"""Triton-Ascend kernel for the MLA latent split + q/k RMSNorm.

The fused ``qkv_a_proj`` writes one row-major [T, q_lora + kv_lora + rope]
tensor. Splitting it gives three *strided* views, and ``npu_rms_norm`` follows
aclnn's dense-stride contract, so the framework inserts a ``contiguous()``
before each of the two norms -- a copy that buys nothing, since the reduction
dimension of each slice is already contiguous.

This kernel reads the strided slices directly and emits ``q_norm``, ``k_norm``
and the rope slice in one pass, halving the launches (three ops -> one) and
dropping both staging copies.

It is *not* the vendor ``sgl_kernel_npu.norm.fused_split_qk_norm``: Kimi-K3
disables that one as numerically non-equivalent
(``models/kimi_k3.py``: ``_disable_npu_fused_split_qk_norm``). This kernel
keeps the reference arithmetic exactly -- accumulate the sum of squares in
float32, scale by ``rsqrt(mean + eps)``, multiply by the float32 weight, and
cast once on the way out -- so it can be checked element-wise against
``npu_rms_norm`` before it is switched on.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.hardware_backend.npu.triton_ops.utils import npu_vector_cores

# Rows per tile (BLOCK_T). A tile holds, per element of the widest slice, the
# bf16 load (2 B), its fp32 cast (4 B), the squared temporary (4 B), the fp32
# product (4 B) and the cast on the way out (2 B): ~16 B. At the K3 widths the
# q slice (1536) and the k slice (512) can be live together, so a tile of 4
# rows is ~4 * 2048 * 16 B = 128 KiB against a 192 KiB Ascend910B UB, and 8
# rows would be ~256 KiB. The cap is a constant for that reason, not a tunable.
_MAX_BLOCK_T = 4


def _pow2_split(dim: int):
    """dim -> (head, tail): the largest power of two <= dim, and the remainder
    rounded up to one. 1536 -> (1024, 512), 512 -> (512, 0), 96 -> (64, 32).

    ``tl.arange`` wants a power of two, and rounding the whole width up (2048
    for 1536) spends a quarter of every load, square and store on masked
    zeros. Two exact blocks spend none at the K3 widths; only a tail that is
    not itself a power of two keeps a mask, and only on its own columns.
    """
    head = 1 << (int(dim).bit_length() - 1)
    rest = int(dim) - head
    return head, (triton.next_power_of_2(rest) if rest else 0)


@triton.jit
def _rmsnorm_tile(
    x_ptr,
    x_row,
    col0,
    w_head,
    w_tail,
    out_ptr,
    rows,
    row_mask,
    eps,
    DIM: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    """RMSNorm of x[rows, col0 : col0 + DIM] -> out[rows, :DIM], a tile at a time.

    The sum of squares comes back as a [BLOCK_T] vector (``axis=1``) and is
    broadcast straight back over the tile, so a row never reduces to a scalar
    that has to leave the vector unit and return.
    """
    src = x_ptr + rows[:, None] * x_row + col0
    dst = out_ptr + rows[:, None] * DIM
    head_cols = tl.arange(0, HEAD)
    head_mask = row_mask[:, None] & (head_cols < DIM)[None, :]
    a = tl.load(src + head_cols[None, :], mask=head_mask, other=0.0).to(tl.float32)
    if TAIL > 0:
        tail_cols = HEAD + tl.arange(0, TAIL)
        tail_mask = row_mask[:, None] & (tail_cols < DIM)[None, :]
        b = tl.load(src + tail_cols[None, :], mask=tail_mask, other=0.0).to(tl.float32)
        scale = tl.rsqrt((tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)) / DIM + eps)
        tl.store(
            dst + tail_cols[None, :],
            (b * scale[:, None] * w_tail[None, :]).to(out_ptr.dtype.element_ty),
            mask=tail_mask,
        )
        tl.store(
            dst + head_cols[None, :],
            (a * scale[:, None] * w_head[None, :]).to(out_ptr.dtype.element_ty),
            mask=head_mask,
        )
    else:
        scale = tl.rsqrt(tl.sum(a * a, axis=1) / DIM + eps)
        tl.store(
            dst + head_cols[None, :],
            (a * scale[:, None] * w_head[None, :]).to(out_ptr.dtype.element_ty),
            mask=head_mask,
        )


@triton.jit
def _split_qk_rmsnorm_kernel(
    x_ptr,  # [T, q_dim + k_dim + r_dim]
    q_weight_ptr,  # [q_dim]
    k_weight_ptr,  # [k_dim]
    q_out_ptr,  # [T, q_dim]
    k_out_ptr,  # [T, k_dim]
    r_out_ptr,  # [T, r_dim]
    x_row,  # element stride of an x row
    n_rows,
    q_eps,
    k_eps,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    R_DIM: tl.constexpr,
    Q_HEAD: tl.constexpr,
    Q_TAIL: tl.constexpr,
    K_HEAD: tl.constexpr,
    K_TAIL: tl.constexpr,
    Q_TAIL_LOAD: tl.constexpr,  # max(Q_TAIL, 1)
    K_TAIL_LOAD: tl.constexpr,  # max(K_TAIL, 1)
    BLOCK_R: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """q_out = rmsnorm(x[:, :Q]), k_out = rmsnorm(x[:, Q:Q+K]), r_out = x[:, Q+K:].

    Tiled over ``BLOCK_T`` rows as well as the columns -- the shape the vendor's
    K3-tuned RMSNorm uses (``triton_ascend_kernels`` rmsnorm_situ_optim): one
    [BLOCK_T, D] load per slice, a vector of row sums, one [BLOCK_T, D] store.
    The row-at-a-time form this replaces measured 21.7 us a call with the
    vector unit busy for 4.8 of them at T = 4, against 3.2 us for the KV write
    kernel on the same grid and the same rows; what it did and that kernel
    does not is reduce every row to a scalar, twice.
    """
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    # Column-only values, shared by every tile: load them once, not per row.
    q_head_cols = tl.arange(0, Q_HEAD)
    wq_head = tl.load(
        q_weight_ptr + q_head_cols, mask=q_head_cols < Q_DIM, other=0.0
    ).to(tl.float32)
    # The tail load is unconditional and sized at least 1: with no tail its mask
    # is all false and it reads nothing. That keeps every name bound once, to
    # one shape -- a rebind inside a constexpr branch is something the
    # interpreter accepts whether or not the compiler does.
    q_tail_cols = Q_HEAD + tl.arange(0, Q_TAIL_LOAD)
    wq_tail = tl.load(
        q_weight_ptr + q_tail_cols, mask=q_tail_cols < Q_DIM, other=0.0
    ).to(tl.float32)
    k_head_cols = tl.arange(0, K_HEAD)
    wk_head = tl.load(
        k_weight_ptr + k_head_cols, mask=k_head_cols < K_DIM, other=0.0
    ).to(tl.float32)
    k_tail_cols = K_HEAD + tl.arange(0, K_TAIL_LOAD)
    wk_tail = tl.load(
        k_weight_ptr + k_tail_cols, mask=k_tail_cols < K_DIM, other=0.0
    ).to(tl.float32)
    r_cols = tl.arange(0, BLOCK_R)
    r_col_mask = r_cols < R_DIM
    tile = tl.arange(0, BLOCK_T)

    for row0 in range(start_row, end_row, BLOCK_T):
        rows = row0 + tile
        row_mask = rows < end_row

        _rmsnorm_tile(
            x_ptr, x_row, 0, wq_head, wq_tail, q_out_ptr, rows, row_mask, q_eps,
            Q_DIM, Q_HEAD, Q_TAIL,
        )
        _rmsnorm_tile(
            x_ptr, x_row, Q_DIM, wk_head, wk_tail, k_out_ptr, rows, row_mask, k_eps,
            K_DIM, K_HEAD, K_TAIL,
        )

        r_mask = row_mask[:, None] & r_col_mask[None, :]
        r_vals = tl.load(
            x_ptr + rows[:, None] * x_row + Q_DIM + K_DIM + r_cols[None, :],
            mask=r_mask,
            other=0.0,
        )
        tl.store(r_out_ptr + rows[:, None] * R_DIM + r_cols[None, :], r_vals, mask=r_mask)


def _launch_shape(n_rows: int):
    """(programs, BLOCK_T). Tiles first, programs second: a decode step of four
    rows is one program with one tile, not four programs of a row each, and a
    prefill still spreads one chunk of tiles to every vector core."""
    block_t = min(triton.next_power_of_2(n_rows), _MAX_BLOCK_T)
    n_tiles = -(-n_rows // block_t)
    return max(1, min(n_tiles, npu_vector_cores())), block_t


def split_qk_rmsnorm(
    qkv_latent: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    q_eps: float,
    k_eps: float,
):
    """[T, q_lora + kv_lora + rope] -> (q [T, q_lora], k_nope [T, 1, kv_lora],
    k_pe [T, 1, rope]), the two normed parts and the untouched rope slice."""
    assert qkv_latent.dim() == 2, f"expected [T, D], got {tuple(qkv_latent.shape)}"
    total = q_lora_rank + kv_lora_rank + qk_rope_head_dim
    assert qkv_latent.shape[1] == total, (
        f"latent width {qkv_latent.shape[1]} != {q_lora_rank} + {kv_lora_rank} + "
        f"{qk_rope_head_dim}"
    )
    assert qkv_latent.stride(1) == 1, "the latent rows must be contiguous"
    n_rows = qkv_latent.shape[0]
    q = qkv_latent.new_empty((n_rows, q_lora_rank))
    k_nope = qkv_latent.new_empty((n_rows, 1, kv_lora_rank))
    k_pe = qkv_latent.new_empty((n_rows, 1, qk_rope_head_dim))
    if n_rows == 0:
        return q, k_nope, k_pe
    n_programs, block_t = _launch_shape(n_rows)
    q_head, q_tail = _pow2_split(q_lora_rank)
    k_head, k_tail = _pow2_split(kv_lora_rank)
    _split_qk_rmsnorm_kernel[(n_programs,)](
        qkv_latent,
        q_weight,
        k_weight,
        q,
        k_nope,
        k_pe,
        qkv_latent.stride(0),
        n_rows,
        q_eps,
        k_eps,
        Q_DIM=q_lora_rank,
        K_DIM=kv_lora_rank,
        R_DIM=qk_rope_head_dim,
        Q_HEAD=q_head,
        Q_TAIL=q_tail,
        K_HEAD=k_head,
        K_TAIL=k_tail,
        Q_TAIL_LOAD=max(q_tail, 1),
        K_TAIL_LOAD=max(k_tail, 1),
        BLOCK_R=triton.next_power_of_2(qk_rope_head_dim),
        BLOCK_T=block_t,
    )
    return q, k_nope, k_pe
