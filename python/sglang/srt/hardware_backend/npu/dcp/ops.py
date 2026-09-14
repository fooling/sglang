"""Pure-torch helpers for decode context parallel (DCP) on the Ascend NPU path.

Layout (see ``PagedTokenToKVPoolAllocator`` built with ``page_size * dcp_size``):
a request position ``p`` lives at virtual slot ``v = page * P * c + p % (P * c)``;
rank ``v % c`` owns it and stores it at physical slot ``v // c``. Hence rank
``r`` finds its tokens of virtual page ``page`` in physical page ``page``,
position-ordered, and every rank uses the same block table
``req_to_token[:, 0:len:P*c] // (P*c)`` with block size ``P``.

Nothing here imports torch_npu, so the math is unit-testable on CPU. Functions
that communicate take a ``GroupCoordinator``-like ``group`` argument.
"""

import math
from typing import List, NamedTuple, Sequence, Union

import torch

from sglang.srt.layers.dcp.layout import get_dcp_lens


def dcp_physical_write_loc(
    loc: torch.Tensor, dcp_size: int, dcp_rank: int, pad_slot: int = 0
) -> torch.Tensor:
    """Map virtual write locs to this rank's physical slots.

    Tokens owned by another rank (and negative locs) go to ``pad_slot`` so the
    write keeps a static shape (graph-safe, no boolean indexing).
    """
    if dcp_size == 1:
        return loc
    owned = (loc >= 0) & (loc % dcp_size == dcp_rank)
    return torch.where(owned, loc // dcp_size, loc.new_full((), pad_slot))


def dcp_block_tables(
    req_to_token_rows: torch.Tensor, max_len, page_size: int, dcp_size: int
) -> torch.Tensor:
    """Per-rank FIA block table: virtual page ids (== physical page ids)."""
    stride = page_size * dcp_size
    return (req_to_token_rows[:, 0:max_len:stride] // stride).to(torch.int32)


def dcp_local_seq_lens(
    seq_lens: Union[torch.Tensor, Sequence[int]], dcp_size: int, dcp_rank: int
) -> Union[torch.Tensor, List[int]]:
    """KV length this rank holds per request (owner rule pos % c == rank)."""
    if isinstance(seq_lens, torch.Tensor):
        return get_dcp_lens(seq_lens, dcp_size, dcp_rank)
    if dcp_size == 1:
        return list(seq_lens)
    return [int(n) // dcp_size + int(dcp_rank < int(n) % dcp_size) for n in seq_lens]


def dcp_verify_history_local_lens(
    seq_lens_with_w: Union[torch.Tensor, Sequence[int]],
    w: int,
    dcp_size: int,
    dcp_rank: int,
) -> Union[torch.Tensor, List[int]]:
    """History KV length this rank holds for a target-verify window.

    ``seq_lens_with_w`` already counts the ``w`` verify tokens (written to the
    cache before attention); history is the global prefix before them,
    ``max(len - w, 0)`` (0 for idle / graph-padding rows), sharded by the
    owner rule pos % c == rank.
    """
    if isinstance(seq_lens_with_w, torch.Tensor):
        history = (seq_lens_with_w - w).clamp_min(0)
    else:
        history = [max(int(n) - int(w), 0) for n in seq_lens_with_w]
    return dcp_local_seq_lens(history, dcp_size, dcp_rank)


def dcp_interleave_pages(gathered: torch.Tensor, total_len: int) -> torch.Tensor:
    """[c, npages, P, *tail] (rank-major) -> [npages*P*c, *tail][:total_len].

    Row ``j`` of physical page ``page`` on rank ``r`` holds position
    ``page * P * c + j * c + r``, so page-major, row-major, rank-minor order is
    the global position order.
    """
    c, npages, page_size = gathered.shape[:3]
    tail = gathered.shape[3:]
    rows = gathered.movedim(0, 2).reshape(npages * page_size * c, *tail)
    return rows[:total_len]


class DcpPrefixChunk(NamedTuple):
    """Global prefix positions [start, end) and the DCP pages that cover them."""

    start: int
    end: int
    first_page: int  # index into the request's DCP block table
    num_pages: int
    row_offset: int  # start - first_page * page_size * dcp_size


def dcp_prefix_chunk_plan(
    prefix_len: int, chunk_tokens: int, page_size: int, dcp_size: int
) -> List[DcpPrefixChunk]:
    """Split a request's global prefix [0, prefix_len) into chunks.

    Depends only on global lengths, so every DCP rank derives the same plan and
    issues the same all-gathers in the same order (none for prefix_len == 0).
    """
    assert chunk_tokens > 0, f"chunk_tokens must be positive, got {chunk_tokens}"
    stride = page_size * dcp_size
    plan = []
    for start in range(0, int(prefix_len), chunk_tokens):
        end = min(start + chunk_tokens, int(prefix_len))
        first_page = start // stride
        num_pages = (end + stride - 1) // stride - first_page
        plan.append(
            DcpPrefixChunk(
                start, end, first_page, num_pages, start - first_page * stride
            )
        )
    return plan


def dcp_gather_chunk_rows(
    local_pages: torch.Tensor, row_offset: int, num_rows: int, group
) -> torch.Tensor:
    """Globally ordered rows of one prefix chunk from every rank's local pages.

    local_pages: [num_pages, P, *tail], this rank's physical pages selected by
    the (rank independent) DCP block table, so all ranks share the shape.
    Returns rows [row_offset, row_offset + num_rows) of the interleaved pages.
    """
    c = group.world_size
    gathered = group.all_gather(local_pages.contiguous(), dim=0)
    gathered = gathered.view(c, *local_pages.shape)
    return dcp_interleave_pages(gathered, row_offset + num_rows)[row_offset:]


def lse_combine(
    outs: torch.Tensor, lses: torch.Tensor, base_e: bool = True
) -> torch.Tensor:
    """Exact softmax merge of N partial attentions over disjoint KV shards.

    outs: [N, B, H, D], lses: [N, B, H]. A shard with -inf/NaN LSE (no KV on
    that rank) contributes nothing; if every shard is empty the result is 0.
    """
    out_dtype = outs.dtype
    lses = lses.float()
    lses = torch.where(torch.isnan(lses), torch.full_like(lses, -math.inf), lses)
    if not base_e:
        lses = lses * math.log(2.0)
    valid = torch.isfinite(lses)
    max_lse = torch.where(valid, lses, torch.full_like(lses, -math.inf)).amax(dim=0)
    max_lse = torch.where(torch.isfinite(max_lse), max_lse, torch.zeros_like(max_lse))
    weights = torch.where(valid, torch.exp(lses - max_lse), torch.zeros_like(lses))
    denom = weights.sum(dim=0, keepdim=True)
    weights = torch.where(denom > 0, weights / denom.clamp_min(1e-30), weights)
    outs = torch.where(
        valid.unsqueeze(-1), outs.float(), torch.zeros_like(outs.float())
    )
    return (weights.unsqueeze(-1) * outs).sum(dim=0).to(out_dtype)


def _lse_pack_cols(dtype: torch.dtype) -> int:
    elem = torch.empty((), dtype=dtype).element_size()
    assert 4 % elem == 0, f"cannot pack an fp32 LSE into {dtype} columns"
    return 4 // elem


def _dcp_a2a_packed_exchange(out: torch.Tensor, lse: torch.Tensor, group):
    """Packed A2A exchange: out [B, N*h, D], lse [B, N*h] -> ([N, B, h, D], [N, B, h]).

    send[j] carries this rank's partial for the heads rank j keeps, with the
    fp32 LSE reinterpreted as trailing ``out.dtype`` columns, so output + LSE
    move in one all_to_all_single. The buffer keeps the float dtype (no uint8
    byte view) for HCCL. recv[j] = rank j's partial for this rank's heads;
    the returned out keeps ``out.dtype`` and the LSE is float32.
    """
    n = group.world_size
    b, heads, d = out.shape
    assert heads % n == 0, f"num_heads ({heads}) must be divisible by dcp ({n})"
    h = heads // n
    cols = _lse_pack_cols(out.dtype)
    assert (d + cols) % cols == 0, f"head dim {d} not packable with {cols} cols"

    send = out.new_empty((n, b, h, d + cols))
    send[..., :d] = out.view(b, n, h, d).transpose(0, 1)
    send.view(torch.float32)[..., d // cols] = lse.float().view(b, n, h).transpose(0, 1)
    recv = torch.empty_like(send)
    # NPU-DCP: verify on device: HCCL all_to_all_single inside a captured NPU graph.
    group.all_to_all_single(recv.view(-1), send.view(-1))

    return recv[..., :d], recv.view(torch.float32)[..., d // cols]


def lse_logsumexp_valid(lses: torch.Tensor, base_e: bool = True) -> torch.Tensor:
    """Natural-log merged LSE over dim 0, ignoring invalid (non-finite) shards.

    lses: [N, ...]; an element with no valid shard gets -inf.
    """
    lses = lses.float()
    if not base_e:
        lses = lses * math.log(2.0)
    lses = torch.where(torch.isfinite(lses), lses, torch.full_like(lses, -math.inf))
    return torch.logsumexp(lses, dim=0)


def _natural_lse(lse: torch.Tensor, base_e: bool) -> torch.Tensor:
    lse = lse.float()
    return lse if base_e else lse * math.log(2.0)


def dcp_merge_a2a(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    base_e: bool = True,
    return_lse: bool = False,
):
    """A2A merge (packed exchange + pure-torch ``lse_combine``): out [B, N*h, D],
    lse [B, N*h] -> [B, h, D]; with ``return_lse`` also the merged natural-log
    LSE [B, h] (float32, -inf where no shard is valid)."""
    if group.world_size == 1:
        return (out, _natural_lse(lse, base_e)) if return_lse else out
    recv_out, recv_lse = _dcp_a2a_packed_exchange(out, lse, group)
    merged = lse_combine(recv_out, recv_lse, base_e=base_e)
    if not return_lse:
        return merged
    return merged, lse_logsumexp_valid(recv_lse, base_e=base_e)


# Finite stand-in for an invalid LSE (+inf FIA sentinel for an empty local KV,
# -inf, NaN): exp(-1e30 - lse) underflows to 0 against any finite shard, and
# when every shard is invalid the zeroed outputs merge to 0.
_INVALID_LSE = -1e30


def _torch_npu_attention_update(lse_list, out_list, update_type):
    import torch_npu  # lazy: keep this module importable on CPU

    return torch_npu.npu_attention_update(lse_list, out_list, update_type)


# Swapped for attention_update_reference in CPU tests.
_attention_update_op = _torch_npu_attention_update


def attention_update_reference(
    lse_list: Sequence[torch.Tensor],
    out_list: Sequence[torch.Tensor],
    update_type: int = 0,
):
    """Pure-torch ``torch_npu.npu_attention_update`` (documented formula).

    lse_i [T] float32, out_i [T, D]; natural log:
    lsemax = max_i lse_i; lse = lsemax + log sum_i exp(lse_i - lsemax);
    out = sum_i out_i * exp(lse_i - lse). Returns (out, lse or None).
    """
    lses = torch.stack([l.float() for l in lse_list])
    outs = torch.stack([o.float() for o in out_list])
    lse_max = lses.amax(dim=0)
    lse = lse_max + torch.log(torch.exp(lses - lse_max).sum(dim=0))
    out = (outs * torch.exp(lses - lse).unsqueeze(-1)).sum(dim=0)
    return out, (lse if update_type == 1 else None)


def npu_attention_update(
    lse_list: Sequence[torch.Tensor],
    out_list: Sequence[torch.Tensor],
    return_lse: bool = False,
):
    """Merge partial attentions over disjoint KV: lse_i [T], out_i [T, D] -> [T, D] fp32.

    Invalid (non-finite) shard LSEs are sanitised first (vllm-ascend notes FIA
    returns +inf for an empty local KV and only its fused Triton merge skips
    it); an element with no valid shard merges to 0, like ``lse_combine``.
    With ``return_lse`` (update_type=1) also returns the merged LSE [T]
    (float32, natural log) = logsumexp over the valid shards, -inf where no
    shard is valid.
    """
    lses, outs = [], []
    any_valid = None
    for lse, out in zip(lse_list, out_list):
        lse = lse.float()
        out = out.float()
        valid = torch.isfinite(lse)
        any_valid = valid if any_valid is None else any_valid | valid
        lses.append(torch.where(valid, lse, lse.new_full((), _INVALID_LSE)))
        outs.append(torch.where(valid.unsqueeze(-1), out, out.new_zeros(())))
    if not return_lse:
        # NPU-DCP: verify on device: torch_npu.npu_attention_update (update_type=0)
        # inside a captured NPU graph, and bit-level agreement with lse_combine.
        out, _ = _attention_update_op(lses, outs, 0)
        return out
    # NPU-DCP: verify on device: torch_npu.npu_attention_update(update_type=1)
    # returns the merged lse as float32 [T] (natural log, same shape as each
    # input lse) next to out [T, D], also inside a captured NPU graph.
    out, lse = _attention_update_op(lses, outs, 1)
    lse = lse.float().reshape(any_valid.shape)
    lse = torch.where(any_valid, lse, lse.new_full((), -math.inf))
    return out, lse


def dcp_merge_a2a_vllm(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    base_e: bool = True,
    return_lse: bool = False,
):
    """vllm-ascend style A2A merge: out [B, N*h, D], lse [B, N*h] -> [B, h, D]
    (with ``return_lse`` also the merged natural-log LSE [B, h] float32).

    Mirrors ``_process_attn_out_lse`` + ``_npu_attention_update``: fp32
    ``cat(out, lse)`` permuted to [N*h, D+1, B], one all_to_all_single (chunk
    j of dim 0 = head group j goes to rank j, as in ``dcp_merge_a2a``), then
    ``npu_attention_update`` over the N received shards.
    """
    n = group.world_size
    if n == 1:
        return (out, _natural_lse(lse, base_e)) if return_lse else out
    out_dtype = out.dtype
    b, heads, d = out.shape
    assert heads % n == 0, f"num_heads ({heads}) must be divisible by dcp ({n})"
    h = heads // n
    lse = lse.float()
    if not base_e:
        lse = lse * math.log(2.0)
    send = torch.cat([out.float(), lse.unsqueeze(-1)], dim=-1)
    send = send.permute(1, 2, 0).contiguous()  # [N*h, D+1, B]
    recv = torch.empty_like(send)
    # NPU-DCP: verify on device: HCCL all_to_all_single inside a captured NPU graph.
    group.all_to_all_single(recv, send)
    # recv chunk j = rank j's partial for this rank's heads.
    x = recv.permute(2, 0, 1).reshape(b, n, h, d + 1).permute(1, 0, 2, 3)
    outs, lses = x.split([d, 1], dim=-1)  # [N, B, h, D], [N, B, h, 1]
    merged = npu_attention_update(
        list(lses.reshape(n, b * h).unbind(0)),
        list(outs.reshape(n, b * h, d).unbind(0)),
        return_lse=return_lse,
    )
    if return_lse:
        merged, merged_lse = merged
        return merged.view(b, h, d).to(out_dtype), merged_lse.view(b, h)
    return merged.view(b, h, d).to(out_dtype)


def dcp_merge_a2a_npu(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    base_e: bool = True,
    return_lse: bool = False,
):
    """A2A merge (packed exchange + ``npu_attention_update``): out [B, N*h, D],
    lse [B, N*h] -> [B, h, D] (with ``return_lse`` also the merged natural-log
    LSE [B, h] float32).

    Same single all_to_all_single as ``dcp_merge_a2a`` (out in model dtype,
    fp32 LSE packed as trailing columns); the received shards are cast to
    float32 and merged by the sanitising ``npu_attention_update``.
    """
    n = group.world_size
    if n == 1:
        return (out, _natural_lse(lse, base_e)) if return_lse else out
    out_dtype = out.dtype
    b, heads, d = out.shape
    h = heads // n
    recv_out, recv_lse = _dcp_a2a_packed_exchange(out, lse, group)
    recv_lse = recv_lse.float()
    if not base_e:
        recv_lse = recv_lse * math.log(2.0)
    merged = npu_attention_update(
        list(recv_lse.reshape(n, b * h).unbind(0)),
        list(recv_out.float().reshape(n, b * h, d).unbind(0)),
        return_lse=return_lse,
    )
    if return_lse:
        merged, merged_lse = merged
        return merged.view(b, h, d).to(out_dtype), merged_lse.view(b, h)
    return merged.view(b, h, d).to(out_dtype)


def dcp_merge_ag_rs(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    base_e: bool = True,
    return_lse: bool = False,
):
    """AG+RS merge: out [B, N*h, D], lse [B, N*h] -> [B, h, D] (with
    ``return_lse`` also the merged natural-log LSE [B, h] float32)."""
    n = group.world_size
    if n == 1:
        return (out, _natural_lse(lse, base_e)) if return_lse else out
    b, heads, d = out.shape
    lse = lse.float().contiguous()
    lses = group.all_gather(lse, dim=0).view(n, b, heads)
    # NaN and FIA's +inf empty-shard sentinel both mean "no local KV".
    invalid = torch.isnan(lses) | torch.isposinf(lses)
    lses = torch.where(invalid, torch.full_like(lses, -math.inf), lses)
    lse = lses[group.rank_in_group]
    if base_e:
        global_lse = torch.logsumexp(lses, dim=0)
        scale = torch.exp(lse - global_lse)
    else:
        global_lse = torch.logsumexp(lses * math.log(2.0), dim=0) / math.log(2.0)
        scale = torch.pow(2.0, lse - global_lse)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    corrected = torch.nan_to_num(out.float(), nan=0.0, posinf=0.0, neginf=0.0)
    # Reduce in fp32 like cp_lse_ag_out_rs_mla, then cast back.
    corrected = (corrected * scale.unsqueeze(-1)).contiguous()
    merged = group.reduce_scatter_along_dim(corrected, dim=1).to(out.dtype)
    if not return_lse:
        return merged
    h = heads // n
    rank = group.rank_in_group
    merged_lse = lse_logsumexp_valid(lses[:, :, rank * h : (rank + 1) * h], base_e)
    return merged, merged_lse


def dcp_merge_with_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    comm_backend: str,
    merge_impl: str,
    base_e: bool = True,
):
    """Cross-rank merge that also returns the merged LSE.

    out [B, N*h, D], lse [B, N*h] -> (out [B, h, D], lse [B, h] float32 natural
    log, -inf where no rank holds KV). ``comm_backend`` 'a2a' selects by
    ``merge_impl`` ('npu' / 'vllm' / 'torch', as SGLANG_NPU_DCP_MERGE_IMPL);
    anything else uses AG+RS.
    """
    if comm_backend == "a2a":
        merges = dict(npu=dcp_merge_a2a_npu, vllm=dcp_merge_a2a_vllm, torch=dcp_merge_a2a)
        if merge_impl not in merges:
            raise ValueError(
                "SGLANG_NPU_DCP_MERGE_IMPL must be 'npu', 'vllm' or 'torch', "
                f"got {merge_impl!r}."
            )
        merge = merges[merge_impl]
    else:
        merge = dcp_merge_ag_rs
    return merge(out, lse, group, base_e=base_e, return_lse=True)


def mla_decode_with_lse_torch(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv_pages: torch.Tensor,
    k_rope_pages: torch.Tensor,
    block_table: torch.Tensor,
    local_lens: Union[torch.Tensor, Sequence[int]],
    scale: float,
):
    """Reference single-query MLA decode over this rank's local KV pages.

    q_nope [B, H, Dc], q_rope [B, H, Dr]; c_kv_pages [pages, P, (1,) Dc],
    k_rope_pages [pages, P, (1,) Dr] in logical (token-major) order;
    block_table [B, max_pages]. Returns (out [B, H, Dc], lse [B, H]) with a
    natural-log LSE; a request with local_len == 0 gets out = 0, lse = -inf.
    """
    bsz, heads, d_c = q_nope.shape
    page_size = c_kv_pages.shape[1]
    c_kv_pages = c_kv_pages.reshape(c_kv_pages.shape[0], page_size, -1)
    k_rope_pages = k_rope_pages.reshape(k_rope_pages.shape[0], page_size, -1)
    if isinstance(local_lens, torch.Tensor):
        local_lens = local_lens.tolist()

    out = torch.zeros(bsz, heads, d_c, dtype=torch.float32, device=q_nope.device)
    lse = torch.full((bsz, heads), -math.inf, dtype=torch.float32, device=q_nope.device)
    for i in range(bsz):
        n = int(local_lens[i])
        if n <= 0:
            continue
        npages = (n + page_size - 1) // page_size
        pages = block_table[i, :npages].long()
        kv = c_kv_pages[pages].reshape(-1, c_kv_pages.shape[-1])[:n].float()
        kr = k_rope_pages[pages].reshape(-1, k_rope_pages.shape[-1])[:n].float()
        scores = (
            torch.einsum("hd,td->ht", q_nope[i].float(), kv)
            + torch.einsum("hd,td->ht", q_rope[i].float(), kr)
        ) * scale
        lse[i] = torch.logsumexp(scores, dim=-1)
        out[i] = torch.softmax(scores, dim=-1) @ kv
    return out.to(q_nope.dtype), lse
