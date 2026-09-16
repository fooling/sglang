"""Single-NPU microbenchmark for the DSA sparse attention shape swap.

With attention TP the op is called per rank as [T, H/tp, D]; every rank gathers
the same topk KV for all T tokens while computing 1/tp of the heads.  Exchanging
the query over the attention-TP domain turns that into [T/tp, H, D]: the same
flops, but the KV gather happens once per token for the whole domain.

This script measures only the op, on one card, with no communication: it times
the two shapes side by side so the premise ("the second shape is much faster per
unit of work") can be checked before running a distributed server.  The a2a and
its two transposes are NOT included -- see the "budget" column for how much time
the exchange may take before the swap stops paying.

Run:
    python benchmark/kernels/attention/bench_npu_dsa_sparse_attn_shapes.py
    python benchmark/kernels/attention/bench_npu_dsa_sparse_attn_shapes.py \
        --num-tokens 64 --attn-tp 8 --topk 2048 --seq-len 8192
"""

import argparse
import time

import torch

try:
    import torch_npu
except ImportError as exc:  # pragma: no cover - this script only runs on NPU
    raise SystemExit(f"this benchmark needs torch_npu: {exc}")

from sglang.srt.hardware_backend.npu.attention.fp8_contracts import (
    DSA_KV_QUANT_TILE_SIZE,
    get_dsa_fp8_packed_cache_dim,
)

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64


def _build_inputs(num_tokens, num_heads, topk, seq_len, page_size, device):
    packed_dim = get_dsa_fp8_packed_cache_dim(
        kv_lora_rank=KV_LORA_RANK, qk_rope_head_dim=QK_ROPE_HEAD_DIM
    )
    num_pages = (seq_len + page_size - 1) // page_size
    query = torch.randn(
        num_tokens,
        num_heads,
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.zeros(
        num_pages * page_size, packed_dim, dtype=torch.uint8, device=device
    ).view(torch.float8_e4m3fn)
    indices = torch.randint(
        0, seq_len, (num_tokens, 1, topk), dtype=torch.int32, device=device
    )
    block_table = (
        torch.arange(num_pages, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .repeat(num_tokens, 1)
    )
    seq_q = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device)
    seq_kv = torch.full((num_tokens,), seq_len, dtype=torch.int32, device=device)
    return query, kv, indices, block_table, seq_q, seq_kv, packed_dim


def _time_op(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1e6  # us


def _run_shape(num_tokens, num_heads, topk, seq_len, page_size, scale, device):
    (query, kv, indices, block_table, seq_q, seq_kv, packed_dim) = _build_inputs(
        num_tokens, num_heads, topk, seq_len, page_size, device
    )
    paged_kv = kv.view(-1, page_size, 1, packed_dim)

    def call():
        return torch_npu.npu_kv_quant_sparse_flash_attention(
            query=query,
            key=paged_kv,
            value=paged_kv,
            sparse_indices=indices,
            scale_value=scale,
            key_quant_mode=2,
            value_quant_mode=2,
            key_dequant_scale=None,
            value_dequant_scale=None,
            actual_seq_lengths_query=seq_q,
            actual_seq_lengths_kv=seq_kv,
            block_table=block_table,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=DSA_KV_QUANT_TILE_SIZE,
            rope_head_dim=QK_ROPE_HEAD_DIM,
        )

    return _time_op(call)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=128)
    parser.add_argument("--attn-tp", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--scale", type=float, default=0.1)
    args = parser.parse_args()

    device = "npu"
    torch.manual_seed(0)

    print(
        f"tokens={args.num_tokens} q_heads={args.q_heads} topk={args.topk} "
        f"kv_len={args.seq_len} page={args.page_size}\n"
    )
    header = f"{'attn_tp':>8} {'sharded heads':>14} {'exchanged':>10} {'speedup':>8} {'budget':>9}"
    print(header)
    print("-" * len(header))

    for tp in args.attn_tp:
        if args.q_heads % tp or args.num_tokens % tp:
            print(f"{tp:>8}  skipped (not divisible)")
            continue
        sharded = _run_shape(
            args.num_tokens,
            args.q_heads // tp,
            args.topk,
            args.seq_len,
            args.page_size,
            args.scale,
            device,
        )
        exchanged = _run_shape(
            args.num_tokens // tp,
            args.q_heads,
            args.topk,
            args.seq_len,
            args.page_size,
            args.scale,
            device,
        )
        print(
            f"{tp:>8} {sharded:>13.1f}us {exchanged:>9.1f}us "
            f"{sharded / exchanged:>7.2f}x {sharded - exchanged:>8.1f}us"
        )

    print(
        "\nbudget = time the two all-to-alls plus their transposes may take per "
        "layer\nbefore the exchange stops paying for itself."
    )


if __name__ == "__main__":
    main()
