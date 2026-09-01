# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton Indexer logits over the packed DSA Indexer cache.

The packed cache stores a page as ``page_size * index_head_dim`` bytes of FP8 K
followed by ``page_size * 4`` bytes of FP32 scale, so that a HiCache page moves
as one DMA block per Indexer layer instead of two.  Nothing in the vendor
Ascend op set reads that page: ``npu_quant_lightning_indexer`` takes ``key`` and
``key_dequant_scale`` as two contiguous PA_ND tensors, and the K half of a
packed page has a pitch of ``page_size * (index_head_dim + 4)``.

This kernel reads the page in place.  It addresses it the way
``_set_k_and_s_triton_kernel`` in ``index_buf_accessor.py`` writes it: the same
buffer is passed twice, once viewed FP8 and once viewed FP32, K offsets counted
in FP8 elements and scale offsets divided by 4.

The score is the Indexer's, transcribed from ``fp8_paged_mqa_logits_torch``
(``sglang/srt/layers/attention/dsv4/indexer.py``), which reads this same packed
layout on the CUDA side::

    logits[t, s] = (sum_h w[t, h] * relu(q[t, h] . k[s])) * k_scale[s]

with the query dequant scale folded into ``w``.  ReLU is positively homogeneous
and both scales are non-negative, so folding a scale in before or after the ReLU
does not change the result -- which is why the deepgemm reference test can apply
them in a different order and still agree.

Portable Triton: no CUDA intrinsics, so a Triton-Ascend build compiles it for
the AI Core.  Correctness here is checked under ``TRITON_INTERPRET=1``; nothing
about its tiling has been tuned on any device.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# One FP32 dequant scale per token, stored after a page's FP8 K rows.
INDEX_K_SCALE_BYTES = 4

# tl.dot wants every extent of the contraction to be at least this wide.
_MIN_DOT_EXTENT = 16


@triton.jit
def _packed_indexer_logits_kernel(
    cache_fp8_ptr,  # packed cache viewed FP8
    cache_fp32_ptr,  # the same buffer viewed FP32
    q_ptr,  # (num_tokens, num_heads, head_dim) fp32
    w_ptr,  # (num_tokens, num_heads) fp32, query dequant scale folded in
    block_table_ptr,  # (num_requests, max_pages) int32
    token_request_ptr,  # (num_tokens,) int32
    token_position_ptr,  # (num_tokens,) int32, position of the token in its kv run
    kv_len_ptr,  # (num_requests,) int32
    logits_ptr,  # (num_tokens, max_pages * PAGE_SIZE) fp32
    block_table_stride,
    q_stride_token,
    q_stride_head,
    logits_stride_token,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,  # page_size * (head_dim + 4), FP8 elements
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,  # page_size * head_dim
):
    token = tl.program_id(0)
    page_slot = tl.program_id(1)

    request = tl.load(token_request_ptr + token)
    kv_len = tl.load(kv_len_ptr + request)
    first = page_slot * PAGE_SIZE
    if first >= kv_len:
        return

    page = tl.load(block_table_ptr + request * block_table_stride + page_slot)

    heads = tl.arange(0, NUM_HEADS)
    dims = tl.arange(0, HEAD_DIM)
    slots = tl.arange(0, PAGE_SIZE)

    q = tl.load(q_ptr + token * q_stride_token + heads[:, None] * q_stride_head + dims[None, :])
    w = tl.load(w_ptr + token * NUM_HEADS + heads)

    # K transposed straight out of the page: (head_dim, page_size).  The page
    # base is in FP8 elements, which are bytes, so the K half needs no offset.
    k_offsets = page * BUF_NUMEL_PER_PAGE + slots[None, :] * HEAD_DIM + dims[:, None]
    k = tl.load(cache_fp8_ptr + k_offsets).to(tl.float32)

    # "// 4" b/c the scale half is FP32 while the page base is counted in bytes.
    s_offsets = (
        page * (BUF_NUMEL_PER_PAGE // 4) + (S_OFFSET_NBYTES_IN_PAGE // 4) + slots
    )
    k_scale = tl.load(cache_fp32_ptr + s_offsets)

    scores = tl.dot(q, k)  # (num_heads, page_size)
    scores = tl.maximum(scores, 0.0)
    logits = tl.sum(scores * w[:, None], axis=0) * k_scale

    # sparse_mode=3: a query token sees keys up to its own position only.
    position = tl.load(token_position_ptr + token)
    positions = first + slots
    visible = (positions < kv_len) & (positions <= position)
    logits = tl.where(visible, logits, float("-inf"))

    tl.store(logits_ptr + token * logits_stride_token + positions, logits)


def packed_indexer_supported(
    num_heads: int, head_dim: int, page_size: int
) -> bool:
    """Whether the kernel's tl.dot can express this shape.

    Every extent of the contraction has to reach ``_MIN_DOT_EXTENT``, and Triton
    wants powers of two for the ranges this kernel builds.
    """
    extents = (num_heads, head_dim, page_size)
    return all(
        extent >= _MIN_DOT_EXTENT and extent & (extent - 1) == 0 for extent in extents
    )


def packed_indexer_logits(
    *,
    cache: torch.Tensor,
    query: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    token_request: torch.Tensor,
    token_position: torch.Tensor,
    kv_lens: torch.Tensor,
    index_head_dim: int,
    page_size: int,
) -> torch.Tensor:
    """Indexer logits for every query token, ``-inf`` where a key is not visible.

    ``cache`` is one layer's packed Indexer cache, any shape whose flat bytes are
    ``(page_num, page_size * (index_head_dim + 4))`` uint8.  Returns
    ``(num_tokens, max_pages * page_size)`` fp32.
    """
    num_tokens, num_heads, head_dim = query.shape
    if head_dim != index_head_dim:
        raise ValueError(
            f"query head_dim {head_dim} != index_head_dim {index_head_dim}"
        )
    if not packed_indexer_supported(num_heads, head_dim, page_size):
        raise ValueError(
            f"packed Indexer kernel needs power-of-two extents of at least "
            f"{_MIN_DOT_EXTENT}: got num_heads={num_heads}, head_dim={head_dim}, "
            f"page_size={page_size}"
        )
    row = index_head_dim + INDEX_K_SCALE_BYTES
    flat = cache.reshape(-1)
    if flat.numel() % (page_size * row) != 0:
        raise ValueError(
            f"packed cache is {flat.numel()} bytes, not a multiple of the "
            f"{page_size * row}-byte page"
        )

    max_pages = block_table.shape[1]
    logits = torch.full(
        (num_tokens, max_pages * page_size),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    _packed_indexer_logits_kernel[(num_tokens, max_pages)](
        flat.view(torch.float8_e4m3fn),
        flat.view(torch.float32),
        query.contiguous(),
        weights.contiguous(),
        block_table.contiguous(),
        token_request.contiguous(),
        token_position.contiguous(),
        kv_lens.contiguous(),
        logits,
        block_table.shape[1],
        query.stride(0),
        query.stride(1),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=page_size * row,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
    )
    return logits


def packed_indexer_topk(
    *,
    cache: torch.Tensor,
    query: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    block_table: torch.Tensor,
    cumulative_seq_lengths_query: torch.Tensor,
    seq_lengths_key: torch.Tensor,
    index_head_dim: int,
    page_size: int,
    sparse_count: int,
) -> torch.Tensor:
    """Top-``sparse_count`` key positions per query token, from the packed cache.

    The per-token bookkeeping -- which request a token belongs to and where it
    sits in that request's kv run -- is derived here rather than in the kernel,
    so the kernel stays one program per (token, page) with no ragged logic.
    """
    num_tokens = query.shape[0]
    cumulative = cumulative_seq_lengths_query.to(torch.int64).tolist()
    kv_lens = seq_lengths_key.to(torch.int64)

    token_request = torch.empty(num_tokens, dtype=torch.int32, device=query.device)
    token_position = torch.empty(num_tokens, dtype=torch.int32, device=query.device)
    start = 0
    for request, end in enumerate(cumulative):
        if end <= start:
            continue
        kv_len = int(kv_lens[request])
        token_request[start:end] = request
        # The last query token of a request sits at kv_len - 1.
        first_position = kv_len - (end - start)
        token_position[start:end] = torch.arange(
            first_position, first_position + (end - start), device=query.device
        ).to(torch.int32)
        start = end

    folded = weights.to(torch.float32) * query_dequant_scale.to(torch.float32).reshape(
        num_tokens, -1
    )
    logits = packed_indexer_logits(
        cache=cache,
        query=query.to(torch.float32),
        weights=folded,
        block_table=block_table,
        token_request=token_request,
        token_position=token_position,
        kv_lens=kv_lens.to(torch.int32),
        index_head_dim=index_head_dim,
        page_size=page_size,
    )

    out = torch.full(
        (num_tokens, sparse_count), -1, dtype=torch.int32, device=query.device
    )
    start = 0
    for request, end in enumerate(cumulative):
        if end <= start:
            continue
        kv_len = int(kv_lens[request])
        take = min(sparse_count, kv_len)
        out[start:end, :take] = (
            logits[start:end, :kv_len].topk(take, dim=-1).indices.to(torch.int32)
        )
        start = end
    return out
