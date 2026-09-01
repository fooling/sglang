"""Indexer top-k over the packed DSA Indexer cache.

The packed cache keeps a page's FP8 Indexer K rows and their FP32 dequant
scales in one allocation -- ``page_size * index_head_dim`` bytes of K followed
by ``page_size * 4`` bytes of scale -- which is what lets a HiCache page move in
one transfer block per Indexer layer instead of two
(:attr:`NPUMLATokenToKVPool.index_k_with_scale_buffer`).  It is the same layout
``fp8_paged_mqa_logits_torch`` reads on the CUDA side.

``torch_npu.npu_quant_lightning_indexer`` cannot read it: it takes ``key`` and
``key_dequant_scale`` as two separate contiguous PA_ND tensors, and slicing the
K half out of a packed page leaves a page pitch of
``page_size * (index_head_dim + 4)``.  An op that accepts that pitch -- or that
reads the scale inline, the way ``npu_kv_quant_sparse_flash_attention`` already
does with ``key_quant_mode=2`` -- is what this module dispatches to when one is
registered.

When no such op is registered this dispatches to the Triton kernel in
``sglang/kernels/ops/attention/dsa/packed_indexer_triton.py``, which reads the
packed page in place -- portable Triton, so a Triton-Ascend build compiles it
for the AI Core.  Failing that, :func:`unpack_index_k_with_scale` materializes
the two views the vendor op wants; that is a copy, and it is the copy the packed
layout exists to avoid, so it is a correctness path rather than a fast one.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# One FP32 dequant scale per token, stored after a page's FP8 K rows.
INDEX_K_SCALE_BYTES = 4

_NATIVE_OP_NAME = "npu_quant_lightning_indexer_packed"


def native_packed_indexer_op():
    """The registered packed-cache Indexer op, or None when nothing provides it.

    Looked up per call rather than cached at import: the op may be installed by
    a shim (CANNON registers it on ``torch.ops.npu``) after this module is
    imported.
    """
    namespace = getattr(torch.ops, "npu", None)
    if namespace is None:
        return None
    return getattr(namespace, _NATIVE_OP_NAME, None)


def _triton_packed_indexer():
    """``(supported, topk)`` from the Triton kernel, or None when it will not load.

    Imported lazily: Triton is optional, and on a build without a backend for
    this device importing it is the failure, not calling it.
    """
    try:
        from sglang.kernels.ops.attention.dsa.packed_indexer_triton import (
            packed_indexer_supported,
            packed_indexer_topk,
        )
    except Exception:
        return None
    return packed_indexer_supported, packed_indexer_topk


def unpack_index_k_with_scale(
    key_with_scale: torch.Tensor,
    index_head_dim: int,
    k_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split one layer's packed cache into the contiguous K and scale tensors.

    ``key_with_scale`` is ``(page_num, page_size, 1, index_head_dim + 4)`` uint8,
    whose per-page bytes are ``page_size * index_head_dim`` of K then
    ``page_size * 4`` of FP32 scale.  Returns ``(key, key_scale)`` shaped
    ``(page_num, page_size, 1, index_head_dim)`` and ``(page_num, page_size, 1)``
    -- both contiguous, i.e. both copies.
    """
    page_num, page_size = key_with_scale.shape[0], key_with_scale.shape[1]
    row = index_head_dim + INDEX_K_SCALE_BYTES
    if key_with_scale.shape[-1] != row:
        raise ValueError(
            f"packed Indexer page row is {key_with_scale.shape[-1]} bytes, "
            f"expected {row} for index_head_dim={index_head_dim}"
        )
    flat = key_with_scale.reshape(page_num, page_size * row)
    k_bytes = page_size * index_head_dim
    key = (
        flat[:, :k_bytes]
        .reshape(page_num, page_size, 1, index_head_dim)
        .contiguous()
        .view(k_dtype)
    )
    key_scale = (
        flat[:, k_bytes:]
        .contiguous()
        .view(torch.float32)
        .reshape(page_num, page_size, 1)
    )
    return key, key_scale


def quant_lightning_indexer_packed(
    *,
    query: torch.Tensor,
    key_with_scale: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    index_head_dim: int,
    sparse_count: int,
    layout_query: str = "TND",
    sparse_mode: int = 3,
    split_op: Optional[object] = None,
) -> torch.Tensor:
    """Indexer top-k against the packed cache.

    Dispatches to the packed-cache op when one is registered.  Otherwise unpacks
    and calls ``split_op`` -- the vendor ``npu_quant_lightning_indexer`` -- so the
    scores come from the same kernel as the split layout and only the layout
    differs.  The caller passes ``split_op`` rather than this module importing
    ``torch_npu``, which keeps the fallback testable off an NPU.
    """
    native = native_packed_indexer_op()
    if native is not None:
        return native(
            query,
            key_with_scale,
            weights,
            query_dequant_scale,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table,
            index_head_dim,
            sparse_count,
            layout_query,
            sparse_mode,
        )
    triton_topk = _triton_packed_indexer()
    if triton_topk is not None and query.ndim == 3:
        page_size = key_with_scale.shape[1]
        supported, topk = triton_topk
        if supported(query.shape[1], index_head_dim, page_size):
            return topk(
                cache=key_with_scale,
                query=query,
                weights=weights,
                query_dequant_scale=query_dequant_scale,
                block_table=block_table,
                cumulative_seq_lengths_query=actual_seq_lengths_query,
                seq_lengths_key=actual_seq_lengths_key,
                index_head_dim=index_head_dim,
                page_size=page_size,
                sparse_count=sparse_count,
            )
    if split_op is None:
        raise RuntimeError(
            "No packed-cache Indexer op is registered, the Triton kernel does "
            "not cover this shape, and no split-layout op was supplied to fall "
            f"back to. Register torch.ops.npu.{_NATIVE_OP_NAME}, or run with "
            "SGLANG_NPU_ENABLE_PACKED_INDEXER_CACHE=0."
        )
    key, key_scale = unpack_index_k_with_scale(key_with_scale, index_head_dim)
    return split_op(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_dequant_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query=layout_query,
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        query_quant_mode=0,
        key_quant_mode=0,
    )
