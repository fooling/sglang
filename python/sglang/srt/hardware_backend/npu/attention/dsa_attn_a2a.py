# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Attention-TP all-to-all for the NPU DSA sparse attention op.

Why
---
With attention TP the sparse attention op is called per rank as

    query [num_tokens, q_head_num / attn_tp, head_dim]
    -> attn_out [num_tokens, q_head_num / attn_tp, head_dim]

The head count is the axis this op's efficiency turns on, and attention TP
shards exactly that axis.  The DSA KV is multi-query (kv head num == 1), so one
gathered KV entry feeds ``q_head_num / attn_tp`` multiply-accumulates instead of
``q_head_num``: the query side of the call is too narrow to pay for the KV it
moves, and the op lands far below the throughput the same flops reach at the
full head count.

Nothing forces the query to stay where it is, though.  Every attn-TP rank holds
the *whole* KV cache, so the tokens can be split instead of the heads:

    [T, H / tp, D]  --all-to-all-->  [T / tp, H, D]
    sparse attention on [T / tp, H, D]
    [T / tp, H, D]  --all-to-all-->  [T, H / tp, D]

The flops per rank are identical -- ``T * H / tp`` token-head pairs either way.
What changes is the shape the op is handed, and, as a second-order effect, that
each rank now gathers KV for ``1 / tp`` of the tokens rather than all of them.
Only the query and the output move; key/value stay put because they are
replicated and head-count 1.

Layout
------
``all_to_all_single`` splits and concatenates along dim 0, so the peer axis has
to be dim 0 on both sides:

    scatter:  q[T_pad, Hl, D].view(tp, T_pad/tp, Hl, D)   # chunk j -> rank j
              a2a -> recv[i] = head shard i of our token chunk
              permute(1, 0, 2, 3) -> [T_pad/tp, tp * Hl, D]
    gather:   o[T_pad/tp, tp * Hl, D].view(T_pad/tp, tp, Hl, D)
              transpose(0, 1)                             # head shard j -> rank j
              a2a -> recv[j] = token chunk j of our head shard
              view -> [T_pad, Hl, D]

Global head index is ``src_rank * (H / tp) + local_head``, which is exactly the
head order a column-parallel q projection produces, so no head permutation is
needed anywhere else.

Metadata
--------
``sparse_indices`` / ``actual_seq_lengths_query`` / ``actual_seq_lengths_kv`` /
``block_table`` all describe tokens, so they have to be narrowed to the token
window this rank now owns.  Two shapes of that problem:

* ``uniform`` (decode, target-verify, draft-extend): every request contributes
  the same number of query tokens, so a token window is a whole number of
  requests.  The per-rank metadata is a plain row slice, and the op sees
  ``B / tp`` segments instead of ``B``.  Query rows past the last request (DP
  attention pads the query to the global maximum) become padding slots with KV
  length 0.
* ``general`` (extend/prefill): the window cuts sequences.  Every batch stays in
  the segment list, with its query length clamped to the window (zero outside
  it) and its KV length shortened to the window end so that the op's
  bottom-right query/KV alignment still puts each query token at its true
  position, plus one trailing segment for the query rows no batch accounts for.
  ``block_table`` keeps its rows and gains one zeroed row for that segment.

Both transforms are elementwise device ops on static shapes: no host sync, no
data-dependent shapes, graph-capturable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.utils import is_npu

_is_npu = is_npu()

__all__ = [
    "DsaA2APlan",
    "make_dsa_a2a_plan",
    "scatter_query",
    "gather_attn_out",
    "local_sparse_indices",
    "local_seq_metadata_uniform",
    "local_seq_metadata_general",
]


@dataclass(frozen=True)
class DsaA2APlan:
    """Static description of one attention-TP query exchange."""

    tp_size: int
    tp_rank: int

    # token axis
    num_tokens: int  # T, before padding
    padded_num_tokens: int  # T_pad, multiple of tp_size
    tokens_per_rank: int  # T_pad // tp_size

    # batch (segment) axis
    num_batches: int  # B, the requests that carry real metadata rows
    num_segments: int  # batch slots the token axis spans; >= B under token padding
    padded_num_batches: int  # B_pad; == B in general mode
    batches_per_rank: int  # B_pad // tp_size; 0 in general mode

    # tokens contributed by every batch; 0 when not uniform
    tokens_per_batch: int
    uniform: bool

    @property
    def token_begin(self) -> int:
        return self.tp_rank * self.tokens_per_rank

    @property
    def token_end(self) -> int:
        return (self.tp_rank + 1) * self.tokens_per_rank

    @property
    def batch_begin(self) -> int:
        return self.tp_rank * self.batches_per_rank

    @property
    def batch_end(self) -> int:
        return (self.tp_rank + 1) * self.batches_per_rank

    @property
    def token_padding(self) -> int:
        return self.padded_num_tokens - self.num_tokens

    @property
    def batch_padding(self) -> int:
        return self.padded_num_batches - self.num_batches


def make_dsa_a2a_plan(
    num_tokens: int,
    num_batches: int,
    tp_size: int,
    tp_rank: int,
    tokens_per_batch: int = 0,
) -> DsaA2APlan:
    """Build the exchange plan.

    ``tokens_per_batch > 0`` selects the uniform mode, where the token axis
    divides into equal per-request slots; batch slots are then padded to a
    multiple of ``tp_size`` so a token chunk never splits a request.  The token
    axis may already carry padding of its own (DP attention pads the query to
    the global maximum while the metadata stays at the local batch size), so the
    slot count can exceed ``num_batches``; the surplus slots are treated exactly
    like the ones this function adds.  Otherwise only the token axis is padded.
    """
    assert tp_size >= 1 and 0 <= tp_rank < tp_size
    assert num_tokens > 0

    uniform = tokens_per_batch > 0
    if uniform:
        assert num_tokens % tokens_per_batch == 0, (
            f"uniform mode expects num_tokens ({num_tokens}) to be a multiple of "
            f"tokens_per_batch ({tokens_per_batch})"
        )
        num_segments = num_tokens // tokens_per_batch
        assert num_segments >= num_batches, (
            f"uniform mode expects at least one batch slot per request, got "
            f"{num_segments} slots for {num_batches} requests"
        )
        padded_num_batches = -(-num_segments // tp_size) * tp_size
        batches_per_rank = padded_num_batches // tp_size
        padded_num_tokens = padded_num_batches * tokens_per_batch
    else:
        num_segments = num_batches
        padded_num_batches = num_batches
        batches_per_rank = 0
        padded_num_tokens = -(-num_tokens // tp_size) * tp_size

    return DsaA2APlan(
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_tokens=num_tokens,
        padded_num_tokens=padded_num_tokens,
        tokens_per_rank=padded_num_tokens // tp_size,
        num_batches=num_batches,
        num_segments=num_segments,
        padded_num_batches=padded_num_batches,
        batches_per_rank=batches_per_rank,
        tokens_per_batch=tokens_per_batch if uniform else 0,
        uniform=uniform,
    )


def _all_to_all(group, output: torch.Tensor, input_: torch.Tensor) -> None:
    """Even-split all-to-all over flat views (dim 0 is the peer axis)."""
    if group is None or getattr(group, "world_size", 1) == 1:
        output.copy_(input_)
        return
    out_flat, in_flat = output.view(-1), input_.reshape(-1)
    if _is_npu and hasattr(group, "_all_to_all_single"):
        # Same reason as the NPU bypass in GroupCoordinator.all_gather_into_tensor:
        # go straight to the collective instead of through the custom-op shim,
        # which Dynamo rewrites on this backend.
        group._all_to_all_single(out_flat, in_flat)
    else:
        group.all_to_all_single(out_flat, in_flat)


def scatter_query(
    query: torch.Tensor,
    plan: DsaA2APlan,
    group,
) -> torch.Tensor:
    """[T, H/tp, D] -> [T_pad/tp, H, D] (this rank's token chunk, all heads)."""
    num_tokens, heads_per_rank, head_dim = query.shape
    assert (
        num_tokens == plan.num_tokens
    ), f"query rows ({num_tokens}) do not match the plan ({plan.num_tokens})"
    n = plan.tp_size

    if plan.token_padding:
        query = torch.nn.functional.pad(query, (0, 0, 0, 0, 0, plan.token_padding))
    send = query.contiguous().view(n, plan.tokens_per_rank, heads_per_rank, head_dim)
    recv = torch.empty_like(send)
    _all_to_all(group, recv, send)

    # recv[i] holds head shard i for our token chunk -> [T_pad/tp, H, D]
    return (
        recv.permute(1, 0, 2, 3)
        .contiguous()
        .view(plan.tokens_per_rank, n * heads_per_rank, head_dim)
    )


def gather_attn_out(
    attn_out: torch.Tensor,
    plan: DsaA2APlan,
    group,
) -> torch.Tensor:
    """[T_pad/tp, H, D] -> [T, H/tp, D] (all tokens, this rank's head shard)."""
    tokens_per_rank, num_heads, head_dim = attn_out.shape
    assert tokens_per_rank == plan.tokens_per_rank
    n = plan.tp_size
    assert num_heads % n == 0, f"head count ({num_heads}) not divisible by tp ({n})"
    heads_per_rank = num_heads // n

    send = (
        attn_out.view(tokens_per_rank, n, heads_per_rank, head_dim)
        .transpose(0, 1)
        .contiguous()
    )
    recv = torch.empty_like(send)
    _all_to_all(group, recv, send)

    # recv[j] holds token chunk j of our head shard -> [T_pad, H/tp, D]
    out = recv.view(plan.padded_num_tokens, heads_per_rank, head_dim)
    return out[: plan.num_tokens]


def local_sparse_indices(
    sparse_indices: torch.Tensor,
    plan: DsaA2APlan,
) -> torch.Tensor:
    """Rows [token_begin, token_end) of the topk index table, -1 padded.

    ``sparse_indices`` is [T, K] or [T, 1, K]; every attn-TP rank computes the
    same table (the DSA indexer is replicated), so no communication is needed.
    """
    begin, end = plan.token_begin, plan.token_end
    rows = sparse_indices.shape[0]
    if end <= rows:
        return sparse_indices[begin:end]

    tail_shape = (end - max(begin, rows),) + tuple(sparse_indices.shape[1:])
    tail = torch.full(
        tail_shape, -1, dtype=sparse_indices.dtype, device=sparse_indices.device
    )
    if begin >= rows:
        return tail
    return torch.cat([sparse_indices[begin:rows], tail], dim=0)


def local_seq_metadata_uniform(
    plan: DsaA2APlan,
    actual_seq_lengths_kv: torch.Tensor,
    block_table: Optional[torch.Tensor],
    q_len_cumsum_cache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Per-rank (query cumsum, kv lengths, block table) for the uniform mode.

    Returns the cumulative query lengths of this rank's ``B/tp`` batches
    (``u, 2u, ...``), their KV lengths and their block-table rows.  Padded
    batches get KV length 0 and a zeroed block-table row, which is exactly how
    graph-mode padding is already fed to the op.

    The query cumsum is rebuilt from the plan rather than sliced from the
    caller's, because the caller's covers all ``B`` batches while this rank
    keeps ``B/tp`` of them.  That is only valid while every producer of
    ``actual_seq_lengths_q`` in this mode emits a plain arange of step ``u``
    (ascend_backend.py builds it that way for decode, target-verify and
    draft-extend, eager and graph alike); the caller checks the token count
    against ``B * u`` before selecting this mode, which is necessary but not
    sufficient, so keep the two in step.
    """
    assert plan.uniform
    begin, end = plan.batch_begin, plan.batch_end
    device = actual_seq_lengths_kv.device

    if q_len_cumsum_cache is not None:
        q_cumsum = q_len_cumsum_cache
    else:
        u = plan.tokens_per_batch
        q_cumsum = torch.arange(
            u, plan.batches_per_rank * u + 1, u, dtype=torch.int32, device=device
        )

    kv = _slice_rows(
        actual_seq_lengths_kv, begin, end, plan.num_batches, plan.padded_num_batches
    )
    bt = (
        None
        if block_table is None
        else _slice_rows(
            block_table, begin, end, plan.num_batches, plan.padded_num_batches
        )
    )
    return q_cumsum, kv, bt


def _slice_rows(
    tensor: torch.Tensor, begin: int, end: int, valid_rows: int, padded_rows: int
) -> torch.Tensor:
    """``tensor[begin:end]``, zeroing everything past row ``valid_rows``.

    Metadata buffers can be longer than the batch count (graph mode keeps them
    at the captured size), so rows past the last real batch are stale, not
    absent: zero them rather than slicing them.
    """
    if end <= valid_rows:
        return tensor[begin:end]

    padded = torch.zeros(
        (padded_rows,) + tuple(tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    keep = min(valid_rows, tensor.shape[0])
    padded[:keep].copy_(tensor[:keep])
    return padded[begin:end]


def local_seq_metadata_general(
    plan: DsaA2APlan,
    actual_seq_lengths_q: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Per-rank (query cumsum, kv lengths) for a window that cuts sequences.

    ``actual_seq_lengths_q`` is the cumulative query length per batch, in
    extend-local coordinates; ``actual_seq_lengths_kv`` is prefix + extend.  For
    the window ``[s, e)``:

        q_cumsum'[b] = clamp(q_cumsum[b] - s, 0, e - s)
        kv'[b]       = clamp(kv[b] - relu(q_cumsum[b] - e), min=0)

    Shortening the KV length by the part of the batch that lives past the window
    keeps the op's bottom-right alignment honest: query token ``i`` of the
    truncated segment still sits at KV position ``kv'[b] - len' + i``, its true
    position in the sequence.  Batches outside the window collapse to
    zero-length segments.

    One trailing segment is always appended, covering the query rows the batches
    do not account for (KV length 0, all-``-1`` sparse indices, one zeroed
    block-table row).  In extend the token axis is padded but ``extend_seq_lens``
    is not, so the query tensor is routinely longer than the last cumulative
    length -- and a rank whose whole window falls in that padding would otherwise
    hand the op query rows that belong to no segment at all.  When the window is
    fully covered the appended segment is empty, like any other out-of-window
    batch.
    """
    s, e = plan.token_begin, plan.token_end
    chunk = plan.tokens_per_rank
    assert actual_seq_lengths_q.shape[0] == actual_seq_lengths_kv.shape[0], (
        f"query cumsum ({actual_seq_lengths_q.shape[0]}) and KV lengths "
        f"({actual_seq_lengths_kv.shape[0]}) must describe the same batches"
    )

    # actual_seq_lengths_kv can still be the CPU mirror here; align both sides
    device = device if device is not None else actual_seq_lengths_q.device
    cum_q = actual_seq_lengths_q.to(device=device, dtype=torch.int64)
    kv = actual_seq_lengths_kv.to(device=device, dtype=torch.int64)

    q_local = (cum_q - s).clamp_(min=0, max=chunk)
    kv_local = (kv - (cum_q - e).clamp_(min=0)).clamp_(min=0)
    # batches with no query token in this window keep a meaningless KV length
    # from the clamp above; zero it so the segment is empty on both axes
    seg_len = q_local - torch.cat(
        [torch.zeros(1, dtype=q_local.dtype, device=device), q_local[:-1]]
    )
    kv_local = torch.where(seg_len > 0, kv_local, torch.zeros_like(kv_local))

    # trailing segment for every query row no batch accounts for; its length is
    # chunk - q_local[-1] >= 0, so this is a no-op segment when the window is
    # fully covered and the whole window when it is pure padding
    q_local = torch.cat(
        [q_local, torch.full((1,), chunk, dtype=q_local.dtype, device=device)]
    )
    kv_local = torch.cat(
        [kv_local, torch.zeros(1, dtype=kv_local.dtype, device=device)]
    )
    if block_table is not None:
        block_table = torch.cat(
            [block_table, block_table.new_zeros((1,) + tuple(block_table.shape[1:]))]
        )
    return q_local.to(torch.int32), kv_local.to(torch.int32), block_table
