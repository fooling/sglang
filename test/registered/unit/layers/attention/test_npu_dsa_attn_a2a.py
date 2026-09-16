"""CPU coverage for the attention-TP query all-to-all of the NPU DSA sparse attention.

The exchange turns a per-rank call of shape [T, H/tp, D] into [T/tp, H, D] so the
sparse KV gather is done once per token instead of once per token per rank.  The
tests here pin down the two things that can silently corrupt results:

* the head/token transposition (which head shard and which token chunk end up
  where), simulated for all ranks in one process, and
* the metadata narrowing (query cumsum, KV lengths, block table, sparse
  indices), checked end to end against a reference sparse attention.
"""

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention.dsa_attn_a2a import (
    gather_attn_out,
    local_seq_metadata_general,
    local_seq_metadata_uniform,
    local_sparse_indices,
    make_dsa_a2a_plan,
    scatter_query,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


# --------------------------------------------------------------------------- #
# all-to-all simulation
# --------------------------------------------------------------------------- #


class _RecordGroup:
    """Captures the send buffer of one rank; leaves the output untouched."""

    def __init__(self, world_size, sends, rank):
        self.world_size = world_size
        self._sends = sends
        self._rank = rank

    def all_to_all_single(self, output, input_):
        self._sends[self._rank] = input_.detach().clone()


class _ReplayGroup:
    """Serves rank ``rank`` the chunks every other rank sent to it."""

    def __init__(self, world_size, sends, rank):
        self.world_size = world_size
        self._sends = sends
        self._rank = rank

    def all_to_all_single(self, output, input_):
        n = self.world_size
        out = output.view(n, -1)
        for src in range(n):
            out[src].copy_(self._sends[src].view(n, -1)[self._rank])


def _run_all_ranks(fn, world_size):
    """Run ``fn(rank, group)`` for every rank, resolving the all-to-all in lockstep.

    ``fn`` is executed twice: once to record what each rank sends, once to let
    each rank read what it was sent.  The results of the second pass are the
    real ones.
    """
    sends = {}
    for rank in range(world_size):
        fn(rank, _RecordGroup(world_size, sends, rank))
    return [
        fn(rank, _ReplayGroup(world_size, sends, rank)) for rank in range(world_size)
    ]


# --------------------------------------------------------------------------- #
# reference sparse attention (mirrors the op's TND / paged-KV semantics)
# --------------------------------------------------------------------------- #


def _ref_sparse_attention(
    query,  # [T, H, Dq]
    kv_cache,  # [num_slots, Dq]  (value = first Dv columns, as in MLA)
    sparse_indices,  # [T, K] positions inside the sequence, -1 = empty
    q_cumsum,  # [B] cumulative query lengths
    kv_lens,  # [B]
    block_table,  # [B, max_blocks]
    page_size,
    value_dim,
    scale,
):
    num_tokens, num_heads, _ = query.shape
    out = torch.zeros(num_tokens, num_heads, value_dim, dtype=torch.float32)
    begin = 0
    for b in range(q_cumsum.shape[0]):
        end = int(q_cumsum[b])
        assert end >= begin, "cumulative query lengths must be non-decreasing"
        kv_len = int(kv_lens[b])
        for t in range(begin, min(end, num_tokens)):
            positions = [
                int(p) for p in sparse_indices[t].tolist() if 0 <= int(p) < kv_len
            ]
            if not positions:
                continue
            slots = [
                int(block_table[b, p // page_size]) * page_size + p % page_size
                for p in positions
            ]
            keys = kv_cache[slots].to(torch.float32)  # [n, Dq]
            logits = torch.einsum("hd,nd->hn", query[t].to(torch.float32), keys) * scale
            probs = torch.softmax(logits, dim=-1)
            out[t] = torch.einsum("hn,nd->hd", probs, keys[:, :value_dim])
        begin = max(end, begin)
    return out


# --------------------------------------------------------------------------- #
# plan arithmetic
# --------------------------------------------------------------------------- #


def test_plan_uniform_pads_batches_not_just_tokens():
    plan = make_dsa_a2a_plan(
        num_tokens=14, num_batches=7, tp_size=4, tp_rank=2, tokens_per_batch=2
    )
    assert plan.uniform
    assert plan.padded_num_batches == 8
    assert plan.batches_per_rank == 2
    assert plan.padded_num_tokens == 16
    assert plan.tokens_per_rank == 4
    assert (plan.token_begin, plan.token_end) == (8, 12)
    assert (plan.batch_begin, plan.batch_end) == (4, 6)


def test_plan_general_pads_tokens_only():
    plan = make_dsa_a2a_plan(num_tokens=10, num_batches=3, tp_size=4, tp_rank=3)
    assert not plan.uniform
    assert plan.padded_num_tokens == 12
    assert plan.tokens_per_rank == 3
    assert (plan.token_begin, plan.token_end) == (9, 12)
    assert plan.padded_num_batches == 3


def test_plan_rejects_inconsistent_uniform_shape():
    with pytest.raises(AssertionError):
        make_dsa_a2a_plan(
            num_tokens=13, num_batches=7, tp_size=4, tp_rank=0, tokens_per_batch=2
        )


# --------------------------------------------------------------------------- #
# transposition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("num_tokens,tp_size", [(8, 4), (7, 4), (5, 2), (3, 4)])
def test_scatter_then_gather_is_identity(num_tokens, tp_size):
    heads_per_rank, head_dim = 3, 5
    torch.manual_seed(0)
    global_q = torch.randn(num_tokens, heads_per_rank * tp_size, head_dim)

    def scatter(rank, group):
        plan = make_dsa_a2a_plan(num_tokens, num_tokens, tp_size, rank)
        shard = global_q[:, rank * heads_per_rank : (rank + 1) * heads_per_rank, :]
        return scatter_query(shard.contiguous(), plan, group)

    scattered = _run_all_ranks(scatter, tp_size)

    # every rank now holds all heads of its own token chunk
    for rank, got in enumerate(scattered):
        plan = make_dsa_a2a_plan(num_tokens, num_tokens, tp_size, rank)
        want = torch.zeros(plan.tokens_per_rank, global_q.shape[1], head_dim)
        real = max(min(plan.token_end, num_tokens) - plan.token_begin, 0)
        if real > 0:
            want[:real] = global_q[plan.token_begin : plan.token_begin + real]
        torch.testing.assert_close(got, want)

    def gather(rank, group):
        plan = make_dsa_a2a_plan(num_tokens, num_tokens, tp_size, rank)
        return gather_attn_out(scattered[rank], plan, group)

    gathered = _run_all_ranks(gather, tp_size)
    for rank, got in enumerate(gathered):
        want = global_q[:, rank * heads_per_rank : (rank + 1) * heads_per_rank, :]
        assert got.shape == want.shape
        torch.testing.assert_close(got, want)


def test_single_rank_is_a_noop():
    q = torch.randn(5, 4, 6)
    plan = make_dsa_a2a_plan(5, 5, tp_size=1, tp_rank=0)
    scattered = scatter_query(q, plan, group=None)
    torch.testing.assert_close(scattered, q)
    torch.testing.assert_close(gather_attn_out(scattered, plan, group=None), q)


def test_local_sparse_indices_pads_with_minus_one():
    idx = torch.arange(5 * 3, dtype=torch.int32).view(5, 3)
    plan = make_dsa_a2a_plan(num_tokens=5, num_batches=5, tp_size=2, tp_rank=1)
    got = local_sparse_indices(idx, plan)
    assert got.shape == (3, 3)
    torch.testing.assert_close(got[:2], idx[3:5])
    assert bool((got[2] == -1).all())


# --------------------------------------------------------------------------- #
# end-to-end equivalence
# --------------------------------------------------------------------------- #


def _build_case(seq_lens, extend_lens, topk, page_size, num_heads, dims, seed=0):
    """Returns query, kv cache, sparse indices, cumsums, kv lens and block table."""
    torch.manual_seed(seed)
    head_dim, value_dim = dims
    num_batches = len(seq_lens)
    max_blocks = max((s + page_size - 1) // page_size for s in seq_lens)
    num_slots = num_batches * max_blocks * page_size

    kv_cache = torch.randn(num_slots, head_dim)
    # a deliberately shuffled block table, so a wrong batch row cannot pass
    block_table = torch.randperm(num_batches * max_blocks).view(num_batches, max_blocks)

    num_tokens = sum(extend_lens)
    query = torch.randn(num_tokens, num_heads, head_dim)

    sparse_indices = torch.full((num_tokens, topk), -1, dtype=torch.int32)
    row = 0
    for b, (kv_len, ext) in enumerate(zip(seq_lens, extend_lens)):
        prefix = kv_len - ext
        for i in range(ext):
            # causal: token i of the extend part may look at positions <= prefix + i
            limit = prefix + i + 1
            count = min(topk, limit)
            perm = torch.randperm(limit)[:count]
            sparse_indices[row, :count] = perm.to(torch.int32)
            row += 1

    q_cumsum = torch.tensor(extend_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32)
    return query, kv_cache, sparse_indices, q_cumsum, kv_lens, block_table


def _reference_full(case, page_size, value_dim, scale):
    query, kv_cache, sparse_indices, q_cumsum, kv_lens, block_table = case
    return _ref_sparse_attention(
        query,
        kv_cache,
        sparse_indices,
        q_cumsum,
        kv_lens,
        block_table,
        page_size,
        value_dim,
        scale,
    )


def _a2a_result(case, plan_kwargs, tp_size, page_size, value_dim, scale, uniform):
    """Run the exchanged path on every rank and stitch the per-rank outputs back."""
    query, kv_cache, sparse_indices, q_cumsum, kv_lens, block_table = case
    num_tokens, num_heads, head_dim = query.shape
    assert num_heads % tp_size == 0
    heads_per_rank = num_heads // tp_size

    def scatter(rank, group):
        plan = make_dsa_a2a_plan(tp_rank=rank, tp_size=tp_size, **plan_kwargs)
        shard = query[:, rank * heads_per_rank : (rank + 1) * heads_per_rank, :]
        return scatter_query(shard.contiguous(), plan, group)

    local_q = _run_all_ranks(scatter, tp_size)

    local_out = []
    for rank in range(tp_size):
        plan = make_dsa_a2a_plan(tp_rank=rank, tp_size=tp_size, **plan_kwargs)
        idx = local_sparse_indices(sparse_indices, plan)
        if uniform:
            q_cum_l, kv_l, bt_l = local_seq_metadata_uniform(plan, kv_lens, block_table)
        else:
            q_cum_l, kv_l, bt_l = local_seq_metadata_general(
                plan, q_cumsum, kv_lens, block_table
            )
        out = _ref_sparse_attention(
            local_q[rank],
            kv_cache,
            idx,
            q_cum_l,
            kv_l,
            bt_l,
            page_size,
            value_dim,
            scale,
        )
        local_out.append(out.to(query.dtype))

    def gather(rank, group):
        plan = make_dsa_a2a_plan(tp_rank=rank, tp_size=tp_size, **plan_kwargs)
        return gather_attn_out(local_out[rank], plan, group)

    gathered = _run_all_ranks(gather, tp_size)
    return torch.cat(gathered, dim=1)  # [T, H, Dv] in global head order


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("batch_size", [8, 7])
def test_decode_path_matches_reference(tp_size, batch_size):
    page_size, topk, num_heads = 4, 6, 8
    head_dim, value_dim = 6, 4
    scale = 0.5
    seq_lens = [3 + 5 * i for i in range(batch_size)]
    extend_lens = [1] * batch_size
    case = _build_case(
        seq_lens, extend_lens, topk, page_size, num_heads, (head_dim, value_dim)
    )

    want = _reference_full(case, page_size, value_dim, scale)
    got = _a2a_result(
        case,
        dict(num_tokens=batch_size, num_batches=batch_size, tokens_per_batch=1),
        tp_size,
        page_size,
        value_dim,
        scale,
        uniform=True,
    )
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_target_verify_path_matches_reference(tp_size):
    page_size, topk, num_heads = 4, 6, 8
    head_dim, value_dim = 6, 4
    scale = 0.5
    batch_size, draft_tokens = 5, 3
    seq_lens = [7 + 4 * i for i in range(batch_size)]
    extend_lens = [draft_tokens] * batch_size
    case = _build_case(
        seq_lens, extend_lens, topk, page_size, num_heads, (head_dim, value_dim)
    )

    want = _reference_full(case, page_size, value_dim, scale)
    got = _a2a_result(
        case,
        dict(
            num_tokens=batch_size * draft_tokens,
            num_batches=batch_size,
            tokens_per_batch=draft_tokens,
        ),
        tp_size,
        page_size,
        value_dim,
        scale,
        uniform=True,
    )
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tp_size", [2, 3, 4])
def test_extend_path_matches_reference(tp_size):
    """Ragged sequences: the token window cuts batches in the middle."""
    page_size, topk, num_heads = 4, 6, 12
    head_dim, value_dim = 6, 4
    scale = 0.5
    extend_lens = [5, 1, 9, 2]
    seq_lens = [9, 6, 9, 11]  # prefix + extend
    case = _build_case(
        seq_lens, extend_lens, topk, page_size, num_heads, (head_dim, value_dim)
    )

    want = _reference_full(case, page_size, value_dim, scale)
    got = _a2a_result(
        case,
        dict(num_tokens=sum(extend_lens), num_batches=len(extend_lens)),
        tp_size,
        page_size,
        value_dim,
        scale,
        uniform=False,
    )
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_general_metadata_keeps_query_positions():
    """A cut batch keeps its true KV positions via the shortened KV length."""
    extend_lens = [5, 3]
    seq_lens = [10, 8]
    q_cumsum = torch.tensor(extend_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32)

    # tp=2 over 8 tokens: rank 0 owns tokens [0, 4), cutting batch 0 (5 tokens)
    plan0 = make_dsa_a2a_plan(num_tokens=8, num_batches=2, tp_size=2, tp_rank=0)
    q0, kv0, _ = local_seq_metadata_general(plan0, q_cumsum, kv_lens)
    # the third entry is the always-appended trailing segment, empty here
    assert q0.tolist() == [4, 4, 4]  # 4 of batch 0, none of batch 1
    # batch 0 kept 4 of its 5 extend tokens -> its KV window ends one token early
    assert kv0.tolist() == [9, 0, 0]

    plan1 = make_dsa_a2a_plan(num_tokens=8, num_batches=2, tp_size=2, tp_rank=1)
    q1, kv1, _ = local_seq_metadata_general(plan1, q_cumsum, kv_lens)
    assert q1.tolist() == [1, 4, 4]  # the 5th token of batch 0, then all of batch 1
    assert kv1.tolist() == [10, 8, 0]


def test_uniform_metadata_pads_missing_batches():
    kv_lens = torch.tensor([11, 12, 13], dtype=torch.int32)
    block_table = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    plan = make_dsa_a2a_plan(
        num_tokens=3, num_batches=3, tp_size=2, tp_rank=1, tokens_per_batch=1
    )
    q_cum, kv, bt = local_seq_metadata_uniform(plan, kv_lens, block_table)
    assert q_cum.tolist() == [1, 2]
    assert kv.tolist() == [13, 0]  # padded batch gets no KV
    assert bt.tolist() == [[5, 6], [0, 0]]


def test_uniform_metadata_ignores_rows_past_the_batch_count():
    """Graph-mode buffers outlive the batch: stale rows must not leak in."""
    # buffer sized for 8 requests, only 3 of them real this step
    kv_lens = torch.tensor([11, 12, 13, 99, 99, 99, 99, 99], dtype=torch.int32)
    block_table = torch.arange(16, dtype=torch.int32).view(8, 2)
    plan = make_dsa_a2a_plan(
        num_tokens=3, num_batches=3, tp_size=2, tp_rank=1, tokens_per_batch=1
    )
    _, kv, bt = local_seq_metadata_uniform(plan, kv_lens, block_table)
    assert kv.tolist() == [13, 0]
    assert bt.tolist() == [[4, 5], [0, 0]]


def test_general_metadata_covers_query_rows_no_batch_accounts_for():
    """Extend pads the token axis but not extend_seq_lens: the gap needs a segment."""
    extend_lens = [3, 2]  # 5 real tokens
    q_cumsum = torch.tensor(extend_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    kv_lens = torch.tensor([9, 8], dtype=torch.int32)

    # the query tensor carries 12 rows: 5 real, 7 of DP / alignment padding
    for rank in range(4):
        plan = make_dsa_a2a_plan(num_tokens=12, num_batches=2, tp_size=4, tp_rank=rank)
        q_cum, kv, _ = local_seq_metadata_general(plan, q_cumsum, kv_lens)
        # the segments must account for every query row this rank holds
        assert int(q_cum[-1]) == plan.tokens_per_rank
        assert len(q_cum) == len(kv) == 3
        assert all(
            int(q_cum[i]) <= int(q_cum[i + 1]) for i in range(len(q_cum) - 1)
        ), q_cum.tolist()

    # rank 2 owns tokens [6, 9): pure padding, so one full-width empty-KV segment
    plan = make_dsa_a2a_plan(num_tokens=12, num_batches=2, tp_size=4, tp_rank=2)
    q_cum, kv, bt = local_seq_metadata_general(
        plan, q_cumsum, kv_lens, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    )
    assert q_cum.tolist() == [0, 0, 3]
    assert kv.tolist() == [0, 0, 0]
    assert bt.tolist() == [[1, 2], [3, 4], [0, 0]]


@pytest.mark.parametrize("tp_size", [2, 4])
def test_extend_with_padded_token_axis_matches_reference(tp_size):
    """Same as the ragged case, but the query tensor is longer than the batches."""
    page_size, topk, num_heads = 4, 6, 12
    head_dim, value_dim = 6, 4
    scale = 0.5
    extend_lens = [5, 1, 3]
    seq_lens = [9, 6, 7]
    case = _build_case(
        seq_lens, extend_lens, topk, page_size, num_heads, (head_dim, value_dim)
    )
    query, kv_cache, sparse_indices, q_cumsum, kv_lens, block_table = case

    # pad the token axis the way DP attention does, leaving the extra rows
    # unaccounted for by any batch (-1 sparse indices, as _pad_topk_indices does)
    padded_tokens = 16
    pad = padded_tokens - query.shape[0]
    query = torch.cat([query, torch.zeros(pad, num_heads, head_dim)])
    sparse_indices = torch.cat(
        [sparse_indices, torch.full((pad, topk), -1, dtype=sparse_indices.dtype)]
    )
    padded_case = (query, kv_cache, sparse_indices, q_cumsum, kv_lens, block_table)

    want = _reference_full(padded_case, page_size, value_dim, scale)
    got = _a2a_result(
        padded_case,
        dict(num_tokens=padded_tokens, num_batches=len(extend_lens)),
        tp_size,
        page_size,
        value_dim,
        scale,
        uniform=False,
    )
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_decode_with_dp_padded_token_axis_matches_reference(tp_size):
    """DP attention pads the query past the batch; uniform mode must absorb it."""
    page_size, topk, num_heads = 4, 6, 8
    head_dim, value_dim = 6, 4
    scale = 0.5
    batch_size, padded_tokens = 5, 8
    seq_lens = [3 + 5 * i for i in range(batch_size)]
    case = _build_case(
        seq_lens, [1] * batch_size, topk, page_size, num_heads, (head_dim, value_dim)
    )
    query, kv_cache, sparse_indices, q_cumsum, kv_lens, block_table = case
    pad = padded_tokens - batch_size
    query = torch.cat([query, torch.zeros(pad, num_heads, head_dim)])
    sparse_indices = torch.cat(
        [sparse_indices, torch.full((pad, topk), -1, dtype=sparse_indices.dtype)]
    )
    padded_case = (query, kv_cache, sparse_indices, q_cumsum, kv_lens, block_table)

    want = _reference_full(padded_case, page_size, value_dim, scale)
    got = _a2a_result(
        padded_case,
        dict(num_tokens=padded_tokens, num_batches=batch_size, tokens_per_batch=1),
        tp_size,
        page_size,
        value_dim,
        scale,
        uniform=True,
    )
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_uniform_plan_absorbs_surplus_token_slots():
    plan = make_dsa_a2a_plan(
        num_tokens=8, num_batches=5, tp_size=4, tp_rank=3, tokens_per_batch=1
    )
    assert plan.uniform
    assert (plan.num_batches, plan.num_segments) == (5, 8)
    assert plan.padded_num_batches == 8
    assert (plan.batch_begin, plan.batch_end) == (6, 8)

    kv_lens = torch.arange(11, 16, dtype=torch.int32)  # only 5 real rows
    block_table = torch.arange(10, dtype=torch.int32).view(5, 2)
    q_cum, kv, bt = local_seq_metadata_uniform(plan, kv_lens, block_table)
    assert q_cum.tolist() == [1, 2]
    assert kv.tolist() == [0, 0]  # both slots are padding
    assert bt.tolist() == [[0, 0], [0, 0]]
