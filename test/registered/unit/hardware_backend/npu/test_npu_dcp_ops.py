"""CPU unit tests for the Ascend NPU decode-context-parallel (DCP) helpers in
``sglang.srt.hardware_backend.npu.dcp.ops``.

Collectives are simulated in one process: every DCP rank runs in its own
thread against a barrier-synchronised fake ``GroupCoordinator``.

Usage:
    python -m pytest test_npu_dcp_ops.py -v
"""

import math
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.model_overrides import kimi_k3 as kimi_k3_overrides
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dcp import ops as dcp_ops
from sglang.srt.hardware_backend.npu.dcp.ops import (
    attention_update_reference,
    dcp_block_tables,
    dcp_gather_chunk_rows,
    dcp_interleave_pages,
    dcp_local_seq_lens,
    dcp_merge_a2a,
    dcp_merge_a2a_npu,
    dcp_merge_a2a_vllm,
    dcp_merge_ag_rs,
    dcp_physical_write_loc,
    dcp_prefix_chunk_plan,
    lse_combine,
    mla_decode_with_lse_torch,
    npu_attention_update,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _SharedGroupState:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, timeout=60)
        self.inputs = {}


class _FakeGroup:
    """Per-rank view of a simulated GroupCoordinator."""

    def __init__(self, state: _SharedGroupState, rank: int):
        self.state = state
        self.world_size = state.world_size
        self.rank_in_group = rank
        self._call = 0

    def _exchange(self, tensor: torch.Tensor):
        call = self._call
        self._call += 1
        self.state.inputs[(call, self.rank_in_group)] = tensor.clone()
        self.state.barrier.wait()
        return [self.state.inputs[(call, r)] for r in range(self.world_size)]

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.cat(self._exchange(input_), dim=dim)

    def all_to_all_single(self, output: torch.Tensor, input: torch.Tensor) -> None:
        inputs = self._exchange(input)
        n = self.world_size
        chunk = input.numel() // n
        recv = [
            inp.view(-1)[self.rank_in_group * chunk : (self.rank_in_group + 1) * chunk]
            for inp in inputs
        ]
        output.view(-1).copy_(torch.cat(recv))

    def reduce_scatter_along_dim(self, input_: torch.Tensor, dim: int = -1):
        total = torch.stack(self._exchange(input_)).sum(dim=0)
        return total.chunk(self.world_size, dim=dim)[self.rank_in_group].contiguous()


def _run_ranks(world_size: int, fn):
    """Run fn(rank, group) on one thread per rank; return per-rank results."""
    state = _SharedGroupState(world_size)
    results = [None] * world_size
    errors = []

    def target(rank):
        try:
            results[rank] = fn(rank, _FakeGroup(state, rank))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            state.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def _build_layout(seq_lens, page_size, dcp_size, seed=0):
    """Allocate virtual KV slots like the DCP scheduler does (allocator page
    size = page_size * dcp_size) and return (req_to_token, num_pages)."""
    stride = page_size * dcp_size
    max_len = max(seq_lens)
    total_pages = sum((n + stride - 1) // stride for n in seq_lens) + 3
    allocator = PagedTokenToKVPoolAllocator(
        size=total_pages * stride,
        page_size=stride,
        dtype=torch.float32,
        device="cpu",
        kvcache=object(),
        need_sort=False,
    )
    # Scramble the free list so requests get non-contiguous pages.
    g = torch.Generator().manual_seed(seed)
    allocator.free_pages = allocator.free_pages[
        torch.randperm(len(allocator.free_pages), generator=g)
    ]
    req_to_token = torch.zeros(
        (len(seq_lens), (max_len + stride - 1) // stride * stride), dtype=torch.int32
    )
    for i, n in enumerate(seq_lens):
        need = (n + stride - 1) // stride * stride
        if need == 0:
            continue
        locs = allocator.alloc(need)
        assert locs is not None
        req_to_token[i, :need] = locs.to(torch.int32)
    return req_to_token, total_pages + 1


class TestWriteLoc(CustomTestCase):
    def test_matches_triton_kernel_math(self):
        g = torch.Generator().manual_seed(0)
        for c in (1, 2, 4, 8):
            loc = torch.randint(0, 4096, (257,), generator=g)
            loc[:3] = torch.tensor([0, -1, c])
            for r in range(c):
                got = dcp_physical_write_loc(loc, c, r)
                # set_mla_kv_buffer_kernel: valid = loc % c == r -> loc // c.
                for x, y in zip(loc.tolist(), got.tolist()):
                    if c == 1:
                        self.assertEqual(y, x)
                    elif x >= 0 and x % c == r:
                        self.assertEqual(y, x // c)
                    else:
                        self.assertEqual(y, 0)
                self.assertEqual(got.shape, loc.shape)
                self.assertEqual(got.dtype, loc.dtype)

    def test_every_token_owned_once(self):
        loc = torch.arange(64, 512)
        c = 4
        owned = sum((dcp_physical_write_loc(loc, c, r) != 0).long() for r in range(c))
        self.assertTrue(torch.all(owned == 1))


class TestLocalSeqLens(CustomTestCase):
    def test_list_and_tensor(self):
        lens = list(range(0, 40))
        for c in (1, 2, 3, 8):
            for r in range(c):
                as_list = dcp_local_seq_lens(lens, c, r)
                as_tensor = dcp_local_seq_lens(torch.tensor(lens), c, r).tolist()
                ref = [sum(1 for p in range(n) if p % c == r) for n in lens]
                self.assertEqual(as_list, ref)
                self.assertEqual(as_tensor, ref)


class TestLayout(CustomTestCase):
    def test_block_table_reads_owned_tokens_in_order(self):
        for c in (2, 4, 8):
            for page_size in (1, 4, 16):
                seq_lens = [1, 3, page_size * c, page_size * c * 2 + 5, 37]
                req_to_token, num_pages = _build_layout(seq_lens, page_size, c)
                block_table = dcp_block_tables(
                    req_to_token, max(seq_lens), page_size, c
                )
                self.assertEqual(block_table.dtype, torch.int32)
                for r in range(c):
                    # Physical pool of this rank; each slot stores (req, pos).
                    pool = torch.full((num_pages * page_size, 2), -1)
                    for i, n in enumerate(seq_lens):
                        v = req_to_token[i, :n].long()
                        phys = dcp_physical_write_loc(v, c, r)
                        mine = v % c == r
                        pool[phys[mine], 0] = i
                        pool[phys[mine], 1] = torch.arange(n)[mine]
                    pages = pool.view(num_pages, page_size, 2)
                    local = dcp_local_seq_lens(seq_lens, c, r)
                    for i, n in enumerate(seq_lens):
                        npg = (local[i] + page_size - 1) // page_size
                        rows = pages[block_table[i, :npg].long()].reshape(-1, 2)
                        rows = rows[: local[i]]
                        expect = [p for p in range(n) if p % c == r]
                        self.assertTrue(torch.all(rows[:, 0] == i))
                        self.assertEqual(rows[:, 1].tolist(), expect)


class TestInterleave(CustomTestCase):
    def test_restores_global_order(self):
        for c in (2, 4, 8):
            for page_size in (1, 4, 16):
                npages = 3
                total = npages * page_size * c
                pos = torch.arange(total).view(npages, page_size, c)
                gathered = pos.permute(2, 0, 1).unsqueeze(-1).float()
                for total_len in (0, 1, total // 2 + 1, total):
                    rows = dcp_interleave_pages(gathered, total_len)
                    self.assertEqual(rows.shape, (total_len, 1))
                    self.assertEqual(rows[:, 0].long().tolist(), list(range(total_len)))

    def test_gather_chunk_rows(self):
        c, page_size, chunk_tokens = 4, 4, 7
        prefix_lens = [5, 16, 33, 0]
        req_to_token, num_pages = _build_layout(prefix_lens, page_size, c, seed=3)
        tail = 3

        def rank_fn(r, group):
            pool = _owned_pool(prefix_lens, req_to_token, num_pages, page_size, c, r)
            per_req = []
            for i, n in enumerate(prefix_lens):
                rows = []
                for ch in dcp_prefix_chunk_plan(n, chunk_tokens, page_size, c):
                    pages = _chunk_block_ids(req_to_token[i], ch, page_size, c)
                    rows.append(
                        dcp_gather_chunk_rows(
                            pool[pages], ch.row_offset, ch.end - ch.start, group
                        )
                    )
                per_req.append(torch.cat(rows) if rows else torch.empty(0, tail))
            return per_req

        for per_req in _run_ranks(c, rank_fn):
            for i, n in enumerate(prefix_lens):
                rows = per_req[i]
                self.assertEqual(rows.shape, (n, tail))
                self.assertTrue(torch.all(rows[:, 0] == i))
                self.assertEqual(rows[:, 1].long().tolist(), list(range(n)))


def _owned_pool(prefix_lens, req_to_token, num_pages, page_size, c, r, vals=None):
    """Rank r's physical pages [num_pages, P, tail]; default rows (req, pos, -pos)."""
    tail = 3 if vals is None else vals[0].shape[-1]
    pool = torch.zeros(num_pages * page_size, tail)
    for i, n in enumerate(prefix_lens):
        v = req_to_token[i, :n].long()
        mine = v % c == r
        if vals is None:
            rows = torch.stack(
                [
                    torch.full((n,), float(i)),
                    torch.arange(n).float(),
                    -torch.arange(n).float(),
                ],
                dim=-1,
            )
        else:
            rows = vals[i]
        pool[(v // c)[mine]] = rows[mine]
    return pool.view(num_pages, page_size, tail)


def _chunk_block_ids(req_row, chunk, page_size, c):
    stride = page_size * c
    first = chunk.first_page * stride
    return (req_row[first : first + chunk.num_pages * stride : stride] // stride).long()


class TestPrefixChunkPlan(CustomTestCase):
    def test_covers_prefix(self):
        for page_size, c, chunk_tokens in [
            (1, 2, 3),
            (4, 4, 16),
            (4, 4, 7),
            (16, 8, 100),
            (128, 8, 65536),
            (2, 3, 5),
        ]:
            stride = page_size * c
            for prefix_len in (
                0,
                1,
                chunk_tokens,
                chunk_tokens + 1,
                3 * stride + 5,
                257,
            ):
                plan = dcp_prefix_chunk_plan(prefix_len, chunk_tokens, page_size, c)
                self.assertEqual(len(plan), math.ceil(prefix_len / chunk_tokens))
                pos = 0
                for idx, ch in enumerate(plan):
                    self.assertEqual(ch.start, pos)
                    self.assertGreater(ch.end, ch.start)
                    if idx < len(plan) - 1:
                        self.assertEqual(ch.end - ch.start, chunk_tokens)
                    self.assertEqual(ch.first_page, ch.start // stride)
                    self.assertEqual(
                        ch.first_page + ch.num_pages, math.ceil(ch.end / stride)
                    )
                    self.assertEqual(ch.row_offset, ch.start - ch.first_page * stride)
                    self.assertTrue(0 <= ch.row_offset < stride)
                    # The selected pages hold every row of the chunk.
                    self.assertLessEqual(
                        ch.row_offset + ch.end - ch.start, ch.num_pages * stride
                    )
                    pos = ch.end
                self.assertEqual(pos, prefix_len)

    def test_unaligned_example(self):
        # P=4, c=2 -> 8 tokens per page; chunks of 5 over 13 prefix tokens.
        plan = dcp_prefix_chunk_plan(13, 5, 4, 2)
        self.assertEqual(
            [tuple(ch) for ch in plan],
            [(0, 5, 0, 1, 0), (5, 10, 0, 2, 5), (10, 13, 1, 1, 2)],
        )


def _attn_with_lse(q, k, v, scale, causal):
    """q [T, H, d], k [S, d], v [S, dv] -> (lse [T, H], out [T, H, dv]).

    causal: q token i sees k[: S - T + i + 1] (bottom-right aligned)."""
    t, s = q.shape[0], k.shape[0]
    scores = torch.einsum("thd,sd->ths", q, k) * scale
    if causal:
        allowed = torch.arange(s)[None, :] <= torch.arange(t)[:, None] + (s - t)
        scores = scores.masked_fill(~allowed[:, None, :], -math.inf)
    return torch.logsumexp(scores, dim=-1), torch.softmax(scores, dim=-1) @ v


class TestChunkedPrefixExactness(CustomTestCase):
    """c ranks: per-chunk (no mask) + current (causal) attention merged with
    npu_attention_update == single-rank causal attention over prefix+current."""

    HEADS, D_K, D_V = 3, 12, 10

    def test_exact(self):
        cases = [
            # (c, page_size, chunk_tokens, prefix_lens, extend_lens)
            (2, 4, 11, [37, 0, 8], [5, 3, 1]),
            (4, 4, 16, [100, 3], [7, 1]),
            (3, 2, 7, [50], [4]),
        ]
        for seed, (c, page_size, chunk_tokens, prefix_lens, extend_lens) in enumerate(
            cases
        ):
            g = torch.Generator().manual_seed(seed)
            scale = 1.0 / math.sqrt(self.D_K)
            pk = [torch.randn(n, self.D_K, generator=g) for n in prefix_lens]
            pv = [torch.randn(n, self.D_V, generator=g) for n in prefix_lens]
            ck = [torch.randn(n, self.D_K, generator=g) for n in extend_lens]
            cv = [torch.randn(n, self.D_V, generator=g) for n in extend_lens]
            qs = [
                torch.randn(n, self.HEADS, self.D_K, generator=g) for n in extend_lens
            ]
            req_to_token, num_pages = _build_layout(prefix_lens, page_size, c, seed)
            vals = [torch.cat([pk[i], pv[i]], dim=-1) for i in range(len(prefix_lens))]
            self.assertGreater(max(prefix_lens), 2 * chunk_tokens)

            def rank_fn(r, group):
                pool = _owned_pool(
                    prefix_lens, req_to_token, num_pages, page_size, c, r, vals
                )
                outs = []
                for i, n in enumerate(prefix_lens):
                    lse_list, out_list = [], []
                    for ch in dcp_prefix_chunk_plan(n, chunk_tokens, page_size, c):
                        pages = _chunk_block_ids(req_to_token[i], ch, page_size, c)
                        rows = dcp_gather_chunk_rows(
                            pool[pages], ch.row_offset, ch.end - ch.start, group
                        )
                        k_rows, v_rows = rows.split([self.D_K, self.D_V], dim=-1)
                        lse, out = _attn_with_lse(qs[i], k_rows, v_rows, scale, False)
                        lse_list.append(lse.reshape(-1))
                        out_list.append(out.reshape(-1, self.D_V))
                    lse, out = _attn_with_lse(qs[i], ck[i], cv[i], scale, True)
                    lse_list.append(lse.reshape(-1))
                    out_list.append(out.reshape(-1, self.D_V))
                    outs.append(
                        npu_attention_update(lse_list, out_list).view(
                            -1, self.HEADS, self.D_V
                        )
                    )
                return outs

            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                per_rank = _run_ranks(c, rank_fn)
            for i in range(len(prefix_lens)):
                _, ref = _attn_with_lse(
                    qs[i],
                    torch.cat([pk[i], ck[i]]),
                    torch.cat([pv[i], cv[i]]),
                    scale,
                    True,
                )
                for r in range(c):
                    err = (per_rank[r][i] - ref).abs().max().item()
                    self.assertLess(err, 1e-5, (seed, i, r))


class TestExactness(CustomTestCase):
    """Single-rank full softmax attention == c local shards + lse_combine."""

    D_C, D_R, HEADS = 32, 8, 6

    def _run(self, c, page_size, seq_lens, seed):
        g = torch.Generator().manual_seed(seed)
        bsz = len(seq_lens)
        scale = 1.0 / math.sqrt(self.D_C + self.D_R)
        q_nope = torch.randn(bsz, self.HEADS, self.D_C, generator=g)
        q_rope = torch.randn(bsz, self.HEADS, self.D_R, generator=g)
        kv = [torch.randn(n, self.D_C, generator=g) for n in seq_lens]
        kr = [torch.randn(n, self.D_R, generator=g) for n in seq_lens]

        ref_out, ref_lse = [], []
        for i in range(bsz):
            scores = (q_nope[i] @ kv[i].T + q_rope[i] @ kr[i].T) * scale
            ref_lse.append(torch.logsumexp(scores, dim=-1))
            ref_out.append(torch.softmax(scores, dim=-1) @ kv[i])
        ref_out, ref_lse = torch.stack(ref_out), torch.stack(ref_lse)

        req_to_token, num_pages = _build_layout(seq_lens, page_size, c, seed)
        block_table = dcp_block_tables(req_to_token, max(seq_lens), page_size, c)
        outs, lses = [], []
        saw_empty = False
        for r in range(c):
            c_kv = torch.zeros(num_pages * page_size, 1, self.D_C)
            k_rope = torch.zeros(num_pages * page_size, 1, self.D_R)
            for i, n in enumerate(seq_lens):
                v = req_to_token[i, :n].long()
                mine = v % c == r
                c_kv[(v // c)[mine], 0] = kv[i][mine]
                k_rope[(v // c)[mine], 0] = kr[i][mine]
            local = dcp_local_seq_lens(torch.tensor(seq_lens), c, r)
            saw_empty |= bool((local == 0).any())
            out, lse = mla_decode_with_lse_torch(
                q_nope,
                q_rope,
                c_kv.view(num_pages, page_size, 1, self.D_C),
                k_rope.view(num_pages, page_size, 1, self.D_R),
                block_table,
                local,
                scale,
            )
            self.assertTrue(torch.all(out[local == 0] == 0))
            self.assertTrue(torch.all(torch.isneginf(lse[local == 0])))
            outs.append(out)
            lses.append(lse)
        merged = lse_combine(torch.stack(outs), torch.stack(lses))
        merged_lse = torch.logsumexp(torch.stack(lses), dim=0)
        return merged, merged_lse, ref_out, ref_lse, saw_empty, outs, lses

    def test_exact_merge(self):
        cases = [
            (2, 1, [1, 2, 7, 30]),
            (4, 4, [1, 3, 16, 50, 97]),
            (8, 16, [1, 5, 8, 129, 300]),
            (3, 2, [2, 11, 12]),
        ]
        for seed, (c, page_size, seq_lens) in enumerate(cases):
            merged, merged_lse, ref_out, ref_lse, saw_empty, _, _ = self._run(
                c, page_size, seq_lens, seed
            )
            if min(seq_lens) < c:
                self.assertTrue(saw_empty)
            self.assertLess((merged - ref_out).abs().max().item(), 1e-5)
            self.assertLess((merged_lse - ref_lse).abs().max().item(), 1e-4)

    def test_exact_merge_a2a_vllm(self):
        # Full heads split into c groups of h; rank r keeps head group r.
        cases = [(2, 1, [1, 2, 7, 30]), (3, 2, [2, 11, 12]), (6, 4, [1, 5, 40])]
        for seed, (c, page_size, seq_lens) in enumerate(cases):
            h = self.HEADS // c
            _, _, ref_out, _, saw_empty, outs, lses = self._run(
                c, page_size, seq_lens, seed
            )
            self.assertTrue(saw_empty)
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                got = _run_ranks(
                    c, lambda r, grp: dcp_merge_a2a_vllm(outs[r], lses[r], grp)
                )
            for r in range(c):
                self.assertEqual(got[r].shape, (len(seq_lens), h, self.D_C))
                expect = ref_out[:, r * h : (r + 1) * h]
                self.assertLess((got[r] - expect).abs().max().item(), 1e-5)

    def test_exact_merge_a2a_npu(self):
        # Packed bf16/fp32 exchange + npu_attention_update, incl. empty ranks.
        cases = [(2, 1, [1, 2, 7, 30]), (3, 2, [2, 11, 12]), (6, 4, [1, 5, 40])]
        for seed, (c, page_size, seq_lens) in enumerate(cases):
            h = self.HEADS // c
            _, _, ref_out, _, saw_empty, outs, lses = self._run(
                c, page_size, seq_lens, seed
            )
            self.assertTrue(saw_empty)
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                got = _run_ranks(
                    c, lambda r, grp: dcp_merge_a2a_npu(outs[r], lses[r], grp)
                )
            for r in range(c):
                self.assertEqual(got[r].shape, (len(seq_lens), h, self.D_C))
                expect = ref_out[:, r * h : (r + 1) * h]
                self.assertLess((got[r] - expect).abs().max().item(), 1e-5)

    def test_base2_lse(self):
        outs = torch.randn(4, 3, 5, 7)
        lses = torch.randn(4, 3, 5)
        lses[1, 0, 0] = -math.inf
        lses[2, 1, 1] = float("nan")
        base_e = lse_combine(outs, lses, base_e=True)
        base_2 = lse_combine(outs, lses / math.log(2.0), base_e=False)
        self.assertLess((base_e - base_2).abs().max().item(), 1e-5)

    def test_all_empty_is_zero(self):
        outs = torch.full((3, 2, 4, 5), float("nan"))
        lses = torch.full((3, 2, 4), -math.inf)
        self.assertTrue(torch.all(lse_combine(outs, lses) == 0))


class TestMerge(CustomTestCase):
    def _inputs(self, n, bsz, h, d, dtype, seed=0):
        g = torch.Generator().manual_seed(seed)
        outs = [torch.randn(bsz, n * h, d, generator=g).to(dtype) for _ in range(n)]
        lses = [torch.randn(bsz, n * h, generator=g) * 3 for _ in range(n)]
        # An empty shard for one (request, head) on rank 0.
        lses[0][0, 0] = -math.inf
        return outs, lses

    def _expected(self, outs, lses, rank, h):
        o = torch.stack(outs)[:, :, rank * h : (rank + 1) * h]
        l = torch.stack(lses)[:, :, rank * h : (rank + 1) * h]
        return lse_combine(o, l)

    def test_a2a_matches_combine(self):
        n, bsz, h, d = 4, 3, 2, 16
        for dtype in (torch.bfloat16, torch.float32):
            outs, lses = self._inputs(n, bsz, h, d, dtype)
            got = _run_ranks(n, lambda r, grp: dcp_merge_a2a(outs[r], lses[r], grp))
            for r in range(n):
                self.assertEqual(got[r].shape, (bsz, h, d))
                self.assertEqual(got[r].dtype, dtype)
                self.assertTrue(
                    torch.equal(got[r], self._expected(outs, lses, r, h)), dtype
                )

    def test_attention_update_reference_matches_combine(self):
        g = torch.Generator().manual_seed(5)
        n, t, d = 4, 13, 9
        outs = torch.randn(n, t, d, generator=g)
        lses = torch.randn(n, t, generator=g) * 4
        ref, ref_lse = attention_update_reference(list(lses), list(outs), 1)
        expect = lse_combine(outs.view(n, t, 1, d), lses.view(n, t, 1)).view(t, d)
        self.assertLess((ref - expect).abs().max().item(), 1e-5)
        self.assertLess(
            (ref_lse - torch.logsumexp(lses, dim=0)).abs().max().item(), 1e-4
        )
        self.assertIsNone(attention_update_reference(list(lses), list(outs), 0)[1])

    def test_attention_update_sanitises_invalid_lse(self):
        g = torch.Generator().manual_seed(6)
        n, t, d = 3, 8, 5
        outs = torch.randn(n, t, d, generator=g)
        lses = torch.randn(n, t, generator=g)
        lses[0, 0] = math.inf  # FIA sentinel for an empty local KV
        outs[0, 0] = float("nan")
        lses[1, 1] = -math.inf
        lses[2, 2] = float("nan")
        lses[:, 3] = math.inf  # no valid shard -> 0
        outs[:, 3] = float("nan")
        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            got = npu_attention_update(list(lses), list(outs))
        expect = lse_combine(outs.view(n, t, 1, d), lses.view(n, t, 1)).view(t, d)
        self.assertTrue(torch.all(torch.isfinite(got)))
        self.assertTrue(torch.all(got[3] == 0))
        self.assertLess((got - expect).abs().max().item(), 1e-5)

    def test_a2a_vllm_matches_combine(self):
        n, bsz, h, d = 4, 3, 2, 16
        for dtype in (torch.bfloat16, torch.float32):
            outs, lses = self._inputs(n, bsz, h, d, dtype)
            # rank 2 has no local KV for request 1: FIA's +inf sentinel.
            lses[2][1] = math.inf
            outs[2][1] = float("nan")
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                got = _run_ranks(
                    n, lambda r, grp: dcp_merge_a2a_vllm(outs[r], lses[r], grp)
                )
            tol = 1e-5 if dtype == torch.float32 else 2e-2
            for r in range(n):
                self.assertEqual(got[r].shape, (bsz, h, d))
                self.assertEqual(got[r].dtype, dtype)
                expect = self._expected(outs, lses, r, h)
                diff = (got[r].float() - expect.float()).abs().max().item()
                self.assertLess(diff, tol, dtype)

    def test_a2a_npu_matches_combine(self):
        n, bsz, h, d = 4, 3, 2, 16
        for dtype in (torch.bfloat16, torch.float32):
            outs, lses = self._inputs(n, bsz, h, d, dtype)
            # rank 2 has no local KV for request 1: FIA's +inf sentinel.
            lses[2][1] = math.inf
            outs[2][1] = float("nan")
            with patch.object(
                dcp_ops, "_attention_update_op", attention_update_reference
            ):
                got = _run_ranks(
                    n, lambda r, grp: dcp_merge_a2a_npu(outs[r], lses[r], grp)
                )
                vllm = _run_ranks(
                    n, lambda r, grp: dcp_merge_a2a_vllm(outs[r], lses[r], grp)
                )
            for r in range(n):
                self.assertEqual(got[r].shape, (bsz, h, d))
                self.assertEqual(got[r].dtype, dtype)
                self.assertTrue(torch.all(torch.isfinite(got[r])))
                expect = self._expected(outs, lses, r, h)
                # Same inputs (bf16 out is exact in fp32), so bit-equal to vllm.
                self.assertTrue(torch.equal(got[r], vllm[r]), dtype)
                tol = 1e-5 if dtype == torch.float32 else 2e-2
                diff = (got[r].float() - expect.float()).abs().max().item()
                self.assertLess(diff, tol, dtype)

    def test_a2a_npu_base2(self):
        n, bsz, h, d = 2, 2, 3, 4
        outs, lses = self._inputs(n, bsz, h, d, torch.float32, seed=3)
        lses2 = [l / math.log(2.0) for l in lses]
        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            e = _run_ranks(n, lambda r, grp: dcp_merge_a2a_npu(outs[r], lses[r], grp))
            b2 = _run_ranks(
                n,
                lambda r, grp: dcp_merge_a2a_npu(outs[r], lses2[r], grp, base_e=False),
            )
        for r in range(n):
            self.assertLess((e[r] - b2[r]).abs().max().item(), 1e-5)

    def test_a2a_vllm_base2(self):
        n, bsz, h, d = 2, 2, 3, 4
        outs, lses = self._inputs(n, bsz, h, d, torch.float32, seed=3)
        lses2 = [l / math.log(2.0) for l in lses]
        with patch.object(dcp_ops, "_attention_update_op", attention_update_reference):
            e = _run_ranks(n, lambda r, grp: dcp_merge_a2a_vllm(outs[r], lses[r], grp))
            b2 = _run_ranks(
                n,
                lambda r, grp: dcp_merge_a2a_vllm(outs[r], lses2[r], grp, base_e=False),
            )
        for r in range(n):
            self.assertLess((e[r] - b2[r]).abs().max().item(), 1e-5)

    def test_bf16_lse_pack_roundtrip(self):
        lse = torch.tensor([0.123456789, -1e30, 3.4e38, -math.inf, 7.0])
        buf = torch.empty(5, 2, dtype=torch.bfloat16)
        buf.view(torch.float32)[:, 0] = lse
        self.assertTrue(torch.equal(buf.clone().view(torch.float32)[:, 0], lse))

    def test_ag_rs_matches_a2a(self):
        n, bsz, h, d = 4, 3, 3, 8
        outs, lses = self._inputs(n, bsz, h, d, torch.float32, seed=1)
        a2a = _run_ranks(n, lambda r, grp: dcp_merge_a2a(outs[r], lses[r], grp))
        ag_rs = _run_ranks(n, lambda r, grp: dcp_merge_ag_rs(outs[r], lses[r], grp))
        for r in range(n):
            self.assertEqual(ag_rs[r].shape, (bsz, h, d))
            self.assertLess((a2a[r] - ag_rs[r]).abs().max().item(), 1e-5)

    def test_ag_rs_base2(self):
        n, bsz, h, d = 2, 2, 2, 4
        outs, lses = self._inputs(n, bsz, h, d, torch.float32, seed=2)
        lses2 = [l / math.log(2.0) for l in lses]
        e = _run_ranks(n, lambda r, grp: dcp_merge_ag_rs(outs[r], lses[r], grp))
        b2 = _run_ranks(
            n, lambda r, grp: dcp_merge_ag_rs(outs[r], lses2[r], grp, base_e=False)
        )
        for r in range(n):
            self.assertLess((e[r] - b2[r]).abs().max().item(), 1e-5)


def _import_ascend_backend():
    mocked = {
        name: MagicMock()
        for name in (
            "torch_npu",
            "torch_npu.contrib",
            "sgl_kernel_npu",
            "sgl_kernel_npu.attention",
            "sgl_kernel_npu.attention.sinks_attention",
            "sglang.srt.speculative",
            "sglang.srt.speculative.decoupled_spec_io",
            "sglang.srt.speculative.spec_info",
            "sglang.srt.speculative.eagle_info",
        )
        if name not in sys.modules
    }
    with patch.dict(sys.modules, mocked):
        from sglang.srt.hardware_backend.npu.attention import ascend_backend
    return ascend_backend


def _fake_fia_bsnd(query, key, value, *, num_heads, num_key_value_heads, **kw):
    """BSND FIA stand-in: (out [1, T, H, Dv], lse [1, H, T, 1])."""
    assert kw["softmax_lse_flag"] and kw["input_layout"] == "BSND"
    assert num_heads == num_key_value_heads == query.shape[2]
    if kw["sparse_mode"] == 0:
        assert kw["atten_mask"] is None
        causal = False
    else:
        assert kw["sparse_mode"] == 3 and kw["atten_mask"] is not None
        causal = True
    t, s = query.shape[1], key.shape[1]
    scores = torch.einsum("thd,shd->hts", query[0], key[0]) * kw["scale"]
    if causal:
        allowed = torch.arange(s)[None, :] <= torch.arange(t)[:, None] + (s - t)
        scores = scores.masked_fill(~allowed, -math.inf)
    lse = torch.logsumexp(scores, dim=-1)  # [H, T]
    out = torch.einsum("hts,shd->thd", torch.softmax(scores, dim=-1), value[0])
    return out[None], lse[None, :, :, None]


_REAL_TORCH_TENSOR = torch.tensor


def _cpu_tensor(*args, **kwargs):
    kwargs.pop("device", None)
    return _REAL_TORCH_TENSOR(*args, **kwargs)


def _real_ascend_backend(
    *, dcp_size, allocator_page_size, is_draft_worker=False, page=4, heads=8
):
    """Run the real AscendAttnBackend.__init__ on CPU with NPU deps mocked."""
    backend_mod = _import_ascend_backend()
    req_to_token = torch.arange(2 * 64, dtype=torch.int32).view(2, 64) + 8
    model_runner = SimpleNamespace(
        device="cpu",
        page_size=page,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            attention_arch=backend_mod.AttentionArch.MLA,
            kv_lora_rank=6,
            qk_rope_head_dim=2,
            qk_nope_head_dim=6,
            hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]),
            context_len=64,
            num_attention_heads=heads,
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=object(),
        spec_algorithm=SimpleNamespace(
            is_dspark=lambda: False,
            get_num_tokens_per_req_for_target_verify=lambda n, is_draft_worker: n,
        ),
        is_draft_worker=is_draft_worker,
        is_hybrid_swa=False,
        server_args=None,
        ps=SimpleNamespace(attn_cp_size=1),
    )
    if allocator_page_size is not None:
        model_runner.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=allocator_page_size
        )
    parallel = SimpleNamespace(attn_tp_size=1, attn_dcp_size=dcp_size, attn_dcp_rank=0)
    with patch.object(torch, "tensor", _cpu_tensor), patch.object(
        backend_mod, "get_parallel", return_value=parallel
    ), patch.object(
        backend_mod,
        "get_spec",
        return_value=SimpleNamespace(speculative_num_draft_tokens=None),
    ), patch.object(
        backend_mod,
        "get_flags",
        return_value=SimpleNamespace(
            capture=SimpleNamespace(enable_torch_compile=False)
        ),
    ), patch.object(
        backend_mod, "AscendAttnMaskBuilder", MagicMock()
    ), patch.object(
        backend_mod, "AscendTorchNativeAttnBackend", MagicMock()
    ), patch.object(
        backend_mod, "DllmConfig", MagicMock(from_server_args=lambda _: None)
    ), patch.object(
        backend_mod, "is_fia_nz", return_value=False
    ):
        backend = backend_mod.AscendAttnBackend(model_runner)
    return backend_mod, backend, model_runner


class TestDcpTargetBackendInit(CustomTestCase):
    """Real AscendAttnBackend with DCP on: __init__ refuses an allocator that
    was not widened to page_size * dcp_size."""

    PAGE = 4

    def test_requires_widened_allocator(self):
        dcp = 4
        _, backend, _ = _real_ascend_backend(
            dcp_size=dcp, allocator_page_size=self.PAGE * dcp, page=self.PAGE
        )
        self.assertEqual((backend.dcp_size, backend.page_size), (dcp, self.PAGE))
        for bad in (self.PAGE, self.PAGE * 2, None):
            with self.assertRaisesRegex(RuntimeError, r"page_size \* dcp_size"):
                _real_ascend_backend(
                    dcp_size=dcp, allocator_page_size=bad, page=self.PAGE
                )
        # DCP off: no allocator requirement.
        _, plain, _ = _real_ascend_backend(dcp_size=1, allocator_page_size=None)
        self.assertEqual(plain.dcp_size, 1)

    def test_eager_decode_block_table_width_from_host_lens(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        dcp = 2
        backend_mod, backend, mr = _real_ascend_backend(
            dcp_size=dcp, allocator_page_size=self.PAGE * dcp, page=self.PAGE
        )
        seq_lens_cpu = [5, 19]
        widths = []
        real_block_tables = backend_mod.dcp_block_tables

        def spy(rows, max_len, page_size, dcp_size):
            widths.append(max_len)
            return real_block_tables(rows, max_len, page_size, dcp_size)

        for spec_info, extra in ((None, 0), (SimpleNamespace(), 2)):
            backend.speculative_step_id = extra - 1
            fb = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                batch_size=2,
                # Stale device lengths: using them (a device-scalar slice bound,
                # i.e. a host sync) would change the width.
                seq_lens=torch.tensor([5, 3]),
                seq_lens_cpu=torch.tensor(seq_lens_cpu),
                spec_info=spec_info,
                spec_algorithm=None,
                req_pool_indices=torch.tensor([1, 0]),
                extend_seq_lens=None,
                extend_seq_lens_cpu=[1, 1],
                out_cache_loc=None,
            )
            widths.clear()
            with patch.object(backend_mod, "dcp_block_tables", spy), patch.object(
                torch, "tensor", _cpu_tensor
            ):
                backend.init_forward_metadata(fb)
            max_len = max(seq_lens_cpu) + extra
            self.assertEqual(widths, [max_len])
            self.assertIsInstance(widths[0], int)
            stride = self.PAGE * dcp
            expect = (
                mr.req_to_token_pool.req_to_token[[1, 0], 0:max_len:stride] // stride
            )
            self.assertTrue(
                torch.equal(backend.forward_metadata.block_tables, expect.int())
            )


class TestNpuMlaPoolDcpHelpers(CustomTestCase):
    """NPUMLATokenToKVPool CPU backup / restore translates virtual locs under
    DCP; KV relocation refuses to run."""

    C, P, D_C, D_R, LAYERS = 2, 2, 3, 2, 2

    def _pool(self, pages):
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMLATokenToKVPool,
        )

        pool = object.__new__(NPUMLATokenToKVPool)
        pool.layer_num = self.LAYERS
        pool.start_layer = 0
        pool.index_head_dim = None
        pool.kv_lora_rank = self.D_C
        pool.qk_rope_head_dim = self.D_R
        pool.cpu_offloading_chunk_size = 3
        shape = (self.LAYERS, pages + 1, self.P, 1)
        pool.k_buffer = torch.zeros(*shape, self.D_C)
        pool.v_buffer = torch.zeros(*shape, self.D_R)
        return pool

    def test_cpu_copy_roundtrip_and_move_guard(self):
        from sglang.srt.hardware_backend.npu import memory_pool_npu

        c, seq_len, pages = self.C, 7, 4
        stride = self.P * c
        # Request on virtual page 1, restored onto virtual page 3.
        old_virtual = torch.arange(seq_len) + 1 * stride
        new_virtual = torch.arange(seq_len) + 3 * stride
        fake_npu = SimpleNamespace(synchronize=lambda: None)
        for rank in range(c):
            parallel = SimpleNamespace(
                dcp_enabled=True, attn_dcp_size=c, attn_dcp_rank=rank
            )
            pool = self._pool(pages)
            owned = old_virtual % c == rank
            k_rows = torch.randn(self.LAYERS, seq_len, self.D_C)
            v_rows = torch.randn(self.LAYERS, seq_len, self.D_R)
            for layer in range(self.LAYERS):
                phys = old_virtual[owned] // c
                pool.k_buffer[layer].view(-1, self.D_C)[phys] = k_rows[layer][owned]
                pool.v_buffer[layer].view(-1, self.D_R)[phys] = v_rows[layer][owned]
            with patch.object(
                memory_pool_npu, "get_parallel", return_value=parallel
            ), patch.object(torch, "npu", fake_npu, create=True):
                backup = pool.get_cpu_copy(old_virtual)
                pool.k_buffer.zero_()
                pool.v_buffer.zero_()
                pool.load_cpu_copy(backup, new_virtual)
                with self.assertRaisesRegex(NotImplementedError, "decode context"):
                    pool.move_kv_cache(new_virtual, old_virtual)
            for layer in range(self.LAYERS):
                k_view = pool.k_buffer[layer].view(-1, self.D_C)
                v_view = pool.v_buffer[layer].view(-1, self.D_R)
                phys = new_virtual[owned] // c
                self.assertTrue(torch.equal(k_view[phys], k_rows[layer][owned]))
                self.assertTrue(torch.equal(v_view[phys], v_rows[layer][owned]))
                # Only this rank's physical rows (and pad slot 0) are written.
                touched = torch.zeros(k_view.shape[0], dtype=torch.bool)
                touched[phys] = True
                touched[0] = True
                self.assertTrue(torch.all(k_view[~touched] == 0))


class TestAscendBackendChunkedPrefix(CustomTestCase):
    """AscendAttnBackend._forward_extend_mla_prefix_dcp on c simulated ranks
    (fake BSND FIA, reference npu_attention_update) == full causal attention."""

    HEADS, NOPE, ROPE, V, RANK = 2, 6, 4, 5, 8

    def test_matches_full_attention(self):
        backend_mod = _import_ascend_backend()
        c, page_size, chunk_tokens = 2, 4, 11
        prefix_lens, extend_lens = [29, 0, 16], [4, 3, 1]
        g = torch.Generator().manual_seed(7)
        h, dk = self.HEADS, self.NOPE + self.ROPE
        kv_b = torch.randn(h * (self.NOPE + self.V), self.RANK, generator=g) * 0.3
        latent = [torch.randn(n, self.RANK, generator=g) for n in prefix_lens]
        rope = [torch.randn(n, self.ROPE, generator=g) for n in prefix_lens]
        total_t = sum(extend_lens)
        q = torch.randn(total_t, h, dk, generator=g)
        k = torch.randn(total_t, h, dk, generator=g)
        v = torch.randn(total_t, h, self.V, generator=g)
        scale = 1.0 / math.sqrt(dk)
        req_to_token, num_pages = _build_layout(prefix_lens, page_size, c, seed=7)
        stride = page_size * c
        vals = [torch.cat([latent[i], rope[i]], -1) for i in range(len(prefix_lens))]
        block_ids = torch.cat(
            [req_to_token[i, :n:stride] // stride for i, n in enumerate(prefix_lens)]
        ).to(torch.int32)

        def project(x):
            out = x.reshape(-1, self.RANK) @ kv_b.T
            return out.view(*x.shape[:-1], kv_b.shape[0])

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=h,
            tp_k_head_num=h,
            v_head_dim=self.V,
            scaling=scale,
            kv_b_proj=lambda x: (project(x),),
        )
        local = threading.local()
        collectives = []

        def rank_fn(r, group):
            local.group = group
            pool = _owned_pool(
                prefix_lens, req_to_token, num_pages, page_size, c, r, vals
            )
            backend = object.__new__(backend_mod.AscendAttnBackend)
            backend.dcp_size = c
            backend.page_size = page_size
            backend.dcp_prefix_chunk_tokens = chunk_tokens
            backend.qk_nope_head_dim = self.NOPE
            backend.fia_mask = torch.ones(1, dtype=torch.bool)
            backend.forward_metadata = SimpleNamespace(
                extend_seq_lens_cpu_int=torch.tensor(extend_lens, dtype=torch.int32),
                prefix_lens=torch.tensor(prefix_lens),
                prefix_npages=[(n + stride - 1) // stride for n in prefix_lens],
                flatten_prefix_block_tables=block_ids,
            )
            buffers = pool.unsqueeze(2).split([self.RANK, self.ROPE], dim=-1)
            backend.token_to_kv_pool = SimpleNamespace(
                get_key_buffer=lambda _: buffers[0].contiguous(),
                get_value_buffer=lambda _: buffers[1].contiguous(),
            )
            out = backend._forward_extend_mla_prefix_dcp(q, k, v, layer)
            collectives.append((r, group._call))
            return out

        fake_ops = SimpleNamespace(
            npu=SimpleNamespace(npu_fused_infer_attention_score=_fake_fia_bsnd)
        )
        with patch.object(
            dcp_ops, "_attention_update_op", attention_update_reference
        ), patch.object(backend_mod, "is_fia_nz", return_value=False), patch.object(
            backend_mod,
            "get_parallel",
            side_effect=lambda: SimpleNamespace(dcp_group=local.group),
        ), patch.object(
            backend_mod, "torch", SimpleNamespace(**{**vars(torch), "ops": fake_ops})
        ):
            per_rank = _run_ranks(c, rank_fn)

        expect_calls = sum(math.ceil(n / chunk_tokens) for n in prefix_lens)
        self.assertEqual(sorted(collectives), [(r, expect_calls) for r in range(c)])
        offset = 0
        refs = []
        for i, n in enumerate(prefix_lens):
            t = extend_lens[i]
            kv = project(latent[i]).view(n, h, self.NOPE + self.V)
            k_nope, v_pre = kv.split([self.NOPE, self.V], dim=-1)
            k_pre = torch.cat([k_nope, rope[i][:, None].expand(-1, h, -1)], -1)
            out, _ = _fake_fia_bsnd(
                q[None, offset : offset + t],
                torch.cat([k_pre, k[offset : offset + t]])[None],
                torch.cat([v_pre, v[offset : offset + t]])[None],
                num_heads=h,
                num_key_value_heads=h,
                softmax_lse_flag=True,
                input_layout="BSND",
                sparse_mode=3,
                atten_mask=True,
                scale=scale,
            )
            refs.append(out[0])
            offset += t
        ref = torch.cat(refs).reshape(total_t, h * self.V)
        for r in range(c):
            self.assertEqual(per_rank[r].shape, (total_t, h * self.V))
            self.assertLess((per_rank[r] - ref).abs().max().item(), 1e-5)


class TestEnvDefaults(CustomTestCase):
    def test_defaults(self):
        self.assertEqual(envs.SGLANG_NPU_DCP_MERGE_IMPL.get(), "npu")
        self.assertTrue(envs.SGLANG_NPU_DCP_PAD_HEADS.get())
        self.assertEqual(envs.SGLANG_NPU_DCP_PREFIX_CHUNK_TOKENS.get(), 65536)

    def test_merge_impl_selects_path(self):
        mocked = {
            name: MagicMock()
            for name in (
                "torch_npu",
                "sgl_kernel_npu",
                "sgl_kernel_npu.norm",
                "sgl_kernel_npu.norm.fused_split_qk_norm",
            )
            if name not in sys.modules
        }
        with patch.dict(sys.modules, mocked):
            from sglang.srt.hardware_backend.npu.modules import (
                deepseek_v2_attention_mla_npu as mla_npu,
            )

        calls = []

        def fake(name):
            return lambda out, lse, group, base_e=True: calls.append(name)

        parallel = SimpleNamespace(dcp_comm_backend="a2a", dcp_group=None)
        with patch.object(mla_npu, "get_parallel", return_value=parallel), patch.object(
            mla_npu, "dcp_merge_a2a_npu", fake("npu")
        ), patch.object(
            mla_npu, "dcp_merge_a2a_vllm", fake("vllm")
        ), patch.object(mla_npu, "dcp_merge_a2a", fake("torch")), patch.object(
            mla_npu, "dcp_merge_ag_rs", fake("ag_rs")
        ):
            out, lse = torch.zeros(1, 2, 3), torch.zeros(1, 2)
            mla_npu._npu_dcp_merge_mla_decode(out, lse)
            with envs.SGLANG_NPU_DCP_MERGE_IMPL.override("vllm"):
                mla_npu._npu_dcp_merge_mla_decode(out, lse)
            with envs.SGLANG_NPU_DCP_MERGE_IMPL.override("torch"):
                mla_npu._npu_dcp_merge_mla_decode(out, lse)
            with envs.SGLANG_NPU_DCP_MERGE_IMPL.override("bogus"):
                with self.assertRaises(ValueError):
                    mla_npu._npu_dcp_merge_mla_decode(out, lse)
            parallel.dcp_comm_backend = "ag_rs"
            mla_npu._npu_dcp_merge_mla_decode(out, lse)
        self.assertEqual(calls, ["npu", "vllm", "torch", "ag_rs"])


class TestKimiK3NpuDcpConfig(CustomTestCase):
    @staticmethod
    def _server_args(**kwargs):
        fields = dict(
            dcp_size=8,
            dcp_comm_backend="ag_rs",
            dcp_replicate_q_proj=None,
            speculative_algorithm=None,
            enable_hierarchical_cache=False,
            disaggregation_mode="null",
            attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
        )
        fields.update(kwargs)
        return SimpleNamespace(**fields)

    def _resolve(self, **kwargs):
        with patch.object(
            kimi_k3_overrides,
            "get_platform",
            return_value=SimpleNamespace(is_npu=True, is_sm100=False),
        ):
            return kimi_k3_overrides._kimi_k3_overrides(
                self._server_args(**kwargs), None
            )

    def test_default_replicates_q_with_a2a(self):
        self.assertEqual(
            self._resolve(),
            {"dcp_replicate_q_proj": True, "dcp_comm_backend": "a2a"},
        )
        self.assertEqual(
            self._resolve(attention_backend="ascend"),
            {"dcp_replicate_q_proj": True, "dcp_comm_backend": "a2a"},
        )

    def test_no_replicate_keeps_comm_backend(self):
        self.assertEqual(self._resolve(dcp_replicate_q_proj=False), {})
        self.assertEqual(
            self._resolve(dcp_replicate_q_proj=False, dcp_comm_backend="a2a"), {}
        )

    def test_explicit_replicate(self):
        self.assertEqual(
            self._resolve(dcp_replicate_q_proj=True), {"dcp_comm_backend": "a2a"}
        )

    def test_rejections(self):
        rejected = [
            dict(speculative_algorithm="DSPARK"),
            dict(enable_hierarchical_cache=True),
            dict(disaggregation_mode="decode"),
            dict(dcp_comm_backend="fi_a2a"),
            dict(decode_attention_backend="cutedsl_mla"),
        ]
        for kwargs in rejected:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                self._resolve(**kwargs)
        with envs.SGLANG_NPU_USE_MLAPO.override(True):
            with self.assertRaises(ValueError):
                self._resolve()

    def test_dcp_disabled_is_untouched(self):
        self.assertEqual(self._resolve(dcp_size=1), {})


if __name__ == "__main__":
    unittest.main()
