"""Triton Indexer logits read the packed page in place.

The packed Indexer cache exists so a HiCache page moves as one DMA block per
layer, which costs it the contiguity the vendor op needs.  This kernel is what
makes the layout usable without paying that back as an unpack copy, so what has
to hold is that reading the packed page gives the same answer as reading the two
split buffers.  These tests run the kernel under Triton's interpreter, so they
need no accelerator and check semantics rather than tiling.
"""

import os
import unittest

# Must precede the Triton import: the interpreter is selected when the kernel is
# decorated, not when it is launched.
os.environ.setdefault("TRITON_INTERPRET", "1")

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

try:
    from sglang.kernels.ops.attention.dsa.packed_indexer_npu import (
        INDEX_K_SCALE_BYTES,
        packed_indexer_supported,
        packed_indexer_topk,
    )

    _HAS_TRITON = True
except Exception:  # pragma: no cover - exercised only where Triton is absent
    _HAS_TRITON = False

PAGE_SIZE = 16
HEAD_DIM = 32
NUM_HEADS = 16
PAGE_NUM = 6
SPARSE_COUNT = 5


def _build_cache(generator):
    """The same content in both layouts: (packed, key_fp32, scale_fp32)."""
    key = torch.randint(
        -60,
        60,
        (PAGE_NUM, PAGE_SIZE, HEAD_DIM),
        generator=generator,
        dtype=torch.int32,
    ).to(torch.float8_e4m3fn)
    scale = (
        torch.randint(
            1, 9, (PAGE_NUM, PAGE_SIZE), generator=generator, dtype=torch.int32
        ).to(torch.float32)
        * 0.125
    )
    row = HEAD_DIM + INDEX_K_SCALE_BYTES
    packed = torch.empty((PAGE_NUM, PAGE_SIZE * row), dtype=torch.uint8)
    k_bytes = PAGE_SIZE * HEAD_DIM
    packed[:, :k_bytes] = key.reshape(PAGE_NUM, k_bytes).view(torch.uint8)
    packed[:, k_bytes:] = (
        scale.reshape(PAGE_NUM, PAGE_SIZE, 1)
        .contiguous()
        .view(torch.uint8)
        .reshape(PAGE_NUM, PAGE_SIZE * INDEX_K_SCALE_BYTES)
    )
    return packed, key.to(torch.float32), scale


def _split_layout_topk(key, scale, query, weights, query_scale, block_table, seq_q, seq_k):
    """Golden: gather from the split buffers and score with a separate expression."""
    num_tokens = sum(seq_q)
    want = torch.full((num_tokens, SPARSE_COUNT), -1, dtype=torch.int32)
    start = 0
    for request, q_len in enumerate(seq_q):
        kv_len = seq_k[request]
        pages = block_table[request].tolist()
        k_rows = torch.cat([key[p] for p in pages])[:kv_len]
        s_rows = torch.cat([scale[p] for p in pages])[:kv_len]
        for local in range(q_len):
            token = start + local
            position = kv_len - q_len + local
            row = torch.full((kv_len,), float("-inf"))
            for pos in range(position + 1):
                acc = 0.0
                for head in range(NUM_HEADS):
                    dot = float(torch.dot(query[token, head], k_rows[pos]))
                    acc += (
                        max(dot, 0.0)
                        * float(weights[token, head])
                        * float(query_scale[token, 0])
                    )
                row[pos] = acc * float(s_rows[pos])
            take = min(SPARSE_COUNT, kv_len)
            want[token, :take] = row.topk(take).indices.to(torch.int32)
        start += q_len
    return want


@unittest.skipUnless(_HAS_TRITON, "Triton is not installed")
class TestPackedIndexerTriton(CustomTestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20260901)
        self.packed, self.key, self.scale = _build_cache(generator)
        # Ragged on purpose: a kv length that is not a whole number of pages, and
        # a request whose query run is shorter than another's.
        self.seq_q = [3, 2]
        self.seq_k = [35, 20]
        self.block_table = torch.tensor([[4, 1, 5, 0], [2, 3, 0, 0]], dtype=torch.int32)
        num_tokens = sum(self.seq_q)
        self.query = torch.randint(
            -60,
            60,
            (num_tokens, NUM_HEADS, HEAD_DIM),
            generator=generator,
            dtype=torch.int32,
        ).to(torch.float32)
        self.weights = torch.rand((num_tokens, NUM_HEADS), generator=generator)
        self.query_scale = torch.rand((num_tokens, 1), generator=generator) + 0.5

    def _run(self, **overrides):
        kwargs = dict(
            cache=self.packed,
            query=self.query,
            weights=self.weights,
            query_dequant_scale=self.query_scale,
            block_table=self.block_table,
            cumulative_seq_lengths_query=torch.tensor([3, 5], dtype=torch.int32),
            seq_lengths_key=torch.tensor(self.seq_k, dtype=torch.int32),
            index_head_dim=HEAD_DIM,
            page_size=PAGE_SIZE,
            sparse_count=SPARSE_COUNT,
        )
        kwargs.update(overrides)
        return packed_indexer_topk(**kwargs)

    def test_matches_the_split_layout(self):
        got = self._run()
        want = _split_layout_topk(
            self.key,
            self.scale,
            self.query,
            self.weights,
            self.query_scale,
            self.block_table,
            self.seq_q,
            self.seq_k,
        )
        self.assertEqual(int((got != want).sum()), 0)
        # A run that selected nothing would also compare equal to an empty want.
        self.assertTrue(bool((got >= 0).all()))

    def test_indices_stay_inside_the_kv_run(self):
        got = self._run()
        start = 0
        for request, q_len in enumerate(self.seq_q):
            kv_len = self.seq_k[request]
            selected = got[start : start + q_len]
            self.assertTrue(bool((selected < kv_len).all()))
            start += q_len

    def test_causal_mask_holds_for_every_token(self):
        got = self._run()
        start = 0
        for request, q_len in enumerate(self.seq_q):
            kv_len = self.seq_k[request]
            for local in range(q_len):
                position = kv_len - q_len + local
                self.assertTrue(bool((got[start + local] <= position).all()))
            start += q_len

    def test_short_run_pads_with_minus_one(self):
        got = self._run(
            seq_lengths_key=torch.tensor([3, 20], dtype=torch.int32),
            block_table=torch.tensor(
                [[4, 0, 0, 0], [2, 3, 0, 0]], dtype=torch.int32
            ),
        )
        # kv_len 3 < sparse_count 5: the tail is unfilled, not a repeated index.
        self.assertTrue(bool((got[:3, 3:] == -1).all()))
        self.assertTrue(bool((got[:3, :3] >= 0).all()))

    def test_unsupported_shapes_are_refused_not_miscomputed(self):
        self.assertTrue(packed_indexer_supported(NUM_HEADS, HEAD_DIM, PAGE_SIZE))
        # tl.dot needs power-of-two extents of at least 16.
        self.assertFalse(packed_indexer_supported(8, HEAD_DIM, PAGE_SIZE))
        self.assertFalse(packed_indexer_supported(NUM_HEADS, HEAD_DIM, 24))
        with self.assertRaisesRegex(ValueError, "power-of-two extents"):
            self._run(query=self.query[:, :8])


if __name__ == "__main__":
    unittest.main()
