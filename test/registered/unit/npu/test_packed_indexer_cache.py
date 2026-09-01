"""Packed DSA Indexer cache: FP8 K and its FP32 scale share a page.

The split layout gives the HiCache transfer two components per Indexer layer --
page_size * index_head_dim bytes of K and page_size * 4 bytes of scale -- which
is two DMA blocks per (layer, page).  Packing them into one page makes it one
block.  These tests check that the packed page holds exactly what the two split
buffers held, that the transfer carries one component instead of two, and that
every path that cannot read the packed layout fails loudly rather than quietly
reading the wrong bytes.
"""

import contextlib
import enum
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.layers.attention.dsa.packed_indexer import (
    native_packed_indexer_op,
    quant_lightning_indexer_packed,
    unpack_index_k_with_scale,
)
from sglang.srt.mem_cache.pool_host import mla as mla_mod
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 8
LAYER_NUM = 6
INDEXER_LAYER_IDS = (0, 2, 4)
NUM_INDEXER_LAYERS = len(INDEXER_LAYER_IDS)
HEAD_DIM = 128
DEVICE_PAGES = 16
SCALE_BYTES = 4

META_HEADER = 6
META_STRIDE = 9


class _Direction(enum.Enum):
    H2D = 1
    D2H = 2


class _FakeKvExchange:
    """Reference model of kv_exchange_copy; records one entry per DMA block."""

    def __init__(self, tensors, host_indices, device_indices):
        self._flat_by_ptr = {}
        for tensor in tensors:
            if tensor is None or tensor.numel() == 0:
                continue
            self._flat_by_ptr[tensor.data_ptr()] = tensor.view(torch.uint8).reshape(-1)
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.blocks = []

    def kv_exchange_copy(self, meta, device):
        vals = [int(v) for v in meta.tolist()]
        num_components, num_pages, page_size, direction = vals[:4]
        for comp in range(num_components):
            base = META_HEADER + META_STRIDE * comp
            (
                dev_ptr,
                host_ptr,
                dev_layer_stride,
                dev_page_stride,
                host_page_stride,
                host_layer_stride,
                width,
                lo,
                hi,
            ) = vals[base : base + META_STRIDE]
            dev = self._flat_by_ptr[dev_ptr]
            host = self._flat_by_ptr[host_ptr]
            for page in range(num_pages):
                dev_page = int(self.device_indices[page * page_size]) // page_size
                host_page = int(self.host_indices[page * page_size]) // page_size
                for layer in range(lo, hi):
                    d = dev_page * dev_page_stride + layer * dev_layer_stride
                    h = host_page * host_page_stride + layer * host_layer_stride
                    if direction == _Direction.H2D.value:
                        dev[d : d + width] = host[h : h + width]
                    else:
                        host[h : h + width] = dev[d : d + width]
                    self.blocks.append((comp, width))
        return 0

    def widths(self):
        return sorted({width for _, width in self.blocks})


@contextlib.contextmanager
def _installed(fake):
    module = types.ModuleType("memfabric_hybrid")
    module.offload = SimpleNamespace(kv_exchange_copy=fake.kv_exchange_copy)
    npu_ns = getattr(torch, "npu", None)
    created_npu = npu_ns is None
    if created_npu:
        torch.npu = SimpleNamespace(synchronize=lambda: None)
    try:
        with mock.patch.dict(sys.modules, {"memfabric_hybrid": module}), mock.patch.object(
            mla_mod, "TransferDirection", _Direction, create=True
        ), mock.patch.object(mla_mod, "_is_npu", True), mock.patch.object(
            mla_mod, "ascendc_io_enabled", lambda: True
        ), mock.patch.object(
            torch.npu, "synchronize", lambda: None, create=True
        ):
            yield
    finally:
        if created_npu:
            del torch.npu


def _make_pools(packed: bool):
    with envs.SGLANG_NPU_ENABLE_PACKED_INDEXER_CACHE.override(packed):
        device_pool = NPUMLATokenToKVPool(
            size=DEVICE_PAGES * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=64,
            qk_rope_head_dim=16,
            layer_num=LAYER_NUM,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=HEAD_DIM,
            start_layer=0,
            end_layer=LAYER_NUM,
            indexer_layer_ids=INDEXER_LAYER_IDS,
            enable_npu_quant_lightning_indexer=True,
        )
    host_pool = MLATokenToKVPoolHost(
        device_pool,
        host_to_device_ratio=1.0,
        host_size=0,
        page_size=PAGE_SIZE,
        layout="page_first_kv_split",
        pin_memory=False,
        device="cpu",
    )
    return device_pool, host_pool


@contextlib.contextmanager
def _torch_scatter_nd_update():
    """torch stand-in for npu_scatter_nd_update_, which needs a card (or a
    forwarder that accepts uint8) to run.  Same semantics for the 1-D index
    form the split writers use: rows named by indices[:, 0] take updates."""
    import torch_npu

    def _impl(target, indices, updates):
        target[indices.reshape(-1)] = updates.reshape(
            (-1,) + tuple(target.shape[1:])
        ).to(target.dtype)
        return target

    with mock.patch.object(
        torch_npu, "npu_scatter_nd_update_", _impl, create=True
    ):
        yield


def _sample_writes(num_tokens=5):
    loc = torch.tensor([0, 3, 9, 17, 40][:num_tokens], dtype=torch.int64)
    key = (
        torch.arange(num_tokens * HEAD_DIM, dtype=torch.float32).reshape(
            num_tokens, HEAD_DIM
        )
        % 61
        - 30
    ).to(torch.float8_e4m3fn)
    scale = torch.arange(1, num_tokens + 1, dtype=torch.float32) * 0.25
    return loc, key, scale


def _tensors(device_pool, host_pool):
    return [
        device_pool.k_buffer,
        device_pool.v_buffer,
        device_pool.index_k_buffer,
        device_pool.index_k_scale_buffer,
        device_pool.index_k_with_scale_buffer,
        host_pool.k_buffer,
        host_pool.v_buffer,
        host_pool.index_k_buffer,
        host_pool.index_k_scale_buffer,
        host_pool.index_k_with_scale_buffer,
    ]


def _page_indices(pages):
    return torch.cat(
        [
            torch.arange(p * PAGE_SIZE, (p + 1) * PAGE_SIZE, dtype=torch.int64)
            for p in pages
        ]
    )


class TestPackedIndexerCache(CustomTestCase):
    def setUp(self):
        self.host_pages = [5, 2, 9]
        self.device_pages = [11, 3, 7]
        self.host_indices = _page_indices(self.host_pages)
        self.device_indices = _page_indices(self.device_pages)
        self.num_pages = len(self.host_pages)
        self.packed_width = PAGE_SIZE * (HEAD_DIM + SCALE_BYTES)
        self.k_width = PAGE_SIZE * HEAD_DIM
        self.scale_width = PAGE_SIZE * SCALE_BYTES

    def test_packed_page_is_k_then_scale(self):
        pool, _ = _make_pools(packed=True)
        self.assertTrue(pool.packed_indexer_cache)
        self.assertIsNone(pool.index_k_buffer)
        self.assertIsNone(pool.index_k_scale_buffer)
        self.assertEqual(
            pool.index_k_with_scale_buffer.shape,
            (NUM_INDEXER_LAYERS, DEVICE_PAGES + 1, PAGE_SIZE, 1, HEAD_DIM + SCALE_BYTES),
        )
        # One page is one contiguous run, which is what collapses the two
        # transfer components into one.
        page_bytes = pool.index_k_with_scale_buffer[0].stride(0)
        self.assertEqual(page_bytes, self.packed_width)

    def test_packed_write_matches_the_split_writes(self):
        loc, key, scale = _sample_writes()
        split_pool, _ = _make_pools(packed=False)
        with _torch_scatter_nd_update():
            split_pool.set_index_k_buffer(2, loc, key)
            split_pool.set_index_k_scale_buffer(2, loc, scale)

        packed_pool, _ = _make_pools(packed=True)
        packed_pool.set_index_k_with_scale_buffer(2, loc, key, scale)

        got_key, got_scale = unpack_index_k_with_scale(
            packed_pool.get_index_k_with_scale_buffer(2), HEAD_DIM
        )
        torch.testing.assert_close(
            got_key.float(), split_pool.get_index_k_buffer(2).float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            got_scale,
            split_pool.get_index_k_scale_buffer(2).reshape(got_scale.shape),
            rtol=0,
            atol=0,
        )
        # And something was actually written.
        self.assertGreater(float(got_scale.abs().sum()), 0.0)

    def test_transfer_carries_one_component_instead_of_two(self):
        split_pool, split_host = _make_pools(packed=False)
        packed_pool, packed_host = _make_pools(packed=True)
        for host in (split_host, packed_host):
            for buf in (
                host.index_k_buffer,
                host.index_k_scale_buffer,
                host.index_k_with_scale_buffer,
            ):
                if buf is not None:
                    flat = buf.view(torch.uint8).reshape(-1)
                    flat.copy_(
                        (torch.arange(flat.numel(), dtype=torch.int64) % 251).to(
                            torch.uint8
                        )
                    )

        split = self._run(split_pool, split_host, _Direction.H2D)
        packed = self._run(packed_pool, packed_host, _Direction.H2D)

        self.assertIn(self.k_width, split.widths())
        self.assertIn(self.scale_width, split.widths())
        self.assertIn(self.packed_width, packed.widths())
        self.assertNotIn(self.scale_width, packed.widths())
        self.assertEqual(
            len([b for b in split.blocks if b[1] in (self.k_width, self.scale_width)]),
            2 * NUM_INDEXER_LAYERS * self.num_pages,
        )
        self.assertEqual(
            len([b for b in packed.blocks if b[1] == self.packed_width]),
            NUM_INDEXER_LAYERS * self.num_pages,
        )

    def test_transferred_packed_pages_hold_the_split_bytes(self):
        split_pool, split_host = _make_pools(packed=False)
        packed_pool, packed_host = _make_pools(packed=True)
        # Same logical content on both host mirrors: page-major K then scale.
        for layer in range(NUM_INDEXER_LAYERS):
            for page in range(split_host.index_k_buffer.shape[0]):
                k = torch.arange(PAGE_SIZE * HEAD_DIM, dtype=torch.int64)
                k = ((k + layer * 7 + page * 13) % 251).to(torch.uint8)
                s = ((torch.arange(PAGE_SIZE * SCALE_BYTES) + page) % 251).to(
                    torch.uint8
                )
                split_host.index_k_buffer[page, layer].view(torch.uint8).reshape(
                    -1
                ).copy_(k)
                split_host.index_k_scale_buffer[page, layer].view(
                    torch.uint8
                ).reshape(-1).copy_(s)
                packed_host.index_k_with_scale_buffer[page, layer].reshape(-1).copy_(
                    torch.cat([k, s])
                )

        self._run(split_pool, split_host, _Direction.H2D)
        self._run(packed_pool, packed_host, _Direction.H2D)

        for layer_id in INDEXER_LAYER_IDS:
            key, scale = unpack_index_k_with_scale(
                packed_pool.get_index_k_with_scale_buffer(layer_id), HEAD_DIM
            )
            torch.testing.assert_close(
                key.view(torch.uint8),
                split_pool.get_index_k_buffer(layer_id).view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                scale.reshape(-1),
                split_pool.get_index_k_scale_buffer(layer_id).reshape(-1),
                rtol=0,
                atol=0,
            )

    def test_disagg_region_count_halves(self):
        split_pool, _ = _make_pools(packed=False)
        packed_pool, _ = _make_pools(packed=True)

        self.assertEqual(
            len(split_pool.get_state_buf_infos()[0]), 2 * NUM_INDEXER_LAYERS
        )
        self.assertEqual(len(packed_pool.get_state_buf_infos()[0]), NUM_INDEXER_LAYERS)
        self.assertEqual(
            packed_pool.get_state_layer_ids(), list(INDEXER_LAYER_IDS)
        )
        self.assertEqual(
            len(split_pool.get_contiguous_buf_infos()[0])
            - len(packed_pool.get_contiguous_buf_infos()[0]),
            NUM_INDEXER_LAYERS,
        )
        self.assertEqual(
            len(packed_pool.get_kv_layer_ids()),
            len(packed_pool.get_contiguous_buf_infos()[0]),
        )

    def test_split_accessors_refuse_the_packed_cache(self):
        pool, _ = _make_pools(packed=True)
        with self.assertRaisesRegex(RuntimeError, "get_index_k_with_scale_buffer"):
            pool.get_index_k_buffer(2)
        with self.assertRaisesRegex(RuntimeError, "get_index_k_with_scale_buffer"):
            pool.get_index_k_scale_buffer(2)

    def test_paths_that_cannot_read_the_packed_page_fail_loudly(self):
        pool, host = _make_pools(packed=True)
        with self.assertRaisesRegex(NotImplementedError, "L3 zero-copy"):
            host.get_page_buffer_meta(torch.arange(PAGE_SIZE, dtype=torch.int64))
        fake = _FakeKvExchange(
            _tensors(pool, host), self.host_indices, self.device_indices
        )
        with _installed(fake), mock.patch.object(
            mla_mod, "ascendc_io_enabled", lambda: False
        ), mock.patch.object(
            torch.distributed, "get_rank", lambda *a, **k: 0
        ), self.assertRaisesRegex(
            RuntimeError, "AscendC kv_exchange"
        ):
            host.load_to_device_per_layer(
                pool, self.host_indices, self.device_indices, 0, "kernel_ascend"
            )

    def test_dispatch_prefers_a_registered_packed_op(self):
        pool, _ = _make_pools(packed=True)
        loc, key, scale = _sample_writes()
        pool.set_index_k_with_scale_buffer(2, loc, key, scale)
        packed = pool.get_index_k_with_scale_buffer(2)
        sentinel = torch.zeros(1)
        calls = []

        # 8 heads is below tl.dot's minimum extent, so the Triton kernel
        # declines this shape and the dispatch has to reach the fallback.
        args = dict(
            query=torch.zeros(2, 8, HEAD_DIM),
            key_with_scale=packed,
            weights=torch.zeros(2, 8),
            query_dequant_scale=torch.zeros(2, 1),
            actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([PAGE_SIZE], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            index_head_dim=HEAD_DIM,
            sparse_count=4,
        )

        def split_op(**kwargs):
            calls.append(kwargs)
            return sentinel

        # A shim may already register the op in this environment, so the
        # no-native cases name an empty namespace rather than assuming one.
        empty = SimpleNamespace()
        with mock.patch.object(torch.ops, "npu", empty, create=True):
            self.assertIsNone(native_packed_indexer_op())
            # No native op and no fallback: say so instead of guessing.
            with self.assertRaisesRegex(RuntimeError, "No packed-cache Indexer op"):
                quant_lightning_indexer_packed(**args)
            # Fallback unpacks and hands the vendor op two contiguous tensors.
            self.assertIs(
                quant_lightning_indexer_packed(**args, split_op=split_op), sentinel
            )
        self.assertEqual(calls[0]["key"].shape[-1], HEAD_DIM)
        self.assertTrue(calls[0]["key"].is_contiguous())
        self.assertTrue(calls[0]["key_dequant_scale"].is_contiguous())

        # A registered native op wins over the fallback.
        native = SimpleNamespace(
            **{"npu_quant_lightning_indexer_packed": lambda *a: "native"}
        )
        with mock.patch.object(torch.ops, "npu", native, create=True):
            self.assertIsNotNone(native_packed_indexer_op())
            self.assertEqual(
                quant_lightning_indexer_packed(**args, split_op=split_op), "native"
            )

    def _run(self, device_pool, host_pool, direction):
        fake = _FakeKvExchange(
            _tensors(device_pool, host_pool), self.host_indices, self.device_indices
        )
        with _installed(fake):
            host_pool._transfer_ascendc_sparse_copy(
                device_pool, self.host_indices, self.device_indices, direction
            )
        return fake


if __name__ == "__main__":
    unittest.main()
