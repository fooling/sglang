"""Packed DSA Indexer cache as a HiCache transfer format.

The split layout gives the HiCache transfer two components per Indexer layer --
page_size * index_head_dim bytes of K and page_size * 4 bytes of scale -- which
is two DMA blocks per (layer, page).  Packing them into one page makes it one
block, at the cost of a layout no Indexer op reads.

This variant does not ask the op to read it.  The split buffers stay and remain
the only truth; the packed page is transfer scratch that a load lands in and a
write-back is built from.  The model code is untouched -- writes go where they
always went.  These tests pin that the transfer carries one component instead of
two, that a loaded page reaches the split buffers with the bytes the split
transfer would have delivered, and that a write-back sends what the split
buffers hold.
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


def unpack_index_k_with_scale(packed, head_dim):
    """Split one layer's packed cache into contiguous K and scale tensors.

    Written here rather than imported: this variant deliberately has no unpack
    helper on the call path, so the test needs its own reading of the layout to
    check the pool's against.
    """
    page_num, page_size = packed.shape[0], packed.shape[1]
    row = head_dim + SCALE_BYTES
    flat = packed.reshape(page_num, page_size * row)
    k_bytes = page_size * head_dim
    key = (
        flat[:, :k_bytes]
        .reshape(page_num, page_size, 1, head_dim)
        .contiguous()
        .view(torch.float8_e4m3fn)
    )
    scale = (
        flat[:, k_bytes:]
        .contiguous()
        .view(torch.float32)
        .reshape(page_num, page_size, 1)
    )
    return key, scale


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
        # Both representations exist: the op still reads the split pair.
        self.assertIsNotNone(pool.index_k_buffer)
        self.assertIsNotNone(pool.index_k_scale_buffer)
        self.assertEqual(
            pool.index_k_with_scale_buffer.shape,
            (NUM_INDEXER_LAYERS, DEVICE_PAGES + 1, PAGE_SIZE, 1, HEAD_DIM + SCALE_BYTES),
        )
        # One page is one contiguous run, which is what collapses the two
        # transfer components into one.
        page_bytes = pool.index_k_with_scale_buffer[0].stride(0)
        self.assertEqual(page_bytes, self.packed_width)

    def test_writes_do_not_touch_the_scratch(self):
        """The model code is unchanged, so a write lands in the split buffers
        only; the packed page is built when a transfer needs it."""
        loc, key, scale = _sample_writes()
        pool, _ = _make_pools(packed=True)
        before = pool.index_k_with_scale_buffer.clone()
        with _torch_scatter_nd_update():
            pool.set_index_k_buffer(2, loc, key)
            pool.set_index_k_scale_buffer(2, loc, scale)

        pages, offsets = loc // PAGE_SIZE, loc % PAGE_SIZE
        torch.testing.assert_close(
            pool.get_index_k_buffer(2)[pages, offsets, 0].float(),
            key.float(),
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.equal(pool.index_k_with_scale_buffer, before))

        # pack_index_pages builds it, and then it holds what was written.
        pool.pack_index_pages(_page_indices(sorted(set(pages.tolist()))))
        packed_key, packed_scale = unpack_index_k_with_scale(
            pool.get_index_k_with_scale_buffer(2), HEAD_DIM
        )
        torch.testing.assert_close(
            packed_key[pages, offsets, 0].float(), key.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            packed_scale[pages, offsets, 0], scale, rtol=0, atol=0
        )

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

    def test_a_load_reaches_the_buffers_the_op_reads(self):
        """The point of the variant: a packed load is published into the split
        buffers, so the Indexer op sees the same bytes it would have seen from a
        split transfer without ever being told about the packed page."""
        split_pool, split_host = _make_pools(packed=False)
        packed_pool, packed_host = _make_pools(packed=True)
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

        before = packed_pool.index_k_buffer.clone()
        self._run(split_pool, split_host, _Direction.H2D)
        self._run(packed_pool, packed_host, _Direction.H2D)

        for layer_id in INDEXER_LAYER_IDS:
            torch.testing.assert_close(
                packed_pool.get_index_k_buffer(layer_id).view(torch.uint8),
                split_pool.get_index_k_buffer(layer_id).view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                packed_pool.get_index_k_scale_buffer(layer_id),
                split_pool.get_index_k_scale_buffer(layer_id),
                rtol=0,
                atol=0,
            )
        # The unpack really ran; an unpublished load would have left these alone.
        self.assertFalse(torch.equal(packed_pool.index_k_buffer, before))
        # And only the loaded pages moved.
        untouched = [p for p in range(DEVICE_PAGES + 1) if p not in self.device_pages]
        torch.testing.assert_close(
            packed_pool.index_k_buffer[:, untouched],
            before[:, untouched],
            rtol=0,
            atol=0,
        )

    def test_disagg_keeps_the_split_regions(self):
        split_pool, _ = _make_pools(packed=False)
        packed_pool, _ = _make_pools(packed=True)

        # Unchanged on purpose: a page arriving over RDMA is used directly, with
        # no unpack step to run afterwards, so disagg keeps the split pair.
        self.assertEqual(
            len(packed_pool.get_state_buf_infos()[0]), 2 * NUM_INDEXER_LAYERS
        )
        self.assertEqual(
            len(packed_pool.get_contiguous_buf_infos()[0]),
            len(split_pool.get_contiguous_buf_infos()[0]),
        )
        self.assertEqual(
            len(packed_pool.get_kv_layer_ids()),
            len(packed_pool.get_contiguous_buf_infos()[0]),
        )

    def test_packed_scratch_is_charged_to_memory(self):
        split_pool, _ = _make_pools(packed=False)
        packed_pool, _ = _make_pools(packed=True)
        page_num = DEVICE_PAGES + 1
        scratch = page_num * PAGE_SIZE * NUM_INDEXER_LAYERS * (HEAD_DIM + SCALE_BYTES)
        self.assertEqual(
            packed_pool.get_kv_size_bytes() - split_pool.get_kv_size_bytes(), scratch
        )

    def test_write_back_sends_what_the_split_buffers_hold(self):
        """D2H packs from the split buffers, so a write-back carries them even
        though nothing kept the scratch current."""
        split_pool, split_host = _make_pools(packed=False)
        packed_pool, packed_host = _make_pools(packed=True)
        for pool in (split_pool, packed_pool):
            flat = pool.index_k_buffer.view(torch.uint8).reshape(-1)
            flat.copy_((torch.arange(flat.numel(), dtype=torch.int64) % 251).to(torch.uint8))
            flat = pool.index_k_scale_buffer.reshape(-1)
            flat.copy_(torch.arange(flat.numel(), dtype=torch.float32) * 0.5)

        self._run(split_pool, split_host, _Direction.D2H)
        self._run(packed_pool, packed_host, _Direction.D2H)

        for page in self.host_pages:
            for layer in range(NUM_INDEXER_LAYERS):
                packed_page = packed_host.index_k_with_scale_buffer[page, layer]
                key, scale = (
                    split_host.index_k_buffer[page, layer],
                    split_host.index_k_scale_buffer[page, layer],
                )
                k_bytes = PAGE_SIZE * HEAD_DIM
                torch.testing.assert_close(
                    packed_page.reshape(-1)[:k_bytes],
                    key.view(torch.uint8).reshape(-1),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    packed_page.reshape(-1)[k_bytes:].contiguous().view(torch.float32),
                    scale.reshape(-1),
                    rtol=0,
                    atol=0,
                )
        self.assertGreater(
            float(packed_host.index_k_with_scale_buffer.abs().sum()), 0.0
        )

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
