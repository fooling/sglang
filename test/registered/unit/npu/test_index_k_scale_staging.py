"""Page-major staging for the NPU Indexer FP32 scale cache.

The device Indexer scale cache is layer-major, so one page's scales are one
512 B block per Indexer layer; the host mirror is page-first, where the same
page's scales are already a single contiguous run.  The staging mirror gives
the transfer a device-side view with the host's shape, collapsing those blocks
into one per page.  These tests drive the real meta construction through a
reference model of ``offload.kv_exchange_copy`` and check that the staging path
is byte-equivalent to the per-(layer, page) path it replaces.
"""

import contextlib
import enum
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.pool_host import mla as mla_mod
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 128
LAYER_NUM = 12
# GLM-5.2 shape in miniature: one Indexer layer every 4 transformer layers.
INDEXER_LAYER_IDS = (0, 4, 8)
NUM_INDEXER_LAYERS = len(INDEXER_LAYER_IDS)
DEVICE_PAGES = 16

# offload.kv_exchange_copy meta: 6 int64 header, then 9 int64 per component.
META_HEADER = 6
META_STRIDE = 9


class _Direction(enum.Enum):
    """Stand-in for sgl_kernel_npu.kvcacheio.TransferDirection."""

    H2D = 1
    D2H = 2


class _FakeKvExchange:
    """Reference model of the acc_offload kv_exchange_copy kernel.

    Reads the meta exactly as documented -- for every component, every page and
    every layer in ``[lo, hi)``, copy ``width`` bytes between
    ``base + page * page_stride + layer * layer_stride`` on both sides -- and
    records one entry per block so tests can count DMA blocks.
    """

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

    def scale_blocks(self, width):
        return [block for block in self.blocks if block[1] == width]


@contextlib.contextmanager
def _installed(fake):
    """Install the reference kernel and the NPU-only guards the path needs."""
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
        ), mock.patch.object(
            # load_to_device_per_layer's rank-0 debug log; no process group here.
            torch.distributed,
            "get_rank",
            lambda *args, **kwargs: 0,
        ):
            yield
    finally:
        if created_npu:
            del torch.npu


def _make_pools():
    device_pool = NPUMLATokenToKVPool(
        size=DEVICE_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=64,
        qk_rope_head_dim=16,
        layer_num=LAYER_NUM,
        device="cpu",
        enable_memory_saver=False,
        index_head_dim=128,
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


def _page_indices(pages):
    """Token-slot index list for whole pages, as the transfer paths pass it."""
    return torch.cat(
        [torch.arange(p * PAGE_SIZE, (p + 1) * PAGE_SIZE, dtype=torch.int64) for p in pages]
    )


def _fill_host_scale(host_pool):
    scale = host_pool.index_k_scale_buffer
    values = torch.arange(scale.numel(), dtype=torch.float32) + 1.0
    scale.copy_(values.reshape(scale.shape))


def _fill_device_scale(device_pool):
    scale = device_pool.index_k_scale_buffer
    values = torch.arange(scale.numel(), dtype=torch.float32) + 1000.0
    scale.copy_(values.reshape(scale.shape))


def _tensors(device_pool, host_pool):
    return [
        device_pool.k_buffer,
        device_pool.v_buffer,
        device_pool.index_k_buffer,
        device_pool.index_k_scale_buffer,
        device_pool.index_k_scale_staging,
        host_pool.k_buffer,
        host_pool.v_buffer,
        host_pool.index_k_buffer,
        host_pool.index_k_scale_buffer,
    ]


class TestIndexKScaleStaging(CustomTestCase):
    def setUp(self):
        # Deliberately non-identity and different on the two sides, so a
        # component that mixes up the host and device page axes cannot pass.
        self.host_pages = [5, 2, 9]
        self.device_pages = [11, 3, 7]
        self.host_indices = _page_indices(self.host_pages)
        self.device_indices = _page_indices(self.device_pages)
        self.num_pages = len(self.host_pages)
        self.scale_width = NUM_INDEXER_LAYERS * PAGE_SIZE * 4
        self.legacy_width = PAGE_SIZE * 4

    def _run(self, device_pool, host_pool, direction, *, staging, **kwargs):
        saved = device_pool.index_k_scale_staging
        if not staging:
            device_pool.index_k_scale_staging = None
        fake = _FakeKvExchange(
            _tensors(device_pool, host_pool), self.host_indices, self.device_indices
        )
        try:
            with _installed(fake):
                host_pool._transfer_ascendc_sparse_copy(
                    device_pool,
                    self.host_indices,
                    self.device_indices,
                    direction,
                    **kwargs,
                )
        finally:
            device_pool.index_k_scale_staging = saved
        return fake

    def test_staging_layout_mirrors_the_host_scale_buffer(self):
        device_pool, host_pool = _make_pools()
        staging = device_pool.index_k_scale_staging
        host_scale = host_pool.index_k_scale_buffer

        self.assertEqual(
            staging.shape, (1, DEVICE_PAGES + 1, NUM_INDEXER_LAYERS, PAGE_SIZE)
        )
        # One page is one contiguous run of every Indexer layer on both sides,
        # and the two runs have the same pitch -- that is what lets the layer
        # axis collapse to a single block.
        self.assertEqual(staging.stride(1) * 4, self.scale_width)
        self.assertEqual(host_scale.stride(0) * 4, self.scale_width)
        self.assertEqual(staging.stride(1) * 4, host_scale.stride(0) * 4)
        # The op-facing cache keeps its layer-major layout untouched.
        self.assertEqual(
            device_pool.index_k_scale_buffer.shape,
            (NUM_INDEXER_LAYERS, DEVICE_PAGES + 1, PAGE_SIZE, 1),
        )
        self.assertEqual(
            device_pool.index_k_scale_buffer.stride(2) * 4 * PAGE_SIZE,
            self.legacy_width,
        )

    def test_h2d_matches_the_per_layer_transfer(self):
        legacy_pool, legacy_host = _make_pools()
        _fill_host_scale(legacy_host)
        self._run(legacy_pool, legacy_host, _Direction.H2D, staging=False)

        staged_pool, staged_host = _make_pools()
        _fill_host_scale(staged_host)
        fake = self._run(staged_pool, staged_host, _Direction.H2D, staging=True)

        torch.testing.assert_close(
            staged_pool.index_k_scale_buffer,
            legacy_pool.index_k_scale_buffer,
            rtol=0,
            atol=0,
        )
        # And the transfer actually happened -- an all-zero pass would match too.
        self.assertGreater(float(staged_pool.index_k_scale_buffer.abs().sum()), 0.0)
        self.assertEqual(len(fake.scale_blocks(self.scale_width)), self.num_pages)

    def test_h2d_block_count_drops_to_one_per_page(self):
        legacy_pool, legacy_host = _make_pools()
        _fill_host_scale(legacy_host)
        legacy = self._run(legacy_pool, legacy_host, _Direction.H2D, staging=False)

        staged_pool, staged_host = _make_pools()
        _fill_host_scale(staged_host)
        staged = self._run(staged_pool, staged_host, _Direction.H2D, staging=True)

        self.assertEqual(
            len(legacy.scale_blocks(self.legacy_width)),
            NUM_INDEXER_LAYERS * self.num_pages,
        )
        self.assertEqual(len(staged.scale_blocks(self.scale_width)), self.num_pages)
        self.assertEqual(
            len(legacy.blocks) - len(staged.blocks),
            (NUM_INDEXER_LAYERS - 1) * self.num_pages,
        )
        # Same bytes moved, fewer blocks.
        self.assertEqual(
            sum(width for _, width in legacy.scale_blocks(self.legacy_width)),
            sum(width for _, width in staged.scale_blocks(self.scale_width)),
        )

    def test_h2d_leaves_untouched_pages_alone(self):
        device_pool, host_pool = _make_pools()
        _fill_host_scale(host_pool)
        _fill_device_scale(device_pool)
        before = device_pool.index_k_scale_buffer.clone()

        self._run(device_pool, host_pool, _Direction.H2D, staging=True)

        after = device_pool.index_k_scale_buffer
        untouched = [p for p in range(DEVICE_PAGES + 1) if p not in self.device_pages]
        torch.testing.assert_close(
            after[:, untouched], before[:, untouched], rtol=0, atol=0
        )
        for page in self.device_pages:
            self.assertFalse(torch.equal(after[:, page], before[:, page]))

    def test_d2h_matches_the_per_layer_transfer(self):
        legacy_pool, legacy_host = _make_pools()
        _fill_device_scale(legacy_pool)
        self._run(legacy_pool, legacy_host, _Direction.D2H, staging=False)

        staged_pool, staged_host = _make_pools()
        _fill_device_scale(staged_pool)
        fake = self._run(staged_pool, staged_host, _Direction.D2H, staging=True)

        torch.testing.assert_close(
            staged_host.index_k_scale_buffer,
            legacy_host.index_k_scale_buffer,
            rtol=0,
            atol=0,
        )
        self.assertGreater(float(staged_host.index_k_scale_buffer.abs().sum()), 0.0)
        self.assertEqual(len(fake.scale_blocks(self.scale_width)), self.num_pages)

    def test_round_trip_through_staging_is_lossless(self):
        device_pool, host_pool = _make_pools()
        _fill_device_scale(device_pool)
        original = device_pool.index_k_scale_buffer.clone()

        self._run(device_pool, host_pool, _Direction.D2H, staging=True)
        device_pool.index_k_scale_buffer.zero_()
        self._run(device_pool, host_pool, _Direction.H2D, staging=True)

        for page in self.device_pages:
            torch.testing.assert_close(
                device_pool.index_k_scale_buffer[:, page],
                original[:, page],
                rtol=0,
                atol=0,
            )

    def test_scale_rides_only_the_first_group(self):
        device_pool, host_pool = _make_pools()
        _fill_host_scale(host_pool)
        fake = _FakeKvExchange(
            _tensors(device_pool, host_pool), self.host_indices, self.device_indices
        )
        with _installed(fake), mock.patch.dict(
            "os.environ", {"SGLANG_HICACHE_LAYER_GROUP_SIZE": "1"}
        ):
            for layer_id in range(LAYER_NUM):
                host_pool.load_to_device_per_layer(
                    device_pool,
                    self.host_indices,
                    self.device_indices,
                    layer_id,
                    "kernel_ascend",
                )

        # One layer per group means LAYER_NUM launches, but the scale is not
        # sliced per group: it rides the first one and is never resent.
        self.assertEqual(len(fake.scale_blocks(self.scale_width)), self.num_pages)
        self.assertEqual(len(fake.scale_blocks(self.legacy_width)), 0)
        legacy_pool, legacy_host = _make_pools()
        _fill_host_scale(legacy_host)
        self._run(legacy_pool, legacy_host, _Direction.H2D, staging=False)
        torch.testing.assert_close(
            device_pool.index_k_scale_buffer,
            legacy_pool.index_k_scale_buffer,
            rtol=0,
            atol=0,
        )

    def test_falls_back_when_layer_counts_disagree(self):
        """A pool whose Indexer layer count differs from the host mirror's.

        The staging block spans all Indexer layers of a page, so its width is
        only correct when both sides agree; the per-layer component is not, and
        is what a mismatched pool (e.g. an MTP draft pool) must keep using.
        """
        device_pool, host_pool = _make_pools()
        _fill_host_scale(host_pool)
        # Same page geometry, one Indexer layer fewer than the host mirror.
        narrow = device_pool.index_k_scale_staging[:, :, :-1, :].contiguous()
        saved = device_pool.index_k_scale_staging
        device_pool.index_k_scale_staging = narrow
        try:
            fake = _FakeKvExchange(
                _tensors(device_pool, host_pool), self.host_indices, self.device_indices
            )
            with _installed(fake):
                host_pool._transfer_ascendc_sparse_copy(
                    device_pool,
                    self.host_indices,
                    self.device_indices,
                    _Direction.H2D,
                )
        finally:
            device_pool.index_k_scale_staging = saved

        self.assertEqual(len(fake.scale_blocks(self.scale_width)), 0)
        self.assertEqual(
            len(fake.scale_blocks(self.legacy_width)),
            NUM_INDEXER_LAYERS * self.num_pages,
        )
        legacy_pool, legacy_host = _make_pools()
        _fill_host_scale(legacy_host)
        self._run(legacy_pool, legacy_host, _Direction.H2D, staging=False)
        torch.testing.assert_close(
            device_pool.index_k_scale_buffer,
            legacy_pool.index_k_scale_buffer,
            rtol=0,
            atol=0,
        )

    def test_falls_back_when_the_device_pool_has_no_staging(self):
        device_pool, host_pool = _make_pools()
        _fill_host_scale(host_pool)
        fake = self._run(device_pool, host_pool, _Direction.H2D, staging=False)

        self.assertEqual(
            len(fake.scale_blocks(self.legacy_width)),
            NUM_INDEXER_LAYERS * self.num_pages,
        )
        self.assertGreater(float(device_pool.index_k_scale_buffer.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
