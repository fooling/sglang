"""Unit tests for sglang.srt.distributed.parallel_state — no server, no model loading."""

import unittest

import sglang.srt.platforms as platforms_mod
from sglang.srt.distributed.parallel_state import get_default_distributed_backend
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


# Worked example: an out-of-tree platform that overrides the torch backend.
# Future plugin authors can pattern-match against this in their own tests.
class _OverridingPlatform(SRTPlatform):
    device_type = "cuda"

    def get_torch_distributed_backend_str(self) -> str:
        return "fake_backend"


# Mirrors all in-tree platforms today: no override, so DeviceMixin's default
# body returns _DEVICE_TO_DISTRIBUTED_BACKEND.get(device_type, "gloo").
class _DefaultPlatform(SRTPlatform):
    device_type = "cuda"


class TestGetDefaultDistributedBackend(CustomTestCase):
    """Cover all paths of get_default_distributed_backend.

    When ``device == current_platform.device_type`` the dispatcher asks
    ``current_platform.get_torch_distributed_backend_str()``; otherwise it
    looks up ``_DEVICE_TO_DISTRIBUTED_BACKEND[device]`` for cross-device
    queries (e.g. auxiliary "cpu"/gloo groups on a CUDA process).

    Tests use real SRTPlatform subclasses so the assertions stay close to
    how an actual out-of-tree plugin would interact with this dispatcher.

    ``current_platform`` is exposed via ``__getattr__`` on the platforms
    module backed by the ``_current_platform`` singleton, so the test
    overrides that singleton directly and restores it in tearDown.
    """

    def setUp(self):
        self._saved_platform = platforms_mod._current_platform

    def tearDown(self):
        platforms_mod._current_platform = self._saved_platform

    def _install(self, platform: SRTPlatform) -> None:
        platforms_mod._current_platform = platform

    def test_overriding_platform_supplies_backend(self):
        self._install(_OverridingPlatform())
        self.assertEqual(get_default_distributed_backend("cuda"), "fake_backend")

    def test_overriding_platform_skipped_for_non_active_device(self):
        # Even when current_platform overrides get_torch_distributed_backend_str,
        # callers asking for a different device (e.g. an auxiliary "cpu" gloo
        # group on a CUDA process) must keep going through the dict.
        self._install(_OverridingPlatform())
        self.assertEqual(get_default_distributed_backend("cpu"), "gloo")

    def test_default_platform_uses_device_mixin_table(self):
        # No override: DeviceMixin's default body looks up the device_type in
        # _DEVICE_TO_DISTRIBUTED_BACKEND, so cuda still resolves to nccl.
        self._install(_DefaultPlatform())
        self.assertEqual(get_default_distributed_backend("cuda"), "nccl")

    def test_unknown_device_returns_gloo_default(self):
        self._install(_DefaultPlatform())
        self.assertEqual(get_default_distributed_backend("unobtanium"), "gloo")



class TestDcpCollectiveBackend(CustomTestCase):
    """The DCP group does not inherit the zbal backend.

    SGLANG_ZBAL_LOCAL_MEM_SIZE repoints _DEVICE_TO_DISTRIBUTED_BACKEND["npu"]
    to "zbal", and every model-parallel group is built from the world group's
    backend, so the switch reaches groups that have nothing to do with the
    allocator or the DeepEP buffer it exists for. The DCP attention reduction
    does an all_to_all_single per layer, so its group keeps hccl.
    """

    def test_zbal_is_not_inherited(self):
        from sglang.srt.distributed.parallel_state import _dcp_collective_backend

        self.assertEqual(_dcp_collective_backend("zbal"), "hccl")

    def test_every_other_backend_is_left_alone(self):
        from sglang.srt.distributed.parallel_state import _dcp_collective_backend

        for backend in ("hccl", "nccl", "gloo", "mooncake", "xccl"):
            self.assertEqual(_dcp_collective_backend(backend), backend, backend)


if __name__ == "__main__":
    unittest.main()
