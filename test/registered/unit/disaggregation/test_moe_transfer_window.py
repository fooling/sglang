import threading
import time
import unittest

from sglang.srt.disaggregation.moe_transfer_window import (
    MoeTransferWindow,
    get_moe_transfer_window,
    maybe_install_moe_transfer_window_hooks,
    moe_window_layer_group_size,
    moe_window_pacing_enabled,
    reset_moe_transfer_window_for_test,
    set_moe_transfer_window_for_test,
    slice_for_moe_window,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeEvent:
    """Models torch_npu ``NPUEvent`` semantics, not CUDA's.

    Two Ascend behaviours matter here and are reproduced faithfully:

    * an event that was never recorded is *uncreated*, and ``NPUEvent::query``
      returns **True** for it (``if (!is_created_) return true;``);
    * ``record()`` goes through torch_npu's task queue, so the ACL record lands
      after the host call returns, and ``query()`` returns False until then
      (``GetTaskQueueEnable() && !IsEventRecorded(...)``).

    ``land()`` models the queue consuming the record; ``complete()`` models the
    device reaching the event.
    """

    def __init__(self):
        self.created = False
        self.landed = False
        self.completed = False

    def record(self, stream=None):
        self.created = True
        self.landed = True  # tests that care about the queue call land() again
        self.completed = False

    def record_but_not_landed(self):
        self.created = True
        self.landed = False
        self.completed = False

    def query(self):
        if not self.created:
            return True  # Ascend: unrecorded event queries as complete
        if not self.landed:
            return False  # record still sitting in torch_npu's task queue
        return self.completed

    def complete(self):
        self.completed = True


def _fake_window(**kwargs):
    kwargs.setdefault("wait_timeout_s", 0.05)
    kwargs.setdefault("poll_interval_s", 0.0005)
    kwargs.setdefault("ring_size", 8)
    return MoeTransferWindow(event_factory=FakeEvent, **kwargs)


class _FakeDispatcher:
    """Minimal stand-in for BaseDispatcher's hook registration surface."""

    def __init__(self):
        self.post_dispatch_hooks = []
        self.pre_combine_hooks = []

    def register_post_dispatch_hook(self, hook):
        self.post_dispatch_hooks.append(hook)
        return object()

    def register_pre_combine_hook(self, hook):
        self.pre_combine_hooks.append(hook)
        return object()

    def enqueue_layer(self):
        """Host enqueues one MoE layer (device has not run it yet)."""
        for hook in self.post_dispatch_hooks:
            hook(self, object())
        for hook in self.pre_combine_hooks:
            hook(self, object())


class TestSliceForMoeWindow(CustomTestCase):
    def test_slices_cover_input_in_order(self):
        items = list(range(78))
        slices = slice_for_moe_window(items, 8)

        self.assertEqual(len(slices), 10)
        self.assertEqual([len(s) for s in slices], [8] * 9 + [6])
        self.assertEqual([x for s in slices for x in s], items)

    def test_non_positive_group_size_disables_slicing(self):
        items = list(range(5))
        self.assertEqual(slice_for_moe_window(items, 0), [items])
        self.assertEqual(slice_for_moe_window(items, -1), [items])

    def test_group_size_at_or_above_length_is_single_slice(self):
        items = list(range(5))
        self.assertEqual(slice_for_moe_window(items, 5), [items])
        self.assertEqual(slice_for_moe_window(items, 99), [items])

    def test_empty_input(self):
        self.assertEqual(slice_for_moe_window([], 8), [])


class TestDeviceTimeWindow(CustomTestCase):
    """The window must track the DEVICE phase, not the host's hook position."""

    def _slot_events(self, window, index):
        return window._ring[index][0], window._ring[index][1]

    def test_host_ahead_of_device_is_not_a_window(self):
        # Host enqueued the layer, but the device has not reached the GEMM yet:
        # neither event has completed, so we must NOT admit a transfer.
        window = _fake_window()
        window.force_waiter_for_test()
        window.open()
        window.close()

        self.assertFalse(window.device_in_gemm())

    def test_device_inside_gemm_is_a_window(self):
        window = _fake_window()
        window.force_waiter_for_test()
        window.open()
        window.close()
        start, end = self._slot_events(window, 0)

        start.complete()  # device reached the GEMM start
        self.assertTrue(window.device_in_gemm())

        end.complete()  # device finished the GEMM
        self.assertFalse(window.device_in_gemm())

    def test_open_without_close_still_counts_as_in_gemm(self):
        # Host has not enqueued combine yet, but the device is already running
        # the GEMM. That is a valid window.
        window = _fake_window()
        window.force_waiter_for_test()
        window.open()
        start, _ = self._slot_events(window, 0)
        start.complete()

        self.assertTrue(window.device_in_gemm())

    def test_ring_tracks_device_across_host_runahead(self):
        # Host enqueues 5 layers while the device is still on layer 0.
        window = _fake_window(ring_size=8)
        window.force_waiter_for_test()
        for _ in range(5):
            window.open()
            window.close()

        self.assertFalse(window.device_in_gemm())

        # Device now enters layer 2's GEMM only.
        start2, end2 = self._slot_events(window, 2)
        start2.complete()
        self.assertTrue(window.device_in_gemm())
        end2.complete()
        self.assertFalse(window.device_in_gemm())

    def test_no_events_recorded_while_nothing_waits(self):
        # Steady state: hooks must not allocate or record events.
        window = _fake_window()
        window.open()
        window.close()

        self.assertEqual(window._ring, [])
        self.assertEqual(window.stats.windows_recorded, 0)

    def test_unrecorded_event_querying_true_is_not_a_window(self):
        """Ascend returns query()==True for an event that was never recorded.

        The ring is pre-allocated with fresh events, so if device_in_gemm()
        trusted event state instead of our own bookkeeping it would report a
        window for every untouched slot.
        """
        window = _fake_window()
        window.force_waiter_for_test()
        window._ensure_ring()

        # Every slot's events are unrecorded and therefore query() == True.
        self.assertTrue(all(ev.query() for slot in window._ring for ev in slot[:2]))
        self.assertFalse(window.device_in_gemm())

    def test_record_still_queued_is_not_a_window(self):
        """torch_npu record() is async; query() is False until it lands."""
        window = _fake_window()
        window.force_waiter_for_test()
        window.open()
        start, _ = self._slot_events(window, 0)
        start.record_but_not_landed()

        self.assertFalse(window.device_in_gemm())

        start.landed = True
        start.complete()
        self.assertTrue(window.device_in_gemm())

    def test_wait_admits_when_device_enters_gemm(self):
        window = _fake_window(wait_timeout_s=5.0)
        results = []

        def _worker():
            results.append(window.wait_for_window())

        thread = threading.Thread(target=_worker)
        thread.start()
        # Let the worker register itself as a waiter before the hooks fire.
        time.sleep(0.05)
        window.open()
        window.close()
        self.assertEqual(results, [])  # device has not reached the GEMM
        self._slot_events(window, 0)[0].complete()
        thread.join(timeout=5.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [True])
        self.assertEqual(window.stats.slices_in_window, 1)

    def test_wait_fails_open_when_device_never_enters(self):
        window = _fake_window(wait_timeout_s=0.05)

        start = time.perf_counter()
        admitted = window.wait_for_window()
        elapsed = time.perf_counter() - start

        self.assertFalse(admitted)
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertLess(elapsed, 2.0)
        self.assertEqual(window.stats.slices_timed_out, 1)

    def test_wait_fails_open_without_device_events(self):
        window = MoeTransferWindow(wait_timeout_s=5.0, event_factory=None)
        window._event_factory_resolved = True  # simulate "no usable events"
        window._event_factory = None

        start = time.perf_counter()
        self.assertFalse(window.wait_for_window())
        # Must not burn the timeout when pacing is simply unavailable.
        self.assertLess(time.perf_counter() - start, 1.0)
        self.assertEqual(window.stats.slices_unpaced, 1)


class TestHookInstallation(CustomTestCase):
    def tearDown(self):
        reset_moe_transfer_window_for_test()

    def test_disabled_by_default(self):
        self.assertFalse(moe_window_pacing_enabled())
        dispatcher = _FakeDispatcher()
        self.assertFalse(maybe_install_moe_transfer_window_hooks(dispatcher))
        self.assertEqual(dispatcher.post_dispatch_hooks, [])
        self.assertEqual(dispatcher.pre_combine_hooks, [])

    def test_none_dispatcher_is_ignored(self):
        with envs.SGLANG_DISAGGREGATION_ENABLE_MOE_WINDOW_PACING.override(True):
            self.assertFalse(maybe_install_moe_transfer_window_hooks(None))

    def test_hooks_installed_once(self):
        with envs.SGLANG_DISAGGREGATION_ENABLE_MOE_WINDOW_PACING.override(True):
            set_moe_transfer_window_for_test(_fake_window())
            dispatcher = _FakeDispatcher()

            self.assertTrue(maybe_install_moe_transfer_window_hooks(dispatcher))
            # Second call on the same dispatcher must not double-register.
            self.assertTrue(maybe_install_moe_transfer_window_hooks(dispatcher))
            self.assertEqual(len(dispatcher.post_dispatch_hooks), 1)
            self.assertEqual(len(dispatcher.pre_combine_hooks), 1)

    def test_transfer_slices_land_inside_device_gemm_phases(self):
        """End-to-end: host runs ahead, device walks the layers, and every
        admitted slice is issued while the device is inside a GEMM."""
        with envs.SGLANG_DISAGGREGATION_ENABLE_MOE_WINDOW_PACING.override(True):
            window = _fake_window(wait_timeout_s=5.0, ring_size=16)
            set_moe_transfer_window_for_test(window)
            dispatcher = _FakeDispatcher()
            maybe_install_moe_transfer_window_hooks(dispatcher)

            slices = slice_for_moe_window(list(range(24)), 8)
            observed = []
            done = threading.Event()

            def _transfer_worker():
                for _ in slices:
                    admitted = window.wait_for_window(timeout_s=5.0)
                    observed.append((admitted, window.device_in_gemm()))
                done.set()

            thread = threading.Thread(target=_transfer_worker)
            thread.start()
            time.sleep(0.05)  # worker registers as waiter

            # Host enqueues 6 layers well ahead of the device.
            for _ in range(6):
                dispatcher.enqueue_layer()

            # Device now walks those layers one GEMM at a time.
            for i in range(6):
                if done.is_set():
                    break
                start, end = window._ring[i][0], window._ring[i][1]
                start.complete()
                time.sleep(0.02)  # GEMM is "running"
                end.complete()
                time.sleep(0.005)
            thread.join(timeout=5.0)

            self.assertFalse(thread.is_alive())
            self.assertEqual(len(observed), len(slices))
            for admitted, in_gemm in observed:
                self.assertTrue(admitted)
                self.assertTrue(in_gemm)

    def test_window_config_reads_env(self):
        with envs.SGLANG_DISAGGREGATION_MOE_WINDOW_LAYER_GROUP.override(4):
            self.assertEqual(moe_window_layer_group_size(), 4)
        with envs.SGLANG_DISAGGREGATION_MOE_WINDOW_WAIT_MS.override(12.5):
            reset_moe_transfer_window_for_test()
            self.assertAlmostEqual(get_moe_transfer_window().wait_timeout_s, 0.0125)


if __name__ == "__main__":
    unittest.main()
