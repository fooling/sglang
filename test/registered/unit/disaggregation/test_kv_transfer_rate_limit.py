import threading
import unittest

from sglang.srt.disaggregation.ascend.kv_transfer_flow import (
    KvTransferPacer,
    get_kv_transfer_pacer,
    reset_for_test,
    split_blocks,
)
from sglang.srt.distributed.collective_gate import collective_bracket, gate
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MIB = 1024 * 1024


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = 0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps += 1
        self.t += seconds


class FakeEvent:
    """torch_npu semantics: an event never recorded queries as complete."""

    def __init__(self):
        self.created = False
        self.done = False

    def record(self):
        self.created = True
        self.done = False

    def query(self):
        return True if not self.created else self.done


def _pacer(busy_mbps, quiet_mbps, is_busy=lambda: False):
    clock = FakeClock()
    return (
        KvTransferPacer(
            busy_bytes_per_s=busy_mbps * MIB,
            quiet_bytes_per_s=quiet_mbps * MIB,
            is_busy=is_busy,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        ),
        clock,
    )


class TestPacing(CustomTestCase):
    def test_first_send_is_free_then_paced(self):
        pacer, clock = _pacer(10, 0, is_busy=lambda: True)
        self.assertEqual(pacer.send(10 * MIB), 0.0)
        self.assertEqual(pacer.send(10 * MIB), 1.0)
        self.assertEqual(clock.sleeps, 1)

    def test_long_run_rate_is_exact(self):
        pacer, clock = _pacer(10, 0, is_busy=lambda: True)
        for _ in range(100):
            pacer.send(MIB)
        self.assertAlmostEqual(clock.t, 9.9, places=9)

    def test_one_sleep_per_send_no_polling_loop(self):
        pacer, clock = _pacer(10, 0, is_busy=lambda: True)
        for _ in range(10):
            pacer.send(MIB)
        self.assertEqual(clock.sleeps, 9)

    def test_quiet_state_is_unlimited_by_default(self):
        pacer, clock = _pacer(10, 0, is_busy=lambda: False)
        for _ in range(10):
            pacer.send(1000 * MIB)
        self.assertEqual(clock.t, 0.0)
        self.assertEqual(clock.sleeps, 0)

    def test_release_drops_a_deadline_parked_while_busy(self):
        busy = [True]
        pacer, clock = _pacer(1, 0, is_busy=lambda: busy[0])
        pacer.send(100 * MIB)  # parks a long deadline at the slow rate
        busy[0] = False
        self.assertEqual(pacer.send(100 * MIB), 0.0)
        busy[0] = True
        self.assertEqual(pacer.send(MIB), 0.0)  # stale deadline did not leak

    def test_quiet_cap_applies_when_configured(self):
        pacer, clock = _pacer(10, 100, is_busy=lambda: False)
        pacer.send(100 * MIB)
        self.assertAlmostEqual(pacer.send(1), 1.0, places=6)

    def test_oversized_send_returns_and_never_spins(self):
        pacer, clock = _pacer(10, 0, is_busy=lambda: True)
        pacer.send(MIB)
        self.assertAlmostEqual(pacer.send(10000 * MIB), 0.1, places=9)
        self.assertAlmostEqual(pacer.send(1), 1000.0, places=6)

    def test_zero_and_negative_are_free(self):
        pacer, clock = _pacer(1, 0, is_busy=lambda: True)
        self.assertEqual(pacer.send(0), 0.0)
        self.assertEqual(pacer.send(-5), 0.0)

    def test_probe_failure_falls_open_and_logs_once(self):
        def boom():
            raise RuntimeError("probe exploded")

        pacer, clock = _pacer(1, 0, is_busy=boom)
        logger = "sglang.srt.disaggregation.ascend.kv_transfer_flow"
        with self.assertLogs(logger, level="WARNING") as cm:
            pacer.send(100 * MIB)
            pacer.send(100 * MIB)
        self.assertEqual(clock.t, 0.0)  # not throttled
        self.assertEqual(len(cm.output), 1)

    def test_threads_share_the_rate_and_all_progress(self):
        pacer, clock = _pacer(10, 0, is_busy=lambda: True)
        done = [0] * 4

        def worker(i):
            for _ in range(10):
                pacer.send(MIB)
                done[i] += 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(done, [10, 10, 10, 10])
        self.assertEqual(pacer.bytes_sent, 40 * MIB)


class TestCollectiveGate(CustomTestCase):
    def tearDown(self):
        reset_for_test()

    def test_disabled_gate_is_never_busy_and_brackets_are_inert(self):
        self.assertFalse(gate.installed)
        with collective_bracket:
            pass
        self.assertFalse(gate.is_busy())

    def test_never_recorded_event_reads_quiet(self):
        # Ascend returns query()==True for an event that was never recorded,
        # so a fresh worker must read quiet rather than permanently throttled.
        gate.install()
        gate._event = FakeEvent()
        self.assertFalse(gate.is_busy())

    def test_busy_while_host_is_inside_a_collective(self):
        gate.install()
        gate._event = FakeEvent()
        with collective_bracket:
            self.assertTrue(gate.is_busy())  # covered by the counter

    def test_stays_busy_until_the_device_finishes(self):
        gate.install()
        event = FakeEvent()
        gate._event = event
        with collective_bracket:
            pass
        # Host has left, but the device has not reached the end event yet.
        self.assertTrue(gate.is_busy())
        event.done = True
        self.assertFalse(gate.is_busy())

    def test_nested_collectives_stay_busy_until_all_exit(self):
        gate.install()
        gate._event = FakeEvent()
        with collective_bracket:
            with collective_bracket:
                self.assertTrue(gate.is_busy())
            self.assertTrue(gate.is_busy())

    def test_query_failure_falls_open(self):
        class Boom(FakeEvent):
            def query(self):
                raise RuntimeError("query exploded")

        gate.install()
        gate._event = Boom()
        gate._event.created = True
        self.assertFalse(gate.is_busy())


class TestSplitBlocks(CustomTestCase):
    def test_bytes_and_order_are_preserved(self):
        blocks = [(i, i + 1000, 5 * MIB) for i in range(10)]
        slices = list(split_blocks(blocks, 16 * MIB))
        flat = [b for s in slices for b in s]
        self.assertEqual(flat, blocks)
        self.assertTrue(all(sum(b[2] for b in s) <= 16 * MIB for s in slices))

    def test_block_larger_than_the_slice_gets_its_own(self):
        blocks = [(0, 0, 100 * MIB), (1, 1, MIB)]
        slices = list(split_blocks(blocks, 16 * MIB))
        self.assertEqual(slices, [[blocks[0]], [blocks[1]]])

    def test_empty(self):
        self.assertEqual(list(split_blocks([], 16 * MIB)), [])


class _RecordingEngine:
    """Stands in for the memfabric TransferEngine, recording block counts."""

    def __init__(self):
        self.calls = []
        self.on_call = None

    def batch_transfer_sync(self, session_id, src, dst, lengths):
        self.calls.append(len(lengths))
        if self.on_call is not None:
            self.on_call(session_id)
        return 0


def _make_manager(failed_sessions=()):
    """A real AscendKVManager with __init__ skipped.

    Only the transfer engine is faked, so the override under test, its
    ``super()`` call into MooncakeKVManager, and the session bookkeeping are
    all the production ones.
    """
    from sglang.srt.disaggregation.ascend.conn import AscendKVManager

    mgr = AscendKVManager.__new__(AscendKVManager)
    mgr.failed_sessions = set(failed_sessions)
    mgr.session_lock = threading.Lock()
    mgr.engine = _RecordingEngine()
    return mgr


class TestTransferAbort(CustomTestCase):
    def tearDown(self):
        reset_for_test()

    def _blocks(self, n, size):
        return [(i, i + 10000, size) for i in range(n)]

    def test_disabled_passes_the_whole_batch_through_untouched(self):
        reset_for_test()
        mgr = _make_manager()
        blocks = self._blocks(8, 8 * MIB)
        self.assertEqual(mgr._transfer_data("s1", blocks), 0)
        # One engine call with all 8 blocks: byte-identical to the old path.
        self.assertEqual(mgr.engine.calls, [8])

    def test_enabled_slices_and_preserves_every_block(self):
        reset_for_test()
        mgr = _make_manager()
        blocks = self._blocks(8, 8 * MIB)  # 64 MiB over a 16 MiB slice
        with envs.SGLANG_DISAGGREGATION_KV_TRANSFER_MBPS_DURING_COLLECTIVE.override(
            1_000_000  # effectively no wait; this exercises the slicing glue
        ):
            self.assertEqual(mgr._transfer_data("s1", blocks), 0)
        self.assertEqual(mgr.engine.calls, [2, 2, 2, 2])  # 8 blocks, 2 per slice

    def test_dead_session_stops_before_the_first_slice(self):
        reset_for_test()
        mgr = _make_manager(failed_sessions={"s1"})
        with envs.SGLANG_DISAGGREGATION_KV_TRANSFER_MBPS_DURING_COLLECTIVE.override(
            1_000_000
        ):
            self.assertNotEqual(mgr._transfer_data("s1", self._blocks(8, 8 * MIB)), 0)
        self.assertEqual(mgr.engine.calls, [])  # nothing pushed at a dead peer

    def test_session_dying_mid_transfer_stops_at_the_next_slice(self):
        reset_for_test()
        mgr = _make_manager()

        def die_after_two(session_id):
            if len(mgr.engine.calls) == 2:
                mgr.failed_sessions.add(session_id)

        mgr.engine.on_call = die_after_two
        with envs.SGLANG_DISAGGREGATION_KV_TRANSFER_MBPS_DURING_COLLECTIVE.override(
            1_000_000
        ):
            status = mgr._transfer_data("s1", self._blocks(16, 8 * MIB))
        self.assertNotEqual(status, 0)
        # Stopped at the next slice instead of pushing all 128 MiB at a corpse.
        self.assertEqual(len(mgr.engine.calls), 2)

    def test_other_sessions_are_unaffected(self):
        reset_for_test()
        mgr = _make_manager(failed_sessions={"dead"})
        with envs.SGLANG_DISAGGREGATION_KV_TRANSFER_MBPS_DURING_COLLECTIVE.override(
            1_000_000
        ):
            self.assertEqual(mgr._transfer_data("alive", self._blocks(4, MIB)), 0)
        self.assertEqual(mgr.engine.calls, [4])


class TestConflictingFeature(CustomTestCase):
    def tearDown(self):
        reset_for_test()

    def test_sibling_pacing_feature_is_reported(self):
        # The MoE-GEMM-window feature paces the same transfer by a conflicting
        # rule; a merge of both branches would apply cleanly and then fight.
        from sglang.srt.utils.common import temp_set_env

        reset_for_test()
        logger = "sglang.srt.disaggregation.ascend.kv_transfer_flow"
        # allow_sglang: this key has no descriptor on this branch -- it belongs
        # to the sibling feature we are warning about.
        with temp_set_env(
            allow_sglang=True, SGLANG_DISAGGREGATION_ENABLE_MOE_WINDOW_PACING="1"
        ):
            with envs.SGLANG_DISAGGREGATION_KV_TRANSFER_MBPS_DURING_COLLECTIVE.override(
                3072
            ):
                with self.assertLogs(logger, level="ERROR") as cm:
                    self.assertIsNotNone(get_kv_transfer_pacer())
        self.assertIn("MOE_WINDOW_PACING", cm.output[0])

    def test_no_complaint_when_the_sibling_is_off(self):
        reset_for_test()
        with envs.SGLANG_DISAGGREGATION_KV_TRANSFER_MBPS_DURING_COLLECTIVE.override(
            3072
        ):
            with self.assertNoLogs(
                "sglang.srt.disaggregation.ascend.kv_transfer_flow", level="ERROR"
            ):
                self.assertIsNotNone(get_kv_transfer_pacer())


class TestStats(CustomTestCase):
    def test_stats_counts_sends_and_bytes(self):
        pacer, _ = _pacer(10, 0, is_busy=lambda: True)
        for _ in range(5):
            pacer.send(MIB)
        st = pacer.stats()
        self.assertEqual(st["sends"], 5)
        self.assertEqual(st["bytes_sent"], 5 * MIB)
        self.assertGreater(st["wait_seconds"], 0.0)

    def test_unlimited_sends_are_counted_too(self):
        pacer, _ = _pacer(10, 0, is_busy=lambda: False)
        pacer.send(MIB)
        self.assertEqual(pacer.stats()["sends"], 1)


class TestConfiguration(CustomTestCase):
    def tearDown(self):
        reset_for_test()

    def test_off_by_default(self):
        reset_for_test()
        self.assertIsNone(get_kv_transfer_pacer())
        self.assertFalse(gate.installed)

    def test_enabled_installs_the_gate(self):
        reset_for_test()
        with envs.SGLANG_DISAGGREGATION_KV_TRANSFER_MBPS_DURING_COLLECTIVE.override(
            3072
        ):
            pacer = get_kv_transfer_pacer()
        self.assertIsNotNone(pacer)
        self.assertTrue(gate.installed)


if __name__ == "__main__":
    unittest.main()
