# Copyright 2023-2026 SGLang Team
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
"""Pace PD KV transfers into the MoE expert-GEMM window.

On Ascend, the prefill->decode KV RDMA and the MoE DeepEP all-to-all share the
same NPU egress (device RDMA / UB) and HBM read bandwidth.  The KV transfer is
issued by background CPU threads with no relation to the forward, so a chunk's
transfer lands on top of the *next* chunk's dispatch/combine traffic.

Every MoE layer has a phase where the network is idle: in
``FusedMoE.forward_deepep`` the sequence is

    dispatch()  ->  run_moe_core()  ->  combine()
                    ^^^^^^^^^^^^^^ grouped expert GEMM, network idle

so the dispatcher's ``post_dispatch`` / ``pre_combine`` hooks bracket exactly
that phase.

WHY DEVICE EVENTS AND NOT A HOST FLAG
-------------------------------------
The hooks run on the *host* thread, but the phase we care about is a *device*
phase.  ``DeepEPDispatcher.dispatch`` ends in ``dispatch_b``, which only calls
``event.current_stream_wait()`` -- it enqueues a stream wait and returns
immediately.  The host therefore races ahead and is typically several layers in
front of the device, so "the host is between post_dispatch and pre_combine"
says almost nothing about where the device is.

Instead each hook records a device event on the forward stream, and the
transfer worker polls those events with ``Event.query()``.  The device is
inside a GEMM iff some slot has its start event completed and its end event not
yet completed.  A ring of slots is required precisely because the host run-ahead
means several layers are in flight at once.

ASCEND (CANN) EVENT SEMANTICS THIS RELIES ON
--------------------------------------------
Verified against torch_npu ``NPUEvent`` / ``AclInterface``, not assumed from
CUDA:

* ``Event.query()`` maps to ``aclrtQueryEventStatus`` and compares against
  ``ACL_EVENT_RECORDED_STATUS_COMPLETE``.  It is a host-side status read and
  never synchronizes the stream (``NPUEvent::query``).
* An event that was never recorded is lazily *uncreated*, and
  ``NPUEvent::query`` returns **true** for it (``if (!is_created_) return
  true;``).  So "query() is true" alone does NOT mean a GEMM finished -- every
  read below is therefore gated on our own ``start_recorded`` /
  ``end_recorded`` bookkeeping, never on the event state alone.
* ``NPUEvent::record`` goes through torch_npu's task queue
  (``LaunchRecordEventTask``), so the ACL record lands after the host call
  returns.  While it is still queued, ``query()`` deliberately returns false
  (``GetTaskQueueEnable() && !IsEventRecorded(...)``).  That is the safe
  direction for us: a not-yet-landed record reads as "no window", so we fall
  back to sending unpaced rather than sending at the wrong time.
* Event reuse: when CANN exposes ``aclrtCreateEventExWithFlag`` torch_npu
  creates ``ACL_EVENT_SYNC`` events, which per its own comment "can be reused
  naturally, aclrtResetEvent is not supported" and have "no limit on the number
  of events" -- so re-recording a ring slot every pass is correct and we must
  NOT call ``reset()`` (``NPUEvent::reset`` only accepts ``ACL_EVENT_EXTERNAL``
  events and would raise).  On older CANN it falls back to the legacy
  ``aclrtCreateEventWithFlag``, which *does* cap the number of live events --
  hence RING_SIZE is a tunable and defaults to a modest 64 pairs.

Nothing here changes *what* is transferred, only *when* the RDMA is issued; the
KV bytes are already final when the chunk is enqueued.  Every wait is bounded
and fails open, so a stalled or absent MoE forward can never hold a request:
the worker transfers anyway once the deadline passes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, TypeVar

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MoeTransferWindowStats:
    """Cheap counters; read by tests and for on-NPU triage."""

    windows_recorded: int = 0
    slices_in_window: int = 0
    slices_timed_out: int = 0
    slices_unpaced: int = 0  # no device-event support; sent immediately

    def as_dict(self) -> dict:
        return {
            "windows_recorded": self.windows_recorded,
            "slices_in_window": self.slices_in_window,
            "slices_timed_out": self.slices_timed_out,
            "slices_unpaced": self.slices_unpaced,
        }


class MoeTransferWindow:
    """Tracks, in device time, whether a MoE expert GEMM is currently running.

    ``open`` / ``close`` are called from the forward thread via dispatcher
    hooks and only record device events; ``wait_for_window`` is called from the
    PD transfer worker threads and polls those events.

    Event recording is skipped entirely while no transfer is waiting, so the
    steady-state cost on the forward path is one atomic read per hook.
    """

    def __init__(
        self,
        wait_timeout_s: float,
        poll_interval_s: float = 0.0002,
        ring_size: int = 64,
        event_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._wait_timeout_s = wait_timeout_s
        self._poll_interval_s = poll_interval_s
        self._ring_size = max(1, ring_size)
        self._event_factory = event_factory
        self._event_factory_resolved = event_factory is not None
        # Each slot: [start_event, end_event, start_recorded, end_recorded]
        self._ring: List[list] = []
        self._slot = -1
        self._waiters = 0
        self._stats = MoeTransferWindowStats()

    # -- config -------------------------------------------------------------

    @property
    def wait_timeout_s(self) -> float:
        return self._wait_timeout_s

    @property
    def stats(self) -> MoeTransferWindowStats:
        return self._stats

    def _resolve_event_factory(self) -> Optional[Callable[[], object]]:
        """Look up the device Event class lazily, once."""
        if self._event_factory_resolved:
            return self._event_factory
        self._event_factory_resolved = True
        try:
            from sglang.srt.utils import get_device_module

            device_module = get_device_module()
            event_cls = getattr(device_module, "Event", None)
            if event_cls is not None:
                # enable_timing must stay off: we only need recorded status,
                # and timing-enabled events are more expensive on CANN.
                probe = event_cls()
                probe.query()
                self._event_factory = event_cls
        except Exception as exc:  # pragma: no cover - device dependent
            logger.warning(
                "MoE transfer window disabled: device events unavailable (%s). "
                "KV transfers will be issued without pacing.",
                exc,
            )
            self._event_factory = None
        return self._event_factory

    def _ensure_ring(self) -> bool:
        factory = self._resolve_event_factory()
        if factory is None:
            return False
        if not self._ring:
            self._ring = [
                [factory(), factory(), False, False] for _ in range(self._ring_size)
            ]
        return True

    # -- forward-thread side ------------------------------------------------

    def open(self, stream=None) -> None:
        """Expert GEMM is about to be enqueued: mark its start on the stream."""
        # Fast path: nothing is waiting, so do not pay for an event record.
        if self._waiters == 0:
            return
        with self._lock:
            if self._waiters == 0 or not self._ensure_ring():
                return
            self._slot = (self._slot + 1) % self._ring_size
            slot = self._ring[self._slot]
            slot[2] = False
            slot[3] = False
            _record(slot[0], stream)
            slot[2] = True
            self._stats.windows_recorded += 1

    def close(self, stream=None) -> None:
        """Combine is about to be enqueued: mark the GEMM's end on the stream."""
        if self._waiters == 0:
            return
        with self._lock:
            if self._slot < 0 or not self._ring:
                return
            slot = self._ring[self._slot]
            if not slot[2] or slot[3]:
                return
            _record(slot[1], stream)
            slot[3] = True

    # -- transfer-thread side -----------------------------------------------

    def device_in_gemm(self) -> bool:
        """True iff the device is currently inside a recorded GEMM phase."""
        with self._lock:
            if not self._ring:
                return False
            slots = list(self._ring)
        for start_ev, end_ev, start_recorded, end_recorded in slots:
            # NOTE: on Ascend an unrecorded event queries as True, so the
            # *_recorded flags -- not the event state -- decide whether a slot
            # carries information at all.
            if not start_recorded:
                continue
            # query() -> aclrtQueryEventStatus: host-side, never synchronizes.
            if not _query(start_ev):
                continue  # device has not reached this GEMM yet
            if end_recorded and _query(end_ev):
                continue  # device already finished this GEMM
            return True
        return False

    def wait_for_window(self, timeout_s: Optional[float] = None) -> bool:
        """Block until the device is inside a MoE expert GEMM.

        Returns ``True`` if the caller was admitted inside a window and
        ``False`` if the deadline passed first (or the device has no usable
        events).  A ``False`` return is not an error -- the caller must proceed
        with the transfer regardless, so an idle or non-MoE forward never
        stalls a request.
        """
        if timeout_s is None:
            timeout_s = self._wait_timeout_s

        with self._lock:
            self._waiters += 1
            has_events = self._ensure_ring()
        try:
            if not has_events:
                self._stats.slices_unpaced += 1
                return False
            deadline = time.perf_counter() + timeout_s
            while True:
                if self.device_in_gemm():
                    self._stats.slices_in_window += 1
                    return True
                if time.perf_counter() >= deadline:
                    self._stats.slices_timed_out += 1
                    return False
                time.sleep(self._poll_interval_s)
        finally:
            with self._lock:
                self._waiters -= 1

    # -- test support -------------------------------------------------------

    def force_waiter_for_test(self, count: int = 1) -> None:
        """Pretend a transfer is pending so the hooks record events."""
        with self._lock:
            self._waiters += count


def _record(event, stream) -> None:
    try:
        if stream is not None:
            event.record(stream)
        else:
            event.record()
    except TypeError:  # pragma: no cover - backend signature differences
        event.record()


def _query(event) -> bool:
    try:
        return bool(event.query())
    except Exception:  # pragma: no cover - device dependent
        return True


_global_window: Optional[MoeTransferWindow] = None
_global_window_lock = threading.Lock()


def moe_window_pacing_enabled() -> bool:
    return envs.SGLANG_DISAGGREGATION_ENABLE_MOE_WINDOW_PACING.get()


def get_moe_transfer_window() -> MoeTransferWindow:
    """Return the process-global window, creating it on first use."""
    global _global_window
    if _global_window is None:
        with _global_window_lock:
            if _global_window is None:
                _global_window = MoeTransferWindow(
                    wait_timeout_s=(
                        envs.SGLANG_DISAGGREGATION_MOE_WINDOW_WAIT_MS.get() / 1000.0
                    ),
                    poll_interval_s=(
                        envs.SGLANG_DISAGGREGATION_MOE_WINDOW_POLL_US.get() / 1e6
                    ),
                    ring_size=envs.SGLANG_DISAGGREGATION_MOE_WINDOW_RING_SIZE.get(),
                )
    return _global_window


def set_moe_transfer_window_for_test(window: Optional[MoeTransferWindow]) -> None:
    global _global_window
    with _global_window_lock:
        _global_window = window


def reset_moe_transfer_window_for_test() -> None:
    set_moe_transfer_window_for_test(None)


def slice_for_moe_window(items: Sequence[T], group_size: int) -> List[List[T]]:
    """Split a transfer's per-layer params into window-sized slices.

    ``group_size <= 0`` disables slicing (one slice with everything), which is
    the byte-for-byte equivalent of the unpaced path.
    """
    if not items:
        return []
    if group_size <= 0 or group_size >= len(items):
        return [list(items)]
    return [
        list(items[start : start + group_size])
        for start in range(0, len(items), group_size)
    ]


def moe_window_layer_group_size() -> int:
    return envs.SGLANG_DISAGGREGATION_MOE_WINDOW_LAYER_GROUP.get()


# ---------------------------------------------------------------------------
# Dispatcher hook installation
# ---------------------------------------------------------------------------

_HOOKS_INSTALLED_ATTR = "_sglang_moe_transfer_window_hooks"


def maybe_install_moe_transfer_window_hooks(dispatcher) -> bool:
    """Install the GEMM-phase event hooks on ``dispatcher``, once per instance.

    Returns whether hooks are installed on this dispatcher (including when a
    previous call already installed them).  The hooks are permanent: unlike the
    one-shot fine-grained dual-stream hooks they must fire on every layer of
    every forward, so they never remove themselves.
    """
    if dispatcher is None or not moe_window_pacing_enabled():
        return False
    if getattr(dispatcher, _HOOKS_INSTALLED_ATTR, False):
        return True

    window = get_moe_transfer_window()

    def _open_window_hook(_dispatcher, dispatch_output):
        # The grouped expert GEMM is enqueued right after this returns.
        window.open()
        return None

    def _close_window_hook(_dispatcher, combine_input):
        # Combine puts traffic back on the wire right after this returns.
        window.close()
        return None

    dispatcher.register_post_dispatch_hook(_open_window_hook)
    dispatcher.register_pre_combine_hook(_close_window_hook)
    setattr(dispatcher, _HOOKS_INSTALLED_ATTR, True)
    logger.info(
        "PD KV transfer will be paced into the MoE expert-GEMM window "
        "(layer_group=%d, wait=%.1fms).",
        moe_window_layer_group_size(),
        window.wait_timeout_s * 1000.0,
    )
    return True
