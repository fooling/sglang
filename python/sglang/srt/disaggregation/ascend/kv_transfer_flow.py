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
"""Rate-limit the prefill->decode KV transfer while collectives are running.

On Ascend the KV RDMA and the model's collectives share the same NPU egress and
HBM read bandwidth, and the transfer is issued by background CPU threads with
no relation to the forward -- so a chunk's KV lands on top of the next chunk's
AllGather / ReduceScatter / all-to-all.  This caps the rate while the device is
busy with a collective and lifts the cap as soon as it is not.

Only the issue rate changes.  The bytes, their order, and every existing
failure path are untouched.

Pacing
------
Deadline pacing: remember the instant the flow may next send, wait until then,
and push it out by ``amount / rate``::

    start     = max(next_send, now)
    sleep(start - now)
    next_send = start + amount / rate

No token counter to refill, no burst budget to size, no loop -- and therefore
no way to spin.  A flow that has been quiet has ``next_send`` in the past, so
its next send is free and pacing resumes right after: the self-sizing version
of a burst allowance.

One sleep per call is enough because the caller meters in bounded slices (see
``SLICE_BYTES``): a single wait is a couple of milliseconds, so a gate that
flips is picked up on the very next slice.

Liveness
--------
The wait depends only on the wall clock, so nothing the gate does -- including
never observing a collective at all -- can hold a transfer.  The during-collective
rate is required to be positive, and a gate that cannot read its device reports
quiet, which does not throttle.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from sglang.srt.distributed.collective_gate import gate
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024

# One metered slice. Small enough that a single wait stays in the low
# milliseconds, large enough that the per-call memfabric overhead stays noise.
SLICE_BYTES = 16 * _MIB

TransferBlock = Tuple[int, int, int]  # (src_addr, dst_addr, length)

# Periodic debug line so the throttling can be read off a running server.
_LOG_EVERY_N_SENDS = 256


class KvTransferPacer:
    """Paces one byte stream at a rate chosen by the collective gate."""

    def __init__(
        self,
        busy_bytes_per_s: float,
        quiet_bytes_per_s: float,
        is_busy: Callable[[], bool] = gate.is_busy,
        now_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._busy_rate = float(busy_bytes_per_s)
        self._quiet_rate = float(quiet_bytes_per_s)
        self._is_busy = is_busy
        self._now = now_fn
        self._sleep = sleep_fn
        self._lock = threading.Lock()
        self._next_send = now_fn()
        self._probe_failed = False
        self.bytes_sent = 0
        self.wait_seconds = 0.0
        self.sends = 0

    def _rate(self) -> Optional[float]:
        """Bytes per second right now; ``None`` means unlimited."""
        try:
            busy = self._is_busy()
        except Exception as exc:  # pragma: no cover - device dependent
            if not self._probe_failed:
                self._probe_failed = True
                logger.warning(
                    "KV transfer pacer: collective probe failed (%s); not "
                    "throttling from now on.",
                    exc,
                )
            return None
        rate = self._busy_rate if busy else self._quiet_rate
        return rate if rate > 0 else None

    def send(self, nbytes: int) -> float:
        """Wait until ``nbytes`` may go out. Returns seconds waited."""
        if nbytes <= 0:
            return 0.0
        with self._lock:
            rate = self._rate()
            now = self._now()
            if rate is None:
                # Unlimited right now; do not carry a stale deadline into a
                # later throttled moment.
                self._next_send = now
                self.bytes_sent += nbytes
                self.sends += 1
                return 0.0
            start = self._next_send if self._next_send > now else now
            wait = start - now
            self._next_send = start + nbytes / rate
            self.bytes_sent += nbytes
            self.wait_seconds += wait
            self.sends += 1
            report = self.sends % _LOG_EVERY_N_SENDS == 0
        if wait > 0:
            self._sleep(wait)
        if report:
            logger.debug("KV transfer pacer: %s", self.stats())
        return wait

    def stats(self) -> dict:
        return {
            "sends": self.sends,
            "bytes_sent": self.bytes_sent,
            "wait_seconds": round(self.wait_seconds, 6),
        }


def split_blocks(
    blocks: Sequence[TransferBlock], slice_bytes: int = SLICE_BYTES
) -> Iterator[List[TransferBlock]]:
    """Group transfer blocks into slices of at most ``slice_bytes``.

    Order is preserved and every block appears exactly once.  A block larger
    than the slice gets one to itself rather than being split, because a block
    is a single contiguous RDMA range.
    """
    current: List[TransferBlock] = []
    current_bytes = 0
    for block in blocks:
        length = block[2]
        if current and current_bytes + length > slice_bytes:
            yield current
            current = []
            current_bytes = 0
        current.append(block)
        current_bytes += length
    if current:
        yield current


_pacer: Optional[KvTransferPacer] = None
_resolved = False
_lock = threading.Lock()


def get_kv_transfer_pacer() -> Optional[KvTransferPacer]:
    """Return the pacer, or ``None`` when rate limiting is off."""
    global _pacer, _resolved
    if _resolved:
        return _pacer
    with _lock:
        if _resolved:
            return _pacer
        busy_mbps = envs.SGLANG_DISAGGREGATION_KV_TRANSFER_MBPS_DURING_COLLECTIVE.get()
        _resolved = True
        if busy_mbps <= 0:
            return None
        quiet_mbps = envs.SGLANG_DISAGGREGATION_KV_TRANSFER_MBPS.get()
        _pacer = KvTransferPacer(
            busy_bytes_per_s=busy_mbps * _MIB,
            quiet_bytes_per_s=max(0, quiet_mbps) * _MIB,
        )
        _warn_if_conflicting_pacing_feature()
        gate.install()
        logger.info(
            "PD KV transfer limited to %d MiB/s while a collective is running "
            "(%s otherwise).",
            busy_mbps,
            f"{quiet_mbps} MiB/s" if quiet_mbps > 0 else "unlimited",
        )
    return _pacer


# The sibling feature on npu/pd-kv-moe-gemm-window paces the same transfer by a
# different rule (it holds slices back until the device is inside a MoE expert
# GEMM). The two branches touch different functions, so a merge of both would
# apply cleanly and then quietly fight: this limiter keeps the rate down through
# the collective-dense region, which is exactly the region the window feature is
# trying to release into. Checked through os.environ because that feature's
# variable is not declared on this branch.
_CONFLICTING_ENV = "SGLANG_DISAGGREGATION_ENABLE_MOE_WINDOW_PACING"


def _warn_if_conflicting_pacing_feature() -> None:
    value = os.environ.get(_CONFLICTING_ENV, "")
    if value.strip().lower() in ("", "0", "false", "off", "none"):
        return
    logger.error(
        "%s is set alongside the KV transfer rate limit. Both pace the same "
        "transfer by conflicting rules and the combination performs worse than "
        "either alone; enable exactly one.",
        _CONFLICTING_ENV,
    )


def reset_for_test() -> None:
    global _pacer, _resolved
    with _lock:
        _pacer = None
        _resolved = False
    gate.reset_for_test()
