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
"""Tells whether this rank's interconnect is currently busy with a collective.

Consumers bracket a collective with ``collective_bracket`` and ask ``is_busy()``.
The only consumer today is the PD KV transfer limiter, which slows itself down
while a collective is running and releases as soon as one is not.

It lives under ``distributed/`` because the brackets sit in ``parallel_state``
and ``communication_op``; importing upward into ``disaggregation`` from there
would be a cycle.

How it decides
--------------
Two pieces of state, both cheap:

* a host-side counter of collectives the host is currently inside;
* one reused device event, recorded as each collective is handed to the stream.

``is_busy()`` is ``counter > 0 or not last_end.query()``.  The counter covers
the window before the first event is ever recorded; the event covers the far
larger window where the host has already moved on but the device has not caught
up.  ``query()`` maps to ``aclrtQueryEventStatus`` on Ascend: a host-side read
that never synchronizes the stream.

Why one event is enough: the bracketed collectives are synchronous calls on the
main compute stream, so they complete in stream order.  The most recent end
event completing therefore implies every earlier one has, and re-recording the
same event as each collective goes by simply tracks the latest.

Error direction
---------------
The dangerous answer would be "quiet" while the device is in fact mid
collective, and three things rule it out: the counter covers the pre-record
window; stream ordering makes a single event sufficient; and ``exit`` records
*before* decrementing, so a reader that sees ``counter == 0`` is guaranteed the
record already happened.

What remains is conservative.  Because the host runs ahead, the event keeps
being re-recorded onto later collectives, so through the collective-dense part
of a forward ``is_busy()`` stays true and the gaps between layers are not
released.  This is region-level rather than interval-level gating: it throttles
somewhat more than strictly necessary, never less.

Constraints this rests on
-------------------------
**Bracket only synchronous collectives on the main compute stream.**  The
"latest end event completing implies every earlier one has" argument holds only
while every bracketed collective is submitted to the same stream.  Bracketing a
call on a side stream would let the single event be recorded there while an
earlier main-stream collective is still running, and the gate would then read
quiet while the device is busy -- the one error direction this design rules
out.  There is no cheap runtime assert for this (sampling the current stream on
every collective costs more than the gate), so it is a review-time invariant.

**Do not install this gate on a role that captures graphs.**  ``exit`` records
on the main compute stream, so inside a capture region the record would be
captured too, and the lock would break a Dynamo trace.  Today only the PD
prefill sender installs it and prefill does not capture, but a future NPU
piecewise-compile path would have to move the bracket inside the custom op.

An event that was never recorded returns ``query() == True`` on Ascend
(``NPUEvent::query``: ``if (!is_created_) return true;``), which lands exactly
right here -- before any collective has run, and forever on a decode-only
worker, the gate reads quiet and nothing is throttled.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class _CollectiveGate:
    """Module singleton. Attribute reads only; no ``global`` statements.

    TODO: the state here (and the pacer's) belongs in a ``get_resources()``
    named slot per ``sglang-runtime-context``; it is a module singleton today
    because the consumer runs on the PD transfer worker threads, where the
    per-forward contextvar layer does not reach.
    """

    def __init__(self) -> None:
        self.installed = False
        self._lock = threading.Lock()
        self._inflight = 0
        self._event = None
        self._event_broken = False

    def install(self) -> None:
        self.installed = True

    def _event_or_none(self):
        if self._event is not None or self._event_broken:
            return self._event
        try:
            from sglang.srt.utils import get_device_module

            event_cls = getattr(get_device_module(), "Event", None)
            if event_cls is None:
                self._event_broken = True
            else:
                self._event = event_cls()
        except Exception as exc:  # pragma: no cover - device dependent
            self._event_broken = True
            logger.warning(
                "collective gate: device events unavailable (%s); the KV "
                "transfer will not be throttled.",
                exc,
            )
        return self._event

    def _enter(self) -> None:
        with self._lock:
            self._inflight += 1

    def _exit(self) -> None:
        with self._lock:
            event = self._event_or_none()
            if event is not None:
                try:
                    event.record()
                except Exception:  # pragma: no cover - device dependent
                    self._event_broken = True
                    self._event = None
            # Record first, then release the counter, so a reader that sees
            # zero in flight is guaranteed the event is already recorded.
            self._inflight = max(0, self._inflight - 1)

    def is_busy(self) -> bool:
        with self._lock:
            if self._inflight > 0:
                return True
            event = self._event
        if event is None:
            return False
        try:
            return not event.query()
        except Exception:  # pragma: no cover - device dependent
            if not self._event_broken:
                self._event_broken = True
                logger.warning(
                    "collective gate: event query failed; reading quiet from "
                    "now on, i.e. not throttling."
                )
            return False

    def reset_for_test(self) -> None:
        with self._lock:
            self.installed = False
            self._inflight = 0
            self._event = None
            self._event_broken = False


gate = _CollectiveGate()


class _CollectiveBracket:
    """Marks one collective. One attribute read when the feature is off.

    Only valid around a *synchronous* collective on the *main compute stream* --
    see the module docstring; a side-stream call would invert the gate's error
    direction.
    """

    __slots__ = ()

    def __enter__(self):
        if gate.installed:
            gate._enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if gate.installed:
            gate._exit()
        return False


collective_bracket = _CollectiveBracket()
