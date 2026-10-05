"""Virtual clock for the timeout interception points.

scheduler.py reads the wall clock in exactly three ways (grepped, see
below) -- all through the module-level ``import time`` binding in
``managers/scheduler.py``:

    scheduler.py:1712   deadline = time.perf_counter() - timeout_s   (_abort_on_running_timeout)
    scheduler.py:2999   deadline = time.perf_counter() - timeout_s   (_abort_on_waiting_timeout)
    time.sleep(...)                                                  (idle-loop backoff, not a deadline read)

Both deadline reads use ``time.perf_counter()``; ``time.monotonic()`` also
appears elsewhere in scheduler.py (perf logging) but not on a timeout path
this prototype needs to prove. Because ``SGLANG_REQ_RUNNING_TIMEOUT`` /
``SGLANG_REQ_WAITING_TIMEOUT`` default to -1 (disabled, per environ.py:588-589
and the task brief), neither call site fires in run_smoke.py's default
config -- the harness exercises the virtual clock directly (step 7) rather
than through a triggered timeout, and that gap is called out explicitly in
FEASIBILITY.md rather than silently assumed to work end-to-end.

The interception mechanism: monkeypatch the ``time`` *name* inside
``sglang.srt.managers.scheduler``'s module namespace to a shim object whose
``.perf_counter`` / ``.monotonic`` read the virtual clock and whose
``.sleep`` advances it instead of blocking. This only rebinds the one name
scheduler.py's own global namespace points at -- it does not touch the real
stdlib ``time`` module, so nothing outside scheduler.py (or any other module
explicitly patched the same way) is affected. Zero source-line changes to
scheduler.py.
"""

from __future__ import annotations

import time as _real_time
from types import ModuleType


class VirtualClock:
    """A monotonically increasing counter, advanced only by explicit calls."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def monotonic(self) -> float:
        return self._t

    def perf_counter(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("VirtualClock cannot go backwards")
        self._t += seconds
        return self._t

    def sleep(self, seconds: float) -> None:
        """Advance the virtual clock instead of blocking the real thread."""
        self.advance(seconds)


class _ClockShim(ModuleType):
    """A stand-in for the ``time`` module, backed by one VirtualClock.

    Anything not overridden (time.time(), time.strftime(), ...) falls
    through to the real ``time`` module via ``__getattr__``, so a module
    patched to use this shim keeps working for everything except the three
    functions the sim cares about.
    """

    def __init__(self, clock: VirtualClock):
        super().__init__("sglang.srt.sim._clock_shim")
        self._clock = clock

    def perf_counter(self) -> float:
        return self._clock.perf_counter()

    def monotonic(self) -> float:
        return self._clock.monotonic()

    def sleep(self, seconds: float) -> None:
        self._clock.sleep(seconds)

    def __getattr__(self, name):
        return getattr(_real_time, name)


def patch_module_clock(module, clock: VirtualClock) -> _ClockShim:
    """Rebind ``module``'s ``time`` name to a VirtualClock-backed shim.

    Returns the shim (so a caller can restore with
    ``module.time = real_time_module`` if needed). Only valid for modules
    that did ``import time`` at module scope (scheduler.py does).
    """
    shim = _ClockShim(clock)
    module.time = shim
    return shim
