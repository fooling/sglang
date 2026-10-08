"""Every selftest has to pass in a process of its own, and run_smoke has to run.

Why this file exists. test_sim_interception.py already asserts
``selftest_execution_shim() is True``, and the whole suite was green for
three days while ``run_smoke.py`` could not get past step 0. By the time
that assert runs, an earlier test in the same process has already published
config and left ``_ATTN_DP_SIZE`` set, so ``_init_sim_dp_attention``
early-returns and the path a fresh process actually takes is never
exercised. The order dependency produced a false green.

These tests spend a process per check, which is the only way to say
anything about what a runner will hit on its first line.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SELFTESTS = ["execution", "kv", "transfer", "clock"]

_SNIPPET = (
    "from sglang.srt.sim import register\n"
    "register.install()\n"
    "ok = getattr(register, 'selftest_{name}_shim')()\n"
    "raise SystemExit(0 if ok else 1)\n"
)


def _offline_env():
    """The sim never reaches the network; a test that hangs on a download is
    a test that cannot fail fast."""
    return dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        no_proxy="*",
        NO_PROXY="*",
    )


def _tail(text: str, n: int = 2000) -> str:
    return text[-n:] if text else ""


@pytest.mark.parametrize("name", SELFTESTS)
def test_selftest_passes_in_a_fresh_process(name):
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(name=name)],
        capture_output=True,
        text=True,
        timeout=300,
        env=_offline_env(),
    )
    assert proc.returncode == 0, (
        f"selftest_{name}_shim does not pass in a process of its own "
        f"(exit={proc.returncode}); a runner hits exactly this on its first "
        f"line:\n{_tail(proc.stdout)}\n{_tail(proc.stderr)}"
    )


def test_run_smoke_exits_zero():
    """The end-to-end guard: this is the thing that was broken while the
    suite was green."""
    script = Path(__file__).resolve().parents[1] / "run_smoke.py"
    assert script.exists(), script
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
        env=_offline_env(),
    )
    assert proc.returncode == 0, (
        f"run_smoke.py exited {proc.returncode}:\n"
        f"{_tail(proc.stdout)}\n{_tail(proc.stderr)}"
    )
    assert "all 4 interception-point selftests passed: True" in proc.stdout
