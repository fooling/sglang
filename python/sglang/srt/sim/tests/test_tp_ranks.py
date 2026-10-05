"""Multi-rank: the interception design is not single-rank-only.

"TP / EP / PD are not covered" was written down as a limitation. TP and EP are
not: one process per rank, a real gloo group, every rank running
event_loop_normal with nothing but the mock backend, and all ranks deciding
the same schedule.

The run spawns processes, so it is the slow test in this suite -- kept because
the claim it checks is one the deck makes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "run_k3_tp_sim.py"


def _run(tp_size: int, ep_size: int) -> str:
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", no_proxy="*", NO_PROXY="*",
    )
    out = subprocess.run(
        [sys.executable, str(RUNNER), str(tp_size), str(ep_size)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert out.returncode == 0, f"rc={out.returncode}\n{out.stdout[-3000:]}"
    return out.stdout


@pytest.fixture(scope="module")
def tp2_ep2() -> str:
    return _run(2, 2)


def test_two_tp_ranks_run_the_whole_workload(tp2_ep2):
    assert "TP=2 EP=2 RAN THE WHOLE WORKLOAD: True" in tp2_ep2
    assert "rank exit codes: [0, 0]" in tp2_ep2


def test_both_ranks_are_in_a_real_group_of_two(tp2_ep2):
    """Not two independent single-rank runs: one group, two members.

    ep world=2 as well, so expert parallelism is actually on rather than
    silently collapsing to 1 (a tp=2/ep=1 control run reports ep world=1).
    """
    for rank in (0, 1):
        assert f"[rank{rank}] tp world=2 rank_in_group={rank} ep world=2" in tp2_ep2


def test_every_rank_decides_the_same_schedule(tp2_ep2):
    """In TP each rank batches on its own copy of the state.

    A divergence here is a real bug: the ranks would be running different
    forwards. The runner compares the full per-batch schedule, not a count.
    """
    assert "all ranks agree on the schedule: True (11 forward batches)" in tp2_ep2
    assert "all ranks agree on the finishes: True" in tp2_ep2


def test_the_multi_rank_schedule_matches_the_single_rank_one(tp2_ep2):
    """Adding a rank must not change what the scheduler decides.

    The single-rank run is 11 forward batches with a1 stopping on its scripted
    token; so is this one.
    """
    for rank in (0, 1):
        assert f"[rank{rank}] forward batches=11 finished=6/6 stop=all finished" in (
            tp2_ep2
        )
    assert '"a1": "FINISH_MATCHED_TOKEN"' in tp2_ep2


def test_the_platform_blocker_is_sglangs_own_not_the_sims():
    """Why the runner sets SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0.

    SGLang's node-locality check (in_the_same_node_as, reached from the shm
    broadcaster) deadlocks on a box with no /dev/shm. That is reproducible with
    no sim code at all, so it is a platform limit rather than a limit of the
    interception design -- and it is avoided with SGLang's own switch, not a
    patch. Recorded here so nobody reads the env var as a sim workaround.
    """
    from sglang.srt.environ import envs
    from sglang.srt.sim import run_k3_tp_sim

    assert envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get() is True  # default on
    src = RUNNER.read_text()
    assert 'env.setdefault("SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "0")' in src
    assert run_k3_tp_sim.DIST_INIT.startswith("tcp://")
