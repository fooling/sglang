"""PD disaggregation: the control plane comes up, the data plane has no bytes.

"PD is not covered" was written down as a limitation. Its control plane is
not: both roles construct on a Scheduler with no weights and no device, and
SGLang builds its own PD queues around them.

Where it genuinely stops is the KV transfer, and that stop is a fidelity
limit, not a design one: the sim allocates KV indices for real but has no
tensors, so there is nothing to register with the transfer engine and nothing
to move between a prefill and a decode instance.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "run_pd_sim.py"


@pytest.fixture(scope="module")
def pd_out() -> str:
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", no_proxy="*", NO_PROXY="*",
    )
    out = subprocess.run(
        [sys.executable, str(RUNNER)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert out.returncode == 0, f"rc={out.returncode}\n{out.stdout[-3000:]}"
    return out.stdout


def test_both_roles_construct(pd_out):
    assert "BOTH PD ROLES CONSTRUCTED ON THE SIM BACKEND: True" in pd_out
    assert "[prefill] ROLE OK True" in pd_out
    assert "[decode] ROLE OK True" in pd_out


def test_sglangs_own_pd_queues_are_what_got_built(pd_out):
    """Not objects of ours standing in for them -- the module's own classes."""
    assert "[prefill] disagg_prefill_bootstrap_queue: PrefillBootstrapQueue" in pd_out
    assert "[decode] disagg_decode_prealloc_queue: DecodePreallocQueue" in pd_out
    assert "[decode] disagg_decode_transfer_queue: DecodeTransferQueue" in pd_out


def test_pd_picked_the_transfer_face_the_sim_occupies(pd_out):
    """The sim replaces the ascend engine, so PD has to be pointed at it.

    The default backend is mooncake, whose library is absent here and whose
    engine therefore comes back None -- which is how this was found. Selecting
    ascend is a server arg, not a patch, and on an Ascend box it is the right
    engine anyway.
    """
    for role in ("prefill", "decode"):
        assert f"[{role}] transfer engine class from: sglang.srt.sim" in pd_out


def test_the_kv_transfer_has_nothing_to_carry(pd_out):
    """Stated, not papered over: zero buffers to register, on both roles."""
    for role in ("prefill", "decode"):
        assert f"[{role}] kv buffers to register: 0" in pd_out


def test_the_sim_kv_cache_still_refuses_to_hand_out_values():
    """The buffer-info accessor returning empty must not have softened the rest.

    get_contiguous_buf_infos and maybe_get_custom_mem_pool are pool accessors
    and answer honestly. Anything that hands out KV *contents* must still
    raise, or zeros would silently stand in for real attention state.
    """
    import torch

    from sglang.srt.sim.cpu_kv import SimKVCache

    kv = SimKVCache(dtype=torch.float16)
    assert kv.get_contiguous_buf_infos() == ([], [], [])
    assert kv.maybe_get_custom_mem_pool() is None
    raisers = [n for n in ("get_key_buffer", "get_value_buffer", "get_kv_buffer",
                           "set_kv_buffer")
               if hasattr(kv, n)]
    assert raisers, "SimKVCache has no value accessors left to guard"
    for name in raisers:
        with pytest.raises(NotImplementedError):
            getattr(kv, name)(0)
