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


def test_sglang_ships_its_own_fake_pd_transfer_path():
    """Testing PD without real transfers is SGLang's own capability.

    This is the answer to "can PD be exercised without device memory": yes,
    and not by anything of ours. A request whose bootstrap_host is the
    FAKE_BOOTSTRAP_HOST sentinel (or the backend set to "fake") makes both
    sides pick the fake sender/receiver -- prefill.py:334 and decode.py:718
    through _is_fake_transfer. What real memory buys is byte correctness and
    bandwidth, not the handshake.
    """
    from sglang.srt.disaggregation.decode import DecodePreallocQueue  # noqa: F401
    from sglang.srt.disaggregation.fake.conn import FakeKVReceiver, FakeKVSender
    from sglang.srt.disaggregation.utils import (
        FAKE_BOOTSTRAP_HOST,
        KVClassType,
        TransferBackend,
        get_kv_class,
    )

    assert FAKE_BOOTSTRAP_HOST == "2.2.2.2"
    assert get_kv_class(TransferBackend.FAKE, KVClassType.SENDER) is FakeKVSender
    assert get_kv_class(TransferBackend.FAKE, KVClassType.RECEIVER) is FakeKVReceiver
    # both roles consult the same predicate, so neither side needs a real move
    import inspect

    from sglang.srt.disaggregation import decode as decode_mod
    from sglang.srt.disaggregation import prefill as prefill_mod

    assert "FAKE_BOOTSTRAP_HOST" in inspect.getsource(prefill_mod)
    assert "_is_fake_transfer" in inspect.getsource(decode_mod)


def test_the_transfer_engine_reports_a_move_and_can_be_charged_for_it():
    """The mock engine is where "how long did the KV move take" comes from.

    0 is success -- the same value SGLang's own code returns for a layer with
    nothing to transfer. The duration comes from the cost seam, like the
    forward slice, which is what a performance simulation needs from PD.
    """
    from sglang.srt.sim import register

    eng = register.MockTransferEngine("127.0.0.1", 0, "prefill")
    assert eng.get_session_id().startswith("sim-session")
    assert eng.batch_transfer_sync("s", [1, 2], [3, 4], [64, 64]) == 0
    assert eng.transfers[-1] == ("s", 2, 128)

    # The seam is checked by what it is asked, not by moving the shared clock:
    # advancing it here would leak into the K3 run's own clock reconciliation
    # (which is exactly how this was caught).
    asked = []

    def cost(nbytes):
        asked.append(nbytes)
        return 0.0

    before = register.virtual_clock().perf_counter()
    register.transfer_cost_hook(cost)
    try:
        eng.batch_transfer_sync("s", [1], [2], [1_000_000])
    finally:
        register.transfer_cost_hook(None)
    assert asked == [1_000_000]
    assert register.virtual_clock().perf_counter() == before


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
