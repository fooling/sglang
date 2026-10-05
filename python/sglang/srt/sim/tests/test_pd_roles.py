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


def test_the_kv_counts_are_real_even_though_nothing_moves(pd_out):
    """The sizes have to be right, or PD would look free in a timing run.

    A cost model computes a transfer from bytes. A pool reporting zero bytes
    would make every PD handoff cost nothing -- so the shape comes off the
    model config: one region per full-attention layer, bytes per token from
    the MLA latent width, and the span consistent with the pool size.
    """
    import re

    for role in ("prefill", "decode"):
        m = re.search(
            rf"\[{role}\] kv regions=(\d+) \(one per full-attention layer\) "
            rf"item=(\d+)B span=(\d+)B total=(\d+)B", pd_out)
        assert m, f"{role} 没报 KV 字节数：{pd_out[-600:]}"
        regions, item, span, total = (int(x) for x in m.groups())
        assert regions > 0 and item > 0
        assert total == regions * span
        assert f"[{role}] moving a 48-token prompt would be" in pd_out


def test_the_kv_shape_is_the_configs_not_ours():
    """Derived from the model config, including the hybrid layer split.

    On K3 only the full-attention layers hold KV pages -- the linear-attention
    layers keep state in the mamba pool -- so the KV layer count is 8, not 32.
    MLA keeps one latent entry per token, so head_num is 1 and head_dim is
    kv_lora_rank + qk_rope_head_dim.
    """
    import torch

    from sglang.srt.configs.kimi_linear import KimiLinearConfig
    from sglang.srt.sim.cpu_kv import kv_shape_from_config
    from sglang.srt.sim.run_k3_sim import K3_TEXT_CONFIG

    cfg = KimiLinearConfig(**{k: v for k, v in K3_TEXT_CONFIG.items()
                              if k != "architectures"})
    shape = kv_shape_from_config(cfg, torch.float16)
    assert shape["total_layers"] == 32
    assert shape["kv_layers"] == len(cfg.full_attention_layer_ids) == 8
    assert shape["head_num"] == 1
    assert shape["head_dim"] == cfg.kv_lora_rank + cfg.qk_rope_head_dim == 576
    assert shape["bytes_per_token"] == 576 * 2        # fp16


def test_a_shapeless_kv_cache_refuses_rather_than_reporting_zero():
    """Zero bytes would be believed by a cost model, so it has to raise."""
    import torch

    from sglang.srt.sim.cpu_kv import SimKVCache

    bare = SimKVCache(dtype=torch.float16)
    with pytest.raises(RuntimeError, match="no shape"):
        bare.get_contiguous_buf_infos()


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


def test_the_kv_operators_are_really_called_with_real_shapes():
    """The seam is the operator body, not the call site.

    These accessors used to raise, on the reasoning that returning zeros would
    hide a dependency. For a simulator that is the wrong cut: raising blocks
    the path, so the KV write, the PD transfer and the offload never get
    exercised. What a mock needs is the real shape, the call actually
    happening, and an empty operator behind it.

    Counting is what replaces raising as the guard: the test asserts the
    operator ran and with which shapes, so a silently skipped call fails here.
    """
    import torch

    from sglang.srt.sim.cpu_kv import SimKVCache

    kv = SimKVCache(dtype=torch.float16)
    kv.size = 64
    kv.apply_shape(dict(kv_layers=8, head_num=1, head_dim=576,
                        bytes_per_token=1152))

    # real shapes out
    k = kv.get_key_buffer(0)
    assert tuple(k.shape) == (64, 1, 576) and k.dtype is torch.float16
    assert tuple(kv.get_kv_buffer_shape()[0]) == (64, 1, 576)

    # the write operator is callable, and the call is recorded
    layer = type("L", (), {"layer_id": 3})()
    kv.set_kv_buffer(layer, torch.arange(8),
                     torch.zeros((8, 1, 576), dtype=torch.float16))
    assert kv.ops_called["set_kv_buffer"] == 1
    assert kv.op_shapes["set_kv_buffer"] == [{"layer": 3, "slots": 8}]

    # the offload operators too, one entry per KV layer
    copied = kv.get_cpu_copy(list(range(5)))
    assert len(copied) == 8 and tuple(copied[0].shape) == (5, 1, 576)
    kv.load_cpu_copy(copied, list(range(5)))
    assert kv.ops_called["get_cpu_copy"] == kv.ops_called["load_cpu_copy"] == 1

    # an empty operator must still refuse a shape that cannot be right
    with pytest.raises(ValueError, match="slots but cache_k"):
        kv.set_kv_buffer(layer, torch.arange(8),
                         torch.zeros((7, 1, 576), dtype=torch.float16))


def test_a_shapeless_kv_cache_still_refuses():
    """No model config means no shape, and a zero shape is not an answer."""
    import torch

    from sglang.srt.sim.cpu_kv import SimKVCache

    bare = SimKVCache(dtype=torch.float16)
    with pytest.raises(RuntimeError, match="no shape"):
        bare.get_contiguous_buf_infos()
    with pytest.raises(RuntimeError, match="no shape"):
        bare.get_key_buffer(0)
