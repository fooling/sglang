"""The sim has to be on the NPU branch, and the stand-in must never invent numbers.

is_npu() gates 174 call sites. On the non-NPU branch the sim would exercise a
different attention backend, different layer implementations and a different
transfer engine -- a control plane no 910 runs, which is the opposite of what
the sim claims. These tests pin that down, and pin down the one property that
makes a stand-in honest: it either computes or it raises.
"""

import subprocess
import sys

import pytest
import torch

from sglang.srt.sim import fake_npu


def test_process_is_on_the_npu_branch():
    from sglang.srt.utils.common import is_npu

    assert is_npu() is True, (
        "this process is on the non-NPU branch; conftest.py has to install the "
        "stand-in before anything under sglang.srt.layers is imported"
    )
    fake_npu.assert_npu_branch()


def test_npu_device_strings_parse_but_allocate_on_cpu():
    """torch.device("npu:0") has to be legal -- parallel_state.py:339 builds it
    -- while the bytes land on CPU, because there is no device memory."""
    dev = torch.device("npu:0")
    assert dev.type == "npu"
    t = torch.ones(4, device="npu:0")
    assert t.device.type == "cpu", "an allocation reached a device that is not there"


def test_an_uncovered_op_raises_instead_of_returning_a_tensor():
    """The property the whole stand-in rests on.

    A faked tensor would be indistinguishable from a computed one, and every
    number downstream of it would be fiction. So an op with no plain-torch
    equivalent raises, and the message names it.
    """
    import torch_npu

    with pytest.raises(fake_npu.NpuOpNotInSim) as excinfo:
        torch_npu.npu_fused_infer_attention_score_v2(1, 2)
    assert "npu_fused_infer_attention_score_v2" in str(excinfo.value)


def test_covered_op_actually_computes():
    """npu_scatter_nd_update_ is the KV write; it has to really write."""
    import torch_npu

    buf = torch.zeros(4, 2)
    idx = torch.tensor([[1], [3]])
    torch_npu.npu_scatter_nd_update_(buf, idx, torch.ones(2, 2) * 5)
    assert buf[1].tolist() == [5.0, 5.0] and buf[3].tolist() == [5.0, 5.0]
    assert buf[0].tolist() == [0.0, 0.0]


def test_runner_reports_the_npu_branch_in_a_fresh_process():
    """What a runner prints on its first step, in a process of its own."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "from sglang.srt.sim.fake_npu import install_fake_npu, assert_npu_branch\n"
         "install_fake_npu()\n"
         "assert_npu_branch()\n"
         "from sglang.srt.runtime_context import attention_backends\n"
         "from sglang.srt.sim.run_smoke import build_server_args\n"
         "build_server_args()\n"
         "print('BACKEND', attention_backends()[0])\n"],
        capture_output=True, text=True, timeout=300,
        env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "HF_DATASETS_OFFLINE": "1", "no_proxy": "*", "NO_PROXY": "*",
             "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "BACKEND ascend" in proc.stdout, proc.stdout[-2000:]
