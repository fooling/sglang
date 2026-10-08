"""The sim runs the NPU branch, so the tests have to run it too.

``is_npu()`` is lru_cached and 34 modules under sglang.srt.layers freeze
``_is_npu = is_npu()`` at import time, so the branch is settled by whichever
import lands first. pytest imports conftest.py before any test module, which
makes this the only hook early enough. Tests that ran on the non-NPU branch
would be exercising a different codebase than the runners do.
"""

from sglang.srt.sim.fake_npu import install_fake_npu

install_fake_npu()
