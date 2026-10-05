"""Probe: how far does a REAL Scheduler.__init__ get in the sim?

Not a test -- a measuring stick. Run it to see the current blocker:

    cd ~/repo/sglang && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 no_proxy='*' \
      perl -e 'alarm 420; exec @ARGV' python/.venv/bin/python \
      python/sglang/srt/sim/probe_scheduler_init.py

State at last run: clears __init__ end to end and prints SCHEDULER INIT OK.
The run itself lives in run_k3_server_sim.py; this file stays as the place a
future blocker gets measured.

What it took to get here, and why neither step is a mock shortcut:

1. The hybrid request pool. Kimi-K3 is a hybrid architecture --
   KimiK3DeltaAttention (models/kimi_k3.py:1584) is linear attention carrying
   recurrent state, and configs/kimi_linear.py carries KimiLinearCacheParams /
   kda_layers. build_kv_cache therefore asserts HybridReqToTokenPool, and the
   sim used to hand it a plain ReqToTokenPool. Fixed by building the pool from
   SGLang's own mambaish_config(model_config) (mock_worker._build_req_pool) --
   the sim only moves the device to CPU; the shape comes off the config.
   So K3's KV side is MLA pages PLUS a linear-attention state pool, not MLA
   alone.

2. Overlap off. run_batch's overlap branch (scheduler.py:3901) calls
   forward_stream.wait_stream, and there is no device stream here. Turned off
   through the server arg SGLang already has (disable_overlap_schedule), not by
   patching run_batch -- the test suite asserts run_batch stays the module's own
   function.
"""
import warnings, traceback; warnings.filterwarnings('ignore')
# 平台探测：macOS 没有 lscpu。与 get_available_gpu_memory 同类，按 backend 处理。
import sglang.srt.utils.common as _c
import sglang.srt.utils as _u
import sglang.srt.utils.numa_utils as _nu
_fake_cpu_ids = lambda *a, **k: ["0"]
for _m in (_c, _u, _nu):
    for _n, _f in (("parse_lscpu_topology", lambda *a, **k: []),
                   ("get_physical_cpus_by_numa", lambda *a, **k: {0: [0]}),
                   ("get_cpu_ids_by_node", _fake_cpu_ids)):
        if hasattr(_m, _n):
            setattr(_m, _n, _f)
from sglang.srt.sim import register, run_k3_sim as K
register.install()
mc, cfg_dir = K.build_k3_model_config()
import sglang.srt.sim.run_smoke as RS
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
sa = ServerArgs(model_path=str(cfg_dir), device="cpu",
                attention_backend="torch_native", tokenizer_path=str(cfg_dir),
                skip_tokenizer_init=True, disable_cuda_graph=True, tp_size=1)
set_global_server_args_for_scheduler(sa)
print('model_path ->', sa.model_path)
from sglang.srt.server_args import PortArgs
from sglang.srt.managers.scheduler import Scheduler
try:
    pa = PortArgs.init_new(sa)
except Exception as e:
    print('PortArgs FAIL:', type(e).__name__, str(e)[:120]); raise SystemExit(1)
print('PortArgs OK')
try:
    s = Scheduler(server_args=sa, port_args=pa, gpu_id=0, tp_rank=0,
                  moe_ep_rank=0, pp_rank=0, attn_cp_rank=0, moe_dp_rank=0, dp_rank=None)
    print('SCHEDULER INIT OK ->', type(s).__name__)
except Exception:
    tb = traceback.format_exc().strip().splitlines()
    print('INIT FAIL:', tb[-1])
    for l in tb[-8:-1]: print('   ', l.strip())
