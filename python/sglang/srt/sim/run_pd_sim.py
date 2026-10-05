"""Both PD-disaggregation roles, constructed on the sim backend.

"PD is not covered" was listed as a limitation. Its control plane is not: with
disaggregation_mode=prefill / decode and the ascend transfer backend -- the one
the sim shim replaces -- SGLang builds its own PD queues
(PrefillBootstrapQueue, DecodePreallocQueue, DecodeTransferQueue) on a
Scheduler with no weights and no device.

Where it does stop is the data plane, and that stop is honest: SimKVCache
allocates KV *indices* for real but has no tensors, so
get_contiguous_buf_infos returns nothing to register and a prefill/decode pair
has no bytes to move between them. PD's KV transfer is the one thing here that
needs real KV memory.

One role per process, because each one installs process-wide parallel state.

    cd ~/repo/sglang && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 no_proxy='*' \
      perl -e 'alarm 600; exec @ARGV' python/.venv/bin/python \
      python/sglang/srt/sim/run_pd_sim.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import sglang.srt.utils as _u
import sglang.srt.utils.common as _c
import sglang.srt.utils.numa_utils as _nu

for _m in (_c, _u, _nu):
    for _n, _f in (("parse_lscpu_topology", lambda *a, **k: []),
                   ("get_physical_cpus_by_numa", lambda *a, **k: {0: [0]}),
                   ("get_cpu_ids_by_node", lambda *a, **k: ["0"])):
        if hasattr(_m, _n):
            setattr(_m, _n, _f)

ROLES = ("prefill", "decode")
# What each role's own queues are expected to be, by SGLang's class names.
EXPECTED = {
    "prefill": {"disagg_prefill_bootstrap_queue": "PrefillBootstrapQueue"},
    "decode": {"disagg_decode_prealloc_queue": "DecodePreallocQueue",
               "disagg_decode_transfer_queue": "DecodeTransferQueue"},
}


def build_role(mode: str) -> int:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import (
        PortArgs,
        ServerArgs,
        set_global_server_args_for_scheduler,
    )
    from sglang.srt.sim import register
    from sglang.srt.sim.run_k3_sim import build_k3_model_config

    register.install()
    _mc, cfg_dir = build_k3_model_config()
    server_args = ServerArgs(
        model_path=str(cfg_dir), tokenizer_path=str(cfg_dir), device="cpu",
        attention_backend="torch_native", skip_tokenizer_init=True,
        disable_overlap_schedule=True, max_total_tokens=256,
        max_mamba_cache_size=64, tp_size=1,
        disaggregation_mode=mode,
        # The sim replaces the ascend engine; the default backend is mooncake,
        # whose library is absent here, so its engine comes back None. Choosing
        # ascend is what points PD at the face the sim actually occupies.
        disaggregation_transfer_backend="ascend",
    )
    set_global_server_args_for_scheduler(server_args)
    port_args = PortArgs.init_new(server_args)
    scheduler = Scheduler(
        server_args=server_args, port_args=port_args, gpu_id=0, tp_rank=0,
        moe_ep_rank=0, pp_rank=0, attn_cp_rank=0, moe_dp_rank=0, dp_rank=None,
    )

    print(f"[{mode}] disaggregation_mode={scheduler.disaggregation_mode}", flush=True)
    ok = True
    for attr, cls in EXPECTED[mode].items():
        got = type(getattr(scheduler, attr, None)).__name__
        ok = ok and got == cls
        print(f"[{mode}] {attr}: {got} (want {cls})", flush=True)

    # The data plane: nothing to register, because there are no KV tensors.
    kv = scheduler.token_to_kv_pool_allocator.get_kvcache()
    ptrs, lens, item_lens = kv.get_contiguous_buf_infos()
    print(f"[{mode}] kv buffers to register: {len(ptrs)} "
          f"(sim has indices, no tensors)", flush=True)
    ok = ok and (len(ptrs), len(lens), len(item_lens)) == (0, 0, 0)

    # The transfer engine PD picked must be the sim's, not a real one.
    import sglang.srt.disaggregation.ascend.conn as conn_mod

    engine_mod = conn_mod.AscendTransferEngine.__module__
    print(f"[{mode}] transfer engine class from: {engine_mod}", flush=True)
    ok = ok and engine_mod.startswith("sglang.srt.sim")

    print(f"[{mode}] ROLE OK {ok}", flush=True)
    # A PD role starts its bootstrap server on a background thread and keeps
    # it up -- that is what a real PD instance does, so the process will not
    # exit on its own. Report and leave hard.
    sys.stdout.flush()
    os._exit(0 if ok else 2)


def main() -> int:
    env = dict(os.environ)
    env.setdefault("SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "0")
    results = {}
    for mode in ROLES:
        role_env = dict(env)
        # Each role is its own single-rank process; let it take a free port
        # rather than inheriting one a previous run may still hold.
        role_env.pop("SIM_DIST_INIT", None)
        p = subprocess.run(
            [sys.executable, __file__, "--role", mode],
            env=role_env, capture_output=True, text=True, timeout=300,
        )
        for line in p.stdout.splitlines():
            if line.startswith("["):
                print("  " + line)
        if p.returncode != 0:
            tail = [ln for ln in p.stderr.splitlines()
                    if ln.strip() and not ln.startswith("W")][-6:]
            for ln in tail:
                print(f"  [{mode}] ! {ln[:150]}")
        results[mode] = p.returncode
    ok = all(rc == 0 for rc in results.values())
    print(f"\n  BOTH PD ROLES CONSTRUCTED ON THE SIM BACKEND: {ok} {results}")
    print("  (control plane up; KV transfer has no bytes to move -- see module docstring)")
    return 0 if ok else 2


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--role":
        raise SystemExit(build_role(sys.argv[2]))
    raise SystemExit(main())
