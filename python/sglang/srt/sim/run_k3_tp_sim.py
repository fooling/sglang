"""The same workload, but on a tensor-parallel group of real Schedulers.

One process per TP rank, exactly as a real deployment does it: each builds its
own Scheduler with the sim backend installed, they form a gloo TP group, and
every rank runs event_loop_normal. Only rank 0 owns the tokenizer IPC and
scripts the arrivals; the other ranks receive the requests the way they
normally do, through the scheduler's own _broadcast_reqs_across_ranks.

What this is for: "TP / EP / PD are not covered" was listed as a limitation of
the interception design. It is not one. Construction and the whole run work on
a multi-rank group with nothing but the mock backend.

One real platform constraint, and it is not ours: SGLang's node-locality check
(in_the_same_node_as, reached from the shm broadcaster) deadlocks on this macOS
box, which has no /dev/shm. Verified with no sim code in the picture at all --
plain init_distributed_environment + initialize_model_parallel on two gloo
ranks hangs in the same line. SGLang's own switch avoids it:

    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0

    cd ~/repo/sglang && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 no_proxy='*' \
      SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
      perl -e 'alarm 600; exec @ARGV' python/.venv/bin/python \
      python/sglang/srt/sim/run_k3_tp_sim.py 2
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Platform probes must be neutralised before ServerArgs publishes (see
# run_k3_server_sim for why).
import sglang.srt.utils as _u
import sglang.srt.utils.common as _c
import sglang.srt.utils.numa_utils as _nu

for _m in (_c, _u, _nu):
    for _n, _f in (("parse_lscpu_topology", lambda *a, **k: []),
                   ("get_physical_cpus_by_numa", lambda *a, **k: {0: [0]}),
                   ("get_cpu_ids_by_node", lambda *a, **k: ["0"])):
        if hasattr(_m, _n):
            setattr(_m, _n, _f)

DIST_INIT = "tcp://127.0.0.1:29701"


def _server_args(cfg_dir: str, tp_size: int, ep_size: int = 1):
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(
        model_path=cfg_dir,
        tokenizer_path=cfg_dir,
        device="cpu",
        attention_backend="torch_native",
        skip_tokenizer_init=True,
        disable_overlap_schedule=True,
        max_total_tokens=256,
        max_mamba_cache_size=64,
        tp_size=tp_size,
        ep_size=ep_size,
    )


def run_rank(rank: int, tp_size: int, cfg_dir: str, port_args_path: str,
             ep_size: int = 1) -> int:
    """One TP rank: build a Scheduler, run SGLang's loop, report."""
    os.environ["SIM_DIST_INIT"] = DIST_INIT

    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import set_global_server_args_for_scheduler
    from sglang.srt.sim import register
    from sglang.srt.sim.run_k3_server_sim import (
        SimDriver,
        open_input_socket,
        open_output_socket,
    )
    from sglang.srt.sim.run_k3_sim import WORKLOAD, scripted_token

    register.install()
    server_args = _server_args(cfg_dir, tp_size, ep_size)
    set_global_server_args_for_scheduler(server_args)
    port_args = pickle.loads(Path(port_args_path).read_bytes())

    scheduler = Scheduler(
        server_args=server_args, port_args=port_args, gpu_id=rank, tp_rank=rank,
        moe_ep_rank=0, pp_rank=0, attn_cp_rank=0, moe_dp_rank=0, dp_rank=None,
    )
    if getattr(scheduler, "ngram_embedding_manager", None) is None:
        from types import SimpleNamespace

        scheduler.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, *a, **k: batch
        )
    scheduler.sim_input_ipc_name = str(port_args.scheduler_input_ipc_name)
    scheduler.sim_output_ipc_name = str(port_args.tokenizer_ipc_name)
    # Same scripted tokens on every rank: in TP the ranks must sample the same
    # thing, so a per-rank script would desync the schedule.
    scheduler.tp_worker.model_runner.token_script = scripted_token

    from sglang.srt.distributed import parallel_state as ps_mod

    tp = ps_mod.get_tp_group()
    ep_world = None
    for getter in ("get_moe_ep_group", "get_ep_group"):
        fn = getattr(ps_mod, getter, None)
        if fn is None:
            continue
        try:
            ep_world = fn().world_size
            break
        except Exception:
            continue
    print(f"[rank{rank}] tp world={tp.world_size} rank_in_group={tp.rank_in_group}"
          f" ep world={ep_world}"
          f" kv={scheduler.token_to_kv_pool_allocator.available_size()}"
          f" reqpool={scheduler.req_to_token_pool.size}", flush=True)

    # Only rank 0 talks to the tokenizer. The others get the requests through
    # the scheduler's own cross-rank broadcast.
    ctx_in = push = ctx_out = pull = None
    if rank == 0:
        ctx_in, push = open_input_socket(scheduler)
        ctx_out, pull = open_output_socket(scheduler)

    driver = SimDriver(scheduler, push, pull)
    assert scheduler.scripted_scheduler_hook is None
    scheduler.scripted_scheduler_hook = driver
    import dataclasses

    scheduler.request_receiver = dataclasses.replace(
        scheduler.request_receiver, scripted_scheduler_hook=driver
    )
    scheduler.gracefully_exit = False

    Scheduler.event_loop_normal(scheduler)
    driver._drain_outputs()
    driver._attribute_finishes()

    schedule = [(r["mode"], r["bs"], tuple(r["rids"])) for r in driver.rows]
    finished = dict(sorted(driver.finished_from_reqs.items()))
    print(f"[rank{rank}] forward batches={len(driver.rows)}"
          f" finished={len(finished)}/{len(WORKLOAD)}"
          f" stop={driver.stop_reason}", flush=True)
    print(f"[rank{rank}] SCHEDULE {json.dumps(schedule)}", flush=True)
    print(f"[rank{rank}] FINISHED {json.dumps(finished)}", flush=True)

    if push is not None:
        push.close(); ctx_in.term()
        pull.close(); ctx_out.term()
    return 0 if len(finished) == len(WORKLOAD) else 2


def main(tp_size: int = 2, ep_size: int = 1) -> int:
    from sglang.srt.server_args import (
        PortArgs,
        set_global_server_args_for_scheduler,
    )
    from sglang.srt.sim.run_k3_sim import banner, build_k3_model_config

    banner(f"step 1: one process per rank (tp_size={tp_size}, ep_size={ep_size})")
    _mc, cfg_dir = build_k3_model_config()
    server_args = _server_args(str(cfg_dir), tp_size, ep_size)
    set_global_server_args_for_scheduler(server_args)
    port_args = PortArgs.init_new(server_args)
    pa_path = tempfile.mktemp(suffix=".portargs")
    Path(pa_path).write_bytes(pickle.dumps(port_args))
    print(f"  config dir: {cfg_dir}")
    print(f"  gloo rendezvous: {DIST_INIT}")

    banner("step 2: every rank runs SGLang's own event_loop_normal")
    env = dict(os.environ)
    # SGLang's own switch; without it its node-locality check deadlocks on a
    # box with no /dev/shm, with or without any sim code (see module docstring).
    env.setdefault("SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "0")
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "--rank", str(r), str(tp_size),
             str(cfg_dir), pa_path, str(ep_size)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True,
        )
        for r in range(tp_size)
    ]
    outs = []
    for p in procs:
        outs.append(p.communicate()[0])
    rcs = [p.returncode for p in procs]

    banner("step 3: result")
    schedules, finishes = {}, {}
    for r, out in enumerate(outs):
        for line in (out or "").splitlines():
            if line.startswith(f"[rank{r}]"):
                print("  " + line)
            if " SCHEDULE " in line:
                schedules[r] = json.loads(line.split(" SCHEDULE ", 1)[1])
            if " FINISHED " in line:
                finishes[r] = json.loads(line.split(" FINISHED ", 1)[1])

    ok = all(rc == 0 for rc in rcs) and len(schedules) == tp_size
    if ok:
        # Every rank must have decided the same schedule: in TP the ranks run
        # the same batching on their own copy of the state, so a divergence
        # here is a real bug, not a cosmetic difference.
        ok = all(schedules[r] == schedules[0] for r in schedules)
        print(f"  all ranks agree on the schedule: {ok}"
              f" ({len(schedules[0])} forward batches)")
        same_finish = all(finishes[r] == finishes[0] for r in finishes)
        print(f"  all ranks agree on the finishes: {same_finish}"
              f" {sorted(finishes[0])}")
        ok = ok and same_finish
    print(f"  rank exit codes: {rcs}")
    print(f"\n  TP={tp_size} EP={ep_size} RAN THE WHOLE WORKLOAD: {ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rank":
        raise SystemExit(
            run_rank(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5],
                     int(sys.argv[6]) if len(sys.argv) > 6 else 1)
        )
    raise SystemExit(main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 2,
        int(sys.argv[2]) if len(sys.argv) > 2 else 1,
    ))
