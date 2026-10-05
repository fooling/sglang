"""The page-31 flow, but on a REAL Scheduler built by Scheduler.__init__.

run_k3_sim.py drives SGLang's scheduling methods on a hand-built stub. This
one goes one level further: it constructs a real ``Scheduler`` -- ZMQ port
args, a single-rank gloo TP group, the real ModelConfig, init_model_worker,
the memory pools (including the hybrid request pool K3's linear attention
needs) -- with the sim backend installed, and then runs the same workload.

Nothing about the model is real: no checkpoint, no attention backend, no
kernels. Everything about the control plane is.

    cd ~/repo/sglang && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 no_proxy='*' \
      perl -e 'alarm 600; exec @ARGV' python/.venv/bin/python \
      python/sglang/srt/sim/run_k3_server_sim.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

# Platform probes must be neutralised before ServerArgs publishes: resolution
# asks for host memory capacity, which shells out to lscpu (Linux-only).
import sglang.srt.utils as _u
import sglang.srt.utils.common as _c
import sglang.srt.utils.numa_utils as _nu

for _m in (_c, _u, _nu):
    for _n, _f in (("parse_lscpu_topology", lambda *a, **k: []),
                   ("get_physical_cpus_by_numa", lambda *a, **k: {0: [0]}),
                   ("get_cpu_ids_by_node", lambda *a, **k: ["0"])):
        if hasattr(_m, _n):
            setattr(_m, _n, _f)

import torch

from sglang.srt.sim import register
from sglang.srt.sim.run_k3_sim import (
    EOS_SCRIPT,
    WORKLOAD,
    banner,
    build_k3_model_config,
    make_k3_req,
    scripted_token,
)

LAST_RUN: dict = {}


def build_real_scheduler():
    """A genuine Scheduler object, sim backend installed, no weights."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import (
        PortArgs,
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    register.install()
    model_config, cfg_dir = build_k3_model_config()
    server_args = ServerArgs(
        model_path=str(cfg_dir),
        tokenizer_path=str(cfg_dir),
        device="cpu",
        attention_backend="torch_native",  # keeps KV writes off the Triton kernel
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,  # overlap needs device streams
        tp_size=1,
    )
    set_global_server_args_for_scheduler(server_args)
    port_args = PortArgs.init_new(server_args)
    scheduler = Scheduler(
        server_args=server_args, port_args=port_args, gpu_id=0, tp_rank=0,
        moe_ep_rank=0, pp_rank=0, attn_cp_rank=0, moe_dp_rank=0, dp_rank=None,
    )
    scheduler.tp_worker.model_runner.token_script = scripted_token

    # With ngram speculative decoding off, __init__ leaves this None, but
    # get_next_batch_to_run (scheduler.py:3332) calls through unconditionally.
    # Hand it a pass-through rather than skipping the call.
    if getattr(scheduler, "ngram_embedding_manager", None) is None:
        from types import SimpleNamespace

        scheduler.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, *a, **k: batch
        )
    return scheduler, cfg_dir


def main(prebuilt=None) -> int:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    banner("step 1: build a real Scheduler (mock backend, no weights)")
    # One Scheduler per process: it holds a gloo group and ZMQ sockets, so a
    # caller that already built one passes it in rather than building a second.
    sched, cfg_dir = prebuilt if prebuilt is not None else build_real_scheduler()
    mr = sched.tp_worker.model_runner
    print(f"  Scheduler built: {type(sched).__name__}")
    print(f"  worker={type(sched.tp_worker).__name__} runner={type(mr).__name__}")
    print(f"  config dir holds: {sorted(p.name for p in cfg_dir.iterdir())}")
    print(f"  req pool: {type(sched.req_to_token_pool).__name__}"
          f" (hybrid = K3's linear attention needs a state pool)")
    print(f"  kv allocator: {type(sched.token_to_kv_pool_allocator).__name__}"
          f" size={sched.token_to_kv_pool_allocator.available_size()}")
    print(f"  torch.cuda.is_available()={torch.cuda.is_available()}")

    banner("step 2: workload (arrives in waves)")
    reqs = {rid: make_k3_req(rid, plen, mnt) for rid, plen, mnt, _ in WORKLOAD}
    for rid, plen, mnt, arrive in WORKLOAD:
        eos = f", EOS at output #{EOS_SCRIPT[rid]}" if rid in EOS_SCRIPT else ""
        print(f"  {rid}: prompt={plen} max_new={mnt} arrives before step {arrive}{eos}")
    sched.waiting_queue = []

    banner("step 3: scheduler loop on the real Scheduler")
    running_batch = ScheduleBatch(
        reqs=[], batch_is_full=False, device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    last_batch, steps, idle = None, 0, 0
    finished: dict[str, str] = {}
    modes: list[str] = []
    print(f"  {'step':>4} {'mode':<8} {'bs':>3} {'kvfree':>7} {'wait':>5}  "
          f"{'batch':<30} events")
    while steps < 60:
        steps += 1
        newly = [rid for rid, _p, _m, a in WORKLOAD if a == steps]
        if newly:
            sched.waiting_queue.extend(reqs[r] for r in newly)

        plan = Scheduler.get_next_batch_to_run(
            sched, running_batch=running_batch, last_batch=last_batch
        )
        running_batch, batch = plan.running_batch, plan.batch_to_run
        kvfree = sched.token_to_kv_pool_allocator.available_size()
        if batch is None:
            print(f"  {steps:>4} {'(idle)':<8} {'-':>3} {kvfree:>7} "
                  f"{len(sched.waiting_queue):>5}  {'--':<30} "
                  f"{'arrived ' + str(newly) if newly else ''}")
            idle += 1
            if not sched.waiting_queue or (
                idle >= 3 and not [a for _r, _p, _m, a in WORKLOAD if a > steps]
            ):
                break
            last_batch = None
            continue
        idle = 0
        mode = "prefill" if batch.forward_mode.is_extend() else "decode"
        modes.append(mode)
        in_batch = list(batch.reqs)

        result = Scheduler.run_batch(sched, batch)
        Scheduler.process_batch_result(sched, batch, result)

        events = []
        if newly:
            events.append(f"arrived {newly}")
        for req in in_batch:
            if req.finished() and req.rid not in finished:
                finished[req.rid] = type(req.finished_reason).__name__
                events.append(f"{req.rid} {finished[req.rid].replace('FINISH_', '')}")
        print(f"  {steps:>4} {mode:<8} {batch.batch_size():>3} "
              f"{sched.token_to_kv_pool_allocator.available_size():>7} "
              f"{len(sched.waiting_queue):>5}  "
              f"{str([r.rid for r in batch.reqs]):<30} {'; '.join(events)}")
        last_batch = batch
        if len(finished) == len(WORKLOAD):
            break

    banner("step 4: result")
    by_reason: dict = {}
    for rid, reason in finished.items():
        by_reason.setdefault(reason, []).append(rid)
    print(f"  steps: {steps}  prefills: {modes.count('prefill')}  "
          f"decodes: {modes.count('decode')}")
    print(f"  finished: {len(finished)} / {len(WORKLOAD)}")
    for reason, rids in sorted(by_reason.items()):
        print(f"    {reason}: {sorted(rids)}")
    for rid, _plen, mnt, _a in WORKLOAD:
        print(f"    {rid}: output_len={len(reqs[rid].output_ids)} (max_new={mnt}) "
              f"reason={finished.get(rid, 'NOT FINISHED')}")
    kv_free = sched.token_to_kv_pool_allocator.available_size()
    evictable = sched.tree_cache.evictable_size()
    print(f"  kv at end: free={kv_free} + radix evictable={evictable} "
          f"= {kv_free + evictable}")
    print(f"  waiting_queue left: {len(sched.waiting_queue)}")

    ok = len(finished) == len(WORKLOAD)
    print(f"\n  ALL REQUESTS FINISHED ON A REAL SCHEDULER: {ok}")
    LAST_RUN.clear()
    LAST_RUN.update(
        scheduler_cls=type(sched).__name__,
        worker_cls=type(sched.tp_worker).__name__,
        req_pool_cls=type(sched.req_to_token_pool).__name__,
        steps=steps, modes=modes, finished=finished, by_reason=by_reason,
        reqs=reqs, workload=WORKLOAD, kv_free_end=kv_free,
        radix_evictable=evictable, waiting_left=len(sched.waiting_queue),
    )
    return 0 if ok else 2


def run(prebuilt=None) -> dict:
    main(prebuilt)
    return dict(LAST_RUN)


if __name__ == "__main__":
    raise SystemExit(main())
