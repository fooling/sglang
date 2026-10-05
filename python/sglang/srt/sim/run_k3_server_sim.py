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

import time
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
    FILLER_TOKEN,
    WORKLOAD,
    banner,
    build_k3_model_config,
    scripted_token,
)

LAST_RUN: dict = {}


def open_input_socket(scheduler):
    """Stand where the tokenizer manager stands: bind the scheduler's input.

    Not a shim. The scheduler already connected a real zmq.PULL to this
    endpoint in __init__; this binds the PUSH side, so recv_requests ->
    process_input_requests -> handle_generate_request all run as SGLang wrote
    them, and the Req objects are built by the scheduler, not by us.
    """
    import zmq

    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.bind(scheduler.sim_input_ipc_name)
    return ctx, push


def tokenized_request(rid: str, prompt_len: int, max_new: int):
    """What the tokenizer manager would hand the scheduler.

    normalize() is the tokenizer manager's job (it is what fills stop_strs,
    which update_finish_state then reads); the sim does it here because that
    process is out of scope, not because anything is being faked.
    """
    from array import array

    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.sampling.sampling_params import SamplingParams

    sp = SamplingParams(max_new_tokens=max_new)
    sp.normalize(None)
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        # typecode "q", same as the tokenizer manager (:1346). Req.output_ids
        # is array("q") and _refresh_fill_ids concatenates the two, so a
        # different typecode raises on the first prefill.
        input_ids=array("q", [FILLER_TOKEN] * prompt_len),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=sp,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


def deliver(scheduler, push, rids):
    """Send, then let SGLang receive and dispatch. Returns the new Reqs."""
    from sglang.srt.managers.io_struct import sock_send
    from sglang.srt.managers.scheduler import Scheduler

    spec = {rid: (plen, mnt) for rid, plen, mnt, _a in WORKLOAD}
    for rid in rids:
        plen, mnt = spec[rid]
        sock_send(push, tokenized_request(rid, plen, mnt))

    before = {id(r) for r in scheduler.waiting_queue}
    got: list = []
    for _ in range(80):  # the socket is non-blocking; give delivery a moment
        recv = scheduler.request_receiver.recv_requests()
        if recv:
            Scheduler.process_input_requests(scheduler, recv)
            got += [r for r in scheduler.waiting_queue if id(r) not in before]
            if len(got) >= len(rids):
                break
        time.sleep(0.02)
    assert len(got) == len(rids), (
        f"sent {rids} but the scheduler queued {[r.rid for r in got]}"
    )
    return got


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
        # Tight on purpose: the six prompts want 456 pages, so admission
        # has to refuse and defer. A user cap SGLang already
        # has -- _apply_token_constraints applies it to whatever the
        # memory profile reports, and _derive_pool_sizes still runs.
        max_total_tokens=144,
        # Required, not a sim convenience: with a hybrid model and this
        # left at None, SGLang's own resolve_max_num_reqs divides it by
        # the mamba ratio (kv_cache_configurator.py:2004) and raises.
        # A real K3 deployment has to pass it too.
        max_mamba_cache_size=64,
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
    # The scheduler CONNECTS a zmq.PULL here; the binder is normally the
    # tokenizer manager process. The driver below binds a PUSH and takes
    # that position, so arrivals go through the real socket.
    scheduler.sim_input_ipc_name = str(port_args.scheduler_input_ipc_name)
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

    banner("step 2: workload (arrives in waves, through the real socket)")
    reqs: dict = {}
    for rid, plen, mnt, arrive in WORKLOAD:
        eos = f", EOS at output #{EOS_SCRIPT[rid]}" if rid in EOS_SCRIPT else ""
        print(f"  {rid}: prompt={plen} max_new={mnt} arrives before step {arrive}{eos}")
    ctx, push = open_input_socket(sched)
    print(f"  driver bound {sched.sim_input_ipc_name}"
          f" (the tokenizer manager's position)")
    sched.waiting_queue = []

    banner("step 3: scheduler loop on the real Scheduler")
    running_batch = ScheduleBatch(
        reqs=[], batch_is_full=False, device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    last_batch, steps, idle = None, 0, 0
    finished: dict[str, str] = {}
    modes: list[str] = []
    # per-step queue depth: how the page can claim admission deferred anyone
    waits: list[int] = []
    print(f"  {'step':>4} {'mode':<8} {'bs':>3} {'kvfree':>7} {'wait':>5}  "
          f"{'batch':<30} events")
    while steps < 60:
        steps += 1
        newly = [rid for rid, _p, _m, a in WORKLOAD if a == steps]
        if newly:
            # real socket -> recv_requests -> process_input_requests ->
            # handle_generate_request: the Req objects are SGLang's own
            for req in deliver(sched, push, newly):
                reqs[req.rid] = req

        plan = Scheduler.get_next_batch_to_run(
            sched, running_batch=running_batch, last_batch=last_batch
        )
        running_batch, batch = plan.running_batch, plan.batch_to_run
        kvfree = sched.token_to_kv_pool_allocator.available_size()
        if batch is None:
            print(f"  {steps:>4} {'(idle)':<8} {'-':>3} {kvfree:>7} "
                  f"{len(sched.waiting_queue):>5}  {'--':<30} "
                  f"{'arrived ' + str(newly) if newly else ''}")
            # event_loop_normal calls this on the no-batch branch (:3825);
            # calling it keeps every step of the loop body covered.
            Scheduler.on_idle(sched)
            idle += 1
            if not sched.waiting_queue or (
                idle >= 3 and not [a for _r, _p, _m, a in WORKLOAD if a > steps]
            ):
                break
            last_batch = None
            continue
        idle = 0
        waits.append(len(sched.waiting_queue))
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

    push.close()
    ctx.term()

    ok = len(finished) == len(WORKLOAD)
    print(f"\n  ALL REQUESTS FINISHED ON A REAL SCHEDULER: {ok}")
    LAST_RUN.clear()
    LAST_RUN.update(
        scheduler_cls=type(sched).__name__,
        worker_cls=type(sched.tp_worker).__name__,
        req_pool_cls=type(sched.req_to_token_pool).__name__,
        steps=steps, modes=modes, waits=waits, finished=finished,
        by_reason=by_reason,
        reqs=reqs, workload=WORKLOAD, kv_free_end=kv_free,
        radix_evictable=evictable, waiting_left=len(sched.waiting_queue),
        max_running_requests=sched.max_running_requests,
        req_pool_size=sched.req_to_token_pool.size,
        req_cls_module=type(next(iter(reqs.values()))).__module__,
    )
    return 0 if ok else 2


def run(prebuilt=None) -> dict:
    main(prebuilt)
    return dict(LAST_RUN)


if __name__ == "__main__":
    raise SystemExit(main())
