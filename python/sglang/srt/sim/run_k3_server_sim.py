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

import dataclasses
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

# How long to wait on the output socket once the engine has gone idle, and
# how many such waits before calling a missing result missing.
OUTPUT_POLL_MS = 200
IDLE_ROUNDS_BEFORE_GIVING_UP = 25


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
    # With skip_tokenizer_init the scheduler sends results straight to
    # tokenizer_ipc_name (ipc_channels:58), not to the detokenizer.
    scheduler.sim_output_ipc_name = str(port_args.tokenizer_ipc_name)
    return scheduler, cfg_dir


def open_output_socket(scheduler):
    """Bind the side the tokenizer / detokenizer process binds.

    The scheduler's output socket is a zmq.PUSH that CONNECTS (ipc_channels:58),
    so without a bound peer the results go nowhere. Binding it here means
    process_batch_result's stream really lands somewhere and can be read back.
    """
    import zmq

    from sglang.srt.server_args import get_global_server_args

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.bind(scheduler.sim_output_ipc_name)
    return ctx, pull


class SimDriver:
    """Scripts the arrivals and watches each iteration from inside the loop.

    Both methods are SGLang's own scheduler-hook interface: run_batch calls
    on_run_batch (scheduler.py:3874) and recv_requests calls step(). Using them
    means event_loop_normal runs for real -- this object only sends requests,
    reads the outputs the engine publishes, and decides when to stop.
    """

    def __init__(self, scheduler, push, pull, max_iters: int = 4000):
        self.sched = scheduler
        self.push = push
        self.pull = pull
        self.max_iters = max_iters
        self.iter = 0
        self.rows: list[dict] = []
        self.finished: dict[str, str] = {}
        self.completion: dict[str, int] = {}
        self.arrived: dict[int, list] = {}
        self.sent: set = set()
        # Reqs the engine put in a batch, kept so finishes can be attributed to
        # the batch that produced them. The authoritative ledger is still what
        # the engine published on the output socket; the two are reconciled at
        # the end.
        self.seen: dict = {}
        self.finished_from_reqs: dict[str, str] = {}
        self.idle_rounds = 0
        self.stop_reason = "not stopped"

    # ---- SGLang calls this at the top of recv_requests ----
    def step(self) -> None:
        self.iter += 1
        self._drain_outputs()
        self._attribute_finishes()

        newly = [rid for rid, _p, _m, a in WORKLOAD if a == self.iter]
        if newly:
            deliver_async(self.push, newly)
            self.arrived[self.iter] = newly
            self.sent.update(newly)

        # Yield so zmq's I/O thread can actually move the messages. Without
        # this the loop spins through every iteration in a few milliseconds
        # and the outputs arrive late or not at all -- a real deployment has a
        # tokenizer process on the other end polling its own socket.
        time.sleep(0.001)

        all_sent = len(self.sent) == len(WORKLOAD)
        if all_sent and self.sched.is_fully_idle():
            # Nothing left to launch: block on the output socket instead of
            # spinning, so a result that is still in flight is not mistaken
            # for a result that never came.
            if len(self.finished) < len(WORKLOAD):
                self.pull.poll(timeout=OUTPUT_POLL_MS)
                self._drain_outputs()
            self.idle_rounds += 1
        else:
            self.idle_rounds = 0

        done = len(self.finished) == len(WORKLOAD)
        if done or self.idle_rounds > IDLE_ROUNDS_BEFORE_GIVING_UP or (
            self.iter > self.max_iters
        ):
            # the loop checks this at the top of the next iteration
            self.stop_reason = (
                "all finished" if done
                else "idle with outputs missing"
                if self.idle_rounds > IDLE_ROUNDS_BEFORE_GIVING_UP
                else "iteration cap"
            )
            self.sched.gracefully_exit = True

    # ---- SGLang calls this inside run_batch ----
    def on_run_batch(self, batch) -> None:
        for req in batch.reqs:
            self.seen[req.rid] = req
        self.rows.append(
            dict(
                forward_iter=batch.forward_iter,
                mode="prefill" if batch.forward_mode.is_extend() else "decode",
                bs=batch.batch_size(),
                rids=[r.rid for r in batch.reqs],
                wait=len(self.sched.waiting_queue),
                # free pages at the moment this batch is launched
                kvfree=self.sched.token_to_kv_pool_allocator.available_size(),
                arrived=self.arrived.get(self.iter, []),
                events=[],
            )
        )

    def _attribute_finishes(self) -> None:
        """Pin each finish to the batch that produced it.

        step() runs right after the previous iteration's process_batch_result,
        so rows[-1] is the batch whose forward produced this finish.
        """
        for rid, req in self.seen.items():
            if rid in self.finished_from_reqs or not req.finished():
                continue
            name = type(req.finished_reason).__name__
            self.finished_from_reqs[rid] = name
            if self.rows:
                self.rows[-1]["events"].append(
                    f"{rid} {name.replace('FINISH_', '')}"
                )

    # ---- reading what the engine published ----
    def _drain_outputs(self) -> None:
        import zmq

        from sglang.srt.managers.io_struct import sock_recv

        while True:
            try:
                out = sock_recv(self.pull, zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            rids = getattr(out, "rids", None)
            if not rids:
                continue
            reasons = getattr(out, "finished_reasons", [None] * len(rids))
            done = getattr(out, "completion_tokens", [None] * len(rids))
            for i, rid in enumerate(rids):
                if done[i] is not None:
                    self.completion[rid] = done[i]
                reason = reasons[i] if i < len(reasons) else None
                if reason and rid not in self.finished:
                    name = reason.get("type") if isinstance(reason, dict) else str(reason)
                    self.finished[rid] = str(name)


def deliver_async(push, rids) -> None:
    """Send without waiting: the engine's own recv_requests picks them up."""
    from sglang.srt.managers.io_struct import sock_send

    spec = {rid: (plen, mnt) for rid, plen, mnt, _a in WORKLOAD}
    for rid in rids:
        plen, mnt = spec[rid]
        sock_send(push, tokenized_request(rid, plen, mnt))


def main(prebuilt=None) -> int:
    from sglang.srt.managers.scheduler import Scheduler

    banner("step 1: build a real Scheduler (mock backend, no weights)")
    # One Scheduler per process: it holds a gloo group and ZMQ sockets, so a
    # caller that already built one passes it in rather than building a second.
    sched, cfg_dir = prebuilt if prebuilt is not None else build_real_scheduler()
    mr = sched.tp_worker.model_runner
    print(f"  Scheduler built: {type(sched).__name__}")
    print(f"  worker={type(sched.tp_worker).__name__} runner={type(mr).__name__}")
    print(f"  config dir holds: {sorted(p.name for p in cfg_dir.iterdir())}")
    print(f"  req pool: {type(sched.req_to_token_pool).__name__}"
          f" size={sched.req_to_token_pool.size}"
          f" (hybrid: K3's linear attention needs a state pool)")
    print(f"  kv allocator: {type(sched.token_to_kv_pool_allocator).__name__}"
          f" size={sched.token_to_kv_pool_allocator.available_size()}")
    print(f"  max_running_requests: {sched.max_running_requests}")
    print(f"  torch.cuda.is_available()={torch.cuda.is_available()}")

    banner("step 2: wire both ends of the real IPC")
    ctx_in, push = open_input_socket(sched)
    ctx_out, pull = open_output_socket(sched)
    print(f"  input  <- {sched.sim_input_ipc_name}")
    print(f"  output -> {sched.sim_output_ipc_name}")
    for rid, plen, mnt, arrive in WORKLOAD:
        eos = f", EOS at output #{EOS_SCRIPT[rid]}" if rid in EOS_SCRIPT else ""
        print(f"  {rid}: prompt={plen} max_new={mnt} arrives at iter {arrive}{eos}")

    banner("step 3: run SGLang's own event_loop_normal")
    driver = SimDriver(sched, push, pull)
    # SGLang's own scheduler-hook attribute: run_batch calls on_run_batch and
    # recv_requests calls step(). The loop below is SGLang's, start to finish.
    assert sched.scripted_scheduler_hook is None
    sched.scripted_scheduler_hook = driver
    # The receiver is a frozen dataclass that captured the hook at __init__
    # time. Rebuild it with dataclasses.replace -- still SGLang's own class
    # and every other field untouched, which is exactly what
    # init_request_receiver would have handed it had the hook existed then.
    sched.request_receiver = dataclasses.replace(
        sched.request_receiver, scripted_scheduler_hook=driver
    )
    sched.gracefully_exit = False

    Scheduler.event_loop_normal(sched)
    driver._drain_outputs()  # whatever the last batch published
    driver._attribute_finishes()
    # The two records must agree: what the Reqs say and what the engine
    # published on the socket. A mismatch means the output path dropped a
    # result, which would otherwise pass unnoticed.
    assert set(driver.finished) == set(driver.finished_from_reqs), (
        f"output stream says {sorted(driver.finished)} but the Reqs say "
        f"{sorted(driver.finished_from_reqs)}"
    )

    print(f"  {'fwd':>4} {'mode':<8} {'bs':>3} {'kvfree':>7} {'wait':>5}  "
          f"{'batch':<30} events")
    for row in driver.rows:
        ev = []
        if row["arrived"]:
            ev.append(f"arrived {row['arrived']}")
        ev += row["events"]
        kv = row.get("kvfree")
        print(f"  {row['forward_iter']:>4} {row['mode']:<8} {row['bs']:>3} "
              f"{(kv if kv is not None else '-'):>7} {row['wait']:>5}  "
              f"{str(row['rids']):<30} {'; '.join(ev)}")

    banner("step 4: result, read off what the engine published")
    modes = [r["mode"] for r in driver.rows]
    finished = driver.finished
    by_reason: dict = {}
    for rid, reason in finished.items():
        by_reason.setdefault(reason, []).append(rid)
    print(f"  loop iterations: {driver.iter}  forward batches: {len(driver.rows)}"
          f"  prefills: {modes.count('prefill')}  decodes: {modes.count('decode')}")
    print(f"  finished: {len(finished)} / {len(WORKLOAD)}")
    for reason, rids in sorted(by_reason.items()):
        print(f"    {reason}: {sorted(rids)}")
    for rid, _plen, mnt, _a in WORKLOAD:
        print(f"    {rid}: completion_tokens={driver.completion.get(rid)} "
              f"(max_new={mnt}) reason={finished.get(rid, 'NOT FINISHED')}")
    kv_free = sched.token_to_kv_pool_allocator.available_size()
    evictable = sched.tree_cache.evictable_size()
    print(f"  kv at end: free={kv_free} + radix evictable={evictable} "
          f"= {kv_free + evictable}")
    print(f"  waiting_queue left: {len(sched.waiting_queue)}")
    print(f"  loop exited by: {driver.stop_reason} (gracefully_exit={sched.gracefully_exit})")

    push.close(); ctx_in.term()
    pull.close(); ctx_out.term()

    ok = len(finished) == len(WORKLOAD)
    print(f"\n  ALL REQUESTS FINISHED ON A REAL SCHEDULER LOOP: {ok}")
    LAST_RUN.clear()
    LAST_RUN.update(
        scheduler_cls=type(sched).__name__,
        worker_cls=type(sched.tp_worker).__name__,
        req_pool_cls=type(sched.req_to_token_pool).__name__,
        loop_iters=driver.iter, rows=driver.rows, modes=modes,
        waits=[r["wait"] for r in driver.rows],
        finished=finished, by_reason=by_reason, completion=driver.completion,
        workload=WORKLOAD, kv_free_end=kv_free, radix_evictable=evictable,
        waiting_left=len(sched.waiting_queue),
        max_running_requests=sched.max_running_requests,
        req_pool_size=sched.req_to_token_pool.size,
        outputs_were_read=bool(driver.completion),
        finished_from_reqs=driver.finished_from_reqs,
        req_cls_module=type(next(iter(driver.seen.values()))).__module__,
        stop_reason=driver.stop_reason,
    )
    return 0 if ok else 2


def run(prebuilt=None) -> dict:
    main(prebuilt)
    return dict(LAST_RUN)


if __name__ == "__main__":
    raise SystemExit(main())
