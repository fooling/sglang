"""Smoke harness for the "仿真接入点" prototype (feat/sim-interception).

Does NOT start any server / HTTP endpoint. It constructs the real SGLang
control-plane objects directly in-process -- a ``Scheduler.__new__(Scheduler)``
stub carrying the real ``SchedulePolicy`` / ``PrefillAdder`` / ``ScheduleBatch``
/ ``RadixCache`` / ``TokenToKVPoolAllocator`` / ``ReqToTokenPool`` objects on
CPU torch, with the four interception points (worker/model-runner, KV pool,
transfer engine, clock) occupied by the sim/mock classes registered in
``register.py`` -- and drives the real prefill-admission and decode-retraction
code paths.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    no_proxy='*' NO_PROXY='*' \
    perl -e 'alarm 120; exec @ARGV' \
    python/.venv/bin/python python/sglang/srt/sim/run_smoke.py
"""

from __future__ import annotations

import sys
from array import array
from types import SimpleNamespace

import torch

# Before anything under sglang.srt.layers is imported: 34 modules freeze
# _is_npu = is_npu() at module scope and is_npu is lru_cached, so the branch
# is decided by whichever import lands first. The sim runs the NPU branch --
# that is the control plane a 910 runs; the non-NPU one is a different
# codebase and proving things about it would prove nothing.
from sglang.srt.sim.fake_npu import assert_npu_branch, install_fake_npu

install_fake_npu()

from sglang.srt.sim import register
from sglang.srt.sim.mock_model_runner import MockModelRunner
from sglang.srt.sim.mock_worker import MockWorker
from sglang.srt.sim.virtual_clock import VirtualClock


def banner(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def build_server_args():
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    # attention_backend="torch_native": the real default is resolved against
    # an actual GPU probe that this harness never runs (sa.device stayed
    # None), and support_triton(None) is True -- that fed the real Triton
    # kernel write_req_to_token_pool_triton into alloc_for_extend, which
    # cannot launch without a GPU. torch_native is the one backend name
    # support_triton() explicitly excludes (sglang/srt/utils.py:1332-1333),
    # which routes alloc_for_extend's write_cache_indices through its plain
    # Python/tensor fallback loop instead -- that fallback is what C1 means
    # by "the control plane does not query the device": it is reachable
    # because of a config choice (no Triton/CUDA kernel in the path), not
    # because scheduler.py/schedule_batch.py were changed.
    server_args = ServerArgs(
        model_path="dummy", device="npu", attention_backend="ascend"
    )
    set_global_server_args_for_scheduler(server_args)
    # Real Scheduler.__init__ (scheduler.py:1099-1104) derives
    # pp_max_micro_batch_size from max_running_requests // pp_size and
    # publishes it via get_context().override(...) the first time it is
    # unset. The sim harness skips Scheduler.__init__ entirely (it builds a
    # __new__ stub instead), so it must do that one override itself or
    # get_num_allocatable_reqs (scheduler.py:3349) divides by None.
    get_context().override(
        "sim.run_smoke", pp_max_micro_batch_size=1 << 20
    )
    return server_args


def build_model_config(vocab_size: int = 32000):
    return SimpleNamespace(
        vocab_size=vocab_size,
        is_encoder_decoder=False,
        hf_text_config=SimpleNamespace(),
        context_len=8192,
    )


def make_req(rid, text_len: int, max_new_tokens: int = 16):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    ids = array("q", list(range(1, text_len + 1)))
    return Req(
        rid=rid,
        origin_input_text="x" * text_len,
        origin_input_ids=ids,
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
    )


def build_scheduler_stub(
    *,
    kv_pool_size: int,
    req_pool_size: int,
    model_config,
    mock_worker,
    clock: VirtualClock,
):
    """A Scheduler.__new__(Scheduler) carrying real control-plane objects.

    Pattern lifted from
    test/registered/unit/managers/test_scheduler_chunked_req_gate.py, which
    SGLang's own unit tests already use to exercise real bound Scheduler
    methods without booting the full engine.
    """
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.schedule_policy import SchedulePolicy
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        size=kv_pool_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    req_to_token_pool = ReqToTokenPool(
        size=req_pool_size,
        max_context_len=model_config.context_len,
        device="cpu",
        enable_memory_saver=False,
    )
    tree_cache = RadixCache.create_simulated(
        mock_allocator=token_to_kv_pool_allocator, page_size=1
    )
    # create_simulated() leaves req_to_token_pool=None ("a radix cache
    # without memory pools for simulation purpose" -- its own docstring).
    # That is fine for prefill admission (schedule_policy only reads
    # evictable_size()/full_evictable_size()/inc_lock_ref() off the tree),
    # but retract_decode()'s release path calls
    # tree_cache.cache_finished_req(), which indexes
    # self.req_to_token_pool.req_to_token directly -- so this prototype
    # wires the real pool in, same object the Scheduler stub itself uses.
    tree_cache.req_to_token_pool = req_to_token_pool
    policy = SchedulePolicy(
        policy="fcfs",
        tree_cache=tree_cache,
        enable_hierarchical_cache=False,
        enable_priority_scheduling=False,
        schedule_low_priority_values_first=False,
    )

    s = Scheduler.__new__(Scheduler)
    s.device = "cpu"
    s.model_config = model_config
    s.tp_worker = mock_worker
    s.page_size = 1
    s.tree_cache = tree_cache
    s.policy = policy
    s.req_to_token_pool = req_to_token_pool
    s.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    s.enable_overlap = False
    s.spec_algorithm = SpeculativeAlgorithm.NONE
    s.waiting_queue = []
    s.chunked_req = None
    # None == unlimited chunk budget (scheduler.py:1220-1222 normalizes any
    # <=0 server_args value to None; passing -1 raw here fed PrefillAdder a
    # rem_chunk_tokens of -1 and silently admitted zero requests).
    s.chunked_prefill_size = None
    s.enable_dynamic_chunking = False
    s.max_prefill_tokens = 1 << 20
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 1 << 20
    s.max_running_requests = 1 << 20
    s.prefill_delayer = None
    s.min_free_slots_delayer = None
    s.enable_hierarchical_cache = False
    s.enable_hicache_storage = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.enable_lora = False
    s.lora_drainer = None
    s.dllm_config = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.enable_fpm = False
    s.enable_hisparse = False
    s.require_mlp_sync = False
    s.beam_coordinator = SimpleNamespace(
        pending_member_rows=lambda batch: 0, retire_group=lambda req: None
    )
    s.load_inquirer = SimpleNamespace(
        _get_num_pending_tokens=lambda chunk_deduct=0: 0
    )
    s.pool_stats_observer = None
    s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
    s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
    s.clock = clock
    s.truncation_align_size = None
    s.enable_priority_scheduling = False
    return s


def main() -> int:
    banner("step 0: register sim shims at the 4 interception points")
    assert_npu_branch()
    print("  is_npu() = True (torch_npu stand-in installed; no NPU present)")
    register.install()
    ok = register.selftest_all()
    print(f"\n  all 4 interception-point selftests passed: {ok}")
    if not ok:
        print("  ABORTING: an interception point did not actually take effect.")
        return 1

    banner("step 1: publish ServerArgs (CPU, no device queries)")
    build_server_args()
    print("torch.cuda.is_available() =", torch.cuda.is_available())

    model_config = build_model_config()
    clock = register.get_shared_clock()
    mock_model_runner = MockModelRunner(model_config=model_config, device="cpu")
    mock_worker = MockWorker(model_runner=mock_model_runner)

    KV_POOL_SIZE = 256
    REQ_POOL_SIZE = 32
    sched = build_scheduler_stub(
        kv_pool_size=KV_POOL_SIZE,
        req_pool_size=REQ_POOL_SIZE,
        model_config=model_config,
        mock_worker=mock_worker,
        clock=clock,
    )

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    banner("step 2: put 10 Req objects into waiting_queue")
    reqs = [make_req(f"req-{i}", text_len=20 + i * 8, max_new_tokens=8) for i in range(10)]
    sched.waiting_queue = list(reqs)
    for r in reqs:
        print(f"  queued {r.rid}: prompt_len={len(r.origin_input_ids)}")

    running_batch = ScheduleBatch(
        reqs=[],
        batch_is_full=False,
        device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )

    banner("step 3: drive prefill admission (Scheduler.get_new_batch_prefill)")
    round_no = 0
    all_admitted = []
    while sched.waiting_queue and round_no < 10:
        round_no += 1
        avail_before = token_to_kv_pool_allocator_available(sched)
        req_pool_avail_before = sched.req_to_token_pool.available_size()
        plan = Scheduler.get_new_batch_prefill(sched, running_batch)
        new_batch = plan.batch_to_run
        running_batch = plan.running_batch
        admitted = new_batch.reqs if new_batch is not None else []
        all_admitted.extend(admitted)
        avail_after = token_to_kv_pool_allocator_available(sched)
        print(
            f"  round {round_no}: available_size before={avail_before} after={avail_after} "
            f"req_pool_available_before={req_pool_avail_before} "
            f"admitted_this_round={len(admitted)} ({[r.rid for r in admitted]}) "
            f"waiting_queue_left={len(sched.waiting_queue)}"
        )
        if new_batch is not None:
            # Fold the freshly admitted prefill batch into the running batch,
            # same as get_next_batch_to_run's last_batch merge, and move it to
            # decode. prepare_for_decode mirrors what scheduler.py does after
            # a one-step forward in the real engine.
            if running_batch is new_batch:
                pass
            elif running_batch.is_empty():
                running_batch = new_batch
            else:
                running_batch.merge_batch(new_batch)
        else:
            break

    print(
        f"\n  TOTAL admitted across {round_no} round(s): {len(all_admitted)} / "
        f"{len(reqs)} requests, running_batch.batch_size()={running_batch.batch_size()}"
    )

    banner("step 4: move running_batch to decode, check_decode_mem")
    running_batch.prepare_for_decode()
    print(
        f"  running_batch is now decode, batch_size={running_batch.batch_size()}, "
        f"token_to_kv_pool_allocator.available_size()={sched.token_to_kv_pool_allocator.available_size()}"
    )
    fits = running_batch.check_decode_mem()
    print(f"  check_decode_mem() with full pool -> fits={fits}")

    banner("step 5: shrink the KV pool and force retract_decode")
    # Drain the allocator to near-empty directly (simulating heavy external
    # pressure) so the next check_decode_mem() call fails and retract_decode
    # actually has work to do.
    allocator = sched.token_to_kv_pool_allocator
    starving_alloc = allocator.alloc(allocator.available_size() - 1)
    print(
        f"  drained allocator down to available_size={allocator.available_size()} "
        f"(kept 1 free page) before retraction"
    )
    fits_after_drain = running_batch.check_decode_mem()
    print(f"  check_decode_mem() after drain -> fits={fits_after_drain}")
    if not fits_after_drain:
        before_bs = running_batch.batch_size()
        retracted_reqs, new_ratio, aborted = running_batch.retract_decode()
        after_bs = running_batch.batch_size()
        print(
            f"  retract_decode(): batch_size {before_bs} -> {after_bs}, "
            f"retracted={len(retracted_reqs)} ({[r.rid for r in retracted_reqs]}), "
            f"aborted={len(aborted)}, new_token_ratio={new_ratio}"
        )
    else:
        print("  pool still fits after drain -- retract_decode not exercised "
              "(this is a FAILURE of the smoke test's setup, not a finding)")
    # restore the pages we borrowed purely to prove the drain happened on a
    # real tensor, not to pretend the scenario is realistic
    if starving_alloc is not None:
        allocator.free(starving_alloc)

    banner("step 6: try run_batch / forward on the mock model runner")
    try:
        fb_like = SimpleNamespace(
            batch_size=running_batch.batch_size(),
            seq_lens=running_batch.seq_lens,
        )
        logits = mock_model_runner.forward(fb_like)
        next_tokens = mock_model_runner.sample(logits)
        print(
            f"  mock forward() logits.shape={tuple(logits.shape)}, "
            f"sample() next_token_ids.shape={tuple(next_tokens.shape)}"
        )
    except Exception as e:  # noqa: BLE001 -- smoke test must report, not hide
        print(f"  FAILED: {type(e).__name__}: {e}")

    banner("step 7: virtual clock sanity check for the timeout interception points")
    t0 = clock.monotonic()
    clock.advance(5.0)
    t1 = clock.monotonic()
    print(f"  virtual monotonic: {t0} -> {t1} (advanced by {t1 - t0}s, no wall-clock sleep)")

    banner("step 8: drive the REAL _abort_on_waiting_timeout off the virtual clock")
    # scheduler.py:2994-3021, unmodified. SGLANG_REQ_WAITING_TIMEOUT defaults
    # to -1 (disabled) per environ.py:588 -- enable it here, put a req in the
    # waiting queue whose wait_queue_entry_time is already behind the
    # (virtual) deadline, and confirm the real method aborts it using only
    # the virtual clock -- no real sleep, no wall-clock read.
    import os as _os

    from sglang.srt.managers.scheduler import Scheduler

    _os.environ["SGLANG_REQ_WAITING_TIMEOUT"] = "10"
    sent = []
    sched.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(
            send_output=lambda msg, req: sent.append(req.rid)
        )
    )
    # entry_time must be in (0, deadline) to trip the abort (scheduler.py:3001
    # checks `0 < entry_time < deadline`); deadline = now - timeout_s.
    stale_req = make_req("stale-req", text_len=5)
    stale_req.time_stats.wait_queue_entry_time = 1.0  # entered at virtual t=1
    fresh_req = make_req("fresh-req", text_len=5)
    fresh_req.time_stats.wait_queue_entry_time = clock.monotonic()  # entered just now
    sched.waiting_queue = [stale_req, fresh_req]
    print(
        f"  before: waiting_queue={[r.rid for r in sched.waiting_queue]}, "
        f"virtual now={clock.monotonic()}, timeout_s=10"
    )
    Scheduler._abort_on_waiting_timeout(sched)
    print(
        f"  after:  waiting_queue={[r.rid for r in sched.waiting_queue]}, "
        f"aborted_and_sent={sent}"
    )
    del _os.environ["SGLANG_REQ_WAITING_TIMEOUT"]

    banner("step 9: drive the REAL _abort_on_running_timeout off the virtual clock")
    # scheduler.py:1704-1717, unmodified. Same virtual-clock mechanism as
    # step 8, exercised on the sibling timeout (running batch, not waiting
    # queue) so both of scheduler.py's timeout call sites are proven, not
    # just the one.
    _os.environ["SGLANG_REQ_RUNNING_TIMEOUT"] = "10"
    victim = running_batch.reqs[0]
    victim.time_stats.forward_entry_time = 1.0  # entered forward at virtual t=1
    print(
        f"  before: victim={victim.rid} to_finish={victim.to_finish} "
        f"virtual now={clock.monotonic()}, timeout_s=10"
    )
    Scheduler._abort_on_running_timeout(sched, running_batch)
    # _abort_on_running_timeout only marks req.to_finish (scheduler.py:1715);
    # turning that into finished()==True happens later in the real pipeline
    # (process_batch_result), out of scope here.
    print(f"  after:  victim={victim.rid} to_finish={victim.to_finish}")
    del _os.environ["SGLANG_REQ_RUNNING_TIMEOUT"]

    banner("step 10: which NPU entry points this run actually reached")
    counts = __import__("sglang.srt.sim.fake_npu", fromlist=["op_counts"]).op_counts()
    if counts:
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {n}")
    else:
        print("  none -- the control plane came up without calling into torch_npu")

    return 0


def token_to_kv_pool_allocator_available(sched) -> int:
    return sched.token_to_kv_pool_allocator.available_size()


if __name__ == "__main__":
    sys.exit(main())
