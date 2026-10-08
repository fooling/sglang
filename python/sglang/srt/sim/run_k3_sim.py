"""Full control-plane pass for a Kimi-K3 shaped model, weights never loaded.

What this exercises that ``run_smoke.py`` does not:

* the *loop* -- real ``Scheduler.get_next_batch_to_run`` every iteration,
  prefill then decode then decode..., until every request is done;
* **continuous batching** -- requests arrive in waves, so a later prefill is
  admitted while earlier requests are still decoding, and SGLang's own
  ``last_batch`` path merges them;
* **two different finish reasons** -- one request is handed an EOS token
  mid-flight and must end via ``FINISH_MATCHED_TOKEN``, the rest run out their
  ``max_new_tokens`` and end via ``FINISH_LENGTH``;
* **a KV pool too small for the whole workload**, so admission has to refuse
  and defer instead of taking everything in one go.

No checkpoint is downloaded and nothing lands on disk beyond a temporary
``config.json``: ``ModelConfig`` only needs the architecture, so a K3-shaped
config is enough to get a real ``AttentionArch.MLA`` model config. Forward and
sampling are mocked; everything on the decision path is SGLang's own code.

Shapes come from the repo's own ``KimiLinearConfig`` defaults (vocab 163840,
hidden 4096, 32 heads) plus self-consistent MLA dims. They are example
parameters -- this does not claim to be any released checkpoint.

Run:

    cd ~/repo/sglang && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 no_proxy='*' \
      perl -e 'alarm 600; exec @ARGV' python/.venv/bin/python \
      python/sglang/srt/sim/run_k3_sim.py
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from array import array
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import torch

# Before anything under sglang.srt.layers is imported -- see fake_npu's module
# docstring for why the branch is decided by import order.
from sglang.srt.sim.fake_npu import assert_npu_branch, install_fake_npu

install_fake_npu()

from sglang.srt.sim import register
from sglang.srt.sim.mock_model_runner import MockModelRunner
from sglang.srt.sim.mock_worker import MockWorker
from sglang.srt.sim.run_smoke import build_scheduler_stub, build_server_args

K3_TEXT_CONFIG = {
    "model_type": "kimi_linear",
    "architectures": ["KimiK3LinearForCausalLM"],
    "vocab_size": 163840,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    # the class's own default, not a number of ours (configs/kimi_linear.py:21)
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 4096,
    "eos_token_id": 2,
    "n_routed_experts": 32,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 1408,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    # K3 is a hybrid: most layers are KDA (linear attention, carrying a state
    # instead of KV pages), a few stay full attention. SGLang reads this off
    # the config and hands the scheduler a HybridReqToTokenPool.
    # Filled in below by kda_layer_pattern() -- the layer numbers are 1-BASED
    # (is_kda_layer does `(layer_idx + 1) in kda_layers`,
    # configs/kimi_linear.py:172), which is easy to get wrong by one.
    "linear_attn_config": None,
}

# How many layers per full-attention layer. One example of the config format,
# not a sourced fact about any released checkpoint: the format itself is two
# explicit layer lists, so a real config.json drops straight in (see
# build_k3_model_config's config_path argument).
FULL_ATTN_EVERY = 4


def kda_layer_pattern(num_layers: int, full_every: int = FULL_ATTN_EVERY) -> dict:
    """The two 1-based layer lists the config format wants.

    1-based because is_kda_layer tests ``(layer_idx + 1) in kda_layers``.
    Writing them 0-based silently shifts the whole pattern -- which is exactly
    what happened before this helper existed.
    """
    layers = range(1, num_layers + 1)
    return {
        "kda_layers": [i for i in layers if i % full_every != 0],
        "full_attn_layers": [i for i in layers if i % full_every == 0],
    }


K3_TEXT_CONFIG["linear_attn_config"] = {
    "num_heads": 32,
    "head_dim": 128,
    "short_conv_kernel_size": 4,
    **kda_layer_pattern(K3_TEXT_CONFIG["num_hidden_layers"]),
}

EOS_ID = 2
FILLER_TOKEN = 7

# Tight on purpose: the workload needs more pages than the pool holds, so
# admission must refuse and defer rather than take everything at once.
KV_POOL_SIZE = 320
REQ_POOL_SIZE = 64
MAX_STEPS = 60

# (rid, prompt_len, max_new_tokens, arrives_before_step)
WORKLOAD = [
    ("a0", 48, 8, 1),
    ("a1", 64, 12, 1),
    ("a2", 80, 6, 1),
    ("b0", 96, 5, 3),
    ("b1", 112, 4, 3),
    ("c0", 56, 3, 6),
]
EOS_SCRIPT = {"a1": 5}   # a1 gets EOS as its 5th output token

LAST_RUN: dict = {}

# Extra stub surface only the full loop touches. Every field is inert: it
# switches the matching feature off so the plain single-instance path runs.
# Nothing in SGLang is modified.
# Per-iteration bookkeeping a real Scheduler.__init__ would zero out. These
# are not decisions -- they are counters and last-seen handles.
STUB_BOOKKEEPING = {
    "_pending_chunked_abort_req": None,
    "chunked_req": None,
    "last_batch": None,
    "forward_ct": 0,
    "_sched_idled": True,            # scheduler.py starts idle
    "cur_batch_for_debug": None,
    "_prefill_decode_interval_remaining": 0,
    "dllm_config": None,             # no diffusion-LLM config published
    "lora_drainer": None,            # only built when LoRA is on
    "prefill_delayer": None,         # only built when the delayer is on
    "min_free_slots_delayer": None,  # ditto
    # Step-time accounting, same initial values as Scheduler.__init__
    # (:2208-2210). These only became reachable once the forward started
    # charging a time slice: _record_step_counters gates on 0 < step_us, and
    # before that step_us was always 0, so the engine's own step-time ledger
    # never ran. With a slice it does, which is the point.
    "_prev_step": None,
    "total_prefill_uncached_tokens": 0,
    "total_prefill_busy_us": 0,
    "decode_moment_totals": None,  # replaced below; a fresh list per stub
}


def resolved_feature_flags() -> dict:
    """Read every feature flag from the SAME source Scheduler.__init__ reads.

    Hardcoding these would mean the harness, not the config, decides what the
    scheduler does -- which is exactly the thing this prototype must not do.
    """
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.scheduler import require_mlp_sync
    from sglang.srt.runtime_context import (
        get_disagg,
        get_lora,
        get_memory,
        get_schedule,
    )

    schedule = get_schedule()
    return {
        "enable_hisparse": get_memory().enable_hisparse,
        "enable_hierarchical_cache": get_memory().enable_hierarchical_cache,
        "enable_hicache_storage": get_memory().hicache_storage_backend is not None,
        "enable_lora": get_lora().enable_lora,
        "require_mlp_sync": require_mlp_sync(),
        "disaggregation_mode": DisaggregationMode(get_disagg().disaggregation_mode),
        "prefill_decode_interval": schedule.prefill_decode_interval,
        "enable_priority_preemption": getattr(
            schedule, "enable_priority_scheduling", False
        ),
        "enable_dynamic_chunking": getattr(schedule, "enable_dynamic_chunking", False),
        "is_mixed_chunk": getattr(schedule, "enable_mixed_chunk", False),
        "enable_fpm": getattr(schedule, "enable_fpm", False),
        "is_hybrid_swa": False,  # K3 is MLA, not hybrid SWA
    }


def extend_stub_for_full_loop(sched) -> list[str]:
    added = []
    for k, v in {**STUB_BOOKKEEPING, **resolved_feature_flags()}.items():
        if not hasattr(sched, k):
            # one list per stub, never a shared default
            setattr(sched, k, [0.0] * 6 if k == "decode_moment_totals" else v)
            added.append(k)
    tracker = getattr(sched, "new_token_ratio_tracker", None)
    if tracker is not None and not hasattr(tracker, "decay_step"):
        tracker.decay_step = lambda: None
        added.append("new_token_ratio_tracker.decay_step")
    if not hasattr(sched, "ngram_embedding_manager"):
        sched.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, *a, **k: batch
        )
        added.append("ngram_embedding_manager")
    if not hasattr(sched, "dp_attn_adapter"):
        sched.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, need_sync=None: batch,
            maybe_convert_decode_to_extend=lambda batch: batch,
        )
        added.append("dp_attn_adapter")
    added += _extend_for_real_forward_path(sched)
    return added


class _ZeroCall(int):
    """Reads as 0, calls as a no-op -- so one sink covers counters and hooks."""

    def __call__(self, *a, **k):
        return None


class _MetricsSink:
    """Metrics are a side channel, never a decision: everything is 0 / no-op.

    Nothing here feeds a scheduling decision -- if it ever did, this sink would
    be hiding it, so the architecture test pins the decision functions instead.
    """

    def __getattr__(self, _name):
        return _ZeroCall(0)


def _extend_for_real_forward_path(sched) -> list[str]:
    """What Scheduler.run_batch / process_batch_result need beyond the above.

    Two rules here:
      * anything that *decides* gets the real object (FutureMap, the real
        SchedulerBatchResultProcessor -- it is the code that writes tokens,
        judges finish and releases KV);
      * anything that only *reports* (metrics, load publishing) gets a sink.
    """
    import torch
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    added = []
    flat = {
        "scripted_scheduler_hook": None,
        "forward_sleep_time": None,
        "is_generation": True,
        "enable_pdmux": False,
        "enable_dp_attention": False,
        "enable_unified_memory": False,
        "return_health_check_ipcs": False,
        "_prev_step": None,
        "load_snapshot_writer": None,
        "profiler_manager": SimpleNamespace(_profile_batch_predicate=lambda b: None),
    }
    for k, v in flat.items():
        if not hasattr(sched, k):
            setattr(sched, k, v)
            added.append(k)
    # these two already exist on the stub; make sure the methods process_batch_result
    # calls are on them (load reporting is a side channel, not a decision)
    for obj_name, meth in (("load_inquirer", "get_loads"),
                           ("load_publisher", "publish_load_stat")):
        obj = getattr(sched, obj_name, None)
        if obj is None:
            obj = SimpleNamespace()
            setattr(sched, obj_name, obj)
            added.append(obj_name)
        if not hasattr(obj, meth):
            setattr(obj, meth, lambda *a, **k: None)
            added.append(f"{obj_name}.{meth}")
    if not hasattr(sched, "model_worker"):
        sched.model_worker = sched.tp_worker
        added.append("model_worker")
    if not isinstance(getattr(sched, "metrics_reporter", None), _MetricsSink):
        sched.metrics_reporter = _MetricsSink()
        added.append("metrics_reporter")
    # beam search is off: every hook is a no-op, commit_decode yields no groups
    for name, ret in (("maybe_select_and_relay", None), ("pending_member_rows", 0),
                      ("retire_group", None), ("commit_decode", ()),
                      ("commit_prefill", ()), ("finalize_groups", ())):
        if not hasattr(sched.beam_coordinator, name):
            setattr(sched.beam_coordinator, name, lambda *a, _r=ret, **k: _r)
            added.append(f"beam_coordinator.{name}")

    # the real relay buffer, not a fake: it is control-plane plumbing
    if getattr(sched, "future_map", None) is None:
        sched.future_map = FutureMap(
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.NONE,
            req_to_token_pool=sched.req_to_token_pool,
        )
        added.append("future_map(real)")

    # the real result processor: this is the code under test, not a stub
    if getattr(sched, "batch_result_processor", None) is None:
        sched.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=sched.disaggregation_mode,
            enable_overlap=False,
            enable_overlap_mlx=False,
            model_config=sched.model_config,
            token_to_kv_pool_allocator=sched.token_to_kv_pool_allocator,
            tree_cache=sched.tree_cache,
            hisparse_coordinator=None,
            req_to_token_pool=sched.req_to_token_pool,
            decode_offload_manager=None,
            metrics_collector=_MetricsSink(),
            metrics_reporter=_MetricsSink(),
            draft_worker=None,
            model_worker=sched.tp_worker,
            logprob_result_processor=_MetricsSink(),
            output_streamer=_MetricsSink(),
            beam_coordinator=sched.beam_coordinator,
            abort_request=lambda *a, **k: None,
        )
        added.append("batch_result_processor(real)")
    return added


def banner(t: str) -> None:
    print(f"\n{'=' * 18} {t} {'=' * 18}")


def build_k3_model_config(config_path: "str | Path | None" = None):
    """A real ModelConfig for a K3-shaped model -- config.json only, no weights.

    ``config_path`` takes a real released config.json and uses it verbatim.
    Nothing downstream cares where the shapes came from: ModelConfig parses it,
    mambaish_config reads the layer pattern off it, and the pools are sized
    from whatever it says. The example below exists because the real file is
    not in this offline environment, not because it could not be used.

    Also settable from the environment, so a run can be pointed at a real
    config without editing code:

        SIM_K3_CONFIG_JSON=/path/to/config.json
    """
    from sglang.srt.configs.model_config import ModelConfig

    path = config_path or os.environ.get("SIM_K3_CONFIG_JSON")
    if path:
        src = Path(path)
        d = Path(tempfile.mkdtemp(prefix="k3-cfg-real-"))
        (d / "config.json").write_text(src.read_text())
        return ModelConfig(model_path=str(d), trust_remote_code=True), d

    d = Path(tempfile.mkdtemp(prefix="k3-cfg-"))
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "architectures": ["KimiK3LinearForCausalLM"],
                "torch_dtype": "bfloat16",
                "text_config": K3_TEXT_CONFIG,
            }
        )
    )
    return ModelConfig(model_path=str(d), trust_remote_code=True), d


def make_k3_req(rid: str, prompt_len: int, max_new: int):
    """A Req that knows its EOS, with sampling params normalized.

    tokenizer_manager normally calls normalize() before a Req reaches the
    scheduler; without it stop_strs stays None and the real
    update_finish_state trips on len(None).
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = Req(
        rid=rid,
        origin_input_text="x" * prompt_len,
        origin_input_ids=array("q", list(range(1, prompt_len + 1))),
        sampling_params=SamplingParams(max_new_tokens=max_new),
        eos_token_ids={EOS_ID},
    )
    req.sampling_params.normalize(None)
    return req


def scripted_token(req) -> int:
    """What the mock sampler hands back for this request this step."""
    eos_at = EOS_SCRIPT.get(req.rid)
    if eos_at is not None and len(req.output_ids) + 1 == eos_at:
        return EOS_ID
    return FILLER_TOKEN


def main() -> int:
    banner("step 0: interception points")
    assert_npu_branch()
    print("  is_npu() = True (torch_npu stand-in installed; no NPU present)")
    register.install()
    if not register.selftest_all():
        print("  ABORTING: an interception point did not take effect.")
        return 1
    build_server_args()
    print(f"  all four installed; torch.cuda.is_available()={torch.cuda.is_available()}")

    banner("step 1: K3 model config (no checkpoint, no download)")
    model_config, cfg_dir = build_k3_model_config()
    print(f"  config dir holds: {sorted(p.name for p in cfg_dir.iterdir())}")
    print(f"  model_type={model_config.hf_config.model_type} "
          f"arch={model_config.hf_config.architectures} "
          f"attention_arch={model_config.attention_arch}")
    print(f"  vocab={model_config.vocab_size} hidden={model_config.hidden_size} "
          f"layers={model_config.num_hidden_layers} ctx={model_config.context_len}")
    print(f"  MLA: kv_lora_rank={model_config.kv_lora_rank} "
          f"qk_nope={model_config.qk_nope_head_dim} "
          f"qk_rope={model_config.qk_rope_head_dim} "
          f"v_head={model_config.v_head_dim} head_dim={model_config.head_dim}")
    print("  weights loaded: none")

    clock = register.get_shared_clock()
    runner = MockModelRunner(model_config=model_config, device="cpu")
    runner.token_script = scripted_token  # mock sampler reads this
    sched = build_scheduler_stub(
        kv_pool_size=KV_POOL_SIZE,
        req_pool_size=REQ_POOL_SIZE,
        model_config=model_config,
        mock_worker=MockWorker(model_runner=runner),
        clock=clock,
    )

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    stub_extras = extend_stub_for_full_loop(sched)
    print(f"  stub extras for the full loop ({len(stub_extras)}): {stub_extras}")

    banner("step 2: workload (arrives in waves)")
    reqs = {rid: make_k3_req(rid, plen, mnt) for rid, plen, mnt, _ in WORKLOAD}
    total_prompt = sum(p for _r, p, _m, _a in WORKLOAD)
    print(f"  KV pool = {KV_POOL_SIZE} pages, total prompt tokens = {total_prompt}"
          f"  -> the pool cannot hold the whole workload at once")
    for rid, plen, mnt, arrive in WORKLOAD:
        eos = f", EOS scripted at output #{EOS_SCRIPT[rid]}" if rid in EOS_SCRIPT else ""
        print(f"  {rid}: prompt={plen} max_new={mnt} arrives before step {arrive}{eos}")
    sched.waiting_queue = []

    banner("step 3: scheduler loop")
    running_batch = ScheduleBatch(
        reqs=[], batch_is_full=False, device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    last_batch = None
    finished: dict[str, str] = {}
    modes: list[str] = []
    logits_shape = ()
    retractions: list = []
    arrivals: list = []
    steps = 0
    idle_streak = 0
    print(f"  {'step':>4} {'mode':<8} {'bs':>3} {'kvfree':>7} {'wait':>5}  "
          f"{'batch':<30} events")
    while steps < MAX_STEPS:
        steps += 1
        newly = [rid for rid, _p, _m, a in WORKLOAD if a == steps]
        if newly:
            sched.waiting_queue.extend(reqs[r] for r in newly)
            arrivals.append((steps, newly))

        before_running = {r.rid for r in running_batch.reqs}
        plan = Scheduler.get_next_batch_to_run(
            sched, running_batch=running_batch, last_batch=last_batch
        )
        running_batch = plan.running_batch
        batch = plan.batch_to_run
        back_in_queue = sorted({r.rid for r in sched.waiting_queue} & before_running)
        if back_in_queue:
            retractions.append((steps, back_in_queue))

        if batch is None:
            print(f"  {steps:>4} {'(idle)':<8} {'-':>3} "
                  f"{sched.token_to_kv_pool_allocator.available_size():>7} "
                  f"{len(sched.waiting_queue):>5}  {'--':<30} "
                  f"{'arrived ' + str(newly) if newly else ''}")
            idle_streak += 1
            future = [a for _r, _p, _m, a in WORKLOAD if a > steps]
            if not sched.waiting_queue or (idle_streak >= 3 and not future):
                break
            last_batch = None
            continue

        idle_streak = 0
        mode = "prefill" if batch.forward_mode.is_extend() else "decode"
        modes.append(mode)

        in_batch = list(batch.reqs)
        # SGLang's own forward entry and output handling. The harness no longer
        # sequences "append token -> judge finish -> release KV" itself; that
        # whole sequence is process_batch_result's job and it runs here.
        result = Scheduler.run_batch(sched, batch)
        Scheduler.process_batch_result(sched, batch, result)
        logits_shape = tuple(result.logits_output.next_token_logits.shape)

        events = []
        if newly:
            events.append(f"arrived {newly}")
        if back_in_queue:
            events.append(f"RETRACTED {back_in_queue}")
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
    print(f"  steps: {steps}   prefills: {modes.count('prefill')}   "
          f"decodes: {modes.count('decode')}")
    print(f"  arrivals: {arrivals}")
    print(f"  retractions: {retractions if retractions else 'none'}")
    print(f"  finished: {len(finished)} / {len(WORKLOAD)}")
    for reason, rids in sorted(by_reason.items()):
        print(f"    {reason}: {sorted(rids)}")
    for rid, plen, mnt, _a in WORKLOAD:
        r = reqs[rid]
        print(f"    {rid}: output_len={len(r.output_ids)} (max_new={mnt}) "
              f"reason={finished.get(rid, 'NOT FINISHED')}")
    print(f"  waiting_queue left: {len(sched.waiting_queue)}")
    kv_free = sched.token_to_kv_pool_allocator.available_size()
    evictable = sched.tree_cache.evictable_size()
    print(f"  kv at end: free={kv_free} + radix evictable={evictable} "
          f"= {kv_free + evictable} / {KV_POOL_SIZE}  (nothing leaked; the radix "
          f"cache keeps finished prefixes for reuse)")
    print(f"  req slots: {sched.req_to_token_pool.available_size()} / {REQ_POOL_SIZE} free")

    all_done = len(finished) == len(WORKLOAD)
    print(f"\n  ALL REQUESTS FINISHED: {all_done}")

    LAST_RUN.clear()
    LAST_RUN.update(
        model_config=model_config, workload=WORKLOAD, reqs=reqs, finished=finished,
        by_reason=by_reason, steps=steps, modes=modes, arrivals=arrivals,
        retractions=retractions,
        kv_free_end=kv_free, radix_evictable=evictable,
        req_slots_free=sched.req_to_token_pool.available_size(),
        kv_pool_size=KV_POOL_SIZE, logits_shape=logits_shape,
        waiting_left=len(sched.waiting_queue), stub_extras=stub_extras,
        result_processor_cls=type(sched.batch_result_processor).__name__,
        future_map_cls=type(sched.future_map).__name__,
        ran_real_forward_path=True,
    )
    return 0 if all_done else 2


def run() -> dict:
    """Run the full pass and hand back the facts, for tests to assert on."""
    main()
    return dict(LAST_RUN)


if __name__ == "__main__":
    raise SystemExit(main())
