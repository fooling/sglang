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
import tempfile
import warnings
from array import array
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import torch

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
    "num_hidden_layers": 8,
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
STUB_EXTRAS = {
    "_pending_chunked_abort_req": None,
    "chunked_req": None,
    "enable_fpm": False,
    "dllm_config": None,
    "enable_hisparse": False,
    "enable_hierarchical_cache": False,
    "enable_hicache_storage": False,
    "enable_priority_preemption": False,
    "is_hybrid_swa": False,
    "enable_lora": False,
    "lora_drainer": None,
    "require_mlp_sync": False,
    "disaggregation_mode": None,
    "last_batch": None,
    "forward_ct": 0,
    "_sched_idled": False,
    "cur_batch_for_debug": None,
    "prefill_delayer": None,
    "min_free_slots_delayer": None,
    "enable_dynamic_chunking": False,
    "is_mixed_chunk": False,
    "_prefill_decode_interval_remaining": 0,
    "prefill_decode_interval": 0,  # 0 = feature off (scheduler.py:1251)
}


def extend_stub_for_full_loop(sched) -> list[str]:
    added = []
    for k, v in STUB_EXTRAS.items():
        if not hasattr(sched, k):
            setattr(sched, k, v)
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
    return added


def banner(t: str) -> None:
    print(f"\n{'=' * 18} {t} {'=' * 18}")


def build_k3_model_config():
    """A real ModelConfig for a K3-shaped model -- config.json only, no weights."""
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
    from sglang.srt.configs.model_config import ModelConfig

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
    sched = build_scheduler_stub(
        kv_pool_size=KV_POOL_SIZE,
        req_pool_size=REQ_POOL_SIZE,
        model_config=model_config,
        mock_worker=MockWorker(model_runner=runner),
        clock=clock,
    )

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.common import release_kv_cache
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
        logits = runner.forward(
            SimpleNamespace(batch_size=batch.batch_size(), seq_lens=batch.seq_lens)
        )
        runner.sample(logits)

        events = []
        if newly:
            events.append(f"arrived {newly}")
        if back_in_queue:
            events.append(f"RETRACTED {back_in_queue}")
        for req in batch.reqs:
            req.output_ids.append(scripted_token(req))
            req.update_finish_state()  # the REAL finish-condition code
            if req.finished() and req.rid not in finished:
                finished[req.rid] = type(req.finished_reason).__name__
                events.append(f"{req.rid} {finished[req.rid].replace('FINISH_', '')}")
                # SGLang's own release entry (mem_cache/common.py:201) -- in the
                # real engine process_batch_result calls it. Without it the
                # finished request's pages never come back and later arrivals
                # can never be admitted.
                release_kv_cache(req, sched.tree_cache)

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
        kv_pool_size=KV_POOL_SIZE, logits_shape=tuple(logits.shape),
        waiting_left=len(sched.waiting_queue), stub_extras=stub_extras,
    )
    return 0 if all_done else 2


def run() -> dict:
    """Run the full pass and hand back the facts, for tests to assert on."""
    main()
    return dict(LAST_RUN)


if __name__ == "__main__":
    raise SystemExit(main())
