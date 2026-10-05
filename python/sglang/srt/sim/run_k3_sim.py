"""Full control-plane pass for a Kimi-K3 shaped model, weights never loaded.

What this adds over ``run_smoke.py``: run_smoke drives admission once and
stops. This drives the *loop* -- the real ``Scheduler.get_next_batch_to_run``
every iteration, prefill then decode then decode..., applying sampled tokens
and letting the real ``Req.update_finish_state`` decide when each request
ends, until the queue drains.

No checkpoint is downloaded and nothing is written to disk beyond a temporary
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
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from types import SimpleNamespace

from sglang.srt.sim import register
from sglang.srt.sim.mock_model_runner import MockModelRunner
from sglang.srt.sim.mock_worker import MockWorker
from sglang.srt.sim.run_smoke import build_scheduler_stub, build_server_args, make_req

# KimiLinearConfig class defaults + self-consistent MLA dims. Example
# parameters, not a released checkpoint.
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

KV_POOL_SIZE = 4096
REQ_POOL_SIZE = 64
MAX_STEPS = 200

# Filled by main(); run() returns a copy so tests can assert on real numbers.
LAST_RUN: dict = {}



# Extra stub surface that only the full loop touches. run_smoke's stub covers
# get_new_batch_prefill; get_next_batch_to_run reaches further (chunked-abort
# bookkeeping, dllm, hisparse, dp-attn, prefill/decode interval arming).
# Every field here is inert -- it switches the corresponding feature off so the
# plain single-instance path is what runs. Nothing in SGLang is modified.
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
    "prefill_decode_interval": 0,
}


def _passthrough_dp_attn_adapter():
    """DP-attention off: both hooks hand the batch straight back.

    scheduler.py calls maybe_prepare_mlp_sync_batch / maybe_convert_decode_to_extend
    unconditionally; with dp-attn disabled they are identity in the real code too.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda batch, need_sync=None: batch,
        maybe_convert_decode_to_extend=lambda batch: batch,
    )


def extend_stub_for_full_loop(sched) -> list[str]:
    """Fill in only the attributes the full loop needs; report which were missing."""
    added = []
    for k, v in STUB_EXTRAS.items():
        if not hasattr(sched, k):
            setattr(sched, k, v)
            added.append(k)
    tracker = getattr(sched, "new_token_ratio_tracker", None)
    if tracker is not None and not hasattr(tracker, "decay_step"):
        # The decode path (scheduler.py:3789) decays this every step; the real
        # tracker only adjusts a float used for retraction headroom.
        tracker.decay_step = lambda: None
        added.append("new_token_ratio_tracker.decay_step")
    if not hasattr(sched, "ngram_embedding_manager"):
        # ngram speculative decoding off: scheduler.py:3332 calls through
        # unconditionally, so hand the batch back untouched.
        from types import SimpleNamespace

        sched.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, *a, **k: batch
        )
        added.append("ngram_embedding_manager")
    if not hasattr(sched, "dp_attn_adapter"):
        sched.dp_attn_adapter = _passthrough_dp_attn_adapter()
        added.append("dp_attn_adapter")
    if added:
        print(f"  stub extras added for the full loop ({len(added)}): {added}")
    return added


def banner(t: str) -> None:
    print(f"\n{'=' * 20} {t} {'=' * 20}")


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
    print(f"  config.json at {cfg_dir}")
    print(
        f"  model_type={model_config.hf_config.model_type} "
        f"arch={model_config.hf_config.architectures} "
        f"attention_arch={model_config.attention_arch}"
    )
    print(
        f"  vocab={model_config.vocab_size} hidden={model_config.hidden_size} "
        f"layers={model_config.num_hidden_layers} ctx={model_config.context_len}"
    )
    print(
        f"  MLA: kv_lora_rank={model_config.kv_lora_rank} "
        f"qk_nope={model_config.qk_nope_head_dim} qk_rope={model_config.qk_rope_head_dim} "
        f"v_head={model_config.v_head_dim} head_dim={model_config.head_dim}"
    )
    print("  weights loaded: none (MockModelRunner does not touch a checkpoint)")

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
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    stub_extras = extend_stub_for_full_loop(sched)

    banner("step 2: queue requests")
    specs = [(24, 4), (40, 6), (56, 3), (72, 5), (88, 2), (104, 4)]
    reqs = [
        make_req(f"k3-{i}", text_len=plen, max_new_tokens=mnt)
        for i, (plen, mnt) in enumerate(specs)
    ]
    for r in reqs:
        # tokenizer_manager normally does this before a Req reaches the
        # scheduler; without it stop_strs stays None and the real
        # update_finish_state trips. No tokenizer needed when stop_strs is empty.
        r.sampling_params.normalize(None)
    sched.waiting_queue = list(reqs)
    for r, (plen, mnt) in zip(reqs, specs):
        print(f"  {r.rid}: prompt={plen} tok, max_new_tokens={mnt}")

    banner("step 3: run the scheduler loop to completion")
    running_batch = ScheduleBatch(
        reqs=[], batch_is_full=False, device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    last_batch = None
    finished: dict[str, str] = {}
    modes: list[str] = []
    steps = 0
    print(f"  {'step':>4} {'mode':<8} {'bs':>3} {'kv_free':>8}  {'reqs':<34} finished")
    while steps < MAX_STEPS:
        steps += 1
        plan = Scheduler.get_next_batch_to_run(
            sched, running_batch=running_batch, last_batch=last_batch
        )
        running_batch = plan.running_batch
        batch = plan.batch_to_run
        if batch is None:
            print(f"  {steps:>4} {'(idle)':<8} {'-':>3} "
                  f"{sched.token_to_kv_pool_allocator.available_size():>8}  "
                  f"{'--':<34} queue={len(sched.waiting_queue)}")
            break

        mode = "prefill" if batch.forward_mode.is_extend() else "decode"
        modes.append(mode)
        # Mock forward + sample: shapes follow the real batch, values are fake.
        # ScheduleBatch.batch_size is a method; MockModelRunner wants the
        # shape-carrying fields a ForwardBatch would expose.
        fb_like = SimpleNamespace(
            batch_size=batch.batch_size(), seq_lens=batch.seq_lens
        )
        logits = runner.forward(fb_like)
        next_tokens = runner.sample(logits)

        just_finished = []
        for req, tok in zip(batch.reqs, next_tokens.tolist()):
            req.output_ids.append(int(tok))
            req.update_finish_state()  # the REAL finish-condition code
            if req.finished():
                just_finished.append(req.rid)
                finished[req.rid] = type(req.finished_reason).__name__

        print(
            f"  {steps:>4} {mode:<8} {batch.batch_size():>3} "
            f"{sched.token_to_kv_pool_allocator.available_size():>8}  "
            f"{str([r.rid for r in batch.reqs]):<34} {just_finished}"
        )

        last_batch = batch
        if len(finished) == len(reqs):
            break

    banner("step 4: result")
    print(f"  steps run: {steps}")
    print(f"  finished: {len(finished)} / {len(reqs)}")
    for rid in (r.rid for r in reqs):
        print(f"    {rid}: {finished.get(rid, 'NOT FINISHED')} "
              f"output_len={len(next(r for r in reqs if r.rid == rid).output_ids)}")
    print(f"  waiting_queue left: {len(sched.waiting_queue)}")
    print(f"  kv pages free at end: {sched.token_to_kv_pool_allocator.available_size()}"
          f" / {KV_POOL_SIZE}")
    print(f"  logits dtype/shape of last forward: {logits.dtype} {tuple(logits.shape)}")

    all_done = len(finished) == len(reqs)
    print(f"\n  ALL REQUESTS FINISHED: {all_done}")
    LAST_RUN.clear()
    LAST_RUN.update(
        model_config=model_config,
        specs=specs,
        reqs=reqs,
        finished=finished,
        steps=steps,
        modes=modes,
        kv_free_end=sched.token_to_kv_pool_allocator.available_size(),
        kv_pool_size=KV_POOL_SIZE,
        logits_shape=tuple(logits.shape),
        waiting_left=len(sched.waiting_queue),
        stub_extras=stub_extras,
    )
    return 0 if all_done else 2


def run() -> dict:
    """Run the full pass and hand back the facts, for tests to assert on."""
    main()
    return dict(LAST_RUN)


if __name__ == "__main__":
    raise SystemExit(main())
