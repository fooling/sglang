"""Tests for the sim-interception prototype.

What these cover that ``run_smoke.py`` does not: ``run_smoke`` only *prints*
numbers -- if the admission ledger, the retraction path or an interception
point silently changed behaviour, it would still exit 0. Everything below
asserts, so a regression fails.

Run:

    cd ~/repo/sglang && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 no_proxy='*' \
      perl -e 'alarm 600; exec @ARGV' python/.venv/bin/python -m pytest \
      python/sglang/srt/sim/tests/ -q

Claims under test (same numbering as FEASIBILITY.md):
  C1  control plane imports and runs with no device libraries
  C2  the four selection points can be taken over
  C3  batching decisions read only Python-side state (queues, index ledgers,
      startup constants) -- no device query
  C4  the KV allocator is index bookkeeping, so device='cpu' is enough
  C5  a starved KV pool really drives retract_decode

Plus two guards that are not claims but invariants of this branch:
  G1  no pre-existing SGLang source file is modified
  G2  the attention-backend *name* must stay on a non-Triton branch
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sim import register
from sglang.srt.sim.cpu_kv import (
    build_cpu_req_to_token_pool,
    build_cpu_token_to_kv_pool_allocator,
)
from sglang.srt.sim.mock_model_runner import MockModelRunner
from sglang.srt.sim.mock_worker import MockWorker
from sglang.srt.sim.run_smoke import (
    build_model_config,
    build_scheduler_stub,
    build_server_args,
    make_req,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
BASE_COMMIT = "5b33b51793"  # branch point of feat/sim-interception
SIM_PREFIX = "python/sglang/srt/sim/"

CONTROL_PLANE_MODULES = [
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.schedule_policy",
    "sglang.srt.managers.schedule_batch",
    "sglang.srt.managers.tp_worker",
    "sglang.srt.mem_cache.allocator.token",
    "sglang.srt.mem_cache.memory_pool",
    "sglang.srt.mem_cache.kv_cache_configurator",
    "sglang.srt.model_executor.model_runner",
    "sglang.srt.managers.tokenizer_manager",
    "sglang.srt.disaggregation.ascend.conn",
]

KV_POOL_SIZE = 256
REQ_POOL_SIZE = 32


# ───────────────────────── fixtures ─────────────────────────
@pytest.fixture(scope="session", autouse=True)
def _shims():
    """Install the four interception points once for the whole session."""
    register.install()
    build_server_args()
    yield
    register.uninstall()


def _fresh_stub():
    """A Scheduler stub carrying real control-plane objects, built from scratch.

    Rebuilt per test so ledger state never leaks between tests.
    """
    model_config = build_model_config()
    clock = register.get_shared_clock()
    runner = MockModelRunner(model_config=model_config, device="cpu")
    worker = MockWorker(model_runner=runner)
    sched = build_scheduler_stub(
        kv_pool_size=KV_POOL_SIZE,
        req_pool_size=REQ_POOL_SIZE,
        model_config=model_config,
        mock_worker=worker,
        clock=clock,
    )
    return sched, clock


def _empty_batch():
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return ScheduleBatch(
        reqs=[],
        batch_is_full=False,
        device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


def _run_admission(sched, reqs, max_rounds: int = 10):
    """Drive the REAL Scheduler.get_new_batch_prefill until it stops admitting.

    Returns (admitted_rids, running_batch, per_round_available_size).
    """
    from sglang.srt.managers.scheduler import Scheduler

    sched.waiting_queue = list(reqs)
    running_batch = _empty_batch()
    admitted, avails = [], []
    for _ in range(max_rounds):
        if not sched.waiting_queue:
            break
        avails.append(sched.token_to_kv_pool_allocator.available_size())
        plan = Scheduler.get_new_batch_prefill(sched, running_batch)
        new_batch, running_batch = plan.batch_to_run, plan.running_batch
        if new_batch is None:
            break
        admitted.extend(r.rid for r in new_batch.reqs)
        if running_batch is new_batch:
            pass
        elif running_batch.is_empty():
            running_batch = new_batch
        else:
            running_batch.merge_batch(new_batch)
    return admitted, running_batch, avails


# ───────────────────────── G1: no source modified ─────────────────────────
def test_no_preexisting_sglang_source_modified():
    """The whole proposition is "install sim classes without touching SGLang".

    If this fails, the prototype no longer demonstrates what it claims.
    """
    committed = subprocess.run(
        ["git", "diff", "--name-only", f"{BASE_COMMIT}..HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    offenders = [p for p in committed if not p.startswith(SIM_PREFIX)]
    assert not offenders, f"committed changes outside {SIM_PREFIX}: {offenders}"

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    tracked_dirty = [
        ln[3:] for ln in dirty
        if not ln.startswith("??") and not ln[3:].startswith(SIM_PREFIX)
    ]
    assert not tracked_dirty, f"uncommitted edits to tracked files: {tracked_dirty}"


# ───────────────────────── C1: no device libs ─────────────────────────
def test_no_device_libraries_in_play():
    assert torch.cuda.is_available() is False
    assert "torch_npu" not in sys.modules


@pytest.mark.parametrize("module", CONTROL_PLANE_MODULES)
def test_control_plane_module_imports(module):
    assert importlib.import_module(module) is not None


# ───────────────────────── C4: index bookkeeping ─────────────────────────
def test_kv_allocator_is_index_bookkeeping_on_cpu():
    alloc = build_cpu_token_to_kv_pool_allocator(size=64)
    assert alloc.available_size() == 64

    idx = alloc.alloc(10)
    assert idx is not None and len(idx) == 10
    assert idx.device.type == "cpu"
    assert alloc.available_size() == 54

    alloc.free(idx)
    assert alloc.available_size() == 64

    # free_pages is an index tensor, not storage: it lives on CPU and its
    # length *is* the answer available_size() gives.
    assert alloc.free_pages.device.type == "cpu"
    assert alloc.available_size() == len(alloc.free_pages) + len(alloc.release_pages)


def test_kv_allocator_overdraw_returns_none_not_raise():
    alloc = build_cpu_token_to_kv_pool_allocator(size=8)
    assert alloc.alloc(9) is None
    assert alloc.available_size() == 8, "a failed alloc must not consume pages"


def test_req_to_token_pool_available_size_is_free_slot_count():
    pool = build_cpu_req_to_token_pool(size=8, max_context_len=32)
    assert pool.available_size() == len(pool.free_slots) == 8


# ───────────────────────── C3: decisions read Python state only ──────────
def test_num_allocatable_reqs_follows_req_pool_ledger():
    """The admission quota is min(pp budget, req_to_token_pool.available_size()).

    Drain the req pool and the quota must follow it down -- no device query
    anywhere in that path.
    """
    from sglang.srt.managers.scheduler import Scheduler

    sched, _ = _fresh_stub()
    batch = _empty_batch()
    avail0 = sched.req_to_token_pool.available_size()
    # running_batch must be passed explicitly: scheduler.py:3352 falls back to
    # self.running_batch, which only exists on a fully-constructed Scheduler.
    quota0 = Scheduler.get_num_allocatable_reqs(sched, 0, running_batch=batch)
    assert quota0 == min(quota0, avail0)

    occupy = [make_req(f"occupy-{i}", text_len=4) for i in range(avail0 - 2)]
    assert sched.req_to_token_pool.alloc(occupy) is not None
    assert sched.req_to_token_pool.available_size() == 2

    quota1 = Scheduler.get_num_allocatable_reqs(sched, 0, running_batch=batch)
    assert quota1 == min(quota0, 2), (
        f"quota did not follow the ledger: quota0={quota0} quota1={quota1}"
    )


def test_prefill_admission_is_deterministic():
    """Same requests + same ledger => same batch. Twice, from scratch."""
    runs = []
    for _ in range(2):
        sched, _ = _fresh_stub()
        reqs = [make_req(f"req-{i}", text_len=20 + i * 8, max_new_tokens=8)
                for i in range(10)]
        admitted, _, _ = _run_admission(sched, reqs)
        runs.append(admitted)
    assert runs[0] == runs[1], f"admission not deterministic: {runs[0]} vs {runs[1]}"
    assert runs[0], "no request was admitted at all -- the harness is broken"


def test_prefill_admission_consumes_the_kv_ledger():
    sched, _ = _fresh_stub()
    reqs = [make_req(f"req-{i}", text_len=20 + i * 8, max_new_tokens=8)
            for i in range(10)]
    before = sched.token_to_kv_pool_allocator.available_size()
    admitted, running_batch, _ = _run_admission(sched, reqs)
    after = sched.token_to_kv_pool_allocator.available_size()

    assert before == KV_POOL_SIZE
    assert after < before, "admission consumed no KV pages"
    assert running_batch.batch_size() == len(admitted)

    admitted_tokens = sum(
        len(r.origin_input_ids) for r in reqs if r.rid in set(admitted)
    )
    assert before - after >= admitted_tokens, (
        f"consumed {before - after} pages for {admitted_tokens} prompt tokens"
    )


def test_admission_stops_when_pool_is_too_small():
    """A pool that cannot hold the first prompt must admit nobody."""
    model_config = build_model_config()
    clock = register.get_shared_clock()
    runner = MockModelRunner(model_config=model_config, device="cpu")
    sched = build_scheduler_stub(
        kv_pool_size=4,  # smaller than any prompt below
        req_pool_size=REQ_POOL_SIZE,
        model_config=model_config,
        mock_worker=MockWorker(model_runner=runner),
        clock=clock,
    )
    reqs = [make_req(f"big-{i}", text_len=64, max_new_tokens=8) for i in range(3)]
    admitted, _, _ = _run_admission(sched, reqs)
    assert admitted == [], f"admitted {admitted} into a 4-page pool"


# ───────────────────────── C5: back-pressure ─────────────────────────
def test_retract_decode_fires_when_kv_pool_is_starved():
    sched, _ = _fresh_stub()
    reqs = [make_req(f"req-{i}", text_len=20 + i * 8, max_new_tokens=8)
            for i in range(10)]
    admitted, running_batch, _ = _run_admission(sched, reqs)
    assert len(admitted) >= 2, "need at least 2 running reqs to retract one"

    running_batch.prepare_for_decode()
    assert running_batch.check_decode_mem() is True

    allocator = sched.token_to_kv_pool_allocator
    allocator.alloc(allocator.available_size() - 1)
    assert allocator.available_size() == 1
    assert running_batch.check_decode_mem() is False, (
        "a 1-page pool still reported that the next decode step fits"
    )

    before_bs = running_batch.batch_size()
    avail_before = allocator.available_size()
    retracted, new_ratio, _aborted = running_batch.retract_decode()

    assert len(retracted) >= 1, "retract_decode returned nothing on a starved pool"
    assert running_batch.batch_size() == before_bs - len(retracted)
    assert allocator.available_size() > avail_before, (
        "retraction did not give KV pages back"
    )
    assert new_ratio > 0


# ───────────────────────── C2: the four selection points ─────────────────
def test_execution_interception_yields_mock_runner():
    assert register.selftest_execution_shim() is True


def test_kv_interception_yields_cpu_pools():
    assert register.selftest_kv_shim() is True


def test_transfer_interception_yields_mock_engine():
    assert register.selftest_transfer_shim() is True


def test_clock_interception_is_virtual():
    assert register.selftest_clock_shim() is True


def test_virtual_clock_drives_the_real_waiting_timeout():
    """scheduler.py's _abort_on_waiting_timeout, unmodified, off the virtual clock.

    Behaviour assertion, not a print: the stale request must be dropped and
    the fresh one kept, with no wall-clock sleep.
    """
    from sglang.srt.managers.scheduler import Scheduler

    sched, clock = _fresh_stub()
    sent = []
    sched.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(
            send_output=lambda msg, req: sent.append(req.rid)
        )
    )
    stale = make_req("stale", text_len=5)
    stale.time_stats.wait_queue_entry_time = max(clock.monotonic(), 1.0)
    clock.advance(100.0)
    # fresh enters *after* the jump, so only stale is behind the deadline
    fresh = make_req("fresh", text_len=5)
    fresh.time_stats.wait_queue_entry_time = clock.monotonic()
    sched.waiting_queue = [stale, fresh]

    os.environ["SGLANG_REQ_WAITING_TIMEOUT"] = "10"
    try:
        Scheduler._abort_on_waiting_timeout(sched)
    finally:
        del os.environ["SGLANG_REQ_WAITING_TIMEOUT"]

    assert [r.rid for r in sched.waiting_queue] == ["fresh"]
    assert sent == ["stale"]


def test_waiting_timeout_is_off_by_default():
    """environ.py ships both timeouts at -1; nothing may be aborted then."""
    from sglang.srt.managers.scheduler import Scheduler

    sched, clock = _fresh_stub()
    sched.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=lambda msg, req: None)
    )
    stale = make_req("stale", text_len=5)
    stale.time_stats.wait_queue_entry_time = 1.0
    sched.waiting_queue = [stale]
    os.environ.pop("SGLANG_REQ_WAITING_TIMEOUT", None)
    clock.advance(100.0)
    Scheduler._abort_on_waiting_timeout(sched)
    assert [r.rid for r in sched.waiting_queue] == ["stale"]


def test_virtual_clock_drives_the_real_running_timeout():
    from sglang.srt.managers.scheduler import Scheduler

    sched, clock = _fresh_stub()
    reqs = [make_req(f"req-{i}", text_len=20 + i * 8, max_new_tokens=8)
            for i in range(4)]
    _admitted, running_batch, _ = _run_admission(sched, reqs)
    assert running_batch.batch_size() >= 1

    victim = running_batch.reqs[0]
    victim.time_stats.forward_entry_time = 1.0
    assert victim.to_finish is None

    os.environ["SGLANG_REQ_RUNNING_TIMEOUT"] = "10"
    try:
        clock.advance(100.0)
        Scheduler._abort_on_running_timeout(sched, running_batch)
    finally:
        del os.environ["SGLANG_REQ_RUNNING_TIMEOUT"]

    assert victim.to_finish is not None, "running timeout did not mark the victim"


# ───────────────────────── G2: backend name must stay non-Triton ─────────
def test_support_triton_semantics_are_what_the_shim_assumes():
    """If upstream changes this, the sim's backend choice stops being safe.

    support_triton(None) is True, i.e. *not* naming a backend selects the
    Triton kernel in alloc_for_extend -- which cannot launch without a GPU.
    """
    from sglang.srt.utils.common import support_triton

    assert support_triton(None) is True
    assert support_triton("torch_native") is False
    assert support_triton("intel_amx") is False
    assert support_triton("triton") is True


def test_alloc_for_extend_still_branches_on_support_triton():
    """The constraint only exists because this call site exists. Pin it."""
    import inspect

    from sglang.srt.mem_cache import allocation

    src = inspect.getsource(allocation)
    assert "support_triton(prefill_backend)" in src, (
        "alloc_for_extend no longer branches on support_triton -- recheck "
        "whether the backend-name constraint still applies"
    )


def test_sim_server_args_select_a_non_triton_backend():
    """Assert on the pair alloc_for_extend actually reads, not on the raw field.

    alloc_for_extend calls ``attention_backends()`` (runtime_context.py:1951),
    which falls back to ``attention_backend`` when the split prefill/decode
    fields are unset -- so that is what has to be non-Triton.
    """
    from sglang.srt.runtime_context import attention_backends
    from sglang.srt.utils.common import support_triton

    build_server_args()
    prefill, decode = attention_backends()
    assert prefill is not None, "sim must name a prefill backend, not leave it None"
    assert support_triton(prefill) is False, (
        f"prefill backend {prefill!r} routes KV writes through the Triton kernel"
    )
    assert decode is None or support_triton(decode) is False, (
        f"decode backend {decode!r} routes through the Triton kernel"
    )


# ───────────────────────── K3 full control-plane pass ─────────────────────
@pytest.fixture(scope="module")
def k3_run():
    """One full prefill->decode->finish pass over a K3-shaped model config."""
    from sglang.srt.sim import run_k3_sim

    return run_k3_sim.run()


def test_k3_model_config_is_mla_and_needs_no_checkpoint(k3_run):
    from sglang.srt.configs.model_config import AttentionArch

    mc = k3_run["model_config"]
    assert mc.hf_config.model_type == "kimi_k3"
    assert "KimiK3LinearForCausalLM" in mc.hf_config.architectures
    assert mc.attention_arch == AttentionArch.MLA
    assert mc.vocab_size == 163840
    # the config dir holds config.json and nothing else -- no weights on disk
    cfg_dir = Path(mc.model_path)
    assert sorted(p.name for p in cfg_dir.iterdir()) == ["config.json"]


def test_k3_full_loop_finishes_every_request(k3_run):
    reqs, finished = k3_run["reqs"], k3_run["finished"]
    assert len(finished) == len(reqs), (
        f"only {len(finished)}/{len(reqs)} finished: {finished}"
    )
    assert set(finished.values()) == {"FINISH_LENGTH"}
    assert k3_run["waiting_left"] == 0


def test_k3_output_length_matches_max_new_tokens(k3_run):
    """The REAL update_finish_state decided these --每条都必须正好停在预算上."""
    for req, (_prompt_len, max_new) in zip(k3_run["reqs"], k3_run["specs"]):
        assert len(req.output_ids) == max_new, (
            f"{req.rid}: output_len={len(req.output_ids)} but max_new_tokens={max_new}"
        )


def test_k3_loop_is_one_prefill_then_decodes(k3_run):
    modes = k3_run["modes"]
    assert modes[0] == "prefill"
    assert set(modes[1:]) == {"decode"}
    # steps == the longest request's budget: prefill emits token 1, then decodes
    assert k3_run["steps"] == max(m for _p, m in k3_run["specs"])


def test_k3_kv_accounting_is_exact(k3_run):
    """Pages consumed must equal prompt tokens + decode tokens, to the page."""
    prompt_tokens = sum(p for p, _m in k3_run["specs"])
    output_tokens = sum(len(r.output_ids) for r in k3_run["reqs"])
    # one page per prompt token at prefill; the prefill step emits the first
    # output token without a further page, so decode pages = outputs - reqs
    decode_pages = output_tokens - len(k3_run["reqs"])
    consumed = k3_run["kv_pool_size"] - k3_run["kv_free_end"]
    assert consumed == prompt_tokens + decode_pages, (
        f"consumed={consumed} prompt={prompt_tokens} decode={decode_pages}"
    )


def test_k3_logits_carry_the_real_vocab(k3_run):
    assert k3_run["logits_shape"][1] == k3_run["model_config"].vocab_size == 163840


def test_k3_full_loop_needs_only_inert_stub_extras(k3_run):
    """Record the stub surface the full loop needs; a jump means SGLang moved."""
    extras = k3_run["stub_extras"]
    assert len(extras) <= 12, f"full loop now needs {len(extras)} stub fields: {extras}"
    for required in ("dp_attn_adapter", "ngram_embedding_manager",
                     "prefill_decode_interval"):
        assert required in extras
