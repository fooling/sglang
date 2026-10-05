"""The page-31 flow on a Scheduler built by ``Scheduler.__init__``.

test_sim_interception.py drives SGLang's scheduling methods on a hand-built
stub, which leaves one question open: does the real constructor survive the
sim backend? These tests answer it by building a real ``Scheduler`` once and
asserting on what came out -- the classes, the pools, and the run.

The build costs a few seconds (gloo group, ModelConfig, pools), so both the
constructed scheduler and the run are module-scoped fixtures.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture(scope="module")
def built():
    """One Scheduler for the whole module.

    It owns a gloo process group and ZMQ sockets, so building a second one in
    the same process is not just wasteful -- when the first build fails, the
    second blocks, and the suite hangs instead of reporting the failure. Build
    once, and turn a build failure into a red test here.
    """
    from sglang.srt.sim.run_k3_server_sim import build_real_scheduler

    try:
        return build_real_scheduler()
    except Exception as exc:  # noqa: BLE001 -- must not escape as a hang
        pytest.fail(f"Scheduler.__init__ did not survive the sim backend: {exc!r}")


@pytest.fixture(scope="module")
def sched(built):
    return built[0]


@pytest.fixture(scope="module")
def run(built):
    from sglang.srt.sim import run_k3_server_sim

    return run_k3_server_sim.run(prebuilt=built)


# ───────────────────────── what the constructor produced ─────────────────────


def test_it_is_sglangs_own_scheduler_class(sched):
    """Not a stub, not a subclass of ours: the module's own Scheduler."""
    from sglang.srt.managers.scheduler import Scheduler

    assert type(sched) is Scheduler
    assert type(sched).__module__ == "sglang.srt.managers.scheduler"


def test_the_worker_is_ours_but_the_scheduler_around_it_is_not(sched):
    """The seam is the worker. Everything holding it stays SGLang's."""
    assert type(sched.tp_worker).__module__.startswith("sglang.srt.sim")
    for attr in ("tree_cache", "req_to_token_pool", "token_to_kv_pool_allocator"):
        obj = getattr(sched, attr)
        assert obj is not None, attr
        assert type(obj).__module__.startswith("sglang.srt.mem_cache"), (
            f"{attr} is a {type(obj).__module__} object -- the pools must stay "
            "SGLang's own, only their device memory is ours"
        )


def test_k3_gets_a_hybrid_request_pool(sched):
    """K3 is hybrid: KDA layers need a state pool next to the MLA pages.

    This is not something the sim chose. mambaish_config(model_config) reads
    it off the config, and SGLang's own HybridReqToTokenPool is what comes
    back -- so a deck that describes only the MLA side is describing half.
    """
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    assert isinstance(sched.req_to_token_pool, HybridReqToTokenPool)
    assert type(sched.req_to_token_pool).__module__ == (
        "sglang.srt.mem_cache.memory_pool"
    )


def test_the_layer_pattern_is_1_based_and_counted_by_sglang(run):
    """How many layers are linear is SGLang's answer, not a number we typed.

    ``kda_layers`` is 1-BASED: is_kda_layer tests ``(layer_idx + 1) in
    kda_layers`` (configs/kimi_linear.py:172). Written 0-based the whole
    pattern shifts by one and the counts come out wrong while everything
    still runs -- which is exactly what happened before. So the counts here
    come from the config object's own linear_layer_ids /
    full_attention_layer_ids, and the ratio is asserted.
    """
    lin, full = run["linear_layer_ids"], run["full_attn_layer_ids"]
    n = run["num_hidden_layers"]
    assert len(lin) + len(full) == n
    assert not set(lin) & set(full)
    assert sorted(lin + full) == list(range(n))
    # one full-attention layer every FULL_ATTN_EVERY, and it is the last of
    # each group -- so 3 linear to 1 full
    from sglang.srt.sim.run_k3_sim import FULL_ATTN_EVERY

    assert full == [i for i in range(n) if (i + 1) % FULL_ATTN_EVERY == 0]
    assert len(lin) == len(full) * (FULL_ATTN_EVERY - 1)


def test_the_layer_count_is_the_config_classs_own_default():
    """32 layers because KimiLinearConfig says 32, not because we picked it."""
    from sglang.srt.configs.kimi_linear import KimiLinearConfig
    from sglang.srt.sim.run_k3_sim import K3_TEXT_CONFIG

    assert K3_TEXT_CONFIG["num_hidden_layers"] == (
        KimiLinearConfig().num_hidden_layers
    )


def test_a_real_config_json_can_be_dropped_in(tmp_path):
    """The example config is a stand-in, not a requirement.

    build_k3_model_config takes a path (or SIM_K3_CONFIG_JSON) and uses the
    file verbatim, so a released config.json needs no code change -- the
    layer pattern, dims and dtype all come off whatever it says.
    """
    import json

    from sglang.srt.sim.run_k3_sim import K3_TEXT_CONFIG, build_k3_model_config

    text = dict(K3_TEXT_CONFIG)
    text["num_hidden_layers"] = 12
    text["linear_attn_config"] = {
        **K3_TEXT_CONFIG["linear_attn_config"],
        "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11],
        "full_attn_layers": [4, 8, 12],
    }
    real = tmp_path / "config.json"
    real.write_text(json.dumps({
        "model_type": "kimi_k3",
        "architectures": ["KimiK3LinearForCausalLM"],
        "torch_dtype": "bfloat16",
        "text_config": text,
    }))

    mc, _d = build_k3_model_config(config_path=real)
    tc = mc.hf_text_config
    assert tc.num_hidden_layers == 12
    assert tc.full_attention_layer_ids == [3, 7, 11]
    assert len(tc.linear_layer_ids) == 9


def test_the_decision_methods_on_a_real_instance_are_sglangs(sched):
    """Same guarantee as the stub tests, now on the constructed object."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    owners = (
        (type(sched), "sglang.srt.managers.scheduler", (
            "get_next_batch_to_run",
            "get_new_batch_prefill",
            "update_running_batch",
            "run_batch",
            "process_batch_result",
            "handle_generate_request",
        )),
        # admission and retraction live on the batch, not the scheduler
        (ScheduleBatch, "sglang.srt.managers.schedule_batch", (
            "check_decode_mem",
            "retract_decode",
            "prepare_for_extend",
            "prepare_for_decode",
        )),
    )
    for cls, want_mod, names in owners:
        for name in names:
            fn = getattr(cls, name, None)
            assert fn is not None, f"{name} vanished from {cls.__name__}"
            assert fn.__module__ == want_mod, (
                f"{cls.__name__}.{name} now comes from {fn.__module__} -- that "
                "is a decision function, not a backend seam"
            )


def test_the_pool_size_is_sglangs_arithmetic_not_the_sims(sched):
    """The sim answers "how many bytes are free", never "how many tokens".

    Pool size is an admission input. The seam is _profile_available_bytes;
    everything after it -- the pool configurator, the --max-total-tokens cap,
    page alignment, _derive_pool_sizes -- is SGLang's. Proof: the cap set in
    ServerArgs is what came out the other end.
    """
    import sglang.srt.mem_cache.kv_cache_configurator as kvc_mod

    assert kvc_mod.KVCacheConfigurator._profile_available_bytes.__module__.startswith(
        "sglang.srt.sim"
    )
    for name in ("_resolve_memory_pool_config", "config_from_budget",
                 "_apply_token_constraints", "resolve_max_num_reqs",
                 "_derive_pool_sizes"):
        fn = getattr(kvc_mod.KVCacheConfigurator, name)
        assert fn.__module__ == "sglang.srt.mem_cache.kv_cache_configurator", (
            f"{name} now comes from {fn.__module__} -- that is pool sizing, "
            "which feeds admission"
        )
    cap = sched.server_args.max_total_tokens
    assert cap == 256
    # order-independent: whatever the run has consumed, the pages are all
    # still accounted for between the free list and the radix tree
    accounted = (sched.token_to_kv_pool_allocator.available_size()
                 + sched.tree_cache.evictable_size())
    assert accounted == cap, f"pool is {accounted}, cap is {cap}"


def test_overlap_is_off_by_config_not_by_patching(sched):
    """Overlap wants device streams, so the sim runs without it.

    The honest way to do that is the server arg SGLang already has. If this
    ever became a monkeypatch of run_batch, the test above would catch it;
    this one records that the flag is what did it.
    """
    assert sched.server_args.disable_overlap_schedule is True
    assert getattr(sched, "enable_overlap", False) is False


def test_arrivals_go_through_sglangs_own_receive_path(sched):
    """Requests arrive on the real socket, not by poking waiting_queue.

    The driver binds the endpoint the tokenizer manager normally binds; the
    scheduler had already connected its zmq.PULL in __init__. So recv_requests,
    the dispatcher and handle_generate_request are all SGLang's, and the Req
    objects are built by the scheduler rather than by the harness.
    """
    import zmq

    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.request_receiver import (
        SchedulerRequestReceiver,
    )

    # the socket is real and it is the scheduler that connected it
    assert isinstance(sched.recv_from_tokenizer, zmq.Socket)
    assert type(sched.request_receiver) is SchedulerRequestReceiver
    for cls, name, mod in (
        (SchedulerRequestReceiver, "recv_requests",
         "sglang.srt.managers.scheduler_components.request_receiver"),
        (SchedulerRequestReceiver, "_pull_raw_reqs",
         "sglang.srt.managers.scheduler_components.request_receiver"),
        (Scheduler, "process_input_requests", "sglang.srt.managers.scheduler"),
        (Scheduler, "handle_generate_request", "sglang.srt.managers.scheduler"),
    ):
        assert getattr(cls, name).__module__ == mod, (
            f"{cls.__name__}.{name} is no longer SGLang's -- the request path "
            "must not be reimplemented by the harness"
        )


def test_the_reqs_were_built_by_sglang(run):
    assert run["req_cls_module"] == "sglang.srt.managers.schedule_batch"


def test_concurrency_is_capped_by_the_state_pool_not_by_us(run, sched):
    """A hybrid model's concurrency limit comes off the state pool.

    64 state slots at 3 slots per request is 21 running requests -- SGLang's
    own arithmetic (resolve_max_num_reqs), and it is the request pool's real
    size, not the KV page count. Anyone sizing K3 concurrency has to use this
    number, so the deck must not quote the raw slot count.
    """
    assert sched.server_args.max_mamba_cache_size == 64
    assert run["max_running_requests"] == 21
    assert run["req_pool_size"] == 21
    assert run["req_pool_size"] != sched.server_args.max_mamba_cache_size


def test_the_output_path_runs_and_is_read_back(sched, run):
    """Outputs leave through SGLang's own streamer and are read off the socket.

    The scheduler's output socket is a PUSH that connects, so the driver binds
    the peer the tokenizer/detokenizer process would bind. completion_tokens
    and the finish reasons in the result come from those messages, which is
    why this is a path and not a claim.
    """
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )

    assert type(sched.output_streamer) is SchedulerOutputStreamer
    assert type(sched.output_streamer).stream_output.__module__ == (
        "sglang.srt.managers.scheduler_components.output_streamer"
    )
    assert run["outputs_were_read"] is True
    assert sorted(run["completion"]) == ["a0", "a1", "a2", "b0", "b1", "c0"]


def test_the_loop_itself_is_sglangs(sched, run):
    """event_loop_normal ran; the driver only observes through its own hook.

    The hook interface (on_run_batch / step) is what SGLang calls from inside
    run_batch and recv_requests, so using it does not displace any loop code.
    """
    from sglang.srt.managers.scheduler import Scheduler

    assert Scheduler.event_loop_normal.__module__ == "sglang.srt.managers.scheduler"
    assert Scheduler.on_idle.__module__ == "sglang.srt.managers.scheduler"
    # the run went through the loop, not through a hand-rolled while
    assert run["loop_iters"] >= len(run["rows"])
    assert run["stop_reason"] == "all finished"
    assert len(run["rows"]) == 11


# ──────────────────── the forward time slice: given, not computed ────────────


def test_the_time_slice_is_supplied_through_a_backend_seam():
    """The duration comes from the sim's own worker, not from SGLang.

    This is the interface the real design fills from the offline cost library:
    hand it the batch, get seconds back. Nothing in SGLang is replaced to make
    it work -- which is what makes it a backend mechanism rather than a patch.
    """
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.sim import mock_worker, register

    assert hasattr(register, "forward_cost_hook")
    assert callable(mock_worker.default_forward_cost)
    # the seam lives on the sim worker, which is itself the execution shim
    assert TpModelWorker.__module__.startswith("sglang.srt.sim")
    # and the cost is charged inside the sim's forward, nowhere else
    src = inspect.getsource(mock_worker._sim_forward_batch_generation)
    assert "_charge_forward_time(batch)" in src


def test_the_engine_accounts_the_forward_at_the_slice_it_was_given(run):
    """SGLang's own numbers add up to the slices, to the nanosecond.

    If anything else had moved the clock -- a real sleep, a stray wall-clock
    read -- these two would differ.
    """
    assert abs(run["clock_elapsed"] - run["clock_expected"]) < 1e-9
    expected = (run["prefill_slice_s"] * run["modes"].count("prefill")
                + run["decode_slice_s"] * run["modes"].count("decode"))
    assert abs(run["clock_elapsed"] - expected) < 1e-9
    # the last row's clock is the run's end, and the last finisher's
    # completion stamp -- taken by SGLang, not by us -- matches it
    last_clock = run["rows"][-1]["clock_after"]
    assert abs(max(t["completion"] for t in run["timing"].values()) - last_clock) < 1e-9


def test_the_queue_wait_the_engine_reports_is_the_deferral_we_caused(run):
    """b1 was deferred, and SGLang's own queue time says so in model time.

    b1 is admitted at the 8th forward; the slices burned between its arrival
    and that batch are what its queue wait has to be. This is the number a
    performance simulator exists to produce, and it comes out of the engine.
    """
    rows = run["rows"]
    admitted = next(i for i, r in enumerate(rows) if "b1" in r["rids"])
    # arrivals are delivered in step(), which runs before that iteration's
    # forward is charged -- so the queue clock starts at clock_before
    arrival_clock = next(
        r["clock_before"] for r in rows if "b1" in r["arrived"]
    )
    burned = rows[admitted]["clock_before"] - arrival_clock
    assert run["timing"]["b1"]["queued"] > 0
    assert abs(run["timing"]["b1"]["queued"] - burned) < 1e-9
    # and a request that was never deferred waited no model time at all
    assert run["timing"]["a0"]["queued"] == 0.0


def test_the_engine_spends_the_slice_in_its_own_step_ledger(run):
    """The slice is consumed, not just stored -- SGLang's own counters move.

    _record_step_counters (scheduler.py:4266) only accumulates when
    0 < step_us, where step_us is the gap between two consecutive launch
    timestamps. With a clock that never moved that was always 0, so the
    engine's step-time ledger was dead. The supplied slices make it live:
    the decode buckets below are microseconds of decode busy time, built out
    of the 8 ms we handed over.

    total_prefill_busy_us stays 0 on purpose -- that branch needs two
    back-to-back prefill forwards and this workload never has them, which is
    worth asserting so nobody reads the zero as a failure.
    """
    buckets = run["decode_moment_totals"]
    busy_us = [v for v in buckets if v >= 1000]
    assert busy_us, f"no decode busy time accumulated: {buckets}"
    slice_us = run["decode_slice_s"] * 1e6
    for v in busy_us:
        assert v % slice_us == 0, (
            f"{v} us is not a whole number of {slice_us} us decode slices"
        )
    assert sum(busy_us) <= run["clock_elapsed"] * 1e6
    assert run["total_prefill_busy_us"] == 0


def test_the_kv_write_operator_really_ran_for_every_forward(run):
    """The write path is exercised, not skipped -- the operator body is empty.

    One call per KV layer per forward, with the slots that forward allocated.
    An operator that is never invoked would leave this at zero, which is the
    failure a mock has to be able to detect about itself.
    """
    layers, forwards = run["kv_layers"], len(run["rows"])
    assert layers > 0
    assert run["kv_ops_called"].get("set_kv_buffer") == layers * forwards
    shapes = run["kv_write_shapes"]
    assert {s["layer"] for s in shapes} == set(range(layers))
    assert all(s["slots"] > 0 for s in shapes)
    # every layer sees the same slot count within one forward
    per_forward = [shapes[i:i + layers] for i in range(0, len(shapes), layers)]
    for group in per_forward:
        assert len({s["slots"] for s in group}) == 1


def test_the_kv_byte_count_is_real_so_a_transfer_can_be_costed(run):
    """bytes per token times layers is what a PD transfer would cost."""
    assert run["kv_bytes_per_token"] == 576 * 2      # MLA latent, fp16
    assert run["kv_layers"] == 8                     # full-attention layers only
    moved = run["kv_bytes_per_token"] * run["kv_layers"]
    assert moved == 9216                             # one token, all KV layers


def test_the_clock_face_is_observability_not_decision():
    """Why the slice needs nothing from SGLang's scheduling logic.

    The clock face rebinds the ``time`` name in two modules: the scheduler
    (where the timeout deadlines are read) and req_time_stats (where every
    per-request timestamp is taken). Only the two timeout gates feed a
    decision, and both default to disabled -- so with them off, the slice
    changes what the engine *reports* and nothing it *decides*. Verified by
    running with the face removed: the 11-row schedule came out identical and
    only the reported latencies changed (to real laptop time).
    """
    from sglang.srt.environ import envs
    from sglang.srt.sim import register

    assert register.CLOCK_FACE_MODULES == (
        "sglang.srt.managers.scheduler",
        "sglang.srt.observability.req_time_stats",
    )
    assert envs.SGLANG_REQ_WAITING_TIMEOUT.get() == -1
    assert envs.SGLANG_REQ_RUNNING_TIMEOUT.get() == -1


def test_the_per_row_timeline_needs_no_patch_at_all(run):
    """The timeline on the page is ours, built from the slices we handed over.

    It does not depend on the clock face: the row clocks come from the sim's
    own VirtualClock. So a deployment unwilling to rebind SGLang's time source
    still gets the timeline -- it just loses the engine's self-reported
    latencies.
    """
    rows = run["rows"]
    assert rows[0]["clock_before"] == 0.0
    assert all(
        rows[i]["clock_after"] == rows[i + 1]["clock_before"]
        for i in range(len(rows) - 1)
    )
    assert rows[-1]["clock_after"] == run["clock_elapsed"]


# ─────────────────────────────── what the run did ────────────────────────────


def test_every_request_finishes(run):
    assert len(run["finished"]) == len(run["workload"]) == 6
    assert run["waiting_left"] == 0


def test_both_finish_reasons_are_reached(run):
    """Length cutoff on five, a scripted EOS on a1 -- not one path twice.

    by_reason comes off the output messages (the engine's public wording);
    finished_from_reqs comes off the Req objects. Both are checked because
    they are two different records of the same event.
    """
    assert sorted(run["by_reason"]) == ["length", "stop"]
    assert run["by_reason"]["stop"] == ["a1"]
    assert run["finished_from_reqs"]["a1"] == "FINISH_MATCHED_TOKEN"
    assert sorted(set(run["finished_from_reqs"].values())) == [
        "FINISH_LENGTH", "FINISH_MATCHED_TOKEN"
    ]


def test_the_two_finish_records_reconcile(run):
    """What the engine published and what the Reqs hold must be the same set.

    A result dropped on the output path would otherwise pass unnoticed.
    """
    assert set(run["finished"]) == set(run["finished_from_reqs"])
    assert len(run["finished"]) == 6


def test_output_lengths_match_the_stopping_rule(run):
    """LENGTH requests stop exactly at max_new; a1 stops early at its EOS.

    The counts are the engine's own completion_tokens, read back off the
    output socket -- not something the harness counted.
    """
    from sglang.srt.sim.run_k3_sim import EOS_SCRIPT

    assert run["outputs_were_read"] is True
    for rid, _plen, max_new, _arrive in run["workload"]:
        got = run["completion"][rid]
        want = EOS_SCRIPT[rid] if rid in EOS_SCRIPT else max_new
        assert got == want, f"{rid}: completion_tokens={got} want={want}"


def test_the_waves_produce_more_than_one_prefill(run):
    """Three arrival waves must interleave with decode, not front-load."""
    assert run["modes"].count("prefill") == 3
    assert run["modes"].count("decode") == 8
    assert run["modes"][0] == "prefill"
    # a prefill after decode has started is what continuous batching means
    assert "prefill" in run["modes"][run["modes"].index("decode") :]


def test_admission_actually_refused_and_deferred(run):
    """The pool is smaller than the prompts, so someone must be made to wait.

    Without this the run would only show that everything fits -- which is the
    easy case and proves nothing about admission.
    """
    waits = run["waits"]
    assert max(waits) > 0, f"nobody was ever deferred: {waits}"
    # and the deferral resolves: a later prefill picks the waiters up
    first_wait = next(i for i, w in enumerate(waits) if w > 0)
    assert "prefill" in run["modes"][first_wait + 1 :], (
        "someone waited and no later prefill admitted them"
    )
    assert waits[-1] == 0 and run["waiting_left"] == 0


def test_kv_accounting_closes_at_the_end(run):
    """Freed pages plus what the radix tree still holds = the whole pool."""
    total = run["kv_free_end"] + run["radix_evictable"]
    assert run["kv_free_end"] > 0
    assert run["radix_evictable"] > 0
    assert total == 256, f"pool does not close: {total}"
