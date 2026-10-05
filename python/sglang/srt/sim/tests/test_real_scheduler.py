"""The page-31 flow on a Scheduler built by ``Scheduler.__init__``.

test_sim_interception.py drives SGLang's scheduling methods on a hand-built
stub, which leaves one question open: does the real constructor survive the
sim backend? These tests answer it by building a real ``Scheduler`` once and
asserting on what came out -- the classes, the pools, and the run.

The build costs a few seconds (gloo group, ModelConfig, pools), so both the
constructed scheduler and the run are module-scoped fixtures.
"""

from __future__ import annotations

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
    assert cap == 144
    assert sched.token_to_kv_pool_allocator.available_size() == cap


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
    assert len(run["rows"]) == 12


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
    assert run["modes"].count("prefill") == 4
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
    assert total == 144, f"pool does not close: {total}"
