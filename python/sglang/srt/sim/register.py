"""Shims that occupy the four interception points with sim/mock classes.

Mechanism for all four: monkeypatch the *name* a selection point resolves at
call time (a module attribute, or a class attribute looked up through
``self``) -- never edit the selection point's own source line. This mirrors
exactly how the production code already varies behavior by platform (e.g.
``use_mlx()`` at scheduler.py:953): a name is resolved right before use, so
replacing what that name is bound to is sufficient.

Each ``install_*`` function patches one interception point and returns an
``unpatch`` callable. Each has a paired ``selftest_*`` function that calls
the REAL bound method of the REAL production class against a minimal stub
``self``, and asserts the sim object is what got constructed -- i.e. the
proof is "the real scheduler code path picked up our class", not "we built
an object that looks right in isolation".

================================================================================
Interception point                  | Real selection site           | Shim target
================================================================================
1. Execution (worker class)         | scheduler.py:956/960          | sglang.srt.managers.tp_worker.TpModelWorker
2. KV (pool + allocator)            | kv_cache_configurator.py       | KVCacheConfigurator.configure (the one
                                     | _build_token_to_kv_pool (dispatch)| method model_runner.py:878 actually calls;
                                     |                                | see FEASIBILITY.md for why the dispatch's
                                     |                                | internal branches at :1305/:1334/:1472/:1765
                                     |                                | are NOT independent selection points)
3. Transfer (PD engine)             | disaggregation/ascend/conn.py:42 | sglang.srt.disaggregation.ascend.conn.AscendTransferEngine
4. Clock (timeout deadlines)        | scheduler.py:1712 / :2999     | sglang.srt.managers.scheduler.time (module-rebind)
================================================================================

No line in any of these four files is edited. register.install() only
monkeypatches already-existing module/class attributes.
"""

from __future__ import annotations

import os

from types import SimpleNamespace
from typing import Callable, List

from sglang.srt.sim.virtual_clock import VirtualClock, patch_module_clock

_unpatchers: List[Callable[[], None]] = []


# ---------------------------------------------------------------------------
# 1. Execution interception: scheduler.py:956/960 worker-class selection
# ---------------------------------------------------------------------------
def _init_sim_parallel_state(
    rank: int = 0, world_size: int = 1, tp_size: int = 1,
    ep_size: int = 1, init_method: str = "tcp://127.0.0.1:29591",
) -> None:
    """A gloo process group, as the real worker would set up.

    The scheduler reads the TP group (scheduler.py:1107); a sim that skipped it
    would be hiding a real dependency. gloo needs no accelerator, so this works
    for one rank and for several -- several means several processes, one per
    rank, same as a real deployment.
    """
    import torch.distributed as dist
    from sglang.srt.distributed import parallel_state

    if parallel_state._TP is not None:
        return
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", init_method=init_method,
            world_size=world_size, rank=rank,
        )
    parallel_state.init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=rank,
        distributed_init_method=init_method, backend="gloo",
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=ep_size,
        backend="gloo",
    )


class SimTpModelWorker:
    """Drop-in for TpModelWorker's constructor signature (tp_worker.py:315-328).

    Built so ``Scheduler.init_tp_model_worker`` (scheduler.py:944-960), run
    unmodified, ends up with ``self.tp_worker`` being this class -- without
    scheduler.py ever importing ``sim``.
    """

    def __init__(self, server_args, gpu_id, ps, nccl_port, **kwargs):
        from sglang.srt.sim.mock_model_runner import MockModelRunner
        from sglang.srt.sim.mock_worker import MockWorker

        # Same thing the real TpModelWorker does: resolve the model config
        # from server_args. Only the weights are skipped -- a sim worker that
        # invented its own config would be deciding shapes for the scheduler.
        model_config = getattr(server_args, "_sim_model_config", None)
        if model_config is None:
            try:
                from sglang.srt.configs.model_config import ModelConfig

                model_config = ModelConfig.from_server_args(server_args)
            except Exception:
                model_config = SimpleNamespace(vocab_size=32000, context_len=8192)
        tp_size = int(getattr(server_args, "tp_size", 1) or 1)
        ep_size = int(getattr(server_args, "ep_size", 1) or 1)
        rank = int(getattr(ps, "tp_rank", 0) or 0)
        _init_sim_parallel_state(
            rank=rank, world_size=tp_size, tp_size=tp_size, ep_size=ep_size,
            init_method=os.environ.get(
                "SIM_DIST_INIT", "tcp://127.0.0.1:29591"
            ),
        )
        model_runner = MockModelRunner(
            model_config=model_config, device="cpu", ps=ps
        )
        self._mock_worker = MockWorker(
            model_runner=model_runner, server_args=server_args, gpu_id=gpu_id
        )

    def __getattr__(self, name):
        # Delegate everything to the wrapped MockWorker so an instance of
        # SimTpModelWorker is usable anywhere TpModelWorker would be.
        return getattr(self._mock_worker, name)


def install_execution_shim() -> Callable[[], None]:
    import sglang.srt.managers.tp_worker as tp_worker_mod

    original = tp_worker_mod.TpModelWorker
    tp_worker_mod.TpModelWorker = SimTpModelWorker

    def unpatch():
        tp_worker_mod.TpModelWorker = original

    return unpatch


def selftest_execution_shim() -> bool:
    """Calls the REAL Scheduler.init_tp_model_worker bound method."""
    from sglang.srt.managers.scheduler import Scheduler

    stub = Scheduler.__new__(Scheduler)
    stub.server_args = SimpleNamespace(_sim_model_config=None)
    stub.ps = SimpleNamespace(gpu_id=0)
    stub.nccl_port = 29500

    Scheduler.init_tp_model_worker(stub)

    ok = isinstance(stub.tp_worker, SimTpModelWorker)
    print(
        f"  [execution] Scheduler.init_tp_model_worker(stub) -> "
        f"tp_worker={type(stub.tp_worker).__name__}  model_runner="
        f"{type(stub.tp_worker.model_runner).__name__}  ok={ok}"
    )
    return ok


# ---------------------------------------------------------------------------
# 2. KV interception: the two device-bound steps inside
#    KVCacheConfigurator.configure (its only caller is model_runner.py:878).
#
#    Cut deliberately *inside* configure, not around it:
#      _profile_available_bytes     -- reads free device memory; pure backend
#      _resolve_memory_pool_config  -- budget -> token count, incl. the user cap
#                                      and page alignment; CONTROL PLANE, left
#                                      to SGLang
#      _derive_pool_sizes           -- arithmetic on the config; CONTROL PLANE,
#                                      left to SGLang, it feeds admission
#      _init_pools                  -- constructs the pool classes; pure backend
#    Replacing configure, or even _resolve_memory_pool_config, would have taken
#    the size derivation with it -- i.e. we would be deciding an admission
#    input, not adapting a backend.
# ---------------------------------------------------------------------------
def _sim_profile_available_bytes(self, pre_model_load_memory: int) -> int:
    """How many bytes are free for KV. The one genuinely device-bound step.

    Cut here rather than one level up at _resolve_memory_pool_config: that
    method is mostly arithmetic SGLang owns -- config_from_budget, the pool
    configurator, _apply_token_constraints (where --max-total-tokens lands),
    resolve_max_num_reqs. Taking the whole method would have meant the sim
    deciding the pool size, i.e. deciding an admission input. Taking only the
    probe leaves SGLang to turn a byte budget into a token count.

    The budget itself is a scenario knob: a run that wants admission to refuse
    sets --max-total-tokens, which SGLang then applies to whatever comes back
    from here.
    """
    return int(getattr(self.model_config, "_sim_kv_budget_bytes", 8 << 30))


def _sim_init_pools(self, *, sizes, req_to_token_pool, token_to_kv_pool_allocator):
    """Build CPU pools instead of device pools; sizes come from SGLang's own
    _derive_pool_sizes, so admission sees numbers SGLang derived."""
    from sglang.srt.mem_cache.kv_cache_configurator import _InitializedPools
    from sglang.srt.sim.cpu_kv import (
        build_cpu_token_to_kv_pool_allocator,
        build_req_to_token_pool,
    )

    max_context_len = getattr(self.model_config, "_sim_max_context_len", None) or (
        getattr(self.model_config, "context_len", 128)
    )
    allocator = build_cpu_token_to_kv_pool_allocator(size=sizes.max_total_num_tokens)
    return _InitializedPools(
        req_to_token_pool=req_to_token_pool
        or build_req_to_token_pool(
            self.model_config,
            size=sizes.max_running_requests or sizes.max_total_num_tokens,
            max_context_len=max_context_len,
        ),
        token_to_kv_pool=allocator.get_kvcache(),  # None -- no device KVCache in sim
        token_to_kv_pool_allocator=token_to_kv_pool_allocator or allocator,
    )


def install_kv_shim() -> Callable[[], None]:
    """Swap only the two device-bound steps: the memory probe and the pool
    constructors. configure(), the budget-to-tokens math, the user cap and the
    size derivation all stay SGLang's own code."""
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    import sglang.srt.mem_cache.kv_cache_configurator as kvc_mod

    originals = {
        "_profile_available_bytes": KVCacheConfigurator._profile_available_bytes,
        "_init_pools": KVCacheConfigurator._init_pools,
    }
    KVCacheConfigurator._profile_available_bytes = _sim_profile_available_bytes
    KVCacheConfigurator._init_pools = _sim_init_pools
    # configure() logs free device memory on the way out. That probe is a
    # device query like any other -- on a box with no accelerator it shells
    # out to lscpu and dies. Same treatment as the pools: swap the probe,
    # leave configure()'s own flow alone.
    original_probe = kvc_mod.get_available_gpu_memory
    kvc_mod.get_available_gpu_memory = lambda *a, **k: 0.0

    def uninstall() -> None:
        for name, fn in originals.items():
            setattr(KVCacheConfigurator, name, fn)
        kvc_mod.get_available_gpu_memory = original_probe

    return uninstall


class SpecStub:
    """configure() asks the spec algorithm whether it is none; sim has none."""

    @staticmethod
    def is_none() -> bool:
        return True


def selftest_kv_shim() -> bool:
    """Calls the REAL (now-patched) bound methods of the REAL configurator.

    Scope note: this used to call ``configure`` end to end against a
    hand-built ``self``. That worked only while the shim cut *above* the
    pool arithmetic. Now that the seam is just the memory probe, everything
    after it is SGLang's own math and needs a real ModelConfig -- so the
    end-to-end check moved to tests/test_real_scheduler.py, which builds a
    real Scheduler and asserts the pool size came out of SGLang's own
    _apply_token_constraints. What a stub can still prove is exactly what is
    checked here: the two shimmed steps are in place and do the backend job.
    """
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
    cfg.model_config = SimpleNamespace(
        _sim_kv_budget_bytes=4 << 20,
        _sim_max_context_len=64,
        hf_config=SimpleNamespace(
            architectures=["KimiK3LinearForCausalLM"], model_type="kimi_k3"
        ),
    )
    cfg.spec_algorithm = SpecStub()
    cfg.is_hybrid_swa = False
    cfg.is_draft_worker = False
    cfg.req_to_token_pool = None
    cfg.token_to_kv_pool_allocator = None
    cfg.device = "cpu"
    cfg.gpu_id = 0

    # step 1: the memory probe. Returns a byte budget -- NOT a token count,
    # which is the whole point of cutting here.
    budget = KVCacheConfigurator._profile_available_bytes(
        cfg, pre_model_load_memory=0
    )
    # step 2: pool construction, with sizes SGLang would have derived.
    sizes = SimpleNamespace(max_total_num_tokens=100, max_running_requests=100)
    pools = KVCacheConfigurator._init_pools(
        cfg, sizes=sizes, req_to_token_pool=None, token_to_kv_pool_allocator=None
    )

    ok = (
        budget == (4 << 20)
        and pools.token_to_kv_pool_allocator.device == "cpu"
        and pools.token_to_kv_pool_allocator.available_size() == 100
        and pools.req_to_token_pool.device == "cpu"
        and isinstance(pools.req_to_token_pool, ReqToTokenPool)
    )
    print(
        f"  [kv] _profile_available_bytes(stub) -> {budget} bytes; "
        f"_init_pools(stub) -> allocator.available_size()="
        f"{pools.token_to_kv_pool_allocator.available_size()} "
        f"req_to_token_pool={type(pools.req_to_token_pool).__name__} ok={ok}"
    )
    return ok


# ---------------------------------------------------------------------------
# 3. Transfer interception: disaggregation/ascend/conn.py:42
# ---------------------------------------------------------------------------
class MockTransferEngine:
    """Replaces AscendTransferEngine; no memfabric / NPU dependency."""

    def __init__(self, hostname, npu_id, disaggregation_mode):
        self.hostname = hostname
        self.npu_id = npu_id
        self.disaggregation_mode = disaggregation_mode
        self.registered = []

    def batch_register(self, ptrs, lens):
        self.registered.append((list(ptrs), list(lens)))


def install_transfer_shim() -> Callable[[], None]:
    import sglang.srt.disaggregation.ascend.conn as conn_mod

    original = conn_mod.AscendTransferEngine
    conn_mod.AscendTransferEngine = MockTransferEngine

    def unpatch():
        conn_mod.AscendTransferEngine = original

    return unpatch


def selftest_transfer_shim() -> bool:
    """Calls the REAL AscendKVManager.init_engine bound method."""
    from sglang.srt.disaggregation.ascend.conn import AscendKVManager

    stub = AscendKVManager.__new__(AscendKVManager)
    stub.kv_args = SimpleNamespace(gpu_id=0)
    stub.disaggregation_mode = SimpleNamespace(value="prefill")

    AscendKVManager.init_engine(stub)

    ok = isinstance(stub.engine, MockTransferEngine)
    print(f"  [transfer] AscendKVManager.init_engine(stub) -> engine={type(stub.engine).__name__} ok={ok}")
    return ok


# ---------------------------------------------------------------------------
# 4. Clock interception: scheduler.py:1712 / :2999 (time.perf_counter)
# ---------------------------------------------------------------------------
_shared_clock = VirtualClock()


# Modules whose ``time`` name the clock face covers. scheduler.py is where the
# timeout deadlines are read; req_time_stats.py is where SGLang stamps every
# per-request timestamp it later reports as queue / prefill / decode duration.
# Both have to be on the same clock, or the engine would time a forward that
# the sim decided takes 40 ms against a real wall clock and report ~0.
CLOCK_FACE_MODULES = (
    "sglang.srt.managers.scheduler",
    "sglang.srt.observability.req_time_stats",
)


def install_clock_shim() -> Callable[[], None]:
    import importlib

    originals = {}
    for name in CLOCK_FACE_MODULES:
        mod = importlib.import_module(name)
        originals[name] = mod.time
        patch_module_clock(mod, _shared_clock)

    def unpatch():
        for name, original in originals.items():
            importlib.import_module(name).time = original

    return unpatch


def forward_cost_hook(fn) -> None:
    """Install what decides how long a forward takes.

    This is the seam the real design fills from the offline cost library: it
    is handed the batch and returns seconds. Nothing here computes a duration
    from the model -- the number is supplied, which is the whole point.
    """
    import sglang.srt.sim.mock_worker as mw

    mw.FORWARD_COST = fn


def virtual_clock() -> "VirtualClock":
    return _shared_clock


def selftest_clock_shim() -> bool:
    """Calls scheduler.py's own time.perf_counter() through its module global."""
    import sglang.srt.managers.scheduler as scheduler_mod

    before = scheduler_mod.time.perf_counter()
    _shared_clock.advance(10.0)
    after = scheduler_mod.time.perf_counter()

    ok = (after - before) == 10.0
    print(
        f"  [clock] scheduler.time.perf_counter(): {before} -> {after} "
        f"(advanced only by explicit VirtualClock.advance(), no real sleep) ok={ok}"
    )
    return ok


# ---------------------------------------------------------------------------
def install() -> None:
    """Install all four shims. Idempotent-ish: call once per process."""
    _unpatchers.extend(
        [
            install_execution_shim(),
            install_kv_shim(),
            install_transfer_shim(),
            install_clock_shim(),
        ]
    )


def uninstall() -> None:
    while _unpatchers:
        _unpatchers.pop()()


def get_shared_clock() -> VirtualClock:
    return _shared_clock


def selftest_all() -> bool:
    results = [
        selftest_execution_shim(),
        selftest_kv_shim(),
        selftest_transfer_shim(),
        selftest_clock_shim(),
    ]
    return all(results)


if __name__ == "__main__":
    install()
    ok = selftest_all()
    print("\nALL FOUR INTERCEPTION POINTS OK:", ok)
