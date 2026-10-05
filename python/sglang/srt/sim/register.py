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

from types import SimpleNamespace
from typing import Callable, List

from sglang.srt.sim.virtual_clock import VirtualClock, patch_module_clock

_unpatchers: List[Callable[[], None]] = []


# ---------------------------------------------------------------------------
# 1. Execution interception: scheduler.py:956/960 worker-class selection
# ---------------------------------------------------------------------------
class SimTpModelWorker:
    """Drop-in for TpModelWorker's constructor signature (tp_worker.py:315-328).

    Built so ``Scheduler.init_tp_model_worker`` (scheduler.py:944-960), run
    unmodified, ends up with ``self.tp_worker`` being this class -- without
    scheduler.py ever importing ``sim``.
    """

    def __init__(self, server_args, gpu_id, ps, nccl_port, **kwargs):
        from sglang.srt.sim.mock_model_runner import MockModelRunner
        from sglang.srt.sim.mock_worker import MockWorker

        model_config = getattr(server_args, "_sim_model_config", None) or SimpleNamespace(
            vocab_size=32000, context_len=8192
        )
        model_runner = MockModelRunner(model_config=model_config, device="cpu")
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
#      _resolve_memory_pool_config  -- profiles GPU memory; pure backend, must go
#      _derive_pool_sizes           -- arithmetic on the config; CONTROL PLANE,
#                                      left to SGLang, it feeds admission
#      _init_pools                  -- constructs the pool classes; pure backend
#    Replacing configure wholesale would have taken the size derivation with it,
#    i.e. we would be changing scheduling inputs, not adapting a backend.
# ---------------------------------------------------------------------------
def _sim_resolve_memory_pool_config(self, pre_model_load_memory: int):
    """No device to profile: hand back the sim's pool size as a real config."""
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    size = getattr(self.model_config, "_sim_max_total_num_tokens", 256)
    return MemoryPoolConfig(max_total_num_tokens=size, max_running_requests=size)


def _sim_init_pools(self, *, sizes, req_to_token_pool, token_to_kv_pool_allocator):
    """Build CPU pools instead of device pools; sizes come from SGLang's own
    _derive_pool_sizes, so admission sees numbers SGLang derived."""
    from sglang.srt.mem_cache.kv_cache_configurator import _InitializedPools
    from sglang.srt.sim.cpu_kv import (
        build_cpu_req_to_token_pool,
        build_cpu_token_to_kv_pool_allocator,
    )

    max_context_len = getattr(self.model_config, "_sim_max_context_len", 128)
    allocator = build_cpu_token_to_kv_pool_allocator(size=sizes.max_total_num_tokens)
    return _InitializedPools(
        req_to_token_pool=req_to_token_pool
        or build_cpu_req_to_token_pool(
            size=sizes.max_total_num_tokens, max_context_len=max_context_len
        ),
        token_to_kv_pool=allocator.get_kvcache(),  # None -- no device KVCache in sim
        token_to_kv_pool_allocator=token_to_kv_pool_allocator or allocator,
    )


def install_kv_shim() -> Callable[[], None]:
    """Swap only the two device-bound steps; configure() and the size
    derivation stay SGLang's own code."""
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    import sglang.srt.mem_cache.kv_cache_configurator as kvc_mod

    originals = {
        "_resolve_memory_pool_config": KVCacheConfigurator._resolve_memory_pool_config,
        "_init_pools": KVCacheConfigurator._init_pools,
    }
    KVCacheConfigurator._resolve_memory_pool_config = _sim_resolve_memory_pool_config
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
    """Calls the REAL (now-patched) KVCacheConfigurator.configure bound method."""
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
    cfg.model_config = SimpleNamespace(
        _sim_max_total_num_tokens=100,
        _sim_max_context_len=64,
        # _derive_pool_sizes asks the model config what family this is
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
    result = KVCacheConfigurator.configure(cfg, pre_model_load_memory=0)

    ok = (
        result.max_total_num_tokens == 100
        and result.token_to_kv_pool_allocator.device == "cpu"
        and result.token_to_kv_pool_allocator.available_size() == 100
        and result.req_to_token_pool.device == "cpu"
    )
    print(
        f"  [kv] KVCacheConfigurator.configure(stub) -> "
        f"allocator.available_size()={result.token_to_kv_pool_allocator.available_size()} "
        f"req_to_token_pool.available_size()={result.req_to_token_pool.available_size()} "
        f"ok={ok}"
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


def install_clock_shim() -> Callable[[], None]:
    import sglang.srt.managers.scheduler as scheduler_mod

    original = scheduler_mod.time
    patch_module_clock(scheduler_mod, _shared_clock)

    def unpatch():
        scheduler_mod.time = original

    return unpatch


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
