"""A stand-in for torch_npu, so the NPU branch runs on a machine with no NPU.

Why this exists. ``is_npu()`` (utils/common.py:183) is ``hasattr(torch, "npu")``
plus ``torch.npu.is_available()``, and 174 call sites branch on it. Without
torch_npu every one of them takes the non-NPU path: a different attention
backend, different layer implementations, a different transfer engine --
that is, the control plane we exercise is *not* the one a 910 runs. The
sim's whole claim is that the control plane is SGLang's own, so it has to
be the NPU one.

What this is not. It does not implement Ascend kernels. Every
``torch_npu.<op>`` goes through one dispatch table:

  * ops listed in ``_CPU_OPS`` run a plain-torch equivalent on CPU -- these
    are the ones whose *shape and bookkeeping* the scheduler depends on;
  * everything else raises ``NpuOpNotInSim`` naming the op. It never returns
    a plausible-looking tensor it did not compute. A silently faked number
    is worse than a crash: it would be indistinguishable from a result.

Every call is counted (``op_counts()``), so a run can report exactly which
NPU entry points the branch reached.

Install it before importing anything under sglang.srt.layers -- 34 modules
hold ``_is_npu = is_npu()`` at module scope, and ``is_npu`` is lru_cached, so
the answer is frozen at first import. ``import sglang`` and
``import sglang.srt.sim`` touch neither (checked), so the runners can import
this module first and call install_fake_npu() before anything else.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from collections import Counter
from typing import Any, Callable, Dict

import torch

_counts: Counter = Counter()
_installed = False


class NpuOpNotInSim(NotImplementedError):
    """Raised when the NPU branch reaches an op this stand-in does not cover.

    The message carries the op name so the gap is a fact, not a mystery.
    """


def op_counts() -> Dict[str, int]:
    """Which NPU entry points this process actually reached, and how often."""
    return dict(_counts)


def reset_op_counts() -> None:
    _counts.clear()


# --------------------------------------------------------------------------
# The ops that have a plain-torch equivalent. Kept deliberately small: each
# one is here because the control plane reads its *shape* or its side effect,
# not because the sim claims to reproduce an Ascend kernel's arithmetic.
# --------------------------------------------------------------------------
def _npu_scatter_nd_update_(self: torch.Tensor, indices: torch.Tensor,
                            updates: torch.Tensor) -> torch.Tensor:
    """In-place scatter: the KV write. index_put_ is the torch spelling."""
    idx = tuple(indices[..., i] for i in range(indices.shape[-1]))
    self.index_put_(idx, updates.to(self.dtype))
    return self


def _npu_rms_norm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6):
    var = x.float().pow(2).mean(-1, keepdim=True)
    out = (x.float() * torch.rsqrt(var + epsilon)).to(x.dtype) * gamma
    return out, var


_CPU_OPS: Dict[str, Callable[..., Any]] = {
    "npu_scatter_nd_update_": _npu_scatter_nd_update_,
    "npu_rms_norm": _npu_rms_norm,
}

# Plumbing, not arithmetic: these install patches, set flags or hand back
# handles. Returning None changes nothing the sim measures, so a no-op here is
# honest in a way that a made-up tensor never is. Keep the two lists apart --
# the moment something that *computes* lands in here, the sim starts inventing
# numbers.
_NOOPS = {
    "_apply_patches",
    "npu_config",
    "set_option",
    "set_compile_mode",
    "init_dump",
    "finalize_dump",
}


class _StandInModule(types.ModuleType):
    """A torch_npu module (or submodule) whose attributes are counted calls.

    An attribute that is not in ``_CPU_OPS`` still *resolves* -- imports have
    to succeed for the branch to be exercised at all -- but calling it raises.
    The failure names the op, so "the NPU branch reached something we do not
    cover" is a fact with an address, not a mystery.
    """

    def __init__(self, fullname: str, concrete: Dict[str, Any] | None = None,
                 is_pkg: bool = False) -> None:
        super().__init__(fullname)
        self.__version__ = "sim-stand-in"
        self.__spec__ = importlib.machinery.ModuleSpec(
            fullname, loader=None, is_package=is_pkg
        )
        self.__file__ = "<sglang.srt.sim.fake_npu>"
        if is_pkg:
            self.__path__ = []  # a package, so torch_npu.x imports resolve
        for k, v in (concrete or {}).items():
            setattr(self, k, v)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        short = name
        impl = _CPU_OPS.get(short)
        label = f"{self.__name__}.{name}"
        noop = short in _NOOPS

        def _call(*args, **kwargs):
            _counts[label] += 1
            if noop:
                return None
            if impl is None:
                raise NpuOpNotInSim(
                    f"{label} was reached by the NPU branch and this stand-in "
                    f"does not implement it. Add a plain-torch equivalent to "
                    f"fake_npu._CPU_OPS, or route it to the functional "
                    f"executor -- do not return a made-up tensor."
                )
            return impl(*args, **kwargs)

        return _call


def _register(fullname: str, concrete: Dict[str, Any] | None = None,
              is_pkg: bool = False) -> "_StandInModule":
    """Put a stand-in in sys.modules *and* on its parent as an attribute.

    Both are needed: ``import a.b`` reads sys.modules, but ``a.b.c`` reads the
    attribute -- profile_utils.py:30 does the second with torch_npu.profiler.
    """
    mod = _StandInModule(fullname, concrete=concrete, is_pkg=is_pkg)
    sys.modules[fullname] = mod
    if "." in fullname:
        parent_name, _, child = fullname.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            object.__setattr__(parent, child, mod)
    return mod


class _StandInFinder:
    """Resolve any torch_npu.* import to a stand-in submodule.

    Hardcoding the submodule list would break the next time SGLang reaches for
    one; an import that fails would silently push the process back onto the
    non-NPU branch, which is the one thing this file exists to prevent.
    """

    ROOTS = ("torch_npu", "torchair", "sgl_kernel_npu", "mindspore", "mindie")

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root not in self.ROOTS:
            return None
        if fullname in sys.modules:
            return None
        return _register(fullname, is_pkg=True).__spec__


class _HcclOptions:
    """torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options, as far as
    parallel_state.py:101 uses it: an object it fills in and hands to
    init_process_group. The sim's groups are gloo, so nothing reads it back."""

    def __init__(self) -> None:
        self.hccl_config = {}
        self.global_ranks_in_group = []
        self.group_name = ""


class _ProcessGroupHCCL:
    Options = _HcclOptions


class _CompilerConfig:
    """torchair.configs.compiler_config.CompilerConfig, as far as
    get_compiler_backend (utils/common.py:1027-1043) uses it: a mode and a
    debug flag it sets, nothing it reads back."""

    def __init__(self) -> None:
        self.mode = "max-autotune"
        self.debug = types.SimpleNamespace(run_eagerly=False)


class _NullProfiler:
    """torch_npu.profiler.profile: the sim takes no device traces."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def start(self):
        return None

    def stop(self):
        return None

    def step(self):
        return None


_PROFILER_ATTRS = {
    "ProfilerActivity": types.SimpleNamespace(CPU="cpu", NPU="npu"),
    "ProfilerLevel": types.SimpleNamespace(Level0=0, Level1=1, Level2=2),
    "ExportType": types.SimpleNamespace(Text="text", Db="db"),
    "profile": _NullProfiler,
    "_ExperimentalConfig": lambda *a, **k: types.SimpleNamespace(),
    "tensorboard_trace_handler": lambda *a, **k: None,
    "schedule": lambda *a, **k: None,
}


class _Stream:
    def __init__(self, *a, **k):
        pass

    def synchronize(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Event:
    def __init__(self, *a, **k):
        pass

    def record(self, *a, **k):
        return None

    def synchronize(self):
        return None

    def elapsed_time(self, other):
        return 0.0


def _register_npu_device_type() -> None:
    """Make torch.device("npu:0") a legal device string.

    Real torch_npu registers Ascend as torch's PrivateUse1 backend and renames
    it to "npu"; distributed/parallel_state.py:339 builds that device string
    directly, so without the rename the branch dies on a string parse. Nothing
    is ever allocated on it here -- the sim's tensors are CPU tensors; this
    only makes the name legal.
    """
    try:
        torch.utils.rename_privateuse1_backend("npu")
    except Exception:
        pass


def _npu_namespace(total_bytes: int) -> types.SimpleNamespace:
    """torch.npu, with CPU semantics. Memory numbers are supplied, not probed:
    the sim's memory answer comes from the KV shim, not from a device query."""
    free = total_bytes

    return types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        current_device=lambda: 0,
        set_device=lambda *a, **k: None,
        synchronize=lambda *a, **k: None,
        empty_cache=lambda *a, **k: None,
        manual_seed_all=lambda *a, **k: None,
        get_device_name=lambda *a, **k: "Ascend910B2",
        get_device_properties=lambda *a, **k: types.SimpleNamespace(
            name="Ascend910B2", total_memory=total_bytes, multi_processor_count=24
        ),
        get_device_capability=lambda *a, **k: (0, 0),
        mem_get_info=lambda *a, **k: (free, total_bytes),
        memory_allocated=lambda *a, **k: 0,
        max_memory_allocated=lambda *a, **k: 0,
        reset_peak_memory_stats=lambda *a, **k: None,
        current_stream=lambda *a, **k: _Stream(),
        stream=lambda *a, **k: _Stream(),
        set_stream_limit=lambda *a, **k: None,
        Stream=_Stream,
        Event=_Event,
        NPUGraph=_Stream,
        graph=lambda *a, **k: _Stream(),
    )


def _neutralize_torch_compile() -> None:
    """Make @torch.compile an identity decorator for this process.

    On the NPU branch the import chain resolves the inductor backend while a
    class body is still being defined (kvfp4_tensor.py:66, reached from
    mem_cache/memory_pool.py:55), and this environment's triton is too old for
    torch._inductor.runtime.triton_compat to import at all:
    "module 'triton.language.core' has no attribute 'view'". The same import
    succeeds on the non-NPU branch, so it is the branch that drags inductor in.

    Compiling is irrelevant here either way: the sim never executes a compiled
    kernel -- the forward is intercepted. Declared as a stand-in rather than
    worked around silently.
    """
    real = torch.compile

    def _identity(model=None, **kwargs):
        if model is None:
            return lambda fn: fn
        return model

    _identity._sim_stand_in_for = real
    torch.compile = _identity


def _patch_get_device_module() -> None:
    """torch.get_device_module() has to hand back the npu module.

    Its no-argument form asks torch._C._get_accelerator(), which on a real
    Ascend box answers through C++ PrivateUse1 hooks that only torch_npu can
    register -- from Python it raises. parallel_state.py:111 calls it while a
    class body is being built (``stream: torch.get_device_module().Stream``),
    so the branch cannot even be imported without this.
    """
    real = torch.get_device_module

    def _get_device_module(device=None):
        if device is None or str(device).startswith("npu"):
            return torch.npu
        return real(device)

    torch.get_device_module = _get_device_module


_FACTORIES = ("ones", "zeros", "empty", "full", "arange", "tensor",
              "as_tensor", "eye", "randn", "rand", "zeros_like", "ones_like",
              "empty_like")


def _cpu_instead_of_npu(device):
    """npu:* -> cpu. Everything else untouched."""
    if device is None:
        return None
    if isinstance(device, torch.device):
        return torch.device("cpu") if device.type == "npu" else device
    if isinstance(device, str) and device.startswith("npu"):
        return "cpu"
    return device


def _redirect_device_allocations() -> None:
    """Allocate on CPU whatever the branch asks to allocate on the NPU.

    This is the sim's standing claim made literal: there is no device memory.
    The control plane allocates small tensors on the device as it comes up
    (parallel_state.py:398 builds an active-ranks vector, for one), and with
    PrivateUse1 registered but no kernels behind it those calls die.

    Only the *allocation* is redirected. torch.device("npu:0") still parses,
    still reports .type == "npu", and still compares as itself -- so the code
    under test keeps taking the NPU branch; what changes is where the bytes
    land, which is the one thing the sim never claimed to reproduce.
    """
    for name in _FACTORIES:
        real = getattr(torch, name, None)
        if real is None:
            continue

        def _wrap(_real=real):
            def _f(*args, **kwargs):
                if "device" in kwargs:
                    kwargs["device"] = _cpu_instead_of_npu(kwargs["device"])
                return _real(*args, **kwargs)

            return _f

        setattr(torch, name, _wrap())

    _real_to = torch.Tensor.to

    def _to(self, *args, **kwargs):
        if args and _cpu_instead_of_npu(args[0]) != args[0]:
            args = (_cpu_instead_of_npu(args[0]),) + args[1:]
        if "device" in kwargs:
            kwargs["device"] = _cpu_instead_of_npu(kwargs["device"])
        return _real_to(self, *args, **kwargs)

    torch.Tensor.to = _to
    torch.Tensor.npu = lambda self, *a, **k: self


def install_fake_npu(total_memory_bytes: int = 64 << 30,
                     neutralize_compile: bool = True) -> None:
    """Make is_npu() true without an NPU. Idempotent."""
    global _installed
    if _installed:
        return
    if neutralize_compile:
        _neutralize_torch_compile()
    if "torch_npu" not in sys.modules:
        _register("torch_npu", is_pkg=True)
        # The submodules SGLang imports by name and then reads attributes off
        # of; anything else is covered by the finder below.
        _register("torch_npu.multiprocessing", is_pkg=True)
        _register("torch_npu.multiprocessing.reductions",
                  concrete={"_rebuild_npu_tensor_original": lambda *a, **k: None})
        _register("torch_npu.profiler", concrete=_PROFILER_ATTRS, is_pkg=True)
        _register("torch_npu._C", concrete={"_weak_ref_tensor": lambda t: t},
                  is_pkg=True)
        _register("torch_npu._C._distributed_c10d",
                  concrete={"ProcessGroupHCCL": _ProcessGroupHCCL})
        _register("torch_npu.contrib", concrete={"transfer_to_npu": None},
                  is_pkg=True)
        # torchair is what the NPU branch asks for as its compile backend
        # (utils/common.py:1027). The sim never compiles, so the backend is a
        # name nothing calls -- but the import has to succeed or the branch
        # raises before any sim code runs.
        _register("torchair", concrete={"get_npu_backend": lambda **k: "eager"},
                  is_pkg=True)
        _register("torchair.configs", is_pkg=True)
        _register("torchair.configs.compiler_config",
                  concrete={"CompilerConfig": _CompilerConfig})
        sys.meta_path.insert(0, _StandInFinder())
    _register_npu_device_type()
    if not hasattr(torch, "npu"):
        torch.npu = _npu_namespace(total_memory_bytes)
    _patch_get_device_module()
    _redirect_device_allocations()
    sys.modules.setdefault("torch.npu", torch.npu)
    _installed = True


def assert_npu_branch() -> None:
    """Fail loudly if the process is not actually on the NPU branch.

    A sim that silently fell back to the CPU branch would still run and still
    print numbers -- and they would describe a control plane no 910 runs.
    """
    from sglang.srt.utils.common import is_npu

    if not is_npu():
        raise RuntimeError(
            "is_npu() is False: this process is on the non-NPU branch. "
            "install_fake_npu() has to run before anything under "
            "sglang.srt.layers is imported (34 modules freeze _is_npu at "
            "module scope, and is_npu is lru_cached)."
        )
