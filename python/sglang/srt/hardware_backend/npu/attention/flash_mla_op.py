"""Access to the FlashMLA custom op, which ships outside SGLang.

``flash_mla_with_kvcache`` lives in the CANN ``custom_transformer`` vendor
package, not in torch_npu, so it is only callable when **both** halves of that
package are present in the process:

1. the aicore binaries, made visible to aclnn by sourcing the vendor's env
   script (it exports ``ASCEND_CUSTOM_OPP_PATH``)::

       source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

   A vendor directory unpacked somewhere else works the same way -- source the
   ``bin/set_env.bash`` inside it (e.g. a deployment's own
   ``kimi-k3/vendors/custom_transformer/bin/set_env.bash``);

2. the Python wrapper ``cann_ops_transformer`` (shipped as
   ``cann_ops_transformer-1.0.0-py3-none-any.whl``), importable either by
   installing the wheel or by putting it on ``PYTHONPATH``.

The import is deferred to first use so that a process without the package can
still import the Ascend attention backend: CPU unit tests, a CUDA host, and any
NPU deployment that does not take the FlashMLA path (which is selected by
``SGLANG_NPU_USE_FIAS_V2_BSND`` together with DSPARK speculative decoding).
A missing package is reported once, with the two steps above, instead of
surfacing as a bare ``ModuleNotFoundError`` at backend import time.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

_SETUP_HINT = (
    "The FlashMLA custom op (cann_ops_transformer) is not importable. It ships "
    "outside SGLang and needs both halves of the CANN custom_transformer vendor "
    "package:\n"
    "  1. source the vendor env script so aclnn can find the kernels, e.g.\n"
    "     source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/"
    "custom_transformer/bin/set_env.bash\n"
    "     (or the bin/set_env.bash of the vendor directory your deployment "
    "unpacks, e.g. <deploy>/kimi-k3/vendors/custom_transformer/bin/set_env.bash)\n"
    "  2. make the python wrapper importable -- install "
    "cann_ops_transformer-1.0.0-py3-none-any.whl or add it to PYTHONPATH.\n"
    "Without it, run without the FlashMLA path: unset "
    "SGLANG_NPU_USE_FIAS_V2_BSND."
)

# (with_kvcache, metadata) once imported; None until the first attempt.
_ops: Optional[Tuple[Callable[..., Any], Callable[..., Any]]] = None
_import_error: Optional[BaseException] = None


def _load() -> Optional[Tuple[Callable[..., Any], Callable[..., Any]]]:
    """Import the op pair once, caching both success and failure."""
    global _ops, _import_error
    if _ops is not None or _import_error is not None:
        return _ops
    try:
        from cann_ops_transformer.ops.attention.flash_mla_with_kvcache import (
            flash_mla_with_kvcache as _with_kvcache,
            flash_mla_with_kvcache_metadata as _metadata,
        )
    except BaseException as exc:  # ImportError, or a load error from the vendor pkg
        _import_error = exc
        return None
    _ops = (_with_kvcache, _metadata)
    return _ops


def is_flash_mla_available() -> bool:
    """True when the FlashMLA op can be called in this process."""
    return _load() is not None


def require_flash_mla(reason: str) -> None:
    """Fail at startup, not mid-forward, when a config needs the op.

    ``reason`` names the configuration that selected the FlashMLA path.
    """
    if _load() is not None:
        return
    raise RuntimeError(f"{reason}\n{_SETUP_HINT}") from _import_error


def _get(index: int, name: str) -> Callable[..., Any]:
    ops = _load()
    if ops is None:
        raise RuntimeError(f"{name} is unavailable.\n{_SETUP_HINT}") from _import_error
    return ops[index]


def flash_mla_with_kvcache(*args, **kwargs):
    """Paged FlashMLA attention; see the vendor package for the signature."""
    return _get(0, "flash_mla_with_kvcache")(*args, **kwargs)


def flash_mla_with_kvcache_metadata(*args, **kwargs):
    """Host-side tiling metadata consumed by :func:`flash_mla_with_kvcache`."""
    return _get(1, "flash_mla_with_kvcache_metadata")(*args, **kwargs)
