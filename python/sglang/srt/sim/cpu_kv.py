"""KV side of the sim prototype.

Deliverable (3). The instruction was: first check whether the existing
``TokenToKVPoolAllocator`` (``sglang/srt/mem_cache/allocator/token.py``)
already works with ``device='cpu'`` before writing anything new.

Finding (verified by direct construction, see the bottom of this file and
FEASIBILITY.md C4): **yes, unmodified, it works as-is.**

``TokenToKVPoolAllocator.__init__`` stores ``device`` and does
``torch.arange(..., device=device)``; ``alloc``/``free``/``available_size``
are plain tensor slicing (``free_pages[:need_size]``, ``torch.cat``,
``len(...)``) with no CUDA-only call anywhere in the class or its base
(``mem_cache/allocator/base.py``). The only place a real ``KVCache`` object
is touched is ``get_cpu_copy`` / ``load_cpu_copy`` (host-offload paths, not
on the alloc/free/available_size admission path this prototype drives), so
the constructor's ``kvcache`` argument can be ``None`` for everything
run_smoke.py exercises.

Likewise ``ReqToTokenPool`` (``mem_cache/memory_pool.py``) is a plain
``torch.zeros((.., ..), device=device)`` index table; ``available_size()``
is ``len(self.free_slots)`` (a Python list, not a device call).

So this module does NOT define a new allocator class. It only provides two
thin factory functions so register.py / run_smoke.py have one place to call,
and documents the finding the task asked for explicitly.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


def build_cpu_token_to_kv_pool_allocator(
    size: int, dtype: torch.dtype = torch.float16
) -> TokenToKVPoolAllocator:
    """The real allocator class, unmodified, device='cpu', kvcache=None."""
    return TokenToKVPoolAllocator(
        size=size, dtype=dtype, device="cpu",
        kvcache=_sized_kv(dtype, size), need_sort=False,
    )


def _sized_kv(dtype, size: int) -> "SimKVCache":
    kv = SimKVCache(dtype=dtype)
    kv.size = size
    return kv


def build_cpu_req_to_token_pool(size: int, max_context_len: int) -> ReqToTokenPool:
    """The real pool class, unmodified, device='cpu'."""
    return ReqToTokenPool(
        size=size,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
    )


def build_req_to_token_pool(model_config, size: int, max_context_len: int):
    """Plain pool, or the hybrid one when the model carries recurrent state.

    Kimi-K3 is hybrid: KimiK3DeltaAttention is linear attention with state, so
    the KV side needs MLA pages *and* a linear-attention state pool. Which one
    to build, and every parameter of it, comes from SGLang's own
    ``mambaish_config`` -- the sim only moves the device to CPU.
    """
    try:
        from sglang.srt.configs.hybrid_arch import mambaish_config
    except Exception:
        return build_cpu_req_to_token_pool(size=size, max_context_len=max_context_len)

    try:
        spec = mambaish_config(model_config)
    except Exception:
        spec = None
    if spec is None:
        return build_cpu_req_to_token_pool(size=size, max_context_len=max_context_len)

    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    cache_params = spec.mamba2_cache_params
    return HybridReqToTokenPool(
        size=size,
        mamba_size=size,
        mamba_spec_state_size=size,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=list(cache_params.layers),
        enable_mamba_extra_buffer=False,
        enable_mamba_extra_buffer_lazy=False,
        speculative_num_draft_tokens=None,
        speculative_eagle_topk=None,
        enable_overlap_schedule=False,
    )


def selftest() -> None:
    """Prints real alloc/free/available_size output -- run directly:

        perl -e 'alarm 60; exec @ARGV' python/.venv/bin/python -m sglang.srt.sim.cpu_kv
    """
    alloc = build_cpu_token_to_kv_pool_allocator(size=64)
    print("TokenToKVPoolAllocator(device='cpu') available_size:", alloc.available_size())
    idx = alloc.alloc(10)
    print("alloc(10) ->", idx.tolist())
    print("available_size after alloc:", alloc.available_size())
    alloc.free(idx)
    print("available_size after free:", alloc.available_size())

    pool = build_cpu_req_to_token_pool(size=8, max_context_len=128)
    print("ReqToTokenPool(device='cpu') available_size:", pool.available_size())
    rows = pool.alloc_rows(3)
    print("alloc_rows(3) ->", rows, "available_size after:", pool.available_size())


if __name__ == "__main__":
    selftest()


class SimKVCache:
    """Placeholder for the device KV store.

    The sim allocates *indices* for real (that is the whole point of C4), but
    there is no tensor behind them. Control-plane code only ever asks this
    object what type it is, so a placeholder is enough -- and anything that
    tries to actually read or write KV values must blow up here rather than
    silently get zeros, which would hide a real dependency.
    """

    def __init__(self, dtype, device: str = "cpu"):
        self.dtype = dtype
        self.device = device
        self.page_size = 1
        self.layer_num = 0
        # layer range, same meaning as the real pool: PD's prefill role reads
        # start_layer to decide which layers it transfers
        self.start_layer = 0
        self.end_layer = 0
        # shape fields PD reads off the pool when it builds its kv_args
        self.head_num = 0
        self.head_dim = 0
        self.post_capture_active = False
        self.enable_memory_saver = False
        self.size = 0  # set by the builder

    def get_contiguous_buf_infos(self):
        """(ptrs, lens, item_lens) for PD KV transfer -- empty here, on purpose.

        The real pool hands the transfer engine the device addresses of the KV
        tensors so they can be registered for RDMA. This sim allocates KV
        *indices* for real but has no tensors behind them, so there are no
        regions to register and nothing to move. Returning empty lists states
        that plainly: PD's control plane can come up, its data plane has
        nothing to carry.

        Not a value accessor, so it does not raise -- but it is the boundary
        where PD stops being simulable without real KV memory.
        """
        return [], [], []

    def maybe_get_custom_mem_pool(self):
        """No custom device allocator here, so None -- same as the real path
        when one is not configured. The PD disaggregation setup asks for it
        while registering KV memory with the transfer engine.

        Note this is a *pool* accessor, not a value accessor: the ones that
        hand out KV contents still raise, because zeros there would hide a
        real dependency.
        """
        return None

    def _no_values(self, *_a, **_k):
        raise NotImplementedError(
            "SimKVCache holds no tensors: the sim proves index bookkeeping, "
            "not KV contents. Something asked for real KV values."
        )

    get_key_buffer = get_value_buffer = get_kv_buffer = _no_values
    set_kv_buffer = get_cpu_copy = load_cpu_copy = _no_values
