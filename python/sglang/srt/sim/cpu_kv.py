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
        size=size, dtype=dtype, device="cpu", kvcache=None, need_sort=False
    )


def build_cpu_req_to_token_pool(size: int, max_context_len: int) -> ReqToTokenPool:
    """The real pool class, unmodified, device='cpu'."""
    return ReqToTokenPool(
        size=size,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
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
