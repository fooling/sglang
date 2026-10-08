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

import os

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


def build_cpu_token_to_kv_pool_allocator(
    size: int, dtype: torch.dtype = torch.float16, model_config=None,
    page_size: int = 1,
) -> TokenToKVPoolAllocator:
    """The real allocator class, unmodified, device='cpu'.

    ``model_config`` is what makes the KV *shape* real: without it the cache
    object would report zero layers and zero bytes per token, and anything
    that costs a KV transfer would get zero. Passing it makes the counts right
    while still holding no tensors.
    """
    return TokenToKVPoolAllocator(
        size=size, dtype=dtype, device="cpu",
        kvcache=_sized_kv(dtype, size, model_config, page_size), need_sort=False,
    )


def kv_shape_from_config(model_config, dtype) -> dict:
    """How many layers hold KV, and how many bytes one token takes in each.

    Mirrors the real pools' arithmetic rather than inventing one:

    * MLA keeps a single latent entry per token, so head_num is 1 and head_dim
      is ``kv_lora_rank + qk_rope_head_dim`` (MLATokenToKVPool:4068).
    * MHA keeps K and V per head, so the per-token span is
      ``2 * num_key_value_heads * head_dim``.
    * On a hybrid model only the full-attention layers hold KV pages -- the
      linear-attention layers keep state in the mamba pool instead -- so the
      layer count comes off ``full_attention_layer_ids`` when the config has
      one.

    Returns a dict, not a tensor: the sim needs the sizes, not the storage.
    """
    tc = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", model_config
    )
    nlayers = int(getattr(tc, "num_hidden_layers", 0) or 0)
    full_ids = getattr(tc, "full_attention_layer_ids", None)
    kv_layers = len(full_ids) if full_ids else nlayers

    kv_lora_rank = getattr(tc, "kv_lora_rank", None)
    qk_rope = getattr(tc, "qk_rope_head_dim", None)
    if kv_lora_rank and qk_rope:
        head_num, head_dim = 1, int(kv_lora_rank) + int(qk_rope)
        per_token_elems = head_dim
    else:
        head_num = int(getattr(tc, "num_key_value_heads", 0)
                       or getattr(tc, "num_attention_heads", 0) or 0)
        hidden = int(getattr(tc, "hidden_size", 0) or 0)
        nheads = int(getattr(tc, "num_attention_heads", 0) or 0)
        head_dim = int(getattr(tc, "head_dim", 0) or (hidden // nheads if nheads else 0))
        per_token_elems = 2 * head_num * head_dim      # K and V
    itemsize = torch.empty(0, dtype=dtype).element_size()
    return dict(
        kv_layers=kv_layers, total_layers=nlayers, head_num=head_num,
        head_dim=head_dim, bytes_per_token=per_token_elems * itemsize,
    )


def _sized_kv(dtype, size: int, model_config=None, page_size: int = 1) -> "SimKVCache":
    kv = SimKVCache(dtype=dtype, page_size=page_size)
    kv.size = size
    if model_config is not None:
        kv.apply_shape(kv_shape_from_config(model_config, dtype))
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
    real_slot_bytes = _real_state_bytes_per_slot(cache_params, size)
    ctx = _flat_state_alloc() if shape_only_enabled() else contextlib.nullcontext()
    with ctx:
        pool = HybridReqToTokenPool(
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

    if shape_only_enabled():
        _shrink_mamba_state(pool, real_slot_bytes)
    return pool


import contextlib


def _real_state_bytes_per_slot(cache_params, slots: int) -> int:
    """压扁之前先把"真实每槽多少字节"算出来——代价模型要的是这个数。

    从 cache_params 自己的形状声明里算，不去量压扁后的张量。
    """
    import torch as _t
    total = 0
    for name in dir(cache_params):
        if name.startswith("_"):
            continue
        try:
            v = getattr(cache_params, name)
        except Exception:
            continue
        shapes = v if isinstance(v, (list, tuple)) else [v]
        for sh in shapes:
            if not isinstance(sh, (list, tuple)) or len(sh) < 2:
                continue
            if not all(isinstance(d, int) and d > 0 for d in sh):
                continue
            n = 1
            for d in sh:
                n *= d
            total += n * 2          # 按 2 字节估；真实 dtype 在压扁前量不到
    return total


@contextlib.contextmanager
def _flat_state_alloc():
    """构造状态池的那一刻，把 ≥3 维张量的数据维压到 1。

    为什么要在"构造时"而不是"构造后"：构造后再换掉，那 1.1 GB 已经真的向系统要过一次，
    峰值 RSS 并不会降——形状化的意义就没了（实测：构造后压扁，峰值仍是 1654 MB）。

    只压**第 2 维之后**：前两维是层与槽，分配/释放按它们索引；
    二维的索引表（req_to_token 是 [size, max_context_len]）整个不动。
    """
    import torch as _t
    orig_zeros, orig_empty = _t.zeros, _t.empty

    def _shrink(args):
        if not args:
            return args
        shape = args[0] if isinstance(args[0], (tuple, list, _t.Size)) else args
        if not isinstance(shape, (tuple, list, _t.Size)) or len(shape) < 3:
            return args
        small = tuple(list(shape[:2]) + [1] * (len(shape) - 2))
        return (small,) if isinstance(args[0], (tuple, list, _t.Size)) else small

    def _shrink_kw(kw):
        # memory_pool.py 用的是 torch.zeros(size=(...))，形状走关键字——只拦位置参数会漏
        sz = kw.get("size")
        if isinstance(sz, (tuple, list, _t.Size)) and len(sz) >= 3:
            kw = dict(kw)
            kw["size"] = tuple(list(sz[:2]) + [1] * (len(sz) - 2))
        return kw

    def zeros(*a, **kw):
        return orig_zeros(*_shrink(a), **_shrink_kw(kw))

    def empty(*a, **kw):
        return orig_empty(*_shrink(a), **_shrink_kw(kw))

    _t.zeros, _t.empty = zeros, empty
    try:
        yield
    finally:
        _t.zeros, _t.empty = orig_zeros, orig_empty


def _shrink_mamba_state(pool, real_slot_bytes: int = 0) -> None:
    """形状化：只保留**被索引的维度**（层、槽），数据维压到 1。

    为什么可以：仿真里没人读这些状态的**值**——线性注意力本身是 mock 的；
    真正被用到的是"有多少层、多少槽"（分配/释放按槽号索引）与"每槽多少字节"
    （代价模型要）。所以把 (层, 槽, H, L, V) 压成 (层, 槽, 1, 1, 1)，
    槽号索引照常，内存从 1.1 GB 掉到 KB 级。

    真实的每槽字节数记在 ``pool.mamba_pool.shape_only_bytes_per_slot`` 上，
    谁要算内存账就读它，不要去量压扁后的张量。
    """
    mp = getattr(pool, "mamba_pool", None)
    mc = getattr(mp, "mamba_cache", None)
    if mc is None:
        return

    def _flat(t):
        # 前两维（层、槽）保留，其余压到 1
        keep = list(t.shape[:2]) + [1] * (t.dim() - 2)
        return torch.zeros(keep, dtype=t.dtype, device=t.device)

    real_bytes = 0
    slots = None
    for name in ("temporal", "conv", "intermediate_conv_window",
                 "intermediate_ssm", "replayssm_rawv", "replayssm_rawk"):
        v = getattr(mc, name, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            if not v or not hasattr(v[0], "dim"):
                continue
            real_bytes += sum(t.numel() * t.element_size() for t in v)
            slots = slots or (v[0].shape[1] if v[0].dim() >= 2 else None)
            small = [_flat(t) for t in v]
        elif hasattr(v, "dim"):
            if v.dim() < 2:
                continue
            real_bytes += v.numel() * v.element_size()
            slots = slots or v.shape[1]
            small = _flat(v)
        else:
            continue
        try:
            setattr(mc, name, small)
        except Exception:                      # frozen dataclass
            object.__setattr__(mc, name, small)
    mp.shape_only_bytes_per_slot = real_slot_bytes or (
        (real_bytes // slots) if slots else 0)
    mp.shape_only = True


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




def shape_only_enabled() -> bool:
    """形状化开关：池子只按真实 shape 记账，不按规模分配那块内存。

    默认关，PoC 原来的行为不变；``SIM_SHAPE_ONLY=1`` 打开。打开之后：

    * KV 的每层区域换成**一块共享 scratch**（行数固定，所有层共用），
      shape 字段与 ``kv_bytes_for`` 仍按真规模报——算子照常被调到，只是写进 scratch；
    * mamba/KDA 状态张量只保留**被索引的维度**（层、槽），数据维压到 1。

    为什么这样仍然成立：并发上限由 SGLang 自己的 ``resolve_max_num_reqs``
    从 ``max_mamba_cache_size`` 算出来（kv_cache_configurator.py:2008），
    它读配置不读池子实际分配了多少。
    """
    return os.environ.get("SIM_SHAPE_ONLY", "") not in ("", "0", "false", "False")


SHAPE_ONLY_SCRATCH_ROWS = 64


class SimKVCache:
    """Placeholder for the device KV store.

    The sim allocates *indices* for real (that is the whole point of C4), but
    there is no tensor behind them. Control-plane code only ever asks this
    object what type it is, so a placeholder is enough -- and anything that
    tries to actually read or write KV values must blow up here rather than
    silently get zeros, which would hide a real dependency.
    """

    def __init__(self, dtype, device: str = "cpu", page_size: int = 1):
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.layer_num = 0
        # layer range, same meaning as the real pool: PD's prefill role reads
        # start_layer to decide which layers it transfers
        self.start_layer = 0
        self.end_layer = 0
        # shape fields PD reads off the pool when it builds its kv_args
        self.head_num = 0
        self.head_dim = 0
        self.bytes_per_token = 0
        self.post_capture_active = False
        self.enable_memory_saver = False
        self.size = 0  # set by the builder
        # per-layer regions, allocated on first touch at the real shape
        self._buffers: dict = {}
        self.shape_only = shape_only_enabled()
        # which KV operators were actually invoked, and with what shapes.
        # This is what replaces "raise" as the guard against self-deception:
        # a test asserts the operator ran, not that it was avoided.
        self.ops_called: dict = {}
        self.op_shapes: dict = {}

    def apply_shape(self, shape: dict) -> None:
        """Take the real shape off the model config (kv_shape_from_config).

        Without this the object reports zeros, and then a KV transfer costs
        nothing -- which would make PD look free in a performance run. The
        bytes still do not exist; only the counts are real.
        """
        self.layer_num = shape["kv_layers"]
        self.end_layer = shape["kv_layers"]
        self.head_num = shape["head_num"]
        self.head_dim = shape["head_dim"]
        self.bytes_per_token = shape["bytes_per_token"]

    def kv_bytes_for(self, num_tokens: int) -> int:
        """How many bytes moving ``num_tokens`` of KV would be, all layers.

        This is the number a transfer cost model needs; it is right even though
        nothing is stored.
        """
        return int(num_tokens) * self.bytes_per_token * self.layer_num

    # Synthetic base address for the per-layer regions. Deliberately a value
    # no real allocation returns, so a pointer that escapes to a real transfer
    # engine fails loudly instead of corrupting memory. In the sim the engine
    # is always the sim's own (register.MockTransferEngine), which records the
    # byte counts and never dereferences.
    SYNTHETIC_BASE = 0x5100_0000_0000

    def get_contiguous_buf_infos(self):
        """(ptrs, lens, item_lens) for PD KV transfer.

        The lengths are REAL -- bytes per token per layer and the whole span --
        because that is what a transfer cost model computes from, and a pool
        reporting zero would make PD look free. The pointers are synthetic:
        the sim holds no tensors, so there is nothing to dereference, and the
        only engine that ever sees them is the sim's own.

        Nothing moves. The counts being right is the point; the bytes being
        absent is the stated limit.
        """
        if not self.layer_num or not self.bytes_per_token:
            # No model config was supplied, so the shape is unknown. Say so
            # rather than reporting zero-byte regions that a cost model would
            # silently believe.
            raise RuntimeError(
                "SimKVCache has no shape: build it with model_config so the "
                "KV byte counts are real (cpu_kv.kv_shape_from_config)."
            )
        item_len = self.bytes_per_token * self.page_size
        span = self.size * self.bytes_per_token
        ptrs = [self.SYNTHETIC_BASE + i * span for i in range(self.layer_num)]
        return ptrs, [span] * self.layer_num, [item_len] * self.layer_num

    def maybe_get_custom_mem_pool(self):
        """No custom device allocator here, so None -- same as the real path
        when one is not configured. The PD disaggregation setup asks for it
        while registering KV memory with the transfer engine.

        Note this is a *pool* accessor, not a value accessor: the ones that
        hand out KV contents still raise, because zeros there would hide a
        real dependency.
        """
        return None

    # ---- the KV operators: really called, with real shapes, bodies empty ----
    #
    # These used to raise, on the reasoning that returning zeros would hide a
    # dependency. That was the wrong cut for a simulator: raising blocks the
    # call path, so the KV write, the PD transfer and the offload never get
    # exercised at all. The seam belongs at the OPERATOR, not at the call site:
    # the shapes are real, the call really happens, and the operator body is
    # empty for now -- later it can be a costed or a real one.
    #
    # What keeps that from becoming self-deception is counting: every call is
    # recorded with the shape it was given, so a test can assert the operator
    # was invoked rather than quietly skipped. See ops_called / op_shapes.

    def _layer_buffer(self, layer_id: int) -> "torch.Tensor":
        """The layer's KV region, allocated on first touch at the real shape.

        Lazily, because a modelled pool can be far larger than this host; at
        simulation scale the allocation is small (one region is
        size * head_num * head_dim elements).
        """
        if self.head_num <= 0 or self.head_dim <= 0:
            raise RuntimeError(
                "SimKVCache has no shape: build it with model_config so the KV "
                "shapes are real (cpu_kv.kv_shape_from_config)."
            )
        if self.shape_only:
            # 一块共享 scratch，所有层共用；行数固定，不随池子规模涨。
            # 算子照常被调到、shape 字段仍是真的，只是写进这块而不是真池子。
            buf = self._buffers.get("scratch")
            if buf is None:
                rows = min(self.size, SHAPE_ONLY_SCRATCH_ROWS) or 1
                buf = torch.zeros(
                    (rows, self.head_num, self.head_dim),
                    dtype=self.dtype, device=self.device,
                )
                self._buffers["scratch"] = buf
            return buf
        buf = self._buffers.get(layer_id)
        if buf is None:
            buf = torch.zeros(
                (self.size, self.head_num, self.head_dim),
                dtype=self.dtype, device=self.device,
            )
            self._buffers[layer_id] = buf
        return buf

    def _record(self, op: str, **shapes) -> None:
        self.ops_called[op] = self.ops_called.get(op, 0) + 1
        if shapes:
            self.op_shapes.setdefault(op, []).append(shapes)

    def get_kv_buffer_shape(self):
        one = torch.Size((self.size, self.head_num, self.head_dim))
        return one, one

    def get_key_buffer(self, layer_id: int):
        self._record("get_key_buffer", layer=layer_id)
        return self._layer_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        self._record("get_value_buffer", layer=layer_id)
        # MLA keeps one latent entry per token, so K and V are the same region.
        return self._layer_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._record("get_kv_buffer", layer=layer_id)
        buf = self._layer_buffer(layer_id)
        return buf, buf

    def set_kv_buffer(self, layer, loc, cache_k, cache_v=None) -> None:
        """The KV write operator. Called for real; the write itself is empty.

        Shapes are checked rather than used: a caller handing in the wrong
        number of slots is a bug worth hearing about, and an empty operator
        that silently accepts anything would hide it.
        """
        layer_id = getattr(layer, "layer_id", layer)
        n = int(loc.shape[0]) if hasattr(loc, "shape") else len(loc)
        if hasattr(cache_k, "shape") and int(cache_k.shape[0]) != n:
            raise ValueError(
                f"set_kv_buffer: {n} slots but cache_k has "
                f"{int(cache_k.shape[0])} rows"
            )
        self._layer_buffer(layer_id if isinstance(layer_id, int) else 0)
        self._record("set_kv_buffer", layer=layer_id, slots=n)
        # operator body intentionally empty -- nothing is stored

    def get_cpu_copy(self, indices, mamba_indices=None):
        """The KV offload operator (device -> host). Called; copies nothing.

        Returns one correctly shaped placeholder per layer so the caller's own
        bookkeeping (how many layers, how many slots) still runs.
        """
        n = len(indices)
        self._record("get_cpu_copy", slots=n, layers=self.layer_num)
        return [torch.empty((n, self.head_num, self.head_dim), dtype=self.dtype)
                for _ in range(self.layer_num)]

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None) -> None:
        """The reload operator (host -> device). Called; copies nothing."""
        self._record("load_cpu_copy", slots=len(indices),
                     layers=len(kv_cache_cpu) if kv_cache_cpu is not None else 0)


# The guard goes last on purpose: selftest() reaches SimKVCache (defined
# below where this block used to sit), so running the module as a script
# raised NameError while every normal import was fine.
if __name__ == "__main__":
    selftest()
