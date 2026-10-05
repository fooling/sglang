"""A fake ModelRunner for the execution interception point.

Deliverable (2) of the sim-interception prototype: it does not load weights
and does not build a device memory pool (no ``req_to_token_pool`` /
``token_to_kv_pool_allocator`` of its own -- those live on the Scheduler stub
in ``run_smoke.py``, exactly mirroring how the real ``ModelRunner`` leaves KV
pool allocation to ``Scheduler.init_memory_pools`` / ``alloc_memory_pool``).

``forward()`` / ``sample()`` only need to return the right *shapes*; the
values are meaningless (zeros / random), which is enough to drive the
control-plane smoke test in run_smoke.py and to prove the execution
interception point is pluggable without touching scheduler.py or
tp_worker.py source.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


class MockModelRunner:
    """Stands in for ``sglang.srt.model_executor.model_runner.ModelRunner``.

    Real ModelRunner is constructed at tp_worker.py:483
    (``self.model_runner = ModelRunner(**kwargs)``) and does, among other
    things: load_model() (real weights, real device), init_memory_pool
    (device KV cache), init_attention_backends() (model_runner.py:993,
    resolve-by-name at :1008). MockModelRunner intentionally implements none
    of that -- it only has to satisfy what the *control plane* (scheduler /
    schedule_policy) reads off ``tp_worker.model_runner`` during admission,
    plus a forward()/sample() pair with correct output shapes so a caller
    further down the pipeline (real or mock) does not get a shape mismatch.
    """

    def __init__(
        self,
        *,
        model_config,
        device: str = "cpu",
        vocab_size: int | None = None,
        ps=None,
    ):
        self.model_config = model_config
        self.device = device
        # The parallel state the worker was handed. Not decoration: SGLang's
        # own resolve_max_num_reqs divides by ps.attn_dp_size, so a sim that
        # invented a number here would be deciding an admission input.
        self.ps = ps
        self.vocab_size = vocab_size or getattr(model_config, "vocab_size", 32000)

        # Read by schedule_policy / scheduler admission code
        # (scheduler.py:3461-3465 BLOCK_M lookup, scheduler.py:3647
        # prefill_aware_swa). Deliberately bare -- hasattr(...,
        # "extend_attention_block_m") must be False so the real fallback
        # (prefill_tile_block_m = 64) fires, same as any non-Triton backend.
        self.attn_backend = SimpleNamespace()
        self.prefill_aware_swa = False

        # Mirrors the real ModelRunner's memory_pool_config / pool handles
        # being None until Scheduler.init_target_memory_pool() allocates
        # them -- the sim harness never calls that, since KV lives on the
        # Scheduler stub directly (see run_smoke.py build_scheduler_stub).
        self.memory_pool_config = None
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        # the device KV store; the sim has indices but no tensors
        self.token_to_kv_pool = None
        self.attn_backend = None
        self.ngram_embedding_manager = None
        self.canary_manager = None
        self.mtp_draft_device_pools = {}

    # -- pool allocation: SGLang's own, not ours -----------------------
    def alloc_memory_pool(self, memory_pool_config=None):
        """Run SGLang's own ``KVCacheConfigurator.configure``.

        Mirrors ModelRunner.alloc_memory_pool (model_runner.py:872). The sim
        does not get to decide how big the pool is: configure() ->
        config_from_budget -> the pool configurator -> _apply_token_constraints
        (which is where ``--max-total-tokens`` lands) -> _derive_pool_sizes all
        run as SGLang wrote them. Only two steps inside are answered by the
        sim, both shimmed in register.py and both genuinely device-bound:
        _profile_available_bytes (how many bytes are free for KV -- there is no
        device to profile) and _init_pools (which pool classes to construct).

        Every field below is either read off the real ModelConfig / ServerArgs
        or derived by the same call the real runner uses; the stand-ins are the
        device stream and the model object, neither of which exists here.
        """
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.model_executor.model_runner_components.layer_setup import (
            resolve_layer_indices,
        )
        from sglang.srt.runtime_context import get_model, get_schedule, get_server_args
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        if memory_pool_config is not None:
            self.memory_pool_config = memory_pool_config

        mc = self.model_config
        server_args = get_server_args()
        # same derivations as ModelRunner.__init__ (:361-377, :1212)
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.page_size = get_schedule().page_size
        self.is_hybrid_swa = mc.is_hybrid_swa
        self.is_hybrid_swa_compress = mc.is_hybrid_swa_compress
        self.use_mla_backend = mc.attention_arch == AttentionArch.MLA
        self.is_draft_worker = False
        self.draft_model_idx = 0
        self.dtype = mc.dtype
        self.sliding_window_size = None
        self.spec_aux_config = None
        self.gpu_id = 0
        self.layer_info = resolve_layer_indices(
            model=None, model_config=mc,
            is_draft_worker=self.is_draft_worker,
            spec_algorithm=self.spec_algorithm,
        )
        # dtype of the KV store: the server arg if set, else the model dtype
        self.kv_cache_dtype_str = get_model().kv_cache_dtype or "auto"
        self.kv_cache_dtype = mc.dtype

        configurator = KVCacheConfigurator(
            device=self.device,
            gpu_id=self.gpu_id,
            ps=self.ps,
            pp_group=None,
            model=None,  # stand-in: no weights are loaded
            model_config=mc,
            server_args=server_args,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_cache_dtype_str=self.kv_cache_dtype_str,
            model_dtype=self.dtype,
            page_size=self.page_size,
            sliding_window_size=self.sliding_window_size,
            spec_algorithm=self.spec_algorithm,
            is_draft_worker=self.is_draft_worker,
            post_capture_kv_active=False,
            spec_aux_config=self.spec_aux_config,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_swa_compress=self.is_hybrid_swa_compress,
            use_mla_backend=self.use_mla_backend,
            layer_info=self.layer_info,
            forward_stream=None,  # stand-in: no device stream
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_model_idx=self.draft_model_idx,
        )
        self.kv_cache_configurator = configurator
        result = configurator.configure(pre_model_load_memory=0)

        self.max_total_num_tokens = result.max_total_num_tokens
        self.max_running_requests = result.max_running_requests
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool = result.token_to_kv_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.memory_pool_config = result.memory_pool_config
        return result

    @property
    def effective_max_total_num_tokens(self):
        """Same derivation as the real runner (model_runner.py:1337).

        Read by the PD prefill role (disaggregation/prefill.py:169). Derived
        here rather than stored so it cannot drift from the pools.
        """
        capacity = self.max_total_num_tokens
        pool = getattr(self, "req_to_token_pool", None)
        if pool is not None and hasattr(pool, "schedulable_token_capacity"):
            return pool.schedulable_token_capacity(capacity)
        return capacity

    @property
    def max_token_pool_size(self):
        return self.effective_max_total_num_tokens

    # -- the two methods the task asks for -----------------------------
    def forward(self, forward_batch: Any) -> torch.Tensor:
        """Return zero logits shaped [batch_size, vocab_size].

        ``forward_batch`` only needs a ``batch_size`` attribute here; the
        real ForwardBatch carries far more (input_ids, positions, kv
        locations, attn metadata...) that a real attention backend would
        read. This mock does not implement an attention backend, so it
        cannot consume those -- that is the real boundary of what this
        prototype proves (see FEASIBILITY.md C2 discussion).
        """
        batch_size = getattr(forward_batch, "batch_size", None)
        if batch_size is None:
            seq_lens = getattr(forward_batch, "seq_lens", None)
            batch_size = int(seq_lens.shape[0]) if seq_lens is not None else 1
        return torch.zeros(
            (batch_size, self.vocab_size), dtype=torch.float32, device=self.device
        )

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy-argmax over the mock logits -> next_token_ids [batch_size]."""
        return torch.argmax(logits, dim=-1)

    # -- stubs for completeness / future growth -------------------------
    def account_preloaded_weights(self, *_args, **_kwargs) -> None:
        return None


def _sample_for_batch(self, logits, batch):
    """Sampling hook the sim can script per request.

    ``self.token_script`` (if set) maps rid -> the token to emit next; anything
    unscripted falls back to greedy argmax over the mock logits.
    """
    import torch

    script = getattr(self, "token_script", None)
    if script is None:
        return torch.argmax(logits, dim=-1)
    return torch.tensor(
        [script(req) for req in batch.reqs], dtype=torch.long, device="cpu"
    )


MockModelRunner.sample_for_batch = _sample_for_batch
