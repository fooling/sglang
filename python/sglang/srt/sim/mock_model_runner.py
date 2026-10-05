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
    ):
        self.model_config = model_config
        self.device = device
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
