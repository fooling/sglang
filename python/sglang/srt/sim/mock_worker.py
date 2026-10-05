"""A fake TpModelWorker for the execution interception point.

Deliverable (2): the scheduler.py:956/:960 selection point picks
``MlxTpModelWorker`` or ``TpModelWorker`` by platform; ``register.py``
monkeypatches that import target so a sim run gets ``MockWorker`` instead,
with zero changes to scheduler.py's own source.

This class exposes exactly the attributes the real control-plane code (as
grepped from ``managers/scheduler.py``, see below) reads off
``self.tp_worker``. The ones this prototype does not exercise (LoRA, remote
weight transfer, cuda graphs, embedding/split-prefill forwards) are stubs
that raise ``NotImplementedError`` on first use rather than silently
no-op'ing -- the task's "don't fake success" rule applies here too: if
run_smoke.py or a future caller needs one of these, it must fail loudly, not
pretend.

Attributes/methods the real Scheduler reads off ``self.tp_worker``
(grepped from scheduler.py; only the first four are touched by
run_smoke.py's admission/decode path):
    model_runner, model_config, preloaded_weights_bytes, get_worker_info,
    alloc_memory_pool, get_memory_pool, forward_batch_embedding,
    forward_batch_split_prefill, get_pad_input_ids_func,
    graph_memory_usage, graph_time_usage, init_attention_backends,
    init_cuda_graphs, init_weights_send_group_for_remote_instance,
    load_lora_adapter(_from_tensors), unload_lora_adapter,
    send/receive_weights_to_remote_instance, start/finalize_startup_weight_load,
    weight_load_time, target_worker (draft worker only).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class MockWorker:
    def __init__(self, *, model_runner, server_args: Any = None, gpu_id: int = 0):
        self.model_runner = model_runner
        self.model_config = model_runner.model_config
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.preloaded_weights_bytes = 0
        self.weight_load_time = 0.0

    # -- exercised by run_smoke.py / Scheduler.init_memory_pools --------
    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None:
        """Allocate the pools. Same signature as TpModelWorker (:407).

        Allocating pools is the worker's job (backend side), but how big they
        are is not -- that is an admission input. The standalone harness builds
        its own pools on the Scheduler stub instead, in which case this is a
        no-op because the runner already has them.
        """
        if req_to_token_pool is not None:
            self.model_runner.req_to_token_pool = req_to_token_pool
        if token_to_kv_pool_allocator is not None:
            self.model_runner.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        if self.model_runner.token_to_kv_pool_allocator is not None:
            return None
        # Hand it back to the runner, which runs SGLang's own
        # KVCacheConfigurator (mirrors TpModelWorker.alloc_memory_pool ->
        # ModelRunner.alloc_memory_pool, tp_worker.py:420). The sim must not
        # pick the pool size here: that number is an admission input.
        self.model_runner.alloc_memory_pool(memory_pool_config)
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool_allocator = self.model_runner.token_to_kv_pool_allocator
        return None

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def get_worker_info(self):
        """Same 12-tuple the real TpModelWorker returns (tp_worker.py:542-563).

        Shapes and limits come from the pools and the model config, not from
        invented numbers -- the scheduler derives admission limits from these.
        """
        rt = self.model_runner.req_to_token_pool
        kv = self.model_runner.token_to_kv_pool
        max_total = getattr(self.model_runner, "max_total_num_tokens", None) or (
            self.model_runner.token_to_kv_pool_allocator.available_size()
        )
        ctx = getattr(self.model_runner.model_config, "context_len", 4096)
        max_req_len = min(ctx - 1, max_total - 1)
        from sglang.srt.runtime_context import get_schedule

        schedule = get_schedule()
        return (
            rt.schedulable_token_capacity(max_total)
            if hasattr(rt, "schedulable_token_capacity")
            else max_total,
            schedule.max_prefill_tokens,
            getattr(self.model_runner, "max_running_requests", max_total),
            schedule.max_queued_requests,
            max_req_len,
            max_req_len - 5,
            0,                      # random_seed
            self.model_runner.device,
            None,                   # forward_stream: no device stream in the sim
            rt.size,
            rt.max_context_len,
            getattr(kv, "size", max_total),
        )


    def get_pad_input_ids_func(self):
        return None

    # -- not exercised by this prototype; fail loudly if ever called ----
    def _unimplemented(self, name: str):
        raise NotImplementedError(
            f"MockWorker.{name}: not implemented in the sim-interception "
            "prototype (out of scope for run_smoke.py's control-plane path). "
            "See FEASIBILITY.md."
        )

    def init_attention_backends(self, *a, **kw):
        """No attention backend in the sim: nothing to initialise.

        model_runner.attn_backend stays None, so scheduler.py:3461's
        hasattr(attn_backend, "extend_attention_block_m") is False and the
        prefill tile budget takes the documented fallback of 64 -- the same
        branch the real Ascend backend takes.
        """
        return None

    def _unused_init_attention_backends(self, *a, **kw):
        self._unimplemented("init_attention_backends")

    def init_cuda_graphs(self, *a, **kw):
        return None

    def forward_batch_embedding(self, *a, **kw):
        self._unimplemented("forward_batch_embedding")

    def forward_batch_split_prefill(self, *a, **kw):
        self._unimplemented("forward_batch_split_prefill")

    def load_lora_adapter(self, *a, **kw):
        self._unimplemented("load_lora_adapter")

    def load_lora_adapter_from_tensors(self, *a, **kw):
        self._unimplemented("load_lora_adapter_from_tensors")

    def unload_lora_adapter(self, *a, **kw):
        self._unimplemented("unload_lora_adapter")

    def init_weights_send_group_for_remote_instance(self, *a, **kw):
        self._unimplemented("init_weights_send_group_for_remote_instance")

    def send_weights_to_remote_instance(self, *a, **kw):
        self._unimplemented("send_weights_to_remote_instance")

    def start_startup_weight_load(self, *a, **kw):
        self._unimplemented("start_startup_weight_load")

    def finalize_startup_weight_load(self, *a, **kw):
        self._unimplemented("finalize_startup_weight_load")

    @property
    def graph_memory_usage(self):
        """No CUDA graphs in the sim: nothing captured, so nothing to report."""
        return {}

    @property
    def graph_time_usage(self):
        return {}


# How long a forward takes, in seconds, given the batch. The real design fills
# this from the offline cost library; the default below is a plain per-mode
# number, supplied rather than computed -- a stand-in with the same interface.
# Installed via register.forward_cost_hook().
FORWARD_COST = None

PREFILL_SLICE_S = 0.040
DECODE_SLICE_S = 0.008


def default_forward_cost(batch) -> float:
    """A given time slice per forward: 40 ms for a prefill, 8 ms for a decode.

    Deliberately not derived from the model, the operator layer or the batch
    shape. The point of the seam is that a number from elsewhere is accepted
    as the forward's duration.
    """
    return PREFILL_SLICE_S if batch.forward_mode.is_extend() else DECODE_SLICE_S


def _charge_forward_time(batch) -> float:
    """Advance the virtual clock by this forward's cost, and return it.

    Every timestamp SGLang takes afterwards -- the time_stats it stamps per
    request, the timeout deadlines -- reads that same clock, so the engine
    accounts for the forward at the duration it was given.
    """
    from sglang.srt.sim import register

    cost_fn = FORWARD_COST or default_forward_cost
    cost = float(cost_fn(batch))
    if cost < 0:
        raise ValueError(f"forward cost must not be negative: {cost}")
    register.virtual_clock().advance(cost)
    return cost


def _write_kv(model_runner, batch) -> None:
    """Call the KV write operator for this forward, at the real shape.

    The operator body is empty -- nothing is stored -- but the call happens for
    every KV layer with the slots this batch allocated, so the write path is
    exercised rather than skipped. That is the difference between a mock and a
    hole: swap in a costed or a real operator later and nothing above it
    changes.
    """
    import torch

    kv = getattr(model_runner, "token_to_kv_pool", None)
    if kv is None or not getattr(kv, "layer_num", 0):
        return
    loc = getattr(batch, "out_cache_loc", None)
    if loc is None or not hasattr(loc, "shape") or loc.shape[0] == 0:
        return
    n = int(loc.shape[0])
    shaped = torch.zeros((n, kv.head_num, kv.head_dim), dtype=kv.dtype)
    for layer_id in range(kv.layer_num):
        kv.set_kv_buffer(
            SimpleNamespace(layer_id=layer_id), loc, shaped, shaped
        )


def _sim_forward_batch_generation(self, batch, **kwargs):
    """What the real TpModelWorker returns, with mocked numbers.

    The real worker builds a ForwardBatch, runs the model and samples. Here the
    shapes follow the real ScheduleBatch; only the values are mock. Returning a
    genuine GenerationBatchResult is what lets Scheduler.run_batch and
    process_batch_result run as SGLang's own code.

    The one thing that is not mocked away is time: the forward is charged the
    duration the cost seam supplies, on the clock SGLang itself reads.
    """
    import torch
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.utils import GenerationBatchResult

    bs = batch.batch_size()
    vocab = self.model_runner.vocab_size
    _charge_forward_time(batch)
    _write_kv(self.model_runner, batch)
    logits = torch.zeros((bs, vocab), dtype=torch.float32, device="cpu")
    next_token_ids = self.model_runner.sample_for_batch(logits, batch)
    return GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=logits),
        next_token_ids=next_token_ids,
        can_run_cuda_graph=False,
    )


MockWorker.forward_batch_generation = _sim_forward_batch_generation



MockWorker.is_hybrid_swa = property(lambda self: False)


