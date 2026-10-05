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
    def alloc_memory_pool(self) -> None:
        """No-op: KV pools live on the Scheduler stub directly in this
        prototype (see run_smoke.py build_scheduler_stub), so there is
        nothing for the worker to allocate."""
        return None

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def get_worker_info(self):
        return {
            "model_runner": "MockModelRunner",
            "device": self.model_runner.device,
            "weights_loaded": False,
        }

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
        self._unimplemented("init_attention_backends")

    def init_cuda_graphs(self, *a, **kw):
        self._unimplemented("init_cuda_graphs")

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
        return 0

    @property
    def graph_time_usage(self):
        return 0.0


def _sim_forward_batch_generation(self, batch, **kwargs):
    """What the real TpModelWorker returns, with mocked numbers.

    The real worker builds a ForwardBatch, runs the model and samples. Here the
    shapes follow the real ScheduleBatch; only the values are mock. Returning a
    genuine GenerationBatchResult is what lets Scheduler.run_batch and
    process_batch_result run as SGLang's own code.
    """
    import torch
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.utils import GenerationBatchResult

    bs = batch.batch_size()
    vocab = self.model_runner.vocab_size
    logits = torch.zeros((bs, vocab), dtype=torch.float32, device="cpu")
    next_token_ids = self.model_runner.sample_for_batch(logits, batch)
    return GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=logits),
        next_token_ids=next_token_ids,
        can_run_cuda_graph=False,
    )


MockWorker.forward_batch_generation = _sim_forward_batch_generation
