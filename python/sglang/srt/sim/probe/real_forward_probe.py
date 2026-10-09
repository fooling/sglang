"""探针：让真模型的 forward 真跑一遍，算子体全是 mock，把调用序记下来。

这是「执行拦截面从整个 forward 下沉到单个算子」的可行性验证，**不是**产品代码。
跑法（仓根目录）：
    python/.venv/bin/python python/sglang/srt/sim/probe/real_forward_probe.py \
        python/sglang/srt/sim/probe/tiny_qwen2

2026-10-09 的结果：forward 走到底，logits [2, 1024]；dispatcher 记到 136 次算子调用，
attention 被拦到 2 次（两层各一次，带真实 shape），未覆盖的 npu 算子 0 个。

四个让它成立的杠杆，按重要性排：
1 **load_format="dummy"**：SGLang 自带 DummyModelLoader，不要 checkpoint 就能把模型建出来。
2 **一个本地的极小 config.json**（2 层 / hidden 128 / vocab 1024）：ModelConfig 从本地目录读，
  不碰网络；形状保真度在阶段一不重要，要真形状时把 config 改大即可。
3 **runtime_context.publish(server_args, role="scheduler")**：一次发布整棵配置树。
  不发布就会在 _initialize_model 里撞 "config namespace 'exec' not published"。
4 **forward_npu -> forward_native 的整体改指**：凡是自己就带着 forward_native 的层
  （本次 15 个类）都改指过去。**不自己实现任何 npu 算子**——用上游维护的那份等价实现，
  和 support_triton 压 False 走 torch 回退是同一个套路。

还没做的（阶段一剩下的）：
- ForwardBatch 是手搓的。真路径要走 ScheduleBatch.get_model_worker_batch() -> init_new()。
- SimAttnBackend 只记形状、返回零张量，没有 KV 读写，也没有 metadata 构造。
- 算子描述还只是 (算子名, 进出形状)，缺 dtype/格式/非张量参数/并行上下文/调用点身份。
- 时长一律不记。
"""
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TINY = os.path.join(_HERE, "tiny_qwen2")
# <repo>/python/sglang/srt/sim/probe -> <repo>/python
_PY_ROOT = os.path.abspath(os.path.join(_HERE, *[os.pardir] * 4))


def main(TINY: str = _DEFAULT_TINY) -> int:
    if _PY_ROOT not in sys.path:
        sys.path.insert(0, _PY_ROOT)
    from sglang.srt.sim.fake_npu import install_fake_npu
    install_fake_npu()

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from sglang.srt.server_args import ServerArgs
    from sglang.srt import runtime_context as rc
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.model_loader import get_model
    import sglang.srt.distributed as dist

    sa = ServerArgs(model_path=TINY, dtype='float32', device='npu',
                    attention_backend='ascend', load_format='dummy', disable_cuda_graph=True)
    rc.publish(sa, role='scheduler')
    dist.init_distributed_environment(world_size=1, rank=0, local_rank=0,
        distributed_init_method='tcp://127.0.0.1:29593', backend='gloo')
    dist.initialize_model_parallel(tensor_model_parallel_size=1)
    mc = ModelConfig(model_path=TINY, dtype='float32')
    model = get_model(model_config=mc, load_config=LoadConfig(load_format='dummy'),
                      device_config=DeviceConfig(device='cpu'))
    print('model built:', type(model).__name__)

    # ── 算子拦截面 1/3：attention 后端（唯一需要设备 kernel 的地方）──
    ATTN_CALLS = []
    class SimAttnBackend:
        def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
            ATTN_CALLS.append(dict(layer=layer.layer_id, q=tuple(q.shape),
                                   k=None if k is None else tuple(k.shape),
                                   heads=layer.tp_q_head_num, vdim=layer.v_head_dim))
            return q.new_zeros((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))

    import sglang.srt.layers.radix_attention as ra
    ra.get_attn_backend = lambda *a, **k: SimAttnBackend()

    # ── 顶替 rope：和 Triton 那次同一个套路——上游自己就带着 forward_native，
    #    不自己实现 npu_mrope，改走它的等价实现。
    import gc as _gc, inspect as _inspect
    def _route_npu_to_native():
        """凡是自己就带着 forward_native 的层，把 forward_npu 改指向它。
        不自己实现任何 npu 算子——用上游维护的那份等价实现。"""
        done = []
        for mod in list(sys.modules.values()):
            if not mod or not getattr(mod, '__name__', '').startswith('sglang.srt.layers'):
                continue
            for nm, obj in vars(mod).items():
                if not _inspect.isclass(obj):
                    continue
                if 'forward_npu' in obj.__dict__ and hasattr(obj, 'forward_native'):
                    obj.forward_npu = obj.forward_native
                    done.append(f'{obj.__module__.split(".")[-1]}.{obj.__name__}')
        return sorted(set(done))
    _routed = _route_npu_to_native()
    print(f'forward_npu -> forward_native 改指 {len(_routed)} 个类:', _routed[:8], '...' if len(_routed)>8 else '')

    # ── 算子拦截面 2/3：torch dispatcher（其余全部算子）──
    OPS = []
    class Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            OPS.append((str(func),
                        tuple(tuple(a.shape) for a in args if isinstance(a, torch.Tensor)),
                        tuple(out.shape) if isinstance(out, torch.Tensor) else None))
            return out

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    NTOK = 6
    fb = ForwardBatch.__new__(ForwardBatch)
    fb.forward_mode = ForwardMode.EXTEND
    fb.batch_size = 2
    fb.input_ids = torch.randint(0, mc.vocab_size, (NTOK,))
    fb.positions = torch.arange(NTOK)
    fb.seq_lens = torch.tensor([3, 3])
    fb.out_cache_loc = torch.arange(NTOK)
    fb.attn_backend = SimAttnBackend()
    fb.token_to_kv_pool = None
    fb.extend_seq_lens = torch.tensor([3, 3])
    fb.extend_prefix_lens = torch.tensor([0, 0])
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    fb.capture_hidden_mode = CaptureHiddenMode.NULL
    fb.return_logprob = False
    fb.extend_seq_lens_cpu = [3, 3]
    fb.extend_logprob_start_lens_cpu = None
    fb.temp_scaled_logprobs = False
    fb.top_p_normalized_logprobs = False
    fb.spec_info = None
    fb.input_embeds = None
    fb.token_ids_logprobs = None
    fb.dp_padding_mode = None
    fb.global_num_tokens_cpu = None
    fb.gathered_buffer = None
    fb.sampling_info = None
    fb.next_token_logits_buffer = None

    try:
        with torch.no_grad(), Recorder():
            out = model.forward(fb.input_ids, fb.positions, fb)
        print('FORWARD OK ->', type(out).__name__,
              getattr(out, 'next_token_logits', torch.empty(0)).shape)
    except Exception as e:
        print('forward ->', type(e).__name__, e)
        traceback.print_exc(limit=8)

    from sglang.srt.sim.fake_npu import op_counts
    print('--- 这一跑碰到的 npu 入口 ---', dict(op_counts()))
    print(f'--- attention 调用 {len(ATTN_CALLS)} 次 ---')
    for c in ATTN_CALLS: print('   ', c)
    print(f'--- dispatcher 拦到 {len(OPS)} 次算子调用，去重后 ---')
    seen = {}
    for name, ins, o in OPS:
        seen.setdefault(name, 0)
        seen[name] += 1
    for k, v in sorted(seen.items(), key=lambda x: -x[1]):
        print(f'   {v:4d}  {k}')
    return 0


# 模块体不许是裸的：这个文件在 sglang 包树里，被任何收集器 import 到都不该跑起来。
if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TINY))
