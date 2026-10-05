# 仿真接入点原型 — 可行性报告

分支 `feat/sim-interception`，基线 `5b33b51793`。解释器
`~/repo/sglang/python/.venv/bin/python`（3.12.14 / torch 2.13.0 /
`torch.cuda.is_available()==False` / 无 CANN、torch_npu）。全部命令前缀：

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 no_proxy='*' NO_PROXY='*'
```

交付物都在 `python/sglang/srt/sim/`：`mock_model_runner.py`、`mock_worker.py`、
`cpu_kv.py`、`virtual_clock.py`、`register.py`、`run_smoke.py`。

---

## ① 跑通了什么（原样输出）

### 1. `cpu_kv.py` 独立自测 —— 验证 C4

```
perl -e 'alarm 60; exec @ARGV' python/.venv/bin/python -m sglang.srt.sim.cpu_kv
```

```
TokenToKVPoolAllocator(device='cpu') available_size: 64
alloc(10) -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
available_size after alloc: 54
available_size after free: 64
ReqToTokenPool(device='cpu') available_size: 8
alloc_rows(3) -> [6, 7, 8] available_size after: 5
```

真实的 `TokenToKVPoolAllocator` / `ReqToTokenPool`，`device='cpu'`，**零改动**。

### 2. `register.py` 独立自测 —— 验证四个拦截面各自可装

```
perl -e 'alarm 90; exec @ARGV' python/.venv/bin/python -m sglang.srt.sim.register
```

```
  [execution] Scheduler.init_tp_model_worker(stub) -> tp_worker=SimTpModelWorker  model_runner=MockModelRunner  ok=True
  [kv] KVCacheConfigurator.configure(stub) -> allocator.available_size()=100 req_to_token_pool.available_size()=100 ok=True
  [transfer] AscendKVManager.init_engine(stub) -> engine=MockTransferEngine ok=True
  [clock] scheduler.time.perf_counter(): 0.0 -> 10.0 (advanced only by explicit VirtualClock.advance(), no real sleep) ok=True

ALL FOUR INTERCEPTION POINTS OK: True
```

四个 selftest 都是对**真实生产类的真实绑定方法**（`Scheduler.init_tp_model_worker`、
`KVCacheConfigurator.configure`、`AscendKVManager.init_engine`、
`scheduler.py` 自己的 `time.perf_counter()`）直接调用，不是关起门来测一个孤立对象。

### 3. `run_smoke.py` 完整跑一遍 —— 验证 C1/C3/C4/C5 + 组 batch 全链路

```
perl -e 'alarm 120; exec @ARGV' python/.venv/bin/python python/sglang/srt/sim/run_smoke.py
```

（完整输出，已去掉 torch 启动期的无关 warning）：

```
==================== step 0: register sim shims at the 4 interception points ====================
  [execution] Scheduler.init_tp_model_worker(stub) -> tp_worker=SimTpModelWorker  model_runner=MockModelRunner  ok=True
  [kv] KVCacheConfigurator.configure(stub) -> allocator.available_size()=100 req_to_token_pool.available_size()=100 ok=True
  [transfer] AscendKVManager.init_engine(stub) -> engine=MockTransferEngine ok=True
  [clock] scheduler.time.perf_counter(): 0.0 -> 10.0 (advanced only by explicit VirtualClock.advance(), no real sleep) ok=True

  all 4 interception-point selftests passed: True

==================== step 1: publish ServerArgs (CPU, no device queries) ====================
torch.cuda.is_available() = False

==================== step 2: put 10 Req objects into waiting_queue ====================
  queued req-0: prompt_len=20
  queued req-1: prompt_len=28
  queued req-2: prompt_len=36
  queued req-3: prompt_len=44
  queued req-4: prompt_len=52
  queued req-5: prompt_len=60
  queued req-6: prompt_len=68
  queued req-7: prompt_len=76
  queued req-8: prompt_len=84
  queued req-9: prompt_len=92

==================== step 3: drive prefill admission (Scheduler.get_new_batch_prefill) ====================
  round 1: available_size before=256 after=76 req_pool_available_before=32 admitted_this_round=5 (['req-0', 'req-1', 'req-2', 'req-3', 'req-4']) waiting_queue_left=5
  round 2: available_size before=76 after=76 req_pool_available_before=27 admitted_this_round=0 ([]) waiting_queue_left=5

  TOTAL admitted across 2 round(s): 5 / 10 requests, running_batch.batch_size()=5

==================== step 4: move running_batch to decode, check_decode_mem ====================
  running_batch is now decode, batch_size=5, token_to_kv_pool_allocator.available_size()=71
  check_decode_mem() with full pool -> fits=True

==================== step 5: shrink the KV pool and force retract_decode ====================
  drained allocator down to available_size=1 (kept 1 free page) before retraction
  check_decode_mem() after drain -> fits=False
  retract_decode(): batch_size 5 -> 4, retracted=1 (['req-4']), aborted=0, new_token_ratio=1.0

==================== step 6: try run_batch / forward on the mock model runner ====================
  mock forward() logits.shape=(4, 32000), sample() next_token_ids.shape=(4,)

==================== step 7: virtual clock sanity check for the timeout interception points ====================
  virtual monotonic: 10.0 -> 15.0 (advanced by 5.0s, no wall-clock sleep)

==================== step 8: drive the REAL _abort_on_waiting_timeout off the virtual clock ====================
  before: waiting_queue=['stale-req', 'fresh-req'], virtual now=15.0, timeout_s=10
  after:  waiting_queue=['fresh-req'], aborted_and_sent=['stale-req']

==================== step 9: drive the REAL _abort_on_running_timeout off the virtual clock ====================
  before: victim=req-0 to_finish=None virtual now=15.0, timeout_s=10
  after:  victim=req-0 to_finish=<sglang.srt.managers.schedule_batch.FINISH_ABORT object at 0x139954a40>
```

要点核对：
- step3 第 1 轮：256 个 slot，5 个请求（20/28/36/44/52 token，共 180 token）全部准入，
  `available_size` 256→76（消耗 180，吻合）；第 2 轮因 `running_batch.batch_is_full`
  仍为真（本原型不驱动真实 forward 把它翻回去，和真引擎语义一致）admit 0，
  循环正确退出。这是 `scheduler.get_new_batch_prefill` /
  `schedule_policy.PrefillAdder` **真实代码**跑出来的,不是重写的简化版。
- step4/5：`check_decode_mem()` 在满池时 `fits=True`，把分配器几乎耗尽后
  `fits=False`，`retract_decode()` 真的退回 1 个请求（`req-4`，LIFO 顺序符合
  `_get_decode_retraction_order` 的默认策略）。
- step8/9：两处超时读点（`_abort_on_waiting_timeout` / `_abort_on_running_timeout`,
  scheduler.py:2994/:1704）在**只有虚拟钟推进、没有真实 `time.sleep`** 的情况下，
  被真实触发。

---

## ② 没跑通什么（原样报错，含调试过程中真实命中的坑）

这些都是在把 run_smoke.py 从零搭起来的过程中真实命中、然后修掉的；全部列出
是因为它们本身就是"控制面到底依赖什么"的证据，而不是噪音。

**1) `get_parallel().pp_max_micro_batch_size` 是 None**
```
File ".../scheduler.py", line 3349, in get_num_allocatable_reqs
    pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
```
原因：真实 `Scheduler.__init__`（scheduler.py:1099-1104）会在这个值未设置时算一次
默认值并 `get_context().override(...)` 发布；本原型跳过了 `__init__`（用
`Scheduler.__new__` 搭 stub），没人做这一步。**修法**：在 harness 里补一次同样的
`get_context().override("sim.run_smoke", pp_max_micro_batch_size=1<<20)`，不改
scheduler.py。

**2) `Scheduler` stub 缺字段**（`truncation_align_size`、`enable_priority_scheduling`
等），都是单纯的 `AttributeError`，照着报错把字段按合理默认值补上即可，不是逻辑
问题。

**3) `chunked_prefill_size=-1` 喂给 `PrefillAdder` 导致静默 0 准入**（不是异常，是
错误结果：round 1 `admitted_this_round=0`）。原因：真实 scheduler.py:1220-1222 把
`<=0` 规整成 `None`（无限 chunk 预算），本原型一开始直接把 -1 传进去，被
`PrefillAdder` 当成"只剩 -1 个 token 的 chunk 预算"。**修法**：stub 里用 `None`。

**4) `running_batch.spec_algorithm` 是 `None`**
```
File ".../schedule_batch.py", line 3240, in prepare_for_decode
    if not self.spec_algorithm.is_none():
AttributeError: 'NoneType' object has no attribute 'is_none'
```
`ScheduleBatch(reqs=[], batch_is_full=False)` 直接构造时没传
`spec_algorithm`，默认是 `None` 而不是 `SpeculativeAlgorithm.NONE`。**修法**：
构造时显式传。

**5) 真·Triton 内核挡路（这是本次最重要的一个"没跑通"）**
```
File ".../mem_cache/allocation.py", line 357, in alloc_for_extend
    write_cache_indices(...)
File ".../mem_cache/allocation.py", line 77, in write_cache_indices
    write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](...)
TypeError: 'function' object is not subscriptable
```
`write_cache_indices`（`mem_cache/allocation.py:54`）按
`support_triton(prefill_backend)` 在"Triton 内核"和"纯 Python/张量循环回退"之间二选
一；`prefill_backend` 来自 `attention_backends()`（读 `get_exec().kernel.*`）。本原型
的 `ServerArgs(model_path="dummy")` 没显式给 `device`/`attention_backend`，解析出来
是 `None`，而 `support_triton(None)` 返回 `True`——于是走了真实 Triton 内核，在无
GPU 的 CPU 进程上无法 launch。**这不是靠改 scheduler.py 或 schedule_batch.py 能解
决的**，是一个配置选择：把 `ServerArgs(attention_backend="torch_native")` 显式给够
（`support_triton()` 明确把 `"torch_native"` 排除在外，sglang/srt/utils.py:1332-
1333），`write_cache_indices` 就走到了纯 Python fallback（`mem_cache/allocation.py:
87-103`，逐 request `.item()` + `req_to_token_pool.write(...)`，没有任何设备专属调
用）。**记入结论**：C1 的"控制面不碰设备"成立，但前提是"选了一个不含 Triton/CUDA
内核的 attention backend"这个配置分支——这件事本身也是"按配置选实现"的一个例
证，只是选择点不在四个拦截面清单里，而在 `attention_backends()`/
`support_triton()` 这条路上。

**6) `tree_cache.req_to_token_pool` 是 `None`**
```
File ".../radix_cache.py", line 478, in cache_finished_req
    kv_indices = self.req_to_token_pool.req_to_token[...]
AttributeError: 'NoneType' object has no attribute 'req_to_token'
```
`RadixCache.create_simulated()` 自己的 docstring 就说"a radix cache without
memory pools for simulation purpose"——它默认不带 `req_to_token_pool`，这对纯准
入路径没问题（只读 `evictable_size()` 这类），但 `retract_decode()` 的释放路径
会调用 `tree_cache.cache_finished_req()`，直接按 `req_to_token_pool.req_to_token`
取值。**修法**：把 harness 自己那份真实 `ReqToTokenPool` 对象也挂到
`tree_cache.req_to_token_pool` 上（同一个对象，不是另造一个）。

**最终仍未验到的（诚实列出，没有去凑）**：
- `Scheduler.run_batch` / `process_batch_result`（scheduler.py:3861/:4205）本身没
  有被调用——run_smoke.py 第 6 步只是直接调 `mock_model_runner.forward()` /
  `.sample()`，没有通过真实的 `ForwardBatch` 构造路径。再往下挖需要
  `model_executor/forward_batch_info.py` 的 `ForwardBatch.init_new` 读
  `ScheduleBatch` 的哪些字段，以及一个真实或假的 attention backend 对象，工作量
  明显超出本轮"验证拦截点是否可行"的目标，老实标注为**没验到**。
- `mlx` 分支（scheduler.py:953 的 `use_mlx()` 为真那条路）完全没碰，只验了
  `else` 分支（`TpModelWorker`）。
- KV / 传输两个拦截面的 selftest 证明了"这个类/方法可以被猴补丁替换"，但**没有
  跟一次真实的 `ModelRunner` 启动串起来**——那需要真实或极度拟真的 `hf_config`、
  `ParallelState`、`layer_info` 等一整套 `KVCacheConfigurator` 构造参数，这部分
  本轮没有做,详见 ③ 的说明。
- 真实 NPU / memfabric 传输路径，以及任何需要网络/多进程的 PD 分离流程，完全没
  碰（机器上也没有 NPU）。

---

## ③ C1–C5 逐条判定

### C1：控制面（scheduler / schedule_policy / schedule_batch）不依赖设备库，CPU torch 就能跑
**证实，但有一个前提条件**：`attention_backend` 必须选到一个不含 Triton/CUDA 内核
的分支（见②-5）。选对之后，`get_new_batch_prefill` → `PrefillAdder` →
`ScheduleBatch.prepare_for_extend` → `alloc_for_extend` → `check_decode_mem` →
`retract_decode` 整条路径，在 `torch.cuda.is_available()==False` 的真实 CPU 进程
里全部跑通，没有抛出任何设备相关异常。
复跑命令：
```
perl -e 'alarm 120; exec @ARGV' python/.venv/bin/python python/sglang/srt/sim/run_smoke.py
```

### C2：四个选择点能装进仿真实现类，不改 SGLang 源码
**证实（零源码改动，`git status`/`git diff` 可核）**，但范围要讲清楚：
- 执行面（scheduler.py:956/960）：端到端证实——`register.selftest_execution_shim()`
  调用真实 `Scheduler.init_tp_model_worker`，产出 `tp_worker` 是我们的
  `SimTpModelWorker`。
- 时钟面（scheduler.py:1704/1712、2994/2999）：端到端证实——run_smoke.py 第
  8/9 步直接调真实的 `Scheduler._abort_on_waiting_timeout` /
  `_abort_on_running_timeout`，只靠虚拟钟推进就触发了超时逻辑。
- KV 面：**选择点本身与任务书给的行号不完全一致，这是我核实后需要更正的一处**
  （见④）。真正的选择点是 `KVCacheConfigurator.configure`（被
  `model_runner.py:878` 调用一次），而不是 `kv_cache_configurator.py:1305/
  1334/1472/1765`——那四行是同一个方法内部 if/elif 链的四个分支构造调用,不是
  四个独立的"选择点"。装在 `.configure` 这一层，`register.selftest_kv_shim()`
  对**真实的 `KVCacheConfigurator.configure` 绑定方法**调用证实可行；但没有跟一
  次真实 `ModelRunner`/`KVCacheConfigurator.__init__` 串起来（那需要真实
  `hf_config`/`ParallelState`/`layer_info`，超出本轮范围）——这部分记为**没验
  到**。run_smoke.py 自己的组 batch 演示绕开了 `KVCacheConfigurator`，直接用
  `cpu_kv.py` 构造真实分配器/池对象（因为 C4 已经证明这样做和走配置器选出来的结
  果在语义上等价：都是同一个 `TokenToKVPoolAllocator`/`ReqToTokenPool` 类）。
- 传输面（disaggregation/ascend/conn.py:42）：`register.selftest_transfer_shim()`
  对真实 `AscendKVManager.init_engine` 绑定方法调用证实可行；同样没有跟一次真实
  disaggregation 启动串起来——记为**没验到**。
复跑命令：
```
perl -e 'alarm 90; exec @ARGV' python/.venv/bin/python -m sglang.srt.sim.register
git status --porcelain | grep -v '^?? python/sglang/srt/sim/'   # 确认零改动
```

### C3：组 batch 决策输入只有 Python 侧状态，没有一次设备查询
**证实**。run_smoke.py 全程 `torch.cuda.is_available()==False`（这是这台机器的
真实状态，不是 mock 出来的），而第 3-5-8-9 步没有任何异常——如果
`get_new_batch_prefill`/`retract_decode`/两处超时检查里有任何一次真实设备查询
（如 `.cuda()`、`torch.cuda.synchronize()` 之类)，在这台机器上会直接抛异常而不
是静默过去。额外佐证：② 中列出的六个问题里，没有一个是"要连 GPU/NPU"，全部是
"stub 缺字段"或"配置选错分支"这两类纯 Python 问题。

### C4：KV 分配器管的是索引不是显存，`device='cpu'` 下同样的 arange/切片/cat 能给出同样的 `available_size()`
**证实，且是直接证据而非推论**：`TokenToKVPoolAllocator`/`ReqToTokenPool` 两个
类**完全未修改**，直接用 `device='cpu'` 构造，`alloc`/`free`/`available_size`
的返回值和真实数值完全符合预期（见①-1 的输出）。`cpu_kv.py` 因此没有重新定义
一个分配器类——按任务要求,这件事本身就是一条该报告的结论。
复跑命令：
```
perl -e 'alarm 60; exec @ARGV' python/.venv/bin/python -m sglang.srt.sim.cpu_kv
```

### C5：decode 侧 KV 不够会走 `retract_decode` 退回队列
**证实**。run_smoke.py 第 5 步把分配器几乎耗尽（只留 1 个 page）后，
`check_decode_mem()` 返回 `False`，紧接着调用的**真实** `ScheduleBatch.
retract_decode()`（未经任何改写）把 batch 从 5 个请求退到 4 个，`retracted_reqs
=['req-4']`，`aborted=0`。
复跑命令：同 C1。

---

## ④ 任务书里的问题 / 我发现的额外问题

**任务书里站不住的一处**：KV 拦截面给的四个行号
（`kv_cache_configurator.py:1305/:1334/:1472/:1765`）实际上是
`KVCacheConfigurator._build_token_to_kv_pool`（一个约 90 行的 if/elif 分发方法)
内部四个不同分支各自构造具体池类（`NPUMLATokenToKVPool`、`NPUMHATokenToKVPool`
等）的那一行。它们不是四个可以独立拦截的"选择点"——要在那个层级拦截,得把
`_build_token_to_kv_pool` 整个方法换掉,或者把分发条件(`current_platform.
is_out_of_tree()`、`get_exec().kernel.attention_backend=="ascend"`、
`self.use_mla_backend`...）全部伪造成某个分支成立,成本和拦截
`.configure()`（它唯一的调用方只有 `model_runner.py:878`）几乎一样,但后者干净
得多。本报告把 KV 拦截面落在 `KVCacheConfigurator.configure`,已在②③中说明并更
正。

**额外发现的问题**：
1. **"不改源码"在 C1 这条链路上隐含了一个配置前提**：`ServerArgs()` 默认不解析
   出一个不含 Triton 内核的 `attention_backend`；必须显式
   `attention_backend="torch_native"` 才能让 `alloc_for_extend` 避开真实 Triton
   kernel launch。这不是改了源码，但确实是"必须对，不然会在 CPU 机器上崩"的一
   个隐藏前提,原任务书没有提到这一点,建议写进后续任何"CPU 模拟 sglang 控制面"
   的操作手册里。
2. **`RadixCache.create_simulated()` 的默认行为和 `retract_decode` 的真实需求不
   完全匹配**：它的 docstring 明确说"没有内存池用于仿真",但 `retract_decode`
   的释放路径（经 `release_kv_cache` → `tree_cache.cache_finished_req`）需要
   `req_to_token_pool`。本原型的修法是把同一个真实 `ReqToTokenPool` 对象挂上去,
   但这意味着 `create_simulated()` 这个"官方仿真入口"本身并不能脱离
   `req_to_token_pool` 单独支撑完整的 decode-retraction 路径——这对任何后续想复
   用 `create_simulated()` 做纯仿真(不碰真实池对象)的人是一个值得知道的边界。
3. **`KVCacheConfigurator` 用 `@dataclass(slots=True)`**：想往 `__new__` 出来的
   实例上挂任意 sim 专用字段(本来想挂 `_sim_max_total_num_tokens` 之类)会直接
   `AttributeError`,因为没有 `__dict__`。这类 slots 类在用"真方法 + 假 self"的
   测试手法时,传参渠道只能走已声明的槽位(本原型借用了 `model_config`)。以后
   如果要给 `KVCacheConfigurator` 加测试钩子,得留意这一点。
4. **`/repo/sglang` 里已经存在一个独立的"排程仿真器"**
   （`python/sglang/srt/debug_utils/schedule_simulator/`,连带
   `test/registered/debug_utils/test_schedule_simulator.py`）。它是一整套**平行
   重新实现**(自己的 `SimRequest`/`GPUState`/`SchedulerPolicy`),完全不触碰
   `managers/scheduler.py`、`schedule_policy.py`、`schedule_batch.py` 这些真实
   控制面代码——跟本任务要验的"在真实选择点装仿真类、驱动真实控制面代码"是两件
   不同的事,不能互相替代。值得让后续读到这份报告的人知道,免得重复发明或误用。
5. **调度决策逻辑完全未改动**：② 列出的六处修复全部是"给 stub 补齐真实
   `Scheduler.__init__` 本该设置的字段"或"选对配置分支",没有一处碰了
   `get_new_batch_prefill`/`PrefillAdder`/`retract_decode` 内部的准入、退避判断
   逻辑本身——这点我在改动过程中反复核对过,逐条列在②里就是为了让这件事可核验,
   不是我自己说了算。

---

## 结论

这套"按既有选择点装仿真实现类"的接入方案，在 control-plane 这一侧**可行**，且
比预想的更干净：四个列出的拦截面里，执行面和时钟面做到了端到端证实（真实绑定
方法 + 真实触发路径）；KV 面和传输面证实"可被猴补丁接管"，但没有做端到端集成
(受限于构造一个真实 `ModelRunner` 所需的配置量,这不是这套方案本身的缺陷,是本
轮验证深度的边界)。

前提条件(按重要性排序):
1. `attention_backend` 必须选到非 Triton/CUDA 的分支,否则纯 Python 回退路径
   不会被触发,CPU 进程会在尝试 launch 真实 kernel 时崩溃——这与"仿真接入点"方
   案无关,是 sglang 本身"按配置选内核实现"机制的一部分,但必须被正确配置。
2. 凡是脱离 `Scheduler.__init__` 真正跑过的字段(本报告②中枚举的六处),仿真
   harness 必须自己补齐,否则会在意料之外的地方抛 `AttributeError`——这对"仿真
   不改源码"成立,但说明仿真层需要紧跟 `Scheduler.__init__` 的实际赋值,维护成
   本不是零。
3. KV/传输两个拦截面若要支撑真实引擎启动(而不只是 harness 直接构造对象),还需
   要补一层"伪造最小 `ModelRunner` 配置"的工作,这部分本轮诚实地标为没验到。
