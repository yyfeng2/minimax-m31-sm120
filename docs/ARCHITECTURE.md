# 架构与补丁设计

本文说明：硬件约束 → 官方链在 SM120 的断点 → 三层补丁如何绕开 → 运行时数据流 →
每个补丁文件的职责。故障点的逐条证据见 [SM120_FAULT_POINTS.md](SM120_FAULT_POINTS.md)。

---

## 1. 硬件约束：SM120 缺了什么

RTX PRO 6000 Blackwell（SM120，compute capability 12.0）与数据中心
B200/B300（SM100/103/107）同属 Blackwell 家族，但**没有 tcgen05 / TMEM（UMMA）**。
官方 MiniMax-M3 的 NVFP4 高性能链路几乎全部建立在这套硬件上：

| 链路 | 依赖 | SM120 结果 |
|---|---|---|
| CuTe-DSL prefill（`cute/src/sm100/q8kv8_indexer_prefill.py`） | tcgen05 MMA + TMA | 编译/运行不可用 |
| CUTLASS Q8KV4 decode（`decode_q8kv4/`） | tcgen05 | `Failed to plan fmha_sm100: invalid argument` |
| MSA（`sparse_attention_msa.py` / `msa_cutlass_sparse_decode.py`） | SM100 fmha C++ plan | capture 阶段崩溃 |
| FlashInfer TRT-LLM / CuTe-DSL MoE | SM100 cubins | 被 oracle 过滤 |
| NVVM `sm120a` codegen | SM120a 指令集 | 与 tcgen05 内核不匹配 |

SM120 实际具备的是 **`mma.sync` 块缩放 MMA**：

- `kind::mxf4nvf4.block_scale.scale_vec::4X` —— FP4（E2M1）输入 + E4M3 块缩放；
- `kind::mxf8f6f4.block_scale.scale_vec::1X` —— FP8（E4M3/E5M2）输入 + E8M0 缩放。

因此本工程的策略是：**官方 nvfp4 主链整体退出，改用 SM120 原生指令重写关键路径**，
而不是继续放宽 gate。

---

## 2. 三层补丁

### v1 降级层（`patch_port.py`）

目标：让服务在 SM120 上先能稳定跑起来，且不引入任何 SM100 代码。

在 vLLM 两个选择器入口插入平台判断：

| 位置 | 逻辑 |
|---|---|
| `models/minimax_m3/common/sparse_attention.py::select_main_backend_and_impl_cls` | `nvfp4 + SM120` → 主 attention **完全退出 MSA**，选 `MiniMaxM3SparseBackend + MiniMaxM3SparseTritonImpl`（Triton 稀疏 attend/decode；page 128 与 `--block-size=128` 对齐） |
| `models/minimax_m3/common/indexer.py::select_indexer_impl_cls` | `nvfp4 + SM120` → 选 `MiniMaxM3IndexerTritonImpl`（与主模型共享 `topk_indices_buffer` / `block_table`） |

逃生开关：`M3_SM120_KEEP_NVFP4=1` 保留官方路由（用于复现崩溃，不可用于生产）。

v1.1 的修正：仅降级 KV dtype 不够，MSA(fp8) 的 `should_prepare_decode_metadata`
仍会调用 C++ `kv_mode3` plan 并在 SM120 报 `invalid argument`，所以主选择器必须
**直接离开 MSA 类**。

### v2 原生注意力（`patch_native.py` + `models/minimax_m3/common/ops/msa_sm120/`）

目标：decode 阶段不再把 NVFP4 KV 反量化成 BF16，直接在 FP4/FP8 张量核上计算。

| 阶段 | 实现 |
|---|---|
| QK^T | `mma.sync … kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`，直接读 cache 中的 E2M1 码 + E4M3/16 block scale；Q 现场量化为 NVFP4 |
| PV | `mma.sync … kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`，P 用 E4M3 + E8M0 幂次 scale；V 在 smem 中转 E4M3（NVFP4 的 V scale 沿 head 维，与 PV 收缩轴不对齐） |
| split-K | 自研 merge kernel，grid `(T, Hkv, nchunk)`，按 ~256 CTA 自适应 chunk 数；`w_c = exp2(m_c − M)`，`out = Σ w_c O_c / Σ w_c l_c` |

注入点：

1. 模块级 `_msa_sm120_enabled()` / `_msa_sm120_load()`，开关 `M3_SM120_NATIVE=0` 回退；
2. `__init__` 中在**模型加载期**编译并加载 JIT 扩展（不能发生在 CUDA graph capture 内）；
3. `forward` 的 decode 分支改调 `msa_decode_nvfp4(...)`；prefill 保持 Triton
   （prefill 不需要张量核）。

内核源码（镜像内 `models/minimax_m3/common/ops/msa_sm120/`）：

- `msa_nvfp4_sm120.cu` —— kernel + merge + pybind（不依赖 torch 头）；
- `msa_nvfp4.py` —— JIT loader、stride 校验、split-K 参数、buffer 缓存；
- `__init__.py`。

JIT 产物缓存在 `/data/vllm-cache/root-cache/torch_extensions/msa_nvfp4_sm120/`。

### v3 原生 MoE（`patch_moe_sm120.py`）

根因：M3 在 `models/minimax_m3/nvidia/model.py` 声明
`qat_activation_key=kNvfp4DynamicToken`（per-token 激活 scale）。唯一实现该
scheme 的 FlashInfer TRT-LLM 后端是 SM100-only；SM120 上所有 `FLASHINFER_*`
被 oracle 过滤，最终落到 MARLIN W4A16（把 FP4 权重反量化回 BF16）。

方案：只改激活 scheme，不动 oracle 与内核：

```python
qat_activation_key=(
    kNvfp4DynamicToken
    if os.environ.get("M3_SM120_MOE_W4A4", "1") == "0"
    else kNvfp4Dynamic
),
```

`kNvfp4Dynamic`（per-tensor 动态外层 scale + 16 宽 E4M3 block scale）正是
FlashInfer CUTLASS SM120 NVFP4 MoE 原生实现的 scheme。本 checkpoint 的
FC1/FC2 `input_global_scale` 是 1.0 占位，per-token → per-tensor 只改外层 scale
粒度，不改量化载荷。

选中的证据：启动日志 `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`；
4 个 TP worker 各映射 5 段 `fused_moe_120.so`，内含 549 个
`MainloopSm120ArrayTmaWarpSpecializedBlockScaled` 符号。

### reasoning 层（`patches/reasoning/`，parser 烤进 `-reason5` 镜像，模板在模型目录）

M3 的思考块由 `<mm:think>…</mm:think>` 包裹。官方模板**不读** `thinking_mode`
（"Current thinking mode: adaptive." 是写死字符串），reasoning parser 的
`_initial_in_reasoning` 只认 `thinking_mode == "enabled"`。结果：档位失效、
流式推理文本漏进 `content`。

本工程用「模板 + parser」双补丁把 `reasoning_effort` 变成唯一开关：

- 模板按 effort 折算实际模式并预填标记（`enabled → <mm:think>`、`disabled → </mm:think>`）；
- parser 补丁用同一套映射设置 `_initial_in_reasoning`，保证流式/非流式一致；
- 默认（不传 effort）→ `none` → 关闭思考。

详见 [REASONING.md](REASONING.md)。

---

## 3. 运行时数据流

```
请求
 └─ OpenAI API (chat_completion/serving.py)
     ├─ _effective_chat_template_kwargs  ← 请求 chat_template_kwargs
     ├─ tokenizer.apply_chat_template     ← /data/MiniMax-M3.1-Flash/chat_template.jinja
     └─ _make_parser(...)                 ← patched minimax_m3_reasoning_parser.py
         │
         ▼
     vLLM engine (TP=4)
         ├─ indexer: Triton（v1 降级）
         ├─ sparse attention
         │    ├─ prefill: Triton 稀疏 attend（不反量化 nvfp4 常量，读 packed cache）
         │    └─ decode : msa_decode_nvfp4（v2，FP4 QK + FP8 PV）
         ├─ MoE: FlashInfer CUTLASS SM120 NVFP4 W4A4（v3）
         └─ draft: dspark 7 tokens，FlashInfer XQA + fp8 KV（Q 必须 bf16）
         │
         ▼
     detokenizer → reasoning parser（流式切分 content / reasoning）
```

KV / 并发参数：

- 主 KV：nvfp4（E2M1 码 + E4M3 block scale，132 B/128 tok/head → 72 B 页槽）；
- indexer KV：fp8；
- draft KV：fp8；
- `--max-model-len=1048576`、`--max-num-seqs=8`、`--block-size=128`、
  `--disable-hybrid-kv-cache-manager`。

---

## 4. 镜像构建流水线

`build/build_moe_native.sh`（当前生产镜像的构建脚本）：

1. 从**干净**的 `minimax-m31-flash` 启动临时容器；
2. 把 `/data/vllm-patch` 下 `models/`、`third_party/`、`v1/` 的 `.py` 覆盖到
   site-packages（vLLM）——这些是已含 gate 修正的完整文件；
3. 额外覆盖 flashinfer 的 `prims_ts/decode.py` 与 `context.py`（白名单加 `(12,0)`）；
4. 做三处断言：`120a` 已进 `decode_q8kv4/jit.py`、`12, 0` 已进 `decode.py`；
5. 复制两份源文件到 `/tmp/patchwork`，依次运行
   `patch_port.py` → `patch_native.py` → `patch_moe_sm120.py`，再写回；
6. 拷贝 `msa_sm120/` 三个文件（`.cu` 不在 `*.py` 循环范围内）；
7. 校验 `PORT_OK / NATIVE_OK / MOE_OK / PY_OK`；
8. `docker commit` 成新 tag，再起一个临时容器并
   `docker commit --change='ENTRYPOINT ["vllm","serve"]'` 修复入口
   （直接 commit 运行态会把 entrypoint 固化成 `/bin/sh`）。

`build_native.sh` / `build_port.sh` 是前两层的对应版本，结构相同、少一层补丁。

---

## 5. 补丁文件索引（`patches/vllm/`）

| 文件 | 层 | 作用 |
|---|---|---|
| `patch_port.py` | v1 | 主 attention / indexer 选择器加 SM120 降级 |
| `patch_native.py` | v2 | 在 `sparse_attention.py` 注入原生 decode 调用 |
| `patch_moe_sm120.py` | v3 | `model.py` 激活 scheme 切换 |
| `patch_m3.py` / `patch_m3_2.py` / `patch_m3_3.py` | 早期 nvfp4 线 | 放宽 SM100 gate 到 SM120（归档，供对照，非当前路线） |
| `models/minimax_m3/common/sparse_attention.py` | v1+v2 | 主 attention 选择器与 Triton 实现（已 patch） |
| `models/minimax_m3/common/indexer.py` | v1 | indexer 选择器（已 patch） |
| `models/minimax_m3/common/dspark.py` | 兼容 | SM120 上 FlashInfer 只接受 bf16 query，避免 FP8 query 触发 assert |
| `models/minimax_m3/common/ops/sparse_attn.py` | v1 | Triton 稀疏 kernel 的 NVFP4 packed 页读取 |
| `models/minimax_m3/nvidia/sparse_attention_msa.py` | 兼容 | MSA 设备判断（SM100/120 家族） |
| `models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py` | 兼容 | MSA decode 设备判断 |
| `models/minimax_m3/common/ops/msa_sm120/` | v2 | 自研 SM120 原生 kernel + loader |
| `flashinfer/attention/prims_ts/{decode,context}.py` | 兼容 | FlashInfer TS 白名单加 `(12,0)` |
| `third_party/fmha_sm100/jit.py` 等 4 个 | 兼容 | fmha_sm100 JIT 架构白名单加 `120a` |
| `third_party/fmha_sm100/cute/…`、`q8kv4_prefill_adapter.py` | 兼容 | CuTe 路径 capability 白名单 |
| `v1/attention/backends/flashinfer.py`、`flashinfer_ts.py` | 兼容 | vLLM FlashInfer 后端设备/能力判断 |

> 注意：`models/minimax_m3/nvidia/model.py` 不在覆盖列表中，它由
> `patch_moe_sm120.py` 在构建时直接改容器内文件。

---

## 6. 为什么不做的事

- **不放行 nvfp4 gate**：gate 只是白名单，真正缺的是 tcgen05，放行只会把崩溃
  从启动期推迟到 forward。
- **不启用 B12x**：flashinfer 的 SM12x CuTe-DSL MoE 支持 `swigluoai_uninterleave`
  （alpha 1.702 / limit 7.0，正是 M3 参数），但被 oracle / 激活白名单 /
  weight-prep 三道 gate 挡住；官方称 1.36×，列为后续探索项。
- **不反复重建整镜像**：reasoning 层只把 parser 覆盖进镜像（`-reason5` 仅两层
  COPY）；模板放在模型目录，改模板只覆盖文件 + 重启容器，回滚还原备份即可。
