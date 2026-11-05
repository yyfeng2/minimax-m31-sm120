# MiniMax-M3.1-Flash on RTX PRO 6000 (SM120) — 移植与原生内核工程

本项目是 node2（`192.168.20.12`，4× RTX PRO 6000 Blackwell / SM120，TP=4）上
部署 MiniMax-M3.1-Flash 的完整工程归档：源码补丁、镜像构建脚本、部署 compose、
验证/压测脚本、排障与性能报告。

**当前生产镜像**：`docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-moe-native-reason5`
（`sha256:373b65bae955…`，25.8 GB）
**当前部署文件**：`/data/docker-compose-m31.yml`（本仓库 `deploy/` 内同名副本）
**服务状态**：1M 上下文、NVFP4 KV、SM120 原生 FP4/FP8 稀疏 decode、
FlashInfer CUTLASS SM120 NVFP4 W4A4 MoE、思考默认关闭、五档 `reasoning_effort`
（parser 烤在镜像里；模板以模型目录那份为权威，无 bind mount）。

---

## 1. 为什么需要这套补丁

官方 MiniMax-M3 的 NVFP4 主链是 **SM100 系专用**：

- CuTe-DSL prefill、CUTLASS Q8KV4 decode、NVVM `sm120a` codegen 均依赖
  **tcgen05 / TMEM（UMMA）**；
- RTX PRO 6000（SM120）**没有 tcgen05**，放行 capability gate 只会把崩溃推迟到
  forward 阶段（`Failed to plan fmha_sm100: invalid argument`、CUTLASS decode 崩）。

本工程的做法不是「继续放行」，而是换成 SM120 真正具备的能力：

| 层 | 改动 | 结果 |
|---|---|---|
| v1 降级层 | 主 attention 完全退出 MSA，改走 Triton 稀疏后端；indexer 改 Triton；KV 按 nvfp4 读路径 | 服务能稳定启动并推理 |
| v2 原生注意力 | 自研 CUDA 内核，decode 直接消费打包的 NVFP4 KV（不反量化），QK 用 `mma.sync kind::mxf4nvf4`，PV 用 `kind::mxf8f6f4` | 注意力 kernel 2.16×（T=1），端到端持平（注意力仅占单步 ~6%） |
| v3 原生 MoE | 激活 scheme 由 per-token 改为 `kNvfp4Dynamic`，命中 FlashInfer CUTLASS SM120 NVFP4 W4A4 | MoE 全链路不再反量化；tok/s 与 MARLIN 持平，logprob 略优 |
| reasoning 层 | 模板 + parser 补丁，`reasoning_effort` 五档（low/medium/high/xhigh/max） | 思考默认关闭；开启后思考进 `reasoning` 字段，正文干净；parser 烤进镜像，模板随模型目录 |

详细根因见 [docs/SM120_FAULT_POINTS.md](docs/SM120_FAULT_POINTS.md)，
设计与数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 2. 目录结构

```
minimax-m31-sm120/
├── README.md                      本文件
├── MANIFEST.md / MANIFEST.md5     归档清单、node2 来源映射、完整性校验
├── deploy/                        各阶段 docker compose（m31 为当前生产配置）
├── build/                         镜像构建脚本（从干净基础镜像重放补丁）
├── patches/
│   ├── vllm/                      覆盖进镜像的 vLLM/flashinfer 源文件 + 三个补丁器
│   │   ├── patch_port.py          v1 降级层
│   │   ├── patch_native.py        v2 原生注意力注入
│   │   ├── patch_moe_sm120.py     v3 MoE 激活 scheme
│   │   ├── legacy/                早期一次性脚本（已被上面的补丁器取代）
│   │   └── ...                    被替换的源码（models/、third_party/、v1/、flashinfer/）
│   └── reasoning/                 思考模板与 reasoning parser 补丁 + 一键安装脚本
├── scripts/                       验证、压测、MoE 探针、思考测试、启动监控
├── probes/                        硬件能力探针与内核正确性测试（CUDA / Triton）
├── results/                       实测结果 JSON（MoE 精度、分布、prefill）
├── tools/recon/                   定位 gate 时用的侦察脚本（过程存档）
└── docs/
    ├── ARCHITECTURE.md            硬件约束、三层补丁设计、数据流、补丁索引
    ├── RUNBOOK.md                 部署/回滚/排障/验收命令
    ├── REASONING.md               思考开关设计（模板 + parser）与调用方式
    ├── SM120_FAULT_POINTS.md      故障点全量清单（已修/硬阻塞/可绕路）
    ├── SM120_NATIVE_REPORT.md     v2 原生注意力报告（正确性 + 微基准）
    ├── SM120_MOE_REPORT.md        v3 原生 MoE 报告（根因 + 精度 + 性能）
    └── m31_todo.txt               部署阶段的工作清单存档
```

---

## 3. 快速使用（node2）

```bash
# 启动当前生产配置
cd /data && docker compose -p data -f docker-compose-m31.yml up -d

# 健康检查（首次启动约 210–270 s，含编译与 CUDA graph capture）
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health

# 普通请求（默认不思考）
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer REPLACE_WITH_YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash","messages":[{"role":"user","content":"你好"}],"max_tokens":512}'

# 开启思考（high 或 max；low/none/不传 = 关闭）
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer REPLACE_WITH_YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash","messages":[{"role":"user","content":"9.11 和 9.9 哪个大？"}],
       "chat_template_kwargs":{"reasoning_effort":"high"},"max_tokens":1500}'
```

思考开关、流式切分与 `max_tokens` 注意事项见 [docs/REASONING.md](docs/REASONING.md)。

---

## 4. 关键实测数据

### 4.1 上下文与显存

| 项 | 值 |
|---|---|
| `max_model_len` | **1,048,576**（模型原生 `max_position_embeddings=1048576`，`rope_theta=1e7`，无需插值） |
| `gpu_memory_utilization` | 0.95 |
| KV cache（nvfp4） | 26.78 GiB → **1,633,920 tokens** |
| 每卡权重 / 峰值激活 / CUDA graph | 59.56 GiB / 10.44 GiB / 1.76 GiB |

1M 上下文是显存问题：nvfp4 KV 比 fp8 省 38%，只有 nvfp4 能让 1M 落在可达区间
（fp8 需 23.75 GiB，物理出界）。

### 4.2 Prefill / TTFT（单请求，冷 APC）

| prompt | TTFT |
|---|---|
| 4k | 0.51 s |
| 16k | 2.05 s |
| 64k | 8.4 s |
| 128k | 17.2 s |

稳定 **7.6–8.0k tok/s**。chunked prefill 上限 8192，并发场景聚合 prefill 不超过
单请求值（32k×8 表现为排队，TTFT 中位 18.6 s）。

### 4.3 Decode 吞吐（temperature=0，3 次中位）

| 负载 | tok/s |
|---|---|
| solo 1×384 | ~126–132 |
| g16 ×256 | ~770 |
| g64 ×256 | ~1,388 |

MoE W4A4 与 MARLIN W4A16 基本持平（权重带宽主导）。

> **精度口径修订（2026-10-25）**：原先记录的「精度 logprob 略优
> （−3.8878 vs −3.9440）」经配对检验**不成立**——paired t=0.698、
> 95% CI [-0.102, +0.214] 含 0，且 27% 的位置 |Δ|>1 nat（MoE 路由离散翻转
> 所致）。mean logprob 不能作跨配置精度判据；精度请用
> `scripts/task_accuracy.py` 的下游任务正确率。详见
> [docs/SM120_MOE_REPORT.md §4.1](docs/SM120_MOE_REPORT.md)。

### 4.4 思考层

| 请求 | content | reasoning | 泄漏 |
|---|---|---|---|
| 不传 effort | 80 字 | 0 | 无 |
| `low` | 83 字 | 0 | 无 |
| `high` | 258 字 | 394（r_tokens 114） | 无 |
| 流式 `high` | 218 字 | 482 | 无 |

---

## 5. 镜像谱系

| 镜像 tag | 内容 | 状态 |
|---|---|---|
| `minimax-m31-flash` | 官方多模态底座（干净） | 构建起点 |
| `minimax-m31-sm120` | nvfp4 gate 放宽（另一条线） | 真实 forward 崩在 CUTLASS decode |
| `minimax-m31-sm120-port` | v1 降级层（fp8/Triton） | 可用，A/B 基线 |
| `minimax-m31-sm120-native` | v1 + v2 原生注意力 | 可用，A/B 基线 |
| `minimax-m31-sm120-moe-native` | v1 + v2 + v3 原生 MoE | 基线（A/B 对照） |
| **`minimax-m31-sm120-moe-native-reason5`** | 上者 + 五档 reasoning（内置 parser + 模型目录模板，`/opt/m31` 为备用副本） | **当前生产** |

前四个镜像是逐层 `docker commit` 叠加的产物；`-reason5` 由 `build/Dockerfile.reason5`
（仅 COPY，无 RUN）从 `-moe-native` 构建，`build/build_reasoning_image.sh` 一键重放并
在 `docker run --network=none` 容器里校验。离线镜像 tar 在 node2
`/data/minimax-m31-sm120-moe-native.tar.gz`（10.1 GB，不入库；不含 reason5）。

---

## 6. 运行开关（环境变量）

| 变量 | 默认 | 作用 |
|---|---|---|
| `M3_SM120_NATIVE` | 1 | 0 = 回退 Triton 在线反量化注意力（v1 行为），用于 A/B |
| `M3_SM120_MOE_W4A4` | 1 | 0 = 回退 checkpoint 原 per-token scheme（MARLIN W4A16），用于 A/B |
| `M3_SM120_KEEP_NVFP4` | 未设 | 1 = 保留官方 nvfp4 路由（SM120 上会崩，仅供复现） |
| `M3_NATIVE` / `M3_MOE_W4A4` | — | compose 层别名，分别注入上面两个开关 |

---

## 7. 演进时间线（2026-10-24 至 10-25）

| 阶段 | 动作 | 结果 |
|---|---|---|
| 1 | 官方 nvfp4 链路放行 SM120 gate（镜像 `minimax-m31-sm120`） | 启动能过，真实 forward 崩在 CUTLASS decode |
| 2 | v1 降级层：退出 MSA，fp8/Triton（`-sm120-port`） | 服务可用；开启 A/B 基线 |
| 3 | v2 自研 SM120 FP4/FP8 decode kernel（`-sm120-native`） | kernel 2.16×（T=1），数值误差在 E4M3-P/V 固有量化底 |
| 4 | v3 MoE 原生 W4A4（`-sm120-moe-native`） | 全链路不反量化；logprob 略优，tok/s 持平 |
| 5 | 上下文提到 1M + util 0.95 | KV 26.78 GiB / 1,633,920 tokens；prefill 7.6–8.0k tok/s |
| 6 | 思考层模板 + parser 补丁，`reasoning_effort` 两态 | 流式/非流式切分一致，无标签泄漏 |
| 7 | 思考默认关闭，仅显式 effort 开启 | 默认正文直出；显式档位才进 `reasoning` |
| 8 | 恢复官方五档（low/medium/high/xhigh/max）+ 补丁烤进镜像（`-reason5`） | 五档均可独立触发；`/tokenize` 验证渲染用模型目录模板（low 213 vs 官方 194 tokens），无 bind mount |

---

## 8. 已知限制与边界

- **不能走官方 nvfp4 主链**：SM120 无 tcgen05，gate 放行只会推迟崩溃，不要尝试。
- **B12x 尚未启用**：flashinfer 0.7.2 的 SM12x CuTe-DSL MoE 支持
  `swigluoai_uninterleave`，但被三道 gate 挡住（oracle 不选、激活白名单不收、
  weight-prep 只收 silu/gelu_tanh/relu2）。官方称 SM12x 上 B12x W4A16 比 Marlin
  快 1.36×，属后续可探索项。
- **prefill 无并行度优化**：自研 decode 内核的 split-K 在 prefill 下恒为 1
  （`_num_chunks` 中 T 是 chunk 长度而非前缀长度）。
- **思考层**：模板/parser 补丁在启动时编译缓存，改完必须重启容器；
  `max_tokens` 不足时思考块会耗尽预算，`content` 为空。
- **API key 已脱敏**：文档与脚本中的 `REPLACE_WITH_YOUR_API_KEY`
  为占位符，部署前请替换为你自己的 key（可用
  `sed -i 's/REPLACE_WITH_YOUR_API_KEY/<你的key>/g' -r README.md deploy/ docs/ scripts/ patches/`）。
