# MiniMax-M3.1-Flash on RTX PRO 6000 (SM120) — 原生 FP4/FP8 稀疏 decode

日期 2026-10-24 · node2 (192.168.20.12) · 4× RTX PRO 6000 Blackwell · TP=4 · vLLM
镜像 `docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-native`

## 1. 做了什么

`MiniMaxM3SparseTritonNvfp4Impl.forward` 的 **decode** 分支在 SM120 上改走自研
CUDA kernel，直接消费 packing 后的 NVFP4 KV cache（**不反量化**）：

| 阶段 | 指令 | 说明 |
|---|---|---|
| QK^T | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` | 读 cache 里的 E2M1 码 + E4M3/16 block scale；Q 现场量化成 NVFP4 |
| PV | `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0` | P 用 E4M3 + 2 的幂 E8M0 scale；V 码在 smem 里转 E4M3（NVFP4 的 V scale 沿 head 维、与 PV 收缩轴不对齐，无法直接做 scale-native MMA，与官方 SM100 Q8KV4 的做法一致） |
| split-K | 自研 merge kernel | grid `(T, Hkv, nchunk)`；`w_c = exp2(m_c − M)`，`out = Σ w_c O_c / Σ w_c l_c`。`nchunk` 按 ~256 CTA 自适应（batch1/spec8 → 16，T=64 → 4，T≥256 → 1 免 merge） |

prefill 仍走 Triton（不需要张量核）；`M3_SM120_NATIVE=0` 可运行时切回 Triton 在线反量化路径。

文件：
- `models/minimax_m3/common/ops/msa_sm120/msa_nvfp4_sm120.cu`（kernel + merge + pybind，无 torch 头依赖）
- `models/minimax_m3/common/ops/msa_sm120/msa_nvfp4.py`（JIT loader、stride 校验、split-K 参数、buffer 缓存）
- `/data/patch_native.py`（注入点 patch，供 `build_native.sh` 重放）

## 2. 正确性证据（容器内实测）

| 检查 | 结果 |
|---|---|
| staged Q / Qsf / K / Ksf / Vt 字节级对比 | mismatch = 0（Q pad 区除外，不参与计算） |
| QK scores vs torch 参考 | `max|err| = 6.924e-08`（|s| max 1.192） |
| 输出 vs 精确 softmax 参考 | `max|err| = 1.53e-02`, mean|out| = 2.35e-02 —— 即 E4M3-P + E4M3-V 的固有量化底（单 P 项 1.21e-2、单 V 项 1.17e-2） |
| split-K 自洽（chunks 1 vs 2/8/16） | max 2.1e-03 / mean 3.2e-04（来自每个 chunk 独立的 P E4M3 量化，非 merge 误差） |
| PV atom（随机逐行 E8M0 scale） | 与参考差 5.6e-17 |

## 3. 性能：注意力 kernel 本体（真实 shape：Hq=16, Hkv=1, hd=128, topk=16）

`/tmp/msa_sm120/bench_attn.py`，CUDA event 中位数，单位 µs/次/层/rank：

| shape | Triton 在线反量化 | 原生 FP4/FP8 | 倍数 |
|---|---|---|---|
| T=1 (batch 1) | 88.1 | 40.8 | **2.16×** |
| T=8 (batch1 + spec8) | 87.4 | 41.4 | **2.11×** |
| T=64 (batch 8) | 109.4 | 61.7 | **1.77×** |
| T=256 (batch 32) | 273.6 | 148.9 | **1.84×** |

## 4. 端到端 tok/s（同一 compose，仅换镜像；3 次取中位）

| 负载 | -port（Triton 反量化） | -native | 差异 |
|---|---|---|---|
| solo 384 tok | 126 tok/s | 121 tok/s | 噪声内 |
| g4 × 256 tok | 307 tok/s | 314 tok/s | 噪声内 |
| g8 × 256 tok | 450 tok/s | 470 tok/s | +4%（噪声内） |

结论：attention 只占 decode 单步的约 6%（60 层 × 41µs ≈ 2.5ms / ~40ms），
所以端到端差异被 MoE/权重读取淹没。要吃到 SM120 FP4 的大头，下一步必须把
MoE 的 MARLIN nvfp4 反量化路径换成 FlashInfer SM120 原生 `moe_nvfp4_w4a4`。

## 5. 复现 / 回退

```bash
# 重建镜像（含 v1 降级层 + v2 原生 kernel）
bash /data/build_native.sh

# 部署
docker compose -p data -f /data/docker-compose-native.yml up -d

# 回退到 Triton 反量化（同镜像）
docker compose -p data -f /data/docker-compose-native.yml up -d  # 并设 M3_NATIVE=0
# 或回到旧 tag
M31_IMAGE=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-port \
  docker compose -p data -f /data/docker-compose-native.yml up -d
```

JIT 产物缓存在 `/data/vllm-cache/root-cache/torch_extensions/msa_nvfp4_sm120/`，
换容器不重编。
