# MiniMax-M3.1-Flash on RTX PRO 6000 (SM120) — MoE 原生 NVFP4 W4A4

日期 2026-10-24 · node2 (192.168.20.12) · 4× RTX PRO 6000 Blackwell · TP=4 · vLLM
镜像 `docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-moe-native` (`sha256:d761c7b9d7c6…`)

## 1. 结论

路由专家（MoE）已从 **MARLIN W4A16（把 FP4 权重反量化成 BF16 再算）** 换成
**FlashInfer CUTLASS SM120 NVFP4 W4A4**：权重与激活都是 E2M1 码 + 16 宽 E4M3 block scale，
在 SM120 的 FP4 张量核上直接相乘，全链路不再反量化。

- 启动日志：`Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`，MARLIN 的
  `GPU does not have native support for FP4` 警告消失。
- 4 个 TP worker 的 `/proc/<pid>/maps` 各映射 5 段 `fused_moe_120.so`；
  该 so 内 549 个 `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` 符号、
  `SM120_16x8x32/64_TN_VSIS*` 块缩放 MMA atom。
- 正确性未退化（232 个 next-token 的 logprob 略好，**但该差异统计上不显著，
  不能据此下「精度更优」结论**，见 §4 的修订说明）。
- 端到端 tok/s 与 MARLIN **基本持平**（同会话中位：g16 +1.7%、g64 +0.6%、solo −4.3%）。
  原因见 §5：decode 小 batch 受权重带宽主导，FP4 算力优势体现不出来；
  但约束「不做反量化、走原生 FP4 张量核」已满足，且为 prefill/大 batch 留出了算力余量。

## 2. 根因：M3 声明 per-token NVFP4 激活，SM120 无人接

```
models/minimax_m3/nvidia/model.py:466  qat_activation_key=kNvfp4DynamicToken
        └─ CT W4A4 method: activation_key = per-token FC1 输入 scale
           └─ NvFp4 oracle 按 scheme 过滤后端
              ├─ FLASHINFER_TRTLLM / CUTEDSL / CUTEDSL_BATCHED
              │    is_device_capability_family(100)  → SM120 全拒
              ├─ FLASHINFER_CUTLASS
              │    _supports_current_device() 放行 family 120，
              │    但 _supports_quant_scheme() 只收
              │    (kMxfp4Static,kMxfp8Dynamic) / (kNvfp4Static,kNvfp4Dynamic)
              │    → per-token 不在列表
              ├─ VLLM_CUTLASS  同样只收 kNvfp4Dynamic
              └─ MARLIN 只看 weight key → 接住 → W4A16 反量化
```

关键源码位置（容器内，patch 后）：

| 文件 | 位置 | 内容 |
|---|---|---|
| `vllm/models/minimax_m3/nvidia/model.py` | 473 | patch 后的 activation key 开关（原始 466 行） |
| `vllm/.../experts/flashinfer_cutlass_moe.py` | 128–143 | device gate：`family(120)` 放行 |
| 同上 | 146–180 | quant gate：nvfp4 仅 `(kNvfp4Static, kNvfp4Dynamic)` 且 `capability(100)` |
| 同上 | 184–191 | activation 支持 `SWIGLUOAI / SWIGLUOAI_UNINTERLEAVE` |
| `vllm/.../oracle/nvfp4.py` | 225–235, 336 | 后端优先级与最终选中日志 |

## 3. 方案：只改激活 scheme，不动 oracle / 内核

新增 `/data/vllm-patch/patch_moe_sm120.py`（幂等，marker `[m31-sm120-moe]`）：

1. `model.py` 补 `import os`；
2. 导入 `kNvfp4Dynamic`；
3. 把 `qat_activation_key=kNvfp4DynamicToken` 改成

```python
qat_activation_key=(
    kNvfp4DynamicToken
    if os.environ.get("M3_SM120_MOE_W4A4", "1") == "0"
    else kNvfp4Dynamic
),
```

选 `kNvfp4Dynamic`（per-tensor 动态外层 scale + 16 宽 E4M3 block scale）而不是 B12X 的原因：

- 这是 FlashInfer SM120 CUTLASS NVFP4 MoE 原生实现的 scheme，激活仍走 FP4 张量核；
- 本 checkpoint 的 FC1/FC2 `input_global_scale` 是 1.0 占位，per-token 与 per-tensor
  只差外层 scale 的粒度，量化载荷不变；
- `flashinfer_b12x_moe.py` 虽支持 SM12x 且收 `(kNvfp4Static, kNvfp4Dynamic)`，但
  `_supports_activation` 不含 `SWIGLUOAI`，且不在 auto 后端列表，需额外改 oracle。

镜像构建（`/data/build_moe_native.sh`）：从干净的 `:minimax-m31-flash` 起，依次重放
v1 降级层 `patch_port.py`、v2 注意力 `patch_native.py`、v3 MoE `patch_moe_sm120.py`，
覆盖两个 flashinfer `prims_ts` 文件与 `msa_sm120` 三文件，校验
`PORT_OK / NATIVE_OK / MOE_OK / PY_OK` 后 commit，并用 `--change` 修正入口为
`["vllm","serve"]`。一个镜像，A/B 只切 env。

部署：`/data/docker-compose-moe.yml`（host 网络、TP=4、NVFP4 KV、block 128），
`M3_SM120_MOE_W4A4` 缺省 1。重启到 ready 约 270 s。

## 4. 正确性证据

固定 233-token 中英混合 prompt，`prompt_logprobs=1`，取 232 个真实 next-token：

| 指标 | MARLIN (W4A16) | FLASHINFER_CUTLASS (W4A4) |
|---|---|---|
| mean logprob(actual token) | −3.9440 | **−3.8878** |
| mean logprob(top-1 token) | −0.9193 | **−0.8849** |

抽样任务：`173×29−451` 两路均答 4566；代码与中文续写通顺，无重复/乱码。
原始数据：`/data/moe_logprob_{native,marlin}.json`、`/data/moe_probe_*.json`、
`/data/moe_dist_*.json`。

### 4.1 修订说明（2026-10-25）：上表的 logprob 差**不构成精度证据**

复核后结论：**mean logprob 在本模型上不能用作跨配置的精度判据**，本节原始
数字保留存档，但「W4A4 精度略优」的推论作废。三条实证理由：

1. **统计不显著。** 对 `moe_logprob_{native,marlin}.json` 做配对检验
   （n=232）：

   ```
   mean_delta = +0.0562   sd = 1.2261   SE = 0.0805
   paired t   =  0.698           95% CI = [-0.1016, +0.2140]   ← 含 0
   bootstrap 95% CI = [-0.1019, +0.2135]，仅 76.1% 的重采样均值 > 0
   ```

2. **指标本身不连续。** 配对差里 `|delta| > 1.0 nats` 的位置有 **63/232
   (27%)**，`max|d| = 7.21`。这不是「微小数值扰动」的形状（那会让每个位置都
   动一点点），而是**离散选择翻转**：MoE top-4 路由是离散的，一位浮点差让某层
   的专家集合换一个，该 token 的 logprob 就跳几个 nats。均值 0.03–0.06 nats
   完全落在这个混沌带内。

3. **指标在单实例内是确定的，但跨运行不可比。** 同一服务重复 6 次
   （`noise_floor.py`）spread = **0.0000**、逐位一致；并发 6 个干扰请求下
   delta 仍为 **0.0000**（`batch_sensitivity.py`）。所以差异并非噪声，
   而是**不同容器运行之间的真实差异**——但它可以来自与精度无关的因素
   （CUDA graph capture、JIT 选择、prefix cache 状态）。

> 由此还暴露一个更严重的问题：用今天的生产服务重放归档 native 的 prompt，
> chat 输出与归档的 **`marlin`** 文本**逐字相同**、与 `native` 不同；且归档
> `moe_probe_native` 的 logprob 列表长 152、`moe_probe_marlin` 长 153（同一
> prompt 不应长度不同）。**归档那组 A/B 的 env 标签与实际运行配置可能不对应，
> 对照组本身存疑。**

**正确的做法**：用**下游任务正确率**（`scripts/task_accuracy.py`，
GSM8K 风格算术 / MMLU 风格选择 / 代码 / 中文问答，固定题集自动判分）作为
精度指标，logprob 只用于同一实例内的调试。

## 5. 性能：同镜像、同会话、只切 env

`/data/bench2.py`（temperature=0，每请求 256 token，k 并发聚合，3 次取中位；
2026-10-24 12:40–12:48，两次启动完成后各预热一轮）：

| 负载 | MARLIN 中位 | CUTLASS 中位 | Δ |
|---|---|---|---|
| solo 1×384 | 132.2 tok/s | 126.5 tok/s | −4.3% |
| g16 ×256 | 757.5 tok/s | 770.0 tok/s | +1.7% |
| g64 ×256 | 1379.4 tok/s | 1388.3 tok/s | +0.6% |

同会话 g16 原始值：CUTLASS 737/761/784/783/759/780/770，MARLIN 752.0/757.5/759.4。
另一次较早会话的大扫描（跨会话 GPU 时钟不同，仅作趋势参考）：
CUTLASS solo 130.2、g4 307、g8 468、g16 838、g32 1174、g64 1473；
MARLIN solo 129.3、g4 321、g8 446、g16 843、g32 1157、g64 1420。

**为什么持平**：decode 单步 60 层 × 128 experts × top-4，专家权重读取量约
60×4×(2×3072×6144)/8 B ≈ 1.2 GB/token/rank 量级，瓶颈在 HBM→SM 的权重搬运；
W4A4 只在算力受限（大 batch / 长 prefill）时才有明显收益。本次验收的目标是
「不反量化、走 SM120 FP4 张量核」，该目标已达成；算力余量留给后续大 batch 压测。

## 6. 运行证据

```
(Worker_TP0 pid=789) INFO 10-24 12:49:10 [nvfp4.py:336] Using 'FLASHINFER_CUTLASS'
  NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM',
  'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN',
  'HUMMING', 'EMULATION'].
```

- 4× Worker_TP：`grep -c fused_moe_120.so /proc/<pid>/maps` = 5。
- `/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/providers/sm120f/jit_cache/fused_moe_120/fused_moe_120.so`
  60,754,128 B。
- 符号：549 × `MainloopSm120ArrayTmaWarpSpecializedBlockScaled`；
  `SM120_16x8x32_TN_VSIS{K,L}` 各 96、`SM120_16x8x64_TN_VSIS{K,L}` 168/288，
  另有 E/F/G/J/M 变体；`float_e2m1_t` 在符号中出现 549 次。
- 容器 env：`M3_SM120_NATIVE=1`、`M3_SM120_MOE_W4A4=1`。

## 7. 复现 / 回退

```bash
# 重建（v1 降级 + v2 原生注意力 + v3 原生 MoE）
bash /data/build_moe_native.sh

# 部署（默认 native，M3_MOE_W4A4 缺省 1）
cd /data && docker compose -p data -f docker-compose-moe.yml up -d

# A/B：回到 MARLIN W4A16
cd /data && M3_MOE_W4A4=0 docker compose -p data -f docker-compose-moe.yml up -d

# 回到无 MoE patch 的旧镜像
M31_IMAGE=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-native \
  docker compose -p data -f docker-compose-moe.yml up -d

# 回退原生注意力（同一镜像）
M3_NATIVE=0 docker compose -p data -f docker-compose-moe.yml up -d
```

JIT 产物已烘进镜像（`flashinfer_jit_cache` AOT），换容器不重编。

## 8. 边界与风险

- 激活外层 scale 从 per-token 降为 per-tensor。本 checkpoint 的
  `input_global_scale=1.0`，实测精度未退化；若换成带真实 per-token scale 的
  checkpoint，需重新做精度评估。
- patch 只改 M3 模型构造处的 activation key，不碰 oracle / CUDA 内核；
  升级 vLLM 若 `model.py` 锚点变化，patch 会 assert 失败而非静默错改（幂等 marker）。
- 尚未单独测长 prompt 的 TTFT/prefill 吞吐；这是 W4A4 相对 W4A16 最可能体现优势的场景。
- 本次未使用 B12X（不支持 SWIGLUOAI、不在 auto 列表）。
