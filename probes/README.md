# 探针与内核测试

这些文件用于证明「SM120 到底支持什么」，以及自研内核的数值正确性。
运行环境是 node2 上的 vLLM 容器（需要 torch / CUDA / triton），
大多通过 `torch.utils.cpp_extension.load` 现场编译。

## 指令能力探针

| 文件 | 验证内容 | 结论 |
|---|---|---|
| `sm120_native_mma_probe.cu` | SM120 是否有可用的块缩放 MMA | 有 `mma.sync kind::mxf4nvf4 / mxf8f6f4`，无 tcgen05 |
| `sm120_kind_matrix_probe.cu` | 各 `kind::` × scale_vec × 数据类型的组合矩阵 | 确定 QK 用 4X、PV 用 1X |
| `sm120_cvt_probe.cu` / `sm120_cvt_probe2.cu` | E2M1/E4M3 转换指令与 packing 行为 | 确定 nibble 顺序 |
| `test_mma_probe.py` | fragment 布局（哪个 lane 持有哪些元素） | 布局与 kernel 实现一致 |
| `triton_sm120_fp4_probe.py` | Triton 是否把 `tl.dot_scaled` 降级到 SM120 原生 MMA | MXFP4 可用；NVFP4（E4M3 scale, group16）不可用 |
| `triton_mixed_probe.py` | 单例 PTX：e4m3 × e2m1 的 `dot_scaled` 降级 | 不能直接用于 Q8KV4 混合 |

## NVFP4 KV 布局探针

| 文件 | 验证内容 |
|---|---|
| `nvfp4_write_probe.py` | `fused_minimax_m3_qknorm_rope_kv_insert` 是否支持 nvfp4 cache；72 B 页槽的 nibble 顺序 |
| `nvfp4_layout_probe.py` | 单个非零元素的确定性布局（nibble、scale 字节覆盖范围、K/V slot） |
| `nvfp4_layout_probe2.py` | 与 fmha reader 一致的 scale 布局：K `token*8 + d_block`；V `(token//4)*32 + token%4 + 4*d_block` |
| `nvfp4_triton_probe.py` | Triton 反量化与 torch 参考（按真实页布局）逐元素对比 |

## 内核正确性

| 文件 | 验证内容 |
|---|---|
| `test_nvfp4_attn.py` | Triton nvfp4 稀疏 attend / decode 与反量化参考对比 |
| `test_msa_native.py` | 自研 SM120 kernel：staged 字节级对比、QK scores、最终输出、split-K 自洽性 |

`test_msa_native.py` 的验收数字记录在
[`../docs/SM120_NATIVE_REPORT.md`](../docs/SM120_NATIVE_REPORT.md) §2。
