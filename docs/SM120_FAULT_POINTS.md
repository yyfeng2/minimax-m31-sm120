# MiniMax-M3.1-Flash @ RTX PRO 6000 (SM120) 故障点全景分析

独立分析（未参考另一会话结论，源码为 node2 容器 `vllm-openai:minimax-m31-flash` 内的 vLLM 实时拷贝）。
源码副本：`C:\Users\yyfeng14\Desktop\azy\qoder_analysis\`（models/minimax_m3 + third_party/fmha_sm100 + v1/attention）。

## 0. 根本结论

SM120（RTX PRO 6000，消费级/工作站 Blackwell）**没有 TMEM（Tensor Memory）与 tcgen05/UMMA 矩阵引擎**，
而 M3 官方 nvfp4 路径的全部核心 kernel 都是 SM100/103/107 数据中心 Blackwell 专用：

- `third_party/fmha_sm100/csrc/include/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`：`SM100_TMEM_LOAD_32dp32b32x` / `SM100_TMEM_STORE_*`、TMEM 流水握手
- `decode_q8kv4/csrc/include/sm100/collective/sm100_fmha_mma_tma_warpspecialized.hpp`：`cute::SM100_MMA_F8F6F4_TS`（tcgen05 UMMA）、`UMMA::make_runtime_instr_desc`
- `cute/src/sm100/fwd/atten_fwd_nvfp4_kv.py`：`cute.arch.get_max_tmem_alloc_cols("sm_100")`、`SoftmaxSm100`、`sm100_utils`（CuTe-DSL 编译期也依赖 NVVM 对 sm120a 生成 tcgen05 → 工具链缺口，即 NVVM backend compilation failed）
- 随包 cutlass 头：`tmem_allocator_sm100.hpp`、`mma_sm100_umma.hpp`、`tmem_capacity_sm100.hpp` 等

且模型结构封死了绕路空间：`config.json` 60 层全部 `sparse_attention_freq=1`（全稀疏）、`sparse_block_size=128`、`sparse_topk_blocks=16`——
这正好命中 CUTLASS 后端的硬性几何要求（`page_size == _PAGE_SIZE(128) and topk_blocks == _TOPK(16)`），而稀疏 attention 是模型结构特性而非可关选项。

## 1. 致命故障点（硬阻塞，gate 放行无法解决）

| # | 位置 | 机制 | 触发阶段 | 表现 |
|---|---|---|---|---|
| D1 | `msa_cutlass_sparse_decode.py:325` `supports_cutlass_sparse_decode` 的 `or nvfp4` | nvfp4 KV 时无条件强制 CUTLASS，无视 `msa_decode_backend`；nvfp4 类明确「no Triton fallback / no BF16-query kernel」（sparse_attention_msa.py nvfp4 分支注释） | 60 层全部 decode forward | CUTLASS q8kv4 kernel 在 SM120 上 `unspecified launch failure`（cudaError 719，TMEM/tcgen05 指令非法） |
| D2 | `decode_q8kv4/csrc/include/sm100/collective/*` CUTLASS kernel 本体 | SM100_MMA_F8F6F4（tcgen05 UMMA）+ TMEM load/store，SM120 硬件缺失 | 同上 | launch failure；`120a` 只是 ptxas 编译目标，硬件不具备。QPML4/QMUL4（`mul.rn.satfinite.e2m1x4.e4m3x4`）在 sm120a 也被 ptxas 拒绝，JIT 静默降级 `fp16_fallback`——与 host 缓存里只存在 `decode_attention_plan_fp16_fallback_120a` 产物吻合 |
| P1 | nvfp4 prefill：`sparse_attention_msa.py` prefill 分支 → `fmha_sm100.sparse.sparse_atten_func` → cute CuTe-DSL kernel（`atten_fwd_nvfp4_kv.py` 等） | CuTe-DSL 用 NVVM/MLIR 编译，NVVM 无法为 sm120a 生成 tcgen05 → `NVVM backend compilation failed / unsupported operation`；kernel 类本身以 sm_100 为基准（`get_max_tmem_alloc_cols("sm_100")`） | 首个含 prefill 的 forward | 编译期直接失败。注：`_supports_blackwell_prefill`（cute/interface.py:696）只放行 (10,0)/(10,3)，sm120 会落入通用 nvfp4 变体——即此路径 |
| I1 | indexer 的 cute 打分 kernel：`cute/q8_indexer_interface.py`（`_SUPPORTED_CAPABILITIES={(10,0),(10,3),(10,7),(12,0)}` 已放行 12,0）但 kernel 源码 `cute/src/sm100/q8kv4_indexer_decode.py` / `q8kv8_indexer_prefill.py` 为 tcgen05 文件 | gate 放行 ≠ 硬件具备；CuTe-DSL 编译 sm120a 同样撞 NVVM 工具链缺口 | indexer decode/prefill 打分 | 预期与 P1 同型失败（尚未实测到，若 decode 主链先崩会掩盖它） |
| M1 | 硬件层：SM120 无 TMEM/tcgen05（ptxas 对 sm_120a 拒绝 tcgen05.alloc） | 上述全部 kernel 的共同根 | — | 任何「放行 gate」的 patch 都只把失败点往后推 |

## 2. 已修 / 已定位的 gate 类故障点（放行即可，但仅解决启动期）

以下 gate 原代码只认 family(100)，SM120 被挡；已通过 16 轮 patch 固化进 `minimax-m31-sm120` 镜像：

| 位置 | 原 gate | 影响 | 状态 |
|---|---|---|---|
| `flashinfer.py:474` 大页 block size | `family(100)` | 标准 FLASHINFER 只报 [16,32,64]，与主模型 [128] 无公共 block（`No common block size for 128`） | 已放行 family(120)（SM12x 有 XQA/trtllm decode，`supports_trtllm_attention` 本身支持） |
| `flashinfer.py:878` page_size assert | `family(100)` | `Unexpected FlashInfer page size 128 without trtllm-gen GQA` | 已放行 (100 or 120) |
| flashinfer pip 库 `attention/prims_ts/decode.py:55`、`context.py:65` | `_SUPPORTED_COMPUTE_CAPABILITIES=((10,0),(10,3),(10,7))` | draft FLASHINFER_TS `attention-ts decode requires SM100a/B200…` | 已加 (12,0) 放行；但随后撞上 prims_ts 的 NVVM/cutlass-dsl 工具链缺口（同 M1 根因）→ 决策改用标准 FLASHINFER |
| dspark draft KV block | FLASHINFER 仅 [16,32,64] | block size 冲突 | 随 flashinfer.py 大页放行解决（block 128 可公共） |

## 3. 被动降级点（不阻塞启动，SM120 上静默关闭）

| 位置 | gate | 后果 |
|---|---|---|
| `model.py:169` QKV fused epilogue | `family(100)` | SM120 关闭融合，正常回退，无故障 |
| `ops/dense_gemm1_mxfp8.py:569` GEMM1 融合 | `family(100)` | 回退分离路径 |
| `ops/finalize_allreduce_norm.py:740` 融合 MoE finalize | `family(100)` + NVLink multicast | 回退常规 all-reduce+norm；还依赖 multicast 探测（RTX PRO 6000 若无 NVSwitch multicast 也 False） |
| `msa_cutlass_sparse_decode.py:301` `_min_cutlass_batch_size` | 精确 (10,3)（B300/EAGLE3 专属） | 用默认最小 batch，无害 |
| `sparse_attention_msa.py:190` prefill_workspace | 精确 107（Rubin 专属） | 不启用，无害 |
| `cute/interface.py:696` `_supports_blackwell_prefill` | `in ((10,0),(10,3))` | SM120 落入通用 cute nvfp4 变体 → 进入 P1 失败 |
| `cute/interface.py` Rubin flags（`enable_fp16_softmax`/`enable_2x_fp8`） | `== _RUBIN_CAPABILITY(10,7)` | 自动关闭，无害 |
| `flashinfer.py:1072` draft 对 SM120 允许 nvfp4+XQA | 已含 family(120) 分支 | 官方本就预期 SM120 用 XQA + fp8 路径 |

## 4. 非阻塞告警

- `symm_mem.py:66`：`SYMM_MEM_ALL_REDUCE_MAX_SIZES` 无 capability 12.0 条目 → SymmMem 通信器禁用（日志 WARNING），回退 NCCL，不致命。
- `custom_all_reduce.py:110`：custom all-reduce 精确匹配 (10,0)/(10,3) → SM120 禁用，回退 NCCL。TP=4 时 all-reduce 全走 NCCL，性能损失非故障。
- `index_topk.py:712`：split_k 打分**只对 (12,0) 启用**——官方已为 SM120 写过的优化点，佐证官方对 SM120 的定位是 Triton/XQA 路线。

## 5. 结构性结论

1. **nvfp4 KV 与 SM120 不可共存**。`indexer.py:534-547` 的 nvfp4 白名单虽写了 120，但其依赖的三件套（indexer cute kernel、CUTLASS decode、cute prefill）全部需要 TMEM/tcgen05。放行白名单 = 把启动期错误推迟到 forward。
2. **官方给 SM120 留的路是 fp8-KV 路线**：indexer 回退 `MiniMaxM3IndexerTritonImpl`（Triton，`use_msa` 要求 is_sm100 故 SM120 自动落此分支）；MSA 回退 Triton sparse decode/prefill（`use_cutlass_decode=False` 时 `_bf16_query` + Triton）；draft 用 FLASHINFER/XQA（SM12x 有原生支持）；b12x backend 是官方 SM12x paged attention（fp8 KV 限定）。
3. **唯一现实的可行配置**：`--kv-cache-dtype fp8` + `indexer_kv_dtype: fp8` + 保持 `--block-size 128` + dspark（FLASHINFER/XQA，fp8 KV）+ `--num-speculative-tokens 7` 不变。预期显著损失：长上下文 KV 容量减半、prefill/decode 吞吐低于 H100/B200 设计值，但这是 SM120 硬件能力对应的路径。
4. **patch 已修项只解决了「启动期 gate」**，解决不了「kernel 硬件依赖」。要跑 nvfp4 只有两条重路径：改写 CUTLASS kernel 到 SM120 的 wgmma/HMMA 路线（工程量=重写 FMHA），或等 NVIDIA/vLLM 上游补 SM120 的 nvfp4 FMHA。

## 6. 建议验证清单（fp8 路线）

- [x] `MiniMaxM3MSADecodeBackend = Literal["triton","cutlass"]`（vllm/config/attention.py，容器实测）；**默认即 `triton`**（common/sparse_attention.py:406）；compose 未显式指定 cutlass
- [ ] 确认 `indexer_kv_dtype=fp8` 时 indexer 选 Triton impl（日志 `selected Triton (no fmha_sm100)`）
- [ ] **fp8 路线关键确认**：fp8+family(120)+backend 默认 cutlass 时 `supports_cutlass_sparse_decode` 仍返回 True → 仍会进 SM100 CUTLASS kernel 崩溃。需将 `--msa-decode-backend triton`（或等效配置）显式设为 triton，并确认 compose 中该参数名
- [ ] dspark 的 draft attention 在 SM120 用 FLASHINFER/XQA 实测 block 128
- [ ] 若 nvfp4 为硬性要求：本卡无解，需换 B 系卡（SM100/103）

（注：fp8 的 decode 默认仍走 CUTLASS，而 CUTLASS 模板本身 SM100-only；fp8 路线必须让 decode 落 triton——nvfp4 档则被 `or nvfp4` 强制锁死 CUTLASS，连 triton 选项都不给，这正是第 5 节结论 1 的代码依据。）
