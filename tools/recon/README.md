# 侦察脚本（recon）

这些脚本是最初定位 SM120 断点时用的排查工具，跑在 node2 的 vLLM 容器里
（多数靠 `docker exec minimax-m31 bash /path/xxx.sh`，或先 `docker cp` 进去）。
路径写死为 `/usr/local/lib/python3.12/dist-packages/vllm`，只适用于该镜像。

| 脚本 | 用途 |
|---|---|
| `locate_nvfp4.sh` | 搜索 NVFP4 / MSA 的 capability gate 与报错原文（SM100-only 提示） |
| `query_nvfp4*.sh` | 查 `is_device_capability_family` 定义、GPU capability、`platforms/` 与 `vllm/` 两套路径差异 |
| `query_jit.sh` | 查 fmha_sm100 JIT 的 `_DEFAULT_ARCHS` / arch 常量里是否只有 100/103/107 |
| `query_archfinal.sh` | 追 `_target_arch` 的最终来源（决定是否生成 `sm120a`） |
| `query_msa.sh` | 查 `supports_cutlass_sparse_decode` 的实现与设备条件 |
| `query_msa2.sh` | 查 `sparse_attention_msa.py` 指定行段（可在本机经 ssh 执行） |
| `export_orig.sh` | 从干净镜像导出 6 个原始文件，作为 patch 基线 |
| `export_nvfp4add.sh` | 导出 nvfp4 链路其余相关文件到 `/data/nvfp4add` 供离线分析 |

结论汇总在 [../../docs/SM120_FAULT_POINTS.md](../../docs/SM120_FAULT_POINTS.md)，
因此这些脚本只作为「证据是怎么挖出来的」的过程存档，复现问题时可重跑。
