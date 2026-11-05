# vLLM 补丁树

这些文件是**覆盖式补丁**：构建时原样复制进镜像的
`/usr/local/lib/python3.12/dist-packages/`（见 `../../build/build_moe_native.sh`），
之后再用三个补丁器做文本级注入。

## 应用顺序

```bash
# 1) 同步覆盖树到 node2（目录结构保持）
scp -r * root@192.168.20.12:/data/vllm-patch/

# 2) 构建（会自动按顺序跑补丁器）
ssh root@192.168.20.12 'bash /data/build_moe_native.sh'
```

补丁器顺序（脚本内固定）：

1. `patch_port.py` —— v1 降级层（marker `# [m31-sm120-port]`）
2. `patch_native.py` —— v2 原生注意力（marker `# [m31-sm120-native]`）
3. `patch_moe_sm120.py` —— v3 MoE（marker `[m31-sm120-moe]`）

三者都幂等：重复执行会打印 `ALREADY_PATCHED` / `MOE_PATCH_ALREADY`。

## 目录 → 镜像路径

| 仓库路径 | 镜像内路径 |
|---|---|
| `models/...` | `site-packages/vllm/models/...` |
| `third_party/fmha_sm100/...` | `site-packages/vllm/third_party/fmha_sm100/...` |
| `v1/attention/backends/...` | `site-packages/vllm/v1/attention/backends/...` |
| `flashinfer/attention/prims_ts/...` | `site-packages/flashinfer/attention/prims_ts/...` |

`patch_m3*.py` 是早期 nvfp4 路线（放宽 SM100 gate）的归档，**不在当前构建流程中**，
仅用于对照与复现崩溃。

`legacy/` 是更早的一次性产物：`m31_sm120_port_v1.py`（v1 降级逻辑的独立脚本版，
已被 `patch_port.py` 取代）、`commit_patch.sh` / `commit_patch2.sh`（早期直接把
文件 `docker cp` 进容器再 commit 的做法，已被 `build_*.sh` 取代）。

## 运行时开关

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `M3_SM120_NATIVE` | 1 | 0 = Triton 在线反量化 decode |
| `M3_SM120_MOE_W4A4` | 1 | 0 = MARLIN W4A16（checkpoint 原 scheme） |
| `M3_SM120_KEEP_NVFP4` | 未设 | 1 = 保留官方 nvfp4 路由（SM120 会崩） |
