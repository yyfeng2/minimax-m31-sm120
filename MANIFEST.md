# 归档清单与来源映射

本目录是 node2 上部署产物的离线归档。除下列「未归档」项外，运行所需的代码、
补丁、配置、脚本、报告均已收齐（约 1.5 MB，122 个文件；`MANIFEST.md5` 自身
不纳入校验）。

## 目录 → node2 来源

| 本地目录 | node2 来源 | 说明 |
|---|---|---|
| `deploy/` | `/data/docker-compose*.yml` | 各阶段 compose；`docker-compose-m31.yml` 为当前生产 |
| `build/` | `/data/build_*.sh`、`install_native.sh`、`check_loaded.sh` | 镜像构建与临时安装脚本 |
| `patches/vllm/` | `/data/vllm-patch/` | 覆盖进镜像的源文件 + 三个补丁器 |
| `patches/vllm/patch_m3*.py` | `/data/patch_m3*.py` | 早期 nvfp4 路线归档 |
| `patches/vllm/legacy/` | 本地 `Desktop\azy\` | 更早的一次性脚本 |
| `patches/reasoning/` | `/data/m3_patches/`、模型目录、`/data/m31_patches/` | 五档模板（`.effort5` 为生产版）+ parser 补丁 + 挂载版安装脚本（已停用，供回滚） |
| `scripts/` | `/data/*.py` | 验收、压测、MoE 探针 |
| `probes/` | `/data/*probe*.py`、`*.cu` + 本地测试 | 指令能力与内核正确性验证 |
| `results/` | `/data/moe_*.json`、`prefill_native.json` | 实测原始数据 |
| `docs/` | `/data/*.md` + 本地 | 报告、架构、运维、思考层文档 |
| `tools/recon/` | 本地 `Desktop\azy\query_*.sh` 等 | 排查 gate 时用的侦察脚本 |

## 未归档（体积或环境相关）

| 项 | 位置 | 原因 |
|---|---|---|
| 生产镜像离线包（10.1 GB） | node2 `/data/minimax-m31-sm120-moe-native.tar.gz` | 体积；可用 `build/build_moe_native.sh` 重建 |
| 模型权重（约 240 GB） | node2 `/data/MiniMax-M3.1-Flash` | 体积；需单独下载 |
| vLLM 源码快照 | node2 `/data/vllm_src.tgz`、`/data/nvfp4add/` | 上游代码，可重新导出 |
| 启动/下载日志 | node2 `/data/*.log` | 临时数据 |
| JIT / CUDA graph 缓存 | node2 `/data/vllm-cache/` | 可重建 |

## 关键外部依赖

- **镜像底座**：`docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-flash`
- **模型**：`MiniMax/MiniMax-M3.1-Flash`（本地 `/data/MiniMax-M3.1-Flash`）
- **GPU**：4× NVIDIA RTX PRO 6000 Blackwell（SM120，无 tcgen05），TP=4
- **软件**：CUDA 13.x、torch 2.14、vLLM `0.1.dev23341+g8744dd2fb`、flashinfer 0.7.2

## 完整性校验

```bash
md5sum -c MANIFEST.md5
```
