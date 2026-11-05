# 部署配置

| 文件 | 用途 |
|---|---|
| **`docker-compose-m31.yml`** | **当前生产**：`minimax-m31-sm120-moe-native-reason5`，1M 上下文、nvfp4 KV、五档 reasoning（parser 烤进镜像、模板在模型目录，无 bind mount） |
| `docker-compose-native.yml` | v1+v2 阶段：原生注意力，512K 上下文 |
| `docker-compose-port.yml` | v1 阶段：fp8 KV + Triton（A/B 基线） |
| `docker-compose-nvfp4.yml` | 更早的 nvfp4 KV + Triton 反量化（A/B 基线） |
| `docker-compose-moe.yml` | v3 首次部署（512K，已被 m31 取代） |
| `docker-compose-1m.yml` | 1M 提档实验（util 0.92，已被 m31 取代） |
| `docker-compose.yml` | 最早的门槛放宽版（`minimax-m31-sm120`，会崩，仅供复现） |

统一启动方式（项目名必须为 `data`）：

```bash
cd /data && docker compose -p data -f docker-compose-m31.yml up -d
```

镜像可通过 `M31_IMAGE` 覆盖，A/B 开关为 `M3_NATIVE` / `M3_MOE_W4A4`
（见 `../docs/RUNBOOK.md` §3）。
