# 运维手册（部署 / 验收 / 回滚 / 排障）

所有命令默认在 **node2**（`root@192.168.20.12`）执行。SSH 没有配别名，
请直接用 IP（`ssh node2` 会报 `Could not resolve hostname`）。

---

## 1. 部署当前生产版本

```bash
# 1) compose（本仓库 deploy/docker-compose-m31.yml 与 /data 下一致）
scp deploy/docker-compose-m31.yml root@192.168.20.12:/data/

# 2) 起服务（项目名必须是 data，与现网一致）
ssh root@192.168.20.12
cd /data && docker compose -p data -f docker-compose-m31.yml up -d

# 3) 等 ready（首次约 210–270 s：JIT、CUDA graph capture）
for i in $(seq 1 40); do
  c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)
  [ "$c" = 200 ] && { echo READY; break; }; sleep 10
done
```

关键参数（均在 compose 的 `command:` 中）：

```
--max-model-len=1048576        1M 上下文
--max-num-seqs=8               并发上限
--gpu-memory-utilization=0.95  KV 26.78 GiB / 1,633,920 tokens
--kv-cache-dtype=nvfp4         只有 nvfp4 让 1M 落在显存可达区间
--block-size=128
--reasoning-parser=minimax_m3
--tool-call-parser=minimax_m3
--enable-auto-tool-choice
--speculative-config={"method":"dspark","num_speculative_tokens":7,...}
```

**不要**写 `--default-chat-template-kwargs=...`：思考默认关闭，由客户端
`reasoning_effort` 显式开启（见 [REASONING.md](REASONING.md)）。

---

## 2. 基础验收

```bash
KEY=REPLACE_WITH_YOUR_API_KEY

# a) 健康
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health

# b) 普通对话（默认不思考）
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash","messages":[{"role":"user","content":"用两句话解释光合作用。"}],"max_tokens":1200}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);m=d["choices"][0]["message"];r=(m.get("reasoning") or m.get("reasoning_content") or "");t=m.get("content") or "";print("content",len(t),"reasoning",len(r),"leak","mm:think" in t)'

# c) 开启思考
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash","messages":[{"role":"user","content":"9.11 和 9.9 哪个大？"}],"chat_template_kwargs":{"reasoning_effort":"high"},"max_tokens":1500}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);m=d["choices"][0]["message"];r=(m.get("reasoning") or m.get("reasoning_content") or "");t=m.get("content") or "";print("content",len(t),"reasoning",len(r),"r_tokens",d["usage"]["completion_tokens_details"]["reasoning_tokens"])'
```

期望（2026-10-24 实测）：b) `content>0 / reasoning=0 / leak=False`；
c) `reasoning>0 / content>0 / leak=False`。

流式验收要逐帧检查 `delta.content` 中不出现 `<mm:think>` / `</mm:think>`，
且思考帧只出现在 `delta.reasoning`（本 build 字段名；部分版本为
`delta.reasoning_content`，客户端应两者都读）。

---

## 3. 回滚

### 3.1 只回滚某一层（同一镜像，改 env）

```bash
# 回退原生注意力 → Triton 反量化（v1）
M3_NATIVE=0 docker compose -p data -f docker-compose-m31.yml up -d

# 回退原生 MoE → MARLIN W4A16（v2 镜像行为）
M3_MOE_W4A4=0 docker compose -p data -f docker-compose-m31.yml up -d
```

### 3.2 回滚到旧镜像

```bash
M31_IMAGE=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-native \
  docker compose -p data -f docker-compose-m31.yml up -d

M31_IMAGE=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-port \
  docker compose -p data -f docker-compose-m31.yml up -d
```

### 3.3 回滚 reasoning 补丁

生产自 2026-10-25 起用 `-reason5` 镜像：parser 烤在镜像里，模板由**模型目录**那份
决定（`--chat-template` 默认注释）。回滚分两种：

```bash
# A) 回到官方模板（最快）：模型目录换 .orig 后重启
cp /data/MiniMax-M3.1-Flash/chat_template.jinja.orig /data/MiniMax-M3.1-Flash/chat_template.jinja
cd /data && docker compose -p data -f docker-compose-m31.yml up -d --force-recreate

# B) 回到 v1 挂载版：compose 用 backup（parser 改回 /data/m3_patches bind mount）
cp /data/docker-compose-m31.yml.bak-<ts> /data/docker-compose-m31.yml
cp /data/m31_patches/chat_template.jinja.effort5 /data/MiniMax-M3.1-Flash/chat_template.jinja
cd /data && docker compose -p data -f docker-compose-m31.yml up -d --force-recreate
```

模板与 parser 的完整历史备份：`.orig`（官方）、`.pre-effort`（首轮）、
`.pre-maxfix`、`.pre-defaultoff`（默认关闭前的版本）、`.effort-final`（v1，开关两态）、
`.effort5`（v2，五档，当前）。

---

## 4. 重建镜像

### 4.1 准备补丁树

构建脚本从 `/data/vllm-patch` 读取覆盖文件，因此先把仓库内容同步过去：

```bash
# 本地 → node2
scp -r patches/vllm/* root@192.168.20.12:/data/vllm-patch/
```

`/data/vllm-patch` 目录结构必须保持：

```
vllm-patch/
├── models/minimax_m3/...          # 覆盖到 site-packages/vllm/models/...
├── third_party/fmha_sm100/...     # 覆盖到 site-packages/vllm/third_party/...
├── v1/attention/backends/...      # 覆盖到 site-packages/vllm/v1/...
├── flashinfer/attention/prims_ts/ # 覆盖到 site-packages/flashinfer/...
└── patch_port.py patch_native.py patch_moe_sm120.py
```

### 4.2 构建

```bash
bash /data/build_moe_native.sh     # 生产镜像（v1+v2+v3）
# 或前两层：
bash /data/build_native.sh         # v1+v2
bash /data/build_port.sh           # 仅 v1
```

脚本内置校验，构建日志必须看到：

```
COPY_ALL_OK
FLASHINFER_OK
PORT_OK
NATIVE_OK      # 仅 build_native / build_moe_native
MOE_OK         # 仅 build_moe_native
PY_OK
moe-native 镜像 sha256:… entrypoint=[vllm serve]
```

任何一行缺失都说明覆盖或补丁没生效，**不要**部署该镜像。

### 4.3 构建期常见坑

- 直接 `docker commit` 运行态容器会把 entrypoint 固化成 `/bin/sh`；脚本末尾的
  `--change='ENTRYPOINT ["vllm","serve"]'` 是必需的。
- `*.py` 覆盖循环不包含 `.cu`，`msa_sm120/` 三个文件需单独拷贝（脚本已处理）。
- 补丁器全部是幂等的（按 marker 判断），但锚点变化会 `assert` 失败而不是静默错改；
  升级 vLLM 基础镜像后需先核对锚点。

### 4.4 思考层镜像（`-reason5`，2026-10-25 起生产）

五档 reasoning 补丁已烤进镜像，不再用 bind mount：

```bash
bash build/build_reasoning_image.sh root@192.168.20.12
```

产出 `…:minimax-m31-sm120-moe-native-reason5`，内含覆盖后的
`vllm/reasoning/minimax_m3_reasoning_parser.py`，以及一份备用模板
`/opt/m31/chat_template.jinja`。

生产以**模型目录**的 `/data/MiniMax-M3.1-Flash/chat_template.jinja`（md5 `f90d3923…`）
为权威副本：compose 里的 `--chat-template` 已注释，渲染走模型目录那份，改模板只需
覆盖该文件 + `up -d --force-recreate`。若要强制以镜像 `/opt/m31` 为准，取消 compose
里 `--chat-template=/opt/m31/chat_template.jinja` 的注释再重启。

校验输出必须看到 `PY_OK / TEMPLATE_OK / PARSER_OK / PARSER_STATE_OK / VERIFY_OK`
以及两个 md5（模板 `f90d3923…`、parser `14e6fc54…`）。本机没有 docker bridge，
Dockerfile 里不能写 `RUN`，校验放在 build 之后用 `docker run --network=none` 执行。

---

## 5. 排障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `unrecognized arguments: --default-chat-template-args=…`，容器重启循环 | 旗标拼写错误，正确是 `--default-chat-template-kwargs`（带 s） | 修正 compose 后 `up -d` |
| 启动时报 `Failed to plan fmha_sm100: invalid argument` | 仍选中 MSA 类（只降级了 KV dtype，没退出 MSA） | 确认镜像是 `-port` 及以后（v1.1 起直接退出 MSA） |
| 日志出现 `GPU does not have native support for FP4` / `Using MARLIN` | MoE 补丁未生效或 `M3_SM120_MOE_W4A4=0` | 查 `build_moe_native.sh` 的 `MOE_OK`；确认 env |
| 改了模板但行为没变 | vLLM 启动时缓存/编译模板 | 覆盖模型目录那份后 `up -d --force-recreate` |
| 改了 `/opt/m31/chat_template.jinja` 无效 | 生产没用 `--chat-template`，渲染读模型目录 | 改模型目录那份；或取消 compose 里 `--chat-template` 的注释后重启 |
| 流式时思考文本出现在 `content` | 请求用了 `thinking_mode=adaptive`，本 build 下绕过 reasoning parser | 改用 `reasoning_effort=high/max` |
| `content` 为 `null` / 空 | `max_tokens` 被思考块耗尽 | 提高 `max_tokens`（≥1500 / max 档 ≥3000）或加 `thinking_token_budget` |
| 首次启动很久没 ready | JIT + CUDA graph capture | 正常需 210–270 s，观察 `docker logs -f minimax-m31` |
| 1M 上下文启动失败并提示 KV 不足 | 显存预算不够（fp8 KV 物理出界） | 保持 `--kv-cache-dtype=nvfp4`、util 0.95；不要用 fp8 KV 跑 1M |
| SSH `Could not resolve hostname node2` | 未配 SSH 别名 | 用 `root@192.168.20.12` |

日志与指标：

```bash
docker logs -f minimax-m31 | grep -E "Using '.*' NvFp4 MoE backend|m31-sm120|error|Error"
curl -s http://127.0.0.1:8000/metrics | grep -E 'prompt_tokens_total|iteration_tokens_total_sum|kv_cache_usage'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

---

## 6. 压测方法（避免假数）

`scripts/prefill_bench.py`、`scripts/bench_m31.py`、`scripts/bench2.py` 均已按下列
口径实现，复测时请保持一致：

1. **每个请求一条独立 PRNG token-id prompt**，seed 必须包含请求序号（`rep`）。
   否则第二轮起 prefix cache（APC）全命中，曾报出 25 万 tok/s 的假数据。
2. TTFT / prefill 吞吐必须与 `/metrics` 的 `prompt_tokens_total`、
   `iteration_tokens_total_sum` 增量**交叉校验**；两者对不上就作废。
3. 请求内 `temperature=0`，每个负载跑 3 次取中位。
4. 长 prompt 并发（如 32k×8）受 chunked prefill 上限 8192 约束，表现为排队而非
   聚合吞吐提升，报告时区分「TTFT 中位/最差」与「聚合 prefill」。

参考结果（单请求冷 APC）：4k→0.51 s、16k→2.05 s、64k→8.4 s、128k→17.2 s，
稳定 7.6–8.0k tok/s。

---

## 7. 备份清单（node2）

| 路径 | 内容 |
|---|---|
| `/data/MiniMax-M3.1-Flash/chat_template.jinja` | 当前思考模板（权威，五档，md5 `f90d3923…`） |
| 同目录 `.orig` / `.pre-effort` / `.pre-maxfix` / `.pre-defaultoff` / `.effort-final` / `.effort5` | 模板历史版本 |
| `/data/m3_patches/minimax_m3_reasoning_parser.py` | v2 parser（已烤进镜像，仅回滚挂载用） |
| `/data/m3_patches/minimax_m3_reasoning_parser.py.orig` | 官方 parser |
| `/data/m31_patches/` | 模板副本（`.effort5` 与模型目录同 md5） |
| `/data/docker-compose-m31.yml` | 当前生产 compose（`.bak` 为上一版） |
| `/data/minimax-m31-sm120-moe-native.tar.gz` | 生产镜像离线包（10.1 GB） |
| `/data/vllm-patch/` | 构建用覆盖树（本仓库 `patches/vllm/` 的来源） |
