# 思考层（reasoning）设计与调用

M3 每次生成会先输出思考块 `<mm:think>…</mm:think>`（常见 900–3400 token），
再由正文。本工程把「是否思考 / 思考多深」统一收敛到 `reasoning_effort` 一个开关，
并且**默认关闭**。

> **2026-10-25 更新（v2）**：官方 `low / medium / high / xhigh / max` 五档已全部
> 恢复，五档都是「思考开启」，深度依次递增；`none / minimal` 与不传仍然关闭。
> 详见 §2.5。旧版把 `low` 误映射为关闭、`medium/high/xhigh` 共用同一句指令。

---

## 1. 官方实现的两个缺陷

1. **模板不读 `thinking_mode`**：官方 `chat_template.jinja` 里
   `Current thinking mode: adaptive.` 是写死的字符串（全文 0 处引用该变量），
   所以 `thinking_mode=enabled/disabled/adaptive` 三个档位行为完全相同。
2. **parser 只认一个值**：`MiniMaxM3ReasoningParser.__init__` 中

   ```python
   self._initial_in_reasoning = chat_kwargs.get("thinking_mode") == "enabled"
   ```

   只有显式 `enabled` 才从「思考中」开始切分。实测本 build 下
   `thinking_mode=adaptive` 的**流式**请求会绕过 reasoning parser，
   推理文本直接漏进 `content`（非流式反而是对的）。

此外模型是「先想完再答」：如果 `<mm:think>` 没闭合，正文不会开始。因此
`max_tokens` 太小时整轮预算被思考耗尽，`content` 为空——这是预算问题，不是
parser bug。

---

## 2. 方案：effort 驱动，模板与 parser 同源

### 2.1 档位映射

| 请求参数 | 实际模式 | 生成前缀 | 行为 |
|---|---|---|---|
| 不传 `reasoning_effort` | disabled | `</mm:think>` | **默认**，正文直出 |
| `reasoning_effort=none` / `minimal` | disabled | `</mm:think>` | 不思考 |
| `reasoning_effort=low` | enabled | `<mm:think>` | 轻思考：几句话内给结论 |
| `reasoning_effort=medium` | enabled | `<mm:think>` | 逐步推理，保持紧凑 |
| `reasoning_effort=high` | enabled | `<mm:think>` | 逐步推理 + 考虑备选 + 自检 |
| `reasoning_effort=xhigh` | enabled | `<mm:think>` | 拆解问题、验证中间结果与边界 |
| `reasoning_effort=max` | enabled | `<mm:think>` | 最大深度：穷举、验证、复核 |
| `thinking_mode=enabled`（显式） | enabled | `<mm:think>` | 兼容旧调用（不带档位） |
| `thinking_mode=disabled`（显式） | disabled | `</mm:think>` | 兼容旧调用 |

显式 `thinking_mode` 优先；否则由 effort 折算；都没有 → disabled。
未知的非空 effort 一律按 `medium` 处理（宁可多想，不可静默关闭）。

> 与官方默认值的差异：官方 API 缺省 `reasoning_effort=max`，本部署缺省**关闭**。
> 这是有意的部署策略（单卡 126 tok/s，思考 token 直接吃墙钟），需要时逐请求打开。

### 2.2 模板补丁（`patches/reasoning/chat_template.jinja.effort5`）

在 `<thinking_instructions>` 段落中把 effort / thinking_mode 归一化：

```jinja
{%- set _effort = (reasoning_effort | default(effort, true)) | string | lower | trim -%}
{%- if _effort in ['', 'default', 'auto', 'adaptive'] -%}
    {%- set _effort = 'default' -%}
{%- elif _effort in ['none', 'minimal', 'off', 'no'] -%}
    {%- set _effort = 'none' -%}
{%- elif _effort not in ['low', 'medium', 'high', 'xhigh', 'max'] -%}
    {%- set _effort = 'medium' -%}
{%- endif -%}
{%- set _thinking_mode = (thinking_mode | default('', true)) | string | lower | trim -%}
{%- if _thinking_mode in ['', 'adaptive', 'auto'] -%}
    {%- if _effort == 'default' or _effort == 'none' -%}{%- set _thinking_mode = 'disabled' -%}
    {%- else -%}{%- set _thinking_mode = 'enabled' -%}{%- endif -%}
{%- endif -%}
```

指令文本五档各一句（low 简短、medium 紧凑、high 自检、xhigh 验证中间结果、
max 最大深度），并保留官方原生的档位标签——**只在思考开启且确实给了档位时**
输出，避免与预填的 `</mm:think>` 自相矛盾：

```jinja
{%- if _thinking_mode == 'enabled' and _effort != 'default' -%}
    {{- '\n\n<effort>' ~ _effort ~ '</effort>' }}
{%- endif -%}
```

生成提示（generation prompt）使用同一套折算，并预填标记：

```jinja
{%- if _tm == 'enabled' -%}{{- think_begin_token }}
{%- elif _tm == 'disabled' -%}{{- think_end_token }}{%- endif -%}
```

预填的作用是让模型从正确状态开始：关闭时直接跳过思考块，开启时保证
`<mm:think>` 一定出现，parser 才有确定的起点。

### 2.3 parser 补丁（`patches/reasoning/minimax_m3_reasoning_parser.effort5.py`）

把 `__init__` 里的单行替换为同一套映射：

```python
_tm  = str(chat_kwargs.get("thinking_mode") or "").strip().lower()
_eff = str(chat_kwargs.get("reasoning_effort") or chat_kwargs.get("effort") or "").strip().lower()
_on, _off = ("low", "medium", "high", "xhigh", "max"), ("none", "minimal", "off", "no")
if _tm in ("enabled", "on", "true", "1", "enable"):
    self._initial_in_reasoning = True
elif _tm in ("disabled", "off", "false", "0", "disable"):
    self._initial_in_reasoning = False
elif _eff in _on:
    self._initial_in_reasoning = True
elif _eff in _off:
    self._initial_in_reasoning = False
elif _eff:                       # 未知档位：模板按 medium 处理，这里必须一致
    self._initial_in_reasoning = True
else:
    self._initial_in_reasoning = False   # 默认不思考
```

模板与 parser 必须**同时**存在且映射一致，否则会出现「模板预填了
`<mm:think>`、parser 却以为在正文」的错位，表现为推理文本进入 `content`。

### 2.4 部署方式：parser 烤进镜像，模板放模型目录（2026-10-25 起）

生产镜像 `…:minimax-m31-sm120-moe-native-reason5` 由 `build/build_reasoning_image.sh`
在 `-moe-native` 之上构建：parser 烤进镜像，模板另存一份备用副本。

| 文件 | 权威位置 | 生效方式 |
|---|---|---|
| 五档模板 | `/data/MiniMax-M3.1-Flash/chat_template.jinja` | 模型目录挂载进容器，tokenizer 启动时读取 |
| 五档 parser | 镜像内 `.../vllm/reasoning/minimax_m3_reasoning_parser.py` | 覆盖官方实现，自动生效 |

镜像里另存了一份同 md5 的模板副本 `/opt/m31/chat_template.jinja`，**默认不启用**，
只在想强制以镜像为准时用 `--chat-template=/opt/m31/chat_template.jinja` 覆盖
（compose 里保留成注释行）。

> 2026-10-25 13:20 起模板**直接替换模型目录那一份**：这样任何读该模型的工具
> （tokenizer 自检、其他框架、离线渲染）拿到的都是五档版，不再有「模型目录是官方、
> 运行时是 /opt」的两份分叉。parser 不能放模型目录，所以仍烤在镜像里。

compose 现状（`deploy/docker-compose-m31.yml`）：

```yaml
image: ${M31_IMAGE:-…:minimax-m31-sm120-moe-native-reason5}
volumes:
  - /data/MiniMax-M3.1-Flash:/data/MiniMax-M3.1-Flash   # 权重 + 五档模板
  - /data/vllm-cache/root-cache:/root/.cache
  # 无 m3_patches 挂载
command:
  # - --chat-template=/opt/m31/chat_template.jinja      # 需要时取消注释
```

改模板 = 改模型目录那份（并同步仓库/镜像副本）；改 parser = 重建镜像：

```bash
cp patches/reasoning/chat_template.jinja.effort5 \
   /data/MiniMax-M3.1-Flash/chat_template.jinja
bash build/build_reasoning_image.sh root@192.168.20.12   # parser 或镜像内副本变了才需要
cd /data && docker compose -p data -f docker-compose-m31.yml up -d --force-recreate
```

**改完必须重启**：vLLM 在启动时编译/缓存模板。

> 更早的 v1（2026-10-24 至 10-25 13:12）用 bind mount 挂 parser 到 site-packages；
> 该方式已停用，`/data/m3_patches` 与各模板备份仅作回滚用。

### 2.5 为什么和硬件无关（SM120 适用性）

档位是**纯 prompt 侧**机制，不碰 kernel、不碰 KV 布局：

- 模板只往系统提示里写一行 `Current thinking mode / Thinking effort` 和
  `<effort>LEVEL</effort>`，外加预填一个 `<mm:think>` / `</mm:think>` 标记，
  全部走正常 tokenizer 路径；
- 我们的 SM120 三件套（Triton sparse attention / 自写 NVFP4-FP8 decode kernel /
  FlashInfer CUTLASS NVFP4 MoE）只决定**每个 token 怎么算**，与生成多少思考 token
  无关，因此五档全部可用，无需重新编译 CUDA 代码；
- 唯一代价是 token 数与墙钟。本部署 KV 实测 26.78 GiB / 1,633,920 token
  ≈ **17.2 KiB/token**；8 并发 × 2 万思考 token ≈ 2.7 GiB，相对 KV 池可忽略。
  真正的成本是时间：单请求 decode ≈ 126 tok/s，`max` 档 3k 思考 token 约多花
  25 s。

结论：**可以在现有 RTX PRO 6000 (SM120) 配置上直接适配**，不需要改镜像或内核，
只需换模板 + parser 并重启一次。

### 2.6 离线渲染验证（2026-10-25）

用容器内 tokenizer 直接渲染（`scripts/render_template_check.py`，14 个用例，
不经过服务）：

| 用例 | 指令档位 | `<effort>` | 前缀 |
|---|---|---|---|
| 不传 | default | 无 | `</mm:think>` |
| none / minimal | none | 无 | `</mm:think>` |
| low / medium / high / xhigh / max | 各自独立文案 | 对应档位 | `<mm:think>` |
| `effort=low`（别名） | low | `low` | `<mm:think>` |
| 未知值 `ultra` | 归一为 medium | `medium` | `<mm:think>` |
| `thinking_mode=disabled` + `max` | disabled | **无**（不再矛盾） | `</mm:think>` |

parser 侧同样用真实 tokenizer 实例化验证 `_initial_in_reasoning`：low/medium/
high/xhigh/max/未知值 → True，none/minimal/不传 → False。

> 注：第一版渲染脚本传的是 `chat_template_kwargs=kw`，transformers 5.x 不再
> 解包该参数（vLLM 侧是 `**resolved_kwargs` 直接展开），导致 11 个用例渲染结果
> 完全相同——那是脚本 bug，不是模板 bug；已改成 `**kw` 并加上 `effort` 别名与
> 未知值用例。

---

## 3. 调用示例

```bash
KEY=REPLACE_WITH_YOUR_API_KEY
URL=http://192.168.20.12:8000/v1/chat/completions

# 默认：不思考
curl -s $URL -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash","messages":[{"role":"user","content":"你好"}],"max_tokens":512}'

# 开启思考（推荐：直接放在顶层，vLLM 协议原生字段）
curl -s $URL -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash","messages":[{"role":"user","content":"9.11 和 9.9 哪个大？"}],
       "reasoning_effort":"high","max_tokens":1500}'

# 五档任选：low / medium / high / xhigh / max
curl -s $URL -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash","messages":[{"role":"user","content":"9.11 和 9.9 哪个大？"}],
       "reasoning_effort":"low","max_tokens":1500}'

# 等价写法：走 chat_template_kwargs（顶层字段未设置时才会生效）
curl -s $URL -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash","messages":[{"role":"user","content":"9.11 和 9.9 哪个大？"}],
       "chat_template_kwargs":{"reasoning_effort":"xhigh"},"max_tokens":3000}'

# 流式：思考帧在 delta.reasoning，正文帧在 delta.content
curl -sN $URL -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash","messages":[{"role":"user","content":"9.11 和 9.9 哪个大？"}],
       "reasoning_effort":"high","max_tokens":1500,"stream":true}'
```

客户端渲染建议：把 `reasoning` 收进可折叠的「已深度思考」块（类似 GLM
客户端），`content` 作为正文；不要假设 `content` 立即有值。

> **字段名注意**：本镜像（vLLM `0.1.dev23341`）的非流式消息与流式 `delta`
> 都使用 **`reasoning`**，而不是常见文档里的 `reasoning_content`。
> 兼容写法：`msg.get("reasoning") or msg.get("reasoning_content")`。

`max_tokens` 参考：简单问答 ≥512；开放题/数学题 ≥1500；`max` 档建议 ≥3000。

### 3.1 硬性思考预算 `thinking_token_budget`（可选，本 build 自带）

本镜像（vLLM `0.1.dev23341`）的 `ChatCompletionRequest` 还暴露
`thinking_token_budget: int | None`，由 `vllm/v1/worker/gpu/sample/thinking_budget.py`
在采样侧强制：思考 token 达到预算即强制写出 `</mm:think>`（取自 parser 的
`reasoning_end_str`），正文继续生成。它与 effort 正交，可用来给墙钟兜底：

| 档位 | 建议 `thinking_token_budget` |
|---|---|
| low | 256–512 |
| medium | 1024 |
| high | 2048 |
| xhigh / max | 4096（或留空，交给模型自决） |

> 该字段在协议与采样层都已存在，但本部署尚未实测；随下一次重启的冒烟一起验证。
> 只设 `max_tokens` 不设预算时，思考有可能吃光整轮预算导致 `content` 为空。

---

## 4. 验收记录

### 4.1 v1（2026-10-24 重启后实测，「关闭 / 开启」两态）

| 场景 | content | reasoning | reasoning_tokens | 泄漏 |
|---|---|---|---|---|
| 默认（不传）非流式 | 80 | 0 | 0 | 否 |
| `low` 非流式（v1 映射为关闭） | 83 | 0 | 0 | 否 |
| `high` 非流式 | 258 | 394 | 114 | 否 |
| 默认 流式 | 100 | 0 | — | 否 |
| `high` 流式 | 218 | 482 | — | 否 |
| 女驸马 @8192 | 929 | 1454 | — | 否 |
| 显式 `thinking_mode=adaptive` @8192 | 810 | 3258 | — | 否 |

另外验证过：`max` 档（reasoning 822 / r_tokens 214）、多轮 `high→low` 切换、
`low`/`high` 下的 tool calls 均正常。注意 v1 里 `low` 是**关闭**，与官方语义相反，
v2 已修正。

### 4.2 v2（2026-10-25，五档；已重启 prod 并实测）

`apply_reasoning_patch.sh` 执行：铺文件 → 备份 → `--force-recreate` → READY →
冒烟。同一 prompt（17 只羊，`temperature=0`，`max_tokens=4096`）逐档各一条：

| 档位 | content | reasoning（字符） | reasoning_tokens | 泄漏 |
|---|---|---|---|---|
| 默认（不传） | 156 | 0 | 0 | 否 |
| `low` | 160 | 493 | — | 否 |
| `medium` | 210 | 373 | 137 | 否 |
| `high` | 188 | 383 | — | 否 |
| `xhigh` | 320 | 823 | 279 | 否 |
| `max` | 329 | 862 | — | 否 |

结论：五档都能独立进入思考、正文与思考分流正确、无 `<mm:think>` 泄漏；
`xhigh`/`max` 明显更长。但**单样本不足以证明「档位严格单调」**——本例
`low`(493) > `medium`(373) ≈ `high`(383)，说明长度主要由模型自身判断决定，
本文的档位差异是「提示强度」而非硬配额。需要硬上限请用 §3.1 的
`thinking_token_budget`。

安装后校验：模板 md5 `f90d3923…`、parser md5 `14e6fc54…`。

> 2026-10-25 13:12 起补丁烤入 `-reason5` 镜像；13:20 起模板改为**直接替换模型目录
> 那一份**并去掉 `--chat-template`（§2.4）。容器 `docker inspect` 的 mounts 只剩
> 模型目录与 `~/.cache`，无 `m3_patches`；`/tokenize` 带 `reasoning_effort=low`
> 时 prompt 以 `<mm:think>` 结尾、末尾带 `<effort>low</effort>`（194 → 213 tokens），
> 而官方模板两者都没有——此时渲染路径里唯一的五档副本就是模型目录那份，由此
> 证明它已生效。四档实跑长度与挂载阶段完全一致（`temperature=0`：
> 默认 156/0、low 160/493、high 188/383、max 329/862，content/reasoning 字符数）。

---

## 5. 回滚

### 5.1 回到官方模板

```bash
cp /data/MiniMax-M3.1-Flash/chat_template.jinja.orig /data/MiniMax-M3.1-Flash/chat_template.jinja
cd /data && docker compose -p data -f docker-compose-m31.yml up -d --force-recreate
```

`--chat-template` 默认是注释状态，所以模型目录那份就是唯一来源；如果之前手动
启用了它，记得一并注释掉。

### 5.2 回到 v1 挂载版

```bash
# 1) compose 换回挂载版（backup 里有整份）
cp /data/docker-compose-m31.yml.bak-<ts> /data/docker-compose-m31.yml
# 2) 模板：v2 在模型目录，官方用 .orig
cp /data/m31_patches/chat_template.jinja.effort5 /data/MiniMax-M3.1-Flash/chat_template.jinja
# 3) parser 挂载依赖 /data/m3_patches（当前即 v2）
md5sum /data/m3_patches/minimax_m3_reasoning_parser.py   # 期望 14e6fc54…
cd /data && docker compose -p data -f docker-compose-m31.yml up -d --force-recreate
```

模板备份链：`.orig`（官方）→ `.pre-effort` → `.pre-maxfix` → `.pre-defaultoff`
→ `.effort-final`（v1，两态）→ `.effort5`（v2，五档）。当前权威副本：
模型目录 `/data/MiniMax-M3.1-Flash/chat_template.jinja`（md5 `f90d3923…`）；
同 md5 的副本还有 `/data/m31_patches/`、镜像 `/opt/m31/`（备用）与仓库
`patches/reasoning/`。
