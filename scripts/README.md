# 脚本

这些脚本最初写在 node2 的 `/data` 下，默认访问 `http://127.0.0.1:8000`，
在 node2 本机（或容器网络内）执行。

| 脚本 | 作用 |
|---|---|
| `prefill_bench.py` | 长 prompt TTFT / prefill 吞吐；每请求独立 PRNG token-id prompt（避免 APC 假数），并读 `/metrics` 交叉校验 |
| `verify_m31.py` | 校验模型 safetensors 分片结构与 index 一致性（不依赖 torch） |
| `check_m31.py` | 对比 ModelScope 远端文件尺寸，检查模型下载完整性 |
| `bench_m31.py` / `bench2.py` | decode 吞吐压测（solo / 多并发），用于镜像与 MoE 后端 A/B |
| `moe_probe.py` | MoE A/B：固定 prompt 的逐 token logprob + 3 组贪心生成（native / marlin） |
| `moe_logprob.py` | 232 个真实 next-token 的 logprob 统计（精度对比的原始数据来源） |
| `moe_dist.py` | 短/长上下文下的 next-token 分布对比（贪心 token 与 top-5） |
| `m3_think_test.py` | 五档思考冒烟（默认关闭 + low…max，打印 content/reasoning 长度与标签泄漏；只打印不断言单调） |
| `render_template_check.py` | 离线渲染 14 个档位用例（真实 tokenizer + 模板文件），打印指令行、`<effort>` 与预填标记 |
| `reasoning_consistency_check.py` | 模板与 parser 的一致性校验（同一批用例，断言两侧开关判定一致） |
| `gen_m31.py` | 简单生成请求（可传 prompt），用于人工抽查 |

MoE 相关脚本的 `<tag>` 参数（`native` / `marlin`）决定输出文件名，
结果已归档在 `../results/`。

## 压测口径（重要）

- 每条请求必须是**不同前缀**的输入；`prefill_bench.py` 用 PRNG token id 并以
  请求序号作 seed。
- 报数前必须与 `/metrics` 的 `prompt_tokens_total` 增量对上；对不上即作废。
- `temperature=0`，每个负载 3 次取中位。
