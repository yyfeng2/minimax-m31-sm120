#!/usr/bin/env python3
"""校验 safetensors 分片结构，并与 model.safetensors.index.json 交叉比对（无需 torch）。"""
import json, os, struct, sys

L = "/data/MiniMax-M3.1-Flash"
idx = json.load(open(os.path.join(L, "model.safetensors.index.json")))
wmap = idx["weight_map"]

per_shard = {}
for t, shard in wmap.items():
    per_shard.setdefault(shard, set()).add(t)

bad, have = [], set()
for shard in sorted(per_shard):
    p = os.path.join(L, shard)
    try:
        fsize = os.path.getsize(p)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > fsize:
                bad.append("%s: 头部长度非法 %d" % (shard, n))
                continue
            head = json.loads(f.read(n))
        names = {k for k in head if k != "__metadata__"}
        end = max((v["data_offsets"][1] for v in head.values()
                   if v.get("data_offsets")), default=0)
        if fsize < 8 + n + end:
            bad.append("%s: 文件被截断 size=%d 需要>=%d" % (shard, fsize, 8 + n + end))
        if names != per_shard[shard]:
            bad.append("%s: 权重集合与 index 不符 实有=%d index=%d"
                       % (shard, len(names), len(per_shard[shard])))
        have |= names
    except Exception as e:
        bad.append("%s: %s: %s" % (shard, type(e).__name__, e))

print("分片数=%d  index 声明权重=%d  实际匹配=%d" % (len(per_shard), len(wmap), len(have & set(wmap))))
print("权重结构校验通过" if (have == set(wmap) and not bad) else "存在问题:")
for b in bad[:10]:
    print("  !", b)

for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
          "generation_config.json", "special_tokens_map.json", "added_tokens.json",
          "preprocessor_config.json", "chat_template.jinja"):
    fp = os.path.join(L, f)
    if not os.path.exists(fp):
        print("  缺失文件 ", f); bad.append(f); continue
    if f.endswith(".jinja"):
        print("  存在       ", f); continue
    try:
        json.load(open(fp)); print("  JSON 合法  ", f)
    except Exception as e:
        print("  JSON 损坏  ", f, e); bad.append(f)

print("RESULT:", "ALL OK" if not bad else "FAILED (%d)" % len(bad))
sys.exit(1 if bad else 0)
