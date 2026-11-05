"""Prompt-token logprob A/B probe (actual next-token logprob, not top-1).

  python3 /data/moe_logprob.py <tag>

Scores a fixed ~600-token mixed EN/ZH continuation through /v1/completions
with prompt_logprobs=1 and extracts, per position, the logprob assigned to the
prompt's own token (rank != 1) and the top-1 logprob.
"""
import json
import sys
import statistics as st
import urllib.request

BASE = "http://127.0.0.1:8000"

TEXT = (
    "A paged KV cache stores the key and value tensors of every attention "
    "layer in fixed-size pages of 128 tokens. Each request owns a block table "
    "that maps logical page indices to physical pages in a shared pool, so "
    "memory is allocated on demand and pages can be shared between requests "
    "with identical prefixes. During decode the engine selects a small set of "
    "pages with a lightweight indexer, loads them into shared memory, and runs "
    "QK^T with FP4 tensor cores followed by an online softmax and a PV "
    "product in FP8. Split-K partitions the selected pages across several "
    "thread blocks; each block keeps a running maximum, a running denominator "
    "and a partial output, and a small merge kernel combines the partials "
    "with weights exp(m_i - M). 分页 KV cache 把每条序列的键值张量按固定大小"
    "切页，用块表把逻辑页映射到共享显存池中的物理页，从而避免为最大长度预"
    "留连续空间。解码时索引器先挑出少量关键页，再在张量核上完成 QK 乘积、"
    "在线 softmax 和 PV 累加；分块并行只改变归约顺序，不改变数学结果。"
)


def post(path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main(tag):
    o = post(
        "/v1/completions",
        {
            "model": "MiniMax-M3.1-Flash",
            "prompt": TEXT,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 1,
        },
    )
    entries = o["choices"][0]["prompt_logprobs"] or []
    actual, top1 = [], []
    for e in entries:
        if not e:
            continue
        vals = list(e.values())
        act = [v for v in vals if v.get("rank", 1) != 1]
        if act:
            actual.append(float(act[0]["logprob"]))
        else:
            actual.append(float(vals[0]["logprob"]))
        top1.append(max(float(v["logprob"]) for v in vals))
    out = {
        "prompt_tokens": (o.get("usage") or {}).get("prompt_tokens"),
        "n": len(actual),
        "actual": actual,
        "top1": top1,
    }
    json.dump(out, open(f"/data/moe_logprob_{tag}.json", "w"))
    print(f"[{tag}] prompt_tokens={out['prompt_tokens']} n={len(actual)} "
          f"mean_actual={st.mean(actual):.4f} mean_top1={st.mean(top1):.4f}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
