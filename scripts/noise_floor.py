"""Measure the RUN-TO-RUN noise floor of the logprob probe.

Same server, same config, same prompt, repeated N times. The spread of these
runs is the noise floor: any A/B delta smaller than it is not evidence of a
precision difference.

Usage: python3 /data/noise_floor.py [N] [--out /data]
"""
import argparse
import json
import statistics as st
import urllib.request

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

BASE = "http://127.0.0.1:8000"
KEY = "REPLACE_WITH_YOUR_API_KEY"


def one():
    req = urllib.request.Request(
        BASE + "/v1/completions",
        json.dumps({"model": "MiniMax-M3.1-Flash", "prompt": TEXT,
                    "max_tokens": 1, "temperature": 0.0,
                    "prompt_logprobs": 1}).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    o = json.load(urllib.request.urlopen(req, timeout=900))
    actual = []
    for e in o["choices"][0].get("prompt_logprobs") or []:
        if not e:
            continue
        vals = list(e.values())
        act = [v for v in vals if v.get("rank", 1) != 1]
        actual.append(float(act[0]["logprob"]) if act else float(vals[0]["logprob"]))
    return actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=5)
    ap.add_argument("--out", default="/data")
    a = ap.parse_args()
    runs = []
    for i in range(a.n):
        v = one()
        runs.append(v)
        print(f"run {i+1}: n={len(v)} mean_actual={st.mean(v):+.4f}", flush=True)
    means = [st.mean(v) for v in runs]
    print(f"\nmean of means = {st.mean(means):+.4f}   spread = "
          f"{max(means)-min(means):.4f}   stdev = "
          f"{st.stdev(means) if len(means)>1 else 0:.4f}")
    base = runs[0]
    for i, v in enumerate(runs[1:], 2):
        n = min(len(base), len(v))
        d = [v[j] - base[j] for j in range(n)]
        print(f"paired run{i} vs run1: mean_delta={st.mean(d):+.4f} "
              f"max|d|={max(abs(x) for x in d):.4f} "
              f"better={sum(1 for x in d if x>1e-6)} "
              f"worse={sum(1 for x in d if x<-1e-6)}")
    json.dump({"means": means, "runs": runs},
              open(f"{a.out}/noise_floor.json", "w"))
    print("SAVED", f"{a.out}/noise_floor.json")


if __name__ == "__main__":
    main()
