"""Is the prompt_logprob metric itself batch/layout sensitive?

A mean delta of 0.03 nats with individual positions moving 6.8 nats is NOT
the signature of a small numeric perturbation (that would move every position
by ~the same tiny amount).  The signature of *chaotic amplification* is:
   small float differences in one forward pass -> a routing decision flips
   -> one expert set changes -> that token's logprob moves a lot.

This script tests whether the metric moves when only the BATCH CONTEXT
changes (same prompt, same server, same config):
   * run alone
   * run while N concurrent unrelated requests are in flight
   * run after a longer prompt has occupied the KV pool
If the metric moves, then any cross-container logprob delta is dominated by
batch/layout effects and cannot be used as a precision metric.

Usage: python3 /data/batch_sensitivity.py [--key KEY]
"""
import argparse
import json
import statistics as st
import threading
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
NOISE = "请简要说明什么是张量并行。 " * 400


def post(key, path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path, json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def logprob(key):
    o = post(key, "/v1/completions", {
        "model": "MiniMax-M3.1-Flash", "prompt": TEXT, "max_tokens": 1,
        "temperature": 0.0, "prompt_logprobs": 1})
    actual = []
    for e in o["choices"][0].get("prompt_logprobs") or []:
        if not e:
            continue
        vals = list(e.values())
        act = [v for v in vals if v.get("rank", 1) != 1]
        actual.append(float(act[0]["logprob"]) if act else float(vals[0]["logprob"]))
    return actual


def fire_noise(key, n, stop):
    def one():
        try:
            post(key, "/v1/chat/completions", {
                "model": "MiniMax-M3.1-Flash",
                "messages": [{"role": "user", "content": NOISE}],
                "temperature": 0.0, "max_tokens": 256})
        except Exception:
            pass
        stop.set()
    ts = [threading.Thread(target=one) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="REPLACE_WITH_YOUR_API_KEY")
    a = ap.parse_args()

    solo = logprob(a.key)
    print(f"solo          n={len(solo)} mean={st.mean(solo):+.4f}", flush=True)

    # same prompt, but with 6 concurrent unrelated requests in flight
    stop = threading.Event()
    th = threading.Thread(target=fire_noise, args=(a.key, 6, stop))
    th.start()
    under_load = logprob(a.key)
    stop.set()
    th.join()
    print(f"under_load    n={len(under_load)} mean={st.mean(under_load):+.4f}", flush=True)

    n = min(len(solo), len(under_load))
    d = [under_load[i] - solo[i] for i in range(n)]
    print(f"delta solo->load: mean={st.mean(d):+.4f} max|d|={max(abs(x) for x in d):.4f} "
          f"identical={sum(1 for x in d if x==0)}/{n}", flush=True)

    json.dump({"solo": solo, "under_load": under_load},
              open("/data/batch_sensitivity.json", "w"))
    print("SAVED /data/batch_sensitivity.json")


if __name__ == "__main__":
    main()
