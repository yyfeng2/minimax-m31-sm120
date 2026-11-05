"""Precision probe v2 — separates the PREFILL path from the DECODE path.

Why two paths:
  * `prompt_logprobs` on a single-forward prompt exercises the *prefill*
    attention (Triton sparse attend, always the same in every SM120 config)
    plus MoE.  It is insensitive to the native decode kernel.
  * Greedy generation with `logprobs=1` exercises the *decode* path: every
    token after the first attends over the growing KV cache through the
    config-selected decode kernel (native FP4/FP8 vs Triton dequant).

Both are reported per-position so runs can be paired exactly.

Usage: python3 /data/accuracy_probe2.py <tag> [--key KEY] [--out DIR]
Output: <out>/accuracy2_<tag>.json
"""
import argparse
import json
import statistics as st
import urllib.request

LOGPROB_TEXT = (
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

# Greedy continuation: same prompt, N decode steps. Long enough that the
# decode path dominates (64 steps), short enough to stay fast.
GEN_PROMPT = (
    "Explain, step by step, how a decode step computes attention over the "
    "selected pages of a paged KV cache, then summarize the memory-traffic "
    "advantages of the design."
)

TASKS = [
    ("math_mul", "math", "Compute 173*29-451. Show the arithmetic and end with 'ANSWER: <number>'.",
     lambda t: "4566" in t.replace(",", "")),
    ("math_frac", "math", "A train travels 240 km in 3 hours, then 180 km in 2 hours. "
     "What is its average speed for the whole trip? End with 'ANSWER: <number> km/h'.",
     lambda t: "84" in t),
    ("math_seq", "math", "What is the 20th Fibonacci number? End with 'ANSWER: <number>'.",
     lambda t: "6765" in t),
    ("code_fn", "code", "Write a Python function split_kv(pages, k) that splits a list into k "
     "contiguous chunks, then show one example.", lambda t: "def split_kv" in t and ":" in t),
    ("code_sort", "code", "Write a Python one-liner that returns the second largest number in a list "
     "named nums. Put the code in a single ```python block.",
     lambda t: "sorted" in t or "heap" in t or "max" in t),
    ("zh_kv", "zh", "用三句话解释什么是分页 KV cache，并说明它为什么能节省显存。",
     lambda t: len(t) > 30 and ("KV" in t or "kv" in t)),
    ("zh_sci", "zh", "简述扩散模型的前向过程和反向过程，以及为什么采样步数会影响质量。",
     lambda t: len(t) > 40 and ("噪声" in t or "去噪" in t)),
    ("en_fact", "en", "Explain in a short paragraph why FP8 block scaling needs a per-block "
     "scale factor, and what happens without one.",
     lambda t: len(t) > 80 and ("scale" in t.lower())),
    ("instr_list", "instr", "List exactly three primary colors as a JSON array and nothing else.",
     lambda t: t.strip().startswith("[") and t.strip().rstrip(".").endswith("]")),
    ("instr_upper", "instr", "Reply with the word BANANA in uppercase letters only. No other text.",
     lambda t: t.strip() == "BANANA"),
]


def post(base, key, path, body, timeout=1800):
    req = urllib.request.Request(
        base + path, json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def prefill_logprob(base, key):
    """prompt_logprobs: single forward -> prefill attention path + MoE."""
    o = post(base, key, "/v1/completions", {
        "model": "MiniMax-M3.1-Flash", "prompt": LOGPROB_TEXT,
        "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1})
    entries = o["choices"][0].get("prompt_logprobs") or []
    actual, top1 = [], []
    for e in entries:
        if not e:
            continue
        vals = list(e.values())
        act = [v for v in vals if v.get("rank", 1) != 1]
        actual.append(float(act[0]["logprob"]) if act else float(vals[0]["logprob"]))
        top1.append(max(float(v["logprob"]) for v in vals))
    return {"prompt_tokens": (o.get("usage") or {}).get("prompt_tokens"),
            "n": len(actual),
            "mean_actual": st.mean(actual) if actual else float("nan"),
            "mean_top1": st.mean(top1) if top1 else float("nan"),
            "actual": actual, "top1": top1}


def decode_logprob(base, key, max_tokens=64):
    """Greedy generation with output logprobs -> decode attention path."""
    o = post(base, key, "/v1/completions", {
        "model": "MiniMax-M3.1-Flash", "prompt": GEN_PROMPT,
        "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 1})
    lp = o["choices"][0].get("logprobs") or {}
    toks = lp.get("tokens") or []
    tlp = lp.get("token_logprobs") or []
    text = o["choices"][0].get("text") or ""
    return {"n": len(toks), "tokens": toks, "token_logprobs": tlp, "text": text,
            "mean_token_logprob": (st.mean([x for x in tlp if x is not None])
                                   if any(x is not None for x in tlp) else float("nan"))}


def run_tasks(base, key, max_tokens=800):
    out = {}
    for tid, kind, prompt, checker in TASKS:
        o = post(base, key, "/v1/chat/completions", {
            "model": "MiniMax-M3.1-Flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": max_tokens})
        msg = o["choices"][0]["message"]
        text = msg.get("content") or ""
        rsn = msg.get("reasoning") or msg.get("reasoning_content") or ""
        u = o.get("usage") or {}
        ok = bool(checker(text))
        out[tid] = {"kind": kind, "pass": ok, "len": len(text),
                    "completion_tokens": u.get("completion_tokens"),
                    "reasoning_len": len(rsn), "text": text}
        print(f"  {tid:12s} {'PASS' if ok else 'FAIL'}  len={len(text):5d} "
              f"ct={u.get('completion_tokens')}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--key", default="REPLACE_WITH_YOUR_API_KEY")
    ap.add_argument("--out", default="/data")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"

    pl = prefill_logprob(base, a.key)
    print(f"[{a.tag}] PREFILL logprob n={pl['n']} mean_actual={pl['mean_actual']:.4f} "
          f"mean_top1={pl['mean_top1']:.4f}", flush=True)

    dl = decode_logprob(base, a.key)
    print(f"[{a.tag}] DECODE  logprob n={dl['n']} "
          f"mean_token_logprob={dl['mean_token_logprob']:.4f}", flush=True)

    print(f"[{a.tag}] tasks:", flush=True)
    tk = run_tasks(base, a.key)
    npass = sum(1 for v in tk.values() if v["pass"])
    print(f"[{a.tag}] TASK SCORE {npass}/{len(tk)}", flush=True)

    res = {"tag": a.tag, "prefill_logprob": pl, "decode_logprob": dl,
           "tasks": tk, "npass": npass, "ntask": len(tk)}
    path = f"{a.out}/accuracy2_{a.tag}.json"
    json.dump(res, open(path, "w"), ensure_ascii=False)
    print("SAVED", path, flush=True)


if __name__ == "__main__":
    main()
