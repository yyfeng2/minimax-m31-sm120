"""A/B probe for the SM120 native W4A4 NVFP4 MoE vs the MARLIN W4A16 path.

Runs on node2 next to the served container:
  python3 /data/moe_probe.py native|marlin

Collects per-token prompt logprobs for a fixed technical prompt plus three
greedy generations (math / code / Chinese).  Output: /data/moe_probe_<tag>.json
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"

PROMPT = (
    "MiniMax-M3 uses a block-sparse attention with a 128-token page and a "
    "top-k indexer. The server stores the KV cache as NVFP4 (E2M1 payload "
    "plus E4M3 block scales) and runs decode with split-K online softmax. "
    "Explain, step by step, how a decode step computes attention over the "
    "selected pages, then summarize the memory-traffic advantages."
)


def post(path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def probe_logprobs(tag, out):
    o = post(
        "/v1/completions",
        {
            "model": "MiniMax-M3.1-Flash",
            "prompt": PROMPT,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 1,
        },
    )
    entries = o["choices"][0].get("prompt_logprobs") or []
    vals = []
    for e in entries:
        if not e:
            continue
        it = e.values() if isinstance(e, dict) else e
        for v in it:
            if isinstance(v, dict) and "logprob" in v:
                vals.append(float(v["logprob"]))
    mean = sum(vals) / len(vals) if vals else float("nan")
    print(f"[{tag}] n={len(vals)} mean_logprob={mean:.6f}", flush=True)
    out[tag] = vals


def gen(tag, out, prompt, n=700):
    o = post(
        "/v1/chat/completions",
        {
            "model": "MiniMax-M3.1-Flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": n,
        },
    )
    txt = o["choices"][0]["message"]["content"]
    u = o.get("usage") or {}
    print(f"[{tag}] completion_tokens={u.get('completion_tokens')} text_len={len(txt)}", flush=True)
    out[tag] = txt


if __name__ == "__main__":
    tag = sys.argv[1]
    out = {}
    probe_logprobs(tag, out)
    gen("math", out, "Compute 173*29-451. Show the arithmetic and end with 'ANSWER: <number>'.")
    gen("code", out, "Write a Python function split_kv(pages, k) that splits a list into k contiguous chunks, then show one example.")
    gen("zh", out, "用三句话解释什么是分页 KV cache，并说明它为什么能节省显存。")
    json.dump(out, open(f"/data/moe_probe_{tag}.json", "w"), ensure_ascii=False)
    print("SAVED", tag, flush=True)
