"""Next-token distribution probe for MoE A/B (W4A4 native vs MARLIN W4A16).

  python3 /data/moe_dist.py <tag>

For two contexts (short ~150 tok, long ~1.5k tok) it records the greedy next
token and the top-5 next-token distribution.  Output /data/moe_dist_<tag>.json
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"

SHORT = (
    "MiniMax-M3 uses a block-sparse attention with a 128-token page and a "
    "top-k indexer. The server stores the KV cache as NVFP4 (E2M1 payload "
    "plus E4M3 block scales) and runs decode with split-K online softmax. "
    "Explain, step by step, how a decode step computes attention over the "
    "selected pages, then summarize the memory-traffic advantages."
)

LONG = (
    "The following is a technical description of a paged KV cache in an LLM "
    "serving engine. A decode step reads the KV pages selected by the indexer, "
    "computes QK^T in FP4, applies online softmax with a running max and "
    "running denominator, and accumulates the PV product in FP8. Pages are "
    "128 tokens long and are referenced through a per-request block table. "
    "Split-K partitions the selected pages across chunks and merges the "
    "partial outputs with a second reduction pass. "
) * 8 + "In one sentence, what is the dominant cost of this decode step?"


def post(path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def probe(name, prompt, out):
    o = post(
        "/v1/completions",
        {
            "model": "MiniMax-M3.1-Flash",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5,
        },
    )
    ch = o["choices"][0]
    lp = ch.get("logprobs") or {}
    top = lp.get("top_logprobs") or []
    entry = top[0] if top else {}
    if isinstance(entry, list):  # normalize possible list-of-dicts
        entry = {d["token"]: d["logprob"] for d in entry}
    tokens = lp.get("tokens") or []
    out[name] = {
        "text": ch.get("text"),
        "prompt_tokens": (o.get("usage") or {}).get("prompt_tokens"),
        "token": tokens[0] if tokens else None,
        "token_logprob": (lp.get("token_logprobs") or [None])[0],
        "top5": sorted(entry.items(), key=lambda kv: -kv[1])[:5],
    }
    print(f"[{name}] tokens={out[name]['prompt_tokens']} greedy={out[name]['token']!r} "
          f"lp={out[name]['token_logprob']} top5={out[name]['top5'][:3]}", flush=True)


if __name__ == "__main__":
    tag = sys.argv[1]
    out = {}
    probe("short", SHORT, out)
    probe("long", LONG, out)
    json.dump(out, open(f"/data/moe_dist_{tag}.json", "w"), ensure_ascii=False)
    print("SAVED", tag, flush=True)
