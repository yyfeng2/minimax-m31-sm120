#!/usr/bin/env python3
"""Prefill/TTFT probe with exact input-token control (nvfp4 SM120 build).

Each request carries an explicit token-id prompt (built from a PRNG stream) so
(a) the input length is exact, and (b) no two requests share a prefix, i.e. the
APC cache cannot shorten the prefill being measured.  max_tokens=1 keeps decode
out of the number, so wall ~= queue + prefill + one decode step.
"""
import argparse
import concurrent.futures as cf
import json
import random
import statistics
import time
import urllib.request

API = "http://127.0.0.1:8000"
MODEL = "MiniMax-M3.1-Flash"
VOCAB = 200_000


def post(path: str, payload: dict, timeout: int = 1800):
    req = urllib.request.Request(
        API + path, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def token_ids(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, VOCAB) for _ in range(n)]


def ask(n_in: int, seed: int):
    body = {
        "model": MODEL,
        "temperature": 0.0,
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,
        "prompt": token_ids(n_in, seed),
    }
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    with post("/v1/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            if ttft is None and obj.get("choices"):
                if (obj["choices"][0].get("text") or "") != "":
                    ttft = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    return {
        "ttft": ttft if ttft is not None else wall,
        "wall": wall,
        "pt": usage.get("prompt_tokens", 0),
    }


def engine_counter(name: str) -> float:
    txt = urllib.request.urlopen(API + "/metrics", timeout=60).read().decode()
    for line in txt.splitlines():
        if line.startswith(f"vllm:{name}{{"):
            return float(line.rsplit(" ", 1)[1])
    return float("nan")


def run(n_in, c, reps, rows, tag):
    per_rep = []
    for rep in range(reps):
        seq = list(range(c))

        # The seed MUST vary with rep: otherwise rep1 replays rep0's token stream
        # verbatim and APC serves it from cache (measured: 32k in 0.13s).
        def job(i, rep=rep):
            base = int(time.time() * 1000) ^ (n_in * 1_000_003)
            return ask(n_in, base + rep * 100_003 + i * 7919)

        with cf.ThreadPoolExecutor(max_workers=c) as ex:
            rs = list(ex.map(job, seq))
        ttfts = sorted(r["ttft"] for r in rs)
        pt = rs[0]["pt"]
        tps = pt * c / max(r["wall"] for r in rs)
        per_rep.append((ttfts, tps, pt))
        print(
            f"  rep{rep} in={pt} c={c} TTFT med={statistics.median(ttfts):.3f}s "
            f"max={ttfts[-1]:.3f}s prefill={tps:.0f} tok/s",
            flush=True,
        )
    rows.append(
        {
            "n_in": n_in,
            "c": c,
            "prompt_tokens": per_rep[0][2],
            "ttft_med": statistics.median(r[0][len(r[0]) // 2] for r in per_rep),
            "ttft_max": max(r[0][-1] for r in per_rep),
            "prefill_tps": statistics.median(r[1] for r in per_rep),
        }
    )
    print(f"[{tag}] engine prompt_tokens_total now={engine_counter('prompt_tokens_total'):.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cases", default="4096:1,16384:1,65536:1,131072:1,32768:4,32768:8")
    a = ap.parse_args()

    ask(512, 1)  # warm
    base = engine_counter("prompt_tokens_total")
    rows = []
    for case in a.cases.split(","):
        n, c = case.split(":")
        print(f"[{a.tag}] case {n} x{c}", flush=True)
        run(int(n), int(c), a.reps, rows, a.tag)

    json.dump(
        {"base_prompt_tokens_total": base, "rows": rows},
        open(f"/data/prefill_{a.tag}.json", "w"),
        indent=1,
    )
    print(f"[{a.tag}] engine-prompt-tokens delta={engine_counter('prompt_tokens_total')-base:.0f}")
    for r in rows:
        print(
            f"[{a.tag}] in={r['prompt_tokens']:>7} c={r['c']:>2} "
            f"TTFT_med={r['ttft_med']:7.3f}s TTFT_max={r['ttft_max']:7.3f}s "
            f"prefill={r['prefill_tps']:8.0f} tok/s"
        )


if __name__ == "__main__":
    main()
