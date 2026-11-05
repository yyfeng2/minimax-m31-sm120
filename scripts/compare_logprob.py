"""Paired per-position comparison of prompt logprobs across runs.

The prompt is fixed, so the logprob the model assigns to each prompt token is
computed in a single forward pass and is directly comparable between runs
(position i in run A vs position i in run B).

Usage: python3 compare_logprob.py tag1 tag2 [tag3 ...]
(files: /data/accuracy_<tag>.json or /data/moe_logprob_<tag>.json)
"""
import json
import os
import statistics as st
import sys


def load(tag):
    for path in (f"/data/accuracy_{tag}.json", f"/data/moe_logprob_{tag}.json"):
        if os.path.exists(path):
            d = json.load(open(path))
            lp = d.get("logprob") if "logprob" in d else d
            return path, lp["actual"], lp["top1"]
    raise SystemExit(f"no data for {tag}")


def main():
    tags = sys.argv[1:]
    data = {}
    for t in tags:
        p, a, t1 = load(t)
        data[t] = (a, t1)
        print(f"{t:26s} n={len(a):4d} mean_actual={st.mean(a):+.4f}  ({p})")

    base = tags[0]
    ba, bt = data[base]
    print(f"\npaired vs {base} (per-position actual logprob delta):")
    for t in tags[1:]:
        a, t1 = data[t]
        n = min(len(ba), len(a))
        d = [a[i] - ba[i] for i in range(n)]
        win = sum(1 for x in d if x > 1e-6)
        loss = sum(1 for x in d if x < -1e-6)
        same = n - win - loss
        print(f"  {t:24s} n={n:4d} mean_delta={st.mean(d):+.4f} "
              f"median_delta={st.median(d):+.4f} max|d|={max(abs(x) for x in d):.4f} | "
              f"better={win} worse={loss} same={same}")


if __name__ == "__main__":
    main()
