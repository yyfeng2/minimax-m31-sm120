"""Paired comparison of task-accuracy runs across configs.

Key design point: temperature=0 makes each config deterministic, but a config
change still flips discrete decisions (MoE routing, top-k page selection), so
individual items can flip for reasons unrelated to "quality".  A single
config change that flips k items gives an exact McNemar-style reading:
the informative quantity is the *imbalance* between items that go
fail->pass and pass->fail, not the raw score difference.

Usage: python3 /data/taskacc_compare.py <tag_base> <tag_1> [<tag_2> ...]
       [--id-key id rep]
"""
import argparse
import json
import os
import sys


def load(tag, out="/data"):
    for p in (f"{out}/taskacc_{tag}.json", f"/data/taskacc_{tag}.json"):
        if os.path.exists(p):
            return json.load(open(p))
    raise SystemExit(f"no data for {tag}")


def key_of(rec):
    return (rec["id"], rec["rep"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", default="/data")
    a = ap.parse_args()

    data = {}
    for t in a.tags:
        d = load(t, a.out)
        data[t] = {key_of(r): r for r in d["results"]}
        cats = d["cat_summary"]
        line = f"{t:22s} strict={d['strict']}/{d['nitems']} loose={d['loose']}/{d['nitems']} "
        line += "  " + " ".join(f"{c}:{p}/{tt}" for c, (p, tt) in sorted(cats.items()))
        print(line)

    base = a.tags[0]
    print(f"\nMcNemar-style paired flips vs {base}:")
    for t in a.tags[1:]:
        keys = sorted(set(data[base]) & set(data[t]))
        b2g = [k for k in keys if not data[base][k]["pass"] and data[t][k]["pass"]]
        g2b = [k for k in keys if data[base][k]["pass"] and not data[t][k]["pass"]]
        same = len(keys) - len(b2g) - len(g2b)
        n_disc = len(b2g) + len(g2b)
        print(f"  {t:22s} common={len(keys)} unchanged={same} "
              f"gain={len(b2g)} loss={len(g2b)} net={len(b2g)-len(g2b):+d}")
        if b2g:
            print("      gained:", " ".join(sorted(k[0] for k in b2g)))
        if g2b:
            print("      lost  :", " ".join(sorted(k[0] for k in g2b)))
        # exact binomial two-sided p for the imbalance (McNemar exact)
        if n_disc:
            from math import comb
            k = min(len(b2g), len(g2b))
            p2 = sum(comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc) * 2
            p2 = min(1.0, p2)
            print(f"      McNemar exact two-sided p = {p2:.4f} "
                  f"({'significant' if p2 < 0.05 else 'NOT significant'} at 0.05)")
        else:
            print("      no discordant pairs -> configs agree on every item")


if __name__ == "__main__":
    main()
