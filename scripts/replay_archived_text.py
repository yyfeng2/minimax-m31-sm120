"""Decisive check: replay the ARCHIVED 10-24 prompt text on today's server.

If today's logprob for the archived text equals the archived value (-3.8878
for native / -3.9440 for marlin), the server config did not change the
numerics and the earlier delta was a prompt-text mismatch.  If it differs,
the config change is responsible.

Usage: python3 /data/replay_archived_text.py
"""
import ast
import json
import statistics as st
import urllib.request

BASE = "http://127.0.0.1:8000"
KEY = "REPLACE_WITH_YOUR_API_KEY"


def get_text(path, varname):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == varname:
                    return ast.literal_eval(node.value)
    raise SystemExit("not found")


def logprob(text):
    req = urllib.request.Request(
        BASE + "/v1/completions",
        json.dumps({"model": "MiniMax-M3.1-Flash", "prompt": text,
                    "max_tokens": 1, "temperature": 0.0,
                    "prompt_logprobs": 1}).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    o = json.load(urllib.request.urlopen(req, timeout=900))
    actual, top1 = [], []
    for e in o["choices"][0].get("prompt_logprobs") or []:
        if not e:
            continue
        vals = list(e.values())
        act = [v for v in vals if v.get("rank", 1) != 1]
        actual.append(float(act[0]["logprob"]) if act else float(vals[0]["logprob"]))
        top1.append(max(float(v["logprob"]) for v in vals))
    return actual, top1, (o.get("usage") or {}).get("prompt_tokens")


def main():
    arch_text = get_text("/data/moe_logprob.py", "TEXT")
    mine_text = get_text("/data/accuracy_probe2.py", "LOGPROB_TEXT")
    print(f"archived text len={len(arch_text)}   mine len={len(mine_text)}   "
          f"IDENTICAL={arch_text == mine_text}", flush=True)

    a, t1, ntok = logprob(arch_text)
    print(f"ARCHIVED TEXT today: n={len(a)} prompt_tokens={ntok} "
          f"mean_actual={st.mean(a):+.4f} mean_top1={st.mean(t1):+.4f}", flush=True)

    for tag in ("native", "marlin"):
        p = f"/data/moe_logprob_{tag}.json"
        try:
            d = json.load(open(p))
        except Exception:
            continue
        old = d["actual"]
        n = min(len(old), len(a))
        delta = [a[i] - old[i] for i in range(n)]
        print(f"vs archived {tag} (mean {st.mean(old):+.4f}): "
              f"mean_delta={st.mean(delta):+.4f} max|d|={max(abs(x) for x in delta):.4f} "
              f"identical={sum(1 for x in delta if x==0)}/{n}", flush=True)

    json.dump({"archived_text": arch_text, "actual": a, "top1": t1,
               "prompt_tokens": ntok},
              open("/data/replay_archived_text.json", "w"), ensure_ascii=False)
    print("SAVED /data/replay_archived_text.json")


if __name__ == "__main__":
    main()
