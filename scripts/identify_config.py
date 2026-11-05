"""Which archived config does TODAY's prod actually correspond to?

moe_probe.py used a *different* prompt (PROMPT) than moe_logprob.py (TEXT).
Its archived outputs contain the per-token logprob list for that prompt, so
replaying that exact prompt today tells us, element by element, whether today's
prod is bit-identical to the archived 'native' run or the archived 'marlin' run.

Usage: python3 /data/identify_config.py
"""
import ast
import json
import statistics as st
import urllib.request

BASE = "http://127.0.0.1:8000"
KEY = "REPLACE_WITH_YOUR_API_KEY"


def get_prompt(path, varname):
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
    actual = []
    for e in o["choices"][0].get("prompt_logprobs") or []:
        if not e:
            continue
        vals = list(e.values())
        act = [v for v in vals if v.get("rank", 1) != 1]
        actual.append(float(act[0]["logprob"]) if act else float(vals[0]["logprob"]))
    return actual


def main():
    prompt = get_prompt("/data/moe_probe.py", "PROMPT")
    today = logprob(prompt)
    print(f"prompt_tokens={len(today)} mean_today={st.mean(today):+.4f}", flush=True)
    for tag in ("native", "marlin"):
        p = f"/data/moe_probe_{tag}.json"
        try:
            d = json.load(open(p))
        except Exception as e:
            print(tag, "missing", e)
            continue
        old = d[0] if isinstance(d, list) else d.get("logprobs")
        if old is None:
            # the archived file stores a dict of task->text plus a 'logprobs' key
            for k, v in d.items():
                if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                    old = v
                    break
        if not old:
            print(tag, "no logprob list; keys=", list(d)[:8])
            continue
        n = min(len(old), len(today))
        delta = [today[i] - old[i] for i in range(n)]
        same = sum(1 for x in delta if x == 0)
        print(f"vs archived {tag:7s} (mean {st.mean(old):+.4f}): "
              f"mean_delta={st.mean(delta):+.4f} max|d|={max(abs(x) for x in delta):.4f} "
              f"bit-identical={same}/{n}", flush=True)
    json.dump({"today": today}, open("/data/identify_config.json", "w"))
    print("SAVED /data/identify_config.json")


if __name__ == "__main__":
    main()
