"""Check whether the failed math task is a *thinking-policy* artifact or a
numeric-precision artifact: re-run it with reasoning_effort=high.

Usage: python3 /data/reasoning_policy_check.py [--key KEY]
"""
import argparse
import json
import urllib.request

BASE = "http://127.0.0.1:8000"
TASKS = [
    ("math_frac", "A train travels 240 km in 3 hours, then 180 km in 2 hours. "
     "What is its average speed for the whole trip? End with 'ANSWER: <number> km/h'.",
     lambda t: "70" in t),
    ("math_mul", "Compute 173*29-451. Show the arithmetic and end with 'ANSWER: <number>'.",
     lambda t: "4566" in t.replace(",", "")),
]


def post(key, path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path, json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run(key, tid, prompt, checker, effort):
    kw = {}
    if effort:
        kw["chat_template_kwargs"] = {"reasoning_effort": effort}
    o = post(key, "/v1/chat/completions", {
        "model": "MiniMax-M3.1-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 3000, **kw})
    msg = o["choices"][0]["message"]
    text = msg.get("content") or ""
    rsn = msg.get("reasoning") or msg.get("reasoning_content") or ""
    u = o.get("usage") or {}
    ok = bool(checker(text))
    print(f"  {tid:10s} effort={effort or 'none':5s} {'PASS' if ok else 'FAIL'} "
          f"len={len(text):5d} rsn={len(rsn):5d} ct={u.get('completion_tokens')}", flush=True)
    if not ok:
        print("     content tail:", text[-160:].replace("\n", " "), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="REPLACE_WITH_YOUR_API_KEY")
    a = ap.parse_args()
    out = {}
    for effort in (None, "high"):
        for tid, prompt, checker in TASKS:
            out[f"{tid}@{effort or 'none'}"] = run(a.key, tid, prompt, checker, effort)
    json.dump(out, open("/data/reasoning_policy_check.json", "w"))
    print("SAVED /data/reasoning_policy_check.json")


if __name__ == "__main__":
    main()
