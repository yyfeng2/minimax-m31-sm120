"""Replay the archived 10-24 probe prompts against the live server and diff
the generated text against the archived native/marlin outputs.

If today's text is byte-identical to the archived native text, then the
config changed but the numerics did not -> the logprob delta comes from the
container/config, not from a precision regression.

Usage: python3 /data/replay_text.py
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"
KEY = "REPLACE_WITH_YOUR_API_KEY"

PROMPTS = {
    "math": ("Compute 173*29-451. Show the arithmetic and end with 'ANSWER: <number>'."),
    "code": ("Write a Python function split_kv(pages, k) that splits a list into k "
             "contiguous chunks, then show one example."),
    "zh": "用三句话解释什么是分页 KV cache，并说明它为什么能节省显存。",
}


def post(path, body):
    req = urllib.request.Request(
        BASE + path, json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    return json.load(urllib.request.urlopen(req, timeout=900))


def main():
    arch = {}
    for tag in ("native", "marlin"):
        try:
            arch[tag] = json.load(open(f"/data/moe_probe_{tag}.json"))
        except Exception:
            pass
    out = {}
    for k, p in PROMPTS.items():
        o = post("/v1/chat/completions", {
            "model": "MiniMax-M3.1-Flash",
            "messages": [{"role": "user", "content": p}],
            "temperature": 0.0, "max_tokens": 700})
        txt = o["choices"][0]["message"].get("content") or ""
        out[k] = txt
        line = f"{k:6s} today_len={len(txt):5d}"
        for tag in arch:
            a = arch[tag].get(k)
            if a is not None:
                line += f"  {tag}: len={len(a):5d} identical={a == txt}"
        print(line, flush=True)
    json.dump(out, open("/data/replay_text_today.json", "w"), ensure_ascii=False)
    print("SAVED /data/replay_text_today.json")


if __name__ == "__main__":
    main()
