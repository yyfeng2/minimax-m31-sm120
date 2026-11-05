"""思考档位冒烟：逐档请求，打印 content / reasoning 切分与标记泄漏。

默认（不传 effort）应为关闭；low/medium/high/xhigh/max 应产生 reasoning。
只打印，不做长度断言——单样本下档位长度不保证单调（见 docs/REASONING.md §4.2）。

Usage: python3 m3_think_test.py [base_url]
       KEY=... python3 m3_think_test.py http://192.168.20.12:8000
"""
import json
import os
import sys
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
KEY = os.environ.get("KEY", "REPLACE_WITH_YOUR_API_KEY")
URL = BASE + "/v1/chat/completions"


def call(payload):
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


base = dict(model="MiniMax-M3.1-Flash", max_tokens=1024, temperature=0)
MSG = [{"role": "user", "content": "12月有31天吗？请先说理由再回答"}]
CASES = [
    ("absent", None),
    ("none", {"reasoning_effort": "none"}),
    ("low", {"reasoning_effort": "low"}),
    ("medium", {"reasoning_effort": "medium"}),
    ("high", {"reasoning_effort": "high"}),
    ("xhigh", {"reasoning_effort": "xhigh"}),
    ("max", {"reasoning_effort": "max"}),
    ("tm_disabled+max", {"thinking_mode": "disabled", "reasoning_effort": "max"}),
]

for tag, kw in CASES:
    p = dict(base, messages=MSG)
    if kw is not None:
        p["chat_template_kwargs"] = kw
    r = call(p)
    m = r["choices"][0]["message"]
    t = m.get("content") or ""
    rsn = m.get("reasoning") or m.get("reasoning_content") or ""
    print(f"=== {tag:15s} content={len(t):4d} reasoning={len(rsn):4d} "
          f"leak={'mm:think' in t}")
    print("    content  :", repr(t)[:160])
    print("    reasoning:", repr(rsn)[:120])
