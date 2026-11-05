"""Render the M3 chat template for every reasoning_effort level (offline).

Read-only: uses the tokenizer directly, does NOT touch the running server.
Usage: python3 /data/render_template_check.py [template_path]

Note (2026-10-25): transformers 5.x no longer exposes `chat_template_kwargs=`;
the template variables must be passed as plain **kwargs.  The first version of
this harness passed `chat_template_kwargs=kw`, so the template received a
variable literally named `chat_template_kwargs` and every case rendered as the
"absent" default -- that was a harness bug, not a template bug.
"""
import sys

from transformers import AutoTokenizer

MODEL = "/data/MiniMax-M3.1-Flash"
MSGS = [{"role": "user", "content": "1+1=?"}]
CASES = [
    ("absent", {}),
    ("none", {"reasoning_effort": "none"}),
    ("minimal", {"reasoning_effort": "minimal"}),
    ("low", {"reasoning_effort": "low"}),
    ("medium", {"reasoning_effort": "medium"}),
    ("high", {"reasoning_effort": "high"}),
    ("xhigh", {"reasoning_effort": "xhigh"}),
    ("max", {"reasoning_effort": "max"}),
    ("as_effort_low", {"effort": "low"}),
    ("unknown_eff", {"reasoning_effort": "ultra"}),
    ("tm_disabled", {"thinking_mode": "disabled"}),
    ("tm_disabled_max", {"thinking_mode": "disabled", "reasoning_effort": "max"}),
    ("tm_enabled", {"thinking_mode": "enabled"}),
    ("tm_enabled_low", {"thinking_mode": "enabled", "reasoning_effort": "low"}),
]


def main():
    tmpl = sys.argv[1] if len(sys.argv) > 1 else None
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tmpl:
        tok.chat_template = open(tmpl, encoding="utf-8").read()
    print(f"template: {tmpl or '(model default)'}  md5={__import__('hashlib').md5(tok.chat_template.encode()).hexdigest()}")
    seen = {}
    for name, kw in CASES:
        s = tok.apply_chat_template(
            MSGS, add_generation_prompt=True, tokenize=False, **kw)
        head = s.split("<model_id>")[0]
        inst = "(none)"
        for line in head.splitlines():
            if line.startswith("Current thinking mode:"):
                inst = line.strip()
        effort_tag = ""
        if "<effort>" in s:
            effort_tag = "<effort>" + s.split("<effort>")[1].split("</effort>")[0] + "</effort>"
        tail = s[-30:]
        prefix = "THINK_OPEN" if tail.rstrip().endswith("<mm:think>") else (
                 "THINK_CLOSE" if tail.rstrip().endswith("</mm:think>") else "NO_PREFIX")
        key = (inst, effort_tag, prefix)
        seen.setdefault(key, []).append(name)
        print(f"--- {name:13s} kw={kw}")
        print(f"    instruction: {inst[:200]}")
        print(f"    effort_tag : {effort_tag or '(absent)'}")
        print(f"    gen_prefix : {prefix}")
    print()
    print(f"distinct (instruction, effort_tag, prefix) combos: {len(seen)} / {len(CASES)} cases")
    for key, names in seen.items():
        print(f"  {len(names)}x {names}")
    print("RENDER CHECK DONE")


if __name__ == "__main__":
    main()
