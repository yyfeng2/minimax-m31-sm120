"""Template/parser consistency check for the M3 5-level reasoning_effort patch.

Renders the chat template and instantiates the reasoning parser for the same
request kwargs, then asserts that both agree on "thinking on / off" and that
the rendered prompt carries the expected <effort> tag.

Usage (inside the vLLM container):
    python3 reasoning_consistency_check.py <template.jinja> <parser.py>
"""
import importlib.util
import sys

from transformers import AutoTokenizer

MODEL = "/data/MiniMax-M3.1-Flash"
MSGS = [{"role": "user", "content": "1+1=?"}]

# name, kwargs, expected_on, expected_effort_tag
CASES = [
    ("absent", {}, False, None),
    ("none", {"reasoning_effort": "none"}, False, None),
    ("minimal", {"reasoning_effort": "minimal"}, False, None),
    ("low", {"reasoning_effort": "low"}, True, "low"),
    ("medium", {"reasoning_effort": "medium"}, True, "medium"),
    ("high", {"reasoning_effort": "high"}, True, "high"),
    ("xhigh", {"reasoning_effort": "xhigh"}, True, "xhigh"),
    ("max", {"reasoning_effort": "max"}, True, "max"),
    ("alias_effort_low", {"effort": "low"}, True, "low"),
    ("unknown_ultra", {"reasoning_effort": "ultra"}, True, "medium"),
    ("tm_disabled", {"thinking_mode": "disabled"}, False, None),
    ("tm_disabled_max", {"thinking_mode": "disabled", "reasoning_effort": "max"}, False, None),
    ("tm_enabled", {"thinking_mode": "enabled"}, True, None),
    ("tm_enabled_low", {"thinking_mode": "enabled", "reasoning_effort": "low"}, True, "low"),
]


def load_parser(path):
    spec = importlib.util.spec_from_file_location("m3_parser_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MiniMaxM3ReasoningParser


def main():
    tmpl_path, parser_path = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.chat_template = open(tmpl_path, encoding="utf-8").read()
    Parser = load_parser(parser_path)

    ok = True
    print(f"{'case':16s} {'tmpl_on':8s} {'parser_on':10s} {'effort_tag':12s} verdict")
    for name, kw, exp_on, exp_tag in CASES:
        text = tok.apply_chat_template(
            MSGS, add_generation_prompt=True, tokenize=False, **kw)
        tail = text[-30:].rstrip()
        tmpl_on = tail.endswith("<mm:think>")
        if not (tmpl_on or tail.endswith("</mm:think>")):
            tmpl_on = None  # no prefill at all

        tag = None
        if "<effort>" in text:
            tag = text.split("<effort>")[1].split("</effort>")[0]

        parser_on = Parser(tok, chat_template_kwargs=kw)._initial_in_reasoning

        good = (tmpl_on == exp_on) and (parser_on == exp_on) and (tag == exp_tag)
        ok &= good
        print(f"{name:16s} {str(tmpl_on):8s} {str(parser_on):10s} {str(tag):12s} "
              f"{'OK' if good else 'MISMATCH exp_on=%s exp_tag=%s' % (exp_on, exp_tag)}")

    print("ALL CONSISTENT" if ok else "INCONSISTENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
