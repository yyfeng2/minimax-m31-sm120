#!/bin/sh
# Runs INSIDE the freshly built -reason5 image (--network=none).
# Fails loudly if the baked reasoning layer is missing or unparsable.
set -e

T=/opt/m31/chat_template.jinja
P=/usr/local/lib/python3.12/dist-packages/vllm/reasoning/minimax_m3_reasoning_parser.py

md5sum "$T" "$P"

python3 -m py_compile "$P"
echo PY_OK

python3 - <<'PY'
from jinja2.sandbox import ImmutableSandboxedEnvironment as E

t = open("/opt/m31/chat_template.jinja", encoding="utf-8").read()
E(trim_blocks=True, lstrip_blocks=True).parse(t)
assert "Thinking effort" in t, "effort ladder missing"
assert "thinking_mode" in t, "thinking_mode handling missing"
print("TEMPLATE_OK", len(t), "bytes")
PY

grep -q "SM120 patch" "$P" && echo PARSER_OK
grep -q "_initial_in_reasoning" "$P" && echo PARSER_STATE_OK

# Production renders with the model-dir copy (bind-mounted, same md5); the
# /opt/m31 file verified above is the spare used only when the serve command
# re-enables --chat-template.
if [ -e /data/MiniMax-M3.1-Flash/chat_template.jinja ]; then
  echo "NOTE: model dir is mounted here; its copy is authoritative unless --chat-template is set"
fi
echo VERIFY_OK
