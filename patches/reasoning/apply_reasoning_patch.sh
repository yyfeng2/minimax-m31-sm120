#!/usr/bin/env bash
# 安装 / 更新 M3 思考层补丁（模板 + reasoning parser），然后重启并冒烟。
#
#   bash apply_reasoning_patch.sh [root@192.168.20.12]
#   NO_RESTART=1 bash apply_reasoning_patch.sh      # 只铺文件不重启
#
# 2026-10-25: v2 恢复官方 5 档 reasoning_effort（low/medium/high/xhigh/max），
#             none/minimal 与「不传」仍然关闭；默认仍是关闭。
#
# !! 已停用于生产 !! 2026-10-25 起生产改用 -reason5 镜像（parser 内置，
#   模板以模型目录为准，无 bind mount），重建请用
#   build/build_reasoning_image.sh。本脚本保留用于回滚到挂载版（见
#   docs/REASONING.md §5.2）。
#
# 幂等：可重复执行；每次覆盖前生成带时间戳的 .bak-* 备份。
set -euo pipefail

HOST="${1:-root@192.168.20.12}"
DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE=/data/docker-compose-m31.yml
MODEL_TMPL=/data/MiniMax-M3.1-Flash/chat_template.jinja
PATCH_DIR=/data/m3_patches
PARSER=$PATCH_DIR/minimax_m3_reasoning_parser.py
KEY=REPLACE_WITH_YOUR_API_KEY

echo "[1/5] 上传补丁文件到 $HOST"
scp "$DIR/chat_template.jinja.effort5"             "$HOST:/tmp/chat_template.jinja.new"
scp "$DIR/minimax_m3_reasoning_parser.effort5.py"  "$HOST:/tmp/minimax_m3_reasoning_parser.new"

echo "[2/5] 安装并校验"
ssh "$HOST" 'bash -s' <<'EOF'
set -euo pipefail
TS=$(date +%Y%m%d-%H%M%S)
MODEL_TMPL=/data/MiniMax-M3.1-Flash/chat_template.jinja
PATCH_DIR=/data/m3_patches
PARSER=$PATCH_DIR/minimax_m3_reasoning_parser.py
COMPOSE=/data/docker-compose-m31.yml

mkdir -p "$PATCH_DIR"

# 首次安装时保留官方 parser
if [ ! -f "$PARSER.orig" ]; then
  if [ -f "$PARSER" ]; then cp -a "$PARSER" "$PARSER.orig"; else
    echo "ERROR: $PARSER 不存在，无法保存 .orig"; exit 1
  fi
fi

install -b -S ".bak-$TS" -m 0644 /tmp/chat_template.jinja.new "$MODEL_TMPL"
install -b -S ".bak-$TS" -m 0644 /tmp/minimax_m3_reasoning_parser.new "$PARSER"

# 内容校验
grep -q "reasoning_effort" "$MODEL_TMPL"
grep -q "SM120 patch" "$PARSER"
grep -q "_initial_in_reasoning" "$PARSER"

# compose 必须把 parser 绑定挂载进容器
if ! grep -q "m3_patches/minimax_m3_reasoning_parser.py" "$COMPOSE"; then
  echo "ERROR: $COMPOSE 缺少 parser 绑定挂载，思考切分会失效"
  echo "需要: - /data/m3_patches/minimax_m3_reasoning_parser.py:"
  echo "        /usr/local/lib/python3.12/dist-packages/vllm/reasoning/minimax_m3_reasoning_parser.py:ro"
  exit 1
fi

python3 - <<'PY'
import pathlib
p = pathlib.Path("/data/m3_patches/minimax_m3_reasoning_parser.py")
src = p.read_text()
compile(src, str(p), "exec")
print("parser syntax OK")
PY
echo "files installed: template + parser"
EOF

if [ "${NO_RESTART:-0}" = "1" ]; then
  echo "[3/5] NO_RESTART=1，跳过重启"
  exit 0
fi

echo "[3/5] 重启容器（模板在启动时编译，必须重启才生效）"
ssh "$HOST" "cd /data && docker compose -p data -f $COMPOSE up -d --force-recreate"

echo "[4/5] 等待 ready（最多 400s）"
ssh "$HOST" 'for i in $(seq 1 40); do
  c=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && { echo READY; exit 0; }
  sleep 10
done; echo "TIMEOUT"; exit 1'

echo "[5/5] 冒烟：默认关闭；low/high/max 开启且无泄漏"
ssh "$HOST" "KEY=$KEY python3 -" <<'PY'
import json, os, urllib.request
KEY = os.environ["KEY"]
URL = "http://127.0.0.1:8000/v1/chat/completions"

def call(payload):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

msgs = [{"role": "user",
         "content": "一个农夫有17只羊，除了9只以外全死了，还剩几只？请先推理再回答。"}]
base = {"model": "MiniMax-M3.1-Flash", "messages": msgs, "max_tokens": 4096,
        "temperature": 0}

def probe(tag, kw):
    resp = call({**base, **({"chat_template_kwargs": kw} if kw else {})})
    m = resp["choices"][0]["message"]
    t = m.get("content") or ""
    r = m.get("reasoning") or m.get("reasoning_content") or ""
    print(f"{tag:8s} content={len(t):4d} reasoning={len(r):4d} "
          f"leak={'mm:think' in t}")
    assert t and "mm:think" not in t, f"{tag}: content 为空或标记泄漏"
    return len(r)

r_off  = probe("default", None)
r_low  = probe("low",    {"reasoning_effort": "low"})
r_high = probe("high",   {"reasoning_effort": "high"})
r_max  = probe("max",    {"reasoning_effort": "max"})

assert r_off == 0, "默认档不应产生 reasoning"
assert r_low > 0 and r_high > 0 and r_max > 0, "各思考档必须产生 reasoning"
print(f"reasoning 长度: low={r_low} high={r_high} max={r_max}")
print("SMOKE OK")
PY
