#!/bin/bash
# m31-sm120-reason5: 在 -moe-native 镜像上烤入五档 reasoning 补丁，不再依赖挂载。
#
#   bash build_reasoning_image.sh [root@192.168.20.12]
#
# 产出：docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-moe-native-reason5
#   .../vllm/reasoning/minimax_m3_reasoning_parser.py  ← 五档 parser（覆盖官方实现，生产使用）
#   /opt/m31/chat_template.jinja                       ← 五档模板备用副本（生产不用）
#
# 生产模板在模型目录 /data/MiniMax-M3.1-Flash/chat_template.jinja（同 md5），
# 随模型目录挂载进容器，compose 的 --chat-template 已注释；镜像内这份只在
# 取消注释 --chat-template 时才会生效。注意模型目录是 bind mount，镜像里
# 放同名文件会被宿主覆盖，所以备用副本放在 /opt。
#
# 本机没有 docker bridge 网络，Dockerfile 里不能写 RUN；校验放在 build 之后
# 用 `docker run --network=none` 跑 build/verify_reason5.sh。
set -euo pipefail

HOST="${1:-root@192.168.20.12}"
BASE=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-moe-native
DEST=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-moe-native-reason5
DIR="$(cd "$(dirname "$0")/.." && pwd)"
CTX=/data/m31-image-reason5

echo "[1/4] 上传构建上下文到 $HOST:$CTX"
ssh "$HOST" "mkdir -p $CTX"
scp "$DIR/build/Dockerfile.reason5"                        "$HOST:$CTX/Dockerfile"
scp "$DIR/build/verify_reason5.sh"                         "$HOST:$CTX/verify_reason5.sh"
scp "$DIR/patches/reasoning/chat_template.jinja.effort5"    "$HOST:$CTX/chat_template.jinja"
scp "$DIR/patches/reasoning/minimax_m3_reasoning_parser.effort5.py" \
                                                           "$HOST:$CTX/minimax_m3_reasoning_parser.py"

echo "[2/4] docker build（FROM $BASE）"
ssh "$HOST" "cd $CTX && docker build -t $DEST ."

echo "[3/4] 镜像内校验（--network=none）"
ssh "$HOST" "docker run --rm --network=none -v $CTX:/ctx:ro --entrypoint /bin/sh $DEST -c 'sh /ctx/verify_reason5.sh'"

echo "[4/4] 结果"
ssh "$HOST" "docker inspect $DEST --format 'reason5 镜像 {{.Id}} entrypoint={{.Config.Entrypoint}} size={{.Size}}'"
