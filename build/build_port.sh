#!/bin/bash
# m31-sm120-port: 在 nvfp4 gate patch 的 vllm-patch 树之上叠加 SM120 降级层（v1），
# 固化独立镜像 minimax-m31-sm120-port（基础镜像保持干净）。
# 前置：/data/vllm-patch 已含 13 个 vllm 文件 + 2 个 flashinfer 文件的 gate 修正；
#       /data/patch_port.py 为 v1 降级 patch。
set -e
IMG=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-flash
DEST=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-port
C=portbuild
docker rm -f $C >/dev/null 2>&1 || true
docker run --name $C --entrypoint /bin/sh -v /data/vllm-patch:/p:ro -v /data:/d:ro $IMG -c '
set -e
V=/usr/local/lib/python3.12/dist-packages/vllm
cd /p
N=0
for f in $(find . -name "*.py" -path "./models/*" -o -name "*.py" -path "./third_party/*" -o -name "*.py" -path "./v1/*"); do
  g=/usr/local/lib/python3.12/dist-packages/${f#./}
  g2=$V/${f#./}
  if [ -f "$g2" ]; then cp -a "$f" "$g2"; N=$((N+1)); else echo "MISS $f"; fi
done
echo "COPIED=$N"
F=/usr/local/lib/python3.12/dist-packages/flashinfer/attention/prims_ts
cp /p/flashinfer/attention/prims_ts/decode.py $F/decode.py
cp /p/flashinfer/attention/prims_ts/context.py $F/context.py
grep -q "120a" $V/third_party/fmha_sm100/decode_q8kv4/jit.py && echo COPY_ALL_OK
grep -q "12, 0" $F/decode.py && echo FLASHINFER_OK
# v1 降级层：复制被 patch 文件到可写工作区 -> 应用 -> 写回镜像树
mkdir -p /tmp/patchwork/models/minimax_m3/common
cp $V/models/minimax_m3/common/sparse_attention.py /tmp/patchwork/models/minimax_m3/common/sparse_attention.py
cp $V/models/minimax_m3/common/indexer.py /tmp/patchwork/models/minimax_m3/common/indexer.py
python3 /p/patch_port.py
cp /tmp/patchwork/models/minimax_m3/common/sparse_attention.py $V/models/minimax_m3/common/sparse_attention.py
cp /tmp/patchwork/models/minimax_m3/common/indexer.py $V/models/minimax_m3/common/indexer.py
grep -q "m31-sm120-port" $V/models/minimax_m3/common/sparse_attention.py \
  && grep -q "m31-sm120-port" $V/models/minimax_m3/common/indexer.py && echo PORT_OK
'
docker commit -m 'm31-sm120-port v1: nvfp4->fp8 degrade for SM120' $C $DEST >/dev/null
docker rm -f $C >/dev/null
# commit 会把运行态 entrypoint 固化成 /bin/sh，对镜像自身再 commit 一次恢复官方入口
C2=fixent
docker rm -f $C2 >/dev/null 2>&1 || true
docker run -d --name $C2 --entrypoint /bin/sh $DEST -c 'sleep 1' >/dev/null
sleep 2
docker commit --change='ENTRYPOINT ["vllm","serve"]' $C2 $DEST >/dev/null
docker rm -f $C2 >/dev/null
docker inspect $DEST --format 'port镜像 {{.Id}} entrypoint={{.Config.Entrypoint}}'
