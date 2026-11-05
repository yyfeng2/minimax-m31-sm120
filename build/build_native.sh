#!/bin/bash
# m31-sm120-native: 在 m31-sm120-port（v1 降级层 + Triton nvfp4 读路径）基础上，
# 叠加 SM120 原生 FP4/FP8 张量核稀疏 decode 内核（common/ops/msa_sm120）。
# 基础镜像仍为干净的 minimax-m31-flash；产出独立 tag，便于与 -port 做 A/B。
#   -port  : Triton on-load NVFP4 dequant decode（基线）
#   -native: mma.sync kind::mxf4nvf4 (QK) + kind::mxf8f6f4 (PV) 原生 decode
set -e
IMG=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-flash
DEST=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120-native
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
# v1 降级层
mkdir -p /tmp/patchwork/models/minimax_m3/common
cp $V/models/minimax_m3/common/sparse_attention.py /tmp/patchwork/models/minimax_m3/common/sparse_attention.py
cp $V/models/minimax_m3/common/indexer.py /tmp/patchwork/models/minimax_m3/common/indexer.py
python3 /p/patch_port.py
# v2 原生内核层（SM120 原生 FP4/FP8 decode）
python3 /p/patch_native.py
cp /tmp/patchwork/models/minimax_m3/common/sparse_attention.py $V/models/minimax_m3/common/sparse_attention.py
cp /tmp/patchwork/models/minimax_m3/common/indexer.py $V/models/minimax_m3/common/indexer.py
# 内核源码（.cu 不被上面的 *.py 循环覆盖，单独拷贝）
D=$V/models/minimax_m3/common/ops/msa_sm120
mkdir -p $D
cp /p/models/minimax_m3/common/ops/msa_sm120/__init__.py $D/
cp /p/models/minimax_m3/common/ops/msa_sm120/msa_nvfp4.py $D/
cp /p/models/minimax_m3/common/ops/msa_sm120/msa_nvfp4_sm120.cu $D/
grep -q "m31-sm120-port" $V/models/minimax_m3/common/sparse_attention.py \
  && grep -q "m31-sm120-port" $V/models/minimax_m3/common/indexer.py && echo PORT_OK
grep -q "m31-sm120-native" $V/models/minimax_m3/common/sparse_attention.py \
  && grep -q "msa_decode_nvfp4" $V/models/minimax_m3/common/sparse_attention.py \
  && test -f $D/msa_nvfp4_sm120.cu && echo NATIVE_OK
python3 -m py_compile $V/models/minimax_m3/common/sparse_attention.py $D/msa_nvfp4.py && echo PY_OK
'
docker commit -m "m31-sm120-native v2: SM120 FP4/FP8 tensor-core sparse decode" $C $DEST >/dev/null
docker rm -f $C >/dev/null
C2=fixent
docker rm -f $C2 >/dev/null 2>&1 || true
docker run -d --name $C2 --entrypoint /bin/sh $DEST -c "sleep 1" >/dev/null
sleep 2
docker commit --change="ENTRYPOINT [\"vllm\",\"serve\"]" $C2 $DEST >/dev/null
docker rm -f $C2 >/dev/null
docker inspect $DEST --format "native镜像 {{.Id}} entrypoint={{.Config.Entrypoint}}"
