#!/bin/bash
set -e
IMG=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-flash
DEST=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-sm120
P=/usr/local/lib/python3.12/dist-packages/vllm
C=m31fix
docker rm -f $C >/dev/null 2>&1 || true
docker run --name $C --entrypoint /bin/sh -v /data/vllm-patch:/p:ro $IMG -c '
set -e
P=/usr/local/lib/python3.12/dist-packages/vllm
cp /p/models/minimax_m3/common/sparse_attention.py        $P/models/minimax_m3/common/sparse_attention.py
cp /p/models/minimax_m3/common/indexer.py                $P/models/minimax_m3/common/indexer.py
cp /p/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py $P/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py
cp /p/models/minimax_m3/nvidia/sparse_attention_msa.py   $P/models/minimax_m3/nvidia/sparse_attention_msa.py
cp /p/third_party/fmha_sm100/jit.py                      $P/third_party/fmha_sm100/jit.py
cp /p/third_party/fmha_sm100/q8kv4_prefill_adapter.py    $P/third_party/fmha_sm100/q8kv4_prefill_adapter.py
# 校验
grep -l "120" $P/third_party/fmha_sm100/jit.py >/dev/null && echo COPY_VERIFY_OK
'
docker commit $C $DEST
docker rm -f $C >/dev/null 2>&1 || true
echo "DONE_IMAGE=$DEST"
docker images | grep minimax-m31
