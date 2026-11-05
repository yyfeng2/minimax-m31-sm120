#!/bin/bash
# 从镜像导出原始文件 -> 冻结 host patch 目录(只导出6个原始文件供 patch)
set -e
IMG=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-flash
P=/usr/local/lib/python3.12/dist-packages/vllm
OUT=/data/vllm-patch
rm -rf $OUT
mkdir -p $OUT/models/minimax_m3/common $OUT/models/minimax_m3/nvidia $OUT/third_party/fmha_sm100
docker run --rm --entrypoint /bin/sh -v $OUT:/out $IMG -c '
P=/usr/local/lib/python3.12/dist-packages/vllm
cp $P/models/minimax_m3/common/sparse_attention.py        /out/models/minimax_m3/common/
cp $P/models/minimax_m3/common/indexer.py                /out/models/minimax_m3/common/
cp $P/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py /out/models/minimax_m3/nvidia/
cp $P/models/minimax_m3/nvidia/sparse_attention_msa.py   /out/models/minimax_m3/nvidia/
cp $P/third_party/fmha_sm100/jit.py                      /out/third_party/fmha_sm100/
cp $P/third_party/fmha_sm100/q8kv4_prefill_adapter.py    /out/third_party/fmha_sm100/
echo EXPORT_DONE
'
find $OUT -type f | sort
echo "--- 文件类型校验 ---"
file $OUT/models/minimax_m3/nvidia/*.py $OUT/third_party/fmha_sm100/*.py
