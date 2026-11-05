#!/bin/bash
# 导出 nvfp4 sm120 链路剩余相关文件（原始版）到 host，供分析/patch
set -e
IMG=docker.m.daocloud.io/vllm/vllm-openai:minimax-m31-flash
OUT=/data/nvfp4add
rm -rf $OUT; mkdir -p $OUT
docker run --rm --entrypoint /bin/sh -v $OUT:/out $IMG -c '
BS=/usr/local/lib/python3.12/dist-packages/vllm/third_party/fmha_sm100
mkdir -p /out/prefill_q8kv4 /out/decode_q8kv4 /out/cute /out/models
cp $BS/prefill_q8kv4/jit.py /out/prefill_q8kv4/
cp $BS/prefill_q8kv4/build.py /out/prefill_q8kv4/
cp $BS/decode_q8kv4/jit.py /out/decode_q8kv4/
cp $BS/decode_q8kv4/build.py /out/decode_q8kv4/
cp $BS/decode_q8kv4/_build_utils.py /out/decode_q8kv4/
cp $BS/decode_q8kv4/q8kv4_decode_adapter.py /out/decode_q8kv4/ 2>/dev/null || true
cp $BS/cute/q8kv8_indexer_prefill.py /out/cute/
cp $BS/cute/q8_indexer_interface.py /out/cute/
cp $BS/q8kv4_decode_adapter.py /out/
cp /usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py /out/models/
cp /usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/nvidia/indexer_nvfp4.py /out/models/
echo EXPORT_DONE
'
find $OUT -type f | sort
