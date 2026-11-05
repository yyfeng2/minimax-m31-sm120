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
cd /p
for f in \
  models/minimax_m3/common/sparse_attention.py \
  models/minimax_m3/common/indexer.py \
  models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py \
  models/minimax_m3/nvidia/sparse_attention_msa.py \
  v1/attention/backends/flashinfer_ts.py \
  v1/attention/backends/flashinfer.py \
  third_party/fmha_sm100/jit.py \
  third_party/fmha_sm100/q8kv4_prefill_adapter.py \
  third_party/fmha_sm100/prefill_q8kv4/jit.py \
  third_party/fmha_sm100/decode_q8kv4/jit.py \
  third_party/fmha_sm100/decode_q8kv4/_build_utils.py \
  third_party/fmha_sm100/cute/q8_indexer_interface.py \
  third_party/fmha_sm100/cute/src/sm100/q8kv8_indexer_prefill.py ; do
    mkdir -p "$P/$(dirname "$f")"
    cp "/p/$f" "$P/$f"
done
grep -q 120a $P/third_party/fmha_sm100/prefill_q8kv4/jit.py && echo COPY_ALL_OK
# flashinfer(第三方) gate 修正
F=/usr/local/lib/python3.12/dist-packages/flashinfer/attention/prims_ts
cp /p/flashinfer/attention/prims_ts/decode.py $F/decode.py
cp /p/flashinfer/attention/prims_ts/context.py $F/context.py
grep -q "12, 0" $F/decode.py && echo FLASHINFER_OK
'
docker commit --change 'ENTRYPOINT ["vllm", "serve"]' $C $DEST | tail -1
docker rm -f $C >/dev/null 2>&1 || true
docker inspect --format "entrypoint={{.Config.Entrypoint}}" $DEST
echo DONE
