#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
echo "===== 1. sparse_attention_msa.py 495-560 ====="
ssh -p 22 root@192.168.20.12 "sed -n '495,560p' $P/models/minimax_m3/nvidia/sparse_attention_msa.py" 2>/dev/null || \
docker exec minimax-m31 sed -n '495,560p' $P/models/minimax_m3/nvidia/sparse_attention_msa.py
echo
echo "===== 2. Q8KV4 prefill supports only 的检查点 ====="
grep -rn "Q8KV4 prefill attention supports" $P/models/minimax_m3/ $P/third_party/fmha_sm100/ | cut -c1-160
