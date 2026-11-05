#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
cd $P
echo "===== A. is_device_capability_family / is_device_capability 定义 ====="
grep -rn "def is_device_capability_family\|def is_device_capability\b" --include=*.py platforms/ vllm/ | cut -c1-160 | head -6
F=$(grep -rln "def is_device_capability_family" --include=*.py platforms/ vllm/ | head -1)
echo "--- file: $F ---"
grep -n "def is_device_capability_family" -A 14 "$F" | cut -c1-120 | head -20
echo
echo "===== B. flashinfer_ts 440-530 ====="
sed -n '440,530p' v1/attention/backends/flashinfer_ts.py | cut -c1-200 | head -95
echo
echo "===== C. indexer.py 505-560 (nvfp4 gate) ====="
sed -n '505,560p' models/minimax_m3/common/indexer.py | cut -c1-180 | head -60
echo
echo "===== D. sparse_attention_msa.py 178-200 && 470-495 ====="
sed -n '176,200p' models/minimax_m3/nvidia/sparse_attention_msa.py | cut -c1-180
echo "---"
sed -n '475,495p' models/minimax_m3/nvidia/sparse_attention_msa.py | cut -c1-180
echo
echo "===== E. nvfp4 quantize kernel SM gate ====="
grep -rn "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping\|def.*nvfp4.*kv\|kv.*nvfp4" v1/kv_cache_utils.py vllm/_custom_ops/ 2>/dev/null | grep -avi "import\|^.*#" | cut -c1-180 | head -10
grep -rn "12.0\|device_capability.major\|capability" vllm/_custom_ops/cutlass/__init__.py 2>/dev/null | grep -i nvfp4 | cut -c1-180 | head -5
