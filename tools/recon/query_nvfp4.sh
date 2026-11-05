#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
cd $P
echo "===== 1. is_device_capability_family 定义 ====="
grep -rn "def is_device_capability_family" --include=*.py platform/ | cut -c1-160
echo
echo "===== 2. kv_cache_interface.py 90-135 ====="
sed -n '90,135p' v1/kv_cache_interface.py
echo
echo "===== 3. flashinfer_ts.py 关键 gate ====="
grep -n "get_supported_kernel_block_sizes\|MultipleOf\|is_device_capability_family\|kv_cache_dtype\|supported_kernel\|FLASHINFER" v1/attention/backends/flashinfer_ts.py | cut -c1-180 | head -30
echo
echo "===== 4. minimax_m3/common/indexer.py ====="
grep -rn "get_supported_kernel_block_sizes\|128\|MultipleOf\|nvfp4" models/minimax_m3/common/indexer.py | cut -c1-160 | head -15
echo
echo "===== 5. sparse_attention_msa.py SM 检查 ====="
grep -rn "capability\|device_capability\|sm_\|cc_\|SM12\|120\|compile.*sm\|cuda_arch" models/minimax_m3/nvidia/sparse_attention_msa.py | cut -c1-160 | head -20
echo
echo "===== 6. use_prefill_query_quantization 消费位置 ====="
grep -rn "use_prefill_query_quantization" --include=*.py . | grep -avi test | cut -c1-200 | head -12
