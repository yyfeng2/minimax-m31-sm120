#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
cd $P
echo "===== A. GPU capability ====="
python -c "import torch; c=torch.cuda.get_device_capability(0); print('capability =', c); import vllm.platforms as P2; print('device name:', torch.cuda.get_device_name(0))" 2>&1 | grep -aiE "capability|device name|error" | head -4
echo
echo "===== B. is_device_capability* 定义位置 ====="
grep -rn "def is_device_capability_family\|def is_device_capability\b" --include=*.py . 2>/dev/null | grep -avi "test\|_cross\|util" | cut -c1-150 | head -6
echo
echo "===== C. flashinfer_ts supported_kv_cache_dtypes ====="
sed -n '444,456p' v1/attention/backends/flashinfer_ts.py | cut -c1-150
echo
echo "===== D. indexer_nvfp4 里的 SM/kernel 检查 ====="
grep -n "capability\|cuda_arch\|sm_\|compile\|is_device\|120\|107\|103\|fmha\|cutlass" models/minimax_m3/nvidia/indexer_nvfp4.py 2>/dev/null | cut -c1-150 | head -15
echo
echo "===== E. CUTLASS 支持哪些 SM（device cap whitelist）====="
ls vllm/_custom_ops/cutlass/ 2>/dev/null | head
grep -rn "is_device_capability(1[02]\|capability.major == 1[02]\|(12,\|(10," vllm/_custom_ops/cutlass/*.py 2>/dev/null | grep -aiE "supports|capab" | cut -c1-150 | head -8
echo
echo "===== F. is_device_capability(120) 或 family(120) 现成调用 ====="
grep -rn "is_device_capability_family(120)\|is_device_capability(12" --include=*.py . 2>/dev/null | grep -avi "test" | cut -c1-120 | head -8
echo
echo "===== G. nvfp4 quantize kernel 定义/gate ====="
grep -rn "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping" --include=*.py v1/ | grep -avi test | cut -c1-150 | head -6
