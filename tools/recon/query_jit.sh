#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm/third_party/fmha_sm100
echo "===== A. _DEFAULT_ARCHS / arch 常量 ====="
grep -n "_DEFAULT_ARCHS\s*=\|_ARCHS\s*=\|120\|107\|103" $P/jit.py | grep -aivE "def |#" | head -25
echo
echo "===== B. _target_arch 实现 ====="
grep -n "_target_arch" -A 12 $P/jit.py | head -30
echo
echo "===== C. _INDEXER_MODULE_VARIANTS 全表 ====="
grep -n "_INDEXER_MODULE_VARIANTS" -A 20 $P/jit.py | head -30
echo
echo "===== D. q8kv4 adapters arch gate ====="
grep -n "_SUPPORTED_ARCHES\|120\|107\|103\|device_capability" $P/q8kv4_prefill_adapter.py $P/q8kv4_decode_adapter.py 2>/dev/null | cut -c1-140 | head -20
echo
echo "===== E. sparse 相关 JIT variant 名 ====="
grep -rn "107a\|103a\|120a\|variant" $P/jit.py | grep -aiE "arch|variant|=" | cut -c1-150 | head -20
