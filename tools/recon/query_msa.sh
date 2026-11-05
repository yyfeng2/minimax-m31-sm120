#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
cd $P
echo "===== A. supports_cutlass_sparse_decode 定义 ====="
grep -rn "def supports_cutlass_sparse_decode" -A 24 --include=*.py models/ | head -32 | cut -c1-170
echo
echo "===== B. sparse_attention_msa.py 算子库 import ====="
sed -n '1,60p' models/minimax_m3/nvidia/sparse_attention_msa.py | grep -aE "^import|^from|cutlass|fmha|msa" | cut -c1-160 | head -20
echo
echo "===== C. MSA decode 内核来源（CUTLASS/fmha）====="
grep -rn "MSACutlassDecode\|cutlass.*decode\|sparse_decode\|fmha_sm100" models/minimax_m3/nvidia/sparse_attention_msa.py | cut -c1-170 | grep -aiv "plan_cache\|min_batch" | head -20
echo
echo "===== D. JIT 加载路径 520-565 ====="
sed -n '520,565p' third_party/fmha_sm100/jit.py | cut -c1-160
echo
echo "===== E. nvcc / CUDA 版本（JIT 前提）====="
which nvcc && nvcc --version 2>/dev/null | grep -a release | head -1
python -c "import torch; print('torch cuda:', torch.version.cuda)" 2>/dev/null
echo
echo "===== F. is_nvfp4_kv_cache 判定 ====="
grep -rn "def is_nvfp4_kv_cache" -A 8 --include=*.py . | grep -avi test | head -12 | cut -c1-150
