#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm/third_party/fmha_sm100
echo "===== A. def _target_arch 全包搜索 ====="
grep -rn "_target_arch" $P/ | grep -aviE "\.pyc|_target_arch(device)" | cut -c1-130 | head
grep -rn "def _target_arch" -A 10 $P/jit.py $P/*.py 2>/dev/null | head -18
echo
echo "===== B. _DEFAULT_ARCHS 的使用点 ====="
grep -rn "_DEFAULT_ARCHS" $P/jit.py | cut -c1-150
echo
echo "===== C. variant 编译 arch 选择（240-300）====="
sed -n '238,300p' $P/jit.py | cut -c1-160
echo
echo "===== D. _MODULES 完整定义（q8kv4/reduction/sparse_topk/plan）====="
grep -n '"_sparse\|"plan\|"q8kv4\|"reduction\|"sparse_topk' -A 4 $P/jit.py | grep -aE '"[a-z]|"archs|\(.*\.cu' | cut -c1-160 | head -24
echo
echo "===== E. sparse_topk 的 fgmma/tile Arch — 编译宏 ====="
grep -rn "sm100\|sm103\|sm107\|sm120\|_SM_\|MAJOR" $P/_MODULES/*.py $P/*.jinja $P/csrc/*.h 2>/dev/null | head -8 | cut -c1-150
