#!/bin/bash
P=/usr/local/lib/python3.12/dist-packages/vllm
cd $P
echo "===== A. GPU capability (raw) ====="
python -c "
import torch
print('cap=', torch.cuda.get_device_capability(0))
print('name=', torch.cuda.get_device_name(0))
print('cc_supported=', torch.cuda.get_device_capability())
" 2>/dev/null | grep -ivE "warn|deprecat"
echo
echo "===== B. fmha_sm100 包结构/加载方式 ====="
ls third_party/fmha_sm100/ 2>/dev/null | head
grep -rn "load_inline\|cuda_arch\|compute_capability\|capability\|arch" third_party/fmha_sm100/*.py 2>/dev/null | grep -aiE "jit|load_inline|arch|capab" | cut -c1-160 | head -12
echo "--- fmha_sm100 顶层 init ---"
find third_party/fmha_sm100 -maxdepth 2 -name "*.py" | head -10
echo
echo "===== C. CUTLASS 库位置与 SM 支持 ====="
python - <<'PY' 2>/dev/null
import importlib
try:
    import vllm._custom_ops.cutlass as c
    print('cutlass module ok, file:', getattr(c,'__file__',None))
except Exception as e:
    print('cutlass import:', type(e).__name__, e)
PY
find . -maxdepth 4 -type d -name "cutlass" 2>/dev/null | head
grep -rln "def get_supported_capabilities\|SM120\|sm120\|(12, 0)" compilation/ 2>/dev/null | head -3
echo
echo "===== D. 该镜像 vLLM 版本 ====="
python -c "import vllm; print(vllm.__version__)" 2>/dev/null
