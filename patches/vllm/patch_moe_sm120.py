#!/usr/bin/env python3
"""[m31-sm120-moe] MiniMax-M3.1 routed-expert NVFP4 activation scheme.

M3 declares qat_activation_key=kNvfp4DynamicToken (per-token FC1 input
scale).  The only backend that implements per-token NVFP4 activations is
FlashInfer TRT-LLM, whose routed-MoE cubins are SM100-only.  On SM120 every
FLASHINFER_* backend is therefore filtered out by the NvFp4 oracle and vLLM
falls back to MARLIN, which dequantizes the FP4 weights back to BF16 -- i.e.
no FP4 tensor-core compute at all.

With M3_SM120_MOE_W4A4=1 (default) the FC1 input scheme becomes kNvfp4Dynamic
(per-tensor dynamic outer scale + 16-wide E4M3 block scales), which is exactly
what FlashInfer's SM120 CUTLASS NVFP4 MoE kernel implements natively
(mma.sync kind::mxf4nvf4).  The routed experts then run W4A4 on the FP4
tensor cores.  This checkpoint's FC1/FC2 input global scales are 1.0
placeholders, so only the outer-scale granularity changes, not the payload.

M3_SM120_MOE_W4A4=0 keeps the checkpoint scheme (MARLIN W4A16 fallback) for
A/B testing.
"""
import os
import sys

V = "/usr/local/lib/python3.12/dist-packages/vllm"
P = os.path.join(V, "models/minimax_m3/nvidia/model.py")

src = open(P).read()
if "[m31-sm120-moe]" in src:
    print("MOE_PATCH_ALREADY")
    sys.exit(0)

# 1) os import
if "\nimport os\n" not in src:
    old = "from collections.abc import Iterable\n\nimport torch"
    assert src.count(old) == 1, "import anchor"
    src = src.replace(old, "import os\nfrom collections.abc import Iterable\n\nimport torch")
    print("MOE_ADDED_IMPORT_OS")

# 2) kNvfp4Dynamic import
old = (
    "from vllm.model_executor.layers.quantization.utils.quant_utils import (\n"
    "    kMxfp8Dynamic,\n"
    "    kNvfp4DynamicToken,\n"
    ")"
)
new = (
    "from vllm.model_executor.layers.quantization.utils.quant_utils import (\n"
    "    kMxfp8Dynamic,\n"
    "    kNvfp4Dynamic,\n"
    "    kNvfp4DynamicToken,\n"
    ")"
)
assert src.count(old) == 1, "quant_utils import anchor"
src = src.replace(old, new)

# 3) activation scheme switch
old = "            qat_activation_key=kNvfp4DynamicToken,\n"
new = (
    "            # [m31-sm120-moe] W4A4 NVFP4 on SM120: FlashInfer CUTLASS MoE\n"
    "            # implements kNvfp4Dynamic (per-tensor dynamic outer scale +\n"
    "            # 16-wide E4M3 block scales).  The per-token variant is\n"
    "            # SM100-TRTLLM only, which would leave SM120 on the MARLIN\n"
    "            # W4A16 dequant fallback.  M3_SM120_MOE_W4A4=0 restores it.\n"
    "            qat_activation_key=(\n"
    "                kNvfp4DynamicToken\n"
    "                if os.environ.get(\"M3_SM120_MOE_W4A4\", \"1\") == \"0\"\n"
    "                else kNvfp4Dynamic\n"
    "            ),\n"
)
assert src.count(old) == 1, "qat_activation_key anchor"
src = src.replace(old, new)

open(P, "w").write(src)
print("MOE_PATCH_OK")
