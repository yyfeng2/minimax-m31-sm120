#!/usr/bin/env python3
"""放宽 MiniMax M3 nvfp4 链路的 SM100-only 门槛到 SM120（RTX PRO 6000）。
   就地替换 + 命中计数校验，未全部命中则回滚并报错。"""
import shutil, sys, pathlib

import os
BASE = pathlib.Path(os.environ.get("VLLM_BASE", "/usr/local/lib/python3.12/dist-packages/vllm"))

# (相对路径, [(旧, 新), ...])
EDITS = [
    ("models/minimax_m3/common/sparse_attention.py", [
        ("""        and current_platform.is_device_capability_family(100)
        and topk_blocks in (4, 8, 16, 32)
        and kv_cache_dtype != "fp8_e5m2"
""",
         """        and (current_platform.is_device_capability_family(100)
             or current_platform.is_device_capability_family(120))
        and topk_blocks in (4, 8, 16, 32)
        and kv_cache_dtype != "fp8_e5m2"
"""),
    ]),
    ("models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py", [
        ("""        (decode_backend == "cutlass" or nvfp4)
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and (kv_cache_dtype in ("fp8", "fp8_e4m3") or nvfp4)
""",
         """        (decode_backend == "cutlass" or nvfp4)
        and current_platform.is_cuda()
        and (current_platform.is_device_capability_family(100)
             or current_platform.is_device_capability_family(120))
        and (kv_cache_dtype in ("fp8", "fp8_e4m3") or nvfp4)
"""),
    ]),
    ("models/minimax_m3/nvidia/sparse_attention_msa.py", [
        ("""        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            raise ValueError("MiniMax M3 NVFP4 KV cache requires an SM100-family GPU")
""",
         """        if not (
            current_platform.is_cuda()
            and (
                current_platform.is_device_capability_family(100)
                or current_platform.is_device_capability_family(120)
            )
        ):
            raise ValueError(
                "MiniMax M3 NVFP4 KV cache requires an SM100/SM120-family GPU"
            )
"""),
    ]),
    ("models/minimax_m3/common/indexer.py", [
        ("""                current_platform.is_device_capability(100)
                or current_platform.is_device_capability(103)
                or current_platform.is_device_capability(107)
""",
         """                current_platform.is_device_capability(100)
                or current_platform.is_device_capability(103)
                or current_platform.is_device_capability(107)
                or current_platform.is_device_capability(120)
"""),
    ]),
    (pathlib.Path("third_party/fmha_sm100/jit.py"), [
        ('_DEFAULT_ARCHS = ("100a", "103a", "100f")',
         '_DEFAULT_ARCHS = ("100a", "103a", "100f", "120a")'),
    ]),
    (pathlib.Path("third_party/fmha_sm100/q8kv4_prefill_adapter.py"), [
        ("_SUPPORTED_ARCHES = ((10, 0), (10, 3), (10, 7))",
         "_SUPPORTED_ARCHES = ((10, 0), (10, 3), (10, 7), (12, 0))"),
    ]),
]

dirty = []
for rel, pairs in EDITS:
    p = BASE / rel
    src = p.read_text()
    for old, new in pairs:
        n = src.count(old)
        if n != 1:
            print(f"!! {rel}: 期望命中 1 处, 实际 {n} 处 —— {old[:60]!r}")
            dirty.append(rel)
            continue
        src = src.replace(old, new)
    if rel not in dirty:
        p.write_text(src)
        print(f"OK  {rel}")

if dirty:
    print("存在未命中, 已改文件保留, 请人工核对")
    sys.exit(1)
print("ALL PATCHED")
