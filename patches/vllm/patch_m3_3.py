#!/usr/bin/env python3
"""第三轮：flashinfer prims_ts 白名单加 SM120。"""
import sys, pathlib

BASE = pathlib.Path("/data/vllm-patch")

EDITS = [
    ("flashinfer/attention/prims_ts/decode.py", [
        ("_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))",
         "_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7), (12, 0))"),
    ]),
    ("flashinfer/attention/prims_ts/context.py", [
        ("_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))",
         "_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7), (12, 0))"),
    ]),
]

dirty = []
for rel, pairs in EDITS:
    p = BASE / rel
    if not p.exists():
        print(f"!! 缺失 {rel}"); dirty.append(rel); continue
    src = p.read_text()
    for old, new in pairs:
        n = src.count(old)
        if n != 1:
            print(f"!! {rel}: 命中 {n} 处: {old[:50]!r}")
            dirty.append(rel)
            continue
        src = src.replace(old, new)
    if rel not in dirty:
        p.write_text(src)
        print(f"OK  {rel}")

sys.exit(1 if dirty else 0)
