#!/usr/bin/env python3
"""第二轮：把 Q8KV4 prefill/decode/indexer 的 arch 白名单扩展 SM120。"""
import sys, pathlib

BASE = pathlib.Path("/data/vllm-patch")

EDITS = [
    ("v1/attention/backends/flashinfer_ts.py", [
        ("_TS_CAPABILITIES = ((10, 0), (10, 3), (10, 7))",
         "_TS_CAPABILITIES = ((10, 0), (10, 3), (10, 7), (12, 0))"),
    ]),
    ("third_party/fmha_sm100/prefill_q8kv4/jit.py", [
        ('_SUPPORTED_ARCHES = ("100a", "103a", "107a")',
         '_SUPPORTED_ARCHES = ("100a", "103a", "107a", "120a")'),
    ]),
    ("third_party/fmha_sm100/decode_q8kv4/jit.py", [
        ('        supported_arches=("100a", "103a", "107a"),',
         '        supported_arches=("100a", "103a", "107a", "120a"),'),
    ]),
    ("third_party/fmha_sm100/decode_q8kv4/_build_utils.py", [
        ('    supported_arches=("100a", "103a"),',
         '    supported_arches=("100a", "103a", "120a"),'),
    ]),
    ("third_party/fmha_sm100/cute/q8_indexer_interface.py", [
        ("_SUPPORTED_CAPABILITIES = frozenset({(10, 0), (10, 3), (10, 7)})",
         "_SUPPORTED_CAPABILITIES = frozenset({(10, 0), (10, 3), (10, 7), (12, 0)})"),
    ]),
    ("third_party/fmha_sm100/cute/src/sm100/q8kv8_indexer_prefill.py", [
        ("    supported_compute_capabilities = frozenset({(10, 0), (10, 3), (10, 7)})",
         "    supported_compute_capabilities = frozenset({(10, 0), (10, 3), (10, 7), (12, 0)})"),
    ]),
]

dirty = []
for rel, pairs in EDITS:
    p = BASE / rel
    if not p.exists():
        print(f"!! 缺失文件 {rel}"); dirty.append(rel); continue
    src = p.read_text()
    for old, new in pairs:
        n = src.count(old)
        if n != 1:
            print(f"!! {rel}: 命中 {n} 处（期望 1）: {old[:50]!r}")
            dirty.append(rel)
            continue
        src = src.replace(old, new)
    if rel not in dirty:
        p.write_text(src)
        print(f"OK  {rel}")

sys.exit(1 if dirty else 0)
