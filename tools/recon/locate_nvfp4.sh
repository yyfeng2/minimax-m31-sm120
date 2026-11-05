#!/bin/bash
# 定位 nvfp4 KV / MSA backend 的 gate
cd /usr/local/lib/python3.12/dist-packages/vllm
echo "== 报错原文 =="
grep -rn "supports an NVFP4\|only with the SM" --include=*.py . 2>/dev/null | grep -avi test | cut -c1-200 | head -5
echo "== backends 列表 =="
ls v1/attention/backends/ | grep -avi test_
echo "== backends/__init__ 中 msa 相关 =="
grep -n "msa\|MSA" v1/attention/backends/__init__.py | cut -c1-180 | head -15
echo "== nvfp4 出现文件(含 msa/mla) =="
grep -rln "nvfp4" v1/attention/backends/ | head
echo "== attention backend 校验处 =="
grep -rn "kv_cache_dtype.*nvfp4\|nvfp4.*kv_cache_dtype" v1/ --include=*.py | grep -avi "test\|backends/flashinfer\|backends/mla/pdfa" | cut -c1-200 | head -20
