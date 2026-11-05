#!/bin/bash
set -e
docker cp /tmp/__init__.py minimax-m31:/tmp/msa_sm120/__init__.py
docker cp /data/patch_native.py minimax-m31:/tmp/patch_native.py
docker exec minimax-m31 bash -lc '
set -e
V=/usr/local/lib/python3.12/dist-packages/vllm
D=$V/models/minimax_m3/common/ops/msa_sm120
mkdir -p $D
cp /tmp/msa_sm120/__init__.py /tmp/msa_sm120/msa_nvfp4.py /tmp/msa_sm120/msa_nvfp4_sm120.cu $D/
mkdir -p /tmp/patchwork/models/minimax_m3/common
cp $V/models/minimax_m3/common/sparse_attention.py /tmp/patchwork/models/minimax_m3/common/sparse_attention.py
python3 /tmp/patch_native.py /tmp/patchwork/models/minimax_m3/common/sparse_attention.py
python3 -m py_compile /tmp/patchwork/models/minimax_m3/common/sparse_attention.py && echo COMPILE_OK
grep -c m31-sm120-native /tmp/patchwork/models/minimax_m3/common/sparse_attention.py
'
