"""SM120 native (FP4/FP8 tensor-core) MiniMax-M3 sparse decode attention.

Thin loader around ``msa_nvfp4_sm120.cu``. The kernel reads the packed NVFP4 KV
cache in place for QK^T (``mma.sync kind::mxf4nvf4``) and runs PV on the FP8
block-scaled MMA; see the .cu header for the exact contract.

The extension is JIT-built once per process and cached under
``TORCH_EXTENSIONS_DIR``. The build is triggered by :func:`build` (called from
the backend constructor, i.e. before CUDA-graph capture) so no compilation or
``.item()`` sync ever happens inside a captured graph.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch

_EXT = None
_LOG2E = 1.4426950408889634
# Layer scale tensors are constant; keep the converted floats (and a reference so
# the data_ptr stays valid) instead of syncing on every decode step.
_SCALE_CACHE: dict[int, tuple[torch.Tensor, float]] = {}

ScaleLike = Union[float, int, torch.Tensor, None]

# Split-K (flash-decoding) tuning: keep ~this many CTAs resident; the decode
# grid is (T, Hkv, nchunk) and each chunk owns topk/nchunk pages.
_TARGET_CTAS = 256
_MAX_CHUNKS = 32
_PART_BUFFERS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _num_chunks(T: int, num_kv_heads: int, topk_cap: int) -> int:
    """Chunk count for this launch (constant per CUDA-graph bucket)."""
    env = os.environ.get("MSA_SM120_CHUNKS")
    if env:
        return max(1, min(int(env), topk_cap, _MAX_CHUNKS))
    n = _TARGET_CTAS // max(1, T * num_kv_heads)
    return max(1, min(topk_cap, 16, n))


def _part_buffers(T: int, hq: int, nchunk: int, device: torch.device):
    """fp32 partial O + (m, l), cached per shape so CUDA graphs see stable
    addresses. Total size is bounded by _TARGET_CTAS * hq * 128 floats."""
    key = (T, hq, nchunk, device.type, device.index)
    buf = _PART_BUFFERS.get(key)
    if buf is None:
        o = torch.empty((nchunk, T, hq, 128), dtype=torch.float32, device=device)
        ml = torch.empty((nchunk, T, hq, 2), dtype=torch.float32, device=device)
        buf = (o, ml)
        _PART_BUFFERS[key] = buf
    return buf


def _scale_float(s: ScaleLike) -> float:
    if s is None:
        return 1.0
    if isinstance(s, (float, int)):
        return float(s)
    key = id(s)
    hit = _SCALE_CACHE.get(key)
    if hit is not None and hit[0] is s:
        return hit[1]
    v = float(s.reshape(-1)[0].item())
    _SCALE_CACHE[key] = (s, v)
    return v


def build() -> None:
    """Compile/load the extension now (idempotent)."""
    _ext()


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msa_nvfp4_sm120.cu")
        _EXT = load(
            name="msa_nvfp4_sm120",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-gencode=arch=compute_120a,code=sm_120a",
                "--expt-relaxed-constexpr",
            ],
            verbose=os.environ.get("MSA_SM120_VERBOSE", "0") == "1",
        )
    return _EXT


def msa_decode_nvfp4(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    decode_query_len: int,
    scale: float,
    k_scale: ScaleLike = None,
    v_scale: ScaleLike = None,
    debug_scores: Optional[torch.Tensor] = None,
    debug_stage: Optional[torch.Tensor] = None,
) -> None:
    """q [T, Hq, 128] bf16; kv uint8 [pages, 2*Hkv, 128, 72]; topk [Hkv, T, topk].

    ``topk_indices`` may be a transposed view (the vLLM buffer is [T, Hkv, topk]),
    so its inner strides are passed through; q/out/kv/block_table must be dense.
    ``k_scale``/``v_scale`` accept a python float or a 1-element tensor.
    """
    hq = int(q.shape[1])
    num_kv_heads = int(topk_indices.shape[0])
    assert q.shape[-1] == 128 and out.shape[-1] == 128
    assert q.is_contiguous() and out.is_contiguous()
    assert hq % num_kv_heads == 0
    assert topk_indices.stride(-1) == 1 and topk_indices.dtype == torch.int32
    assert block_table.stride(-1) == 1 and seq_lens.is_contiguous()
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(-1) == 1
    assert kv_cache.stride(1) == 128 * 72
    assert kv_cache.stride(0) == 2 * num_kv_heads * 128 * 72
    bt_stride = int(block_table.stride(0)) if block_table.dim() == 2 else int(seq_lens.numel())
    T = int(q.shape[0])
    nchunk = _num_chunks(T, num_kv_heads, int(topk_indices.shape[-1]))
    if nchunk > 1:
        o_part, ml_part = _part_buffers(T, hq, nchunk, q.device)
        op, mp = o_part.data_ptr(), ml_part.data_ptr()
    else:
        op = mp = 0
    _ext().msa_decode(
        q.data_ptr(),
        kv_cache.data_ptr(),
        topk_indices.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        out.data_ptr(),
        T,
        int(topk_indices.shape[-1]),
        int(topk_indices.stride(0)),
        int(topk_indices.stride(1)),
        int(decode_query_len),
        num_kv_heads,
        bt_stride,
        op,
        mp,
        nchunk,
        float(scale) * _LOG2E * _scale_float(k_scale),
        _scale_float(v_scale),
        debug_scores.data_ptr() if debug_scores is not None else 0,
        debug_stage.data_ptr() if debug_stage is not None else 0,
        torch.cuda.current_stream().cuda_stream,
    )
