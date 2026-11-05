#!/usr/bin/env python3
"""m31-sm120-port v2: route MiniMax-M3 sparse *decode* attention to the SM120
native NVFP4 x FP8 tensor-core kernel.

What changes (single file: ``models/minimax_m3/common/sparse_attention.py``):

  1. Module helpers ``_msa_sm120_enabled()`` / ``_msa_sm120_load()``. The native
     module is imported lazily; the switch is ``M3_SM120_NATIVE=0`` (A/B with the
     Triton on-load dequant path, which stays intact).
  2. ``MiniMaxM3SparseTritonNvfp4Impl.__init__``: build/load the JIT extension
     at model-init time, i.e. before CUDA-graph capture. Nothing is compiled or
     synchronized inside a captured graph.
  3. ``MiniMaxM3SparseTritonNvfp4Impl.forward``: the decode branch calls
     ``msa_decode_nvfp4`` (mma.sync kind::mxf4nvf4 for QK^T, kind::mxf8f6f4 for
     PV), consuming the packed cache in place -- no dequantization. Prefill keeps
     the Triton kernel, which never touches tensor cores.

The kernel sources live in ``common/ops/msa_sm120/`` inside the image.

Usage: patch_native.py [path-to-sparse_attention.py]
Defaults to the /tmp/patchwork copy used by the build script.
"""

import pathlib
import sys

SPARSE = pathlib.Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/tmp/patchwork/models/minimax_m3/common/sparse_attention.py"
)
MARKER = "# [m31-sm120-native]"

# ---------- 1. module-level helpers ----------
ANCHOR_IMPORTS = (
    "from vllm.utils.torch_utils import get_dtype_size, nvfp4_kv_cache_full_dim\n"
)

HELPERS = (
    ANCHOR_IMPORTS
    + """
# [m31-sm120-native] SM120 native decode attention: the packed NVFP4 KV cache is
# consumed in place by mma.sync kind::mxf4nvf4 (QK^T) and kind::mxf8f6f4 (PV),
# so decode never dequantizes K/V. Prefill keeps the Triton path. Set
# M3_SM120_NATIVE=0 to A/B against the Triton on-load dequant kernel.
_MSA_SM120_NATIVE = os.environ.get("M3_SM120_NATIVE", "1") != "0"
_msa_sm120_mod = None


def _msa_sm120_enabled() -> bool:
    return (
        _MSA_SM120_NATIVE
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    )


def _msa_sm120_load():
    global _msa_sm120_mod
    if _msa_sm120_mod is None:
        from vllm.models.minimax_m3.common.ops.msa_sm120 import msa_nvfp4

        _msa_sm120_mod = msa_nvfp4
    return _msa_sm120_mod
"""
)

# ---------- 2. build the extension before graph capture ----------
INIT_OLD = """        super().__init__(*args, **kwargs)
        # nvfp4 is a quantized cache but stores packed E2M1 codes, not fp8.
        self.use_fp8_kv = False
"""

INIT_NEW = """        super().__init__(*args, **kwargs)
        # nvfp4 is a quantized cache but stores packed E2M1 codes, not fp8.
        self.use_fp8_kv = False
        # [m31-sm120-native] Compile/load the tensor-core kernel at init time
        # (model load), never inside a CUDA-graph capture.
        if _msa_sm120_enabled():
            _msa_sm120_load().build()
            logger.info_once(
                "[m31-sm120-native] SM120 native FP4/FP8 sparse decode enabled "
                "(QK: mma.sync kind::mxf4nvf4 over the packed NVFP4 cache, "
                "PV: kind::mxf8f6f4). Set M3_SM120_NATIVE=0 for the Triton "
                "on-load dequant path."
            )
"""

# ---------- 3. decode call ----------
DECODE_OLD = """        # Unit-scaled by default; the Triton kernels fold these in per tile.
        k_scale = getattr(layer, "_k_scale", None)
        v_scale = getattr(layer, "_v_scale", None)

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            minimax_m3_sparse_attn_decode(
                q[:nd],
                kv_cache,
                topk[:, :nd, :],
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:nd],
                d.decode_query_len,
                k_scale=k_scale,
                v_scale=v_scale,
            )
"""

DECODE_NEW = """        # Unit-scaled by default; the Triton kernels fold these in per tile.
        k_scale = getattr(layer, "_k_scale", None)
        v_scale = getattr(layer, "_v_scale", None)

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            if _msa_sm120_enabled():
                # [m31-sm120-native] FP4 QK^T + FP8 PV on SM120 tensor cores.
                # The *_float attributes are plain python floats (no .item()
                # sync) and are constant for the whole run.
                _msa_sm120_load().msa_decode_nvfp4(
                    q[:nd],
                    kv_cache,
                    topk[:, :nd, :],
                    d.block_table,
                    d.seq_lens,
                    out[:nd],
                    d.decode_query_len,
                    self.scale,
                    getattr(layer, "_k_scale_float", k_scale),
                    getattr(layer, "_v_scale_float", v_scale),
                )
            else:
                minimax_m3_sparse_attn_decode(
                    q[:nd],
                    kv_cache,
                    topk[:, :nd, :],
                    d.block_table,
                    d.seq_lens,
                    self.num_kv_heads,
                    self.scale,
                    out[:nd],
                    d.decode_query_len,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
"""


def main() -> int:
    src = SPARSE.read_text()
    if MARKER in src:
        print("ALREADY_PATCHED", SPARSE)
        return 0
    for name, old in (("imports", ANCHOR_IMPORTS), ("init", INIT_OLD), ("decode", DECODE_OLD)):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"anchor {name}: expected 1 occurrence, found {n}")
    src = src.replace(ANCHOR_IMPORTS, HELPERS, 1)
    src = src.replace(INIT_OLD, INIT_NEW, 1)
    src = src.replace(DECODE_OLD, DECODE_NEW, 1)
    if MARKER not in src or "msa_decode_nvfp4" not in src:
        raise SystemExit("patch did not apply")
    SPARSE.write_text(src)
    print("PATCHED", SPARSE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
