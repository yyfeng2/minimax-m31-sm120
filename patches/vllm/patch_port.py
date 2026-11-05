#!/usr/bin/env python3
"""m31-sm120-port 移植分支 v1: nvfp4+SM120 自动降级为 fp8 KV。

根因（见 SM120_fault_points.md）：RTX PRO 6000 (SM120) 无 TMEM/tcgen05，
nvfp4 主链（cute CuTe-DSL prefill + CUTLASS q8kv4 decode）全部 SM100/103/107 专用。
但 vLLM 自带的完整 Triton 稀疏回退链（MiniMaxM3SparseTritonImpl，bf16/fp8 KV）
在 SM120 上从未被尝试；官方 nvfp4 白名单把 SM120 放行反而导致直接崩溃。

本 patch 在两个选择器入口插入 SM120 降级逻辑（官方 fp8 链零改动复用）：

  A. common/sparse_attention.py::select_main_backend_and_impl_cls
     nvfp4 + SM120 且非 MSA 后端  -> 改按 "fp8" 选择（MiniMaxM3SparseBackend +
     MiniMaxM3SparseTritonImpl，page 128 与 compose 的 --block-size 128 一致）。
     KV scale 默认 1.0（官方未发布 M3 校准值；fp8_e4m3 动态范围 ~448 覆盖
     activation 幅值，精度待冒烟测试确认）。
  B. common/indexer.py::select_indexer_impl_cls
     nvfp4 + SM120 -> 改按 "fp8" 选择。topk_blocks=16 != MSA 要求的档位组合
     （use_msa 需 is_sm100，SM120 恒 False）-> MiniMaxM3IndexerTritonImpl，
     与主模型共享 topk_indices_buffer / block_table。
  C. 逃生开关：env M3_SM120_KEEP_NVFP4=1 保留官方 nvfp4 行为（用于复现崩溃）。

注意：indexer_kv_dtype 保持 "nvfp4" 时，model.py:1078 的
kv_k_scale=self._k_scale if self.use_nvfp4_kv else None 分支由
use_nvfp4_kv（get_kv_quant_mode(kv_cache_dtype)）单独决定，主 KV 走 fp8 后
该参数仅 nvfp4 主链消费——若冒烟发现 fp8 主链误收 nvfp4 scale，需追加 model.py
的对应条件 patch。

目标文件为镜像 /usr/local/lib/python3.12/dist-packages/vllm 的源文件，
经 patch -> commit 进 minimax-m31-sm120-port 镜像使用。

v1.1（部署实测修正）：
  - 首轮部署报错 `Failed to plan fmha_sm100: invalid argument`（启动 capture 阶段）：
    compose 用 nvfp4 时官方选择 MSA 类，v1 只降级了 cache dtype，未离开 MSA 类；
    MSAImpl(fp8) 的 should_prepare_decode_metadata 在 batch>=min_batch 时仍调
    C++ kv_mode3 plan -> 在 SM120 上 invalid argument（且 forward 期
    fmha_sm100 decode/run 同为 tcgen05 模板，运行时同样会崩）。
  - 修正：主选择器对 SM120 **完全退出 MSA**（use_msa=False）->
    MiniMaxM3SparseBackend + MiniMaxM3SparseTritonImpl（Triton 稀疏
    attend/decode，bf16 Q 直读、fp8 KV 反量化，纯 SM120 原生路径）。
  - indexer 侧（fp8 档）SM120 本来就不满足 use_msa（需 family(100)），
    自动落 Triton indexer，无需额外处理。
"""
import sys
import pathlib

BASE = pathlib.Path("/tmp/patchwork")

SPARSE_REL = "models/minimax_m3/common/sparse_attention.py"
INDEXER_REL = "models/minimax_m3/common/indexer.py"

MARKER = "# [m31-sm120-port]"

# ---------- A. sparse_attention.py ----------
SPARSE_OLD = """    use_msa = (
        current_platform.is_cuda()
        and (current_platform.is_device_capability_family(100)
             or current_platform.is_device_capability_family(120))
        and topk_blocks in (4, 8, 16, 32)
        and kv_cache_dtype != "fp8_e5m2"
    )"""

SPARSE_NEW = """    use_msa = (
        current_platform.is_cuda()
        and (current_platform.is_device_capability_family(100)
             or current_platform.is_device_capability_family(120))
        and topk_blocks in (4, 8, 16, 32)
        and kv_cache_dtype != "fp8_e5m2"
    )
    # [m31-sm120-port] SM120 (RTX PRO 6000 / consumer Blackwell) has no
    # TMEM/tcgen05. The entire MSA chain is SM100-class only: the fp8
    # fmha_sm100 C++ plan fails at startup ("Failed to plan fmha_sm100:
    # invalid argument") and the nvfp4 CUTLASS/q8kv4 kernels fail at run
    # ("unspecified launch failure"). Exit MSA entirely on SM120 and use
    # the official Triton sparse backend (bf16 Q, fp8 KV dequant), which is
    # SM120-native. Toggle: M3_SM120_KEEP_NVFP4=1 restores official routing.
    if (
        use_msa
        and current_platform.is_device_capability_family(120)
        and os.environ.get("M3_SM120_KEEP_NVFP4", "") != "1"
    ):
        logger.info_once(
            "[m31-sm120-port] SM120 GPU: MSA (fmha_sm100) requires SM100-class "
            "tcgen05 hardware; routing MiniMax M3 sparse attention to the "
            "Triton backend. Set M3_SM120_KEEP_NVFP4=1 to disable.",
        )
        use_msa = False"""

SPARSE_OLD2 = """    use_nvfp4 = get_kv_quant_mode(kv_cache_dtype).is_nvfp4
    if use_nvfp4 and kv_cache_dtype != "nvfp4":
        raise ValueError(
            "MiniMax M3 sparse attention supports only the 'nvfp4' NVFP4 KV "
            f"cache dtype, got {kv_cache_dtype}"
        )"""

SPARSE_NEW2 = """    use_nvfp4 = get_kv_quant_mode(kv_cache_dtype).is_nvfp4
    if use_nvfp4 and kv_cache_dtype != "nvfp4":
        raise ValueError(
            "MiniMax M3 sparse attention supports only the 'nvfp4' NVFP4 KV "
            f"cache dtype, got {kv_cache_dtype}"
        )
    # [m31-sm120-port] SM120 never selects MSA (patch above), so an nvfp4 KV
    # cache would hit "NVFP4 KV cache only with the SM100 MSA backend";
    # degrade the main KV to fp8_e4m3 + Triton. KV scales default to 1.0
    # (official default). Toggle: M3_SM120_KEEP_NVFP4=1.
    if (
        use_nvfp4
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and os.environ.get("M3_SM120_KEEP_NVFP4", "") != "1"
    ):
        logger.info_once(
            "[m31-sm120-port] SM120 GPU: degrading MiniMax M3 main KV cache "
            "nvfp4 -> fp8 (Triton sparse attend/decode).",
        )
        kv_cache_dtype = "fp8"
        use_nvfp4 = False"""

SPARSE_IMPORT_OLD = "import torch\n"
SPARSE_IMPORT_NEW = "import os\n\nimport torch\n"

# ---------- B. indexer.py ----------
INDEXER_OLD = """    is_sm100 = is_cuda and current_platform.is_device_capability_family(100)
    if indexer_kv_dtype == "nvfp4":
        if not ("""

INDEXER_NEW = """    is_sm100 = is_cuda and current_platform.is_device_capability_family(100)
    # [m31-sm120-port] nvfp4 indexers (cute src/sm100/*.py) are
    # tcgen05/TMEM-only and crash on SM120 even though the whitelist below
    # admits it. Degrade the index KV to fp8 + Triton indexer.
    # Toggle: M3_SM120_KEEP_NVFP4=1.
    if (
        indexer_kv_dtype == "nvfp4"
        and is_cuda
        and current_platform.is_device_capability_family(120)
        and os.environ.get("M3_SM120_KEEP_NVFP4", "") != "1"
    ):
        logger.info_once(
            "[m31-sm120-port] SM120 GPU: degrading MiniMax M3 indexer KV "
            "nvfp4 -> fp8 (Triton indexer). Set M3_SM120_KEEP_NVFP4=1 to disable.",
        )
        indexer_kv_dtype = "fp8"
    if indexer_kv_dtype == "nvfp4":
        if not ("""

INDEXER_IMPORT_OLD = "import torch\n"
INDEXER_IMPORT_NEW = "import os\n\nimport torch\n"

EDITS = [
    (SPARSE_REL, [
        (SPARSE_IMPORT_OLD, SPARSE_IMPORT_NEW),
        (SPARSE_OLD, SPARSE_NEW),
        (SPARSE_OLD2, SPARSE_NEW2),
    ]),
    (INDEXER_REL, [
        (INDEXER_IMPORT_OLD, INDEXER_IMPORT_NEW),
        (INDEXER_OLD, INDEXER_NEW),
    ]),
]


def main() -> int:
    rc = 0
    for rel, edits in EDITS:
        path = BASE / rel
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"FAIL {rel}: {exc}")
            return 1
        if MARKER in text:
            print(f"SKIP {rel}: already patched")
            continue
        for old, new in edits:
            count = text.count(old)
            if count != 1:
                print(f"FAIL {rel}: anchor count={count} for {old[:60]!r}")
                rc = 1
                break
            text = text.replace(old, new)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"OK   {rel}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
