"""Kernel-level check: Triton nvfp4 sparse attend vs a torch reference.

Builds a random packed NVFP4 cache in the official page layout, then compares
``minimax_m3_sparse_attn`` (prefill) and ``minimax_m3_sparse_attn_decode``
against a dequant reference that reads the same bytes.
"""

import torch

from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)

torch.manual_seed(0)
DEV = "cuda"
H = 8
HKV = 2
HD = 128
SLOTS = 2 * HKV
PAGES = 4
SEQ = 8
GQA = H // HKV
SM_SCALE = 1.0 / (HD**0.5)
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=DEV
)

# uint8 [pages, 2*Hkv, 128, 72]: 64 B packed codes + 8 B E4M3 block scales.
cache = torch.randint(0, 256, (PAGES, SLOTS, SPARSE_BLOCK_SIZE, 72), dtype=torch.uint8, device=DEV)
# Slot bytes: 8192 E2M1 then 1024 E4M3. Random E4M3 can be NaN; pin it to 1.0.
_flat = cache.view(PAGES * SLOTS, 128 * 72)
_scale_plane = _flat[:, 128 * 64 :]
_scale_plane[(_scale_plane & 0x7F) == 0x7F] = 0x38


def dequant(page: int, head: int, is_v: bool) -> torch.Tensor:
    """[128, 128] float32 dequantized K or V for one KV head."""
    slot = cache.view(PAGES * SLOTS, 128 * 72)[page * SLOTS + 2 * head + int(is_v)]
    codes = slot[: 128 * 64].view(SPARSE_BLOCK_SIZE, 64)
    sf = slot[128 * 64 :].view(torch.float8_e4m3fn).to(torch.float32)  # [1024]
    pos = torch.arange(SPARSE_BLOCK_SIZE, device=DEV)[:, None]
    dim = torch.arange(HD, device=DEV)[None, :]
    byte = codes[:, (torch.arange(HD, device=DEV) // 2)].to(torch.int16)
    nib = (byte >> (4 * (dim % 2))) & 0xF
    val = E2M1[(nib & 7).long()] * torch.where(nib >= 8, -1.0, 1.0)
    if is_v:
        # V scales interleave four tokens: (t//4)*32 + t%4 + 4*(d//16).
        sf_off = (pos // 4) * 32 + pos % 4 + 4 * (dim // 16)
    else:
        sf_off = pos * 8 + dim // 16
    return val * sf[sf_off]  # [128, 128]


def reference(
    q: torch.Tensor, page: int, seq_len: int, qpos: torch.Tensor, gscale: float = 1.0
) -> torch.Tensor:
    """Causal attention over positions [0, seq_len) of one page."""
    out = torch.zeros_like(q, dtype=torch.float32)
    qf = q.float()
    for h in range(HKV):
        k = dequant(page, h, False)[:seq_len] * gscale  # [seq, D]
        v = dequant(page, h, True)[:seq_len] * gscale
        qh = qf[:, h * GQA : (h + 1) * GQA]  # [Q, GQA, D]
        scores = torch.einsum("qgd,kd->qgk", qh, k) * SM_SCALE
        pos = torch.arange(seq_len, device=DEV)
        scores = scores.masked_fill(
            pos[None, None, :] > qpos[:, None, None], float("-inf")
        )
        p = torch.softmax(scores, dim=-1)
        out[:, h * GQA : (h + 1) * GQA] = torch.einsum("qgk,kd->qgd", p, v)
    return out


def rel_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt())


page = 1
block_table = torch.tensor([[page]], dtype=torch.int32, device=DEV)
page_zero = torch.tensor([0], dtype=torch.int32, device=DEV)
k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEV)
v_scale = torch.tensor(1.0, dtype=torch.float32, device=DEV)

# ---- prefill: 8 query tokens, one request, block 0 selected ----
q = torch.randn(SEQ, H, HD, dtype=torch.bfloat16, device=DEV)
topk = torch.full((HKV, SEQ, 16), -1, dtype=torch.int32, device=DEV)
topk[:, :, 0] = 0
out = torch.empty_like(q)
minimax_m3_sparse_attn(
    q,
    cache,
    topk,
    block_table,
    torch.tensor([0, SEQ], dtype=torch.int32, device=DEV),
    torch.tensor([SEQ], dtype=torch.int32, device=DEV),
    torch.tensor([0], dtype=torch.int32, device=DEV),
    SEQ,
    HKV,
    SM_SCALE,
    out,
    k_scale=k_scale,
    v_scale=v_scale,
)
ref = reference(q, page, SEQ, torch.arange(SEQ, device=DEV))
print(f"prefill nvfp4 rel_rmse = {rel_rmse(out, ref):.3e}")

# ---- decode: one query token at seq_len 8 ----
qd = torch.randn(1, H, HD, dtype=torch.bfloat16, device=DEV)
topk_d = torch.full((HKV, 1, 16), -1, dtype=torch.int32, device=DEV)
topk_d[:, :, 0] = 0
outd = torch.empty_like(qd)
minimax_m3_sparse_attn_decode(
    qd,
    cache,
    topk_d,
    block_table,
    torch.tensor([SEQ], dtype=torch.int32, device=DEV),
    HKV,
    SM_SCALE,
    outd,
    1,
    k_scale=k_scale,
    v_scale=v_scale,
)
refd = reference(qd, page, SEQ, torch.tensor([SEQ - 1], device=DEV))
print(f"decode  nvfp4 rel_rmse = {rel_rmse(outd, refd):.3e}")

# ---- scalar global scale must be folded in ----
k2 = torch.tensor(2.0, dtype=torch.float32, device=DEV)
out2 = torch.empty_like(q)
minimax_m3_sparse_attn(
    q,
    cache,
    topk,
    block_table,
    torch.tensor([0, SEQ], dtype=torch.int32, device=DEV),
    torch.tensor([SEQ], dtype=torch.int32, device=DEV),
    torch.tensor([0], dtype=torch.int32, device=DEV),
    SEQ,
    HKV,
    SM_SCALE,
    out2,
    k_scale=k2,
    v_scale=k2,
)
ref2 = reference(q, page, SEQ, torch.arange(SEQ, device=DEV), gscale=2.0)
print(f"global  x2    rel_rmse = {rel_rmse(out2, ref2):.3e}")
