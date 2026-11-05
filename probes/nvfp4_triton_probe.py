"""Standalone Triton NVFP4 dequant probe against the vLLM packed page layout.

Fills a real cache with the vLLM write op, dequantizes it both in torch (byte
semantics worked out from the fmha readers) and in Triton, and compares.
"""
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm import _custom_ops as ops  # noqa: E402


@triton.jit
def _e2m1_codes_to_f16(codes):
    mag = codes & 7
    bits = tl.where(
        mag == 0,
        0,
        (((mag >> 1) + 14) << 10) | tl.where(mag > 1, (mag & 1) << 9, 0),
    )
    bits = (bits | ((codes & 8) << 12)).to(tl.uint16)
    return bits.to(tl.float16, bitcast=True)


@triton.jit
def nvfp4_dequant_kv(cache_ptr, out_k, out_v, n_pos,
                     head_dim: tl.constexpr, BLOCK_N: tl.constexpr):
    off_n = tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, head_dim)
    nmask = off_n < n_pos
    # ---- K: slot 0 ----
    kc = tl.load(cache_ptr + off_n[None, :] * 64 + (off_d[:, None] // 2),
                 mask=nmask[None, :], other=0)
    kc = (kc >> ((off_d[:, None] & 1) * 4)) & 0xF
    k = _e2m1_codes_to_f16(kc)
    ksf = tl.load(cache_ptr + 8192 + off_n[None, :] * 8 + (off_d[:, None] // 16),
                  mask=nmask[None, :], other=0)
    ksf = ksf.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    k = (k * ksf).to(tl.float32)
    tl.store(out_k + off_d[:, None] * n_pos + off_n[None, :], k, mask=nmask[None, :])
    # ---- V: slot 1 ----
    vbase = cache_ptr + 9216
    vc = tl.load(vbase + off_n[:, None] * 64 + (off_d[None, :] // 2),
                 mask=nmask[:, None], other=0)
    vc = (vc >> ((off_d[None, :] & 1) * 4)) & 0xF
    v = _e2m1_codes_to_f16(vc)
    vsf_off = ((off_n[:, None] // 4) * 32 + (off_n[:, None] % 4)
               + 4 * (off_d[None, :] // 16))
    vsf = tl.load(vbase + 8192 + vsf_off, mask=nmask[:, None], other=0)
    vsf = vsf.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    v = (v * vsf).to(tl.float32)
    tl.store(out_v + off_n[:, None] * head_dim + off_d[None, :], v,
             mask=nmask[:, None])


dev = "cuda:0"
NH, NKV, NIQ, HD, BLK, NB = 8, 1, 2, 128, 128, 1
EPS = 1e-6
N = 100
torch.manual_seed(1)

qkv = torch.randn(N, (NH + 2 * NKV + NIQ + 1) * HD, dtype=torch.bfloat16, device=dev)
q_w = torch.zeros(HD, dtype=torch.bfloat16, device=dev)
positions = torch.zeros(N, dtype=torch.int64, device=dev)
slot = torch.arange(N, dtype=torch.int64, device=dev)
cache = torch.zeros(NB, 2 * NKV, BLK, 72, dtype=torch.uint8, device=dev)
icache = torch.zeros(NB, NIQ, BLK, HD, dtype=torch.float8_e4m3fn, device=dev)
cs = torch.zeros(1, 64, dtype=torch.bfloat16, device=dev)
cs[:, :32] = 1.0
one = torch.ones(1, dtype=torch.float32, device=dev)
ops.fused_minimax_m3_qknorm_rope_kv_insert(
    qkv, q_w, q_w, cs, positions, NH, NKV, 64, EPS,
    q_w, q_w, NIQ, slot, slot, cache, icache, BLK,
    None, None, "nvfp4", False, None, 1.0, one, one, True,
)

out_k = torch.zeros(HD, N, dtype=torch.float32, device=dev)
out_v = torch.zeros(N, HD, dtype=torch.float32, device=dev)
nvfp4_dequant_kv[(1,)](
    cache[0].reshape(-1).contiguous(), out_k, out_v, N,
    head_dim=HD, BLOCK_N=BLK,
)

# ---- torch reference from raw bytes ----
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)


def conv(codes):
    return E2M1[(codes & 7).long()] * torch.where((codes >> 3) != 0, -1.0, 1.0)


flat = cache[0].reshape(-1)
data = flat[:8192].reshape(128, 64)
ksf = flat[8192:9216].view(torch.float8_e4m3fn).float().reshape(128, 8)
k_ref = torch.stack([conv(data & 0xF), conv((data >> 4) & 0xF)], -1).reshape(128, 128)
k_ref = (k_ref * ksf.repeat_interleave(16, -1)).T  # [D, N]

vdata = flat[9216:9216 + 8192].reshape(128, 64)
vsf = flat[9216 + 8192:9216 + 9216].view(torch.float8_e4m3fn).float()
t = torch.arange(128, device=dev)
v_off = (t // 4) * 32 + (t % 4)
v_ref = torch.empty(128, 128, device=dev)
for b in range(8):
    v_ref[:, b * 16:(b + 1) * 16] = vsf[v_off + 4 * b].unsqueeze(-1)
v_ref = torch.stack([conv(vdata & 0xF), conv((vdata >> 4) & 0xF)], -1).reshape(128, 128) * v_ref

ek = (out_k - k_ref[:, :N]).abs()
ev = (out_v - v_ref[:N]).abs()
print("K max abs err", ek.max().item(), "mean", ek.mean().item())
print("V max abs err", ev.max().item(), "mean", ev.mean().item())

# against the pre-quantization K/V (quantization error only)
qk = qkv.view(N, -1, HD)
k_true = qk[:, NH, :].float()
k_true = k_true / torch.sqrt(k_true.pow(2).mean(-1, keepdim=True) + EPS)
print("K vs pre-quant rel_rmse",
      ((out_k - k_true.T).pow(2).mean().sqrt() / k_true.pow(2).mean().sqrt()).item())
print("V vs pre-quant rel_rmse",
      ((out_v - qk[:, NH + NKV, :].float()).pow(2).mean().sqrt()
       / qk[:, NH + NKV, :].float().pow(2).mean().sqrt()).item())
