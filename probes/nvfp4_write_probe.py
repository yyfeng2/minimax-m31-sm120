"""SM120 probe: does fused_minimax_m3_qknorm_rope_kv_insert support kv_cache_dtype=nvfp4?

Writes a known K/V pattern through the op, then dequantizes the packed 72-byte
head slots with both nibble orders and reports the error against the reference
(normed + partial-RoPE at position 0, where RoPE is identity).
"""
import math
import sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm import _custom_ops as ops  # noqa: E402

torch.manual_seed(0)
dev = "cuda:0"
N, NH, NKV, NIQ, HD = 4, 8, 1, 2, 128
BLK, NB = 128, 2
EPS = 1e-6

print("cc =", torch.cuda.get_device_capability())

qkv = torch.randn(N, (NH + 2 * NKV + NIQ + 1) * HD, dtype=torch.bfloat16, device=dev)
qkv_snapshot = qkv.clone()
q_w = torch.zeros(HD, dtype=torch.bfloat16, device=dev)
k_w = torch.zeros(HD, dtype=torch.bfloat16, device=dev)
positions = torch.zeros(N, dtype=torch.int64, device=dev)
slot_mapping = torch.arange(N, dtype=torch.int64, device=dev)
kv_cache = torch.zeros(NB, 2 * NKV, BLK, 72, dtype=torch.uint8, device=dev)
iq_w = torch.zeros(HD, dtype=torch.bfloat16, device=dev)
index_cache = torch.zeros(NB, NIQ, BLK, HD, dtype=torch.float8_e4m3fn, device=dev)
index_slot = torch.arange(N, dtype=torch.int64, device=dev)
k_scale = torch.ones(1, dtype=torch.float32, device=dev)
v_scale = torch.ones(1, dtype=torch.float32, device=dev)

cos_sin = torch.zeros(1, 64, dtype=torch.float32, device=dev)
cos_sin[:, :32] = 1.0  # cos = 1, sin = 0 at position 0

for cs_dtype in (torch.float32, torch.bfloat16):
    try:
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, q_w, k_w, cos_sin.to(cs_dtype), positions, NH, NKV, 64, EPS,
            iq_w, iq_w, NIQ, slot_mapping, index_slot, kv_cache, index_cache, BLK,
            None, None, "nvfp4", False, None, 1.0, k_scale, v_scale, True,
        )
        print(f"OP OK with cos_sin_cache dtype {cs_dtype}")
        break
    except Exception as exc:  # noqa: BLE001
        print(f"OP FAILED with cos_sin_cache dtype {cs_dtype}: {type(exc).__name__}: {exc}")
        if cs_dtype is torch.bfloat16:
            raise SystemExit(1)

print("cache nonzero bytes:", int((kv_cache != 0).sum()))

# ---- reference K/V (pos 0: RoPE identity; Gemma RMSNorm with 1+w, w=0) ----
qk = qkv_snapshot.view(N, -1, HD)
k_ref = qk[:, NH : NH + NKV].float()
v_ref = qk[:, NH + NKV : NH + 2 * NKV].float()
k_ref = k_ref / torch.sqrt(k_ref.pow(2).mean(-1, keepdim=True) + EPS)
v_ref = v_ref / torch.sqrt(v_ref.pow(2).mean(-1, keepdim=True) + EPS)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant(slot, low_nibble_first: bool):
    flat = slot.reshape(-1)  # 9216 bytes: [128 x 64 data][128 x 8 scales]
    data = flat[:8192].reshape(128, 64).contiguous()
    sf = flat[8192:9216].reshape(128, 8).contiguous().view(torch.float8_e4m3fn).float()
    lo = (data & 0x0F).long()
    hi = ((data >> 4) & 0x0F).long()

    def conv(codes):
        mag = codes & 7
        sign = torch.where((codes >> 3) != 0, -1.0, 1.0)
        return E2M1.to(codes.device)[mag] * sign

    a, b = conv(lo), conv(hi)
    if not low_nibble_first:
        a, b = b, a
    out = torch.stack([a, b], dim=-1).reshape(128, 128)  # element 2i, 2i+1
    sf = sf.repeat_interleave(16, dim=-1)  # one E4M3 scale per 16 elements
    return out * sf


for name, ref, slot_idx in (("K", k_ref, 0), ("V", v_ref, 1)):
    slot = kv_cache[0, slot_idx]  # [128, 72]
    for low_first in (True, False):
        got = dequant(slot, low_first)[:N]
        err = (got - ref).abs()
        rel = err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
        print(f"{name} low_nibble_first={low_first}: rel_rmse={rel.item():.4f} "
              f"max_abs={err.max().item():.4f}")
