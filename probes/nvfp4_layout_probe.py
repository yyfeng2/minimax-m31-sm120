"""Deterministic NVFP4 slot layout probe: one nonzero k element at a known dim.

Reveals (a) which nibble holds the even element, (b) which of the 8 E4M3 scale
bytes covers which 16-element group, (c) whether the op's K slot is cache[:,0]
and V slot cache[:,1].
"""
import sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm import _custom_ops as ops  # noqa: E402

torch.manual_seed(0)
dev = "cuda:0"
NH, NKV, NIQ, HD, BLK, NB = 8, 1, 2, 128, 128, 1
EPS = 1e-6
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def run(onehot_dim: int):
    N = 1
    qkv = torch.zeros(N, (NH + 2 * NKV + NIQ + 1) * HD, dtype=torch.bfloat16, device=dev)
    k_off = NH * HD
    v_off = (NH + NKV) * HD
    qkv[0, k_off + onehot_dim] = 1.0
    qkv[0, v_off + onehot_dim] = 1.0
    q_w = torch.zeros(HD, dtype=torch.bfloat16, device=dev)
    iq_w = torch.zeros(HD, dtype=torch.bfloat16, device=dev)
    positions = torch.zeros(N, dtype=torch.int64, device=dev)
    slot = torch.zeros(N, dtype=torch.int64, device=dev)
    cache = torch.zeros(NB, 2 * NKV, BLK, 72, dtype=torch.uint8, device=dev)
    icache = torch.zeros(NB, NIQ, BLK, HD, dtype=torch.float8_e4m3fn, device=dev)
    cs = torch.zeros(1, 64, dtype=torch.bfloat16, device=dev)
    cs[:, :32] = 1.0
    one = torch.ones(1, dtype=torch.float32, device=dev)
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, q_w, q_w, cs, positions, NH, NKV, 64, EPS,
        iq_w, iq_w, NIQ, slot, slot, cache, icache, BLK,
        None, None, "nvfp4", False, None, 1.0, one, one, True,
    )
    for side, name in ((0, "K"), (1, "V")):
        flat = cache[0, side].reshape(-1)
        data = flat[:8192].reshape(128, 64)
        sf = flat[8192:9216].reshape(128, 8).view(torch.float8_e4m3fn).float()
        nz = (data[0] != 0).nonzero().flatten().tolist()
        sf_nz = (sf[0] != 0).nonzero().flatten().tolist()
        vals = []
        for b in nz[:4]:
            byte = int(data[0, b])
            vals.append((b, hex(byte), byte & 0xF, (byte >> 4) & 0xF))
        print(f"dim={onehot_dim} {name}: nonzero data bytes {nz} -> "
              f"(byte, hex, lo, hi) {vals}")
        print(f"   scale nonzero idx {sf_nz} values {[round(float(x),3) for x in sf[0][sf_nz].tolist()]}")


for d in (5, 21, 127):
    run(d)
