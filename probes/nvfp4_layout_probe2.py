"""Verify the vllm NVFP4 slot scale layout the fmha readers use:
K: scale byte at token*8 + d_block
V: scale byte at (token//4)*32 + token%4 + 4*d_block
Data planes: token*64 + byte_in_head, low nibble = even dim.
"""
import sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm import _custom_ops as ops  # noqa: E402

dev = "cuda:0"
NH, NKV, NIQ, HD, BLK, NB = 8, 1, 2, 128, 128, 1
EPS = 1e-6
N = 5
DIM = 21  # d_block 1

qkv = torch.zeros(N, (NH + 2 * NKV + NIQ + 1) * HD, dtype=torch.bfloat16, device=dev)
k_off, v_off = NH * HD, (NH + NKV) * HD
qkv[:, k_off + DIM] = 1.0
qkv[:, v_off + DIM] = 1.0
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

for side, name in ((0, "K"), (1, "V")):
    flat = cache[0, side].reshape(-1)
    data = flat[:8192].reshape(128, 64)
    sf = flat[8192:9216].view(torch.float8_e4m3fn).float()
    tok_rows = (data != 0).any(-1).nonzero().flatten().tolist()
    byte_cols = (data != 0).any(0).nonzero().flatten().tolist()
    sf_idx = (sf > 0.01).nonzero().flatten().tolist()
    print(f"{name}: data rows {tok_rows}, byte cols {byte_cols}, "
          f"scale bytes {sf_idx} values {[round(float(x),3) for x in sf[sf_idx].tolist()]}")
    for t in range(N):
        exp = t * 8 + 1 if name == "K" else (t // 4) * 32 + (t % 4) + 4
        print(f"   token {t}: expected scale offset {exp}, actual value "
              f"{float(sf[exp]):.3f}")
