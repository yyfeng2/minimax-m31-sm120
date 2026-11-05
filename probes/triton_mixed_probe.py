"""Single-case PTX check: e4m3 x e2m1 dot_scaled lowering on SM120."""
import torch
import triton
import triton.language as tl


@triton.jit
def mm(q_ptr, b_ptr, sq_ptr, sb_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K // 2)
    rk8 = tl.arange(0, K // 32)
    q = tl.load(q_ptr + rm[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    sq = tl.load(sq_ptr + rm[:, None] * (K // 32) + rk8[None, :])
    sb = tl.load(sb_ptr + rn[:, None] * (K // 32) + rk8[None, :])
    c = tl.dot_scaled(q, sq, "e4m3", b, sb, "e2m1", out_dtype=tl.float32)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


dev = "cuda:0"
M = N = 64
K = 128
q = (torch.randn(M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
b = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8, device=dev)
sq = torch.full((M, K // 32), 127, dtype=torch.uint8, device=dev)
sb = torch.full((N, K // 32), 127, dtype=torch.uint8, device=dev)
c = torch.zeros(M, N, device=dev)
h = mm[(1,)](q, b, sq, sb, c, M=M, N=N, K=K)
torch.cuda.synchronize()
ptx = h.asm["ptx"]
lines = sorted({l.strip() for l in ptx.splitlines() if "mma.sync" in l or "wgmma" in l})
for l in lines[:12]:
    print(l[:140])
print("total mma lines", len(lines), "bf16 cvt", ptx.count("cvt.rn.bf16"),
      "f16 cvt", ptx.count("cvt.rn.f16"), "e2m1 cvt", ptx.count("e2m1"))
