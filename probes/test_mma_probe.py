"""Validate the SM120 MMA fragment layouts in isolation."""
import os
import sys

import torch
from torch.utils.cpp_extension import load

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(
    name="probe_mma",
    sources=[os.path.join(HERE, "probe_mma.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
    verbose=False,
)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)


def deq_codes(packed, scales, nrow, nk):
    """packed [nrow, nk/2] uint8; scales [nrow, nk//16] e4m3 -> values [nrow, nk]."""
    lo = (packed & 0xF).long()
    hi = ((packed >> 4) & 0xF).long()
    v = torch.stack([E2M1[lo & 7] * torch.where(lo >= 8, -1.0, 1.0),
                     E2M1[hi & 7] * torch.where(hi >= 8, -1.0, 1.0)], dim=-1)
    v = v.reshape(nrow, nk // 16, 16)
    return (v * scales.float().unsqueeze(-1)).reshape(nrow, nk)


def main():
    torch.manual_seed(0)
    # ---------------- QK probe
    a_codes = torch.randint(0, 256, (16, 32), device=DEV, dtype=torch.uint8)
    a_sf = (0.02 + 0.2 * torch.rand(16, 4, device=DEV)).to(torch.float8_e4m3fn)
    b_codes = torch.randint(0, 256, (128, 32), device=DEV, dtype=torch.uint8)
    b_sf = (0.02 + 0.2 * torch.rand(128, 4, device=DEV)).to(torch.float8_e4m3fn)
    out = torch.zeros(16, 128, device=DEV, dtype=torch.float32)
    ext.qk_probe(a_codes.data_ptr(), a_sf.data_ptr(), b_codes.data_ptr(),
                 b_sf.data_ptr(), out.data_ptr(),
                 torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    A = deq_codes(a_codes, a_sf, 16, 64).double()
    B = deq_codes(b_codes, b_sf, 128, 64).double()
    ref = A @ B.T
    err = (out.double() - ref).abs()
    print("QK probe: max|err| = %.3e   |ref| max = %.3f   rel = %.3e"
          % (err.max().item(), ref.abs().max().item(),
             (err / ref.abs().clamp_min(1e-3)).max().item()))
    print("  kernel[0,:8] =", [round(x, 4) for x in out[0, :8].tolist()])
    print("  ref   [0,:8] =", [round(x, 4) for x in ref[0, :8].tolist()])
    per_row = err.max(dim=1).values
    print("  err per row:", [round(x, 3) for x in per_row.tolist()])
    colgrp = err.reshape(16, 16, 8).amax(dim=(0, 1))
    print("  err per col group (8 cols):", [round(x, 3) for x in colgrp.tolist()])
    per_col = err.amax(dim=0)
    print("  err per col[:40]:", [round(x, 3) for x in per_col[:40].tolist()])

    # ---------------- PV probe
    p = (torch.rand(16, 32, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    psf = torch.full((16,), 127, device=DEV, dtype=torch.uint8)  # 2^0
    v = (torch.rand(32, 32, device=DEV)).to(torch.float8_e4m3fn)  # [dim][key]
    outp = torch.zeros(16, 32, device=DEV, dtype=torch.float32)
    ext.pv_probe(p.view(torch.uint8).data_ptr(), psf.data_ptr(),
                 v.view(torch.uint8).data_ptr(), outp.data_ptr(),
                 torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    refp = p.double() @ v.double().T  # [16 heads][32 dims]
    errp = (outp.double() - refp).abs()
    print("PV probe: max|err| = %.3e   |ref| max = %.3f   rel = %.3e"
          % (errp.max().item(), refp.abs().max().item(),
             (errp / refp.abs().clamp_min(1e-3)).max().item()))
    print("  kernel[0,:6] =", [round(x, 4) for x in outp[0, :6].tolist()])
    print("  ref   [0,:6] =", [round(x, 4) for x in refp[0, :6].tolist()])


if __name__ == "__main__":
    main()
