"""Probe: does Triton lower tl.dot_scaled to native SM120 block-scaled MMA?

Covers MXFP4 (e2m1 x e2m1, e8m0, group 32) and the Q8KV4 mix
(e4m3 x e2m1, e8m0, group 32).  Also tries NVFP4 (e4m3 scales, group 16).
Dumps PTX and checks correctness against a torch reference.
"""
import os
import sys

import torch
import triton
import triton.language as tl

DEV = "cuda:0"
torch.manual_seed(0)


@triton.jit
def scaled_mm(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr,
              M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              GROUP: tl.constexpr, SCALE_E4M3: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K // 2)
    rk8 = tl.arange(0, K // GROUP)
    a = tl.load(a_ptr + rm[:, None] * (K // 2) + rk[None, :])          # e2m1 packed
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])                 # e2m1 packed along K
    if SCALE_E4M3:
        sa = tl.load(sa_ptr + rm[:, None] * (K // GROUP) + rk8[None, :]).to(
            tl.float8e4nv, bitcast=True)
        sb = tl.load(sb_ptr + rn[:, None] * (K // GROUP) + rk8[None, :]).to(
            tl.float8e4nv, bitcast=True)
    else:
        sa = tl.load(sa_ptr + rm[:, None] * (K // GROUP) + rk8[None, :])
        sb = tl.load(sb_ptr + rn[:, None] * (K // GROUP) + rk8[None, :])
    c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", out_dtype=tl.float32)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


@triton.jit
def scaled_mm_q8kv4(q_ptr, b_ptr, sq_ptr, sb_ptr, c_ptr,
                    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    GROUP: tl.constexpr, SCALE_E4M3: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K // 2)
    rk8 = tl.arange(0, K // GROUP)
    q = tl.load(q_ptr + rm[:, None] * K + tl.arange(0, K)[None, :])     # e4m3
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])                  # e2m1 packed
    if SCALE_E4M3:
        sq = tl.load(sq_ptr + rm[:, None] * (K // GROUP) + rk8[None, :]).to(
            tl.float8e4nv, bitcast=True)
        sb = tl.load(sb_ptr + rn[:, None] * (K // GROUP) + rk8[None, :]).to(
            tl.float8e4nv, bitcast=True)
    else:
        sq = tl.load(sq_ptr + rm[:, None] * (K // GROUP) + rk8[None, :])
        sb = tl.load(sb_ptr + rn[:, None] * (K // GROUP) + rk8[None, :])
    c = tl.dot_scaled(q, sq, "e4m3", b, sb, "e2m1", out_dtype=tl.float32)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


@triton.jit
def scaled_mm_fp8(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  GROUP: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk8 = tl.arange(0, K // GROUP)
    a = tl.load(a_ptr + rm[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N + rn[None, :])
    sa = tl.load(sa_ptr + rm[:, None] * (K // GROUP) + rk8[None, :])
    sb = tl.load(sb_ptr + rn[:, None] * (K // GROUP) + rk8[None, :])
    c = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", out_dtype=tl.float32)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)


def deq_e2m1(packed, K):
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    nib = torch.stack([lo, hi], -1).reshape(*packed.shape[:-1], K).long()
    return E2M1[nib & 7] * torch.where(nib >= 8, -1.0, 1.0)


def e8m0_to_float(u):
    return torch.pow(2.0, u.float() - 127.0)


def e4m3_scale_to_float(u):
    return u.view(torch.float8_e4m3fn).float()


def check(tag, c, ref):
    err = (c - ref).abs()
    rel = err.max() / ref.abs().max().clamp_min(1e-9)
    print(f"  {tag}: max_abs={err.max().item():.6g} rel={rel.item():.3g} "
          f"finite={torch.isfinite(c).all().item()}")


def find_ptx():
    import glob
    paths = sorted(glob.glob(os.path.expanduser("/tmp/triton_dump/**/*.ptx"), recursive=True))
    hits = {}
    for p in paths:
        txt = open(p).read()
        lines = sorted({ln.strip() for ln in txt.splitlines()
                        if "mma.sync" in ln and "aligned" in ln})
        hits[p] = (txt.count("mxf4nvf4"), txt.count("mxf8f6f4"),
                   txt.count("tcgen05"), lines)
    return hits


def main():
    M = N = 64
    K = 128
    for group, sc_e4m3 in ((32, False), (16, True)):
        tag = f"e2m1x e2m1 group={group} scale={'e4m3' if sc_e4m3 else 'e8m0'}"
        print(tag)
        a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=DEV)
        b = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8, device=DEV)
        if sc_e4m3:
            sa = torch.randint(120, 132, (M, K // group), dtype=torch.uint8, device=DEV)
            sb = torch.randint(120, 132, (N, K // group), dtype=torch.uint8, device=DEV)
            saf = e4m3_scale_to_float(sa)
            sbf = e4m3_scale_to_float(sb)
        else:
            sa = torch.randint(125, 130, (M, K // group), dtype=torch.uint8, device=DEV)
            sb = torch.randint(125, 130, (N, K // group), dtype=torch.uint8, device=DEV)
            saf = e8m0_to_float(sa)
            sbf = e8m0_to_float(sb)
        c = torch.zeros(M, N, dtype=torch.float32, device=DEV)
        try:
            scaled_mm[(1,)](a, b, sa, sb, c, M=M, N=N, K=K, GROUP=group,
                            SCALE_E4M3=sc_e4m3)
            torch.cuda.synchronize()
            av = deq_e2m1(a, K) * saf.repeat_interleave(group, -1)
            bv = deq_e2m1(b.T.contiguous(), K).T  # [K, N]
            bv = bv * sbf.repeat_interleave(group, -1).T
            check(tag, c, av @ bv)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:400]}")

    for group, sc_e4m3 in ((32, False), (16, True)):
        tag = f"e4m3 x e2m1 group={group} scale={'e4m3' if sc_e4m3 else 'e8m0'}"
        print(tag)
        q = (torch.randn(M, K, device=DEV) * 0.5).to(torch.float8_e4m3fn)
        b = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8, device=DEV)
        if sc_e4m3:
            sq = torch.randint(120, 132, (M, K // group), dtype=torch.uint8, device=DEV)
            sb = torch.randint(120, 132, (N, K // group), dtype=torch.uint8, device=DEV)
            sqf = e4m3_scale_to_float(sq)
            sbf = e4m3_scale_to_float(sb)
        else:
            sq = torch.randint(125, 130, (M, K // group), dtype=torch.uint8, device=DEV)
            sb = torch.randint(125, 130, (N, K // group), dtype=torch.uint8, device=DEV)
            sqf = e8m0_to_float(sq)
            sbf = e8m0_to_float(sb)
        c = torch.zeros(M, N, dtype=torch.float32, device=DEV)
        try:
            scaled_mm_q8kv4[(1,)](q, b, sq, sb, c, M=M, N=N, K=K, GROUP=group,
                                  SCALE_E4M3=sc_e4m3)
            torch.cuda.synchronize()
            qv = q.float() * sqf.repeat_interleave(group, -1)
            bv = deq_e2m1(b.T.contiguous(), K).T
            bv = bv * sbf.repeat_interleave(group, -1).T
            check(tag, c, qv @ bv)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:400]}")

    print("e4m3 x e4m3 group=32 scale=e8m0")
    q = (torch.randn(M, K, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(K, N, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    sq = torch.randint(125, 130, (M, K // 32), dtype=torch.uint8, device=DEV)
    sb = torch.randint(125, 130, (N, K // 32), dtype=torch.uint8, device=DEV)
    c = torch.zeros(M, N, dtype=torch.float32, device=DEV)
    try:
        scaled_mm_fp8[(1,)](q, b, sq, sb, c, M=M, N=N, K=K, GROUP=32)
        torch.cuda.synchronize()
        qv = q.float() * e8m0_to_float(sq).repeat_interleave(32, -1)
        bv = b.float() * e8m0_to_float(sb).repeat_interleave(32, -1).T
        check("e4m3 x e4m3", c, qv @ bv)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:400]}")

    print("PTX mma lines:")
    for p, (a, b_, t, lines) in find_ptx().items():
        print(f"  {os.path.basename(os.path.dirname(p))[:8]} "
              f"mxf4nvf4={a} mxf8f6f4={b_} tcgen05={t}")
        for ln in lines:
            print("    ", ln[:150])


if __name__ == "__main__":
    sys.exit(main())
