"""Numerical check for the SM120 native FP4/FP8 sparse decode kernel.

Builds a packed NVFP4 cache exactly as vLLM lays it out
(per head slot: 8192 B E2M1 data plane + 1024 B E4M3 block-scale plane,
K sf = pos*8 + db, V sf = (pos//4)*32 + 4*db + pos%4), dequantizes it in
torch, and compares both the raw QK scores and the final output.
"""

import math
import sys

import torch

sys.path.insert(0, r"/tmp/msa_sm120")
from msa_nvfp4 import msa_decode_nvfp4  # noqa: E402

DEV = "cuda"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)
THRESH = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)


def e2m1_round(x):
    # Round-to-nearest-even on the E2M1 grid (matches hardware
    # cvt.rn.satfinite.e2m1x2.f32): exact ties go to the even code.
    a = x.abs()
    idx = torch.zeros_like(a, dtype=torch.long)
    tie = torch.zeros_like(a, dtype=torch.bool)
    for t in THRESH:
        idx += (a > t).long()
        tie |= (a == t)
    idx += (tie & (idx % 2 == 1)).long()
    val = E2M1[idx] * torch.sign(x)
    code = idx | torch.where(x < 0, torch.tensor(8, device=DEV), torch.tensor(0, device=DEV))
    return val, code


def quant_q(q, block=16):
    t, h, d = q.shape
    x = q.reshape(t, h, d // block, block)
    amax = x.abs().amax(-1)
    scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
    scale = scale.to(torch.float8_e4m3fn).float()
    val, code = e2m1_round(x / scale.unsqueeze(-1))
    return code.reshape(t, h, d), scale.reshape(t, h, d // block),         (val * scale.unsqueeze(-1)).reshape(t, h, d)


def build_cache(pages, hkv):
    """Returns flat slot tensor [pages, 2*hkv, 9216] plus dequantized K/V."""
    nslot = 2 * hkv
    codes = torch.randint(0, 16, (pages, nslot, 128, 128), device=DEV, dtype=torch.int32)
    data = (codes[..., 0::2] & 0xF) | ((codes[..., 1::2] & 0xF) << 4)  # [., 128, 64]
    sf_k = (0.02 + 0.2 * torch.rand(pages, nslot, 128, 8, device=DEV)).to(torch.float8_e4m3fn)
    sf_v = (0.02 + 0.2 * torch.rand(pages, nslot, 128, 8, device=DEV)).to(torch.float8_e4m3fn)

    flat = torch.zeros(pages, nslot, 9216, device=DEV, dtype=torch.uint8)
    flat[..., :8192] = data.reshape(pages, nslot, 8192)

    pos = torch.arange(128, device=DEV).view(128, 1)
    db = torch.arange(8, device=DEV).view(1, 8)
    idx_k = (pos * 8 + db).reshape(1, 1, 1024).expand(pages, hkv, 1024)
    idx_v = ((pos // 4) * 32 + 4 * db + (pos % 4)).reshape(1, 1, 1024).expand(pages, hkv, 1024)
    flat[:, 0::2].narrow(-1, 8192, 1024).scatter_(
        -1, idx_k, sf_k[:, 0::2].view(torch.uint8).reshape(pages, hkv, 1024))
    flat[:, 1::2].narrow(-1, 8192, 1024).scatter_(
        -1, idx_v, sf_v[:, 1::2].view(torch.uint8).reshape(pages, hkv, 1024))

    sign = torch.where(codes >= 8, -1.0, 1.0)
    vals = E2M1[codes & 7] * sign  # [pages, nslot, 128, 128]
    k = (vals[:, 0::2].reshape(pages, hkv, 128, 8, 16) * sf_k[:, 0::2].float().unsqueeze(-1))
    v = (vals[:, 1::2].reshape(pages, hkv, 128, 8, 16) * sf_v[:, 1::2].float().unsqueeze(-1))
    return flat, k.reshape(pages, hkv, 128, 128), v.reshape(pages, hkv, 128, 128)


def main():
    torch.manual_seed(0)
    hkv, hq, hd = 4, 64, 128
    pages, topk_cap = 24, 8
    num_reqs, dql = 2, 1
    T = num_reqs * dql
    scale = 1.0 / math.sqrt(hd)

    flat, k, v = build_cache(pages, hkv)
    kv_cache = flat.view(pages, 2 * hkv, 128, 72)

    seq_lens = torch.tensor([301, 129], device=DEV, dtype=torch.int32)
    q = (torch.randn(T, hq, hd, device=DEV) * 0.5).to(torch.bfloat16)

    topk = torch.full((hkv, T, topk_cap), -1, device=DEV, dtype=torch.int32)
    nblk = [(int(s) + 127) // 128 for s in seq_lens.tolist()]
    for t in range(T):
        nb = nblk[t // dql]
        ids = torch.randperm(nb, device=DEV)[: min(nb, topk_cap)]
        topk[:, t, : ids.numel()] = ids.to(torch.int32).sort().values

    block_table = torch.arange(pages, device=DEV, dtype=torch.int32).reshape(num_reqs, -1)
    block_table = block_table[:, : max(nblk) + 1].contiguous()

    out = torch.zeros(T, hq, hd, device=DEV, dtype=torch.bfloat16)
    dbg = torch.zeros(hkv * 16, 128, device=DEV, dtype=torch.float32)

    stage = torch.zeros(58240, device=DEV, dtype=torch.uint8)  # 31104 + p/ps + 3 page dumps
    msa_decode_nvfp4(q, kv_cache, topk, block_table, seq_lens, out, dql, scale,
                     None, None, dbg, stage)
    torch.cuda.synchronize()

    # ---- staging checks (token 0, kv head 0, first selected page)
    qcode, qsf, qdeq = quant_q(q.float())
    st = stage.cpu()
    # Q codes: [16][80] with 64 code bytes per row
    q_bytes = torch.zeros(16, 80, dtype=torch.uint8)
    q_bytes[:, :64] = (qcode[0, 0:16, 0::2] & 0xF).to(torch.uint8) |                       ((qcode[0, 0:16, 1::2] & 0xF).to(torch.uint8) << 4)
    gq = st[:1280].reshape(16, 80)
    print("stage Q  : mismatch bytes =", int((gq[:, :64] != q_bytes[:, :64]).sum()),
          " pad nonzero =", int(gq[:, 64:].ne(0).sum()))
    gqs = st[1280:1408].reshape(16, 8).float()
    eqs = qsf[0, 0:16].to(torch.float8_e4m3fn).view(torch.uint8).float().cpu()
    print("stage Qsf: mismatch =", int((gqs != eqs).sum()))

    pg = int(block_table[0, topk[0, 0, 0]].item())
    gk = st[1408:11648].reshape(128, 80)
    ek = kv_cache[pg, 0].reshape(-1)[:8192].reshape(128, 64)
    print("stage K  : mismatch bytes =", int((gk[:, :64].cpu() != ek.cpu()).sum()))
    gks = st[11648:12672].reshape(128, 8)
    eks = kv_cache[pg, 0].reshape(-1)[8192:].reshape(128, 8)
    print("stage Ksf: mismatch bytes =", int((gks.cpu() != eks.cpu()).sum()))

    gvt = st[12672:31104].reshape(128, 144)[:, :128]
    ev = v[pg, 0].T.contiguous().to(torch.float8_e4m3fn).view(torch.uint8)  # [dim][key]
    ev = (v[pg, 0].T.cpu() * 16.0).to(torch.float8_e4m3fn).view(torch.uint8)
    dd = (gvt.float() - ev.float()).abs()
    print("stage Vt : mismatch bytes =", int((gvt != ev).sum()),
          " max |diff| =", float(dd.max()))

    # ---- reference
    ref_out = torch.zeros(T, hq, hd, device=DEV, dtype=torch.float64)
    ref_scores0 = torch.zeros(hq, 128, device=DEV, dtype=torch.float64)
    for t in range(T):
        r = t // dql
        kvl = int(seq_lens[r])
        for hh in range(hkv):
            kb = topk[hh, t]
            kb = kb[kb >= 0]
            kk = k[block_table[r, kb.long()], hh].reshape(-1, hd)[:kvl]
            vv = v[block_table[r, kb.long()], hh].reshape(-1, hd)[:kvl]
            qh = qdeq[t, hh * 16:(hh + 1) * 16].double()
            s = qh @ kk.double().T * scale * 1.4426950408889634
            if t == 0:
                ref_scores0[hh * 16:(hh + 1) * 16] = s[:, :128]
            p = torch.softmax(s, dim=-1)
            ref_out[t, hh * 16:(hh + 1) * 16] = p @ vv.double()

    s_err = (dbg.double() - ref_scores0).abs()
    den = ref_scores0.abs().clamp_min(1e-3)
    print("scores: max|err| = %.3e   max rel = %.3e   (|s| max %.3f)"
          % (s_err.max().item(), (s_err / den).max().item(), ref_scores0.abs().max().item()))
    bad = (s_err / den) > 1e-2
    if bad.any():
        rows, cols = bad.nonzero(as_tuple=True)
        print("  first bad score cells (row, col):",
              list(zip(rows[:8].tolist(), cols[:8].tolist())),
              " total bad:", int(bad.sum()))
        r0, c0 = int(rows[0]), int(cols[0])
        print("  kernel=%.6f ref=%.6f" % (dbg[r0, c0].item(), ref_scores0[r0, c0].item()))

    o_err = (out.double() - ref_out).abs()
    print("out: max|err| = %.3e   mean|out| = %.3e   rel = %.3e"
          % (o_err.max().item(), ref_out.abs().mean().item(),
             (o_err / ref_out.abs().clamp_min(1e-3)).max().item()))
    print("out sample kernel:", [round(x, 5) for x in out[0, 0, :6].tolist()])
    print("out sample ref   :", [round(x, 5) for x in ref_out[0, 0, :6].tolist()])


if __name__ == "__main__":
    main()
