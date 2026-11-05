// SM120 (RTX PRO 6000 Blackwell) MiniMax-M3 sparse decode attention over a
// packed NVFP4 KV cache, using *native* block-scaled tensor-core MMA:
//
//   QK^T : mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//          .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//          -> reads the E2M1 codes and the E4M3/16 block scales of the cache
//             in place; no dequantization of K, no Q down-conversion.
//   PV   : mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
//          .m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
//          -> probabilities become E4M3 with exact power-of-two E8M0 scales;
//             V codes are converted (code * per-dim E4M3 scale) to E4M3 in
//             shared memory. The NVFP4 cache quantizes V along the head dim
//             (per token, 16-value blocks), so its scales are not aligned with
//             the PV contraction axis (keys); a scale-native PV MMA is
//             mathematically impossible for this cache layout. This is the
//             same conversion the official SM100 Q8KV4 kernel performs
//             (mul.rn.satfinite.e4m3x4), but SM120 has no QMUL4, so the
//             conversion is done with cvt + fp16 arithmetic here.
//
// Cache contract (vLLM MiniMax-M3 NVFP4 KV cache, verified against the Triton
// dequant in vllm/models/minimax_m3/common/ops/sparse_attn.py):
//   kv uint8 [pages, 2*Hkv, 128, 72]; one head slot = 9216 B =
//       [ data plane 8192 B | scale plane 1024 B ]
//   data  byte at pos*64 + dim/2; low nibble = even dim.
//   K sf  byte at pos*8 + dim/16.
//   V sf  byte at (pos/4)*32 + 4*(dim/16) + pos%4.
//   Q   bf16 [T, Hq, 128], Hq = 16 * Hkv, head_dim = 128, page = 128.
//   topk int32 [Hkv, T, topk]: logical page ids (relative to the request), -1 pad.
//   block_table int32 [num_reqs, bt_stride], seq_lens int32 [num_reqs].
//
// Optional debug buffers (pass nullptr in production):
//   debug_scores float [Hkv*16, 128]: page-0 QK scores after scaling/masking.
//   debug_stage  uint8 [58240]: staged Q/Qs/K/Ks/Vt, then P/Ps, then the
//                running O/l after each of the first 3 pages.
//   Both are written only for token 0.
//
// Split-K (flash-decoding): the grid is (T, Hkv, nchunk); chunk z owns the
// contiguous slice [z*per, min(topk, (z+1)*per)) of the top-k pages, so each
// CTA runs the same online-softmax recurrence over fewer pages. With
// nchunk > 1 each CTA writes an unnormalized fp32 partial O plus (m, l) and
// ``msa_merge_kernel`` combines them:
//   w_c = exp2(m_c - M), out = sum_c w_c O_c / sum_c w_c l_c, M = max_c m_c.
// nchunk == 1 writes the bf16 output directly (no merge, no scratch).
// The chunk count is chosen on the host to keep ~256 CTAs busy (see
// msa_nvfp4.py); it is constant per CUDA-graph bucket.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>

#ifdef __CUDACC__
#define DEV __device__ __forceinline__
#else
#define DEV inline
#endif

namespace msa_sm120 {

constexpr int kHeadDim = 128;
constexpr int kPage = 128;
constexpr int kHqPerKv = 16;
constexpr int kWarps = 4;

constexpr int kSlotBytes = kPage * 72;      // 9216
constexpr int kDataBytes = kPage * 64;      // 8192
constexpr int kVScale = 16;                 // E4M3 headroom for V values

constexpr int kSQStride = 80;    // 64B codes + pad
constexpr int kSKStride = 80;
constexpr int kSVTStride = 144;  // 128B + pad (bank-conflict free 4-row/4-lane)
constexpr int kSPStride = 144;

// E2M1 nibble -> fp16 bits (exact: 0, .5, 1, 1.5, 2, 3, 4, 6 and negatives).
DEV uint32_t e2m1_nib_to_f16(int n) {
  int mag = n & 7;
  int e = (mag >> 1) & 3;
  uint32_t bits = (mag == 0) ? 0u
                  : (e == 0 ? 0x3800u
                            : (uint32_t)(((e + 14) << 10) | ((mag & 1) << 9)));
  return bits | ((uint32_t)(n & 8) << 12);
}

DEV __half2 e2m1_pair_to_half2(uint32_t byte) {
  uint32_t lo = e2m1_nib_to_f16(byte & 0xF);
  uint32_t hi = e2m1_nib_to_f16((byte >> 4) & 0xF);
  uint32_t packed = lo | (hi << 16);
  return *reinterpret_cast<__half2*>(&packed);
}

// NVFP4 quantizer for Q: hardware round-to-nearest-even on the E2M1 grid.
// Same conversion CUTLASS/vLLM use (cvt.rn.satfinite.e2m1x2.f32), which puts
// the FIRST source in the HIGH nibble.
DEV uint8_t e2m1x2_pack(float lo, float hi) {
  uint32_t tmp;
  asm("{\n"
      ".reg .b8 b0;\n"
      ".reg .b8 b1;\n"
      ".reg .b8 b2;\n"
      ".reg .b8 b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %1, %2;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(tmp)
      : "f"(hi), "f"(lo));
  return (uint8_t)(tmp & 0xFFu);
}

// cvt.rn.satfinite.e4m3x2.f32 puts the FIRST source in the HIGH byte.
DEV uint16_t e4m3x2_pack(float lo, float hi) {
  uint16_t h;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h) : "f"(hi), "f"(lo));
  return h;
}

DEV uint8_t e4m3_encode(float v) {
  return (uint8_t)(e4m3x2_pack(v, v) & 0xFFu);
}

DEV __half e4m3_to_half(uint8_t b) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)b));
  return __ushort_as_half((unsigned short)(h2 & 0xFFFFu));
}

DEV uint16_t e4m3x2_from_half2(__half2 a) {
  uint32_t r = *reinterpret_cast<uint32_t*>(&a);
  uint16_t h;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h) : "r"(r));
  return h;
}

DEV void mma_qk_fp4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                    uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
        "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

DEV void mma_pv_fp8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                    uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
        "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

struct Smem {
  alignas(16) uint8_t q[kHqPerKv * kSQStride];      // 1280
  alignas(16) uint8_t qs[kHqPerKv * 8];             // 128
  alignas(16) uint8_t k[kPage * kSKStride];         // 10240
  alignas(16) uint8_t ks[kPage * 8];                // 1024
  alignas(16) uint8_t vt[kHeadDim * kSVTStride];    // 18432
  alignas(16) uint8_t p[kHqPerKv * kSPStride];      // 2304
  alignas(16) uint8_t ps[kHqPerKv * 4];             // 64
  alignas(16) float wmax[kWarps][kHqPerKv];
  alignas(16) float wsum[kWarps][kHqPerKv];
  alignas(16) float mrun[kHqPerKv];
  alignas(16) float lrun[kHqPerKv];
  alignas(16) float mnew[kHqPerKv];
  alignas(16) float alpha[kHqPerKv];
};

__global__ void __launch_bounds__(kWarps * 32) msa_decode_kernel(
    const __nv_bfloat16* __restrict__ q,      // [T, 16*Hkv, 128]
    const uint8_t* __restrict__ kv,           // [pages, 2*Hkv, 9216]
    const int32_t* __restrict__ topk,         // [Hkv, T, topk_cap]
    const int32_t* __restrict__ block_table,  // [num_reqs, bt_stride]
    const int32_t* __restrict__ seq_lens,     // [num_reqs]
    __nv_bfloat16* __restrict__ out,          // [T, 16*Hkv, 128]
    float* __restrict__ o_part,               // [nchunk, T, 16*Hkv, 128] or null
    float2* __restrict__ ml_part,             // [nchunk, T, 16*Hkv] or null
    int T, int topk_cap, int topk_stride_h, int topk_stride_t,
    int decode_query_len, int num_kv_heads, int bt_stride, int nchunk,
    float scale_log2e, float v_global, float* debug_scores, uint8_t* debug_stage) {
  extern __shared__ uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);

  const int token = blockIdx.x;
  const int kvh = blockIdx.y;
  const int chunk = blockIdx.z;
  const int per_chunk = (topk_cap + nchunk - 1) / nchunk;
  const int i_begin = chunk * per_chunk;
  const int i_end_lim = i_begin + per_chunk;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int t0 = lane & 3;
  const int t1 = lane >> 2;

  const int req = token / decode_query_len;
  const int qoff = token - req * decode_query_len;
  const int seq_len = seq_lens[req];
  const int qpos = seq_len - decode_query_len + qoff;
  const int kv_len = qpos + 1 > 0 ? qpos + 1 : 0;
  const int num_blocks = (kv_len + kPage - 1) / kPage;
  const int hq = kvh * kHqPerKv;

  // ---------------- Q: load + quantize to NVFP4 (E2M1 codes, E4M3/16 scales)
  {
    const int h = tid >> 3;
    const int blk = tid & 7;
    const __nv_bfloat16* qrow =
        q + ((size_t)token * (size_t)(kHqPerKv * num_kv_heads) + (size_t)hq + h) * kHeadDim +
        blk * 16;
    float x[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      x[i] = __bfloat162float(qrow[i]);
      amax = fmaxf(amax, fabsf(x[i]));
    }
    float sc = amax > 0.f ? amax / 6.0f : 1.0f;
    uint8_t sb = e4m3_encode(sc);
    float sdec = __half2float(e4m3_to_half(sb));
    uint8_t bytes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // Divide (not multiply by 1/sc) so exact ties round like the reference.
      bytes[i] = e2m1x2_pack(sdec > 0.f ? x[2 * i] / sdec : 0.f,
                             sdec > 0.f ? x[2 * i + 1] / sdec : 0.f);
    }
    s.qs[h * 8 + blk] = sb;
    uint32_t* dst = reinterpret_cast<uint32_t*>(s.q + h * kSQStride + blk * 8);
    dst[0] = *reinterpret_cast<uint32_t*>(&bytes[0]);
    dst[1] = *reinterpret_cast<uint32_t*>(&bytes[4]);
  }
  if (tid < kHqPerKv) {
    s.mrun[tid] = -CUDART_INF_F;
    s.lrun[tid] = 0.f;
  }

  float o[4][4];
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    o[g][0] = o[g][1] = o[g][2] = o[g][3] = 0.f;
  }

  const size_t topk_base = (size_t)kvh * topk_stride_h + (size_t)token * topk_stride_t;
  const size_t pageStride = (size_t)(2 * num_kv_heads) * kSlotBytes;
  const float qk_scale = scale_log2e;

  int first_valid = -1;

  for (int i = i_begin; i < topk_cap && i < i_end_lim; ++i) {
    const int blk = topk[topk_base + i];
    if (blk < 0 || blk >= num_blocks) continue;
    if (first_valid < 0) first_valid = i;
    const int page = block_table[(size_t)req * bt_stride + blk];
    // live keys inside this logical block (topk ids are page-aligned)
    const int key_limit = kv_len - blk * kPage;
    const uint8_t* kbase = kv + (size_t)page * pageStride + (size_t)(2 * kvh) * kSlotBytes;
    const uint8_t* vbase = kbase + kSlotBytes;

    // ---------------- load K codes + block scales (16B per thread step)
    {
      const uint8_t* kdata = kbase;
      const uint8_t* ksf = kbase + kDataBytes;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int idx = tid + j * 128;   // 0..1023
        int row = idx >> 3;
        int seg = idx & 7;
        uint2 v = *reinterpret_cast<const uint2*>(kdata + row * 64 + seg * 8);
        *reinterpret_cast<uint2*>(s.k + row * kSKStride + seg * 8) = v;
      }
      int row = tid;
      uint2 sv = *reinterpret_cast<const uint2*>(ksf + row * 8);
      *reinterpret_cast<uint2*>(s.ks + row * 8) = sv;
    }

    // ---------------- load V codes + scales, convert to E4M3, transpose to [dim][key]
    {
      const uint8_t* vdata = vbase;
      const uint8_t* vsf = vbase + kDataBytes;
      const __half2 kx16 = __float2half2_rn((float)kVScale);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int unit = tid + j * 128;   // 0..1023
        int k4 = unit >> 5;         // token quad
        int d4 = unit & 31;         // dim quad
        int sb = d4 >> 2;           // 16-dim block index
        const uint8_t* sfu = vsf + k4 * 32 + 4 * sb;
        uint32_t w[4];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
          int row = 4 * k4 + ii;
          uint16_t codes = *reinterpret_cast<const uint16_t*>(vdata + row * 64 + 2 * d4);
          __half2 sc2 = __hmul2(__half2half2(e4m3_to_half(sfu[ii])), kx16);
          __half2 a01 = __hmul2(e2m1_pair_to_half2(codes & 0xFFu), sc2);
          __half2 a23 = __hmul2(e2m1_pair_to_half2((codes >> 8) & 0xFFu), sc2);
          uint32_t e01 = e4m3x2_from_half2(a01);
          uint32_t e23 = e4m3x2_from_half2(a23);
          w[ii] = e01 | (e23 << 16);
        }
        // 4x4 byte transpose: out[j].byte[ii] = w[ii].byte[j]
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          uint32_t lo = __byte_perm(w[0], w[1], (jj) | ((4 + jj) << 4) | (jj << 8) | ((4 + jj) << 12));
          uint32_t hi = __byte_perm(w[2], w[3], (jj) | ((4 + jj) << 4) | (jj << 8) | ((4 + jj) << 12));
          uint32_t outw = __byte_perm(lo, hi, 0x5410);
          *reinterpret_cast<uint32_t*>(s.vt + (4 * d4 + jj) * kSVTStride + 4 * k4) = outw;
        }
      }
    }

    __syncthreads();

    if (debug_stage != nullptr && chunk == 0 && token == 0 && kvh == 0 &&
        i == first_valid) {
      for (int j = tid; j < kHqPerKv * kSQStride; j += 128) debug_stage[j] = s.q[j];
      for (int j = tid; j < kHqPerKv * 8; j += 128) debug_stage[1280 + j] = s.qs[j];
      for (int j = tid; j < kPage * kSKStride; j += 128) debug_stage[1408 + j] = s.k[j];
      for (int j = tid; j < kPage * 8; j += 128) debug_stage[11648 + j] = s.ks[j];
      for (int j = tid; j < kHeadDim * kSVTStride; j += 128) debug_stage[12672 + j] = s.vt[j];
    }

    // ---------------- QK^T: native FP4 x FP4 block-scaled MMA
    // One MMA covers k = 64 of the 128 head dims, so c = 0,1 accumulate into
    // the same d[g][]: d[g][0..3] = keys 32*warp+8g+2t0 (+1), heads t1 / t1+8.
    float d[4][4];
    {
      const int mq = 8 * (lane & 1) + t1;  // SFA row for this lane
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int key = 32 * warp + 8 * g + t1;
        const uint8_t* krow = s.k + key * kSKStride;
        const uint8_t* ksrow = s.ks + key * 8;
        d[g][0] = d[g][1] = d[g][2] = d[g][3] = 0.f;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          uint32_t a0 = *reinterpret_cast<const uint32_t*>(s.q + t1 * kSQStride + 32 * c + 4 * t0);
          uint32_t a1 = *reinterpret_cast<const uint32_t*>(s.q + (t1 + 8) * kSQStride + 32 * c + 4 * t0);
          uint32_t a2 = *reinterpret_cast<const uint32_t*>(s.q + t1 * kSQStride + 32 * c + 16 + 4 * t0);
          uint32_t a3 = *reinterpret_cast<const uint32_t*>(s.q + (t1 + 8) * kSQStride + 32 * c + 16 + 4 * t0);
          uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow + 32 * c + 4 * t0);
          uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + 32 * c + 16 + 4 * t0);
          uint32_t sfa = *reinterpret_cast<const uint32_t*>(s.qs + mq * 8 + 4 * c);
          uint32_t sfb = *reinterpret_cast<const uint32_t*>(ksrow + 4 * c);
          mma_qk_fp4(d[g], a0, a1, a2, a3, b0, b1, sfa, sfb);
        }
      }
    }

    // ---------------- scale + mask + row max
    {
      float mx0 = -CUDART_INF_F, mx1 = -CUDART_INF_F;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int key0 = 32 * warp + 8 * g + 2 * t0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            float v = d[g][r * 2 + j] * qk_scale;
            if (key0 + j >= key_limit) v = -CUDART_INF_F;
            d[g][r * 2 + j] = v;
            if (r == 0) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
          }
        }
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      if (t0 == 0) {
        s.wmax[warp][t1] = mx0;
        s.wmax[warp][t1 + 8] = mx1;
      }
    }

    if (debug_scores != nullptr && chunk == 0 && token == 0 && i == first_valid) {
      float* dbg = debug_scores + (size_t)(kvh * kHqPerKv) * kPage;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int key0 = 32 * warp + 8 * g + 2 * t0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          int row = t1 + 8 * r;
          dbg[row * kPage + key0 + 0] = d[g][r * 2 + 0];
          dbg[row * kPage + key0 + 1] = d[g][r * 2 + 1];
        }
      }
    }

    __syncthreads();

    if (tid < kHqPerKv) {
      float pm = fmaxf(fmaxf(s.wmax[0][tid], s.wmax[1][tid]),
                       fmaxf(s.wmax[2][tid], s.wmax[3][tid]));
      float mn = fmaxf(s.mrun[tid], pm);
      if (mn == -CUDART_INF_F) mn = 0.f;  // page without any live key
      s.mnew[tid] = mn;
      s.alpha[tid] = exp2f(s.mrun[tid] - mn);  // first page: exp2(-inf) = 0
      s.mrun[tid] = mn;
    }
    __syncthreads();

    // ---------------- probabilities, per-32-key E8M0 scale, pack E4M3
    const float mn0 = s.mnew[t1];
    const float mn1 = s.mnew[t1 + 8];
    float sum0 = 0.f, sum1 = 0.f, amax0 = 0.f, amax1 = 0.f;
    {
#pragma unroll
      for (int g = 0; g < 4; ++g) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float mn = r == 0 ? mn0 : mn1;
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            float p = exp2f(d[g][r * 2 + j] - mn);
            d[g][r * 2 + j] = p;
            if (r == 0) {
              sum0 += p;
              amax0 = fmaxf(amax0, p);
            } else {
              sum1 += p;
              amax1 = fmaxf(amax1, p);
            }
          }
        }
      }
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
      amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffffu, amax0, 1));
      amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffffu, amax0, 2));
      amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffffu, amax1, 1));
      amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffffu, amax1, 2));
      if (t0 == 0) {
        s.wsum[warp][t1] = sum0;
        s.wsum[warp][t1 + 8] = sum1;
      }
    }
    // E8M0 scale: P * 2^-e with amax*2^-e in [128, 256) -> no E4M3 saturation,
    // 7 bits of headroom below the max.
    int e0 = 0, e1 = 0;
    {
      if (amax0 > 0.f) {
        int e = (int)floorf(log2f(amax0)) - 7;
        e0 = e < -127 ? -127 : e;
      }
      if (amax1 > 0.f) {
        int e = (int)floorf(log2f(amax1)) - 7;
        e1 = e < -127 ? -127 : e;
      }
      if (t0 == 0) {
        s.ps[t1 * 4 + warp] = (uint8_t)(e0 + 127);
        s.ps[(t1 + 8) * 4 + warp] = (uint8_t)(e1 + 127);
      }
    }
    const float mul0 = exp2f((float)(-e0));
    const float mul1 = exp2f((float)(-e1));
    {
      for (int g = 0; g < 4; ++g) {
        const int kb = 32 * warp + 8 * g + 2 * t0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float mul = r == 0 ? mul0 : mul1;
          float p0 = d[g][r * 2 + 0] * mul;
          float p1 = d[g][r * 2 + 1] * mul;
          // P row = head, byte index = key within the page.
          *reinterpret_cast<uint16_t*>(s.p + (t1 + 8 * r) * kSPStride + kb) = e4m3x2_pack(p0, p1);
        }
      }
    }
    __syncthreads();

    if (debug_stage != nullptr && chunk == 0 && token == 0 && kvh == 0 &&
        i == first_valid) {
      for (int j = tid; j < kHqPerKv * kSPStride; j += 128) debug_stage[31104 + j] = s.p[j];
      for (int j = tid; j < kHqPerKv * 4; j += 128) debug_stage[31104 + 2304 + j] = s.ps[j];
    }

    if (tid < kHqPerKv) {
      float ws = s.wsum[0][tid] + s.wsum[1][tid] + s.wsum[2][tid] + s.wsum[3][tid];
      s.lrun[tid] = s.lrun[tid] * s.alpha[tid] + ws;
    }

    // ---------------- rescale O, then PV: native FP8 x FP8 block-scaled MMA
    {
      const float a0 = s.alpha[t1];
      const float a1 = s.alpha[t1 + 8];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        o[g][0] *= a0;
        o[g][1] *= a0;
        o[g][2] *= a1;
        o[g][3] *= a1;
      }
      const int ma = 8 * (lane & 1) + t1;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t sfa = (uint32_t)s.ps[ma * 4 + c];
        const uint32_t sfb = 127u;  // E8M0 1.0: V is stored as absolute E4M3
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const int dim = 32 * warp + 8 * g + t1;
          uint32_t a0r = *reinterpret_cast<const uint32_t*>(s.p + t1 * kSPStride + 32 * c + 4 * t0);
          uint32_t a1r = *reinterpret_cast<const uint32_t*>(s.p + (t1 + 8) * kSPStride + 32 * c + 4 * t0);
          uint32_t a2r = *reinterpret_cast<const uint32_t*>(s.p + t1 * kSPStride + 32 * c + 16 + 4 * t0);
          uint32_t a3r = *reinterpret_cast<const uint32_t*>(s.p + (t1 + 8) * kSPStride + 32 * c + 16 + 4 * t0);
          uint32_t b0 = *reinterpret_cast<const uint32_t*>(s.vt + dim * kSVTStride + 32 * c + 4 * t0);
          uint32_t b1 = *reinterpret_cast<const uint32_t*>(s.vt + dim * kSVTStride + 32 * c + 16 + 4 * t0);
          mma_pv_fp8(o[g], a0r, a1r, a2r, a3r, b0, b1, sfa, sfb);
        }
      }
    }
    __syncthreads();

    if (debug_stage != nullptr && chunk == 0 && token == 0 && kvh == 0) {
      // running O and l after each processed page (3 pages max in the test)
      uint8_t* dbg2 = debug_stage + 33472 + (size_t)i * (kHqPerKv * kHeadDim * 4 + 64);
      float* fb = reinterpret_cast<float*>(dbg2);
      for (int g = 0; g < 4; ++g) {
        const int dm = 32 * warp + 8 * g + 2 * t0;
        fb[t1 * kHeadDim + dm + 0] = o[g][0];
        fb[t1 * kHeadDim + dm + 1] = o[g][1];
        fb[(t1 + 8) * kHeadDim + dm + 0] = o[g][2];
        fb[(t1 + 8) * kHeadDim + dm + 1] = o[g][3];
      }
      if (tid < kHqPerKv) fb[kHqPerKv * kHeadDim + tid] = s.lrun[tid];
    }
  }

  // ---------------- epilogue
  // V was scaled by kVScale on the way into E4M3; fold that back here.
  const float vg = v_global * (1.f / (float)kVScale);
  if (nchunk > 1) {
    const size_t hq_all = (size_t)kHqPerKv * num_kv_heads;
    float* op = o_part + (((size_t)chunk * T + token) * hq_all + (size_t)hq) * kHeadDim;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int dim = 32 * warp + 8 * g + 2 * t0;
      op[(size_t)t1 * kHeadDim + dim + 0] = o[g][0] * vg;
      op[(size_t)t1 * kHeadDim + dim + 1] = o[g][1] * vg;
      op[(size_t)(t1 + 8) * kHeadDim + dim + 0] = o[g][2] * vg;
      op[(size_t)(t1 + 8) * kHeadDim + dim + 1] = o[g][3] * vg;
    }
    if (tid < kHqPerKv) {
      ml_part[((size_t)chunk * T + token) * hq_all + (size_t)hq + tid] =
          make_float2(s.mrun[tid], s.lrun[tid]);
    }
    return;
  }
  const float l0 = s.lrun[t1];
  const float l1 = s.lrun[t1 + 8];
  const float r0 = l0 > 0.f ? vg / l0 : 0.f;
  const float r1 = l1 > 0.f ? vg / l1 : 0.f;
  __nv_bfloat16* obase =
      out + ((size_t)token * (size_t)(kHqPerKv * num_kv_heads) + (size_t)hq) * kHeadDim;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const int dim = 32 * warp + 8 * g + 2 * t0;
    obase[(size_t)t1 * kHeadDim + dim + 0] = __float2bfloat16(o[g][0] * r0);
    obase[(size_t)t1 * kHeadDim + dim + 1] = __float2bfloat16(o[g][1] * r0);
    obase[(size_t)(t1 + 8) * kHeadDim + dim + 0] = __float2bfloat16(o[g][2] * r1);
    obase[(size_t)(t1 + 8) * kHeadDim + dim + 1] = __float2bfloat16(o[g][3] * r1);
  }
}

// Split-K merge: combine nchunk unnormalized partials (see the header comment).
// One CTA per (token, kv head); 128 threads cover 16 heads x 128 dims in 16
// strided passes.
__global__ void __launch_bounds__(128) msa_merge_kernel(
    const float* __restrict__ o_part,   // [nchunk, T, Hq, 128]
    const float2* __restrict__ ml_part, // [nchunk, T, Hq]
    __nv_bfloat16* __restrict__ out, int T, int num_kv_heads, int nchunk) {
  constexpr int kMaxChunks = 32;
  __shared__ float w[kMaxChunks][kHqPerKv];
  __shared__ float den[kHqPerKv];

  const int token = blockIdx.x;
  const int kvh = blockIdx.y;
  const int tid = threadIdx.x;
  const size_t hq_all = (size_t)kHqPerKv * num_kv_heads;
  const size_t ghead = (size_t)kvh * kHqPerKv;

  if (tid < kHqPerKv) {
    const size_t base = (size_t)token * hq_all + ghead + tid;
    float M = -CUDART_INF_F;
    for (int c = 0; c < nchunk; ++c) {
      const float2 ml = ml_part[(size_t)c * (size_t)T * hq_all + base];
      if (ml.y > 0.f) M = fmaxf(M, ml.x);
    }
    float d = 0.f;
    for (int c = 0; c < nchunk; ++c) {
      const float2 ml = ml_part[(size_t)c * (size_t)T * hq_all + base];
      const float wc = (ml.y > 0.f) ? exp2f(ml.x - M) : 0.f;
      w[c][tid] = wc;
      d += wc * ml.y;
    }
    den[tid] = d;
  }
  __syncthreads();

  __nv_bfloat16* obase = out + ((size_t)token * hq_all + ghead) * kHeadDim;
  for (int idx = tid; idx < kHqPerKv * kHeadDim; idx += 128) {
    const int h = idx >> 7;
    const int dim = idx & (kHeadDim - 1);
    const size_t off = ((size_t)token * hq_all + ghead + h) * kHeadDim + dim;
    float acc = 0.f;
    for (int c = 0; c < nchunk; ++c) {
      acc += w[c][h] * o_part[(size_t)c * (size_t)T * hq_all * kHeadDim + off];
    }
    const float d = den[h];
    obase[(size_t)h * kHeadDim + dim] =
        __float2bfloat16(d > 0.f ? acc / d : 0.f);
  }
}

}  // namespace msa_sm120

// Torch-free binding: the Python side passes raw device pointers and the
// current stream, so this file only needs cuda_runtime + pybind11.
#include <pybind11/pybind11.h>

void msa_decode(int64_t q, int64_t kv, int64_t topk, int64_t block_table,
                int64_t seq_lens, int64_t out, int64_t T, int64_t topk_cap,
                int64_t topk_stride_h, int64_t topk_stride_t,
                int64_t decode_query_len, int64_t num_kv_heads, int64_t bt_stride,
                int64_t o_part, int64_t ml_part, int64_t nchunk,
                double scale_log2e, double v_global, int64_t debug_scores,
                int64_t debug_stage, int64_t stream) {
  const int nk = (int)nchunk;
  dim3 grid((unsigned)T, (unsigned)num_kv_heads, (unsigned)nk);
  msa_sm120::msa_decode_kernel<<<grid, 128, sizeof(msa_sm120::Smem),
                                 (cudaStream_t)stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const uint8_t*>(kv),
      reinterpret_cast<const int32_t*>(topk),
      reinterpret_cast<const int32_t*>(block_table),
      reinterpret_cast<const int32_t*>(seq_lens),
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<float*>(o_part),
      reinterpret_cast<float2*>(ml_part),
      (int)T, (int)topk_cap, (int)topk_stride_h, (int)topk_stride_t,
      (int)decode_query_len, (int)num_kv_heads,
      (int)bt_stride, nk, (float)scale_log2e, (float)v_global,
      reinterpret_cast<float*>(debug_scores),
      reinterpret_cast<uint8_t*>(debug_stage));
  if (nk > 1) {
    dim3 mgrid((unsigned)T, (unsigned)num_kv_heads);
    msa_sm120::msa_merge_kernel<<<mgrid, 128, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const float*>(o_part),
        reinterpret_cast<const float2*>(ml_part),
        reinterpret_cast<__nv_bfloat16*>(out), (int)T, (int)num_kv_heads, nk);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("msa_decode", &msa_decode, "SM120 native FP4/FP8 sparse decode (NVFP4 cache)");
}
