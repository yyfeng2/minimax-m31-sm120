// Probe SM120a support for fp4/fp8 conversion instructions used by the native kernel.
#include <cstdint>
#include <cstdio>

__global__ void probe(const float* f, const uint16_t* p, const uint32_t* q, uint32_t* out) {
  uint32_t o = 0, o2 = 0, o3 = 0, o4 = 0;
  uint16_t packed = p[0];
  uint32_t scale = q[0];
#if TEST == 1
  // V dequant: 4x E2M1 codes * 4x E4M3 scales -> 4x E4M3
  asm volatile("mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %1, %2;" : "=r"(o) : "h"(packed), "r"(scale));
#elif TEST == 2
  // Q/P quantize: two f32 -> 2x E2M1 packed in low byte
  asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=r"(o) : "f"(f[0]), "f"(f[1]));
#elif TEST == 3
  // scale encode: two f32 -> 2x E4M3
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=r"(o) : "f"(f[0]), "f"(f[1]));
#elif TEST == 4
  // E2M1x4 -> 4 f16 (dequant codes to half for block amax / reference)
  asm volatile("cvt.rn.f16x2.e2m1x2 %0, %1, %2;" : "=r"(o) : "h"(packed), "h"(packed));
#elif TEST == 5
  // QK native: already proven
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+r"(o), "+r"(o2), "+r"(o3), "+r"(o4)
      : "r"(q[0]), "r"(q[1]), "r"(q[2]), "r"(q[3]), "r"(q[4]), "r"(q[5]), "r"(q[6]),
        "h"((uint16_t)0), "h"((uint16_t)0), "r"(q[7]), "h"((uint16_t)0), "h"((uint16_t)0));
#elif TEST == 6
  // FP8 PV native: mxf8f6f4 1X ue8m0, e4m3 x e4m3
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+r"(o), "+r"(o2), "+r"(o3), "+r"(o4)
      : "r"(q[0]), "r"(q[1]), "r"(q[2]), "r"(q[3]), "r"(q[4]), "r"(q[5]), "r"(q[6]),
        "h"((uint16_t)0), "h"((uint16_t)0), "r"(q[7]), "h"((uint16_t)0), "h"((uint16_t)0));
#endif
  out[0] = o;
  out[1] = o2;
  out[2] = o3;
  out[3] = o4;
}

int main() { return 0; }
