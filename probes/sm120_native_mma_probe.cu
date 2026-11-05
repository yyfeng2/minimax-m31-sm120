// Probe: does ptxas accept block-scaled MMA for sm_120a?
//  1) mixed FP8 x FP4 with E8M0 block scales (mxf8f6f4)
//  2) FP4 x FP4 with E4M3 block scales (mxf4nvf4, NVFP4)
#include <cstdint>

__global__ void mma_mxf8f6f4_mixed(const uint32_t* A, const uint32_t* B,
                                  const uint32_t* SA, const uint32_t* SB, float* D) {
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  uint32_t a0 = A[0], a1 = A[1], a2 = A[2], a3 = A[3];
  uint32_t b0 = B[0], b1 = B[1];
  uint32_t sfa = SA[0], sfb = SB[0];
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
  D[threadIdx.x] = d0 + d1 + d2 + d3;
}

__global__ void mma_mxf4nvf4(const uint32_t* A, const uint32_t* B,
                             const uint32_t* SA, const uint32_t* SB, float* D) {
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  uint32_t a0 = A[0], a1 = A[1], a2 = A[2], a3 = A[3];
  uint32_t b0 = B[0], b1 = B[1];
  uint32_t sfa = SA[0], sfb = SB[0];
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
  D[threadIdx.x] = d0 + d1 + d2 + d3;
}
