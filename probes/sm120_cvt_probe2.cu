#include <cstdint>
__global__ void probe(const float* f, const uint16_t* p, uint32_t* out) {
  uint16_t h = 0;
  uint32_t o = 0;
#if TEST == 1
  asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=h"(h) : "f"(f[0]), "f"(f[1]));
#elif TEST == 2
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h) : "f"(f[0]), "f"(f[1]));
#elif TEST == 3
  asm volatile("cvt.rn.f16x2.e2m1x2 %0, %1;" : "=r"(o) : "h"(p[0]));
#elif TEST == 4
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(o) : "h"(p[0]));
#elif TEST == 5
  asm volatile("cvt.rn.satfinite.e2m1x2.f16x2 %0, %1;" : "=h"(h) : "r"(o));
#elif TEST == 6
  asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h) : "r"(o));
#endif
  out[0] = (uint32_t)h + o;
}
