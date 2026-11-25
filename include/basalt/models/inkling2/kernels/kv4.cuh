// include/basalt/models/inkling2/kernels/kv4.cuh - Inkling 2's NVFP4 K/V rows (--kv nvfp4): one E4M3 scale byte per
// 16 values and E2M1 codes, in a row that is 80 bytes for the keys and for the values alike, so every cell address in
// the engine is one number.  The keys (and every query read against them) are turned by the normalized 128-point
// Hadamard transform first: orthonormal, so a dot product is unchanged before rounding, and the norm is preserved, so
// the qk clip still bounds it.  The values are averaged, not matched, so they are not turned.
//
// Row layout (80 bytes, 16-byte aligned):
//   [0..63]   128 E2M1 codes, two a byte, value 2i in byte i's low nibble
//   [64..79]  one E4M3 scale byte per 8 values (group g covers values 8g..8g+7) - the row exactly full
//
// Eight values a scale rather than sixteen: the same 80 bytes (the eight bytes that used to be padding hold the second
// half of the scales), the same read cost, half the group's amax range to lose, which is what a four-bit plane's error is
// dominated by.
//
// The scale's rounding is the dense stack's: scale = E4M3(amax / 6), codes = E2M1(v / decode(scale)) - see
// basalt/kernels/kv_fp4.cuh, whose helpers this file reuses so both stacks round identically.  Device inlines only.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_fp16.h>

#include <basalt/kernels/kv_fp4.cuh>
#endif

namespace basalt::models::inkling2 {

constexpr int kKvF4CodeBytes = 64;                   ///< 128 four-bit codes
constexpr int kKvF4Group = 8;                        ///< values one E4M3 scale byte covers
constexpr int kKvF4Groups = 16;                      ///< scale bytes a row (64 code bytes + 16 scales = 80)
constexpr int kKvF4Pairs = 2;                        ///< lanes a group spans (four values a lane)
constexpr float kKvF4InvSqrt128 = 0.08838834764831845f;   // 1/sqrt(128)

#if defined(__CUDACC__)
/// The normalized 128-point Hadamard transform of one row, held by one warp: four values a lane, index 4*lane + j.
/// Two stages inside the lane (h = 1, 2) and five across the lanes (h = 4, 8, 16, 32, 64 as shfl masks 1, 2, 4, 8, 16).
/// Orthonormal and its own inverse.  Every lane of the warp must call it.
__device__ __forceinline__ void kv4_hadamard128(float (&v)[4], int lane) {
#pragma unroll
    for (int h = 1; h < 4; h <<= 1) {   // inside the lane: index bits 0 and 1
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j & h) continue;   // each pair once, from the element whose bit is clear
            const float a = v[j], b = v[j ^ h];
            v[j] = a + b;
            v[j ^ h] = a - b;
        }
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {   // across the lanes: index bits 2..6
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = v[j], b = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? b - a : a + b;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= kKvF4InvSqrt128;
}

/// The group's E4M3 scale byte.  Two candidates are measured - amax / 6, the largest code exactly on the group's peak,
/// and 0.9 x that, which spends a little headroom to shrink the step for the other seven values - and the group keeps
/// whichever leaves the smaller squared error.  Free at read time (the byte says which was used), a handful of flops
/// per 8 values at write time.  Every lane of the group returns the same byte.
__device__ __forceinline__ uint32_t k4_scales(const float (&v)[4], int lane) {
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));   // two lanes a group: 8 values
    (void) lane;
    // four candidates: the peak exactly on code 6, and three that spend some of the peak's range on the other seven
    // values (an outlier-heavy group wants the smaller step far more than it wants its maximum represented)
    constexpr float kCand[4] = {1.0f, 0.9f, 0.8f, 0.7f};
    uint32_t best = 0;
    float best_e = 1e30f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint32_t byte = basalt::kernels::kv4_group_scale(a * kCand[c]);
        const float s = basalt::kernels::kv4_e4m3_value(byte);
        float e = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float q = basalt::kernels::kv4_e2m1_value14(basalt::kernels::kv4_group_code(v[j], s)) * (s * 16384.0f);
            e += (v[j] - q) * (v[j] - q);
        }
        e += __shfl_xor_sync(0xffffffffu, e, 1);   // the group's total, in both lanes
        if (e < best_e) {
            best_e = e;
            best = byte;
        }
    }
    return best;
}

/// The lane's four values against a decoded group scale: two code bytes at lane * 2, and, for the group's first lane,
/// its scale byte at kKvF4CodeBytes + group.
__device__ __forceinline__ void k4_store_codes(void* row, int lane, const float (&v)[4], uint32_t scale_byte) {
    uint8_t* p = static_cast<uint8_t*>(row);
    const float s = basalt::kernels::kv4_e4m3_value(scale_byte);
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const uint32_t lo = basalt::kernels::kv4_group_code(v[2 * b], s);
        const uint32_t hi = basalt::kernels::kv4_group_code(v[2 * b + 1], s);
        p[lane * 2 + b] = (uint8_t) (lo | (hi << 4));
    }
    if ((lane & (kKvF4Pairs - 1)) == 0) p[kKvF4CodeBytes + lane / kKvF4Pairs] = (uint8_t) scale_byte;   // lanes 0, 2, 4, ... write 16
}

/// A key row: the 128-point transform, then the group scale, then the codes at lane * 2.
__device__ __forceinline__ void store_k4(void* row, int lane, float (&v)[4]) {
    kv4_hadamard128(v, lane);
    k4_store_codes(row, lane, v, k4_scales(v, lane));
}

/// A value row: the same encoding without the transform (values are averaged, not matched).
__device__ __forceinline__ void store_v4nv(void* row, int lane, const float (&v)[4]) {
    k4_store_codes(row, lane, v, k4_scales(v, lane));
}

/// The eight scale bytes of a row whose scales piece is already in registers (a loader reads bytes 64..79 as one uint4).
__device__ __forceinline__ void k4_piece_scales(const uint4 u, uint32_t (&s)[kKvF4Groups]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) s[4 * i + j] = (w[i] >> (8 * j)) & 0xFFu;
}

/// A row's eight scale bytes as numbers the loader can use without decoding them twice.
__device__ __forceinline__ void k4_row_scales(const uint8_t* row, uint32_t (&s)[kKvF4Groups]) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(row + kKvF4CodeBytes);
#pragma unroll
    for (int i = 0; i < kKvF4Groups / 4; ++i) {
        s[4 * i + 0] = w[i] & 0xFFu;
        s[4 * i + 1] = (w[i] >> 8) & 0xFFu;
        s[4 * i + 2] = (w[i] >> 16) & 0xFFu;
        s[4 * i + 3] = (w[i] >> 24) & 0xFFu;
    }
}

/// 32 fp16 values from one 16-byte code piece (32 codes = four groups of eight) and their four scale bytes, s[0..3].
__device__ __forceinline__ void k4_unpack32(const uint4 u, const uint32_t* s, uint32_t (&h)[16]) {
    const uint4 a = basalt::kernels::kv4_decode8(u.x, s[0]), b = basalt::kernels::kv4_decode8(u.y, s[1]);
    const uint4 c = basalt::kernels::kv4_decode8(u.z, s[2]), d = basalt::kernels::kv4_decode8(u.w, s[3]);
    h[0] = a.x; h[1] = a.y; h[2] = a.z; h[3] = a.w;
    h[4] = b.x; h[5] = b.y; h[6] = b.z; h[7] = b.w;
    h[8] = c.x; h[9] = c.y; h[10] = c.z; h[11] = c.w;
    h[12] = d.x; h[13] = d.y; h[14] = d.z; h[15] = d.w;
}

/// 16 fp16 values from one 8-byte code piece (16 codes = two groups of eight) and their two scale bytes.
__device__ __forceinline__ void k4_unpack16(const uint2 u, uint32_t s0, uint32_t s1, uint32_t (&h)[8]) {
    const uint4 a = basalt::kernels::kv4_decode8(u.x, s0), b = basalt::kernels::kv4_decode8(u.y, s1);
    h[0] = a.x; h[1] = a.y; h[2] = a.z; h[3] = a.w;
    h[4] = b.x; h[5] = b.y; h[6] = b.z; h[7] = b.w;
}

#endif  // __CUDACC__

/// Parity entry point (kernels_parity's test): one warp a row, four values a lane, in -> H(in).  Defined in ops.cu, so a
/// host-only translation unit can call it without a kernel launch.
void kv4_hadamard128_test(const float* in, float* out, int rows, void* stream);

}  // namespace basalt::models::inkling2
