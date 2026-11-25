// include/basalt/models/inkling2/kernels/kv4.cuh - Inkling 2's NVFP4 K/V rows (--kv nvfp4): one E4M3 scale byte per
// 16 values and E2M1 codes, in a row that is 80 bytes for the keys and for the values alike, so every cell address in
// the engine is one number.  The keys (and every query read against them) are turned by the normalized 128-point
// Hadamard transform first: orthonormal, so a dot product is unchanged before rounding, and the norm is preserved, so
// the qk clip still bounds it.  The values are averaged, not matched, so they are not turned.
//
// Row layout (80 bytes, 16-byte aligned):
//   [0..63]   128 E2M1 codes, two a byte, value 2i in byte i's low nibble
//   [64..71]  one E4M3 scale byte per 16 values (group g covers values 16g..16g+15)
//   [72..79]  zero (padding to the key row's 80 bytes)
//
// The scale's rounding is the dense stack's: scale = E4M3(amax / 6), codes = E2M1(v / decode(scale)) - see
// basalt/kernels/kv_fp4.cuh, whose helpers this file reuses so both stacks round identically.  Device inlines only.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include <basalt/kernels/kv_fp4.cuh>

namespace basalt::models::inkling2 {

constexpr int kKvF4CodeBytes = 64;                   ///< 128 four-bit codes
constexpr int kKvF4Group = 16;                       ///< values one E4M3 scale byte covers
constexpr int kKvF4Groups = 8;                       ///< scale bytes a row
constexpr int kKvF4Pairs = 4;                        ///< lanes a group spans (four values a lane)
constexpr float kKvF4InvSqrt128 = 0.08838834764831845f;   // 1/sqrt(128)

/// The normalized 128-point Hadamard transform of one row, held by one warp: four values a lane, index 4*lane + j.
/// Two stages inside the lane (h = 1, 2) and five across the lanes (h = 4, 8, 16, 32, 64 as shfl masks 1, 2, 4, 8, 16).
/// Orthonormal and its own inverse.  Every lane of the warp must call it.
__device__ __forceinline__ void kv4_hadamard128(float (&v)[4], int lane) {
#pragma unroll
    for (int h = 1; h < 4; h <<= 1) {   // inside the lane: index bits 0 and 1
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = v[j], b = v[j ^ h];
            v[j] = (j & h) ? b - a : a + b;
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

/// The group's E4M3 scale byte: the largest magnitude over the four lanes of the group (lanes 4g..4g+3, shfl masks 1
/// and 2), floored and divided by 6 as the dense path does.  Every lane of the group returns the same byte.
__device__ __forceinline__ uint32_t k4_scales(const float (&v)[4], int lane) {
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
    (void) lane;
    return basalt::kernels::kv4_group_scale(a);
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
    if ((lane & (kKvF4Pairs - 1)) == 0) p[kKvF4CodeBytes + lane / kKvF4Pairs] = (uint8_t) scale_byte;
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

/// 32 fp16 values from one 16-byte code piece (32 codes = two groups) and their two scale bytes.
__device__ __forceinline__ void k4_unpack32(const uint4 u, uint32_t s0, uint32_t s1, uint32_t (&h)[16]) {
    const uint4 a = basalt::kernels::kv4_decode8(u.x, s0), b = basalt::kernels::kv4_decode8(u.y, s0);
    const uint4 c = basalt::kernels::kv4_decode8(u.z, s1), d = basalt::kernels::kv4_decode8(u.w, s1);
    h[0] = a.x; h[1] = a.y; h[2] = a.z; h[3] = a.w;
    h[4] = b.x; h[5] = b.y; h[6] = b.z; h[7] = b.w;
    h[8] = c.x; h[9] = c.y; h[10] = c.z; h[11] = c.w;
    h[12] = d.x; h[13] = d.y; h[14] = d.z; h[15] = d.w;
}

/// 16 fp16 values from one 8-byte code piece (16 codes = one group) and its scale byte.
__device__ __forceinline__ void k4_unpack16(const uint2 u, uint32_t s, uint32_t (&h)[8]) {
    const uint4 a = basalt::kernels::kv4_decode8(u.x, s), b = basalt::kernels::kv4_decode8(u.y, s);
    h[0] = a.x; h[1] = a.y; h[2] = a.z; h[3] = a.w;
    h[4] = b.x; h[5] = b.y; h[6] = b.z; h[7] = b.w;
}

}  // namespace basalt::models::inkling2
