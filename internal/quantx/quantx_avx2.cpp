//go:build !darwin
#include "quantx.h"
#include <immintrin.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx2,fma")
#endif

/**
 * block_q2_k defines the structure for a QuantX 2-bit quantized block.
 * It contains 256 elements, each represented by 2 bits.
 */
typedef struct {
    float d;       // Block-level scale factor
    float dmin;    // Block-level minimum value (bias)
    uint8_t qs[64]; // 256 elements * 2 bits/element = 64 bytes
} block_q2_k;

extern "C" {

/**
 * dequantize_q2_k_avx2 dequantizes Q2_K blocks to float32 using AVX2 and FMA.
 * Optimized for Zen 3 (Ryzen 5900HX) by interleaving two blocks to saturate FMA ports
 * and using vectorized unpacking with shuffle/mask strategy.
 */
void dequantize_q2_k_avx2(const void* vx, float* vy, int k) {
    const block_q2_k* x = (const block_q2_k*)vx;
    const int nb = k / 256;

    // Constants for bit extraction and expansion
    const __m256i m3 = _mm256_set1_epi32(0x03);
    const __m256i v_shifts = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);

    // Pre-defined masks for _mm256_shuffle_epi8 to expand 8 bytes into 8 int32s.
    const __m256i m01 = _mm256_setr_epi8(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1);
    const __m256i m23 = _mm256_setr_epi8(2,2,2,2, 2,2,2,2, 2,2,2,2, 2,2,2,2, 3,3,3,3, 3,3,3,3, 3,3,3,3, 3,3,3,3);
    const __m256i m45 = _mm256_setr_epi8(4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 5,5,5,5, 5,5,5,5, 5,5,5,5, 5,5,5,5);
    const __m256i m67 = _mm256_setr_epi8(6,6,6,6, 6,6,6,6, 6,6,6,6, 6,6,6,6, 7,7,7,7, 7,7,7,7, 7,7,7,7, 7,7,7,7);
    const __m256i m89 = _mm256_setr_epi8(8,8,8,8, 8,8,8,8, 8,8,8,8, 8,8,8,8, 9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9);
    const __m256i mAB = _mm256_setr_epi8(10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10, 11,11,11,11, 11,11,11,11, 11,11,11,11, 11,11,11,11);
    const __m256i mCD = _mm256_setr_epi8(12,12,12,12, 12,12,12,12, 12,12,12,12, 12,12,12,12, 13,13,13,13, 13,13,13,13, 13,13,13,13, 13,13,13,13);
    const __m256i mEF = _mm256_setr_epi8(14,14,14,14, 14,14,14,14, 14,14,14,14, 14,14,14,14, 15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15);

    for (int i = 0; i < nb; i++) {
        const __m256 v_scale = _mm256_set1_ps(x[i].d);
        const __m256 v_bias  = _mm256_set1_ps(x[i].dmin);
        const uint8_t* qs = x[i].qs;
        float* y = vy + i * 256;

        // Process 256 elements in 4 groups of 64.
        for (int j = 0; j < 64; j += 16) {
            // Load 16 bytes and broadcast to both 128-bit lanes.
            __m256i v_raw = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(qs + j)));

            // Unpack 64 elements into 8 registers of 8 int32s each.
            __m256i v0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(v_raw, m01), v_shifts), m3);
            __m256i v1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(v_raw, m23), v_shifts), m3);
            __m256i v2 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(v_raw, m45), v_shifts), m3);
            __m256i v3 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(v_raw, m67), v_shifts), m3);
            __m256i v4 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(v_raw, m89), v_shifts), m3);
            __m256i v5 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(v_raw, mAB), v_shifts), m3);
            __m256i v6 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(v_raw, mCD), v_shifts), m3);
            __m256i v7 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(v_raw, mEF), v_shifts), m3);

            // Convert to float, apply scale and bias using FMA, and store.
            _mm256_storeu_ps(y + 0,  _mm256_fmadd_ps(_mm256_cvtepi32_ps(v0), v_scale, v_bias));
            _mm256_storeu_ps(y + 8,  _mm256_fmadd_ps(_mm256_cvtepi32_ps(v1), v_scale, v_bias));
            _mm256_storeu_ps(y + 16, _mm256_fmadd_ps(_mm256_cvtepi32_ps(v2), v_scale, v_bias));
            _mm256_storeu_ps(y + 24, _mm256_fmadd_ps(_mm256_cvtepi32_ps(v3), v_scale, v_bias));
            _mm256_storeu_ps(y + 32, _mm256_fmadd_ps(_mm256_cvtepi32_ps(v4), v_scale, v_bias));
            _mm256_storeu_ps(y + 40, _mm256_fmadd_ps(_mm256_cvtepi32_ps(v5), v_scale, v_bias));
            _mm256_storeu_ps(y + 48, _mm256_fmadd_ps(_mm256_cvtepi32_ps(v6), v_scale, v_bias));
            _mm256_storeu_ps(y + 56, _mm256_fmadd_ps(_mm256_cvtepi32_ps(v7), v_scale, v_bias));

            y += 64;
        }
    }
}

}
