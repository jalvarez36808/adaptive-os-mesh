//go:build !darwin
#include "quantx.h"
#include <immintrin.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl")
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
 * dequantize_q2_k_avx512 dequantizes Q2_K blocks to float32 using AVX512.
 * Optimized for Tiger Lake by using 512-bit registers and broadcast/shuffle strategy.
 */
void dequantize_q2_k_avx512(const void* vx, float* vy, int k) {
    const block_q2_k* x = (const block_q2_k*)vx;
    const int nb = k / 256;

    // Constants for bit extraction and expansion
    const __m512i m3 = _mm512_set1_epi32(0x03);
    const __m512i v_shifts = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);

    // Pre-defined masks for _mm512_shuffle_epi8 to expand 16 bytes into 64 int32s.
    // Each 128-bit lane of the mask picks one byte from the broadcasted 16-byte source.
    // Using _mm512_set_epi8 (arguments are in reverse order compared to _mm512_setr_epi8).
    const __m512i m0123 = _mm512_set_epi8(
        3,3,3,3, 3,3,3,3, 3,3,3,3, 3,3,3,3,
        2,2,2,2, 2,2,2,2, 2,2,2,2, 2,2,2,2,
        1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
        0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
    );
    const __m512i m4567 = _mm512_set_epi8(
        7,7,7,7, 7,7,7,7, 7,7,7,7, 7,7,7,7,
        6,6,6,6, 6,6,6,6, 6,6,6,6, 6,6,6,6,
        5,5,5,5, 5,5,5,5, 5,5,5,5, 5,5,5,5,
        4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4
    );
    const __m512i m89AB = _mm512_set_epi8(
        11,11,11,11, 11,11,11,11, 11,11,11,11, 11,11,11,11,
        10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10,
        9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9,
        8,8,8,8, 8,8,8,8, 8,8,8,8, 8,8,8,8
    );
    const __m512i mCDEF = _mm512_set_epi8(
        15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
        14,14,14,14, 14,14,14,14, 14,14,14,14, 14,14,14,14,
        13,13,13,13, 13,13,13,13, 13,13,13,13, 13,13,13,13,
        12,12,12,12, 12,12,12,12, 12,12,12,12, 12,12,12,12
    );

    for (int i = 0; i < nb; i++) {
        const __m512 v_scale = _mm512_set1_ps(x[i].d);
        const __m512 v_bias  = _mm512_set1_ps(x[i].dmin);
        const uint8_t* qs = x[i].qs;
        float* y = vy + i * 256;

        // Process 256 elements in 4 groups of 64.
        for (int j = 0; j < 64; j += 16) {
            // Load 16 bytes and broadcast to all 4 lanes of 512-bit register.
            __m512i v_raw = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)(qs + j)));

            // Unpack 64 elements into 4 registers of 16 int32s each.
            __m512i v0 = _mm512_and_si512(_mm512_srlv_epi32(_mm512_shuffle_epi8(v_raw, m0123), v_shifts), m3);
            __m512i v1 = _mm512_and_si512(_mm512_srlv_epi32(_mm512_shuffle_epi8(v_raw, m4567), v_shifts), m3);
            __m512i v2 = _mm512_and_si512(_mm512_srlv_epi32(_mm512_shuffle_epi8(v_raw, m89AB), v_shifts), m3);
            __m512i v3 = _mm512_and_si512(_mm512_srlv_epi32(_mm512_shuffle_epi8(v_raw, mCDEF), v_shifts), m3);

            // Convert to float, apply scale and bias using FMA, and store.
            _mm512_storeu_ps(y + 0,  _mm512_fmadd_ps(_mm512_cvtepi32_ps(v0), v_scale, v_bias));
            _mm512_storeu_ps(y + 16, _mm512_fmadd_ps(_mm512_cvtepi32_ps(v1), v_scale, v_bias));
            _mm512_storeu_ps(y + 32, _mm512_fmadd_ps(_mm512_cvtepi32_ps(v2), v_scale, v_bias));
            _mm512_storeu_ps(y + 48, _mm512_fmadd_ps(_mm512_cvtepi32_ps(v3), v_scale, v_bias));

            y += 64;
        }
    }
}

}
