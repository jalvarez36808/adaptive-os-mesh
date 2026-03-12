#include "quantx.h"
//go:build !darwin
#include <cuda_runtime.h>
#include <stdint.h>

typedef struct {
    float d;
    float dmin;
    uint8_t qs[64];
} block_q2_k;

/**
 * dequantize_q2_k_kernel is a basic CUDA kernel for Q2_K dequantization.
 * Each thread processes 8 elements (2 bytes) from a block.
 * A warp (32 threads) processes one full block of 256 elements.
 */
__global__ void dequantize_q2_k_kernel(const block_q2_k* __restrict__ x, float* __restrict__ y, int k) {
    int i = blockIdx.x;
    int tid = threadIdx.x; // 0-31

    if (i * 256 >= k) return;

    // Use shared memory for scale and bias to avoid redundant global memory reads
    // and take advantage of fast shared memory broadcast on Ampere.
    __shared__ float s_d;
    __shared__ float s_dmin;

    if (tid == 0) {
        s_d = x[i].d;
        s_dmin = x[i].dmin;
    }
    __syncthreads();

    float d = s_d;
    float dmin = s_dmin;

    // Each thread processes 8 elements (2 bytes)
    // Load 2 bytes at once as uint16_t using read-only cache
    const uint16_t* qs16 = (const uint16_t*)x[i].qs;
    uint16_t q16 = __ldg(&qs16[tid]);

    uint8_t q0 = q16 & 0xFF;
    uint8_t q1 = q16 >> 8;

    float4 out0, out1;

    // Extract 4 elements from first byte
    // Ampere's FFMA is very fast, so we keep the math simple
    out0.x = (float)(q0 & 0x03) * d + dmin;
    out0.y = (float)((q0 >> 2) & 0x03) * d + dmin;
    out0.z = (float)((q0 >> 4) & 0x03) * d + dmin;
    out0.w = (float)((q0 >> 6) & 0x03) * d + dmin;

    // Extract 4 elements from second byte
    out1.x = (float)(q1 & 0x03) * d + dmin;
    out1.y = (float)((q1 >> 2) & 0x03) * d + dmin;
    out1.z = (float)((q1 >> 4) & 0x03) * d + dmin;
    out1.w = (float)((q1 >> 6) & 0x03) * d + dmin;

    // Vectorized store of 8 floats (32 bytes per thread)
    // This ensures coalesced writes to global memory
    float4* y4 = (float4*)(y + i * 256 + tid * 8);
    y4[0] = out0;
    y4[1] = out1;
}

extern "C" {

/**
 * dequantize_q2_k_cuda launches the dequantization kernel on the GPU.
 * Assumes vx and vy are already on the device.
 */
void dequantize_q2_k_cuda(const void* vx, float* vy, int k) {
    const block_q2_k* d_x = (const block_q2_k*)vx;
    float* d_y = vy;

    int nb = k / 256;
    if (nb == 0) return;

    // Launch with one warp per block
    dequantize_q2_k_kernel<<<nb, 32>>>(d_x, d_y, k);
}

}
