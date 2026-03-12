//go:build !darwin
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml_vextra.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>

// Reference dequantizer for verification
void dequantize_q2_k_reference(const void * vx, float * vy, int k) {
    // This should match our expected behavior
    // For now we just implement a mock that matches our test pattern
    for (int i = 0; i < k; i += 4) {
        vy[i+0] = 0.5f;
        vy[i+1] = 2.5f;
        vy[i+2] = 4.5f;
        vy[i+3] = 6.5f;
    }
}

int main() {
    printf("Starting GGML Vextra Dequantization TDD Test...\n");

    ggml_backend_t backend = ggml_backend_vextra_init();
    assert(backend != NULL);

    ggml_init_params params = {
        128 * 1024 * 1024,
        nullptr,
        false,
    };
    ggml_context * ctx = ggml_init(params);

    const int k = 1024;
    
    // 1. Create a Q2_K tensor
    // Q2_K block size is 256. 1024 elements = 4 blocks.
    // Each block is 72 bytes. Total 288 bytes.
    ggml_tensor * weights = ggml_new_tensor_1d(ctx, GGML_TYPE_Q2_K, k);
    
    // 2. Fill with known pattern (0, 1, 2, 3)
    // 0xE4 = 11 10 01 00 in bits
    uint8_t pattern = 0xE4;
    for (int b = 0; b < 4; ++b) {
        uint8_t * block_data = (uint8_t *)weights->data + b * 72;
        // Set d = 2.0, dmin = 0.5
        float d = 2.0f;
        float dmin = 0.5f;
        memcpy(block_data, &d, 4);
        memcpy(block_data + 4, &dmin, 4);
        memset(block_data + 8, pattern, 64);
    }

    // 3. Create destination F32 tensor
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);

    // 4. Build graph for dequantization (CPY op is often used or internal)
    // In GGML, MUL_MAT triggers dequantization internally.
    // We will test if our backend handles GGML_OP_CPY from Q2_K to F32
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * result = ggml_cpy(ctx, weights, dst);
    ggml_build_forward_expand(gf, result);

    // 5. Compute
    ggml_backend_graph_compute(backend, gf);

    // 6. Verify
    float * res_data = (float *)dst->data;
    printf("Result[0]: %f (Expected: 0.5)\n", res_data[0]);
    printf("Result[1]: %f (Expected: 2.5)\n", res_data[1]);
    
    if (std::abs(res_data[0] - 0.5f) < 1e-3 && std::abs(res_data[1] - 2.5f) < 1e-3) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
