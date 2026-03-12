//go:build !darwin
#include "ggml.h"
#include <cstdio>

int main() {
    printf("Testing GGML link...\n");
    struct ggml_init_params params = {
        16 * 1024 * 1024,
        NULL,
        false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ggml_init() failed\n");
        return 1;
    }
    printf("GGML context initialized successfully.\n");
    ggml_free(ctx);
    return 0;
}
