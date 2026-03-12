//go:build !darwin
package ggml

/*
#cgo LDFLAGS: -L. -lggml_vextra -lggml -lggml-base -Wl,-rpath,.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml_vextra.h"
#include <stdlib.h>

// Mock for scheinfer_route_task if not linked with Go
// In a real build, Go provides this.
// char * scheinfer_route_task(size_t size) { return strdup("CPU_AVX2"); }
*/
import "C"
import (
	"testing"
	"github.com/groovy-byte/agent-mesh-core/internal/controller"
)

func TestScheInferGGMLIntegration(t *testing.T) {
	// 1. Initialize Go Scheduler
	controller.InitializeGlobalScheduler(16*1024*1024, "NVIDIA GeForce RTX 3070 Laptop GPU", 86, false)

	// 2. Initialize Vextra Backend
	backend := C.ggml_backend_vextra_init()
	if backend == nil {
		t.Fatal("Failed to initialize Vextra backend")
	}
	defer C.ggml_backend_free(backend)

	// 3. Create GGML context
	params := C.struct_ggml_init_params{
		mem_size: 16 * 1024 * 1024,
		no_alloc: false,
	}
	ctx := C.ggml_init(params)
	defer C.ggml_free(ctx)

	// 4. Create a matrix multiplication (to trigger ScheInfer)
	// Small matrix (< 16MB)
	weights := C.ggml_new_tensor_2d(ctx, C.GGML_TYPE_F32, 128, 128)
	input := C.ggml_new_tensor_1d(ctx, C.GGML_TYPE_F32, 128)
	
	gf := C.ggml_new_graph(ctx)
	result := C.ggml_mul_mat(ctx, weights, input)
	C.ggml_build_forward_expand(gf, result)

	// 5. Execute - this calls back into Go's scheinfer_route_task
	status := C.ggml_backend_graph_compute(backend, gf)
	if status != C.GGML_STATUS_SUCCESS {
		t.Errorf("Graph compute failed: %d", status)
	}
	
	// If it didn't crash and returned SUCCESS, the C++ -> Go callback worked!
	t.Log("Integration successful: C++ backend successfully called Go scheduler.")
}
