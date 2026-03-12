//go:build !darwin
package ggml

/*
#cgo LDFLAGS: -lggml -lggml-base
#include "ggml.h"
#include "gguf.h"
#include "ggml_vextra.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type GGUFModel struct {
	ctx  *C.struct_ggml_context
	gctx *C.struct_gguf_context
}

func LoadModel(path string) (*GGUFModel, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var ggmlCtx *C.struct_ggml_context
	params := C.struct_gguf_init_params{
		no_alloc: false,
		ctx:      &ggmlCtx,
	}

	gctx := C.gguf_init_from_file(cPath, params)
	if gctx == nil {
		return nil, fmt.Errorf("failed to load GGUF model from %s", path)
	}

	return &GGUFModel{
		ctx:  ggmlCtx,
		gctx: gctx,
	}, nil
}

func (m *GGUFModel) Free() {
	if m.gctx != nil {
		C.gguf_free(m.gctx)
	}
	if m.ctx != nil {
		C.ggml_free(m.ctx)
	}
}

func (m *GGUFModel) GetTensorCount() int {
	return int(C.gguf_get_n_tensors(m.gctx))
}
