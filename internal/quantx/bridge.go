//go:build !darwin

package quantx

/*
#cgo CFLAGS: -I. -O3 -mavx2
#cgo CXXFLAGS: -I. -O3 -mavx2 -std=c++11
#include "quantx.h"
*/
import "C"

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

// BlockQ2K represents a quantized 2-bit block.
type BlockQ2K struct {
	D    float32
	Dmin float32
	Qs   [64]uint8
}

// hasAVX512 detects if the CPU supports both AVX512 Foundation and Byte/Word instructions.
// Tiger Lake and newer CPUs support these, enabling 512-bit SIMD dequantization.
var hasAVX512 = cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW

// HasAVX512 returns true if the current CPU supports AVX512 Foundation and Byte/Word instructions.
func HasAVX512() bool {
	return hasAVX512
}

/**
 * DequantizeQ2K wraps the C++ dequantization kernels.
 * It dynamically selects the best implementation (AVX512 or AVX2) based on CPU capabilities.
 * @param vx Pointer to the quantized blocks.
 * @param vy Output float32 slice.
 * @param k Number of elements to dequantize.
 */
func DequantizeQ2K(vx unsafe.Pointer, vy []float32, k int) {
	if len(vy) < k {
		panic("quantx: output buffer size is less than k")
	}
	
	if hasAVX512 {
		C.dequantize_q2_k_avx512(
			vx,
			(*C.float)(unsafe.Pointer(&vy[0])),
			C.int(k),
		)
	} else {
		C.dequantize_q2_k_avx2(
			vx,
			(*C.float)(unsafe.Pointer(&vy[0])),
			C.int(k),
		)
	}
}
