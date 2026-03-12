//go:build darwin

package quantx

import "errors"

// BlockQ2K represents a quantized 2-bit block.
type BlockQ2K struct {
	D    float32
	Dmin float32
	Qs   [64]uint8
}

// HasAVX512 returns false on Darwin as Apple Silicon doesn't support AVX.
func HasAVX512() bool {
	return false
}

// DequantizeQ2K panics on macOS because local quantx kernels shouldn't be executed here.
func DequantizeQ2K(vx interface{}, vy []float32, k int) {
	panic("quantx: DequantizeQ2K should not be called on macOS. Route through Ollama instead.")
}

// GetGpuInfo returns stubs since we bypass bare-metal GPU control on Mac.
func GetGpuInfo() (string, uint64, error) {
	return "Apple Silicon (Ollama Managed)", 0, nil
}

// GetGpuComputeCapability returns 0 indicating no direct CUDA capability.
func GetGpuComputeCapability() int {
	return 0
}

// DequantizeQ2KCUDA returns an error since CUDA is not present on macOS.
func DequantizeQ2KCUDA(vx interface{}, vy []float32, k int) error {
	return errors.New("cuda: build tag not provided on darwin")
}
