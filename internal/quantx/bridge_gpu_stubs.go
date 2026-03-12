//go:build !cuda && !darwin

package quantx

import "errors"

/**
 * GetGpuInfo returns stubs when CUDA is not enabled.
 */
func GetGpuInfo() (string, uint64, error) {
	return "No GPU (CUDA disabled)", 0, errors.New("cuda: build tag not provided")
}

/**
 * GetGpuComputeCapability returns 0 when CUDA is not enabled.
 */
func GetGpuComputeCapability() int {
	return 0
}

/**
 * DequantizeQ2KCUDA returns an error when CUDA is not enabled.
 */
func DequantizeQ2KCUDA(vx interface{}, vy []float32, k int) error {
	return errors.New("cuda: build tag not provided")
}
