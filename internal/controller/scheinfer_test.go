package controller

import (
	"runtime"
	"testing"
)

func TestScheInferRouting(t *testing.T) {
	// Mock 16MB L3 cache
	l3Size := uint64(16 * 1024 * 1024)
	
	tests := []struct {
		name       string
		gpuName    string
		computeCap int
		avx512     bool
		dataSize   uint64
		expected   string
	}{
		{
			name:       "Small task on Ampere",
			gpuName:    "NVIDIA GeForce RTX 3070 Laptop GPU",
			computeCap: 86,
			dataSize:   1024 * 1024, // 1MB
			expected:   "CPU_AVX2",
		},
		{
			name:       "Large task on Ampere",
			gpuName:    "NVIDIA GeForce RTX 3070 Laptop GPU",
			computeCap: 86,
			dataSize:   32 * 1024 * 1024, // 32MB
			expected:   "GPU_CUDA",
		},
		{
			name:       "Small task on Pascal",
			gpuName:    "NVIDIA GeForce GTX 1070",
			computeCap: 61,
			dataSize:   1024 * 1024, // 1MB
			expected:   "CPU_AVX2",
		},
		{
			name:       "Large task on Pascal",
			gpuName:    "NVIDIA GeForce GTX 1070",
			computeCap: 61,
			dataSize:   32 * 1024 * 1024, // 32MB
			expected:   "GPU_VULKAN",
		},
		{
			name:       "Large task on AVX-512 (No GPU)",
			gpuName:    "",
			computeCap: 0,
			avx512:     true,
			dataSize:   32 * 1024 * 1024, // 32MB
			expected:   "CPU_AVX512",
		},
		{
			name:       "Large task on No GPU",
			gpuName:    "",
			computeCap: 0,
			avx512:     false,
			dataSize:   32 * 1024 * 1024, // 32MB
			expected:   "CPU_AVX2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewScheInfer(l3Size, tt.gpuName, tt.computeCap, tt.avx512)
			got := s.RouteTask(tt.dataSize)
			
			// Override expectations if tests are running on macOS where MAC_OLLAMA is forced
			expected := tt.expected
			if runtime.GOOS == "darwin" {
				expected = "MAC_OLLAMA"
			}

			if got != expected {
				t.Errorf("RouteTask() = %v, want %v", got, expected)
			}
		})
	}
}
