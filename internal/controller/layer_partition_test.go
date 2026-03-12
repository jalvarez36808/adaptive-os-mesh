package controller

import (
	"runtime"
	"testing"
)

func TestLayerPartitioning(t *testing.T) {
	s := NewScheInfer(16*1024*1024, "NVIDIA GeForce RTX 3070 Laptop GPU", 86, false)
	
	// Define model layers
	// Layer 0-15: Vision/Encoding (Heavy GPU usage preferred)
	// Layer 16-31: Reasoning/Decoding (CPU cache preferred for latency)
	
	tests := []struct {
		layerID  int
		expected string
	}{
		{0,  "GPU_CUDA"}, // Vision layer -> Node A GPU
		{10, "GPU_CUDA"},
		{20, "CPU_AVX2"}, // Reasoning layer -> Node A CPU (Low latency reasoning)
		{30, "CPU_AVX2"},
	}

	for _, tt := range tests {
		got := s.RouteLayer(tt.layerID)
		expected := tt.expected
		if runtime.GOOS == "darwin" {
			expected = "MAC_OLLAMA"
		}
		if got != expected {
			t.Errorf("Layer %d routing failed: got %s, want %s", tt.layerID, got, expected)
		}
	}
}
