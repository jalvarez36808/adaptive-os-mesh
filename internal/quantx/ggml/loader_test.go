//go:build !darwin
package ggml

import (
	"testing"
)

func TestLoadModel(t *testing.T) {
	model, err := LoadModel("non_existent.gguf")
	if err == nil {
		t.Fatal("Expected error for non-existent file")
	}
	if model != nil {
		t.Fatal("Model should be nil on error")
	}
}
