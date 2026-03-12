package controller

import "C"
import (
	"log"
	"runtime"
	"fmt"
)

var globalScheduler *ScheInfer

// InitializeGlobalScheduler sets the scheduler used by the C backend
func InitializeGlobalScheduler(l3Size uint64, gpuName string, computeCap int, avx512 bool) {
	globalScheduler = NewScheInfer(l3Size, gpuName, computeCap, avx512)
}

//export scheinfer_route_task
func scheinfer_route_task(dataSizeBytes uint64) *C.char {
	if globalScheduler == nil {
		return C.CString("CPU_AVX2")
	}
	provider := globalScheduler.RouteTask(dataSizeBytes)
	return C.CString(provider)
}

// ScheInfer handles intelligent task routing based on hardware topology
type ScheInfer struct {
	l3CacheSize uint64
	hasCuda     bool
	hasVulkan   bool
	hasAvx512   bool
	gpuName     string
}

func NewScheInfer(l3Size uint64, gpuName string, computeCap int, avx512 bool) *ScheInfer {
	return &ScheInfer{
		l3CacheSize: l3Size,
		gpuName:     gpuName,
		hasAvx512:   avx512,
		// Ampere (7.0+) or better preferred for CUDA path
		hasCuda:     gpuName != "" && computeCap >= 70,
		// Pascal/Turing fallback to Vulkan
		hasVulkan:   gpuName != "" && computeCap >= 60 && computeCap < 70,
	}
}

// RouteTask determines the optimal execution provider for a given tensor size.
func (s *ScheInfer) RouteTask(dataSizeBytes uint64) string {
	// If running on macOS, bypass all low-level hardware inference logic 
	// and route directly to the local Ollama API.
	if runtime.GOOS == "darwin" {
		log.Printf("[ScheInfer] macOS Node detected: Routing to MAC_OLLAMA")
		return "MAC_OLLAMA"
	}

	// If the data fits within the CPU's L3 cache, route to CPU to avoid PCIe transfer overhead.
	if dataSizeBytes < s.l3CacheSize {
		log.Printf("[ScheInfer] Cache-Resident Task (%d KB): Routing to CPU AVX2", dataSizeBytes/1024)
		return "CPU_AVX2"
	}

	// For larger tensors, prefer a GPU if an appropriate one is available.
	if s.hasCuda {
		log.Printf("[ScheInfer] High-Throughput Task (%d MB): Routing to NVIDIA CUDA (Ampere+)", dataSizeBytes/(1024*1024))
		return "GPU_CUDA"
	}

	if s.hasVulkan {
		log.Printf("[ScheInfer] Legacy GPU Task (%d MB): Routing to Vulkan", dataSizeBytes/(1024*1024))
		return "GPU_VULKAN"
	}

	// If no suitable GPU is found, use the high-performance AVX-512 CPU path as a fallback.
	if s.hasAvx512 {
		log.Printf("[ScheInfer] AVX-512 Optimized Path (%d MB): Routing to Vector Tier", dataSizeBytes/(1024*1024))
		return "CPU_AVX512"
	}

	log.Printf("[ScheInfer] DRAM-Bound Task (%d MB): Routing to standard CPU path", dataSizeBytes/(1024*1024))
	return "CPU_AVX2"
}

// RouteLayer implements Smart Layer Partitioning (Task 10.2)
func (s *ScheInfer) RouteLayer(layerID int) string {
	if runtime.GOOS == "darwin" {
		return "MAC_OLLAMA"
	}

	// Heuristic: Pin first half of layers to GPU (Compute-heavy), 
	// second half to CPU (Latency-sensitive reasoning)
	if layerID < 16 {
		if s.hasCuda { return "GPU_CUDA" }
	}
	
	// Routing to CPU for late-stage reasoning/decoding
	return "CPU_AVX2"
}

// GetMeshCapability returns the node's performance profile for the Global Mesh
func (s *ScheInfer) GetMeshCapability() string {
	if runtime.GOOS == "darwin" {
		return "MAC:OLLAMA"
	}

	profile := fmt.Sprintf("CPU:%d cores", runtime.NumCPU())
	if s.hasCuda {
		profile += " | GPU:CUDA(Ampere)"
	} else if s.hasAvx512 {
		profile += " | CPU:AVX-512"
	}
	return profile
}
