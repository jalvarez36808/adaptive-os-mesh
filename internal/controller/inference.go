package controller

import (
	"context"
	"fmt"
	"log"
	"time"
	pb "github.com/groovy-byte/agent-mesh-core/proto"
)

// InferenceController handles hardware-aware LLM requests
type InferenceController struct {
	scheduler    *ScheInfer
	ollamaClient *OllamaClient
}

func NewInferenceController(scheduler *ScheInfer) *InferenceController {
	return &InferenceController{
		scheduler:    scheduler,
		ollamaClient: NewOllamaClient("", ""), // Uses default http://localhost:11434 and llama3.2
	}
}

// Generate processes the LLM request by selecting the optimal hardware path
func (c *InferenceController) Generate(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	start := time.Now()
	
	// --- Phase 8: Hardware-Aware Selection ---
	// We use the expected KV cache size or prompt size to decide the routing
	// Prompt size is a good proxy for data transfer impact
	dataSize := uint64(len(req.Prompt))
	if req.ExpectedKvCacheBytes > 0 {
		dataSize = req.ExpectedKvCacheBytes
	}

	hardwarePath := c.scheduler.RouteTask(dataSize)
	log.Printf("[Inference] Request from %s. Size: %d bytes. Path: %s", req.AgentId, dataSize, hardwarePath)

	// --- macOS Ollama Path ---
	if hardwarePath == "MAC_OLLAMA" {
		log.Printf("[Inference] Executing via Ollama...")
		resp, err := c.ollamaClient.Generate(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("failed inference via ollama: %w", err)
		}
		
		return &pb.InferenceResponse{
			Text:          resp.Response,
			TokensUsed:    uint32(resp.EvalCount), // rough equivalent
			HardwarePath:  hardwarePath,
			LatencyMs:     float32(time.Since(start).Milliseconds()),
			ThroughputGbs: 0.0, // Not applicable for external API
			Avx512Usage:   false,
		}, nil
	}

	// Simulation: Actual LLM work
	// In a full implementation, this would call llama.cpp or a Gemini bridge
	simulatedText := fmt.Sprintf("[Hardware: %s] Simulated response for: %s", hardwarePath, req.Prompt)
	
	// Simulate processing time based on path
	latency := 50 * time.Millisecond
	throughput := float32(8.2) // Default AVX2 throughput GB/s
	avx512Used := false

	if hardwarePath == "CPU_AVX512" {
		latency = 35 * time.Millisecond
		throughput = 12.5 // Simulated Tiger Lake AVX512 throughput
		avx512Used = true
	} else if hardwarePath == "GPU_CUDA" || hardwarePath == "GPU_VULKAN" {
		latency = 120 * time.Millisecond // Transfer overhead simulation
		throughput = 25.0
	}
	
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(latency):
	}

	return &pb.InferenceResponse{
		Text:          simulatedText,
		TokensUsed:    uint32(len(simulatedText) / 4), // Rough estimate
		HardwarePath:  hardwarePath,
		LatencyMs:     float32(time.Since(start).Milliseconds()),
		ThroughputGbs: throughput,
		Avx512Usage:   avx512Used,
	}, nil
}
