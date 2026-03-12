package main

import (
	"context"
	"fmt"
	"log"

	"github.com/groovy-byte/agent-mesh-core/internal/controller"
	pb "github.com/groovy-byte/agent-mesh-core/proto"
)

func main() {
	fmt.Println("=== macOS Ollama Integration E2E Test ===")

	// 1. Initialize scheinfer and controller
	// L3 cache 16MB, "Apple M-Series", compute cap 0, avx512 false
	scheduler := controller.NewScheInfer(16*1024*1024, "Apple M-Series", 0, false)
	ctrl := controller.NewInferenceController(scheduler)

	// 2. Create a test request
	// This should route to MAC_OLLAMA and call the local Ollama instance
	req := &pb.InferenceRequest{
		AgentId:     "test-agent-01",
		Prompt:      "Explain the significance of the year 1991 in computer science.",
		MaxTokens:   50,
		Temperature: 0.7,
	}

	fmt.Printf("Sending request to inference controller...\n")
	resp, err := ctrl.Generate(context.Background(), req)
	if err != nil {
		log.Fatalf("Generate failed: %v", err)
	}

	fmt.Printf("\n--- Response ---\n")
	fmt.Printf("Hardware Path: %s\n", resp.HardwarePath)
	fmt.Printf("Tokens Used: %d\n", resp.TokensUsed)
	fmt.Printf("Latency: %.2f ms\n", resp.LatencyMs)
	fmt.Printf("Text:\n%s\n", resp.Text)
}
