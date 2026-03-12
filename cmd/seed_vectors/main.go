package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/groovy-byte/agent-mesh-core/internal/controller"
)

func main() {
	fmt.Println("=== Vector Store Seeder ===")

	client := controller.NewOllamaClient("", "")
	store := controller.NewVectorStore(client)
	ctx := context.Background()

	storePath := "vectors.json"

	// Load existing vectors if they exist
	if _, err := os.Stat(storePath); err == nil {
		if err := store.LoadFromFile(storePath); err != nil {
			log.Printf("Warning: could not load existing store: %v", err)
		}
	}

	// Seed documents about the Adaptive OS Mesh project
	documents := []struct {
		content string
		source  string
	}{
		{
			content: "The Adaptive OS Mesh is a hardware-aware distributed agent framework designed for high-performance inference and cross-node orchestration.",
			source:  "project_overview",
		},
		{
			content: "ScheInfer handles intelligent task routing based on hardware topology, including L3 cache size, GPU availability, and SIMD capabilities.",
			source:  "scheinfer_docs",
		},
		{
			content: "On macOS, the mesh routes all inference tasks through a MAC_OLLAMA hardware path, delegating to a local Ollama instance via HTTP REST API.",
			source:  "mac_integration",
		},
		{
			content: "The mesh controller uses gRPC for strategic reasoning tasks and NATS for fast operational heartbeats between agents.",
			source:  "communication_layer",
		},
		{
			content: "Agents in the mesh have roles: OPERATIONAL for fast-path NATS tasks, and STRATEGIC for gRPC-based reasoning and planning.",
			source:  "agent_roles",
		},
		{
			content: "The Value of Contribution metric tracks how much each agent influences other agents in the mesh network.",
			source:  "voc_metrics",
		},
		{
			content: "Quantization uses 2-bit Q2_K format with optimized AVX2 and AVX512 SIMD kernels for dequantization on x86 Linux nodes.",
			source:  "quantx_docs",
		},
		{
			content: "The KV Cache manager implements a ring buffer for efficient key-value cache management during autoregressive LLM inference.",
			source:  "kv_cache_docs",
		},
	}

	fmt.Printf("Seeding %d documents...\n\n", len(documents))

	for i, doc := range documents {
		fmt.Printf("[%d/%d] Embedding: %.60s...\n", i+1, len(documents), doc.content)
		if err := store.AddDocument(ctx, doc.content, doc.source); err != nil {
			log.Fatalf("Failed to seed document: %v", err)
		}
	}

	// Save to disk
	if err := store.SaveToFile(storePath); err != nil {
		log.Fatalf("Failed to save vector store: %v", err)
	}

	fmt.Printf("\n✅ Seeded %d documents and saved to %s\n", store.Count(), storePath)
}
