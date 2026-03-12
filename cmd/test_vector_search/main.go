package main

import (
	"context"
	"fmt"
	"log"

	"github.com/groovy-byte/agent-mesh-core/internal/controller"
)

func main() {
	fmt.Println("=== Vector Store Search Test ===")

	client := controller.NewOllamaClient("", "")
	store := controller.NewVectorStore(client)

	// Load the seeded vectors
	if err := store.LoadFromFile("vectors.json"); err != nil {
		log.Fatalf("Failed to load vectors: %v", err)
	}

	fmt.Printf("Loaded %d documents\n\n", store.Count())

	// Run test queries
	queries := []string{
		"How does task routing work?",
		"What hardware does the Mac use?",
		"How do agents communicate with each other?",
	}

	ctx := context.Background()
	for _, query := range queries {
		fmt.Printf("🔍 Query: %s\n", query)
		results, err := store.Search(ctx, query, 3)
		if err != nil {
			log.Fatalf("Search failed: %v", err)
		}
		for i, r := range results {
			fmt.Printf("   [%d] (%.4f) %s — %.80s\n", i+1, r.Similarity, r.Document.Source, r.Document.Content)
		}
		fmt.Println()
	}
}
