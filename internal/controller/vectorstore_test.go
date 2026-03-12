package controller

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

func mockEmbedServer(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/embed" {
			var req EmbedRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("Failed to decode embed request: %v", err)
			}

			// Return deterministic embeddings based on content for testing
			var embedding []float64
			switch {
			case contains(req.Input, "distributed"):
				embedding = []float64{0.9, 0.1, 0.2, 0.0}
			case contains(req.Input, "cooking"):
				embedding = []float64{0.0, 0.1, 0.8, 0.9}
			case contains(req.Input, "mesh"):
				embedding = []float64{0.85, 0.15, 0.25, 0.05}
			default:
				embedding = []float64{0.5, 0.5, 0.5, 0.5}
			}

			resp := EmbedResponse{
				Model:      "nomic-embed-text",
				Embeddings: [][]float64{embedding},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsSubstring(s, substr))
}

func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestVectorStoreAddAndSearch(t *testing.T) {
	server := mockEmbedServer(t)
	defer server.Close()

	client := NewOllamaClient(server.URL, "test-model")
	store := NewVectorStore(client)
	ctx := context.Background()

	// Add documents
	if err := store.AddDocument(ctx, "distributed systems are complex", "paper1"); err != nil {
		t.Fatalf("AddDocument failed: %v", err)
	}
	if err := store.AddDocument(ctx, "cooking pasta requires boiling water", "recipe1"); err != nil {
		t.Fatalf("AddDocument failed: %v", err)
	}

	if store.Count() != 2 {
		t.Fatalf("Expected 2 documents, got %d", store.Count())
	}

	// Search for something related to distributed systems
	results, err := store.Search(ctx, "mesh networking topology", 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}

	// The distributed systems doc should rank higher than cooking
	if results[0].Document.Source != "paper1" {
		t.Errorf("Expected 'paper1' as top result, got '%s'", results[0].Document.Source)
	}

	// Similarity should be > 0
	if results[0].Similarity <= 0 {
		t.Errorf("Expected positive similarity, got %f", results[0].Similarity)
	}
}

func TestVectorStorePersistence(t *testing.T) {
	server := mockEmbedServer(t)
	defer server.Close()

	client := NewOllamaClient(server.URL, "test-model")
	store := NewVectorStore(client)
	ctx := context.Background()

	store.AddDocument(ctx, "distributed computing fundamentals", "textbook")

	// Save
	tmpFile := t.TempDir() + "/test_vectors.json"
	if err := store.SaveToFile(tmpFile); err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Fatalf("Expected file to exist at %s", tmpFile)
	}

	// Load into a new store
	store2 := NewVectorStore(client)
	if err := store2.LoadFromFile(tmpFile); err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	if store2.Count() != 1 {
		t.Fatalf("Expected 1 document after load, got %d", store2.Count())
	}
}

func TestOllamaEmbed(t *testing.T) {
	server := mockEmbedServer(t)
	defer server.Close()

	client := NewOllamaClient(server.URL, "test-model")

	embedding, err := client.Embed(context.Background(), "distributed systems")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if len(embedding) != 4 {
		t.Fatalf("Expected 4-dimensional embedding, got %d", len(embedding))
	}

	if embedding[0] != 0.9 {
		t.Errorf("Expected first dimension 0.9, got %f", embedding[0])
	}
}
