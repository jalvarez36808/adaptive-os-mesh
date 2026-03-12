package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"sync"
)

// Document represents a single entry in the vector store
type Document struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	Source    string    `json:"source"`
	Embedding []float64 `json:"embedding"`
}

// SearchResult represents a ranked search hit
type VectorSearchResult struct {
	Document   Document
	Similarity float64
}

// VectorStore is a thread-safe, in-memory vector database
type VectorStore struct {
	mu        sync.RWMutex
	documents []Document
	client    *OllamaClient
	nextID    int
}

// NewVectorStore creates a new empty vector store
func NewVectorStore(client *OllamaClient) *VectorStore {
	return &VectorStore{
		documents: []Document{},
		client:    client,
		nextID:    1,
	}
}

// AddDocument embeds text via Ollama and stores it in the vector store
func (vs *VectorStore) AddDocument(ctx context.Context, content, source string) error {
	embedding, err := vs.client.Embed(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to embed document: %w", err)
	}

	vs.mu.Lock()
	defer vs.mu.Unlock()

	doc := Document{
		ID:        fmt.Sprintf("doc_%d", vs.nextID),
		Content:   content,
		Source:    source,
		Embedding: embedding,
	}
	vs.nextID++
	vs.documents = append(vs.documents, doc)

	log.Printf("[VectorStore] Added document %s from '%s' (%d dimensions)", doc.ID, source, len(embedding))
	return nil
}

// Search finds the top-K most similar documents to the query
func (vs *VectorStore) Search(ctx context.Context, query string, topK int) ([]VectorSearchResult, error) {
	queryEmbedding, err := vs.client.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	vs.mu.RLock()
	defer vs.mu.RUnlock()

	if len(vs.documents) == 0 {
		return nil, nil
	}

	results := make([]VectorSearchResult, 0, len(vs.documents))
	for _, doc := range vs.documents {
		sim := cosineSimilarity(queryEmbedding, doc.Embedding)
		results = append(results, VectorSearchResult{
			Document:   doc,
			Similarity: sim,
		})
	}

	// Sort by similarity descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if topK > len(results) {
		topK = len(results)
	}

	return results[:topK], nil
}

// Count returns the number of documents in the store
func (vs *VectorStore) Count() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.documents)
}

// SaveToFile persists the vector store to a JSON file
func (vs *VectorStore) SaveToFile(path string) error {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	data, err := json.MarshalIndent(vs.documents, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal vector store: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write vector store file: %w", err)
	}

	log.Printf("[VectorStore] Saved %d documents to %s", len(vs.documents), path)
	return nil
}

// LoadFromFile loads a previously saved vector store from a JSON file
func (vs *VectorStore) LoadFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read vector store file: %w", err)
	}

	var docs []Document
	if err := json.Unmarshal(data, &docs); err != nil {
		return fmt.Errorf("failed to unmarshal vector store: %w", err)
	}

	vs.mu.Lock()
	defer vs.mu.Unlock()

	vs.documents = docs
	// Set nextID to be past the highest existing ID
	vs.nextID = len(docs) + 1

	log.Printf("[VectorStore] Loaded %d documents from %s", len(docs), path)
	return nil
}

// cosineSimilarity computes the cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	denominator := math.Sqrt(normA) * math.Sqrt(normB)
	if denominator == 0 {
		return 0
	}

	return dotProduct / denominator
}
