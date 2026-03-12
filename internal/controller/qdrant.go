package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	pb "github.com/groovy-byte/agent-mesh-core/proto"
)

// SoftThrottle implements the Value of Coordination (VoC) logic
type SoftThrottle struct {
	mu             sync.RWMutex
	recentSearches map[string]time.Time
	window         time.Duration
}

func NewSoftThrottle() *SoftThrottle {
	return &SoftThrottle{
		recentSearches: make(map[string]time.Time),
		window:         5 * time.Second,
	}
}

func (t *SoftThrottle) IsNovel(query string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	lastTime, exists := t.recentSearches[query]
	if exists && time.Since(lastTime) < t.window {
		return false
	}
	t.recentSearches[query] = time.Now()
	return true
}

type QdrantController struct {
	throttle    *SoftThrottle
	serviceURL  string
	vectorStore *VectorStore
}

func NewQdrantController() *QdrantController {
	client := NewOllamaClient("", "")
	vs := NewVectorStore(client)

	// Try to load persisted vectors on startup
	storePath := "vectors.json"
	if _, err := os.Stat(storePath); err == nil {
		if err := vs.LoadFromFile(storePath); err != nil {
			log.Printf("[Qdrant] Warning: could not load vector store: %v", err)
		}
	}

	return &QdrantController{
		throttle:    NewSoftThrottle(),
		serviceURL:  "http://127.0.0.1:5000/search",
		vectorStore: vs,
	}
}

func (q *QdrantController) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	if !q.throttle.IsNovel(req.Query) {
		return &pb.SearchResponse{
			ReasoningContext: "THROTTLED: Redundant semantic search detected.",
		}, nil
	}

	// On macOS, use the in-process vector store if populated
	if runtime.GOOS == "darwin" && q.vectorStore.Count() > 0 {
		log.Printf("[Mesh] 🔍 Vector Search (In-Process): %s", req.Query)

		maxResults := int(req.MaxResults)
		if maxResults == 0 {
			maxResults = 3
		}

		results, err := q.vectorStore.Search(ctx, req.Query, maxResults)
		if err != nil {
			log.Printf("[Mesh] ⚠️ Vector search failed, falling back: %v", err)
			return q.performGrepFallback(req.Query)
		}

		pbResults := make([]*pb.SearchResult, len(results))
		for i, r := range results {
			pbResults[i] = &pb.SearchResult{
				Source:  r.Document.Source,
				Content: r.Document.Content,
				Score:   float32(r.Similarity),
			}
		}

		return &pb.SearchResponse{
			Results:          pbResults,
			ReasoningContext: "Grounded in local vector store (Ollama nomic-embed-text).",
		}, nil
	}

	log.Printf("[Mesh] 🔍 Grounded Search (Service): %s", req.Query)

	searchReq := map[string]interface{}{
		"query":       req.Query,
		"limit":       req.MaxResults,
		"collections": []string{"llama_research", "research_corpus"},
	}
	
	jsonBody, err := json.Marshal(searchReq)
	if err != nil {
		return nil, fmt.Errorf("marshal failed: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", q.serviceURL, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("request creation failed: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{
		Timeout: 2 * time.Second, // Aggressive timeout for fast fallback
	}
	resp, err := client.Do(httpReq)
	
	// Fallback if service is down or errors
	if err != nil || (resp != nil && resp.StatusCode != http.StatusOK) {
		log.Printf("[Mesh] ⚠️ Qdrant Service unavailable, falling back to local grep search.")
		return q.performGrepFallback(req.Query)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return q.performGrepFallback(req.Query)
	}

	var results []struct {
		Source  string  `json:"source"`
		Content string  `json:"content"`
		Score   float32 `json:"score"`
	}
	if err := json.Unmarshal(body, &results); err != nil {
		return q.performGrepFallback(req.Query)
	}

	pbResults := make([]*pb.SearchResult, len(results))
	for i, r := range results {
		pbResults[i] = &pb.SearchResult{
			Source:  r.Source,
			Content: r.Content,
			Score:   r.Score,
		}
	}

	return &pb.SearchResponse{
		Results:          pbResults,
		ReasoningContext: "Grounded in local Qdrant service.",
	}, nil
}

func (q *QdrantController) performGrepFallback(query string) (*pb.SearchResponse, error) {
	// Extract keywords for grep
	words := strings.Fields(query)
	keyword := "mesh"
	if len(words) > 0 {
		keyword = words[0]
	}

	cmd := exec.Command("grep", "-ri", "-m", "5", keyword, "/home/groovy-byte/agent-mesh-core/local_research")
	out, _ := cmd.CombinedOutput()
	
	lines := strings.Split(string(out), "\n")
	results := []*pb.SearchResult{}
	
	for _, line := range lines {
		if line == "" { continue }
		parts := strings.SplitN(line, ":", 2)
		source := "Research Corpus"
		content := line
		if len(parts) == 2 {
			source = parts[0]
			content = parts[1]
		}
		
		results = append(results, &pb.SearchResult{
			Source:  source,
			Content: strings.TrimSpace(content),
			Score:   0.5,
		})
		if len(results) >= 3 { break }
	}

	if len(results) == 0 {
		results = append(results, &pb.SearchResult{
			Source:  "Operational Fallback",
			Content: "No direct keyword matches in local cache. Escalating to base reasoning.",
			Score:   0.3,
		})
	}

	return &pb.SearchResponse{
		Results:          results,
		ReasoningContext: "Grounded via Local Grep Fallback (Qdrant Offline).",
	}, nil
}
