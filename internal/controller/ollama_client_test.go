package controller

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	pb "github.com/groovy-byte/agent-mesh-core/proto"
)

func TestOllamaClient_Generate(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("Expected POST request, got %s", r.Method)
		}
		if r.URL.Path != "/api/generate" {
			t.Errorf("Expected path /api/generate, got %s", r.URL.Path)
		}

		var req OllamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("Failed to decode request body: %v", err)
		}

		if req.Prompt != "Test prompt" {
			t.Errorf("Expected prompt 'Test prompt', got '%s'", req.Prompt)
		}
		if req.Options.Temperature != 0.5 {
			t.Errorf("Expected temperature 0.5, got %f", req.Options.Temperature)
		}
		if req.Options.NumPredict != 50 {
			t.Errorf("Expected num_predict 50, got %d", req.Options.NumPredict)
		}

		resp := OllamaResponse{
			Model:     "test-model",
			Response:  "Test response",
			Done:      true,
			EvalCount: 10,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer mockServer.Close()

	client := NewOllamaClient(mockServer.URL, "test-model")

	req := &pb.InferenceRequest{
		Prompt:      "Test prompt",
		MaxTokens:   50,
		Temperature: 0.5,
	}

	resp, err := client.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.Response != "Test response" {
		t.Errorf("Expected response 'Test response', got '%s'", resp.Response)
	}
	if resp.EvalCount != 10 {
		t.Errorf("Expected EvalCount 10, got %d", resp.EvalCount)
	}
}
