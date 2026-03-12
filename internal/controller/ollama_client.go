package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	pb "github.com/groovy-byte/agent-mesh-core/proto"
)

// OllamaClient provides an interface to communicate with a local Ollama instance
type OllamaClient struct {
	BaseURL    string
	Model      string
	HTTPClient *http.Client
}

// NewOllamaClient initializes a new client for Ollama
func NewOllamaClient(baseURL, model string) *OllamaClient {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	if model == "" {
		model = "qwen3.5" // Default model, can be overridden by environment or config
	}
	return &OllamaClient{
		BaseURL: baseURL,
		Model:   model,
		HTTPClient: &http.Client{
			// Setting a lenient timeout as LLM generation can take tens of seconds
			Timeout: 120 * time.Second,
		},
	}
}

// OllamaRequest represents the expected payload for the /api/generate endpoint
type OllamaRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	Stream  bool           `json:"stream"`
	Options *OllamaOptions `json:"options,omitempty"`
}

// OllamaOptions contains generation knobs like temperature and max tokens
type OllamaOptions struct {
	Temperature float32 `json:"temperature,omitempty"`
	NumPredict  uint32  `json:"num_predict,omitempty"`
}

// OllamaResponse represents the JSON response from /api/generate
type OllamaResponse struct {
	Model              string `json:"model"`
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int64  `json:"eval_duration"`
}

// Generate sends an InferenceRequest to Ollama and returns the parsed response
func (c *OllamaClient) Generate(ctx context.Context, req *pb.InferenceRequest) (*OllamaResponse, error) {
	url := fmt.Sprintf("%s/api/generate", c.BaseURL)

	ollamaReq := OllamaRequest{
		Model:  c.Model,
		Prompt: req.Prompt,
		Stream: false, // We use single-shot REST response for simpler architecture right now
	}

	if req.Temperature > 0 || req.MaxTokens > 0 {
		ollamaReq.Options = &OllamaOptions{
			Temperature: req.Temperature,
			NumPredict:  req.MaxTokens,
		}
	}

	reqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ollama request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama http request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode ollama response: %w", err)
	}

	return &ollamaResp, nil
}

// --- Embedding Support ---

// EmbedRequest represents the payload for the /api/embed endpoint
type EmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// EmbedResponse represents the JSON response from /api/embed
type EmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float64 `json:"embeddings"`
}

// Embed generates vector embeddings for the given text using the nomic-embed-text model
func (c *OllamaClient) Embed(ctx context.Context, text string) ([]float64, error) {
	url := fmt.Sprintf("%s/api/embed", c.BaseURL)

	embedReq := EmbedRequest{
		Model: "nomic-embed-text",
		Input: text,
	}

	reqBody, err := json.Marshal(embedReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embed request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create embed http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama embed request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama embed returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var embedResp EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, fmt.Errorf("failed to decode embed response: %w", err)
	}

	if len(embedResp.Embeddings) == 0 {
		return nil, fmt.Errorf("ollama returned no embeddings")
	}

	return embedResp.Embeddings[0], nil
}
