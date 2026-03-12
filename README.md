# Adaptive OS Mesh — macOS (Apple Silicon)

A hardware-aware distributed agent framework, adapted for macOS via local [Ollama](https://ollama.com) integration.

## Overview

On macOS, the Adaptive OS Mesh bypasses the x86-specific SIMD kernels (AVX2/AVX512) and CUDA/Vulkan GPU paths used on Linux nodes. Instead, all inference tasks are routed through a new `MAC_OLLAMA` hardware path, which delegates LLM generation to a locally running Ollama instance. Ollama handles Apple Silicon GPU acceleration (via Metal) transparently.

### Architecture on macOS

```
┌──────────────────────────────────────────────┐
│           Adaptive OS Mesh Node              │
│                                              │
│  InferenceRequest                            │
│        │                                     │
│        ▼                                     │
│   ┌──────────┐    runtime.GOOS == "darwin"   │
│   │ ScheInfer │──────────────────────┐       │
│   └──────────┘                      │       │
│        │                            ▼       │
│   (Linux: AVX2/CUDA)     ┌──────────────┐   │
│                          │ OllamaClient │   │
│                          └──────┬───────┘   │
│                                 │           │
└─────────────────────────────────┼───────────┘
                                  │ HTTP POST
                                  ▼
                      ┌────────────────────┐
                      │  Ollama (localhost)│
                      │  :11434/api/gen    │
                      │  Metal / ARM NEON   │
                      └────────────────────┘
```

### Key Differences from Linux

| Feature           | Linux Nodes                           | macOS Node                 |
| ----------------- | ------------------------------------- | -------------------------- |
| Inference Backend | AVX2/AVX512 C++ kernels, CUDA, Vulkan | Ollama (Metal-accelerated) |
| CGO Required      | Yes                                   | No                         |
| ScheInfer Route   | `CPU_AVX2`, `GPU_CUDA`, etc.          | `MAC_OLLAMA`               |
| Mesh Capability   | `CPU:N cores \| GPU:CUDA(Ampere)`     | `MAC:OLLAMA`               |

## Prerequisites

| Requirement           | How to Install                                                              |
| --------------------- | --------------------------------------------------------------------------- |
| **Go 1.25+**          | `brew install go`                                                           |
| **Python 3.10+**      | `brew install python@3.13`                                                  |
| **Ollama**            | `brew install ollama` or [ollama.com/download](https://ollama.com/download) |
| **Ollama Models**     | `ollama pull qwen3.5` and `ollama pull nomic-embed-text`                    |
| **NATS Server**       | `brew install nats-server`                                                  |
| **Protobuf Compiler** | `brew install protobuf`                                                     |

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/jalvarez36808/adaptive-os-mesh.git
    cd adaptive-os-mesh
    ```

2.  Install Go dependencies:

    ```bash
    go mod tidy
    ```

3.  Set up a Python virtual environment and install dependencies:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

4.  Verify Ollama is running with a model loaded:

    ```bash
    ollama list
    ```

    You should see both `qwen3.5` and `nomic-embed-text`. If not, pull them:

    ```bash
    ollama pull qwen3.5
    ollama pull nomic-embed-text
    ```

5.  Verify the project builds:
    ```bash
    go build ./...
    ```

## Running Tests

Run the controller tests (includes ScheInfer routing and Ollama client tests):

```bash
go test ./internal/controller/... -v
```

> **Note**: The `internal/quantx/...` tests are excluded on macOS via `//go:build !darwin` tags, as they require x86 C++ toolchains.

## Usage

1.  **Start Ollama** (if not already running as a background service):

    ```bash
    ollama serve
    ```

2.  **Start the NATS Server** (with JetStream enabled for KV cache sync):

    ```bash
    nats-server -p 4222 -js &
    ```

3.  **Quick E2E Test** — Verify Ollama integration works end-to-end:

    ```bash
    go run ./cmd/test_ollama/main.go
    ```

4.  **Seed the Vector Store** — Embed project documents for semantic search:

    ```bash
    go run ./cmd/seed_vectors/main.go
    ```

5.  **Test Semantic Search** — Query the vector store:

    ```bash
    go run ./cmd/test_vector_search/main.go
    ```

## Configuration

The Ollama client defaults can be overridden in `ollama_client.go`:

| Setting   | Default                  | Description               |
| --------- | ------------------------ | ------------------------- |
| `BaseURL` | `http://localhost:11434` | Ollama API endpoint       |
| `Model`   | `qwen3.5`                | Model used for generation |
| `Timeout` | `120s`                   | HTTP client timeout       |

## Files Modified for macOS

These files were changed or created to enable macOS compatibility:

| File                                   | Change                                              |
| -------------------------------------- | ---------------------------------------------------- |
| `internal/quantx/bridge.go`            | Added `//go:build !darwin`                           |
| `internal/quantx/bridge_cuda.go`       | Added `!darwin` to build tag                         |
| `internal/quantx/bridge_gpu_stubs.go`  | Added `!darwin` to build tag                         |
| `internal/quantx/bridge_darwin.go`     | **[NEW]** macOS stub                                 |
| `internal/quantx/*.cpp`, `*.cu`        | Added `//go:build !darwin`                           |
| `internal/quantx/ggml/*.go`, `*.cpp`   | Added `//go:build !darwin`                           |
| `internal/controller/scheinfer.go`     | Added `MAC_OLLAMA` routing                           |
| `internal/controller/ollama_client.go` | **[NEW]** Ollama HTTP client (generate + embed)      |
| `internal/controller/inference.go`     | Wired Ollama into `Generate()`                       |
| `internal/controller/vectorstore.go`   | **[NEW]** In-process vector store + cosine similarity|
| `internal/controller/qdrant.go`        | Wired vector store as primary search on macOS        |
| `cmd/seed_vectors/main.go`             | **[NEW]** CLI to seed documents into vector store    |
| `cmd/test_vector_search/main.go`       | **[NEW]** CLI to test semantic search                |

## License

Apache-2.0
