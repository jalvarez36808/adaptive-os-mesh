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
                      │  Ollama (localhost) │
                      │  :11434/api/gen    │
                      │  Metal / ARM NEON   │
                      └────────────────────┘
```

### Key Differences from Linux

| Feature | Linux Nodes | macOS Node |
|---|---|---|
| Inference Backend | AVX2/AVX512 C++ kernels, CUDA, Vulkan | Ollama (Metal-accelerated) |
| CGO Required | Yes | No |
| ScheInfer Route | `CPU_AVX2`, `GPU_CUDA`, etc. | `MAC_OLLAMA` |
| Mesh Capability | `CPU:N cores \| GPU:CUDA(Ampere)` | `MAC:OLLAMA` |

## Prerequisites

| Requirement | How to Install |
|---|---|
| **Go 1.22+** | `brew install go` |
| **Python 3.10+** | `brew install python@3.13` |
| **Ollama** | `brew install ollama` or [ollama.com/download](https://ollama.com/download) |
| **An Ollama Model** | `ollama pull qwen3.5` (or any model you prefer) |
| **NATS Server** | `brew install nats-server` |
| **Protobuf Compiler** | `brew install protobuf` |

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/groovy-byte/adaptive-os-mesh.git
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
    You should see at least one model (e.g., `qwen3.5`). If not, pull one:
    ```bash
    ollama pull qwen3.5
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

2.  **Start the NATS Server**:
    ```bash
    nats-server -p 4222
    ```

3.  **Quick E2E Test** — Verify Ollama integration works end-to-end:
    ```bash
    go run ./cmd/test_ollama/main.go
    ```
    Expected output:
    ```
    === macOS Ollama Integration E2E Test ===
    Sending request to inference controller...
    [ScheInfer] macOS Node detected: Routing to MAC_OLLAMA
    [Inference] Executing via Ollama...

    --- Response ---
    Hardware Path: MAC_OLLAMA
    Tokens Used: <N>
    Latency: <N> ms
    Text: <generated text from Ollama>
    ```

## Configuration

The Ollama client defaults can be overridden in `ollama_client.go`:

| Setting | Default | Description |
|---|---|---|
| `BaseURL` | `http://localhost:11434` | Ollama API endpoint |
| `Model` | `qwen3.5` | Model used for generation |
| `Timeout` | `120s` | HTTP client timeout |

## Files Modified for macOS

These files were changed or created to enable macOS compatibility:

| File | Change |
|---|---|
| `internal/quantx/bridge.go` | Added `//go:build !darwin` |
| `internal/quantx/bridge_cuda.go` | Added `!darwin` to build tag |
| `internal/quantx/bridge_gpu_stubs.go` | Added `!darwin` to build tag |
| `internal/quantx/bridge_darwin.go` | **[NEW]** macOS stub |
| `internal/quantx/*.cpp`, `*.cu` | Added `//go:build !darwin` |
| `internal/quantx/ggml/*.go`, `*.cpp` | Added `//go:build !darwin` |
| `internal/controller/scheinfer.go` | Added `MAC_OLLAMA` routing |
| `internal/controller/ollama_client.go` | **[NEW]** Ollama HTTP client |
| `internal/controller/inference.go` | Wired Ollama into `Generate()` |

## License

Apache-2.0
