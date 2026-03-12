# Adaptive OS Mesh

A hardware-aware distributed agent framework designed for high-performance inference and cross-node orchestration.

## Overview

The Adaptive OS Mesh is an experimental framework that leverages heterogeneous hardware across a distributed network of nodes. It features dynamic SIMD kernel selection, topology-aware task routing, and real-time performance monitoring.

### Key Architectural Layers

1.  **Hardware-Aware Kernels (`internal/quantx`)**: 
    - Optimized C++ kernels for 2-bit quantization (Q2_K).
    - Dynamic runtime selection between AVX2 and AVX512 (Tiger Lake optimized).
    - Fail-safe stubs for systems without hardware accelerators.

2.  **Strategic Mesh Controller (`cmd/vextra`)**: 
    - Centralized control plane using gRPC for strategic reasoning and NATS for operational heartbeats.
    - Integrated with **ScheInfer** for intelligent workload distribution.

3.  **ScheInfer: Topology-Aware Routing**:
    - Adaptive task routing based on L3 cache boundaries and accelerator availability.
    - Tiered execution: `CPU_AVX2` (Cache-resident) -> `GPU_CUDA` (Large tensors) -> `CPU_AVX512` (Vector-optimized fallback).

4.  **Vextra TUI (`cmd/vextra_tui`)**: 
    - Real-time observability dashboard.
    - Tracks **THRPT** (Throughput in GB/s) and Value of Contribution (VoC) metrics.

## Hardware Support

- **Node A (Research/Compute)**: 16MB L3, AMD Ryzen 5900HX (AVX2).
- **Node B (Desktop/GPU)**: 8MB L3, Intel i7-7700, NVIDIA GTX 1070.
- **Node C (Edge/Vector)**: 12MB L3, Intel Tiger Lake (AVX512 Optimized).

## Getting Started

### Prerequisites

- **Go**: 1.26.0+
- **Python**: 3.10+ (for benchmarking)
- **Protobuf Compiler**: `protoc`
- **NATS Server**: For operational messaging.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/[your-username]/agent-mesh-core.git
    cd agent-mesh-core
    ```

2.  Install Go dependencies:
    ```bash
    go mod tidy
    ```

3.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Provision the environment:
    ```bash
    ./start_node_c.sh  # Example for Node C
    ```

### Running Tests

Execute the unified test suite:
```bash
go test ./internal/quantx/... -v
npm test  # For integrity and dashboard tests
```

### Usage

1.  **Start the NATS Server**:
    The mesh controller requires a NATS server for operational messaging.
    ```bash
    nats-server -p 4222
    ```

2.  **Run the Vextra Controller**:
    In a separate terminal, start the main controller.
    ```bash
    go run ./cmd/vextra
    ```

3.  **Launch the TUI**:
    To monitor the mesh in real-time, run the TUI.
    ```bash
    go run ./cmd/vextra_tui
    ```

## License

Apache-2.0
