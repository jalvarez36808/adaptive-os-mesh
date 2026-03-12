//go:build cuda && !darwin

package quantx

/*
#cgo CFLAGS: -I/opt/cuda/include
#cgo LDFLAGS: -L/opt/cuda/lib64 -lcudart ${SRCDIR}/quantx_cuda.o
#include "quantx.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

// Helper functions to manage CUDA memory from Go
static void* cuda_malloc(size_t size) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) return NULL;
    return ptr;
}

static void cuda_memcpy_to_device(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

static void cuda_memcpy_from_device(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

static void cuda_free(void* ptr) {
    cudaFree(ptr);
}

static void cuda_sync() {
    cudaDeviceSynchronize();
}

static char* get_gpu_name() {
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    char* name = (char*)malloc(256);
    strncpy(name, prop.name, 256);
    return name;
}

static size_t get_gpu_vram() {
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.totalGlobalMem;
}

static int get_gpu_compute_capability() {
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.major * 10 + prop.minor;
}
*/
import "C"
import (
	"unsafe"
	"errors"
)

/**
 * GetGpuInfo returns the name and total VRAM of the first CUDA device.
 */
func GetGpuInfo() (string, uint64, error) {
	cName := C.get_gpu_name()
	defer C.free(unsafe.Pointer(cName))
	name := C.GoString(cName)
	vram := uint64(C.get_gpu_vram())
	if vram == 0 {
		return "", 0, errors.New("cuda: no device found or failed to get properties")
	}
	return name, vram, nil
}

/**
 * GetGpuComputeCapability returns the CUDA compute capability (e.g., 86 for Ampere).
 */
func GetGpuComputeCapability() int {
	return int(C.get_gpu_compute_capability())
}

/**
 * DequantizeQ2KCUDA dequantizes Q2_K blocks using the GPU with Auto-Fallback.
 * If CUDA initialization fails or memory allocation fails, it transparently
 * falls back to the AVX2 CPU implementation.
 */
func DequantizeQ2KCUDA(vx unsafe.Pointer, vy []float32, k int) error {
	if len(vy) < k {
		return errors.New("quantx: output buffer size is less than k")
	}

	// Try CUDA path
	err := runCudaDequantize(vx, vy, k)
	if err != nil {
		// LOG: Falling back to CPU
		DequantizeQ2K(vx, vy, k)
		return nil
	}

	return nil
}

func runCudaDequantize(vx unsafe.Pointer, vy []float32, k int) error {
	numBlocks := k / 256
	inputSize := C.size_t(numBlocks * 72)
	outputSize := C.size_t(k * 4)

	dVx := C.cuda_malloc(inputSize)
	if dVx == nil {
		return errors.New("cuda: failed to allocate device memory")
	}
	defer C.cuda_free(dVx)

	dVy := C.cuda_malloc(outputSize)
	if dVy == nil {
		return errors.New("cuda: failed to allocate device memory")
	}
	defer C.cuda_free(dVy)

	C.cuda_memcpy_to_device(dVx, vx, inputSize)
	C.dequantize_q2_k_cuda(dVx, (*C.float)(dVy), C.int(k))
	C.cuda_sync()
	C.cuda_memcpy_from_device(unsafe.Pointer(&vy[0]), dVy, outputSize)

	return nil
}

// CudaBuffer represents a buffer on the GPU
type CudaBuffer struct {
	Ptr  unsafe.Pointer
	Size int
}

func NewCudaBuffer(size int) (*CudaBuffer, error) {
	ptr := C.cuda_malloc(C.size_t(size))
	if ptr == nil {
		return nil, errors.New("cuda: failed to allocate device memory")
	}
	return &CudaBuffer{Ptr: ptr, Size: size}, nil
}

func (b *CudaBuffer) Free() {
	C.cuda_free(b.Ptr)
}

func (b *CudaBuffer) CopyToDevice(src unsafe.Pointer, size int) {
	C.cuda_memcpy_to_device(b.Ptr, src, C.size_t(size))
}

func (b *CudaBuffer) CopyFromDevice(dst unsafe.Pointer, size int) {
	C.cuda_memcpy_from_device(dst, b.Ptr, C.size_t(size))
}

/**
 * DequantizeQ2KCUDAKernel runs the dequantization kernel on data already on the GPU.
 * This is used for high-performance benchmarking.
 */
func DequantizeQ2KCUDAKernel(dVx, dVy unsafe.Pointer, k int) {
	C.dequantize_q2_k_cuda(dVx, (*C.float)(dVy), C.int(k))
	C.cuda_sync()
}
