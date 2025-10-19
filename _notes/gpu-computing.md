---
title: GPU Computing
date: 2025-10-16
excerpt: A concise walkthrough of the GPU Computing course, updated with modern CUDA idioms, profiling tips, and cross-platform notes.
tags:
  - cuda
  - parallelism
  - high-performance-computing
---

## Why GPUs?

Graphics processors expose thousands of lightweight cores designed to execute the same instruction on many data elements. That makes them the workhorse for dense numerical workloads such as linear algebra, particle simulations, and image processing. The trade-off is that developers must express workloads with massive parallelism and organize memory traffic carefully to keep the hardware saturated.

Key architectural traits:
- **Throughput-oriented design:** GPUs devote transistor budget to arithmetic logic units and wide memory buses instead of complex branch prediction.
- **Hierarchy of parallelism:** Threads are grouped into warps (32 threads on NVIDIA devices), warps aggregate into thread blocks, and thread blocks compose a grid. Each step of the hierarchy maps to scheduler units on the GPU.
- **Explicit data movement:** The CPU (host) orchestrates work, while the GPU (device) executes kernels. Transferring data between host and device memory remains a major performance limiter.

## Bulk-Synchronous Parallel Mental Model

The Bulk-Synchronous Parallel (BSP) model, introduced in the 1990s, remains a useful abstraction for reasoning about GPU workloads:
1. **Compute:** Each virtual processor works on its chunk of data.
2. **Communicate:** Partial results are exchanged (typically via shared or global memory).
3. **Synchronize:** A barrier ensures all processors see a consistent state before the next superstep.

Two conclusions follow immediately:
- Always strive for **latency hiding** by launching more virtual processors than physical compute resources (`v >> p`). The runtime overlaps memory stalls with useful work.
- Design kernels with clear superstep boundaries. CUDA provides `__syncthreads()` for intra-block barriers and `cudaDeviceSynchronize()` for host-side synchronization.

## Scaling Laws Refresher

- **Moore's law** (transistor density) explains the historical cadence of GPU improvements, but modern gains come from specialized accelerators, denser packaging, and software optimizations rather than pure frequency bumps.
- **Amdahl's law** bounds speedup when a fraction `S` of a workload is serial:
  \[
    \text{Speedup}_\text{max} = \frac{1}{S + \frac{P}{N}}
  \]
  Diminishing returns appear once additional streaming multiprocessors (SMs) sit idle.
- **Gustafson's law** complements Amdahl by emphasizing that larger problem sizes often unlock additional parallelism. When scaling to larger datasets, GPUs can achieve near-linear speedups.
- **Roofline analysis** combines arithmetic intensity (FLOPs per byte) and achievable bandwidth to reveal whether a kernel is compute-bound or memory-bound. Move kernels up and right on the roofline by tiling, fusion, and mixed precision.

## CUDA Programming Model

A CUDA application is split between host code (C/C++ running on the CPU) and device code (kernels running on the GPU).

```cuda
__global__ void mat_add(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int N) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    const int idx = row * N + col;
    C[idx] = A[idx] + B[idx];
  }
}

int main() {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x,
            (N + block.y - 1) / block.y);
  mat_add<<<grid, block>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();
}
```

Takeaways:
- `<<<grid, block>>>` defines the launch configuration. Each kernel call should be checked with `cudaGetLastError()` in production code.
- Warps execute in lock-step. Branching inside a warp causes **divergence** and serializes execution; reorganize data to minimize divergent branches.
- The compiler can inline device functions (`__device__`) and mark kernels as callable from host/device (`__host__ __device__`). Recursive calls and dynamic memory allocation are limited on device code to keep the execution model predictable.

### Occupancy & Latency Hiding

Occupancy measures how many warps run concurrently on an SM. High occupancy helps hide memory latency, but more threads are not always better—register pressure or shared memory use can throttle parallelism. Use `nvcc --ptxas-options=-v`, `nvdisasm`, or Nsight Compute to spot register spills and adjust block size.

## Memory Hierarchy & Access Patterns

| Memory space | Scope | Latency | Typical usage |
|--------------|-------|---------|----------------|
| Registers    | Thread | ~1 cycle | Scalar temporaries |
| Shared       | Block  | ~10 cycles | Tiling, producer-consumer patterns |
| L1 / L2 cache| Device | tens of cycles | Recently reused data |
| Global       | Device | 400–800 cycles | Primary storage |
| Constant     | Device | cached | Broadcast coefficients |
| Texture      | Device | cached & filtered | Spatial locality, interpolation |

Rules of thumb:
- **Coalesced loads**: consecutive threads in a warp should touch consecutive 32-byte segments. Misaligned or strided accesses degrade bandwidth.
- **Shared memory banks**: modern GPUs expose 32 banks. Accessing `shared[threadIdx.x]` is conflict-free; strided patterns like `shared[threadIdx.x * k]` may serialize unless `k` and the bank count are coprime.
- Prefer `cudaMallocAsync` or `cudaMallocManaged` for long-lived allocations, and use page-locked (`cudaHostAlloc`) buffers for frequent host↔device transfers.

## Case Study: Tiled Matrix Multiplication

The following kernel demonstrates how tiling in shared memory increases arithmetic intensity. It includes minor fixes compared to the lecture handout (bounds checks, consistent tile size, and loop order).

```cuda
constexpr int TILE = 32;

__global__ void matmul_tiled(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
  for (int tileIdx = 0; tileIdx < (N + TILE - 1) / TILE; ++tileIdx) {
    const int tiledCol = tileIdx * TILE + threadIdx.x;
    const int tiledRow = tileIdx * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < N && tiledCol < N)
        ? A[row * N + tiledCol]
        : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (tiledRow < N && col < N)
        ? B[tiledRow * N + col]
        : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < N && col < N) {
    C[row * N + col] = acc;
  }
}
```

Enhancements to explore:
- Use `float4` loads/stores for higher bandwidth when matrices are properly aligned.
- Switch to mixed precision (TF32/FP16) with tensor cores via CUDA WMMA APIs for dramatic speedups.
- Fuse bias addition or activation functions to reduce extra passes over memory.

## Streams, Events, and Overlap

CUDA streams allow independent sequences of work to run concurrently. Use multiple streams to overlap data transfers and kernel execution:

```cpp
cudaStream_t stream_h2d, stream_compute;
cudaStreamCreate(&stream_h2d);
cudaStreamCreate(&stream_compute);

cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream_h2d);
cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream_h2d);

cudaEvent_t ready;
cudaEventCreate(&ready);
cudaEventRecord(ready, stream_h2d);
cudaStreamWaitEvent(stream_compute, ready, 0);

matmul_tiled<<<grid, block, 0, stream_compute>>>(d_A, d_B, d_C, N);
```

Events also help build **CUDA Graphs**, which capture launch dependencies for amortized launch overhead—especially handy for inference workloads with repetitive structure.

## Profiling & Debugging Workflow

1. **Start with sanity checks:** `cuda-memcheck`, `thrust::device_vector`, and unit tests running on small matrix sizes catch indexing bugs early.
2. **Profile kernels:** Nsight Systems shows high-level timing and overlap; Nsight Compute drills into per-kernel metrics like achieved occupancy, DRAM throughput, and instruction mix.
3. **Iterate systematically:** Inspect the roofline placement, optimize memory access, then tune math optimizations. After each change, rerun profiling to confirm the intended improvement.
4. **Automate regression tests:** Measure FLOPs, throughput, and correctness using CI pipelines to detect performance regressions when dependencies or drivers change.

## Directive-Based and Portable Alternatives

| Model      | Key idea | Best for | CUDA analogue |
|------------|----------|----------|---------------|
| OpenACC    | Pragmas annotate loops for acceleration | Porting existing Fortran/C codes with minimal refactoring | `#pragma acc parallel loop` ↔ manual CUDA kernels |
| OpenMP Offload | Modern OpenMP adds `target` directives for GPU execution | Teams with existing OpenMP expertise | Cooperative kernel launches |
| SYCL / oneAPI | Single-source C++ template model | Cross-vendor portability (Intel, AMD, NVIDIA) | CUDA streams and kernels |
| OpenCL     | Explicit platform/device API | Fine-grained control with vendor neutrality | CUDA driver API |

Most large projects mix and match: prototype in CUDA for peak performance, wrap kernels in higher-level abstractions, and fall back to portable back-ends when needed.

## Memory Consistency & Synchronization

- **Cache coherence** on GPUs is scoped: L1 caches may retain stale data unless you use memory fences or avoid shared data structures. Use `__threadfence()` to flush writes from a thread to global memory and `__threadfence_block()` when shared memory ordering suffices.
- **Memory consistency** governs instruction order. Cooperative Groups and C++20 `<cuda/barrier>` utilities provide finer-grained synchronization patterns beyond `__syncthreads()` when you need to coordinate subsets of threads.
- Atomic operations (`atomicAdd`, `atomicCAS`, etc.) serialize access; prefer warp-level primitives (`__shfl_sync`, `__ballot_sync`) to reduce contention.

## Further Reading

- *Programming Massively Parallel Processors* by Kirk & Hwu for architectural fundamentals.
- NVIDIA's CUDA Best Practices Guide for continuously updated optimization strategies.
- The Nsight Compute instruction latency tables to budget your shared vs. global memory accesses.

Armed with these principles, the original lecture material becomes a springboard for building production-ready GPU pipelines that scale with evolving hardware.
