---
layout: default
title: GPU Computing
date: 2025-10-16
excerpt: A concise walkthrough of the GPU Computing course, updated with modern CUDA idioms, profiling tips, and cross-platform notes.
tags:
  - cuda
  - parallelism
  - high-performance-computing
---

# Chapter 1: The GPU Computing Model

Welcome to the world of GPU computing! This study book will guide you, from first principles, through the architecture of modern Graphics Processing Units (GPUs) and teach you how to harness their massive parallel processing power using the CUDA programming model. We will begin by exploring why GPUs are so effective for certain computational problems and how their fundamental design differs from that of a traditional Central Processing Unit (CPU).

## 1.1 Why Use a GPU? CPU vs. GPU Architectures

At first glance, CPUs and GPUs are both processors built from silicon. However, their internal architectures are designed to solve problems in fundamentally different ways. A CPU is a master of latency-sensitive tasks, designed to execute a single sequence of instructions (a thread) as quickly as possible. A GPU, in contrast, is a master of throughput-sensitive tasks, designed to execute thousands of parallel threads simultaneously.

This design philosophy is visible in their physical layouts, or "die shots."

- A CPU die is typically composed of a small number of very powerful, complex cores. A significant portion of the silicon is dedicated to sophisticated control logic and large caches to minimize the time it takes to fetch data and instructions, thereby reducing latency for a single task.
- A GPU die is packed with hundreds or even thousands of smaller, simpler cores. Less silicon is devoted to complex control logic and large caches; instead, the architecture prioritizes raw computational units to maximize throughput—the total number of calculations performed across the entire chip per second.

### System-Level Integration and Performance

In a typical computer system, the CPU and GPU are distinct components with their own dedicated memory systems, connected via an I/O bridge.

A diagram of a modern system illustrates this separation: The CPU is connected to its Host Memory (system RAM) through a high-speed memory interface. The GPU, a separate component on the peripheral bus (like PCIe), is connected to its own dedicated, high-bandwidth GPU Memory.

The performance differences, particularly in memory bandwidth and computational throughput, are staggering. Consider a comparison between a high-end server CPU and a data center GPU:

| Component | Metric | Performance |
| --- | --- | --- |
| CPU (e.g., AMD EPYC 9754) | Host Memory Bandwidth | ~460 GB/s |
| CPU (e.g., AMD EPYC 9754) | Double Precision (DP) TFLOP/s | ~6 TFLOP/s |
| GPU (e.g., NVIDIA H100) | GPU Memory Bandwidth | ~3.3 TB/s |
| GPU (e.g., NVIDIA H100) | Double Precision (DP) TFLOP/s | ~34-67 TFLOP/s |
| GPU (e.g., NVIDIA H100) | Single Precision (SP) TFLOP/s | ~67-490 TFLOP/s |

The GPU's memory bandwidth can be over 7 times higher than the CPU's, and its computational throughput can be an order of magnitude greater. This massive throughput is precisely what we aim to leverage with GPU computing. The CUDA (Compute Unified Device Architecture) platform allows us to use this power not just for graphics, but for general-purpose computing tasks.

## 1.2 The GPU Architecture for General-Purpose Computing

While originally designed for the graphics pipeline (processing vertices, pixels, etc.), the architecture of GPUs has been generalized for computation. The NVIDIA G80 architecture was a pivotal step in this evolution.

A simplified diagram for general-purpose processing shows how a program interacts with the GPU:

1. The Host (the CPU) sends a command to the GPU to start a computation.
2. An Input Assembler and Thread Execution Manager on the GPU receive this command.
3. The work is distributed across an array of Streaming Multiprocessors (SMs).

A Streaming Multiprocessor (SM) is the fundamental processing unit of a CUDA-capable GPU. You can think of it as a group of simple cores that execute threads in parallel. Each SM has its own execution units, schedulers, and a small, fast, on-chip memory called Shared Memory.

All SMs on the GPU can access a large, shared Global Memory through a system of parallel data caches. The host CPU initiates data transfers to and from this Global Memory to set up computations and retrieve results. This load/store architecture, where data is explicitly moved between different memory spaces, is a central concept in GPU programming.

---

# Chapter 2: Introduction to CUDA Programming

Now that we understand the hardware philosophy behind GPUs, let's explore how to program them. CUDA is an extension of the C programming language created by NVIDIA that exposes the GPU's parallel architecture directly to the developer.

## 2.1 The CUDA Programming Model

A CUDA program is a hybrid program consisting of two parts: a host part that runs on the CPU and a device part that runs on the GPU.

- The CPU (host) part is responsible for serial or low-parallelism tasks, such as setting up data, managing memory transfers, and launching computations on the GPU.
- The GPU (device) part handles massively parallel operations by executing kernels across many threads.

## 2.2 The Thread Hierarchy

The most fundamental concept in CUDA is the thread hierarchy. When you launch a computation on the GPU, you are launching a kernel function that is executed by a grid of threads. This hierarchy is organized into three levels:

- Thread: The smallest unit of execution. A single thread executes one instance of the kernel code.
- Block: A group of threads. Threads within the same block can cooperate by sharing data through a fast, on-chip shared memory and can synchronize their execution using barriers.
- Grid: A group of blocks. A kernel is launched as a single grid of thread blocks. Blocks within a grid are executed independently and in any order, and they cannot directly synchronize with each other.

The following diagram illustrates this structure. A Grid is composed of multiple Blocks, which can be arranged in one, two, or three dimensions. Each Block, in turn, contains multiple Threads, which can also be arranged in one, two, or three dimensions. For example, the diagram shows a Grid of Blocks arranged in a 2x2 configuration. One of these blocks, Block (1,1), is shown expanded, containing an array of Threads.

This hierarchical structure allows you to naturally map the parallelism in your problem onto the GPU hardware. For example, to process a 2D image, you might launch a 2D grid of blocks, where each block processes a tile of the image and each thread within a block processes a single pixel.

## 2.3 Launching a Kernel

A CUDA kernel is a function that runs on the device. You define it in your C/C++ code using the `__global__` declaration specifier.

A kernel is defined like a C function but with the `__global__` prefix, which indicates it can be called from the host and is executed on the device. A kernel function must have a void return type.

```c++
// Kernel function declaration
__global__ void MyKernel(float* data) {
    // Kernel code executed by each thread
}
```

To execute this function, you call it from the host using a special `<<< ... >>>` syntax, known as the execution configuration. This tells the CUDA runtime how many threads to launch.

```c++
kernel_name<<<numBlocks, threadsPerBlock>>>(arguments);
```

- `numBlocks`: The number of thread blocks to launch in the grid.
- `threadsPerBlock`: The number of threads to launch in each block.

### Unique Thread Identification

Inside a kernel, each thread needs a way to identify itself so it can work on a unique piece of data. CUDA provides built-in variables for this purpose:

- `threadIdx`: A 3-component vector (x, y, z) that contains the unique index of a thread within its block.
- `blockIdx`: A 3-component vector (x, y, z) that contains the unique index of a block within its grid.
- `blockDim`: A 3-component vector (x, y, z) that contains the dimensions of the block (the number of threads in each dimension).
- `gridDim`: A 3-component vector (x, y, z) that contains the dimensions of the grid (the number of blocks in each dimension).

Let's see a simple example of adding two matrices, where each thread computes one element of the result.

```c++
// Kernel to add two N x N matrices
__global__ void matAdd(float A[N][N], float B[N][N], float C[N][N]) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  C[i][j] = A[i][j] + B[i][j];
}

int main() {
  // Kernel invocation for an N x N problem
  dim3 dimBlock(N, N); // Use N x N threads per block
  matAdd<<<1, dimBlock>>>(A, B, C); // Launch one block
}
```

In this simple case, we launch a single block (1) with NxN threads (`dimBlock`). Each thread uses its `threadIdx.x` and `threadIdx.y` to find its unique (i, j) coordinate and computes a single element `C[i][j]`.

### Scaling Up with Grids

The previous example only works if the matrix size `N` is small enough to fit within a single thread block (e.g., up to 1024 threads total). To handle larger problems, we must launch a grid of multiple blocks.

When using multiple blocks, we need a way to calculate a global index for each thread across the entire grid. The standard formula for a 1D problem is:

```c++
int global_index = blockIdx.x * blockDim.x + threadIdx.x;
```

Let's break this down:

- `blockIdx.x * blockDim.x`: This calculates the starting index for the current block. For example, if each block has 256 threads (`blockDim.x`), then block 0 starts at index 0, block 1 starts at index 256, block 2 starts at index 512, and so on.
- `+ threadIdx.x`: This adds the thread's local index within the block to get its unique global index.

Here is the revised `matAdd` kernel that can handle any size matrix by using a grid of blocks.

```c++
__global__ void matAdd(float A[N][N], float B[N][N], float C[N][N]) {
  // Calculate the global row and column index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary check to prevent writing out of bounds
  if (i < N && j < N) {
    C[i][j] = A[i][j] + B[i][j];
  }
}

int main() {
  // Set the block size (e.g., 16x16 threads)
  dim3 dimBlock(16, 16);

  // Calculate the grid size needed to cover the entire N x N matrix
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);

  // Launch the kernel
  matAdd<<<dimGrid, dimBlock>>>(A, B, C);
}
```

Key Improvements:

1. Global Index Calculation: The kernel now correctly computes global `i` and `j` indices, allowing threads from different blocks to work on different parts of the matrix.
2. Boundary Check: The `if (i < N && j < N)` statement is crucial. Because we must launch a whole number of blocks, the total number of threads launched might be greater than the number of elements in our matrix. This check ensures that only threads corresponding to valid matrix elements perform a write, preventing memory corruption.
3. Grid Calculation: The formula `(N + dimBlock.x - 1) / dimBlock.x` is a standard C/C++ integer arithmetic trick for calculating the ceiling of a division. It ensures we launch enough blocks to cover all `N` elements. For example, if `N=50` and `dimBlock.x=16`, the calculation is `(50 + 16 - 1) / 16 = 65 / 16`, which results in 4 in integer division, correctly launching 4 blocks to cover the 50 elements.

### Choosing Block and Grid Sizes

- Threads per Block: This should be a multiple of the warp size (typically 32, a concept we'll cover later). A common starting point is 128, 256, or 512 threads per block. The ideal number balances resource usage with the ability to hide memory latency. A range of 100-1000 threads is often optimal.
- Blocks per Grid: You should launch enough blocks to keep all the SMs on the GPU busy. A good heuristic is to launch at least twice as many blocks as there are SMs on your GPU.

## 2.4 Thread Communication and Synchronization

A key feature of the CUDA model is that threads within the same block can cooperate. This is achieved through two main mechanisms:

- Shared Memory: A small, fast, on-chip memory that is shared by all threads in a block. Access to shared memory is much faster than global memory, making it ideal for caching frequently used data or for intermediate results.
- Barrier Synchronization: Threads in a block can be synchronized by calling the `__syncthreads()` intrinsic. When a thread reaches this function, it pauses until every other thread in its block has also reached the same point. This is essential for coordinating memory accesses, for example, ensuring all threads have finished loading data into shared memory before any thread starts consuming it.

Important Limitation: Threads from different blocks cannot directly communicate or synchronize with each other. They operate independently. The only way they can "communicate" is by reading and writing to global memory. However, the guarantees for when writes from one block become visible to another are very weak. If you need global synchronization across all threads, you must terminate the current kernel and launch a new one.

## 2.5 The CUDA Memory Hierarchy

Understanding the memory hierarchy is critical for writing high-performance CUDA code. A thread has access to several distinct memory spaces, each with different characteristics regarding scope, lifetime, and speed.

A diagram of the memory hierarchy shows that each Thread has its own private Registers. A group of threads in a Block shares a common Shared Memory. All blocks in the Grid can access the larger but slower Global Memory. The Host (CPU) also interacts with the device via this Global Memory.

### Global Memory

- Scope: Accessible by all threads in the grid, as well as the host (CPU).
- Lifetime: Persists for the lifetime of the application, beyond the execution of any single kernel.
- Characteristics: Large (often many gigabytes) but has high latency. This is the primary memory used for transferring data between the host and the device. Accesses to global memory are very sensitive to access patterns, and uncoalesced (scattered) accesses can severely degrade performance.

You manage global memory from the host using the CUDA runtime API:

- `cudaMalloc(&d_ptr, size)`: Allocates `size` bytes of memory on the device and returns a pointer in `d_ptr`.
- `cudaFree(d_ptr)`: Frees device memory.
- `cudaMemcpy(dst, src, size, type)`: A blocking function to copy data between host and device. The `type` can be `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, or `cudaMemcpyDeviceToDevice`.
- `cudaMemcpyAsync(...)`: A non-blocking version for overlapping data transfers with computation.

The standard workflow for a CUDA program looks like this:

```c++
// 1. Allocate memory on the host (CPU) and device (GPU)
void *h_mem = malloc(SIZE);
void *d_mem;
cudaMalloc(&d_mem, SIZE);

// 2. Transfer input data from host to device
cudaMemcpy(d_mem, h_mem, SIZE, cudaMemcpyHostToDevice);

// 3. Launch one or more kernels to compute on the device
kernel1<<<...>>>(d_mem, ...);
kernel2<<<...>>>(d_mem, ...);

// 4. Transfer results from device back to host
cudaMemcpy(h_mem, d_mem, SIZE, cudaMemcpyDeviceToHost);

// 5. Free allocated memory
cudaFree(d_mem);
free(h_mem);
```

Note: When calling a kernel, you can only pass pointers to device memory (like `d_mem`), not host memory.

### Shared Memory

- Scope: Accessible only by threads within the same block.
- Lifetime: Persists only for the lifetime of the block. Once a block finishes executing, its shared memory is gone.
- Characteristics: Very fast on-chip memory. In the best case, access latency is similar to registers. It is organized into banks, and parallel access is possible as long as threads do not access addresses in the same bank (a "bank conflict"). Bank conflicts cause accesses to be serialized, reducing performance.

## 2.6 CUDA Language Extensions

CUDA extends C/C++ with special specifiers for declaring variables and functions.

### Variable Declaration Specifiers

These specifiers determine where a variable is stored and its scope.

| Location Specifier | Memory Space | Scope | Lifetime |
| --- | --- | --- | --- |
| `__device__ float var;` | Global Memory | All threads + Host API | Application |
| `__constant__ float var;` | Constant Memory | All threads + Host API | Application |
| `__shared__ float var;` | Shared Memory | All threads in block | Block |
| `texture <float> ref;` | Texture Memory | All threads + Host API | Application |

A key function related to shared memory is `__syncthreads()`. This intrinsic creates a barrier, forcing all threads in a block to wait until everyone has reached this point. It is essential for managing dependencies when using shared memory, ensuring that data is fully written before it is read by other threads.

### Function Declaration Specifiers

These specifiers determine where a function is executed and where it can be called from.

| Declaration Specifier | Executed On | Callable From |
| --- | --- | --- |
| `__device__ float Func()` | Device | Device |
| `__global__ void Kernel()` | Device | Host |
| `__host__ float Func()` | Host | Host |

- `__global__` defines a kernel, which can only be called from the host.
- `__device__` functions can only be called from other `__device__` or `__global__` functions.
- `__host__` is the default and can be combined with `__device__` to create a function that can be compiled for and called from both the CPU and GPU.
- Device functions have several restrictions: they do not support recursion, variable numbers of arguments, or non-static variable declarations inside the function.

### Type Specifiers

CUDA introduces several built-in types:

- Vector types: Such as `float2`, `float4`, `int2`, `int4`, which are simple structs containing 2 or 4 components. These are useful for representing coordinates or colors and can lead to more efficient memory access.
- `dim3` type: A struct based on `uint3` used for specifying dimensions for grids and blocks. Unspecified components are automatically initialized to 1.

## 2.7 Compilation and Execution

CUDA code is compiled using the `nvcc` (NVIDIA C Compiler) driver. `nvcc` is a powerful tool that separates the host and device code.

1. It processes the CUDA source code, separating host (`__host__`) code from device (`__global__`, `__device__`) code.
2. The host code is compiled by a standard C++ compiler like `g++` or `clang`.
3. The device code is compiled into PTX (Parallel Thread Execution) code.

PTX is a virtual machine and instruction set architecture (ISA) for GPUs. It acts as a stable assembly-like language for the GPU. This is a key part of CUDA's forward compatibility. When you compile your code, `nvcc` can embed the PTX in your executable. When you run your application, the GPU driver performs a final Just-In-Time (JIT) compilation step, translating the PTX into the specific machine code for the target GPU (e.g., GF100, GK110, GP100) you are running on.

Finally, `nvcc` links the compiled host and device code with the necessary CUDA libraries (`cudart`, `cuda`) to produce the final executable.

## 2.8 A Complete Example: SAXPY

SAXPY stands for Scalar Alpha X Plus Y. It is a common, simple vector operation used to benchmark computational performance. The operation is defined by the formula:

```text
 y[i] = \alpha \cdot x[i] + y[i] 
```

Here, `x` and `y` are vectors, α (alpha) is a scalar, and `i` is the index of the element. This is an ideal problem for GPU acceleration because the calculation for each element `y[i]` is completely independent of all other elements.

### Serial CPU Implementation

A standard C implementation of SAXPY uses a simple `for` loop.

```c++
// Kernel function (CPU)
void saxpy_serial(int n, float alpha, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = alpha * x[i] + y[i];
  }
}
```

### Parallel CUDA Implementation

The CUDA version replaces the loop with a kernel where each thread processes one element.

```c++
// Kernel function (CUDA device)
__global__ void saxpy_parallel(int n, float alpha, float *x, float *y) {
  // Compute the global index for this thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check to avoid writing past the end of the arrays
  if (i < n) {
    y[i] = alpha * x[i] + y[i];
  }
}
```

This is a perfect demonstration of the SPMD model. Every thread runs this exact same code, but because each thread has a unique `i` calculated from `blockIdx.x` and `threadIdx.x`, each thread operates on a different element of the vectors `x` and `y`.

### Performance Considerations: Pinned Memory

Initial performance tests often show that even for large vectors, the GPU version can be slower than the CPU version. This is usually because the time taken to transfer data between host and device memory (`cudaMemcpy`) dominates the total runtime.

One way to significantly reduce this data movement cost is to use pinned memory (or page-locked memory). By default, host memory allocated with `malloc` is pageable, meaning the operating system can move it around in physical memory. For the GPU to access this data, the CUDA driver must first copy it into a temporary, pinned buffer before transferring it to the device.

By allocating host memory directly as pinned memory, we eliminate this extra copy. Pinned memory is a scarce resource, so it should be used judiciously.

```c++
float *h_x, *h_y;

// Standard pageable memory allocation
// h_x = (float*) malloc(N * sizeof(float));
// h_y = (float*) malloc(N * sizeof(float));

// Pinned memory allocation
cudaMallocHost((void**)&h_x, N * sizeof(float));
cudaMallocHost((void**)&h_y, N * sizeof(float));
```

Using `cudaMallocHost` instead of `malloc` can lead to a significant reduction in data transfer times, making the GPU's computational advantage more apparent.

## 2.9 Device Properties and Common Errors

You can query the properties of the GPU in your system to make informed decisions about kernel launch configurations. The `deviceQuery` utility provides a survey of these properties.

### GPU Property Survey (Examples)

| Property | GeForce GTX 480 (CC 2.0) | Tesla K20c (CC 3.5) | RTX 2080Ti (CC 7.5) |
| --- | --- | --- | --- |
| Global Memory | 1.5 GB | 5 GB | 11 GB |
| Multiprocessors (SMs) | 15 | 13 | 68 |
| Cores (Total) | 480 | 2496 | 4352 |
| Shared Memory / Block | 48 KB | 48 KB | 48 KB |
| Registers / Block | 32k | 64k | 64k |
| Warp Size | 32 | 32 | 32 |
| Max Threads / Block | 1024 | 1024 | 1024 |
| Max Block Dimension | 1024 x 1024 x 64 | 1024 x 1024 x 64 | 1024 x 1024 x 64 |
| Max Grid Dimension | 65535 x 65535 x 65535 | 2G x 65535 x 65535 | 2G x 65535 x ? |

### Common CUDA Errors

- CUDA Error: the launch timed out and was terminated: The kernel took too long to execute. This often happens on systems with a graphical display, where the OS will kill a kernel to prevent the screen from freezing. A common solution is to stop the X11 server.
- CUDA Error: unspecified launch failure: This is a generic error that often indicates a segmentation fault inside the kernel, such as accessing an array out of bounds or dereferencing an invalid pointer.
- CUDA Error: invalid configuration argument: The kernel launch configuration is invalid. Common causes include requesting too many threads per block (e.g., > 1024) or requesting more resources (shared memory, registers) per thread than are available on the SM.
- error: identifier "__eh_curr_region" is undefined: A compiler problem often related to using non-static allocation for shared memory. Ensure shared memory arrays are declared with static sizes.

## 2.10 Summary

This introduction to CUDA has covered the fundamental concepts needed to start writing parallel programs for GPUs. Compared to traditional CPU programming, the CUDA model presents a new paradigm.

Main differences from CPU programming:

- Sophisticated Resource Planning: You must manually manage the hierarchy of threads, blocks, and grids.
- Manual Data Movements: Data must be explicitly transferred between the host and device.
- Limited Memory Capacity: GPU memory, while fast, is often smaller than system RAM.
- Direct Hardware Control: The model gives an experienced user direct control over the hardware, offering plenty of opportunities for performance optimization.

Once understood, the programming model is remarkably straightforward and powerful, allowing you to unlock massive parallelism for a wide range of computational problems.
