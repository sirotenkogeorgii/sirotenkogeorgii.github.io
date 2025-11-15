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

# GPU Computing

## The Need for Massive Parallelism

For decades, the engine driving computational progress was Moore's Law. Originally stated in 1965 and revised in 1975, it famously observed that the number of transistors on an integrated circuit doubles approximately every two years. This exponential growth allowed single-processor performance to increase at a staggering rate, primarily by increasing clock frequencies.

However, this era of "free" performance scaling came to an end due to physical limitations, a phenomenon known as the end of Dennard Scaling. As transistors became smaller and denser, the power required to run them at ever-higher frequencies became unsustainable, generating too much heat. The industry hit a "power wall."

To continue advancing performance, computer architects shifted their focus from making a single processing core faster to using more cores working in parallel. This transition from complex, latency-optimized cores to numerous, simpler, throughput-oriented cores is the defining characteristic of the post-Dennard era and the primary reason for the rise of massively parallel architectures like the GPU.

**Performance can be viewed through two primary lenses:**

- **Instruction-Level Performance:** This classical view measures performance as the product of instructions executed per cycle and the clock frequency. $$\text{Perf}(\mathrm{ops/s}) = \frac{\text{Instructions}}{\text{cycle}} \times \text{frequency}$$
- **Power-Constrained Performance:** In the modern era, a more critical view is performance per watt, or energy efficiency. $$\text{Perf}(\mathrm{ops/s}) = \underbrace{\text{Power}(\mathrm{W})}_{\text{fixed}} \times \text{Efficiency}(\mathrm{ops/Joule})$$

GPUs are designed to excel in the second paradigm. By replicating many smaller, energy-efficient processing units and often running them at lower frequencies, they achieve massive parallelism and superior overall throughput and efficiency for suitable workloads.

## From Gaming Graphics to General-Purpose Computing

The Graphics Processing Unit, or GPU, began its life with a singular purpose: rendering graphics for video games. The core task of graphics involves performing the same mathematical operations (like matrix transformations and color calculations) on millions of pixels simultaneously. This inherently parallel workload led to the development of highly specialized, parallel hardware.

The graphics pipeline consists of several key stages:

| Stage | Purpose |
| --- | --- |
| Vertex Shader | A programmable stage that transforms the 3D position and texture coordinates of vertices. |
| Triangle Assembly | Connects vertices to form triangles. |
| Rasterization | Determines which pixels on the screen are covered by each triangle. |
| Fragment Shader | A programmable stage that calculates the final color for each affected pixel. |

Around 2007, innovators realized that this powerful, programmable parallel engine could be harnessed for tasks beyond graphics. This marked the birth of General-Purpose GPU (GPGPU) computing. NVIDIA's CUDA (Compute Unified Device Architecture) platform was a pioneering effort that provided a software layer allowing developers to program the GPU directly for general-purpose tasks, unlocking its power for scientific computing, climate research, molecular dynamics, and, most notably, machine learning.

The impact has been profound, enabling computational milestones to be reached years ahead of schedule. For instance, the leap from Tera-FLOP/s (trillion floating-point operations per second) systems in 1997 to Peta-FLOP/s (quadrillion) in 2012 took 15 years. The subsequent jump to Exa-FLOP/s (quintillion) took only 10 years, largely driven by the adoption of GPU accelerators.

## CPU vs. GPU: A Tale of Two Architectures

At first glance, a Central Processing Unit (CPU) and a GPU are both silicon chips with processing cores. However, their internal architectures are fundamentally different, reflecting their distinct design philosophies.

- **A CPU is a latency-oriented device.** It is designed to execute a single thread of instructions as fast as possible. To achieve this, a significant portion of its die area is dedicated to sophisticated control logic, large caches, branch predictors, and speculative execution units. This complexity minimizes the time (latency) for any given task, making it ideal for general-purpose, sequential workloads like operating systems and desktop applications.
- **A GPU is a throughput-oriented device.** Its goal is to execute thousands of parallel threads simultaneously to maximize the total amount of work done. To achieve this, it dedicates the vast majority of its silicon to a massive number of simpler arithmetic logic units (ALUs). It sacrifices single-thread performance and complex control logic in favor of raw parallel processing power. Instead of minimizing latency, a GPU tolerates latency by having so many threads that while some are waiting for data from memory, others can be actively executing.

### Mind the Memory Hierarchy

The architectural differences extend to the memory system. A diagram of a typical CPU memory hierarchy shows a large main memory (TBs) with relatively low bandwidth (e.g., 20 GB/s), supported by several layers of increasingly fast and small caches (LLC, L2, L1) and a tiny set of registers.

A GPU memory hierarchy, in contrast, is designed for massive bandwidth to feed its thousands of threads. It features its own dedicated high-bandwidth memory (HBM) on the device, with bandwidth reaching nearly 2 TB/s in modern architectures. Within the GPU, a diagram would illustrate that each thread has private registers, threads within a block share a fast on-chip Shared Memory (or L1 cache), and all threads on the GPU can access a larger L2 cache and the main global GPU memory. The bandwidth at the register file level can be enormous, reaching tens of TB/s, which is essential for keeping the parallel execution units fed with data.

### Fundamental Laws of Performance Scaling

When moving from serial to parallel computing, it is crucial to understand the theoretical limits of performance improvement. Two laws are fundamental: Moore's Law, which we have discussed, and Amdahl's Law.

#### Amdahl's Law

Amdahl's Law is a model used to find the maximum expected improvement to an overall system when only a part of the system is improved. In parallel computing, it helps us understand the limits of speed-up.

The central idea is that every program contains a serial part and a parallel part.

- **The parallel fraction ($p$):** The portion of the program's execution time that can be perfectly parallelized.
- **The serial fraction ($s$):** The portion that must be run sequentially on a single processor.
- **Relationship:** By definition, $s + p = 1$.

If we use $N$ parallel execution units (e.g., processor cores), the parallel part of the program can be sped up by a factor of $N$. The serial part, however, receives no speed-up. The overall speed-up, a, is therefore given by the formula:

$$
a = \frac{s + p}{s + \frac{p}{N}} = \frac{1}{s + \frac{p}{N}} = \frac{1}{(1 - p) + \frac{p}{N}}
$$


The key insight from Amdahl's Law is that the serial fraction s places a hard limit on the maximum possible speed-up, regardless of how many processors $N$ you add. As $N$ approaches infinity, the term $p/N$ approaches zero, and the maximum speed-up converges to $1/s$. For example, if $90$% of your program is parallel ($p = 0.9$), the serial fraction is $10$% ($s = 0.1$). The maximum speed-up you can ever achieve is $1 / 0.1 = 10 \times$, even with a million cores.

Gene Amdahl, who formulated this law in 1967, originally used it to argue that the single-processor approach was superior. However, his law can be viewed from different perspectives:

- **Optimistic View:** The law, as stated, doesn't account for the overhead of parallelization (e.g., communication, synchronization), which in reality makes achieving the theoretical speed-up even harder.
- **Pessimistic View:** The law assumes a fixed problem size. In practice, as we get more processors ($N$), we often want to solve larger problems. Gustafson's Law (1988) offers an alternative perspective, suggesting that for larger problems, the parallel fraction $p$ can increase, leading to better scalability. Furthermore, sometimes using more processors can lead to superlinear speed-up ($a > N$) due to caching effects, where a larger total cache size allows the problem to fit entirely in faster memory.

#### The GPU Programming Model: A Glimpse

To manage the immense parallelism, GPUs employ a specific execution and programming model. While the hardware itself is a multi-core vector architecture (SIMD - Single Instruction, Multiple Data), it presents a simpler abstraction to the programmer.

- **Hardware View (SIMD):** The hardware groups threads together and executes them on wide vector units. A single instruction is fetched and executed simultaneously on multiple data elements.
- **Software View (SIMT):** The programmer writes code for a single scalar thread, and the hardware and compiler manage the complexity of grouping these threads into "warps" or "wavefronts" for execution. This is known as SIMT (Single Instruction, Multiple Thread). It gives the programmer the illusion of writing simple scalar code while the hardware provides the efficiency of a vector architecture.

This model is a near-perfect incarnation of the Bulk-Synchronous Parallel (BSP) model proposed by Leslie Valiant in 1990. The BSP model structures parallel computation into a sequence of "supersteps," where each step consists of:

1. Compute: All processors perform local computations in parallel.
2. Communicate: Processors exchange necessary data.
3. Synchronize: A barrier synchronization ensures all processors have completed the step before moving to the next.

A key concept in this model is parallel slackness, which refers to having many more virtual processors (threads) than physical processors. This slackness ($v ≫ p$) is precisely what GPUs leverage to hide memory latency and schedule computation efficiently.

> GPUs hide long memory latencies by running many more independent threads ($v$) than they have execution lanes/pipelines ($p$). When some threads stall on memory, the hardware instantly swaps to other ready threads—so the lanes stay busy. That “excess” of runnable work over hardware lanes is the slackness $v ≫ p$.

## CUDA

### CUDA and GPU Overview

In a typical computer system, the CPU and GPU are distinct components with their own dedicated memory systems, connected via an I/O bridge.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/cpu-gpu-diagram.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
    <figcaption>CPU + GPU System</figcaption>
  </figure>
</div>

A diagram of a modern system illustrates this separation: The CPU is connected to its Host Memory (system RAM) through a high-speed memory interface. The GPU, a separate component on the peripheral bus (like PCIe), is connected to its own dedicated, high-bandwidth GPU Memory.

The performance differences, particularly in memory bandwidth and computational throughput, are staggering. The GPU's memory bandwidth can be over 7 times higher than the CPU's, and its computational throughput can be an order of magnitude greater. This massive throughput is precisely what we aim to leverage with GPU computing. The CUDA (Compute Unified Device Architecture) platform allows us to use this power not just for graphics, but for general-purpose computing tasks.



#### The GPU Architecture for General-Purpose Computing

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/gpu-graphics.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <figcaption>G80 Architecture for graphics processing</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/gpu-general.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <figcaption>G80 Architecture for general-purpose processing</figcaption>
  </figure>
</div>

While originally designed for the graphics pipeline (processing vertices, pixels, etc.), the architecture of GPUs has been generalized for computation. The NVIDIA G80 architecture was a pivotal step in this evolution.

A simplified diagram for general-purpose processing shows how a program interacts with the GPU:

1. The Host (the CPU) sends a command to the GPU to start a computation.
2. An Input Assembler and Thread Execution Manager on the GPU receive this command.
3. The work is distributed across an array of Streaming Multiprocessors (SMs).

A Streaming Multiprocessor (SM) is the fundamental processing unit of a CUDA-capable GPU. You can think of it as a group of simple cores that execute threads in parallel. Each SM has its own execution units, schedulers, and a small, fast, on-chip memory called Shared Memory.

All SMs on the GPU can access a large, shared Global Memory through a system of parallel data caches. The host CPU initiates data transfers to and from this Global Memory to set up computations and retrieve results. This load/store architecture, where data is explicitly moved between different memory spaces, is a central concept in GPU programming.

### CUDA Programming

CUDA is an extension of the C programming language created by NVIDIA that exposes the GPU's parallel architecture directly to the developer. CUDA extends C with three main abstractions:
1. **hierarchy of threads**
2. **shared memory**
3. **barrier synchronization**

#### CUDA Programming Model

A CUDA program is a hybrid program consisting of two parts: a host part that runs on the CPU and a device part that runs on the GPU.

- The **CPU (host) part** is responsible for serial or low-parallelism tasks, such as setting up data, managing memory transfers, and launching computations on the GPU.
- The **GPU (device) part** handles massively parallel operations by executing kernels across many threads.

#### Thread Hierarchy

The most fundamental concept in CUDA is the thread hierarchy. When you launch a computation on the GPU, you are launching a kernel function that is executed by a grid of threads. This hierarchy is organized into three levels:

- **Thread**: The smallest unit of execution. A single thread executes one instance of the kernel code.
- **Block**: 
  - A group of threads. All blocks are equal size. 
  - Threads within the same block can cooperate by sharing data through a fast, on-chip **shared memory** and can synchronize their execution using **barriers**. 
  - Threads from different blocks cannot interact. Exception: **global memory**. 
- **Grid**: A group of blocks. A kernel is launched as a single grid of thread blocks. Blocks within a grid are executed independently and in any order, and they cannot directly synchronize with each other.

This hierarchical structure allows you to naturally map the parallelism in your problem onto the GPU hardware. For example, to process a 2D image, you might launch a 2D grid of blocks, where each block processes a tile of the image and each thread within a block processes a single pixel.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/thread_hierarchy.png' | relative_url }}" alt="Thread hierarchy 1" loading="lazy">
    <figcaption>Thread hierarchy 1</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/thread_communication.png' | relative_url }}" alt="Thread hierarchy 2" loading="lazy">
    <figcaption>Thread hierarchy 2</figcaption>
  </figure>
</div>


#### Launching a Kernel

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

**Unique Thread Identification**

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

**Scaling Up with Grids**

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
// Super fine-grained: one thread computes one element
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
  // For N=50, grid size = (50+16-1)/16 = 4.0625 => 4.
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);

  // Launch the kernel
  matAdd<<<dimGrid, dimBlock>>>(A, B, C);
  //     grid size, block size
}
```

**Key Improvements**

1. Global Index Calculation: The kernel now correctly computes global `i` and `j` indices, allowing threads from different blocks to work on different parts of the matrix.
2. Boundary Check: The `if (i < N && j < N)` statement is crucial. Because we must launch a whole number of blocks, the total number of threads launched might be greater than the number of elements in our matrix. This check ensures that only threads corresponding to valid matrix elements perform a write, preventing memory corruption.
3. Grid Calculation: The formula `(N + dimBlock.x - 1) / dimBlock.x` is a standard C/C++ integer arithmetic trick for calculating the ceiling of a division. It ensures we launch enough blocks to cover all `N` elements. For example, if `N=50` and `dimBlock.x=16`, the calculation is `(50 + 16 - 1) / 16 = 65 / 16`, which results in 4 in integer division, correctly launching 4 blocks to cover the 50 elements.

**Choosing Block and Grid Sizes**

- Threads per Block: This should be a multiple of the warp size (typically 32, a concept we'll cover later). A common starting point is 128, 256, or 512 threads per block. The ideal number balances resource usage with the ability to hide memory latency. A range of 100-1000 threads is often optimal.
- Blocks per Grid: You should launch enough blocks to keep all the SMs on the GPU busy. A good heuristic is to launch at least twice as many blocks as there are SMs on your GPU.
- Number of blocks is limited: $512 \times 512 \times 64 \to 1024 \times 1024 \times 64$. Depends on GPU.
- Number of blocks is limited. Depends on GPU.

#### Thread Communication and Synchronization

A key feature of the CUDA model is that threads within the same block can cooperate. This is achieved through two main mechanisms:

- **Shared Memory**: A small, fast, on-chip memory that is shared by all threads in a block. Access to shared memory is much faster than global memory, making it ideal for caching frequently used data or for intermediate results.
- **Barrier Synchronization**: Threads in a block can be synchronized by calling the `__syncthreads()` intrinsic. When a thread reaches this function, it pauses until every other thread in its block has also reached the same point. This is essential for coordinating memory accesses, for example, ensuring all threads have finished loading data into shared memory before any thread starts consuming it.

> **Important Limitation:** Threads from different blocks cannot directly communicate or synchronize with each other. They operate independently. The only way they can "communicate" is by reading and writing to global memory. However, the guarantees for when writes from one block become visible to another are very weak. If you need global synchronization across all threads, you must terminate the current kernel and launch a new one.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/gpu_global_memory.png' | relative_url }}" alt="GPU global memory" loading="lazy">
    <figcaption>GPU global memory</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/gpu_shared_memory.png' | relative_url }}" alt="GPU shared memory" loading="lazy">
    <figcaption>GPU shared memory</figcaption>
  </figure>
</div>

### The CUDA Memory Hierarchy

Understanding the memory hierarchy is critical for writing high-performance CUDA code. A thread has access to several distinct memory spaces, each with different characteristics regarding scope, lifetime, and speed.

A diagram of the memory hierarchy shows that each Thread has its own private Registers. A group of threads in a Block shares a common Shared Memory. All blocks in the Grid can access the larger but slower Global Memory. The Host (CPU) also interacts with the device via this Global Memory.

**Global Memory**

- **Scope:** Accessible by all threads in the grid (R/W), as well as the host (CPU), communication between host and device.
- **Lifetime:** Persists for the lifetime of the application, beyond the execution of any single kernel.
- **Characteristics:** Large (often many gigabytes) but has high latency. This is the primary memory used for transferring data between the host and the device. Accesses to global memory are very sensitive to access patterns, and uncoalesced (scattered) accesses can severely degrade performance.

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

> **Note:** When calling a kernel, you can only pass pointers to device memory (like `d_mem`), not host memory.

**Shared Memory**

- **Scope:** Accessible only by threads within the same block.
- **Lifetime:** Persists only for the lifetime of the block. Once a block finishes executing, its shared memory is gone.
- **Characteristics:** Very fast on-chip memory. In the best case, access latency is similar to registers. It is organized into banks, and parallel access is possible as long as threads do not access addresses in the same bank (a "bank conflict"). Bank conflicts cause accesses to be serialized, reducing performance.

#### CUDA Language Extensions

CUDA extends C/C++ with special specifiers for declaring variables and functions.

**Variable Declaration Specifiers**

These specifiers determine where a variable is stored and its scope.

| Location Specifier | Memory Space | Scope | Lifetime |
| --- | --- | --- | --- |
| `__device__ float var;` | Global Memory | All threads + Host API | Application |
| `__constant__ float var;` | Constant Memory | All threads + Host API | Application |
| `__shared__ float var;` | Shared Memory | All threads in block | Block |
| `texture <float> ref;` | Texture Memory | All threads + Host API | Application |

`__device__` can be combined with others.

A key function related to shared memory is `__syncthreads()`. This intrinsic creates a barrier, forcing all threads in a block to wait until everyone has reached this point. It is essential for managing dependencies when using shared memory, ensuring that data is fully written before it is read by other threads.

**Function Declaration Specifiers**

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

**Type Specifiers**

CUDA introduces several built-in types:

- Vector types: Such as `float2`, `float4`, `int2`, `int4`, which are simple structs containing 2 or 4 components. These are useful for representing coordinates or colors and can lead to more efficient memory access.
- `dim3` type: A struct based on `uint3` used for specifying dimensions for grids and blocks. Unspecified components are automatically initialized to 1.

#### Compilation and Execution

CUDA code is compiled using the `nvcc` (NVIDIA C Compiler) driver. `nvcc` is a powerful tool that separates the host and device code.

1. It processes the CUDA source code, separating host (`__host__`) code from device (`__global__`, `__device__`) code.
2. The host code is compiled by a standard C++ compiler like `g++` or `clang`.
3. The device code is compiled into PTX (Parallel Thread Execution) code.

PTX is a virtual machine and instruction set architecture (ISA) for GPUs. It acts as a stable assembly-like language for the GPU. This is a key part of CUDA's forward compatibility. When you compile your code, `nvcc` can embed the PTX in your executable. When you run your application, the GPU driver performs a final Just-In-Time (JIT) compilation step, translating the PTX into the specific machine code for the target GPU (e.g., GF100, GK110, GP100) you are running on.

Finally, `nvcc` links the compiled host and device code with the necessary CUDA libraries (`cudart`, `cuda`) to produce the final executable.

### A Complete Example: SAXPY

SAXPY stands for Scalar Alpha X Plus Y. It is a common, simple vector operation used to benchmark computational performance. The operation is defined by the formula:

$$
 y[i] = \alpha \cdot x[i] + y[i] 
$$

Here, `x` and `y` are vectors, α (alpha) is a scalar, and `i` is the index of the element. This is an ideal problem for GPU acceleration because the calculation for each element `y[i]` is completely independent of all other elements.

#### Serial CPU Implementation

A standard C implementation of SAXPY uses a simple `for` loop.

```c++
// Kernel function (CPU)
void saxpy_serial(int n, float alpha, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = alpha * x[i] + y[i];
  }
}
```

#### Parallel CUDA Implementation

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

#### Performance Considerations: Pinned Memory

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

### Device Properties and Common Errors

You can query the properties of the GPU in your system to make informed decisions about kernel launch configurations. The `deviceQuery` utility provides a survey of these properties.

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

- **CUDA Error: the launch timed out and was terminated:** The kernel took too long to execute. This often happens on systems with a graphical display, where the OS will kill a kernel to prevent the screen from freezing. A common solution is to stop the X11 server.
- **CUDA Error: unspecified launch failure:** This is a generic error that often indicates a segmentation fault inside the kernel, such as accessing an array out of bounds or dereferencing an invalid pointer.
- **CUDA Error: invalid configuration argument:** The kernel launch configuration is invalid. Common causes include requesting too many threads per block (e.g., > 1024) or requesting more resources (shared memory, registers) per thread than are available on the SM.
- **error: identifier "__eh_curr_region" is undefined:** A compiler problem often related to using non-static allocation for shared memory. Ensure shared memory arrays are declared with static sizes.


## The Modern GPU Architecture

### Vector Architectures: The Foundation of Efficiency

The underlying hardware of a GPU is a vector machine leveraging **Vector ISAs (Instruction Set Architectures)**, which are efficient in three key ways:

  * **Compact:** A single instruction defines many operations, amortizing the cost of instruction fetch/decode and reducing branches.
  * **Parallel:** The operations are data-parallel (no dependencies), simplifying the hardware.
  * **Expressive:** Vector memory instructions describe regular access patterns, allowing the hardware to prefetch data and amortize memory latency.


### The GK110 Architecture: A High-Level View

> **Note:** We will use the NVIDIA GK110 "Kepler" architecture as a representative example.

The GK110 chip consists of up to 15 **Streaming Multiprocessors (SMX)**, which are the main computational engines.  These are connected to Memory Controllers (MCs) and a large L2 cache, communicating with the CPU and host memory via a PCIe 3.0 interface.

### The Streaming Multiprocessor (SMX): The "GPU Core"

The **Streaming Multiprocessor**, often abbreviated as **SM** (or **SMX** in the Kepler architecture), is the true "core" of the GPU where threads are scheduled and instructions are executed. Each GK110 SMX contains:

  * **192 SP** (Single-Precision) units
  * **64 DP** (Double-Precision) units
  * **32 Load/Store (LD/ST)** units
  * **32 Special Function Units (SFUs)** (for sine, cosine, etc.)

To manage execution, each SMX also contains **4 warp schedulers**. A key design philosophy of the GK110 was optimizing for **performance-per-watt** by reducing clock frequency (which has a cubic relationship with power) while increasing parallelism.

-----

## The GPU Memory Hierarchy

Understanding the memory hierarchy is critical for high-performance GPU programming. Unlike CPUs with deep, transparent cache hierarchies, GPUs feature a complex, multi-level memory hierarchy that is **manually controlled by the programmer**. On a GPU, caches are used less for reducing latency and more for reducing memory contention and **coalescing** memory accesses.

### A Collaborative Approach

The GPU philosophy is built on collaboration:

  * **Collaborative Computing:** In CUDA, you typically launch one thread per output element, grouped into **thread blocks**. Schedulers use the massive number of threads (parallel slack) to keep hardware busy.
  * **Collaborative Memory Access:** Memory access should be a team sport. Threads within a block work together to load data efficiently. The memory controllers are optimized to exploit this, especially through **memory coalescing**.

> **Key Takeaway:** If you do something on a GPU, do it collaboratively with all threads.

### The Levels of Memory

The GPU memory hierarchy can be understood by its scope—what threads can "see" which memory.

| Memory Space | Scope | GK110 Size (Per SM/Device) | Description |
| :--- | :--- | :--- | :--- |
| **Registers** | Per Thread | 64k total per thread block | Fastest memory. Private to a single thread. |
| **Local Memory** | Per Thread | Part of Global Memory | Slow, off-chip memory. Private to a thread. Used for **register spilling**. |
| **Shared Memory** | Per Thread Block | 16-48 kB per SM | Fast, on-chip memory. Shared by all threads in a block. Acts as a user-managed **scratchpad**. |
| **L1 Cache** | Per Thread Block | 16-48 kB per SM | On-chip cache. Shares physical hardware with Shared Memory. |
| **Read-Only Cache** | Per Device | 48 kB per SM | Cache for constant and texture data. |
| **L2 Cache** | Per Device | 1.5 MB | Large on-chip cache shared by all SMs. Acts as a victim cache. |
| **Global Memory** | Per Device | \~6 GB (GDDR, off-chip) | Main GPU memory. Large but slow (400-600 cycle latency). |
| **Host Memory** | System-wide | Multiple TBs (off-device) | Main system RAM, connected to the CPU. Accessible via PCIe. |

### Deeper Dive into Memory Types

#### Registers and Local Memory

Each thread has private **registers**, the fastest memory. The total number of registers on an SM is finite (64k per block on GK110). If a thread requires too many registers (max 255), the compiler performs **register spilling**, moving some variables to **Local Memory**.

Despite its name, **Local Memory** is not on-chip; it is a private section of the slow, off-chip **Global Memory**. Stores to local memory are cached in the L1 cache.

#### Shared Memory and L1 Cache

Each thread block has access to fast, on-chip **Shared Memory**. This is a critical optimization tool, used as an explicit, user-controlled cache (**scratchpad**) to orchestrate data movement, reduce global memory traffic, and enable inter-thread communication.

On Kepler, the L1 Cache and Shared Memory are backed by the same physical hardware and can be configured in different size ratios (e.g., 48kB Shared/16kB L1). The L1 cache is not coherent and also serves spills from registers.

#### Global Memory and L2 Cache

**Global Memory** refers to the large pool of GDDR memory on the graphics card. It has very high bandwidth but also very high latency (400-600 cycles). The **L2 Cache** is a large, on-chip cache shared by all SMs, designed to reduce contention on the global memory subsystem. The GPU's memory subsystem is fully featured, with support for virtual addresses, an MMU, and a TLB.

#### Host Memory

This is the main system RAM attached to the CPU. Data must be transferred between host and GPU global memory.

  * `cudaMemcpy`: This function explicitly transfers data using the GPU's DMA (Direct Memory Access) engines.
  * **Pinned Memory:** Standard host memory is "pageable" (unpinned). The GPU must copy it to a "staging buffer" first. **Pinning** memory (e.g., with `cudaMallocHost`) locks it in physical RAM, allowing for autonomous device access and faster transfers.
  * **Zero Copy:** On modern GPUs (Compute Capability \>= 2.0), threads can directly operate on pinned host memory.

A system diagram shows the CPU/Host Memory connection (\~64 GB/s) is vastly different from the GPU-GDDR connection (\~460 GB/s) and internal GPU memory bandwidth (\~3.3 TB/s).

### Global Memory Coalescing: The Key to Bandwidth

**Coalescing** is the process of combining many fine-grained memory accesses from multiple threads in a warp into a single, large GDDR memory transaction. This is paramount for achieving high bandwidth.

On Kepler, the L1 cache line size is 128 bytes (latency-optimized) and the L2 cache line size is 32 bytes (bandwidth-optimized). When threads in a warp access memory, the ideal pattern is for them to access contiguous, aligned locations.

#### Access Penalties

  * **Offset Access:** `data[addr + offset]`. If a warp's access crosses a cache line boundary, it may require fetching 5 cache lines instead of 4, reducing effective bandwidth.
  * **Strided Access:** `data[addr * stride]`. A stride of 2 means only half the data loaded into a cache line is used, resulting in 50% load/store efficiency.

The solution is to **manually control data movement**. A common pattern is to have threads collaboratively load a "tile" of data from global memory into shared memory in a coalesced manner, then perform computation using the fast shared memory.

> One of the GPU’s main advantages is memory bandwidth: **coalescing is of utmost importance\!**

-----

## GPU Execution and Scheduling

GPU execution is designed to manage hundreds of thousands of threads to hide the massive latencies of memory access.

### Latency Hiding and Tolerance

Latency is the delay between requesting and receiving data (e.g., 400-600 cycles for global memory). GPUs primarily use **multi-threading** to tolerate this latency.

| Property | Relaxed Consistency Models | Prefetching | Multi-Threading | Block Data Transfer |
| :--- | :--- | :--- | :--- | :--- |
| Types of latency | Write | Write | Write, Read | Write, Read |
| Synchronization | Write | Read | Read | - |
| Software requirements | Labeling sync ops | Predictability | **Explicit extra concurrency** | Identifying and orchestrating block transfers |
| Extra hardware support | Little | Little | **Substantial** | Not in processor, but in memory system |
| Supported in systems? | Yes | Yes | **Yes** | (Yes) |

As the table shows, multi-threading requires substantial hardware support but relies on the software providing explicit extra concurrency—which is the massive number of threads launched in a CUDA kernel.

### The Warp: The Unit of Scheduling

The GPU hardware does not manage individual threads. Instead, it groups them into a **warp**.

  * A **warp** is a group of **32 consecutive threads** from a thread block.
  * This size (32) is an NVIDIA implementation detail but is fundamental to performance.
  * Warps are the fundamental units for the scheduler.
  * All threads in a warp execute the same instruction at the same time in lock-step.

On Kepler:

  * Up to 1024 threads can be in a **thread block**.
  * One thread block executes entirely on **one SM**.
  * Each thread block is divided into **warps** of 32 threads.
  * One SM can hold multiple thread blocks (up to 4) and up to 32 warps per block.

### The SM Scheduler at Work

Each SM has its own scheduler(s) to keep its execution units busy. This is called **Fine-Grained Multi-Threading (FGMT)**. The scheduling loop is:

1.  Select a thread block and allocate its resources (registers, shared memory).
2.  From that block's warps, select one that is **ready** (operands are available).
3.  Fetch and issue the instruction for the selected warp.
4.  Repeat, allocating resources to new blocks until the SM is full.
5.  If an executing warp **stalls** (e.g., on a memory access), the scheduler **immediately switches context** to another ready warp (this switch is zero-cost).
6.  When all warps in a block finish, its resources are deallocated.

The goal of FGMT is **latency hiding**. With enough active warps, the scheduler can almost always find work, keeping the functional units busy. This process interleaves execution, ensuring the pipeline is always full.

### Hardware Multi-Threading Example (G80)

An older G80 architecture provides a clear example. Assume 4 warp contexts, a 50-cycle memory stall, and a memory access every 20 cycles. You need at least 3 warps to hide latency and **4 warps for full utilization**.

A timing diagram for this G80 example shows four warps (T0, T1, T2, T3).

1.  At time 0, warp T0 begins execution.
2.  After 20 cycles, it issues a memory access and enters a **stall** state.
3.  The scheduler immediately switches to warp T1, which executes for 20 cycles and stalls.
4.  The scheduler switches to T2, and then T3.
5.  By the time T3 stalls, 60 cycles have passed, and the 50-cycle memory access for T0 is complete. T0 is now in a **waiting** state, ready to be scheduled again.

This cycle of executing, stalling, and switching between ready warps ensures the ALUs are constantly fed with instructions.

### Scoreboarding and Instruction Issue

Modern schedulers use a **scoreboard**, a hardware table that tracks the status of instructions for all active warps. It tracks dependencies, resource availability, and outputs. This allows for **Out-of-Order (OOO) execution** *among warps*, preventing data hazards. The scheduler can pick any warp whose dependencies are resolved.

The Kepler scheduler checks the scoreboard and issues an instruction to a ready warp using a prioritized round-robin scheme. The instruction is then broadcast to all 32 threads in that warp.

### The Challenge of Branch Divergence

Since all 32 threads in a warp execute the same instruction, `if-else` statements can cause **branch divergence**.

If some threads in a warp take the `if` path and others take the `else` path, the hardware serializes the execution:

1.  It disables the `else` threads using a **write-mask**.
2.  All threads in the warp traverse the `if` block.
3.  The write-mask is inverted: `if` threads are disabled, `else` threads are enabled.
4.  All threads in the warp traverse the `else` block.

The total execution time is the *sum* of both paths.

```c++
// Kernel 1: High divergence within a warp
// Thread 0 takes 'if', threads 1-31 take 'else'.
// This is slow due to serialization.
__global__ void kernel1(float* out) {
  int id = threadIdx.x;
  if (id % 32 == 0) {
    out[id] = complex_function_call();
  } else {
    out[id] = 0;
  }
}

// Kernel 2: No divergence within a warp
// All threads in the first warp (0-31) take the 'if' path together.
// All threads in other warps take the 'else' path together.
// This is fast.
__global__ void kernel2(float* out) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < 32) {
    out[id] = complex_function_call();
  } else {
    out[id] = 0;
  }
}
```

-----

## Performance and Optimization Summary

### Key Performance Considerations

The most common CUDA performance issues are:

  * **Memory Coalescing:** Failing to access global memory in a coalesced pattern (due to strides or offsets) wastes memory bandwidth.
  * **Latency Hiding:** Not launching enough threads/warps leaves the scheduler with no work when a stall occurs, idling the hardware. Conversely, too many threads/block can cause **register spilling** to slow local memory.
  * **Divergent Branching:** When threads within a warp follow different control flow paths, their execution is serialized, nullifying parallelism.

### Summary of Key Concepts

  * **Memory Hierarchy:** GPUs feature a manually-controlled, flat memory hierarchy. Caches are primarily for coalescing accesses, not hiding latency.
  * **Parallel Slackness:** The BSP model's concept of $v \gg p$ is key to the GPU's latency hiding capabilities.
  * **The Warp:** The instruction stream on a GPU corresponds to a **thread warp** (32 threads), not a single thread.
  * **Optimization:** Performance optimization revolves around maximizing memory coalescing, ensuring sufficient active warps, and minimizing branch divergence.

### Advanced Memory Analysis: Pointer Chasing

It is possible to empirically analyze a GPU's memory subsystem using **pointer chasing**. This involves creating a long linked list in memory and measuring the time to traverse it. By varying the stride and total size, you can infer properties of the caches and TLB.

An analysis of the GeForce 8800 GTX revealed:

  * **L1 Cache:** A sharp latency increase at 5.5 kB (implying a 5 kB L1 cache) and at a 32-byte stride (implying a 32-byte cache line).
  * **L2 Cache:** Similar analysis suggested a 24-way set-associative L2.
  * **TLB:** A latency increase at 128 MB pointed to a TLB. Saturation at a 512 kB size indicated a 512 kB page size, and further tests suggested a 16-entry, fully-associative TLB.