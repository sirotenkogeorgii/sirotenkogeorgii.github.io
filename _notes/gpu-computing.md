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

<style>
  .accordion summary {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    background-color: var(--accent-soft, #f5f6ff);
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--accent-strong, #2c3e94);
    border-radius: 0.25rem;
  }
</style>

# GPU Computing

[summary](/subpages/gpu-computing/summary)

- https://www.youtube.com/watch?v=_qSP455IekE
- https://www.youtube.com/watch?v=OSpy-HoR0ac
- https://www.youtube.com/watch?v=1Goq8Yc3dfo
- https://www.youtube.com/watch?v=7JYUejkK-Fo
- https://www.youtube.com/watch?v=K9anz4aB0S0
- https://www.youtube.com/watch?v=ajKpsY3ojAo

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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Amdahl's Law)</span></p>

Amdahl's Law models the maximum expected improvement to an overall system when only a part of the system is improved. Every program contains a serial part and a parallel part.

- **The parallel fraction ($p$):** The portion of the program's execution time that can be perfectly parallelized.
- **The serial fraction ($s$):** The portion that must be run sequentially on a single processor.
- **Relationship:** By definition, $s + p = 1$.

If we use $N$ parallel execution units, the overall speed-up $a$ is:

$$
a = \frac{s + p}{s + \frac{p}{N}} = \frac{1}{s + \frac{p}{N}} = \frac{1}{(1 - p) + \frac{p}{N}}
$$

The serial fraction $s$ places a hard limit on the maximum possible speed-up. As $N \to \infty$, the maximum speed-up converges to $1/s$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Amdahl's Law)</span></p>

If $90$% of your program is parallel ($p = 0.9$), the serial fraction is $10$% ($s = 0.1$). The maximum speed-up you can ever achieve is $1 / 0.1 = 10 \times$, even with a million cores.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Perspectives on Amdahl's Law)</span></p>

Gene Amdahl, who formulated this law in 1967, originally used it to argue that the single-processor approach was superior. However, his law can be viewed from different perspectives:

- **Optimistic View:** The law doesn't account for the overhead of parallelization (e.g., communication, synchronization), which in reality makes achieving the theoretical speed-up even harder.
- **Pessimistic View:** The law assumes a fixed problem size. Gustafson's Law (1988) suggests that for larger problems, the parallel fraction $p$ can increase, leading to better scalability. Furthermore, sometimes using more processors can lead to superlinear speed-up ($a > N$) due to caching effects.

</div>

#### The GPU Programming Model: A Glimpse

To manage the immense parallelism, GPUs employ a specific execution and programming model. While the hardware itself is a multi-core vector architecture (SIMD - Single Instruction, Multiple Data), it presents a simpler abstraction to the programmer.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SIMD vs. SIMT)</span></p>

- **Hardware View (SIMD):** The hardware groups threads together and executes them on wide vector units. A single instruction is fetched and executed simultaneously on multiple data elements.
- **Software View (SIMT):** The programmer writes code for a single scalar thread, and the hardware and compiler manage the complexity of grouping these threads into "warps" or "wavefronts" for execution. This is known as SIMT (Single Instruction, Multiple Thread). It gives the programmer the illusion of writing simple scalar code while the hardware provides the efficiency of a vector architecture.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bulk-Synchronous Parallel Model)</span></p>

This model is a near-perfect incarnation of the **Bulk-Synchronous Parallel (BSP)** model proposed by Leslie Valiant in 1990. The BSP model structures parallel computation into a sequence of "supersteps," where each step consists of:

1. **Compute:** All processors perform local computations in parallel.
2. **Communicate:** Processors exchange necessary data.
3. **Synchronize:** A barrier synchronization ensures all processors have completed the step before moving to the next.

A key concept in this model is **parallel slackness**, which refers to having many more virtual processors (threads) than physical processors. This slackness ($v \gg p$) is precisely what GPUs leverage to hide memory latency and schedule computation efficiently.

</div>

> GPUs hide long memory latencies by running many more independent threads ($v$) than they have execution lanes/pipelines ($p$). When some threads stall on memory, the hardware instantly swaps to other ready threads—so the lanes stay busy. That “excess” of runnable work over hardware lanes is the slackness $v ≫ p$.

## CUDA

### CUDA and GPU Overview

In a typical computer system, the CPU and GPU are distinct components with their own dedicated memory systems, connected via an I/O bridge.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/cpu-gpu-diagram.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>CPU + GPU System</figcaption>
</figure>

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Streaming Multiprocessor)</span></p>

A **Streaming Multiprocessor (SM)** is the fundamental processing unit of a CUDA-capable GPU. It is a group of simple cores that execute threads in parallel. Each SM has its own execution units, schedulers, and a small, fast, on-chip memory called Shared Memory.

</div>

All SMs on the GPU can access a large, shared Global Memory through a system of parallel data caches. The host CPU initiates data transfers to and from this Global Memory to set up computations and retrieve results. This load/store architecture, where data is explicitly moved between different memory spaces, is a central concept in GPU programming.

### CUDA Programming

CUDA is an extension of the C programming language created by NVIDIA that exposes the GPU's parallel architecture directly to the developer. CUDA extends C with three main abstractions:
1. **hierarchy of threads**
2. **shared memory**
3. **barrier synchronization**

#### CUDA Programming Model

A CUDA program is a hybrid program consisting of two parts: a host part that runs on the CPU and a device part that runs on the GPU.

- The **CPU (host) part** is responsible for serial or low-parallelism tasks, such as setting up data, managing memory transfers, and launching computations on the GPU.
- The **GPU (device) part** handles massively parallel operations by executing kernels across many threads, SPMD-style.

### Two different types of parallelism

Fine-grain data-level parallelism (DLP): Do the *same* operation on *many* data elements at once.
Thread-level parallelism (TLP): Run multiple (more independent) *threads* concurrently, each with its own instruction stream.

More:
<div class="accordion">
  <details markdown="1">
    <summary>Two different types of parallelism</summary>

Fine-grain **data-level parallelism (DLP)** and **thread-level parallelism (TLP)** are two different ways to run work in parallel. The key difference is *what* gets replicated and scheduled: **operations on many data elements** (DLP) vs **independent instruction streams** (TLP).

## Fine-grain Data-Level Parallelism (DLP)

**Idea:** Do the *same* operation on *many* data elements at once.

* **Unit of parallel work:** individual data elements (or small vectors of elements)
* **How it’s exploited:** SIMD / vector units, GPU warps/wavefronts, vectorized CPU instructions
* **Control flow:** usually **one** control stream (“single instruction”) applied across many lanes
* **Best for:** dense, regular computations: linear algebra, image filters, convolution, elementwise ops, large array loops
* **Typical hardware mapping:**

  * CPU SIMD (AVX/NEON): 4–64-ish elements per instruction depending on type/width
  * GPUs: warps of threads executing in lockstep (SIMT behaves like DLP when threads follow same path)

**Example mental model:**
`for i in 0..N: C[i] = A[i] + B[i]`
Vectorization/GPU executes many `i` in parallel.

**Main limitation:** if elements follow different branches (divergence) or memory accesses are irregular, efficiency drops.

---

## Thread-Level Parallelism (TLP)

**Idea:** Run multiple (more independent) *threads* concurrently, each with its own instruction stream.

* **Unit of parallel work:** threads (or tasks)
* **How it’s exploited:** multi-core CPUs, SMT/Hyper-Threading, GPU thread blocks (at a higher level), OS threads, OpenMP, pthreads, CUDA blocks, TBB tasks
* **Control flow:** **many** independent instruction streams (each thread can take different branches)
* **Best for:** coarse/medium-grain parallel work: serving many requests, running different pipeline stages, parallel search, independent simulations, producer–consumer patterns
* **Typical hardware mapping:**

  * CPU cores run different threads truly concurrently
  * SMT runs multiple threads per core to hide stalls
  * GPUs run huge numbers of threads too, but within a warp execution is still lockstep (so it’s a mix: TLP at grid/block level, DLP-like at warp level)

**Example mental model:**
Thread 1 handles user A’s request, thread 2 handles user B’s request; or thread 1 updates physics for one region, thread 2 for another.

**Main limitation:** overheads and coordination—thread creation/scheduling, synchronization, contention, false sharing.

---

## Quick comparison

* **DLP:** parallelize *across data* (same code, many elements) → *vector lanes / warps*
* **TLP:** parallelize *across threads/tasks* (different control flows possible) → *cores / hardware threads*

A useful rule of thumb:

* If your program is “one loop over big arrays”: **DLP first** (vectorize/GPU-style).
* If your program is “many independent jobs or stages”: **TLP first** (threads/tasks).

</div>


#### Thread Hierarchy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Thread Hierarchy)</span></p>

The most fundamental concept in CUDA is the thread hierarchy. When you launch a computation on the GPU, you are launching a kernel function that is executed by a grid of threads. This hierarchy is organized into three levels:

- **Thread**: The smallest unit of execution. A single thread executes one instance of the kernel code.
- **Block**:
  - A group of threads. All blocks are equal size.
  - Threads within the same block can cooperate by sharing data through a fast, on-chip **shared memory** and can synchronize their execution using **barriers**.
  - Threads from different blocks cannot interact. Exception: **global memory**.
- **Grid**: A group of blocks. A kernel is launched as a single grid of thread blocks. Blocks within a grid are executed independently and in any order, and they cannot directly synchronize with each other.

</div>

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

**Important: It is always a 3D vector, even if you logically use 2D, for example, for matrix multiplication.**

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

In this simple case, we launch a single block (1) with $N\times N$ threads (`dimBlock` of type `dim3`). Each thread uses its `threadIdx.x` and `threadIdx.y` to find its unique (i, j) coordinate and computes a single element `C[i][j]`.

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

1. **Global Index Calculation:** The kernel now correctly computes global `i` and `j` indices, allowing threads from different blocks to work on different parts of the matrix.
2. **Boundary Check:** The `if (i < N && j < N)` statement is crucial. Because we must launch a whole number of blocks, the total number of threads launched might be greater than the number of elements in our matrix. This check ensures that only threads corresponding to valid matrix elements perform a write, preventing memory corruption.
3. **Grid Calculation:** The formula `(N + dimBlock.x - 1) / dimBlock.x` is a standard C/C++ integer arithmetic trick for calculating the ceiling of a division. It ensures we launch enough blocks to cover all `N` elements. For example, if `N=50` and `dimBlock.x=16`, the calculation is `(50 + 16 - 1) / 16 = 65 / 16`, which results in 4 in integer division, correctly launching 4 blocks to cover the 50 elements.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/grid_sketch.png' | relative_url }}" alt="Thread hierarchy 1" loading="lazy">
  <!-- <figcaption>Thread hierarchy 1</figcaption> -->
</figure>

**Choosing Block and Grid Sizes**

- **Threads per Block:** This should be a multiple of the warp size (typically 32, a concept we'll cover later). A common starting point is $128$, $256$, or $512$ threads per block. The ideal number balances resource usage with the ability to hide memory latency. 
  - **A range of $100-1000$ threads is often optimal.**
- **Blocks per Grid:** You should launch enough blocks to keep all the SMs on the GPU busy.
  - **A good heuristic is to launch at least twice as many blocks as there are SMs on your GPU.**
- **Number of blocks is limited:** $512 \times 512 \times 64 \to 1024 \times 1024 \times 64$. Depends on GPU.
- **Number of blocks is limited. Depends on GPU.**

#### Thread Communication and Synchronization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Thread Communication Mechanisms)</span></p>

A key feature of the CUDA model is that threads within the same block can cooperate. This is achieved through two main mechanisms:

- **Shared Memory**: A small, fast, on-chip memory that is shared by all threads in a block. Access to shared memory is much faster than global memory, making it ideal for caching frequently used data or for intermediate results.
- **Barrier Synchronization**: Threads in a block can be synchronized by calling the `__syncthreads()` intrinsic. When a thread reaches this function, it pauses until every other thread in its block has also reached the same point. This is essential for coordinating memory accesses, for example, ensuring all threads have finished loading data into shared memory before any thread starts consuming it.

</div>

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Global Memory)</span></p>

- **Scope:** Accessible by all threads in the grid (R/W), as well as the host (CPU), communication between host and device.
- **Lifetime:** Persists for the lifetime of the application, beyond the execution of any single kernel.
- **Characteristics:** Large (often many gigabytes) but has high latency. This is the primary memory used for transferring data between the host and the device. Accesses to global memory are very sensitive to access patterns, and uncoalesced (scattered) accesses can severely degrade performance.

</div>

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Shared Memory)</span></p>

- **Scope:** Accessible only by threads within the same block.
- **Lifetime:** Persists only for the lifetime of the block. Once a block finishes executing, its shared memory is gone.
- **Characteristics:** Very fast on-chip memory.
  - In the best case, access latency is similar to registers. Typically, $16$–$32$ banks with 32-bit width.
  - It is organized into banks, and
  - parallel access is possible as long as threads do not access addresses in the same bank (a "bank conflict"). Bank conflicts cause accesses to be serialized, reducing performance.

</div>

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

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/cuda_program_compilation1.png' | relative_url }}" alt="Thread hierarchy 1" loading="lazy">
  <!-- <figcaption>Thread hierarchy 1</figcaption> -->
</figure>

### A Complete Example: SAXPY

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SAXPY)</span></p>

**SAXPY** stands for **S**calar **A**lpha **X** **P**lus **Y**. It is a common, simple vector operation used to benchmark computational performance:

$$
y[i] = \alpha \cdot x[i] + y[i]
$$

Here, `x` and `y` are vectors, $\alpha$ is a scalar, and `i` is the index of the element. This is an ideal problem for GPU acceleration because the calculation for each element `y[i]` is completely independent of all other elements.

</div>

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pinned Memory)</span></p>

**Pinned memory** (or page-locked memory) is host memory that the OS cannot page out or relocate. By default, host memory allocated with `malloc` is pageable, meaning the operating system can move it around in physical memory. For the GPU to access this data, the CUDA driver must first copy it into a temporary, pinned buffer before transferring it to the device.

By allocating host memory directly as pinned memory, we eliminate this extra copy. Pinned memory is a scarce resource, so it should be used judiciously.

</div>

```c++
float *h_x;
float *h_y;
float *d_x;
float *d_y;

if (USE_PINNED_MEMORY) {
  cudaMallocHost ( (void**) &h_x, N*sizeof(float) );
  cudaMallocHost ( (void**) &h_y, N*sizeof(float) );
} else {
  h_x = (float*) malloc ( N*sizeof(float) );
  h_y = (float*) malloc ( N*sizeof(float) );
}

cudaMalloc ( (void**)&d_x, N*sizeof(float) );
cudaMalloc ( (void**)&d_y, N*sizeof(float) );
```

Using `cudaMallocHost` instead of `malloc` can lead to a significant reduction in data transfer times, making the GPU's computational advantage more apparent.

**What `cudaMalloc` actually is?**
<div class="accordion">
  <details markdown="1">
    <summary>What `cudaMalloc` actually is</summary>

`malloc()` isn’t “bad” in general — the *problem is what the CUDA driver must do with the memory it gives you* when you call `cudaMemcpy`.

## What’s the issue with ordinary `malloc()` host memory?

Memory you get from `malloc()` is **pageable** host memory:

* The OS is allowed to **page it out** to disk, and/or
* **move** (remap) its physical pages around in RAM at any time.

That’s great for a general-purpose OS, but it’s awkward for GPU transfers because **DMA (direct memory access)** engines on the GPU want to read/write a **stable set of physical pages**.

So when you do:

```cpp
cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
```

the CUDA driver cannot reliably let the GPU DMA directly from `h_x` (because the pages might move). To make the transfer safe, the driver typically:

1. Allocates/uses an internal **pinned (page-locked) staging buffer**
2. **Copies** your data from the pageable `malloc()` buffer into that pinned buffer (CPU-side copy)
3. Performs the actual DMA transfer from the pinned buffer to the GPU

That extra step (2) is the “hidden copy” your paragraph refers to. For large copies — or for many repeated transfers — it can become a dominant cost, and make the GPU version look slower overall.

## How does `cudaMallocHost()` solve it?

`cudaMallocHost()` allocates **pinned (page-locked) host memory**:

* The OS is not allowed to page it out or relocate it.
* The CUDA driver/GPU can DMA to/from that host memory **directly**.

So the transfer path becomes:

* **Pinned host buffer → DMA → device** (no extra staging copy)

That removes the driver’s need to do the intermediate “pageable → pinned” copy, which can substantially reduce host↔device transfer time and CPU overhead.

## Key tradeoff (why pinned memory is “scarce”)

Pinned memory has real costs:

* It **reduces how much RAM the OS can page/manage**, which can hurt overall system performance.
* Too much pinned memory can lead to **worse** performance (OS pressure, reduced caching flexibility).

So it’s best used for:

* Frequently transferred buffers
* Large transfers
* Streaming pipelines (especially combined with `cudaMemcpyAsync`), where pinned memory is required for true async H2D/D2H.

That’s the core story: `malloc()` gives pageable memory → driver must stage into pinned memory anyway → extra copy. `cudaMallocHost()` gives pinned memory up front → enables direct DMA → faster transfers (when transfers are the bottleneck).

You’re basically on the right track — **for a single `cudaMemcpy`**, using `cudaMallocHost` often *looks like* you’re just “doing the driver’s staging manually”.

But there are two important differences:

## 1) You remove the *extra* copy (you don’t just “move it earlier”)

With pageable `malloc` memory, the path is usually:

* **your data in pageable buffer (malloc)**
* **driver copies it into a pinned staging buffer**  ← extra CPU copy
* **GPU DMA from pinned staging buffer to device**

With `cudaMallocHost`, the path is:

* **your data already in pinned buffer**
* **GPU DMA from that pinned buffer to device**

So step “copy into pinned staging” is not “merged” with anything — it’s **gone**.

If you allocate pinned memory but still *fill it* from some other pageable buffer, then yes, you reintroduce a copy — but that’s a copy you only do if your program design forces it. The point is: **your real working buffer can itself be pinned**, so you can produce/consume data directly in it.

## 2) The driver’s staging is not necessarily reusable the way your pinned buffer is

When you call `cudaMemcpy` from pageable memory, the driver may:

* allocate staging buffers on demand,
* manage them internally,
* potentially copy in chunks,
* and you don’t control reuse/lifetime.

If you allocate `h_x` with `cudaMallocHost`, you get:

* a pinned buffer with a stable lifetime,
* reuse across many transfers,
* predictable behavior,
* and (crucially) it enables true overlap with `cudaMemcpyAsync` (next point).

## 3) Pinned memory unlocks *true async copies and overlap*

This is often the *real* win.

* Transfers from **pageable** memory often can’t be truly asynchronous: the runtime may have to stage/copy first, which forces synchronization.
* Transfers from **pinned** memory can be used with `cudaMemcpyAsync` so you can overlap:

  * CPU preparing the next chunk
  * GPU computing on the current chunk
  * DMA copying the next/previous chunk

That overlap is impossible (or much weaker) if every transfer requires an internal staging copy.

---

### Quick mental model

* With `malloc`: **pageable → (hidden copy to pinned) → DMA → GPU**
* With `cudaMallocHost`: **pinned → DMA → GPU**

So it’s not “doing step1+step2 in one step”; it’s **eliminating the staging copy entirely**, *provided your program works directly with that pinned buffer*.

  </details>
</div>

**Why CUDA driver cannot reliably let the GPU DMA directly from `h_x` if it is paged?**
<div class="accordion">
  <details markdown="1">
    <summary>Why CUDA driver cannot reliably let the GPU DMA directly from `h_x` if it is paged?</summary>

Because **DMA needs stable, physical memory**, while pageable (“paged”) memory from `malloc()` does **not** guarantee that.

Here’s the core issue:

## What GPU DMA actually needs

A DMA engine transfers data by reading/writing **physical RAM addresses** (or an IOMMU mapping to physical pages). For the transfer to run safely, the device driver must be able to say:

* “These exact physical pages back `h_x`”
* “They will remain valid and won’t be moved or paged out for the whole duration of the DMA”

## What pageable `malloc()` memory allows the OS to do

With pageable memory, the OS can, at any time:

1. **Page it out** (write the page to disk/swap and reuse the RAM frame for something else)
2. **Move it** (e.g., through memory compaction, NUMA balancing, copy-on-write effects, re-mapping)
3. **Remap it** so the *virtual address* stays the same, but the *physical pages* underneath change

So even if `h_x` is a valid pointer in your process (virtual address), the **physical backing pages can change** while the GPU is mid-transfer.

## What could go wrong if the GPU DMA’d directly anyway

Imagine the GPU starts DMA from the physical page that currently backs `h_x`. While it’s transferring:

* the OS pages that memory out or remaps it
* that physical page frame may get reused for something else

Now the GPU might:

* read garbage / someone else’s data (corruption + security issue),
* trigger an I/O fault the GPU can’t handle like a CPU can,
* or write into memory that no longer belongs to your buffer.

CPUs can handle page faults by trapping into the OS. **A GPU DMA engine cannot “page fault” in the same way** during a raw DMA copy (there are more modern mechanisms like on-demand paging / unified memory, but that’s a different system than a plain `cudaMemcpy` from pageable host memory).

## So what the driver does instead

To make it safe, the driver uses pinned (page-locked) memory:

* pinned memory pages are **locked in RAM**
* the OS cannot page them out or move them
* the driver can build a stable list of physical pages (scatter/gather) for DMA

If your source is pageable (`malloc()`), the driver can’t lock arbitrary pages “just in time” without complications (and doing so repeatedly is expensive and can block), so it often copies into its own pinned staging buffer and DMA’s from there.

## Why `cudaMallocHost()` fixes it

`cudaMallocHost()` allocates memory that is **already page-locked**, so the driver can safely set up DMA from that buffer directly, with no risk that the OS will relocate/page it out mid-transfer.

  </details>
</div>


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

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/GK110-arch.png' | relative_url }}" alt="GPU global memory" loading="lazy">
    <figcaption>GK110 architecture</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/GK110-sm.png' | relative_url }}" alt="GPU shared memory" loading="lazy">
    <figcaption>GK110 SM (Streaming Multiprocessor)</figcaption>
  </figure>
</div>

### The Streaming Multiprocessor (SMX): The "GPU Core"

The **Streaming Multiprocessor**, often abbreviated as **SM** (or **SMX** in the Kepler architecture), is the true "core" of the GPU where threads are scheduled and instructions are executed. Each GK110 SMX contains:

  * **192 SP** (Single-Precision) units
  * **64 DP** (Double-Precision) units
  * **32 Load/Store (LD/ST)** units
  * **32 Special Function Units (SFUs)** (for sine, cosine, etc.)

To manage execution, each SMX also contains **4 warp schedulers**. A key design philosophy of the GK110 was optimizing for **performance-per-watt** by reducing clock frequency (which has a cubic relationship with power) while increasing parallelism.

## The GPU Memory Hierarchy

Understanding the memory hierarchy is critical for high-performance GPU programming. Unlike CPUs with deep, transparent cache hierarchies, GPUs feature a complex, multi-level memory hierarchy that is **manually controlled by the programmer**. 

On a GPU, caches are used less for reducing latency and more for reducing memory contention and **coalescing** memory accesses.

### A Collaborative Approach

The GPU philosophy is built on collaboration:

  * **Collaborative Computing:** In CUDA, you typically launch one thread per output element, grouped into **thread blocks**. Schedulers use the massive number of threads (parallel slack) to keep hardware busy. SIMT – Single Instruction, Multiple Threads.
  * **Collaborative Memory Access:** Memory access should be a team sport. Thread-collective computation and memory accesses. Threads within a block work together to load data efficiently. The memory controllers (MCs) are optimized to exploit that concurrency, especially through **memory coalescing**.

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Victim Cache)</span></p>

A victim cache is **a small, fast, fully associative cache that sits between a primary cache (like a direct-mapped L1) and the next level (L2/memory) to store recently evicted data blocks**, preventing "conflict misses" by giving those "victims" a second chance for quick retrieval, significantly boosting performance by reducing penalties from needing to fetch data from slower memory. Proposed by Norman Jouppi in 1990, it leverages temporal locality, storing discarded lines in case they're needed soon, improving hit rates for memory-intensive programs. 

**How it works**

**Placement:** It's placed in the "refill path" of a main cache, often a direct-mapped L1.
**Eviction:** When a block is kicked out of the L1 cache, it goes into the victim cache instead of being discarded.
**Lookup:** If the L1 misses, the system checks the victim cache.
**Hit:** If found in the victim cache, the data is quickly returned to the CPU, and the block might swap back to L1.
**Miss:** If not in the victim cache, it's fetched from the next level (L2/memory), and the evicted block goes to the victim cach

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/GK110-memory-hierarchy.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>GK110 Memory Hierarchy</figcaption>
</figure>

### Deeper Dive into Memory Types

#### Registers and Local Memory

Each thread has private **registers**, the fastest memory. The total number of registers on an SM is finite (64k per block on GK110). If a thread requires too many registers (max 255), the compiler performs **register spilling**, moving some variables to **Local Memory**.

Despite its name, **Local Memory** is not on-chip; it is a private section of the slow, off-chip **Global Memory**. Stores to local memory are cached in the L1 cache.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Local memory)</span></p>

“Local memory” is *not* a separate physical memory. It’s a **per-thread region in device DRAM** (global memory) that the compiler uses when registers aren’t enough (spills, large per-thread arrays, etc.). The word “local” refers to **scope/visibility** (only that thread can address it), not location.

So why does it get cached in L1?

#### Because the cache decision is based on **where the data lives**, not on its “local vs global” name

Local memory accesses compile down to the same kind of instructions as global memory accesses (loads/stores to an address in device memory). Since the data is in **device DRAM**, it naturally goes through the GPU’s normal memory hierarchy:

**thread → (L1 / per-SM cache) → L2 → DRAM**

Whether an address is “a spilled slot for thread 17” or “an element of a global array” doesn’t change that it’s ultimately a **global-memory address**. If the architecture’s caching policy says “these loads/stores are L1-cacheable”, then local memory traffic benefits too.

#### Why caching local memory is useful

Even though local memory is in DRAM, it often has access patterns that make caching pay off:

* **Spills tend to be reused soon** (save a value, do some work, load it back)
* **Per-thread arrays** may be accessed multiple times in a loop
* The GPU can hide latency better if some of those accesses hit in L1 instead of going to DRAM

#### Important nuance: “stores are cached” is architecture/policy dependent

On many NVIDIA GPUs, **local memory is cached by L1 and L2** (at least for loads; stores may go through write buffers and be visible via L2, with L1 behavior depending on generation and configuration). The key idea remains:

* **Local memory is in global memory (DRAM)**
* **It uses the same cache hierarchy as global memory**
* “Local” just means “private to a thread”

So there’s no contradiction: local memory is a *logical* category (private addressing), but it’s *physically* global memory, so it can be cached in L1 like other global-memory accesses.

</div>

<div class="accordion">
  <details>
    <summary>Register spilling</summary>
    <p><strong>Register spilling</strong> is the process where a computer's compiler moves data from CPU registers to slower memory (like RAM) because it has run out of registers to store all the necessary temporary variables. This is done to allow program execution to continue, but it negatively impacts performance because accessing memory is much slower than using registers.</p>

    <h3>How register spilling works</h3>
    <ul>
      <li><strong>Insufficient registers:</strong> A program needs to store more temporary variables than the CPU has available registers to hold them all at a given time.</li>
      <li><strong>Compiler's solution:</strong> The compiler selects one or more variables to "spill" to memory.</li>
      <li><strong>Data transfer:</strong> The compiler generates "spill code" to move the variable's data from a register to main memory.</li>
      <li><strong>Register reuse:</strong> The freed register can now be used for another variable.</li>
      <li><strong>Data retrieval:</strong> When the program needs the spilled variable again, the compiler generates more code to move the data back from memory into a register.</li>
    </ul>

    <h3>When is happens</h3>
    <p>The decision of which variables to spill is primarily made during compile time, but the actual act of moving the data to and from memory happens at runtime when the program executes.</p>
    <ol>
      <li><strong>Compile time decision:</strong> The compiler analyzes the code's register pressure (how many variables are needed at once) and determines that not all variables can fit into the physical registers. It then generates machine code with explicit instructions to store (spill) data to stack memory and reload it when needed.</li>
      <li><strong>Runtime execution:</strong> When the program runs, the processor executes those specific "load" and "store" instructions inserted by the compiler, moving the data between registers and main memory as necessary.</li>
    </ol>
    <p>Therefore, the planning occurs at compile time, and the execution of the spill operations occurs at runtime.</p>

    <h3>Why register spilling is a performance bottleneck</h3>
    <ul>
      <li><strong>Speed difference:</strong> Register access is significantly faster than memory access, which is why compilers try to keep as many variables as possible in registers.</li>
      <li><strong>Performance hit:</strong> Each time data is spilled to memory and then loaded back, it introduces a delay, slowing down the program.</li>
    </ul>

    <h3>How to avoid or reduce register spilling</h3>
    <ul>
      <li><strong>Compiler optimizations:</strong> Use different compiler flags or settings to try and guide the compiler to reduce spilling, though the effectiveness can vary.</li>
      <li><strong>Code structure:</strong> Rewriting code to use fewer temporary variables can help.</li>
      <li><strong>Shared memory:</strong> On GPUs, using shared memory (which is faster than global memory) for variables that would otherwise be spilled is a common optimization strategy.</li>
    </ul>
  </details>
</div>

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Memory Coalescing)</span></p>

**Coalescing** is the process of **combining many fine-grained memory accesses from multiple threads in a warp into a single, large GDDR memory transaction**. This is paramount for achieving high bandwidth.

</div>

On Kepler, 
* the L1 cache line size is 128 bytes (**latency-optimized**)
  * **latency-optimized, warp-aligned, spatially coherent** $\implies$ **large cache line size**
* the L2 cache line size is 32 bytes (**bandwidth-optimized**)
  * **bandwidth-optimized, multi-client, fine-grained access, matched to GDDR transfer size** $\implies$ **small cache line size**

When threads in a warp access memory, the ideal pattern is for them to access contiguous, aligned locations.

**Misaligned accesses:** One warp is scheduled, but accesses misaligned addresses.

#### Access Penalties

  * **Offset Access:** `data[addr + offset]`. If a warp's access crosses a cache line boundary, it may require fetching 5 cache lines instead of 4, reducing effective bandwidth.
  * **Strided Access:** `data[addr * stride]`. A stride of 2 means only half the data loaded into a cache line is used, resulting in 50% load/store efficiency.

The solution is to **manually control data movement**. A common pattern is to have threads collaboratively load a "tile" of data from global memory into shared memory in a coalesced manner, then perform computation using the fast shared memory.

> One of the GPU’s main advantages is memory bandwidth: **coalescing is of utmost importance\!**

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/global_memory_access_penalty1.png' | relative_url }}" alt="GPU global memory" loading="lazy">
    <!-- <figcaption>GK110 architecture</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/global_memory_access_penalty2.png' | relative_url }}" alt="GPU shared memory" loading="lazy">
    <!-- <figcaption>GK110 SM (Streaming Multiprocessor)</figcaption> -->
  </figure>
</div>

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Warp)</span></p>

The GPU hardware does not manage individual threads. Instead, it groups them into a **warp**.

  * A **warp** is a group of **32 consecutive threads** from a thread block.
  * This size (32) is an NVIDIA implementation detail but is fundamental to performance.
  * Warps are the fundamental units for the scheduler.
  * All threads in a warp execute the same instruction at the same time in lock-step.

</div>

On Kepler:

  * Up to 1024 threads can be in a **thread block**.
  * One thread block executes entirely on **one SM**.
  * Each thread block is divided into **warps** of 32 threads.
  * One SM can hold multiple thread blocks (up to 4) and up to 32 warps per block.

### The SM Scheduler at Work

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fine-Grained Multi-Threading)</span></p>

Each SM has its own scheduler(s) to keep its execution units busy. This is called **Fine-Grained Multi-Threading (FGMT)**. The scheduling loop is:

1.  Select a thread block and allocate its resources (registers, shared memory).
2.  From that block's warps, select one that is **ready** (operands are available).
3.  Fetch and issue the instruction for the selected warp.
4.  Repeat, allocating resources to new blocks until the SM is full.
5.  If an executing warp **stalls** (e.g., on a memory access), the scheduler **immediately switches context** to another ready warp (this switch is zero-cost).
6.  When all warps in a block finish, its resources are deallocated.

The goal of FGMT is **latency hiding**. With enough active warps, the scheduler can almost always find work, keeping the functional units busy.

</div>

### Hardware Multi-Threading Example (G80)

An older G80 architecture provides a clear example. Assume 
* 4 warp contexts and maximum 1 being executed simultaneously, 
* Explicit 32x SIMD instructions
  * 32 ALUs execute a single SIMD instruction
* Register file (RF) is shared among contexts 
  * One register entry (vector) has 32 words (each 32bit)
  * RF: 16 entries -> Max. of 4 registers/warp

**Simplifying assumptions:**
* memory stall every 50-cycle.
* memory access every 20 cycles. 
* you need at least 3 warps to hide latency and **4 warps for full utilization**.

A timing diagram for this G80 example shows four warps (T0, T1, T2, T3).

1.  At time 0, warp T0 begins execution.
2.  After 20 cycles, it issues a memory access and enters a **stall** state.
3.  The scheduler immediately switches to warp T1, which executes for 20 cycles and stalls.
4.  The scheduler switches to T2, and then T3.
5.  By the time T3 stalls, 60 cycles have passed, and the 50-cycle memory access for T0 is complete. T0 is now in a **waiting** state, ready to be scheduled again.

This cycle of executing, stalling, and switching between ready warps ensures the ALUs are constantly fed with instructions.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/example_for_hardware_multithreading_g80.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>Example for Hardware Multi-Threading G80</figcaption>
</figure>

### Scoreboarding and Instruction Issue

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Scoreboarding)</span></p>

Modern schedulers use a **scoreboard**, a hardware table that tracks the status of instructions for all active warps.

* Tracks dependencies, resource availability, and outputs.
* Allows for **Out-of-Order (OOO) execution** *among warps*, to prevent **data hazards**.
* Scheduler can pick any warp whose dependencies are resolved.

</div>

The Kepler scheduler 
1. **checks** the scoreboard
2. **issues** an instruction to a ready warp using a **prioritized round-robin scheme**. 
   * The instruction is then broadcast to all 32 threads in that warp.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/kepler_scheduler_instruction_issue.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>Thread Scheduling on Kepler</figcaption>
</figure>

<div class="accordion">
  <details markdown="1">
    <summary>Data </summary>

Data hazards in computer architecture are **pipeline stalls caused by instruction dependencies**, where an instruction needs data from a previous instruction that hasn't finished yet, leading to incorrect results, most commonly **Read After Write (RAW)** hazards, but also **Write After Read (WAR)** and **Write After Write (WAW)** hazards in parallel systems. These are managed with techniques like **forwarding (bypassing)** (sending data directly) or **stalling the pipeline** (inserting NOPs/bubbles) to ensure data correctness, especially when a load instruction precedes its use. 

**Types of Data Hazards**

* **Read After Write (RAW) / True Dependency:** A subsequent instruction reads a register before a preceding instruction writes to it (e.g., ADD R1, R2, R3 then SUB R4, R1, R5).
* **Write After Read (WAR) / Anti-dependency:** A subsequent instruction writes to a register that a preceding instruction reads from (e.g., SUB R4, R1, R5 then ADD R1, R2, R3).
* **Write After Write (WAW) / Output Dependency:** Two instructions write to the same destination register (e.g., ADD R1, R2, R3 then SUB R1, R4, R5). 

**Causes in Pipelining**

* **Data Dependency:** Instructions need results from earlier instructions still in the pipeline.
* **Timing Mismatch:** The write-back stage for data might be later than the instruction decode stage that needs it, causing a read of an old value. 

**Handling Methods**

* **Stalling (Pipeline Bubbles):** The pipeline pauses, inserting no-operation (NOP) instructions to wait for data to become available.
* **Forwarding (Bypassing):** Data is sent directly from the output of one stage (like ALU) to the input of a later stage that needs it, bypassing the register file.
* **Out-of-Order Execution:** Processors reorder instructions to avoid stalls, though WAR/WAW hazards become more complex. 

  </details>
</div>

### The Challenge of Branch Divergence

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Branch Divergence)</span></p>

Since all 32 threads in a warp execute the same instruction, `if-else` statements can cause **branch divergence**.

If some threads in a warp take the `if` path and others take the `else` path, the hardware serializes the execution:

1.  It disables the `else` threads using a **write-mask**.
2.  All threads in the warp traverse the `if` block.
3.  The write-mask is inverted: `if` threads are disabled, `else` threads are enabled.
4.  All threads in the warp traverse the `else` block.

The total execution time is the *sum* of both paths.

</div>

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

## Performance and Optimization Summary

### Key Performance Considerations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key Performance Pitfalls)</span></p>

The most common CUDA performance issues are:

  * **Memory Coalescing:** Failing to access global memory in a coalesced pattern (due to strides or offsets) wastes memory bandwidth.
  * **Latency Hiding:** Not launching enough threads/warps leaves the scheduler with no work when a stall occurs, idling the hardware. Conversely, too many threads/block can cause **register spilling** to slow local memory.
  * **Divergent Branching:** When threads within a warp follow different control flow paths, their execution is serialized, nullifying parallelism.

</div>

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

## Optimizing with Shared Memory

### Matrix Multiplication: A Case Study

Matrix multiplication is one of the most fundamental operations in scientific computing and artificial intelligence. While simple in principle, it serves as the perfect case study for mastering the advanced optimization techniques required to unlock the true power of a GPU.

We focus on this operation because:
* **Ubiquity:** It is the computational core of Deep Learning and physical simulations.
* **Optimization Depth:** It features complex memory access patterns that highlight critical performance differences between naive and optimized code.
* **Balance:** It is complex enough to require sophisticated optimization (tiling, shared memory) but simple enough to understand comprehensively.

#### Analyzing the Problem

The goal is to compute a result matrix, $C$, by multiplying two input matrices, $A$ and $B$:

$$C = A \cdot B$$

For this analysis, we assume square matrices with dimensions $n \times n$, stored in **row-major order**. This means an element at `[row][column]` is accessed in a flat memory array as `M[row * width + column]`.

#### Computational Cost (FLOPs)

To calculate a single element $C[i][j]$, we perform a dot product of the $i$-th row of $A$ and the $j$-th column of $B$. This requires $n$ multiplications and $n-1$ additions, approximated as $2n$ Floating-Point Operations (FLOPs).

With $n^2$ elements in the output matrix, the total computational cost $f$ is:

$$f = n^2 \text{ elements} \times 2n \text{ FLOPs/element} = 2n^3 \text{ FLOPs}$$

#### Memory Access Cost and Intensity

Performance is rarely limited by pure math; it is limited by memory.

* **Unique Accesses:** Ideally, assuming perfect caching, we load each element of $A$, $B$, and $C$ exactly once. This yields $m_{unique}=3n^2$ unique elements.
* **Total Accesses (Naive):** Without caching, every calculation fetches data from main memory. To compute $n^2$ elements, we perform $2n^3$ reads.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Computational Intensity)</span></p>

**Computational Intensity ($r$)** is the ratio of arithmetic operations to memory operations (FLOPs/Byte). A high ratio indicates a compute-bound algorithm (good); a low ratio indicates a memory-bound algorithm (bad).

</div>

Assuming a perfect cache, the intensity is:

$$r = \frac{f}{m_{unique}} = \frac{2n^3}{3n^2} = O(n)$$

This linear relationship suggests that as matrix size $n$ grows, the algorithm *should* become increasingly compute-intensive. However, this is only true if we can effectively utilize the memory hierarchy to avoid re-fetching data from slow global memory.

### CPU Baseline and Tiling

#### The Naive CPU Approach

A standard implementation uses three nested loops.

```c++
void MatrixMulOnHost(float* M, float* N, float* P, int Width) {
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            float sum = 0;
            for (int k = 0; k < Width; ++k) {
                float a = M[i * Width + k];
                float b = N[k * Width + j];
                sum += a * b;
            }
            P[i * Width + j] = sum;
        }
    }
}
````

This code hits the **Memory Wall**. For small matrices fitting in the CPU cache, performance is high. As matrices grow larger than the cache (e.g., \> 1500 $\times$ 1500), the CPU stalls waiting for data from the main system RAM.

<div class="accordion">
  <details>
    <summary>Cache-aware implementation (CPU)</summary>
    <blockquote>
      <p>See the original post: <a href="https://siboehm.com/articles/22/Fast-MMM-on-CPU" target="_blank" rel="noopener">https://siboehm.com/articles/22/Fast-MMM-on-CPU</a></p>
    </blockquote>

    <p>Multidimensional matrices are represented in memory using a strided representation. In most programming languages you expect the matrix to be row-continuous, meaning that iterating through a single row by incrementing the column results in sequential memory access.</p>

    <div class="note-figure">
      <img src="{{ '/assets/images/notes/gpu-computing/physical-logical-matrix-layout.png' | relative_url }}" alt="Physical vs. logical matrix layout" loading="lazy">
      <p class="caption">Physical vs. logical view of a matrix in row-major order.</p>
    </div>

    <p>This makes it clear why the inner, most important loop of our matrix multiplication is very cache unfriendly. Normally, the processor loads data from memory using fixed-size cache lines, commonly 64 Byte large. When iterating over the row of A, we incur a cache miss on the first entry. However, for matrix B, we walk down the rows, occurring a cache-miss at every step.</p>

    <div class="note-figure">
      <img src="{{ '/assets/images/notes/gpu-computing/column-miss.png' | relative_url }}" alt="Cache misses when iterating down a matrix column" loading="lazy">
      <p class="caption">Walking down a column triggers a cache miss per step.</p>
    </div>

    <p>To fix this, we reorder the two inner-most loops:</p>

    {% highlight cpp %}
template <int rows, int columns, int inners>
inline void matmulImplLoopOrder(const float *left, const float *right,
                                float *result) {
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < columns; col++) {
        result[row * columns + col] +=
          left[row * columns + inner] * right[inner * columns + col];
      }
    }
  }
}
    {% endhighlight %}

    <p>The improvement is quite spectacular, bringing runtime down to 89ms. A 16x improvement! Our inner loops now iterate through B &amp; C in a memory sequential manner. The only time we do a large jump in memory access is when our middle loop finishes and we need to fetch the first row of B again. Since we’re now only computing a partial result in the inner loop, we cannot perform the accumulation in a single register anymore.</p>

    <div class="note-figure">
      <img src="{{ '/assets/images/notes/gpu-computing/seq-column.png' | relative_url }}" alt="Sequential column access after loop reordering" loading="lazy">
      <p class="caption">Loop reordering makes column access sequential and cache friendly.</p>
    </div>

    <p>Looking at the compiled output on <a href="https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgBYBQogFUAztgAKAD24AGfgCsp5eiyahUAUgBMAIUtXyKxqiIEh1ZpgDC6egFc2TA3cAGQImbAA5PwAjbFIDAAd0JWIXJi9ffwSklKEQsMi2GLiee2xHZyERIhZSInS/AJKHbCdUqpqiPIjo2IMlatr6zKaBztDuwt6eAEp7dB9SVE4uADd0AkwAajYWIgBZH3oASTZ4%2Blp6dF2AKk3SAmAkInJNmkubzcYaZ9f3olvSNglIciNMzAB2GxaACCm1eZE2EFCRDu6AA7pszABmAAimy02KsqIx2I8mx4Wgs0kJxNstmmmMhZhhcLhglIiORm1CYQ52LxBKxRJ5sUxWLJFKpNJFpDp1gZEKhsNZKvZnOEmwwGSYYoFNK1DTFEsp1KFmu8DTlVgVTJZKvtcMBwPoRDMAFYrKR0ZtbpLpJjrObte68XTcczlQ77V9XR6vRjfSaA8KmLyQz67g8nu6rDKM37k0HLW6cYSIw6IaW7XDK%2BXGVXoZWuLN6Nw3fwAlwdOR0NwPFbNkp5otsAGsXxyERtM3ZgBrEBurSGbjSfhsBdLzvd3tcfhKEBLqdd5vkOCwFDYdQtHwkChUCA1YBKVTGMoiJDozsTjCnBi7VIvmE9Dvp%2B07kD%2B8QML0wA8BYJQQVBpAAPI3iBaJbvwl4tNCpBPtwmFXqgVSEJ2/CCMIYgSJwMhyMIyhqJox7kPoJRGCYaBWoYBBRPukCzOg8QVEw%2B67nMCxLH0RAkYBb4fuh3ATkQgLLBOaKkCw8QKSerZcO25AYT23C4IRN4IuoAAcABsAC0ln%2BsAqCoOSFgAHQ8Ii/bWLYLz4MQCKWOO0z8EeOjTLMSDYCwOBxBALYrmuG76WBO57gek7TmF5Dzouy5cFiHbJfh6XHrMZ7IGg6C/owd7UAh1VoGxMHSFoS50C6sT7hAURgVEoQ1AAnlp4GVRwwhIUw9CDUxOA7CYkjTQQgKtCsQJgVhqA3ip/DImUYH0Nx6mkP1Xg4GBSkEOuvAnm8LBPgAagQ2Bokh8TMEN5GiOIkg0R99EaGBLGGMYpicftPHwPxgmpCJ1lIVimzWTQNAsP01k7EQSAIzsixIPySAo2iZT0Jse5lC0QluEwngWsM5DBOMBRFFkyRCUMjTkIkLOpF0jNTKU5RtKMbN9GTrSVKMPM9MU9hCzT7P9B0kuTMUsxDuJ1GTspQ1osYRAvUQRxMIIWlxbpBVMTuFk2XZmqNeS0guVojseZxmy%2BbeY4lJsXhVaKAUzMFGXhZF0XUHOiU6auSUW0V%2B6HkHuUWOb25FSFM7kCtpDJK40hAA%3D%3D" target="_blank" rel="noopener">compiler explorer</a>, the loop reordering also enabled vectorization. With the naive loop order the compiler was already using the VFMADD instruction, but only on a single fp32 at a time. The relevant parts of the assembly look like this:</p>

    {% highlight asm %}
; In the loop setup, load a single fp32 from the current A row
; and broadcast it to all 8 entries of the ymm0 register
; vbroadcastss ymm0, dword ptr [rsi + 4*r8]

; In each instruction, load 8 entries from 
; the current row of B into a ymm register
vmovups ymm1, ymmword ptr [rbx + 4*rbp - 96]
vmovups ymm2, ymmword ptr [rbx + 4*rbp - 64]
vmovups ymm3, ymmword ptr [rbx + 4*rbp - 32]
vmovups ymm4, ymmword ptr [rbx + 4*rbp]
; In each instruction, multipy the current entry of A (ymm0) times 
; the entries of C (ymm1-4) and add partial results from C (memory load) 
vfmadd213ps ymm1, ymm0, ymmword ptr [rcx + 4*rbp - 96] ; ymm1 = (ymm0 * ymm1) + mem
vfmadd213ps ymm2, ymm0, ymmword ptr [rcx + 4*rbp - 64] ; ymm2 = (ymm0 * ymm2) + mem
vfmadd213ps ymm3, ymm0, ymmword ptr [rcx + 4*rbp - 32] ; ymm3 = (ymm0 * ymm3) + mem
vfmadd213ps ymm4, ymm0, ymmword ptr [rcx + 4*rbp] ; ymm4 = (ymm0 * ymm4) + mem
; Store the partial results back to C's memory
vmovups ymmword ptr [rcx + 4*rbp - 96], ymm1
vmovups ymmword ptr [rcx + 4*rbp - 64], ymm2
vmovups ymmword ptr [rcx + 4*rbp - 32], ymm3
vmovups ymmword ptr [rcx + 4*rbp], ymm4
    {% endhighlight %}
  </details>
</div>

#### The Tiling (Blocking) Strategy

To overcome the memory wall, we use **tiling**. We divide the matrices into small sub-matrices (tiles) that fit into the cache. We load a tile from $A$ and $B$, perform all possible multiplications between them, and then move to the next tile.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Locality of Reference)</span></p>

Tiling maximizes **Locality of Reference**:

  * **Temporal Locality:** Reusing data while it is in the cache.
  * **Spatial Locality:** Accessing data elements physically close to each other.

</div>

<!-- end list -->

```c++
// Blocked Matrix Multiplication
void MatrixMulOnHostBlocked(float* M, float* N, float* P, int Width, int blockSize) {
    for (int ii = 0; ii < Width; ii += blockSize) {
        for (int jj = 0; jj < Width; jj += blockSize) {
            for (int kk = 0; kk < Width; kk += blockSize) {
                // Process small blockSize x blockSize tiles here
                for (int i = ii; i < min(ii+blockSize, matWidth); ++i) {
                    for (int j = jj; j < min(jj+blockSize, matWidth); ++j){
                        float sum = 0;
                        for (int k = kk; k < Width; ++k) {
                            float a = M[i * width + k];
                            float b = N[k * width + j];
                            sum += a * b;
                        }
                        P[i * Width + j] += sum;
                  }
                }
            }
        }
    }
}
```

### Porting to the GPU

#### Naive CUDA Implementation

Our first GPU attempt assigns one thread to compute one element of the output matrix $P$.

```c++
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width) {
    // Calculate global row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0;

    // Each thread reads entire row of A and col of B from Global Memory
    for (int k = 0; k < Width; ++k) {
        Pvalue += Md[row * Width + k] * Nd[k * Width + col];
    }

    Pd[row * Width + col] = Pvalue;
}
```

#### The Memory Bandwidth Bottleneck

While this kernel runs in parallel, it suffers from **abysmal computational intensity**.

  * Every multiply-add (2 FLOPs) requires two float reads (8 Bytes).
  * Intensity = 0.25 FLOPs/Byte.
  * Required Bandwidth for 13 TFLOP/s (RTX 2080 Ti) $\approx$ **52 TB/s**.
  * Actual Hardware Bandwidth $\approx$ **616 GB/s**.

The hardware provides nearly 100x less bandwidth than this naive algorithm requires. The GPU spends almost all its time waiting for data from global memory.

## Optimizing with Shared Memory

To solve the bandwidth bottleneck, we must program the **Shared Memory**. This is a user-managed L1 cache (scratchpad) that is orders of magnitude faster than global memory.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/shared_memory_tiling.png' | relative_url }}" alt="Tiling with Shared Memory" loading="lazy">
  <figcaption>Collaborative Loading into Shared Memory</figcaption>
</figure>

#### The Algorithm

We adapt the tiling strategy for the GPU architecture:

1.  **Collaborative Load:** A thread block collectively loads a tile of $A$ and a tile of $B$ from Global Memory into Shared Memory.
2.  **Synchronize:** `__syncthreads()` ensures the tile is fully loaded.
3.  **Compute:** Threads perform dot products using the fast data in Shared Memory.
4.  **Repeat:** Move to the next tile.

This increases computational intensity by a factor of `TILE_WIDTH`. For a $16 \times 16$ tile, we reduce global memory traffic by 16x.

#### Shared Memory Kernel

```c++
#define TILE_WIDTH 16

__global__ void MM_SM(float* Md, float* Nd, float* Pd, int Width) {
    // static shared memory allocation
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    // Loop over the Md and Nd tiles required to compute the Pd element
    for (int m = 0; m < Width / TILE_WIDTH; ++m) {

        // --- Phase 1: Collaborative Loading ---
        // Each thread loads one element of Mds and Nds
        Mds[ty][tx] = Md[row * Width + (m * TILE_WIDTH + tx)];
        Nds[ty][tx] = Nd[(m * TILE_WIDTH + ty) * Width + col];

        // Ensure all threads have loaded data before computing
        __syncthreads();

        // --- Phase 2: Compute ---
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        // Ensure computation is done before overwriting shared mem in next iter
        __syncthreads();
    }

    Pd[row * Width + col] = Pvalue;
}
```

#### Bank Conflicts

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bank Conflicts)</span></p>

Shared memory is divided into **32 banks** (like parallel filing cabinets). Ideally, threads in a warp (32 threads) access different banks simultaneously.

  * **Conflict-Free:** Thread $i$ accesses Bank $i$.
  * **Bank Conflict:** Multiple threads in a warp access the *same* bank. The hardware serializes these requests, destroying performance.

</div>

A common cause is strided access. If threads access `array[threadIdx.x * 2]`, they only hit even banks, potentially causing 2-way conflicts. The implementation above generally avoids this by loading $16 \times 16$ tiles where `tx` maps directly to columns.

#### Advanced Optimizations

To close the gap to theoretical peak performance, further techniques are required:

  * **Data Dependencies:** We use `__syncthreads()` to handle Read-After-Write (RAW) and Write-After-Read (WAR) hazards.
  * **Dynamic Allocation:** Using `extern __shared__` allows sizing shared memory at runtime rather than compile time.
    * `extern` say: "This shared-memory array is declared here, but its size is not known at compile time; it will be provided at kernel launch."
  * **Thread Coarsening:** Having one thread compute multiple output elements (e.g., a $4 \times 4$ patch) increases register reuse.
  * **Double Buffering:** Loading the *next* tile into registers while computing the *current* tile hides memory latency completely.

#### Summary

  * **Performance is Memory-Bound:** High-performance computing is often less about math and more about data movement.
  * **Hierarchy is King:** Tiling (blocking) is the fundamental technique to exploit the memory hierarchy.
  * **Shared Memory:** This is the programmer's primary tool for maximizing computational intensity ($O(n)$ data reuse).
  * **Synchronization:** When using shared memory, `__syncthreads()` is essential for correctness to manage RAW and WAR hazards.

## Chapter 5 - Scheduling and Optimization

This chapter focuses on the crucial topic of scheduling and explores a series of powerful optimization techniques, using a common parallel algorithm—reduction—as our practical, step-by-step example.

### A Refresher on GPU Scheduling

Before we optimize, we must understand how the GPU executes our code. The CUDA programming model provides abstractions like thread blocks and grids, but the hardware has its own way of managing the work.

#### The Thread Hierarchy and Parallel Slackness

As a reminder, the CUDA programming model organizes threads into a three-level hierarchy:

1. A **thread** is the smallest unit of execution, running a single instance of your kernel function.
2. A **thread block**, also known as a Cooperative Thread Array (**CTA**), is a group of threads that can cooperate by sharing data through a fast, on-chip shared memory and can synchronize their execution.
3. A **grid** is composed of all the thread blocks launched by a single kernel call.

The GPU scheduler maps these CTAs to the physical processing units on the chip, called Streaming Multiprocessors (SMs). An important concept here is parallel slackness. To achieve high performance, we deliberately launch far more threads and CTAs than there are physical SMs. Why? The primary reason is to hide latency. When a group of threads is stuck waiting for a slow operation, like fetching data from global memory, the SM can instantly switch to executing another group of threads that is ready to run. This keeps the computational units busy and maximizes throughput.

A key rule of this model is that there are no dependency guarantees between different CTAs. They can be executed in any order, concurrently or sequentially, which gives the scheduler maximum flexibility.

#### The Warp: The True Unit of Scheduling

While we, as programmers, think in terms of threads and blocks, the GPU hardware schedules threads in groups of 32, known as a warp. A warp is a fundamental, architecture-dependent concept. All 32 threads in a warp execute the same instruction at the same time on an SM. This is a form of SIMT (Single Instruction, Multiple Thread) execution.

Let's clarify the abstractions:

* **CTAs (Thread Blocks)**: This is a user-defined abstraction. You decide how many threads are in a block and what they do. The execution of CTAs is opaque to the user; you don't control which SM they run on or when.
* **Warps**: This is a hardware-level abstraction defined by the Just-In-Time (JIT) compiler and the GPU architecture. The execution of warps is transparent to the user; you don't directly manage them, but their behavior has profound performance implications.

Imagine a CTA with 128 threads. The hardware doesn't see 128 individual threads; it sees four warps (warp 0: threads 0-31, warp 1: threads 32-63, etc.). The SM will pick a ready warp and execute one instruction for all 32 of its threads, then pick another ready warp, and so on.

The slide content describes a diagram that visually represents this hierarchy. It shows a Grid composed of multiple CTAs (or thread blocks). Each CTA has access to its own private shared memory. All CTAs in the grid can access the larger, but slower, global memory. Within a CTA, threads are grouped into Warps, and each individual thread has its own private thread-local memory (registers). This illustrates the memory and execution hierarchy of a modern GPU.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/chapter5_memory_threads_hierarchy.png' | relative_url }}" alt="GPU global memory" loading="lazy">
    <figcaption>Memory-Thread hierarchy</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/chapter5_wrap_scheduling.png' | relative_url }}" alt="GPU shared memory" loading="lazy">
    <figcaption>Wrap Scheduling</figcaption>
  </figure>
</div>

### Common CUDA Performance Issues

Before we can fix problems, we need to know what they are. Several common issues can bottleneck GPU performance:

* **Memory Coalescing**: Inefficient access patterns to global memory can dramatically slow down data transfers.
* **Latency Hiding**: If there isn't enough parallel slackness, the SMs will stall while waiting for memory, leaving the cores idle.
* **Divergent Branching**: When threads within the same warp take different paths in an if-else statement, performance suffers because the hardware must execute both paths sequentially.
* **Shared Memory Bank Conflicts**: Inefficient access patterns to shared memory by threads within a block can cause access serialization, slowing down computation.
* **Instruction Overhead**: Spending too many cycles on control flow (loops, address calculations) instead of actual computation can limit performance.

We will see nearly all of these issues in our upcoming example.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/chapter5_cuda_performance_issues_optimizations.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Memory-Thread hierarchy</figcaption>
</figure>

### The Parallel Reduction Problem: A Case Study in Optimization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parallel Reduction)</span></p>

A **reduction** is a common parallel operation where an array of elements is "reduced" to a single value using an associative operator like addition, multiplication, or finding the maximum value.

</div>

#### Examples of reduction include:

* Calculating a global sum: $s = \sum_{i=0}^{N} f(x_i)$
* Calculating a global product: $p = \prod_{i=0}^{N} f(x_i)$
* Building a histogram: $h_k = \sum_{i=0}^{N} (x_i = k) ? 1 : 0$

Reduction is a perfect candidate for optimization analysis because it is a **memory-bound operation**. This means its performance is limited by the speed at which it can read data from memory, not by the speed of the calculations. Therefore, our **key performance metric will be effective bandwidth GB/s** (gigabytes per second). The problem often occuring due to inefficient access patterns,causing the process to wait for data. Also it is an optimization example for scheduling issues that we are considering.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Effective Bandwidth)</span></p>

**Effective bandwidth** is a real-world measure of data transfer speed, accounting for inefficiencies like latency, protocol overhead, and packet loss, unlike **theoretical (asymptotic) bandwidth**. It quantifies the actual data throughput for specific applications, representing the sustainable rate under typical conditions, often lower than the maximum possible speed.

</div>

### The Multi-CTA Challenge: Global Synchronization

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/chapter5_partial_results_communication.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Partial results communication/synchronization</figcaption>
</figure>

To process a large array, we must launch many CTAs, each responsible for reducing a chunk of the input data. This produces a partial result for each CTA. But how do we combine these partial results into a final, single value?

This would be simple if we had a global synchronization mechanism that could make all CTAs across the entire GPU wait for each other. However, *CUDA provides no such feature*.

The lecture notes say that there is no mechanism for block synchronisation, but can we synchronise blocks within one SM?

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question:</span><span class="math-callout__name">The lecture notes say that there is no mechanism for block synchronisation, but can we synchronise blocks within one SM?</span></p>

Not in the way people usually mean by “block synchronization.”

#### What you *can’t* do

There is **no general, safe “block barrier”** even if the blocks happen to be on the **same SM**, because:

* CUDA provides **no primitive** like `__syncblocks()` (within-SM or otherwise).
* You also have **no guarantee** that two particular blocks are resident on the same SM at the same time (unless you use special features like clusters, and even then it’s not “free-form sync”).

#### What you *can* do (practically)

You can synchronize **within a block**:

* `__syncthreads()` (and the cooperative groups equivalents) is a true barrier **for threads in the same block**.

You can sometimes coordinate **between blocks on the same SM**, but it’s **not a true barrier** and is generally **unsafe unless you can prove residency**:

* **Atomic + polling in global memory**: one block waits until another sets a flag.
  This can still deadlock if the other block isn’t resident yet (same core issue as global barriers).
* **Only safe if all participating blocks are guaranteed resident simultaneously**, i.e.
  
  $$\#\text{CTAs} \le \#\text{SMs} \cdot b_r$$
  
  and even then you’re relying on assumptions about progress and scheduling.

#### The “real” supported ways to sync across blocks

* **End the kernel** and start a new kernel (kernel launch boundary is a global sync, assuming default stream / proper stream sync).
* **Cooperative Groups grid synchronization** (`grid_group::sync`) via *cooperative launch* (supported on certain GPUs and with constraints). This is the “official” in-kernel global barrier, but it requires that all blocks be able to be resident and meet other requirements.

So: **within one SM**, you can *sometimes hack coordination* if you ensure all relevant blocks are resident, but CUDA does **not** provide a general “sync blocks within one SM” mechanism like `__syncthreads()` provides within a block.

</div>

#### Why is there no global synchronization?

1. **Scalability**: A global barrier would be extremely expensive to implement in hardware across a device with a high SM count.
2. **Scheduling Guarantees**: GPU scheduling is non-preemptive. A CTA, once scheduled on an SM, runs to completion. If a CTA were to wait at a global barrier for a CTA that hasn't even been scheduled yet, it could lead to a deadlock, where the entire GPU grinds to a halt. This would also conflict with the principle of parallel slackness needed to hide memory latency. The number of CTAs that could be synchronized would be limited by the number of resident blocks per SM, according to the formula:  #\text{CTAs} $\leq$ #\text{SMs} $\cdot$ $b_r$  where $b_r$ is the number of resident blocks per SM.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example:</span><span class="math-callout__name">The deadlock example that would the global synchronization cause.</span></p>

Close, but the "cycle of CTAs waiting on each other" isn't the usual deadlock mechanism here.

The classic deadlock is simpler:

1. You launch **more blocks than can be resident at once**:
   
   $$\#\text{CTAs} > \#\text{SMs} \cdot b_r$$
   
2. The GPU schedules up to $\#\text{SMs} \cdot b_r$ blocks. These become **resident** and start running.
3. All resident blocks reach the **global barrier** and **wait** there.
4. While waiting, they **still occupy the SM resources** (registers, shared memory, block slots).
5. Because the SMs are "full" of waiting resident blocks, **no new blocks can become resident**, so the remaining (not-yet-scheduled) CTAs never start.
6. But the barrier can't release until **all CTAs arrive** → the ones that haven't started can't arrive → **deadlock**.

So it's not really a "waiting chain forming a cycle." It's more like:

* **scheduled/resident CTAs are waiting for unscheduled CTAs** to reach the barrier,
* but **unscheduled CTAs can't be scheduled** because scheduled CTAs are parked at the barrier holding all resources.

That's why the text says global synchronization would only be safe if all CTAs can be resident simultaneously:

$$\#\text{CTAs} \le \#\text{SMs} \cdot b_r$$

</div>

### The Solution: Kernel Decomposition

The standard solution is kernel decomposition. We write a kernel that performs a partial reduction, where each CTA writes its partial sum to global memory. After this first kernel completes, we launch it again on the smaller array of partial sums. **A kernel completion boundary acts as a de facto global synchronization point**. Because the reduction operation is the same at each level, we can reuse the same kernel code. **Negligible HW overhead, low SW overhead**

<div class="math-callout math-callout--remark">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">"Launch it again on the smaller array … reuse the same kernel code"</span></p>
  <p>Often true, but in practice people frequently use a different final-stage kernel (or do the last step on CPU) for efficiency. Still, "can reuse the same kernel" is fine as a conceptual statement.</p>
</div>

The figure depicts this as a tree-based reduction. A large array is at the bottom. The first kernel launch has many CTAs (CTA 0, CTA 1, CTA 2, etc.) that each compute a partial sum. These partial sums form a new, smaller array. A second kernel launch then reduces this smaller array, and so on, until a single final value remains.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/chapter5_kernel_decomposition.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Kernel decomposition</figcaption>
</figure>

**Question(s):** Where the intermediate results are stored? Are they returned from the kernel as an output? Could something happen to the global memory of the device between kernel launches?
<div class="accordion">
  <details markdown="1">
    <summary>Intermediate results</summary>

### The first answer

**Intermediate results are stored in GPU global memory (device memory).**
They are **not “returned” from the kernel** in the function-call sense.

#### How it actually works

A CUDA kernel:

* does **not return values** (its return type is `void`),
* instead **writes results into memory** that was passed to it as a pointer.

For kernel decomposition in a reduction:

1. **Host code allocates device memory**

   ```c
   float* d_input;
   float* d_partial;
   cudaMalloc(&d_input,   N * sizeof(float));
   cudaMalloc(&d_partial, numBlocks * sizeof(float));
   ```

2. **First kernel launch**

   * Each CTA reduces part of `d_input`
   * Each CTA writes **one partial sum** to `d_partial[blockIdx.x]`

   ```c
   reduce_kernel<<<numBlocks, blockSize>>>(d_input, d_partial, N);
   ```

3. **Kernel finishes → implicit synchronization point**

   * When the next kernel is launched **in the same stream**, CUDA guarantees the first kernel is complete.

4. **Second kernel launch**

   * Now `d_partial` becomes the input
   * Output goes to another (even smaller) buffer, or reused memory

   ```c
   reduce_kernel<<<numBlocks2, blockSize>>>(d_partial, d_partial2, numBlocks);
   ```

5. Repeat until one value remains.

6. **Final result**

   * Either copied back to the host with `cudaMemcpy`,
   * or used directly by another kernel.

#### Key points to remember

* **Kernels communicate only through memory**, not return values.
* **Global memory persists across kernel launches**, so it’s the natural place to store intermediate results.
* The **kernel launch boundary** (with proper stream ordering) is what provides the global synchronization.
* Registers and shared memory **do not persist** after a kernel finishes.

#### Mental model

Think of a kernel like:

> “Run this function *in parallel*, and **write your outputs into this array**.”

### The second answer

#### Could it happen that between kernel runs the global will be cleaned?

No — **device global memory is not automatically “cleaned” between kernel launches.**

If a kernel writes intermediate results into global memory, those values **stay there** until something overwrites them or you free that allocation.

#### What *can* change it

* Another kernel writes into the same memory region.
* You call something like `cudaMemset()` (or `cudaMemsetAsync`) on that allocation.
* You free the memory (`cudaFree`) and later reallocate (then its contents are not guaranteed).
* A device reset or process exit (then everything is gone).
* Rarely, errors/undefined behavior (out-of-bounds writes) can corrupt memory.

#### Two important “not guaranteed” cases

* **Freshly allocated memory is uninitialized**: after `cudaMalloc`, the contents are unspecified (could look like “old junk”). Don’t assume it’s zeroed.
* **Freed then reallocated memory**: also unspecified.

So for kernel decomposition reductions, it’s safe to assume intermediate results persist **as long as you keep the allocation alive and don’t overwrite it**.

  </details>
</div>

### A Step-by-Step Guide to Optimizing Reduction

We will now analyze six different versions of a reduction kernel, each fixing a performance problem from the previous one. Our test case will be an array of 4 million elements, and we will analyze performance by varying the number of threads per CTA.

#### Reduction #1: Interleaved Addressing with Divergence

This is our naive, starting-point implementation. The basic idea is that in each step of a loop, half of the active threads will fetch a value from a neighboring thread, add it to their own, and store the result. This process repeats until only the first thread holds the final sum. `blockDim.x` must be a power-of-two.

```c++
__global__ void Reduction0a_kernel( int *out, int *in, size_t N ) {
    extern __shared__ int sPartials[];
    const int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    // Each thread loads one element from global to shared mem
    sPartials[tid] = in[i];
    __syncthreads();
    
    // Do reduction in shared mem
    for ( unsigned int s = 1; s < blockDim.x; s *= 2 ) {
        if ( tid % ( 2 * s ) == 0 ) {
            sPartials[tid] += sPartials[tid + s];
        }
        __syncthreads();
    }
    
    if ( tid == 0 ) {
        out[blockIdx.x] = sPartials[0];
    }
}
```

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/chapter5_interleaved_addressing.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Interleaved Addressing</figcaption>
</figure>

#### Code Explanation:

* `extern __shared__ int sPartials[];`: Declares a dynamically sized shared memory array. Its size is specified at kernel launch.
* `sPartials[tid] = in[i];`: Each thread loads one element from the global input array in into the shared memory array `sPartials`.
* `__syncthreads();`: This is a crucial barrier. It ensures that all threads in the block have completed their load from global memory before any thread proceeds to the reduction phase.
* `for (unsigned int s = 1; s < blockDim.x; s *= 2)`: This loop performs the reduction. The variable `s` represents the stride between the two elements being added. It doubles in each iteration (1, 2, 4, 8...).
* `if (tid % ( 2 * s ) == 0)`: This condition selects which threads are active. In the first iteration (`s=1`), threads 0, 2, 4, ... are active. In the second (`s=2`), threads 0, 4, 8, ... are active, and so on.
* `__syncthreads();`: Another barrier inside the loop is essential. It prevents a race condition where one thread might read a value from `sPartials` in the next iteration before another thread has finished writing its new sum to that same location in the current iteration.
* `if (tid == 0)`: After the loop, thread 0 of the block holds the partial sum for the entire block, which it writes out to the global output array out.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multiple Blocks and Shared Memory)</span></p>

The code could be used with multiple blocks, not just one, because **`sPartials` is in shared memory**, and **shared memory is per-block**.

So even though every block has threads with the same `tid` values (0, 1, 2, …), each block has its **own private shared-memory array**:

* Block 0: `sPartials_block0[tid]`
* Block 1: `sPartials_block1[tid]`
* Block 2: `sPartials_block2[tid]`
* …

They do **not** alias each other. What *is* shared across blocks is **global memory**, e.g. `in[i]` and `out[blockIdx.x]`.

So:
* `sPartials[tid] = in[i];` is safe across blocks (different shared memory instances).
* `out[blockIdx.x] = ...;` is also safe because each block writes a different `blockIdx.x`.

</div>

#### Problem Identified: Branch Divergence

The `if (tid % (2*s) == 0)` check is the source of a major performance problem. Within a warp of 32 threads, some threads will satisfy this condition while others will not. For example, when `s=1`, half the threads in a warp will pass the check, and half will fail. Since all threads in a warp execute the same instruction, the hardware must execute the if block for the active threads and then wait while the other threads do nothing. This effectively halves the utilization of the SM.

#### Reduction #2: Non-Divergent Interleaved Addressing

We can eliminate branch divergence by re-arranging the work so that all threads in a warp either all participate or all do not. We can achieve this by making the active threads contiguous.

```c++
// do reduction in shared mem
for ( unsigned int s = 1; s < blockDim.x; s *= 2 ) {
    int index = 2 * s * tid;
    if ( index < blockDim.x ) {
        sPartials[index] += sPartials[index + s];
    }
    __syncthreads();
}
```

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/chapter5_interleaved_addressing_nondivergent.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Interleaved nondivergent addressing</figcaption>
</figure>

#### Code Explanation:

* `int index = 2 * s * tid;`: Instead of checking `tid % (2 * s)`, we directly calculate the index each thread will work on.
* `if (index < blockDim.x)`: Now, threads 0, 1, 2, ... are the active threads. Since `tid` is consecutive, the first few warps will have all their threads pass the condition, while the later warps will have all their threads fail. This avoids intra-warp divergence.

#### Problem Identified: Shared Memory Bank Conflicts

We fixed one problem but created another. Shared memory is organized into physical memory banks. To achieve high bandwidth, consecutive threads should access consecutive banks. In this new code, the access pattern is `sPartials[index]` and `sPartials[index + s]`. The stride `s` causes threads in the same warp to access memory locations that fall into the same bank, leading to a bank conflict. The hardware must serialize these requests, reducing the effective shared memory bandwidth. The best access pattern for shared memory is typically for each thread `tid` to access `sPartials[tid]`.

#### Reduction #3: Non-Divergent Sequential Addressing

To fix the bank conflicts, we change the indexing logic again. This version ensures that active threads are contiguous and that their memory accesses are sequential, avoiding bank conflicts.

```c++
// do reduction in shared mem
for ( unsigned int s = blockDim.x / 2; s > 0; s >>= 1 ) {
    if ( tid < s ) {
        sPartials[tid] += sPartials[tid + s];
    }
    __syncthreads();
}
```

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/chapter5_sequential_addressing_nondivergent.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Sequential nondivergent addressing</figcaption>
</figure>

#### Code Explanation:

* `for (unsigned int s = blockDim.x / 2; ... )`: The loop now counts down. The variable `s` represents the number of active threads in each iteration, which is also the offset for the addition. It starts with half the threads in the block being active.
* `if (tid < s)`: This is a simple, non-divergent check. Threads `0` to `s-1` are active.
* `sPartials[tid] += sPartials[tid + s];`: This is a "block access" pattern. Thread `tid` reads from `tid` and `tid + s`. Since `s` is large initially, this doesn't cause bank conflicts, and as `s` gets smaller, the accesses remain efficient.

#### Problem Identified: Idle Threads

This version is much better, but it's still not perfect. In the very first iteration of the loop, half of the threads in the block (`tid >= s`) are completely idle. In the next iteration, three-quarters are idle, and so on. While we have solved divergence and bank conflicts, we are now underutilizing the computational resources of the SM.

#### Reduction #4: First Add During Load

We can solve the idle thread problem by giving every thread more work to do at the beginning. Instead of each thread loading just one element from global memory, we can have each thread load two elements and perform the first addition right away.

To do this, we launch the kernel with half the number of blocks but twice the grid stride for each thread.

```c++
// Previous version's load:
// unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
// sPartials[tid] = in[i];

// New version:
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

// Perform first level of reduction
// read from global memory, write to local memory
sPartials[tid] = in[i] + in[i+blockDim.x];
__syncthreads();
    
for ( unsigned int s = blockDim.x / 2; s > 0; s >>= 1 ) {
    if ( tid < s ) {
        sPartials[tid] += sPartials[tid + s];
    }
    __syncthreads();
}
```

#### Code Explanation:

* `unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;`: Each block now covers a region twice the size of `blockDim.x`.
* `sPartials[tid] = in[i] + in[i+blockDim.x];`: Each thread now performs two loads (`in[i]` and `in[i+blockDim.x]`) and one add, storing the result directly into shared memory. This uses all threads in the block productively from the very beginning.
* We basically just skip the first phase of summation (reducing the problem by two) via the first add, but we still will utilise only the half the threads in the second phrase.

#### Problem Identified: Instruction Overhead

We are getting much closer to peak performance, but there is still room for improvement. The for loop itself, with its address arithmetic and control flow, introduces instruction overhead. These are instructions that don't perform the core computation (the additions) but are necessary to manage the loop. We can reduce this overhead by unrolling the loop.

#### Reduction #5: Unrolling the Last Warp

Number of active threads decreases over time. A warp (32 threads) has a special property: all instructions within it are synchronous. The scheduler broadcasts a single instruction to all 32 threads. This means that if we are in a situation where the number of active threads is 32 or fewer (i.e., only one warp is left doing work), we no longer need the `__syncthreads()` call. The natural lock-step execution of the warp guarantees synchronization. We also no longer need the `if (tid < s)` check, as the inactive threads in the warp can simply be told to nullify their output.

We can exploit this by "unrolling" the last few iterations of the loop—specifically, the iterations where `s <= 32`.

```c++
__global__ void Reduction0e_kernel( int *out, int *in, bool echo ) {
    extern __shared__ int sPartials[];
    const int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    
    // Perform first level of reduction
    sPartials[tid] = in[i] + in[i+blockDim.x];
    __syncthreads();
    
    // Loop for s > 32
    for ( unsigned int s = blockDim.x / 2; s > 32; s >>= 1 ) {
        if ( tid < s ) {
            sPartials[tid] += sPartials[tid + s];
        }
        __syncthreads();
    }
    
    // Unrolled additions for the last warp (s <= 32)
    if ( tid < 32 && blockDim.x >= 64) sPartials[tid] += sPartials[tid + 32];
    if ( tid < 16 && blockDim.x >= 32) sPartials[tid] += sPartials[tid + 16];
    if ( tid <  8 && blockDim.x >= 16) sPartials[tid] += sPartials[tid + 8];
    if ( tid <  4 && blockDim.x >=  8) sPartials[tid] += sPartials[tid + 4];
    if ( tid <  2 && blockDim.x >=  4) sPartials[tid] += sPartials[tid + 2];
    if ( tid <  1 && blockDim.x >=  2) sPartials[tid] += sPartials[tid + 1];
    
    if ( tid == 0 ) {
        out[blockIdx.x] = sPartials[0];
    }
}
```

#### Code Explanation:

* The for loop now only runs for the initial reduction steps where more than one warp is active (`s > 32`).
* The subsequent if statements handle the final six reduction steps manually. These lines have no `__syncthreads()` calls, reducing instruction overhead and removing synchronization latency for the final warp. The extra `blockDim.x >= N` checks ensure this code works correctly for block sizes smaller than 64.

#### Reduction #6: Complete Unrolling with Templates

Why stop at unrolling the last warp? We could unroll the entire loop. The problem is that the number of loop iterations depends on `blockDim.x`, which is a runtime parameter. To unroll completely, the compiler needs to know the loop bounds at compile time.

We can solve this using C++ templates. A template allows us to write a generic function where a parameter (like the block size) can be specified at compile time. The compiler will then generate a specialized, highly optimized version of that function for the specific block size we provide.

```c++
template <unsigned int blockSize>
__global__ void Reduction0f_kernel( int *out, int *in, bool echo ) {
    extern __shared__ int sPartials[];
    const int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    sPartials[tid] = in[i] + in[i+blockSize];
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sPartials[tid] += sPartials[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sPartials[tid] += sPartials[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sPartials[tid] += sPartials[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sPartials[tid] += sPartials[tid + 64]; } __syncthreads(); }

    // No __syncthreads() needed for the last warp
    if ( tid < 32 && blockSize >= 64) sPartials[tid] += sPartials[tid + 32];
    if ( tid < 16 && blockSize >= 32) sPartials[tid] += sPartials[tid + 16];
    // ... and so on for the rest of the unrolled steps ...
    if ( tid < 1 && blockSize >= 2) sPartials[tid] += sPartials[tid + 1];

    if ( tid == 0 ) {
       out[blockIdx.x] = sPartials[0];
    }
}
```

Since the kernel now requires the block size at compile time, we need a "wrapper" function on the host to call the correct version based on the runtime `dimBlock` parameter. A switch statement is perfect for this.

```c++
void Reduction0f_wrapper ( int dimGrid, int dimBlock, int smemSize, int *out, int *in, bool echo ) {
    switch ( dimBlock ) {
        case 1024: Reduction0f_kernel<1024><<< dimGrid, dimBlock, smemSize >>>(out, in, echo); break;
        case 512:  Reduction0f_kernel<512><<< dimGrid, dimBlock, smemSize >>>(out, in, echo); break;
        case 256:  Reduction0f_kernel<256><<< dimGrid, dimBlock, smemSize >>>(out, in, echo); break;
        // ... cases for other power-of-two block sizes ...
        case 1:    Reduction0f_kernel<1><<< dimGrid, dimBlock, smemSize >>>(out, in, echo); break;
    }
}
```

This version provides the compiler with maximum information, allowing it to generate the most optimized code possible by completely removing the loop structure. Interestingly, the performance data shows this version was slightly slower than the partial unrolling, which might be due to factors like increased code size impacting the instruction cache. This is a great lesson: optimization is experimental, and you must measure the results.

### Performance Summary

The following table summarizes the throughput achieved by each optimization. The `maxThr` column shows the thread count per block that yielded the peak performance for that version.

| Version | 32 | 64 | 128 | 256 | 512 | 1024 | maxThr | maxBW (GB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Interleaved, Divergent (intrlvd div) | 7.39 | 12.57 | 16.77 | 14.67 | 12.33 | 9.05 | 128 | 16.77 |
| 2. Interleaved, Non-Divergent (intrlvd non-div) | 10.46 | 18.33 | 23.88 | 18.96 | 14.5 | 10.02 | 128 | 23.88 |
| 3. Sequential, Non-Divergent (seq. non-div) | 11.05 | 19.54 | 30.83 | 27.51 | 23.67 | 17.99 | 128 | 30.83 |
| 4. First Add During Load (first add) | 21.68 | 37.15 | 58.03 | 51.31 | 43.75 | 33.66 | 128 | 58.03 |
| 5. Unrolling Last Warp (unrolling) | 22.59 | 36.91 | 68.38 | 62.35 | 53.06 | 43.78 | 128 | 68.38 |
| 6. Complete Unrolling (templated) | 26.47 | 41.19 | 42.98 | 40.01 | 34.1 | 29.78 | 128 | 42.98 |

This progression clearly shows how systematically identifying and eliminating bottlenecks—from branch divergence to instruction overhead—can lead to massive performance gains (from 16.77 GB/s to a peak of 68.38 GB/s).

### A Framework for Optimization

The techniques we used in the reduction example can be categorized into a broader framework. When you optimize CUDA code, you can make changes at three different levels.

1. Algorithmic Optimizations

This involves changing the high-level parallel algorithm itself—the global pattern of work and synchronization.

* Hierarchical Tree: Our reduction algorithm is a classic example.
* Associativity: We exploited the associative property of addition $a + (b + c) = (a + b) + c$ to reorder the operations for parallelism.
* Algorithm Cascading: A further optimization (not shown) would be to have each thread sum multiple elements sequentially before starting the parallel reduction. This increases Instruction-Level Parallelism (ILP) and reduces the total number of synchronization steps needed.

2. Code Optimizations

Here, the overall algorithm stays the same, but the implementation of the kernel code is changed for better hardware performance.

* Addressing Changes: We modified our indexing to improve memory coalescing and eliminate shared memory bank conflicts.
* Loop Unrolling: We unrolled the loop for the last warp (and later, the entire loop) to reduce instruction overhead.
* Templating: We used C++ templates to generate specialized, highly-optimized code based on compile-time parameters.
* Warp Shuffle Operations: A more advanced technique (not shown) that allows threads within a warp to exchange data directly without using shared memory, further reducing latency.

3. Scheduling Optimizations

This level of optimization leaves the algorithm and code untouched. Instead, it focuses on how kernels are launched and how work is managed.

* Kernel Launch Parameters: Tuning the grid and block dimensions (`dimGrid`, `dimBlock`) can have a huge impact on performance.
* Overlapped Copy & Execute: Using CUDA streams to overlap data transfers between the host and device with kernel execution to keep all parts of the GPU busy.

### Summary

This chapter provided a deep dive into the world of GPU performance optimization. By understanding the hardware's scheduling behavior and systematically addressing common bottlenecks, you can transform a simple parallel algorithm into a highly efficient one.

Key Takeaways:

1. Set the Right Goal: First, identify whether your application is memory-bound (measure in GB/s) or compute-bound (measure in GFLOP/s). This determines where you should focus your optimization efforts.
2. Identify Bottlenecks: Systematically look for issues like memory access patterns, branch divergence, instruction overhead, and resource underutilization.
3. Optimize Systematically: Start with the broadest algorithmic optimizations, then move to fine-grained code optimizations, and finally tune the scheduling optimizations.
4. Know When to Stop: The goal of optimization is not just raw performance, but performance balanced with code readability, maintainability, and portability. A hyper-optimized but unreadable kernel may be a long-term liability.

## Chapter 6 - Profiling and Understanding GPU Performance

Welcome to the world of performance analysis! Writing a parallel program that runs correctly is only the first step. The next, and often more challenging, step is to make it run fast. In high-performance computing, we are constantly chasing the maximum possible speedup. This chapter will introduce you to the fundamental concepts that define and limit the performance of your GPU applications. As one expert noted, “There is no lower bound how bad a baseline can be,” which serves as a humble reminder that there is always room for improvement.

### What is Performance? The Concept of Intensity

At its core, a program's performance is a balancing act between two fundamental activities: computation and memory access. Your GPU can perform trillions of calculations per second, but it can only do so if it has the data it needs. Fetching that data from memory takes time. This relationship is captured by a crucial metric called Arithmetic Intensity.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Arithmetic Intensity)</span></p>

**Arithmetic Intensity** is defined as the ratio of floating-point operations (FLOPs) performed for every byte of data moved from memory:

$$r = \frac{f}{b}$$

where $r$ is the arithmetic intensity in FLOPs/Byte, $f$ is the number of floating-point operations, and $b$ is the number of bytes of memory accessed.

</div>

Think of it like a chef in a kitchen. If the chef spends a lot of time chopping, mixing, and cooking (computation) for every ingredient they grab from the pantry (memory access), their arithmetic intensity is high. They are making efficient use of their time at the cooking station. If they constantly run back and forth to the pantry for a single ingredient each time, their intensity is low, and the pantry access becomes the bottleneck.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Types of Intensity)</span></p>

It's important to distinguish between three related concepts:

* **Algorithmic Intensity:** The inherent ratio of operations to memory accesses in the pure, mathematical algorithm.
* **Computational Intensity:** The actual ratio achieved by your specific code implementation. Caching can reduce memory accesses and increase computational intensity.
* **Machine Intensity:** The ratio of a hardware's peak FLOPs/sec to its peak memory bandwidth (Bytes/sec). This represents the intensity an application needs to achieve to fully utilize the processor's computational power.

</div>

To achieve peak performance on a GPU, an extreme amount of computational intensity and data reuse is required.

### Memory-Bound vs. Compute-Bound: Where is the Bottleneck?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Boundedness Classification)</span></p>

Based on their arithmetic intensity, algorithms can be classified into three categories:

* **Memory-Bound:** Limited by the speed of memory access. A simple vector addition (`C[i] = A[i] + B[i]`) is a classic example: for every three memory operations (two reads, one write), it performs only one addition.
* **Compute-Bound:** Limited by the raw computational power of the processor. A dense matrix multiplication is a good example, as it reuses data many times.
* **IO-Bound:** Limited by Input/Output operations. In GPU computing, this often describes the PCIe bottleneck, where performance is limited by host-to-device data transfer time.

Understanding which category your application falls into is the first step toward optimizing it.

</div>

### The Roofline Model: A Visual Guide to Performance

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Roofline Model)</span></p>

The **Roofline Model** is a powerful tool for visualizing performance limitations. The model is a 2D plot with Arithmetic Intensity $r$ (FLOPs/Byte) on the x-axis and Attainable Performance $p$ (GFLOP/s) on the y-axis.

The "roofline" consists of two parts:

1. **The Slanted Roof:** Represents peak memory bandwidth ($m_p$, in GB/s). Performance is proportional to intensity: $p = m_p \cdot r$.
2. **The Flat Roof:** Represents peak compute performance ($f_p$, in GFLOP/s). Once intensity is high enough, the GPU's computational units are fully saturated.

The attainable performance is:

$$p = \min(m_p \cdot r, \; f_p)$$

If you are under the slanted part, you are **memory-bound**. If under the flat part, you are **compute-bound**.

</div>

### How to Optimize Performance

The Roofline Model clarifies which type of optimization will be most effective.

If your application is Compute-Bound (hitting the flat part of the roof), you should focus on optimizing floating-point performance:

* Balance the number of additions and multiplications.
* Improve Instruction-Level Parallelism (ILP) to help the processor's superscalar architecture execute more instructions simultaneously.
* Make effective use of SIMD (Single Instruction, Multiple Data) instructions, which perform the same operation on multiple data points at once.

If your application is Memory-Bound (hitting the slanted part of the roof), you must optimize memory usage:

* Use software prefetching to fetch data from memory before it's actually needed.
* Structure your code to avoid load stalls, where the processor idles waiting for data.
* Ensure memory affinity by having threads access data that is physically close to them (e.g., in NUMA architectures).
* Avoid non-local data accesses whenever possible.

Critically, the arithmetic intensity $r$ of an application is not always fixed; it can vary and often scales with the problem size. Furthermore, effective caching is a primary way to optimize. By keeping frequently used data in fast, on-chip caches, you reduce the number of accesses to slower main memory. This reduction in memory traffic directly increases your application's effective arithmetic intensity, pushing it to the right on the Roofline plot and unlocking higher performance.

## Chapter 2: Introduction to GPU Profiling

Once you understand the theoretical limits of performance, the next step is to measure what your application is actually doing. This is the job of a profiler.

### What is Profiling?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Profiling)</span></p>

**Profiling** is the process of analyzing an application's behavior to understand its performance characteristics. It involves collecting data about both its static and dynamic properties.

* **Static Behavior:** Properties of the code itself, independent of any specific run (e.g., instruction count, instruction types).
* **Dynamic Behavior:** What happens when the code is executed (e.g., cache hit/miss rates, scheduler decisions, thread occupancy, memory stalls).

</div>

To gather this data, profilers rely on hardware performance counters. These are special registers built into the processor that can count events like cache misses, instructions executed, or cycles the processor was stalled. However, these counters are an expensive and limited resource. Accessing them can be costly and will inevitably affect the performance of the code you are trying to measure. This is known as profiling overhead.

### Levels of Profiling: From C++ to Machine Code

When we write a CUDA C++ program, it goes through several stages of compilation before it can run on the GPU. We can analyze performance at any of these levels:

1. C/C++: The high-level source code we write.
2. IR (Intermediate Representation): A lower-level, platform-agnostic representation of the code, such as LLVM IR.
3. PTX (Parallel Thread Execution): An assembly-like language for NVIDIA GPUs. It's a stable instruction set that can be compiled for different GPU architectures.
4. SASS (Shader Assembly): The native, machine-level assembly language for a specific GPU architecture. This is what the hardware actually executes.

Profiling at the SASS level gives you the most accurate and detailed view of what the hardware is doing, as it's closest to the metal.

### Prerequisites for Profiling

Before you start optimizing for performance, you must follow two critical steps:

1. Ensure Correctness First: Performance profiling is meaningless if your program produces the wrong results. Use tools like cuda-memcheck to find and fix memory errors, such as segmentation faults and memory leaks, before you begin any performance analysis.
2. Compile with Correct Flags: To get the most accurate and useful profiling data, you need to compile your code correctly.
  * Enable compiler optimizations (e.g., nvcc -O2). This ensures you are profiling the code as it would run in a production environment.
  * Include debug information (e.g., nvcc -lineinfo). This allows the profiler to map the low-level SASS instructions back to the original lines in your C++/CUDA source code, making it much easier to identify which parts of your code are causing bottlenecks.

## Chapter 3: NVIDIA's Professional Profiling Toolkit

NVIDIA provides a powerful suite of tools called Nsight for profiling and debugging GPU applications. For performance analysis, we will focus on two key components: Nsight Compute and Nsight Systems.

### Nsight Compute: Deep-Diving into Kernels

Nsight Compute is the primary tool for detailed analysis of individual CUDA kernels. It can collect an immense amount of data—nearly 1,700 different metrics on a modern GPU like the TU102—giving you an unprecedented view into your kernel's execution.

Nsight Compute offers two interfaces:

* A command-line interface (CLI) called ncu.
* A graphical user interface (GUI) called nv-nsight-cu.

A common workflow is to use ncu on a remote server (where the powerful GPU is) to collect performance data and save it to a report file. You can then download this file and open it with the nv-nsight-cu GUI on your local machine for in-depth visual analysis.

By default, ncu prints its results to the console (stdout). To save the results, you use the --export or -o flag.

```bash
# To run a profiler on an application and save the report
ncu --export my_report.ncu-rep ./my_application
```

### Working with Metrics

Querying all 1,700+ metrics at once is overwhelming and inefficient. Instead, ncu provides predefined sets of metrics for common analysis tasks. You can list these sets with the command:

```bash
ncu --list-sets
```

This will display a table of available sets, such as default, detailed, and full, showing how many metrics each set collects.

| Identifier | Sections | Enabled | Estimated Metrics |
| --- | --- | --- | --- |
| default | LaunchStats, Occupancy, SpeedOfLight | yes | 36 |
| detailed | ComputeWorkloadAnalysis, InstructionStats, LaunchStats, MemoryWorkloadAnalysis, Occupancy, SchedulerStats, SourceCounters, SpeedOfLight, SpeedOfLight_RooflineChart, WarpStateStats | no | 172 |
| full | ComputeWorkloadAnalysis, InstructionStats, LaunchStats, MemoryWorkloadAnalysis, MemoryWorkloadAnalysis_Chart, MemoryWorkloadAnalysis_Tables, Nvlink_Tables, Nvlink_Topology, Occupancy, SchedulerStats, SourceCounters, SpeedOfLight, SpeedOfLight_RooflineChart, WarpStateStats | no | 177 |
| source | SourceCounters | no | 58 |

You can also create custom profiling runs by combining sets, sections, and individual metrics.

```bash
# Run with the 'default' set, but also collect the 'SourceCounters' section
# and one specific metric related to shared memory instructions.
ncu --set default --section SourceCounters --metrics sm__sass_inst_executed_op_shared ./my_application
```

### Nsight Systems: Analyzing the Entire Application

While Nsight Compute is for kernels, Nsight Systems is designed to analyze the performance of the entire system, focusing particularly on CPU-GPU interactions. It helps you identify high-level bottlenecks, such as:

* Time spent transferring data over the PCIe bus.
* Gaps in GPU execution where the GPU is idle waiting for the CPU.
* How different CUDA API calls and kernel launches overlap (or fail to overlap) over time.

Like Nsight Compute, it has a CLI (nsys) and a GUI (nsight-sys).

A powerful feature of Nsight Systems is the ability to add custom annotations to your host code using the NVTX (NVIDIA Tools Extension) library. This lets you mark specific regions of your C++ code, which will then appear as labeled ranges in the profiler's timeline, making it easy to correlate profiler output with your application's logic.

To use NTX, you need to include the header and link against the library:

```c
#include <nvToolsExt.h>

// Link your application with -lnvToolsExt
```

You can then bracket sections of your code with nvtxRangePush and nvtxRangePop:

```c++
// This code block will appear as a labeled "sleeping" range in the nsys GUI.
nvtxRangePush("sleeping");
sleep(100);
nvtxRangePop();
```


## Chapter 4: Case Study: Profiling a Matrix Multiplication Kernel

Let's apply these concepts to a real-world example: profiling a highly optimized matrix multiplication routine from the cuBLAS library. We'll examine how the performance changes dramatically not just with the size of the matrices, but with their shape.

### The High Cost of Detailed Profiling

First, it is crucial to understand that profiling has an overhead. The more metrics you collect, the more the profiler interferes with the application's execution, slowing it down. This happens because the GPU has a limited number of hardware performance counters. To collect many metrics, the profiler must re-run the kernel multiple times (called "passes" or "replays"), collecting a different subset of metrics each time.

Consider this example of running a test on a 1024x1024 matrix multiplication (SGEMM):

1. Baseline (No Profiling): The application runs extremely fast, achieving over 8,000 GFLOP/s.

```bash
$ ./cuBLAS-test-sm75 1024 1024 1024
SGEMM (  1024 x   1024 x   1024):     0.0002 sec,    8363.55 GFLOP/s
```

2. Profiling with the default set: Using ncu with the default set requires 8 passes. The execution time balloons from 0.2 milliseconds to over half a second, and the measured performance plummets to just 3.46 GFLOP/s.

```bash
$ ncu -f --set default -o <file> ./cuBLAS-test-sm75 1024 1024 1024
==PROF== Profiling "volta_sgemm_128x64_nn" - 2: 0%....50%....100% - 8 passes
SGEMM (  1024 x   1024 x   1024):     0.5779 sec,       3.46 GFLOP/s
```

3. Profiling with the full set: Using the full set is even more expensive, requiring 33 passes. The execution takes 1.7 seconds, and measured performance is a paltry 1.17 GFLOP/s.

```bash
$ ncu -f --set full --section ComputeWorkloadAnalysis -o <file> ./cuBLAS-test-sm75 1024 1024 1024
==PROF== Profiling "volta_sgemm_128x64_nn" - 2: 0%....50%....100% - 33 passes
SGEMM (  1024 x   1024 x   1024):     1.7117 sec,       1.17 GFLOP/s
```

This is a critical lesson: the performance numbers you see during a detailed profiling run are not the true performance of your application; they are the performance under heavy observation. You must always establish a non-profiled baseline first.

### The Challenge of Skewed Matrices

The peak performance of libraries like cuBLAS is often benchmarked using square matrices (e.g., $1024 \times 1024$). But what happens if the matrices are "skewed"—for instance, very tall and thin, or very short and wide?

Let's consider the matrix multiplication $C = A \cdot B$, where the dimensions are $m \times k$ for matrix $A$ and $k \times n$ for matrix $B$. We will keep the total amount of work roughly the same but dramatically alter the shapes of $A$ and $B$.

The lecture slide presents a bar chart that illustrates this scenario.

* Description of the Diagram: The chart plots GFLOP/s (y-axis) against various matrix dimensions m-n-k (x-axis). The first bar, representing a square matrix 1024-1024-1024, shows a very high performance of nearly 8400 GFLOP/s. As the matrices become more skewed (e.g., 2048-512-1024, 4096-256-1024, and so on, up to an extreme 1048576-1-1024), the performance drops dramatically, eventually falling below 500 GFLOP/s. This shows a substantial performance loss even though the total number of floating-point operations remains identical.

This massive performance degradation suggests that the skewed shapes are causing a major problem with memory access patterns.

### Using Nsight Compute to Uncover the Truth

To diagnose this, we can use ncu to profile the application for each skewed matrix configuration. A simple shell for loop can automate this process, saving a unique report file for each run.

```bash
# This loop iterates, making the 'm' dimension larger and 'n' smaller each time,
# while keeping 'k' and the total work constant.
for ((i=1;i<=1024;i*=2)); do
  ncu -f --set full -o cuBLAS-skewed-$((1024*$i))-$((1024/$i))-$((1024)) \
  ./cuBLAS-test-sm75 $((1024*$i)) $((1024/$i)) $((1024))
done
```

After collecting the data, we can import the report files (.ncu-rep) and examine specific metrics related to memory performance. Some key metrics of interest include:

* L1 Cache Hit Rate: l1tex__t_sector_hit_rate.pct
* L2 Cache Hit Rate: lts__t_sector_hit_rate.pct
* Shared Memory Accesses: sass__inst_executed_shared_loads, sass__inst_executed_shared_stores
* Global Memory Traffic (L1 to L2): l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
* Global Memory Traffic (L2 to DRAM): dram__sectors_read.sum

### Analyzing Memory Traffic and Cache Performance

By plotting these metrics against the different matrix shapes, the source of the problem becomes clear.

**Global Read Traffic:**

* Description of the Diagram: A line graph shows two metrics: "L1->L2 LD [MB]" and "DRAM RD [MB]". For square-like matrices on the left, both traffic volumes are low. As the matrices become highly skewed to the right, the traffic from the L1 cache to the L2 cache (the blue line) explodes, increasing from a small amount to nearly 3000 MB. In contrast, the traffic from the L2 cache to main DRAM (the red line) stays relatively flat and low. This indicates that data is being constantly evicted from the L1 cache and must be re-fetched from L2, but the L2 cache is large enough to absorb most of this traffic, preventing a complete collapse from DRAM access. The read traffic amplification factor ranges from 1.25x for a moderately skewed case to an enormous 256x for the most skewed case.

**Cache Hit Rates:**

* Description of the Diagram: A line graph plots the L1 and L2 hit rates. For square matrices, both hit rates are very high (L1 at ~95%, L2 at ~85%). As the matrix shape skews, the L1 hit rate plummets dramatically, falling close to 0% for the most extreme cases. The L2 hit rate also declines but remains much higher, confirming that L2 is catching most of the L1 misses.

**Internal Kernel Switching:** An interesting detail revealed by the profiling data is that the cuBLAS library is not using the same kernel for all matrix shapes. It intelligently selects different internal implementations based on the problem size and shape. For example:

* 1024-1024-1024 uses volta_sgemm_128x64_nn
* 32768-32-1024 uses volta_sgemm_128x32_sliced1x4_nn
* 524288-2-1024 uses gemmSN_NN_kernel
* 1048576-1-1024 uses a combination of kernel and splitKreduce_kernel

This shows the complexity of high-performance libraries, which contain multiple specialized algorithms to handle different types of inputs. However, even with these specialized kernels, the fundamental problem of poor data locality in skewed matrices leads to catastrophic cache performance and a massive drop in overall GFLOP/s.

## Chapter 5: Architectural Deep Dive: Independent Thread Scheduling

The way a GPU schedules and executes threads is fundamental to its performance, especially when dealing with complex control flow (like if-else statements). This execution model, known as SIMT (Single Instruction, Multiple Thread), has evolved significantly. Understanding this evolution helps explain why certain programming patterns are more efficient than others.

### The Classic SIMT Model (Pascal and Earlier)

On older architectures like Pascal, a warp (a group of 32 threads) operated like a single unit with one program counter (PC) and one call stack. To handle branches where some threads take an if path and others take the else path, the hardware used an active mask.

1. Divergence: When an if statement is encountered, threads that don't meet the condition are "masked off" (made inactive).
2. Execution: The GPU executes the entire if block for the active threads.
3. Mask Inversion: The active mask is then inverted. The threads that just ran are masked off, and the threads that originally failed the condition are made active.
4. Execution: The GPU executes the else block for the newly active threads.
5. Reconvergence: After the else block, all threads in the warp become active again and proceed together.

* Description of the Diagram: A flow chart shows a single execution path. At a divergence point (if), the path for A; B; is executed first, followed by the path for X; Y;. Only after both serialized paths are complete does the execution reconverge to execute Z;.

The major drawback of this model is branch serialization. Even though different threads are doing different work, they cannot do it at the same time. The hardware must execute each branch path sequentially, causing a significant performance penalty for divergent code. This model could also lead to deadlock if threads within a warp tried to synchronize with each other across a divergent branch.

### The Modern SIMT Model (Volta and Later)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Independent Thread Scheduling)</span></p>

Starting with the Volta architecture, NVIDIA introduced **Independent Thread Scheduling (ITS)**. In the ITS model, the GPU maintains the execution state (program counter, register state) for each individual thread. A **schedule optimizer** dynamically groups active threads from the same warp that are executing the same instruction. Threads can now diverge and reconverge at sub-warp granularity.

</div>

* A schedule optimizer is now responsible for dynamically grouping active threads from the same warp that are executing the same instruction and issuing that instruction to the SIMT execution units.
* Threads can now diverge and reconverge at a sub-warp granularity.
* Description of the Diagram: A flow chart shows the execution path diverging. The A; B; block and the X; Y; block are shown side-by-side, indicating they can be scheduled more flexibly. Crucially, after A; B; finishes, its threads can immediately start executing Z; without waiting for the X; Y; block to complete. Similarly, threads from the other branch can start Z; as soon as they are done.

Execution is still SIMT at the core—the hardware still executes one common instruction across multiple threads at a time. However, the scheduler can now group any threads from a warp that are at the same point in the code, rather than being constrained by a single warp-wide program counter.

One subtlety is that the hardware does not automatically force a full warp reconvergence at the point where the branches would logically meet (e.g., at statement Z in the example). This is a conservative approach because code in one branch might produce data needed by another branch if synchronization were involved.

To force a reconvergence point and ensure all threads in a warp have reached a specific point before any proceed, developers can use the __syncwarp() intrinsic.

### Implications for Developers and Starvation-Free Algorithms

Independent Thread Scheduling enables starvation-free algorithms. This means that if multiple threads are contending for a shared resource (like a lock), the system guarantees that any given thread will eventually be scheduled and make progress.

Consider a lock (mutual exclusion):

* Thread #0 acquires a lock.
* Thread #1, which needs the lock, is scheduled to run. It spins, waiting for the lock.
* In the old model, if Thread #0 and #1 were in the same warp, Thread #0 might never be scheduled again to release the lock, causing a deadlock.
* With ITS, the scheduler will eventually give Thread #0 a chance to run, allowing it to release the lock and ensuring the system makes forward progress.

If you need the old, stricter warp-synchronous behavior for certain algorithms (like a warp-level reduction), you can:

* Use sync-variant primitives like __shfl_down_sync().
* Explicitly call __syncwarp().
* Compile your code for an older architecture (e.g., nvcc -arch=compute_60 -code=sm_70) to force the compiler to generate code compatible with the old scheduling model.

## Chapter 6: A Survey of Profiling Tools and Techniques

While NVIDIA's Nsight suite is the industry standard, it's part of a broader ecosystem of tools and research projects for performance analysis. Understanding these alternatives provides a more complete picture of the field.

### An Overview of Available Tools

We can categorize profiling tools based on their underlying technology:

| Category | Examples | Pros | Cons |
| --- | --- | --- | --- |
| Hardware Counter-Based | nvprof (legacy), Nsight | Provides detailed hardware metrics (cache hits, etc.) | Heavy performance impact, slowdown due to kernel replays. |
| GPU Simulators | GPGPU-Sim, Multi2Sim, Barra | Extremely detailed cycle-accurate analysis. | Very slow; often lag behind the latest hardware generations. |
| Instrumentation-Based | GPU Ocelot, SASSI, NVBit, CUDA Flux | Fast, low overhead, allows for custom profiling logic. | Cannot measure hardware metrics; lifetime of research tools is limited. |
| CUDA API Trace | Part of Nsight Systems | Traces calls to the CUDA runtime API. | - |

### CUDA Flux: An LLVM-Based Instrumentation Profiler

CUDA Flux is a research tool developed at Heidelberg University that offers a lightweight alternative to hardware counter-based profiling. It works by instrumenting the code at the compiler level.

### The LLVM Framework and CUDA

Modern compilers like clang use the LLVM Compiler Framework. This framework has a modular design:

1. Front-end: Parses source code (like C++) into an Intermediate Representation (IR).
2. Middle-end: Performs optimizations on the IR. This is where CUDA Flux hooks in.
3. Back-end: Converts the optimized IR into machine code (PTX and SASS for GPUs).

CUDA Flux adds a custom "pass" to the LLVM middle-end that injects extra instructions into the code to count how many times each basic block is executed.

### How CUDA Flux Works

1. PTX Processing: It first analyzes the PTX assembly for each kernel to create a summary of how many instructions are in each basic block (a straight-line sequence of code with no branches in or out, except at the beginning and end).
2. Instrumentation: It then instruments the code at the IR level. At the beginning of each basic block, it inserts an instruction to increment a counter specific to that block.
3. Calculation: After the kernel runs, the tool uses the execution counts for each basic block and the instruction summary to calculate the total number of PTX instructions executed.

This profiling can be done at different granularities: for a single warp, a single CTA (thread block), or the entire grid.

### Advantages and Limitations

Advantages:

* Fine-grained: Provides detailed instruction counts.
* Low Overhead: The time taken does not depend on the number of metrics being monitored, avoiding the kernel replay issue of ncu.
* Accessible: PTX is a more stable and accessible target for analysis than the constantly changing SASS.

Limitations:

* PTX, not SASS: It profiles PTX instructions, which is one step removed from what the hardware actually runs. The mapping is not always one-to-one.
* Build System Modification: It requires changing the build system to use clang++ instead of nvcc, which can be complex.
* Clang Limitations: It may not support all the newest CUDA features, such as texture memory.

### Excursion: Predictive Performance Modeling

A cutting-edge area of research is predictive performance modeling. The goal is to predict the performance (time, power, energy) of an application on a processor without actually running it, or at least without running it on every possible hardware configuration. This is incredibly useful for:

* Making runtime scheduling decisions.
* Exploring performance on hardware you don't have access to.
* Guiding co-design of future hardware and software.

### GPU Mangrove: A Portable Prediction Model

GPU Mangrove is a research project that uses a machine learning approach for performance prediction.

**Methodology:**

1. Feature Extraction: It uses a tool like CUDA Flux to extract a set of portable code features from a kernel. These features depend only on the code and its inputs, not on the target hardware. Examples include:
    * Instructions executed (total FLOPs, memory ops, etc.)
    * Memory footprint
    * Kernel launch configuration (grid and block dimensions)
    * Computational intensity
2. Model Training: For a specific GPU, it measures the actual execution time and power consumption of a large suite of diverse kernels (189 unique kernels from benchmarks like Rodinia and SHOC). It then trains a RandomForest machine learning model to learn the relationship between the portable code features and the measured performance on that GPU.
3. Prediction: To predict the performance of a new kernel on that GPU, it extracts its portable features and feeds them into the trained model.

**Results:** This approach has proven to be quite effective.

* Accuracy: Achieved prediction accuracy of 8.86–52.0% for execution time and an impressive 1.84–2.94% for power consumption across five different GPUs.
* Speed: Prediction is very fast, taking only 15-108 milliseconds.

This type of learning-based model represents a powerful new way to reason about performance in our increasingly heterogeneous and complex computing landscape.

## Chapter 7: Optimizing GPU Applications: A Case Study in N-Body Simulations

Welcome to the next chapter in our exploration of GPU computing. So far, we have covered the fundamentals of the GPU architecture and the CUDA programming model. Now, we will dive deeper into one of the most critical aspects of high-performance computing: optimization.

Effective GPU programming is not just about writing code that runs in parallel; it's about writing code that leverages the specific strengths of the GPU architecture to achieve maximum performance. In this chapter, we will explore advanced optimization techniques by examining a classic and computationally demanding problem: the N-body simulation. We will start with a simple, "naive" implementation and progressively refine it, demonstrating how architectural awareness can lead to dramatic speedups.

### Advanced Memory Layout Optimizations

Before we tackle the N-body problem, we must first discuss a foundational optimization strategy that is crucial for GPU performance: how we arrange our data in memory.

#### The High Cost of Memory Access

In any modern computer system, from a laptop CPU to a high-end GPU, accessing data from main memory is one of the most expensive operations an application can perform. It takes significantly more time and energy than performing an arithmetic calculation. Therefore, a primary goal of performance optimization is to minimize and streamline memory access. On a GPU, where thousands of threads can request data simultaneously, this becomes paramount.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(AoS vs. SoA)</span></p>

- **Array of Structures (AoS):** All fields of one element are stored contiguously. Intuitive but causes non-coalesced GPU memory access.
- **Structure of Arrays (SoA):** All values of one field are stored contiguously. Naturally suited for coalesced memory access on GPUs.

</div>

#### Array of Structures (AoS): The Intuitive Approach

When programming in languages like C or C++, the most common way to group related data is using a struct. For a particle simulation, we might define a particle like this:

```c++
// Define the structure for a single particle
struct p_t {
  float x, y, z;      // Position
  float vx, vy, vz;   // Velocity
  float mass;
};

// Create an array to hold many particles
p_t particles[MAX_SIZE];
```

This is called an Array of Structures (AoS). In memory, the data for each particle is laid out contiguously. The position, velocity, and mass of particle[0] are stored together, followed immediately by all the data for particle[1], and so on.

Memory Layout (AoS): [x0, y0, z0, vx0, ..., m0] [x1, y1, z1, vx1, ..., m1] [x2, ...] 

This layout is intuitive and works well for many single-threaded applications where you typically process one entire object at a time.

#### Structure of Arrays (SoA): The GPU-Friendly Approach

An alternative data layout is the Structure of Arrays (SoA). Instead of one large array of particle structures, we create a single structure that contains arrays for each attribute.

```c++
// Define a structure containing arrays for each particle attribute
struct p_t {
  float x[MAX_SIZE];
  float y[MAX_SIZE];
  float z[MAX_SIZE];
  float vx[MAX_SIZE];
  float vy[MAX_SIZE];
  float vz[MAX_SIZE];
  float mass[MAX_SIZE];
};

// Create the single structure instance
p_t particles;
```

In this layout, all the x-positions are stored together, all the y-positions are stored together, and so on.

Memory Layout (SoA): [x0, x1, x2, ...] [y0, y1, y2, ...] [z0, z1, z2, ...] ...

While this might seem less intuitive and requires more pointers to manage, it is often the superior choice for GPU applications. To understand why, we need to revisit the concept of memory coalescing.

#### A Refresher on Memory Coalescing

As we've learned, threads on a GPU are grouped into warps (typically 32 threads). When threads in a warp access global memory, the GPU tries to service all of their requests with a single, large memory transaction. This is called coalesced memory access.

* Coalesced Access: Occurs when threads in a warp access consecutive memory addresses. This is the ideal, most efficient scenario, as a single transaction can satisfy all 32 threads at once.
* Non-Coalesced Access: Occurs when threads access scattered, non-consecutive memory locations. This forces the GPU to issue multiple memory transactions to satisfy the requests of a single warp, leading to a significant performance penalty.

#### Visualizing Memory Access: AoS vs. SoA

The source context provides a helpful diagram to visualize this difference. Let's describe it.

Imagine four threads (Thread 1, 2, 3, 4) in a warp, each tasked with processing a particle's position. Let's say each thread needs to read the x coordinate of its assigned particle.

Case 1: Array of Structures (AoS) The memory is laid out as [x1, y1, z1, m1], [x2, y2, z2, m2], and so on.

* Thread 1 wants x1.
* Thread 2 wants x2.
* Thread 3 wants x3.
* Thread 4 wants x4.

The data they want (x1, x2, x3, x4) is separated by other data (y, z, m). This is a non-coalesced access pattern. The memory controller has to fetch a larger chunk of memory for each thread and discard the unneeded y, z, and m data, or issue multiple separate transactions.

Case 2: Structure of Arrays (SoA) The memory is laid out as [x1, x2, x3, x4, ...], [y1, y2, y3, y4, ...], etc.

* Thread 1 wants x1.
* Thread 2 wants x2.
* Thread 3 wants x3.
* Thread 4 wants x4.

Here, the data they need is located in consecutive memory locations. The GPU can satisfy all four requests with a single memory transaction. This is a perfectly coalesced access.

While AoS can sometimes be made to work by using data types like float4 to pack values together, the SoA layout is naturally suited for coalesced memory access on GPUs and is a key technique for memory-bound applications.

### The Tiling Technique for Data Reuse

Another powerful optimization is tiling, which focuses on maximizing the use of data once it has been loaded into the GPU's faster, on-chip memory.

#### What is Tiling?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tiling)</span></p>

**Tiling** is a technique used to divide a large, repetitive computation into smaller, regular blocks or "tiles." By processing the problem one tile at a time, we can manage resource usage more effectively and improve data locality. The primary goal is to maximize **data reuse**: load data from slow global memory into fast shared memory once, then use it as many times as possible before discarding.

</div>

Imagine you need to multiply two very large matrices. Instead of loading entire rows and columns from slow global memory for every single calculation, you can:

1. Load a small sub-matrix (a tile) from each input matrix into the fast shared memory.
2. Perform all possible calculations using only the data within those tiles.
3. Load the next set of tiles and repeat.

This makes the computation more regular and often independent of the overall problem size. The tiling approach breaks a large $N \times N$ matrix into smaller $p \times p$ sub-matrices. A thread block loads the data for one sub-problem into shared memory, computes all interactions within that tile, synchronizes, and then moves to the next tile.

#### A Practical Application: N-Body Simulations

Now, let's apply these concepts to a real-world problem: simulating the interactions of a large number of particles, a task known as an N-body simulation.

##### What are N-Body Simulations?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(N-Body Simulation)</span></p>

**N-body simulations** are a class of problems where the goal is to simulate the evolution of a system of $N$ bodies (particles) that interact with each other through a fundamental force like gravity or electromagnetism. The computational complexity is $O(N^2)$ for all-pairs calculations, making them extremely well-suited for GPU acceleration.

</div>

These simulations are foundational in many scientific fields:

* Astrophysics: Simulating the formation and evolution of galaxies, where each "body" is a star or a cluster of stars interacting via gravitational forces. N-body simulations were instrumental in the 1990s discovery of dark energy and are used today to model phenomena like the collision of neutron stars.
* Biomolecular Systems: Modeling the folding of proteins or the behavior of viruses like the Satellite Tobacco Mosaic Virus (STMV). Here, the "bodies" are atoms, and the interactions are complex electrostatic and Van der Waals forces. Simulating these systems is a massive computational challenge, with a single day of simulation time for STMV (100 million atoms) requiring petascale computing power.

#### The Computational Challenge of N-Body Problems

The core of an N-body simulation is calculating the total force exerted on each body by every other body in the system. If there are N bodies, then for each body, we must calculate the force from the other N-1 bodies. This leads to a computational complexity of O(N^2). For a system with a million bodies, this is a trillion interactions per time step!

This makes the naive all-pairs calculation computationally bound, meaning the processor's speed is the main bottleneck. The memory requirement is only $O(N)$ (to store positions and velocities), but the compute cost is $O(N^2)$.

To make this tractable for enormous systems, scientists use hierarchical algorithms like the Barnes-Hut algorithm ($O(N \log N)$) which approximate the forces from distant clusters of bodies. However, for interactions within a cluster, an all-pairs method ($O(k^2)$ for a cluster of size $k$) is still used, and this portion is extremely well-suited for GPUs.

#### The Physics Behind the Simulation

The simulation is based on Newton's second law of motion:

$$F(x(t)) = m \frac{d^2x(t)}{dt^2}$$

We approximate the solution to this differential equation by discretizing time into small steps ($\delta t$). In each step, we calculate forces, update velocities, and then update positions.

The gravitational force ($f_{ij}$) between two bodies $i$ and $j$ with masses $m_i$, $m_j$ and positions $x_i$, $x_j$ is given by:

$$f_{ij} = G \cdot \frac{m_i m_j}{\|d_{ij}\|^2} \cdot \frac{d_{ij}}{\|d_{ij}\|}$$

where $d_{ij} = x_j - x_i$ is the vector between them and $G$ is the gravitational constant.

To avoid a division-by-zero error if two particles get too close ($\|d_{ij}\| \to 0$), a **softening factor** ($\epsilon^2$) is introduced in the denominator:

$$f_{ij} = G \cdot \frac{m_i m_j\, d_{ij}}{(\|d_{ij}\|^2 + \epsilon^2)^{3/2}}$$

The total force $F_i$ on body $i$ is the sum of the forces from all other bodies $j$:

$$F_i = \sum_{j=1}^{N} f_{ij} = G\, m_i \sum_{j=1}^{N} \frac{m_j\, d_{ij}}{(\|d_{ij}\|^2 + \epsilon^2)^{3/2}}$$

Once we have the total force $F_i$, we can update the velocity and position using a numerical integration method like the **Leapfrog Verlet** algorithm:

$$v_i\!\left(t + \tfrac{1}{2}\delta t\right) = v_i\!\left(t - \tfrac{1}{2}\delta t\right) + \delta t\, \frac{F_i}{m_i}$$

$$x_i(t + \delta t) = x_i(t) + \delta t \cdot v_i\!\left(t + \tfrac{1}{2}\delta t\right)$$

### Implementing an N-Body Simulation on the GPU

Our goal is to implement the force calculation step on the GPU, as it is the most computationally expensive part (O(N^2)).

#### Initial Design: One Thread Per Body

A natural way to partition the problem for a GPU is to assign one thread to calculate the total force for one body. Each thread will:

1. Load the position and mass of its assigned body.
2. Iterate through all other N-1 bodies.
3. For each other body, calculate the interaction force and add it to an accumulator.
4. Write the final total force back to global memory.

This approach requires communication (each thread needs to read the data of all other bodies) and offers a prime opportunity to optimize for data re-use. We will explore two implementations: a naive one and a tiled one.

#### A Naive GPU Implementation

Let's start with a straightforward implementation. First, a helper function to calculate the interaction between two bodies. This function can be used by both the CPU (__host__) and GPU (__device__).

##### bodyBodyInteraction Helper Function

This function takes the properties of two bodies and calculates the force components (fx, fy, fz) that body 1 exerts on body 0. It performs approximately 16 single-precision floating-point operations (FLOPs).

```c++
__host__ __device__ void bodyBodyInteraction(
    float *fx, float *fy, float *fz,
    float x0, float y0, float z0,
    float x1, float y1, float z1, float mass1,
    float softeningSquared)
{
    // Calculate distance vector components
    float dx = x1 - x0;
    float dy = y1 - y0;
    float dz = z1 - z0;

    // Calculate squared distance and add softening factor
    float distSqr = dx*dx + dy*dy + dz*dz;
    distSqr += softeningSquared;

    // Calculate 1 / (dist^3) using the fast reciprocal square root intrinsic
    float invDist = rsqrtf(distSqr);
    float invDistCube =  invDist * invDist * invDist;
    float s = mass1 * invDistCube;

    // Calculate force components
    *fx = dx * s;
    *fy = dy * s;
    *fz = dz * s;
}
```


##### The ComputeNBodyGravitation_GPU_AOS Kernel

Now for the main kernel. This kernel uses the AoS data layout with packed float4 values to improve memory access. Each thread calculates the total force for one body i.

```c++
__global__ void ComputeNBodyGravitation_GPU_AOS(
    float *force, float *posMass, size_t N, float softeningSquared)
{
    // Outer loop to handle cases where N > number of threads (grid-stride loop)
    for (int i = blockIdx.x*blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x*gridDim.x)
    {
        // Accumulator for total force on body 'i'
        float acc[3] = {0};

        // Load position and mass of body 'i'
        float4 me = ((float4 *) posMass)[i];
        float myX = me.x; float myY = me.y; float myZ = me.z;

        // Inner loop to iterate through all other bodies 'j'
        for (int j = 0; j < N; j++) {
            // Load position and mass of body 'j'
            float4 body = ((float4 *) posMass)[j];

            // Calculate interaction force
            float fx, fy, fz;
            bodyBodyInteraction(
                &fx, &fy, &fz, myX, myY, myZ,
                body.x, body.y, body.z, body.w, // .w component stores mass
                softeningSquared);

            // Accumulate the force
            acc[0] += fx; acc[1] += fy; acc[2] += fz;
        }

        // Write the final total force to global memory
        force[3*i+0] = acc[0];
        force[3*i+1] = acc[1];
        force[3*i+2] = acc[2];
    }
}
```


##### Code Breakdown

* i = blockIdx.x*blockDim.x + threadIdx.x: This is the standard formula to calculate a unique global index for each thread. The outer for loop is a grid-stride loop, which makes the kernel flexible—it works correctly even if we launch fewer threads than the number of bodies N.
* float4 me = ((float4 *) posMass)[i]: We load the data for the thread's assigned body (me). By casting the posMass pointer to float4*, we are telling the hardware to perform a single 16-byte load, which can be more efficient. The x, y, z components store position, and the w component stores mass.
* for (int j = 0; j < N; j++): This is the critical inner loop. The thread iterates through all N bodies in the system.
* float4 body = ((float4 *) posMass)[j]: Inside the loop, the thread loads the data for each body j from global memory.
* Data Reuse: Notice that me is loaded once and reused N times inside the inner loop. The data for body is loaded from global memory in every single iteration. However, because all threads in the GPU are executing this same inner loop, they will all be requesting the same body data at roughly the same time. This means the data for body j will be loaded from global memory and can be temporarily stored in the L1/L2 caches, benefiting all threads that need it.

#### Performance and the Power of Loop Unrolling

Even in this naive implementation, we can apply a simple but effective optimization: loop unrolling. This technique reduces branch overhead by expanding the loop body, performing more work per iteration. We can hint to the compiler to do this using a #pragma.

```c++
#pragma unroll 16
for (int j = 0; j < N; j++) {
    // ... body of the loop ...
}
```

The optimal unroll factor must be found empirically by testing different values. The results show a significant performance gain:

| Version | Unroll Factor | Body-Body Interactions per Second [G] |
| --- | --- | --- |
| GPU Naive | 1 (none) | 25.0 |
| GPU Naive | 2 | 30.0 |
| GPU Naive | 16 | 34.3 |

Loop unrolling improves performance by over 37% in this case.

### An Optimized Tiled Implementation Using Shared Memory

The naive version relies on the GPU's hardware caches to exploit data reuse. We can achieve even better performance by explicitly managing data reuse with tiling and shared memory.

#### Applying the Tiling Strategy to N-Body

The idea is to break the N \times N interaction calculation into smaller tiles. Each thread block will work on one tile at a time.

1. Each thread is still responsible for one body i.
2. The inner loop that iterates over all j bodies is tiled.
3. In each step of the tiled loop, all threads in the block cooperate to load a "tile" of blockDim.x bodies into shared memory.
4. A synchronization barrier (__syncthreads()) is used to ensure all threads have finished loading before any thread starts computing.
5. Each thread then iterates through the bodies in the shared memory tile, accumulating forces.
6. Another synchronization barrier is used before the block proceeds to load the next tile.

This strategy drastically reduces global memory traffic. A tile of body data is loaded once from global memory into fast shared memory, and then every thread in the block can reuse it blockDim.x times.

#### Code Breakdown: The Tiled ComputeNBodyGravitation_Shared Kernel

```c++
__global__ void ComputeNBodyGravitation_Shared(
    float *force, float *posMass, float softeningSquared, size_t N)
{
    // Dynamically allocated shared memory for a tile of bodies
    extern __shared__ float4 shPosMass[];

    // Grid-stride loop for each thread to work on a body 'i'
    for (int i = blockIdx.x*blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x*gridDim.x)
    {
        float acc[3] = {0};
        float4 myPosMass = ((float4 *) posMass)[i];

        // Outer loop that strides through the N bodies in tiles of size blockDim.x
        #pragma unroll 32 // Unroll factor can be applied here too
        for (int j = 0; j < N; j += blockDim.x) {
            // Cooperative load: each thread loads one body into the shared memory tile
            shPosMass[threadIdx.x] = ((float4 *) posMass)[j + threadIdx.x];

            // Synchronize to ensure all data is loaded before computation
            __syncthreads();

            // Inner loop iterates over the tile in shared memory
            for (size_t k = 0; k < blockDim.x; k++) {
                float fx, fy, fz;
                // Read body data from FAST shared memory
                float4 bodyPosMass = shPosMass[k];
                bodyBodyInteraction(
                    &fx, &fy, &fz,
                    myPosMass.x, myPosMass.y, myPosMass.z,
                    bodyPosMass.x, bodyPosMass.y, bodyPosMass.z, bodyPosMass.w,
                    softeningSquared);
                acc[0] += fx; acc[1] += fy; acc[2] += fz;
            }

            // Synchronize to ensure all computations are done before loading the next tile
            __syncthreads();
        }
        force[3*i+0] = acc[0]; force[3*i+1] = acc[1]; force[3*i+2] = acc[2];
    }
}
```

##### Key Differences

* extern __shared__ float4 shPosMass[]: This declares a dynamically sized shared memory array. The actual size is specified during kernel launch.
* for (int j = 0; j < N; j += blockDim.x): This is the tiling loop. Instead of incrementing j by 1, we jump by the block size.
* shPosMass[threadIdx.x] = ...: This is the cooperative load. Each thread threadIdx.x in the block is responsible for loading one body's data into the corresponding slot in shPosMass. This is a perfectly coalesced read from global memory.
* __syncthreads(): This is a barrier synchronization. It forces all threads in the block to wait at this point until every single thread has reached it. This is essential to prevent a thread from trying to read data from shPosMass before another thread has finished writing it.
* float4 bodyPosMass = shPosMass[k]: Inside the innermost loop, the body data is now read from the extremely fast on-chip shared memory, not slow global memory. This is the source of the performance gain.

#### Performance Analysis: The Impact of Shared Memory

Adding tiling and shared memory provides another significant performance boost on top of our previous optimizations.

| Version | Unroll Factor | Body-Body Interactions per Second [G] |
| --- | --- | --- |
| GPU Naive | 16 | 34.3 |
| GPU Shmem | 1 (none) | 38.2 |
| GPU Shmem | 2 | 44.5 |
| GPU Shmem | 4 | 45.2 |

The optimized version with shared memory achieves 45.2 G-interactions/sec, a 32% improvement over the best naive version and an 80% improvement over the original unoptimized kernel.

### A Look at CPU Optimizations and Performance Comparison

While GPUs are excellent for this problem, it's insightful to see how an optimized CPU version performs. Modern CPUs also have parallel capabilities through SIMD (Single Instruction, Multiple Data) vector units, such as SSE or AVX.

#### Vectorization on the CPU

The provided code snippet for bodyBodyInteraction using __m128 data types is an example of a CPU implementation using SSE intrinsics. __m128 is a 128-bit data type that can hold four 32-bit floating-point numbers. Functions like _mm_add_ps perform a parallel add on all four floats at once. This is a form of fine-grained parallelism available on the CPU.

```c++
// Example of a CPU SSE vectorized function
inline void bodyBodyInteraction(__m128& fx, /*...*/) {
    // r_01  [3 FLOPS]
    __m128 dx = _mm_sub_ps( x1, x0 );
    // ... more SIMD operations ...
    // s = m_j * invDistCube [1 FLOP]
    __m128 s = _mm_mul_ps(mass1,invDistCube);
    // ... accumulate results ...
    fx = _mm_add_ps( fx, _mm_mul_ps(dx, s) );
}
```


This is fundamentally different from the GPU's SIMT (Single Instruction, Multiple Thread) model. In SIMD, the programmer explicitly manages vectors of data. In SIMT, the programmer writes code for a single scalar thread, and the hardware executes many of these threads in parallel on its vector-like units.

#### Performance Showdown: CPU vs. GPU

The performance comparison uses an Intel E5-2670 CPU and a GK104 GPU.

| Version | Body-Body Interactions per Second [G] |
| --- | --- |
| CPU naive, single thread | 0.017 |
| CPU SSE, single thread | 0.307 |
| CPU SSE, 32 threads | 5.650 |
| GPU naive, best unroll | 34.3 |
| GPU shmem, best unroll | 45.2 |

Analysis:

* Vectorizing the CPU code (SSE) gives a 18x speedup over the naive single-threaded version.
* Using all 32 threads of the multicore CPU gives another 18x speedup.
* However, even the best multi-threaded, vectorized CPU implementation (5.65 G-interactions/sec) is significantly slower than the GPU.
* The optimized GPU implementation is 8x faster than the highly optimized 32-thread CPU version, showcasing why GPUs are the prime architecture for problems like N-body simulations.

### Chapter Summary and Further Optimizations

#### Recap of Key Techniques

In this chapter, we saw how to transform a naive GPU implementation into a highly-optimized one, resulting in a dramatic performance increase. The key takeaways are:

* Data Layout Matters: Choosing a Structure of Arrays (SoA) layout is often critical for achieving coalesced memory access on the GPU.
* Maximize Data Reuse: The tiling technique, combined with explicit management of on-chip shared memory, is a powerful way to reduce expensive global memory traffic.
* Compiler Optimizations: Simple directives like #pragma unroll can provide significant speedups by reducing instruction and branch overhead.
* N-Body is a Prime Example for GPUs: The massive data parallelism and high arithmetic intensity of N-body problems make them exceptionally well-suited for the GPU's many-core architecture.

#### Other Optimization Paths

The optimizations don't stop here. The lecture slides mention several other advanced techniques that were skipped for brevity but are worth knowing:

* Warp Shuffle Instructions (__shfl()): These are special instructions that allow threads within the same warp to exchange data directly without needing to use shared memory. This can be even faster than shared memory but requires restructuring the algorithm to work at the warp level (e.g., tiling at a size of 32) instead of the block level.
* Constant Memory: For data that is read-only and accessed by all threads, placing it in constant memory can be beneficial. The GPU has a dedicated cache for constant memory, which is optimized for broadcasting a single value to all threads in a warp. However, updating constant memory requires a host-to-device transfer, making it unsuitable for data that changes every time step, as in our N-body simulation.

## Chapter 7: A Deep Dive into GPU Computing: Architecture and Programming

### Chapter 1: Overcoming the Host-Device Bottleneck

Welcome to the world of high-performance computing with GPUs! In previous discussions, we've focused on how to write code that runs in parallel on the GPU itself. Now, we'll address a critical, real-world challenge: getting data to and from the GPU efficiently. This chapter explores the fundamental bottleneck in GPU computing—the connection between the CPU and the GPU—and introduces CUDA Streams, a powerful technique for hiding data transfer latency and unlocking even more performance.

#### 1.1 The GPU as a Peripheral Device: A Tale of Two Speeds

A modern computer system is a collection of specialized components. The CPU (Central Processing Unit) acts as the general-purpose brain, while the GPU (Graphics Processing Unit) serves as a highly parallel co-processor, or an accelerator. The CPU and its main memory (the host) are connected to the GPU and its dedicated memory (the device) via a peripheral interface, typically the PCIe (Peripheral Component Interconnect Express) bus.

This architectural separation is the source of a major performance challenge. To understand why, let's examine the bandwidth—the rate at which data can be moved—at different points in the system.

A diagram of the system architecture reveals a stark contrast:

* On-Device Memory Bandwidth: The connection between the GPU processor and its own dedicated memory is incredibly fast. Modern GPUs can have a memory bandwidth of up to 3.3 TB/s. This is like having a multi-lane superhighway for data.
* PCIe Bus Bandwidth: The connection between the host system and the GPU device is significantly slower. The PCIe interface typically offers a bandwidth of around 64 GB/s. This is more like a local access road.

This massive difference creates a bandwidth mismatch. The GPU can process data at a phenomenal rate (e.g., 34-67 TFLOP/s for double-precision), but it can only receive that data from the host at a much slower pace. If we are not careful, the GPU will spend most of its time waiting for data to arrive, completely wasting its computational power. This is often referred to as being memory-bound or, more specifically, PCIe-bound.

Therefore, a key objective for any high-performance GPU application is to overlap communication and computation—that is, to keep the GPU busy with calculations while new data is being transferred over the PCIe bus.

#### 1.2 Exploiting Task Parallelism on the Host

Up to this point, our focus has been on data parallelism: taking a single task (like adding two vectors) and splitting the data across thousands of GPU threads to be processed simultaneously.

Now, we introduce a new concept: task parallelism. This involves breaking down the entire workflow into independent tasks that can be executed concurrently on the host side. Instead of performing our operations in a strict sequence:

1. Copy data from Host to Device (H2D)
2. Execute a kernel on the Device
3. Copy results from Device to Host (D2H)

We want to orchestrate these steps so that, for example, the GPU is executing a kernel on one chunk of data at the same time it is receiving the next chunk of data from the host. This is where CUDA Streams come in.

#### 1.3 Introducing CUDA Streams

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(CUDA Stream)</span></p>

A **CUDA Stream** is an ordered queue of operations submitted to the GPU. Key properties:

* **Ordered Execution:** Within a single stream, operations execute in FIFO order.
* **Asynchronous Execution:** When the CPU issues a command to a stream, it returns immediately, allowing the CPU to continue queuing more work.
* **Inter-Stream Independence:** Operations in different streams have no guaranteed execution order relative to each other, enabling concurrency.

</div>

*Analogy: The Supermarket Checkout.* A single checkout lane (the default stream) serializes everything. Multiple checkout lanes (multiple CUDA Streams) allow one lane to process a large order (kernel execution) while another handles a quick transaction (data transfer).

By placing independent operations into different streams, we can enable the GPU's hardware to execute them concurrently, effectively hiding the latency of data transfers behind useful computation.

#### 1.4 The Levels of Concurrency

CUDA Streams unlock a higher level of concurrency, which we can call coarse-grained concurrency, to complement the fine-grained concurrency we already know. Let's break down the different types of parallelism a modern GPU can exploit:

1. Fine-Grained Concurrency (Within a Kernel): This is the parallelism at the instruction and thread level that we are familiar with. It involves overlapping the execution of instructions and threads to hide memory latency within the GPU itself.
2. Coarse-Grained Concurrency (Enabled by Streams):
  * CPU/GPU Concurrency: While the GPU is busy executing tasks from a stream, the CPU is free to perform other work, including queuing up more tasks for the GPU.
  * Concurrent Copy & Execute: This is the primary goal. A memory copy operation (e.g., cudaMemcpyAsync) can execute at the same time as a kernel, provided they are in different streams and the hardware supports it.
  * Concurrent Kernel Execution: Modern GPUs (Compute Capability 2.x and later) can execute multiple kernels from different streams simultaneously.
  * Multi-GPU Concurrency: If a host system has multiple GPUs, each can operate in parallel on its own set of streams, dramatically increasing throughput.

#### 1.5 Managing Host-Device Synchronization

Since the CPU queues up work asynchronously, we need mechanisms to check on the GPU's progress and ensure that results are ready before we try to use them. CUDA provides three primary ways to manage synchronization.

##### Context-Based Synchronization

This is the most heavy-handed approach. It blocks the CPU until all previously issued CUDA operations on the device have completed, regardless of which stream they were in.

* Functions: cudaDeviceSynchronize() is the most common function for this. Many blocking calls, like the standard cudaMemcpy(), also have this effect implicitly.
* When to Use: Use this when you need a hard barrier, for example, right before the CPU needs to access the final results calculated by the GPU.

##### Stream-Based Synchronization

This offers more granular control by targeting a specific stream. It blocks the CPU until all operations in a particular stream have completed.

* cudaStreamSynchronize(stream): This function will pause the host thread until the specified stream is empty.
* cudaStreamQuery(stream): This is a non-blocking alternative. It checks the status of the stream and immediately returns either cudaSuccess (if all operations in the stream are complete) or cudaErrorNotReady (if the GPU is still working). This is useful for building more complex scheduling logic on the host.

##### Event-Based Synchronization

This is the most fine-grained and flexible method. An event is like a marker or a checkpoint that you can place into a stream.

1. Record an Event: You use cudaEventRecord(event, stream) to place an event into a specific stream. When the GPU processes all operations in the queue up to that point, the event is considered "recorded," and a timestamp is captured.
2. Wait for an Event: You can then have the CPU wait for that specific event to be recorded using cudaEventSynchronize(event). This call will block until the event marker has been passed in its stream.
3. Query an Event: Similar to streams, you can use the non-blocking cudaEventQuery(event) to check if an event has been recorded without halting the CPU.

Events are powerful because they allow you to synchronize based on specific points in your workflow, rather than waiting for an entire stream or the entire device to finish.


---


### Chapter 2: Programming with CUDA Streams

Now that we understand the theory behind CUDA Streams, let's put it into practice. This chapter will walk you through the process of converting a standard, sequential GPU workflow into a pipelined, high-performance one using streams. We'll examine the necessary API calls, look at code examples, and discuss important architectural details that can affect performance.

#### 2.1 The Default Stream and Sequential Execution

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Default Stream)</span></p>

When you launch a kernel or call `cudaMemcpy` without specifying a stream, you are using the **default stream** (stream 0). It is a **synchronizing stream**: an operation in the default stream will wait for all preceding operations in all other streams to complete before it begins, and any subsequent operation in any other stream will wait for the default stream operation to complete. This creates an inherent synchronization point, making overlap impossible with the default stream.

</div>

Consider this typical sequence of operations for a SAXPY kernel, which computes y[i] = \alpha \cdot x[i] + y[i]:

```c++
// All operations below are implicitly in the default stream
// 1. Copy input data X to device
cudaMemcpy(dx, hx, numBytes, cudaMemcpyHostToDevice);

// 2. Launch the kernel
saxpy<<<numBlocks, blockSize>>>(dx, dy, alpha, N);

// 3. Copy result data Y back to host
cudaMemcpy(hy, dy, numBytes, cudaMemcpyDeviceToHost);
```


Because all three operations are in the default stream, they will execute in a strictly sequential order. The kernel will not start until the first cudaMemcpy is completely finished, and the second cudaMemcpy will not start until the kernel is completely finished. This serialization is exactly what we want to avoid and is a prime example of the "serial fraction" described by Amdahl's Law, which limits the potential speedup of any parallel program.

##### Device Overlap Capability

Most modern CUDA devices support a feature called "Device Overlap," often referred to as "Concurrent copy and execute." This is the hardware capability that allows a kernel to run at the same time as a data transfer. You can programmatically check for this capability:

```c++
int dev_count;
cudaGetDeviceCount(&dev_count);

for (int i = 0; i < dev_count; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    if (prop.deviceOverlap) {
        // This device supports concurrent copy and execute
    }
}
```


Without this hardware feature, using streams for overlap is impossible. Fortunately, it is standard on nearly all modern GPUs.

#### 2.2 Pipelining with Multiple Streams

To achieve overlap, we need to break our problem into smaller, independent pieces and process them in a pipeline. The strategy is as follows:

1. Divide Data: Split your large input and output data structures (e.g., arrays) into smaller segments or chunks.
2. Create Streams: Create two or more non-default streams.
3. Process in a Loop: Loop through the data segments, assigning each segment's workflow (Copy-Execute-Copy) to a different stream.

This creates a pipeline with three distinct phases:

* Fill: In the beginning, the first few stages of the pipeline are being filled. For example, Stream 1 is copying data while the GPU is otherwise idle. Then, Stream 2 starts copying while Stream 1 starts computing.
* Steady State: The pipeline is full. This is the ideal state where data is being copied in for chunk N+1, the kernel is executing on chunk N, and results are being copied out for chunk N-1, all at the same time.
* Drain: As the loop finishes, the final chunks work their way through the now-emptying pipeline.

The effectiveness of this technique depends on computational intensity. If the kernel is too fast compared to the data transfer time, the pipeline will stall, waiting for data. Conversely, if the data transfers are much faster than the kernel, the benefit of overlap is minimal. We will analyze this trade-off mathematically in the next chapter.

#### 2.3 Implementing a Multi-Stream Workflow

Let's see how to implement this in code. First, we need to know the relevant API calls.

1. Creating a Stream: A stream is represented by the cudaStream_t type. You create one with cudaStreamCreate().

```c++
cudaStream_t my_stream;
cudaStreamCreate(&my_stream);
```


2. Asynchronous Memory Copies: To use streams, you must use the asynchronous version of cudaMemcpy, which is cudaMemcpyAsync(). It takes an additional argument: the stream ID.

```c++
cudaMemcpyAsync(dst, src, count, kind, stream);
```


Important Note: Asynchronous memory transfers require the host memory to be page-locked (or "pinned"). You must allocate host memory using cudaMallocHost() or cudaHostAlloc() instead of the standard malloc(). This prevents the operating system from moving the memory while the GPU is trying to access it via Direct Memory Access (DMA).

3. Launching a Kernel in a Stream: The kernel launch configuration is extended to include the stream ID as a fourth optional parameter (after grid dimensions, block dimensions, and shared memory size).

```c++
kernel_name<<<grid, block, shared_mem_size, stream>>>(args...);
```


##### Example: Multi-Stream SAXPY (Version 1)

Below is a conceptual code snippet showing how to process a large SAXPY operation by breaking it into two segments and processing them in two different streams.

```c++
// Create two streams
cudaStream_t stream0, stream1;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);

// Allocate device memory for each stream's segment
float *d_A0, *d_B0, *d_C0; // for stream 0
float *d_A1, *d_B1, *d_C1; // for stream 1
// ... calls to cudaMalloc for each pointer ...

// Assume h_A, h_B, h_C are pointers to large, page-locked host arrays
int segSize = n / 2; // For simplicity, let's just do two segments

// --- Issue commands for the first segment to stream 0 ---
// Copy inputs for segment 0
cudaMemcpyAsync(d_A0, h_A, segSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
cudaMemcpyAsync(d_B0, h_B, segSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
// Launch kernel for segment 0
saxpy<<<segSize/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, ...);
// Copy output for segment 0
cudaMemcpyAsync(h_C, d_C0, segSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);

// --- Issue commands for the second segment to stream 1 ---
// Copy inputs for segment 1
cudaMemcpyAsync(d_A1, h_A + segSize, segSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_B1, h_B + segSize, segSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
// Launch kernel for segment 1
saxpy<<<segSize/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, ...);
// Copy output for segment 1
cudaMemcpyAsync(h_C + segSize, d_C1, segSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);

// Don't forget to synchronize before using the results on the CPU!
cudaDeviceSynchronize();
```


The intent here is that while the saxpy kernel for stream0 is running, the cudaMemcpyAsync operations for stream1 can also be running, achieving our desired overlap. However, due to the architecture of older GPUs, this might not happen as we expect.

#### 2.4 Architecture Matters: Fermi vs. Kepler and Newer

The way a GPU executes commands from streams depends heavily on its architecture.

* Fermi Architecture (and older): These GPUs have a single work queue for the copy engine and a single work queue for the compute engine. Even though we issued commands to two different software streams, they are all fed into the same two hardware queues. In our "Version 1" code, the device driver would schedule the operations like this:
  1. Copy Queue: H2D(A0), H2D(B0), D2H(C0), H2D(A1), H2D(B1), D2H(C1)
  2. Execute Queue: Kernel(0), Kernel(1)
* The problem is that the D2H(C0) operation (copying the result for stream 0) is placed in the copy queue before the input copies for stream 1 (H2D(A1) and H2D(B1)). Since operations within a queue are serial, the input copies for the second segment cannot begin until the output copy for the first segment is complete, destroying our desired overlap.
* Kepler Architecture (and newer): These GPUs introduced a feature called "Hyper-Q," which provides multiple hardware work queues (e.g., 32 queues each for copy and execute). This allows different software streams to map to different hardware queues, enabling true concurrent execution. On a Kepler or newer GPU, the "Version 1" code would likely achieve the desired overlap.

##### A More Robust Approach (Version 2)

To ensure overlap even on older hardware, we can reorder the commands we issue from the host. The goal is to issue all independent operations first to allow the hardware scheduler more flexibility.

```c++
// ... stream creation and memory allocation as before ...

// --- Issue all H2D copy commands first ---
cudaMemcpyAsync(d_A0, h_A, segSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
cudaMemcpyAsync(d_B0, h_B, segSize * sizeof(float), cudaMemcpyHostToDevice, stream0);

cudaMemcpyAsync(d_A1, h_A + segSize, segSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_B1, h_B + segSize, segSize * sizeof(float), cudaMemcpyHostToDevice, stream1);

// --- Issue all kernel launches ---
saxpy<<<segSize/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, ...);
saxpy<<<segSize/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, ...);

// --- Issue all D2H copy commands last ---
cudaMemcpyAsync(h_C, d_C0, segSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);
cudaMemcpyAsync(h_C + segSize, d_C1, segSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);
```


By issuing all input copies first, followed by all kernel launches, we maximize the opportunity for the GPU to overlap the execution of Kernel(0) with the input copies for stream 1 (H2D(A1) and H2D(B1)). This reordering makes the code more robust across different GPU architectures.

#### 2.5 Common Pitfalls: Implicit Synchronization

When using streams, you must be careful to avoid operations that implicitly synchronize the entire device, as they will destroy any overlap you've worked to create. These operations act like a call to cudaDeviceSynchronize(), forcing all previously issued work to complete.

Be cautious with the following types of operations:

* **Page-locked host memory allocation:** `cudaMallocHost()` or `cudaHostAlloc()`.
* **Device memory allocation/deallocation:** `cudaMalloc()` or `cudaFree()`.
* **Synchronous memory operations:** Any memory function that does not have the `Async` suffix, such as `cudaMemcpy()` or `cudaMemset()`.
* **L1/Shared Memory configuration changes:** `cudaDeviceSetCacheConfig()`.

Always use the Async versions of functions where available and perform all necessary memory allocations before you begin your pipelined stream loop.


---


### Chapter 3: Advanced Topics and Modern Alternatives

While CUDA Streams are a powerful tool for performance optimization, they represent a manual approach that increases programmer complexity. In this chapter, we will analyze when using streams is mathematically justified. We will then explore more modern CUDA features—Unified Memory and Peer-to-Peer Access—that aim to simplify host-device memory management, sometimes at the cost of performance, but always with the benefit of simpler code.

#### 3.1 Is Streaming Always Worth It? An Analysis of Arithmetic Intensity

The goal of streaming is to hide the time it takes to transfer data over the PCIe bus ($t_{\text{PCIe}}$) by overlapping it with computation time ($t_{\text{COMP}}$). This strategy is only effective if the computation is long enough to mask the transfer. We can formalize this with the concept of Arithmetic Intensity.

**Arithmetic Intensity** ($r$) is defined as the ratio of floating-point operations (FLOPs) performed to the number of bytes transferred to or from memory:

$$r = \frac{\text{FLOPs}}{\text{Byte}}$$

Let's derive the condition required to successfully hide the PCIe latency. We'll make the following assumptions:

| Variable | Description | Units |
| --- | --- | --- |
| $N$ | The number of float elements in our segment. | elements |
| $b$ | The bandwidth of the PCIe bus. | Bytes/s |
| $c$ | The peak compute performance of the GPU. | FLOPs/s |
| $r$ | The arithmetic intensity of our kernel. | FLOPs/Byte |

The total amount of data to be transferred for a segment of $N$ floats is $4N$ bytes (since a float is 4 bytes). The time required for this transfer is:

$$t_{\text{PCIe}} = \frac{4N}{b}$$

The total number of floating-point operations performed on this segment is the arithmetic intensity multiplied by the number of bytes, which is $r \cdot (4N)$. The time required for this computation is:

$$t_{\text{COMP}} = \frac{\text{Total FLOPs}}{\text{Performance}} = \frac{r \cdot (4N)}{c}$$

To completely hide the data transfer latency, the computation time must be greater than or equal to the transfer time: $t_{\text{COMP}} \ge t_{\text{PCIe}}$.

Substituting our expressions for time:

$$\frac{r \cdot (4N)}{c} \ge \frac{4N}{b}$$

We can cancel out the $4N$ term from both sides and rearrange the inequality to solve for $r$:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Streaming Effectiveness Condition)</span></p>

For streaming to effectively hide PCIe latency, the arithmetic intensity $r$ of the kernel must satisfy:

$$r \ge \frac{c}{b}$$

where $c$ is the GPU's peak compute performance (FLOPs/s) and $b$ is the PCIe bandwidth (Bytes/s). If the kernel performs too few calculations per byte of data, it will finish long before the next chunk arrives, and the GPU will sit idle.

</div>

#### 3.2 Simplifying Memory Management: Unified Virtual Addressing (UVA)

The complexity of manually managing memory buffers and cudaMemcpyAsync calls led NVIDIA to develop simpler memory models. The first step in this direction was Unified Virtual Addressing (UVA).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Unified Virtual Addressing)</span></p>

With **UVA**, the CPU and all GPUs in a system share a single virtual address space. A pointer, regardless of whether it points to host or device memory, has a unique address. However, while the GPU can access host memory directly, doing so is extremely slow as data must still travel over the PCIe bus. The programmer is still responsible for explicit locality optimizations. UVA primarily simplifies pointer management in multi-GPU applications.

</div>

#### 3.3 The Future is Automatic: Unified Memory (UM)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Unified Memory)</span></p>

**Unified Memory (UM)** creates a pool of managed memory accessible to both the CPU and the GPU through a single pointer. The CUDA runtime automatically migrates data between host and device on demand. When the CPU accesses managed data, it is ensured to be in host memory. When a GPU kernel accesses it, it is automatically paged to device memory.

</div>

Here is how you would write a SAXPY application using Unified Memory:

```c++
// Allocate managed memory accessible by both host and device
float *X, *Y;
cudaMallocManaged(&X, N * sizeof(float));
cudaMallocManaged(&Y, N * sizeof(float));

// Initialize data on the CPU
for (int i = 0; i < N; ++i) {
    X[i] = ...;
    Y[i] = ...;
}

// Launch kernel. The CUDA runtime will automatically move X and Y
// to the device if they are not already there.
saxpy<<<numBlocks, blockSize>>>(X, Y, ...);

// Wait for the kernel to finish before accessing data on the CPU
cudaDeviceSynchronize();

// Use the results on the CPU. The runtime will move the data back.
use_data(Y);

// Free the managed memory
cudaFree(X);
cudaFree(Y);
```


Notice the complete absence of cudaMemcpy calls! This dramatically simplifies the code.

##### Performance of Unified Memory

While UM is convenient, the automated data migration has overhead. The source context provides bar charts comparing the performance of matrix multiplication for different memory strategies on a Pascal-generation GPU. The strategies are:

* malloc: Traditional C-style host allocation with explicit cudaMemcpy.
* pinned: Page-locked host allocation (cudaMallocHost) with explicit cudaMemcpy.
* UM: Unified Memory with on-demand migration.
* UM prefetch: Unified Memory where the programmer provides hints to the runtime about where data will be needed next, allowing it to pre-migrate the data.

The charts illustrate the breakdown of time spent in host2device copy, kernel execution, and device2host copy for different matrix sizes:

* 1k x 1k Matrix: For smaller problem sizes, the overhead of UM's page faulting mechanism is significant, making it slower than traditional pinned memory with manual copies.
* 4k x 4k Matrix: As the problem size grows, the kernel execution time becomes more dominant. The convenience of UM starts to become more competitive, though still slightly slower than the manual approach.
* 8k x 8k Matrix: For very large problems, the kernel time dwarfs the transfer and overhead time. Here, the performance of UM is very close to that of manual memory management, and using prefetching hints can close the gap even further.

The conclusion is that Unified Memory offers a promising trade-off between programmer productivity and performance, especially for applications where development speed is critical or memory access patterns are complex.

#### 3.4 Direct Peer-to-Peer Access for Multi-GPU Systems

In systems with multiple GPUs connected by a high-speed interconnect like NVLink, it's possible for one GPU to directly access the memory of another without involving the host CPU. This is known as Peer-to-Peer (P2P) Access.

You must first enable this capability between the GPUs:

```c++
// Check if GPU 0 can access GPU 1's memory
int can_access_peer;
cudaDeviceCanAccessPeer(&can_access_peer, gpuid_0, gpuid_1);

if (can_access_peer) {
    // Enable access from GPU 0 to GPU 1
    cudaSetDevice(gpuid_0);
    cudaDeviceEnablePeerAccess(gpuid_1, 0);

    // Enable access from GPU 1 to GPU 0
    cudaSetDevice(gpuid_1);
    cudaDeviceEnablePeerAccess(gpuid_0, 0);
}
```


Once P2P access is enabled, you can perform a cudaMemcpy directly between the buffers of two different GPUs by specifying cudaMemcpyDefault. Even more powerfully, a kernel running on gpu0 can directly read from or write to a pointer that resides in gpu1's memory, as if it were its own.

```c++
// Example: Copy from GPU 1 to GPU 0
cudaMemcpy(gpu0_buf, gpu1_buf, buf_size, cudaMemcpyDefault);

// Example: Kernel on GPU 0 directly accesses GPU 1 memory
// (inside a kernel launched on GPU 0)
gpu0_buf[idx] = gpu1_buf[idx];
```


This capability is essential for scaling applications across multiple GPUs efficiently.


---


### Chapter 4: Summary and Key Takeaways

This lecture has taken us beyond single-kernel optimization and into the critical domain of system-level performance. We've explored the challenges posed by the host-device communication bottleneck and learned powerful techniques to mitigate it.

#### 4.1 Recap: Streams for Latency Hiding

* **The Problem:** The PCIe bus connecting the host and device is much slower than the GPU's internal memory and compute capabilities, creating a data transfer bottleneck.
* **The Solution:** We can introduce task-level parallelism using CUDA Streams to overlap communication (data transfers) with computation (kernel execution).
* **The "How":** By dividing work into independent segments and placing the operations for each segment into a different stream, we can create a pipeline. This requires using asynchronous functions like `cudaMemcpyAsync` and launching kernels into specific streams.
* **The Catch:** This technique adds complexity for the programmer. For older GPUs (Fermi-class), careful ordering of API calls is required to achieve overlap due to limited hardware queues.
* **The Prerequisite:** Overlapping is only effective if the application has sufficient computational intensity. The ratio of computations to data bytes ($r$) must be high enough to successfully hide the data transfer time ($r \ge c/b$).

#### 4.2 Recap: Unified Memory as an Alternative

* **The Goal:** Simplify the programming model by reducing the burden of manual memory management.
* **Unified Virtual Addressing (UVA):** Provides a single address space for the entire system, but still requires the programmer to manually move data for good performance. Accessing host memory from the GPU is possible but can be extremely slow.
* **Unified Memory (UM):** Automates the movement of data between host and device based on where it is being accessed (page migration). This eliminates the need for explicit `cudaMemcpy` calls, resulting in simpler, more maintainable code.
* **The Trade-off:** The convenience of UM comes with some performance overhead from the automated page migration system. However, for large problems and on modern hardware, its performance can be very competitive with manual methods, making it a compelling alternative.


## Stencil Computations

### Introduction to Stencil Computations

In the world of high-performance computing, many problems involve updating values in a large, organized grid. Whether we are simulating weather patterns, analyzing medical images, or calculating how heat moves through a metal part, we often use a technique called **Stencil Computation**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stencil)</span></p>

A **stencil** is a geometric pattern used to update elements in a regular array (a grid). The value of a center element is updated based on the values of its neighbors. This specific "neighborhood" of cells used for the calculation is the stencil.

* **Iterative Kernel:** A kernel is a small function or program designed to run in parallel. In stencil computations, this kernel is iterative, meaning it runs repeatedly over the array to evolve the data over time.
* **Example (6-point stencil):** A 3D stencil where a central cube is updated using values from six direct neighbors: top, bottom, left, right, front, and back.

</div>

**Real-World Applications:**

1. **Image Processing:** Used in blob analysis, which helps computers identify and categorize distinct shapes within an image.
2. **Partial Differential Equations (PDEs):** Mathematical equations that describe how physical quantities change over space and time.
3. **Computational Fluid Dynamics (CFD):** Engineers use stencils to simulate the flow of liquids and gases, essential for designing airplanes, cars, and turbines.

While stencils are used for regular grids, more complex, irregular grids (like those used to model complex mechanical parts) often require Finite Element Methods (FEM).

### The Mathematics of PDEs and Finite Differences

Most stencil computations aim to approximate the solutions to Partial Differential Equations (PDEs).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial Differential Equation)</span></p>

A **Partial Differential Equation** is a function of multiple independent variables (like position $x$ and time $t$). These equations are used in physics, biology, economics, and chemistry to describe phenomena like heat transfer, Newtonian gravity, seismic wave propagation, and electrostatics.

</div>

#### Finite-Difference Methods (FDM)

Since computers cannot solve complex calculus equations exactly, we use **Finite-Difference Methods**. These methods approximate a derivative (the rate of change) by looking at the difference between two points separated by a small spacing $k$.

Using Taylor's polynomial, we can approximate the derivative of a function $f(x)$:

$$
f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2 + \dots
$$

In practice, we use three main types of "differences" to approximate these changes:

| Method | Formula Approximation | Error Level |
| --- | --- | --- |
| Forward Difference | $f_x \approx \frac{f(x + k) - f(x)}{k}$ | $O(k)$ |
| Backward Difference | $f_x \approx \frac{f(x) - f(x - k)}{k}$ | $O(k)$ |
| Central Difference | $f_x \approx \frac{f(x + k) - f(x - k)}{2k}$ | $O(k^2)$ |

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Error Order)</span></p>

$O(k)$ represents the "order of error." An $O(k^2)$ method is generally much more accurate than an $O(k)$ method because the error shrinks faster as the spacing $k$ gets smaller.

</div>

#### Discretizing the Grid

To solve these on a GPU, we turn continuous space into a grid. We define points $x_i = ih$ and $t_j = jk$, where $h$ is the space step and $k$ is the time step. This allows us to represent a physical object as a series of indexed points in a computer's memory.

### Solving the Heat Equation: Explicit vs. Implicit Methods

The Heat Equation describes how thermal conductivity ($c$) transports heat through a material over time. It is a parabolic equation, and there are two primary ways to solve it using stencils.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Explicit Method)</span></p>

In the **Explicit Method**, we calculate the state of a point at the next time step ($j+1$) using only the values from the current time step ($j$):

$$
u_{i,j+1} = r\, u_{i-1,j} + (1 - 2r)\,u_{i,j} + r\, u_{i+1,j}
$$

where $r = \frac{ck}{h^2}$.

* **Pros:** Very simple to calculate. Each point can be computed independently.
* **Cons:** Unstable. Requires impractically small time steps ($k < \frac{h^2}{2c}$).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Implicit Method)</span></p>

The **Implicit Method** calculates the next time step by solving a system of equations where multiple unknown points at $t+k$ are related to each other.

* **Pros:** Numerically stable. Much larger time steps are possible.
* **Cons:** Computationally intensive, requiring solving a linear system.

</div>

#### The Crank-Nicholson Method (CNM)

This is a hybrid approach. It averages the explicit and implicit methods to achieve higher accuracy ($O(h^2 + k^2)$). For a simulation requiring 4 digits of accuracy, an implicit method might need 1 million points, while the Crank-Nicholson Method would only need 560 points to achieve the same result.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(GPU Connection)</span></p>

Because implicit methods and hybrid methods like CNM are computationally heavy but involve massive amounts of similar calculations, they are perfectly suited for the parallel architecture of a GPU.

</div>

### Optimizing Stencil Codes for GPU Architecture

Stencil codes are typically **memory-bound**. This means the GPU spends more time waiting for data to arrive from memory than it does actually performing math.

#### Domain Decomposition and Halos

To parallelize a stencil, we divide the data grid into chunks and assign each chunk to a thread block.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Halo)</span></p>

Because each point needs its neighbor's data, thread blocks must store an overlapping area of data from neighboring blocks. This overlap is called the **halo**.

</div>

#### Strategies for Memory Optimization

1. **1D vs 2D Partitioning:** 2D partitioning is common but can lead to "poorly aligned" memory access for vertical halos. 1D partitioning (strips) can sometimes be more efficient for communication.
2. **Shared Memory (Scratchpad):** Shared memory is a fast, small memory area on the GPU. Instead of reading from the slow global memory every time, threads can load a "tile" of the grid into shared memory, use it, and then move on.
3. **Marching Planes:** When dealing with 3D data, we don't have enough shared memory to store everything. Instead, we keep only three planes (top, middle, bottom) in shared memory and "march" through the 3D volume, cycling the buffers as we go.

### Directive-Based Programming (OpenACC)

Writing raw CUDA code can be difficult for beginners. **OpenACC** is a directive-based standard that allows you to parallelize code by adding simple "pragmas" (hints to the compiler) to standard C, C++, or Fortran.

#### The Execution Model

OpenACC uses a host-directed model: the CPU (host) manages the program and "offloads" compute-heavy loops to the GPU (accelerator).

Parallelism in OpenACC is organized into three levels:

1. **Gang:** The coarsest level (similar to a CUDA thread block).
2. **Worker:** A middle level (similar to a CUDA warp).
3. **Vector:** The finest level (similar to individual threads or SIMD operations).

#### Basic Syntax and Directives

To tell the compiler to run a loop on the GPU, you use:

```c
#pragma acc kernels
{
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
```

**Key Directives and Clauses:**

* **`parallel` vs `kernels`:** The `parallel` directive is more explicit—the programmer is responsible for identifying parallelism. The `kernels` directive gives the compiler more freedom to find and optimize parallelism on its own.
* **Data Management:** Moving data between the CPU and GPU is expensive. The `#pragma acc data` directive allows you to keep data on the GPU across multiple loops to save time.

| Clause | Description |
| --- | --- |
| `copy` | Moves data to the device at the start and back to the host at the end. |
| `copyin` | Only moves data from host to device. |
| `copyout` | Only moves data from device to host. |
| `present` | Tells the compiler the data is already on the GPU. |

#### Performance Optimization in OpenACC

A common problem is **pointer aliasing**, where the compiler isn't sure if two memory pointers overlap. To fix this and allow parallelization, use the `restrict` keyword:

```c
float* restrict x; // Tells the compiler this pointer is unique
```

You can also use the `async` clause to allow the GPU to work on one task while the CPU continues with another, enabling overlapping execution.

### Practical Application: Connected Component Labeling

A great example of stencil computation in image processing is **Connected Component Labeling (CCL)**. The goal is to identify and color separate "blobs" or shapes in a black-and-white image.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Connected Component Labeling)</span></p>

1. **Thresholding:** Convert a color image to black and white (bitmapping).
2. **Initial Labeling:** Every pixel is given a unique label.
3. **Merging (Stencil Step):** Each pixel looks at its neighbors (using a 4-way or 8-way stencil). If a neighbor has a lower label value, the pixel updates its own label to match.
4. **Iteration:** This merging step repeats until no labels change across the entire image.

</div>

**Optimization Techniques for CCL:**

* **Marching Order:** Reading data in "Row-Major" order (across rows) provides better cache hits, while "Column-Major" order can sometimes offer better coalescing (combining multiple memory requests into one), which is vital for GPU performance.
* **Wavefront Merging:** This optimization uses shared memory to store previous results, reducing the number of times the GPU has to reach out to slow global memory. Threads process the image in "waves," ensuring that as many pixels as possible are updated in a single pass.

Stencil computations represent a perfect marriage between physical mathematics and parallel hardware. While memory limits often present a challenge, techniques like domain decomposition and productivity tools like OpenACC make it possible to simulate complex physical systems with incredible speed and accuracy.



## GPU Memory Model: Coherence and Consistency

### A Review of GPU Architecture

To understand how memory works in a GPU, we must first recall what a GPU actually is. Depending on whether you are looking at it from a programmer’s perspective (software) or an engineer’s perspective (hardware), the GPU takes on two different roles.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Software View: Many-Core Scalar Architecture)</span></p>

From a software perspective, the GPU is a programmable **many-core scalar architecture**. It is designed to handle a massive number of scalar threads. The GPU manages a huge number of these workers to exploit **parallel slackness**, which means having so many threads available that the hardware can always find work to do, even if some threads are waiting for data. This software model is known as **SIMT** (Single Instruction, Multiple Threads) and is a near-perfect implementation of the **BSP** (Bulk Synchronous Parallel) model.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hardware View: Multi-Core Vector Architecture)</span></p>

Under the hood, the GPU is actually a programmable **multi-core vector architecture**. While the software thinks it is dealing with individual scalar threads, the hardware actually operates using **SIMD** (Single Instruction, Multiple Data). The hardware creates an "illusion" of scalar threads by packing them into compound units. In essence, a GPU is a vector architecture that cleverly hides its vector units from the programmer.

</div>

### Foundations of Shared Memory

In parallel computing, we often use **Shared Memory Multiprocessors**. This describes a system where multiple execution units (processors or cores) share a single address space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Process)</span></p>

A **process** is defined as a single virtual address space that contains one or more threads of control. By definition, multiple threads within a process share the same address space.

</div>

#### Mapping Virtual to Physical Addresses

Computers use **Virtual Addresses (VA)** to manage memory. However, the actual data lives at a **Physical Address (PA)** in the hardware.

* **Shared Segments:** Portions of the address space can be shared such that multiple virtual addresses map to a single physical address.
* **Structured Address Space:** Typically, the virtual address space is organized into private segments (accessible only by one thread) and shared segments (accessible by all).

#### Communication and Synchronization

In a shared memory system, threads communicate by writing to and reading from shared addresses. For example, if Thread 0 performs a store operation to a shared memory location, that change should eventually be visible when Thread 1 performs a load operation from that same location. This reliance on memory operations—including **atomic operations** (operations that happen completely or not at all, without interruption)—is the basis for synchronization.

### Designing Communication Abstractions

A communication abstraction acts as a contract between the hardware and the software, much like an ISA (Instruction Set Architecture). It defines how data is moved and managed. There are five key pillars to this design:

1. **Naming:** Determines what data can be identified. In shared memory, threads "name" locations in their registers and virtual address spaces (code, stack, heap). In message passing, naming involves local addresses and process identifiers.
2. **Operations:** The actions performed on named data.
   * *Shared Memory:* Uses simple load and store instructions, plus atomic read-modify-write operations.
   * *Message Passing:* Uses send and receive operations, which are generally more complex than simple memory loads.
3. **Ordering:** Defines the sequence in which operations appear to happen. For a single thread, we follow sequential program order (top-to-bottom). In parallel systems, determining the order is much harder because threads operate independently.
4. **Communication/Replication:** Refers to how data is physically moved or copied across the system.
5. **Performance:** Focuses on the efficiency of data transfer, involving the application, the communication hardware, and the physical medium.

### The Coherence Problem

The goal of a memory system is to reduce latency (the time it takes to access data). We use caches—small, fast storage areas near the processor—to achieve this. However, caches introduce a major challenge in multi-core systems: **Coherence**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cache Coherence)</span></p>

**Coherence** is the protocol that ensures all processors see the same (most recent) value for a specific memory location. Imagine a system with two processors, P0 and P1, each with its own local cache. If both processors have a copy of variable $A$ (initially 0), and P0 changes $A$ to 1, P1 might still see 0 in its own local cache. Coherence protocols prevent this inconsistency.

</div>

#### Cache Write Policies

How a cache handles updates affects coherence:

* **Write-back (WB):** Updates are only made to the local cache and written to main memory later. This creates a significant coherence problem because other processors won’t see the change until the "write-back" occurs.
* **Write-through (WT):** Every update to the cache is immediately written to main memory. This makes coherence easier to manage but can be slower due to constant memory traffic.

#### Scalable Coherence Protocols

In older or smaller systems, we used a **Shared Bus**. A bus is like a single hallway; only one person can talk at a time. This is not scalable (it doesn’t work well as you add more cores).

* **Snooping:** Processors "snoop" on the bus to see if others are changing data they have cached. However, most "snoops" result in no action, wasting bandwidth.
* **Directory Protocol:** Instead of broadcasting every change to everyone, the system maintains a directory. It only sends updates to the specific processors that are known to hold a copy of that specific cache line.

### Understanding Memory Consistency

While coherence ensures that everyone eventually sees the same value for a single address, **consistency** deals with the ordering of operations across *different* addresses.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coherence vs. Consistency)</span></p>

* **Coherence:** Focuses on the "last write" to a *single address*. It is usually invisible to the software.
* **Consistency:** Defines constraints on the order in which memory operations to *different locations* become visible. It is a contract between the programmer and the architecture and is visible to the software.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Store Buffering)</span></p>

Consider two threads:

* Thread 0: Sets `a = 1`, then checks if `b == 0`.
* Thread 1: Sets `b = 1`, then checks if `a == 0`.

In a "perfect" world, it should be impossible for both `if` statements to be true. However, in reality, both can be true if the hardware uses write buffering (delaying the actual store to memory).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Producer-Consumer)</span></p>

* Thread 0 (Producer): Sets `a = 1`, then sets `flag = 1`.
* Thread 1 (Consumer): Waits for `flag == 1`, then prints `a`.

In many modern systems, Thread 1 might print 0 instead of 1. This happens because the system might reorder the operations, making the `flag` update visible before the data `a` update is visible.

</div>

### Relaxed Consistency and Performance

We relax consistency (allow reordering) for one primary reason: **Performance**. Maintaining a strict global order is incredibly expensive and prevents hardware optimizations like:

* **Out-of-Order (OOO) execution:** Processors doing work out of sequence to stay busy.
* **Store Buffers:** Temporarily holding writes to avoid waiting for slow memory.

| Model | Description | Impact on Optimization |
| --- | --- | --- |
| Strict Consistency | Global order required. | Disaster! No OOO allowed. |
| Sequential Consistency (SC) | Operations appear in a sequential order that respects program order. | Very restrictive; limits most optimizations. |
| Processor Consistency | Reorders loads across stores; allows FIFO store buffers. | Common in x86 architectures. |
| Relaxed Consistency | Allows unordered, coalescing store buffers. | High performance; used in GPUs. |
| Data-Race-Free (DRF) | Programmer must mark races using strong operations. | Found in C++ and Java. |

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sequential Consistency)</span></p>

A system is **Sequentially Consistent** if the result of execution is the same as if all operations were executed in some sequential order, and the operations of each individual processor appear in the sequence in the order specified by its program.

* **Requirement:** A processor must wait for a store to complete before issuing the next operation.
* **Problem:** This prevents "latency hiding"—the ability to overlap memory access with other work.

</div>

### GPU Coherence and Consistency Models

GPUs use a **relaxed memory consistency** model because they are built to prioritize throughput and tolerate high latency.

#### The Data-Race-Free (DRF) Model

The GPU consistency model is **Data-Race-Free (DRF)**, similar to the models used in C++ or Java. It relies on **Release-Acquire** semantics:

* **Release Store:** A "release" operation ensures that all memory writes performed by the thread before the release become visible to other threads.
* **Acquire Load:** An "acquire" operation ensures that after observing the load, the thread will also see everything the releasing thread did before its release.

#### Scoped Memory

In CUDA, you can define the scope of consistency:

* `thread`: Only within the single thread.
* `cta`: Within the thread block.
* `device`: Across the entire GPU.
* `system`: Across the GPU and CPU.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Release-Acquire Sequence)</span></p>

Below is an example of how a producer-consumer relationship is handled using CUDA atomics to ensure consistency.

```c++
// Thread 0 on Processor 0 (Producer)
x = 42;
// Create a reference to a flag in device scope
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
// Store with ‘release’ semantics to ensure x=42 is visible first
flag.store(1, memory_order_release);

// Thread 1 on Processor 1 (Consumer)
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
// Load with ‘acquire’ semantics to ensure we see the producer’s writes
while(flag.load(memory_order_acquire) != 1);
// Because of Release-Acquire, this assertion is guaranteed to pass
assert(x == 42);
```

1. Thread 0 writes data to `x`.
2. Thread 0 then performs a store on a flag using `memory_order_release`. This acts as a "fence," pushing the write of `x` out to memory before the flag is updated.
3. Thread 1 loops (polls) the flag using `memory_order_acquire`.
4. Once Thread 1 sees the flag change to 1, the `memory_order_acquire` ensures it also sees the most recent value of `x`.

</div>

#### GPU Memory Hierarchy and Coherence

The GPU hierarchy is typically "flat" compared to CPUs:

* **L1 Cache:** Exclusive to a Streaming Multiprocessor (SM). It is private and only provides consistency guarantees for the start and end of a thread block.
* **L2 Cache:** Shared across the whole GPU.
* **Write-Through Policy:** GPUs often use a Write-Through policy for L1 caches to make writes globally visible to the L2 cache immediately.
* **Software-Controlled Coherence:** Instead of expensive hardware protocols like snooping, GPUs often use "software-controlled" coherence. This involves invalidating caches at kernel completion boundaries. Because GPUs can tolerate high latency, they don’t need the complex, non-scalable hardware coherence found in CPUs.


## Deep Dive into the SIMT Execution Model

To understand how a GPU processes data, we must look past the code we write and see how the hardware actually executes it. GPU computing relies on a model called **SIMT** (Single Instruction, Multiple Threads).

### The Illusion of Independence

As a programmer, CUDA allows you to see the system as thousands of independent scalar threads. You write code as if a single thread is performing a calculation, and the GPU simply "multiplies" that effort. However, this is a clever illusion maintained by the hardware.

In reality, the GPU hardware bundles these individual threads into groups of 32, known as a **Warp**. These warps run in "lockstep" on hardware that is fundamentally SIMD (Single Instruction, Multiple Data). Think of a warp as a single thread of vector instructions. If you tell one thread in a warp to add two numbers, all 32 threads in that warp perform an addition at the exact same moment.

### Handling Divergent Control Flow

A major challenge arises when threads within the same warp want to do different things—a concept called **Divergent Control Flow**.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Masked Execution)</span></p>

Imagine an `if-else` statement:

```c++
foo[] = {4, 8, 12, 16};
v = foo[tid.x];
if (v < 10)
    v = 0;  // Path C
else
    v = 10; // Path D
w = bar[tid.x] + v; // Path E
```

Some threads (where `v < 10`) want to execute Path C, while others want to execute Path D. Because the hardware runs in lockstep, it cannot do both simultaneously. Instead, the GPU uses **Masked Execution**:

1. It serializes the paths.
2. It first executes Path C while "masking out" (deactivating) the threads that need Path D.
3. It then executes Path D while masking out the threads that already finished Path C.
4. Finally, it re-converges all threads at Path E.

This leads to a loss of efficiency. If only one thread takes a specific branch, the other 31 threads sit idle, waiting for their turn.

</div>

### The SIMT Stack and Re-convergence

To manage this divergence, the hardware utilizes a **Per-warp stack** that stores Program Counters (PCs) and Active Masks.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SIMT Stack Components)</span></p>

* **Active Mask:** A bitmask where each bit represents whether a specific thread in the warp is "active" or "inactive" for the current instruction.
* **Re-convergence:** The hardware attempts to find the **Immediate Post-Dominator** (the point where the divergent paths meet again) to resume full parallel execution as quickly as possible.

</div>

### Inside the SIMT Core

The GPU is composed of several **SIMT Core Clusters**, each containing multiple **SIMT Cores** (often called Streaming Multiprocessors).

A SIMT Core is split into a **Front End** and a **Back End** (Datapath):

* **Front End:** Handles the "virtualization" of threads. It includes the Instruction Cache (I-Cache), Instruction Buffer (I-Buffer), and the Scoreboard (which tracks when operands are ready).
* **Back End (SIMD Datapath):** This is where the heavy lifting happens. It contains the Arithmetic Logic Units (ALUs), Register Files, and the Memory Subsystem (including Shared Memory, L1 Cache, and Texture/Constant Caches).

The core uses **Fine-grained Multithreading** to hide latency. While one warp is waiting for data to arrive from slow off-chip memory, the Warp Scheduler instantly switches to another warp that is ready to work. Because the Register Values for all active threads stay on the core, this switch has **zero overhead**.

## Flexible Synchronization with Cooperative Groups

Historically, CUDA provided the `__syncthreads()` function to synchronize threads within a block. While effective, it is "rigid"—it's all-or-nothing for the entire block. **Cooperative Groups (CG)** is a modern API that allows for much more flexible, fine-grained synchronization.

### Core Concepts of Thread Groups

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Thread Group)</span></p>

A **Thread Group** is a generic type in the CG API that can represent any subset of threads. This allows you to perform collective operations (like synchronization or data shuffling) on specific groups rather than the whole block.

Supported hierarchies include:

* **Warp-Level Groups:** Represented by `coalesced_group`, allowing threads within a warp to cooperate using Warp-level intrinsics like `shfl` (shuffle).
* **Block-Level Groups:** Represented by `thread_block`.
* **Grid Groups:** Allows for synchronization across the entire grid (all blocks). This requires a "Cooperative Launch" and hardware with Compute Capability $\geq 6.0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Block-Level Synchronization)</span></p>

Using Cooperative Groups improves code readability and provides a standardized way to handle synchronization.

```c++
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void blockSyncKernel(float* data) {
    // Get a handle to the current thread block group
    thread_block block = this_thread_block();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform initial parallel computation
    data[idx] = data[idx] * 2.0f;

    // Synchronize ONLY the threads in this specific block
    block.sync();

    // This code only runs once every thread in the block has reached the sync point
    data[idx] += 1.0f;
}
```

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Warp-Level Aggregated Atomics)</span></p>

A powerful feature of CG is the ability to create opportunistic groups of threads that happen to be executing together (coalesced).

```c++
__device__ int atomicAggInc(int *ptr) {
    // Create a group containing all currently active threads in the warp
    cg::coalesced_group g = cg::coalesced_threads();
    int prev;

    // Elect the first active thread (rank 0) to perform a single atomic operation
    if (g.thread_rank() == 0) {
        // Increment the global counter by the total number of threads in this group
        prev = atomicAdd(ptr, g.size());
    }

    // Broadcast the original value back to all threads in the group
    // Each thread then adds its own rank to get a unique index
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}
```

In this code, we reduce the pressure on global memory by combining multiple individual atomic increments into a single "bulk" increment performed by one leader thread.

</div>

## Accelerated Matrix Operations with Tensor Cores

As Deep Learning became dominant, NVIDIA introduced **Tensor Cores**. These are specialized hardware units designed specifically for Matrix-Multiply-Accumulate (MMA) operations.

### What is a Tensor Core?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tensor Core)</span></p>

Standard CUDA cores handle one calculation at a time. A **Tensor Core** operates on entire **Matrix Tiles** (e.g., $4 \times 4$ or $8 \times 8$ matrices). It performs a **Fused Multiply-Add (FMA)**, calculating $D = A \cdot B + C$.

The primary advantage is **Mixed-Precision Arithmetic:**

* **Input** ($A$ and $B$): Usually lower precision like FP16 (16-bit floating point).
* **Accumulation** ($C$ and $D$): Can be higher precision like FP32 to maintain accuracy.

By using Tensor Cores, a GPU can generate significantly more Multiply-Accumulate (MAC) operations per clock cycle than standard cores.

</div>

### The WMMA API

To use Tensor Cores in CUDA C++, you use the **WMMA (Warp Matrix Multiply Accumulate)** API. This API introduces **Fragments**, which are specialized objects that hold matrix tiles in the registers of a warp.

**Fragment Types:**

1. `matrix_a` and `matrix_b`: The input matrices.
2. `accumulator`: Holds the intermediate and final results.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Matrix Multiplication with WMMA)</span></p>

Below is a simplified example of how a warp processes a $16 \times 16 \times 16$ matrix operation.

```c++
// Define matrix tile dimensions supported by the hardware
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_example(half *a, half *b, float *c) {
    // Declare the fragments (placeholders for matrix tiles in registers)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize the accumulator fragment with zeros
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the K-dimension to perform the multiplication
    for (int i = 0; i < K; i += WMMA_K) {
        // Load tiles from memory into fragments
        wmma::load_matrix_sync(a_frag, a + (warpM * WMMA_M) + i * lda, lda);
        wmma::load_matrix_sync(b_frag, b + i + (warpN * WMMA_N) * ldb, ldb);

        // Perform the specialized Matrix-Multiply-Accumulate operation
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store the final result fragment back to global memory
    wmma::store_matrix_sync(c + (warpM * WMMA_M) + (warpN * WMMA_N) * ldc,
                            acc_frag, ldc, wmma::mem_col_major);
}
```

</div>

### Evolution of Tensor Cores

NVIDIA has updated Tensor Core technology with every major architecture generation:

| Generation | Architecture | Key Additions |
| --- | --- | --- |
| Gen 1 | Volta | Introduced FP16/FP32 support |
| Gen 2 | Turing | Added integer formats (INT8, INT4) for faster inference |
| Gen 3 | Ampere | Added TF32 (Tensor Float 32) and BF16 (BFloat16) |
| Gen 4 | Hopper | Introduced FP8 support for massive HPC and AI scaling |
| Gen 5 | Blackwell | Added FP6 and FP4 support; introduced Tensor Memory (TMEM) |

## Performance Scaling and the Future of GPU Computing

As we move into the **Post-Dennard Scaling** era, simply shrinking transistors no longer provides "free" power and performance gains.

### The Power-Performance Limit

The performance of a modern processor is fundamentally limited by its power consumption. The relationship can be expressed as:

$$
\text{perf [ops/s]} = P\,[\text{W}] \cdot \eta\,[\text{ops/J}]
$$

where $P$ is the total Power in Watts and $\eta$ is the Energy Efficiency (operations per Joule).

Energy consumption is the sum of two parts:

1. **Computation Energy** ($\epsilon_{\text{op}}$): The energy cost of the math itself.
2. **Data Movement Energy** ($\epsilon_{\text{mem}}$): The energy cost of moving data from RAM to the processor. Moving data is often far more expensive than the calculation itself!

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Hardware Lottery)</span></p>

There is a concept known as the **Hardware Lottery**, suggesting that the success of certain AI models (like Deep Neural Networks) is partly because they happen to run very well on current GPU architectures. This creates a bias toward "standard" solutions.

</div>

### Emerging Alternatives

To overcome the power limits of CMOS (the current standard chip technology), researchers are exploring:

* **Analog Computations:** Using electrical or photonic (light-based) signals to perform math.
* **Resistive Memory:** Performing calculations directly inside the memory.
* **Specialized AI Chips:** Examples include GraphCore IPU, Cerebras, and SiPearl.

### Course Summary

Throughout this series, we have covered the foundational and advanced aspects of GPU computing:

* **CUDA Basics:** Thread hierarchies, shared memory, and barrier synchronization.
* **Architecture:** The SMX architecture, memory hierarchies, and warp scheduling.
* **Optimizations:** Bank conflict resolution, loop unrolling, and templating.
* **Advanced Features:** SIMT divergence management, Cooperative Groups for flexible syncing, and Tensor Cores for AI acceleration.
