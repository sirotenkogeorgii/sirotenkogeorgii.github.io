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

**Table of Contents**
- TOC
{:toc}

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

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/The-bulk-synchronous-parallel-computing-paradigm.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>The Bulk Synchronous Parallel Computing Paradigm.</figcaption>
</figure>

> GPUs hide long memory latencies by running many more independent threads ($v$) than they have execution lanes/pipelines ($p$). When some threads stall on memory, the hardware instantly swaps to other ready threads—so the lanes stay busy. That “excess” of runnable work over hardware lanes is the slackness $v ≫ p$.

## CUDA

### CUDA and GPU Overview

In a typical computer system, the CPU and GPU are distinct components with their own dedicated memory systems, connected via an I/O bridge (PCIe).

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/cpu-gpu-diagram.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>CPU + GPU System</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(GPU and CPU memory bandwidth)</span></p>

* The **GPU's memory bandwidth can be over 7x higher than the CPU's**, and its computational throughput an order of magnitude greater. 
* **CUDA** (Compute Unified Device Architecture) allows us to leverage this power for general-purpose computing tasks.

</div>

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Global Memory is used by SMs and host CPU)</span></p>

* All SMs on the GPU can access a large, shared **Global Memory** through a system of parallel data caches. 
* The **host CPU** initiates data transfers to and from this Global Memory to **set up computations** and **retrieve results**. 
* This load/store architecture, where data is explicitly moved between **different memory spaces**, is a central concept in GPU programming.

</div>

### CUDA Programming

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(CUDA)</span></p>

**CUDA** is an extension of the C programming language created by NVIDIA that exposes the GPU's parallel architecture directly to the developer. CUDA extends C with three main abstractions:
1. **hierarchy of threads**
2. **shared memory**
3. **barrier synchronization**

</div>

#### CUDA Programming Model

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(CUDA Programm)</span></p>

A **CUDA program** is a hybrid program consisting of two parts: a host part that runs on the CPU and a device part that runs on the GPU.

- The **CPU (host) part** is responsible for serial or low-parallelism tasks, such as setting up data, managing memory transfers, and launching computations on the GPU.
- The **GPU (device) part** handles massively parallel operations by executing kernels across many threads, SPMD-style.

</div>

### Two Types of Parallelism

CUDA programs exploit two complementary forms of parallelism:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Data-Level vs. Thread-Level Parallelism)</span></p>

- **Data-Level Parallelism (DLP):** Perform the *same* operation on *many* data elements simultaneously. Exploited by SIMD/vector units and GPU warps. Best for dense, regular computations (linear algebra, convolutions, elementwise ops). The main limitation is branch divergence and irregular memory access.
- **Thread-Level Parallelism (TLP):** Run multiple *independent threads* concurrently, each with its own instruction stream. Exploited by multi-core CPUs, SMT, and GPU thread blocks. Best for coarse-grained parallel work (serving requests, pipeline stages, independent simulations). The main limitation is coordination overhead (synchronization, contention, false sharing).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(GPU combines both)</span></p>

**GPU combines both:** TLP at the grid/block level and DLP-like execution at the warp level. A useful rule of thumb:

* If your program is “one loop over big arrays”: **DLP first** (vectorize/GPU-style).
* If your program is “many independent jobs or stages”: **TLP first** (threads/tasks).

</div>

#### Thread Hierarchy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Thread Hierarchy)</span></p>

The most fundamental concept in CUDA is the **thread hierarchy**. When you launch a computation on the GPU, you are launching a kernel function that is executed by a grid of threads. This hierarchy is organized into three levels:

- **Thread**: The smallest unit of execution. A single thread executes one instance of the kernel code.
- **Block**:
  - A group of threads. All blocks are equal size.
  - Threads within the same block can cooperate by sharing data through a fast, on-chip **shared memory** and can synchronize their execution using **barriers**.
  - Threads from different blocks cannot interact.
    - Exception: **global memory**.
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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(Defining a Kernel)</span></p>

A CUDA kernel is a function that runs on the device. You **define** it in your C/C++ code using the `__global__` declaration specifier.

A kernel is defined like a C function but with the `__global__` prefix, which indicates it can be called from the host and is executed on the device. A kernel function must have a **void return type**.

```c++
// Kernel function declaration
__global__ void MyKernel(float* data) {
    // Kernel code executed by each thread
}
```

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(Executing a Kernel)</span></p>

To **execute this function**, you call it from the host using a special `<<< ... >>>` syntax, known as the execution configuration. This tells the CUDA runtime how many threads to launch.

```c++
kernel_name<<<numBlocks, threadsPerBlock>>>(arguments);
```

- `numBlocks`: The **number of thread blocks** to launch in the **grid**.
- `threadsPerBlock`: The **number of threads** to launch in each **block**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Unique Thread Identification)</span></p>

Inside a kernel, each thread needs a way to identify itself so it can **work on a unique piece of data**. CUDA provides built-in variables for this purpose:

- `threadIdx`: A 3-component vector (x, y, z) that contains the **unique index of a thread within its block**.
- `blockIdx`: A 3-component vector (x, y, z) that contains the **unique index of a block within its grid**.
- `blockDim`: A 3-component vector (x, y, z) that contains the **dimensions of the block** (the number of threads in each dimension).
- `gridDim`: A 3-component vector (x, y, z) that contains the **dimensions of the grid** (the number of blocks in each dimension).

**Important: It is always a 3D vector, even if you logically use 2D, for example, for matrix multiplication.**

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Simple matrix addition)</span></p>

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

</div>

**Scaling Up with Grids**

The previous example only works if the matrix size `N` is small enough to fit within a single thread block (e.g., up to 1024 threads total). To handle larger problems, we must launch a grid of multiple blocks.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Using multiple blocks)</span></p>

When using multiple blocks, we need a way to calculate a global index for each thread across the entire grid. The standard formula for a 1D problem is:

```c++
int global_index = blockIdx.x * blockDim.x + threadIdx.x;
```

Let's break this down:

- `blockIdx.x * blockDim.x`: This calculates the starting index for the current block. For example, if each block has 256 threads (`blockDim.x`), then block 0 starts at index 0, block 1 starts at index 256, block 2 starts at index 512, and so on.
- `+ threadIdx.x`: This adds the thread's local index within the block to get its unique global index.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Revised `matAdd` kernel with any size matrix)</span></p>

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

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/grid_sketch.png' | relative_url }}" alt="Thread hierarchy 1" loading="lazy">
  <!-- <figcaption>Thread hierarchy 1</figcaption> -->
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Choosing Block and Grid Sizes)</span></p>

- **Threads per Block:** This should be a multiple of the warp size (typically 32, a concept we'll cover later). A common starting point is $128$, $256$, or $512$ threads per block. The ideal number balances resource usage with the ability to hide memory latency. 
  - **A range of $100-1000$ threads is often optimal.**
- **Blocks per Grid:** You should launch enough blocks to keep all the SMs on the GPU busy.
  - **A good heuristic is to launch at least twice as many blocks as there are SMs on your GPU.**
- **Number of blocks is limited:** $512 \times 512 \times 64 \to 1024 \times 1024 \times 64$, depending on the GPU generation.

</div>

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Memory Hierarchy: registers, shared memory, global memory)</span></p>

A diagram of the memory hierarchy shows that each Thread has its own private Registers. A group of threads in a Block shares a common Shared Memory. All blocks in the Grid can access the larger but slower Global Memory. The Host (CPU) also interacts with the device via this Global Memory.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Global Memory)</span></p>

- **Scope:** Accessible by all threads in the grid (R/W), as well as the host (CPU), communication between host and device.
- **Lifetime:** Persists for the lifetime of the application, beyond the execution of any single kernel.
- **Characteristics:** Large (often many gigabytes) but has high latency. This is the primary memory used for transferring data between the host and the device. Accesses to global memory are very sensitive to access patterns, and uncoalesced (scattered) accesses can severely degrade performance.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(CUDA global memory runtime API)</span></p>

You manage **global memory from the host using the CUDA runtime API**:

- `cudaMalloc(&d_ptr, size)`: Allocates `size` bytes of memory on the device and returns a pointer in `d_ptr`.
- `cudaFree(d_ptr)`: Frees device memory.
- `cudaMemcpy(dst, src, size, type)`: A blocking function to copy data between host and device. The `type` can be `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, or `cudaMemcpyDeviceToDevice`.
- `cudaMemcpyAsync(...)`: A non-blocking version for overlapping data transfers with computation.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Workflow for a CUDA program)</span></p>

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

</div>

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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(Variable Declaration Specifiers)</span></p>

These specifiers determine where a variable is stored and its scope.

| Location Specifier | Memory Space | Scope | Lifetime |
| --- | --- | --- | --- |
| `__device__ float var;` | Global Memory | All threads + Host API | Application |
| `__constant__ float var;` | Constant Memory | All threads + Host API | Application |
| `__shared__ float var;` | Shared Memory | All threads in block | Block |
| `texture <float> ref;` | Texture Memory | All threads + Host API | Application |

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(Function Declaration Specifiers)</span></p>

These specifiers determine where a function is executed and where it can be called from.

| Declaration Specifier | Executed On | Callable From |
| --- | --- | --- |
| `__device__ float Func()` | Device | Device |
| `__global__ void Kernel()` | Device | Host |
| `__host__ float Func()` | Host | Host |

- `__global__` defines a kernel, which can only be called from the host.
- `__device__` functions can only be called from other `__device__` or `__global__` functions.
- `__host__` is the default and can be combined with `__device__` to create a function that can be compiled for and called from both the CPU and GPU.
- **Device functions have several restrictions:** they do not support recursion, variable numbers of arguments, or non-static variable declarations inside the function.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(Type Specifiers)</span></p>

CUDA introduces several built-in types:

- **Vector types:** Such as `float2`, `float4`, `int2`, `int4`, which are simple structs containing 2 or 4 components. These are useful for representing coordinates or colors and can lead to more efficient memory access.
- **`dim3` type:** A struct based on `uint3` used for specifying dimensions for grids and blocks. Unspecified components are automatically initialized to 1.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(`__device__` can be combined with others)</span></p>

A key function related to shared memory is `__syncthreads()`. This intrinsic creates a barrier, forcing all threads in a block to wait until everyone has reached this point. It is essential for managing dependencies when using shared memory, ensuring that data is fully written before it is read by other threads.

</div>

#### Compilation and Execution

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(CUDA compiler)</span></p>

CUDA code is compiled using the `nvcc` (**NVIDIA C Compiler**) driver. `nvcc` is a powerful tool that separates the host and device code.

1. It processes the CUDA source code, separating host (`__host__`) code from device (`__global__`, `__device__`) code.
2. The host code is compiled by a standard C++ compiler like `g++` or `clang`.
3. The device code is compiled into PTX (Parallel Thread Execution) code.

Finally, `nvcc` links the compiled host and device code with the necessary CUDA libraries (`cudart`, `cuda`) to produce the final executable.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(PTX (Parallel Thread Execution))</span></p>

* **PTX** is a virtual machine and instruction set architecture (ISA) for GPUs. It acts as a stable assembly-like language for the GPU. 
* This is a key part of CUDA's forward compatibility. 
  * **When you compile your code**, `nvcc` can embed the PTX in your executable. 
  * **When you run your application**, the GPU driver performs a final Just-In-Time (JIT) compilation step, translating the PTX into the specific machine code for the target GPU (e.g., GF100, GK110, GP100) you are running on.

</div>

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Serial CPU Implementation of SAXPY)</span></p>

A standard C implementation of SAXPY uses a simple `for` loop.

```c++
// Kernel function (CPU)
void saxpy_serial(int n, float alpha, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = alpha * x[i] + y[i];
  }
}
```

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Parallel CUDA Implementation of SAXPY)</span></p>

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

This is a perfect demonstration of the SPMD model. Every thread runs this exact same code, but because each thread has a unique `i` calculated from `blockIdx.x` and 
`threadIdx.x`, each thread operates on a different element of the vectors `x` and `y`.

</div>

#### Performance Considerations: Pinned Memory

Initial performance tests often show that even for large vectors, the GPU version can be slower than the CPU version. This is usually because the time taken to transfer data between host and device memory (`cudaMemcpy`) dominates the total runtime.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pinned Memory)</span></p>

**Pinned memory** (or page-locked memory) is host memory that the OS cannot page out or relocate. By default, host memory allocated with `malloc` is pageable, meaning the operating system can move it around in physical memory. For the GPU to access this data, the CUDA driver must first copy it into a temporary, pinned buffer before transferring it to the device.

By allocating host memory directly as pinned memory, we eliminate this extra copy. Pinned memory is a scarce resource, so it should be used judiciously.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Pinned Memory)</span></p>

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

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(`cudaMallocHost` vs. `malloc`)</span></p>

Using `cudaMallocHost` instead of `malloc` can lead to a significant reduction in data transfer times, making the GPU's computational advantage more apparent.

</div>

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
* Streaming pipelines (especially combined with `cudaMemcpyAsync`), where pinned memory is required for true async `H2D`/`D2H`.

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

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Common CUDA Errors)</span></p>

- **the launch timed out and was terminated:** Kernel took too long. Common on systems with a graphical display where the OS kills kernels to prevent screen freezing. Solution: stop the X11 server.
- **unspecified launch failure:** Often indicates a segfault inside the kernel (out-of-bounds access or invalid pointer).
- **invalid configuration argument:** Invalid launch configuration — too many threads per block (> 1024) or too many resources per SM.
- **identifier "__eh_curr_region" is undefined:** Compiler issue related to non-static shared memory allocation. Declare shared memory arrays with static sizes.

</div>

## The Modern GPU Architecture

### Vector Architectures: The Foundation of Efficiency

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector ISA)</span></p>

**Vector ISA:** an instruction set architecture that provides instructions for performing the same operation over multiple data elements arranged as a vector.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Vector ISA Efficiency)</span></p>

The underlying hardware of a GPU is a vector machine leveraging **Vector ISAs (Instruction Set Architectures)**, which are efficient in three key ways:

  * **Compact:** A single instruction defines many operations, amortizing the cost of instruction fetch/decode and reducing branches.
  * **Parallel:** The operations are data-parallel (no dependencies), simplifying the hardware.
  * **Expressive:** Vector memory instructions describe regular access patterns, allowing the hardware to prefetch data and amortize memory latency.

</div>

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Kepler's SM's (SMX))</span></p>

The **Streaming Multiprocessor**, often abbreviated as **SM** (or **SMX** in the Kepler architecture), is the true "core" of the GPU where threads are scheduled and instructions are executed. Each GK110 SMX contains:

  * **192 SP** (Single-Precision) units
  * **64 DP** (Double-Precision) units
  * **32 Load/Store (LD/ST)** units
  * **32 Special Function Units (SFUs)** (for sine, cosine, etc.)

To manage execution, each SMX also contains **4 warp schedulers**. A key design philosophy of the GK110 was optimizing for **performance-per-watt** by reducing clock frequency (which has a cubic relationship with power) while increasing parallelism.

</div>

## The GPU Memory Hierarchy

Understanding the memory hierarchy is critical for high-performance GPU programming. Unlike CPUs with deep, transparent cache hierarchies, GPUs feature a complex, multi-level memory hierarchy that is **manually controlled by the programmer**. 

On a GPU, caches are used less for reducing latency and more for reducing memory contention and **coalescing** memory accesses.

### A Collaborative Approach

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(GPU's Collaborative Approach)</span></p>

The GPU philosophy is built on collaboration:

  * **Collaborative Computing:** In CUDA, you typically launch one thread per output element, grouped into **thread blocks**. Schedulers use the massive number of threads (parallel slack) to keep hardware busy. SIMT – Single Instruction, Multiple Threads.
  * **Collaborative Memory Access:** Memory access should be a team sport. Thread-collective computation and memory accesses. Threads within a block work together to load data efficiently. The memory controllers (MCs) are optimized to exploit that concurrency, especially through **memory coalescing**.

> **Key Takeaway:** If you do something on a GPU, do it collaboratively with all threads.

</div>

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Host memory)</span></p>

This is the main system RAM attached to the CPU. Data must be transferred between host and GPU global memory.

  * `cudaMemcpy`: This function explicitly transfers data using the GPU's DMA (Direct Memory Access) engines.
  * **Pinned Memory:** Standard host memory is "pageable" (unpinned). The GPU must copy it to a "staging buffer" first. **Pinning** memory (e.g., with `cudaMallocHost`) locks it in physical RAM, allowing for autonomous device access and faster transfers.
  * **Zero Copy:** On modern GPUs (Compute Capability \>= 2.0), threads can directly operate on pinned host memory.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Host memory is much slower than GPU memory)</span></p>

A system diagram shows the CPU/Host Memory connection (\~64 GB/s) is vastly different from the GPU-GDDR connection (\~460 GB/s) and internal GPU memory bandwidth (\~3.3 TB/s).

</div>

### Global Memory Coalescing: The Key to Bandwidth

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Memory Coalescing)</span></p>

**Coalescing** is the process of **combining many fine-grained memory accesses from multiple threads in a warp into a single, large GDDR memory transaction**. This is paramount for achieving high bandwidth.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Kepler's cache lines)</span></p>

* the L1 cache line size is 128 bytes (**latency-optimized**)
  * **latency-optimized, warp-aligned, spatially coherent** $\implies$ **large cache line size**
* the L2 cache line size is 32 bytes (**bandwidth-optimized**)
  * **bandwidth-optimized, multi-client, fine-grained access, matched to GDDR transfer size** $\implies$ **small cache line size**

</div>

When threads in a warp access memory, the ideal pattern is for them to access contiguous, aligned locations.

**Misaligned accesses:** One warp is scheduled, but accesses misaligned addresses.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Access Penalties)</span></p>

  * **Offset Access:** `data[addr + offset]`. If a warp's access crosses a cache line boundary, it may require fetching 5 cache lines instead of 4, reducing effective bandwidth.
  * **Strided Access:** `data[addr * stride]`. A stride of 2 means only half the data loaded into a cache line is used, resulting in 50% load/store efficiency.

The solution is to **manually control data movement**: threads collaboratively load a "tile" from global memory into shared memory in a coalesced pattern, then compute using the fast shared memory.

</div>

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(On Kepler)</span></p>

* Up to 1024 threads can be in a **thread block**.
* One thread block executes entirely on **one SM**.
* Each thread block is divided into **warps** of 32 threads.
* One SM can hold multiple thread blocks (up to 4) and up to 32 warps per block.

</div>

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Hardware Multi-Threading on G80)</span></p>

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

</div>

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Kepler's scheduler)</span></p>

The Kepler scheduler 
1. **checks** the scoreboard
2. **issues** an instruction to a ready warp using a **prioritized round-robin scheme**. 
   * The instruction is then broadcast to all 32 threads in that warp.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/kepler_scheduler_instruction_issue.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>Thread Scheduling on Kepler</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Data Hazards)</span></p>

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Matrix Multiplication Summary)</span></p>

  * **Performance is Memory-Bound:** High-performance computing is often less about math and more about data movement.
  * **Hierarchy is King:** Tiling (blocking) is the fundamental technique to exploit the memory hierarchy.
  * **Shared Memory:** The programmer's primary tool for maximizing computational intensity ($O(n)$ data reuse).
  * **Synchronization:** `__syncthreads()` is essential for correctness when using shared memory to manage RAW and WAR hazards.

</div>

## Scheduling and Optimization

This section applies the optimization techniques discussed so far to **parallel reduction** — a classic case study that exposes all key GPU performance pitfalls. The GPU scheduler maps CTAs (thread blocks) to SMs, groups threads into warps of 32, and exploits parallel slackness ($v \gg p$) to hide latency by switching between ready warps at zero cost.

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

The reduction example below will demonstrate all of the key performance pitfalls discussed earlier — memory coalescing, latency hiding, divergent branching, bank conflicts, and instruction overhead — and how to fix them systematically.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/chapter5_cuda_performance_issues_optimizations.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>CUDA performance issues and optimizations</figcaption>
</figure>

### The Parallel Reduction Problem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parallel Reduction)</span></p>

A **reduction** is a common parallel operation where an array of elements is "reduced" to a single value using an associative operator like addition, multiplication, or finding the maximum value.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Reductions)</span></p>

Examples of reduction include:

* **Global sum:** $s = \sum_{i=0}^{N} f(x_i)$
* **Global product:** $p = \prod_{i=0}^{N} f(x_i)$
* **Histogram:** $h_k = \sum_{i=0}^{N} (x_i = k) ? 1 : 0$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimization of Reduction)</span></p>

Reduction is a perfect candidate for optimization analysis because it is **memory-bound**: performance is limited by the speed at which data can be read from memory, not by the speed of the arithmetic. Therefore, our key performance metric will be **effective bandwidth (GB/s)**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Effective Bandwidth)</span></p>

**Effective bandwidth** is a real-world measure of data transfer speed, accounting for inefficiencies like latency, protocol overhead, and packet loss, unlike **theoretical (asymptotic) bandwidth**. It quantifies the actual data throughput for specific applications, representing the sustainable rate under typical conditions, often lower than the maximum possible speed.

</div>

### The Multi-CTA Challenge: Global Synchronization

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/chapter5_partial_results_communication.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Partial results communication/synchronization</figcaption>
</figure>

To process a large array, we must launch many CTAs, each responsible for reducing a chunk of the input data. This produces a partial result for each CTA. But how do we combine these partial results into a final, single value?

This would be simple if we had a global synchronization mechanism that could make all CTAs across the entire GPU wait for each other. However, *CUDA provides no such feature*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(Can we synchronise blocks within one SM?)</span></p>

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
2. **Scheduling Guarantees**: GPU scheduling is non-preemptive. A CTA, once scheduled on an SM, runs to completion. If a CTA were to wait at a global barrier for a CTA that hasn't even been scheduled yet, it could lead to a deadlock, where the entire GPU grinds to a halt. This would also conflict with the principle of parallel slackness needed to hide memory latency. The number of CTAs that could be synchronized would be limited by the number of resident blocks per SM, according to the formula: 
   
   $$\#\text{CTAs} \leq \#\text{SMs} \cdot b_r$$
   
   where $b_r$ is the number of resident blocks per SM.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Deadlock from Global Synchronization)</span></p>

The classic deadlock scenario works as follows:

1. You launch **more blocks than can be resident at once**:
   
   $$\#\text{CTAs} > \#\text{SMs} \cdot b_r$$
   
2. The GPU schedules up to #$\text{SMs} \cdot b_r$ blocks. These become **resident** and start running.
3. All resident blocks reach the **global barrier** and **wait** there.
4. While waiting, they **still occupy the SM resources** (registers, shared memory, block slots).
5. Because the SMs are "full" of waiting resident blocks, **no new blocks can become resident**, so the remaining (not-yet-scheduled) CTAs never start.
6. But the barrier can't release until **all CTAs arrive** → the ones that haven't started can't arrive → **deadlock**.

In short, **resident CTAs wait for unscheduled CTAs** to reach the barrier, but **unscheduled CTAs cannot be scheduled** because resident CTAs are parked at the barrier holding all SM resources. Global synchronization would therefore only be safe if all CTAs can be resident simultaneously:

$$\#\text{CTAs} \le \#\text{SMs} \cdot b_r$$

</div>

### The Solution: Kernel Decomposition

The standard solution is kernel decomposition. We write a kernel that performs a partial reduction, where each CTA writes its partial sum to global memory. After this first kernel completes, we launch it again on the smaller array of partial sums. **A kernel completion boundary acts as a de facto global synchronization point**. Because the reduction operation is the same at each level, we can reuse the same kernel code. **Negligible HW overhead, low SW overhead**

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Kernel Reuse in Practice)</span></p>

In practice, the final reduction stage often uses a different, simpler kernel (or is completed on the CPU). However, the same kernel *can* be reused at every level since the reduction operation is identical.

</div>

The figure depicts this as a tree-based reduction. A large array is at the bottom. The first kernel launch has many CTAs (CTA 0, CTA 1, CTA 2, etc.) that each compute a partial sum. These partial sums form a new, smaller array. A second kernel launch then reduces this smaller array, and so on, until a single final value remains.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/chapter5_kernel_decomposition.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Kernel decomposition</figcaption>
</figure>

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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Three Levels of CUDA Optimization)</span></p>

The techniques used in the reduction example generalize into three optimization levels:

**1. Algorithmic Optimizations** — changing the high-level parallel algorithm itself:

* *Hierarchical Tree:* The reduction algorithm is a classic example.
* *Associativity:* Exploiting $a + (b + c) = (a + b) + c$ to reorder operations for parallelism.
* *Algorithm Cascading:* Having each thread sum multiple elements sequentially before the parallel reduction, increasing ILP and reducing synchronization steps.

**2. Code Optimizations** — same algorithm, better implementation:

* *Addressing Changes:* Modifying indexing to improve memory coalescing and eliminate bank conflicts.
* *Loop Unrolling:* Reducing instruction overhead for the last warp (and the entire loop via templates).
* *Warp Shuffle Operations:* Exchanging data within a warp directly, bypassing shared memory.

**3. Scheduling Optimizations** — tuning how kernels are launched:

* *Kernel Launch Parameters:* Tuning grid and block dimensions for occupancy.
* *Overlapped Copy & Execute:* Using CUDA streams to overlap data transfers with kernel execution.

</div>

### Summary

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimization Workflow)</span></p>

1. **Set the Right Goal:** Identify whether your application is memory-bound (measure in GB/s) or compute-bound (measure in GFLOP/s).
2. **Identify Bottlenecks:** Systematically look for issues like memory access patterns, branch divergence, instruction overhead, and resource underutilization.
3. **Optimize Systematically:** Start with algorithmic optimizations, then code optimizations, and finally scheduling optimizations.
4. **Know When to Stop:** Balance raw performance with code readability and maintainability.

</div>

## Profiling and Understanding GPU Performance

Writing a parallel program that runs correctly is only the first step; making it run *fast* is the real challenge.

### Arithmetic Intensity

Computational intensity — the ratio of FLOPs to memory operations — is the key metric for performance analysis. Here we formalize it precisely.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Arithmetic Intensity)</span></p>

**Arithmetic Intensity** is defined as the ratio of floating-point operations (FLOPs) performed for every byte of data moved from memory:

$$r = \frac{f}{b}$$

where 

* $r$ is the arithmetic intensity in **FLOPs/Byte**
* $f$ is the number of **floating-point operations**
* $b$ is the number of **bytes of memory accessed**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Types of Intensity)</span></p>

* **Arithmetic Intensity:** (algorithm) The inherent ratio of operations to memory accesses in the pure, mathematical algorithm.
* **Computational Intensity:** (implementation) The actual ratio achieved by a specific code implementation. Caching can reduce memory accesses and increase it.
* **Machine Intensity:** (hardware capability) The ratio of peak FLOPs/sec to peak memory bandwidth (Bytes/sec). This is the intensity an application needs to fully utilize the hardware.

All are not equal in general.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_computational_intensity_required_for_peak_compute_performance.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>Extreme Amount of Computational Intensity (Data Reuse) Required</figcaption>
</figure>

### Memory-Bound vs. Compute-Bound

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Boundedness Classification)</span></p>

Based on arithmetic intensity, algorithms are classified as:

* **Memory-Bound:** Limited by the speed of memory access (e.g., vector addition: one FLOP per three memory operations).
* **Compute-Bound:** Limited by the raw computational power (e.g., dense matrix multiplication with high data reuse).
* **IO-Bound:** Limited by host-device data transfer time (PCIe bottleneck).

Identifying which category your application falls into is the first step toward optimizing it.

</div>

### The Roofline Model: A Visual Guide to Performance

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Roofline Model)</span></p>

The **Roofline Model** is a powerful tool for visualizing performance limitations. The model is a 2D plot with Arithmetic Intensity $r$ (FLOPs/Byte) on the x-axis and Attainable Performance $p$ (GFLOP/s) on the y-axis.

The "roofline" consists of two parts:

1. **The Slanted Roof:** Represents peak memory bandwidth ($m_p$, in GB/s). Performance is proportional to intensity: 
   
   $$p = m_p \cdot r$$

2. **The Flat Roof:** Represents peak compute performance ($f_p$, in GFLOP/s). Once intensity is high enough, the GPU's computational units are fully saturated.

The attainable performance is:

$$p = \min(m_p \cdot r, \; f_p)$$

If you are under the slanted part, you are **memory-bound**. If under the flat part, you are **compute-bound**.

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_roofline_model_1.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <!-- <figcaption>Explicit Method</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_roofline_model_2.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <!-- <figcaption>Implicit Method</figcaption> -->
  </figure>
</div>

<div class="math-callout math-callout--pro" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Algorithms could be divided into three classes)</span></p>

**If compute-bound** (hitting the flat roof).
* Algorithm includes plenty of memory accesses, but for each memory access only few calculations are performed.
* Execution time dominated by memory accesses
* **Optimization:** balance additions/multiplications, improve ILP, and exploit SIMD instructions.

**If memory-bound** (hitting the slanted roof).
* Algorithm includes plenty of integer and floating point operations; for each memory access many calculations are performed
* Execution time dominated by computations
* **Optimization:** use software prefetching, avoid load stalls, ensure memory affinity, and avoid non-local data accesses.

**IO-bound:** (limited in performance by IO operations)
* Usually disk or network access
* In the context of GPUs: PCIe bottleneck affecting host-device data movements

Arithmetic intensity is not fixed — it can scale with problem size, and effective caching directly increases the *effective* intensity by reducing main memory traffic.

</div>

### GPU Profiling

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Profiling)</span></p>

**Profiling** is the process of analyzing an application's behavior to understand its performance characteristics. It involves collecting data about both its static and dynamic properties.

* **Static Behavior:** Properties of the code itself, independent of any specific run (e.g., instruction count, instruction types).
* **Dynamic Behavior:** What happens when the code is executed (e.g., cache hit/miss rates, scheduler decisions, thread occupancy, memory stalls).

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Caveat</span><span class="math-callout__name">(Profiling is not free)</span></p>

* To gather this data, profilers rely on **hardware performance counters**. 
* These are special registers built into the processor that can count events like cache misses, instructions executed, or cycles the processor was stalled. 
* However, these counters are an expensive and limited resource. Accessing them can be costly and will inevitably affect the performance of the code you are trying to measure. 
* This is known as **profiling overhead**.

</div>

### Levels of Profiling: From C++ to Machine Code

When we write a CUDA C++ program, it goes through several stages of compilation before it can run on the GPU. We can analyze performance at any of these levels:

1. **C/C++:** The high-level source code we write.
2. **IR (Intermediate Representation):** A lower-level, platform-agnostic representation of the code, such as LLVM IR.
3. **PTX (Parallel Thread Execution):** An assembly-like language for NVIDIA GPUs. It's a stable instruction set that can be compiled for different GPU architectures.
4. **SASS (Shader Assembly):** The native, machine-level assembly language for a specific GPU architecture. This is what the hardware actually executes.

Profiling at the **SASS level gives you the most accurate and detailed view of what the hardware is doing**, as it's **closest to the metal**.

### Prerequisites for Profiling

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Before You Profile)</span></p>

1. **Ensure Correctness First:** Use tools like `cuda-memcheck` to find and fix memory errors before any performance analysis.
2. **Compile with Correct Flags:**
   * Enable optimizations (`nvcc -O2`) to profile production-like code.
   * Include debug information (`nvcc -lineinfo`) to map SASS instructions back to source lines.

</div>

## NVIDIA's Professional Profiling Toolkit

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nsight)</span></p>

NVIDIA provides a powerful suite of tools called Nsight for profiling and debugging GPU applications. For performance analysis, we will focus on two key components: **Nsight Compute** and **Nsight Systems**.

</div>

### Nsight Compute: Deep-Diving into Kernels

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nsight Compute)</span></p>

**Nsight Compute** is the primary tool for detailed analysis of individual CUDA kernels. It can collect an immense amount of data—nearly 1,700 different metrics on a modern GPU like the TU102—giving you an unprecedented view into your kernel's execution.

Nsight Compute offers two interfaces:

* A **command-line interface** (CLI) called `ncu`.
* A **graphical user interface** (GUI) called `nv-nsight-cu`.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Workflow</span><span class="math-callout__name">(Nsight Compute)</span></p>

A common workflow is to use `ncu` on a remote server (where the powerful GPU is) to collect performance data and save it to a report file. You can then download this file and open it with the `nv-nsight-cu` GUI on your local machine for in-depth visual analysis.

By default, `ncu` prints its results to the console (stdout). To save the results, you use the `--export` or `-o` flag.

```bash
# To run a profiler on an application and save the report
ncu --export my_report.ncu-rep ./my_application
```

</div>

### Working with Metrics

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Nsight Compute: Listing Metrics)</span></p>

Querying all 1,700+ metrics at once is overwhelming and inefficient. Instead, `ncu` provides predefined sets of metrics for common analysis tasks. You can list these sets with the command:

```bash
ncu --list-sets
```

or **custom combinations** of sets, sections, and metrics

```bash
ncu --set default --section SourceCounters --metrics sm__sass_inst_executed_op_shared <app>
```

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_ncu_list_sets.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
  <!-- <figcaption>Implicit Method</figcaption> -->
</figure>

</div>

### Nsight Systems: Analyzing the Entire Application

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nsight Systems)</span></p>

While Nsight Compute is for kernels, **Nsight Systems** is designed to analyze the performance of the entire system, focusing particularly on CPU-GPU interactions. It helps you identify high-level bottlenecks, such as:

* Time spent transferring data over the PCIe bus.
* Gaps in GPU execution where the GPU is idle waiting for the CPU.
* How different CUDA API calls and kernel launches overlap (or fail to overlap) over time.

Like Nsight Compute, it has a **CLI (`nsys`)** and a **GUI (`nsight-sys`)**.

Recording and analyzing can be separated:
* Record into file using `nsys profile <app>`, download for local use.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Nsight Systems Annotations)</span></p>

A powerful feature of Nsight Systems is the ability to add **custom annotations** to your host code using the **NVTX (NVIDIA Tools Extension)** library. This lets you mark specific regions of your C++ code, which will then appear as labeled ranges in the profiler's timeline, making it easy to correlate profiler output with your application's logic.

To use NTX, you need to include the header and link against the library:

```c
#include <nvToolsExt.h>

// Link your application with -lnvToolsExt
```

You can then bracket sections of your code with `nvtxRangePush` and `nvtxRangePop`:

```c++
// This code block will appear as a labeled "sleeping" range in the nsys GUI.
nvtxRangePush("sleeping");
sleep(100);
nvtxRangePop();
```

</div>

### Case Study: Profiling a Matrix Multiplication Kernel

We now apply these tools to a concrete example: profiling a `cuBLAS` matrix multiplication routine and examining how performance changes with matrix shape.

### The High Cost of Detailed Profiling

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name">(Profiling Has Overhead)</span></p>

First, it is crucial to understand that profiling has an overhead. The more metrics you collect, the more the profiler interferes with the application's execution, slowing it down. This happens because the GPU has a limited number of hardware performance counters. To collect many metrics, the profiler must re-run the kernel multiple times (called "passes" or "replays"), collecting a different subset of metrics each time.

**This is a critical lesson:** the performance numbers you see during a detailed profiling run are not the true performance of your application; they are the performance under heavy observation. You must always establish a non-profiled baseline first.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_ncu_profiling_expensive.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
  <figcaption>`ncu` profiling can be expensive...</figcaption>
</figure>

### The Challenge of Skewed Matrices

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The Challenge of Skewed Matrices)</span></p>

* The peak performance of libraries like `cuBLAS` is often benchmarked using square matrices (e.g., $1024 \times 1024$). **Peak performance assumption only holds true for square matrices**.
* But what happens if the matrices are **"skewed"**—for instance, very tall and thin, or very short and wide?

Let's consider the matrix multiplication $C = A \cdot B$, where the dimensions are $m \times k$ for matrix $A$ and $k \times n$ for matrix $B$. We will keep the total amount of work roughly the same but dramatically alter the shapes of $A$ and $B$.

This **massive performance degradation** suggests that the **skewed shapes** are causing a **major problem with memory access patterns**.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_skewed_matrices.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
  <figcaption>The chart plots GFLOP/s (y-axis) against various matrix dimensions $m-n-k$ (x-axis). The first bar, representing a square matrix 1024-1024-1024, shows a very high performance of nearly 8400 GFLOP/s. As the matrices become more skewed (e.g., 2048-512-1024, 4096-256-1024, and so on, up to an extreme 1048576-1-1024), the performance drops dramatically, eventually falling below 500 GFLOP/s. This shows a substantial performance loss even though the total number of floating-point operations remains identical.</figcaption>
</figure>

### Using Nsight Compute to Uncover the Truth

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Using Nsight Compute to Uncover the Truth)</span></p>

To diagnose this, we can use `ncu` to profile the application for each skewed matrix configuration. A simple shell for loop can automate this process, saving a unique report file for each run.

```bash
# This loop iterates, making the 'm' dimension larger and 'n' smaller each time,
# while keeping 'k' and the total work constant.
for ((i=1;i<=1024;i*=2)); do
  ncu -f --set full -o cuBLAS-skewed-$((1024*$i))-$((1024/$i))-$((1024)) \
  ./cuBLAS-test-sm75 $((1024*$i)) $((1024/$i)) $((1024))
done
```

After collecting the data, we can import the report files (`.ncu-rep`) and examine specific metrics related to memory performance. Some key metrics of interest include:

* **L1 Cache Hit Rate:** `l1tex__t_sector_hit_rate.pct`
* **L2 Cache Hit Rate:** `lts__t_sector_hit_rate.pct`
* **Shared Memory Accesses:** `sass__inst_executed_shared_loads`, `sass__inst_executed_shared_stores`
* **Global Memory Traffic (L1 to L2):** `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
* **Global Memory Traffic (L2 to DRAM):** `dram__sectors_read.sum`

</div>

### Analyzing Memory Traffic and Cache Performance

By plotting these metrics against the different matrix shapes, the source of the problem becomes clear.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Analysis</span><span class="math-callout__name">(Global Read Traffic)</span></p>

**Global Read Traffic:**

**Description of the Diagram:** A line graph shows two metrics: "L1->L2 LD [MB]" and "DRAM RD [MB]". For square-like matrices on the left, both traffic volumes are low. As the matrices become highly skewed to the right, the traffic from the L1 cache to the L2 cache (the blue line) explodes, increasing from a small amount to nearly 3000 MB. In contrast, the traffic from the L2 cache to main DRAM (the red line) stays relatively flat and low. This indicates that data is being constantly evicted from the L1 cache and must be re-fetched from L2, but the L2 cache is large enough to absorb most of this traffic, preventing a complete collapse from DRAM access. The read traffic amplification factor ranges from 1.25x for a moderately skewed case to an enormous 256x for the most skewed case.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_skewed_matrices_global_read_traffic.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <figcaption>Global Read Traffic</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_skewed_matrices_global_shared_mem_transaction.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <figcaption>Shared memory VS. global memory transactions</figcaption>
  </figure>
</div>

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Analysis</span><span class="math-callout__name">(Cache Hit Rates)</span></p>

* **Description of the Diagram:** A line graph plots the L1 and L2 hit rates. For square matrices, both hit rates are very high (L1 at ~95%, L2 at ~85%). As the matrix shape skews, the L1 hit rate plummets dramatically, falling close to 0% for the most extreme cases. The L2 hit rate also declines but remains much higher, confirming that L2 is catching most of the L1 misses.

**Internal Kernel Switching:** An interesting detail revealed by the profiling data is that the `cuBLAS` library is not using the same kernel for all matrix shapes. It intelligently selects different internal implementations based on the problem size and shape. For example:

* 1024-1024-1024 uses `volta_sgemm_128x64_nn`
* 32768-32-1024 uses `volta_sgemm_128x32_sliced1x4_nn`
* 524288-2-1024 uses `gemmSN_NN_kernel`
* 1048576-1-1024 uses a combination of kernel and `splitKreduce_kernel`

This shows the complexity of high-performance libraries, which contain multiple specialized algorithms to handle different types of inputs. However, even with these specialized kernels, the fundamental problem of poor data locality in skewed matrices leads to catastrophic cache performance and a massive drop in overall GFLOP/s.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_skewed_matrices_cache_hits_rate.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
  <figcaption>Cache hit rates and internals</figcaption>
</figure>

</div>

### Independent Thread Scheduling

The SIMT execution model has evolved significantly across GPU generations. Understanding this evolution explains why certain programming patterns are more efficient than others.

#### The Classic SIMT Model (Pascal and Earlier)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Scheduling Before Volta)</span></p>

On older architectures, a warp operated as a single unit with **one program counter (PC) and one call stack**. As described in the branch divergence section, divergent `if-else` paths were serialized using an active mask. The major drawback is that this model could also lead to **deadlock** if threads within a warp tried to synchronize with each other across a divergent branch.

</div>

### The Modern SIMT Model (Volta and Later)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Independent Thread Scheduling)</span></p>

Starting with the Volta architecture, NVIDIA introduced **Independent Thread Scheduling (ITS)**. In the ITS model, the GPU maintains the execution state (program counter, register state) for each individual thread. A **schedule optimizer** dynamically groups active threads from the same warp that are executing the same instruction. Threads can now diverge and reconverge at sub-warp granularity.

**Each thread now has its own program counter and register state bookkeeping, but instruction issue (and execution) is still at warp granularity.**

Execution is still SIMT at the core — the hardware executes one instruction across multiple threads. However, the scheduler can now group *any* threads from a warp that are at the same PC, rather than being constrained by a single warp-wide program counter. Crucially, after one branch finishes, its threads can immediately proceed without waiting for the other branch to complete.

**One subtlety:** the hardware does not automatically force full warp reconvergence at the join point. To explicitly reconverge, developers use the `__syncwarp()` intrinsic.

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_pascal_simt_model.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <figcaption>Pascal's (and before) SIMT model</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_06_volta_simt_model.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <figcaption>Volta's (and after) SIMT model</figcaption>
  </figure>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Starvation-Free Algorithms)</span></p>

ITS enables **starvation-free algorithms**: if threads within a warp contend for a lock, the scheduler guarantees every thread will eventually be scheduled. In the old model, a thread holding a lock might never be re-scheduled if another thread in the same warp was spinning, causing deadlock.

<!-- To recover strict warp-synchronous behavior (e.g., for warp-level reductions), use `__shfl_down_sync()`, call `__syncwarp()`, or compile for an older architecture (`-arch=compute_60`). -->

**Consider a lock (mutual exclusion):**
* Thread #0 holds the lock, but thread #1 is scheduled for execution and impedes the progress of thread #0.
* Volta’s ITS: thread #0 will eventually (question of when, not if) be scheduled for execution.

</div>

<!-- ### A Survey of Profiling Tools

Beyond NVIDIA's Nsight suite, several alternative profiling tools exist:

| Category | Examples | Pros | Cons |
| --- | --- | --- | --- |
| Hardware Counter-Based | nvprof (legacy), Nsight | Provides detailed hardware metrics (cache hits, etc.) | Heavy performance impact, slowdown due to kernel replays. |
| GPU Simulators | GPGPU-Sim, Multi2Sim, Barra | Extremely detailed cycle-accurate analysis. | Very slow; often lag behind the latest hardware generations. |
| Instrumentation-Based | GPU Ocelot, SASSI, NVBit, CUDA Flux | Fast, low overhead, allows for custom profiling logic. | Cannot measure hardware metrics; lifetime of research tools is limited. |
| CUDA API Trace | Part of Nsight Systems | Traces calls to the CUDA runtime API. | - |

### CUDA Flux: An LLVM-Based Instrumentation Profiler

**CUDA Flux** (Heidelberg University) is a lightweight alternative to hardware-counter profiling. It hooks into the LLVM middle-end, injecting per-basic-block counters at the IR level. After kernel execution, it combines execution counts with PTX instruction summaries to compute total instructions executed — at warp, CTA, or grid granularity.

| | Advantages | Limitations |
| --- | --- | --- |
| 1 | Fine-grained instruction counts | Profiles PTX, not SASS (one step removed from hardware) |
| 2 | Low overhead — no kernel replays | Requires `clang++` instead of `nvcc` |
| 3 | PTX is more stable than SASS | May not support all CUDA features (e.g., texture memory) |

### Predictive Performance Modeling (GPU Mangrove)

**GPU Mangrove** uses a machine-learning approach to predict kernel performance without running on every target GPU. It extracts portable code features (instruction counts, memory footprint, launch configuration, computational intensity) using tools like CUDA Flux, then trains a RandomForest model on measured execution time and power from 189 benchmark kernels. Results: 8.86–52.0% accuracy for execution time and 1.84–2.94% for power consumption across five GPUs, with prediction taking only 15–108 ms. -->

## N-Body Simulations: A Case Study in Optimization

We now apply the optimization principles discussed so far to a classic, computationally demanding problem: the N-body simulation. We start with a naive implementation and progressively refine it, demonstrating how architectural awareness leads to dramatic speedups.

### Memory Layout: AoS vs. SoA

Before tackling the N-body problem, we address a foundational optimization: how data is arranged in memory. Since memory access is far more expensive than arithmetic, streamlining access patterns is critical.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(AoS vs. SoA)</span></p>

- **Array of Structures (AoS):** All fields of one element are stored contiguously. Intuitive but causes non-coalesced GPU memory access.
- **Structure of Arrays (SoA):** All values of one field are stored contiguously. Naturally suited for coalesced memory access on GPUs.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Array of Structures (AoS): The Intuitive Approach)</span></p>

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

This is called an **Array of Structures (AoS)**. In memory, the data for each particle is laid out contiguously. The position, velocity, and mass of `particle[0]` are stored together, followed immediately by all the data for `particle[1]`, and so on.

**Memory Layout (AoS):** `[x0, y0, z0, vx0, ..., m0] [x1, y1, z1, vx1, ..., m1] [x2, ...]`

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(AoS is good for single-threaded applications)</span></p>

This layout is intuitive and works well for many single-threaded applications where you **typically process one entire object at a time**.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Structure of Arrays (SoA): The GPU-Friendly Approach)</span></p>

An alternative data layout is the **Structure of Arrays (SoA)**. Instead of one large array of particle structures, we create a single structure that contains arrays for each attribute.

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

In this layout, all the `x`-positions are stored together, all the `y`-positions are stored together, and so on.

**Memory Layout (SoA):** `[x0, x1, x2, ...] [y0, y1, y2, ...] [z0, z1, z2, ...] ...`


</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(SoA is superior choice for GPU applications)</span></p>

While this might seem less intuitive and requires more pointers to manage, it is often the **superior choice for GPU applications**. To understand why, we need to revisit the concept of memory coalescing.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Coalescing encourages SoA)</span></p>

Recall that coalesced memory access requires threads in a warp to access consecutive addresses. With **AoS**, threads reading the `x` coordinate of consecutive particles access non-contiguous locations (separated by `y`, `z`, `m` fields) — a non-coalesced pattern. With **SoA**, the `x` values of all particles are contiguous, yielding perfectly coalesced access.

While AoS can sometimes be improved using packed types like `float4`, SoA is the naturally GPU-friendly layout for memory-bound applications.

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_07_coalesced_aos.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <figcaption>Non-Coalesced AoS vs. Coalesced SoA</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_07_uncoalesced_aos_coalesced_soa.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <figcaption>Coalesced AoS --- Packed Values</figcaption>
  </figure>
</div>

### Applying Tiling to N-Body Simulations

The tiling principle from matrix multiplication — load a tile into shared memory, compute all interactions within it, then move on — applies directly to the N-body problem.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(N-Body Simulation)</span></p>

**N-body simulations** are a class of problems where the goal is to simulate the evolution of a system of $N$ bodies (particles) that interact with each other through a fundamental force like gravity or electromagnetism. The computational complexity is $O(N^2)$ for all-pairs calculations, making them extremely well-suited for GPU acceleration.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(N-Body Simulation)</span></p>

Applications include astrophysics (galaxy formation, neutron star collisions) and biomolecular simulation (protein folding, virus modeling).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(N-Body Simulation Complexity)</span></p>

* The all-pairs force calculation has $O(N^2)$ computational complexity but only $O(N)$ memory, making it **compute-bound** and ideal for GPUs. 
* For very large $N$, hierarchical algorithms like Barnes-Hut ($O(N \log N)$) approximate distant interactions, but the all-pairs kernel remains the inner loop.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Physics Behind the Simulation)</span></p>

The simulation is based on Newton's second law of motion:

$$F(x(t)) = m \frac{d^2x(t)}{dt^2}$$

We approximate the solution to this differential equation by discretizing time into small steps ($\delta t$). In each step, we calculate forces, update velocities, and then update positions.

The gravitational force ($f_{ij}$) between two bodies $i$ and $j$ with masses $m_i$, $m_j$ and positions $x_i$, $x_j$ is given by:

$$f_{ij} = G \cdot \frac{m_i m_j}{\|d_{ij}\|^2} \cdot \frac{d_{ij}}{\|d_{ij}\|}$$

where $d_{ij} = x_j - x_i$ is the vector between them and $G$ is the gravitational constant.

To avoid a division-by-zero error if two particles get too close ($\lVert d_{ij}\rVert \to 0$), a **softening factor** ($\epsilon^2$) is introduced in the denominator:

$$f_{ij} = G \cdot \frac{m_i m_j\, d_{ij}}{(\|d_{ij}\|^2 + \epsilon^2)^{3/2}}$$

The total force $F_i$ on body $i$ is the sum of the forces from all other bodies $j$:

$$F_i = \sum_{j=1}^{N} f_{ij} = G\, m_i \sum_{j=1}^{N} \frac{m_j\, d_{ij}}{(\|d_{ij}\|^2 + \epsilon^2)^{3/2}}$$

Once we have the total force $F_i$, we can update the velocity and position using a numerical integration method like the **Leapfrog Verlet** algorithm:

$$v_i\!\left(t + \tfrac{1}{2}\delta t\right) = v_i\!\left(t - \tfrac{1}{2}\delta t\right) + \delta t\, \frac{F_i}{m_i}$$

$$x_i(t + \delta t) = x_i(t) + \delta t \cdot v_i\!\left(t + \tfrac{1}{2}\delta t\right)$$

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_07_single_body_force.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <figcaption>Single Body Force</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_07_force_matrix.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <figcaption>Force Matrix</figcaption>
  </figure>
</div>

### Implementing an N-Body Simulation on the GPU

Our goal is to implement the force calculation step on the GPU, as it is the most computationally expensive part $O(N^2)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Initial Design: One Thread Per Body)</span></p>

A natural way to partition the problem for a GPU is to assign one thread to calculate the total force for one body. Each thread will:

1. Load the position and mass of its assigned body.
2. Iterate through all other $N-1$ bodies.
3. For each other body, calculate the interaction force and add it to an accumulator.
4. Write the final total force back to global memory.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Requires thread communication)</span></p>

This **approach requires communication** (each thread needs to read the data of all other bodies) and offers a prime opportunity to optimize for data re-use. We will explore two implementations: a naive one and a tiled one.

</div>

#### A Naive GPU Implementation

Let's start with a straightforward implementation. First, a helper function to calculate the interaction between two bodies. This function can be used by both the CPU (`__host__`) and GPU (`__device__`).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(bodyBodyInteraction Helper Function)</span></p>

This function takes the properties of two bodies and calculates the force components (`fx`, `fy`, `fz`) that body 1 exerts on body 0. It performs approximately 16 single-precision floating-point operations (FLOPs).

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
    float invDist = rsqrtf(distSqr); // rsqrtf is a single-precision floating-point function that computes the reciprocal (inverse) square root of a number
    float invDistCube =  invDist * invDist * invDist;
    float s = mass1 * invDistCube;

    // Calculate force components
    *fx = dx * s;
    *fy = dy * s;
    *fz = dz * s;
}
```

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(ComputeNBodyGravitation_GPU_AOS Kernel)</span></p>

Now for the main kernel. This kernel uses the AoS data layout with packed `float4` values to improve memory access. Each thread calculates the total force for one body `i`.

```c++
__global__ void ComputeNBodyGravitation_GPU_AOS(float *force, float *posMass, size_t N, float softeningSquared){
    // Outer loop to handle cases where N > number of threads (grid-stride loop)
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
    {
        // Accumulator for total force on body 'i'
        float acc[3] = {0};

        // Load position and mass of body 'i'
        float4 me = ((float4*)posMass)[i];
        float myX = me.x; float myY = me.y; float myZ = me.z;

        // Inner loop to iterate through all other bodies 'j'
        for (int j = 0; j < N; j++) {
            // Load position and mass of body 'j'
            float4 body = ((float4*)posMass)[j];

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

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Code Breakdown)</span></p>

* `i = blockIdx.x*blockDim.x + threadIdx.x`: This is the standard formula to calculate a unique global index for each thread. The outer for loop is a **grid-stride loop**: the stride is `blockDim.x * gridDim.x`, i.e. the total number of threads in the grid. If fewer threads are launched than bodies (e.g. 1024 threads for 10000 bodies), each thread handles multiple bodies by striding forward by the grid size—thread 0 handles bodies 0, 1024, 2048, etc. This decouples the launch configuration from the problem size $N$.
* `float4 me = ((float4 *) posMass)[i]`: We load the data for the thread's assigned body (`me`). By casting the posMass pointer to `float4*`, we are telling the hardware to perform a single 16-byte load, which can be more efficient. The `x`, `y`, `z` components store position, and the `w` component stores mass.
* `for (int j = 0; j < N; j++)`: This is the critical inner loop. The thread iterates through all $N$ bodies in the system.
* `float4 body = ((float4 *) posMass)[j]`: Inside the loop, the thread loads the data for each body `j` from global memory.
* **Data Reuse:** Notice that `me` is loaded once and reused $N$ times inside the inner loop. The data for body is loaded from global memory in every single iteration. However, because **all threads in the GPU are executing this same inner loop, they will all be requesting the same body data at roughly the same time. This means the data for body `j` will be loaded from global memory and can be temporarily stored in the L1/L2 caches, benefiting all threads that need it.**

</div>

#### Performance and the Power of Loop Unrolling

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Optimization</span><span class="math-callout__name">(Loop Unrolling using Pragma)</span></p>

Even in this naive implementation, we can apply a simple but effective optimization: **loop unrolling**. This technique reduces branch overhead by expanding the loop body, performing more work per iteration. We can hint to the compiler to do this using a `#pragma`.

```c++
#pragma unroll 16
for (int j = 0; j < N; j++) {
    // ... body of the loop ...
}
```

* `#pragma` unroll 1 will prevent the compiler from ever unrolling a loop.
* If no number is specified after `#pragma unroll`, the loop is completely unrolled if its trip count is constant, otherwise it is not unrolled at all.

The optimal unroll factor must be found empirically by testing different values. The results show a significant performance gain:

| Version | Unroll Factor | Body-Body Interactions per Second [G] |
| --- | --- | --- |
| GPU Naive | 1 (none) | 25.0 |
| GPU Naive | 2 | 30.0 |
| GPU Naive | 16 | 34.3 |

Loop unrolling improves performance by over 37% in this case.

</div>

### An Optimized Tiled Implementation Using Shared Memory

The naive version relies on the GPU's hardware caches to exploit data reuse. We can achieve even better performance by explicitly managing data reuse with tiling and **shared memory**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Optimized Tiled Implementation Using Shared Memory)</span></p>

The idea is to break the $N \times N$ interaction calculation into smaller tiles. Each thread block will work on one tile at a time.

1. Each thread is still responsible for one body `i`.
2. The inner loop that iterates over all `j` bodies is tiled.
3. In each step of the tiled loop, all threads in the block cooperate to load a "tile" of `blockDim.x` bodies into shared memory.
4. A synchronization barrier (`__syncthreads()`) is used to ensure all threads have finished loading before any thread starts computing.
5. Each thread then iterates through the bodies in the shared memory tile, accumulating forces.
6. Another synchronization barrier is used before the block proceeds to load the next tile.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tiled ComputeNBodyGravitation_Shared Kernel)</span></p>

This strategy drastically reduces global memory traffic. A tile of body data is loaded once from global memory into fast shared memory, and then every thread in the block can reuse it `blockDim.x` times.

```c++
__global__ void ComputeNBodyGravitation_Shared(float *force, float *posMass, float softeningSquared, size_t N){
    // Dynamically allocated shared memory for a tile of bodies
    extern __shared__ float4 shPosMass[];

    // Grid-stride loop for each thread to work on a body 'i'
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
    {
        float acc[3] = {0};
        float4 myPosMass = ((float4*)posMass)[i];

        // Outer loop that strides through the N bodies in tiles of size blockDim.x
        #pragma unroll 32 // Unroll factor can be applied here too
        for (int j = 0; j < N; j += blockDim.x) {
            // Cooperative load: each thread loads one body into the shared memory tile
            shPosMass[threadIdx.x] = ((float4*)posMass)[j + threadIdx.x];

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

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key Differences)</span></p>

* `extern __shared__ float4 shPosMass[]`: This declares a dynamically sized shared memory array. The actual size is specified during kernel launch.
* `for (int j = 0; j < N; j += blockDim.x)`: This is the tiling loop. Instead of incrementing `j` by 1, we jump by the block size. The tiles are **non-overlapping**: with 4 threads and $N=16$, the first tile loads bodies $[0,1,2,3]$, the second $[4,5,6,7]$, etc. Each thread (e.g. thread 0, assigned to body $i=0$) accumulates forces from all $N$ bodies across all tiles. The data reuse comes from the fact that one cooperative global memory load per tile serves every thread in the block.
* `shPosMass[threadIdx.x] = ...`: This is the cooperative load. Each thread `threadIdx.x` in the block is responsible for loading one body's data into the corresponding slot in `shPosMass`. This is a perfectly coalesced read from global memory.
* `__syncthreads()`: This is a barrier synchronization. It forces all threads in the block to wait at this point until every single thread has reached it. This is essential to prevent a thread from trying to read data from `shPosMass` before another thread has finished writing it.
* `float4 bodyPosMass = shPosMass[k]`: Inside the innermost loop, the body data is now read from the extremely fast on-chip shared memory, not slow global memory. This is the source of the performance gain.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Analysis</span><span class="math-callout__name">(The Impact of Shared Memory)</span></p>

Adding tiling and shared memory provides another significant performance boost on top of our previous optimizations.

| Version | Unroll Factor | Body-Body Interactions per Second [G] |
| --- | --- | --- |
| GPU Naive | 16 | 34.3 |
| GPU Shmem | 1 (none) | 38.2 |
| GPU Shmem | 2 | 44.5 |
| GPU Shmem | 4 | 45.2 |

The optimized version with shared memory achieves 45.2 G-interactions/sec, a **32% improvement over the best naive version and an 80% improvement over the original unoptimized kernel**.

</div>

### A Look at CPU Optimizations and Performance Comparison

While GPUs are excellent for this problem, it's insightful to see how an optimized CPU version performs. Modern CPUs also have parallel capabilities through SIMD (Single Instruction, Multiple Data) vector units, such as SSE or AVX.

#### Vectorization on the CPU

The provided code snippet for `bodyBodyInteraction` using `__m128` data types is an example of a CPU implementation using SSE intrinsics. `__m128` is a 128-bit data type that can hold four 32-bit floating-point numbers. Functions like `_mm_add_ps` perform a parallel add on all four floats at once. This is a form of fine-grained parallelism available on the CPU.

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

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Analysis</span><span class="math-callout__name">(CPU vs. GPU)</span></p>

The performance comparison uses an Intel E5-2670 CPU and a GK104 GPU.

| Version | Body-Body Interactions per Second [G] |
| --- | --- |
| CPU naive, single thread | 0.017 |
| CPU SSE, single thread | 0.307 |
| CPU SSE, 32 threads | 5.650 |
| GPU naive, best unroll | 34.3 |
| GPU shmem, best unroll | 45.2 |

* Vectorizing the CPU code (SSE) gives a 18x speedup over the naive single-threaded version.
* Using all 32 threads of the multicore CPU gives another 18x speedup.
* However, even the best multi-threaded, vectorized CPU implementation (5.65 G-interactions/sec) is significantly slower than the GPU.
* The optimized GPU implementation is 8x faster than the highly optimized 32-thread CPU version, showcasing why GPUs are the prime architecture for problems like N-body simulations.

</div>

### N-Body Summary

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(N-Body Optimization Takeaways)</span></p>

* **Data Layout Matters:** Choosing SoA layout is often critical for coalesced memory access on the GPU.
* **Maximize Data Reuse:** Tiling combined with explicit shared memory management reduces expensive global memory traffic.
* **Compiler Optimizations:** Simple directives like `#pragma unroll` can provide significant speedups by reducing instruction and branch overhead.
* **Further Techniques:**
  * *Warp Shuffle Instructions* (`__shfl()`): Allow threads within a warp to exchange data directly without shared memory, but require restructuring the algorithm to work at warp granularity.
  * *Constant Memory:* Beneficial for read-only data accessed by all threads, but unsuitable for data that changes every time step.

</div>

## CUDA Streams and Host-Device Communication

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(The Host-Device Bottleneck)</span></p>

The CPU (host) and GPU (device) are connected via PCIe, which creates a massive bandwidth mismatch:

* **On-device memory bandwidth:** up to ~3.3 TB/s
* **PCIe bus bandwidth:** ~64 GB/s

The GPU can compute at 34–67 TFLOP/s but can only receive data at a fraction of that rate. If data transfers are not overlapped with computation, the GPU sits idle — a condition known as being **PCIe-bound**. The key objective is to **overlap communication and computation**.

To hide this latency, we employ **task parallelism**: instead of sequentially performing `H2D` copy → kernel → `D2H` copy, we overlap these operations across independent data chunks using **CUDA Streams**.

**Up to now:** kernels to exploit data parallelism, host code still sequential
**Now:** exploit task parallelism on the host side GPUs are accelerators, thus CPU overhead should be as small as possible

</div>

### CUDA Streams

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(CUDA Stream)</span></p>

A **CUDA Stream** is an **ordered queue** of operations submitted to the GPU. Key properties:

* **Ordered Execution:** Within a single stream, operations execute in FIFO order.
* **Asynchronous Execution:** When the CPU issues a command to a stream, it returns immediately, allowing the CPU to continue queuing more work.
* **Inter-Stream Independence:** Operations in different streams have no guaranteed execution order relative to each other, enabling concurrency.

If CUDA events pop out of the queue, previous operations have completed (FIFO)

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_08_multiple_streams.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
    <figcaption>If CUDA events pop out of the queue, previous operations have completed (FIFO)</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_08_multiple_streams_conceptual_view.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <figcaption>Multiple CUDA streams - conceptual view</figcaption>
  </figure>
</div>


*Analogy: The Supermarket Checkout.* A single checkout lane (the default stream) serializes everything. Multiple checkout lanes (multiple CUDA Streams) allow one lane to process a large order (kernel execution) while another handles a quick transaction (data transfer).

By placing independent operations into different streams, we can enable the GPU's hardware to execute them concurrently, effectively hiding the latency of data transfers behind useful computation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coarse-Grained Concurrency in GPUs)</span></p>

Streams unlock **coarse-grained concurrency**: CPU/GPU concurrency, concurrent copy & execute, concurrent kernel execution, and multi-GPU parallelism — complementing the **fine-grained thread-level concurrency** within a kernel.

</div>

### Host-Device Synchronization

Since the CPU queues up work asynchronously, we need mechanisms to ensure results are ready before using them. CUDA provides three synchronization granularities:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Synchronization Granularities)</span></p>

**Context-Based Synchronization** — blocks the CPU until *all* CUDA operations complete:

* `cudaDeviceSynchronize()`: Blocks until all operations on all streams finish.
* Many blocking calls like `cudaMemcpy()` implicitly synchronize.

**Stream-Based Synchronization** — targets a *specific stream*:

* `cudaStreamSynchronize(stream)`: Pauses the host thread until all outstanding CUDA operations in a stream have completed.
* `cudaStreamQuery(stream)`: Non-blocking check; returns `cudaSuccess` or `cudaErrorNotReady`.

**Event-Based Synchronization** — the most *fine-grained* method:

1. `cudaEventRecord(event, stream)`: Places a marker into a stream.
2. `cudaEventSynchronize(event)`: Blocks the CPU until the event marker is reached.
3. `cudaEventQuery(event)`: Non-blocking check of event status.

Events allow synchronization at specific points in the workflow rather than waiting for an entire stream or device.

</div>

### Programming with CUDA Streams

#### The Default Stream and Sequential Execution

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Default Stream)</span></p>

When you launch a kernel or call `cudaMemcpy` without specifying a stream, you are using the **default stream** (stream 0). It is a **synchronizing stream**: an operation in the default stream will wait for all preceding operations in all other streams to complete before it begins, and any subsequent operation in any other stream will wait for the default stream operation to complete. This creates an inherent synchronization point, making overlap impossible with the default stream.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Default Stream)</span></p>

Consider this typical sequence of operations for a SAXPY kernel, which computes y[i] = $\alpha$ $\cdot$ x[i] + y[i]:

```c++
// All operations below are implicitly in the default stream
// 1. Copy input data X to device
cudaMemcpy(dx, hx, numBytes, cudaMemcpyHostToDevice);

// 2. Launch the kernel
saxpy<<<numBlocks, blockSize>>>(dx, dy, alpha, N);

// 3. Copy result data Y back to host
cudaMemcpy(hy, dy, numBytes, cudaMemcpyDeviceToHost);
```

Because all three operations are in the default stream, they will execute in a strictly sequential order. The kernel will not start until the first `cudaMemcpy` is completely finished, and the second `cudaMemcpy` will not start until the kernel is completely finished. **This serialization is exactly what we want to avoid** and is a prime example of the "serial fraction" described by Amdahl's Law, which limits the potential speedup of any parallel program.

</div>

##### Device Overlap Capability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">("Device Overlap" in CUDA)</span></p>

Most modern CUDA devices support a feature called **"Device Overlap"**, often referred to as "Concurrent copy and execute." This is the hardware capability that **allows a kernel to run at the same time as a data transfer**. You can programmatically check for this capability:

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

**Without this hardware feature, using streams for overlap is impossible**. Fortunately, it is standard on nearly all modern GPUs.

</div>

#### Pipelining with Multiple Streams

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Pipelining with Multiple Streams)</span></p>

To achieve overlap, we need to break our problem into smaller, independent pieces and process them in a pipeline. The strategy is as follows:

1. **Divide Data:** Split your large input and output data structures (e.g., arrays) into smaller segments or chunks.
2. **Create Streams:** Create two or more non-default streams.
3. **Process in a Loop:** Loop through the data segments, assigning each segment's workflow (Copy-Execute-Copy) to a different stream.

This creates a pipeline with three distinct phases:

* **Fill:** In the beginning, the first few stages of the pipeline are being filled. For example, **Stream 1** is copying data while the GPU is otherwise idle. Then, **Stream 2** starts copying while **Stream 1** starts computing.
* **Steady State:** The pipeline is full. This is the ideal state where data is being copied in for chunk $N+1$, the kernel is executing on chunk $N$, and results are being copied out for chunk $N-1$, all at the same time.
* **Drain:** As the loop finishes, the final chunks work their way through the now-emptying pipeline.

The effectiveness of this technique depends on computational intensity. If the kernel is too fast compared to the data transfer time, the pipeline will stall, waiting for data. Conversely, if the data transfers are much faster than the kernel, the benefit of overlap is minimal. We will analyze this trade-off mathematically in the next chapter.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(Effectivness of Pipelining with Multiple Streams)</span></p>

The effectiveness of this technique depends on computational intensity. If the kernel is too fast compared to the data transfer time, the pipeline will stall, waiting for data. Conversely, if the data transfers are much faster than the kernel, the benefit of overlap is minimal. We will analyze this trade-off mathematically in the next chapter.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_08_pipelining_multiple_streams.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>Pipelining Data Transfers with Kernel Execution</figcaption>
</figure>

#### Implementing a Multi-Stream Workflow

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Building Blocks for Multi-Stream Workflow)</span></p>

Let's see how to implement this in code. First, we need to know the relevant API calls.

1. **Creating a Stream:** A stream is represented by the `cudaStream_t` type. You create one with `cudaStreamCreate()`.

   ```c++
   cudaStream_t my_stream;
   cudaStreamCreate(&my_stream);
   ```

2. **Asynchronous Memory Copies:** To use streams, you must use the asynchronous version of `cudaMemcpy`, which is `cudaMemcpyAsync()`. It takes an additional argument: the stream ID.

   ```c++
   cudaMemcpyAsync(dst, src, count, kind, stream);
   ```

   **Important Note:** Asynchronous memory transfers require the host memory to be page-locked (or "pinned"). You must allocate host memory using `cudaMallocHost()` or `cudaHostAlloc()` instead of the standard `malloc()`. This prevents the operating system from moving the memory while the GPU is trying to access it via Direct Memory Access (DMA).

3. **Launching a Kernel in a Stream:** The kernel launch configuration is extended to include the stream ID as a fourth optional parameter (after grid dimensions, block dimensions, and shared memory size).

   ```c++
   kernel_name<<<grid, block, shared_mem_size, stream>>>(args...);
   ```

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Multi-Stream SAXPY (Version 1))</span></p>

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

The intent here is that while the saxpy kernel for `stream0` is running, the `cudaMemcpyAsync` operations for `stream1` can also be running, achieving our desired overlap. However, due to the architecture of older GPUs, this might not happen as we expect.

</div>

#### Architecture Matters: Fermi vs. Kepler and Newer

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Issues Using Streams)</span></p>

The way a GPU executes commands from streams depends heavily on its architecture.

* **Fermi Architecture (and older):** These GPUs have a single work queue for the copy engine and a single work queue for the compute engine. Even though we issued commands to two different software streams, they are all fed into the same two hardware queues. In our "Version 1" code, the device driver would schedule the operations like this:
  1. **Copy Queue:** `H2D(A0)`, `H2D(B0)`, `D2H(C0)`, `H2D(A1)`, `H2D(B1)`, `D2H(C1)`
  2. **Execute Queue:** `Kernel(0)`, `Kernel(1)`
* The problem is that the `D2H(C0)` operation (copying the result for stream 0) is placed in the copy queue before the input copies for stream 1 (`H2D(A1)` and `H2D(B1)`). Since operations within a queue are serial, the input copies for the second segment cannot begin until the output copy for the first segment is complete, destroying our desired overlap.
* **Kepler Architecture (and newer):** These GPUs introduced a feature called "Hyper-Q," which provides multiple hardware work queues (e.g., 32 queues each for copy and execute). This allows different software streams to map to different hardware queues, enabling true concurrent execution. On a Kepler or newer GPU, the "Version 1" code would likely achieve the desired overlap.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Multi-Stream SAXPY: More Robust Approach (Version 2))</span></p>

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

By issuing all input copies first, followed by all kernel launches, we maximize the opportunity for the GPU to overlap the execution of `Kernel(0)` with the input copies for stream 1 (`H2D(A1)` and `H2D(B1)`). This reordering makes the code more robust across different GPU architectures.

</div>

#### Common Pitfalls: Implicit Synchronization

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implicit Synchronization)</span></p>

The following operations act like `cudaDeviceSynchronize()`, destroying any stream overlap:

* **Page-locked host memory allocation:** `cudaMallocHost()` or `cudaHostAlloc()`.
* **Device memory allocation/deallocation:** `cudaMalloc()` or `cudaFree()`.
* **Synchronous memory operations:** Any function without the `Async` suffix (e.g., `cudaMemcpy()`, `cudaMemset()`).
* **L1/Shared Memory configuration changes:** `cudaDeviceSetCacheConfig()`.

Always use the `Async` versions and perform all memory allocations before your pipelined stream loop.

</div>

### Advanced Topics and Modern Alternatives

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(When Is Streaming Justfied?)</span></p>

* While CUDA Streams are powerful, they add programmer complexity. The key question is: **when is streaming mathematically justified?** 
* Beyond streams, modern CUDA features like **Unified Memory** and **Peer-to-Peer Access** simplify host-device memory management, sometimes at the cost of performance.

</div>

#### Is Streaming Always Worth It? An Analysis of Arithmetic Intensity

The goal of streaming is to hide the time it takes to transfer data over the PCIe bus ($t_{\text{PCIe}}$) by overlapping it with computation time ($t_{\text{COMP}}$). This strategy is only effective if the **computation is long enough to mask the transfer**. We can formalize this with the concept of Arithmetic Intensity.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name">(Arithmetic Intensity)</span></p>

**Arithmetic Intensity** ($r$) is defined as the ratio of floating-point operations (FLOPs) performed to the number of bytes transferred to or from memory:

$$r = \frac{\text{FLOPs}}{\text{Byte}}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Streaming Effectiveness Condition)</span></p>

For streaming to effectively hide PCIe latency, the arithmetic intensity $r$ of the kernel must satisfy:

$$r \ge \frac{c}{b}$$

where 
* $c$ is the **GPU's peak compute performance** (FLOPs/s)
* $b$ is the **PCIe bandwidth** (Bytes/s). 
  
If the kernel performs too few calculations per byte of data, it will finish long before the next chunk arrives, and the GPU will sit idle.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let's derive the condition required to successfully hide the PCIe latency. We'll make the following assumptions:

| Variable | Description | Units |
| --- | --- | --- |
| $N$ | The number of float elements in our segment. | elements |
| $b$ | The bandwidth of the PCIe bus. | Bytes/s |
| $c$ | The peak compute performance of the GPU. | FLOPs/s |
| $r$ | The arithmetic intensity of our kernel. | FLOPs/Byte |

The **total amount of data to be transferred** for a segment of $N$ floats is $4N$ bytes (since a float is 4 bytes). The time required for this transfer is:

$$t_{\text{PCIe}} = \frac{4N}{b}$$

The **total number of floating-point operations performed** on this segment is the arithmetic intensity multiplied by the number of bytes, which is $r \cdot (4N)$. The time required for this computation is:

$$t_{\text{COMP}} = \frac{\text{Total FLOPs}}{\text{Performance}} = \frac{r \cdot (4N)}{c}$$

To completely hide the data transfer latency, the computation time must be greater than or equal to the transfer time: $t_{\text{COMP}} \ge t_{\text{PCIe}}$.

Substituting our expressions for time:

$$\frac{r \cdot (4N)}{c} \ge \frac{4N}{b}$$

We can cancel out the $4N$ term from both sides and rearrange the inequality to solve for $r$.

</details>
</div>

#### Unified Virtual Addressing (UVA)

The complexity of manually managing memory buffers and `cudaMemcpyAsync` calls led NVIDIA to develop simpler memory models. The first step in this direction was **Unified Virtual Addressing (UVA)**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Unified Virtual Addressing)</span></p>

With **UVA**, the CPU and all GPUs in a system share a single virtual address space. A pointer, regardless of whether it points to host or device memory, has a unique address. However, while the GPU can access host memory directly, doing so is extremely slow as data must still travel over the PCIe bus. The programmer is still responsible for explicit locality optimizations:

* Single virtual address space for all memory in the system
* GPU code can access all memory
* Manual locality optimizations (cudaMemcpy)

**UVA primarily simplifies pointer management in multi-GPU applications.**

</div>

#### Unified Memory (UM)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Unified Memory)</span></p>

**Unified Memory (UM)** creates a pool of managed memory accessible to both the CPU and the GPU through a single pointer (the pool is shared between CPU and GPU). The CUDA runtime automatically migrates data between host and device on demand. When the CPU accesses managed data, it is ensured to be in host memory. When a GPU kernel accesses it, it is automatically paged to device memory.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(SAXPY application using Unified Memory)</span></p>

```c++
// Allocate managed memory accessible by both host and device
float *X, *Y; // unified pointers
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

Notice the complete absence of `cudaMemcpy` calls! This dramatically simplifies the code.

</div>

##### Performance of Unified Memory

<!-- <div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Analysis</span><span class="math-callout__name">(Memory Management Comparison)</span></p>

While UM is convenient, the automated data migration has overhead. The source context provides bar charts comparing the performance of matrix multiplication for different memory strategies on a Pascal-generation GPU. The strategies are:

* **malloc:** Traditional C-style host allocation with explicit `cudaMemcpy`.
* **pinned:** Page-locked host allocation (`cudaMallocHost`) with explicit `cudaMemcpy`.
* **UM:** Unified Memory with on-demand migration.
* **UM prefetch:** Unified Memory where the programmer provides hints to the runtime about where data will be needed next, allowing it to pre-migrate the data.

The charts illustrate the breakdown of time spent in `host2device` copy, kernel execution, and `device2host` copy for different matrix sizes:

* **$1k \times 1k$ Matrix:** For smaller problem sizes, the overhead of UM's page faulting mechanism is significant, making it slower than traditional pinned memory with manual copies.
* **$4k \times 4k$ Matrix:** As the problem size grows, the kernel execution time becomes more dominant. The convenience of UM starts to become more competitive, though still slightly slower than the manual approach.
* **$8k \times 8k$ Matrix:** For very large problems, the kernel time dwarfs the transfer and overhead time. Here, the performance of UM is very close to that of manual memory management, and using prefetching hints can close the gap even further.

</div> -->

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/pascal_unified_memory_comparison.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <!-- <figcaption>CPU + GPU System</figcaption> -->
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Unified Memory Trade-off)</span></p>

Unified Memory offers a promising trade-off between programmer productivity and performance, especially for applications where development speed is critical or memory access patterns are complex.

</div>

#### Peer-to-Peer Access for Multi-GPU Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Peer-to-Peer Access for Multi-GPU Systems)</span></p>

In systems with multiple GPUs connected by a high-speed interconnect like NVLink, it's possible for one GPU to directly access the memory of another without involving the host CPU. This is known as **Peer-to-Peer (P2P) Access**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(Enabling P2P Access)</span></p>

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

Once P2P access is enabled, you can perform a `cudaMemcpy` directly between the buffers of two different GPUs by specifying `cudaMemcpyDefault`. Even more powerfully, a kernel running on `gpu0` can directly read from or write to a pointer that resides in `gpu1`'s memory, as if it were its own.

This capability is essential for scaling applications across multiple GPUs efficiently.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(P2P Access)</span></p>

```c++
// Example: Copy from GPU 1 to GPU 0
cudaMemcpy(gpu0_buf, gpu1_buf, buf_size, cudaMemcpyDefault);

// Example: Kernel on GPU 0 directly accesses GPU 1 memory
// (inside a kernel launched on GPU 0)
gpu0_buf[idx] = gpu1_buf[idx];
```

</div>

### Summary: CUDA Streams and Host-Device Communication

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Streams for Latency Hiding)</span></p>

* **Problem:** The PCIe bus is much slower than the GPU's internal memory and compute capabilities.
* **Solution:** Task-level parallelism via CUDA Streams overlaps communication with computation.
* **Mechanism:** Divide work into independent segments, place each segment's workflow into a different stream using `cudaMemcpyAsync` and stream-specific kernel launches.
* **Prerequisite:** Only effective if the arithmetic intensity satisfies $r \ge c/b$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Unified Memory as an Alternative)</span></p>

* **UVA:** Provides a single address space but still requires manual data movement for good performance.
* **Unified Memory (UM):** Automates data migration between host and device based on access patterns. Eliminates explicit `cudaMemcpy` calls at the cost of some migration overhead.
* **Trade-off:** For large problems on modern hardware, UM performance is competitive with manual methods.

</div>

## Stencil Computations

### Introduction to Stencil Computations

In the world of high-performance computing, many problems involve updating values in a large, organized grid. Whether we are simulating weather patterns, analyzing medical images, or calculating how heat moves through a metal part, we often use a technique called **Stencil Computation**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stencil)</span></p>

A **stencil** is a geometric pattern used to update elements in a regular array (a grid). The value of a center element is updated based on the values of its neighbors. This specific "neighborhood" of cells used for the calculation is the stencil.

* **Iterative Kernel:** A kernel is a small function or program designed to run in parallel. In stencil computations, this kernel is iterative, meaning it runs repeatedly over the array to evolve the data over time.
* **Example (6-point stencil):** A 3D stencil where a central cube is updated using values from six direct neighbors: top, bottom, left, right, front, and back.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Real-World Applications of Stencils)</span></p>

1. **Image Processing:** Used in blob analysis, which helps computers identify and categorize distinct shapes within an image.
2. **Partial Differential Equations (PDEs):** Mathematical equations that describe how physical quantities change over space and time.
3. **Computational Fluid Dynamics (CFD):** Engineers use stencils to simulate the flow of liquids and gases, essential for designing airplanes, cars, and turbines.

While stencils are used for regular grids, more complex, irregular grids (like those used to model complex mechanical parts) often require Finite Element Methods (FEM).

</div>

### The Mathematics of PDEs and Finite Differences

Most stencil computations aim to approximate the solutions to Partial Differential Equations (PDEs).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial Differential Equation)</span></p>

A **Partial Differential Equation** is a function of multiple independent variables (like position $x$ and time $t$). These equations are used in physics, biology, economics, and chemistry to describe phenomena like heat transfer, Newtonian gravity, seismic wave propagation, and electrostatics.

</div>

#### Finite-Difference Methods (FDM)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Finite-Difference Methods (FDM))</span></p>

Since computers cannot solve complex calculus equations exactly, we use **Finite-Difference Methods**. These methods approximate a derivative (the rate of change) by looking at the difference between two points separated by a small spacing $k$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Derivative Approximation Methods)</span></p>

Using Taylor's polynomial, we can approximate the derivative of a function $f(x)$:

$$
f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2 + \dots
$$

In practice, we use three main types of "differences" to approximate these changes:

| Method | Formula Approximation | Error Level |
| --- | --- | --- |
| **Forward Difference** | $f_x \approx \frac{f(x + k) - f(x)}{k}$ | $O(k)$ |
| **Backward Difference** | $f_x \approx \frac{f(x) - f(x - k)}{k}$ | $O(k)$ |
| **Central Difference** | $f_x \approx \frac{f(x + k) - f(x - k)}{2k}$ | $O(k^2)$ |

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Error Order)</span></p>

$O(k)$ represents the "order of error." An $O(k^2)$ method is generally much more accurate than an $O(k)$ method because the error shrinks faster as the spacing $k$ gets smaller.

</div>

#### Discretizing the Grid

To solve these on a GPU, we turn continuous space into a grid. We define points $x_i = ih$ and $t_j = jk$, where $h$ is the space step and $k$ is the time step. This allows us to represent a physical object as a series of indexed points in a computer's memory.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_finite_differences.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>Finite Differences.</figcaption>
</figure>

### Solving the Heat Equation: Explicit vs. Implicit Methods

The **Heat Equation** describes how thermal conductivity ($c$) transports heat through a material over time. It is a parabolic equation, and there are two primary ways to solve it using stencils.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Explicit Method)</span></p>

In the **Explicit Method**, we calculate the state of a point at the next time step ($j+1$) using only the values from the current time step ($j$):

$$u_{i,j+1} = r\, u_{i-1,j} + (1 - 2r)\,u_{i,j} + r\, u_{i+1,j}$$

where 

* $r = \frac{ck}{h^2}$
* $c$ - thermal conductivity or how fast heat $u(x, t)$ is transported through material.

* **Pros:** Very simple to calculate. Each point can be computed independently.
* **Cons:** Unstable. Requires impractically small time steps ($k < \frac{h^2}{2c}$).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Implicit Method)</span></p>

The **Implicit Method** calculates the next time step by solving a system of equations where multiple unknown points at $t+k$ are related to each other:

$$u_{i,j} = -r\, u_{i-1,j+1} + (1 + 2r)\,u_{i,j+1} - r\, u_{i+1,j+1}$$

* **Pros:** Numerically stable. Much larger time steps are possible.
* **Cons:** Computationally intensive, requiring solving a linear system. (Here GPUs come in)

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_explicit_method.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <figcaption>Explicit Method</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_implicit_method.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <figcaption>Implicit Method</figcaption>
  </figure>
</div>

#### The Crank-Nicholson Method (CNM)

This is a hybrid approach. It applies weighted average ($\beta\cdot\dots\(1-\beta)\cdot\dots$) the explicit and implicit methods to achieve higher accuracy ($O(h^2 + k^2)$). For a simulation requiring 4 digits of accuracy, an implicit method might need 1 million points, while the Crank-Nicholson Method would only need 560 points to achieve the same result.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(GPU Connection)</span></p>

Because implicit methods and hybrid methods like CNM are computationally heavy but involve massive amounts of similar calculations, they are perfectly suited for the parallel architecture of a GPU.

</div>

### Optimizing Stencil Codes for GPU Architecture

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Stencil is memory-bound)</span></p>

Stencil codes are typically **memory-bound**. This means the GPU spends more time waiting for data to arrive from memory than it does actually performing math.

</div>

#### Domain Decomposition and Halos

To parallelize a stencil, we divide the data grid into chunks and assign each chunk to a thread block.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Halo)</span></p>

Because each point needs its neighbor's data, thread blocks must store an overlapping area of data from neighboring blocks. This overlap is called the **halo**.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_parallelization_of_stencil_codes.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>Parallelization of Stencil Codes.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Strategies for Memory Optimization)</span></p>

1. **1D vs 2D Partitioning:** 2D partitioning is common but can lead to "poorly aligned" memory access for vertical halos (vertical halos are poorly aligned in memory). 1D partitioning (strips) can sometimes be more efficient for communication (only horizontal).
2. **Shared Memory (Scratchpad):** Shared memory is a fast, small memory area on the GPU. Instead of reading from the slow global memory every time, threads can load a "tile" of the grid into shared memory, use it, and then move on.
3. **Marching Planes:** When dealing with 3D data, we don't have enough shared memory to store everything. Instead, we keep only three planes (top, middle, bottom) in shared memory and "march" through the 3D volume, cycling the buffers as we go.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_marching_through_3-dimensional_data_array.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>Marching through a 3-dimensional data array.</figcaption>
</figure>

### Directive-Based Programming (OpenACC)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(OpenACC)</span></p>

Writing raw CUDA code can be difficult for beginners. **OpenACC** is a directive-based standard that allows you to parallelize code by adding simple "pragmas" (hints to the compiler) to standard C, C++, or Fortran.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(OpenACC Execution Model)</span></p>

OpenACC uses a **host-directed model** with an attached accelerator device:
* CPU (host) manages the program and "offloads" compute-heavy loops to the GPU (accelerator).
* Three parallelism levels

Parallelism in OpenACC is organized into three levels:

1. **Gang Parallelism:** The coarsest level (CUDA: multiple thread blocks -> grid level).
   1. Fully parallel execution across execution units
   2. Limited support for synchronization
2. **Worker Parallelism:** A middle level (CUDA: warps at block level).
   1. Multiple threads on a single execution unit
   2. Latency hiding techniques
3. **Vector Parallelism:** The finest level (CUDA: threads at block level).
   1. Multiple operations per thread
   2. SIMD/vector operations

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Choosing parallelism type)</span></p>

**Programmer has to identify appropriate parallelism type**
* Fully-parallel loop (no dependencies) $\implies$ **gang**
* Vectorizable loop but with dependencies $\implies$ **fine-grain parallelism**, or **sequential implementation**

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Basic Syntax and Directives in OpenACC)</span></p>

To tell the compiler to run a loop on the GPU, you use:

```c
#pragma acc kernels
{
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
```

</div>

**Key Directives and Clauses:**

* **`parallel` vs `kernels`:** The `parallel` directive is more explicit—the programmer is responsible for identifying parallelism. The `kernels` directive gives the compiler more freedom to find and optimize parallelism on its own.
* **Data Management:** Moving data between the CPU and GPU is expensive. The `#pragma acc data` directive allows you to keep data on the GPU across multiple loops to save time.

| Clause | Description |
| --- | --- |
| `copy` | Moves data to the device at the start and back to the host at the end. |
| `copyin` | Only moves data from host to device. |
| `copyout` | Only moves data from device to host. |
| `present` | Tells the compiler the data is already on the GPU. |

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_openacc_1_compilation.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <figcaption>SAXPY – OPENACC #1 (Compilation)</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_openacc_1_running.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <figcaption>SAXPY – OPENACC #1 (Running)</figcaption>
  </figure>
</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_openacc_2.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <figcaption>SAXPY – OPENACC #2</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_openacc_3.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <figcaption>SAXPY – OPENACC #3</figcaption>
  </figure>
</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_openacc_4.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
  <figcaption>SAXPY – OPENACC #4</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_performance_results_of_pragmas.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
  <!-- <figcaption>SAXPY – OPENACC #4</figcaption> -->
</figure>

#### Performance Optimization in OpenACC

A common problem is **pointer aliasing**, where the compiler isn't sure if two memory pointers overlap. To fix this and allow parallelization, use the `restrict` keyword:

```c
float* restrict x; // Tells the compiler this pointer is unique
```

You can also use the `async` clause to allow the GPU to work on one task while the CPU continues with another, enabling overlapping execution.

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_09_openacc_optimization.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
  <!-- <figcaption>SAXPY – OPENACC #4</figcaption> -->
</figure>

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name">(Mapping Virtual to Physical Addresses)</span></p>

Computers use **Virtual Addresses (VA)** to manage memory. However, the actual data lives at a **Physical Address (PA)** in the hardware.

* **Shared Segments:** Portions of the address space can be shared such that multiple virtual addresses map to a single physical address.
* **Structured Address Space:** Typically, the virtual address space is organized into private segments (accessible only by one thread) and shared segments (accessible by all).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Communication and Synchronization in Shared Memory)</span></p>

In a shared memory system, threads communicate by writing to and reading from shared addresses. For example, if Thread 0 performs a store operation to a shared memory location, that change should eventually be visible when Thread 1 performs a load operation from that same location. This reliance on memory operations—including **atomic operations** (operations that happen completely or not at all, without interruption)—is the basis for synchronization.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_10_shared_memory_two_threads.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <!-- <figcaption>64 floating-point FMA mixed-precision operations per clock</figcaption> -->
</figure>

### Designing Communication Abstractions

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Communication)</span></p>

**The problem:** When multiple processors or threads need to work together, they must exchange data. But *how* should they communicate? There are two fundamentally different approaches, and a **communication abstraction** is a framework (a contract between hardware and software, like an ISA) that lets us compare them systematically.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solutions</span><span class="math-callout__name">(Shared Memory and Message Passing)</span></p>

The two approaches are:

* **Shared memory:** Threads communicate *implicitly* through a common address space. Think of a whiteboard in a shared office -- thread A writes `x = 42` on the whiteboard, thread B walks over and reads it. Nobody explicitly "sends" anything. This is simple to program (it looks like ordinary single-threaded code with loads and stores), but raises hard questions: what if A is still writing when B reads? What if B has a cached copy that is now stale?
* **Message passing:** Threads communicate *explicitly* by sending and receiving messages. Think of email -- thread A packs up the data and calls `MPI_Send(data, to=B)`, thread B explicitly calls `MPI_Recv(from=A)`. More work for the programmer, but the communication is visible and easier to reason about.

**GPUs chose the shared memory model, which is why the rest of this lecture focuses on coherence and consistency** -- problems that arise specifically because communication is implicit.

</div>

A communication abstraction has **five pillars**, each of which looks different depending on which approach is used:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Pillar 1</span><span class="math-callout__name">(Naming)</span></p>

**Naming:** How do you refer to (identify, point at) the data you want to communicate?

 * *Shared Memory:* A thread can refer to two kinds of storage: its own **private registers** (not shared -- only that thread can see them) and **memory addresses** in the virtual address space (organized into code, stack, and heap segments). All threads within the same process share one virtual address space, so if thread A writes to address `0x1000`, thread B can read from the same address `0x1000`. The "name" for shared data is simply a memory address. Under the hood, accessing a shared variable compiles to ordinary `load`/`store` instructions on these virtual addresses.
   * How does this work physically? There are two hardware designs:
     * **Global physical address space** (e.g. GPU, SMP): All processors share the same physical RAM. Virtual address `0x1000` in thread A and in thread B maps to the same DRAM chip. Simple and fast.
     * **Independent local physical address spaces** (e.g. NUMA cluster): Each processor has its own physical RAM. If thread B on machine B accesses a virtual address whose data lives on machine A's RAM, there is no local copy -- the hardware triggers a **page fault**. The OS intercepts this, fetches the page from machine A over the network, and maps it locally. This is called **Distributed Shared Memory (DSM)** -- it looks like shared memory to the programmer, but is secretly message passing underneath.
 * *Message Passing:* You don't name a memory address at all. Instead you name a *destination process* (by its rank/ID) and attach a *tag* to the message. The hardware handles the physical transport, but matching incoming messages to the right `recv` call and buffering them is done in software.
 * *Example:* To share a value `x`, shared memory just does `x = 42` (naming the address of `x`). Message passing does `MPI_Send(&x, 1, MPI_INT, dest=3, tag=0)` (naming process 3).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Pillar 2</span><span class="math-callout__name">(Operations)</span></p>

**Operations:** Now that you can name data, what can you actually *do* with it?

* *Shared Memory:* The operations are just ordinary CPU instructions -- **loads** (read from memory into a register) and **stores** (write from a register to memory). On CISC architectures (like x86) you can operate directly on memory addresses; on RISC architectures (like ARM) you must first load into a register, operate, then store back. On top of basic loads/stores, shared memory also provides **atomic read-modify-write** operations (e.g. `atomicAdd`, `compare-and-swap`) that guarantee no other thread can interfere mid-operation. This is how you build locks, counters, etc.
* *Message Passing:* The operations are fundamentally different -- **send** and **receive**. A thread packs data into a message and sends it; another thread posts a receive to get it. There are also **collective operations** that coordinate many processes at once (e.g. `MPI_Bcast` sends data from one process to all, `MPI_Reduce` combines values from all processes into one). This is more complex than a simple `load` -- you have to specify who you're talking to, how much data, what format, etc.
* *Example:* To atomically increment a shared counter: in shared memory, one instruction -- `atomicAdd(&counter, 1)`. In message passing, process B must send a request to process A (the "owner" of the counter), A increments it, and sends back the new value. Three messages for one increment.
* *Key difference:* Shared memory operations are simple and look like normal code. Message passing operations are explicit and verbose, but that explicitness makes communication visible and easier to optimize.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Pillar 3</span><span class="math-callout__name">(Ordering)</span></p>

**Ordering:** When multiple threads run simultaneously and issue loads/stores (or sends/receives), in what order do those operations actually take effect? This is surprisingly tricky.

* *Shared Memory:* Each thread individually follows **sequential program order** -- its own instructions execute top-to-bottom as written. But across threads there is no inherent ordering. If thread A does `x = 1` then `y = 2`, and thread B does `a = load(y)` then `b = load(x)`, the question is: can B see `y = 2` but still see the old `x = 0`? In a "perfect" world, no. In reality, **yes** -- hardware may reorder stores (via store buffers), execute out-of-order, or delay cache updates. This is the core problem that consistency models address.
* *Message Passing:* MPI provides **strong ordering** between a given sender-receiver pair: if process A sends message 1 then message 2 to process B, B is guaranteed to receive them in that order. Internally, MPI matches incoming messages by `(sender, tag)` using a linear search through a queue, and `MPI_Recv(ANY_SOURCE, ANY_TAG)` simply returns the first match.
* *Example:* Thread A does `x = 1; flag = 1`. Thread B spins on `while(flag == 0); print(x)`. You'd expect B to print 1. But in shared memory with relaxed ordering, B might see `flag = 1` (the store arrived) but still read `x = 0` (that store hasn't arrived yet). This is why we need memory fences and acquire/release semantics. In MPI, if A sends `x` then sends `flag` to B in two separate messages, B receives `x` first -- the ordering is guaranteed.
* *Why it matters for performance:* Enforcing strict global ordering is extremely expensive -- it prevents store buffers, out-of-order execution, and overlapping memory transactions. So real hardware **relaxes** ordering and lets the programmer insert explicit fences where ordering matters. GPUs are a very radical example of this relaxation.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Pillar 4</span><span class="math-callout__name">(Communication/Replication)</span></p>

**Communication/Replication:** How does data physically get from one processor to another?

*Shared Memory:* Communication is **implicit** -- it happens through the cache/memory hierarchy without the programmer doing anything explicit. When thread A writes `x = 42`, the value goes into A's cache. When thread B later reads `x`, the hardware must somehow get that value from A's cache to B. This is done through **cache coherence protocols** (snooping, directory-based) that automatically propagate and replicate data between caches. The programmer never sees this -- it's all handled by hardware. The downside is that you can't easily control *when* or *how* data moves, making optimization harder.
*Message Passing:* Communication is **explicit** -- the programmer decides exactly what data to send, when, and to whom. The data is physically copied from the sender's local memory into a network buffer, transmitted over the interconnect (e.g. InfiniBand, Ethernet), and placed into the receiver's local memory. There is no hidden data movement. The programmer has full control, which makes optimization easier but programming harder.
*Replication:* In shared memory, caches automatically create *replicas* of data -- multiple caches can hold copies of the same address. This improves read performance (data is close to each processor) but creates the coherence problem (keeping replicas in sync). In message passing, there are no automatic replicas; each process has its own private copy, and the programmer explicitly manages any duplication.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Pillar 5</span><span class="math-callout__name">(Performance)</span></p>

**Performance:** How efficient is the data transfer? This is ultimately what all the design choices above are trying to optimize.
 
*Shared Memory:* Performance depends on the cache hit rate, the coherence protocol overhead, and the memory hierarchy latency. Communication is "free" when data is already in the local cache (a few cycles) but very expensive on a cache miss (hundreds of cycles to fetch from remote memory). The implicit nature makes it hard to predict and optimize -- the programmer can't easily tell which accesses will be cache hits vs. misses.
*Message Passing:* Performance depends on message latency (time to send one message), bandwidth (data rate for large messages), and software overhead (packing/unpacking, matching). Communication cost is explicit and predictable -- the programmer knows exactly when data is being moved and can overlap communication with computation (e.g. `MPI_Isend` for non-blocking sends). The downside is the fixed overhead per message, which makes fine-grained communication (many small messages) expensive.
*Trade-off:* Shared memory is easier to program and efficient for fine-grained, irregular communication patterns (just read/write addresses). Message passing is better for coarse-grained, structured communication patterns where the programmer can batch data into large messages and overlap with computation.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_10_PRAM.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>**Symmetric Multiprocessors** (SMP) and **Chip Multiprocessors** (CMP) are the most successful parallel machines ever</figcaption>
</figure>

### The Coherence Problem

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Coherence Problem)</span></p>

* The goal of a memory system is to **reduce latency** (the time it takes to access data).
* We **use caches—small**, fast storage areas near the processor—to achieve this. 
* However, caches introduce a major challenge in multi-core systems: **Coherence**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cache Coherence)</span></p>

**Coherence** is the protocol that ensures all processors see the same (most recent) value for a specific memory location. Imagine a system with two processors, `P0` and `P1`, each with its own local cache. If both processors have a copy of variable $A$ (initially 0), and `P0` changes $A$ to 1, `P1` might still see 0 in its own local cache. Coherence protocols prevent this inconsistency.

</div>

#### Cache Write Policies

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Cache Write Policies)</span></p>

How a cache handles updates affects coherence:

* **Write-back (WB):** Updates are only made to the local cache and written to main memory later. This creates a significant coherence problem because other processors won’t see the change until the "write-back" occurs.
* **Write-through (WT):** Every update to the cache is immediately written to main memory. This makes coherence easier to manage but can be slower due to constant memory traffic.

</div>

#### Scalable Coherence Protocols

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Shared Bus Directory Protocol)</span></p>

A **shared bus** is a single physical wire that connects all processors to memory. Only one transaction can use the bus at a time -- if P0 is writing, everyone else must wait. This constraint is actually *useful* for coherence: because all processors see bus transactions in the same order, there is a natural serialization point. If P0 writes `A=1` and P2 writes `A=2`, one goes first on the bus, and every processor sees them in that same sequence. However, this single-wire design is not scalable -- with many cores, the bus becomes a bottleneck (all cores compete for one shared wire).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Snooping (bus-based coherence))</span></p>

**Snooping** is *not* periodic polling -- it is **passive, continuous listening**. Every processor’s cache controller is physically wired to the bus and sees every transaction that passes by, automatically. When P0 writes to address `A`, that write appears on the bus as an electrical signal. P1’s cache controller immediately checks: "do I have address `A` in my cache?" If yes, it invalidates or updates its copy. If no, it does nothing.

The bandwidth problem: snooping uses **broadcast** -- every write is announced to *all* processors. With 64 cores, every single write goes to all 64 cache controllers. But most of them don’t have a copy of that cache line, so they check and do nothing. The bus bandwidth is consumed by these useless announcements. This is why broadcast-based snooping does not scale.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Directory Protocol (the scalable alternative))</span></p>

Instead of broadcasting, the system maintains a **directory** -- a table (stored in main memory alongside the data) that records, for each cache line, which processors currently hold a copy. The flow:

1. P0 reads cache line `X` -- the directory records "P0 has X."
2. P2 also reads `X` -- the directory now records "P0 and P2 have X."
3. P0 writes to `X` -- P0 contacts the directory. The directory looks up who else has `X` (just P2) and sends an invalidation **only to P2**. Not to P1, P3, ..., P63.

This changes broadcast (tell everyone) to **multicast** (tell only the sharers), which scales much better. The trade-off is the storage cost for the directory itself and the extra latency of an indirection (going through the directory instead of directly snooping the bus).

</div>

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

* **Thread 0:** Sets `a = 1`, then checks if `b == 0`.
* **Thread 1:** Sets `b = 1`, then checks if `a == 0`.

In a "perfect" world, it should be impossible for both `if` statements to be true. However, in reality, both can be true if the hardware uses write buffering (delaying the actual store to memory).

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Producer-Consumer)</span></p>

* **Thread 0 (Producer):** Sets `a = 1`, then sets `flag = 1`.
* **Thread 1 (Consumer):** Waits for `flag == 1`, then prints `a`.

In many modern systems, **Thread 1** might print 0 instead of 1. This happens because the system might **reorder the operations**, making the `flag` update visible before the data `a` update is visible.

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_10_coherence.png' | relative_url }}" alt="G80 architecture for graphics processing" loading="lazy">
    <!-- <figcaption>SAXPY – OPENACC #2</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/gpu-computing/lecture_10_consistency.png' | relative_url }}" alt="G80 architecture for general-purpose processing" loading="lazy">
    <!-- <figcaption>SAXPY – OPENACC #3</figcaption> -->
  </figure>
</div>

### Relaxed Consistency and Performance

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Strict global ordering is expensive)</span></p>

* We relax consistency (allow reordering) for one primary reason: **Performance**. 
* Maintaining a strict global order is incredibly expensive and prevents hardware optimizations like:

* **Out-of-Order (OOO) execution:** Processors doing work out of sequence to stay busy.
* **Store Buffers:** Temporarily holding writes to avoid waiting for slow memory.
* **Sliced (banked) caches**

As we will see, GPUs are a very radical example of such relaxations.

</div>

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
* **Problem:** This prevents "latency hiding"—the ability to overlap memory access with other work. Memory operations happen (start and end) atomically:
  * Must wait for a store to complete before issuing next operation
  * After a load, issuing processor waits for load to complete, before issuing next operation

Easily implemented with a shared bus
* Bus as **synchronization point**, serializing all accesses

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Problems with Sequential Consistency)</span></p>

* **Aspect 1:** difficult to implement efficiently in hardware
  * No concurrency among memory access
  * Strict ordering of memory accesses at each processor (node)
  * Essentially precludes out-of-order CPUs
* **Aspect 2:** unnecessarily restrictive
  * Most parallel programs won‘t notice out-of-order accesses
* **Aspect 3:** conflicts with latency hiding techniques
  * Which relies on many concurrent outstanding requests

**Fixing SC performance**
* Revert to a less strict consistency model (relaxed or weak consistency)
* Programmer specifies when ordering matters

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_10_coco_nutshell.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>Coherence and Consistency in a nutshell</figcaption>
</figure>

### GPU Coherence and Consistency Models

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(GPU: relaxed memory consistency)</span></p>

GPUs use a **relaxed memory consistency** model because they are built to prioritize throughput and tolerate high latency.

</div>

#### The Data-Race-Free (DRF) Model

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Data-Race-Free (DRF))</span></p>

The GPU consistency model is **Data-Race-Free (DRF)**, similar to the models used in C++ or Java. It relies on **Release-Acquire** semantics:

* **Release Store:** A "release" operation ensures that all memory writes performed by the thread before the release become visible to other threads.
* **Acquire Load:** An "acquire" operation ensures that after observing the load, the thread will also see everything the releasing thread did before its release.

</div>

#### Scoped Memory

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Scoped Memory)</span></p>

In CUDA, you can define the scope of consistency:

* `thread`: Only within the single thread.
* `cta`: Within the thread block.
* `device`: Across the entire GPU.
* `system`: Across the GPU and CPU.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Release-Acquire Sequence)</span></p>

Below is an example of how a producer-consumer relationship is handled using CUDA atomics to ensure consistency.

```c++
// Thread 0 on SM 0 (Producer)
x = 42;
// Create a reference to a flag in device scope
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
// Store with ‘release’ semantics to ensure x=42 is visible first
flag.store(1, memory_order_release);
////////////////////////////////////////////////////////////////////

// Thread 1 on SM 1 (Consumer)
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
// Load with ‘acquire’ semantics to ensure we see the producer’s writes
while(flag.load(memory_order_acquire) != 1);
// Because of Release-Acquire, this assertion is guaranteed to pass
assert(x == 42);
////////////////////////////////////////////////////////////////////
```

1. **Thread 0** writes data to `x`.
2. **Thread 0** then performs a store on a flag using `memory_order_release`. This acts as a "fence," pushing the write of `x` out to memory before the flag is updated.
3. **Thread 1** loops (polls) the flag using `memory_order_acquire`.
4. Once **Thread 1** sees the flag change to 1, the `memory_order_acquire` ensures it also sees the most recent value of `x`.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_10_coherence_in_gpus_cpus.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>Coherence in GPUs and CPUs</figcaption>
</figure>

### Address Space Views: From Uniprocessor to GPU

To understand *why* GPUs can get away without hardware coherence, it helps to build up from the simplest case and see what breaks at each step.

#### Uniprocessor (Single Memory Controller)

With a single processor and a single memory controller, there is **no coherence problem** even with caches -- there’s only one entity reading and writing, so the cache is always consistent with the processor’s view.

However, there is a different issue: **homonyms** -- the same virtual address can refer to different physical data depending on which process is running. For example, virtual address `0x1000` in process A maps to physical address `0x5000`, but in process B it maps to `0x9000`. If process A’s cache line for `0x1000` is still in the cache when process B runs, B might read A’s stale data.

Solutions:
* **Flush caches on context switch** (`WBINVD` instruction): simple but expensive -- you throw away all cached data every time the OS switches processes.
* **ID-tagged caches using ASID (Address Space Identifier):** Each cache line is tagged with the process ID, so the hardware can distinguish "address `0x1000` belonging to process A" from "address `0x1000` belonging to process B." No flushing needed.

#### Multiprocessor with a Single Memory Controller

Now add multiple processors, each with its own cache, but still a single shared memory controller. This is where coherence becomes necessary: multiple caches can hold copies of the same address, and one processor might update its copy without the others knowing.

The key insight: the **memory controller acts as a natural synchronization point**. Since all memory requests funnel through this single controller, it can enforce coherence -- it sees every read and write and can coordinate invalidations or updates. This is the classic SMP (Symmetric Multiprocessor) setup.

#### Multiprocessor with Multiple Memory Controllers

As systems grow, a single memory controller becomes a bottleneck (just like a single bus). So we add multiple memory controllers, each responsible for a portion of the address space.

Now the question: **which memory controller is responsible for a given address?** This is determined by **static mapping** -- a fixed function (e.g. based on address bits) that maps each address range to a specific memory controller. For example, addresses `0x0000-0x3FFF` go to MC0, `0x4000-0x7FFF` go to MC1, etc.

Each memory controller now acts as a synchronization point for its own address range. All of them together must coordinate to maintain coherence.

*Excursion -- COMA (Cache-Only Memory Architecture):* An alternative where there is no static mapping at all. Main memory is treated as a giant cache, and data migrates dynamically to wherever it’s needed. This avoids the problem of data being "far" from the processor that needs it, but is more complex to implement.

#### Adding a Cache Hierarchy (CPU-style)

Modern CPUs add a cache hierarchy: exclusive **L1/L2 caches per core** and a shared **L3/LLC (Last-Level Cache)** that all cores can access. The LLC is typically **sliced (banked)** -- divided into multiple slices that can be accessed concurrently, improving bandwidth.

The important point: this cache hierarchy does not change the coherence picture. The same coherence protocols (snooping or directory) apply. Having a shared LLC just adds another level where data can be found before going to main memory.

#### GPU Memory Hierarchy -- L2 (LLC)

The GPU’s L2 cache is also sliced, similar to CPU LLCs. But there’s a crucial difference: **GPUs don’t need hardware coherence for the L2**.

Why? Because the GPU’s L2 slices are part of the **fixed address mapping** -- each L2 slice is paired with a memory controller, and the address-to-slice mapping is static. A given address always goes to the same L2 slice. There is no situation where two different L2 slices hold conflicting copies of the same address, because a given address is *owned* by exactly one slice.

Trade-offs of this design:
* **Latency increases:** A core (SM) might need to access an L2 slice on the other side of the chip. This adds latency compared to a local cache.
* **GPUs don’t care:** GPUs are designed to tolerate memory latency (by switching to other warps while waiting). CPUs would care because they are latency-sensitive.
* **Effective cache size can be reduced:** If data is not equally distributed among the memory controllers, some L2 slices fill up while others sit empty. But cache size is less critical for GPUs than for CPUs (again, latency tolerance).

#### GPU Memory Hierarchy -- L1

The L1 cache is **local to each Streaming Multiprocessor (SM)** and is private to the thread blocks running on that SM. This is where things get interesting for coherence:

* **Exclusive cache:** The L1 is not shared between SMs. Two SMs *can* have copies of the same address in their L1 caches.
* **Consistency guarantees only at thread block boundaries:** The GPU only guarantees that memory is consistent at the **start and end of a thread block’s life**. Within a thread block, threads on the same SM use `__syncthreads()` for synchronization. Between SMs, there are no guarantees during execution.
* **Write-through, no write-allocate policy:** When an SM writes to global memory through L1, the write goes directly through to the L2 (write-through). The L1 does not allocate a cache line for the written data (no write-allocate). This means writes are always immediately visible at the L2 level.
* **Invalidation at kernel completion boundaries:** When a kernel finishes, the L1 caches are invalidated -- all cached data is thrown away. The next kernel starts with clean L1 caches. This eliminates the possibility of stale data across kernel launches.
* **No memory traffic from invalidation:** Because L1 uses write-through, all data is already in L2 at the time of invalidation. Invalidating L1 just discards local copies -- no data needs to be written back. This is essentially free.

This is **software-controlled coherence**: instead of complex hardware protocols tracking every cache line, the GPU relies on the programming model (BSP-like: compute phase, then synchronize) to ensure correctness. The hardware just needs to invalidate at boundaries.

#### GPU Memory Architecture (Detailed)

<figure>
  <img src="{{ ‘/assets/images/notes/gpu-computing/lecture_10_gpu_memory_architecture.png’ | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>GPU memory architecture</figcaption>
</figure>

The data path from SM to DRAM:
1. Each SM has an **L1 cache** and **shared memory**, connected via a MUX (they share the same on-chip SRAM).
2. Below the SM, **address-sliced crossbars** provide a high-bandwidth, contention-free path into the memory system. The crossbar routes each address to the correct L2 slice based on address bits.
3. The **L2 cache (LLC)** is split into slices, each paired with a memory controller and GDDR channel.

Cache policies:
* **L1:** 128B cache line size (= 32 threads x 4B, matching a warp), **write-invalidate** (on a write, invalidate the local copy and write through), **no write-allocate** (writes don’t bring data into L1).
* **L2:** 32B cache line size (stores and over-fetches), **write-back** (dirty data stays in L2 until evicted), **write-allocate** (a write miss brings the line into L2 first).

GPU kernels are typically **write-once** -- each thread writes its output once. This means L1 write-allocate is unnecessary (why cache a value you’ll never read again?), which is why L1 uses no-write-allocate.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why is L2’s cache line smaller than L1’s?)</span></p>

On CPUs, L2 lines are typically the same size or larger than L1 lines, and L2 *contains* L1’s data (inclusive hierarchy) or holds evicted lines (exclusive hierarchy). So it seems counterintuitive that GPU L1 = 128B but L2 = 32B.

The reason is that GPU L1 and L2 are **not in a containment relationship** -- they serve different roles with different access patterns:

* **L1 (128B) is optimized for reading in bulk.** The line size matches one coalesced warp access: 32 threads x 4B = 128B. One L1 line serves an entire warp in a single transaction.
* **L2 (32B) is optimized for fine-grained memory management.** It handles traffic from *all* SMs, including small write-through stores (one thread writes 4B, not 128B). Using 32B granularity means a small store doesn’t force L2 to fetch/allocate 128B just to update a few bytes. It also reduces false sharing when different SMs access different 32B portions of the same 128B region.

L1 is write-through, so writes skip L1 and go directly to L2. L1 is invalidated entirely at kernel boundaries. L2 is not "holding evicted L1 data" -- it’s an independent, memory-side cache with its own granularity chosen for its own workload.

**But if coalescing always fetches 128B into L1, when does 32B matter?** The interaction depends on the operation:

* **Reads (L1 miss):** L1 needs 128B but doesn’t talk to DRAM directly -- it goes through L2. That 128B request is fulfilled by **4 separate 32B L2 sectors**. Some of those sectors might already be in L2 (hit), while others must come from DRAM (miss). So L2 can serve a *partial* hit -- e.g. 3 of 4 sectors cached, only 1 fetched from DRAM. This is more efficient than treating 128B as all-or-nothing.
* **Writes:** L1 is write-through, no write-allocate. The new value is sent directly to L2 and is **not stored** in L1. However, if the written address already has a cached copy in L1, that copy is **invalidated** (write-invalidate policy) so no stale data can be read. The next read to that address will miss in L1 and fetch the fresh value from L2. Because the write goes to L2 at its native 32B granularity, a single thread writing 4B does not trigger a 128B fetch.
* **Non-coalesced reads:** If a warp’s accesses are scattered, L1 might fetch multiple 128B lines but use only a few bytes from each. At L2, these become separate 32B sector accesses -- finer-grained, reducing wasted DRAM bandwidth.

In short: coalescing determines how many **L1 lines** are touched. L2’s 32B granularity determines how efficiently those L1 misses (and all writes) are handled at the memory side.

**Why invalidate L1 on write instead of updating it?** One might ask: why not just change the value in the L1 cache line and also write through to L2? That way L1 stays warm for future reads. The answer is that **there is no coherence protocol between L1 caches**. If SM0 and SM1 both have address X cached in their L1, and SM0 writes `X = 42`, updating SM0’s L1 would make SM0 correct -- but SM1’s L1 still has the stale value, and there is no hardware mechanism to notify SM1. By invalidating SM0’s L1 copy instead, the GPU forces all future reads (from any SM, including SM0) through L2 -- the single source of truth (each address maps to exactly one L2 slice, so L2 is inherently coherent). This sacrifices L1 hit rate on writes to avoid needing expensive inter-SM coherence hardware. Two additional reasons this makes sense: (1) GPU kernels are typically write-once (the written value is never read back, so caching it in L1 wastes capacity), and (2) L1 never holds dirty data, keeping the hardware simple -- no write-back logic, and invalidation at kernel boundaries is free.

**In summary, L1 is effectively a read-only cache.** A write invalidates the *specific cache line* containing the written address (not the entire L1 -- the entire L1 is only invalidated at kernel completion boundaries). The typical lifecycle of an L1 cache line is:

1. Warp reads address X → L1 miss → fetch 128B from L2 → cache in L1.
2. Subsequent reads of X → L1 hit (fast).
3. Some thread writes to X → that L1 line is invalidated, value goes to L2.
4. Next read of X → L1 miss again → fetch fresh value from L2.

This is why GPU code that reads data many times benefits greatly from L1 (e.g. the N-body inner loop reading body positions), but code that interleaves reads and writes to the same address gets no L1 benefit.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Coalescing is not a CUDA method)</span></p>

Coalescing is **not** a CUDA API call or a feature you enable -- it is a **hardware behavior** that happens automatically. When a warp of 32 threads issues a load instruction, the memory hardware examines all 32 addresses simultaneously and groups them by which cache lines they fall into. It then issues **one memory transaction per cache line touched**:

* Thread 0 reads addr 0, thread 1 reads addr 4, ..., thread 31 reads addr 124 → all 32 addresses fall within one 128B cache line → **1 transaction** (coalesced).
* Thread 0 reads addr 0, thread 1 reads addr 512, thread 2 reads addr 1024, ... → addresses span many cache lines → **many transactions** (uncoalesced).

Coalescing is simply the natural consequence of how cache lines work -- the hardware always fetches a whole cache line regardless. A "coalesced" access just means the warp’s access pattern happens to align with cache line boundaries, so all 32 threads are served by the minimum number of transactions. The programmer’s role is to arrange data layouts so that consecutive threads access consecutive addresses (e.g. SoA over AoS). The hardware does the rest.

</div>

### How L1, L2, and Global Memory Are Connected

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(GPU cache hierarchy is not like a CPU)</span></p>

In a CPU, caches form a nested hierarchy close to the core: `Core → L1 → L2 → L3/LLC → DRAM`. Each level is physically near the previous one, optimized for latency, and typically inclusive (data in L1 is also in L2).

In a GPU, L1 and L2 are **not in a containment relationship** -- they are two independent caches connected by a crossbar:

```
SM → L1 → address-sliced crossbar → L2 slice 0 → Memory Controller 0 → GDDR
                                   → L2 slice 1 → Memory Controller 1 → GDDR
                                   → L2 slice 2 → Memory Controller 2 → GDDR
                                   → ...
```

The critical difference: **L2 is not next to L1 -- it’s on the other side of the chip, physically paired with the memory controllers.** The address-sliced crossbar routes each request to the correct L2 slice based on address bits (static mapping).

| Design choice | CPU rationale | GPU rationale |
| --- | --- | --- |
| L2 location | Close to cores (minimize latency) | Next to memory controllers (maximize bandwidth) |
| Inclusion | L1 ⊂ L2 (coherence simplicity) | Independent (no coherence between L1s) |
| L1 write policy | Write-back (keep data close) | Write-through (L1 is a read-only cache) |
| L2 role | Catch L1 evictions, reduce L3 traffic | Shared source of truth, absorb writes from all SMs |

CPUs optimize for **latency** -- every cache level is as close and fast as possible. GPUs optimize for **bandwidth** -- L2 is placed where the memory bandwidth is (next to the controllers), and the latency is hidden by switching warps.

</div>

#### Data Flow: Reads

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(L1 always operates at 128B granularity)</span></p>

L1 has no concept of a "partial" cache line -- it either has the full 128B line or it doesn’t.

</div>

1. Warp issues a load → L1 checks: do I have this 128B line?
2. **L1 hit:** Serve immediately (fast, a few cycles). L2 is not involved.
3. **L1 miss:** The request travels through the crossbar to the specific L2 slice that *owns* that address (determined by static mapping).
4. The hardware needs the full 128B for L1. This maps to **4 × 32B sectors** at L2. Crucially, which 4 sectors is determined by **alignment**, not by which address was accessed. L1 lines are 128B-aligned (they always start at addresses that are multiples of 128), so the 4 sectors are the fixed group that falls within that 128B boundary:
   ```
   L2 sectors:  [0:32B][1:32B][2:32B][3:32B][4:32B][5:32B][6:32B][7:32B][8:32B]...
                |_________ L1 line 0 _________||_________ L1 line 1 _________|
                addr 0-127                      addr 128-255
   ```
   For example, if the accessed address falls in sector 5 (addr 160-191), the 128B-aligned L1 line starts at addr 128, so L1 fetches sectors **{4, 5, 6, 7}** -- not {5, 6, 7, 8}. The grouping is fixed by alignment, not by the access point.
5. **How does L2 find these 4 sectors?** Caches are not arrays where consecutive addresses sit next to each other in physical SRAM. They are lookup structures (like hash tables). Each address maps to a specific location via index bits extracted from the address:
   ```
   Address: [  tag  |  index  |  offset  ]
                          ↓
                  direct lookup into this "set" in the cache
   ```
   The hardware does not search the whole L2. It extracts the index bits, jumps directly to the right set, and checks the tag. Each of the 4 sectors is looked up independently by its own address. They don’t need to be physically adjacent in L2’s SRAM -- they just need to be retrievable by address, which they always are.

   Furthermore, all 4 sectors share the same high-order address bits (they fall within the same 128B-aligned region), which guarantees they all map to the **same L2 slice**. So the hardware only talks to one L2 slice and does 4 independent lookups within it:
   ```
   Sector 4 addr: 0b...0001_00|000  → L2 slice X
   Sector 5 addr: 0b...0001_01|000  → L2 slice X (same high bits)
   Sector 6 addr: 0b...0001_10|000  → L2 slice X (same high bits)
   Sector 7 addr: 0b...0001_11|000  → L2 slice X (same high bits)
   ```

6. L2 checks each of the 4 sectors independently:
   * Sectors that are **L2 hits** are served directly from L2.
   * Sectors that are **L2 misses** are fetched by the paired memory controller from GDDR.
7. Once all 4 sectors are gathered, the complete 128B line is assembled and delivered to L1.

This is where L2’s 32B granularity pays off: if 3 of 4 sectors are already in L2, only 1 sector (32B) needs to come from GDDR -- not the full 128B.

**Important caveat:** An L1 miss always requires the **full 128B line**, even if the address you actually need is just 4 bytes in one sector. If L2 has only the sector you missed on (say sector 5) but not the other three, the hardware must still fetch the missing sectors from GDDR:

* Sector 4: L2 miss → fetch from GDDR
* Sector 5: L2 hit → served from L2
* Sector 6: L2 miss → fetch from GDDR
* Sector 7: L2 miss → fetch from GDDR

Three GDDR fetches even though the address you cared about was in L2. The L1 line won’t be delivered until all 4 sectors are gathered. This is a real cost of the large L1 line size. GPUs accept this trade-off because: (1) **coalesced access is the common case** -- well-written GPU code has consecutive threads accessing consecutive addresses, so all 128B is typically useful; (2) **GDDR is optimized for burst transfers** -- fetching 32B vs. 128B doesn’t save as much as you’d think, since DRAM naturally transfers data in bursts (often 32-64B per burst); (3) **latency is hidden** -- while waiting for missing sectors, the SM switches to other warps.

#### Data Flow: Writes

1. SM issues a store → L1 invalidates its copy of that line (if cached).
2. The write travels through the crossbar to the owning L2 slice.
3. L2 handles it with write-back policy -- the dirty data stays in L2 and is **not** immediately written to GDDR.

#### When Does L2 Write Back to GDDR?

L2 uses write-back, so dirty data can stay in L2 indefinitely. It is only written to GDDR when something forces it out:

1. **Eviction:** L2 is full and a new sector needs space. If the evicted sector is dirty, it gets written back to GDDR. This is the normal, ongoing mechanism.
2. **Data must leave the GPU:** For example, `cudaMemcpy(device→host)` -- the driver ensures L2 dirty data is flushed to GDDR before the transfer begins, so the CPU reads correct values.
3. **Explicit synchronization:** Certain CUDA operations (e.g. memory visibility operations at `system` scope) may trigger an L2 flush to ensure data is visible outside the GPU.

**Kernel completion does NOT flush L2.** Only L1 is invalidated at kernel boundaries. L2 persists across kernel launches -- if kernel A writes results that kernel B needs, those results are already in L2. There's no reason to push them to GDDR just because the kernel ended. L2 only writes back when something *outside* L2 needs the data.

This is also why L1 and L2 use different write policies: L1 uses write-through because there are many L1 caches that are not coherent with each other -- writes must reach the single source of truth (L2) immediately. But there is only one L2 (logically), so there is no urgency to push data further to GDDR until it's actually needed.


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Does flushing L2 hurt a concurrently running kernel?)</span></p>

If kernel A finishes and the CPU copies its results back (`cudaMemcpyAsync` on stream 1), while kernel B is still running on stream 2, does the L2 flush hurt kernel B?

First, note that `cudaDeviceSynchronize()` waits for **all** work on **all** streams to finish -- so you cannot have "A done, B still running" with a device-wide synchronize. This scenario only arises with **CUDA streams**, where each stream synchronizes independently.

In the streams case, kernel B is not hurt because:

1. **The flush is targeted, not a full L2 wipe.** The hardware/driver only ensures the specific addresses being transferred are written back from L2 to GDDR. Kernel B's data (at different addresses) stays in L2 untouched.
2. **Write-back does not necessarily invalidate.** Flushing pushes dirty data to GDDR but can leave a clean copy in L2. Kernel B can still hit on it.
3. **Even in the worst case** (an L2 line gets evicted), the data is still in GDDR. Kernel B would see cache misses and re-fetch -- a performance hit, not a correctness issue.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(When is L2 fully flushed?)</span></p>

In normal GPU operation, **a full L2 flush essentially never happens.** L2 is designed to persist as the source of truth. It is important to distinguish **flush** (write dirty data to GDDR, keep clean copy) from **invalidate** (discard all cached data):

| Situation | Full L2 flush/invalidate? | What actually happens |
| --- | --- | --- |
| Eviction (ongoing) | No | Individual dirty sectors written back as they're replaced. Normal cache behavior. |
| `cudaMemcpy` D→H | No | Targeted write-back of specific addresses being transferred. |
| Kernel completion | No | Only L1 is invalidated. L2 persists. |
| `cudaDeviceSynchronize()` | No | Just waits for work to finish. Doesn't touch L2. |
| `cudaCtxResetPersistingL2Cache()` | Yes | Explicit API to reset L2 persistent cache reservations (CUDA 11+). |
| `cudaDeviceReset()` | Yes | Entire GPU state is torn down, including all caches. |
| Context destruction | Yes | CUDA context is destroyed, GPU state is cleaned up. |

A full L2 invalidation is drastic -- it only happens when the GPU context is being destroyed or explicitly reset. Data flows in and out of L2 through evictions and targeted write-backs, but the cache as a whole is never wholesale cleared during normal execution. This is fundamentally different from L1, which is invalidated at every kernel boundary.

</div>

#### L1 Eviction

When L1 evicts a cache line (to make room for a new one), it simply **discards** the full 128B line. Since L1 is write-through, there is never dirty data in L1 -- everything has already been written to L2. No write-back is needed, making eviction free.

### Summary: Why GPUs Don’t Need CPU-Style Coherence

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key Takeaways)</span></p>

* **Coherence is an artificial problem introduced by caches.** Without caches there would be no coherence issue -- everyone would just read from the same memory. Caches create replicas, and replicas can diverge.
* **CPUs provide strong coherence guarantees** because they must support legacy code, arbitrary user programs, and are latency-sensitive (they need caches close to the core, which forces complex coherence).
* **GPUs provide very relaxed consistency** because:
  * There is no legacy GPU code -- the programming model was designed from scratch with these constraints in mind.
  * The execution model is **BSP-like** (Bulk Synchronous Parallel): compute, then synchronize. Synchronization points are essentially the start and end of a thread block’s life. This constrained model can be leveraged as the foundation for consistency.
  * GPUs can **tolerate latency** (by switching warps), so they can live with small caches and place the LLC far away next to the memory controllers -- eliminating the need for coherence between L2 slices.

</div>

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

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(__syncthreads() is good, but rigid)</span></p>

Historically, CUDA provided the `__syncthreads()` function to synchronize threads within a block. While effective, it is "rigid"—it's all-or-nothing for the entire block. **Cooperative Groups (CG)** is a modern API that allows for much more flexible, fine-grained synchronization.

</div>

### Core Concepts of Thread Groups

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Thread Group)</span></p>

A **Thread Group** is a generic type in the CG API that can represent any subset of threads. This allows you to perform collective operations (like synchronization or data shuffling) on specific groups rather than the whole block.

Supported hierarchies include:

* **Warp-Level Groups:** Represented by `coalesced_group` (`cooperative_groups::coalesced_group`), allowing threads within a warp to cooperate using Warp-level intrinsics like `__shfl__` (shuffle). 
* **Block-Level Groups:** Represented by `thread_block` (`cooperative_groups::thread_block`).
* **Grid Groups:** Allows for synchronization across the entire grid (all blocks). This requires a "Cooperative Launch" and hardware with Compute Capability $\geq 6.0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Block-Level Synchronization)</span></p>

Using **Cooperative Groups**

* `thread_block block = this_thread_block();`
* `block.sync();`

improves code readability and provides a standardized way to handle synchronization.

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

An equivalent way to synchronize is `cg::synchronize(block)`.

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

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(More effective CUDA cores for MMA)</span></p>

As Deep Learning became dominant, NVIDIA introduced **Tensor Cores**. These are specialized hardware units designed specifically for Matrix-Multiply-Accumulate (MMA) operations. Standard CUDA cores can do these operations, but Tensor Cores are specialized hardware units that handle matrix ops much faster with mixed-precision arithmetic. Key changes: wider, specialized ALUs

</div>

### What is a Tensor Core?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tensor Core)</span></p>

Standard CUDA cores handle one calculation at a time. A **Tensor Core** operates on entire **Matrix Tiles** (e.g., $4 \times 4$ or $8 \times 8$ matrices). It performs a **Fused Multiply-Add (FMA)**, calculating $D = A \cdot B + C$.

The primary advantage is **Mixed-Precision Arithmetic:**

* **Input** ($A$ and $B$): Usually lower precision like `FP16` (16-bit floating point).
* **Accumulation** ($C$ and $D$): Can be higher precision like `FP32` to maintain accuracy.

By using Tensor Cores, a GPU can generate significantly more Multiply-Accumulate (MAC) operations per clock cycle than standard cores.

</div>

<figure>
  <img src="{{ '/assets/images/notes/gpu-computing/lecture_11_4_floating-point_FMA_mixed-precision_operations_per_clock.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>64 floating-point FMA mixed-precision operations per clock</figcaption>
</figure>

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

__global__ void wmma_example(half *a, half *b, float *c,
                             int M, int N, int K,
                             float alpha, float beta)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x*blockDim.x+threadIdx.x)/warpSize;
    int warpN = (blockIdx.y*blockDim.y+threadIdx.y);

    // Declare the fragments (placeholders for matrix tiles in registers)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the accumulator fragment with zeros
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the K-dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
    
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {

          // Load the inputs
          wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
          wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
          
          // Perform the matrix multiplication
          wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
      
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
      
      for(int i=0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}
```

</div>

### Evolution of Tensor Cores

NVIDIA has updated Tensor Core technology with every major architecture generation:

| Generation | Architecture | Key Additions |
| --- | --- | --- |
| Gen 1 | Volta | Introduced Tensor Cores supporting FP16 for inputs and FP32 accumulation. (V100 GPU contains 640 Tensor Cores (8 per SM))|
| Gen 2 | Turing | Added integer formats (`INT8`, `INT4`) for faster inference |
| Gen 3 | Ampere | More Tensor Cores per SM, higher throughput. Mixed-precision support: TF32, FP16, BF16, INT8, FP8 (E4M3/E5M2), etc. |
| Gen 4 | Hopper | Introduced `FP8` support for massive HPC and AI scaling |
| Gen 5 | Blackwell | Added `FP6` and `FP4` support; introduced Tensor Memory (`TMEM`) - previous generations operated on SM & RF. Replaced warp-synchronous MMA with a single-thread instruction (`tcgen05.mma`) |

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

[Cerebras Architecture Deep Dive: First Look Inside the HW/SW Co-Design for Deep Learning](https://www.cerebras.ai/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw-sw-co-design-for-deep-learning)