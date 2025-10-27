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

# GPU Computing: Architecture and Programming

## Notation & Acronyms

| Term/Symbol | Definition |
| --- | --- |
| ASIC | Application-Specific Integrated Circuit |
| BSP | Bulk-Synchronous Parallel |
| CPU | Central Processing Unit |
| CUDA | Compute Unified Device Architecture (NVIDIA's parallel computing platform) |
| DP | Double Precision (Floating-Point) |
| FPGA | Field-Programmable Gate Array |
| GPGPU | General-Purpose computing on Graphics Processing Units |
| HBM2 | High Bandwidth Memory 2 |
| ML | Machine Learning |
| SIMD | Single Instruction, Multiple Data |
| SIMT | Single Instruction, Multiple Threads |
| SM | Streaming Multiprocessor |
| SP | Single Precision (Floating-Point) |
| a | Speed-up |
| e | Efficiency (of parallelization) |
| p | Parallel fraction of a program |
| s | Serial fraction of a program |
| N | Number of parallel processing units |
| t_{ser}(M) | Time of the best serial implementation for input size M |
| t_{par}(M, N) | Time of the best parallel implementation for input size M on N units |
| P | Power |
| f | Frequency |
| C | Capacitance |
| V | Voltage |
| I_{leakage} | Leakage Current |

---

## Executive Summary

This document provides a foundational overview of GPU computing, synthesizing the core principles of its architecture, programming models, and performance characteristics. The central theme is the deliberate architectural divergence of GPUs from traditional CPUs, a shift driven by the end of Dennard scaling and the demand for immense computational throughput.

### Key Architectural & Performance Insights

- Throughput over Latency: CPUs are latency-oriented, employing complex mechanisms like deep pipelines, branch prediction, and speculative execution to minimize the execution time of a single thread. In contrast, GPUs are throughput-oriented. They utilize thousands of simpler, in-order processing cores and massive multithreading to hide, rather than minimize, the long latency of memory accesses. The goal is to keep the vast number of functional units busy with useful work from other threads while one thread waits for data.
- Massive Parallelism: The physical design of a GPU reflects its purpose. Die shots reveal a landscape dominated by replicated computational units (Streaming Multiprocessors or SMs) and a relatively small area dedicated to control logic and caches. This contrasts sharply with a CPU die, where large caches and sophisticated control logic consume a significant portion of the silicon budget. This design enables a GPU to execute tens of thousands of threads concurrently.
- Memory Hierarchy: While both CPUs and GPUs have a memory hierarchy, their characteristics are tailored to their respective tasks. GPUs feature extremely high-bandwidth memory (e.g., HBM2 providing nearly 2 TB/s) to feed their numerous cores, though often with less total capacity than system RAM. The on-chip memory hierarchy, including large register files and software-managed shared memory, is critical for achieving high performance by staging data close to the execution units.
- Performance Limits and Amdahl's Law: The effectiveness of a parallel architecture like a GPU is fundamentally constrained by the serial portion of an application. Amdahl's Law formally states that the maximum speed-up is capped by this serial fraction, regardless of how many parallel processors are added. This highlights the critical importance of designing algorithms that are overwhelmingly parallel to leverage the full potential of a GPU.
- Dual View of Parallelism (SIMT vs. SIMD): From a programmer's perspective (the software view), a GPU executes scalar programs on thousands of independent threads, a model known as Single Instruction, Multiple Threads (SIMT). This simplifies programming. From the hardware's perspective, these scalar threads are bundled into groups (warps) and executed on wide vector units, an implementation of Single Instruction, Multiple Data (SIMD). The hardware creates the illusion of scalar processing, effectively hiding the underlying vector nature of the architecture. This approach aligns closely with the principles of the Bulk-Synchronous Parallel (BSP) model, using "parallel slackness" (many more virtual threads than physical cores) to efficiently pipeline computation and communication.

In essence, GPU computing represents a paradigm shift from optimizing single-thread performance to maximizing aggregate system throughput. Success in this domain requires understanding the hardware's capabilities—its immense parallelism and high-bandwidth memory—and designing algorithms that can exploit these strengths while being mindful of inherent limitations like the serial bottleneck described by Amdahl's Law.

---

## 1. Introduction and Motivation for GPU Computing

The primary objective of this study is to understand the architecture of modern GPUs, create and optimize GPU programs using frameworks like CUDA, and analyze the factors that determine their performance.

### 1.1 The Evolution from Graphics to General-Purpose Computing

Initially designed for the gaming industry to accelerate graphics rendering, Graphics Processing Units (GPUs) have evolved into powerful, programmable parallel processors. The graphics pipeline—involving vertex and fragment shaders that perform a vast number of parallel floating-point operations—provided a natural foundation for general-purpose computation. Since approximately 2007, with the introduction of frameworks like NVIDIA's CUDA, GPUs have been increasingly used for non-graphical tasks, a field known as General-Purpose GPU (GPGPU) computing.

This transition has enabled tremendous performance gains in scientific and commercial domains, including:

* Computational Fluid Dynamics (CFD)
* Cosmology
* Molecular Dynamics
* Weather and Climate Research
* Machine Learning

The performance trajectory has been staggering, with supercomputers crossing the Tera-FLOP/s barrier in 1997, the Peta-FLOP/s barrier in 2012, and the Exa-FLOP/s barrier in 2022. Much of this advancement is attributable to the integration of massively parallel accelerators like GPUs.

### 1.2 CPU vs. GPU: A Tale of Two Architectures

The fundamental difference between a CPU and a GPU lies in their design philosophy. A CPU is optimized for low-latency access to cached data sets and excels at single-thread performance. A GPU is optimized for high-throughput computation and excels at parallel execution. A visual inspection of their die shots reveals this distinction: CPUs dedicate significant silicon to large caches and complex control logic, whereas GPUs dedicate most of their area to arithmetic logic units (ALUs).

The following table provides a quantitative comparison across several generations of processors, illustrating the architectural divergence.

| Feature | CPU (Broadwell, 2016) | GPU (Kepler, 2012) | GPU (Pascal, 2016) | GPU (Volta, 2017) |
| --- | --- | --- | --- | --- |
| Core Count | 22 cores (2 FP-ALUs/core) | 13 SMs (192 SP, 64 DP ALUs/SM) | 56 SMs (64 SP, 32 DP ALUs/SM) | 84 SMs (64 SP, 32 DP ALUs/SM) |
| Frequency | 2.2 - 3.6 GHz | 0.7 GHz | 1.328 - 1.480 GHz | 1.455 GHz |
| Use Mode | Latency-Oriented | Throughput-Oriented | Throughput-Oriented | Throughput-Oriented |
| Latency Treatment | Minimization | Toleration | Toleration | Toleration |
| Programming Model | 10s of threads | 10,000s+ of threads | 10,000s+ of threads | 10,000s+ of threads |
| Peak Performance | 633.6 GF/s (DP) | 1,165 GF/s (DP), ~3.5 TF/s (SP) | 5.3 TF/s (DP), ~10.6 TF/s (SP) | 7.5 TF/s (DP), ~15 TF/s (SP) |
| Memory Bandwidth | 76.8 GB/s (DDR4) | 250 GB/s (GDDR5) | 720 GB/s (HBM2) | Not listed, but high |
| Memory Capacity | 1.54 TB | 5 GB | 16 GB | 32 GB |
| Die Size | 456 mm² | 550 mm² | 610 mm² | 815 mm² |
| Transistor Count | 7.2 billion | 7.1 billion | 15.3 billion | 21.1 billion |
| Technology | 14nm | 28nm | 16nm FinFET | 12nm FFN |
| Power | 145 W | 250 W | 300 W | 300 W |
| Power Efficiency | ~4.37 GF/Watt (DP) | ~4.66 GF/Watt (DP) | ~17.66 GF/Watt (DP) | ~25 GF/Watt (DP) |

### 1.3 The Memory Hierarchy Divide

The memory systems of CPUs and GPUs are also distinct. GPUs pair their massive computational power with extremely high-bandwidth memory systems to prevent starvation. The on-chip memory hierarchy, including very large register files and software-managed shared memories, is crucial for performance.

Figure (from slides): Memory Hierarchy Comparison. The slides depict a comparison of memory hierarchies for an Intel Sandy Bridge CPU and several generations of NVIDIA GPUs (GK110, GP100, GA100). The data indicates a clear trend: GPUs have significantly larger and higher-bandwidth on-chip memory resources (registers, shared memory) and vastly higher main memory bandwidth compared to contemporary CPUs.

| Level | Intel Sandy Bridge (CPU) | GK110 (GPU) | GP100 (GPU) | GA100 (GPU) |
| --- | --- | --- | --- | --- |
| Registers | ~1 kB (5 TB/s) | ~4 MB (40 TB/s) | 14 MB | 32 MB |
| L1 / SM Memory | 512 kB | ~1 MB | ~4 MB | 24 MB |
| L2 / LLC | 8 MB (500 GB/s) | 1.5 MB (500 GB/s) | 4 MB | 40 MB |
| Main Memory | Terabytes (20 GB/s) | 4 GB (150 GB/s) | 16 GB (800 GB/s) | 48 GB (1.9 TB/s) |

**Key Takeaway:** GPUs achieve superior performance and energy efficiency on parallel workloads by trading single-thread performance for massive throughput. This is accomplished through an architecture that dedicates most of its silicon to simple, replicated processing cores and is supported by a high-bandwidth memory system designed to feed them.

### Exercises

1. Explain the primary reason GPUs were historically well-suited for transitioning from graphics rendering to general-purpose computing.
2. Using the provided table, calculate the ratio of peak single-precision (SP) to double-precision (DP) performance for the Kepler, Pascal, and Volta GPUs. What trend do you observe?
3. Describe the key differences in the memory hierarchy between the Sandy Bridge CPU and the GA100 GPU in terms of capacity and bandwidth at different levels.

---

## 2. Performance Scaling and Its Fundamental Limits

The drive toward parallel architectures like GPUs is a direct consequence of the breakdown of traditional processor scaling models. Understanding these models and their limitations is essential to appreciating why GPU architectures are designed the way they are.

### 2.1 Moore's Law and Its Evolution

Definition 1. Moore's Law. An observation made by Gordon Moore. The 1975 revision states that the number of transistors on an integrated circuit doubles approximately every two years.

This exponential growth has driven the semiconductor industry for decades. Derived "laws" suggested that CPU performance would double every 18 months and memory capacity would quadruple every three years.

Today, Moore's Law is still considered "alive," as transistors continue to shrink (approaching 3nm). However, the industry faces significant challenges:

* Physical Limits: The end of silicon scaling is in sight.
* Economic Limits: Prototyping advanced nodes is different from mass production.
* Design Response: To continue increasing transistor counts, chips are becoming larger (often limited by the reticle size) and new paradigms like chiplets (assembling multiple smaller dies) are being adopted.

### 2.2 The Post-Dennard Scaling Era

Classical Dennard scaling observed that as transistors shrank, their power density remained constant, allowing frequency and performance to increase without a corresponding increase in power consumption. This "free lunch" ended in the mid-2000s.

The governing equation for dynamic power is: `P = afCV^2 + V I_{leakage}` where a is the activity factor, f is frequency, C is capacitance, and V is voltage. As frequency scaling stalled, architects sought performance elsewhere.

This led to a transition from complex, latency-minimizing microarchitectures to massively parallel ones.

* Pre-Transition (CPU): Performance was sought by increasing Instructions Per Cycle (IPC) through speculative and out-of-order execution, deep pipelines, and branch prediction. This approach is energy-intensive.
* Post-Transition (GPU): Performance is achieved through massive replication of simpler, energy-efficient, in-order cores. Frequency is often reduced to maintain power efficiency.

Performance scaling shifted from Regime I (IPC × Frequency) to Regime II (Power × Efficiency), where efficiency is measured in Operations/Joule. This favors specialization and architectural heterogeneity.

### 2.3 Amdahl's Law: The Ceiling on Speed-up

While adding more processors seems like a straightforward way to increase performance, the achievable speed-up is limited by the portion of the code that cannot be parallelized.

Definition 2. Speed-up and Efficiency. Given a problem of size M, the speed-up (a) achieved by using N parallel units is the ratio of the best serial execution time to the parallel execution time. $a = \frac{t_{old}}{t_{new}} = \frac{t_{ser}(M)}{t_{par}(M, N)}$ A speed-up of a(N) = N is considered linear. A speed-up of a(N) > N is superlinear and typically arises from caching effects where the larger aggregate cache of a parallel system reduces memory latency. The efficiency (e) measures how well the parallel resources are utilized. $e = \frac{t_{ser}(M)}{N \cdot t_{par}(M, N)}$

Theorem 1. Amdahl's Law. The maximum speed-up of a program is limited by its serial fraction. If a fraction p of a program's execution time can be parallelized and the remaining fraction s = 1-p is purely serial, the speed-up on N processors is: $a = \frac{1}{s + \frac{p}{N}} = \frac{1}{(1 - p) + \frac{p}{N}}$ As the number of processors $N \to \infty$, the term $\frac{p}{N} \to 0$, and the maximum speed-up is limited by: $\lim_{N\to\infty} a = \frac{1}{s} = \frac{1}{1-p}$

Key Insight: Gene Amdahl's key insight was that speed-up is fundamentally bounded by the serial fraction s, not the number of processors N. For example, if 10% of a program is serial (s=0.1), the maximum possible speed-up is 1/0.1 = 10×, even with an infinite number of processors.

Amdahl's argument was originally used to claim that the single-processor approach was superior. However, his model is optimistic (it ignores parallel overheads like communication and synchronization) and pessimistic (it does not account for scaling the problem size with the number of processors, as described by Gustafson's Law).

**Key Takeaway:** The end of Dennard scaling forced a move to parallel architectures. Amdahl's Law provides a crucial, sobering model for the limits of this approach: to achieve significant speed-up, applications must be overwhelmingly parallel.

### Exercises

1. A program spends 20% of its time on serial operations. According to Amdahl's Law, what is the maximum theoretical speed-up that can be achieved on a massively parallel processor?
2. If an application achieves a speed-up of 16 on 32 processors, what is its parallel efficiency?
3. Explain why the end of Dennard scaling made architectures like GPUs more attractive for high-performance computing.

---

## 3. GPU Architectural and Programming Models

A modern GPU can be understood through two complementary perspectives: the abstract software model presented to the programmer and the concrete hardware implementation that executes the code.

### 3.1 A Dual View of the GPU

1. Software View: A Programmable Many-Core Scalar Architecture (SIMT) From the programmer's perspective, a GPU is a device that executes a single program on a huge number of scalar threads. This model is called Single Instruction, Multiple Threads (SIMT). Each thread has its own state (program counter, registers) and can execute an independent control flow path. This model operates in lock-step and is designed to exploit parallel slackness—having far more threads ready to run than there are physical execution units.
2. Hardware View: A Programmable Multi-Core Vector Architecture (SIMD) From the hardware's perspective, a GPU is a collection of multi-core processors (SMs), where each core contains wide vector execution units. To achieve efficiency, the hardware bundles scalar threads from the SIMT model into groups (typically called warps) and executes them in lock-step on these vector units. This is an implementation of Single Instruction, Multiple Data (SIMD). The hardware manages the execution of these groups, creating the illusion of independent scalar threads for the programmer. In essence, a GPU is a vector architecture that hides its vector units.

Figure (from slides): Massively Parallel Microarchitecture. The slides show a schematic of a GPU Streaming Multiprocessor (SM). It depicts multiple "warp schedulers" that dispatch instructions to a large collection of simple, replicated cores (for floating-point, integer, load/store, and special functions). These cores share resources like a register file, shared memory, and L1 cache. This contrasts sharply with the diagram of a complex CPU core, which features extensive logic for out-of-order execution, speculation, and branch prediction.

### 3.2 The Bulk-Synchronous Parallel (BSP) Model

In 1990, Leslie Valiant described the Bulk-Synchronous Parallel (BSP) model, which serves as an excellent theoretical framework for understanding GPU execution. The BSP model organizes computation into a sequence of supersteps. Each superstep consists of three phases:

1. Compute: All processors perform local computations concurrently.
2. Communicate: Processors exchange necessary data.
3. Synchronize: A barrier synchronization ensures all processors have completed the superstep before proceeding to the next.

A key concept in Valiant's model is parallel slackness, where the number of virtual processors (v) is much larger than the number of physical processors (p), i.e., $v \gg p$. This slack allows the system to hide the latency of communication and synchronization by scheduling computation from other virtual processors. The SIMT model, with its tens of thousands of threads, is a near-perfect incarnation of this principle.

### 3.3 Summary of the GPU Computing Paradigm

GPU computing leverages GPUs for non-graphical tasks to achieve superior performance and energy efficiency.

Key Characteristics and Differences from CPUs:

* Parallelism: Employs vastly more parallelism (tens of thousands of threads vs. tens).
* Latency: Tolerates memory latency through massive multithreading, rather than minimizing it with large caches.
* Model: Uses an offload compute model, where the CPU (host) manages data and dispatches parallel computations (kernels) to the GPU (device).
* Limitations:
  * Single-thread performance is very low.
  * On-board memory capacity is typically smaller than system RAM.
  * Not yet a fully general-purpose programming environment (though this is evolving).

### Exercises

1. Explain the difference between the SIMT and SIMD models of parallelism. How does a GPU use both concepts?
2. What is "parallel slackness" and how does it help a GPU tolerate memory latency?
3. Describe a "superstep" in the Bulk-Synchronous Parallel (BSP) model and relate it to how a GPU kernel might execute.

---

# GPU Computing: Architecture and Programming

## Executive Summary

This document provides a comprehensive overview of GPU architecture and the CUDA programming model, synthesizing foundational concepts for high-throughput computing. The central theme is the architectural divergence between CPUs, which are latency-optimized, and GPUs, which are throughput-optimized. This distinction is visually represented by their die layouts: CPUs dedicate significant silicon to large caches and complex control logic for single-thread performance, whereas GPUs dedicate the vast majority of their area to a massive number of parallel processing cores.

The performance disparity is stark. A representative high-end CPU like the AMD EPYC 9754 delivers approximately 6 TFLOP/s of double-precision performance with 460 GB/s of memory bandwidth. In contrast, a high-end GPU like the NVIDIA H100 SXM offers 34-67 TFLOP/s (DP) and a staggering 3.3 TB/s of memory bandwidth. This massive parallelism is harnessed to hide the high latency of memory access; while one group of threads waits for data, the hardware schedules other groups to perform computation, ensuring the processing units remain saturated.

The CUDA programming model provides a C-based framework to explicitly manage this parallelism. It is built on three core abstractions: a hierarchical organization of threads, a software-managed on-chip shared memory, and barrier synchronization primitives. A CUDA program is partitioned into a host (CPU) component, which handles sequential logic and orchestrates data transfers, and a device (GPU) component, which consists of highly parallel functions called kernels.

A kernel is executed by a grid of thread blocks. Each block contains a group of threads that can cooperate via fast shared memory and synchronize their execution using barriers. This hierarchical model allows for scaling computation across problems of varying sizes. Programmers manage the GPU's distinct memory spaces: large but high-latency global memory for bulk data, low-latency on-chip shared memory for inter-thread cooperation within a block, and private registers for each thread. Effective CUDA programming hinges on meticulous management of data movement between these memory spaces and structuring computation to maximize parallel execution while respecting the resource constraints of the hardware. The SAXPY case study demonstrates how a simple for loop on a CPU is transformed into a parallel kernel where each thread independently processes a single data element, showcasing the Single Program, Multiple Data (SPMD) paradigm that underpins GPU computing.

## Notation & Acronyms

| Term | Definition |
| --- | --- |
| CPU | Central Processing Unit |
| CUDA | Compute Unified Device Architecture; NVIDIA's parallel computing platform. |
| DLP | Data-Level Parallelism |
| GPU | Graphics Processing Unit |
| ISA | Instruction Set Architecture |
| PTX | Parallel Thread Execution; a virtual machine and ISA for GPUs. |
| SAXPY | Single-Precision A-X Plus Y; a common vector operation (y = \alpha x + y). |
| SM | Streaming Multiprocessor; a core computational unit on a GPU. |
| SPMD | Single Program, Multiple Data; an execution model where all processors run the same program on different data. |
| TLP | Thread-Level Parallelism |

---

## 1. Motivation: Throughput vs. Latency

The fundamental difference between CPUs and GPUs lies in their design philosophy. CPUs are optimized for low latency on sequential tasks, dedicating significant chip area to complex control logic and large caches to minimize the execution time of a single thread. In contrast, GPUs are designed for high throughput, dedicating most of their silicon to a vast number of simpler arithmetic logic units (ALUs).

This architectural divergence results in dramatic differences in theoretical performance.

Table 1: Comparison of a High-End CPU and GPU

| Metric | CPU (AMD EPYC 9754) | GPU (NVIDIA H100 SXM) |
| --- | --- | --- |
| Peak FP64 Perf. | ~6 TFLOP/s | 34-67 TFLOP/s |
| Peak FP32 Perf. | N/A | 67-490 TFLOP/s |
| Memory Bandwidth | 460 GB/s | 3.3 TB/s |

The GPU achieves its performance by executing thousands of threads concurrently. This massive thread-level parallelism (TLP) is used to hide memory latency: when one group of threads stalls waiting for data from global memory, the GPU scheduler simply switches to another group that is ready to execute, keeping the computational units busy.

## 2. GPU Microarchitecture

Early GPUs were designed for graphics pipelines. The NVIDIA G80 architecture marked a significant step toward general-purpose computing by organizing the hardware around a scalable array of Streaming Multiprocessors (SMs).

Figure (from slides): G80 Architecture for General-Purpose Processing This diagram illustrates the core components of a general-purpose GPU. A host interface connects the GPU to the CPU via a PCIe bus. On the device, a Thread Execution Manager dispatches work to an array of SMs. Each SM has access to a parallel data cache and can perform load/store operations to a large, shared Global Memory.

* Streaming Multiprocessor (SM): The primary computational unit of the GPU. A GPU contains many SMs, each capable of executing hundreds of threads concurrently.
* Global Memory: The main device memory, accessible by all SMs and the host. It is large (gigabytes) but has high latency.

## 3. The CUDA Programming Model

CUDA is a C-based programming model that exposes the GPU's parallelism to the programmer. A CUDA program consists of two parts:

1. Host Code: Runs on the CPU. It is typically responsible for sequential logic, memory management, and launching computations on the device.
2. Device Code: Runs on the GPU. This code, written in functions called kernels, executes in parallel across many threads.

This model is built upon three key abstractions:

1. A Hierarchy of Threads: A scalable model for organizing parallel work.
2. Shared Memory: A low-latency, on-chip memory space for cooperating threads.
3. Barrier Synchronization: A mechanism to coordinate threads within a functional group.

### 3.1 Thread Hierarchy

CUDA organizes threads into a three-level hierarchy:

1. Thread: The basic unit of execution. Each thread has a private set of registers and executes an instance of the kernel function.
2. Thread Block (or Block): A group of threads that execute on a single SM. Threads within a block can cooperate by sharing data through a fast, on-chip shared memory and can synchronize their execution.
3. Grid: A collection of all thread blocks that execute a single kernel launch. Blocks within a grid are executed independently and can run in any order.

Figure (from slides): CUDA Thread Hierarchy This diagram shows a Grid composed of multiple Blocks. Each Block is a 3D array of Threads. A thread is uniquely identified within the entire grid by its block index and its index within that block.

Threads and blocks are identified using built-in, multi-dimensional index variables:

* threadIdx.{x,y,z}: The index of a thread within its block.
* blockIdx.{x,y,z}: The index of a block within its grid.
* blockDim.{x,y,z}: The dimensions of the thread block.

### 3.2 Kernel Definition and Launch

A kernel is a C function that is executed on the GPU by N threads. It is defined using the `__global__` specifier.

```cpp
// Defines a kernel function
__global__ void myKernel(float* data) {
    // Kernel body executed by each thread
}
```

The host launches a kernel using a special `<<<...>>>` syntax, which specifies the grid and block dimensions.

```
kernel_name <<< gridDim, blockDim >>> (argument_list);
```

* gridDim: The number of blocks in the grid (e.g., `dim3(10, 1, 1)`).
* blockDim: The number of threads in each block (e.g., `dim3(256, 1, 1)`).

Example 1: Vector Addition Kernel The following kernel adds two vectors A and B, storing the result in C. Each thread is responsible for computing one element of the output vector.

```cpp
// A kernel for vector addition that can handle inputs of any size N.
__global__ void vecAdd(float* A, float* B, float* C, int N) {
    // Calculate the global index of the thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to ensure no out-of-bounds access
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000;
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed to cover all N elements
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}
```

Key Takeaway: Global Index Calculation A thread's unique global index is computed from its block and thread indices. For a 1D grid of 1D blocks, the formula is: $\text{globalID} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$ This allows a single kernel code (SPMD) to operate on different data elements by mapping each thread to a specific data index. The `if (i < N)` guard is crucial for handling problem sizes that are not an exact multiple of the block size.

### 3.3 Thread Communication and Synchronization

* Within a Block: Threads within the same block can communicate via shared memory and atomic operations. They can synchronize their execution using `__syncthreads()`, which acts as a barrier.
* Between Blocks: Threads from different blocks cannot directly communicate or synchronize. Blocks execute independently. Communication between blocks requires writing results to global memory, terminating the current kernel, and launching a subsequent kernel to read those results.

### 3.4 Sizing Recommendations

* Threads per Block: Typically between 100 and 1000. This range is large enough to provide concurrency for latency hiding but small enough to not over-consume resources (registers, shared memory) on an SM.
* Blocks per Grid: Should be at least twice the number of SMs on the GPU to ensure all SMs are kept busy.

### Exercises

1. Conceptual: Why are threads from different blocks unable to synchronize using a barrier?
2. Calculation: A 2D matrix of size 2000x3000 needs to be processed. If you use a 2D thread block of size 16x16, what are the `dimGrid` dimensions you would specify in the kernel launch?
3. Applied: Write the global index calculation for a 2D grid of 2D blocks, assigning one thread to each element of a 2D matrix.

---

## 4. GPU Memory Hierarchy

Effective CUDA programming requires explicit management of data across several distinct memory spaces.

Figure (from slides): CUDA Memory Hierarchy This diagram illustrates the memory spaces available to CUDA threads. Each thread has its own private registers. Each thread block has a shared memory space accessible to all threads within that block. All threads in the grid can access the large global memory. The host CPU has its own memory, separate from the device's global memory.

### 4.1 Global Memory

* Scope: Accessible by all threads in the grid and by the host (via API calls).
* Lifetime: Application lifetime.
* Characteristics: Large (gigabytes), but has very high latency. Accesses are sensitive to memory access patterns (coalescing). It is the primary medium for transferring data between the host and device.

API for Global Memory Management:

* `cudaMalloc(&d_ptr, size)`: Allocates memory on the device.
* `cudaFree(d_ptr)`: Frees device memory.
* `cudaMemcpy(dst, src, size, type)`: Transfers data between host and device. The type can be `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`.

### 4.2 Shared Memory

* Scope: Accessible only by threads within a single block.
* Lifetime: Thread block lifetime.
* Characteristics: On-chip, very low latency (approaching register speed in the best case). It is a software-managed cache, ideal for data reuse and inter-thread communication. Shared memory is organized into banks; concurrent accesses to different banks can proceed in parallel, but accesses to the same bank (a "bank conflict") are serialized, degrading performance.
* Declaration: Declared within a kernel using the `__shared__` specifier.

### 4.3 Registers

* Scope: Private to a single thread.
* Lifetime: Thread lifetime.
* Characteristics: Fastest memory space. Used for local variables. The number of available registers per SM is limited.

### 4.4 Variable and Function Specifiers

CUDA uses specifiers to declare the location of variables and the execution space of functions.

Table 2: Variable Declaration Specifiers

| Specifier | Location | Access From | Lifetime |
| --- | --- | --- | --- |
| `__device__ float var;` | Global Memory | Device / Host | Program |
| `__constant__ float var;` | Constant Memory | Device / Host | Program |
| `__shared__ float var;` | Shared Memory | Threads | Thread Block |
| `texture <float> ref;` | Texture Memory | Device / Host | Program |

Table 3: Function Declaration Specifiers

| Declaration | Executed on | Callable from |
| --- | --- | --- |
| `__global__ void KernelFunc()` | Device | Host |
| `__device__ float DeviceFunc()` | Device | Device |
| `__host__ float HostFunc()` | Host | Host |

`__host__` and `__device__` can be combined to compile a function for both the CPU and GPU. Device functions have limitations, including no support for recursion, variable argument counts, or non-static variable declarations.

Key Takeaway: Memory Consistency CUDA has a relaxed memory consistency model. Writes to shared or global memory by one thread are not guaranteed to be visible to other threads without explicit synchronization. The `__syncthreads()` intrinsic acts as a barrier and a memory fence, ensuring that all memory writes made by threads in a block are visible to all other threads in the same block after the barrier.

---

## 5. CUDA Compilation and Execution

CUDA code is compiled using the NVIDIA CUDA Compiler (`nvcc`).

1. `nvcc` separates host code from device code.
2. Host code is compiled by a standard C++ compiler (like `g++`, `clang`).
3. Device code (`__global__` and `__device__` functions) is compiled into PTX (Parallel Thread Execution), a virtual ISA for GPUs.
4. The PTX code is then further compiled, either ahead-of-time or just-in-time (JIT) by the driver, into the native binary for the target GPU architecture.
5. Finally, `nvcc` links the compiled host code, the device code, and the CUDA runtime libraries (`cudart`) into a single executable.

## 6. Device Properties

Different GPUs have vastly different capabilities. The `deviceQuery` utility provides a survey of a device's properties.

Table 4: Properties of Various NVIDIA GPU Architectures

| Property | GeForce GTX 480 (CC 2.0) | Tesla K20c (CC 3.5) | RTX 2080Ti (CC 7.5) |
| --- | --- | --- | --- |
| Global Memory | 1.5 GB | 5 GB | 11 GB |
| Multiprocessors (SMs) | 15 | 13 | 68 |
| Cores (Total) | 480 | 2496 | 4352 |
| Shared Memory / Block | 48 KB | 48 KB | 48 KB |
| Registers / Block | 32k | 64k | 64k |
| Warp Size | 32 | 32 | 32 |
| Max Threads / Block | 1024 | 1024 | 1024 |
| Max Grid Dim | 65535 x 65535 x 65535 | 2G x 65535 x 65535 | N/A |
| Concurrent Copy & Exec | Yes (1 engine) | Yes (2 engines) | Yes (3 engines) |

## 7. Case Study: SAXPY

SAXPY (Scalar Alpha X Plus Y) is a simple vector operation defined as $y_i = \alpha \cdot x_i + y_i$. It serves as a good introductory example for comparing CPU and GPU implementations.

Serial CPU Implementation:

```cpp
// Kernel function (CPU)
void saxpy_serial(int n, float alpha, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i] + y[i];
    }
}
```

Parallel CUDA Implementation:

```cpp
// Kernel function (CUDA device)
__global__ void saxpy_parallel(int n, float alpha, float *x, float *y) {
    // Compute the global index from thread and block IDs
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid writing past the allocated memory
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}
```

Performance Considerations

Initial performance analysis shows a huge advantage for the GPU when data transfer times are excluded. However, the cost of moving data between host and device memory can be a significant bottleneck.

Pinned Memory: By default, host memory allocated with `malloc` is pageable. The operating system can move this memory in physical RAM. For DMA transfers to the GPU, the driver must first copy this data to a temporary, pinned (non-pageable) buffer. This extra copy adds overhead.

Using `cudaMallocHost()` allocates pinned memory directly, allowing the GPU to transfer data without the intermediate copy, significantly reducing data movement costs. However, pinned memory is a scarce system resource and should be used judiciously.

```cpp
float *h_x;
// Allocate pinned host memory instead of using malloc
cudaMallocHost((void**)&h_x, N * sizeof(float));
```

## 8. Common Errors and Troubleshooting

* CUDA Error: the launch timed out and was terminated: The kernel took too long to execute. This can happen with infinite loops or on systems where the GPU is also driving a display (a watchdog timer kills long-running kernels). On dedicated systems, stopping the X11 server can help.
* CUDA Error: unspecified launch failure: Often indicates a segmentation fault within the kernel, such as an out-of-bounds memory access.
* CUDA Error: invalid configuration argument: The kernel launch parameters (`gridDim`, `blockDim`) are invalid. Common causes include requesting more threads per block than the device supports (e.g., > 1024) or requesting more resources (shared memory, registers) per block than are available on an SM.
* Compiler Error: identifier "__eh_curr_region" is undefined: This can occur when using dynamically sized shared memory with a C++ compiler. Using statically allocated shared memory often resolves it.

## 9. Summary

The transition from CPU to GPU programming represents a shift in paradigm. While it offers direct control over powerful hardware and immense potential for parallelism, it also increases the programmer's burden. Key differences include:

* Sophisticated Resource Planning: Programmers must carefully manage the constraints on registers, shared memory, and thread counts to achieve optimal performance.
* Manual Data Movements: Data must be explicitly transferred between the host and device, and between different levels of the device memory hierarchy.
* Limited Memory Capacity: GPU global memory is typically smaller than host system RAM.

Understanding these concepts provides a strong foundation for harnessing the massive computational power of modern GPUs.

## Final Exercises

1. Conceptual: The slides ask, "Did you see any vector instructions today?" The SAXPY kernel uses simple scalar arithmetic ($y[i] = \alpha*x[i] + y[i]$). How does the GPU achieve parallelism without explicit vector instructions in the code?
2. Applied: Describe the complete sequence of CUDA API calls required to run the `saxpy_parallel` kernel on a vector of 1 million floats, including memory allocation, data transfer, kernel execution, and result retrieval.
3. Performance: Why is pinned memory faster for host-to-device data transfers than standard pageable memory? What is the main drawback of using it?
