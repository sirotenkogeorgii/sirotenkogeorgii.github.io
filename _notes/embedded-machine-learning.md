---
layout: default
title: Embedded Machine Learning
date: 2025-10-16
excerpt: ...
tags:
  - machine-learning
  - high-performance-computing
---

<!-- This study book serves as a comprehensive guide to the introductory concepts of Embedded Machine Learning, designed for students with no prior background in the field. It synthesizes the core ideas from the initial lecture, explaining all fundamental concepts, terminology, and mathematical principles from the ground up. -->

# Embedded Machine Learning


## 1. Introduction to Embedded Machine Learning

This course explores the intersection of state-of-the-art Deep Neural Networks (DNNs) and resource-constrained embedded devices. A central theme is the challenge of not only making complex models run on resource-constraint embedded devices but also embedding these models in the real world, which necessitates a robust understanding and treatment of uncertainty and resource-efficient deep neural networks.


---


## 2. The Landscape of Modern Machine Learning

### The Challenge: Mismatch Between ANNs and Embedded Hardware

The scale of data used to train these models is massive and growing. The complexity ranges from smaller datasets that can be trained on a laptop to enormous collections that require supercomputing resources.

There is an extreme mismatch between the computational demands of modern ANNs and the capabilities of mobile or embedded processors. Large models like ResNet-50, which perform well on high-power servers, are difficult to deploy on devices with strict power and memory constraints.

### The Hardware Lottery Hypothesis

This mismatch leads to a concept known as the Hardware Lottery Hypothesis.

The "Hardware Lottery" suggests that tooling has played a disproportionately large role in deciding which ideas succeed and which fail. The hardware that is readily available and highly optimized (like GPUs for matrix operations) determines which research directions thrive.

Because ANNs fundamentally rely on matrix-matrix operations, they perform exceptionally well on GPUs, which are designed for exactly this kind of computation. As a result, most ML researchers tend to ignore hardware constraints and focus on architectures that fit this paradigm, such as Convolutions and Transformers. This has led to massive models like GPT-3 (175B parameters, 800GB of state) and AlphaFold-2 (23TB of training data).

This raises an important question: what if a different type of processor existed, e.g. one that excelled at processing large graphs? This could have led to the dominance of alternative models like probabilistic graphical models, sum-product networks, or graph neural networks. 

**Processor specialization is considered harmful for innovation because it can prevent alternative algorithmic approaches from being explored.**


---


## 3. Fundamentals of Supervised Learning: Regression

This section covers the foundational concepts of machine learning, including learning, generalization, model selection, regularization, and overfitting, using the example of regression.

### Introduction to Supervised Learning

In supervised learning, the goal is to learn a predictive function from a labeled dataset. We are given a set of examples and asked to predict an output for new, unseen data.

Supervised Learning Problem: Given a training set of $N$ samples, $(x^{(i)}, t^{(i)})$, find a good prediction function, $y = h_{\theta}(x)$, that can generalize to new data.

#### Key Terminology

* $x^{(i)}$: The input features of the $i$-th training sample.
* $t^{(i)}$: The target variable (or label) of the $i$-th sample.
* $(x^{(i)}, t^{(i)})$: A single training sample or observation.
* Training Set: The complete set of $N$ training samples.
* $h_{\theta}(x)$: The prediction function (or hypothesis) we are trying to find (learn).
* $\theta$: The parameters (or weights) of the model that the learning algorithm will adjust.

Problems in supervised learning can be categorized as:

* Classification: When the target variable $t^{(i)}$ is discrete (e.g., 'cat', 'dog', 'bird').
* Regression: When the target variable $t^{(i)}$ is continuous (e.g., the price of a house).

### Linear Regression

The simplest form of regression is linear regression. Here, we assume the relationship between the input features and the output is linear. For an input $x$ with $D$ features, the model is:

$$ h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_Dx_D $$

This can be written more compactly using vector notation. By setting $x_0 = 1$, we can absorb the $\theta_0$ term (known as the model intercept or bias):

$$ h_{\theta}(x) = \sum_{d=1}^{D} \theta_d x_d = \theta^T x $$

The goal of learning is to find the optimal parameters $\theta$ that make our model's predictions, $h(x)$, as close as possible to the true target values, $t$, for all $N$ samples in our training set. To measure "how close" we are, we use a cost function (also known as an error function or loss function).

A common choice is the sum of squared residuals, which forms the basis of the least-squares method. The cost function for linear regression is defined as:

$$ \mathcal{L}(\theta) = \frac{1}{2} \sum_{n=1}^{N} \left(h_{\theta}\left(x^{(n)}\right) - t^{(n)}\right)^2 $$

The $\frac{1}{2}$ factor is included for mathematical convenience to simplify the derivative later.

#### Minimizing Error with Gradient Descent

To find the best parameters $\theta$, we need to find the values that minimize the cost function $\mathcal{L}(\theta)$. The most common algorithm for this is gradient descent.

Gradient Descent is an iterative optimization algorithm that starts with an initial guess for $\theta$ and repeatedly adjusts the parameters in the direction that most steeply decreases the cost function.

The update rule for each parameter $\theta_d$ is:

$$ \theta_d := \theta_d - \alpha \frac{\partial}{\partial\theta_d} \mathcal{L}(\theta) $$

Here, $\alpha$ is the learning rate. The term $\frac{\partial}{\partial\theta_d} \mathcal{L}(\theta)$ is the partial derivative of the cost function, which tells us the slope (or gradient) of the cost function with respect to that parameter.

Using the chain rule:

$$ \frac{\partial}{\partial\theta_d} \mathcal{L}(\theta) = \sum_{n=1}^{N} (h_{\theta}(x) - t)x_d $$

Plugging this back into the update rule gives us the final update equation:

$$ \theta_d := \theta_d + \alpha \sum_{n} \left(t^{(n)} - h_{\theta}\left(x^{(n)}\right)\right)x_d^{(n)} $$

This rule shows that the magnitude of the update for each parameter is proportional to the error term $\left(t^{(n)} - h_{\theta}\left(x^{(n)}\right)\right)$.

#### Batch vs. Stochastic Gradient Descent

There are different strategies for deciding which training samples to use for each update step.

1. **Batch Gradient Descent**
    - In each step, the algorithm looks at every training sample ($\forall n \in N$) to compute the gradient.
    - The update rule is: $\theta_d := \theta_d + \alpha \sum_{n=1}^{N} \left(t^{(n)} - h_{\theta}\left(x^{(n)}\right)\right)x_d^{(n)}$ for all parameters $d$.
    - This is repeated until the algorithm converges.
    - Advantage: Guaranteed to find the optimal solution for convex functions like our linear regression cost function.
    - Disadvantage: Very expensive computationally, as the entire dataset must be processed for a single update.
2. **Stochastic (or Incremental) Gradient Descent (SGD)**
    - Instead of the entire dataset, the algorithm randomly selects one training sample at a time to perform an update.
    - The update rule is performed for each sample $n$ in the dataset: $\theta_d := \theta_d + \alpha\left(t^{(n)} - h_{\theta}\left(x^{(n)}\right)\right)x_d^{(n)}$ for all $d$.
    - Advantage: Makes progress much faster, as parameters are updated after every single sample. It is much more efficient for large datasets.
    - Disadvantage: The path to the minimum is "noisier" and may not converge to the exact minimum, but it usually gets very close.

[Placeholder: Two graphs comparing Batch vs. Stochastic Gradient Descent. The first shows Batch GD taking a smooth, direct path to the minimum of a cost function contour plot. The second shows SGD taking a noisy, zigzagging path toward the same minimum.]

### Polynomial Curve Fitting

Consider a training set of $N$ observations of an input $x$ and a target $t$. Our goal is to predict new values $\hat{y}$ for new inputs $\hat{x}$.

Our model can be a polynomial of order $M$:

$$ h(x, w) = w_0 + w_1x + w_2x^2 + \dots + w_Mx^M = \sum_{m=0}^{M} w_m x^m $$

Although this function is nonlinear with respect to the input $x$, it is still a linear model because it is a linear function of its coefficients $w$.

$$ \mathcal{L}(w) = \frac{1}{2} \sum_{n=1}^{N} \left(h(x_n, w) - t_n\right)^2 $$

Because this is a quadratic function of the coefficients $w$, it has a unique solution $w^*$.

#### The Problem of Overfitting and the Importance of Generalization

A critical decision in polynomial fitting is choosing the order of the polynomial, $M$. This is a problem of model selection.

Generalization refers to a model's ability to make accurate predictions for new, unseen data. A model that generalizes well has learned the underlying pattern in the data, not just the noise.

Overfitting occurs when a model is too complex (e.g., a high-order polynomial $M$) and starts to fit the random noise in the training data instead of the true underlying relationship. Such a model performs very well on the training data but fails to generalize to new data.

To identify overfitting, we split our data into a training set and a test set. The model is trained only on the training set. We then evaluate its performance on both sets.

* Training error: The error calculated on the training set.
* Test error: The error calculated on the test set.

We often use the Root-Mean-Square (RMS) Error to compare performance on datasets of different sizes:

$$ \mathcal{L}_{RMS} = \sqrt{\frac{2\mathcal{L}(w)}{N}} $$

A large gap between a low training error and a high test error is a clear sign of overfitting.

[Placeholder: Graph showing the relationship between Model "complexity" and Error. The y-axis is "Error" and the x-axis is "Model 'complexity'". The "Training error" curve consistently decreases as complexity increases. The "Test error" curve decreases initially, hits a minimum ("Best fit"), and then starts to increase, indicating the regions of "Underfitting" and "Overfitting".]

The ideal model complexity also depends on the size of the dataset. A more complex model can be justified if there is a sufficiently large amount of data to prevent it from overfitting.

#### Controlling Overfitting with Regularization

A common technique to control overfitting is regularization.

Regularization involves adding a penalty term to the cost function to discourage the model's coefficients from becoming too large. This "shrinks" the coefficients, leading to a simpler, smoother model that is less likely to overfit.

The regularized cost function is:

$$ \tilde{\mathcal{L}}(w) = \frac{1}{2} \sum_{n=1}^{N} \left(h(x_n, w) - t_n\right)^2 + \frac{\lambda}{2} \|w\|^2 $$

Where:

- $\lambda$ is the regularization parameter, which controls the relative importance of the penalty term.

This specific type of regularization is known as ridge regression, weight decay, or L2 regularization. Finding the optimal value for $M$ or $\lambda$ is typically done using a third dataset called a validation set.


---


## 5. Summary and Next Steps

### Key Takeaways

* This course aims to bridge the gap between complex Deep Neural Networks (DNNs) and the constraints of real-world hardware (HW).
* The linear regression example introduced fundamental concepts: learning model parameters, the importance of generalization and model selection, and the problems of overfitting and how to combat it with regularization.
* The Hardware Lottery hypothesis suggests that the prevalence of specialized hardware (like GPUs) can prevent algorithmic innovation by favoring models that fit the existing hardware paradigm.
* While linear models are foundational, they are not universal approximators—they cannot represent any arbitrary function.

### Looking Ahead: Artificial Neural Networks

Artificial Neural Networks (ANNs) are universal approximators. They have the capacity to learn extremely complex, non-linear relationships in data. However, this power comes at a price: increased complexity, a need for vast amounts of data and computation, and reduced interpretability. The following lectures will delve into the architecture and application of these powerful models.



## Embedded Machine Learning Lecture 02: Massively Parallel Architectures

This study guide provides a comprehensive overview of massively parallel architectures, focusing on the Graphics Processing Unit (GPU) as a key enabler for modern machine learning. We will explore the historical context, microarchitectural shifts, software models, and hardware components that define GPU computing.


---


### 1. Introduction to GPUs and their Evolution

#### The Origins of GPUs in Graphics

Initially, the primary application for Graphics Processing Units (GPUs) was in the gaming industry. Every gaming console contains a powerful GPU designed to render increasingly photorealistic graphics. The core task of computer graphics involves performing a vast number of multi-dimensional, floating-point operations in parallel, a workload for which GPUs are perfectly suited.

Over time, these graphics pipelines became programmable, opening the door for new applications. Around 2007, a significant shift occurred as developers began to leverage GPUs for general-purpose computing, a field now known as GPGPU. This was largely enabled by platforms like NVIDIA's CUDA (Compute Unified Device Architecture).

Despite their power, the specialized nature of GPUs led to some skepticism in the early days, exemplified by a quote attributed to a "known (sane) scientist" in 2013:

> GPUs are simply crippled processors.

#### The Shift from Single-Core Scaling to Parallelism

To understand why GPUs became so important, we must first look at the history of traditional processor design, which was driven by Moore's Law (the observation that the number of transistors on a chip doubles approximately every two years) and Dennard Scaling (which predicted that power density would remain constant as transistors got smaller).

#### The Era of Moore & Dennard Scaling

Traditional CPU microarchitectures were designed to maximize the performance of a single instruction stream. They relied on several key principles:

* More Transistors & Frequency Scaling: Use the increasing transistor budget to build more complex logic and increase the clock frequency.
* Latency Minimization & Hiding: Employ multiple, deep pipelines to break down instructions into smaller steps and hide the latency of slow operations.
* Exploiting Locality: Use caches to take advantage of spatial locality (accessing data near recently accessed data) and temporal locality (accessing the same data again soon).
* Aggressive Speculation: Since performance was limited by control flow (like branches) and data dependencies, CPUs use complex hardware to predict what will happen next. This includes:
  * Speculative fetch and speculative pre-fetch to get instructions and data from memory before they are officially needed.
  * Speculative execution to run instructions ahead of time.
  * Complex mechanisms like a Reorder Buffer and Reservation Stations to manage out-of-order execution and ensure correctness.

[Placeholder: Diagram illustrating a complex, speculative CPU microarchitecture driven by Moore's Law, showing components like instruction fetch, decode, reorder buffer, reservation stations, functional units, and caches.]

The Post-Dennard Transition

Around the mid-2000s, Dennard Scaling broke down. While transistors continued to shrink, it was no longer possible to increase clock frequencies without generating too much heat and consuming too much power. This led to a fundamental shift in processor design. The new paradigm, known as the Post-Dennard era, embraced massive parallelism.

Instead of building one extremely complex and fast core, the focus shifted to:

* Replication: Using the vast transistor budget to place many simpler, slower cores on a single chip.
* Energy Efficiency: Reducing the clock frequency and using simpler in-order pipelines (which execute instructions in the order they are received) to dramatically improve energy efficiency. The power consumption of a chip is related to frequency (\( f \)) and voltage (\( V \)) by the formula: \( P = afCV^2 + \frac{V I_{\text{leakage}}}{f^3} \)

This transition marks the move toward massively parallel microarchitectures, the very foundation of modern GPUs.

[Placeholder: Diagram of a massively parallel GPU microarchitecture, highlighting the replication of simple, in-order cores, warp schedulers, and a shared memory/L1 cache.]


---


### 2. The Power of Vector Architectures

GPUs are fundamentally vector architectures. This means they are designed to perform the same operation on multiple data elements simultaneously. This is achieved through a Vector Instruction Set Architecture (ISA), which has several key advantages.

* Compact: A single instruction can define operations on an entire vector (a collection of data points). This amortizes the cost of fetching, decoding, and issuing instructions, as one instruction does the work of many scalar instructions. It also reduces the frequency of branch instructions.
* Parallel: The operations on the elements of a vector are inherently data-parallel, meaning there are no dependencies between them. This eliminates the need for complex hardware to detect parallelism (unlike in complex CPUs) and allows the operations to be executed simultaneously on parallel data paths.
* Expressive: Vector instructions can describe regular memory access patterns, such as accessing a continuous block of memory. This allows the hardware to prefetch data efficiently or use wide, multi-banked memory to accelerate access. The high latency of accessing the first element can be amortized over the entire sequential pattern.

[Placeholder: Diagram showing a 4x SIMD example, where a single instruction stream is applied to a data pool by four parallel Processing Units (PUs).]

#### The GPU View: SIMT vs. SIMD

There are two ways to view a GPU's execution model:

1. **Software View (The Illusion):** From a programmer's perspective, a GPU appears as a programmable many-core scalar architecture. You write code for a single thread, and the GPU executes this code across a huge number of scalar threads. This model is called SIMT, or Single Instruction, Multiple Threads. The threads appear to run independently, operating in a lock-step fashion.
2. **Hardware View (The Reality):** In reality, the GPU hardware is a programmable multi-core vector architecture. The hardware takes the scalar threads written by the programmer and bundles them into compound units (called warps) that are executed on vector-like hardware. This model is SIMD, or Single Instruction, Multiple Data.

In essence, a GPU is a vector architecture that cleverly hides its vector units from the programmer, creating the powerful and easy-to-use illusion of countless independent scalar threads.


---


### 3. The GPU Software and Programming Model

The GPU programming model is designed to leverage massive parallelism efficiently, a concept described by Leslie Valiant's Bulk-Synchronous Parallel (BSP) model back in 1990.

#### Bulk-Synchronous Parallelism and Parallel Slackness

The BSP model organizes computation into a series of supersteps. Each superstep consists of three phases:

1. Compute: All processors perform computations locally.
2. Communicate: Processors exchange necessary data.
3. Synchronize: All processors wait at a barrier until every processor has finished the communication phase.

A key concept here is parallel slackness, which is the ratio of virtual processors (v) to physical processors (p). Valiant argued that for optimal performance and scalability, it's best to have far more virtual processors than physical ones (v >> p). This "slack" gives the hardware schedulers the flexibility to pipeline computation and communication efficiently, hiding latencies and keeping the physical cores busy. While this model is extremely scalable, it can be inefficient for unbalanced workloads.

#### Collaborative Computing and Memory Access

The core philosophy of GPU programming is collaboration.

If you do something on a GPU, do it collaboratively with all threads.

* Collaborative Computation: The typical approach is to assign one thread to compute one output element.
* Collaborative Memory Access: Similarly, memory access is structured with one thread per data element.

The GPU's schedulers exploit the previously mentioned parallel slackness to manage these thousands of threads, ensuring that the hardware is effectively utilized for both computation and memory transfers.

[Placeholder: Diagram illustrating the concept of GPU collaborative computing, where a large output data set is computed by many threads, which then access memory (GDDR) collaboratively.]

#### Thread Synchronization

Coordinating thousands of threads requires efficient synchronization mechanisms.

* Barrier Synchronization (within a Thread Block): Threads are organized into groups called thread blocks (or Cooperative Thread Arrays, CTAs). Within a single block, threads can synchronize at a barrier with negligible cost. The hardware uses a simple counter to track how many threads in a warp (a group of threads) have reached the barrier. Interactions between different thread blocks are not allowed during a kernel's execution.
* Grid-Level Synchronization: To synchronize all thread blocks in a grid (the full set of threads for a computation), one must launch a new kernel (a function that runs on the GPU). This is a much slower operation, taking approximately 2-50 microseconds.
* Atomic Operations: For fine-grained synchronization, GPUs support atomic operations on data in L2 cache or global memory. These include atomic read-modify-write operations (like add, min, max, and, or, xor) and atomic exchange or compare-and-swap. These are "fire-and-forget" operations tied to L2 latency.

#### The Explicit Memory Hierarchy

Unlike CPUs, which have largely transparent caches, GPUs feature an explicit memory hierarchy that must be managed manually by the programmer.

* Manual Data Management: The programmer is responsible for filling and spilling GPU global memory from host (CPU) memory, as well as managing the on-chip shared memory.
* Simplified Coherence: This explicit control simplifies hardware design, as there are no complex cache coherence protocols. Coherence is only guaranteed at kernel completion boundaries, meaning the programmer is responsible for ensuring data consistency.

The hierarchy consists of several levels, each with different sizes, speeds, and scopes:

| Memory Level | Scope | Typical Size (per unit) |
| --- | --- | --- |
| Registers | Single Thread | ~64k per thread block |
| Shared Memory / L1 Cache | Thread Block | 16-48kB |
| Read-only data Cache | GPU | 48kB |
| L2 Cache | GPU | 1.5MB+ |
| GDDR (off-chip) | GPU Card | 6GB+ |
| Host Memory (off-device) | System | Multiple TBs |

[Placeholder: Diagram of the explicit GPU memory hierarchy, showing the progression from on-device registers and shared memory to off-chip GDDR and off-device host memory, associated with different levels of the software model (Thread, Thread Block, Multiple Kernels).]

The performance differences between these levels are enormous, making effective memory management crucial for high performance.

[Placeholder: Diagram comparing the memory hierarchies of an Intel Sandy Bridge CPU with NVIDIA's GK110, GP100, and GA100 GPUs, highlighting differences in cache sizes, shared memory (SM), and register file capacities and bandwidths.]


---


### 4. GPU Hardware Architecture

#### Top-Level View

At a high level, a GPU is composed of several key components:

* SIMT Core Clusters: These are the main processing engines, often called Streaming Multiprocessors (SMs). A GPU contains many of these.
* Interconnection Network: A high-speed network that connects the core clusters to each other and to the memory system.
* Memory Controllers: These manage data flow between the GPU and the off-chip memory.
* GDDR Modules: High-bandwidth Graphics Double Data Rate memory, which serves as the GPU's main memory (global memory).

[Placeholder: High-level block diagram of a GPU's top-level architecture, showing multiple SIMT Core Clusters connected via an interconnection network to memory controllers and off-chip GDDR memory modules.]

#### The Streaming Multiprocessor (SM)

The Streaming Multiprocessor (SM) is the heart of the GPU. It is a multi-threaded, data-parallel processor. A typical modern SM, like the Kepler GK110's SMX, contains:

* CUDA Cores: A large number of simple processing cores for integer and single-precision floating-point (SP FPU) arithmetic (e.g., 192).
* Specialized Units:
  * Double-precision floating-point units (DP FPUs) (e.g., 64).
  * Load/Store Units (LSUs) for memory access (e.g., 32).
  * Special Function Units (SFUs) for transcendental functions like sine and square root (e.g., 32).
* Registers: A very large register file (e.g., 64,000 registers).
* Warp Schedulers: Schedulers that manage the execution of warps (groups of 32 threads) on the cores (e.g., 4 schedulers, each with 2-way dispatch).

[Placeholder: Detailed diagram of an NVIDIA Kepler GK110 Streaming Multiprocessor (SMX) architecture.]

#### The SIMT Execution Model in Practice

The SIMT model elegantly handles divergent control flow (e.g., if-else statements) where threads within the same warp take different paths. The hardware handles this by serializing the execution.

Consider the following pseudo-code, where `tid.x` is the thread's ID:

```text
A: v = foo[tid.x];
B: if (v < 10)
C:    v = 0;
   else
D:    v = 10;
E: w = bar[tid.x] + v;
```


If four threads (T1-T4) in a warp encounter this code, and the values for T1/T2 are less than 10 while the values for T3/T4 are not, execution proceeds as follows:

1. All threads execute instructions A and B together.
2. At the if statement, the threads diverge.
3. Threads T1 and T2 (which satisfy the condition) execute instruction C. Meanwhile, threads T3 and T4 are inactive.
4. Threads T3 and T4 (which do not satisfy the condition) execute instruction D. Meanwhile, threads T1 and T2 are inactive.
5. After both paths are complete, the threads reconverge and all execute instruction E together.

The programmer sees independent scalar threads, but the hardware efficiently manages the divergence by executing each branch path sequentially while disabling the threads not on that path.

[Placeholder: Flowchart illustrating the SIMT execution model for divergent control flow.]


---


### 5. Recent Trends and Future Directions

"Lay Back and Wait" Performance Scaling

For certain well-structured problems like large matrix multiplications (SGEMM), GPU performance has scaled tremendously over generations, often without requiring significant code changes from the user, especially when using optimized libraries like cuBLAS.

[Placeholder: Graph of Square SGEMM performance using cuBLAS, showing GFLOP/s scaling up to nearly 12,000 as matrix size increases to 16,384.] [Placeholder: Graph of Square SGEMM performance using custom code, showing GFLOP/s scaling up to around 1,200 as matrix size increases to 8192.]

#### NVIDIA's TensorCores

Deep learning training and inference are dominated by matrix-multiply-accumulate (MMA) operations. To accelerate this specific workload, NVIDIA introduced TensorCores, which are specialized hardware units built directly into the SMs.

**Key Concept:** TensorCores are wider, specialized ALUs designed to perform a fused multiply-add (FMA) operation on small matrix tiles (e.g., 4x4 or 8x8) using mixed-precision arithmetic. This allows them to generate many more MAC (multiply-accumulate) operations per clock cycle than standard CUDA cores.

A warp of threads provides a matrix operation, like \( D = A \cdot B + C \), to be processed by the TensorCore. These can be accessed through low-level APIs like CUDA C++ WMMA API or more commonly through high-level libraries like cuBLAS and cuDNN by setting the appropriate math mode (e.g., CUBLAS_TENSOR_OP_MATH).

[Placeholder: Diagram illustrating a 16x16x16 matrix operation D = A * B + C, representing the function of an NVIDIA TensorCore.]

#### The Future of Hardware Performance

While GPUs are nearly perfect for standard Deep Neural Networks (DNNs) and Large Language Models (LLMs), the demand for more performance continues.

* Future GPU scaling comes at a tremendous cost in power consumption.
* The dominance of GPUs may create a bias towards standard DNN-based solutions, a phenomenon known as the hardware lottery.

The billion-dollar question is: what are promising alternatives?

* Emerging hardware like analog computers (electrical or photonic) and resistive memory show promise but often come with imperfections like noise, non-linearities, and saturation effects.
* Other candidates include approximate/near-threshold-voltage computing, cryogenic computing, quantum computing, and neuromorphic computing.
* A key challenge is that these alternatives are often not general-purpose and may require disruptive changes to the entire computing stack.


---


### 6. Summary: Why GPUs are Perfect for Neural Networks

The relationship between GPUs and neural networks is deeply synergistic.

* Core Computation: The fundamental operation in neural networks is matrix multiplication, expressed as \( Y = W \cdot X \). This involves a massive number of independent but very similar computations.
* Scalability: The computation required scales with the size of the matrices (N) as \( \mathcal{O}(N^3) \), while the memory required scales as \( \mathcal{O}(N^2) \). This makes the problem computationally intensive with highly structured parallelism.
* GPU Architecture: GPUs, as massively parallel vector-based processors, are perfectly suited for this workload. They are excellent at compute-heavy tasks and are highly energy-efficient, delivering many operations per Watt. Their main limitation is memory capacity.

#### Key Takeaways

* GPUs are massively parallel BSP processors that excel at data parallelism.
* They make programming thousands of cores relatively easy through the SIMT abstraction.
* Their architecture is perfectly aligned with the constraints of today's CMOS technology, where compute cycles are cheap, but data movement is expensive.
* Specialized units like TensorCores are designed to further reduce overheads like instruction fetch and data movement for key workloads.

#### Additional Reading

* Nicholas Wilt, CUDA Handbook, Pearson Education, 2019
* Tor Aamodt, Wilson Wai Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architectures (Synthesis Lectures on Computer Architecture), Morgan Claypool, 2018


---


### 7. Backup: Energy Consumption in Computing

A critical insight from modern computer architecture is that energy is dominated by data movement, not computation.

* The cost of a typical operation (e.g., a multiply) can be less than 5pJ, while the total energy for the instruction, including fetch, control, and register file access, might be 75pJ. Vectorization helps amortize this overhead.
* The energy cost of moving data increases by orders of magnitude as you move further from the processing unit: from on-chip SRAM to off-die DRAM.
* Using reduced precision arithmetic (e.g., 8-bit integers instead of 32-bit floats) can significantly reduce both computation and data movement costs.

[Placeholder: Bar chart showing the energy cost in picojoules (pJ) for different operations, including 8-bit integer multiply (~0.2 pJ), 32-bit float multiply (~3.7 pJ), and accessing different sizes of SRAM and DRAM, with DRAM access being the most expensive by far.]
