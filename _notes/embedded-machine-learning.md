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


The ideal model complexity also depends on the size of the dataset. A more complex model can be justified if there is a sufficiently large amount of data to prevent it from overfitting.

#### Controlling Overfitting with Regularization

A common technique to control overfitting is regularization.

Regularization involves adding a penalty term to the cost function to discourage the model's coefficients from becoming too large. This "shrinks" the coefficients, leading to a simpler, smoother model that is less likely to overfit.

The regularized cost function is:

$$ \tilde{\mathcal{L}}(w) = \frac{1}{2} \sum_{n=1}^{N} \left(h(x_n, w) - t_n\right)^2 + \frac{\lambda}{2} \|w\|^2 $$

Where:

- $\lambda$ is the regularization parameter, which controls the relative importance of the penalty term.

This specific type of regularization is known as ridge regression, weight decay, or L2 regularization. Finding the optimal value for $M$ or $\lambda$ is typically done using a third dataset called a validation set.


### Summary and Next Steps

**Key Takeaways**

* This course aims to bridge the gap between complex Deep Neural Networks (DNNs) and the constraints of real-world hardware (HW).
* The linear regression example introduced fundamental concepts: learning model parameters, the importance of generalization and model selection, and the problems of overfitting and how to combat it with regularization.
* The Hardware Lottery hypothesis suggests that the prevalence of specialized hardware (like GPUs) can prevent algorithmic innovation by favoring models that fit the existing hardware paradigm.
* While linear models are foundational, they are not universal approximators—they cannot represent any arbitrary function.

**Looking Ahead: Artificial Neural Networks**

Artificial Neural Networks (ANNs) are universal approximators. They have the capacity to learn extremely complex, non-linear relationships in data. However, this power comes at a price: increased complexity, a need for vast amounts of data and computation, and reduced interpretability. The following lectures will delve into the architecture and application of these powerful models.

---


Embedded Machine Learning: A Technical Reference

Table of Contents

1. Chapter 1: The Shift to Massively Parallel Architectures
  * 1.1 The End of an Era: Dennard Scaling and the Limits of Single-Core Performance
  * 1.2 The Rise of the GPU for General-Purpose Computing
2. Chapter 2: Principles of GPU Architecture and Execution
  * 2.1 The Core Concept: Vector Processing and SIMD
  * 2.2 Two Perspectives on a Single Architecture: SIMT vs. SIMD
  * 2.3 The Programming Model: Bulk-Synchronous Parallelism
  * 2.4 Execution in Practice: Threads, Warps, and Control Flow Divergence
3. Chapter 3: The Critical Role of the Memory Hierarchy
  * 3.1 Navigating the Explicit Memory Hierarchy
  * 3.2 Bandwidth, Capacity, and Generational Improvements
  * 3.3 The Dominant Cost of Data Movement
4. Chapter 4: Hardware-Software Co-Design for Machine Learning
  * 4.1 Why Neural Networks Are a Perfect Match for GPUs
  * 4.2 Hardware Specialization: The NVIDIA Tensor Core
5. Chapter 5: Future Trends and Broader Implications
  * 5.1 The Limits of Current Scaling and the Search for Alternatives
  * 5.2 The "Hardware Lottery": How Architecture Shapes Research


--------------------------------------------------------------------------------


Chapter 1: The Shift to Massively Parallel Architectures

The landscape of high-performance computing has been fundamentally reshaped by the physical limits of semiconductor technology. The strategies that drove decades of exponential growth in single-processor performance have given way to a new paradigm: massive parallelism. This chapter explores the technological transition that necessitated this shift and introduced the Graphics Processing Unit (GPU) as a central player in general-purpose computing, particularly for machine learning.

1.1 The End of an Era: Dennard Scaling and the Limits of Single-Core Performance

For many years, processor design was driven by two complementary principles: Moore's Law, which predicted the doubling of transistors on a chip approximately every two years, and Dennard Scaling, which observed that as transistors shrank, their power density remained constant. This allowed manufacturers to increase clock frequencies and add complex features without a corresponding surge in power consumption.

This era produced sophisticated Central Processing Unit (CPU) microarchitectures designed to maximize the performance of a single instruction stream. These designs are characterized by:

* Deep, Multiple Pipelines: Breaking down instructions into many small steps to be executed concurrently.
* Latency Minimization and Hiding: Using techniques like caching to reduce the time it takes to access data from memory.
* Aggressive Speculation: Employing features like speculative fetch, pre-fetch, and execution, alongside branch predictors, to guess the program's future path and execute instructions ahead of time. This requires complex correctness-checking hardware like reorder buffers.
* Focus on Locality: Architectures heavily optimized for spatial and temporal data locality, assuming predictable control flow.

The breakdown of Dennard Scaling around 2005 marked a turning point. As transistors continued to shrink, leakage current became a significant problem, and power density began to increase. Consequently, simply increasing clock frequency was no longer viable due to thermal constraints. The fundamental power equation illustrates this challenge:

P = afCV^2 + V I_{leakage} / f^3

Where P is power, a is the activity factor, f is frequency, C is capacitance, and V is voltage. The industry was forced to find a new way to utilize the ever-increasing transistor budget from Moore's Law. The solution was to transition from complex single-core designs to simpler, replicated, and massively parallel architectures.

1.2 The Rise of the GPU for General-Purpose Computing

Originally designed for the highly parallelizable tasks of 3D graphics rendering (e.g., vertex shaders, rasterization, and fragment shaders), GPUs were built on principles of massive parallelism from the outset. Their architecture was ideal for performing large numbers of multi-dimensional floating-point operations simultaneously.

Around 2007, with the introduction of programming frameworks like NVIDIA's CUDA (Compute Unified Device Architecture), it became feasible to harness this parallel power for non-graphics tasks. This marked the beginning of General-Purpose computing on GPUs (GPGPU). The architectural philosophy of a GPU stands in stark contrast to that of a traditional CPU.

Feature	Traditional CPU (Pre-Post-Dennard)	Massively Parallel GPU
Primary Goal	Minimize latency for a single thread	Maximize aggregate throughput of many threads
Core Design	Complex, speculative, out-of-order	Simple, replicated, in-order pipelines
Parallelism	Instruction-Level Parallelism (ILP)	Data-Level Parallelism (DLP), Thread-Level Parallelism (TLP)
Control Logic	Large portion of die area for speculation, branch prediction	Minimal control logic, relies on programmer to expose parallelism
Energy Strategy	High frequency, aggressive performance	Lower frequency, massive replication for energy efficiency

This architectural divergence makes GPUs exceptionally well-suited for workloads that exhibit high data parallelism, a characteristic central to modern machine learning algorithms.

Chapter 2: Principles of GPU Architecture and Execution

To effectively leverage a GPU for machine learning, it is crucial to understand its underlying architectural model, which presents a unique abstraction to the programmer while relying on a different hardware reality. This interplay between the software view and the hardware implementation defines both the power and the programming challenges of GPUs.

2.1 The Core Concept: Vector Processing and SIMD

At its heart, a GPU is a vector processor. It utilizes a Single Instruction, Multiple Data (SIMD) execution model. This model is built upon Vector Instruction Set Architectures (ISAs), which provide significant advantages for parallel workloads:

* Compactness: A single instruction defines an operation to be performed on an entire vector of data. This amortizes the cost of instruction fetch, decode, and issue, and reduces the frequency of branches.
* Parallelism: The operations are inherently data-parallel, with no dependencies between them. This eliminates the need for complex hardware to detect parallelism, allowing for straightforward execution on parallel data paths.
* Expressiveness: Vector memory instructions can describe regular access patterns (e.g., continuous blocks of data), enabling hardware to prefetch data or use wide memory interfaces to hide the high latency of the first element access over a large sequence.

A simplified diagram of SIMD execution shows a single instruction stream dispatched to multiple processing units (PUs), each operating on a different element from a pool of data.

2.2 Two Perspectives on a Single Architecture: SIMT vs. SIMD

While the hardware operates on a SIMD basis, the programming abstraction presented to the developer is Single Instruction, Multiple Threads (SIMT). This is a crucial distinction:

* Software View (SIMT): The programmer writes code for a single scalar thread, as if it were an independent entity. This greatly simplifies programming, as one can reason about the logic for a single data element. The program is then launched with a huge number of these threads.
* Hardware View (SIMD): The GPU hardware takes these individual scalar threads and bundles them into groups called warps (typically 32 threads in NVIDIA terminology). The threads within a warp are executed in lock-step on the vector-like hardware. A single instruction is fetched and executed for the entire warp.

In essence, the SIMT model is an illusion created by the hardware to hide the underlying vector units, making the massive parallelism of the GPU more accessible.

2.3 The Programming Model: Bulk-Synchronous Parallelism

The GPU computing model aligns closely with the Bulk-Synchronous Parallel (BSP) model described by Leslie G. Valiant in 1990. This model structures computation into a sequence of "supersteps," where each superstep consists of three phases:

1. Compute: All processors perform computations independently on their local data.
2. Communicate: Processors exchange data as needed.
3. Synchronize: A barrier synchronization ensures all processors have completed the current superstep before moving to the next.

A key concept in this model is parallel slackness, defined as the ratio of virtual processors (v) to physical processors (p). For efficient scheduling and pipelining of computation and communication, it is ideal to have many more virtual processors (threads) than physical processors (v \gg p). GPU schedulers leverage this slackness to hide memory latency by swapping in other ready warps while one warp is waiting for data. This model is extremely scalable but can be inefficient for unbalanced parallelism.

2.4 Execution in Practice: Threads, Warps, and Control Flow Divergence

When a program contains conditional logic (e.g., an if-else statement), the threads within a warp may need to take different execution paths. This is known as control flow divergence. Since all threads in a warp execute the same instruction at any given time, the hardware handles divergence by serializing the execution paths.

Consider the following pseudo-code, where tid.x is the thread's unique ID:

A: v = foo[tid.x];
B: if (v < 10)
C:    v = 0;
   else
D:    v = 10;
E: w = bar[tid.x] + v;


If a warp contains four threads (T1, T2, T3, T4) where T1 and T2 satisfy the if condition and T3 and T4 do not, the execution proceeds as follows:

1. A, B: All threads execute instructions A and B in lock-step.
2. C: Threads T1 and T2 execute instruction C, while T3 and T4 are disabled (masked off).
3. D: Threads T3 and T4 execute instruction D, while T1 and T2 are disabled.
4. E: All threads reconverge and execute instruction E in lock-step.

Divergence can lead to performance degradation, as some processing units are idle while others execute a branch. Therefore, minimizing divergence within a warp is a key optimization strategy.

Synchronization between threads is handled differently based on scope:

* Within a Thread Block: Barrier synchronization is fast and has negligible cost, managed by a simple counter for arriving threads within a warp.
* Across the Grid (Global): Synchronization between different thread blocks (CTAs) is not directly allowed. It requires relaunching the kernel, which incurs a significant latency of approximately 2-50 microseconds.
* Atomic Operations: For fine-grained coordination, atomic read-modify-write operations (e.g., add, min, max) can be performed on L2 cache or global memory.

Chapter 3: The Critical Role of the Memory Hierarchy

In massively parallel architectures, performance is often limited not by the speed of computation but by the ability to supply the processing cores with data. GPUs feature a deep and explicitly managed memory hierarchy, and understanding how to navigate it is paramount for achieving high performance.

3.1 Navigating the Explicit Memory Hierarchy

Unlike CPUs, which have largely transparent caching systems, GPUs expose a significant portion of their memory hierarchy to the programmer. This requires manual management of data movement but provides fine-grained control for optimization.

* Host Memory: System RAM on the host machine (CPU side), typically multiple terabytes. Data must be explicitly transferred to the GPU device.
* GDDR (Global Memory): The GPU's main off-chip memory, analogous to system DRAM. It has high bandwidth but also high latency. (e.g., 6GB).
* L2 Cache: A large, shared cache that serves all cores on the GPU (e.g., 1.5MB).
* L1 Cache / Shared Memory: Each Streaming Multiprocessor (SM), a cluster of cores, has its own on-chip scratchpad memory. This can be configured as a combination of L1 cache and programmer-managed Shared Memory. Shared memory allows for explicit, low-latency data sharing between threads within the same thread block (e.g., 16-48kB).
* Read-only Data Cache: A cache optimized for constant data that is read by many threads (e.g., 48kB).
* Registers: The fastest memory available, private to each thread. GPUs have a very large register file per thread block (e.g., 64k registers).

This explicit management simplifies hardware for coherence and consistency, as the only guarantee is that memory operations are complete at kernel boundaries. The responsibility for maintaining data coherence falls to the software.

3.2 Bandwidth, Capacity, and Generational Improvements

The GPU memory hierarchy is characterized by a trade-off between speed, size, and proximity to the cores. As GPU generations evolve, the capacity and bandwidth at each level increase dramatically, but the fundamental structure remains.

The following table provides a comparison of memory subsystems across several GPU architectures and a representative CPU architecture.

Architecture	Registers	Shared Memory (SM) / L1	Last Level Cache (LLC)	GPU Memory (GDDR)
Intel Sandy Bridge (CPU)	~1kB @ 5TB/s	512kB (L1)	8MB @ 500GB/s	Main Memory (20GB/s)
NVIDIA Kepler GK110	~4MB @ 40TB/s	~1MB	1.5MB @ 500GB/s	4GB @ 150GB/s
NVIDIA Pascal GP100	14MB	~4MB	4MB	16GB @ 800GB/s
NVIDIA Ampere GA100	32MB	24MB	40MB	48GB @ 1.9TB/s

This data illustrates the HW-ML Interplay: GPUs prioritize massive on-chip memory bandwidth (registers, shared memory) to feed their thousands of cores, making them suitable for algorithms that can be structured to reuse data intensely from these fast memory tiers.

3.3 The Dominant Cost of Data Movement

A pivotal insight from research by M. Horowitz is that energy consumption in modern CMOS technology is dominated by data movement, not computation.

* An 8-bit integer multiply might cost around 0.2 pJ.
* A 32-bit float multiply might cost around 3.7 pJ.
* Reading 64 bits from an 8kB SRAM costs 100 pJ.
* Reading 64 bits from off-chip DRAM can cost up to 10,000 pJ.

For a typical instruction, the computation itself might account for less than 5 pJ of a total 75 pJ cost, with the rest consumed by instruction fetch, control overhead, and register file access. This disparity underscores the importance of Resource Constraints in system design. The core optimization strategies directly stem from this reality:

1. Vectorization (SIMD): Amortizes the high cost of instruction fetch and control across many data elements.
2. Reduced Precision: Using lower-precision data types (e.g., 16-bit floats or 8-bit integers) drastically reduces the energy cost of both computation and data movement.
3. Data Reuse: Maximizing the use of data once it is loaded into the fastest memory tiers (registers and shared memory) is the most effective way to improve performance and energy efficiency.

Chapter 4: Hardware-Software Co-Design for Machine Learning

The convergence of GPU architecture and the computational patterns of deep neural networks is a prime example of hardware-software co-design. The parallel, computationally intensive nature of neural networks maps almost perfectly onto the strengths of GPUs, leading to the development of specialized hardware features to further accelerate these workloads.

4.1 Why Neural Networks Are a Perfect Match for GPUs

At the heart of most neural networks lies matrix multiplication. The core operation in a dense layer is given by:

Y = W \cdot X

Where W is the weight matrix, X is the input vector (or batch of vectors), and Y is the output. This operation exhibits properties that align perfectly with GPU architecture:

* Structured Parallelism: The calculation of each element in the output matrix Y is an independent dot product. For an N \times N matrix multiplication, there are N^2 such independent computations.
* Computational Intensity: The total number of computations scales with O(N^3), while the memory required scales with O(N^2). This high compute-to-memory ratio means that once the weight and input matrices are loaded into fast memory, a massive amount of computation can be performed, maximizing data reuse.
* Energy Efficiency: Because GPUs can perform so many operations per Watt, they are highly energy-efficient for these dense computational tasks.

This natural fit means that GPUs provide exceptional performance scaling for standard Deep Neural Networks (DNNs) and Large Language Models (LLMs). The performance graphs for SGEMM (Single-precision General Matrix Multiply) show that performance increases steadily with matrix size, especially when using optimized libraries like cuBLAS that are tuned to the specific hardware.

4.2 Hardware Specialization: The NVIDIA Tensor Core

To further capitalize on the dominance of matrix operations in deep learning, NVIDIA introduced Tensor Cores, which are specialized hardware units integrated into their SMs.

* Core Function: Tensor Cores are designed to execute Matrix-Multiply-Accumulate (MMA) operations with extreme efficiency. Instead of using standard CUDA cores, these wider, specialized ALUs perform a fused multiply-add (FMA) on small matrix tiles (e.g., 4x4 or 8x8).
* Mixed-Precision Arithmetic: A key innovation is their use of mixed-precision. They can take two 16-bit floating-point (FP16) matrices as input, multiply them, and accumulate the result into a 32-bit floating-point (FP32) matrix. This significantly boosts throughput and reduces memory bandwidth requirements without a catastrophic loss of numerical precision for most deep learning models.
* Operation: A single Tensor Core can perform a complete matrix operation, such as the 16x16x16 operation shown below, generating multiple MAC operations per clock cycle. The operation is conceptually:

D_{(16 \times 16)} = A_{(16 \times 16)} \cdot B_{(16 \times 16)} + C_{(16 \times 16)}

Programmers can access Tensor Cores either through low-level APIs like the CUDA C++ WMMA (Warp-Matrix-Multiply-Accumulate) API or, more commonly, through high-level libraries like cuBLAS and cuDNN. Using libraries simply involves setting a math mode flag and ensuring the data types and matrix dimensions are compatible. For example, with cuBLAS:

// Set the math mode to allow the use of Tensor Cores
cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// Call the matrix multiplication function with appropriate data types
// (e.g., CUDA_R_16F for inputs, CUDA_R_32F for output)
cublasGemmEx(...);


The Tensor Core is a powerful example of the HW-ML Interplay, where a dominant software workload (deep learning) directly drove the creation of specialized hardware to accelerate it.

Chapter 5: Future Trends and Broader Implications

While the current GPU paradigm is incredibly successful for deep learning, the relentless demand for more performance continues to push the boundaries of technology. This final chapter considers the limits of current approaches and the broader impact of hardware on the direction of ML research.

5.1 The Limits of Current Scaling and the Search for Alternatives

The future scaling of GPU performance is expected to come at a tremendous cost in power consumption. This unsustainable trend is driving research into alternative computing paradigms that might offer better efficiency for specific workloads. Promising alternatives include:

* Approximate/Near-Threshold-Voltage Computing: Operating circuits at very low voltages to save power, at the cost of increased susceptibility to errors.
* Analog Computing: Using the physics of continuous electrical or photonic systems to perform computations, which can be extremely power-efficient but often suffer from noise and non-linearities.
* Cryogenic and Quantum Computing: Exotic technologies that offer fundamentally new ways to compute but are disruptive to the existing software stack.
* Neuromorphic Computing: Hardware inspired by the structure and function of the biological brain.

A common theme among many of these emerging hardware platforms is that they come with imperfections, such as noise, saturation effects, and non-linearities. This connects to the challenge of building ML models that are robust to Real-World Data and imperfect hardware, a key concern in embedded ML. Currently, no general-purpose replacement for CMOS-based digital computing is on the horizon.

5.2 The "Hardware Lottery": How Architecture Shapes Research

The immense success and widespread availability of GPUs for deep learning have created a phenomenon known as the "Hardware Lottery." This concept suggests that an algorithmic idea's success is not just a function of its inherent quality but also of how well it maps to available hardware.

* Bias Towards "Standard" DNNs: Because GPUs are exceptionally good at dense matrix multiplications, there is a strong incentive for researchers to develop models based on these operations.
* Stifling of Alternative Models: Novel ideas that might not fit the GPU's massively parallel, data-synchronous model may be underexplored, not because they are inherently inferior, but because they cannot be demonstrated at scale on current hardware.

This raises a critical question: Do GPUs create a bias towards specific types of ML solutions? The answer appears to be yes, highlighting a deep and complex interplay where hardware availability can guide the trajectory of an entire research field. The "billion-dollar question" remains: what promising alternatives are being overlooked due to this architectural dominance?



Embedded Machine Learning: A Technical Reference

Table of Contents

1. Chapter 1: Foundations of Neural Networks for Embedded Systems
  * 1.1 From Linear Regression to Universal Approximators
  * 1.2 The Multi-Layer Perceptron (MLP)
  * 1.3 Mathematical Formalism of a Neural Network
  * 1.4 Core Components: Activation Functions
  * 1.5 Training Neural Networks: Forward and Backward Propagation
  * 1.6 Understanding Convolutional Neural Networks (CNNs)
  * 1.7 Key CNN Architectures: AlexNet & VGG16
  * 1.8 The Simplicity Wall: A Core Challenge in Embedded ML


--------------------------------------------------------------------------------


Chapter 1: Foundations of Neural Networks for Embedded Systems

This chapter bridges the gap between fundamental machine learning concepts, such as linear regression, and the more complex, powerful models required for tasks on embedded systems. We will explore the architecture of Artificial Neural Networks (ANNs), the mathematics that govern their operation, and the core building blocks like convolutional layers that are essential for processing real-world sensor data like images. The central theme is understanding why these models are structured the way they are and how their computational patterns relate to hardware constraints.

1.1 From Linear Regression to Universal Approximators

Simple models like linear regression are foundational but are not universal approximators; they cannot model arbitrarily complex functions. This limitation is particularly acute in embedded applications that deal with high-dimensional data, such as images, where each pixel represents a new dimension (the "curse of dimensionality").

Artificial Neural Networks (ANNs) overcome this by stacking layers of interconnected "neurons," enabling them to model highly complex, non-linear relationships. The Universal Approximation Theorem states that a neural network with appropriate weights can represent a wide variety of functions. As we add more intermediate layers, known as hidden layers, we create Deep Neural Networks (DNNs), which possess even greater model capacity.

While the structure of ANNs is loosely inspired by biological neurons and synapses, the term "biologically inspired" is often met with skepticism in the field. It is crucial to distinguish these models from spiking neural networks, which more closely mimic biological processes but are typically non-differentiable and thus trained differently.

The diagram below illustrates the problem of overfitting, a critical challenge when dealing with limited or noisy real-world data. A high-capacity model (red line, polynomial of order M=9) can perfectly fit the training data but fails to generalize to the true underlying function (green line, sin(2πx)). Regularization (cyan line, lambda=0.001) is a technique to penalize model complexity, preventing overfitting and leading to better performance on unseen data.



1.2 The Multi-Layer Perceptron (MLP)

The Multi-Layer Perceptron (MLP) is a foundational class of feedforward ANN. It consists of an input layer, one or more hidden layers, and an output layer. Each neuron in a layer is connected to every neuron in the subsequent layer through a weighted edge.

Consider an image classification task like MNIST, which involves classifying 28x28 pixel images of handwritten digits into 10 classes. An MLP for this task would have 28 * 28 = 784 input neurons and 10 output neurons.

The computation for a single neuron k in a given layer is defined as:

y_k = f(\sum_j(w_{k,j} \cdot x_j) + b_k)


Where:

* y_k is the output of neuron k.
* f is a non-linear activation function (e.g., Sigmoid, ReLU).
* w_{k,j} is the weight connecting neuron j from the previous layer to neuron k.
* x_j is the activation (output) of neuron j from the previous layer.
* b_k is the bias term for neuron k.

This can be expressed more compactly for an entire layer l using vector notation:

x_l = f(W_l \cdot x_{l-1} + b_l)


Here, x_l is the activation vector for layer l, W_l is the weight matrix, x_{l-1} is the activation vector from the previous layer, and b_l is the bias vector.

1.3 Mathematical Formalism of a Neural Network

Understanding the underlying linear algebra is crucial for grasping the computational workload of neural networks, which directly impacts their performance on resource-constrained hardware.

Notation Conventions:

* Matrix: Bold uppercase (e.g., \mathbf{W}), with elements w_{k,j} (row k, column j).
* Vector: Bold lowercase (e.g., \mathbf{x}), with elements x_i. Vectors are treated as column vectors by default. A row vector is denoted with a transpose, \mathbf{x}^T.

The core operation in an MLP layer is a matrix-vector multiplication, which is computationally intensive. The entire forward pass of an L-layer MLP can be represented as a chain of these operations:

y(W, x_0) = x_L = W_L \oplus f(W_{L-1} \oplus f(... \oplus f(W_1 \oplus x_0))...)


Here, \oplus represents the combined operation of matrix multiplication and bias addition. Often, the bias vector b is incorporated into the weight matrix W by adding a constant input activation (e.g., x_0 = 1), simplifying the notation and hardware implementation.

HW-ML Interplay: The dominance of matrix-vector and matrix-matrix multiplications is a defining characteristic of DNNs. This has driven the development of specialized hardware like GPUs and TPUs, which use massive parallelism (e.g., SIMD, systolic arrays) to accelerate these specific linear algebra operations. Efficiently managing memory to feed these computational units is a primary challenge in embedded ML hardware design.

1.4 Core Components: Activation Functions

Activation functions, or non-linearities, are applied after the linear transformation in each neuron. They are essential for allowing the network to learn complex patterns; without them, a multi-layer network would be mathematically equivalent to a single linear layer.

The choice of activation function impacts training dynamics and computational cost. The Rectified Linear Unit (ReLU) is the most prevalent due to its simplicity and effectiveness in mitigating the "vanishing gradients" problem during training.

The diagram below shows several common activation functions and their derivatives (dashed red lines).



A comparison of these functions is provided in the table below:

Function	Formula	Output Range	Key Characteristics
Sigmoid	f(x) = \frac{1}{1 + e^{-x}}	[0, 1]	Smooth, but can suffer from vanishing gradients.
Tanh	f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}	[-1, 1]	Zero-centered, which can help with optimization.
ReLU	f(x) = \max(x, 0)	[0, \infty)	Computationally efficient (a simple threshold). Avoids vanishing gradients for positive inputs but can lead to "dead neurons" (neurons that never activate).
Leaky ReLU	f(x) = \begin{cases} x, & x \ge 0 \\ \alpha x, & x < 0 \end{cases}	(-\infty, \infty)	A variation of ReLU that allows a small, non-zero gradient for negative inputs, preventing dead neurons.
ELU	f(x) = \begin{cases} x, & x \ge 0 \\ e^x - 1, & x < 0 \end{cases}	(-1, \infty)	Aims to combine the benefits of ReLU with smoother gradients for negative values.

The Softmax Function for Classification

For classification tasks, the raw output of the final layer is often uncalibrated. The Softmax function is typically applied to convert these outputs into a multinomial probability distribution over K classes.

\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}


The outputs of softmax sum to 1, providing a more interpretable result. However, it's important not to confuse this with a mathematically rigorous probability. The function can be numerically unstable due to the exponentiation of large numbers. A common trick is to subtract the maximum value from all inputs before applying the exponent, which is mathematically equivalent but more stable:

\frac{e^{x_i}}{\sum e^{x_j}} = \frac{Ce^{x_i}}{C\sum e^{x_j}} = \frac{e^{x_i + \log C}}{\sum e^{x_j + \log C}} \quad \text{where } \log C = -\max_j x_j


1.5 Training Neural Networks: Forward and Backward Propagation

Training a neural network involves adjusting its weights (W) to minimize a loss function (\mathcal{L}) that measures the discrepancy between the network's predictions and the true targets. This is an optimization problem solved using gradient-based methods.

The overall loss for a dataset \mathcal{D} = \{(x_1, t_1), ..., (x_N, t_N)\} is:

\mathcal{L}(W; \mathcal{D}) = \sum_{n=1}^{N} l(y(W, x_n), t_n) + \lambda r(W)


This consists of:

1. Data Term: l(y, t), which penalizes incorrect predictions.
2. Regularizer: r(W), such as an \ell_1 or \ell_2-norm on the weights, which penalizes model complexity to prevent overfitting. \lambda is a hyperparameter that balances the two terms.

The training process iteratively updates the weights in the opposite direction of the gradient of the loss function. This is known as gradient descent:

W := W - \eta \nabla_W \mathcal{L}(W; \mathcal{D})


Here, \eta is the learning rate, a hyperparameter controlling the step size.

Backpropagation

Backpropagation is the algorithm used to efficiently compute the gradient \nabla_W \mathcal{L}. It involves two main phases:

1. Forward Pass: An input is passed through the network to compute the output and the final loss.
2. Backward Pass: The algorithm propagates the gradient of the loss backward from the output layer to the input layer, using the chain rule of calculus to compute the partial derivative of the loss with respect to each weight in the network.

The chain rule allows us to decompose the gradient calculation layer by layer. For a function z = f(x, y), the gradient of the loss \mathcal{L} with respect to the inputs x and y is:

\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial x} \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial y} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial y}


The term \frac{\partial \mathcal{L}}{\partial z} is the "upstream gradient" coming from the next layer, while \frac{\partial z}{\partial x} and \frac{\partial z}{\partial y} are the "local gradients" of the current operation. This process is repeated for every layer and every operation (matrix multiplication, activation function, etc.).

The loss landscape of modern ANNs is highly non-convex. Architectural choices, such as the use of skip connections (as in ResNet), can significantly smooth this landscape, making optimization more stable and effective, as illustrated in the visualization below.



1.6 Understanding Convolutional Neural Networks (CNNs)

While MLPs are powerful, they are inefficient for high-dimensional spatial data like images. They fail to account for the spatial correlation between pixels and have a massive number of parameters. Convolutional Neural Networks (CNNs) are designed specifically to address these issues.

Key properties of convolutional layers:

1. Spatially Local Correlation (Receptive Field): Neurons are only connected to a small, local patch of the input from the previous layer. This leverages the fact that pixels closer to each other are more strongly related.
2. Shared Weights: The same set of weights (a "filter" or "kernel") is applied across all patches of the input. This drastically reduces the number of parameters compared to a fully-connected layer and allows the network to detect features (like edges or textures) regardless of their position in the image.

An input to a convolutional layer, such as a color image, is a 3D tensor of shape (Channels, Width, Height). The layer applies a set of learnable filters, and each filter produces a 2D "output feature map." The "depth" of the output layer is the number of filters used.

The convolution operation for an output feature map O from an input feature map I using weights W and biases B is defined as:

O[z][u][x][y] = \sum_{k=0}^{C-1} \sum_{i=0}^{S-1} \sum_{j=0}^{R-1} I[z][k][Ux + i][Uy + j] \cdot W[u][k][i][ j] + B[u]


Where:

* N is the batch size (z).
* M is the number of output filters (u).
* C is the number of input channels (k).
* R, S are the filter height and width.
* H, W are the input height and width.
* E, F are the output height and width.
* U is the stride.

Resource Constraints: The principles of receptive fields and shared weights are fundamental optimizations that make deep learning on images computationally feasible. By reducing the number of parameters, they significantly lower the memory required to store the model. Furthermore, the regular, repeating pattern of the convolution operation is highly amenable to hardware acceleration, allowing for immense data reuse and computational efficiency.

The diagram below visualizes a DenseNet, an advanced CNN architecture where feature maps from early layers are directly connected to all subsequent layers within a block, promoting feature reuse and improving gradient flow.



1.7 Key CNN Architectures: AlexNet & VGG16

AlexNet was a pioneering deep CNN that demonstrated the power of the architecture on the ImageNet dataset. Its structure, shown below, consists of a stack of convolutional layers followed by fully-connected layers for classification.

The architecture processes a 227x227x3 input image through five convolutional layers (Conv1-Conv5) and three fully-connected layers (FC6-FC8). The convolutional layers progressively reduce the spatial dimensions (227x227 -> 55x55 -> 27x27 -> 13x13) while increasing the number of channels (depth), extracting increasingly complex features. The final fully-connected layers perform the classification based on these high-level features.



VGG16 is another influential architecture known for its simplicity, exclusively using 3x3 convolutional filters stacked in increasing depth. Both models highlight the common design pattern of using convolutional layers for feature extraction and fully-connected layers for final classification.

1.8 The Simplicity Wall: A Core Challenge in Embedded ML

Despite their architectural complexity, a critical insight is that ANNs spend the vast majority of their computational time on a very simple and repetitive operation: matrix multiplication. This is true for both fully-connected layers (explicit matrix-vector products) and convolutional layers (which can be implemented as large matrix-matrix products via Toeplitz matrices).

This concept is referred to as the "Simplicity Wall". While we design intricate networks, their execution boils down to a massive number of multiply-accumulate (MAC) operations. This has profound implications for embedded systems:

* HW-ML Interplay: The efficiency of an embedded ML system is not just about the algorithm but is fundamentally determined by the hardware's ability to perform matrix multiplications at high speed and with low power.
* Resource Constraints: The key to deploying complex models on small devices is not just algorithmic innovation, but co-designing algorithms and hardware to optimize these fundamental linear algebra operations. Techniques like Quantization (reducing the precision of numbers) and Pruning (removing unnecessary weights) directly target the reduction of this computational and memory burden.



Embedded Machine Learning: A Technical Guide to Optimization and Differentiation

Table of Contents

1. Chapter 1: The Foundations of Neural Network Training
  * 1.1 The Goal: Minimizing the Loss Function
  * 1.2 Backpropagation: The Engine of Learning
2. Chapter 2: Automatic Differentiation
  * 2.1 The Need for an Automated Approach
  * 2.2 The Computational Graph
  * 2.3 The Chain Rule: A Practical Perspective
  * 2.4 Autograd in Practice: PyTorch
  * 2.5 The Autograd Algorithm
3. Chapter 3: Advanced Calculus for Neural Networks
  * 3.1 From Scalar to Vector Calculus
  * 3.2 Backpropagation: The Vectorized View
4. Chapter 4: Gradient-Based Optimization Algorithms
  * 4.1 Gradient Descent: The Core Concept
  * 4.2 Variants of Gradient Descent
  * 4.3 Challenges in Optimization
  * 4.4 Momentum: Accelerating Convergence
  * 4.5 Adaptive Learning Rate Methods
    * Adagrad
    * Adadelta and RMSprop
    * Adam: Adaptive Moment Estimation
  * 4.6 Visualizing Optimizer Performance
5. Chapter 5: Conclusion and Key Takeaways


--------------------------------------------------------------------------------


Chapter 1: The Foundations of Neural Network Training

The process of training an Artificial Neural Network (ANN) is fundamentally an optimization problem. It involves systematically adjusting the network's parameters to minimize the difference between its predictions and the actual target values. This chapter revisits the core mathematical concepts that enable this learning process: the loss function and backpropagation.

1.1 The Goal: Minimizing the Loss Function

The training process begins with a dataset \mathcal{D} containing N input-target pairs, denoted as \mathcal{D} = \{(x_1, t_1), \dots, (x_N, t_N)\}. The network's parameters, collectively known as weights W, are initially set to random values. The objective is to refine these weights to solve a specific task.

This refinement is guided by a Loss Function, \mathcal{L}, which quantifies the model's error. A common formulation for the loss function is:

\mathcal{L}(W; \mathcal{D}) = \sum_{n=1}^{N} l(y(W, x_n), t_n) + \lambda r(W)


This equation consists of two primary components:

1. Data Term: The term l(y(W, x_n), t_n) is the error function that penalizes incorrect predictions. Here, y(W, x_n) is the model's output for input x_n, and t_n is the ground truth target. The sum accumulates this error over all data points.
2. Regularizer: The term r(W) is a regularizer, such as the \ell_1-norm or \ell_2-norm, which penalizes large weight values to prevent overfitting. The hyperparameter \lambda controls the trade-off between fitting the data and maintaining simple weights.

The goal of training is to find the set of weights W that minimizes this loss function \mathcal{L}.

1.2 Backpropagation: The Engine of Learning

Backpropagation is the algorithm used to efficiently compute the gradients of the loss function with respect to the network's weights. A gradient is a vector of partial derivatives, indicating the direction of the steepest ascent of the loss function. To minimize the loss, we adjust the weights in the opposite direction of the gradient.

This process is known as gradient-based optimization. The core update rule is an iterative process defined as:

W := W - \eta \nabla_W \mathcal{L}(W; \mathcal{D})


Where:

* W represents the network's weights.
* \eta is the learning rate, a hyperparameter that controls the size of each update step.
* \nabla_W \mathcal{L}(W; \mathcal{D}) is the gradient of the loss function with respect to the weights W. The gradient operator is defined as \nabla_x = (\frac{\partial}{\partial x_1}, \dots, \frac{\partial}{\partial x_n}).

Essentially, for each weight W_n in the network, we compute its partial derivative \frac{\partial}{\partial W_n} with respect to the overall loss and take a small step in the opposing direction. The key operations underpinning this entire process are partial derivatives and linear algebra.

A neural network is a nested function. For an L-layer network, the output y for an input x_0 can be expressed as:

y(W, x_0) = x_L = f(W_L \oplus f(W_{L-1} \oplus (\dots \oplus f(W_1 \oplus x_0) \dots )))


To find the gradient of the loss l with respect to the input x_0, the chain rule of calculus is applied repeatedly, propagating the derivatives backward through the layers:

\frac{\partial l}{\partial x_0} = \frac{\partial l}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_{L-1}} \cdot \dots \cdot \frac{\partial x_i}{\partial x_{i-1}} \cdot \dots \cdot \frac{\partial x_2}{\partial x_1} \cdot \frac{\partial x_1}{\partial x_0}


The central question for efficient training is: what is the most computationally effective way to compute the gradient for all parameters (weights W and biases b)? This leads to the concept of automatic differentiation.

Chapter 2: Automatic Differentiation

Manually calculating the derivatives for a deep neural network with millions of parameters is not only tedious but also computationally inefficient and prone to error. Automatic differentiation, or Autograd, is the solution that modern machine learning frameworks employ to automate this crucial step.

2.1 The Need for an Automated Approach

Consider a simple model with a regularized loss function:

* Model: z = wx + b, y = \sigma(z)
* Loss Function: \mathcal{L}_{reg} = \mathcal{L} + \lambda\mathcal{R} = \frac{1}{2}(y - t)^2 + \frac{\lambda}{2}w^2

If we substitute the model into the loss function, we get:

\mathcal{L}_{reg} = \frac{1}{2}(\sigma(wx + b) - t)^2 + \frac{\lambda}{2}w^2


Calculating the partial derivatives \frac{\partial\mathcal{L}_{reg}}{\partial w} and \frac{\partial\mathcal{L}_{reg}}{\partial b} by hand, as one would in a calculus class, reveals several inefficiencies:

\frac{\partial\mathcal{L}_{reg}}{\partial w} = (\sigma(wx + b) - t)\sigma'(wx + b)x + \lambda w


\frac{\partial\mathcal{L}_{reg}}{\partial b} = (\sigma(wx + b) - t)\sigma'(wx + b)


This manual process is cumbersome and reveals redundant calculations. For example, the term (\sigma(wx + b) - t)\sigma'(wx + b) is computed for both derivatives. In a deep network with many layers and shared parameters, these redundancies multiply, leading to massive computational waste. Autograd solves this by systematically breaking down the computation and reusing intermediate results.

2.2 The Computational Graph

Autograd systems represent computations as a Computational Graph, which is a directed acyclic graph (DAG).

* Vertices (Nodes): Represent variables or the results of operations.
* Edges: Represent dependencies between variables, directed from inputs to outputs.

The graph is constructed in a topological ordering, meaning that parents (inputs) always appear before their children (outputs).

For our simple example (z = wx + b, etc.), the computational graph would show inputs w, x, b, t flowing through intermediate operations to produce the final output, \mathcal{L}_{reg}.

 Description: A diagram illustrating the flow from inputs w, x, b to an intermediate node z. z then feeds into y. y and input t feed into L. Separately, w feeds into R. Finally, L and R combine to form the final output L_reg.

The backpropagation process works by traversing this graph backward, from the final output back to the inputs, applying the chain rule at each vertex to compute gradients.

2.3 The Chain Rule: A Practical Perspective

To streamline the backpropagation process, a new notation is introduced. Instead of viewing \frac{\partial\mathcal{L}}{\partial v} as an expression to be evaluated, we treat it as a computed value, denoted as \bar{v}.

* Definition: \bar{v} = \frac{\partial\mathcal{L}}{\partial v}

This notation emphasizes that \bar{v} (read as "v-bar") is a quantity we compute and store, promoting the idea of data reuse. The chain rule is then expressed as:

\bar{v} = \frac{\partial\mathcal{L}}{\partial v} = \frac{\partial\mathcal{L}}{\partial u} \frac{\partial u}{\partial v} = \bar{u} \frac{\partial u}{\partial v}


This simplifies the multivariate chain rule as well. For a function f(a(v), b(v)), the gradient \bar{v} is:

\bar{v} = \frac{\partial\mathcal{L}}{\partial v} = \frac{\partial\mathcal{L}}{\partial a} \frac{\partial a}{\partial v} + \frac{\partial\mathcal{L}}{\partial b} \frac{\partial b}{\partial v} = \bar{a} \frac{\partial a}{\partial v} + \bar{b} \frac{\partial b}{\partial v}


Here, \bar{a} and \bar{b} are previously computed values, which highlights the efficiency of this backward pass.

2.4 Autograd in Practice: PyTorch

PyTorch's autograd engine is a prime example of these concepts in action. It tracks operations on tensors to build the computational graph dynamically.

Core Mechanism: requires_grad

To signal to autograd that operations on a tensor should be tracked, its requires_grad attribute is set to True.

import torch

# Tensors a and b will be tracked by autograd
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# Define a computation
q = 3*a**3 - b**2


The operation creating q is now part of the computational graph. When we call .backward(), autograd computes the gradients of q with respect to its dependencies (a and b).

# The gradient passed to backward() must match the shape of the output tensor
external_grad = torch.tensor([1., 1.])
q.backward(gradient=external_grad)

# The computed gradients are stored in the .grad attribute of the leaf tensors
print(a.grad) # ∂q/∂a = 9a^2 = 9 * [4, 9] = [36, 81]
print(b.grad) # ∂q/∂b = -2b = -2 * [6, 4] = [-12, -8]


Full Training Loop Example

In a typical training loop, autograd is used to compute gradients for all model parameters.

import torch
from torchvision.models import resnet18, ResNet18_Weights

# 1. Initialize the model and optimizer
model = resnet18(weights=ResNet18_Weights.DEFAULT)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# 2. Prepare dummy data
data = torch.rand(1, 3, 64, 64)   # Example input image
labels = torch.rand(1, 1000)      # Example target labels

# 3. Forward Pass: Compute predictions
prediction = model(data)

# 4. Compute Loss
loss = (prediction - labels).sum()

# 5. Backward Pass: Autograd computes gradients for all parameters
#    with requires_grad=True and stores them in their .grad attribute.
loss.backward()

# 6. Gradient Descent: The optimizer updates parameters using the stored gradients
optim.step()


This elegant process abstracts away the complexity of differentiation, allowing developers to focus on model architecture.

2.5 The Autograd Algorithm

The backward pass of Autograd follows a precise algorithm that leverages the topological ordering of the computational graph.

Let v_1, \dots, v_N be the vertices of the graph in topological order.

* Pa(v_i): The set of parent vertices of v_i.
* Ch(v_i): The set of child vertices of v_i.

Algorithm Steps:

1. Forward Pass: Compute the value of each vertex v_i as a function of its parents Pa(v_i), for i = 1, \dots, N.
2. Backward Pass Initialization: Initialize the gradient for the final output vertex, \bar{v}_N = 1. This is because \frac{d\mathcal{L}_{reg}}{d\mathcal{L}_{reg}} = 1 by convention.
3. Backward Iteration: Iterate backward from i = N-1 down to 1:  \bar{v}i = \sum{j \in Ch(v_i)} \bar{v}_j \frac{dv_j}{dv_i}  This step sums the incoming gradients from all children, each weighted by the local partial derivative, effectively applying the multivariate chain rule.

This algorithm ensures that every partial derivative is computed exactly once and that intermediate gradient values (\bar{v}_j) are reused, maximizing computational efficiency.

Chapter 3: Advanced Calculus for Neural Networks

While understanding backpropagation with scalar values is intuitive, modern deep learning relies on matrix and vector operations for performance. This requires extending our understanding to vector calculus.

3.1 From Scalar to Vector Calculus

The principles of differentiation remain the same, but the operations are applied to vectors and matrices. The key mathematical tool is the Jacobian matrix, which contains all first-order partial derivatives of a vector-valued function.

For a function y = f(x) where y \in \mathbb{R}^m and x \in \mathbb{R}^n, the Jacobian J is an m \times n matrix:

J = \begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{pmatrix}


Using this, the chain rule for backpropagation can be expressed in a compact, vectorized form. For a node x, its gradient \bar{x} can be computed from the gradient of its child y as:

\bar{x} = (\frac{\partial y}{\partial x})^T \bar{y} = J^T \bar{y}


Below are key vectorized derivatives for common neural network operations:

* Matrix-vector multiplication (y = Wx):
  * Partial derivative with respect to input x: \frac{\partial y}{\partial x} = W. The gradient update is \bar{x} = W^T\bar{y}.
  * Partial derivative with respect to weights W: The gradient update is \bar{W} = \bar{y}x^T, which is the outer product of \bar{y} and x.
* Element-wise operation (y = e^x):
  * The Jacobian is a diagonal matrix. The gradient update is an element-wise multiplication (Hadamard product): \bar{x} = e^x \circ \bar{y}.

3.2 Backpropagation: The Vectorized View

Revisiting the two-layer neural network example, the scalar update rules for backpropagation can be directly translated into their more efficient vectorized counterparts.

Scalar Calculation for a single element	Vectorized Calculation for the entire layer
\bar{y}_k = y_k - t_k	\mathbf{\bar{y}} = \mathbf{y} - \mathbf{t}
\bar{W}^{(2)}_{k,j} = \bar{y}_k \cdot h_j	\mathbf{\bar{W}}^{(2)} = \mathbf{\bar{y}}\mathbf{h}^T
\bar{b}^{(2)}_k = \bar{y}_k	\mathbf{\bar{b}}^{(2)} = \mathbf{\bar{y}}
\bar{h}_k = \sum_j \bar{y}_j \cdot W^{(2)}_{j,k}	\mathbf{\bar{h}} = (\mathbf{W}^{(2)})^T\mathbf{\bar{y}}
\bar{z}_k = \bar{h}_k \cdot \sigma'(z_k)	\mathbf{\bar{z}} = \mathbf{\bar{h}} \circ \sigma'(\mathbf{z})
\bar{W}^{(1)}_{k,j} = \bar{z}_k \cdot x_j	\mathbf{\bar{W}}^{(1)} = \mathbf{\bar{z}}\mathbf{x}^T
\bar{b}^{(1)}_k = \bar{z}_k	\mathbf{\bar{b}}^{(1)} = \mathbf{\bar{z}}

HW-ML Interplay: This transition from scalar loops to vectorized operations is not merely a notational convenience. It is fundamental to performance on modern hardware. CPUs and especially GPUs are designed for SIMD (Single Instruction, Multiple Data) operations. Vector and matrix multiplications can be executed in parallel, leading to orders-of-magnitude speedups over scalar-based loops. Efficiently implementing ML algorithms on embedded systems requires leveraging this hardware parallelism.

Chapter 4: Gradient-Based Optimization Algorithms

Once gradients are computed via backpropagation, an optimizer uses them to update the model's weights. The choice of optimizer significantly impacts training speed and final model performance.

4.1 Gradient Descent: The Core Concept

The fundamental principle of gradient descent is to iteratively take steps in the direction opposite to the gradient of the cost function. This is visualized as descending a hill in the "loss landscape" to find the point of minimum cost (loss).

The update rule remains:

W := W - \eta\nabla_W\mathcal{L}(W; \mathcal{D})


4.2 Variants of Gradient Descent

The way the dataset \mathcal{D} is used to compute the gradient leads to three main variants:

Variant	Gradient Calculation	Update Frequency	Pros	Cons
Batch Gradient Descent	Uses the entire training dataset for each gradient calculation.	Few updates per epoch.	Guaranteed to converge to the optimum (for convex surfaces). Stable updates.	Slow. Computationally expensive and memory-intensive for large datasets.
Stochastic Gradient Descent (SGD)	Uses a single training sample (x^{(i)}, y^{(i)}) for each gradient calculation.	Many updates per epoch (one per sample).	Fast, cheap updates. Stochasticity can help escape local minima.	High variance in updates, leading to a noisy convergence path. Requires careful learning rate tuning.
Mini-batch SGD	Uses a small subset (mini-batch) of n training samples.	Balanced. More updates than Batch GD, fewer than SGD.	Reduces update variance compared to SGD, leading to more stable convergence. Computationally efficient.	Introduces a new hyperparameter: the mini-batch size.

HW-ML Interplay: For Mini-batch SGD, the batch size is often chosen as a power of two (e.g., 32, 64, 128). This aligns with the memory architecture and parallel processing capabilities of GPUs, leading to higher computational efficiency.

4.3 Challenges in Optimization

Training deep neural networks presents several challenges:

1. Choosing the Learning Rate (\eta): A single learning rate is applied to all parameters, but some may require larger or smaller updates. Manually tuning this and creating schedules (e.g., decaying the learning rate over time) is difficult and adds more hyperparameters.
2. Complex Loss Landscapes: The loss function for a neural network is non-convex and filled with numerous local minima and saddle points, which can stall the optimization process.

Advanced optimizers aim to address these issues automatically.

4.4 Momentum

Momentum is an optimization technique that helps accelerate SGD in the relevant direction and dampens oscillations. It computes the weight update based on the current gradient and an exponentially decaying average of past gradients.

The update is calculated as follows:

1. Compute the velocity vector v_t: v_t = \gamma v_{t-1} + \eta\nabla_\theta J(\theta)
2. Update the parameters \theta: \theta := \theta - v_t

Here, \gamma is the momentum term (usually close to 0.9), acting like friction. This method can be visualized as a heavy ball rolling down the loss surface; it accumulates momentum in consistent directions and is less affected by noisy, oscillating gradients.

4.5 Adaptive Learning Rate Methods

These methods eliminate the need to manually tune the learning rate by adapting it for each parameter based on its update history.

Adagrad

Adagrad adapts the learning rate on a per-parameter basis, performing larger updates for infrequent parameters and smaller updates for frequent ones. It accumulates the sum of squared gradients for each parameter over time.

* Update Rule: \theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} \cdot g_{t,i}
* Accumulator: G_t is a diagonal matrix where each diagonal element G_{t,ii} is the sum of the squares of the gradients of \theta_i up to time step t.

Problem: The accumulator G_t only grows. This causes the effective learning rate to shrink continuously, potentially stalling training prematurely.

Adadelta and RMSprop

Adadelta and RMSprop resolve Adagrad's diminishing learning rate issue by restricting the history of gradients. Instead of summing all past squared gradients, they use an exponentially decaying average, keeping a "moving window" of recent gradient information.

* Decaying Average: E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma)g_t^2
* Update Rule (RMSprop): \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t

Adadelta further refines this by also keeping a decaying average of squared parameter updates, removing the need for a global learning rate \eta altogether.

Adam: Adaptive Moment Estimation

Adam is arguably the most popular and recommended optimizer. It combines the ideas of Momentum and RMSprop. It maintains two exponentially decaying averages:

1. m_t: An estimate of the mean of the gradients (the first moment), similar to Momentum.
2. v_t: An estimate of the uncentered variance of the gradients (the second moment), similar to RMSprop.

* First Moment: m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t
* Second Moment: v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2

Adam also includes a bias-correction step to account for the fact that m_t and v_t are initialized to zero and thus biased towards zero in the initial steps.

* Bias Correction: \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
* Final Update Rule: \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t

Adam behaves like a heavy ball with friction, accelerating in flat regions and slowing down in steep ones, providing stable and efficient convergence.

Resource Constraints: While powerful, optimizers like Adam come with a memory cost. They must store additional state for each parameter (the moving averages m_t and v_t). For memory-constrained embedded systems, the simpler SGD or Momentum might be preferable if Adam's memory overhead is too high.

4.6 Visualizing Optimizer Performance

The slides provide visualizations that compare the trajectories of different optimizers on complex loss surfaces.

* Contour Plot: A 2D contour plot shows how optimizers navigate a landscape with a steep ravine. SGD struggles, moving slowly. Momentum overshoots and oscillates but makes faster progress down the ravine. Adaptive methods like Adagrad, Adadelta, and RMSprop converge more directly toward the minimum.
* Saddle Point Surface: A 3D surface plot illustrates a saddle point. This visualization shows how different algorithms might slow down or escape such points, highlighting the superior navigation capabilities of adaptive methods.

These diagrams effectively demonstrate that while plain SGD is robust, advanced optimizers like Adam generally converge faster and more reliably, making them a strong default choice.

Chapter 5: Conclusion and Key Takeaways

This module has detailed the fundamental mechanics of training a neural network, from the mathematical necessity of backpropagation to the practical implementation of automatic differentiation and the suite of advanced optimization algorithms.

Key Insights:

1. Backpropagation and SGD are Foundational: These methods are robust, powerful, and relatively simple to use. Their main requirement is that all components of the neural network must be differentiable.
2. Automatic Differentiation is Essential for Efficiency: Modern frameworks like PyTorch use computational graphs and the chain rule to automatically and efficiently compute gradients. This system avoids redundant calculations and scales to massive models, a critical feature for both training large models and for optimizing computation on resource-constrained devices.
3. Vectorization is Key to Hardware Performance: The shift from scalar-based calculations to vectorized operations is crucial for leveraging the parallel processing capabilities of modern hardware (CPUs, GPUs), which is a central theme in embedded machine learning.
4. Advanced Optimizers Improve Convergence: While SGD is a viable option, adaptive methods like Adam offer significant improvements in convergence speed and stability by adjusting the learning rate per-parameter and incorporating momentum. Adam is strongly recommended as a starting point, but its additional state (memory usage) should be considered for embedded applications.

Understanding the interplay between these algorithms and the underlying hardware constraints is the bridge between theoretical machine learning and practical, efficient deployment on embedded systems.



Embedded Machine Learning - Chapter 5: Regularization

Table of Contents

1. Chapter 1: The Core Problem: Overfitting and the Need for Regularization
  * 1.1 Defining Regularization
  * 1.2 Visualizing Overfitting
  * 1.3 Philosophical Underpinnings: No Free Lunch and Occam's Razor
2. Chapter 2: The Statistical Foundation: The Bias-Variance Tradeoff
  * 2.1 Probability Review: Expectation and Variance
  * 2.2 Decomposing Model Error
  * 2.3 The Tradeoff in Practice
3. Chapter 3: Foundational Regularization Strategies
  * 3.1 Limiting Training Time: Early Stopping
  * 3.2 Limiting Model Size: Parameter Reduction and Bottlenecks
4. Chapter 4: Data-Centric Regularization: Augmentation
  * 4.1 The Principle of Data Augmentation
  * 4.2 Advanced Augmentation: Mixup, Cutout, and CutMix
  * 4.3 Automated Augmentation Policies
  * 4.4 Implementation in PyTorch
5. Chapter 5: Parameter-Based Regularization: L1 & L2 Norms
  * 5.1 Penalizing Complexity: The Regularized Cost Function
  * 5.2 L2 Regularization (Weight Decay / Ridge)
  * 5.3 L1 Regularization (LASSO) and Sparsity
  * 5.4 A Geometric Interpretation of L1 and L2 Norms
6. Chapter 6: Stochastic Regularization Methods
  * 6.1 The Power of Ensembles
  * 6.2 Dropout: An Efficient Ensemble Approximation
  * 6.3 DropBlock: Adapting Dropout for Convolutional Layers
7. Chapter 7: Summary and Implications for Embedded Systems


--------------------------------------------------------------------------------


Chapter 1: The Core Problem: Overfitting and the Need for Regularization

In the pursuit of deploying machine learning models on resource-constrained embedded systems, a primary challenge is creating models that are both compact and robust. The phenomenon of overfitting stands in direct opposition to this goal, resulting in models that are unnecessarily complex and perform poorly on new, unseen data. This chapter introduces the concept of regularization as the primary tool to combat overfitting and build efficient, generalizable models.

1.1 Defining Regularization

Regularization is formally defined as any modification applied to a learning algorithm that is intended to reduce its generalization error but not its training error. Generalization error is the measure of how accurately a model can predict outcomes for previously unseen data, which is the ultimate benchmark of a model's utility.

In essence, regularization techniques introduce a penalty for model complexity, guiding the learning process towards simpler solutions that are more likely to capture the true underlying patterns in the data rather than memorizing the noise and idiosyncrasies of the training set.

Embedded ML Pillar: Resource Constraints & Real-World Data Regularization is not just an abstract statistical tool; it is a fundamental enabler for embedded ML. By promoting simpler models, regularization directly leads to models with fewer parameters and lower precision requirements, reducing memory footprint and computational load. Furthermore, it enhances a model's robustness to the noisy, imperfect data typically encountered by embedded sensors in the real world.

1.2 Visualizing Overfitting

The tension between model complexity and performance is best understood visually. Consider fitting a polynomial function to a set of noisy data points sampled from a sine wave.

A provided set of diagrams illustrates this scenario with polynomial models of varying degrees (M):

* M=0 & M=1: These models (a constant and a line) are too simple. They fail to capture the underlying sinusoidal pattern, resulting in high error on both training and test data. This is known as underfitting.
* M=3: This cubic polynomial provides a good approximation of the underlying sine wave. It captures the general trend without being overly influenced by the noise in individual data points. This represents a well-fit model.
* M=9: This high-degree polynomial fits the training data points perfectly. However, it achieves this by oscillating wildly between points, creating a complex curve that has completely lost the original sinusoidal signal. This model will perform extremely poorly on new data. This is a classic example of overfitting.

This dynamic is captured by the relationship between training error and test error. As model complexity increases, the training error consistently decreases. However, the test error follows a U-shaped curve: it decreases initially (as the model moves from underfitting to a good fit) and then begins to rise sharply as the model starts overfitting. The goal of regularization is to find the "best fit" point at the bottom of this curve.

1.3 Philosophical Underpinnings: No Free Lunch and Occam's Razor

Two principles guide the philosophy behind regularization:

1. "No Free Lunch" Theorem: This theorem states that no single machine learning algorithm is universally the best for all tasks. The choice of method must be tailored to the specific problem at hand. The same logic applies to regularization; the right technique depends on the model architecture and data characteristics.
2. Occam's Razor: This principle advocates for simplicity. When faced with multiple hypotheses that explain the data equally well, one should always choose the simplest one. Regularization is the practical application of Occam's Razor to machine learning, penalizing complexity to favor simpler, and thus more generalizable, models.

Chapter 2: The Statistical Foundation: The Bias-Variance Tradeoff

To understand why regularization works, we must first decompose a model's error into its fundamental statistical components: bias and variance. This tradeoff is central to machine learning and dictates the balance between model simplicity and complexity.

2.1 Probability Review: Expectation and Variance

Let's define two key statistical measures for a random variable X. The prediction of a neural network can be conceptualized as drawing a sample from a complex, well-informed random variable.

* Expectation (E(X)): This describes the average or mean value of the random variable.
  * For a discrete variable: E(X) = \sum_{x} x p_x(x)
  * For a continuous variable: E(X) = \int_{-\infty}^{\infty} x f_x(x) dx
* Variance (Var(X)): This measures the spread or variability of the data around the expectation. It is the expectation of the squared deviation from the mean.
  * Var(X) = E((X - E(X))^2) = E(X^2) - E(X)^2

The standard deviation, std(X), is simply the square root of the variance, \sqrt{Var(X)}.

2.2 Decomposing Model Error

Using these concepts, we can analyze the error of a model's predictions.

* Bias: The bias of an estimator measures the difference between the model's average prediction and the correct value we are trying to predict. It quantifies the model's systematic error. A high-bias model is too simple and fails to capture the underlying data structure (underfitting).  Bias(\hat{y}_m) = E(\hat{y}_m) - y 
* Variance: The variance of an estimator measures the variability of the model's prediction for a given data point. It quantifies how much the model's predictions would change if it were trained on a different training set. A high-variance model is overly sensitive to the training data, capturing noise as if it were signal (overfitting).  Var(\hat{y}_m) 
* Mean Squared Error (MSE): The Mean Squared Error is a common metric for a model's performance. It can be decomposed into three components:  MSE = E[(\hat{y}_m - y)^2] = Bias(\hat{y}_m)^2 + Var(\hat{y}_m) + \text{Bayes Error}  The Bayes Error is the irreducible, inherent error due to noise in the data itself. It represents the lower bound of error for any possible model.

2.3 The Tradeoff in Practice

The MSE decomposition reveals the fundamental bias-variance tradeoff.

* Underfitting models are characterized by high bias and low variance. They are consistently wrong in the same way, regardless of the specific training data.
* Overfitting models are characterized by low bias and high variance. On average, their predictions are centered around the correct value, but for any specific training set, they can be wildly inaccurate due to their sensitivity to noise.

A diagram illustrating the generalization error curve shows that as model complexity increases, bias tends to decrease while variance tends to increase. The total error is minimized at a point of optimal complexity where these two components are balanced.

Regularization is the set of techniques used to navigate this tradeoff. Specifically, regularization methods intentionally increase bias slightly in order to achieve a much greater reduction in variance, leading to a lower overall MSE.

Embedded ML Pillar: HW-ML Interplay The bias-variance tradeoff has direct hardware implications. High-variance (overfit) models often have large, sensitive weight values that require higher precision (e.g., 32-bit floats) to represent accurately. By trading some bias for lower variance, regularization produces models with smaller, more stable weights, which are more amenable to quantization—the process of converting weights to low-precision formats like 8-bit integers. This drastically reduces model size and improves computational efficiency on simple hardware.

Chapter 3: Foundational Regularization Strategies

Beyond manipulating data or model parameters directly, regularization can be achieved by controlling the training process itself or by making static architectural choices.

3.1 Limiting Training Time: Early Stopping

Early stopping is an intuitive and effective regularization technique. The core idea is to monitor the model's performance on a separate validation dataset during training and halt the process when the validation error stops decreasing and begins to rise, even if the training error is still falling.

* Intuition: During the initial epochs of training, a model's weights are typically small. As training progresses, the weights grow in magnitude to fit the training data more closely. By stopping training early, we prevent the weights from becoming excessively large and complex, which is a hallmark of overfitting. This has an effect similar to explicit weight decay methods like L2 regularization.
* Practical Challenges: In practice, the validation error can fluctuate due to the stochastic nature of optimization algorithms like Stochastic Gradient Descent (SGD). This can make it difficult to determine the precise optimal stopping point.

32. Limiting Model Size: Parameter Reduction and Bottlenecks

The number of parameters in a model is a direct measure of its capacity. A straightforward way to limit capacity and thus regularize a model is to reduce its parameter count.

One common architectural pattern for this is the use of linear bottleneck layers. A diagram illustrates a comparison between a direct 100-unit -> 100-unit layer and a bottleneck architecture 100-unit -> 10-unit -> 100-unit.

* Parameter Count:
  * Direct Layer: The number of connections (parameters) is approximately 100 \times 100 = 10,000.
  * Bottleneck Layer: The number of parameters is (100 \times 10) + (10 \times 100) = 2,000.
* Expressivity: While additional linear layers do not increase the theoretical expressivity of a network (a series of linear transformations is equivalent to a single linear transformation), the bottleneck structure forces the network to learn a compressed, low-dimensional representation of the data in the 10-unit layer. This constraint acts as a powerful regularizer.

The primary drawback of such static parameter reduction is that it can make the model architecture too simple to learn complex but essential patterns in the data. This motivates the need for more dynamic regularization approaches.

Embedded ML Pillar: Resource Constraints Static parameter reduction through methods like bottleneck layers is one of the most direct ways to design a model for an embedded target. Fewer parameters mean a smaller memory footprint for storing the model and fewer computations (MAC operations) required during inference, leading to lower latency and energy consumption.

Chapter 4: Data-Centric Regularization: Augmentation

One of the most effective ways to combat overfitting and improve model robustness is to train on more data. When collecting new data is impractical, Data Augmentation provides a way to synthetically expand the training set.

4.1 The Principle of Data Augmentation

Data augmentation involves creating new, realistic training samples by applying a series of transformations to the existing data. For image data, common transformations include:

* Geometric Transformations: Flipping, rotating, cropping, shearing, and translating.
* Color and Photometric Transformations: Adjusting brightness, contrast, or color saturation.
* Noise Injection: Adding random noise to pixels.

The key principle is that these transformations should preserve the core semantic meaning of the data. For example, a horizontally flipped image of a cat is still a valid image of a cat.

Important Rule: Data augmentation should only be applied to the training set. The test set must remain untouched to serve as an unbiased benchmark of the model's real-world performance.

4.2 Advanced Augmentation: Mixup, Cutout, and CutMix

More advanced techniques go beyond simple transformations and involve combining or occluding images. These "regional dropout" strategies force the model to learn from partial information, making it more robust.

Method	Description	Labeling Strategy
Cutout	Randomly removes a square patch from an input image, forcing the model to use context from surrounding features.	The original label is preserved (e.g., Dog: 1.0).
Mixup	Creates a new image by taking a weighted linear interpolation of two existing images.	The labels are mixed with the same weights (e.g., Dog: 0.5, Cat: 0.5).
CutMix	Cuts a patch from one image and pastes it onto another.	Labels are mixed proportionally to the area of the combined patches (e.g., Dog: 0.6, Cat: 0.4).

These methods have proven highly effective in training strong, generalizable classifiers.

4.3 Automated Augmentation Policies

Manually designing an effective augmentation pipeline is a task-specific and time-consuming process. AutoAugment is a technique that automates this discovery process using reinforcement learning.

* Mechanism: A controller network (an RNN) learns to search for optimal augmentation "sub-policies." Each sub-policy consists of a sequence of operations (e.g., ShearX, Invert), each with an associated probability and magnitude.
* Performance vs. Cost: AutoAugment can discover policies that significantly outperform manually designed ones. However, this comes at a tremendous computational cost; the source notes a training time of approximately 1,000 GPU hours for the SVHN dataset.
* Alternatives: More recent methods like TrivialAugment have been shown to achieve comparable or better performance than AutoAugment with a much simpler search strategy, making them highly recommended.

A rule of thumb for applying augmentation is to consider it when the model's training loss is significantly lower than its test loss, which is a clear sign of overfitting.

4.4 Implementation in PyTorch

In frameworks like PyTorch, implementing data augmentation is straightforward using the transforms library.

# Example of a training data transformation pipeline
transforms_train = transforms.Compose([
   transforms.RandomCrop(32, padding=4),
   transforms.RandomRotation(3.8),
   transforms.RandomVerticalFlip(p=0.4),
   transforms.RandomHorizontalFlip(p=0.3),
   transforms.RandomAffine(translate=(0.8,0.9), shear=[0.2,0.4,0.7]),
   transform.ToTensor(),
   # ... other transforms like normalization
])

# The test set should typically only have basic preprocessing
transforms_test = tranforms.Compose([
   transforms.ToTensor(),
   # ... other transforms like normalization
])


Code Explanation:

* transforms.Compose: Chains multiple transformation steps together.
* RandomCrop, RandomRotation, etc.: These functions apply their respective transformations with a certain probability or within a specified range. They are applied on-the-fly during data loading, so the model sees a slightly different version of an image in each epoch.
* transform.ToTensor(): Converts the image data into a PyTorch tensor, the standard data structure for the framework.

The choice of transformations is highly task-dependent. For instance, RandomHorizontalFlip is effective for general object recognition but would be detrimental for a digit recognition task (e.g., confusing a '6' with a '9').

Embedded ML Pillar: Real-World Data Data augmentation is the primary technique for dealing with data scarcity and improving robustness to real-world variations. An embedded device (e.g., a camera sensor) will encounter objects from different angles, under varying lighting conditions, and with partial occlusions. Augmentation simulates this variability during training, resulting in a model that generalizes far better "in the wild."

Chapter 5: Parameter-Based Regularization: L1 & L2 Norms

One of the most direct ways to control model complexity is to impose a penalty on the magnitude of its parameters (weights). This is achieved by adding a regularization term to the model's cost function.

5.1 Penalizing Complexity: The Regularized Cost Function

The standard cost function J(w) is modified to include a regularization term R(w), scaled by a hyperparameter \lambda.

\mathcal{J}(w) = \frac{1}{N} \sum_{n=1}^{N} \mathcal{L}(y(w, x_n), t_n) + \lambda \mathcal{R}(w)


* \mathcal{L} is the loss function (e.g., squared error).
* \mathcal{R}(w) is the regularization penalty, which is a function of the model's weights.
* \lambda is the regularization strength, which controls the relative importance of the penalty term. A larger \lambda imposes a stronger penalty, leading to simpler models.

5.2 L2 Regularization (Weight Decay / Ridge)

L2 Regularization, also known as Weight Decay or Ridge Regression, uses the squared magnitude of the weight vector as its penalty term.

\mathcal{R}(w) = \frac{1}{2} \|w\|_2^2 = \frac{1}{2} w^T w = \frac{1}{2} \sum_j w_j^2


The effect of this penalty is most apparent in the update rule for Stochastic Gradient Descent (SGD):

w_j := w_j - \eta \frac{\partial \mathcal{J}}{\partial w_j} = w_j - \eta (\frac{\partial \mathcal{L}}{\partial w_j} + \lambda w_j) = (1 - \eta \lambda)w_j - \eta \frac{\partial \mathcal{L}}{\partial w_j}


During each update step, the weight w_j is first shrunk by a factor of (1 - \eta \lambda) before the standard gradient update is applied. This is why it's called "weight decay"—it constantly pushes weights towards zero.

Intuition: Overfitting is often correlated with excessively large weights. A model with large weights is highly sensitive to small changes in its input. For example, in a simple neuron y = w_1 x_1 + w_2 x_2, a weight vector of w = [-9, 11] will produce large output fluctuations compared to a more stable vector like w = [1, 1]. L2 regularization penalizes large weights, promoting smoother, more stable models. A table of weights for the M=9 polynomial fit shows this empirically: without regularization, weights can be in the thousands, while L2 regularization constrains them to small single-digit values.

5.3 L1 Regularization (LASSO) and Sparsity

L1 Regularization, also known as LASSO (Least Absolute Shrinkage and Selection Operator), uses the sum of the absolute values of the weights as its penalty.

\mathcal{R}(w) = \|w\|_1 = \sum_j |w_j|


While L2 regularization shrinks weights towards zero, L1 regularization can push weights to be exactly zero. This property is known as sparsity.

A comparison of the L1 and L2 norms and their gradients reveals why:

* L2 Norm: Has a parabolic shape (w^2). Its gradient is linear (2w), so the penalty's "pull" towards zero weakens as the weight gets smaller.
* L1 Norm: Has a V-shape (|w|). Its gradient is a constant +1 or -1 (for non-zero w). This means it applies a constant "pull" towards zero, regardless of the weight's magnitude, making it very effective at eliminating small, unimportant weights completely.

Embedded ML Pillar: HW-ML Interplay & Resource Constraints The sparsity induced by L1 regularization is a critical concept in embedded ML. A sparse model has many zero-valued weights. This means the corresponding multiply-accumulate operations during inference can be skipped entirely. This leads to:

1. Model Compression: The model can be stored in a sparse format, saving memory.
2. Latency & Energy Reduction: Skipping computations directly reduces inference time and power consumption. Modern hardware accelerators are increasingly being designed with support for sparse computations to exploit this property.

5.4 A Geometric Interpretation of L1 and L2 Norms

The behavior of L1 and L2 regularization can be visualized by considering the optimization problem. The goal is to find the weights w that minimize the loss function (represented by elliptical contours) subject to the constraint that the regularization penalty \mathcal{R}(w) is less than some constant.

* L2 Constraint: The L2 norm constraint \|w\|_2^2 \le s defines a circular region (in 2D). The loss ellipse is likely to touch this circle at a point where neither weight is zero.
* L1 Constraint: The L1 norm constraint \|w\|_1 \le s defines a diamond-shaped region. The elliptical contours of the loss function are much more likely to make first contact with this region at one of its sharp corners. At these corners, one of the weight components is exactly zero.

This visualization intuitively explains why L1 regularization promotes sparse solutions, while L2 regularization promotes small but generally non-zero weights.

Chapter 6: Stochastic Regularization Methods

Introducing randomness into the training process is another powerful form of regularization. These methods, known as stochastic regularization, help prevent the model from memorizing the training data by constantly presenting it with varied perspectives.

6.1 The Power of Ensembles

An ensemble is a collection of multiple models whose predictions are averaged to produce a final output. By training several models independently (e.g., on different random subsets of the data, a technique called bagging), the variance of the final prediction can be significantly reduced.

Ensembles are often highly accurate but are computationally expensive, as they require training and running multiple full models. This makes them generally unsuitable for resource-constrained embedded systems.

6.2 Dropout: An Efficient Ensemble Approximation

Dropout is a clever and computationally efficient technique that approximates the training of a massive ensemble of neural networks.

* Mechanism: During each training step, every neuron in a layer is "dropped" (i.e., its output is set to zero) with a certain probability p. This creates a new, "thinned" network architecture for each training batch.
* Intuition:
  1. Ensemble Effect: Training with dropout is like training an exponential number of different smaller networks that share weights. A diagram shows how a single base network can be seen as an ensemble of all its possible sub-networks.
  2. Preventing Co-adaptation: It prevents neurons from becoming overly reliant on the output of a few specific preceding neurons. Each neuron must learn to be more robust and independently useful.
  3. Implicit L2 Regularization: A neuron with a very large outgoing weight becomes a liability, as its high activation, when present, will cause large fluctuations in the network's output when it is randomly dropped. The optimization process will therefore favor smaller, more distributed weights, similar to the effect of L2 regularization.

Inference Time: Dropout is only applied during training. At test time, to compensate for the fact that all neurons are now active, the outgoing weights from the dropout layer are scaled down by a factor of (1-p).

6.3 DropBlock: Adapting Dropout for Convolutional Layers

Standard dropout has been found to be less effective in convolutional neural networks (CNNs). This is due to the strong spatial correlation in feature maps. If a single activation is dropped, its neighbors likely contain very similar information, allowing the signal to propagate anyway.

DropBlock is an adaptation that addresses this by dropping entire contiguous rectangular blocks of a feature map rather than individual, random activations. This forces the network to learn more robust features that are not reliant on specific local information.

Chapter 7: Summary and Implications for Embedded Systems

Regularization is a cornerstone of modern machine learning, rivaled in importance only by optimization. It encompasses any technique that reduces a model's generalization error without harming its performance on the training data.

Several key themes emerge from the study of regularization:

* Conceptual Overlap: Many different techniques achieve similar goals through different means. There are strong conceptual links between ensembles, dropout, and data augmentation. Likewise, early stopping, L2 weight decay, and bottleneck architectures all serve to constrain the magnitude or number of model weights.
* The Power of Stochasticity: Introducing randomness through methods like SGD, dropout, and data augmentation is a powerful, though not fully understood, way to improve model generalization.

For the embedded systems engineer, regularization is not merely a tool for improving accuracy—it is a critical part of the model design and compression pipeline.

Sneak Preview & Final Takeaway: A crucial insight for embedded ML is that model compression techniques like pruning (removing weights) and quantization (reducing precision) often introduce a form of noise or stochasticity into the model. This process can act as a regularizer itself, leading to the remarkable outcome where models can become both smaller and more accurate. This synergy between regularization and model compression is a central theme in designing efficient machine learning solutions for hardware-constrained environments.




Embedded Machine Learning: A Technical Reference

Table of Contents

* Chapter 1: Principles of Neural Architecture Design
  * 1.1 The Bias-Variance Tradeoff in Model Design
  * 1.2 Regularization: Taming Model Complexity
* Chapter 2: Stabilizing Training with Normalization
  * 2.1 The Challenge of Training Deep Architectures
  * 2.2 Introduction to Batch Normalization (BN)
  * 2.3 Mathematical Formulation and Algorithm
  * 2.4 Practical Implementation and Considerations
  * 2.5 Why Batch Normalization is Effective
  * 2.6 Alternatives for Different Constraints
* Chapter 3: Foundations of Convolutional Architectures
  * 3.1 The Convolution Operation: A Recap
  * 3.2 Quantifying Convolutional Layers: Cost Analysis
  * 3.3 Case Study: AlexNet Architecture
  * 3.4 Downsampling with Pooling Layers
* Chapter 4: Designing for Efficiency: Modern Architectures
  * 4.1 Grouped Convolutions for Parallelism and Efficiency
  * 4.2 The Inception Architecture: Going Deeper Efficiently
  * 4.3 The Residual Architecture (ResNet): Overcoming Degradation
  * 4.4 Evolving Architectures: DenseNet
* Chapter 5: Summary and Key Metrics
  * 5.1 Comparative Analysis of Architectures
  * 5.2 Core Efficiency Metrics for Embedded ML
  * 5.3 Concluding Remarks


--------------------------------------------------------------------------------


Chapter 1: Principles of Neural Architecture Design

The design of a neural network architecture is a fundamental task that dictates not only its predictive accuracy but also its computational cost. In embedded machine learning, where resources are scarce, finding an architecture that balances performance with efficiency is paramount. This chapter explores the foundational principles of model complexity and the techniques used to control it.

1.1 The Bias-Variance Tradeoff in Model Design

A central challenge in machine learning is developing models that generalize well to new, unseen data. The generalization error of a model can be decomposed into three components: bias, variance, and irreducible error.

* Bias is the error from erroneous assumptions in the learning algorithm. High bias can cause a model to miss relevant relations between features and target outputs, a condition known as underfitting.
* Variance is the error from sensitivity to small fluctuations in the training set. High variance can cause a model to capture random noise instead of the intended output, a condition known as overfitting.

The relationship between model capacity and these error components is illustrated in the classic bias-variance tradeoff curve.

A diagram illustrates the relationship between Model Capacity (x-axis) and Error (y-axis). As model capacity increases, Bias (a dotted purple line) decreases. Conversely, Variance (a dotted blue line) increases. The Generalization Error (a solid green curve) is the sum of these, forming a U-shape. The lowest point of this curve represents the "Best fit," where the model is neither underfitting (low capacity) nor overfitting (high capacity).

In embedded systems, overly complex models (high variance) are undesirable not just for their poor generalization but also because they demand more memory and computational power. The goal is to find the "best fit" architecture that minimizes generalization error while respecting hardware constraints.

1.2 Regularization: Taming Model Complexity

Regularization refers to a set of techniques used to prevent overfitting by penalizing model complexity. It modifies the learning objective by adding a penalty term to the standard loss function. The generalized objective function, J(w), can be expressed as:

\mathcal{J}(\mathbf{w}) = \frac{1}{N} \sum_{n=1}^{N} \mathcal{L}(y(\mathbf{w}, \mathbf{x}_n), t_n) + \lambda \mathcal{R}(\mathbf{w})


Where:

* \mathcal{L} is the loss function.
* \mathcal{R}(\mathbf{w}) is the regularization term that penalizes the magnitude of the model weights \mathbf{w}.
* \lambda is the regularization parameter that controls the strength of the penalty.

Key regularization techniques include:

* Weight Decay (L1/L2 Regularization): This method adds a penalty proportional to the L1 or L2 norm of the weight vector, encouraging the model to learn smaller, simpler weight configurations.
* Data Augmentation: This technique artificially expands the training dataset by creating modified copies of existing data (e.g., rotating, cropping, or flipping images). It helps the model learn to be invariant to such transformations, improving its ability to handle real-world data variations.
* Dropout: Proposed by Srivastava et al., dropout is a technique where, during training, randomly selected neurons are ignored ("dropped out"). This prevents neurons from co-adapting too much. During inference, the full network is used, but weights are scaled to account for the dropout during training. This acts as a form of model averaging, improving generalization.

Chapter 2: Stabilizing Training with Normalization

As neural networks grow deeper, the training process becomes more complex and unstable. This chapter examines the challenges posed by deep architectures and introduces normalization techniques, particularly Batch Normalization, designed to stabilize training, accelerate convergence, and improve model performance.

2.1 The Challenge of Training Deep Architectures

Deep neural architectures are composed of sequentially connected layers, creating a strong layer interdependence. When the parameters of one layer are updated during training, the distribution of that layer's output changes. This phenomenon has cascading effects on all downstream layers, which must constantly adapt to a shifting input distribution.

This constant shifting, sometimes referred to as internal covariate shift, makes training difficult for several reasons:

* Unstable Training: The optimizer must account for these cascading effects, which can destabilize the learning process.
* Difficult Coordination: Coordinating weight updates across many layers is challenging due to second- and higher-order interactions.
* Vanishing/Exploding Gradients: Small changes in early layers can be amplified or diminished as they propagate through the network, leading to unstable gradients.

2.2 Introduction to Batch Normalization (BN)

Batch Normalization (BN) is a technique of adaptive reparametrization designed to address the difficulties of training deep networks. By normalizing the activations of a layer, BN helps to mitigate the interdependency between layers and stabilize the learning process.

The core idea is to standardize the activations within a mini-batch to have a mean of zero and a standard deviation of one (i.e., a standard Gaussian distribution). This reparametrization coordinates updates across layers, making the training process more robust.

A diagram illustrates the effect of BN. It shows a set of input signals (x1, x2, x3, x4) with highly interdependent distributions being transformed by a layer. The output signals (y1) have similarly interdependent distributions. After applying BN, the normalized signals are shown with mitigated interdependency, leading to more stable learning for the subsequent layer.

2.3 Mathematical Formulation and Algorithm

While fully whitening a layer's inputs is computationally expensive, BN offers a simplified and effective approach. It normalizes each scalar feature independently over a mini-batch of size m. The Batch Normalizing Transform is detailed in Algorithm 1.

Algorithm 1: Batch Normalizing Transform

Input: Values of x over a mini-batch: B = \{x_1, ..., x_m\}; Parameters to be learned: \gamma, \beta. Output: \{y_i = BN_{\gamma, \beta}(x_i)\}

1. Calculate mini-batch mean (\mu_B):  \mu_B \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i 
2. Calculate mini-batch variance (\sigma_B^2):  \sigma_B^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2 
3. Normalize (\hat{x}_i):  \hat{x}_i \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}  Here, \epsilon is a small constant for numerical stability, preventing division by zero.
4. Scale and Shift (y_i):  y_i \leftarrow \gamma \hat{x}i + \beta \equiv BN{\gamma, \beta}(x_i)  Here, \gamma (scale) and \beta (shift) are trainable parameters.

Introducing the learnable parameters \gamma and \beta seems counter-intuitive, as they can reverse the normalization. However, this is crucial for the learning dynamics. It allows the network to represent the identity transform and ensures that the mean of the transformed activations is determined solely by the easily learnable parameter \beta, rather than by complex interactions in previous layers. This makes the optimization problem much easier for Gradient Descent (GD).

2.4 Practical Implementation and Considerations

When incorporating BN into a network, several practical points are important:

1. Redundant Bias: The standard bias term b in a layer (e.g., in Wx + b) becomes redundant when using BN, as its effect is canceled out by the mean subtraction and replaced by the learnable shift parameter \beta. Therefore, the layer can be simplified to Wx.
2. Placement: BN can be applied either before or after the non-linear activation function. While the original paper applied it before, applying it after often yields better results in practice.
3. Interaction with Dropout: As a rule of thumb, it is often advised not to use BN and Dropout together, as their regularizing effects can interact in complex ways.
4. Inference Time: During testing, we may process single inputs, making mini-batch statistics unavailable. To handle this, the mini-batch mean \mu and variance \sigma^2 are replaced with running averages, \mu' and \sigma'^2, which are computed and stored during the training phase. This allows the normalization to be applied consistently at inference time, irrespective of batch size. This adaptation is crucial for Resource Constraints in embedded deployment, where batch processing may not be feasible.

2.5 Why Batch Normalization is Effective

Several theories explain the remarkable effectiveness of BN:

1. Reduces Internal Covariate Shift (Original Theory): By stabilizing the distribution of activations, BN ensures that downstream layers do not have to constantly adapt to shifting inputs. The use of running averages at test time makes the test set distribution "similar enough" to the training set, leading to better generalization.
2. Improves Gradient Flow: BN mitigates the interdependency between layers during training, leading to a more consistent scale of activations. This improved gradient flow makes the network less sensitive to hyperparameters like the learning rate.
3. Smoothes the Optimization Landscape: By normalizing over a mini-batch rather than the full dataset, BN introduces a small amount of noise. This noise acts as a regularizer and has been shown to smooth the optimization landscape, making it easier for the optimizer to find a good minimum. This is beneficial for dealing with noisy, real-world data.

2.6 Alternatives for Different Constraints

Batch Normalization's performance is highly dependent on the mini-batch size. For embedded applications where memory is limited and small batch sizes are common, this can be a significant drawback. Several alternatives have been developed:

A diagram shows four 3D tensors representing feature maps with axes (N, C, H, W). The blue shaded area in each indicates the values used for normalization.

* Batch Norm (BN): Normalizes across the batch (N) and spatial (H, W) dimensions for each channel (C).
* Layer Norm (LN): Normalizes across the channel (C) and spatial (H, W) dimensions for each sample in the batch (N). It is independent of batch size.
* Instance Norm (IN): Normalizes across the spatial (H, W) dimensions for each channel (C) and each sample (N) independently.
* Group Norm (GN): Divides channels into groups and normalizes within each group across the spatial dimensions. It is also independent of batch size and shows performance comparable to BN.

These alternatives, especially Layer Norm and Group Norm, are critical for scenarios with small mini-batches, a common situation under the Resource Constraints of embedded systems.

Chapter 3: Foundations of Convolutional Architectures

Convolutional Neural Networks (CNNs) are the cornerstone of modern computer vision. Their architecture is designed to exploit the spatial structure of data like images, making them highly effective and computationally efficient compared to fully connected networks. This chapter revisits the convolution operation and analyzes its computational cost.

3.1 The Convolution Operation: A Recap

The core of a CNN is the convolution layer, which applies a set of learnable filters to an input feature map to produce an output feature map.

A diagram illustrates the 3D convolution process. A set of M filters, each of size R x S x C, are convolved with an input feature map (ifmap) of size H x W x C. Each filter produces one channel of the M-channel output feature map (ofmap) of size E x F x M.

The operation is defined by the following equation, which calculates a single output activation value:  O[z][u][x][y] = \sum_{k=0}^{C-1} \sum_{i=0}^{S-1} \sum_{j=0}^{R-1} I[z][k][U \cdot x + i][U \cdot y + j] \cdot W[u][k][i][j] + B[u] 

The dimensions involved are:

* Input Feature Map (ifmap, I):
  * N: Batch size
  * C: Number of input channels
  * H, W: Height and width of the input plane
* Filters (Weights, W):
  * M: Number of filters (determines output channels)
  * R, S: Height and width of the filter plane
* Output Feature Map (ofmap, O):
  * E, F: Height and width of the output plane, calculated as:  E = \frac{H - R + 2P}{U} + 1 \quad , \quad F = \frac{W - S + 2P}{U} + 1  where U is the stride and P is the padding.
* Biases (B): A single bias term B[u] for each output channel u.

3.2 Quantifying Convolutional Layers: Cost Analysis

Understanding the computational and memory costs of a network is essential for embedded ML. We can quantify a layer by its number of parameters and the number of Multiply-Accumulate (MAC) operations required. A MAC operation, c = \sum_n a_n b_n, is the fundamental computation in a dot product.

Layer Type	Number of Parameters	Number of MAC Operations
Convolution (Conv)	Weights: W_c = R \cdot S \cdot C \cdot M <br> Biases: B_c = M <br> Total: P_c = W_c + B_c	MAC_c = (E \cdot F) \cdot (R \cdot S \cdot C) \cdot M
Fully Connected (FC)	Weights: W_f = (\text{Inputs}) \cdot (\text{Outputs}) <br> Biases: B_f = \text{Outputs} <br> Total: P_f = W_f + B_f	MAC_f = (\text{Inputs}) \cdot (\text{Outputs})

This analysis highlights a key aspect of HW-ML Interplay: convolutions are computationally intensive, scaling with the output map size (E, F) and filter size (R, S, C). Efficient architecture design focuses on minimizing these terms.

3.3 Case Study: AlexNet Architecture

AlexNet, a pioneering CNN, demonstrated the power of deep convolutional networks. Its architecture consists of a sequence of convolutional layers, pooling layers, and fully connected layers.

A diagram shows the architecture of AlexNet. It starts with a large input volume (227x227x3), which is processed by a sequence of five convolutional layers and three max-pooling layers. The feature map size progressively shrinks in spatial dimensions (H, W) while the depth (C) generally increases. Finally, the features are flattened and passed through three fully connected layers, culminating in a 1000-way softmax output.

The detailed breakdown of AlexNet reveals the computational and parameter costs at each stage:

Layer	Output Dim (H, W, C)	Filter/Pool (R, S)	Stride (U)	Padding (P)	Groups (G)	Neurons (O)	MACs	Parameters
Input	227, 227, 3	-	-	-	-	-	-	-
CONV-1	55, 55, 96	11, 11	4	0	1	-	105,415,200	34,944
POOL-1	27, 27, 96	3, 3	2	-	-	-	-	-
CONV-2	27, 27, 256	5, 5	1	2	2	-	223,948,800	307,456
POOL-2	13, 13, 256	3, 3	2	-	-	-	-	-
CONV-3	13, 13, 384	3, 3	1	1	1	-	149,520,384	885,120
CONV-4	13, 13, 384	3, 3	1	1	2	-	112,140,288	663,936
CONV-5	13, 13, 256	3, 3	1	1	2	-	74,760,192	442,624
POOL-3	6, 6, 256	3, 3	2	-	-	-	-	-
FC-1	1, 1, 4096	-	-	-	-	4096	37,748,736	37,752,832
FC-2	1, 1, 4096	-	-	-	-	4096	16,777,216	16,781,312
FC-3	1, 1, 1000	-	-	-	-	1000	4,096,000	4,097,000
Total							724,406,816	60,965,224

Notice that the vast majority of parameters are in the FC layers, while the vast majority of MACs are in the CONV layers. This insight drives modern architecture design to reduce or eliminate costly FC layers.

3.4 Downsampling with Pooling Layers

Pooling layers are a critical component of CNNs, used to reduce the spatial dimensions (height and width) of the feature maps. This has two primary benefits:

1. Reduced Computational Complexity: By decreasing the output map size (H and W), pooling reduces the number of MACs required in subsequent convolutional layers, as MAC_c is proportional to E \cdot F. Note that pooling does not change the number of parameters in the network.
2. Translation Invariance: Pooling creates invariance to small translations in the input. For tasks like image classification, the exact location of a feature is often less important than its presence. While convolutional layers are equivariant (a translation in the input results in a corresponding translation in the output), pooling layers provide a degree of invariance.

A diagram shows a 4x4 grid of numbers undergoing 2x2 pooling with a stride of 2. In Max Pooling, the maximum value from each 2x2 block is taken. In Average Pooling, the average value is taken.

During backpropagation, a max pooling gate acts as a "gradient router," passing the downstream gradient only to the neuron that had the maximum value during the forward pass. While essential, too much pooling can be harmful, as it discards spatial information that may be valuable.

Chapter 4: Designing for Efficiency: Modern Architectures

The evolution from early CNNs like AlexNet to modern architectures has been driven by the pursuit of higher accuracy with greater computational efficiency. This chapter explores key architectural innovations—Grouped Convolutions, Inception modules, and Residual connections—that enable deeper, more powerful, and more resource-friendly networks suitable for embedded systems.

4.1 Grouped Convolutions for Parallelism and Efficiency

A grouped convolution divides the input channels into g groups. A separate set of filters is then applied to each group independently. The final output is the concatenation of the results from all groups along the channel axis.

This technique, first used in AlexNet to distribute the model across two GPUs, has profound implications for efficiency.

A diagram contrasts a standard convolution with a grouped convolution. In the standard version, c_2 filters, each with depth c_1, process the input tensor. In the grouped version (with g groups), the input tensor is split into g chunks along the channel axis. Each chunk (of depth c_1/g) is processed by a separate set of c_2/g filters. The outputs are then concatenated.

By splitting the computation, grouped convolutions reduce both the number of parameters and the number of MAC operations by a factor of g.

* Parameters: W_{cg} = g \cdot (R \cdot S \cdot \frac{C}{g} \cdot \frac{M}{g}) = \frac{W_c}{g}
* MACs: MAC_{cg} = g \cdot ((E \cdot F) \cdot (R \cdot S \cdot \frac{C}{g}) \cdot \frac{M}{g}) = \frac{MAC_c}{g}

This is a prime example of HW-ML Interplay: the architecture is designed for parallel execution and significantly reduces the computational load, a critical requirement for resource-constrained hardware.

4.2 The Inception Architecture: Going Deeper Efficiently

The Inception architecture (GoogLeNet) was motivated by the need to deploy powerful models on mobile devices. It achieves this by drastically reducing the number of parameters while building a deeper, 22-layer network.

Core Ideas:

1. Eliminate FC Layers: Like many modern networks, Inception replaces the parameter-heavy fully connected layers at the end with convolutions.
2. Factorize Convolutions: It replaces large convolutions with multiple smaller ones. For example, two stacked 3x3 filters have a similar receptive field to one 5x5 filter but with fewer parameters (2 \cdot (9CM) vs. 25CM).

The Inception Module: The key innovation is the Inception module, a "network-in-network" design. The naïve version applies multiple convolutions (1x1, 3x3, 5x5) and a max pooling operation in parallel to the same input layer. The outputs are then concatenated.

A diagram of the naïve Inception module shows a "Previous layer" block feeding into four parallel branches: 1x1 convolutions, 3x3 convolutions, 5x5 convolutions, and 3x3 max pooling. The outputs of these branches are all fed into a "Filter concatenation" block.

The Role of 1x1 Convolutions: The major challenge with the naïve module is that concatenating the outputs rapidly increases the channel depth, making subsequent layers computationally expensive. The solution is to use 1x1 convolutions for dimensionality reduction.

A 1x1 convolution is essentially a fully connected layer applied across the channels at each spatial location. By controlling the number of 1x1 filters (M), we can reduce the number of input channels (C) for the more expensive 3x3 and 5x5 convolutions.

A diagram of the improved Inception module shows 1x1 convolution blocks inserted before the 3x3 and 5x5 convolutions, and after the max pooling layer. These "bottleneck" layers reduce the channel depth before the main computation.

The impact of this technique on resource constraints is enormous. Consider a 5x5 convolution on a 28x28x192 input:

Configuration	Parameters	MACs
Plain 5x5x32 convolution	153,600	120,422,400
With 1x1 Bottleneck:		
-> Initial 1x1x16 (reduction)	3,072	2,408,448
-> New 5x5x32 on reduced input	12,800	10,035,200
Total with Bottleneck	15,872	12,443,648

By first reducing the channels from 192 to 16 with a 1x1 convolution, the number of parameters is reduced by ~10x and the number of MACs by ~10x. This is a foundational technique for building efficient deep networks.

4.3 The Residual Architecture (ResNet): Overcoming Degradation

As networks become deeper, a "degradation" problem emerges: accuracy gets saturated and then degrades rapidly. This is not caused by overfitting but by the difficulty of optimizing very deep networks, often due to vanishing gradients.

The Residual Network (ResNet) addresses this by introducing "shortcut" or "skip" connections. Instead of learning a direct mapping H(x), a residual module learns the residual mapping \mathcal{F}(x) = H(x) - x. The output of the module is then \mathcal{F}(x, W) + x.

A diagram illustrates a standard residual module. The input passes through two 3x3 convolutional layers. A "shortcut" connection routes the original input around these layers, and it is added to their output via element-wise addition before the final ReLU activation.

Benefits of Residual Connections: The add gate acts as a gradient distributor. During backpropagation, the gradient can flow directly through the identity shortcut, bypassing the weight layers. This creates a more direct path for gradients from later layers to earlier ones, mitigating the vanishing gradient problem. The backward path equation demonstrates this:  \frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( 1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i) \right)  The 1 in the expression ensures that the gradient can always propagate, even if the gradients through the weight layers are small.

The Bottleneck Module: For deeper ResNets, a more efficient "bottleneck" module is used, inspired by Inception. It uses a stack of 1x1, 3x3, and 1x1 convolutions. The first 1x1 reduces the dimension, the 3x3 performs the main convolution, and the final 1x1 restores the dimension, saving significant computation.

4.4 Evolving Architectures: DenseNet

Building on the idea of shortcut connections, DenseNet proposes an even more connected architecture. Instead of adding feature maps, a DenseNet layer concatenates the output of a layer with the feature maps of all preceding layers.

* ResNet: x_l = \mathcal{F}_l(x_{l-1}) + x_{l-1}
* DenseNet: x_l = \mathcal{F}_l([x_0, x_1, ..., x_{l-1}])

This encourages feature reuse and further improves gradient flow but requires careful management of feature map dimensionality via "Transition Layers."

Chapter 5: Summary and Key Metrics

The design of neural architectures for embedded systems is an exercise in balancing accuracy with efficiency. This final chapter summarizes the key trends and metrics that guide the development of resource-aware machine learning models.

5.1 Comparative Analysis of Architectures

The evolution of ImageNet-winning architectures demonstrates a clear trend: as models get deeper, they also become more efficient in terms of parameters and computation required to achieve a certain level of accuracy.

Architecture	Layers	Parameters	MACs	ImageNet Top-5 Error
AlexNet	7	61M	724M	16.4%
VGG	16	138M	2.8G	7.4%
Inception	69	24M	5.7G	5.6%
ResNet	152	60M	11G	4.5%

This table shows that deeper is empirically more efficient than wider. Innovations like the Inception and Residual modules allow for much deeper networks (152 layers in ResNet vs. 7 in AlexNet) without a proportional explosion in parameters.

5.2 Core Efficiency Metrics for Embedded ML

When evaluating a neural network for an embedded application, three key metrics related to resource constraints must be considered:

Metric	Description	FC Layer Cost	Convolution Cost	Grouped Conv Cost
Parameters	The number of weights and biases. Dictates model size (storage). Corresponds to weight state.	High	Moderate	Low
Units	The number of activations produced by a layer. Dictates memory for intermediate results. Corresponds to activation state.	Low	High	High
MACs	The number of Multiply-Accumulate operations. Dictates computational load and latency.	High	Very High	Moderate

Architectural choices directly impact these metrics. For instance, grouped convolutions reduce parameters and MACs, while 1x1 convolutions are used to manage the number of units (activations) and subsequently reduce MACs in following layers.

5.3 Concluding Remarks

The foundation of modern architectures for image classification and beyond is the convolution operation. While highly effective, it is computationally intensive. The success of deep learning on embedded platforms hinges on architectural designs that explicitly manage this complexity. Techniques like grouped convolutions, 1x1 bottleneck layers, and residual connections are not just methods for improving accuracy; they are essential tools for aligning the demands of machine learning algorithms with the constraints of hardware. This intricate HW-ML Interplay is the central theme of efficient neural architecture design.





Embedded Machine Learning: A Technical Guide to Unsafe Optimizations

Table of Contents

1. Chapter 1: The High Cost of Deep Learning on Embedded Systems
  * 1.1 Defining the Resource Challenge
  * 1.2 The Trade-Off: Accuracy vs. Efficiency
2. Chapter 2: A Framework for Model Optimization
  * 2.1 The DNN Compute Stack
  * 2.2 Safe vs. Unsafe Optimizations
3. Chapter 3: The Physics of Computation: Energy and Precision
  * 3.1 A Primer on Numerical Formats: Floating-Point vs. Fixed-Point
  * 3.2 The Picojoule Economy: Quantifying Energy Costs
4. Chapter 4: Quantization - Trading Precision for Performance
  * 4.1 Core Concepts of Quantization
  * 4.2 Uniform Quantization
  * 4.3 Non-Uniform Quantization
  * 4.4 Hardware Acceleration and Low-Precision Arithmetic
  * 4.5 Training Strategies for Quantized Networks
  * 4.6 A Survey of Quantization Techniques and Performance
5. Chapter 5: Pruning - Engineering Sparsity in Neural Networks
  * 5.1 The Principle of Pruning: Inspired by Nature
  * 5.2 The Pruning Workflow and Criteria
  * 5.3 Pruning Granularity: Unstructured vs. Structured Sparsity
  * 5.4 The Impact of Retraining and Regularization


--------------------------------------------------------------------------------


Chapter 1: The High Cost of Deep Learning on Embedded Systems

Deep Neural Networks (DNNs) have achieved state-of-the-art performance across numerous domains, from image recognition to natural language processing. However, this success comes at a steep price: DNNs are profoundly compute- and memory-intensive. This inherent cost creates a significant barrier to their deployment on resource-constrained embedded systems, where power, memory, and processing capabilities are strictly limited.

1.1 Defining the Resource Challenge

The primary objective in embedded machine learning is to bridge the gap between the high resource demands of modern DNNs and the modest capabilities of edge devices. To quantify this challenge, consider a standard ImageNet classification task using 224x224 pixel images. A widely used architecture like ResNet50 requires a staggering amount of resources to achieve its baseline accuracy:

* Computational Load: 3.9 GFLOPs (Giga Floating-Point Operations) per inference.
* Model Size (Parameters): 102 MB for storing the network weights.
* Runtime Memory (Activations): 187 MB for storing intermediate feature maps during a forward pass.

These figures are often orders of magnitude greater than the available SRAM or processing power of a typical microcontroller unit (MCU) or a low-power digital signal processor (DSP). The central goal, therefore, is to drastically reduce these computational and memory requirements while preserving the model's predictive quality.

1.2 The Trade-Off: Accuracy vs. Efficiency

There is a direct and observable relationship between a model's complexity and its accuracy. Generally, deeper and wider networks with more parameters and higher computational demands yield better results.

The provided course materials illustrate this through a series of graphs plotting the Top-1 accuracy of various well-known architectures (ResNet, ResNext, DenseNet, MobileNetV1/V2) against key resource metrics:

* Accuracy vs. FLOPs: A graph shows that as the number of GFLOPs increases, the Top-1 accuracy generally improves. For instance, ResNet50 is explicitly marked at 3.87 GFLOPs, achieving approximately 76% accuracy. Architectures like ResNext push beyond 8 GFLOPs to gain a couple of percentage points in accuracy, while the MobileNet family operates below 1 GFLOP, sacrificing some accuracy for immense computational savings.
* Accuracy vs. Parameters: A similar trend is visible when comparing accuracy to the number of model parameters (measured in millions). ResNet50 is shown at 102 MB of parameters. The graph demonstrates that models with more parameters tend to be more accurate, highlighting the memory footprint challenge.
* Accuracy vs. Activations: Runtime memory is another critical constraint. A third graph plots accuracy against the size of activations (in MB), with ResNet50 requiring 187 MB. This illustrates the demand on volatile memory (SRAM or DRAM) during inference.

These visualizations underscore the fundamental trade-off at the heart of embedded ML. To make these powerful models viable at the edge, we must employ optimization techniques that intelligently navigate this landscape, reducing FLOPs and memory footprint with minimal impact on accuracy.

Chapter 2: A Framework for Model Optimization

Optimizing a DNN for embedded deployment is a multi-faceted problem that involves interventions at various levels of abstraction, from high-level architectural choices to low-level hardware implementations. This requires a holistic view of the entire system.

2.1 The DNN Compute Stack

The process of designing and deploying an efficient DNN can be visualized as a layered stack. Each layer presents opportunities for optimization:

1. Neural Architecture: The highest level of abstraction, where the fundamental structure of the network is defined. Choices made here have the most significant impact on overall performance.
2. Compression & Algorithm: This layer involves techniques applied to a given architecture to reduce its complexity. This is the primary focus of this lecture, covering methods like quantization and pruning.
3. Hardware: The physical foundation. The choice of processor—be it a general-purpose CPU, a massively parallel GPU, or a specialized Deep Learning Processing Unit (DPU) like an ASIC or FPGA—dictates the constraints and opportunities for optimization.

The interplay between these layers is crucial. For example, an algorithmic choice like using 1-bit precision (an extreme form of quantization) is only effective if the underlying hardware is designed to execute bitwise operations efficiently.

2.2 Safe vs. Unsafe Optimizations

DNN optimizations can be broadly classified into two categories based on their impact on the model's output.

Safe Optimizations

Safe optimizations are techniques that reduce resource consumption without any possibility of altering the model's predictive accuracy. These are primarily hardware- and system-level improvements that exploit the predictable nature of DNN workloads.

* Key Characteristics:
  * Guaranteed preservation of model accuracy.
  * Focus on efficient data movement and computation.
* Examples:
  * Shorter communication paths: Designing hardware layouts that minimize the distance data has to travel between memory and processing units.
  * Data reuse: Exploiting locality by caching frequently accessed weights or activations to minimize transfers from slower, energy-intensive main memory.
  * Dedicated architectures: Building specialized hardware (e.g., systolic arrays) that are perfectly matched to the core computations in DNNs, like matrix multiplication.

Unsafe Optimizations

Unsafe optimizations are algorithmic modifications that achieve resource savings by changing the model itself, which carries the risk of degrading its accuracy. The "art" of embedded ML lies in applying these techniques aggressively while managing the potential accuracy loss.

* Key Characteristics:
  * Potential impact on model accuracy, which must be carefully measured and mitigated.
  * Focus on reducing the intrinsic complexity of the model and its operations.
* Examples:
  * Compression & Pruning: Reducing the total number of operations and the model's size by eliminating redundant parameters (weights).
  * Quantization: Reducing the numerical precision of operations and operands (e.g., moving from 32-bit floating-point to 8-bit integers or even binary values). This is a cornerstone of efficient embedded inference.

Chapter 3: The Physics of Computation: Energy and Precision

Before diving into specific optimization techniques, it is essential to understand the physical cost of arithmetic and memory access. The choice of numerical format and data movement strategy has a direct and quantifiable impact on energy consumption and latency, which are the ultimate currencies in embedded systems.

3.1 A Primer on Numerical Formats: Floating-Point vs. Fixed-Point

The precision with which numbers are represented is a fundamental design choice.

Floating-Point Arithmetic

Floating-point numbers offer a huge dynamic range, allowing them to represent both very small and very large values. This is why they are the default for training deep learning models. A standard floating-point number is composed of three parts: a sign bit (S), an exponent (E), and a significand or mantissa (F). The value is calculated as:  v = (-1)^S \cdot (1 + F) \cdot 2^{(E-\text{bias})}  The primary drawback is complexity. Floating-point units (FPUs) are larger and more power-hungry than simpler integer units.

Format	Sign (S)	Exponent (E)	Significand (F)	Dynamic Range
float64	1 bit	11 bits	52 bits	~2 \times 10^{\pm 308}
float32	1 bit	8 bits	23 bits	~2 \times 10^{\pm 38}
float16	1 bit	5 bits	10 bits	~1 \times 10^{-4 \text{ to } +5}
bfloat16	1 bit	8 bits	7 bits	Same as float32

The bfloat16 (Brain Floating-Point) format, introduced by Google and Intel, is particularly relevant. It maintains the 8-bit exponent of float32, preserving its large dynamic range, while reducing the significand to 7 bits. This was a direct response to the observation that dynamic range is more critical than precision for many deep learning workloads.

Fixed-Point Arithmetic

Fixed-point representation uses integers where the position of the binary point is implicitly fixed. This results in a smaller dynamic range but offers significant hardware advantages:

* Pros: Addition and multiplication are less complex, leading to smaller and more energy-efficient hardware.
* Cons: The limited dynamic range makes them susceptible to overflow or underflow errors if not handled carefully.

A critical insight for hardware design is how cost scales with bit width (N):

* Addition cost (Energy, Area) scales linearly: \propto N
* Multiplication cost (Energy, Area) scales quadratically: \propto N^2

This quadratic scaling for multipliers provides a powerful incentive to reduce precision wherever possible.

3.2 The Picojoule Economy: Quantifying Energy Costs

To truly appreciate the importance of optimization, one must "learn to love the picojoule." The energy cost of different operations varies by orders of magnitude. The following table, based on data from M. Horowitz, illustrates the stark differences.

Operation Type	Precision	Energy Cost (pJ)	Notes
Integer Add	8-bit	0.03	
	32-bit	0.1	~3x more than 8-bit
Integer Mult	8-bit	0.2	
	32-bit	3.1	~15x more than 8-bit
FP Add	16-bit	0.4	
	32-bit	0.9	
FP Mult	16-bit	1.1	
	32-bit	3.7	
Memory Access	8kB Cache (SRAM)	10	
	32kB Cache (SRAM)	20	
	1MB Cache (SRAM)	100	
	DRAM Access	1300 - 2600	~100x more than cache

Key Takeaways:

1. Computation is cheap, memory access is expensive. A single access to external DRAM can cost more than a thousand simple arithmetic operations. This reality makes exploiting locality and minimizing memory transfers the single most important principle in high-performance system design.
2. Precision has a high cost. A 32-bit integer multiply is about 15 times more expensive than an 8-bit multiply.
3. Addition is much cheaper than multiplication. This motivates the use of architectures or techniques that can substitute multiplications with additions or even cheaper bitwise operations.

Chapter 4: Quantization - Trading Precision for Performance

Quantization is an unsafe optimization technique that involves mapping continuous or high-precision values to a smaller set of discrete, low-precision values. It is one of the most effective methods for reducing memory footprint, decreasing latency, and lowering power consumption for DNN inference.

4.1 Core Concepts of Quantization

At its core, a quantizer (Q) is a piece-wise constant function that maps an input value x from a given interval to a corresponding quantization level q_l.

A diagram from the lecture materials illustrates this by showing a histogram of a typical bell-shaped weight distribution. Dashed vertical lines represent "interval thresholds" (\delta^l). Any weight value falling between two thresholds is mapped to a single, representative value for that interval.

For a real number a_i \in [0, 1], a uniform k-bit quantizer can be defined by the formula:  a_q^i = \frac{1}{2^k - 1} \cdot \text{round}((2^k - 1)a_i)  This function maps the continuous input a_i to one of 2^k discrete levels. For example, with k=2 bits, the continuous range [0, 1] would be mapped to the discrete set {0.00, 0.25, 0.50, 0.75, 1.00}.

Quantization can be applied to weights, activations, and even gradients during training. The goal is to find the lowest possible bit-width that maintains an acceptable level of accuracy.

4.2 Uniform Quantization

In uniform quantization, the quantization levels are equidistant, meaning the step size \Delta between any two adjacent levels is constant: q_{i+1} - q_i = \Delta, \forall i.

* Advantages:
  * Simplicity: It is easy to store and compute. The quantized values can be represented by simple integers requiring only \log_2(L) bits (where L is the number of levels), and the constant step size \Delta can be stored once per layer or tensor.
  * Efficiency: If weights (W) and activations (A) are quantized identically, computations become simple integer arithmetic, which is highly efficient on most hardware.
* Disadvantages:
  * Limited Capacity: A uniform grid may not be the most efficient way to represent data that has a non-uniform distribution (like the bell-shaped curve of typical DNN weights). It may waste representation capacity on ranges with few values and lack fidelity in dense regions.

A simple yet powerful example of uniform quantization is binary quantization, where all values are mapped to just two levels, often using the sign function:  Q(x) = \begin{cases} +1 & \text{if } x \geq 0 \\ -1 & \text{if } x < 0 \end{cases} 

4.3 Non-Uniform Quantization

Non-uniform quantization uses quantization levels that are not equidistant. This allows the representation to adapt to the underlying distribution of the data, placing more quantization levels in regions where values are most frequent.

* Advantages:
  * Improved Model Capacity: By matching the representation to the data distribution, it can achieve higher accuracy than uniform quantization for the same number of bits.
* Disadvantages:
  * Complexity: It is more complex to implement. In addition to the \log_2(L) bits for the quantized values, the actual quantization levels q_l must also be stored (e.g., in a lookup table). Computation may require table lookups, which can add overhead.

A common technique is to make the quantization levels (or scaling factors) trainable, allowing them to be learned during the training process. For example, a ternary quantizer might use a trainable threshold \Delta^l to map weights:  w_i^l = \begin{cases} W_p^l & \text{if } w^l > \Delta^l \\ 0 & \text{if } |w^l| \leq \Delta^l \\ -W_n^l & \text{if } w^l < -\Delta^l \end{cases}  Here, the threshold \Delta^l can be defined as a fraction of the maximum weight value, where the fraction is a learnable parameter.

4.4 Hardware Acceleration and Low-Precision Arithmetic

The true power of quantization is unlocked by hardware designed to exploit low-precision data. The core operation in most DNNs is the Multiply-Accumulate (MAC), which is central to matrix multiplication and convolution.

A provided diagram compellingly contrasts MAC units at different bit-widths:

* 32-Bit MAC: A standard unit takes two 32-bit inputs (A, B) and performs a fused multiply-add (FMA) with a 32-bit accumulator C. This takes a certain number of cycles (e.g., 6).
* 8-Bit MAC: Within the same area or cycle count, multiple 8-bit operations can be performed in parallel. The diagram shows four 8-bit multipliers operating simultaneously. Their results are then summed by a tree of adders. Although the total cycle count might increase (e.g., to 36) due to the multi-stage process, the throughput (operations per second) is significantly higher.
* 1-Bit MAC: At the extreme, multiplication becomes a simple bitwise XNOR operation, and accumulation becomes counting the number of set bits (popcount). The diagram shows 32 1-bit operations happening in parallel, using EOR (equivalent to XNOR for accumulation purposes) and CNT (count) blocks. This results in a massively parallel and energy-efficient design, completing in fewer cycles (e.g., 15) than the 8-bit version for a vector operation.

XNOR-Based Binary Multiplication

For binary networks where weights and activations are represented by {-1, +1}, the expensive multiplication can be replaced. By mapping {-1, +1} to {0, 1}, the dot product can be calculated efficiently. The multiplication of two binarized values is equivalent to the XNOR operation.

The dot product c of two binary vectors a and b of length N can be calculated as:  c = a \cdot b = 2 \cdot \text{popc}(\text{xnor}(a, b)) - N  Where popc is the population count (counting the number of 1s). This transforms the core DNN operation from a sequence of floating-point multiplies and adds into simple, fast, and low-power bitwise logic.

4.5 Training Strategies for Quantized Networks

Applying quantization naively can significantly degrade accuracy. To counteract this, several training strategies have been developed.

1. Post-Training Quantization (PTQ): This is the simplest method. A fully trained, high-precision model is converted to a lower precision after training is complete. It requires a small "calibration dataset" (a representative sample of the training data) to determine the quantization parameters (e.g., scale and zero-point). While easy to apply, PTQ often results in a noticeable loss of accuracy, especially for very low bit-widths.
2. Quantization-Aware Training (QAT): This method simulates the effects of quantization during the training process itself.
  * Forward Pass: Weights and/or activations are "fake quantized." They are quantized to a low precision (e.g., 8-bit integer) and then immediately de-quantized back to high precision (e.g., 32-bit float) before being used in the computation. This injects quantization noise into the training, forcing the model to learn weights that are robust to this effect.
  * Backward Pass: Gradients are computed using the high-precision values, allowing for stable training updates.
3. A significant challenge in QAT is that the quantization function is piece-wise constant, meaning its gradient is zero or undefined everywhere. This would stall training. To overcome this, the Straight-Through Estimator (STE) is used. STE simply passes the gradient through the quantization function as if it were an identity function (i.e., assuming a gradient of 1). The gradient for a weight w is approximated as:  \frac{\partial\mathcal{L}}{\partial w} = \frac{\partial\mathcal{L}}{\partial f} \frac{\partial f}{\partial w} \approx \frac{\partial\mathcal{L}}{\partial f} \tilde{f}'(w)  where \tilde{f}'(w) is a surrogate derivative, often just set to 1. This allows the high-precision weights to be updated based on gradients computed from the quantized weights.
4. Re-training/Fine-tuning: A hybrid approach where a pre-trained model is quantized and then fine-tuned for a few epochs using QAT. This can often recover much of the accuracy lost during initial quantization.

4.6 A Survey of Quantization Techniques and Performance

Numerous quantization schemes have been proposed, varying in which components they quantize (weights, activations, or both) and to what bit-width.

Method	Weights (W)	Activations (A)
BNN	{-1,+1}	{-1,+1}
XNOR-Net	{-S,+S}	{-1,+1}
DoReFa-Net	{-S,+S}	{0,+1} (k-bit)
TWN	{-S,0,+S}	float32
TTQ	{-Sn,0,+Sp}	float32
HWGQ	XNOR	2-bit

Performance data for these methods on the AlexNet/ImageNet task shows that while extreme 1-bit quantization (BNN, XNOR) suffers a significant accuracy drop, more moderate schemes can approach the full-precision baseline. For example, TTQ (with 2-bit weights and 32-bit activations) achieves 79.7% Top-5 accuracy, very close to the 80.3% of the 32-bit baseline.

Graphs in the lecture illustrate these trade-offs:

* An "Improvement factor" graph shows that memory footprint improves exponentially as operand bit-width decreases (e.g., a 2^4 factor improvement going from 16-bit to 1-bit). Latency also improves, though less dramatically.
* A "Test error" graph shows that as the number of bits increases from 1 to 3, the test error for various methods (Lq-Net, BNN, DoReFa, etc.) drops, approaching the full-precision baseline error rate.

Chapter 5: Pruning - Engineering Sparsity in Neural Networks

Pruning is another powerful unsafe optimization technique aimed at reducing model size and computational cost. It operates by removing "unimportant" connections or neurons from a trained network, effectively setting their corresponding weights to zero.

5.1 The Principle of Pruning: Inspired by Nature

The concept of pruning is biologically inspired. A diagram titled "Evolution of Human Brain During Life" shows that the density of synaptic connections in the human brain peaks around age 6 and is then gradually reduced by age 14. This process of synaptic pruning is believed to refine and optimize neural circuits. Similarly, DNNs are often over-parameterized, containing significant redundancy that can be removed without harming, and sometimes even improving, generalization.

5.2 The Pruning Workflow and Criteria

Pruning is not a single action but a process, typically involving three stages:

1. Train: A standard, dense network is trained to convergence.
2. Prune: A certain fraction of the network's weights are set to zero based on a specific criterion.
3. Fine-tune: The now-sparse network is retrained for a few epochs to allow the remaining weights to adjust and recover any accuracy lost during pruning.

This process is often performed iteratively (Train -> Prune -> Fine-tune -> Prune -> Fine-tune -> ...) to achieve higher levels of sparsity without catastrophic drops in accuracy. This contrasts with one-shot pruning, where the entire pruning process happens at once.

The critical question is which connections to remove. Common criteria include:

* Weight Magnitude Pruning: This is the simplest and most common method. Weights with the smallest absolute values are removed, based on the heuristic that they have the least influence on the network's output. A threshold t is set, and any weight w_i where |w_i| \le t is pruned.
* Gradient Magnitude Pruning: This method considers not just the weight's size but also its impact on the loss function. It prunes weights based on the magnitude of the weight multiplied by its corresponding gradient: |w_i \cdot g_i| \le t.

5.3 Pruning Granularity: Unstructured vs. Structured Sparsity

The way in which weights are removed has profound implications for hardware performance.

Unstructured (Fine-Grained) Pruning

This approach removes individual weights anywhere in the network based on the pruning criterion.

* Pros: Offers the highest potential for accuracy at a given sparsity level, as it provides maximum flexibility.
* Cons: Extremely difficult to accelerate on parallel hardware like GPUs and CPUs. A sparse matrix resulting from unstructured pruning requires an index to store the location of each non-zero element. This leads to indirect memory accesses, which are inefficient and break memory coalescing patterns, often resulting in performance that is no better, or even worse, than the original dense matrix. The Compressed Sparse Row (CSR) format, which uses data arrays (d), column indices (i), and row pointers (r), exemplifies this overhead.

Structured (Coarse-Grained) Pruning

This approach removes entire groups of weights at once, such as entire filters/channels in a convolutional layer or rows/columns in a fully connected layer.

* Pros: Creates smaller, dense matrices that are perfectly suited for hardware acceleration. It preserves the regular computational patterns that parallel processors are designed for.
* Cons: Less flexible than unstructured pruning, which can lead to a greater loss in accuracy for the same number of removed parameters.

Advanced techniques aim to make this structure learnable. Parameterized Structured Pruning divides a weight tensor W into sub-tensors, each representing a structure. Each structure w_i is associated with a learnable parameter \alpha_i. The structure is kept or pruned based on whether |\alpha_i| is above a threshold \epsilon. This is implemented using a thresholding function v_i(\alpha_i):  w_i^{qi} = w_i \cdot v_i(\alpha_i) \quad \text{where} \quad v_i(\alpha_i) = \begin{cases} 0 & \text{if } |\alpha_i| < \epsilon \\ \alpha_i & \text{if } |\alpha_i| \geq \epsilon \end{cases}  Since this function is not differentiable at the threshold, STE is used during backpropagation to update the \alpha_i parameters, allowing the network to learn which structures to keep.

5.4 The Impact of Retraining and Regularization

Fine-tuning after pruning is mandatory to recover model accuracy. The process of pruning and retraining fundamentally alters the weight distribution of the model.

A diagram shows two histograms:

1. Weight distribution before pruning: A typical Gaussian-like distribution centered narrowly around zero.
2. Weight distribution after pruning and retraining: The distribution becomes bimodal. The peak at zero is gone (due to pruning), and the remaining weights have been pushed away from zero during retraining, forming two distinct clusters. This shows that retraining strengthens the remaining connections to compensate for those that were removed.

The choice of regularization during training also plays a role.

* L1 Regularization (\mathcal{R}_{L1}(w) = \sum_j |w_j|) encourages sparsity by pushing weights towards exact zero.
* L2 Regularization (\mathcal{R}_{L2}(w) = \frac{1}{2} \sum_j w_j^2) encourages small weights but does not force them to be exactly zero.

Studies on AlexNet/ImageNet show that with retraining, L2 regularization ultimately performs best for pruning. Without retraining, L1 is a better option, but the key takeaway is that retraining is essential for achieving high performance with pruned models.




Embedded Machine Learning: A Technical Reference

Table of Contents

1. Chapter 1: Unsafe Optimizations II - Quantization
  1. 1.1 Introduction to Model Compression
  2. 1.2 A Hardware-Aware Metric: Bit Operations (BOPS)
  3. 1.3 Fundamentals of Quantization
  4. 1.4 A Taxonomy of Quantization Techniques
  5. 1.5 Practical Quantization: A ResNet Case Study
  6. 1.6 The Challenge of Real-World Data: Calibration Mismatch
  7. 1.7 Advanced Quantization-Aware Training (QAT) Methods
  8. 1.8 Hardware-Software Interplay: Architectures and Methods
  9. 1.9 Common Pitfalls and Best Practices
  10. 1.10 Automated Search for Compression Parameters


--------------------------------------------------------------------------------


Chapter 1: Unsafe Optimizations II - Quantization

This chapter provides a deep dive into quantization, a critical model compression technique for deploying machine learning models on resource-constrained embedded systems. The central theme is the interplay between ML algorithms, their numerical representation, and the underlying hardware's limitations in terms of memory, energy, and latency.

1.1 Introduction to Model Compression

Deploying complex neural networks on embedded devices necessitates model compression to meet strict constraints on memory footprint, computational power, and energy consumption. The two primary families of compression techniques are Pruning and Quantization.

* Pruning: This technique involves removing redundant parameters (weights or neurons) from a trained network. The goal is to reduce the model's size and the number of computations required during inference. As described in a diagram, pruning can be unstructured, where individual connections (synapses) are removed based on criteria like their magnitude, or structured, where entire neurons or filters are removed. While effective, the efficiency gains from pruning are highly dependent on hardware support for sparse computations.
* A diagram contrasts a fully connected network layer with pruned versions. "Before pruning," all neurons are interconnected. "After pruning" shows two outcomes:
  1. Pruning synapses: Individual connections are removed, leading to a sparse weight matrix.
  2. Pruning neurons: Entire neurons and all their associated connections are eliminated, reducing the layer's dimensions.
* Quantization: This technique reduces the numerical precision of a model's parameters (weights) and, optionally, its intermediate calculations (activations). By representing floating-point numbers (e.g., Float32) with lower-bit integers (e.g., INT8, INT4, or even binary values), quantization drastically reduces memory footprint and can enable faster, more energy-efficient integer-based arithmetic on compatible hardware.

The choice of data type, number format, and bit width, as well as whether the quantization is applied homogeneously across the model or heterogeneously (per-layer, per-filter, etc.), are key design decisions. Like pruning, the ultimate efficiency of quantization is deeply tied to the target hardware architecture.

1.2 A Hardware-Aware Metric: Bit Operations (BOPS)

To fairly compare different model compression techniques, especially those using custom or low-precision data types, a suitable hardware-abstract metric is required. The traditional metric of Multiply-Accumulate operations (MACs) is often insufficient, as it does not capture the cost variations associated with different bit widths.

The Bit Operations (BOPS) metric provides a more accurate measure of computational complexity for fixed-point arithmetic. For a standard convolutional layer, the BOPS can be approximated by considering the bit widths of weights and activations.

Let's define the parameters for a convolutional layer:

* b_w: bit width of weights
* b_a: bit width of activations
* n: number of input channels
* m: number of output channels (filters)
* k: filter size (e.g., k \times k)

The total number of MACs per output element is nk^2. The maximum value of the output accumulator depends on the input bit widths and the number of accumulated products. For a binary-coded accumulator, the required bit width (b_o) can be estimated as:

b_o = b_a + b_w + \log_2(nk^2)


The total BOPS for the convolutional layer, accounting for both multiplications and accumulations, is then given by:

\text{BOPS}_{\text{conv}} \approx mnk^2 \left( \underbrace{b_a b_w}_{\text{mult.}} + \underbrace{b_o}_{\text{acc.}} \right)


This formula highlights the HW-ML Interplay: reducing the bit width of weights (b_w) and activations (b_a) quadratically reduces the multiplication cost and linearly reduces the accumulation cost, leading to significant computational savings. A scatter plot of various models illustrates this trade-off, showing model accuracy versus billions of bit operations. Techniques like UNIQ, QNN, and XNOR aim to push models into the high-accuracy, low-BOPS regime.

1.3 Fundamentals of Quantization

Quantization is the process of mapping a continuous or large set of values to a smaller, discrete set. In machine learning, this typically means mapping 32-bit floating-point numbers to low-bit integers.

The core quantization mapping is defined by the following equation:

q = Q(x) = \text{clip}(\text{round}(\frac{x}{s} + z), q_{\text{min}}, q_{\text{max}})


Where:

* Glossary of Terms:
  * x: The original, continuous (floating-point) value to be quantized.
  * q: The quantized integer representation of x.
  * s: The scale factor, a positive float that determines the step size or resolution of the quantization.
  * z: The zero-point, an integer that specifies which quantized value corresponds to the real value of 0. This allows for an offset in the mapping.
  * q_min, q_max: The minimum and maximum allowed values for the target integer data type (e.g., -128 and 127 for signed INT8).

To recover the original value (with some error), a dequantization step is performed:

\hat{x} = (q - z) \cdot s


Here, \hat{x} is the reconstructed floating-point approximation of x. The difference between x and \hat{x} is the quantization error, which arises from both rounding and clipping.

A visual representation of this process is shown in a graph titled "Effect of Int2 Quantization on Function Approximation," where a smooth sine wave (Float32) is approximated by a coarse, step-like function (Int2), clearly showing the information loss inherent in the process.

Symmetric vs. Asymmetric Quantization

The choice of s and z defines the quantization scheme.

* Symmetric Quantization: The range of real values is mapped symmetrically around zero. This is achieved by setting the zero-point z=0. The scale is typically calculated as:  s = \frac{\max(|x|)}{q_{\text{max}}}  This scheme is commonly used for weights, which often have a distribution centered around zero.
* Asymmetric Quantization: The range is not necessarily centered at zero. This requires both a scale and a non-zero zero-point. The parameters are calculated as:  s = \frac{x_{\text{max}} - x_{\text{min}}}{q_{\text{max}} - q_{\text{min}}}, \quad z = \text{round}(q_{\text{min}} - \frac{x_{\text{min}}}{s})  This scheme is often used for activations, especially after a ReLU function, where all values are non-negative.

Static vs. Dynamic Quantization

The method for determining the range (x_{\text{min}}, x_{\text{max}}) also defines a key characteristic.

* Static Scaling: The scale (s) and zero-point (z) are pre-computed offline using a representative calibration dataset. These parameters are then fixed and used for all inferences at runtime. This approach has zero runtime overhead but is vulnerable if the deployment data distribution differs from the calibration data.
* Dynamic Scaling: The range (x_{\text{min}}, x_{\text{max}}), and thus s and z, are computed on-the-fly for each input or batch at runtime. This is more robust to varying input distributions but introduces computational overhead during inference.

1.4 A Taxonomy of Quantization Techniques

The design space of quantization is vast. Understanding its taxonomy is crucial for making informed decisions.

Dimension	Options	Description & Trade-offs
Procedure (Timing)	Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)	PTQ is applied to a pre-trained model without retraining. It is fast and simple, often effective for INT8. QAT simulates quantization effects during the training loop, allowing the model to adapt to the precision loss. It is more complex and time-consuming but necessary to regain accuracy with aggressive quantization (e.g., INT4).
Target	Weight-Only vs. Weights + Activations	Weight-only quantization primarily reduces model size and memory bandwidth. Computations are often still performed in higher precision. Weights + Activations (W+A) quantization is required to enable fully integer-based pipelines, which yields maximum speedups and energy savings on compatible hardware.
Range Determination	Static vs. Dynamic	Static uses a fixed range from a calibration set, offering fast inference but sensitivity to data distribution shifts. Dynamic calculates ranges at runtime, providing robustness at the cost of latency overhead.
Granularity	Per-Tensor vs. Per-Channel vs. Per-Group	Per-tensor uses a single s and z for an entire weight tensor. It is simple but can be suboptimal if value distributions vary widely within the tensor. Per-channel (for CNNs) computes a separate s and z for each output channel's filter weights, offering much better robustness. Per-group is a trade-off, sharing a scale across a small group of channels.

A practical escalation strategy for applying quantization is:

1. Start with INT8 PTQ.
2. If accuracy drops, improve the calibration process (e.g., use percentile ranges instead of min/max) or increase granularity (e.g., move from per-tensor to per-channel).
3. Consider using mixed precision, keeping sensitive layers at a higher precision.
4. As a final resort, use the more complex QAT process to retrain the model.

1.5 Practical Quantization: A ResNet Case Study

Let's examine the quantization of the first convolutional layer (conv1) of a ResNet model to illustrate these concepts.

INT8 Quantization Example

* ResNet conv1 Weights:
  * A histogram of the original Float32 weights shows a bell-shaped distribution sharply peaked at zero, with values ranging from approximately -1.0 to 1.0.
  * Method: Symmetric min/max quantization for a signed INT8 range [-128, 127].
  * Parameters: scale = 0.007972, zero_point = 0.
  * Result: The quantized integer codes (shown in an inset plot) mirror the original distribution. The dequantized histogram (overlaid on the original) shows a very close approximation, indicating minimal information loss.
* ResNet ReLU Activations:
  * A histogram of the Float32 activations shows a distribution skewed to the right, starting at zero (due to ReLU) with a long tail. Outliers are present.
  * Method: Asymmetric min/max quantization for a signed INT8 range [-128, 127] (though values will be positive).
  * Parameters: scale = 0.01287, zero_point = -128.
  * Result: The dequantized histogram again closely follows the original, demonstrating that INT8 is often sufficient for both weights and activations in CNNs with proper calibration.

INT4 Quantization Example (Aggressive)

* ResNet conv1 Weights:
  * Method: Symmetric min/max for a signed INT4 range [-7, 7].
  * Parameters: scale = 0.1452, zero_point = 0.
  * Result: The dequantized histogram is much coarser. The values are visibly grouped into a few discrete bins, indicating significant precision loss. The large scale value means low resolution.
* ResNet ReLU Activations:
  * Method: Asymmetric min/max for an unsigned INT4 range [0, 15].
  * Parameters: scale = 0.2187, zero_point = 0.
  * Result: The quantization effect is dramatic. The smooth distribution is replaced by a few sharp peaks, showing that many distinct activation values have been mapped to the same integer code. This level of aggressive quantization typically requires QAT to maintain acceptable accuracy.

1.6 The Challenge of Real-World Data: Calibration Mismatch

A core assumption of static activation quantization is that the data distribution seen during deployment will match the distribution of the calibration set (calibration ≈ deployment). When this assumption is violated, the pre-computed scale and zero-point become suboptimal, leading to accuracy degradation.

Sources of Mismatch:

* Lighting/Exposure Shifts: Changes in lighting can cause the "right tail" of activation distributions to grow, as more high-magnitude activations appear.
* Sensor/Optics Changes: Different cameras have unique noise profiles, color responses, and lens artifacts, altering the overall input distribution.
* Compression Artifacts: Using different image formats (e.g., RAW vs. JPEG) or streaming introduces artifacts like blocking and ringing, which can distort activation histograms.

Effect of Mismatch: An illustration shows activation histograms for ResNet-18 on CIFAR-10 under "Clean calibration" and "Mismatch calib" conditions. Under mismatch, the distribution's tail extends further. This forces a min/max-based calibration to select a larger range, which in turn increases the scale factor. A larger scale means lower resolution for the bulk of the values, increasing quantization error. The example shows the min/max scale increasing from 0.01287 (clean) to 0.01672 (mismatch).

Mitigation Strategies:

1. Diversify the calibration set to be more representative of real-world conditions.
2. Use more robust range selection methods like percentile or MSE-based calibration, which are less sensitive to outliers.
3. Increase quantization granularity (e.g., per-channel scales).
4. Employ mixed precision, keeping sensitive layers (often the first and last) in higher precision.

1.7 Advanced Quantization-Aware Training (QAT) Methods

When PTQ is insufficient, QAT methods integrate quantization into the training process. These techniques allow the model to learn weights that are more robust to the effects of low-precision arithmetic.

Trained Ternary Quantization (TTQ)

TTQ quantizes weights to three values: positive (W_p), negative (-W_n), and zero. It uniquely learns the optimal scaling factors W_p and W_n during training.

The process, illustrated in a flow diagram, is as follows:

1. Forward Pass (Inference Time): Full-precision weights are quantized to an intermediate ternary representation {-t, 0, +t} based on a threshold \Delta_l. This is then scaled by the learned values W_p and W_n to produce the final ternary weight.
2. Backward Pass (Training Time): The loss gradient is backpropagated through the network. Crucially, TTQ computes two sets of gradients:
  * gradient1: Propagates back to the full-precision weights to learn the ternary assignments.
  * gradient2: Propagates back to the scale factors W_p and W_n to learn the optimal ternary values.

The quantization function is:  \tilde{w}_i = \begin{cases} W_p & : w_i > \Delta_l \ 0 & : |w_i| \le \Delta_l \ -W_n & : w_i < -\Delta_l \end{cases}  where the threshold \Delta_l is a hyperparameter: \Delta_l = t \cdot \max(|w|); t \in [0,1].

DoReFa-Net

DoReFa-Net is a comprehensive framework for training networks with arbitrary bit widths for weights (W), activations (A), and gradients (G). It uses deterministic quantization for weights and activations and stochastic quantization for gradients. It provides a thorough treatment of the Straight-Through Estimator (STE), a key technique for backpropagating gradients through non-differentiable quantization functions. Experimental results on AlexNet show the performance for various W-A-G configurations, highlighting the importance of gradient precision.

LQ-Nets (Learned Quantization Networks)

LQ-Nets introduce a learnable quantizer that creates data-adaptive, non-uniform quantization levels. This can significantly reduce quantization error compared to uniform methods.

The key idea is to represent a quantized number not with a fixed-power-of-2 basis, but as a linear combination of a trainable basis vector v \in \mathbb{R}^K:  q_l = v^T b_l  where b_l is a binary coding vector. By making v a trainable parameter, the network learns the optimal placement of quantization levels for its specific weight distribution. This combines the benefits of uniform quantization (efficiency) with non-uniform quantization (accuracy). Diagrams illustrate how a 2-bit and 3-bit learned basis can create a non-uniform staircase function of quantization levels.

1.8 Hardware-Software Interplay: Architectures and Methods

The true benefits of quantization are realized when the algorithm is co-designed with the target hardware in mind.

HW Excursion: Bit-Serial Multiplication

For hardware designed to handle arbitrary precision, bit-serial computation is a viable option. Instead of a parallel multiplier, operations are serialized over the bits of the operands.

A multiplication c = a \cdot b can be decomposed into bit-level operations. If a and b are N-bit and M-bit fixed-point integers respectively, the product is:  c = a \cdot b = \sum_{n=1}^{N} \sum_{m=1}^{M} 2^{n+m} \cdot \text{popc}(\text{and}(a_n, b_m))  The complexity is O(NM), directly proportional to the bit widths. While a single bit-serial operation has high latency, its simple logic is suitable for massive parallelism, enabling competitive throughput. A graph comparing operand bit width to performance improvements shows that for latency with bit-serial logic, the improvement factor scales exponentially as bit width decreases.

DeepChip's Reduce-and-Scale (RAS) Quantization

The DeepChip project focuses on model compression for resource-constrained devices like mobile ARM processors. Its Reduce-and-Scale (RAS) method is a prime example of HW-aware quantization.

RAS combines several techniques:

1. Weight Quantization: Uses a TTQ-like approach to quantize weights to ternary values {-Wn, 0, Wp}. The scale factors W_p and W_n are independent, asymmetric, and trained via SGD.
2. Activation Quantization: Activations are first bounded using a Bounded ReLU function (a' = \text{clip}(a, 0, 1)) and then quantized to a k-bit fixed-point representation, similar to DoReFa-Net.
3. Space-Efficient Data Structures: Instead of storing the full ternary weight matrix, RAS uses a parameter converter to create a compressed representation. This involves run-length encoding principles, storing only the signs and the distances (indices) between non-zero values. This reduces cardinality and is amenable to further compression like Huffman coding.
4. Efficient Operator Library: The core computation is reformulated to avoid costly multiplications. The output is calculated by summing the relevant activations and performing only two multiplications by the scale factors W_p and W_n per output channel.  c = W_p^l \cdot \sum_{i \in i_p^l} a_i + W_n^l \cdot \sum_{i \in i_n^l} a_i  This leverages the fact that integer additions are significantly cheaper and more energy-efficient than multiplications on typical processors (e.g., on an ARM chip, an int16 ADD is ~2x faster and >30x more energy-efficient than an int16 FMA).

Results on AlexNet/ImageNet show that DeepChip's method achieves higher accuracy (79.0% Top-5) and a smaller memory footprint (25 MB) than a baseline BNN or standard INT8 quantization, while maintaining a competitive inference rate.

1.9 Common Pitfalls and Best Practices

Applying quantization effectively requires avoiding common mistakes.

Common Pitfalls:

* Taxonomy Confusion: Incorrectly assuming PTQ is always weight-only.
* Unrepresentative Calibration Data: Using a calibration set that doesn't reflect real-world deployment conditions (lighting, sensors, etc.).
* Blind Min/Max for Activations: Allowing outliers to inflate the quantization range, which wastes resolution for the majority of values.
* Ignoring Signed/Unsigned: Failing to use unsigned integers for non-negative data like ReLU activations.
* Wrong Granularity Default: Using per-tensor quantization for CNN weights, where per-channel is a much safer and more robust baseline.
* Ignoring Sensitive Layers: Quantizing all layers uniformly, when the first and last layers often require higher precision.
* Not Measuring the Right Signals: Failing to analyze activation histograms, saturation rates, and layer-wise error to diagnose issues.

Rule of Thumb: Start with a simple, robust baseline: uniform INT8 PTQ, with per-channel granularity for weights and percentile-based calibration for activations. Only escalate to finer granularity, mixed precision, or full QAT if this baseline fails.

1.10 Automated Search for Compression Parameters

The vast design space of compression (pruning ratios, bit widths per layer, quantization schemes) makes manual tuning difficult. Automated approaches use search algorithms to find the optimal compression policy for a given hardware target.

* GALEN: Combines quantization and pruning by using reinforcement learning (RL) to predict a compression policy. It performs a layer-wise sensitivity analysis and incorporates real-world hardware latency measurements, going beyond simple metrics like FLOPs or BOPs.
* HAQ (Hardware-Aware Automated Quantization): Also uses RL to find a mixed-precision quantization policy (2-8 bits) under a latency budget, which is approximated using a lookup table.

These methods represent the frontier of model compression, where hardware expertise is encoded into an automated algorithm that can navigate the complex trade-offs between accuracy, latency, and model size.




Embedded Machine Learning: A Technical Guide to Efficient Inference

Table of Contents

1. Chapter 1: The Illusion of Sparsity: From Theory to Hardware-Accelerated Pruning
  * 1.1 The Sparsity-Speed Fallacy
  * 1.2 Understanding Sparsity Formats and Granularity
  * 1.3 The Economics of Sparsity: A Break-Even Model
  * 1.4 Practical Pruning Methodologies in PyTorch
  * 1.5 The Path to Real Speedup: Deployment Strategies
  * 1.6 Advanced Techniques: Learning Sparsity with PSP
  * 1.7 Benchmarking and Measurement: A Practical Guide
  * 1.8 Case Study: Pruning a Simple CNN on MNIST
  * 1.9 Strategic Pruning: Layer-Wise Sparsity Allocation
  * 1.10 Conclusion: A Systems-Level View of Compression


--------------------------------------------------------------------------------


Chapter 1: The Illusion of Sparsity: From Theory to Hardware-Accelerated Pruning

This chapter bridges the critical gap between the theoretical promise of model compression and the practical realities of hardware performance. We will dissect the concept of sparsity—the introduction of zero-valued weights into a neural network—and explore why simply increasing the number of zeros often fails to deliver corresponding speedups in inference. The central theme is the HW-ML Interplay: the best optimization is not the one with the highest theoretical compression, but the one your hardware can execute efficiently.

1.1 The Sparsity-Speed Fallacy

A common misconception in machine learning optimization is that sparsity directly translates to speed. For instance, if a model has 80% of its weights pruned to zero, one might intuitively expect a 5x speedup. However, the practical reality is often closer to 1x—no speedup at all—without substantial, hardware-aware effort.

This discrepancy arises because modern processors and ML frameworks are highly optimized for dense, structured computations like GEMM (General Matrix-Matrix Multiplication). The performance of a sparse model depends entirely on how that sparsity is represented and executed.

There are three primary approaches to handling sparsity, each with vastly different performance implications:

1. Mask-Only Sparsity: This is the most common method within ML frameworks like PyTorch. A binary mask is applied to the weight tensor, effectively zeroing out certain values during computation (e.g., y = x ⋅ (W ⊙ M)). However, the underlying weight tensor W remains dense in memory, and the computation is still performed by a dense GEMM kernel. The hardware sees the same tensor shapes and executes the same schedule, resulting in little to no speedup.
2. Structured Pruning (Rewiring): This method removes entire structural components of the network, such as neurons, attention heads, or convolutional channels. This physically alters the model's architecture, resulting in smaller weight matrices and activation tensors. These smaller tensors can then be processed by standard, efficient dense kernels, leading to real, measurable speedups.
3. Hardware-Structured Sparsity: This advanced technique enforces a specific, hardware-friendly sparsity pattern (e.g., N:M sparsity, where N out of M consecutive weights are non-zero). The weights are then packed into a compact format, and a specialized sparse kernel that understands this pattern is used for computation. This provides significant speedups but requires direct hardware support.

The fundamental reason sparsity does not equal speed is that computational work is not skipped unless the kernel is explicitly designed to do so. Overheads associated with sparse formats—such as storing indices, managing indirection, and load imbalance across parallel processing units—can easily negate any savings from reduced computations. Furthermore, in memory-bound scenarios, where performance is limited by data movement rather than computation, removing floating-point operations (FLOPs) provides no benefit.

1.2 Understanding Sparsity Formats and Granularity

The effectiveness of pruning is deeply tied to its granularity, which dictates the scale at which weights are removed. This choice has direct consequences for both model accuracy and hardware efficiency.

* Fine-Grained (Unstructured) Pruning: Removes individual weights, offering the highest potential for preserving accuracy. However, the resulting random sparsity pattern is difficult for massively parallel processors like GPUs to exploit. These processors thrive on structured, predictable computation, and the overhead of skipping individual zero-valued weights often outweighs the benefit.
* Coarse-Grained (Structured) Pruning: Removes groups of weights (e.g., entire rows, columns, or channels). This structure is highly compatible with parallel hardware, avoiding performance issues like memory access misalignment (coalescing) and branch divergence. This is the key to achieving practical speedups on modern CPUs and GPUs.

To handle unstructured sparsity, specialized data formats are required. A common example is the Compressed Sparse Row (CSR) format.

Glossary: Compressed Sparse Row (CSR) A format for storing sparse matrices that avoids storing zero elements. It uses three arrays:

1. Data Array (d): A flat array containing all non-zero values.
2. Column Index (i): An array storing the column index for each value in d.
3. Row Pointer (r): An array of size M+1 (for M rows) where r[k] stores the index in d where row k begins.

Consider the following dense 4x4 matrix D with 8 non-zero elements: D = [[0, 5, 3, 0], [6, 1, 0, 4], [0, 0, 0, 0], [2, 0, 1, 4]]

In dense format, this requires 4 × 4 = 16 units of storage. In CSR format, it would be represented as:

* d ∈ ℝ⁸ = (5, 3, 6, 1, 4, 2, 1, 4)
* i ∈ ℝ⁸ = (1, 2, 0, 1, 3, 0, 2, 3)
* r ∈ ℝ⁵ = (0, 2, 5, 5, 8)

The CSR representation requires 8 (data) + 8 (indices) + 5 (pointers) = 21 units of storage. In this case, the overhead from metadata (i and r) makes the sparse format larger than the dense one. Moreover, accessing elements requires indirect, data-dependent lookups, which is inefficient on parallel hardware.

A compromise between dense and fully unstructured sparse formats is the Block-Sparse Row (BSR) format. BSR divides the matrix into equally sized dense blocks (e.g., 2x2). The format is similar to CSR, but the pointers and indices refer to these blocks, and the data array stores a list of dense blocks. This approach amortizes the metadata overhead over entire blocks, improving computational regularity.

1.3 The Economics of Sparsity: A Break-Even Model

To formalize the trade-offs, we can model the time taken for a dense vs. a sparse operation. The total time is determined by the maximum of either the compute time or the memory access time.

The execution time for a dense operation (T_dense) can be modeled as: T_{dense} \approx max\left( \frac{F}{P}, \frac{B}{BW} \right)

The execution time for a sparse operation (T_sparse) is more complex: T_{sparse} \approx max\left( \frac{(1-s)F}{P_{eff}}, \frac{B_{sparse}}{BW} \right) + T_{overhead}

Glossary of Terms: | Term | Description | | :--- | :--- | | s | Sparsity (fraction of zeros). | | F | Total FLOPs for the dense operation. | | P | Peak theoretical throughput of the hardware (FLOPs/sec). | | P_eff | Effective sparse throughput, which is often much lower than P. | | B | Total bytes moved for the dense operation. | | BW| Memory bandwidth (bytes/sec). | | B_sparse| Bytes moved for the sparse operation, including metadata (B_{sparse} = B_{activations} + B_{non-zero\_weights} + B_{metadata}). | | T_overhead | Time cost for packing, format conversion, and handling load imbalance. |

This model reveals critical insights into the HW-ML Interplay:

* Memory-Bound Regimes: If the operation is limited by memory bandwidth (B/BW is the dominant term), reducing FLOPs (F) through sparsity may have no effect on latency.
* Unstructured Sparsity Overheads: Unstructured sparsity often hurts performance by lowering P_eff (due to irregular computation) and adding significant metadata B_meta and overhead T_overhead. This shifts the break-even point—the sparsity level at which T_sparse becomes less than T_dense—to very high levels, often over 90%.
* Structured Sparsity Advantage: Structured approaches (like block sparsity or N:M patterns) reduce metadata and overhead, leading to a higher P_eff. This allows them to achieve a speedup over dense operations at much lower sparsity levels.

A diagram illustrating the "Speedup vs sparsity: break-even effect" shows this relationship clearly. A plot with Sparsity on the x-axis and Speedup on the y-axis depicts two curves. The curve for "Unstructured" sparsity remains flat at 1.0 (no speedup) until a high sparsity of ~85%, after which it begins to rise. In contrast, the curve for "Structured" sparsity begins to show a speedup at a much lower sparsity of ~50%, demonstrating its superior hardware efficiency.

1.4 Practical Pruning Methodologies in PyTorch

Unstructured Pruning

This technique removes individual weights based on their magnitude. PyTorch's torch.nn.utils.prune module provides a simple interface for this.

A visual representation of a weight matrix before and after 30% unstructured pruning shows a random, salt-and-pepper pattern of zeroed-out (black) elements in the pruned matrix.

# Unstructured pruning of a fully-connected layer
import torch.nn as nn
import torch.nn.utils.prune as prune

fc = nn.Linear(in_features=10, out_features=6)

# Prune 30% of the weights with the smallest L1 magnitude
prune.l1_unstructured(fc, name='weight', amount=0.3)


This applies a weight_mask to the layer. To make the change permanent (though still stored as a dense tensor with zeros), one would call prune.remove(fc, 'weight').

Structured Pruning

This method removes entire groups of weights. The importance of a group (e.g., a neuron or a filter) is typically determined by calculating a norm over its weights. The choice between L1 and L2 norm can influence which structures are pruned.

* L1 Norm: $∥w∥_1 = \sum_j |w_j|$. Each weight contributes linearly. This norm is less sensitive to a single large weight within a group.
* L2 Norm (Squared): $∥w∥_2 = \sum_j w_j^2$. Larger weights have a disproportionately greater impact on the norm. A single large weight can "save" its group from being pruned, even if other weights are small.

Consider two weight vectors, a = [3, 0, 0, 0] and b = [2, 1, 0, 0].

* Their L1 norms are equal: $∥a∥_1 = 3$ and $∥b∥_1 = 3$.
* Their squared L2 norms differ: $∥a∥_2^2 = 9$ and $∥b∥_2^2 = 5$. The L2 norm prioritizes vector a due to its single large-magnitude weight.

Visualizations of L1-pruned vs. L2-pruned weight matrices show distinct differences. With output neuron pruning, entire rows are zeroed out. The specific rows chosen can differ between L1 and L2-norm criteria, reflecting their different sensitivities to weight distributions.

# Structured pruning of an FC layer's output neurons (dim=0)
prune.ln_structured(fc, name='weight', amount=0.3, n=1, dim=0) # n=1 for L1-norm

# Structured pruning of an FC layer's input features (dim=1)
prune.ln_structured(fc, name='weight', amount=0.3, n=1, dim=1)

# Structured pruning of a Conv layer's output channels (dim=0)
conv = nn.Conv2d(16, 32, kernel_size=3)
prune.ln_structured(conv, name='weight', amount=0.3, n=1, dim=0)


Note: Pruning the last fully-connected (FC) layer's outputs or the first FC layer's inputs can be problematic as it changes the model's output or input dimensions, respectively.

Threshold-Based Pruning

Instead of removing a fixed fraction of weights, thresholding removes all weights whose magnitude falls below a certain value. This can be implemented in PyTorch by creating a custom mask.

# Custom pruning based on an absolute magnitude threshold
threshold = 0.15
mask = (fc.weight.data.abs() >= threshold) # Create a boolean mask

# Apply the custom mask to the layer
prune.custom_from_mask(fc, name='weight', mask=mask)


This approach can be extended to structured pruning by first aggregating weight magnitudes (e.g., L2 norm of each convolutional filter), applying a threshold to these aggregate scores, and then creating a mask that zeros out all weights belonging to the low-magnitude structures.

1.5 The Path to Real Speedup: Deployment Strategies

As established, simply using PyTorch's default pruning utilities does not yield a speedup. This is because:

1. Dense Kernels are Used: The backend still invokes a dense GEMM or conv kernel, as the tensor shapes are unchanged. The operation weight = weight_orig * weight_mask is an element-wise multiplication that occurs before the main computation.
2. No Sparse Representation: A mask is not a true sparse format like CSR or BSR. The hardware has no information to skip zero-valued elements.
3. prune.remove is Deceptive: This function finalizes the pruning by creating a new dense weight tensor that explicitly contains the zeros, which is then passed to the dense kernel.

To achieve actual performance gains, one must follow a hardware-aware deployment path.

Deployment Path	Description & How it Works	Best For	Key Trade-offs
Mask-Only	Uses weight_mask to zero out weights but runs on dense kernels. The operation is y = x ⋅ (W ⊙ M).	Research, ablations, sensitivity analysis, exploring pruning schedules.	No speedup. Must be explicitly reported as running on dense kernels.
Structured Rewiring	Physically removes entire channels, neurons, or heads. This reduces the tensor dimensions, creating a smaller, dense model.	Achieving portable latency/throughput gains on general-purpose hardware (CPUs, GPUs).	Requires changes to the network architecture. May need careful fine-tuning and plumbing of tensor shapes.
Hardware Pattern	Enforces a specific pattern (e.g., N:M), packs weights into a special format, and uses specialized sparse kernels.	Maximizing speed on accelerators with dedicated structured sparsity support.	Pattern constraints can impact accuracy. Incurs packing/format conversion overhead. Limited operator coverage.

Case Study: NVIDIA's 2:4 Structured Sparsity

A prime example of the "Hardware Pattern" path is NVIDIA's 2:4 structured sparsity, supported by Ampere and later GPU architectures.

* The Pattern: In every contiguous group of four weights, a maximum of two can be non-zero.
* The Mechanism: Models are trained or fine-tuned to adhere to this constraint. The sparse weights are then compressed, storing only the non-zero data values and their corresponding indices. The diagram shows this compression process, where a sparse weight matrix is converted into a compact representation before the dot product.
* The Hardware: NVIDIA's Sparse Tensor Cores (SPTCs) are designed to process this compressed data format at up to 2x the speed of dense Tensor Cores.
* The Format: The underlying storage format is a variant like Blocked-Ellpack (Blocked-ELL), which, similar to CSR/BSR, is optimized for regular memory access patterns on the GPU.

1.6 Advanced Techniques: Learning Sparsity with PSP

Traditional pruning methods are often a multi-stage process: train a dense model, apply a heuristic criterion to prune, and then fine-tune to recover accuracy. Parameterized Structured Pruning (PSP) offers a more integrated approach by learning the sparsity pattern during training.

The core idea of PSP is to parametrize the pruning decision for entire structures (e.g., weights, columns, channels). The different levels of structure are visualized in a diagram showing (a) individual weights, (b) columns, (c) channels, (d) shapes, and (e) layers.

For each structural sub-tensor $w_i$, a learnable parameter $\alpha_i$ is introduced. During the forward pass, the sub-tensor is dynamically masked: q_i = w_i \cdot v_i(\alpha_i) where $v_i(\alpha_i)$ is a thresholding function: v_i(\alpha_i) = \begin{cases} 0 & \text{if } |\alpha_i| < \epsilon \\ \alpha_i & \text{if } |\alpha_i| \geq \epsilon \end{cases}

Since this thresholding is non-differentiable, the Straight-Through Estimator (STE) is used to approximate its gradient during backpropagation. The gradients update the $\alpha_i$ parameters, allowing the network to learn which structures are important. A regularization term (L1 or L2) is applied to the $\alpha_i$ parameters to encourage sparsity.

Update rules for $\alpha_i$, using gradient descent with momentum $\mu$ and a learning rate $\eta$, can incorporate L2 weight decay ($-λη \cdot \alpha_i(t)) or L1 regularization ($-λη \cdot sign(\alpha_i(t))`). L2 regularization tends to produce better-separated weight distributions and has shown superior performance.

A validation error plot for ResNet-56 on CIFAR-10 with column pruning demonstrates PSP's effectiveness. The plot shows that "PSP (weight decay)" achieves a lower validation error than "L1 norm" pruning across a wide range of sparsity levels, indicating it finds a better trade-off between accuracy and compression.

1.7 Benchmarking and Measurement: A Practical Guide

Accurate and honest performance measurement is non-negotiable in embedded ML. As the German adage says, "Wer misst, misst Mist" (He who measures, measures rubbish). Rigorous benchmarking is essential to avoid misleading results.

What to Report (Minimum Set)

* Quality: Accuracy/loss, with the evaluation protocol specified.
* Latency: p50 (median) and p90 (90th percentile) latency in milliseconds for a fixed batch size and input shape.
* Throughput: Samples per second or tokens per second.
* Memory: Peak GPU memory usage.
* Energy (Optional): Joules per inference or per training step.

It is crucial to be explicit about the scope of the timing:

* Kernel-only: Timing just the matmul or conv operation.
* Layer: Timing the full layer forward pass.
* End-to-end: Timing the entire inference pipeline, including data transfers, pre/post-processing, and any format conversions (e.g., packing for sparse execution). Only an end-to-end measurement under a fixed, stated workload can be trusted.

Measurement Protocol (Rules of Thumb)

1. Fix the Environment: Use the same hardware, data type (dtype), batch size, and input shape for all comparisons.
2. Warmup: Run 20–100 iterations before starting measurements to let the GPU reach a stable state.
3. Measure Many Iterations: Average results over 200–1000 iterations for statistical stability.
4. Synchronize GPU: On GPUs, kernel launches are asynchronous. Wrap timers with a synchronization call (e.g., torch.cuda.synchronize()) to ensure the measurement captures the full execution time.
5. Report Variability: Report mean and standard deviation, or p50/p90 percentiles, not just a single number.

Common Measurement Traps

* No GPU Synchronization: This times only the kernel launch, not its execution, leading to artificially low latency numbers.
* Comparing Different Workloads: Comparing results with different batch sizes, input shapes, or data types is invalid.
* Ignoring Overheads: Timing only the matrix multiplication while ignoring the cost of packing data into a sparse format.
* Ignoring Memory-Bound Effects: Assuming fewer FLOPs automatically means less time, which is false in memory-bound scenarios.

1.8 Case Study: Pruning a Simple CNN on MNIST

To demonstrate these concepts, a simple CNN is trained on the MNIST dataset.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


With standard training (5 epochs, Adam optimizer, lr=0.001), the model achieves a baseline accuracy of 99.02%.

Scenario	Sparsity	Accuracy
Baseline	0%	99.02%
Pruning (No Fine-Tuning)	49.96%	90.14%
Pruning (1 Epoch Fine-Tuning)	49.96%	98.91%

Pruning nearly 50% of the weights without fine-tuning causes a significant accuracy drop of ~9%. However, just one epoch of fine-tuning (FT) restores accuracy to near-baseline levels. Histograms of weight distributions show that after pruning, many weights are clustered at zero. Fine-tuning allows the remaining weights to adjust and recover their original distribution shape, restoring the model's representational capacity.

A comprehensive plot comparing unstructured, L1, L2, and threshold pruning with varying epochs of fine-tuning across a spectrum of sparsities shows that fine-tuning is universally critical. All methods experience a sharp accuracy drop-off as sparsity increases, but fine-tuning consistently pushes this "cliff" to higher sparsity levels, enabling more aggressive compression without sacrificing performance.

1.9 Strategic Pruning: Layer-Wise Sparsity Allocation

Applying a uniform sparsity ratio across all layers is rarely optimal. Different layers have varying degrees of redundancy and sensitivity to pruning. A more effective strategy is to allocate sparsity non-uniformly based on layer characteristics.

1. Sensitivity-Based Allocation (Quality-Driven): The goal is to maximize accuracy for a given global sparsity budget. This involves measuring the sensitivity of each layer (e.g., by observing the accuracy drop when pruning it slightly). More sparsity is allocated to robust, redundant layers, while sensitive, brittle layers are pruned less aggressively.
2. Compute-Aware Allocation (Efficiency-Driven): The goal is to maximize latency or energy savings for a target accuracy. This strategy prioritizes pruning layers that are computationally expensive (high FLOPs or long execution time) and where pruning leads to tangible changes in tensor shapes or enables specialized kernels.

A practical rule of thumb is to prune large fully-connected layers heavily, as they are often over-parameterized, and to be more cautious with early convolutional layers, which tend to learn fundamental features.

1.10 Conclusion: A Systems-Level View of Compression

Pruning and other model compression techniques like quantization are not purely algorithmic problems; they are optimization problems under a strict systems constraint. The effectiveness of any compression method is determined by the interplay between the algorithm, the model architecture, and the target hardware.

The optimal strategy depends entirely on the end goal:

* If the goal is minimum model size for storage or transmission, unstructured pruning is sufficient, provided the model is stored and loaded in an efficient sparse format.
* If the goal is portable latency reduction on general-purpose hardware, structured pruning that physically rewires the network to create smaller, dense tensors is the most reliable path.
* If the goal is maximum performance on specialized hardware, one must enforce kernel-supported patterns like N:M sparsity and leverage dedicated hardware acceleration.

Ultimately, successful model compression for embedded systems requires a holistic view, recognizing that performance is a function of the entire tuple: {data, neural architecture, hardware architecture}.
