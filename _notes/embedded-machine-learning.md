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

## 1. Motivation

This course explores the intersection of state-of-the-art Deep Neural Networks (DNNs) and resource-constrained embedded devices. A central theme is the challenge of not only making complex models run on resource-constraint embedded devices but also embedding these models in the real world, which necessitates a robust understanding and treatment of uncertainty and resource-efficient deep neural networks.

## 2. Introduction to Machine Learning and GPU

### 2.1 The Landscape of Modern Machine Learning

#### The Challenge: Mismatch Between ANNs and Embedded Hardware

The scale of data used to train these models is massive and growing. The complexity ranges from smaller datasets that can be trained on a laptop to enormous collections that require supercomputing resources.

There is an extreme mismatch between the computational demands of modern ANNs and the capabilities of mobile or embedded processors. Large models like ResNet-50, which perform well on high-power servers, are difficult to deploy on devices with strict power and memory constraints.

### The Hardware Lottery Hypothesis

This mismatch leads to a concept known as the Hardware Lottery Hypothesis.

The "Hardware Lottery" suggests that tooling has played a disproportionately large role in deciding which ideas succeed and which fail. The hardware that is readily available and highly optimized (like GPUs for matrix operations) determines which research directions thrive.

Because ANNs fundamentally rely on matrix-matrix operations, they perform exceptionally well on GPUs, which are designed for exactly this kind of computation. As a result, most ML researchers tend to ignore hardware constraints and focus on architectures that fit this paradigm, such as Convolutions and Transformers. This has led to massive models like GPT-3 (175B parameters, 800GB of state) and AlphaFold-2 (23TB of training data).

This raises an important question: what if a different type of processor existed, e.g. one that excelled at processing large graphs? This could have led to the dominance of alternative models like probabilistic graphical models, sum-product networks, or graph neural networks. 

**Processor specialization is considered harmful for innovation because it can prevent alternative algorithmic approaches from being explored.**

### 2.2 Fundamentals of Supervised Learning: Regression

This section covers the foundational concepts of machine learning, including learning, generalization, model selection, regularization, and overfitting, using the example of regression.

#### Introduction to Supervised Learning

In supervised learning, the goal is to learn a predictive function from a labeled dataset. We are given a set of examples and asked to predict an output for new, unseen data.

Supervised Learning Problem: Given a training set of $N$ samples, $(x^{(i)}, t^{(i)})$, find a good prediction function, $y = h_{\theta}(x)$, that can generalize to new data.

##### Key Terminology

* $x^{(i)}$: The input features of the $i$-th training sample.
* $t^{(i)}$: The target variable (or label) of the $i$-th sample.
* $(x^{(i)}, t^{(i)})$: A single training sample or observation.
* Training Set: The complete set of $N$ training samples.
* $h_{\theta}(x)$: The prediction function (or hypothesis) we are trying to find (learn).
* $\theta$: The parameters (or weights) of the model that the learning algorithm will adjust.

Problems in supervised learning can be categorized as:

* **Classification**: When the target variable $t^{(i)}$ is discrete (e.g., 'cat', 'dog', 'bird').
* **Regression**: When the target variable $t^{(i)}$ is continuous (e.g., the price of a house).

### Linear Regression

The simplest form of regression is linear regression. Here, we assume the relationship between the input features and the output is linear. For an input $x$ with $D$ features, the model is:

$$ h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_Dx_D $$

This can be written more compactly using vector notation. By setting $x_0 = 1$, we can absorb the $\theta_0$ term (known as the model intercept or bias):

$$ h_{\theta}(x) = \sum_{d=1}^{D} \theta_d x_d = \theta^\top x $$

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

**Where:**

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

High-performance computing has undergone a **fundamental paradigm shift**.
The exponential gains from single-core performance have stalled due to **physical limits of semiconductor scaling**, forcing a move toward **massive parallelism**.

This chapter explains:

* Why traditional CPU scaling broke down
* Why **GPUs** became central to modern machine learning

---

### 1.1 The End of an Era: Dennard Scaling and Single-Core Limits

For decades, processor performance improved due to two principles:

* **Moore’s Law**
  → Transistor counts double roughly every two years
* **Dennard Scaling**
  → Smaller transistors consume proportionally less power

This allowed CPUs to:

* Increase clock frequency
* Add complex control logic
* Improve single-thread performance without overheating

#### Characteristics of Classical CPU Design

Traditional CPUs focused on **single-instruction stream performance**, using:

* **Deep, multiple pipelines**
  → Instructions split into many stages
* **Latency minimization and hiding**
  → Caches, prefetching
* **Aggressive speculation**
  → Branch prediction, speculative execution
  → Requires reorder buffers and rollback logic
* **Strong locality assumptions**
  → Predictable control flow, temporal & spatial locality

#### The Breakdown (~2005)

Dennard Scaling **failed** due to:

* Leakage currents
* Rising power density
* Thermal limits

Increasing clock frequency became infeasible.

##### Power Model


$$P = a f C V^2 + \frac{V I_{\text{leakage}}}{f^3}$$


Where:
* $f$: frequency
* $V$: voltage
* $C$: capacitance

**Key implication:**
Power grows too quickly with frequency → performance scaling must come from **parallelism**, not speed.

> **Turning point:**
> The industry shifted from **complex single cores** to **many simple cores**.

---

### 1.2 The Rise of the GPU for General-Purpose Computing

GPUs were originally designed for **graphics workloads**, which are:

* Highly parallel
* Floating-point intensive
* Data-synchronous

This made them a natural fit for non-graphics workloads.

#### GPGPU Breakthrough (≈ 2007)

* Introduction of **CUDA**
* Enabled **General-Purpose GPU Computing (GPGPU)**

#### CPU vs GPU Design Philosophy

| Aspect          | CPU                     | GPU                 |
| --------------- | ----------------------- | ------------------- |
| Primary Goal    | Minimize latency        | Maximize throughput |
| Core Design     | Complex, speculative    | Simple, replicated  |
| Parallelism     | Instruction-level (ILP) | Data + thread-level |
| Control Logic   | Heavy speculation       | Minimal control     |
| Energy Strategy | High frequency          | Massive replication |

> **Key takeaway:**
> GPUs trade latency for **throughput**, making them ideal for machine learning.

---

## Chapter 2 — Principles of GPU Architecture and Execution

GPUs expose a **distinct programming abstraction** that differs fundamentally from CPUs.

Understanding this abstraction is essential for efficient ML implementation.

---

### 2.1 Vector Processing and SIMD

At the hardware level, GPUs are **vector processors**.

They use **SIMD (Single Instruction, Multiple Data)** execution.

#### Advantages of SIMD

* **Instruction compactness**
  → One instruction applies to many data elements. A single instruction defines an operation to be performed on an entire vector of data. This amortizes the cost of instruction fetch, decode, and issue, and reduces the frequency of branches.
* **Implicit parallelism**
  → No dependency analysis required. The operations are inherently data-parallel, with no dependencies between them. This eliminates the need for complex hardware to detect parallelism, allowing for straightforward execution on parallel data paths.
* **Efficient memory access**
  → Wide loads, predictable patterns. Vector memory instructions can describe regular access patterns (e.g., continuous blocks of data), enabling hardware to prefetch data or use wide memory interfaces to hide the high latency of the first element access over a large sequence.

> SIMD amortizes instruction overhead across many operations.

---

### 2.2 SIMT vs SIMD: Two Views of One Machine

Although hardware executes SIMD, programmers see **SIMT**.

#### Software View — SIMT
* Programmer writes **scalar thread code**, as if a thread was an independent entity. This greatly simplifies programming, as one can reason about the logic for a single data element.
* Millions of such logical threads launched

#### Hardware View — SIMD
* Threads grouped into **warps** (typically 32 threads in NVIDIA terminology)
* The threads within a warp are executed in **lock-step** on the vector-like hardware.
* A **single instruction** is fetched and executed for the **entire warp**.

> **SIMT is an abstraction** that hides vector hardware for usability. The SIMT model is an illusion created by the hardware to hide the underlying vector units, making the massive parallelism of the GPU more accessible.

---

#### 2.3 The Programming Model: Bulk-Synchronous Parallelism

GPU execution aligns closely with the **BSP model**.

Each computation proceeds in **supersteps**:

1. **Compute** — local operations
2. **Communicate** — data exchange
3. **Synchronize** — barrier

#### Parallel Slackness

$$\text{Slackness} = \frac{v}{p}, \quad v \gg p$$

* $v$: virtual processors (threads)
* $p: physical processors

GPUs exploit slackness to **hide memory latency** by swapping in other ready warps while one warp is waiting for data.

---


### 2.4 Control Flow Divergence

Conditional branches inside a warp cause **divergence**.

#### Example

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/SIMT_execution_model.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>SIMT execution model</figcaption>
</figure>

1. A, B: All threads execute instructions A and B in lock-step.
2. C: Threads T1 and T2 execute instruction C, while T3 and T4 are disabled (masked off).
3. D: Threads T3 and T4 execute instruction D, while T1 and T2 are disabled.
4. E: All threads reconverge and execute instruction E in lock-step.

```text
foo[] = {4,8,12,16};
v = foo[tid.x];
if (v < 10)
    v = 0;
else
    v = 10;
```

Execution behavior:

* Each branch path is executed **serially**
* Non-participating threads are masked

> **Cost:**
> Divergence reduces effective parallelism.

#### Synchronization Scope
* **Within block:** fast barrier
* **Across blocks:** kernel relaunch (because interactions among CTAs are not allowed) (requires μs latency)
* **Atomic ops:** fine-grained coordination (tied to L2 latency)

---

## Chapter 3 — The Critical Role of the Memory Hierarchy

In GPUs, **data movement dominates cost**.

Performance is often **memory-bound**, not compute-bound.

---

### 3.1 Explicit Memory Hierarchy

Unlike CPUs, GPU memory is **explicitly managed**.

#### Memory Levels (Slow → Fast)

<div class="gd-grid">
  <figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/simp_core_cluster.png' | relative_url }}" alt="a" loading="lazy">
   <figcaption>SIMT Core Cluster</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/embedded-machine-learning/GPU_memory.png' | relative_url }}" alt="GPU shared memory" loading="lazy">
    <figcaption>Accessing Memeory</figcaption>
  </figure>
</div>

So in practice:
> **On NVIDIA GPUs, “SIMT core cluster” ≈ SM**
— but the terms are **not universally identical**.

* **Host Memory** — system RAM on the host machine (CPU side)
* **Global (GDDR)** — high bandwidth, high latency (GPU's main off-chip memory)
* **L2 Cache** — large shared cache that serves all cores on the GPU (e.g., 1.5MB).
* **Shared Memory / L1 Cache** — per Streaming Multiprocessor (SM), programmer-managed. SM, a cluster of cores, has its own on-chip scratchpad memory. This can be configured as a combination of L1 cache and programmer-managed Shared Memory. Shared memory allows for explicit, low-latency data sharing between threads within the same thread block (e.g., 16-48kB).
* **Read-only Data Cache** - optimized for constant data that is read by many threads (e.g., 48kB).
* **Registers** — private to each thread, fastest

> **Guarantee:**
> Memory coherence only at kernel boundaries.

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/GPU_arch_top_level.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>GPU Architecture Top-Level View</figcaption>
</figure>

---

### 3.2 Bandwidth and Capacity Across Generations

| Architecture       | Registers | Shared/L1 | LLC   | Global  |
| ------------------ | --------- | --------- | ----- | ------- |
| CPU (Sandy Bridge) | ~1kB      | 512kB     | 8MB   | 20GB/s  |
| Kepler GK110       | ~4MB      | ~1MB      | 1.5MB | 150GB/s |
| Pascal GP100       | 14MB      | ~4MB      | 4MB   | 800GB/s |
| Ampere GA100       | 32MB      | 24MB      | 40MB  | 1.9TB/s |

> GPUs prioritize **on-chip bandwidth** over cache sophistication.

---

### 3.3 Data Movement Dominates Energy

Energy cost comparison:

* 8-bit multiply: **~0.2 pJ**
* 32-bit float multiply: **~3.7 pJ**
* SRAM access: **~100 pJ**
* DRAM access: **~10,000 pJ**

> **Key insight:**
> Computation is cheap.
> **Moving data is expensive.**

#### Core Optimization Principles

1. **Vectorization (SIMD)**: amortizes the high cost of *instruction fetch* and control *across many data elements*.
2. **Reduced precision**: using lower-precision data types (e.g., 16-bit floats or 8-bit integers)
3. **Aggressive data reuse**: registers and shared memory

---

## Chapter 4 — Hardware–Software Co-Design for ML

Deep learning workloads align almost perfectly with GPU strengths.

---

### 4.1 Why Neural Networks Fit GPUs

Core operation:

$$\mathbf{Y} = \mathbf{W} \cdot \mathbf{X}$$

Properties:
* Massive **data parallelism**: The calculation of each element in the output matrix is an independent dot product. There are $N^2$ such independent computations.
* High **compute-to-memory ratio**: High compute-to-memory ratio means that once the weight and input matrices are loaded into fast memory, a massive amount of computation can be performed, maximizing data reuse. Compute scales with $O(N^3)$, memory with $O(N^2)$.
* Excellent **energy efficiency**: GPUs can perform so many operations per Watt, they are highly energy-efficient for these dense computational tasks.

This explains GPU dominance in:
* DNNs
* LLMs
* SGEMM workloads

The performance graphs for SGEMM (Single-precision General Matrix Multiply) show that performance increases steadily with matrix size, especially when using optimized libraries like cuBLAS that are tuned to the specific hardware.

---

### 4.2 Tensor Cores

Tensor Cores are **specialized MMA units**.

#### Key Features

<div class="gd-grid">
  <figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/nvidea_tensorcores1.png' | relative_url }}" alt="a" loading="lazy">
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/embedded-machine-learning/nvidea_tensorcores2.png' | relative_url }}" alt="GPU shared memory" loading="lazy">
  </figure>
</div>

* Tile-based matrix operations
* Mixed precision (FP16 → FP32)
* Massive throughput per cycle

Example operation:

$$D = A \cdot B + C$$

Accessed via:
* cuBLAS
* cuDNN
* WMMA API

The Tensor Core is a powerful example of the HW-ML Interplay, where a dominant software workload (deep learning) directly drove the creation of specialized hardware to accelerate it.

> Tensor Cores are a **direct result of ML-driven hardware design**.

### Chapter 5: Future Trends and Broader Implications

#### 5.2 The "Hardware Lottery": How Architecture Shapes Research

The immense success and widespread availability of GPUs for deep learning have created a phenomenon known as the "Hardware Lottery." This concept suggests that an algorithmic idea's success is not just a function of its inherent quality but also of how well it maps to available hardware.

- Bias Towards "Standard" DNNs: Because GPUs are exceptionally good at dense matrix multiplications, there is a strong incentive for researchers to develop models based on these operations.
- Stifling of Alternative Models: Novel ideas that might not fit the GPU's massively parallel, data-synchronous model may be underexplored, not because they are inherently inferior, but because they cannot be demonstrated at scale on current hardware.

This raises a critical question: Do GPUs create a bias towards specific types of ML solutions? The answer appears to be yes, highlighting a deep and complex interplay where hardware availability can guide the trajectory of an entire research field. The "billion-dollar question" remains: what promising alternatives are being overlooked due to this architectural dominance?

---

## Embedded Machine Learning: A Technical Reference

### Table of Contents

1. Chapter 1: Foundations of Neural Networks for Embedded Systems
   - 1.1 From Linear Regression to Universal Approximators
   - 1.2 The Multi-Layer Perceptron (MLP)
   - 1.3 Mathematical Formalism of a Neural Network
   - 1.4 Core Components: Activation Functions
   - 1.5 Training Neural Networks: Forward and Backward Propagation
   - 1.6 Understanding Convolutional Neural Networks (CNNs)
   - 1.7 Key CNN Architectures: AlexNet & VGG16
   - 1.8 The Simplicity Wall: A Core Challenge in Embedded ML


---


# Chapter 1 — Foundations of Neural Networks for Embedded Systems

This chapter bridges **fundamental ML models** (e.g., linear regression) and the **higher-capacity models** needed for embedded tasks. We study:

* **Artificial Neural Networks (ANNs):** architecture + mathematical model
* **Core building blocks:** MLPs, activations, convolution
* **Embedded perspective:** how computation patterns map to **resource constraints** (memory, bandwidth, power)

> **Central theme:** ANNs are structured the way they are because their computation must be feasible under hardware limits.

---

## 1.1 From Linear Regression to Universal Approximators

### Motivation

Linear regression is foundational, but **not a universal approximator**: it cannot represent arbitrarily complex functions.

This limitation is especially severe for embedded perception tasks (e.g., images), due to:

* **High dimensionality** (each pixel adds a dimension)
* The **curse of dimensionality** in modeling complex decision boundaries

### Why ANNs help

**ANNs** stack layers of interconnected “neurons,” enabling **non-linear function approximation**.
With more intermediate layers (**hidden layers**), we obtain **Deep Neural Networks (DNNs)** with greater capacity.

> **Highlight — Universal Approximation Theorem**
> A sufficiently large neural network (with suitable weights and non-linearities) can approximate a wide variety of functions.

### Biological analogy (important distinction)

ANNs are *loosely* inspired by biology, but “biologically inspired” is often viewed skeptically.

* **ANNs:** differentiable, trained via gradient-based optimization
* **Spiking neural networks:** closer to biology but often non-differentiable → trained differently

### Overfitting (conceptual note)

The referenced diagram illustrates **overfitting** and **regularization**:

* High-capacity model fits training points but fails to generalize
* Regularization penalizes complexity → improves generalization on unseen data

> **Highlight — Overfitting vs. Generalization**
> Strong fit to training data does not imply correct modeling of the underlying function.

---

## 1.2 The Multi-Layer Perceptron (MLP)

### Definition

An **MLP** is a feedforward ANN with:

* input layer
* ≥ 1 hidden layer
* output layer

Each neuron is connected to every neuron in the next layer (**fully connected**).

### Example: MNIST

For $28\times 28$ grayscale images:

* input dimension: $28 \cdot 28 = 784$
* output classes: $10$

### Neuron computation (scalar form)

$$y_k = f\Big(\sum_j (w_{k,j}, x_j) + b_k\Big)$$

**Where**

* $y_k$: output of neuron $k$
* $f$: activation function (e.g., Sigmoid, ReLU)
* $w_{k,j}$: weight from neuron $j$ to neuron $k$
* $x_j$: input activation
* $b_k$: bias

### Layer computation (vector form)

$$\mathbf{x}_\ell = f(\mathbf{W}_\ell \mathbf{x}*{\ell-1} + \mathbf{b}*\ell)$$

> **Highlight — Workload identity**
> The dominant compute is **matrix–vector multiplication** plus a non-linearity.

---

## 1.3 Mathematical Formalism of a Neural Network

Understanding the linear algebra is essential because it determines:

* compute intensity
* memory traffic
* suitability for acceleration on embedded hardware

### Notation conventions

* **Matrix:** $\mathbf{W}$, elements $w_{k,j}$ (row $k$, column $j$)
* **Vector:** $\mathbf{x}$, elements $x_i$
* Vectors are column vectors by default; row vector: $\mathbf{x}^\top$

### Full forward pass (L-layer MLP)

$$y(\mathbf{W}, \mathbf{x}_0) = \mathbf{x}_L = \mathbf{W}_L \oplus f(\mathbf{W}_{L-1} \oplus f(\cdots \oplus f(\mathbf{W}_1 \oplus \mathbf{x}_0)\cdots))$$

$\oplus$ denotes matrix multiplication + bias addition.

#### Bias folding (implementation detail)

Bias $\mathbf{b}$ can be incorporated into $\mathbf{W}$ by appending a constant activation (e.g., $x_0=1$).
This often simplifies notation and hardware implementations.

> **Highlight — HW–ML Interplay**
> The dominance of matrix operations in DNNs has driven specialized accelerators (GPUs/TPUs) using SIMD, systolic arrays, etc.
> The primary bottleneck often becomes **feeding the compute units efficiently** (memory system).

---

## 1.4 Core Components: Activation Functions

### Why activations are necessary

Without non-linearities, stacking linear layers collapses into a single linear map:

> **Key result:**
> A multi-layer network *without activations* is equivalent to **one linear layer**.

### Practical choice: ReLU

ReLU is widely used due to:

* computational simplicity
* better training stability (mitigates vanishing gradients for positive region)

### Common activation functions

| Function   | Formula                                  | Output Range       | Key Characteristics                               |
| ---------- | ---------------------------------------- | ------------------ | ------------------------------------------------- |
| Sigmoid    | ( \frac{1}{1 + e^{-x}} )                 | ([0,1])            | Smooth; can suffer vanishing gradients            |
| Tanh       | ( \frac{e^x-e^{-x}}{e^x+e^{-x}} )        | ([-1,1])           | Zero-centered; may still saturate                 |
| ReLU       | ( \max(x,0) )                            | ([0,\infty))       | Very efficient; risk of “dead neurons”            |
| Leaky ReLU | ( x ) if (x\ge0), ( \alpha x ) otherwise | ((-\infty,\infty)) | Prevents dead neurons via non-zero negative slope |
| ELU        | ( x ) if (x\ge0), ( e^x-1 ) otherwise    | ((-1,\infty))      | Smooth negative region; aims for stable gradients |

> **Observation**: Any non-linear function can be used.

<figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/activation_functions.png' | relative_url }}" alt="a" loading="lazy">
</figure>

---

### Softmax for classification

$$\mathrm{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

Properties:

* outputs sum to 1
* often interpreted as a probability distribution (but not necessarily a calibrated probability)

#### Numerical stability trick

Subtract the maximum input before exponentiation:

$$\frac{e^{x_i}}{\sum e^{x_j}} = \frac{e^{x_i - \max_j x_j}}{\sum e^{x_j - \max_j x_j}}$$

> **Highlight — Stability rule**
> Always stabilize softmax via subtracting $\max(\mathbf{x})$ to prevent overflow.

---

## 1.5 Training Neural Networks: Forward and Backward Propagation

<figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/NN_training_parallelism.png' | relative_url }}" alt="GPU shared memory" loading="lazy">
</figure>

Training adjusts weights $\mathbf{W}$ to minimize a loss function (\mathcal{L}).

### Dataset loss with regularization

$$\mathcal{L}(\mathbf{W}; \mathcal{D}) = \sum_{n=1}^{N} \ell(y(\mathbf{W}, \mathbf{x}_n), \mathbf{t}_n) + \lambda r(\mathbf{W})$$

Components:

1. **Data term** $\ell(\cdot)$: penalizes prediction error
2. **Regularizer** $r(\mathbf{W})$: penalizes complexity (e.g., $\ell_1$, $\ell_2$)

   * $\lambda$: controls the strength of regularization

### Gradient descent update

$$\mathbf{W} := \mathbf{W} - \eta \nabla_{\mathbf{W}} \mathcal{L}$$

* $\eta$: learning rate

---

### Backpropagation (core algorithm)

Backprop efficiently computes $\nabla_{\mathbf{W}} \mathcal{L}$ via:

1. **Forward pass:** compute output and loss
2. **Backward pass:** propagate gradients using the **chain rule**

Chain rule example for $z=f(x,y)$:

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial z}\cdot \frac{\partial z}{\partial x}, \qquad \frac{\partial \mathcal{L}}{\partial y}=\frac{\partial \mathcal{L}}{\partial z}\cdot \frac{\partial z}{\partial y}
$$

Interpretation:

* upstream gradient: $\frac{\partial \mathcal{L}}{\partial z}$
* local gradients: $\frac{\partial z}{\partial x}, \frac{\partial z}{\partial y}$

> **Highlight — Practical insight**
> Modern ANN loss surfaces are highly non-convex; architectural choices (e.g., ResNet skip connections) can make optimization more stable by improving gradient flow.

---

## 1.6 Understanding Convolutional Neural Networks (CNNs)

MLPs are inefficient for images because they:

* ignore spatial structure
* require enormous parameter counts

CNNs address this by enforcing two key design constraints.

### Core properties

1. **Local connectivity (receptive fields)**
   Neurons connect only to local patches → captures spatial correlation.

2. **Weight sharing**
   Same filter/kernel applied across locations → fewer parameters and translation-tolerant feature detection.

> **Highlight — Why CNNs matter on embedded devices**
> Locality + sharing drastically reduce model size and enable regular compute patterns that accelerators exploit.

### Tensor shapes

An image is a 3D tensor:

$$(\text{Channels}, \text{Width}, \text{Height})$$

A convolution layer applies $M$ filters → produces $M$ output feature maps.

### Convolution definition

$$
O[z][u][x][y] =\sum_{k=0}^{C-1}\sum_{i=0}^{S-1}\sum_{j=0}^{R-1} I[z][k][Ux+i][Uy+j]\cdot W[u][k][i][j] * B[u]
$$

**Where**
* $z$: batch index
* $u$: output filter index
* $k$: input channel
* $R,S$: filter height/width
* $U$: stride

> **Highlight — Resource constraints**
> Convolutions enable **data reuse** and map well to hardware due to regular access and repeated MAC patterns.

<figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/other_convolutions.png' | relative_url }}" alt="GPU shared memory" loading="lazy">
   <figcaption>Other Convolutions</figcaption>
</figure>

---

## 1.7 Key CNN Architectures: AlexNet & VGG16

### AlexNet

A pioneering deep CNN for ImageNet:

* input: 227×227×3
* 5 convolution layers (Conv1–Conv5)
* 3 fully-connected layers (FC6–FC8)

Trend through the network:

* spatial resolution decreases
* channel depth increases
* features become more abstract

### VGG16

Known for architectural simplicity:

* stacks many 3×3 convolutions
* follows the same pattern: conv feature extraction → FC classification

> **Highlight — Common CNN pattern**
> Convolutional layers perform feature extraction; fully connected layers perform final classification.

---

## 1.8 The “Simplicity Wall” in Embedded ML

Despite complex architectures, most runtime is spent on one primitive:

> **Key claim:**
> ANNs spend most compute on **matrix multiplication / MAC operations**.

This holds for:

* fully-connected layers (matrix–vector products)
* convolution layers (often transformed into matrix–matrix products via im2col/Toeplitz-like constructions)

### Implications for embedded systems

* **HW–ML Interplay:** efficiency depends on hardware MAC throughput and memory feeding
* **Resource constraints:** deployment depends on reducing compute + memory

Typical techniques:

* **Quantization:** lower precision → less memory + faster MAC
* **Pruning:** remove redundant weights → fewer MACs + smaller models

> **Highlight — Embedded conclusion**
> Embedded ML success depends on co-design: algorithms must match the hardware’s ability to execute large numbers of MACs efficiently.

---

Below is a **formally reformatted, study-ready version** with improved hierarchy and **highlighting via callouts, bold emphasis, and “Key idea / Formula / Pitfall” blocks**. I did not change the meaning; only presentation and structure.

---

# Chapter 1 — The Foundations of Neural Network Training

Training an **Artificial Neural Network (ANN)** is an **optimization problem**: adjust parameters to minimize the discrepancy between predictions and targets. This chapter revisits the mathematical core of learning:

* **Loss functions** (objective definition)
* **Backpropagation** (efficient gradient computation)

---

## 1.1 The Goal: Minimizing the Loss Function

### Dataset and parameters

Let the dataset be

$$\mathcal{D} = {(x_1, t_1), \dots, (x_N, t_N)},$$

with $N$ input–target pairs. The network parameters (weights) are denoted $W$ and are typically initialized randomly.

### Loss function (objective)

A common formulation is:

$$\mathcal{L}(W; \mathcal{D}) = \sum_{n=1}^{N} \ell(y(W, x_n), t_n) + \lambda r(W)$$

#### Components

1. **Data term**
   
   $$\ell(y(W, x_n), t_n)$$
   
   Penalizes incorrect predictions. The sum aggregates error over the dataset.

2. **Regularizer**
   
   $$r(W)$$
   
   Penalizes large weights (e.g., $\ell_1$, $\ell_2$) to reduce overfitting.
   $\lambda$ controls the trade-off between **data fit** and **model simplicity**.

> **Highlight — Training objective**
> Find parameters $W$ that minimize $\mathcal{L}(W; \mathcal{D})$.

---

## 1.2 Backpropagation: The Engine of Learning

Backpropagation computes the **gradient of the loss** w.r.t. all parameters efficiently.

### Gradient-based update rule

Training proceeds by iterative updates:

$$W := W - \eta \nabla_W \mathcal{L}(W; \mathcal{D})$$

**Where**

* $W$: weights
* $\eta$: learning rate
* $\nabla_W \mathcal{L}$: gradient of loss w.r.t. $W$
* Gradient operator:
  
  $$\nabla_x = \Big(\frac{\partial}{\partial x_1}, \dots, \frac{\partial}{\partial x_n}\Big)$$

> **Highlight — Direction of change**
> Gradients point uphill (steepest ascent). Minimization updates go **against** the gradient.

### Neural network as a nested function

For an $L$-layer network:

$$y(W, x_0) = x_L =f\Big(W_L \oplus f\big(W_{L-1} \oplus ( \dots \oplus f(W_1 \oplus x_0)\dots )\big)\Big)$$

$\oplus$ denotes the linear transform (e.g., matrix multiplication + bias add).

### Chain rule structure of backprop

To compute gradients back to the input:

$$
\frac{\partial \ell}{\partial x_0} =\frac{\partial \ell}{\partial x_L}
\cdot
\frac{\partial x_L}{\partial x_{L-1}}
\cdot
\dots
\cdot
\frac{\partial x_1}{\partial x_0}
$$

> **Key question (efficiency)**
> How do we compute gradients for **all** parameters $(W, b)$ in a computationally efficient way?
> This motivates **automatic differentiation**.

---

# Chapter 2 — Automatic Differentiation

Manual differentiation for deep networks is:

* error-prone
* inefficient
* full of redundant computation

Modern frameworks rely on **automatic differentiation (autograd)**.

---

## 2.1 The Need for an Automated Approach

### Example model (scalar)

* $z = wx + b$
* $y = \sigma(z)$
* Regularized loss:
  
  $$\mathcal{L}_{reg} = \frac{1}{2}(y-t)^2 + \frac{\lambda}{2}w^2=\frac{1}{2}\big(\sigma(wx+b)-t\big)^2 + \frac{\lambda}{2}w^2$$

### Manual gradients

$$\frac{\partial \mathcal{L}_{reg}}{\partial w} = (\sigma(wx+b)-t),\sigma'(wx+b),x + \lambda w$$

$$\frac{\partial \mathcal{L}_{reg}}{\partial b}=(\sigma(wx+b)-t),\sigma'(wx+b)$$

> **Highlight — Redundancy**
> The term $(\sigma(wx+b)-t)\sigma'(wx+b)$ appears in multiple derivatives.
> In deep networks, such redundancies multiply dramatically.

Autograd avoids recomputation by:

* decomposing computation into primitives
* caching intermediates
* reusing them during the backward pass

---

## 2.2 The Computational Graph

Autograd represents computation as a **directed acyclic graph (DAG)**:

* **Nodes:** variables or results of operations
* **Edges:** dependencies (inputs → outputs)

The graph is built in **topological order** (parents before children).
Backprop traverses the graph **backwards**, applying the chain rule at each node.

> **Highlight — Operational view of backprop**
> Backprop is reverse traversal of the computational graph, accumulating gradients.

---

## 2.3 Chain Rule as a Computation ($v$-bar notation)

Define:

$$\bar{v} = \frac{\partial \mathcal{L}}{\partial v}$$

This emphasizes that $\bar{v}$ is a **computed, stored value**.

### Single-parent case

If $u = u(v)$:
[
\bar{v}
=======

# \frac{\partial \mathcal{L}}{\partial v}

# \frac{\partial \mathcal{L}}{\partial u}\frac{\partial u}{\partial v}

\bar{u}\frac{\partial u}{\partial v}
]

### Multi-parent case

For $f(a(v), b(v))$:

$$\bar{v} = \bar{a}\frac{\partial a}{\partial v} + \bar{b}\frac{\partial b}{\partial v}$$

> **Highlight — Efficiency principle**
> Compute $\bar{a}, \bar{b}$ once and reuse them across all dependent gradients.

---

## 2.4 Autograd in Practice: PyTorch

PyTorch builds computational graphs dynamically by tracking tensor operations.

### Core mechanism: `requires_grad`

Setting `requires_grad=True` tells PyTorch to track operations.

```python
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

q = 3*a**3 - b**2
```

Calling `.backward()` computes gradients:

```python
external_grad = torch.tensor([1., 1.])
q.backward(gradient=external_grad)

print(a.grad)  # ∂q/∂a = 9a^2
print(b.grad)  # ∂q/∂b = -2b
```

> **Highlight — Where gradients appear**
> Gradients of leaf tensors are stored in `.grad`.

### Full training loop (typical structure)

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data)
loss = (prediction - labels).sum()

loss.backward()
optim.step()
```

> **Highlight — Separation of concerns**
> Autograd computes gradients; the optimizer updates parameters.

---

## 2.5 The Autograd Algorithm (graph perspective)

Let $v_1, \dots, v_N$ be vertices in topological order.

* $Pa(v_i)$: parents of $v_i$
* $Ch(v_i)$: children of $v_i$

### Algorithm

1. **Forward pass:** compute values $v_i$ from $Pa(v_i)$
2. **Initialize backward:** $\bar{v}_N = 1$
3. **Backward pass:** for $i = N-1, \dots, 1$
   
   $$\bar{v}_i = \sum_{j \in Ch(v_i)} \bar{v}_j \frac{dv_j}{dv_i}$$

> **Highlight — Guarantee**
> Each local derivative is computed once, and intermediate gradients are reused.

---

# Chapter 3 — Advanced Calculus for Neural Networks

Deep learning relies on vector/matrix operations, requiring **vector calculus**.

---

## 3.1 From Scalar to Vector Calculus (Jacobian)

For $y=f(x)$ with $y\in\mathbb{R}^m$, $x\in\mathbb{R}^n$, the Jacobian is:

$$
J
=
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \
\vdots & \ddots & \vdots \
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{pmatrix}
$$

Backprop becomes:

$$\bar{x} = \Big(\frac{\partial y}{\partial x}\Big)^\top \bar{y} = J^\top \bar{y}$$

### Common vectorized derivatives

**1) Matrix-vector product (y = Wx)**

* $\frac{\partial y}{\partial x} = W$ → $\bar{x} = W^\top\bar{y}$
* $\bar{W} = \bar{y}x^\top$ (outer product)

**2) Element-wise operation (y=e^x)**

* Jacobian diagonal → $\bar{x} = e^x \circ \bar{y}$ (Hadamard product)

---

## 3.2 Backpropagation: Vectorized View (Two-layer example)

| Scalar (single element)                      | Vectorized (whole layer)                                        |
| -------------------------------------------- | --------------------------------------------------------------- |
| $\bar{y}_k = y_k - t_k$                      | $\mathbf{\bar{y}} = \mathbf{y} - \mathbf{t}$                    |
| $\bar{W}^{(2)}_{k,j} = \bar{y}_k h_j$        | $\mathbf{\bar{W}}^{(2)} = \mathbf{\bar{y}}\mathbf{h}^\top$         |
| $\bar{b}^{(2)}_k = \bar{y}_k$                | $\mathbf{\bar{b}}^{(2)} = \mathbf{\bar{y}}$                     |
| $\bar{h}_k = \sum_j \bar{y}_j W^{(2)}_{j,k}$ | $\mathbf{\bar{h}} = (\mathbf{W}^{(2)})^\top \mathbf{\bar{y}}$      |
| $\bar{z}_k = \bar{h}_k \sigma'(z_k)$         | $\mathbf{\bar{z}} = \mathbf{\bar{h}} \circ \sigma'(\mathbf{z})$ |
| $\bar{W}^{(1)}_{k,j} = \bar{z}_k x_j$        | $\mathbf{\bar{W}}^{(1)} = \mathbf{\bar{z}}\mathbf{x}^\top$         |
| $\bar{b}^{(1)}_k = \bar{z}_k$                | $\mathbf{\bar{b}}^{(1)} = \mathbf{\bar{z}}$                     |

> **Highlight — HW–ML Interplay**
> Vectorization is not just notation: it enables efficient SIMD execution on CPUs/GPUs and is essential for embedded performance.

---

# Chapter 4 — Gradient-Based Optimization Algorithms

Once gradients are computed, an optimizer updates parameters.

---

## 4.1 Gradient Descent (core rule)

$$W := W - \eta\nabla_W\mathcal{L}(W; \mathcal{D})$$

---

## 4.2 Variants of Gradient Descent

| Variant        | Gradient source | Update frequency   | Pros                            | Cons                           |
| -------------- | --------------- | ------------------ | ------------------------------- | ------------------------------ |
| Batch GD       | Full dataset    | Few updates/epoch  | Stable; converges for convex    | Slow; expensive                |
| SGD            | Single sample   | Many updates/epoch | Cheap; can escape shallow traps | Noisy; sensitive to (\eta)     |
| Mini-batch SGD | Small batch     | Balanced           | Stable + efficient              | Batch size is a hyperparameter |

> **Highlight — Hardware note**
> Batch sizes are often powers of two (32/64/128) to match GPU memory/parallelism.

---

## 4.3 Challenges in Optimization

1. **Choosing $\eta$**: one learning rate for all parameters is rarely ideal
2. **Non-convex landscapes**: local minima and saddle points complicate training

---

## 4.4 Momentum

Momentum maintains a velocity vector:

1. $v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$
2. $\theta := \theta - v_t$

* $\gamma \approx 0.9$

> **Highlight — Intuition**
> Momentum accumulates consistent gradient directions and reduces oscillation.

---

## 4.5 Adaptive Learning Rate Methods

### Adagrad

$$\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}}, g_{t,i}$$

Form the sum of the squares of the gradients for $\theta_i$ up to time step $t$ into a diagonal matrix (accumulator)

$$G_t = \sum_{\tau=1}^t g_{\tau}g_{\tau}^\top$$

Issue: $G_t$ grows → learning rate shrinks → training may stall.

### RMSprop / Adadelta

Use exponential moving average:

$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma)g_t^2$$

RMSprop update:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

### Adam

Maintains:

* first moment $m_t$ (mean)
* second moment $v_t$ (variance)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

Bias towards zero correction:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\beta_1 = 0.9 \qquad \beta_2 = 0.999$$

Update:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$

> **Highlight — Embedded constraint**
> Adam stores extra state per parameter $(m_t, v_t)$. On memory-constrained systems, SGD/Momentum may be preferable.

---

## 4.6 Visualizing Optimizer Performance (interpretation)

* **Ravine contours:** SGD progresses slowly; Momentum accelerates but oscillates; adaptive methods converge more directly.
* **Saddle points:** adaptive methods often navigate more effectively.

> **Highlight — Practical takeaway**
> SGD is robust; Adam typically converges faster and more reliably in practice.

---

# Chapter 5 — Conclusion and Key Takeaways

1. **Backpropagation + SGD are foundational**
   Requires differentiable components; remains the core of modern training.

2. **Automatic differentiation is essential**
   Computational graphs + chain rule avoid redundant work and scale to large models.

3. **Vectorization is central to hardware efficiency**
   Enables SIMD and fast matrix kernels; crucial for embedded ML.

4. **Advanced optimizers improve convergence**
   Adam is a strong default but consider memory overhead on embedded devices.

> **Bridge to embedded ML:**
> Effective training and deployment require understanding how optimization and differentiation interact with **hardware constraints** (compute, memory, bandwidth, power).

Below is a **formal, study-ready reformatted version** with clearer hierarchy and **highlighting via callouts, bold emphasis, and structured “Key idea / Formula / Embedded ML pillar” blocks**. Content is preserved; only layout and emphasis are improved.

---

## Chapter 1 — The Core Problem: Overfitting and the Need for Regularization

Deploying ML on embedded systems requires models that are both:

* **compact** (memory/compute constraints), and
* **robust** (noisy real-world sensors).

**Overfitting** directly conflicts with both goals: it creates models that are **unnecessarily complex** and that **generalize poorly**.

> **Key idea**
> Regularization is the main toolbox for reducing overfitting and improving generalization—often while also improving deployability on constrained hardware.

---

### 1.1 Defining Regularization

**Definition (formal):**
Regularization is **any modification** to a learning algorithm that **reduces generalization error** (test error) **without necessarily reducing training error**.

* **Generalization error:** performance on *unseen* data (the relevant benchmark)

Regularization typically works by penalizing complexity, steering learning toward **simpler hypotheses** that are more likely to reflect true structure rather than noise.

> **Embedded ML Pillar — Resource Constraints & Real-World Data**
> Regularization enables embedded ML by:
>
> * reducing parameter magnitudes and/or counts → smaller memory footprint
> * improving robustness to noisy sensor data → more stable real-world behavior

---

### 1.2 Visualizing Overfitting

Consider polynomial regression on noisy samples from a sine wave.

#### Model complexity examples

* **$M = 0$ or $M = 1$**: too simple → high training and test error (**underfitting**)
* **$M = 3$**: captures the underlying trend → good generalization (**well-fit**)
* **$M = 9$**: fits training points almost perfectly but oscillates wildly → poor test performance (**overfitting**)

#### Training vs test error behavior

* Training error typically **decreases monotonically** with complexity.
* Test error often follows a **U-shaped curve**:

  * decreases initially (less underfitting)
  * then increases (overfitting)

> **Key idea**
> Regularization aims to position the model near the **minimum of test error**, not the minimum of training error.

---

### 1.3 Philosophical Underpinnings: No Free Lunch and Occam’s Razor

1. **No Free Lunch Theorem**
   No algorithm is universally best across all tasks → technique choice depends on the problem.
   The same applies to regularization: which method works best depends on **data + architecture + constraints**.

2. **Occam’s Razor**
   Prefer the simplest explanation that fits the data (always choose the simplest of multiple
competing hypotheses, given they perform equally well).
   Regularization is the practical mechanism that enforces this preference by penalizing complexity.

---

## Chapter 2 — The Statistical Foundation: The Bias–Variance Tradeoff

Regularization can be understood as a method for controlling the **bias–variance tradeoff**, which explains why test performance improves when complexity is constrained.

---

### 2.1 Probability Review: Expectation and Variance

Let $X$ be a random variable.

#### Expectation

* Discrete:
  
  $$\mathbb{E}[X] = \sum_x x,p_X(x)$$
  
* Continuous:
  
  $$\mathbb{E}[X] = \int_{-\infty}^{\infty} x, f_X(x),dx$$
  
#### Variance

$$\mathrm{Var}(X)=\mathbb{E}\big[(X-\mathbb{E}[X])^2\big] =\mathbb{E}[X^2]-\mathbb{E}[X]^2$$

Standard deviation:

$$\mathrm{std}(X)=\sqrt{\mathrm{Var}(X)}$$

---

### 2.2 Decomposing Model Error

For a model estimator $\hat{y}_m$:

* **Bias**
  
  $$\mathrm{Bias}(\hat{y}_m)=\mathbb{E}[\hat{y}_m]-y$$
  
  Systematic error; high bias → underfitting.

* **Variance**
  
  $$\mathrm{Var}(\hat{y}_m)$$
  
  Sensitivity to the training set; high variance → overfitting.

* **MSE decomposition**
  
  $$\mathrm{MSE} = \mathbb{E}\big[(\hat{y}_m-y)^2\big] \mathrm{Bias}(\hat{y}_m)^2 + \mathrm{Var}(\hat{y}_m) + \text{Bayes Error}$$

> **Bayes Error**
> Irreducible error due to inherent noise in the data.

<div class="gd-grid">
  <figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/bias_tradeoff1.png' | relative_url }}" alt="a" loading="lazy">
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/embedded-machine-learning/bias_tradeoff2.png' | relative_url }}" alt="a" loading="lazy">
  </figure>
</div>

---

### 2.3 The Tradeoff in Practice

* **Underfitting:** high bias, low variance
* **Overfitting:** low bias, high variance

As model complexity increases:

* bias tends to decrease
* variance tends to increase

Regularization typically:

* **increases bias slightly**
* **reduces variance substantially**
  → lower total MSE.

> **Embedded ML Pillar — HW–ML Interplay**
> Overfit models often contain large, sensitive weights requiring higher precision.
> Regularization promotes smaller, more stable weights → better suited for **quantization** (e.g., int8), reducing memory and improving efficiency.

---

## Chapter 3 — Foundational Regularization Strategies

Regularization can be achieved through:

* training control (procedural)
* architecture constraints (structural)

---

### 3.1 Limiting Training Time: Early Stopping

**Idea:** stop training when validation error stops improving.

* During early epochs, weights are typically smaller.
* Continued training often increases weight magnitudes to fit noise.

> **Effect (intuition)**
> Early stopping behaves similarly to weight decay: it prevents overly large weights from forming.

**Practical issue:** validation curves can fluctuate under SGD → requires patience rules / smoothing.

---

### 3.2 Limiting Model Size: Parameter Reduction and Bottlenecks

Model capacity is strongly linked to **parameter count**.

#### Bottleneck architecture example

Compare:

* direct: 100 → 100
* bottleneck: 100 → 10 → 100

Parameter counts:

* direct: (100 \cdot 100 = 10{,}000)
* bottleneck: (100 \cdot 10 + 10 \cdot 100 = 2{,}000)

> **Key idea**
> Bottlenecks force the network to learn a compressed representation, acting as a structural regularizer.

**Tradeoff:** too much reduction can remove necessary capacity.

> **Embedded ML Pillar — Resource Constraints**
> Parameter reduction is one of the most direct routes to embedded deployment:
> fewer parameters → smaller model + fewer MACs → lower latency and energy.

---

## Chapter 4 — Data-Centric Regularization: Augmentation

When real data is limited, augment it synthetically.

---

### 4.1 The Principle of Data Augmentation

Create additional training samples via transformations that preserve semantics.

Common for images:

* geometric: flip/rotate/crop/translate/shear
* photometric: brightness/contrast/saturation
* noise injection

> **Rule (strict)**
> Apply augmentation **only** to the training set.
> The test set must remain an unbiased evaluation benchmark.

---

### 4.2 Advanced Augmentation: Mixup, Cutout, CutMix

These methods force robustness under partial information or mixed evidence.

| Method | Description                               | Label strategy                        |
| ------ | ----------------------------------------- | ------------------------------------- |
| Cutout | remove a random square region             | keep original label                   |
| Mixup  | interpolate two images linearly           | interpolate labels with same weights  |
| CutMix | paste a patch from one image onto another | mix labels proportional to patch area |

---

### 4.3 Automated Augmentation Policies

**AutoAugment**

* uses a controller (RNN) to search for augmentation policies via RL
* can outperform hand-designed pipelines
* but is expensive (e.g., ~1000 GPU hours cited for SVHN in the notes)

**TrivialAugment**

* simpler policy selection
* often comparable performance with far less search overhead

> **Rule of thumb**
> Consider augmentation when training loss ≪ test loss (clear overfitting signal).

---

### 4.4 Implementation in PyTorch

```python
# Example training augmentation pipeline
transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(3.8),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomAffine(translate=(0.8, 0.9), shear=[0.2, 0.4, 0.7]),
    transforms.ToTensor(),
])

# Test-time: basic preprocessing only
transforms_test = transforms.Compose([
    transforms.ToTensor(),
])
```

**Notes**

* `transforms.Compose`: chains transforms for on-the-fly augmentation.
* Task dependence matters:

  * horizontal flips may help generic object recognition
  * can harm digit recognition (e.g., 6 vs 9)

> **Embedded ML Pillar — Real-World Data**
> Augmentation simulates real deployment variation: viewpoint, lighting, occlusion—improving robustness “in the wild.”

---

## Chapter 5 — Parameter-Based Regularization: L1 & L2 Norms

Add penalties on parameter magnitude to control complexity.

---

### 5.1 Penalizing Complexity: Regularized Cost Function

$$\mathcal{J}(w) = \frac{1}{N}\sum_{n=1}^{N}\mathcal{L}(y(w,x_n),t_n) + \lambda \mathcal{R}(w)$$$$

* $\lambda$: regularization strength
* $\mathcal{R}(w)$: penalty term

> **Key idea**
> Larger $\lambda$ pushes toward simpler (more constrained) solutions.

---

### 5.2 L2 Regularization (Weight Decay / Ridge)

Penalty:

$$\mathcal{R}(w)=\frac{1}{2}\lvert w\rvert_2^2=\frac{1}{2}w^Tw=\frac{1}{2}\sum_j w_j^2$$

SGD update:

$$w_j := (1-\eta\lambda)w_j - \eta \frac{\partial \mathcal{L}}{\partial w_j}$$


> **Highlight — Weight decay mechanism**
> Each step shrinks weights by $(1-\eta\lambda)$, then applies the standard gradient update.

**Intuition:** large weights amplify sensitivity to input → encourages overfitting. L2 promotes smoother, more stable models.

---

### 5.3 L1 Regularization (LASSO) and Sparsity

Penalty:

$$\mathcal{R}(w)=\lvert w\rvert_1=\sum_j \lvert w_j\rvert$$

Key property: L1 can drive weights to **exactly zero** → **sparsity**.

Why:

* L2 gradient scales with $w$ → pull weakens near zero
* L1 gradient is constant sign (for nonzero $w$) → persistent pull to zero

> **Embedded ML Pillar — HW–ML Interplay & Resource Constraints**
> Sparsity enables:
>
> 1. compressed storage (sparse formats)
> 2. skipping MACs → lower latency and energy
>    Hardware increasingly supports sparse compute.

---

### 5.4 Geometric Interpretation of L1 vs L2

View optimization as minimizing loss contours under a norm constraint.

* **L2 ball:** circle in 2D → solution typically has both coordinates nonzero
* **L1 ball:** diamond in 2D → contours often touch at corners → one coordinate becomes exactly zero

> **Key insight**
> The geometry explains why L1 encourages sparsity and L2 encourages small-but-nonzero weights.

<div class="gd-grid">
  <figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/norms.png' | relative_url }}" alt="a" loading="lazy">
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/embedded-machine-learning/ridge_lasso.png' | relative_url }}" alt="a" loading="lazy">
  </figure>
</div>

---

## Chapter 6 — Stochastic Regularization Methods

Randomness during training can improve generalization by preventing memorization.

---

### 6.1 The Power of Ensembles

An ensemble averages predictions from multiple models (e.g., bagging).
This reduces variance but is computationally expensive.

> **Embedded note**
> Ensembles are often impractical for embedded inference due to memory and compute costs.

---

### 6.2 Dropout: Efficient Ensemble Approximation

During training:

* each neuron is dropped with probability $p$
* each batch sees a different “thinned” network

Effects:

1. approximates training many sub-networks sharing weights
2. prevents co-adaptation
3. implicitly encourages smaller outgoing weights (L2-like effect)

**Inference-time adjustment:** dropout is disabled; weights (or activations) are scaled by $(1-p)$.

> **Highlight — Train vs test behavior**
> Dropout is a training-only perturbation; inference uses the full network with scaling.

---

### 6.3 DropBlock: Dropout for Convolutional Layers

In CNNs, individual activation dropout is less effective due to spatial correlation.
**DropBlock** drops contiguous blocks → forces robustness to missing spatial regions.

---

## Chapter 7 — Summary and Implications for Embedded Systems

**Regularization** reduces generalization error and is central to building robust models.

### Key themes

* Many techniques overlap conceptually:

  * early stopping ↔ weight decay ↔ bottlenecks
  * dropout ↔ ensembles
  * augmentation ↔ stochasticity
* Randomness (SGD, dropout, augmentation) often improves generalization.

> **Embedded takeaway**
> Regularization is not only about accuracy. It is part of the embedded design pipeline because it promotes:
>
> * smaller, more stable weights (better quantization)
> * fewer effective parameters (better compression)
> * robustness to real-world sensor noise

### Sneak preview: compression as regularization

Pruning and quantization introduce noise/constraints that can act as regularizers.
This can lead to the non-intuitive outcome: **models becoming both smaller and more accurate**.

---










# Chapter 1 — Principles of Neural Architecture Design

Neural architecture design determines both:

* **predictive accuracy**, and
* **computational cost** (parameters, memory traffic, MACs).

In **embedded machine learning**, resources are scarce, so architecture design becomes an explicit exercise in **accuracy–efficiency tradeoffs**.

> **Key objective (embedded)**
> Find architectures that minimize generalization error *and* fit within hardware budgets (memory, compute, latency, energy).

---

## 1.1 Bias–Variance Tradeoff in Model Design

A model’s generalization error can be decomposed into:

* **bias**
* **variance**
* **irreducible error** (noise floor)

### Definitions

* **Bias**: error from overly simplistic assumptions.
  High bias → **underfitting** (misses relevant relationships).
* **Variance**: sensitivity to small changes in the training set.
  High variance → **overfitting** (fits noise as if signal).

### Capacity vs. error (classical curve)

As model capacity increases:

* **bias decreases**
* **variance increases**
* total **generalization error** follows a **U-shape**

> **Best-fit region**
> The bottom of the U-curve corresponds to a model that is neither underfitting nor overfitting.

### Embedded implication

Overly complex architectures are undesirable because:

* they generalize poorly (high variance), and
* they require more **memory** and **compute**

> **Embedded note**
> The goal is not “maximum capacity,” but *optimal capacity under constraints*.

---

## 1.2 Regularization: Taming Model Complexity

**Regularization** prevents overfitting by explicitly discouraging complexity.
It modifies the learning objective by adding a penalty term.

### Regularized objective

$$\mathcal{J}(\mathbf{w}) = \frac{1}{N}\sum_{n=1}^{N}\mathcal{L}(y(\mathbf{w}, \mathbf{x}_n), t_n) + \lambda \mathcal{R}(\mathbf{w})$$

**Where**

* $\mathcal{L}$: task loss (data fit)
* $\mathcal{R}(\mathbf{w})$: complexity penalty
* $\lambda$: penalty strength (regularization parameter)

> **Highlight**
> Regularization typically increases bias slightly to reduce variance substantially, improving generalization.

### Main regularization techniques (high-level)

* **Weight decay ($L1$/$L2$)**
  Penalizes weight magnitude → encourages smaller/simpler solutions.
* **Data augmentation**
  Creates transformed training samples (rotate/crop/flip, etc.) → encourages invariances and robustness.
* **Dropout (Srivastava et al.)**
  Randomly drops neurons during training → prevents co-adaptation; during inference uses full network with scaled weights.

> **Embedded note**
> Regularization is also an efficiency enabler: it supports smaller weights, improved robustness, and often easier compression/quantization.

---

# Chapter 2 — Stabilizing Training with Normalization

Deep networks introduce training instability due to strong layer interdependence.
Normalization methods—especially **Batch Normalization (BN)**—stabilize activations, accelerate convergence, and improve performance.

---

## 2.1 The Challenge of Training Deep Architectures

In deep stacks, updating earlier layers shifts the distribution of downstream inputs.

This causes:

* **unstable training** (cascading distribution drift)
* difficult coordination across layers (higher-order interactions)
* **vanishing/exploding gradients**

> **Key mechanism**
> Distribution shifts at one layer propagate forward, forcing downstream layers to continually adapt.

<figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/batch_normalization_distribution.png' | relative_url }}" alt="a" loading="lazy">
</figure>

---

## 2.2 Batch Normalization (BN): Concept

Batch Normalization is an **adaptive reparametrization** that normalizes activations over a mini-batch, reducing interdependency and improving training stability.

### Core idea

Within a mini-batch, enforce:

* mean ≈ 0
* variance ≈ 1

> **Effect (high level)**
> BN reduces activation scale drift, stabilizes gradient flow, and makes optimization less sensitive to hyperparameters.

<figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/batch_normalization.png' | relative_url }}" alt="a" loading="lazy">
</figure>

---

## 2.3 Mathematical Formulation and Algorithm

BN normalizes each scalar feature independently over a mini-batch $B=\lbrace x_1,\dots,x_m\rbrace$.

### Algorithm 1 — Batch Normalizing Transform

**Input:** mini-batch $B=\lbrace x_1,\dots,x_m\rbrace$
**Learnable parameters:** $\gamma, \beta$
**Output:** $y_i = BN_{\gamma,\beta}(x_i)$

1. **Mini-batch mean**
   
   $$\mu_B \leftarrow \frac{1}{m}\sum_{i=1}^{m}x_i$$
   
2. **Mini-batch variance**
   
   $$\sigma_B^2 \leftarrow \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2$$
   
3. **Normalize**
   
   $$\hat{x}_i \leftarrow \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$$
   
   $\epsilon$: numerical stability constant.
4. **Scale and shift**
   
   $$y_i \leftarrow \gamma \hat{x}_i + \beta \equiv BN_{\gamma,\beta}(x_i)$$

### Why $\gamma$ and $\beta$ exist

Although $\gamma,\beta$ can undo normalization, they are crucial:

* allow the network to represent the identity transform
* make the mean controllable via $\beta$
* simplify optimization for gradient descent

> **Highlight**
> BN normalizes *but still preserves representational flexibility* through learnable scale/shift.

---

## 2.4 Practical Implementation and Considerations

1. **Bias redundancy**
   In $Wx+b$, the bias $b$ becomes redundant because BN subtracts a mean and introduces $\beta$.
   Often simplifies to using $Wx$.

2. **Placement**
   BN can be placed before or after activation:

   * original paper: before non-linearity
   * practice: after can work better depending on architecture

3. **Interaction with dropout**
   BN and dropout can interact unpredictably. A common rule: avoid using both together unless justified.

4. **Inference-time statistics**
   At test time, batches may be too small (even size 1). BN then uses running averages:

   * $\mu'$, $(\sigma'^2)$ accumulated during training

> **Embedded note — Resource constraints**
> In embedded inference, batch processing may be infeasible. Reliance on running averages enables consistent inference normalization.

---

## 2.5 Why Batch Normalization Works (Explanatory Theories)

1. **Reduced internal covariate shift (original argument)**
   * Stabilizes activation distributions; running statistics bridge train/test behavior.

2. **Improved gradient flow**
   * BN mitigates the interdependency between layers during training
   * More consistent layer activation scaling → reduced sensitivity to learning rate.

3. **Smoother optimization landscape**
   * Mini-batch normalization introduces noise (normalizes over a mini-batch) → acts as a mild regularizer and can improve robustness to real-world noise.

> **Observation:** BN smoothes the optimization landscape
---

## 2.6 Alternatives for Different Constraints

BN depends on mini-batch statistics and can degrade with small batch sizes—common in embedded training or memory-limited settings.

### Common alternatives (normalization axes)

* **Batch Norm (BN):** normalize across $N, H, W$ per channel $C$
* **Layer Norm (LN):** normalize across $C, H, W$ per sample (batch-size independent)
* **Instance Norm (IN):** normalize across $H, W$ per channel per sample
* **Group Norm (GN):** normalize within groups of channels across $H, W$ (batch-size independent; often comparable to BN)

> **Embedded note**
> LN and GN are critical when batch sizes are small or variable.

<figure>
   <img src="{{ '/assets/images/notes/embedded-machine-learning/other_normalizations.png' | relative_url }}" alt="a" loading="lazy">
</figure>

---

# Chapter 3 — Foundations of Convolutional Architectures

CNNs exploit spatial structure in images, achieving better performance and efficiency than fully connected designs for vision tasks.

---

## 3.1 Convolution Operation (recap)

A convolution layer applies $M$ learnable filters of size $R \times S \times C$ to an input feature map $H \times W \times C$, producing an output feature map $E \times F \times M$.

### Convolution definition

$$O[z][u][x][y] \sum_{k=0}^{C-1} \sum_{i=0}^{S-1} \sum_{j=0}^{R-1} I[z][k][U x+i][U y+j]\cdot W[u][k][i][j] * B[u]$$

### Dimensions

* Input: $N$ (batch), $C$ (channels), $(H,W)$ (spatial)
* Filters: $M$ (output channels), $(R,S)$ (filter size)
* Output:
  
  $$E = \frac{H - R + 2P}{U} + 1, \qquad F = \frac{W - S + 2P}{U} + 1$$
  
  with stride $U$, padding $P$

---

## 3.2 Quantifying Convolutional Layers: Cost Analysis

Embedded ML requires explicit accounting of:

* **Parameters** (storage)
* **MACs** (compute)
* **Units / activations** (intermediate memory)

| Layer Type | Parameters                                                      | MAC Operations                   |
| ---------- | --------------------------------------------------------------- | -------------------------------- |
| Conv       | $W_c=RSCM$, $B_c=M$, $P_c=W_c+B_c$                              | $MAC_c=(EF)(RSC)M$               |
| FC         | $W_f=\text{In}\cdot\text{Out}$, $B_f=\text{Out}$,$(P_f=W_f+B_f$ | $MAC_f=\text{In}\cdot\text{Out}$ |

> **Highlight — HW–ML interplay**
> Conv cost scales strongly with $E\cdot F$ and $R\cdot S\cdot C$. Efficient design targets these terms.

---

## 3.3 Case Study: AlexNet

Key empirical insight from the layer breakdown:

* **Most parameters** are in **FC layers**
* **Most MACs** are in **CONV layers**

> **Design implication**
> Modern architectures reduce or remove FC layers, often using global pooling and convolutional heads to lower parameter cost.

---

## 3.4 Downsampling with Pooling Layers

Pooling reduces spatial resolution, providing:

1. **Lower compute**
   Subsequent convolutions become cheaper because $MAC_c \propto E\cdot F$.

2. **Translation invariance**
   Presence of features matters more than exact location.

Backprop note:

* max pooling acts as a **gradient router**, passing gradient only to the argmax location.

> **Caution**
> Excessive pooling discards spatial detail and can harm tasks needing localization.

---

# Chapter 4 — Designing for Efficiency: Modern Architectures

Modern CNNs pursue deeper networks without proportional growth in cost. Key mechanisms include:

* grouped convolutions
* factorization + bottlenecks
* residual connections
* dense connectivity

---

## 4.1 Grouped Convolutions

Split input channels into $g$ groups; convolve each independently; concatenate outputs.

### Efficiency

Grouped conv reduces:

* parameters by factor $g$
* MACs by factor $g$

$$W_{cg}=\frac{W_c}{g}, \qquad MAC_{cg}=\frac{MAC_c}{g}$$

> **Highlight — Embedded relevance**
> Grouped convolutions are a direct efficiency tool: fewer MACs and fewer parameters under the same spatial resolution.

---

## 4.2 Inception: Going Deeper Efficiently

Inception (GoogLeNet) targets mobile deployment by reducing parameters while increasing depth.

### Core ideas

1. Replace parameter-heavy FC layers with convolutional structures.
2. Factorize large filters into stacks of smaller ones:
   * two $3\times3$ ≈ one $5\times5$ receptive field with fewer parameters.

### Inception module

Parallel branches (e.g., $1\times1$, $3\times3$, $5\times5$, pooling), then concatenate.

### Role of $1\times1$ convolutions (bottlenecks)

$1\times1$ conv performs channel mixing and dimensionality reduction, reducing the channel depth before expensive filters.

**Example (from notes)**
A $5\times5$ conv on $28\times28\times192$:

| Configuration              | Parameters | MACs        |
| -------------------------- | ---------- | ----------- |
| Plain $5\times5\times32$   | 153,600    | 120,422,400 |
| With $1\times1$ bottleneck | 15,872     | 12,443,648  |

> **Highlight**
> Bottlenecks can reduce parameters and MACs by ~10× while preserving functional capacity.

---

## 4.3 ResNet: Overcoming Degradation with Residual Connections

As networks deepen, accuracy can degrade due to optimization difficulty (not necessarily overfitting). ResNet introduces skip connections to improve gradient flow.

### Residual mapping

Instead of learning $H(x)$, learn:

$$\mathcal{F}(x)=H(x)-x, \qquad \text{output}=\mathcal{F}(x,W)+x$$

### Gradient flow benefit

$$\frac{\partial \mathcal{L}}{\partial x_l} =\frac{\partial \mathcal{L}}{\partial x_L} \left( 1+\frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}\mathcal{F}(x_i,W_i) \right)$$

> **Highlight**
> The “(1)” term ensures a direct gradient path through the identity shortcut, mitigating vanishing gradients.

### Bottleneck ResNet module

Stack $1\times1$ → $3\times3$ → $1\times1$ to reduce compute while enabling very deep networks.

---

## 4.4 DenseNet

DenseNet concatenates feature maps from all previous layers:

* ResNet:
  
  $$x_l=\mathcal{F}_l(x_{l-1})+x_{l-1}$$
  
* DenseNet:
  
  $$x_l=\mathcal{F}_l([x_0,x_1,\dots,x_{l-1}])$$
  

Benefits:

* feature reuse
* strong gradient flow
* requires transition layers to manage growth in channels

---

# Chapter 5 — Summary and Key Metrics

Architecture design for embedded ML balances **accuracy** with **efficiency**.

---

## 5.1 Comparative Analysis (ImageNet-era trend)

| Architecture | Layers | Parameters | MACs | Top-5 Error |
| ------------ | ------ | ---------- | ---- | ----------- |
| AlexNet      | 7      | 61M        | 724M | 16.4%       |
| VGG          | 16     | 138M       | 2.8G | 7.4%        |
| Inception    | 69     | 24M        | 5.7G | 5.6%        |
| ResNet       | 152    | 60M        | 11G  | 4.5%        |

> **Highlight — Empirical lesson**
> “Deeper” can be more efficient than “wider,” when paired with architectural innovations (bottlenecks, factorization, skip connections).

---

## 5.2 Core Efficiency Metrics for Embedded ML

| Metric         | Meaning                              | FC cost | Conv cost | Grouped conv cost |
| -------------- | ------------------------------------ | ------- | --------- | ----------------- |
| **Parameters** | model storage (weight state)         | High    | Moderate  | Low               |
| **Units**      | activation memory (activation state) | Low     | High      | High              |
| **MACs**       | compute/latency/energy               | High    | Very High | Moderate          |

> **Design guidance**
>
> * Grouped conv reduces **MACs + parameters**
> * (1\times1) conv manages **units** and reduces downstream MACs
> * Residual connections improve optimization → allow deeper (often more accurate) models without instability

---

## 5.3 Concluding Remarks

Convolutions remain the core primitive for vision but are computationally expensive. Efficient embedded deployment depends on architectures that explicitly manage complexity through:

* **grouped convolutions**
* **(1\times1) bottlenecks**
* **factorized filters**
* **residual (and dense) connectivity**

> **Central theme**
> Efficient neural architecture design is fundamentally **HW–ML co-design**: aligning algorithmic structure with hardware constraints.

---













# Lecture 7 — Unsafe Optimizations 1

Deep Neural Networks (DNNs) deliver state-of-the-art accuracy in vision and language, but they are inherently **compute- and memory-intensive**. This creates a deployment barrier on embedded devices, where **power, memory, and compute throughput** are strictly limited.

**Key problem**: Modern DNN accuracy is purchased with high FLOPs, large parameter storage, and large runtime activation memory.
**Embedded objective**: Reduce FLOPs, parameter storage, and activation memory **aggressively**, while maintaining acceptable accuracy.

## Trade-Off: Accuracy vs. Efficiency

Model performance correlates strongly with model complexity:
* **deeper/wider models** $\implies$ typically **higher accuracy**
* but also $\implies$ **more FLOPs, more parameters, larger activations**
* **Runtime memory** is a major bottleneck on embedded targets.
   * Example anchor: **ResNet50 ≈ 187 MB** activations.

**Key takeaway**: Embedded deployment requires navigating an explicit **accuracy–resource trade-off**: reduce FLOPs and memory with minimal accuracy loss.


## Safe vs. Unsafe Optimizations

Optimizations can be classified by whether they can change the model’s output.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Safe optimization)</span></p>

**Safe optimization** methods reduce resources without changing accuracy.

**Characteristics:**
* guaranteed preservation of model behavior
* typically hardware/system-level improvements

Safe optimizations are “free” in accuracy, but **require careful hardware/system design**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Safe optimization methods)</span></p>

* **shorter communication paths** (layout / interconnect optimization)
* **data reuse** (caching weights/activations; exploiting locality)
* **dedicated compute architectures** (e.g., systolic arrays for GEMM)

</div>


<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Unsafe Optimization)</span></p>

* **pruning** (remove weights/structures)
* **quantization** (reduce numerical precision: FP32 → INT8 → binary)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Unsafe optimization methods)</span></p>

**Unsafe optimization** methods modify the model/representation to reduce intrinsic cost, potentially affecting accuracy.

**Characteristics:**
* may degrade accuracy; must be measured/mitigated
* core “engineering” challenge in embedded ML

Most embedded ML performance gains come from unsafe optimizations applied carefully.

</div>

## The Physics of Computation: Energy and Precision

Energy and latency depend strongly on:
* **numerical format (bit-width)**
* **data movement** (cache vs DRAM)

**Central principle:** Computation is often cheaper than memory access; optimizing data movement is crucial.

### Numerical Formats: Floating-Point vs Fixed-Point

Here we will focus on the **trade-offs between hardware efficiency (energy/area) and numerical precision**. In modern computing — especially for Artificial Intelligence and Deep Learning — we are moving away from high-precision "perfect" math toward "good enough" math because it is significantly cheaper and faster.

Specifically, we will talk about the mechanical difference between how computers represent numbers and how much physical "space" (area) and electricity (energy) those operations consume.

### Floating-point arithmetic (The "Flexible" Way)

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example:</span><span class="math-callout__name">(Floating-Point)</span></p>

Floating-point works like **scientific notation** ($1.275 \times 10^1$). The decimal point "floats" to the most significant digit, and we store the "zoom level" (exponent) separately.

To represent **12.75** in standard 32-bit Floating Point (IEEE 754):
* **Convert to Binary:** $1100.11$
* **Normalize it:** Shift the point so only one "1" is on the left: $1.10011 \times 2^3$.
* **The Three Parts:**
  * **Sign (1 bit):** `0` (it’s positive).
  * **Exponent (8 bits):** The "3" is stored with a bias (usually +127), so $127 + 3 = 130$. In binary: `10000010`.
  * **Fraction/Mantissa (23 bits):** We store the digits after the point: `10011` followed by eighteen `0`s.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definiton</span><span class="math-callout__name">(Scientific notation, Multiplication)</span></p>

Floating-point provides large dynamic range; Below is scientific notation for computers. **Formula:**

$$v = (-1)^S \cdot (1 + F) \cdot 2^{(E-\text{bias})}$$

$$P_s=A_s \oplus B_s \qquad P_E = A_E + B_E \qquad P_F = A_F \cdot B_F$$

* **Structure**:
  * **sign bit** $S$ (positive or negative).
  * **exponent/fraction** $E$ (moves the decimal point, allowing for huge ranges).
  * **significand** $F$ (the actual digits).

</div>

* **The Distribution Problem:** Floating point is "dense" near zero. The slide notes that half of all representable numbers fall between $−1$ and $1$. This is great for weights in neural networks, which are usually small.
* **Complexity:** Floating point hardware is much more complex because to multiply two numbers, you have to add their exponents ($P_E=A_E+B_E$) and multiply their fractions ($P_F=A_F\dot B_F$) separately.
* **Cost:** FPUs are large and power-hungry relative to integer units.

| Format   | Sign | Exponent | Significand | Dynamic range                     |
| -------- | ---- | -------- | ----------- | --------------------------------- |
| float64  | 1b    | 11b       | 52b          | $\sim 2\times 10^{\pm308}$        |
| float32  | 1b    | 8b        | 23b          | $\sim 2\times 10^{\pm38}$         |
| float16  | 1b    | 5b        | 10b          | $\sim 10^{-4}$ to $10^{+5}$       |
| bfloat16 | 1b    | 8b        | 7b           | similar exponent range to float32 |

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The "bfloat16" Revolution)</span></p>

**bfloat16 relevance:** retains `float32`-like exponent range with fewer mantissa bits → often sufficient for deep learning workloads.
* **The Innovation:** Developed by Google/Intel for AI.
* **Why it matters:** Standard `float16` has a small range. `bfloat16` **uses the same 8-bit exponent as** `float32` but chops the fraction down to 7 bits.
* **The Result:** It has the same "reach" as a big 32-bit number but is much smaller and faster to process. This is the current industry standard for training AI.

If you look at the `bfloat16`, it is a "hybrid" designed for AI. It keeps the big "zoom range" of a 32-bit float but uses a very short 7-bit fraction. This gives AI models the ability to handle massive ranges of numbers without the energy cost of high precision.

The primary trade-off of **bfloat16** is that it sacrifices **precision** (fine-grained detail) to give you a massive **dynamic range** (the ability to handle very large and very small numbers).

**The "Precision vs. Range" Trade-off**

Look at how **bfloat16** compares to the standard **float16** (Half Precision):

* **float16:** Has **10 bits** for the fraction. This means it is more precise—it can tell the difference between $0.1234$ and $0.12335$. However, it only has **5 bits** for the exponent, so it can't represent any number larger than **65,504**.
* **bfloat16:** Has only **7 bits** for the fraction. It is "blurry"—it might see $0.1234$ and $0.12335$ as the same number. But, because it has **8 bits** for the exponent (just like a big 32-bit float), it can represent numbers up to **$3.4\times 10^38$**.

**Why AI prefers this trade-off**

You might wonder: *Why would we want less precision?* In Deep Learning, "Range" is much more important than "Precision" because of a problem called **Gradient Underflow.**

* During training, the computer calculates "gradients" (tiny adjustments to weights). These numbers are often incredibly small (e.g., $0.0000001$).
* If you use **float16**, that number might be too small for the 5-bit exponent to handle, and it gets rounded to **zero**. If the adjustments become zero, the AI stops learning.
* **bfloat16** handles these tiny numbers easily. AI researchers found that neural networks are "robust"—they don't mind if the numbers are a little blurry (low precision), as long as the numbers don't disappear entirely (dynamic range).

**The "Hardware Area" Trade-off**
The lecture mentions: `mult: {Energy, Area} \propto N^2[bits]`. This is a hidden win for bfloat16.
* The physical size of a multiplier on a chip depends mostly on the **significand (fraction)** bits.
* Since bfloat16 only has **7 bits** of fraction (compared to float16's 10 bits), a bfloat16 multiplier is physically **smaller and uses less energy** on the silicon than a float16 multiplier, even though they both use 16 bits total.

**Summary Table: The Catch**

| Format | The "Pro" | The "Con" (Trade-off) |
| --- | --- | --- |
| **float16** | Higher precision; better for "final" models (inference). | Small range; prone to crashing/errors during training. |
| **bfloat16** | Massive range; makes training stable and easy. | Very low precision (only ~2.5 decimal digits of accuracy). |

> **Pro Tip:** If you are doing **scientific simulations** (like landing a rocket), you would *never* use bfloat16 because the rounding errors would accumulate and the rocket would miss. But for **AI** (recognizing a cat), bfloat16 is the "Goldilocks" format.

</div>

### Fixed-point arithmetic (The "Cheap" Way)

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Fixed-Point)</span></p>

In a fixed-point system, we decide ahead of time exactly where the "point" goes. Imagine we have an **8-bit** system where we decide the first 4 bits are for the whole number and the last 4 bits are for the fraction.
* **The Number:** $12.75$
* **Binary of 12:** `1100`
* **Binary of 0.75:** `1100` (Since $0.75 = 1/2 + 1/4$, which is $2^{-1} + 2^{-2}$)
* **The Result:** `1100.1100`

**The Hardware Perspective:** The computer doesn't actually store a "point." It just sees the integer `11001100` (which is 204 in decimal). The programmer just remembers to divide by 16 at the end. This makes addition as fast as regular integer math.

</div>

Fixed-point uses integers with an implicit binary point:
* **smaller dynamic range than floating point**
* **simpler hardware → better energy efficiency**

* **Scaling:** Addition complexity grows linearly with the number of bits ($N$). Multiplication is much harder; it grows quadratically ($N^2$). If you double the bits, a multiplier becomes four times as large/expensive.
* **The Trade-off:** Fixed point is very energy-efficient but has a **small dynamic range**. You can’t represent a very tiny number and a very huge number at the same time without losing precision.

> **Highlight:** Quadratic multiplier scaling creates strong pressure to reduce precision.

| Feature | Fixed-Point (e.g., `1100.1100`) | Floating-Point (e.g., $1.10011 \times 2^3$) |
| :--- | :--- | :--- |
| **Metaphor** | **Ruler**: The marks are always the same distance apart. | **Map**: You can zoom in for high detail or zoom out to see the whole world. |
| **Precision** | Always the same (e.g., always accurate to 0.01). | High precision for small numbers; low precision for huge numbers. |
| **Hardware** | Very simple (uses the Integer Unit). | Complex (requires a dedicated Floating Point Unit). |
| **Best Use** | Money (cents), simple sensors, or low-power microcontrollers. | 3D Graphics, AI training, and complex physics simulations. |

### The Picojoule Economy: Energy Costs

This is the "So What?" table. It uses data from Mark Horowitz (a famous Stanford professor) to show that **energy is the ultimate constraint in computing**.


Energy varies dramatically by operation type.

| Operation    | Precision | Energy (pJ) | Note        |
| ------------ | --------- | ----------- | ----------- |
| Integer add  | 8-bit     | 0.03        |             |
|              | 32-bit    | 0.1         | $3×$         |
| Integer mult | 8-bit     | 0.2         |             |
|              | 32-bit    | 3.1         | $~15×$        |
| FP add       | 16-bit    | 0.4         |             |
|              | 32-bit    | 0.9         |             |
| FP mult      | 16-bit    | 1.1         |             |
|              | 32-bit    | 3.7         |             |
| SRAM access  | 8 kB      | 10          |             |
| SRAM access  | 32 kB     | 20          |             |
| SRAM access  | 1 MB      | 100         |             |
| DRAM access  | —         | 1300–2600   | $~100×$ cache |

**1. The Energy Gap (Integer vs. Floating Point)**
* Look at the tables. An 8-bit Integer Add costs **0.03 pJ**, while a 16-bit Floating Point Add costs **0.4 pJ**.
* **Takeaway:** Floating point operations are roughly **10x more expensive** in terms of energy than integer operations. If you can do your math with integers (Fixed Point), your battery lasts 10x longer.

**2. The Scaling Law**
* The slide reiterates: **ADD scales with $n$, MULT with $n^2$**. A 32-bit integer addition costs **0.1 pJ**.
* A 32-bit integer multiplication costs **3.1 pJ**.
* **Takeaway:** Multiplications are the "energy hogs" of computing. This is why researchers try to design neural networks that use more additions and fewer multiplications.

**3. The "Memory Wall" (The Most Important Part)**
* Look at the **Memory** table on the right.
* A 32-bit Floating Point multiplication costs **3.7 pJ**.
* Reading that same number from **DRAM (Main Memory)** costs **1300–2600 pJ**.
* **The Shocking Truth:** It costs **hundreds of times more energy** to simply "fetch" a number from memory than it does to actually perform the math.


#### Key takeaways

1. **Memory access dominates energy.** DRAM is extremely expensive.
2. **Precision is costly.** 32-bit mult ≫ 8-bit mult.
3. **Multiplication ≫ addition.** Encourages architectures that reduce multiplications (or replace them with bitwise ops).

> **Design rule**
> Maximize locality/data reuse and reduce operand precision where possible.

**Summary: The Big Picture**
If you are designing a chip or software for something like a smartphone or a data center, these slides tell you three things:
* **Lower Precision is Better:** Use bfloat16 or 8-bit integers whenever possible to save 10x–30x energy.
* **Avoid Multiplication:** It’s much more "expensive" than addition because it scales quadratically ($n^2$).
* **Data Movement is the Enemy:** The real "cost" of AI isn't the math; it's moving data from the RAM to the Processor. You must "exploit locality" (keep data in the small, cheap 8kB Cache) to avoid the massive energy cost of DRAM.

> **The final message:** "Need for reduced precision, avoid memory accesses." If you can follow those two rules, you can run much larger AI models on much smaller devices.

## Quantization: Trading Precision for Performance

While the previous topic compared different "languages" (float vs. fixed), this topic explains how to translate a high-precision model into a low-precision one. **Quantization** maps high-precision values to a discrete low-precision set. It is one of the most effective unsafe optimizations for embedded inference.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definiton</span><span class="math-callout__name">(Quantizer)</span></p>

The **Quantizer** $Q$, which is a mathematical function that maps a continuous range of numbers into a set of discrete "buckets". A quantizer $Q$ maps an input $x$ into discrete levels $\lbrace q_\ell\rbrace$, typically using thresholds $\lbrace\delta^\ell\rbrace$.

</div>

**Neural Network Application:** These functions are applied to three main areas of an AI model: the Weights (stored parameters), the Activations (outputs of each layer), and sometimes the Gradients (used during training).
**Trade-off:** Quantization limits "model capacity," meaning the AI might become less "smart" because it can't represent subtle differences in data anymore.

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/weight_distribution.png' | relative_url }}" alt="a" loading="lazy">
</figure>


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Quantization functions)</span></p>

* **Piece-wise Constant Function:** The quantizer is like a staircase. Any input value falling within a certain "step" (interval) gets snapped to the same value (quantization level).
* **Uniform vs. Non-Uniform:** 
  * **Uniform:** All the "steps" on the staircase are exactly the same height/width ($\Delta$). This is easy for hardware to compute but might not fit the data perfectly.
  * **Non-Uniform:** Steps can be different sizes (useful if you have many numbers near zero and few large numbers).
* **Binary Quantization:** The simplest version is the **Sign Function**. If a number is $\ge 0$, it becomes $+1$. If it’s $< 0$, it becomes $-1$. This turns a complex floating-point number into a single bit ($0$ or $1$):

$$
Q(x)=
\begin{cases}
+1 & x \ge 0 \\
-1 & x < 0
\end{cases}
$$

</div>

| Type | Uniform | Non-Uniform | Bits |
| --- | --- | --- | --- |
| **Binary** | $\lbrace -1, +1 \rbrace$ | $\lbrace W_P, W_N \rbrace$ | 1 |
| **Ternary** | $\lbrace -1, 0, +1 \rbrace$ | $\lbrace W_P, 0, W_N \rbrace$ | 2 |
| **Quaternary-** | $\text{Na}$ | $\lbrace W_P, 0, W_N, 0, W_N, 1 \rbrace$ | 2 |
| **Quaternary+** | $\text{Na}$ | $\lbrace W_P, 0, W_N, 0, W_N, 1 \rbrace$ | 2 |

**Bits Column**: The number of binary bits required to store one value in that format.

### Uniform Quantization

Again, Quantizer $Q$: piece-wise constant function
* Input values in given quantization interval mapped to corresponding quantization level
* Apply to activations/weights(/gradients)

$$Q(x) = \underbrace{q_l}_{\text{quantization level } l} \text{ if } x\in \underbrace{(t_l, t_{l+1}]}_{\text{quantization invervals}}$$

Uniform quantization uses constant step size (all level are equidistant):

$$q_{i+1} - q_i = \Delta \quad \forall i$$

**Advantages**
* **Hardware Efficiency:** Uniform quantization is preferred in hardware design because it makes it "easy to store  and compute" using standard bitwidths (like $\log_2(L)$ bits). Since intervals are equally spaced, you only need to store $\log_2(L)$ bits where $L$ is the number of levels
* **No need to store thresholds:** Given min/max range and number of levels, you can compute any threshold

**Disadvantages**
* may waste representation capacity when distributions are non-uniform (e.g., bell-shaped weights)

> **Important note:** "Keep activation function in mind when quantizing" - ReLU outputs are $[0,\infty)$ so you'd use asymmetric quantization, while tanh outputs [-1,1] so symmetric quantization works better.

### Uniform $k$-bit quantizer on $[0,1]$

For $a_i \in [0,1]$:

$$a_i^q = \frac{1}{2^k - 1}\cdot \mathrm{round}\big((2^k-1)a_i\big)$$

Example: $k=2$ maps $[0,1]$ into $5$ representable values
$\lbrace 0, 0.25, 0.5, 0.75, 1\rbrace$ (as stated in the notes).

Quantization can be applied to:
* **weights**
* **activations**
* (sometimes) **gradients**

**Visualizing Precision ($k$)**:
* **$k=1$ (Blue):** Shows a "hard" jump at $0.5$. It provides the most energy savings but results in the highest error.
* **$k=4$ and $k=8$ (Orange/Red):** As k increases, the "staircase" becomes much finer, closely approximating the ideal diagonal line.

**The Trade-off:** By visualizing 10 possible input values on the x-axis, one can mathematically reason about the error — the further the colored line is from the diagonal, the more "knowledge" the AI loses.

**Optimization goal:** Use the smallest bit-width that preserves acceptable accuracy.

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/uniform_quantization_k_bits.png' | relative_url }}" alt="a" loading="lazy">
</figure>

### Non-Uniform Quantization

Non-uniform quantization allocates more levels where values are dense.

**Advantages**
* **Imporves model capacity:** higher accuracy at same bit-width by matching data distribution

**Disadvantages**
* **Slightly more storage:** requires storing quantization levels (lookup tables) ($\log_2(L)$ bits + the levels)

> **Highlight**
> Learnable thresholds/scales allow the quantizer to adapt to layer-specific distributions (remember bell curve figure).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example:</span><span class="math-callout__name">(Trainable ternary quantizer)</span></p>

$$
w^i_l =
\begin{cases}
W^p_l : w_l > \Delta_l \\
0 : \lvert w_l\rvert\le \Delta_l \\ 
- W_l^n & w_l < -\Delta_l
\end{cases}
$$

with $\Delta_l$ defined via a learnable fraction of the maximum weight: 
$\Delta_l = t\cdot\max(\lvert w\rvert);t\in [0,1]$

</div>

### Hardware Acceleration and Low-Precision MACs

DNNs are dominated by **MAC operations** (matrix multiplication/convolution): 

```python
sum = sum + A_ik * B_kj # <- Multiply-Accumulate (MAC) operation
```

This **multiply-accumulate (MAC)** is the bottleneck operation repeated billions of times in deep learning. Lower precision enables more parallelism per area/energy.

### Hardware intuition (bit-width scaling)

* **Cycle count** = How long ONE operation takes (latency)
* **Throughput** = How many operations you can complete per second (total work done)

#### **32-Bit MAC (Standard Precision) - 6 Cycles**

The standard approach uses **full 32-bit floating-point arithmetic**:
* Takes two 32-bit inputs (int32x1_t shown in blue)
* Uses a **Fused Multiply-Add (FMA)** unit
* Completes in **6 clock cycles**
* This is what happens in typical CPUs/GPUs with FP32 operations

#### **8-Bit MAC - 36 Cycles**

* You need **four 8-bit multiply units (MUL)** running in parallel
* Intermediate results are stored in wider formats (`2int16x2_t`, then `2int32x1_t`)
* Multiple **addition (ADD)** stages to accumulate partial products
* The bit-width doubling occurs: when you multiply two n-bit numbers, you get a 2n-bit result
* Formula shown: **add: n+1 bits, mult: 2n bits**

The increased cycle count comes from **managing bit-width expansion** and **accumulation overhead**. However, the energy consumption and area are much lower than 32-bit, and you can pack more operations in parallel.

#### **1-Bit MAC (Binary) - 15 Cycles**

This is the most radical optimization using **binary neural networks** (weights are $\pm 1$):
* Multiplication by $\pm 1$ becomes just a **sign flip**
* The slide shows **XNOR operations** instead of multiplications
* Multiple **XOR gates** performing bitwise operations
* **CNT (count)** operations to accumulate bit patterns
* Multiple **ADD** stages to combine results

#### In Hardware Terms

**32-bit MAC unit:**
* 6 cycles per operation
* Large, power-hungry circuit (lots of transistors for 32-bit multipliers)
* Maybe you fit **1,000 units** on a chip

**1-bit MAC unit:**
* 15 cycles per operation (2.5× slower!)
* Tiny circuit (just XNOR gates and counters)
* You might fit **100,000 units** on the same chip (100× more!)

**The math:**
* 32-bit: 1,000 units ÷ 6 cycles = **167 operations per cycle**
* 1-bit: 100,000 units ÷ 15 cycles = **6,667 operations per cycle**

The 1-bit system is **40× faster overall** despite taking 2.5× longer per individual operation!

**Why This Matters for Neural Networks**
Neural networks need to do **billions** of MAC operations. What matters is:
* How many you can do **simultaneously** (parallelism)
* Total **energy** consumed
* **Chip area** available

The 1-bit approach wins because:
* **Massive parallelism:** Fit way more units in parallel
* **Lower power:** Each XNOR gate uses ~1/100th the power of a 32-bit multiplier
* **Higher total throughput:** Complete the entire neural network inference faster

So yes, each individual 1-bit operation is slower (15 cycles vs 6), but you're running **thousands more in parallel**, giving you much higher overall performance.

This is why modern AI accelerators (like Google's TPU) use massive arrays of low-precision arithmetic units rather than fewer high-precision ones!

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/qnns_eigen_extension.png' | relative_url }}" alt="a" loading="lazy">
</figure>

### XNOR-based binary multiplication

> **`popc` instruction:** The popcnt instructions, short for population count, are used to count the amount of 1s in a numbers binary representation.
> **`xnor` instruction:** Exclusive NOR instruction performs logical equality, outputting a high (1) if all inputs are the same (all 0s or all 1s) and a low (0) if inputs differ, essentially checking for similarity; while not a single common instruction in basic x86, it's implemented via XOR/NOT or specific vector instructions

**Coding (XNOR):**

| a | b | !(a^b) |
| --- | --- | --- |
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

For binary networks with $\lbrace -1,+1\rbrace$, map to $\lbrace 0, 1\rbrace$.

* $c = a\cdot b = popc(xnor(a,b)) - (N - popc(xnor(a,b))) = 2 \cdot popc(xnor(a,b)) - N$
  * $N$ is a length the vectors $a$ and $b$.
  * $-1 := 0$, $+1 := 1$
* Eg. a = [10010], b = [11100]
  * $c = 1 \cdot 1 + -1 \cdot 1 + -1 \cdot 1 + 1 \cdot -1 + -1 \cdot -1 = -1$
  * $c = 2 \cdot popc(xnor(a,b)) - N$
  * $= 2 \cdot popc([10001]) - N = 2 \cdot 2 - 5 = -1$
* Eg. a = [00000], b = [00000]
  * $c = 5$
* Eg. a = [11111], b = [00000]
  * $c = -5$

* While 15 cycles seems long, the teacher likely emphasized:
  * **XNOR gates are tiny** compared to multipliers
  * **Power consumption is orders of magnitude lower**
  * You can fit **massively more** 1-bit operations on a chip
  * The cycle count is misleading - throughput matters more

> **Key idea**
> Replace multiply-add with XNOR + popcount → extremely efficient on suitable hardware.

## Training Strategies for Quantized Networks

Applying quantization naively can significantly degrade accuracy. To counteract this, several training strategies have been developed.

### 1) Post-Training Quantization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definiton</span><span class="math-callout__name">(Post-Training Quantization)</span></p>

A fully trained, high-precision model is converted to a lower precision after training is complete. It **requires a small "calibration dataset"** (a representative sample of the training data) to **determine the quantization parameters** (e.g., scale and zero-point). While easy to apply, PTQ often **results in a noticeable loss of accuracy**, especially for very low bit-widths.

</div>

### 2) Quantization-Aware Training

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definiton</span><span class="math-callout__name">(Quantization-Aware Training)</span></p>

This method **simulates the effects of quantization during the training** itself:
* **Forward Pass:** Weights and/or activations are "fake quantized." They are quantized to a low precision (e.g., 8-bit integer) and then immediately de-quantized back to high precision (e.g., 32-bit float) before being used in the computation. This injects quantization noise into the training, forcing the model to learn weights that are robust to this effect.
* **Backward Pass:** Gradients are computed using the high-precision values, allowing for stable training updates.

</div>

### 3) Straight-Through Estimator (STE)

A significant challenge in QAT is that the quantization function is piece-wise constant, meaning its gradient is zero or undefined everywhere. This would stall training. To overcome this, the **Straight-Through Estimator (STE)** is used. 

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definiton</span><span class="math-callout__name">(Straight-Through Estimator)</span></p>

**Straight-Through Estimator** simply passes the gradient through the quantization function as if it were an identity function (i.e., assuming a gradient of 1). The gradient for a weight $w$ is approximated as:

$$
\frac{\partial\mathcal{L}}{\partial w}
\approx
\frac{\partial\mathcal{L}}{\partial f}\frac{\partial f}{\partial w} \approx \frac{\partial\mathcal{L}}{\partial f} \tilde{f}'(w),
$$

where $\tilde{f}'(w)$ is a surrogate derivative, often just set to 1. This allows the high-precision weights to be updated based on gradients computed from the quantized weights.

</div>

### 4) Fine-tuning

**Re-training/Fine-tuning:** A hybrid approach where a pre-trained model is quantized and then fine-tuned for a few epochs using QAT. This can often recover much of the accuracy lost during initial quantization.

> **Highlight**
> QAT + fine-tuning is often necessary for low-bit quantization without major accuracy loss.

## 4.6 Survey: Quantization Schemes and Performance

| Method     | Weights $W$     | Activations $A$  |
| ---------- | --------------- | ---------------- |
| BNN        | $\lbrace -1,+1\rbrace$       | $\lbrace -1,+1\rbrace$        |
| XNOR-Net   | $\lbrace -S,+S\rbrace$       | $\lbrace -1,+1\rbrace$        |
| DoReFa-Net | $\lbrace -S,+S\rbrace$       | $\lbrace 0,+1\rbrace$ (k-bit) |
| TWN        | $\lbrace -S,0,+S\rbrace$     | float32          |
| TTQ        | $\lbrace -S_n,0,+S_p\rbrace$ | float32          |
| HWGQ       | XNOR                       | 2-bit            |

**Empirical trend:**
Performance data for these methods on the AlexNet/ImageNet task shows that while extreme 1-bit quantization (BNN, XNOR) suffers a significant accuracy drop, more moderate schemes can approach the full-precision baseline. For example, TTQ (with 2-bit weights and 32-bit activations) achieves 79.7% Top-5 accuracy, very close to the 80.3% of the 32-bit baseline.

**Graphs in the lecture illustrate these trade-offs:**
* An "Improvement factor" graph shows that memory footprint improves exponentially as operand bit-width decreases (e.g., a $2^4$ factor improvement going from 16-bit to 1-bit). Latency also improves, though less dramatically.
* A "Test error" graph shows that as the number of bits increases from 1 to 3, the test error for various methods (Lq-Net, BNN, DoReFa, etc.) drops, approaching the full-precision baseline error rate.

> **Trade-off summary**
> Lower bits → strong memory/latency improvements, but accuracy recovers substantially by 2–3 bits depending on method.
> **Key observation:** DNNs contain plenty of redundancy. Nonuniform quantization outperforms uniform quantization

## Pruning: Engineering Sparsity in Neural Networks

**Pruning** is another powerful unsafe optimization technique aimed at reducing model size and computational cost. It operates by removing "unimportant" connections or neurons from a trained network, effectively setting their corresponding weights to zero.

### Principle: Inspired by Nature

The concept of pruning is biologically inspired. A diagram titled "Evolution of Human Brain During Life" shows that the density of synaptic connections in the human brain peaks around age 6 and is then gradually reduced by age 14. This process of synaptic pruning is believed to refine and optimize neural circuits. Similarly, DNNs are often over-parameterized, containing significant redundancy that can be removed without harming, and sometimes even improving, generalization.

> **Key hypothesis**
> Many weights are not critical for prediction and can be eliminated with minimal accuracy loss (especially with retraining).

### Pruning Workflow and Criteria

Pruning is not a single action but a process, typically involving three stages:
1. **Train:** A standard, dense network is trained to convergence.
2. **Prune:** A certain fraction of the network's weights are set to zero based on a specific criterion.
3. **Fine-tune:** The now - sparse network is retrained for a few epochs to allow the remaining weights to adjust and recover any accuracy lost during pruning.

Often performed iteratively:

$$\text{Train} \rightarrow \text{Prune} \rightarrow \text{Fine-tune} \rightarrow \cdots$$

to achieve higher levels of sparsity without catastrophic drops in accuracy. This contrasts with **one-shot pruning**, where the entire pruning process happens at once.

#### Common pruning criteria

The critical question is which connections to remove. Common criteria include:
* **Magnitude pruning:** Weights with the smallest absolute values are removed, based on the heuristic that they have the least influence on the network's output.
  
  $$\lvert w_i\rvert \le t \Rightarrow w_i \gets 0$$
  
* **Gradient-based (saliency) pruning:** This method considers not just the weight's size but also its impact on the loss function.
  
  $$\lvert w_i g_i\rvert \le t \Rightarrow w_i \gets 0$$

> **Highlight**
> Magnitude pruning is simple and widely used; gradient-based measures attempt to capture impact on loss.

### Granularity: Unstructured vs Structured Sparsity

Pruning granularity strongly affects hardware speedups.

#### Unstructured (fine-grained) pruning

This approach removes individual weights anywhere in the network based on the pruning criterion.

**Pros**
* Offers the highest potential for accuracy at a given sparsity level, as it provides maximum flexibility.

**Cons**
* Extremely difficult to accelerate on parallel hardware like **GPUs** and **CPUs**.
* A sparse matrix resulting from unstructured pruning requires an index to store the location of each non-zero element. This leads to indirect memory accesses, which are inefficient and break memory coalescing patterns, often resulting in performance that is no better, or even worse, than the original dense matrix. The **Compressed Sparse Row (CSR)** format, which uses data arrays (d), column indices (i), and row pointers (r), exemplifies this overhead.
  * requires indices (e.g., CSR format: data $d$, column indices $i$, row pointers $r$)
  * irregular memory access breaks coalescing/locality and can negate speedups

> **Embedded note**
> Unstructured sparsity often compresses storage but may not reduce latency unless hardware explicitly supports sparse compute efficiently.

#### Structured (coarse-grained) pruning

This approach removes entire groups of weights at once, such as entire filters/channels in a convolutional layer or rows/columns in a fully connected layer:
* channels/filters in CNNs
* rows/columns in FC layers
* blocks/tiles

**Pros**
* Creates smaller, **dense matrices** that are perfectly suited for hardware acceleration. It preserves the regular computational patterns that parallel processors are designed for.

**Cons**
* Less flexible than unstructured pruning, which can lead to a greater loss in accuracy for the same number of removed parameters.

#### Parameterized structured pruning (learnable structure selection)

Advanced techniques aim to make this structure learnable. **Parameterized Structured Pruning** divides a weight tensor $W$ into sub-tensors, each representing a structure. Each structure $w_i$ is associated with a learnable parameter $\alpha_i$. The structure is kept or pruned based on whether $\lvert\alpha_i\rvert$ is above a threshold $\epsilon$. This is implemented using a thresholding function $v_i(\alpha_i)$:

$$
w_i^{qi}=w_i\cdot v_i(\alpha_i),
\qquad \text{where} \qquad 
v_i(\alpha_i)=
\begin{cases}
0 & \lvert \alpha_i\rvert<\epsilon \\
\alpha_i & \lvert \alpha_i\rvert\ge \epsilon
\end{cases}
$$

Since this function is not differentiable at the threshold, **STE** is used during backpropagation to update the $\alpha_i$ parameters, allowing the network to learn which structures to keep.

Thus, gradient of $\alpha_i$ is claculated folllowing the chain rule
* Trained together with weights using GD based on loss J, but regularized and pruned independently.

**Update rule 1:**

$$\Delta\alpha_i(t+1) := \mu\Delta\alpha_i(t)-\eta\frac{\partial J}{\partial\alpha_i(t)} - \lambda\eta\cdot\alpha_i(t)$$

**Update rule 2:**

$$\Delta\alpha_i(t+1) := \mu\Delta\alpha_i(t)-\eta\frac{\partial J}{\partial\alpha_i(t)} - \lambda\eta\cdot\text{sign}(\alpha_i(t))$$

Surprisingly, **option 1 outperforms option 2**. Different learning dynamics, seen in weight distributions L2 produces unimodal, bimodal and trimodal distributions with clear distinctions, while L1 lacks those distinctions

### Impact of Retraining and Regularization

Fine-tuning after pruning is mandatory to recover model accuracy. The process of pruning and retraining fundamentally alters the weight distribution of the model.

#### Observed weight distribution shift (from notes)

A diagram shows two histograms:
1. **Weight distribution before pruning:** A typical Gaussian-like distribution centered narrowly around zero.
2. **Weight distribution after pruning and retraining:** The distribution becomes bimodal. The peak at zero is gone (due to pruning), and the remaining weights have been pushed away from zero during retraining, forming two distinct clusters. This shows that retraining strengthens the remaining connections to compensate for those that were removed.

#### Role of regularization

The choice of regularization during training also plays a role.

* **L1**
  
  $$\mathcal{R}_{L1}(w)=\sum_j \lvert w_j\rvert$$
  
  encourages sparsity (pushes weights toward zero)
* **L2**
  
  $$\mathcal{R}_{L2}(w)=\frac{1}{2}\sum_j w_j^2$$
  
  encourages small weights but does not force them to be exactly zero.

**Empirical note:** For AlexNet/ImageNet in the notes:
* with retraining, **L2** ultimately performs best for pruning
* without retraining, **L1** is better
* but the key takeaway is that **retraining is essential** for achieving high performance with pruned models.

> **Key takeaway**
> Pruning is not a one-step operation; iterative pruning + fine-tuning is the standard route to high sparsity with acceptable accuracy.














# Unsafe Optimizations II — Quantization

This chapter provides a deep dive into **quantization**, a critical model compression technique for deploying machine learning models on resource-constrained embedded systems. The central theme is the interplay between ML algorithms, their numerical representation, and the underlying hardware's limitations in terms of memory, energy, and latency.

## 1.1 Introduction to Model Compression

Deploying complex neural networks on embedded devices necessitates model compression to meet strict constraints on memory footprint, computational power, and energy consumption. The two primary families of compression techniques are Pruning and Quantization.

* **Pruning:** This technique involves removing redundant parameters (weights or neurons) from a trained network. The goal is to reduce the model's size and the number of computations required during inference. As described in a diagram, pruning can be **unstructured**, where individual connections (synapses) are removed based on criteria like their magnitude, or **structured**, where entire neurons or filters are removed. While effective, the efficiency gains from pruning are highly dependent on hardware support for sparse computations.
* A diagram contrasts a fully connected network layer with pruned versions. "Before pruning," all neurons are interconnected. "After pruning" shows two outcomes:
  1. **Pruning synapses:** Individual connections are removed, leading to a sparse weight matrix.
  2. **Pruning neurons:** Entire neurons and all their associated connections are eliminated, reducing the layer's dimensions.
* **Quantization:** This technique reduces the numerical precision of a model's parameters (weights) and, optionally, its intermediate calculations (activations). By representing floating-point numbers (e.g., **Float32**) with lower-bit integers (e.g., **INT8, INT4**, or even binary values), quantization drastically reduces memory footprint and can enable faster, more energy-efficient integer-based arithmetic on compatible hardware.

The choice of data type, number format, and bit width, as well as whether the quantization is applied homogeneously across the model or heterogeneously (per-layer, per-filter, etc.), are key design decisions. Like pruning, the ultimate efficiency of quantization is deeply tied to the target hardware architecture.

## 1.2 A Hardware-Aware Metric: Bit Operations (BOPS)

To fairly compare different model compression techniques, especially those using custom or low-precision data types, a suitable hardware-abstract metric is required. The traditional metric of **Multiply-Accumulate operations (MACs)** is often insufficient, as it does not capture the cost variations associated with different bit widths.

### **Why MACs aren’t enough**

A MAC count treats every multiply-accumulate as equal:

* 1 MAC with 32-bit float
* 1 MAC with 8-bit int
* 1 MAC with 1-bit XNOR-popcount style arithmetic

…all counted the same. But on real hardware, those have drastically different cost/energy/area. So we want a *hardware-abstract but bit-aware* metric.

The **Bit Operations (BOPS)** metric provides a more accurate measure of computational complexity for fixed-point arithmetic. For a standard convolutional layer, the BOPS can be approximated by considering the bit widths of weights and activations.

### **What BOPS is trying to approximate**

**BOPS (Bit Operations)** approximates cost by:
* **Multiplication cost** grows roughly with the *product* of operand bit widths.
* **Addition/accumulation cost** grows roughly with the accumulator bit width.

It’s not a perfect model of every accelerator, but it captures the big effect: **lower precision is cheaper**.

Let's define the parameters for a convolutional layer:
* $b_w$: bit width of weights
* $b_a$: bit width of activations
* $n$: number of input channels
* $m$: number of output channels (filters)
* $k$: filter size (e.g., $k \times k$)

The total number of MACs per output element is $\approx nk^2$:

$$\underbrace{n k^2}_{\text{products}} \text{ multiplications} \quad + \quad (n k^2 - 1)\text{ additions}$$

People usually call that “$nk^2$ MACs”.

### **Where the accumulator bit width $b_o$ comes from**

**Max size of one product**

If activations have bit width $b_a$ and weights have bit width $b_w$, the *maximum magnitude* of each (unsigned) value is about $2^{b_a}$ and $2^{b_w} (more precisely $2^{b_a}-1$, etc.). A product therefore can be as large as roughly:

$$2^{b_a} \cdot 2^{b_w} = 2^{b_a + b_w}$$

So **one multiply produces a value needing about $b_a + b_w$ bits**.

*(If signed two’s complement is used, you typically need an extra sign bit / careful bounds; more on that at the end.)*

**Summing $nk^2$ products**

Worst case, all products have the same sign and add up. So the max sum is approximately:

$$(nk^2)\cdot 2^{b_a+b_w}$$

To store a value up to $X$, you need about $\log_2(X)$ bits. Therefore:

$$b_o \approx \log_2!\left((nk^2)\cdot 2^{b_a+b_w}\right) = (b_a+b_w) + \log_2(nk^2)$$

So:

$$\boxed{b_o = b_a + b_w + \log_2(nk^2)}$$

In practice you’d use a ceiling:

$$b_o = b_a + b_w + \left\lceil \log_2(nk^2)\right\rceil$$

**Intuition:** every time you double the number of terms you sum, you need about **one more accumulator bit**.

### **Where the BOPS formula comes from**

They model total per-layer bit ops (again per output spatial position) as:

$$\text{BOPS}_{\text{conv}} \approx mnk^2\Big(\underbrace{b_ab_w}_{\text{mult}} + \underbrace{b_o}_{\text{acc}}\Big)$$

### Why multiplication cost $\sim b_a b_w$

**The schoolbook binary view (most intuitive)**

Let
* $a$ be $b_a$-bit
* $w$ be $b_w$-bit

Write $a$ in bits:

$$a=\sum_{i=0}^{b_a-1} a_i 2^i,\quad a_i\in{0,1}$$

Then

$$a\cdot w=\left(\sum_{i=0}^{b_a-1} a_i 2^i\right)w=\sum_{i=0}^{b_a-1} a_i (w\ll i)$$

So multiplication becomes:

* For each bit $a_i$, either add **0** or add a **shifted copy of $w$**.

Now, how expensive is “$a_i (w\ll i)$” at the bit level?

* $a_i$ is 0/1, so it “gates” each bit of $w$.
* That’s basically $b_w$ AND operations (one per bit of $w$) to form the partial product row.
* You do that for each of the $b_a$ bits of $a$.

So just forming partial products costs about:

$$b_a \cdot b_w$$

bit operations (ANDs).

Then you still have to add those rows up (more cost), but many BOPS formulas bundle “multiply cost” as scaling with $b_a b_w$ because that’s the dominant size term and matches how multiplier hardware scales.

**Hardware intuition (area/energy scaling)**

Common multiplier circuits (e.g., array multipliers) are literally a 2D grid of bit-cells roughly sized $b_a \times b_w$. Even with optimizations (Booth encoding, Wallace trees), the **resource/energy** still grows roughly with operand width product.

So $b_a b_w$ is a compact proxy for:

* “how many bit-level interactions do we need between the two numbers?”

Key takeaway: **halve both bit-widths → multiplication cost drops ~4×** (quadratic effect).

**Why accumulation cost $\sim b_o$**

Adding two $b_o$-bit numbers costs work proportional to $b_o$ in basic adder models (ripple-carry $~O(b_o)$, more advanced adders still grow with width in area/energy).

Key takeaway: **halve precision → accumulator work drops ~2×** (linear-ish effect), and you *also* reduced $b_o$ because it contains $b_a+b_w$.

### “HW–ML interplay” from the equation (the trade-off)

This formula highlights the **HW-ML Interplay**: 
reducing the bit width of weights ($b_w$) and activations ($b_a$) quadratically reduces the multiplication cost and linearly reduces the accumulation cost, leading to significant computational savings. A scatter plot of various models illustrates this trade-off, showing model accuracy versus billions of bit operations. Techniques like UNIQ, QNN, and XNOR aim to push models into the high-accuracy, low-BOPS regime. From

$$
mnk^2(b_ab_w + b_o)
\quad\text{with}\quad
b_o = b_a+b_w+\log_2(nk^2)
$$

you can see:

* **Multiplication part:** $b_ab_w$ → *quadratic* benefit when reducing both.
* **Accumulation part:** $b_o$ → roughly *linear* in $b_a+b_w$, plus a fixed-ish $\log_2(nk^2)$ term determined by layer shape.

So quantization buys you large savings mostly because multiplies get way cheaper, and accumulators also shrink but less dramatically.

That’s the “push models into high-accuracy, low-BOPS”: keep accuracy while making $b_a, b_w$ small.


## 1.3 Fundamentals of Quantization

**Quantization** is the process of mapping a continuous or large set of values to a smaller, discrete set. In machine learning, this typically means mapping 32-bit floating-point numbers to low-bit integers.

The core quantization mapping is defined by the following equation:

$$q = Q(x) = \text{clip}(\text{round}(\frac{x}{s} + z), q_{\text{min}}, q_{\text{max}})$$

Where:
* **Glossary of Terms:**
  * $x$: The original, continuous (floating-point) value to be quantized.
  * $q$: The quantized integer representation of $x$.
  * $s$: The **scale factor**, a positive float that determines the step size or resolution of the quantization.
  * $z$: The **zero-point**, an integer that specifies which quantized value corresponds to the real value of 0. This allows for an offset in the mapping.
  * $q_{min}$, $q_{max}$: The minimum and maximum allowed values for the target integer data type (e.g., -128 and 127 for signed `INT8`).

To recover the original value (with some error), a dequantization step is performed:

$$\hat{x} = (q - z) \cdot s$$

Here, $\hat{x}$ is the reconstructed floating-point approximation of $x$. The difference between $x$ and $\hat{x}$ is the **quantization error**, which arises from both rounding and clipping.

A visual representation of this process is shown in a graph titled "Effect of Int2 Quantization on Function Approximation," where a smooth sine wave (Float32) is approximated by a coarse, step-like function (Int2), clearly showing the information loss inherent in the process.

### Symmetric vs. Asymmetric Quantization

The choice of $s$ and $z$ defines the quantization scheme.

* **Symmetric Quantization:** The range of real values is mapped symmetrically around zero. This is achieved by setting the zero-point $z=0$. The scale is typically calculated as:

$$s = \frac{\max(\lvert x\rvert)}{q_{\text{max}}}$$

This scheme is commonly used for weights, which often have a distribution centered around zero.
* **Asymmetric Quantization:** The range is not necessarily centered at zero. This requires both a scale and a non-zero zero-point. The parameters are calculated as: 

$$s = \frac{x_{\text{max}} - x_{\text{min}}}{q_{\text{max}} - q_{\text{min}}}, \quad z = \text{round}(q_{\text{min}} - \frac{x_{\text{min}}}{s})$$

This scheme is often used for activations, especially after a ReLU function, where all values are non-negative.

### Static vs. Dynamic Quantization

The method for determining the range ($x_{\text{min}}, x_{\text{max}}$) also defines a key characteristic.

* **Static Scaling:** The scale ($s$) and zero-point ($z$) are pre-computed offline using a representative **calibration dataset**. These parameters are then fixed and used for all inferences at runtime. This approach has zero runtime overhead but is vulnerable if the deployment data distribution differs from the calibration data.
* **Dynamic Scaling:** The range ($x_{\text{min}}, x_{\text{max}}$), and thus $s$ and $z$, are computed on-the-fly for each input or batch at runtime. This is more robust to varying input distributions but introduces computational overhead during inference.

## 1.4 A Taxonomy of Quantization Techniques

The design space of quantization is vast. Understanding its taxonomy is crucial for making informed decisions.

| Dimension | Options | Description & Trade-offs |
| :--- | :--- | :--- |
| **Procedure (Timing)** | **Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)** | **PTQ** (Post-Training Quantization) is applied to a pre-trained model without retraining; it is fast and simple, often effective for **INT8**. **QAT** (Quantization-Aware Training) simulates quantization effects during the training loop, allowing the model to adapt. It is more complex but necessary for aggressive quantization (e.g., **INT4**). |
| **Target** | **Weight-Only vs. Weights + Activations** | **Weight-only** reduces model size and memory bandwidth, but computations often remain in higher precision. **Weights + Activations (W+A)** enables fully integer-based pipelines, yielding maximum speedups and energy savings on compatible hardware. |
| **Range Determination** | **Static vs. Dynamic** | **Static** uses a fixed range from a calibration set, offering fast inference but sensitivity to data shifts. **Dynamic** calculates ranges at runtime, providing robustness at the cost of latency overhead. |
| **Granularity** | **Per-Tensor vs. Per-Channel vs. Per-Group** | **Per-tensor** uses a single $s$ and $z$ for an entire weight tensor; it is simple but can be suboptimal. **Per-channel** computes separate scales for each output channel, offering better robustness. **Per-group** is a middle-ground trade-off, sharing a scale across small groups of channels. |

A practical escalation strategy for applying quantization is:

1. Start with **INT8 PTQ**.
2. If accuracy drops, improve the **calibration** process (e.g., use percentile ranges instead of min/max) or increase **granularity** (e.g., move from per-tensor to per-channel).
3. Consider using **mixed precision**, keeping sensitive layers at a higher precision.
4. As a final resort, use the more complex **QAT** process to retrain the model.

## 1.5 Practical Quantization: A ResNet Case Study

Let's examine the quantization of the first convolutional layer (`conv1`) of a ResNet model to illustrate these concepts.

### INT8 Quantization Example

* **ResNet `conv1` Weights:**
  * A histogram of the original Float32 weights shows a bell-shaped distribution sharply peaked at zero, with values ranging from approximately -1.0 to 1.0.
  * **Method:** Symmetric `min/max` quantization for a signed **INT8** range `[-128, 127]`.
  * **Parameters:** `scale = 0.007972`, `zero_point = 0`.
  * **Result:** The quantized integer codes (shown in an inset plot) mirror the original distribution. The dequantized histogram (overlaid on the original) shows a very close approximation, indicating minimal information loss.
* **ResNet ReLU Activations:**
  * A histogram of the **Float32** activations shows a distribution skewed to the right, starting at zero (due to ReLU) with a long tail. Outliers are present.
  * **Method:** Asymmetric `min/max` quantization for a signed **INT8** range `[-128, 127]` (though values will be positive).
  * **Parameters:** `scale = 0.01287`, `zero_point = -128`.
  * **Result:** The dequantized histogram again closely follows the original, demonstrating that **INT8** is often sufficient for both weights and activations in CNNs with proper calibration.

### INT4 Quantization Example (Aggressive)

* **ResNet `conv1` Weights:**
  * **Method:** Symmetric min/max for a signed INT4 range `[-7, 7]`.
  * **Parameters:** scale = 0.1452, zero_point = 0.
  * **Result:** The dequantized histogram is much coarser. The values are visibly grouped into a few discrete bins, indicating significant precision loss. The large `scale` value means low resolution.
* **ResNet ReLU Activations:**
  * **Method:** Asymmetric `min/max` for an unsigned **INT4** range `[0, 15]`.
  * **Parameters:** `scale = 0.2187`, `zero_point = 0`.
  * **Result:** The quantization effect is dramatic. The smooth distribution is replaced by a few sharp peaks, showing that many distinct activation values have been mapped to the same integer code. This level of aggressive quantization typically requires QAT to maintain acceptable accuracy.

## 1.6 The Challenge of Real-World Data: Calibration Mismatch

A core assumption of **static activation quantization** is that the data distribution seen during deployment will match the distribution of the calibration set (`calibration ≈ deployment`). When this assumption is violated, the pre-computed scale and zero-point become suboptimal, leading to accuracy degradation.

**Sources of Mismatch:**
* **Lighting/Exposure Shifts:** Changes in lighting can cause the "right tail" of activation distributions to grow, as more high-magnitude activations appear.
* **Sensor/Optics Changes:** Different cameras have unique noise profiles, color responses, and lens artifacts, altering the overall input distribution.
* **Compression Artifacts:** Using different image formats (e.g., RAW vs. JPEG) or streaming introduces artifacts like blocking and ringing, which can distort activation histograms.

**Effect of Mismatch:** An illustration shows activation histograms for ResNet-18 on CIFAR-10 under "Clean calibration" and "Mismatch calib" conditions. Under mismatch, the distribution's tail extends further. This forces a min/max-based calibration to select a larger range, which in turn increases the scale factor. A larger scale means lower resolution for the bulk of the values, increasing quantization error. The example shows the min/max scale increasing from `0.01287` (clean) to `0.01672` (mismatch).

**Mitigation Strategies:**
1. **Diversify the calibration set** to be more representative of real-world conditions.
2. Use more robust range selection methods like **percentile** or **MSE-based** calibration, which are less sensitive to outliers.
3. Increase quantization **granularity** (e.g., per-channel scales).
4. Employ **mixed precision**, keeping sensitive layers (often the first and last) in higher precision.

## 1.7 Advanced Quantization-Aware Training (QAT) Methods

When PTQ is insufficient, QAT methods integrate quantization into the training process. These techniques allow the model to learn weights that are more robust to the effects of low-precision arithmetic.

### Trained Ternary Quantization (TTQ)

TTQ quantizes weights to three values: positive ($W_p$), negative ($-W_n$), and zero. It uniquely learns the optimal scaling factors $W_p$ and $W_n$ during training.

The process, illustrated in a flow diagram, is as follows:
1. **Forward Pass (Inference Time):** Full-precision weights are quantized to an intermediate ternary representation $\lbrace -t, 0, +t\rbrace$ based on a threshold $\Delta_l$. This is then scaled by the learned values $W_p$ and $W_n$ to produce the final ternary weight.
2. **Backward Pass (Training Time):** The loss gradient is backpropagated through the network. Crucially, TTQ computes two sets of gradients:
  * `gradient1`: Propagates back to the full-precision weights to learn the ternary assignments.
  * `gradient2`: Propagates back to the scale factors $W_p$ and $W_n$ to learn the optimal ternary values.

The quantization function is: 

$$\tilde{w}_i = \begin{cases} W_p & : w_i > \Delta_l \\ 0 & : \lvert w_i\rvert \le \Delta_l \\ -W_n & : w_i < -\Delta_l \end{cases}$$  

where the threshold $\Delta_l$ is a hyperparameter: $\Delta_l = t \cdot \max(\lvert w\rvert); t \in [0,1]$.

### DoReFa-Net

DoReFa-Net is a comprehensive framework for training networks with arbitrary bit widths for weights ($W$), activations ($A$), and gradients ($G$). It uses deterministic quantization for weights and activations and stochastic quantization for gradients. It provides a thorough treatment of the **Straight-Through Estimator (STE)**, a key technique for backpropagating gradients through non-differentiable quantization functions. Experimental results on AlexNet show the performance for various W-A-G configurations, highlighting the importance of gradient precision.

### LQ-Nets (Learned Quantization Networks)

LQ-Nets introduce a **learnable quantizer** that creates data-adaptive, non-uniform quantization levels. This can significantly reduce quantization error compared to uniform methods.

The key idea is to represent a quantized number not with a fixed-power-of-2 basis, but as a linear combination of a trainable basis vector $v \in \mathbb{R}^K$:

$$q_l = v^T b_l$$

where $b_l$ is a binary coding vector. By making $v$ a trainable parameter, the network learns the optimal placement of quantization levels for its specific weight distribution. This combines the benefits of uniform quantization (efficiency) with non-uniform quantization (accuracy). Diagrams illustrate how a 2-bit and 3-bit learned basis can create a non-uniform staircase function of quantization levels.

## 1.8 Hardware-Software Interplay: Architectures and Methods

The true benefits of quantization are realized when the algorithm is co-designed with the target hardware in mind.

### HW Excursion: Bit-Serial Multiplication

For hardware designed to handle arbitrary precision, **bit-serial computation** is a viable option. Instead of a parallel multiplier, operations are serialized over the bits of the operands.

A multiplication $c = a \cdot b$ can be decomposed into bit-level operations. If $a$ and $b$ are $N$-bit and $M$-bit fixed-point integers respectively, the product is: 

$$c = a \cdot b = \sum_{n=1}^{N} \sum_{m=1}^{M} 2^{n+m} \cdot \text{popc}(\text{and}(a_n, b_m))$$

The complexity is $O(NM)$, directly proportional to the bit widths. While a single bit-serial operation has high latency, its simple logic is suitable for massive parallelism, enabling competitive throughput. A graph comparing operand bit width to performance improvements shows that for latency with bit-serial logic, the improvement factor scales exponentially as bit width decreases.

### DeepChip's Reduce-and-Scale (RAS) Quantization

The **DeepChip** project focuses on model compression for resource-constrained devices like mobile **ARM** processors. Its **Reduce-and-Scale (RAS)** method is a prime example of HW-aware quantization.

**RAS combines several techniques:**
1. **Weight Quantization:** Uses a TTQ-like approach to quantize weights to ternary values $\lbrace -W_n, 0, W_p\rbrace$. The scale factors $W_p$ and $W_n$ are independent, asymmetric, and trained via SGD.
2. **Activation Quantization:** Activations are first bounded using a **Bounded ReLU** function ($a' = \text{clip}(a, 0, 1)$) and then quantized to a $k$-bit fixed-point representation, similar to DoReFa-Net.
3. **Space-Efficient Data Structures:** Instead of storing the full ternary weight matrix, RAS uses a **parameter converter** to create a compressed representation. This involves run-length encoding principles, storing only the signs and the distances (indices) between non-zero values. This reduces cardinality and is amenable to further compression like Huffman coding.
4. **Efficient Operator Library:** The core computation is reformulated to avoid costly multiplications. The output is calculated by summing the relevant activations and performing only two multiplications by the scale factors $W_p$ and $W_n$ per output channel.

$$c = W_p^l \cdot \sum_{i \in i_p^l} a_i + W_n^l \cdot \sum_{i \in i_n^l} a_i$$

This leverages the fact that integer additions are significantly cheaper and more energy-efficient than multiplications on typical processors (e.g., on an ARM chip, an int16 ADD is ~2x faster and >30x more energy-efficient than an int16 FMA).

Results on AlexNet/ImageNet show that DeepChip's method achieves higher accuracy (79.0% Top-5) and a smaller memory footprint (25 MB) than a baseline BNN or standard INT8 quantization, while maintaining a competitive inference rate.

## 1.9 Common Pitfalls and Best Practices

Applying quantization effectively requires avoiding common mistakes.

**Common Pitfalls:**
* **Taxonomy Confusion:** Incorrectly assuming PTQ is always weight-only.
* **Unrepresentative Calibration Data:** Using a calibration set that doesn't reflect real-world deployment conditions (lighting, sensors, etc.).
* **Blind Min/Max for Activations:** Allowing outliers to inflate the quantization range, which wastes resolution for the majority of values.
* **Ignoring Signed/Unsigned:** Failing to use unsigned integers for non-negative data like ReLU activations.
* **Wrong Granularity Default:** Using per-tensor quantization for CNN weights, where per-channel is a much safer and more robust baseline.
* **Ignoring Sensitive Layers:** Quantizing all layers uniformly, when the first and last layers often require higher precision.
* **Not Measuring the Right Signals:** Failing to analyze activation histograms, saturation rates, and layer-wise error to diagnose issues.

**Rule of Thumb:** Start with a simple, robust baseline: uniform **INT8 PTQ**, with **per-channel** granularity for weights and **percentile-based** calibration for activations. Only escalate to finer granularity, mixed precision, or full QAT if this baseline fails.

## 1.10 Automated Search for Compression Parameters

The vast design space of compression (pruning ratios, bit widths per layer, quantization schemes) makes manual tuning difficult. Automated approaches use search algorithms to find the optimal compression policy for a given hardware target.

* **GALEN:** Combines quantization and pruning by using reinforcement learning (RL) to predict a compression policy. It performs a layer-wise sensitivity analysis and incorporates real-world hardware latency measurements, going beyond simple metrics like FLOPs or BOPs.
* **HAQ (Hardware-Aware Automated Quantization):** Also uses RL to find a mixed-precision quantization policy (2-8 bits) under a latency budget, which is approximated using a lookup table.

These methods represent the frontier of model compression, where hardware expertise is encoded into an automated algorithm that can navigate the complex trade-offs between accuracy, latency, and model size.


















# Unsafe Optimization 3

This chapter bridges the critical gap between the theoretical promise of model compression and the practical realities of hardware performance. We will dissect the concept of sparsity—the introduction of zero-valued weights into a neural network—and explore why simply increasing the number of zeros often fails to deliver corresponding speedups in inference. The central theme is the HW-ML Interplay: the best optimization is not the one with the highest theoretical compression, but the one your hardware can execute efficiently.

## The Sparsity-Speed Fallacy

A common misconception in machine learning optimization is that sparsity directly translates to speed. For instance, if a model has 80% of its weights pruned to zero, one might intuitively expect a 5x speedup. However, the practical reality is often closer to 1x—no speedup at all—without substantial, hardware-aware effort.

This discrepancy arises because modern processors and ML frameworks are highly optimized for dense, structured computations like GEMM (General Matrix-Matrix Multiplication). The performance of a sparse model depends entirely on how that sparsity is represented and executed.

There are three primary approaches to handling sparsity, each with vastly different performance implications:

1. **Mask-Only Sparsity:** This is the most common method within ML frameworks like PyTorch. A binary mask is applied to the weight tensor, effectively zeroing out certain values during computation (e.g., $y = x \cdot (W ⊙ M)$). However, the underlying weight tensor W remains dense in memory, and the computation is still performed by a dense GEMM kernel. The hardware sees the same tensor shapes and executes the same schedule, resulting in little to no speedup.
2. **Structured Pruning (Rewiring):** This method removes entire structural components of the network, such as neurons, attention heads, or convolutional channels. This physically alters the model's architecture, resulting in smaller weight matrices and activation tensors. These smaller tensors can then be processed by standard, efficient dense kernels, leading to real, measurable speedups.
3. **Hardware-Structured Sparsity:** This advanced technique enforces a specific, hardware-friendly sparsity pattern (e.g., N:M sparsity, where N out of M consecutive weights are non-zero). The weights are then packed into a compact format, and a specialized sparse kernel that understands this pattern is used for computation. This provides significant speedups but requires direct hardware support.

The fundamental reason sparsity does not equal speed is that computational work is not skipped unless the kernel is explicitly designed to do so. Overheads associated with sparse formats—such as storing indices, managing indirection, and load imbalance across parallel processing units—can easily negate any savings from reduced computations. Furthermore, in memory-bound scenarios, where performance is limited by data movement rather than computation, removing floating-point operations (FLOPs) provides no benefit.

## 1.2 Understanding Sparsity Formats and Granularity

The effectiveness of pruning is deeply tied to its granularity, which dictates the scale at which weights are removed. This choice has direct consequences for both model accuracy and hardware efficiency.

* **Fine-Grained (Unstructured) Pruning:** Removes individual weights, offering the highest potential for preserving accuracy. However, the resulting random sparsity pattern is difficult for massively parallel processors like GPUs to exploit. These processors thrive on structured, predictable computation, and the overhead of skipping individual zero-valued weights often outweighs the benefit. See performance bugs for GPUs such as memory coalescing, branch divergence, vectorization for CPUs
* **Coarse-Grained (Structured) Pruning:** Removes groups of weights (e.g., entire rows, columns, or channels). This structure is highly compatible with parallel hardware, avoiding performance issues like memory access misalignment (coalescing) and branch divergence. This is the key to achieving practical speedups on modern CPUs and GPUs. Coarse-grained pruning is fastest/most effective on processors

### Compressed Sparse Row (CSR) format

To handle unstructured sparsity, specialized data formats are required. A common example is the **Compressed Sparse Row (CSR) format**.

Glossary: Compressed Sparse Row (CSR) A format for storing sparse matrices that avoids storing zero elements. It uses three arrays:
1. **Data Array ($d$):** A flat array containing all non-zero values.
2. **Column Index ($i$):** An array storing the column index for each value in $d$.
3. **Row Pointer ($r$):** An array of size $M+1$ (for $M$ rows) where $\text{r}[k]$ stores the index in $d$ where row $k$ begins.

Consider the following dense 4x4 matrix D with 8 non-zero elements: 

$$D = [[0, 5, 3, 0], [6, 1, 0, 4], [0, 0, 0, 0], [2, 0, 1, 4]]$$

In dense format, this requires 4 × 4 = 16 units of storage. In CSR format, it would be represented as:
* $d \in \mathbb{R}^8 = (5, 3, 6, 1, 4, 2, 1, 4)$
* $i \in \mathbb{R}^8 = (1, 2, 0, 1, 3, 0, 2, 3)$
* $r \in \mathbb{R}^5 = (0, 2, 5, 5, 8)$

The CSR representation requires 8 (data) + 8 (indices) + 5 (pointers) = 21 units of storage. In this case, the overhead from metadata ($i$ and $r$) makes the sparse format larger than the dense one. Moreover, accessing elements requires indirect, data-dependent lookups, which is inefficient on parallel hardware.

### Block Sparse Row (BSR) format

A compromise between dense and fully unstructured sparse formats is the **Block-Sparse Row (BSR) format**:
* BSR divides the matrix into equally sized dense blocks $o\times p$. 
* Each block can either be fully zero or contain nonzero entries. 
* The format is similar to CSR, but $r$ and $i$ refer to blocks, and $d$ is now a list of arrays (each of size $o\times p$). 

This approach amortizes the metadata overhead over entire blocks, improving computational regularity. Let's consider the same example:

$$D = [[0, 5, 3, 0], [6, 1, 0, 4], [0, 0, 0, 0], [2, 0, 1, 4]]$$

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/bsr_block_division_examples.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>Examples of BSR block size</figcaption>
</figure>

## The Economics of Sparsity: A Break-Even Model

To formalize the trade-offs, we can model the time taken for a dense vs. a sparse operation. The total time is determined by the maximum of either the compute time or the memory access time.

The execution time for a dense operation ($T_\text{dense}$) can be modeled as: $T_\text{dense} \approx \max\left( \frac{F}{P}, \frac{B}{BW} \right)$
The execution time for a sparse operation ($T_\text{sparse}$) is more complex: $T_\text{sparse} \approx \max\left( \frac{(1-s)F}{P_\text{eff}}, \frac{B_\text{sparse}}{BW} \right) + T_\text{overhead}$

| Term        | Description |
|:-----------|:------------|
| $s$          | Sparsity (fraction of zeros). |
| $F$          | Total FLOPs for the dense operation. |
| $P$          | Peak theoretical throughput of the hardware (FLOPs/sec). |
| $P_\text{eff}$      | Effective sparse throughput, which is often much lower than $P$. |
| $B$          | Total bytes moved for the dense operation. |
| $BW$         | Memory bandwidth (bytes/sec). |
| $B_\text{sparse}$   | Bytes moved for the sparse operation, including metadata ($B_\text{sparse} = B_\text{activations} + B_\text{non-zero-weights} + B_\text{metadata}$). |
| $T_\text{overhead}$ | Time cost for packing, format conversion, and handling load imbalance. |


This model reveals critical insights into the HW-ML Interplay:
* **Memory-Bound Regimes:** If the operation is limited by memory bandwidth ($B$/$BW$ is the dominant term), reducing FLOPs ($F$) through sparsity may have no effect on latency.
* **Unstructured Sparsity Overheads:** Unstructured sparsity often hurts performance by lowering $P_\text{eff}$ (due to irregular computation) and adding significant metadata $B_\text{meta}$ and overhead $T_\text{overhead}$. This shifts the break-even point—the sparsity level at which $T_\text{sparse}$ becomes less than $T_\text{dense}$—to very high levels, often over 90%.
* **Structured Sparsity Advantage:** Structured approaches (like block sparsity or N:M patterns) reduce metadata and overhead, leading to a higher $P_\text{eff}$. This allows them to achieve a speedup over dense operations at much lower sparsity levels.

A diagram below shows this relationship clearly. A plot with Sparsity on the x-axis and Speedup on the y-axis depicts two curves. The curve for "Unstructured" sparsity remains flat at 1.0 (no speedup) until a high sparsity of ~85%, after which it begins to rise. In contrast, the curve for "Structured" sparsity begins to show a speedup at a much lower sparsity of ~50%, demonstrating its superior hardware efficiency.

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/speedup_vs_sparsity.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>Speedup vs sparsity: break-even effect</figcaption>
</figure>

## Practical Pruning Methodologies in PyTorch

### Unstructured Pruning

This technique removes individual weights based on their magnitude. PyTorch's `torch.nn.utils.prune` module provides a simple interface for this.
A visual representation of a weight matrix before and after $30%$ unstructured pruning shows a random, salt-and-pepper pattern of zeroed-out (black) elements in the pruned matrix.

Unstructured pruning of a fully-connected layer:
```python
import torch.nn as nn
import torch.nn.utils.prune as prune

fc = nn.Linear(in_features=10, out_features=6)

# Prune 30% of the weights with the smallest L1 magnitude
prune.l1_unstructured(fc, name='weight', amount=0.3)
```

This applies a `weight_mask` to the layer. To make the change permanent (though still stored as a dense tensor with zeros), one would call `prune.remove(fc, 'weight')`.

### Structured Pruning

This method removes entire groups of weights. The importance of a group (e.g., a neuron or a filter) is typically determined by calculating a norm over its weights. The choice between L1 and L2 norm can influence which structures are pruned.

* **L1 Norm:** $\|w\|_1 = \sum_j \lvert w_j \rvert$. Each weight contributes linearly. This norm is less sensitive to a single large weight within a group.
* **L2 Norm (Squared):** $\|w\|_2 = \sum_j w_j^2$. Larger weights have a disproportionately greater impact on the norm. A single large weight can "save" its group from being pruned, even if other weights are small.

Consider two weight vectors, $a = [3, 0, 0, 0]$ and $b = [2, 1, 0, 0]$.

* **Their L1 norms are equal:** $\|a\|_1 = 3$ and $\|b\|_1 = 3$.
* **Their squared L2 norms differ:** $\|a\|_2^2 = 9$ and $\|b\|_2^2 = 5$. The L2 norm prioritizes vector a due to its single large-magnitude weight.

Visualizations of L1-pruned vs. L2-pruned weight matrices show distinct differences. With output neuron pruning, entire rows are zeroed out. The specific rows chosen can differ between L1 and L2-norm criteria, reflecting their different sensitivities to weight distributions.

```python
# Structured pruning of an FC layer's output neurons (dim=0)
prune.ln_structured(fc, name='weight', amount=0.3, n=1, dim=0) # n=1 for L1-norm

# Structured pruning of an FC layer's input features (dim=1)
prune.ln_structured(fc, name='weight', amount=0.3, n=1, dim=1)

# Structured pruning of a Conv layer's output channels (dim=0)
conv = nn.Conv2d(16, 32, kernel_size=3)
prune.ln_structured(conv, name='weight', amount=0.3, n=1, dim=0)
```

Note: Pruning the last fully-connected (FC) layer's outputs or the first FC layer's inputs can be problematic as it changes the model's output or input dimensions, respectively.

### Threshold-Based Pruning

Instead of removing a fixed fraction of weights, thresholding removes all weights whose magnitude falls below a certain value. This can be implemented in PyTorch by creating a custom mask.

```python
# Custom pruning based on an absolute magnitude threshold
threshold = 0.15
mask = (fc.weight.data.abs() >= threshold) # Create a boolean mask

# Apply the custom mask to the layer
prune.custom_from_mask(fc, name='weight', mask=mask)
```

This approach can be extended to structured pruning by first aggregating weight magnitudes (e.g., L2 norm of each convolutional filter), applying a threshold to these aggregate scores, and then creating a mask that zeros out all weights belonging to the low-magnitude structures.

## The Path to Real Speedup: Deployment Strategies

As established, simply using PyTorch's default pruning utilities does not yield a speedup. This is because:

1. **Dense Kernels are Used:** The backend still invokes a dense GEMM or conv kernel, as the tensor shapes are unchanged. The operation `weight = weight_orig * weight_mask` is an element-wise multiplication that occurs before the main computation.
2. **No Sparse Representation:** A mask is not a true sparse format like CSR or BSR. The hardware has no information to skip zero-valued elements.
3. **`prune.remove` is Deceptive:** This function finalizes the pruning by creating a new dense weight tensor that explicitly contains the zeros, which is then passed to the dense kernel.

To achieve actual performance gains, one must follow a hardware-aware deployment path.

## Deployment Path	Description & How it Works	Best For	Key Trade-offs

* **Mask-Only**
  * Uses `weight_mask` to zero out weights but runs on dense kernels. The operation is $y = x \cdot (W ⊙ M)$.	Research, ablations, sensitivity analysis, exploring pruning schedules.	No speedup. Must be explicitly reported as running on dense kernels.
    * **Mask:** $y = x \cdot (W ⊙ M)$
    * **Dense:** $y = x \cdot W$
* **Structured Rewiring**	
  * Physically removes entire channels, neurons, or heads. This reduces the tensor dimensions, creating a smaller, dense model.	Achieving portable latency/throughput gains on general-purpose hardware (CPUs, GPUs).	Requires changes to the network architecture. May need careful fine-tuning and plumbing of tensor shapes.
* **Hardware Pattern**	
  * Enforces a specific pattern (e.g., N:M), packs weights into a special format, and uses specialized sparse kernels.	Maximizing speed on accelerators with dedicated structured sparsity support.	Pattern constraints can impact accuracy. Incurs packing/format conversion overhead. Limited operator coverage.

## Case Study: NVIDIA's 2:4 Structured Sparsity

A prime example of the "Hardware Pattern" path is NVIDIA's 2:4 structured sparsity, supported by Ampere and later GPU architectures.

#TODO: what if we have less or more that eactly two non-zero weights or zero weights in a group that does not allow us to separate the matrix in such groups or the weights configurations does not allow us that all rows in the row pointer array must have the same number of blocks?

* **The Pattern:** In every contiguous group of four weights, a maximum of two can be non-zero.
* **The Mechanism:** Models are trained or fine-tuned to adhere to this constraint. The sparse weights are then compressed, storing only the non-zero data values and their corresponding indices. The diagram shows this compression process, where a sparse weight matrix is converted into a compact representation before the dot product.
* **The Hardware:** NVIDIA's Sparse Tensor Cores (SPTCs) are designed to process this compressed data format at up to 2x the speed of dense Tensor Cores.
* **The Format:** The underlying storage format is a variant like Blocked-Ellpack (Blocked-ELL), which, similar to CSR/BSR, is optimized for regular memory access patterns on the GPU.

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/NVIDIASparsityPattern.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>NVIDIA’s 2:4 Sparsity Pattern</figcaption>
</figure>

## Benchmarking and Measurement: A Practical Guide

Accurate and honest performance measurement is non-negotiable in embedded ML. As the German adage says, "Wer misst, misst Mist" (He who measures, measures rubbish). Rigorous benchmarking is essential to avoid misleading results.

### What to Report (Minimum Set)

* **Quality:** Accuracy/loss, with the evaluation protocol specified.
* **Latency:** p50 (median) and p90 (90th percentile) latency in milliseconds for a fixed batch size and input shape.
* **Throughput:** Samples per second or tokens per second.
* **Memory:** Peak GPU memory usage.
* **Energy (Optional):** Joules per inference or per training step.

It is crucial to be explicit about the scope of the timing:

* **Kernel-only:** Timing just the matmul or conv operation.
* **Layer:** Timing the full layer forward pass.
* **End-to-end:** Timing the entire inference pipeline, including data transfers, pre/post-processing, and any format conversions (e.g., packing for sparse execution). Only an end-to-end measurement under a fixed, stated workload can be trusted.

### Measurement Protocol (Rules of Thumb)

1. **Fix the Environment:** Use the same hardware, data type (`dtype`), batch size, and input shape for all comparisons.
2. **Warmup:** Run 20–100 iterations before starting measurements to let the GPU reach a stable state.
3. **Measure Many Iterations:** Average results over 200–1000 iterations for statistical stability.
4. **Synchronize GPU:** On GPUs, kernel launches are asynchronous. Wrap timers with a synchronization call (e.g., `torch.cuda.synchronize()`) to ensure the measurement captures the full execution time.
  * `torch.cuda.synchronize()`: wait for all kernels in all streams on a CUDA device to complete.
5. **Report Variability:** Report mean and standard deviation, or p50/p90 percentiles, not just a single number.

### Common Measurement Traps

* **No GPU Synchronization:** This times only the kernel launch, not its execution, leading to artificially low latency numbers.
* **Comparing Different Workloads:** Comparing results with different batch sizes, input shapes, or data types is invalid.
* **Ignoring Overheads:** Timing only the matrix multiplication while ignoring the cost of packing data into a sparse format.
* **Ignoring Memory-Bound Effects:** Assuming fewer FLOPs automatically means less time, which is false in memory-bound scenarios.

## Case Study: Pruning a Simple CNN on MNIST

To demonstrate these concepts, a simple CNN is trained on the MNIST dataset.

```python
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
```


With standard training (5 epochs, Adam optimizer, lr=0.001), the model achieves a baseline accuracy of 99.02%.

| Scenario                         | Sparsity | Accuracy |
|:---------------------------------|:--------:|:--------:|
| Baseline                         |   0%     |  99.02%  |
| Pruning (No Fine-Tuning)         |  49.96%  |  90.14%  |
| Pruning (1 Epoch Fine-Tuning)    |  49.96%  |  98.91%  |

Pruning nearly 50% of the weights without fine-tuning causes a significant accuracy drop of ~9%. However, just one epoch of fine-tuning (FT) restores accuracy to near-baseline levels. Histograms of weight distributions show that after pruning, many weights are clustered at zero. Fine-tuning allows the remaining weights to adjust and recover their original distribution shape, restoring the model's representational capacity.

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/weights_distribution_pruning_ft.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>Weights Distribution</figcaption>
</figure>

A comprehensive plot comparing unstructured, L1, L2, and threshold pruning with varying epochs of fine-tuning across a spectrum of sparsities shows that fine-tuning is universally critical. All methods experience a sharp accuracy drop-off as sparsity increases, but fine-tuning consistently pushes this "cliff" to higher sparsity levels, enabling more aggressive compression without sacrificing performance.

## Strategic Pruning: Layer-Wise Sparsity Allocation

Why uniform sparsity is rarely optimal:
* **Layers differ in sensitivity (some are brittle, some redundant)**
* Layers differ in compute share (**prune where FLOPs/time actually are**)
* Layers differ in kernel/shape constraints (**some pruning creates real size reduction, some doesn’t**)

Applying a uniform sparsity ratio across all layers is rarely optimal. Different layers have varying degrees of redundancy and sensitivity to pruning. A more effective strategy is to allocate sparsity non-uniformly based on layer characteristics:
1. **Sensitivity-Based Allocation (Quality-Driven):** The goal is to maximize accuracy for a given global sparsity budget. This involves measuring the sensitivity of each layer (e.g., by observing the accuracy drop when pruning it slightly). More sparsity is allocated to robust, redundant layers, while sensitive, brittle layers are pruned less aggressively.
2. **Compute-Aware Allocation (Efficiency-Driven):** The goal is to maximize latency or energy savings for a target accuracy. This strategy prioritizes pruning layers that are computationally expensive (*high FLOPs or long execution time*) and where pruning leads to tangible changes in tensor shapes or enables specialized kernels.

A practical rule of thumb is to **prune large fully-connected layers heavily, as they are often over-parameterized**, and to be **more cautious with early convolutional layers, which tend to learn fundamental features**.

## Advanced Techniques: Learning Sparsity with PSP

Traditional pruning methods are often a multi-stage process: train a dense model, apply a heuristic criterion to prune, and then fine-tune to recover accuracy. **Parameterized Structured Pruning (PSP)** offers a more integrated approach by learning the sparsity pattern during training.

The core idea of PSP is to parametrize the pruning decision for entire structures (e.g., weights, columns, channels). The different levels of structure are visualized in a diagram showing (a) individual weights, (b) columns, (c) channels, (d) shapes, and (e) layers.

For each structural sub-tensor $w_i$, a learnable parameter $\alpha_i$ is introduced. During the forward pass, the sub-tensor is dynamically masked: 

$$q_i = w_i \cdot v_i(\alpha_i) $$

where $v_i(\alpha_i)$ is a thresholding function: 

$$v_i(\alpha_i) = \begin{cases} 0 & \text{if } \lvert \alpha_i\rvert < \epsilon \\ \alpha_i & \text{if } \lvert \alpha_i\rvert \geq \epsilon \end{cases}$$

Since this thresholding is non-differentiable, the Straight-Through Estimator (STE) is used to approximate its gradient during backpropagation. The gradients update the $\alpha_i$ parameters, allowing the network to learn which structures are important. A regularization term (L1 or L2) is applied to the $\alpha_i$ parameters to encourage sparsity.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/embedded-machine-learning/PSP1.png' | relative_url }}" alt="GPU global memory" loading="lazy">
    <!-- <figcaption>GK110 architecture</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/embedded-machine-learning/PSP2.png' | relative_url }}" alt="GPU shared memory" loading="lazy">
    <!-- <figcaption>GK110 SM (Streaming Multiprocessor)</figcaption> -->
  </figure>
</div>

Update rules for $\alpha_i$, using gradient descent with momentum $\mu$ and a learning rate $\eta$, can incorporate L2 weight decay ($-\lambda \eta \cdot \alpha_i(t)$) or L1 regularization ($-\lambda \eta \cdot \text{sign}(\alpha_i(t))$). L2 regularization tends to produce better-separated weight distributions and has shown superior performance.

A validation error plot for ResNet-56 on CIFAR-10 with column pruning demonstrates PSP's effectiveness. The plot shows that "PSP (weight decay)" achieves a lower validation error than "L1 norm" pruning across a wide range of sparsity levels, indicating it finds a better trade-off between accuracy and compression.

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/PSP_plot.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>NVIDIA’s 2:4 Sparsity Pattern</figcaption>
</figure>

## Conclusion: A Systems-Level View of Compression

Pruning and other model compression techniques like quantization are not purely algorithmic problems; they are optimization problems under a strict systems constraint. The effectiveness of any compression method is determined by the interplay between the algorithm, the model architecture, and the target hardware.

The optimal strategy depends entirely on the end goal:
* If the goal is minimum model size for storage or transmission, unstructured pruning is sufficient, provided the model is stored and loaded in an efficient sparse format.
* If the goal is portable latency reduction on general-purpose hardware, structured pruning that physically rewires the network to create smaller, dense tensors is the most reliable path.
* If the goal is maximum performance on specialized hardware, one must enforce kernel-supported patterns like N:M sparsity and leverage dedicated hardware acceleration.

Ultimately, successful model compression for embedded systems requires a holistic view, recognizing that performance is a function of the entire tuple: {data, neural architecture, hardware architecture}.

<figure>
  <img src="{{ '/assets/images/notes/embedded-machine-learning/pruning_landscape.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>Pruning Landscape</figcaption>
</figure>

---

(lecture 10)
Chapter 1: The Optimization Imperative in Embedded ML

The core challenge in Embedded Machine Learning is bridging the gap between computationally intensive ML algorithms and the severe resource constraints of edge devices. This involves a deep understanding of how both software and hardware can be co-designed to achieve efficiency in terms of energy, memory, and latency.

The "Simplicity Wall" of Deep Neural Networks

Deep Neural Networks (DNNs), despite their complex behavior, are computationally uniform. They spend the vast majority of their execution time performing matrix multiplications and convolutions, which are fundamentally composed of Multiply-Accumulate (MAC) operations. This phenomenon is known as the "Simplicity Wall." This computational regularity, characterized by static loop-trip counts and minimal control overhead, makes DNNs highly predictable and an ideal target for hardware acceleration. It allows architects to design specialized processors that excel at this one task, leading to significant performance and efficiency gains.

The Energy Cost of Computation and Data Movement

A foundational principle in embedded systems design is that moving data is far more expensive than computing on it. This is especially true when data has to be fetched from off-chip memory. As documented by M. Horowitz in "Computing's energy problem," the energy cost increases exponentially as data moves further from the processing unit.

The following table, derived from this research, illustrates the orders-of-magnitude difference in energy consumption for various operations and memory accesses.

Operation / Memory Access	Energy (pJ)	Location
8-bit Integer MULT	0.2	On-Die
32-bit Float MULT	3.7	On-Die
8KB SRAM Read (64 bit)	10	On-Die
32KB SRAM Read (64 bit)	20	On-Die
1MB SRAM Read (64 bit)	100	On-Die
DRAM Read	2,000	Off-Die

This stark reality highlights a primary goal of Embedded ML optimization: minimize data movement. The most effective hardware architectures are those that maximize data reuse, keeping operands and intermediate results in fast, low-power SRAM (Static Random-Access Memory) close to the compute units, and avoiding costly trips to off-chip DRAM (Dynamic Random-Access Memory).

Safe vs. Unsafe Optimizations

Optimizations in ML can be categorized into two main types based on their impact on model accuracy.

Safe Optimizations

Safe optimizations are techniques that improve performance without any impact on the model's accuracy. These are purely architectural or data-layout improvements that exploit the structure of the ML workload.

* Shorter communication paths: Designing hardware to keep compute and local memory physically close.
* Data reuse: Structuring computation to use each piece of data loaded from memory multiple times before discarding it.

These principles directly lead to the development of dedicated architectures, such as array-based processors, which are designed from the ground up to minimize data transfer volume.

Unsafe Optimizations

Unsafe optimizations are techniques that can potentially alter the model's accuracy. They achieve efficiency by modifying the model itself or the precision of its calculations. While they risk a drop in accuracy, the performance gains can be substantial, making them essential for deployment on highly constrained devices.

* Pruning: Reducing the model size and operation count by removing redundant weights or connections.
* Quantization: Reducing the precision of operations and operands. This is a powerful technique where, for instance, 32-bit floating-point numbers are converted to 8-bit fixed-point integers (Quantization) or even single bits (Binarization). This dramatically reduces memory footprint and energy consumption, as lower-precision arithmetic is much cheaper.

Chapter 2: The Rise of Array Processors for ML Acceleration

The "Simplicity Wall" of DNNs has driven a paradigm shift in processor design, moving away from general-purpose cores towards highly specialized, parallel architectures. This trend is visible across the industry, with a plethora of ML accelerators being developed in both research and commercial sectors.

Processor Specialization Driven by ML

The demand for efficient ML execution has led to an explosion of specialized hardware. A survey of ML accelerators reveals a clear trend: specialization leads to greater efficiency. A plot of speed (GOP/s) versus power (W) shows that ASICs (Application-Specific Integrated Circuits) consistently achieve a higher ratio of Tera-Operations per second per Watt (TOPs/W) compared to more general-purpose platforms like FPGAs and CPUs.

This landscape includes a wide range of devices:

* Research Accelerators: Eyeriss, EIE/ESE, NeuroCube, DianNao
* Commercial Products: Google TPU/Edge TPU, NVIDIA TensorCores, Intel Nervana, Apple Neural Engine, ARM Trillium, GraphCore IPU.

One notable research architecture, Neurocube, exemplifies the focus on memory. It uses 3D die stacking with Through-Silicon Vias (TSVs) to place multiple DRAM dies directly on top of a logic die. This vertically integrated design drastically shortens the path between memory and computation, directly addressing the high energy cost of data movement.

The Systolic Array: A Paradigm for Data Reuse

The most successful architectural pattern to emerge for DNN acceleration is the systolic array. This architecture replaces a single complex processing element (PE) with an array of many simpler, interconnected PEs.

Key Principle: Data is "pumped" through the array of PEs in a rhythmic, systolic fashion, similar to how blood is pumped by the heart. Each PE performs a small computation (typically a MAC operation) on the data it receives and then passes the result to its neighbor. This structure allows for immense data reuse, achieving high throughput without proportionally increasing memory bandwidth requirements. For example, replacing a single PE capable of 5 MOP/s with a 4-PE array can achieve 20 MOP/s with the same memory interface.

A visual inspection of a modern ML accelerator chip, such as the one developed by the Eyeriss project at MIT, reveals this regularity. The die is dominated by a large, grid-like Spatial PE Array, with a significant portion of the remaining area dedicated to an On-Chip Buffer to feed this array.

How Systolic Arrays Perform Matrix Multiplication

Consider the matrix multiplication C = A \cdot B. In a systolic array, the elements of matrices A and B are streamed into the array from different directions (e.g., A from the left, B from the top). As the data waves intersect within the array, each PE accumulates a partial product. For instance, at a specific time step t, a PE might multiply a_ij and b_jk. In the next time step, a_ij moves to the next PE in its row, and b_jk moves to the next PE in its column. The partial sums for the output matrix C remain stationary within the PEs, accumulating values over time until the computation is complete. This orchestrated flow ensures that each value fetched from memory is used in multiple computations across the array.

Architectural Comparison: Array vs. Multi-Core

Array-based processors represent a fundamentally different design philosophy compared to traditional multi-core vector processors.

Feature	Multi-core Vector Processor	Array-based Processor
Control	Complex control logic per core	Simple, often centralized control
Processing Units	Few, powerful ALUs per core	Many simple ALUs (PEs)
Memory	Deep memory hierarchy (L1/L2/L3 caches)	Flat, large scratchpad memory (On-Chip Buffer)
Register File	Large Register File (RF) per core	Minimal or no RF per PE
Data Flow	Programmer-managed data movement	Implicit, hardware-managed dataflow

This specialization makes array processors incredibly efficient for their target workload (matrix multiplication) but less flexible for general-purpose tasks.

Chapter 3: Case Study: Google's Tensor Processing Unit (TPU)

The Google TPU is one of the first and most prominent production-class ML accelerators. It is essentially a coprocessor designed to offload matrix multiplications from a host CPU, and its design philosophy provides valuable insights into building hardware tailored to real-world ML workloads.

The Roofline Model: Analyzing Performance Bottlenecks

To understand the performance of a given architecture, the Roofline Model is an invaluable tool. It plots the attainable performance of a processor (in GFLOP/s or TOP/s) as a function of an application's operational intensity.

* Operational Intensity (r): The ratio of floating-point operations (FLOPs) to bytes of data moved from memory. It is calculated as r = \frac{[FLOPs]}{[Bytes]}. This metric quantifies how much computation is done per byte of data.
* Peak Compute Performance (f): The theoretical maximum performance of the processor's compute units (the "flat" part of the roofline).
* Peak Memory Bandwidth (m): The maximum rate at which the processor can access memory (the "sloped" part of the roofline).

The attainable performance (a) is given by the formula: a = \min(m \cdot r, f)

An application is compute-bound if its operational intensity is high enough that its performance is limited by f. It is memory-bound if its performance is limited by the sloped line, m \cdot r. The Roofline model visually indicates whether an application's performance would benefit more from faster compute or higher memory bandwidth.

A Roofline plot comparing a Google TPU, an NVIDIA K80 GPU, and an Intel Haswell CPU on various ML workloads (MLP, LSTM, CNN) clearly demonstrates the TPU's advantage. For workloads with high operational intensity, the TPU's "roof" is significantly higher, allowing it to achieve over an order of magnitude more TeraOps/sec than the GPU or CPU. This shows it is exceptionally well-designed for compute-bound ML inference tasks.

TPU v1 Architecture Deep Dive

The architecture of the first-generation TPU (TPU v1) is a direct implementation of the systolic array concept.

A block diagram of the TPU v1 reveals the following data flow:

1. Host Interface: Instructions and data enter from the host system via a PCIe Gen3 x16 Interface.
2. Unified Buffer: Data (primarily activations) is stored in a large on-chip buffer (24 MiB of SRAM). This acts as the local storage to feed the compute engine.
3. Weight Fetcher: Weights are streamed from off-chip DDR3 DRAM into a Weight FIFO buffer.
4. Matrix Multiply Unit: The heart of the TPU is a 256x256 Systolic Array, capable of performing 64K MAC operations per cycle. Activations from the Unified Buffer and weights from the Weight FIFO are fed into this array.
5. Post-processing: The results from the systolic array are passed to Accumulators and then through hardware units for Activation functions (e.g., ReLU), Normalization, and Pooling.
6. The final results are written back to the Unified Buffer and eventually sent back to the host.

The key is the massive internal bandwidth. The Unified Buffer can supply data to the Matrix Multiply Unit at 167 GiB/s, keeping the systolic array fed and achieving high utilization. This architecture is purpose-built to maximize computation while minimizing off-chip memory access, directly aligning with the energy cost principles.

Workload Evolution Across TPU Generations

Google's internal workloads have evolved significantly, and subsequent TPU versions have adapted to these changes. The distribution of model types running on TPUs highlights shifting priorities in the ML field.

DNN Model	TPU v1 (Inference)	TPU v3 (Train & Inf.)	TPU v4 Lite (Inference)	TPU v4 (Training)
MLP/DLRM	61%	27%	25%	24%
RNN	29%	21%	29%	2%
CNN	5%	24%	18%	12%
Transformer	--	21%	28% (BERT)	57% (LLM)

Data for TPU v4 Training is from a 30-day period in Oct 2022. Over 90% of training at Google now runs on TPUs.

The most dramatic shift is the rise of Transformer models, which grew from non-existent on v1 to dominating the v4 training workload (57%). In response, newer TPUs like the v4 have introduced specialized hardware such as the SparseCore for efficiently handling embeddings common in these models and Optical Circuit Switches (OSC) for high-speed interconnects in large-scale training pods.

Chapter 4: Field-Programmable Gate Arrays (FPGA) for Flexible Acceleration

While ASICs like the TPU offer peak performance and efficiency, they lack flexibility. Field-Programmable Gate Arrays (FPGAs) offer a middle ground, providing customizable hardware acceleration with the ability to be reconfigured after manufacturing.

Core Architecture of an FPGA

An FPGA is an integrated circuit containing an array of programmable hardware blocks. Its primary components are:

* Configurable Logic Blocks (CLBs): These are the fundamental building blocks of an FPGA. A typical CLB, or logic cell, contains one or more Look-Up Tables (LUTs), a Full Adder (FA), and a D-type Flip-Flop (DFF). LUTs can be programmed to implement any boolean logic function of a few inputs (e.g., a 4-input LUT can implement any function of 4 variables).
* Programmable Interconnect: A rich network of programmable wiring that allows CLBs and other resources to be connected in arbitrary ways.
* Specialized Modules: Modern FPGAs also include hardened, high-performance blocks for common tasks:
  * BRAM (Block RAM): Dedicated blocks of on-chip SRAM for efficient data storage.
  * DSP (Digital Signal Processor): Specialized slices designed to perform arithmetic operations, like multiplication and accumulation, very efficiently. These are critical for ML workloads.
* I/O Blocks: Interfaces for connecting the FPGA to the outside world (e.g., PCIe, Ethernet, DDR memory).

FPGAs are programmed using Hardware Description Languages (HDLs) like Verilog or VHDL, although High-Level Synthesis (HLS) tools that compile C/C++ into hardware designs are becoming increasingly common.

FPGA vs. GPU: A Comparison for ML Workloads

FPGAs and GPUs are both highly parallel processors, but they have distinct strengths and weaknesses for ML acceleration.

Feature	GPU	FPGA
Concurrency	High concurrency at reduced frequency	High concurrency at reduced frequency
Memory Hierarchy	Flat memory hierarchy (Scratchpad/Shared Memory)	Flat memory hierarchy (BRAM)
Programming Model	Data-parallel kernels (e.g., CUDA)	Partly data-parallel, via HLS or HDL
Latency Tolerance	BSP-like block data transfer	BSP-like block data transfer
Efficiency Driver	Extreme specialization (e.g., TensorCores)	Flexible specialization (custom data paths)
Network-on-Chip (NOC)	Non-blocking	Blocking
3D Die Stacking	Difficult	Possible
Main Benefit	Raw performance	Power efficiency (wattage) and I/O flexibility
Main Drawback	Amount of data movements required	Memory performance can be a bottleneck

The key advantage of an FPGA is its flexibility. It allows designers to create custom data paths and precision for their specific neural network, avoiding the overhead of a fixed instruction set. This is particularly beneficial for highly quantized models.

Chapter 5: FINN: An End-to-End FPGA Deployment Framework

To bridge the gap between high-level ML frameworks and the complexity of FPGA programming, tools like FINN have been developed. FINN is an open-source, end-to-end framework from AMD/Xilinx for deploying quantized neural networks on Xilinx FPGAs.

The FINN Tool Flow

The FINN framework provides a complete pipeline from model training to hardware deployment, composed of three key open-source projects:

1. Brevitas (Training): A PyTorch library for quantization-aware training (QAT). It allows developers to train neural networks while simulating the effects of low-precision arithmetic, ensuring that the model maintains high accuracy after quantization.
2. FINN Compiler (Synthesis): This is the core of the framework. It takes a pretrained, quantized model (in ONNX format) and translates it into a hardware design. It performs a series of automated transformations and optimizations, generates High-Level Synthesis (HLS) C++ code, and drives the synthesis process to create the final hardware bitstream.
3. PYNQ (Deployment): Python on Zynq (PYNQ) is an open-source project that provides a Python-based execution environment on Xilinx FPGAs. It simplifies interacting with the custom-generated accelerator from a host processor, making it easy to integrate into larger applications.

FPGA Execution Models: Loopback vs. Dataflow

The FINN framework supports two primary architectural patterns for executing neural networks on an FPGA.

Option 1: Loopback Architecture

Also known as a "multi-layer offload architecture," this model uses a single, maximally-sized, homogeneous compute array. The network is executed layer-by-layer. For each layer, the weights are loaded from off-chip memory, a batch of input images (or activations from the previous layer) is processed, and the results are written back to off-chip memory.

* Pros: Can support arbitrarily deep networks since weights are not stored permanently on-chip.
* Cons: High traffic to external memory, which can become a performance and energy bottleneck.

Option 2: Dataflow Architecture

This is the original architecture proposed in the FINN paper and is highly efficient for streaming applications. In this model, a custom, pipelined hardware architecture is created for the entire network. Each layer gets its own dedicated, heterogeneously-sized compute array tailored to its specific requirements.

* How it works: Input images are streamed into the pipeline. Intermediate activations are passed directly from one layer's compute array to the next via on-chip FIFOs or BRAM, without ever touching external memory. All model weights are stored in on-chip memory.
* Pros: Extremely high throughput and low latency, as it avoids the off-chip memory bottleneck. Very power efficient.
* Cons: The entire network must fit within the FPGA's on-chip resources (BRAM for weights, logic for compute), which limits the maximum model size.

The dataflow architecture is a prime example of HW-ML Interplay, where the hardware is perfectly tailored to the structure of a specific neural network, leading to massive efficiency gains, particularly for highly quantized networks (< 5 bits) that can leverage the FPGA's integer processing capabilities.

Key Steps in the FINN Compiler

The FINN compiler automates the complex process of hardware generation through a series of steps:

1. Import Model: A pretrained network is imported in the ONNX format, using custom operators to represent quantized layers.
2. Network Preparation & Transformation: A series of "streamlining" passes are performed to prepare the graph for HLS. This includes converting nodes into a hardware-friendly representation and setting performance parameters like parallelism.
3. HLS Code Generation: The compiler generates C++ code from templates using a dedicated HLS library (finn-hlslib). This code describes the behavior of the custom dataflow accelerator.
4. HLS Synthesis & Deployment: The generated code is synthesized using the Vitis HLS tool to produce a hardware bitstream, which can then be deployed to a PYNQ-enabled board like the Xilinx Ultra96.

Chapter 6: Performance Analysis and Conclusion

The ultimate goal of these advanced architectures and frameworks is to deliver high performance within a strict power budget. Analysis using emulators and real-world benchmarks demonstrates the effectiveness of these hardware-aware optimization strategies.

Optimizing Dataflow with the CAMUY Emulator

The performance of a systolic array is highly dependent on its dimensions (height and width). The CAMUY Emulator is a tool used to analyze the performance of different systolic array configurations for various DNN architectures (e.g., AlexNet, VGG-16, ResNet, MobileNetV3).

By simulating the execution of these networks, CAMUY can generate heatmaps showing the normalized data movement cost for different array dimensions. The results consistently show that there is a "sweet spot" for the array shape. For many popular CNNs, square-like or slightly rectangular dimensions (e.g., 128x128, 256x64) result in the lowest data movement cost, while extremely skewed "vector-like" dimensions (e.g., 8x2048, 2048x8) are highly inefficient. This analysis is crucial for designing optimal ASICs or configuring FPGA overlays.

Comparative Benchmark: ResNet on a 5W Power Budget

A performance comparison for a ResNet variant on the CIFAR-10 dataset, constrained to a 5 Watt power budget, powerfully illustrates the trade-offs between different platforms and optimization techniques.

The plot of Test Accuracy vs. Throughput (frames per second) reveals several key insights:

* FPGA (Xilinx Ultra96) with Quantization: Achieves very high accuracy (~95.5%) at a moderate throughput (~500 fps). This highlights the strength of FPGAs in executing low-bit precision models with minimal accuracy loss.
* CPU (ARM Cortex-A57) with Optimizations:
  * Using the Gemmlowp library, it achieves low throughput.
  * With binarization, throughput is high (~1800 fps) but accuracy drops significantly (to ~87%).
  * With reduce-and-scale, it offers a balance between the two.
* GPU (NVIDIA Nano) with Structured Pruning: Shows a clear trade-off curve. Higher throughput can be achieved by increasing pruning, but this comes at the cost of reduced accuracy. It occupies a space between the high-accuracy FPGA and the high-throughput, low-accuracy CPU methods.

This benchmark demonstrates the core principle of Embedded ML: there is no single "best" solution. The optimal choice of hardware (FPGA, GPU, CPU) and optimization strategy (quantization, pruning, binarization) depends entirely on the specific application's requirements for accuracy, throughput, and power consumption. The tight coupling of ML model design and hardware architecture is essential for pushing the boundaries of what is possible on edge devices.

---

Chapter 1: The New Era of Specialized ML Processors

This chapter introduces the rapidly expanding landscape of hardware designed specifically for machine learning and explores the fundamental principle that hardware capabilities often dictate which algorithmic ideas thrive.

The Cambrian Explosion of AI Hardware

The field of machine learning is no longer confined to the domain of software and algorithms. A vast and diverse ecosystem of specialized processors has emerged, each designed to accelerate AI workloads with unprecedented efficiency. This hardware landscape includes a wide array of architectures from academic research projects to commercial silicon deployed at scale.

Notable examples of these specialized processors include:

* Data Center & High Performance: Google TPU, NVIDIA TensorCore, Tesla Dojo (D1), GraphCore IPU, Habana Gaudi 2, Cerebras WSE-2, Groq TSP, Huawei Davinci, Baidu Kunlun.
* Edge & Embedded Devices: Google Edge TPU (Coral), Apple NeuralEngine, ARM Helium, GreenWaves GAP9, Kinara Ara-1, Samsung Neural Processing Unit.
* Research & Novel Architectures: MIT Eyeriss, DianNao and its variants, UC Berkeley Gemmini, GT NeuroCube, LightOn OPU, LightMatter.

This proliferation signifies a fundamental shift in computing: general-purpose architectures are no longer sufficient to meet the demands of modern AI.

The Hardware Lottery Hypothesis

The success of certain machine learning models, particularly Deep Neural Networks (DNNs), is not solely a result of their algorithmic elegance. It is deeply intertwined with the available hardware. This concept is captured by the Hardware Lottery Hypothesis, which posits that "Tooling […] has played a disproportionately large role in deciding which ideas succeed and which fail."

* Embedded ML Pillar (HW-ML Interplay): The dominance of DNNs is a prime example of this hypothesis. DNNs rely heavily on matrix-matrix operations, a task for which Graphics Processing Units (GPUs) are exceptionally well-suited due to their massively parallel architecture. This perfect match between algorithm and hardware accelerated the deep learning revolution.

However, this creates a feedback loop where researchers, often ignoring the underlying hardware, gravitate towards models that perform well on existing systems. This raises a critical question: what if a different type of processor, one excelling at processing large graphs, had been widely available? Perhaps models like probabilistic graphical models, sum-product networks, or graph neural networks would dominate the field today.

The enormous scale of modern models like GPT-3 (175 billion parameters, requiring 800GB of state) and Alphafold-2 (trained on 23TB of data) further underscores the critical dependence on hardware that can efficiently manage such resource demands.

Chapter 2: Why General-Purpose Processors Fall Short for Deep Learning

To understand the need for specialization, we must first analyze why the sophisticated architectures of modern Central Processing Units (CPUs), honed over decades by Moore's Law and Dennard Scaling, are fundamentally ill-suited for the computational patterns of deep learning.

A Look Inside a Modern CPU Microarchitecture

A modern high-performance CPU is a marvel of complexity designed for general-purpose, single-thread performance. A typical microarchitecture, as illustrated in the lecture diagrams, includes an intricate pipeline with numerous components dedicated to optimizing the execution of varied and unpredictable code.

Key features include:

* Instruction Fetch & Branch Prediction: An Instruction Fetch unit retrieves instructions from an Instruction Cache. A sophisticated Branch Predictor guesses the outcome of conditional branches (bne, beq) to keep the pipeline full.
* Decoding and Reordering: Instructions are decoded and placed into an Issue Buffer. A Reorder Buffer and Reservation Stations manage out-of-order execution, allowing the CPU to execute instructions as their data becomes available, not just in their programmed sequence.
* Speculative Execution: The CPU performs speculative execution, executing instructions before it's certain they are needed (e.g., past a predicted branch). It later checks for correctness and discards results if the speculation was wrong.
* Complex Memory Hierarchy: A multi-level cache system (L1, L2, L3 Caches) and DRAM are managed with complex replacement strategies and speculative pre-fetchers to hide memory latency.

These innovations—speculation, out-of-order execution, and deep pipelines—are essential for accelerating programs with complex, data-dependent control flow but represent significant overhead in terms of energy and die area.

The Unique Properties of Deep Neural Network Workloads

In stark contrast to general-purpose code, DNN workloads exhibit a set of highly regular and predictable properties:

1. High Computational Intensity: DNNs are dominated by large linear algebra kernels, primarily matrix multiplications and convolutions. This means there are many arithmetic operations for each memory access.
2. Massive Data-Level Parallelism: The same operation is performed independently on vast amounts of data (e.g., applying a filter to every patch of an image).
3. Associativity: The mathematical properties of these operations provide flexibility in the ordering and scheduling of computations.
4. Predictable Control Flow: During inference, the computational graph is static. There are no data-dependent branches, meaning the entire sequence of operations is known in advance.

Shedding the Architectural Baggage

* Embedded ML Pillar (Resource Constraints & HW-ML Interplay): The predictable and parallel nature of DNNs means that most of the architectural innovations of the last few decades are not only unnecessary but actively detrimental to efficiency.

For a DNN accelerator, there is no need for:

* Speculation and Branch Prediction: Control flow is known ahead of time.
* Complex Cache Agents: Data access patterns can be explicitly managed by a compiler to optimize for locality, rendering complex pre-fetching and replacement policies redundant.
* Out-of-Order (OOO) Execution: The massive available parallelism provides more than enough work to keep functional units busy without reordering instructions.
* Complex Multi-threading (SMT): Data-level parallelism is a more direct and efficient way to achieve high utilization than instruction-level multi-threading.

By eliminating this overhead, specialized processors can dedicate more silicon area and power budget to what truly matters for ML: raw computational units and high-bandwidth on-chip memory.

Chapter 3: The Philosophy of Processor Specialization

Having established the mismatch between general-purpose CPUs and ML workloads, we now explore the principles and benefits of designing specialized hardware tailored specifically for AI.

Benefits of Tailored Hardware Design

Specialization allows architects to rethink the entire processor stack, from the instruction set to the memory system, to achieve maximum efficiency for a target domain.

* Instruction Set Architecture (ISA): Designers can create simple ISAs optimized for tensor and vector operations, reducing the energy and complexity of fetching and decoding instructions. In some cases, instructions can be avoided entirely in favor of configurable finite state machines.
* Data Types: There is no need to adhere to legacy data types. Specialized processors heavily utilize quantization, a process of reducing the numerical precision of model weights and activations (e.g., from 32-bit floating-point to 8-bit fixed-point integers). This drastically reduces memory footprint and energy consumption for both storage and computation.
* Memory Structures: General-purpose caches, designed to handle unpredictable access patterns, can be replaced with software-managed memories like scratchpads, queues, and FIFOs. These structures give the compiler direct control over data placement and movement, enabling optimal data reuse and minimizing power-hungry off-chip DRAM accesses.

The Efficiency-Flexibility Trade-off

Processor design exists on a spectrum between flexibility and efficiency.

* Software-Only: A pure software solution on a general-purpose CPU is highly flexible but offers the lowest performance and efficiency.
* Hardware-Only (ASIC): A fully custom Application-Specific Integrated Circuit (ASIC) offers the highest performance and efficiency but is completely inflexible.
* Co-design: The goal of modern ML accelerators is to find a sweet spot, creating programmable hardware that sacrifices general-purpose flexibility for massive gains in the target ML domain. This co-design approach avoids sacrificing either innovation (flexibility) or performance (efficiency).

A Multi-Layered Approach to Specialization

Optimizing for ML is not a single-step process. It involves a holistic approach that spans algorithms, architecture, and circuits.

Optimization Layer	Technique	Hardware/Software Implication
Algorithms	Data Types (Quantization, Approx. Computing)	Requires architectural support for special data types and non-standard forms of computing.
	Model Sparsity (Pruning, Efficient Coding)	Benefits from architectural support for compression and sparse data formats.
Architecture	Mapping & Scheduling (Compiler)	High-level optimizations like data reuse and scheduling map the algorithm to the hardware.
	Data Movement (Compiler)	The compiler must explicitly manage the memory hierarchy to minimize energy.
Circuits	Energy Efficiency	Techniques like clock gating ensure energy proportionality (no work, no power).
	Novel Technologies	Emerging solutions like analog circuits, memristor arrays, and 3D die stacking push efficiency further.

Example of Extreme Specialization: The FINN Framework

The FINN framework is an example of pushing specialization to its logical extreme.

* Core Idea: It compiles a neural network architecture directly onto the fabric of an FPGA (Field-Programmable Gate Array) using a dataflow execution model.
* HW-ML Interplay: It specifically targets highly quantized networks (often with fewer than 5 bits of precision) to fully leverage the integer-processing capabilities of FPGA logic blocks (CLBs) and on-chip memory (BRAMs).
* Architecture: The framework uses predefined, modular building blocks for different neural network layers (convolution, pooling, fully connected). The diagram shows these units chained together: an input stream flows through a sliding window unit into a Matrix-vector-threshold unit (for convolution), then to a Pool unit, and finally to another Matrix-vector-threshold unit (for the fully connected layer). This creates a physical pipeline on the FPGA that perfectly mirrors the model's structure.

Chapter 4: Architectural Deep Dive: Google Edge TPU

The Google Edge TPU, the accelerator inside Coral devices, is a prime example of a commercial product built on the principles of specialization for edge inference.

System Overview and Architecture

The Edge TPU is an ASIC designed to accelerate Convolutional Neural Networks (CNNs). Its core is a systolic array, a grid of interconnected Processing Elements (PEs) optimized for matrix multiplication.

The architecture diagram reveals a chip organized around this PE array:

* A Controller fetches instructions and manages the flow of data.
* On-chip Memory (Activation, Instruction, Parameter Memory) acts as a high-bandwidth buffer to feed the PEs. This is likely fast SRAM.
* The chip communicates with slower, off-chip DRAM for storing the main model parameters and larger activation maps.
* Each PE in the (nx, ny) grid contains its own local memory (Core Memory) and multiple Compute Lanes, which perform the fundamental Multiply-Accumulate (MAC) operations.

The operational flow involves activations flowing through the array, being multiplied by weights stored in the PEs, and accumulating partial results. These results are then either stored in PE memory for further processing or written back to DRAM.

The Mandatory Development Workflow

The high degree of specialization in the Edge TPU imposes a strict and mandatory workflow for model deployment. Users cannot simply run any ML model; it must be specifically prepared for the hardware.

* Embedded ML Pillar (HW-ML Interplay): The toolchain is designed to transform a standard model into a format the hardware can execute efficiently. The official documentation states: “It supports only TensorFlow Lite models that are fully 8-bit quantized and then compiled specifically for the Edge TPU.”

The workflow is as follows:

1. Train Model: A model is trained in TensorFlow using standard 32-bit floating-point numbers.
2. Quantize: The model undergoes quantization. This can be done post-training or, for better accuracy, through quantization-aware training. The result is a TensorFlow Lite model with 8-bit fixed-point numbers.
3. Compile: The quantized TensorFlow Lite model is passed to the Edge TPU compiler, which generates a .tflite file containing operations optimized for the systolic array.
4. Deploy: The compiled model is deployed to Coral hardware, where it is executed by the Edge TPU runtime.

The diagram of the software stack shows two primary paths for application development:

* Option A (Python): A Python application uses the TensorFlow Lite Python API and optionally the PyCoral API to interact with the Edge TPU runtime (libedgetpu.so).
* Option B (C/C++): A C/C++ application links directly against the Edge TPU runtime API (edgetpu.h) and optionally the libcoral API.

Hardware Specifications and Performance

Analysis of the Edge TPU reveals that its performance is often bound by memory bandwidth, not raw compute power. The compiler uses a technique called "parameter caching" by allocating a scratchpad memory to reduce expensive DRAM accesses, though this benefit diminishes for larger models. Consequently, inference latency is primarily determined by memory bandwidth, with the PE count having less impact.

The following table compares different internal versions of the Edge TPU, highlighting the design trade-offs:

Parameter	V1	V2	V3
Clock Frequency (MHz)	800	1066	1066
# of (X, Y)-PEs	(4, 4)	(4, 4)	(4, 1)
PE Memory	2MB	384kB	2MB
# of Cores per PE	4	1	8
Core Memory	32kB	32kB	8kB
# of Compute Lanes	64	64	32
Mem Bandwidth (GB/s)	17	32	32
Peak TOP/s	26.2	8.73	8.73

Chapter 5: Architectural Deep Dive: MIT Eyeriss and Dataflow Optimization

The MIT Eyeriss project provides a deep dive into one of the most critical aspects of accelerator design: optimizing the dataflow, which is the orchestration of data movement between memory and compute units.

The Primacy of Data Movement

* Embedded ML Pillar (Resource Constraints): In any modern processor, data movement is far more expensive in terms of energy and latency than computation. An arithmetic operation like a Multiply-Accumulate (MAC) is orders of magnitude more efficient than reading the operands from off-chip DRAM. Therefore, the primary goal of an efficient accelerator is to minimize memory traffic by maximizing data reuse.

A basic dot product operation involves reading a filter weight and an input feature map (ifmap) activation, performing a MAC operation, and writing back the updated partial sum (psum). To improve efficiency, data must be kept as close to the ALUs as possible in the memory hierarchy.

Exploiting Data Reuse in Convolutions

Convolutional layers, which dominate many vision models, offer significant opportunities for data reuse. A diagram illustrating these opportunities shows three primary types:

1. Convolutional Reuse: In standard convolutions, the sliding window operation means the same filter weights are applied to overlapping input ifmaps, and the same input activations are multiplied by different weights. This allows for the reuse of both activations and weights.
2. Fmap Reuse: Across different filters applied to the same input, the entire input activation fmap can be reused. This applies to both CONV and Fully Connected (FC) layers.
3. Filter Reuse: When processing a batch of inputs (batch size > 1), the same filter weights can be reused across multiple different input fmaps.

A Taxonomy of Dataflow Strategies

The way data is moved and reused is defined by the accelerator's dataflow. Different strategies prioritize the reuse of different data types by making them "stationary" (i.e., holding them in a PE's local memory for multiple cycles).

Dataflow Strategy	Description	Key Characteristic	Examples
Weight Stationary (WS)	Each PE holds a weight stationary and processes a stream of input activations broadcast to it.	Maximizes weight reuse.	neuFlow, Google Edge TPU
Output Stationary (OS)	Each PE is responsible for accumulating a specific partial sum (psum) and holds it stationary.	Minimizes psum movement.	ShiDianNao
No Local Reuse (NLR)	Forgoes complex local reuse logic to dedicate more area to a larger shared buffer.	Area-efficient, but less energy-efficient.	Google TPU, DianNao
Row Stationary (RS)	A novel approach that aims to maximize reuse for all data types simultaneously.	Decomposes convolutions into 1D primitives.	MIT Eyeriss

The Eyeriss Innovation: Row Stationary (RS) Dataflow

The Row Stationary (RS) dataflow was developed to overcome the limitations of previous strategies, which were often optimal only for specific layer shapes. The core idea is to maximize data reuse for all three data types: filter weights, ifmaps, and psums.

* Mechanism: RS achieves this by processing convolutions one row of filter weights at a time. A row of filter weights is kept stationary within a PE's register file. Input activations are multicast to the PEs, and partial sums are accumulated spatially across the PE array.
* Benefit: This approach maximizes the use of the smallest, most energy-efficient memory—the register file—for filter data. It also facilitates reuse of ifmap data that is passed between adjacent PEs and allows psums to be accumulated locally before being sent to the global buffer.
* Spatial Unrolling: The process is spatially unrolled across the PE array. As shown in the diagrams, each PE processes a single row of a convolution. Multiple PEs work in parallel on different rows of the same filter or on different output columns. This can be extended to handle multiple channels, filters, and ifmaps in a batch, fully exploiting all available parallelism.

Eyeriss Hardware Architecture

The Eyeriss chip is a physical embodiment of the Row Stationary dataflow. The architecture diagram shows:

* A 14x12 PE Array forms the computational core.
* A Global Buffer (108KB SRAM) serves as the central on-chip memory, buffering data to and from the PE array. This minimizes access to the much slower off-chip DRAM.
* Data paths for Filter, Input Fmap, and Output Fmap feed into and out of the Global Buffer. The input path includes a decompression (Decomp) unit, while the output path has compression (Comp) and ReLU activation units.
* The entire accelerator communicates with Off-Chip DRAM via a 64-bit interface.

The key feature is the inter-PE communication network, which allows ifmaps and psums to be passed directly between neighboring PEs, which is essential for the efficiency of the RS dataflow.

TPU vs. Eyeriss: A Philosophical Comparison

While both are DNN accelerators, they represent different design philosophies:

* Google TPU: A commercial, general-purpose ML accelerator designed for usability and broad applicability (NLR dataflow). It uses a systolic array without direct inter-PE communication, relying on a unified buffer.
* MIT Eyeriss: A research accelerator focused on maximizing energy efficiency through sophisticated dataflow optimization (RS dataflow). It features a more complex memory hierarchy with inter-PE communication to support its advanced data reuse strategy.

Chapter 6: A Survey of the Modern Accelerator Landscape

Beyond the deep dives into the Edge TPU and Eyeriss, the field is rich with other innovative architectures.

Graphcore's Intelligence Processing Unit (IPU)

The Graphcore IPU is designed for both training and inference in data centers. Its architecture diverges significantly from GPU-like designs.

* Massive Parallelism: A diagram of the IPU shows it is composed of 1472 independent IPU-Tiles, each with its own IPU-Core and dedicated In-Processor-Memory.
* SRAM-centric Design: It replaces the traditional cache hierarchy with 900MB of distributed on-chip SRAM, providing extremely high memory bandwidth (47.5 TB/s per IPU). This design choice is based on the premise that ML workloads have predictable memory access patterns that can be managed by a compiler, making complex caches unnecessary.
* MIMD Architecture: Each core can run its own program thread in parallel (supporting 8832 independent threads per IPU), making it a Multiple Instruction, Multiple Data (MIMD) machine. This provides great flexibility for complex and irregular models, such as those with high sparsity.
* Software: The IPU is programmed using the Poplar C++ framework, which represents computations as a graph of vertices (code) and edges (data dependencies on tensors).

Tenstorrent Wormhole

Tenstorrent's architecture is another example of extreme specialization, tailored for neural networks like convolutions and transformers.

* Specialized Cores: It uses an array of simple "Baby-RISC V" cores, stripping away unnecessary features to focus on raw NN performance.
* Scalability: The Wormhole chip is designed to be extendable into a multi-chip mesh, allowing for the construction of very large systems.
* Open Software: Tenstorrent emphasizes an open software ecosystem, aiming to provide greater flexibility and community involvement.

Chapter 7: Summary and Future Directions

This review of machine learning processors reveals several key themes and points toward future trends in hardware-aware AI.

Key Takeaways:

* Specialization is Key: Processor specialization provides significant performance and efficiency gains for ML workloads but requires sacrificing general-purpose flexibility and often imposes specific software toolchains.
* Tensor Operations Dominate: As of today, tensor and matrix operations are the most critical computations in deep learning. This has led to the prevalence of array-based processors (systolic arrays, spatial arrays) with ISAs that natively support these operations.
* Data Reuse is Paramount: The high cost of data movement makes dataflow optimization the most important challenge in accelerator design. Strategies like Weight Stationary, Output Stationary, and Row Stationary are all attempts to maximize data reuse and minimize off-chip memory access.

Future Trends and Unsafe Optimizations: The pursuit of ever-greater efficiency is leading researchers to explore more aggressive, or "unsafe," optimizations at the circuit and device level:

* Reduced Precision Data Types: Moving beyond 8-bit integers to even lower precision formats.
* Analog Computing: Performing computation in the analog domain to potentially reduce power consumption, though this introduces challenges with noise and precision.
* Emerging Memory Technologies: Utilizing novel memory devices like memristors for in-memory computing, where computation happens directly where data is stored, eliminating the memory-compute bottleneck entirely.
