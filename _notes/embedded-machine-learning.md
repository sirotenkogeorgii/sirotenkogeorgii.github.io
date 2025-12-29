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













Below is a **formal, study-ready reformatted version** with clearer hierarchy and **highlighting via callouts, emphasis, and “Key idea / Formula / Embedded note” blocks**. Meaning is unchanged; only formatting and readability are improved.

---

# Chapter 1 — The High Cost of Deep Learning on Embedded Systems

Deep Neural Networks (DNNs) deliver state-of-the-art accuracy in vision and language, but they are inherently **compute- and memory-intensive**. This creates a deployment barrier on embedded devices, where **power, memory, and compute throughput** are strictly limited.

> **Key problem**
> Modern DNN accuracy is purchased with high FLOPs, large parameter storage, and large runtime activation memory.

---

## 1.1 Defining the Resource Challenge

Embedded ML aims to close the gap between:

* **DNN resource demand**, and
* **edge device capability** (MCU/DSP/NPU constraints).

### Example: ImageNet @ 224×224 (ResNet50)

A baseline architecture such as ResNet50 requires:

* **Compute:** **3.9 GFLOPs** per inference
* **Parameters (weights):** **102 MB**
* **Runtime activations:** **187 MB**

These requirements typically exceed the available SRAM and compute budget of many embedded targets.

> **Embedded objective**
> Reduce FLOPs, parameter storage, and activation memory **aggressively**, while maintaining acceptable accuracy.

---

## 1.2 Trade-Off: Accuracy vs. Efficiency

Model performance correlates strongly with model complexity:

* deeper/wider models → typically higher accuracy
* but also → more FLOPs, more parameters, larger activations

### Observed trends (from course graphs)

The materials compare architectures (ResNet/ResNeXt/DenseNet/MobileNet) across three axes:

1. **Accuracy vs. FLOPs**

   * Higher GFLOPs usually increases Top-1 accuracy.
   * Example anchor: **ResNet50 ≈ 3.87 GFLOPs → ~76% Top-1**.
   * **ResNeXt** pushes beyond ~8 GFLOPs for smaller accuracy gains.
   * **MobileNet** families operate below ~1 GFLOP with reduced accuracy but large compute savings.

2. **Accuracy vs. Parameters**

   * More parameters often correlate with better accuracy.
   * Example anchor: **ResNet50 ≈ 102 MB** weights.

3. **Accuracy vs. Activations**

   * Runtime memory is a major bottleneck on embedded targets.
   * Example anchor: **ResNet50 ≈ 187 MB** activations.

> **Key takeaway**
> Embedded deployment requires navigating an explicit **accuracy–resource trade-off**: reduce FLOPs and memory with minimal accuracy loss.

---

# Chapter 2 — A Framework for Model Optimization

Optimizing a DNN for embedded deployment is a multi-layer problem. Effective design requires a holistic view across architecture, algorithms, and hardware.

---

## 2.1 The DNN Compute Stack

Efficient deployment can be viewed as a layered stack:

1. **Neural architecture**
   Defines the structure of computation and memory access patterns; largest leverage.

2. **Compression & algorithm**
   Techniques applied to a chosen architecture (e.g., **quantization**, **pruning**).

3. **Hardware**
   Execution substrate (CPU/GPU/DSP/NPU/ASIC/FPGA) determines what optimizations translate to real gains.

> **HW–ML Interplay (principle)**
> An algorithmic idea only produces speed/energy benefits if the hardware can exploit it (e.g., 1-bit compute needs fast bitwise datapaths).

---

## 2.2 Safe vs. Unsafe Optimizations

Optimizations can be classified by whether they can change the model’s output.

---

### Safe Optimizations (accuracy-preserving)

**Definition:** reduce resources without changing accuracy.

**Characteristics**

* guaranteed preservation of model behavior
* typically hardware/system-level improvements

**Examples**

* shorter communication paths (layout / interconnect optimization)
* data reuse (caching weights/activations; exploiting locality)
* dedicated compute architectures (e.g., systolic arrays for GEMM)

> **Highlight**
> Safe optimizations are “free” in accuracy, but require careful hardware/system design.

---

### Unsafe Optimizations (accuracy-risking)

**Definition:** modify the model/representation to reduce intrinsic cost, potentially affecting accuracy.

**Characteristics**

* may degrade accuracy; must be measured/mitigated
* core “engineering” challenge in embedded ML

**Examples**

* **pruning** (remove weights/structures)
* **quantization** (reduce numerical precision: FP32 → INT8 → binary)

> **Key point**
> Most embedded ML performance gains come from unsafe optimizations applied carefully.

---

# Chapter 3 — The Physics of Computation: Energy and Precision

Energy and latency depend strongly on:

* **numerical format (bit-width)**
* **data movement** (cache vs DRAM)

> **Central principle**
> Computation is often cheaper than memory access; optimizing data movement is crucial.

---

## 3.1 Numerical Formats: Floating-Point vs Fixed-Point

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

Floating-point provides large dynamic range; Below is scientific notation for computers. **Formula:**

$$v = (-1)^S \cdot (1 + F) \cdot 2^{(E-\text{bias})}$$

$$P_s=A_s \bigoplus B_s \qquad P_E = A_E + B_E \qquad P_F = A_F \dot B_F$$

* **Structure**:
  * **sign bit** $S$ (positive or negative).
  * **exponent/fraction** $E$ (moves the decimal point, allowing for huge ranges).
  * **significand** $F$ (the actual digits).
* **The Distribution Problem:** Floating point is "dense" near zero. The slide notes that half of all representable numbers fall between $−1$ and $1$. This is great for weights in neural networks, which are usually small.
* Complexity: Floating point hardware is much more complex because to multiply two numbers, you have to add their exponents ($P_E=A_E+B_E$) and multiply their fractions ($P_F=A_F\dot B_F$) separately.
* **Cost:** FPUs are large and power-hungry relative to integer units.

| Format   | Sign | Exponent | Significand | Dynamic range                     |
| -------- | ---- | -------- | ----------- | --------------------------------- |
| float64  | 1    | 11       | 52          | $\sim 2\times 10^{\pm308}$        |
| float32  | 1    | 8        | 23          | $\sim 2\times 10^{\pm38}$         |
| float16  | 1    | 5        | 10          | $\sim 10^{-4} to 10^{+5}$       |
| bfloat16 | 1    | 8        | 7           | similar exponent range to float32 |

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

The lecture mentions: `mult: {Energy, Area} ∝ $N^2$[bits]`. This is a hidden win for bfloat16.
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
  <p class="math-callout__title"><span class="math-callout__label">Example:</span><span class="math-callout__name">(Fixed-Point)</span></p>

In a fixed-point system, we decide ahead of time exactly where the "point" goes. Imagine we have an **8-bit** system where we decide the first 4 bits are for the whole number and the last 4 bits are for the fraction.
* **The Number:** $12.75$
* **Binary of 12:** `1100`
* **Binary of 0.75:** `1100` (Since $0.75 = 1/2 + 1/4$, which is $2^{-1} + 2^{-2}$)
* **The Result:** `1100.1100`

**The Hardware Perspective:** The computer doesn't actually store a "point." It just sees the integer `11001100` (which is 204 in decimal). The programmer just remembers to divide by 16 at the end. This makes addition as fast as regular integer math.

</div>

Fixed-point uses integers with an implicit binary point:

* smaller dynamic range than floating point
* simpler hardware → better energy efficiency

* **Scaling:** Addition complexity grows linearly with the number of bits ($N$). Multiplication is much harder; it grows quadratically ($N^2$). If you double the bits, a multiplier becomes four times as large/expensive.
* **The Trade-off:** Fixed point is very energy-efficient but has a **small dynamic range**. You can’t represent a very tiny number and a very huge number at the same time without losing precision.

> **Highlight**
> Quadratic multiplier scaling creates strong pressure to reduce precision.

| Feature | Fixed-Point (e.g., `1100.1100`) | Floating-Point (e.g., $1.10011 \times 2^3$) |
| :--- | :--- | :--- |
| **Metaphor** | A **Ruler**: The marks are always the same distance apart. | A **Map**: You can zoom in for high detail or zoom out to see the whole world. |
| **Precision** | Always the same (e.g., always accurate to 0.01). | High precision for small numbers; low precision for huge numbers. |
| **Hardware** | Very simple (uses the Integer Unit). | Complex (requires a dedicated Floating Point Unit). |
| **Best Use** | Money (cents), simple sensors, or low-power microcontrollers. | 3D Graphics, AI training, and complex physics simulations. |

## 3.2 The Picojoule Economy: Energy Costs

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
Look at the tables. An 8-bit Integer Add costs **0.03 pJ**, while a 16-bit Floating Point Add costs **0.4 pJ**.
* **Takeaway:** Floating point operations are roughly **10x more expensive** in terms of energy than integer operations. If you can do your math with integers (Fixed Point), your battery lasts 10x longer.

**2. The Scaling Law**
The slide reiterates: **ADD scales with $n$, MULT with $n^2$**. A 32-bit integer addition costs **0.1 pJ**.
* A 32-bit integer multiplication costs **3.1 pJ**.
* **Takeaway:** Multiplications are the "energy hogs" of computing. This is why researchers try to design neural networks that use more additions and fewer multiplications.

**3. The "Memory Wall" (The Most Important Part)**
Look at the **Memory** table on the right.
* A 32-bit Floating Point multiplication costs **3.7 pJ**.
* Reading that same number from **DRAM (Main Memory)** costs **1300–2600 pJ**.
* **The Shocking Truth:** It costs **hundreds of times more energy** to simply "fetch" a number from memory than it does to actually perform the math.


### Key takeaways

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

# Chapter 4 — Quantization: Trading Precision for Performance

Quantization maps high-precision values to a discrete low-precision set. It is one of the most effective unsafe optimizations for embedded inference.

## 4.1 Core Concepts

A quantizer $Q$ maps an input $x$ into discrete levels $\lbrace q_\ell\rbrace$, typically using thresholds $\lbrace\delta^\ell\rbrace$.

### Uniform k-bit quantizer on $[0,1]$

For $a_i \in [0,1]$:

$$a_q^i = \frac{1}{2^k - 1}\cdot \mathrm{round}\big((2^k-1)a_i\big)$$


Example: $k=2$ maps [0,1] into 5 representable values
$\lbrace 0, 0.25, 0.5, 0.75, 1\rbrace$ (as stated in the notes).

Quantization can be applied to:

* **weights**
* **activations**
* (sometimes) **gradients**

> **Optimization goal**
> Use the smallest bit-width that preserves acceptable accuracy.

---

## 4.2 Uniform Quantization

Uniform quantization uses constant step size:

$$q_{i+1} - q_i = \Delta$$

**Advantages**

* simple representation (integers + shared scale)
* efficient integer arithmetic when weights and activations are aligned

**Disadvantages**

* may waste representation capacity when distributions are non-uniform (e.g., bell-shaped weights)

### Binary quantization (extreme uniform case)

$$
Q(x)=
\begin{cases}
+1 & x \ge 0 \\
-1 & x < 0
\end{cases}
$$

---

## 4.3 Non-Uniform Quantization

Non-uniform quantization allocates more levels where values are dense.

**Advantages**

* higher accuracy at same bit-width by matching data distribution

**Disadvantages**

* requires storing quantization levels (lookup tables)
* potential runtime overhead

### Example: trainable ternary quantizer

$$
w_i^l =
\begin{cases}
W_p^l & w^l > \Delta^l\
0 & \lvert w^l\rvert\le \Delta^l\ * W_n^l & w^l < -\Delta^l
\end{cases}
$$

with $\Delta^l$ defined via a learnable fraction of the maximum weight.

> **Highlight**
> Learnable thresholds/scales allow the quantizer to adapt to layer-specific distributions.

---

## 4.4 Hardware Acceleration and Low-Precision MACs

DNNs are dominated by **MAC operations** (matrix multiplication/convolution). Lower precision enables more parallelism per area/energy.

### Hardware intuition (bit-width scaling)

* **32-bit MAC:** fewer parallel units; higher energy/area per MAC
* **8-bit MAC:** many MACs can run in parallel within the same resources
* **1-bit MAC:** multiplication becomes bitwise logic; accumulation becomes popcount

---

### XNOR-based binary multiplication

For binary networks with $\lbrace -1,+1\rbrace$, map to $\lbrace 0, 1\rbrace$. Then:

$$c = a \cdot b = 2\cdot \mathrm{popc}(\mathrm{xnor}(a,b)) - N$$

where $\mathrm{popc}$ counts set bits, and $N$ is vector length.

> **Key idea**
> Replace multiply-add with XNOR + popcount → extremely efficient on suitable hardware.

---

## 4.5 Training Strategies for Quantized Networks

Quantization may harm accuracy unless training accounts for quantization effects.

### 1) Post-Training Quantization (PTQ)

* quantize after training
* needs calibration data for scale/zero-point
* can lose accuracy at low bit widths

### 2) Quantization-Aware Training (QAT)

* forward pass: fake-quantize (quantize → dequantize) to inject quantization noise
* backward pass: update full-precision parameters

### 3) Straight-Through Estimator (STE)

Quantization is piecewise constant ⇒ true gradient is zero/undefined. STE approximates gradient as identity:

$$
\frac{\partial\mathcal{L}}{\partial w}
\approx
\frac{\partial\mathcal{L}}{\partial f},\tilde{f}'(w),
\quad \tilde{f}'(w)\approx 1
$$

### 4) Fine-tuning

Quantize a pre-trained model then QAT fine-tune to recover accuracy.

> **Highlight**
> QAT + fine-tuning is often necessary for low-bit quantization without major accuracy loss.

---

## 4.6 Survey: Quantization Schemes and Performance

| Method     | Weights $W$     | Activations $A$  |
| ---------- | --------------- | ---------------- |
| BNN        | $\lbrace -1,+1\rbrace$       | $\lbrace -1,+1\rbrace$        |
| XNOR-Net   | $\lbrace -S,+S\rbrace$       | $\lbrace -1,+1\rbrace$        |
| DoReFa-Net | $\lbrace -S,+S\rbrace$       | $\lbrace 0,+1\rbrace$ (k-bit) |
| TWN        | $\lbrace -S,0,+S\rbrace$     | float32          |
| TTQ        | $\lbrace -S_n,0,+S_p\rbrace$ | float32          |
| HWGQ       | XNOR                       | 2-bit            |

**Empirical trend (from notes):**

* 1-bit schemes (BNN/XNOR) lose significant accuracy
* moderate quantization (e.g., TTQ with low-bit weights + FP activations) approaches FP baseline
* example: TTQ Top-5 ≈ 79.7% vs baseline 80.3% (AlexNet/ImageNet)

> **Trade-off summary**
> Lower bits → strong memory/latency improvements, but accuracy recovers substantially by 2–3 bits depending on method.

---

# Chapter 5 — Pruning: Engineering Sparsity in Neural Networks

Pruning removes unimportant connections by setting weights to zero, reducing model size and potentially compute.

---

## 5.1 Principle: Inspired by Nature

Biological analogy: synaptic density peaks early in life and decreases later (“synaptic pruning”).
Similarly, DNNs are often over-parameterized and contain redundancy that can be removed.

> **Key hypothesis**
> Many weights are not critical for prediction and can be eliminated with minimal accuracy loss (especially with retraining).

---

## 5.2 Pruning Workflow and Criteria

Pruning is typically a pipeline:

1. **Train** dense network to convergence
2. **Prune** weights/structures using a criterion
3. **Fine-tune** sparse model to recover accuracy

Often performed iteratively:

$$\text{Train} \rightarrow \text{Prune} \rightarrow \text{Fine-tune} \rightarrow \cdots$$

versus one-shot pruning.

### Common pruning criteria

* **Magnitude pruning**
  
  $$\lvert w_i\rvert \le t \Rightarrow w_i \gets 0$$
  
* **Gradient-based (saliency) pruning**
  
  $$\lvert w_i g_i\rvert \le t \Rightarrow w_i \gets 0$$

> **Highlight**
> Magnitude pruning is simple and widely used; gradient-based measures attempt to capture impact on loss.

---

## 5.3 Granularity: Unstructured vs Structured Sparsity

Pruning granularity strongly affects hardware speedups.

---

### Unstructured (fine-grained) pruning

Removes individual weights anywhere.

**Pros**

* best accuracy–sparsity flexibility

**Cons**

* difficult to accelerate on CPUs/GPUs
* requires indices (e.g., CSR format: data $d$, column indices $i$, row pointers $r$)
* irregular memory access breaks coalescing/locality and can negate speedups

> **Embedded note**
> Unstructured sparsity often compresses storage but may not reduce latency unless hardware explicitly supports sparse compute efficiently.

---

### Structured (coarse-grained) pruning

Removes groups:

* channels/filters in CNNs
* rows/columns in FC layers
* blocks/tiles

**Pros**

* results in smaller *dense* tensors
* preserves regular compute patterns → easier speedups on standard accelerators

**Cons**

* less flexible → may lose more accuracy for the same number of removed weights

#### Parameterized structured pruning (learnable structure selection)

Associate each structure $w_i$ with $\alpha_i$ and gate via:

$$
w_i^{qi}=w_i\cdot v_i(\alpha_i),
\quad
v_i(\alpha_i)=
\begin{cases}
0 & \lvert \alpha_i\rvert<\epsilon\
\alpha_i & \lvert \alpha_i\rvert\ge \epsilon
\end{cases}
$$

Non-differentiability handled using **STE** to learn $\alpha_i$.

---

## 5.4 Impact of Retraining and Regularization

Fine-tuning after pruning is typically required.

### Observed weight distribution shift (from notes)

* **before pruning:** narrow Gaussian-like centered around zero
* **after pruning + retraining:** bimodal distribution; remaining weights move away from zero
  → surviving connections strengthen to compensate

### Role of regularization

* **L1**
  
  $$\mathcal{R}_{L1}(w)=\sum_j \lvert w_j\rvert$$
  
  encourages sparsity (pushes weights toward zero)
* **L2**
  
  $$\mathcal{R}_{L2}(w)=\frac{1}{2}\sum_j w_j^2$$
  
  encourages small weights but not exact zeros

**Empirical note:** For AlexNet/ImageNet in the notes:

* with retraining, **L2** ultimately performs best for pruning
* without retraining, **L1** is better
* main takeaway: **retraining is essential**

> **Key takeaway**
> Pruning is not a one-step operation; iterative pruning + fine-tuning is the standard route to high sparsity with acceptable accuracy.

---











Below is a **formally reformatted, study-ready version** with **highlighting via callouts, bold emphasis, “Key idea / Formula / Practical note” blocks**, and improved structure. Content is preserved; edits are formatting-only.

---

# Unsafe Optimizations II — Quantization

Quantization is a central compression technique for deploying neural networks on **resource-constrained embedded systems**. The core theme is **HW–ML interplay**: the numerical representation (bit-width, format, granularity) must match the target hardware’s constraints in **memory, energy, and latency**.

> **Learning objective**
> Understand how quantization changes *representation*, how this changes *compute cost*, and when it changes *accuracy*—and how to mitigate that accuracy loss.

---

## 1.1 Model Compression: Pruning vs. Quantization

Deploying modern models on embedded devices requires reducing:

* **model storage** (weights),
* **runtime memory** (activations),
* **compute** (MAC/FLOP-equivalent),
* **energy** (dominated by data movement).

### Pruning (structure removal)

**Definition:** remove redundant parameters from a trained model.

* **Unstructured pruning:** remove individual connections (creates sparse matrices).
* **Structured pruning:** remove higher-level structures (neurons, channels, filters).

> **Hardware dependence**
> Pruning speedups require hardware/runtime support for sparse execution; otherwise, sparse indexing overhead can negate gains.

### Quantization (precision reduction)

**Definition:** reduce numerical precision of:

* **weights**, and optionally
* **activations** (and rarely gradients during training).

Typical conversion: **Float32 → INT8/INT4 → binary/ternary**.

> **Why quantization is powerful**
> Fewer bits reduce memory bandwidth and enable faster, lower-energy arithmetic (integer / bitwise), *if hardware supports it*.

---

## 1.2 Hardware-Aware Metric: Bit Operations (BOPS)

### Motivation

Counting MACs alone can be misleading because **compute cost depends on bit-width**. BOPS provides a more hardware-abstract measure of fixed-point compute complexity.

### Setup (conv layer)

Let:

* $b_w$: weight bit-width
* $b_a$: activation bit-width
* $n$: input channels
* $m$: output channels
* $k$: kernel size (for $k \times k$)

Per output element, MAC count is $n k^2$. The accumulator width (binary-coded) is approximated by:

$$b_o = b_a + b_w + \log_2(nk^2)$$

### BOPS for convolution (approx.)

$$
\text{BOPS}_{\text{conv}} \approx m n k^2\left( \underbrace{b_a b_w}*{\text{multiplication}} + \underbrace{b_o}_{\text{accumulation}} \right)
$$

> **Highlight — scaling behavior**
> Reducing $b_a$ and $b_w$ yields:
>
> * **quadratic** reduction in multiplication cost $(b_a b_w)$
> * **linear** reduction in accumulation cost $(b_o)$

---

## 1.3 Fundamentals of Quantization

Quantization maps high-precision values to low-precision discrete codes.

### Core mapping (affine quantization)

$$q = Q(x) = \mathrm{clip}\Big(\mathrm{round}\big(\frac{x}{s}+z\big),\ q_{\min},q_{\max}\Big)$$

**Glossary**

* $x$: original float value
* $q$: quantized integer code
* $s>0$: scale (step size)
* $z$: zero-point (integer code corresponding to real 0)
* $q_{\min}, q_{\max}$: integer range (e.g., [-128,127] for signed INT8)

### Dequantization (reconstruction)

$$\hat{x} = (q - z)\cdot s$$

> **Quantization error sources**
> * rounding error
> * clipping/saturation (when values exceed representable range)

---

### Symmetric vs. Asymmetric quantization

#### Symmetric (typically for weights)

Set $z=0$; map range symmetrically around zero.

$$s = \frac{\max(\lvert x\rvert)}{q_{\max}}$$

**Common use-case:** weights often have distributions centered near 0.

#### Asymmetric (often for activations)

Allow nonzero $z$, useful for nonnegative activations (e.g., after ReLU).

$$s = \frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}}, \qquad z=\mathrm{round}\Big(q_{\min}-\frac{x_{\min}}{s}\Big)$$

> **Practical note**
> Post-ReLU activations are nonnegative → unsigned quantization is often preferred.

---

### Static vs. Dynamic quantization

#### Static scaling

* compute $s,z$ offline from a calibration set
* fixed at inference
* **no runtime overhead**

**Risk:** distribution shift (calibration ≠ deployment).

#### Dynamic scaling

* compute range (and thus $s,z$) per input/batch at runtime
* **more robust** to input variation
* **adds inference overhead**

---

## 1.4 Taxonomy of Quantization Techniques

| Dimension               | Options                                | Key trade-offs                                                                                                                |
| ----------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Procedure**           | PTQ vs QAT                             | PTQ: fast/simple (often good for INT8). QAT: training-time simulation, needed for aggressive low-bit (INT4, ternary, binary). |
| **Target**              | Weight-only vs Weights+Activations     | Weight-only reduces storage/bandwidth; W+A enables integer-only pipelines for maximum latency/energy gains.                   |
| **Range determination** | Static vs Dynamic                      | Static is fast but sensitive to mismatch; dynamic is robust but adds overhead.                                                |
| **Granularity**         | Per-tensor vs Per-channel vs Per-group | Per-tensor is simplest; per-channel is more robust for CNN weights; per-group trades complexity and robustness.               |

### Practical escalation strategy

1. Start with **INT8 PTQ**.
2. If accuracy drops: improve calibration (percentile/MSE) and/or use **per-channel** weights.
3. Use **mixed precision** for sensitive layers.
4. If still insufficient: apply **QAT**.

---

## 1.5 Practical Quantization: ResNet Case Study

### INT8 (often sufficient)

#### conv1 weights (Float32 → INT8)

* distribution: bell-shaped around 0
* method: **symmetric min/max**, signed INT8 ([-128,127])
* example parameters: $s=0.007972$, $z=0$
* result: dequantized histogram closely matches original → low error

#### ReLU activations (Float32 → INT8)

* distribution: nonnegative, right-skewed, with outliers
* method: **asymmetric min/max**
* example parameters: $s=0.01287$, $z=-128$
* result: good match with proper calibration

> **Interpretation**
> With reasonable calibration, INT8 preserves most signal for both weights and activations in common CNNs.

---

### INT4 (aggressive)

#### conv1 weights (Float32 → INT4)

* method: symmetric min/max, signed INT4 ([-7,7])
* example parameters: $s=0.1452$, $z=0$
* effect: coarse binning → visible discretization → higher error

#### ReLU activations (Float32 → INT4)

* method: asymmetric min/max, unsigned INT4 ([0,15])
* example parameters: $s=0.2187$, $z=0$
* effect: activation distribution collapses into few peaks → strong information loss

> **Practical conclusion**
> INT4 usually requires **QAT** or more careful calibration/mixed precision to preserve accuracy.

---

## 1.6 Real-World Data Challenge: Calibration Mismatch

Static activation quantization assumes:

$$\text{calibration distribution} \approx \text{deployment distribution}$$

When violated, accuracy can degrade.

### Common mismatch sources

* lighting/exposure shifts (activation tails grow)
* sensor/optics changes (noise/color response changes)
* compression artifacts (JPEG/streaming distortions)

### Mechanism of failure

If tails extend, min/max calibration expands range → scale $s$ increases → resolution for typical values decreases → quantization error increases.

Example trend (from notes): scale increases from **0.01287 → 0.01672** under mismatch.

### Mitigation strategies

1. diversify calibration data
2. use robust range selection (percentile, MSE-based)
3. increase granularity (per-channel)
4. mixed precision for sensitive layers (often first/last)

> **Embedded note**
> Deployment conditions vary; calibration should be designed as an engineering artifact, not an afterthought.

---

## 1.7 Quantization-Aware Training (QAT) Methods

When PTQ fails, QAT integrates quantization effects into training to learn robustness.

---

### Trained Ternary Quantization (TTQ)

Quantizes weights to three values: $\lbrace +W_p, 0, -W_n\rbrace$ with learned scales.

#### Quantization rule

$$
\tilde{w}_i =
\begin{cases}
W_p & w_i>\Delta_l\
0 & \lvert w_i\rvert\le \Delta_l\
-W_n & w_i<-\Delta_l
\end{cases}
\quad
\text{with}\quad
\Delta_l=t\cdot \max(\lvert w\rvert),\ t\in[0,1]
$$

**Training idea:** learn both the ternary assignments and the scale factors $(W_p, W_n)$ via backprop.

---

### DoReFa-Net

A framework for arbitrary bit-widths for:

* weights $W$
* activations $A$
* gradients $G$

Uses deterministic quantization for $W/A$ and stochastic quantization for gradients, relying on **STE** for backprop through non-differentiable quantizers.

> **Key point**
> Gradient precision can significantly influence trainability and final accuracy.

---

### LQ-Nets (Learned Quantization Networks)

Learns **non-uniform quantization levels** via a trainable basis vector $v\in\mathbb{R}^K$:

$$q_l = v^T b_l$$

where $b_l$ is a binary coding vector.

**Intuition:** learned levels match the weight distribution better than uniform grids, reducing quantization error at fixed bit-width.

---

## 1.8 Hardware–Software Interplay: Architectures and Methods

Quantization gains are maximized only when matched to hardware.

---

### Bit-serial multiplication (arbitrary precision hardware)

If $a$ is N-bit and $b$ is M-bit, multiply via bit-level decomposition:

$$c=a\cdot b = \sum_{n=1}^{N}\sum_{m=1}^{M}2^{n+m}\cdot \mathrm{popc}\big(\mathrm{and}(a_n,b_m)\big)$$

Complexity:

$$O(NM)$$

Latency increases per operation, but logic is simple and can be heavily parallelized.

---

### DeepChip: Reduce-and-Scale (RAS)

A HW-aware quantization approach targeting resource-constrained processors (e.g., mobile ARM).

**Components**

1. **Ternary weight quantization** (TTQ-like): $\lbrace -W_n,0,W_p\rbrace$ with learned asymmetric scales.
2. **Activation quantization:** bound activations (bounded ReLU) then k-bit fixed-point quantize.
3. **Space-efficient storage:** store signs and distances between nonzeros (run-length-like), compressible (e.g., Huffman).
4. **Operator reformulation:** avoid expensive multiplications by turning compute into sums + few scalings:

$$c = W_p^l\cdot \sum_{i\in i_p^l}a_i + W_n^l\cdot \sum_{i\in i_n^l}a_i$$

**Hardware motivation:** integer adds can be far cheaper than integer FMA on many embedded CPUs.

> **Highlight**
> This is an example of algorithm–hardware co-design: the compression method is shaped by the cost model of the target processor.

---

## 1.9 Pitfalls and Best Practices

### Common pitfalls

* confusing PTQ with weight-only by default
* unrepresentative calibration data
* blind min/max for activations (outliers dominate range)
* wrong signed/unsigned choice (ReLU activations)
* per-tensor weights for CNNs (often too brittle)
* quantizing all layers equally (first/last often sensitive)
* not monitoring saturation, histograms, and layerwise errors

### Robust baseline (recommended rule-of-thumb)

* **Uniform INT8 PTQ**
* **Per-channel weights** (CNNs)
* **Percentile-based activation calibration**

Escalate only if needed:

* finer granularity → mixed precision → QAT.

---

## 1.10 Automated Search for Compression Policies

The design space (bit-widths per layer, pruning ratios, calibration methods) is too large for manual tuning.

### Hardware-aware automated approaches

* **GALEN:** reinforcement learning to jointly select pruning + quantization policies, uses sensitivity analysis and real latency measurements.
* **HAQ:** RL-based mixed-precision quantization (2–8 bits) under a latency budget, often using lookup-table latency models.

> **Direction of the field**
> Compression is increasingly treated as an optimization problem constrained by real hardware latency/energy, solved via automated search.

---












# The Illusion of Sparsity: From Theory to Hardware-Accelerated Pruning

This chapter addresses the gap between **theoretical compression** and **real hardware speedups**. Sparsity (many zero weights) only helps if the **runtime kernel** and **hardware** can *skip work efficiently*. The key theme is **HW–ML interplay**: the best pruning method is the one that your target system can execute efficiently.

---

## 1.1 The Sparsity–Speed Fallacy

A common misconception:

> **Fallacy:** “80% zeros ⇒ 5× speedup”
> **Reality:** Often ≈ **1×** without hardware-aware execution.

### Why sparsity often does not accelerate inference

Modern processors and ML frameworks are optimized for **dense, structured kernels** (e.g., GEMM). If your sparse model still runs **dense kernels**, the FLOPs are not actually skipped.

### Three practical approaches to sparsity

#### 1) Mask-only sparsity (framework-level masking)

Compute is still dense:

$$y = x \cdot (W \odot M)$$

* **Storage:** $W$ is still dense (mask is extra)
* **Execution:** dense GEMM/conv kernel still runs
* **Outcome:** typically **no speedup**

> **Use-case**
> Good for *research*, sensitivity analysis, pruning schedules—not for inference acceleration.

#### 2) Structured pruning (rewiring)

Remove whole structures (channels/neurons/heads), producing **smaller dense tensors**.

* **Storage:** smaller weight matrices
* **Execution:** standard dense kernels on smaller tensors
* **Outcome:** real, portable speedups on CPUs/GPUs

> **Key point**
> Speed comes from *smaller shapes*, not from zeros.

#### 3) Hardware-structured sparsity (pattern-constrained + packed)

Enforce a hardware-friendly pattern (e.g., **N:M**) and use a **specialized sparse kernel**.

* **Storage:** packed nonzeros + metadata for pattern
* **Execution:** sparse kernel explicitly skips work
* **Outcome:** significant speedups if hardware supports it

> **Trade-off**
> Pattern constraints may reduce accuracy; operator coverage may be limited.

---

## 1.2 Sparsity Formats and Granularity

Pruning granularity determines both:

* **accuracy retention** (flexibility), and
* **hardware efficiency** (regularity).

### Fine-grained (unstructured) pruning

* Removes individual weights (random “salt-and-pepper” zeros)
* **Pros:** highest flexibility; accuracy usually best at a given sparsity
* **Cons:** poor hardware utilization; irregular memory access; low parallel efficiency

### Coarse-grained (structured) pruning

* Removes groups (rows/cols/channels/filters)
* **Pros:** compatible with dense kernels; predictable access; fewer overheads
* **Cons:** less flexible; may cost more accuracy at same compression

> **Rule of thumb**
> If you want speed on general hardware: prefer **structured pruning**.

---

### CSR format (Compressed Sparse Row)

**Goal:** store sparse matrices without explicitly storing zeros.

CSR stores:

1. **Data array** $d$: nonzero values
2. **Column indices** $i$: column index per value in $d$
3. **Row pointer** $r$: start offsets in $d$ for each row

#### Example dense matrix

$$
D=
\begin{bmatrix}
0&5&3&0\\
6&1&0&4\\
0&0&0&0\\
2&0&1&4
\end{bmatrix}
$$

CSR representation:

* $d = (5, 3, 6, 1, 4, 2, 1, 4)$
* $i = (1, 2, 0, 1, 3, 0, 2, 3)$
* $r = (0, 2, 5, 5, 8)$

**Storage comparison**

* Dense: $16$ values
* CSR: $8$ data + $8$ indices + $5$ pointers = $21$ units

> **Highlight — metadata overhead**
> CSR can be **larger than dense** at moderate sparsity and introduces **indirection**, which is costly on parallel hardware.

---

### BSR format (Block Sparse Row)

BSR divides the matrix into **dense blocks** (e.g., $2\times2$) and sparsifies at block granularity.

* **Benefit:** metadata overhead amortized over blocks
* **Benefit:** better computational regularity than CSR
* **Trade-off:** less flexible than fully unstructured sparsity

---

## 1.3 The Economics of Sparsity: Break-Even Model

Latency is governed by the max of **compute time** and **memory time**.

### Dense operation time

$$T_{\text{dense}} \approx \max\left(\frac{F}{P}, \frac{B}{BW}\right)$$


### Sparse operation time


$$T_{\text{sparse}} \approx \max\left(\frac{(1-s)F}{P_{\text{eff}}}, \frac{B_{\text{sparse}}}{BW}\right)+T_{\text{overhead}}$$


#### Glossary

| Term                  | Meaning                                     |
| --------------------- | ------------------------------------------- |
| $s$                   | sparsity fraction (zeros)                   |
| $F$                   | dense FLOPs                                 |
| $P$                   | peak throughput                             |
| $P_{\text{eff}}$      | effective sparse throughput (often $\ll P$) |
| $B$                   | bytes moved (dense)                         |
| $BW$                  | memory bandwidth                            |
| $B_{\text{sparse}}$   | bytes moved incl. metadata                  |
| $T_{\text{overhead}}$ | packing + conversion + imbalance overhead   |

### Key implications

#### Memory-bound regime

If $\frac{B}{BW}$ dominates, reducing FLOPs may not reduce latency.

> **Practical consequence**
> “Less compute” does not help when performance is dominated by **data movement**.

#### Unstructured sparsity

* lowers $P_{\text{eff}}$ due to irregular execution
* increases $B_{\text{sparse}}$ (metadata)
* increases $T_{\text{overhead}}$

→ break-even sparsity can be extremely high (often **> 90%**).

#### Structured sparsity

* reduces metadata/overhead
* improves $P_{\text{eff}}$

→ speedups can appear at much lower sparsity (e.g., ~50%).

> **Highlight — break-even effect**
> Structured sparsity yields earlier speedups than unstructured sparsity.

---

## 1.4 Practical Pruning in PyTorch

### Unstructured pruning (mask-based)

```python
import torch.nn as nn
import torch.nn.utils.prune as prune

fc = nn.Linear(in_features=10, out_features=6)

# Prune 30% smallest-magnitude weights (L1 criterion)
prune.l1_unstructured(fc, name='weight', amount=0.3)
```

* Adds a **weight_mask** (and keeps dense tensor)
* `prune.remove(...)` makes it “permanent” but still **dense with zeros**

> **Important**
> This is typically **mask-only sparsity** → does not imply speedup.

---

### Structured pruning (group removal)

Structured pruning removes entire rows/cols/channels, determined via a norm criterion.

#### $L1$ vs $L2$ sensitivity

* $\lvert w\rvert_1 = \sum_j \lvert w_j\rvert$: linear contribution
* $\lvert w\rvert_2^2 = \sum_j w_j^2$: large weights dominate

Example vectors:

* $a=(3,0,0,0)^\top$
* $b=(2,1,0,0)^\top$

$$\lvert a\rvert_1=\lvert b\rvert_1=3 \quad\text{but}\quad \lvert a\rvert_2^2=9, \lvert b\rvert_2^2=5$$


> **Interpretation**
> L2 can preserve groups with a single large weight; L1 treats weight mass more evenly.

Structured pruning in code:

```python
import torch.nn.utils.prune as prune

# Output neuron pruning (rows): dim=0
prune.ln_structured(fc, name='weight', amount=0.3, n=1, dim=0)  # n=1 → L1

# Input feature pruning (cols): dim=1
prune.ln_structured(fc, name='weight', amount=0.3, n=1, dim=1)

# Conv output channel pruning: dim=0
conv = nn.Conv2d(16, 32, kernel_size=3)
prune.ln_structured(conv, name='weight', amount=0.3, n=1, dim=0)
```

> **Caution**
> Pruning first/last layers structurally can break interface dimensions (input/output).

---

### Threshold-based pruning (custom mask)

```python
threshold = 0.15
mask = (fc.weight.data.abs() >= threshold)
prune.custom_from_mask(fc, name='weight', mask=mask)
```

* Removes all weights below a magnitude threshold
* Extensible to structured pruning via aggregate scores per group

---

## 1.5 Path to Real Speedup: Deployment Strategies

Masking alone does not accelerate inference because:

1. **dense kernels** still execute
2. masks are not sparse formats (CSR/BSR)
3. `prune.remove` produces dense tensors with zeros

### Deployment options

| Path                    | How it works                                    | Best for                            | Trade-offs                                 |
| ----------------------- | ----------------------------------------------- | ----------------------------------- | ------------------------------------------ |
| **Mask-only**           | (W\odot M) then dense kernels                   | research, ablations                 | no speedup                                 |
| **Structured rewiring** | remove channels/neurons → smaller dense tensors | portable speedups                   | architectural changes                      |
| **Hardware pattern**    | enforce N:M, pack weights, use sparse kernels   | max speed on supported accelerators | constrained pattern, overhead, limited ops |

---

### Case study: NVIDIA 2:4 structured sparsity

* Pattern: in each group of 4 weights, at most **2** nonzero
* Weights are packed and processed by **Sparse Tensor Cores**
* Potential: up to **~2×** throughput vs dense tensor cores (for supported ops/pattern)

> **Key constraint**
> Speedups only materialize when you satisfy the pattern *and* execute with the appropriate sparse kernel.

## 1.6 Learning Sparsity During Training: PSP

Parameterized Structured Pruning (PSP) learns the sparsity pattern by introducing trainable parameters $\alpha_i$ for each structure.

For a structural sub-tensor $w_i$:

$$q_i = w_i \cdot v_i(\alpha_i)$$

with thresholding:

$$
v_i(\alpha_i)=
\begin{cases}
0 & \lvert \alpha_i\rvert <\epsilon \
\alpha_i & \lvert \alpha_i\rvert \ge\epsilon
\end{cases}
$$

Because the threshold is non-differentiable, PSP uses **STE** during backprop to update $\alpha_i. Regularization ($L1$ or $L2$) on $\alpha_i$ encourages sparsity.

> **Empirical note (from lecture narrative)**
> PSP with weight decay can outperform heuristic L1 pruning across many sparsity levels, improving the accuracy–compression trade-off.

## 1.7 Benchmarking and Measurement

Performance reporting must be precise and reproducible.

### Minimum reporting set

* **Quality:** accuracy/loss + evaluation protocol
* **Latency:** p50 and p90 (ms), fixed batch size and shape
* **Throughput:** samples/s (or tokens/s)
* **Memory:** peak usage
* **Energy (optional):** J/inference

### Timing scope (must be explicit)

* **Kernel-only** vs **Layer** vs **End-to-end**
* End-to-end must include:

  * preprocessing / postprocessing
  * transfers
  * packing/format conversion

> **Trustworthy comparisons require end-to-end measurements under fixed workload.**

---

### Measurement protocol (rules of thumb)

1. Fix hardware, dtype, batch size, input shape
2. Warm up 20–100 iterations
3. Measure 200–1000 iterations
4. **Synchronize GPU** (e.g., `torch.cuda.synchronize()`)
5. Report variability (std or p50/p90)

### Common traps

* timing without GPU sync (measures launch, not execution)
* comparing different workloads (invalid)
* ignoring packing overhead
* assuming fewer FLOPs ⇒ less time (false when memory-bound)

---

## 1.8 Case Study: Pruning a Simple CNN on MNIST

Baseline training yields ~99% accuracy. After ~50% pruning:

* without fine-tuning: large accuracy drop
* with fine-tuning: accuracy recovers close to baseline

> **Key takeaway**
> **Fine-tuning is mandatory** to recover accuracy after pruning.

---

## 1.9 Layer-Wise Sparsity Allocation

Uniform sparsity across layers is rarely optimal.

### (1) Sensitivity-based (quality-driven)

* prune robust layers more
* prune sensitive layers less
* objective: maximize accuracy for a global sparsity budget

### (2) Compute-aware (efficiency-driven)

* prioritize pruning where it reduces **real latency**
* target layers with high execution time and kernel support
* objective: maximize speed/energy benefit under accuracy constraint

> **Practical heuristic**
> Prune large FC layers aggressively; be cautious with early conv layers.

---

## 1.10 Conclusion: Compression as a Systems Problem

Pruning effectiveness is determined by the tuple:

$$(\text{data}, \text{architecture}, \text{hardware})$$


### Strategy selection depends on goal

* **min model size:** unstructured sparse storage may be sufficient
* **portable latency gains:** structured rewiring (smaller dense tensors)
* **max accelerator speed:** pattern-constrained sparsity (N:M) + sparse kernels

> **Final principle**
> Sparsity is valuable only when it is **representable, executable, and benchmarked** end-to-end on the target system.

---
