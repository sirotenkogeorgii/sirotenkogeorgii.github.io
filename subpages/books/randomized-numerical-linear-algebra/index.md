---
layout: default
title: Randomized Numerical Algebra
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Section 1: Introduction

## 1.1 Our World

**Numerical linear algebra** (NLA) concerns algorithms for computations on matrices with numerical entries. Originally driven by applications in the physical sciences, it now provides the foundation for vast swaths of applied and computational mathematics. More recently, NLA has also been motivated by developments in machine learning and data science.

There are two factors that increasingly present obstacles to scaling linear algebra computations:

- **Space and power constraints in hardware.** Chips today have billions of transistors packed into a very small amount of space. The breakdown of Dennard's Law and the sunsetting of Moore's Law represent the infamous *power wall* and the post-Moore era of heterogeneous computing. The end result: "more powerful processors" are just scaled-out versions of "less powerful processors." Any algorithm that does not parallelize well is fundamentally limited.
- **NLA's maturity as a field.** Software can only improve so much without algorithmic innovations. Most algorithmic breakthroughs in recent years have required carefully exploiting structures present in specific problems.

This monograph concerns **randomized numerical linear algebra**, or **RandNLA**. Algorithms in this realm offer compelling advantages in a wide variety of settings: unrivaled combinations of efficiency and reliability for massive problems, fine-grained control when balancing accuracy and computational cost, and practicality even with elementary MATLAB or Python implementations.

**Randomized algorithms** in RandNLA are probabilistic in nature. They use randomness as part of their internal logic to make decisions or compute estimates. These algorithms do not presume a distribution over possible inputs, nor do they assume the inputs somehow possess intrinsic uncertainty. Rather, they use randomness as a tool, to find and exploit structures in problem data that would seem "hidden" from the perspective of classical NLA.

### Finding Hidden Structures

Consider the problem of highly overdetermined least squares, i.e., solving

$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2,$$

where $\boldsymbol{A}$ has $m \gg n$ rows. If the columns of $\boldsymbol{A}$ are orthonormal then it can be solved in $O(mn)$ time by setting $\boldsymbol{x} = \boldsymbol{A}^*\boldsymbol{b}$. The trouble is that the columns of $\boldsymbol{A}$ are very unlikely to be orthogonal, and standard algorithms take $O(mn^2)$ time.

However, if we could find an $n \times n$ matrix $\boldsymbol{C}$ for which $\boldsymbol{A}\boldsymbol{C}^{-1}$ was column-orthonormal, we could compute the exact solution in time $O(mn + n^3)$. Randomization can be used to quickly identify a basis in which $\boldsymbol{A}$ is *nearly* column-orthonormal, and one can reliably solve to $\epsilon$-error in time

$$O\!\left(mn\log\tfrac{1}{\epsilon} + n^3\right).$$

### 1.1.1 Four Value Propositions of Randomization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Time Complexity and FLOP Counts)</span></p>

In the sequential RAM model of computing, an algorithm's **time complexity** is its worst-case total number of reads, writes, and elementary arithmetic operations, as a function of input size. It is standard to describe complexity asymptotically with big-$O$ notation.

In NLA, arithmetic operations are presumed to be floating point operations ("flops") by default, and it is common to refer to an algorithm's **flop count** as a function of input size. Flop counts almost always agree asymptotically with time complexity, but flop counts are often given with explicit constant factors.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Value Proposition 1</span><span class="math-callout__name">(Linear or Nearly-Linear Complexity)</span></p>

*For many important linear algebra problems, randomization offers entirely new avenues of computing approximate solutions with linear or nearly-linear complexity.*

For example, a Cholesky decomposition of a dense matrix of order $n$ takes $n^3/3$ flops. For $n = 10{,}000$ this takes about one second on a higher-end laptop. If the semantic notion of problem size is $n$ and one wants to solve a problem ten times as large, the calculation with $n = 100{,}000$ takes over 15 minutes.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(1.1.1)</span></p>

One of RandNLA's success stories is a fast algorithm for computing sparse approximate Cholesky decompositions of so-called *graph Laplacians*. In order to keep the length of this monograph under control, the authors have opted *not* to include algorithms that only apply to sparse matrices. However, they do provide algorithms for computing approximate eigendecompositions of regularized positive semidefinite matrices, and these algorithms can be used to solve linear systems faster than Cholesky in certain applications.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Value Proposition 2</span><span class="math-callout__name">(Fastest Non-Galactic Algorithms)</span></p>

*For a handful of important linear algebra problems, the asymptotically fastest (non-galactic) algorithms for computing accurate solutions are, in fact, randomized.*

The problem of multiplying two $n \times n$ matrices is one of the most fundamental in NLA. The fastest practical method (Strassen's algorithm) runs in time $O(n^{\log_2 7})$. The trouble with "fast" algorithms is that they have massive constants hidden in their big-$O$ complexity. Such algorithms are called *galactic*.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Value Proposition 3</span><span class="math-callout__name">(Reducing Data Movement)</span></p>

*Randomization creates a wealth of opportunities to reduce and redirect data movement. Randomized algorithms based on this principle are significantly faster than the best-available deterministic methods by wall-clock time.*

The RAM model of computing fails to account for the fact that moving data from main memory, through different levels of cache, and onward to processor registers is *much* more expensive than elementary arithmetic on the same data. Even if the time complexities of two algorithms match up to and including constant factors, their performance by wall-clock time can differ by orders of magnitude.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Value Proposition 4</span><span class="math-callout__name">(Lower-Precision Arithmetic)</span></p>

*In RandNLA, it is natural to perform computations on sketches of matrices in lower-precision arithmetic. Depending on how the sketch is constructed, one can be (nearly) certain of avoiding degenerate situations that are known to cause common deterministic algorithms to fail.*

Finite-precision arithmetic and exact arithmetic are different beasts, and this has real consequences for NLA. Certain computations can be performed with lower precision without compromising the accuracy of a final result.

</div>

### 1.1.2 What Is, and Isn't, Subject to Randomness

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sketching Operators and Sketching Distributions)</span></p>

We are concerned with algorithms that use random linear dimension reduction maps called **sketching operators**. The sketching operators used in RandNLA come in a wide variety of forms. They can be as simple as operators for selecting rows or columns from a matrix, and they can be even more complicated than algorithms for computing Fast Fourier Transforms. We refer to a distribution over sketching operators as a **sketching distribution**.

*For the vast majority of RandNLA algorithms, randomness is only used when sampling from the sketching distribution.*

Upon specifying a seed for the random number generator involved in sampling, RandNLA algorithms become every bit as deterministic as classical algorithms.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sketch)</span></p>

When a sketching operator is applied to a large data matrix, it produces a smaller matrix called a **sketch**. A wealth of different outcomes can be achieved through different methods for processing a sketch and using the processed representation downstream.

*Some processing schemes inevitably yield rough approximations to the solution of a given problem. Other processing schemes can lead to high-accuracy approximations, if not exact solutions, under mild assumptions.*

</div>

A popular trend in algorithm analysis is to employ a two-part approach. First, characterize algorithm output in terms of some simple property of the sketch. Second, employ results from random matrix theory to bound the probability that the sketch will possess the desired property.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Confidently Managing Uncertainty)</span></p>

The performance of a numerical algorithm is characterized by the accuracy of its solutions and the cost it incurs to produce those solutions. Most randomized algorithms "gamble" with only one of the two performance metrics, accuracy or cost. Through optional algorithm parameters, users retain fine-grained control over one of these two metrics.

There is a general trend in RandNLA algorithms of becoming more predictable as they are applied to larger problems. At large enough scales, many randomized algorithms are nearly as predictable as deterministic ones.

</div>

## 1.2 This Monograph, from an Astronaut's-Eye View

This monograph started as a development plan for two C++ libraries for RandNLA, primarily working within a shared-memory dense-matrix data model.

- The first library, **RandBLAS**, concerns basic sketching and is the subject of Section 2. The hope is that RandBLAS will grow to become a community standard for RandNLA.
- The second library, **RandLAPACK**, concerns algorithms for traditional linear algebra problems (Sections 3 to 5, on least-squares and optimization, low-rank approximation, and additional possibilities, respectively) and advanced sketching functionality (Sections 6 and 7).

The monograph is written to be modular and accessible, without sacrificing depth. There are almost no technical dependencies across Sections 3 to 7. The authors make liberal use of appendices for proofs, background on special topics, low-level algorithm implementation notes, and high-level algorithm pseudocode.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Drivers and Computational Routines)</span></p>

Most algorithms are designated as either **drivers** or **computational routines**. These terms are borrowed from LAPACK's API. In general, drivers solve higher-level problems than computational routines, and their implementations tend to use a small number of computational routines.

- *Drivers* are only for traditional linear algebra problems.
- *Computational routines* address a mix of traditional linear algebra problems and specialized problems that are only of interest in RandNLA.

Sections 3 and 4 cover drivers *and* the computational routines behind them. Section 5 also covers drivers but at less depth. The advanced sketching functionality in Sections 6 and 7 would *only* be considered for computational routines.

</div>

## 1.3 This Monograph, from a Bird's-Eye View

### Basic Sketching (Section 2)

This section documents work toward developing a RandBLAS standard. It begins with remarks on the Basic Linear Algebra Subprograms (BLAS).

- Section 2.1 addresses high-level design questions for a RandBLAS standard, arriving at the conclusion that it should provide functionality for *data-oblivious sketching* (sketching without consideration to the numerical properties of the data).
- Section 2.2 summarizes a variety of concepts in sketching: geometric interpretations, quality measures, and "standard" properties for the first and second moments of sketching operator distributions.
- Sections 2.3 to 2.5 review three types of sketching operator distributions: dense sketching operators (e.g., Gaussian matrices), sparse sketching operators, and sketching operators based on subsampled fast trigonometric transforms.
- Section 2.6 presents elementary sketching operations not naturally represented by a linear transformation acting only on the columns or only on the rows of a matrix.

### Least Squares and Optimization (Section 3)

This section covers driver-level functionality, discussing drivers *and* computational routines.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Saddle Point Problems)</span></p>

The framework describes all problems in terms of an $m \times n$ data matrix $\boldsymbol{A}$ where $m \ge n$. Given $\boldsymbol{A}$, any pair of vectors $(\boldsymbol{b}, \boldsymbol{c})$ of respective lengths $(m, n)$ can be considered along with a parameter $\mu \ge 0$ to define "primal" and "dual" saddle point problems. The primal problem is always

$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \left\lbrace \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2 + \mu\|\boldsymbol{x}\|_2^2 + 2\boldsymbol{c}^*\boldsymbol{x} \right\rbrace. \qquad (P_\mu)$$

The dual problem takes one of two forms, depending on the value of $\mu$:

$$\min_{\boldsymbol{y} \in \mathbb{R}^m} \left\lbrace \|\boldsymbol{A}^*\boldsymbol{y} - \boldsymbol{c}\|_2^2 + \mu\|\boldsymbol{y} - \boldsymbol{b}\|_2^2 \right\rbrace \quad \text{if } \mu > 0$$

$$\min_{\boldsymbol{y} \in \mathbb{R}^m} \left\lbrace \|\boldsymbol{y} - \boldsymbol{b}\|_2^2 \;:\; \boldsymbol{A}^*\boldsymbol{y} = \boldsymbol{c} \right\rbrace \quad \text{if } \mu = 0. \qquad (D_\mu)$$

Special cases include overdetermined and underdetermined least squares, as well as ridge regression with tall or wide matrices.

For a positive semidefinite linear operator $\boldsymbol{G}$ and a positive parameter $\mu$, the **regularized quadratic** problem is

$$\min_{\boldsymbol{w}} \boldsymbol{w}^*(\boldsymbol{G} + \mu\boldsymbol{I})\boldsymbol{w} - 2\boldsymbol{h}^*\boldsymbol{w}. \qquad (R_\mu)$$

</div>

- **Drivers:** Section 3.2.1 covers *sketch-and-solve* for overdetermined least squares. Sections 3.2.2 and 3.2.3 cover high-accuracy methods using randomization to find a *preconditioner*. Section 3.2.4 reinterprets algorithms for approximate kernel ridge regression (KRR) as sketch-and-solve algorithms.
- **Computational routines:** Section 3.3 covers preconditioner generation for saddle point problems when $m \gg n$, low-memory preconditioners for regularized saddle point problems, and deterministic iterative algorithms needed for randomized preconditioning algorithms.

### Low-rank Approximation (Section 4)

Low-rank approximation problems take the following form: given an $m \times n$ target matrix $\boldsymbol{A}$, compute suitably structured factor matrices $\boldsymbol{E}$, $\boldsymbol{F}$, and $\boldsymbol{G}$ where

$$\hat{\boldsymbol{A}} := \underset{m \times n}{\boldsymbol{E}} \underset{m \times k}{\phantom{x}} \underset{k \times k}{\boldsymbol{F}} \underset{k \times n}{\boldsymbol{G}}$$

approximates $\boldsymbol{A}$, with $k \ll \min\lbrace m, n\rbrace$.

- **Problem classes:** Section 4.1 covers SVD, eigendecomposition, principal component analysis, and *submatrix-oriented decompositions* (CUR, one-sided interpolative decompositions, two-sided interpolative decompositions).
- **Drivers:** Section 4.2 covers RandNLA algorithms for SVD, Hermitian eigendecomposition, CUR, and two-sided interpolative decomposition. Includes *Nystrom approximation* of positive semidefinite matrices.
- **Computational routines:** QB decomposition, column subset selection (CSS) / one-sided ID, power iteration, partial column-pivoted matrix decompositions, and norm estimation methods.

### Further Possibilities for Drivers (Section 5)

- Section 5.1 covers *multi-purpose matrix decompositions*: unpivoted QR, Householder QRCP, preconditioned Cholesky QR, and decompositions known as UTV, URV, QLP.
- Section 5.2 addresses randomized algorithms for the solution of unstructured linear systems.
- Section 5.3 considers estimating the trace of a linear operator, including the Girard-Hutchinson estimator and stochastic Lanczos quadrature for computing $\operatorname{tr}(f(\boldsymbol{B}))$ where $\boldsymbol{B}$ is Hermitian and $f$ is a matrix function.

### Advanced Sketching: Leverage Score Sampling (Section 6)

Leverage scores constitute measures of importance for the rows or columns of a matrix. They can be used to define data-aware sketching operators that implement row or column sampling.

- Section 6.1 introduces three types of leverage scores: standard leverage scores, subspace leverage scores, and ridge leverage scores.
- Section 6.2 covers randomized algorithms for approximating leverage scores.

### Advanced Sketching: Tensor Product Structures (Section 7)

Tensor computations are the domain of *multilinear algebra*. This section reviews efficient methods for sketching matrices with Kronecker product or Khatri-Rao product structures. Section 7.2.5 covers data-aware sketching methods based on leverage score sampling, including methods to efficiently sample from the *exact* leverage score distributions of tall matrices with Kronecker and Khatri-Rao product structures.

## 1.5 Notation and Terminology

### Matrices and Vectors

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Matrix Notation and Conventions)</span></p>

Let $\boldsymbol{A}$ be an $m \times n$ matrix or linear operator.

- $\boldsymbol{A}^*$ denotes its **adjoint** (transpose, in the real case).
- $\boldsymbol{A}^\dagger$ denotes its **pseudo-inverse**.
- $\boldsymbol{A}$ is called **Hermitian** if $\boldsymbol{A}^* = \boldsymbol{A}$ and **positive semidefinite** (psd) if it is Hermitian and all of its eigenvalues are nonnegative.
- Vectors have column orientations by default, so the standard inner product of two vectors $\boldsymbol{u}, \boldsymbol{v}$ is $\boldsymbol{u}^*\boldsymbol{v}$.
- A vector of length $n$ is called an *$n$-vector*.
- An $m \times n$ matrix called "tall" means $m \ge n$; "very tall" means $m \gg n$. Analogous conventions for "wide" and "very wide."

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(QR Decomposition)</span></p>

For $m \ge n$, a **QR decomposition** of $\boldsymbol{A}$ consists of an $m \times n$ column-orthonormal matrix $\boldsymbol{Q}$ and an upper-triangular matrix $\boldsymbol{R}$ for which $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{R}$. This is typically the *economic* QR decomposition. If $\boldsymbol{A}$ has rank $k < \min(m, n)$, then $\boldsymbol{Q}$ is $m \times k$ and $\boldsymbol{R}$ is $k \times n$.

**QR decomposition with column pivoting (QRCP):** If $J = (j_1, \ldots, j_n)$ is a permutation of $[\![n]\!]$, then $\boldsymbol{A}[:, J] = [\boldsymbol{a}_{j_1}, \boldsymbol{a}_{j_2}, \ldots, \boldsymbol{a}_{j_n}]$. QRCP produces an index vector $J$ and factors $(\boldsymbol{Q}, \boldsymbol{R})$ that provide a QR decomposition of $\boldsymbol{A}[:, J]$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Singular Value Decomposition)</span></p>

Let $\boldsymbol{A}$ have rank $k$. Its **singular value decomposition** (SVD) takes the form $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^*$, where the matrices $(\boldsymbol{U}, \boldsymbol{V})$ have $k$ orthonormal columns and $\boldsymbol{\Sigma} = \operatorname{diag}(\sigma_1, \ldots, \sigma_k)$ is a square matrix with sorted entries $\sigma_1 \ge \cdots \ge \sigma_k > 0$.

The SVD can also be written as a sum of rank-one matrices: $\boldsymbol{A} = \sum_{i=1}^{k} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^*$, where $(\boldsymbol{u}_i, \boldsymbol{v}_i)$ are the $i$-th columns of $(\boldsymbol{U}, \boldsymbol{V})$ respectively. This is typically the *compact* SVD.

</div>

### Notation Table

| Symbol | Meaning |
| --- | --- |
| $A_{ij}$ or $\boldsymbol{A}[i,j]$ | $(i,j)$-th entry of matrix $\boldsymbol{A}$ |
| $\boldsymbol{a}_i$ or $\boldsymbol{A}[:,i]$ | $i$-th column of $\boldsymbol{A}$ |
| $[\![m]\!]$ | index set of integers from 1 to $m$ |
| $\boldsymbol{A}[I,:]$ | submatrix of (permuted) rows of $\boldsymbol{A}$ |
| $\boldsymbol{A}[:,J]$ | submatrix of (permuted) columns of $\boldsymbol{A}$ |
| $\boldsymbol{S}$ | sketching operator |
| $\boldsymbol{I}_k$ | identity matrix of size $k \times k$ |
| $\|\boldsymbol{x}\|_2$ | Euclidean norm of a vector $\boldsymbol{x}$ |
| $\|\boldsymbol{A}\|_2$ | spectral norm of $\boldsymbol{A}$ |
| $\|\boldsymbol{A}\|_{\mathrm{F}}$ | Frobenius norm of $\boldsymbol{A}$ |
| $\operatorname{cond}(\boldsymbol{A})$ | Euclidean condition number of $\boldsymbol{A}$ |
| $\lambda_i(\boldsymbol{A})$ | $i$-th largest eigenvalue of $\boldsymbol{A}$ |
| $\sigma_i(\boldsymbol{A})$ | $i$-th largest singular value of $\boldsymbol{A}$ |
| $\boldsymbol{A}^*$ | adjoint (transpose, in the real case) of $\boldsymbol{A}$ |
| $\boldsymbol{A}^\dagger$ | Moore-Penrose pseudoinverse of $\boldsymbol{A}$ |
| $\boldsymbol{A}^{1/2}$ | Hermitian matrix square root |
| $\boldsymbol{A} \preceq \boldsymbol{B}$ | $\boldsymbol{B} - \boldsymbol{A}$ is positive semidefinite |
| $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{R}$ | QR decomposition (economic, by default) |
| $(\boldsymbol{Q}, \boldsymbol{R}, J) = \operatorname{qrcp}(\boldsymbol{A})$ | QR with column-pivoting: $\boldsymbol{A}[:,J] = \boldsymbol{Q}\boldsymbol{R}$ |
| $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^*$ | singular value decomposition (compact, by default) |
| $\boldsymbol{R} = \operatorname{chol}(\boldsymbol{G})$ | upper triangular Cholesky factor of $\boldsymbol{G} = \boldsymbol{R}^*\boldsymbol{R}$ |

### Probability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probability Conventions)</span></p>

- A **Rademacher** random variable uniformly takes values in $\lbrace +1, -1 \rbrace$.
- "iid" expands to *independent and identically distributed*.
- $X \sim \mathcal{D}$: $X$ is a random variable following a distribution $\mathcal{D}$.
- $\mathbb{E}[\boldsymbol{X}]$: expected value of a random matrix $\boldsymbol{X}$.
- $\operatorname{var}(X)$: variance of a random variable $X$.
- $\Pr\lbrace E \rbrace$: probability of the event $E$.
- When we say a matrix "randomly" performs some operation, the matrix itself only performs deterministic calculations — randomness only comes into play when the matrix is first constructed.
- Unqualified use of the term "random" before performing an action with a finite set of outcomes (such as sampling components from a vector, applying a permutation, etc.) means the randomness is uniform over the space of possible actions.

</div>

# Section 2: Basic Sketching

The **BLAS** (Basic Linear Algebra Subprograms) were originally a collection of Fortran routines for computations including vector scaling, vector addition, and applying Givens rotations. They were later extended to operations such as matrix-vector multiplication, triangular solves, matrix-matrix multiplication, block triangular solves, and symmetric rank-$k$ updates. These routines have been organized into three *levels* called BLAS 1, BLAS 2, and BLAS 3.

Over the years the BLAS have evolved into a *community standard*, with implementations targeting different machine architectures in many programming languages. This section summarizes progress on the design of a "**RandBLAS**" library, which is to be to RandNLA as BLAS is to classical NLA.

## 2.1 A High-Level Plan

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Premise</span><span class="math-callout__name">(RandBLAS)</span></p>

The RandBLAS's defining purpose should be to facilitate implementation of high-level RandNLA algorithms.

</div>

This premise suggests that RandBLAS should be concerned with **data-oblivious sketching** — sketching without consideration to the numerical properties of a dataset. Three categories of operations are identified:

1. Sampling a random sketching operator from a prescribed distribution.
2. Applying a sampled sketching operator to a data matrix.
3. Sketching that is not naturally expressed as applying a single linear operator to a data matrix.

These categories are somewhat analogous to BLAS 1, BLAS 2, and BLAS 3. Note that data-oblivious sketching is not the only kind of value in RandNLA. *Data-aware* sketching operators such as those derived from power iteration are extremely important for low-rank approximation, and methods based on leverage scores are useful for kernel ridge regression and certain tensor computations.

### 2.1.1 Random Number Generation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Counter-Based Random Number Generators)</span></p>

For reproducibility's sake, RandBLAS should include a specification for *random number generators* (RNGs). The recommended approach is **counter-based random number generators** (CBRNGs). A CBRNG returns a random number upon being called with two integer parameters: the **counter** and the **key**. The time required does not depend on either parameter.

- A serial application can set the key at the outset and never change it.
- Parallel applications can use different keys across different threads.
- Sequential calls with a fixed key should use different values for the counter.
- For a fixed key, a CBRNG with a $p$-bit integer counter defines a stream of random numbers with period length $2^p$.

CBRNGs are preferable to traditional state-based RNGs such as the Mersenne Twister because they maximize flexibility in the order in which a sketching operator is generated. For example, the $(i,j)$-th entry of a dense $d \times m$ sketching operator can be generated with counter $c + (i + dj)$, making these computations embarrassingly parallel.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Shift-Register RNGs)</span></p>

The CBRNGs in Random123 are significantly more expensive than the state-based shift-register RNGs developed by Blackman and Vigna. In fact, their generators are so fast that one can implement a method for applying a Gaussian sketching operator to a sparse matrix that beats Intel MKL's sparse-times-dense matrix multiplication methods. However, in applications where processing the sketch downstream was more expensive than computing the sketch itself, CBRNGs' longer runtimes were inconsequential. This longer runtime can be viewed as a price we pay for prioritizing reproducibility of sketching across compute environments with different levels of parallelism.

*State-based RNGs may be preferable to CBRNGs if sketching is a bottleneck in a RandNLA algorithm and where the cost of random number generation decisively affects the cost of sketching.*

</div>

### 2.1.2 Portability, Reproducibility and Exception Handling

RandBLAS should have a procedural API and make use of no special data structures beyond elementary structs. RandLAPACK will expose RandBLAS functionality through a suitable object-oriented linear operator interface.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(2.1.1 — Bitwise Reproducibility)</span></p>

Making the BLAS bitwise reproducible is challenging because floating-point addition is not associative, and the order of summation can vary depending on the use of parallelism, vectorization, and other matters. Summation algorithms that guarantee bitwise reproducibility do exist. These algorithms may become practical on hardware that implements the latest IEEE 754 floating point standard, which includes a recommended instruction for bitwise-reproducible summation. However, these matters are left to future work.

</div>

A key source of exceptions in NLA is the presence of `NaN`s or `Inf`s in problem data. Extremely sparse sketching matrices might not even read every entry of a data matrix, and so they might miss a `NaN` or `Inf`. Those routines will be clearly marked as carrying this risk. The majority of routines in RandBLAS and RandLAPACK will *not* carry this risk: they will propagate `NaN`s and `Inf`s.

## 2.2 Helpful Things to Know About Sketching

The purpose of sketching is to enact dimension-reduction so that computations of interest can be performed on a smaller matrix called a *sketch*.

- Sketching operators applied to the *left* of a data matrix must *must be wide* (more columns than rows).
- Sketching operators applied to the *right* of a data matrix *must be tall* (more rows than columns).

In left-sketching we require that $\boldsymbol{S}\boldsymbol{A}$ has fewer rows than $\boldsymbol{A}$, and in right-sketching we require that $\boldsymbol{A}\boldsymbol{S}$ has fewer columns than $\boldsymbol{A}$.

### 2.2.1 Geometric Interpretations of Sketching

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Left-Sketching and Right-Sketching)</span></p>

**Left-sketching:** Sketching $\boldsymbol{A}$ from the left preserves its number of columns. It is suitable for estimating *right* singular vectors. We often interpret a left-sketch $\boldsymbol{S}\boldsymbol{A}$ as a compression of the range of $\boldsymbol{A}$ to a space of lower ambient dimension. In the special case when $\operatorname{rank}(\boldsymbol{S}\boldsymbol{A}) = \operatorname{rank}(\boldsymbol{A})$, the quality of the compression is completely independent from the spectrum of $\boldsymbol{A}$.

**Right-sketching:** Sketching $\boldsymbol{A}$ from the right preserves its number of rows, and is suitable for estimating *left* singular vectors. The right-sketch $\boldsymbol{A}\boldsymbol{S}$ can be interpreted as a sample from the range of $\boldsymbol{A}$ using a test matrix $\boldsymbol{S}$. When $\operatorname{rank}(\boldsymbol{A}\boldsymbol{S}) \ll \operatorname{rank}(\boldsymbol{A})$ it is a "lossy sample" from a much larger "population."

**Equivalence:** Left-sketching and right-sketching can be reduced to one another by replacing $\boldsymbol{A}$ and $\boldsymbol{S}$ by their adjoints. A left-sketch $\boldsymbol{S}\boldsymbol{A}$ is equivalent to the right-sketch $\boldsymbol{A}^*\boldsymbol{S}^*$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Canonical Extension of Sketching Distributions)</span></p>

If $\mathcal{D}_{d,m}$ is a distribution over wide $d \times m$ sketching operators, it is *canonically extended* to a distribution over tall $n \times d$ sketching operators by sampling $\boldsymbol{T}$ from $\mathcal{D}_{d,n}$ and then returning the adjoint $\boldsymbol{S} = \boldsymbol{T}^*$.

Since "data matrices" are typically $m \times n$, a typical left-sketching operator requires $m$ columns, and a typical right-sketching operator requires $n$ rows.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Embedding and Sampling Regimes)</span></p>

Sketching in the **embedding regime** is the use of a sketching operator that is *larger* than the data to be sketched. Sketching in the **sampling regime** is the use of a sketching operator that is *far smaller* than the data to be sketched.

The size of an operator (or matrix) is quantified by the product of its number of rows and number of columns.

- Sketching in the **embedding regime** is nearly universal in randomized algorithms for least squares and related problems (Section 3).
- Sketching in the **sampling regime** is the foundation of randomized algorithms for low-rank approximation (Section 4).

We tend to see sketching in the embedding regime happen from the left, and sketching in the sampling regime happen from the right (though these are tendencies of exposition, not absolute rules).

</div>

### 2.2.2 Sketch Quality

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subspace Embedding)</span></p>

Let $\boldsymbol{S}$ be a $d \times m$ sketching operator and $L$ be a linear subspace of $\mathbb{R}^m$. We say that $\boldsymbol{S}$ **embeds** $L$ into $\mathbb{R}^d$ and that it does so with **distortion** $\delta \in [0,1]$ if $\boldsymbol{x} \in L$ implies

$$(1 - \delta)\|\boldsymbol{x}\|_2 \le \|\boldsymbol{S}\boldsymbol{x}\|_2 \le (1 + \delta)\|\boldsymbol{x}\|_2.$$

We often call such an operator a **$\delta$-embedding**.

When $L$ is the range of a matrix $\boldsymbol{A}$, $\boldsymbol{S}$ is a $\delta$-embedding for $L$ if and only if the following two-sided linear matrix inequality holds:

$$(1 - \delta)^2 \boldsymbol{A}^*\boldsymbol{A} \preceq (\boldsymbol{S}\boldsymbol{A})^*(\boldsymbol{S}\boldsymbol{A}) \preceq (1 + \delta)^2 \boldsymbol{A}^*\boldsymbol{A}.$$

The distortion of $\boldsymbol{S}$ as an embedding for $\operatorname{range}(\boldsymbol{A})$ is a measurement of how well the Gram matrix of $\boldsymbol{S}\boldsymbol{A}$ approximates that of $\boldsymbol{A}$.

Note: for $\boldsymbol{S}$ to be a subspace embedding for $L$ it is necessary that $d \ge \dim(L)$, so subspace embeddings can only be achieved when "sketching in the embedding regime."

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Effective Distortion)</span></p>

Subspace embedding distortion is the most common measure of sketch quality, but it is not invariant under scaling of $\boldsymbol{S}$ (i.e., replacing $\boldsymbol{S} \leftarrow t\boldsymbol{S}$ for $t \ne 0$). This is a significant limitation since many RandNLA algorithms *are* invariant under scaling of $\boldsymbol{S}$.

The **effective distortion** of a sketching operator $\boldsymbol{S}$ for a subspace $L$ is

$$\mathscr{D}_e(\boldsymbol{S}; L) = \inf\lbrace \delta \;:\; 0 \le \delta \le 1,\; 0 < t,\; t\boldsymbol{S} \text{ is a } \delta\text{-embedding for } L \rbrace.$$

This is the minimum distortion that *any* sketching operator $t\boldsymbol{S}$ can achieve for $L$, optimizing over $t > 0$. Effective distortion is far more relevant for least squares and optimization, and it will always be no larger than the standard notion of distortion.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Oblivious Subspace Embedding — OSE Property)</span></p>

Data-oblivious subspace embeddings (OSEs) were first used in RandNLA by Sarlos. Consider a distribution $\mathcal{D}$ over wide $d \times m$ matrices. We say that $\mathcal{D}$ has the **OSE property with parameters $(\delta, n, p)$** if, for every $n$-dimensional subspace $L \subset \mathbb{R}^m$, we have

$$\Pr\lbrace \boldsymbol{S} \sim \mathcal{D} \text{ is a } \delta\text{-embedding for } L \rbrace \ge 1 - p.$$

Theoretical analyses of sketching distributions often concern bounding $d$ as a function of $(\delta, n, p)$ to ensure $\mathcal{D}$ satisfies the OSE property. Naturally, all else equal, we would like to achieve the OSE property for smaller values of $d$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Johnson-Lindenstrauss Embedding)</span></p>

Let $\boldsymbol{S}$ be a $d \times m$ sketching operator and $L$ be a finite point set in $\mathbb{R}^m$. We say that $\boldsymbol{S}$ is a **Johnson-Lindenstrauss embedding** (or "JL embedding") for $L$ with distortion $\delta$ if, for all distinct $\boldsymbol{x}, \boldsymbol{y}$ in $L$, we have

$$1 - \delta \le \frac{\|\boldsymbol{S}(\boldsymbol{x} - \boldsymbol{y})\|_2^2}{\|\boldsymbol{x} - \boldsymbol{y}\|_2^2} \le 1 + \delta.$$

The JL Lemma (as the result is now known) is remarkable for two reasons:
1. The requisite value for $d$ did not depend on the ambient dimension $m$ and was only logarithmic in $|L|$.
2. The construction of the transformation $\boldsymbol{S}$ was *data-oblivious* — a scaled orthogonal projection.

</div>

### 2.2.3 (In)essential Properties of Sketching Distributions

Distributions $\mathcal{D}$ over wide sketching operators are typically designed so that, for $\boldsymbol{S} \sim \mathcal{D}$, the mean and covariance matrices are

$$\mathbb{E}[\boldsymbol{S}] = \boldsymbol{0} \qquad \text{and} \qquad \mathbb{E}[\boldsymbol{S}^*\boldsymbol{S}] = \boldsymbol{I}.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Scale-Agnosticism)</span></p>

The property $\mathbb{E}[\boldsymbol{S}] = \boldsymbol{0}$ is important in RandNLA. However, there is flexibility in the scale of the covariance matrix — in most situations it suffices for the covariance matrix to be a scalar multiple of the identity.

$\mathbb{E}[\boldsymbol{S}^*\boldsymbol{S}] = \boldsymbol{I}$ is equivalent to $\boldsymbol{S}$ preserving squared Euclidean norms in expectation. However, the vast majority of algorithms in this monograph rely on sketching preserving *relative norms*, in the sense that $\|\boldsymbol{S}\boldsymbol{u}\|_2/\|\boldsymbol{S}\boldsymbol{v}\|_2$ should be close to $\|\boldsymbol{u}\|_2/\|\boldsymbol{v}\|_2$. Such a property is clearly unaffected if $\boldsymbol{S}$ is replaced by $t\boldsymbol{S}$ for some $t \ne 0$.

This section therefore ignores most matters of scaling that is applied equally to all entries of a sketching operator, and regularly describes sketching operators as having entries in $[-1, 1]$ even though it is more common to have entries in $[-v, v]$ for some positive $v$ (which is set to achieve an identity covariance matrix).

</div>

**Partial sketching.** Suppose we want to approximate $\boldsymbol{H} = \boldsymbol{A}^*\boldsymbol{A} + \boldsymbol{G}$ (where $\boldsymbol{G}$ is $n \times n$ psd and $\boldsymbol{A}$ is very tall $m \times n$) by a *partial sketch*

$$\boldsymbol{H}_{\text{sk}} = (\boldsymbol{S}_o\boldsymbol{A})^*(\boldsymbol{S}_o\boldsymbol{A}) + \boldsymbol{G},$$

where $\boldsymbol{S}_o$ is a $d \times m$ sketching operator. One can go beyond distortion by lifting to a higher-dimensional space. Letting $\sqrt{\boldsymbol{G}}$ denote the Hermitian square root, define the augmented sketching operator and augmented data matrix

$$\boldsymbol{S} = \begin{bmatrix} \boldsymbol{S}_o & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{I} \end{bmatrix} \qquad \text{and} \qquad \boldsymbol{A}_{\boldsymbol{G}} = \begin{bmatrix} \boldsymbol{A} \\ \sqrt{\boldsymbol{G}} \end{bmatrix}.$$

This lets us express $\boldsymbol{H} = \boldsymbol{A}_{\boldsymbol{G}}^*\boldsymbol{A}_{\boldsymbol{G}}$ and $\boldsymbol{H}_{\text{sk}} = (\boldsymbol{S}\boldsymbol{A}_{\boldsymbol{G}})^*(\boldsymbol{S}\boldsymbol{A}_{\boldsymbol{G}})$. The statistical properties of $\boldsymbol{H}_{\text{sk}}$ as an approximation to $\boldsymbol{H}$ can then be understood in terms of how $\boldsymbol{S}$ preserves (or distorts) the range of $\boldsymbol{A}_{\boldsymbol{G}}$.

**Row sampling from block matrices.** A ridge regression problem with tall $m \times n$ data matrix $\boldsymbol{A}$ and regularization parameter $\mu$ can be lifted to an ordinary least squares problem with data matrix $\boldsymbol{A}_\mu := [\boldsymbol{A};\; \sqrt{\mu}\boldsymbol{I}]$. Sketching $\boldsymbol{A}_\mu$ by a row sampling operator $\boldsymbol{S}$ naturally produces $\boldsymbol{S}\boldsymbol{A} = [\boldsymbol{S}_o\boldsymbol{A};\; \sqrt{\mu}\boldsymbol{I}]$ — the map $\boldsymbol{A}_\mu \mapsto \boldsymbol{S}\boldsymbol{A}_\mu$ would *not* sample uniformly at random from $\boldsymbol{A}_\mu$, even if $\boldsymbol{S}_o$ sampled rows of $\boldsymbol{A}$ uniformly. Partial sketching is thus a way of incorporating non-uniform row sampling into other sketching distributions.

## 2.3 Dense Sketching Operators

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dense Sketching Operators)</span></p>

The RandBLAS should provide methods for sampling sketching operators with iid entries drawn from distinguished distributions. Three types stand out:

- **Rademacher sketching operators:** entries are $\pm 1$ with equal probability.
- **Uniform sketching operators:** entries are uniform over $[-1, 1]$.
- **Gaussian sketching operators:** entries follow the standard normal distribution.

The RandBLAS should also support sampling row-orthonormal or column-orthonormal matrices uniformly at random from the set of all such matrices (**Haar operators**).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Equivalence of iid-Dense Operators)</span></p>

The theoretical results for Gaussian operators are especially strong. However, there is little practical difference in the performance of RandNLA algorithms between any of the three entrywise iid operators. This practical equivalence has theoretical support through *universality principles* in high-dimensional probability.

RandNLA algorithms tend to be very robust to the quality of the random number generator. As a result, it is not necessary to sample from the Gaussian distribution with high statistical accuracy. This is due in part to the success of *sub-Gaussian distributions* as an analysis framework. From an implementation standpoint, there is likely no need to sample from the Gaussian distribution beyond single precision — even half-precision may suffice for practical purposes.

</div>

**Sampling iid-dense sketching operators.** Sampling from the Rademacher or uniform distributions is the most basic operation of random number generators. For Gaussian sampling, two transformations are of interest: Box-Muller (easy to implement and parallelize) and the Ziggurat transform (far more efficient on a single thread but does not parallelize well).

**Applying iid-dense sketching operators.** If a dense sketching operator is realized explicitly in memory then it can (and should) be applied by an appropriate BLAS function, most likely `gemm`. There is potential for reduced memory or communication requirements if a sketching operator is applied without ever fully allocating it in-memory, using counter-based random number generators.

**Sampling and applying Haar operators.** For left-sketching, the Haar distribution is the uniform distribution over row-orthonormal matrices. The naive approach requires $O(d^2 m)$ time. A more efficient approach — constructing the operator as a composition of suitable Householder reflectors — costs only $O(dm)$ time and has the secondary benefit of not needing to form the sketching operator explicitly.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intended Use-Cases for Dense Operators)</span></p>

Using the embedding/sampling regime terminology from Section 2.2.1, dense sketching operators are commonly used for "sketching in the sampling regime." They are the workhorses of randomized algorithms for low-rank approximation. They also have applications in certain randomized algorithms for ridge regression and some full-rank matrix decomposition problems.

These distributions are much less useful for sketching dense matrices "in the embedding regime" — they are more expensive to apply to dense matrices than many other types of sketching operators. They *might* be of interest in the embedding regime if applied to sparse or otherwise structured data matrices.

</div>

## 2.4 Sparse Sketching Operators

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Taxonomy of Sparse Sketching Operators)</span></p>

We use the term **short-axis vector** in reference to the columns of a wide matrix or rows of a tall matrix. The term **long-axis vector** is defined analogously, as the rows of a wide matrix or columns of a tall matrix. Three families of sparse sketching operators:

- **Short-axis-sparse (SASOs):** The short-axis vectors are independent of one another. Each short-axis vector has a fixed (and very small) number of nonzeros. Typically, the indices of the nonzeros in each short-axis vector are sampled uniformly without replacement.
- **Long-axis-sparse (LASOs):** The long-axis vectors are independent of one another. For a given long-axis vector, the indices for its nonzeros are sampled with replacement according to a prescribed probability distribution. The value of a given nonzero is affected by the number of times its index appears in the sample.
- **Iid-sparse:** Mathematically, these can be described as starting with an iid-dense sketching operator and "zeroing-out" entries in an iid-manner with some high probability. These are *not* recommended for the RandBLAS due to weaker theoretical guarantees and less predictable structure.

</div>

### 2.4.1 Short-Axis-Sparse Sketching Operators (SASOs)

SASOs include sketching operators known as sparse Johnson-Lindenstrauss transforms, the Clarkson-Woodruff transform, CountSketch, and OSNAPs.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SASO Construction)</span></p>

The short-axis vectors of a SASO should be independent of one another. One can select the locations of nonzero elements in two ways. For a wide $d \times m$ operator, we can:

1. Sample $k$ indices uniformly from $[\![d]\!]$ without replacement, once for each column, or
2. Divide $[\![d]\!]$ into $k$ contiguous subsets of equal size, and then for each column select one index from each of the $k$ index sets.

The nonzero values in a SASO's short-axis vector are canonically independent Rademachers. Alternatively, they can be drawn from other sub-Gaussian distributions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(2.4.1 — Naming Conventions)</span></p>

The concept of what we would call a "wide SASO" is referred to in the literature as an *OSNAP*. We prefer "SASO" over "OSNAP" for two reasons. First, it pairs naturally with the abbreviation *LASO* for long-axis-sparse sketching operators. Second, the literature consistently describes OSNAPs as having a fixed number of nonzeros per column, which is only appropriate for left sketching.

</div>

### 2.4.2 Long-Axis-Sparse Sketching Operators (LASOs)

This category includes row and column sampling, LESS embeddings, and LESS-uniform operators.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(LASO Construction)</span></p>

In the wide case, a LASO has independent rows and a fixed upper bound on the number of nonzeros per row. All rows are sampled with reference to a distribution $\boldsymbol{p}$ over $[\![m]\!]$ (which can be uniform) and a positive integer $k$. Construction begins by sampling $t_1, \ldots, t_k$ from $[\![m]\!]$ with replacement according to $\boldsymbol{p}$. Then we initialize

$$\boldsymbol{S}[i,:] = \frac{1}{\sqrt{dk}} \left(\sqrt{\frac{b_1}{p_1}}, \ldots, \sqrt{\frac{b_m}{p_m}}\right),$$

where $b_j$ is the number of times the index $j$ appeared in the sample $(t_1, \ldots, t_k)$. We finish constructing the row by multiplying each nonzero entry by an iid copy of a mean-zero random variable of unit variance (e.g., a standard Gaussian). Such a LASO will have at most $k$ nonzeros per row and hence at most $dk$ nonzeros in total — much smaller than $mk$ nonzeros required by a SASO with the same parameters.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(2.4.2 — Scale and Leverage Scores)</span></p>

The scaling factor $1/\sqrt{dk}$ appearing in the LASO initialization is the same for all rows (i.e., for each long-axis vector). This factor is necessary so that once the nonzeros in $\boldsymbol{S}$ are multiplied by mean-zero unit-variance random variables, we have $\mathbb{E}[\boldsymbol{S}^*\boldsymbol{S}] = \boldsymbol{I}_m$. This scaling matters when one cares about subspace embedding distortion or when one is only sketching a portion of the problem data.

The quality of sketches produced by LASOs when $\boldsymbol{p}$ is uniform depends on the **leverage scores** of the matrix. If $\boldsymbol{p}$ is the leverage score distribution of some matrix then the sketching operator is known as a **Leverage Score Sparsified (LESS)** embedding.

</div>

## 2.5 Subsampled Fast Trigonometric Transforms

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fast Trigonometric Transforms)</span></p>

**Fast trigonometric transforms** (or *fast trig transforms*) are orthogonal or unitary operators that take $m$-vectors to $m$-vectors in $O(m \log m)$ time or better. The most important examples are:

- **Discrete Fourier Transform** (for complex-valued inputs)
- **Discrete Cosine Transform** (for real-valued inputs)
- **Walsh-Hadamard Transform** (exists only when $m$ is a power of two; equivalent to a Kronecker product of $\log_2 m$ Discrete Fourier Transforms of size $2 \times 2$; involves no multiplications and no branching)

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subsampled Randomized Fast Trig Transforms — SRFTs)</span></p>

Formally, a $d \times m$ SRFT takes the form

$$\boldsymbol{S} = \sqrt{m/d}\;\boldsymbol{R}\boldsymbol{F}\boldsymbol{D},$$

where $\boldsymbol{D}$ is a diagonal matrix of independent Rademachers, $\boldsymbol{F}$ is a fast trig transform that maps $m$-vectors to $m$-vectors, and $\boldsymbol{R}$ randomly samples $d$ components from an $m$-vector. For added robustness one can define SRFTs slightly differently, replacing $\boldsymbol{S}$ by $\boldsymbol{S}\boldsymbol{\Pi}$ for a permutation matrix $\boldsymbol{\Pi}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(SRFTs)</span></p>

- A $d \times m$ SRFT can be applied to an $m \times n$ matrix in as little as $O(mn \log d)$ time.
- Theoretical guarantees for SRFTs are usually established assuming $\boldsymbol{F}$ is the Walsh-Hadamard transform, and do not rely on tuning sparsity parameters.
- **Drawback:** SRFTs are notoriously difficult to implement efficiently. Even their best-case $O(mn \log d)$ complexity is higher than the $O(mnk)$ complexity of a SASO that is wide with $k \ll \log d$ nonzeros per column.
- **Memory advantage:** If one overwrites $\boldsymbol{A}$ by the $m \times n$ matrix $\boldsymbol{B} := \sqrt{m/d}\;\boldsymbol{F}\boldsymbol{D}\boldsymbol{A}$ in $O(mn \log m)$ time, then $\boldsymbol{S}\boldsymbol{A}$ can be accessed as a submatrix of rows of $\boldsymbol{B}$ without losing access to $\boldsymbol{A}$ or $\boldsymbol{A}^*$ as linear operators.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Block SRFTs)</span></p>

Let $p$ be a positive integer, $r = m/p$ be greater than $d$, and $\boldsymbol{R}$ be a matrix that randomly samples $d$ components from an $r$-vector. For each index $i \in [\![p]\!]$, we introduce a $d \times r$ sketching operator

$$\boldsymbol{S}_i = \sqrt{r/d}\;\boldsymbol{D}_i^{\text{post}}\boldsymbol{R}\boldsymbol{F}\boldsymbol{D}_i^{\text{pre}},$$

where $\boldsymbol{D}_i^{\text{post}}$ and $\boldsymbol{D}_i^{\text{pre}}$ are diagonal matrices filled with independent Rademachers. The **block SRFT** is defined columnwise as $\boldsymbol{S} = [\boldsymbol{S}_1 \;\; \boldsymbol{S}_2 \;\; \ldots \;\; \boldsymbol{S}_p]$.

Block SRFTs can effectively leverage parallel hardware using *serial* implementations of the fast trig transform. For $\boldsymbol{A}$ distributed block row-wise among $p$ processors, $\boldsymbol{S}\boldsymbol{A} = \sum_{i \in [\![p]\!]} \boldsymbol{S}_i \boldsymbol{A}_i$, where each $\boldsymbol{S}_i \boldsymbol{A}_i$ is computed locally.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(2.5.1 — Navigating the SRFT Literature)</span></p>

The development of SRFTs began with *fast Johnson-Lindenstrauss transforms* (FJLTs), which replace the matrix "$\boldsymbol{R}$" in the SRFT construction by a particular type of sparse matrix. FJLTs were first used in RandNLA for least squares and low-rank approximation. The jump from FJLTs to SRFTs was made independently by Drineas et al. and Woolfe et al. for usage in least squares and low-rank approximation, respectively.

SRFTs are sometimes called *randomized orthonormal systems* or (with slight abuse of terminology) FJLTs. A type of "SRFTs without subsampling" has been successfully used to approximate Gaussian matrices needed in random features approaches to kernel ridge regression.

</div>

## 2.6 Multi-Sketch and Quadratic-Sketch Routines

For many years, the performance bottleneck in NLA algorithms has been data movement, rather than FLOPs performed on the data. The fastest randomized algorithms for low-rank matrix approximation involve computing multiple sketches of a data matrix. Such *multi-sketching* presents new challenges and opportunities in the development of optimized implementations with minimal data movement.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Types of Multi-Sketching)</span></p>

The RandBLAS should include functionality for at least three types of multi-sketching:

1. Generate $\boldsymbol{S}$ and compute $\boldsymbol{Y}_1 = \boldsymbol{A}\boldsymbol{S}$ and $\boldsymbol{Y}_2 = \boldsymbol{A}^*\boldsymbol{A}\boldsymbol{S}$.
2. Generate independent $\boldsymbol{S}_1, \boldsymbol{S}_2$, and compute $\boldsymbol{Y}_1 = \boldsymbol{A}\boldsymbol{S}_1$ and $\boldsymbol{Y}_2 = \boldsymbol{S}_2\boldsymbol{A}$. Algorithms using this typically need to retain $\boldsymbol{S}_1$ or $\boldsymbol{S}_2$ for later use.
3. Generate independent $\boldsymbol{S}_1, \boldsymbol{S}_2, \boldsymbol{S}_3, \boldsymbol{S}_4$, and compute $\boldsymbol{Y}_1 = \boldsymbol{A}\boldsymbol{S}_1$, $\boldsymbol{Y}_2 = \boldsymbol{S}_2\boldsymbol{A}$, and $\boldsymbol{Y}_3 = \boldsymbol{S}_3\boldsymbol{A}\boldsymbol{S}_4$.

In all cases, these primitives are only used for sketching in the sampling regime.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quadratic Sketching)</span></p>

By "quadratic sketch," we mean a linear sketch of $\boldsymbol{A}^*\boldsymbol{A}$ or $\boldsymbol{A}\boldsymbol{A}^*$. This operation is ubiquitous in algorithms for low-rank approximation. As with multi-sketching, all uses of quadratic sketching entail sketching in the sampling regime.

It is not possible to fundamentally accelerate this kind of sketching by using fast sketching operators, so it would be reasonable for RandBLAS's quadratic sketching methods to only support dense sketching operators. In essence, this asks for a high-performance implementation of the composition of the BLAS 3 functions `syrk` and `gemm`: $(\boldsymbol{A}, \boldsymbol{S}) \mapsto \boldsymbol{A}\boldsymbol{A}^*\boldsymbol{S}$.

</div>

# Section 3: Least Squares and Optimization

Numerical linear algebra is the backbone of the most widely-used algorithms for continuous optimization. The connections between optimization and linear algebra are often introduced with *least squares problems*. By adopting a suitable perspective, one can use randomization in essentially the same way to solve a wealth of different quadratic optimization problems.

All least squares problems are described in terms of an $m \times n$ data matrix $\boldsymbol{A}$ with at least as many rows as columns. The overdetermined problem is

$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2$$

for a vector $\boldsymbol{b}$ in $\mathbb{R}^m$, while the underdetermined problem is

$$\min_{\boldsymbol{y} \in \mathbb{R}^m} \lbrace \|\boldsymbol{y}\|_2^2 \mid \boldsymbol{A}^*\boldsymbol{y} = \boldsymbol{c} \rbrace$$

for a vector $\boldsymbol{c}$ in $\mathbb{R}^n$.

## 3.1 Problem Classes

This section covers drivers for two related classes of optimization problems: minimizing regularized positive definite quadratics (Section 3.1.1) and certain generalizations of overdetermined and underdetermined least squares which we refer to as *saddle point problems* (Section 3.1.2). Problems in both classes can naturally be transformed to equivalent linear algebra problems.

### 3.1.1 Minimizing Regularized Quadratics

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regularized Quadratic Problem)</span></p>

Let $\boldsymbol{G}$ be a positive semidefinite (psd) linear operator, and let $\mu$ be a positive regularization parameter. One of the main topics of this section is algorithms for computing approximate solutions to problems of the form

$$\min_{\boldsymbol{x}} \boldsymbol{x}^*(\boldsymbol{G} + \mu\boldsymbol{I})\boldsymbol{x} - 2\boldsymbol{h}^*\boldsymbol{x}. \qquad (3.1)$$

Note that solving (3.1) is equivalent to solving $(\boldsymbol{G} + \mu\boldsymbol{I})\boldsymbol{x} = \boldsymbol{h}$. In some contexts, $\boldsymbol{G}$ is $n \times n$, and in others, it is $m \times m$.

- Methods for solving to **higher accuracy** will access $\boldsymbol{G}$ repeatedly by matrix-matrix and matrix-vector multiplication.
- Methods for solving to **lower accuracy** may only entail selecting a subset of columns of $\boldsymbol{G}$, or performing a single matrix-matrix multiplication $\boldsymbol{G}\boldsymbol{S}$ with a tall and thin matrix $\boldsymbol{S}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(3.1.1 — Implicit Gram Matrices)</span></p>

If the linear operator $\boldsymbol{G}$ implements the action of an implicit Gram matrix $\boldsymbol{A}^*\boldsymbol{A}$ (with $\boldsymbol{A}$ known) then it would be preferable to reformulate (3.1) as the primal saddle point problem (3.2), with $\boldsymbol{b} = \boldsymbol{0}$ and $\boldsymbol{c} = \boldsymbol{h}$.

</div>

**Amenable problem structures.** The suitability of methods for problem (3.1) depends on how many eigenvalues of $\boldsymbol{G}$ are larger than $\mu$. Supposing $\boldsymbol{G}$ is $n \times n$, it is desirable that the number of such eigenvalues is much less than $n$. The data $(\boldsymbol{G}, \mu)$ that arise in practical KRR problems usually have this property.

### 3.1.2 Solving Least Squares and Basic Saddle Point Problems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Saddle Point Problems)</span></p>

We frame generalized least squares problems as complementary formulations of a common *saddle point problem*. The defining data consists of a tall $m \times n$ matrix $\boldsymbol{A}$, an $m$-vector $\boldsymbol{b}$, an $n$-vector $\boldsymbol{c}$, and a scalar $\mu \ge 0$. The **primal** saddle point problem is

$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2 + \mu\|\boldsymbol{x}\|_2^2 + 2\boldsymbol{c}^*\boldsymbol{x}. \qquad (3.2)$$

When $\mu$ is positive, the **dual** saddle point problem is

$$\min_{\boldsymbol{y} \in \mathbb{R}^m} \|\boldsymbol{A}^*\boldsymbol{y} - \boldsymbol{c}\|_2^2 + \mu\|\boldsymbol{y} - \boldsymbol{b}\|_2^2. \qquad (3.3)$$

In the limit as $\mu$ tends to zero, the dual canonically becomes

$$\min_{\boldsymbol{y} \in \mathbb{R}^m} \lbrace \|\boldsymbol{y} - \boldsymbol{b}\|_2^2 \;:\; \boldsymbol{A}^*\boldsymbol{y} = \boldsymbol{c} \rbrace. \qquad (3.4)$$

**Special cases:**
- Primal reduces to **ridge regression** when $\boldsymbol{c} = \boldsymbol{0}$, and to **overdetermined least squares** when both $\boldsymbol{c}$ and $\mu$ are zero.
- When $\boldsymbol{b} = \boldsymbol{0}$, the dual amounts to **ridge regression with a wide data matrix** or **basic underdetermined least squares**, depending on $\mu$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Benefits of the Saddle Point Viewpoint)</span></p>

Adopting this more general optimization-based viewpoint on least squares problems has two major benefits:

1. It extends least squares problems to include **linear terms** in the objective. The linear terms in (3.3) and (3.4) are obtained by expanding $\|\boldsymbol{y} - \boldsymbol{b}\|_2^2 = \|\boldsymbol{y}\|_2^2 - 2\boldsymbol{b}^*\boldsymbol{y} + \|\boldsymbol{b}\|_2^2$ and ignoring the constant term.
2. It renders the primal and dual problems **equivalent for most algorithmic purposes**. The equivalence is based on formulating the optimality conditions for these problems in a so-called *saddle point system* over the variables $(\boldsymbol{x}, \boldsymbol{y})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Ill-Posed Saddle Point Problems)</span></p>

The saddle point problems can be ill-posed when $\mu$ is zero and $\boldsymbol{A}$ is rank-deficient. Specifically, when $\mu = 0$ and $\boldsymbol{c}$ is not orthogonal to the kernel of $\boldsymbol{A}$, the primal problem (3.2) has no optimal solution and the dual problem (3.4) has no feasible solution. In this setting, **canonical solutions** are assigned by considering the limit as $\mu$ tends to zero:

$$\boldsymbol{x} = (\boldsymbol{A}^*\boldsymbol{A})^\dagger(\boldsymbol{A}^*\boldsymbol{b} - \boldsymbol{c}) \qquad \text{and} \qquad \boldsymbol{y} = (\boldsymbol{A}^*)^\dagger\boldsymbol{c} + (\boldsymbol{I} - \boldsymbol{A}\boldsymbol{A}^\dagger)\boldsymbol{b},$$

which are related through the identity $\boldsymbol{y} = \boldsymbol{b} - \boldsymbol{A}\boldsymbol{x}$.

</div>

**Amenable problem structures.** This section focuses on methods for solving these optimization problems to high accuracy. If $m \gg n$, then these methods are efficient regardless of numerical aspects of the problem data $(\boldsymbol{A}, \boldsymbol{b}, \boldsymbol{c}, \mu)$. If $m$ is only slightly larger than $n$, then the methods will only be effective when $\boldsymbol{G} := \boldsymbol{A}^*\boldsymbol{A}$ and $\mu$ have the properties alluded to in Section 3.1.1.

## 3.2 Drivers

Four families of drivers are presented. Two belong to the paradigm of **sketch-and-precondition** (capable of computing accurate approximations). The other two belong to **sketch-and-solve** (less expensive, but only suitable for rough approximations).

### 3.2.1 Sketch-and-Solve for Overdetermined Least Squares

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sketch-and-Solve)</span></p>

Sketch-and-solve is a broad paradigm within RandNLA. For least squares, one samples a sketching operator $\boldsymbol{S}$, and returns

$$(\boldsymbol{S}\boldsymbol{A})^\dagger(\boldsymbol{S}\boldsymbol{b}) \in \operatorname{argmin}_{\boldsymbol{x}} \|\boldsymbol{S}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b})\|_2^2 \qquad (3.6)$$

as a proxy for the solution to $\min_{\boldsymbol{x}} \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Sketch-and-Solve Error Bound)</span></p>

If $\boldsymbol{S}$ is a subspace embedding for $V = \operatorname{range}([\boldsymbol{A}, \boldsymbol{b}])$ with distortion $\delta$, then

$$\|\boldsymbol{A}(\boldsymbol{S}\boldsymbol{A})^\dagger(\boldsymbol{S}\boldsymbol{b}) - \boldsymbol{b}\|_2 \le \left(\frac{1 + \delta}{1 - \delta}\right)\|\boldsymbol{A}\boldsymbol{A}^\dagger\boldsymbol{b} - \boldsymbol{b}\|_2. \qquad (3.7)$$

Note that (3.6) is invariant under scaling of $\boldsymbol{S}$. This implies that (3.7) also holds when $\delta$ is the effective distortion of $\boldsymbol{S}$ for $V$.

</div>

**Error estimation.** Since sketch-and-solve algorithms are most suitable for computing rough approximations, it is important to have methods for estimating the error. Such estimates can either be used to inform downstream processing or to determine if a more accurate solution (computed by a more expensive algorithm) might be needed. It is especially important that these methods work well in regimes where sketch-and-solve has a compelling computational profile, such as when $m \gg dn$.

**Application to tensor decomposition.** The benefits of sketch-and-solve for least squares manifest most prominently when: (1) $m$ is extremely large, so $\boldsymbol{A}$ is not stored explicitly, and (2) $\boldsymbol{A}$ supports relatively cheap access to individual rows $\boldsymbol{A}[i,:]$. This situation arises in alternating least squares approaches to tensor decomposition.

**Methods for the sketched subproblem.** Direct methods for (3.6) require computing an orthogonal decomposition of $\boldsymbol{S}\boldsymbol{A}$, such as a QR decomposition or an SVD, in $O(dn^2)$ time. In this context, sketch-and-solve can be used as a preprocessing step for sketch-and-precondition methods at essentially no added cost. Therefore if a direct method is being considered for sketch-and-solve, then sketch-and-precondition methods should also be viable when $m \in O(dn)$.

### 3.2.2 Sketch-and-Precondition for Least Squares and Saddle Point Problems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sketch-and-Precondition)</span></p>

The *sketch-and-precondition* approach to overdetermined least squares was introduced by Rokhlin and Tygert. When the $m \times n$ matrix $\boldsymbol{A}$ is very tall, the method is capable of producing accurate solutions with less expense than direct methods. It starts by computing a $d \times n$ sketch $\boldsymbol{A}^{\text{sk}} = \boldsymbol{S}\boldsymbol{A}$ in the embedding regime (i.e., $d \gtrsim n$). The sketch is decomposed by QR with column pivoting $\boldsymbol{A}^{\text{sk}}\boldsymbol{\Pi} = \boldsymbol{Q}\boldsymbol{R}$, which defines a preconditioner $\boldsymbol{M} = \boldsymbol{\Pi}\boldsymbol{R}^{-1}$. If the parameters for the sketching operator distribution were chosen appropriately, then $\boldsymbol{A}\boldsymbol{M}$ will be nearly-orthogonal with high probability. The near-orthogonality of $\boldsymbol{A}\boldsymbol{M}$ ensures rapid convergence of an iterative method for the least squares problem's preconditioned normal equations. If $T_{\text{sk}}$ denotes the time complexity of computing $\boldsymbol{S}\boldsymbol{A}$, then the typical asymptotic FLOP count to solve to $\epsilon$-error is

$$O(T_{\text{sk}} + dn^2 + mn\log(1/\epsilon)). \qquad (3.8)$$

Importantly, this complexity has no dependence on the condition number of $\boldsymbol{A}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Nystrom Preconditioning for Saddle Point Problems)</span></p>

If a saddle point problem features regularization (i.e., if $\mu > 0$) and if $\boldsymbol{A}$ has rapid spectral decay, then randomized methods can be used to find a good preconditioner in far less than $O(n^3)$ time, no matter the specific value of $m \ge n$. This is possible by borrowing ideas from *Nystrom preconditioning*, which we introduce in Section 3.2.3 for the related problem of minimizing regularized quadratics. As a novel contribution, Section 3.3.3 explains how Nystrom preconditioning can naturally be adapted to saddle point problems.

</div>

**Algorithms.** Sketch-and-precondition algorithms can take different approaches to sketching, preconditioner generation, and choice of the eventual iterative solver.

- *Blendenpik* used SRFT sketching operators, obtained its preconditioner by unpivoted QR of $\boldsymbol{A}^{\text{sk}}$, and used LSQR as its underlying iterative method.
- *LSRN* used Gaussian sketching operators, obtained its preconditioner through an SVD of $\boldsymbol{A}^{\text{sk}}$, and defaulted to the Chebyshev semi-iterative method for its iterative solver.

Both algorithms initialize the iterative solver at the solution from a sketch-and-solve approach in the vein of Section 3.2.1. This presolve step is negligible in cost but should save several iterations when solving to a prescribed accuracy. It also plays an important role in handling overdetermined least squares problems when $\boldsymbol{b}$ is in the range of $\boldsymbol{A}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(SP01: Blendenpik-Like Approach to Overdetermined Least Squares)</span></p>

**function** $\texttt{SP01}(\boldsymbol{A}, \boldsymbol{b}, \epsilon, L)$

**Inputs:** $\boldsymbol{A}$ is $m \times n$ and $\boldsymbol{b}$ is an $m$-vector. We require $m \ge n$ and expect $m \gg n$. The iterative solver's termination criteria are governed by $\epsilon$ and $L$: it stops if the solution reaches error $\epsilon \ge 0$ according to the solver's error metric, or if the solver completes $L \ge 1$ iterations.

**Output:** An approximate solution to (3.2), with $\boldsymbol{c} = \boldsymbol{0}$ and $\mu = 0$.

**Abstract subroutines and tuning parameters:**
- $\texttt{SketchOpGen}$ generates an oblivious sketching operator.
- $\texttt{sampling\_factor} \ge 1$ is the size of the embedding dimension relative to $n$.

1. $d = \min\lbrace \lceil n \cdot \texttt{sampling\_factor} \rceil, m \rbrace$
2. $\boldsymbol{S} = \texttt{SketchOpGen}(d, m)$
3. $[\boldsymbol{A}^{\text{sk}}, \boldsymbol{b}_{\text{sk}}] = \boldsymbol{S}[\boldsymbol{A}, \boldsymbol{b}]$
4. $\boldsymbol{Q}, \boldsymbol{R} = \texttt{qr\_econ}(\boldsymbol{A}^{\text{sk}})$
5. $\boldsymbol{z}_o = \boldsymbol{Q}^*\boldsymbol{b}_{\text{sk}}$ &ensp; *(i.e., $\boldsymbol{R}^{-1}\boldsymbol{z}_o$ solves $\min_{\boldsymbol{x}} \lbrace \|\boldsymbol{S}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b})\|_2^2 \rbrace$)*
6. $\boldsymbol{A}_{\text{precond}} = \boldsymbol{A}\boldsymbol{R}^{-1}$ &ensp; *(as a linear operator)*
7. $\boldsymbol{z} = \texttt{iterative\_ls\_solver}(\boldsymbol{A}_{\text{precond}}, \boldsymbol{b}, \epsilon, L, \boldsymbol{z}_o)$
8. **return** $\boldsymbol{R}^{-1}\boldsymbol{z}$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(SPS2: Sketch, Transform Saddle Point Problem to Least Squares, and Precondition)</span></p>

**function** $\texttt{SPS2}(\boldsymbol{A}, \boldsymbol{b}, \boldsymbol{c}, \mu, \epsilon, L)$

**Inputs:** $\boldsymbol{A}$ is $m \times n$, $\boldsymbol{b}$ is an $m$-vector, $\boldsymbol{c}$ is an $n$-vector, and $\mu$ is a nonnegative regularization parameter. We require $m \ge n$ and expect $m \gg n$. The iterative solver's termination criteria are governed by $\epsilon$ and $L$.

**Output:** Approximate solutions to (3.2) *and* its dual problem.

**Abstract subroutines and tuning parameters:** Same as SP01.

1. $d = \min\lbrace \lceil n \cdot \texttt{sampling\_factor} \rceil, m \rbrace$
2. $\boldsymbol{S} = \texttt{SketchOpGen}(d, m)$
3. **if** $\mu > 0$ **then** set $\boldsymbol{S} \leftarrow \begin{bmatrix} \boldsymbol{S} & \boldsymbol{0} \\\\ \boldsymbol{0} & \boldsymbol{I}_n \end{bmatrix}$, $\boldsymbol{A} \leftarrow \begin{bmatrix} \boldsymbol{A} \\\\ \sqrt{\mu}\boldsymbol{I}_n \end{bmatrix}$, $\boldsymbol{b} \leftarrow \begin{bmatrix} \boldsymbol{b} \\\\ \boldsymbol{0} \end{bmatrix}$
4. $\boldsymbol{A}^{\text{sk}} = \boldsymbol{S}\boldsymbol{A}$
5. $\boldsymbol{U}, \boldsymbol{\Sigma}, \boldsymbol{V}^* = \texttt{svd}(\boldsymbol{A}^{\text{sk}})$
6. $\boldsymbol{M} = \boldsymbol{V}\boldsymbol{\Sigma}^\dagger$
7. $\boldsymbol{b}_{\text{mod}} = \boldsymbol{b}$
8. **if** $\boldsymbol{c} \ne \boldsymbol{0}$ **then:**
   - $\hat{\boldsymbol{v}} = \boldsymbol{U}\boldsymbol{\Sigma}^\dagger\boldsymbol{V}^*\boldsymbol{c}$
   - $\boldsymbol{b}_{\text{shift}} = \boldsymbol{S}^*\hat{\boldsymbol{v}}$
   - $\boldsymbol{b}_{\text{mod}} = \boldsymbol{b}_{\text{mod}} - \boldsymbol{b}_{\text{shift}}$
9. $\boldsymbol{z}_o = \boldsymbol{U}^*\boldsymbol{S}\boldsymbol{b}_{\text{mod}}$
10. $\boldsymbol{A}_{\text{precond}} = \boldsymbol{A}\boldsymbol{M}$ &ensp; *(defined implicitly, as a linear operator)*
11. $\boldsymbol{z} = \texttt{iterative\_ls\_solver}(\boldsymbol{A}_{\text{precond}}, \boldsymbol{b}_{\text{mod}}, \epsilon, L, \boldsymbol{z}_o)$
12. $\boldsymbol{x} = \boldsymbol{M}\boldsymbol{z}$
13. $\boldsymbol{y} = \boldsymbol{b}[:m] - \boldsymbol{A}[:m,:]\boldsymbol{x}$
14. **return** $\boldsymbol{x}$, $\boldsymbol{y}$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Novelty of Algorithm SPS2)</span></p>

While Algorithm SP01 is standard, Algorithm SPS2 is somewhat novel. Using the same data that might be computed during a standard sketch-and-precondition algorithm for simple overdetermined least squares, it transforms any saddle point problem — primal or dual — into an equivalent primal saddle point problem with $\boldsymbol{c} = \boldsymbol{0}$. To the authors' knowledge, no such conversion routines have been described in the literature. The conversion is advantageous because it opens the possibility of using iterative solvers with excellent numerical properties that are specific to least squares problems.

</div>

**When is sketch-and-precondition asymptotically faster than QR?** Using SRFTs as sketching operators, it takes $O(mn\log d)$ time to apply a $d \times m$ SRFT to an $m \times n$ matrix. Plugging $T_{\text{sk}} = mn\log d$ into (3.8), the "typical" runtime for sketch-and-precondition with an SRFT is

$$O(mn\log d + dn^2 + mn\log(1/\epsilon)). \qquad (3.9)$$

Taking $d = sn$ for small constants $s$ (e.g., $s = 4$) sufficed to accurately describe algorithm runtime in practice. However, the theoretical analysis needed $d \in \Omega(n^2)$ to bound a random multiplicative factor $F$ on the $mn\log(1/\epsilon)$ term with high probability. The best theoretical runtime guarantee was originally obtained by plugging $d = n^2$ into (3.9). Theoretical guarantees improved following developments in the analysis of SRFTs: taking $d = n\log n$ sufficed to bound $F$ with high probability. This is the appropriate bound to use when comparing the theoretical asymptotic runtime of SRFT-based sketch-and-precondition to other algorithms. In practice, it is still preferred to use $d = sn$ for some small $s > 1$, since the resulting preconditioned matrices tend to be extremely well-conditioned.

### 3.2.3 Nystrom PCG for Minimizing Regularized Quadratics

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nystrom Preconditioned Conjugate Gradient)</span></p>

Nystrom preconditioned conjugate gradient (Nystrom PCG) is a recently-proposed method for solving problems of the form (3.1) to fairly high accuracy. It computes approximate solutions to linear systems $(\boldsymbol{G} + \mu\boldsymbol{I})\boldsymbol{x} = \boldsymbol{h}$ where $\boldsymbol{G}$ is $n \times n$ and psd.

The randomness in Nystrom PCG is encapsulated in an initial phase where it computes a low-rank approximation of $\boldsymbol{G}$ by a so-called "Nystrom approximation." A rank-$\ell$ Nystrom approximation leads to a preconditioner $\boldsymbol{P}$ which can be stored in $O(\ell n)$ space and applied in $O(\ell n)$ time.

We presume access to a low-rank approximation

$$\hat{\boldsymbol{G}} = \boldsymbol{V}\operatorname{diag}(\boldsymbol{\lambda})\boldsymbol{V}^* \qquad (3.10)$$

where $\boldsymbol{V}$ is a column-orthonormal $n \times \ell$ matrix that approximates the dominant $\ell$ eigenvectors of $\boldsymbol{G}$ and $\lambda_1 \ge \cdots \ge \lambda_\ell > 0$ are the approximated eigenvalues. The data $(\boldsymbol{V}, \boldsymbol{\lambda}, \mu)$ is then used to define a preconditioner

$$\boldsymbol{P}^{-1} = \boldsymbol{V}\operatorname{diag}(\boldsymbol{\lambda} + \mu)^{-1}\boldsymbol{V}^* + (\mu + \lambda_\ell)^{-1}(\boldsymbol{I}_n - \boldsymbol{V}\boldsymbol{V}^*). \qquad (3.11)$$

</div>

Let $\kappa$ denote the condition number of $\boldsymbol{G}_p := \boldsymbol{P}^{-1/2}(\boldsymbol{G} + \mu\boldsymbol{I})\boldsymbol{P}^{-1/2}$. It is well-known that each iteration of PCG requires one matrix-vector multiply with $\boldsymbol{G}$, one matrix-vector multiply with $\boldsymbol{P}^{-1}$, and reduces the error of the candidate solution to (3.1) by a multiplicative factor $(\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$. One can expect that $\kappa$ will be $O(1)$ if the $\ell^{\text{th}}$-largest eigenvalue of $\boldsymbol{G}$ is smaller than $\mu$. Indeed, Nystrom PCG is most effective for problems when this threshold is crossed at some $\ell \ll n$.

Assuming that $(\boldsymbol{V}, \boldsymbol{\lambda})$ are very good estimates for the top $\ell$ eigenpairs of $\boldsymbol{G}$ *and* that $\lambda_\ell(\boldsymbol{G}) \approx \lambda_{\ell+1}(\boldsymbol{G})$, the condition number of $\boldsymbol{G}_p$ should be near

$$\kappa_\ell(\boldsymbol{G}, \mu) := (\lambda_\ell(\boldsymbol{G}) + \mu)/(\lambda_n(\boldsymbol{G}) + \mu).$$

Taking this for granted, the preconditioner (3.11) can only be effective if $\ell \ll n$ is large enough so that $\kappa_\ell(\boldsymbol{G}, \mu)$ is bounded by a small constant. Using the fact that $\kappa_\ell(\boldsymbol{G}, u) \le 1 + \lambda_\ell(\boldsymbol{G})/\mu$, a good preconditioner is possible when $\lambda_\ell(\boldsymbol{G})/\mu$ is $O(1)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(3.2.1 — Exact Eigenvectors)</span></p>

The argument above can be made more rigorous by assuming that $\boldsymbol{V}$ is an $n \times (\ell - 1)$ matrix that contains the *exact* leading $\ell - 1$ eigenvectors of $\boldsymbol{G}$, and that $\lambda_1, \ldots, \lambda_\ell$ are the *exact* leading $\ell$ eigenvalues of $\boldsymbol{G}$. In this case, the condition number of $\boldsymbol{G}_p$ will be equal to $\kappa_\ell(\boldsymbol{G}, \mu)$, which will be at most $1 + \lambda_\ell/\mu$.

</div>

### 3.2.4 Sketch-and-Solve for Minimizing Regularized Quadratics

Randomization offers several avenues for solving problems of the form (3.1) to modest accuracy. We describe two possible methods through novel interpretations of existing work on KRR. For notation, $\boldsymbol{G}$ is $m \times m$, $\mu = m\lambda$ for some $\lambda > 0$, and the optimization variable in (3.1) is denoted "$\boldsymbol{\alpha}$" rather than "$\boldsymbol{x}$."

**A one-shot fallback on Nystrom approximations.** Rather than solving (3.1) directly, it has been suggested that one solve

$$(\boldsymbol{A}\boldsymbol{A}^* + m\lambda\boldsymbol{I})\hat{\boldsymbol{\alpha}} = \boldsymbol{h},$$

where $\boldsymbol{A}\boldsymbol{A}^*$ is a Nystrom approximation of $\boldsymbol{G}$. The computation of $\boldsymbol{A}$ only requires access to $\boldsymbol{G}$ by a single sketch $\boldsymbol{G}\boldsymbol{S}$ for a tall $m \times n$ sketching operator $\boldsymbol{S}$. In the KRR context, it is especially popular for $\boldsymbol{S}$ to be a column sampling operator. There is an equivalence between computing $\hat{\boldsymbol{\alpha}}$ and solving a dual saddle point problem with matrix $\boldsymbol{A}$ and other data $(\boldsymbol{b}, \boldsymbol{c}, \mu) = (\boldsymbol{h}, \boldsymbol{0}, m\lambda)$. That dual saddle point problem can naturally be approached by sketch-and-precondition methods from Section 3.2.2.

**Applying a random subspace constraint.** By taking the gradient of the objective function in (3.1) and multiplying the gradient by the positive definite matrix $\boldsymbol{G}$, we can recast (3.1) as minimizing

$$Q(\boldsymbol{\alpha}) = \boldsymbol{\alpha}^*(\boldsymbol{G}^2 + m\lambda\boldsymbol{G})\boldsymbol{\alpha} - 2\boldsymbol{h}^*\boldsymbol{G}\boldsymbol{\alpha}.$$

A sketch-and-solve approach is to minimize $Q(\boldsymbol{\alpha})$ subject to a constraint that $\boldsymbol{\alpha}$ is in the range of a very tall $m \times n$ sketching operator $\boldsymbol{S}$. The constrained minimization problem is equivalent to minimizing $\boldsymbol{z} \mapsto Q(\boldsymbol{S}\boldsymbol{z})$ over $n$-vectors $\boldsymbol{z}$. This in turn is equivalent to solving a highly overdetermined least squares problem, with an $(m + n) \times n$ data matrix $\boldsymbol{A} = [\boldsymbol{G}\boldsymbol{S};\; \sqrt{m\lambda}\boldsymbol{R}]$ where $\boldsymbol{R}$ is any matrix for which $\boldsymbol{R}^*\boldsymbol{R} = \boldsymbol{S}^*\boldsymbol{G}\boldsymbol{S}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(3.2.2 — On Required Sketches)</span></p>

Note that the approach of applying a random subspace constraint presumes access to the sketches $\boldsymbol{h}^*\boldsymbol{G}\boldsymbol{S}$, $\boldsymbol{S}^*\boldsymbol{G}\boldsymbol{S}$, and $\boldsymbol{S}^*\boldsymbol{G}^2\boldsymbol{S}$, and advocates for solving the resulting $n$-dimensional minimization problem by a direct method in $O(n^3)$ time. However, no guidance is given on how to compute the sketch $\boldsymbol{S}^*\boldsymbol{G}^2\boldsymbol{S}$. From what we can tell, the most efficient way of doing this would be to form the Gram matrix at cost $O(mn^2)$ assuming access to the sketch $\boldsymbol{G}\boldsymbol{S}$.

</div>

## 3.3 Computational Routines

To contextualize the computational routines that follow, we begin in Section 3.3.1 with a brief discussion of optimality conditions for saddle point problems. From there, we present in Sections 3.3.2 and 3.3.3 two families of methods for generating preconditioners needed by saddle point drivers; our presentation of both families includes novel observations that lead to improved efficiency and numerical stability. Then in Section 3.3.4 we discuss deterministic preconditioned iterative methods for positive definite systems and saddle point problems.

**Routines not detailed here.** The driver from Section 3.2.3 requires methods to compute Nystrom approximations, which are described in Section 4. The drivers from Section 3.2.4 would benefit from specialized data-aware methods for sketching kernel matrices, which are discussed in Section 6.

### 3.3.1 Technical Background: Optimality Conditions for Saddle Point Problems

We call an $n$-vector $\boldsymbol{x}$ **primal-optimal** if it solves (3.2). Analogously, an $m$-vector $\boldsymbol{y}$ shall be called **dual-optimal** if it solves (3.3) when $\mu$ is positive or (3.4) when $\mu$ is zero.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Saddle Point System)</span></p>

Primal-dual optimal solutions can be characterized with *saddle point systems*. These are a class of $2 \times 2$ block linear systems that arise broadly in computational mathematics and especially in optimization. We are interested in saddle point systems of the form

$$\begin{bmatrix} \boldsymbol{I} & \boldsymbol{A} \\\\ \boldsymbol{A}^* & -\mu\boldsymbol{I} \end{bmatrix} \begin{bmatrix} \boldsymbol{y} \\\\ \boldsymbol{x} \end{bmatrix} = \begin{bmatrix} \boldsymbol{b} \\\\ \boldsymbol{c} \end{bmatrix}. \qquad (3.12)$$

A solution to such a system always exists when $\mu$ is positive or when the tall matrix $\boldsymbol{A}$ is full-rank. Given that assumption:

- A point $\tilde{\boldsymbol{x}}$ is primal-optimal if and only if there is a $\tilde{\boldsymbol{y}}$ for which $(\tilde{\boldsymbol{x}}, \tilde{\boldsymbol{y}})$ solve (3.12).
- A point $\tilde{\boldsymbol{y}}$ is dual-optimal if and only if there is an $\tilde{\boldsymbol{x}}$ for which $(\tilde{\boldsymbol{x}}, \tilde{\boldsymbol{y}})$ solve (3.12).

</div>

Saddle point systems are often reformulated into equivalent positive semidefinite systems. The reformulation takes the system's upper block to *define* $\boldsymbol{y} = \boldsymbol{b} - \boldsymbol{A}\boldsymbol{x}$, and then substitutes that expression into the system's lower block. This gives us the **normal equations**

$$(\boldsymbol{A}^*\boldsymbol{A} + \mu\boldsymbol{I})\boldsymbol{x} = \boldsymbol{A}^*\boldsymbol{b} - \boldsymbol{c}. \qquad (3.13)$$

Therefore one can solve (3.12) by first solving (3.13) and then setting $\boldsymbol{y} = \boldsymbol{b} - \boldsymbol{A}\boldsymbol{x}$.

Thinking in terms of the normal equations helps with the design of preconditioners. When accurate solutions are desired, however, it is preferable to employ reformulations that reduce the need for matrix-vector products with the linear operator $\boldsymbol{A}^*\boldsymbol{A}$. Such reformulations start by defining an augmented data matrix $\boldsymbol{A}_\mu = [\boldsymbol{A};\; \sqrt{\mu}\boldsymbol{I}_n]$. For dual saddle point problems, one solves

$$\min \lbrace \|\Delta\boldsymbol{y}\|_2^2 \;:\; \Delta\boldsymbol{y} \in \mathbb{R}^{m+n},\; (\boldsymbol{A}_\mu)^*\Delta\boldsymbol{y} = \boldsymbol{c} - \boldsymbol{A}^*\boldsymbol{b} \rbrace, \qquad (3.14)$$

and subsequently recovers the dual-optimal solution $\boldsymbol{y} = [b_1 + \Delta y_1, \ldots, b_m + \Delta y_m]$. For primal saddle point problems, one computes some $\boldsymbol{b}_{\text{shift}} \in \mathbb{R}^{m+n}$ satisfying $(\boldsymbol{A}_\mu)^*\boldsymbol{b}_{\text{shift}} = \boldsymbol{c}$ and then defines $\boldsymbol{b}_\mu = [\boldsymbol{b}; \boldsymbol{0}_n] - \boldsymbol{b}_{\text{shift}}$. Any solution to the resulting problem

$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \lbrace \|\boldsymbol{A}_\mu\boldsymbol{x} - \boldsymbol{b}_\mu\|_2^2 \rbrace \qquad (3.15)$$

is primal-optimal.

**Inconsistent saddle point systems.** Suppose that $\mu$ is zero. Under this assumption, (3.12) is inconsistent if and only if $\boldsymbol{c}$ is not in the range of $\boldsymbol{A}^*$. Since $\boldsymbol{c} \notin \operatorname{range}(\boldsymbol{A}^*)$ is equivalent to $\boldsymbol{c} \notin \ker(\boldsymbol{A})^\perp$, we see that inconsistency of (3.12) is equivalent to (3.2) having no optimal solution. Therefore a saddle point system is consistent if and only if its associated saddle point problems are well-posed; for ill-posed problems, recall that we canonically assign solutions per (3.5).

### 3.3.2 Preconditioning Least Squares and Saddle Point Problems: Tall Data Matrices

There is a simple unifying framework for preconditioner generation of the kind used in sketch-and-precondition methods. The framework is applicable to any least squares or saddle point problem (3.2)-(3.4) in the regime $m \gg n$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sketch and Orthogonalize Framework)</span></p>

Begin by defining a sketch $\boldsymbol{A}^{\text{sk}} = \boldsymbol{S}\boldsymbol{A}$ where the sketching operator $\boldsymbol{S}$ has $d \gtrsim n$ rows. We also define the augmented matrices

$$\boldsymbol{A}_\mu = \begin{bmatrix} \boldsymbol{A} \\\\ \sqrt{\mu}\boldsymbol{I} \end{bmatrix} \quad \text{and} \quad \boldsymbol{A}_\mu^{\text{sk}} = \begin{bmatrix} \boldsymbol{A}^{\text{sk}} \\\\ \sqrt{\mu}\boldsymbol{I} \end{bmatrix}.$$

We say that a matrix $\boldsymbol{M}$ **orthogonalizes** $\boldsymbol{A}_\mu^{\text{sk}}$ if the columns of $\boldsymbol{A}_\mu^{\text{sk}}\boldsymbol{M}$ are an orthonormal basis for the range of $\boldsymbol{A}_\mu^{\text{sk}}$. Such a matrix is called a **valid preconditioner** for $\boldsymbol{A}_\mu$ if, in addition, $\operatorname{rank}(\boldsymbol{A}_\mu^{\text{sk}}) = \operatorname{rank}(\boldsymbol{A}_\mu)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.3.1 — Spectrum of Preconditioned Matrix)</span></p>

Let $\boldsymbol{U}$ be a matrix whose columns form an orthonormal basis for the range of $\boldsymbol{A}$. If $\boldsymbol{M}$ is a valid preconditioner for $\boldsymbol{A}$, then the spectrum of $\boldsymbol{A}\boldsymbol{M}$ is equal to that of $(\boldsymbol{S}\boldsymbol{U})^\dagger$.

*If $\boldsymbol{M}$ is a valid preconditioner for $\boldsymbol{A}_\mu$, then the condition number of $\boldsymbol{A}_\mu\boldsymbol{M}$ does not depend on that of $\boldsymbol{A}_\mu$.*

</div>

**QR-based preconditioning in the full-rank case.** QR-based preconditioning when $\mu = 0$ is very simple; one need only run Householder QR on $\boldsymbol{A}^{\text{sk}}$ and return $\boldsymbol{M} = \boldsymbol{R}^{-1}$ as a linear operator. Householder-type representations of $\boldsymbol{A}^{\text{sk}}$'s QR decomposition are especially useful since they require a modest amount of added workspace on top of storing $\boldsymbol{A}^{\text{sk}}$.

The case with $\mu > 0$ is more complicated if we want to avoid forming $\boldsymbol{A}_\mu^{\text{sk}}$ explicitly. Suppose we have an initial QR decomposition $\boldsymbol{A}^{\text{sk}} = \boldsymbol{Q}_0\boldsymbol{R}_0$. It is easy to show that the factor $\boldsymbol{R}$ from a QR decomposition of $\boldsymbol{A}_\mu^{\text{sk}}$ is the same as the triangular factor from a QR decomposition of $\hat{\boldsymbol{R}} := [\boldsymbol{R}_0;\; \sqrt{\mu}\boldsymbol{I}]$. There are specialized algorithms for QR decomposition of matrices given by an implicit vertical concatenation of a triangular matrix and a diagonal matrix; these specialized algorithms only require $O(n)$ additional workspace.

If $\boldsymbol{A}$ is not too ill-conditioned then the same preconditioner can be obtained by Cholesky-decomposing the regularized Gram matrix $(\boldsymbol{A}_\mu^{\text{sk}})^*(\boldsymbol{A}_\mu^{\text{sk}}) = (\boldsymbol{A}^{\text{sk}})^*(\boldsymbol{A}^{\text{sk}}) + \mu\boldsymbol{I}$. This approach is simple to implement, and its time and space requirements are unaffected by whether or not $\mu$ is zero. However, it requires solving via the normal equations, which is not a numerically stable approach.

**QR-based preconditioning in the rank-deficient case.** Suppose $\mu = 0$ and let $k = \operatorname{rank}(\boldsymbol{A}^{\text{sk}}) \lesssim n$. One can use a variety of methods to compute preconditioners that are *morally triangular* in the sense that they are of the form $\boldsymbol{M} = \boldsymbol{P}\boldsymbol{R}^{-1}$ for an $n \times k$ partial-permutation matrix $\boldsymbol{P}$ and a triangular matrix $\boldsymbol{R}$. As long as the preconditioner orthogonalizes $\boldsymbol{A}^{\text{sk}}$, we can postprocess $\boldsymbol{z}_{\text{sol}} = \operatorname{argmin} \|\boldsymbol{A}\boldsymbol{M}\boldsymbol{z} - \boldsymbol{b}\|_2^2$ to obtain $\boldsymbol{x}_{\text{sol}} = \boldsymbol{M}\boldsymbol{z}_{\text{sol}}$ which solves $\min \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2$.

**SVD-based preconditioners.** Let us denote the SVD of $\boldsymbol{A}^{\text{sk}}$ by $\boldsymbol{U}\operatorname{diag}(\boldsymbol{\sigma})\boldsymbol{V}^*$.

First consider preconditioner generation when $\mu = 0$. In this case we must account for the fact that $\boldsymbol{A}^{\text{sk}}$ might be rank-deficient. Letting $k$ denote the rank of $\boldsymbol{A}^{\text{sk}}$, the SVD-based preconditioner is the $n \times k$ matrix

$$\boldsymbol{M} = \left[\frac{\boldsymbol{v}_1}{\sigma_1}, \ldots, \frac{\boldsymbol{v}_k}{\sigma_k}\right].$$

This construction is important, because it can be shown that if $\boldsymbol{z}_*$ solves

$$\min_{\boldsymbol{z}} \|\boldsymbol{A}\boldsymbol{M}\boldsymbol{z} - \boldsymbol{b}\|_2^2 + \boldsymbol{c}^*\boldsymbol{M}\boldsymbol{z} \qquad (3.16)$$

then $\boldsymbol{x} = \boldsymbol{M}\boldsymbol{z}_*$ satisfies the canonical solution (3.5). In particular, (3.16) has a unique optimal solution and computing $\boldsymbol{z}_*$ is a well-posed problem.

SVD-based preconditioning is conceptually simpler when $\mu$ is positive, since in that case it does not matter if $\boldsymbol{A}^{\text{sk}}$ is rank-deficient. The singular values of $\boldsymbol{A}_\mu^{\text{sk}}$ are $\hat{\sigma}_i = \sqrt{\sigma_i^2 + \mu}$, and its right singular vectors are the same as those of $\boldsymbol{A}^{\text{sk}}$. These observations alone are sufficient to recover the preconditioner

$$\boldsymbol{M} = \boldsymbol{V}\operatorname{diag}\left(\frac{1}{\hat{\sigma}_1}, \ldots, \frac{1}{\hat{\sigma}_n}\right)$$

which orthogonalizes $\boldsymbol{A}_\mu^{\text{sk}}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(3.3.2 — Computational Complexity)</span></p>

The default algorithm for SVD is currently divide-and-conquer. Two somewhat-outdated algorithms for computing the SVD are described in standard references on numerical linear algebra. The choice of algorithm depends on whether the left singular vectors need to be computed.

</div>

### 3.3.3 Preconditioning Least Squares and Saddle Point Problems: Data Matrices with Fast Spectral Decay

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nystrom Preconditioner for Saddle Point Problems)</span></p>

**Interpreting the Nystrom preconditioner.** Recall from Section 3.2.3 that the Nystrom preconditioning approach to solving $(\boldsymbol{G} + \mu\boldsymbol{I})\boldsymbol{x} = \boldsymbol{h}$ starts by constructing a low-rank approximation of $\boldsymbol{G}$. That approximation defines a preconditioner $\boldsymbol{P}$ satisfying three properties:

1. $\boldsymbol{P}$ is positive definite.
2. $(\boldsymbol{G} + \mu\boldsymbol{I})\boldsymbol{P}^{-1}$ is well-conditioned on a subspace $L$ that contains $\boldsymbol{G}$'s dominant eigenspaces.
3. $\boldsymbol{P}$ acts as the identity on $L^\perp$ (the orthogonal complement of $L$).

Such a preconditioner will be effective when the action of $\boldsymbol{G}$ on $L^\perp$ is "not too pronounced" compared to that of $\mu\boldsymbol{I}$. More formally, if we define the restricted spectral norm of $\boldsymbol{G}$ on $L^\perp$ as

$$\|\boldsymbol{G}\|_{L^\perp} = \max\lbrace \|\boldsymbol{G}\boldsymbol{z}\|_2 \;:\; \boldsymbol{z} \in L^\perp, \|\boldsymbol{z}\|_2 = 1 \rbrace$$

then the preconditioner will be effective if $\|\boldsymbol{G}\|_{L^\perp}/\mu$ is $O(1)$.

</div>

**Adaptation to saddle point problems.** Nystrom preconditioners can naively be used for regularized saddle point problems by taking $\boldsymbol{G} = \boldsymbol{A}^*\boldsymbol{A}$ and considering the normal equations (3.13). However, the numerical properties of iterative least squares solvers that only access $\boldsymbol{A}^*\boldsymbol{A}$ tend to be less robust than those of iterative solvers that access $\boldsymbol{A}$ and $\boldsymbol{A}^*$ separately (i.e., solvers such as LSQR). This motivates having an extension of the Nystrom preconditioner to be compatible with the latter type of solver.

Towards this end, let us express $\boldsymbol{P}^{-1}$ with a (possibly non-symmetric) matrix square-root $\boldsymbol{M}$, satisfying the relation $\boldsymbol{P}^{-1} = \boldsymbol{M}\boldsymbol{M}^*$. The basic criteria for a $\boldsymbol{P}$ as a Nystrom preconditioner for the normal equations (3.13) can be stated with $\boldsymbol{M}$ as follows:

1. $\boldsymbol{A}_\mu\boldsymbol{M}$ should be well-conditioned on a subspace $L$ that includes the dominant right singular vectors of $\boldsymbol{A}_\mu$.
2. We should have $\boldsymbol{A}_\mu\boldsymbol{M}\boldsymbol{x} = \boldsymbol{A}_\mu\boldsymbol{x}$ for all $\boldsymbol{x} \in L^\perp$.

Whether such a preconditioner will be effective can be stated with the "restricted spectral norm" as defined above. Specifically, $\boldsymbol{M}$ should be effective if the above conditions hold and $\|\boldsymbol{A}\|_{L^\perp}/\sqrt{\mu}$ is $O(1)$.

### 3.3.4 Deterministic Preconditioned Iterative Solvers

Most of the drivers described in Section 3.2 amount to using randomization to obtain a preconditioner and then calling a traditional iterative solver that can make use of that randomized preconditioner. We note that many factors can affect the ideal choice of iterative method in a given setting.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Iterative Solvers for Preconditioned Systems)</span></p>

- **CG** (Conjugate Gradient): The most broadly applicable solver in our context. It applies to the regularized positive definite system (3.1) and the normal equations of the primal saddle point problem (3.13).
- **CGLS**: Applies when $\boldsymbol{c}$ is zero. It is equivalent to CG on the normal equations in exact arithmetic, but is more stable than CG in finite-precision arithmetic.
- **LSQR**: Applies when at least one of $\boldsymbol{c}$ or $\boldsymbol{b}$ is zero. When considered for overdetermined problems it is algebraically (but not numerically) equivalent to CGLS. It is more stable than CG and CGLS for ill-conditioned problems.
- **CS** (Chebyshev Semi-Iterative Method): Applies to the same class of systems as CG. It has fewer synchronization points in each iteration and so can take better advantage of parallelism. It requires a knowledge of an upper bound and a lower bound on the eigenvalues of the system matrix.
- **LSMR**: Applies to the same problems as LSQR. For overdetermined least squares it is algebraically equivalent to MINRES on the normal equations. The residual of the normal equations will decrease with each iteration, which makes it safer to stop early compared to LSQR.

</div>

These algorithms vary in how they accommodate preconditioners. Some require implicitly preconditioning the problem data, calling the "unpreconditioned" solver, then applying some (cheap) postprocessing to the returned solution. Any standard library implementing the drivers from Section 3 should include computational methods for (preconditioned) CG and LSQR. LSQR is most naturally applied to dual saddle point problems by reformulation to (3.14) and to primal saddle point problems by reformulation to (3.15).

## 3.4 Other Optimization Functionality

Here, we briefly discuss other RandNLA algorithms of note for least squares or optimization, often commenting on how they fit into plans for RandLAPACK.

**Facilitating second-order optimization algorithms.** Many second-order optimization algorithms need to solve sequences of saddle point systems, where $\boldsymbol{A}, \boldsymbol{b}, \boldsymbol{c}$ vary continuously from one iteration to the next. RandLAPACK will support such use-cases *indirectly* through methods that help amortize the dominant computational cost of a single randomized algorithm across multiple saddle point solves.

The most common way for $\boldsymbol{A}$ to vary is by a reweighting: $\boldsymbol{A} = \boldsymbol{W}\boldsymbol{A}_o$ for a fixed matrix $\boldsymbol{A}_o$ and an iteration-dependent matrix $\boldsymbol{W}$. The matrix $\boldsymbol{W}$ is typically (but not universally) a matrix square root of the Hessian of some separable convex function. The randomized algorithms described in this section will only be useful for such problems when $\boldsymbol{W}$ and its adjoint can be applied to $m$-vectors in $O(m)$ time. This condition is satisfied in limited but important situations such as in algorithms for logistic regression, linear programming, and iteratively-reweighted least squares.

**Stochastic Newton and subsampled Newton methods.** Newton Sketch is a prototype algorithm developed over two papers which is closely related to subsampled Newton methods. Each is suited to optimization problems that feature non-quadratic objective functions or problems with constraints other than linear equations. These methods entail sampling a new sketching operator (and applying it to a new data matrix) in each iteration. They are also well-suited to situations where sketching is essentially free but matrix-matrix products are relatively expensive, such as in distributed and streaming settings.

# Section 4: Low-rank Approximation

*Low-rank approximation* is a workhorse approach for reduced run time, reduced storage requirements, or improved interpretability when processing massive matrices. Given a target matrix $\boldsymbol{A}$, the task is to produce a suitably factored representation of a low-rank matrix $\hat{\boldsymbol{A}}$ of the same dimensions which approximates $\boldsymbol{A}$. We express a low-rank approximation as computing factor matrices $\boldsymbol{E}$ and $\boldsymbol{F}$ where

$$\boldsymbol{A} \approx \hat{\boldsymbol{A}} := \underset{m \times k}{\boldsymbol{E}} \;\underset{k \times n}{\boldsymbol{F}}$$

for some $k \ll \min\lbrace m, n \rbrace$. It is very common to have a $k \times k$ "inner factor" between $\boldsymbol{E}$ and $\boldsymbol{F}$.

The rank $k$ is a tuning parameter that controls the trade-off between approximation accuracy and data compression. Matrices that are well-approximated by a low-rank matrix but whose exact rank needed for a good approximation is unknown are called *numerically low-rank*.

## 4.1 Problem Classes

Low-rank approximation is naturally formalized as an optimization problem; one chooses $\hat{\boldsymbol{A}}$ and its factors to minimize a loss function subject to constraints. The most common loss functions are distances $\hat{\boldsymbol{A}} \mapsto \|\boldsymbol{A} - \hat{\boldsymbol{A}}\|$ induced by the Frobenius or spectral norms. Our two problem classes are:

- **Spectral decompositions** (Section 4.1.1): low-rank SVD and Hermitian eigendecomposition.
- **Submatrix-oriented decompositions** (Section 4.1.2): CUR and interpolative decompositions.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(4.1.1)</span></p>

Low-rank approximations that impose no requirements on $\hat{\boldsymbol{A}}$'s representation are briefly addressed in Section 4.3.6 in the context of computational routines. Methods for low-rank approximation with other representations (e.g., QR, UTV, LU, nonnegative factorization) are discussed in Section 4.4.

</div>

### 4.1.1 Spectral Decompositions

**Singular value decomposition.** The SVD is widely used to compute low-rank approximations and as a workhorse algorithm for PCA. Given an $m \times n$ matrix $\boldsymbol{A}$ where $m \ge n$, its SVD is

$$\boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^*, \quad \boldsymbol{U} \in \mathbb{R}^{m \times n},\; \boldsymbol{\Sigma} \in \mathbb{R}^{n \times n},\; \boldsymbol{V} \in \mathbb{R}^{n \times n},$$

where $\boldsymbol{U}$ and $\boldsymbol{V}$ are column-orthonormal and $\boldsymbol{\Sigma} = \operatorname{diag}(\sigma_1, \ldots, \sigma_n)$ with $\sigma_1 \ge \ldots \ge \sigma_n \ge 0$. The SVD can also be expressed as a sum of rank-one matrices: $\boldsymbol{A} = \sum_{i=1}^{n} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^*$.

For a matrix $\boldsymbol{A}$ with approximate low-rank structure, we obtain approximations with low exact rank by truncating the sum at some $k < r = \operatorname{rank}(\boldsymbol{A})$:

$$\boldsymbol{A} \approx \hat{\boldsymbol{A}}_k := \sum_{i=1}^{k} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^* = \boldsymbol{U}_k \boldsymbol{\Sigma}_k \boldsymbol{V}_k^*.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Eckart-Young-Mirsky)</span></p>

Truncating trailing singular values yields optimal rank-constrained approximations:

$$\hat{\boldsymbol{A}}_k \in \operatorname{argmin}_{\operatorname{rank}(\hat{\boldsymbol{A}}') = k} \|\boldsymbol{A} - \hat{\boldsymbol{A}}'\|.$$

This holds for every $k \in [\![r]\!]$ and for any unitarily invariant matrix norm, including the spectral norm and Frobenius norm. The reconstruction errors are:

$$\|\boldsymbol{A} - \hat{\boldsymbol{A}}_k\|_2 = \sigma_{k+1}(\boldsymbol{A}) \quad \text{and} \quad \|\boldsymbol{A} - \hat{\boldsymbol{A}}_k\|_{\mathrm{F}} = \sqrt{\sum_{j > k} \sigma_j^2(\boldsymbol{A})}.$$

</div>

**Hermitian eigendecomposition.** A matrix is called *Hermitian* if $\boldsymbol{A} = \boldsymbol{A}^*$. For real matrices this is the same as being symmetric. The eigendecomposition of a Hermitian matrix $\boldsymbol{A}$ is

$$\boldsymbol{A} = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V}^*, \quad \boldsymbol{V} \in \mathbb{R}^{n \times n},\; \boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \ldots, \lambda_n),$$

where $\boldsymbol{V}$ is orthogonal and $\boldsymbol{\Lambda}$ is real. A Hermitian matrix is *positive semidefinite* (psd) if $\lambda_i \ge 0$ for all $i$. We sort eigenvalues by decreasing absolute value: $|\lambda_1| \ge \cdots \ge |\lambda_n|$. Low-rank approximations are:

$$\boldsymbol{A} \approx \hat{\boldsymbol{A}}_k := \sum_{i=1}^{k} \lambda_i \boldsymbol{v}_i \boldsymbol{v}_i^* = \boldsymbol{V}_k \boldsymbol{\Lambda}_k \boldsymbol{V}_k^*.$$

A rank-$k$ eigendecomposition requires almost half the storage of a rank-$k$ SVD, and algorithms for computing low-rank eigendecompositions can leverage structure in the matrix for improved efficiency.

**Connections to principal component analysis.** PCA is a linear dimension reduction technique that forms $k$ new variables (components) $\boldsymbol{Z} = [\boldsymbol{z}_1, \ldots, \boldsymbol{z}_k]$ as linear combinations of the variables $\boldsymbol{X} = [\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n] \in \mathbb{R}^{m \times n}$ (assumed mean-centered). Specifically, $\boldsymbol{Z} = \boldsymbol{X}\boldsymbol{W}$ where $\boldsymbol{W} \in \mathbb{R}^{n \times k}$ are chosen so that $\boldsymbol{z}_1$ accounts for most of the variability in the data. PCA can be formulated as

$$\boldsymbol{w}_1 := \operatorname{argmax}_{\|\boldsymbol{w}\|_2^2 = 1} \operatorname{Var}(\boldsymbol{X}\boldsymbol{w}),$$

where $\operatorname{Var}(\boldsymbol{X}\boldsymbol{w}) := \frac{1}{m-1}\|\boldsymbol{X}\boldsymbol{w}\|_2^2$. Defining the sample covariance matrix $\boldsymbol{C} := \frac{1}{m-1}\boldsymbol{X}^*\boldsymbol{X}$, the weights $\boldsymbol{w}_1, \ldots, \boldsymbol{w}_k$ are found by diagonalizing $\boldsymbol{C} = \boldsymbol{W}\boldsymbol{\Lambda}\boldsymbol{W}^*$ and retaining the top $k$ eigenvectors.

- If the covariance matrix $\boldsymbol{C}$ is given directly (but $\boldsymbol{X}$ is not accessible), employ a low-rank Hermitian eigendecomposition.
- If the mean-centered data matrix $\boldsymbol{X}$ is given, the low-rank SVD is preferable: the $k$ weights $\boldsymbol{W} = [\boldsymbol{w}_1, \ldots, \boldsymbol{w}_k]$ are the top $k$ right singular vectors $\boldsymbol{V}_k = [\boldsymbol{v}_1, \ldots, \boldsymbol{v}_k]$, and the eigenvalues of $\boldsymbol{C}$ are $\frac{1}{m-1}\boldsymbol{\Sigma}^2$.

### 4.1.2 Submatrix-oriented Decompositions

Here we describe four types of submatrix-oriented decompositions: a *CUR decomposition* and three types of *interpolative decompositions*. Their value propositions include:

- Reduced storage requirements compared to spectral decompositions, especially for sparse data matrices.
- More transparent data interpretation when data is modeled as a matrix for convenience.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(CUR Decomposition)</span></p>

A **CUR decomposition** is a low-rank approximation of the form

$$\boldsymbol{A} \approx \underset{m \times k}{\boldsymbol{C}} \;\underset{k \times k}{\boldsymbol{U}} \;\underset{k \times n}{\boldsymbol{R}},$$

where $\boldsymbol{C}$ and $\boldsymbol{R}$ are formed by small subsets of actual columns and rows of $\boldsymbol{A}$, and the *linking matrix* $\boldsymbol{U}$ is chosen so that some norm of $\boldsymbol{A} - \boldsymbol{C}\boldsymbol{U}\boldsymbol{R}$ is small.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation for CUR)</span></p>

CUR decompositions have far lower storage requirements than partial SVDs when $\boldsymbol{A}$ is sparse. Experts often have a clear understanding of the meaning of certain columns and rows in a matrix, and this meaning is preserved by CUR. In contrast, SVD forms linear combinations of columns or rows that can destroy structures such as sparsity or nonnegativity.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Subtleties of CUR)</span></p>

CUR is unique among submatrix-oriented decompositions in that it involves taking products of submatrices of $\boldsymbol{A}$. If $\boldsymbol{A}$ has low numerical rank and its CUR is computed to high accuracy, then all three factors ($\boldsymbol{C}$, $\boldsymbol{U}$, $\boldsymbol{R}$) will be ill-conditioned, particularly $\boldsymbol{U}$.

</div>

**Interpolative decompositions.** Low-rank interpolative decompositions (IDs) come in three flavors sharing two common features: (1) exactly one of the factors is a submatrix of $\boldsymbol{A}$, and (2) the factors that are *not* submatrices of $\boldsymbol{A}$, called *interpolation matrices*, are subject to regularity conditions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(One-Sided Interpolative Decompositions)</span></p>

A **column ID** is an approximation $\boldsymbol{A} \approx \boldsymbol{C}\boldsymbol{X}$ where $\boldsymbol{C} = \boldsymbol{A}[:,J]$ is a small number of columns of $\boldsymbol{A}$ and $\boldsymbol{X}$ is a wide $k \times n$ interpolation matrix. A **row ID** is an approximation $\boldsymbol{A} \approx \boldsymbol{Z}\boldsymbol{R}$ where $\boldsymbol{R} = \boldsymbol{A}[I,:]$ is a small number of rows and $\boldsymbol{Z}$ is a tall $m \times k$ interpolation matrix. The submatrices $\boldsymbol{C}$ and $\boldsymbol{R}$ are represented by ordered index sets $J$ and $I$ called *skeleton indices*. The interpolation matrices satisfy the regularity condition

$$\boldsymbol{X}[:,J] = \boldsymbol{I}_k = \boldsymbol{Z}[I,:].$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(4.1.3 — Quality of Low-Rank IDs)</span></p>

Let $\tilde{\boldsymbol{A}}$ be any rank-$k$ approximation of $\boldsymbol{A}$ satisfying $\|\boldsymbol{A} - \tilde{\boldsymbol{A}}\|_2 \le \epsilon$. If $\hat{\boldsymbol{A}} = \tilde{\boldsymbol{A}}[:,J]\boldsymbol{X}$ for some $k \times n$ matrix $\boldsymbol{X}$ and an index vector $J$, then $\hat{\boldsymbol{A}} = \boldsymbol{A}[:,J]\boldsymbol{X}$ is a low-rank column ID satisfying

$$\|\boldsymbol{A} - \hat{\boldsymbol{A}}\|_2 \le (1 + \|\boldsymbol{X}\|_2)\epsilon.$$

Furthermore, if $|X_{ij}| \le M$ for all $(i,j)$, then $\|\boldsymbol{X}\|_2 \le \sqrt{1 + M^2 k(n-k)}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Two-Sided Interpolative Decomposition)</span></p>

A **two-sided ID** extends a one-sided ID by considering simultaneous row and column IDs. In the low-rank case, it is an approximation of the form

$$\boldsymbol{A} \approx \underset{m \times k}{\boldsymbol{Z}} \;\underset{k \times k}{\boldsymbol{A}[I,J]} \;\underset{k \times n}{\boldsymbol{X}},$$

where the inner factor is a submatrix of $\boldsymbol{A}$ and moderate requirements are placed on the outer factors (the interpolation matrices).

</div>

**Relationships between submatrix-oriented decompositions.** Algorithms for one-sided ID are the main building blocks of algorithms for low-rank two-sided ID. Several algorithms for CUR also depend on algorithms for one-sided ID. Algorithms for one-sided ID are designated as *computational routines*, while methods for CUR and two-sided ID are designated as *drivers*.

CUR and two-sided ID are "dual" to one another: for CUR, the outer factors are submatrices of $\boldsymbol{A}$ and no requirements are placed on the inner factor; for two-sided ID, the inner factor is a submatrix of $\boldsymbol{A}$ and moderate requirements are placed on the outer factors. If we specify the outer factors by ordered index sets $J$ and $I$, then a CUR decomposition is

$$\boldsymbol{A} \approx \boldsymbol{A}[:,J] \;\boldsymbol{U}\; \boldsymbol{A}[I,:].$$

### 4.1.3 On Accuracy Metrics

The main error metrics in low-rank approximation are norm-induced distances. One measures the distance from the approximation to the target, most often in the spectral or Frobenius norms.

**Distance to optimal approximations.** Truncating $\boldsymbol{A}$'s SVD at rank $k$ gives an optimal rank-$k$ approximation in any unitarily invariant norm. However, this truncation is non-unique when $\boldsymbol{A}$ has more than one singular value equal to $\sigma_k$. If $\boldsymbol{A}$ has multiple singular values *close to* $\sigma_k$, then extremely small perturbations to $\boldsymbol{A}$ can result in large changes to the singular vectors. It is harder to estimate the dominant $k$ singular vectors than to find a rank-$k$ approximation that is "near optimal."

**Intractability of computing optimal approximations.** With submatrix-oriented decompositions, finding an "optimal" ID with constraint $|X_{ij}| \le M$ is NP-hard even when $\hat{\boldsymbol{A}}$ has exact rank $k$ and $M = 1$. Algorithms such as strong rank-revealing QR can be applied with typical runtime $O(mnk)$, while ensuring $\max_{ij} |X_{ij}| \le 2$.

**Distance relative to a reference approximation.** One considers the distance between $\boldsymbol{A}$ and $\hat{\boldsymbol{A}}$ *relative to* that between $\boldsymbol{A}$ and some reference matrix $\boldsymbol{A}_r$:

$$\|\boldsymbol{A} - \hat{\boldsymbol{A}}\| \le (1 + \epsilon)\|\boldsymbol{A} - \boldsymbol{A}_r\|.$$

The reference matrices $\boldsymbol{A}_r$ are usually not available in practice and may not be optimal for the formal low-rank approximation problem at hand.

## 4.2 Drivers

There exist many randomized algorithms for computing low-rank approximations. This section focuses on algorithms that take the two-stage approach: first one constructs a "simple" representation of $\hat{\boldsymbol{A}}$ with the aid of randomization, and then one deterministically converts that representation into a more useful form.

Two important concepts for drivers:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(QB Decomposition)</span></p>

A **QB decomposition** is a simple representation useful for SVD and eigendecomposition. The representation takes the form $\hat{\boldsymbol{A}} = \boldsymbol{Q}\boldsymbol{B}$ for a tall matrix $\boldsymbol{Q}$ with orthonormal columns and $\boldsymbol{B} = \boldsymbol{Q}^*\boldsymbol{A}$. The QB decomposition involves explicit construction of and access to both $\boldsymbol{Q}$ and $\boldsymbol{B}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Column Subset Selection)</span></p>

**Column subset selection** (CSS) is the problem of selecting from a matrix a set of columns that is "good" in some sense. CSS algorithms largely characterize algorithms for *one-sided ID*. A one-sided ID can be used for the simple representation of $\hat{\boldsymbol{A}}$ when working toward an SVD, eigendecomposition, two-sided ID, or CUR decomposition.

</div>

### 4.2.1 Methods for SVD

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(SVD1: QB-Backed Low-Rank SVD)</span></p>

**function** $\texttt{SVD1}(\boldsymbol{A}, k, \epsilon, s)$

**Inputs:** $\boldsymbol{A}$ is an $m \times n$ matrix. The returned approximation will have rank at most $k$. The randomized phase attempts to approximate $\boldsymbol{A}$ to within $\epsilon$ error, but will not produce an approximation of rank greater than $k + s$.

**Output:** The compact SVD of a low-rank approximation of $\boldsymbol{A}$.

**Abstract subroutines:** $\texttt{QBDecomposer}$ generates a QB decomposition; it tries to reach a prescribed error tolerance but may stop early if it reaches a prescribed rank limit.

1. $\boldsymbol{Q}, \boldsymbol{B} = \texttt{QBDecomposer}(\boldsymbol{A}, k + s, \epsilon)$ &ensp; $\#$ $\boldsymbol{Q}\boldsymbol{B} \approx \boldsymbol{A}$
2. $r = \min\lbrace k, \text{number of columns in } \boldsymbol{Q}\rbrace$
3. $\boldsymbol{U}, \boldsymbol{\Sigma}, \boldsymbol{V}^* = \texttt{svd}(\boldsymbol{B})$
4. $\boldsymbol{U} = \boldsymbol{U}[:, \, :r]$, $\boldsymbol{V} = \boldsymbol{V}[:, \, :r]$, $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}[:r, \, :r]$
5. $\boldsymbol{U} = \boldsymbol{Q}\boldsymbol{U}$
6. **return** $\boldsymbol{U}, \boldsymbol{\Sigma}, \boldsymbol{V}^*$

</div>

The oversampling parameter $s$ is recommended to be a small positive number (e.g., $s = 5$ or $s = 10$) to account for the fact that the trailing singular vectors of $\boldsymbol{Q}\boldsymbol{B}$ may not be good estimates for the corresponding singular vectors of $\boldsymbol{A}$.

**Converting from an ID.** If $\hat{\boldsymbol{A}}$ is given in *any* compact representation then it can be losslessly converted into an SVD without ever accessing $\boldsymbol{A}$. For example, given a column ID $\hat{\boldsymbol{A}} = \boldsymbol{C}\boldsymbol{X}$, one computes a QR decomposition $\boldsymbol{C} = \boldsymbol{Q}\boldsymbol{R}$, sets $\boldsymbol{B} = \boldsymbol{R}\boldsymbol{X}$, and then proceeds with $(\boldsymbol{Q}, \boldsymbol{B})$ as in Algorithm SVD1.

**Single-pass algorithms.** For very large problems, the main measure of an algorithm's complexity is the number of times it moves $\boldsymbol{A}$ through fast memory. There are three algorithms for low-rank SVD which move $\boldsymbol{A}$ through fast memory only once, each using multi-sketching.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(4.2.1 — Pass-Efficient Algorithms)</span></p>

Algorithms designed to minimize the number of views of a matrix are usually analyzed in the *pass efficient model* for algorithm complexity. Early work on randomized pass-efficient and single-pass algorithms can be found in [FKV04; DKM06a; DKM06b].

</div>

### 4.2.2 Methods for Hermitian Eigendecomposition

Each randomized algorithm for low-rank SVD has a corresponding version specialized to Hermitian matrices.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(EVD1: QB-Backed Low-Rank Eigendecomposition)</span></p>

**function** $\texttt{EVD1}(\boldsymbol{A}, k, \epsilon, s)$

**Inputs:** $\boldsymbol{A}$ is an $n \times n$ Hermitian matrix. The returned approximation will have rank at most $k$. The randomized phase attempts to approximate $\boldsymbol{A}$ to within $\epsilon$ error, but will not produce an approximation of rank greater than $k + s$.

**Output:** Approximations of the dominant eigenvectors and eigenvalues of $\boldsymbol{A}$.

1. $\boldsymbol{Q}, \boldsymbol{B} = \texttt{QBDecomposer}(\boldsymbol{A}, k + s, \epsilon/2)$
2. $\boldsymbol{C} = \boldsymbol{B}\boldsymbol{Q}$ &ensp; $\#$ since $\boldsymbol{B} = \boldsymbol{Q}^*\boldsymbol{A}$, we have $\boldsymbol{C} = \boldsymbol{Q}^*\boldsymbol{A}\boldsymbol{Q}$
3. $\boldsymbol{U}, \boldsymbol{\lambda} = \texttt{eigh}(\boldsymbol{C})$ &ensp; $\#$ full Hermitian eigendecomposition
4. $r = \min\lbrace k, \text{number of entries in } \boldsymbol{\lambda}\rbrace$
5. $P = \texttt{argsort}(|\boldsymbol{\lambda}|)[:r]$
6. $\boldsymbol{U} = \boldsymbol{U}[:, P]$, $\boldsymbol{\lambda} = \boldsymbol{\lambda}[P]$
7. $\boldsymbol{V} = \boldsymbol{Q}\boldsymbol{U}$
8. **return** $\boldsymbol{V}, \boldsymbol{\lambda}$

</div>

The main difference between EVD1 and SVD1 is that $\epsilon$ is scaled down by a factor $1/2$ before being passed to the $\texttt{QBDecomposer}$, so that if $s = 0$ and $\|\boldsymbol{Q}\boldsymbol{B} - \boldsymbol{A}\| \le \epsilon$ then $\|\hat{\boldsymbol{A}} - \boldsymbol{A}\| \le \epsilon$.

**Converting from an ID.** One can use the symmetry of $\boldsymbol{A}$ to canonically approximate an initial row ID $\tilde{\boldsymbol{A}} = \boldsymbol{Z}\boldsymbol{A}[I,:] \approx \boldsymbol{A}$ by the Hermitian matrix $\hat{\boldsymbol{A}} = \boldsymbol{Z}\boldsymbol{A}[I,I]\boldsymbol{Z}^*$. This compact representation makes it easy to compute its eigendecomposition by a lossless process. The ID approach is of interest only when it moves $\boldsymbol{A}$ through fast memory *once*.

**Nystrom approximations for positive semidefinite matrices.** Suppose the $n \times n$ matrix $\boldsymbol{A}$ is psd. The *Nystrom approximation of $\boldsymbol{A}$ with respect to a matrix $\boldsymbol{X}$* is

$$\hat{\boldsymbol{A}} = (\boldsymbol{A}\boldsymbol{X})(\boldsymbol{X}^*\boldsymbol{A}\boldsymbol{X})^\dagger(\boldsymbol{A}\boldsymbol{X})^*.$$

In RandNLA, we set $\boldsymbol{X}$ to a sketching operator and produce a compact spectral decomposition $\hat{\boldsymbol{A}} = \boldsymbol{V}\operatorname{diag}(\boldsymbol{\lambda})\boldsymbol{V}^*$. For any given type of sketching operator, low-rank approximation of psd matrices by Nystrom approximations tend to be more accurate than comparable algorithms for general Hermitian eigendecomposition.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Disambiguating "Nystrom Approximation")</span></p>

The literature on randomized algorithms for Nystrom approximation heavily emphasizes column selection operators, stemming from an analogy between sampling columns of kernel matrices in machine learning and the Nystrom method from integral equation theory. However, *the term "Nystrom approximation" shall be used for any approximation of the form $(\boldsymbol{A}\boldsymbol{X})(\boldsymbol{X}^*\boldsymbol{A}\boldsymbol{X})^\dagger(\boldsymbol{A}\boldsymbol{X})^*$, even when $\boldsymbol{X}$ is a general sketching operator.*

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(EVD2: Nystrom-Based Eigendecomposition for PSD Matrices)</span></p>

**function** $\texttt{EVD2}(\boldsymbol{A}, k, s)$

**Inputs:** $\boldsymbol{A}$ is an $n \times n$ psd matrix. The returned approximation will have rank at most $k$, but the sketching operator can have rank as high as $k + s$.

**Output:** Approximations of the dominant eigenvectors and eigenvalues of $\boldsymbol{A}$.

1. $\boldsymbol{S} = \texttt{TallSketchOpGen}(\boldsymbol{A}, k + s)$
2. $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{S}$
3. $\nu = \sqrt{n} \cdot \epsilon_{\text{mach}} \cdot \|\boldsymbol{Y}\|$ &ensp; $\#$ regularize for numerical stability
4. $\boldsymbol{Y} = \boldsymbol{Y} + \nu\boldsymbol{S}$
5. $\boldsymbol{R} = \texttt{chol}(\boldsymbol{S}^*\boldsymbol{Y})$ &ensp; $\#$ $\boldsymbol{R}$ is upper-triangular, $\boldsymbol{R}^*\boldsymbol{R} = \boldsymbol{S}^*(\boldsymbol{A} + \nu\boldsymbol{I})\boldsymbol{S}$
6. $\boldsymbol{B} = \boldsymbol{Y}(\boldsymbol{R}^*)^{-1}$ &ensp; $\#$ $\boldsymbol{B}$ has $n$ rows and $k + s$ columns
7. $\boldsymbol{V}, \boldsymbol{\Sigma}, \boldsymbol{W}^* = \texttt{svd}(\boldsymbol{B})$ &ensp; $\#$ can discard $\boldsymbol{W}$
8. $\boldsymbol{\lambda} = \operatorname{diag}(\boldsymbol{\Sigma}^2)$
9. $r = \min\lbrace k, \text{number of entries in } \boldsymbol{\lambda} \text{ greater than } \nu\rbrace$
10. $\boldsymbol{\lambda} = \boldsymbol{\lambda}[:r] - \nu$ &ensp; $\#$ undo regularization
11. $\boldsymbol{V} = \boldsymbol{V}[:, :r]$
12. **return** $\boldsymbol{V}, \boldsymbol{\lambda}$

</div>

Unlike Algorithms SVD1 and EVD1, Algorithm EVD2 has no control of approximation error.

### 4.2.3 Methods for CUR and Two-Sided ID

**CUR by falling back on CSS.** The simplest approach computes the row and column indices $(I, J)$ in one stage and then computes the linking matrix $\boldsymbol{U}$ in a second stage. The column indices $J$ are obtained by a randomized algorithm for CSS on $\boldsymbol{A}$, then the row indices $I$ are obtained by CSS on $\boldsymbol{C}^* = \boldsymbol{A}[:,J]^*$. Two canonical choices for the linking matrix:

$$\boldsymbol{U}_{\text{proj}} = (\boldsymbol{A}[:,J])^\dagger \boldsymbol{A}\, (\boldsymbol{A}[I,:])^\dagger$$

and

$$\boldsymbol{U}_{\text{sub}} = \boldsymbol{A}[I,J]^\dagger.$$

It is generally preferable to use $\boldsymbol{U}_{\text{proj}}$ as the linking matrix, as the approximation error incurred will never be larger than when using $\boldsymbol{U}_{\text{sub}}$, and the computation of $\boldsymbol{U}_{\text{proj}}$ is better conditioned.

**CUR by a combination of column ID and CSS.** Given data $(\boldsymbol{X}, J)$ from a column ID, one recovers the row index set $I$ and $\boldsymbol{U}$ for a CUR decomposition by running CSS on $\boldsymbol{C}^* = \boldsymbol{A}[:,J]^*$ and setting $\boldsymbol{U} = \boldsymbol{X}\,(\boldsymbol{A}[I,:])^\dagger$. This requires only one application of a pseudo-inverse.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(CURD1: CUR by Randomizing an Initial ID)</span></p>

**function** $\texttt{CURD1}(\boldsymbol{A}, k, s)$

**Inputs:** $\boldsymbol{A}$ is an $m \times n$ matrix. The returned approximation will have rank at most $k$. The $\texttt{ColumnID}$ subroutine can use sketching operators of rank up to $k + s$.

**Output:** A low-rank CUR decomposition of $\boldsymbol{A}$.

1. **if** $m \ge n$ **then**
   - $\boldsymbol{X}, J = \texttt{ColumnID}(\boldsymbol{A}, k, s)$ &ensp; $\#$ $\boldsymbol{A}[:,J]\boldsymbol{X} \approx \boldsymbol{A}$
   - $\boldsymbol{Q}, \boldsymbol{T}, I = \texttt{qrcp}(\boldsymbol{A}[:,J]^*)$ &ensp; $\#$ only care about $I$
   - $I = I[:k]$
   - $\boldsymbol{U} = \boldsymbol{X}\,(\boldsymbol{A}[I,:])^\dagger$
2. **else**
   - $\boldsymbol{Z}^*, I = \texttt{ColumnID}(\boldsymbol{A}^*, k, s)$ &ensp; $\#$ $\boldsymbol{Z}\boldsymbol{A}[I,:] \approx \boldsymbol{A}$
   - $\boldsymbol{Q}, \boldsymbol{T}, J = \texttt{qrcp}(\boldsymbol{A}[I,:])$ &ensp; $\#$ only care about $J$
   - $J = J[:k]$
   - $\boldsymbol{U} = (\boldsymbol{A}[:,J])^\dagger \boldsymbol{Z}$
3. **return** $J, \boldsymbol{U}, I$

</div>

**Two-sided ID via one-sided ID.** Two-sided IDs are canonically computed by a simple reduction to one-sided ID: first obtain $(\boldsymbol{X}, J)$ by a column ID of $\boldsymbol{A}$ and then obtain $(I, \boldsymbol{Z})$ by a row ID of $\boldsymbol{A}[:,J]$. The initial column ID will be computed by a randomized algorithm, while the full-rank row ID $\boldsymbol{A}[:,J] = \boldsymbol{Z}\boldsymbol{A}[I,J]$ is computed by a deterministic method.

## 4.3 Computational Routines

Sections 4.3.1 to 4.3.4 cumulatively cover QB, column ID, CSS, and building blocks for the same.

### 4.3.1 Power Iteration

Given a matrix $\boldsymbol{A}$, suppose we sketch $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{S}$ using a very tall sketching operator $\boldsymbol{S}$. In a low-rank approximation context, it is generally preferable for $\operatorname{range}(\boldsymbol{Y})$ to be well-aligned with the span of $\boldsymbol{A}$'s dominant left singular vectors. This is facilitated by having $\operatorname{range}(\boldsymbol{S})$ well-aligned with the span of $\boldsymbol{A}$'s dominant *right* singular vectors.

A basic approach to **power iteration** makes alternating applications of $\boldsymbol{A}$ and $\boldsymbol{A}^*$ to an initial data-oblivious sketching operator $\boldsymbol{S}_o$, producing a data-aware sketching operator such as

$$\boldsymbol{S} = (\boldsymbol{A}^*\boldsymbol{A})^q \boldsymbol{S}_o \quad \text{or} \quad \boldsymbol{S} = (\boldsymbol{A}^*\boldsymbol{A})^q \boldsymbol{A}^* \boldsymbol{S}_o,$$

for some parameter $q \ge 0$. Practical implementations need to incorporate some form of stabilization in between the successive applications of $\boldsymbol{A}$ and $\boldsymbol{A}^*$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(4.3.1)</span></p>

The closest relative to Algorithm 8 (the general formulation of power iteration given in Appendix C.2.1) in the literature is probably [ZM20, Algorithm 3.3]. However, the core idea behind this algorithm was explored earlier by Bjarkason.

</div>

### 4.3.2 Orthogonal Projections: QB and Rangefinders

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(QB Decomposition and Rangefinder)</span></p>

Given a matrix $\boldsymbol{A}$, a **QB decomposition** is given by a pair of matrices $(\boldsymbol{Q}, \boldsymbol{B})$ where $\boldsymbol{Q}$ is column-orthonormal and $\boldsymbol{B} = \boldsymbol{Q}^*\boldsymbol{A}$. It is intended that $\boldsymbol{Q}\boldsymbol{B}$ serve as an approximation of $\boldsymbol{A}$. An algorithm that computes only the factor $\boldsymbol{Q}$ from a QB decomposition is called a **rangefinder**.

</div>

The value of QB decompositions stems from how they define approximations by orthogonal projection: $\hat{\boldsymbol{A}} = \boldsymbol{Q}\boldsymbol{B} = \boldsymbol{Q}\boldsymbol{Q}^*\boldsymbol{A}$. QB algorithms do *not necessarily* first compute $\boldsymbol{Q}$ and then set $\boldsymbol{B} = \boldsymbol{Q}^*\boldsymbol{A}$. The benefit of the rangefinder abstraction is that it considers an equivalent problem while setting aside the potentially-nuanced matter of computing $\boldsymbol{B}$.

**Rangefinder algorithms and basic QB.** The simplest rangefinders are based on power iteration. One prepares a data-aware sketching operator $\boldsymbol{S}$, computes $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{S}$, and returns $\boldsymbol{Q} = \texttt{orth}(\boldsymbol{Y})$. More advanced rangefinders use block Krylov subspace methods.

**Iterative QB algorithms.** The most effective QB algorithms work by building $(\boldsymbol{Q}, \boldsymbol{B})$ *iteratively*. Each iteration entails matrix multiplications with $\boldsymbol{A}$ (as a rangefinder step), adds columns to $\boldsymbol{Q}$ and rows to $\boldsymbol{B}$, and makes a suitable in-place update to $\boldsymbol{A}$. Iterations terminate once some metric of approximation error $\boldsymbol{A} \approx \boldsymbol{Q}\boldsymbol{B}$ falls below a certain level.

Iterative QB methods were improved by [YGL18]. Algorithm 2 of [YGL18] uses a power-iteration-based rangefinder to compute new blocks for $(\boldsymbol{Q}, \boldsymbol{B})$ and efficiently updates the Frobenius error $\|\boldsymbol{A} - \boldsymbol{Q}\boldsymbol{B}\|_{\mathrm{F}}$ as iterations proceed, giving complete control over the Frobenius norm error of the returned approximation.

**Stopping criteria for iterative QB algorithms.** The Frobenius norm is easily computed for sparse and dense matrices stored explicitly in memory. However, it can be difficult to compute for abstract linear operators accessed only via matrix-vector multiplies. One approach is to use a randomized Frobenius norm estimator as part of the QB decomposition. For high-quality approximations, bounding error in *spectral norm* is often preferred (the Frobenius norm is only used because it is cheap to compute). Section 4.3.5 reviews randomized algorithms for estimating matrix norms.

### 4.3.3 Column-Pivoted Matrix Decompositions

Column-pivoted matrix decompositions are deterministic algorithms that provide a natural way to compute one-sided IDs. QR decomposition with column pivoting (QRCP) is the most important example. Given an $m \times n$ matrix $\boldsymbol{A}$ and a target rank $k$, QRCP produces an index vector $J$, a column-orthonormal matrix $\boldsymbol{Q}$, and an upper-triangular matrix $\boldsymbol{R}$ for which $\boldsymbol{A}[:,J] = \boldsymbol{Q}\boldsymbol{R}$. Truncating to $k$ columns yields a column ID.

QRCP can also be used for CSS and as a building block within the drivers for CUR and two-sided ID (as seen in Algorithm CURD1). QRCP is typically applied to a *sketch* of $\boldsymbol{A}$ rather than to $\boldsymbol{A}$ directly, making the deterministic cost manageable even for very large matrices.

### 4.3.4 One-Sided ID and CSS

One-sided ID and CSS are closely related. A one-sided ID algorithm produces both skeleton indices and an interpolation matrix, while a CSS algorithm only produces the skeleton indices. Any one-sided ID algorithm can be used for CSS by discarding the interpolation matrix.

Randomized algorithms for one-sided ID typically proceed in two stages: (1) compute a sketch of the data matrix, and (2) apply a deterministic column-pivoted decomposition to the sketch. The sketch is computed using techniques from Section 2 (data-oblivious sketching operators) or Section 4.3.1 (power iteration for data-aware sketching). The deterministic step is almost always QRCP applied to the sketch.

### 4.3.5 Estimating Matrix Norms

Norm estimation is important for solving low-rank approximation problems to fixed accuracy. Randomized algorithms for estimating the spectral norm are well-studied in the NLA literature. Two classical methods stand out:

- **Power method** for estimating $\|\boldsymbol{A}\|_2$ or $\|\boldsymbol{A} - \boldsymbol{Q}\boldsymbol{B}\|_2$, based on iterating $\boldsymbol{v} \mapsto \boldsymbol{A}^*\boldsymbol{A}\boldsymbol{v} / \|\boldsymbol{A}^*\boldsymbol{A}\boldsymbol{v}\|$.
- **Probabilistic bounds** using the fact that $\|\boldsymbol{A}\boldsymbol{g}\|_2$ for a random Gaussian vector $\boldsymbol{g}$ concentrates around $\|\boldsymbol{A}\|_{\mathrm{F}}$.

These techniques are useful as stopping criteria for iterative QB algorithms and for certifying that a computed low-rank approximation has achieved a desired accuracy.

### 4.3.6 Oblique Projections

Oblique projections provide low-rank approximations that are cheap to compute but do not have meaningfully-structured factors. Given $\boldsymbol{A}$ and sketches $\boldsymbol{Y}_1 = \boldsymbol{A}\boldsymbol{S}_1$ and $\boldsymbol{Y}_2 = \boldsymbol{S}_2\boldsymbol{A}$, an oblique projection approximation takes the form

$$\hat{\boldsymbol{A}} = \boldsymbol{Y}_1 (\boldsymbol{S}_2 \boldsymbol{Y}_1)^\dagger \boldsymbol{Y}_2.$$

This framework does not impose orthogonality or submatrix constraints on the factors. The main appeal is computational efficiency: it requires only two sketches and a small pseudo-inverse computation.

## 4.4 Other Low-Rank Approximations

Low-rank approximation methods exist for representations beyond the spectral and submatrix-oriented decompositions covered in Sections 4.1 to 4.3. These include QR, UTV, URV, QLP, LU, and nonnegative matrix factorization (NMF). Some of these alternative representations arise naturally in specific application contexts where additional structure constraints are needed.

## 4.5 Existing Libraries

Several libraries implement randomized algorithms for low-rank approximation. Notable implementations exist in MATLAB, Python (via SciPy and scikit-learn), and C++. Since 2016, the `pca` function in scikit-learn defaults to a randomized algorithm when $d = \min\lbrace m, n\rbrace \ge 500$ and $k < 0.8d$.

# Section 5: Further Possibilities for Drivers

This section covers multi-purpose matrix decompositions, the solution of unstructured linear systems, and trace estimation. These are the last problems that might be handled by "drivers" in a high-level RandNLA library. The randomized algorithms here do not aim for an asymptotic speedup over deterministic methods. Rather, the aim is to significantly reduce time-to-solution by taking better advantage of modern computing hardware.

## 5.1 Multi-purpose Matrix Decompositions

Early in the year 2000, the IEEE publication *Computing in Science & Engineering* published a list of the top ten algorithms of the twentieth century. Among this list was *the decompositional approach to matrix computation*, on which G. W. Stewart gave the following remark.

> The underlying principle of the decompositional approach of matrix computation is that it is not the business of matrix algorithmists to solve particular problems but to construct computational platforms from which a variety of problems can be solved. This approach, which was in full swing by the mid-1960s, has revolutionized matrix computation.

This section covers three decompositions that provide broad platforms for problem solving, addressed in an order where randomization offers increasing benefits over purely deterministic algorithms.

### 5.1.1 QR Decomposition of Tall Matrices

Algorithms for computing unpivoted QR decompositions are true workhorses of numerical linear algebra. They are the foundation for the preferred algorithms for solving least squares problems with full-rank data matrices, and they are also an important ingredient in preprocessing for more expensive algorithms.

#### Orthogonality in the Standard Inner Product

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cholesky QR)</span></p>

**Cholesky QR** is a method for computing unpivoted QR decompositions of matrices with linearly independent columns. Given a full-column-rank matrix $\boldsymbol{A}$, the QR decomposition $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{R}$ can be computed as:

1. Compute a Cholesky decomposition of the Gram matrix $\boldsymbol{A}^*\boldsymbol{A} = \boldsymbol{R}^*\boldsymbol{R}$.
2. Perform a matrix-matrix triangular solve to obtain $\boldsymbol{Q} = \boldsymbol{A}\boldsymbol{R}^{-1}$.

The factor $\boldsymbol{R}$ is simply the upper-triangular Cholesky factor of $\boldsymbol{A}^*\boldsymbol{A}$. Implementing Cholesky QR only requires three functions: `syrk` from BLAS, `potrf` from LAPACK, and `trsm` from BLAS. These functions parallelize extremely well, so Cholesky QR can offer substantial speedups over Householder QR for very tall matrices.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cholesky QR Limitations)</span></p>

Despite the speed advantage of Cholesky QR, it is rarely used in practice since it is unsuitable for even moderately ill-conditioned matrices. Recently it has been shown that randomization can overcome this limitation by **preconditioning** Cholesky QR to ensure stability [FGL21]. The algorithm is called `RCholeskyQR2`.

</div>

#### Orthogonality in a Sketched Inner Product

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Randomized Gram-Schmidt — RGS)</span></p>

In [BG22], Balabanov and Grigori propose a **randomized Gram-Schmidt** (RGS) process that orthogonalizes $n$ vectors in $\mathbb{R}^m$ with respect to a sketched inner product

$$\langle \boldsymbol{u}, \boldsymbol{v} \rangle_{\boldsymbol{S}} = (\boldsymbol{S}\boldsymbol{u})^*(\boldsymbol{S}\boldsymbol{v}).$$

We call such vectors $\boldsymbol{S}$-**orthogonal** or **sketch-orthogonal**. When RGS is run on the columns of a matrix $\boldsymbol{A}$, values computed during sketched projections are assembled in an upper-triangular matrix $\boldsymbol{R}$ so that $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{R}$ and $(\boldsymbol{S}\boldsymbol{Q})^*(\boldsymbol{S}\boldsymbol{Q}) = \boldsymbol{I}_n$.

One can choose the distribution from which $\boldsymbol{S}$ is drawn so that $\boldsymbol{Q}$ will be nearly-orthonormal with respect to the standard inner product, with high probability. Empirical and theoretical results show RGS is faster than classic Gram-Schmidt but as stable as modified Gram-Schmidt.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch-Orthogonal QR Variants)</span></p>

The idea of computing QR decompositions where $\boldsymbol{Q}$ is sketch-orthogonal can be taken in several directions. A block version of RGS is proposed in [BG21]. Taking the block size to be the full number of columns, one can compute $\boldsymbol{R}$ by Householder QR on $\boldsymbol{S}\boldsymbol{A}$ and then represent $\boldsymbol{Q} = \boldsymbol{A}\boldsymbol{R}^{-1}$ as a linear operator. This procedure is the essence of sketch-and-precondition for least squares [RT08], and the algorithm is called `RCholeskyQR`.

</div>

### 5.1.2 QR Decomposition with Column Pivoting

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(QR with Column Pivoting — QRCP)</span></p>

Given a matrix $\boldsymbol{A}$, produce a column-orthogonal matrix $\boldsymbol{Q}$, an upper-triangular matrix $\boldsymbol{R}$, and a permutation vector $J$ so that

$$\boldsymbol{A}[:,J] = \boldsymbol{Q}\boldsymbol{R}.$$

The diagonal entries of $\boldsymbol{R}$ should approximate $\boldsymbol{A}$'s singular values, and the columns of $\boldsymbol{Q}$ should approximate $\boldsymbol{A}$'s left singular vectors. These stipulations reflect QRCP's main use cases: low-rank approximation and solving ill-conditioned least squares problems.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pivoting Strategy)</span></p>

If $m \ge n$, then for any permutation vector $J$, the economic QR decomposition of $\boldsymbol{A}[:,J]$ is unique. Therefore $J$ completely determines how well the columns of $\boldsymbol{Q}$ (resp., diagonal entries of $\boldsymbol{R}$) approximate the left singular vectors of $\boldsymbol{A}$ (resp., singular values of $\boldsymbol{A}$). The method of choosing pivots that sees the widest use today — a simple method based on column norms — was first described in [BG65].

</div>

#### An Established Randomized Algorithm for General Matrices

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(HQRRP — Householder QR with Randomization for Pivoting)</span></p>

This algorithm was first developed by Martinsson [Mar15] and Duersch and Gu [DG17], and then refined by Martinsson, Quintana-Ortí, Heavner, and van de Geijn [MHG17]. The factor $\boldsymbol{Q}$ from HQRRP is an $m \times m$ operator defined by $n$ Householder reflectors. The algorithm can run much faster than standard QRCP methods by processing the matrix in column blocks, casting the majority of its operations in terms of BLAS 3 (instead of about half BLAS 2).

**Outline:** Given a block size $b$ and an oversampling parameter $s$ (typical values: $b = 64$, $s = 10$), HQRRP starts by forming a thin $(b+s) \times n$ sketch $\boldsymbol{Y} = \boldsymbol{S}\boldsymbol{A}$, and then enters the following iterative loop:

1. Use any QRCP method to find $P_{\text{block}}$: the first $b$ pivots for $\boldsymbol{Y}$.
2. Process the panel $\boldsymbol{A}[:,P_{\text{block}}]$ by QRCP.
3. Suitably update $(\boldsymbol{A}, \boldsymbol{Y})$ and return to Step 1.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(HQRRP Performance)</span></p>

If done appropriately (particularly, by Duersch and Gu's method [DG17]) then the leading term in the FLOP count for HQRRP is identical to that of unpivoted Householder QR. The one downside is that the diagonal entries of $\boldsymbol{R}$ are not guaranteed to decrease across block boundaries. Two opportunities to improve performance are: (1) mixed-precision arithmetic for sketching and QRCP on the sketch, and (2) calling unpivoted QR on $\boldsymbol{A}_{\text{panel}}$ in the second phase of processing a block.

</div>

#### A Novel Randomized Algorithm for Very Tall Matrices

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(7 — QRCP via Sketch-and-Precondition and Cholesky QR)</span></p>

This algorithm overcomes the limitation of the preconditioned Cholesky QR methodology from [FGL21] of requiring full-rank data matrices. It uses a randomized preconditioner based on QRCP.

**Function:** $[\boldsymbol{Q}, \boldsymbol{R}, J] = \texttt{sap\_chol\_qrcp}(\boldsymbol{A}, d)$

**Inputs:** A matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, an integer $d$ satisfying $n \le d \ll m$.

**Output:** Column-orthonormal $\boldsymbol{Q} \in \mathbb{R}^{m \times k}$, upper-triangular $\boldsymbol{R} \in \mathbb{R}^{k \times n}$, and a permutation vector $J$ of length $n$.

1. $\boldsymbol{S} = \texttt{SketchOpGen}(d,m)$ — generate a $d \times m$ oblivious sketching operator.
2. $[\boldsymbol{Q}^{\text{sk}}, \boldsymbol{R}^{\text{sk}}, J] = \operatorname{qrcp}(\boldsymbol{S}\boldsymbol{A})$ — so that $\boldsymbol{S}\boldsymbol{A}[:,J] = \boldsymbol{Q}^{\text{sk}}\boldsymbol{R}^{\text{sk}}$.
3. $k = \operatorname{rank}(\boldsymbol{R}^{\text{sk}})$.
4. $\boldsymbol{A}^{\text{pre}} = \boldsymbol{A}[:,J[:k]](\boldsymbol{R}^{\text{sk}}[:k,:k])^{-1}$ — precondition.
5. $[\boldsymbol{Q}, \boldsymbol{R}^{\text{pre}}] = \texttt{chol\_qr}(\boldsymbol{A}^{\text{pre}})$ — Cholesky QR of preconditioned matrix.
6. $\boldsymbol{R} = \boldsymbol{R}^{\text{pre}}\boldsymbol{R}^{\text{sk}}[:k,:]$.
7. Return $\boldsymbol{Q}$, $\boldsymbol{R}$, $J$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(5.1.1)</span></p>

This monograph was released as a technical report in November 2022. It has come to our attention that Algorithm 7 was discovered slightly earlier by Balabanov; it is termed `RRRCholesyQR2` in arXiv:2210.09953:v2 [Bal22b].

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.1.2 — Correctness of Algorithm 7)</span></p>

Consider the context of Algorithm 7. If $\operatorname{rank}(\boldsymbol{S}\boldsymbol{A}) = \operatorname{rank}(\boldsymbol{A})$ then $\boldsymbol{A}[:,J] = \boldsymbol{Q}\boldsymbol{R}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conditioning of Algorithm 7)</span></p>

A practical implementation needs to consider finite-precision arithmetic. In particular, one cannot use the exact rank of $\boldsymbol{R}^{\text{sk}}$; a tolerance-based scheme would be needed. The main concern is the condition number of $\boldsymbol{A}^{\text{pre}}$. If that matrix is not well-conditioned, the factor $\boldsymbol{Q}$ from Cholesky QR may not be orthonormal to machine precision. More generally, if $\operatorname{cond}(\boldsymbol{A}^{\text{pre}}) \ge \epsilon^{-1/2}$ (where $\epsilon$ is the working precision), then it is possible for Cholesky QR to fail outright.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.1.3 — Conditioning of Preconditioned Matrix)</span></p>

Consider the context of Algorithm 7 and let $\boldsymbol{U}$ be an orthonormal basis for the range of $\boldsymbol{A}$. If $\operatorname{rank}(\boldsymbol{S}\boldsymbol{A}) = \operatorname{rank}(\boldsymbol{A})$, then the singular values of $\boldsymbol{A}^{\text{pre}}$ are the reciprocals of the singular values of $\boldsymbol{S}\boldsymbol{U}$.

Therefore if the distribution of the sketching operator is chosen judiciously, the algorithm should return an accurate decomposition with extremely high probability — the condition number of $\boldsymbol{A}^{\text{pre}}$ depends on neither the conditioning of $\boldsymbol{A}$ nor that of $\boldsymbol{A}^{\text{sk}}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Application to Matrices with Any Aspect Ratio)</span></p>

Although Cholesky QR only applies to very tall matrices, one could apply it to any $m \times n$ matrix $\boldsymbol{A}$ (with $m \ge n$) by processing the matrix in blocks. It would be natural to use Cholesky QR as the subroutine for processing a block of columns of $\boldsymbol{A}$ in HQRRP.

However, HQRRP's update rule for $\boldsymbol{A}$ requires that each panel's orthogonal factor is represented as a composition of $b$ Householder reflectors (each $m \times m$), whereas Cholesky QR returns an explicit $m \times b$ column-orthonormal $\boldsymbol{Q}$. This can be resolved using `sorhr_col` in LAPACK, which amounts to unpivoted LU factorization. While pairing Cholesky QR with `sorhr_col` will reduce its speed benefit, it may still be faster than Householder QR (`GEQRF`) and Tall-and-Skinny QR (`GEQR`) in certain settings.

</div>

### 5.1.3 UTV, URV, and QLP Decompositions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(UTV Decomposition)</span></p>

If QRCP cannot be relied upon to provide an adequate surrogate for the SVD, then one can consider decompositions of the form

$$\boldsymbol{A} = \boldsymbol{U}\boldsymbol{T}\boldsymbol{V}^*,$$

where $\boldsymbol{U}, \boldsymbol{V}$ are column-orthogonal and $\boldsymbol{T}$ is square and triangular.

- This recovers the SVD when $\boldsymbol{T}$ is the diagonal matrix of singular values of $\boldsymbol{A}$.
- This recovers QRCP when $\boldsymbol{V}$ is a permutation matrix.

These decompositions were first meaningfully studied by Stewart [Ste92; Ste93; Ste99]. They are known by various names, including *UTV*, *URV*, and *QLP*.

</div>

#### Deterministic Algorithms

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Stewart's UTV Algorithm)</span></p>

Stewart's best-known algorithm for UTV (see [Ste99]) proceeds as follows:

1. Run QRCP on the original matrix: $\boldsymbol{A} = \boldsymbol{Q}_1\boldsymbol{R}_1(\boldsymbol{P}_1)^*$.
2. Run QRCP on $(\boldsymbol{R}_1)^*$, to obtain $\boldsymbol{R}_1 = \boldsymbol{P}_2(\boldsymbol{R}_2)^*(\boldsymbol{Q}_2)^*$.
3. Grouping terms, we find

$$\boldsymbol{A} = \underbrace{(\boldsymbol{Q}_1\boldsymbol{P}_2)}_{\boldsymbol{U}} (\boldsymbol{R}_2)^* \underbrace{(\boldsymbol{P}_1\boldsymbol{Q}_2)^*}_{\boldsymbol{V}^*}.$$

Note that $\boldsymbol{T} = (\boldsymbol{R}_2)^*$ is *lower* triangular. The diagonal of $\boldsymbol{T}$ can track the singular values of $\boldsymbol{A}$ much better than the diagonal of $\boldsymbol{R}_1$, since the successive QRCP calls resemble QR iteration.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stewart's UTV Properties)</span></p>

Stewart's algorithm can be modified to interleave the computation of $\boldsymbol{R}_1$ with factoring $\boldsymbol{R}_1$ [Ste99, §5]. The resulting method, like QRCP, can be stopped early at a specified rank or once some accuracy metric is satisfied.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complete Orthogonal Decomposition — COD)</span></p>

There is a notion of a UTV decomposition that is not the SVD, not QRCP, and yet predates Stewart's UTV by several decades. The **complete orthogonal decomposition** (COD) is computed by one call to QRCP followed by one call to unpivoted QR [HL69]. Its main use is to facilitate the application of a pseudoinverse $\boldsymbol{A}^\dagger$ when $\boldsymbol{A}$ is rank-deficient.

</div>

#### Randomized Algorithms

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Randomized UTV Algorithms)</span></p>

- The first randomized algorithm for UTV was described in [DDH07, §5]. It used a random orthogonal transformation as a preconditioner for computing a COD, making it safe to replace the usual call to QRCP with a call to unpivoted QR.
- This approach was extended with power iteration ideas through the **PowerURV** algorithm [GM18, §3]. PowerURV obtains better approximations of the SVD than Stewart's UTV without using any pivoted QR decompositions.
- The earliest randomized algorithm for UTV that computes the decomposition *incrementally* is given in [MQH19, Figure 4]. It is the first full-rank UTV algorithm to use sketching (random dimension reduction) rather than random rotations.
- One can also trivially incorporate randomization into Stewart's UTV by using HQRRP for the requisite QRCP calls. The speedup would be fundamentally limited, but should still be observable for $n \times n$ matrices even when $n$ is as small as a few thousand.

</div>

## 5.2 Solving Unstructured Linear Systems

Two broad methodologies have emerged for incorporating randomization into general linear solvers. The first aims to ameliorate the cost of common safeguards that are applied to fast but potentially unreliable direct methods. The second aims to restructure computations in existing general-purpose iterative methods.

### 5.2.1 Direct Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(LU and LDL Decompositions)</span></p>

Direct methods for solving systems of linear equations center on finding a factored representation of the system matrix.

- **LU decomposition** of a general $n \times n$ matrix: $\boldsymbol{A} = \boldsymbol{L}\boldsymbol{U}$, where $\boldsymbol{L}$ is lower-triangular with unit diagonal ($L_{ii} = 1$) and $\boldsymbol{U}$ is upper-triangular.
- **LDL decomposition** for Hermitian matrices: $\boldsymbol{A} = \boldsymbol{L}\boldsymbol{D}\boldsymbol{L}^*$, where $\boldsymbol{L}$ is unit lower-triangular and $\boldsymbol{D}$ is block diagonal with blocks of size one and two.

Once in hand, they can be used to solve linear systems involving $\boldsymbol{A}$ in $O(n^2)$ operations. However, there are some nonsingular matrices for which these decompositions do not exist or cannot be computed stably in finite precision. Therefore they need to be carefully modified to ensure reliability.

</div>

#### Stability Through Randomized Pivoting

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pivoted LU and LDL)</span></p>

Pivoting is the standard paradigm to modify LU and LDL for improved numerical stability.

- **Partial pivoting** for LU: $\boldsymbol{P}\boldsymbol{A} = \boldsymbol{L}\boldsymbol{U}$.
- **Complete pivoting** for LU: $\boldsymbol{P}_1\boldsymbol{A}\boldsymbol{P}_2 = \boldsymbol{L}\boldsymbol{U}$.

Gaussian elimination with partial pivoting (GEPP) is substantially faster than GECP but has weaker theoretical guarantees. In [MG15], Melgaard and Gu propose a **randomized algorithm for partially pivoted LU** that makes pivoting decisions in a manner similar to HQRRP. The randomized algorithm achieves efficiency comparable to GEPP while also satisfying GECP-like element-growth bounds with high probability.

For LDL, pivoted decompositions take the form $\boldsymbol{A} = (\boldsymbol{P}\boldsymbol{L})\boldsymbol{D}(\boldsymbol{P}\boldsymbol{L})^*$. Notable methods include Bunch-Kaufman [BK77] and bounded Bunch-Kaufman (which uses rook pivoting) [AGL98]. In [FXG18], a randomized algorithm for pivoted LDL is proposed that is as stable as GECP and yet only slightly slower than Bunch-Kaufman.

</div>

#### Stability Through Randomized Rotations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Recursive Butterfly Transformations — RBTs)</span></p>

A **butterfly matrix** of size $2d \times 2d$ is a two-by-two block matrix with $d \times d$ diagonal matrices in each of the four blocks. A **recursive butterfly transformation** (RBT) is a product of a chain of matrices, each with butterfly matrices as diagonal blocks.

- RBTs of order $n$ (i.e., size $n \times n$) are usually analyzed when $n$ is a power of two.
- An RBT of order $n = 2^\ell$ can be applied to an $n$-vector in $O(n\ell)$ time.
- We are interested in RBTs that are both **orthogonal** and **random**, so the same FFT-like algorithm to apply the RBT can also apply its inverse.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Parker's RBT Approach)</span></p>

One of the major contributions of [Par95] was to prove that for any nonsingular matrix $\boldsymbol{A}$ of order $n$, one can sample $\boldsymbol{B}_1, \boldsymbol{B}_2$ iid from a certain distribution $\mathcal{D}_n$ over orthogonal RBTs so that $\boldsymbol{B}_1\boldsymbol{A}\boldsymbol{B}_2$ has an **unpivoted** LU decomposition with high probability. Put another way,

$$\boldsymbol{A} = (\boldsymbol{B}_1)^*\boldsymbol{L}\boldsymbol{U}(\boldsymbol{B}_2)^*$$

exists with high probability. The high speed at which RBTs can be applied and their excellent data locality properties have led to substantial interest in RBTs from the HPC community [BDH+13; LLD20].

The idea of using RBTs to precondition an "unsafe" unpivoted method also applies to LDL. In this case, one obtains factorizations of the form $\boldsymbol{A} = (\boldsymbol{B}\boldsymbol{L})\boldsymbol{D}(\boldsymbol{B}\boldsymbol{L})^*$, where $\boldsymbol{B}$ is the random RBT [BBB+14; BDR+17].

Remarkably, although RBTs seem predicated on destroying sparsity structure, the random RBT methodology can be applied to sparse matrices without catastrophic fill-in [BLR14].

</div>

### 5.2.2 Iterative Methods

#### Background on GMRES

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(GMRES and Krylov Subspaces)</span></p>

**GMRES** is a well-known iterative method for solving linear systems of the form $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ where $\boldsymbol{A}$ is $n \times n$ and nonsingular. The trajectory $(\boldsymbol{x}_p)_{p \ge 1}$ it generates has a simple variational characterization: $\boldsymbol{x}_p$ minimizes $L(\boldsymbol{x}) = \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2$ over all vectors $\boldsymbol{x}$ in the $p$-dimensional **Krylov subspace**

$$K_p = \operatorname{span}\lbrace \boldsymbol{b}, \boldsymbol{A}\boldsymbol{b}, \ldots, \boldsymbol{A}^{p-1}\boldsymbol{b} \rbrace.$$

The standard implementation uses the **Arnoldi process**, a specialization of modified Gram-Schmidt to orthogonalize implicitly-defined matrices $\boldsymbol{K}_p = [\boldsymbol{b}, \boldsymbol{A}\boldsymbol{b}, \ldots, \boldsymbol{A}^{p-1}\boldsymbol{b}]$. The Arnoldi process maintains a column-orthonormal matrix $\boldsymbol{V}_p$ where $\operatorname{range}(\boldsymbol{V}_p) = K_p$, and an **Arnoldi decomposition** $\boldsymbol{A}\boldsymbol{V}_p = \boldsymbol{V}_{p+1}\boldsymbol{H}_p$ in terms of an $n \times (p+1)$ column-orthonormal $\boldsymbol{V}_{p+1}$ and a $(p+1) \times p$ upper-Hessenberg matrix $\boldsymbol{H}_p$.

Letting $T_{\text{mv}}(\boldsymbol{A})$ denote the cost of a matrix-vector multiply with $\boldsymbol{A}$, the Arnoldi decomposition up to step $p$ can be computed in time $O(pT_{\text{mv}}(\boldsymbol{A}) + np^2)$.

</div>

#### Randomized GMRES: Arnoldi Decompositions in a Sketch-Orthogonal Basis

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketched Arnoldi Process)</span></p>

The method from [BG22, §4.2] can be interpreted as using a "sketched Arnoldi process" based on sketched Gram-Schmidt. It works by building up $\boldsymbol{V}_p$ so that its columns are $\boldsymbol{S}$-orthogonal in the sense of (5.1), where $\boldsymbol{S}$ is a $d \times n$ sketching operator ($p \lesssim d \ll n$). Along the way, it maintains an Arnoldi-like decomposition $\boldsymbol{A}\boldsymbol{V}_p = \boldsymbol{V}_{p+1}\boldsymbol{H}_p$, where $\boldsymbol{V}_{p+1}$ is likewise $\boldsymbol{S}$-orthogonal.

If $\delta$ is the effective distortion of $\boldsymbol{S}$ for the subspace $K_{p+1}$, then the solution $\boldsymbol{x}_{\text{sk}}$ obtained by the sketched Arnoldi approach will satisfy

$$\|\boldsymbol{A}\boldsymbol{x}_{\text{sk}} - \boldsymbol{b}\|_2 \le (1 + \delta)\|\boldsymbol{A}\boldsymbol{x}_\star - \boldsymbol{b}\|_2.$$

The big-$O$ time complexity is unchanged relative to the classic Arnoldi process. However, the flop count can be up to a factor of two smaller, the sketched process makes better use of BLAS 2 over BLAS 1, and it has fewer synchronization points compared to modified Gram-Schmidt.

</div>

#### Randomized GMRES: Handling General Non-Orthogonal Bases

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-Orthogonal Krylov Bases)</span></p>

Both classic GMRES and the sketched Arnoldi variant maintain Arnoldi-like decompositions $\boldsymbol{A}\boldsymbol{V}_p$ at a cost of $O(np^2)$ time complexity. This cost cannot be asymptotically reduced by forgoing the decomposition of $\boldsymbol{A}\boldsymbol{V}_p$, since building $\boldsymbol{V}_p$ with full orthogonalization already takes $O(np^2)$ time.

In [NT21], Nakatsukasa and Tropp identified that the unconstrained formulation

$$\min_{\boldsymbol{z}} \|\boldsymbol{A}\boldsymbol{V}_p\boldsymbol{z} - \boldsymbol{b}\|_2^2$$

has precisely the form needed to benefit from randomized algorithms, independent from how $\boldsymbol{V}_p$ and $\boldsymbol{A}\boldsymbol{V}_p$ are generated. One can compute $\boldsymbol{V}_p$ by a truncated $k$-step Arnoldi process for some $k \ll p$ in $O(npk)$ time. Alternatively, the Chebyshev method may be practical if one has knowledge of the spectrum of $\boldsymbol{A}$.

The solution is then obtained via sketch-and-solve by factoring $\boldsymbol{S}\boldsymbol{A}\boldsymbol{V}_p$. In exact arithmetic, the solutions from this method would coincide with those of sketched Arnoldi. However, in finite-precision arithmetic, if one is too lax in building the basis matrix $\boldsymbol{V}_p$ then the condition number of $\boldsymbol{A}\boldsymbol{V}_p$ can explode as $p$ increases.

</div>

#### Nested Randomization in Block-Projection and Block-Descent Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sketch-and-Project)</span></p>

**Sketch-and-project** is a template iterative algorithm for solving linear systems of the form $\boldsymbol{F}\boldsymbol{z} = \boldsymbol{g}$, where $\boldsymbol{F} \in \mathbb{R}^{M \times m}$ has at least as many rows as columns ($M \ge m$) [GR15]. Its special cases include **randomized Kaczmarz** [SV08] and **randomized block Kaczmarz** [NT14]. It also has variants specifically designed for overdetermined least squares problems [GIG21].

These methods share a significant weakness: their convergence rates worsen as one considers larger and larger problems. They are most likely to be useful when one cannot fit an $m \times m$ matrix in memory, but the subproblems encountered in sketch-and-project are amenable to methods from Section 3.

</div>

## 5.3 Trace Estimation

Many scientific computing and machine learning applications require estimating the trace of a square linear operator $\boldsymbol{A}$ that is represented implicitly. Randomized methods are especially effective for such problems.

### 5.3.1 Trace Estimation by Sampling

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Girard-Hutchinson Estimator)</span></p>

Let $\boldsymbol{A}$ be $n \times n$ and $\lbrace \boldsymbol{e}_1, \ldots, \boldsymbol{e}_n \rbrace$ be the standard basis vectors in $\mathbb{R}^n$. One can compute $\operatorname{tr}(\boldsymbol{A})$ exactly with $n$ matrix-vector products:

$$\operatorname{tr}(\boldsymbol{A}) = \sum_{i=1}^n \boldsymbol{e}_i^*\boldsymbol{A}\boldsymbol{e}_i.$$

Randomization allows estimating this quantity using $m \ll n$ matrix-vector multiplications. If $\boldsymbol{\omega} \sim \mathcal{D}$ is a random vector satisfying $\mathbb{E}[\boldsymbol{\omega}\boldsymbol{\omega}^*] = \boldsymbol{I}_n$, then $\operatorname{tr}(\boldsymbol{A}) = \mathbb{E}[\boldsymbol{\omega}^*\boldsymbol{A}\boldsymbol{\omega}]$. Drawing $m$ independent vectors $\boldsymbol{\omega}_i \sim \mathcal{D}$, the **Girard-Hutchinson estimator** is

$$\operatorname{tr}(\boldsymbol{A}) \approx \frac{1}{m}\sum_{i=1}^m \boldsymbol{\omega}_i^*\boldsymbol{A}\boldsymbol{\omega}_i.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Choice of Distribution for Trace Estimation)</span></p>

The idea for this method goes back to 1987 with work by Girard [Gir87], who proposed that $\mathcal{D}$ be the uniform distribution over the $\ell_2$ hypersphere with radius $\sqrt{n}$. Shortly thereafter, Hutchinson proposed taking $\mathcal{D}$ as a distribution over Rademacher random vectors [Hut90].

- **Hutchinson's choice** of $\mathcal{D}$ minimizes the variance of the estimator when $\boldsymbol{A}$ is fixed.
- **Girard's choice** minimizes the worst-case variance over sets of matrices that are closed under conjugation by unitary matrices.

Such estimators require $m \in \Omega(1/\epsilon^2)$ samples to approximate $\operatorname{tr}(\boldsymbol{A})$ to within $\epsilon$ error for some constant failure probability.

</div>

### 5.3.2 Trace Estimation with Help from Low-Rank Approximation

#### Compress and Trace

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compress-and-Trace Method)</span></p>

In [SAI17], Saibaba, Alexanderian, and Ipsen propose two randomized algorithms for estimating the trace of a psd linear operator $\boldsymbol{A}$. When $\boldsymbol{A}$ is accessible by matrix-vector products, the proposed method begins with a rangefinder step to find a column-orthonormal $n \times m$ matrix $\boldsymbol{Q}$ where $\boldsymbol{Q}\boldsymbol{Q}^*\boldsymbol{A}\boldsymbol{Q}\boldsymbol{Q}^* \approx \boldsymbol{A}$. The method then approximates

$$\operatorname{tr}(\boldsymbol{A}) \approx \operatorname{tr}(\boldsymbol{Q}\boldsymbol{Q}^*\boldsymbol{A}\boldsymbol{Q}\boldsymbol{Q}^*) = \operatorname{tr}(\boldsymbol{Q}^*\boldsymbol{A}\boldsymbol{Q}).$$

This can provide better relative error bounds than a Girard-Hutchinson estimator if $\boldsymbol{A}$'s spectral decay is sufficiently fast and $\boldsymbol{Q}$ is obtained by power iteration.

For the case where $\boldsymbol{A} = \log(\boldsymbol{I} + \boldsymbol{B})$ for a psd matrix $\boldsymbol{B}$, one finds $\boldsymbol{Q}$ so that $\boldsymbol{Q}\boldsymbol{Q}^*\boldsymbol{B}\boldsymbol{Q}\boldsymbol{Q}^*$ is a good low-rank approximation of $\boldsymbol{B}$, and then

$$\operatorname{tr}(\boldsymbol{A}) \approx \sum_{i=1}^m \log\!\left(1 + \lambda_i(\boldsymbol{Q}^*\boldsymbol{B}\boldsymbol{Q})\right).$$

</div>

#### Split, Trace, and Approximate

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Hutch++)</span></p>

In [MMM+21], Meyer et al. combined ideas from low-rank approximation with the Girard-Hutchinson estimator to obtain **Hutch++**. The algorithm proceeds as follows:

1. Sample a matrix $\boldsymbol{Q}$ uniformly at random from the set of $n \times m$ column-orthonormal matrices.
2. Define the low-rank approximation $\hat{\boldsymbol{A}} = \boldsymbol{Q}\boldsymbol{Q}^*\boldsymbol{A}\boldsymbol{Q}\boldsymbol{Q}^*$ and compute $\operatorname{tr}(\hat{\boldsymbol{A}}) = \operatorname{tr}(\boldsymbol{Q}^*\boldsymbol{A}\boldsymbol{Q})$.
3. Apply Girard-Hutchinson to the **deflated matrix** $\boldsymbol{\Delta} = (\boldsymbol{I} - \boldsymbol{Q}\boldsymbol{Q}^*)\boldsymbol{A}(\boldsymbol{I} - \boldsymbol{Q}\boldsymbol{Q}^*)$.
4. $\operatorname{tr}(\boldsymbol{A}) = \operatorname{tr}(\hat{\boldsymbol{A}}) + \operatorname{tr}(\boldsymbol{\Delta})$.

For psd matrices, Hutch++ can (with some small fixed failure probability) compute $\operatorname{tr}(\boldsymbol{A})$ to within $\epsilon$ relative error using only $O(1/\epsilon)$ matrix-vector products — a substantial improvement over the $O(1/\epsilon^2)$ products required by plain Girard-Hutchinson estimators. In fact, this sample complexity cannot be improved when considering a large class of algorithms [MMM+21, Theorems 4.1 and 4.2].

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hutch++ Extensions)</span></p>

- Persson, Cortinovis, and Kressner extended Hutch++ so that it can proceed adaptively, only terminating once some error tolerance has been achieved (up to a *controllable* failure probability) [PCK21].
- The modified Hutch++ method notably accommodates **symmetric indefinite matrices** $\boldsymbol{A}$. The accuracy guarantees for indefinite matrices cannot be as strong as those for positive definite matrices — relative error guarantees are essentially impossible when $\operatorname{tr}(\boldsymbol{A}) = 0$. Persson et al. therefore provide *additive* error guarantees in this setting.

</div>

#### Leveraging the Exchangeability Principle

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(XTrace)</span></p>

In [ETW23], Epperly, Tropp, and Webber develop a trace estimator based on the **exchangeability principle**. This principle stipulates that if an algorithm computes its estimate based on $m$ pairs $\lbrace (\boldsymbol{\omega}_i, \boldsymbol{A}\boldsymbol{\omega}_i)\rbrace_{i=1}^m$ where $\boldsymbol{\omega}_i$ are iid, then the minimum-variance unbiased estimator for $\operatorname{tr}(\boldsymbol{A})$ must be invariant under relabelings $\lbrace \boldsymbol{\omega}_i\rbrace_{i=1}^m \leftarrow \lbrace \boldsymbol{\omega}_{\sigma(i)}\rbrace_{i=1}^m$ for permutations $\sigma$.

Hutch++ does not respect the exchangeability principle since it uses randomness in two distinct stages. The **XTrace** algorithm can be thought of as a symmetrized version of Hutch++. Its $j$-th run uses $\boldsymbol{Q}_j = \operatorname{orth}([\boldsymbol{A}\boldsymbol{\omega}_i|_{i \ne j})$ and estimates $\operatorname{tr}(\boldsymbol{\Delta}_j)$ by $\boldsymbol{\omega}_j^*\boldsymbol{\Delta}_j\boldsymbol{\omega}_j$. A careful implementation achieves the same asymptotic complexity as Hutch++. XTrace also comes with adaptive-stopping and variance estimation methods.

</div>

### 5.3.3 Estimating the Trace of $f(\boldsymbol{B})$ via Integral Quadrature

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Lanczos Quadrature — SLQ)</span></p>

**Stochastic Lanczos quadrature** (SLQ) was introduced to the linear algebra community in [BFG96], popularized by [UCS17], and has since been extended in a few different ways [CH22; PK22; CTU22].

Consider how any function $f : \mathbb{R} \to \mathbb{R}$ can canonically be extended to act on a Hermitian matrix by acting separately on the eigenvalues. If $\boldsymbol{B} = \sum_{i=1}^n \lambda_i \boldsymbol{u}_i \boldsymbol{u}_i^*$ is the eigendecomposition, then

$$f(\boldsymbol{B}) = \sum_{i=1}^n f(\lambda_i)\boldsymbol{u}_i\boldsymbol{u}_i^*.$$

SLQ provides methods for approximating $\operatorname{tr}(f(\boldsymbol{B}))$ via quadrature-based methods that apply whenever $f$ is sufficiently smooth and $\boldsymbol{B}$ is Hermitian.

</div>

#### Technical Background

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Riemann-Stieltjes Integrals and Gaussian Quadrature)</span></p>

Let $\mu$ be a real-valued function on $\mathbb{R}$. The expression $\int_{\mathbb{R}} f(t)\,\mathrm{d}\mu(t)$ is called the **Riemann-Stieltjes integral** of $f$ against $\mu$. For our purposes $\mu$ is nondecreasing with $\mu(t) = \mu(L)$ for $t \le L$ and $\mu(t) = \mu(U)$ for $t \ge U$. Under these assumptions, the integral is well-defined whenever $f$ is continuous.

An $s$-point **quadrature rule** specifies $s$-vectors $\boldsymbol{w}$ (weights) and $\boldsymbol{\theta}$ (nodes) to approximate

$$\int_{\mathbb{R}} f(t)\,\mathrm{d}\mu(t) \approx \sum_{\ell=1}^s w_\ell f(\theta_\ell).$$

**Gaussian quadrature** achieves optimal sample complexity: it is exact for polynomials up to degree $2s - 1$, and no rule can guarantee exactness for polynomials of degree higher than $2s - 1$ with only $s$ samples.

There is a deep connection between Gaussian quadrature and orthogonal polynomials. Under our assumptions on $\mu$, one can define an inner product $\langle p, q \rangle_\mu = \int_{\mathbb{R}} p(t)q(t)\,\mathrm{d}\mu(t)$, leading to an orthonormal polynomial basis whose three-term recurrence coefficients form a tridiagonal **Jacobi matrix** $J$ of size $s \times s$. The nodes and weights of Gaussian quadrature against $\mu$ can be recovered from an eigendecomposition of $J$.

</div>

#### Stochastic Lanczos Quadrature: Approximating Girard-Hutchinson

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(SLQ Method)</span></p>

A Girard-Hutchinson estimator for the trace of $f(\boldsymbol{B})$ takes the form

$$T = \frac{1}{m}\sum_{i=1}^m T_i \quad \text{where} \quad T_i = \boldsymbol{\omega}_i^*f(\boldsymbol{B})\boldsymbol{\omega}_i$$

for independent random vectors $\boldsymbol{\omega}_i$ drawn from a suitable distribution. Each sample can be expressed as a Riemann-Stieltjes integral: upon setting

$$\mu_i(t) = \sum_{j=1}^n |\boldsymbol{\omega}_i^*\boldsymbol{u}_j|^2 \, u(t - \lambda_j),$$

where $u$ is the Heaviside step function, the identity $T_i = \int_{\mathbb{R}} f(t)\,\mathrm{d}\mu_i(t)$ is immediate from the definition of $f(\boldsymbol{B})$.

Rather than computing $f(\boldsymbol{B})\boldsymbol{\omega}_i$ directly, SLQ approximates each $T_i$ via **Lanczos quadrature**: the Lanczos algorithm computes an orthonormal basis for the $s$-dimensional Krylov subspace $\operatorname{span}\lbrace \boldsymbol{\omega}_i, \boldsymbol{B}\boldsymbol{\omega}_i, \ldots, \boldsymbol{B}^{(s-1)}\boldsymbol{\omega}_i \rbrace$, producing the Jacobi matrix whose eigendecomposition yields the nodes and weights for Gaussian quadrature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(SLQ Implementation Notes)</span></p>

SLQ entails approximating $m$ samples of the form $\boldsymbol{\omega}_i^*f(\boldsymbol{B})\boldsymbol{\omega}_i$, where each $\boldsymbol{\omega}_i$ is an independent random vector drawn from some distribution $\mathcal{D}$. The quality of each approximate sample depends on the number of nodes allowed in the Gaussian quadrature rule, and hence on the number of steps $s$ in the Lanczos algorithm.

- Taking $s$ steps of the Lanczos algorithm always requires $s - 1$ matrix-vector products with $\boldsymbol{B}$.
- At one end of the tradeoff, $s$ iterations of Lanczos with full orthogonalization costs $O(ns)$ storage and $O(ns^2)$ arithmetic.
- At the other end, performing no reorthogonalization reduces the costs to only $O(n)$ storage and $O(ns)$ arithmetic.
- SLQ is a powerful tool with important applications in Gaussian process regression. For an implementation that scales to petascale problems by running on GPU farms, see the **IMATE** Python package [Ame22].

</div>

#### Beyond Stochastic Lanczos Quadrature

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Accelerated Quadrature-Based Methods)</span></p>

Let $\boldsymbol{A} = f(\boldsymbol{B})$. The convergence rate of SLQ for estimating $\operatorname{tr}(\boldsymbol{A})$ can only be as good as a Girard-Hutchinson estimator. As such, one needs $m \in \Omega(1/\epsilon^2)$ samples to estimate $\operatorname{tr}(\boldsymbol{A})$ to within $\epsilon$ error. This leaves substantial room for improvement in the case when $\boldsymbol{A}$ is positive definite, where Hutch++ could make do with $m \in \Omega(1/\epsilon)$ queries to $\boldsymbol{A}$.

Luckily, it is possible to extend SLQ to use similar splitting techniques that Hutch++ employs for its variance reduction; see [CH22] and [PK22] for details.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Spectral Density Estimation)</span></p>

One of SLQ's remarkable properties is that its quadrature rule for approximating $T_i = \int_{\mathbb{R}} f(t)\,\mathrm{d}\mu_i(t)$ does not depend on $f$. As such, if the quadrature nodes and weights are computed to estimate $\operatorname{tr}(f(\boldsymbol{B}))$ for one function $f$, one can use those same nodes and weights to compute an estimate for $\operatorname{tr}(g(\boldsymbol{B}))$ for another function $g$. This gives motivation for directly estimating the function

$$\phi(t) = \sum_{j=1}^n u(t - \lambda_j),$$

which satisfies $\int_{\mathbb{R}} f(t)\,\mathrm{d}\phi(t) = \operatorname{tr}(f(\boldsymbol{B}))$ for all continuous functions $f : \mathbb{R} \to \mathbb{R}$. Note that $\phi$ is a nonnegative nondecreasing function with $\lim_{t \to \infty} \phi(t) = n$. As such, $\phi/n$ is a cumulative probability distribution function that can be uniquely identified with the spectrum of $\boldsymbol{B}$.

The problem of estimating $\phi$ is a particular case of **spectral density estimation**. In particular, [CTU22] provides a systematic treatment of quadrature-based trace estimation algorithms based on this perspective.

</div>

# Section 6: Advanced Sketching — Leverage Score Sampling

Leverage scores quantify the extent to which a low-dimensional subspace aligns with coordinate subspaces. They are fundamental to RandNLA theory since they determine how well a matrix can be approximated through sketching by row or column selection, and thus indirectly how well a matrix can be approximated by sparse data-oblivious sketching methods. They have algorithmic uses in least squares and low-rank approximation among other topics. More broadly, they play a key role in statistical regression diagnostics.

The computational value of leverage scores stems from how they induce data-aware probability distributions over the rows or columns of a matrix. *Leverage score sampling* refers to sketching by row or column sampling according to a leverage score distribution (or an approximation thereof). The quality of sketches produced by leverage score sampling is relatively insensitive to numerical properties of the matrix to be sketched — this can be contrasted with sketching by uniform row or column sampling, which can perform very poorly on certain families of matrices.

Leverage score distributions can be computed exactly with standard deterministic algorithms. However, exact computation is expensive except in very specific cases. Therefore in practice it is necessary to use randomized algorithms to *approximate* leverage score distributions.

## 6.1 Definitions and Background

Three types of leverage scores are covered here, along with corresponding approaches to leverage score sampling:

1. **Standard leverage scores** — applicable to sketching in the embedding regime, primarily for highly overdetermined least squares problems or other saddle point problems with tall data matrices.
2. **Subspace leverage scores** — used for sketching in the sampling regime with applications in low-rank approximation problems.
3. **Ridge leverage scores** — specifically for approximating psd matrices (typically kernel matrices) in the presence of explicit regularization.

### 6.1.1 Standard Leverage Scores

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Standard Leverage Scores)</span></p>

Let $U$ be an $n$-dimensional linear subspace of $\mathbb{R}^m$ and $\boldsymbol{P}_U$ be the orthogonal projector from $\mathbb{R}^m$ to $U$. The $i$-th **leverage score** of $U$ is

$$\ell_i(U) = \|\boldsymbol{P}_U \boldsymbol{\delta}_i\|_2^2 = \boldsymbol{P}_U[i, i],$$

where $\boldsymbol{\delta}_i$ is the $i$-th standard basis vector.

</div>

Collectively, leverage scores describe how well the subspace $U$ aligns with the standard basis in $\mathbb{R}^m$. They have algorithmic implications when we consider induced *leverage score distributions*, defined by

$$p_i(U) = \frac{\ell_i(U)}{\sum_{j=1}^{m} \ell_j(U)} = \frac{\ell_i(U)}{n}.$$

Given a matrix $\boldsymbol{A}$, one can associate as many sets of leverage scores to that matrix as one can associate subspaces to $\boldsymbol{A}$. Two of the most important such subspaces are $U = \operatorname{range}(\boldsymbol{A})$ and $V = \operatorname{range}(\boldsymbol{A}^*)$. Moving forward, we routinely replace $U$ by $\boldsymbol{A}$ with the understanding that $U = \operatorname{range}(\boldsymbol{A})$.

**Formulas for leverage scores.** There are many concrete ways to express the leverage scores of a tall $m \times n$ matrix $\boldsymbol{A}$. Using the matrix itself:

$$\ell_j(\boldsymbol{A}) = \boldsymbol{A}[j,:]\,(\boldsymbol{A}^*\boldsymbol{A})^\dagger\,\boldsymbol{A}[j,:]^*.$$

We can also use *any* matrix $\boldsymbol{U}$ whose columns form an orthonormal basis for $U = \operatorname{range}(\boldsymbol{A})$, since $\boldsymbol{P} = \boldsymbol{U}\boldsymbol{U}^*$:

$$\ell_j(\boldsymbol{A}) = \|\boldsymbol{U}[j,:]\|_2^2 = (\boldsymbol{U}\boldsymbol{U}^*)[j,j].$$

The subspace perspective shows that leverage scores are unchanged if $\boldsymbol{A}$ is replaced by $\boldsymbol{A}\boldsymbol{A}^*$. More generally, if $\boldsymbol{A} = \boldsymbol{E}\boldsymbol{F}$ and $\boldsymbol{F}$ has full row-rank then the leverage scores of $\boldsymbol{E}$ match those of $\boldsymbol{A}$.

#### Probabilistic Guarantees of Sketching via Row Sampling

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(6.1.1 — Row Sampling Quality)</span></p>

Suppose $\boldsymbol{A}$ is an $m \times n$ matrix of rank $n$. Suppose $\boldsymbol{S}$ is a wide $d \times m$ sketching operator that implements row sampling according to a probability distribution $\boldsymbol{q}$. Our measure of sketch quality is the smallest $\epsilon \in (0,1)$ where $\boldsymbol{y} \in \operatorname{range}(\boldsymbol{A})$ implies

$$(1 - \epsilon)\|\boldsymbol{y}\|_2^2 \le \|\boldsymbol{S}\boldsymbol{y}\|_2^2 \le (1 + \epsilon)\|\boldsymbol{y}\|_2^2.$$

If

$$r := \min_{j \in [\![m]\!]} \frac{q_j}{p_j(\boldsymbol{A})},$$

then for all $0 < \epsilon < 1$ we have

$$\Pr\left\lbrace \text{sketch fails for } (\boldsymbol{S}, \boldsymbol{A}, \epsilon) \right\rbrace \le 2n \left(\frac{\exp(\epsilon)}{(1+\epsilon)^{(1+\epsilon)}}\right)^{rd/n}$$

and $\exp(\epsilon) < (1+\epsilon)^{(1+\epsilon)}$.

</div>

The proposition's basic message is that the probability of $\boldsymbol{S}\boldsymbol{A}$ being a good sketch improves as $\boldsymbol{q}$ gets closer to the leverage score distribution $\boldsymbol{p}(\boldsymbol{A})$, where "closer" means that the value $r$ becomes larger.

With exact leverage score sampling we are fortunate to have $r = 1$, and so it suffices for the embedding dimension to satisfy

$$d_{\text{lev}} \in O\!\left(\frac{n \log n}{\epsilon^2}\right).$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coherence)</span></p>

The **coherence** of $\boldsymbol{A}$ is defined as

$$\mathscr{C}(\boldsymbol{A}) := m \max_{i \in [\![m]\!]} \ell_i(\boldsymbol{A}).$$

Coherence is bounded by $n \le \mathscr{C}(\boldsymbol{A}) \le m$. Uniform sampling leads to $r = n / \mathscr{C}(\boldsymbol{A})$, and so the embedding dimension for uniform sampling should be on the order of

$$d_{\text{unif}} \in O\!\left(\frac{\mathscr{C}(\boldsymbol{A}) \log n}{\epsilon^2}\right).$$

This is no better than leverage score sampling, and it can be *much* worse.

</div>

When using *approximate* leverage scores with a bound $q_j \ge \beta\, p_j(\boldsymbol{A})$ for all $j$, one has $\beta \le r$ and setting $d = d_{\text{lev}}/\beta$ would suffice to achieve the same guarantees as leverage score sampling.

#### Preconditioned Leverage Score Sampling, Hidden in Plain Sight

Many data-oblivious sketching operators can be described as applying a "rotation" and then performing coordinate subsampling. Two examples:

- A wide $d \times m$ Haar sketching operator $\boldsymbol{S}$ can be viewed as a composition of an $m \times m$ orthogonal matrix followed by a coordinate sampling operator.
- The diagonal sign flip and the fast trig transform in an SRFT amounts to a rotation, and the full action of the SRFT is just applying coordinate sampling to the rotated input.

In both cases, the rotation acts as a type of *preconditioner* for sampling — a transformation that converts a given problem into a related form that is more suitable for sampling methods. The example of SRFTs is especially informative, since using an embedding dimension $d \in O(n \log n)$ suffices for a $d \times m$ SRFT to be a subspace embedding with constant distortion with high probability.

### 6.1.2 Subspace Leverage Scores

The standard leverage scores are not suitable for low-rank approximation. Two problems arise: (1) asking for a low-rank approximation of an invertible matrix with many small singular values makes both row and column leverage scores uniform and uninformative, and (2) the map from a matrix to its leverage scores is not locally continuous at $\boldsymbol{A}$ whenever $\boldsymbol{A}$ is rank-deficient.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rank-$k$ Leverage Scores)</span></p>

These shortcomings can be partially addressed with the concept of **subspace leverage scores**, also called **rank-$k$ leverage scores** and **leverage scores relative to the best rank-$k$ approximation**.

Expressing the $m \times n$ matrix $\boldsymbol{A}$ by its compact SVD, $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^*$, the rank-$k$ leverage scores for its range are

$$\ell_j^k(\boldsymbol{A}) = \|\boldsymbol{U}[j, {:}k]\|_2^2.$$

The rank-$k$ leverage scores can be nonuniform regardless of the aspect ratio of the matrix. So long as $k < \operatorname{rank}(\boldsymbol{A})$, the rank-$k$ leverage scores of both $\operatorname{range}(\boldsymbol{A})$ and $\operatorname{range}(\boldsymbol{A}^*)$ can be nonuniform.

</div>

The problem of discontinuity of the map from a matrix to its rank-$k$ subspace leverage scores can still persist. These problems are less troublesome if one assumes that the $k$-th spectral gap $\sigma_k(\boldsymbol{A}) - \sigma_{k+1}(\boldsymbol{A})$ is bounded away from zero.

The rank-$k$ leverage score distribution is

$$p_j^k(\boldsymbol{A}) = \frac{\ell_j^k(\boldsymbol{A})}{\sum_{i=1}^{m} \ell_i^k(\boldsymbol{A})}.$$

If $\boldsymbol{S}$ denotes a $d \times m$ row-sampling operator induced by $\boldsymbol{p}^k(\boldsymbol{A})$, then the sketch $\boldsymbol{Y} = \boldsymbol{S}\boldsymbol{A}$ leads naturally to the approximation $\hat{\boldsymbol{A}} = \boldsymbol{A}\boldsymbol{Y}^\dagger \boldsymbol{Y}$. Letting $\boldsymbol{A}_k$ denote some best-rank-$k$ approximation of $\boldsymbol{A}$ in a unitarily invariant matrix norm $\|\cdot\|$, it is possible to choose $d$ sufficiently large so that

$$\|\boldsymbol{A} - \hat{\boldsymbol{A}}\| \lesssim \|\boldsymbol{A} - \boldsymbol{A}_k\|$$

holds with high probability. One may need $d \gg k$ to have any chance that this holds.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(6.1.2)</span></p>

One rarely samples according to an exact rank-$k$ leverage score distribution in practice. Rather, one uses randomized algorithms to approximate them. The key fact that enables this approximation is that leverage scores ("standard" or "subspace") are preserved if we replace $\boldsymbol{A}$ by $\boldsymbol{A}\boldsymbol{A}^*$. Moreover, as leverage scores quantify a notion of eigenvector localization, in many applications one has domain knowledge that eigenvalues should be localized, and this could be used to construct approximations.

</div>

### 6.1.3 Ridge Leverage Scores

Ridge leverage scores are used to approximate matrices in the presence of explicit regularization. That is, we are given an $m \times m$ psd matrix $\boldsymbol{K}$ and a positive regularization parameter $\lambda$, and we approximate $\boldsymbol{K} + \lambda\boldsymbol{I}$ by $\hat{\boldsymbol{K}} + \lambda\boldsymbol{I}$ where $\hat{\boldsymbol{K}}$ is a psd matrix of rank $n \ll m$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Effective Rank)</span></p>

What rank $n$ is needed for $(\hat{\boldsymbol{K}} + \lambda\boldsymbol{I})^{-1}$ to approximate $(\boldsymbol{K} + \lambda\boldsymbol{I})^{-1}$ up to some fixed accuracy? This is determined by the quantity $\operatorname{tr}(\boldsymbol{K}(\boldsymbol{K} + \lambda\boldsymbol{I})^{-1})$, which is called the **effective rank** of $\boldsymbol{K}$. Using $\mu_i$ to denote the $i$-th largest eigenvalue of $\boldsymbol{K}$, we can express the effective rank as

$$\operatorname{tr}(\boldsymbol{K}(\boldsymbol{K} + \lambda\boldsymbol{I})^{-1}) = \sum_{i=1}^{m} \frac{\mu_i}{\mu_i + \lambda}.$$

</div>

Since we are working with psd matrices it is natural to define $\hat{\boldsymbol{K}}$ as a Nyström approximation of $\boldsymbol{K}$ with respect to some sketching operator $\boldsymbol{S}$. This leaves the question of how to choose the distribution for $\boldsymbol{S}$. Taking $\boldsymbol{S}$ as a column-selection operator is especially appealing in settings where psd matrices arising in applications are defined implicitly through pairwise evaluations of a *kernel function* on a given dataset.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ridge Leverage Scores)</span></p>

The **ridge leverage scores** of $(\boldsymbol{K}, \lambda)$ are

$$\ell_i(\boldsymbol{K}; \lambda) = \left(\boldsymbol{K}\,(\boldsymbol{K} + \lambda\boldsymbol{I})^{-1}\right)[i, i].$$

In certain cases — particularly for estimating ridge leverage scores — it can be convenient to express these quantities in terms of a matrix $\boldsymbol{B}$ that satisfies $\boldsymbol{K} = \boldsymbol{B}\boldsymbol{B}^*$ and that has at least as many rows as columns. Specifically, by expressing $\boldsymbol{B}$ in terms of its compact SVD, one can show that

$$\ell_i(\boldsymbol{K}; \lambda) = \boldsymbol{b}_i^*\,(\boldsymbol{B}^*\boldsymbol{B} + \lambda\boldsymbol{I})^{-1}\,\boldsymbol{b}_i,$$

where $\boldsymbol{b}_i^*$ is the $i$-th row of $\boldsymbol{B}$.

</div>

## 6.2 Approximation Schemes

Computing leverage scores exactly is an expensive proposition. If $\boldsymbol{A}$ is a tall $m \times n$ matrix, then it takes $O(mn^2)$ time to compute the standard leverage scores exactly. If one is interested in subspace leverage scores and $k$ is small, then one can in principle use Krylov methods to approximate the dominant $k$ singular vectors in far less than $O(mn^2)$ time. If we want to compute the ridge leverage scores of an $m \times m$ matrix $\boldsymbol{K}$ exactly, then the straightforward implementation takes $O(m^3)$ time.

These facts necessitate the development of efficient and reliable methods for *leverage score estimation*.

### 6.2.1 Standard Leverage Scores

Suppose the $m \times n$ matrix $\boldsymbol{A}$ is very tall, i.e., $m \gg n$. Here we summarize a method by Drineas et al. that can compute approximate leverage scores, to within a constant multiplicative error factor, in $O(mn \log m)$ time — roughly the time it takes to implement a random projection — with some constant failure probability bounded away from one. This can offer improved efficiency over straightforward $O(mn^2)$ approaches when $m \gg n$ and yet $m \in o(2^n)$.

We set the stage by expressing leverage scores as follows:

$$\ell_j(\boldsymbol{A}) = \|\boldsymbol{\delta}_j^*\boldsymbol{U}\|_2^2 = \|\boldsymbol{\delta}_j^*\boldsymbol{U}\boldsymbol{U}^*\|_2^2 = \|\boldsymbol{\delta}_j^*\boldsymbol{A}\boldsymbol{A}^\dagger\|_2^2,$$

where the second equality follows from unitary invariance of the spectral norm. The method proceeds by approximating two operations: first we approximate the pseudoinverse of $\boldsymbol{A}$ and then we approximate the matrix-matrix product $\boldsymbol{A}\boldsymbol{A}^\dagger$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Approximate Standard Leverage Scores via Double Sketching)</span></p>

**Input:** Very tall $m \times n$ matrix $\boldsymbol{A}$.

1. Apply a wide $d_1 \times m$ SRFT $\boldsymbol{S}_1$ to the left of $\boldsymbol{A}$. Let $\boldsymbol{U}_1\boldsymbol{\Sigma}_1\boldsymbol{V}_1^*$ be an SVD of this $d_1 \times n$ sketched matrix $\boldsymbol{S}_1\boldsymbol{A}$.

2. Approximate: $\hat{\ell}_j(\boldsymbol{A}) = \|\boldsymbol{\delta}_j^*\boldsymbol{A}(\boldsymbol{S}_1\boldsymbol{A})^\dagger\|_2^2 = \|\boldsymbol{\delta}_j^*\boldsymbol{A}\boldsymbol{V}_1\boldsymbol{\Sigma}_1^{-1}\|_2^2$ at a cost of $O(d_1 n^2)$.

3. To avoid the $O(mn^2)$ cost of multiplying $\boldsymbol{A}$ with $\boldsymbol{V}_1\boldsymbol{\Sigma}_1^{-1}$, apply a tall sketching operator $\boldsymbol{S}_2$ of size $n \times d_2$ to the right of $\boldsymbol{V}_1\boldsymbol{\Sigma}_1^{-1}$ before multiplying it by $\boldsymbol{A}$:

$$\hat{\hat{\ell}}_j(\boldsymbol{A}) = \|\boldsymbol{\delta}_j^*\boldsymbol{A}(\boldsymbol{V}_1\boldsymbol{\Sigma}_1^{-1}\boldsymbol{S}_2)\|_2^2.$$

**Cost:** $O(d_1 n^2 + d_2 mn)$, i.e., $O(mn \log m)$ overall with appropriate $d_1, d_2$.

</div>

This estimation method can be adapted to efficiently compute "cross-leverage scores," as well as subspace leverage scores. It also has natural adjustments to make it faster. For example, replacing the SRFT $\boldsymbol{S}_1$ by $\tilde{\boldsymbol{S}}_1 = \boldsymbol{F}\boldsymbol{C}$ where $\boldsymbol{C}$ is a CountSketch and $\boldsymbol{F}$ is an SRFT that further compresses the output of $\boldsymbol{C}$; or replacing $\boldsymbol{S}_1$ by a SASO (a generalized CountSketch), which yields a similar speed-up.

### 6.2.2 Subspace Leverage Scores

There is a wide range of possibilities for estimating subspace leverage scores. Two methods are described here, both slightly adapted from [DMM+12].

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Approximate Rank-$k$ Leverage Scores via QB Decomposition)</span></p>

**Goal:** Estimate the rank-$k$ leverage scores of $\boldsymbol{A}$ for some $k \ll \min\lbrace m, n\rbrace$.

1. Compute a rank-$(k + s)$ QB decomposition of $\boldsymbol{A}$ (e.g., via a method from Section 4.3.2): $\boldsymbol{A} \approx \boldsymbol{Q}\boldsymbol{B}$, where the user specifies an oversampling parameter $s \in O(k)$.

2. Compute the top $k$ left singular vectors of $\boldsymbol{B}$ by some traditional method. Let $\boldsymbol{U}_k$ denote the $(k+s) \times k$ matrix of such leading left singular vectors.

3. Form the columns of $\boldsymbol{Q}\boldsymbol{U}_k$ as approximations of the leading $k$ left singular vectors of $\boldsymbol{A}$. The row-norms of this matrix define the approximate rank-$k$ leverage scores.

**Analysis:** When viewed as random variables, the returned leverage scores coincide with those of a rank-$k$ approximation $\hat{\boldsymbol{A}}$ where

$$\mathbb{E}\|\hat{\boldsymbol{A}} - \boldsymbol{A}\|_{\mathrm{F}}^2 \le (1 + \epsilon) \sum_{j > k} \sigma_j(\boldsymbol{A})^2.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Approximate Rank-$k$ Leverage Scores via Power Iteration)</span></p>

This is a two-stage method to find the leverage scores of a rank-$k$ approximation to $\boldsymbol{A}$ that is near-optimal in spectral norm.

**Stage 1:** Rather than setting $\boldsymbol{Q} = \operatorname{orth}(\boldsymbol{A}\boldsymbol{S})$ for Gaussian $\boldsymbol{S}$, set $\boldsymbol{S} = (\boldsymbol{A}^*\boldsymbol{A})^q \boldsymbol{S}_0$ for Gaussian $\boldsymbol{S}_0$. Compute $\boldsymbol{S}_{q+1} = (\boldsymbol{A}\boldsymbol{A}^*)^q \boldsymbol{A}\boldsymbol{S}_0$ from an $n \times 2k$ Gaussian operator $\boldsymbol{S}_0$.

**Stage 2:** Approximate leverage scores of $\boldsymbol{S}_{q+1}$ — call them $\hat{\ell}_i$ — from any method that ensures

$$|\hat{\ell}_i - \ell_i(\boldsymbol{S}_{q+1})| \le \epsilon\, \ell_i(\boldsymbol{S}_{q+1}).$$

These approximations are the estimates for the rank-$k$ leverage scores of $\boldsymbol{A}$. Specifically, for the prescribed $q$, the estimated leverage scores are within a factor $\frac{1-\epsilon}{2(1+\epsilon)}$ of the leverage scores of a rank-$k$ matrix $\hat{\boldsymbol{A}}$ that satisfies

$$\mathbb{E}\|\hat{\boldsymbol{A}} - \boldsymbol{A}\|_2 \le (1 + \epsilon/10)\,\sigma_{k+1}(\boldsymbol{A}).$$

</div>

### 6.2.3 Ridge Leverage Scores

A wide variety of algorithms have been devised to estimate ridge leverage scores or carry out approximate ridge leverage score sampling.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Simple Ridge Leverage Score Estimation)</span></p>

The simplest such algorithm, proposed in [AM15] alongside the definition of ridge leverage scores, proceeds as follows:

1. Start with a distribution $\boldsymbol{p} = (p_i)_{i \in [\![m]\!]}$ over the column index set of $\boldsymbol{K}$.

2. Construct a column selection operator $\boldsymbol{S}$ with $n$ columns, where each column is independently set to $\boldsymbol{\delta}_i \in \mathbb{R}^m$ with probability $p_i$.

3. Compute the Nyström approximation of $\boldsymbol{K}$ with respect to $\boldsymbol{S}$. Suppose the approximation is represented as $\hat{\boldsymbol{K}} = \boldsymbol{B}\boldsymbol{B}^*$ for an $m \times n$ matrix $\boldsymbol{B}$.

4. Using $\boldsymbol{b}_i \in \mathbb{R}^n$ for the $i$-th row of $\boldsymbol{B}$, take $\tilde{\ell}_i := \boldsymbol{b}_i^*(\boldsymbol{B}^*\boldsymbol{B} + \lambda\boldsymbol{I})^{-1}\boldsymbol{b}_i$ as an approximation for the $i$-th ridge leverage score of $\boldsymbol{K}$ with regularization $\lambda$.

One can start with $\boldsymbol{p} = (1/m)_{i \in [\![m]\!]}$ as the uniform distribution over columns of $\boldsymbol{K}$. An alternative starting point is $\boldsymbol{p} = \operatorname{diag}(\boldsymbol{K}) / \operatorname{tr}(\boldsymbol{K})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Iterative Methods for Ridge Leverage Scores)</span></p>

Iterative methods should be used if accurate approximations to ridge leverage scores are desired. Notably, most of the iterative methods in the literature *simultaneously* estimate the ridge leverage scores and sample columns from $\boldsymbol{K}$ according to the estimates. This algorithmic structure blurs the distinction between approximating ridge leverage scores and producing a Nyström approximation of $\boldsymbol{K}$ via column selection.

Letting $d = \operatorname{tr}(\boldsymbol{K}(\boldsymbol{K} + \lambda\boldsymbol{I})^{-1})$ denote the effective rank of $\boldsymbol{K}$, one can construct an approximation $\hat{\boldsymbol{K}}$ of rank $n \in O(d \log d)$ for which $\|\boldsymbol{K} - \hat{\boldsymbol{K}}\|_2 \le \lambda$ holds with high probability. Furthermore, these approximations can be constructed in time $O(mn^2)$ using only $n$ column samples from $\boldsymbol{K}$.

</div>

## 6.3 Special Topics and Further Reading

### 6.3.1 Leverage Score Sparsified Embeddings

The concept of long-axis-sparse operators from Section 2.4.2 is based on the *Leverage Score Sparsified* or *LESS* embeddings of Dereziński et al. [DLD+21].

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(LESS Embeddings)</span></p>

Let $\boldsymbol{S}$ be a random $d \times m$ long-axis-sparse operator ($d \ll m$) with sparsity parameter $k$ and sampling distribution $\boldsymbol{p} = (p_1, \ldots, p_m)$. The idea of LESS embeddings is that varying $k$ should provide a way to interpolate between the low cost of sketching by row sampling and the high cost of sketching by Gaussian operators, while still obtaining a sketch that is meaningfully Gaussian-like.

Indeed, if $k \approx n = \operatorname{rank}(\boldsymbol{A})$, then the resulting sketching operator is nearly indistinguishable from a dense sub-Gaussian operator (such as Gaussian or Rademacher), despite the reduction from $O(dmn)$ time to $O(dn^2)$.

</div>

As with other uses of leverage scores, approximate leverage scores suffice for LESS embeddings; and the computational cost of a LESS embedding is typically dominated by the cost of estimating the leverage scores of $\boldsymbol{A}$. The use of leverage scores in the sparsification pattern is essential for theoretically showing that a LESS embedding exhibits nearly identical performance to a Gaussian operator for all matrices $\boldsymbol{A}$.

### 6.3.2 Determinantal Point Processes

In many data analysis applications, submatrix-oriented decompositions such as Nyström approximation via column selection are desirable for their interpretability. Here we may wish to produce a very small but high-quality sketch of the matrix $\boldsymbol{A}$, using a method more refined (albeit slower) than leverage score sampling.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Determinantal Point Process)</span></p>

Let $\boldsymbol{A}$ be an $m \times m$ psd matrix. A **Determinantal Point Process** (DPP) is a distribution over index subsets $J \subseteq [\![m]\!]$ such that:

$$\mathbb{P}(J = S) = \frac{\det(\boldsymbol{A}[S, S])}{\det(\boldsymbol{A} + \boldsymbol{I})}.$$

This formulation is known as an **L-ensemble**, and it is also sometimes called **volume sampling**. Unlike leverage score sampling, individual indices sampled in a DPP are not drawn independently, but rather jointly, to minimize redundancies in the sampling process. A DPP can be viewed as an extension of leverage score sampling that incorporates dependencies between the samples, inducing diversity in the selected subset.

</div>

DPP sampling can be used to construct improved Nyström approximations $\hat{\boldsymbol{A}} = (\boldsymbol{A}\boldsymbol{S})(\boldsymbol{S}^*\boldsymbol{A}\boldsymbol{S})^\dagger(\boldsymbol{A}\boldsymbol{S})^*$ where the selection matrix $\boldsymbol{S}$ corresponds to the random subset $J$. Strong guarantees for this approach in terms of the nuclear norm error relative to the best rank-$k$ approximation:

$$\|\hat{\boldsymbol{A}} - \boldsymbol{A}\|_* \le (1 + \epsilon)\|\boldsymbol{A}_k - \boldsymbol{A}\|_*,$$

where $k$ is the target rank and the subset size $|J|$ is chosen to be equal or slightly larger than $k$. DPPs have also found applications in machine learning as a method for constructing diverse and interpretable data representations.

Two classes of efficient DPP sampling methods exploit the connection between DPPs and ridge leverage scores:

1. **Intermediate sampling + trimming:** Use intermediate sampling with ridge leverage scores to produce a larger index set $T$, which is then trimmed down to produce a smaller DPP sample $J \subseteq T$.

2. **Iterative refinement via Markov chain:** Start with an initial subset $J_1$, then gradually update it by swapping out one index at a time, producing a sequence of subsets $J_1, J_2, J_3, \ldots$, which rapidly converges to a DPP distribution.

### 6.3.3 Further Variations on Leverage Scores

In the case of tall data matrices $\boldsymbol{A}$, leverage scores are useful for finding data-aware sketching operators $\boldsymbol{S}$ so that the Euclidean norms of vectors in the range of $\boldsymbol{S}\boldsymbol{A}$ are comparable to the Euclidean norms of vectors in the range of $\boldsymbol{A}$. A related concept called *Lewis weights* can be used for matrix approximation where we want $\boldsymbol{S}$ to approximately preserve the $p$-norm of vectors in the range of $\boldsymbol{A}$ for some $p \ne 2$.

A more recently proposed concept samples according to the probabilities

$$p_i = \frac{\left\|(\boldsymbol{A}^*\boldsymbol{A})^{-1}\,\boldsymbol{A}[i,:]\right\|_2}{\sum_{j=1}^{m} \left\|(\boldsymbol{A}^*\boldsymbol{A})^{-1}\,\boldsymbol{A}[j,:]\right\|_2}$$

in order to estimate the *variability* of sketch-and-solve solutions to overdetermined least squares problems.

# Section 7: Advanced Sketching — Tensor Product Structures

This section considers efficient sketching of data with tensor product structure. We specifically focus on implicit matrices with Kronecker and Khatri–Rao product structure. These structures are of interest in RandNLA due to their prominent role in certain randomized algorithms for tensor decomposition. A secondary point of interest is that the operators discussed in this section can also be used for sketching unstructured matrices — for example, as alternatives to unstructured test vectors in norm and trace estimation. In this case, the main benefit would not be improved speed but reduced storage requirements for storing the sketching operator.

## 7.1 The Kronecker and Khatri–Rao Products

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kronecker Product)</span></p>

Suppose that $\boldsymbol{B}$ is an $m \times n$ matrix and $\boldsymbol{C}$ is a $p \times q$ matrix. The **Kronecker product** of $\boldsymbol{B}$ and $\boldsymbol{C}$ is the $mp \times nq$ matrix

$$\boldsymbol{B} \otimes \boldsymbol{C} = \begin{bmatrix} \boldsymbol{B}[1,1] \cdot \boldsymbol{C} & \boldsymbol{B}[1,2] \cdot \boldsymbol{C} & \cdots & \boldsymbol{B}[1,n] \cdot \boldsymbol{C} \\ \boldsymbol{B}[2,1] \cdot \boldsymbol{C} & \boldsymbol{B}[2,2] \cdot \boldsymbol{C} & \cdots & \boldsymbol{B}[2,n] \cdot \boldsymbol{C} \\ \vdots & \vdots & & \vdots \\ \boldsymbol{B}[m,1] \cdot \boldsymbol{C} & \boldsymbol{B}[m,2] \cdot \boldsymbol{C} & \cdots & \boldsymbol{B}[m,n] \cdot \boldsymbol{C} \end{bmatrix}.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Khatri–Rao Product)</span></p>

If $\boldsymbol{B}$ and $\boldsymbol{C}$ have the same number of columns (i.e., if $n = q$), then their **Khatri–Rao product** is the $mp \times n$ matrix

$$\boldsymbol{B} \odot \boldsymbol{C} = \begin{bmatrix} \boldsymbol{B}[:,1] \otimes \boldsymbol{C}[:,1] & \boldsymbol{B}[:,2] \otimes \boldsymbol{C}[:,2] & \cdots & \boldsymbol{B}[:,n] \otimes \boldsymbol{C}[:,n] \end{bmatrix}.$$

The Khatri–Rao product is sometimes also referred to as the *matching columnwise Kronecker product*. The Kronecker and Khatri–Rao products for more than two matrices are defined in the obvious way.

</div>

Note that for two vectors $\boldsymbol{x}$ and $\boldsymbol{y}$ we have

$$\boldsymbol{x} \otimes \boldsymbol{y} = \boldsymbol{x} \odot \boldsymbol{y} = \operatorname{vec}(\boldsymbol{x} \circ \boldsymbol{y}),$$

where $\circ$ denotes the outer product and $\operatorname{vec}(\cdot)$ is an operator that turns a matrix into a vector by vertically concatenating its columns. We also use $\circledast$ to denote the elementwise (Hadamard) product.

Matrices with Kronecker and Khatri–Rao product structure tend to be *very* large. For example, consider matrices $\boldsymbol{B}_1, \ldots, \boldsymbol{B}_L$, all of size $m \times n$. Their Kronecker product $\boldsymbol{B}_1 \otimes \cdots \otimes \boldsymbol{B}_L$ is an $m^L \times n^L$ matrix and their Khatri–Rao product $\boldsymbol{B}_1 \odot \cdots \odot \boldsymbol{B}_L$ is an $m^L \times n$ matrix. The exponential dependence on $L$ means that these products can become very large even if the matrices $\boldsymbol{B}_1, \ldots, \boldsymbol{B}_L$ are not especially large. Even just forming and storing these products may therefore be prohibitively expensive.

Kronecker and Khatri–Rao product matrices feature prominently in algorithms for tensor decomposition (i.e., decomposition of multidimensional arrays into sums and products of more elementary objects). They also appear in a variety of other contexts when sketching techniques are helpful, such as for representation of polynomial kernels, when fitting polynomial chaos expansion models in surrogate modeling, multi-dimensional spline fitting, and in PDE inverse problems.

## 7.2 Sketching Operators

The sketching operators in Sections 7.2.1–7.2.4 are all oblivious, whereas the sampling-based methods in Section 7.2.5 are not. All of the oblivious sketching operators discussed here could also be applied to *unstructured* matrices — yielding no speed benefit compared to their unstructured counterparts, but reducing the storage requirement compared to traditional dense sketching operators of the kind supported by the RandBLAS.

### 7.2.1 Row-Structured Tensor Sketching Operators

Three types of sketching operators whose rows can be applied to Kronecker and Khatri–Rao product matrices very efficiently are described here.

#### Khatri–Rao Products of Elementary Sketching Operators

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Khatri–Rao Sketching Operator)</span></p>

The most basic row-structured sketching operator takes the form

$$\boldsymbol{S} = (\boldsymbol{S}_1 \odot \boldsymbol{S}_2 \odot \cdots \odot \boldsymbol{S}_L)^*,$$

where each $\boldsymbol{S}_k$ is an appropriate random matrix of size $m_k \times d$ for $k \in [\![L]\!]$. Such an operator maps $(m_1 \cdots m_L)$-vectors to $d$-vectors. It can be efficiently applied to Kronecker product vectors, which in turn means that it can be applied efficiently (column-wise) to both Kronecker and Khatri–Rao product matrices.

Consider vectors $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_L$ where $\boldsymbol{x}_k$ is a length-$m_k$ vector. The operator is applied to a vector $\boldsymbol{v} = \boldsymbol{x}_1 \otimes \cdots \otimes \boldsymbol{x}_L$ via the formula

$$\boldsymbol{S}\boldsymbol{v} = (\boldsymbol{S}_1^*\boldsymbol{x}_1) \circledast (\boldsymbol{S}_2^*\boldsymbol{x}_2) \circledast \cdots \circledast (\boldsymbol{S}_L^*\boldsymbol{x}_L).$$

</div>

#### Row-wise Vectorized Tensors

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vectorized Tensor Sketching Operators)</span></p>

Rakhshan and Rabusseau [RR20] propose a distribution of sketching operators for which the $i$-th row is given by $\boldsymbol{S}[i,:] = \operatorname{vec}(\mathcal{X}_i)^*$, where $\mathcal{X}_i$ is a tensor in some factorized format and vec returns a vectorized version of its input.

Two cases are considered:

1. **CP format:** $\mathcal{X}_i$ is defined elementwise via

$$\mathcal{X}_i[j_1, j_2, \ldots, j_L] = \sum_{r=1}^{R} \boldsymbol{a}_r^{(i,1)}[j_1] \cdot \boldsymbol{a}_r^{(i,2)}[j_2] \cdots \boldsymbol{a}_r^{(i,L)}[j_L],$$

where the vector entries are drawn independently from an appropriately scaled Gaussian distribution.

2. **Tensor train format:** $\mathcal{X}_i$ is defined elementwise via

$$\mathcal{X}_i[j_1, j_2, \ldots, j_L] = \boldsymbol{A}_{j_1}^{(i,1)} \boldsymbol{A}_{j_2}^{(i,2)} \cdots \boldsymbol{A}_{j_L}^{(i,L)},$$

where each matrix $\boldsymbol{A}_{j_n}^{(i,n)}$ is of size $R_n \times R_{n+1}$ with $R_1 = R_{L+1} = 1$ so the product is a scalar.

For both constructs, the inner product of $\operatorname{vec}(\mathcal{X}_i)$ and Kronecker product vectors can be computed efficiently due to the special structure of the CP and tensor train formats.

</div>

#### Two-Stage Operators

Iwen et al. [INR+20] propose a two-stage sketching procedure for mapping $(m_1 \cdots m_L)$-vectors to $d$-vectors. The first step consists of applying a row-structured matrix $(\boldsymbol{S}_1 \otimes \cdots \otimes \boldsymbol{S}_L)$, where each $\boldsymbol{S}_k$ is a sketching operator of size $p_k \times m_k$. This maps the $(m_1 \cdots m_L)$-vector to an intermediate embedding space of dimension $(p_1 \cdots p_L)$. This is then followed by another sketching operator $\boldsymbol{T}$ of size $d \times (p_1 \cdots p_L)$ which maps the intermediate representation to the final $d$-dimensional space.

### 7.2.2 The Kronecker SRFT

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kronecker SRFT)</span></p>

Kronecker SRFTs are a variant of the SRFTs discussed in Section 2.5. They can be applied very efficiently to a Kronecker product vector without forming the vector explicitly. The Kronecker SRFT that maps $(m_1 \cdots m_L)$-vectors to $d$-vectors takes the form

$$\boldsymbol{S} = \sqrt{\frac{m_1 \cdots m_L}{d}}\,\boldsymbol{R}\left(\bigotimes_{k=1}^{L} \boldsymbol{F}_k\right)\left(\bigotimes_{k=1}^{L} \boldsymbol{D}_k\right),$$

where each $\boldsymbol{D}_k$ is a diagonal $m_k \times m_k$ matrix of independent Rademachers, each $\boldsymbol{F}_k$ is an $m_k \times m_k$ fast trigonometric transform, and $\boldsymbol{R}$ randomly samples $d$ components from an $(m_1 \cdots m_L)$-vector.

The Kronecker SRFT replaces the $\boldsymbol{F}$ and $\boldsymbol{D}$ operators in the standard SRFT by Kronecker products of smaller operators of the same form. With $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_L$ defined as in Section 7.2.1, the operator can be applied efficiently to $\boldsymbol{x}_1 \otimes \cdots \otimes \boldsymbol{x}_L$ via the formula

$$\boldsymbol{S}(\boldsymbol{x}_1 \otimes \cdots \otimes \boldsymbol{x}_L) = \sqrt{\frac{m_1 \cdots m_L}{d}}\,\boldsymbol{R}\left(\bigotimes_{k=1}^{L} \boldsymbol{F}_k \boldsymbol{D}_k \boldsymbol{x}_k\right).$$

Only those entries in $\bigotimes_k \boldsymbol{F}_k \boldsymbol{D}_k \boldsymbol{x}_k$ that are sampled by $\boldsymbol{R}$ need to be computed.

</div>

### 7.2.3 TensorSketch

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(TensorSketch)</span></p>

A **TensorSketch** operator is a kind of structured CountSketch that can be applied very efficiently to Kronecker product matrices. The improved computational efficiency comes at the cost of needing a larger embedding dimension than CountSketch. TensorSketch was first proposed in [Pag13] for fast approximate matrix multiplication.

Let $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_L$ be as in Section 7.2.1. A TensorSketch, which we denote by $\boldsymbol{S}$, maps an $(m_1 \cdots m_L)$-vector $\boldsymbol{v} = \boldsymbol{x}_1 \otimes \cdots \otimes \boldsymbol{x}_L$ to a $d$-vector via the formula

$$\boldsymbol{S}\boldsymbol{v} = \operatorname{DFT}^{-1}\left(\bigcircledast_{k=1}^{L} \operatorname{DFT}(\boldsymbol{S}_k \boldsymbol{x}_k)\right),$$

where each $\boldsymbol{S}_k$ is an independent CountSketch that maps $m_k$-vectors to $d$-vectors. Here, DFT denotes the discrete Fourier transform which can be efficiently applied using fast Fourier transform (FFT) methods. TensorSketches use the fact that polynomials can be multiplied using the DFT, which is why DFT and its inverse appear in the formula above.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(7.2.1)</span></p>

We have not investigated whether fast trig transforms other than the discrete Fourier transform (e.g., the discrete cosine transform) can be used for this type of sketching operator.

</div>

### 7.2.4 Recursive Sketching

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Embedding Dimension Scaling)</span></p>

In order to achieve theoretical guarantees, the sketching operators discussed so far require an embedding dimension $d$ which scales *exponentially* with $L$ when embedding a vector of the form $\boldsymbol{x}_1 \otimes \cdots \otimes \boldsymbol{x}_L$. Ahle et al. [AKK+20] propose sketching operators that are computed recursively and have the remarkable property that their requisite embedding dimensions scale *polynomially* with $L$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Recursive Sketching Operator)</span></p>

Suppose $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_L$ are $m$-vectors and that $L = 2^q$ for a positive integer $q$. The recursive sketching operator first computes

$$\boldsymbol{y}_k^{(0)} = \boldsymbol{T}_k \boldsymbol{x}_k \quad \text{for} \quad k \in [\![L]\!],$$

where $\boldsymbol{T}_1, \ldots, \boldsymbol{T}_L$ are independent SASOs (e.g., CountSketches) that map $m$-vectors to $d$-vectors. The $d$-vectors $\boldsymbol{y}_1^{(0)}, \ldots, \boldsymbol{y}_L^{(0)}$ are now combined pairwise into $L/2 = 2^{q-1}$ vectors by computing

$$\boldsymbol{y}_k^{(1)} = \boldsymbol{S}_k(\boldsymbol{y}_{2k-1}^{(0)} \otimes \boldsymbol{y}_{2k}^{(0)}) \quad \text{for} \quad k \in [\![L/2]\!],$$

where $\boldsymbol{S}_1, \ldots, \boldsymbol{S}_{L/2}$ are independent sketching operators that map $d^2$-vectors to $d$-vectors. If the initial $\boldsymbol{T}_1, \ldots, \boldsymbol{T}_L$ are CountSketches then the $\boldsymbol{S}_i$ are canonically TensorSketches. If instead $\boldsymbol{T}_1, \ldots, \boldsymbol{T}_L$ are more general SASOs then the $\boldsymbol{S}_i$ are canonically Kronecker SRFTs.

Regardless of which configuration we use, the pairwise combination of vectors is repeated for a total of $q = \log_2(L)$ steps until a single $d$-vector remains, which is the embedding of $\boldsymbol{x}_1 \otimes \cdots \otimes \boldsymbol{x}_L$. When $L$ is not a power of two, this is handled by adding additional vectors $\boldsymbol{x}_k = \boldsymbol{e}_1$ for $k = L+1, \ldots, 2^{\lceil \log_2(L) \rceil}$.

Recursive sketching operators are linear despite their somewhat complicated description.

</div>

The recursive sketching operator by [AKK+20] can be described by a binary tree, with each node corresponding to an appropriate sketching operator. Ma and Solomonik [MS22] generalize this idea by allowing for other graph structures, but limit nodes in these graphs to be associated with Gaussian sketching operators. Under this framework, they develop a structured sketching operator whose embedding dimension only scales *linearly* with $L$. These operators can be adapted for efficient application to vectors with general tensor network structure which includes Kronecker products of vectors as a special case.

### 7.2.5 Leverage Score Sampling for Implicit Matrices with Tensor Product Structures

Consider the problem of sketching and solving a least squares problem

$$\min_{\boldsymbol{x}} \|\boldsymbol{A}\boldsymbol{X} - \boldsymbol{Y}\|_{\mathrm{F}}$$

when the columns of $\boldsymbol{A}$ have tensor product structure and $\boldsymbol{Y}$ is a thin unstructured matrix. The sketching operators discussed so far in this section can be efficiently applied to $\boldsymbol{A}$. However, since $\boldsymbol{Y}$ lacks structure, these sketching operators require accessing all nonzero elements of $\boldsymbol{Y}$. This can be prohibitively expensive in applications such as:

- **Tensor decomposition:** One typically solves a sequence of least squares problems for which $\boldsymbol{A}$ is structured and $\boldsymbol{Y}$ contains all the entries of the tensor being decomposed. When $\boldsymbol{Y}$ has a fixed proportion of nonzero entries, the cost will scale exponentially with the number of tensor indices — a manifestation of the *curse-of-dimensionality*.

- **Surrogate modeling:** $\boldsymbol{A}$ contains evaluations of a multivariate polynomial on a structured quadrature grid and $\boldsymbol{Y}$ (which will now be a column vector) contains the outputs of an expensive data generation process.

In both example applications, it is clearly desirable to avoid using all entries of $\boldsymbol{Y}$. As discussed in Section 6, leverage score sampling can be used to sketch-and-solve least squares problems without accessing all entries of the right-hand side $\boldsymbol{Y}$ while still providing performance guarantees.

#### Kronecker Product Structure

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Leverage Score Factorization for Kronecker Products)</span></p>

Consider a Kronecker product $\boldsymbol{A} = \boldsymbol{A}_1 \otimes \cdots \otimes \boldsymbol{A}_L$ of $m_k \times n_k$ matrices $\boldsymbol{A}_k$. It is possible to perform *exact* leverage score sampling on $\boldsymbol{A}$ without even forming it.

Let $(p_i)$ be the leverage score sampling distribution of $\boldsymbol{A}$ and let $(p_{i_k})$ be the leverage score sampling distribution of $\boldsymbol{A}_k$ for $k \in [\![L]\!]$. For any $i \in [\![\prod_{k=1}^{L} m_k]\!]$ and corresponding multi-index $(i_1, \ldots, i_L)$ satisfying

$$\boldsymbol{A}[i,:] = \boldsymbol{A}_1[i_1,:] \otimes \cdots \otimes \boldsymbol{A}_L[i_L,:],$$

it holds that

$$p_i = p_{i_1}^{(1)} p_{i_2}^{(2)} \cdots p_{i_L}^{(L)}.$$

Therefore, instead of drawing an index $i$ according to $(p_i)$, one can draw the index $i_k$ according to $(p_{i_k}^{(k)})$ for each $k \in [\![L]\!]$. Due to the factorization above, the row corresponding to the drawn index can be computed and rescaled without constructing $\boldsymbol{A}$.

</div>

#### Khatri–Rao Product Structure

Sampling according to the leverage scores of a Khatri–Rao product matrix $\boldsymbol{A} = \boldsymbol{A}_1 \odot \cdots \odot \boldsymbol{A}_L$ is more challenging than it is for a Kronecker product matrix. Two categories of approaches have been proposed:

**Category 1: Sampling according to Kronecker product leverage scores.** As noted by [CPL+16; BBK18], the leverage scores of $\boldsymbol{A}$ can be upper bounded by

$$\ell_i(\boldsymbol{A}) \le \prod_{k=1}^{L} \ell_{i_k}(\boldsymbol{A}_k),$$

where $(i_1, \ldots, i_L)$ is the multi-index corresponding to $i$. Using the expression on the right-hand side as an approximation to the exact leverage scores on the left-hand side provides theoretical performance guarantees when this approach is used for sketch-and-solve in least squares problems.

**Category 2: Sampling according to exact or high-accuracy approximations of leverage scores.** Malik [Mal22] proposes a different approach for the Khatri–Rao product least squares problem. By combining some of the ideas for fast leverage score estimation (see Section 6.2) and recursive sketching (see Section 7.2.4) with a sequential sampling approach, he improves the sampling and computational complexities. An upshot of this work is a method for efficiently sampling a Khatri–Rao product matrix according to its *exact* leverage score distribution without forming the matrix.

## 7.3 Partial Updates to Kronecker Product Sketches

The structured sketching operators discussed in Section 7.2 are notable in that they are defined in terms of multiple smaller sketching operators. Here we discuss situations when it is advantageous to reuse some of these smaller sketches across multiple calls to the structured sketching operator. The examples come from works that use sketching in tensor decomposition algorithms.

By tensor, we mean multi-index arrays containing real numbers. A tensor with $L$ indices is called an *$L$-way tensor*. Vectors and matrices are one- and two-way tensors, respectively. Calligraphic capital letters (e.g., $\mathcal{X}$) are used to denote tensors with three or more indices. Much like matrix decomposition, the purpose of tensor decomposition is to decompose a tensor into some number of simpler components.

### 7.3.1 Background on the CP Decomposition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(CP Decomposition)</span></p>

The **CANDECOMP/PARAFAC (CP) decomposition** (also known as the canonical polyadic decomposition) decomposes an $L$-way tensor $\mathcal{X}$ of size $m_1 \times m_2 \times \cdots \times m_L$ into a sum of $R$ rank-1 tensors:

$$\mathcal{X} = \sum_{r=1}^{R} \boldsymbol{a}_r^{(1)} \circ \boldsymbol{a}_r^{(2)} \circ \cdots \circ \boldsymbol{a}_r^{(L)},$$

where $\circ$ denotes the outer product. The $m_n \times R$ matrices $\boldsymbol{A}^{(n)} = \begin{bmatrix} \boldsymbol{a}_1^{(n)} & \cdots & \boldsymbol{a}_R^{(n)} \end{bmatrix}$ for $n \in [\![L]\!]$ are called **factor matrices**. When $R$ is sufficiently large, we can express the factor matrices as the solution to

$$\arg\min_{\boldsymbol{A}^{(1)},\ldots,\boldsymbol{A}^{(L)}} \left\|\mathcal{X} - \sum_{r=1}^{R} \boldsymbol{a}_r^{(1)} \circ \boldsymbol{a}_r^{(2)} \circ \cdots \circ \boldsymbol{a}_r^{(L)}\right\|_{\mathrm{F}},$$

where $\|\cdot\|_{\mathrm{F}}$ denotes the Frobenius norm as generalized to tensors.

</div>

It is computationally intractable to solve the CP optimization problem in the general case. However, the problem admits several heuristics that are effective in practice.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Alternating Least Squares for CP Decomposition)</span></p>

One of the most popular heuristics is **alternating minimization**, wherein one solves for only one factor matrix at a time while keeping the others fixed. That is, one solves a sequence of problems of the form

$$\boldsymbol{A}^{(n)} = \arg\min_{\boldsymbol{A}} \left\|\mathcal{X} - \sum_{r=1}^{R} \boldsymbol{a}_r^{(1)} \circ \cdots \circ \boldsymbol{a}_r^{(n-1)} \circ \boldsymbol{a}_r \circ \boldsymbol{a}_r^{(n+1)} \circ \cdots \circ \boldsymbol{a}_r^{(L)}\right\|_{\mathrm{F}}$$

for $n \in [\![L]\!]$. This alternating minimization approach is called **alternating least squares (ALS)**. ALS cycles through the indices $n \in [\![L]\!]$ multiple times until some termination criteria is met.

</div>

**Formulating and solving the least squares problem.** Introducing flattened representations of $\mathcal{X}$: for $n \in [\![L]\!]$, the $m_n \times \left(\prod_{j \ne n} m_j\right)$ matrix $\boldsymbol{X}_{(n)}$ is given by horizontally concatenating the mode-$n$ fibers as columns. Next, the flattened tensorization of the factor matrices is

$$\boldsymbol{A}^{\ne n} := \boldsymbol{A}^{(L)} \odot \cdots \odot \boldsymbol{A}^{(n+1)} \odot \boldsymbol{A}^{(n-1)} \odot \cdots \odot \boldsymbol{A}^{(1)} =: \bigodot_{\substack{j=L \\ j \ne n}}^{1} \boldsymbol{A}^{(j)}.$$

In terms of these matrices, the ALS update rule for the $n$-th factor matrix is

$$\boldsymbol{A}^{(n)} = \arg\min_{\boldsymbol{A}} \left\|\boldsymbol{A}^{\ne n} \boldsymbol{A}^* - \boldsymbol{X}_{(n)}^*\right\|_{\mathrm{F}}.$$

The Gram matrix for this least squares problem can be computed efficiently by the formula

$$\boldsymbol{A}^{\ne n *}\boldsymbol{A}^{\ne n} = (\boldsymbol{A}^{(L)*}\boldsymbol{A}^{(L)}) \circledast \cdots \circledast (\boldsymbol{A}^{(n+1)*}\boldsymbol{A}^{(n+1)}) \circledast (\boldsymbol{A}^{(n-1)*}\boldsymbol{A}^{(n-1)}) \circledast \cdots \circledast (\boldsymbol{A}^{(1)*}\boldsymbol{A}^{(1)}).$$

The ALS update rule for the $n$-th factor matrix becomes

$$\boldsymbol{A}^{(n)} = \boldsymbol{X}_{(n)} \boldsymbol{A}^{\ne n} (\boldsymbol{A}^{\ne n *}\boldsymbol{A}^{\ne n})^\dagger.$$

The most expensive part of this update is computing $\boldsymbol{X}_{(n)}\boldsymbol{A}^{\ne n}$, which is the computational bottleneck. This is why row-sampling sketching operators have been successful in ALS algorithms that use sketch-and-solve for the least squares subproblems.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(7.3.1 — Gram Matrix Conditioning)</span></p>

Although it is cheap to form the Gram matrix via the Hadamard product formula, there is potential for *very* bad conditioning even when $L$ is small. We do not know how seriously the poor conditioning affects ALS approaches to CP decomposition in practice.

</div>

### 7.3.2 Sketching for the CP Decomposition

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Kronecker SRFT Sketching for CP-ALS)</span></p>

Battaglino et al. [BBK18] apply the Kronecker SRFT from Section 7.2.2 to the least squares problem. Letting $\boldsymbol{T}_j$ and $\boldsymbol{F}_j$ be of size $m_j \times m_j$, the sketching operator used before solving for the $n$-th factor matrix is

$$\boldsymbol{S}_n = \sqrt{\frac{\prod_{\substack{j=1 \\ j \ne n}}^{L} m_j}{d}}\,\boldsymbol{R}\left(\bigotimes_{\substack{j=L \\ j \ne n}}^{1} \boldsymbol{T}_j\right)\left(\bigotimes_{\substack{j=L \\ j \ne n}}^{1} \boldsymbol{F}_j\right).$$

A key optimization: rather than drawing new $\boldsymbol{R}$ and $(\boldsymbol{F}_j)_{j \ne n}$ every time the operator is applied ($L$ times for every execution of the for loop), Battaglino et al. instead propose drawing $\boldsymbol{F}_1, \ldots, \boldsymbol{F}_L$ once and then reusing them throughout the algorithm, only drawing $\boldsymbol{R}$ anew for each least squares problem. This reduces the computational cost considerably since it allows for greater reuse of various computed quantities — specifically, the expensive application of the full Kronecker SRFT to the unstructured matrix $\boldsymbol{X}_{(n)}^*$ does not have to be computed for every least squares problem.

</div>

Larsen and Kolda [LK20] also sketch the least squares problems. They use the efficient leverage score sampling scheme for Khatri–Rao products discussed in Section 7.2.5. This approach also allows for some reuse between subsequent sketches: since each $\boldsymbol{A}^{(k)}$ only changes for every $L$-th least squares problem, the probability distribution $(p_{i_k}^{(k)})$ can be used in $L - 1$ least squares problems before it needs to be recomputed.

### 7.3.3 Background on the Tucker Decomposition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tucker Decomposition)</span></p>

The **Tucker decomposition** is another popular method that decomposes an $L$-way tensor $\mathcal{X}$ of size $m_1 \times m_2 \times \cdots \times m_L$ into

$$\sum_{r_1=1}^{R_1} \sum_{r_2=1}^{R_2} \cdots \sum_{r_L=1}^{R_L} \mathcal{G}[r_1, r_2, \ldots, r_L]\,\boldsymbol{a}_{r_1}^{(1)} \circ \boldsymbol{a}_{r_2}^{(2)} \circ \cdots \circ \boldsymbol{a}_{r_L}^{(L)},$$

where the so-called **core tensor** $\mathcal{G}$ is of size $R_1 \times R_2 \times \cdots \times R_L$. The $m_n \times R_n$ matrices $\boldsymbol{A}^{(n)} = \begin{bmatrix} \boldsymbol{a}_1^{(n)} & \cdots & \boldsymbol{a}_{R_n}^{(n)} \end{bmatrix}$ for $n \in [\![L]\!]$ are called **factor matrices**.

Similarly to the CP decomposition, the core tensor and factor matrices can be initialized randomly and then updated iteratively via ALS:

$$\text{For } n \text{ in } [\![L]\!]: \quad \boldsymbol{A}^{(n)} = \arg\min_{\boldsymbol{A}} \|\boldsymbol{B}^{\ne n}\boldsymbol{G}_{(n)}^*\boldsymbol{A}^* - \boldsymbol{X}_{(n)}^*\|_{\mathrm{F}},$$

$$\mathcal{G} = \arg\min_{\mathcal{Z}} \|\boldsymbol{B}\operatorname{vec}(\mathcal{Z}) - \operatorname{vec}(\mathcal{X})\|_{\mathrm{F}},$$

where $\boldsymbol{B}^{\ne n} = \boldsymbol{A}^{(L)} \otimes \cdots \otimes \boldsymbol{A}^{(n+1)} \otimes \boldsymbol{A}^{(n-1)} \otimes \cdots \otimes \boldsymbol{A}^{(1)}$ and $\boldsymbol{B} = \boldsymbol{A}^{(L)} \otimes \cdots \otimes \boldsymbol{A}^{(1)}$. The least squares problems are highly overdetermined when $(R_n)_{n \in [\![L]\!]}$ are small compared to $(m_n)_{n \in [\![L]\!]}$.

</div>

### 7.3.4 Sketching for the Tucker Decomposition

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(TensorSketch for Tucker-ALS)</span></p>

Malik and Becker [MB18] apply the TensorSketch discussed in Section 7.2.3 to these problems. From a straightforward adaption to matrix Kronecker products, we have that the TensorSketch of the design matrix $\boldsymbol{B}^{\ne n}$ is computed via the formula

$$\operatorname{DFT}^{-1}\left(\left(\bigodot_{\substack{j=L \\ j \ne n}}^{1} \left(\operatorname{DFT}(\boldsymbol{S}_j \boldsymbol{A}^{(j)})\right)^\top\right)^\top\right),$$

where $\boldsymbol{S}_j$ is a $d \times m_j$ CountSketch, and where $\top$ denotes transpose without complex conjugation.

Instead of drawing new CountSketches for every application of TensorSketch, [MB18, Alg. 2] draw two sets of independent CountSketches at the start of the algorithm: $(\boldsymbol{S}_j^{(1)})_{j=1}^L$ where $\boldsymbol{S}_j^{(1)}$ is of size $d_1 \times m_j$, and $(\boldsymbol{S}_j^{(2)})_{j=1}^L$ where $\boldsymbol{S}_j^{(2)}$ is of size $d_2 \times m_j$. These two sets are then reused throughout the algorithm: $(\boldsymbol{S}_j^{(1)})$ are used for the factor matrix updates and $(\boldsymbol{S}_j^{(2)})$ are used for the core tensor update. Using two sets of sketching operators with $d_2 > d_1$ allows choosing a larger embedding dimension for the latter (larger) problem, achieving considerable improvement in run time.

Moreover, it is possible to compute all relevant sketches of unfoldings of $\mathcal{X}$ at the start of the algorithm, leading to an algorithm that requires only a single pass of $\mathcal{X}$ in order to decompose it.

</div>

### 7.3.5 Implementation Considerations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(RandLAPACK Design for Tensor Sketching)</span></p>

The structured sketches discussed in Section 7.2 are most appropriate to implement in RandLAPACK rather than RandBLAS. In order to facilitate the applications discussed in Section 7.3, it should be possible to update or redraw specific components of the sketching operator after it has been created. For example, when applying the Kronecker SRFT in an ALS algorithm for CP decomposition, we want to keep the random diagonal matrices $\boldsymbol{F}_1, \ldots, \boldsymbol{F}_L$ fixed but draw a new sampling operator $\boldsymbol{R}$ before each application of $\boldsymbol{S}_n$.

In the applications above, components are shared across the $L$ different sketching operators that are used when updating the $L$ different factor matrices. Rather than defining $L$ different sketching operators with shared components, it is better to define a single operator that contains all components and which allows "leaving one component out" when applied to a matrix or vector. A user should be able to supply a parameter $n$ which indicates that the $n$-th term in the Kronecker products should be left out when computing the sketch.

</div>

# Appendix A: Details on Basic Sketching

This appendix covers sketching theory and implementation of sketching operators. Its contents are relevant to Sections 2, 3, and 6.

## A.1 Subspace Embeddings and Effective Distortion

Let $\boldsymbol{S}$ be a wide $d \times m$ sketching operator and $L$ be a linear subspace of $\mathbb{R}^m$. Recall that $\boldsymbol{S}$ *embeds* $L$ into $\mathbb{R}^d$ *with distortion* $\delta \in [0, 1]$ if

$$(1 - \delta)\|\boldsymbol{x}\|_2 \le \|\boldsymbol{S}\boldsymbol{x}\|_2 \le (1 + \delta)\|\boldsymbol{x}\|_2$$

holds for all $\boldsymbol{x}$ in $L$. We often use the term $\delta$-*embedding* for such an operator. Note that if $\boldsymbol{S}$ is a $\delta$-embedding and $\delta' > \delta$, then $\boldsymbol{S}$ is also a $\delta'$-embedding.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Distortion and Effective Distortion)</span></p>

The **distortion** of $\boldsymbol{S}$ for $L$ is the smallest $\delta$ for which $\boldsymbol{S}$ is a $\delta$-embedding for $L$:

$$\mathscr{D}(\boldsymbol{S}; L) = \inf\lbrace \delta : 0 \le \delta \le 1, \;\boldsymbol{S} \text{ is a } \delta\text{-embedding for } L \rbrace.$$

In this notation, we have $\mathscr{D}(\boldsymbol{S}; L) \ge 1$ when $\ker \boldsymbol{S} \cap L$ is nontrivial. If there is a unit vector $\boldsymbol{x}$ in $L$ where $\|\boldsymbol{S}\boldsymbol{x}\| > 2$, then $\mathscr{D}(\boldsymbol{S}; L) = +\infty$.

Subspace embedding distortion has a significant limitation in that it depends on the scale of $\boldsymbol{S}$, while many RandNLA algorithms have no such dependence. This shortcoming leads us to define the **effective distortion** of $\boldsymbol{S}$ for $L$ as

$$\mathscr{D}_e(\boldsymbol{S}; L) = \inf_{t > 0} \mathscr{D}(t\boldsymbol{S}; L).$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Restricted Singular Values and Condition Numbers)</span></p>

**Restricted singular values** are a fairly general concept of use in random matrix theory. They are measures of an operator's "size" when considered from different vantage points within a set of interest. Formally, the largest and smallest restricted singular values of a sketching operator $\boldsymbol{S}$ for a subspace $L$ are

$$\sigma_{\max}(\boldsymbol{S}; L) = \max_{\boldsymbol{x} \in L} \lbrace \|\boldsymbol{S}\boldsymbol{x}\|_2 \;:\; \|\boldsymbol{x}\|_2 = 1 \rbrace$$

and

$$\sigma_{\min}(\boldsymbol{S}; L) = \min_{\boldsymbol{x} \in L} \lbrace \|\boldsymbol{S}\boldsymbol{x}\|_2 \;:\; \|\boldsymbol{x}\|_2 = 1 \rbrace.$$

The **restricted condition number** of $\boldsymbol{S}$ on $L$ is

$$\operatorname{cond}(\boldsymbol{S}; L) = \frac{\sigma_{\max}(\boldsymbol{S}; L)}{\sigma_{\min}(\boldsymbol{S}; L)},$$

where we take $c/0 = +\infty$ for any $c \ge 0$.

More concrete descriptions can be obtained by considering any matrix $\boldsymbol{U}$ whose columns provide an orthonormal basis for $L$. With this one can see that $\sigma_{\min}(\boldsymbol{S}; L)$ and $\sigma_{\max}(\boldsymbol{S}; L)$ coincide with the extreme singular values of $\boldsymbol{S}\boldsymbol{U}$, and that $\operatorname{cond}(\boldsymbol{S}; L) = \operatorname{cond}(\boldsymbol{S}\boldsymbol{U})$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.1.1 — Effective Distortion via Condition Numbers)</span></p>

Let $L$ be a linear subspace and $\boldsymbol{S}$ be a sketching operator on $L$. Let $\kappa = \operatorname{cond}(\boldsymbol{S}; L)$. The effective distortion of $\boldsymbol{S}$ for $L$ is

$$\mathscr{D}_e(\boldsymbol{S}; L) = \frac{\kappa - 1}{\kappa + 1},$$

where we take $(\infty - 1)/(\infty + 1) = 1$.

*Proof sketch.* The scaled sketching operator $t\boldsymbol{S}$ is a $\delta$-embedding for $L$ if and only if $(1-\delta)/t \le \sigma_n$ and $(1+\delta)/t \le \sigma_1$, where $\sigma_1 := \sigma_{\max}(\boldsymbol{S}; L)$ and $\sigma_n := \sigma_{\min}(\boldsymbol{S}; L)$. The optimal $t$ that permits minimum $\delta$ is $t_* = 2/(\sigma_1 + \sigma_n)$. Plugging this in yields $\delta \ge (\sigma_1 - \sigma_n)/(\sigma_1 + \sigma_n) = (\kappa - 1)/(\kappa + 1)$.

</div>

### A.1.1 Effective Distortion of Gaussian Operators

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gaussian Effective Distortion Asymptotics)</span></p>

It is informative to consider the concepts of restricted condition numbers and effective distortion for Gaussian sketching operators. Suppose that our $d \times m$ sketching operator $\boldsymbol{S}$ has iid mean-zero Gaussian entries, and consider an $n$-dimensional subspace $L$ in $\mathbb{R}^m$. By rotational invariance of Gaussian distribution, we can infer that the distribution of $\operatorname{cond}(\boldsymbol{S}; L)$ coincides with that of $\operatorname{cond}(\tilde{\boldsymbol{S}})$ for a $d \times n$ Gaussian matrix $\tilde{\boldsymbol{S}}$.

Specifically, letting $d = sn$ for a constant $s > 1$, results by Silverstein [Sil85] and Geman [Gem80] imply

$$\operatorname{cond}(\boldsymbol{S}; L) \to \frac{\sqrt{s} + 1}{\sqrt{s} - 1} \quad \text{almost surely as} \quad n \to \infty.$$

This can be turned around using Proposition A.1.1 to obtain

$$\mathscr{D}_e(\boldsymbol{S}; L) \to \frac{1}{\sqrt{s}} \quad \text{almost surely as} \quad n \to \infty.$$

These facts hold for any fixed $n$-dimensional subspace $L$. They justify aggressively small choices of embedding dimension when using Gaussian sketching operators in randomized algorithms for least squares.

</div>

## A.2 Short-Axis-Sparse Sketching Operators

In this appendix we make liberal use of the abbreviation *SASO* (for "short-axis-sparse sketching operator") introduced in Section 2.4.1. Without loss of generality, we describe SASOs in the wide case, i.e., when $\boldsymbol{S}$ is $d \times m$ with $d \ll m$.

### A.2.1 Implementation Notes

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Constructing SASOs Column-wise)</span></p>

It is extremely cheap to construct and store a wide SASO. The construction is embarrassingly parallel across columns provided one uses CBRNGs (counter-based random number generators).

**Partitioned construction:** Row indices are partitioned into index sets $I_1, \ldots, I_k$ of roughly equal size. Given such a partition, the indices of nonzeros for a given column are chosen by taking one element (independently and uniformly) from each of the index sets $I_j$. The naive implementation can sample these row indices with $k$ parallel calls to the CBRNG.

**Without-replacement construction:** Row indices for a column are chosen by selecting $k$ elements from $[\![d]\!]$ uniformly without replacement. This can be done in $O(km)$ time by using Fisher-Yates sampling.

Given $T$ threads, the natural modification to the algorithm takes $O(mk/T)$ time and requires $O(Td)$ workspace. The constants in the big-$O$ notation are small.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Storage Formats for SASOs)</span></p>

It is reasonable for a standard library to restrict SASOs to only the most common sparse matrix formats. Both compressed sparse row (CSR) and compressed sparse column (CSC) are worth considering. CSC is the more natural of the two since (wide) SASOs are constructed columnwise. If CSR format is preferred, then we recommend constructing $\boldsymbol{S}$ columnwise while retaining extra data to facilitate conversion to CSR.

</div>

#### Applying a Wide SASO

There are four combinations of storage formats to consider for $(\boldsymbol{S}, \boldsymbol{A})$ when computing the sketch $\boldsymbol{A}_{\text{sk}} = \boldsymbol{S}\boldsymbol{A}$:

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Application Strategies</span><span class="math-callout__name">(SASO $\times$ Data Matrix)</span></p>

**$\boldsymbol{S}$ is CSC, $\boldsymbol{A}$ is row-major.** Given $P$ processors, partition the row index set $[\![d]\!]$ into $I_1, \ldots, I_P$ and have each processor be responsible for its own block of rows. The $p$-th processor computes its row block by streaming over the columns of $\boldsymbol{S}$ and rows of $\boldsymbol{A}$, accumulating outer products:

$$\boldsymbol{A}_{\text{sk}}[I_p, :] = \sum_{\ell \in [\![m]\!]} \boldsymbol{S}[I_p, \ell]\boldsymbol{A}[\ell, :].$$

**$\boldsymbol{S}$ is CSR, $\boldsymbol{A}$ is row-major.** The $d$ rows of $\boldsymbol{A}_{\text{sk}}$ are computed separately from one another. An individual row is given by $\boldsymbol{A}_{\text{sk}}[i,:] = \boldsymbol{S}[i,:]\boldsymbol{A}$. If $R$ is the number of nonzeros in $\boldsymbol{S}[i,:]$ then computing $\boldsymbol{A}_{\text{sk}}[i,:]$ only requires $R$ rows from $\boldsymbol{A}$. Since the columns of $\boldsymbol{S}$ are independent, $R$ is a sum of $m$ iid Bernoulli random variables with mean $k/d$. We expect to access $mk/d$ rows of $\boldsymbol{A}$ per row of the sketch.

**$\boldsymbol{S}$ is CSC, $\boldsymbol{A}$ is column-major.** The $n$ columns of $\boldsymbol{A}_{\text{sk}}$ are computed separately from one another. An individual column is given by $\boldsymbol{A}_{\text{sk}}[:,j] = \sum_{\ell \in [\![m]\!]} \boldsymbol{S}[:,\ell]\boldsymbol{A}[\ell, j]$. Based on preliminary experiments, this method has mediocre single-thread performance, but has excellent scaling properties for many-core machines.

**$\boldsymbol{S}$ is CSR, $\boldsymbol{A}$ is column-major.** The most efficient algorithm may be to convert $\boldsymbol{S}$ to CSC and then apply the preferred method when $\boldsymbol{S}$ is CSC and $\boldsymbol{A}$ is column-major.

</div>

### A.2.2 Theory and Practical Usage

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(SASO Theory)</span></p>

A precursor to the SASOs we consider is described in [DKS10], which sampled row indices for nonzero entries from $[\![d]\!]$ with replacement. The first theoretical analysis of the SASOs we consider was conducted in [KN12] and concerned the distributional Johnson-Lindenstrauss property. Shortly thereafter, [CW13] and [MM13] studied OSE properties for SASOs with a single nonzero per column; the latter referred to the construction as "CountSketch."

Theoretical analyses for OSE properties of general SASOs (i.e., those with more than one nonzero per column) were first carried out by [NN13; KN14] and subsequently improved by [BDN15; Coh16]. Much of the SASO analysis has been through the lens of "hashing functions," and does not require that the columns of the sketching operator are fully independent. We do not know of any practical advantage to SASOs with partial dependence across the columns.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A.2.1 — Navigating the Literature)</span></p>

[CW17] is a longer journal version of [CW13]. [KN14] and [KN12] have the same title, and the former is considered a more developed journal version of the latter.

</div>

#### Selecting Parameters for SASOs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(SASO Parameter Selection)</span></p>

When $d$ is fixed the sketch quality increases rapidly with $k$ before reaching a plateau. As one point of reference, there is no real benefit in $k$ being larger than eight when embedding the range of a $100{,}000 \times 2{,}000$ matrix into a space with ambient dimension $d = 6{,}000$. Furthermore, for the data matrices tested, the restricted condition numbers of those sketching operators were tightly concentrated at $O(1)$. Extensive experiments with parameter selection for SASOs in a least squares context are given in [Ura13].

</div>

## A.3 Theory for Sketching by Row Selection

Here we prove Proposition 6.1.1. The proof is inspired by [Tro20, Problem 5.13], which begins with the following adaptation of [Tro15, Theorem 5.1.1].

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(A.3.1 — Matrix Chernoff Bound)</span></p>

Consider an independent family $\lbrace \boldsymbol{X}_1, \ldots, \boldsymbol{X}_s \rbrace \subset \mathbb{H}^n$ of random psd matrices that satisfy $\lambda_{\max}(\boldsymbol{X}_i) \le L$ almost surely. Let $\boldsymbol{Y} = \sum_{i=1}^{s} \boldsymbol{X}_i$, and define the mean parameters

$$\mu_{\max} = \lambda_{\max}(\mathbb{E}\boldsymbol{Y}) \quad \text{and} \quad \mu_{\min} = \lambda_{\min}(\mathbb{E}\boldsymbol{Y}).$$

One has that

$$\Pr\lbrace \lambda_{\max}(\boldsymbol{Y} - (1+t)\mathbb{E}\boldsymbol{Y}) \ge 0 \rbrace \le n \left[\frac{\exp(t)}{(1+t)^{(1+t)}}\right]^{\mu_{\max}/L} \quad \text{for } t > 0,$$

$$\Pr\lbrace \lambda_{\max}((1-t)\mathbb{E}\boldsymbol{Y} - \boldsymbol{Y}) \ge 0 \rbrace \le n \left[\frac{\exp(-t)}{(1-t)^{(1-t)}}\right]^{\mu_{\min}/L} \quad \text{for } t \in (0, 1).$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.3.2 — Adaptation of Proposition 6.1.1)</span></p>

Suppose $\boldsymbol{A}$ is an $m \times n$ matrix of rank $n$, $\boldsymbol{q}$ is a distribution over $[\![m]\!]$, and $t$ is in $(0, 1)$. Let $\boldsymbol{S}$ be a $d \times m$ sketching operator with rows that are distributed iid as

$$\boldsymbol{S}[i,:] = \frac{\boldsymbol{\delta}_j}{\sqrt{d q_j}} \quad \text{with probability } q_j,$$

and let $E(t, \boldsymbol{S})$ denote the event that

$$(1-t)\|\boldsymbol{y}\|_2^2 \le \|\boldsymbol{S}\boldsymbol{y}\|_2^2 \le (1+t)\|\boldsymbol{y}\|_2^2 \quad \forall\; \boldsymbol{y} \in \operatorname{range} \boldsymbol{A}.$$

Using $r := \min_{j \in [\![m]\!]} \frac{q_j}{p_j(\boldsymbol{A})}$, we have

$$\Pr\lbrace E(t, \boldsymbol{S}) \text{ fails} \rbrace \le 2n\left(\frac{\exp(t)}{(1+t)^{(1+t)}}\right)^{rd/n}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition A.3.2</summary>

We consider the Gram matrices $\boldsymbol{G} = \boldsymbol{A}^*\boldsymbol{A}$ and $\boldsymbol{G}_{\text{sk}} = \boldsymbol{A}^*\boldsymbol{S}^*\boldsymbol{S}\boldsymbol{A}$. The event $E(t, \boldsymbol{S})$ is equivalent to

$$(1 - t)\boldsymbol{I}_n \preceq \boldsymbol{G}^{-1/2}\boldsymbol{G}_{\text{sk}}\boldsymbol{G}^{-1/2} \preceq (1 + t)\boldsymbol{I}_n.$$

The sketched Gram matrix can be expressed as a sum of $d$ outer products of rows of $\boldsymbol{S}\boldsymbol{A}$. Each of the $d$ outer products is conjugated by $\boldsymbol{G}^{-1/2}$ to obtain our matrices $\lbrace \boldsymbol{X}_1, \ldots, \boldsymbol{X}_d \rbrace$. That is, we set

$$\boldsymbol{X}_i = \boldsymbol{G}^{-1/2}\left((\boldsymbol{S}\boldsymbol{A})[i,:]\right)^*\left((\boldsymbol{S}\boldsymbol{A})[i,:]\right)\boldsymbol{G}^{-1/2},$$

so that $\boldsymbol{Y} = \sum_{i=1}^{d} \boldsymbol{X}_i$ satisfies $\mathbb{E}\boldsymbol{Y} = \boldsymbol{I}_n$. A union bound provides

$$\Pr\lbrace E(t, \boldsymbol{S}) \text{ fails} \rbrace \le \Pr\lbrace \lambda_{\max}(\boldsymbol{Y}) \ge 1 + t \rbrace + \Pr\lbrace 1 - t \ge \lambda_{\min}(\boldsymbol{Y}) \rbrace.$$

Our particular choice of $\boldsymbol{Y}$ leads to $\mu_{\min} = \mu_{\max} = 1$. The larger of the two probability bounds in Theorem A.3.1 is the one involving $\exp(t)/(1+t)^{(1+t)}$. Therefore we have

$$\Pr\lbrace E(t, \boldsymbol{S}) \text{ fails} \rbrace \le 2n\left(\exp(t)/(1+t)^{(1+t)}\right)^{1/L}.$$

The smallest value that guarantees $\lambda_{\max}(\boldsymbol{X}_i) \le L$ is

$$L = \frac{1}{d} \max_{j \in [\![m]\!]} \left\lbrace \frac{\ell_j(\boldsymbol{A})}{q_j} \right\rbrace = \frac{n}{d} \max_{j \in [\![m]\!]} \frac{p_j(\boldsymbol{A})}{q_j},$$

and the proposition's claim follows from substituting $r = \min_{j \in [\![m]\!]} q_j / p_j(\boldsymbol{A})$.

</details>
</div>

# Appendix B: Details on Least Squares and Optimization

This appendix covers a few distinct topics. Appendix B.1 proves a novel result relevant to sketch-and-precondition algorithms for saddle point problems, and connects this result to the idea of effective distortion. Appendix B.2 provides background from classical NLA on what it means to compute an "accurate" solution to a least squares problem (overdetermined or underdetermined). Appendix B.3 derives limiting solutions of saddle point problems as the regularization parameter tends to zero from above. Appendix B.4 gives background on kernel ridge regression and details a new approach to sketch-and-solve of regularized quadratics.

## B.1 Quality of Preconditioners

Here we consider preconditioners of the kind described in Section 3.3.2. These are obtained by sketching a tall $m \times n$ data matrix $\boldsymbol{A}$ in the embedding regime, factoring the sketch, and using the factorization to construct an orthogonalizer.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(B.1.1 — Preconditioner Singular Values)</span></p>

Consider a sketch $\boldsymbol{A}_{\text{sk}} = \boldsymbol{S}\boldsymbol{A}$ and a matrix $\boldsymbol{U}$ whose columns are an orthonormal basis for $\operatorname{range}(\boldsymbol{A})$. If $\operatorname{rank}(\boldsymbol{A}_{\text{sk}}) = \operatorname{rank}(\boldsymbol{A})$ and the columns of $\boldsymbol{A}_{\text{sk}}\boldsymbol{M}$ are an orthonormal basis for the range of $\boldsymbol{A}_{\text{sk}}$, then singular values of $\boldsymbol{A}\boldsymbol{M}$ are the reciprocals of the singular values of $\boldsymbol{S}\boldsymbol{U}$.

This proposition is a linear algebraic result — there is no randomness. When applied to randomized algorithms, the randomness enters only via the construction of the sketch.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(B.1.2 — Pseudoinverse Identity)</span></p>

Suppose $\boldsymbol{A}_{\text{sk}}$ is a tall $d \times n$ matrix and that $\boldsymbol{M}$ is a full-column-rank matrix for which the columns of $\boldsymbol{A}_{\text{sk}}\boldsymbol{M}$ form an orthonormal basis for $\operatorname{range}(\boldsymbol{A}_{\text{sk}})$. If $\boldsymbol{B}$ is a full-row-rank matrix for which $\boldsymbol{A}_{\text{sk}} = \boldsymbol{A}_{\text{sk}}\boldsymbol{M}\boldsymbol{B}$, then we have $\boldsymbol{M} = \boldsymbol{B}^\dagger$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition B.1.1</summary>

Let $k = \operatorname{rank}(\boldsymbol{A})$. It suffices to prove the statement where $\boldsymbol{U}$ is the $m \times k$ matrix containing the left singular vectors of $\boldsymbol{A}$. Working with the compact SVD $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^*$, where $\boldsymbol{V}$ is $n \times k$ and $\boldsymbol{\Sigma}$ is invertible, we can replace $\boldsymbol{S}\boldsymbol{U}$ by its economic QR factorization $\boldsymbol{S}\boldsymbol{U} = \boldsymbol{Q}\boldsymbol{R}$ to see

$$\boldsymbol{A}_{\text{sk}} = \boldsymbol{Q}\boldsymbol{R}\boldsymbol{\Sigma}\boldsymbol{V}^*.$$

Since $\operatorname{rank}(\boldsymbol{A}_{\text{sk}}) = k$ it must be that $\operatorname{rank}(\boldsymbol{S}\boldsymbol{U}) = k$, so $\boldsymbol{R}$ is invertible and $\boldsymbol{Q}$ provides an orthonormal basis for the range of $\boldsymbol{A}_{\text{sk}}$.

By assumption on $\boldsymbol{M}$, the matrix $\boldsymbol{A}_{\text{sk}}\boldsymbol{M}$ is *also* an orthonormal basis for the range of $\boldsymbol{A}_{\text{sk}}$. Therefore there exists a $k \times k$ orthogonal matrix $\boldsymbol{P}$ where $\boldsymbol{Q}\boldsymbol{P} = \boldsymbol{A}_{\text{sk}}\boldsymbol{M}$. We can rewrite as

$$\boldsymbol{A}_{\text{sk}} = \boldsymbol{A}_{\text{sk}}\boldsymbol{M}\left(\boldsymbol{P}^*\boldsymbol{R}\boldsymbol{\Sigma}\boldsymbol{V}^*\right).$$

Since the left factor is $\boldsymbol{A}_{\text{sk}}\boldsymbol{M}$, we have $\boldsymbol{A}_{\text{sk}} = \boldsymbol{A}_{\text{sk}}\boldsymbol{M}\left(\boldsymbol{P}^*\boldsymbol{R}\boldsymbol{\Sigma}\boldsymbol{V}^*\right)$. Abbreviate $\boldsymbol{B} = \boldsymbol{P}^*\boldsymbol{R}\boldsymbol{\Sigma}\boldsymbol{V}^*$ and apply Lemma B.1.2 to infer that $\boldsymbol{B} = \boldsymbol{M}^\dagger$. Invoking the column-orthonormality of $\boldsymbol{V}$ and invertibility of $(\boldsymbol{\Sigma}, \boldsymbol{R}, \boldsymbol{P})$ we further have $\boldsymbol{B}^\dagger = \boldsymbol{M} = \boldsymbol{V}\boldsymbol{\Sigma}^{-1}\boldsymbol{R}^{-1}\boldsymbol{P}$. Plug in this expression for $\boldsymbol{M}$ to see that

$$\boldsymbol{A}\boldsymbol{M} = (\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^*)(\boldsymbol{V}\boldsymbol{\Sigma}^{-1}\boldsymbol{R}^{-1}\boldsymbol{P}) = \boldsymbol{U}\boldsymbol{R}^{-1}\boldsymbol{P}.$$

The proof is completed by noting that the singular values of $\boldsymbol{R}^{-1}$ are the reciprocals of the singular values of $\boldsymbol{Q}\boldsymbol{R} = \boldsymbol{S}\boldsymbol{U}$.

</details>
</div>

### B.1.1 Effective Distortion in Sketch-and-Precondition

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to Effective Distortion)</span></p>

Recall from Appendix A.1 that if the columns of $\boldsymbol{U}$ are an orthonormal basis for a linear subspace $L$, then the restricted condition number of $\boldsymbol{S}$ on $L$ is $\operatorname{cond}(\boldsymbol{S}; L) = \operatorname{cond}(\boldsymbol{S}\boldsymbol{U})$.

This identity combines with Proposition B.1.1 to make for a remarkable fact. Namely, if $L = \operatorname{range}(\boldsymbol{A})$ and $\boldsymbol{M}$ is an orthogonalizer of a sketch $\boldsymbol{S}\boldsymbol{A}$, then

$$\operatorname{cond}(\boldsymbol{S}; L) = \operatorname{cond}(\boldsymbol{A}\boldsymbol{M}).$$

If $\boldsymbol{A}$ is an $m \times n$ matrix ($m \gg n$) in a saddle point problem, and if that problem is approached by the sketch-and-precondition methodology from Section 3.2.2, then the condition number of the preconditioned matrix handed to the iterative solver is equal to the restricted condition number of $\boldsymbol{S}$ on $\operatorname{range}(\boldsymbol{A})$.

By invoking Proposition A.1.1 we obtain the following expression for the effective distortion of $\boldsymbol{S}$ for $L$:

$$\mathscr{D}_e(\boldsymbol{S}; L) = \frac{\operatorname{cond}(\boldsymbol{A}\boldsymbol{M}) - 1}{\operatorname{cond}(\boldsymbol{A}\boldsymbol{M}) + 1}.$$

The right-hand-side is none other than the convergence rate of LSQR (or CGLS) for a least squares problem with data matrix $\boldsymbol{A}\boldsymbol{M}$. This shows a deep connection between effective distortion and the venerated sketch-and-precondition paradigm in RandNLA.

</div>

## B.2 Basic Error Analysis for Least Squares Problems

When solving a computational problem numerically it is inevitable that the computed solutions deviate from the problem's exact solution. This is a simple consequence of working in finite-precision arithmetic. Furthermore, for large-scale computations it is often of interest to trade off computational complexity with solution accuracy; this has led to algorithms that produce approximate solutions even when run in exact arithmetic.

### B.2.1 Concepts: Forward and Backward Error

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Forward and Backward Error)</span></p>

The **forward error** of an approximate solution to a computational problem is its distance to the problem's exact solution. Forward error is easy to interpret, but it is not without its limitations: it can rarely be computed in practice (since we do not have access to the exact solution), and an analysis for one algorithm may not be repurposable for another.

These shortcomings motivate the ideas of **backward error** and **sensitivity analysis**, wherein one asks the following questions, respectively:

- How much do we need to perturb the problem data so that the computed solution exactly solves the perturbed problem?
- How does a small perturbation to a given problem change that problem's exact solution?

The connection between the two concepts is clear: any bound on backward error can be combined with sensitivity analysis to obtain an estimate of forward error. The idea of sensitivity analysis is especially powerful since it is agnostic to the source of the problem's perturbation.

</div>

The combination of backward error and sensitivity analysis can therefore be used to establish *a priori* guarantees on algorithm numerical behavior and *a posteriori* guarantees on the quality of an approximate solution.

### B.2.2 Basic Sensitivity Analysis for Least Squares Problems

The analysis results here involve a tall $m \times n$ matrix $\boldsymbol{A}$. The overdetermined problem induced by $\boldsymbol{A}$ and an $m$-vector $\boldsymbol{b}$ is

$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2,$$

while the underdetermined problem induced by $\boldsymbol{A}$ and an $n$-vector $\boldsymbol{c}$ is

$$\min_{\boldsymbol{y} \in \mathbb{R}^m} \left\lbrace \|\boldsymbol{y}\|_2^2 \;:\; \boldsymbol{A}^*\boldsymbol{y} = \boldsymbol{c} \right\rbrace.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(B.2.2 — Sensitivity of Overdetermined Least Squares)</span></p>

Suppose $\boldsymbol{b}$ is neither in the range of $\boldsymbol{A}$ nor the kernel of $\boldsymbol{A}^*$, and let $\boldsymbol{x} = \boldsymbol{A}^\dagger \boldsymbol{b}$ be the optimal solution of the overdetermined least squares problem with data $(\boldsymbol{A}, \boldsymbol{b})$. Consider perturbations $\delta\boldsymbol{b}$ and $\delta\boldsymbol{A}$ where $\|\delta\boldsymbol{A}\|_2 < \sigma_n(\boldsymbol{A})$. Define

$$\epsilon = \max\left\lbrace \frac{\|\delta\boldsymbol{A}\|_2}{\|\boldsymbol{A}\|_2},\; \frac{\|\delta\boldsymbol{b}\|_2}{\|\boldsymbol{b}\|_2} \right\rbrace$$

together with some auxiliary quantities

$$\sin\theta = \frac{\|\boldsymbol{b} - \boldsymbol{A}\boldsymbol{x}\|_2}{\|\boldsymbol{b}\|_2} \quad \text{and} \quad \nu = \frac{\|\boldsymbol{A}\boldsymbol{x}\|_2}{\sigma_n(\boldsymbol{A})\|\boldsymbol{x}\|_2}.$$

The perturbation $\delta\boldsymbol{x}$ necessary for $\boldsymbol{x} + \delta\boldsymbol{x}$ to solve the least squares problem with data $(\boldsymbol{A} + \delta\boldsymbol{A}, \boldsymbol{b} + \delta\boldsymbol{b})$ satisfies

$$\frac{\|\delta\boldsymbol{x}\|_2}{\|\boldsymbol{x}\|_2} \le \epsilon\left\lbrace \frac{\nu}{\cos\theta} + \kappa(\boldsymbol{A})(1 + \nu\tan\theta) \right\rbrace + O(\epsilon^2).$$

Under the modest geometric assumption that $\theta$ is bounded away from $\pi/2$, since $\nu \le \kappa(\boldsymbol{A})$ holds for all nonzero $\boldsymbol{x}$, this simplifies to

$$\frac{\|\delta\boldsymbol{x}\|_2}{\|\boldsymbol{x}\|_2} \lesssim \epsilon\left\lbrace \kappa(\boldsymbol{A}) + \frac{\|\boldsymbol{b} - \boldsymbol{A}\boldsymbol{x}\|_2}{\|\boldsymbol{b}\|_2}\,\kappa(\boldsymbol{A})^2 \right\rbrace.$$

The significance of this bound is that it shows the dependence of $\|\delta\boldsymbol{x}\|_2$ on the *square* of the condition number of $\boldsymbol{A}$. This dependence is a fundamental obstacle to solving least squares problems to a high degree of accuracy when measured by forward error.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(B.2.3 — Sensitivity of the Residual)</span></p>

Under the hypothesis and notation of Theorem B.2.2, we have

$$\frac{\|\boldsymbol{A}(\delta\boldsymbol{x})\|_2}{\|\boldsymbol{b} - \boldsymbol{A}\boldsymbol{x}\|_2} \le \epsilon\left\lbrace \frac{1}{\sin\theta} + \kappa(\boldsymbol{A})\left(\frac{1}{\nu\tan\theta} + 1\right) \right\rbrace + O(\epsilon^2).$$

This can be seen as a sensitivity analysis result for dual saddle point problems. Under mild geometric assumptions, if $\theta$ is sufficiently bounded away from 0 and $\pi/2$:

$$\frac{\|\delta\boldsymbol{y}\|_2}{\|\boldsymbol{y}\|_2} \lesssim \epsilon\,\kappa(\boldsymbol{A}).$$

This shows there is more hope for solving dual saddle point problems to a high degree of forward error accuracy, at least by comparison to primal saddle point problems.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(B.2.4 — Sensitivity of Underdetermined Least Squares)</span></p>

Let $\boldsymbol{y} = (\boldsymbol{A}^*)^\dagger \boldsymbol{c}$ solve the underdetermined least squares problem with data $(\boldsymbol{A}, \boldsymbol{c})$ for a nonzero vector $\boldsymbol{c}$. Consider perturbations $\delta\boldsymbol{c}$ and $\delta\boldsymbol{A}$ where

$$\epsilon = \max\left\lbrace \frac{\|\delta\boldsymbol{c}\|_2}{\|\boldsymbol{c}\|_2},\; \frac{\|\delta\boldsymbol{A}\|_2}{\|\boldsymbol{A}\|_2} \right\rbrace < \sigma_n(\boldsymbol{A}).$$

The perturbation $\delta\boldsymbol{y}$ needed for $\boldsymbol{y} + \delta\boldsymbol{y}$ to solve the underdetermined least squares problem with data $(\boldsymbol{A} + \delta\boldsymbol{A}, \boldsymbol{c} + \delta\boldsymbol{c})$ satisfies

$$\frac{\|\delta\boldsymbol{y}\|_2}{\|\boldsymbol{y}\|_2} \le 3\,\epsilon\,\kappa(\boldsymbol{A}) + O(\epsilon^2).$$

</div>

### B.2.3 Sharper Sensitivity Analysis for Overdetermined Problems

The analysis results in Appendix B.2.2 have notable limitations: they hide constants in $O(\epsilon^2)$ terms. Luckily there are more precise results in the literature that work with different notions of error.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(B.2.5 — Relative Error Bound for Perturbed Problem)</span></p>

Consider an overdetermined least squares problem with data $(\boldsymbol{A}_o, \boldsymbol{b}_o)$ that is solved by $\boldsymbol{x}_o = (\boldsymbol{A}_o)^\dagger(\boldsymbol{b}_o)$; consider also perturbed problem data $\boldsymbol{A} = \boldsymbol{A}_o + \delta\boldsymbol{A}_o$ and $\boldsymbol{b} = \boldsymbol{b}_o + \delta\boldsymbol{b}_o$ together with solution $\boldsymbol{x} = \boldsymbol{A}^\dagger \boldsymbol{b}$. If we have $\operatorname{rank}(\boldsymbol{A}_o) = \operatorname{rank}(\boldsymbol{A}) = n$ and define

$$\epsilon_A = \frac{\|\delta\boldsymbol{A}_o\|_2}{\|\boldsymbol{A}_o\|_2} \quad \text{and} \quad \epsilon_b = \frac{\|\delta\boldsymbol{b}_o\|}{\|\boldsymbol{A}_o\|_2 \|\boldsymbol{x}\|_2},$$

then we have

$$\frac{\|\boldsymbol{x}_o - \boldsymbol{x}\|_2}{\|\boldsymbol{x}\|_2} \le (\epsilon_A + \epsilon_b)\kappa(\boldsymbol{A}_o) + \epsilon_A \frac{[\kappa(\boldsymbol{A}_o)]^2\,\|\boldsymbol{y}\|_2}{\|\boldsymbol{A}_o\|_2\|\boldsymbol{x}\|_2}$$

for $\boldsymbol{y} = \boldsymbol{b} - \boldsymbol{A}\boldsymbol{x}$.

This bound obtains a relative error bound normalized by the solution of the *perturbed problem* rather than the original problem. It is useful for understanding how solutions exhibit different sensitivity to perturbations to the data matrix compared to perturbations to the right-hand-side. Even better bounds can be obtained by assuming *structured perturbations* — e.g., if $\operatorname{range}(\boldsymbol{A}_o) = \operatorname{range}(\boldsymbol{A})$, then the sensitivity depends only linearly on $\kappa(\boldsymbol{A}_o)$.

</div>

### B.2.4 Simple Constructions to Bound Backward Error

Here we describe two methods for constructing perturbations to problem data that render an approximate solution exact. By computing the norms of these perturbations, we can obtain upper bounds on (normwise) backward error. Such bounds are useful as termination criteria for iterative solvers.

The notation matches that of Theorem B.2.5: our original least squares problem has data $(\boldsymbol{A}_o, \boldsymbol{b}_o)$ and $\boldsymbol{x}$ is an *approximate* solution.

#### Inconsistent Overdetermined Problems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Backward Error Construction — Inconsistent Case)</span></p>

Letting $\boldsymbol{r} = \boldsymbol{b}_o - \boldsymbol{A}_o\boldsymbol{x}$, we define

$$\delta\boldsymbol{A}_o = \frac{\boldsymbol{r}\boldsymbol{r}^*\boldsymbol{A}_o}{\|\boldsymbol{r}\|_2^2} \quad \text{and} \quad \boldsymbol{A} = \boldsymbol{A}_o + \delta\boldsymbol{A}_o,$$

and subsequently

$$\delta\boldsymbol{b}_o = -(\delta\boldsymbol{A}_o)\boldsymbol{x} \quad \text{and} \quad \boldsymbol{b} = \boldsymbol{b}_o + \delta\boldsymbol{b}_o.$$

Then $\boldsymbol{x}$ satisfies the normal equations $\boldsymbol{A}^*(\boldsymbol{b} - \boldsymbol{A}\boldsymbol{x}) = \boldsymbol{0}$, and therefore it solves the overdetermined least squares problem with data $(\boldsymbol{A}, \boldsymbol{b})$.

This construction was first given in [Ste77, Theorem 3.2]. It is especially nice since the perturbation is rank-1, and so its spectral norm

$$\|\delta\boldsymbol{A}_o\|_2 = \frac{\|\boldsymbol{A}_o^*\boldsymbol{r}\|_2}{\|\boldsymbol{r}\|_2}$$

is easily computed at runtime by an iterative solver for overdetermined least squares. Furthermore, if the iterative solver is LSQR and we assume exact arithmetic, the perturbation will satisfy $\delta\boldsymbol{A}_o\boldsymbol{x} = \boldsymbol{0}$, so $\delta\boldsymbol{b}_o$ is zero when running LSQR in exact arithmetic.

</div>

#### Consistent Overdetermined Problems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Backward Error Construction — Consistent Case)</span></p>

The perturbations above are not suitable for least squares problems where the optimal residual $(\boldsymbol{I} - \boldsymbol{A}_o\boldsymbol{A}_o^\dagger)\boldsymbol{b}_o$ is zero or nearly zero. In these situations one should use a perturbation designed for consistent linear systems.

Let $\delta\boldsymbol{b}_o$ be an arbitrary $m$-vector. We set $\delta\boldsymbol{A}_o$ as a function of $\delta\boldsymbol{b}_o$ in the following way:

$$\delta\boldsymbol{A}_o = \frac{(\delta\boldsymbol{b}_o + \boldsymbol{b}_o - \boldsymbol{A}_o\boldsymbol{x})\,\boldsymbol{x}^*}{\|\boldsymbol{x}\|_2^2}.$$

It can be seen that $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ upon taking $\boldsymbol{b} = \boldsymbol{b}_o + \delta\boldsymbol{b}_o$ and $\boldsymbol{A} = \boldsymbol{A}_o + \delta\boldsymbol{A}_o$, so $\boldsymbol{x}$ trivially solves the perturbed least squares problem.

The construction for LSQR considers two tolerance parameters $\epsilon_A, \epsilon_b \in [0, 1)$, and sets $\delta\boldsymbol{b}_o$ as follows:

$$\delta\boldsymbol{b}_o = \left(\frac{\epsilon_b\|\boldsymbol{b}_o\|_2}{\epsilon_b\|\boldsymbol{b}_o\|_2 + \epsilon_A\|\boldsymbol{A}_o\|\|\boldsymbol{x}\|_2}\right)(\boldsymbol{A}_o\boldsymbol{x} - \boldsymbol{b}_o).$$

The parameters $\epsilon_A$ and $\epsilon_b$ indicate the (relative) sizes of perturbations to $(\boldsymbol{A}_o, \boldsymbol{b}_o)$ that a user deems allowable. We recommend setting $\epsilon_A = 0$ if one wants to think only in terms of a single tolerance for consistent overdetermined problems — this ensures that $\delta\boldsymbol{A}_o = \boldsymbol{0}$.

</div>

### B.2.5 More Advanced Concepts

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Componentwise Backward Error)</span></p>

Some of the earliest work on backward-error analysis for solutions to linear systems focused on componentwise backward error for direct methods [OP64]. A principal shortcoming of componentwise error metrics is that they are expensive to compute, especially as stopping criteria for iterative solvers. [ADR92] investigates metrics for componentwise backward error suitable for iterative solvers.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forward Error Estimation at Runtime)</span></p>

The "backward error plus sensitivity analysis" approach may overestimate forward error. Alternative estimates are available for some Krylov subspace methods such as PCG, wherein one uses algorithm-specific recurrences to estimate forward error in the Euclidean norm or the norm induced by $\boldsymbol{A}_\mu := [\boldsymbol{A}; \sqrt{\mu}]$. These error bounds are more accurate when used with a good preconditioner, which we can generally expect to have when using the randomized algorithms described herein.

It is not easy to apply sensitivity analysis results to compute forward error bounds at runtime. A primary obstacle is the need to have accurate estimates for the extreme singular values of $\boldsymbol{A}_o$ or the perturbation $\boldsymbol{A}$ (depending on the sensitivity analysis result in question). If $\boldsymbol{M}$ is an SVD-based preconditioner then the singular values and right singular vectors of a sketch $\boldsymbol{S}\boldsymbol{A}_o$ can be used as approximations to the reciprocals of the singular values of $\boldsymbol{A}_o$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(B.2.7 — Norm Ambiguity)</span></p>

As a minor detail, we point out that the norm of $\boldsymbol{A}_o$ in the backward error construction is deliberately ambiguous. While the spectral norm would probably be most natural, the formal LSQR algorithm replaces $\|\boldsymbol{A}_o\|$ by an *estimate* of $\|\boldsymbol{A}_o\|_{\mathrm{F}}$ that monotonically increases from one iteration to the next; see [PS82, §5.3].

</div>

## B.3 Ill-Posed Saddle Point Problems

Our saddle point formulations of least squares problems can be problematic when $\boldsymbol{A}$ is rank-deficient and $\mu$ is zero, in which case our problems can actually be infeasible or unbounded below. This appendix uses a limiting analysis to define *canonical solutions* to saddle point problems in these settings.

We begin by recalling

$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2 + \mu\|\boldsymbol{x}\|_2^2 + 2\boldsymbol{c}^*\boldsymbol{x}, \qquad (P_\mu)$$

$$\min_{\boldsymbol{y} \in \mathbb{R}^m} \|\boldsymbol{A}^*\boldsymbol{y} - \boldsymbol{c}\|_2^2 + \mu\|\boldsymbol{y} - \boldsymbol{b}\|_2^2, \qquad (D_\mu, \;\mu > 0)$$

$$\min_{\boldsymbol{y} \in \mathbb{R}^m} \left\lbrace \|\boldsymbol{y} - \boldsymbol{b}\|_2^2 \;:\; \boldsymbol{A}^*\boldsymbol{y} = \boldsymbol{c} \right\rbrace. \qquad (D_0)$$

We also note the following form of solutions to the dual, when $\mu$ is positive:

$$\boldsymbol{y}(\mu) = (\boldsymbol{A}\boldsymbol{A}^* + \mu\boldsymbol{I})^{-1}(\boldsymbol{A}\boldsymbol{c} + \mu\boldsymbol{b}).$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(B.3.1 — Limiting Dual Solution)</span></p>

For any tall $m \times n$ matrix $\boldsymbol{A}$, any $m$-vector $\boldsymbol{b}$, and any $n$-vector $\boldsymbol{c}$, we have

$$\lim_{\mu \downarrow 0} \boldsymbol{y}(\mu) = (\boldsymbol{A}^*)^\dagger\boldsymbol{c} + (\boldsymbol{I} - \boldsymbol{A}\boldsymbol{A}^\dagger)\boldsymbol{b}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition B.3.1</summary>

Let $k = \operatorname{rank}(\boldsymbol{A})$. If $k = 0$ then the claim is trivial since $\boldsymbol{y}(\mu) = \boldsymbol{b}$ for all $\mu > 0$. Assume $k \ge 1$. Consider how the compact SVD $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^*$ lets us express

$$\boldsymbol{A}\boldsymbol{A}^* + \mu\boldsymbol{I}_m = \boldsymbol{H}_\mu + \boldsymbol{G}_\mu$$

in terms of the Hermitian matrices $\boldsymbol{H}_\mu = \boldsymbol{U}(\boldsymbol{\Sigma}^2 + \mu\boldsymbol{I}_k)\boldsymbol{U}^*$ and $\boldsymbol{G}_\mu = \mu(\boldsymbol{I}_m - \boldsymbol{U}\boldsymbol{U}^*)$. Since $\boldsymbol{H}_\mu\boldsymbol{G}_\mu = \boldsymbol{G}_\mu\boldsymbol{H}_\mu = \boldsymbol{0}$, we have $(\boldsymbol{A}\boldsymbol{A}^* + \mu\boldsymbol{I})^{-1} = \boldsymbol{H}_\mu^\dagger + \boldsymbol{G}_\mu^\dagger$. Furthermore:

$$\boldsymbol{H}_\mu^\dagger\boldsymbol{A}\boldsymbol{c} = \boldsymbol{U}(\boldsymbol{\Sigma}^2 + \mu\boldsymbol{I}_k)^{-1}\boldsymbol{\Sigma}\boldsymbol{V}^*\boldsymbol{c}, \quad \boldsymbol{G}_\mu^\dagger\boldsymbol{A}\boldsymbol{c} = 0,$$

$$\mu\boldsymbol{H}_\mu^\dagger\boldsymbol{b} = \boldsymbol{U}(\boldsymbol{\Sigma}^2/\mu + \boldsymbol{I}_k)^{-1}\boldsymbol{U}^*\boldsymbol{b}, \quad \mu\boldsymbol{G}_\mu^\dagger\boldsymbol{b} = (\boldsymbol{I}_m - \boldsymbol{U}\boldsymbol{U}^*)\boldsymbol{b}.$$

We find that

$$\boldsymbol{y}(\mu) = \boldsymbol{H}_\mu^\dagger(\boldsymbol{A}\boldsymbol{c} + \mu\boldsymbol{b}) + \boldsymbol{G}_\mu^\dagger(\boldsymbol{A}\boldsymbol{c} + \mu\boldsymbol{b}) \to \boldsymbol{U}\boldsymbol{\Sigma}^{-1}\boldsymbol{V}^*\boldsymbol{c} + (\boldsymbol{I}_m - \boldsymbol{U}\boldsymbol{U}^*)\boldsymbol{b}.$$

This is equivalent to the desired claim since $(\boldsymbol{A}^*)^\dagger = \boldsymbol{U}\boldsymbol{\Sigma}^{-1}\boldsymbol{V}^*$ and $\boldsymbol{A}\boldsymbol{A}^\dagger = \boldsymbol{U}\boldsymbol{U}^*$.

</details>
</div>

In light of the above proposition, we take $\boldsymbol{y}(0) = (\boldsymbol{A}^*)^\dagger\boldsymbol{c} + (\boldsymbol{I} - \boldsymbol{A}\boldsymbol{A}^\dagger)\boldsymbol{b}$ as our canonical solution to the dual problem when $\mu = 0$.

Now let $\boldsymbol{x}(\mu)$ denote the solution to $(P_\mu)$ parameterized by $\mu > 0$:

$$\boldsymbol{x}(\mu) = (\boldsymbol{A}^*\boldsymbol{A} + \mu\boldsymbol{I})^{-1}(\boldsymbol{A}^*\boldsymbol{b} - \boldsymbol{c}).$$

If $\boldsymbol{c}$ is orthogonal to the kernel of $\boldsymbol{A}$, then we have

$$\lim_{\mu \downarrow 0} \boldsymbol{x}(\mu) = (\boldsymbol{A}^*\boldsymbol{A})^\dagger(\boldsymbol{A}^*\boldsymbol{b} - \boldsymbol{c}) =: \boldsymbol{x}(0).$$

We actually take the limit above as our canonical solution to the primal problem $(P_0)$ regardless of whether or not $\boldsymbol{c}$ is orthogonal to the kernel of $\boldsymbol{A}$.

## B.4 Minimizing Regularized Quadratics

### B.4.1 A Primer on Kernel Ridge Regression

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kernel Ridge Regression)</span></p>

**Kernel ridge regression (KRR)** is a type of nonparametric regression for learning real-valued nonlinear functions $f : \mathcal{X} \to \mathbb{R}$. It can be formulated as a linear algebra problem: we are given $\lambda > 0$, an $m \times m$ psd "kernel matrix" $\boldsymbol{K}$, and a vector of observations $\boldsymbol{h}$ in $\mathbb{R}^m$; we want to solve

$$\arg\min_{\boldsymbol{\alpha} \in \mathbb{R}^m} \frac{1}{m}\|\boldsymbol{K}\boldsymbol{\alpha} - \boldsymbol{h}\|_2^2 + \lambda\,\boldsymbol{\alpha}^*\boldsymbol{K}\boldsymbol{\alpha}.$$

Equivalently, we want to solve the *KRR normal equations* $(\boldsymbol{K} + m\lambda\boldsymbol{I})\boldsymbol{\alpha} = \boldsymbol{h}$. The normal equations formulation makes it clear that KRR is an instance of the regularized quadratic problem $(R_\mu)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reproducing Kernel Hilbert Space)</span></p>

The kernel function $k$ induces a **reproducing kernel Hilbert space**, $\mathcal{H}$, of real-valued functions on $\mathcal{X}$. This space is (up to closure) equal to the set of real-linear combinations of functions $\boldsymbol{y} \mapsto k^{\boldsymbol{u}}(\boldsymbol{y}) := k(\boldsymbol{y}, \boldsymbol{u})$ parameterized by $\boldsymbol{u} \in \mathcal{X}$.

If the function $\boldsymbol{y} \mapsto f(\boldsymbol{y}) = \sum_{i=1}^{m} \alpha_i k(\boldsymbol{y}, \boldsymbol{x}_i)$ is parameterized by $\boldsymbol{\alpha} \in \mathbb{R}^m$ and $\lbrace \boldsymbol{x}_i \rbrace_{i=1}^{m} \subset \mathcal{X}$, then its squared norm is given by

$$\|f\|_{\mathcal{H}}^2 = \sum_{i=1}^{m}\sum_{j=1}^{m} \alpha_i\alpha_j k(\boldsymbol{x}_i, \boldsymbol{x}_j).$$

Using the kernel matrix $\boldsymbol{K}$ with entries $K_{ij} = k(\boldsymbol{x}_i, \boldsymbol{x}_j)$, we can express that squared norm as $\|f\|_{\mathcal{H}}^2 = \boldsymbol{\alpha}^*\boldsymbol{K}\boldsymbol{\alpha}$.

KRR problem data consists of observations $\lbrace (\boldsymbol{x}_i, h_i) \rbrace_{i=1}^{m} \subset \mathcal{X} \times \mathbb{R}$ and a positive regularization parameter $\lambda$. We presume there are functions $g$ in $\mathcal{H}$ for which $g(\boldsymbol{x}_i) \approx h_i$, and we try to obtain such a function by solving

$$\min_{g \in \mathcal{H}} \frac{1}{m}\sum_{i=1}^{m}(g(\boldsymbol{x}_i) - h_i)^2 + \lambda\|g\|_{\mathcal{H}}^2.$$

It follows that the solution is in the span of the functions $\lbrace k^{x_i} \rbrace_{i=1}^{m}$. Specifically, the solution is $g_\star = \sum_{i=1}^{m} \alpha_i k^{x_i}$ where $\boldsymbol{\alpha}$ solves the KRR optimization problem.

</div>

#### Why Is Ridge Regression a Special Case of KRR?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Ridge Regression as KRR)</span></p>

Suppose we have an $m \times n$ matrix $\boldsymbol{X} = [\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n]$ with linearly independent columns, and that we want to estimate a linear functional $\hat{g} : \mathbb{R}^m \to \mathbb{R}$ given access to the $n$ point evaluations $(\boldsymbol{x}_i, \hat{g}(\boldsymbol{x}_i))_{i=1}^{n}$.

Given a regularization parameter $\lambda > 0$, ridge regression concerns finding the linear function $g : \mathbb{R}^m \to \mathbb{R}$ that minimizes

$$L(\boldsymbol{g}) = \|\boldsymbol{X}^*\boldsymbol{g} - \boldsymbol{h}\|_2^2 + \lambda n\|\boldsymbol{g}\|_2^2,$$

where $\boldsymbol{h} = \boldsymbol{X}^*\hat{\boldsymbol{g}}$. The essential part is showing that the optimal estimate $\boldsymbol{g}$ is in the range of $\boldsymbol{X}$. Using the orthogonal projector $\boldsymbol{P}$ onto the range of $\boldsymbol{X}$ and $\boldsymbol{X}^*\boldsymbol{P}\boldsymbol{g} = \boldsymbol{X}^*\boldsymbol{g}$:

$$L(\boldsymbol{g}) = \|\boldsymbol{X}^*\boldsymbol{P}\boldsymbol{g} - \boldsymbol{h}\|_2^2 + \lambda n(\|\boldsymbol{P}\boldsymbol{g}\|_2^2 + \|(\boldsymbol{I} - \boldsymbol{P})\boldsymbol{g}\|_2^2) \ge L(\boldsymbol{P}\boldsymbol{g}),$$

so $\boldsymbol{g} = \boldsymbol{X}\boldsymbol{\alpha}$ for some $\boldsymbol{\alpha} \in \mathbb{R}^n$. The following problems are then equivalent:

$$\arg\min\lbrace L(g) : g \text{ is a linear functional on } \mathbb{R}^m \rbrace,$$

$$\arg\min\lbrace \|\boldsymbol{X}^*\boldsymbol{X}\boldsymbol{\alpha} - \boldsymbol{h}\|_2^2 + \lambda n\|\boldsymbol{X}\boldsymbol{\alpha}\|_2^2 : \boldsymbol{\alpha} \in \mathbb{R}^n \rbrace, \text{ and}$$

$$\arg\min\lbrace \|\boldsymbol{X}\boldsymbol{\alpha} - \hat{\boldsymbol{g}}\|_2^2 + \lambda n\|\boldsymbol{\alpha}\|_2^2 : \boldsymbol{\alpha} \in \mathbb{R}^n \rbrace.$$

The second of these problems is KRR with a scaled objective and the $n \times n$ kernel matrix $\boldsymbol{K} = \boldsymbol{X}^*\boldsymbol{X}$. The last is ridge regression in the familiar form.

</div>

### B.4.2 Efficient Sketch-and-Solve for Regularized Quadratics

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Sketch-and-Precondition for Regularized Quadratics)</span></p>

Let $\boldsymbol{G}$ be an $m \times m$ psd matrix and $\mu$ be a positive regularization parameter. The sketch-and-solve approach to KRR from [AM15] can be considered generically as a sketch-and-solve approach to the regularized quadratic minimization problem $(R_\mu)$. The generic formulation is to approximate $\boldsymbol{G} \approx \boldsymbol{A}\boldsymbol{A}^*$ with an $m \times n$ matrix $\boldsymbol{A}$ ($m \gg n$) and then solve

$$(\boldsymbol{A}\boldsymbol{A}^* + \mu\boldsymbol{I})\,\boldsymbol{z} = \boldsymbol{h}.$$

Identifying $\boldsymbol{b} = \boldsymbol{h}/\mu$, $\boldsymbol{c} = \boldsymbol{0}$, and $\boldsymbol{y} = \boldsymbol{z}$ shows that this amounts to a dual saddle point problem of the form $(D_\mu)$.

Here we explain how the sketch-and-precondition paradigm can efficiently be applied to solve this under the assumption that $\boldsymbol{A}\boldsymbol{A}^*$ defines a Nyström approximation of $\boldsymbol{G}$.

Let $\boldsymbol{S}_o$ be an initial $m \times n$ sketching operator. The resulting sketch $\boldsymbol{Y} = \boldsymbol{G}\boldsymbol{S}_o$ and factor $\boldsymbol{R} = \operatorname{chol}(\boldsymbol{S}_o^*\boldsymbol{Y})$ together define $\boldsymbol{A} = \boldsymbol{Y}\boldsymbol{R}^{-1}$. For the sketching phase of preconditioner generation, we sample a $d \times m$ operator $\boldsymbol{S}$ (with $d \gtrsim n$) and set

$$\boldsymbol{A}_\mu^{\text{sk}} = \begin{bmatrix} \boldsymbol{S} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{I} \end{bmatrix} \boldsymbol{A}_\mu = \begin{bmatrix} \boldsymbol{S}\boldsymbol{Y} \\ \sqrt{\mu}\boldsymbol{R} \end{bmatrix}\boldsymbol{R}^{-1}.$$

We then compute the SVD of the augmented matrix

$$\begin{bmatrix} \boldsymbol{S}\boldsymbol{Y} \\ \sqrt{\mu}\boldsymbol{R} \end{bmatrix} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^*$$

and find that setting $\boldsymbol{M} = \boldsymbol{R}\boldsymbol{V}\boldsymbol{\Sigma}^{-1}$ satisfies $\boldsymbol{A}_\mu^{\text{sk}}\boldsymbol{M} = \boldsymbol{U}$. The preconditioned linear operator $\boldsymbol{A}_\mu\boldsymbol{M}$ (and its adjoint) should be applied in the iterative solver by noting the identity

$$\begin{bmatrix} \boldsymbol{A} \\ \sqrt{\mu}\boldsymbol{I} \end{bmatrix}\boldsymbol{M} = \begin{bmatrix} \boldsymbol{Y} \\ \sqrt{\mu}\boldsymbol{R} \end{bmatrix}\boldsymbol{V}\boldsymbol{\Sigma}^{-1}.$$

This identity is important since it shows that $\boldsymbol{R}^{-1}$ need never be applied at any point in the sketch-and-precondition approach.

</div>

# Appendix C: Details on Low-Rank Approximation

## C.1 Theory for Submatrix-Oriented Decompositions

### C.1.1 Approximation Quality in Low-Rank ID

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(C.1.1 — Approximation Quality in Low-Rank ID)</span></p>

Let $\tilde{\boldsymbol{A}}$ be any rank-$k$ approximation of $\boldsymbol{A}$ that satisfies the spectral norm error bound $\|\boldsymbol{A} - \tilde{\boldsymbol{A}}\|_2 \le \epsilon$. If $\tilde{\boldsymbol{A}} = \tilde{\boldsymbol{A}}[:,J]\boldsymbol{X}$ for some $k \times n$ matrix $\boldsymbol{X}$ and an index vector $J$, then $\hat{\boldsymbol{A}} = \boldsymbol{A}[:,J]\boldsymbol{X}$ is a low-rank column ID that satisfies

$$\|\boldsymbol{A} - \hat{\boldsymbol{A}}\|_2 \le (1 + \|\boldsymbol{X}\|_2)\,\epsilon.$$

Furthermore, if $|X_{ij}| \le M$ for all $(i,j)$, then

$$\|\boldsymbol{X}\|_2 \le \sqrt{1 + M^2 k(n-k)}.$$

</div>

*Proof sketch.* By the triangle inequality,

$$\|\boldsymbol{A} - \hat{\boldsymbol{A}}\|_2 = \|(\boldsymbol{A} - \tilde{\boldsymbol{A}}) + (\tilde{\boldsymbol{A}} - \hat{\boldsymbol{A}})\|_2 \le \|\boldsymbol{A} - \tilde{\boldsymbol{A}}\|_2 + \|\tilde{\boldsymbol{A}} - \hat{\boldsymbol{A}}\|_2.$$

The first term is bounded by $\epsilon$ by assumption. For the second term, using $\tilde{\boldsymbol{A}} - \hat{\boldsymbol{A}} = (\tilde{\boldsymbol{A}}[:,J] - \boldsymbol{A}[:,J])\boldsymbol{X}$ and submultiplicativity of the spectral norm gives $\|\tilde{\boldsymbol{A}} - \hat{\boldsymbol{A}}\|_2 \le \|\tilde{\boldsymbol{A}}[:,J] - \boldsymbol{A}[:,J]\|_2 \|\boldsymbol{X}\|_2 \le \epsilon \|\boldsymbol{X}\|_2$.

For the bound on $\|\boldsymbol{X}\|_2$: since $\tilde{\boldsymbol{A}} = \boldsymbol{C}\boldsymbol{X}$ has rank $k$ and $\boldsymbol{X}$ is $k \times n$, we can infer that $\boldsymbol{X}[:,J] = \boldsymbol{I}_k$. Then there exists a permutation $P$ such that $\boldsymbol{X}[:,P] = [\boldsymbol{I}_k, \boldsymbol{V}]$ for a $k \times (n-k)$ matrix $\boldsymbol{V}$ with $|V_{ij}| \le M$. Since permuting columns does not change the spectral norm, it suffices to bound $\|[\boldsymbol{I}_k, \boldsymbol{V}]\|_2$. Using $\|\boldsymbol{W}\|_2 \le \sqrt{\|\boldsymbol{U}\|_2^2 + \|\boldsymbol{V}\|_{\mathrm{F}}^2}$ for any block matrix $\boldsymbol{W} = [\boldsymbol{U}, \boldsymbol{V}]$, combined with $\|\boldsymbol{I}_k\|_2 = 1$ and $\|\boldsymbol{V}\|_{\mathrm{F}}^2 \le M^2 k(n-k)$, yields the result.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(C.1.2)</span></p>

The bound on $\|\boldsymbol{X}\|_2$ is not the best possible. Looking at the final steps of the proof, it suffices for $M$ to bound the entries of $\boldsymbol{X}$ that are *not* part of the identity submatrix.

</div>

### C.1.2 Truncation in Column-Pivoted Matrix Decompositions

This part examines how changing the basis of a column-pivoted decomposition can affect approximation quality when truncating these decompositions.

The matrix of interest $\boldsymbol{G}$ is $r \times c$ and is given through a decomposition $\boldsymbol{G}\boldsymbol{P} = \boldsymbol{F}\boldsymbol{T}$ for a permutation matrix $\boldsymbol{P}$ and upper-triangular $\boldsymbol{T}$. Let $w = \min\lbrace c, r \rbrace$ columns (so $\boldsymbol{T}$ has as many rows). We use $\mathcal{U}$ to denote the set of invertible upper-triangular matrices of order $w$. For a positive integer $k < \operatorname{rank}(\boldsymbol{G})$, the matrix-valued function $g_k$ is defined on $\mathcal{U}$ as

$$g_k(\boldsymbol{U}) = \boldsymbol{F}(\boldsymbol{U}^{-1})[:,:k]\,\boldsymbol{U}[:k,:]\,\boldsymbol{T}\boldsymbol{P}^*.$$

Note that for every diagonal $\boldsymbol{D} \in \mathcal{U}$ we have $g_k(\boldsymbol{D}) = (\boldsymbol{F}[:,:k])(\boldsymbol{T}[:k,:])\boldsymbol{P}^*$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(C.1.3 — Truncation in Column-Pivoted Decompositions)</span></p>

Partition the factors $\boldsymbol{F}$ and $\boldsymbol{T}$ into blocks $[\boldsymbol{F}_1, \boldsymbol{F}_2]$ and $[\boldsymbol{T}_1; \boldsymbol{T}_2]$ so that $\boldsymbol{F}_1$ has $k$ columns and $\boldsymbol{T}_1$ has $k$ rows. If $\boldsymbol{U}_\star$ is an optimal solution to

$$\min_{\boldsymbol{U} \in \mathcal{U}} \|\boldsymbol{G} - g_k(\boldsymbol{U})\|_{\mathrm{F}},$$

then the following identity holds in any unitarily invariant norm:

$$\|\boldsymbol{G} - g_k(\boldsymbol{U}_\star)\| = \left\|\left(\boldsymbol{I} - \boldsymbol{F}_1 \boldsymbol{F}_1^\dagger\right)\boldsymbol{F}_2 \boldsymbol{T}_2\right\|.$$

Furthermore, $\|\boldsymbol{G} - g_k(\boldsymbol{I}_{w \times w})\| = \|\boldsymbol{F}_2 \boldsymbol{T}_2\|$, and the identity matrix is optimal if and only if $\operatorname{range}(\boldsymbol{F}_2)$ and $\operatorname{range}(\boldsymbol{F}_1)$ are orthogonal.

</div>

*Proof sketch.* Partition $\boldsymbol{T}$ further into blocks:

$$\boldsymbol{T} = \begin{bmatrix} \boldsymbol{T}_1 \\ \boldsymbol{T}_2 \end{bmatrix} = \begin{bmatrix} \boldsymbol{T}_{11} & \boldsymbol{T}_{12} \\ \boldsymbol{0} & \boldsymbol{T}_{22} \end{bmatrix}$$

where $\boldsymbol{T}_{11}$ is $k \times k$. Introduce $\boldsymbol{U} \in \mathcal{U}$ and partition it similarly into blocks $\boldsymbol{U}_{11}, \boldsymbol{U}_{12}, \boldsymbol{U}_{22}$. Setting $\boldsymbol{V} = \boldsymbol{U}^{-1}$ and partitioning $\boldsymbol{V} = [\boldsymbol{V}_1, \boldsymbol{V}_2]$, one derives

$$g_k(\boldsymbol{U}) = \boldsymbol{F}\boldsymbol{V}_1\boldsymbol{U}_1\boldsymbol{T}\boldsymbol{P}^*$$

Using the block matrix-inversion identity $\boldsymbol{F}\boldsymbol{V}_1 = \boldsymbol{F}_1\boldsymbol{U}_{11}^{-1}$ and the block multiply $\boldsymbol{U}_1\boldsymbol{T} = \boldsymbol{U}_{11}\boldsymbol{T}_1 + \boldsymbol{U}_{12}\boldsymbol{T}_2$, we obtain

$$g_k(\boldsymbol{U}) = \boldsymbol{F}_1\left(\boldsymbol{T}_1 + \boldsymbol{U}_{11}^{-1}\boldsymbol{U}_{12}\boldsymbol{T}_2\right)\boldsymbol{P}^*.$$

Since $\boldsymbol{G} = \boldsymbol{F}\boldsymbol{T}\boldsymbol{P}^*$, the difference is

$$\boldsymbol{G} - g_k(\boldsymbol{U}) = (\boldsymbol{F}_2\boldsymbol{T}_2 - \boldsymbol{F}_1\boldsymbol{U}_{11}^{-1}\boldsymbol{U}_{12}\boldsymbol{T}_2)\boldsymbol{P}^*.$$

In any unitarily invariant norm, $\|\boldsymbol{G} - g_k(\boldsymbol{U})\| = \|\boldsymbol{F}_2\boldsymbol{T}_2 - \boldsymbol{F}_1\boldsymbol{U}_{11}^{-1}\boldsymbol{U}_{12}\boldsymbol{T}_2\|$. The claim about diagonal $\boldsymbol{U}$ follows since $\boldsymbol{U}_{12} = \boldsymbol{0}$. For the optimization, replacing $\boldsymbol{U}_{11}^{-1}\boldsymbol{U}_{12}$ by a general $k \times (w-k)$ matrix $\boldsymbol{M}$, the optimal $\boldsymbol{M}_\star = \boldsymbol{F}_1^\dagger \boldsymbol{F}_2$.

## C.2 Computational Routine Interfaces and Implementations

The design space for low-rank approximation algorithms is large. This appendix illustrates its breadth and depth with pseudocode for computational routines needed for four drivers: `SVD1`, `EVD1`, `EVD2`, and `CURD1`. All pseudocode uses Python-style zero-based indexing.

Three interfaces are central to low-rank approximation:

- **$\boldsymbol{Y} = \texttt{Orth}(\boldsymbol{X})$** returns an orthonormal basis for the range of a tall input matrix; the number of columns in $\boldsymbol{Y}$ will never be larger than that of $\boldsymbol{X}$ and may be smaller. The simplest implementation is to return the orthogonal factor from an economic QR decomposition of $\boldsymbol{X}$.

- **$\boldsymbol{S} = \texttt{SketchOpGen}(\ell, k)$** returns an $\ell \times k$ oblivious sketching operator sampled from some predetermined distribution. The most common distributions used for low-rank approximation were covered in Section 2.3. In actual implementations, this function would accept an input representing the state of the random number generator.

- **$\boldsymbol{Y} = \texttt{Stabilizer}(\boldsymbol{X})$** has similar semantics to `Orth`. It differs in that it only requires $\boldsymbol{Y}$ to be better-conditioned than $\boldsymbol{X}$ while preserving its range. The relaxed semantics open up the possibility of methods that are less expensive than computing an orthonormal basis, such as taking the lower-triangular factor from an LU decomposition with column pivoting.

### C.2.1 Power Iteration for Data-Aware Sketching

When a `TallSketchOpGen` is called with parameters $(\boldsymbol{A}, k)$, it produces an $n \times k$ sketching operator where $\operatorname{range}(\boldsymbol{S})$ is reasonably well-aligned with the subspace spanned by the $k$ leading right singular vectors of $\boldsymbol{A}$. One extreme case of interest is to return an oblivious sketching operator without reading any entries of $\boldsymbol{A}$.

This method uses a $p$-step power iteration technique. When $p = 0$, the method returns an oblivious sketching operator. It is recommended that one use $p > 0$ (e.g., $p \in \lbrace 2, 3 \rbrace$) when the singular values of $\boldsymbol{A}$ exhibit "slow" decay.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(8 — TSOG1: TallSketchOpGen via Power Method)</span></p>

**function** $\texttt{TSOG1}(\boldsymbol{A}, k)$

**Inputs:**
- $\boldsymbol{A}$ is $m \times n$, and $k \ll \min\lbrace m, n \rbrace$ is a positive integer.

**Output:**
- $\boldsymbol{S}$ is $n \times k$, intended for later use in computing $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{S}$.

**Tuning parameters:**
- $p \ge 0$ controls the number of steps in the power method. It equals the total number of matrix-matrix multiplications involving $\boldsymbol{A}$ or $\boldsymbol{A}^*$. If $p = 0$ then this function returns an oblivious sketching operator.
- $q \ge 1$ is the number of matrix-matrix multiplications with $\boldsymbol{A}$ or $\boldsymbol{A}^*$ that accumulate before the stabilizer is called.

**Procedure:**
1. $p_{\text{done}} = 0$
2. If $p$ is even: $\boldsymbol{S} = \texttt{SketchOpGen}(n, k)$; else: $\boldsymbol{S} = \boldsymbol{A}^* \texttt{SketchOpGen}(m, k)$, $p_{\text{done}} \mathrel{+}= 1$, and stabilize $\boldsymbol{S}$ if $p_{\text{done}} \bmod q = 0$.
3. **while** $p - p_{\text{done}} \ge 2$ **do:**
   - $\boldsymbol{S} = \boldsymbol{A}\boldsymbol{S}$, $p_{\text{done}} \mathrel{+}= 1$, stabilize if $p_{\text{done}} \bmod q = 0$.
   - $\boldsymbol{S} = \boldsymbol{A}^*\boldsymbol{S}$, $p_{\text{done}} \mathrel{+}= 1$, stabilize if $p_{\text{done}} \bmod q = 0$.
4. **return** $\boldsymbol{S}$

</div>

### C.2.2 RangeFinders and QB Decompositions

A general `RangeFinder` takes in a matrix $\boldsymbol{A}$ and a target rank parameter $k$, and returns a matrix $\boldsymbol{Q}$ of rank $d = \min\lbrace k, \operatorname{rank}(\boldsymbol{A}) \rbrace$ such that the range of $\boldsymbol{Q}$ is an approximation to the space spanned by $\boldsymbol{A}$'s top $d$ left singular vectors.

The rangefinder problem may also be viewed as follows: given $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ and a target rank $k \ll \min(m,n)$, find a matrix $\boldsymbol{Q}$ with $k$ columns such that the error $\|\boldsymbol{A} - \boldsymbol{Q}\boldsymbol{Q}^*\boldsymbol{A}\|$ is "reasonably" small. Some `RangeFinder` implementations are iterative and can accept a target accuracy as a third argument.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(9 — RF1: RangeFinder via Single Row Sketch)</span></p>

**function** $\texttt{RF1}(\boldsymbol{A}, k)$

**Inputs:**
- $\boldsymbol{A}$ is $m \times n$, and $k \ll \min\lbrace m, n \rbrace$ is a positive integer.

**Output:**
- $\boldsymbol{Q}$ is a column-orthonormal matrix with $d = \min\lbrace k, \operatorname{rank}(\boldsymbol{A}) \rbrace$ columns. We have $\operatorname{range}(\boldsymbol{Q}) \subset \operatorname{range}(\boldsymbol{A})$; it is intended that $\operatorname{range}(\boldsymbol{Q})$ is an approximation to the space spanned by $\boldsymbol{A}$'s top $d$ left singular vectors.

**Procedure:**
1. $\boldsymbol{S} = \texttt{TallSketchOpGen}(\boldsymbol{A}, k)$ — $\boldsymbol{S}$ is $n \times k$
2. $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{S}$
3. $\boldsymbol{Q} = \texttt{Orth}(\boldsymbol{Y})$
4. **return** $\boldsymbol{Q}$

</div>

The conceptual goal of QB decomposition algorithms is to produce an approximation $\|\boldsymbol{A} - \boldsymbol{Q}\boldsymbol{B}\| \le \epsilon$ (for some unitarily-invariant norm), where $\operatorname{rank}(\boldsymbol{Q}\boldsymbol{B}) \le \min\lbrace k, \operatorname{rank}(\boldsymbol{A}) \rbrace$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(10 — QB1: QBDecomposer via RangeFinder)</span></p>

**function** $\texttt{QB1}(\boldsymbol{A}, k, \epsilon)$

**Inputs:**
- $\boldsymbol{A}$ is an $m \times n$ matrix and $k \ll \min\lbrace m, n \rbrace$ is a positive integer.
- $\epsilon$ is a target for the relative error $\|\boldsymbol{A} - \boldsymbol{Q}\boldsymbol{B}\| / \|\boldsymbol{A}\|$ measured in some unitarily-invariant norm. This parameter is passed directly to the `RangeFinder`.

**Output:**
- $\boldsymbol{Q}$: an $m \times d$ matrix returned by the underlying `RangeFinder`.
- $\boldsymbol{B} = \boldsymbol{Q}^*\boldsymbol{A}$ is $d \times n$; we can be certain that $d \le \min\lbrace k, \operatorname{rank}(\boldsymbol{A}) \rbrace$. The matrix $\boldsymbol{Q}\boldsymbol{B}$ is a low-rank approximation of $\boldsymbol{A}$.

**Procedure:**
1. $\boldsymbol{Q} = \texttt{RangeFinder}(\boldsymbol{A}, k, \epsilon)$
2. $\boldsymbol{B} = \boldsymbol{Q}^*\boldsymbol{A}$
3. **return** $\boldsymbol{Q}, \boldsymbol{B}$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(11 — QB2: Fully-Adaptive QBDecomposer)</span></p>

**function** $\texttt{QB2}(\boldsymbol{A}, k, \epsilon)$

**Inputs:**
- $\boldsymbol{A}$ is an $m \times n$ matrix and $k \ll \min\lbrace m, n \rbrace$ is a positive integer.
- $\epsilon$ is a target for the relative error $\|\boldsymbol{A} - \boldsymbol{Q}\boldsymbol{B}\|_{\mathrm{F}} / \|\boldsymbol{A}\|_{\mathrm{F}}$. This parameter is used as a termination criterion upon reaching the desired accuracy.

**Output:**
- $\boldsymbol{Q}$: an $m \times d$ matrix combined of successive outputs from the underlying `RangeFinder`.
- $\boldsymbol{B} = \boldsymbol{Q}^*\boldsymbol{A}$ is $d \times n$; $d \le \min\lbrace k, \operatorname{rank}(\boldsymbol{A}) \rbrace$. The matrix $\boldsymbol{Q}\boldsymbol{B}$ is a low-rank approximation of $\boldsymbol{A}$.

**Tuning parameter:** block\_size $\ge 1$ — at every iteration (except possibly the final), block\_size columns are added to $\boldsymbol{Q}$.

**Procedure:**
1. $d = 0$, $\boldsymbol{Q} = [\ ] \in \mathbb{R}^{m \times d}$, $\boldsymbol{B} = [\ ] \in \mathbb{R}^{d \times n}$, squared\_error $= \|\boldsymbol{A}\|_{\mathrm{F}}^2$
2. **while** $k > d$ **do:**
   - block\_size $= \min\lbrace$block\_size$, k - d\rbrace$
   - $\boldsymbol{Q}_i = \texttt{RangeFinder}(\boldsymbol{A},$ block\_size$)$
   - $\boldsymbol{Q}_i = \texttt{Orth}(\boldsymbol{Q}_i - \boldsymbol{Q}(\boldsymbol{Q}^*\boldsymbol{Q}_i))$ — for numerical stability
   - $\boldsymbol{B}_i = \boldsymbol{Q}_i^*\boldsymbol{A}$ — original matrix $\boldsymbol{A}$ is valid here
   - $\boldsymbol{B} = \begin{bmatrix} \boldsymbol{B} \\ \boldsymbol{B}_i \end{bmatrix}$, $\boldsymbol{Q} = [\boldsymbol{Q},\ \boldsymbol{Q}_i]$
   - $d \mathrel{+}=$ block\_size
   - $\boldsymbol{A} = \boldsymbol{A} - \boldsymbol{Q}_i\boldsymbol{B}_i$ — modification can be implicit, but is required by the `RangeFinder` call
   - squared\_error $=$ squared\_error $- \|\boldsymbol{B}_i\|_{\mathrm{F}}^2$ — compute by a stable method
   - **if** squared\_error $\le \epsilon^2$ **then break**
3. **return** $\boldsymbol{Q}, \boldsymbol{B}$

</div>

The third and final QB algorithm also builds up its approximation incrementally. It is called *pass-efficient* because it does not access the data matrix $\boldsymbol{A}$ within its main loop. The algorithm can use a requested error tolerance as an early-stopping criterion. This function should never be called with $k = \min\lbrace m, n \rbrace$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(12 — QB3: Pass-Efficient and Partially Adaptive QBDecomposer)</span></p>

**function** $\texttt{QB3}(\boldsymbol{A}, k, \epsilon)$

**Inputs:**
- $\boldsymbol{A}$ is an $m \times n$ matrix and $k \ll \min\lbrace m, n \rbrace$ is a positive integer.
- $\epsilon$ is a target for the relative error $\|\boldsymbol{A} - \boldsymbol{Q}\boldsymbol{B}\|_{\mathrm{F}} / \|\boldsymbol{A}\|_{\mathrm{F}}$. Used as a termination criterion.

**Output:**
- $\boldsymbol{Q}$: an $m \times d$ matrix combined of successively-computed orthonormal bases $\boldsymbol{Q}_i$.
- $\boldsymbol{B} = \boldsymbol{Q}^*\boldsymbol{A}$ is $d \times n$; $d \le \min\lbrace k, \operatorname{rank}(\boldsymbol{A}) \rbrace$. The matrix $\boldsymbol{Q}\boldsymbol{B}$ is a low-rank approximation of $\boldsymbol{A}$.

**Tuning parameter:** block\_size is a positive integer; at every iteration (except possibly the last), block\_size columns are added to $\boldsymbol{Q}$.

**Procedure:**
1. $\boldsymbol{Q} = [\ ] \in \mathbb{R}^{m \times 0}$, $\boldsymbol{B} = [\ ] \in \mathbb{R}^{0 \times n}$, squared\_error $= \|\boldsymbol{A}\|_{\mathrm{F}}^2$
2. $\boldsymbol{S} = \texttt{TallSketchOpGen}(\boldsymbol{A}, k)$
3. $\boldsymbol{G} = \boldsymbol{A}\boldsymbol{S}$, $\boldsymbol{H} = \boldsymbol{A}^*\boldsymbol{G}$ — can be done in one pass over $\boldsymbol{A}$
4. max\_blocks $= \lceil k / \text{block\_size} \rceil$, $i = 0$
5. **while** $i <$ max\_blocks **do:**
   - $b_{\text{start}} = i \cdot \text{block\_size} + 1$, $b_{\text{end}} = \min\lbrace (i+1) \cdot \text{block\_size},\, k \rbrace$
   - $\boldsymbol{S}_i = \boldsymbol{S}[:,\, b_{\text{start}} : b_{\text{end}}]$
   - $\boldsymbol{Y}_i = \boldsymbol{G}[:,\, b_{\text{start}} : b_{\text{end}}] - \boldsymbol{Q}(\boldsymbol{B}\boldsymbol{S}_i)$
   - $\boldsymbol{Q}_i, \boldsymbol{R}_i = \texttt{qr}(\boldsymbol{Y}_i)$ — next three lines are for numerical stability
   - $\boldsymbol{Q}_i = \boldsymbol{Q}_i - \boldsymbol{Q}(\boldsymbol{Q}^*\boldsymbol{Q}_i)$
   - $\boldsymbol{Q}_i, \hat{\boldsymbol{R}}_i = \texttt{qr}(\boldsymbol{Q}_i)$, $\boldsymbol{R}_i = \hat{\boldsymbol{R}}_i \boldsymbol{R}_i$
   - $\boldsymbol{B}_i = (\boldsymbol{H}[:,\, b_{\text{start}} : b_{\text{end}}])^* - (\boldsymbol{Y}_i\boldsymbol{Q})\boldsymbol{B} - (\boldsymbol{B}\boldsymbol{S}_i)^*\boldsymbol{B}$
   - $\boldsymbol{B}_i = (\boldsymbol{R}_i^*)^{-1}\boldsymbol{B}_i$ — in-place triangular solve
   - $\boldsymbol{B} = \begin{bmatrix} \boldsymbol{B} \\ \boldsymbol{B}_i \end{bmatrix}$, $\boldsymbol{Q} = [\boldsymbol{Q},\ \boldsymbol{Q}_i]$
   - squared\_error $=$ squared\_error $- \|\boldsymbol{B}_i\|_{\mathrm{F}}^2$ — compute by a stable method
   - $i \mathrel{+}= 1$
   - **if** squared\_error $\le \epsilon^2$ **then break**
6. **return** $\boldsymbol{Q}, \boldsymbol{B}$

</div>

### C.2.3 ID and Subset Selection

The collective design space of algorithms for ID, subset selection, and CUR is very large. This appendix presents one randomized algorithm for one-sided ID and an analogous randomized algorithm for subset selection.

Two deterministic functions are needed. The first — called as $\boldsymbol{Q}, \boldsymbol{R}, J = \texttt{qrcp}(\boldsymbol{F}, k)$ — returns data for an economic QR decomposition with column pivoting, where the decomposition is restricted to rank $k$ and may be incomplete. The second deterministic function is the canonical way to use QRCP for one-sided ID. It produces a column ID when the final argument "axis" is set to one; otherwise, it produces a row ID. When used for column ID, it's typical for $\boldsymbol{Y} \in \mathbb{R}^{\ell \times w}$ to be (very) wide and for $k$ to be only slightly smaller than $\ell$ (say, $\ell/2 \le k \le \ell$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(13 — osid_qrcp: Deterministic One-Sided ID Based on QRCP)</span></p>

**function** $\texttt{osid\_qrcp}(\boldsymbol{Y}, k, \text{axis})$

**Inputs:**
- $\boldsymbol{Y}$ is an $\ell \times w$ matrix, typically a sketch of some larger matrix.
- $k$ is an integer, typically close to $\min\lbrace \ell, w \rbrace$.
- axis is an integer: equals 1 for row ID and 2 for column ID.

**Outputs:**
- When axis $= 1$: $\boldsymbol{Z}$ is $\ell \times k$ and $I$ is a length-$k$ index vector, satisfying $\boldsymbol{Y}[I,:] = (\boldsymbol{Z}\boldsymbol{Y}[I,:])[I,:]$.
- When axis $= 2$: $\boldsymbol{X}$ is $k \times w$ and $J$ is a length-$k$ index vector, satisfying $\boldsymbol{Y}[:,J] = (\boldsymbol{Y}[:,J]\boldsymbol{X})[:,J]$.

**Procedure (axis $= 2$):**
1. $(\ell, w) =$ shape of $\boldsymbol{Y}$; assert $k \le \min\lbrace \ell, w \rbrace$
2. $\boldsymbol{Q}, \boldsymbol{R}, J = \texttt{qrcp}(\boldsymbol{Y}, k)$
3. $\boldsymbol{T} = (\boldsymbol{R}[:k, :k])^{-1}\,\boldsymbol{R}[:k, k+1:]$ — use trsm from BLAS 3
4. $\boldsymbol{X} = \texttt{zeros}(k, w)$, $\boldsymbol{X}[:,J] = [\boldsymbol{I}_{k \times k},\, \boldsymbol{T}]$, $J = J[:k]$
5. **return** $\boldsymbol{X}, J$

**Procedure (axis $= 1$):** Compute $\boldsymbol{X}, I = \texttt{osid\_qrcp}(\boldsymbol{Y}^*, k, \text{axis} = 1)$, then set $\boldsymbol{Z} = \boldsymbol{X}^*$ and **return** $\boldsymbol{Z}, I$.

</div>

The one-sided ID interface is $\boldsymbol{M}, P = \texttt{OneSidedID}(\boldsymbol{A}, k, s, \text{axis})$. The output value $\boldsymbol{M}$ is the interpolation matrix and $P$ is the length-$k$ vector of skeleton indices. When axis $= 1$ we are considering a row ID and so obtain the approximation $\hat{\boldsymbol{A}} = \boldsymbol{M}\boldsymbol{A}[P,:]$ to $\boldsymbol{A}$. When axis $= 2$, we are considering the low-rank column ID $\hat{\boldsymbol{A}} = \boldsymbol{A}[:,P]\boldsymbol{M}$. Implementations of this interface perform internal calculations with sketches of rank $k + s$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(14 — OSID1: OneSidedID via Sketch Re-purposing)</span></p>

**function** $\texttt{OSID1}(\boldsymbol{A}, k, \text{axis})$

**Inputs:**
- $\boldsymbol{A}$ is an $m \times n$ matrix and $k \ll \min\lbrace m, n \rbrace$ is a positive integer.
- axis is an integer, equal to 1 for row ID or 2 for column ID.

**Tuning parameter:** $s$ is a nonnegative integer. The algorithm internally works with a sketch of rank $k + s$.

**Procedure (axis $= 1$, row ID):**
1. $\boldsymbol{S} = \texttt{TallSketchOpGen}(\boldsymbol{A}, k + s)$
2. $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{S}$
3. $\boldsymbol{Z}, I = \texttt{osid\_qrcp}(\boldsymbol{Y}, k, \text{axis} = 0)$
4. **return** $\boldsymbol{Z}, I$

**Procedure (axis $= 2$, column ID):**
1. $\boldsymbol{S} = \texttt{TallSketchOpGen}(\boldsymbol{A}^*, k + s)^*$
2. $\boldsymbol{Y} = \boldsymbol{S}\boldsymbol{A}$
3. $\boldsymbol{X}, J = \texttt{osid\_qrcp}(\boldsymbol{Y}, k, \text{axis} = 1)$
4. **return** $\boldsymbol{X}, J$

</div>

Consider the following interface for (randomized) row and column subset selection algorithms: $P = \texttt{RowOrColSelection}(\boldsymbol{A}, k, s, \text{axis})$. The index vector $P$ and oversampling parameter is understood in the same way as the `OneSidedID` interface. Implementations are supposed to perform internal calculations with sketches of rank $k + s$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(15 — ROCS1: RowOrColSelection via QRCP on a Sketch)</span></p>

**function** $\texttt{ROCS1}(\boldsymbol{A}, k, s, \text{axis})$

**Inputs:**
- $\boldsymbol{A}$ is an $m \times n$ matrix and $k \ll \min\lbrace m, n \rbrace$ is a positive integer.
- axis is an integer, equal to 1 for row selection or 2 for column selection.

**Output:**
- $I$: a row selection vector of length $k$, **or** $J$: a column selection vector of length $k$.

**Tuning parameter:** $s$ is a nonnegative integer. The algorithm internally works with a sketch of rank $k + s$.

**Procedure (axis $= 1$, row selection):**
1. $\boldsymbol{S} = \texttt{TallSketchOpGen}(\boldsymbol{A}, k + s)$
2. $\boldsymbol{Y} = \boldsymbol{A}\boldsymbol{S}$
3. $\boldsymbol{Q}, \boldsymbol{R}, I = \texttt{qrcp}(\boldsymbol{Y}^*)$
4. **return** $I[:k]$

**Procedure (axis $= 2$, column selection):**
1. $\boldsymbol{S} = \texttt{TallSketchOpGen}(\boldsymbol{A}^*, k + s)^*$
2. $\boldsymbol{Y} = \boldsymbol{S}\boldsymbol{A}$
3. $\boldsymbol{Q}, \boldsymbol{R}, J = \texttt{qrcp}(\boldsymbol{Y})$
4. **return** $J[:k]$

</div>

# Appendix D: Correctness of Preconditioned Cholesky QRCP

This appendix proves Proposition 5.1.2. We begin with a detailed statement of the algorithm.

Let $\boldsymbol{A}$ be $m \times n$ and $\boldsymbol{S}$ be $d \times m$ with $n \le d \ll m$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Preconditioned Cholesky QRCP — Detailed Statement)</span></p>

1. Compute the sketch $\boldsymbol{A}^{\text{sk}} = \boldsymbol{S}\boldsymbol{A}$.
2. Decompose $[\boldsymbol{Q}^{\text{sk}}, \boldsymbol{R}^{\text{sk}}, J] = \texttt{qrcp}(\boldsymbol{A}^{\text{sk}})$:
   - $J$ is a permutation vector for the index set $[\![n]\!]$.
   - Abbreviating $\boldsymbol{A}_J^{\text{sk}} = \boldsymbol{A}^{\text{sk}}[:,J]$, we have $\boldsymbol{A}_J^{\text{sk}} = \boldsymbol{Q}^{\text{sk}}\boldsymbol{R}^{\text{sk}}$.
   - Let $k = \operatorname{rank}(\boldsymbol{A}^{\text{sk}})$.
   - $\boldsymbol{Q}^{\text{sk}}$ is $m \times k$ and column-orthonormal.
   - $\boldsymbol{R}^{\text{sk}} = [\boldsymbol{R}_1^{\text{sk}}, \boldsymbol{R}_2^{\text{sk}}]$ is $k \times n$ upper-triangular.
   - $\boldsymbol{R}_1^{\text{sk}}$ is $k \times k$ and nonsingular.
3. Abbreviate $\boldsymbol{A}_J = \boldsymbol{A}[:,J]$ and explicitly form $\boldsymbol{A}^{\text{pre}} = \boldsymbol{A}_J[:,:k](\boldsymbol{R}_1^{\text{sk}})^{-1}$.
4. Compute an unpivoted QR decomposition $\boldsymbol{A}^{\text{pre}} = \boldsymbol{Q}\boldsymbol{R}^{\text{pre}}$.
   - If $\operatorname{rank}(\boldsymbol{A}) = k$ then $\boldsymbol{Q}$ is an orthonormal basis for the range of $\boldsymbol{A}$.
   - We assume this decomposition is exact.
5. Explicitly form $\boldsymbol{R} = \boldsymbol{R}^{\text{pre}}\boldsymbol{R}^{\text{sk}}$.

</div>

The goal is to show that $\boldsymbol{A}[:,J] = \boldsymbol{Q}\boldsymbol{R}$ under the assumption that $\operatorname{rank}(\boldsymbol{S}\boldsymbol{A}) = \operatorname{rank}(\boldsymbol{A})$.

By steps 3 and 4, we know that

$$\boldsymbol{R}^{\text{pre}} = \boldsymbol{Q}^*\boldsymbol{A}_J[:,:k](\boldsymbol{R}_1^{\text{sk}})^{-1}.$$

Combining this with the characterization of $\boldsymbol{R}$ from steps 2e and 5:

$$\boldsymbol{R} = \boldsymbol{Q}^*\boldsymbol{A}_J[:,:k](\boldsymbol{R}_1^{\text{sk}})^{-1}[\boldsymbol{R}_1^{\text{sk}},\ \boldsymbol{R}_2^{\text{sk}}].$$

Expanding further:

$$\boldsymbol{R} = \boldsymbol{Q}^*\boldsymbol{A}_J[:,:k][\boldsymbol{I}_{k \times k},\ (\boldsymbol{R}_1^{\text{sk}})^{-1}\boldsymbol{R}_2^{\text{sk}}].$$

Since $\boldsymbol{Q}$ is an orthonormal basis for the range of $\boldsymbol{A}$ and consequently $\boldsymbol{A}_J$, we have

$$\boldsymbol{Q}\boldsymbol{R} = \boldsymbol{A}_J[:,:k][\boldsymbol{I}_{k \times k},\ (\boldsymbol{R}_1^{\text{sk}})^{-1}\boldsymbol{R}_2^{\text{sk}}]. \tag{D.1}$$

The claim is established columnwise: we show $\boldsymbol{Q}\boldsymbol{R}[:,\ell] = \boldsymbol{A}_J[:,\ell]$ for all $1 \le \ell \le n$.

**Case $\ell \le k$.** Let $\boldsymbol{\delta}_\ell^n$ be the $\ell$-th standard basis vector in $\mathbb{R}^n$. Then

$$\boldsymbol{Q}\boldsymbol{R}[:,\ell] = \boldsymbol{Q}\boldsymbol{R}\boldsymbol{\delta}_\ell^n = \boldsymbol{A}_J[:,:k][\boldsymbol{I}_{k \times k},\ (\boldsymbol{R}_1^{\text{sk}})^{-1}\boldsymbol{R}_2^{\text{sk}}]\boldsymbol{\delta}_\ell^n = \boldsymbol{A}_J[:,:k]\boldsymbol{\delta}_\ell^k = \boldsymbol{A}_J[:,\ell].$$

**Case $\ell > k$.** We have

$$\boldsymbol{Q}\boldsymbol{R}[:,\ell] = \boldsymbol{A}_J[:,:k]((\boldsymbol{R}_1^{\text{sk}})^{-1}\boldsymbol{R}_2^{\text{sk}})[:,\ell - k].$$

Let $\boldsymbol{\gamma} = ((\boldsymbol{R}_1^{\text{sk}})^{-1}\boldsymbol{R}_2^{\text{sk}})[:,\ell - k]$. We need to show that $\boldsymbol{A}_J[:,k]\boldsymbol{\gamma} = \boldsymbol{A}_J[:,\ell]$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(D.0.1)</span></p>

If $\boldsymbol{A}_J^{\text{sk}}[:,\ell] = \boldsymbol{A}_J^{\text{sk}}[:,:k]\boldsymbol{u}$ for some $\boldsymbol{u} \in \mathbb{R}^k$, then $\boldsymbol{A}_J[:,\ell] = \boldsymbol{A}_J[:,:k]\boldsymbol{u}$.

</div>

*Proof.* To simplify notation, define $\boldsymbol{X} = \boldsymbol{A}_J[:,:k]$ and $\boldsymbol{y} = \boldsymbol{A}_J[:,\ell]$.

Suppose to the contrary that $\boldsymbol{y} \ne \boldsymbol{X}\boldsymbol{u}$ and $\boldsymbol{S}\boldsymbol{y} = \boldsymbol{S}\boldsymbol{X}\boldsymbol{u}$. Then $\boldsymbol{S}\boldsymbol{X}\boldsymbol{u} - \boldsymbol{S}\boldsymbol{y} = \boldsymbol{0}$. Define $U = \ker(\boldsymbol{S}[\boldsymbol{X},\,\boldsymbol{y}])$ and $V = \ker([\boldsymbol{X},\,\boldsymbol{y}])$. Clearly, $U$ contains $V$. Additionally, if $U$ contains a nonzero vector that is not in $V$, then $\dim(U) > \dim(V)$, which would imply $\operatorname{rank}(\boldsymbol{S}[\boldsymbol{X},\,\boldsymbol{y}]) < \operatorname{rank}([\boldsymbol{X},\,\boldsymbol{y}])$.

If $\boldsymbol{S}\boldsymbol{X}\boldsymbol{u} - \boldsymbol{S}\boldsymbol{y} = \boldsymbol{0}$, then $(\boldsymbol{u}, -1)$ is a nonzero vector in $U$ that is not in $V$. However, by our assumption, the sketch does not drop rank. Consequently, no such vector $(\boldsymbol{u}, -1)$ can exist, and we must have $\boldsymbol{y} = \boldsymbol{X}\boldsymbol{u}$. $\square$

We now prove that $\boldsymbol{A}_J^{\text{sk}}[:,:k]\boldsymbol{\gamma} = \boldsymbol{A}_J^{\text{sk}}[:,\ell]$. Start by noting that $\boldsymbol{A}_J^{\text{sk}}[:,:k] = \boldsymbol{Q}^{\text{sk}}\boldsymbol{R}_1^{\text{sk}}$. Plugging in the definition of $\boldsymbol{\gamma}$:

$$\boldsymbol{A}_J^{\text{sk}}[:,:k]\boldsymbol{\gamma} = \boldsymbol{Q}^{\text{sk}}\boldsymbol{R}_1^{\text{sk}}(\boldsymbol{R}_1^{\text{sk}})^{-1}(\boldsymbol{R}_2^{\text{sk}})[:,\ell - k] = \boldsymbol{Q}^{\text{sk}}(\boldsymbol{R}_2^{\text{sk}})[:,\ell - k].$$

Using $\boldsymbol{R}_2^{\text{sk}}[:,\ell - k] = \boldsymbol{R}^{\text{sk}}[:,\ell]$:

$$\boldsymbol{A}_J^{\text{sk}}[:,:k]\boldsymbol{\gamma} = (\boldsymbol{Q}^{\text{sk}}\boldsymbol{R}^{\text{sk}})[:,\ell] = \boldsymbol{A}_J^{\text{sk}}[:,\ell].$$

Combining Proposition D.0.1 with the above proves the desired result $\boldsymbol{A}[:,J] = \boldsymbol{Q}\boldsymbol{R}$, which is Proposition 5.1.2.

# Appendix E: Bootstrap Methods for Error Estimation

Whenever a randomized algorithm produces a solution, a question immediately arises: is the solution sufficiently accurate? In many situations, it is possible to estimate numerically the error of the solution using the available problem data — a process often referred to as *(a posteriori) error estimation*. In addition to resolving uncertainty about the quality of a solution, error estimation enables computations to be done more adaptively. For instance, error estimates can determine if additional iterations should be performed, or if tuning parameters should be modified. In this way, error estimates can help to incrementally refine a rough initial solution so that "just enough" work is done to reach a desired level of accuracy.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Priori vs. A Posteriori)</span></p>

*A posteriori* error estimation should be contrasted with *a priori error bounds* often used in theoretical development of RandNLA algorithms, in which one bounds rather than estimates the error, and does so in a worst-case way that does not depend on the problem data.

</div>

## E.1 Bootstrap Methods in a Nutshell

Bootstrap methods have been studied extensively in the statistics literature for more than four decades, and they comprise a very general framework for quantifying uncertainty. One of the most common uses in statistics is to assess the accuracy of parameter estimates. This use-case provides the connection between bootstrap methods and error estimation in RandNLA: an exact solution to a linear algebra problem can be viewed as an "unknown parameter," and a randomized algorithm can be viewed as providing an "estimate" of that parameter. A random sketch of a matrix can also be viewed as a "dataset" from which the estimate of the "population" quantity is computed. When bootstrap methods are applied in RandNLA, the rows or columns of a sketched matrix often play the role of "data vectors."

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bootstrap Error Estimation Framework)</span></p>

Suppose the existence of some fixed but unknown "true parameter" $\theta \in \mathbb{R}$. We estimate this parameter by a value $\hat{\theta}$ depending on random samples from some probability distribution. The error of $\hat{\theta}$ is defined as $\hat{\epsilon} = |\hat{\theta} - \theta|$, which is both random and unknown.

It is natural to seek the tightest upper bound on $\hat{\epsilon}$ that holds with a specified probability $1 - \alpha$. This ideal bound is known as the $(1-\alpha)$-quantile of $\hat{\epsilon}$:

$$q_{1-\alpha} = \inf\lbrace t \in [0, \infty) \mid \mathbb{P}(\hat{\epsilon} \le t) \ge 1 - \alpha \rbrace.$$

An error estimation problem is considered solved if it is possible to construct a quantile estimate $\hat{q}_{1-\alpha}$ such that the inequality $\hat{\epsilon} \le \hat{q}_{1-\alpha}$ holds with probability close to $1 - \alpha$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bootstrap Sampling Procedure)</span></p>

The bootstrap approach to estimating $q_{1-\alpha}$ is based on imagining a scenario where it is possible to generate many independent samples $\hat{\epsilon}_1, \ldots, \hat{\epsilon}_N$ of the random variable $\hat{\epsilon}$. The key idea is to generate "approximate samples" of $\hat{\epsilon}$, which *can* be done in practice.

Consider a generic situation where $\hat{\theta} = f(X_1, \ldots, X_n)$ for some function $f$. A **bootstrap sample** of $\hat{\epsilon}$, denoted $\mathring{\hat{\epsilon}}$, is computed as follows:

- Sample $n$ points $\lbrace \hat{X}_i \rbrace_{i=1}^n$ with replacement from the original dataset $\lbrace X_i \rbrace_{i=1}^n$.
- Compute $\mathring{\hat{\theta}} := f(\hat{X}_1, \ldots, \hat{X}_n)$.
- Compute $\mathring{\hat{\epsilon}} := |\mathring{\hat{\theta}} - \hat{\theta}|$.

By performing $N$ independent iterations, a collection of bootstrap samples $\mathring{\hat{\epsilon}}_1, \ldots, \mathring{\hat{\epsilon}}_N$ can be generated. The desired quantile estimate $\hat{q}_{1-\alpha}$ is the smallest number $t \ge 0$ for which

$$\frac{1}{N}\sum_{i=1}^N \mathbb{I}\lbrace \mathring{\hat{\epsilon}}_i \le t \rbrace \ge 1 - \alpha$$

where $\mathbb{I}\lbrace \cdot \rbrace$ is the $\lbrace 0,1 \rbrace$-valued indicator function. This quantity is also known as the empirical $(1-\alpha)$-quantile.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition for the Bootstrap)</span></p>

The random variable $\mathring{\hat{\theta}}$ can be viewed as a "perturbed version" of $\hat{\theta}$, where the perturbing mechanism is designed so that the deviations of $\mathring{\hat{\theta}}$ around $\hat{\theta}$ are statistically similar to the deviations of $\hat{\theta}$ around $\theta$. Equivalently, this means that the histogram of $\mathring{\hat{\epsilon}}_1, \ldots, \mathring{\hat{\epsilon}}_N$ will serve as a good approximation to the distribution of the actual random error variable $\hat{\epsilon}$. Furthermore, it turns out that this approximation is asymptotically valid (i.e., $n \to \infty$) and supported by quantitative guarantees in a broad range of situations.

</div>

## E.2 Sketch-and-Solve Least Squares

There is a direct analogy between the general bootstrap discussion and the sketch-and-solve setting for least squares. The "true parameter" $\theta$ is the exact solution $\boldsymbol{x}_\star = \operatorname{argmin}_{\boldsymbol{x} \in \mathbb{R}^n} \|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2$. The dataset $X_1, \ldots, X_n$ corresponds to the sketches $[\hat{\boldsymbol{A}}, \hat{\boldsymbol{b}}] = \boldsymbol{S}[\boldsymbol{A}, \boldsymbol{b}]$. The estimate $\hat{\theta}$ corresponds to the sketch-and-solve solution $\hat{\boldsymbol{x}} = \operatorname{argmin}_{\boldsymbol{x} \in \mathbb{R}^n} \|\hat{\boldsymbol{A}}\boldsymbol{x} - \hat{\boldsymbol{b}}\|_2^2$. The error variable can be defined as $\hat{\epsilon} = \rho(\hat{\boldsymbol{x}}, \boldsymbol{x}_\star)$, for a preferred metric $\rho$, such as that induced by the $\ell_2$ or $\ell_\infty$ norms.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Method 1 — Bootstrap Error Estimation for Sketch-and-Solve Least Squares)</span></p>

**Input:** A positive integer $B$, the sketches $\hat{\boldsymbol{A}} \in \mathbb{R}^{d \times n}$, $\hat{\boldsymbol{b}} \in \mathbb{R}^d$, and $\hat{\boldsymbol{x}} \in \mathbb{R}^n$.

**For** $\ell \in [\![B]\!]$ **do in parallel:**

1. Draw a vector $I := (i_1, \ldots, i_d)$ by sampling $d$ numbers with replacement from $[\![d]\!]$.
2. Form the matrix $\mathring{\hat{\boldsymbol{A}}} := \hat{\boldsymbol{A}}[I,:]$ and vector $\mathring{\hat{\boldsymbol{b}}} := \hat{\boldsymbol{b}}[I]$.
3. Compute

$$\mathring{\hat{\boldsymbol{x}}} := \operatorname{argmin}_{\boldsymbol{x} \in \mathbb{R}^n} \|\mathring{\hat{\boldsymbol{A}}}\boldsymbol{x} - \mathring{\hat{\boldsymbol{b}}}\|_2 \qquad \text{and} \qquad \mathring{\hat{\epsilon}}_\ell := \|\mathring{\hat{\boldsymbol{x}}} - \hat{\boldsymbol{x}}\|.$$

**Return:** The estimate $\text{quantile}[\mathring{\hat{\epsilon}}_1, \ldots, \mathring{\hat{\epsilon}}_B;\, 1-\alpha]$ for the $(1-\alpha)$-quantile of $\|\hat{\boldsymbol{x}} - \boldsymbol{x}_\star\|$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Computational Characteristics of Method 1)</span></p>

The for loop can be implemented in an embarrassingly parallel manner, which is typical of most bootstrap methods. The method only relies on access to sketched quantities, and hence does not require any access to the full matrix $\boldsymbol{A}$. Likewise, the computational cost of the method is independent of the number of rows of $\boldsymbol{A}$.

</div>

## E.3 Sketch-and-Solve One-Sided SVD

We call the problem of computing the singular values and right singular vectors of a matrix a "one-sided SVD." We use the term "sketch-and-solve one-sided SVD" for an algorithm that approximates the top $k$ singular values and singular vectors of $\boldsymbol{A}$ by those of a sketch $\hat{\boldsymbol{A}} = \boldsymbol{S}\boldsymbol{A}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Error Variables for One-Sided SVD)</span></p>

Let $\lbrace (\sigma_j, \boldsymbol{v}_j) \rbrace_{j=1}^k$ denote the top $k$ singular values and right singular vectors of $\boldsymbol{A}$, and $\lbrace (\hat{\sigma}_j, \hat{\boldsymbol{v}}_j) \rbrace_{j=1}^k$ the corresponding quantities for $\hat{\boldsymbol{A}}$. Supposing that error is measured uniformly over $j \in [\![k]\!]$, the error variables are

$$\epsilon_\Sigma := \max_{j \in [\![k]\!]} |\hat{\sigma}_j - \sigma_j| \qquad \text{and} \qquad \epsilon_V := \max_{j \in [\![k]\!]} \rho(\hat{\boldsymbol{v}}_j, \boldsymbol{v}_j).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Method 2 — Bootstrap Error Estimation for Sketch-and-Solve SVD)</span></p>

**Input:** The sketch $\hat{\boldsymbol{A}} \in \mathbb{R}^{d \times n}$ and its top $k$ singular values and right singular vectors $(\hat{\sigma}_1, \hat{\boldsymbol{v}}_1), \ldots, (\hat{\sigma}_k, \hat{\boldsymbol{v}}_k)$, a number of samples $B$, a parameter $\alpha \in (0, 1)$.

**For** $\ell \in [\![B]\!]$ **do in parallel:**

1. Form $\mathring{\hat{\boldsymbol{A}}} \in \mathbb{R}^{d \times n}$ by sampling $d$ rows from $\hat{\boldsymbol{A}}$ with replacement.
2. Compute the top $k$ singular values and right singular vectors of $\mathring{\hat{\boldsymbol{A}}}$, denoted $\mathring{\hat{\sigma}}_1, \ldots, \mathring{\hat{\sigma}}_k$ and $\mathring{\hat{\boldsymbol{v}}}_1, \ldots, \mathring{\hat{\boldsymbol{v}}}_k$. Then compute the bootstrap samples:

$$\mathring{\hat{\epsilon}}_{\Sigma,\ell} := \max_{j \in [\![k]\!]} |\mathring{\hat{\sigma}}_j - \hat{\sigma}_j|$$

$$\mathring{\hat{\epsilon}}_{V,\ell} := \max_{j \in [\![k]\!]} \rho(\mathring{\hat{\boldsymbol{v}}}_j, \hat{\boldsymbol{v}}_j).$$

**Return:** The estimates $\text{quantile}[\mathring{\hat{\epsilon}}_{\Sigma,1}, \ldots, \mathring{\hat{\epsilon}}_{\Sigma,B};\, 1-\alpha]$ and $\text{quantile}[\mathring{\hat{\epsilon}}_{V,1}, \ldots, \mathring{\hat{\epsilon}}_{V,B};\, 1-\alpha]$ for the $(1-\alpha)$-quantiles of $\epsilon_\Sigma$ and $\epsilon_V$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Extensions of Method 2)</span></p>

Although Method 2 is only presented with regard to singular values and right singular vectors, it is also possible to apply a variant of it to estimate the errors of approximate left singular vectors.

Another technique to estimate error in the setting of sketch-and-solve one-sided SVD is through the spectral norm $\|\hat{\boldsymbol{A}}^*\hat{\boldsymbol{A}} - \boldsymbol{A}^*\boldsymbol{A}\|_2$. Due to the Weyl and Davis-Kahan inequalities, an upper bound on $\|\hat{\boldsymbol{A}}^*\hat{\boldsymbol{A}} - \boldsymbol{A}^*\boldsymbol{A}\|_2$ directly implies upper bounds on the errors of all the sketched singular values $\hat{\sigma}_1, \ldots, \hat{\sigma}_n$ and sketched right singular vectors $\hat{\boldsymbol{v}}_1, \ldots, \hat{\boldsymbol{v}}_n$. The quantiles of the error variable $\|\hat{\boldsymbol{A}}^*\hat{\boldsymbol{A}} - \boldsymbol{A}^*\boldsymbol{A}\|_2$ can also be estimated via the bootstrap.

</div>