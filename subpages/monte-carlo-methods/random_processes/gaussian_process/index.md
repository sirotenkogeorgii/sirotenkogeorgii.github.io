---
title: Gaussian Processes
layout: default
noindex: true
---

# Gaussian Processes

## Basic Gaussian Processes

A **Gaussian Process (GP)** is a way to define a probability distribution over **functions**—so instead of putting a prior on parameters, you put a prior on the *entire shape* of a function.

More formally: a stochastic process $f(x)$ (or $\{f(x_t)=\tilde{X}_t, t\in \mathcal{T}\}$) is a Gaussian Process if for any finite set of inputs $x_1,\dots,x_n$, the vector of function values $\{\tilde{X_t}, t\in \mathcal{T}\}$

$$[f(x_1), \dots, f(x_n)]$$

is jointly Gaussian. Or equivalently, if any linear combination $\sum_{i=1}^n \alpha_i f(x_i)=\sum_{i=1}^n \alpha_i \tilde{X}_i$ is Gaussian.

A GP is fully specified by:

* a **mean function** $m(x_t)=\tilde{\mu}(x_t) = \mathbb{E}[f(x_t)]$ defined on the index set $\mathcal{T}$.
* a **covariance (kernel) function** $k(x_t,x_{t'})= \tilde{\Sigma}\_{t,t'} = \text{Cov}(f(x_t), f(x_{t'}))$ defined on the index set $\mathcal{T} \times \mathcal{T}$.

We write:

$$f(x) \sim \mathcal{GP}(m(x), k(x,x'))$$

* You believe the true function is smooth / periodic / rough / etc.
* You encode that belief with a **kernel**.
* After seeing data, you update that belief to get a **posterior distribution over functions**.

So a GP gives:

* a **prediction**
* and a **principled uncertainty estimate** at every $x$

<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/GP1.png' | relative_url }}" alt="GP1" loading="lazy">
  </figure>
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/GP2.png' | relative_url }}" alt="GP2" loading="lazy">
  </figure>
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/GP3.png' | relative_url }}" alt="GP3" loading="lazy">
  </figure>
</div>

#### Stationarity

For general stochastic processes strict-sense stationarity implies wide-sense stationarity but not every wide-sense stationary stochastic process is strict-sense stationary. However, for a Gaussian stochastic process the two concepts are equivalent. 

$\textbf{Property (Stationarity):}$ A Gaussian stochastic process is strict-sense stationary if and only $\iff$ wide-sense stationary.

#### Example

There is an **explicit** representation for stationary Gaussian processes. A simple example of this representation is

$$X_{t}=\cos(at)\xi_{1}+\sin(at)\xi_{2}$$

are independent random variables with the standard normal distribution.

#### Example: Gaussian Process Regression (the classic use)

Given noisy observations:

$$y = f(x) + \epsilon,\quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

the posterior predictive distribution at a new point (x_*) is Gaussian:

$$p(f_* \mid x_*, X, y) = \mathcal{N}(\mu_*, \sigma_*^2)$$

where $\mu_*$ and $\sigma_*^2$ come from kernel-based linear algebra.

#### Why kernels matter

The kernel determines the shape properties of sampled functions:

* **RBF (squared exponential):** very smooth functions
* **Matérn:** controlled roughness (often more realistic)
* **Periodic:** repeating structure
* **Linear:** equivalent to Bayesian linear regression

### Strengths

* Excellent **uncertainty calibration**
* Works great with **small-to-medium data**
* Very flexible via kernel design
* Naturally Bayesian

---
### Algorithm (Gaussian Process Generator)
1. Form the mean vector $\mu = (\mu_1,\dots,\mu_n)^\top$ and covariance matrix $\Sigma = (\Sigma_{ij})$ by setting $\mu_i = \tilde\mu_{t_i}$ and $\Sigma_{ij} = \tilde\Sigma_{t_i,t_j}$.
2. Compute the Cholesky factorization $\Sigma = AA^\top$.
3. Sample independent standard normal variables $Z_1,\dots,Z_n \sim \mathcal N(0,1)$, and define $Z = (Z_1,\dots,Z_n)^\top$.
4. Produce the sample $X = \mu + AZ$.

---

<figure>
  <img src="{{ 'assets/images/notes/monte-carlo-methods/GP_sketch.jpg' | relative_url }}" alt="GP sketch" loading="lazy">
</figure>

**Magic behind Cholsky factorization in this context:**
<div class="accordion">
  <details markdown="1">
    <summary>Magic behind Cholsky factorization in this context</summary>

### Start with independent Gaussians

$$Z \sim \mathcal{N}(0, I)$$

This means:

* each $Z_i$ is standard normal
* and they’re independent:
  
  $$\text{Cov}(Z_i, Z_j) = 0 \quad (i \neq j).$$
  

### We want correlated outputs

We want

$$X \sim \mathcal{N}(\mu, \Sigma).$$


So we look for a matrix $A$ such that

$$\Sigma = A A^\top$$


Cholesky gives exactly that (with a nice lower triangular $A$).

### The transformation

Define:

$$X = \mu + A Z$$


### What happens to correlations?

Look at the covariance:

$$\text{Cov}(X) = \text{Cov}(A Z)$$


Pull out the constant matrix:

$$\text{Cov}(A Z) = A \text{Cov}(Z) A^\top$$


But $\text{Cov}(Z) = I$, so:

$$\text{Cov}(X) = A I A^\top = A A^\top = \Sigma$$


That’s the whole magic.
**The matrix $A$ is chosen specifically so that this equality holds.**

**One-line takeaway:**
Cholesky gives a matrix $A$ that acts like a covariance “square root,” so when you apply it to independent Gaussian noise, the output inherits exactly the covariance (and thus correlations) you want.

  </details>
</div>
