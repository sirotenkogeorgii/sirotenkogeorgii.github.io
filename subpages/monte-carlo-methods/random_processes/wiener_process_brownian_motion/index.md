---
title: Wiener Process and Brownian Motion
layout: default
noindex: true
---

# Wiener Process and Brownian Motion

$\textbf{Definition (Wiener process):}$ A **Wiener process** is a stochastic process $W = \rbrace W_t, t \ge 0\lbrace$ defined by these properties:

1. **Independent increments:**
   For any t_1$ < t_2 \le t_3 < t_4$, the increments

   $$W_{t_4} - W_{t_3} \quad \text{and} \quad W_{t_2} - W_{t_1}$$

   are independent. Equivalently, for $t > s$, the increment $W_t - W_s$ does not depend on the past trajectory $\lbrace W_u, 0 \le u \le s\rbrace$.

2. **Gaussian stationary increments:**
   For all $t \ge s \ge 0$,
   
   $$W_t - W_s \sim \mathcal{N}(0, t-s)$$

   Variance of the difference increases linearly. 
   
3. **Continuous paths:**
   The sample paths of $\lbrace W_t\rbrace$ are continuous, and $W_0 = 0$.

> **Remark**: The Wiener process is fundamental in probability theory and serves as the building block for many other stochastic processes. It can be interpreted as the continuous-time version of a random walk. Two example paths over ([0,1]) are shown in Figure 4.9.

<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/wiener_process_on_interval.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Two example paths over the interval [0,1]</figcaption>
  </figure>
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/wiener_process_with_different_sigma.png' | relative_url }}" alt="b" loading="lazy">
    <figcaption>Wiener process with different sigmas</figcaption>
  </figure>
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/wiener_process_3d.png' | relative_url }}" alt="c" loading="lazy">
    <figcaption>3D Wiener process</figcaption>
  </figure>
</div>

A Wiener process could be viewed as a homogeneous Gaussian process $X(t)$ with independent increments. Also Wiener process serves as one of the models of **Brownian motion**.

> **Remark**: A **homogeneous Gaussian process** means a stationary (translation-invariant) Gaussian process — i.e., shifting the input space doesn’t change its probabilistic behavior. "Stationary" means the probabilistic behavior of a process doesn’t change when you shift time/space.

More details on homogeneous GP:

<div class="accordion">
  <details markdown="1">
    <summary>homogeneous Gaussian process</summary>

A **homogeneous Gaussian process** usually means a **stationary (translation-invariant) Gaussian process** — i.e., shifting the input space doesn’t change its probabilistic behavior.

Formally, a (real-valued) Gaussian process $\lbrace X(t): t \in \mathbb{R}^d\rbrace$ is **homogeneous** if:

1. **Constant mean**
   
   $$\mathbb{E}[X(t)] = m \quad \text{for all } t$$

2. **Covariance depends only on displacement**
   
   $$\mathrm{Cov}(X(s), X(t)) = C(t-s)$$
   
   So the covariance is a function of the **difference** $h=t-s$, not of the absolute locations.

Because it’s Gaussian, specifying the mean and covariance fully determines all finite-dimensional distributions.
  </details>
</div>

The key consequences for $W = \lbrace W_t, t \ge 0\rbrace$ include:

1. **Gaussian process characterization:**
   $W$ is a Gaussian process with
   
   $$\mathbb{E}[W_t] = 0, \qquad \text{Cov}(W_s, W_t) = \min{s,t}$$
   
   It is the unique Gaussian process with continuous paths having these mean and covariance properties.

2. **Markov property:**
   $W$ is a time-homogeneous strong Markov process. In particular, for any finite stopping time $\tau$ and any $t \ge 0$,
   
   $$(W_{\tau+t}\mid W_u, u \le \tau) \sim (W_{\tau+t}\mid W_\tau)$$
   
    That line is basically saying: once you know the value at time $τ$, the whole past before $τ$ gives you no extra information about the future after $τ$.

The simulation method below relies on the **Markov** and **Gaussian** properties of the Wiener process.

---
### Algorithm (Simulating a Wiener Process)

1. Choose a set of distinct time points
   
   $$0 = t_0 < t_1 < t_2 < \cdots < t_n$$
   
   at which you want to simulate the process.

2. Generate independent standard normal variables
   
   $$Z_1,\ldots,Z_n \stackrel{iid}{\sim} \mathcal{N}(0,1),$$
   
   and compute
   
   $$W_{t_k} = \sum_{i=1}^k \sqrt{t_i - t_{i-1}}, Z_i, \quad k = 1,\ldots,n$$
---

This method is **exact** in the sense that the values $\lbrace W_{t_k}\rbrace$ are sampled from the correct distributions. However, it only produces a **discrete set of points** from the underlying continuous-time process.

To build a continuous approximation of the path, one can use **linear interpolation** between successive simulated points. Specifically, on each interval $[t_{k-1}, t_k]$, approximate $\lbrace W_s\rbrace$ by

$$\widehat{W}_s = \frac{W_{t_k}(s - t_{k-1}) + W_{t_{k-1}}(t_k - s)} {t_k - t_{k-1}}, \qquad s \in [t_{k-1}, t_k].$$


The path can also be refined adaptively using a **Brownian bridge**.

A process $\lbrace B_t, t \ge 0\rbrace$ defined by

$$B_t = \mu t + \sigma W_t, \qquad t \ge 0,$$

where $\lbrace W_t\rbrace$ is a Wiener process, is called a **Brownian motion** with **drift** $\mu$ and **diffusion coefficient** $\sigma^2$.
When $\mu = 0$ and $\sigma^2 = 1$, this reduces to **standard Brownian motion** (i.e., the Wiener process).

The simulation of Brownian motion at times $t_1,\ldots,t_n$ follows directly from this definition.

---
### Algorithm (Simulating Brownian Motion)

1. First generate $W_{t_1},\ldots,W_{t_n}$ from a Wiener process.

2. Then set
   
   $$B_{t_i} = \mu t_i + \sigma W_{t_i}, \quad i = 1,\ldots,n.$$
---

If $\lbrace W_{t,i}, t \ge 0\rbrace$ for $i = 1,\ldots,n$ are independent Wiener processes and

$$W_t = (W_{t,1},\ldots,W_{t,n}),$$

then $\lbrace W_t, t \ge 0\rbrace$ is called an **$n$-dimensional Wiener process**.

