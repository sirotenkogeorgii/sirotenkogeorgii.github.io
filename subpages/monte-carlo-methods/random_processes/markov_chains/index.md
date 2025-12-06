---
title: Markov Chains
layout: default
noindex: true
---

# Markov Chains

In probability theory and statistics, a **Markov chain** or **Markov process** is a stochastic process describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event. Two famous classes of Markov process are the **Markov chain** and **Brownian motion**.


Formally, A **Markov chain** is a stochastic process $\lbrace X_t, t \in \mathcal T\rbrace$ indexed by a *countable* set $\mathcal T \subset \mathbb{R}$ that satisfies the **Markov property**:

$$(X_{t+s}\mid X_u,, u \le t) \sim (X_{t+s}\mid X_t).$$

That is, the future depends on the past only through the present state.

Here we focus only on the main ideas relevant to simulation (#TODO add more stuff). Throughout, we assume the index set is $\mathcal T = \lbrace 0,1,2,\dots\rbrace$.

A direct implication of the Markov property is that the chain can be simulated **step by step** (i.e., sequentially) as $X_0, X_1, \ldots$, using the following general procedure.

---
### Algorithm (Generating a Markov Chain)

1. Sample $X_0$ from its initial distribution. Set $t=0$.
2. Sample $X_{t+1}$ from the conditional distribution of $X_{t+1}$ given $X_t$.
3. Increase $t$ by 1 and repeat Step 2.

---

<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/Markov_chain_sketch.jpg' | relative_url }}" alt="Markov chain sketch" loading="lazy">
  </figure>
</div>

The conditional distribution of $X_{t+1}$ given $X_t$ is commonly specified in one of two ways:

* **Via a recurrence form:**
  
  $$X_{t+1} = g(t, X_t, U_t), \quad t=0,1,2,\dots$$
  
  where $g$ is easy to compute, and $U_t$ is a random variable that is easy to generate, *possibly* with a distribution depending on $X_t$ and $t$.

* **Via an explicit conditional distribution P(X_{t+1} \mid X_t)** that is known and straightforward to sample from.

An important example of the second case arises when the chain $\lbrace X_0,X_1,\dots\rbrace$ has a **discrete state space** $E$ and is **time-homogeneous** (a system or process's rules (like transition probabilities) don't change over time). In this setting, the process is fully determined by:

* the distribution of $X_0$, and
* the **one-step transition matrix** $P = (p_{ij})$, where
  
  $$p_{ij} = \mathbb P(X_{t+1}=j \mid X_t=i), \quad i,j \in E$$

Thus, if $X_n=i$, the distribution of $X_{n+1}$ is the discrete distribution given by the $i$-th row of $P$. This gives a more specific simulation method.

---
### Algorithm 4.5 (Generating a Time-Homogeneous Markov Chain on a Discrete State Space)

1. Sample $X_0$ from the initial distribution. Set $t=0$.
2. Sample $X_{t+1}$ from the discrete distribution given by the $X_t$-th row of $P$.
3. Set $t=t+1$ and repeat Step 2.

---

## Random Walks on an $n$-Dimensional Hypercube


A typical example of a Markov chain that is specified by a recurrence relation is the **random walk process**.

$$X_{t+1} = X_t + U_t,\quad t=1,2,\dots$$

where $U_1,U_2,\ldots$ is an iid sequence of random variables from some discrete or continuous distribution.

**State space:** vertices of the unit $n$-hypercube $E = \lbrace 0,1\rbrace^n$ (i.e., binary vectors of length $n$).

Let $e_1,\dots,e_n$ be the standard basis vectors in $\mathbb{R}^n$. Start with any $X_0 \in \lbrace 0,1\rbrace^n$.

**Transition rule:**

$$X_{t+1} = X_t + e_{I_t} \pmod{2}$$

with $$I_t \stackrel{\text{iid}}{\sim} \text{Uniform}\lbrace 1,\dots,n\rbrace$$

**Interpretation:**
At each step, choose a coordinate uniformly at random and **flip that bit**.

**Consequence:**
The chain moves to one of the $n$ adjacent vertices with equal probability.

<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/random_walk.jpg' | relative_url }}" alt="Random Walk on a 3-Dimensional Hypercube" loading="lazy">
    <figcaption>Random Walk on a 3-Dimensional Hypercube</figcaption>
  </figure>
</div>

### TODO: Add PAS2 source

### TODO: Add Wiki source