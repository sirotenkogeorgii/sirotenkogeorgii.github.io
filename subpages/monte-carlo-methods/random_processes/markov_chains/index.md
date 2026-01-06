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

<figure>
  <img src="{{ 'assets/images/notes/monte-carlo-methods/Markov_chain_sketch.jpg' | relative_url }}" alt="Markov chain sketch" loading="lazy">
</figure>

The conditional distribution of $X_{t+1}$ given $X_t$ is commonly specified in one of two ways:

* **Via a recurrence form:**
  
  $$X_{t+1} = g(t, X_t, U_t), \quad t=0,1,2,\dots$$
  
  where $g$ is easy to compute, and $U_t$ is a random variable that is easy to generate, *possibly* with a distribution depending on $X_t$ and $t$.

* **Via an explicit conditional distribution $P(X_{t+1} \mid X_t)$** that is known and straightforward to sample from.

An important example of the second case arises when the chain $\lbrace X_0,X_1,\dots\rbrace$ has a **discrete state space** $E$ and is **time-homogeneous** (a system or process's rules (like transition probabilities) don't change over time). In this setting, the process is fully determined by:

* the distribution of $X_0$, and
* the **one-step transition matrix** $P = (p_{ij})$ (if the number of states is finite), where
  
  $$p_{ij} = \mathbb P(X_{t+1}=j \mid X_t=i), \quad i,j \in E$$

Thus, if $X_n=i$, the distribution of $X_{n+1}$ is the discrete distribution given by the $i$-th row of $P$. This gives a more specific simulation method.

---
### Algorithm 4.5 (Generating a Time-Homogeneous Markov Chain on a Discrete State Space)

1. Sample $X_0$ from the initial distribution. Set $t=0$.
2. Sample $X_{t+1}$ from the discrete distribution given by the $X_t$-th row of $P$.
3. Set $t=t+1$ and repeat Step 2.

---

### Stationary Distribution

A probability mass function $𝜋 ∶ Γ → [0, 1]$ is called a **stationary distribution** if 

$$\pi(\omega_k) = \sum_{j=1}^M \pi(w_j)P_{jk},$$

or equivalently the row-vector $𝜋 = (𝜋(𝜔_1), … , 𝜋(𝜔_𝑀))$ is a right-eigenvector of $𝑃$ with eigenvalue $1$, i.e. $𝜋 = 𝜋𝑃$.

We can make two observations: 

* **Limit Behavior:**
If the sequence of distributions $\mathbf{p}_n = \mathbf{p}_0 P^n$ converges as $n \to \infty$,  the resulting limit $\pi$ is stationary. This is shown by taking the limit of the evolution equation:

$$\pi(\omega_k) = \lim_{n\to\infty} \sum_{j=1}^{M} \mathbf{p}_{n-1}(\omega_j) P_{jk} = \sum_{j=1}^{M} \pi(\omega_j) P_{jk}$$

* **Spectral Properties & Existence:**
  * Since $P \mathbf{1} = \mathbf{1}$ (where $\mathbf{1}$ is a column vector of ones), $P$ has an eigenvalue of 1.
  * All other eigenvalues satisfy $\lvert\lambda\rvert \le 1$.
  * Since $P$ contains only non-negative entries, the **Perron-Frobenius Theorem** applies. It states that for the eigenvalue 1, there exists a right-eigenvector with non-negative entries. Normalizing this vector yields a stationary probability distribution.

**Code example:**
<div class="accordion">
  <details markdown="1">
    <summary>Code example</summary>

```python
def stationary_distributions(P):
    eigenvalues, eigenvectors = np.linalg.eig(np.transpose(P))
    return [eigenvectors[:,i]/np.sum(eigenvectors[:,i]) for i in range(len(eigenvalues)) if np.abs(eigenvalues[i]-1) < 1e-10]

print("Eigenvalues: ",np.linalg.eig(transition_P)[0])
for pi in stationary_distributions(transition_P):
    print("Stationary distribution: ",pi)
```

  </details>
</div>

#### Irreducible Transition Matrix

<figure>
  <img src="{{ 'assets/images/notes/monte-carlo-methods/reducible-matrix.jpg' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Reducible Matrix</figcaption>
</figure>

> An irreducible matrix is a square matrix that cannot be rearranged (by permuting rows and columns) into a block upper triangular form, meaning it can't be split into smaller, independent matrix blocks along the diagonal, which signifies a "connectedness" in its underlying graph representation where every node can reach every other node. This concept is crucial in Perron-Frobenius theory, Markov chains, and graph theory for understanding system dynamics and connectivity. 
>
> Key Characteristics:
> * **No Block Decomposition**: A matrix is reducible if it can be permuted into [[A, B], [0, C]] form (where A, C are square, and B is some matrix), but an irreducible one cannot.
> * **Graph Connectivity**: Its associated directed graph (digraph) is strongly connected, meaning there's a path between any two nodes (vertices).
> * **Permutation Similarity**: It's not similar to any block-triangular matrix (except trivial ones with one block). 
>
> **Why it Matters**:
> * Perron-Frobenius Theorem: Guarantees unique positive eigenvalues and related properties for irreducible, non-negative matrices.
> * Markov Chains: Irreducible matrices represent systems where all states are mutually reachable, leading to stable long-term behavior (stationary distributions).
> * Computational Efficiency: Identifying reducibility helps simplify problems by breaking them into smaller, solvable parts, as reducible matrix problems are easier. 

<figure>
  <img src="{{ 'assets/images/notes/monte-carlo-methods/non-unique-stationary-distribution.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Non-unique stationary distribution</figcaption>
</figure>

Example above has two linearly independent stationary distribution: 
* Stationary distribution 1: [0.5 0.5 0. 0. ]
* Stationary distribution 2: [0. 0. 0.5 0.5 ]

It happens due to the fact that the state space consists of two components that have no transitions among each other. To prevent this phenomenon from happening it is useful
to consider the property of irreducibility. 

The transition matrix $𝑃$ is **irreducible** if for every pair of states $𝑥, 𝑦 ∈ Γ$ there is a positive chance for the Markov chain started at $𝑥$ to eventually reach $𝑦$. 

In the pictorial representation as a directed graph this property is equivalent to the graph being **strongly connected**, meaning that every vertex is reachable along an oriented path from every other vertex. It can be shown that an irreducible transition matrix $𝑃$ always possesses a unique invariant distribution.

Does this imply that every irreducible Markov chain approaches the invariant distribution? Almost, but not quite! There is one more thing that can go wrong: the Markov chain can be **periodic** with period $𝑡 = 2, 3, …$, meaning that there exists a state $𝑥 ∈ Γ$ such that the Markov chain can only return to $𝑥$ after a number a transitions that is a multiple of $𝑡$. It is not difficult to convince yourself that if the Markov chain is irreducible and this holds for one state $𝑥$, then it holds for every state in $Γ$. An example of a periodic Markov chain with period 2 is the following.

<figure>
  <img src="{{ 'assets/images/notes/monte-carlo-methods/periodic-markov-chain.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Periodic Markov Chain</figcaption>
</figure>

A Markov chain cannot converge to a single steady state if it is **periodic**.

Eigenvalues: [-1.0 1.0 0. 0.]

* Consider a chain that strictly alternates between state set $A=\lbrace 0,1\rbrace$ and state set $B=\lbrace 2,3\rbrace$.
* If you start in set $A$ (at $t=0$), you will be in set $A$ at all even times ($t=2n$) and set $B$ at all odd times ($t=2n+1$).
* The distribution of $𝑋_𝑛$ cannot converge as $𝑛 → ∞$, because $ℙ(𝑋_{2𝑛} ∈ \lbrace 0, 1\rbrace) = 1$ while $ℙ(𝑋_{2𝑛+1} ∈ \lbrace 0, 1\rbrace) = 0$ for all $𝑛$ and probability oscillates between 0 and 1 rather than settling - the distribution does not converge.

* **Spectral Interpretation:** In linear algebra terms, periodicity means there are multiple eigenvalues located on the unit circle in the complex plane.
* **Aperiodicity in Practice:** Note that in practice it is not a particularly strong assumption on an irreducible Markov chain: $P_{ii} > 0$ for some state $𝜔_𝑖$ already guarantees the Markov chain is aperiodic. It means if an irreducible chain has even a single state with a "self-loop" (a probability $P_{ii} > 0$ of staying in the same state), the entire chain is guaranteed to be aperiodic.

##### Convergence Analysis (Spectral Decomposition)

If a transition matrix $P$ is both **irreducible** and **aperiodic**, then its eigenvalues behave in a specific way that guarantees convergence:
1. There is a **unique** right-eigenvector with eigenvalue $\lambda_1 = 1$. This corresponds to the stationary distribution, $\pi$.
2. All other eigenvalues have an absolute value strictly less than 1 ($\lvert\lambda_i\rvert < 1$).

**The Proof Logic:**
Assuming $P$ is diagonalizable (which is true for chains satisfying detailed balance), we can express the starting distribution $\mathbf{p}_0$ as a sum of eigenvectors: $\mathbf{p}_0=\sum_i c_i v_i$. eigenvalues $\lambda_i$ are eigenvectors and, where $\lambda_1 = 1$ and $v_1 = \pi$ is the stationary distribution:

$$\lim_{n\to \infty} p_0 P^n = \lim_{n\to \infty} v_i \lambda_i^n c_i = c_1 \lambda_1 = c_1 \pi$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Convergence to Stationary Distribution)</span></p>

If a transition matrix $P$ on a finite state space is **irreducible** and **aperiodic**, it has a unique stationary distribution $\pi$. The chain will converge to this distribution $\lim_{n\to\infty} \mathbb{P}(X_n = x) = \pi(x)$ regardless of the starting state $X_0$.

</div>

#### The Ergodic Theorem (Sample Means)

While the previous theorem deals with the distribution of a single state at time $n$, we are often interested in the **sample mean** (the average of a function over time).

If $𝑋$ is a random variable on $Γ$ with probability density function 𝜋, then this statement is equivalent to $𝑋_𝑛$ converging (𝑑)
in distribution to 𝑋 as 𝑛 → ∞

* **The Challenge:** In standard statistics, the Law of Large Numbers applies to independent (i.i.d.) samples. However, Markov chain samples are highly **correlated** (the next state depends on the previous one).
* **The Solution:** Despite this correlation, an analogue to the Law of Large Numbers exists for Markov chains, known as the Ergodic Theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Ergodic Theorem)</span></p>

If the transition matrix $P$ on a finite state space is **irreducible**, then for any function $f: \Gamma \to \mathbb{R}$, the sample mean $\bar f_n = \frac{1}{n}\sum_{i=1}^n f(X_i)$ converges to the expected value with probability $1$.

</div>

**Key Distinction:** Unlike the convergence of the distribution (which requires aperiodicity), the Ergodic Theorem generally only requires the chain to be **irreducible**.

There are two different convergence in Markov chains:
* **Stationarity (snapshot convergence)**: it is denoted as $\pi$ probabiltiy vector over states and $\pi_i = \lim_{n\to\infty} P(X_n = i)$. If $P(X_n = i)$ oscillates for $n\to\infty$, then it will never converge. Like in the image example above, in pendulum case or in any periodic process.
* **Ergodic Theorem (mean over long time convergence)** (or the Ergodic Theorem for Markov chains): we denote it as $\pi$ as well here and **ergodicity** means $\pi_i = \lim_{n\to\infty} \frac{1}{N}\sum_{i=0}^{N-1} [X_i == i]$. In example from the image above we have $\pi = (0.25, 0.25, 0.25, 0.25)^\top$, meaning the process is ergodic, but not stationary.
 
> **Intuition** (Stationarity vs. Ergodicity)
> * *Stationarity* = a distribution that looks unchanged after one step. Stationary distribution $π$ is a photograph: what the distribution looks like at a given time if you’re in equilibrium.
> * *Ergodicity* is about the movie: if you follow one trajectory the long-run time averages you measure in that movie match the equilibrium averages.

##### Cesàro convergence vs. Ergodicity

They’re related, but they’re **not the same thing**. The cleanest way to separate them is:

* **Cesàro convergence** = a statement about **averaging distributions (or matrices) over time**.
* **Ergodicity** = a property of the **dynamics** that makes **time averages along a single trajectory** match **space averages under $\pi$** (and in some texts also implies “snapshot convergence”/mixing).

**Cesàro convergence (a.k.a. averaging the snapshots)**

For a Markov chain with transition matrix $P$ and stationary distribution $\pi$, define the rank-one limit matrix

$$\Pi := \mathbf{1}\pi \quad\text{(each row equals }\pi\text{)}.$$

**Cesàro convergence of the chain** typically refers to:

$$\frac{1}{N}\sum_{n=0}^{N-1} P^n \longrightarrow \Pi \quad (N\to\infty).$$

Equivalently, for any initial distribution $\mu$,

$$\frac{1}{N}\sum_{n=0}^{N-1} \mu P^n \longrightarrow \pi.$$

Key point: this can hold even when $P^n$ itself **does not converge** (e.g., periodic chains). *Averaging kills the oscillation*.

**Ergodicity (Markov chains): time averages along one path**

The **ergodic theorem for Markov chains** is about a *single run* $X_0,X_1,\dots$. For a (typically) **irreducible positive recurrent** chain with stationary $\pi$, for any integrable function $f$,

$$\frac{1}{N}\sum_{n=0}^{N-1} f(X_n) \xrightarrow{a.s.} \mathbb{E}_\pi[f].$$

Special case $f=\mathbf 1\lbrace x=i\rbrace$:

$$\frac{1}{N}\sum_{n=0}^{N-1}\mathbf 1{X_n=i} \xrightarrow{a.s.} \pi_i.$$

Key point: this is **stronger in a different direction** than Cesàro convergence:
* Cesàro convergence averages **probability distributions** $\mu P^n$.
* The ergodic theorem averages **observations along a sample path**.

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

<figure>
  <img src="{{ 'assets/images/notes/monte-carlo-methods/random_walk.jpg' | relative_url }}" alt="Random Walk on a 3-Dimensional Hypercube" loading="lazy">
  <figcaption>Random Walk on a 3-Dimensional Hypercube</figcaption>
</figure>

### TODO: Add PAS2 source

### TODO: Add Wiki source