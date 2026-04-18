---
layout: default
title: Stochastic Differential Equations
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Stochastic Differential Equations

## 1. Introduction

The equation we obtain by allowing randomness in the coeﬃcients of a diﬀerential equation is called a **stochastic diﬀerential equation**. This will be made more precise later. It is clear that any solution of a stochastic diﬀerential equation must involve some randomness, i.e. we can only hope to be able to say something about the probability distributions of the solutions.

### 1.1 Stochastic Analogs of Classical Diﬀerential Equations

$\textbf{Example (Population Growth with Noise):}$ A population model with a randomly fluctuating growth rate is described by the stochastic differential equation: 

$$\frac{dN}{dt} = a(t)N(t), \quad N(0) = N_0$$

where $N(t)$ is the population size at time $t$, and the relative growth rate $a(t)$ is given by:  

$$a(t) = r(t) + \text{“noise”}$$

Here, $r(t)$ is a nonrandom (deterministic?) function. We do not know the exact behaviour of the noise term, only its probability distribution. 

### 1.2 Filtering Problems

$\textbf{Example (The Filtering Problem):}$ Given a system described by a stochastic differential equation, such as the RLC circuit example, observations $Z(s)$ are taken at times $s \le t$. These measurements are also corrupted by noise:  

$$Z(s) = Q(s) + \text{“noise”}$$

Basically, we have two sources of noise. The filtering problem is to find the best estimate of the system's state $Q(t)$ based on the history of noisy observations $\lbrace Z_s\rbrace_{s \le t}$.

In 1960 Kalman and in 1961 Kalman and Bucy proved what is now known as the Kalman-Bucy filter. Basically the filter gives a procedure for estimating the state of a system which satisfies a "noisy" linear diﬀerential equation, based on a series of "noisy" observations. Almost immediately the discovery found applications in aerospace engineering (Ranger, Mariner, Apollo etc.) and it now has a broad range of applications.

$\textbf{Example (Stochastic Solution to the Dirichlet Problem):}$ Given a domain $U \subset \mathbb{R}^n$ and a continuous function $f$ on the boundary $\partial U$, the Dirichlet problem is to find a function $\tilde{f}$ continuous on the closure $\bar{U}$ such that:

1. $\tilde{f} = f$ on $\partial U$.
2. $\tilde{f}$ is harmonic in $U: \Delta \tilde{f} := \sum_{i=1}^n \frac{\partial^2 \tilde{f}}{\partial x_i^2} = 0$ in $U$. The solution can be expressed stochastically. The value $\tilde{f}(x)$ is the expected value of $f$ at the first exit point from $U$ of a Brownian motion starting at $x \in U$.

$\textbf{Example (An Optimal Stopping Problem):}$ An asset price $X_t$ at time $t$ evolves according to the stochastic differential equation:

$$\frac{dX_t}{dt} = rX_t + \alpha X_t \cdot \text{"noise"}$$

where $r$ and $\alpha$ are known constants. Given a discount rate $\rho$, the optimal stopping problem is to determine the selling time that maximizes the expected discounted profit.

$\textbf{Example (An Optimal Portfolio Problem):}$ An investor allocates a fortune $X_t$ between two investments:

1. A risky asset with price $p_1(t)$ governed by:
   
   $$\frac{dp_1}{dt} = (a + \alpha \cdot \text{"noise"})p_1, \quad a > 0$$ 

2. A safe asset with price $p_2(t)$ governed by: 
   
   $$\frac{dp_2}{dt} = bp_2, \quad 0 < b < a$$
   
   Let $u_t \in [0, 1]$ be the fraction of the fortune $X_t$ invested in the risky asset at time $t$. Given a utility function $U$ and a terminal time $T$, the stochastic control problem is to find the optimal portfolio strategy $\lbrace u_t\rbrace_{0 \le t \le T}$ that solves:

$$\max_{0 \le u_t \le 1} \lbrace E[U(X_T^{(u)})] \rbrace$$


$\textbf{Example (Pricing of European Call Options):}$ A European call option grants the right, but not the obligation, to buy a unit of a risky asset at a specified price $K$ (the strike price) at a specified future time $T$ (the maturity time). The problem is to determine the fair price of this option at time $t=0$.

## 2. Some Mathematical Preliminaries

### 2.1 Probability Spaces, Random Variables and Stochastic Processes

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sigma-algebra)</span></p>

Given a set $\Omega$, a family $\mathcal{F}$ of subsets of $\Omega$ is a **$\sigma$-algebra** if it satisfies: 
1. $\emptyset \in \mathcal{F}$
2. $F \in \mathcal{F} \implies F^C \in \mathcal{F}$, where $F^C = \Omega \setminus F$
3. $A_1, A_2, \dots \in \mathcal{F} \implies A := \bigcup_{i=1}^\infty A_i \in \mathcal{F}$ 
   
The pair $(\Omega, \mathcal{F})$ is a measurable space.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probability Measure)</span></p>

A **probability measure** $P$ on a measurable space $(\Omega, \mathcal{F})$ is a function $P: \mathcal{F} \to [0, 1]$ such that: 

* **(a)** $P(\emptyset) = 0, P(\Omega) = 1 $
* **(b)** If $\{A_i\}_{i=1}^\infty$ is a disjoint collection of sets in $\mathcal{F}$ (i.e., $A_i \cap A_j = \emptyset $for $i \ne j$), then:  

$$P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)$$

The triple $(\Omega, \mathcal{F}, P)$ is a probability space.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complete Probability Space)</span></p>

A probability space $(\Omega, \mathcal{F}, P)$ is complete if $\mathcal{F}$ contains all subsets $G$ of $\Omega$ with $P$-outer measure zero, where the outer measure is defined as:  

$$P^*(G) := \inf \lbrace P(F); F \in \mathcal{F}, G \subset F\rbrace = 0$$

</div>

> **Remark**: Subsets $F \in \mathcal{F}$ are called $\mathcal{F}$-measurable sets or events. $P(F)$ is the probability that event $F$ occurs. If $P(F)=1$, the event occurs almost surely (a.s.).

$\textbf{Definition } (\textbf{Generated } \sigma \textbf{-algebra:})$ Given a family $\mathcal{U}$ of subsets of $\Omega$, the $\sigma$-algebra generated by $\mathcal{U}$, denoted $\mathcal{H}_\mathcal{U}$, is the smallest $\sigma$-algebra containing $\mathcal{U}$:

$$\mathcal{H}_\mathcal{U} = \bigcap \lbrace\mathcal{H}; \mathcal{H} \text{ is a } \sigma\text{-algebra on } \Omega, \mathcal{U} \subset \mathcal{H}\rbrace$$  

If $\Omega$ is a topological space and $\mathcal{U}$ is the collection of all open subsets, $\mathcal{B} = \mathcal{H}_\mathcal{U}$ is the Borel $\sigma$-algebra.

$\textbf{Definition } (\mathcal{F}\textbf{-measurable Function:})$ A function $Y: \Omega \to \mathbb{R}^n$ is $\mathcal{F}$-measurable if for all open sets $U \subset \mathbb{R}^n$:  

$$Y^{-1}(U) := \lbrace \omega \in \Omega ; Y(\omega) \in U\rbrace \in \mathcal{F}$$

This is equivalent to the condition holding for all Borel sets $U \subset \mathbb{R}^n$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\sigma$-algebra generated by a Function)</span></p>

For a function $X: \Omega \to \mathbb{R}^n$, the **$\sigma$-algebra generated by $X$**, denoted $\mathcal{H}_X$, is the smallest $\sigma$-algebra on $\Omega$ making $X$ measurable. It consists of all sets $X^{-1}(B)$ for all Borel sets $B \in \mathcal{B}$ on $\mathbb{R}^n$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Doob-Dynkin Lemma)</span></p>

Let $X, Y: \Omega \to \mathbb{R}^n$ be two functions. $Y$ is $\mathcal{H}_X$-measurable if and only if there exists a Borel measurable function $g: \mathbb{R}^n \to \mathbb{R}^n$ such that $Y = g(X)$.

</div>


$\textbf{Definition (Random Variable and Expectation):}$ Let $(\Omega, \mathcal{F}, P)$ be a complete probability space. A random variable $X$ is an $\mathcal{F}$-measurable function $X: \Omega \to \mathbb{R}^n$. The distribution of $X$ is the probability measure $\mu_X$ on $\mathbb{R}^n$ defined by $\mu_X(B) = P(X^{-1}(B))$. If $\int_\Omega \|X(\omega)\| dP(\omega) < \infty$, the expectation of $X$ is:  

$$\mathbb{E}[X] := \int_\Omega X(\omega) dP(\omega) = \int_{\mathbb{R}^n} x d\mu_X(x)$$

For a Borel measurable function $f: \mathbb{R}^n \to \mathbb{R}$, if 

$$\int_\Omega \|f(X(\omega))\| dP(\omega) < \infty:  \mathbb{E}[f(X)] := \int_\Omega f(X(\omega)) dP(\omega) = \int_{\mathbb{R}^n} f(x) d\mu_X(x)$$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Independence of Events)</span></p>

Two events $A, B \in \mathcal{F}$ are independent if $P(A \cap B) = P(A) \cdot P(B)$. A collection 

$\mathcal{A} = \lbrace \mathcal{H}\_i ; i \in I\rbrace$ 

of families of measurable sets is independent if for any distinct indices $i_1, \dots, i_k$ and any choice of sets $H_{i_j} \in \mathcal{H}_{i_j}$:

$$P(H_{i_1} \cap \cdots \cap H_{i_k}) = P(H_{i_1}) \cdots P(H_{i_k})$$

A collection of random variables $\lbrace X_i ; i \in I\rbrace$ is independent if the collection of their generated $\sigma$-algebras $\lbrace\mathcal{H}_{X_i}\rbrace$ is independent.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Expectation of the product of independent rvs)</span></p>

If two random variables $X, Y: \Omega \to \mathbb{R}$ are independent and $\mathbb{E}[\|X\|] < \infty, \mathbb{E}[\|Y\|] < \infty$, then $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Process)</span></p>

A **stochastic process** is a parametrized collection of random variables $\lbrace X_t\rbrace_{t \in T}$ defined on a probability space $(\Omega, \mathcal{F}, P)$ and taking values in $\mathbb{R}^n$. The parameter space $T$ is typically $[0, \infty)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

For a fixed $t \in T, X_t(\omega)$ is a random variable. For a fixed $\omega \in \Omega$, the function $t \mapsto X_t(\omega)$ is a path of the process. A process can also be viewed as a function of two variables, $(t, \omega) \mapsto X(t, \omega)$, or as a probability measure on the space of all functions from $T$ to $\mathbb{R}^n$, denoted $(\mathbb{R}^n)^T$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Finite-Dimensional Distributions)</span></p>

The **finite-dimensional distributions** of a process $\lbrace X_t\rbrace_{t \in T}$ are the measures $\mu_{t_1, \dots, t_k}$ on $\mathbb{R}^{nk}$ for $k=1, 2, \dots$ defined by:  

$$\mu_{t_1, \dots, t_k}(F_1 \times \cdots \times F_k) = P[X_{t_1} \in F_1, \dots, X_{t_k} \in F_k]$$

for Borel sets $F_i \subset \mathbb{R}^n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kolmogorov's Extension Theorem)</span></p>

Let $\lbrace \nu_{t_1, \dots, t_k}\rbrace$ be a family of probability measures on $\mathbb{R}^{nk}$ for all $t_1, \dots, t_k \in T$ and $k \in \mathbb{N}$. If this family satisfies the consistency conditions: 
* (K1) $\nu_{t_{\sigma(1)}, \dots, t_{\sigma(k)}}(F_1 \times \cdots \times F_k) = \nu_{t_1, \dots, t_k}(F_{\sigma^{-1}(1)} \times \cdots \times F_{\sigma^{-1}(k)})$ for all permutations $\sigma$.
* (K2) $\nu_{t_1, \dots, t_k}(F_1 \times \cdots \times F_k) = \nu_{t_1, \dots, t_k, t_{k+1}, \dots, t_{k+m}}(F_1 \times \cdots \times F_k \times \mathbb{R}^n \times \cdots \times \mathbb{R}^n)$ for all $m \in \mathbb{N}$. 
  
Then there exists a probability space $(\Omega, \mathcal{F}, P)$ and a stochastic process $\lbrace X_t\rbrace_{t \in T}$ on it such that its finite-dimensional distributions are given by the family $\nu$.

</div>

### 2.2 An Important Example: Brownian Motion

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Construction of Brownian Motion)</span></p>

An $n$-dimensional Brownian motion starting at $x \in \mathbb{R}^n$ is constructed via Kolmogorov's extension theorem. The finite-dimensional distributions are specified for $0 \le t_1 \le \cdots \le t_k$ by defining a measure $\nu_{t_1, \dots, t_k}$ on 
 
$$\mathbb{R}^{nk}:  \nu_{t_1, \dots, t_k}(F_1 \times \cdots \times F_k) = \int_{F_1 \times \cdots \times F_k} p(t_1, x, x_1) p(t_2-t_1, x_1, x_2) \cdots p(t_k-t_{k-1}, x_{k-1}, x_k) dx_1 \cdots dx_k$$  
 
where the transition density function p(t, x, y) is the Gaussian kernel:
 
$$p(t, x, y) = (2\pi t)^{-n/2} \exp\left(-\frac{\|x-y\|^2}{2t}\right) \text{ for } t > 0$$
 
and $p(0, x, y)dy = \delta_x(y)$ is the point mass at $x$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Brownian Motion)</span></p>

A stochastic process $\lbrace B_t\rbrace_{t \ge 0}$ whose finite-dimensional distributions are given by the construction above is called a **Brownian motion** starting at $x$. By construction, $P^x(B_0=x)=1$.

</div>

> **Remark: (Properties of Brownian Motion)** 
> 
> * (i) Gaussian Process: For $0 \le t_1 \le \cdots \le t_k$, the random vector $Z = (B_{t_1}, \dots, B_{t_k}) \in \mathbb{R}^{nk}$ has a multivariate normal distribution. Its characteristic function is:  
> 
> $$E^x\left[\exp\left(i \sum_{j=1}^{nk} u_j Z_j\right)\right] = \exp\left(-\frac{1}{2} \sum_{j,m} u_j c_{jm} u_m + i \sum_j u_j M_j\right)$$  
> 
> The mean vector is $M = E^x[Z] = (x, x, \dots, x) \in \mathbb{R}^{nk}$. The covariance matrix $C=[c_{jm}]$ has entries corresponding to the block matrix components 
> 
> $$E^x[(B_{t_i}-x)(B_{t_j}-x)^T] = \min(t_i, t_j) I_n,$$
> 
> resulting in:  
> 
> $$C = \begin{pmatrix} t_1 I_n & t_1 I_n & \cdots & t_1 I_n \\ t_1 I_n & t_2 I_n & \cdots & t_2 I_n \\ \vdots & \vdots & \ddots & \vdots \\ t_1 I_n & t_2 I_n & \cdots & t_k I_n \end{pmatrix}$$
> 
> From this, for all $t,s \ge 0$:  
> 
> $$E^x[B_t] = x   E^x[(B_t - x)(B_s - x)] = n \min(s, t)   E^x[(B_t-B_s)^2] = n(t-s) \quad \text{if } t \ge s  $$
> 
> * (ii) Independent Increments: For any $0 \le t_1 < t_2 < \cdots < t_k$, the random variables $B_{t_1}, B_{t_2}-B_{t_1}, \dots, B_{t_k}-B_{t_{k-1}}$ are independent.

*Proof:* Since the increments are jointly normal, independence is equivalent to being uncorrelated. For $t_i < t_j$: 

$$E^x[(B_{t_i} - B_{t_{i-1}})(B_{t_j} - B_{t_{j-1}})] = n(\min(t_i, t_j) - \min(t_{i-1}, t_j) - \min(t_i, t_{j-1}) + \min(t_{i-1}, t_{j-1})).$$ 

Assuming $t_{i-1} < t_i < t_{j-1} < t_j$: 

$$E^x[(B_{t_i} - B_{t_{i-1}})(B_{t_j} - B_{t_{j-1}})] = n(t_i - t_{i-1} - t_i + t_{i-1}) = 0.$$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Version of a Stochastic Process)</span></p>

A process $\lbrace X_t\rbrace$ is a **version (or modification) of a process** $\lbrace Y_t\rbrace$ if for all $t$:  

$$P(\lbrace\omega; X_t(\omega) = Y_t(\omega)\rbrace) = 1$$

</div>

> **Remark**: Versions of a process have the same finite-dimensional distributions but may have different path properties.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kolmogorov's Continuity Theorem)</span></p>

Suppose a process $\lbrace X_t\rbrace_{t \ge 0}$ satisfies the condition that for all $T > 0$, there exist positive constants $\alpha, \beta, D$ such that for 

$$0 \le s, t \le T:  \mathbb{E}[\|X_t - X_s\|^\alpha] \le D \cdot \|t - s\|^{1+\beta}$$

Then there exists a continuous version of $X$.

</div>

> **Remark**: For Brownian motion $\lbrace B_t\rbrace$, the following holds:  
> 
> $$E^x[\|B_t - B_s\|^4] = n(n+2)\|t-s\|^2$$
> 
> This satisfies the condition of Kolmogorov's continuity theorem with $\alpha=4, \beta=1, D=n(n+2)$. Therefore, a continuous version of Brownian motion exists, and this version is assumed henceforth.

> **Remark**: If $B_t = (B_t^{(1)}, \dots, B_t^{(n)})$ is an $n$-dimensional Brownian motion, then the component processes $\lbrace B_t^{(j)}\rbrace_{t \ge 0}$ for $j=1, \dots, n$ are independent, $1$-dimensional Brownian motions.

## 3. Itô Integrals

### 3.1 Construction of the Itô Integral

We now turn to the question of finding a reasonable mathematical interpretation of the "noise" term. Consider the population growth equation from the Introduction:

$$\frac{dN}{dt} = (r(t) + \text{"noise"})N(t)$$

or more generally equations of the form

$$\frac{dX}{dt} = b(t, X_t) + \sigma(t, X_t) \cdot \text{"noise"}$$

where $b$ and $\sigma$ are given functions. Concentrating on the 1-dimensional case, it is natural to look for a stochastic process $W_t$ to represent the noise term, so that

$$\frac{dX}{dt} = b(t, X_t) + \sigma(t, X_t) \cdot W_t$$

Based on many situations, one is led to assume that $W_t$ has these properties:

1. $t_1 \ne t_2 \Rightarrow W_{t_1}$ and $W_{t_2}$ are independent.
2. $\lbrace W_t\rbrace$ is stationary, i.e. the joint distribution of $\lbrace W_{t_1+t}, \dots, W_{t_k+t}\rbrace$ does not depend on $t$.
3. $E[W_t] = 0$ for all $t$.

However, it turns out there does *not* exist any "reasonable" stochastic process satisfying (1) and (2): such a $W_t$ cannot have continuous paths. If we require $E[W_t^2] = 1$ then the function $(t,\omega) \to W_t(\omega)$ cannot even be measurable with respect to the $\sigma$-algebra $\mathcal{B} \times \mathcal{F}$, where $\mathcal{B}$ is the Borel $\sigma$-algebra on $[0,\infty]$. Nevertheless it is possible to represent $W_t$ as a generalized stochastic process called the **white noise process**.

We avoid this kind of construction and rather rewrite the equation in a form that suggests a replacement of $W_t$ by a proper stochastic process. Let $0 = t_0 < t_1 < \cdots < t_m = t$ and consider a discrete version:

$$X_{k+1} - X_k = b(t_k, X_k)\Delta t_k + \sigma(t_k, X_k)W_k\Delta t_k$$

where $X_j = X(t_j)$, $W_k = W_{t_k}$, $\Delta t_k = t_{k+1} - t_k$. We abandon the $W_k$-notation and replace $W_k\Delta t_k$ by $\Delta V_k = V_{t_{k+1}} - V_{t_k}$, where $\lbrace V_t\rbrace_{t \ge 0}$ is some suitable stochastic process. The assumptions (1)–(3) on $W_t$ suggest that $V_t$ should have **stationary independent increments with mean 0**. It turns out that the only such process with continuous paths is the Brownian motion $B_t$. Thus we put $V_t = B_t$ and obtain:

$$X_k = X_0 + \sum_{j=0}^{k-1} b(t_j, X_j)\Delta t_j + \sum_{j=0}^{k-1} \sigma(t_j, X_j)\Delta B_j$$

Taking the limit as $\Delta t_j \to 0$, we should obtain

$$X_t = X_0 + \int_0^t b(s, X_s)ds + \int_0^t \sigma(s, X_s)dB_s$$

and we adopt the convention that $X_t = X_t(\omega)$ is a stochastic process satisfying this integral equation. The goal of this chapter is to prove the existence, in a certain sense, of $\int_0^t f(s,\omega)dB_s(\omega)$ where $B_t(\omega)$ is 1-dimensional Brownian motion starting at the origin.

Suppose $0 \le S < T$ and $f(t,\omega)$ is given. We want to define

$$\int_S^T f(t,\omega)dB_t(\omega)$$

It is reasonable to start with a definition for a simple class of functions $f$ and then extend by an approximation procedure. Thus, let us first assume that $f$ has the form

$$\phi(t,\omega) = \sum_{j \ge 0} e_j(\omega) \cdot \mathcal{X}_{[j \cdot 2^{-n}, (j+1)2^{-n})}(t)$$

where $\mathcal{X}$ denotes the characteristic (indicator) function and $n$ is a natural number. For such functions it is reasonable to define

$$\int_S^T \phi(t,\omega)dB_t(\omega) = \sum_{j \ge 0} e_j(\omega)[B_{t_{j+1}} - B_{t_j}](\omega)$$

where $t_k = t_k^{(n)}$ is defined as $k \cdot 2^{-n}$ if $S \le k \cdot 2^{-n} \le T$, as $S$ if $k \cdot 2^{-n} < S$, and as $T$ if $k \cdot 2^{-n} > T$.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Choice of Evaluation Point Matters)</span></p>

Without further assumptions on $e_j(\omega)$ this leads to difficulties. Choose

$$\phi_1(t,\omega) = \sum_{j \ge 0} B_{j \cdot 2^{-n}}(\omega) \cdot \mathcal{X}_{[j \cdot 2^{-n}, (j+1)2^{-n})}(t)$$

$$\phi_2(t,\omega) = \sum_{j \ge 0} B_{(j+1)2^{-n}}(\omega) \cdot \mathcal{X}_{[j \cdot 2^{-n}, (j+1)2^{-n})}(t)$$

Then

$$E\left[\int_0^T \phi_1(t,\omega)dB_t(\omega)\right] = \sum_{j \ge 0} E[B_{t_j}(B_{t_{j+1}} - B_{t_j})] = 0$$

since $\lbrace B_t\rbrace$ has independent increments. But

$$E\left[\int_0^T \phi_2(t,\omega)dB_t(\omega)\right] = \sum_{j \ge 0} E[B_{t_{j+1}} \cdot (B_{t_{j+1}} - B_{t_j})] = \sum_{j \ge 0} E[(B_{t_{j+1}} - B_{t_j})^2] = T$$

So, in spite of the fact that both $\phi_1$ and $\phi_2$ appear to be very reasonable approximations to $f(t,\omega) = B_t(\omega)$, their integrals are not close to each other at all, no matter how large $n$ is.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Paths of Brownian Motion)</span></p>

This only reflects the fact that the variations of the paths of $B_t$ are too big to enable us to define the integral in the Riemann-Stieltjes sense. In fact, one can show that the paths $t \to B_t$ of Brownian motion are nowhere differentiable, almost surely. In particular, the total variation of the path is infinite, a.s.

</div>

In general it is natural to approximate a given function $f(t,\omega)$ by

$$\sum_j f(t_j^*,\omega) \cdot \mathcal{X}_{[t_j, t_{j+1})}(t)$$

where the points $t_j^*$ belong to the intervals $[t_j, t_{j+1}]$, and then define $\int_S^T f(t,\omega)dB_t(\omega)$ as the limit of $\sum_j f(t_j^*,\omega)[B_{t_{j+1}} - B_{t_j}](\omega)$ as $n \to \infty$. However, the example above shows that — unlike the Riemann-Stieltjes integral — it does make a difference here what points $t_j^*$ we choose. The following two choices have turned out to be the most useful ones:

1. $t_j^* = t_j$ (the left end point), which leads to the **Itô integral**, denoted by $\int_S^T f(t,\omega)dB_t(\omega)$.
2. $t_j^* = (t_j + t_{j+1})/2$ (the mid point), which leads to the **Stratonovich integral**, denoted by $\int_S^T f(t,\omega) \circ dB_t(\omega)$.

In any case one must restrict oneself to a special class of functions $f(t,\omega)$ in order to obtain a reasonable definition of the integral. We present Itô's choice $t_j^* = t_j$. The approximation procedure works successfully provided that $f$ has the property that each of the functions $\omega \to f(t_j,\omega)$ *only depends on the behaviour of $B_s(\omega)$ up to time $t_j$*. This leads to the following important concepts:

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Filtration Generated by Brownian Motion)</span></p>

Let $B_t(\omega)$ be $n$-dimensional Brownian motion. Then we define $\mathcal{F}_t = \mathcal{F}_t^{(n)}$ to be the $\sigma$-algebra generated by the random variables $B_s(\cdot)$; $s \le t$. In other words, $\mathcal{F}_t$ is the smallest $\sigma$-algebra containing all sets of the form

$$\lbrace\omega; B_{t_1}(\omega) \in F_1, \cdots, B_{t_k}(\omega) \in F_k\rbrace$$

where $t_j \le t$ and $F_j \subset \mathbb{R}^n$ are Borel sets, $j \le k = 1, 2, \dots$ (We assume that all sets of measure zero are included in $\mathcal{F}_t$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of $\mathcal{F}_t$)</span></p>

One often thinks of $\mathcal{F}_t$ as "the history of $B_s$ up to time $t$". A function $h(\omega)$ will be $\mathcal{F}_t$-measurable if and only if $h$ can be written as the pointwise a.e. limit of sums of functions of the form $g_1(B_{t_1})g_2(B_{t_2})\cdots g_k(B_{t_k})$, where $g_1, \dots, g_k$ are bounded continuous functions and $t_j \le t$ for $j \le k$, $k = 1, 2, \dots$. Intuitively, that $h$ is $\mathcal{F}_t$-measurable means that the value of $h(\omega)$ can be decided from the values of $B_s(\omega)$ for $s \le t$. For example, $h_1(\omega) = B_{t/2}(\omega)$ is $\mathcal{F}_t$-measurable, while $h_2(\omega) = B_{2t}(\omega)$ is not. Note that $\mathcal{F}_s \subset \mathcal{F}_t$ for $s < t$ (i.e. $\lbrace\mathcal{F}_t\rbrace$ is *increasing*) and that $\mathcal{F}_t \subset \mathcal{F}$ for all $t$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Adapted Process)</span></p>

Let $\lbrace\mathcal{N}_t\rbrace_{t \ge 0}$ be an increasing family of $\sigma$-algebras of subsets of $\Omega$. A process $g(t,\omega): [0,\infty) \times \Omega \to \mathbb{R}^n$ is called **$\mathcal{N}_t$-adapted** if for each $t \ge 0$ the function $\omega \to g(t,\omega)$ is $\mathcal{N}_t$-measurable.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Class $\mathcal{V}$)</span></p>

Let $\mathcal{V} = \mathcal{V}(S,T)$ be the class of functions $f(t,\omega): [0,\infty) \times \Omega \to \mathbb{R}$ such that

1. $(t,\omega) \to f(t,\omega)$ is $\mathcal{B} \times \mathcal{F}$-measurable, where $\mathcal{B}$ denotes the Borel $\sigma$-algebra on $[0,\infty)$.
2. $f(t,\omega)$ is $\mathcal{F}_t$-adapted.
3. $E\left[\int_S^T f(t,\omega)^2 dt\right] < \infty$.

</div>

#### The Itô Integral

For functions $f \in \mathcal{V}$ we will now show how to define the **Itô integral**

$$\mathcal{I}[f](\omega) = \int_S^T f(t,\omega)dB_t(\omega)$$

where $B_t$ is 1-dimensional Brownian motion. The idea is natural: First we define $\mathcal{I}[\phi]$ for a simple class of functions $\phi$. Then we show that each $f \in \mathcal{V}$ can be approximated (in an appropriate sense) by such $\phi$'s and we use this to define $\int f dB$ as the limit of $\int \phi dB$ as $\phi \to f$.

A function $\phi \in \mathcal{V}$ is called **elementary** if it has the form

$$\phi(t,\omega) = \sum_j e_j(\omega) \cdot \mathcal{X}_{(t_j, t_{j+1})}(t)$$

Note that since $\phi \in \mathcal{V}$ each function $e_j$ must be $\mathcal{F}_{t_j}$-measurable. For elementary functions $\phi(t,\omega)$ we define the integral as

$$\int_S^T \phi(t,\omega)dB_t(\omega) = \sum_{j \ge 0} e_j(\omega)[B_{t_{j+1}} - B_{t_j}](\omega)$$

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(The Itô Isometry — Elementary Case)</span></p>

If $\phi(t,\omega)$ is bounded and elementary then

$$E\left[\left(\int_S^T \phi(t,\omega)dB_t(\omega)\right)^2\right] = E\left[\int_S^T \phi(t,\omega)^2 dt\right]$$

</div>

*Proof.* Put $\Delta B_j = B_{t_{j+1}} - B_{t_j}$. Then

$$E[e_i e_j \Delta B_i \Delta B_j] = \begin{cases} 0 & \text{if } i \ne j \\ E[e_j^2] \cdot (t_{j+1} - t_j) & \text{if } i = j \end{cases}$$

using that $e_i e_j$ and $\Delta B_j$ are independent if $i < j$. Thus

$$E\left[\left(\int_S^T \phi \, dB\right)^2\right] = \sum_{i,j} E[e_i e_j \Delta B_i \Delta B_j] = \sum_j E[e_j^2] \cdot (t_{j+1} - t_j) = E\left[\int_S^T \phi^2 \, dt\right]$$

The idea is now to use the isometry to extend the definition from elementary functions to functions in $\mathcal{V}$. We do this in several steps:

**Step 1.** Let $g \in \mathcal{V}$ be bounded and $g(\cdot,\omega)$ continuous for each $\omega$. Then there exist elementary functions $\phi_n \in \mathcal{V}$ such that $E\left[\int_S^T (g - \phi_n)^2 dt\right] \to 0$ as $n \to \infty$.

**Step 2.** Let $h \in \mathcal{V}$ be bounded. Then there exist bounded functions $g_n \in \mathcal{V}$ such that $g_n(\cdot,\omega)$ is continuous for all $\omega$ and $n$, and $E\left[\int_S^T (h - g_n)^2 dt\right] \to 0$.

**Step 3.** Let $f \in \mathcal{V}$. Then there exists a sequence $\lbrace h_n\rbrace \subset \mathcal{V}$ such that $h_n$ is bounded for each $n$ and $E\left[\int_S^T (f - h_n)^2 dt\right] \to 0$ as $n \to \infty$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Itô Integral)</span></p>

Let $f \in \mathcal{V}(S,T)$. Then the **Itô integral** of $f$ (from $S$ to $T$) is defined by

$$\int_S^T f(t,\omega)dB_t(\omega) = \lim_{n \to \infty} \int_S^T \phi_n(t,\omega)dB_t(\omega) \quad (\text{limit in } L^2(P))$$

where $\lbrace\phi_n\rbrace$ is a sequence of elementary functions such that

$$E\left[\int_S^T (f(t,\omega) - \phi_n(t,\omega))^2 dt\right] \to 0 \quad \text{as } n \to \infty$$

Note that such a sequence $\lbrace\phi_n\rbrace$ exists by Steps 1–3 above. Moreover, by the isometry the limit exists and does not depend on the actual choice of $\lbrace\phi_n\rbrace$, as long as the convergence condition holds.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(The Itô Isometry)</span></p>

$$E\left[\left(\int_S^T f(t,\omega)dB_t\right)^2\right] = E\left[\int_S^T f^2(t,\omega)dt\right] \quad \text{for all } f \in \mathcal{V}(S,T)$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Convergence of Itô Integrals)</span></p>

If $f(t,\omega) \in \mathcal{V}(S,T)$ and $f_n(t,\omega) \in \mathcal{V}(S,T)$ for $n = 1, 2, \dots$ and $E\left[\int_S^T (f_n(t,\omega) - f(t,\omega))^2 dt\right] \to 0$ as $n \to \infty$, then

$$\int_S^T f_n(t,\omega)dB_t(\omega) \to \int_S^T f(t,\omega)dB_t(\omega) \quad \text{in } L^2(P) \text{ as } n \to \infty$$

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Itô Integral of $B_s$)</span></p>

Assume $B_0 = 0$. Then

$$\int_0^t B_s \, dB_s = \frac{1}{2}B_t^2 - \frac{1}{2}t$$

</div>

*Proof.* Put $\phi_n(s,\omega) = \sum B_j(\omega) \cdot \mathcal{X}_{(t_j, t_{j+1})}(s)$, where $B_j = B_{t_j}$. Then

$$E\left[\int_0^t (\phi_n - B_s)^2 ds\right] = E\left[\sum_j \int_{t_j}^{t_{j+1}} (B_j - B_s)^2 ds\right] = \sum_j \int_{t_j}^{t_{j+1}} (s - t_j) ds = \sum_j \frac{1}{2}(t_{j+1} - t_j)^2 \to 0$$

as $\Delta t_j \to 0$. So by the convergence corollary

$$\int_0^t B_s \, dB_s = \lim_{\Delta t_j \to 0} \int_0^t \phi_n \, dB_s = \lim_{\Delta t_j \to 0} \sum_j B_j \Delta B_j$$

Now $\Delta(B_j^2) = B_{j+1}^2 - B_j^2 = (B_{j+1} - B_j)^2 + 2B_j(B_{j+1} - B_j) = (\Delta B_j)^2 + 2B_j \Delta B_j$, and therefore, since $B_0 = 0$,

$$B_t^2 = \sum_j \Delta(B_j^2) = \sum_j (\Delta B_j)^2 + 2\sum_j B_j \Delta B_j$$

or

$$\sum_j B_j \Delta B_j = \frac{1}{2}B_t^2 - \frac{1}{2}\sum_j (\Delta B_j)^2$$

Since $\sum_j (\Delta B_j)^2 \to t$ in $L^2(P)$ as $\Delta t_j \to 0$, the result follows. The extra term $-\frac{1}{2}t$ shows that the Itô stochastic integral does not behave like ordinary integrals.

### 3.2 Some Properties of the Itô Integral

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Properties of the Itô Integral)</span></p>

Let $f, g \in \mathcal{V}(0,T)$ and let $0 \le S < U < T$. Then

1. $\int_S^T f \, dB_t = \int_S^U f \, dB_t + \int_U^T f \, dB_t$ for a.a. $\omega$
2. $\int_S^T (cf + g) \, dB_t = c \cdot \int_S^T f \, dB_t + \int_S^T g \, dB_t$ ($c$ constant) for a.a. $\omega$
3. $E\left[\int_S^T f \, dB_t\right] = 0$
4. $\int_S^T f \, dB_t$ is $\mathcal{F}_T$-measurable.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Filtration and Martingale)</span></p>

A **filtration** (on $(\Omega, \mathcal{F})$) is a family $\mathcal{M} = \lbrace\mathcal{M}_t\rbrace_{t \ge 0}$ of $\sigma$-algebras $\mathcal{M}_t \subset \mathcal{F}$ such that $0 \le s < t \Rightarrow \mathcal{M}_s \subset \mathcal{M}_t$ (i.e. $\lbrace\mathcal{M}_t\rbrace$ is increasing). An $n$-dimensional stochastic process $\lbrace M_t\rbrace_{t \ge 0}$ on $(\Omega, \mathcal{F}, P)$ is called a **martingale** with respect to a filtration $\lbrace\mathcal{M}_t\rbrace_{t \ge 0}$ (and with respect to $P$) if

1. $M_t$ is $\mathcal{M}_t$-measurable for all $t$,
2. $E[\|M_t\|] < \infty$ for all $t$, and
3. $E[M_s \mid \mathcal{M}_t] = M_t$ for all $s \ge t$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Brownian Motion is a Martingale)</span></p>

Brownian motion $B_t$ in $\mathbb{R}^n$ is a martingale w.r.t. the $\sigma$-algebras $\mathcal{F}_t$ generated by $\lbrace B_s; s \le t\rbrace$, because

$$E[\|B_t\|]^2 \le E[\|B_t\|^2] = \|B_0\|^2 + nt$$

and if $s \ge t$ then

$$E[B_s \mid \mathcal{F}_t] = E[B_s - B_t + B_t \mid \mathcal{F}_t] = E[B_s - B_t \mid \mathcal{F}_t] + E[B_t \mid \mathcal{F}_t] = 0 + B_t = B_t$$

Here we have used that $E[(B_s - B_t) \mid \mathcal{F}_t] = E[B_s - B_t] = 0$ since $B_s - B_t$ is independent of $\mathcal{F}_t$, and $E[B_t \mid \mathcal{F}_t] = B_t$ since $B_t$ is $\mathcal{F}_t$-measurable.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Doob's Martingale Inequality)</span></p>

If $M_t$ is a martingale such that $t \to M_t(\omega)$ is continuous a.s., then for all $p \ge 1, T \ge 0$ and all $\lambda > 0$

$$P\left[\sup_{0 \le t \le T} \|M_t\| \ge \lambda\right] \le \frac{1}{\lambda^p} \cdot E[\|M_T\|^p]$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Continuous Version of the Itô Integral)</span></p>

Let $f \in \mathcal{V}(0,T)$. Then there exists a $t$-continuous version of

$$\int_0^t f(s,\omega)dB_s(\omega); \quad 0 \le t \le T$$

i.e. there exists a $t$-continuous stochastic process $J_t$ on $(\Omega, \mathcal{F}, P)$ such that

$$P\left[J_t = \int_0^t f \, dB\right] = 1 \quad \text{for all } t, 0 \le t \le T$$

</div>

From now on we shall always assume that $\int_0^t f(s,\omega)dB_s(\omega)$ means a $t$-continuous version of the integral.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Martingale Property and Maximal Inequality)</span></p>

Let $f(t,\omega) \in \mathcal{V}(0,T)$ for all $T$. Then

$$M_t(\omega) = \int_0^t f(s,\omega)dB_s$$

is a martingale w.r.t. $\mathcal{F}_t$ and

$$P\left[\sup_{0 \le t \le T} \|M_t\| \ge \lambda\right] \le \frac{1}{\lambda^2} \cdot E\left[\int_0^T f(s,\omega)^2 ds\right]; \quad \lambda, T > 0$$

</div>

### 3.3 Extensions of the Itô Integral

The Itô integral $\int f \, dB$ can be defined for a larger class of integrands $f$ than $\mathcal{V}$. First, the measurability condition (2) of Definition 3.1.4 can be relaxed to the following:

(2)' There exists an increasing family of $\sigma$-algebras $\mathcal{H}_t$; $t \ge 0$ such that
&nbsp;&nbsp;&nbsp;&nbsp;a) $B_t$ is a martingale with respect to $\mathcal{H}_t$ and
&nbsp;&nbsp;&nbsp;&nbsp;b) $f_t$ is $\mathcal{H}_t$-adapted.

Note that (a) implies that $\mathcal{F}_t \subset \mathcal{H}_t$. The essence of this extension is that we can allow $f_t$ to depend on more than $\mathcal{F}_t$ as long as $B_t$ remains a martingale with respect to the "history" of $f_s$; $s \le t$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multi-Dimensional Itô Integral)</span></p>

Let $B = (B_1, B_2, \dots, B_n)$ be $n$-dimensional Brownian motion. Then $\mathcal{V}_{\mathcal{H}}^{m \times n}(S,T)$ denotes the set of $m \times n$ matrices $v = [v_{ij}(t,\omega)]$ where each entry $v_{ij}(t,\omega)$ satisfies conditions (1) and (3) of Definition 3.1.4 and (2)' above, with respect to some filtration $\mathcal{H} = \lbrace\mathcal{H}_t\rbrace_{t \ge 0}$.

If $v \in \mathcal{V}_{\mathcal{H}}^{m \times n}(S,T)$ we define, using matrix notation

$$\int_S^T v \, dB = \int_S^T \begin{pmatrix} v_{11} & \cdots & v_{1n} \\ \vdots & & \vdots \\ v_{m1} & \cdots & v_{mn} \end{pmatrix} \begin{pmatrix} dB_1 \\ \vdots \\ dB_n \end{pmatrix}$$

to be the $m \times 1$ matrix (column vector) whose $i$'th component is the following sum of (extended) 1-dimensional Itô integrals:

$$\sum_{j=1}^n \int_S^T v_{ij}(s,\omega)dB_j(s,\omega)$$

</div>

The next extension of the Itô integral consists of weakening condition (3) of Definition 3.1.4 to

(3)' $\quad P\left[\int_S^T f(s,\omega)^2 ds < \infty\right] = 1$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Class $\mathcal{W}$)</span></p>

$\mathcal{W}_{\mathcal{H}}(S,T)$ denotes the class of processes $f(t,\omega) \in \mathbb{R}$ satisfying (1) of Definition 3.1.4 and (2)', (3)' above. Similarly to the notation for $\mathcal{V}$ we put $\mathcal{W}_{\mathcal{H}} = \bigcap_{T > 0} \mathcal{W}_{\mathcal{H}}(0,T)$ and in the matrix case we write $\mathcal{W}_{\mathcal{H}}^{m \times n}(S,T)$ etc. If $\mathcal{H} = \mathcal{F}^{(n)}$ we write $\mathcal{W}(S,T)$ instead of $\mathcal{W}_{\mathcal{F}^{(n)}}(S,T)$ etc.

</div>

Let $B_t$ denote 1-dimensional Brownian motion. If $f \in \mathcal{W}_{\mathcal{H}}$ one can show that for all $t$ there exist step functions $f_n \in \mathcal{W}_{\mathcal{H}}$ such that $\int_0^t \|f_n - f\|^2 ds \to 0$ in probability. For such a sequence one has that $\int_0^t f_n(s,\omega)dB_s$ converges in probability to some random variable and the limit only depends on $f$, not on the sequence $\lbrace f_n\rbrace$. Thus we may define

$$\int_0^t f(s,\omega)dB_s(\omega) = \lim_{n \to \infty} \int_0^t f_n(s,\omega)dB_s(\omega) \quad \text{(limit in probability) for } f \in \mathcal{W}_{\mathcal{H}}$$

As before there exists a $t$-continuous version of this integral. Note, however, that this integral is not in general a martingale. It is, however, a *local* martingale.

#### A Comparison of Itô and Stratonovich Integrals

Let us now return to our original question: We have argued that the mathematical interpretation of the white noise equation

$$\frac{dX}{dt} = b(t, X_t) + \sigma(t, X_t) \cdot W_t$$

is that $X_t$ is a solution of the integral equation

$$X_t = X_0 + \int_0^t b(s, X_s)ds + \int_0^t \sigma(s, X_s)dB_s$$

for some suitable interpretation of the last integral. The Itô interpretation of an integral of the form $\int_0^t f(s,\omega)dB_s(\omega)$ is just one of several reasonable choices. The Stratonovich integral is another possibility, leading (in general) to a different result. The question remains: *Which interpretation makes the "right" mathematical model?*

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stratonovich as Physical Limit)</span></p>

Here is an argument that indicates that the **Stratonovich** interpretation in some situations may be the most appropriate: Choose $t$-continuously differentiable processes $B_t^{(n)}$ such that for a.a. $\omega$

$$B^{(n)}(t,\omega) \to B(t,\omega) \quad \text{as } n \to \infty$$

uniformly (in $t$) in bounded intervals. For each $\omega$ let $X_t^{(n)}(\omega)$ be the solution of the corresponding (deterministic) differential equation

$$\frac{dX_t}{dt} = b(t, X_t) + \sigma(t, X_t)\frac{dB_t^{(n)}}{dt}$$

Then $X_t^{(n)}(\omega) \to X_t(\omega)$ as $n \to \infty$ uniformly in bounded intervals, and this limit $X_t$ coincides with the solution obtained by using **Stratonovich** integrals, i.e.

$$X_t = X_0 + \int_0^t b(s, X_s)ds + \int_0^t \sigma(s, X_s) \circ dB_s$$

This implies that $X_t$ is the solution of the following **modified Itô equation**:

$$X_t = X_0 + \int_0^t b(s, X_s)ds + \frac{1}{2}\int_0^t \sigma'(s, X_s)\sigma(s, X_s)ds + \int_0^t \sigma(s, X_s)dB_s$$

where $\sigma'$ denotes the derivative of $\sigma(t,x)$ with respect to $x$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Itô vs Stratonovich: When to Use Which)</span></p>

Therefore, from the physical-limit point of view it seems reasonable to use the Stratonovich interpretation. On the other hand, the specific feature of the Itô model of "not looking into the future" seems to be a reason for choosing the Itô interpretation in many cases, for example in biology. Note that the Stratonovich and Itô interpretations coincide if $\sigma(t,x)$ does not depend on $x$. This is the situation in the linear case handled in the filtering problem.

In general one can say that the Stratonovich integral has the advantage of leading to ordinary chain rule formulas under a transformation (change of variable), i.e. there are no second order terms in the Stratonovich analogue of the Itô transformation formula. This property makes the Stratonovich integral natural to use for example in connection with stochastic differential equations on manifolds.

However, Stratonovich integrals are not martingales, as we have seen that Itô integrals are. This gives the Itô integral an important computational advantage, even though it does not behave so nicely under transformations. For our purposes the Itô integral will be most convenient, so we will base our discussion on that from now on.

</div>

## 4. The Itô Formula and the Martingale Representation Theorem

### 4.1 The 1-Dimensional Itô Formula

Example 3.1.9 illustrates that the basic definition of Itô integrals is not very useful when we try to evaluate a given integral. This is similar to the situation for ordinary Riemann integrals, where we do not use the basic definition but rather the fundamental theorem of calculus plus the chain rule in the explicit calculations.

In this context, however, we have no differentiation theory, only integration theory. Nevertheless it turns out that it is possible to establish an Itô integral version of the chain rule, called the Itô formula. The Itô formula is, as we will show by examples, very useful for evaluating Itô integrals.

From the example

$$\int_0^t B_s dB_s = \frac{1}{2}B_t^2 - \frac{1}{2}t \quad \text{or} \quad \frac{1}{2}B_t^2 = \frac{1}{2}t + \int_0^t B_s dB_s$$

we see that the image of the Itô integral $B_t = \int_0^t dB_s$ by the map $g(x) = \frac{1}{2}x^2$ is not again an Itô integral of the form $\int_0^t f(s,\omega)dB_s(\omega)$ but a combination of a $dB_s$-and a $ds$-integral:

$$\frac{1}{2}B_t^2 = \int_0^t \frac{1}{2}ds + \int_0^t B_s dB_s$$

It turns out that if we introduce **Itô processes** (also called **stochastic integrals**) as sums of a $dB_s$-and a $ds$-integral then this family of integrals is stable under smooth maps.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(1-Dimensional Itô Process)</span></p>

Let $B_t$ be 1-dimensional Brownian motion on $(\Omega, \mathcal{F}, P)$. A **1-dimensional Itô process** (or stochastic integral) is a stochastic process $X_t$ on $(\Omega, \mathcal{F}, P)$ of the form

$$X_t = X_0 + \int_0^t u(s,\omega)ds + \int_0^t v(s,\omega)dB_s$$

where $v \in \mathcal{W}_{\mathcal{H}}$, so that $P\left[\int_0^t v(s,\omega)^2 ds < \infty \text{ for all } t \ge 0\right] = 1$. We also assume that $u$ is $\mathcal{H}_t$-adapted and $P\left[\int_0^t \|u(s,\omega)\|ds < \infty \text{ for all } t \ge 0\right] = 1$.

If $X_t$ is an Itô process the equation is sometimes written in the shorter differential form

$$dX_t = u \, dt + v \, dB_t$$

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Differential form of $\frac{1}{2}B_t^2$)</span></p>

The equation $\frac{1}{2}B_t^2 = \int_0^t \frac{1}{2}ds + \int_0^t B_s dB_s$ may be represented by

$$d\!\left(\frac{1}{2}B_t^2\right) = \frac{1}{2}dt + B_t dB_t$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The 1-Dimensional Itô Formula)</span></p>

Let $X_t$ be an Itô process given by $dX_t = u \, dt + v \, dB_t$. Let $g(t,x) \in C^2([0,\infty) \times \mathbb{R})$ (i.e. $g$ is twice continuously differentiable on $[0,\infty) \times \mathbb{R}$). Then

$$Y_t = g(t, X_t)$$

is again an Itô process, and

$$dY_t = \frac{\partial g}{\partial t}(t, X_t)dt + \frac{\partial g}{\partial x}(t, X_t)dX_t + \frac{1}{2}\frac{\partial^2 g}{\partial x^2}(t, X_t) \cdot (dX_t)^2$$

where $(dX_t)^2 = (dX_t) \cdot (dX_t)$ is computed according to the rules

$$dt \cdot dt = dt \cdot dB_t = dB_t \cdot dt = 0, \quad dB_t \cdot dB_t = dt$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Expanded form of the Itô formula)</span></p>

If we substitute $dX_t = u \, dt + v \, dB_t$ and use the multiplication rules, the Itô formula becomes

$$g(t, X_t) = g(0, X_0) + \int_0^t \left(\frac{\partial g}{\partial s}(s, X_s) + u_s \frac{\partial g}{\partial x}(s, X_s) + \frac{1}{2}v_s^2 \cdot \frac{\partial^2 g}{\partial x^2}(s, X_s)\right)ds + \int_0^t v_s \cdot \frac{\partial g}{\partial x}(s, X_s) dB_s$$

where $u_s = u(s,\omega)$, $v_s = v(s,\omega)$. Note that this is an Itô process in the sense of Definition 4.1.1.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Recovering $\int B_s \, dB_s$ via Itô's Formula)</span></p>

Let us return to $I = \int_0^t B_s dB_s$ from Chapter 3. Choose $X_t = B_t$ and $g(t,x) = \frac{1}{2}x^2$. Then $Y_t = g(t, B_t) = \frac{1}{2}B_t^2$. By Itô's formula,

$$dY_t = \frac{\partial g}{\partial t}dt + \frac{\partial g}{\partial x}dB_t + \frac{1}{2}\frac{\partial^2 g}{\partial x^2}(dB_t)^2 = B_t dB_t + \frac{1}{2}(dB_t)^2 = B_t dB_t + \frac{1}{2}dt$$

Hence $d(\frac{1}{2}B_t^2) = B_t dB_t + \frac{1}{2}dt$, which gives $\frac{1}{2}B_t^2 = \int_0^t B_s dB_s + \frac{1}{2}t$, as in Chapter 3.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Computing $\int_0^t s \, dB_s$)</span></p>

What is $\int_0^t s \, dB_s$? From classical calculus it seems reasonable that a term of the form $tB_t$ should appear, so we put $g(t,x) = tx$ and $Y_t = g(t,B_t) = tB_t$. Then by Itô's formula,

$$dY_t = B_t dt + t \, dB_t + 0$$

i.e. $d(tB_t) = B_t dt + t \, dB_t$, or

$$tB_t = \int_0^t B_s ds + \int_0^t s \, dB_s$$

Therefore

$$\int_0^t s \, dB_s = tB_t - \int_0^t B_s ds$$

which is reasonable from an integration-by-parts point of view.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Integration by Parts)</span></p>

Suppose $f(s,\omega) = f(s)$ only depends on $s$ and that $f$ is continuous and of bounded variation in $[0,t]$. Then

$$\int_0^t f(s)dB_s = f(t)B_t - \int_0^t B_s \, df_s$$

Note that it is crucial for the result to hold that $f$ does not depend on $\omega$.

</div>

#### Sketch of Proof of the Itô Formula

Substituting $dX_t = u \, dt + v \, dB_t$ into the formula and using the multiplication rules, we get the equivalent expression. Using Taylor's theorem we write

$$g(t, X_t) = g(0, X_0) + \sum_j \frac{\partial g}{\partial t}\Delta t_j + \sum_j \frac{\partial g}{\partial x}\Delta X_j + \frac{1}{2}\sum_j \frac{\partial^2 g}{\partial t^2}(\Delta t_j)^2 + \sum_j \frac{\partial^2 g}{\partial t \partial x}(\Delta t_j)(\Delta X_j) + \frac{1}{2}\sum_j \frac{\partial^2 g}{\partial x^2}(\Delta X_j)^2 + \sum_j R_j$$

As $\Delta t_j \to 0$, the first sum converges to $\int_0^t \frac{\partial g}{\partial s}(s,X_s)ds$ and the second to $\int_0^t \frac{\partial g}{\partial x}(s,X_s)dX_s$. For the key term $\sum_j \frac{\partial^2 g}{\partial x^2}(\Delta X_j)^2$, since $u$ and $v$ are elementary we get

$$\sum_j \frac{\partial^2 g}{\partial x^2}(\Delta X_j)^2 = \sum_j \frac{\partial^2 g}{\partial x^2}u_j^2(\Delta t_j)^2 + 2\sum_j \frac{\partial^2 g}{\partial x^2}u_j v_j(\Delta t_j)(\Delta B_j) + \sum_j \frac{\partial^2 g}{\partial x^2}v_j^2 \cdot (\Delta B_j)^2$$

The first two terms tend to 0 as $\Delta t_j \to 0$. The last term converges to $\int_0^t \frac{\partial^2 g}{\partial x^2}v^2 ds$ in $L^2(P)$, and this is often expressed shortly by the striking formula

$$(dB_t)^2 = dt$$

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Regularity requirements)</span></p>

Note that it is enough that $g(t,x)$ is $C^2$ on $[0,\infty) \times U$, if $U \subset \mathbb{R}$ is an open set such that $X_t(\omega) \in U$ for all $t \ge 0$, $\omega \in \Omega$. Moreover, it is sufficient that $g(t,x)$ is $C^1$ w.r.t. $t$ and $C^2$ w.r.t. $x$.

</div>

### 4.2 The Multi-Dimensional Itô Formula

We now turn to the situation in higher dimensions.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-Dimensional Itô Process)</span></p>

Let $B(t,\omega) = (B_1(t,\omega), \dots, B_m(t,\omega))$ denote $m$-dimensional Brownian motion. If each of the processes $u_i(t,\omega)$ and $v_{ij}(t,\omega)$ ($1 \le i \le n$, $1 \le j \le m$) satisfies the conditions given in Definition 4.1.1, then we can form the following $n$ Itô processes

$$\begin{cases} dX_1 = u_1 dt + v_{11}dB_1 + \cdots + v_{1m}dB_m \\ \vdots \\ dX_n = u_n dt + v_{n1}dB_1 + \cdots + v_{nm}dB_m \end{cases}$$

Or, in matrix notation simply $dX(t) = u \, dt + v \, dB(t)$, where

$$X(t) = \begin{pmatrix} X_1(t) \\ \vdots \\ X_n(t) \end{pmatrix}, \quad u = \begin{pmatrix} u_1 \\ \vdots \\ u_n \end{pmatrix}, \quad v = \begin{pmatrix} v_{11} & \cdots & v_{1m} \\ \vdots & & \vdots \\ v_{n1} & \cdots & v_{nm} \end{pmatrix}, \quad dB(t) = \begin{pmatrix} dB_1(t) \\ \vdots \\ dB_m(t) \end{pmatrix}$$

Such a process $X(t)$ is called an **$n$-dimensional Itô process**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The General Itô Formula)</span></p>

Let $dX(t) = u \, dt + v \, dB(t)$ be an $n$-dimensional Itô process as above. Let $g(t,x) = (g_1(t,x), \dots, g_p(t,x))$ be a $C^2$ map from $[0,\infty) \times \mathbb{R}^n$ into $\mathbb{R}^p$. Then the process

$$Y(t,\omega) = g(t, X(t))$$

is again an Itô process, whose component number $k$, $Y_k$, is given by

$$dY_k = \frac{\partial g_k}{\partial t}(t,X)dt + \sum_i \frac{\partial g_k}{\partial x_i}(t,X)dX_i + \frac{1}{2}\sum_{i,j} \frac{\partial^2 g_k}{\partial x_i \partial x_j}(t,X)dX_i \, dX_j$$

where $dB_i \, dB_j = \delta_{ij} dt$, $dB_i \, dt = dt \, dB_i = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Bessel Process)</span></p>

Let $B = (B_1, \dots, B_n)$ be Brownian motion in $\mathbb{R}^n$, $n \ge 2$, and consider

$$R(t,\omega) = \|B(t,\omega)\| = (B_1^2(t,\omega) + \cdots + B_n^2(t,\omega))^{1/2}$$

i.e. the distance to the origin of $B(t,\omega)$. The function $g(t,x) = \|x\|$ is not $C^2$ at the origin, but since $B_t$ never hits the origin a.s. when $n \ge 2$, Itô's formula still works and we get

$$dR = \sum_{i=1}^n \frac{B_i dB_i}{R} + \frac{n-1}{2R}dt$$

The process $R$ is called the **$n$-dimensional Bessel process** because its generator is the Bessel differential operator $Af(x) = \frac{1}{2}f''(x) + \frac{n-1}{2x}f'(x)$.

</div>

### 4.3 The Martingale Representation Theorem

Let $B(t) = (B_1(t), \dots, B_n(t))$ be $n$-dimensional Brownian motion. In Chapter 3 (Corollary 3.2.6) we proved that if $v \in \mathcal{V}^n$ then the Itô integral

$$X_t = X_0 + \int_0^t v(s,\omega)dB(s); \quad t \ge 0$$

is always a martingale w.r.t. filtration $\mathcal{F}_t^{(n)}$ (and w.r.t. the probability measure $P$). In this section we prove that the converse is also true: Any $\mathcal{F}_t^{(n)}$-martingale (w.r.t. $P$) can be represented as an Itô integral. This result, called the **martingale representation theorem**, is important for many applications, for example in mathematical finance.

We first establish some auxiliary results.

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Density of Smooth Functionals)</span></p>

Fix $T > 0$. The set of random variables

$$\lbrace\phi(B_{t_1}, \dots, B_{t_n}); \; t_i \in [0,T], \; \phi \in C_0^\infty(\mathbb{R}^n), \; n = 1, 2, \dots\rbrace$$

is dense in $L^2(\mathcal{F}_T, P)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Density of Exponential Martingales)</span></p>

The linear span of random variables of the type

$$\exp\left\lbrace\int_0^T h(t)dB_t(\omega) - \frac{1}{2}\int_0^T h^2(t)dt\right\rbrace; \quad h \in L^2[0,T] \text{ (deterministic)}$$

is dense in $L^2(\mathcal{F}_T, P)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Itô Representation Theorem)</span></p>

Let $F \in L^2(\mathcal{F}_T^{(n)}, P)$. Then there exists a unique stochastic process $f(t,\omega) \in \mathcal{V}^n(0,T)$ such that

$$F(\omega) = E[F] + \int_0^T f(t,\omega)dB(t)$$

</div>

*Proof sketch (for $n=1$).* First assume that $F$ has the exponential form $F(\omega) = \exp\left\lbrace\int_0^T h(t)dB_t(\omega) - \frac{1}{2}\int_0^T h^2(t)dt\right\rbrace$ for some $h(t) \in L^2[0,T]$. Define

$$Y_t(\omega) = \exp\left\lbrace\int_0^t h(s)dB_s(\omega) - \frac{1}{2}\int_0^t h^2(s)ds\right\rbrace; \quad 0 \le t \le T$$

Then by Itô's formula, $dY_t = Y_t(h(t)dB_t - \frac{1}{2}h^2(t)dt) + \frac{1}{2}Y_t(h(t)dB_t)^2 = Y_t h(t)dB_t$, so that

$$Y_t = 1 + \int_0^t Y_s h(s) dB_s; \quad t \in [0,T]$$

Therefore $F = Y_T = 1 + \int_0^T Y_s h(s)dB_s$ and hence $E[F] = 1$. So the representation holds in this case. By linearity it also holds for linear combinations of functions of this form. If $F \in L^2(\mathcal{F}_T, P)$ is arbitrary, we approximate $F$ by linear combinations $F_n$ of the exponential form. Then for each $n$ we have $F_n(\omega) = E[F_n] + \int_0^T f_n(s,\omega)dB_s(\omega)$ where $f_n \in \mathcal{V}(0,T)$. By the Itô isometry

$$E[(F_n - F_m)^2] = (E[F_n - F_m])^2 + \int_0^T E[(f_n - f_m)^2]dt \to 0 \quad \text{as } n, m \to \infty$$

so $\lbrace f_n\rbrace$ is a Cauchy sequence converging to some $f \in \mathcal{V}(0,T)$, and

$$F = \lim_{n \to \infty} F_n = E[F] + \int_0^T f \, dB$$

Uniqueness follows from the Itô isometry: if $F = E[F] + \int_0^T f_1 \, dB_t = E[F] + \int_0^T f_2 \, dB_t$ with $f_1, f_2 \in \mathcal{V}(0,T)$, then $0 = E[(\int_0^T (f_1 - f_2)dB_t)^2] = \int_0^T E[(f_1 - f_2)^2]dt$ and therefore $f_1 = f_2$ for a.a. $(t,\omega)$.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Martingale Representation Theorem)</span></p>

Let $B(t) = (B_1(t), \dots, B_n(t))$ be $n$-dimensional. Suppose $M_t$ is an $\mathcal{F}_t^{(n)}$-martingale (w.r.t. $P$) and that $M_t \in L^2(P)$ for all $t \ge 0$. Then there exists a unique stochastic process $g(s,\omega)$ such that $g \in \mathcal{V}^{(n)}(0,t)$ for all $t \ge 0$ and

$$M_t(\omega) = E[M_0] + \int_0^t g(s,\omega)dB(s) \quad \text{a.s., for all } t \ge 0$$

</div>

*Proof (for $n=1$).* By the Itô representation theorem applied to $T = t$, $F = M_t$, we have that for all $t$ there exists a unique $f^{(t)}(s,\omega) \in L^2(\mathcal{F}_t, P)$ such that

$$M_t(\omega) = E[M_t] + \int_0^t f^{(t)}(s,\omega)dB_s(\omega) = E[M_0] + \int_0^t f^{(t)}(s,\omega)dB_s(\omega)$$

Now assume $0 \le t_1 < t_2$. Then $M_{t_1} = E[M_{t_2} \mid \mathcal{F}_{t_1}] = E[M_0] + \int_0^{t_1} f^{(t_2)}(s,\omega)dB_s(\omega)$. But we also have $M_{t_1} = E[M_0] + \int_0^{t_1} f^{(t_1)}(s,\omega)dB_s(\omega)$. Hence, comparing and using the Itô isometry, $f^{(t_1)}(s,\omega) = f^{(t_2)}(s,\omega)$ for a.a. $(s,\omega) \in [0,t_1] \times \Omega$. So we can define $f(s,\omega)$ for a.a. $s \in [0,\infty) \times \Omega$ by setting $f(s,\omega) = f^{(N)}(s,\omega)$ if $s \in [0,N]$, and then

$$M_t = E[M_0] + \int_0^t f(s,\omega)dB_s(\omega) \quad \text{for all } t \ge 0$$

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The process $f(t,\omega)$ and Malliavin derivative)</span></p>

The process $f(t,\omega)$ in the Itô representation theorem can be expressed in terms of the Frechet derivative and also in terms of the Malliavin derivative of $F(\omega)$.

</div>

## 5. Stochastic Differential Equations

### 5.1 Examples and Some Solution Methods

We now return to the possible solutions $X_t(\omega)$ of the stochastic differential equation

$$\frac{dX_t}{dt} = b(t, X_t) + \sigma(t, X_t)W_t, \quad b(t,x) \in \mathbb{R},\, \sigma(t,x) \in \mathbb{R}$$

where $W_t$ is 1-dimensional "white noise". As discussed in Chapter 3 the Itô interpretation is that $X_t$ satisfies the stochastic integral equation

$$X_t = X_0 + \int_0^t b(s, X_s)\,ds + \int_0^t \sigma(s, X_s)\,dB_s$$

or in differential form

$$dX_t = b(t, X_t)\,dt + \sigma(t, X_t)\,dB_t$$

Therefore, to get from the first form to the second we formally replace the white noise $W_t$ by $\frac{dB_t}{dt}$ and multiply by $dt$. It is natural to ask:

**(A)** Can one obtain existence and uniqueness theorems for such equations? What are the properties of the solutions?

**(B)** How can one solve a given such equation?

We will first consider question (B) by looking at some simple examples, and then in Section 5.2 we will discuss (A). It is the Itô formula that is the key to the solution of many stochastic differential equations.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(5.1.1 — Population Growth Model)</span></p>

Let us return to the population growth model:

$$\frac{dN_t}{dt} = a_t N_t, \quad N_0 \text{ given}$$

where $a_t = r_t + \alpha W_t$, $W_t = $ white noise, $\alpha = $ constant. Let us assume that $r_t = r = $ constant. By the Itô interpretation this equation is equivalent to (here $\sigma(t,x) = \alpha x$):

$$dN_t = rN_t\,dt + \alpha N_t\,dB_t$$

or

$$\frac{dN_t}{N_t} = r\,dt + \alpha\,dB_t$$

Hence

$$\int_0^t \frac{dN_s}{N_s} = rt + \alpha B_t \quad (B_0 = 0)$$

To evaluate the integral on the left hand side we use the Itô formula for the function $g(t,x) = \ln x$, $x > 0$, and obtain

$$d(\ln N_t) = \frac{1}{N_t} \cdot dN_t + \frac{1}{2}\left(-\frac{1}{N_t^2}\right)(dN_t)^2 = \frac{dN_t}{N_t} - \frac{1}{2}\alpha^2\,dt$$

Hence $\frac{dN_t}{N_t} = d(\ln N_t) + \frac{1}{2}\alpha^2\,dt$, so from the integral equation above we conclude

$$\ln \frac{N_t}{N_0} = \left(r - \tfrac{1}{2}\alpha^2\right)t + \alpha B_t$$

or

$$N_t = N_0 \exp\!\left((r - \tfrac{1}{2}\alpha^2)t + \alpha B_t\right)$$

For comparison, the *Stratonovich* interpretation would give $d\overline{N}_t = r\overline{N}_t\,dt + \alpha \overline{N}_t \circ dB_t$, with solution

$$\overline{N}_t = N_0 \exp(rt + \alpha B_t)$$

The solutions $N_t, \overline{N}_t$ are both processes of the type $X_t = X_0 \exp(\mu t + \alpha B_t)$ ($\mu, \alpha$ constants). Such processes are called **geometric Brownian motions**. They are important also as models for stochastic prices in economics.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Expected Value of $N_t$)</span></p>

It seems reasonable that if $B_t$ is independent of $N_0$ we should have $E[N_t] = E[N_0]e^{rt}$, i.e. the same as when there is no noise in $a_t$. To verify, let $Y_t = e^{\alpha B_t}$ and apply Itô's formula:

$$dY_t = \alpha e^{\alpha B_t}dB_t + \tfrac{1}{2}\alpha^2 e^{\alpha B_t}dt$$

so $Y_t = Y_0 + \alpha \int_0^t e^{\alpha B_s}\,dB_s + \tfrac{1}{2}\alpha^2 \int_0^t e^{\alpha B_s}\,ds$. Since $E[\int_0^t e^{\alpha B_s}\,dB_s] = 0$ (Theorem 3.2.1 (iii)), we get

$$E[Y_t] = E[Y_0] + \tfrac{1}{2}\alpha^2 \int_0^t E[Y_s]\,ds$$

i.e. $\frac{d}{dt}E[Y_t] = \tfrac{1}{2}\alpha^2 E[Y_t]$, $E[Y_0] = 1$. So $E[Y_t] = e^{\frac{1}{2}\alpha^2 t}$, and therefore — as anticipated — we obtain

$$E[N_t] = E[N_0]e^{rt}$$

For the Stratonovich solution, the same calculation gives $E[\overline{N}_t] = E[N_0]e^{(r + \frac{1}{2}\alpha^2)t}$.

</div>

Now that we have found the explicit solutions $N_t$ and $\overline{N}_t$, we can use our knowledge about the behaviour of $B_t$ to gain information on these solutions. For example, for the Itô solution $N_t$ we get the following:

1. If $r > \frac{1}{2}\alpha^2$ then $N_t \to \infty$ as $t \to \infty$, a.s.
2. If $r < \frac{1}{2}\alpha^2$ then $N_t \to 0$ as $t \to \infty$, a.s.
3. If $r = \frac{1}{2}\alpha^2$ then $N_t$ will fluctuate between arbitrary large and arbitrary small values as $t \to \infty$, a.s.

These conclusions are direct consequences of the formula for $N_t$ together with the following basic result about 1-dimensional Brownian motion $B_t$:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.1.2 — The Law of Iterated Logarithm)</span></p>

$$\limsup_{t \to \infty} \frac{B_t}{\sqrt{2t \log \log t}} = 1 \quad a.s.$$

</div>

For the Stratonovich solution $\overline{N}_t$ we get by the same argument that $\overline{N}_t \to 0$ a.s. if $r < 0$ and $\overline{N}_t \to \infty$ a.s. if $r > 0$. Thus the two solutions have fundamentally different properties and it is an interesting question what solution gives the best description of the situation.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(5.1.3 — RLC Circuit with Noise)</span></p>

Let us return to the equation

$$LQ_t'' + RQ_t' + \frac{1}{C}Q_t = F_t = G_t + \alpha W_t$$

We introduce the vector $X = X(t,\omega) = \begin{pmatrix} X_1 \\ X_2 \end{pmatrix} = \begin{pmatrix} Q_t \\ Q_t' \end{pmatrix}$ and obtain the system

$$\begin{cases} X_1' = X_2 \\ LX_2' = -RX_2 - \frac{1}{C}X_1 + G_t + \alpha W_t \end{cases}$$

or, in matrix notation,

$$dX = dX(t) = AX(t)\,dt + H(t)\,dt + K\,dB_t$$

where

$$dX = \begin{pmatrix} dX_1 \\ dX_2 \end{pmatrix},\; A = \begin{pmatrix} 0 & 1 \\ -\frac{1}{CL} & -\frac{R}{L} \end{pmatrix},\; H(t) = \begin{pmatrix} 0 \\ \frac{1}{L}G_t \end{pmatrix},\; K = \begin{pmatrix} 0 \\ \frac{\alpha}{L} \end{pmatrix}$$

and $B_t$ is a 1-dimensional Brownian motion. Thus we are led to a *2-dimensional stochastic differential equation*. We rewrite it as

$$\exp(-At)\,dX(t) - \exp(-At)AX(t)\,dt = \exp(-At)[H(t)\,dt + K\,dB_t]$$

Using a 2-dimensional version of the Itô formula (Theorem 4.2.1) applied to the coordinate functions of $g(t,x_1,x_2) = \exp(-At)\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$, we obtain that

$$d(\exp(-At)X(t)) = (-A)\exp(-At)X(t)\,dt + \exp(-At)\,dX(t)$$

Substituting, this gives

$$\exp(-At)X(t) - X(0) = \int_0^t \exp(-As)H(s)\,ds + \int_0^t \exp(-As)K\,dB_s$$

or

$$X(t) = \exp(At)\bigl[X(0) + \exp(-At)KB_t + \int_0^t \exp(-As)[H(s) + AKB_s]\,ds\bigr]$$

by integration by parts (Theorem 4.1.5).

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(5.1.4 — Brownian Motion on the Unit Circle)</span></p>

Choose $X = B$, 1-dimensional Brownian motion, and $g(t,x) = e^{ix} = (\cos x, \sin x) \in \mathbb{R}^2$ for $x \in \mathbb{R}$. Then

$$Y = g(t, X) = e^{iB} = (\cos B, \sin B)$$

is by Itô's formula again an Itô process. Its coordinates $Y_1, Y_2$ satisfy

$$\begin{cases} dY_1(t) = -\sin(B)\,dB - \tfrac{1}{2}\cos(B)\,dt \\ dY_2(t) = \cos(B)\,dB - \tfrac{1}{2}\sin(B)\,dt \end{cases}$$

Thus the process $Y = (Y_1, Y_2)$, which we could call *Brownian motion on the unit circle*, is the solution of the stochastic differential equations

$$\begin{cases} dY_1 = -\tfrac{1}{2}Y_1\,dt - Y_2\,dB \\ dY_2 = -\tfrac{1}{2}Y_2\,dt + Y_1\,dB \end{cases}$$

Or, in matrix notation,

$$dY = -\frac{1}{2}Y\,dt + KY\,dB, \quad \text{where } K = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

</div>

### 5.2 An Existence and Uniqueness Result

We now turn to the existence and uniqueness question (A) above.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.2.1 — Existence and Uniqueness for SDEs)</span></p>

Let $T > 0$ and $b(\cdot,\cdot): [0,T] \times \mathbb{R}^n \to \mathbb{R}^n$, $\sigma(\cdot,\cdot): [0,T] \times \mathbb{R}^n \to \mathbb{R}^{n \times m}$ be measurable functions satisfying

$$\lvert b(t,x)\rvert + \lvert \sigma(t,x)\rvert \le C(1 + \lvert x\rvert); \quad x \in \mathbb{R}^n,\, t \in [0,T]$$

for some constant $C$, (where $\lvert\sigma\rvert^2 = \sum \lvert\sigma_{ij}\rvert^2$) and such that

$$\lvert b(t,x) - b(t,y)\rvert + \lvert\sigma(t,x) - \sigma(t,y)\rvert \le D\lvert x - y\rvert; \quad x,y \in \mathbb{R}^n,\, t \in [0,T]$$

for some constant $D$. Let $Z$ be a random variable which is independent of the $\sigma$-algebra $\mathcal{F}_\infty^{(m)}$ generated by $B_s(\cdot)$, $s \ge 0$ and such that $E[\lvert Z\rvert^2] < \infty$.

Then the stochastic differential equation

$$dX_t = b(t, X_t)\,dt + \sigma(t, X_t)\,dB_t, \quad 0 \le t \le T,\; X_0 = Z$$

has a unique $t$-continuous solution $X_t(\omega)$ with the property that $X_t(\omega)$ is adapted to the filtration $\mathcal{F}_t^Z$ generated by $Z$ and $B_s(\cdot)$; $s \le t$ and

$$E\!\left[\int_0^T \lvert X_t\rvert^2\,dt\right] < \infty$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On the Growth and Lipschitz Conditions)</span></p>

The linear growth condition and the Lipschitz condition are natural in view of the following two simple examples from deterministic differential equations (i.e. $\sigma = 0$):

**a)** The equation $\frac{dX_t}{dt} = X_t^2$, $X_0 = 1$ (corresponding to $b(x) = x^2$, which does not satisfy the linear growth condition) has the unique solution $X_t = \frac{1}{1 - t}$, $0 \le t < 1$. Thus it is impossible to find a global solution in this case. More generally, the linear growth condition ensures that the solution does not *explode*, i.e. that $\lvert X_t(\omega)\rvert$ does not tend to $\infty$ in a finite time.

**b)** The equation $\frac{dX_t}{dt} = 3X_t^{2/3}$, $X_0 = 0$ has more than one solution. In fact, for any $a > 0$ the function $X_t = \begin{cases} 0 & \text{for } t \le a \\ (t-a)^3 & \text{for } t > a \end{cases}$ solves it. In this case $b(x) = 3x^{2/3}$ does not satisfy the Lipschitz condition at $x = 0$. Thus the Lipschitz condition guarantees that the equation has a **unique** solution.

</div>

*Proof of Theorem 5.2.1.* **Uniqueness.** The uniqueness follows from the Itô isometry (Corollary 3.1.7) and the Lipschitz property: Let $X_1(t,\omega) = X_t(\omega)$ and $X_2(t,\omega) = \hat{X}_t(\omega)$ be solutions with initial values $Z, \hat{Z}$ respectively. Put $a(s,\omega) = b(s, X_s) - b(s, \hat{X}_s)$ and $\gamma(s,\omega) = \sigma(s, X_s) - \sigma(s, \hat{X}_s)$. Then

$$E[\lvert X_t - \hat{X}_t\rvert^2] = E\!\left[\left(Z - \hat{Z} + \int_0^t a\,ds + \int_0^t \gamma\,dB_s\right)^2\right]$$

$$\le 3E[\lvert Z - \hat{Z}\rvert^2] + 3tE\!\left[\int_0^t a^2\,ds\right] + 3E\!\left[\int_0^t \gamma^2\,ds\right]$$

$$\le 3E[\lvert Z - \hat{Z}\rvert^2] + 3(1+t)D^2 \int_0^t E[\lvert X_s - \hat{X}_s\rvert^2]\,ds$$

So the function $v(t) = E[\lvert X_t - \hat{X}_t\rvert^2]$ satisfies $v(t) \le F + A\int_0^t v(s)\,ds$ where $F = 3E[\lvert Z - \hat{Z}\rvert^2]$ and $A = 3(1+T)D^2$. By the Gronwall inequality (Exercise 5.17) we conclude that

$$v(t) \le F\exp(At)$$

Now assume that $Z = \hat{Z}$. Then $F = 0$ and so $v(t) = 0$ for all $t \ge 0$. Hence $P[\lvert X_t - \hat{X}_t\rvert = 0 \text{ for all } t \in \mathbb{Q} \cap [0,T]] = 1$. By continuity of $t \to \lvert X_t - \hat{X}_t\rvert$ it follows that

$$P[\lvert X_1(t,\omega) - X_2(t,\omega)\rvert = 0 \text{ for all } t \in [0,T]] = 1$$

and the uniqueness is proved.

**Existence.** The proof of the existence is similar to the familiar existence proof for ordinary differential equations: Define $Y_t^{(0)} = X_0$ and $Y_t^{(k)} = Y_t^{(k)}(\omega)$ inductively as follows

$$Y_t^{(k+1)} = X_0 + \int_0^t b(s, Y_s^{(k)})\,ds + \int_0^t \sigma(s, Y_s^{(k)})\,dB_s$$

Then, similar computation as for the uniqueness above gives

$$E[\lvert Y_t^{(k+1)} - Y_t^{(k)}\rvert^2] \le (1+T)3D^2 \int_0^t E[\lvert Y_s^{(k)} - Y_s^{(k-1)}\rvert^2]\,ds$$

for $k \ge 1$, $t \le T$, and by induction on $k$ we obtain

$$E[\lvert Y_t^{(k+1)} - Y_t^{(k)}\rvert^2] \le \frac{A_2^{k+1} t^{k+1}}{(k+1)!}; \quad k \ge 0,\, t \in [0,T]$$

for some suitable constant $A_2$ depending only on $C, D, T$ and $E[\lvert X_0\rvert^2]$. By the martingale inequality (Theorem 3.2.4) and the Borel–Cantelli lemma,

$$P\!\left[\sup_{0 \le t \le T} \lvert Y_t^{(k+1)} - Y_t^{(k)}\rvert > 2^{-k} \text{ for infinitely many } k\right] = 0$$

Thus, for a.a. $\omega$ there exists $k_0 = k_0(\omega)$ such that $\sup_{0 \le t \le T}\lvert Y_t^{(k+1)} - Y_t^{(k)}\rvert \le 2^{-k}$ for $k \ge k_0$. Therefore the sequence

$$Y_t^{(n)}(\omega) = Y_t^{(0)}(\omega) + \sum_{k=0}^{n-1}(Y_t^{(k+1)}(\omega) - Y_t^{(k)}(\omega))$$

is uniformly convergent in $[0,T]$, for a.a. $\omega$. Denote the limit by $X_t = X_t(\omega)$. Then $X_t$ is $t$-continuous for a.a. $\omega$ since $Y_t^{(n)}$ is $t$-continuous for all $n$. Moreover, $X_t(\cdot)$ is $\mathcal{F}_t^Z$-measurable for all $t$, since $Y_t^{(n)}(\cdot)$ has this property for all $n$.

Next, note that for $m > n \ge 0$ we have by the induction estimate

$$E[\lvert Y_t^{(m)} - Y_t^{(n)}\rvert^2]^{1/2} \le \sum_{k=n}^{m-1}\left[\frac{(A_2 t)^{k+1}}{(k+1)!}\right]^{1/2} \to 0 \quad \text{as } n \to \infty$$

So $\lbrace Y_t^{(n)}\rbrace$ converges in $L^2(P)$ to a limit $Y_t$. A subsequence of $Y_t^{(n)}(\omega)$ will then converge $\omega$-pointwise to $Y_t(\omega)$ and therefore we must have $Y_t = X_t$ a.s. In particular, $X_t$ satisfies the integrability condition.

It remains to show that $X_t$ satisfies the SDE. For all $n$ we have

$$Y_t^{(n+1)} = X_0 + \int_0^t b(s, Y_s^{(n)})\,ds + \int_0^t \sigma(s, Y_s^{(n)})\,dB_s$$

Now $Y_t^{(n+1)} \to X_t$ as $n \to \infty$, uniformly in $t \in [0,T]$ for a.a. $\omega$. By the Fatou lemma we have $E[\int_0^T \lvert X_t - Y_t^{(n)}\rvert^2\,dt] \to 0$. It follows by the Itô isometry that $\int_0^t \sigma(s, Y_s^{(n)})\,dB_s \to \int_0^t \sigma(s, X_s)\,dB_s$ and by the Hölder inequality that $\int_0^t b(s, Y_s^{(n)})\,ds \to \int_0^t b(s, X_s)\,ds$ in $L^2(P)$. Therefore, taking the limit we obtain the SDE for $X_t$. $\square$

### 5.3 Weak and Strong Solutions

The solution $X_t$ found above is called a **strong solution**, because the version $B_t$ of Brownian motion is given in advance and the solution $X_t$ constructed from it is $\mathcal{F}_t^Z$-adapted. If we are only given the functions $b(t,x)$ and $\sigma(t,x)$ and ask for a pair of processes $((\tilde{X}_t, \tilde{B}_t), \mathcal{H}_t)$ on a probability space $(\Omega, \mathcal{H}, P)$ such that the SDE holds, then the solution $\tilde{X}_t$ (or more precisely $(\tilde{X}_t, \tilde{B}_t)$) is called a **weak solution**. Here $\mathcal{H}_t$ is an increasing family of $\sigma$-algebras such that $\tilde{X}_t$ is $\mathcal{H}_t$-adapted and $\tilde{B}_t$ is an $\mathcal{H}_t$-Brownian motion, i.e. $\tilde{B}_t$ is a Brownian motion, and $\tilde{B}_t$ is a martingale w.r.t. $\mathcal{H}_t$.

A strong solution is of course also a weak solution, but the converse is not true in general.

The uniqueness that we obtain from Theorem 5.2.1 is called **strong** or **pathwise** uniqueness, while **weak uniqueness** simply means that any two solutions (weak or strong) are identical in law, i.e. have the same finite-dimensional distributions.

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(5.3.1 — Weak Uniqueness under Lipschitz Conditions)</span></p>

If $b$ and $\sigma$ satisfy the conditions of Theorem 5.2.1 then we have: A solution (weak or strong) of the SDE is weakly unique.

</div>

*Sketch of proof.* Let $((\tilde{X}_t, \tilde{B}_t), \tilde{\mathcal{H}}_t)$ and $((\hat{X}_t, \hat{B}_t), \hat{\mathcal{H}}_t)$ be two weak solutions. Let $X_t$ and $Y_t$ be the strong solutions constructed from $\tilde{B}_t$ and $\hat{B}_t$, respectively, as above. Then the same uniqueness argument applies to show that $X_t = \tilde{X}_t$ and $Y_t = \hat{X}_t$ for all $t$, a.s. Therefore it suffices to show that $X_t$ and $Y_t$ must be identical in law. We show this by proving by induction that if $X_t^{(k)}, Y_t^{(k)}$ are the processes in the Picard iteration defined by the iterative scheme with Brownian motions $\tilde{B}_t$ and $\hat{B}_t$, then $(X_t^{(k)}, \tilde{B}_t)$ and $(Y_t^{(k)}, \hat{B}_t)$ have the same law for all $k$. $\square$

From a modelling point of view the weak solution concept is often natural, because it does not specify beforehand the explicit representation of the white noise. Moreover, the concept is convenient for mathematical reasons, because there are stochastic differential equations which have *no strong solutions* but still a (weakly) *unique weak solution*.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(5.3.2 — The Tanaka Equation)</span></p>

Consider the 1-dimensional stochastic differential equation

$$dX_t = \operatorname{sign}(X_t)\,dB_t; \quad X_0 = 0$$

where $\operatorname{sign}(x) = \begin{cases} +1 & \text{if } x \ge 0 \\ -1 & \text{if } x < 0 \end{cases}$.

Note that $\sigma(t,x) = \sigma(x) = \operatorname{sign}(x)$ does not satisfy the Lipschitz condition, so Theorem 5.2.1 does not apply. Indeed, **the equation has no strong solution**.

To see this, let $\hat{B}_t$ be a Brownian motion generating the filtration $\hat{\mathcal{F}}_t$ and define $Y_t = \int_0^t \operatorname{sign}(\hat{B}_s)\,d\hat{B}_s$. By the Tanaka formula (Exercise 4.10) we have $Y_t = \lvert\hat{B}_t\rvert - \lvert\hat{B}_0\rvert - \hat{L}_t(\omega)$ where $\hat{L}_t(\omega)$ is the local time for $\hat{B}_t(\omega)$ at 0. It follows that $Y_t$ is measurable w.r.t. the $\sigma$-algebra $\mathcal{G}_t$ generated by $\lvert\hat{B}_s\rvert$; $s \le t$, which is clearly strictly contained in $\hat{\mathcal{F}}_t$. Hence the $\sigma$-algebra $\mathcal{N}_t$ generated by $Y_s(\cdot)$; $s \le t$ is also strictly contained in $\hat{\mathcal{F}}_t$.

Now suppose $X_t$ is a strong solution. Then by Theorem 8.4.2 it follows that $X_t$ is a Brownian motion w.r.t. the measure $P$. Let $\mathcal{M}_t$ be the $\sigma$-algebra generated by $X_s(\cdot)$; $s \le t$. Since $(\operatorname{sign}(x))^2 = 1$ we can rewrite the equation as $dB_t = \operatorname{sign}(X_t)\,dX_t$. By the above argument applied to $\hat{B}_t = X_t$, $Y_t = B_t$ we conclude that $\mathcal{F}_t$ is strictly contained in $\mathcal{M}_t$. But this contradicts that $X_t$ is a strong solution. Hence strong solutions of the Tanaka equation do not exist.

To find a **weak solution**, we simply choose $X_t$ to be *any* Brownian motion $\hat{B}_t$. Then we define $\tilde{B}_t = \int_0^t \operatorname{sign}(X_s)\,dX_s$, i.e. $d\tilde{B}_t = \operatorname{sign}(X_t)\,dX_t$. Then $dX_t = \operatorname{sign}(X_t)\,d\tilde{B}_t$, so $X_t$ is a weak solution.

Finally, **weak uniqueness** follows from Theorem 8.4.2, which implies that any weak solution $X_t$ must be a Brownian motion w.r.t. $P$.

</div>

## 6. The Filtering Problem

### 6.1 Introduction

Problem 3 in the introduction is a special case of the following general *filtering problem*:

Suppose the state $X_t \in \mathbb{R}^n$ at time $t$ of a system is given by a stochastic differential equation

$$\frac{dX_t}{dt} = b(t, X_t) + \sigma(t, X_t)W_t, \quad t \ge 0$$

where $b: \mathbb{R}^{n+1} \to \mathbb{R}^n$, $\sigma: \mathbb{R}^{n+1} \to \mathbb{R}^{n \times p}$ satisfy the conditions (5.2.1), (5.2.2) and $W_t$ is $p$-dimensional white noise. The Itô interpretation of this equation is

$$dX_t = b(t, X_t)\,dt + \sigma(t, X_t)\,dU_t$$

where $U_t$ is $p$-dimensional Brownian motion. We also assume that the distribution of $X_0$ is known and independent of $U_t$. Similarly to the 1-dimensional situation (Theorem 3.3.6) there is an explicit several-dimensional formula which expresses the *Stratonovich* interpretation:

$$dX_t = \tilde{b}(t, X_t)\,dt + \sigma(t, X_t)\,dU_t, \quad \text{where}$$

$$\tilde{b}_i(t,x) = b_i(t,x) + \frac{1}{2}\sum_{j=1}^{p}\sum_{k=1}^{n}\frac{\partial \sigma_{ij}}{\partial x_k}\sigma_{kj}; \quad 1 \le i \le n$$

From now on we will use the Itô interpretation.

In the continuous version of the filtering problem we assume that the observations $H_t \in \mathbb{R}^m$ are performed continuously and are of the form

$$H_t = c(t, X_t) + \gamma(t, X_t) \cdot \widetilde{W}_t$$

where $c: \mathbb{R}^{n+1} \to \mathbb{R}^m$, $\gamma: \mathbb{R}^{n+1} \to \mathbb{R}^{m \times r}$ are functions satisfying (5.2.1) and $\widetilde{W}_t$ denotes $r$-dimensional white noise, independent of $U_t$ and $X_0$. To obtain a tractable mathematical interpretation we introduce

$$Z_t = \int_0^t H_s\,ds$$

and thereby obtain the stochastic integral representation

$$dZ_t = c(t, X_t)\,dt + \gamma(t, X_t)\,dV_t, \quad Z_0 = 0$$

where $V_t$ is $r$-dimensional Brownian motion, independent of $U_t$ and $X_0$.

**The filtering problem** is the following: Given the observations $Z_s$ satisfying the observation equation for $0 \le s \le t$, what is the best estimate $\hat{X}_t$ of the state $X_t$ of the system based on these observations?

By saying that the estimate $\hat{X}_t$ is *based on the observations* $\lbrace Z_s;\, s \le t\rbrace$ we mean that

$$\hat{X}_t(\cdot) \text{ is } \mathcal{G}_t\text{-measurable, where } \mathcal{G}_t \text{ is the } \sigma\text{-algebra generated by } \lbrace Z_s(\cdot),\, s \le t\rbrace$$

By saying that $\hat{X}_t$ is the *best* such estimate we mean that

$$\int_\Omega \lvert X_t - \hat{X}_t\rvert^2\,dP = E[\lvert X_t - \hat{X}_t\rvert^2] = \inf\lbrace E[\lvert X_t - Y\rvert^2];\, Y \in \mathcal{K}\rbrace$$

where $\mathcal{K} = \mathcal{K}_t = \mathcal{K}(Z,t) := \lbrace Y: \Omega \to \mathbb{R}^n;\, Y \in L^2(P) \text{ and } Y \text{ is } \mathcal{G}_t\text{-measurable}\rbrace$.

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.1.1 — Conditional Expectation as Projection)</span></p>

Let $\mathcal{H} \subset \mathcal{F}$ be a $\sigma$-algebra and let $X \in L^2(P)$ be $\mathcal{F}$-measurable. Put $\mathcal{N} = \lbrace Y \in L^2(P);\, Y \text{ is } \mathcal{H}\text{-measurable}\rbrace$ and let $\mathcal{P}_\mathcal{N}$ denote the (orthogonal) projection from the Hilbert space $L^2(P)$ into the subspace $\mathcal{N}$. Then

$$\mathcal{P}_\mathcal{N}(X) = E[X \mid \mathcal{H}]$$

</div>

*Proof.* Recall that $E[X \mid \mathcal{H}]$ is by definition the $P$-unique function from $\Omega$ to $\mathbb{R}$ such that (i) $E[X \mid \mathcal{H}]$ is $\mathcal{H}$-measurable and (ii) $\int_A E[X \mid \mathcal{H}]\,dP = \int_A X\,dP$ for all $A \in \mathcal{H}$. Now $\mathcal{P}_\mathcal{N}(X)$ is $\mathcal{H}$-measurable and $\int_\Omega Y(X - \mathcal{P}_\mathcal{N}(X))\,dP = 0$ for all $Y \in \mathcal{N}$. In particular, $\int_A (X - \mathcal{P}_\mathcal{N}(X))\,dP = 0$ for all $A \in \mathcal{H}$, i.e. $\int_A \mathcal{P}_\mathcal{N}(X)\,dP = \int_A X\,dP$ for all $A \in \mathcal{H}$. Hence, by uniqueness, $\mathcal{P}_\mathcal{N}(X) = E[X \mid \mathcal{H}]$. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(6.1.2 — Optimal Filter as Conditional Expectation)</span></p>

$$\hat{X}_t = \mathcal{P}_{\mathcal{K}_t}(X_t) = E[X_t \mid \mathcal{G}_t]$$

</div>

This is the basis for the general Fujisaki–Kallianpur–Kunita equation of filtering theory.

### 6.2 The 1-Dimensional Linear Filtering Problem

From now on we will concentrate on the linear case, which allows an explicit solution in terms of a stochastic differential equation for $\hat{X}_t$ (the *Kalman–Bucy filter*).

In the *linear* filtering problem the system and observation equations have the form:

$$\text{(linear system)} \quad dX_t = F(t)X_t\,dt + C(t)\,dU_t; \quad F(t) \in \mathbb{R}^{n \times n},\, C(t) \in \mathbb{R}^{n \times p}$$

$$\text{(linear observations)} \quad dZ_t = G(t)X_t\,dt + D(t)\,dV_t; \quad G(t) \in \mathbb{R}^{m \times n},\, D(t) \in \mathbb{R}^{m \times r}$$

To focus on the main ideas, we first consider the 1-dimensional case:

$$\text{(linear system)} \quad dX_t = F(t)X_t\,dt + C(t)\,dU_t; \quad F(t),\, C(t) \in \mathbb{R}$$

$$\text{(linear observations)} \quad dZ_t = G(t)X_t\,dt + D(t)\,dV_t; \quad G(t),\, D(t) \in \mathbb{R}$$

We assume (see (5.2.1)) that $F, G, C, D$ are bounded on bounded intervals. Based on our interpretation of $Z_t$ we assume $Z_0 = 0$. We also assume that $X_0$ is normally distributed (and independent of $\lbrace U_t\rbrace$, $\lbrace V_t\rbrace$). Finally we assume that $D(t)$ is bounded away from 0 on bounded intervals.

Here is an outline of the solution of the filtering problem in this case.

**Step 1.** Let $\mathcal{L} = \mathcal{L}(Z,t)$ be the closure in $L^2(P)$ of functions which are linear combinations of the form $c_0 + c_1 Z_{s_1}(\omega) + \cdots + c_k Z_{s_k}(\omega)$ with $s_j \le t$, $c_j \in \mathbb{R}$. Let $\mathcal{P}_\mathcal{L}$ denote the projection from $L^2(P)$ onto $\mathcal{L}$. Then $\hat{X}_t = \mathcal{P}_\mathcal{K}(X_t) = E[X_t \mid \mathcal{G}_t] = \mathcal{P}_\mathcal{L}(X_t)$. Thus, the best $Z$-measurable estimate of $X_t$ coincides with the best $Z$-linear estimate of $X_t$.

**Step 2.** Replace $Z_t$ by the *innovation process* $N_t$:

$$N_t = Z_t - \int_0^t (GX)_s^\wedge\,ds, \quad \text{where } (GX)_s^\wedge = \mathcal{P}_{\mathcal{L}(Z,s)}(G(s)X_s) = G(s)\hat{X}_s$$

i.e. $dN_t = G(t)(X_t - \hat{X}_t)\,dt + D(t)\,dV_t$.

Then: (i) $N_t$ has orthogonal increments; (ii) $\mathcal{L}(N,t) = \mathcal{L}(Z,t)$, so $\hat{X}_t = \mathcal{P}_{\mathcal{L}(N,t)}(X_t)$.

**Step 3.** If we put $dR_t = \frac{1}{D(t)}dN_t$, then $R_t$ is a 1-dimensional Brownian motion. Moreover, $\mathcal{L}(N,t) = \mathcal{L}(R,t)$ and

$$\hat{X}_t = \mathcal{P}_{\mathcal{L}(N,t)}(X_t) = \mathcal{P}_{\mathcal{L}(R,t)}(X_t) = E[X_t] + \int_0^t \frac{\partial}{\partial s}E[X_t R_s]\,dR_s$$

**Step 4.** Find an explicit expression for $X_t$ by solving the linear SDE $dX_t = F(t)X_t\,dt + C(t)\,dU_t$. The result is

$$X_t = \exp\!\left(\int_0^t F(s)\,ds\right)\!\left[X_0 + \int_0^t \exp\!\left(-\int_0^s F(u)\,du\right)C(s)\,dU_s\right]$$

**Step 5.** Substitute the formula for $X_t$ from Step 4 into $E[X_t R_s]$ and use Step 3 to obtain a stochastic differential equation for $\hat{X}_t$.

#### Step 1. $Z$-Linear and $Z$-Measurable Estimates

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.2.2 — Linear vs Measurable Estimates for Gaussian Systems)</span></p>

Let $X, Z_s$; $s \le t$ be random variables in $L^2(P)$ and assume that $(X, Z_{s_1}, Z_{s_2}, \dots, Z_{s_n}) \in \mathbb{R}^{n+1}$ has a normal distribution for all $s_1, s_2, \dots, s_n \le t$, $n \ge 1$. Then

$$\mathcal{P}_\mathcal{L}(X) = E[X \mid \mathcal{G}] = \mathcal{P}_\mathcal{K}(X)$$

In other words, the best $Z$-linear estimate for $X$ coincides with the best $Z$-measurable estimate in this case.

</div>

*Proof.* Put $\check{X} = \mathcal{P}_\mathcal{L}(X)$, $\widetilde{X} = X - \check{X}$. Then we claim that $\widetilde{X}$ is independent of $\mathcal{G}$: Recall that a random variable $(Y_1, \dots, Y_k) \in \mathbb{R}^k$ is normal iff $c_1Y_1 + \cdots + c_kY_k$ is normal, for all choices of $c_1, \dots, c_k \in \mathbb{R}$. And an $L^2$-limit of normal variables is again normal. Therefore $(\widetilde{X}, Z_{s_1}, \dots, Z_{s_n})$ is normal for all $s_1, \dots, s_n \le t$. Since $E[\widetilde{X}Z_{s_j}] = 0$, $\widetilde{X}$ and $Z_{s_j}$ are uncorrelated, for $1 \le j \le n$. It follows that $\widetilde{X}$ and $(Z_{s_1}, \dots, Z_{s_n})$ are independent. So $\widetilde{X}$ is independent from $\mathcal{G}$ as claimed. But then $E[\mathcal{X}_G(X - \check{X})] = E[\mathcal{X}_G\widetilde{X}] = E[\mathcal{X}_G] \cdot E[\widetilde{X}] = 0$ for all $G \in \mathcal{G}$, i.e. $\int_G X\,dP = \int_G \check{X}\,dP$. Since $\check{X}$ is $\mathcal{G}$-measurable, we conclude that $\check{X} = E[X \mid \mathcal{G}]$. $\square$

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Normal Distribution and Worst-Case Estimation)</span></p>

There is a curious interpretation of this result: Suppose $X, \lbrace Z_t\rbrace_{t \in T}$ are $L^2(P)$-functions with given covariances. Among all possible distributions of $(X, Z_{t_1}, \dots, Z_{t_n})$ with these covariances, the *normal* distribution will be the "worst" w.r.t. estimation, in the following sense: For any distribution we have $E[(X - E[X \mid \mathcal{G}])^2] \le E[(X - \mathcal{P}_\mathcal{L}(X))^2]$, with *equality* for the normal distribution, by Lemma 6.2.2.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.2.3 — Gaussianity of $(X_t, Z_t)$)</span></p>

$M_t = \begin{bmatrix} X_t \\ Z_t \end{bmatrix} \in \mathbb{R}^2$ is a Gaussian process.

</div>

*Proof.* We may regard $M_t$ as the solution of a 2-dimensional linear SDE of the form $dM_t = H(t)M_t\,dt + K(t)\,dB_t$, $M_0 = \begin{bmatrix} X_0 \\ 0 \end{bmatrix}$, where $H(t) \in \mathbb{R}^{2 \times 2}$, $K(t) \in \mathbb{R}^{2 \times 2}$ and $B_t$ is 2-dimensional Brownian motion. Use Picard iteration: $M_t^{(n)}$ is Gaussian for all $n$ and $M_t^{(n)} \to M_t$ in $L^2(P)$ (see the proof of Theorem 5.2.1) and therefore $M_t$ is Gaussian. $\square$

#### Step 2. The Innovation Process

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Innovation Process)</span></p>

The **innovation process** $N_t$ is defined as follows:

$$N_t = Z_t - \int_0^t (GX)_s^\wedge\,ds, \quad \text{where } (GX)_s^\wedge = \mathcal{P}_{\mathcal{L}(Z,s)}(G(s)X_s) = G(s)\hat{X}_s$$

i.e. $dN_t = G(t)(X_t - \hat{X}_t)\,dt + D(t)\,dV_t$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.2.4 — Representation of $\mathcal{L}(Z,T)$)</span></p>

$\mathcal{L}(Z,T) = \lbrace c_0 + \int_0^T f(t)\,dZ_t;\, f \in L^2[0,T],\, c_0 \in \mathbb{R}\rbrace$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.2.5 — Properties of the Innovation Process)</span></p>

**(i)** $N_t$ has orthogonal increments.

**(ii)** $E[N_t^2] = \int_0^t D^2(s)\,ds$.

**(iii)** $\mathcal{L}(N,t) = \mathcal{L}(Z,t)$ for all $t \ge 0$.

**(iv)** $N_t$ is a Gaussian process.

</div>

*Proof.* (i): If $s < t$ and $Y \in \mathcal{L}(Z,s)$ we have

$$E[(N_t - N_s)Y] = E\!\left[\left(\int_s^t G(r)(X_r - \hat{X}_r)\,dr + \int_s^t D(r)\,dV_r\right)Y\right] = 0$$

since $X_r - \hat{X}_r \perp \mathcal{L}(Z,s)$ for $r \ge s$ and $V$ has independent increments.

(ii): By Itô's formula, $d(N_t^2) = 2N_t\,dN_t + D^2\,dt$. So $E[N_t^2] = E[\int_0^t 2N_s\,dN_s] + \int_0^t D^2(s)\,ds$. Now $\int_0^t N_s\,dN_s = \lim_{\Delta t_j \to 0} \sum N_{t_j}[N_{t_{j+1}} - N_{t_j}]$, so since $N$ has orthogonal increments, $E[\int_0^t N_s\,dN_s] = 0$, and (ii) follows.

(iii): It is clear that $\mathcal{L}(N,t) \subset \mathcal{L}(Z,t)$ for all $t \ge 0$. To establish the opposite inclusion we use Lemma 6.2.4 and the theory of Volterra integral equations to show that for all $h \in L^2[0,t]$ there exists $f \in L^2[0,t]$ such that $Z_{t_1} = \int_0^t \mathcal{X}_{[0,t_1]}(s)\,dZ_s$ can be recovered from $\int_0^t f(r)c(r)\,dr + \int_0^t f(s)\,dN_s$, which shows $\mathcal{L}(N,t) \supset \mathcal{L}(Z,t)$.

(iv): $\hat{X}_t$ is a limit in $L^2(P)$ of linear combinations of the form $c_0 + c_1 Z_{s_1} + \cdots + c_k Z_{s_k}$ with $s_k \le t$. Therefore $(\hat{X}_{t_1}, \dots, \hat{X}_{t_m})$ is a limit of $m$-dimensional random variables whose components are linear combinations of this form and hence Gaussian. It follows that $N_t = Z_t - \int_0^t G(s)\hat{X}_s\,ds$ is Gaussian, by a similar argument. $\square$

#### Step 3. The Innovation Process and Brownian Motion

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.2.6 — $R_t$ is Brownian Motion)</span></p>

Let $N_t = Z_t - \int_0^t G(s)\hat{X}_s\,ds$ be the innovation process. Define

$$dR_t = \frac{1}{D(t)}\,dN_t; \quad t \ge 0,\, R_0 = 0$$

Then $R_t$ is a 1-dimensional Brownian motion.

</div>

*Proof.* Observe that (i) $R_t$ has continuous paths; (ii) $R_t$ has orthogonal increments (since $N_t$ has); (iii) $R_t$ is Gaussian (since $N_t$ is); (iv) $E[R_t] = 0$ and $E[R_t R_s] = \min(s,t)$. For (iv), by Itô's formula $d(R_t^2) = 2R_t\,dR_t + dt$, so $E[R_t^2] = t$ since $R_t$ has orthogonal increments. Then for $s < t$: $E[R_t R_s] = E[(R_t - R_s)R_s] + E[R_s^2] = E[R_s^2] = s$. Properties (i), (iii), (iv) constitute one of the many characterizations of a 1-dimensional Brownian motion. $\square$

Since $\mathcal{L}(N,t) = \mathcal{L}(R,t)$, we conclude that $\hat{X}_t = \mathcal{P}_{\mathcal{L}(R,t)}(X_t)$.

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.2.7 — Representation of $\hat{X}_t$)</span></p>

$$\hat{X}_t = E[X_t] + \int_0^t \frac{\partial}{\partial s}E[X_t R_s]\,dR_s$$

</div>

*Proof.* From Lemma 6.2.4 we know that $\hat{X}_t = c_0(t) + \int_0^t g(s)\,dR_s$ for some $g \in L^2[0,t]$, $c_0(t) \in \mathbb{R}$. Taking expectations we see that $c_0(t) = E[\hat{X}_t] = E[X_t]$. We have $(X_t - \hat{X}_t) \perp \int_0^t f(s)\,dR_s$ for all $f \in L^2[0,t]$. Therefore

$$E\!\left[X_t \int_0^t f(s)\,dR_s\right] = E\!\left[\hat{X}_t \int_0^t f(s)\,dR_s\right] = E\!\left[\int_0^t g(s)\,dR_s \int_0^t f(s)\,dR_s\right] = \int_0^t g(s)f(s)\,ds$$

by the Itô isometry. In particular, choosing $f = \mathcal{X}_{[0,r]}$ we obtain $E[X_t R_r] = \int_0^r g(s)\,ds$, or $g(r) = \frac{\partial}{\partial r}E[X_t R_r]$, as asserted. $\square$

#### Step 4. An Explicit Formula for $X_t$

This is easily obtained using Itô's formula, as in the examples in Chapter 5. The result is

$$X_t = \exp\!\left(\int_0^t F(s)\,ds\right)\!\left[X_0 + \int_0^t \exp\!\left(-\int_0^s F(u)\,du\right)C(s)\,dU_s\right]$$

More generally, if $0 \le r \le t$:

$$X_t = \exp\!\left(\int_r^t F(s)\,ds\right)X_r + \int_r^t \exp\!\left(\int_s^t F(u)\,du\right)C(s)\,dU_s$$

#### Step 5. The Stochastic Differential Equation for $\hat{X}_t$

We now combine the previous steps to obtain the solution of the filtering problem, i.e. a stochastic differential equation for $\hat{X}_t$. Starting with the formula from Lemma 6.2.7:

$$\hat{X}_t = E[X_t] + \int_0^t f(s,t)\,dR_s$$

where $f(s,t) = \frac{\partial}{\partial s}E[X_t R_s]$, we use that $R_s = \int_0^s \frac{G(r)}{D(r)}(X_r - \hat{X}_r)\,dr + V_s$ and the explicit formula for $X_t$ to obtain

$$E[X_t R_s] = \int_0^s \frac{G(r)}{D(r)}\exp\!\left(\int_r^t F(v)\,dv\right)S(r)\,dr$$

where $S(r) = E[(\widetilde{X}_r)^2] = E[(X_r - \hat{X}_r)^2]$ is **the mean square error of the estimate** at time $r \ge 0$. Thus

$$f(s,t) = \frac{G(s)}{D(s)}\exp\!\left(\int_s^t F(v)\,dv\right)S(s)$$

We claim that $S(t)$ satisfies the (deterministic) **Riccati equation**:

$$\frac{dS}{dt} = 2F(t)S(t) - \frac{G^2(t)}{D^2(t)}S^2(t) + C^2(t), \quad S(0) = E[(X_0 - E[X_0])^2]$$

Differentiating the expression for $\hat{X}_t$ and substituting back, we arrive at:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(6.2.8 — The 1-Dimensional Kalman–Bucy Filter)</span></p>

The solution $\hat{X}_t = E[X_t \mid \mathcal{G}_t]$ of the 1-dimensional linear filtering problem

$$\text{(linear system)} \quad dX_t = F(t)X_t\,dt + C(t)\,dU_t; \quad F(t),\, C(t) \in \mathbb{R}$$

$$\text{(linear observations)} \quad dZ_t = G(t)X_t\,dt + D(t)\,dV_t; \quad G(t),\, D(t) \in \mathbb{R}$$

(with conditions as stated earlier) satisfies the stochastic differential equation

$$d\hat{X}_t = \left(F(t) - \frac{G^2(t)S(t)}{D^2(t)}\right)\hat{X}_t\,dt + \frac{G(t)S(t)}{D^2(t)}\,dZ_t; \quad \hat{X}_0 = E[X_0]$$

where $S(t) = E[(X_t - \hat{X}_t)^2]$ satisfies the (deterministic) Riccati equation

$$\frac{dS}{dt} = 2F(t)S(t) - \frac{G^2(t)}{D^2(t)}S^2(t) + C^2(t), \quad S(0) = E[(X_0 - E[X_0])^2]$$

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(6.2.9 — Noisy Observations of a Constant Process)</span></p>

Consider the simple case $dX_t = 0$, i.e. $X_t = X_0$; $E[X_0] = 0$, $E[X_0^2] = a^2$, with observations $dZ_t = X_t\,dt + m\,dV_t$; $Z_0 = 0$ (corresponding to $H_t = X_t + mW_t$, white noise).

The Riccati equation becomes $\frac{dS}{dt} = -\frac{1}{m^2}S^2$, $S(0) = a^2$, giving $S(t) = \frac{a^2 m^2}{m^2 + a^2 t}$; $t \ge 0$.

This gives the equation for $\hat{X}_t$: $d\hat{X}_t = -\frac{a^2}{m^2 + a^2 t}\hat{X}_t\,dt + \frac{a^2}{m^2 + a^2 t}\,dZ_t$; $\hat{X}_0 = 0$,

which yields $\hat{X}_t = \frac{a^2}{m^2 + a^2 t}Z_t$; $t \ge 0$. This is the continuous analogue of Example 6.2.1.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(6.2.10 — Noisy Observations of a Brownian Motion)</span></p>

If we modify the preceding example slightly, so that $dX_t = c\,dU_t$; $E[X_0] = 0$, $E[X_0^2] = a^2$, $c$ constant, with observations $dZ_t = X_t\,dt + m\,dV_t$, the Riccati equation becomes

$$\frac{dS}{dt} = -\frac{1}{m^2}S^2 + c^2, \quad S(0) = a^2$$

In all cases the mean square error $S(t)$ tends to $mc$ as $t \to \infty$. For simplicity let us put $a = 0$, $m = c = 1$. Then $S(t) = \tanh(t)$.

The equation for $\hat{X}_t$ is $d\hat{X}_t = -\tanh(t)\hat{X}_t\,dt + \tanh(t)\,dZ_t$; $\hat{X}_0 = 0$, which gives

$$\hat{X}_t = \frac{1}{\cosh(t)}\int_0^t \sinh(s)\,dZ_s$$

If we return to the interpretation of $Z_t = \int_0^t H_s\,ds$, where $H_s$ are the "original" observations, we can write

$$\hat{X}_t = \frac{1}{\cosh(t)}\int_0^t \sinh(s)H_s\,ds$$

so $\hat{X}_t$ is approximately (for large $t$) a weighted average of the observations $H_s$, with increasing emphasis on observations as time increases.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Comparison with Exponentially Weighted Moving Average)</span></p>

It is interesting to compare formula (6.2.31) with established formulas in forecasting. For example, the *exponentially weighted moving average* $\widetilde{X}_n$ suggested by C.C. Holt in 1958 is given by $\widetilde{X}_n = (1-\alpha)^n Z_0 + \alpha \sum_{k=1}^n (1-\alpha)^{n-k}Z_k$, where $\alpha$ is a constant; $0 \le \alpha \le 1$. This may be written $\widetilde{X}_n = \beta^{-n}Z_0 + (\beta - 1)\beta^{-n-1}\sum_{k=1}^n \beta^k Z_k$ where $\beta = \frac{1}{1-\alpha}$ (assuming $\alpha < 1$), which is a discrete version of the continuous Kalman–Bucy filter formula.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(6.2.11 — Estimation of a Parameter)</span></p>

Suppose we want to estimate the value of a (constant) parameter $\theta$, based on observations $Z_t$ satisfying the model $dZ_t = \theta M(t)\,dt + N(t)\,dB_t$, where $M(t), N(t)$ are known functions. In this case the stochastic differential equation for $\theta$ is of course $d\theta = 0$, so the Riccati equation for $S(t) = E[(\theta - \hat{\theta}_t)^2]$ is

$$\frac{dS}{dt} = -\left(\frac{M(t)S(t)}{N(t)}\right)^2$$

which gives $S(t) = \left(S_0^{-1} + \int_0^t M(s)^2 N(s)^{-2}\,ds\right)^{-1}$ and the Kalman–Bucy filter is

$$\hat{\theta}_t = \frac{\hat{\theta}_0 S_0^{-1} + \int_0^t M(s)N(s)^{-2}\,dZ_s}{S_0^{-1} + \int_0^t M(s)^2 N(s)^{-2}\,ds}$$

This estimate coincides with the maximum likelihood estimate in classical estimation theory if $S_0^{-1} = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(6.2.12 — Noisy Observations of a Population Growth)</span></p>

Consider a simple growth model ($r$ constant) $dX_t = rX_t\,dt$, $E[X_0] = b > 0$, $E[(X_0 - b)^2] = a^2$, with observations $dZ_t = X_t\,dt + m\,dV_t$; $m$ constant.

The corresponding Riccati equation $\frac{dS}{dt} = 2rS - \frac{1}{m^2}S^2$, $S(0) = a^2$, gives the logistic curve

$$S(t) = \frac{2rm^2}{1 + Ke^{-2rt}}; \quad \text{where } K = \frac{2rm^2}{a^2} - 1$$

So the equation for $\hat{X}_t$ becomes $d\hat{X}_t = \left(r - \frac{S}{m^2}\right)\hat{X}_t\,dt + \frac{S}{m^2}\,dZ_t$; $\hat{X}_0 = b$.

For simplicity let us assume $a^2 = 2rm^2$, so that $S(t) = 2rm^2$ for all $t$. Then $d(\exp(rt)\hat{X}_t) = \exp(rt)2r\,dZ_t$, or

$$\hat{X}_t = \exp(-rt)\!\left[\int_0^t 2r\exp(rs)\,dZ_s + b\right]$$

As in Example 6.2.10 this may be written $\hat{X}_t = \exp(-rt)\left[\int_0^t 2r\exp(rs)H_s\,ds + b\right]$ if $Z_t = \int_0^t H_s\,ds$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(6.2.13 — Constant Coefficients)</span></p>

Now consider the system $dX_t = FX_t\,dt + C\,dU_t$; $F, C$ constants $\ne 0$, with observations $dZ_t = GX_t\,dt + D\,dV_t$; $G, D$ constants $\ne 0$. The corresponding Riccati equation $S' = 2FS - \frac{G^2}{D^2}S^2 + C^2$, $S(0) = a^2$ has the solution

$$S(t) = \frac{\alpha_1 - K\alpha_2 \exp\!\left(\frac{(\alpha_2 - \alpha_1)G^2}{D^2}t\right)}{1 - K\exp\!\left(\frac{(\alpha_2 - \alpha_1)G^2}{D^2}t\right)}$$

where $\alpha_1 = G^{-2}(FD^2 - D\sqrt{F^2D^2 + G^2C^2})$, $\alpha_2 = G^{-2}(FD^2 + D\sqrt{F^2D^2 + G^2C^2})$, and $K = \frac{a^2 - \alpha_1}{a^2 - \alpha_2}$. For large $s$ we have $S(s) \approx \alpha_2$. This gives

$$\hat{X}_t \approx \hat{X}_0 \exp(-\beta t) + \frac{G\alpha_2}{D^2}\exp(-\beta t)\int_0^t \exp(\beta s)\,dZ_s$$

where $\beta = D^{-1}\sqrt{F^2D^2 + G^2C^2}$.

</div>

### 6.3 The Multidimensional Linear Filtering Problem

Finally we formulate the solution of the $n$-dimensional linear filtering problem:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(6.3.1 — The Multi-Dimensional Kalman–Bucy Filter)</span></p>

The solution $\hat{X}_t = E[X_t \mid \mathcal{G}_t]$ of the multi-dimensional linear filtering problem

$$\text{(linear system)} \quad dX_t = F(t)X_t\,dt + C(t)\,dU_t; \quad F(t) \in \mathbb{R}^{n \times n},\, C(t) \in \mathbb{R}^{n \times p}$$

$$\text{(linear observations)} \quad dZ_t = G(t)X_t\,dt + D(t)\,dV_t; \quad G(t) \in \mathbb{R}^{m \times n},\, D(t) \in \mathbb{R}^{m \times r}$$

satisfies the stochastic differential equation

$$d\hat{X}_t = (F - SG^T(DD^T)^{-1}G)\hat{X}_t\,dt + SG^T(DD^T)^{-1}\,dZ_t; \quad \hat{X}_0 = E[X_0]$$

where $S(t) := E[(X_t - \hat{X}_t)(X_t - \hat{X}_t)^T] \in \mathbb{R}^{n \times n}$ satisfies the **matrix Riccati equation**

$$\frac{dS}{dt} = FS + SF^T - SG^T(DD^T)^{-1}GS + CC^T; \quad S(0) = E[(X_0 - E[X_0])(X_0 - E[X_0])^T]$$

The condition on $D(t) \in \mathbb{R}^{m \times r}$ is now that $D(t)D(t)^T$ is invertible for all $t$ and that $(D(t)D(t)^T)^{-1}$ is bounded on every bounded $t$-interval.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Extensions and Nonlinear Filtering)</span></p>

A similar solution can be found for the more general situation where the system and observation equations contain additional terms depending on $Z_t$:

$$dX_t = [F_0(t) + F_1(t)X_t + F_2(t)Z_t]\,dt + C(t)\,dU_t$$

$$dZ_t = [G_0(t) + G_1(t)X_t + G_2(t)Z_t]\,dt + D(t)\,dV_t$$

An account of non-linear filtering theory is also given in Pardoux (1979) and Davis (1984).

</div>

## 7. Diffusions: Basic Properties

The solution of a stochastic differential equation may be thought of as the mathematical description of the motion of a small particle in a moving fluid. Such stochastic processes are called *(Itô) diffusions*. In this chapter we establish the most basic properties and results about Itô diffusions.

In a stochastic differential equation of the form

$$dX_t = b(t, X_t)\,dt + \sigma(t, X_t)\,dB_t$$

where $X_t \in \mathbb{R}^n$, $b(t,x) \in \mathbb{R}^n$, $\sigma(t,x) \in \mathbb{R}^{n \times m}$ and $B_t$ is $m$-dimensional Brownian motion, we call $b$ the **drift coefficient** and $\sigma$ — or sometimes $\tfrac{1}{2}\sigma\sigma^T$ — the **diffusion coefficient** (see Theorem 7.3.3).

### 7.1 The Markov Property

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.1.1 — Time-Homogeneous Itô Diffusion)</span></p>

A *(time-homogeneous)* Itô diffusion is a stochastic process $X_t(\omega) = X(t,\omega)\colon [0,\infty) \times \Omega \to \mathbb{R}^n$ satisfying a stochastic differential equation of the form

$$dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t, \quad t \ge s;\quad X_s = x$$

where $B_t$ is $m$-dimensional Brownian motion and $b\colon \mathbb{R}^n \to \mathbb{R}^n$, $\sigma\colon \mathbb{R}^n \to \mathbb{R}^{n \times m}$ satisfy the conditions in Theorem 5.2.1, which in this case simplify to:

$$\lvert b(x) - b(y)\rvert + \lvert \sigma(x) - \sigma(y)\rvert \le D\lvert x - y\rvert; \quad x, y \in \mathbb{R}^n$$

where $\lvert \sigma \rvert^2 = \sum \lvert \sigma_{ij}\rvert^2$.

</div>

We denote the (unique) solution by $X_t = X_t^{s,x}$; $t \ge s$. If $s = 0$ we write $X_t^x$ for $X_t^{0,x}$. Note that $b$ and $\sigma$ do not depend on $t$ but only on $x$. The resulting process $X_t(\omega)$ is **time-homogeneous**: the processes $\lbrace X_{s+h}^{s,x}\rbrace_{h \ge 0}$ and $\lbrace X_h^{0,x}\rbrace_{h \ge 0}$ have the same $P^0$-distributions.

We introduce the probability laws $Q^x$ of $\lbrace X_t\rbrace_{t \ge 0}$ for $x \in \mathbb{R}^n$. Intuitively, $Q^x$ gives the distribution of $\lbrace X_t\rbrace_{t \ge 0}$ assuming that $X_0 = x$:

$$Q^x[X_{t_1} \in E_1, \ldots, X_{t_k} \in E_k] = P^0[X_{t_1}^x \in E_1, \ldots, X_{t_k}^x \in E_k]$$

where $E_i \subset \mathbb{R}^n$ are Borel sets.

As before let $\mathcal{F}_t^{(m)}$ be the $\sigma$-algebra generated by $\lbrace B_r;\, r \le t\rbrace$. Similarly we let $\mathcal{M}_t$ be the $\sigma$-algebra generated by $\lbrace X_r;\, r \le t\rbrace$. We have established earlier (Theorem 5.2.1) that $X_t$ is measurable with respect to $\mathcal{F}_t^{(m)}$, so $\mathcal{M}_t \subseteq \mathcal{F}_t^{(m)}$.

The **Markov property** states that the future behaviour of the process given what has happened up to time $t$ is the same as the behaviour obtained when starting the process at $X_t$.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.1.2 — The Markov Property for Itô Diffusions)</span></p>

Let $f$ be a bounded Borel function from $\mathbb{R}^n$ to $\mathbb{R}$. Then, for $t, h \ge 0$,

$$E^x[f(X_{t+h}) \mid \mathcal{F}_t^{(m)}](\omega) = E^{X_t(\omega)}[f(X_h)]$$

Here $E^x$ denotes the expectation w.r.t. the probability measure $Q^x$. Thus $E^y[f(X_h)]$ means $E[f(X_h^y)]$, where $E$ denotes the expectation w.r.t. $P^0$. The right hand side means the function $E^y[f(X_h)]$ evaluated at $y = X_t(\omega)$.

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Theorem 7.1.2)</summary>

Since for $r \ge t$, $X_r(\omega) = X_t(\omega) + \int_t^r b(X_u)\,du + \int_t^r \sigma(X_u)\,dB_u$, by uniqueness $X_r(\omega) = X_r^{t, X_t}(\omega)$. Defining $F(x,t,r,\omega) = X_r^{t,x}(\omega)$ for $r \ge t$, we have $X_r(\omega) = F(X_t, t, r, \omega)$ for $r \ge t$. Since $\omega \to F(x,t,r,\omega)$ is independent of $\mathcal{F}_t^{(m)}$, the result follows by the properties of conditional expectation (approximating $g(x,\omega) = f \circ F(x,t,t+h,\omega)$ by functions of the form $\sum \phi_k(x)\psi_k(\omega)$) and using time-homogeneity.

</details>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Theorem 7.1.2 states that $X_t$ is a Markov process w.r.t. the family of $\sigma$-algebras $\lbrace\mathcal{F}_t^{(m)}\rbrace_{t \ge 0}$. Since $\mathcal{M}_t \subseteq \mathcal{F}_t^{(m)}$, this implies that $X_t$ is also a Markov process w.r.t. $\lbrace\mathcal{M}_t\rbrace_{t \ge 0}$:

$$E^x[f(X_{t+h}) \mid \mathcal{M}_t] = E^{X_t}[f(X_h)]$$

</div>

### 7.2 The Strong Markov Property

The strong Markov property states that the Markov relation continues to hold if the deterministic time $t$ is replaced by a random time $\tau(\omega)$ of a type called a **stopping time** (or **Markov time**).

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.2.1 — Stopping Time)</span></p>

Let $\lbrace\mathcal{N}_t\rbrace$ be an increasing family of $\sigma$-algebras (of subsets of $\Omega$). A function $\tau\colon \Omega \to [0, \infty]$ is called a *(strict)* **stopping time** w.r.t. $\lbrace\mathcal{N}_t\rbrace$ if

$$\lbrace\omega;\, \tau(\omega) \le t\rbrace \in \mathcal{N}_t, \quad \text{for all } t \ge 0$$

In other words, it should be possible to decide whether or not $\tau \le t$ has occurred on the basis of the knowledge of $\mathcal{N}_t$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(7.2.2 — First Exit Time)</span></p>

Let $U \subset \mathbb{R}^n$ be open. Then the **first exit time**

$$\tau_U := \inf\lbrace t > 0;\, X_t \notin U\rbrace$$

is a stopping time w.r.t. $\lbrace\mathcal{M}_t\rbrace$. More generally, if $H \subset \mathbb{R}^n$ is any set, the first exit time from $H$ is $\tau_H = \inf\lbrace t > 0;\, X_t \notin H\rbrace$.

If we include the sets of measure $0$ in $\mathcal{M}_t$ (which we do) then $\mathcal{M}_t$ is right-continuous, i.e. $\mathcal{M}_t = \mathcal{M}_{t+} = \bigcap_{s > t} \mathcal{M}_s$, and therefore $\tau_H$ is a stopping time for any Borel set $H$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.2.3 — $\sigma$-algebra $\mathcal{N}_\tau$)</span></p>

Let $\tau$ be a stopping time w.r.t. $\lbrace\mathcal{N}_t\rbrace$ and let $\mathcal{N}_\infty$ be the smallest $\sigma$-algebra containing $\mathcal{N}_t$ for all $t \ge 0$. Then the $\sigma$-algebra $\mathcal{N}_\tau$ consists of all sets $N \in \mathcal{N}_\infty$ such that

$$N \cap \lbrace\tau \le t\rbrace \in \mathcal{N}_t \quad \text{for all } t \ge 0$$

In the case $\mathcal{N}_t = \mathcal{M}_t$: $\mathcal{M}_\tau$ = the $\sigma$-algebra generated by $\lbrace X_{\min(s,\tau)};\, s \ge 0\rbrace$. Similarly, if $\mathcal{N}_t = \mathcal{F}_t^{(m)}$, we get $\mathcal{F}_\tau^{(m)}$ = the $\sigma$-algebra generated by $\lbrace B_{s \wedge \tau};\, s \ge 0\rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.2.4 — The Strong Markov Property for Itô Diffusions)</span></p>

Let $f$ be a bounded Borel function on $\mathbb{R}^n$, $\tau$ a stopping time w.r.t. $\mathcal{F}_t^{(m)}$, $\tau < \infty$ a.s. Then

$$E^x[f(X_{\tau+h}) \mid \mathcal{F}_\tau^{(m)}] = E^{X_\tau}[f(X_h)] \quad \text{for all } h \ge 0$$

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Theorem 7.2.4)</summary>

By the strong Markov property for Brownian motion, $\widetilde{B}_v = B_{\tau + v} - B_\tau$ ($v \ge 0$) is again a Brownian motion independent of $\mathcal{F}_\tau^{(m)}$. Therefore $\lbrace X_{\tau+h}^{\tau,x}\rbrace_{h \ge 0}$ must coincide a.e. with the strongly unique solution $Y_h$ of $Y_h = x + \int_0^h b(Y_v)\,dv + \int_0^h \sigma(Y_v)\,d\widetilde{B}_v$. Since $\lbrace Y_h\rbrace$ is independent of $\mathcal{F}_\tau^{(m)}$, we conclude that $\lbrace X_{\tau+h}^{\tau,x}\rbrace_{h \ge 0}$ has the same law as $\lbrace X_h^{0,x}\rbrace_{h \ge 0}$. Using the same approximation technique as in Theorem 7.1.2, the result follows.

</details>

This extends to multiple time points: if $f_1, \ldots, f_k$ are bounded Borel functions on $\mathbb{R}^n$, $\tau$ an $\mathcal{F}_t^{(m)}$-stopping time with $\tau < \infty$ a.s., then for $0 \le h_1 \le h_2 \le \cdots \le h_k$:

$$E^x[f_1(X_{\tau+h_1})f_2(X_{\tau+h_2})\cdots f_k(X_{\tau+h_k}) \mid \mathcal{F}_\tau^{(m)}] = E^{X_\tau}[f_1(X_{h_1})\cdots f_k(X_{h_k})]$$

**The shift operator.** For $t \ge 0$ define $\theta_t\colon \mathcal{H} \to \mathcal{H}$ where $\mathcal{H}$ is the set of all $\mathcal{M}_\infty$-measurable functions. If $\eta = g_1(X_{t_1})\cdots g_k(X_{t_k})$ ($g_i$ Borel measurable, $t_i \ge 0$) we put $\theta_t \eta = g_1(X_{t_1+t})\cdots g_k(X_{t_k+t})$. It follows that

$$E^x[\theta_\tau \eta \mid \mathcal{F}_\tau^{(m)}] = E^{X_\tau}[\eta]$$

for all stopping times $\tau$ and all bounded $\eta \in \mathcal{H}$.

#### Hitting Distribution, Harmonic Measure and the Mean Value Property

Let $H \subset \mathbb{R}^n$ be measurable and let $\tau_H$ be the first exit time from $H$ for an Itô diffusion $X_t$. Let $\alpha$ be another stopping time, $g$ a bounded continuous function on $\mathbb{R}^n$. Put

$$\eta = g(X_{\tau_H})\mathcal{X}_{\lbrace\tau_H < \infty\rbrace}, \quad \tau_H^\alpha = \inf\lbrace t > \alpha;\, X_t \notin H\rbrace$$

Then

$$\theta_\alpha \eta \cdot \mathcal{X}_{\lbrace\alpha < \infty\rbrace} = g(X_{\tau_H^\alpha})\mathcal{X}_{\lbrace\tau_H^\alpha < \infty\rbrace}$$

In particular, if $\alpha = \tau_G$ with $G \subset\subset H$ measurable, $\tau_H < \infty$ a.s. $Q^x$, then $\tau_H^\alpha = \tau_H$ and so

$$\theta_{\tau_G} g(X_{\tau_H}) = g(X_{\tau_H})$$

Using the strong Markov property, for any bounded measurable $f$:

$$E^x[f(X_{\tau_H})] = E^x[E^{X_{\tau_G}}[f(X_{\tau_H})]] = \int_{\partial G} E^y[f(X_{\tau_H})] \cdot Q^x[X_{\tau_G} \in dy]$$

for $x \in G$.

Define **the harmonic measure** of $X$ on $\partial G$, $\mu_G^x$, by

$$\mu_G^x(F) = Q^x[X_{\tau_G} \in F] \quad \text{for } F \subset \partial G, \, x \in G$$

Then the function

$$\phi(x) = E^x[f(X_{\tau_H})]$$

satisfies the **mean value property**:

$$\phi(x) = \int_{\partial G} \phi(y)\,d\mu_G^x(y), \quad \text{for all } x \in G$$

for all Borel sets $G \subset\subset H$. This is an important ingredient in the solution of the generalized Dirichlet problem.

### 7.3 The Generator of an Itô Diffusion

It is fundamental for many applications that we can associate a second order partial differential operator $A$ to an Itô diffusion $X_t$. The basic connection is that $A$ is the **generator** of the process $X_t$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.3.1 — Generator of an Itô Diffusion)</span></p>

Let $\lbrace X_t\rbrace$ be a (time-homogeneous) Itô diffusion in $\mathbb{R}^n$. The (infinitesimal) generator $A$ of $X_t$ is defined by

$$Af(x) = \lim_{t \downarrow 0} \frac{E^x[f(X_t)] - f(x)}{t}; \quad x \in \mathbb{R}^n$$

The set of functions $f\colon \mathbb{R}^n \to \mathbb{R}$ such that the limit exists at $x$ is denoted by $\mathcal{D}_A(x)$, while $\mathcal{D}_A$ denotes the set of functions for which the limit exists for all $x \in \mathbb{R}^n$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(7.3.2)</span></p>

Let $Y_t = Y_t^x$ be an Itô process in $\mathbb{R}^n$ of the form

$$Y_t^x(\omega) = x + \int_0^t u(s,\omega)\,ds + \int_0^t v(s,\omega)\,dB_s(\omega)$$

where $B$ is $m$-dimensional. Let $f \in C_0^2(\mathbb{R}^n)$ (i.e. $f \in C^2(\mathbb{R}^n)$ with compact support), and let $\tau$ be a stopping time w.r.t. $\lbrace\mathcal{F}_t^{(m)}\rbrace$, with $E^x[\tau] < \infty$. Assume that $u(t,\omega)$ and $v(t,\omega)$ are bounded on the set of $(t,\omega)$ such that $Y(t,\omega)$ belongs to the support of $f$. Then

$$E^x[f(Y_\tau)] = f(x) + E^x\left[\int_0^\tau \left(\sum_i u_i(s,\omega)\frac{\partial f}{\partial x_i}(Y_s) + \tfrac{1}{2}\sum_{i,j}(vv^T)_{i,j}(s,\omega)\frac{\partial^2 f}{\partial x_i \partial x_j}(Y_s)\right)ds\right]$$

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Lemma 7.3.2)</summary>

Apply Itô's formula to $Z = f(Y)$. Writing $Y_i$, $B_k$ for the coordinates of $Y$ and $B$ respectively:

$$dZ = \sum_i \frac{\partial f}{\partial x_i}(Y)\,dY_i + \tfrac{1}{2}\sum_{i,j}\frac{\partial^2 f}{\partial x_i \partial x_j}(Y)\,dY_i\,dY_j$$

Using $(vdB)_i \cdot (vdB)_j = (vv^T)_{ij}\,dt$ we get

$$f(Y_t) = f(Y_0) + \int_0^t \left(\sum_i u_i \frac{\partial f}{\partial x_i}(Y) + \tfrac{1}{2}\sum_{i,j}(vv^T)_{i,j}\frac{\partial^2 f}{\partial x_i \partial x_j}(Y)\right)ds + \sum_{i,k}\int_0^t v_{ik}\frac{\partial f}{\partial x_i}(Y)\,dB_k$$

The Itô integral terms vanish in expectation (using that $E^x[\int_0^\tau g(Y_s)\,dB_s] = 0$ for bounded $g$ by a stopping argument).

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.3.3 — Formula for the Generator)</span></p>

Let $X_t$ be the Itô diffusion $dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t$. If $f \in C_0^2(\mathbb{R}^n)$ then $f \in \mathcal{D}_A$ and

$$Af(x) = \sum_i b_i(x)\frac{\partial f}{\partial x_i} + \tfrac{1}{2}\sum_{i,j}(\sigma\sigma^T)_{i,j}(x)\frac{\partial^2 f}{\partial x_i \partial x_j}$$

</div>

This follows directly from Lemma 7.3.2 (with $\tau = t$) and the definition of $A$.

We let $L = L_X$ denote the differential operator given by the right hand side of Theorem 7.3.3. From the theorem, $A_X$ and $L_X$ coincide on $C_0^2(\mathbb{R}^n)$.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(7.3.4 — Generator of Brownian Motion)</span></p>

The $n$-dimensional Brownian motion is the solution of $dX_t = dB_t$, i.e. $b = 0$ and $\sigma = I_n$. The generator of $B_t$ is

$$Af = \tfrac{1}{2}\sum_i \frac{\partial^2 f}{\partial x_i^2}; \quad f = f(x_1,\ldots,x_n) \in C_0^2(\mathbb{R}^n)$$

i.e. $A = \tfrac{1}{2}\Delta$, where $\Delta$ is the Laplace operator.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(7.3.5 — The Graph of Brownian Motion)</span></p>

Let $B$ denote 1-dimensional Brownian motion and let $X = \binom{X_1}{X_2}$ be the solution of the stochastic differential equation

$$dX_1 = dt;\; X_1(0) = t_0 \quad \text{and} \quad dX_2 = dB;\; X_2(0) = x_0$$

i.e. $dX = b\,dt + \sigma\,dB$ with $b = \binom{1}{0}$ and $\sigma = \binom{0}{1}$. In other words, $X$ may be regarded as the graph of Brownian motion. The generator $A$ of $X$ is

$$Af = \frac{\partial f}{\partial t} + \tfrac{1}{2}\frac{\partial^2 f}{\partial x^2}; \quad f = f(t,x) \in C_0^2(\mathbb{R}^n)$$

</div>

### 7.4 The Dynkin Formula

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.4.1 — Dynkin's Formula)</span></p>

Let $f \in C_0^2(\mathbb{R}^n)$. Suppose $\tau$ is a stopping time, $E^x[\tau] < \infty$. Then

$$E^x[f(X_\tau)] = f(x) + E^x\left[\int_0^\tau Af(X_s)\,ds\right]$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

1. If $\tau$ is the first exit time of a bounded set, $E^x[\tau] < \infty$, then Dynkin's formula holds for any function $f \in C^2$.
2. For a more general version of Theorem 7.4.1 see Dynkin (1965 I), p. 133.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(7.4.2 — Expected Exit Time and Recurrence/Transience of Brownian Motion)</span></p>

**Expected exit time from a ball.** Consider $n$-dimensional Brownian motion $B = (B_1, \ldots, B_n)$ starting at $a = (a_1,\ldots,a_n) \in \mathbb{R}^n$ ($n \ge 1$) with $\lvert a\rvert < R$. The expected first exit time $\tau_K$ of $B$ from the ball $K = \lbrace x \in \mathbb{R}^n;\, \lvert x\rvert < R\rbrace$ is

$$E^a[\tau_K] = \frac{1}{n}(R^2 - \lvert a\rvert^2)$$

This is obtained by applying Dynkin's formula with $X = B$, $\tau = \sigma_k = \min(k, \tau_K)$, and $f(x) = \lvert x\rvert^2$, noting that $\Delta f = 2n$ so $\tfrac{1}{2}\Delta f = n$.

**Recurrence and transience.** Next assume $n \ge 2$ and $\lvert b\rvert > R$. What is the probability that $B$ starting at $b$ ever hits $K$?

- **$n = 2$:** Brownian motion is **recurrent** in $\mathbb{R}^2$: $P^b[T_K < \infty] = 1$.
- **$n > 2$:** Brownian motion is **transient** in $\mathbb{R}^n$ for $n > 2$: $\lim_{k \to \infty} p_k = P^b[T_K < \infty] = \left(\frac{\lvert b\rvert}{R}\right)^{2-n}$.

These results are derived by considering the first exit time $\alpha_k$ from annuli $A_k = \lbrace x;\, R < \lvert x\rvert < 2^k R\rbrace$ and applying Dynkin's formula with appropriate functions ($f(x) = -\log\lvert x\rvert$ for $n = 2$ and $f(x) = \lvert x\rvert^{2-n}$ for $n > 2$) that are harmonic in $A_k$.

</div>

### 7.5 The Characteristic Operator

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.5.1 — Characteristic Operator)</span></p>

Let $\lbrace X_t\rbrace$ be an Itô diffusion. The **characteristic operator** $\mathcal{A} = \mathcal{A}_X$ of $\lbrace X_t\rbrace$ is defined by

$$\mathcal{A}f(x) = \lim_{U \downarrow x} \frac{E^x[f(X_{\tau_U})] - f(x)}{E^x[\tau_U]}$$

where the $U$'s are open sets $U_k$ decreasing to the point $x$, in the sense that $U_{k+1} \subset U_k$ and $\bigcap_k U_k = \lbrace x\rbrace$, and $\tau_U = \inf\lbrace t > 0;\, X_t \notin U\rbrace$ is the first exit time from $U$ for $X_t$. The set of functions $f$ such that this limit exists for all $x \in \mathbb{R}^n$ (and all $\lbrace U_k\rbrace$) is denoted by $\mathcal{D}_\mathcal{A}$. If $E^x[\tau_U] = \infty$ for all open $U \ni x$, we define $\mathcal{A}f(x) = 0$.

</div>

It turns out that $\mathcal{D}_A \subseteq \mathcal{D}_\mathcal{A}$ always and that $\mathcal{A}f = Af$ for all $f \in \mathcal{D}_A$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.5.2 — Trap)</span></p>

A point $x \in \mathbb{R}^n$ is called a **trap** for $\lbrace X_t\rbrace$ if

$$Q^x(\lbrace X_t = x \text{ for all } t\rbrace) = 1$$

In other words, $x$ is a trap if and only if $\tau_{\lbrace x\rbrace} = \infty$ a.s. $Q^x$. For example, if $b(x_0) = \sigma(x_0) = 0$, then $x_0$ is a trap for $X_t$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(7.5.3)</span></p>

If $x$ is not a trap for $X_t$, then there exists an open set $U \ni x$ such that $E^x[\tau_U] < \infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.5.4 — Characteristic Operator Equals the Differential Operator)</span></p>

Let $f \in C^2$. Then $f \in \mathcal{D}_\mathcal{A}$ and

$$\mathcal{A}f = \sum_i b_i \frac{\partial f}{\partial x_i} + \tfrac{1}{2}\sum_{i,j}(\sigma\sigma^T)_{i,j}\frac{\partial^2 f}{\partial x_i \partial x_j}$$

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Theorem 7.5.4)</summary>

Let $L$ denote the operator defined by the right hand side. If $x$ is a trap for $\lbrace X_t\rbrace$ then $\mathcal{A}f(x) = 0$. Choose a bounded open set $V$ with $x \in V$, modify $f$ to $f_0$ outside $V$ so that $f_0 \in C_0^2(\mathbb{R}^n)$. Then $0 = Af_0(x) = Lf_0(x) = Lf(x)$. If $x$ is not a trap, choose a bounded open set $U \ni x$ with $E^x[\tau_U] < \infty$. By Dynkin's formula:

$$\left\lvert\frac{E^x[f(X_\tau)] - f(x)}{E^x[\tau]} - Lf(x)\right\rvert \le \sup_{y \in U}\lvert Lf(x) - Lf(y)\rvert \to 0 \text{ as } U \downarrow x$$

since $Lf$ is a continuous function.

</details>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

We have now obtained that an Itô diffusion is a continuous, strong Markov process such that the domain of its characteristic operator includes $C^2$. Thus an Itô diffusion is a *diffusion* in the sense of Dynkin (1965 I).

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(7.5.5 — Brownian Motion on the Unit Circle)</span></p>

The characteristic operator of the process $Y = \binom{Y_1}{Y_2}$ from Example 5.1.4 satisfying

$$dY_1 = -\tfrac{1}{2}Y_1\,dt - Y_2\,dB, \quad dY_2 = -\tfrac{1}{2}Y_2\,dt + Y_1\,dB$$

is

$$\mathcal{A}f(y_1, y_2) = \tfrac{1}{2}\left[y_2^2\frac{\partial^2 f}{\partial y_1^2} - 2y_1 y_2\frac{\partial^2 f}{\partial y_1 \partial y_2} + y_1^2\frac{\partial^2 f}{\partial y_2^2} - y_1\frac{\partial f}{\partial y_1} - y_2\frac{\partial f}{\partial y_2}\right]$$

This is because $dY = -\tfrac{1}{2}Y\,dt + KY\,dB$ where $K = \begin{pmatrix}0 & -1\\1 & 0\end{pmatrix}$, so $b(y_1,y_2) = \binom{-\frac{1}{2}y_1}{-\frac{1}{2}y_2}$, $\sigma(y_1,y_2) = \binom{-y_2}{y_1}$, and $a = \tfrac{1}{2}\sigma\sigma^T = \tfrac{1}{2}\begin{pmatrix}y_2^2 & -y_1 y_2\\-y_1 y_2 & y_1^2\end{pmatrix}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(7.5.6 — X-Harmonic Extension)</span></p>

Let $D$ be an open subset of $\mathbb{R}^n$ such that $\tau_D < \infty$ a.s. $Q^x$ for all $x$. Let $\phi$ be a bounded, measurable function on $\partial D$ and define

$$\widetilde{\phi}(x) = E^x[\phi(X_{\tau_D})]$$

($\widetilde{\phi}$ is called the $X$-harmonic extension of $\phi$.) Then if $U$ is open, $x \in U \subset\subset D$, we have by the strong Markov property:

$$E^x[\widetilde{\phi}(X_{\tau_U})] = E^x[E^{X_{\tau_U}}[\phi(X_{\tau_D})]] = E^x[\phi(X_{\tau_D})] = \widetilde{\phi}(x)$$

So $\widetilde{\phi} \in \mathcal{D}_\mathcal{A}$ and $\mathcal{A}\widetilde{\phi} = 0$ in $D$, in spite of the fact that in general $\widetilde{\phi}$ need not even be continuous in $D$.

</div>

## 8. Other Topics in Diffusion Theory

In this chapter we study some other important topics in diffusion theory and related areas. Some of these topics are not strictly necessary for the remaining chapters, but they are all central in the theory of stochastic analysis and essential for further applications.

### 8.1 Kolmogorov's Backward Equation. The Resolvent

Let $X_t$ be an Itô diffusion in $\mathbb{R}^n$ with generator $A$. If we choose $f \in C_0^2(\mathbb{R}^n)$ and $\tau = t$ in Dynkin's formula (Theorem 7.4.1), we see that

$$u(t,x) = E^x[f(X_t)]$$

is differentiable with respect to $t$ and $\frac{\partial u}{\partial t} = E^x[Af(X_t)]$.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.1.1 — Kolmogorov's Backward Equation)</span></p>

Let $f \in C_0^2(\mathbb{R}^n)$.

**a)** Define $u(t,x) = E^x[f(X_t)]$. Then $u(t,\cdot) \in \mathcal{D}_A$ for each $t$ and

$$\frac{\partial u}{\partial t} = Au, \quad t > 0,\, x \in \mathbb{R}^n$$

$$u(0,x) = f(x), \quad x \in \mathbb{R}^n$$

where the right hand side is to be interpreted as $A$ applied to the function $x \to u(t,x)$.

**b)** Moreover, if $w(t,x) \in C^{1,2}(\mathbb{R} \times \mathbb{R}^n)$ is a bounded function satisfying the above PDE and initial condition, then $w(t,x) = u(t,x)$.

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Theorem 8.1.1)</summary>

**Part a):** Let $g(x) = u(t,x)$. Since $t \to u(t,x)$ is differentiable, using the Markov property:

$$\frac{E^x[g(X_r)] - g(x)}{r} = \frac{u(t+r,x) - u(t,x)}{r} \to \frac{\partial u}{\partial t} \text{ as } r \downarrow 0$$

Hence $Au = \frac{\partial u}{\partial t}$ as asserted.

**Part b):** If $w(t,x)$ satisfies $\widetilde{A}w := -\frac{\partial w}{\partial t} + Aw = 0$ for $t > 0$ and $w(0,x) = f(x)$, define the process $Y_t = (s-t, X_t^{0,x})$ in $\mathbb{R}^{n+1}$. Then $Y_t$ has generator $\widetilde{A}$ and by Dynkin's formula, letting $R \to \infty$ and choosing $t = s$, we get $w(s,x) = E^x[f(X_s)] = u(s,x)$.

</details>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Semigroup Interpretation)</span></p>

If we introduce the operator $Q_t\colon f \to E^\bullet[f(X_t)]$ then $u(t,x) = (Q_t f)(x)$ and Kolmogorov's backward equation can be rewritten as:

$$\frac{d}{dt}(Q_t f) = Q_t(Af) \quad \text{and} \quad \frac{d}{dt}(Q_t f) = A(Q_t f)$$

The equivalence of these two forms amounts to saying that $Q_t$ and $A$ commute. Formally, the solution is $Q_t = e^{tA}$ and therefore $Q_t A = AQ_t$.

</div>

The operator $A$ is in general unbounded. However, its inverse (at least if a positive multiple of the identity is subtracted from $A$) can be expressed explicitly via the diffusion $X_t$:

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(8.1.2 — Resolvent Operator)</span></p>

For $\alpha > 0$ and $g \in C_b(\mathbb{R}^n)$ we define the **resolvent operator** $R_\alpha$ by

$$R_\alpha g(x) = E^x\left[\int_0^\infty e^{-\alpha t} g(X_t)\,dt\right]$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(8.1.3)</span></p>

$R_\alpha g$ is a bounded continuous function.

</div>

This follows from $R_\alpha g(x) = \int_0^\infty e^{-\alpha t} E^x[g(X_t)]\,dt$ and the Feller-continuity of Itô diffusions (Lemma 8.1.4 below).

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(8.1.4 — Feller Continuity)</span></p>

Let $g$ be a lower bounded, measurable function on $\mathbb{R}^n$ and define, for fixed $t \ge 0$, $u(x) = E^x[g(X_t)]$.

- **a)** If $g$ is lower semicontinuous, then $u$ is lower semicontinuous.
- **b)** If $g$ is bounded and continuous, then $u$ is continuous. In other words, any Itô diffusion $X_t$ is **Feller-continuous**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.1.5 — The Resolvent as Inverse of $\alpha - A$)</span></p>

**a)** If $f \in C_0^2(\mathbb{R}^n)$ then $R_\alpha(\alpha - A)f = f$ for all $\alpha > 0$.

**b)** If $g \in C_b(\mathbb{R}^n)$ then $R_\alpha g \in \mathcal{D}_A$ and $(\alpha - A)R_\alpha g = g$ for all $\alpha > 0$.

In other words, $R_\alpha$ and $\alpha - A$ are inverse operators.

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Theorem 8.1.5)</summary>

**Part a):** By Dynkin's formula, $R_\alpha(\alpha - A)f(x) = \alpha \int_0^\infty e^{-\alpha t}E^x[f(X_t)]\,dt - \int_0^\infty e^{-\alpha t}E^x[Af(X_t)]\,dt$. Integration by parts using $\frac{d}{dt}E^x[f(X_t)] = E^x[Af(X_t)]$ gives $E^x[f(X_0)] = f(x)$.

**Part b):** By the strong Markov property:

$$E^x[R_\alpha g(X_t)] = \int_0^\infty e^{-\alpha s} E^x[g(X_{t+s})]\,ds$$

Integration by parts gives $E^x[R_\alpha g(X_t)] = \alpha \int_0^\infty e^{-\alpha s}\int_t^{t+s}E^x[g(X_v)]\,dv\,ds$. This implies $R_\alpha g \in \mathcal{D}_A$ and $A(R_\alpha g) = \alpha R_\alpha g - g$.

</details>

### 8.2 The Feynman–Kac Formula. Killing

The Feynman–Kac formula is a useful generalization of Kolmogorov's backward equation.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.2.1 — The Feynman–Kac Formula)</span></p>

Let $f \in C_0^2(\mathbb{R}^n)$ and $q \in C(\mathbb{R}^n)$. Assume that $q$ is lower bounded.

**a)** Put

$$v(t,x) = E^x\left[\exp\!\left(-\int_0^t q(X_s)\,ds\right)f(X_t)\right]$$

Then

$$\frac{\partial v}{\partial t} = Av - qv; \quad t > 0,\, x \in \mathbb{R}^n$$

$$v(0,x) = f(x); \quad x \in \mathbb{R}^n$$

**b)** Moreover, if $w(t,x) \in C^{1,2}(\mathbb{R} \times \mathbb{R}^n)$ is bounded on $K \times \mathbb{R}^n$ for each compact $K \subset \mathbb{R}$ and $w$ solves the above PDE and initial condition, then $w(t,x) = v(t,x)$.

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Theorem 8.2.1)</summary>

**Part a):** Let $Y_t = f(X_t)$ and $Z_t = \exp(-\int_0^t q(X_s)\,ds)$. Then $dZ_t = -Z_t q(X_t)\,dt$. By the product rule, $d(Y_t Z_t) = Y_t\,dZ_t + Z_t\,dY_t$ (since $dZ_t \cdot dY_t = 0$). Since $Y_t Z_t$ is an Itô process, $v(t,x) = E^x[Y_t Z_t]$ is differentiable w.r.t. $t$. A direct computation gives $\frac{\partial v}{\partial t} + q(x)v(t,x) = Av$.

**Part b):** Define the process $H_t = (s-t, X_t^{0,x}, Z_t)$ where $Z_t = z + \int_0^t q(X_s)\,ds$. Then $H_t$ is an Itô diffusion with generator $A_H \phi = -\frac{\partial \phi}{\partial s} + A\phi + q(x)\frac{\partial \phi}{\partial z}$. Apply Dynkin's formula with $\phi(s,x,z) = \exp(-z)w(s,x)$, noting that $A_H \phi = 0$. Taking $R \to \infty$ and $t = s$ gives $w(s,x) = v(s,x)$.

</details>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(About Killing a Diffusion)</span></p>

In Theorem 7.3.3 we saw that the generator of an Itô diffusion $dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t$ is a partial differential operator of the form

$$Lf = \sum a_{ij}\frac{\partial^2 f}{\partial x_i \partial x_j} + \sum b_i \frac{\partial f}{\partial x_i}$$

where $[a_{ij}] = \tfrac{1}{2}\sigma\sigma^T$, $b = [b_i]$. It is natural to ask if one can find processes whose generator has the form

$$Lf = \sum a_{ij}\frac{\partial^2 f}{\partial x_i \partial x_j} + \sum b_i \frac{\partial f}{\partial x_i} - cf$$

where $c(x) \ge 0$ is bounded and continuous. The answer is yes: a process $\widetilde{X}_t$ with this generator is obtained by **killing** $X_t$ at a random **killing time** $\zeta$. We put $\widetilde{X}_t = X_t$ if $t < \zeta$ and leave $\widetilde{X}_t$ undefined (or send it to a "coffin state" $\partial$) if $t \ge \zeta$. Then $\widetilde{X}_t$ is also a strong Markov process and

$$E^x[f(\widetilde{X}_t)] = E^x\left[f(X_t) \cdot e^{-\int_0^t c(X_s)\,ds}\right]$$

The function $c(x)$ is interpreted as the **killing rate**: $c(x) = \lim_{t \downarrow 0} \frac{1}{t}Q^x[X_0 \text{ is killed in } (0,t]]$.

</div>

### 8.3 The Martingale Problem

If $dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t$ is an Itô diffusion in $\mathbb{R}^n$ with generator $A$ and if $f \in C_0^2(\mathbb{R}^n)$, then by Itô's formula (Theorem 7.3.1):

$$f(X_t) = f(x) + \int_0^t Af(X_s)\,ds + \int_0^t \nabla f^T(X_s)\sigma(X_s)\,dB_s$$

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.3.1 — Martingale Characterization of Itô Diffusions)</span></p>

If $X_t$ is an Itô diffusion in $\mathbb{R}^n$ with generator $A$, then for all $f \in C_0^2(\mathbb{R}^n)$ the process

$$M_t = f(X_t) - \int_0^t Af(X_r)\,dr$$

is a martingale w.r.t. $\lbrace\mathcal{M}_t\rbrace$.

</div>

Identifying each $\omega \in \Omega$ with $\omega_t = X_t^x(\omega)$, the probability space $(\Omega, \mathcal{M}, Q^x)$ is identified with $((\mathbb{R}^n)^{[0,\infty)}, \mathcal{B}, \widetilde{Q}^x)$ where $\mathcal{B}$ is the Borel $\sigma$-algebra on $(\mathbb{R}^n)^{[0,\infty)}$. The measure $\widetilde{Q}^x$ solves the **martingale problem** for $A$ in the following sense:

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(8.3.2 — The Martingale Problem)</span></p>

Let $L$ be a semi-elliptic differential operator of the form

$$L = \sum b_i \frac{\partial}{\partial x_i} + \sum a_{ij}\frac{\partial^2}{\partial x_i \partial x_j}$$

where the coefficients $b_i, a_{ij}$ are locally bounded Borel measurable functions on $\mathbb{R}^n$. We say that a probability measure $\widetilde{P}^x$ on $((\mathbb{R}^n)^{[0,\infty)}, \mathcal{B})$ **solves the martingale problem for $L$** (starting at $x$) if the process

$$M_t = f(\omega_t) - \int_0^t Lf(\omega_r)\,dr, \quad M_0 = f(x) \quad \text{a.s. } \widetilde{P}^x$$

is a $\widetilde{P}^x$-martingale w.r.t. $\mathcal{B}_t$, for all $f \in C_0^2(\mathbb{R}^n)$. The martingale problem is called **well posed** if there is a unique measure $\widetilde{P}^x$ solving it.

</div>

The argument of Theorem 8.3.1 shows that $\widetilde{Q}^x$ solves the martingale problem for $A$ whenever $X_t$ is a weak solution of $dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t$.

Conversely, if $\widetilde{P}^x$ solves the martingale problem for $L = \sum b_i \frac{\partial}{\partial x_i} + \tfrac{1}{2}\sum (\sigma\sigma^T)_{ij}\frac{\partial^2}{\partial x_i \partial x_j}$ starting at $x$ for all $x \in \mathbb{R}^n$, then there exists a weak solution $X_t$ of the SDE. Moreover, the weak solution is a Markov process if and only if the martingale problem for $L$ is well posed. Under the Lipschitz conditions of Theorem 5.2.1, we conclude:

$$\widetilde{Q}^x \text{ is the unique solution of the martingale problem for } L$$

A spectacular result of Stroock and Varadhan (1979) is that the martingale problem for $L = \sum b_i \frac{\partial}{\partial x_i} + \sum a_{ij}\frac{\partial^2}{\partial x_i \partial x_j}$ has a unique solution if $[a_{ij}]$ is everywhere positive definite, $a_{ij}(x)$ is continuous, $b(x)$ is measurable, and there exists a constant $D$ such that $\lvert b(x)\rvert + \lvert a(x)\rvert^{1/2} \le D(1 + \lvert x\rvert)$ for all $x \in \mathbb{R}^n$.

### 8.4 When is an Itô Process a Diffusion?

The Itô formula gives that if we apply a $C^2$ function $\phi\colon U \subset \mathbb{R}^n \to \mathbb{R}^n$ to an Itô process $X_t$, the result $\phi(X_t)$ is another Itô process. A natural question: if $X_t$ is an Itô diffusion, will $\phi(X_t)$ be an Itô diffusion too? The answer is no in general, but it may be yes in some cases.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(8.4.1 — The Bessel Process)</span></p>

Let $n \ge 2$. The process $R_t(\omega) = \lvert B(t,\omega)\rvert = (B_1(t,\omega)^2 + \cdots + B_n(t,\omega)^2)^{1/2}$ satisfies

$$dR_t = \sum_{i=1}^n \frac{B_i\,dB_i}{R_t} + \frac{n-1}{2R_t}\,dt$$

This is *not* directly of the form (5.2.3), but one can show that $Y_t := \int_0^t \sum_{i=1}^n \frac{B_i}{\lvert B\rvert}\,dB_i$ coincides in law with 1-dimensional Brownian motion $\widetilde{B}_t$. Then $dR_t = \frac{n-1}{2R_t}\,dt + d\widetilde{B}$, which by weak uniqueness (Lemma 5.3.1) shows that $R_t$ is an Itô diffusion with generator

$$Af(x) = \tfrac{1}{2}f''(x) + \frac{n-1}{2x}f'(x)$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.4.2 — Characterization of Itô Integrals Coinciding with Brownian Motion)</span></p>

An Itô process $dY_t = v\,dB_t$, $Y_0 = 0$ with $v(t,\omega) \in \mathcal{V}_H^{n \times m}$ coincides in law with $n$-dimensional Brownian motion if and only if

$$vv^T(t,\omega) = I_n \quad \text{for a.a. } (t,\omega) \text{ w.r.t. } dt \times dP$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.4.3 — When an Itô Process Coincides in Law with a Diffusion)</span></p>

Let $X_t$ be an Itô diffusion given by $dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t$, $b \in \mathbb{R}^n$, $\sigma \in \mathbb{R}^{n \times m}$, $X_0 = x$, and let $Y_t$ be an Itô process given by $dY_t = u(t,\omega)\,dt + v(t,\omega)\,dB_t$, $u \in \mathbb{R}^n$, $v \in \mathbb{R}^{n \times m}$, $Y_0 = x$.

Then $X_t \simeq Y_t$ (coincide in law) if and only if

$$E^x[u(t,\cdot) \mid \mathcal{N}_t] = b(Y_t^x) \quad \text{and} \quad vv^T(t,\omega) = \sigma\sigma^T(Y_t^x)$$

for a.a. $(t,\omega)$ w.r.t. $dt \times dP$, where $\mathcal{N}_t$ is the $\sigma$-algebra generated by $\lbrace Y_s;\, s \le t\rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

1. The drift $u(t,\cdot)$ need not be $\mathcal{N}_t$-measurable. For instance, if $B_1, B_2$ are independent 1-dimensional Brownian motions and $dY_t = B_1(t)\,dt + dB_2(t)$, then $Y_t$ can be regarded as noisy observations of $B_1(t)$, and $B_1(t,\omega)$ cannot be $\mathcal{N}_t$-measurable.

2. The diffusion coefficient $v(t,\omega)$ also need not be $\mathcal{N}_t$-adapted. For instance, $dY_t = \text{sign}(B_t)\,dB_t$ gives $\lvert B_t\rvert = \lvert B_0\rvert + \int_0^t \text{sign}(B_s)\,dB_s + L_t$ by Tanaka's formula, where $L_t$ is the local time of $B_t$ at $0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(8.4.5 — How to Recognize a Brownian Motion)</span></p>

Let $dY_t = u(t,\omega)\,dt + v(t,\omega)\,dB_t$ be an Itô process in $\mathbb{R}^n$. Then $Y_t$ is a Brownian motion if and only if

$$E^x[u(t,\cdot) \mid \mathcal{N}_t] = 0 \quad \text{and} \quad vv^T(t,\omega) = I_n$$

for a.a. $(t,\omega)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Image of a Diffusion under a $C^2$ Map)</span></p>

Using Theorem 8.4.3, one may investigate when the image $Y_t = \phi(X_t)$ of an Itô diffusion $X_t$ by a $C^2$-function $\phi$ coincides in law with an Itô diffusion $Z_t$. Applying the criterion (8.4.3), one obtains:

$$\phi(X_t) \sim Z_t \quad \text{if and only if} \quad A[f \circ \phi] = \widehat{A}[f] \circ \phi$$

for all second order polynomials $f(x_1,\ldots,x_n) = \sum a_i x_i + \sum c_{ij}x_i x_j$ (and hence for all $f \in C_0^2$), where $A$ and $\widehat{A}$ are the generators of $X_t$ and $Z_t$ respectively.

</div>

### 8.5 Random Time Change

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Time Change and Its Inverse)</span></p>

Let $c(t,\omega) \ge 0$ be an $\mathcal{F}_t$-adapted process. Define the **time change** $\beta_t$ with **time change rate** $c(t,\omega)$ by

$$\beta_t = \beta(t,\omega) = \int_0^t c(s,\omega)\,ds$$

Note that $\beta(t,\omega)$ is $\mathcal{F}_t$-adapted and for each $\omega$ the map $t \to \beta_t(\omega)$ is non-decreasing. Define $\alpha_t = \alpha(t,\omega)$ by

$$\alpha_t = \inf\lbrace s;\, \beta_s > t\rbrace$$

Then $\alpha_t$ is a right-inverse of $\beta_t$: $\beta(\alpha(t,\omega),\omega) = t$ for all $t \ge 0$. Moreover, $t \to \alpha_t(\omega)$ is right-continuous. If $c(s,\omega) > 0$ for a.a. $(s,\omega)$ then $\beta_t$ is strictly increasing, $\alpha_t$ is continuous, and $\alpha_t$ is also a left-inverse: $\alpha(\beta(t,\omega),\omega) = t$. In general, $\omega \to \alpha(t,\omega)$ is an $\lbrace\mathcal{F}_s\rbrace$-stopping time for each $t$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.5.1 — Time Change Producing a Diffusion)</span></p>

Let $X_t, Y_t$ be as in Theorem 8.4.3 and let $\beta_t$ be a time change with right-inverse $\alpha_t$ as above. Assume that

$$u(t,\omega) = c(t,\omega)b(Y_t) \quad \text{and} \quad vv^T(t,\omega) = c(t,\omega) \cdot \sigma\sigma^T(Y_t)$$

for a.a. $t, \omega$. Then $Y_{\alpha_t} \simeq X_t$.

</div>

This result allows us to recognize time changes of Brownian motion:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.5.2 — Time Change of Brownian Motion)</span></p>

Let $dY_t = v(t,\omega)\,dB_t$, $v \in \mathbb{R}^{n \times m}$, $B_t \in \mathbb{R}^m$, be an Itô integral in $\mathbb{R}^n$, $Y_0 = 0$, and assume that

$$vv^T(t,\omega) = c(t,\omega)I_n$$

for some process $c(t,\omega) \ge 0$. Let $\alpha_t, \beta_t$ be as above. Then $Y_{\alpha_t}$ is an $n$-dimensional Brownian motion.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(8.5.3)</span></p>

Let $dY_t = \sum_{i=1}^n v_i(t,\omega)\,dB_i(t,\omega)$, $Y_0 = 0$, where $B = (B_1,\ldots,B_n)$ is Brownian motion in $\mathbb{R}^n$. Then

$$\widehat{B}_t := Y_{\alpha_t} \quad \text{is a 1-dimensional Brownian motion}$$

where $\alpha_t$ is defined by $\beta_s = \int_0^s \left\lbrace\sum_{i=1}^n v_i^2(r,\omega)\right\rbrace dr$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(8.5.4)</span></p>

Let $Y_t, \beta_s$ be as in Corollary 8.5.3. Assume that $\sum_{i=1}^n v_i^2(r,\omega) > 0$ for a.a. $(r,\omega)$. Then there exists a Brownian motion $\widehat{B}_t$ such that

$$Y_t = \widehat{B}_{\beta_t}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.5.7 — Time Change Formula for Itô Integrals)</span></p>

Suppose $c(s,\omega)$ and $\alpha(s,\omega)$ are $s$-continuous, $\alpha(0,\omega) = 0$ for a.a. $\omega$ and that $E[\alpha_t] < \infty$. Let $B_s$ be an $m$-dimensional Brownian motion and let $v(s,\omega) \in \mathcal{V}_H^{n \times m}$ be bounded and $s$-continuous. Define

$$\widetilde{B}_t = \lim_{k \to \infty}\sum_j \sqrt{c(\alpha_j,\omega)}\,\Delta B_{\alpha_j} = \int_0^{\alpha_t}\sqrt{c(s,\omega)}\,dB_s$$

Then $\widetilde{B}_t$ is an ($m$-dimensional) $\mathcal{F}_{\alpha_t}^{(m)}$-Brownian motion and $\widetilde{B}_t$ is a martingale w.r.t. $\mathcal{F}_{\alpha_t}^{(m)}$, and

$$\int_0^{\alpha_t} v(s,\omega)\,dB_s = \int_0^t v(\alpha_r,\omega)\sqrt{\alpha_r'(\omega)}\,d\widetilde{B}_r \quad \text{a.s. } P$$

where $\alpha_r'(\omega) = \frac{1}{c(\alpha_r, \omega)}$ for a.a. $r \ge 0$, a.a. $\omega \in \Omega$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(8.5.8 — Brownian Motion on the Unit Sphere in $\mathbb{R}^n$; $n > 2$)</span></p>

Apply the function $\phi(x) = x \cdot \lvert x\rvert^{-1}$ to $n$-dimensional Brownian motion $B = (B_1,\ldots,B_n)$. The result is a stochastic integral $Y = (Y_1,\ldots,Y_n) = \phi(B)$ with

$$dY = \frac{1}{\lvert B\rvert}\cdot\sigma(Y)\,dB + \frac{1}{\lvert B\rvert^2}b(Y)\,dt$$

where $\sigma_{ij}(Y) = \delta_{ij} - Y_i Y_j$ and $b(y) = -\frac{n-1}{2}\cdot y$.

Performing the time change $Z_t(\omega) = Y_{\alpha(t,\omega)}(\omega)$ with $\beta(t,\omega) = \int_0^t \frac{1}{\lvert B\rvert^2}\,ds$, one gets $dZ = \sigma(Z)\,d\widetilde{B} + b(Z)\,dt$. Hence $Z$ is a diffusion on the unit sphere $S$ with characteristic operator

$$\mathcal{A}f(y) = \tfrac{1}{2}\left(\Delta f(y) - \sum_{i,j}y_i y_j \frac{\partial^2 f}{\partial y_i \partial y_j}\right) - \frac{n-1}{2}\cdot\sum_i y_i\frac{\partial f}{\partial y_i}; \quad \lvert y\rvert = 1$$

Since $Z$ is invariant under orthogonal transformations, it is called **Brownian motion on the unit sphere** $S$.

More generally, given a Riemannian manifold $M$ with metric tensor $g = [g_{ij}]$, one may define **Brownian motion on $M$** as a diffusion on $M$ whose characteristic operator $\mathcal{A}$ in local coordinates $x_i$ is given by $\tfrac{1}{2}$ times the Laplace–Beltrami operator:

$$\Delta_M = \frac{1}{\sqrt{\det(g)}}\cdot\sum_i \frac{\partial}{\partial x_i}\left(\sqrt{\det(g)}\sum_j g^{ij}\frac{\partial}{\partial x_j}\right)$$

where $[g^{ij}] = [g_{ij}]^{-1}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(8.5.9 — Harmonic and Analytic Functions)</span></p>

Let $B = (B_1, B_2)$ be 2-dimensional Brownian motion. Apply a $C^2$ function $\phi(x_1,x_2) = (u(x_1,x_2), v(x_1,x_2))$ to $B$. Put $Y = (Y_1, Y_2) = \phi(B_1,B_2)$. By Itô's formula:

$$dY = b(B_1,B_2)\,dt + \sigma(B_1,B_2)\,dB$$

with $b = \tfrac{1}{2}\binom{\Delta u}{\Delta v}$ and $\sigma = D_\phi$ (the derivative of $\phi$). So $Y = \phi(B_1,B_2)$ is a martingale if and only if $\phi$ is harmonic (i.e. $\Delta\phi = 0$).

If $\phi$ is harmonic, then by Corollary 8.5.3, $\phi(B_1,B_2) = (\widetilde{B}^{(1)}_{\beta_1}, \widetilde{B}^{(2)}_{\beta_2})$ where $\widetilde{B}^{(1)}, \widetilde{B}^{(2)}$ are (not necessarily independent) 1-dimensional Brownian motions, and $\beta_1(t,\omega) = \int_0^t \lvert\nabla u\rvert^2(B_1,B_2)\,ds$, $\beta_2(t,\omega) = \int_0^t \lvert\nabla v\rvert^2(B_1,B_2)\,ds$.

Since $\sigma\sigma^T = \begin{pmatrix}\lvert\nabla u\rvert^2 & \nabla u \cdot \nabla v\\ \nabla u \cdot \nabla v & \lvert\nabla v\rvert^2\end{pmatrix}$, the condition for $Y_{\alpha_t}$ to be a 2-dimensional Brownian motion (by Theorem 8.5.2) is $\sigma\sigma^T = \lvert\nabla u\rvert^2 I_2$ (in addition to $\Delta u = \Delta v = 0$), i.e.:

$$\lvert\nabla u\rvert^2 = \lvert\nabla v\rvert^2 \quad \text{and} \quad \nabla u \cdot \nabla v = 0$$

These are precisely the **Cauchy–Riemann equations**: the function $\phi(x+iy) = \phi(x,y)$ regarded as a complex function is either **analytic** or **conjugate analytic**. This is a theorem of P. Lévy: $\phi(B_1,B_2)$ is — after a change of time scale — again Brownian motion in the plane if and only if $\phi$ is either analytic or conjugate analytic.

</div>

### 8.6 The Girsanov Theorem

The Girsanov theorem is fundamental in the general theory of stochastic analysis and very important in many applications, for example in economics (Chapter 12). Basically, it says that if we change the *drift* coefficient of a given Itô process (with a nondegenerate diffusion coefficient), the law of the process will not change dramatically — it will be absolutely continuous w.r.t. the law of the original process, and we can compute explicitly the Radon–Nikodym derivative.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.6.1 — Lévy Characterization of Brownian Motion)</span></p>

Let $X(t) = (X_1(t),\ldots,X_n(t))$ be a continuous stochastic process on a probability space $(\Omega, \mathcal{H}, Q)$ with values in $\mathbb{R}^n$. Then the following are equivalent:

- **a)** $X(t)$ is a Brownian motion w.r.t. $Q$.
- **b)** (i) $X(t) = (X_1(t),\ldots,X_n(t))$ is a martingale w.r.t. $Q$ (and w.r.t. its own filtration) and
  (ii) $X_i(t)X_j(t) - \delta_{ij}t$ is a martingale w.r.t. $Q$ (and w.r.t. its own filtration) for all $i, j \in \lbrace 1,2,\ldots,n\rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Condition (ii) may be replaced by the condition that the cross-variation processes $\langle X_i, X_j\rangle_t$ satisfy $\langle X_i, X_j\rangle_t = \delta_{ij}t$ a.s., for $1 \le i, j \le n$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(8.6.2 — Conditional Expectation under Change of Measure)</span></p>

Let $\mu$ and $\nu$ be two probability measures on a measurable space $(\Omega, \mathcal{G})$ such that $d\nu(\omega) = f(\omega)\,d\mu(\omega)$ for some $f \in L^1(\mu)$. Let $X$ be a random variable on $(\Omega, \mathcal{G})$ such that $E_\nu[\lvert X\rvert] = \int_\Omega \lvert X(\omega)\rvert f(\omega)\,d\mu(\omega) < \infty$. Let $\mathcal{H}$ be a $\sigma$-algebra, $\mathcal{H} \subset \mathcal{G}$. Then

$$E_\nu[X \mid \mathcal{H}] \cdot E_\mu[f \mid \mathcal{H}] = E_\mu[fX \mid \mathcal{H}] \quad \text{a.s.}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.6.3 — The Girsanov Theorem I)</span></p>

Let $Y(t) \in \mathbb{R}^n$ be an Itô process of the form

$$dY(t) = a(t,\omega)\,dt + dB(t); \quad t \le T,\; Y_0 = 0$$

where $T \le \infty$ is a given constant and $B(t)$ is $n$-dimensional Brownian motion. Put

$$M_t = \exp\!\left(-\int_0^t a(s,\omega)\,dB_s - \tfrac{1}{2}\int_0^t a^2(s,\omega)\,ds\right); \quad t \le T$$

Assume that $a(s,\omega)$ satisfies **Novikov's condition**:

$$E\left[\exp\!\left(\tfrac{1}{2}\int_0^T a^2(s,\omega)\,ds\right)\right] < \infty$$

where $E = E_P$ is the expectation w.r.t. $P$. Define the measure $Q$ on $(\Omega, \mathcal{F}_T^{(n)})$ by

$$dQ(\omega) = M_T(\omega)\,dP(\omega)$$

Then $Y(t)$ is an $n$-dimensional Brownian motion w.r.t. the probability law $Q$, for $t \le T$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

1. The transformation $P \to Q$ given by $dQ = M_T\,dP$ is called the **Girsanov transformation of measures**.
2. The Novikov condition is sufficient to guarantee that $\lbrace M_t\rbrace_{t \le T}$ is a martingale (w.r.t. $\mathcal{F}_t^{(n)}$ and $P$). Actually, the result holds if we only assume that $\lbrace M_t\rbrace_{t \le T}$ is a martingale (see Karatzas and Shreve (1991)).
3. Since $M_t$ is a martingale, we actually have $M_T\,dP = M_t\,dP$ on $\mathcal{F}_t^{(n)}$ for $t \le T$.

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Theorem 8.6.3)</summary>

For simplicity assume $a(s,\omega)$ is bounded. By Theorem 8.6.1 it suffices to verify:

**(i)** $Y(t) = (Y_1(t),\ldots,Y_n(t))$ is a martingale w.r.t. $Q$.

**(ii)** $Y_i(t)Y_j(t) - \delta_{ij}t$ is a martingale w.r.t. $Q$.

For (i): Put $K_i(t) = M_t Y_i(t)$. By Itô's formula, $dK_i(t) = M_t(dB_i(t) - Y_i(t)\sum_k a_k(t)\,dB_k(t)) = M_t \gamma^{(i)}(t)\,dB(t)$, so $K_i(t)$ is a martingale w.r.t. $P$. By Lemma 8.6.2:

$$E_Q[Y_i(t) \mid \mathcal{F}_s] = \frac{E[M_t Y_i(t) \mid \mathcal{F}_s]}{E[M_t \mid \mathcal{F}_s]} = \frac{K_i(s)}{M_s} = Y_i(s)$$

The proof of (ii) is similar.

</details>

Theorem 8.6.3 states that for all Borel sets $F_1,\ldots,F_k \subset \mathbb{R}^n$ and all $t_1,t_2,\ldots,t_k \le T$, $k = 1,2,\ldots$ we have

$$Q[Y(t_1) \in F_1,\ldots,Y(t_k) \in F_k] = P[B(t_1) \in F_1,\ldots,B(t_k) \in F_k]$$

An equivalent way of expressing this is to say that $Q \ll P$ ($Q$ is absolutely continuous w.r.t. $P$) with Radon-Nikodym derivative

$$\frac{dQ}{dP} = M_T \quad \text{on } \mathcal{F}_T^{(n)}$$

Note that $M_T(\omega) > 0$ a.s., so we also have that $P \ll Q$. Hence the two measures $Q$ and $P$ are equivalent. Therefore we get

$$P[Y(t_1) \in F_1,\ldots,Y(t_k) \in F_k] > 0 \iff P[B(t_1) \in F_1,\ldots,B(t_k) \in F_k] > 0$$

for all $t_1,\ldots,t_k \in [0,T]$.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.6.4 — The Girsanov Theorem II)</span></p>

Let $Y(t) \in \mathbb{R}^n$ be an Itô process of the form

$$dY(t) = \beta(t,\omega)\,dt + \theta(t,\omega)\,dB(t); \quad t \le T$$

where $B(t) \in \mathbb{R}^m$, $\beta(t,\omega) \in \mathbb{R}^n$ and $\theta(t,\omega) \in \mathbb{R}^{n \times m}$. Suppose there exist processes $u(t,\omega) \in \mathcal{W}_H^m$ and $\alpha(t,\omega) \in \mathcal{W}_H^n$ such that

$$\theta(t,\omega)u(t,\omega) = \beta(t,\omega) - \alpha(t,\omega)$$

and assume that $u(t,\omega)$ satisfies **Novikov's condition**

$$E\left[\exp\!\left(\tfrac{1}{2}\int_0^T u^2(s,\omega)\,ds\right)\right] < \infty$$

Put

$$M_t = \exp\!\left(-\int_0^t u(s,\omega)\,dB_s - \tfrac{1}{2}\int_0^t u^2(s,\omega)\,ds\right); \quad t \le T$$

and

$$dQ(\omega) = M_T(\omega)\,dP(\omega) \quad \text{on } \mathcal{F}_T^{(m)}$$

Then

$$\widehat{B}(t) := \int_0^t u(s,\omega)\,ds + B(t); \quad t \le T$$

is a Brownian motion w.r.t. $Q$ and in terms of $\widehat{B}(t)$ the process $Y(t)$ has the stochastic integral representation

$$dY(t) = \alpha(t,\omega)\,dt + \theta(t,\omega)\,d\widehat{B}(t)$$

</div>

<details class="accordion" markdown="1">
<summary>Proof (Theorem 8.6.4)</summary>

It follows from Theorem 8.6.3 that $\widehat{B}(t)$ is a Brownian motion w.r.t. $Q$. So, substituting into the SDE and using $\theta u = \beta - \alpha$:

$$\begin{aligned}
dY(t) &= \beta(t,\omega)\,dt + \theta(t,\omega)(d\widehat{B}(t) - u(t,\omega)\,dt) \\
&= [\beta(t,\omega) - \theta(t,\omega)u(t,\omega)]\,dt + \theta(t,\omega)\,d\widehat{B}(t) \\
&= \alpha(t,\omega)\,dt + \theta(t,\omega)\,d\widehat{B}(t)
\end{aligned}$$

Note that if $n = m$ and $\theta \in \mathbb{R}^{n \times n}$ is invertible, then the process $u(t,\omega)$ is given uniquely by

$$u(t,\omega) = \theta^{-1}(t,\omega)[\beta(t,\omega) - \alpha(t,\omega)]$$

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.6.5 — The Girsanov Theorem III)</span></p>

Let $X(t) = X^x(t) \in \mathbb{R}^n$ and $Y(t) = Y^x(t) \in \mathbb{R}^n$ be an Itô diffusion and an Itô process, respectively, of the forms

$$dX(t) = b(X(t))\,dt + \sigma(X(t))\,dB(t); \quad t \le T,\; X(0) = x$$

$$dY(t) = [\gamma(t,\omega) + b(Y(t))]\,dt + \sigma(Y(t))\,dB(t); \quad t \le T,\; Y(0) = x$$

where the functions $b\colon\mathbb{R}^n \to \mathbb{R}^n$ and $\sigma\colon\mathbb{R}^n \to \mathbb{R}^{n \times m}$ satisfy the conditions of Theorem 5.2.1 and $\gamma(t,\omega) \in \mathcal{W}_H^n$, $x \in \mathbb{R}^n$. Suppose there exists a process $u(t,\omega) \in \mathcal{W}_H^m$ such that

$$\sigma(Y(t))u(t,\omega) = \gamma(t,\omega)$$

and assume that $u(t,\omega)$ satisfies Novikov's condition. Define $M_t$, $Q$ and $\widehat{B}(t)$ as in Theorem 8.6.4. Then

$$dY(t) = b(Y(t))\,dt + \sigma(Y(t))\,d\widehat{B}(t)$$

Therefore, the $Q$-law of $Y^x(t)$ is the same as the $P$-law of $X^x(t)$, for $t \le T$.

</div>

<details class="accordion" markdown="1">
<summary>Proof (Theorem 8.6.5)</summary>

The representation follows by applying Theorem 8.6.4 with $\theta(t,\omega) = \sigma(Y(t))$, $\beta(t,\omega) = \gamma(t,\omega) + b(Y(t))$, $\alpha(t,\omega) = b(Y(t))$. Then the conclusion about laws follows from the weak uniqueness of solutions of stochastic differential equations (Lemma 5.3.1).

</details>

The Girsanov theorem III can be used to produce **weak solutions** of stochastic differential equations. Suppose $Y_t$ is a known weak or strong solution to the equation

$$dY_t = b(Y_t)\,dt + \sigma(Y_t)\,dB(t)$$

where $b\colon\mathbb{R}^n \to \mathbb{R}^n$, $\sigma\colon\mathbb{R}^n \to \mathbb{R}^{n \times m}$ and $B(t) \in \mathbb{R}^m$. We wish to find a weak solution $X(t)$ of a related equation

$$dX_t = a(X_t)\,dt + \sigma(X_t)\,dB(t)$$

where the drift function is changed to $a\colon\mathbb{R}^n \to \mathbb{R}^n$. Suppose we can find a function $u_0\colon\mathbb{R}^n \to \mathbb{R}^m$ such that

$$\sigma(y)u_0(y) = b(y) - a(y); \quad y \in \mathbb{R}^n$$

(If $n = m$ and $\sigma$ is invertible we choose $u_0 = \sigma^{-1} \cdot (b - a)$.)

Then if $u(t,\omega) = u_0(Y_t(\omega))$ satisfies Novikov's conditions, we have, with $Q$ and $\widehat{B}_t = \widehat{B}(t)$ as before, that

$$dY_t = a(Y_t)\,dt + \sigma(Y_t)\,d\widehat{B}_t$$

Thus we have found a Brownian motion $(\widehat{B}_t, Q)$ such that $Y_t$ satisfies the equation with drift $a$. Therefore $(Y_t, \widehat{B}_t)$ is a weak solution.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(8.6.6 — Weak Solution via Girsanov)</span></p>

Let $a\colon\mathbb{R}^n \to \mathbb{R}^n$ be a bounded, measurable function. Then we can construct a weak solution $X_t = X_t^x$ of the stochastic differential equation

$$dX_t = a(X_t)\,dt + dB_t; \quad X_0 = x \in \mathbb{R}^n$$

We proceed with $\sigma = I$, $b = 0$ and

$$dY_t = dB_t; \quad Y_0 = x$$

Choose $u_0 = \sigma^{-1} \cdot (b - a) = -a$ and define

$$M_t = \exp\!\left\lbrace -\int_0^t u_0(Y_s)\,dB_s - \tfrac{1}{2}\int_0^t u_0^2(Y_s)\,ds\right\rbrace = \exp\!\left\lbrace \int_0^t a(B_s)\,dB_s - \tfrac{1}{2}\int_0^t a^2(B_s)\,ds\right\rbrace$$

Fix $T < \infty$ and put $dQ = M_T\,dP$ on $\mathcal{F}_T^{(m)}$. Then

$$\widehat{B}_t := -\int_0^t a(B_s)\,ds + B_t$$

is a Brownian motion w.r.t. $Q$ for $t \le T$ and $dB_t = a(Y_t)\,dt + d\widehat{B}_t$. Hence if we set $Y_0 = x$, the pair $(Y_t, \widehat{B}_t)$ is a weak solution. By weak uniqueness the $Q$-law of $Y_t = B_t$ coincides with the $P$-law of $X_t^x$, so that

$$E[f_1(X_{t_1}^x) \cdots f_k(X_{t_k}^x)] = E_Q[f_1(Y_{t_1}) \cdots f_k(Y_{t_k})] = E[M_T f_1(B_{t_1}) \cdots f_k(B_{t_k})]$$

for all $f_1,\ldots,f_k \in C_0(\mathbb{R}^n)$; $t_1,\ldots,t_k \le T$.

</div>

## 9. Applications to Boundary Value Problems

### 9.1 The Combined Dirichlet–Poisson Problem. Uniqueness

We now use results from the preceding chapters to solve the following generalization of the Dirichlet problem stated in the introduction.

Let $D$ be a domain (open connected set) in $\mathbb{R}^n$ and let $L$ denote a **semi-elliptic** partial differential operator on $C^2(\mathbb{R}^n)$ of the form

$$L = \sum_{i=1}^n b_i(x)\frac{\partial}{\partial x_i} + \sum_{i,j=1}^n a_{ij}(x)\frac{\partial^2}{\partial x_i \partial x_j}$$

where $b_i(x)$ and $a_{ij}(x) = a_{ji}(x)$ are continuous functions. (By saying that $L$ is *semi-elliptic* (resp. *elliptic*) we mean that all the eigenvalues of the symmetric matrix $a(x) = [a_{ij}(x)]_{i,j=1}^n$ are *non-negative* (resp. *positive*) for all $x$.)

**The Combined Dirichlet–Poisson Problem.** Let $\phi \in C(\partial D)$ and $g \in C(D)$ be given functions. Find $w \in C^2(D)$ such that

$$Lw = -g \quad \text{in } D$$

and

$$\lim_{\substack{x \to y \\ x \in D}} w(x) = \phi(y) \quad \text{for all } y \in \partial D$$

The idea of the solution is the following: First we find an Itô diffusion $\lbrace X_t\rbrace$ whose generator $A$ coincides with $L$ on $C_0^2(\mathbb{R}^n)$. To achieve this we simply choose $\sigma(x) \in \mathbb{R}^{n \times n}$ such that

$$\tfrac{1}{2}\sigma(x)\sigma^T(x) = [a_{ij}(x)]$$

We assume that $\sigma(x)$ and $b(x) = [b_i(x)]$ satisfy the conditions of Theorem 5.2.1. Next we let $X_t$ be the solution of

$$dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t$$

where $B_t$ is $n$-dimensional Brownian motion. As usual let $E^x$ denote expectation with respect to the probability law $Q^x$ of $X_t$ starting at $x \in \mathbb{R}^n$. Then our candidate for the solution $w$ is

$$w(x) = E^x[\phi(X_{\tau_D}) \cdot \mathcal{X}_{\lbrace\tau_D < \infty\rbrace}] + E^x\!\left[\int_0^{\tau_D} g(X_t)\,dt\right]$$

provided that $\phi$ is bounded and

$$E^x\!\left[\int_0^{\tau_D} \lvert g(X_t)\rvert\,dt\right] < \infty \quad \text{for all } x$$

The Dirichlet–Poisson problem consists of two parts:
1. Existence of solution.
2. Uniqueness of solution.

The uniqueness problem turns out to be simpler and therefore we handle this first.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.1.1 — Uniqueness Theorem (1))</span></p>

Suppose $\phi$ is bounded and $g$ satisfies $E^x\!\left[\int_0^{\tau_D} \lvert g(X_t)\rvert\,dt\right] < \infty$ for all $x$. Suppose $w \in C^2(D)$ is bounded and satisfies

$$Lw = -g \quad \text{in } D$$

and

$$\lim_{t \uparrow \tau_D} w(X_t) = \phi(X_{\tau_D}) \cdot \mathcal{X}_{\lbrace\tau_D < \infty\rbrace} \quad \text{a.s. } Q^x \text{ for all } x$$

Then

$$w(x) = E^x[\phi(X_{\tau_D}) \cdot \mathcal{X}_{\lbrace\tau_D < \infty\rbrace}] + E^x\!\left[\int_0^{\tau_D} g(X_t)\,dt\right]$$

</div>

<details class="accordion" markdown="1">
<summary>Proof (Theorem 9.1.1)</summary>

Let $\lbrace D_k\rbrace_{k=1}^\infty$ be an increasing sequence of open sets $D_k$ such that $D_k \subset\subset D$ and $D = \bigcup_{k=1}^\infty D_k$. Define $\alpha_k = k \wedge \tau_{D_k}$; $k = 1,2,\ldots$. Then by the Dynkin formula and $Lw = -g$:

$$w(x) = E^x[w(X_{\alpha_k})] + E^x\!\left[\int_0^{\alpha_k} g(X_t)\,dt\right]$$

By the boundary condition, $w(X_{\alpha_k}) \to \phi(X_{\tau_D}) \cdot \mathcal{X}_{\lbrace\tau_D < \infty\rbrace}$ pointwise boundedly a.s. $Q^x$. Hence

$$E^x[w(X_{\alpha_k})] \to E^x[\phi(X_{\tau_D}) \cdot \mathcal{X}_{\lbrace\tau_D < \infty\rbrace}] \quad \text{as } k \to \infty$$

Moreover,

$$E^x\!\left[\int_0^{\alpha_k} g(X_t)\,dt\right] \to E^x\!\left[\int_0^{\tau_D} g(X_t)\,dt\right] \quad \text{as } k \to \infty$$

since $\int_0^{\alpha_k} g(X_t)\,dt \to \int_0^{\tau_D} g(X_t)\,dt$ a.s. and $\left\lvert\int_0^{\alpha_k} g(X_t)\,dt\right\rvert \le \int_0^{\tau_D} \lvert g(X_t)\rvert\,dt$, which is $Q^x$-integrable. Combining gives the result.

</details>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(9.1.2 — Uniqueness Theorem (2))</span></p>

Suppose $\phi$ is bounded and $g$ satisfies $E^x\!\left[\int_0^{\tau_D} \lvert g(X_t)\rvert\,dt\right] < \infty$ for all $x$. Suppose

$$\tau_D < \infty \quad \text{a.s. } Q^x \text{ for all } x$$

Then if $w \in C^2(D)$ is a bounded solution of the combined Dirichlet–Poisson problem, we have

$$w(x) = E^x[\phi(X_{\tau_D})] + E^x\!\left[\int_0^{\tau_D} g(X_t)\,dt\right]$$

</div>

### 9.2 The Dirichlet Problem. Regular Points

We now consider the more complicated question of existence of solution. It is convenient to split the combined Dirichlet–Poisson problem in two parts: The **Dirichlet problem** and the **Poisson problem**.

**The Dirichlet Problem.** Let $\phi \in C(\partial D)$ be a given function. Find $u \in C^2(D)$ such that

$$Lu = 0 \quad \text{in } D$$

and

$$\lim_{\substack{x \to y \\ x \in D}} u(x) = \phi(y) \quad \text{for all } y \in \partial D$$

**The Poisson Problem.** Let $g \in C(D)$ be a given function. Find $v \in C^2(D)$ such that

$$Lv = -g \quad \text{in } D$$

and

$$\lim_{\substack{x \to y \\ x \in D}} v(x) = 0 \quad \text{for all } y \in \partial D$$

Note that if $u$ and $v$ solve the Dirichlet and the Poisson problem, respectively, then $w := u + v$ solves the combined Dirichlet–Poisson problem.

For simplicity we assume in this section that $\tau_D < \infty$ a.s. $Q^x$ for all $x$. In view of Corollary 9.1.2 the question of existence of a solution of the Dirichlet problem can be restated as follows: When is

$$u(x) := E^x[\phi(X_{\tau_D})]$$

a solution? Unfortunately, in general this function $u$ need not be in $C^2(U)$. In fact, it need not even be continuous. Moreover, it need not satisfy the boundary condition, either.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(9.2.1 — Failure of classical Dirichlet solution)</span></p>

Let $X(t) = (X_1(t), X_2(t))$ be the solution of the equations $dX_1(t) = dt$, $dX_2(t) = 0$, so that $X(t) = X(0) + t(1,0) \in \mathbb{R}^2$; $t \ge 0$. Let

$$D = ((0,1) \times (0,1)) \cup ((0,2) \times (0,\tfrac{1}{2}))$$

and let $\phi$ be a continuous function on $\partial D$ such that $\phi = 1$ on $\lbrace 1\rbrace \times [\tfrac{1}{2},1]$, $\phi = 0$ on $\lbrace 2\rbrace \times [0,\tfrac{1}{2}]$, and $\phi = 0$ on $\lbrace 0\rbrace \times [0,1]$. Then

$$u(t,x) = E^{t,x}[\phi(X_{\tau_D})] = \begin{cases} 1 & \text{if } x \in (\tfrac{1}{2}, 1) \\ 0 & \text{if } x \in (0, \tfrac{1}{2}) \end{cases}$$

so $u$ is not even continuous. Moreover, $\lim_{t \to 0^+} u(t,x) = 1 \ne \phi(0,x)$ if $\tfrac{1}{2} < x < 1$, so the boundary condition does not hold.

</div>

However, the function $u(x)$ defined by $E^x[\phi(X_{\tau_D})]$ will solve the Dirichlet problem in a weaker, stochastic sense: The boundary condition is replaced by the stochastic (pathwise) boundary condition $\lim_{t \uparrow \tau_D} u(X_t) = \phi(X_{\tau_D})$ and the condition $Lu = 0$ is replaced by a condition related to

$$\mathcal{A}\,u = 0$$

where $\mathcal{A}$ is the characteristic operator of $X_t$ (Section 7.5).

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(9.2.2 — X-harmonic Function)</span></p>

Let $f$ be a locally bounded, measurable function on $D$. Then $f$ is called **$X$-harmonic** in $D$ if

$$f(x) = E^x[f(X_{\tau_U})]$$

for all $x \in D$ and all bounded open sets $U$ with $\overline{U} \subset D$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(9.2.3)</span></p>

**a)** Let $f$ be $X$-harmonic in $D$. Then $\mathcal{A}f = 0$ in $D$.

**b)** Conversely, suppose $f \in C^2(D)$ and $\mathcal{A}f = 0$ in $D$. Then $f$ is $X$-harmonic.

</div>

<details class="accordion" markdown="1">
<summary>Proof (Lemma 9.2.3)</summary>

**(a)** follows directly from the formula for $\mathcal{A}$.

**(b)** follows from the Dynkin formula: Choose $U$ as in Definition 9.2.2. Then

$$E^x[f(X_{\tau_U})] = \lim_{k \to \infty} E^x[f(X_{\tau_U \wedge k})] = f(x) + \lim_{k \to \infty} E^x\!\left[\int_0^{\tau_U \wedge k} (Lf)(X_s)\,ds\right] = f(x)$$

since $Lf = \mathcal{A}f = 0$ in $U$.

</details>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(9.2.4)</span></p>

Let $\phi$ be a bounded measurable function on $\partial D$ and put

$$u(x) = E^x[\phi(X_{\tau_D})]; \quad x \in D$$

Then $u$ is $X$-harmonic. Thus, in particular, $\mathcal{A}u = 0$.

</div>

<details class="accordion" markdown="1">
<summary>Proof (Lemma 9.2.4)</summary>

From the mean value property (7.2.9) we have, if $\overline{V} \subset D$:

$$u(x) = \int_{\partial V} u(y)\,Q^x[X_{\tau_V} \in dy] = E^x[u(X_{\tau_V})]$$

</details>

**The Stochastic Dirichlet Problem.** Given a bounded measurable function $\phi$ on $\partial D$, find a function $u$ on $D$ such that

$$(i)_s \quad u \text{ is } X\text{-harmonic}$$

$$(ii)_s \quad \lim_{t \uparrow \tau_D} u(X_t) = \phi(X_{\tau_D}) \quad \text{a.s. } Q^x, \; x \in D$$

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.2.5 — Solution of the Stochastic Dirichlet Problem)</span></p>

Let $\phi$ be a bounded measurable function on $\partial D$.

**a)** *(Existence)* Define

$$u(x) = E^x[\phi(X_{\tau_D})]$$

Then $u$ solves the stochastic Dirichlet problem $(i)_s$, $(ii)_s$.

**b)** *(Uniqueness)* Suppose $g$ is a bounded function on $D$ such that
1. $g$ is $X$-harmonic
2. $\lim_{t \uparrow \tau_D} g(X_t) = \phi(X_{\tau_D})$ a.s. $Q^x$, $x \in D$.

Then $g(x) = E^x[\phi(X_{\tau_D})]$, $x \in D$.

</div>

<details class="accordion" markdown="1">
<summary>Proof (Theorem 9.2.5)</summary>

**a)** It follows from Lemma 9.2.4 that $(i)_s$ holds. Fix $x \in D$. Let $\lbrace D_k\rbrace$ be an increasing sequence of open sets such that $D_k \subset\subset D$ and $D = \bigcup_k D_k$. Put $\tau_k = \tau_{D_k}$, $\tau = \tau_D$. Then by the strong Markov property

$$u(X_{\tau_k}) = E^{X_{\tau_k}}[\phi(X_\tau)] = E^x[\phi(X_\tau) \mid \mathcal{F}_{\tau_k}]$$

Now $M_k = E^x[\phi(X_\tau) \mid \mathcal{F}_{\tau_k}]$ is a bounded (discrete time) martingale so by the martingale convergence theorem:

$$\lim_{k \to \infty} u(X_{\tau_k}) = \lim_{k \to \infty} E^x[\phi(X_\tau) \mid \mathcal{F}_{\tau_k}] = \phi(X_\tau)$$

both pointwise for a.a. $\omega$ and in $L^p(Q^x)$, for all $p < \infty$. Moreover, for each $k$ the process

$$N_t = u(X_{\tau_k \vee (t \wedge \tau_{k+1})}) - u(X_{\tau_k}); \quad t \ge 0$$

is a martingale w.r.t. $\mathcal{G}_t = \mathcal{F}_{\tau_k \vee (t \wedge \tau_{k+1})}$. So by the martingale inequality:

$$Q^x\!\left[\sup_{\tau_k \le r \le \tau_{k+1}} \lvert u(X_r) - u(X_{\tau_k})\rvert > \epsilon\right] \le \frac{1}{\epsilon^2} E^x[\lvert u(X_{\tau_{k+1}}) - u(X_{\tau_k})\rvert^2] \to 0$$

as $k \to \infty$, for all $\epsilon > 0$. This gives $(ii)_s$.

**b)** Let $D_k, \tau_k$ be as in a). Then since $g$ is $X$-harmonic we have $g(x) = E^x[g(X_{\tau_k})]$ for all $k$. So by (2) and bounded convergence

$$g(x) = \lim_{k \to \infty} E^x[g(X_{\tau_k})] = E^x[\phi(X_{\tau_D})]$$

as asserted.

</details>

Finally we return to the original Dirichlet problem. We have already seen that a solution need not exist. However, it turns out that for a large class of processes $X_t$ we do get a solution (for all $D$) if we reduce the requirement in the boundary condition to hold only for a subset of the boundary points $y \in \partial D$ called the **regular** boundary points. Before we define regular points we need the following auxiliary lemmas.

(As before we let $\mathcal{M}_t$ and $\mathcal{M}_\infty$ denote the $\sigma$-algebras generated by $X_s$; $s \le t$ and by $X_s$; $s \ge 0$ respectively).

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(9.2.6 — The 0–1 Law)</span></p>

Let $H \in \bigcap_{t > 0} \mathcal{M}_t$. Then either $Q^x(H) = 0$ or $Q^x(H) = 1$.

</div>

<details class="accordion" markdown="1">
<summary>Proof (Lemma 9.2.6)</summary>

From the strong Markov property (7.2.5) we have $E^x[\theta_t \eta \mid \mathcal{M}_t] = E^{X_t}[\eta]$ for all bounded, $\mathcal{M}_\infty$-measurable $\eta\colon\Omega \to \mathbb{R}$. This implies that

$$\int_H \theta_t \eta \cdot dQ^x = \int_H E^{X_t}[\eta]\,dQ^x, \quad \text{for all } t$$

First assume that $\eta = \eta_k = g_1(X_{t_1}) \cdots g_k(X_{t_k})$, where each $g_i$ is bounded and continuous. Then letting $t \to 0$ we obtain

$$\int_H \eta\,dQ^x = \lim_{t \to 0} \int_H \theta_t \eta\,dQ^x = \lim_{t \to 0} \int_H E^{X_t}[\eta]\,dQ^x = Q^x(H) E^x[\eta]$$

by Feller continuity (Lemma 8.1.4) and bounded convergence. Approximating the general $\eta$ by functions $\eta_k$ as above we conclude that

$$\int_H \eta\,dQ^x = Q^x(H) E^x[\eta]$$

for all bounded $\mathcal{M}_\infty$-measurable $\eta$. If we put $\eta = \mathcal{X}_H$ we obtain $Q^x(H) = (Q^x(H))^2$, which completes the proof.

</details>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(9.2.7)</span></p>

Let $y \in \mathbb{R}^n$. Then

$$\text{either} \quad Q^y[\tau_D = 0] = 0 \quad \text{or} \quad Q^y[\tau_D = 0] = 1$$

</div>

In other words, either a.a. paths $X_t$ starting from $y$ stay within $D$ for a positive period of time or a.a. paths $X_t$ starting from $y$ leave $D$ immediately. In the last case we call the point $y$ **regular**.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(9.2.8 — Regular and Irregular Points)</span></p>

A point $y \in \partial D$ is called **regular** for $D$ (w.r.t. $X_t$) if

$$Q^y[\tau_D = 0] = 1$$

Otherwise the point $y$ is called **irregular**.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(9.2.9 — Regular points for Brownian motion)</span></p>

Corollary 9.2.7 may seem hard to believe at first glance. For example, if $X_t$ is a 2-dimensional Brownian motion $B_t$ and $\overline{D}$ is the square $[0,1] \times [0,1]$, one might think that, starting from $(\tfrac{1}{2},0)$, say, half of the paths will stay in the upper half plane and half in the lower, for a positive period of time. However, Corollary 9.2.7 says that this is not the case: Either they all stay in $D$ initially or they all leave $D$ immediately. Symmetry considerations imply that the first alternative is impossible. Thus $(\tfrac{1}{2},0)$, and similarly all the other points of $\partial D$, are regular for $D$ w.r.t. $B_t$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(9.2.10 — Irregular points for the graph of Brownian motion)</span></p>

Let $D = [0,1] \times [0,1]$ and let $L$ be the parabolic differential operator

$$Lf(t,x) = \frac{\partial f}{\partial t} + \frac{1}{2} \cdot \frac{\partial^2 f}{\partial x^2}; \quad (t,x) \in \mathbb{R}^2$$

Here $b = \binom{1}{0}$ and $a = [a_{ij}] = \frac{1}{2}\binom{0\ 0}{0\ 1}$, so if we choose $\sigma = \binom{0\ 0}{1\ 0}$, we have $\frac{1}{2}\sigma\sigma^T = a$. This gives the following SDE for the Itô diffusion $X_t$ associated with $L$:

$$dX_t = \binom{1}{0}dt + \binom{0\ 0}{1\ 0}\binom{dB_t^{(1)}}{dB_t^{(2)}}$$

In other words, $X_t = \binom{t+t_0}{B_t}$, $X_0 = \binom{t_0}{x}$, where $B_t$ is 1-dimensional Brownian motion. So we end up with the graph of Brownian motion. In this case it is not hard to see that the irregular points of $\partial D$ consist of the open line $\lbrace 0\rbrace \times (0,1)$, the rest of the boundary points being regular.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(9.2.11 — Wiener criterion)</span></p>

Let $\Delta = \lbrace(x,y); x^2 + y^2 < 1\rbrace \subset \mathbb{R}^2$ and let $\lbrace\Delta_n\rbrace$ be a sequence of disjoint open discs in $\Delta$ centered at $(2^{-n},0)$, respectively, $n = 1,2,\ldots$. Put

$$D = \Delta \setminus \overline{\bigcup_{n=1}^\infty \Delta_n}$$

Then it is easy to see that all the points of $\partial\Delta \cup \bigcup_{n=1}^\infty \partial\Delta_n$ are regular for $D$ w.r.t. 2-dimensional Brownian motion $B_t$. But what about the point 0? The answer depends on the sizes of the discs $\Delta_n$. More precisely, if $r_n$ is the radius of $\Delta_n$ then 0 is a regular point for $D$ if and only if

$$\sum_{n=1}^\infty \frac{n}{\log \frac{1}{r_n}} = \infty$$

This is a consequence of the famous **Wiener criterion**.

</div>

Having defined regular points we now formulate the announced generalized version of the Dirichlet problem:

**The Generalized Dirichlet Problem.** Given a domain $D \subset \mathbb{R}^n$ and $L$ and $\phi$ as before, find a function $u \in C^2(D)$ such that

$$Lu = 0 \quad \text{in } D$$

and

$$\lim_{\substack{x \to y \\ x \in D}} u(x) = \phi(y) \quad \text{for all \textit{regular} } y \in \partial D$$

First we establish that if a solution of this problem exists, it must be the solution of the stochastic Dirichlet problem found in Theorem 9.2.5, provided that $X_t$ satisfies **Hunt's condition** (H):

**(H):** Every semipolar set for $X_t$ is polar for $X_t$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Semipolar, Thin, and Polar Sets)</span></p>

A **semipolar** set is a countable union of *thin* sets. A measurable set $G \subset \mathbb{R}^n$ is called **thin** (for $X_t$) if $Q^x[T_G = 0] = 0$ for all $x$, where $T_G = \inf\lbrace t > 0; X_t \in G\rbrace$ is the first hitting time of $G$. (Intuitively: for all starting points the process does not hit $G$ immediately, a.s.)

A measurable set $F \subset \mathbb{R}^n$ is called **polar** (for $X_t$) if $Q^x[T_F < \infty] = 0$ for all $x$. (Intuitively: for all starting points the process *never* hits $F$, a.s.)

Clearly every polar set is semipolar, but the converse need not be true (consider the process in Example 9.2.1). However, condition (H) does hold for Brownian motion. It follows from the Girsanov theorem that condition (H) holds for all Itô diffusions whose diffusion coefficient matrix has a bounded inverse and whose drift coefficient satisfies the Novikov condition for all $T < \infty$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(9.2.12)</span></p>

Let $U \subset D$ be open and let $I$ denote the set of irregular points of $U$. Then $I$ is a semipolar set.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.2.13)</span></p>

Suppose $X_t$ satisfies Hunt's condition (H). Let $\phi$ be a bounded continuous function on $\partial D$. Suppose there exists a bounded $u \in C^2(D)$ such that

- (i) $Lu = 0$ in $D$
- (ii)$_r$ $\displaystyle\lim_{\substack{x \to y \\ x \in D}} u(x) = \phi(y)$ for all regular $y \in \partial D$

Then $u(x) = E^x[\phi(X_{\tau_D})]$.

</div>

<details class="accordion" markdown="1">
<summary>Proof (Theorem 9.2.13)</summary>

Let $\lbrace D_k\rbrace$ be as in the proof of Theorem 9.1.1. By Lemma 9.2.3 b) $u$ is $X$-harmonic and therefore

$$u(x) = E^x[u(X_{\tau_k})] \quad \text{for all } x \in D_k \text{ and all } k$$

If $k \to \infty$ then $X_{\tau_k} \to X_{\tau_D}$ and so $u(X_{\tau_k}) \to \phi(X_{\tau_D})$ if $X_{\tau_D}$ is regular. From Lemma 9.2.12 we know that the set $I$ of irregular points of $\partial D$ is semipolar. So by condition (H) the set $I$ is polar and therefore $X_{\tau_D} \notin I$ a.s. $Q^x$. Hence

$$u(x) = \lim E^x[u(X_{\tau_k})] = E^x[\phi(X_{\tau_D})]$$

as claimed.

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.2.14 — Solution of the Generalized Dirichlet Problem)</span></p>

Suppose $L$ is **uniformly elliptic** in $D$, i.e. the eigenvalues of $[a_{ij}]$ are bounded away from 0 in $D$. Let $\phi$ be a bounded continuous function on $\partial D$. Put

$$u(x) = E^x[\phi(X_{\tau_D})]$$

Then $u \in C^{2+\alpha}(D)$ for all $\alpha < 1$ and $u$ solves the Dirichlet problem, i.e.

- (i) $Lu = 0$ in $D$.
- (ii)$_r$ $\displaystyle\lim_{\substack{x \to y \\ x \in D}} u(x) = \phi(y)$ for all regular $y \in \partial D$.

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Theorem 9.2.14)</summary>

Choose an open ball $\Delta$ with $\overline{\Delta} \subset D$ and let $f \in C(\partial\Delta)$. Then, from the general theory of partial differential equations, for all $\alpha < 1$ there exists a continuous function $u$ on $\overline{\Delta}$ such that $u\vert_\Delta \in C^{2+\alpha}(\Delta)$ and $Lu = 0$ in $\Delta$, $u = f$ on $\partial\Delta$. Since $u\vert_\Delta \in C^{2+\alpha}(\Delta)$ we have: If $K$ is any compact subset of $\Delta$ there exists a constant $C$ only depending on $K$ and the $C^\alpha$-norms of the coefficients of $L$ such that

$$\|u\|_{C^{2+\alpha}(K)} \le C(\|Lu\|_{C^\alpha(\Delta)} + \|u\|_{C(\Delta)}) \le C\|f\|_{C(\partial\Delta)}$$

By uniqueness (Theorem 9.2.13) we know that $u(x) = \int f(y)\,d\mu_x(y)$ where $d\mu_x = Q^x[X_{\tau_\Delta} \in dy]$ is the first exit distribution of $X_t$ from $\Delta$. From the Schauder estimate it follows that $\|\mu_{x_1} - \mu_{x_2}\| \le C\lvert x_1 - x_2\rvert^\alpha$ for $x_1, x_2 \in K$. So if $g$ is any bounded measurable function on $\partial\Delta$, the function $\widetilde{g}(x) = \int g(y)\,d\mu_x(y) = E^x[g(X_{\tau_\Delta})]$ belongs to $C^\alpha(K)$. Since $u(x) = E^x[u(X_{\tau_U})]$ for all open sets $U$ with $\overline{U} \subset D$ (Lemma 9.2.4), this applies to $g = u$ and we conclude that $u \in C^\alpha(M)$ for any compact subset $M$ of $D$.

We may therefore apply the solution to the problem $Lu = 0$ in $\Delta$, $u = f$ on $\partial\Delta$ once more, this time with $f = u$, and this way we obtain that $u(x) = E^x[u(X_{\tau_D})]$ belongs to $C^{2+\alpha}(M)$ for any compact $M \subset D$. Therefore (i) holds by Lemma 9.2.3 a).

To obtain (ii)$_r$ we apply a theorem from the theory of parabolic differential equations: The Kolmogorov backward equation $Lv = \frac{\partial v}{\partial t}$ has a fundamental solution $v = p(t,x,y)$ jointly continuous in $t,x,y$ for $t > 0$ and bounded in $x,y$ for each fixed $t > 0$. It follows that the process $X_t$ is a **strong Feller process**, in the sense that $x \to E^x[f(X_t)] = \int f(y)p(t,x,y)\,dy$ is continuous for all $t > 0$ and all bounded, measurable functions $f$. In general we have: If $X_t$ is a strong Feller Itô diffusion and $D \subset \mathbb{R}^n$ is open then

$$\lim_{\substack{x \to y \\ x \in D}} E^x[\phi(X_{\tau_D})] = \phi(y)$$

for all regular $y \in \partial D$ and bounded $\phi \in C(\partial D)$. Therefore $u$ satisfies property (ii)$_r$.

</details>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(9.2.15 — Boundary condition can fail at irregular points)</span></p>

We have already seen (Example 9.2.1) that condition (9.1.3) does not hold in general. This example shows that it need not hold even when $L$ is elliptic: Consider Example 9.2.11 again, in the case when the point 0 is not regular. Choose $\phi \in C(\partial D)$ such that $\phi(0) = 1$, $0 \le \phi(y) < 1$ for $y \in \partial D \setminus \lbrace 0\rbrace$. Since $\lbrace 0\rbrace$ is polar for $B_t$ (see Exercise 9.7 a) we have $B_{\tau_D}^0 \ne 0$ a.s. and therefore

$$u(0) = E^0[\phi(B_{\tau_D})] < 1$$

By a slight extension of the mean value property (7.2.9) we get

$$E^0[u(X_{\sigma_k})] = E^0[\phi(X_{\tau_D})] = u(0) < 1$$

where $\sigma_k = \inf\lbrace t > 0; B_t \notin D \cap \lbrace\lvert x\rvert < \frac{1}{k}\rbrace\rbrace$, $k = 1,2,\ldots$. This implies that it is impossible that $u(x) \to 1$ as $x \to 0$. Therefore (9.1.3) does not hold in this case.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(9.2.16 — The infinite strip)</span></p>

Let $D$ denote the infinite strip $D = \lbrace(t,x) \in \mathbb{R}^2; x < R\rbrace$, where $R \in \mathbb{R}$, and let $L$ be the differential operator

$$Lf(t,x) = \frac{\partial f}{\partial t} + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}; \quad f \in C^2(D)$$

An Itô diffusion whose generator coincides with $L$ on $C_0^2(\mathbb{R}^2)$ is (see Example 9.2.10) $X_t = (s + t, B_t)$; $t \ge 0$, and all the points of $\partial D$ are regular for this process. It is not hard to see that $\tau_D < \infty$ a.s. Assume that $\phi$ is a bounded continuous function on $\partial D = \lbrace(t,R); t \in \mathbb{R}\rbrace$. Then by Theorem 9.2.5 the function

$$u(s,x) = E^{s,x}[\phi(X_{\tau_D})]$$

is the solution of the stochastic Dirichlet problem. Using the Laplace transform it is possible to find the distribution of the first time $\hat{\tau}$ that $B_t$ reaches the value $R$:

$$P^x[\hat{\tau} \in dt] = g(x,t)\,dt$$

where

$$g(x,t) = \begin{cases} (R-x)(2\pi t^3)^{-1} \exp\!\left(-\frac{(R-x)^2}{2t}\right); & t > 0 \\ 0; & t \le 0 \end{cases}$$

Thus the solution $u$ may be written

$$u(s,x) = \int_0^\infty \phi(s+t, R)\,g(x,t)\,dt = \int_s^\infty \phi(r, R)\,g(x, r-s)\,dr$$

From the explicit formula for $u$ it is clear that $\frac{\partial u}{\partial s}$ and $\frac{\partial^2 u}{\partial x^2}$ are continuous and we conclude that $Lu = 0$ in $D$ by Lemma 9.2.3. So $u$ satisfies (9.2.13). For property (9.2.14), it is not hard to verify directly that if $\lvert y\rvert = R$, $t_1 > 0$ then for all $\epsilon > 0$ there exists $\delta > 0$ such that $\lvert x - y\rvert < \delta$, $\lvert t - t_1\rvert < \delta \Rightarrow Q^{t,x}[X_{\tau_D} \in N] \ge 1 - \epsilon$, where $N = [t_1 - \epsilon, t_1 + \epsilon] \times \lbrace y\rbrace$. And this is easily seen to imply (9.2.14).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

As the above example (and Example 9.2.1) shows, an Itô diffusion need not be a *strong* Feller process. However, we have seen that it is always a Feller process (Lemma 8.1.4).

</div>

### 9.3 The Poisson Problem

Let $L = \sum a_{ij}\frac{\partial^2}{\partial x_i \partial x_j} + \sum b_i \frac{\partial}{\partial x_i}$ be a semi-elliptic partial differential operator on a domain $D \subset \mathbb{R}^n$ as before and let $X_t$ be an associated Itô diffusion. For the same reasons as in Section 9.2 we generalize the problem to the following:

**The Generalized Poisson Problem.** Given a continuous function $g$ on $D$, find a $C^2$ function $v$ in $D$ such that

$$Lv = -g \quad \text{in } D$$

and

$$\lim_{\substack{x \to y \\ x \in D}} v(x) = 0 \quad \text{for all \textit{regular} } y \in \partial D$$

Again we will first study a stochastic version of the problem and then investigate the relation between the corresponding stochastic solution and the solution (if it exists) of the generalized problem.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.3.1 — Solution of the Stochastic Poisson Problem)</span></p>

Assume that

$$E^x\!\left[\int_0^{\tau_D} \lvert g(X_s)\rvert\,ds\right] < \infty \quad \text{for all } x \in D$$

Define

$$v(x) = E^x\!\left[\int_0^{\tau_D} g(X_s)\,ds\right]$$

Then

$$\mathcal{A}v = -g \quad \text{in } D$$

and

$$\lim_{t \uparrow \tau_D} v(X_t) = 0 \quad \text{a.s. } Q^x, \text{ for all } x \in D$$

</div>

<details class="accordion" markdown="1">
<summary>Proof (Theorem 9.3.1)</summary>

Choose $U$ open, $x \in U \subset\subset D$. Put $\eta = \int_0^{\tau_D} g(X_s)\,ds$, $\tau = \tau_U$. Then by the strong Markov property (7.2.5)

$$\frac{E^x[v(X_\tau)] - v(x)}{E^x[\tau]} = \frac{1}{E^x[\tau]}(E^x[E^{X_\tau}[\eta]] - E^x[\eta])$$

$$= \frac{1}{E^x[\tau]}(E^x[E^x[\theta_\tau \eta \mid \mathcal{F}_\tau]] - E^x[\eta]) = \frac{1}{E^x[\tau]} E^x[\theta_\tau \eta - \eta]$$

Approximate $\eta$ by sums of the form $\eta^{(k)} = \sum g(X_{t_i})\mathcal{X}_{\lbrace t_i < \tau_D\rbrace}\Delta t_i$. Since $\theta_\tau \eta^{(k)} = \sum g(X_{t_i+\tau})\mathcal{X}_{\lbrace t_i+\tau < \tau_D\rbrace}\Delta t_i$ for all $k$, we see that $\theta_\tau \eta = \int_\tau^{\tau_D} g(X_s)\,ds$. Therefore

$$\frac{E^x[v(X_\tau)] - v(x)}{E^x[\tau]} = \frac{-1}{E^x[\tau]} E^x\!\left[\int_0^\tau g(X_s)\,ds\right] \to -g(x) \quad \text{as } U \downarrow x$$

since $g$ is continuous. This proves $\mathcal{A}v = -g$.

Put $H(x) = E^x\!\left[\int_0^{\tau_D} \lvert g(X_s)\rvert\,ds\right]$. Let $D_k, \tau_k$ be as in the proof of Theorem 9.2.5. Then by the same argument as above we get

$$E^x[H(X_{\tau_k \wedge t})] = E^x\!\left[E^x\!\left[\int_{\tau_k \wedge t}^{\tau_D} \lvert g(X_s)\rvert\,ds \,\Big\vert\, \mathcal{F}_{\tau_k \wedge t}\right]\right] = E^x\!\left[\int_{\tau_k \wedge t}^{\tau_D} \lvert g(X_s)\rvert\,ds\right] \to 0$$

as $t \to \tau_D$, $k \to \infty$, by dominated convergence. This implies $\lim_{t \uparrow \tau_D} v(X_t) = 0$.

</details>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

For functions $g$ satisfying the integrability condition, define the operator $\mathcal{R}$ by

$$(\mathcal{R}g)(x) = \check{g}(x) = E^x\!\left[\int_0^{\tau_D} g(X_s)\,ds\right]$$

Then $\mathcal{A}(\mathcal{R}g) = -g$, i.e. the operator $-\mathcal{R}$ is a right inverse of the operator $\mathcal{A}$. Similarly, if we define

$$\mathcal{R}_\alpha g(x) = E^x\!\left[\int_0^{\tau_D} g(X_s)e^{-\alpha s}\,ds\right] \quad \text{for } \alpha \ge 0$$

then the same method of proof as in Theorem 8.1.5 gives that $(\mathcal{A} - \alpha)\mathcal{R}_\alpha g = -g$; $\alpha \ge 0$.

Thus we may regard the operator $\mathcal{R}_\alpha$ as a generalization of the resolvent operator $R_\alpha$ discussed in Chapter 8.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.3.2 — Uniqueness Theorem for the Poisson Equation)</span></p>

Assume that $X_t$ satisfies Hunt's condition (H). Assume that the integrability condition $E^x\!\left[\int_0^{\tau_D} \lvert g(X_s)\rvert\,ds\right] < \infty$ holds and that there exists a function $v \in C^2(D)$ and a constant $C$ such that

$$\lvert v(x)\rvert \le C\!\left(1 + E^x\!\left[\int_0^{\tau_D} \lvert g(X_s)\rvert\,ds\right]\right) \quad \text{for all } x \in D$$

and with the properties

$$Lv = -g \quad \text{in } D$$

$$\lim_{\substack{x \to y \\ x \in D}} v(x) = 0 \quad \text{for all regular points } y \in \partial D$$

Then $v(x) = E^x\!\left[\int_0^{\tau_D} g(X_s)\,ds\right]$.

</div>

<details class="accordion" markdown="1">
<summary>Proof (Theorem 9.3.2)</summary>

Let $D_k, \tau_k$ be as in the proof of Theorem 9.2.5. Then by Dynkin's formula

$$E^x[v(X_{\tau_k})] - v(x) = E^x\!\left[\int_0^{\tau_k} (Lv)(X_s)\,ds\right] = -E^x\!\left[\int_0^{\tau_k} g(X_s)\,ds\right]$$

By dominated convergence we obtain

$$v(x) = \lim_{k \to \infty}\!\left(E^x[v(X_{\tau_k})] + E^x\!\left[\int_0^{\tau_k} g(X_s)\,ds\right]\right) = E^x\!\left[\int_0^{\tau_D} g(X_s)\,ds\right]$$

since $X_{\tau_D}$ is a regular point a.s. by condition (H) and Lemma 9.2.12.

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.3.3 — Solution of the Combined Stochastic Dirichlet and Poisson Problem)</span></p>

Assume that $\tau_D < \infty$ a.s. holds. Let $\phi \in C(\partial D)$ be bounded and let $g \in C(D)$ satisfy

$$E^x\!\left[\int_0^{\tau_D} \lvert g(X_s)\rvert\,ds\right] < \infty \quad \text{for all } x \in D$$

Define

$$w(x) = E^x[\phi(X_{\tau_D})] + E^x\!\left[\int_0^{\tau_D} g(X_s)\,ds\right], \quad x \in D$$

**a)** Then

$$\mathcal{A}w = -g \quad \text{in } D$$

and

$$\lim_{t \uparrow \tau_D} w(X_t) = \phi(X_{\tau_D}) \quad \text{a.s. } Q^x, \text{ for all } x \in D$$

**b)** Moreover, if there exists a function $w_1 \in C^2(D)$ and a constant $C$ such that

$$\lvert w_1(x)\rvert < C\!\left(1 + E^x\!\left[\int_0^{\tau_D} \lvert g(X_s)\rvert\,ds\right]\right), \quad x \in D$$

and $w_1$ satisfies $\mathcal{A}w_1 = -g$ and $\lim_{t \uparrow \tau_D} w_1(X_t) = \phi(X_{\tau_D})$, then $w_1 = w$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

With an approach similar to the one used in Theorem 9.2.14 one can prove that if $L$ is uniformly elliptic in $D$ and $g \in C^\alpha(D)$ (for some $\alpha > 0$) is bounded, then the function $w$ given by $w(x) = E^x[\phi(X_{\tau_D})] + E^x\!\left[\int_0^{\tau_D} g(X_s)\,ds\right]$ solves the Dirichlet–Poisson problem, i.e. $Lw = -g$ in $D$ and $\lim_{\substack{x \to y \\ x \in D}} w(x) = \phi(y)$ for all regular $y \in \partial D$.

</div>

#### The Green Measure

The solution $v$ given by the formula $v(x) = E^x\!\left[\int_0^{\tau_D} g(X_s)\,ds\right]$ may be rewritten as follows:

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(9.3.4 — The Green Measure)</span></p>

The **Green measure** (of $X_t$ w.r.t. $D$ at $x$), $G(x,\cdot)$, is defined by

$$G(x, H) = E^x\!\left[\int_0^{\tau_D} \mathcal{X}_H(X_s)\,ds\right], \quad H \subset \mathbb{R}^n \text{ Borel set}$$

or equivalently

$$\int f(y)\,G(x,dy) = E^x\!\left[\int_0^{\tau_D} f(X_s)\,ds\right], \quad f \text{ bounded, continuous}$$

In other words, $G(x,H)$ is the expected length of time the process stays in $H$ before it exits from $D$. If $X_t$ is Brownian motion, then $G(x,H) = \int_H G(x,y)\,dy$, where $G(x,y)$ is the classical Green function w.r.t. $D$ and $dy$ denotes Lebesgue measure.

</div>

Note that using the Fubini theorem we obtain the following relation between the Green measure $G$ and the *transition measure* for $X_t$ in $D$, $Q_t^D(x,H) = Q^x[X_t \in H, t < \tau_D]$:

$$G(x,H) = E^x\!\left[\int_0^\infty \mathcal{X}_H(X_s) \cdot \mathcal{X}_{[0,\tau_D)}(s)\,ds\right] = \int_0^\infty Q_t^D(x,H)\,dt$$

From this we get

$$v(x) = E^x\!\left[\int_0^{\tau_D} g(X_s)\,ds\right] = \int_D g(y)\,G(x,dy)$$

which is the familiar formula for the solution of the Poisson equation in the classical case.

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(9.3.5 — The Green Formula)</span></p>

Let $E^x[\tau_D] < \infty$ for all $x \in D$ and assume that $f \in C_0^2(\mathbb{R}^n)$. Then

$$f(x) = E^x[f(X_{\tau_D})] - \int_D (L_X f)(y)\,G(x,dy)$$

In particular, if $f \in C_0^2(D)$ we have

$$f(x) = -\int_D (L_X f)(y)\,G(x,dy)$$

(As before $L_X = \sum b_i \frac{\partial}{\partial x_i} + \frac{1}{2}\sum (\sigma\sigma^T)_{ij}\frac{\partial^2}{\partial x_i \partial x_j}$ when $dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t$.)

</div>

<details class="accordion" markdown="1">
<summary>Proof (Corollary 9.3.5)</summary>

By Dynkin's formula and the Green measure formula we have

$$E^x[f(X_{\tau_D})] = f(x) + E^x\!\left[\int_0^{\tau_D} (L_X f)(X_s)\,ds\right] = f(x) + \int_D (L_X f)(y)\,G(x,dy)$$

</details>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Combining the Green formula with the resolvent properties, we see that if $E^x[\tau_K] < \infty$ for all compacts $K \subset D$ and all $x \in D$, then $-\mathcal{R}$ is the inverse of the operator $\mathcal{A}$ on $C_0^2(D)$:

$$\mathcal{A}(\mathcal{R}f) = \mathcal{R}(\mathcal{A}f) = -f, \quad \text{for all } f \in C_0^2(D)$$

More generally, for all $\alpha \ge 0$ we get the following analogue of Theorem 8.1.5:

$$(\mathcal{A} - \alpha)(\mathcal{R}_\alpha f) = \mathcal{R}_\alpha(\mathcal{A} - \alpha)f = -f \quad \text{for all } f \in C_0^2(D)$$

This is valid for all stopping times $\tau \le \infty$ and all $f \in C_0^2(\mathbb{R}^n)$, and follows from the useful extension of the Dynkin formula:

$$E^x[e^{-\alpha\tau}f(X_\tau)] = f(x) + E^x\!\left[\int_0^\tau e^{-\alpha s}(\mathcal{A} - \alpha)f(X_s)\,ds\right]$$

if $\alpha > 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(9.3.6 — Green function of Brownian motion on an interval)</span></p>

If $X_t = B_t$ is 1-dimensional Brownian motion in a bounded interval $(a,b) \subset \mathbb{R}$, then we can compute the Green function $G(x,y)$ explicitly. To this end, choose a bounded continuous function $g\colon(a,b) \to \mathbb{R}$ and let us compute

$$v(x) := E^x\!\left[\int_0^{\tau_D} g(B_t)\,dt\right]$$

By Corollary 9.1.2 we know that $v$ is the solution of the differential equation $\frac{1}{2}v''(x) = -g(x)$; $x \in (a,b)$, $v(a) = v(b) = 0$. Integrating twice and using the boundary conditions we get

$$v(x) = \frac{2(x-a)}{b-a}\int_a^b\!\left(\int_a^y g(z)\,dz\right)dy - 2\int_a^x\!\left(\int_a^y g(z)\,dz\right)dy$$

Changing the order of integration we can rewrite this as $v(x) = \int_a^b g(y)G(x,y)\,dy$ where

$$G(x,y) = \frac{2(x-a)(b-y)}{b-a} - 2(x-y) \cdot \mathcal{X}_{(-\infty,x)}(y)$$

We conclude that *the Green function of Brownian motion in the interval $(a,b)$ is given by this formula*.

In higher dimension $n$ the Green function $y \to G(x,y)$ of Brownian motion starting at $x$ will not be continuous at $x$. It will have a logarithmic singularity (i.e. a singularity of order $\ln\frac{1}{\lvert x-y\rvert}$) for $n = 2$ and a singularity of the order $\lvert x - y\rvert^{2-n}$ for $n > 2$.

</div>

## 10. Application to Optimal Stopping

### 10.1 The Time-Homogeneous Case

Problem 5 in the introduction is a special case of a problem of the following type:

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(10.1.1 — The Optimal Stopping Problem)</span></p>

Let $X_t$ be an Itô diffusion on $\mathbb{R}^n$ and let $g$ (*the reward function*) be a given function on $\mathbb{R}^n$, satisfying

- $g(\xi) \ge 0$ for all $\xi \in \mathbb{R}^n$
- $g$ is continuous.

Find a stopping time $\tau^* = \tau^*(x,\omega)$ (called an **optimal stopping time**) for $\lbrace X_t \rbrace$ such that

$$E^x[g(X_{\tau^*})] = \sup_\tau E^x[g(X_\tau)] \quad \text{for all } x \in \mathbb{R}^n,$$

the sup being taken over all stopping times $\tau$ for $\lbrace X_t \rbrace$. We also want to find the corresponding **optimal expected reward**

$$g^*(x) = E^x[g(X_{\tau^*})].$$

Here $g(X_\tau)$ is to be interpreted as $0$ at the points $\omega \in \Omega$ where $\tau(\omega) = \infty$, and as usual $E^x$ denotes the expectation with respect to the probability law $Q^x$ of the process $X_t$; $t \ge 0$ starting at $X_0 = x \in \mathbb{R}^n$.

</div>

We may regard $X_t$ as the state of a game at time $t$, each $\omega$ corresponds to one sample of the game. For each time $t$ we have the option of stopping the game, thereby obtaining the reward $g(X_t)$, or continue the game in the hope that stopping it at a later time will give a bigger reward. The problem is of course that we do not know what state the game is in at future times, only the probability distribution of the "future". Mathematically, this means that the possible "stopping" times we consider really are stopping times in the sense of Definition 7.2.1: The decision whether $\tau \le t$ or not should only depend on the behaviour of the Brownian motion $B_r$ (driving the process $X$) up to time $t$, or perhaps only on the behaviour of $X_r$ up to time $t$. So, among all possible stopping times $\tau$ we are asking for the optimal one, $\tau^*$, which gives the best result "in the long run", i.e. the biggest expected reward.

The discussion of problem (10.1.2)–(10.1.3) also covers the apparently more general problems

$$g^*(s,x) = \sup_\tau E^{(s,x)}[g(\tau, X_\tau)] = E^{(s,x)}[g(\tau^*, X_{\tau^*})]$$

and

$$G^*(s,x) = \sup_\tau E^{(s,x)}\!\left[\int_0^\tau f(t,X_t)\,dt + g(\tau, X_\tau)\right] = E^{(s,x)}\!\left[\int_0^{\tau^*} f(t,X_t)\,dt + g(\tau^*, X_{\tau^*})\right]$$

where $f$ is a given *reward rate function* (satisfying certain conditions).

#### Supermeanvalued and Superharmonic Functions

A basic concept in the solution of the optimal stopping problem is the following:

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(10.1.2 — Supermeanvalued and Superharmonic)</span></p>

A measurable function $f\colon\mathbb{R}^n \to [0,\infty]$ is called **supermeanvalued** (w.r.t. $X_t$) if

$$f(x) \ge E^x[f(X_\tau)]$$

for all stopping times $\tau$ and all $x \in \mathbb{R}^n$.

If, in addition, $f$ is also lower semicontinuous, then $f$ is called **l.s.c. superharmonic** or just **superharmonic** (w.r.t. $X_t$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On Superharmonicity)</span></p>

**1)** In the literature (e.g. Dynkin (1965 II)) one often finds a weaker concept of $X_t$-superharmonicity, defined by the supermeanvalued property plus the stochastic continuity requirement. This weaker concept corresponds to the $X_t$-harmonicity defined in Chapter 9.

**2)** If $f \in C^2(\mathbb{R}^n)$ it follows from Dynkin's formula that $f$ is superharmonic w.r.t. $X_t$ if and only if $\mathcal{A}f \le 0$, where $\mathcal{A}$ is the characteristic operator of $X_t$. This is often a useful criterion.

**3)** If $X_t = B_t$ is Brownian motion in $\mathbb{R}^n$ then the superharmonic functions for $X_t$ coincide with the (nonnegative) superharmonic functions in classical potential theory.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(10.1.3 — Properties of Superharmonic and Supermeanvalued Functions)</span></p>

- **a)** If $f$ is superharmonic (supermeanvalued) and $\alpha > 0$, then $\alpha f$ is superharmonic (supermeanvalued).
- **b)** If $f_1, f_2$ are superharmonic (supermeanvalued), then $f_1 + f_2$ is superharmonic (supermeanvalued).
- **c)** If $\lbrace f_j \rbrace_{j \in J}$ is a family of supermeanvalued functions, then $f(x) := \inf_{j \in J}\lbrace f_j(x)\rbrace$ is supermeanvalued if it is measurable ($J$ is any set).
- **d)** If $f_1, f_2, \cdots$ are superharmonic (supermeanvalued) functions and $f_k \uparrow f$ pointwise, then $f$ is superharmonic (supermeanvalued).
- **e)** If $f$ is supermeanvalued and $\sigma \le \tau$ are stopping times, then $E^x[f(X_\sigma)] \ge E^x[f(X_\tau)]$.
- **f)** If $f$ is supermeanvalued and $H$ is a Borel set, then $\widetilde{f}(x) := E^x[f(X_{\tau_H})]$ is supermeanvalued.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Lemma 10.1.3</summary>

**a)** and **b)** are straightforward.

**c)** Suppose $f_j$ is supermeanvalued for all $j \in J$. Then $f_j(x) \ge E^x[f_j(X_\tau)] \ge E^x[f(X_\tau)]$ for all $j$. So $f(x) = \inf f_j(x) \ge E^x[f(X_\tau)]$, as required.

**d)** Suppose $f_j$ is supermeanvalued, $f_j \uparrow f$. Then $f(x) \ge f_j(x) \ge E^x[f_j(X_\tau)]$ for all $j$, so $f(x) \ge \lim_{j \to \infty} E^x[f_j(X_\tau)] = E^x[f(X_\tau)]$ by monotone convergence. Hence $f$ is supermeanvalued. If each $f_j$ is also lower semicontinuous then if $y_k \to x$ as $k \to \infty$ we have $f_j(x) \le \varliminf_{k \to \infty} f_j(y_k) \le \varliminf_{k \to \infty} f(y_k)$ for each $j$. Hence, by letting $j \to \infty$, $f(x) \le \varliminf_{k \to \infty} f(y_k)$.

**e)** If $f$ is supermeanvalued we have by the Markov property when $t > s$: $E^x[f(X_t)\mid\mathcal{F}_s] = E^{X_s}[f(X_{t-s})] \le f(X_s)$, i.e. the process $\zeta_t = f(X_t)$ is a *supermartingale* w.r.t. the $\sigma$-algebras $\mathcal{F}_t$ generated by $\lbrace B_r; r \le t\rbrace$. Therefore, by Doob's optional sampling theorem, $E^x[f(X_\sigma)] \ge E^x[f(X_\tau)]$ for all stopping times $\sigma, \tau$ with $\sigma \le \tau$ a.s. $Q^x$.

**f)** Suppose $f$ is supermeanvalued. By the strong Markov property (7.2.2) and formula (7.2.6) we have, for any stopping time $\alpha$: $E^x[\widetilde{f}(X_\alpha)] = E^x[E^{X_\alpha}[f(X_{\tau_H})]] = E^x[E^x[\theta_\alpha f(X_{\tau_H})\mid\mathcal{F}_\alpha]] = E^x[\theta_\alpha f(X_{\tau_H})] = E^x[f(X_{\tau_H^\alpha})]$, where $\tau_H^\alpha = \inf\lbrace t > \alpha; X_t \notin H\rbrace$. Since $\tau_H^\alpha \ge \tau_H$ we have by e) $E^x[\widetilde{f}(X_\alpha)] \le E^x[f(X_{\tau_H})] = \widetilde{f}(x)$, so $\widetilde{f}$ is supermeanvalued. $\square$

</details>

#### The Least Superharmonic Majorant

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(10.1.4 — Superharmonic Majorant and Least Superharmonic Majorant)</span></p>

Let $h$ be a real measurable function on $\mathbb{R}^n$. If $f$ is a superharmonic (supermeanvalued) function and $f \ge h$ we say that $f$ is a **superharmonic (supermeanvalued) majorant** of $h$ (w.r.t. $X_t$). The function

$$\overline{h}(x) = \inf_f f(x); \quad x \in \mathbb{R}^n,$$

the inf being taken over all supermeanvalued majorants $f$ of $h$, is called the **least supermeanvalued majorant** of $h$.

Similarly, suppose there exists a function $\widehat{h}$ such that (i) $\widehat{h}$ is a superharmonic majorant of $h$ and (ii) if $f$ is any other superharmonic majorant of $h$ then $\widehat{h} \le f$. Then $\widehat{h}$ is called the **least superharmonic majorant** of $h$.

Note that by Lemma 10.1.3 c) the function $\overline{h}$ is supermeanvalued if it is measurable. Moreover, if $\overline{h}$ is lower semicontinuous, then $\widehat{h}$ exists and $\widehat{h} = \overline{h}$.

</div>

Let $g \ge 0$ and let $f$ be a supermeanvalued majorant of $g$. Then if $\tau$ is a stopping time, $f(x) \ge E^x[f(X_\tau)] \ge E^x[g(X_\tau)]$. So $f(x) \ge \sup_\tau E^x[g(X_\tau)] = g^*(x)$. Therefore we always have

$$\widehat{g}(x) \ge g^*(x) \quad \text{for all } x \in \mathbb{R}^n.$$

What is not so easy to see is that the converse inequality also holds, i.e. that in fact $\widehat{g} = g^*$. We will prove this after establishing useful iterative procedures for calculating $\widehat{g}$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(10.1.5 — Excessive Function)</span></p>

A lower semicontinuous function $f\colon\mathbb{R}^n \to [0,\infty]$ is called **excessive** (w.r.t. $X_t$) if

$$f(x) \ge E^x[f(X_s)] \quad \text{for all } s \ge 0,\ x \in \mathbb{R}^n.$$

</div>

It is clear that a superharmonic function must be excessive. What is not so obvious, is that the converse also holds:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(10.1.6 — Excessive iff Superharmonic)</span></p>

Let $f\colon\mathbb{R}^n \to [0,\infty]$. Then $f$ is excessive w.r.t. $X_t$ if and only if $f$ is superharmonic w.r.t. $X_t$.

</div>

<details class="accordion" markdown="1">
<summary>Proof (special case)</summary>

Let $L$ be the differential operator associated to $X$ (given by the right hand side of (7.3.3)), so that $L$ coincides with the generator $A$ of $X$ on $C_0^2$. We only prove the theorem in the special case when $f \in C^2(\mathbb{R}^n)$ and $Lf$ is bounded: Then by Dynkin's formula we have

$$E^x[f(X_t)] = f(x) + E^x\!\left[\int_0^t Lf(X_r)\,dr\right] \quad \text{for all } t \ge 0,$$

so if $f$ is excessive then $Lf \le 0$. Therefore, if $\tau$ is a stopping time we get $E^x[f(X_{t \wedge \tau})] \le f(x)$ for all $t \ge 0$. Letting $t \to \infty$ we see that $f$ is superharmonic. $\square$

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(10.1.7 — Construction of the Least Superharmonic Majorant)</span></p>

Let $g = g_0$ be a nonnegative, lower semicontinuous function on $\mathbb{R}^n$ and define inductively

$$g_n(x) = \sup_{t \in S_n} E^x[g_{n-1}(X_t)],$$

where $S_n = \lbrace k \cdot 2^{-n};\, 0 \le k \le 4^n\rbrace$, $n = 1, 2, \ldots$ Then $g_n \uparrow \widehat{g}$ and $\widehat{g}$ is the least superharmonic majorant of $g$. Moreover, $\widehat{g} = \overline{g}$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 10.1.7</summary>

Note that $\lbrace g_n \rbrace$ is increasing. Define $\check{g}(x) = \lim_{n \to \infty} g_n(x)$. Then $\check{g}(x) \ge g_n(x) \ge E^x[g_{n-1}(X_t)]$ for all $n$ and all $t \in S_n$. Hence

$$\check{g}(x) \ge \lim_{n \to \infty} E^x[g_{n-1}(X_t)] = E^x[\check{g}(X_t)]$$

for all $t \in S = \bigcup_{n=1}^{\infty} S_n$. Since $\check{g}$ is an increasing limit of lower semicontinuous functions (Lemma 8.1.4), $\check{g}$ is lower semicontinuous. Fix $t \in \mathbb{R}$ and choose $t_k \in S$ such that $t_k \to t$. Then by the Fatou lemma and lower semicontinuity

$$\check{g}(x) \ge \varliminf_{k \to \infty} E^x[\check{g}(X_{t_k})] \ge E^x[\varliminf_{k \to \infty} \check{g}(X_{t_k})] \ge E^x[\check{g}(X_t)].$$

So $\check{g}$ is an excessive function. Therefore $\check{g}$ is superharmonic by Theorem 10.1.6 and hence $\check{g}$ is a superharmonic majorant of $g$. On the other hand, if $f$ is any supermeanvalued majorant of $g$, then clearly by induction $f(x) \ge g_n(x)$ for all $n$ and so $f(x) \ge \check{g}(x)$. This proves that $\check{g}$ is the least supermeanvalued majorant $\overline{g}$ of $g$. So $\check{g} = \widehat{g}$. $\square$

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(10.1.8)</span></p>

Define $h_0 = g$ and inductively

$$h_n(x) = \sup_{t \ge 0} E^x[h_{n-1}(X_t)]; \quad n = 1, 2, \ldots$$

Then $h_n \uparrow \widehat{g}$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Corollary 10.1.8</summary>

Let $h = \lim h_n$. Then clearly $h \ge \check{g} = \widehat{g}$. On the other hand, since $\widehat{g}$ is excessive we have $\widehat{g}(x) \ge \sup_{t \ge 0} E^x[\widehat{g}(X_t)]$. So by induction $\widehat{g} \ge h_n$ for all $n$. Thus $\widehat{g} = h$ and the proof is complete. $\square$

</details>

#### The Main Optimal Stopping Theorems

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(10.1.9 — Existence Theorem for Optimal Stopping)</span></p>

Let $g^*$ denote the optimal reward and $\widehat{g}$ the least superharmonic majorant of a continuous reward function $g \ge 0$.

- **a)** Then $g^*(x) = \widehat{g}(x)$.
- **b)** For $\epsilon > 0$ let $D_\epsilon = \lbrace x;\, g(x) < \widehat{g}(x) - \epsilon \rbrace$. Suppose $g$ is bounded. Then stopping at the first time $\tau_\epsilon$ of exit from $D_\epsilon$ is close to being optimal, in the sense that $\lvert g^*(x) - E^x[g(X_{\tau_\epsilon})]\rvert \le 2\epsilon$ for all $x$.
- **c)** For arbitrary continuous $g \ge 0$ let $D = \lbrace x;\, g(x) < g^*(x) \rbrace$ (the **continuation region**). For $N = 1, 2, \ldots$ define $g_N = g \wedge N$, $D_N = \lbrace x;\, g_N(x) < \widehat{g_N}(x)\rbrace$ and $\sigma_N = \tau_{D_N}$. Then $D_N \subset D_{N+1}$, $D_N \subset D \cap g^{-1}([0,N))$, $D = \bigcup_N D_N$. If $\sigma_N < \infty$ a.s. $Q^x$ for all $N$ then $g^*(x) = \lim_{N \to \infty} E^x[g(X_{\sigma_N})]$.
- **d)** In particular, if $\tau_D < \infty$ a.s. $Q^x$ and the family $\lbrace g(X_{\sigma_N})\rbrace_N$ is uniformly integrable w.r.t. $Q^x$, then $g^*(x) = E^x[g(X_{\tau_D})]$ and $\tau^* = \tau_D$ is an optimal stopping time.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 10.1.9</summary>

First assume that $g$ is bounded and define $\widetilde{g}_\epsilon(x) = E^x[\widehat{g}(X_{\tau_\epsilon})]$ for $\epsilon > 0$. Then $\widetilde{g}_\epsilon$ is supermeanvalued by Lemma 10.1.3 f). We claim that $g(x) \le \widetilde{g}_\epsilon(x) + \epsilon$ for all $x$.

To see this suppose $\beta := \sup_x\lbrace g(x) - \widetilde{g}_\epsilon(x)\rbrace > \epsilon$. Then for all $\eta > 0$ we can find $x_0$ such that $g(x_0) - \widetilde{g}_\epsilon(x_0) \ge \beta - \eta$. On the other hand, since $\widetilde{g}_\epsilon + \beta$ is a supermeanvalued majorant of $g$, we have $\widehat{g}(x_0) \le \widetilde{g}_\epsilon(x_0) + \beta$. Combining these we get $\widehat{g}(x_0) \le g(x_0) + \eta$.

Consider the two possible cases:

**Case 1:** $\tau_\epsilon > 0$ a.s. $Q^{x_0}$. Then by the definition of $D_\epsilon$, $g(x_0) + \eta \ge \widehat{g}(x_0) \ge E^{x_0}[\widehat{g}(X_{t \wedge \tau_\epsilon})] \ge E^{x_0}[(g(X_t) + \epsilon)\chi_{\lbrace t < \tau_\epsilon\rbrace}]$ for all $t > 0$. Hence by the Fatou lemma and lower semicontinuity of $g$, $g(x_0) + \eta \ge g(x_0) + \epsilon$. This is a contradiction if $\eta < \epsilon$.

**Case 2:** $\tau_\epsilon = 0$ a.s. $Q^{x_0}$. Then $\widetilde{g}_\epsilon(x_0) = \widehat{g}(x_0)$, so $g(x_0) \le \widetilde{g}_\epsilon(x_0)$, contradicting $g(x_0) - \widetilde{g}_\epsilon(x_0) \ge \beta - \eta > 0$ for $\eta < \beta$.

Therefore $\widetilde{g}_\epsilon + \epsilon$ is a supermeanvalued majorant of $g$. Therefore $\widehat{g} \le \widetilde{g}_\epsilon + \epsilon = E[\widehat{g}(X_{\tau_\epsilon})] + \epsilon \le E[(g + \epsilon)(X_{\tau_\epsilon})] + \epsilon \le g^* + 2\epsilon$ and since $\epsilon$ was arbitrary we have by (10.1.12) $\widehat{g} = g^*$.

If $g$ is not bounded, let $g_N = \min(N,g)$, $N = 1,2,\ldots$ and let $\widehat{g_N}$ be the least superharmonic majorant of $g_N$. Then $g^* \ge g^*_N = \widehat{g_N} \uparrow h$ as $N \to \infty$, where $h \ge \widehat{g}$. On the other hand $g_N \le \widehat{g_N} \le h$ for all $N$ and therefore $g \le h$. Since $\widehat{g}$ is the least superharmonic majorant of $g$ we conclude that $h = \widehat{g} = g^*$.

Finally, to obtain c) and d) note that $\tau_\epsilon \uparrow \tau_D$ as $\epsilon \downarrow 0$ and $\tau_D < \infty$ a.s. so $E^x[g(X_{\tau_\epsilon})] \to E^x[g(X_{\tau_D})]$ as $\epsilon \downarrow 0$. Using the above bounds and uniform integrability one concludes $g^*(x) = E^x[g(X_{\tau_D})]$ if $g$ is bounded. The unbounded case follows by the same truncation argument. $\square$

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(10.1.10)</span></p>

Suppose there exists a Borel set $H$ such that $\widetilde{g}_H(x) := E^x[g(X_{\tau_H})]$ is a supermeanvalued majorant of $g$. Then

$$g^*(x) = \widetilde{g}_H(x), \quad \text{so } \tau^* = \tau_H \text{ is optimal}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(10.1.11)</span></p>

Let $D = \lbrace x;\, g(x) < \widehat{g}(x)\rbrace$ and put $\widetilde{g}(x) = \widetilde{g}_D(x) = E^x[g(X_{\tau_D})]$. If $\widetilde{g} \ge g$ then $\widetilde{g} = g^*$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Corollary 10.1.11</summary>

Since $X_{\tau_D} \notin D$ we have $g(X_{\tau_D}) \ge \widehat{g}(X_{\tau_D})$ and therefore $g(X_{\tau_D}) = \widehat{g}(X_{\tau_D})$, a.s. $Q^x$. So $\widetilde{g}(x) = E^x[\widehat{g}(X_{\tau_D})]$ is supermeanvalued since $\widehat{g}$ is, and the result follows from Corollary 10.1.10. $\square$

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(10.1.12 — Uniqueness Theorem for Optimal Stopping)</span></p>

Define as before $D = \lbrace x;\, g(x) < g^*(x)\rbrace \subset \mathbb{R}^n$. Suppose there exists an optimal stopping time $\tau^* = \tau^*(x,\omega)$ for the problem (10.1.2) for all $x$. Then

$$\tau^* \ge \tau_D \quad \text{for all } x \in D$$

and

$$g^*(x) = E^x[g(X_{\tau_D})] \quad \text{for all } x \in \mathbb{R}^n.$$

Hence $\tau_D$ is an optimal stopping time for the problem (10.1.2).

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 10.1.12</summary>

Choose $x \in D$. Let $\tau$ be an $\mathcal{F}_t$-stopping time and assume $Q^x[\tau < \tau_D] > 0$. Since $g(X_\tau) < g^*(X_\tau)$ if $\tau < \tau_D$ and $g \le g^*$ always, we have

$$E^x[g(X_\tau)] = \int_{\tau < \tau_D} g(X_\tau)\,dQ^x + \int_{\tau \ge \tau_D} g(X_\tau)\,dQ^x < \int_{\tau < \tau_D} g^*(X_\tau)\,dQ^x + \int_{\tau \ge \tau_D} g^*(X_\tau)\,dQ^x = E^x[g^*(X_\tau)] \le g^*(x),$$

since $g^*$ is superharmonic. This proves (10.1.32).

To obtain (10.1.33), we first choose $x \in D$. Since $\widehat{g}$ is superharmonic we have by (10.1.32) and Lemma 10.1.3 e): $g^*(x) = E^x[g(X_{\tau^*})] \le E^x[\widehat{g}(X_{\tau^*})] \le E^x[\widehat{g}(X_{\tau_D})] = E^x[g(X_{\tau_D})] \le g^*(x)$, which proves (10.1.33) for $x \in D$.

Next, choose $x \in \partial D$ to be an *irregular* boundary point of $D$. Then $\tau_D > 0$ a.s. $Q^x$. Let $\lbrace \alpha_k \rbrace$ be a sequence of stopping times such that $0 < \alpha_k < \tau_D$ and $\alpha_k \to 0$ a.s. $Q^x$, as $k \to \infty$. Then $X_{\alpha_k} \in D$ so by (10.1.32), (7.2.6) and the strong Markov property (7.2.2): $E^x[g(X_{\tau_D})] = E^x[\theta_{\alpha_k} g(X_{\tau_D})] = E^x[E^{X_{\alpha_k}}[g(X_{\tau_D})]] = E^x[g^*(X_{\alpha_k})]$ for all $k$.

Hence by lower semicontinuity and the Fatou lemma $g^*(x) \le E^x[\varliminf_{k \to \infty} g^*(X_{\alpha_k})] \le \varliminf_{k \to \infty} E^x[g^*(X_{\alpha_k})] = E^x[g(X_{\tau_D})]$.

Finally, if $x \in \partial D$ is a *regular* boundary point of $D$ or if $x \notin \overline{D}$ we have $\tau_D = 0$ a.s. $Q^x$ and hence $g^*(x) = E^x[g(X_{\tau_D})]$. $\square$

</details>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Set $U$ and the Continuation Region)</span></p>

Let $\mathcal{A}$ be the characteristic operator of $X$. Assume $g \in C^2(\mathbb{R}^n)$. Define

$$U = \lbrace x;\, \mathcal{A}g(x) > 0\rbrace.$$

Then, with $D$ as before, $U \subset D$. Consequently, from (10.1.32) we conclude that it is *never optimal to stop the process before it exits from $U$*. But there may be cases when $U \ne D$, so that it is optimal to proceed beyond $U$ before stopping. (This is in fact the typical situation.)

To prove $U \subset D$: choose $x \in U$ and let $\tau_0$ be the first exit time from a bounded open set $W \ni x$, $W \subset U$. Then by Dynkin's formula, for $u > 0$: $E^x[g(X_{\tau_0 \wedge u})] = g(x) + E^x\!\left[\int_0^{\tau_0 \wedge u} \mathcal{A}g(X_s)\,ds\right] > g(x)$, so $g(x) < g^*(x)$ and therefore $x \in D$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(10.1.13 — Brownian Motion in $\mathbb{R}^2$)</span></p>

Let $X_t = B_t$ be a Brownian motion in $\mathbb{R}^2$. Using that $B_t$ is recurrent in $\mathbb{R}^2$ (Example 7.4.2) one can show that the only (nonnegative) superharmonic functions in $\mathbb{R}^2$ are the constants. Therefore

$$g^*(x) = \lVert g \rVert_\infty := \sup\lbrace g(y);\, y \in \mathbb{R}^2\rbrace \quad \text{for all } x.$$

So if $g$ is unbounded then $g^* = \infty$ and no optimal stopping time exists. Assume therefore that $g$ is bounded. The continuation region is $D = \lbrace x;\, g(x) < \lVert g \rVert_\infty \rbrace$, so if $\partial D$ is a *polar set* i.e. $\operatorname{cap}(\partial D) = 0$ (where cap denotes the *logarithmic capacity*), then $\tau_D = \infty$ a.s. and no optimal stopping exists. On the other hand, if $\operatorname{cap}(\partial D) > 0$ then $\tau_D < \infty$ a.s. and $E^x[g(B_{\tau_D})] = \lVert g \rVert_\infty = g^*(x)$, so $\tau^* = \tau_D$ is optimal.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(10.1.14 — Brownian Motion in $\mathbb{R}^n$ for $n \ge 3$)</span></p>

The situation is different in $\mathbb{R}^n$ for $n \ge 3$.

**a)** Let $X_t = B_t$ be Brownian motion in $\mathbb{R}^3$ and let the reward function be

$$g(\xi) = \begin{cases} \lvert \xi \rvert^{-1} & \text{for } \lvert \xi \rvert \ge 1 \\\\ 1 & \text{for } \lvert \xi \rvert < 1 \end{cases}; \quad \xi \in \mathbb{R}^3.$$

Then $g$ is superharmonic (in the classical sense) in $\mathbb{R}^3$, so $g^* = g$ everywhere and the best policy is to stop immediately, no matter where the starting point is.

**b)** Let us change $g$ to $h(x) = \lvert x \rvert^{-\alpha}$ for $\lvert x \rvert \ge 1$ and $h(x) = 1$ for $\lvert x \rvert < 1$, for some $\alpha > 1$. Let $H = \lbrace x;\, \lvert x \rvert > 1 \rbrace$ and define $\widetilde{h}(x) = E^x[h(B_{\tau_H})] = P^x[\tau_H < \infty]$. Then by Example 7.4.2

$$\widetilde{h}(x) = \begin{cases} 1 & \text{if } \lvert x \rvert \le 1 \\\\ \lvert x \rvert^{-1} & \text{if } \lvert x \rvert > 1 \end{cases},$$

i.e. $\widetilde{h} = g$ (defined in a)), which is a superharmonic majorant of $h$. Therefore by Corollary 10.1.10 $h^* = \widetilde{h} = g$, $H = D$ and $\tau^* = \tau_H$ is an optimal stopping time.

</div>

#### Reward Functions Assuming Negative Values

The results obtained so far are based on the assumptions that $g(\xi) \ge 0$ and $g$ is continuous. To some extent these assumptions can be relaxed. For example, Theorem 10.1.9 a) still holds if $g \ge 0$ is only assumed to be lower semicontinuous.

The nonnegativity assumption on $g$ can also be relaxed. If $g$ is bounded below, say $g \ge -M$ where $M > 0$ is a constant, then we can put $g_1 = g + M \ge 0$ and apply the theory to $g_1$. Since $E^x[g(X_\tau)] = E^x[g_1(X_\tau)] - M$ if $\tau < \infty$ a.s., we have $g^*(x) = g_1^*(x) - M$, so the problem can be reduced to the optimal stopping problem for the nonnegative function $g_1$.

If $g$ is not bounded below, then the problem (10.1.2)–(10.1.3) is not well-defined unless $E^x[g^-(X_\tau)] < \infty$ for all $\tau$, where $g^-(x) = -\min(g(x), 0)$. If $g$ satisfies the stronger condition that the family $\lbrace g^-(X_\tau);\, \tau \text{ stopping time}\rbrace$ is uniformly integrable, then basically all the theory from the nonnegative case carries over.

### 10.2 The Time-Inhomogeneous Case

Let us now consider the case when the reward function $g$ depends on both time and space, i.e.

$$g = g(t,x)\colon \mathbb{R} \times \mathbb{R}^n \to [0,\infty), \quad g \text{ is continuous}.$$

Then the problem is to find $g_0(x)$ and $\tau^*$ such that

$$g_0(x) = \sup_\tau E^x[g(\tau, X_\tau)] = E^x[g(\tau^*, X_{\tau^*})].$$

To reduce this case to the original case (10.1.2)–(10.1.3) we proceed as follows. Suppose the Itô diffusion $X_t = X_t^x$ has the form $dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t$; $t \ge 0$, $X_0 = x$, where $b\colon\mathbb{R}^n \to \mathbb{R}^n$ and $\sigma\colon\mathbb{R}^n \to \mathbb{R}^{n \times m}$. Define the Itô diffusion $Y_t = Y_t^{(s,x)}$ in $\mathbb{R}^{n+1}$ by

$$Y_t = \begin{bmatrix} s + t \\\\ X_t^x \end{bmatrix}; \quad t \ge 0.$$

Then $dY_t = \widehat{b}(Y_t)\,dt + \widehat{\sigma}(Y_t)\,dB_t$ where $\widehat{b}(\eta) = \widehat{b}(t,\xi) = \begin{bmatrix} 1 \\\\ b(\xi) \end{bmatrix} \in \mathbb{R}^{n+1}$ and $\widehat{\sigma}(\eta) = \widehat{\sigma}(t,\xi) = \begin{bmatrix} 0 \cdots 0 \\\\ \sigma(\xi) \end{bmatrix} \in \mathbb{R}^{(n+1) \times m}$, with $\eta = (t,\xi) \in \mathbb{R} \times \mathbb{R}^n$.

So $Y_t$ is an Itô diffusion starting at $y = (s,x)$. Let $R^y = R^{(s,x)}$ denote the probability law of $\lbrace Y_t \rbrace$ and let $E^y = E^{(s,x)}$ denote the expectation w.r.t. $R^y$. In terms of $Y_t$ the problem (10.2.2) can be written

$$g_0(x) = g^*(0,x) = \sup_\tau E^{(0,x)}[g(Y_\tau)] = E^{(0,x)}[g(Y_{\tau^*})]$$

which is a special case of the problem $g^*(s,x) = \sup_\tau E^{(s,x)}[g(Y_\tau)] = E^{(s,x)}[g(Y_{\tau^*})]$, which is of the form (10.1.2)–(10.1.3) with $X_t$ replaced by $Y_t$.

Note that the characteristic operator $\widehat{\mathcal{A}}$ of $Y_t$ is given by

$$\widehat{\mathcal{A}}\phi(s,x) = \frac{\partial \phi}{\partial s}(s,x) + \mathcal{A}\phi(s,x); \quad \phi \in C^2(\mathbb{R} \times \mathbb{R}^n)$$

where $\mathcal{A}$ is the characteristic operator of $X_t$ (working on the $x$-variables).

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(10.2.1 — Exponential Reward with Brownian Motion)</span></p>

Let $X_t = B_t$ be 1-dimensional Brownian motion and let the reward function be $g(t,\xi) = e^{-\alpha t + \beta \xi}$; $\xi \in \mathbb{R}$, where $\alpha, \beta \ge 0$ are constants. The characteristic operator $\widehat{\mathcal{A}}$ of $Y_t^{s,x} = \begin{bmatrix} s + t \\\\ B_t \end{bmatrix}$ is given by $\widehat{\mathcal{A}}f(s,x) = \frac{\partial f}{\partial s} + \frac{1}{2} \cdot \frac{\partial^2 f}{\partial x^2}$; $f \in C^2$.

Thus $\mathcal{A}g = (-\alpha + \frac{1}{2}\beta^2)g$, so if $\beta^2 \le 2\alpha$ then $g^* = g$ and the best policy is to stop immediately. If $\beta^2 > 2\alpha$ we have $U := \lbrace (s,x);\, \widehat{\mathcal{A}}g(s,x) > 0\rbrace = \mathbb{R}^2$ and therefore by (10.1.35) $D = \mathbb{R}^2$ and hence $\tau^*$ does not exist. If $\beta^2 > 2\alpha$ we can use Theorem 10.1.7 to prove that $g^* = \infty$:

$$\sup_{t \in S_n} E^{(s,x)}[g(Y_t)] = \sup_{t \in S_n} E[e^{-\alpha(s+t) + \beta B_t^x}] = \sup_{t \in S_n}\left[e^{-\alpha(s+t)} \cdot e^{\beta x + \frac{1}{2}\beta^2 t}\right] = g(s,x) \cdot \exp((-\alpha + \tfrac{1}{2}\beta^2)2^n),$$

so $g_n(s,x) \to \infty$ as $n \to \infty$. Hence no optimal stopping exists in this case.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(10.2.2 — When Is the Right Time to Sell the Stocks?)</span></p>

We now return to a specified version of Problem 5 in the introduction. Suppose the price $X_t$ at time $t$ of a person's assets (e.g. a house, stocks, oil ...) varies according to a stochastic differential equation of the form

$$dX_t = rX_t\,dt + \alpha X_t\,dB_t, \quad X_0 = x > 0,$$

where $B_t$ is 1-dimensional Brownian motion and $r, \alpha$ are known constants. Suppose that connected to the sale of the assets there is a fixed fee/tax or transaction cost $a > 0$. Then if the person decides to sell at time $t$ the discounted net of the sale is $e^{-\rho t}(X_t - a)$, where $\rho > 0$ is a given discounting factor. The problem is to find a stopping time $\tau$ that maximizes

$$E^{(s,x)}[e^{-\rho \tau}(X_\tau - a)] = E^{(s,x)}[g(\tau, X_\tau)],$$

where $g(t,\xi) = e^{-\rho t}(\xi - a)$.

The characteristic operator $\widehat{\mathcal{A}}$ of the process $Y_t = (s+t, X_t)$ is given by $\widehat{\mathcal{A}}f(s,x) = \frac{\partial f}{\partial s} + rx\frac{\partial f}{\partial x} + \frac{1}{2}\alpha^2 x^2 \frac{\partial^2 f}{\partial x^2}$; $f \in C^2(\mathbb{R}^2)$.

Hence $\widehat{\mathcal{A}}g(s,x) = -\rho e^{-\rho s}(x-a) + rxe^{-\rho s} = e^{-\rho s}((r-\rho)x + \rho a)$. So

$$U := \lbrace (s,x);\, \widehat{\mathcal{A}}g(s,x) > 0\rbrace = \begin{cases} \mathbb{R} \times \mathbb{R}_+ & \text{if } r \ge \rho \\\\ \lbrace (s,x);\, x < \frac{a\rho}{\rho - r}\rbrace & \text{if } r < \rho. \end{cases}$$

So if $r \ge \rho$ we have $U = D = \mathbb{R} \times \mathbb{R}_+$ so $\tau^*$ does not exist. If $r > \rho$ then $g^* = \infty$ while if $r = \rho$ then $g^*(s,x) = xe^{-\rho s}$.

It remains to examine the case $r < \rho$. (If we regard $\rho$ as the sum of interest rate, inflation and tax etc., this is not an unreasonable assumption in applications.) First we establish that the region $D$ must be invariant w.r.t. $t$, in the sense that $D + (t_0, 0) = D$ for all $t_0$.

Therefore the connected component of $D$ that contains $U$ must have the form $D(x_0) = \lbrace (t,x);\, 0 < x < x_0 \rbrace$ for some $x_0 \ge \frac{a\rho}{\rho - r}$.

Note that $D$ cannot have any other components, for if $V$ is a component of $D$ disjoint from $U$ then $\widehat{\mathcal{A}}g < 0$ in $V$ and so, by Theorem 10.1.9 c), $g^*(y) = g(y)$, which implies $V = \emptyset$.

Put $\tau(x_0) = \tau_{D(x_0)}$ and let us compute $\widetilde{g}(s,x) = \widetilde{g}_{x_0}(s,x) = E^{(s,x)}[g(Y_{\tau(x_0)})]$. From Chapter 9 we know that $f = \widetilde{g}$ is the (bounded) solution of the boundary value problem

$$\frac{\partial f}{\partial s} + rx\frac{\partial f}{\partial x} + \tfrac{1}{2}\alpha^2 x^2 \frac{\partial^2 f}{\partial x^2} = 0 \quad \text{for } 0 < x < x_0$$

$$f(s,x_0) = e^{-\rho s}(x_0 - a).$$

If we try a solution of the form $f(s,x) = e^{-\rho s}\phi(x)$ we get the following 1-dimensional problem

$$-\rho\phi + rx\phi'(x) + \tfrac{1}{2}\alpha^2 x^2 \phi''(x) = 0 \quad \text{for } 0 < x < x_0$$

$$\phi(x_0) = x_0 - a.$$

The general solution is $\phi(x) = C_1 x^{\gamma_1} + C_2 x^{\gamma_2}$, where $C_1, C_2$ are arbitrary constants and

$$\gamma_i = \alpha^{-2}\!\left[\tfrac{1}{2}\alpha^2 - r \pm \sqrt{(r - \tfrac{1}{2}\alpha^2)^2 + 2\rho\alpha^2}\right] \quad (i = 1,2),\ \gamma_2 < 0 < \gamma_1.$$

Since $\phi(x)$ is bounded as $x \to 0$ we must have $C_2 = 0$ and the boundary requirement $\phi(x_0) = x_0 - a$ gives $C_1 = x_0^{-\gamma_1}(x_0 - a)$. We conclude that the bounded solution $f$ of the boundary value problem is

$$\widetilde{g}_{x_0}(s,x) = f(s,x) = e^{-\rho s}(x_0 - a)\!\left(\frac{x}{x_0}\right)^{\gamma_1}.$$

If we fix $(s,x)$ then the value of $x_0$ which maximizes $\widetilde{g}_{x_0}(s,x)$ is easily seen to be given by

$$x_0 = x_{\max} = \frac{a\gamma_1}{\gamma_1 - 1}$$

(note that $\gamma_1 > 1$ if and only if $r < \rho$).

Thus we have arrived at the candidate $\widetilde{g}_{x_{\max}}(s,x)$ for $g^*(s,x) = \sup_\tau E^{(s,x)}[e^{-\rho\tau}(X_\tau - a)]$. To verify that we indeed have $\widetilde{g}_{x_{\max}} = g^*$ it would suffice to prove that $\widetilde{g}_{x_{\max}}$ is a supermeanvalued majorant of $g$ (see Corollary 10.1.10). This problem can be solved more easily by Theorem 10.4.1 (see Example 10.4.2).

The conclusion is therefore that one should sell the assets the first time the price of them reaches the value $x_{\max} = \frac{a\gamma_1}{\gamma_1 - 1}$. The expected discounted profit obtained from this strategy is

$$g^*(s,x) = \widetilde{g}_{x_{\max}}(s,x) = e^{-\rho s}\!\left(\frac{\gamma_1 - 1}{a}\right)^{\gamma_1 - 1}\!\left(\frac{x}{\gamma_1}\right)^{\gamma_1}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The High Contact Principle)</span></p>

The value $x_0 = x_{\max}$ is the only value of $x_0$ which makes the function $x \to \widetilde{g}_{x_0}(s,x)$ continuously differentiable at $x_0$. This is not a coincidence. In fact, it illustrates a general phenomenon which is known as the **high contact** (or smooth fit) **principle**. This principle is the basis of the fundamental connection between optimal stopping and *variational inequalities*.

</div>

### 10.3 Optimal Stopping Problems Involving an Integral

Let

$$dY_t = b(Y_t)\,dt + \sigma(Y_t)\,dB_t, \quad Y_0 = y$$

be an Itô diffusion in $\mathbb{R}^k$. Let $g\colon\mathbb{R}^k \to [0,\infty)$ be continuous and let $f\colon\mathbb{R}^k \to [0,\infty)$ be Lipschitz continuous with at most linear growth. Consider the optimal stopping problem: Find $G^*(y)$ and $\tau^*$ such that

$$G^*(y) = \sup_\tau E^y\!\left[\int_0^\tau f(Y_t)\,dt + g(Y_\tau)\right] = E^y\!\left[\int_0^{\tau^*} f(Y_t)\,dt + g(Y_{\tau^*})\right].$$

This problem can be reduced to our original problem (10.1.2)–(10.1.3) by proceeding as follows: Define the Itô diffusion $Z_t$ in $\mathbb{R}^k \times \mathbb{R} = \mathbb{R}^{k+1}$ by

$$dZ_t = \begin{bmatrix} dY_t \\\\ dW_t \end{bmatrix} := \begin{bmatrix} b(Y_t) \\\\ f(Y_t) \end{bmatrix}dt + \begin{bmatrix} \sigma(Y_t) \\\\ 0 \end{bmatrix}dB_t; \quad Z_0 = z = (y, 0).$$

Then we see that

$$G^*(y) = \sup_\tau E^{(y,0)}[W_\tau + g(Y_\tau)] = \sup_\tau E^{(y,0)}[G(Z_\tau)]$$

with $G(z) := G(y,w) := g(y) + w$; $z = (y,w) \in \mathbb{R}^k \times \mathbb{R}$.

This is again a problem of the type (10.1.2)–(10.1.3) with $X_t$ replaced by $Z_t$ and $g$ replaced by $G$. Note that the connection between the characteristic operators $\mathcal{A}_Y$ of $Y_t$ and $\mathcal{A}_Z$ of $Z_t$ is given by

$$\mathcal{A}_Z\phi(z) = \mathcal{A}_Y\phi(y,w) + f(y)\frac{\partial \phi}{\partial w}, \quad \phi \in C^2(\mathbb{R}^{k+1}).$$

In particular, if $G(y,w) = g(y) + w \in C^2(\mathbb{R}^{k+1})$ then

$$\mathcal{A}_Z G(y,w) = \mathcal{A}_Y g(y) + f(y).$$

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(10.3.1 — Optimal Stopping with Running Reward and Geometric Brownian Motion)</span></p>

Consider the optimal stopping problem

$$\gamma(x) = \sup_\tau E^x\!\left[\int_0^\tau \theta e^{-\rho t} X_t\,dt + e^{-\rho\tau}X_\tau\right],$$

where $dX_t = \alpha X_t\,dt + \beta X_t\,dB_t$; $X_0 = x > 0$ is geometric Brownian motion ($\alpha, \beta, \theta$ constants, $\theta > 0$). We put

$$dY_t = \begin{bmatrix} dt \\\\ dX_t \end{bmatrix} = \begin{bmatrix} 1 \\\\ \alpha X_t \end{bmatrix}dt + \begin{bmatrix} 0 \\\\ \beta X_t \end{bmatrix}dB_t; \quad Y_0 = (s,x)$$

and $dZ_t = \begin{bmatrix} dY_t \\\\ dW_t \end{bmatrix} = \begin{bmatrix} 1 \\\\ \alpha X_t \\\\ e^{-\rho t} X_t \end{bmatrix}dt + \begin{bmatrix} 0 \\\\ \beta X_t \\\\ 0 \end{bmatrix}dB_t$; $Z_0 = (s,x,w)$. Then with $f(y) = f(s,x) = \theta e^{-\rho s}x$, $g(y) = e^{-\rho s}x$ and $G(s,x,w) = e^{-\rho s}x + w$ we have

$$\mathcal{A}_Z G = \frac{\partial G}{\partial s} + \alpha x\frac{\partial G}{\partial x} + \tfrac{1}{2}\beta^2 x^2 \frac{\partial^2 G}{\partial x^2} + \theta e^{-\rho s}x\frac{\partial G}{\partial w} = (-\rho + \alpha + \theta)e^{-\rho s}x.$$

Hence

$$U = \lbrace (s,x,w);\, \mathcal{A}_Z G(s,x,w) > 0\rbrace = \begin{cases} \mathbb{R}^3 & \text{if } \rho < \alpha + \theta \\\\ \emptyset & \text{if } \rho \ge \alpha + \theta. \end{cases}$$

From this we conclude:

- If $\rho \ge \alpha + \theta$ then $\tau^* = 0$ and $G^*(s,x,w) = e^{-\rho s}x + w$.
- If $\alpha < \rho < \alpha + \theta$ then $\tau^*$ does not exist and $G^*(s,x,w) = \frac{\theta x}{\rho - \alpha}e^{-\rho s} + w$.
- If $\rho \le \alpha$ then $\tau^*$ does not exist and $G^* = \infty$.

</div>

### 10.4 Connection with Variational Inequalities

The "high contact principle" says, roughly, that — under certain conditions — the solution $g^*$ of the optimal stopping problem (10.1.2)–(10.1.3) is a $C^1$ function on $\mathbb{R}^n$ if $g \in C^2(\mathbb{R}^n)$. This is useful information which can help us to determine $g^*$. Indeed, this principle is so useful that it is frequently applied in the literature also in cases where its validity has not been rigorously proved.

Fortunately it turns out to be easy to prove a *sufficiency* condition of high contact type, i.e. a kind of verification theorem for optimal stopping, which makes it easy to verify that a given candidate for $g^*$ (that we may have found by guessing or intuition) is actually equal to $g^*$.

In the following we fix a domain $V$ in $\mathbb{R}^k$ and we let

$$dY_t = b(Y_t)\,dt + \sigma(Y_t)\,dB_t, \quad Y_0 = y$$

be an Itô diffusion in $\mathbb{R}^k$. Define $T = T(y,\omega) = \inf\lbrace t > 0;\, Y_t(\omega) \notin V\rbrace$. Let $f\colon\mathbb{R}^k \to \mathbb{R}$, $g\colon\mathbb{R}^k \to \mathbb{R}$ be continuous functions satisfying

- (a) $E^y\!\left[\int_0^T \lvert f(Y_t)\rvert\,dt\right] < \infty$ for all $y \in \mathbb{R}^k$
- (b) the family $\lbrace g^-(Y_\tau);\, \tau \text{ stopping time},\, \tau \le T \rbrace$ is uniformly integrable w.r.t. $R^y$ (the probability law of $Y_t$), for all $y \in \mathbb{R}^k$.

Consider the following problem: Find $\Phi(y)$ and $\tau^* \le T$ such that

$$\Phi(y) = \sup_{\tau \le T} J^\tau(y) = J^{\tau^*}(y),$$

where $J^\tau(y) = E^y\!\left[\int_0^\tau f(Y_t)\,dt + g(Y_\tau)\right]$ for $\tau \le T$. Note that since $J^0(y) = g(y)$ we have $\Phi(y) \ge g(y)$ for all $y \in V$.

We can now formulate the variational inequalities. As usual we let

$$L = L_Y = \sum_{i=1}^k b_i(y)\frac{\partial}{\partial y_i} + \frac{1}{2}\sum_{i,j=1}^k (\sigma\sigma^T)_{ij}(y)\frac{\partial^2}{\partial y_i \partial y_j}$$

be the partial differential operator which coincides with the generator $A_Y$ of $Y_t$ on $C_0^2(\mathbb{R}^k)$.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(10.4.1 — Variational Inequalities for Optimal Stopping)</span></p>

Suppose we can find a function $\phi\colon\overline{V} \to \mathbb{R}$ such that

- **(i)** $\phi \in C^1(V) \cap C(\overline{V})$
- **(ii)** $\phi \ge g$ on $V$ and $\phi = g$ on $\partial V$.

Define $D = \lbrace x \in V;\, \phi(x) > g(x)\rbrace$. Suppose $Y_t$ spends $0$ time on $\partial D$ a.s., i.e.

- **(iii)** $E^y\!\left[\int_0^T \chi_{\partial D}(Y_t)\,dt\right] = 0$ for all $y \in V$

and suppose that

- **(iv)** $\partial D$ is a Lipschitz surface, i.e. $\partial D$ is locally the graph of a function $h\colon\mathbb{R}^{k-1} \to \mathbb{R}$ such that there exists $K < \infty$ with $\lvert h(x) - h(y)\rvert \le K\lvert x - y\rvert$ for all $x,y$.

Moreover, suppose the following:

- **(v)** $\phi \in C^2(V \setminus \partial D)$ and the second order derivatives of $\phi$ are locally bounded near $\partial D$
- **(vi)** $L\phi + f \le 0$ on $V \setminus \overline{D}$
- **(vii)** $L\phi + f = 0$ on $D$
- **(viii)** $\tau_D := \inf\lbrace t > 0;\, Y_t \notin D \rbrace < \infty$ a.s. $R^y$ for all $y \in V$
- **(ix)** the family $\lbrace \phi(Y_\tau);\, \tau \le \tau_D \rbrace$ is uniformly integrable w.r.t. $R^y$, for all $y \in V$.

Then

$$\phi(y) = \Phi(y) = \sup_{\tau \le T} E^y\!\left[\int_0^\tau f(Y_t)\,dt + g(Y_\tau)\right]; \quad y \in V$$

and $\tau^* = \tau_D$ is an optimal stopping time for this problem.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 10.4.1</summary>

By (i), (iv) and (v) we can find a sequence of functions $\phi_j \in C^2(V) \cap C(\overline{V})$, $j = 1,2,\ldots$, such that (a) $\phi_j \to \phi$ uniformly on compact subsets of $\overline{V}$, as $j \to \infty$, (b) $L\phi_j \to L\phi$ uniformly on compact subsets of $V \setminus \partial D$, as $j \to \infty$, (c) $\lbrace L\phi_j\rbrace_{j=1}^\infty$ is locally bounded on $V$.

For $R > 0$ put $T_R = \min(R, \inf\lbrace t > 0;\, \lvert Y_t\rvert \ge R\rbrace)$ and let $\tau \le T$ be a stopping time. Let $y \in V$. Then by Dynkin's formula $E^y[\phi_j(Y_{\tau \wedge T_R})] = \phi_j(y) + E^y\!\left[\int_0^{\tau \wedge T_R} L\phi_j(Y_t)\,dt\right]$.

Hence by (a), (b), (c) and (iii) and the Fatou lemma $\phi(y) = \lim_{j \to \infty} E^y\!\left[\int_0^{\tau \wedge T_R} -L\phi_j(Y_t)\,dt + \phi_j(Y_{\tau \wedge T_R})\right] \ge E^y\!\left[\int_0^{\tau \wedge T_R} -L\phi(Y_t)\,dt + \phi(Y_{\tau \wedge T_R})\right]$.

Therefore, by (ii), (iii), (vi) and (vii), $\phi(y) \ge E^y\!\left[\int_0^{\tau \wedge T_R} f(Y_t)\,dt + g(Y_{\tau \wedge T_R})\right]$.

Hence by the Fatou lemma and (10.4.3), (10.4.4): $\phi(y) \ge \lim_{R \to \infty} E^y\!\left[\int_0^{\tau \wedge T_R} f(Y_t)\,dt + g(Y_{\tau \wedge T_R})\right] \ge E^y\!\left[\int_0^\tau f(Y_t)\,dt + g(Y_\tau)\right]$.

Since $\tau \le T$ was arbitrary, we conclude that $\phi(y) \ge \Phi(y)$ for all $y \in V$. If $y \notin D$ then $\phi(y) = g(y) \le \Phi(y)$ so by the above $\phi(y) = \Phi(y)$ and $\widehat{\tau} = 0$ is optimal for $y \notin D$.

Next, suppose $y \in D$. Let $\lbrace D_k\rbrace_{k=1}^\infty$ be an increasing sequence of open sets $D_k$ such that $\overline{D}_k \subset D$, $D_k$ is compact and $D = \bigcup_{k=1}^\infty D_k$. Put $\tau_k = \inf\lbrace t > 0;\, Y_t \notin D_k\rbrace$. By Dynkin's formula we have for $y \in D_k$:

$\phi(y) = \lim_{j \to \infty} \phi_j(y) = \lim_{j \to \infty} E^y\!\left[\int_0^{\tau_k \wedge T_R} -L\phi_j(Y_t)\,dt + \phi_j(Y_{\tau_k \wedge T_R})\right] = E^y\!\left[\int_0^{\tau_k \wedge T_R} f(Y_t)\,dt + \phi(Y_{\tau_k \wedge T_R})\right]$.

So by uniform integrability and (ii), (vii), (viii) we get $\phi(y) = \lim_{R,K \to \infty} E^y\!\left[\int_0^{\tau_k \wedge T_R} f(Y_t)\,dt + \phi(Y_{\tau_k \wedge T_R})\right] = E^y\!\left[\int_0^{\tau_D} f(Y_t)\,dt + g(Y_{\tau_D})\right] = J^{\tau_D}(y) \le \Phi(y)$.

Combining $\phi(y) \ge \Phi(y)$ and $\phi(y) \le \Phi(y)$ we get $\phi(y) = \Phi(y)$ for all $y \in V$.

Moreover, the stopping time $\widehat{\tau}$ defined by $\widehat{\tau}(y,\omega) = 0$ for $y \notin D$ and $\widehat{\tau}(y,\omega) = \tau_D$ for $y \in D$ is optimal. By Theorem 10.1.12 we conclude that $\tau_D$ is optimal also. $\square$

</details>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(10.4.2 — Revisiting the Stock-Selling Problem)</span></p>

To illustrate Theorem 10.4.1 let us apply it to reconsider Example 10.2.2. Rather than *proving* (10.2.8) and the following properties of $D$, we now simply guess/assume that $D$ has the form

$$D = \lbrace (s,x);\, 0 < x < x_0 \rbrace$$

for some $x_0 > 0$, which is intuitively reasonable. Then we solve (10.2.11) for arbitrary $x_0$ and we arrive at the following candidate $\phi$ for $g^*$:

$$\phi(s,x) = \begin{cases} e^{-\rho s}(x_0 - a)\!\left(\frac{x}{x_0}\right)^{\gamma_1} & \text{for } 0 < x < x_0 \\\\ e^{-\rho s}(x - a) & \text{for } x \ge x_0. \end{cases}$$

The requirement that $\phi \in C^1$ (Theorem 10.4.1 (i)) gives the value (10.2.13) for $x_0$. It is clear that $\phi \in C^2$ outside $\partial D$ and by construction $L\phi = 0$ on $D$. Moreover, conditions (iii), (iv), (viii) and (ix) clearly hold. It remains to verify that

- (ii) $\phi(s,x) > g(s,x)$ for $0 < x < x_0$, i.e. $\phi(s,x) > e^{-\rho s}(x - a)$ for $0 < x < x_0$, and
- (v) $L\phi(s,x) \le 0$ for $x > x_0$, i.e. $Lg(s,x) \le 0$ for $x > x_0$.

This is easily done by direct calculation (assuming $r < \rho$). We conclude that $\phi = g^*$ and $\tau^* = \tau_D$ is optimal (with the value (10.2.13) for $x_0$).

</div>

## 11. Application to Stochastic Control

### 11.1 Statement of the Problem

Suppose that the state of a system at time $t$ is described by an Itô process $X_t$ of the form

$$dX_t = dX_t^u = b(t, X_t, u_t)\,dt + \sigma(t, X_t, u_t)\,dB_t,$$

where $b\colon\mathbb{R} \times \mathbb{R}^n \times U \to \mathbb{R}^n$, $\sigma\colon\mathbb{R} \times \mathbb{R}^n \times U \to \mathbb{R}^{n \times m}$ and $B_t$ is $m$-dimensional Brownian motion. Here $u_t \in U \subset \mathbb{R}^k$ is a parameter whose value we can choose in the given Borel set $U$ at any instant $t$ in order to control the process $X_t$. Thus $u_t = u(t,\omega)$ is a stochastic process. Since our decision at time $t$ must be based upon what has happened up to time $t$, the function $\omega \to u(t,\omega)$ must (at least) be measurable w.r.t. $\mathcal{F}_t^{(m)}$, i.e. the process $u_t$ must be $\mathcal{F}_t^{(m)}$-adapted.

Let $\lbrace X_h^{s,x}\rbrace_{h \ge s}$ be the solution of (11.1.1) such that $X_s^{s,x} = x$, i.e.

$$X_h^{s,x} = x + \int_s^h b(r, X_r^{s,x}, u_r)\,dr + \int_s^h \sigma(r, X_r^{s,x}, u_r)\,dB_r; \quad h \ge s$$

and let the probability law of $X_t$ starting at $x$ for $t = s$ be denoted by $Q^{s,x}$.

Let $F\colon\mathbb{R} \times \mathbb{R}^n \times U \to \mathbb{R}$ (the "utility rate" function) and $K\colon\mathbb{R} \times \mathbb{R}^n \to \mathbb{R}$ (the "bequest" function) be given continuous functions, let $G$ be a fixed domain in $\mathbb{R} \times \mathbb{R}^n$ and let $\widehat{T}$ be the first exit time after $s$ from $G$ for the process $\lbrace X_r^{s,x}\rbrace_{r \ge s}$, i.e.

$$\widehat{T} = \widehat{T}^{s,x}(\omega) = \inf\lbrace r > s;\, (r, X_r^{s,x}(\omega)) \notin G\rbrace \le \infty.$$

Suppose $E^{s,x}\!\left[\int_s^{\widehat{T}} \lvert F^{u_r}(r, X_r)\rvert\,dr + \lvert K(\widehat{T}, X_{\widehat{T}})\rvert \chi_{\lbrace \widehat{T} < \infty\rbrace}\right] < \infty$ for all $s, x, u$, where $F^u(r,z) = F(r,z,u)$. Then we define the **performance function** $J^u(s,x)$ by

$$J^u(s,x) = E^{s,x}\!\left[\int_s^{\widehat{T}} F^{u_r}(r, X_r)\,dr + K(\widehat{T}, X_{\widehat{T}})\chi_{\lbrace \widehat{T} < \infty\rbrace}\right].$$

To obtain an easier notation we introduce $Y_t = (s + t, X_{s+t}^{s,x})$ for $t \ge 0$, $Y_0 = (s,x)$ and observe that $dY_t = dY_t^u = b(Y_t, u_t)\,dt + \sigma(Y_t, u_t)\,dB_t$, with $T := \inf\lbrace t > 0;\, Y_t \notin G\rbrace = \widehat{T} - s$. The performance function may be written in terms of $Y$ as follows, with $y = (s,x)$:

$$J^u(y) = E^y\!\left[\int_0^T F^{u_t}(Y_t)\,dt + K(Y_T)\chi_{\lbrace T < \infty\rbrace}\right].$$

The problem is — for each $y \in G$ — to find the number $\Phi(y)$ and a control $u^* = u^*(t,\omega) = u^*(y,t,\omega)$ such that

$$\Phi(y) := \sup_{u(t,\omega)} J^u(y) = J^{u^*}(y)$$

where the supremum is taken over all $\mathcal{F}_t^{(m)}$-adapted processes $\lbrace u_t\rbrace$ with values in $U$. Such a control $u^*$ — if it exists — is called an **optimal control** and $\Phi$ is called the **optimal performance** or the **value function**.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Types of Controls)</span></p>

Examples of types of control functions that may be considered are:

1. Functions of the form $u(t,\omega) = u(t)$ i.e. not depending on $\omega$. These controls are sometimes called **deterministic** or **open loop controls**.
2. Processes $\lbrace u_t\rbrace$ which are $\mathcal{M}_t$-adapted, i.e. for each $t$ the function $\omega \to u(t,\omega)$ is $\mathcal{M}_t$-measurable, where $\mathcal{M}_t$ is the $\sigma$-algebra generated by $\lbrace X_r^u;\, r \le t\rbrace$. These controls are called **closed loop** or **feedback controls**.
3. The controller has only *partial knowledge* of the state of the system. More precisely, to the controller's disposal are only (noisy) observations $R_t$ of $X_t$, given by an Itô process of the form $dR_t = a(t,X_t)\,dt + \gamma(t,X_t)\,d\widehat{B}_t$, where $\widehat{B}$ is a Brownian motion (not necessarily related to $B$). Hence the control process $\lbrace u_t\rbrace$ must be adapted w.r.t. the $\sigma$-algebra $\mathcal{N}_t$ generated by $\lbrace R_s;\, s \le t\rbrace$. In this situation the stochastic control problem is linked to the filtering problem (Chapter 6). If the equation (11.1.1) is linear and the performance function is integral quadratic (i.e. $F$ and $K$ are quadratic) then the stochastic control problem splits into a linear filtering problem and a corresponding deterministic control problem. This is called the **Separation Principle**.
4. Functions $u(t,\omega)$ of the form $u(t,\omega) = u_0(t, X_t(\omega))$ for some function $u_0\colon\mathbb{R}^{n+1} \to U \subset \mathbb{R}^k$. In this case we assume that $u$ does not depend on the starting point $y = (s,x)$: The value we choose at time $t$ only depends on the state of the system at this time. These are called **Markov controls**.

</div>

### 11.2 The Hamilton-Jacobi-Bellman Equation

Let us first consider only *Markov controls* $u = u(t, X_t(\omega))$. Introducing $Y_t = (s+t, X_{s+t})$ (as explained earlier) the system equation becomes

$$dY_t = b(Y_t, u(Y_t))\,dt + \sigma(Y_t, u(Y_t))\,dB_t.$$

For $v \in U$ and $f \in C_0^2(\mathbb{R} \times \mathbb{R}^n)$ define

$$(L^v f)(y) = \frac{\partial f}{\partial s}(y) + \sum_{i=1}^n b_i(y,v)\frac{\partial f}{\partial x_i} + \sum_{i,j=1}^n a_{ij}(y,v)\frac{\partial^2 f}{\partial x_i \partial x_j}$$

where $a_{ij} = \frac{1}{2}(\sigma\sigma^T)_{ij}$, $y = (s,x)$ and $x = (x_1,\ldots,x_n)$. Then for each choice of the function $u$ the solution $Y_t = Y_t^u$ is an Itô diffusion with generator $A$ given by $(Af)(y) = (L^{u(y)}f)(y)$ for $f \in C_0^2(\mathbb{R} \times \mathbb{R}^n)$.

For $v \in U$ define $F^v(y) = F(y,v)$. The first fundamental result in stochastic control theory is the following:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(11.2.1 — The Hamilton-Jacobi-Bellman (HJB) Equation (I))</span></p>

Define $\Phi(y) = \sup\lbrace J^u(y);\, u = u(Y) \text{ Markov control}\rbrace$.

Suppose $\Phi \in C^2(G) \cap C(\overline{G})$ satisfies $E^y\!\left[\lvert\Phi(Y_\alpha)\rvert + \int_0^\alpha \lvert L^v \Phi(Y_t)\rvert\,dt\right] < \infty$ for all bounded stopping times $\alpha \le T$, all $y \in G$ and all $v \in U$. Moreover, suppose that $T < \infty$ a.s. $Q^y$ for all $y \in G$ and that an optimal Markov control $u^*$ exists. Suppose $\partial G$ is regular for $Y_t^{u^*}$. Then

$$\sup_{v \in U}\lbrace F^v(y) + (L^v \Phi)(y)\rbrace = 0 \quad \text{for all } y \in G$$

and

$$\Phi(y) = K(y) \quad \text{for all } y \in \partial G.$$

The supremum in (11.2.3) is obtained if $v = u^*(y)$ where $u^*(y)$ is optimal. In other words,

$$F(y, u^*(y)) + (L^{u^*(y)}\Phi)(y) = 0 \quad \text{for all } y \in G.$$

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 11.2.1</summary>

The last two statements are easy to prove: Since $u^* = u^*(y)$ is optimal we have $\Phi(y) = J^{u^*}(y) = E^y\!\left[\int_0^T F(Y_s, u^*(Y_s))\,ds + K(Y_T)\right]$.

If $y \in \partial G$ then $T = 0$ a.s. $Q^y$ (since $\partial G$ is regular) and (11.2.4) follows. By the solution of the Dirichlet-Poisson problem (Theorem 9.3.3): $(L^{u^*(y)}\Phi)(y) = -F(y, u^*(y))$ for all $y \in G$, which is (11.2.5).

We proceed to prove (11.2.3). Fix $y = (s,x) \in G$ and choose a Markov control $u$. Let $\alpha \le T$ be a stopping time. Since $J^u(y) = E^y\!\left[\int_0^T F^u(Y_r)\,dr + K(Y_T)\right]$, we get by the strong Markov property (7.2.5), combined with (7.2.6) and (9.3.7):

$$E^y[J^u(Y_\alpha)] = E^y\!\left[E^{Y_\alpha}\!\left[\int_0^T F^u(Y_r)\,dr + K(Y_T)\right]\right] = E^y\!\left[\int_\alpha^T F^u(Y_r)\,dr + K(Y_T)\right] = J^u(y) - E^y\!\left[\int_0^\alpha F^u(Y_r)\,dr\right].$$

So $J^u(y) = E^y\!\left[\int_0^\alpha F^u(Y_r)\,dr\right] + E^y[J^u(Y_\alpha)]$.

Now let $W \subset G$ be of the form $W = \lbrace (r,z) \in G;\, r < t_1\rbrace$ where $s < t_1$. Put $\alpha = \inf\lbrace t \ge 0;\, Y_t \notin W\rbrace$. Suppose an optimal control $u^*(y) = u^*(r,z)$ exists and choose $u(r,z) = v$ if $(r,z) \in W$ and $u(r,z) = u^*(r,z)$ if $(r,z) \in G \setminus W$, where $v \in U$ is arbitrary. Then $\Phi(Y_\alpha) = J^{u^*}(Y_\alpha) = J^u(Y_\alpha)$ and therefore, combining (11.2.6) and (11.2.7) we obtain

$$\Phi(y) \ge J^u(y) = E^y\!\left[\int_0^\alpha F^v(Y_r)\,dr\right] + E^y[\Phi(Y_\alpha)].$$

Since $\Phi \in C^2(G)$ we get by Dynkin's formula $E^y[\Phi(Y_\alpha)] = \Phi(y) + E^y\!\left[\int_0^\alpha (L^u\Phi)(Y_r)\,dr\right]$, which substituted in (11.2.8) gives $\Phi(y) \ge E^y\!\left[\int_0^\alpha F^v(Y_r)\,dr\right] + \Phi(y) + E^y\!\left[\int_0^\alpha (L^v\Phi)(Y_r)\,dr\right]$, or $E^y\!\left[\int_0^\alpha (F^v(Y_r) + (L^v\Phi)(Y_r))\,dr\right] \le 0$.

So $\frac{E^y\!\left[\int_0^\alpha (F^v(Y_r) + (L^v\Phi)(Y_r))\,dr\right]}{E^y[\alpha]} \le 0$ for all such $W$. Letting $t_1 \downarrow s$ we obtain, since $F^v(\cdot)$ and $(L^v\Phi)(\cdot)$ are continuous at $y$, that $F^v(y) + (L^v\Phi)(y) \le 0$, which combined with (11.2.5) gives (11.2.3). $\square$

</details>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of the HJB (I) Equation)</span></p>

The HJB (I) equation states that if an optimal control $u^*$ exists, then we know that its value $v$ at the point $y$ is a point $v$ where the function $v \to F^v(y) + (L^v\Phi)(y)$; $v \in U$ attains its maximum (and this maximum is $0$). Thus the original stochastic control problem is associated to the easier problem of finding the maximum of a real function in $U \subset \mathbb{R}^k$. However, the HJB (I) equation only states that it is *necessary* that $v = u^*(y)$ is the maximum of this function. It is just as important to know if this is also *sufficient*.

</div>

The next result states that (under some conditions) sufficiency holds:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(11.2.2 — The HJB (II) Equation — A Converse of HJB (I))</span></p>

Let $\phi$ be a function in $C^2(G) \cap C(\overline{G})$ such that, for all $v \in U$,

$$F^v(y) + (L^v\phi)(y) \le 0; \quad y \in G$$

with boundary values $\lim_{t \to T} \phi(Y_t) = K(Y_T) \cdot \chi_{\lbrace T < \infty\rbrace}$ a.s. $Q^y$ and such that $\lbrace \phi(Y_\tau)\rbrace_{\tau \le T}$ is uniformly $Q^y$-integrable for all Markov controls $u$ and all $y \in G$.

Then

$$\phi(y) \ge J^u(y) \quad \text{for all Markov controls } u \text{ and all } y \in G.$$

Moreover, if for each $y \in G$ we have found $u_0(y)$ such that $F^{u_0(y)}(y) + (L^{u_0(y)}\phi)(y) = 0$, then $u_0 = u_0(y)$ is a Markov control such that $\phi(y) = J^{u_0}(y)$ and hence $u_0$ must be an optimal control and $\phi(y) = \Phi(y)$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 11.2.2</summary>

Assume that $\phi$ satisfies (11.2.9) and (11.2.10) above. Let $u$ be a Markov control. Since $L^u\phi \le -F^u$ in $G$ we have by Dynkin's formula

$$E^y[\phi(Y_{T_R})] = \phi(y) + E^y\!\left[\int_0^{T_R} (L^u\phi)(Y_r)\,dr\right] \le \phi(y) - E^y\!\left[\int_0^{T_R} F^u(Y_r)\,dr\right]$$

where $T_R = \min\lbrace R, T, \inf\lbrace t > 0;\, \lvert Y_t\rvert \ge R\rbrace\rbrace$ for all $R < \infty$. This gives, by (11.1.4), (11.2.10) and (11.2.11): $\phi(y) \ge E^y\!\left[\int_0^{T_R} F^u(Y_r)\,dr + \phi(Y_{T_R})\right] \to E^y\!\left[\int_0^T F^u(Y_r)\,dr + K(Y_T)\chi_{\lbrace T < \infty\rbrace}\right] = J^u(y)$ as $R \to \infty$, which proves (11.2.12). If $u_0$ is such that (11.2.13) holds, then the calculations above give equality and the proof is complete. $\square$

</details>

The HJB equations (I), (II) provide a very nice solution to the stochastic control problem in the case where only Markov controls are considered. One might feel that considering only Markov controls is too restrictive, but fortunately one can always obtain as good performance with a Markov control as with an arbitrary $\mathcal{F}_t^{(m)}$-adapted control, at least if some extra conditions are satisfied:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(11.2.3 — Markov Controls are Sufficient)</span></p>

Let

$$\Phi_M(y) = \sup\lbrace J^u(y);\, u = u(Y) \text{ Markov control}\rbrace$$

and

$$\Phi_a(y) = \sup\lbrace J^u(y);\, u = u(t,\omega)\ \mathcal{F}_t^{(m)}\text{-adapted control}\rbrace.$$

Suppose there exists an optimal Markov control $u_0 = u_0(Y)$ for the Markov control problem (i.e. $\Phi_M(y) = J^{u_0}(y)$ for all $y \in G$) such that all the boundary points of $G$ are regular w.r.t. $Y_t^{u_0}$ and that $\Phi_M$ is a function in $C^2(G) \cap C(\overline{G})$ satisfying

$$E^y\!\left[\lvert\Phi_M(Y_\alpha)\rvert + \int_0^\alpha \lvert L^u\Phi_M(Y_t)\rvert\,dt\right] < \infty$$

for all bounded stopping times $\alpha \le T$, all adapted controls $u$ and all $y \in G$. Then

$$\Phi_M(y) = \Phi_a(y) \quad \text{for all } y \in G.$$

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 11.2.3</summary>

Let $\phi$ be a function in $C^2(G) \cap C(\overline{G})$ satisfying (11.2.15) and $F^v(y) + (L^v\phi)(y) \le 0$ for all $y \in G$, $v \in U$, and $\phi(y) = K(y)$ for all $y \in \partial G$.

Let $u_t(\omega) = u(t,\omega)$ be an $\mathcal{F}_t^{(m)}$-adapted control. Then $Y_t$ is an Itô process given by $dY_t = b(Y_t, u_t)\,dt + \sigma(Y_t, u_t)\,dB_t$, so by Lemma 7.3.2, with $T_R$ as in (11.2.14):

$$E^y[\phi(Y_{T_R})] = \phi(y) + E^y\!\left[\int_0^{T_R} (L^{u(t,\omega)}\phi)(Y_t)\,dt\right]$$

where $(L^{u(t,\omega)}\phi)(y) = \frac{\partial\phi}{\partial t}(y) + \sum_{i=1}^n b_i(y, u(t,\omega))\frac{\partial\phi}{\partial x_i}(y) + \sum_{i,j=1}^n a_{ij}(y, u(t,\omega))\frac{\partial^2\phi}{\partial x_i \partial x_j}(y)$ with $a_{ij} = \frac{1}{2}(\sigma\sigma^T)_{ij}$. Thus by (11.2.16) and (11.2.17) this gives

$$E^y[\phi(Y_{T_R})] \le \phi(y) - E^y\!\left[\int_0^{T_R} F(Y_t, u(t,\omega))\,dt\right].$$

Letting $R \to \infty$ we obtain $\phi(y) \ge J^u(y)$.

But by Theorem 11.2.1 the function $\phi(y) = \Phi_M(y)$ satisfies (11.2.16) and (11.2.17). So by (11.2.19) we have $\Phi_M(y) \ge \Phi_a(y)$ and Theorem 11.2.3 follows. $\square$

</details>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Minimum Problem)</span></p>

The theory above also applies to the corresponding *minimum* problem $\Psi(y) = \inf_u J^u(y) = J^{u^*}(y)$. To see the connection we note that $\Psi(y) = -\sup_u\lbrace -J^u(y)\rbrace = -\sup_u\!\left\lbrace E^y\!\left[\int_0^T -F^u(Y_t)\,dt - K(Y_t)\right]\right\rbrace$, so $-\Psi$ coincides with the solution $\Phi$ of the problem (11.1.9), but with $F$ replaced by $-F$ and $K$ replaced by $-K$. Using this, we see that the HJB equations apply to $\Psi$ also but with reverse inequalities. For example, equation (11.2.3) for $\Phi$ gets for $\Psi$ the form

$$\inf_{v \in U}\lbrace F^v(y) + (L^v\Psi)(y)\rbrace = 0 \quad \text{for all } y \in G.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(11.2.4 — The Linear Stochastic Regulator Problem)</span></p>

Suppose the state $X_t$ of the system at time $t$ is given by a linear stochastic differential equation:

$$dX_t = (H_t X_t + M_t u_t)dt + \sigma_t dB_t, \quad t \ge s;\; X_s = x$$

and the cost is of the form

$$J^u(s,x) = E^{s,x}\!\left[\int_s^{t_1}\lbrace X_t^T C_t X_t + u_t^T D_t u_t\rbrace dt + X_{t_1}^T R X_{t_1}\right], \quad s \le t_1$$

where all the coefficients $H_t \in \mathbf{R}^{n \times n}$, $M_t \in \mathbf{R}^{n \times k}$, $\sigma_t \in \mathbf{R}^{n \times m}$, $C_t \in \mathbf{R}^{n \times n}$, $D_t \in \mathbf{R}^{k \times k}$ and $R \in \mathbf{R}^{n \times n}$ are $t$-continuous and deterministic. We assume that $C_t$ and $R$ are symmetric, nonnegative definite and $D_t$ is symmetric, positive definite, for all $t$. We also assume that $t_1$ is a deterministic time.

The problem is to choose the control $u = u(t, X_t) \in \mathbf{R}^k$ such that it minimizes $J^u(s,x)$. The aim is to find a control $u$ which makes $\lvert X_t\rvert$ small fast and such that the energy used ($\sim u^T D u$) is small. The sizes of $C_t$ and $R$ reflect the cost of having large values of $\lvert X_t\rvert$, while the size of $D_t$ reflects the cost (energy) of applying large values of $\lvert u_t\rvert$.

The HJB equation for $\Psi(s,x) = \inf_u J^u(s,x)$ becomes

$$0 = \inf_v\lbrace F^v(s,x) + (L^v\Psi)(s,x)\rbrace$$

$$= \frac{\partial\Psi}{\partial s} + \inf_v\!\left\lbrace x^T C_s x + v^T D_s v + \sum_{i=1}^n (H_s x + M_s v)_i \frac{\partial\Psi}{\partial x_i} + \frac{1}{2}\sum_{i,j=1}^n (\sigma_s \sigma_s^T)_{ij}\frac{\partial^2\Psi}{\partial x_i \partial x_j}\right\rbrace$$

for $s < t_1$, with boundary condition $\Psi(t_1, x) = x^T R x$.

We try a solution $\psi$ of the form

$$\psi(t,x) = x^T S_t x + a_t$$

where $S(t) = S_t \in \mathbf{R}^{n \times n}$ is symmetric, nonnegative definite, $a_t \in \mathbf{R}$ and both $a_t$ and $S_t$ are continuously differentiable w.r.t. $t$. To obtain the boundary condition we put

$$S_{t_1} = R, \quad a_{t_1} = 0.$$

Using the ansatz we get

$$F^v(t,x) + (L^v\psi)(t,x) = x^T S_t' x + a_t' + x^T C_t x + v^T D_t v + (H_t x + M_t v)^T(S_t x + S_t^T x) + \sum_{i,j}(\sigma_t \sigma_t^T)_{ij} S_{ij}$$

where $S_t' = \frac{d}{dt}S_t$, $a_t' = \frac{d}{dt}a_t$. The minimum of this expression is obtained when

$$\frac{\partial}{\partial v_i}(F^v(t,x) + (L^v\psi)(t,x)) = 0; \quad i = 1,\dots,k$$

i.e. when $2D_t v + 2M_t^T S_t x = 0$, i.e. when

$$v = -D_t^{-1}M_t^T S_t x.$$

We substitute this value of $v$ and obtain

$$F^v(t,x) + (L^v\psi)(t,x) = x^T(S_t' + C_t - S_t M_t D_t^{-1}M_t^T S_t + 2H_t^T S_t)x + a_t' + tr(\sigma\sigma^T S)_t$$

where $tr$ denotes the (matrix) trace. We obtain that this is $0$ if we choose $S_t$ such that

$$S_t' = -2H_t^T S_t + S_t M_t D_t^{-1} M_t^T S_t - C_t; \quad t < t_1$$

and $a_t$ such that

$$a_t' = -tr(\sigma\sigma^T S)_t; \quad t < t_1.$$

We recognize the equation for $S_t$ as a **Riccati type equation** from linear filtering theory. With the boundary condition $S_{t_1} = R$ it determines $S_t$ uniquely. Combining with the boundary condition $a_{t_1} = 0$ we obtain

$$a_t = \int_t^{t_1} tr(\sigma\sigma^T S)_s\,ds.$$

With such a choice of $S_t$ and $a_t$, by Theorem 11.2.2 we conclude that

$$u^*(t,x) = -D_t^{-1}M_t^T S_t x, \quad t < t_1$$

is an optimal control and the minimum cost is

$$\Psi(s,x) = x^T S_s x + \int_s^{t_1} tr(\sigma\sigma^T S)_t\,dt, \quad s < t_1.$$

This formula shows that the extra cost due to the noise in the system is given by $a_s = \int_s^{t_1} tr(\sigma\sigma^T S)_t\,dt$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Separation Principle)</span></p>

The **Separation Principle** (see Davis (1977), Davis and Vinter (1985) or Fleming and Rishel (1975)) states that if we had only partial knowledge of the state $X_t$ of the system, i.e. if we only had noisy observations

$$dZ_t = g_t X_t dt + \gamma_t d\widetilde{B}_t$$

to our disposal, then the optimal control $u^*(t,\omega)$ (required to be $\mathcal{G}_t$-adapted, where $\mathcal{G}_t$ is the $\sigma$-algebra generated by $\lbrace Z_r;\, r \le t\rbrace$), would be given by

$$u^*(t,\omega) = -D_t^{-1}M_t^T S_t \widehat{X}_t(\omega)$$

where $\widehat{X}_t$ is the filtered estimate of $X_t$ based on the observations $\lbrace Z_r;\, r \le t\rbrace$, given by the Kalman-Bucy filter. This shows that the stochastic control problem in this case splits into a **linear filtering problem** and a **deterministic control problem**.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(11.2.5 — An Optimal Portfolio Selection Problem)</span></p>

Let $X_t$ denote the wealth of a person at time $t$. Suppose that the person has the choice of two different investments. The price $p_1(t)$ at time $t$ of one of the assets is assumed to satisfy the equation

$$\frac{dp_1}{dt} = p_1(a + \alpha W_t)$$

where $W_t$ denotes white noise and $a, \alpha > 0$ are constants measuring the average relative rate of change of $p$ and the size of the noise, respectively. We interpret this as the Itô stochastic differential equation

$$dp_1 = p_1 a\,dt + p_1 \alpha\,dB_t.$$

This investment is called *risky*, since $\alpha > 0$. We assume that the price $p_2$ of the other asset satisfies a similar equation, but with no noise:

$$dp_2 = p_2 b\,dt.$$

This investment is called *safe*. So it is natural to assume $b < a$. At each instant the person can choose how big fraction $u$ of his wealth he will invest in the risky asset, thereby investing the fraction $1 - u$ in the safe one. This gives the following stochastic differential equation for the wealth $X_t = X_t^u$:

$$dX_t = X_t(au + b(1-u))dt + \alpha u X_t dB_t.$$

Suppose that, starting with the wealth $X_t = x > 0$ at time $t$, the person wants to maximize the expected utility of the wealth at some future time $t_0 > t$. If we allow no borrowing (i.e. require $X \ge 0$) and are given a utility function $N\colon [0,\infty) \to [0,\infty)$, $N(0) = 0$ (usually assumed to be increasing and concave), the problem is to find $\Phi(s,x)$ and a (Markov) control $u^* = u^*(t, X_t)$, $0 \le u^* \le 1$, such that

$$\Phi(s,x) = \sup\lbrace J^u(s,x);\; u \text{ Markov control}, 0 \le u \le 1\rbrace = J^{u^*}(s,x)$$

where $J^u(s,x) = E^{s,x}[N(X_T^u)]$ and $T$ is the first exit time from the region $G = \lbrace (r,z);\, r < t_0,\, z > 0\rbrace$. The differential operator $L^v$ has the form

$$(L^v f)(t,x) = \frac{\partial f}{\partial t} + x(av + b(1-v))\frac{\partial f}{\partial x} + \frac{1}{2}\alpha^2 v^2 x^2 \frac{\partial^2 f}{\partial x^2}.$$

The HJB equation becomes

$$\sup_v\lbrace (L^v\Phi)(t,x)\rbrace = 0, \quad \text{for } (t,x) \in G$$

with boundary conditions $\Phi(t,x) = N(x)$ for $t = t_0$, and $\Phi(t,0) = N(0)$ for $t < t_0$.

Therefore, for each $(t,x)$ we try to find the value $v = u(t,x)$ which maximizes

$$\eta(v) = L^v\Phi = \frac{\partial\Phi}{\partial t} + x(b + (a-b)v)\frac{\partial\Phi}{\partial x} + \frac{1}{2}\alpha^2 v^2 x^2 \frac{\partial^2\Phi}{\partial x^2}.$$

If $\Phi_x := \frac{\partial\Phi}{\partial x} > 0$ and $\Phi_{xx} := \frac{\partial^2\Phi}{\partial x^2} < 0$, the solution is

$$v = u(t,x) = -\frac{(a-b)\Phi_x}{x\alpha^2 \Phi_{xx}}.$$

If we substitute this into the HJB equation we get the following nonlinear boundary value problem for $\Phi$:

$$\Phi_t + bx\Phi_x - \frac{(a-b)^2 \Phi_x^2}{2\alpha^2 \Phi_{xx}} = 0 \quad \text{for } t < t_0,\, x > 0$$

$$\Phi(t,x) = N(x) \quad \text{for } t = t_0 \text{ or } x = 0.$$

**Power utility functions.** Important examples of increasing and concave functions are the power functions $N(x) = x^r$ where $0 < r < 1$. We try to find a solution of the form $\phi(t,x) = f(t)x^r$. Substituting we obtain

$$\phi(t,x) = e^{\lambda(t_0 - t)}x^r$$

where $\lambda = br + \frac{(a-b)^2 r}{2\alpha^2(1-r)}$. Using the formula for the optimal control:

$$u^*(t,x) = \frac{a - b}{\alpha^2(1-r)}.$$

If $\frac{a-b}{\alpha^2(1-r)} \in (0,1)$ this is the solution to the problem, in virtue of Theorem 11.2.2. Note that $u^*$ is in fact constant.

**Kelly criterion.** Another interesting choice of the utility function is $N(x) = \log x$, called the Kelly criterion. In this case we may obtain the optimal control directly by evaluating $E^{s,x}[\log(X_T)]$ using Dynkin's formula:

$$E^{s,x}[\log(X_T)] = \log x + E^{s,x}\!\left[\int_s^T \lbrace au(t,X_t) + b(1 - u(t,X_t)) - \tfrac{1}{2}\alpha^2 u^2(t,X_t)\rbrace dt\right]$$

since $L^v(\log x) = av + b(1-v) - \frac{1}{2}\alpha^2 v^2$. So $J^u(s,x) = E^{s,x}[\log(X_T)]$ is maximal if we for all $r, z$ choose $u(s,z)$ to be the value of $v$ which maximizes $av + b(1-v) - \frac{1}{2}\alpha^2 v^2$, i.e. we choose

$$v = u(t, X_t) = \frac{a - b}{\alpha^2} \quad \text{for all } t, \omega.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(11.2.6 — Breakdown of HJB Theory)</span></p>

This example shows that even quite simple — and apparently innocent — stochastic control problems can lead us beyond the reach of the theory developed in this chapter.

Suppose the system is a 1-dimensional Itô integral

$$dX_t = dX_t^u = u(t,\omega)dB_t, \quad t \ge s;\; X_s = x > 0$$

and consider the stochastic control problem

$$\Phi(t,x) = \sup_u E^{t,x}[K(X_\tau^u)]$$

where $\tau$ is the first exit time from $G = \lbrace (r,z);\, r \le t_1,\, z > 0\rbrace$ for $Y_t = (s + t, X_{s+t}^{s,x})$ and $K$ is a given bounded continuous function.

Intuitively, we can think of the system as the state of a game which behaves like an "excited" Brownian motion, where we can control the size $u$ of the excitation at every instant. The purpose of the control is to maximize the expected payoff $K(X_{t_1})$ of the game at a fixed future time $t_1$.

Assuming that $\Phi \in C^2$ and that $u^*$ exists we get by the HJB (I) equation

$$\sup_{v \in \mathbf{R}}\!\left\lbrace \frac{\partial\Phi}{\partial t} + \frac{1}{2}v^2 \frac{\partial^2\Phi}{\partial x^2}\right\rbrace = 0 \quad \text{for } t < t_1,\; \Phi(t_1, x) = K(x).$$

From this we see that we necessarily have $\frac{\partial^2\Phi}{\partial x^2} \le 0$, $v^* \frac{\partial^2\Phi}{\partial x^2} = 0$ and $\frac{\partial\Phi}{\partial t} = 0$ for $t < t_1$. But if $\frac{\partial\Phi}{\partial t} = 0$, then $\Phi(t,x) = \Phi(t_1, x) = K(x)$. However, this cannot possibly be the solution in general, because we have not assumed that $\frac{\partial^2 K}{\partial x^2} \le 0$ — in fact, $K$ was not even assumed to be differentiable.

**What went wrong?** Since the conclusion of the HJB (I) equation was wrong, the assumptions cannot hold. So either $\Phi$ is not $C^2$ or $u^*$ does not exist, or both.

To simplify the problem assume that $K(x) = \begin{cases} x^2 & 0 \le x \le 1 \\\\ 1 & x > 1\end{cases}$.

Then considering the figure above and using some intuition we see that it is optimal to excite as much as possible if $X_t$ is in the strip $0 < x < 1$ to avoid exiting from $G$ in the interval $\lbrace t_1\rbrace \times (0,1)$. Using that $X_t$ is just a time change of Brownian motion, this optimal control leads to a process $X^*$ which jumps immediately to the value $1$ with probability $x$ and to the value $0$ with probability $1-x$, if the starting point is $x \in (0,1)$. If the starting point is $x \in [1,\infty)$ we simply choose our control to be zero. In other words, heuristically we should have

$$u^*(t,x) = \begin{cases} \infty & \text{if } x \in (0,1) \\\\ 0 & \text{if } x \in [1,\infty)\end{cases}$$

with corresponding expected payoff

$$\phi^*(s,x) = E^{s,x}[K(X_{t_1}^*)] = \begin{cases} x & \text{if } 0 \le x \le 1 \\\\ 1 & \text{if } x > 1.\end{cases}$$

Thus our candidate $u^*$ for optimal control is not continuous (not even finite!) and the corresponding optimal process $X_t^*$ is not an Itô diffusion (it is not even continuous). To handle this case mathematically it is necessary to enlarge the family of admissible controls (and the family of corresponding processes).

This last example illustrates the importance of the question of **existence**, in general, both of the optimal control $u^*$ and of the corresponding solution $X_t$ of the stochastic differential equation. With certain conditions on $b, \sigma, F, \partial G$ and assuming that the set of control values is compact, one can show, using general results from nonlinear partial differential equations, that a smooth function $\phi$ exists such that

$$\sup_v\lbrace F^v(y) + (L^v\phi)(y)\rbrace = 0 \quad \text{for } y \in G$$

and $\phi(y) = K(y)$ for $y \in \partial G$. Then by a measurable selection theorem one can find a (measurable) function $u^*(y)$ such that

$$F^{u^*}(y) + (L^{u^*}\phi)(y) = 0$$

for a.a. $y \in G$ w.r.t. Lebesgue measure in $\mathbf{R}^{n+1}$. Even if $u^*$ is only known to be measurable, one can show that the corresponding solution $X_t = X_t^{u^*}$ of the SDE exists (see Stroock and Varadhan (1979) for general results in this direction).

</div>

## 11.3 Stochastic Control Problems with Terminal Conditions

In many applications there are constraints on the types of Markov controls $u$ to be considered, for example in terms of the probabilistic behaviour of $Y_t^u$ at the terminal time $t = T$. Such problems can often be handled by applying a kind of "Lagrange multiplier" method, which we now describe.

Consider the problem of finding $\Phi(y)$ and $u^*(y)$ such that

$$\Phi(y) = \sup_{u \in \mathcal{K}} J^u(y) = J^{u^*}(y)$$

where

$$J^u(y) = E^y\!\left[\int_0^T F^u(Y_t^u)dt + K(Y_T^u)\right]$$

and where the supremum is taken over the space $\mathcal{K}$ of all Markov controls $u\colon \mathbf{R}^{n+1} \to U \subset \mathbf{R}^k$ such that

$$E^y[M_i(Y_T^u)] = 0, \quad i = 1, 2, \dots, l$$

where $M = (M_1, \dots, M_l)\colon \mathbf{R}^{n+1} \to \mathbf{R}^l$ is a given continuous function, with $E^y[\lvert M(Y_T^u)\rvert] < \infty$ for all $y, u$.

Now we introduce a related, but unconstrained problem as follows: For each $\lambda \in \mathbf{R}^l$ and each Markov control $u$ define

$$J_\lambda^u(y) = E^y\!\left[\int_0^T F^u(Y_t^u)dt + K(Y_T^u) + \lambda \cdot M(Y_T^u)\right]$$

where $\cdot$ denotes the inner product in $\mathbf{R}^l$. Find $\Phi_\lambda(y)$ and $u_\lambda^*(y)$ such that

$$\Phi_\lambda(y) = \sup_u J_\lambda^u(y) = J_\lambda^{u_\lambda^*}(y)$$

**without** terminal conditions.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(11.3.1 — Lagrange Multiplier Method for Stochastic Control)</span></p>

Suppose that we for all $\lambda \in \Lambda \subset \mathbf{R}^l$ can find $\Phi_\lambda(y)$ and $u_\lambda^*$ solving the (unconstrained) stochastic control problem (11.3.5)–(11.3.6). Moreover, suppose that there exists $\lambda_0 \in \Lambda$ such that

$$E^y[M(Y_T^{u_{\lambda_0}^*})] = 0.$$

Then $\Phi(y) := \Phi_{\lambda_0}(y)$ and $u^* := u_{\lambda_0}^*$ solves the constrained stochastic control problem (11.3.1)–(11.3.3).

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 11.3.1</summary>

Let $u$ be a Markov control, $\lambda \in \Lambda$. Then by the definition of $u_\lambda^*$ we have

$$E^y\!\left[\int_0^T F^{u_\lambda^*}(Y_t^{u_\lambda^*})dt + K(Y_T^{u_\lambda^*}) + \lambda \cdot M(Y_T^{u_\lambda^*})\right] = J_\lambda^{u_\lambda^*}(y) \ge J_\lambda^u(y) = E^y\!\left[\int_0^T F^u(Y_t^u)dt + K(Y_T^u) + \lambda \cdot M(Y_T^u)\right].$$

In particular, if $\lambda = \lambda_0$ and $u \in \mathcal{K}$ then

$$E^y[M(Y_T^{u_{\lambda_0}^*})] = 0 = E^y[M(Y_T^u)]$$

and hence by the inequality above

$$J^{u_{\lambda_0}^*}(y) \ge J^u(y) \quad \text{for all } u \in \mathcal{K}.$$

Since $u_{\lambda_0}^* \in \mathcal{K}$ the proof is complete. $\square$

</details>

## 12. Application to Mathematical Finance

### 12.1 Market, Portfolio and Arbitrage

In this chapter we describe how the concepts, methods and results in the previous chapters can be applied to give a rigorous mathematical model of finance. We will concentrate on the most fundamental issues and those topics which are most closely related to the theory in this book.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(12.1.1 — Market, Portfolio, Self-Financing)</span></p>

**a)** A **market** is an $\mathcal{F}_t^{(m)}$-adapted $(n+1)$-dimensional Itô process $X(t) = (X_0(t), X_1(t), \dots, X_n(t))$; $0 \le t \le T$ which we will assume has the form

$$dX_0(t) = \rho(t,\omega)X_0(t)dt; \quad X_0(0) = 1$$

and

$$dX_i(t) = \mu_i(t,\omega)dt + \sum_{j=1}^m \sigma_{ij}(t,\omega)dB_j(t) = \mu_i(t,\omega)dt + \sigma_i(t,\omega)dB(t); \quad X_i(0) = x_i$$

where $\sigma_i$ is row number $i$ of the $n \times m$ matrix $[\sigma_{ij}]$; $1 \le i \le n \in \mathbb{N}$.

**b)** The market $\lbrace X(t)\rbrace_{t \in [0,T]}$ is called **normalized** if $X_0(t) \equiv 1$.

**c)** A **portfolio** in the market $\lbrace X(t)\rbrace_{t \in [0,T]}$ is an $(n+1)$-dimensional $(t,\omega)$-measurable and $\mathcal{F}_t^{(m)}$-adapted stochastic process

$$\theta(t,\omega) = (\theta_0(t,\omega), \theta_1(t,\omega), \dots, \theta_n(t,\omega)); \quad 0 \le t \le T.$$

**d)** The **value** at time $t$ of a portfolio $\theta(t)$ is defined by

$$V(t,\omega) = V^\theta(t,\omega) = \theta(t) \cdot X(t) = \sum_{i=0}^n \theta_i(t)X_i(t)$$

where $\cdot$ denotes inner product in $\mathbf{R}^{n+1}$.

**e)** The portfolio $\theta(t)$ is called **self-financing** if

$$dV(t) = \theta(t) \cdot dX(t)$$

i.e. $V(t) = V(0) + \int_0^t \theta(s) \cdot dX(s)$ for $t \in [0, T]$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Comments on Definition 12.1.1)</span></p>

**a)** We think of $X_i(t) = X_i(t,\omega)$ as the price of security/asset number $i$ at time $t$. The assets number $1, \dots, n$ are called *risky* because of the presence of their diffusion terms. They can for example represent stock investments. The asset number $0$ is called *safe* because of the absence of a diffusion term (although $\rho(t,\omega)$ may depend on $\omega$). This asset can for example represent a bank investment. For simplicity we will assume that $\rho(t,\omega)$ is bounded.

**b)** Note that we can always make the market normalized by defining $\overline{X}_i(t) = X_0(t)^{-1}X_i(t)$; $1 \le i \le n$. The market $\overline{X}(t) = (1, \overline{X}_1(t), \dots, \overline{X}_n(t))$ is called the **normalization** of $X(t)$. Thus normalization corresponds to regarding the price $X_0(t)$ of the safe investment as the unit of price (the *numeraire*) and computing the other prices in terms of this unit. Since

$$X_0(t) = \exp\!\left(\int_0^t \rho(s,\omega)ds\right)$$

we have $\xi(t) := X_0^{-1}(t) = \exp\!\left(-\int_0^t \rho(s,\omega)ds\right) > 0$ for all $t \in [0,T]$ and

$$d\overline{X}_i(t) = d(\xi(t)X_i(t)) = \xi(t)[(\mu_i - \rho X_i)dt + \sigma_i dB(t)]; \quad 1 \le i \le n$$

or $d\overline{X}(t) = \xi(t)[dX(t) - \rho(t)X(t)dt]$.

**c)** The components $\theta_0(t,\omega), \dots, \theta_n(t,\omega)$ represent the *number of units* of the securities number $0, \dots, n$, respectively, which an investor holds at time $t$.

**d)** This is simply the total value of all investments held at time $t$.

**e)** The self-financing condition means that changes in portfolio value come only from changes in asset prices — no money is brought in or taken out from the system. If investments $\theta(t_k)$ are made at discrete times $t = t_k$, the increase in the wealth $\Delta V(t_k) = V(t_{k+1}) - V(t_k)$ should be given by $\Delta V(t_k) = \theta(t_k) \cdot \Delta X(t_k)$ where $\Delta X(t_k) = X(t_{k+1}) - X(t_k)$. In the continuous time limit as $\Delta t_k = t_{k+1} - t_k \to 0$, this gives (with the Itô interpretation of the integral) the self-financing condition.

**f)** Note that if $\theta$ is self-financing for $X(t)$ and $\overline{V}^\theta(t) = \theta(t) \cdot \overline{X}(t) = \xi(t)V^\theta(t)$ is the value process of the normalized market, then by Itô's formula:

$$d\overline{V}^\theta(t) = \xi(t)dV^\theta(t) + V^\theta(t)d\xi(t) = \xi(t)\theta(t)[dX(t) - \rho(t)X(t)dt] = \theta(t)d\overline{X}(t).$$

Hence $\theta$ is also self-financing for the normalized market.

</div>

Note that by combining the portfolio value equation and the self-financing condition we get

$$\theta_0(t)X_0(t) + \sum_{i=1}^n \theta_i(t)X_i(t) = V(0) + \int_0^t \theta_0(s)dX_0(s) + \sum_{i=1}^n \int_0^t \theta_i(s)dX_i(s).$$

Hence, if we put $Y_0(t) = \theta_0(t)X_0(t)$, then $dY_0(t) = \rho(t)Y_0(t)dt + dA(t)$, where

$$A(t) = \sum_{i=1}^n\!\left(\int_0^t \theta_i(s)dX_i(s) - \theta_i(t)X_i(t)\right).$$

This equation has the solution $\theta_0(t) = V(0) + \xi(t)A(t) + \int_0^t \rho(s)A(s)\xi(s)ds$. In particular, if $\rho = 0$ this gives $\theta_0(t) = V(0) + A(t)$.

Therefore, if $\theta_1(t), \dots, \theta_n(t)$ are chosen, we can always make the portfolio $\theta(t) = (\theta_0(t), \theta_1(t), \dots, \theta_n(t))$ self-financing by choosing $\theta_0(t)$ accordingly.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(12.1.2 — Admissible Portfolio)</span></p>

A portfolio $\theta(t)$ which satisfies the integrability condition (12.1.5) and which is self-financing is called **admissible** if the corresponding value process $V^\theta(t)$ is $(t,\omega)$ a.s. lower bounded, i.e. there exists $K = K(\theta) < \infty$ such that

$$V^\theta(t,\omega) \ge -K \quad \text{for a.a. } (t,\omega) \in [0,T] \times \Omega.$$

This is the analogue of a *tame* portfolio in the context of Karatzas (1996). The restriction reflects a natural condition in real life finance: there must be a limit to how much debt the creditors can tolerate.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(12.1.3 — Arbitrage)</span></p>

An admissible portfolio $\theta(t)$ is called an **arbitrage** (in the market $\lbrace X_t\rbrace_{t \in [0,T]}$) if the corresponding value process $V^\theta(t)$ satisfies $V^\theta(0) = 0$ and

$$V^\theta(T) \ge 0 \quad \text{a.s.} \quad \text{and} \quad P[V^\theta(T) > 0] > 0.$$

In other words, $\theta(t)$ is an arbitrage if it gives an increase in the value from time $t = 0$ to time $t = T$ a.s., and a strictly positive increase with positive probability. So $\theta(t)$ generates a profit without any risk of losing money.

</div>

Intuitively, the existence of an arbitrage is a sign of lack of equilibrium in the market: No real market equilibrium can exist in the long run if there are arbitrages there. Therefore it is important to be able to determine if a given market allows an arbitrage or not.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(12.1.4 — Why Admissibility Matters)</span></p>

Consider the following market: $dX_0(t) = 0$, $dX_1(t) = dB(t)$, $0 \le t \le T = 1$. Let

$$Y(t) = \int_0^t \frac{dB(s)}{\sqrt{1 - s}} \quad \text{for } 0 \le t < 1.$$

By Corollary 8.5.5 there exists a Brownian motion $\widehat{B}(t)$ such that $Y(t) = \widehat{B}(\beta_t)$ where $\beta_t = \int_0^t \frac{ds}{1-s} = \ln\!\left(\frac{1}{1-t}\right)$ for $0 \le t < 1$.

Let $a \in \mathbf{R}$ be a given constant and define $\tau := \tau_a := \inf\lbrace t > 0;\, \widehat{B}(t) = a\rbrace$ and $\alpha := \alpha_a := \inf\lbrace t > 0;\, Y(t) = a\rbrace$. Then $\tau < \infty$ a.s. and $\tau = \ln\!\left(\frac{1}{1-\alpha}\right)$, so $\alpha < 1$ a.s.

Let $\theta(t) = (\theta_0(t), \theta_1(t))$ be a self-financing portfolio with $\theta_1(t) = \begin{cases} \frac{1}{\sqrt{1-t}} & \text{for } 0 \le t < \alpha \\\\ 0 & \text{for } \alpha \le t \le 1\end{cases}$.

Then $V(t) = \int_0^{t \wedge \alpha} \frac{dB(s)}{\sqrt{1-s}} = Y(t \wedge \alpha)$ for $0 \le t \le 1$, and $V(1) = Y(\alpha) = a$ a.s. This portfolio is self-financing and satisfies the integrability condition, but $\theta(t)$ is **not** admissible because $V(t) = Y(t \wedge \alpha) = \widehat{B}(\ln(\frac{1}{1 - t \wedge \alpha}))$ is not $(t,\omega)$-a.s. lower bounded. This example illustrates that with portfolios only required to be self-financing and satisfy the integrability condition, one can virtually generate any terminal value $V(T,\omega)$ from $V_0 = 0$, even when the risky price process $X_1(t)$ is Brownian motion.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(12.1.5 — Dudley's Representation Theorem)</span></p>

Let $F$ be an $\mathcal{F}_T^{(m)}$-measurable random variable and let $B(t)$ be $m$-dimensional Brownian motion. Then there exists $\phi \in \mathcal{W}^m$ such that

$$F(\omega) = \int_0^T \phi(t,\omega)dB(t).$$

Note that $\phi$ is not unique. This implies that for *any* constant $z$ there exists $\phi \in \mathcal{W}^m$ such that $F(\omega) = z + \int_0^T \phi(t,\omega)dB(t)$. Thus, if we let $m = n$ and interpret $B_1(t) = X_1(t), \dots, B_n(t) = X_n(t)$ as prices, and put $X_0(t) \equiv 1$, this means that we can, with *any* initial fortune $z$, generate *any* $\mathcal{F}_T^{(m)}$-measurable final value $F = V(T)$, as long as we are allowed to choose the portfolio $\phi$ freely from $\mathcal{W}^m$. This again underlines the need for some extra restriction on the family of portfolios allowed, like the admissibility condition.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(12.1.6 — No Arbitrage via Equivalent Local Martingale Measure)</span></p>

Suppose there exists a measure $Q$ on $\mathcal{F}_T^{(m)}$ such that $P \sim Q$ and such that the normalized price process $\lbrace \overline{X}(t)\rbrace_{t \in [0,T]}$ is a local martingale w.r.t. $Q$. Then the market $\lbrace X(t)\rbrace_{t \in [0,T]}$ has no arbitrage.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Lemma 12.1.6</summary>

Suppose $\theta(t)$ is an arbitrage for $\lbrace \overline{X}(t)\rbrace_{t \in [0,T]}$. Let $\overline{V}^\theta(t)$ be the corresponding value process for the normalized market with $\overline{V}^\theta(0) = 0$. Then $\overline{V}^\theta(t)$ is a lower bounded local martingale w.r.t. $Q$, by (12.1.14). Therefore $\overline{V}^\theta(t)$ is a *supermartingale* w.r.t. $Q$, by Exercise 7.12. Hence

$$E_Q[V^\theta(T)] \le V^\theta(0) = 0.$$

But since $V^\theta(T,\omega) \ge 0$ a.s. $P$ we have $V^\theta(T,\omega) \ge 0$ a.s. $Q$ (because $Q \ll P$) and since $P[V^\theta(T) > 0] > 0$ we have $Q[V^\theta(T) > 0] > 0$ (because $P \ll Q$). This implies that $E_Q[V^\theta(T)] > 0$, which contradicts the inequality above. Hence arbitrages do not exist for the normalized price process $\lbrace \overline{X}(t)\rbrace$. It follows that $\lbrace X(t)\rbrace$ has no arbitrage. $\square$

</details>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(12.1.7 — Equivalent (Local) Martingale Measure)</span></p>

A measure $Q \sim P$ such that the normalized process $\lbrace \overline{X}(t)\rbrace_{t \in [0,T]}$ is a (local) martingale w.r.t. $Q$ is called an **equivalent (local) martingale measure**.

Thus Lemma 12.1.6 states that if there exists an equivalent local martingale measure then the market has no arbitrage. In fact, then the market also satisfies the stronger condition "no free lunch with vanishing risk" (NFLVR). Conversely, if the market satisfies the NFLVR condition, then there exists an equivalent martingale measure.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(12.1.8 — Characterization of No Arbitrage)</span></p>

**a)** Suppose there exists a process $u(t,\omega) \in \mathcal{V}^m(0,T)$ such that, with $\widehat{X}(t,\omega) = (X_1(t,\omega), \dots, X_n(t,\omega))$,

$$\sigma(t,\omega)u(t,\omega) = \mu(t,\omega) - \rho(t,\omega)\widehat{X}(t,\omega) \quad \text{for a.a. } (t,\omega)$$

and such that

$$E\!\left[\exp\!\left(\tfrac{1}{2}\int_0^T u^2(t,\omega)dt\right)\right] < \infty.$$

Then the market $\lbrace X(t)\rbrace_{t \in [0,T]}$ has no arbitrage.

**b)** (Karatzas (1996), Th. 0.2.4) Conversely, if the market $\lbrace X(t)\rbrace_{t \in [0,T]}$ has no arbitrage, then there exists an $\mathcal{F}_t^{(m)}$-adapted, $(t,\omega)$-measurable process $u(t,\omega)$ such that

$$\sigma(t,\omega)u(t,\omega) = \mu(t,\omega) - \rho(t,\omega)\widehat{X}(t,\omega)$$

for a.a. $(t,\omega)$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 12.1.8</summary>

**a)** We may assume that $\lbrace X(t)\rbrace$ is normalized, i.e. that $\rho = 0$. Define the measure $Q = Q_u$ on $\mathcal{F}_T^{(m)}$ by

$$dQ(\omega) = \exp\!\left(-\int_0^T u(t,\omega)dB(t) - \frac{1}{2}\int_0^T u^2(t,\omega)dt\right)dP(\omega).$$

Then $Q \sim P$ and by the Girsanov theorem II the process $\widetilde{B}(t) := \int_0^t u(s,\omega)ds + B(t)$ is a $Q$-Brownian motion and in terms of $\widetilde{B}(t)$ we have $dX_i(t) = \mu_i dt + \sigma_i dB(t) = \sigma_i d\widetilde{B}(t)$; $1 \le i \le n$. Hence $X(t)$ is a local $Q$-martingale and the conclusion follows from Lemma 12.1.6.

**b)** Conversely, assume that the market has no arbitrage and is normalized. For $t \in [0,T]$, $\omega \in \Omega$ let $F_t = \lbrace \omega;\, \text{the equation } \sigma u = \mu \text{ has no solution}\rbrace = \lbrace \omega;\, \exists v \text{ with } \sigma^T v = 0 \text{ and } v \cdot \mu \ne 0\rbrace$. Define $\theta_i(t,\omega) = \begin{cases} \text{sign}(v \cdot \mu)v_i & \text{for } \omega \in F_t \\\\ 0 & \text{for } \omega \notin F_t\end{cases}$ for $1 \le i \le n$ and $\theta_0$ according to (12.1.17). Then $\theta(t,\omega)$ is self-financing and generates the value function $V^\theta(t,\omega) - V^\theta(0) = \int_0^t \mathcal{X}_{F_s}(\omega)\lvert v(s,\omega) \cdot \mu(s,\omega)\rvert ds \ge 0$ for all $t \in [0,T]$. Since the market has no arbitrage we must have $\mathcal{X}_{F_t}(\omega) = 0$ for a.a. $(t,\omega)$, i.e. that $\sigma u = \mu$ has a solution for a.a. $(t,\omega)$. $\square$

</details>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(12.1.9 — Detecting Arbitrage)</span></p>

**a)** Consider the price process $X(t)$ given by $dX_0(t) = 0$, $dX_1(t) = 2dt + dB_1(t)$, $dX_2(t) = -dt + dB_1(t) + dB_2(t)$. In this case we have $\mu = \begin{bmatrix} 2 \\\\ -1\end{bmatrix}$, $\sigma = \begin{bmatrix} 1 & 0 \\\\ 1 & 1\end{bmatrix}$ and the system $\sigma u = \mu$ has the unique solution $u = \begin{bmatrix} 2 \\\\ -3\end{bmatrix}$. From Theorem 12.1.8a) we conclude that $X(t)$ has no arbitrage.

**b)** Next, consider the price process $Y(t)$ given by $dY_0(t) = 0$, $dY_1(t) = 2dt + dB_1(t) + dB_2(t)$, $dY_2(t) = -dt - dB_1(t) - dB_2(t)$. Here the system of equations $\sigma u = \mu$ gets the form $\begin{bmatrix} 1 & 1 \\\\ -1 & -1\end{bmatrix}\begin{bmatrix} u_1 \\\\ u_2\end{bmatrix} = \begin{bmatrix} 2 \\\\ -1\end{bmatrix}$ which has no solutions. So the market has an arbitrage, according to Theorem 12.1.8 b). Indeed, if we choose $\theta(t) = (\theta_0, 1, 1)$ we get $V^\theta(T) = V^\theta(0) + \int_0^T 2dt + dB_1(t) + dB_2(t) - dt - dB_1(t) - dB_2(t) = V^\theta(0) + T$. In particular, if we choose $\theta_0$ constant such that $V^\theta(0) = \theta_0 Y_0(0) + Y_1(0) + Y_2(0) = 0$, then $\theta$ will be an arbitrage.

</div>

### 12.2 Attainability and Completeness

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(12.2.1 — Martingale Representation under Change of Measure)</span></p>

Suppose a process $u(t,\omega) \in \mathcal{V}^m(0,T)$ satisfies the condition

$$E\!\left[\exp\!\left(\tfrac{1}{2}\int_0^T u^2(s,\omega)ds\right)\right] < \infty.$$

Define the measure $Q = Q_u$ on $\mathcal{F}_T^{(m)}$ by

$$dQ(\omega) = \exp\!\left(-\int_0^T u(t,\omega)dB(t) - \frac{1}{2}\int_0^T u^2(t,\omega)dt\right)dP(\omega).$$

Then $\widetilde{B}(t) := \int_0^t u(s,\omega)ds + B(t)$ is an $\mathcal{F}_t^{(m)}$-martingale (and hence an $\mathcal{F}_t^{(m)}$-Brownian motion) w.r.t. $Q$ and any $F \in L^2(\mathcal{F}_T^{(m)}, Q)$ has a unique representation

$$F(\omega) = E_Q[F] + \int_0^T \phi(t,\omega)d\widetilde{B}(t)$$

where $\phi(t,\omega)$ is an $\mathcal{F}_t^{(m)}$-adapted, $(t,\omega)$-measurable $\mathbf{R}^m$-valued process such that $E_Q\!\left[\int_0^T \phi^2(t,\omega)dt\right] < \infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(12.2.2 — Admissibility in Original and Normalized Markets)</span></p>

Let $\overline{X}(t) = \xi(t)X(t)$ be the normalized price process. Suppose $\theta(t)$ is an admissible portfolio for the market $\lbrace X(t)\rbrace$ with value process $V^\theta(t) = \theta(t) \cdot X(t)$. Then $\theta(t)$ is also an admissible portfolio for the normalized market $\lbrace \overline{X}(t)\rbrace$ with value process

$$\overline{V}^\theta(t) := \theta(t) \cdot \overline{X}(t) = \xi(t)V^\theta(t)$$

and vice versa. In other words,

$$V^\theta(t) = V^\theta(0) + \int_0^t \theta(s)dX(s); \quad 0 \le t \le T \iff \xi(t)V^\theta(t) = V^\theta(0) + \int_0^t \theta(s)d\overline{X}(s); \quad 0 \le t \le T.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(12.2.3 — Normalized Market Representation)</span></p>

Suppose there exists an $m$-dimensional process $u(t,\omega) \in \mathcal{V}^m(0,T)$ such that, with $\widehat{X}(t,\omega) = (X_1(t,\omega), \dots, X_n(t,\omega))$,

$$\sigma(t,\omega)u(t,\omega) = \mu(t,\omega) - \rho(t,\omega)\widehat{X}(t,\omega) \quad \text{for a.a. } (t,\omega)$$

and $E\!\left[\exp\!\left(\frac{1}{2}\int_0^T u^2(s,\omega)ds\right)\right] < \infty$.

Define the measure $Q = Q_u$ and the process $\widetilde{B}(t)$ as in (12.2.2), (12.2.3), respectively. Then $\widetilde{B}$ is a Brownian motion w.r.t. $Q$ and in terms of $\widetilde{B}$ we have the following representation of the normalized market $\overline{X}(t) = \xi(t)X(t)$:

$$d\overline{X}_0(t) = 0$$

$$d\overline{X}_i(t) = \xi(t)\sigma_i(t)d\widetilde{B}(t); \quad 1 \le i \le n.$$

In particular, if $\int_0^T E_Q[\xi^2(t)\sigma_i^2(t)]dt < \infty$, then $Q$ is an equivalent martingale measure. In any case the normalized value process $\overline{V}^\theta(t)$ of an admissible portfolio $\theta$ is a local $Q$-martingale given by

$$d\overline{V}^\theta(t) = \xi(t)\sum_{i=1}^n \theta_i(t)\sigma_i(t)d\widetilde{B}(t).$$

</div>

**Note.** From now on we assume that there exists a process $u(t,\omega) \in \mathcal{V}^m(0,T)$ satisfying the conditions of Lemma 12.2.3 and we let $Q$ and $\widetilde{B}$ be as described there.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(12.2.4 — Contingent T-claim, Attainability, Completeness)</span></p>

**a)** A (European) contingent **$T$-claim** (or just a $T$-claim or claim) is a lower bounded $\mathcal{F}_T^{(m)}$-measurable random variable $F(\omega)$.

**b)** We say that the claim $F(\omega)$ is **attainable** (in the market $\lbrace X(t)\rbrace_{t \in [0,T]}$) if there exists an admissible portfolio $\theta(t)$ and a real number $z$ such that

$$F(\omega) = V_z^\theta(T) := z + \int_0^T \theta(t)dX(t) \quad \text{a.s.}$$

and such that $\overline{V}^\theta(t) = z + \int_0^t \xi(s)\sum_{i=1}^n \theta_i(s)\sigma_i(s)d\widetilde{B}(s)$; $0 \le t \le T$ is a $Q$-martingale.

If such a $\theta(t)$ exists, we call it a **replicating** or **hedging** portfolio for $F$.

**c)** The market $\lbrace X(t)\rbrace_{t \in [0,T]}$ is called **complete** if every bounded $T$-claim is attainable.

</div>

In other words, a claim $F(\omega)$ is attainable if there exists a real number $z$ such that if we start with $z$ as our initial fortune we can find an admissible portfolio $\theta(t)$ which generates a value $V_z^\theta(T)$ at time $T$ which a.s. equals $F$. In addition we require that the corresponding normalized value process $\overline{V}^\theta(t)$ is a *martingale* and not just a *local* martingale w.r.t. $Q$.

We also note the useful formula: since $\xi(t)V_z^\theta(t) = z + \int_0^t \theta(s)d\overline{X}(s) = z + \int_0^t \phi(s)d\widetilde{B}(s)$, we get

$$\xi(t)V_z^\theta(t) = E_Q[\xi(T)V_z^\theta(T)\,|\,\mathcal{F}_t] = E_Q[\xi(T)F\,|\,\mathcal{F}_t].$$

In particular, $V_z^\theta(t)$ is lower bounded. Hence the market $\lbrace X(t)\rbrace$ is complete.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(12.2.5 — Completeness Criterion)</span></p>

The market $\lbrace X(t)\rbrace$ is complete if and only if $\sigma(t,\omega)$ has a left inverse $\Lambda(t,\omega)$ for a.a. $(t,\omega)$, i.e. there exists an $\mathcal{F}_t^{(m)}$-adapted matrix valued process $\Lambda(t,\omega) \in \mathbf{R}^{m \times n}$ such that

$$\Lambda(t,\omega)\sigma(t,\omega) = I_m \quad \text{for a.a. } (t,\omega).$$

Note that this is equivalent to the property $\operatorname{rank}\,\sigma(t,\omega) = m$ for a.a. $(t,\omega)$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 12.2.5</summary>

**(i)** Assume that $\Lambda(t,\omega)\sigma(t,\omega) = I_m$ holds. Let $Q$ and $\widetilde{B}$ be as in (12.2.2), (12.2.3). Let $F$ be a bounded $T$-claim. We want to prove that there exists an admissible portfolio $\theta(t) = (\theta_0(t), \dots, \theta_n(t))$ and a real number $z$ such that $V_z^\theta(T) = z + \int_0^T \theta(s)dX(s) = F(\omega)$ a.s., then $\overline{V}_z^\theta(t)$ is a $Q$-martingale and $V_z^\theta(T) = F(\omega)$ a.s. By (12.2.16) this is equivalent to $\xi(T)F(\omega) = \overline{V}^\theta(T) = z + \int_0^T \xi(t)\sum_{i=1}^n \theta_i(t)\sigma_i(t)d\widetilde{B}(t)$.

By Lemma 12.2.1 we have a unique representation $\xi(T)F(\omega) = E_Q[\xi(T)F] + \int_0^T \phi(t,\omega)d\widetilde{B}(t) = E_Q[\xi(T)F] + \int_0^T \sum_{j=1}^m \phi_j(t,\omega)d\widetilde{B}_j(t)$ for some $\phi(t,\omega) = (\phi_1, \dots, \phi_m) \in \mathbf{R}^m$. Hence we put $z = E_Q[\xi(T)F]$ and choose $\widehat{\theta}(t) = (\theta_1(t), \dots, \theta_n(t))$ such that $\xi(t)\widehat{\theta}(t)\sigma(t) = \phi(t)$, i.e. such that $\xi(t)\widehat{\theta}(t)\sigma(t) = \phi(t)$. By the left inverse property this equation has the solution $\widehat{\theta}(t,\omega) = X_0(t)\phi(t,\omega)\Lambda(t,\omega)$. By choosing $\theta_0$ according to (12.1.16) the portfolio becomes self-financing. Moreover, $\xi(t)V_z^\theta(t) = E_Q[\xi(T)F\,|\,\mathcal{F}_t]$. In particular, $V_z^\theta(t)$ is lower bounded. Hence the market $\lbrace X(t)\rbrace$ is complete.

**(ii)** Conversely, assume that $\lbrace X(t)\rbrace$ is complete. Then $\lbrace \overline{X}(t)\rbrace$ is complete, so we may assume that $\rho = 0$. The calculation in part a) shows that the value process $V_z^\theta(t)$ generated by an admissible portfolio $\theta(t) = (\theta_0(t), \theta_1(t), \dots, \theta_n(t))$ is $V_z^\theta(t) = z + \int_0^t \widehat{\theta}\sigma\,d\widetilde{B}$ where $\widehat{\theta}(t) = (\theta_1(t), \dots, \theta_n(t))$. Since $\lbrace X(t)\rbrace$ is complete we can hedge any bounded $T$-claim. Choose an $\mathcal{F}_t^{(m)}$-adapted process $\phi(t,\omega) \in \mathbf{R}^m$ such that $E_Q[\int_0^T \phi^2 dt] < \infty$ and define $F(\omega) := \int_0^T \phi(t,\omega)d\widetilde{B}(t)$. Then $E_Q[F^2] < \infty$ so we can find a sequence of bounded $T$-claims $F_k(\omega)$ such that $F_k \to F$ in $L^2(Q)$ and $E_Q[F_k] = 0$. By completeness there exists for all $k$ an admissible portfolio $\theta^{(k)} = (\theta_0^{(k)}, \widehat{\theta}^{(k)})$ such that $V^{\theta^{(k)}}(t) = \int_0^t \widehat{\theta}^{(k)}\sigma\,d\widetilde{B}$ is a $Q$-martingale and $F_k(\omega) = V^{\theta^{(k)}}(T) = \int_0^T \widehat{\theta}^{(k)}\sigma\,d\widetilde{B}$. Then by the Itô isometry the sequence $\lbrace \widehat{\theta}^{(k)}\sigma\rbrace_{k=1}^\infty$ is a Cauchy sequence in $L^2(\lambda \times Q)$. Hence there exists $\psi(t,\omega) \in L^2(\lambda \times Q)$ such that $\widehat{\theta}^{(k)}\sigma \to \psi$ in $L^2(\lambda \times Q)$. But then $\int_0^t \psi\,d\widetilde{B} = \lim_{k \to \infty}\int_0^t \widehat{\theta}^{(k)}\sigma\,d\widetilde{B} = \lim_{k \to \infty} E[F_k \,|\, \widetilde{\mathcal{F}}_t^{(m)}] = E[F \,|\, \widetilde{\mathcal{F}}_t^{(m)}] = \int_0^t \phi\,d\widetilde{B}$ a.s. for all $t \in [0,T]$. Hence by uniqueness we have $\phi(t,\omega) = \psi(t,\omega)$ for a.a. $(t,\omega)$. By taking a subsequence we obtain that for a.a. $(t,\omega)$ there exists a sequence $x^{(k)}(t,\omega) \in \mathbf{R}^m$ such that $x^{(k)}(t,\omega)\sigma(t,\omega) \to \phi(t,\omega)$ as $k \to \infty$. This implies that $\phi(t,\omega)$ belongs to the linear span of the rows $\lbrace \sigma_i(t,\omega)\rbrace_{i=1}^n$ of $\sigma(t,\omega)$. Since $\phi \in L^2(\lambda \times Q)$ was arbitrary, we conclude that the linear span of $\lbrace \sigma_i(t,\omega)\rbrace_{i=1}^n$ is the whole of $\mathbf{R}^m$ for a.a. $(t,\omega)$. So $\operatorname{rank}\,\sigma(t,\omega) = m$ and there exists $\Lambda(t,\omega) \in \mathbf{R}^{m \times n}$ such that $\Lambda(t,\omega)\sigma(t,\omega) = I_m$. $\square$

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(12.2.6 — Consequences of Completeness)</span></p>

**(a)** If $n = m$ then the market is complete if and only if $\sigma(t,\omega)$ is invertible for a.a. $(t,\omega)$.

**(b)** If the market is complete, then $\operatorname{rank}\,\sigma(t,\omega) = m$ for a.a. $(t,\omega)$. In particular, $n \ge m$. Moreover, the process $u(t,\omega)$ satisfying $\sigma u = \mu - \rho\widehat{X}$ is unique.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(12.2.7 — A Complete Market with $n > m$)</span></p>

Define $X_0(t) \equiv 1$ and $\begin{bmatrix} dX_1(t) \\\\ dX_2(t) \\\\ dX_3(t)\end{bmatrix} = \begin{bmatrix} 1 \\\\ 2 \\\\ 3\end{bmatrix}dt + \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 1 & 1\end{bmatrix}\begin{bmatrix} dB_1(t) \\\\ dB_2(t)\end{bmatrix}$. Then $\rho = 0$ and the equation $\sigma u = \mu$ gets the form $\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 1 & 1\end{bmatrix}\begin{bmatrix} u_1 \\\\ u_2\end{bmatrix} = \begin{bmatrix} 1 \\\\ 2 \\\\ 3\end{bmatrix}$ which has the unique solution $u_1 = 1$, $u_2 = 2$. Since $u$ is constant, the Novikov condition holds, and $\operatorname{rank}\,\sigma = 2$, so the market is complete by Theorem 12.2.5. The left inverse is $\Lambda = \begin{bmatrix} 1 & 0 & 0 \\\\ 0 & 1 & 0\end{bmatrix}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(12.2.8 — An Incomplete Market)</span></p>

Let $X_0(t) \equiv 1$ and $dX_1(t) = 2dt + dB_1(t) + dB_2(t)$. Then $\mu = 2$, $\sigma = (1, 1) \in \mathbf{R}^{1 \times 2}$, so $n = 1 < 2 = m$. Hence this market cannot be complete, by Corollary 12.2.6. So there exist bounded $T$-claims which cannot be hedged.

Can we find such a $T$-claim? Let $\theta(t) = (\theta_0(t), \theta_1(t))$ be an admissible portfolio. Then the corresponding value process $V_z^\theta(t)$ is given by $V_z^\theta(t) = z + \int_0^t \theta_1(s)(d\widetilde{B}_1(s) + d\widetilde{B}(s))$. So if $\theta$ hedges a $T$-claim $F(\omega)$ we have $F(\omega) = z + \int_0^T \theta_1(s)(d\widetilde{B}_1(s) + d\widetilde{B}_2(s))$.

Choose $F(\omega) = g(\widetilde{B}_1(T))$, where $g\colon \mathbf{R} \to \mathbf{R}$ is bounded. Then by the Itô representation theorem applied to the 2-dimensional Brownian motion $\widetilde{B}(t) = (\widetilde{B}_1(t), \widetilde{B}_2(t))$ there is a unique $\phi(t,\omega) = (\phi_1(t,\omega), \phi_2(t,\omega))$ such that $g(\widetilde{B}_1(T)) = E_Q[g(\widetilde{B}_1(T))] + \int_0^T \phi_1(s)d\widetilde{B}_1(s) + \phi_2(s)d\widetilde{B}_2(s)$. By the Itô representation theorem applied to $\widetilde{B}_1(t)$ alone, we must have $\phi_2 = 0$, i.e. $g(\widetilde{B}_1(T)) = E_Q[g(\widetilde{B}_1(T))] + \int_0^T \phi_1(s)d\widetilde{B}_1(s)$. Comparing with the hedging equation, we see that no such $\theta_1$ exists. So $F(\omega) = g(\widetilde{B}_1(T))$ cannot be hedged.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Completeness via Uniqueness of Equivalent Martingale Measure)</span></p>

There is a striking characterization of completeness in terms of equivalent martingale measures, due to Harrison and Pliska (1983) and Jacod (1979):

A market $\lbrace X(t)\rbrace$ is complete if and only if there is *one and only one* equivalent martingale measure for the normalized market $\lbrace \overline{X}(t)\rbrace$.

</div>

### 12.3 Option Pricing

#### European Options

Let $F(\omega)$ be a $T$-claim. A **European option** on the claim $F$ is a guarantee to be paid the amount $F(\omega)$ at time $t = T > 0$. How much would you be willing to pay at time $t = 0$ for such a guarantee?

**Buyer's perspective:** If I — the buyer of the option — pay the price $y$ for this guarantee, then I have an initial fortune $-y$ in my investment strategy. With this initial fortune (debt) it must be possible to hedge to time $T$ a value $V_{-y}^\theta(T,\omega)$ which, if the guaranteed payoff $F(\omega)$ is added, gives me a nonnegative result: $V_{-y}^\theta(T,\omega) + F(\omega) \ge 0$ a.s. Thus the maximal price $p = p(F)$ the buyer is willing to pay is

$$p(F) = \sup\lbrace y;\; \text{There exists an admissible portfolio } \theta \text{ such that } V_{-y}^\theta(T,\omega) := -y + \int_0^T \theta(s)dX(s) \ge -F(\omega) \text{ a.s.}\rbrace$$

**Seller's perspective:** If I — the seller — receive the price $z$ for this guarantee, then I can use this as the initial value in an investment strategy. With this initial fortune it must be possible to hedge to time $T$ a value $V_z^\theta(T,\omega)$ which is not less than the amount $F(\omega)$ that I have promised to pay to the buyer: $V_z^\theta(T,\omega) \ge F(\omega)$ a.s. Thus the minimal price $q = q(F)$ the seller is willing to accept is

$$q(F) = \inf\lbrace z;\; \text{There exists an admissible portfolio } \theta \text{ such that } V_z^\theta(T,\omega) := z + \int_0^T \theta(s)dX(s) \ge F(\omega) \text{ a.s.}\rbrace$$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(12.3.1 — Price of a European Contingent Claim)</span></p>

If $p(F) = q(F)$ we call this common value the **price** (at $t = 0$) of the (European) $T$-contingent claim $F(\omega)$.

Two important examples of European contingent claims are:

**a) The European call**, where $F(\omega) = (X_i(T,\omega) - K)^+$ for some $i \in \lbrace 1, 2, \dots, n\rbrace$ and some $K > 0$. This option gives the owner the right (but not the obligation) to *buy* one unit of security number $i$ at the specified price $K$ (the *exercise* price) at time $T$.

**b) The European put** option gives the owner the right (but not the obligation) to *sell* one unit of security number $i$ at a specified price $K$ at time $T$. This option gives the owner the payoff $F(\omega) = (K - X_i(T,\omega))^+$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(12.3.2 — Pricing Bounds and Complete Market Pricing)</span></p>

**a)** Suppose (12.2.12) and (12.2.13) hold and let $Q$ be as in (12.2.2). Let $F$ be a (European) $T$-claim such that $E_Q[\xi(T)F] < \infty$. Then

$$\operatorname{ess\,inf} F(\omega) \le p(F) \le E_Q[\xi(T)F] \le q(F) \le \infty.$$

**b)** Suppose, in addition to the conditions in a), that the market $\lbrace X(t)\rbrace$ is complete. Then the price of the (European) $T$-claim $F$ is

$$p(F) = E_Q[\xi(T)F] = q(F).$$

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 12.3.2</summary>

**a)** Suppose $y \in \mathbf{R}$ and there exists an admissible portfolio $\theta$ such that $V_{-y}^\theta(T,\omega) = -y + \int_0^T \theta(s)dX(s) \ge -F(\omega)$ a.s. i.e., using (12.2.7) and Lemma 12.2.4, $-y + \int_0^T \sum_{i=1}^n \theta_i(s)\xi(s)\sigma_i(s)d\widetilde{B}(s) \ge -\xi(T)F(\omega)$ a.s. Since $\int_0^t \sum_{i=1}^n \theta_i(s)\xi(s)\sigma_i(s)d\widetilde{B}(s)$ is a lower bounded local $Q$-martingale, it is a supermartingale. Hence $E_Q[\int_0^t \sum_{i=1}^n \theta_i \xi\sigma_i d\widetilde{B}] \le 0$. Therefore, taking $Q$-expectation: $y \le E_Q[\xi(T)F]$. Hence $p(F) \le E_Q[\xi(T)F]$, provided such a portfolio $\theta$ exists for some $y \in \mathbf{R}$. The first inequality in (12.3.3) holds because clearly, if $y < F(\omega)$ for a.a. $\omega$, we can choose $\theta = 0$. Hence the first inequality holds.

Similarly, if there exists $z \in \mathbf{R}$ and an admissible portfolio $\theta$ such that $z + \int_0^T \theta(s)dX(s) \ge F(\omega)$ a.s., then $z + \int_0^T \sum_{i=1}^n \theta_i \xi\sigma_i d\widetilde{B} \ge \xi(T)F(\omega)$ a.s. Taking $Q$-expectations: $z \ge E_Q[\xi(T)F]$. If no such $z, \theta$ exist, then $q(F) = \infty > E_Q[\xi(T)F]$.

**b)** Assume also that the market is complete. Define $F_k(\omega) = \begin{cases} k & \text{if } F(\omega) \ge k \\\\ F(\omega) & \text{if } F(\omega) < k\end{cases}$. Then $F_k$ is a bounded $T$-claim, so by completeness we can find (unique) $y_k \in \mathbf{R}$ and $\theta^{(k)}$ such that $-y_k + \int_0^T \theta^{(k)}(s)dX(s) = -F_k(\omega)$ a.s. Since $\int_0^t \sum_{i=1}^n \theta_i^{(k)}\xi\sigma_i d\widetilde{B}$ is a $Q$-martingale, $y_k = E_Q[\xi(T)F_k]$. Hence $p(F) \ge p(F_k) \ge E_Q[\xi(T)F_k] \to E_Q[\xi(T)F]$ as $k \to \infty$, by monotone convergence. Combined with a) this gives $p(F) = E_Q[\xi(T)F]$. A similar argument gives $q(F) = E_Q[\xi(T)F]$. $\square$

</details>

#### How to Hedge an Attainable Claim

We have seen that if $V_z^\theta(t)$ is the value process of an admissible portfolio $\theta(t)$ for the market $\lbrace X(t)\rbrace$, then $\overline{V}_z^\theta(t) := \xi(t)V_z^\theta(t)$ is the value process of $\theta(t)$ for the normalized market $\lbrace \overline{X}(t)\rbrace$ (Lemma 12.2.3). Hence we have

$$\xi(t)V_z^\theta(t) = z + \int_0^t \theta(s)d\overline{X}(s).$$

If (12.2.12) and (12.2.13) hold, then — if $Q$, $\widetilde{B}$ are defined as before — we can rewrite this as

$$\xi(t)V_z^\theta(t) = z + \int_0^t \sum_{i=1}^n \theta_i(s)\xi(s)\sum_{j=1}^m \sigma_{ij}(s)d\widetilde{B}_j(s).$$

Therefore, the portfolio $\theta(t) = (\theta_0(t), \dots, \theta_n(t))$ needed to hedge a given $T$-claim $F$ is given by

$$\xi(t,\omega)(\theta_1(t), \dots, \theta_n(t))\sigma(t,\omega) = \phi(t,\omega)$$

where $\phi(t,\omega) \in \mathbf{R}^m$ is such that $\xi(T)F(\omega) = E_Q[\xi(T)F] + \int_0^T \phi(t,\omega)d\widetilde{B}(t)$ (and $\theta_0(t)$ is given by (12.1.14)).

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(12.3.3 — Explicit Hedging Integrand)</span></p>

Let $Y(t)$ and $Z(t)$ be as in (12.3.10) and (12.3.12), respectively, and assume that $h\colon \mathbf{R}^n \to \mathbf{R}$ is as in (12.3.13). Assume that (12.3.11) and (12.3.16) hold and define $Q$ and $\widetilde{B}(t)$ by (12.3.17) and (12.3.18). Then

$$h(Y(T)) = E_Q^y[h(Y(T))] + \int_0^T \phi(t,\omega)d\widetilde{B}(t)$$

where $\phi = (\phi_1, \dots, \phi_m)$, with

$$\phi_j(t,\omega) = \sum_{i=1}^n \frac{\partial}{\partial y_i}(E^y[h(Z(T-t))])\big\rvert_{y = Y(t)} \sigma_{ij}(Y(t)); \quad 1 \le j \le m.$$

In particular, if $\rho = b = 0$ and $\sigma = I_m$ then $u = 0$, $P = Q$ and $Y(t) = Z(t) = B(t)$. Hence we get the representation

$$h(B(T)) = E^y[h(B(T))] + \int_0^T \sum_{j=1}^m \frac{\partial}{\partial z_j} E^z[h(B(T-t))]\big\rvert_{z = B(t)} dB_j(t).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(12.3.4 — Pricing and Hedging Summary)</span></p>

Let $\lbrace X(t)\rbrace_{t \in [0,T]}$ be a complete market. Suppose (12.2.12) and (12.2.13) hold and let $Q, \widetilde{B}$ be as in (12.2.2), (12.2.3). Let $F$ be a European $T$-claim such that $E_Q[\xi(T)F] < \infty$. Then the price of the claim $F$ is

$$p(F) = E_Q[\xi(T)F].$$

Moreover, to find a replicating (hedging) portfolio $\theta(t) = (\theta_0(t), \dots, \theta_n(t))$ for the claim $F$ we first find (for example by using Theorem 12.3.3 if possible) $\phi \in \mathcal{W}^m$ such that

$$\xi(T)F = E_Q[\xi(T)F] + \int_0^T \phi(t,\omega)d\widetilde{B}(t).$$

Then we choose $\widehat{\theta}(t) = (\theta_1(t), \dots, \theta_n(t))$ such that $\widehat{\theta}(t,\omega)\xi(t,\omega)\sigma(t,\omega) = \phi(t,\omega)$, which has the solution

$$\widehat{\theta}(t,\omega) = X_0(t)\phi(t,\omega)\Lambda(t,\omega)$$

where $\Lambda(t,\omega)$ is the left inverse of $\sigma(t,\omega)$ (Theorem 12.2.5), and we choose $\theta_0(t)$ as in (12.1.14).

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(12.3.5 — Hedging $\exp(Y(T))$ in an Ornstein-Uhlenbeck Market)</span></p>

Suppose the market is $X_0(t) = e^{\rho t}$, $X_1(t) = Y(t)$, where $\rho > 0$ is constant and $Y(t)$ is an Ornstein-Uhlenbeck process $dY(t) = \alpha Y(t)dt + \sigma dB(t)$; $Y(0) = y$ where $\alpha, \sigma$ are constants, $\sigma \ne 0$. How do we hedge the claim $F(\omega) = \exp(Y(T))$?

The portfolio $\theta(t) = (\theta_0(t), \theta_1(t))$ that we seek is given by (12.3.29), i.e. $\theta_1(t,\omega) = e^{\rho t}\sigma^{-1}\phi(t,\omega)$ where $\phi(t,\omega)$ and $V(0)$ are uniquely given by (12.3.9). To find $\phi(t,\omega)$ explicitly we apply Theorem 12.3.3: In this case we choose $h(y) = \exp(y)e^{-\rho T}$ and $dZ(t) = \rho Z(t)dt + dB(t)$. Then $Z(t) = Z(0)e^{\rho t} + \sigma\int_0^t e^{\rho(t-s)}dB(s)$.

Hence $e^{\rho T}E_Q^y[h(Y(T-t))] = E^y[\exp(ye^{\rho(T-t)} + \sigma\int_0^{T-t} e^{\rho(T-t-s)}dB(s))] = \exp\!\left(ye^{\rho(T-t)} + \frac{\sigma^2}{4\rho}(e^{2\rho(T-t)} - 1)\right)$ if $\rho \ne 0$.

This gives $\phi(t,\omega) = \frac{\partial}{\partial y}\exp\!\left(ye^{\rho(T-t)} + \frac{\sigma^2}{4\rho}(e^{2\rho(T-t)} - 1)\right)\big\rvert_{y = Y(t)} \cdot \sigma e^{-\rho T}$ and hence, if $\rho \ne 0$,

$$\theta_1(t) = \exp\!\left\lbrace Y(t)e^{\rho(T-t)} + \frac{\sigma^2}{4\rho}(e^{2\rho(T-t)} - 1)\right\rbrace.$$

If $\rho = 0$ then $\theta_1(t) = \exp\!\left\lbrace Y(t) + \frac{\sigma^2}{2}(T - t)\right\rbrace$.

</div>

#### The Generalized Black & Scholes Model

Let us now specialize to a situation where the market has just two securities $X_0(t), X_1(t)$ where $X_0, X_1$ are Itô processes of the form

$$dX_0(t) = \rho(t,\omega)X_0(t)dt \quad (\text{as before})$$

$$dX_1(t) = \alpha(t,\omega)X_1(t)dt + \beta(t,\omega)X_1(t)dB(t)$$

where $B(t)$ is 1-dimensional and $\alpha(t,\omega), \beta(t,\omega)$ are 1-dimensional processes in $\mathcal{W}$. Note that the solution of the equation for $X_1$ is

$$X_1(t) = X_1(0)\exp\!\left(\int_0^t \beta(s,\omega)dB(s) + \int_0^t (\alpha(s,\omega) - \tfrac{1}{2}\beta^2(s,\omega))ds\right).$$

The equation $\sigma u = \mu - \rho\widehat{X}$ gets the form $X_1(t)\beta(t,\omega)u(t,\omega) = X_1(t)\alpha(t,\omega) - X_1(t)\rho(t,\omega)$ which has the solution

$$u(t,\omega) = \beta^{-1}(t,\omega)[\alpha(t,\omega) - \rho(t,\omega)] \quad \text{if } \beta(t,\omega) \ne 0.$$

So the no-arbitrage condition (12.2.13) holds iff $E\!\left[\exp\!\left(\frac{1}{2}\int_0^T \frac{(\alpha(s,\omega) - \rho(s,\omega))^2}{\beta^2(s,\omega)}ds\right)\right] < \infty$. In this case we have an equivalent martingale measure $Q$ given by (12.2.2) and the market has no arbitrage, by Theorem 12.1.8. Moreover, the market is complete by Corollary 12.2.5. Therefore we get by Theorem 12.3.2 that the price at $t = 0$ of a European option with payoff given by a contingent $T$-claim $F$ is $p(F) = q(F) = E_Q[\xi(T)F]$, provided this quantity is finite.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(12.3.6 — The Generalized Black & Scholes Formula)</span></p>

Suppose $X(t) = (X_0(t), X_1(t))$ is given by

$$dX_0(t) = \rho(t)X_0(t)dt; \quad X_0(0) = 1$$

$$dX_1(t) = \alpha(t,\omega)X_1(t)dt + \beta(t)X_1(t)dB(t); \quad X_1(0) = x_1 > 0$$

where $\rho(t)$, $\beta(t)$ are deterministic and $E\!\left[\exp\!\left(\frac{1}{2}\int_0^T \frac{(\alpha(t,\omega) - \rho(t))^2}{\beta^2(t)}dt\right)\right] < \infty$.

**a)** Then the market $\lbrace X(t)\rbrace$ is complete and the price at time $t = 0$ of the European $T$-claim $F(\omega) = f(X_1(T,\omega))$ where $E_Q[\lvert f(X_1(T,\omega))\rvert] < \infty$ is

$$p = \frac{\xi(T)}{\delta\sqrt{2\pi}}\int_{\mathbf{R}} f\!\left(x_1\exp\!\left[y + \int_0^T (\rho(s) - \tfrac{1}{2}\beta^2(s))ds\right]\right)\exp\!\left(-\frac{y^2}{2\delta^2}\right)dy$$

where $\xi(T) = \exp(-\int_0^T \rho(s)ds)$ and $\delta^2 = \int_0^T \beta^2(s)ds$.

**b)** If $\rho, \alpha, \beta \ne 0$ are constants and $f \in C^1(\mathbf{R})$, then the self-financing portfolio $\theta(t) = (\theta_0(t), \theta_1(t))$ needed to replicate the $T$-claim $F(\omega) = f(X_1(T,\omega))$ is given by

$$\theta_1(t,\omega) = \frac{1}{\sqrt{2\pi(T-t)}}\int_{\mathbf{R}} f'(X_1(t,\omega)\exp\lbrace \beta x + (\rho - \tfrac{1}{2}\beta^2)(T-t)\rbrace) \cdot \exp\!\left(\beta x - \frac{x^2}{2(T-t)} - \tfrac{1}{2}\beta^2(T-t)\right)dx$$

and $\theta_0(t)$ is given by (12.1.14).

</div>

#### American Options

The difference between European and American options is that in the latter case the buyer of the option is free to choose any exercise time $\tau$ before or at the given expiration time $T$ (and the guaranteed payoff may depend on both $\tau$ and $\omega$). This exercise time $\tau$ may be stochastic (depend on $\omega$), but only in such a way that the decision to exercise before or at a time $t$ only depends on the history up to time $t$. More precisely, we require that $\tau$ must be an $\mathcal{F}_t^{(m)}$-stopping time.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(12.3.7 — American Contingent T-claim)</span></p>

An **American contingent $T$-claim** is an $\mathcal{F}_t^{(m)}$-adapted, $(t,\omega)$-measurable and a.s. lower bounded continuous stochastic process $F(t) = F(t,\omega)$; $t \in [0,T]$, $\omega \in \Omega$. An **American option** on such a claim $F(t,\omega)$ gives the owner of the option the right (but not the obligation) to choose any stopping time $\tau(\omega) \le T$ as exercise time for the option, resulting in a payment $F(\tau(\omega), \omega)$ to the owner.

</div>

The pricing argument for American options is analogous to the European case:

**Buyer's price:**

$$p_A(F) = \sup\lbrace y;\; \text{There exists a stopping time } \tau \le T \text{ and an admissible portfolio } \theta \text{ such that } V_{-y}^\theta(\tau(\omega),\omega) := -y + \int_0^{\tau(\omega)} \theta(s)dX(s) \ge -F(\tau(\omega),\omega) \text{ a.s.}\rbrace$$

**Seller's price:**

$$q_A(F) = \inf\lbrace z;\; \text{There exists an admissible portfolio } \theta \text{ such that for all } t \in [0,T] \text{ we have } V_z^\theta(t,\omega) := z + \int_0^t \theta(s)dX(s) \ge F(t,\omega) \text{ a.s.}\rbrace$$

A result analogous to Theorem 12.3.2 can be proved for American options, basically due to Bensoussan (1984) and Karatzas (1988).

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(12.3.8 — Pricing Formula for American Options)</span></p>

**a)** Suppose (12.2.12) and (12.2.13) hold and let $Q$ be as in (12.2.2). Let $F(t) = F(t,\omega)$; $t \in [0,T]$ be an American contingent $T$-claim such that

$$\sup_{\tau \le T} E_Q[\xi(\tau)F(\tau)] < \infty.$$

Then

$$p_A(F) \le \sup_{\tau \le T} E_Q[\xi(\tau)F(\tau)] \le q_A(F) \le \infty.$$

**b)** Suppose, in addition to the conditions in a), that the market $\lbrace X(t)\rbrace$ is complete. Then

$$p_A(F) = \sup_{\tau \le T} E_Q[\xi(\tau)F(\tau)] = q_A(F).$$

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 12.3.8</summary>

**Part a):** Suppose $y \in \mathbf{R}$ and there exists a stopping time $\tau \le T$ and an admissible portfolio $\theta$ such that

$$V_{-y}^\theta(\tau,\omega) = -y + \int_0^\tau \theta(s)dX(s) \ge -F(\tau) \quad \text{a.s.}$$

Then

$$-y + \int_0^\tau \sum_{i=1}^n \theta_i(s)\xi(s)\sigma_i(s)d\widetilde{B}(s) \ge -\xi(\tau)F(\tau) \quad \text{a.s.}$$

Taking expectations with respect to $Q$:

$$y \le E_Q[\xi(\tau)F(\tau)] \le \sup_{\tau \le T} E_Q[\xi(\tau)F(\tau)].$$

Since this holds for all such $y$, we conclude that $p_A(F) \le \sup_{\tau \le T} E_Q[\xi(\tau)F(\tau)]$.

Similarly, suppose $z \in \mathbf{R}$ and there exists an admissible portfolio $\theta$ such that $V_z^\theta(t,\omega) = z + \int_0^t \theta(s)dX(s) \ge F(t,\omega)$ a.s. for all $t \in [0,T]$. Then for any stopping time $\tau \le T$:

$$z \ge \sup_{\tau \le T} E_Q[\xi(\tau)F(\tau)].$$

Since this holds for all such $z$, we get $q_A(F) \ge \sup_{\tau \le T} E_Q[\xi(\tau)F(\tau)]$.

**Part b):** Assume in addition that the market is complete. Choose a stopping time $\tau \le T$. Define

$$F_k(t) = F_k(t,\omega) = \begin{cases} k & \text{if } F(t,\omega) \ge k \\ F(t,\omega) & \text{if } F(t,\omega) < k \end{cases}$$

and put $G_k(\omega) = X_0(T)\xi(\tau)F_k(\tau)$. Then $G_k$ is a bounded $T$-claim, so by completeness we can find $y_k \in \mathbf{R}$ and a portfolio $\theta^{(k)}$ such that

$$-y_k + \int_0^T \theta^{(k)}(s)dX(s) = -G_k(\omega) \quad \text{a.s.}$$

and such that $-y_k + \int_0^t \theta^{(k)}(s)d\overline{X}(s)$ is a $Q$-martingale. Then by (12.2.8)–(12.2.9):

$$-y_k + \int_0^\tau \theta^{(k)}(s)dX(s) = -F_k(\tau) \quad \text{a.s.}$$

and $y_k = E_Q[\xi(\tau)F_k(\tau)]$. This shows that any price of the form $E_Q[\xi(\tau)F_k(\tau)]$ for some stopping time $\tau \le T$ would be acceptable for the buyer. Hence $p_A(F) \ge p_A(F_k) \ge \sup_{\tau \le T} E_Q[\xi(\tau)F_k(\tau)]$.

Letting $k \to \infty$ we obtain by monotone convergence: $p_A(F) \ge \sup_{\tau \le T} E_Q[\xi(\tau)F(\tau)]$.

It remains to show that if we put $z = \sup_{0 \le \tau \le T} E_Q[\xi(\tau)F(\tau)]$, then there exists an admissible portfolio $\theta(s,\omega)$ which **superreplicates** $F(t,\omega)$:

$$z + \int_0^t \theta(s,\omega)dX(s) \ge F(t,\omega) \quad \text{for a.a. } (t,\omega) \in [0,T] \times \Omega.$$

Define the **Snell envelope**:

$$S(t) = \sup_{t \le \tau \le T} E_Q[\xi(\tau)F(\tau) \mid \mathcal{F}_t^{(m)}]; \quad 0 \le t \le T.$$

Then $S(t)$ is a **supermartingale** w.r.t. $Q$ and $\lbrace\mathcal{F}_t^{(m)}\rbrace$, so by the **Doob-Meyer decomposition** we can write

$$S(t) = M(t) - A(t); \quad 0 \le t \le T$$

where $M(t)$ is a $Q, \lbrace\mathcal{F}_t^{(m)}\rbrace$-martingale with $M(0) = S(0) = z$ and $A(t)$ is a nondecreasing process with $A(0) = 0$. By Lemma 12.2.1 we can represent the martingale $M$ as an Itô integral w.r.t. $\widetilde{B}$. Hence

$$z + \int_0^t \phi(s,\omega)d\widetilde{B}(s) = M(t) = S(t) + A(t) \ge S(t); \quad 0 \le t \le T$$

for some $\mathcal{F}_t^{(m)}$-adapted process $\phi(s,\omega)$. Since the market is complete, $\sigma(t,\omega)$ has a left inverse $\Lambda(t,\omega)$. Defining $\widehat{\theta} = (\theta_1, \ldots, \theta_n)$ by $\widehat{\theta}(t,\omega) = X_0(t)\phi(t,\omega)\Lambda(t,\omega)$, we get by Lemma 12.2.3:

$$z + \int_0^t \theta(s,\omega)dX(s) \ge X_0(t)S(t) \ge X_0(t)\xi(t)F(t) = F(t); \quad 0 \le t \le T.$$

</details>

#### The Itô Diffusion Case: Connection to Optimal Stopping

Theorem 12.3.8 shows that pricing an American option is an **optimal stopping problem**. In the general case, the solution can be expressed in terms of the Snell envelope. In the Itô diffusion case, we get an optimal stopping problem of the type discussed in Chapter 10.

Assume the market is an $(n+1)$-dimensional Itô diffusion $X(t) = (X_0(t), X_1(t), \ldots, X_n(t))$; $t \ge 0$ of the form

$$dX_0(t) = \rho(t, X(t))X_0(t)dt; \quad X_0(0) = 1$$

$$dX_i(t) = \mu_i(t, X(t))dt + \sum_{j=1}^m \sigma_{ij}(t, X(t))dB_j(t) = \mu_i(t, X(t))dt + \sigma_i(t, X(t))dB(t); \quad X_i(0) = x_i$$

where $\rho$, $\mu_i$ and $\sigma_{ij}$ are given functions satisfying the conditions of Theorem 5.2.1. Further, assume that the no-arbitrage conditions (12.2.12)–(12.2.13) are satisfied, i.e. there exists $u(t,x) \in \mathbf{R}^{m \times 1}$ such that for all $t$, $x = (x_0, x_1, \ldots, x_n)$:

$$\sigma_i(t,x)u(t,x) = \mu_i(t,x) - \rho(t,x)x_i \quad \text{for } i = 1, \ldots, n$$

and $E^x\!\left[\exp\!\left(\frac{1}{2}\int_0^T u^2(t,X(t))dt\right)\right] < \infty$ for all $x$.

Put

$$M(t) = M(t,\omega) = \exp\!\left(-\int_0^t u(s,X(s))dB(s) - \frac{1}{2}\int_0^t u^2(s,X(s))ds\right)$$

and define the probability measure $Q$ on $\mathcal{F}_T^{(m)}$ by $dQ(\omega) = M(T,\omega)dP(\omega)$.

Now assume that $F(t,\omega)$ is an American contingent $T$-claim of **Markovian type**, i.e. $F(t,\omega) = g(t, X(t,\omega))$ for some continuous, lower bounded function $g\colon \mathbf{R} \times \mathbf{R}^{n+1} \to \mathbf{R}$. Then if the market $\lbrace X(t)\rbrace_{t \in [0,T]}$ is complete, the price $p_A(F)$ of this claim is by Theorem 12.3.8:

$$p_A(F) = \sup_{\tau \le T} E_Q[\xi(\tau)g(\tau, X(\tau))] = \sup_{\tau \le T} E[M(\tau)\xi(\tau)g(\tau, X(\tau))]$$

Define $K(t) = M(t)\xi(t) = \exp\!\left(-\int_0^t u(s,X(s))dB(s) - \int_0^t \left[\frac{1}{2}u^2(s,X(s)) + \rho(s,X(s))\right]ds\right)$. Then $dK(t) = -\rho(t,X(t))K(t)dt - u(t,X(t))K(t)dB(t)$.

Hence if we define the $(n+3)$-dimensional Itô diffusion $Y(t)$ by

$$dY(t) = \begin{bmatrix} dt \\ dK(t) \\ dX_0(t) \\ dX_1(t) \\ \vdots \\ dX_n(t) \end{bmatrix} = \begin{bmatrix} 1 \\ -\rho K \\ \rho X_0 \\ \mu_1 \\ \vdots \\ \mu_n \end{bmatrix}dt + \begin{bmatrix} 0 \\ -uK \\ 0 \\ \sigma_1 \\ \vdots \\ \sigma_n \end{bmatrix}dB(t); \quad Y(0) = y$$

we see that $p_A(F) = \sup_{\tau \le T} E[G(Y(\tau))]$, where $G(y) = G(s,k,x) = kg(s,x)$; $y = (s,k,x) \in \mathbf{R} \times \mathbf{R} \times \mathbf{R}^{n+1}$.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(12.3.9 — American Pricing as Optimal Stopping)</span></p>

The price $p_A(F)$ of an American contingent $T$-claim $F$ of the Markovian form $F(t,\omega) = g(t,X(t,\omega))$ is the solution of the optimal stopping problem $p_A(F) = \sup_{\tau \le T} E[G(Y(\tau))]$, with Itô diffusion $Y(t)$ given above.

</div>

We recognize this as a special case of the optimal stopping problem considered in Theorem 10.4.1. We can therefore use the method there to evaluate $p_A(F)$ in special cases.

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(12.3.10 — American Options in the Black-Scholes Market)</span></p>

Consider the Black and Scholes market:

$$dX_0(t) = \rho X_0(t)dt; \quad X_0(0) = 1$$

$$dX_1(t) = \alpha X_1(t)dt + \beta X_1(t)dB(t); \quad X_1(0) = x_1 > 0$$

where $\rho, \alpha, \beta$ are constants, $\beta \ne 0$. Then equation (12.3.52) becomes $\beta x_1 u(x_1) = \alpha x_1 - \rho x_1$, i.e.

$$u(x_1) = u = \frac{\alpha - \rho}{\beta} \quad \text{for all } x_1.$$

Hence $K(t) = \exp\!\left(-\frac{\alpha - \rho}{\beta}B(t) - \left\lbrace\frac{1}{2}\left(\frac{\alpha - \rho}{\beta}\right)^2 + \rho\right\rbrace t\right)$.

Suppose the American claim is given by $F(t,\omega) = g(t, X_1(t))$ for some continuous lower bounded function $g(t, x_1)$. Then the price of the American option is

$$p_A(F) = \sup_{\tau \le T} E[K(\tau)g(\tau, X_1(\tau))].$$

</div>

#### The American Put Option

To simplify further, assume $\alpha = \rho$ (so that $P = Q$) and $g(t,x_1) = (a - x_1)^+$ where $a > 0$ is a constant. Then the problem becomes to find **the American put option price**:

$$p_A(F) = \sup_{\tau \le T} E[e^{-\rho\tau}(a - X_1(\tau))^+].$$

The owner of the American put option has the right (but not the obligation) to *sell* one stock at a specified price $a$ at any time $\tau$ he chooses before or at the terminal time $T$. If he sells at a time $\tau \le T$ when the market price is $X_1(\tau) < a$, he increases his fortune with the difference $a - X_1(\tau)$. Thus the expression above represents the maximal expected discounted payoff to the owner of the option.

We search for a function $\phi(s,x_1) \in C^1(\mathbf{R}^2)$ satisfying the **variational inequalities** (see Theorem 10.4.1):

$$\phi(s,x_1) \ge e^{-\rho s}(a - x_1)^+ \quad \text{for all } s, x_1$$

$$\frac{\partial\phi}{\partial s} + \rho x_1\frac{\partial\phi}{\partial x_1} + \frac{1}{2}\beta^2 x_1^2\frac{\partial^2\phi}{\partial x_1^2} \le 0 \quad \text{outside } \overline{D}$$

$$\frac{\partial\phi}{\partial s} + \rho x_1\frac{\partial\phi}{\partial x_1} + \frac{1}{2}\beta^2 x_1^2\frac{\partial^2\phi}{\partial x_1^2} = 0 \quad \text{on } D$$

where $D = \lbrace(s,x_1);\; \phi(s,x_1) > e^{-\rho s}(a - x_1)^+\rbrace$ is the **continuation region**.

If such a $\phi$ is found and the additional assumptions of Theorem 10.4.1 hold, then $\phi(s,x_1) = \Phi(s,x_1)$ and hence $p_A(F) = \phi(0, x_1)$ is the option price at time $t = 0$. Moreover,

$$\tau^* = \tau_D = \inf\lbrace t > 0;\; (s + t, X_1(t)) \notin D\rbrace$$

is the corresponding optimal stopping time, i.e. the optimal time to exercise the American option. Unfortunately, even in this case it seems that an explicit analytic solution is very hard (possibly impossible) to find. However, there are interesting partial results and good approximation procedures.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Structure of the Continuation Region)</span></p>

It is known (see Jacka (1991)) that the continuation region $D$ has the form

$$D = \lbrace(t,x_1) \in (0,T) \times \mathbf{R},\; x_1 > f(t)\rbrace$$

i.e. $D$ is the region above the graph of $f$, for some continuous, increasing function $f\colon (0,T) \to \mathbf{R}$. In Barles et al. (1995) it is shown that

$$f(t) \sim a - \beta a\sqrt{(T-t)\lvert\ln(T-t)\rvert} \quad \text{as } t \to T^-.$$

The exact form of the continuation region boundary is still unknown.

</div>

For the corresponding American *call* option the situation is much simpler: it is always optimal to exercise at the terminal time $T$, so the price of the American call coincides with that of a European call (see Exercise 12.14).

## Appendix B: Conditional Expectation

Let $(\Omega, \mathcal{F}, P)$ be a probability space and let $X\colon \Omega \to \mathbf{R}^n$ be a random variable such that $E[\lvert X\rvert] < \infty$. If $\mathcal{H} \subset \mathcal{F}$ is a $\sigma$-algebra, then the **conditional expectation** of $X$ given $\mathcal{H}$, denoted by $E[X \mid \mathcal{H}]$, is defined as follows:

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(B.1 — Conditional Expectation)</span></p>

$E[X \mid \mathcal{H}]$ is the (a.s. unique) function from $\Omega$ to $\mathbf{R}^n$ satisfying:

1. $E[X \mid \mathcal{H}]$ is $\mathcal{H}$-measurable
2. $\int_H E[X \mid \mathcal{H}]\,dP = \int_H X\,dP$, for all $H \in \mathcal{H}$

</div>

The existence and uniqueness of $E[X \mid \mathcal{H}]$ comes from the **Radon-Nikodym theorem**: Let $\mu$ be the measure on $\mathcal{H}$ defined by $\mu(H) = \int_H X\,dP$; $H \in \mathcal{H}$. Then $\mu$ is absolutely continuous w.r.t. $P\vert_\mathcal{H}$, so there exists a $P\vert_\mathcal{H}$-unique $\mathcal{H}$-measurable function $F$ on $\Omega$ such that $\mu(H) = \int_H F\,dP$ for all $H \in \mathcal{H}$. Thus $E[X \mid \mathcal{H}] := F$.

Note that condition (2) is equivalent to: $\int_\Omega Z \cdot E[X \mid \mathcal{H}]\,dP = \int_\Omega Z \cdot X\,dP$ for all $\mathcal{H}$-measurable $Z$.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(B.2 — Properties of Conditional Expectation)</span></p>

Suppose $Y\colon \Omega \to \mathbf{R}^n$ is another random variable with $E[\lvert Y\rvert] < \infty$ and let $a, b \in \mathbf{R}$. Then:

* **a)** $E[aX + bY \mid \mathcal{H}] = aE[X \mid \mathcal{H}] + bE[Y \mid \mathcal{H}]$ (linearity)
* **b)** $E[E[X \mid \mathcal{H}]] = E[X]$ (tower property with full $\sigma$-algebra)
* **c)** $E[X \mid \mathcal{H}] = X$ if $X$ is $\mathcal{H}$-measurable
* **d)** $E[X \mid \mathcal{H}] = E[X]$ if $X$ is independent of $\mathcal{H}$
* **e)** $E[Y \cdot X \mid \mathcal{H}] = Y \cdot E[X \mid \mathcal{H}]$ if $Y$ is $\mathcal{H}$-measurable (pulling out known factors)

</div>

<details class="accordion" markdown="1">
<summary>Proof of (d) and (e)</summary>

**d):** If $X$ is independent of $\mathcal{H}$, then for $H \in \mathcal{H}$:

$$\int_H X\,dP = \int_\Omega X \cdot \mathcal{X}_H\,dP = \int_\Omega X\,dP \cdot \int_\Omega \mathcal{X}_H\,dP = E[X] \cdot P(H)$$

so the constant $E[X]$ satisfies both (1) and (2).

**e):** First establish for $Y = \mathcal{X}_H$ (indicator function) for some $H \in \mathcal{H}$. Then for all $G \in \mathcal{H}$:

$$\int_G Y \cdot E[X \mid \mathcal{H}]\,dP = \int_{G \cap H} E[X \mid \mathcal{H}]\,dP = \int_{G \cap H} X\,dP = \int_G YX\,dP$$

so $Y \cdot E[X \mid \mathcal{H}]$ satisfies both (1) and (2). The result extends to simple functions and then by approximation to general $\mathcal{H}$-measurable $Y$.

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(B.3 — Tower Property)</span></p>

Let $\mathcal{G}, \mathcal{H}$ be $\sigma$-algebras such that $\mathcal{G} \subset \mathcal{H}$. Then

$$E[X \mid \mathcal{G}] = E[E[X \mid \mathcal{H}] \mid \mathcal{G}].$$

</div>

<details class="accordion" markdown="1">
<summary>Proof</summary>

If $G \in \mathcal{G}$ then $G \in \mathcal{H}$ and therefore $\int_G E[X \mid \mathcal{H}]\,dP = \int_G X\,dP$. Hence $E[E[X \mid \mathcal{H}] \mid \mathcal{G}] = E[X \mid \mathcal{G}]$ by uniqueness.

</details>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(B.4 — Jensen's Inequality for Conditional Expectation)</span></p>

If $\phi\colon \mathbf{R} \to \mathbf{R}$ is convex and $E[\lvert\phi(X)\rvert] < \infty$, then

$$\phi(E[X \mid \mathcal{H}]) \le E[\phi(X) \mid \mathcal{H}].$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(B.5)</span></p>

* **(i)** $\lvert E[X \mid \mathcal{H}]\rvert \le E[\lvert X\rvert \mid \mathcal{H}]$
* **(ii)** $\lvert E[X \mid \mathcal{H}]\rvert^2 \le E[\lvert X\rvert^2 \mid \mathcal{H}]$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(B.6)</span></p>

If $X_n \to X$ in $L^2$, then $E[X_n \mid \mathcal{H}] \to E[X \mid \mathcal{H}]$ in $L^2$.

</div>

## Appendix C: Uniform Integrability and Martingale Convergence

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(C.1 — Uniform Integrability)</span></p>

Let $(\Omega, \mathcal{F}, P)$ be a probability space. A family $\lbrace f_j\rbrace_{j \in J}$ of real, measurable functions $f_j$ on $\Omega$ is called **uniformly integrable** if

$$\lim_{M \to \infty}\left(\sup_{j \in J}\left\lbrace\int_{\lbrace\lvert f_j\rvert > M\rbrace}\lvert f_j\rvert\,dP\right\rbrace\right) = 0.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(C.2 — Uniform Integrability Test Function)</span></p>

A function $\psi\colon [0,\infty) \to [0,\infty)$ is called a **u.i. (uniform integrability) test function** if $\psi$ is increasing, convex, and $\lim_{x \to \infty} \frac{\psi(x)}{x} = \infty$.

For example, $\psi(x) = x^p$ is a u.i. test function if $p > 1$, but not if $p = 1$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(C.3 — Test Function Characterization)</span></p>

The family $\lbrace f_j\rbrace_{j \in J}$ is uniformly integrable if and only if there is a u.i. test function $\psi$ such that

$$\sup_{j \in J}\left\lbrace\int \psi(\lvert f_j\rvert)\,dP\right\rbrace < \infty.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(C.4 — Uniform Integrability and $L^1$ Convergence)</span></p>

Suppose $\lbrace f_k\rbrace_{k=1}^\infty$ is a sequence of real measurable functions on $\Omega$ such that $\lim_{k \to \infty} f_k(\omega) = f(\omega)$ for a.a. $\omega$. Then the following are equivalent:

1. $\lbrace f_k\rbrace$ is uniformly integrable
2. $f \in L^1(P)$ and $f_k \to f$ in $L^1(P)$, i.e. $\int \lvert f_k - f\rvert\,dP \to 0$ as $k \to \infty$

</div>

### Convergence Theorems for Martingales

Let $(\Omega, \mathcal{N}, P)$ be a probability space and let $\lbrace\mathcal{N}_t\rbrace_{t \ge 0}$ be an increasing family of $\sigma$-algebras, $\mathcal{N}_t \subset \mathcal{N}$ for all $t$. A stochastic process $N_t\colon \Omega \to \mathbf{R}$ is called a **supermartingale** (w.r.t. $\lbrace\mathcal{N}_t\rbrace$) if $N_t$ is $\mathcal{N}_t$-adapted, $E[\lvert N_t\rvert] < \infty$ for all $t$ and

$$N_t \ge E[N_s \mid \mathcal{N}_t] \quad \text{for all } s > t.$$

Similarly, if the inequality is reversed for all $s > t$, then $N_t$ is called a **submartingale**. And if equality holds, $N_t$ is called a **martingale**.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(C.5 — Doob's Martingale Convergence Theorem I)</span></p>

Let $N_t$ be a right continuous supermartingale with the property that $\sup_{t > 0} E[N_t^-] < \infty$, where $N_t^- = \max(-N_t, 0)$. Then the pointwise limit

$$N(\omega) = \lim_{t \to \infty} N_t(\omega)$$

exists for a.a. $\omega$ and $E[N^-] < \infty$.

</div>

Note, however, that the convergence need not be in $L^1(P)$. In order to obtain this we need uniform integrability:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(C.6 — Doob's Martingale Convergence Theorem II)</span></p>

Let $N_t$ be a right-continuous supermartingale. Then the following are equivalent:

1. $\lbrace N_t\rbrace_{t \ge 0}$ is uniformly integrable
2. There exists $N \in L^1(P)$ such that $N_t \to N$ a.e. $(P)$ and $N_t \to N$ in $L^1(P)$, i.e. $\int \lvert N_t - N\rvert\,dP \to 0$ as $t \to \infty$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(C.7 — $L^p$-Bounded Continuous Martingales Converge)</span></p>

Let $M_t$ be a continuous martingale such that $\sup_{t > 0} E[\lvert M_t\rvert^p] < \infty$ for some $p > 1$. Then there exists $M \in L^1(P)$ such that $M_t \to M$ a.e. $(P)$ and $\int \lvert M_t - M\rvert\,dP \to 0$ as $t \to \infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(C.8 — Discrete $L^p$-Bounded Martingale Convergence)</span></p>

Let $M_k$; $k = 1, 2, \ldots$ be a discrete time martingale and assume that $\sup_k E[\lvert M_k\rvert^p] < \infty$ for some $p > 1$. Then there exists $M \in L^1(P)$ such that $M_k \to M$ a.e. $(P)$ and $\int \lvert M_k - M\rvert\,dP \to 0$ as $k \to \infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(C.9 — Lévy's Upward Theorem)</span></p>

Let $X \in L^1(P)$, let $\lbrace\mathcal{N}_k\rbrace_{k=1}^\infty$ be an increasing family of $\sigma$-algebras, $\mathcal{N}_k \subset \mathcal{F}$ and define $\mathcal{N}_\infty$ to be the $\sigma$-algebra generated by $\lbrace\mathcal{N}_k\rbrace_{k=1}^\infty$. Then

$$E[X \mid \mathcal{N}_k] \to E[X \mid \mathcal{N}_\infty] \quad \text{as } k \to \infty$$

a.e. $P$ and in $L^1(P)$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Corollary C.9</summary>

$M_k := E[X \mid \mathcal{N}_k]$ is a u.i. martingale, so there exists $M \in L^1(P)$ such that $M_k \to M$ a.e. $P$ and in $L^1(P)$, as $k \to \infty$. It remains to prove that $M = E[X \mid \mathcal{N}_\infty]$. Note that

$$\lVert M_k - E[M \mid \mathcal{N}_k]\rVert_{L^1(P)} = \lVert E[M_k \mid \mathcal{N}_k] - E[M \mid \mathcal{N}_k]\rVert_{L^1(P)} \le \lVert M_k - M\rVert_{L^1(P)} \to 0.$$

Hence if $F \in \mathcal{N}_{k_0}$ and $k \ge k_0$: $\int_F (X - M)\,dP = \int_F E[X - M \mid \mathcal{N}_k]\,dP = \int_F (M_k - E[M \mid \mathcal{N}_k])\,dP \to 0$. Therefore $\int_F (X - M)\,dP = 0$ for all $F \in \bigcup_{k=1}^\infty \mathcal{N}_k$, and hence $E[X \mid \mathcal{N}_\infty] = E[M \mid \mathcal{N}_\infty] = M$.

</details>
