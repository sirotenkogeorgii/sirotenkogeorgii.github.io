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

$\textbf{Definition: (sigma-algebra):}$ Given a set $\Omega$, a family $\mathcal{F}$ of subsets of $\Omega$ is a $\sigma$-algebra if it satisfies: 
1. $\emptyset \in \mathcal{F}$
2. $F \in \mathcal{F} \implies F^C \in \mathcal{F}$, where $F^C = \Omega \setminus F$
3. $A_1, A_2, \dots \in \mathcal{F} \implies A := \bigcup_{i=1}^\infty A_i \in \mathcal{F}$ 
   
The pair $(\Omega, \mathcal{F})$ is a measurable space.

$\textbf{Definition (Probability Measure):}$ A probability measure $P$ on a measurable space $(\Omega, \mathcal{F})$ is a function $P: \mathcal{F} \to [0, 1]$ such that: 

* **(a)** $P(\emptyset) = 0, P(\Omega) = 1 $
* **(b)** If $\{A_i\}_{i=1}^\infty$ is a disjoint collection of sets in $\mathcal{F}$ (i.e., $A_i \cap A_j = \emptyset for i \ne j$), then:  

$$P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)$$

The triple $(\Omega, \mathcal{F}, P)$ is a probability space.

$\textbf{Definition (Complete Probability Space):}$ A probability space $(\Omega, \mathcal{F}, P)$ is complete if $\mathcal{F}$ contains all subsets $G$ of $\Omega$ with $P$-outer measure zero, where the outer measure is defined as:  

$$P^*(G) := \inf{P(F); F \in \mathcal{F}, G \subset F} = 0$$

> Remark: Subsets $F \in \mathcal{F}$ are called $\mathcal{F}$-measurable sets or events. $P(F)$ is the probability that event $F$ occurs. If $P(F)=1$, the event occurs almost surely (a.s.).

$\textbf{Definition } (\textbf{Generated } \sigma \textbf{-algebra:})$ Given a family $\mathcal{U}$ of subsets of $\Omega$, the $\sigma$-algebra generated by $\mathcal{U}$, denoted $\mathcal{H}_\mathcal{U}$, is the smallest $\sigma$-algebra containing

$$\mathcal{U}:  \mathcal{H}_\mathcal{U} = \bigcap {\mathcal{H} ; \mathcal{H} \text{ is a } \sigma\text{-algebra on } \Omega, \mathcal{U} \subset \mathcal{H}}$$  

If $\Omega$ is a topological space and $\mathcal{U}$ is the collection of all open subsets, $\mathcal{B} = \mathcal{H}_\mathcal{U}$ is the Borel $\sigma$-algebra.

$\textbf{Definition } (\mathcal{F}\textbf{-measurable Function:})$ A function $Y: \Omega \to \mathbb{R}^n$ is $\mathcal{F}$-measurable if for all open sets

$$U \subset \mathbb{R}^n:  Y^{-1}(U) := {\omega \in \Omega ; Y(\omega) \in U} \in \mathcal{F}$$

This is equivalent to the condition holding for all Borel sets $U \subset \mathbb{R}^n$.

$\textbf{Definition } (\sigma \textbf{-algebra } \textbf{generated by a Function}:)$ For a function $X: \Omega \to \mathbb{R}^n$, the $\sigma$-algebra generated by $X$, denoted $\mathcal{H}_X$, is the smallest $\sigma$-algebra on $\Omega$ making $X$ measurable. It consists of all sets $X^{-1}(B)$ for all Borel sets $B \in \mathcal{B}$ on $\mathbb{R}^n$.

$\textbf{Theorem/Lemma (Doob-Dynkin Lemma):}$ Let $X, Y: \Omega \to \mathbb{R}^n$ be two functions. $Y$ is $\mathcal{H}_X$-measurable if and only if there exists a Borel measurable function $g: \mathbb{R}^n \to \mathbb{R}^n$ such that $Y = g(X)$.

$\textbf{Definition (Random Variable and Expectation):}$ Let $(\Omega, \mathcal{F}, P)$ be a complete probability space. A random variable $X$ is an $\mathcal{F}$-measurable function $X: \Omega \to \mathbb{R}^n$. The distribution of $X$ is the probability measure $\mu_X$ on $\mathbb{R}^n$ defined by $\mu_X(B) = P(X^{-1}(B))$. If $\int_\Omega |X(\omega)| dP(\omega) < \infty$, the expectation of $X$ is:  

$$E[X] := \int_\Omega X(\omega) dP(\omega) = \int_{\mathbb{R}^n} x d\mu_X(x)$$

For a Borel measurable function $f: \mathbb{R}^n \to \mathbb{R}$, if 

$$\int_\Omega |f(X(\omega))| dP(\omega) < \infty:  E[f(X)] := \int_\Omega f(X(\omega)) dP(\omega) = \int_{\mathbb{R}^n} f(x) d\mu_X(x)$$

$\textbf{Definition (Independence):}$ Two events $A, B \in \mathcal{F}$ are independent if $P(A \cap B) = P(A) \cdot P(B)$. A collection $\mathcal{A} = \lbrace\mathcal{H}_i ; i \in I\rbrace$ of families of measurable sets is independent if for any distinct indices $i_1, \dots, i_k$ and any choice of sets $H_{i_j} \in \mathcal{H}_{i_j}$:

$$P(H_{i_1} \cap \cdots \cap H_{i_k}) = P(H_{i_1}) \cdots P(H_{i_k})$$

A collection of random variables $\lbrace X_i ; i \in I\rbrace$ is independent if the collection of their generated $\sigma$-algebras $\lbrace\mathcal{H}_{X_i}\rbrace$ is independent.

> Remark: If two random variables $X, Y: \Omega \to \mathbb{R}$ are independent and $E[|X|] < \infty, E[|Y|] < \infty$, then $E[XY] = E[X]E[Y]$.

$\textbf{Definition (Stochastic Process):}$ A stochastic process is a parametrized collection of random variables $\lbrace X_t\rbrace_{t \in T}$ defined on a probability space $(\Omega, \mathcal{F}, P)$ and taking values in $\mathbb{R}^n$. The parameter space $T$ is typically $[0, \infty)$.

> Remark: For a fixed $t \in T, X_t(\omega)$ is a random variable. For a fixed $\omega \in \Omega$, the function $t \mapsto X_t(\omega)$ is a path of the process. A process can also be viewed as a function of two variables, $(t, \omega) \mapsto X(t, \omega)$, or as a probability measure on the space of all functions from $T$ to $\mathbb{R}^n$, denoted $(\mathbb{R}^n)^T$.

$\textbf{Definition (Finite-Dimensional Distributions):}$ The finite-dimensional distributions of a process $\lbrace X_t\rbrace_{t \in T}$ are the measures $\mu_{t_1, \dots, t_k}$ on $\mathbb{R}^{nk}$ for $k=1, 2, \dots$ defined by:  

$$\mu_{t_1, \dots, t_k}(F_1 \times \cdots \times F_k) = P[X_{t_1} \in F_1, \dots, X_{t_k} \in F_k]$$

for Borel sets $F_i \subset \mathbb{R}^n$.

$\textbf{Theorem/Lemma (Kolmogorov's Extension Theorem):}$ Let $\lbrace \nu_{t_1, \dots, t_k}\rbrace$ be a family of probability measures on $\mathbb{R}^{nk}$ for all $t_1, \dots, t_k \in T$ and $k \in \mathbb{N}$. If this family satisfies the consistency conditions: 
* (K1) $\nu_{t_{\sigma(1)}, \dots, t_{\sigma(k)}}(F_1 \times \cdots \times F_k) = \nu_{t_1, \dots, t_k}(F_{\sigma^{-1}(1)} \times \cdots \times F_{\sigma^{-1}(k)})$ for all permutations $\sigma$. * 
* (K2) $\nu_{t_1, \dots, t_k}(F_1 \times \cdots \times F_k) = \nu_{t_1, \dots, t_k, t_{k+1}, \dots, t_{k+m}}(F_1 \times \cdots \times F_k \times \mathbb{R}^n \times \cdots \times \mathbb{R}^n)$ for all $m \in \mathbb{N}$. Then there exists a probability space $(\Omega, \mathcal{F}, P)$ and a stochastic process $\lbrace X_t\rbrace_{t \in T}$ on it such that its finite-dimensional distributions are given by the family $\nu$.

2.2 An Important Example: Brownian Motion

Remark: (Construction of Brownian Motion) An n-dimensional Brownian motion starting at x \in \mathbb{R}^n is constructed via Kolmogorov's extension theorem. The finite-dimensional distributions are specified for 0 \le t_1 \le \cdots \le t_k by defining a measure \nu_{t_1, \dots, t_k} on \mathbb{R}^{nk}:  \nu_{t_1, \dots, t_k}(F_1 \times \cdots \times F_k) = \int_{F_1 \times \cdots \times F_k} p(t_1, x, x_1) p(t_2-t_1, x_1, x_2) \cdots p(t_k-t_{k-1}, x_{k-1}, x_k) dx_1 \cdots dx_k  where the transition density function p(t, x, y) is the Gaussian kernel:  p(t, x, y) = (2\pi t)^{-n/2} \exp\left(-\frac{|x-y|^2}{2t}\right) \text{ for } t > 0  and p(0, x, y)dy = \delta_x(y) is the point mass at x.

Definition: (Brownian Motion) A stochastic process \{B_t\}_{t \ge 0} whose finite-dimensional distributions are given by the construction above is called a Brownian motion starting at x. By construction, P^x(B_0=x)=1.

Remark: (Properties of Brownian Motion) (i) Gaussian Process: For 0 \le t_1 \le \cdots \le t_k, the random vector Z = (B_{t_1}, \dots, B_{t_k}) \in \mathbb{R}^{nk} has a multivariate normal distribution. Its characteristic function is:  E^x\left[\exp\left(i \sum_{j=1}^{nk} u_j Z_j\right)\right] = \exp\left(-\frac{1}{2} \sum_{j,m} u_j c_{jm} u_m + i \sum_j u_j M_j\right)  The mean vector is M = E^x[Z] = (x, x, \dots, x) \in \mathbb{R}^{nk}. The covariance matrix C=[c_{jm}] has entries corresponding to the block matrix components E^x[(B_{t_i}-x)(B_{t_j}-x)^T] = \min(t_i, t_j) I_n, resulting in:  C = \begin{pmatrix} t_1 I_n & t_1 I_n & \cdots & t_1 I_n \ t_1 I_n & t_2 I_n & \cdots & t_2 I_n \ \vdots & \vdots & \ddots & \vdots \ t_1 I_n & t_2 I_n & \cdots & t_k I_n \end{pmatrix}  From this, for all t,s \ge 0:  E^x[B_t] = x   E^x[(B_t - x)(B_s - x)] = n \min(s, t)   E^x[(B_t-B_s)^2] = n(t-s) \quad \text{if } t \ge s  (ii) Independent Increments: For any 0 \le t_1 < t_2 < \cdots < t_k, the random variables B_{t_1}, B_{t_2}-B_{t_1}, \dots, B_{t_k}-B_{t_{k-1}} are independent.

Proof: Since the increments are jointly normal, independence is equivalent to being uncorrelated. For t_i < t_j: E^x[(B_{t_i} - B_{t_{i-1}})(B_{t_j} - B_{t_{j-1}})] = n(\min(t_i, t_j) - \min(t_{i-1}, t_j) - \min(t_i, t_{j-1}) + \min(t_{i-1}, t_{j-1}})). Assuming t_{i-1} < t_i < t_{j-1} < t_j: E^x[(B_{t_i} - B_{t_{i-1}})(B_{t_j} - B_{t_{j-1}})] = n(t_i - t_{i-1} - t_i + t_{i-1}) = 0.

Definition: (Version of a Stochastic Process) A process \{X_t\} is a version (or modification) of a process \{Y_t\} if for all t:  P({\omega; X_t(\omega) = Y_t(\omega)}) = 1  Remark: Versions of a process have the same finite-dimensional distributions but may have different path properties.

Theorem/Lemma: (Kolmogorov's Continuity Theorem) Suppose a process \{X_t\}_{t \ge 0} satisfies the condition that for all T > 0, there exist positive constants \alpha, \beta, D such that for 0 \le s, t \le T:  E[|X_t - X_s|^\alpha] \le D \cdot |t - s|^{1+\beta}  Then there exists a continuous version of X.

Remark: For Brownian motion \{B_t\}, the following holds:  E^x[|B_t - B_s|^4] = n(n+2)|t-s|^2  This satisfies the condition of Kolmogorov's continuity theorem with \alpha=4, \beta=1, D=n(n+2). Therefore, a continuous version of Brownian motion exists, and this version is assumed henceforth.

Remark: If B_t = (B_t^{(1)}, \dots, B_t^{(n)}) is an n-dimensional Brownian motion, then the component processes \{B_t^{(j)}\}_{t \ge 0} for j=1, \dots, n are independent, 1-dimensional Brownian motions.


