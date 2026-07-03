---
layout: default
title: Numerical Methods for Bayesian Inverse Problems
date: 2025-03-17
excerpt: Notes on inverse problems, regularization, and Bayesian approaches to ill-posed problems.
tags:
  - inverse-problems
  - bayesian-inference
  - numerical-methods
  - singular-value-decomposition
  - evidence-lower-bound
  - markov-chain-monte-carlo
  # - jeffreys-prior
  # - fisher-information
---

<style>
  .accordion summary {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    background-color: var(--accent-soft, #f5f6ff);
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--accent-strong, #2c3e94);
    border-radius: 0.25rem;
  }
</style>

# Numerical Methods for Bayesian Inverse Problems

**Table of Contents**
- TOC
{:toc}

## Problems

[Selected Problems](/subpages/books/numerical_methods_for_bip/problems/)

## Chapter 1: Introduction

### 1.1 A Motivating Example

Many processes in science and engineering can be modelled via differential equations. Assuming complete knowledge of all the necessary parameters, initial and boundary conditions, the solution of such a differential equation allows in principle to fully predict the process.

Consider for example a rod of length 1 with thermal diffusivity coefficient $\alpha$. The temperature at the two ends of the rod is assumed to be 0. Then the temperature distribution $u(x,t)$ satisfies

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \qquad \text{for } 0 < x < 1,\ t > 0, \tag{1.1}$$

with boundary conditions

$$u(0, t) = u(1, t) = 0, \qquad \text{for } t > 0, \tag{1.2}$$

and initial condition

$$u(x, 0) = u_0(x), \qquad \text{for } 0 < x < 1. \tag{1.3}$$

We can now consider the following inverse problem in this simple setting: Given the temperature distribution at some time $T > 0$, can we recover the initial temperature profile $u_0$ at time $t = 0$?

Using the Laplace transform method, the solution to the heat equation (1.1)-(1.3) has the general form

$$u(x, t) = \sum_{n=1}^{\infty} \theta_n e^{-(n\pi)^2 \alpha t} \sin(n\pi x),$$

where $\theta_n$ are the Fourier-sine-coefficients of the initial condition $u_0$, i.e.,

$$u_0(x) = \sum_{n=1}^{\infty} \theta_n \sin(n\pi x).$$

Thus, in principle the coefficients $\theta_n$ of the initial condition $u_0$ can be estimated from measurements of $u(x, T)$ at time $T > 0$. However, consider two initial conditions $u_0^{(1)}$ and $u_0^{(2)}$ with $\theta_1^{(1)} = \theta_1^{(2)} = 1$ that differ only in one single frequency component, i.e.,

$$u_0^{(1)}(x) - u_0^{(2)}(x) = \theta_N \sin(N\pi x), \quad \text{for some } N > 1.$$

At time $T > 0$ the two solutions will differ by

$$u^{(1)}(x, T) - u^{(2)}(x, T) = \theta_N e^{-(N\pi)^2 \alpha T} \sin(N\pi x),$$

which is exponentially small. Therefore, any information about this difference will be lost due to measurement noise for $T$ or $N$ sufficiently large, even if the noise is extremely small.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Inverse Heat Equation)</span></p>

To demonstrate this, consider the case of $\alpha = 0.01$ (which after nondimensionalisation roughly corresponds to a copper rod of length 10cm in dimensionless quantities with time measured in seconds) and let

$$\theta_1^{(1)} = \theta_1^{(2)} = 1, \quad \theta_5^{(2)} = 0.5 \quad \text{and} \quad \theta_i^{(j)} = 0 \quad \text{otherwise}.$$

The solution is plotted at the initial time $t = 0$ and at $t = 1$ and $t = 4$. Even though the two initial conditions clearly differ significantly and the difference is not even particularly oscillatory, it is already very difficult to distinguish the two solutions at $t = 1$. At $t = 4$, it will be impossible to say whether the observed temperature profile came from $u_0^{(1)}$ or from $u_0^{(2)}$.

</div>

### 1.2 What is an Inverse Problem?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.2.1</span><span class="math-callout__name">(Well-Posed Problem)</span></p>

According to **Hadamard** (1865-1963) a problem is called **well-posed**, if

1. a solution exists (**existence**),
2. the solution is unique (**uniqueness**),
3. the solution depends continuously on the input data (**stability**).

If any of these properties is violated, we speak of an **ill-posed** problem.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Well-Posed Problem of System of Linear Equations)</span></p>

Let $X, Y$ be Hilbert spaces and $A : X \to Y$ be linear and bounded (we write $A \in \mathcal{L}(X, Y)$). Then the (forward) problem to compute $y = Ax$ for a given $x \in X$ is clearly well-posed. For the corresponding **inverse problem**, to solve the linear equation $Ax = y$ for a given $y \in Y$ (find $x$), the conditions of Hadamard are:

1. **Existence:** $y \in \mathcal{R}(A)$, i.e. $A$ is surjective.
2. **Uniqueness:** $A$ is injective.
3. **Stability:** $A^{-1}$ is bounded. ($\exists c \forall y \|\|A^{-1}y\|\| \leq c\|\|y\|\|$)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">()</span></p>

For finite dimensional problems, these conditions may be satisfied for a bounded linear operator $A$, although the problem typically gets more and more **ill-conditioned** as the dimension increases. In infinite dimensions on the other hand, it is in general impossible. In particular, for compact operators $A$ the singular values have to accumulate at $0$, which implies that $A^{-1}$ is unbounded. Thus, fundamentally the inverse problem is ill-posed if the forward problem is well-posed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compact operators)</span></p>

A **compact operator** $A \in \mathcal{L}(X, Y)$ between Banach spaces is one that maps bounded sets to *relatively compact* sets — equivalently, every bounded sequence $(x_n) \subset X$ admits a subsequence such that $(Ax_n)$ converges in $Y$. In Hilbert spaces, three equivalent characterisations are useful:

1. **Approximation by finite-rank operators:** $A$ is the norm limit of operators with finite dimensional range, i.e. $A$ is "almost finite dimensional."
2. **Singular values:** $A$ admits an SVD 
   
   $$A = \sum_n \sigma_n \langle \cdot, v_n \rangle u_n \text{ with } \sigma_n \to 0$$

3. **Smoothing:** integral operators 
   
   $$(Af)(x) = \int k(x, y)\, f(y)\, dy$$
   
   with square-integrable kernel $k$ are compact — they "blur" fine-scale features of $f$.

**Why this matters for inverse problems.** In infinite dimensions, the identity operator is *not* compact (the unit ball is not relatively compact, by Riesz's lemma). Hence, if $A$ is compact and injective, $A^{-1}$ cannot be bounded: otherwise $A^{-1}A = I$ would be compact, a contradiction. Concretely, $\sigma_n \to 0$ implies $\sigma_n^{-1} \to \infty$, so $A^{-1}$ amplifies high-frequency components without bound — which is precisely the failure of Hadamard's stability condition. This is why compact forward operators are the canonical source of ill-posed inverse problems: the forward map smooths, and inversion must un-smooth.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why singular values of compact operators accumulate at $0$)</span></p>

Let $A \in \mathcal{L}(X, Y)$ be compact between Hilbert spaces, with SVD $A = \sum_n \sigma_n \langle \cdot, v_n\rangle u_n$, where $(v_n) \subset X$ and $(u_n) \subset Y$ are orthonormal and $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$. We claim $\sigma_n \to 0$.

**Intuition.** Compactness forces "almost finite dimensionality": $A$ must be well-approximated by its finite rank truncations $A_N = \sum_{n \leq N} \sigma_n \langle \cdot, v_n\rangle u_n$. Since $\lVert A - A_N \rVert = \sigma_{N+1}$, approximability forces $\sigma_{N+1} \to 0$.

**Spectral-theorem view.** $A^*A$ is compact and self-adjoint. The spectral theorem for such operators says the spectrum consists of eigenvalues with $0$ as the only possible accumulation point. The $\sigma_n^2$ are those eigenvalues, so $\sigma_n^2 \to 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (by contradiction)</summary>

If $A$ has finite rank, only finitely many $\sigma_n$ are nonzero and we are done. Otherwise $(\sigma_n)$ is non-increasing in $[0, \infty)$ and converges to some $\sigma_* \geq 0$. Suppose $\sigma_* > 0$.

The sequence $(v_n)$ is bounded ($\lVert v_n \rVert = 1$). By compactness of $A$, $(Av_n)$ must have a convergent — hence Cauchy — subsequence. But for $n \neq m$,

$$\lVert Av_n - Av_m \rVert^2 = \lVert \sigma_n u_n - \sigma_m u_m \rVert^2 = \sigma_n^2 + \sigma_m^2 \geq 2\sigma_*^2 > 0,$$

using orthonormality of $(u_n)$. So no subsequence of $(Av_n)$ is Cauchy — contradiction. Hence $\sigma_* = 0$. $\blacksquare$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Ill-posed vs. ill-conditioned)</span></p>

**Ill-posed** is qualitative: one of Hadamard's three conditions fails outright — no solution exists, solutions are non-unique, or $A^{-1}$ is unbounded (discontinuous dependence on the data).

**Ill-conditioned** is quantitative: the problem is technically well-posed (all three conditions hold), but the condition number $\kappa(A) = \lVert A \rVert\, \lVert A^{-1} \rVert$ is large, so small perturbations of the data produce large errors in the solution.

The two live on a spectrum: as $\kappa(A) \to \infty$, an ill-conditioned problem degenerates into an ill-posed one. This is exactly what the proposition above describes: discretising the inverse of a compact operator yields a sequence of well-posed but increasingly ill-conditioned finite dimensional problems, which inherit the ill-posedness of the underlying infinite dimensional problem in the limit.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Focus of this course)</span></p>

In this course, we will study how ill-posed problems can be solved in a numerically stable manner. Our particular focus will lie on the Bayesian approach and Bayesian techniques, but before we get there, we will first study classical approaches.

</div>

### 1.3 Further Examples

We finish this section by presenting a few other, typical examples of inverse problems in applications, which all have an infinite-dimensional state space and often also an infinite-dimensional (or at least high-dimensional) parameter space. Finite dimensional inverse problems, while still possibly leading to non-existence and non-uniqueness problems, do typically not violate Hadamard's third condition of stability in Definition 1.2.1 and -- while still interesting -- are fundamentally not as challenging. Our main focus will be on PDE-constrained Bayesian inverse problems, but there are also some classical examples.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(X-ray Tomography)</span></p>

Given a bounded domain $D$, for simplicity $D \subset \mathbb{R}^2$, representing a cross-sectional slice of the object to be studied. Assume that a pointlike X-ray source is placed on one side of the object. The radiation passes through the object and is detected on the other side by an X-ray film or a digital sensor. It is common to assume that the scattering of the X-rays by the traversed material is insignificant, i.e., only absorption occurs, and that rays are not deflected through interaction with the material. If we further assume that the **mass absorption coefficient** is proportional to the density of the material, the attenuation $\mathrm{d}I$ of the intensity $I(x)$ along a line segment $\mathrm{d}s$ at a point $x \in D$ is given by

$$\mathrm{d}I = -I(x)\theta(x)\,\mathrm{d}s$$

where $\theta(x) \ge 0$ is the mass absorption coefficient of the material. We assume that $\theta$ is compactly supported in $\overline{D}$ and bounded. If an X-ray is transmitted with intensity $I\_0^\ell$ along a straight line $\ell$ towards a receiver, the received intensity $I\_r^\ell$ can be obtained from the equation

$$\log I_r^\ell - \log I_0^\ell = \int_{I_0^\ell}^{I_r^\ell} \frac{\mathrm{d}I}{I} = -\int_\ell \theta(x)\, \mathrm{d}s. \tag{1.4}$$

The inverse problem of X-ray tomography can thus be stated as a problem of integral geometry: Estimate the function $\theta : D \to \mathbb{R}\_+$ from the values of its integrals along a set of straight lines $\lbrace \ell(n, s) : n \in \mathbb{R}^2,\ \lVert n \rVert\_2 = 1,\ s \in \mathbb{R} \rbrace$ passing through $D$, parametrised by their normal vector $n$ and their distance $s > 0$ from the origin. Denoting the data by $y(n, s) := \log\bigl( I\_r^{\ell(n,s)} / I\_0^{\ell(n,s)} \bigr)$, equation (1.4) leads to the linear operator equation

$$y = \mathcal{R}\theta$$

with compact integral operator $\mathcal{R}$, the so-called **Radon transform**.

The nature of the X-ray tomography problem depends on how many lines of integration are available. In the ideal case, we have data along all possible lines passing through the object. The classical results are based on the availability of this complete data. The problem can then be solved explicitly using the **inverse Radon transform**. However, it involves differentiating the data, which is an ill-posed problem in the sense of Hadamard, such that small errors in the data lead to large errors in the solution.

In practice, often only limited-angle data is available, which is further polluted by noise.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian Process Regression or Kriging)</span></p>

Many problems in spatial statistics are of the form that a functional quantity is to be estimated from a few point evaluations, a form of statistical interpolation also called **kriging**.

Let $D \subset \mathbb{R}^d$ be a bounded open set. Consider a field $u \in \mathcal{H} = L^2(D; \mathbb{R}^n)$. Assume that we are given noisy observations $\lbrace y\_k \rbrace\_{k=1}^q$ of a function $g : \mathbb{R}^n \to \mathbb{R}^\ell$ of the field $u$ at a set of points $\lbrace x\_k \rbrace\_{k=1}^q$. Thus

$$y_k = g(u(x_k)) + \eta_k,$$

where the $\lbrace \eta\_k \rbrace\_{k=1}^q$ describe the observational noise. Concatenating data, we have

$$y = \mathcal{G}(u) + \eta,$$

where $y = \bigl( y\_1^\top, \ldots, y\_q^\top \bigr)^\top \in \mathbb{R}^{\ell q}$ and $\eta = \bigl( \eta\_1^\top, \ldots, \eta\_q^\top \bigr)^\top \in \mathbb{R}^{\ell q}$. The observation operator $\mathcal{G}$ maps $V = \bigl( C(\overline{D}) \bigr)^n \subset \mathcal{H}$ to $W = \mathbb{R}^{\ell q}$. The inverse problem is to reconstruct the field $u$ from the data $y$.

In the Bayesian approach to this inverse problem, the unknown function $u \in \mathcal{H}$ is modelled as an $\mathcal{H}$-valued *random field* $U$, which is then conditioned on the observed data $y$. One specifies a so-called *prior measure* $\mu\_U$ on the random field $U$ and determines the so-called *posterior measure* $\mu\_{U \mid y}$ for $U$ given the observed data $y$.

If $g : \mathbb{R}^n \to \mathbb{R}^\ell$ is linear, so that $\mathcal{G}(u) = Au$ for some linear operator $A : V \to W$, if we assume that the observational noise $\eta$ is Gaussian $\mathcal{N}(0, \Sigma)$, with some covariance matrix $\Sigma$, and if the prior measure $\mu\_U$ on the random field $U$ is Gaussian $\mathcal{N}(m\_0, \mathcal{C}\_0)$, with some mean function $m\_0$ and some covariance operator $\mathcal{C}\_0$, then the posterior measure $\mu\_{U \mid y}$ is also Gaussian with a mean and covariance operator that can be computed explicitly, as we will see later in the course. This is called a **Gaussian process** and the data fitting approach is called **Gaussian process regression**. It has many interesting and favourable properties and is very popular in computational statistics.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Subsurface Flow, Heat Conduction, Impedance Tomography)</span></p>

The inverse heat equation, the PDE-constrained inverse problem we used as the motivating example in Section 1.1, is exponentially ill-posed. Another model problem, which is less severely ill-posed, is the problem to identify the diffusion coefficient $a$ in a stationary diffusion or heat conduction problem. Let $D \subseteq \mathbb{R}^d$ be a bounded domain and consider the elliptic PDE

$$-\operatorname{div}(a \nabla u) = f, \qquad u|_{\partial D} = 0. \tag{1.5}$$

If we assume that $f \in H^{-1}(D)$ and $a \in L^\infty(D)$ with $\operatorname{essinf}\_{x \in D} a(x) > 0$, then this problem has a unique weak solution $u \in H\_0^1(D)$. For a bounded linear (observation) operator $B : H\_0^1(D) \to \mathbb{R}^m$ we define the forward operator $\Phi(a) := Bu \in \mathbb{R}^m$; note that the solution $u \in H\_0^1(D)$ of (1.5) depends on $a$, so that $\Phi(a)$ is well-defined. One can show that $\Phi$ is a continuous and measurable function from

$$\lbrace a \in L^\infty(D) \, : \, \operatorname{essinf}_{x \in D} a(x) > 0 \rbrace \to \mathbb{R}^m.$$

This model problem is ubiquitous in many fields of mathematics due to the many important applications it appears in centrally. Whether we are interested in heat conduction, electrostatics, magnetostatics, porous media flow or even radiation shielding, the central mechanism in all those physical processes (in practically relevant regimes) is diffusion and a central question in practice is often how to estimate the diffusion coefficient non-destructively (or with minimal "destruction") from indirect measurements.

The Bayesian inverse problem associated with (1.5) is to find the diffusion coefficient $a \in L^\infty(D)$ from a noisy measurement

$$Y = \Phi(a) + E, \tag{1.6}$$

with $E \sim \mathcal{N}(0, \Gamma)$ for an SPD covariance matrix $\Gamma \in \mathbb{R}^{m \times m}$. In order to do so, we proceed as in the previous example. We specify a prior measure on the unknown diffusion coefficient and then condition on the measured data. We will discuss this example extensively in Sec. 4.1 below. The only important point we want to add here is that, since $u$ depends nonlinearly on $a$, even when the forward operator $\Phi$ in (1.6) is linear as a map of $u$, i.e., $\Phi(a) = Bu$, it is nonlinear as an operator on $a$. This makes it the pre-eminent infinite-dimensional non-linear (Bayesian) inverse problem.

</div>

There are many more examples of important inverse problems in applications -- some of them also studied in our groups in Heidelberg -- such as **inverse scattering** (geophysics, MRT), **inverse source problems** (Tsunami prediction, subsurface pollution), **data assimilation** (weather prediction), **parameter estimation** (pattern formation in developmental biology) or **epidemiology** (COVID-19 modelling and prediction).

For more examples see [Stuart, 2010, Chap. 3] and [Kaipio & Somersalo, 2004, Chap. 6].

## Chapter 2: Linear Inverse Problems and Regularisation

To motivate the remainder of this chapter, we will first consider the finite dimensional setting. However, **the most relevant issue in the numerical treatment of ill-posed problems, namely the lack of continuous dependence on the data, only emerges in infinite dimensions**. Thus, in the remainder of this chapter we analyse infinite dimensional linear inverse problems and introduce regularisation techniques to solve them approximatively in a numerically stable way.

### 2.1 Finite Dimensional Ill-Posed Problems (Matrix Equations)

It suffices to consider matrix equations. Every finite dimensional vector space $X$ over $\mathbb{R}$ is isomorphic to $\mathbb{R}^n$ and every linear operator on $\mathbb{R}^n$ has a matrix representation.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Linear System: Regular Case)</span></p>

**The regular case.** Consider a linear equation system of the form

$$Ax = y \tag{2.1.1}$$

with a symmetric, positive definite (SPD) $n \times n$ square matrix $A \in \mathbb{R}^{n \times n}$. Recall that such a matrix $A$ has $n$ positive, real eigenvalues $\lambda_1 \ge \ldots \ge \lambda_n > 0$ with corresponding eigenvectors $u_i \in \mathbb{R}^n$, $i = 1, \ldots, n$, with $\lVert u_i \rVert = 1$. Furthermore, $A$ has the spectral decomposition

$$A = \sum_{i=1}^{n} \lambda_i u_i u_i^\top \quad \left( = U \Lambda U^\top \right), \tag{2.1.2}$$

where the $i$th column of $U$ is $u_i$ and $\Lambda$ is a diagonal matrix with $\Lambda_{ii} = \lambda_i$. W.l.o.g. assume that $\lambda_1 = \mathcal{O}(1)$, in particular independent of $n$, otherwise rescale $A$ and $y$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Meaning of $\lambda_1 = \mathcal{O}(1)$)</span></p>

$\mathcal{O}(1)$ is big-O of one — a *constant*, independent of $n$. Formally, $f(n) = \mathcal{O}(1)$ means there exists $C > 0$ such that $\lvert f(n) \rvert \le C$ for all $n$: no growth and no decay as $n \to \infty$.

**Why the assumption is made.** We study a sequence of problems indexed by dimension $n$, and the object of interest is the condition number $\kappa(A) = \lambda_1 / \lambda_n$. If $\lambda_1$ itself scaled with $n$, a large $\kappa$ could come from $\lambda_1$ blowing up rather than from $\lambda_n$ collapsing — which would obscure the ill-posedness mechanism. Fixing $\lambda_1 = \mathcal{O}(1)$ by rescaling $A$ and $y$ (which does not change $\kappa$) pins the top of the spectrum to a fixed scale, so that any growth in $\kappa$ is attributable to $\lambda_n \to 0$.

</div>

The **condition number** of $A$ provides a measure for how accurate and stable the system (2.1.1) can be solved. It is given by the ratio of the largest and the smallest eigenvalue of $A$, i.e., $\kappa(A) = \lambda_1 / \lambda_n$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Error bound scaling with noise for regular linear systems)</span></p>

Consider that the data, namely the right hand side $y$, is only available in only a perturbed (or noisy) form as $y^\delta$, such that

$$\lVert y^\delta - y \rVert \le \delta \tag{2.1.3}$$

for some $\delta > 0$ in the Euclidean norm on $\mathbb{R}^n$, and denote by $x^\delta$ the solution of the perturbed system with right hand side $y^\delta$.

Using the condition number and our assumption on the scaling of $\lambda_1$ this can also be expressed as

$$\lVert x^\delta - x \rVert \le \kappa \lambda_1^{-1} \delta = \mathcal{O}(\kappa \delta).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Using the decomposition (2.1.2) of $A$, we get

$$x^\delta - x = \sum_{i=1}^{n} \frac{u_i^\top (y^\delta - y)}{\lambda_i} \, u_i.$$

Since the eigenvectors of $A$ can be chosen to be orthonormal, we can apply the Bessel inequality (two times) to obtain the bound

$$\lVert x^\delta - x \rVert^2 = \sum_{i=1}^{n} \lambda_i^{-2} |u_i^\top (y^\delta - y)|^2 \le \lambda_n^{-2} \lVert y^\delta - y \rVert^2 \le \lambda_n^{-2} \delta^2.$$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Intermediate Steps</summary>

$$x^\delta - x = \sum_{i=1}^{n} \underbrace{\frac{u_i^\top (y^\delta - y)}{\lambda_i}}_{:= c_i} \, u_i = \sum_{i=1}^{n} c_i \, u.$$

$$\lVert x^\delta - x \rVert^2 = \left\|\sum_{i=1}^n c_i u_i\right\|^2 = \left(\sum_{i=1}^n c_i u_i\right)^\top \left(\sum_{j=1}^n c_j u_j\right) = \sum_{i,j} c_i c_j, u_i^\top u_j$$

Because the $u_i$ are orthonormal,

$$
u_i^\top u_j=
\begin{cases}
1,& i=j,\\
0,& i\ne j,
\end{cases}
$$

so all cross terms vanish (remember that $c_i$ is a scalar), leaving only

$$\sum_{i=1}^n |c_i|^2$$

**We are using here Bessel inequality two times.**

</details>
</div>

Using the condition number and our assumption on the scaling of $\lambda_1$ this can also be expressed as

$$\lVert x^\delta - x \rVert \le \kappa \lambda_1^{-1} \delta = \mathcal{O}(\kappa \delta).$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Error bound scaling with noise for regular linear systems)</span></p>

The bound is sharp, which can be seen easily by choosing $y^\delta - y = \delta u_n$. Thus, any growth in the condition number of $A$ directly leads to an amplification of noise in the data in the solution.

Thus, for large condition numbers we say that the problem (2.1.1) is ill-posed -- recall for example that the condition number of the stiffness matrix $A$ in finite element discretisations of elliptic PDEs typically grows like $\mathcal{O}(h^{-2})$, where $h$ is the mesh width. Note however that for finite dimensional problems Hadamard's third condition is not strictly speaking violated and so (2.1.1) is not ill-posed in the sense of Hadamard, it is only **ill-conditioned**, but it is **asymptotically ill-posed** for $\kappa \to \infty$ (e.g. as $h \to 0$ in the FE problem).

On the positive side, the above expansion shows clearly that errors in the low frequency components $i \ll n$, i.e., the components in the direction of eigenvectors corresponding to the larger eigenvalues, are not amplified as much. This is a typical situation in inverse problems (recall the introductory example in Section 1.1).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why low-frequency components are not amplified as much)</span></p>

Read the expansion term-by-term rather than in aggregate. The component of the error along direction $u_i$ has magnitude

$$\bigl\lvert \text{coef}_i \bigr\rvert = \frac{\lvert u_i^\top (y^\delta - y) \rvert}{\lambda_i},$$

so the noise projected onto mode $i$ (numerator, bounded by $\delta$) gets amplified by the factor $1/\lambda_i$. Which $\lambda_i$ you divide by is the whole story:

- **Small $i$ (i.e. $i \ll n$):** $\lambda_i$ is close to $\lambda_1 = \mathcal{O}(1)$, so $1/\lambda_i = \mathcal{O}(1)$. The noise passes through essentially unamplified.
- **Large $i$ (close to $n$):** $\lambda_i$ is close to $\lambda_n$, which is tiny, so $1/\lambda_i \approx \kappa$ is huge. The noise is blown up by the full condition number.

Concretely: if $\lambda_1 = 1$ and $\lambda_n = 10^{-6}$, then noise in the $u_1$-direction is amplified by $1$, while noise in the $u_n$-direction is amplified by $10^6$. The catastrophe lives at the bottom of the spectrum; the top is fine.

The "low frequency $\leftrightarrow$ large $\lambda_i$" convention comes from smoothing forward operators (e.g. compact integral operators): slowly varying eigenvectors are the ones that survive the forward map best and therefore carry the largest eigenvalues / singular values. This is why the heat-equation example in Section 1.1 loses high-frequency information first.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Linear System: Singular Case)</span></p>

**The singular case.** Let us now consider the case that $A$ in (2.1.1) is positive semi-definite, i.e. it has a nontrivial kernel. Since $A^\ast = A^T = A$, we can decompose the vector space in

$$\mathbb{R}^n = \mathcal{N}(A) + \mathcal{R}(A),$$

where $\mathcal{R}$ is the range and $\mathcal{N}$ is the kernel. Let $\lambda_m$ be the smallest nonzero eigenvalue and let $\kappa_{\text{eff}} = \lambda_1 / \lambda_m$ be the **effective condition number**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Solution in singular case)</span></p>

In the singular case the solution is

$$x = \sum_{i=1}^{m} \lambda_i^{-1} u_i u_i^\top y$$

and the problem is solvable (Hadamard's first condition) **iff** $u_i^\top y = 0$ for $i > m$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Error bound scaling with noise for singular linear systems)</span></p>

In the general noisy case, this will usually not be satisfied, but we can for example project the noisy data $y^\delta$ into the range of $A$ via a projection $P : \mathbb{R}^n \to \mathcal{R}(A)$. Now the problem is solvable and the solution $x_P^\delta$ with data $P y^\delta$ satisfies

$$x_P^\delta - x = \sum_{i=1}^{m} \lambda_i^{-1} u_i u_i^\top (P y^\delta - y).$$

Since by construction $u_i^\top P y^\delta = u_i^\top y^\delta$, for $i \le m$, we have

$$\lVert x_P^\delta - x \rVert \le \lambda_m^{-1} \delta = \mathcal{O}(\kappa_{\text{eff}} \delta).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why $u_i^\top P y^\delta = u_i^\top y^\delta$ for $i \le m$)</span></p>

Since $A$ is symmetric PSD, its eigenvectors split as $u_1,\ldots,u_m$ spanning $\mathcal{R}(A)$ (nonzero eigenvalues) and $u_{m+1},\ldots,u_n$ spanning $\mathcal{N}(A)$, with the two subspaces mutually orthogonal. Any vector decomposes uniquely as

$$v = Pv + (I-P)v,\quad Pv \in \mathcal{R}(A),\ (I-P)v \in \mathcal{N}(A).$$

For $i \le m$, $u_i \in \mathcal{R}(A)$, hence $u_i \perp (I-P)v$ and

$$u_i^\top v = u_i^\top P v + \underbrace{u_i^\top (I-P)v}_{=\,0} = u_i^\top P v.$$

Applied to $v = y^\delta$ this gives the claimed identity. Its role in the bound: plugging into (2.1) yields $u_i^\top(P y^\delta - y) = u_i^\top(y^\delta - y)$, so each coefficient is controlled by the raw noise $\lvert u_i^\top(y^\delta - y) \rvert \le \delta$, and summing with $\lambda_i^{-1} \le \lambda_m^{-1}$ yields the $\mathcal{O}(\kappa_{\text{eff}} \delta)$ bound. The projection $P$ silently killed the kernel-direction noise (indices $i > m$, otherwise uncontrolled) without distorting the range-direction noise.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Error bound scaling with noise for singular linear systems)</span></p>

No (arbitrary) contributions in the kernel components are included and the **error amplification is again determined by the smallest nonzero eigenvalue (or equivalently by the effective condition number)**. 

However, in practice it may be difficult to find $P$ without first performing a spectral decomposition of $A$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Outlook to infinite dimensions)</span></p>

In the general case of a linear operator $A$ between two infinite dimensional Hilbert spaces $X$ and $Y$, the range of $A$ and $A^*$ are not necessarily closed. In that case we have

$$X = \mathcal{N}(A) + \overline{\mathcal{R}(A^*)} \quad \text{and} \quad Y = \mathcal{N}(A^*) + \overline{\mathcal{R}(A)}$$

If the range of $A$ is not closed, i.e., $\overline{\mathcal{R}(A)} \neq \mathcal{R}(A)$, then the projection $P$ is not bounded, which leads again to instabilities. Any operator $A$ with eigenvalues arbitrarily close to $0$ will have this behaviour, in particular every compact operator.

</div>

#### Regularisation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Small eigenvalues cause instabilities - regularization)</span></p>

We saw above that small eigenvalues of $A$ are causing instabilities. A natural approach would thus be to approximate the matrix $A$ with a family of matrices with eigenvalues bounded away from zero. One such family is

$$A_\alpha := A + \alpha I, \qquad \alpha > 0.$$

The eigenvalues of $A_\alpha$ are $\lambda_i + \alpha$, $i = 1, \ldots, n$ and the eigenvectors remain unchanged. (small regularization: the problem is still instable. big regularization: we are solving different problem)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Regularization error of linear system)</span></p>

To estimate the **regularisation error** consider again the regular (SPD) case, i.e. $\lambda_n > 0$ and let $x = A^{-1}y$ and $x_\alpha = A_\alpha^{-1} y$. Then

$$x - x_\alpha = \sum_{i=1}^{n} \left( \frac{1}{\lambda_i} - \frac{1}{\lambda_i + \alpha} \right) u_i u_i^\top y = \sum_{i=1}^{n} \frac{\alpha}{\lambda_i(\lambda_i + \alpha)} (u_i^\top y) \, u_i$$

and using again the Bessel inequality we can estimate the regularisation error by

$$E_\alpha(\alpha) := \lVert x - x_\alpha \rVert \le \frac{\alpha}{\lambda_n(\lambda_n + \alpha)} \lVert y \rVert.$$

In particular, we have $E_\alpha \to 0$ as $\alpha \to 0$. 

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Perturbation error of regularized linear system)</span></p>

In the case of a noisy data $y^\delta$, with $x_\alpha^\delta$ the solution of $A_\alpha x_\alpha^\delta = y^\delta$, the spectral decomposition gives

$$x_\alpha^\delta - x_\alpha = \sum_{i=1}^{n} (\lambda_i + \alpha)^{-1} u_i u_i^\top (y^\delta - y).$$

and thus the **perturbation error** can be estimated by

$$E_\delta(\alpha, \delta) := \lVert x_\alpha^\delta - x_\alpha \rVert \le \frac{\delta}{\lambda_n + \alpha}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Error Bound of regularized noisy estimator)</span></p>

Using the triangle inequality the total error between the exact solution and the solution of the regularised problem with noisy data can be bounded by

$$\lVert x - x_\alpha^\delta \rVert \le E_\alpha(\alpha) + E_\delta(\alpha, \delta) \le \left( \underbrace{\frac{\alpha}{\lambda_n(\lambda_n + \alpha)} \lVert y \rVert}_{\text{from regularization}} + \underbrace{\frac{\delta}{\lambda_n + \alpha}}_{\text{from noise}} \right).$$

</div>

(TODO: I guess after rearrangement we can see the ratio of the observarion error and the estimate error, which is instable in some cases, basically making the loss function useless, because it does not give much signal (low loss, but extremely bad / far estimate of the true value))

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Practical error bound)</span></p>

In practice, the exact data is not known, but we can bound $\lVert y \rVert \le \lVert y^\delta \rVert + \delta$ using (2.1.3) and thus obtain

$$\lVert x - x_\alpha^\delta \rVert \le \left( \frac{\alpha}{\lambda_n(\lambda_n + \alpha)} (1 + \delta_{\text{rel}}) + \frac{\delta_{\text{rel}}}{\lambda_n + \alpha} \right) \lVert y^\delta \rVert,$$

where $\delta_{\text{rel}} = \delta / \lVert y^\delta \rVert$ is the relative noise level (or the inverse signal-to-noise ratio).

</div>

(TODO: derive connection with loss function and lagrangian)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Regularisation Trade-off)</span></p>

For fixed $\delta_{\text{rel}}$ the two terms in the error bound behave very differently with respect to $\alpha$. The first term (regularisation error) decreases monotonically as $\alpha \to 0$ while the second one (perturbation error) grows monotonically. The main task in regularisation is thus to determine the optimal $\alpha$ that minimises the total error, either through an a priori choice $\alpha = \alpha(\delta)$ or through an a posteriori choice $\alpha = \alpha(\delta_{\text{rel}})$ that takes into account the size of the data $\lVert y^\delta \rVert$. Any regularisation strategy needs to satisfy $\alpha(\delta) \to 0$ as $\delta \to 0$, so that in the noise-free case the exact solution is recovered.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Generalization to rectangular matrices)</span></p>

The discussion can easily be generalised also to arbitrary rectangular linear equation systems with $A \in \mathbb{R}^{n \times m}$ (and thus also to arbitrary linear operators between finite dimensional vector spaces of possibly different dimension) by considering the normal equations

$$A^\top A x = A^\top y.$$

However, the ill-conditioning is significantly worse since $\kappa(A^T A) = \kappa(A)^2$.

</div>

### 2.2 Generalised Inverse -- The Infinite Dimensional Setting

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Target $y$ is out of range $A$)</span></p>

In this section, throughout $A \in \mathcal{L}(X, Y)$ is a linear bounded operator between the Hilbert spaces $X$ and $Y$, and we are interested in solutions of the linear operator equation

$$Ax = y \tag{2.2.1}$$

for possibly non-injective and/or non-surjective $A$. 
* For $y \notin \mathcal{R}(A)$, (2.2.1) **has no solution** **(Hadamard 1)**. A sensible thing to do is to find $x \in X$ that minimises $\lVert Ax - y \rVert_Y$. 
* On the other hand, for $\mathcal{N}(A) \neq \lbrace 0 \rbrace$ there are **infinitely many solutions** **(Hadamard 2)**. 
  
In that case, we choose the one that minimises $\lVert x \rVert_X$. This leads to the following definition.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.2.1</span><span class="math-callout__name">(Least-Squares and Minimum-Norm Solutions)</span></p>

An element $x \in X$ is called

* **least-squares solution** of $Ax = y$ (more precisely the $Y$-best approximate solution), if

$$\lVert Ax - y \rVert_Y = \min_{z \in X} \lVert Az - y \rVert_Y,$$

* **minimum-norm** (or $(X,Y)$-best approximate) **solution** of $Ax = y$, if 
  * $x$ is least-squares solution and
  * $\lVert x \rVert_X = \min \lbrace \lVert z \rVert_X : z \text{ is least squares solution of } Az = y \rbrace.$

</div>

For $A$ bijective, $x = A^{-1}y$ is the only minimum-norm solution. However, a minimum-norm solution does not have to exist if $\mathcal{R}(A)$ is not closed. To study which $y \in Y$ admit a minimum-norm solution, we introduce an operator that maps $y$ to the minimum-norm solution; it is called **generalised inverse** or **pseudoinverse**.

To do this we first restrict its domain to the range of $A$ to guarantee invertibility before extending the domain as much as possible.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.2.2</span><span class="math-callout__name">(Moore-Penrose Inverse)</span></p>

Let $A \in \mathcal{L}(X, Y)$ and define

$$\tilde{A} := A\big|_{\mathcal{N}(A)^\perp} : \mathcal{N}(A)^\perp \to \mathcal{R}(A). \tag{2.2.2}$$

The **Moore-Penrose** (or **generalised**) **inverse** $A^\dagger$ is the unique, linear extension of $\tilde{A}^{-1}$ with

$$\mathcal{D}(A^\dagger) := \mathcal{R}(A) \oplus \mathcal{R}(A)^\perp, \quad \text{and} \tag{2.2.3}$$

$$\mathcal{N}(A^\dagger) = \mathcal{R}(A)^\perp.$$

</div>

<!-- <figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/pseudo_four_subspaces.png' | relative_url }}" alt="Decomposition X = N(A)^perp + N(A) and Y = R(A) + R(A)^perp; the restriction Ã: N(A)^perp → R(A) is a bijection inverted by A†, which is zero on R(A)^perp" loading="lazy">
  <figcaption>The geometry behind Definition 2.2.2 (rank-1 example $A=\begin{pmatrix}1&1\\0&0\end{pmatrix}$). Every $x \in X$ splits as $x = x_\perp + x_N$ with $x_\perp \in \mathcal{N}(A)^\perp$ and $x_N \in \mathcal{N}(A)$, and every $y \in Y$ splits along $\mathcal{R}(A) \oplus \mathcal{R}(A)^\perp$. The restriction $\tilde{A} : \mathcal{N}(A)^\perp \to \mathcal{R}(A)$ is bijective; $A^\dagger$ extends $\tilde{A}^{-1}$ by zero on $\mathcal{R}(A)^\perp$.</figcaption>
</figure> -->

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Well-Definedness of the Pseudoinverse)</span></p>

Due to the restriction to $\mathcal{N}(A)^\perp\to\mathcal{R}(A)$ the operator $\tilde{A}$ in (2.2.2) is bijective. For arbitrary $y \in \mathcal{D}(A^\dagger)$, an orthogonal decomposition guarantees the existence of $y_1 \in \mathcal{R}(A)$ and $y_2 \in \mathcal{R}(A)^\perp$ such that $y = y_1 + y_2$. Finally, due to $\mathcal{N}(A^\dagger) = \mathcal{R}(A)^\perp$ we have

$$A^\dagger y = A^\dagger y_1 + A^\dagger y_2 = A^\dagger y_1 = \tilde{A}^{-1} y_1, \tag{2.2.4}$$

and thus $A^\dagger$ is well-defined on all of $\mathcal{D}(A^\dagger)$, defined in (2.2.3).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2.3</span><span class="math-callout__name">(Moore-Penrose Equations)</span></p>

The Moore-Penrose inverse $A^\dagger$ satisfies $\mathcal{R}(A^\dagger) = \mathcal{N}(A)^\perp$, as well as the **Moore-Penrose equations**

1. $AA^\dagger A = A$
2. $A^\dagger A A^\dagger = A^\dagger$
3. $A^\dagger A = \mathrm{Id}\_X - P_{\mathcal{N}}$
4. $AA^\dagger = (P_{\overline{\mathcal{R}}})\big\|_{\mathcal{D}(A^\dagger)}$

where $P_{\mathcal{N}}$ and $P_{\overline{\mathcal{R}}}$ are the orthogonal projections to $\mathcal{N}(A)$ and $\overline{\mathcal{R}(A)}$, respectively. *(The Moore-Penrose equations characterise $A^\dagger$ uniquely.)*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.2.3</summary>

As shown in (2.2.4), for all $y \in \mathcal{D}(A^\dagger)$, it follows that $A^\dagger y \in \mathcal{R}(\tilde{A}^{-1}) = \mathcal{N}(A)^\perp$, i.e., $\mathcal{R}(A^\dagger) \subset \mathcal{N}(A)^\perp$. Conversely, it follows from the definition of $\tilde{A}$ that for all $x \in \mathcal{N}(A)^\perp$

$$A^\dagger A x = A^\dagger \tilde{A} x = \tilde{A}^{-1} \tilde{A} x = x,$$

i.e., $x \in \mathcal{R}(A^\dagger)$. Thus, $\mathcal{R}(A^\dagger) = \mathcal{N}(A)^\perp$.

Since orthogonal projections are always closed, $\mathcal{R}(A)^\perp$ is closed and thus $\mathcal{R}(P_{\overline{\mathcal{R}}}) \cap \mathcal{D}(A^\dagger) = \mathcal{R}(A)$. Thus, for all $y \in \mathcal{D}(A^\dagger)$

$$A^\dagger y = \tilde{A}^{-1} P_{\overline{\mathcal{R}}} y \tag{2.2.5}$$

which due to $\tilde{A}^{-1} P_{\overline{\mathcal{R}}} y \in \mathcal{N}(A)^\perp$ implies $AA^\dagger y = P_{\overline{\mathcal{R}}} y$ and thus equation (iv). The proof of the other three Moore-Penrose equations is left as an exercise.

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/pseudo_mp_projections.png' | relative_url }}" alt="A†A acts as projection onto N(A)^perp (kills the kernel component of x); AA† acts as projection onto R(A) (kills the orthogonal-to-range component of y)" loading="lazy">
  <figcaption>Equations (iii) and (iv) of Theorem 2.2.3 read off as orthogonal projections. <em>Left:</em> $A^\dagger A$ kills the $\mathcal{N}(A)$ component of $x$, so $A^\dagger A x \in \mathcal{N}(A)^\perp$. <em>Right:</em> $AA^\dagger$ kills the $\mathcal{R}(A)^\perp$ component of $y$, so $AA^\dagger y \in \mathcal{R}(A)$. Equations (i) and (ii) are then automatic — $A$ is already injective on $\mathcal{N}(A)^\perp$, and $A^\dagger$ already lives in $\mathcal{N}(A)^\perp$.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2.4</span><span class="math-callout__name">(Minimum-Norm Solution via Pseudoinverse)</span></p>

Let $y \in \mathcal{D}(A^\dagger)$. Then $Ax = y$ has a unique minimum-norm solution $x^\dagger \in X$, which is given by

$$x^\dagger = A^\dagger y.$$

The set of all least-squares solutions is $x^\dagger + \mathcal{N}(A)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.2.4</summary>

To show existence of least-squares solutions consider the set

$$S := \lbrace z \in X : Az = P_{\overline{\mathcal{R}}} y \rbrace,$$

which is non-empty, since $P_{\overline{\mathcal{R}}}$ maps $\mathcal{D}(A^\dagger)$ to $\mathcal{R}(A)$. Let $z \in S$. Then, due to the optimality of the orthogonal projection

$$\lVert Az - y \rVert_Y = \lVert P_{\overline{\mathcal{R}}} y - y \rVert_Y = \min_{w \in \mathcal{R}(A)} \lVert w - y \rVert_Y \le \lVert Ax - y \rVert \quad \text{for all } x \in X,$$

i.e., $z$ is least-squares solution of $Ax = y$. Conversely, let $z \in X$ be a least squares solution. Then it follows again from $P_{\overline{\mathcal{R}}} y \in \mathcal{R}(A)$ that

$$\lVert P_{\overline{\mathcal{R}}} y - y \rVert \leq \lVert Az-y \rVert = \min_{x \in X} \lVert Ax - y \rVert = \min_{w \in \mathcal{R}(A)} \lVert w - y \rVert_Y \leq \lVert P_{\overline{\mathcal{R}}} y - y \rVert_Y,$$

i.e., $Az$ is the orthogonal projection of $y$ onto $\overline{\mathcal{R}(A)}$. In summary,

$$\lbrace x \in X : x \text{ is least squares solution of } Ax = y \rbrace = S \neq \emptyset.$$

Each element $z \in S$ can be decomposed uniquely into $x = \tilde{x} + x_0$ with $\tilde{x} \in \mathcal{N}(A)^\perp$ and $x_0 \in \mathcal{N}(A)$, but we have already seen in (2.2.5) that the unique solution to $Az = P_{\overline{\mathcal{R}}} y$ in $\mathcal{N}(A)^\perp$ is

$$\tilde{x} = \tilde{A}^{-1} P_{\overline{\mathcal{R}}} y = A^\dagger y = x^\dagger.$$

Thus, the set of all least-squares solutions is $x^\dagger + \mathcal{N}(A)$. Finally, due to the orthogonality of $x^\dagger=\tilde{x}$ and $x_0$,

$$\lVert z \rVert_X^2 = \lVert x^\dagger + x_0 \rVert_X^2 = \lVert x^\dagger \rVert_X^2 + \lVert x_0 \rVert_X^2 \ge \lVert x^\dagger \rVert_X^2$$

so that $x^\dagger$ is also the unique minimum-norm solution.

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/pseudo_lsq_minnorm.png' | relative_url }}" alt="Two-step picture of A† y: first orthogonally project y onto closure of R(A) (least-squares), then among the affine line of preimages x† + N(A), pick the one in N(A)^perp closest to the origin (minimum-norm)" loading="lazy">
  <figcaption>Two-step picture of $A^\dagger y$ when both Hadamard conditions fail. <em>Left:</em> the least-squares condition is the orthogonal projection $AA^\dagger y = P_{\overline{\mathcal{R}}}\, y$ — among all elements of $\mathcal{R}(A)$, this is the one closest to $y$. <em>Right:</em> the preimage of $P_{\overline{\mathcal{R}}}\, y$ is the affine line $x^\dagger + \mathcal{N}(A)$ of all least-squares solutions; the minimum-norm one is the foot of the perpendicular from the origin and lies in $\mathcal{N}(A)^\perp$. That foot is $x^\dagger = A^\dagger y$.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2.5</span><span class="math-callout__name">(Normal Equations)</span></p>

Let $y \in \mathcal{D}(A^\dagger)$. Then $x \in X$ is least-squares solution of $Ax = y$ iff $x$ satisfies the **normal equations**

$$A^* A x = A^* y.$$

If in addition $x \in \mathcal{N}(A)^\perp$, then $x = x^\dagger$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.2.5</summary>

By Theorem 2.2.4, $x \in X$ is a least-squares solution of $Ax = y$ iff $Ax = P_{\overline{\mathcal{R}}} y$, which is equivalent to $Ax \in \overline{\mathcal{R}(A)}$ and $Ax - y \in \overline{\mathcal{R}(A)}^\perp = \mathcal{N}(A^\ast)$. This in turn is equivalent to $A^\ast(Ax - y) = 0$, i.e. to the normal equations $A^\ast A x = A^\ast y$.

The final part — that $x \in \mathcal{N}(A)^\perp$ implies $x = x^\dagger$ — was already established in the proof of Theorem 2.2.4: among all least-squares solutions $x^\dagger + \mathcal{N}(A)$, the unique one in $\mathcal{N}(A)^\perp$ is $x^\dagger = A^\dagger y$.

</details>
</div>

The minimum-norm solution $x^\dagger$ of $Ax = y$ is the solution of the normal equations with minimum norm, i.e.,

$$x^\dagger = (A^* A)^\dagger A^* y.$$

So far we have considered the generalised inverse on $\mathcal{D}(A^\dagger) = \mathcal{R}(A) \oplus \mathcal{R}(A)^\perp$ without studying this domain in detail. Since orthogonal complements are always closed,

$$\overline{\mathcal{D}(A^\dagger)} = \overline{\mathcal{R}(A)} \oplus \mathcal{R}(A)^\perp = \mathcal{N}(A^*)^\perp \oplus \mathcal{N}(A^*) = Y,$$

i.e., $\mathcal{D}(A^\dagger)$ is dense in $Y$. Thus, $\mathcal{D}(A^\dagger) = Y$ iff $\mathcal{R}(A)$ is closed. Furthermore, for any $y \in \mathcal{R}(A)^\perp = \mathcal{N}(A^\dagger)$ the minimum-norm solution is $x^\dagger = 0$.

The central question is if $\mathcal{R}(A)$ is closed. If it is then $A^\dagger$ is even bounded. Conversely, if there exists any $y \in \overline{\mathcal{R}(A)} \setminus \mathcal{R}(A)$, then $A^\dagger$ cannot be bounded.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2.6</span><span class="math-callout__name">(Boundedness of Pseudoinverse)</span></p>

Let $A \in \mathcal{L}(X, Y)$. Then 

$$A^\dagger \in \mathcal{L}(\mathcal{D}(A^\dagger), X) \iff \mathcal{R}(A) \text{ is closed}$$

(In fact, in that case $\mathcal{D}(A^\dagger) = Y$ and thus $A^\dagger \in \mathcal{L}(Y, X)$.)

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.2.6</summary>

We apply the Closed Graph Theorem, and show first that $A^\dagger$ is closed.

Let $(y_n)\_{n \in \mathbb{N}} \subset \mathcal{D}(A^\dagger)$ be a convergent sequence with $y_n \to y \in Y$ and $A^\dagger y_n \to x \in X$. From Moore-Penrose equation (iv) and the continuity of orthogonal projections it follows that

$$AA^\dagger y_n = P_{\overline{\mathcal{R}}} y_n \to P_{\overline{\mathcal{R}}} y,$$

which due to the continuity of $A$ implies

$$P_{\overline{\mathcal{R}}} y = \lim_{n \to \infty} P_{\overline{\mathcal{R}}} y_n = \lim_{n \to \infty} AA^\dagger y_n = Ax,$$

i.e., $x$ is least-squares solution. Furthermore, $A^\dagger y_n \in \mathcal{R}(A^\dagger) = \mathcal{N}(A)^\perp$ and so

$$A^\dagger y_n \to x \in \mathcal{N}(A)^\perp,$$

since $\mathcal{N}(A)^\perp = \overline{\mathcal{R}(A^*)}$ is closed. Hence, $x$ is in fact the minimum-norm solution of $Ax = y$, i.e., $x = A^\dagger y$, and $A^\dagger$ is closed.

Now let $\mathcal{R}(A)$ be closed. Then $\mathcal{D}(A^\dagger) = Y$ and Theorem A.1.2 implies that $A^\dagger : Y \to X$ is bounded. Conversely, let $A^\dagger$ be bounded on $\mathcal{D}(A^\dagger)$. In that case, since $\mathcal{D}(A^\dagger)$ is dense in $Y$, $A^\dagger$ can be continuously extended to an operator $\overline{A^\dagger} \in \mathcal{L}(Y, X)$ by defining

$$\overline{A^\dagger} y := \lim_{n \to \infty} A^\dagger y_n \quad \text{for some sequence } (y_n)_{n \in \mathbb{N}} \subset \mathcal{D}(A^\dagger) \text{ with } y_n \to y \in Y.$$

Due to its continuity, $A^\dagger$ maps Cauchy sequences to Cauchy sequences, and thus $\overline{A^\dagger}$ is well-defined and bounded. Now let $y \in \overline{\mathcal{R}(A)}$ and $(y_n)\_{n \in \mathbb{N}} \subset \mathcal{R}(A)$ with $y_n \to y$. It follows from Moore-Penrose equation (iv) and the continuity of $A$ that

$$y = P_{\overline{\mathcal{R}}} y = \lim_{n \to \infty} P_{\overline{\mathcal{R}}} y_n = \lim_{n \to \infty} AA^\dagger y_n = A\overline{A^\dagger} y \in \mathcal{R}(A),$$

and thus $\overline{\mathcal{R}(A)} = \mathcal{R}(A)$, which completes the proof.

</details>
</div>

Unfortunately this excludes the most interesting case of a compact operator on a Hilbert space.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.2.7</span><span class="math-callout__name">(Compact Operators Have Unbounded Pseudoinverse)</span></p>

Let $K \in \mathcal{K}(X, Y)$, i.e., $K$ is compact, with infinite dimensional image $\mathcal{R}(K)$. Then $K^\dagger$ is **not** bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 2.2.7</summary>

Suppose that $K^\dagger$ is bounded. Then it follows from Theorem 2.2.6 that $\mathcal{R}(K)$ is closed and

$$\tilde{K} := K\big|_{\mathcal{N}(K)^\perp} : \mathcal{N}(K)^\perp \to \mathcal{R}(K)$$

is a bijective operator with bounded inverse $\tilde{K}^{-1} \in \mathcal{L}(\mathcal{R}(K), \mathcal{N}(K)^\perp)$. Since $K$ is compact, $K \circ \tilde{K}^{-1}$ is also compact. But $K \circ \tilde{K}^{-1}$ is the identity on $\mathcal{R}(K)$, which can only be compact iff $\mathcal{R}(K)$ is finite dimensional.

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/pseudo_compact_unbounded.png' | relative_url }}" alt="Singular values of a compact operator decay to zero, so the amplification factors 1/sigma_n diverge; a Cauchy sequence y^(N) in Y has bounded norm but ||A† y^(N)|| diverges, exhibiting an element of the closure of R(A) outside R(A)" loading="lazy">
  <figcaption>Why compact operators with infinite-dimensional range cannot have a bounded pseudoinverse. <em>Left:</em> $\sigma_n \to 0$ forces the per-mode amplification $1/\sigma_n \to \infty$ (here $\sigma_n = 1/n$). <em>Right:</em> a concrete witness — the partial sums $y^{(N)} = \sum_{n=1}^N n^{-3/2} u_n$ are Cauchy and converge in $Y$ (the norm plateaus at $\sqrt{\zeta(3)}$), yet $\| A^\dagger y^{(N)} \|_X = \sqrt{H_N} \sim \sqrt{\log N}$ diverges. The limit therefore lives in $\overline{\mathcal{R}(A)} \setminus \mathcal{R}(A)$ — exactly the obstruction that Theorem 2.2.6 forbids in the bounded case.</figcaption>
</figure>

### 2.3 Singular Value Decomposition of Compact Operators

We now use an orthonormal system to characterise the Moore-Penrose inverse of compact operators $K \in \mathcal{K}(X, Y)$. To do this for general non-selfadjoint operators we need to generalise the Spectral Theorem. Because of Theorem 2.2.5 we can look at the selfadjoint operator $K^* K$ instead. This leads to the singular value decomposition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.3.1</span><span class="math-callout__name">(Singular Value Decomposition)</span></p>

Let $K \in \mathcal{K}(X, Y)$ with infinite dimensional range $\mathcal{R}(K)$. Then there exists

1. a (null) sequence $(\sigma_n)\_{n \in \mathbb{N}}$ with $\sigma_1 \ge \sigma_2 \ge \ldots > 0$ and $\sigma_n \to 0$ as $n \to \infty$,
2. an orthonormal basis $(u_n)\_{n \in \mathbb{N}} \subset Y$ of $\overline{\mathcal{R}(K)}$, and
3. an orthonormal basis $(v_n)\_{n \in \mathbb{N}} \subset X$ of $\overline{\mathcal{R}(K^*)}$,

   with

   $$Kv_n = \sigma_n u_n \quad \text{and} \quad K^* u_n = \sigma_n v_n, \quad \text{for all } n \in \mathbb{N}, \tag{2.3.1}$$

   and

   $$Kx = \sum_{n \in \mathbb{N}} \sigma_n \langle x, v_n \rangle_X u_n, \quad \text{for all } x \in X. \tag{2.3.2}$$

A sequence $(\sigma_n, u_n, v_n)\_{n \in \mathbb{N}}$ that provides such a **singular value decomposition (SVD)** (2.3.2) of $K$, is called **singular system**.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.3.1</summary>

Since $K^* K : X \to X$ is compact and selfadjoint, it follows from Theorem A.2.6 that there exists a null sequence $(\lambda_n)\_{n \in \mathbb{N}} \subset \mathbb{R} \setminus \lbrace 0 \rbrace$ and an orthonormal system $(v_n)\_{n \in \mathbb{N}} \subset X$ such that

$$K^* K x = \sum \lambda_n \langle x, v_n \rangle_X v_n \quad \text{for all } x \in X.$$

Moreover, $(v_n)$ is an ONB of $\overline{\mathcal{R}(K^* K)}$.

Now, since $\lambda_n = \lambda_n \lVert v_n \rVert_X^2 = \langle \lambda_n v_n, v_n \rangle_X = \langle K^* K v_n, v_n \rangle_X = \lVert K v_n \rVert_X^2 > 0$, we can define for all $n \in \mathbb{N}$

$$\sigma_n := \sqrt{\lambda_n} > 0 \quad \text{and} \quad u_n := \frac{1}{\sigma_n} K v_n \in Y$$

so that $(\sigma_n)$ is a strictly positive null sequence and the first equation in (2.3.1) is satisfied. Moreover,

$$\langle u_i, u_j \rangle_Y = \frac{1}{\sigma_i \sigma_j} \langle K v_i, K v_j \rangle_Y = \frac{1}{\sigma_i \sigma_j} \langle K^* K v_i, v_j \rangle_X = \frac{\lambda_i}{\sigma_i \sigma_j} \langle v_i, v_j \rangle_X = \begin{cases} 1, & \text{if } i = j, \\ 0, & \text{otherwise,} \end{cases}$$

and thus $(u_n)$ is an orthonormal system in $Y$. Furthermore, for all $n \in \mathbb{N}$,

$$K^* u_n = \sigma_n^{-1} K^* K v_n = \sigma_n^{-1} \lambda_n v_n = \sigma_n v_n,$$

i.e., the second equation in (2.3.1) holds.

To show that $(v_n)$ is not only an ONB of $\overline{\mathcal{R}(K^* K)}$ but also of $\overline{\mathcal{R}(K^\ast)}$, it suffices to show that $\overline{\mathcal{R}(K^\ast)} \subset \overline{\mathcal{R}(K^\ast K)}$. Let $x \in \overline{\mathcal{R}(K^\ast)}$. For any $\epsilon > 0$, there exists a

$$y \in \mathcal{N}(K^*)^\perp = \overline{\mathcal{R}(K)} \text{ with } \lVert K^* y - x \rVert_X < \frac{\epsilon}{2} \quad \text{and} \quad \tilde{x} \in X \text{ with } \lVert K\tilde{x} - y \rVert < \frac{\epsilon}{2} \lVert K \rVert_{\mathcal{L}(X,Y)}^{-1},$$

such that $\lVert K^* K\tilde{x} - x \rVert_X \le \lVert K^* K\tilde{x} - K^* y \rVert_X + \lVert K^* y - x \rVert_X < \epsilon$ and thus $x \in \overline{\mathcal{R}(K^* K)}$.

To prove the SVD (2.3.2) consider first an arbitrary $\tilde{x} \in \mathcal{N}(K)^\perp$ and

$$\tilde{x}_N := \sum_{j=1}^{N} \langle \tilde{x}, v_j \rangle_X v_j,$$

i.e., the partial basis representation of $\tilde{x}$ with respect to the ONB $(v_n)$ of $\overline{\mathcal{R}(K^*)} = \mathcal{N}(K)^\perp$. Clearly

$$K\tilde{x}_N = \sum_{j=1}^{N} \langle \tilde{x}, v_j \rangle_X K v_j = \sum_{j=1}^{N} \sigma_j \langle \tilde{x}, v_j \rangle_X u_j.$$

Since $\tilde{x}_N \to \tilde{x}$ and $K$ is bounded,

$$K\tilde{x} = \lim_{N \to \infty} K\tilde{x}_N = \sum_{j=1}^{\infty} \sigma_j \langle \tilde{x}, v_j \rangle_X u_j. \tag{2.3.3}$$

Now, let $x \in X$ be arbitrary. Then, there exist unique $\tilde{x} \in \mathcal{N}(K)^\perp$, $x_0 \in \mathcal{N}(K)$ such that $x = \tilde{x} + x_0$ and

$$\sigma_j \langle x, v_j \rangle_X = \langle x, K^* u_j \rangle_X = \langle Kx, u_j \rangle_Y = \langle K\tilde{x}, u_j \rangle_Y = \sigma_j \langle \tilde{x}, v_j \rangle_X.$$

Substituting this into (2.3.3) and using the fact that $Kx = K\tilde{x}$ leads to the SVD (2.3.2).

Finally, to show that $(u_n) \subset Y$ is an ONB of $\overline{\mathcal{R}(K)}$ let $y \in \overline{\mathcal{R}(K)}$ be arbitrary. Then, there exists a sequence $(x_n) \subset X$ such that

$$y = \lim_{n \to \infty} Kx_n = \lim_{n \to \infty} \sum_{j=1}^{\infty} \langle Kx_n, u_j \rangle_Y u_j = \sum_{j=1}^{\infty} \langle y, u_j \rangle_Y u_j \quad \text{and} \quad \lVert y \rVert_Y^2 = \sum_{j=1}^{\infty} |\langle y, u_j \rangle_Y|^2.$$

This implies that $(u_n)$ is an ONB of $\overline{\mathcal{R}(K)}$.

</details>
</div>

Since eigenvalues $\lambda_n$ of $K^\ast K$ with eigenvector $v_n$ are eigenvalues of $KK^\ast$ with eigenvector $u_n$ as well, (2.3.1) also provides a SVD of $K^\ast$:

$$K^* y = \sum_{n \in \mathbb{N}} \sigma_n \langle y, u_n \rangle_Y v_n \quad \text{for all } y \in Y.$$

We will now use the SVD of $K$ to characterise the domain $\mathcal{D}(K^\dagger) = \mathcal{R}(K) \oplus \mathcal{R}(K)^\perp$ of the Moore-Penrose inverse $K^\dagger$. Recall that minimum-norm solution for $y \in \mathcal{R}(K)^\perp = \mathcal{N}(K^\ast)$ is $x^\dagger = 0$, and conversely $\mathcal{N}(K^\ast)^\perp = \overline{\mathcal{R}(K)}$. Thus, the crucial question is whether an element $y \in \overline{\mathcal{R}(K)}$ also lies in $\mathcal{R}(K)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.3.2</span><span class="math-callout__name">(Picard Condition)</span></p>

Let $K \in \mathcal{K}(X, Y)$ with singular system $(\sigma_n, u_n, v_n)\_{n \in \mathbb{N}}$ and $y \in \overline{\mathcal{R}(K)}$. Then, $y \in \mathcal{R}(K)$ iff the **Picard-condition**

$$\sum_{n \in \mathbb{N}} \sigma_n^{-2} |\langle y, u_n \rangle_Y|^2 < \infty \tag{2.3.4}$$

is satisfied. In this case

$$K^\dagger y = \sum_{n \in \mathbb{N}} \sigma_n^{-1} \langle y, u_n \rangle_Y v_n. \tag{2.3.5}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.3.2</summary>

First let $y \in \mathcal{R}(K)$, i.e. there exists a $x \in X$ with $Kx = y$. Then, using Bessel's inequality

$$\sum_{n \in \mathbb{N}} \sigma_n^{-2} |\langle y, u_n \rangle_Y|^2 = \sum_{n \in \mathbb{N}} \sigma_n^{-2} |\langle x, K^* u_n \rangle_X|^2 = \sum_{n \in \mathbb{N}} |\langle x, v_n \rangle_X|^2 \le \lVert x \rVert_X^2 < \infty.$$

To show the reverse implication, let $y \in \overline{\mathcal{R}(K)}$ and suppose that (2.3.4) holds. Then, the sequence $(s_N)\_{N \in \mathbb{N}}$ of partial sums $s_N := \sum_{n=1}^{N} \sigma_n^{-2} \|\langle y, u_n \rangle_Y\|^2$ is a Cauchy sequence and thus

$$(x_N)_{N \in \mathbb{N}} \quad \text{with} \quad x_N := \sum_{n=1}^{N} \sigma_n^{-1} \langle y, u_n \rangle_Y v_n$$

is also a Cauchy sequence. In other words,

$$\lVert x_N - x_M \rVert_X^2 = \left\lVert \sum_{n=N+1}^{M} \sigma_n^{-1} \langle y, u_n \rangle_Y v_n \right\rVert_X^2 = \sum_{n=N+1}^{M} \sigma_n^{-2}|\langle y, u_n \rangle_Y|^2 = s_M - s_N \to 0,$$

where we used that $(v_n)\_{n \in \mathbb{N}}$ is an orthonormal system in $\overline{\mathcal{R}(K^\ast)}$. Thus, $(x_N)\_{N \in \mathbb{N}} \subset \overline{\mathcal{R}(K^\ast)}$ converges to

$$x := \sum_{n \in \mathbb{N}} \sigma_n^{-1} \langle y, u_n \rangle_Y v_n \in \overline{\mathcal{R}(K^*)} = \mathcal{N}(K)^\perp$$

(since $\overline{\mathcal{R}(K^*)}$ is closed). Now,

$$Kx = \sum_{n \in \mathbb{N}} \sigma_n^{-1} \langle y, u_n \rangle_Y K v_n = \sum_{n \in \mathbb{N}} \langle y, u_n \rangle_Y u_n = P_{\overline{\mathcal{R}(K)}} y = y,$$

so that $y \in \mathcal{R}(K)$.

However, due to Theorem 2.2.4, $x \in \mathcal{N}(K)^\perp$ and $Kx = P_{\overline{\mathcal{R}(K)}} y$ is equivalent to $K^\dagger y = x$.

</details>
</div>

The Picard-condition states that a minimum-norm solution exists only if the coefficients $\langle y, u_n \rangle_Y$ of $y$ with respect to the ONB $(u_n)$ decay faster than the singular values $\sigma_n$. The representation shows clearly how perturbations of $y$ will affect $x^\dagger$: In particular, if $y^\delta = y + \delta u_n$ then

$$\lVert K^\dagger y^\delta - K^\dagger y \rVert_X = \delta \lVert K^\dagger u_n \rVert_X = \sigma_n^{-1} \delta \to \infty \quad \text{for } n \to \infty.$$

Therefore, the faster the singular values decay the more data errors are amplified for a fixed $n$. We call a problem

* **moderately ill-posed**, if there are $c, r > 0$ such that $\sigma_n \ge c n^{-r}$ for all $n \in \mathbb{N}$,
* **severely ill-posed**, if this is not the case, and
* **exponentially ill-posed**, if there are $c, r > 0$ such that $\sigma_n \le c e^{-n^r}$ for all $n \in \mathbb{N}$.

For exponentially ill-posed problems, such as the inverse heat equation in Section 1.1, we can typically expect only very crude estimates for the solution. However, if $\mathcal{R}(K)$ is finite dimensional, the sequence $(\sigma_n)$ truncates at a finite $N$, i.e. $\sigma_n = 0$ for $n > N$ and the error remains bounded; in this case $K^\dagger$ is bounded.

In practice, infinite dimensional problems typically need to be discretised. In general, integral equations or differential equations can not be solved explicitly like the simple one-dimensional inverse heat equation in Section 1.1. So strictly speaking, in practice we will always solve finite dimensional inverse problems. But the problem will be asymptotically ill-posed as the discretisation parameter $h \to 0$, and so we need to find a way to deal with this ill-posedness in a more uniform way, independently of $h$. One way to achieve this is via **regularisation** which we will now discuss in the linear case. The other is to apply a statistical (**Bayesian**) approach, which we will return to in Chapter 4.

### 2.4 Regularisation

We have seen that for $y \in \mathcal{D}(A^\dagger)$ the minimum-norm solution $x^\dagger = A^\dagger y$ of the ill-posed operator equation $Ax = y$ exists. Now consider, as in Section 2.1, the situation where $y$ is known only up to the measurement (or representation) error $\delta$ (the **noise level**), i.e. we only know $y^\delta$ with

$$\lVert y^\delta - y \rVert_Y \le \delta.$$

Since $A^\dagger$ is in general not bounded, $A^\dagger y^\delta$ will normally be a bad approximation to $x^\dagger$, even if $y^\delta \in \mathcal{D}(A^\dagger)$. Thus, in a **regularisation method** we will typically aim to find an approximation $x_\alpha^\delta$, that depends on the one hand continuously on $y^\delta$ and thus on $\delta$, and on the other hand can be selected as close to $x^\dagger$ as the noise level $\delta$ allows by a judicious choice of the **regularisation parameter** $\alpha > 0$. In particular, the choice of $\alpha(\delta)$ should guarantee that $x_{\alpha(\delta)}^\delta \to x^\dagger$ as $\delta \to 0$.

In the case of a linear operator on a Hilbert space this can be achieved by defining a family of regularisation operators that provide bounded replacements of the unbounded pseudoinverse $A^\dagger$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.4.1</span><span class="math-callout__name">(Regularisation)</span></p>

Let $X, Y$ be two Hilbert spaces and $A \in \mathcal{L}(X, Y)$ a bounded, linear operator. A family $(A_\alpha^\dagger)\_{\alpha > 0}$ of linear operators $A_\alpha^\dagger : Y \to X$ is called a **regularisation** of $A^\dagger$ for $\alpha > 0$ if

1. $A_\alpha^\dagger \in \mathcal{L}(Y, X)$ for all $\alpha > 0$,
2. $A_\alpha^\dagger y \to A^\dagger y$ for all $y \in \mathcal{D}(A^\dagger)$, as $\alpha \to 0$.

</div>

Thus, a regularisation is a pointwise approximation of the Moore-Penrose inverse by a sequence of bounded, linear operators. Since $A^\dagger$ is in general not bounded, it follows from Theorem A.1.4 (Banach-Steinhaus) that the convergence will in general **not** be uniform.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.4.2</span><span class="math-callout__name">(Non-Uniform Boundedness)</span></p>

Let $A \in \mathcal{L}(X, Y)$ and $(A_\alpha^\dagger)\_{\alpha > 0} \subset \mathcal{L}(Y, X)$ a regularisation. If $A^\dagger$ is unbounded then the family $(A_\alpha^\dagger)\_{\alpha > 0}$ is not uniformly bounded. In particular, there exists a $y \in Y$ such that $\lVert A_\alpha^\dagger y \rVert_X \to \infty$ as $\alpha \to 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.4.3</span><span class="math-callout__name">(Divergence Outside Domain)</span></p>

Let $A \in \mathcal{L}(X, Y)$ with $A^\dagger$ unbounded and $(A_\alpha^\dagger)\_{\alpha > 0} \subset \mathcal{L}(Y, X)$ a regularisation. If

$$\sup_{\alpha > 0} \lVert A A_\alpha^\dagger \rVert_{\mathcal{L}(Y,Y)} < \infty, \tag{2.4.1}$$

then $\lVert A_\alpha^\dagger y \rVert_X \to \infty$ as $\alpha \to 0$ for all $y \notin \mathcal{D}(A^\dagger)$.

</div>

Since in general $y^\delta \notin \mathcal{D}(A^\dagger)$, to analyse the total error we decompose it as

$$\lVert A_\alpha^\dagger y^\delta - A^\dagger y \rVert_X \le \lVert A_\alpha^\dagger y^\delta - A_\alpha^\dagger y \rVert_X + \lVert A_\alpha^\dagger y - A^\dagger y \rVert_X \le \delta \lVert A_\alpha^\dagger \rVert_{\mathcal{L}(Y,X)} + \lVert A_\alpha^\dagger y - A^\dagger y \rVert_X. \tag{2.4.2}$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Error Decomposition)</span></p>

This decomposition is a fundamental tool of regularisation theory that will be used throughout. The first term represents the (propagated) **data error** that remains unbounded for $\alpha \to 0$ while $\delta > 0$. The second term is the **regularisation error** that due to the pointwise convergence of $A_\alpha^\dagger$ converges to zero as $\alpha \to 0$. Thus, to obtain a meaningful approximation, the regularisation parameter $\alpha$ has to be chosen correctly as a function of $\delta$, in particular such that the total error converges to zero as $\delta \to 0$.

</div>

#### 2.4.1 Parameter Choice

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.4.4</span><span class="math-callout__name">(Parameter Choice Rule)</span></p>

A function $\alpha : \mathbb{R}\_{+} \times Y \to \mathbb{R}\_{+}$, $(\delta, y^\delta) \mapsto \alpha(\delta, y^\delta)$ is called **parameter choice rule**. A regularisation $(A_\alpha^\dagger)\_{\alpha > 0} \subset \mathcal{L}(Y, X)$ of $A^\dagger$ together with a parameter choice rule $\alpha$ is called a **regularisation method** of (2.2.1). The regularisation method $(A_\alpha^\dagger, \alpha)$ is called **convergent** if

$$\lim_{\delta \to 0} \sup \lbrace \lVert A_{\alpha(\delta, y^\delta)}^\dagger y^\delta - A^\dagger y \rVert_X : y^\delta \in Y, \lVert y^\delta - y \rVert_Y \le \delta \rbrace = 0, \quad \text{for all } y \in \mathcal{D}(A^\dagger). \tag{2.4.3}$$

</div>

We distinguish between

* **a priori parameter choice rules** that only depend on $\delta$;
* **a posteriori parameter choice rules** that depend on $\delta$ and $y^\delta$;
* **heuristic rules** that only depend on $y^\delta$.

It can be shown that for all regularisations there exists an a priori rule and thus a convergent regularisation method.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.4.5</span><span class="math-callout__name">(Convergent A Priori Rule)</span></p>

Let $(A_\alpha^\dagger)\_{\alpha > 0}$ be a regularisation and $\alpha : \mathbb{R}\_{+} \to \mathbb{R}\_{+}$ an a-priori rule with

1. $\lim_{\delta \to 0} \alpha(\delta) = 0$,
2. $\lim_{\delta \to 0} \delta \lVert A_{\alpha(\delta)}^\dagger \rVert_{\mathcal{L}(Y,X)} = 0$.

Then $(A_\alpha^\dagger, \alpha)$ is a convergent regularisation method.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.4.5</summary>

Due to the decomposition (2.4.2) it follows that

$$\lVert A_{\alpha(\delta)}^\dagger y^\delta - A^\dagger y \rVert_X \le \delta \lVert A_{\alpha(\delta)}^\dagger \rVert_{\mathcal{L}(Y,X)} + \lVert A_{\alpha(\delta)}^\dagger y - A^\dagger y \rVert_X \to 0 \quad \text{as } \delta \to 0,$$

where we have used (ii) and the pointwise convergence of the regularisation operators under condition (i).

</details>
</div>

Let $y \in \mathcal{D}(A^\dagger)$ and $y^\delta \in Y$ with $c\delta \le \lVert y^\delta - y \rVert_Y \le \delta$ for some $0 < c \le 1$. The main idea of a-posteriori rules can be described as follows: for $x_\alpha^\delta := A_\alpha^\dagger y^\delta$ we consider the **residual**

$$\lVert A x_\alpha^\delta - y^\delta \rVert_Y.$$

Even for $y \in \mathcal{R}(A)$ and the minimum-norm solution $Ax^\dagger = y$ we only have

$$\lVert A x^\dagger - y^\delta \rVert_Y = \lVert y - y^\delta \rVert_Y \ge c\delta.$$

Thus, it makes no sense to expect a smaller residual for the approximation $x_\alpha^\delta$. This motivates:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.4.6</span><span class="math-callout__name">(Discrepancy Principle of Morozov)</span></p>

Given $\delta > 0$ and $y^\delta$, choose $\alpha = \alpha(\delta, y^\delta)$ such that

$$\lVert A x_\alpha^\delta - y^\delta \rVert_Y \le \tau \delta \qquad \text{for some } \tau > 1. \tag{2.4.4}$$

</div>

This principle does not have to be satisfiable: if $y \in \mathcal{D}(A^\dagger)$ such that $y = Ax + y^\perp$ for some $x \in X$ and $0 \neq y^\perp \in \mathcal{R}(A)^\perp$ and $\delta < \frac{1}{2} \lVert y^\perp \rVert_Y$, then even for exact data $y^\delta = y$,

$$\lVert Ax^\dagger - y \rVert_Y = \lVert AA^\dagger y - y \rVert_Y = \lVert P_{\overline{\mathcal{R}(A)}} y - y \rVert_Y = \lVert y^\perp \rVert_Y > 2\delta.$$

Thus, we have to assume that this is not possible. It suffices to assume that $\mathcal{R}(A)$ is dense in $Y$, since in that case $\mathcal{R}(A)^\perp = \lbrace 0 \rbrace$.

A practical approach to implement such an a posteriori rule is to choose a null sequence $(\alpha_n)\_{n \in \mathbb{N}}$, to successively calculate $x_{\alpha_n}^\delta$ for $n = 1, \ldots$ and to terminate the iteration as soon as the discrepancy principle (2.4.4) is satisfied. The following theorem justifies this approach.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.4.7</span><span class="math-callout__name">(Discrepancy Principle Termination)</span></p>

Let $(A_\alpha^\dagger)\_{\alpha > 0}$ be a regularisation of $A \in \mathcal{L}(X, Y)$ with $\mathcal{R}(A)$ dense in $Y$, and suppose that the family $(AA_\alpha^\dagger)\_{\alpha > 0}$ is uniformly bounded. Consider a strictly monotonic null sequence $(\alpha_n)\_{n \in \mathbb{N}}$ and $\tau > 1$. Then for all $y \in \mathcal{D}(A^\dagger)$ and $y^\delta \in Y$ with $\lVert y - y^\delta \rVert_Y \le \delta$ there exists $n^\ast \in \mathbb{N}$ such that

$$\lVert A x_{\alpha_{n^*}}^\delta - y^\delta \rVert_Y \le \tau \delta < \lVert A x_{\alpha_n}^\delta - y^\delta \rVert_Y \qquad \text{for all } n < n^*.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.4.7</summary>

For all $y \in \mathcal{D}(A^\dagger)$, $AA_\alpha^\dagger y$ converges pointwise to $AA^\dagger y = P_{\overline{\mathcal{R}}} y$. Thus, due to the uniform boundedness of $(AA_\alpha^\dagger)\_{\alpha > 0}$ this convergence extends to all $y \in Y = \overline{\mathcal{D}(A^\dagger)}$. This implies for all $y \in \mathcal{D}(A^\dagger) = \mathcal{R}(A)$ and $y^\delta \in Y$ with $\lVert y^\delta - y \rVert_Y \le \delta$ that

$$\lim_{n \to \infty} \lVert A x_{\alpha_n}^\delta - y^\delta \rVert_Y = \lim_{n \to \infty} \lVert AA_{\alpha_n}^\dagger y^\delta - y^\delta \rVert_Y = \lVert P_{\overline{\mathcal{R}}} y^\delta - y^\delta \rVert_Y = \min_{z \in \mathcal{R}(A)} \lVert z - y^\delta \rVert_Y \le \lVert y - y^\delta \rVert_Y \le \delta.$$

The existence of an $n^* \in \mathbb{N}$ then follows directly, since $\tau > 1$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.4.8</span><span class="math-callout__name">(Bakushinskii Veto)</span></p>

*(Bakushinskii, 1985).* Let $(A_\alpha^\dagger)\_{\alpha > 0}$ be a regularisation. If there exists a heuristic parameter choice rule $\alpha \neq \alpha(\delta)$ such that $(A_\alpha^\dagger, \alpha)$ is a convergent regularisation method, then $A^\dagger$ is bounded.

</div>

Heuristic rules do not even assume any knowledge of the noise level $\delta$, which is highly relevant in practice, since often it is hard or impossible to estimate $\delta$ accurately. However, the Bakushinskii veto shows that such a strategy cannot work in general.

#### 2.4.2 Construction of Regularisation Methods

Let us now consider the construction of regularisation methods for linear ill-posed problems. We focus on compact operators $K \in \mathcal{K}(X, Y)$ and recall that stability issues with the Moore-Penrose inverse arose from error amplification through small singular values. Therefore, we aim to construct regularisation methods in such a way that they modify the smallest singular values appropriately.

Thus, recall the SVD of $K^\dagger$ with respect to the singular system $(\sigma_n, u_n, v_n)\_{n \in \mathbb{N}}$ of $K$. It suggests to construct regularisation operators of the form

$$K_\alpha^\dagger y := \sum_{n=1}^{\infty} g_\alpha(\sigma_n) \langle y, u_n \rangle_Y v_n \qquad \text{for } y \in Y,$$

with a suitable function $g_\alpha : \mathbb{R}\_{+} \to \mathbb{R}\_{+}$ that satisfies $g_\alpha(\sigma) \to \frac{1}{\sigma}$ for all $\sigma > 0$ as $\alpha \to 0$. We will see that $(K_\alpha^\dagger)\_{\alpha \ge 0}$ is a regularisation if

$$g_\alpha(\sigma) \le C_\alpha < \infty, \qquad \text{for all } \sigma > 0. \tag{2.4.5}$$

Note that (2.4.5) implies

$$\lVert K_\alpha^\dagger y \rVert_X^2 = \sum_{n=1}^{\infty} (g_\alpha(\sigma_n))^2 |\langle y, u_n \rangle_Y|^2 \le C_\alpha^2 \sum_{n=1}^{\infty} |\langle y, u_n \rangle_Y|^2 \le C_\alpha^2 \lVert y \rVert_Y^2,$$

i.e. $C_\alpha$ is a bound for the norm of $K_\alpha^\dagger$. Moreover, the condition $\lim_{\delta \to 0} \delta \lVert K_{\alpha(\delta)}^\dagger \rVert_{\mathcal{L}(Y,X)} = 0$ can be replaced by

$$\lim_{\delta \to 0} \delta C_{\alpha(\delta)} = 0.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.4.9</span><span class="math-callout__name">(Truncated SVD)</span></p>

Here, all singular values smaller than a prescribed value (controlled by $\alpha$) are ignored (i.e. set to 0). We choose

$$g_\alpha(\sigma) := \begin{cases} \frac{1}{\sigma}, & \text{if } \sigma \ge \alpha, \\ 0, & \text{otherwise.} \end{cases} \tag{2.4.6}$$

Clearly, $g_\alpha(\sigma) \to \frac{1}{\sigma}$ as $\alpha \to 0$ and $C_\alpha = \frac{1}{\alpha}$. Thus, this regularisation with a-priori parameter choice rule leads to a convergent regularisation method provided $\frac{\delta}{\alpha} \to 0$. Furthermore, $\sup_{\sigma, \alpha} \sigma g_\alpha(\sigma) = 1$.

The regularised solution is

$$x_\alpha^\delta := K_\alpha^\dagger y^\delta = \sum_{\sigma_n \ge \alpha} \frac{1}{\sigma_n} \langle y^\delta, u_n \rangle_Y v_n, \qquad \text{for } y^\delta \in Y,$$

motivating the name of the method. The sum in $x_\alpha^\delta$ is always finite for $\alpha > 0$, since zero is the only accumulation point of the sequence $(\sigma_n)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.4.10</span><span class="math-callout__name">(Lavrentiev Regularisation)</span></p>

Here, all singular values are shifted away from zero by $\alpha$, i.e. $g_\alpha(\sigma) = \frac{1}{\sigma + \alpha}$, and

$$x_\alpha^\delta := K_\alpha^\dagger y^\delta = \sum_{n=1}^{\infty} \frac{1}{\sigma_n + \alpha} \langle y^\delta, u_n \rangle_Y v_n, \qquad \text{for } y^\delta \in Y.$$

The computation of this approximation requires an explicit knowledge of the singular system $(\sigma_n, u_n, v_n)\_{n \in \mathbb{N}}$ of $K$, which is not very useful in practice. However, for selfadjoint, positive semidefinite operators $K$ (i.e. $Y = X$, $\lambda_n = \sigma_n$ and $u_n = v_n$) we have

$$(K + \alpha I)x_\alpha^\delta = \sum_{n=1}^{\infty} (\sigma_n + \alpha) \langle x_\alpha^\delta, u_n \rangle_X u_n = \sum_{n=1}^{\infty} \langle y^\delta, u_n \rangle_Y u_n = y^\delta$$

and the regularised solution can be found without explicit knowledge of the singular system of $K$ by solving

$$(K + \alpha I) x_\alpha^\delta = y^\delta.$$

Since $\frac{1}{\sigma + \alpha} \le \frac{1}{\alpha}$, we have again $C_\alpha = \frac{1}{\alpha}$. Furthermore, $g_\alpha(\sigma) \to \frac{1}{\sigma}$ as $\alpha \to 0$ and $\sigma g_\alpha(\sigma) < 1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.4.11</span><span class="math-callout__name">(Tikhonov Regularisation)</span></p>

Here

$$g_\alpha(\sigma) = \frac{\sigma}{\sigma^2 + \alpha},$$

such that

$$x_\alpha^\delta := K_\alpha^\dagger y^\delta = \sum_{n=1}^{\infty} \frac{\sigma_n}{\sigma_n^2 + \alpha} \langle y^\delta, u_n \rangle_Y v_n, \qquad \text{for } y^\delta \in Y.$$

Since $\sigma^2 + \alpha \ge 2\sigma\sqrt{\alpha}$, we can choose $C_\alpha = \frac{1}{2\sqrt{\alpha}}$. Thus, a necessary condition for convergence with a-priori parameter rule is $\frac{\delta}{\sqrt{\alpha}} \to 0$. Furthermore, again $g_\alpha(\sigma) \to \frac{1}{\sigma}$ as $\alpha \to 0$ and $\sigma g_\alpha(\sigma) = \frac{\sigma^2}{\sigma^2 + \alpha} < 1$.

As in Example 2.4.10, $x_\alpha^\delta$ can be computed without explicit knowledge of the singular system, however, in this case for arbitrary $K \in \mathcal{K}(X, Y)$. In particular,

$$(K^* K + \alpha I) x_\alpha^\delta = K^* y^\delta, \tag{2.4.7}$$

a well-posed linear system for $\alpha > 0$. Tikhonov regularisation is in fact equivalent to Lavrentiev regularisation applied to the normal equations.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.4.12</span><span class="math-callout__name">(Landweber Iteration)</span></p>

For $\omega > 0$, consider the fixed point iteration

$$x_0 = 0 \quad \text{and} \quad x_{k+1} = x_k + \omega K^*(y^\delta - Kx_k), \quad \text{for } k \ge 0,$$

to compute regularised solutions $x_k$ of $Kx = y^\delta$. The associated family of regularisation operators $(K_k^\dagger)\_{k \in \mathbb{N}}$ satisfies $K_k^\dagger y^\delta = x_k$. Using the SVD of $K$ and $K^*$ we get

$$\sum_{j=1}^{\infty} \langle x_{k+1}, v_j \rangle_X v_j = \sum_{j=1}^{\infty} \left( (1 - \omega \sigma_j^2) \langle x_k, v_j \rangle_X + \omega \sigma_j \langle y^\delta, u_j \rangle_Y \right) v_j$$

and due to orthogonality

$$\langle x_{k+1}, v_j \rangle_X = (1 - \omega \sigma_j^2) \langle x_k, v_j \rangle_X + \omega \sigma_j \langle y^\delta, u_j \rangle_Y.$$

Since $x_0 = 0$, we get

$$\langle x_k, v_j \rangle_X = \omega \sigma_j \langle y^\delta, u_j \rangle_Y \sum_{i=1}^{k} (1 - \omega \sigma_j^2)^{k-i} = \frac{1 - (1 - \omega \sigma_j^2)^k}{\sigma_j} \langle y^\delta, u_j \rangle_Y.$$

Now we interpret the iteration number as the regularisation parameter and set $\alpha := 1/k$, so that

$$x_\alpha^\delta = K_\alpha^\dagger y^\delta = \sum_{j=1}^{\infty} \frac{1 - (1 - \omega \sigma_j^2)^{1/\alpha}}{\sigma_j} \langle y^\delta, u_j \rangle_Y \, v_j,$$

i.e. $g_\alpha(\sigma) = (1 - (1 - \omega\sigma^2)^{1/\alpha}) \frac{1}{\sigma}$. This function converges to $\frac{1}{\sigma}$ as $\alpha \to 0$ provided $\|1 - \omega\sigma^2\| < 1$. A sufficient condition for $\sigma \in \lbrace \sigma_n \rbrace$ is

$$0 < \omega < 2 \lVert K \rVert_{\mathcal{L}(X,Y)}^{-2}.$$

Since $g_\alpha$ is continuous and $\lim_{\sigma \to 0} g_\alpha(\sigma) = 0$, it is also bounded and $\sigma g_\alpha(\sigma) < 1$ for any $\alpha > 0$.

</div>

Under the stated conditions on $g_\alpha(\sigma)$, we can now prove the convergence of all the above regularisation methods with parameter choice rule $\alpha = \alpha(\delta, y^\delta)$ satisfying $\lim_{\delta \to 0} \delta C_{\alpha(\delta, y^\delta)} = 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.4.13</span><span class="math-callout__name">(Convergence of Spectral Regularisation)</span></p>

Let $g_\alpha : \mathbb{R}\_{+} \to \mathbb{R}\_{+}$ be a piecewise continuous function such that $g_\alpha(\sigma) \to \frac{1}{\sigma}$ for $\sigma > 0$ as $\alpha \to 0$, and suppose that there exist a constant $C_\alpha > 0$ depending on $\alpha$ and a constant $\gamma > 0$ independent of $\alpha$ such that

$$\sigma g_\alpha(\sigma) \le \gamma \qquad \text{and} \qquad g_\alpha(\sigma) \le C_\alpha < \infty \quad \text{for all } \sigma, \alpha > 0.$$

Consider (2.1.1) with $A = K \in \mathcal{K}(X, Y)$, $y \in \mathcal{D}(K^\dagger)$ and perturbed data $y^\delta \in Y$ with $\lVert y - y^\delta \rVert_Y \le \delta$. Then the regularisation method $(K_\alpha^\dagger, \alpha)$ with

$$K_\alpha^\dagger y := \sum_{n=1}^{\infty} g_\alpha(\sigma_n) \langle y, u_n \rangle_Y v_n, \qquad \text{for } y \in Y,$$

and parameter choice rule $\alpha = \alpha(\delta, y^\delta)$ converges provided

$$C_{\alpha(\delta, y^\delta)} \delta \to 0 \quad \text{as} \quad \delta \to 0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.4.13</summary>

To show convergence we bound again the two terms in the decomposition (2.4.2).

First to show that $K_\alpha^\dagger \to K^\dagger$ on $\mathcal{D}(K^\dagger)$ let $y \in D(K^\dagger)$. Then,

$$K_\alpha^\dagger y - K^\dagger y = \sum_{n=1}^{\infty} (g_\alpha(\sigma_n) - \frac{1}{\sigma_n}) \langle y, u_n \rangle_Y v_n = \sum_{n=1}^{\infty} (\sigma_n g_\alpha(\sigma_n) - 1) \langle x^\dagger, v_n \rangle_X v_n.$$

Due to the assumptions on $g_\alpha$, the coefficients in the above expansion satisfy

$$|(\sigma_n g_\alpha(\sigma_n) - 1) \langle x^\dagger, v_n \rangle_X| \le (\gamma + 1) |\langle x^\dagger, v_n \rangle_X|,$$

i.e. the sequence is bounded, and thus

$$\limsup_{\alpha \to 0} \lVert K_\alpha^\dagger y - K^\dagger y \rVert_X^2 \le \limsup_{\alpha \to 0} \sum_{n=1}^{\infty} |\sigma_n g_\alpha(\sigma_n) - 1|^2 |\langle x^\dagger, v_n \rangle_X|^2 \le \sum_{n=1}^{\infty} \lim_{\alpha \to 0} |\sigma_n g_\alpha(\sigma_n) - 1|^2 |\langle x^\dagger, v_n \rangle_X|^2 = 0,$$

since $\sigma g_\alpha(\sigma) \to 1$ pointwise. Thus, $\lVert K_\alpha^\dagger y - K^\dagger y \rVert_X \to 0$ as $\alpha \to 0$, independently of $\delta$.

To bound the propagated data error, note that, for all $\alpha, \delta > 0$,

$$\lVert K_\alpha^\dagger y - K_\alpha^\dagger y^\delta \rVert_X^2 \le \sum_{n=1}^{\infty} g_\alpha(\sigma_n)^2 |\langle y - y^\delta, u_n \rangle_Y|^2 \le C_\alpha^2 \sum_{n=1}^{\infty} |\langle y - y^\delta, u_n \rangle_Y|^2 \le C_\alpha^2 \lVert y - y^\delta \rVert_Y^2 \le (C_\alpha \delta)^2.$$

Thus, under the condition on the limit of $\delta C_{\alpha(\delta, y^\delta)}$ for the parameter choice rule $\alpha(\delta, y^\delta)$, this term also converges with $\delta \to 0$ and the proof is complete.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.4.14</span><span class="math-callout__name">(Data Error Bound)</span></p>

The bound

$$\lVert K_\alpha^\dagger y - K_\alpha^\dagger y^\delta \rVert_X \le C_\alpha \delta$$

in the proof suggests that the propagated data error is of order $\delta$. However, this is not true since $C_\alpha$ depends on $\delta$ and will in general grow with $\delta \to 0$. However, since we required that $C_{\alpha(\delta, y^\delta)} \delta \to 0$ as $\delta \to 0$, $C_\alpha$ grows slower than $\delta$ decreases such that $C_\alpha \delta$ will be of order $\delta^\nu$ for some $0 < \nu < 1$.

</div>

#### 2.4.3 Convergence Rates

A central question in the regularisation of inverse problems is the derivation of error bounds of the form

$$\lVert A_{\alpha(\delta, y^\delta)}^\dagger y^\delta - A^\dagger y \rVert \le \phi(\delta)$$

for some function $\phi : \mathbb{R}\_{+} \to \mathbb{R}\_{+}$ with $\lim_{t \to 0} \phi(t) = 0$ that is independent of $y$. We are interested in particular in the **worst case error**

$$e(y, \delta) := \sup \lbrace \lVert A_{\alpha(\delta, y^\delta)}^\dagger y^\delta - A^\dagger y \rVert_X : y^\delta \in Y \text{ with } \lVert y - y^\delta \rVert_Y \le \delta \rbrace. \tag{2.4.8}$$

This would allow us to provide a priori error bounds for the regularisation method. However, without any further assumptions on $y$ or on $x^\dagger = A^\dagger y$ this hope is futile.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.4.15</span><span class="math-callout__name">(No Uniform Convergence Rate Without Source Conditions)</span></p>

Let $(A_\alpha^\dagger, \alpha)$ be a convergent regularisation method. If there exists a function $\phi : \mathbb{R}\_{+} \to \mathbb{R}\_{+}$ with $\lim_{t \to 0} \phi(t) = 0$ and

$$\sup_{y \in \mathcal{D}(A^\dagger) \text{ s.t. } \lVert y \rVert_Y \le 1} e(y, \delta) \le \phi(\delta), \tag{2.4.9}$$

then $A^\dagger$ is bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.4.15</summary>

Let $y \in \mathcal{D}(A^\dagger)$ with $\lVert y \rVert_Y \le 1$ and $(y_n)\_n \subset \mathcal{D}(A^\dagger)$ with $\lVert y_n \rVert_Y \le 1$ a sequence satisfying $y_n \to y$ as $n \to \infty$. With $\delta_n := \lVert y - y_n \rVert_Y \to 0$ it then follows that

$$\lVert A^\dagger y_n - A^\dagger y \rVert_X \le \lVert A^\dagger y_n - A_{\alpha(\delta_n, y_n)}^\dagger y_n \rVert_X + \lVert A_{\alpha(\delta_n, y_n)}^\dagger y_n - A^\dagger y \rVert_X \le e(y_n, \delta_n) + e(y, \delta_n) \le 2\phi(\delta_n)$$

and thus, $A^\dagger$ is bounded on $\mathcal{D}(A^\dagger) \cap B_Y$. Since $A^\dagger$ is linear the boundedness extends to all of $\mathcal{D}(A^\dagger)$.

</details>
</div>

Thus the convergence can be arbitrarily slow without further assumptions on $y$ or on $x^\dagger$. For compact operators $K \in \mathcal{K}(X, Y)$ (which are smoothing operators), this can be achieved via an abstract smoothness condition on $x^\dagger$, called a **source condition**, namely that

$$x^\dagger \in \mathcal{R}(|K|^\nu), \quad \text{for some } \nu > 0,$$

where $\|K\|^\nu x := \sum_{n \in \mathbb{N}} \sigma_n^\nu \langle x, v_n \rangle_X v_n$, using the singular system $(\sigma_n, u_n, v_n)$ of $K$. This is in fact equivalent to a strengthened Picard-condition

$$\sum_{n \in \mathbb{N}} \sigma_n^{-2(\nu+1)} |\langle y, u_n \rangle_Y|^2 < \infty,$$

i.e. a faster decay of the coefficients of $y$ (or equivalently of $Kx^\dagger$) with respect to $(u_n)$ than necessary to purely guarantee the existence of $x^\dagger$ (as in Theorem 2.3.2).

Under this condition, it can be shown that the convergence rate as $\delta \to 0$ for the total error (with respect to $\delta$) is bounded below by $\frac{\nu}{\nu+1}$ for any regularisation method. A regularisation method is called **order-optimal** for $\nu > 0$, if for all $x^\dagger \in R(\|K\|^\nu)$ there exists a constant $c = c(x^\dagger) > 0$ such that

$$e(Kx^\dagger, \delta) \le c \, \delta^{\frac{\nu}{\nu+1}}.$$

### 2.5 Variational Regularisation and Extensions

The solution of the Tikhonov regularised linear system (2.4.7) is equivalent to minimising the following quadratic functional

$$K_\alpha^\dagger y^\delta := \arg\min_{x \in X} \left\lbrace \frac{1}{2} \lVert Kx - y^\delta \rVert_Y^2 + \frac{\alpha}{2} \lVert x \rVert_X^2 \right\rbrace. \tag{2.5.1}$$

This is how Tikhonov regularisation is typically introduced. In this variational setting, it is easier to generalise to other regularising functionals and to nonlinear inverse problems.

Indeed, if $\Phi(x) := \frac{1}{2} \lVert Kx - y^\delta \rVert_Y^2 + \frac{\alpha}{2} \lVert x \rVert_X^2$, the first-order optimality condition for a minimiser $x^* \in X$ of $\Phi$ is equivalent to setting $\frac{\mathrm{d}}{\mathrm{d}t} \Phi(x^\ast + th)\big\|_{t=0} = 0$ for arbitrary $h \in X$ with $\lVert h \rVert_X = 1$. Expanding, we get

$$\Phi(x + th) = \Phi(x) + t\Big( \langle Kx - y^\delta, Kh \rangle_Y + \alpha \langle x, h \rangle_X \Big) + \frac{t^2}{2} \Big( \lVert Kh \rVert_Y^2 + \alpha \lVert h \rVert_X^2 \Big)$$

and thus

$$0 = \frac{\mathrm{d}}{\mathrm{d}t} \Phi(x^* + th)\big|_{t=0} = \langle Kx^* - y^\delta, Kh \rangle_Y + \alpha \langle x^*, h \rangle_X = \langle K^*(Kx^* - y^\delta) + \alpha x^*, h \rangle_X,$$

which is equivalent to $x^*$ being the solution of (2.4.7), since $h \in X$ with $\lVert h \rVert_X = 1$ was arbitrary.

Let $J : X \to \mathbb{R}$ be a functional on $X$, then **generalised Tikhonov regularisation** seeks the regularised solution as the minimum of

$$\Phi_{\alpha, y^\delta}(x) := \frac{1}{2} \lVert Kx - y^\delta \rVert_Y^2 + \frac{\alpha}{2} J(x).$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.5.1</span><span class="math-callout__name">(Tikhonov-Philipps Regularisation)</span></p>

A simple generalisation of the classical Tikhonov regularisation consists in simply replacing $\frac{1}{2} \lVert x \rVert_X$ by $\frac{1}{2} \lVert Dx \rVert_Z$ for some linear (not necessarily bounded) operator $D : X \to Z$ from $X$ to some Hilbert space $Z$. Then minimising

$$\Phi_{\alpha, y^\delta}(x) := \frac{1}{2} \lVert Kx - y^\delta \rVert_Y^2 + \frac{\alpha}{2} \lVert Dx \rVert_Z^2,$$

constitutes the so-called Tikhonov-Philipps regularisation. It allows to penalise certain properties of $x$ through a suitable choice of $D$. In image processing, a typical choice for $D$ is the gradient operator, i.e. the regularisation functional $J$ is chosen to be the square of the $H^1$-seminorm. This penalises only variations in $x$, but not the size of $x$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.5.2</span><span class="math-callout__name">($\ell^1$-Regularisation)</span></p>

For non-injective operators $K \in \mathcal{L}(l^1, l^2)$ on sequence spaces, a popular choice for $J$ is the $l^1$-norm which is suitable to enforce sparsity in the regularised solution, i.e.,

$$\Phi_{\alpha, y^\delta}(x) := \frac{1}{2} \lVert Kx - y^\delta \rVert_Y^2 + \alpha \sum_{j=1}^{\infty} |x_j|.$$

</div>

As mentioned above, in variational form, Tikhonov regularisation can also easily be generalised to nonlinear inverse problems

$$F(x) = y, \tag{2.5.2}$$

where $F : \mathcal{D}(F) \subset X \to Y$ is a nonlinear bounded operator with domain $\mathcal{D}(F)$ between two Hilbert spaces $X$ and $Y$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.5.3</span><span class="math-callout__name">(Locally Ill-Posed)</span></p>

Let $x \in \mathcal{D}(F)$. The nonlinear equation (2.5.2) is called **locally ill-posed in $x$**, if for every $r > 0$ there exists a sequence $(x_n)\_{n \in \mathbb{N}} \subset B_r(x) \subset \mathcal{D}(F)$ such that

$$F(x_n) \to F(x), \quad \text{but} \quad x_n \not\to x.$$

Otherwise (2.5.2) is called **locally well-posed in $x$**.

</div>

For nonlinear problems, the classical Tikhonov regularisation computes a regularised approximation $x_\alpha^\delta$ of $x$ by minimising the functional

$$\Phi_{\alpha, y^\delta}(x) := \frac{1}{2} \lVert F(x) - y^\delta \rVert_Y^2 + \frac{\alpha}{2} \lVert x \rVert_X^2. \tag{2.5.3}$$

Under certain conditions on the nonlinear operator $F$ it can be shown that together with an a priori parameter choice rule $\alpha = \alpha(\delta)$ such that

$$\alpha(\delta) \to 0 \quad \text{and} \quad \frac{\delta^2}{\alpha(\delta)} \to 0 \quad \text{as } \delta \to 0,$$

(2.5.3) provides a convergent regularisation method for (2.5.2). As in the linear case, the regularisation functional $\lVert \cdot \rVert_X^2$ can again be replaced by another functional $J : X \to \mathbb{R}$ that penalises other features in the solution $x$.

For more details on nonlinear inverse problems see, e.g., [Engl, Hanke, Neubauer, 2000] or [Rieder, 2003].


## Chapter 3: Probability Theory in Banach Spaces

Bertrand's paradox from Exercise sheet 0 shows that one must be careful when introducing the notion of "randomness". In this chapter, among others we formally introduce probability spaces, random variables and most importantly conditional expectations and probabilities. In uncertainty quantification and inverse problems for partial differential equations, we often deal with quantities of interest or random objects belonging to a Sobolev space. For this reason, throughout we concentrate on random variables taking values in separable Banach spaces. For such random variables we will show the existence of so-called "regular conditional distributions", which allows to consider a version of Bayes' theorem in this setting.

Basic concepts of measure theory and the notions of measurability and strong-measurability are summarised in Appendix B. Proofs for the stated results on probability theory that are not given in this chapter, can be found in the book "Wahrscheinlichkeitstheorie" by Achim Klenke or in the lecture notes to "Probability Theory I" by Jan Johannes.

### 3.1 Integration in Banach Spaces

Let $V$ denote a Banach space over the field $\mathbb{R}$ (most results are easily generalized to Banach spaces over $\mathbb{C}$). In this section, we discuss integrals of the type $\int\_\Omega f(\omega) \, \mathrm{d}\mu(\omega)$, where $\mu$ is a $\sigma$-finite measure on the measurable space $(\Omega, \mathcal{A})$ and $f$ maps from $\Omega$ to the Banach space $V$.

Throughout we adhere to the following notational conventions: The norm of elements $x \in V$ is denoted by $\lVert x \rVert\_V$ (or simply $\lVert x \rVert$ in case there's no confusion about $V$) and we write $V' := \mathcal{L}(V; \mathbb{R})$ for the topological dual space of $V$ (the space of continuous linear maps from $V \to \mathbb{R}$). For $v' \in V'$, we denote by $\langle v, v' \rangle\_V$ the dual pairing (or simply $\langle v, v' \rangle$ if there's no confusion about $V$). We consider $V$ as a measurable space equipped with the Borel $\sigma$-algebra $\mathcal{B}(V)$. For a function $f : \Omega \to V$ and a set $B \subseteq V$ we use the shorthand $f^{-1}(B) := \lbrace \omega \in \Omega : f(\omega) \in B \rbrace$.

#### 3.1.1 Bochner Integrals

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\mu$-simple function)</span></p>

We say that $f : \Omega \to V$ is **$\mu$-simple** if

$$f = \sum_{j=1}^{n} \mathbb{1}_{A_j} v_j, \tag{3.1.1}$$

where $v\_j \in V$ and $A\_j \in \mathcal{A}$ such that $\mu(A\_j) < \infty$.

We say that a property holds **$\mu$-almost everywhere (a.e.)** (or $\mu$-almost surely) if there exists a **$\mu$-null-set** $N \in \mathcal{A}$, that is, $\mu(N) = 0$, and the property holds on $\Omega \setminus N$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.1.1</span><span class="math-callout__name">(Strong $\mu$-Measurability)</span></p>

A function $f : \Omega \to V$ is **strongly $\mu$-measurable** iff there exists a sequence $(f\_n)\_{n \in \mathbb{N}}$ of $\mu$-simple functions converging to $f$ $\mu$-a.e.

We call $\tilde{f}$ a **$\mu$-version** of $f$ if $\tilde{f} = f$ $\mu$-a.e. In case there is a $\mu$-version of $f$ that is $\mathcal{A}$-measurable, we say that $f$ is **$\mu$-measurable**.

</div>

As the name suggests, strong $\mu$-measurability is in general indeed stronger than $\mu$-measurability. In case $V$ is a separable Banach space, the two notions are in fact equivalent. This follows by Pettis measurability theorem (Theorem B.3.14).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.1.2</span><span class="math-callout__name">(Bochner Integral)</span></p>

Let $\mu$ be a $\sigma$-finite measure on the measurable space $(\Omega, \mathcal{A})$. A function $f : \Omega \to V$ is called $\mu$**-Bochner integrable** iff the following two conditions are met:

1. there exists a sequence of $\mu$-simple functions $f_n = \sum_{j=1}^{n} \mathbb{1}\_{A_{n,j}} v_{n,j}$ such that $\lim_{n \to \infty} f_n = f$ $\mu$-a.e.,
2. $\lim_{n \to \infty} \int_\Omega \lVert f(\omega) - f_n(\omega) \rVert \, \mathrm{d}\mu(\omega) = 0$.

For a $\mu$-Bochner integrable function we define

$$\int_\Omega f(\omega) \, \mathrm{d}\mu(\omega) := \lim_{n \to \infty} \sum_{j=1}^{n} \mu(A_{n,j}) v_{n,j} \in V. \tag{3.1.2}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.1.3</span></p>

Show that (3.1.2) does not depend on the approximating sequence $(f\_n)\_{n \in \mathbb{N}}$ and is well-defined (i.e. the limit exists in $V$).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.1.4</span><span class="math-callout__name">(Dual Pairing and Bochner Integral)</span></p>

Let $v' \in V'$ and let $f : \Omega \to V$ be Bochner-integrable. Then

$$\left\langle \int_\Omega f(\omega) \, \mathrm{d}\mu(\omega),\, v' \right\rangle = \int_\Omega \langle f(\omega), v' \rangle \, \mathrm{d}\mu(\omega). \tag{3.1.3}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 3.1.4</summary>

For a $\mu$-simple function $f_n = \sum_{j=1}^{n} \mathbb{1}_{A_j} v_j$, due to the linearity of the dual product

$$\left\langle \int_\Omega f_n(\omega) \, \mathrm{d}\mu(\omega),\, v' \right\rangle = \left\langle \sum_{j=1}^{n} v_j \mu(A_j),\, v' \right\rangle = \sum_{j=1}^{n} \langle v_j, v' \rangle \, \mu(A_j) = \int_\Omega \langle f_n(\omega), v' \rangle \, \mathrm{d}\mu(\omega). \tag{3.1.4}$$

Now let $(f_n)_{n \in \mathbb{N}}$ be as in Def. 3.1.2 (the Bochner integral definition). Taking the limit $n \to \infty$ on both sides of (3.1.4) yields (3.1.3). Here we use that $v' : V \to \mathbb{R}$ is continuous and that $\int_\Omega f_n \, \mathrm{d}\mu \to \int_\Omega f \, \mathrm{d}\mu$ in $V$ by assumption (which shows that the left-hand side of (3.1.4) converges to the left-hand side of (3.1.3)), and Def. 3.1.2(ii) (which shows that the right-hand side of (3.1.4) converges to the right-hand side of (3.1.3)).

</details>
</div>

The next theorem is useful to check for Bochner-integrability of a function.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.1.5</span><span class="math-callout__name">(Bochner Integrability Criterion)</span></p>

A strongly $\mu$-measurable function $f : \Omega \to V$ is $\mu$-Bochner integrable iff

$$\int_\Omega \lVert f(\omega) \rVert \, \mathrm{d}\mu(\omega) < \infty$$

(in the sense of the Lebesgue integral) and in this case

$$\left\lVert \int_\Omega f(\omega) \, \mathrm{d}\mu(\omega) \right\rVert \le \int_\Omega \lVert f(\omega) \rVert \, \mathrm{d}\mu(\omega). \tag{3.1.5}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3.1.5</summary>

**($\Rightarrow$)** If $f$ is $\mu$-Bochner integrable, then for the simple functions $(f_n)_{n \in \mathbb{N}}$ as in Def. 3.1.2

$$\int_\Omega \lVert f(\omega) \rVert \, \mathrm{d}\mu(\omega) \le \int_\Omega \lVert f(\omega) - f_n(\omega) \rVert \, \mathrm{d}\mu(\omega) + \int_\Omega \lVert f_n(\omega) \rVert \, \mathrm{d}\mu(\omega).$$

Due to assumption (ii) of Def. 3.1.2 the first term is finite for $n$ large enough. The second term is finite since each $f_n$ is a $\mu$-simple function.

**($\Leftarrow$)** Let $f$ be strongly $\mu$-measurable such that $\int_\Omega \lVert f(\omega) \rVert \, \mathrm{d}\mu(\omega) < \infty$, and let $g_n$ be $\mu$-simple functions satisfying $\lim_{n \to \infty} g_n = f$ $\mu$-a.e. Set

$$f_n := g_n \mathbb{1}_{\lbrace \lVert g_n \rVert \le 2 \lVert f \rVert \rbrace}.$$

Then $f_n$ is $\mu$-simple and $\lim_{n \to \infty} f_n = f$ $\mu$-a.e. Since $\lVert f_n \rVert \le 2 \lVert f \rVert$ pointwise for every $n$, by the dominated convergence theorem

$$\lim_{n \to \infty} \int_\Omega \lVert f - f_n \rVert \, \mathrm{d}\mu = 0.$$

The inequality (3.1.5) is trivial for $\mu$-simple functions, and follows by approximation in the general case.

</details>
</div>

#### 3.1.2 $L^p$-Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^p$-Spaces)</span></p>

Let $(\Omega, \mathcal{A}, \mu)$ be a $\sigma$-finite measure space. For $1 \le p < \infty$ we define $L^p(\Omega, \mu; V)$ to be the **space of all strongly $\mu$-measurable functions** $f : \Omega \to V$ for which

$$\lVert f \rVert_{L^p(\Omega, \mu; V)} := \left( \int_\Omega \lVert f(\omega) \rVert^p \, \mathrm{d}\mu(\omega) \right)^{1/p} < \infty$$

and identifying $\mu$-a.e. equal functions (i.e. elements of $L^p(\Omega, \mu; V)$ are equivalence classes of $\mu$-a.e. equal functions). In case we wish to emphasize the $\sigma$-algebra on $\Omega$ we write $L^p(\Omega, \mathcal{A}, \mu; V)$ (note that if $\mathcal{F} \subseteq \mathcal{A}$ is a sub-$\sigma$-algebra, in general $L^p(\Omega, \mathcal{A}, \mu; V) \neq L^p(\Omega, \mathcal{F}, \mu; V)$). If there's no confusion about $\mu$ or $\mathcal{A}$ we simply write $L^p(\Omega; V)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^\infty$-Space)</span></p>

Similarly, $L^\infty(\Omega; V)$ consists of all equivalence classes of strongly measurable $\mu$-a.e. equal functions endowed with the norm

$$\lVert f \rVert_{L^\infty(\Omega; V)} := \inf \lbrace r \ge 0 : \mu(\lbrace \omega \in \Omega : \lVert f(\omega) \rVert \ge r \rbrace) = 0 \rbrace. \tag{3.1.6}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(...)</span></p>

Without proof we mention that $L^p(\Omega; V) = L^p(\Omega, \mathcal{A}, \mu; V)$ is a Banach space for all $1 \le p \le \infty$. Note that $L^1(\Omega; V)$ consists of all equivalence classes of Bochner-integrable functions.

</div>

#### 3.1.3 Theorem of Radon-Nikodym

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.1.6</span><span class="math-callout__name">(Absolute Continuity of Measures)</span></p>

Given measures $\mu$ and $\nu$ on $(\Omega, \mathcal{A})$, we say that $\nu$ is **absolutely continuous** wrt $\mu$ ($\nu \ll \mu$) if for all $A \in \mathcal{A}$ s.t. $\mu(A) = 0$, we have $\nu(A) = 0$. The two measures are called **equivalent** iff $\mu \ll \nu$ and $\nu \ll \mu$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Radon-Nikodym Derivative)</span></p>

Suppose that $\mu$, $\nu$ are two measures on $(\Omega, \mathcal{A})$. In case there exists an $\mathcal{A}$-measurable $f : \Omega \to \mathbb{R}$ such that for all $A \in \mathcal{A}$

$$\nu(A) = \int_\Omega f(\omega) \mathbb{1}_A(\omega) \, \mathrm{d}\mu(\omega),$$

we call $f$ a **density** of $\nu$ w.r.t. $\mu$. If $\nu$ is $\sigma$-finite, such a density is $\mu$-a.e. unique, and as such this function is called the **Radon-Nikodym derivative** of $\nu$ w.r.t. $\mu$. We denote it by $\frac{\mathrm{d}\nu}{\mathrm{d}\mu} := f$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.1.7</span><span class="math-callout__name">(Radon-Nikodym)</span></p>

Let $\mu$, $\nu$ be two $\sigma$-finite measures on $(\Omega, \mathcal{A})$. Then

$$\nu \ll \mu \quad \Leftrightarrow \quad \text{the Radon-Nikodym derivative } \frac{\mathrm{d}\nu}{\mathrm{d}\mu} \text{ exists.}$$

In this case $\frac{\mathrm{d}\nu}{\mathrm{d}\mu}$ is $\mathcal{A}$-measurable and $\mu$-a.e. finite.

</div>

#### 3.1.4 Transformation of Measures

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.1.8</span><span class="math-callout__name">(Pushforward Measure)</span></p>

Let $(\Omega_1, \mathcal{A}\_1, \mu)$ be a measure space and $(\Omega_2, \mathcal{A}\_2)$ a measurable space. Let $T : \Omega_1 \to \Omega_2$ be measurable. Then

$$T_\sharp \mu(A_2) := \mu(\underbrace{\lbrace \omega \in \Omega_1 : T(\omega) \in A_2 \rbrace}_{= T^{-1}(A_2)}) \qquad \forall A_2 \in \mathcal{A}_2$$

defines a measure on $(\Omega_2, \mathcal{A}_2)$.

We call $T_\sharp \mu$ the **pushforward measure**.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Comment</span><span class="math-callout__name">(Pushforward Measure)</span></p>

* We can say that a pushforward measure is a measure induced by a **source measure space** and a **measurable function** to another measurable space.
  * This understanding is important when we already have two independent measure spaces $(X,\mathcal{A}\_1,\mu)$ and $(Y,\mathcal{A}\_2,\nu)$, and we want to find such a map $T: X\to Y$, which pushes the measure from the source space to a target space, like a constraint in the optimal transport problem: 
  
    $$\min_T \int_X c(x, T(x)) \, \mathrm{d}\mu(x), \qquad \text{subject to } T_\sharp \mu = \nu.$$

  * In the original Monge's optimal transport problem such a transformation $T$ might not even exist!

* Because the **measurable function does not necessarily preserve the measure**, i.e. the preimage could have different measure from the target measure, to make the integration compatible on pairs (preimage, image), allowing change of variables, we take into account the local change of measure (a.k.a.) the determinant of the jacobian of the inverse.

</div>

For real valued measurable functions, we have the usual change of variables formula (for the Lebesgue integrals):

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.1.9</span><span class="math-callout__name">(Change of Variables)</span></p>

Let $T : \Omega_1 \to \Omega_2$ and $f : \Omega_2 \to \mathbb{R}$ be measurable. Then 

$$\int_{\Omega_2} |f(\omega_2)| \, \mathrm{d}T_\sharp\mu(\omega_2) < \infty \iff \int_{\Omega_1} |f \circ T| \, \mathrm{d}\mu(\omega_1) < \infty$$

and in this case

$$\int_{\Omega_1} f \circ T(\omega_1) \, \mathrm{d}\mu(\omega_1) = \int_{\Omega_2} f(\omega_2) \, \mathrm{d}T_\sharp\mu(\omega_2).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.1.10</span><span class="math-callout__name">(Transformation of Densities)</span></p>

Assume that $\mu \ll \lambda$ is a measure on $(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d))$ with density $f := \frac{\mathrm{d}\mu}{\mathrm{d}\lambda}$. In the important case that $T : \mathbb{R}^d \to \mathbb{R}^d$ is a $C^1$-diffeomorphism, we have for all $A \in \mathcal{B}(\mathbb{R}^d)$

$$T_\sharp \mu(A) = \mu(T^{-1}(A)) = \int_{T^{-1}(A)} f(x) \, \mathrm{d}x = \int_A f(T^{-1}(x)) \det \mathrm{d}T^{-1}(x) \, \mathrm{d}x.$$

Hence the density transforms under the pushforward as 

$$\frac{\mathrm{d}T_\sharp \mu}{\mathrm{d}\lambda} = \frac{\mathrm{d}\mu}{\mathrm{d}\lambda} \circ T^{-1} \det \mathrm{d}T^{-1}$$

where $\mathrm{d}T^{-1} : \mathbb{R}^d \to \mathbb{R}^{d \times d}$ denotes the Jacobian matrix of $T^{-1}$.

</div>

### 3.2 Banach-valued Random Variables

Let $V$ be a Banach space and $(\Omega, \mathcal{A}, \mathbb{P})$ a probability space.

A set $A \in \mathcal{A}$ is called an **event**. $\mathbb{P}[A]$ is the **probability** of the event $A$.

Often it is not convenient or possible to work with events. Instead we consider observable quantities of such events. This idea is formalized with the notion of random variables.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.2.2</span><span class="math-callout__name">(Random Variable)</span></p>

Let $(\Omega, \mathcal{A})$ be a measurable space and $V$ a Banach space. Then a measurable function $X : \Omega \to V$ is called a $V$**-valued random variable** (RV).

</div>

It is common practice in probability theory to write $X$ instead of $X(\omega)$, i.e. not to explicitly display the dependence of $X$ on $\omega \in \Omega$.

For a probability space $(\Omega, \mathcal{A}, \mathbb{P})$, a RV $X : \Omega \to V$ induces a probability measure $\mathbb{P}\_X := X_\sharp \mathbb{P}$ on $(V, \mathcal{B}(V))$, i.e. $\mathbb{P}\_X[B] = \mathbb{P}[\lbrace \omega \in \Omega : X(\omega) \in B \rbrace]$. For $B \in \mathcal{B}(V)$ we usually write $\mathbb{P}[X \in B]$ to denote $\mathbb{P}\_X[B]$, which is the probability of the event $\lbrace \omega \in \Omega : X(\omega) \in B \rbrace$, i.e. the probability that $X$ takes a value in $B$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.2.4</span><span class="math-callout__name">(Distribution)</span></p>

1. The measure $\mathbb{P}\_X$ is the **distribution** of $X$.
2. We write $X \sim \mu$ to express that $X$ has distribution $\mu$, i.e. $\mathbb{P}\_X = \mu$.
3. A family of $V$-valued RVs $(X_j)\_{j \in I}$ is called **identically distributed** if $\mathbb{P}\_{X_i} = \mathbb{P}\_{X_j}$ for all $i, j \in I$.
4. For a finite family of RVs $X_j : \Omega \to V_j$, $j = 1, \ldots, n$, the measure $\mathbb{P}\_{X_1, \ldots, X_n} := (X_1, \ldots, X_n)\_\sharp \mathbb{P}$ on $(\times_{j=1}^n V_j, \mathcal{B}(\times_{j=1}^n V_j))$ is the **joint distribution** of the RVs $(X_j)\_{j=1}^n$, and $\mathbb{P}\_{X_j}$ is the **marginal distribution** of $X_j$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.2.5</span><span class="math-callout__name">(Distribution Function and Density)</span></p>

Suppose $X : \Omega \to \mathbb{R}$ is a real valued RV.

1. The function $F_X(x) := \mathbb{P}_X[X \le x]$ is the **distribution function** of $X$.
2. If there exists a nonnegative integrable function $f : \mathbb{R} \to \mathbb{R}$ such that for all $x \in \mathbb{R}$

$$F(x) = \int_{-\infty}^{x} f(t) \, \mathrm{d}t,$$

then $f$ is called the **density function** for $X$. In this case we also write $f = f_X$.

</div>

Note that $f$ is simply the Radon-Nikodym derivative of $\mathbb{P}\_X$ w.r.t. the Lebesgue measure (the "Lebesgue density"), i.e. $f = \frac{\mathrm{d}\mathbb{P}\_X}{\mathrm{d}\lambda}$. For real valued RVs $X_j : \Omega \to \mathbb{R}$ these notions generalise to $n$ RVs: the **joint distribution function** is

$$F_{X_1, \ldots, X_n}(x_1, \ldots, x_n) := \mathbb{P}_{X_1, \ldots, X_n}[X_1 \le x_1, \ldots, X_n \le x_n]$$

and if there exists a nonnegative $f : \mathbb{R} \to \mathbb{R}$ satisfying

$$F_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = \int_{-\infty}^{x_1} \cdots \int_{-\infty}^{x_n} f(t_1, \ldots, t_n) \, \mathrm{d}t_1 \ldots \mathrm{d}t_n$$

then $f$ is the **density function** of $X = (X_1, \ldots, X_n)$. We also write $f(x) = f\_{X\_1, \ldots, X\_n}(x)$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.2.6</span><span class="math-callout__name">(Dice Roll)</span></p>

Let $\Omega = \lbrace 1, \ldots, 6 \rbrace$ be equipped with the $\sigma$-algebra $\mathcal{A} = 2^\Omega$. We interpret each $\omega \in \Omega$ as the outcome of a dice roll, and set

$$X(\omega) = \begin{cases} 0 & \text{if } \omega \text{ is even} \\ 1 & \text{if } \omega \text{ is odd.} \end{cases}$$

Then $X : \lbrace 1, \ldots, 6 \rbrace \to \mathbb{R}$ is an $\mathbb{R}$-valued RV. To model a fair dice, we can define a probability measure $\mathbb{P}$ via $\mathbb{P}[\omega] = \frac{1}{6}$ for each $\omega \in \Omega$.

</div>

It is easy to check that for a RV $X : \Omega \to V$,

$$\sigma(X) := \lbrace X^{-1}(B) : B \in \mathcal{B}(V) \rbrace$$

is a $\sigma$-algebra, called the **$\sigma$-algebra generated by $X$**. It is the smallest $\sigma$-algebra on $\Omega$ w.r.t. which $X$ is measurable, and it can be interpreted as containing all relevant information about the RV $X$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.2.7</span><span class="math-callout__name">($\sigma$-Algebra Generated by the Dice Roll)</span></p>

Consider $X : \lbrace 1, \ldots, 6 \rbrace \to \lbrace 0, 1 \rbrace$ from Example 3.2.6. Then

$$\sigma(X) = \lbrace \lbrace 1, 3, 5 \rbrace, \lbrace 2, 4, 6 \rbrace, \lbrace 1, 2, 3, 4, 5, 6 \rbrace, \emptyset \rbrace.$$

This $\sigma$-algebra contains all relevant information about $X$, namely whether the dice shows an odd or an even number.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.2.8</span><span class="math-callout__name">(Expectation)</span></p>

We say that a RV $X : \Omega \to V$ has finite $k$th moment, iff $\int_\Omega \lVert X(\omega) \rVert^k \, \mathrm{d}\mathbb{P}(\omega) < \infty$.

If $X : \Omega \to V$ has finite first moment, then

$$\mathbb{E}[X] := \int_\Omega X(\omega) \, \mathrm{d}\mathbb{P}(\omega)$$

is the **expectation** of $X$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Covariance operator)</span></p>

For two separable Hilbert spaces $(H_1, \langle \cdot, \cdot \rangle_{H_1})$, $(H_2, \langle \cdot, \cdot \rangle_{H_2})$ and two random variables $X : \Omega \to H_1$, $Y : \Omega \to H_2$ with finite second moments, we define the **covariance operator** $\mathrm{cov}(X, Y) = C : H_2 \to H_1$ by

$$\langle v, Cw \rangle_{H_1} = \int_\Omega \langle X - \mathbb{E}[X], v \rangle_{H_1} \langle Y - \mathbb{E}[Y], w \rangle_{H_2} \, \mathrm{d}\mathbb{P} \qquad \forall v \in H_1, \ w \in H_2.$$

We also set $\mathrm{cov}(X) := \mathrm{cov}(X, X)$. One can show that $\mathrm{cov}(X)$ is a self-adjoint positive trace-class operator.

In case $H = \mathbb{R}$ the **variance** of $X : \Omega \to \mathbb{R}$ is defined as

$$\mathbb{V}(X) := \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.2.9</span><span class="math-callout__name">(Covariance Matrix)</span></p>

Let $X : \Omega \to \mathbb{R}^n$ and $Y : \Omega \to \mathbb{R}^m$ be two random variables. Then $\mathrm{cov}(X, Y)$ is represented by the **covariance matrix** $C \in \mathbb{R}^{n \times m}$ with entries

$$C_{ij} = \mathbb{E}[(X_i - \mathbb{E}[X_i])(Y_j - \mathbb{E}[Y_j])].$$

Under linear transformations the covariance matrix satisfies $\mathrm{cov}(AX, BY) = A \mathrm{cov}(X, Y) B^\top$. In case $X = Y$ we have $C_{ii} = \mathbb{V}(X_i)$.

</div>

Expectations can be computed using the following change of variables formula:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.2.10</span><span class="math-callout__name">(Change of Variables for Expectations)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space, and $(V, \lVert \cdot \rVert_V)$, $(W, \lVert \cdot \rVert_W)$ two separable Banach spaces. Let $X : \Omega \to V$ be a RV and $\varphi : V \to W$ a measurable function. Then $\varphi(X) : \Omega \to W$ is a RV. It holds $\varphi \in L^1(V, \mathbb{P}_X; W)$ iff $\varphi(X) \in L^1(\Omega, \mathbb{P}; W)$ and in this case

$$\mathbb{E}[\varphi(X)] = \int_\Omega \varphi(X(\omega)) \, \mathrm{d}\mathbb{P}(\omega) = \int_V \varphi(v) \, \mathrm{d}\mathbb{P}_X(v).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3.2.10</summary>

Both $\varphi(X) : \Omega \to W$ and $\varphi : V \to W$ are measurable, and thus strongly measurable since $V$ and $W$ are separable. $\mathbb{P}$ and $\mathbb{P}_X$ are probability measures, and thus $\varphi(X)$ is strongly $\mathbb{P}$-measurable and $\varphi$ is strongly $\mathbb{P}_X$-measurable. By Thm. 3.1.9

$$\int_\Omega \lVert \varphi(X(\omega)) \rVert_W \, \mathrm{d}\mathbb{P}(\omega) = \int_V \lVert \varphi(v) \rVert_W \, \mathrm{d}\mathbb{P}_X(v)$$

and hence Thm. 3.1.5 implies $\varphi(X) \in L^1(\Omega, \mathbb{P}; W)$ iff $\varphi \in L^1(V, \mathbb{P}_X; W)$. In this case, Lemma 3.1.4 implies for all $w' \in W'$

$$\left\langle \int_\Omega \varphi(X(\omega)) \, \mathrm{d}\mathbb{P}(\omega), w' \right\rangle = \int_\Omega \langle \varphi(X(\omega)), w' \rangle \, \mathrm{d}\mathbb{P}(\omega) = \int_V \langle \varphi(v), w' \rangle \, \mathrm{d}\mathbb{P}_X(v) = \left\langle \int_V \varphi(v) \, \mathrm{d}\mathbb{P}_X(v), w' \right\rangle,$$

where we used again Thm. 3.1.9 for the real-valued measurable function $\omega \mapsto \langle \varphi(X(\omega)), w' \rangle$ (and the fact that the Lebesgue and Bochner integrals coincide for the integral of real-valued measurable functions w.r.t. $\sigma$-finite measures). Since this equality holds for all $w' \in W'$, we conclude

$$\int_\Omega \varphi(X(\omega)) \, \mathrm{d}\mathbb{P}(\omega) = \int_V \varphi(v) \, \mathrm{d}\mathbb{P}_X(v).$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.2.11</span></p>

With $\varphi(v) = v$ we get

$$\mathbb{E}[X] = \int_V v \, \mathrm{d}\mathbb{P}_X(v).$$

</div>

### 3.3 Independence and Conditionals

#### 3.3.1 Conditional Probability and Independence

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space and let $A, B \in \mathcal{A}$ be two events such that $\mathbb{P}[B] > 0$. For $\omega \in \Omega$, assuming that we already know $\omega \in B$, we want to define the probability that $\omega \in A$ -- **the probability of $A$ given $B$**. Since we know $\omega \in B$, we can interpret $B$ together with the $\sigma$-algebra $\lbrace C \in \mathcal{A} : C \subseteq B \rbrace$ and the probability measure $\tilde{\mathbb{P}} := \frac{\mathbb{P}}{\mathbb{P}[B]}$ as a new probability space. Then the probability of $\omega$ belonging to $A$ equals $\tilde{\mathbb{P}}[A \cap B] = \frac{\mathbb{P}[A \cap B]}{\mathbb{P}[B]}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.1</span><span class="math-callout__name">(Conditional Probability I)</span></p>

The **conditional probability** of $A$ given $B$ is

$$\mathbb{P}[A|B] := \frac{\mathbb{P}[A \cap B]}{\mathbb{P}[B]}.$$

</div>

If the knowledge of $B$ has no influence on the probability of $A$, i.e. $\mathbb{P}[A\mid B] = \mathbb{P}[A]$, we say that the **events are independent**. If $P(B) > 0$, this is equivalent to $\mathbb{P}[A]\mathbb{P}[B] = \mathbb{P}[A \cap B]$. The latter condition is symmetric in $A$ and $B$, as it should be.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.2</span><span class="math-callout__name">(Independent Events)</span></p>

Two events $A$ and $B$ are called **independent** iff

$$\mathbb{P}[A \cap B] = \mathbb{P}[A]\mathbb{P}[B].$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.3.3</span></p>

Show that if $A$ and $B$ are independent, then $A^c$ and $B$ are also independent.

</div>

Next we generalize the notion of independence to $\sigma$-algebras and RVs.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.4</span><span class="math-callout__name">(Independent $\sigma$-Algebras)</span></p>

Let $\mathcal{A}\_i \subseteq \mathcal{A}$ be $\sigma$-algebras on $\Omega$ for all $i \in I$. The $(\mathcal{A}_i)\_{i \in I}$ are **independent** if for all finite subsets $\lbrace k_1, \ldots, k_n \rbrace \subseteq I$ and all events $A_i \in \mathcal{A}\_{k_i}$ holds

$$\mathbb{P}[A_1 \cap \cdots \cap A_n] = \mathbb{P}[A_1] \ldots \mathbb{P}[A_n].$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.5</span><span class="math-callout__name">(Independent Random Variables)</span></p>

Let $X_i : \Omega \to V$ for $i \in I$ be a family of RVs for a Banach space $V$. We say that the $X_i$ are **independent** if for all finite subsets $\lbrace k_1, \ldots, k_n \rbrace \subseteq I$ the $\sigma$-algebras $(\sigma(X_{k_i}))\_{i=1}^n$ are independent or equivalently for all $B_1, \ldots, B_n \in \mathcal{B}(V)$

$$\mathbb{P}[X_{k_1} \in B_1, \ldots, X_{k_n} \in B_n] = \mathbb{P}[X_1 \in B_1] \cdots \mathbb{P}[X_{k_n} \in B_n].$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.3.6</span></p>

Consider the probability space $([0, 1], \mathcal{B}([0, 1]), \lambda)$. Define for $\omega \in [0, 1]$

$$X_n(\omega) := \begin{cases} 1 & \text{if } \frac{k}{2^n} \le \omega \le \frac{k+1}{2^n},\ k \text{ even} \\ -1 & \text{if } \frac{k}{2^n} \le \omega \le \frac{k+1}{2^n},\ k \text{ odd.} \end{cases}$$

Show that the $(X\_n)\_{n \in \mathbb{N}}$ are a family of independent random variables.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.3.7</span><span class="math-callout__name">(Bayes' Formula)</span></p>

Let $A\_1, \ldots, A\_n$ be disjoint events of positive probability such that $\Omega = \bigcup\_{j=1}^{n} A\_j$. Let $B$ be another event with $\mathbb{P}[B] > 0$. Show that for $k \in \lbrace 1, \ldots, n \rbrace$

$$\mathbb{P}[A_k|B] = \frac{\mathbb{P}[B|A_k]\mathbb{P}[A_k]}{\sum_{j=1}^{n} \mathbb{P}[B|A_j]\mathbb{P}[A_j]}.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.3.8</span><span class="math-callout__name">(Independence via Product Measures)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a measure space and $V$ a Banach space. Let $X_i : \Omega \to V$ be RVs for $i = 1, \ldots, n$. Then the $X_i$ are independent if and only if $\mathbb{P}\_{X_1, \ldots, X_n} = \mathbb{P}\_{X_1} \otimes \cdots \otimes \mathbb{P}\_{X_n}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 3.3.8</summary>

**($\Rightarrow$)** Assume that the $X_j$ are independent. Then for all $A_j \in \mathcal{A}$

$$\mathbb{P}_{X_1, \ldots, X_n}[A_1 \times \cdots \times A_n] = \mathbb{P}[X_1 \in A_1, \ldots, X_n \in A_n] = \mathbb{P}[X_1 \in A_1] \cdots \mathbb{P}[X_n \in A_n] = \mathbb{P}_{X_1}[A_1] \cdots \mathbb{P}_{X_n}[A_n].$$

By the uniqueness of the product measure (Thm. B.2.6), this implies $\mathbb{P}_{X_1, \ldots, X_n} = \mathbb{P}_{X_1} \otimes \cdots \otimes \mathbb{P}_{X_n}$.

**($\Leftarrow$)** Conversely, by definition of the product measure, $\mathbb{P}_{X_1, \ldots, X_n} = \mathbb{P}_{X_1} \otimes \cdots \otimes \mathbb{P}_{X_n}$ implies for all $A_j \in \mathcal{A}$ that

$$\mathbb{P}_{X_1, \ldots, X_n}[A_1 \times \cdots \times A_n] = \mathbb{P}_{X_1}[A_1] \cdots \mathbb{P}_{X_n}[A_n],$$

which is exactly the independence condition.

</details>
</div>

For real valued RVs, independence is equivalent to saying that the distribution functions and densities factor.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3.9</span><span class="math-callout__name">(Independence via Factorisation)</span></p>

Let $X_i : \Omega \to \mathbb{R}^m$ be $n$ RVs for $i = 1, \ldots, n$.

1. The RVs are independent iff for $x = (x_1, \ldots, x_n)$: $F_{X_1, \ldots, X_n}(x) = F_{X_1}(x_1) \ldots F_{X_n}(x_n)$.
2. If the RVs have densities, then they are independent iff $f_{X_1, \ldots, X_n}(x) = f_{X_1}(x_1) \ldots f_{X_n}(x_n)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Sketch of Proof of Theorem 3.3.9</summary>

**($\Rightarrow$)** If the $X_j$ are independent, then

$$F_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = \mathbb{P}[X_1 \le x_1, \ldots, X_n \le x_n] = \mathbb{P}[X_1 \le x_1] \cdots \mathbb{P}[X_n \le x_n] = F_{X_1}(x_1) \cdots F_{X_n}(x_n).$$

**($\Leftarrow$)** Conversely, let $A_i = X_i^{-1}(B_i)$ for $B_i \in \mathcal{B}(\mathbb{R}^m)$. Then using Fubini's Theorem

$$\mathbb{P}[A_1 \cap \cdots \cap A_n] = \mathbb{P}[X_1 \in B_1, \ldots, X_n \in B_n] = \int_{B_1 \times \cdots \times B_n} f_{X_1, \ldots, X_n}(x_1, \ldots, x_n) \, \mathrm{d}x_1 \cdots \mathrm{d}x_n$$

$$= \prod_{j=1}^{n} \int_{B_j} f_{X_j}(x_j) \, \mathrm{d}x_j = \prod_{j=1}^{n} \mathbb{P}[X_j \in B_j] = \prod_{j=1}^{n} \mathbb{P}[A_j].$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3.10</span><span class="math-callout__name">(Expectation of Independent Products)</span></p>

Let $X_1, \ldots, X_n : \Omega \to \mathbb{R}$ be independent RVs and such that $\mathbb{E}[\|X_i\|] < \infty$ for all $i = 1, \ldots, n$. Then

$$\mathbb{E}[X_1 \cdots X_n] = \mathbb{E}[X_1] \cdots \mathbb{E}[X_n] < \infty.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3.3.10</summary>

Follows from Thm. 3.2.10 (change of variables for expectations) and Fubini's theorem applied to the product measure $\mathbb{P}_{X_1, \ldots, X_n} = \mathbb{P}_{X_1} \otimes \cdots \otimes \mathbb{P}_{X_n}$ guaranteed by Proposition 3.3.8.

</details>
</div>

#### 3.3.2 Conditional Expectations

Let $X$ be a random variable on $(\Omega, \mathcal{A})$ and let $B \in \mathcal{A}$. Given an event $B$ with $\mathbb{P}[B] > 0$, it is natural to introduce the expectation of $X$ given $B$ as

$$\mathbb{E}[X|B] := \int_B X(\omega) \, \mathrm{d}\mathbb{P}[\omega|B] = \frac{1}{\mathbb{P}[B]} \int_\Omega \mathbb{1}_B(\omega) X(\omega) \, \mathrm{d}\mathbb{P}(\omega).$$

Now let $X$ and $Y$ be two random variables. How can we define the expectation of $X$ given $Y$? Since $Y$ is a random variable, this conditional expectation should also be a random variable.

To motivate the following discussion, consider $X : [0, 1] \to \mathbb{R}$ and $Y : [0, 1] \to \mathbb{R}$ two RVs. Let $\bigcup_{j=1}^{n} A_j = [0, 1]$ be a partition of $[0, 1]$ and suppose that $Y(\omega) = \sum_{j=1}^{n} \mathbb{1}\_{A_j}(\omega) y_j$ is a simple function. If $Y(\omega) = y_j$, then $\omega \in A_j$. Hence the expectation for $X$ is the average of $X$ over $A_j$, i.e.

$$\mathbb{E}[X|A_j] = \frac{1}{\mathbb{P}[A_j]} \int_{A_j} X \, \mathrm{d}\mathbb{P}.$$

We thus set

$$\mathbb{E}[X|Y](\omega) := \frac{1}{\mathbb{P}[A_j]} \int_{A_j} X \, \mathrm{d}\mathbb{P} \qquad \text{if } \omega \in A_j.$$

We make the following observations:

1. $\mathbb{E}[X\mid Y] : [0, 1] \to \mathbb{R}$ is a random variable that is constant on each $A_j$.
2. The actual values $y_j$ taken by $Y$ are irrelevant for the definition of $\mathbb{E}[X\mid Y]$; we merely require the sets $A_j$, or in other words the $\sigma$-algebra $\sigma(Y)$ generated by $Y$.
3. $\mathbb{E}[X\mid Y] : [0, 1] \to \mathbb{R}$ is $\sigma(Y)$-measurable.
4. $\int \mathbb{1}\_A X \, \mathrm{d}\mathbb{P} = \int \mathbb{1}\_A \mathbb{E}[X\mid Y] \, \mathrm{d}\mathbb{P}$ for all $A \in \sigma(Y)$.

The second item motivates us to first introduce expectations of $X$ conditioned on $\sigma$-algebras.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.11</span><span class="math-callout__name">(Conditional Expectation I)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space, $\mathcal{F} \subseteq \mathcal{A}$ a sub-$\sigma$-algebra, $V$ a separable Banach space and $X : \Omega \to V$ a random variable such that $X \in L^1(\Omega, \mu; V)$. A random variable $Z : \Omega \to V$ is called a **conditional expectation of $X$ given $\mathcal{F}$**, iff

1. $Z : \Omega \to V$ is $\mathcal{F}$-measurable,
2. $\int_\Omega \mathbb{1}\_B Z \, \mathrm{d}\mathbb{P} = \int_\Omega \mathbb{1}\_B X \, \mathrm{d}\mathbb{P} \in V$ for all $B \in \mathcal{F}$.

In this case we write $\mathbb{E}[X\mid \mathcal{F}] = Z$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3.12</span><span class="math-callout__name">(Existence and Uniqueness for $V = \mathbb{R}$)</span></p>

Let $V = \mathbb{R}$. Then $\mathbb{E}[X\mid \mathcal{F}]$ exists and is $\mathbb{P}$-a.e. unique.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3.3.12</summary>

**Uniqueness:** Assume $Z$ and $Z'$ both satisfy the conditions of Def. 3.3.11. Let $A = \lbrace \omega \in \Omega : Z(\omega) > Z'(\omega) \rbrace \in \mathcal{F}$. Then

$$\int_\Omega \mathbb{1}_A(\omega)(Z(\omega) - Z'(\omega)) \, \mathrm{d}\mathbb{P}(\omega) = 0$$

and since $Z - Z' > 0$ on $A$ we have $\mathbb{P}[A] = 0$. Similarly, with $B = \lbrace \omega \in \Omega : Z(\omega) < Z'(\omega) \rbrace$ we get $\mathbb{P}[B] = 0$ and thus $Z = Z'$ $\mathbb{P}$-a.e.

**Existence:** Set $X^+ := \max\lbrace 0, X \rbrace$ and $X^- := -\min\lbrace 0, X \rbrace$. For $* \in \lbrace +, - \rbrace$ define

$$\mu^*(A) := \mathbb{E}[X^* \mathbb{1}_A] \qquad \forall A \in \mathcal{F}.$$

Then $\mu^\pm$ are two $\sigma$-finite measures on $(\Omega, \mathcal{F})$. By construction $\mu^\pm \ll \mathbb{P}$ and there exist $\mathcal{F}$-measurable Radon-Nikodym derivatives $Z^\pm : \Omega \to \mathbb{R}$ such that

$$\mu^\pm(A) = \int_A Z^\pm \, \mathrm{d}\mathbb{P} \qquad \forall A \in \mathcal{F}.$$

Then $Z := Z^+ - Z^-$ is $\mathcal{F}$-measurable (the difference of $\mathcal{F}$-measurable $\mathbb{R}$-valued functions is again $\mathcal{F}$-measurable), and for all $A \in \mathcal{F}$

$$\int_\Omega \mathbb{1}_A(\omega) Z \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \mathbb{1}_A(\omega) Z^+ \, \mathrm{d}\mathbb{P}(\omega) - \int_\Omega \mathbb{1}_A(\omega) Z^- \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \mathbb{1}_A(\omega) X(\omega) \, \mathrm{d}\mathbb{P}(\omega).$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.3.13</span></p>

The spaces $L^2(\Omega, \mathcal{A}, \mathbb{P}; \mathbb{R})$ and $L^2(\Omega, \mathcal{F}, \mathbb{P}; \mathbb{R})$ are Hilbert spaces with the $L^2(\Omega, \mathbb{P})$-inner product. It can be shown that for $X \in L^2(\Omega, \mathcal{A}, \mathbb{P}; \mathbb{R})$, $\mathbb{E}[X\mid \mathcal{F}]$ is the orthogonal projection onto the closed subspace $L^2(\Omega, \mathcal{F}, \mathbb{P}; \mathbb{R})$, that is for any $\mathcal{F}$-measurable $Z : \Omega \to \mathbb{R}$

$$\mathbb{E}[(X - \mathbb{E}[X|\mathcal{F}])^2] \le \mathbb{E}[(X - Z)^2]$$

with equality iff $Z = \mathbb{E}[X\mid \mathcal{F}]$ $\mathbb{P}$-a.e.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.3.14</span></p>

For $V = \mathbb{R}$ show that

1. $\mathbb{E}[\mathbb{E}[X\mid\mathcal{F}]] = \mathbb{E}[X]$,
2. $\mathbb{E}[X] = \mathbb{E}[X\mid\mathcal{F}]$ in case $\mathcal{F} = \lbrace \emptyset, \Omega \rbrace$.

</div>

Some further properties of the conditional probability are the following:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3.15</span><span class="math-callout__name">(Properties of Conditional Expectation)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space, $X$ and $Y$ two real-valued RVs in $L^1(\Omega, \mathcal{A}, \mathbb{P}; \mathbb{R})$, and $\mathcal{G} \subseteq \mathcal{F} \subseteq \mathcal{A}$ sub-$\sigma$-algebras. Then

1. *(linearity)* for $\alpha \in \mathbb{R}$, $\mathbb{E}[\alpha X + Y\mid\mathcal{F}] = \alpha \mathbb{E}[X\mid\mathcal{F}] + \mathbb{E}[Y\mid\mathcal{F}]$,
2. *(monotonicity)* if $X \ge Y$ $\mathbb{P}$-a.e., then $\mathbb{E}[X\mid\mathcal{F}] \ge \mathbb{E}[Y\mid\mathcal{F}]$ $\mathbb{P}$-a.e.,
3. *(tower property)* $\mathbb{E}[\mathbb{E}[X\mid\mathcal{F}]\mid\mathcal{G}] = \mathbb{E}[\mathbb{E}[X\mid\mathcal{G}]\mid\mathcal{F}] = \mathbb{E}[X\mid\mathcal{G}]$,
4. *(triangle inequality)* $\mathbb{E}[\|X\|\mid\mathcal{F}] \ge \|\mathbb{E}[X\mid\mathcal{F}]\|$,
5. *(independence)* if $\sigma(X)$ and $\mathcal{F}$ are independent, then $\mathbb{E}[X\mid\mathcal{F}] = \mathbb{E}[X]$,
6. *(Lebesgue dominated convergence)* if $Y \ge 0$ and $(X_n)\_{n \in \mathbb{N}}$ is a sequence of RVs with $\|X_n\| \le Y$ for all $n \in \mathbb{N}$ and $X_n \to X$ $\mathbb{P}$-a.e., then $\lim_{n \to \infty} \mathbb{E}[X_n\mid\mathcal{F}] = \mathbb{E}[X\mid\mathcal{F}]$ $\mathbb{P}$-a.e. and in the sense of $L^1(\Omega, \mathcal{F}, \mathbb{P}; \mathbb{R})$,
7. *(Jensen's inequality)* for $\varphi : \mathbb{R} \to \mathbb{R}$ convex, $\varphi(\mathbb{E}[X\mid\mathcal{F}]) \le \mathbb{E}[\varphi(X)\mid\mathcal{F}]$ $\mathbb{P}$-a.e.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Sketch of Proof of Theorem 3.3.15 (linearity and monotonicity)</summary>

**(i) (linearity):** For $\alpha \in \mathbb{R}$ and $X, Y \in L^1(\Omega, \mathbb{P}; \mathbb{R})$, the function $\mathbb{E}[X\mid\mathcal{F}] + \alpha \mathbb{E}[Y\mid\mathcal{F}]$ is $\mathcal{F}$-measurable, and satisfies for every $A \in \mathcal{F}$

$$\mathbb{E}\bigl[\mathbb{1}_A (\mathbb{E}[X\mid\mathcal{F}] + \alpha \mathbb{E}[Y\mid\mathcal{F}])\bigr] = \mathbb{E}[\mathbb{1}_A \mathbb{E}[X\mid\mathcal{F}]] + \alpha \mathbb{E}[\mathbb{1}_A \mathbb{E}[Y\mid\mathcal{F}]] = \mathbb{E}[\mathbb{1}_A X] + \alpha \mathbb{E}[\mathbb{1}_A Y] = \mathbb{E}[\mathbb{1}_A (X + \alpha Y)].$$

By uniqueness, this equals $\mathbb{E}[X + \alpha Y\mid\mathcal{F}]$.

**(ii) (monotonicity):** Let $A := \lbrace \mathbb{E}[X\mid\mathcal{F}] < \mathbb{E}[Y\mid\mathcal{F}] \rbrace \in \mathcal{F}$. Due to $X \ge Y$, $\mathbb{E}[\mathbb{1}_A(X - Y)] \ge 0$, and thus $\mathbb{P}[A] = 0$.

**(iv) (triangle inequality):** Set $X^+ := \max\lbrace 0, X \rbrace$ and $X^- := -\min\lbrace 0, X \rbrace$, so that $X = X^+ - X^-$. By (i) and (ii)

$$\mathbb{E}[\|X\|\mid\mathcal{F}] = \mathbb{E}[X^+\mid\mathcal{F}] + \mathbb{E}[X^-\mid\mathcal{F}] \ge \mathbb{E}[-X^+\mid\mathcal{F}] + \mathbb{E}[X^-\mid\mathcal{F}] = -\mathbb{E}[X\mid\mathcal{F}] \qquad \mathbb{P}\text{-a.e.}$$

and similarly $\mathbb{E}[\|X\|\mid\mathcal{F}] \ge \mathbb{E}[X\mid\mathcal{F}]$, so $\mathbb{E}[\|X\|\mid\mathcal{F}] \ge \|\mathbb{E}[X\mid\mathcal{F}]\|$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.3.16</span><span class="math-callout__name">(Conditional Expectation is a Contraction)</span></p>

In the setting of Thm. 3.3.12 denote $T(X) = \mathbb{E}[X\mid\mathcal{F}]$. Then $T : L^1(\Omega, \mathcal{A}, \mathbb{P}; \mathbb{R}) \to L^1(\Omega, \mathcal{F}, \mathbb{P}; \mathbb{R})$ is linear and $\lVert T \rVert_{\mathcal{L}(L^1; L^1)} \le 1$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 3.3.16</summary>

According to Thm. 3.3.15(i), $T$ is linear. The bound on the norm of the operator follows from Thm. 3.3.15(iv) (triangle inequality) and the tower property (or Exercise 3.3.14, $\mathbb{E}[\mathbb{E}[X\mid\mathcal{F}]] = \mathbb{E}[X]$):

$$\lVert \mathbb{E}[X\mid\mathcal{F}] \rVert_{L^1} = \mathbb{E}\bigl[\|\mathbb{E}[X\mid\mathcal{F}]\|\bigr] \le \mathbb{E}\bigl[\mathbb{E}[\|X\|\mid\mathcal{F}]\bigr] = \mathbb{E}[\|X\|] = \lVert X \rVert_{L^1}.$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3.17</span><span class="math-callout__name">(Conditional Expectation in Banach Spaces)</span></p>

Let $V$ be a separable Banach space. Then $\mathbb{E}[X\mid\mathcal{F}]$ exists and is $\mathbb{P}$-a.e. unique.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3.3.17 (not examinable)</summary>

For $\mathcal{A}$-simple functions $Y : \Omega \to V$, $Y = \sum\_{j=1}^{n} \mathbb{1}\_{A\_j} v\_j$ with $A\_i \cap A\_j = \emptyset$ for all $i \neq j$, define $\tilde{T}(Y)$ via

$$\tilde{T}(Y)(\omega) := \sum_{j=1}^{n} \mathbb{E}[\mathbb{1}_{A_j}|\mathcal{F}](\omega) v_j = \sum_{j=1}^{n} T(\mathbb{1}_{A_j})(\omega) v_j,$$

with $T$ from Lemma 3.3.16. Then $\tilde{T}$ is a linear operator on the vector space of $V$-valued $\mathcal{A}$-simple functions, and we want to show that it can be extended to a bounded operator on all of $L^1(\Omega, \mathbb{P}; V)$.

Using linearity of $T$ and the fact that $T(\mathbb{1}\_{A\_j}) = \mathbb{E}[\mathbb{1}\_{A\_j}\mid\mathcal{F}]$ takes nonnegative values $\mathbb{P}$-a.e. (why?),

$$\begin{aligned}
\lVert \tilde{T}(Y) \rVert_{L^1(\Omega, \mathcal{F}, \mathbb{P}; V)} &= \int_\Omega \Big\lVert \sum_{j=1}^{n} T(\mathbb{1}_{A_j})(\omega) v_j \Big\rVert \, \mathrm{d}\mathbb{P}(\omega) \le \int_\Omega \sum_{j=1}^{n} |T(\mathbb{1}_{A_j})(\omega)| \lVert v_j \rVert \, \mathrm{d}\mathbb{P}(\omega) \\
&= \int_\Omega \Big| T\Big( \sum_{j=1}^{n} \mathbb{1}_{A_j} \lVert v_j \rVert \Big)(\omega) \Big| \, \mathrm{d}\mathbb{P}(\omega) \le \lVert T \rVert_{\mathcal{L}(L^1; L^1)} \Big\lVert \sum_{j=1}^{n} \mathbb{1}_{A_j} \lVert v_j \rVert \Big\rVert_{L^1(\Omega, \mathbb{P}; \mathbb{R})} \\
&= \lVert T \rVert_{\mathcal{L}(L^1; L^1)} \lVert Y \rVert_{L^1(\Omega, \mathcal{A}, \mathbb{P}; V)}.
\end{aligned}$$

By density of the $\mathcal{B}(V)$-simple functions in $L^1(\Omega, \mathbb{P}; V)$, we conclude that $\tilde{T}$ can be extended to a bounded linear operator $\tilde{T} : L^1(\Omega, \mathcal{A}, \mathbb{P}; V) \to L^1(\Omega, \mathcal{F}, \mathbb{P}; V)$ with $\lVert \tilde{T} \rVert\_{\mathcal{L}(L^1; L^1)} \le \lVert T \rVert\_{\mathcal{L}(L^1; L^1)} = 1$.

Now we show that $\tilde{T}(X) = \mathbb{E}[X\mid\mathcal{F}]$ in the sense of Def. 3.3.11. By definition $\tilde{T}(X)$ is $\mathcal{F}$-measurable. Moreover for $A \in \mathcal{F}$ and $\mathcal{A}$-simple random variables $X : \Omega \to V$ one checks that $\mathbb{E}[\mathbb{1}\_A \tilde{T}(X)] = \mathbb{E}[\mathbb{1}\_A X]$. By density the equality holds for all $X \in L^1(\Omega, \mathcal{A}, \mathbb{P}; V)$, and therefore $\tilde{T}(X)$ is a conditional expectation.

Finally we show that $\mathbb{E}[X\mid\mathcal{F}]$ is $\mathbb{P}$-a.e. unique. Assume that $Z$ and $Z'$ are two conditional expectations. For arbitrary $\varphi \in V'$, $\langle Z, \varphi \rangle$ and $\langle Z', \varphi \rangle$ are (strongly) $\mathcal{F}$-measurable (by Cor. B.3.9) and satisfy $\mathbb{E}[\mathbb{1}\_A \langle Z, \varphi \rangle] = \mathbb{E}[\mathbb{1}\_A \langle Z', \varphi \rangle] = \mathbb{E}[\mathbb{1}\_A \langle X, \varphi \rangle]$ for all $A \in \mathcal{F}$ (see Lemma 3.1.4). This shows that $\langle Z, \varphi \rangle$ and $\langle Z', \varphi \rangle$ are both conditional expectations of $\langle X, \varphi \rangle$, and by Thm. 3.3.12 there exists a $\mathbb{P}$-null set $N \subseteq \Omega$ such that $\langle Z, \varphi \rangle = \langle Z', \varphi \rangle$ on $N^c$. Since $V$ is separable, (as shown earlier) there exists a sequence $\varphi\_n \in V'$ with $\lVert \varphi\_n \rVert\_{V'} = 1$, and such that $\lVert v \rVert = \sup\_{n \in \mathbb{N}} \langle v, \varphi\_n \rangle$ for all $v \in V$. Let $N\_n$ be a $\mathbb{P}$-null set such that $\langle Z, \varphi\_n \rangle = \langle Z', \varphi\_n \rangle$ on $N\_n^c$. Then $N := \bigcup\_{n \in \mathbb{N}} N\_n$ is a null set and $Z = Z'$ on $N^c$.

</details>
</div>

Now we can introduce the conditional expectation of $X$ given $Y$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.18</span><span class="math-callout__name">(Conditional Expectation II)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space, $V$, $W$ two separable Banach spaces and $X : \Omega \to V$, $Y : \Omega \to W$ two random variables such that $X \in L^1(\Omega, \mu; V)$. Then $\mathbb{E}[X\mid Y] := \mathbb{E}[X\mid\sigma(Y)]$ is the **conditional expectation of $X$ given $Y$**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.3.19</span><span class="math-callout__name">(Conditional Expectation for Dice Roll)</span></p>

Let $X : \Omega \to \lbrace 0, 1 \rbrace$ be as in Example 3.2.6, i.e. $X$ is 0 if the dice shows an even number and $X$ is 1 if the dice shows an odd number. Let $Y : \Omega \to \lbrace 0, 1 \rbrace$ with $Y(\omega) = 0$ for $\omega \in \lbrace 1, 2, 3 \rbrace$ and $Y(\omega) = 1$ for $\omega \in \lbrace 4, 5, 6 \rbrace$. Then

$$\mathbb{E}[X|Y](\omega) = \begin{cases} 1/3 & \omega \in \lbrace 1, 2, 3 \rbrace \\ 2/3 & \omega \in \lbrace 4, 5, 6 \rbrace. \end{cases}$$

</div>

For a probability space $(\Omega, \mathcal{A}, \mathbb{P})$ and an event $A \in \mathcal{A}$ it holds $\mathbb{P}[A] = \mathbb{E}[\mathbb{1}\_A]$. This motivates:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.20</span><span class="math-callout__name">(Conditional Probability II)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space and $A \in \mathcal{A}$. Let $\mathcal{F} \subseteq \mathcal{A}$ be a sub-$\sigma$-algebra and let $Y : \Omega \to V$ be a $V$-valued RV. Then we define

1. the **conditional probability of $A$ given $\mathcal{F}$** as $\mathbb{P}[A\mid\mathcal{F}] := \mathbb{E}[\mathbb{1}\_A\mid\mathcal{F}]$ and
2. the **conditional probability of $A$ given $Y$** as $\mathbb{P}[A\mid Y] := \mathbb{E}[\mathbb{1}\_A\mid\sigma(Y)]$.

</div>

Note that $\mathbb{P}[A\mid Y]$ is again a RV, that is $A \mapsto \mathbb{P}[A\mid Y]$ is a mapping from events to RVs. Furthermore, one can show that this mapping is $\sigma$-additive in the sense $\mathbb{P}[\bigcup\_{j \in \mathbb{N}} A\_j \mid Y] = \sum\_{j \in \mathbb{N}} \mathbb{P}[A\_j \mid Y]$ $\mathbb{P}$-a.e. for pairwise disjoint $A\_j \in \mathcal{A}$.

#### 3.3.3 Regular Conditional Distribution

So far we have defined $\mathbb{P}[A\mid B] = \frac{\mathbb{P}[A \cap B]}{\mathbb{P}[B]}$ in case $\mathbb{P}[B] > 0$. The goal of this section is to define the conditional probability $\mathbb{P}[A\mid X = x]$ even if $\mathbb{P}[X = x] = 0$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.3.21</span><span class="math-callout__name">(Why $\mathbb{P}[X=1\mid p=1/2]$ needs care)</span></p>

Let $p$ be a uniformly distributed RV on $[0, 1]$ and let $X$ be a Bernoulli RV, i.e. $X$ takes the value 1 with probability $p$ and the value 0 with probability $1 - p$. What is $\mathbb{P}[X = 1 \mid p = 1/2]$? Our previous definition of conditional probabilities doesn't lead to a meaningful result here since $[p = 1/2]$ is an event of probability 0. Intuitively we expect $\mathbb{P}[X = 1 \mid p = 1/2] = 1/2$, but how to make this rigorous?

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.22</span><span class="math-callout__name">(Transition/Markov Kernel)</span></p>

Let $(\Omega_1, \mathcal{A}_1)$ and $(\Omega_2, \mathcal{A}_2)$ be two measurable spaces. A map $\kappa : \Omega_1 \times \mathcal{A}_2 \to [0, \infty]$ is called a **transition kernel** from $\Omega_1$ to $\Omega_2$, if

1. $\omega \mapsto \kappa(\omega, A_2)$ is $\mathcal{A}_1$-measurable for each $A_2 \in \mathcal{A}_2$,
2. $A_2 \mapsto \kappa(\omega, A_2)$ is a $\sigma$-finite measure on $(\Omega_2, \mathcal{A}_2)$ for each $\omega \in \Omega_1$.

If the measure in (ii) is a probability measure for each $\omega \in \Omega_1$ then $\kappa$ is called a **Markov kernel**.

</div>

Using a Markov kernel we can now define a more "robust" notion of conditional distributions:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.23</span><span class="math-callout__name">(Regular Conditional Distribution)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space and $X : \Omega \to V$ a random variable in a Banach space $V$.

1. Let $\mathcal{F} \subseteq \mathcal{A}$ be a sub-$\sigma$-algebra. Then a Markov kernel $\kappa\_{X\mid\mathcal{F}} : \Omega \times \mathcal{B}(V) \to [0, \infty]$ from $(\Omega, \mathcal{F})$ to $(V, \mathcal{B}(V))$ such that $\kappa\_{X\mid\mathcal{F}}(\omega, B) = \mathbb{P}[X \in B\mid\mathcal{F}](\omega)$ for every $B \in \mathcal{B}(V)$ $\mathbb{P}$-a.e., i.e.

   $$\mathbb{P}[A \cap [X \in B]] = \int_\Omega \mathbb{1}_B(X(\omega)) \mathbb{1}_A(\omega) \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \kappa_{X|\mathcal{F}}(\omega, B) \mathbb{1}_A(\omega) \, \mathrm{d}\mathbb{P}(\omega) \qquad \forall A \in \mathcal{F},\ B \in \mathcal{B}(V),$$

   is called a **regular (version of the) conditional distribution of $X$ given $\mathcal{F}$**.

2. In the special case $\mathcal{F} = \sigma(Y)$, where $Y : \Omega \to W$ is a second RV in a Banach space $W$, the map $\tau\_{X\mid Y} : W \times \mathcal{B}(V) \to [0, 1]$ defined by

   $$\tau_{X|Y}(y, B) := \kappa_{X|\sigma(Y)}(Y^{-1}(y), B)$$

   is called a **regular (version of the) conditional distribution of $X$ given $Y$**. In this case we write

   $$\mathbb{P}[X \in B | Y = y] := \tau_{X|Y}(y, B).$$

</div>

Assuming for the moment that such a $\tau\_{X\mid Y}$ exists, then we have found a meaningful way to define the probability distribution of $X$ given that $Y = y$, namely the measure $B \mapsto \mathbb{P}[X \in B \mid Y = y]$. This is well-defined even if $\lbrace Y = y \rbrace$ is a (nonempty) $\mathbb{P}$-null set. In this sense, $\mathbb{P}[X \in \cdot \mid Y = y]$ can be interpreted as a well behaved conditional probability.

It remains to show that $\tau\_{X\mid Y}$ exists and is unique (in a suitable sense), to which the rest of this section is dedicated. We emphasize that the existence of regular conditional distributions is not trivial, and indeed not always satisfied. However, in the present setting, where $V$ and $W$ are separable Banach spaces, existence does hold.

In fact, the assumptions that $V$ and $W$ are separable Banach spaces could be significantly weakened, in particular it would suffice for $W$ equipped with some $\sigma$-algebra to be a measurable space. Such generalizations are beyond the scope of these lecture notes.

**Existence.** The next theorem shows existence of $\kappa\_{X\mid\mathcal{F}}$ in the case $(V, \mathcal{B}(V)) = (\mathbb{R}, \mathcal{B}(\mathbb{R}))$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3.24</span><span class="math-callout__name">(Existence of Regular Conditional Distribution for $\mathbb{R}$)</span></p>

Let $X : (\Omega, \mathcal{A}, \mathbb{P}) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$ be a real valued random variable and $\mathcal{F} \subseteq \mathcal{A}$ a sub-$\sigma$-algebra. Then there exists a regular version $\kappa_{X\mid\mathcal{F}} : \Omega \times \mathcal{B}(\mathbb{R}) \to [0, \infty]$ of the conditional distribution of $X$ given $\mathcal{F}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3.3.24 (not examinable)</summary>

We proceed as follows: We construct a measurable version of the distribution function of the conditional distribution by first defining it on the countable set of rational numbers, and then extend it to the real numbers. Throughout this proof we write $\kappa$ for $\kappa\_{X\mid\mathcal{F}}$.

**Step 1.** We construct a function $\tilde{F} : \Omega \times \mathbb{R} \to [0, 1]$ such that $q \mapsto \tilde{F}(\omega, q)$ is the distribution function of the measure $\kappa(\omega, \cdot)$. To this end, for every $q \in \mathbb{Q}$ let $\omega \mapsto F(\cdot, q) : (\Omega, \mathcal{F}) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$ be a fixed version of the conditional probability

$$\mathbb{P}[X \in (-\infty, q]|\mathcal{F}] = \mathbb{E}[\mathbb{1}_{X \in (-\infty, q]}|\mathcal{F}] : \Omega \to \mathbb{R}$$

(remember that the conditional probability is only unique $\mathbb{P}$-a.e.). For any $q \le r \in \mathbb{Q}$ it holds $\mathbb{1}\_{X \in (-\infty, q]} \le \mathbb{1}\_{X \in (-\infty, r]}$ and by the monotonicity of the conditional expectation (Thm. 3.3.15(ii)) there is a null set $A\_{q,r} \in \mathcal{F}$ such that

$$F(\omega, q) \le F(\omega, r) \qquad \forall \omega \in \Omega \setminus A_{q,r}.$$

By Lebesgue dominated convergence (cp. Thm. 3.3.15(vi)), there are null sets $B\_q \in \mathcal{F}$ for every $q \in \mathbb{Q}$ such that

$$\lim_{n \to \infty} F\Big(\omega, q + \frac{1}{n}\Big) = \lim_{n \to \infty} \mathbb{E}[\mathbb{1}_{X \in (-\infty, q + \frac{1}{n}]}|\mathcal{F}](\omega) = \mathbb{E}[\mathbb{1}_{X \in (-\infty, q]}|\mathcal{F}](\omega) = F(\omega, q) \qquad \forall \omega \in \Omega \setminus B_q,$$

and by the same argument there exists a null set $C \in \mathcal{F}$ such that

$$\inf_{n \in \mathbb{N}} F(\omega, -n) = \lim_{n \to \infty} F(\omega, -n) = \mathbb{E}[0|\mathcal{F}](\omega) = 0, \qquad \sup_{n \in \mathbb{N}} F(\omega, n) = \lim_{n \to \infty} F(\omega, n) = \mathbb{E}[1|\mathcal{F}](\omega) = 1 \qquad \forall \omega \in \Omega \setminus C.$$

Now set $N := \bigcup\_{q, r \in \mathbb{Q}} A\_{q,r} \cup \bigcup\_{q \in \mathbb{Q}} B\_q \cup C$. Then $N \in \mathcal{F}$ and $\mathbb{P}[N] = 0$. Define

$$\tilde{F}(\omega, z) := \inf \lbrace F(\omega, q) : z < q \in \mathbb{Q} \rbrace \qquad z \in \mathbb{R},\ \omega \in \Omega \setminus N.$$

Then $z \mapsto \tilde{F}(\omega, z)$ is monotonically increasing, right-continuous and satisfies $\lim\_{z \to \infty} F(\omega, z) = 1$ and $\lim\_{z \to \infty} F(\omega, -z) = 0$. As such it is a distribution function, i.e. $\mu\_\omega((a, b]) := F(\omega, b) - F(\omega, a)$ defines a probability measure on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$. For $\omega \in N$ set $F(\omega, z) := F\_0(z)$ where $F\_0$ is an arbitrary fixed probability distribution function, and again $\mu\_\omega((a, b]) := F\_0(b) - F\_0(a)$ defines a probability measure on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$.

**Step 2.** We define $\kappa$ and show that it possesses the properties (i) and (ii) of Def. 3.3.22. For $B \in \mathcal{B}(\mathbb{R})$ set

$$\kappa(\omega, B) := \mu_\omega(B).$$

By construction, for each $\omega \in \Omega$, $\kappa(\omega, \cdot)$ is a probability measure on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$.

It remains to show that for each $B \in \mathcal{B}(\mathbb{R})$ the map $\omega \mapsto \kappa(\omega, B)$ is $\mathcal{F}$-measurable. First let $q \in \mathbb{Q}$ and set $B := (-\infty, q]$. Then

$$\kappa(\omega, B) = F(\omega, q) \mathbb{1}_{N^c}(\omega) + F_0(q) \mathbb{1}_N(\omega).$$

Since $N \in \mathcal{F}$ and $\omega \mapsto F(\omega, q)$ is $\mathcal{F}$-measurable by construction, $\omega \mapsto \kappa(\omega, B)$ is $\mathcal{F}$-measurable. Next, note that with

$$\mathcal{C} := \lbrace (-\infty, q] : q \in \mathbb{Q} \rbrace \tag{3.3.1}$$

it holds $\sigma(\mathcal{C}) = \mathcal{B}(\mathbb{R})$ (i.e. $\mathcal{C}$ generates the Borel-$\sigma$-algebra). We claim that

$$\mathcal{D} := \lbrace B \in \mathcal{B}(\mathbb{R}) : \omega \mapsto \kappa(\omega, B) \text{ is } \mathcal{F}\text{-measurable} \rbrace$$

is a $\sigma$-algebra. In this case $\mathcal{D} \supseteq \sigma(\mathcal{C}) = \mathcal{B}(\mathbb{R})$, which then shows that $\omega \mapsto \kappa(\omega, B)$ is $\mathcal{F}$-measurable for all $B \in \mathcal{B}(\mathbb{R})$.

To show the claim we first point out that $\mathcal{D}$ is a Dynkin-system:

* $\mathbb{R} \in \mathcal{D}$ since $\omega \mapsto \kappa(\omega, \mathbb{R}) = \mu\_\omega(\mathbb{R}) = 1$ is trivially $\mathcal{F}$-measurable,
* for $A, B \in \mathcal{D}$ with $A \subseteq B$ it holds $B \setminus A \in \mathcal{D}$ due to

  $$\kappa(\omega, B \setminus A) = \kappa(\omega, B) - \kappa(\omega, A), \tag{3.3.2}$$

  i.e. $\omega \mapsto \kappa(\omega, B \setminus A)$ is $\mathcal{F}$-measurable since it is the sum of two $\mathcal{F}$-measurable functions (we have used that $A \mapsto \kappa(\omega, A)$ is a probability measure in (3.3.2)),
* for disjoint sets $(A\_j)\_{j \in \mathbb{N}}$ in $\mathcal{D}$ we have $\bigcup\_{j \in \mathbb{N}} A\_j \in \mathcal{D}$ since

  $$\kappa\Big(\omega, \bigcup_{j \in \mathbb{N}} A_j\Big) = \sum_{j \in \mathbb{N}} \kappa(\omega, A_j),$$

  and this sum converges pointwise for every $\omega \in \Omega$ to a number in $[0, 1]$ since $\kappa(\omega, \cdot)$ is a probability measure. Thus $\omega \mapsto \kappa(\omega, \bigcup\_{j \in \mathbb{N}} A\_j) \in \mathbb{R}$ is $\mathcal{F}$-measurable as the pointwise limit of $\mathcal{F}$-measurable functions (cp. Prop. B.3.6).

By Prop. B.1.8 (and because $\mathcal{C}$ satisfies $A, B \in \mathcal{C} \Rightarrow A \cap B \in \mathcal{C}$) we conclude $\mathcal{B}(\mathbb{R}) = \sigma(\mathcal{C}) \subseteq \mathcal{D}$.

**Step 3.** Finally we verify that $\kappa$ satisfies $\kappa(\omega, B) = \mathbb{P}[X \in B\mid\mathcal{F}](\omega)$ of Def. 3.3.23(i) and is thus a regular conditional distribution of $X$ given $\mathcal{F}$.

By definition of $\kappa$, for every $A \in \mathcal{F}$, $q \in \mathbb{Q}$ and $B = (-\infty, q]$

$$\int_\Omega \mathbb{1}_A(\omega) \kappa(\omega, B) \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \mathbb{1}_A(\omega) \mathbb{P}[X \in B|\mathcal{F}](\omega) \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \mathbb{1}_A(\omega) \mathbb{E}[\mathbb{1}_{X \in B}|\mathcal{F}](\omega) \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \mathbb{1}_{A \cap \lbrace X \in B \rbrace}(\omega) \, \mathrm{d}\mathbb{P}(\omega) = \mathbb{P}[A \cap \lbrace X \in B \rbrace]. \tag{3.3.3}$$

Since $\mathcal{C}$ in (3.3.1) generates $\mathcal{B}(\mathbb{R})$, and because of Thm. B.2.5 the left and the right-hand side of (3.3.3) coincide in the sense of finite measures on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$. Thus they are equal for all $B \in \mathcal{B}(\mathbb{R})$.

Now fix $B \in \mathcal{B}(\mathbb{R})$ and assume that there exists $A \in \mathcal{F}$ with $\mathbb{P}[A] > 0$ and such that $\kappa(\omega, B) \neq \mathbb{P}[X \in B\mid\mathcal{F}]$ for all $\omega \in A$. Without loss of generality we can assume that $\kappa(\omega, B) - \mathbb{P}[X \in B\mid\mathcal{F}] > \varepsilon$ for some $\varepsilon > 0$. But then $\int\_\Omega \mathbb{1}\_A(\omega) \kappa(\omega, B) \, \mathrm{d}\mathbb{P}(\omega) - \int\_\Omega \mathbb{1}\_A(\omega) \mathbb{P}[X \in B\mid\mathcal{F}](\omega) \, \mathrm{d}\mathbb{P}(\omega) \ge \varepsilon \mathbb{P}[A] \neq 0$. Thus such $A$ cannot exist and we conclude that $\kappa(\cdot, B) = \mathbb{P}[X \in B\mid\mathcal{F}]$ $\mathbb{P}$-a.e. for every $B \in \mathcal{B}(\mathbb{R})$.

</details>
</div>

To obtain a version of the above theorem for separable Banach spaces $V$, we need the following notion:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.3.25</span><span class="math-callout__name">(Borel Space)</span></p>

Two measurable spaces $(\Omega, \mathcal{A})$ and $(\tilde{\Omega}, \tilde{\mathcal{A}})$ are **isomorphic** if there exists a bijection $\varphi : \Omega \to \tilde{\Omega}$ such that $\varphi$ is $\mathcal{A}/\tilde{\mathcal{A}}$-measurable and $\varphi^{-1}$ is $\tilde{\mathcal{A}}/\mathcal{A}$-measurable. We call $(\Omega, \mathcal{A})$ a **Borel space** if there exists $B \in \mathcal{B}(\mathbb{R})$ such that $(B, \mathcal{B}(B))$ and $(\Omega, \mathcal{A})$ are isomorphic.

</div>

We state without proof:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3.26</span></p>

Let $V$ be a separable Banach space. Then $(V, \mathcal{B}(V))$ is a Borel space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.3.27</span></p>

Let $d \in \mathbb{N}$. Show that $([0, 1]^d, \mathcal{B}([0, 1]^d))$ is a Borel space. *(Hint: Use binary representations.)*

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.3.28</span><span class="math-callout__name">(Existence of Regular Conditional Distribution in Banach Spaces)</span></p>

Let $X : (\Omega, \mathcal{A}, \mathbb{P}) \to V$ be a RV, $(V, \mathcal{B}(V))$ a separable Banach space and $\mathcal{F} \subseteq \mathcal{A}$ a sub-$\sigma$-algebra. Then there exists a regular version $\kappa_{X\mid\mathcal{F}} : \Omega \times \mathcal{B}(V) \to \mathbb{R}$ of the conditional distribution of $X$ given $\mathcal{F}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Corollary 3.3.28</summary>

Let $B \in \mathcal{B}(\mathbb{R})$ and $\varphi : (V, \mathcal{B}(V)) \to (B, \mathcal{B}(B))$ an isomorphism as in Def. 3.3.25, which exists by Thm. 3.3.26. Then $\tilde{X} := \varphi \circ X : \Omega \to \mathbb{R}$ is a real-valued RV, and by Thm. 3.3.24 there exists a regular version $\kappa\_{\tilde{X}\mid\mathcal{F}}$ of the conditional distribution of $\tilde{X}$ given $\mathcal{F}$. Set $\kappa\_{X\mid\mathcal{F}}(\omega, A) := \kappa\_{\tilde{X}\mid\mathcal{F}}(\omega, \varphi(A))$ for all $A \in \mathcal{B}(V)$. Then $\kappa\_{X\mid\mathcal{F}}$ is a regular version of the conditional distribution of $X$ given $\mathcal{F}$.

</details>
</div>

Finally, rather than conditioning on a sub-$\sigma$-algebra, we now wish to condition on $Y = y$ (i.e. on the event $\lbrace Y = y \rbrace \subseteq \Omega$). In order to do so, we need the Doob-Dynkin lemma:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.3.29</span><span class="math-callout__name">(Doob-Dynkin Lemma)</span></p>

Let $\Omega$ be a set and $(\tilde{\Omega}, \tilde{\mathcal{A}})$ a measurable space. Then $\kappa : \Omega \to \mathbb{R}$ is $\sigma(Y)/\mathcal{B}(\mathbb{R})$-measurable iff there exists $\tau : \tilde{\Omega} \to \mathbb{R}$ which is $\tilde{\mathcal{A}}/\mathcal{B}(\mathbb{R})$-measurable such that $\kappa = \tau \circ Y$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 3.3.29 ("$\Leftarrow$" direction)</summary>

**($\Leftarrow$)** If such $\tau$ exists, then $\tau \circ Y = \kappa$ is measurable as a composition of measurable functions.

The other direction can be proved as follows: First assume $\kappa : \Omega \to [0, \infty)$ (i.e. $\kappa$ is nonnegative) and $\kappa$ is $\sigma(Y)/\mathcal{B}(\mathbb{R})$-measurable.

- Define $\kappa_n := \min\lbrace n, 2^{-n} \lfloor 2^n \kappa \rfloor \rbrace$ and observe that $\kappa_n$ is a simple function and $\kappa_n \nearrow \kappa$.
- Use the $\kappa_n$ to construct sets $A_j \in \mathcal{A}$ and numbers $\alpha_j \ge 0$ such that $\kappa = \sum_{j \in \mathbb{N}} \alpha_j \mathbb{1}_{A_j}$.
- By definition of $\sigma(Y)$ there exist sets $B_n \in \tilde{\mathcal{A}}$ such that $Y^{-1}(B_n) = A_n$. Use the $B_n$ to define $\tau$ such that $\tau \circ Y = \kappa$.
- Conclude that $\tau$ also exists under the assumption that $\kappa : \Omega \to \mathbb{R}$ is $\sigma(Y)/\mathcal{B}(\mathbb{R})$-measurable (i.e. is not necessarily nonnegative) by decomposing $\kappa = \kappa^+ - \kappa^-$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3.31</span><span class="math-callout__name">(Existence and Uniqueness of the Conditional Distribution Given $Y$)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space, and $X : \Omega \to V$, $Y : \Omega \to W$ two RVs for two separable Banach spaces $V$ and $W$. Then there exists a regular version of the conditional distribution $\mathbb{P}[X \in \cdot \mid Y = y]$ (in the sense of Def. 3.3.23).

It is unique in the sense that for all other regular versions $\tilde{\tau}$ of the conditional distribution there exists a $\mathbb{P}\_Y$-null set $N \in \mathcal{B}(W)$ such that $\mathbb{P}[X \in \cdot \mid Y = y] = \tilde{\tau}(y, \cdot)$ as probability measures on $(V, \mathcal{B}(V))$, for all $y \in N^c \cap \lbrace Y(\omega) : \omega \in \Omega \rbrace$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3.3.31 (Existence)</summary>

By Corollary 3.3.28 there exists a regular version $\kappa_{X\mid\sigma(Y)}(\omega, B)$ of the conditional distribution. By the Doob-Dynkin Lemma (Lemma 3.3.29, with $(\tilde{\Omega}, \tilde{\mathcal{A}}) = (W, \mathcal{B}(W))$), for every $B \in \mathcal{B}(V)$, there exists $\tau(\cdot, B) : W \to \mathbb{R}$ such that

$$\kappa(\omega, B) = \tau(Y(\omega), B) \qquad \forall \omega \in \Omega.$$

Then $\tau$ satisfies

(i) $y \mapsto \tau(y, B)$ is $\mathcal{B}(W)/\mathcal{B}(\mathbb{R})$-measurable for every $B \in \mathcal{B}(V)$ by definition of $\tau$ in Lemma 3.3.29,

(ii) $B \mapsto \tau(y, B)$ is a probability measure on $(V, \mathcal{B}(V))$ for every $y \in \lbrace Y(\omega) : \omega \in \Omega \rbrace$, since this is true for $B \mapsto \kappa(\omega, B) = \tau(Y(\omega), B) = \tau(y, B)$ and $\omega \in Y^{-1}(y)$,

(iii) for any $B \in \mathcal{B}(V)$ and any $A \in \mathcal{B}(W)$, by Thm. 3.1.9 and the condition in Def. 3.3.23(i) with $\mathcal{F} = \sigma(Y)$ and $[Y \in A] \in \sigma(Y)$

$$\int_W \mathbb{1}_A(y) \tau(y, B) \, \mathrm{d}\mathbb{P}_Y = \int_\Omega \mathbb{1}_A(Y(\omega)) \tau(Y(\omega), B) \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \mathbb{1}_A(Y(\omega)) \kappa(\omega, B) \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \mathbb{1}_A(Y(\omega)) \mathbb{1}_B(X(\omega)) \, \mathrm{d}\mathbb{P}(\omega) = \mathbb{P}[X \in B, Y \in A].$$

This proves the existence of a regular version of the conditional distribution as in Def. 3.3.23(ii).

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 3.3.31 (Uniqueness, not examinable)</summary>

Denote $\tau(y, \cdot) := \mathbb{P}[X \in \cdot \mid Y = y]$ and let $\tilde{\tau}$ be another regular version of the conditional distribution of $X$ given $Y$ that satisfies Def. 3.3.23(ii).

Fix $B \in \mathcal{B}(V)$ and let $A\_n := \lbrace y \in W : \tau(y, B) - \tilde{\tau}(y, B) > \frac{1}{n} \rbrace$. Then $A\_n \in \mathcal{B}(W)$. Due to

$$\int_{A_n} \tau(y, B) \, \mathrm{d}\mathbb{P}_Y(y) = \mathbb{P}[X \in B, Y \in A_n] = \int_{A_n} \tilde{\tau}(y, B) \, \mathrm{d}\mathbb{P}_Y(y),$$

we find $0 = \int\_{A\_n} (\tau(y, B) - \tilde{\tau}(y, B)) \, \mathrm{d}\mathbb{P}\_Y(y) \ge \frac{1}{n} \mathbb{P}\_Y[A\_n]$. Hence $\lbrace y \in W : \tau(y, B) > \tilde{\tau}(y, B) \rbrace = \bigcup\_{n \in \mathbb{N}} A\_n$ is a $\mathbb{P}\_Y$-null set, and by symmetry we conclude that $A\_B := \lbrace y \in W : \tau(y, B) \neq \tilde{\tau}(y, B) \rbrace$ is a $\mathbb{P}\_Y$-null set.

Now fix a dense sequence $(x\_n)\_{n \in \mathbb{N}} \subseteq V$, such that with the open balls $B\_r(x) := \lbrace v \in V : \lVert v - x \rVert\_V < r \rbrace$,

$$\tilde{\mathcal{C}} := \lbrace B_{1/n}(x_m) : n, m \in \mathbb{N} \rbrace = \lbrace \tilde{C}_j : j \in \mathbb{N} \rbrace$$

is a countable basis of the topology of $V$ (i.e. $(\tilde{C}\_j)\_{j \in \mathbb{N}}$ is some fixed enumeration of the countable set $\tilde{\mathcal{C}}$). Then

$$\mathcal{C} := \lbrace \cap_{i \in I} \tilde{C}_i : I \subseteq \mathbb{N},\ |I| < \infty \rbrace = \lbrace C_j : j \in \mathbb{N} \rbrace$$

is a countable set of open sets (why is $\mathcal{C}$ countable?). Since $\tilde{\mathcal{C}}$ is a basis of the topology on $V$, it holds $\sigma(\tilde{\mathcal{C}}) = \mathcal{B}(V)$, and in particular $\sigma(\mathcal{C}) = \mathcal{B}(V)$. Furthermore, $\mathcal{C}$ has the property that for any $C\_i, C\_j \in \mathcal{C}$ also $C\_i \cap C\_j \in \mathcal{C}$ by definition of $\mathcal{C}$. Now choose for every $i \in \mathbb{N}$ a $\mathbb{P}\_Y$-null set $N\_i \in \mathcal{B}(W)$ such that $\tau(y, C\_i) = \tilde{\tau}(y, C\_i)$ for all $y \in W \setminus N\_i$. Then $N := \bigcup\_{i \in \mathbb{N}} N\_i$ is a $\mathbb{P}\_Y$-null set and for all $y \in W \setminus N$ and all $i \in \mathbb{N}$ holds $\tau(y, C\_i) = \tilde{\tau}(y, C\_i)$. Thm. B.2.5 implies $\tau(y, B) = \tilde{\tau}(y, B)$ for all $y \in \lbrace Y(\omega) : \omega \in \Omega \rbrace \setminus N$ and all $B \in \mathcal{B}(V)$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.3.32</span></p>

Due to $\tau$ only being unique in the above sense, we speak of regular *versions* of the conditional distribution. Often we will drop this term, and simply say that $\tau$ is a regular conditional distribution, with the understanding that such a map is only unique $\mathbb{P}\_Y$-a.e.

</div>

#### 3.3.4 Conditional Densities

Suppose that $X : \Omega \to \mathbb{R}^n$ and $Y : \Omega \to \mathbb{R}^m$ are two RVs on the probability space $(\Omega, \mathcal{A}, \mathbb{P})$. Assume that $(X, Y) : \Omega \to \mathbb{R}^{n+m}$ has the joint (measurable) density $f\_{X,Y} : \mathbb{R}^{n+m} \to [0, \infty)$.

Then for any $A \in \mathcal{B}(\mathbb{R}^m)$, by Fubini's theorem

$$\mathbb{P}[Y \in A] = \mathbb{P}[X \in \mathbb{R}^n, Y \in A] = \int_{\mathbb{R}^n \times A} f_{X,Y}(x, y) \, \mathrm{d}(x, y) = \int_A \int_{\mathbb{R}^n} f_{X,Y}(x, y) \, \mathrm{d}x \, \mathrm{d}y.$$

Hence the marginal $Y : \Omega \to \mathbb{R}^m$ has a density, which is given by

$$f_Y(y) := \int_{\mathbb{R}^n} f_{X,Y}(x, y) \, \mathrm{d}x.$$

We also say $f\_Y$ is the **marginal density** of $Y$. We point out that we use here the fact that $y \mapsto \int\_{\mathbb{R}^n} f\_{X,Y}(x, y) \, \mathrm{d}x$ is measurable, which is also a consequence of Fubini's theorem. Next, let us consider a regular version of the conditional distribution $\mathbb{P}[X \in \cdot \mid Y = y]$ of $X$ given $Y = y$. It turns out that in the present setting (a version of the) measure $\mathbb{P}[X \in \cdot \mid Y = y]$ has a density, which we call the **conditional density**, and denote by

$$f_{X|Y}(\cdot | y) := \frac{\mathrm{d}\mathbb{P}[X \in \cdot | Y = y]}{\mathrm{d}\lambda_n}.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.3.33</span><span class="math-callout__name">(Conditional Density Formula)</span></p>

It holds that

$$f_{X|Y}(x|y) = \begin{cases} \frac{f_{X,Y}(x, y)}{f_Y(y)} & \text{if } f_Y(y) \in (0, \infty) \\ f_0(x) & \text{if } f_Y(y) \in \lbrace 0, \infty \rbrace \end{cases} \tag{3.3.4}$$

for some fixed probability density $f\_0$ on $\mathbb{R}^n$, is a density of (a version of) $\mathbb{P}[X \in \cdot \mid Y = y]$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.3.34</span></p>

The set $\lbrace y \in \mathbb{R}^m : f\_Y(y) = 0 \rbrace$ is a $\mathbb{P}\_Y$-null set, and also $\lbrace y \in \mathbb{R}^m : f\_Y(y) = \infty \rbrace$ is a $\lambda$-null set (and thus a $\mathbb{P}\_Y$-null set) since otherwise $\int\_{\mathbb{R}^{n+m}} f\_{X,Y}(x, y) \, \mathrm{d}x \, \mathrm{d}y = \int\_{\mathbb{R}^m} f\_Y(y) \, \mathrm{d}y$ would not be finite. By Thm. 3.3.31, the conditional distribution is only unique $\mathbb{P}\_Y$-a.e., hence in (3.3.4) it doesn't matter how we define $f\_{X\mid Y}(x \mid y)$ for $y$ with $f\_Y(y) \in \lbrace 0, \infty \rbrace$, in agreement with our definition of conditional probabilities. In practice, $f\_{X\mid Y}(\cdot \mid y)$ is only defined for $y$ with $f\_Y(y) \in (0, \infty)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 3.3.33</summary>

The measure $\mathbb{P}[X \in \cdot \mid Y = y]$ on $(\mathbb{R}^n, \mathcal{B}(\mathbb{R}^n))$ exists due to Thm. 3.3.31. For $y \in \mathbb{R}^m$ and $B \in \mathcal{B}(\mathbb{R}^n)$ set

$$\tau(y, B) := \int_B f_{X|Y}(x|y) \, \mathrm{d}x$$

(with $f\_{X\mid Y}(x \mid y)$ as defined in (3.3.4)). This is a version of $\mathbb{P}[X \in \cdot \mid Y = y]$, since

(i) $y \mapsto \tau(y, B)$ is measurable for every $B \in \mathcal{B}(\mathbb{R}^n)$ (this is a consequence of Fubini's theorem),

(ii) $B \mapsto \tau(y, B)$ is a probability measure for every $y \in \mathbb{R}^m$ since $\int\_{\mathbb{R}^n} f\_{X\mid Y}(x \mid y) \, \mathrm{d}x = 1$,

(iii) for every $B \in \mathcal{B}(\mathbb{R}^n)$ and every $A \in \mathcal{B}(\mathbb{R}^m)$, with the $\mathbb{P}\_Y$-null set $N := \lbrace y : f\_Y(y) \in \lbrace 0, \infty \rbrace \rbrace$,

$$\int_A \tau(y, B) \, \mathrm{d}\mathbb{P}_Y(y) = \int_{A \setminus N} \int_B f_Y(y) f_{X|Y}(x|y) \, \mathrm{d}x \, \mathrm{d}y = \int_{A \setminus N} \int_B f_{X,Y}(x, y) \, \mathrm{d}x \, \mathrm{d}y + \int_N \int_B f_{X,Y}(x, y) \, \mathrm{d}x \, \mathrm{d}y = \mathbb{P}[X \in B, Y \in A],$$

where we used $\int\_N \int\_B f\_{X,Y}(x, y) \, \mathrm{d}x \, \mathrm{d}y \le \int\_N f\_Y(y) \, \mathrm{d}y = \mathbb{P}\_Y[N] = 0$.

</details>
</div>

### 3.4 Some Common Distributions (Revision)

#### 3.4.1 Bernoulli

Given a parameter $0 \le p \le 1$, the Bernoulli RV $X \sim \mathrm{Ber}(p)$ is defined such that $\mathbb{P}[X = 1] = p$ and $\mathbb{P}[X = 0] = 1 - p$. This RV can be thought of as representing a coin flip with probability of heads equal to $p$. It is a special case of the binomial distribution with $n = 1$.

#### 3.4.2 Binomial

For $0 \le p \le 1$ and $n \in \mathbb{N}$, we define the binomial RV $X \sim \mathrm{Bin}(n, p)$ with probability mass function

$$\mathbb{P}[X = k] = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0 \ldots n.$$

This mass function can be thought of as the probability of $k$ heads in $n$ independent trials of a Bernoulli RV.

#### 3.4.3 Uniform

For $a < b$, the uniform RV $X \sim \mathrm{uniform}(a, b)$ has probability density function

$$f_X(x) = \begin{cases} \frac{1}{b-a} & \text{if } a \le x \le b \\ 0 & \text{otherwise.} \end{cases}$$

A uniform distribution assigns the same probability mass to all sub-intervals of the same length within its support.

#### 3.4.4 Exponential

For $\lambda > 0$, the exponential RV $X \sim \mathrm{Exp}(\lambda)$ has probability density function

$$f_X(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x \ge 0 \\ 0 & \text{otherwise.} \end{cases}$$

The exponential RV is memoryless: for $0 \le s < t$,

$$\mathbb{P}[X > s + t \mid X > s] = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = \mathbb{P}[X > t].$$

#### 3.4.5 Univariate Gaussian

For $\mu \in \mathbb{R}$ and $\sigma \in (0, \infty)$, the Gaussian RV $X \sim \mathcal{N}(\mu, \sigma^2)$ can be defined by the probability density function

$$f_X(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right).$$

#### 3.4.6 Multivariate Gaussian

First, let $X = (X_1, \ldots, X_n)$ be a vector of real-valued RVs. We say that $X_1, \ldots, X_n$ are "jointly normal" iff $a^\top X$ is Gaussian for every $a \in \mathbb{R}^n$. Equivalently, we say that $X$ has a multivariate Gaussian distribution, $X \sim N(\mu, \Sigma)$. $\mu \in \mathbb{R}^n$ is the mean of $X$ and $\Sigma \in \mathbb{R}^{n \times n}$ is the covariance of $X$. If $\Sigma$ is positive definite, then the probability density of $X$ is

$$f_X(x) = \frac{1}{(2\pi)^{n/2} \det(\Sigma)^{1/2}} \exp\left( -\frac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu) \right).$$

Jointly normal RVs $X_1, \ldots, X_n$ are independent iff they are uncorrelated. All marginal and conditional distributions of the multivariate Gaussian are (multivariate) Gaussian.

Note that if $\Sigma$ is not positive definite (in which case it will be positive semi-definite), $X$ can still be multivariate Gaussian. In this case, it is customary to describe $X$ through its **characteristic function**. For any RV $X$, the characteristic function $\phi_X$ is:

$$\phi_X(\lambda) = \mathbb{E}[e^{i\lambda^\top X}], \quad \lambda \in \mathbb{R}^n.$$

It is thus a function from the real numbers to the complex numbers; it always exists and completely characterizes the distribution. For a multivariate Gaussian, $X \sim N(\mu, \Sigma)$, we have $\phi_X(\lambda) = e^{i\lambda^\top \mu} e^{-\lambda^\top \Sigma \lambda / 2}$.

#### 3.4.7 Chi-squared

A chi-squared distributed RV with $k$ degrees of freedom, $X \sim \chi^2(k)$, is the distribution of a sum of the squares of $k$ independent standard normal RVs. Its probability density function (pdf) is

$$f_X(x) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{\frac{k}{2} - 1} e^{-\frac{x}{2}},$$

where $\Gamma(\cdot)$ is the gamma function.

A chi-squared RV $X \sim \chi^2(k)$ has mean $k$ and variance $2k$. Also note that the sum of chi-squared distributed RVs is also chi-squared distributed. Specifically, if $\lbrace X_i \rbrace_{i=1}^n$ are independent chi-squared variables with $\lbrace k_i \rbrace_{i=1}^n$ degrees of freedom, respectively, then the RV $Y = \sum_{i=1}^n X_i$ is chi-squared distributed with $\sum_{i=1}^n k_i$ degrees of freedom.

### 3.5 Distances and Divergences

Here we consider how to quantify the "difference" between probability measures. Some of these measures of "difference" are distance functions in the proper mathematical sense. Others do not satisfy the triangle inequality and are thus only so-called divergences.

Let $(\Omega, \mathcal{A})$ be a measurable space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.5.1</span><span class="math-callout__name">(Total Variation Distance)</span></p>

The **total variation distance** between two probability measures $\mathbb{P}$ and $\mathbb{Q}$ on $(\Omega, \mathcal{A})$ is defined as:

$$D_{\mathrm{TV}}(\mathbb{P}, \mathbb{Q}) = \sup_{A \in \mathcal{A}} |\mathbb{P}[A] - \mathbb{Q}[A]|.$$

</div>

This is the largest possible difference between the probabilities that the two distributions can assign to the same event.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.5.2</span></p>

If $\mathbb{P} \ll \mu$ and $\mathbb{Q} \ll \mu$, show that $D\_{\mathrm{TV}}(\mathbb{P}, \mathbb{Q}) = \frac{1}{2} \big( \int\_\Omega \big\lvert \frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu} - \frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu} \big\rvert \, \mathrm{d}\mu \big)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.5.3</span><span class="math-callout__name">(Hellinger Distance)</span></p>

Consider two probability measures $\mathbb{P}$ and $\mathbb{Q}$ that are absolutely continuous with respect to a third measure $\mu$ (such a measure always exists, for example $\frac{1}{2}(\mathbb{P} + \mathbb{Q})$). The **Hellinger distance** between $\mathbb{P}$ and $\mathbb{Q}$ is defined as:

$$D_{\mathrm{H}}(\mathbb{P}, \mathbb{Q}) = \left( \frac{1}{2} \int_\Omega \left( \sqrt{\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu}} - \sqrt{\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu}} \right)^2 \mathrm{d}\mu \right)^{1/2}. \tag{3.5.1}$$

</div>

If $\mu \ll \nu$, then

$$D_{\mathrm{H}}(\mathbb{P}, \mathbb{Q}) = \left( \frac{1}{2} \int_\Omega \left( \sqrt{\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu}} - \sqrt{\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu}} \right)^2 \frac{\mathrm{d}\mu}{\mathrm{d}\nu} \mathrm{d}\nu \right)^{1/2} = \left( \frac{1}{2} \int_\Omega \left( \sqrt{\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\nu}} - \sqrt{\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\nu}} \right)^2 \mathrm{d}\nu \right)^{1/2}, \tag{3.5.2}$$

and thus (3.5.1) does not depend on which measure $\mu$ was chosen. In particular, if $\Omega = \mathbb{R}^d$ and $\mathbb{P}$ and $\mathbb{Q}$ are absolutely continuous with respect to the Lebesgue measure $\lambda_d$, then Hellinger distance in (3.5.2) can be expressed through the probability densities $\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\lambda_d}$ and $\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\lambda_d}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.5.4</span></p>

In case $\mathbb{P} \ll \mathbb{Q}$, $D_{\mathrm{H}}(\mathbb{P}, \mathbb{Q}) = (\frac{1}{2} \int_\Omega (1 - \sqrt{\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mathbb{P}}})^2 \, \mathrm{d}\mathbb{P})^{1/2}$, and the normalization constant $\frac{1}{2}$ guarantees $D_{\mathrm{H}}(\mathbb{P}, \mathbb{Q}) \in [0, 1]$. A similar remark can be made about $D\_{\mathrm{TV}}$, cp. Exercise 3.5.2.

</div>

The Kullback-Leibler (KL) divergence (also called relative entropy) is a measure of how one probability distribution diverges from a second.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.5.5</span><span class="math-callout__name">(Kullback-Leibler Divergence)</span></p>

The **Kullback-Leibler divergence** between two probability measures $\mathbb{Q}$ and $\mathbb{P}$ is defined as:

$$D_{\mathrm{KL}}(\mathbb{P} \| \mathbb{Q}) = \begin{cases} \int_\Omega \log\left( \frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mathbb{Q}} \right) \mathrm{d}\mathbb{P} & \text{if } \mathbb{P} \ll \mathbb{Q} \\ \infty & \text{otherwise.} \end{cases}$$

</div>

We note that the KL divergence is not a distance metric, as it is **not symmetric**. In contrast, the total variation distance and the Hellinger distance are distance metrics. However, the KL divergence is non-negative and takes the value 0 iff $\mathbb{P} = \mathbb{Q}$. This result is known as *Gibb's inequality*.

If $\mathbb{P}$ and $\mathbb{Q}$ are equivalent,

$$D_{\mathrm{KL}}(\mathbb{P} \| \mathbb{Q}) = -\int_\Omega \log\left( \frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mathbb{P}} \right) \mathrm{d}\mathbb{P}.$$

If $\Omega = \mathbb{R}^d$ and $\mathbb{P}$ and $\mathbb{Q}$ have densities $p = \frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\lambda_d}$ and $q = \frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\lambda_d}$, the KL divergence can be written as

$$D_{\mathrm{KL}}(\mathbb{P} \| \mathbb{Q}) = \int_\Omega \log\left( \frac{p(x)}{q(x)} \right) p(x) \, \mathrm{d}x.$$

Compared to the total variation distance and the Hellinger distance, the KL divergence has computational advantages in certain situations. We can write the KL divergence as

$$D_{\mathrm{KL}}(\mathbb{P} \| \mathbb{Q}) = \int \log p(x) \, p(x) \, \mathrm{d}x - \int \log q(x) \, p(x) \, \mathrm{d}x,$$

where the second part of the right hand side is called the **cross entropy**,

$$H(\mathbb{P} \| \mathbb{Q}) = -\int \log q(x) \, p(x) \, \mathrm{d}x.$$

Given a set of samples drawn from $\mathbb{P}$, it is possible to compute the cross entropy for a given $\mathbb{Q}$ with known density function without knowing the density function of $\mathbb{P}$. This is particularly useful for many tasks in computational statistics such as importance sampling and density estimation.

The next proposition summarizes the most important relations between the above divergences. The proof is left as an exercise.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.5.6</span><span class="math-callout__name">(Relations Between Divergences)</span></p>

It holds

1. $D_{\mathrm{H}}(\mathbb{P}, \mathbb{Q})^2 \le D_{\mathrm{TV}}(\mathbb{P}, \mathbb{Q}) \le \sqrt{2} D_{\mathrm{H}}(\mathbb{P}, \mathbb{Q})$.

Moreover, if $\mathbb{P}$ and $\mathbb{Q}$ are equivalent

2. $D_{\mathrm{H}}(\mathbb{P}, \mathbb{Q})^2 \le \frac{1}{2} D_{\mathrm{KL}}(\mathbb{P} \| \mathbb{Q})$,
3. $D_{\mathrm{TV}}(\mathbb{P}, \mathbb{Q})^2 \le \frac{1}{2} D_{\mathrm{KL}}(\mathbb{P} \| \mathbb{Q})$.

</div>

Finally, we show how a bound on the Hellinger distance implies a bound on the difference of expectations taken with respect to two different probability measures.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.5.7</span><span class="math-callout__name">(Expectation Bound via Hellinger Distance)</span></p>

Let $\mathbb{P}$ and $\mathbb{Q}$ be two probability measures on a measurable space $(\Omega, \mathcal{A})$. Let $f : \Omega \to V$ be a RV for a separable Banach space $V$, such that $f$ has finite second moments with respect to both $\mathbb{P}$ and $\mathbb{Q}$. Then

$$\lVert \mathbb{E}_{\mathbb{P}}[f] - \mathbb{E}_{\mathbb{Q}}[f] \rVert \le 2(\mathbb{E}_{\mathbb{P}}[\lVert f \rVert^2] + \mathbb{E}_{\mathbb{Q}}[\lVert f \rVert^2])^{1/2} D_{\mathrm{H}}(\mathbb{P}, \mathbb{Q}).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 3.5.7</summary>

Let $\mathbb{P} \ll \mu$ and $\mathbb{Q} \ll \mu$ (such $\mu$ always exists, e.g. $\mu = \frac{1}{2}(\mathbb{P} + \mathbb{Q})$). Then

$$\lVert \mathbb{E}_{\mathbb{P}}[f] - \mathbb{E}_{\mathbb{Q}}[f] \rVert \le \int_\Omega \lVert f \rVert \left| \frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu} - \frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu} \right| \mathrm{d}\mu$$

$$= \int_\Omega \left( \frac{1}{\sqrt{2}} \left| \sqrt{\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu}} - \sqrt{\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu}} \right| \right) \left( \sqrt{2} \lVert f \rVert \left| \sqrt{\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu}} + \sqrt{\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu}} \right| \right) \mathrm{d}\mu$$

$$\le \left( \frac{1}{2} \int_\Omega \left( \sqrt{\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu}} - \sqrt{\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu}} \right)^2 \mathrm{d}\mu \right)^{1/2} \left( 2 \int_\Omega \lVert f \rVert^2 \left( \sqrt{\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu}} + \sqrt{\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu}} \right)^2 \mathrm{d}\mu \right)^{1/2}$$

$$\le \left( \frac{1}{2} \int_\Omega \left( \sqrt{\frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu}} - \sqrt{\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu}} \right)^2 \mathrm{d}\mu \right)^{1/2} \left( 4 \int_\Omega \lVert f \rVert^2 \left( \frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mu} + \frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mu} \right) \mathrm{d}\mu \right)^{1/2}$$

$$= 2 (\mathbb{E}_{\mathbb{P}}[\lVert f \rVert^2] + \mathbb{E}_{\mathbb{Q}}[\lVert f \rVert^2])^{1/2} D_{\mathrm{H}}(\mathbb{P}, \mathbb{Q}).$$

Here we used the Cauchy-Schwarz inequality, the factorisation $a^2 - b^2 = (a-b)(a+b)$ applied to $a = \sqrt{\mathrm{d}\mathbb{P}/\mathrm{d}\mu}$ and $b = \sqrt{\mathrm{d}\mathbb{Q}/\mathrm{d}\mu}$, and $(a+b)^2 \le 2(a^2 + b^2)$.

</details>
</div>


## Chapter 4: Bayesian Inversion

In this chapter we discuss the Bayesian approach towards inverse problems. In contrast to the methods of Chapter 2, in the Bayesian setting all involved quantities are modelled as random variables. As such, the question to be answered is not *what is the value of the unknown variable?*, but rather *what is the distribution of the unknown variable?* It turns out that this is a very powerful viewpoint, leading to a host of numerical methods with broad applications in statistics, applied mathematics and machine learning. Additionally, it has the mathematical advantage of yielding a well-posed inverse problem, as will be discussed in this chapter.

### 4.1 The Bayesian Inverse Problem

As in the previous chapter, we will use capital letters to denote RVs. We denote by $X$ the unknown of primary interest which we wish to identify, by $Y$ an observable quantity, and by $E$ a noise term. In the most general form, the model is described by a possibly nonlinear operator $\Phi$ such that

$$Y = \Phi(X, E).$$

Thus $\Phi$ ties together the three RVs $X$, $Y$ and $E$, and their probability distributions are interdependent. The RV $X$ will also be referred to as the *parameter* that we wish to infer. The RV $Y$ is often called the *measurement*, *observation* or *data*, and $E$ can be interpreted as a *measurement error*. The most common model for the measurement error is that of additive noise, i.e.

$$Y = \Phi(X) + E \tag{4.1.1}$$

and we will concentrate on this situation in the following. The interpretation is that we make an observation of $\Phi(X)$ that is polluted by the noise $E$.

For an underlying probability space $(\Omega, \mathcal{A}, \mathbb{P})$, the following assumptions are made throughout this chapter:

1. $X : \Omega \to V$ is a RV for some separable Banach space $V$. As such it has a distribution $\mathbb{P}_X$, which is called the **prior distribution** (or simply the prior). The prior is interpreted as the information available on $X$, *before* observing $Y$.
2. $E : \Omega \to W$ is a RV onto a second separable Banach space $W$, and $E$ and $X$ are independent.
3. $\Phi : V \to W$ is a Borel-measurable function. We will refer to it as the **forward operator**.

From (4.1.1) we observe that $Y : \Omega \to W$ is also a RV. Assuming that we observe $Y$ (i.e. we are given a realization $Y(\omega) = \Phi(X(\omega)) + E(\omega) \in W$ for some $\omega \in \Omega$), the Bayesian inverse problem is then to determine the distribution of $X$ conditioned on the event $[Y = y]$. Under the present assumptions, this distribution can be interpreted as pooling all available information that we (can) have on $X$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Terminology 4.1.1</span><span class="math-callout__name">(Posterior Distribution and Likelihood)</span></p>

1. Given a realization $y \in W$ of $Y$, the solution to the Bayesian inverse problem is the conditional distribution $\mathbb{P}[X \in \cdot \mid Y = y]$. We call $\mathbb{P}[X \in \cdot \mid Y = y]$ the **posterior distribution** (or simply the posterior) and denote it for short by $\mu\_{X\mid y}$.
2. Conversely, given a realization $x \in V$ of $X$, we will also need the conditional distribution $\mathbb{P}[Y \in \cdot \mid X = x]$, for short denoted by $\mu\_{Y\mid x}$. When it exists, the density of $\mathbb{P}[Y \in \cdot \mid X = x]$ is called the **likelihood** of $Y$ given $X = x$, as it expresses the likelihood of different measurement outcomes for fixed parameter $x$.

</div>

The algorithms discussed in Chapter 2 returned a point estimate $x$ for a given value $y$, for instance assuming the model $y = Ax$ for some matrix $A$. One advantage of Bayesian methods is, that they do not merely deliver point estimates, but acknowledge the fact, that we cannot know the exact value of $x$; for instance, there may exist multiple $x_j$ with $Ax_j = y$ due to $A$ being non-regular. This is reflected in the posterior being a distribution, and thus assigning probabilities to events of the type $[X \in B]$, $B \in \mathcal{B}(V)$. The posterior represents our knowledge and uncertainty about $X$.

In this chapter we investigate how to determine and explore the posterior.

### 4.2 The Finite Dimensional Case

As in Chapter 2, the finite dimensional setting is significantly easier, and in practice we will in general always work in finite (possibly very high) dimensions. Thus, let us restrict first to $V = \mathbb{R}^n$ and $W = \mathbb{R}^m$, and let $X : \Omega \to V$ and $Y : \Omega \to W$ be RVs with joint density $\pi_{X,Y}$.

We first introduce some shorter notation for the occurring distributions. We will assume in the following that real valued RVs are absolutely continuous w.r.t. the Lebesgue measures, and thus have densities.

* The prior distribution $\mathbb{P}_X$ on $(V, \mathcal{B}(V))$ will be denoted by $\mu_X$. If $V = \mathbb{R}^n$, we write $\pi_X(x)$ for its density.
* The posterior distribution $\mathbb{P}[X \in \cdot \mid Y = y]$ on $(V, \mathcal{B}(V))$ is denoted by $\mu_{X\mid y}$. If $V = \mathbb{R}^n$, we write $\pi_{X\mid Y}(x\mid y)$ for its density.
* The conditional distribution $\mathbb{P}[Y \in \cdot \mid X = x]$ is denoted by $\mu_{Y\mid x}$. If $W = \mathbb{R}^m$ we write $\pi_{Y\mid X}(y\mid x)$ for its density.
* The noise distribution $\mathbb{P}\_E$ on $(W, \mathcal{B}(W))$ is denoted by $\mu_E$. If $W = \mathbb{R}^m$ we write $\pi_E(e)$ for its density.

Similarly, we will denote the joint density of $X$ and $Y$ by $\pi_{X,Y}(x, y)$, and the joint density of $X$ and $E$ by $\pi_{X,E}(x, e)$. Note that the standing assumption of $X$ and $E$ being independent implies $\pi_{X,E}(x, e) = \pi_X(x)\pi_E(e)$.

The density $\pi_{Y\mid X}(y\mid x)$ is called the **likelihood**. For a fixed $x \in V$, it describes the probability distribution of the observed quantity $Y$, and thus expresses the likelihood of different measurement outcomes for fixed parameter $x$.

The following is an example of an inverse problem with an infinite dimensional "latent" field in the background, but where both $V = \mathbb{R}$ and $W = \mathbb{R}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.2.1</span><span class="math-callout__name">(Logistic Differential Equation)</span></p>

We consider a logistic differential equation modeling the growth of a population $N(t)$ over time $t \ge 0$, with growth rate $r > 0$ and carrying capacity $k$:

$$\frac{dN}{dt}(t) = rN(t)(k - N(t)), \qquad N(0) = N_0.$$

The solution is given by

$$N(t) = \frac{k}{1 + \exp(-rkt)(\frac{k}{N_0} - 1)}.$$

Assume we are given the values $r = 0.25$ and $N_0 = 2$, and wish to infer $k$. Suppose that we a priorily know $k \in [10, 20]$, motivating the prior $K \sim \mathrm{uniform}(10, 20)$ for $k$. At time $t = 0.5$ we observe $N(t)$, however the observation is polluted by a noise term $E \sim \mathcal{N}(0, 0.5)$ (independent of $K$). The forward operator is

$$\Phi(k) = \frac{k}{1 + \exp(-0.125k)(\frac{k}{2} - 1)}, \qquad k \in [10, 20],$$

the observation is described by $Y = \Phi(K) + E$, with $\mu\_{K,E} \sim \mathrm{uniform}(10, 20) \otimes \mathcal{N}(0, 0.5)$. The joint and posterior density for $Y = 8$ are depicted in Figure 4.1.

</div>

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_figure4.1.png' | relative_url }}" alt="Left: heatmap of the joint density of (k, y) over k in [10, 20] and y in [0, 20], concentrated along a narrow, gently rising band. Right: the posterior density of k given y = 8, a single sharp peak around k = 15.5." loading="lazy">
</figure>

*Figure 4.1: Joint density $\pi_{K,Y}(k,y)$ (left) and posterior density $\pi_{K\mid Y}(k\mid 8)$ (right) in Example 4.2.1.*

#### 4.2.1 Bayes' Theorem in Finite Dimensions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.2.2</span><span class="math-callout__name">(Bayes' Theorem I)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space and $X : \Omega \to \mathbb{R}^n$, $Y : \Omega \to \mathbb{R}^m$ two RVs with joint density $\pi\_{X,Y}$ and marginal densities $\pi\_X(x) = \int\_{\mathbb{R}^m} \pi\_{X,Y}(x, y) \, \mathrm{d}y$ and $\pi\_Y(y) = \int\_{\mathbb{R}^n} \pi\_{X,Y}(x, y) \, \mathrm{d}x$. Let $\pi\_{Y\mid X}(y\mid x)$ be a conditional density of $Y$ given $X$. Then $\mu\_Y$-a.e.

$$\pi_{X|Y}(x|y) = \frac{\pi_{Y|X}(y|x) \pi_X(x)}{\pi_Y(y)}. \tag{4.2.1}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.2.2</summary>

By Prop. 3.3.33 and Rmk. 3.3.34 there exists a $\mathbb{P}\_X$-null set $N_X \subseteq \mathbb{R}^n$ such that for all $x \in N_X^c$

$$\pi_{Y|X}(y|x) = \frac{\pi_{X,Y}(x, y)}{\pi_X(x)} \qquad \text{for } \lambda_m\text{-a.e. } y \in \mathbb{R}^m, \tag{4.2.2}$$

with the denominator being a positive number. With the $\mathbb{P}\_Y$-null set $N_Y := \lbrace y : \pi_Y(y) \in \lbrace 0, \infty \rbrace \rbrace$, set for $y \in N_Y^c$ and $B \in \mathcal{B}(\mathbb{R}^n)$ with $\pi_{X\mid Y}(x\mid y)$ as in (4.2.1)

$$\tau(y, B) := \int_B \pi_{X|Y}(x|y) \, \mathrm{d}x.$$

Then for any $A \in \mathcal{B}(\mathbb{R}^m)$ and any $B \in \mathcal{B}(\mathbb{R}^n)$, using Fubini's theorem, (4.2.1) as a definition of $\pi_{X\mid Y}$, and (4.2.2),

$$
\begin{aligned}
\int_A \tau(y, B) \, \mathrm{d}\mathbb{P}_Y(y) 
&= \int_{A \setminus N_Y} \int_B \pi_{X|Y}(x|y) \pi_Y(y) \, \mathrm{d}x \, \mathrm{d}y \\
&= \int_{A \setminus N_Y} \int_B \pi_{Y|X}(y|x) \pi_X(x) \, \mathrm{d}x \, \mathrm{d}y \\
&= \int_{A \times B} \pi_{X,Y}(x, y) \, \mathrm{d}x \, \mathrm{d}y \\
&= \mathbb{P}[Y \in A, X \in B].
\end{aligned}
$$

Here we have used that $\mathbb{P}\_X[N_X] = \mathbb{P}\_Y[N_Y] = 0$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.2.3</span></p>

Bayes' theorem is often referred to in the form

$$\text{posterior} \propto \text{likelihood} \cdot \text{prior} \tag{4.2.3}$$

where $\propto$ signifies equality of two functions up to a constant (independent of the function argument). In our notation the posterior is $\pi\_{X\mid Y}(x\mid y)$, the likelihood $\pi\_{Y\mid X}(y\mid x)$ and the prior $\pi\_X(x)$. Equality holds up the multiplicative factor $\pi\_Y(y)^{-1}$, which does not depend on $x$ — the argument of the conditional density $x \mapsto \pi\_{X\mid Y}(x\mid y)$. Hence the posterior is proportional to the prior multiplied with the likelihood. The likelihood represents the information obtained through the data and can be interpreted as updating our prior belief ($\pi\_X$) on the parameter.

</div>

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_bayes_update_1d.png' | relative_url }}" alt="Three density curves over x: a broad blue prior centered at 0, a dashed amber likelihood centered at the data y = 2, and a green posterior lying between them, closer to the likelihood and narrower than both." loading="lazy">
</figure>

*The mechanics of (4.2.3) in one dimension: the posterior (green) is the prior (blue) reweighted by the likelihood (amber, scaled for display; as a function of $x$ it need not integrate to one). The posterior concentrates between prior mean and data and is narrower than either factor — the data has updated and sharpened our prior belief.*

#### 4.2.2 Point Estimates

Even though the posterior contains all available information about $X$, it is still desirable to have a point estimate, i.e. a concrete value $x \in V$ which can be interpreted as the "most probable" value of $X$ (in a suitable sense) given that we observed some value $y$ for $Y$. Part of the reason is that the posterior distribution is a measure on $V$ — a possibly high- or even infinite-dimensional Banach space. This precludes visualization of the posterior density and its properties.

Let us assume for the moment that all densities exist. Then we consider the following three point estimates:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Point Estimators)</span></p>

1. **Maximum likelihood (ML):** The maximum likelihood estimate is a popular estimate in statistics. It is defined as a point

   $$x_{\mathrm{ML}} \in \operatorname{argmax}_x \pi_{Y|X}(y|x). \tag{4.2.4}$$

   Here, the likelihood $\pi\_{Y\mid X}(y\mid x)$ is interpreted as a function of $x$, not a density for $Y$. Thus, a value $x$ that maximizes $\pi(y\mid x)$ can be interpreted as "best explaining" the observed data $y$.

2. **Maximum a posteriori (MAP):** A MAP point is a point maximizing the posterior density

   $$x_{\mathrm{MAP}} \in \operatorname{argmax}_x \pi_{X|Y}(x|y). \tag{4.2.5}$$

3. **Conditional mean (CM):** The conditional mean is the posterior expectation of $X$, i.e.

   $$x_{\mathrm{CM}} := \mathbb{E}[X|Y = y] = \int_V x \, \mathrm{d}\mu_{X|y}(x) = \int_V x \pi_{X|Y}(x|y) \, \mathrm{d}x. \tag{4.2.6}$$

</div>

We point out that computing $x_{\mathrm{ML}}$ and $x_{\mathrm{MAP}}$ requires solving an **optimization** problem, while the computation of $x_{\mathrm{CM}}$ requires computing a (high-dimensional) **integral**. For this reason the computational techniques can differ significantly. However, modern Bayesian techniques are often rooted in a combination of optimization, sampling and integration methods.

Moreover, while $x_{\mathrm{ML}}$ and $x_{\mathrm{MAP}}$ need not be unique, $x_{\mathrm{CM}}$ is (in case the expectation exists). The advantage of $x_{\mathrm{CM}}$ is that it is not strongly affected by small changes in the posterior measure, and this will be discussed in more detail in the following sections. Such a statement is not true for $x_{\mathrm{MAP}}$. On the other hand, $x_{\mathrm{CM}}$ has the disadvantage that it does not necessarily correspond to a point with high posterior density. Such a case is often accompanied by the variance of the posterior being high, indicating that we should not be too confident in our point estimate either way. Figure 4.2 visualizes these statements.

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_figure4.2.png' | relative_url }}" alt="Two bimodal posterior densities. Left: the right mode is taller, so the MAP point lies at the right mode while the CM falls between the two modes in a region of low posterior density. Right: the left mode is taller, so the MAP lies at the left mode and the CM again falls between the modes." loading="lazy">
</figure>

*Figure 4.2: MAP and CM for two posterior distributions.*

Thus, apart from computing point estimates we should additionally always compute some other quantities to investigate uncertainty, such as the variance, and the Bayesian approach allows this! A large variance of one of the parameters w.r.t. the posterior may for instance indicate that the data is not very informative about that parameter. In general, we can define **other statistics** about the parameter $X$ via a function $\varphi \in L^1(V, \mathbb{P}_X; \widetilde{W})$ (as in Thm. 3.2.10). Then we may compute the posterior expectation of $\varphi(X)$, i.e.

$$\mathbb{E}[\varphi(X)|Y = y] = \int_V \varphi(x) \, \mathrm{d}\mu_{X|y}(x) = \int_V \varphi(x) \pi_{X|Y}(x|y) \, \mathrm{d}x \in \widetilde{W}. \tag{4.2.7}$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.2.4</span><span class="math-callout__name">(Posterior Variance)</span></p>

A simple example with $V = \mathbb{R}^n$ and $\widetilde{W} = \mathbb{R}$ would be $\varphi(x) = x\_i^2$, i.e., the second moment of $X\_i$ conditioned on $[Y = y]$. From this we can then estimate the posterior variance of parameter $X\_i$ via

$$\mathbb{V}[X_i|Y = y] := \mathbb{E}[X_i^2|Y = y] - x_{\mathrm{CM},i}^2.$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_credible_interval.png' | relative_url }}" alt="Two posterior densities with MAP (solid blue vertical line), CM (dashed red vertical line) and a shaded 90 percent credible interval. Left: a sharp unimodal posterior where MAP and CM nearly coincide and the interval is short. Right: a wide bimodal posterior where MAP sits at the higher mode, CM lies between the modes, and the credible interval is much wider." loading="lazy">
</figure>

*Point estimates alone can be deceptive: for the sharp posterior (left) $x\_{\mathrm{MAP}} \approx x\_{\mathrm{CM}}$ and the 90% credible interval (shaded) is short, so either estimate is trustworthy. For the broad posterior (right) the two estimates disagree and the posterior variance $\mathbb{V}[X \mid Y = y]$ from Example 4.2.4 is an order of magnitude larger — reporting it reveals how little the data has determined the parameter.*

#### 4.2.3 Additive Noise Model

For the additive noise model (4.1.1) we can deduce explicit expressions of the posterior. To this end, we first compute the likelihood and consider immediately the general case of (possibly infinite dimensional) separable Banach spaces $V$ and $W$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.2.5</span><span class="math-callout__name">(Shift Operator)</span></p>

Let $\Phi : V \to W$ be the forward operator in (4.1.1). We introduce the **shift operator** $S^{\Phi(x)} : W \to W$ via

$$S^{\Phi(x)}(e) := e + \Phi(x).$$

This function is measurable, and hence for the probability measure $\mu\_E$ on $(W, \mathcal{B}(W))$, the pushforward $S\_\sharp^{\Phi(x)} \mu\_E$ also is a probability measure on $(W, \mathcal{B}(W))$.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.2.6</span><span class="math-callout__name">(Likelihood)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space, $V$, $W$ two separable Banach spaces, $\Phi : V \to W$ measurable, and $X : \Omega \to V$ as well as $E : \Omega \to W$ two independent RVs. Assume that $x \mapsto S_\sharp^{\Phi(x)} \mu_E(A)$ is measurable for every $A \in \mathcal{B}(W)$.

Then with the RV $Y := \Phi(X) + E : \Omega \to W$ it holds $\mu_X$-a.e.

$$\mu_{Y|x} = S_\sharp^{\Phi(x)} \mu_E.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 4.2.6</summary>

Define for $A \in \mathcal{B}(W)$ and $x \in V$

$$\tau(x, A) := (S_\sharp^{\Phi(x)} \mu_E)(A).$$

Since $\mu_E$ and consequently $S_\sharp^{\Phi(x)} \mu_E$ are probability measures, $A \mapsto \tau(x, A) = S_\sharp^{\Phi(x)} \mu_E(A)$ defines a probability measure on $W$ for every $x$. Measurability of $x \mapsto \tau(x, A)$ holds by assumption.

Next, let $B \in \mathcal{B}(V)$. Then

$$\mathbb{P}[Y \in A, X \in B] = \int_\Omega \mathbb{1}_A(Y(\omega)) \mathbb{1}_B(X(\omega)) \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \underbrace{\mathbb{1}_A(\Phi(X(\omega)) + E(\omega)) \mathbb{1}_B(X(\omega))}_{=:\,\varphi(X, E)} \, \mathrm{d}\mathbb{P}(\omega),$$

where we have defined the function $\varphi : V \times W \to \mathbb{R}$ via $\varphi(x, e) := \mathbb{1}\_A(\Phi(x) + e) \mathbb{1}\_B(x)$. Then $\varphi$ is measurable as a composition of measurable functions. The independence of the RVs $X$ and $E$ implies that $(X, E)_\sharp \mathbb{P} = \mathbb{P}\_{X, E} = \mu_X \otimes \mu_E$, and thus it follows from Thm. 3.1.9 that

$$
\begin{aligned}
\mathbb{P}[Y \in A, X \in B] 
&= \int_\Omega \varphi(X(\omega), E(\omega)) \, \mathrm{d}\mathbb{P}(\omega) \\
&= \int_{V \times W} \mathbb{1}_A(\Phi(x) + e) \mathbb{1}_B(x) \, \mathrm{d}(\mathbb{P}_X \otimes \mathbb{P}_E)(x, e) \\
&= \int_V \mathbb{1}_B(x) \int_W \mathbb{1}_A\bigl(S^{\Phi(x)}(e)\bigr) \, \mathrm{d}\mu_E(e) \, \mathrm{d}\mu_X(x).
\end{aligned}
$$

By a change of variables (again Thm. 3.1.9)

$$\mathbb{P}[Y \in A, X \in B] = \int_V \mathbb{1}_B(x) \int_W \mathbb{1}_A(e) \, \mathrm{d}S_\sharp^{\Phi(x)} \mu_E(e) \, \mathrm{d}\mu_X(x) = \int_V \mathbb{1}_B(x) \tau(x, A) \, \mathrm{d}\mathbb{P}_X(x).$$

Thus $\tau(x, A)$ is a regular conditional distribution of $Y$ given $X$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.2.7</span><span class="math-callout__name">(Posterior for Additive Noise, Finite Dimensional)</span></p>

Let $Y = \Phi(X) + E$ where $X : \Omega \to \mathbb{R}^n$, $E : \Omega \to \mathbb{R}^m$ are independent RVs with densities $\pi\_X$ and $\pi\_E$ and $\Phi : \mathbb{R}^n \to \mathbb{R}^m$ is measurable. Then $\pi\_{Y\mid X}(y\mid x) = \pi\_E(y - \Phi(x))$ $\mu\_X$-a.e. and

$$\pi_{X|Y}(x|y) = \frac{\pi_E(y - \Phi(x)) \, \pi_X(x)}{Z(y)} \qquad \mu_Y\text{-a.e.}, \tag{4.2.8}$$

where

$$Z(y) = \int_{\mathbb{R}^n} \pi_E(y - \Phi(x)) \, \pi_X(x) \, \mathrm{d}x. \tag{4.2.9}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Corollary 4.2.7</summary>

By definition of the shift operator $S^{\Phi(x)}(y) = y + \Phi(x)$, we have for any $A \in \mathcal{B}(\mathbb{R}^m)$

$$
\begin{aligned}
S_\sharp^{\Phi(x)} \mu_E(A) 
&= \mu_E\bigl(\lbrace y \in \mathbb{R}^m : y + \Phi(x) \in A \rbrace\bigr) \\
&= \int_{\mathbb{R}^m} \mathbb{1}_A(y + \Phi(x)) \pi_E(y) \, \mathrm{d}y \\
&= \int_{\mathbb{R}^m} \mathbb{1}_A(y) \pi_E(y - \Phi(x)) \, \mathrm{d}y,
\end{aligned}
$$

and thus $S_\sharp^{\Phi(x)} \mu_E$ has density $y \mapsto \pi_E(y - \Phi(x))$. Hence by Lemma 4.2.6 the conditional density $\pi_{Y\mid X}(y\mid x)$ is equal to $\pi_E(y - \Phi(x))$ for $\mu_X$-a.e. $x \in V$.

The second statement then follows by Thm. 4.2.2 (Bayes' Theorem I) and the observation that by definition of the conditional density, for every $A \in \mathcal{B}(\mathbb{R}^m)$

$$\mathbb{P}[Y \in A, X \in \mathbb{R}^n] = \int_{\mathbb{R}^n} \int_A \pi_{Y|X}(y|x) \, \mathrm{d}y \, \pi_X(x) \, \mathrm{d}x = \int_A \int_{\mathbb{R}^n} \pi_E(y - \Phi(x)) \pi_X(x) \, \mathrm{d}x \, \mathrm{d}y,$$

so that — except on the $\mu_Y$-null set $\lbrace y \in \mathbb{R}^m : Z(y) = \pi_Y(y) = 0 \rbrace$ — $Z(y)$ is equal to the marginal density $\pi_Y(y)$ of $Y$.

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.2.8</span><span class="math-callout__name">(Sigma-Weighted Norm)</span></p>

For $\Sigma \in \mathbb{R}^{m \times m}$ symmetric positive definite (SPD) we define $\lVert x \rVert\_\Sigma^2 := x^\top \Sigma^{-1} x$. When $\Sigma = I\_m$, the $m \times m$ identity matrix, we write as usual $\lVert x \rVert := \lVert x \rVert\_{I\_m}$ (the Euclidean norm).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.2.9</span><span class="math-callout__name">(Additive Gaussian Noise I)</span></p>

Let $E \sim \mathcal{N}(0, \Sigma)$ with SPD $\Sigma \in \mathbb{R}^{m \times m}$ (which is the most common setting for Bayesian inference problems), then the posterior in Cor. 4.2.7 reads

$$\pi_{X|Y}(x|y) \propto \exp\left( -\frac{1}{2} \lVert y - \Phi(x) \rVert_\Sigma^2 \right) \pi_X(x). \tag{4.2.10}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.2.10</span><span class="math-callout__name">(Linear Gaussian Inverse Problem)</span></p>

Let $A \in \mathbb{R}^{m \times n}$ and suppose that for a given $y \in \mathbb{R}^m$ we wish to find $x \in \mathbb{R}^n$ such $Ax = y$. Assume that $y$ is a realization of $Y = AX + E$ with $E \sim \mathcal{N}(0, I\_m)$. As a prior we *choose* $\mu\_X \sim \mathcal{N}(0, \frac{1}{\alpha} I\_n)$ for some fixed $\alpha > 0$, i.e. $\pi\_X(x) = \frac{\alpha^{n/2}}{(2\pi)^{n/2}} \exp\bigl( -\frac{\alpha \lVert x \rVert^2}{2} \bigr)$. Then

1. **ML:** We have 
   
   $$\pi_{Y\mid X}(y\mid x) = \frac{1}{\sqrt{(2\pi)^m}} \exp\left( -\frac{\lVert Ax - y \rVert^2}{2} \right)$$
   
   Maximizing the likelihood is thus equivalent to finding $x$ in 
   
   $$\operatorname{argmin}_x \lVert Ax - y \rVert$$

2. **MAP:** By Example 4.2.9, $\pi\_{X\mid Y}(x\mid y)$ is up to a $y$-dependent constant equal to 
   
   $$\exp\left( -\frac{\lVert Ax - y \rVert^2 + \alpha \lVert x \rVert^2}{2} \right)$$
   
   Therefore a MAP point is a point in 
   
   $$\operatorname{argmin}_x(\lVert Ax - y \rVert^2 + \alpha \lVert x \rVert^2)$$

3. **CM:** The conditional mean is given by

   $$\int_{\mathbb{R}^n} x \pi_{X|Y}(x|y) \, \mathrm{d}x = \frac{\int_{\mathbb{R}^n} x \exp\left( -\frac{\lVert Ax - y \rVert^2 + \alpha \lVert x \rVert^2}{2} \right) \mathrm{d}x}{\int_{\mathbb{R}^n} \exp\left( -\frac{\lVert Ax - y \rVert^2 + \alpha \lVert x \rVert^2}{2} \right) \mathrm{d}x}.$$

</div>

We make the following observations:

* The ML estimate is *not Bayesian*: The joint distribution can be written as $\pi_{X,Y}(x, y) = \pi_{Y\mid X}(y\mid x)\pi_X(x)$ (cp. Prop. 3.3.33). Hence $\pi_{Y\mid X}(y\mid x)$ and as a consequence $x_{\mathrm{ML}}$ are independent of the (choice of) prior $\pi_X(\cdot)$. In the above example determining $x_{\mathrm{ML}}$ amounts to minimizing $\lVert Ax - y \rVert$ in $x$, i.e. to solving the inverse problem without regularization. We have discussed at length in Chapter 2 why this is a bad idea in the context of ill-posed inverse problems. Therefore the ML estimator is not really interesting here.
* The MAP estimate in Example 4.2.10 corresponds to the Tikhonov-regularized solution in (2.5.1) (with $X = \mathbb{R}^n$ and $Y = \mathbb{R}^m$). Thus, using prior information can be interpreted as adding a form of regularization.

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_additive_noise_2d.png' | relative_url }}" alt="Three contour plots over the (x1, x2) plane. Left: circular blue contours of an isotropic Gaussian prior centered at the origin. Middle: an amber diagonal ridge of the likelihood concentrated around the line x1 + x2 = y. Right: green elliptical posterior contours squeezed along that line, with the MAP point marked by a red dot on the diagonal." loading="lazy">
</figure>

*Example 4.2.10 for $n = 2$, $m = 1$, $A = (1, 1)$: the data only informs the direction $x\_1 + x\_2$, so the likelihood (middle) is constant along the dotted line $\lbrace x : Ax = y \rbrace$ — the unregularised problem has no unique solution. Multiplying by the prior (left) yields a proper posterior (right) whose maximiser is exactly the Tikhonov-regularised solution.*

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_noise_concentration.png' | relative_url }}" alt="A dashed gray standard normal prior and four posterior densities for noise levels sigma between 1.5 and 0.15. As sigma decreases the posteriors move from the prior toward the dotted vertical line at the data y = 1.5 and become increasingly narrow and tall." loading="lazy">
</figure>

*The role of the noise level for $Y = X + E$, $E \sim \mathcal{N}(0, \sigma^2)$: for large $\sigma$ the posterior stays close to the prior (the data is barely informative); as $\sigma \to 0$ the likelihood dominates and the posterior concentrates around the data $y$.*

### 4.3 Bayes' Theorem in Infinite Dimensions

We finish by showing a version of Bayes' theorem in the infinite dimensional setting. Our goal is to obtain a statement with a computable density function. In infinite dimensional spaces, there is no Lebesgue measure. Hence we have to consider a Radon-Nikodym derivative of the posterior w.r.t. another measure — the prior.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.3.1</span></p>

Note that it is very natural to assume that the posterior $\mu\_{X\mid y}$ is absolutely continuous w.r.t. the prior $\mu\_X$. Otherwise the posterior would assign positive measure to an event which has a priori been assigned measure 0.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.3.2</span><span class="math-callout__name">(Bayes' Theorem II)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space, $V$, $W$ two separable Banach spaces, $X : \Omega \to V$, $E : \Omega \to W$ two RVs and $\Phi : V \to W$ a measurable function. Suppose further that for some $\sigma$-finite measure $\nu$ on $(W, \mathcal{B}(W))$

1. $S_\sharp^{\Phi(x)} \mu_E \ll \nu$ for all $x \in V$,
2. $(x, y) \mapsto \frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\nu}(y)$ is measurable for $(x, y) \in V \times W$.

Then with the RV $Y = \Phi(X) + E : \Omega \to W$ it holds $\mu\_{X\mid y} \ll \mu\_X$ for $\mu\_Y$-a.e. $y \in W$ and in this case (in particular excluding the $\mu\_Y$-null set where $Z(y) = 0$)

$$\frac{\mathrm{d}\mu_{X|y}}{\mathrm{d}\mu_X}(x) = \frac{1}{Z(y)} \frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\nu}(y), \tag{4.3.1}$$

where

$$Z(y) = \int_V \frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\nu}(y) \, \mathrm{d}\mu_X(x). \tag{4.3.2}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.3.2</summary>

By definition of a conditional density and due to Lemma 4.2.6 (Likelihood), for every $A \in \mathcal{B}(W)$,

$$
\begin{aligned}
\mathbb{P}[Y \in A, X \in V] 
&= \int_V \mu_{Y|x}(A) \, \mathrm{d}\mu_X(x) \\
&= \int_V (S_\sharp^{\Phi(x)} \mu_E)(A) \, \mathrm{d}\mu_X(x) \\
&= \int_V \int_W \mathbb{1}_A(e) \, \mathrm{d}(S_\sharp^{\Phi(x)} \mu_E)(e) \, \mathrm{d}\mu_X(x) \\
&= \int_V \int_W \mathbb{1}_A(e) \frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\nu}(e) \, \mathrm{d}\nu(e) \, \mathrm{d}\mu_X(x) \\
&= \int_W \mathbb{1}_A(e) \underbrace{\int_V \frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\nu}(e) \, \mathrm{d}\mu_X(x)}_{=\,Z(e)} \, \mathrm{d}\nu(e).
\end{aligned}
$$

Here we used that $\frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\nu}(e)$ is jointly measurable in $(x, e)$, which allowed to use Fubini's theorem. This calculation shows that $Z(y)$ is equal to the Radon-Nikodym derivative $\frac{\mathrm{d}\mu_Y}{\mathrm{d}\nu}(y)$ of $\mu_Y$ w.r.t. $\nu$. In particular $N := \lbrace y \in W : Z(y) = 0 \rbrace$ is a $\mu_Y$-null set since $\mu_Y[N] = \int_{[Z(y) = 0]} Z(y) \, \mathrm{d}\nu(y) = 0$.

Define

$$r(x, y) := \begin{cases} \frac{1}{Z(y)} \frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\nu}(y) & \text{if } y \in N^c \\ 1 & \text{if } y \in N. \end{cases}$$

By (ii) and the fact that $y \mapsto Z_Y(y)$ is measurable as a consequence of the Fubini-Tonelli Theorem, we find that $r$ is measurable. Set for $B \in \mathcal{B}(V)$

$$\tau(y, B) := \int_V \mathbb{1}_B(x) r(x, y) \, \mathrm{d}\mu_X(x).$$

By definition of $r$ the map $B \mapsto \tau(y, B)$ is a probability measure for every $y \in W$ (trivially if $y \in N$, and due to the definition of the normalizing factor $Z(y)$ otherwise). Moreover, $y \mapsto \tau(y, B)$ is measurable as a consequence of Fubini-Tonelli. Since $Z(y) = \frac{\mathrm{d}\mu_Y}{\mathrm{d}\nu}(y)$, for each $B \in \mathcal{B}(V)$, $A \in \mathcal{B}(W)$,

$$
\begin{aligned}
\int_W \mathbb{1}_A(y) \tau(y, B) \, \mathrm{d}\mu_Y(y) 
&= \int_{W \times V} \mathbb{1}_B(x) \mathbb{1}_A(y) r(x, y) \, \mathrm{d}\mu_X(x) \, \mathrm{d}\mu_Y(y) \\
&= \int_V \mathbb{1}_B(x) \int_{W \setminus N} \mathbb{1}_A(y) \frac{1}{Z(y)} \frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\nu}(y) \, \mathrm{d}\mu_Y(y) \, \mathrm{d}\mu_X(x) \\
&= \int_V \mathbb{1}_B(x) \int_{W \setminus N} \mathbb{1}_A(y) \frac{1}{Z(y)} \frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\nu}(y) Z(y) \, \mathrm{d}\nu(y) \, \mathrm{d}\mu_X(x) \\
&= \int_V \mathbb{1}_B(x) \int_{W \setminus N} \mathbb{1}_A(y) \, \mathrm{d}(S_\sharp^{\Phi(x)} \mu_E)(y) \, \mathrm{d}\mu_X(x) \\
&= \int_V \mathbb{1}_B(x) \int_W \mathbb{1}_{A \setminus N}(y) \, \mathrm{d}\mu_{Y|x}(y) \, \mathrm{d}\mu_X(x) \\
&= \mathbb{P}[X \in B, Y \in A \setminus N] \\
&= \mathbb{P}[X \in B, Y \in A],
\end{aligned}
$$

where we have used Lemma 4.2.6 and the fact that $\mathbb{P}[Y \in N] = 0$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.3.3</span><span class="math-callout__name">(Additive Gaussian Noise II)</span></p>

Let $V$ be a separable Banach space and $X : \Omega \to V$ a RV. Let $W = \mathbb{R}^m$, $\Phi : V \to \mathbb{R}^m$ measurable, and assume $E : \Omega \to \mathbb{R}^m$ is a RV independent of $X$ and distributed according to $\mathcal{N}(0, \Sigma)$ for an SPD covariance matrix $\Sigma \in \mathbb{R}^{m \times m}$. Then

$$\frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\lambda_m}(e) = \frac{1}{\sqrt{(2\pi)^m \det(\Sigma)}} \exp\left( -\frac{1}{2} \lVert e - \Phi(x) \rVert_\Sigma^2 \right) = \pi_E(e - \Phi(x)).$$

Measurability of $\Phi$ implies that the last expression is measurable in $(e, x)$. Hence, by Thm. 4.3.2,

$$\frac{\mathrm{d}\mu_{X|y}}{\mathrm{d}\mu_X}(x) \propto \exp\left( -\frac{1}{2} \lVert y - \Phi(x) \rVert_\Sigma^2 \right), \tag{4.3.3}$$

which (up to a constant) corresponds to the likelihood. This is to be expected, since we took the Radon-Nikodym derivative of the posterior w.r.t. the prior. Thus we have established a form of (4.2.3) in this setting. Up to a constant, the negative log-likelihood

$$\frac{1}{2} \lVert y - \Phi(x) \rVert_\Sigma^2$$

is the so-called **data misfit potential**. It tells us how well a parameter $x$ fits the observed value $y$.

</div>

### 4.4 Stability and Well-Posedness

In the previous section we have seen that (under certain assumptions), the posterior distribution exists and is unique ($\mu_Y$-a.e.). Thus the Bayesian inverse problem (BIP) possesses a unique solution (Hadamard 1 & 2). To further pursue our investigation of well-posedness, in this section we discuss continuity of the posterior w.r.t. the data (Hadamard 3).

To this end we restrict ourselves to the case where

1. the number of measurements is finite: $W = \mathbb{R}^m$ and $E : \Omega \to \mathbb{R}^m$,
2. the noise has a Lebesgue density: $\mu_E \ll \lambda_m$.

This covers for example Gaussian noise as discussed in Example 4.3.3. It follows from Cor. 4.2.7 that

$$\frac{\mathrm{d}S_\sharp^{\Phi(x)} \mu_E}{\mathrm{d}\lambda_m}(y) = \pi_E(y - \Phi(x)),$$

which is measurable (as a composition of Borel measurable functions). Therefore Thm. 4.3.2 implies that for $\mu_Y$-a.e. $y$ and for every $A \in \mathcal{B}(V)$ the posterior is given by

$$\mu_{X|y}(A) = \frac{1}{Z(y)} \int_V \mathbb{1}_A(x) \pi_E(y - \Phi(x)) \, \mathrm{d}\mu_X(x) \tag{4.4.1a}$$

with normalization constant

$$Z(y) = \int_V \pi_E(y - \Phi(x)) \, \mathrm{d}\mu_X(x). \tag{4.4.1b}$$

To study the stability of the posterior measure, let us consider the Hellinger distance between measures of the type (4.4.1a). Under the above assumptions we have:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.4.1</span><span class="math-callout__name">(Stability of the Posterior)</span></p>

Let $\sqrt{\pi_E} : \mathbb{R}^m \to [0, L]$ be bounded and Lipschitz continuous with Lipschitz constant $L \ge 1$. Let for $i \in \lbrace 1, 2 \rbrace$

$$\nu_i(A) := \frac{1}{Z_i(y_i)} \int_V \mathbb{1}_A(x) \pi_E(y_i - \Phi_i(x)) \, \mathrm{d}\mu_X(x)$$

with $y_i \in \mathbb{R}^m$, $\Phi_i \in L^2(V, \mu_X; \mathbb{R}^m)$ and $Z_i$ as in (4.4.1b), and where $\mu_X$ is a probability measure on $(V, \mathcal{B}(V))$. Then if $\min\lbrace Z_1, Z_2 \rbrace > 0$

$$D_{\mathrm{H}}(\nu_1, \nu_2) \le \frac{2L^2}{\min\lbrace Z_1, Z_2 \rbrace} \left( \lVert y_1 - y_2 \rVert + \lVert \Phi_1 - \Phi_2 \rVert_{L^2(V, \mu_X; \mathbb{R}^m)} \right).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.4.1</summary>

Set $\varphi_i(x) := y_i - \Phi_i(x)$. Using that the Hellinger distance is computed w.r.t. the common dominating measure $\mu_X$ and inserting an intermediate term, we obtain by the elementary inequality $(a - b)^2 \le 2(a - c)^2 + 2(c - b)^2$

$$
\begin{aligned}
D_{\mathrm{H}}(\nu_1, \nu_2)^2 
&= \frac{1}{2} \int_V \left( \sqrt{\frac{\mathrm{d}\nu_1}{\mathrm{d}\mu_X}(x)} - \sqrt{\frac{\mathrm{d}\nu_2}{\mathrm{d}\mu_X}(x)} \right)^2 \mathrm{d}\mu_X(x) \\
&= \frac{1}{2} \int_V \left( \sqrt{\tfrac{1}{Z_1} \pi_E(\varphi_1(x))} - \sqrt{\tfrac{1}{Z_2} \pi_E(\varphi_2(x))} \right)^2 \mathrm{d}\mu_X(x) \\
&\le \underbrace{\int_V \left( \sqrt{\tfrac{1}{Z_1} \pi_E(\varphi_1(x))} - \sqrt{\tfrac{1}{Z_1} \pi_E(\varphi_2(x))} \right)^2 \mathrm{d}\mu_X(x)}_{=:\,I_1} \\
&\quad + \underbrace{\int_V \left( \sqrt{\tfrac{1}{Z_1} \pi_E(\varphi_2(x))} - \sqrt{\tfrac{1}{Z_2} \pi_E(\varphi_2(x))} \right)^2 \mathrm{d}\mu_X(x)}_{=:\,I_2}.
\end{aligned}
$$

**Bounding $I_1$.** Using the Lipschitz continuity of $\sqrt{\pi_E}$ and $\lvert \varphi_1(x) - \varphi_2(x) \rvert^2 = \lvert (y_1 - y_2) - (\Phi_1(x) - \Phi_2(x)) \rvert^2 \le 2\lVert y_1 - y_2 \rVert^2 + 2(\Phi_1(x) - \Phi_2(x))^2$, and since $\mu_X$ is a probability measure,

$$I_1 \le \frac{L^2}{Z_1} \int_V \lvert \varphi_1(x) - \varphi_2(x) \rvert^2 \, \mathrm{d}\mu_X(x) \le \frac{2L^2}{Z_1} \left( \lVert y_1 - y_2 \rVert^2 + \lVert \Phi_1 - \Phi_2 \rVert_{L^2(V, \mu_X; \mathbb{R}^m)}^2 \right). \tag{4.4.2}$$

**Bounding $I_2$.** Pulling the $x$-independent factor out and using $\int_V \pi_E(\varphi_2(x)) \, \mathrm{d}\mu_X(x) = Z_2$ together with $(\sqrt{Z_2} + \sqrt{Z_1})^2 \ge 4 \min\lbrace Z_1, Z_2 \rbrace$,

$$I_2 = Z_2 \left( \frac{1}{\sqrt{Z_1}} - \frac{1}{\sqrt{Z_2}} \right)^2 = \frac{1}{Z_1} (\sqrt{Z_2} - \sqrt{Z_1})^2 = \frac{(Z_2 - Z_1)^2}{Z_1(\sqrt{Z_2} + \sqrt{Z_1})^2} \le \frac{(Z_1 - Z_2)^2}{4 Z_1 \min\lbrace Z_1, Z_2 \rbrace}. \tag{4.4.3}$$

Note that $\pi_E$ is also Lipschitz continuous with Lipschitz constant $2L^2$, since $\sqrt{\pi_E(a)} \le L$ for all $a$ and thus

$$\lvert \pi_E(a) - \pi_E(b) \rvert = \lvert \sqrt{\pi_E(a)} - \sqrt{\pi_E(b)} \rvert \, \lvert \sqrt{\pi_E(a)} + \sqrt{\pi_E(b)} \rvert \le 2L^2 \lvert a - b \rvert.$$

Thus, as in (4.4.2), using the Cauchy-Schwarz inequality for the probability measure $\mu_X$,

$$
\begin{aligned}
\lvert Z_1 - Z_2 \rvert^2 
&= \left( \int_V \lvert \pi_E(\varphi_1(x)) - \pi_E(\varphi_2(x)) \rvert \, \mathrm{d}\mu_X(x) \right)^2 \\
&\le 4L^4 \int_V \lvert \varphi_1(x) - \varphi_2(x) \rvert^2 \, \mathrm{d}\mu_X(x) \le 8L^4 \left( \lVert y_1 - y_2 \rVert^2 + \lVert \Phi_1 - \Phi_2 \rVert_{L^2}^2 \right).
\end{aligned}
$$

Substituting this into (4.4.3) and combining it with (4.4.2) we obtain

$$D_{\mathrm{H}}(\nu_1, \nu_2)^2 \le I_1 + I_2 \le 2L^2 \left( \frac{1}{Z_1} + \frac{L^2}{Z_1 \min\lbrace Z_1, Z_2 \rbrace} \right) \left( \lVert y_1 - y_2 \rVert^2 + \lVert \Phi_1 - \Phi_2 \rVert_{L^2}^2 \right).$$

To conclude we use that $Z_i = \int_V \pi_E(y_i - \Phi_i(x)) \, \mathrm{d}\mu_X(x) \le L^2$ to bound

$$\frac{1}{Z_1} + \frac{L^2}{Z_1 \min\lbrace Z_1, Z_2 \rbrace} = \frac{\min\lbrace Z_1, Z_2 \rbrace + L^2}{Z_1 \min\lbrace Z_1, Z_2 \rbrace} \le \frac{2L^2}{\min\lbrace Z_1, Z_2 \rbrace^2}.$$

Hence $D_{\mathrm{H}}(\nu_1, \nu_2)^2 \le \frac{4L^4}{\min\lbrace Z_1, Z_2 \rbrace^2} (\lVert y_1 - y_2 \rVert^2 + \lVert \Phi_1 - \Phi_2 \rVert_{L^2}^2)$, and the claim follows from $\sqrt{a^2 + b^2} \le \lvert a \rvert + \lvert b \rvert$.

</details>
</div>

This theorem shows that the posterior distribution depends continuously on the data $y$ and the forward operator $\Phi$ in the Hellinger distance. Together with Lemma 3.5.7, this also implies continuity of the conditional mean $x_{\mathrm{CM}}$ with respect to the data.

Assume that $Z(y) > 0$ for all $y \in \mathbb{R}^m$ in (4.4.1b). Then for fixed forward operator $\Phi = \Phi_1 = \Phi_2$, the posterior density depends continuously on the data $y$; in fact the dependence is locally Lipschitz continuous. As a consequence, also the conditional mean estimate depends continuously on the data, cp. Lemma 3.5.7.

Similarly, for fixed data $y = y_1 = y_2$, the posterior depends continuously on the *forward operator*. Such results are of interest, as in practice, the forward operator needs to be replaced by a numerical approximation $\tilde{\Phi}$ to $\Phi$, e.g. when discretising a PDE.

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_hellinger_stability.png' | relative_url }}" alt="Left: three posterior densities for data values y = 0.8, 1.1 and 1.6 that shift gradually to the right as y grows. Right: the Hellinger distance between the posterior for y = 0.8 and the perturbed posterior, plotted against the size of the data perturbation; the curve grows from zero and stays below a dotted straight line through the origin." loading="lazy">
</figure>

*Well-posedness in action (1D problem with Gaussian prior, nonlinear forward map and Gaussian noise): perturbing the data $y$ deforms the posterior continuously (left), and the Hellinger distance $D\_{\mathrm{H}}(\mu\_{X\mid y}, \mu\_{X\mid y'})$ grows at most linearly in $\lVert y - y' \rVert$ (right), exactly as Theorem 4.4.1 predicts.*

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.4.2</span><span class="math-callout__name">(Additive Gaussian Noise III)</span></p>

Let $E \sim \mathcal{N}(0, \Sigma)$ for an SPD matrix $\Sigma \in \mathbb{R}^{m \times m}$, then

$$\sqrt{\pi_E(y)} = \frac{1}{\sqrt{2\pi \det(\Sigma)}} \exp\left( -\frac{1}{4} \lVert y \rVert_\Sigma^2 \right)$$

is Lipschitz continuous (because the derivative is uniformly bounded for all $y \in \mathbb{R}^m$) and $\sqrt{\pi_E} : \mathbb{R}^m \to [0, (2\pi \det(\Sigma))^{-1/2}]$. Thus Thm. 4.4.1 shows continuous dependence of the posterior on the forward operator and the data in this case.

</div>

To conclude this discussion, we investigate continuity of the posterior w.r.t. the prior.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.4.3</span><span class="math-callout__name">(Stability w.r.t. the Prior)</span></p>

Consider two probability measures $\mu\_X$, $\tilde{\mu}\_X$ on the separable Banach space $V$. Let $y \in \mathbb{R}^m$, $\Phi : V \to \mathbb{R}^m$ be measurable, and let the noise density be bounded, i.e. $\pi\_E : \mathbb{R}^m \to [0, L^2]$ for some $L < \infty$. For $A \in \mathcal{B}(V)$ set

$$\nu(A) := \frac{1}{Z} \int_V \mathbb{1}_A(x) \pi_E(y - \Phi(x)) \, \mathrm{d}\mu_X(x)$$

as well as 

$$Z := \int_V \pi_E(y - \Phi(x)) \, \mathrm{d}\mu_X(x)$$

and define $\tilde{\nu}$, $\tilde{Z}$ analogously, but with $\mu_X$ replaced by $\tilde{\mu}\_X$.

Then if $\min\lbrace Z, \tilde{Z} \rbrace > 0$

$$D_{\mathrm{H}}(\nu, \tilde{\nu}) \le \frac{2L^2}{\min\lbrace Z, \tilde{Z} \rbrace} D_{\mathrm{H}}(\mu_X, \tilde{\mu}_X).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.4.3</summary>

Let $\eta$ be a probability measure on $V$ such that $\mu_X \ll \eta$ and $\tilde{\mu}_X \ll \eta$ (e.g. $\eta = \frac{\mu_X + \tilde{\mu}_X}{2}$). Then, writing $\frac{\mathrm{d}\nu}{\mathrm{d}\eta}(x) = \frac{1}{Z} \pi_E(y - \Phi(x)) \frac{\mathrm{d}\mu_X}{\mathrm{d}\eta}(x)$ and analogously for $\tilde{\nu}$, and using $\pi_E \le L^2$ together with $(a - b)^2 \le 2(a - c)^2 + 2(c - b)^2$,

$$
\begin{aligned}
D_{\mathrm{H}}(\nu, \tilde{\nu})^2 
&= \frac{1}{2} \int_V \left( \sqrt{\frac{\mathrm{d}\nu}{\mathrm{d}\eta}(x)} - \sqrt{\frac{\mathrm{d}\tilde{\nu}}{\mathrm{d}\eta}(x)} \right)^2 \mathrm{d}\eta(x) \\
&= \frac{1}{2} \int_V \pi_E(y - \Phi(x)) \left( \sqrt{\tfrac{1}{Z} \tfrac{\mathrm{d}\mu_X}{\mathrm{d}\eta}(x)} - \sqrt{\tfrac{1}{\tilde{Z}} \tfrac{\mathrm{d}\tilde{\mu}_X}{\mathrm{d}\eta}(x)} \right)^2 \mathrm{d}\eta(x) \\
&\le \int_V L^2 \left( \sqrt{\tfrac{1}{Z} \tfrac{\mathrm{d}\mu_X}{\mathrm{d}\eta}(x)} - \sqrt{\tfrac{1}{Z} \tfrac{\mathrm{d}\tilde{\mu}_X}{\mathrm{d}\eta}(x)} \right)^2 \mathrm{d}\eta(x) \\
&\quad + \int_V \pi_E(y - \Phi(x)) \left( \sqrt{\tfrac{1}{Z} \tfrac{\mathrm{d}\tilde{\mu}_X}{\mathrm{d}\eta}(x)} - \sqrt{\tfrac{1}{\tilde{Z}} \tfrac{\mathrm{d}\tilde{\mu}_X}{\mathrm{d}\eta}(x)} \right)^2 \mathrm{d}\eta(x). \tag{4.4.4}
\end{aligned}
$$

The first term equals $\frac{L^2}{Z} \int_V (\sqrt{\frac{\mathrm{d}\mu_X}{\mathrm{d}\eta}} - \sqrt{\frac{\mathrm{d}\tilde{\mu}_X}{\mathrm{d}\eta}})^2 \, \mathrm{d}\eta = \frac{2L^2}{Z} D_{\mathrm{H}}(\mu_X, \tilde{\mu}\_X)^2$. As in (4.4.3) in the proof of Thm. 4.4.1, the second term can be bounded by

$$\tilde{Z} \left( \sqrt{\tfrac{1}{Z}} - \sqrt{\tfrac{1}{\tilde{Z}}} \right)^2 \le \frac{(Z - \tilde{Z})^2}{4 Z \min\lbrace Z, \tilde{Z} \rbrace},$$

and with Prop. 3.5.6 ($D_{\mathrm{TV}} \le \sqrt{2}\, D_{\mathrm{H}}$) and $\pi_E \le L^2$,

$$
\begin{aligned}
\lvert Z - \tilde{Z} \rvert 
&\le L^2 \int_V \left\lvert \frac{\mathrm{d}\mu_X}{\mathrm{d}\eta} - \frac{\mathrm{d}\tilde{\mu}_X}{\mathrm{d}\eta} \right\rvert \mathrm{d}\eta \\ 
&\le 2L^2 \, D_{\mathrm{TV}}(\mu_X, \tilde{\mu}_X) \\
&\le 2\sqrt{2}\, L^2 \, D_{\mathrm{H}}(\mu_X, \tilde{\mu}_X),
\end{aligned}
$$

so that $(Z - \tilde{Z})^2 \le 8L^4 D_{\mathrm{H}}(\mu_X, \tilde{\mu}\_X)^2$. Hence the second term is bounded by $\frac{2L^4}{Z \min\lbrace Z, \tilde{Z} \rbrace} D_{\mathrm{H}}(\mu_X, \tilde{\mu}\_X)^2$. Substituting both bounds into (4.4.4) and using $Z \le L^2$,

$$D_{\mathrm{H}}(\nu, \tilde{\nu})^2 \le \left( \frac{2L^2}{Z} + \frac{2L^4}{Z \min\lbrace Z, \tilde{Z} \rbrace} \right) D_{\mathrm{H}}(\mu_X, \tilde{\mu}_X)^2 \le \frac{4L^4}{\min\lbrace Z, \tilde{Z} \rbrace^2} D_{\mathrm{H}}(\mu_X, \tilde{\mu}_X)^2,$$

where the last step used $\frac{1}{Z} + \frac{L^2}{Z \min\lbrace Z, \tilde{Z} \rbrace} = \frac{\min\lbrace Z, \tilde{Z} \rbrace + L^2}{Z \min\lbrace Z, \tilde{Z} \rbrace} \le \frac{2L^2}{\min\lbrace Z, \tilde{Z} \rbrace^2}$. Taking square roots yields the claim.

</details>
</div>

In all, we have seen that under the stated assumptions (in particular $\sqrt{\pi_E}$ is Lipschitz and bounded, and the normalization constant $Z$ is positive for all $y$) the posterior density defined in (4.4.1) depends w.r.t. the Hellinger distance continuously on

* the forward operator $\Phi \in L^2_{\mu_X}$,
* the data $y \in \mathbb{R}^m$,
* the prior $\mu_X$.

In this sense the inverse problem is well-posed.

### 4.5 Prior Measures

The choice of prior plays an important role in Bayesian inference, and is what distinguishes it from the frequentist approach. In the (finite dim.) Gaussian setting, cf. Example 4.2.10, the prior and the posterior are equivalent measures. In particular, if the prior assigns the value 0 to an event, then the same holds for the posterior. Hence, as a general rule of thumb, apart from excluding physically impossible events, the prior should not be too restrictive.

In this section we discuss a few techniques to construct suitable measures in separable Banach spaces. As a motivation, we first define an important PDE-constrained inverse problem.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.5.1</span><span class="math-callout__name">(PDE-Driven Inverse Problem)</span></p>

Let $D \subseteq \mathbb{R}^d$ be a bounded Lipschitz domain and consider the elliptic PDE

$$-\mathrm{div}(a \nabla u) = f, \qquad u|_{\partial D} = 0. \tag{4.5.1}$$

Here we assume $f \in L^2(D)$ and $a \in L^\infty(D)$ with $\mathrm{essinf}\_{x \in D} a(x) > 0$. As a consequence of the Lax-Milgram Lemma, there exists a unique weak solution $u \in H_0^1(D)$ of (4.5.1). For a bounded linear operator $B : H_0^1(D) \to \mathbb{R}^m$ — the *observation operator* — define the forward operator $\Phi(a) := Bu \in \mathbb{R}^m$; note that the solution $u \in H_0^1(D)$ of (4.5.1) depends on $a$, so that $\Phi(a)$ is well-defined. One can show that $\Phi$ is a continuous function from $\lbrace a \in L^\infty(D) : \mathrm{essinf}\_{x \in D} a(x) > 0 \rbrace \to \mathbb{R}^m$ and thus $\Phi$ is also measurable.

The inverse problem is to find the diffusion coefficient $a \in L^\infty$ from a noisy measurement

$$Y = \Phi(a) + E, \tag{4.5.2}$$

with $E \sim \mathcal{N}(0, \Gamma)$ for an SPD matrix $\Gamma \in \mathbb{R}^{m \times m}$. In order to do so, we proceed as outlined in Sec. 4.1: $a : \Omega \to L^\infty(D)$ is modelled as a RV for a probability space $(\Omega, \mathcal{A}, \mathbb{P})$, and it is distributed according to some prior measure $\mu\_a$ on $L^\infty(D)$, which we need to specify. The posterior can then be determined using Thm. 4.3.2 by conditioning on the data.

</div>

Throughout this section let $D \subseteq \mathbb{R}^d$ be a bounded (closed) Lipschitz domain (the physical domain) and let $(\Omega, \mathcal{A}, \mathbb{P})$ be a probability space.

#### 4.5.1 Mean and Covariance Functions

The map $a : \Omega \times D \to \mathbb{R}$ in Example 4.5.1, which models a random diffusion coefficient in (4.5.1), allows for two interpretations:

* $\omega \mapsto a(\omega, \cdot) \in L^\infty$ is an $L^\infty$-valued RV,
* $(a(\cdot, x))\_{x \in D}$ is a stochastic process, that is, a family of real-valued RVs indexed over $x \in D$.

For now we adopt the second viewpoint, and use the notation $a_x : \Omega \to \mathbb{R}$ instead of $a(\cdot, x)$. Hence $(a_x)\_{x \in D}$ is a collection of RVs.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption 4.5.2</span><span class="math-callout__name">(Mean and Covariance Function)</span></p>

Each $a\_x : \Omega \to \mathbb{R}$ is assumed to have finite first and second moments. For $x, y \in D$, we set

$$m_x := \mathbb{E}[a_x] \in \mathbb{R} \qquad \text{and} \qquad c(x, y) := \mathrm{cov}(a_x, a_y) = \mathbb{E}[(a_x - m_x)(a_y - m_y)] \in \mathbb{R}.$$

We call $m\_x$ the **mean function** and $c$ the **covariance function**.

</div>

The covariance function is symmetric, i.e. $c(x, y) = c(y, x)$, and satisfies for all $n \in \mathbb{N}$

$$\sum_{i,j=1}^{n} s_i s_j c(x_i, x_j) \ge 0 \qquad \forall x_i, x_j \in D, \ \forall s_i, s_j \in \mathbb{R}, \tag{4.5.3}$$

since $\sum_{i,j=1}^{n} s_i s_j c(x_i, x_j) = \mathbb{E}[(\sum_{j=1}^{n} s_j(a_{x_j} - m_{x_j}))^2]$. In other words, the matrix $C = (c(x_i, x_j))\_{i,j=1}^{n}$ is positive semi-definite.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.5.3</span><span class="math-callout__name">(Positive Semi-Definite Function)</span></p>

A function $c : D \times D \to \mathbb{R}$ that satisfies (4.5.3) is called **positive semi-definite**:

$$\sum_{i,j=1}^{n} s_i s_j c(x_i, x_j) \ge 0 \qquad \forall x_i, x_j \in D, \ \forall s_i, s_j \in \mathbb{R}, \tag{4.5.3}$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.5.4</span><span class="math-callout__name">(Centered, Stationary, Isotropic)</span></p>

A stochastic process is called

1. **centered** if $m\_x = 0$ for all $x \in D$;
2. **stationary** if $c$ is translationally invariant, i.e.

   $$c(x + h, y + h) = c(x, y) \qquad \text{for all } x, y \in D \text{ and all } h \in \mathbb{R}^d \text{ s.t. } x + h, y + h \in D;$$

3. **isotropic** if $c$ is translationally and rotationally invariant, i.e. there is a $\rho : \mathbb{R}^+ \to \mathbb{R}$ s.t.

   $$c(x, y) = \rho(\lVert x - y \rVert) \qquad \text{for all } x, y \in D.$$

</div>

In the following, it will be convenient to work with centered processes, but we emphasize that the following discussion also applies to non-centered processes $(a_x)\_{x \in D}$ by considering $\tilde{a}_x := a_x - m_x$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.5.5</span><span class="math-callout__name">(Matérn Covariance Functions)</span></p>

A popular family of covariance functions for isotropic stochastic processes is the **Matérn** family

$$\rho(r) = \frac{\sigma^2}{2^{\nu - 1} \Gamma(\nu)} \left( \frac{2\sqrt{\nu}\, r}{\ell} \right)^{\nu} K_\nu\left( \frac{2\sqrt{\nu}\, r}{\ell} \right),$$

where $K\_\nu$ is the modified Bessel function of order $\nu$ and $\Gamma$ is the Gamma-function. For $\nu = 1/2$ and $\nu = \infty$, we have the special cases

$$\rho(r) = \sigma^2 \exp(-\sqrt{2} r / \ell) \qquad \text{and} \qquad \rho(r) = \sigma^2 \exp(-r^2 / \ell^2).$$

The three parameters $\sigma^2$, $\ell$ and $\nu$ control the (overall) variance, the correlation length and the smoothness of the stochastic process, respectively. The larger $\nu$, the faster $\rho(r)$ decays for $r \to \infty$ and the smoother the behaviour for $r \to 0$. The family is depicted in Figure 4.3.

</div>

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_figure4.3.png' | relative_url }}" alt="Four Matérn covariance curves rho(r) on r in [0, 3], all starting at 1: for nu = 1/2 the curve has a kink at zero and decays slowly; for nu = 3/2, 5/2 and infinity the curves become flatter at the origin and decay faster in the tail." loading="lazy">
</figure>

*Figure 4.3: Matérn covariance functions $\rho(r)$ with $\sigma^2 = 1$, $\ell = 1$ and $\nu \in \lbrace 1/2, 3/2, 5/2, \infty \rbrace$. The exponential covariance ($\nu = 1/2$) is non-differentiable at $r = 0$, while increasing $\nu$ flattens $\rho$ near the origin — the smoothness of $\rho$ at $0$ is what governs the smoothness of the process.*

Let us begin our discussion by relating continuity of the covariance function to a form of continuity of the stochastic process.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.5.6</span><span class="math-callout__name">(Mean-Square Continuity)</span></p>

We say that $(a_x)\_{x \in D}$ is **mean-square continuous**, if for all $x \in D$ holds $a_x \in L^2(\Omega, \mathbb{P}; \mathbb{R})$ and

$$\lim_{y \to x} \mathbb{E}[(a_x - a_y)^2] = 0.$$

This condition can equivalently be stated as $\lim\_{y \to x} \lVert a\_x - a\_y \rVert\_{L^2(\Omega, \mathbb{P})} = 0$.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.5.7</span><span class="math-callout__name">(Covariance Continuity Equivalence)</span></p>

Assume that $a_x$ has finite second moment and $\mathbb{E}[a_x] = 0$ for all $x \in D$. Then the covariance function $c : D \times D \to \mathbb{R}$ is continuous iff the stochastic process $(a_x)\_{x \in D}$ is mean-square continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 4.5.7</summary>

Since the process is centered, we have

$$\mathbb{E}[(a_x - a_y)^2] = \mathbb{E}[a_x^2] - 2\mathbb{E}[a_x a_y] + \mathbb{E}[a_y^2] = c(x, x) - 2c(x, y) + c(y, y).$$

Hence continuity of $c$ implies mean-square continuity of the stochastic process.

Conversely, let $(a_x)\_{x \in D}$ be mean-square continuous. Then

$$
\begin{aligned}
\lvert c(x + s, y + t) - c(x, y) \rvert 
&= \lvert \mathbb{E}[a_{x+s} a_{y+t}] - \mathbb{E}[a_x a_y] \rvert \\
&= \lvert \mathbb{E}[(a_{x+s} - a_x)(a_{y+t} - a_y)] + \mathbb{E}[(a_{x+s} - a_x) a_y] + \mathbb{E}[a_x (a_{y+t} - a_y)] \rvert \\
&\le \lVert a_{x+s} - a_x \rVert_{L^2(\Omega, \mathbb{P})} \lVert a_{y+t} - a_y \rVert_{L^2(\Omega, \mathbb{P})} + \lVert a_{x+s} - a_x \rVert_{L^2(\Omega, \mathbb{P})} \lVert a_y \rVert_{L^2(\Omega, \mathbb{P})} \\
&\quad + \lVert a_x \rVert_{L^2(\Omega, \mathbb{P})} \lVert a_{y+t} - a_y \rVert_{L^2(\Omega, \mathbb{P})}.
\end{aligned}
$$

This term tends to $0$ as $s, t \to 0$ due to the mean-square continuity of $(a_x)\_{x \in D}$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.5.8</span><span class="math-callout__name">(Smoothness of Paths)</span></p>

More generally, the smoothness of the covariance function can be related to the smoothness of *paths* of the random process; a path is a function $a(\omega, \cdot) : D \to \mathbb{R}$ for fixed $\omega \in \Omega$. This will be further discussed in the exercises. Figure 4.4 illustrates the effect for the Matérn family: the larger $\nu$, the smoother the realizations.

</div>

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_figure4.4.png' | relative_url }}" alt="Four panels showing two sample paths each, driven by the same random input, for Matérn smoothness nu = 1/2, 3/2, 5/2 and infinity. The nu = 1/2 paths are jagged and rough; with growing nu the paths become progressively smoother until they are infinitely smooth curves." loading="lazy">
</figure>

*Figure 4.4: Realizations of a centered stochastic process with Matérn covariance ($\sigma^2 = 1$, $\ell = 0.4$) for increasing smoothness parameter $\nu$. All panels use the same underlying random input, so only the effect of $\nu$ is visible: paths go from nowhere-differentiable ($\nu = 1/2$) to analytic ($\nu = \infty$).*

#### 4.5.2 Karhunen-Loève Expansion

For a function $c \in L^2(D \times D)$ in the following we consider the *Hilbert-Schmidt integral operator* defined as

$$T_c f(x) = \int_D c(x, y) f(y) \, \mathrm{d}y \qquad x \in D.$$

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.5.9</span><span class="math-callout__name">(Hilbert-Schmidt Integral Operator)</span></p>

Let $c \in L^2(D \times D, \mathbb{R})$ be symmetric. Then, the associated **Hilbert-Schmidt integral operator** $T_c : L^2(D) \to L^2(D)$, defined for all $x\in D$ via

$$T_c f(x) = \int_D c(x, y) f(y) \, \mathrm{d}y$$

is a compact, self-adjoint operator.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 4.5.9</summary>

For every $f \in L^2(D)$, the Cauchy-Schwarz inequality and Fubini's theorem give

$$\int_D T_c f(x)^2 \, \mathrm{d}x = \int_D \left( \int_D c(x, y) f(y) \, \mathrm{d}y \right)^2 \mathrm{d}x \le \int_D \int_D c(x, y)^2 \, \mathrm{d}y \int_D f(y)^2 \, \mathrm{d}y \, \mathrm{d}x,$$

which shows that $\lVert T_c \rVert_{\mathcal{L}(L^2, L^2)} \le \lVert c \rVert_{L^2(D \times D)}$, i.e. $T_c$ is bounded. Due to the symmetry of $c$, $T_c$ is clearly self-adjoint:

$$\langle T_c f, g \rangle_{L^2} = \int_D \int_D c(x, y) f(y) g(x) \, \mathrm{d}y \, \mathrm{d}x = \langle f, T_c g \rangle_{L^2}.$$

We skip the proof of compactness.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.5.10</span></p>

For a symmetric positive semi-definite $c \in L^2(D \times D; \mathbb{R})$ show that $T\_c : L^2(D) \to L^2(D)$ is a positive operator, i.e. $\langle T\_c f, f \rangle \ge 0$ for all $f \in L^2(D)$.

</div>

Since $T_c : L^2(D) \to L^2(D)$ is a compact, self-adjoint, positive operator, Theorem A.2.6 guarantees the existence of an orthonormal system $(\varphi_j)\_{j \in \mathbb{N}}$ of eigenvectors of $T_c$ with corresponding positive eigenvalues $(\ell_j)\_{j \in \mathbb{N}}$ such that $T_c f = \sum_{j \in \mathbb{N}} \ell_j \langle f, \varphi_j \rangle_{L^2(D)} \varphi_j$, the spectral decomposition of $T_c$; or $T_c f = \sum_{j=1}^{n} \ell_j \langle f, \varphi_j \rangle_{L^2(D)} \varphi_j$, in case $T_c$ has finite, $n$-dimensional range. To avoid distinguishing two cases, for simplicity we assume in the following that $c$ is such that the range of $T_c$ is infinite dimensional.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.5.11</span><span class="math-callout__name">(Trace-Class Operator)</span></p>

Let $H$ be a separable Hilbert space and $A \in \mathcal{L}(H, H)$ a bounded positive linear operator. We say that $A$ is a **trace-class operator** (or $A$ is of trace class) if for an ONB $(\varphi_j)\_{j \in \mathbb{N}}$ of $H$ holds

$$\mathrm{tr}(A) := \sum_{j \in \mathbb{N}} \langle A\varphi_j, \varphi_j \rangle_H < \infty.$$

</div>

One can show that the definition trace $\mathrm{tr}(A)$ does not depend on the choice of ONB $(\varphi_j)\_{j \in \mathbb{N}}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5.12</span><span class="math-callout__name">(Mercer's Theorem)</span></p>

Let $c : D \times D \to \mathbb{R}$ be continuous, positive semi-definite and symmetric. Then there exists an orthonormal system $(\varphi_j)\_{j \in \mathbb{N}}$ in $L^2(D)$ such that $\varphi_j \in C^0(D)$, $T_c \varphi_j = \ell_j \varphi_j$ for a sequence of nonnegative numbers $\ell_j$ satisfying $\ell_j \to 0$ as $j \to \infty$, and

$$c(x, y) = \sum_{j \in \mathbb{N}} \ell_j \varphi_j(x) \varphi_j(y)$$

in the sense of absolute and uniform convergence for all $x, y \in D$. Moreover $T_c$ is a trace-class operator and

$$\mathrm{tr}(T_c) = \int_D c(x, x) \, \mathrm{d}x < \infty.$$

</div>

We skip the proof of Mercer's theorem.

For every $\omega$, we can formally expand $a_x(\omega) = a(\omega, x)$ as a function of $x$ in the $L^2(D)$ orthonormal basis $(\varphi_j)\_{j \in \mathbb{N}}$:

$$a_x(\omega) = a(\omega, x) = \sum_{j \in \mathbb{N}} a_j(\omega) \varphi_j(x) \tag{4.5.4}$$

with the coefficients defined as

$$a_j(\omega) = \int_D a(\omega, x) \varphi_j(x) \, \mathrm{d}x$$

being real-valued RVs. Such an expansion is called a Karhunen-Loève expansion. We next show its convergence.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5.13</span><span class="math-callout__name">(Karhunen-Loève Expansion)</span></p>

Let $a : \Omega \times D \to \mathbb{R}$ be a measurable centered mean-square continuous stochastic process with $a \in L^2(\Omega \times D, \mathbb{P} \otimes \lambda_d; \mathbb{R})$. There exists an orthonormal basis $(\varphi_j)\_{j \in \mathbb{N}} \subseteq L^2(D)$ and nonnegative numbers $(\ell_j)\_{j \in \mathbb{N}}$ (we allow for $\ell_j = 0$) such that with $a_j(\omega) = \int_D a(\omega, x) \varphi_j(x) \, \mathrm{d}x$ and $a_x(\omega) = a(\omega, x)$

$$\lim_{n \to \infty} \sup_{x \in D} \mathbb{E}\left[ \left( a_x - \sum_{j=1}^{n} a_j \varphi_j(x) \right)^2 \right] = 0. \tag{4.5.5}$$

The coefficients $a_j$ satisfy for all $j$ with $\ell_j > 0$

1. $\mathbb{E}[a_j] = 0$,
2. $\mathbb{E}[a_i a_j] = \delta_{ij} \ell_i$ and hence $\mathbb{V}[a_j] = \ell_j$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.5.13</summary>

In the following $(\varphi_j)\_{j \in \mathbb{N}}$ is an orthonormal basis of $L^2(D)$ that is obtained by extending an orthonormal sequence of eigenvectors $\varphi_j$ of $T_c$ with eigenvalues $\ell_j > 0$ to an ONB. We extend the sequence of eigenvalues by zeros, i.e. $\ell_j = 0$ for all $j$ for which $\varphi_j$ belongs to the kernel of $T_c$.

**The coefficients are square-integrable.** Joint measurability and Fubini's theorem imply that $a_j : \Omega \to \mathbb{R}$ is measurable and, using $\int_D \varphi_j(x)^2 \, \mathrm{d}x = 1$ and the Cauchy-Schwarz inequality,

$$\int_\Omega \lvert a_j(\omega) \rvert^2 \, \mathrm{d}\mathbb{P}(\omega) = \int_\Omega \left\lvert \int_D a(\omega, x) \varphi_j(x) \, \mathrm{d}x \right\rvert^2 \mathrm{d}\mathbb{P}(\omega) \le \int_\Omega \int_D a(\omega, x)^2 \, \mathrm{d}x \, \mathrm{d}\mathbb{P}(\omega) = \lVert a \rVert_{L^2(\Omega \times D)}^2.$$

Thus $a_j \in L^2(\Omega, \mathbb{P}; \mathbb{R})$ and $a_j : \Omega \to \mathbb{R}$ is a RV with finite first and second moment for each $j$.

**Property (1).** Since $(a_x)\_{x \in D}$ is centered,

$$\mathbb{E}[a_j] = \int_\Omega a_j(\omega) \, \mathrm{d}\mathbb{P}(\omega) = \int_D \varphi_j(x) \int_\Omega a(\omega, x) \, \mathrm{d}\mathbb{P}(\omega) \, \mathrm{d}x = 0.$$

**Property (2).** Since $c(x, y) = \int_\Omega a(\omega, x) a(\omega, y) \, \mathrm{d}\mathbb{P}(\omega)$ and $\varphi_i$ is an eigenvector of $T_c$ with eigenvalue $\ell_i$,

$$
\begin{aligned}
\mathbb{E}[a_i a_j] 
&= \int_\Omega \int_D a(\omega, x) \varphi_i(x) \, \mathrm{d}x \int_D a(\omega, y) \varphi_j(y) \, \mathrm{d}y \, \mathrm{d}\mathbb{P}(\omega) \\
&= \int_D \int_D \varphi_i(x) \varphi_j(y) c(x, y) \, \mathrm{d}x \, \mathrm{d}y = \int_D \ell_i \varphi_i(y) \varphi_j(y) \, \mathrm{d}y = \delta_{ij} \ell_i,
\end{aligned}
$$

where we used that the $(\varphi_j)\_{j \in \mathbb{N}}$ are orthonormal in $L^2(D)$ in the last step.

**Convergence (4.5.5).** For $x \in D$, define $\epsilon_n(x) := \mathbb{E}[(a_x - \sum_{j=1}^n a_j \varphi_j(x))^2]$; we prove $\sup_{x \in D} \epsilon_n(x) \to 0$. First observe that

$$\epsilon_n(x) = \mathbb{E}[a_x^2] - 2\mathbb{E}\left[ a_x \sum_{j=1}^n a_j \varphi_j(x) \right] + \mathbb{E}\left[ \sum_{i,j=1}^n a_i a_j \varphi_i(x) \varphi_j(x) \right]. \tag{4.5.6}$$

Now

$$\mathbb{E}\left[ a_x \sum_{j=1}^n a_j \varphi_j(x) \right] = \sum_{j=1}^n \int_D \varphi_j(x) \varphi_j(y) \underbrace{\int_\Omega a_x(\omega) a_y(\omega) \, \mathrm{d}\mathbb{P}(\omega)}_{=\,c(x, y)} \mathrm{d}y = \sum_{j=1}^n \ell_j \varphi_j(x)^2,$$

and by property (2)

$$\mathbb{E}\left[ \sum_{i,j=1}^n a_i a_j \varphi_i(x) \varphi_j(x) \right] = \sum_{i,j=1}^n \varphi_i(x) \varphi_j(x) \mathbb{E}[a_i a_j] = \sum_{j=1}^n \ell_j \varphi_j(x)^2.$$

Since $\mathbb{E}[a_x^2] = c(x, x)$ by (4.5.6) we thus find

$$\epsilon_n(x) = c(x, x) - \sum_{j=1}^n \ell_j \varphi_j(x)^2.$$

An application of Mercer's theorem (Thm. 4.5.12), which guarantees absolute and uniform convergence of $\sum_j \ell_j \varphi_j(x)^2$ to $c(x, x)$, concludes the proof.

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_kl_expansion.png' | relative_url }}" alt="Three panels. Left: eigenvalues of the covariance operator on a logarithmic scale decaying rapidly over the first 50 indices. Middle: the four leading eigenfunctions, oscillating sine-like curves with increasing frequency. Right: one random realization reconstructed from 2, 8, 32 and 500 KL terms; the low truncations are smooth approximations that progressively capture the rough reference path." loading="lazy">
</figure>

*Anatomy of the KL expansion for the exponential covariance $c(x, y) = e^{-\lvert x - y \rvert / 0.3}$ on $D = (0, 1)$: the eigenvalues $\ell\_j$ of $T\_c$ decay rapidly (left), the eigenfunctions $\varphi\_j$ oscillate with increasing frequency (middle), and a single realization $\sum\_{j \le s} a\_j \varphi\_j$ converges to the full field as the truncation level $s$ grows (right). The decay of $\ell\_j$ is what makes truncation — and hence computation — feasible.*

Thus, the Karhunen-Loève expansion provides a method to construct measures on function spaces such as $L^2(D)$, respectively to sample from a $L^2(D)$-valued RV: We adopt now the viewpoint that $\omega \mapsto a(\omega, \cdot)$ is an $L^2(D)$-valued RV. A sample from this RV can be drawn by first sampling from the real-valued RVs $(a_j)\_{j \in \mathbb{N}}$, and then computing $\sum_{j \in \mathbb{N}} a_j(\omega) \varphi_j(x)$. In practice, the sum is truncated after $s$ terms and

$$a_{s,x}(\omega) = \sum_{j=1}^{s} a_j(\omega) \varphi_j(x)$$

is obtained as an approximation.

A common situation in practice is that the mean $\mathbb{E}[a]$ and the covariance $\mathrm{cov}(a)$ are known or prescribed and one wishes to obtain a random field with the given expectation and covariance. Note that such a random field is not unique, since the expectation and covariance do not uniquely determine a RV (except in the case of a Gaussian RF).

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.5.14</span><span class="math-callout__name">(Covariance Operator via Hilbert-Schmidt Operator)</span></p>

Let $a : \Omega \to L^2(D)$ be a RV with finite second moment and such that $\mathbb{E}[a] = m \in L^2(D)$. Then $\mathrm{cov}(a) : L^2(D) \to L^2(D)$ is given by $T_c$ where $c(x, y) = \mathbb{E}[(a(\cdot, x) - m(x))(a(\cdot, y) - m(y))]$ and $c \in L^2(D \times D)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 4.5.14</summary>

By definition of the covariance we have for all $f, g \in L^2(D)$ with $C := \mathrm{cov}(a)$, using Fubini's theorem,

$$
\begin{aligned}
\langle g, Cf \rangle_{L^2(D)} 
&= \int_\Omega \langle a(\omega, \cdot) - m(\cdot), f(\cdot) \rangle_{L^2(D)} \, \langle a(\omega, \cdot) - m(\cdot), g(\cdot) \rangle_{L^2(D)} \, \mathrm{d}\mathbb{P}(\omega) \\
&= \int_D \int_D \mathbb{E}[(a(\cdot, x) - m(x))(a(\cdot, y) - m(y))] f(x) \, \mathrm{d}x \, g(y) \, \mathrm{d}y = \langle g, T_c f \rangle_{L^2(D)}.
\end{aligned}
$$

Furthermore, since $\int_\Omega \int_D (a(\omega, x) - m(x))^2 \, \mathrm{d}x \, \mathrm{d}\mathbb{P}(\omega) = \mathbb{E}[\lVert a - m \rVert_{L^2}^2] < \infty$, the Cauchy-Schwarz inequality gives

$$
\begin{aligned}
\int_D \int_D c(x, y)^2 \, \mathrm{d}x \, \mathrm{d}y 
&= \int_D \int_D \mathbb{E}[(a(\cdot, x) - m(x))(a(\cdot, y) - m(y))]^2 \, \mathrm{d}x \, \mathrm{d}y \\
&\le \int_D \int_\Omega (a(\omega, x) - m(x))^2 \, \mathrm{d}\mathbb{P}(\omega) \, \mathrm{d}x \int_D \int_\Omega (a(\omega, y) - m(y))^2 \, \mathrm{d}\mathbb{P}(\omega) \, \mathrm{d}y < \infty,
\end{aligned}
$$

so that $c \in L^2(D \times D)$.

</details>
</div>

Thus, Theorem 4.5.13 and Lemma 4.5.14 show that a RV can be expanded in terms of the eigenfunctions of the covariance operator (under the assumptions of Thm. 4.5.13) to obtain a random field with prescribed mean $\mathbb{E}[a]$ and covariance $\mathrm{cov}(a)$.

#### 4.5.3 Uniform Measures

We now want to construct a similar expansion with real-valued RVs for $L^\infty(D)$ random fields, as needed in Example 4.5.1.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.5.15</span></p>

We note without proof that the pointwise limit $X : \Omega \to V$ of a sequence of RVs $X\_j : (\Omega, \mathcal{A}) \to (V, \mathcal{B}(V))$, $j \in \mathbb{N}$, on a separable Banach space $V$ is measurable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.5.16</span><span class="math-callout__name">(Constructing $L^\infty$ Random Fields)</span></p>

Let $(\varphi_j)\_{j \in \mathbb{N}} \subseteq L^\infty(D)$ such that $\lVert \varphi_j \rVert_{L^\infty(D)} = 1$ for all $j$, $m \in L^\infty(D)$ and $(\ell_j)\_{j \in \mathbb{N}} \in \ell^1(\mathbb{N})$. Furthermore, let $\xi := (\xi_j)\_{j \in \mathbb{N}}$ be a sequence of iid RVs $\xi_j : \Omega \to [-1, 1]$ such that $\xi_j \sim \mathrm{uniform}(-1, 1)$.

Then

$$a(\omega, x) := m(x) + \sum_{j \in \mathbb{N}} \ell_j \xi_j(\omega) \varphi_j(x) \tag{4.5.7}$$

defines a RV $\omega \mapsto a(\omega, \cdot) \in L^\infty(D)$ satisfying $\mathbb{E}[a] = m$ and for all $\omega \in \Omega$

$$\lVert a(\omega, \cdot) \rVert_{L^\infty(D)} \le \lVert m \rVert_{L^\infty(D)} + \sum_{j \in \mathbb{N}} \ell_j, \qquad \mathrm{essinf}_{x \in D} a(\omega, x) \ge \mathrm{essinf}_{x \in D} m(x) - \sum_{j \in \mathbb{N}} \ell_j. \tag{4.5.8}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 4.5.16</summary>

Since each $\xi_j : \Omega \to \mathbb{R}$ is measurable, also $a_n := m + \sum_{j=1}^n \ell_j \xi_j \varphi_j : \Omega \to L^\infty(D)$ is measurable as a sum of measurable functions. Moreover, by the triangle inequality and $\lVert \varphi_j \rVert_{L^\infty(D)} = 1$, $\lvert \xi_j \rvert \le 1$,

$$\lVert a_n(\omega, \cdot) \rVert_{L^\infty(D)} \le \lVert m \rVert_{L^\infty(D)} + \sum_{j \in \mathbb{N}} \ell_j < \infty \qquad \text{for every } n,$$

so that $a_n$ converges pointwise to a measurable function $a : \Omega \to L^\infty(D)$ by Remark 4.5.15. This also implies the first bound in (4.5.8), and the second bound $\mathrm{essinf}_{x \in D} a(\omega, x) \ge \mathrm{essinf}_{x \in D} m(x) - \sum_{j \in \mathbb{N}} \ell_j$ follows similarly, since $\lvert \sum_{j} \ell_j \xi_j(\omega) \varphi_j(x) \rvert \le \sum_j \ell_j$ for a.e. $x \in D$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.5.17</span><span class="math-callout__name">(Continuation of Example 4.5.1)</span></p>

Let $\omega \in \Omega$ be fixed. Assume that $a(\omega, x)$ is expanded in a series as in (4.5.7), where $\varphi\_j \in L^\infty(D)$ with $\lVert \varphi\_j \rVert\_{L^\infty(D)} = 1$ for all $j \in \mathbb{N}$, and the sequence $(\ell\_j)\_{j \in \mathbb{N}} \in \ell^1(\mathbb{N})$ and $m \in L^\infty(D)$ are chosen such that

$$a_- := \operatorname{essinf}_{x \in D} m(x) - \sum_{j \in \mathbb{N}} \ell_j > 0. \tag{4.5.9}$$

The weak formulation of (4.5.1) is: Find $u \in H\_0^1(D)$ such that

$$\int_D a \nabla u(x)^\top \nabla v(x) \, \mathrm{d}x = \langle f, v \rangle \qquad \forall v \in H_0^1(D), \tag{4.5.10}$$

where the last bracket is the $L^2$-inner product. Then for every $\omega \in \Omega$

$$\lVert a(\omega, \cdot) \rVert_{L^\infty(D)} \le \lVert m \rVert_{L^\infty(D)} + \sum_{j \in \mathbb{N}} \ell_j < \infty, \tag{4.5.11}$$

which implies boundedness of the bilinear form on the left-hand side of (4.5.10), while equation (4.5.9) implies its coercivity ($\operatorname{essinf}\_{x \in D} a(\omega, x) \ge a\_- > 0$ by Proposition 4.5.16). Therefore, by the Lax-Milgram Lemma, for each $\omega \in \Omega$ there is a unique weak solution $u \in H\_0^1(D)$ to (4.5.10) for the diffusion coefficient $a(\omega, \cdot) \in L^\infty(D)$. Since the solution depends on $a(\omega, \cdot)$ we also write $u(a(\omega), \cdot)$ (or simply $u(\omega, \cdot)$) to emphasize this dependence.

Now to specify the prior measure $\mu\_a$ on $a(\omega, \cdot) \in L^\infty(D)$ for the Bayesian inverse problem (4.5.2), we denote by $\nu := \otimes\_{j \in \mathbb{N}} \frac{\lambda}{2}$ the distribution of $\xi := (\xi\_j)\_{j \in \mathbb{N}}$ on $[-1, 1]^{\mathbb{N}}$, where $\frac{\lambda}{2}$ is the uniform probability measure on $[-1, 1]$ (i.e. $1/2$ times the Lebesgue measure). Denoting by $T : [-1, 1]^{\mathbb{N}} \to L^\infty(D)$ the operator that maps a sequence $\xi \in [-1, 1]^{\mathbb{N}}$ to a function $a \in L^\infty(D)$, i.e.

$$T(\xi) := m + \sum_{j \in \mathbb{N}} \xi_j \ell_j \varphi_j,$$

the **uniform prior measure** on the random diffusion coefficient $a(\omega, \cdot) \in L^\infty(D)$ is $\mu\_a := T\_\sharp \nu$. It is called uniform because of the uniform distribution on the coefficient sequence, but also because the diffusion coefficient $a(\omega, \cdot)$ is uniformly bounded from above and below for all $\omega \in \Omega$. For computational purposes, it is often much more convenient to formulate the inverse problem in terms of the RV $(\xi\_j)\_{j \in \mathbb{N}} : \Omega \to [-1, 1]^{\mathbb{N}}$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_uniform_field_pde.png' | relative_url }}" alt="Left: four samples of a random diffusion coefficient on [0, 1], wiggly curves starting and ending at the mean value 1 and staying well above a dashed lower bound a-minus. Right: the four corresponding solutions of the elliptic boundary value problem, smooth arch-shaped curves vanishing at both endpoints." loading="lazy">
</figure>

*Example 4.5.17 in one dimension: samples of $a(\omega, \cdot) = m + \sum\_j \ell\_j \xi\_j(\omega) \varphi\_j$ with $m \equiv 1$, $\varphi\_j(x) = \sin(j \pi x)$, $\ell\_j \propto j^{-2}$ and iid $\xi\_j \sim \mathrm{uniform}(-1, 1)$ (left). Every sample stays above $a\_- > 0$ (dashed), so by Lax-Milgram each draw yields a unique solution $u(a(\omega), \cdot)$ of $-(a u')' = f$, $u(0) = u(1) = 0$ (right) — the solution map $a \mapsto u$ is well-defined on the support of the uniform prior.*

Theorem 4.5.13 and Lemma 4.5.14 show that a RV can be expanded in terms of the eigenfunctions of the covariance operator. This also works the other way around in the following sense:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.5.18</span><span class="math-callout__name">(Constructing RVs via KL Expansion)</span></p>

Let $(\varphi_j)\_{j \in \mathbb{N}}$ be an orthonormal system in $L^2(D)$. Let $\xi_j : \Omega \to [-1, 1]$ be a sequence of iid RVs with $\xi_j \sim \mathrm{uniform}(-1, 1)$. Then for a sequence $(\ell_j)\_{j \in \mathbb{N}} \in \ell^2(\mathbb{N})$ and $m \in L^2(D)$

$$a(\omega, x) := m(x) + \sum_{j \in \mathbb{N}} \ell_j \xi_j(\omega) \varphi_j(x)$$

defines a RV $\omega \mapsto a(\omega, \cdot) \in L^2(D)$ where $\mathbb{E}[a] = m$ and $\mathrm{Cov}(a) = T_c$ with $c(x, y) = \frac{1}{3} \sum_{j \in \mathbb{N}} \ell_j^2 \varphi_j(x) \varphi_j(y) \in L^2(D \times D)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 4.5.18</summary>

Set $a_s(\omega, x) := m(x) + \sum_{j=1}^s \ell_j \xi_j(\omega) \varphi_j(x)$. For every $\omega \in \Omega$ it holds that $a_s(\omega, \cdot) \to a(\omega, \cdot)$ in $L^2(D)$ as $s \to \infty$. Measurability of each $\xi_j : \Omega \to \mathbb{R}$ implies measurability of $a_s : \Omega \to L^2(D)$, and thus $a : \Omega \to L^2(D)$ is measurable by Remark 4.5.15.

Note that $a_s \to a$ in the topology of $L^2(\Omega, \mathbb{P}; L^2(D))$: by orthonormality of $(\varphi_j)\_{j \in \mathbb{N}}$ and $\xi_j^2 \le 1$,

$$\lim_{s \to \infty} \mathbb{E}\left[ \lVert a - a_s \rVert_{L^2(D)}^2 \right] = \lim_{s \to \infty} \mathbb{E}\left[ \sum_{j > s} \ell_j^2 \xi_j(\omega)^2 \right] \le \lim_{s \to \infty} \sum_{j > s} \ell_j^2 = 0, \tag{4.5.12}$$

since $(\ell_j)\_{j \in \mathbb{N}} \in \ell^2(\mathbb{N})$. The Cauchy-Schwarz inequality implies in particular $a_s \to a$ in $L^1(\Omega, \mathbb{P}; L^2(D))$, so that $\mathbb{E}[a] = \lim_{s \to \infty} \mathbb{E}[a_s] = m$.

Next, with $\mathbb{E}[\xi_i \xi_j] = 0$ for $i \neq j$ (independence) and $\mathbb{E}[\xi_i^2] = \frac{1}{2} \int_{-1}^1 x^2 \, \mathrm{d}x = \frac{1}{3}$,

$$c_s(x, y) := \mathbb{E}[(a_s(\cdot, x) - m(x))(a_s(\cdot, y) - m(y))] = \sum_{i,j=1}^s \ell_i \ell_j \varphi_i(x) \varphi_j(y) \mathbb{E}[\xi_i \xi_j] = \frac{1}{3} \sum_{i=1}^s \ell_i^2 \varphi_i(x) \varphi_i(y).$$

From (4.5.12) it follows that $c_s \to c = \mathbb{E}[(a(\cdot, x) - m(x))(a(\cdot, y) - m(y))]$ in $L^2(D \times D)$ as $s \to \infty$. An application of Lemma 4.5.14 concludes the proof.

</details>
</div>

#### 4.5.4 Gaussian Measures

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.5.19</span><span class="math-callout__name">(Gaussian Probability Measure)</span></p>

Let $V$ be a separable Banach space. A Borel measure $\mu$ on $(V, \mathcal{B}(V))$ (i.e. a locally finite measure) is called a **Gaussian probability measure** iff for every $f \in V'$ the measure $f_\sharp \mu$ is Gaussian; here Dirac measures are considered to be Gaussian with zero variance. The measure is said to be **centered** if $\int_{\mathbb{R}} x \, \mathrm{d}f_\sharp\mu(x) = 0$ for all $f \in V'$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.5.20</span><span class="math-callout__name">(Fernique's Theorem)</span></p>

It can be shown ("Fernique's theorem") that for every Gaussian measure $\mu$ on $(V, \mathcal{B}(V))$ there exists $\alpha > 0$ such that $\int_V \exp(\alpha \lVert x \rVert_V^2) \, \mathrm{d}\mu(x) < \infty$. Thus a RV with Gaussian distribution has finite moments of all orders.

</div>

We concentrate on the case of separable Hilbert spaces $H$. The expectation and covariance of a probability measure $\mu$ are understood as the expectation and covariance of a RV with distribution $\mu$. Recall from Section 3.2: when they exist, the expectation of $\mu$ is $m := \int_H x \, \mathrm{d}\mu(x) \in H$ and the covariance operator $C : H \to H$ of $\mu$ is the operator satisfying

$$\langle x, Cy \rangle_H = \int_H \langle h - m, x \rangle_H \langle h - m, y \rangle_H \, \mathrm{d}\mu(h) \qquad \forall x, y \in H.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5.21</span><span class="math-callout__name">(Gaussian Measures on Hilbert Spaces)</span></p>

Let $H$ be a separable Hilbert space. Every Gaussian measure $\mu$ on $(H, \mathcal{B}(H))$ has a positive covariance operator $C\_\mu : H \to H$ which is of trace-class and satisfies

$$\mathrm{tr}(C_\mu) = \int_H \lVert x \rVert_H^2 \, \mathrm{d}\mu(x) < \infty.$$

Conversely, for every positive trace-class symmetric operator $C : H \to H$ there exists a Gaussian measure $\mu$ on $H$ with covariance operator $C$.

</div>

We state this result without proof.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.5.22</span><span class="math-callout__name">(Gaussian Random Variable)</span></p>

A RV $X : \Omega \to H$ is called Gaussian, if its distribution is a Gaussian measure on $H$ with expectation $m \in H$ and covariance operator $C$. In this case we write $X \sim \mathcal{N}(m, C)$.

</div>

We mention that, as in the finite dimensional case, Gaussian measures are uniquely determined through their expectation and covariance operator.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.5.23</span></p>

Check that if $X \sim \mathcal{N}(m, C)$ and $h \in H$, then the real-valued RV $\langle X, h \rangle\_H$ is $\mathcal{N}\bigl( \langle m, h \rangle\_H, \langle h, Ch \rangle\_H \bigr)$.

</div>

There holds the following Karhunen-Loève expansion for Gaussian measures, which can be proved similarly to Theorem 4.5.13:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5.24</span><span class="math-callout__name">(Karhunen-Loève Expansion of Gaussian RFs)</span></p>

Let $a : \Omega \to L^2(D)$ be a RV with distribution $\mathcal{N}(m, C)$. Let $(\varphi_j)\_{j \in \mathbb{N}} \subseteq L^2(D)$ be an orthonormal system of eigenvectors of $C$ with positive eigenvalues $(\ell_j)\_{j \in \mathbb{N}}$ such that $Cf = \sum_{j \in \mathbb{N}} \ell_j \langle f, \varphi_j \rangle_{L^2(D)} \varphi_j$. Then

$$a(\omega, x) = m(x) + \sum_{j \in \mathbb{N}} a_j(\omega) \, \varphi_j(x), \qquad a_j \sim \mathcal{N}(0, \ell_j),$$

in the sense of $L^2(\Omega, \mathbb{P}; L^2(D))$ convergence, where the $(a_j)\_{j \in \mathbb{N}}$ are independent real-valued RVs.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 4.5.24 (not examinable)</summary>

Throughout we use the basic property of Gaussian RVs from Exercise 4.5.23 that, for $X \sim \mathcal{N}(m, C)$ and $h \in H = L^2(D)$, one has $\langle X, h \rangle \sim \mathcal{N}(\langle m, h \rangle, \langle h, Ch \rangle)$.

**Step 1: $a(\omega) - m \in H := \overline{\mathrm{span}\lbrace \varphi_j : j \in \mathbb{N} \rbrace}$ $\mathbb{P}$-a.e.** Fix $h \in H^\perp$. Then

$$Ch = \sum_{j \in \mathbb{N}} \ell_j \underbrace{\langle h, \varphi_j \rangle_{L^2(D)}}_{=\,0} \varphi_j = 0.$$

By Exercise 4.5.23, $\langle a - m, h \rangle_{L^2(D)} \sim \mathcal{N}(0, \langle h, Ch \rangle_{L^2(D)}) = \mathcal{N}(0, 0) = \delta_0$, where $\delta_0$ is the Dirac measure at $0 \in \mathbb{R}$. Hence there is a $\mathbb{P}$-null set $N_h \subseteq \Omega$ with $\langle a(\omega) - m, h \rangle_{L^2(D)} = 0$ for every $\omega \in N_h^c$. Since $L^2(D)$ is separable, so is $H^\perp$, and we can pick a dense sequence $(h_n)\_{n \in \mathbb{N}} \subseteq H^\perp$. As $\bigcup_{n \in \mathbb{N}} N_{h_n}$ is a $\mathbb{P}$-null set, we conclude that $\mathbb{P}$-a.e. $\langle a(\omega) - m, h_n \rangle_{L^2(D)} = 0$ for all $n$, and thus $\mathbb{P}$-a.e.

$$(a(\omega) - m) \perp H^\perp \quad \Longleftrightarrow \quad a(\omega) - m \in H.$$

**Step 2: the expansion and the law of the coefficients.** We conclude that $\mathbb{P}$-a.e. in the sense of $L^2(D)$

$$a(\omega) = m + \sum_{j \in \mathbb{N}} a_j(\omega) \varphi_j, \qquad a_j(\omega) = \langle a(\omega) - m, \varphi_j \rangle_{L^2(D)}. \tag{4.5.13}$$

Again by Exercise 4.5.23, $a_j \sim \mathcal{N}(0, \ell_j)$, where we used $\langle \varphi_j, C\varphi_j \rangle_{L^2(D)} = \ell_j$. Independence of the $(a_j)\_{j \in \mathbb{N}}$ follows from them being uncorrelated jointly Gaussian RVs:

$$\mathrm{cov}(a_i, a_j) = \mathbb{E}[\langle a - m, \varphi_i \rangle_{L^2(D)} \langle a - m, \varphi_j \rangle_{L^2(D)}] = \langle \varphi_i, C\varphi_j \rangle_{L^2(D)} = \ell_i \delta_{ij}.$$

**Step 3: convergence.** By (4.5.13) and Parseval's identity,

$$\left\lVert a - \left( m + \sum_{j=1}^s a_j \varphi_j \right) \right\rVert_{L^2(\Omega; L^2(D))}^2 = \int_\Omega \left\lVert a(\omega) - m - \sum_{j=1}^s a_j(\omega) \varphi_j \right\rVert_{L^2(D)}^2 \mathrm{d}\mathbb{P}(\omega) = \sum_{j > s} \mathbb{E}[a_j^2] = \sum_{j > s} \ell_j \to 0,$$

since $C$ is of trace class by Thm. 4.5.21, which implies $\sum_{j \in \mathbb{N}} \ell_j < \infty$.

</details>
</div>

As in the uniform case, the following converse of the previous theorem holds, which provides a method to construct Gaussian RVs. The proof is left as an exercise.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5.25</span><span class="math-callout__name">(Constructing Gaussian RVs)</span></p>

Let $a : \Omega \times D \to \mathbb{R}$ be defined via

$$a(\omega, x) = m(x) + \sum_{j \in \mathbb{N}} a_j(\omega) \varphi_j(x)$$

where $m \in L^2(D)$, $(\varphi_j)\_{j \in \mathbb{N}}$ is an ONS of $L^2(D)$, $(\ell_j)\_{j \in \mathbb{N}} \in \ell^1(\mathbb{N})$ and the $a_j \sim \mathcal{N}(0, \ell_j)$ are independent.

Then $a \sim \mathcal{N}(m, T_c)$ with covariance function $c \in L^2(D \times D)$ given by

$$c(x, y) := \sum_{j \in \mathbb{N}} \ell_j \varphi_j(x) \varphi_j(y) \qquad \forall x, y \in D.$$

</div>

#### 4.5.5 Uninformative Priors

* https://www.youtube.com/results?search_query=jeffreys+prior

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(We have no prior information about the target)</span></p>

Suppose again that we wish to determine $X$ from the measurement $Y$. If we have no prior information about $X$, it is tempting to choose a uniform distribution as a prior. However, for example in case $X : \Omega \to \mathbb{R}$, this leads to an **improper** prior with density $\pi_X \equiv 1$, i.e. $\pi_X$ does not satisfy $\int_{\mathbb{R}} \pi_X(x) \, \mathrm{d}x = 1$. Improper priors may still be used, but are not in line with the theory discussed in this lecture. Furthermore, a uniform distribution should not be interpreted as being "uninformative":

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(My notes)</span></p>

**My impression on this problem:** Is not the problem that with the uniform distribution as a prior all events get probability zero if the event space is not bounded and closed like [a,b], but if the whole space is a sample space, than the whole uniform measure vanishes over the space since it spreads uniformly. So, uniform measure is uninformative measure on [a,b], but not on the whole space.

**Correction of my impression:** The problem is not that all individual events have probability zero. This already happens for ordinary continuous distributions such as the uniform distribution on $[0,1]$. The real problem is that there is no proper probability distribution on $\mathbb R$ with constant density. 

$$\text{there is no probability measure on } \mathbb R \text{ with constant positive density.}$$

If $\pi_X(x)=c$, then $\int_{\mathbb R} c dx$ is infinite for $c>0$ and zero for $c=0$. Hence a “uniform prior on $\mathbb R$” cannot be normalized to have total mass $1$. The improper prior $\pi_X\equiv 1$ should instead be understood as Lebesgue measure, which has infinite total mass.

On a bounded interval such as $[a,b]$, the uniform distribution is a proper probability measure. But even there, calling it “uninformative” is coordinate-dependent.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.5.26</span><span class="math-callout__name">(Uniform Prior is Not Uninformative)</span></p>

Suppose that we wish to find a parameter $X$. Assume that we know (a priori) that $X$ belongs to $[0, 1]$, but we know nothing else about $X$. We may choose the prior $X \sim \mathrm{uniform}(0, 1)$. 

* **Observation I:** Finding $X \in [0, 1]$ is equivalent to finding $X^2 \in [0, 1]$.
* **Observation II:** The RV $X^2$ is *not* uniformly distributed on $[0, 1]$:

  $$\mathbb{P}[X^2 \le a] = \mathbb{P}[X \le \sqrt{a}] = \sqrt{a}$$

  and thus $X^2$ has density $\pi_{X^2}(x) = \frac{1}{2} \frac{1}{\sqrt{x}}$. 

Hence this prior "favours" smaller values of $X^2$ over larger values of $X^2$. 

**This is counterintuitive:** If we have no information about $X \in [0, 1]$, then we also shouldn't have any information about $X^2 \in [0, 1]$.

#TODO: this example feels like a manipulation, because having a prior on X that it belongs only to [0,1] and is distributed there uniformaly is different from the prior on that it belongs only to [0,1] and is distributed there as \sqrt{x}. Yes, X^2 belongs to [0,1] <- X belongs to [0,1], but this is a new random variable although on the same support

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(My notes)</span></p>

**My impression on this problem:** This example feels like a manipulation, because having a prior on $X$ that it belongs only to $[0,1]$ and is distributed there uniformaly is different from the prior on that it belongs only to $[0,1]$ and is distributed there as $\sqrt{x}$. Yes, $X$ belongs to $[0,1]$ $\implies$ $X^2$ belongs to $[0,1]$, but this is a new random variable although on the same support

**Correction of my impression:** This example is not claiming that $X$ and $X^2$ are the same random variable. They are different random variables. The point is that, on $[0,1]$, they are equivalent parametrizations of the same unknown quantity, since the map $x\mapsto x^2$ is one-to-one. So, knowing $X$ is equivalent to knowing $X^2$, because $X=\sqrt{X^2}$.

If we choose $X\sim\mathrm{Unif}(0,1)$, then the induced prior on $Z=X^2$ is not uniform:

$$\mathbb P(Z\le a)=\mathbb P(X\le \sqrt a)=\sqrt a,$$

so

$$\pi_Z(z)=\frac{1}{2\sqrt z}.$$

Thus this prior favors smaller values of $Z$.

Conversely, if we chose $Z=X^2\sim\mathrm{Unif}(0,1)$, then the induced prior on $X=\sqrt Z$ would have density

$$\pi_X(x)=2x,$$

which favors larger values of $X$.

Therefore a uniform prior is not intrinsically uninformative. It is uniform only with respect to a chosen parametrization. Under nonlinear reparametrizations, a uniform prior generally becomes non-uniform.

So the main resolution is:

$$\boxed{\text{Uniform does not mean “no information”; it means “flat in this chosen coordinate.”}}$$

On $[0,1]$, uniform is a valid prior. But it is not coordinate-free. On $\mathbb R$, a uniform probability prior does not exist at all.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Partial Solution</span><span class="math-callout__name">(Then is there a coordinate-free uninformative prior?)</span></p>

**Short answer:** there is no universal coordinate-free “uninformative prior.” But there are several principled substitutes.

The clean distinction is:

$$\text{a prior measure can be coordinate-free, but “uninformative” is not intrinsic.}$$

A density such as $\pi(\theta))$ is always a density with respect to some reference measure, often Lebesgue measure in the chosen coordinate $\theta$. Under a change of variables, the density changes by the Jacobian. So “flat density” is not coordinate-free.

The closest standard notion is the Jeffreys prior.

For a statistical model

$$p(y\mid \theta),$$

define the Fisher information

$$I(\theta) = \mathbb E_\theta\left[\left(\frac{\partial}{\partial \theta}\log p(Y\mid \theta)\right)^2\right].$$

In one dimension, Jeffreys prior is

$$\pi_J(\theta) \propto \sqrt{I(\theta)}.$$

In dimension $d$,

$$\pi_J(\theta) \propto \sqrt{\det I(\theta)}.$$

This is invariant under smooth reparametrization. More geometrically, the Fisher information defines a Riemannian metric on parameter space, and Jeffreys prior is the corresponding volume element:

$$d\Pi_J(\theta) \propto \sqrt{\det I(\theta)},d\theta.$$

So if $\varphi = f(\theta)$, the prior transforms correctly. It does not matter whether you write the parameter as $\theta$, $\theta^2$, $\log \theta$, etc.

**But important caveat:** Jeffreys prior is not “pure ignorance.” It depends on the likelihood model $p(y\mid \theta)$. So it is coordinate-invariant, but not model-free.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.5.27</span><span class="math-callout__name">(Jeffreys Prior)</span></p>

Given a likelihood function $L(y, x) := \pi_{Y\mid X}(y\mid x)$ with $y \in \mathbb{R}^m$, $x \in \mathbb{R}^n$, **Jeffreys prior** is defined as

$$\pi_X(x) \propto \sqrt{\det(I_X(x))},$$

where $I_X(x) \in \mathbb{R}^{n \times n}$ is the **expected Fisher information** of $X$:

$$I_X(x) := \int_{\mathbb{R}^m} \nabla_x \ell(y, x) \cdot \nabla_x \ell(y, x)^\top \, \pi_{Y|X}(y|x) \, \mathrm{d}y, \qquad \ell(y, x) := \log L(y, x).$$

</div>

Jeffreys prior satisfies the following form of "invariance": Suppose that $X$ is a $\mathbb{R}^n$ valued RV and $g : \mathbb{R}^n \to \mathbb{R}^n$ is a diffeomorphism with nonnegative Jacobian determinant $\det Dg : \mathbb{R}^n \to (0, \infty)$. Then $\tilde{X} := g(X)$ is a RV representing another parametrization of $X$. To obtain a prior for the reparametrization $\tilde{X}$, we could now proceed in two ways:

* **(i)** Given the prior $\pi_X(x) \propto \sqrt{\det(I_X(x))}$, the density of $\tilde{X}$ is obtained after a change of variables as $\pi_X(g^{-1}(\tilde{x})) \det Dg^{-1}(\tilde{x})$ 
* **(ii)** We may set $\pi_{\tilde{X}}(\tilde{x}) = \sqrt{\det(I_{\tilde{X}}(\tilde{x}))}$ obtained with the reparametrized likelihood 
  
  $$\pi_{Y\mid\tilde{X}}(y\mid\tilde{x}) = \pi_{Y\mid X}(y\mid g^{-1}(\tilde{x}))$$

It can be shown that both constructions lead to the same prior.

<figure>
  <img src="{{ '/assets/images/notes/bip/bip_jeffreys_prior.png' | relative_url }}" alt="Left: the flat density of a uniform random variable on [0, 1] together with the density of its square, which blows up like one over two square root of x near zero. Right: the U-shaped Jeffreys prior Beta(1/2, 1/2) for the Bernoulli success probability compared with the flat uniform density." loading="lazy">
</figure>

*Left: Example 4.5.26 — if $X \sim \mathrm{uniform}(0, 1)$ then $X^2$ has density $\frac{1}{2\sqrt{x}}$, so "no information about $X$" silently becomes "strong preference for small $X^2$"; the uniform prior is not reparametrization-invariant. Right: for the Bernoulli likelihood, Jeffreys prior (Definition 4.5.27) is the $\mathrm{Beta}(\frac{1}{2}, \frac{1}{2})$ distribution, which differs from the uniform prior precisely so that the construction becomes invariant under reparametrization.*


## Chapter 5: Numerical Methods

After having carefully defined and analysed the well-posedness and stability of Bayesian inverse problems in a very general setting, we can now address the core task of the course, namely to consider some specific problems and solve them numerically.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Summary of the Chapter</span>(Numerical Methods)</p>

* We will start by looking at **some typical examples of inverse problems** in applications, 
  * which all have an **infinite-dimensional state space** and 
  * often also an **infinite-dimensional (or at least high-dimensional) parameter space**. 
* We then analyse the effect of **numerical approximation on the posterior distribution**, 
  * before considering the Gaussian case (for prior and additive noise) with a linear forward operator, which can be solved in closed form. 
* In Section 5.4, we recall some classical and more advanced sampling-based quadrature methods for high dimensions 
  * and in a first attempt apply them directly to compute the conditional mean with respect to the posterior distribution in Bayesian inverse problems. 
* Finally in Sections 5.6-5.8 we present the **main numerical methods applied in general to solve Bayesian inverse problems** in practice: 
  * MCMC methods, 
  * Variational methods,
  * Sequential Monte Carlo.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span>(State Space vs. Parameter Space)</p>

* **State space** represents the set of all possible configurations or values a dynamic system can be in at any given time,
* **Parameter space** represents the set of all possible constant coefficients, weights, or characteristics that define *how* the system behaves or transitions

</div>

### 5.1 Examples

Our main focus will be on *PDE-constrained Bayesian inverse problems*, but more classical examples are typically in the context of integral equations, such as the problem of **X-ray tomography**, or from **spatial statistics**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span>(PDE-constrained Bayesian inverse problems)</p>

* **PDE-constrained Bayesian inverse problems** involve estimating unknown parameters or states in a physical system (modeled by partial differential equations) using noisy observational data. 
* By casting this as a Bayesian inference problem, the goal is to compute the *posterior distribution*, which quantifies the uncertainty of the parameters given both the data and prior physical knowledge.

</div>

#### X-ray Tomography

Given a bounded domain $D$, for simplicity $D \subset \mathbb{R}^2$, representing a cross-sectional slice of the object to be studied. A pointlike X-ray source is placed on one side of the object. The radiation passes through the object and is detected on the other side. It is common to assume that the scattering of the X-rays by the traversed material is insignificant, i.e., only absorption occurs. If we further assume that the **mass absorption coefficient** is proportional to the density of the material, the attenuation $\mathrm{d}I$ of the intensity $I(x)$ along a line segment $\mathrm{d}s$ at a point $x \in D$ is given by

$$\mathrm{d}I = -I(x)\theta(x)\mathrm{d}s$$

where $\theta(x) \ge 0$ is the mass absorption coefficient of the material. If an X-ray is transmitted with intensity $I_0^\ell$ along a straight line $\ell$ towards a receiver, the received intensity $I_r^\ell$ can be obtained from the equation

$$\log I_r^\ell - \log I_0^\ell = \int_{I_0^\ell}^{I_r^\ell} \frac{\mathrm{d}I}{I} = -\int_\ell \theta(x) \, \mathrm{d}s. \tag{5.1.1}$$

The inverse problem of X-ray tomography can thus be stated as a problem of integral geometry: Estimate the function $\theta : D \to \mathbb{R}\_{+}$ from the values of its integrals along a set of straight lines passing through $D$. This leads to the linear operator equation

$$y = \mathcal{R}\theta$$

with compact integral operator $\mathcal{R}$, the so-called **Radon transform**. In the ideal case, we have data along all possible lines passing through the object. The problem can then be solved explicitly using the **inverse Radon transform**. However, it involves differentiating the data, which is an ill-posed problem in the sense of Hadamard.

In practice, often only limited-angle data is available and the data is polluted by electronic noise that can be assumed to be additive Gaussian noise $E \sim \mathcal{N}(0, \Sigma)$ to a good approximation. This can be formulated as an infinite-dimensional (linear) Bayesian inverse problem: $Y = \mathcal{R}\Theta + E$.

#### Gaussian Process Regression (Kriging)

Many problems in spatial statistics are of the form that a functional quantity is to be estimated from a few point evaluations, a form of statistical interpolation also called **kriging**.

Let $D \subset \mathbb{R}^d$ be a bounded open set. Consider a field $u \in \mathcal{H} = L^2(D; \mathbb{R}^n)$. Assume that we are given noisy observations $\lbrace y_k \rbrace_{k=1}^q$ of a function $g : \mathbb{R}^n \to \mathbb{R}^\ell$ of the field $u$ at a set of points $\lbrace x_k \rbrace_{k=1}^q$. Thus $y_k = g(u(x_k)) + \eta_k$, where the $\lbrace \eta_k \rbrace_{k=1}^q$ describe the observational noise. Concatenating data, we have

$$y = \mathcal{G}(u) + \eta,$$

where $y = (y_1^\top, \ldots, y_q^\top)^\top \in \mathbb{R}^{\ell q}$ and $\eta = (\eta_1^\top, \ldots, \eta_q^\top)^\top \in \mathbb{R}^{\ell q}$. The observation operator $\mathcal{G}$ maps $V = (C(\overline{D}))^n \subset \mathcal{H}$ to $W = \mathbb{R}^{\ell q}$. The inverse problem is to reconstruct the field $u$ from the data $y$.

We assume that the observational noise $\eta$ is Gaussian $\mathcal{N}(0, \Sigma)$ and specify a prior measure $\mu_U$ on the random field $U$, which is Gaussian $\mathcal{N}(m_0, \mathcal{C}\_0)$ and determine the posterior measure $\mu_{U\mid y}$ for $U$ given $y$. By Example 4.3.3, the Radon-Nikodym derivative of the posterior w.r.t. the prior is

$$\frac{\mathrm{d}\mu_{U|y}}{\mathrm{d}\mu_U}(u) \propto \exp\left( -\frac{1}{2} \lVert y - \mathcal{G}(u) \rVert_\Sigma^2 \right)$$

which (up to a constant) corresponds to the likelihood. The negative log-likelihood $\frac{1}{2} \lVert y - \mathcal{G}(u) \rVert_\Sigma^2$ is the **data misfit potential**.

If $g : \mathbb{R}^n \to \mathbb{R}^\ell$ is linear, so that $\mathcal{G}(u) = Au$ for some linear operator $A : V \to W$, then the posterior measure $\mu_{U\mid y}$ is also Gaussian with a mean and covariance operator that can be computed explicitly (see Section 5.3). This is called a **Gaussian process** and the data fitting approach is called **Gaussian process regression**.

#### Inverse Heat Equation

Let us return to the motivating example in Section 1.1, the inverse heat equation. The forward problem admitted the explicit solution

$$u(x, t) = \sum_{n=1}^{\infty} \theta_n e^{-(n\pi)^2 \alpha t} \sin(n\pi x),$$

where $\theta_n = \langle u_0, \sin(n\pi \cdot) \rangle_{L^2(0,1)}$ are the Fourier-sine-coefficients of the initial condition $u_0$.

Denoting by $y_n$ the Fourier-sine coefficients of the measured data at time $T > 0$, assuming an additive measurement noise, we obtain the following infinite-dimensional (linear) Bayesian inverse problem: to find posterior distribution $\mu_{\Theta\mid y}$ for the Fourier coefficients $\Theta$ of $u_0$ (understood as a random sequence in $\ell^1$) such that

$$Y = \Lambda \Theta + E$$

with the linear operator $\Lambda : \ell^1 \to \ell^1$ that can be represented as an infinite diagonal matrix with diagonal entries $\Lambda_{nn} := e^{-(n\pi)^2 \alpha T}$.

This viewpoint shows very clearly how the inversion leads to an exponential amplification of any errors in the higher frequencies. In the case of a Gaussian prior and a Gaussian measurement error the posterior distribution is again Gaussian and the problem can be solved explicitly.

#### Subsurface Flow, Heat Conduction, Impedance Tomography

The main model problem we will consider in the rest of this chapter is the problem to identify the diffusion coefficient $a$ in a stationary diffusion problem, the elliptic PDE defined in Example 4.5.1. This model problem is ubiquitous in many fields of mathematics: heat conduction, electrostatics, porous media flow, magnetostatics, or radiation shielding. The Bayesian inverse problem associated with this problem has already been extensively described and analysed in Section 4.5.

There are many more examples of important inverse problems in applications -- such as **inverse scattering** (geophysics, MRT), **inverse source problems** (Tsunami prediction, subsurface pollution), **data assimilation** (weather prediction), **parameter estimation** (pattern formation in developmental biology) or **epidemiology** (COVID-19 modelling and prediction).

### 5.2 Discretisation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span>(Discretization)</p>

To numerically solve such an infinite-dimensional inverse problem it is of course necessary to discretise the problem. The approximation error then leads to a bias in the posterior distribution and in any derived quantities, such as the conditional mean, that needs to be estimated:
* Reduce the infinite-dimensional inverse problem to finite-dimensional surrogate problem via discretization.
* Reduction losses some information introducing a bias in the solution.

The goal is then to estimate/control this bias and show that it vanishes as

$$N\to\infty \quad \text{or} \quad h\to 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span>(Discretization)</p>

**The original mathematical problem remains infinite-dimensional.**
For example, the unknown might be a function

$$a \in L^\infty(D), \qquad u_0 \in L^2(0,1),$$

or an infinite sequence of Fourier coefficients

$$\Theta = (\theta_1,\theta_2,\ldots).$$

A computer cannot store or sample from this object directly, so we choose a finite representation.

**The discretised problem is finite-dimensional.**

For example:

$$u_0(x)=\sum_{n=1}^\infty \theta_n \sin(n\pi x)$$

is replaced by

$$u_{0,N}(x)=\sum_{n=1}^N \theta_n \sin(n\pi x),$$

so the unknown becomes the finite vector

$$(\theta_1,\ldots,\theta_N)\in \mathbb R^N.$$

This is exactly what the chapter says for the inverse heat equation: discretisation is achieved by truncating the infinite series after $N$ terms. For tomography, it says the domain is divided into pixels/voxels, so the unknown function is replaced by finitely many pixel values; for PDE problems, the forward operator may be discretised by finite elements. 

So the situation is:

$$
\text{infinite-dimensional problem}
\quad \leadsto \quad
\text{finite-dimensional approximation depending on } h \text{ or } N.
$$

Here $h$ is a mesh size, and $N$ is the number of retained basis coefficients.

The goal is then to estimate/control this bias and show that it vanishes as

$$N\to\infty \quad \text{or} \quad h\to 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span>(X-ray tomography problem discretization)</p>

For the **X-ray tomography problem**, the domain $D$ is commonly subdivided using a uniform Cartesian grid into pixels (or voxels in three dimensions) and the absorption coefficient $\theta$ is then approximated by a piecewise constant approximation $\theta\_h$, where $h$ denotes the mesh size.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span>(Inverse heat equation problem discretization)</p>

In the **inverse heat equation problem**, the forward problem was already represented in an orthonormal system. In that case the discretisation is very naturally (and in some sense optimally) achieved by truncating the infinite series expansions after a suitable number of terms $N$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span>(Forward operator is typically discretized in real problems)</p>

In simple examples, the forward map is known by a formula; in realistic evolution problems, it is not, so even the map from initial state to later observations must be approximated numerically

In general, however, finding the initial condition of an evolution equation, e.g. in data assimilation for weather forecasting or for tsunami prediction, the forward problem can not be solved explicitly and the forward operator needs to be discretised, e.g. via finite elements.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span>(Forward operator is typically discretized in real problems)</p>

It means: **in simple examples, the forward map is known by a formula; in realistic evolution problems, it is not, so even the map from initial state to later observations must be approximated numerically.**

In the inverse heat equation example, the chapter has an explicit formula:

[
u(x,t)=\sum_{n=1}^{\infty}\theta_n e^{-(n\pi)^2\alpha t}\sin(n\pi x).
]

So the forward operator is explicitly diagonal in the sine basis:

[
\Theta \mapsto \Lambda \Theta,
\qquad
\Lambda_{nn}=e^{-(n\pi)^2\alpha T}.
]

That is very convenient: discretisation just means keeping finitely many Fourier modes. The notes say exactly this: for the inverse heat equation, the forward problem is already represented in an orthonormal system, so discretisation is naturally done by truncating the infinite series after (N) terms. 

The remark says that this is **not the typical situation**.

For weather, tsunami prediction, fluid dynamics, etc., the state (u(t)) solves some complicated evolution equation, schematically

[
\frac{du}{dt}=F(u),
\qquad u(0)=u_0.
]

The inverse problem is:

[
\text{given observations at later times, infer } u_0.
]

So the forward operator is

[
\mathcal G(u_0)
===============

\text{observed quantities obtained by evolving } u_0 \text{ forward in time}.
]

But usually there is no closed formula for (\mathcal G(u_0)). You cannot write down something nice like

[
\mathcal G(u_0)
===============

\sum_{n=1}^{\infty}\theta_n e^{-(n\pi)^2\alpha T}\sin(n\pi x).
]

Instead, to evaluate (\mathcal G(u_0)), you must **numerically solve the PDE**.

That is what “the forward operator needs to be discretised” means. You replace the infinite-dimensional PDE by a finite-dimensional numerical scheme, for example finite elements:

[
u(t,x)
\quad \leadsto \quad
u_h(t,x)=\sum_{i=1}^{N_h} U_i(t)\varphi_i(x),
]

where (\varphi_i) are finite element basis functions. Then the infinite-dimensional evolution equation becomes a finite system of ODEs/algebraic equations for the coefficients

[
U_1(t),\ldots,U_{N_h}(t).
]

So instead of the exact forward operator

[
\mathcal G : u_0 \mapsto y,
]

we compute an approximate forward operator

[
\mathcal G_h : u_{0,h} \mapsto y_h.
]

The notes contrast this with tomography, where the domain is divided into pixels/voxels, and with the inverse heat equation, where one truncates Fourier modes; in the general PDE case, finite elements are one standard way to discretise the forward problem. 

So the remark is saying:

[
\boxed{
\text{In realistic inverse evolution problems, discretisation is not only about the unknown.}
}
]

It is also about the **forward solver**. You approximate the map “initial condition (\mapsto) later state/observations” because the exact map is not available in closed form.

</div>

#### 5.2.1 Finite Element Analysis of the Elliptic Model Problem

Let $D \subseteq \mathbb{R}^d$ be a bounded Lipschitz domain and consider the weak formulation of the elliptic PDE (4.5.1): Find $u \in H_0^1(D)$ such that

$$\int_D a \nabla u(x)^\top \nabla v(x) \, \mathrm{d}x = \int_D f(x) v(x) \, \mathrm{d}x \qquad \forall v \in H_0^1(D), \tag{5.2.1}$$

where for simplicity we assume that $f \in L^2(D)$ and that it is known and not random. Furthermore, as above $a \in L^\infty(D)$ with $\mathrm{essinf}_{x \in D} a(x) > 0$ and the forward (observation) operator $\Phi(a) := Bu \in \mathbb{R}^m$ for some bounded linear operator $B : H_0^1(D) \to \mathbb{R}^m$. The inverse problem is to find the diffusion coefficient $a \in L^\infty(D)$ from noisy measurements

$$Y = \Phi(a) + E, \tag{5.2.2}$$

with $E \sim \mathcal{N}(0, \Sigma)$ for an SPD matrix $\Sigma \in \mathbb{R}^{m \times m}$. We assume that for almost all $\omega \in \Omega$ ($\mathbb{P}$-a.s.), realizations $a(\cdot, \omega)$ of the coefficient function $a$ are strictly positive, lie in $L^\infty(D)$ and satisfy

$$0 < a_{\min}(\omega) \le a(x, \omega) \le a_{\max}(\omega) < \infty \quad \text{for a.e. } x \in D, \tag{5.2.3}$$

where $a_{\min}(\omega) := \mathrm{essinf}\_{x \in D} a(x, \omega)$ and $a_{\max}(\omega) := \mathrm{esssup}\_{x \in D} a(x, \omega)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.2.1</span><span class="math-callout__name">(Well-Posedness of the Elliptic PDE)</span></p>

$\mathbb{P}$-a.s. problem (5.2.1) has a unique solution $u(\cdot, \omega) \in H_0^1(D)$ and

$$|u(\cdot, \omega)|_{H^1(D)} \le C a_{\min}^{-1}(\omega) \lVert f \rVert_{L^2(D)}.$$

If $a_{\min}^{-1} \in L^p(\Omega)$, for some $p \in [1, \infty]$, then

$$\lVert u \rVert_{L^p(\Omega; H_0^1(D))} \le C \lVert a_{\min}^{-1} \rVert_{L^p(\Omega; \mathbb{R})} \lVert f \rVert_{L^2(D)}.$$

</div>

Let $U_h \subset H_0^1(D)$ denote a closed subspace, e.g., the finite element (FE) space of piecewise polynomial functions with respect to a triangulation $\mathcal{T}_h$ of $D$ with mesh width $h > 0$. Suppose $u_h : \Omega \to U_h$ satisfies $\mathbb{P}$-a.s.

$$\int_D a(x, \omega) \nabla u_h(x, \omega)^\top \nabla v_h(x) \, \mathrm{d}x = \int_D f(x) v_h(x) \, \mathrm{d}x \quad \forall v_h \in U_h. \tag{5.2.5}$$

Since $U_h$ is a closed subspace of $H_0^1(D)$ with norm $\|\cdot\|_{H^1(D)}$ all the above results hold in an identical form also for $u_h$:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.2.2</span><span class="math-callout__name">(FE Solution Bounds)</span></p>

The results and the bounds in Theorem 5.2.1 hold under the same assumptions on $a$ and $f$ also for the FE system (5.2.5) and its solution $u_h$.

</div>

To bound the FE error we also need a regularity assumption.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption 5.2.3</span></p>

$\mathbb{P}$-a.s. for all $\omega \in \Omega$, $u(\cdot, \omega) \in H^2(D)$ and there exists a $q \in [1, \infty]$ such that

$$\lVert u \rVert_{L^q(\Omega; H^2(D))} \le C \lVert f \rVert_{L^2(D)}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.2.4</span></p>

Sufficient conditions for $u(\cdot, \omega) \in H^2(D)$ are that $D$ is convex, that $a(\cdot, \omega)$ is Lipschitz continuous and that (5.2.3) holds. For a uniform distribution, as in Section 4.5.3, Assumption 5.2.3 holds with $q = \infty$. However, a more commonly used type of prior distribution, especially in subsurface flow, is a **lognormal distribution** for $a$ with **Matérn covariance**. In that case, $\log a$ is a Gaussian field that admits a Karhunen-Loève expansion as in Theorem 4.5.24, and it can be shown that Assumption 5.2.3 holds for all $q < \infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.2.5</span><span class="math-callout__name">(FE Error Bounds)</span></p>

Let Assumption 5.2.3 hold and let $\sqrt{a_{\max}/a_{\min}} \in L^r(\Omega)$ with $q, r \in [1, \infty]$. Suppose $U_h \subset H_0^1(D)$ is the piecewise linear FE space associated with a triangulation $\mathcal{T}\_h$ of $D$. Then, for any $p \in [1, \infty]$ with $\frac{1}{p} \ge \frac{1}{q} + \frac{1}{r}$,

$$\lVert u - u_h \rVert_{L^p(\Omega; H_0^1(D))} \le Ch \lVert f \rVert_{L^2(D)}.$$

Moreover, for any bounded linear operator $B : H_0^1(D) \to \mathbb{R}^m$,

$$\lVert Bu - Bu_h \rVert_{L^p(\Omega; \mathbb{R}^m)} \le Ch^2. \tag{5.2.6}$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/disc_fe_error_rates.png' | relative_url }}" alt="Left: the reference solution of the 1D elliptic problem together with piecewise linear finite element approximations on meshes with 4, 8 and 16 cells, visibly converging to the smooth curve. Right: log-log plot of the H1 error and the error of the point functional u(0.5) against the mesh size h; the H1 error follows a straight line of slope 1 and the functional error a straight line of slope 2." loading="lazy">
</figure>

*One-dimensional illustration of Theorem 5.2.5: piecewise linear FE approximations of $-(a u')' = 1$ with $a(x) = 2 + 0.9 \sin(2\pi x)$ (left), and the two convergence rates (right) — the $H^1$-error decays like $h$, while the error in the bounded linear functional $u \mapsto u(0.5)$ decays like $h^2$ (measured rates $1.00$ and $2.00$). It is this doubled functional rate that delivers $\alpha = 2$ in the complexity theorems of Section 5.4.*

We are now in a position to extend these results to bound the bias in the posterior measure and in the conditional mean of any derived quantities of interest due to the FE approximation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.2.6</span><span class="math-callout__name">(Discretisation Error in Posterior)</span></p>

Let us assume that $a_{\min}^{-1} \in L^2(\Omega)$ and the assumptions of Theorem 5.2.5 hold for $p = 2$. Suppose $B : H_0^1(D) \to \mathbb{R}^m$ is a bounded linear operator and let the (exact) forward operator $\Phi : L^\infty(D) \to \mathbb{R}^m$ be defined by $\Phi(a) := Bu$. Under the noise model (5.2.2) this induces the posterior measure $\nu := \mu_{a\mid y}$ on the diffusion coefficient $a \in L^\infty(D)$, as described in Chapter 4. In the same way, the discretised observation operator $\Phi_h(a) := Bu_h$ induces an approximate posterior measure $\nu_h$ on $a$ and

$$D_{\mathrm{H}}(\nu, \nu_h) \le Ch^2. \tag{5.2.7}$$

Moreover, for any bounded linear operator $G : H_0^1(D) \to \mathbb{R}$, the approximation error in the posterior expectation of the functional $\Psi(a) := Gu$, which is approximated by $\Psi_h(a) := Gu_h$ can be bounded as

$$\left| \mathbb{E}_\nu[\Psi(a)] - \mathbb{E}_{\nu_h}[\Psi_h(a)] \right| \le Ch^2. \tag{5.2.8}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 5.2.6</summary>

The two measures $\nu$ and $\nu\_h$ satisfy the assumptions of Theorem 4.4.1. Thus, the bound on the Hellinger distance follows directly by applying the bound (5.2.6) in Theorem 4.4.1.

For the bound on the posterior expectations, we note first that due to the additive Gaussian noise assumption, for any measurable function $f : L^\infty(D) \to \mathbb{R}^m$ and for $q \in \mathbb{N}$,

$$\left| \mathbb{E}_\nu[f(a)^q] \right| \le \mathbb{E}_\nu\big[|f(a)|^q\big] \le \frac{1}{Z_\nu} \mathbb{E}_{\mu_a}\big[|f(a)|^q\big], \tag{5.2.9}$$

which follows immediately by taking the supremum of the Radon-Nikodym derivative out of the integral and bounding it by $\frac{1}{Z\_\nu}$, where $Z\_\nu$ is the normalization constant for $\nu$ (as in (4.4.1b)). The analogous bound holds for $\nu\_h$.

Now we use the triangle inequality to separate the error into the FE error in the posterior measure and the FE error in approximating the target functional, i.e.

$$\left| \mathbb{E}_\nu[\Psi(a)] - \mathbb{E}_{\nu_h}[\Psi_h(a)] \right| \le \left| \mathbb{E}_{\nu_h}[\Psi(a) - \Psi_h(a)] \right| + \left| \mathbb{E}_\nu[\Psi(a)] - \mathbb{E}_{\nu_h}[\Psi(a)] \right| \tag{5.2.10}$$

The bound on the first term follows from (5.2.9) and (5.2.6) (with $G$ instead of $B$). For the second term, we proceed as in the proof of Lemma 3.5.7, i.e.

$$\begin{aligned} \left| \mathbb{E}_\nu[\Psi(a)] - \mathbb{E}_{\nu_h}[\Psi(a)] \right| &\le 2D_{\mathrm{H}}(\nu, \nu_h) \left( \int_\Omega |\Psi(a)|^2 \left( \frac{\mathrm{d}\nu}{\mathrm{d}\mu_a} + \frac{\mathrm{d}\nu_h}{\mathrm{d}\mu_a} \right) \mathrm{d}\mu_a \right)^{1/2} \\ &\le Ch^2 \lVert G \rVert_{H^{-1}(D)} \lVert u \rVert_{L^2(\Omega; H_0^1(D))}, \end{aligned}$$

where in the last step we have used (5.2.7), (5.2.9) and the boundedness of $G$. The result then follows from Theorem 5.2.1.

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The error pipeline: from mesh size to posterior bias)</span></p>

Theorem 5.2.6 is the payoff of the whole section, and it is worth pausing on what it actually chains together. Three separate approximation steps happen, and each contributes its own factor:

1. **The PDE solve.** Replacing $u$ by the FE solution $u\_h$ costs $O(h)$ in the $H^1$-norm, but $O(h^2)$ for *functionals* $Bu$ of the solution (Theorem 5.2.5). The doubled rate for functionals is the classical Aubin-Nitsche duality argument from FE theory: $u - u\_h$ is Galerkin-orthogonal to the FE space, so testing it against the solution of a dual problem gains an extra power of $h$.
2. **The likelihood.** The posterior density depends on $a$ only through the forward map $\Phi(a) = Bu$. Since the Gaussian likelihood is locally Lipschitz in $\Phi$, an $O(h^2)$ perturbation of the forward map perturbs the posterior by $O(h^2)$ in Hellinger distance — this is exactly the stability Theorem 4.4.1, applied with $\Phi\_1 = \Phi$ and $\Phi\_2 = \Phi\_h$. Well-posedness of the Bayesian inverse problem is thus not just an abstract nicety: it is the mechanism that transfers discretisation error into posterior bias *without amplification*.
3. **The quantity of interest.** Hellinger distance is precisely the metric that controls differences of expectations of square-integrable functionals (Lemma 3.5.7); it converts the $O(h^2)$ distance between the measures into an $O(h^2)$ bias in the posterior expectation.

The structural point to remember: **the posterior bias inherits the functional rate $h^2$, not the energy rate $h$**, because the data only ever "sees" the PDE solution through the bounded linear functionals $B$ (and the QoI through $G$). This $\alpha = 2$ bias rate is exactly what enters the complexity theorems for MC, MLMC and QMC in Section 5.4.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.2.7</span></p>

**(a)** Similar results can be proved for Fréchet-differentiable nonlinear functionals $G : H_0^1(D) \to \mathbb{R}$ of the PDE solution with $\Psi(a) := G(u)$ or for other Fréchet-differentiable nonlinear functionals $\Psi : L^\infty \to \mathbb{R}$ (with measurable Fréchet derivative).

**(b)** Another approximation error that we have not discussed so far concerns the numerical approximation of the prior distribution. In case of the Karhunen-Loève expansion (cf. Sect. 4.5.2), a natural way to discretise the prior is truncation of the series expansion (4.5.4) at a suitably large index $s \in \mathbb{N}$. For the case of Matérn covariances, both in the uniform and in the lognormal case it can be shown, e.g., in [Graham et al, 2015] that there exists a $\chi > 0$ such that $\lVert Bu_h - Bu_{s,h} \rVert_{L^p(\Omega)} \le Cs^{-\chi}$, where $u_{s,h}$ is the solution to a FE system like (5.2.5) but with truncated coefficient function $a_s$ instead of $a$ and the value of $\chi$ depends on the smoothness parameter in the Matérn covariance. From this it can again be deduced that $D_{\mathrm{H}}(\nu_h, \nu_{s,h}) \le Cs^{-\chi}$.

</div>

### 5.3 Linear Problems and the Laplace Approximation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Linear Gaussian Problems are analytically tractable)</span></p>

If

1. the observation model is
   
   $$Y = \Phi X + E,$$

   where $E$ is additive Gaussian noise,

2. the prior $\mu\_X$ is Gaussian, and

3. the forward operator $\Phi$ is linear,

then the posterior measure $\mu\_{X\mid y}$ is again Gaussian. In particular, its mean and covariance can be written explicitly.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analytically tractable vs. Solvable exactly)</span></p>

In the motivation above we replace **“solvable exactly”** by **“analytically tractable”** or **“posterior is explicit”**, because in infinite dimensions we may still need numerical methods to evaluate the formula in practice. The posterior is exact at the mathematical level, but computing it may still require discretising covariance operators, solving linear systems, or approximating the forward operator.

The danger is that “solvable exactly” might sound like “no numerics are ever needed,” which is not quite true.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.3.1</span><span class="math-callout__name">(Posterior for Linear Gaussian Problems)</span></p>

Let $H$ be a separable Hilbert space and $X : \Omega \to H$ a RV with prior distribution $\mathcal{N}(\overline{x}, C)$ with positive covariance operator $C$. Let $W = \mathbb{R}^m$ and assume $E : \Omega \to \mathbb{R}^m$ is a Gaussian RV independent of $X$ that is distributed according to $\mathcal{N}(0, \Sigma)$ with SPD covariance matrix $\Sigma \in \mathbb{R}^{m \times m}$. Suppose furthermore that the forward operator $\Phi : H \to \mathbb{R}^m$ is linear, i.e. $\Phi(x) = Ax$ and $Y = AX + E$. Then the posterior measure $\mu_{X\mid y}$ is Gaussian $\mathcal{N}(x_{\mathrm{CM}}, C_{X\mid y})$ with

$$x_{\mathrm{CM}} := \overline{x} + CA^*(\Sigma + ACA^*)^{-1}(y - A\overline{x}) \tag{5.3.1}$$

$$C_{X|y} := C - CA^*(\Sigma + ACA^*)^{-1}AC \tag{5.3.2}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 5.3.1</summary>

For the proof we restrict ourselves to $H = \mathbb{R}^s$, but most steps extend straightforwardly also to infinite dimensions.

Given the additive model (4.1.1) for the observable RV $Y$, the RV $Z := \binom{X}{Y} = \binom{I \quad 0}{A \quad I} \binom{X}{E}$ is jointly Gaussian

$$Z \sim \mathcal{N}\left( \binom{\overline{x}}{A\overline{x}}, \begin{pmatrix} C & CA^* \\ AC & \Sigma + ACA^* \end{pmatrix} \right) =: \mathcal{N}(m_Z, C_Z) \tag{5.3.3}$$

This follows directly from the fact that for any random variables $X_1$ and $X_2$ under linear transformations $A_1$, $A_2$ the covariance operator satisfies $\mathrm{cov}(A_1 X_1, A_2 X_2) = A_1 \mathrm{cov}(X_1, X_2) A_2^*$.

Since $\Sigma$ and $C$ are SPD, the bottom-right-block $C_Y := \Sigma + ACA^*$ is also SPD and we can block-$LDL^T$ factorise the covariance matrix of $(X, Y)$ to give

$$\begin{pmatrix} C & CA^* \\ AC & C_Y \end{pmatrix}^{-1} = \begin{pmatrix} I & 0 \\ -C_Y^{-1}AC & I \end{pmatrix} \begin{pmatrix} (C - CA^*C_Y^{-1}AC)^{-1} & 0 \\ 0 & C_Y^{-1} \end{pmatrix} \begin{pmatrix} I & -CA^*C_Y^{-1} \\ 0 & I \end{pmatrix}$$

and thus, since

$$\begin{pmatrix} I & -CA^*C_Y^{-1} \\ 0 & I \end{pmatrix} \begin{pmatrix} x - \overline{x} \\ Y - A\overline{x} \end{pmatrix} = \begin{pmatrix} x - x_{\mathrm{CM}} \\ Y - A\overline{x} \end{pmatrix}$$

with $x_{\mathrm{CM}}$ as defined in (5.3.1) and with $C_{X\mid y} := C - CA^*C_Y^{-1}AC$, it follows that

$$(z - m_Z)^* C_Z^{-1} (z - m_Z) = \begin{pmatrix} x - \overline{x} \\ y - A\overline{x} \end{pmatrix}^* \begin{pmatrix} C & CA^* \\ AC & C_Y \end{pmatrix}^{-1} \begin{pmatrix} x - \overline{x} \\ y - A\overline{x} \end{pmatrix} = \begin{pmatrix} x - x_{\mathrm{CM}} \\ y - A\overline{x} \end{pmatrix}^* \begin{pmatrix} C_{X|y}^{-1} & 0 \\ 0 & C_Y^{-1} \end{pmatrix} \begin{pmatrix} x - x_{\mathrm{CM}} \\ y - A\overline{x} \end{pmatrix}$$

Due to the diagonal structure of the covariance and the formula for conditional densities, $\pi_{X,Y}(x, y) = \pi_{X\mid Y}(x\mid y)\pi(y)$, this completes the proof. The employed technique is sometimes called *"completing the square"*.

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Still need in numerical methods for analytically tractlable problems)</span></p>

* In principle, this solves the problem in the linear Gaussian case and for **low- to intermediate-dimensional** problems. 
  * In that case, it is possible to assemble and factorise $C\_{X\mid y}$ and to perform inference from this posterior distribution. 
* In **high dimensions**, factorisation might be prohibitive and it is necessary to consider alternatives
  * this will be the focus of the next three sections. 
  * However, due to its importance and explicit tractability, there is a large body of literature on efficient numerical methods specifically for the Gaussian case.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Prohibitive factorization of $C\_{X\mid y}$)</span></p>

They are **not mainly referring to the block factorisation of the joint covariance** in

$$\begin{pmatrix} C & CA^* \\ AC & C_Y \end{pmatrix}^{-1} = \begin{pmatrix} I & 0 \\ -C_Y^{-1}AC & I \end{pmatrix} \begin{pmatrix} (C - CA^*C_Y^{-1}AC)^{-1} & 0 \\ 0 & C_Y^{-1} \end{pmatrix} \begin{pmatrix} I & -CA^*C_Y^{-1} \\ 0 & I \end{pmatrix}$$

That factorisation is used in the **proof** to derive the conditional Gaussian formula. The motivation after the proof says that, once we know

$$X\mid Y=y \sim \mathcal N(x_{\mathrm{CM}},C_{X\mid y}),$$

we may still need to **assemble and factorise the posterior covariance**

$$C_{X\mid y} = C-CA^*(\Sigma+ACA^*)^{-1}AC.$$

So the problematic factorisation is usually something like a **Cholesky factorisation** or **square-root factorisation**

$$C_{X\mid y}=LL^T,$$

or equivalently an eigendecomposition

$$C_{X\mid y}=Q\Lambda Q^T.$$

Why do we need that? Because to sample from the posterior

$$X\mid y \sim \mathcal N(x_{\mathrm{CM}},C_{X\mid y}),$$

one standard method is

$$
X = x_{\mathrm{CM}} + L\xi,
\qquad
\xi\sim \mathcal N(0,I).
$$

So we need a factor $L$ satisfying

$$LL^T=C_{X\mid y}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cholesky Decomposition Complexity)</span></p>

* The time complexity of **Cholesky decomposition** is $\mathcal{O}(n^3)$, requiring exactly $\frac{n^{3}}{3}$ floating-point operations (flops) for large matrices. 
* For an $n\times n$ symmetric positive-definite matrix, the space complexity is $\mathcal{O}(n^2)$ because it requires storing the resulting lower triangular matrix.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Conditional mean $x\_{\text{CM}}$ of linear Gaussian case == MAP point)</span></p>

Note that in the linear Gaussian case, the conditional mean $x_{\mathrm{CM}}$ is in fact **identical** to the MAP point, which for $H = \mathbb{R}^s$ can be computed as

$$x_{\mathrm{MAP}} = \operatorname{argmax}_{x \in \mathbb{R}^s} \pi_{X|Y}(x|y) = \operatorname{argmin}_{x \in H} \frac{1}{2} \lVert y - Ax \rVert_\Sigma^2 + \frac{1}{2} \lVert x - \overline{x} \rVert_C^2 = x_{\mathrm{CM}}$$

and is in fact also **identical** to the solution of a generalised Tikhonov-regularised system as discussed in Section 2.5 (with suitable norms on the parameter space $X$ and on the observation space $Y$, in the notation there).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

In the finite-dimensional case $H=\mathbb R^s$, the proof is quite direct.

We use the conventions

$$[|v|_\Sigma^2 := v^\top \Sigma^{-1}v, \qquad |x-\overline{x}|_C^2 := (x-\overline{x})^\top C^{-1}(x-\overline{x}),$$

where $C$ and $\Sigma$ are symmetric positive definite. The theorem in your notes gives the posterior as a Gaussian with mean (x_{\mathrm{CM}}) and covariance (C_{X\mid y}). The proposition then observes that, in the linear Gaussian case, this mean is also the posterior mode/MAP point. 

**Key idea.**
A Gaussian density is maximised at its mean, while Bayes’ formula shows that the negative log-posterior is exactly a quadratic Tikhonov functional.

---

**Step 1: Write the posterior density up to a constant.**

The observation model is

[
Y=AX+E,
\qquad
E\sim \mathcal N(0,\Sigma),
]

so the likelihood is

[
\pi_{Y\mid X}(y\mid x)
\propto
\exp\left(
-\frac12 |y-Ax|_\Sigma^2
\right).
]

The prior is

[
X\sim \mathcal N(\overline{x},C),
]

so

[
\pi_X(x)
\propto
\exp\left(
-\frac12 |x-\overline{x}|_C^2
\right).
]

By Bayes’ formula,

[
\pi_{X\mid Y}(x\mid y)
\propto
\pi_{Y\mid X}(y\mid x)\pi_X(x).
]

Therefore

[
\pi_{X\mid Y}(x\mid y)
\propto
\exp\left(
-\frac12|y-Ax|_\Sigma^2
-\frac12|x-\overline{x}|_C^2
\right).
]

Hence maximising the posterior density is equivalent to minimising

[
J(x)
:=
\frac12|y-Ax|_\Sigma^2
+
\frac12|x-\overline{x}|_C^2.
]

Thus

[
x_{\mathrm{MAP}}
================

\operatorname*{argmin}*{x\in\mathbb R^s}
\left[
\frac12|y-Ax|*\Sigma^2
+
\frac12|x-\overline{x}|_C^2
\right].
]

This proves the MAP/Tikhonov form.

---

**Step 2: Compute the minimiser.**

Since (C^{-1}) and (\Sigma^{-1}) are positive definite, (J) is strictly convex. Hence it has a unique minimiser, characterised by

[
\nabla J(x)=0.
]

Compute:

[
\nabla J(x)
===========

-A^\top\Sigma^{-1}(y-Ax)
+
C^{-1}(x-\overline{x}).
]

Equivalently,

[
A^\top\Sigma^{-1}(Ax-y)
+
C^{-1}(x-\overline{x})
======================

0.

]

Rearranging gives the normal equation

[
\left(A^\top\Sigma^{-1}A+C^{-1}\right)x
=======================================

A^\top\Sigma^{-1}y+C^{-1}\overline{x}.
]

So

[
x_{\mathrm{MAP}}
================

\left(A^\top\Sigma^{-1}A+C^{-1}\right)^{-1}
\left(A^\top\Sigma^{-1}y+C^{-1}\overline{x}\right).
]

This is already the Tikhonov-regularised solution: data misfit plus quadratic prior penalty.

---

**Step 3: Show that this equals the conditional mean.**

The theorem gives

[
x_{\mathrm{CM}}
===============

\overline{x}
+
CA^\top(\Sigma+ACA^\top)^{-1}(y-A\overline{x}).
]

Let

[
C_Y:=\Sigma+ACA^\top,
\qquad
r:=y-A\overline{x}.
]

Then

[
x_{\mathrm{CM}}
===============

\overline{x}+CA^\top C_Y^{-1}r.
]

We check that this satisfies the MAP normal equation.

First,

[
x_{\mathrm{CM}}-\overline{x}
============================

CA^\top C_Y^{-1}r,
]

so

[
C^{-1}(x_{\mathrm{CM}}-\overline{x})
====================================

A^\top C_Y^{-1}r.
]

Next,

[
Ax_{\mathrm{CM}}-y
==================

A\overline{x}+ACA^\top C_Y^{-1}r-y.
]

Since (r=y-A\overline{x}), this becomes

[
Ax_{\mathrm{CM}}-y
==================

-r+ACA^\top C_Y^{-1}r.
]

Using

[
C_Y=\Sigma+ACA^\top,
]

we get

[
I-ACA^\top C_Y^{-1}
===================

\Sigma C_Y^{-1}.
]

Hence

[
Ax_{\mathrm{CM}}-y
==================

-\Sigma C_Y^{-1}r.
]

Therefore

[
A^\top\Sigma^{-1}(Ax_{\mathrm{CM}}-y)
=====================================

-A^\top C_Y^{-1}r.
]

Combining the two pieces,

[
A^\top\Sigma^{-1}(Ax_{\mathrm{CM}}-y)
+
C^{-1}(x_{\mathrm{CM}}-\overline{x})
====================================

-A^\top C_Y^{-1}r
+
A^\top C_Y^{-1}r
================

0.

]

Thus (x_{\mathrm{CM}}) satisfies the unique optimality equation for (J). Therefore

[
\boxed{
x_{\mathrm{MAP}}=x_{\mathrm{CM}}.
}
]

---

**Step 4: Why this is also the mode of the Gaussian posterior.**

From Theorem 5.3.1,

[
X\mid Y=y
\sim
\mathcal N(x_{\mathrm{CM}},C_{X\mid y}).
]

The density of this Gaussian is

[
\pi_{X\mid Y}(x\mid y)
\propto
\exp\left(
-\frac12
|x-x_{\mathrm{CM}}|*{C*{X\mid y}}^2
\right).
]

This is maximised exactly when

[
x=x_{\mathrm{CM}}.
]

So independently of the Tikhonov calculation,

[
x_{\mathrm{MAP}}=x_{\mathrm{CM}}.
]

The Tikhonov calculation identifies the same point as the minimiser of the data-misfit plus prior-penalty functional.

---

**Conclusion.**

In the finite-dimensional linear Gaussian case,

[
\boxed{
x_{\mathrm{MAP}}
=
x_{\mathrm{CM}}
=
\operatorname*{argmin}*{x\in\mathbb R^s}
\left[
\frac12|y-Ax|*\Sigma^2
+
\frac12|x-\overline{x}|_C^2
\right].
}
]

The equality is special to Gaussian posteriors. For non-Gaussian or nonlinear inverse problems, the posterior may be skewed or multimodal, and then the conditional mean and MAP point usually differ.


</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">($x\_{\mathrm{MAP}} = x\_{\mathrm{CM}}$)</span></p>

In the expression above

$$x_{\mathrm{MAP}} = \operatorname{argmax}_{x \in \mathbb{R}^s} \pi_{X|Y}(x|y) = \operatorname{argmin}_{x \in H} \frac{1}{2} \lVert y - Ax \rVert_\Sigma^2 + \frac{1}{2} \lVert x - \overline{x} \rVert_C^2 = x_{\mathrm{CM}}$$

The terms could be interpreted as optimizing two coupled objectives:
* **Match the observed data:** make $Ax$ close to $y$, measured in the noise-weighted norm $\|\cdot\|\_\Sigma$.
* **Remain plausible under the prior:** keep $x$ close to the prior mean $\overline{x}$, measured in the prior-weighted norm $\|\cdot\|\_C$.

The functional

$$\frac12 |y-Ax|_\Sigma^2 + \frac12 |x-\overline{x}|_C^2$$

has exactly the structure you describe:

$$\boxed{ \text{posterior objective} = \text{data misfit} +\text{prior penalty}. }$$

The first term

$$|y-Ax|_\Sigma^2 = (y-Ax)^\top \Sigma^{-1}(y-Ax)$$

means:

$$\text{choose } x \text{ so that the predicted observation } Ax \text{ is close to the actual observation } y.$$

So yes: **match the target/data**.

The second term

$$|x-\overline{x}|_C^2 = (x-\overline{x})^\top C^{-1}(x-\overline{x})$$

means:

$$\text{do not move too far from the prior mean } \overline{x}.$$

But more precisely, it means: **do not move too far in directions that the prior regards as unlikely**. The covariance $C$ matters. If $C$ has large variance in some direction, deviations in that direction are penalized weakly. If $C$ has small variance in some direction, deviations are penalized strongly.

So the final interpretation is:

$$\boxed{ x_{\mathrm{MAP}}=x_{\mathrm{CM}} \text{ is the point that best compromises between explaining the data and remaining prior-plausible.}}$$

This equality is special to the linear Gaussian case. In nonlinear or non-Gaussian problems, the MAP still minimises a similar posterior objective, but it usually no longer equals the conditional mean.

</div>

#### The Laplace Approximation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Preconditioning Technique: Laplace Approximation)</span></p>

* As in numerical optimisation for general, nonlinear deterministic problems, and in particular for the solution of Tikhonov-regularised, nonlinear inverse problems, such as (2.5.3), a powerful way to accelerate numerical methods is a change of metric, also referred to as **preconditioning**. 
* For simplicity, let us restrict ourselves to a finite dimensional parameter space $H = \mathbb{R}^s$ and to additive Gaussian noise $E$ independent of $X$ and distributed according to $\mathcal{N}(0, \Sigma)$ again.

A simple and popular preconditioning technique that is easy to understand and to apply to general Bayesian inverse problems is the so-called **Laplace approximation** of the posterior distribution $\mu\_{X\mid y}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.3.2</span><span class="math-callout__name">(Laplace Approximation)</span></p>

*(Laplace, 1774).* Suppose the forward operator $\Phi : \mathbb{R}^s \to \mathbb{R}^m$ and the prior distribution $\mu\_X(\mathrm{d}x) = \pi\_X(x) \, \mathrm{d}x$ are such that $\Phi, \pi\_X \in C^2(S\_X)$, for $S\_X := \lbrace x \in \mathbb{R}^s : \pi\_X(x) > 0 \rbrace$. Let $\Psi : S\_X \to \mathbb{R}$ be given by $\Psi(x) := \frac{1}{2} \lVert y - \Phi(x) \rVert\_\Sigma^2 - \log \pi\_X(x)$ and assume that $\Psi$ has a unique minimiser $x\_{\mathrm{MAP}} \in S\_X$ satisfying

$$\nabla \Psi(x_{\mathrm{MAP}}) = 0 \quad \text{and} \quad \nabla^2 \Psi(x_{\mathrm{MAP}}) \text{ is SPD.}$$

Then, the **Laplace approximation** of $\mu_{X\mid y}$ is given by the Gaussian measure

$$\mathcal{L}_{\mu_{X|y}} := \mathcal{N}(x_{\mathrm{MAP}}, C_{\mathrm{MAP}}) \quad \text{with} \quad C_{\mathrm{MAP}}^{-1} := \nabla^2 \Psi(x_{\mathrm{MAP}}). \tag{5.3.4}$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/laplace_construction.png' | relative_url }}" alt="Left: the negative log-posterior, an asymmetric valley-shaped curve, together with its quadratic Taylor expansion at the minimiser, a dashed parabola that hugs the curve near the marked MAP point but lies above it on the left flank and below it on the right flank. Right: the corresponding normalised posterior density, visibly skewed with a heavy left tail, against the symmetric Laplace Gaussian; dotted vertical lines mark the MAP point at 0.86 and the conditional mean at 0.54, which do not coincide." loading="lazy">
</figure>

*Definition 5.3.2 visualised (1D model with prior $\mathcal{N}(0, 1)$, forward map $\Phi(x) = e^x$, data $y = e$, noise variance $1$). Left: the negative log-posterior $\Psi$ and its second-order Taylor expansion at the minimiser — value, slope and curvature match at $x\_{\mathrm{MAP}}$, but the parabola lies* above *$\Psi$ on the left flank and* below *it on the right. Right: exponentiating turns the parabola into the Gaussian (5.3.4): a good fit near the mode, but symmetric where the posterior is skewed — where the parabola overshoots $\Psi$, the Gaussian's tail is too light (left), and vice versa (right). Consequently $x\_{\mathrm{CM}} \approx 0.54$ differs markedly from $x\_{\mathrm{MAP}} \approx 0.86$: for nonlinear $\Phi$ the identity $x\_{\mathrm{CM}} = x\_{\mathrm{MAP}}$ of the linear Gaussian case fails. This mismatch is exactly what shrinks in the small-noise limit — see the figure after Theorem 5.3.3.*

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/laplace-approximation.png' | relative_url }}" alt="A single panel showing three densities of a scalar parameter mu between 0 and 12: a broad dashed prior density peaking gently near 6, a thick black posterior density peaking sharply near 3 with a heavy right tail, and a thin red Laplace approximation that coincides with the posterior at the peak but is symmetric, lying above the posterior to the left of the mode and dropping too fast on the right, missing the heavy right tail." loading="lazy">
</figure>

*The same story in a second example (a positive scalar parameter $\mu$ with a broad prior, dashed): the Laplace approximation (red) reproduces the posterior's (black) mode and curvature perfectly but is symmetric by construction — here it overweights the region left of the mode and cuts off the heavy* right *tail that the true posterior inherits from the skewed likelihood. Which side the error lands on depends entirely on the sign of the skewness; what is generic is that mode and curvature are exact while tails and mean are not.*

Finding the minimiser $x_{\mathrm{MAP}}$ can be achieved with classical unconstrained, nonlinear minimisation methods, e.g. quasi-Newton methods with low rank updates (such as SR1 or BFGS) and a globalisation strategy (such as a line search or trust region method).

A common and desirable situation in Bayesian inverse problems is concentration of the posterior around the true parameter, especially in the small noise or large data limit. Optimisation works well in that context and can thus be used to efficiently construct the Laplace approximation as a suitable reference measure to remove the degeneracy and to precondition sampling or Markov chain Monte Carlo algorithms.

To see why the Laplace approximation is useful in this limit, let us first consider a scaled version $n\Psi_n$ of the posterior log-likelihood $\Psi$ with

$$\Psi_n(x) := \frac{1}{2} \lVert y - \Phi(x) \rVert_\Sigma^2 - \frac{1}{n} \log \pi_X(x), \tag{5.3.5}$$

e.g. if the measurement error $E_n$ is assumed to decrease as $n$ increases, such that $E_n \sim \mathcal{N}(0, \frac{1}{n}\Sigma)$. The scaled posterior measure is then

$$\nu_n(\mathrm{d}x) = \frac{1}{Z_n} \exp\left( -n\left(\frac{1}{2} \lVert y - \Phi(x) \rVert_\Sigma^2\right) \right) \mu_x(\mathrm{d}x), \quad Z_n := \int_{\mathbb{R}^s} \exp\left( -n\left(\frac{1}{2} \lVert y - \Phi(x) \rVert_\Sigma^2\right) \right) \mu_x(\mathrm{d}x).$$

We can see that the weight function concentrates more and more around the MAP point $x\_{\mathrm{MAP},n}$ as $n \to \infty$. It is a classical result [Wong, 2001] that integrals with respect to such a measure $\nu\_n$ can be well approximated via integrals with respect to the Laplace approximation, but even a stronger convergence result can be proved. We will only state the result informally and refer to the original paper for a complete statement and for the proof.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.3.3</span><span class="math-callout__name">(Convergence of Laplace Approximation)</span></p>

*(Schillings et al, 2020).* Suppose $\Psi_n \in C^3(S_X)$ and satisfies further technical conditions. Suppose further that $\lim_{n \to \infty} x_{MAP,n} \in S_X$ and $\lim_{n \to \infty} \nabla^2 \Psi_n(x_{MAP,n})$ exist. Then

$$D_{\mathrm{H}}(\nu_n, \mathcal{L}_{\nu_n}) \le Cn^{-1/2}.$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/laplace_concentration.png' | relative_url }}" alt="Four panels. The first three compare the true posterior density (solid) with its Laplace approximation (dashed) for noise scalings n = 1, 10 and 100: at n = 1 the posterior is visibly skewed and the Gaussian misses it, at n = 10 the two curves are close, at n = 100 they coincide. The fourth panel shows the Hellinger distance between posterior and Laplace approximation against n on a log-log scale, following a straight reference line of slope minus one half." loading="lazy">
</figure>

*Theorem 5.3.3 in action for the nonlinear forward map $\Phi(x) = e^x$ with prior $\mathcal{N}(0, 1)$, data $y = e$ and noise variance $1/n$: for $n = 1$ the posterior is visibly skewed and the Gaussian fit at the MAP is poor; by $n = 100$ the two densities are indistinguishable. The Hellinger distance (right, computed numerically) follows the predicted rate $n^{-1/2}$ essentially exactly (measured slope $\to 0.50$).*

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.3.4</span></p>

The scaled posterior log-likelihood in (5.3.5) is the same as the generalised Tikhonov functional (cf. Sect. 2.5)

$$\Psi_{\alpha,\delta}(x) := \frac{1}{2} \lVert y - \Phi(x) \rVert_\Sigma^2 - \alpha \log \pi_X(x) \tag{5.3.6}$$

with penalisation functional $\log \pi_X : \mathbb{R}^s \to \mathbb{R}$, with noise level $\delta = \frac{1}{n}$ and with regularisation parameter $\alpha = \frac{1}{n}$. In the Bayesian setting, adding the regularisation parameter corresponds to "flattening" the prior distribution $\pi_X$ to $\pi_X^{1/n}$ as $n \to \infty$, thus reducing its influence. Note also that the choice $\alpha = 1/n$ for a Gaussian prior in (5.3.6) and for $\delta = 1/n$ leads to a convergent regularisation method in the sense of Sect. 2.4, since $\delta/\sqrt{\alpha} \to 0$ as $\delta \to 0$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(What the Laplace approximation actually does — and when to trust it)</span></p>

The definition looks technical, but the underlying move is a single idea: **replace the negative log-posterior by its second-order Taylor expansion at the mode**. Writing $\pi\_{X\mid Y}(x\mid y) \propto e^{-\Psi(x)}$ and expanding

$$\Psi(x) \approx \Psi(x_{\mathrm{MAP}}) + \underbrace{\nabla \Psi(x_{\mathrm{MAP}})^\top (x - x_{\mathrm{MAP}})}_{= 0 \text{ at the minimiser}} + \frac{1}{2} (x - x_{\mathrm{MAP}})^\top \nabla^2 \Psi(x_{\mathrm{MAP}}) (x - x_{\mathrm{MAP}}),$$

the exponential of the right-hand side is exactly an unnormalised Gaussian density with mean $x\_{\mathrm{MAP}}$ and covariance $(\nabla^2 \Psi(x\_{\mathrm{MAP}}))^{-1}$ — which is (5.3.4). Three consequences are worth internalising:

* **Exactness in the linear Gaussian case.** If $\Phi(x) = Ax$ and the prior is Gaussian, then $\Psi$ is *globally* quadratic, the Taylor expansion has no remainder, and the Laplace approximation coincides with the true posterior of Theorem 5.3.1 (in particular $x\_{\mathrm{CM}} = x\_{\mathrm{MAP}}$). The Laplace approximation therefore measures, in a precise sense, "how far from linear-Gaussian" the problem is.
* **Why the small-noise limit helps.** Theorem 5.3.3 is a Bernstein-von-Mises-type statement: as $n \to \infty$ the posterior mass concentrates in an $O(n^{-1/2})$-neighbourhood of $x\_{\mathrm{MAP},n}$, and on such shrinking neighbourhoods any smooth $\Psi\_n$ is dominated by its quadratic part — the cubic Taylor remainder is what produces the rate $n^{-1/2}$ (hence the $C^3$ assumption). Concentration, usually a *curse* for sampling methods (see Section 5.5.3), is exactly what makes the Laplace approximation *better*.
* **Its role in this course is a preconditioner, not a final answer.** The Laplace approximation is cheap (one optimisation run plus one Hessian), and it is a Gaussian — so we can sample from it, evaluate its density, and use it as a reference measure: as importance distribution (Section 5.5.3), as proposal distribution for MCMC (Section 5.6), or as the update rule in the extended Kalman filter (Section 5.8). The pattern "solve an optimisation problem, then correct the Gaussian ansatz by sampling" recurs throughout the rest of the chapter.

</div>

There is more rigorous mathematical theory on the topic of posterior consistency [Ghosal & van der Vaart, 2017], but we will not discuss this any further. The explicit form of the posterior measure in the linear Gaussian case and the Laplace approximation play a central role in filtering, in the (extended) Kalman filter, and we will come back to this point in Section 5.8.

### 5.4 High-Dimensional Quadrature

Even though mathematically speaking the solution to a Bayesian inverse problem is the posterior distribution $\mu_{X\mid y}$, it is of little practical value (especially in high dimensions). As highlighted already, the central task in Bayesian inference is the computation of expectations of certain functionals of the parameter with respect to the posterior, so called **statistics** or **quantities of interest**.

Care is required, when designing quadrature algorithms in high dimensions; the computational cost of simple tensor product rules of standard 1D quadrature rules explodes as the dimension $s \to \infty$. The workhorses in high dimensions are sampling based methods that do not suffer from this so-called **"curse of dimensionality"**, and in particular Monte Carlo type methods.

#### 5.4.1 Monte Carlo Quadrature

Consider again the general setting of a probability space $(\Omega, \mathcal{A}, \mathbb{P})$ and a RV $X : \Omega \to V$ mapping to a separable Banach space $V$ with measure $\mu_X$, for the moment assumed to be available explicitly.

For any measurable $F : V \to \mathbb{R}$ (for simplicity one-dimensional), given $N$ realisations $x^{(1)}, \ldots, x^{(N)}$ of **independent** RVs $X^{(i)} \sim \mu_X$, a practical method to compute the high-dimensional integral

$$\mathbb{E}[F(X)] = \int_V F(x) \, \mathrm{d}\mu(x), \quad \text{with} \quad \mathbb{E} := \mathbb{E}_{\mu_X}$$

is the **Monte Carlo (MC) method**

$$\frac{1}{N} \sum_{i=1}^{N} F(x^{(i)}) \approx \mathbb{E}[F(X)]$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.4.1</span><span class="math-callout__name">(Monte Carlo Convergence)</span></p>

Let $X \in L^2(\Omega; V)$ and iid $X^{(i)} \sim \mu_X$, $i \in \mathbb{N}$. Then for the associated estimator $\widehat{F(X)}\_N := \frac{1}{N} \sum_{i=1}^{N} F(X^{(i)})$:

$$\widehat{F(X)}_N \xrightarrow{\mathbb{P}\text{-a.s.}}{N \to \infty} \mathbb{E}[F(X)], \tag{5.4.1}$$

$$\sqrt{N}\Big(\widehat{F(X)}_N - \mathbb{E}[F(X)]\Big) \xrightarrow{d}{N \to \infty} \mathcal{N}\Big(0, \mathbb{V}(F(X))\Big), \tag{5.4.2}$$

$$\mathbb{E}\left[\left|\widehat{F(X)}_N - \mathbb{E}[F(X)]\right|^2\right] = \frac{\mathbb{V}(F(X))}{N}. \tag{5.4.3}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 5.4.1</summary>

The convergence results (5.4.1) and (5.4.2) follow directly from the Law of Large Numbers and the Central Limit Theorem. To see (5.4.3), note that due to the independence of the $X^{(i)}$,

$$\begin{aligned}
\mathbb{E}\left[\left|\frac{1}{N}\sum_{i=1}^N F(X^{(i)}) - \mathbb{E}[F(X)]\right|^2\right]
&= \frac{1}{N^2} \mathbb{E}\left[\left(\sum_{i=1}^N \Big(F(X^{(i)}) - \mathbb{E}[F(X)]\Big)\right)^2\right] \\
&= \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N \mathbb{E}\Big[\big(F(X^{(i)}) - \mathbb{E}[F(X)]\big)\big(F(X^{(j)}) - \mathbb{E}[F(X)]\big)\Big] \\
&= \frac{1}{N^2} \sum_{i=1}^N \mathbb{E}\Big[\big(F(X^{(i)}) - \mathbb{E}[F(X)]\big)^2\Big] \\
&\qquad + \frac{1}{N^2} \sum_{i \neq j} \underbrace{\mathbb{E}\Big[F(X^{(i)}) - \mathbb{E}[F(X)]\Big]}_{=0} \, \underbrace{\mathbb{E}\Big[F(X^{(j)}) - \mathbb{E}[F(X)]\Big]}_{=0} \\
&= \frac{1}{N^2} \sum_{i=1}^N \mathrm{Var}\big(F(X^{(i)})\big) = \frac{1}{N} \mathrm{Var}(F(X)).
\end{aligned}$$

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Why Monte Carlo "beats" the curse of dimensionality — and what it costs)</span></p>

The remarkable feature of (5.4.3) is what it does *not* contain: the dimension $s$ of the integration domain. A tensor-product quadrature rule with $n$ points per direction and order $k$ achieves error $O(n^{-k})$ with $N = n^s$ points, i.e. error $O(N^{-k/s})$ — the rate collapses as $s$ grows. The MC rate $N^{-1/2}$ (in root-mean-square) is dimension-independent: the same estimator, the same analysis, whether $s = 3$ or $s = 10^6$ or $V$ is a function space.

The price is twofold, and both points drive the rest of this chapter:

* **The rate is slow.** $N^{-1/2}$ means one extra digit of accuracy costs a factor $100$ in samples. All improvements below (MLMC, QMC) attack precisely this: MLMC reduces the *constant* $\mathbb{V}$ by shifting variance to cheap coarse levels, QMC improves the *rate* towards $N^{-1}$ by giving up on independent random points.
* **The constant is a variance.** The error is governed by $\mathbb{V}(F(X))$, not by smoothness of $F$. This is robust (no derivatives needed) but also means MC cannot exploit smoothness — which is exactly the information QMC uses (Assumption 5.4.8).

</div>

As discussed in Section 5.2, if $V$ is infinite dimensional or $F$ is given as $F(X) = \Psi(\mathcal{G}(X))$ for some operator $\mathcal{G} : V \to W$ and an infinite dimensional latent space $W$ with $\Psi : W \to \mathbb{R}$, it is necessary in practice to discretise the problem. The operator $\mathcal{G}$ could be the Radon transform and $\Psi$ the restriction operator to the measurement along a single line, or $\mathcal{G}$ could be the solution operator for the elliptic PDE that takes the diffusion coefficient $a$ to the solution $u$ and $\Psi$ could be a point evaluation of $u$ at some point in the domain. The general setting is then that $X\_s : \Omega \to V\_s := \mathbb{R}^s$ and $F\_h : V\_s \to \mathbb{R}$ are approximations of $X$ and $F$ with $\mu\_{X\_s} \ll \mu\_X$, parametrised by some parameters $h > 0$ and $s \in \mathbb{N}$, e.g. the FE mesh width and the truncation dimension for the Karhunen-Loève expansion. For simplicity, let us consider only the case $X\_s = X$ and denote by $Q := F(X)$ and $Q\_h := F\_h(X)$ the quantity of interest and its approximation.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.4.2</span><span class="math-callout__name">(Bias-Variance Decomposition)</span></p>

Let $X \in L^2(\Omega; V)$ and iid $X^{(i)} \sim \mu_X$, $i \in \mathbb{N}$. Then

$$\mathbb{E}\left[\left|\widehat{Q}_{h,N}^{\mathrm{MC}} - \mathbb{E}[Q]\right|^2\right] = \big(\mathbb{E}[Q_h - Q]\big)^2 + \frac{\mathbb{V}(Q_h)}{N} \quad \text{with} \quad \widehat{Q}_{h,N}^{\mathrm{MC}} := \frac{1}{N} \sum_{i=1}^{N} F_h(X^{(i)}). \tag{5.4.4}$$

In fact, we also have

$$\sqrt{N}\Big(\widehat{Q}_{h,N}^{\mathrm{MC}} - \mathbb{E}[Q]\Big) \xrightarrow{d}{N \to \infty} \mathcal{N}\Big(\mathbb{E}[Q_h - Q], \mathbb{V}(Q_h)\Big). \tag{5.4.5}$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.4.3</span><span class="math-callout__name">($\varepsilon$-Cost)</span></p>

The $\varepsilon$**-cost** $\mathcal{C}\_\varepsilon(\widehat{Q})$ for any estimator $\widehat{Q}$ of $\mathbb{E}[Q]$ is defined to be the total number of arithmetic operations to achieve

$$\lVert \widehat{Q} - \mathbb{E}[Q] \rVert_{L^2(\Omega)}^2 = \mathbb{E}\left[|\widehat{Q} - \mathbb{E}[Q]|^2\right] \le \varepsilon^2.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.4.4</span><span class="math-callout__name">(Complexity Theorem for MC)</span></p>

Suppose there are constants $\alpha, \gamma > 0$, such that

$$|\mathbb{E}[Q_h - Q]| \le Ch^\alpha, \tag{5.4.6}$$

$$\mathcal{C}(Q_h) \le Ch^{-\gamma}, \tag{5.4.7}$$

as $h \to 0$, where $\mathcal{C}(Y)$ denotes the cost to compute one sample of a RV $Y$. Then, for any $\varepsilon > 0$, there exists $h = h(\varepsilon) > 0$ and $N_{\mathrm{MC}} := N_{\mathrm{MC}}(\varepsilon) \in \mathbb{N}$ such that

$$\mathcal{C}_\varepsilon(\widehat{Q}_{h, N_{\mathrm{MC}}}^{\mathrm{MC}}) \le C\varepsilon^{-2-\gamma/\alpha}.$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Where $\varepsilon^{-2-\gamma/\alpha}$ comes from — the balancing argument)</span></p>

The exponent in Theorem 5.4.4 is not mysterious; it falls out of a two-line budgeting argument that is the template for *every* complexity result in this section. By the bias-variance decomposition (5.4.4), the MSE has two contributions, and we make each smaller than $\varepsilon^2/2$:

* **Bias budget.** $\left(\mathbb{E}[Q\_h - Q]\right)^2 \le C h^{2\alpha} \overset{!}{\le} \varepsilon^2/2$ forces the mesh size $h \sim \varepsilon^{1/\alpha}$. By (5.4.7), one sample at this resolution then costs $\mathcal{C}(Q\_h) \sim h^{-\gamma} \sim \varepsilon^{-\gamma/\alpha}$.
* **Variance budget.** $\mathbb{V}(Q\_h)/N \overset{!}{\le} \varepsilon^2/2$ forces $N \sim \varepsilon^{-2}$, since $\mathbb{V}(Q\_h)$ stays bounded as $h \to 0$.

Total cost: $N \times (\text{cost per sample}) \sim \varepsilon^{-2} \cdot \varepsilon^{-\gamma/\alpha}$. The two factors expose the two independent inefficiencies of plain MC: the $\varepsilon^{-2}$ from the slow sampling rate, and the $\varepsilon^{-\gamma/\alpha}$ from paying full PDE-solve cost for *every single* sample. MLMC (next subsection) attacks the second factor, QMC the first.

</div>

Using (5.4.5), a similar result follows also for the $\varepsilon$-cost defined with respect to convergence in probability. We can deduce the following corollary.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.4.5</span><span class="math-callout__name">(MC Complexity for Elliptic PDE)</span></p>

Let us consider the elliptic PDE in Section 5.2.1 under the assumptions of Theorem 5.2.5 with $p = 2$ and with $a_{\min}^{-1} \in L^2(\Omega)$. Let $Q := Bu$ and $Q_h := Bu_h$. Suppose the FE solution is computed with an optimal multigrid method, such that $\mathcal{C}(Q_h) \le Ch^{-d}$. Then, for any $\varepsilon > 0$, there exists $h = h(\varepsilon) > 0$ and $N_{\mathrm{MC}} := N_{\mathrm{MC}}(\varepsilon) \in \mathbb{N}$ such that

$$\mathcal{C}_\varepsilon(\widehat{Q}_{h, N_{\mathrm{MC}}}^{\mathrm{MC}}) \le C\varepsilon^{-2-d/2}.$$

</div>

#### 5.4.2 Multilevel Monte Carlo

The key idea in multilevel Monte Carlo is to use samples of $Q$ on a hierarchy of different **discretisation levels**, i.e., for different values $h_0 > h_1 > \ldots h_L =: h > 0$ of the discretization parameter with $L \in \mathbb{N}$, and to decompose

$$\mathbb{E}[Q_h] = \mathbb{E}[Q_{h_0}] + \sum_{\ell=1}^{L} \mathbb{E}[Q_{h_\ell} - Q_{h_{\ell-1}}] =: \sum_{\ell=0}^{L} \mathbb{E}[Y_\ell]. \tag{5.4.8}$$

For simplicity, we will choose

$$h_{\ell-1} = r \, h_\ell, \quad \ell = 1, \ldots, L, \quad \text{for some } r \in \mathbb{N} \setminus \lbrace 1 \rbrace \text{ and } h_0 > 0, \tag{5.4.9}$$

i.e. uniform grid refinement for the elliptic PDE. With iid $X_\ell^{(i)} \sim \mu_X$, $\ell, i \in \mathbb{N}$, we define the **multilevel Monte Carlo (MLMC)** estimator for $\mathbb{E}[Q]$ as

$$\widehat{Q}_{L, \lbrace N_\ell \rbrace}^{\mathrm{ML}} := \sum_{\ell=0}^{L} \widehat{Y}_{\ell, N_\ell}^{\mathrm{MC}} = \frac{1}{N_0} \sum_{i=1}^{N_0} F_{h_0}(X_0^{(i)}) + \sum_{\ell=1}^{L} \frac{1}{N_\ell} \sum_{i=1}^{N_\ell} \Big( F_{h_\ell}(X_\ell^{(i)}) - F_{h_{\ell-1}}(X_\ell^{(i)}) \Big) \tag{5.4.10}$$

As in Lemma 5.4.2 for standard MC, the following bias-variance decomposition is a simple consequence of (5.4.8) and the independence of the RVs $X_\ell^{(i)}$:

$$\mathbb{E}\left[\Big(\widehat{Q}_{L, \lbrace N_\ell \rbrace}^{\mathrm{ML}} - \mathbb{E}[Q]\Big)^2\right] = \big(\mathbb{E}[Q_h - Q]\big)^2 + \sum_{\ell=0}^{L} \frac{\mathbb{V}(Y_\ell)}{N_\ell}. \tag{5.4.11}$$

If $\lVert Q_h - Q \rVert_{L^2(\Omega)} \to 0$ as $h \to 0$, i.e., if the RV $Q_h$ converges strongly (samplewise) to $Q$, then $\mathbb{V}(Y_\ell) \to 0$ as $\ell \to \infty$, leading to a huge variance reduction in the MLMC estimator compared to standard MC. In particular, a significantly smaller number $N_L \ll N_{\mathrm{MC}}$ of expensive samples on the finest level $L$ with $h = h_L$ are sufficient to achieve a prescribed tolerance $\varepsilon$ and the slightly larger number of samples $N_0 > N_{\mathrm{MC}}$ necessary on the coarsest level 0 are significantly cheaper than samples on level $L$ under assumption (5.4.7).

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/mlmc_variance_decay.png' | relative_url }}" alt="Two panels of log2 quantities against the level index 0 to 4. Left: the variance of the quantity of interest Q at each level is a flat horizontal line, while the variance of the level correction Y drops steeply along a straight line of slope minus 4. Right: the absolute mean of Q per level is flat while the absolute mean of Y drops along a line of slope minus 2." loading="lazy">
</figure>

*The two measurements that decide everything about MLMC, computed for the 1D elliptic problem with the KL coefficient of Example 4.5.17 ($s = 16$, QoI $u\_h(0.5)$, $4000$ coupled samples per level, $h\_\ell = 2^{-\ell}/8$): the variance of $Q\_{h\_\ell}$ is flat across levels — every level is equally hard for plain MC — while the variance of the correction $Y\_\ell$ drops by a factor $2^4$ per level and its mean by $2^2$, i.e. $\beta = 4$ and $\alpha = 2$ in the assumptions (M1)-(M3) of Theorem 5.4.6. Since $\gamma = d = 1 < \beta$, this is the best-case regime: almost all samples can live on the cheap coarse grids.*

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.4.6</span><span class="math-callout__name">(MLMC Complexity)</span></p>

Let $\varepsilon < e^{-1}$ and let $\alpha, \beta, \gamma > 0$ be such that $\alpha \ge \frac{1}{2}\min\lbrace \beta, \gamma \rbrace$ and such that for all $\ell \in \mathbb{N}_0$

**(M1)** $\|\mathbb{E}[Q_{h_\ell}] - \mathbb{E}[Q]\| \le Ch_\ell^\alpha$, &nbsp; **(M2)** $\mathrm{Var}[Y_\ell] \le Ch_\ell^\beta$, &nbsp; **(M3)** $\mathcal{C}(Y_\ell) \le Ch_\ell^{-\gamma}$.

Then there are $L \in \mathbb{N}$ and $\lbrace N_\ell \rbrace_{\ell=0}^L \subset \mathbb{N}$ such that

$$\mathcal{C}_\varepsilon\Big(\widehat{Q}_{L, \lbrace N_\ell \rbrace}^{ML}\Big) \le C \begin{cases} \varepsilon^{-2}, & \text{if } \beta > \gamma, \\ \varepsilon^{-2} |\log \varepsilon|^2, & \text{if } \beta = \gamma, \\ \varepsilon^{-2-(\gamma-\beta)/\alpha}, & \text{if } \beta < \gamma. \end{cases}$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Reading the MLMC complexity theorem)</span></p>

The telescoping sum (5.4.8) looks like an accounting trick, but it performs a genuine *decoupling of accuracy and cost*. In standard MC, every sample must be computed at the finest resolution $h$, so accuracy (small bias) and cost per sample are welded together. MLMC breaks the weld: the expectation is anchored by many cheap coarse samples ($\mathbb{E}[Q\_{h\_0}]$), and the fine levels only need to estimate the small *corrections* $Y\_\ell = Q\_{h\_\ell} - Q\_{h\_{\ell-1}}$.

The crucial point is that the correction $Y\_\ell$ uses the **same sample** $X\_\ell^{(i)}$ for both resolutions in (5.4.10). Because $Q\_h \to Q$ samplewise, the two evaluations are strongly correlated and their difference has tiny variance: $\mathbb{V}(Y\_\ell) \le C h\_\ell^\beta \to 0$. Few samples suffice exactly where samples are expensive. (Across levels, on the other hand, the estimators $\widehat{Y}\_{\ell,N\_\ell}^{\mathrm{MC}}$ use *independent* samples, which is what makes the variances in (5.4.11) add.)

The three cases in Theorem 5.4.6 answer one question: **which end of the level hierarchy dominates the total cost $\sum\_\ell N\_\ell \, \mathcal{C}(Y\_\ell)$?** With the optimal choice $N\_\ell \propto \sqrt{\mathbb{V}(Y\_\ell)/\mathcal{C}(Y\_\ell)}$ (a Lagrange-multiplier computation: minimise total cost subject to a fixed variance budget), the per-level cost scales like $h\_\ell^{(\gamma - \beta)/2}$:

* $\beta > \gamma$ — variance decays faster than cost grows: the *coarsest* level dominates, and the total cost $\varepsilon^{-2}$ is that of a plain MC estimator whose samples cost $O(1)$. This is the best possible regime for any sampling method with rate $N^{-1/2}$.
* $\beta = \gamma$ — all levels contribute equally; the $\|\log \varepsilon\|^2$ collects the $\sim \|\log \varepsilon\|$ levels.
* $\beta < \gamma$ — cost grows faster than variance decays: the *finest* level dominates and the exponent degrades to $2 + (\gamma - \beta)/\alpha$ — still strictly better than MC's $2 + \gamma/\alpha$.

For the elliptic model problem (Corollary 5.4.7) with piecewise linear FE and multigrid: $\alpha = 2$, $\beta = 4$ (variance of functional corrections decays at twice the bias rate) and $\gamma = d$, so for $d \le 3$ we are in the first two regimes — MLMC removes the *entire* PDE-solve penalty from the complexity.

</div>

A sufficient condition to achieve this asymptotic $\varepsilon$-cost is that

$$N_\ell \propto \left( r^{\frac{\beta+\gamma}{2}} \right)^{-\ell} \tag{5.4.12}$$

with $L$ and $N\_0$ chosen such that both of the two terms on the right hand side of (5.4.11) are equal to $\varepsilon^2/2$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.4.7</span><span class="math-callout__name">(MLMC Complexity for Elliptic PDE)</span></p>

Consider again the elliptic PDE in Section 5.2.1 under the assumptions of Theorem 5.2.5 with $d = 1, 2, 3$, $p = 2$ and $a_{\min}^{-1} \in L^2(\Omega)$. Let $Q := Bu$ and, for $\ell \in \mathbb{N}\_0$, let $Q_{h_\ell} := Bu_{h_\ell}$. Suppose the FE solution on each level is computed with an optimal multigrid method, such that $\mathcal{C}(Q_{h_\ell}) \le Ch_\ell^{-d}$. Then, for any $0 < \varepsilon < e^{-1}$, there exists $L \in \mathbb{N}$ and $\lbrace N_\ell \rbrace_{\ell=0}^L \subset \mathbb{N}$ such that

$$\mathcal{C}_\varepsilon\Big(\widehat{Q}_{L, \lbrace N_\ell \rbrace}^{ML}\Big) \le C\varepsilon^{-2}.$$

</div>

In the case of uniform mesh refinement in two spatial dimensions with $d = 2$ and $r = 2$, since the variance of $Y\_\ell$ decreases with a rate $\beta = 4$, it follows from (5.4.12) that the number of samples can be reduced by a factor of $2^{(4+2)/2} = 8$ from level to level.

#### 5.4.3 Quasi-Monte Carlo

We only consider quasi-Monte Carlo (QMC) methods in the context of the elliptic PDE in Section 5.2.1 with uniform diffusion coefficient $a$, as described in Example 4.5.17.

The Karhunen-Loève expansion is truncated after $s$ terms, and we set $\Xi := (\xi_j)\_{j=1}^s$ with iid $\xi_j \sim \mathrm{uniform}(-1, 1)$, such that $\mu_\Xi$ is the product uniform measure on $V = [-1, 1]^s$. As quantity of interest, we consider a linear functional $B : H_0^1(D) \to \mathbb{R}$ of the PDE solution $u$, which, as a functional of the parameter vector $\Xi$ we denote by $Q := F(\Xi)$. The functional $F : V \to \mathbb{R}$ can then be written as the composition

$$F := B \circ \mathcal{G} \circ T, \quad \text{such that} \quad \Xi \xrightarrow{T} a \xrightarrow{\mathcal{G}} u \xrightarrow{B} Q,$$

where $T : V \to L^\infty(D)$ is the operator defined in Section 4.5.3 and $\mathcal{G} : L^\infty(D) \to H\_0^1(D)$ is the solution operator, mapping the coefficient $a$ to the PDE solution $u$. Similarly, we denote by $Q\_h := F\_h(\Xi)$ with $F\_h = B \circ \mathcal{G}\_h \circ T$ the FE approximation of $Q$, where $\mathcal{G}\_h : L^\infty(D) \to U\_h$ is the FE solution operator, mapping the coefficient $a$ to the FE solution $u\_h$.

QMC methods are formulated as quadrature rules over the unit cube $[0, 1]^s$. Treating $\Xi$ as a deterministic parameter vector $\xi$ distributed according to product uniform measure,

$$\mathbb{E}[Q] \approx \int_{[-1,1]^s} F_h(x) \, \mathrm{d}\mu_\Xi(x) = \int_{[0,1]^s} F_h(2v - \mathbf{1}) \, \mathrm{d}v, \tag{5.4.13}$$

where we used the simple change of variables $x = 2v - \mathbf{1}$ from $[0, 1]$ to $[-1, 1]$. We will use a **randomly shifted rank-1 lattice rule** to approximate (5.4.13). This takes the form

$$\widehat{Q}_{h,N}^{\mathrm{QMC}} = \frac{1}{N} \sum_{i=1}^{N} F_h(\tilde{\Xi}^{(i)}), \quad \text{where} \quad \tilde{\Xi}^{(i)} := 2 \operatorname{frac}\left(\frac{iz}{N} + \Delta\right) - \mathbf{1}, \tag{5.4.14}$$

$z \in \lbrace 1, \ldots, N-1 \rbrace^s$ is a so-called **generating vector**, $\Delta$ is a uniformly distributed **random shift** on $[0, 1]^s$, and "frac" denotes the fractional part function, applied componentwise. To ensure that every one-dimensional projection of the lattice rule has $N$ distinct values we furthermore assume that each component $z\_j$ of $z$ satisfies $\gcd(z\_j, N) = 1$.

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/qmc_lattice_rule.png' | relative_url }}" alt="Left: 55 points of a rank-1 lattice rule with generating vector z = (1, 34) in the unit square, arranged in strikingly regular diagonal lines that cover the square evenly. Right: 55 iid uniform pseudo-random points in the same square, showing visible clumps and gaps." loading="lazy">
</figure>

*Two-dimensional lattice rule with $N = 55$, $z = (1, 34)^\top$, $\Delta = (0, 0)^\top$ (left). The points are deterministic and cover the square far more evenly than $55$ iid uniform samples (right) — no clumps, no holes. The random shift $\Delta$ moves the whole pattern rigidly (modulo 1), which restores unbiasedness without destroying the even coverage.*

Due to the random shift, (5.4.14) is an unbiased estimator of $\mathbb{E}\_{\mu\_\Xi}[F\_h(\Xi)]$ and thus we have — as for MC and MLMC —

$$\mathbb{E}\left[\Big(\widehat{Q}_{h,N}^{\mathrm{QMC}} - \mathbb{E}[Q]\Big)^2\right] = \big(\mathbb{E}[Q_h - Q]\big)^2 + \mathbb{V}\Big(\widehat{Q}_{h,N}^{\mathrm{QMC}}\Big), \tag{5.4.15}$$

where the variance of the QMC estimator is given by

$$\mathbb{V}\Big(\widehat{Q}_{h,N}^{\mathrm{QMC}}\Big) = \mathbb{E}_\Delta\left[\Big(\widehat{Q}_{h,N}^{\mathrm{QMC}} - \mathbb{E}_{\mu_\Xi}[F_h(\Xi)]\Big)^2\right]. \tag{5.4.16}$$

To bound it, we make the following assumption on the integrand $F\_h(\Xi)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption 5.4.8</span></p>

Let $C > 0$ be a constant independent of $s$ and let $(\ell\_j)\_{j \in \mathbb{N}} \in \ell^1(\mathbb{N})$ be as defined in Example 4.5.17. We assume that, for any multi-index $\nu \in \lbrace 0, 1 \rbrace^s$ with $\|\nu\| = \sum\_{j \le s} \nu\_j$,

$$\left| \frac{\partial^{|\nu|} F(\xi)}{\partial \xi^\nu} \right| \le C \frac{|\nu|!}{(\ln 2)^{|\nu|}} \prod_{j=1}^{s} \ell_j^{\nu_j}.$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(What Assumption 5.4.8 is really saying)</span></p>

Decode the bound coordinate by coordinate. Taking $\nu = e\_j$ (one derivative in direction $j$) gives $\|\partial F / \partial \xi\_j\| \lesssim \ell\_j$: the integrand's sensitivity to the $j$-th KL coordinate is controlled by the $j$-th KL weight. Since $(\ell\_j)\_j$ is summable and decaying, **the coordinates are ordered by importance and only the first few matter much** — the integrand is "effectively low-dimensional" even though $s$ may be huge. This is exactly the structure a lattice rule can exploit, and it is the real reason the variance bound in Lemma 5.4.9 is independent of $s$: the curse of dimensionality is not defeated in general, it is defeated *for integrands whose dependence on high coordinates decays*. Mixed derivatives (general $\nu \in \lbrace 0,1 \rbrace^s$) must satisfy the product version of the same bound, with a controlled combinatorial growth $\|\nu\|!/(\ln 2)^{\|\nu\|}$ in the order.

For the elliptic model problem this is not an assumption we impose on the data — it is a *theorem* about the solution map: differentiating the weak form (5.2.1) with respect to $\xi\_j$ shows $\partial\_{\xi\_j} u$ solves the same PDE with right-hand side driven by $\ell\_j \varphi\_j$, and iterating gives precisely such product bounds.

The payoff, comparing rates: randomly shifted lattice rules achieve RMSE close to $N^{-1}$ (take $\delta \to 1/2$, so variance $N^{-1/\delta} \to N^{-2}$) instead of MC's RMSE $N^{-1/2}$ — one extra digit of accuracy costs a factor $10$ instead of $100$ in samples, at essentially no extra cost per sample.

</div>

For linear functionals $B$ on $H\_0^1(D)$, this assumption has been proved in the uniform case. However, it can also be shown for nonlinear functionals.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.4.9</span><span class="math-callout__name">(QMC Variance Bound)</span></p>

Suppose Assumption 5.4.8 holds and $(\ell_j)\_{j \in \mathbb{N}} \in \ell^r(\mathbb{N})$, for some $r \in (0, 1)$. Then, a randomly shifted lattice rule can be constructed via a component-by-component algorithm in $\mathcal{O}(sN \log N)$ cost, such that

$$\mathbb{V}\Big(\widehat{Q}_{h,N}^{\mathrm{QMC}}\Big) \le C \begin{cases} N^{-1/\delta}, & \text{if } r \in (0, 2/3], \\ N^{-(1/r - 1/2)}, & \text{if } r \in (2/3, 1), \end{cases}$$

for any $\delta \in (1/2, 1]$, **independently** of $s$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/qmc_mc_convergence.png' | relative_url }}" alt="Log-log plot of root mean square quadrature error against the number of samples N for the elliptic quantity of interest in 16 dimensions. The Monte Carlo curve decreases along a reference line of slope minus one half; the quasi-Monte Carlo curve with a randomly shifted lattice rule lies an order of magnitude lower and decreases along a reference line of slope minus one." loading="lazy">
</figure>

*Lemma 5.4.9 measured: root mean square quadrature errors for $\mathbb{E}[u\_h(0.5)]$ in the uniform elliptic problem with $s = 16$ KL coordinates. Plain MC follows $N^{-1/2}$; a randomly shifted rank-1 lattice rule (generating vector constructed by the component-by-component algorithm with weights $\gamma\_j = j^{-2}$, RMSE over $40$ random shifts) follows the almost-optimal rate $N^{-1}$ — and is already two orders of magnitude more accurate at moderate $N$, despite the $16$-dimensional integration domain.*

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.4.10</span><span class="math-callout__name">(QMC Complexity for Elliptic PDE)</span></p>

Suppose Assumption 5.4.8 holds and $(\ell_j)\_{j \in \mathbb{N}} \in \ell^r(\mathbb{N})$, for some $r \in (0, 2/3]$. Suppose further that the piecewise linear FE solution is computed with an optimal multigrid method, such that $\mathcal{C}(Q_h) \le Ch^{-d}$. Then, for any $\varepsilon > 0$, there exists $h > 0$ and $N \in \mathbb{N}$ such that

$$\mathcal{C}_\varepsilon(\widehat{Q}_{h,N}^{\mathrm{QMC}}) \le C\varepsilon^{-2\delta - d/2}, \quad \text{for any } \delta \in (1/2, 1].$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.4.11</span></p>

**(a)** It is even possible to combine quasi-Monte Carlo sampling and multilevel estimation and the gains are complementary [Kuo et al, 2015; Kuo et al, 2017], but we will not include these estimators or their analysis here.

**(b)** For smooth random fields, e.g. fast decay of the $\ell\_j$ in Example 4.5.17, even faster convergence rates are possible with higher-order QMC rules [Dick et al, 2014] or with stochastic collocation and sparse grid quadrature rules.

**(c)** Note that due to Remark 5.2.7 and the comments before Lemma 5.4.9, the statements of Corollaries 5.4.5, 5.4.7 and 5.4.10 also hold for Fréchet-differentiable nonlinear functionals $Q := B(u)$ and for nonuniform measures $\mu\_a$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.4.12</span><span class="math-callout__name">(Comparison of Sampling Methods in the Lognormal Case)</span></p>

To compare the approaches, let us consider the elliptic PDE (5.2.1) for $D = (0, 1)^2$ (i.e., $d = 2$) and $f \equiv 1$, with lognormal diffusion coefficient $a \in L^\infty(D)$, i.e., $\log a \sim \mathcal{N}(m, C\_{\nu, \sigma^2, \lambda})$, with Matérn covariance $C\_{\nu, \sigma^2, \lambda}$. The **Matérn covariance function** is defined, for any $x, y \in D$, as

$$c(x, y) := \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{2\sqrt{\nu}\, |x - y|}{\lambda} \right)^{\nu} K_\nu\left( \frac{2\sqrt{\nu}\, |x - y|}{\lambda} \right),$$

where $\Gamma$ and $K\_\nu$ are the Gamma-function and the modified Bessel function (of second kind) of order $\nu$, and where $\nu$, $\sigma^2$ and $\lambda$ are the so-called *smoothness parameter*, *total variance* and *correlation length*, respectively. The quantity of interest is

$$Q(\omega) := \frac{1}{|D^\ast|} \int_{D^\ast} u(x, \omega) \, \mathrm{d}x, \quad \text{with} \quad D^\ast := \left( \tfrac{3}{4}, \tfrac{7}{8} \right) \times \left( \tfrac{7}{8}, 1 \right).$$

For a comparison of the sampling approaches discussed above, we use piecewise linear FEs on a uniform simplicial mesh to discretise the PDE and a truncated Karhunen-Loève expansion to sample from $\log a$ for $\nu = 2.5$. In that case, all the relevant assumptions in Corollaries 5.4.5 and 5.4.7 are satisfied and the assumptions of (the lognormal equivalent of) Corollary 5.4.10 hold with $(\ell\_j)\_{j \in \mathbb{N}} \in \ell^r(\mathbb{N})$ and $r < 2/3$. The theoretical complexity bounds, as well as the theoretical bound for multilevel QMC (MLQMC), are collected in Table 5.1. We see that in dimension $d \ge 2$, the cost of MLQMC is asymptotically optimal, in the sense that even computing a single sample to accuracy $\varepsilon$ has the same asymptotic complexity.

| $d$ | MC | MLMC | QMC | MLQMC | One sample |
|-----|-----|------|-----|-------|------------|
| 1   | 2.5 | 2    | 1.5 | 1     | 0.5        |
| 2   | 3   | 2    | 2   | 1     | 1          |
| 3   | 3.5 | 2    | 2.5 | 1.5   | 1.5        |

*Table 5.1: Theoretical bounds on the order of growth of the $\varepsilon$-cost with respect to $\varepsilon^{-1}$, for the lognormal case for $\nu = 2.5$ (ignoring log-factors and choosing $\delta = 1/2$ in Corollary 5.4.10).*

In numerical experiments (measured $\varepsilon$-costs for $\nu = 2.5$, $\sigma^2 = 0.25$ and $\lambda = 1$, with levels $L = 1, \ldots, 5$ and $h\_0 = \sqrt{2}/8$), these theoretical bounds are attained in practice. For the QMC methods, an embedded lattice rule with weights $\gamma\_j = j^{-2}$ was used, with generating vector taken from the file `lattice-39102-1024-1048576.3600.txt` on Frances Kuo's webpage (UNSW Sydney).

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(How to read Table 5.1)</span></p>

The table rewards a careful look, because it summarises the entire section in one grid:

* **Read along a row** (fixed $d$): each method to the right strictly improves the exponent — MC $\to$ MLMC removes (most of) the PDE-solve penalty $\gamma/\alpha = d/2$; MC $\to$ QMC replaces the sampling exponent $2$ by $2\delta \approx 1$; MLQMC stacks both gains.
* **Read down a column**: for MC and QMC the exponent grows with $d$ (the per-sample PDE solve gets more expensive), while for MLMC it is flat at $2$ — the multilevel construction has completely decoupled the complexity from the spatial dimension, as predicted by the $\beta > \gamma$ / $\beta = \gamma$ cases of Theorem 5.4.6.
* **The "One sample" column is the sanity bound**: no estimator can possibly be cheaper than computing a single realisation of $Q\_h$ with bias $\varepsilon$, which costs $h^{-d} \sim \varepsilon^{-d/2}$. MLQMC matching this bound (for $d \ge 2$, up to log-factors) is the strongest statement one can make: *the quadrature is asymptotically free; only the bias constraint remains.*

</div>

### 5.5 Importance Sampling Estimators for Posterior Expectations

However, crucially, in Bayesian inverse problems we typically only have access to the posterior distribution in unnormalised form, i.e.

$$\frac{\mathrm{d}\mu_{X|y}}{\mathrm{d}\mu_X}(x) \propto \exp\left(-\frac{1}{2}\lVert y - \Phi(x) \rVert_\Sigma^2\right) \quad \text{or} \quad \pi_{X|Y}(x|y) \propto \exp\left(-\frac{1}{2}\lVert y - \Phi(x) \rVert_\Sigma^2\right) \pi_X(x),$$

in the infinite/finite dimensional case for an additive Gaussian noise model, respectively. For simplicity, we will only focus on the case of a finite dimensional parameter $X : \Omega \to \mathbb{R}^s$.

#### 5.5.1 Importance Sampling and Ratio Estimators

A classical technique to sample from a distribution that is only given in unnormalised form and a method that can also be used to reduce the variance in the estimator if we have an approximation for the (normalised or unnormalised) density is **importance sampling**.

Suppose again that we are interested in computing

$$\mathbb{E}_p[F(X)] = \int_{\mathcal{S}} F(x) \, \mathrm{d}\nu(x) = \int_{\mathcal{S}} F(x)p(x) \, \mathrm{d}x,$$

for some RV $X : \Omega \to \mathcal{S} \subset \mathbb{R}^s$ where $\nu$ is a probability measure on $\mathcal{S} \subset \mathbb{R}^s$ with density $p$. If $q$ is a positive probability density function on $\mathbb{R}^s$, then

$$\mathbb{E}_p[F(X)] = \int_{\mathcal{S}} \frac{F(x)p(x)}{q(x)} q(x) \, \mathrm{d}x = \mathbb{E}_q\left[\frac{F(X)p(X)}{q(X)}\right] \tag{5.5.1}$$

By making a multiplicative adjustment to the integrand we compensate for sampling from $q$ instead of $p$. The adjustment factor $w(x) = p(x)/q(x)$ is called the **likelihood ratio** or **importance weight**. The distribution $q$ is the **importance distribution** and $p$ is the **nominal distribution**. It is enough to have $q(x) > 0$ whenever $F(x)p(x) \neq 0$.

The **importance sampling estimator** for $m := \mathbb{E}\_p[F(X)]$ is

$$\widehat{Q}_{q,N}^{\mathrm{IS}} := \frac{1}{N} \sum_{i=1}^{N} \frac{F(X^{(i)})p(X^{(i)})}{q(X^{(i)})} \quad \text{with iid } X^{(i)} \sim q. \tag{5.5.2}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.5.1</span><span class="math-callout__name">(Importance Sampling Variance)</span></p>

Let $q(x) > 0$ whenever $F(x)p(x) \neq 0$. For any $N \in \mathbb{N}$, $\mathbb{E}\_q[\widehat{Q}\_{q,N}^{\mathrm{IS}}] = m = \mathbb{E}\_p[F(X)]$ and $\mathbb{V}\_q(\widehat{Q}_{q,N}^{\mathrm{IS}}) = \sigma_q^2/N$ where

$$\sigma_q^2 = \int_{\mathcal{S}} \frac{(F(x)p(x))^2}{q(x)} \, \mathrm{d}x - m^2 = \int_{\mathcal{S}} \frac{(F(x)p(x) - mq(x))^2}{q(x)} \, \mathrm{d}x. \tag{5.5.3}$$

</div>

Theorem 5.5.1 guides us in selecting a good importance sampling rule. From the first expression in (5.5.3) we see that a better $q$ is one that gives a smaller value of $\int\_{\mathcal{S}} (Fp)^2/q \, \mathrm{d}x$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.5.2</span><span class="math-callout__name">(Optimal Importance Density)</span></p>

Let $\mathbb{E}\_p[\|F(X)\|] \neq 0$. The probability density $q^\ast$ with $q^\ast (x) \propto \|F(x)\|p(x)$ minimises $\sigma_q^2$ over all densities $q$ that are positive when $Fp \neq 0$, i.e. $\sigma_{q^\ast}^2 \le \sigma_q^2$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 5.5.2</summary>

Let $q^\ast(x) = \|F(x)\|p(x) / \mathbb{E}\_p\big[\|F(X)\|\big]$ and let $q$ be an arbitrary density such that $q(x) > 0$ when $F(x)p(x) \neq 0$. Then

$$\begin{aligned}
m^2 + \sigma_{q^\ast}^2 &= \int_{\mathcal{S}} \frac{F(x)^2 p(x)^2}{q^\ast(x)} \, \mathrm{d}x = \int_{\mathcal{S}} \frac{F(x)^2 p(x)^2}{|F(x)| p(x) \big/ \mathbb{E}_p\big[|F(X)|\big]} \, \mathrm{d}x \\
&= \Big( \mathbb{E}_p\big[|F(X)|\big] \Big)^2 = \left( \mathbb{E}_q\left[ \frac{|F(X)| p(X)}{q(X)} \right] \right)^2 \le \mathbb{E}_q\left[ \frac{F(X)^2 p(X)^2}{q(X)^2} \right] = m^2 + \sigma_q^2,
\end{aligned}$$

where the inequality is Jensen's inequality (or equivalently Cauchy-Schwarz) applied to the RV $\|F(X)\| p(X)/q(X)$ under $q$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.5.3</span></p>

If $F(x) > 0$ is positive where $p(x) > 0$ and $m > 0$, then the optimal density $q^\ast = \frac{1}{m}Fp$ has $\sigma\_{q^\ast}^2 = 0$, but it is of no practical interest, because each of the samples in $\widehat{Q}\_{q^\ast,N}^{\mathrm{IS}}$ becomes $F(X^{(i)})p(X^{(i)})/q^\ast (X^{(i)}) = m$, which is available only if we know the final result anyway. Although zero-variance importance sampling densities are not usable, they provide insight into the design of a good importance sampling scheme. It may be good for $q$ to have spikes in the same places that $\|F\|$ does, or where $p$ does, but it is even better to have them where $\|F\|p$ does. The appearance of $q$ in the denominator of $w = p/q$ means that light-tailed importance densities $q$ are dangerous. If we are clever or lucky, then $F$ might be small just where it needs to be to offset the small denominator. But we often need to use the same sample with multiple integrands $F$, and so as a rule $q$ should have tails at least as heavy as $p$ does.

</div>

In the Bayesian setting we can only sample from an unnormalized version of $p$, $p_u(x) = cp(x)$, where $c > 0$ is unknown. The same may be true of $q$, e.g., if we can compute $q_u(x) = bq(x)$ and $b > 0$ might be unknown. In general, $b \neq c$ and thus $p(x)/q(x) \neq p_u(x)/q_u(x)$. However, we may compute the ratio $w_u(x) = p_u(x)/q_u(x) = (c/b)p(x)/q(x)$ and consider the **self-normalized importance sampling estimator** or **ratio estimator**

$$\widehat{Q}_{q,N}^{\mathrm{RE}} := \frac{\sum_{i=1}^{N} F(X^{(i)}) w_u(X^{(i)})}{\sum_{i=1}^{N} w_u(X^{(i)})} = \frac{\frac{1}{N}\sum_{i=1}^{N} F(X^{(i)}) w(X^{(i)})}{\frac{1}{N}\sum_{i=1}^{N} w(X^{(i)})} \quad \text{with iid } X^{(i)} \sim q. \tag{5.5.4}$$

To obtain iid samples of $q$ it suffices to know $q\_u$, and the factor $b/c$ cancels from the numerator and the denominator in (5.5.4), leading to the same estimate as if we had used the desired ratio $w(x) = p(x)/q(x)$ instead of the computable alternative $w\_u(x)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.5.4</span><span class="math-callout__name">(Ratio Estimator Convergence)</span></p>

Let $p, q$ be two probability densities on $\mathbb{R}^s$ with $q(x) > 0$ whenever $p(x) > 0$. Then,

$$\widehat{Q}_{q,N}^{\mathrm{RE}} \xrightarrow{\mathbb{P}\text{-a.s.}}{N \to \infty} \mathbb{E}_p[F(X)] =: m, \tag{5.5.5}$$

but in general $\mathbb{E}\_q[\widehat{Q}\_{q,N}^{\mathrm{RE}}] \neq m$, i.e. the estimator is biased. We also have

$$\sqrt{N}\Big(\widehat{Q}_{q,N}^{\mathrm{RE}} - m\Big) \xrightarrow{d}{N \to \infty} \mathcal{N}(0, \sigma_q^2), \tag{5.5.6}$$

with **asymptotic variance** $\sigma_q^2$, as defined in (5.5.3).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 5.5.4</summary>

Consider the second form of the definition of $\widehat{Q}\_{q,N}^{\mathrm{RE}}$ in (5.5.4). The numerator is equal to $\widehat{Q}\_{q,N}^{\mathrm{IS}}$, which we have already seen is an unbiased estimator of $m$. The strong law of large numbers gives $\mathbb{P}\big( \lim\_{N \to \infty} \widehat{Q}\_{q,N}^{\mathrm{IS}} = m \big) = 1$. Using the same arguments as for the numerator also for the denominator, but with the constant functional $F \equiv 1$, we see that the denominator converges almost surely to $\mathbb{E}\_q[w(X)] = \int p(x) \, \mathrm{d}x = 1$, which implies (5.5.5).

To see that in general $\widehat{Q}\_{q,N}^{\mathrm{RE}}$ is biased, consider $N = 1$, $p \neq q$ and $F(x) = x$. Then the weights cancel completely,

$$\mathbb{E}_q\big[\widehat{Q}_{q,N}^{\mathrm{RE}}\big] = \mathbb{E}_q\left[ \frac{F(X^{(1)}) w(X^{(1)})}{w(X^{(1)})} \right] = \mathbb{E}_q\big[X^{(1)}\big] \neq \mathbb{E}_p\big[X^{(1)}\big] = m.$$

The result in (5.5.6) can be shown using again the Central Limit Theorem.

</details>
</div>

Note that the conditions on $q$ for the ratio estimator are slightly stronger than for the importance sampling estimator, i.e., we need $q(x) > 0$ whenever $p(x) > 0$, rather than only whenever $F(x)p(x) \neq 0$.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The ratio estimator: what is traded for not knowing $Z$)</span></p>

The self-normalised estimator is the workhorse of Bayesian computation, so it pays to be precise about what exactly is lost and kept compared to plain importance sampling:

* **Kept: consistency and the CLT rate.** By (5.5.5)-(5.5.6), $\widehat{Q}\_{q,N}^{\mathrm{RE}}$ still converges a.s. and still fluctuates at scale $N^{-1/2}$, with the *same* asymptotic variance $\sigma\_q^2$ as the (infeasible) importance sampling estimator. Asymptotically, not knowing the normalising constant is free.
* **Lost: unbiasedness at finite $N$.** The estimator is a ratio of two correlated random quantities, and $\mathbb{E}[A/B] \neq \mathbb{E}[A]/\mathbb{E}[B]$. The bias is $O(1/N)$ — one order smaller than the $O(N^{-1/2})$ statistical error, hence usually harmless — but it is structural: no finite-sample trick removes it. (Contrast with MC and MLMC, which were exactly unbiased for $\mathbb{E}[Q\_h]$.)
* **The denominator is the weak point.** The denominator $\frac{1}{N} \sum\_i w(X^{(i)})$ estimates $1$, but if $q$ is a poor match for $p$, the weights $w = p/q$ are wildly variable: most samples get negligible weight and a few dominate — the *weight degeneracy* phenomenon. A useful diagnostic is the **effective sample size** $N\_{\mathrm{eff}} = \big( \sum\_i w\_i \big)^2 / \sum\_i w\_i^2 \in [1, N]$, which measures how many "ideal" samples the weighted ensemble is worth. Exactly this degeneracy, quantified in Lemma 5.5.10 below, is what breaks prior-based importance sampling in the small-noise limit and motivates Section 5.5.3.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise Setting</span><span class="math-callout__name">(Importance sampling)</span></p>

Let $\mu$ be a probability measure on $\mathbb{R}$ with unnormalized density

$$
\tilde f(x)
=
\exp\left(-\frac{x^2}{2}\right)
\left(
\sin^2(6x) + 3\cos^2(x)\sin^2(4x) + 1
\right).
\tag{6.1.1}
$$

We wish to approximate

$$
I = \int_{-\infty}^{\infty} x^2 \, d\mu(x)
$$

using only $\tilde f$. Let $g$ be the standard normal density, i.e.

$$g(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right).$$

We define the unnormalized importance weight

$$w_u(x) := \frac{\tilde f(x)}{g(x)}.\tag{6.1.2}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Importance sampling (a))</span></p>

Explain why this estimator is self-normalized. Derive the estimator

$$\hat I_N^{\mathrm{RE}}=\frac{\sum_{j=1}^N X_j^2 w_u(X_j)}{\sum_{j=1}^N w_u(X_j)},$$

where $X_1,\dots,X_N$ are iid samples from $g$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Write

$$\widetilde f(x) = e^{-x^2/2} \Bigl( \sin^2(6x)+3\cos^2(x)\sin^2(4x)+1 \Bigr),$$

and

$$g(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}.$$

Since $\mu$ is only given through an unnormalised density, we have to normalize it (since $\mu$ is a probability measure given only through an unnormalised density $\widetilde f$, to get the actual density of $\mu$ with respect to Lebesgue measure we need the normalising constant $Z=\int \widetilde f(x)dx$ to obtain the actual Radon–Nikodym derivative $d\mu/dx=\widetilde f/Z$.)

$$\mu(dx)=\frac{\widetilde f(x)}{Z},dx, \qquad Z=\int_{\mathbb R}\widetilde f(x),dx.$$

Therefore

$$I = \int_{\mathbb R}x^2,d\mu(x) = \frac{\int_{\mathbb R}x^2\widetilde f(x),dx}{\int_{\mathbb R}\widetilde f(x),dx}.$$

Now insert $g(x)/g(x)$:

$$I = \frac{\int_{\mathbb R}x^2\frac{\widetilde f(x)}{g(x)}g(x),dx}{\int_{\mathbb R}\frac{\widetilde f(x)}{g(x)}g(x),dx} = \frac{\mathbb E_g[X^2w_u(X)]} {\mathbb E_g[w_u(X)]}.$$

Thus, for iid samples $X_1,\dots,X_N\sim g$,

$$\boxed{\widehat I_N^{\mathrm{RE}} = \frac{\sum_{j=1}^N X_j^2 w_u(X_j)}{\sum_{j=1}^N w_u(X_j)}}.$$

This is **self-normalised** because the normalised empirical weights are

$$\overline w_j = \frac{w_u(X_j)}{\sum_{i=1}^N w_u(X_i)}, \qquad \sum_{j=1}^N \overline w_j=1,$$

so equivalently

$$\widehat I_N^{\mathrm{RE}} = \sum_{j=1}^N \overline w_j X_j^2.$$

Here

$$w_u(x) = \frac{\widetilde f(x)}{g(x)} = \sqrt{2\pi} \Bigl( \sin^2(6x)+3\cos^2(x)\sin^2(4x)+1 \Bigr).$$

The factor $\sqrt{2\pi}$ cancels from numerator and denominator, so in code it is enough to use

$$h(x)=\sin^2(6x)+3\cos^2(x)\sin^2(4x)+1.$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Importance sampling (b))</span></p>

Implement the estimator for a range of sample sizes $N$. Repeat the experiment several times and plot the empirical mean and variance of $\hat I_N^{\mathrm{RE}}$ as functions of $N$.

Estimate the reference value of $I$ by an independent reference run with

$$N_{\mathrm{ref}} = 10^6$$

samples and report the value of $N_{\mathrm{ref}}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**Setup and one key simplification.** Before writing any code it pays to look at the weight (6.1.2) analytically. Since $\widetilde{f}$ carries the *same* Gaussian envelope as the proposal $g$, the exponentials cancel exactly:

$$w_u(x) = \frac{\widetilde{f}(x)}{g(x)} = \sqrt{2\pi} \, \underbrace{\big( \sin^2(6x) + 3\cos^2(x)\sin^2(4x) + 1 \big)}_{=: h(x)}, \qquad 1 \le h(x) \le 5.$$

Two consequences:

* the weights are **uniformly bounded above and below**, $\sqrt{2\pi} \le w\_u(x) \le 5\sqrt{2\pi}$ — the worst weight exceeds the best by at most a factor $5$, whatever $x$ is sampled;
* the constant $\sqrt{2\pi}$ (and any other multiplicative constant) **cancels** in the ratio estimator, exactly as self-normalisation promises — we could equally well work with $h$ alone. This is the whole point of the estimator: it never needs the normalisation constant $Z = \int \widetilde{f}$.

**Implementation.** For each sample size $N \in \lbrace 2^5, \ldots, 2^{17} \rbrace$ we run $R = 200$ independent repetitions of the estimator (vectorised over the repetitions), and one independent reference run with $N\_{\mathrm{ref}} = 10^6$ samples.

```python
import numpy as np

rng = np.random.default_rng(0)

def h(x):                     # bounded modulation factor, 1 <= h <= 5
    return np.sin(6*x)**2 + 3*np.cos(x)**2*np.sin(4*x)**2 + 1.0

def w_u(x):                   # (6.1.2); the Gaussians cancel analytically
    return np.sqrt(2*np.pi) * h(x)

def ratio_estimate(x):        # self-normalised IS estimator of E_mu[X^2]
    w = w_u(x)
    return np.sum(x**2 * w) / np.sum(w)

# independent reference run
N_ref = 10**6
I_ref = ratio_estimate(rng.standard_normal(N_ref))   # 0.828767

# repeated runs over a range of N
Ns, R = 2**np.arange(5, 18), 200
means, variances = [], []
for N in Ns:
    x = rng.standard_normal((R, N))
    w = w_u(x)
    est = (x**2 * w).sum(axis=1) / w.sum(axis=1)     # R ratio estimates
    means.append(est.mean())
    variances.append(est.var(ddof=1))
```

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/sheet6_is_mean_var.png' | relative_url }}" alt="Left: the empirical mean of the self-normalised estimator over 200 repetitions, plotted with two-standard-error bars against N from 32 to 131072 on a logarithmic axis; the error bars shrink steadily and the means settle on the dashed reference line at 0.8288. Right: the empirical variance against N on a log-log scale, following a dashed reference line of slope minus one over four decades." loading="lazy">
  <figcaption>Empirical mean with $\pm 2$ s.e. bars (left) and empirical variance (right) of $\widehat{I}_N^{\mathrm{RE}}$ over $R = 200$ repetitions. The dashed line on the left is the reference value $\widehat{I}_{\mathrm{ref}} \approx 0.8288$ from an independent run with $N_{\mathrm{ref}} = 10^6$ samples; the dashed line on the right has slope $N^{-1}$.</figcaption>
</figure>

**Results and comments.**

* **Reference value.** The independent reference run with $N\_{\mathrm{ref}} = 10^6$ samples gives

  $$\widehat{I}_{\mathrm{ref}} = 0.8288.$$

  As a sanity check (not part of the task), deterministic trapezoidal quadrature of $\int x^2 \widetilde{f} \big/ \int \widetilde{f}$ on a fine grid gives $I \approx 0.8273$; the two agree within one standard error of the reference run ($\mathrm{s.e.} \approx \sqrt{2.4/10^6} \approx 1.6 \cdot 10^{-3}$), as they should.
* **Mean.** The empirical mean is statistically consistent with the reference value for every $N$ — the deviations at small $N$ lie inside the $\pm 2$ s.e. bars. The ratio estimator *is* biased at finite $N$ (numerator and denominator are correlated random quantities), but the bias is $O(1/N)$, one order below the $O(N^{-1/2})$ statistical error, and is invisible at this number of repetitions — consistent with Lemma 5.5.4 of the lecture.
* **Variance.** The empirical variance follows the line $\mathrm{Var} \approx \sigma^2/N$ with $\sigma^2 \approx 2.4$ over four decades — each doubling of $N$ halves the variance. This is the CLT for self-normalised importance sampling: $\sqrt{N}\big( \widehat{I}\_N^{\mathrm{RE}} - I \big) \to \mathcal{N}(0, \sigma\_q^2)$ with the asymptotic variance $\sigma\_q^2$ from the lecture (Lemma 5.5.4, eq. (5.5.6)).

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Importance sampling (c))</span></p>

For the same runs compute the normalized weights

$$\overline w_{u,j}=\frac{w_u(X_j)}{\sum_{i=1}^N w_u(X_i)}$$

and the effective sample size

$$N_{\mathrm{eff}}=\frac{1}{\sum_{j=1}^N \overline w_{u,j}^{\,2}}.$$

Plot $N_{\mathrm{eff}}/N$ against $N$ and comment on the quality of the proposal density $g$ for this example.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**Implementation.** The normalised weights and the effective sample size are computed from the same runs as in (6.1b) — note that any multiplicative constant in $w\_u$ (in particular the $\sqrt{2\pi}$, and the unknown $Z$) cancels in $\overline{w}\_{u,j}$, so $N\_{\mathrm{eff}}$ is computable *without any normalisation knowledge*, just like the estimator itself:

```python
# continuing inside the loop over N from part (b):
    wbar = w / w.sum(axis=1, keepdims=True)   # normalised weights, rows sum to 1
    N_eff = 1.0 / (wbar**2).sum(axis=1)       # effective sample size per run
    ess_fraction = N_eff / N                  # in (0, 1]
```

For a deterministic prediction of where $N\_{\mathrm{eff}}/N$ should settle, expand the definition: by the law of large numbers,

$$\frac{N_{\mathrm{eff}}}{N} = \frac{\big( \frac{1}{N}\sum_j w_u(X_j) \big)^2}{\frac{1}{N}\sum_j w_u(X_j)^2} \; \xrightarrow{N \to \infty} \; \frac{\big( \mathbb{E}_g[w_u] \big)^2}{\mathbb{E}_g[w_u^2]} \approx 0.857,$$

where the limit was evaluated by quadrature.

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/sheet6_is_ess.png' | relative_url }}" alt="The effective sample size fraction plotted against N from 32 to 131072 on a logarithmic axis. The curve is essentially a horizontal line at 0.857 for all N, with small error bars at small N that shrink to invisible, lying exactly on the dashed theoretical limit line." loading="lazy">
  <figcaption>Effective sample size fraction $N_{\mathrm{eff}}/N$ (mean $\pm$ std over the $R = 200$ runs) against $N$. The dashed line is the deterministic limit $(\mathbb{E}_g[w_u])^2 / \mathbb{E}_g[w_u^2] = 0.857$, computed by quadrature. The fraction is flat in $N$ — no weight degeneracy whatsoever.</figcaption>
</figure>

**Comments on the quality of $g$.** The proposal is close to ideal for this target, and the plot shows it: $N\_{\mathrm{eff}}/N \approx 0.86$ *independently of $N$*, i.e. out of every $N$ weighted samples we retain the statistical power of about $0.86\,N$ ideal samples, and this does not deteriorate as $N$ grows. The structural reason is the one identified in part (b):

* **Exact tail match.** $g$ carries the same Gaussian envelope $e^{-x^2/2}$ as $\widetilde{f}$, so the weight $w\_u = \sqrt{2\pi}\, h$ is bounded above *and below* ($1 \le h \le 5$). No single sample can ever dominate the weight sum — the worst possible weight imbalance is a factor $5$. This is precisely the textbook criterion (cf. Remark 5.5.3 of the lecture): the proposal has tails at least as heavy as the target, so light-tail weight blow-up is impossible.
* **What is lost, and why.** The missing $14\%$ is the price of the weight *fluctuation*: $h$ oscillates between $1$ and $5$ across the support, so the weights are not constant (only a constant weight, i.e. $g \propto \widetilde{f}$, would give $N\_{\mathrm{eff}} = N$). Since $h$ oscillates fast compared to the Gaussian, the loss is a fixed, benign constant.
* **The contrast to keep in mind.** This is the opposite of the degeneracy scenario of Section 5.5.3 of the lecture notes, where the target (a concentrating posterior) and the proposal (the prior) drift apart and $N\_{\mathrm{eff}}/N$ collapses like $n^{-s/2}$. Here target and proposal share their global shape and differ only by a bounded, oscillatory factor — importance sampling then works essentially at full efficiency, and increasing $N$ buys variance reduction at the ideal $N^{-1}$ rate seen in part (b).

</details>
</div>

#### 5.5.2 Estimating Posterior Expectations

Now let us return to the Bayesian inverse problem with additive Gaussian noise and assume that $p$ is the density of the posterior $\nu = \mu_{X\mid y}$ such that

$$p_u(x) = \exp\left(-\frac{1}{2}\lVert y - \Phi(x)\rVert_\Sigma^2\right) \pi_X(x)$$

and we have access to a family of approximations $F_h(X)$ of $F(X)$ and $\Phi_h(X)$ of $\Phi(X)$, parametrised by $h > 0$, such that $F_h \to F$ and $\Phi_h \to \Phi$ as $h \to 0$. We will analyse the accuracy and complexity of various ratio estimators for $\mathbb{E}\_p[F(X)]$ based on samples $F_h(X^{(i)})$ with $X^{(i)}$ drawn from some distribution $q$, again possibly given only in unnormalised form.

To simplify the notation let $w_u(x) = Zw(x)$ with $w = p/q$ and $Z := \mathbb{E}\_q[w_u(X)]$, such that

$$\mathbb{E}_p[F(X)] = \frac{\mathbb{E}_q[Q_w]}{Z} \quad \text{with} \quad Q_w := F(X)w_u(X),$$

the quantity of interest times the (unnormalised) weight. Similarly, we write $w_{u,h}(x) = Z_h w_h(x)$ with $w_h = p_h/q$ and $Z_h := \mathbb{E}\_q[w_{u,h}(X)]$, and consider the ratio estimator

$$\widehat{Q}_{q,h,N}^{\mathrm{RE}} := \frac{\widehat{Q}_{w,h}}{\widehat{Z}_h}, \tag{5.5.7}$$

for $\mathbb{E}\_p[Q]$ where again $Q = F(X)$ and $\widehat{Q}\_{w,h}$ and $\widehat{Z}\_h$ are estimators of MC-type for $\mathbb{E}\_q[Q_w]$ and for $Z$, respectively.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.5.5</span><span class="math-callout__name">(MSE of the Ratio Estimator)</span></p>

If $q(x) > 0$ when $p(x) > 0$ and $\lVert \widehat{Q}\_{q,h,N}^{\mathrm{RE}} \rVert_{L^\infty(\Omega)} < \infty$, then

$$\mathbb{E}\left[\Big(\widehat{Q}_{q,h,N}^{\mathrm{RE}} - \mathbb{E}_p[Q]\Big)^2\right] \le CZ^{-2} \left( \mathbb{E}\left[(\widehat{Q}_{w,h} - \mathbb{E}_q[Q_w])^2\right] + \mathbb{E}\left[(\widehat{Z}_h - Z)^2\right] \right), \tag{5.5.8}$$

where $C := 2\max\big(1, \lVert \widehat{Q}\_{q,h,N}^{\mathrm{RE}} \rVert\_{L^\infty(\Omega)}^2\big)$. The expected value is with respect to $q$ in the case of MC and MLMC and with respect to the random shift $\Delta \sim \mathrm{uniform}(0, 1)^s$ in QMC.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 5.5.5</summary>

Rearranging the error and using the triangle inequality, we have

$$\begin{aligned}
\mathbb{E}\left[\Big(\widehat{Q}_{q,h,N}^{\mathrm{RE}} - \mathbb{E}_p[Q]\Big)^2\right]
&= \frac{1}{Z^2} \mathbb{E}\left[\Big(\widehat{Q}_{w,h} - \mathbb{E}_q[Q_w] - \big(\widehat{Q}_{w,h}\big/\widehat{Z}_h\big)\big(\widehat{Z}_h - Z\big)\Big)^2\right] \\
&\le \frac{2}{Z^2} \, \mathbb{E}\left[\Big(\widehat{Q}_{w,h} - \mathbb{E}_q[Q_w]\Big)^2 + \Big(\widehat{Q}_{q,h,N}^{\mathrm{RE}}\Big)^2 \Big(\widehat{Z}_h - Z\Big)^2\right] \\
&\le \frac{2}{Z^2} \max\Big(1, \lVert \widehat{Q}_{q,h,N}^{\mathrm{RE}} \rVert_{L^\infty(\Omega)}^2\Big) \left( \mathbb{E}\left[\Big(\widehat{Q}_{w,h} - \mathbb{E}_q[Q_w]\Big)^2\right] + \mathbb{E}\left[\Big(\widehat{Z}_h - Z\Big)^2\right] \right).
\end{aligned}$$

For the first equality, note that

$$\widehat{Q}_{q,h,N}^{\mathrm{RE}} - \mathbb{E}_p[Q] = \frac{\widehat{Q}_{w,h}}{\widehat{Z}_h} - \frac{\mathbb{E}_q[Q_w]}{Z} = \frac{1}{Z}\left( \widehat{Q}_{w,h} - \mathbb{E}_q[Q_w] - \frac{\widehat{Q}_{w,h}}{\widehat{Z}_h}\big(\widehat{Z}_h - Z\big) \right).$$

</details>
</div>

In the following, let $p = \pi\_{X\mid Y}$ and denote by

$$\widehat{Q}_{q,h,\mathrm{typ}}^{\mathrm{RE}} = \widehat{Q}_{w,h}^{\mathrm{typ}} \Big/ \widehat{Z}_h^{\mathrm{typ}} \quad \text{with} \quad \mathrm{typ} = \mathrm{MC},\ \mathrm{ML},\ \mathrm{QMC},$$

the ratio estimator defined in (5.5.7) for the posterior expectation $\mathbb{E}\_p[Q]$ with $\widehat{Q}\_{w,h}^{\mathrm{typ}}$ chosen to be the MC estimator, the MLMC estimator or the QMC estimator for $\mathbb{E}\_q[Q\_w]$, respectively, and let $\widehat{Z}\_h^{\mathrm{typ}}$ be the corresponding estimator for the normalization constant. Then, Lemma 5.5.5 implies that the convergence and the computational complexity of the ratio estimator (5.5.7) follow directly from the results on the basic estimators in Section 5.4.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.5.6</span><span class="math-callout__name">(Complexity of the Ratio Estimator)</span></p>

Let $\mathrm{typ} = \mathrm{MC}$ or ML and let $q$ be a probability distribution on $\mathbb{R}^s$ with $q(x) > 0$ whenever $p(x) > 0$. Suppose $\lVert \widehat{Q}\_{q,h,N}^{\mathrm{RE}} \rVert_{L^\infty(\Omega)} < \infty$ and the assumptions of Theorems 5.4.4 and 5.4.6 hold with $\alpha, \beta \neq \gamma > 0$. Then, for any $0 < \varepsilon < e^{-1}$ there exists an $h > 0$ and an $N \in \mathbb{N}$, resp. $\lbrace N_\ell \rbrace \subset \mathbb{N}$, such that

$$\mathcal{C}_\varepsilon\Big(\widehat{Q}_{q,h,\mathrm{typ}}^{\mathrm{RE}}\Big) \le C \begin{cases} (Z\varepsilon)^{-2-\gamma/\alpha}, & \text{if typ} = \mathrm{MC}, \\ (Z\varepsilon)^{-2-\max(0,(\gamma-\beta)/\alpha)}, & \text{if typ} = \mathrm{ML}. \end{cases}$$

</div>

A similar result can be proved also for the QMC-based ratio estimator.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.5.7</span></p>

The asymptotic order of the $\varepsilon$-cost is independent of the choice of the importance distribution $q$. However, due to the appearance of the extra factor $Z^{-\eta}$, for some $\eta \ge 2$, the asymptotic constant will **strongly** depend on the choice of $q$, as we will discuss in Section 5.5.3.

</div>

Let us give some more details for all three estimators in the case of the elliptic PDE with uniform diffusion coefficient $a$. As in Section 5.4.3, we assume that $a$ is discretised via a truncated Karhunen-Loève expansion parametrised via $\Xi \sim \mathrm{uniform}(-1, 1)^s$. The first and most obvious choice for the importance distribution is the prior distribution, i.e. $q = \pi\_\Xi$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.5.8</span><span class="math-callout__name">(Complexity of the Prior-Based Ratio Estimator for the Elliptic PDE)</span></p>

Consider the elliptic PDE in Section 5.4.3 with $p = \pi\_{\Xi\mid Y}$ and $q = \pi\_\Xi$. Let $\mathrm{typ} = \mathrm{MC}$, QMC or ML, and consider the ratio estimator $\widehat{Q}\_{q,h,\mathrm{typ}}^{\mathrm{RE}}$ for the posterior expectation $\mathbb{E}\_p[Q]$ of $Q = F(\Xi)$ under the assumptions of Corollaries 5.4.5, 5.4.7, or 5.4.10, respectively. In the QMC case, let $(\ell\_j)\_{j \in \mathbb{N}} \in \ell^r(\mathbb{N})$ with $r < 2/3$; in the MLMC case, let $h\_0$ be sufficiently small.

Let $Q = F(\Xi) := B(u)$ and $\Phi(\Xi) := H(u)$ with $B$ and $H$ two bounded (and sufficiently smooth) functionals of the PDE solution from $H\_0^1(D)$ to $\mathbb{R}$ and $\mathbb{R}^m$, respectively. Then, for any $0 < \varepsilon < e^{-1}$ there exists an $h > 0$ and an $N \in \mathbb{N}$, resp. $\lbrace N\_\ell \rbrace \subset \mathbb{N}$, such that

$$\mathcal{C}_\varepsilon\Big(\widehat{Q}_{q,h,\mathrm{typ}}^{\mathrm{RE}}\Big) \le C \begin{cases} (Z\varepsilon)^{-2-d/2}, & \text{if typ} = \mathrm{MC}, \\ (Z\varepsilon)^{-2}, & \text{if typ} = \mathrm{ML}, \\ (Z\varepsilon)^{-1+\delta-d/2}, & \text{if typ} = \mathrm{QMC}, \quad \text{for any } \delta > 0. \end{cases}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Corollary 5.5.8</summary>

Since the unnormalised weight function $w\_{u,h}(\xi) = \exp\big( -\frac{1}{2} \lVert y - H(u\_h(\xi)) \rVert\_\Sigma^2 \big)$ and the product $F\_h(\xi) w\_{u,h}(\xi) = B(u\_h(\xi)) \exp\big( -\frac{1}{2} \lVert y - H(u\_h(\xi)) \rVert\_\Sigma^2 \big)$ are both sufficiently smooth, nonlinear functionals of the PDE solution, the extension of Theorem 5.2.5 referred to in Remark 5.2.7(a) applies and it is possible to prove analogues of Corollaries 5.4.5, 5.4.7 and 5.4.10 for nonlinear functionals to bound the right hand side of (5.5.8). For a full proof of this extension see [Scheichl, Stuart, Teckentrup, 2017].

Since by definition $q(\xi) > 0$ when $p(\xi) > 0$, we have $Z > 0$. Thus, provided $\lVert \widehat{Q}\_{q,h,N}^{\mathrm{RE}} \rVert\_{L^\infty(\Omega)} < \infty$, we can apply Lemma 5.5.5 and deduce that

$$\mathcal{C}_\varepsilon\Big(\widehat{Q}_{q,h,\mathrm{typ}}^{\mathrm{RE}}\Big) \le C \left( \mathcal{C}_{Z\varepsilon}\Big(\widehat{Q}_{w,h}^{\mathrm{typ}}\Big) + \mathcal{C}_{Z\varepsilon}\Big(\widehat{Z}_h^{\mathrm{typ}}\Big) \right).$$

Note that to compensate the factor $Z^{-2}$ in (5.5.8) we need to scale the required tolerances $\varepsilon$ for the individual estimators for the numerator and the denominator by $Z$.

It remains to verify $\lVert \widehat{Q}\_{q,h,N}^{\mathrm{RE}} \rVert\_{L^\infty(\Omega)} < \infty$. From the assumptions on $B$ and $H$ we deduce that there exist two constants $M\_F, M\_\Phi < \infty$, such that $\|F\_h(\xi)\| \le M\_F$ and $\lVert \Phi\_h(\xi) \rVert\_\Sigma \le M\_\Phi$, for any $\xi \in [-1, 1]^s$ and for any $h > 0$. Thus, recalling that $w\_{u,h}(\xi) \le 1$, we have

$$\Big|\widehat{Q}_{w,h}^{\mathrm{MC}}\Big| = \left| \frac{1}{N} \sum_{i=1}^N F_h(\xi^{(i)}) w_{u,h}(\xi^{(i)}) \right| \le M_F \quad \text{and}$$

$$\widehat{Z}_h^{\mathrm{MC}} = \frac{1}{N} \sum_{i=1}^N w_{u,h}(\xi^{(i)}) = \frac{1}{N} \sum_{i=1}^N \exp\left( -\frac{1}{2} \big\lVert y - \Phi_h(\xi^{(i)}) \big\rVert_\Sigma^2 \right) \ge \exp\left( -\lVert y \rVert_\Sigma^2 - M_\Phi^2 \right) =: M_Z > 0,$$

and thus $\big\|\widehat{Q}\_{q,h,\mathrm{MC}}^{\mathrm{RE}}\big\| \le M\_F / M\_Z < \infty$. The proof for $\mathrm{typ} = \mathrm{QMC}$ is identical.

For $\mathrm{typ} = \mathrm{ML}$, the upper bound follows in the same way. On the other hand, to bound $\widehat{Z}\_h^{\mathrm{ML}}$ we can use the nonlinear extension of Theorem 5.2.5 again to obtain, with $Y\_\ell = w\_{u,h\_\ell} - w\_{u,h\_{\ell-1}}$, that

$$\widehat{Z}_h^{\mathrm{ML}} \ge \widehat{Z}_{h_0}^{\mathrm{MC}} - \sum_{\ell=1}^L \widehat{Y}_{\ell,N_\ell}^{\mathrm{MC}} \ge M_Z - C \sum_{\ell=1}^L h_\ell^2,$$

with a constant $C$ independent of $\lbrace h\_\ell \rbrace$. Thus, if $h\_0$ is sufficiently small, such that $\sum\_{\ell=1}^L h\_\ell^2 < M\_Z / C$, we also have $\big\|\widehat{Q}\_{q,h,\mathrm{ML}}^{\mathrm{RE}}\big\| < \infty$.

</details>
</div>

For $\widehat{Q}\_{q,h,\mathrm{ML}}^{\mathrm{RE}}$, a similar result can also be proved for the case of a lognormal PDE coefficient $a$ (see again [Scheichl, Stuart, Teckentrup, 2017]).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.5.9</span><span class="math-callout__name">(Continuation of Example 5.4.12 — Ratio Estimators in the Lognormal Case)</span></p>

For a numerical comparison we return again to the lognormal case of the elliptic PDE with Matérn covariance on $D = (0, 1)^2$ discretised by piecewise linear FEs. However, in the following experiments we choose $\nu = 1/2$, $\sigma^2 = 1$ and $\lambda = 0.3$, which is a significantly harder case than the one considered in Example 5.4.12. Due to the low regularity, in that case it is only possible to prove

$$\lVert Bu - Bu_h \rVert_{L^p(\Omega; \mathbb{R}^m)} \le Ch$$

for any bounded linear functional $B : H\_0^1(D) \to \mathbb{R}^m$. Thus, the assumptions in Section 5.4 only hold with $\alpha = 1$, $\beta = 2$ and $\gamma = 2$, and the theoretically expected $\varepsilon$-costs are $\mathcal{O}(\varepsilon^{-4})$, $\mathcal{O}(\varepsilon^{-3})$ and $\mathcal{O}(\varepsilon^{-2})$ for $\mathrm{typ} = \mathrm{MC}$, QMC and ML, respectively.

We consider the case of $f \equiv 0$ and mixed boundary conditions, such that

$$u(x) = 1 \text{ for } x_1 = 0, \quad u(x) = 0 \text{ for } x_1 = 1, \quad \text{and} \quad \frac{\partial u}{\partial x_2}(x) = 0 \text{ on the rest of the boundary},$$

leading to a flow of heat (or fluid) from $x\_1 = 0$ to $x\_1 = 1$. The quantity of interest is the outflow over the boundary at $x\_1 = 1$, which can be computed as

$$Q_h = Hu_h = -\int_D a(x, \omega) \nabla u_h(x, \omega)^\top \nabla w_h(x) \, \mathrm{d}x,$$

for a suitably chosen weight function $w\_h$ with $w\_h = 0$ on $x\_1 = 0$ and $w\_h = 1$ on $x\_1 = 1$. The observation functional $B : H\_0^1(D) \to \mathbb{R}^m$ consists of $m$ local averages of the PDE solution $u$ at $m$ uniformly distributed points in $D$. The data $y \in \mathbb{R}^m$ is generated synthetically from a reference solution with $h^\ast = 1/256$, adding noise in the form of a realisation of $E \sim \mathcal{N}(0, \Sigma)$ with $\Sigma = \sigma\_E^2 I$. For more details see [Scheichl, Stuart, Teckentrup, 2017].

The numerical results (for $h = 1/16, \ldots, 1/256$, $m = 9$ and $\sigma\_E^2 = 0.09$) show the following:

* The measured $\varepsilon$-costs of the three ratio estimators attain the predicted rates $\varepsilon^{-4}$, $\varepsilon^{-3}$, $\varepsilon^{-2}$. One can distinguish *dependent* estimators, where the same random samples are used in $\widehat{Q}\_{w,h}^{\mathrm{typ}}$ and $\widehat{Z}\_h^{\mathrm{typ}}$, from *independent* estimators, where different random samples are used; the dependent variants perform better.
* The discretisation errors and the MC sampling errors of the numerator $\widehat{Q}\_{w,h}$, the denominator $\widehat{Z}\_h$ and the ratio estimate itself all converge with the same (predicted) rates, but **the error of the ratio estimate is several orders of magnitude bigger**. This is due to the factor $Z^{-2}$ on the right hand side of (5.5.8).
* Note that $Z \to 0$ as $\sigma\_E^2 \to 0$ (small-noise limit) or as $m \to \infty$ (large-data limit). This blow-up is clearly visible in the measured asymptotic variance of the ratio estimators as functions of $\sigma\_E^2$ and $m$; the growth for the independent ratio estimators is significantly faster than for the dependent ones.

</div>

#### 5.5.3 Data-Informed Importance Distributions -- Preconditioning

The lack of robustness of the ratio estimator with respect to the prior density $q = \pi\_X$ observed in Example 5.5.9 can be clearly seen when considering again the scaled posterior log-likelihood $n\Psi\_n(x)$ with $\Psi\_n$ defined in (5.3.5). In the small noise limit, $Z = \mathbb{E}\_q[w\_u(X)] = \int\_{\mathbb{R}^s} \exp\left(-\frac{n}{2}\lVert y - \Phi(x) \rVert\_\Sigma^2\right) \pi\_X(x) \, \mathrm{d}x \to 0$ as $n \to \infty$, and so the bound on the MSE of the ratio estimator in Lemma 5.5.5 explodes with $n \to \infty$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.5.10</span><span class="math-callout__name">(Prior-Based Ratio Estimator Variance Explosion)</span></p>

*(Schillings, Sprungk, Wacker, 2020).* For a RV $X : \Omega \to \mathbb{R}^s$ and a sufficiently smooth and measurable $F : \mathbb{R}^s \to \mathbb{R}$, consider the scaled posterior log-likelihood $n\Psi_n(x)$ with $\Psi_n$ defined in (5.3.5). Under the assumptions of Theorem 5.3.3 with $p = \pi_{X\mid Y}$ and $q = \pi_X$, there exist $0 < c < C$ such that the asymptotic variance $\sigma_q^2$ of $\widehat{Q}\_{q,N}^{\mathrm{RE}}$ satisfies

$$cn^{s/2} \mathbb{V}_p(F(X)) \le \sigma_q^2 \le Cn^{s/2} \mathbb{V}_p(F(X)).$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Where the $n^{s/2}$ blow-up comes from — a geometric picture)</span></p>

Lemma 5.5.10 looks like a technical variance estimate, but the mechanism is simple enough to reconstruct on a napkin, and doing so explains both the exponent and its dependence on the dimension $s$:

* As $n \to \infty$, the posterior $p = \pi\_{X \mid Y}$ concentrates in an ellipsoid of radius $\sim n^{-1/2}$ *in each of the $s$ directions* around $x\_{\mathrm{MAP},n}$ (this is the Laplace/Bernstein-von-Mises picture from Section 5.3). Its "volume" therefore scales like $n^{-s/2}$.
* Samples are drawn from the **prior**, which stays put. The probability that a prior sample lands inside the region where the posterior actually lives is $\sim \pi\_X(x\_{\mathrm{MAP}}) \cdot n^{-s/2}$; all other samples receive (relatively) negligible weight $w\_u = p\_u/q\_u$.
* So out of $N$ prior samples, only about $N n^{-s/2}$ are "effective" — the effective sample size collapses, and since the MSE of an average scales inversely with the effective number of samples, the variance is inflated by the reciprocal factor $n^{s/2}$. That is exactly the lemma.

Two consequences: the failure is **exponential in the parameter dimension** (for fixed $n$, doubling $s$ squares the penalty), and it is *not* fixed by taking more samples — $N$ would have to grow like $n^{s/2}$, which is precisely the curse of dimensionality that sampling methods were supposed to avoid. The fix has to change $q$, not $N$: move the importance distribution to where the posterior mass is. That is the sense in which the Laplace approximation below acts as a **preconditioner** — same estimator, same rates, but the constant $Z = \mathbb{E}\_q[w\_u]$ is pushed from $\approx 0$ back to $\approx 1$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/is_weight_degeneracy.png' | relative_url }}" alt="Left: the broad shaded prior density and three posterior densities that become dramatically narrower as n grows from 1 to 100; below the axis, 80 prior samples drawn as dots whose size encodes their importance weight at n = 100 — essentially a single large dot near the posterior mode carries all the weight. Right: log-log plot of effective sample size over N against n; the prior-based curve collapses along a reference line of slope minus one half while the Laplace-based curve rises towards one and stays there." loading="lazy">
</figure>

*Weight degeneracy in one picture (1D model $\Phi(x) = e^x$, $y = e$, noise variance $1/n$). Left: the posterior concentrates as $n$ grows while the prior stays put; of $80$ prior samples (dots, sized by their importance weight at $n = 100$) essentially one carries all the weight. Right: the effective sample size $\mathrm{ESS}/N$ of prior-based weights collapses at exactly the predicted rate $n^{-s/2}$ (here $s = 1$), while for Laplace-based importance sampling $\mathrm{ESS}/N \to 1$ — concentration* helps *it, in line with Theorem 5.5.11.*

Let us now instead consider as the importance distribution the Laplace approximation of the posterior, i.e. $q_u$ is the unnormalised density of $\mathcal{L}\_{\mu_{X\mid y}}$. It can be shown using Theorem 5.3.3 that

$$Z = \mathbb{E}_q[w_u(X)] = \mathbb{E}_q\left[\frac{p_u(X)}{q_u(X)}\right] \to 1 \quad \text{as } n \to \infty. \tag{5.5.9}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.5.11</span><span class="math-callout__name">(Laplace-Based Ratio Estimator Convergence)</span></p>

*(Schillings, Sprungk, Wacker, 2020).* Under the assumptions of Lemma 5.5.10 with $p = \pi_{X\mid Y}$ and $q$ the density of the Laplace approximation $\mathcal{L}\_{\mu_{X\mid y}}$, for any $N \in \mathbb{N}$ and $\delta \in [0, 1/2)$,

$$n^\delta \left| \widehat{Q}_{q,N}^{\mathrm{RE}} - \mathbb{E}_p[F(X)] \right| \xrightarrow{\mathbb{P}}{N \to \infty} 0,$$

i.e. the error of the Laplace-based ratio estimator converges in probability to zero as $n \to \infty$, independently of the sample size $N$ and with a rate arbitrarily close to $n^{-1/2}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.5.12</span><span class="math-callout__name">(Robustness of the Laplace-Based Ratio Estimator)</span></p>

The following numerical experiment for the elliptic (P)DE on $(0, 1)$ (i.e. for $d = 1$), with $u = 0$ at $x = 0$ and $x = 1$ and $f(x) = 100x$, is taken from [Schillings, Sprungk, Wacker, 2020]. It is a toy example with uniform $a$, with $\Xi \sim \mathrm{uniform}(-1, 1)^s$, for $s = 1, 2, 3$, and

$$\ell_j \varphi_j(x) = (10j)^{-1} \sin(j\pi x).$$

The data are $m = 2$ (resp. 7) measurements $y\_k = u(x\_k^\ast)$ of the solution at equally spaced points $x\_k^\ast \in (0, 1)$ for $s = 1, 2$ (resp. 3), with measurement noise $E\_n \sim \mathcal{N}\big(0, (100n)^{-1} I\big)$ for $n \in \mathbb{N}$. The quantity of interest is $Q = u(0.5)$.

Estimating the root mean square error of $\widehat{Q}\_{q,\mathrm{QMC}}^{\mathrm{RE}}$ (a randomised lattice rule with 8192 points, averaged over 64 random shifts) as a function of the noise level over many orders of magnitude ($n = 10^2, \ldots, 10^{10}$) reveals exactly the dichotomy predicted by Lemma 5.5.10 and Theorem 5.5.11: for the **prior-based** estimator ($q = \pi\_X$) the RMSE deteriorates dramatically as the noise level decreases — and the more so the larger $s$ — while for the **Laplace-based** importance distribution the RMSE remains uniformly small, in fact slightly *improving* as $n \to \infty$.

</div>

It is also possible to use other preconditioners. For example, one can use **TT-cross approximations**, i.e. low-rank tensor approximations of the unnormalised posterior density $p\_u \propto \pi\_{X\mid Y}$, as the unnormalised importance distribution $q\_u$ [Dolgov, Anaya-Izquierdo, Fox & Scheichl, 2020]. By increasing the ranks in the low-rank approximation, it is possible to make $w\_u(x)$ arbitrarily close to 1. This will be discussed in more detail in Section 5.7.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.5.13</span><span class="math-callout__name">(TT-Preconditioned Ratio Estimators vs. MCMC)</span></p>

Here, we just show how the TT-cross approximation improves the efficiency of the ratio estimator for the elliptic PDE over a prior-based ratio estimator and how it compares to MCMC-based estimators (more details on those in Section 5.6). The setup is almost identical to that in Example 5.5.9, except that $\log a$ is modelled as a (Karhunen-Loève like) expansion with independent *uniform* instead of independent Gaussian coefficients. The observation operator $\Phi : H\_0^1(D) \to \mathbb{R}^m$ and the quantity of interest are the same. We use the same randomised lattice rule, $m = 9$ measurements and a noise $E \sim \mathcal{N}\big(0, \frac{1}{100} I\big)$. For details see [Dolgov et al., 2020].

Comparing the relative sampling errors of various estimators, plotted against the number of samples and also against CPU time — in particular, the TT-based and the prior-based ratio estimators with QMC rules, qRE(TT) and qRE(prior) resp., against three Markov chain Monte Carlo estimators: DRAM [Haario et al, 2001], MALA [Roberts, Tweedie, 1996] and a Metropolis-Hastings algorithm with independent proposals drawn from the TT approximation of the posterior distribution, MetH(TT) — one observes:

* the better rate of convergence of almost $\mathcal{O}(N^{-1})$ for the QMC-based ratio estimators, compared to the $\mathcal{O}(N^{-1/2})$ of the MCMC estimators, and
* how much the TT-based preconditioning helps, both in the case of the ratio estimator and in the MCMC case: qRE(TT) reaches the discretisation-error level orders of magnitude faster than all other methods.

</div>

### 5.6 The Markov Chain Monte Carlo Method

The basic idea of this method is to compute a sequence of RVs $(X_j)\_{j \in \mathbb{N}}$, such that

$$X_j \xrightarrow{d}{j \to \infty} X \sim \mu^{X|y} \tag{5.6.1}$$

Under appropriate conditions it is possible in this setting to prove a strong law of large numbers and a central limit theorem (as for iid samples in Prop. 5.4.1), in particular if the sequence of RVs comes from a **Markov chain** (a well-studied class of **time-discrete stochastic processes**).

#### 5.6.1 Basic Concepts of Markov Chain Theory

Let $H$ be a separable Hilbert space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.6.1</span><span class="math-callout__name">(Markov Chain)</span></p>

A **Markov chain** in $H$ is a sequence of $H$-valued RVs $X_j : \Omega \to H$, satisfying the **Markov property**, i.e.,

$$\mathbb{P}(X_{j+1} \in A | X_1, \ldots, X_j) = \mathbb{P}(X_{j+1} \in A | X_j) \quad \mathbb{P}\text{-a.s.,}$$

for each $j \in \mathbb{N}$ and $A \in \mathcal{B}(H)$ (i.e., the state $X_{j+1}$ of a Markov chain only depends on the previous state $X_j$ and not on the entire history of the chain).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.6.2</span><span class="math-callout__name">(Transition Kernel, Homogeneous Markov Chain)</span></p>

**(a)** A map $K : H \times \mathcal{B}(H) \to [0, 1]$ is called **transition** or **Markov kernel** if
1. for all $x \in H$, $K(x, \cdot)$ is a probability measure on $(H, \mathcal{B}(H))$, and
2. for all $A \in \mathcal{B}(H)$, $K(\cdot, A)$ is a measurable functional from $H$ to $[0, 1]$.

**(b)** A Markov chain $(X_j)\_{j \in \mathbb{N}}$ is **homogeneous**, if there exists a transition kernel $K$, such that for all $j \in \mathbb{N}$, $x \in H$ and $A \in \mathcal{B}(H)$: $K(x, A) = \mathbb{P}(X_{j+1} \in A \mid X_j = x)$ $\mathbb{P}$-a.s.

</div>

We will only consider homogeneous Markov chains. We therefore introduce the following notions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.6.3</span><span class="math-callout__name">(Action of Transition Kernel)</span></p>

Let $K$ be a Markov kernel on $H$ and let $\nu \in \mathcal{P}(H)$. Then we denote by $\nu K$ the probability measure on $(H, \mathcal{B}(H))$ given by

$$(\nu K)(A) := \int_H K(x, A) \nu(\mathrm{d}x) \quad \text{for all } A \in \mathcal{B}(H). \tag{5.6.2}$$

Moreover, we define recursively, for $j \in \mathbb{N}$, the Markov kernel $K^j$ on $H$ by

$$K^j(x, A) := \int_H K^{j-1}(x', A) K(x, \mathrm{d}x') \quad \text{for all } x \in H, \ A \in \mathcal{B}(H). \tag{5.6.3}$$

</div>

In this notation, the distribution of the $j$th state $X_j$ of a Markov chain with transition kernel $K$ and initial distribution $X_1 \sim \nu$ is simply $X_j \sim \nu K^{j-1}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.6.4</span><span class="math-callout__name">(Why $\nu K$ and not $K \nu$?)</span></p>

The definition of $\nu K$ is an abuse of notation, since $K$ denotes a Markov kernel but in $\nu K$ it plays the role of a mapping from $\mathcal{P}(H)$ to $\mathcal{P}(H)$. Also, it seems odd to place $\nu$ on the *left* hand side of $K$ instead of writing $K\nu$. The reason for this is that, in the special case of a discrete state space $H$ with $\|H\| = M$ — where Markov chains have been studied first — the transition kernel $K$ is simply a **(row) stochastic matrix** $K \in [0, 1]^{M \times M}$ (row $i$ holds the distribution of the next state given current state $i$), and thus for a row vector $\nu \in [0, 1]^M$ of initial probabilities, the vector given by $\nu K$ describes the distribution of the next state of the Markov chain.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.6.5</span><span class="math-callout__name">(Invariant Measure, Reversibility)</span></p>

**(a)** Let $\mu \in \mathcal{P}(H)$ and let $K$ be the transition kernel of a Markov chain $(X_j)\_{j \in \mathbb{N}}$ in $H$. The measure $\mu$ is called an **invariant measure** of the Markov chain or **invariant** with respect to $K$ if

$$\mu = \mu K \tag{5.6.4}$$

**(b)** The transition kernel $K$ and the corresponding Markov chain $(X_j)\_{j \in \mathbb{N}}$ are called **$\mu$-reversible** if they satisfy the so-called **detailed balance condition** for all $x, x' \in H$, i.e.,

$$K(x, \mathrm{d}x') \mu(\mathrm{d}x) = K(x', \mathrm{d}x) \mu(\mathrm{d}x'), \tag{5.6.5}$$

where equality holds in the sense of measures on $H \times H$.

</div>

Reversibility means that provided $X\_j \sim \mu$ the jump from $X\_j = x$ to $X\_{j+1} = x'$ has the same probability as the reverse jump from $X\_j = x'$ to $X\_{j+1} = x$, and (5.6.5) is equivalent to

$$\mathbb{P}(X_j \in A, X_{j+1} \in B) = \mathbb{P}(X_j \in B, X_{j+1} \in A), \quad \text{for all } A, B \in \mathcal{B}(H).$$

This property is easier to verify than invariance (5.6.4) itself and we have the following result.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.6.6</span><span class="math-callout__name">(Reversibility Implies Invariance)</span></p>

Let $\mu \in \mathcal{P}(H)$ and let $K : H \times \mathcal{B}(H) \to [0, 1]$ be a $\mu$-reversible transition kernel. Then $\mu$ is invariant with respect to $K$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 5.6.6</summary>

Let $A \in \mathcal{B}(H)$. Then, it follows from the detailed balance condition (5.6.5) that

$$\begin{aligned}
(\mu K)(A) = \int_H K(x, A) \, \mu(\mathrm{d}x) &= \int_H \int_A \underbrace{K(x, \mathrm{d}x') \, \mu(\mathrm{d}x)}_{= K(x', \mathrm{d}x)\,\mu(\mathrm{d}x')} = \int_H \int_A K(x', \mathrm{d}x) \, \mu(\mathrm{d}x') \\
&= \int_A \int_H K(x', \mathrm{d}x) \, \mu(\mathrm{d}x') = \int_A \underbrace{K(x', H)}_{=1} \, \mu(\mathrm{d}x') = \int_A 1 \, \mu(\mathrm{d}x') = \mu(A).
\end{aligned}$$

</details>
</div>

If the transition kernel $K : H \times \mathcal{B}(H) \to [0, 1]$ of a Markov chain is understood as a linear operator from $\mathcal{P}(H)$ to $\mathcal{P}(H)$, then (5.6.4) simply means that $\mu$ is a **fixed point** of $K$ and the Markov chain is a fixed point iteration. Classical convergence results for Markov chains rely on this point of view: they show that $K$ is a contraction and apply the Banach Fixed Point Theorem. However, instead we now introduce a notion of geometric convergence of Markov chains to their invariant distribution.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.6.7</span><span class="math-callout__name">(Geometric Ergodicity)</span></p>

A Markov chain $(X_j)\_{j \in \mathbb{N}}$ in $H$ with transition kernel $K$ is $L^2\_\mu(H)$**-geometrically ergodic** if there exists a number $r \in [0, 1)$ such that for any probability measure $\nu$ which has a density $\frac{\mathrm{d}\nu}{\mathrm{d}\mu} \in L^2_\mu(H)$ w.r.t. $\mu$

$$D_{\mathrm{TV}}(\nu K^j, \mu) \le C_\nu r^j \quad \text{for all } j \in \mathbb{N}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.6.8</span><span class="math-callout__name">(Discrete Markov Chain and Spectral Gap)</span></p>

Consider a Markov chain in a discrete state space with $M$ states and a row-stochastic matrix $K \in [0, 1]^{M \times M}$ representing its transition kernel. Then, $\sum_{j=1}^{M} K_{ij} = 1$ for all $i = 1, \ldots, M$, and the vector of all ones $e$ is a right eigenvector of $K$ to the eigenvalue 1. The corresponding left eigenvector $\mu$ is the invariant measure. If all entries of $K$ are strictly between 0 and 1 then $K$ is called **irreducible** and it follows from the **Perron-Frobenius Theorem** that 1 is in fact **dominant**, i.e. a simple eigenvalue strictly larger in modulus than all other eigenvalues of $K$.

Let the distribution of the initial state of the Markov chain be $\nu \in \mathbb{R}^M$. Then, the distribution $\nu\_j := \nu K^{j-1}$ of the $j$th state represents the $j$th iterate of the **power method** to find the eigenvector corresponding to the dominant eigenvalue of $K$, normalised such that $e^\top \nu\_j = 1$. It is easy to see that the power method converges geometrically to $\mu$:

Let $\lambda\_1, \ldots, \lambda\_M$ be the eigenvalues of $K$ with $1 = \lambda\_1 > \|\lambda\_2\| \ge \|\lambda\_3\| \ge \ldots \ge \|\lambda\_M\|$ and corresponding left eigenvectors $\mu = v\_1, v\_2, \ldots, v\_M$. They form a basis of $\mathbb{R}^M$ and thus $\nu = \sum\_{m=1}^M \alpha\_m v\_m$ for some $\alpha\_1, \ldots, \alpha\_M \in \mathbb{R}$. If we assume that $\alpha\_1 > 0$, then we see easily that

$$\nu_{j+1} = \frac{\nu K^j}{\lVert \nu K^j \rVert_1} = \frac{\sum_{m=1}^M \lambda_m^j \alpha_m v_m}{\sum_{m=1}^M |\lambda_m|^j |\alpha_m|} = \frac{\alpha_1 \mu + \sum_{m=2}^M \lambda_m^j \alpha_m v_m}{|\alpha_1| \left( 1 + \sum_{m=2}^M |\lambda_m|^j |\alpha_m / \alpha_1| \right)} \ \longrightarrow \ \mu \quad \text{as} \quad j \to \infty,$$

since $\|\lambda\_m\| < 1$ for $m \ge 2$. The denominator can be bounded below by $\alpha\_1$. Thus,

$$\lVert \nu_{j+1} - \mu \rVert_1 \le \sum_{m=2}^M |\lambda_m|^j \left| \frac{\alpha_m}{\alpha_1} \right| \le \left( \sum_{m=2}^M \left| \frac{\alpha_m}{\alpha_1} \right| \right) |\lambda_2|^j$$

and the Markov chain converges geometrically with rate $r = \|\lambda\_2\|$, the modulus of the second largest eigenvalue.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.6.9</span></p>

This link between the spectrum of the transition kernel and the geometric ergodicity of the Markov chain is also a key concept in the convergence analysis of Markov chains on general state spaces, leading to the concept of the so-called **spectral gap**. For Markov chains in continuous state spaces, such as $\mathbb{R}^n$, we can distinguish between geometric ergodicity as in Def. 5.6.7 with a constant $C_\nu$ that depends on the initial distribution $\nu$ and **uniform ergodicity** where there exists a (uniform) constant $C < \infty$ for all initial distributions $\nu$.

</div>

If the distribution of $X_j$ converges to $\mu$, then the Markov chain $(X_j)\_{j \in \mathbb{N}}$ can be used for approximate sampling from $\mu$, leading to the very powerful concept of **Markov chain Monte Carlo** methods for the computation of expectations. In particular, the expectation $\mathbb{E}\_\mu[F(X)]$ of a function $F : H \to \mathbb{R}$ of $X$ w.r.t. $\mu$ can be approximated by

$$\widehat{Q}_{N, N_0}^{\mathrm{MCMC}} := \frac{1}{N} \sum_{j=1}^{N} F(X_{j+N_0}), \tag{5.6.6}$$

where $N$ is the sample size and $N_0$ is a so-called **burn-in parameter** to decrease the influence of the initial distribution. In fact, a strong law of large numbers and also a central limit theorem hold for $\widehat{Q}\_{N, N_0}^{\mathrm{MCMC}}$ under appropriate assumptions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.6.10</span><span class="math-callout__name">(Central Limit Theorem for Reversible Markov Chains)</span></p>

Let $(X_j)\_{j \in \mathbb{N}}$ be a $\mu$-reversible and $L^2_\mu(H)$-geometrically ergodic Markov chain and let $F$ be $\mu$-measurable. Then

$$\sqrt{N}\Big(\widehat{Q}_{N, N_0}^{\mathrm{MCMC}} - \mathbb{E}_\mu[F(X)]\Big) \xrightarrow{d}{N \to \infty} \mathcal{N}(0, \sigma_F^2)$$

where $\sigma_F^2 := \lim_{N \to \infty} N \mathbb{V}\Big(\widehat{Q}\_{N, N_0}^{\mathrm{MCMC}}\Big)$ denotes the **asymptotic variance**, which in this case satisfies

$$\sigma_F^2 := \mathbb{V}(F(X_1)) + 2\sum_{j=1}^{\infty} \mathrm{cov}(F(X_1), F(X_{1+j})) < \infty. \tag{5.6.7}$$

</div>

The asymptotic variance $\sigma_F^2$ includes not only the variance of $F(X_1)$ but also the autocovariances $\mathrm{cov}(F(X_1), F(X_{1+j}))$, reflecting the fact that consecutive samples in a Markov chain are correlated.

A proof of Theorem 5.6.10 is beyond the scope of this course, but we can motivate the specific form (5.6.7) of the asymptotic variance. To do this, let us assume that $X\_1 \sim \mu$ (w.l.o.g. with $N\_0 = 0$). We have

$$\begin{aligned}
\mathbb{V}\Big(\widehat{Q}_{N,N_0}^{\mathrm{MCMC}}\Big) = \mathbb{V}\left( \frac{1}{N} \sum_{j=1}^N F(X_{j+N_0}) \right)
&= \frac{1}{N^2} \sum_{j=1}^N \sum_{k=1}^N \mathrm{cov}\big( F(X_{j+N_0}), F(X_{k+N_0}) \big) \\
&= \frac{1}{N^2} \sum_{j=1}^N \mathbb{V}\big( F(X_{j+N_0}) \big) + \frac{1}{N^2} \sum_{j \neq k} \mathrm{cov}\big( F(X_{j+N_0}), F(X_{k+N_0}) \big).
\end{aligned}$$

Since $(X\_j)\_{j \in \mathbb{N}}$ is assumed to be $\mu$-reversible and $X\_1 \sim \mu$, then $X\_j \sim \mu$ for any $j \in \mathbb{N}$, which further implies that $(X\_j, X\_{j+k})$ follows the same distribution as $(X\_1, X\_{1+k})$, for all $j, k \in \mathbb{N}$. Hence,

$$\mathbb{V}(F(X_j)) = \mathbb{V}(F(X_1)), \qquad \mathrm{cov}\big( F(X_{j+N_0}), F(X_{k+N_0}) \big) = \mathrm{cov}\big( F(X_1), F(X_{1+|j-k|}) \big),$$

and thus

$$\mathbb{V}\Big(\widehat{Q}_{N,N_0}^{\mathrm{MCMC}}\Big) \approx \frac{1}{N} \mathbb{V}_\mu\big(F(X_1)\big) + \frac{2}{N} \sum_{j=1}^N \mathrm{cov}_\mu\big( F(X_1), F(X_{1+j}) \big).$$

Of course, the assumption $X\_1 \sim \mu$ is rather academic and, in general, not given in practice. However, since the Markov chain in Theorem 5.6.10 is assumed to be $L^2\_\mu(H)$-geometrically ergodic, the distribution of its $j$th state $X\_j$ converges exponentially fast to $\mu$ as $j \to \infty$ — this is also the justification for the burn-in parameter $N\_0$: discarding the first $N\_0$ states makes the "warm start" assumption approximately true at exponentially small cost in bias.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(How to read $\sigma\_F^2$ — the price of correlation)</span></p>

Comparing Theorem 5.6.10 with the iid CLT in Proposition 5.4.1 isolates exactly what MCMC costs us: the estimator still converges at rate $N^{-1/2}$, but the variance constant is inflated from $\mathbb{V}\_\mu(F(X))$ to $\sigma\_F^2 = \mathbb{V}\_\mu(F(X)) + 2\sum\_{j \ge 1} \mathrm{cov}\_\mu(F(X\_1), F(X\_{1+j}))$. The ratio

$$\mathrm{IACT}_F := \frac{\sigma_F^2}{\mathbb{V}_\mu(F(X))} = 1 + 2\sum_{j=1}^\infty \mathrm{corr}\big(F(X_1), F(X_{1+j})\big)$$

is called the **integrated autocorrelation time**: $N$ correlated MCMC samples are statistically worth only $N / \mathrm{IACT}\_F$ iid samples. This single number is the right figure of merit for comparing MCMC algorithms, and it is controlled by how fast the autocorrelations decay — i.e., by the geometric ergodicity rate $r$ (equivalently, the spectral gap $1 - r$ of the transition kernel, cf. Example 5.6.8 and Remark 5.6.9). Everything in the remainder of this chapter — step-size tuning, pCN, MALA, adaptive and multilevel MCMC — is ultimately an attempt to make $\mathrm{IACT}\_F$ small (and, for infinite-dimensional problems, *bounded in the discretisation dimension*; see Example 5.6.20).

</div>

#### 5.6.2 The Metropolis-Hastings Markov Chain Monte Carlo Method

We will now describe a generic algorithm to (approximately) sample from distributions $\mu$ that are difficult to sample from directly, e.g. because they are only known in unnormalised form, such as the posterior distribution $\mu_{X\mid y}$ in a Bayesian inverse problem. It is based on a Markov chain of proposals and rejection sampling, and was first proposed in 1953 by Metropolis and co-authors before being generalised in 1970 by Hastings.

We focus again first on finite dimensional $H = \mathbb{R}^n$. Let $\mu \in \mathcal{P}(\mathbb{R}^n)$ with $\mu(\mathrm{d}x) \propto p(x) \, \mathrm{d}x$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.6.11</span><span class="math-callout__name">(Proposal Distribution)</span></p>

Let $Q : \mathbb{R}^n \times \mathcal{B}(\mathbb{R}^n) \to [0, 1]$ be a Markov kernel on $\mathbb{R}^n$ with a transition density $q : \mathbb{R}^n \times \mathbb{R}^n \to [0, \infty)$ such that

$$Q(x, A) = \int_A q(x, x') \, \mathrm{d}x' \quad \text{for all } A \in \mathcal{B}(\mathbb{R}^n).$$

The Markov kernel $Q$ is called the **proposal kernel**.

</div>

Given this proposal kernel we can now define the **Metropolis-Hastings algorithm** that allows to sample from the target distribution $\mu$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 1</span><span class="math-callout__name">(Metropolis-Hastings Algorithm)</span></p>

**Input:** Proposal kernel $Q$ with transition density $q$, initial distribution $\nu \in \mathcal{P}(\mathbb{R}^n)$.
**Output:** Realisations $(x_j)\_{j \in \mathbb{N}}$ of a Markov chain $(X_j)\_{j \in \mathbb{N}}$.

1. Draw a realisation $x_1 \sim \nu$.
2. **for** $j = 1, 2, \ldots$ **do**
3. &emsp; Given the current state $X_j = x_j$, draw a realisation $x'$ from $Q(x_j, \cdot)$.
4. &emsp; Compute the acceptance probability

   $$\alpha(x_j, x') := \min\left(1, \frac{p(x') \, q(x', x_j)}{p(x_j) \, q(x_j, x')}\right). \tag{5.6.8}$$

5. &emsp; Draw an independent sample $u_{j+1} \sim \mathrm{uniform}[0, 1]$ and set

   $$x_{j+1} = \begin{cases} x', & \text{if } u_{j+1} \le \alpha(x_j, x'), \\ x_j, & \text{otherwise.} \end{cases}$$

6. **end for**

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.6.12</span><span class="math-callout__name">(Metropolis Kernel is $\mu$-Reversible)</span></p>

The transition kernel $K : \mathbb{R}^n \times \mathcal{B}(\mathbb{R}^n) \to [0, 1]$ of the Markov chain $(X_j)\_{j \in \mathbb{N}}$ produced by Algorithm 1 with proposal kernel $Q$ and acceptance probability $\alpha$ in (5.6.8) is given by

$$K(x, \mathrm{d}x') = \alpha(x, x') Q(x, \mathrm{d}x') + \left(1 - \int_{\mathbb{R}^n} \alpha(x, x'') \, Q(x, \mathrm{d}x'') \right) \delta_x(\mathrm{d}x'), \tag{5.6.9}$$

where $\delta_x \in \mathcal{P}(\mathbb{R}^n)$ denotes the *Dirac-measure* at $x \in \mathbb{R}^n$. The **Metropolis kernel** $K$ is $\mu$-reversible, and thus $\mu$ is invariant with respect to $K$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 5.6.12</summary>

We first show that the Metropolis kernel $K$ is of the form (5.6.9). By definition $K(x, A) = \mathbb{P}(X\_{j+1} \in A \mid X\_j = x)$. We only consider proposal kernels $Q$ with smooth density $q$ and note that in that case the probability that $x' = x$, i.e. that we propose $x$ given $X\_j = x$, is zero. In that case it suffices to study the cases $\mathbb{P}(X\_{j+1} = x \mid X\_j = x)$, i.e., the probability that the proposal is rejected, and $\mathbb{P}(X\_{j+1} \in A \mid X\_j = x)$ for $x \notin A$, i.e., the proposal $x' \in A$ and $x'$ is accepted.

The rejection probability for a proposal is exactly $1 - \alpha(x, x')$ with $x' \sim Q(x, \mathrm{d}x')$. Thus,

$$\mathbb{P}(X_{j+1} = x \mid X_j = x) = \int_{\mathbb{R}^n} \big( 1 - \alpha(x, x') \big) \, Q(x, \mathrm{d}x') = 1 - \int_{\mathbb{R}^n} \alpha(x, x') \, Q(x, \mathrm{d}x').$$

On the other hand, the probability $\mathbb{P}(X\_{j+1} \in A \mid X\_j = x)$ for $x \notin A$ is

$$\mathbb{P}(X_{j+1} \in A \mid X_j = x) = \int_A \alpha(x, x') \, Q(x, \mathrm{d}x').$$

Combining these two cases we obtain (5.6.9).

To show detailed balance, we consider first $A, B \in \mathcal{B}(\mathbb{R}^n)$ with $A \cap B = \emptyset$. W.l.o.g. we can assume that $p(x)q(x, x') > 0$ for all $x, x' \in \mathbb{R}^n$ (otherwise we simply have to restrict the integrations below accordingly). Writing $\mu(\mathrm{d}x) = \frac{1}{c} p(x) \, \mathrm{d}x$ for the (unknown) normalisation constant $c$, and using that on $A \times B$ no rejection term contributes (since $A \cap B = \emptyset$), we have

$$\begin{aligned}
\int_{A \times B} K(x, \mathrm{d}x') \, \mu(\mathrm{d}x) &= \int_A \int_B \alpha(x, x') \, Q(x, \mathrm{d}x') \, \mu(\mathrm{d}x) \\
&= \frac{1}{c} \int_A \int_B \min\left( 1, \frac{p(x') \, q(x', x)}{p(x) \, q(x, x')} \right) p(x) \, q(x, x') \, \mathrm{d}x' \, \mathrm{d}x \\
&= \frac{1}{c} \int_A \int_B \min\big( p(x) \, q(x, x'), \; p(x') \, q(x', x) \big) \, \mathrm{d}x' \, \mathrm{d}x \\
&= \frac{1}{c} \int_A \int_B \min\left( \frac{p(x) \, q(x, x')}{p(x') \, q(x', x)}, \, 1 \right) p(x') \, q(x', x) \, \mathrm{d}x' \, \mathrm{d}x \\
&= \int_A \int_B \alpha(x', x) \, \mu(\mathrm{d}x') \, Q(x', \mathrm{d}x) = \int_{A \times B} K(x', \mathrm{d}x) \, \mu(\mathrm{d}x').
\end{aligned}$$

If $A \cap B \neq \emptyset$ we also need to consider rejections, but detailed balance can again be shown similarly, since $\mathbb{P}(X\_{j+1} = x \mid X\_j = x)$ is clearly symmetric.

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The two design miracles of Metropolis-Hastings)</span></p>

Two features of Algorithm 1 are so fundamental that they deserve to be stated separately:

* **Only ratios of $p$ are ever evaluated.** The acceptance probability (5.6.8) contains $p(x')/p(x\_j)$, so any unknown normalisation constant cancels. This is *the* property that makes MH applicable to Bayesian inverse problems, where the posterior density is only available in unnormalised form — the intractable evidence $Z$ never needs to be computed. (Compare: the ratio estimator of Section 5.5 had to *estimate* $Z$ and paid dearly through the $Z^{-2}$ factor.)
* **The min in (5.6.8) is exactly what detailed balance forces.** In the proof above, the key identity is the symmetry of $\min\big( p(x)q(x, x'), \, p(x')q(x', x) \big)$ in $(x, x')$. Any acceptance rule of the form $\alpha(x, x') = g\big( p(x')q(x', x) / (p(x)q(x, x')) \big)$ with $g(t) \le \min(1, t)$ and $g(t) = t \, g(1/t)$ would also give a $\mu$-reversible chain, but Peskun's ordering shows the Metropolis choice $g(t) = \min(1, t)$ is optimal among these: it accepts as often as possible, which minimises the asymptotic variance $\sigma\_F^2$.

Note also the structure of the Metropolis kernel (5.6.9): it is a *mixture* of a continuous part (accepted moves, density $\alpha \cdot q$) and an atom at the current state $x$ (rejections). The chain is therefore never absolutely continuous — every path contains repeated states, and these repetitions are not wasted: they are precisely the reweighting that corrects the proposal distribution towards $\mu$.

</div>

The big advantage of the Metropolis-Hastings (MH) algorithm is that we only need to be able to evaluate the unnormalised density $p$ of the target measure $\mu$ and the density $q$ of the proposal kernel $Q$. The proposal density is often chosen to be **symmetric**, such that

$$q(x, x') = q(x', x) \qquad \forall x, x' \in \mathbb{R}^n. \tag{5.6.10}$$

In that special case, the acceptance probability simplifies to

$$\alpha(x, x') := \min\left(1, \frac{p(x')}{p(x)}\right). \tag{5.6.11}$$

The 'rule' can be interpreted such that $x'$ is definitely accepted (i.e. with probability 1) if $p(x') \ge p(x)$, and if $p(x') < p(x)$ it is accepted with probability $\frac{p(x')}{p(x)}$. Importantly, in comparison to pure rejection sampling, when a proposal is rejected we include the previous state $x_j$ again as state $x_{j+1}$, i.e. we increase the 'weight' of that state due to its relatively high probability density.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.6.13</span><span class="math-callout__name">(Gaussian Random Walk)</span></p>

The proposal kernel $Q : \mathbb{R}^n \times \mathcal{B}(\mathbb{R}^n) \to [0, 1]$ is chosen to be

$$Q(s; x, \cdot) = \mathcal{N}(x, s^2 I), \tag{5.6.12}$$

where $s > 0$ is the **step size** parameter that can be optimised or calibrated. A well established rule of thumb (which also has some theoretical foundations) is that

$s$ should be chosen such that 

$$\bar{\alpha} := \int_{\mathbb{R}^n} \alpha(x, x') \, Q(s; x, \mathrm{d}x') \mu(\mathrm{d}x) \approx 0.21 \tag{5.6.13}$$

where $\bar{\alpha}$ is called the *mean acceptance rate* which can be estimated on the basis of a short trial run of the Markov chain in practice.

The proposal kernel $Q(s; x, \cdot)$ has a transition density

$$q_s(x, x') \propto \exp\left( -\frac{1}{2s^2} \lVert x' - x \rVert^2 \right)$$

and is thus clearly symmetric, i.e., it satisfies (5.6.10).

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The step-size dilemma behind the magic number 0.21)</span></p>

The tuning rule (5.6.13) resolves a genuine trade-off, and it is worth seeing both failure modes:

* **$s$ too small:** almost every proposal is accepted ($\bar{\alpha} \approx 1$), but the chain moves in tiny increments — consecutive states are almost identical, autocorrelations decay very slowly, and $\mathrm{IACT}\_F$ is huge. High acceptance is *not* a sign of a good sampler.
* **$s$ too large:** proposals jump far from the current state, typically into regions of negligible posterior density, so almost everything is rejected ($\bar{\alpha} \approx 0$) and the chain stays glued to its current state for long stretches — again huge $\mathrm{IACT}\_F$.

The optimum sits in between, and the theoretical foundation (optimal-scaling analysis of random walk MH in the limit $n \to \infty$, for product-form targets) gives the celebrated value $\bar{\alpha} \approx 0.234 \approx 0.21$-$0.23$, with the optimal step size scaling like $s \sim n^{-1/2}$. That the optimal $s$ *shrinks with the dimension* is exactly the dimension-dependence problem that the pCN proposal of Section 5.6.3 is designed to remove.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.6.14</span><span class="math-callout__name">(Geometric Ergodicity of MH)</span></p>

Let $p$ be the unnormalised density of the target $\mu \in \mathcal{P}(\mathbb{R}^n)$. If the proposal kernel $Q$ in Algorithm 1 is such that

$$p(x') > 0 \quad \text{implies} \quad q(x, x') > 0, \quad \text{for all } x \in \mathbb{R}^n, \tag{5.6.14}$$

and if

$$\mathbb{P}\big(\alpha(x_j, X') = 1\big) < 1, \quad \text{for all } j \in \mathbb{N}, \tag{5.6.15}$$

then the Markov chain $(X_j)\_{j \in \mathbb{N}}$ produced by the MH algorithm is $L^2_\mu(\mathbb{R}^n)$-geometrically ergodic and the Central Limit Theorem 5.6.10 applies.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.6.15</span></p>

Condition (5.6.14) on the proposal distribution is similar to the condition required on the importance distribution $q$ for the ratio estimator in Lemma 5.5.4. Condition (5.6.15), on the other hand, guarantees that the Markov chain is **aperiodic**. However, it is somewhat academic, since there is not much point in applying the MH algorithm, when the proposal distribution is so good that proposed states are accepted a.s.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.6.16</span><span class="math-callout__name">(MH for a Bimodal 1D Posterior)</span></p>

Let us consider the MH algorithm for a simple one dimensional posterior distribution. In particular, we consider a RV $X : \Omega \to \mathbb{R}$ with prior distribution $X \sim \mu\_X = \mathcal{N}(0, 1)$, conditioned on the observation $y = 4$ of $Y = X^2 + E$ with $E \sim \mathcal{N}(0, 1)$. Thus,

$$\pi_{X|Y}(x|y) \propto \exp\left( -\tfrac{1}{2}(4 - x^2)^2 \right) \exp\left( -\tfrac{1}{2}x^2 \right) = \exp\left( -\tfrac{1}{2}\big[ (4 - x^2)^2 + x^2 \big] \right),$$

see the left panel of the figure below. Note why the posterior is **bimodal**: the data $y = 4$ only informs $x^2$, so both $x \approx +\sqrt{4}$ and $x \approx -\sqrt{4}$ explain the observation equally well, and the symmetric prior cannot break the tie — the two modes near $\pm 1.95$ are slightly pulled towards the origin by the prior. Bimodality survives here even though prior, noise, and forward map are all as simple as can be; only the *linearity* of $\Phi$ was missing (cf. Theorem 5.3.1).

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/mcmc_bimodal_density.png' | relative_url }}" alt="Left: the unnormalised posterior density exp(-((4-x^2)^2 + x^2)/2), showing two sharp symmetric peaks near x = -2 and x = 2 with a deep valley of essentially zero density at the origin. Right: histogram of relative frequencies of a Metropolis-Hastings chain with 10000 states, matching the normalised posterior density curve closely." loading="lazy">
</figure>

*The unnormalised posterior density from Example 5.6.16 (left) and the histogram of relative frequencies along an MH path with $N = 10^4$ states compared to the true, normalised posterior density (right) — the agreement is very good, including the relative mass of the two modes.*

Let us use the MH algorithm with random walk proposal kernel $Q(s; x, \cdot)$ defined in (5.6.12) to sample from $\pi\_{X\mid Y}(x\mid y)$. The acceptance probability is

$$\alpha(x, x') = \min\left( 1, \frac{\pi_{X|Y}(x'|y)}{\pi_{X|Y}(x|y)} \right) = \min\left( 1, \frac{\exp\left( -\frac{1}{2}\big[ (4 - (x')^2)^2 + (x')^2 \big] \right)}{\exp\left( -\frac{1}{2}\big[ (4 - x^2)^2 + x^2 \big] \right)} \right).$$

The criterion (5.6.13) is satisfied for a step size of roughly $s = 1.5$. As the initial state, we choose $x\_1 = 0$, i.e. $\nu = \delta\_0$. The figure below shows a realisation $(x\_j)\_{j \in \mathbb{N}}$ of the Markov chain produced by the resulting MH algorithm.

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/mcmc_bimodal_paths.png' | relative_url }}" alt="Three trace plots of the same Metropolis-Hastings Markov chain shown at increasing lengths: 100 steps, 1000 steps and 10000 steps. The chain quickly leaves the initial state 0, settles around the two modes near -2 and 2, and hops between them repeatedly; at 10000 steps the two bands around -2 and +2 are clearly visible with frequent switches." loading="lazy">
</figure>

*A path of the Markov chain produced in Example 5.6.16, shown after $10^2$, $10^3$ and $10^4$ steps. Two features are worth noting: (i) repeated values (rejections) appear as horizontal segments — these repetitions are part of the correct weighting, not an artefact; (ii) with step size $s = 1.5$ the proposal is wide enough to jump across the essentially-zero-density valley at $x = 0$, so the chain **mixes between the two modes**. A much smaller step size would produce a chain that looks locally healthy but stays trapped in one mode for an extremely long time (metastability) — its histogram would silently converge to the wrong answer. This is the practical reason why acceptance-rate tuning and multiple diagnostics (trace plots, multiple chains from different initial states) matter.*

</div>

#### 5.6.3 Extension to Infinite Dimensions

Let us briefly discuss the extension to a general, possibly infinite-dimensional, separable Hilbert space $H$. Except for the acceptance probability $\alpha$ in (5.6.8), all the elements of the MH algorithm in Algorithm 1 were not specific to $\mathbb{R}^n$.

In particular, if $\mu \in \mathcal{P}(H)$, $\nu \in \mathcal{P}(H)$ and $Q : H \times \mathcal{B}(H) \to [0, 1]$ are the target measure, a measure for the initial state and a proposal kernel on $H$, respectively, the only remaining question is how to choose $\alpha$, such that the Markov chain produced by Algorithm 1 is $\mu$-reversible. With the probability measures $\rho, \rho^\top \in \mathcal{P}(H \times H)$ defined as

$$\rho(\mathrm{d}x, \mathrm{d}x') := Q(x, \mathrm{d}x') \mu(\mathrm{d}x) \quad \text{and} \quad \rho^\top(\mathrm{d}x, \mathrm{d}x') := \rho(\mathrm{d}x', \mathrm{d}x), \tag{5.6.16}$$

the following proposition can be proved similarly to Proposition 5.6.12.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.6.17</span><span class="math-callout__name">(MH in Infinite Dimensions)</span></p>

If the Radon-Nikodym derivative $\frac{\mathrm{d}\rho^\top}{\mathrm{d}\rho} : H \times H \to [0, \infty)$ exists and we replace the acceptance probability (5.6.8) in Algorithm 1 by

$$\alpha(x_j, x') = \min\left( 1, \frac{\mathrm{d}\rho^\top}{\mathrm{d}\rho}(x_j, x') \right), \tag{5.6.17}$$

then the transition kernel $K : H \times \mathcal{B}(H) \to [0, 1]$ of the Markov chain $(X\_j)\_{j \in \mathbb{N}}$ that is produced by Algorithm 1 with proposal kernel $Q$ is given by

$$K(x, \mathrm{d}x') = \alpha(x, x') Q(x, \mathrm{d}x') + \int_H \big( 1 - \alpha(x, x'') \big) \, Q(x, \mathrm{d}x'') \, \delta_x(\mathrm{d}x')$$

and it is $\mu$-reversible.

</div>

For $H = \mathbb{R}^n$, the two definitions of $\alpha$ in (5.6.17) and (5.6.8) agree. Note that

$$\rho(\mathrm{d}x, \mathrm{d}x') = Q(x, \mathrm{d}x') \mu(\mathrm{d}x) \propto q(x, x') \, p(x) \, \mathrm{d}x' \, \mathrm{d}x,$$

so that, provided $q(x, x') \, p(x) > 0$, we have

$$\frac{\mathrm{d}\rho^\top}{\mathrm{d}\rho}(x, x') = \frac{q(x', x) \, p(x')}{q(x, x') \, p(x)}.$$

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(What $\rho$ and $\rho^\top$ mean — and why densities had to go)</span></p>

The measure $\rho$ is the joint law of one *proposed transition* of the chain in equilibrium: draw the current state $x \sim \mu$, then a proposal $x' \sim Q(x, \cdot)$. Its transpose $\rho^\top$ is the law of the *reversed* pair. Detailed balance of the accepted chain is exactly the statement that acceptance reweights $\rho$ into something symmetric — and the correct reweighting factor is the Radon-Nikodym derivative $\mathrm{d}\rho^\top/\mathrm{d}\rho$. Formula (5.6.17) is therefore not a generalisation *trick* but the coordinate-free way of writing the familiar ratio $\frac{p(x')q(x', x)}{p(x)q(x, x')}$: numerator and denominator, which are each meaningless in infinite dimensions (there is no Lebesgue measure to have densities against), only ever appear through their *ratio*, and the ratio survives as a measure-theoretic object whenever $\rho^\top \ll \rho$.

The catch, and the point of the rest of this subsection: for common proposals, $\rho^\top \ll \rho$ **fails** in infinite dimensions. For a Gaussian random walk on $H$, current state and proposal explore mutually singular Gaussians in the limit, the derivative does not exist, and the "acceptance probability" degenerates — this is the measure-theoretic root of the dimension-dependent collapse of random walk MH observed in Remark 5.6.19 and Example 5.6.20.

</div>

In infinite dimensional spaces the existence of $\frac{\mathrm{d}\rho^\top}{\mathrm{d}\rho}$ is thus not guaranteed for common proposal kernels. A possible way to ensure it in the case of a posterior measure $\mu = \mu\_{X\mid y}$ with

$$\frac{\mathrm{d}\mu_{X|y}}{\mathrm{d}\mu_X}(x) \propto \exp\left(-\frac{1}{2}\lVert y - \Phi(x)\rVert_\Sigma^2\right) =: \exp(-\mathcal{M}(x)),$$

is to choose a proposal kernel $Q$ that is **prior-reversible**, i.e.,

$$\eta(\mathrm{d}x, \mathrm{d}x') := Q(x, \mathrm{d}x')\mu_X(\mathrm{d}x) = Q(x', \mathrm{d}x)\mu_X(\mathrm{d}x') =: \eta^\top(\mathrm{d}x, \mathrm{d}x'). \tag{5.6.18}$$

Multiplying both sides with $\exp[-\mathcal{M}(x) - \mathcal{M}(x')]$, this implies

$$\exp(-\mathcal{M}(x'))Q(x, \mathrm{d}x')\mu_{X|y}(\mathrm{d}x) = \exp(-\mathcal{M}(x))Q(x', \mathrm{d}x)\mu_{X|y}(\mathrm{d}x').$$

Thus, $\frac{\mathrm{d}\rho^\top}{\mathrm{d}\rho}$ is well-defined (under reasonable conditions on $\Phi$) and

$$\alpha(x, x') = \min\left(1, \exp\big(\mathcal{M}(x) - \mathcal{M}(x')\big)\right). \tag{5.6.19}$$

One way to trivially obtain prior-reversibility is an **independence sampler** with $Q(x, \mathrm{d}x') = \mu_X(\mathrm{d}x')$, independently of $x$, which uses independent draws from the prior as proposals. However, this will not work well in practice if the data is informative and the posterior concentrates only on part of the support of $\mu_X$.

A more efficient alternative can be obtained through a slight modification of the Gaussian random walk proposal kernel from Example 5.6.13.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.6.18</span><span class="math-callout__name">(pCN Proposals)</span></p>

Let $X$ be a RV on $H$ with Gaussian prior $\mu_X = \mathcal{N}(0, C)$ and let $\mu_{X\mid y} \in \mathcal{P}(H)$ be a posterior distribution of the usual form with additive Gaussian likelihood, such that

$$\mu_{X|y}(\mathrm{d}x) \propto \exp(-\mathcal{M}(x)) \mu_X(\mathrm{d}x), \quad \text{with} \quad \mathcal{M}(x) = \frac{1}{2}\lVert y - \Phi(x) \rVert_\Sigma^2.$$

Then, Algorithm 1 with the so-called **preconditioned Crank-Nicolson (pCN) proposal kernel**

$$Q(s; x, \cdot) := \mathcal{N}\left(\sqrt{1 - s^2}\, x,\, s^2 C\right), \quad \text{for } s \in (0, 1), \tag{5.6.20}$$

and acceptance probability

$$\alpha(x, x') = \min\big(1, \exp(\mathcal{M}(x) - \mathcal{M}(x'))\big) \tag{5.6.21}$$

produces a $\mu_{X\mid y}$-reversible Markov chain.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 5.6.18</summary>

To see that the pCN kernel in (5.6.20) is prior-reversible, consider $\eta(\mathrm{d}x, \mathrm{d}x') = Q(s; x, \mathrm{d}x') \mu\_X(\mathrm{d}x)$ and let $X, W$ be two independent samples from $\mu\_X = \mathcal{N}(0, C)$. Then,

$$\begin{pmatrix} X \\ X' \end{pmatrix} := \begin{bmatrix} I & 0 \\ \sqrt{1 - s^2}\, I & sI \end{bmatrix} \begin{pmatrix} X \\ W \end{pmatrix} = \begin{pmatrix} X \\ \sqrt{1 - s^2}\, X + sW \end{pmatrix} \sim \eta.$$

As a linear combination of Gaussians, the RV $(X, X')$ is jointly Gaussian, and as in (5.3.3), it follows that

$$\eta = \mathcal{N}\left( \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \begin{bmatrix} C & \sqrt{1 - s^2}\, C \\ \sqrt{1 - s^2}\, C & C \end{bmatrix} \right),$$

which is symmetric and independent of the order of the two RVs $X$ and $X'$. Thus, $\eta = \eta^\top$.

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Why pCN is dimension-independent — the division of labour)</span></p>

The pCN proposal $x' = \sqrt{1 - s^2}\, x + s\, \xi$ with $\xi \sim \mathcal{N}(0, C)$ is an autoregressive (AR(1)) move, not a random walk $x' = x + s\,\xi$, and the difference is precisely the factor $\sqrt{1 - s^2}$ that contracts the current state towards the prior mean. Three observations unpack the construction:

* **The proposal preserves the prior exactly.** If $x \sim \mathcal{N}(0, C)$ then $\mathrm{cov}(x') = (1 - s^2)C + s^2 C = C$ — the coefficients $\sqrt{1 - s^2}$ and $s$ are the *unique* Pythagorean pair with this property. A random walk instead inflates the covariance to $(1 + s^2)C$, and in infinite dimensions $\mathcal{N}(0, C)$ and $\mathcal{N}(0, (1 + s^2)C)$ are **mutually singular** — this is exactly the failure of $\rho^\top \ll \rho$ noted above, and the reason RW-MH degenerates.
* **A clean division of labour.** Prior-reversibility means the proposal handles the (infinite-dimensional, Gaussian) prior part of the posterior *exactly*; the accept/reject step (5.6.21) only has to account for the (finite-dimensional, data-driven) likelihood misfit $\mathcal{M}(x) = \frac{1}{2} \lVert y - \Phi(x) \rVert\_\Sigma^2$. Since the data is finite, the acceptance probability stays non-degenerate no matter how fine the discretisation — the algorithm is **well-defined on $H$ itself**, and any discretisation merely approximates it.
* **The name.** Applying a Crank-Nicolson (trapezoidal) discretisation to the prior-preserving Ornstein-Uhlenbeck-type Langevin equation $\mathrm{d}X = -X \, \mathrm{d}t + \sqrt{2C} \, \mathrm{d}W$ and *preconditioning* by $C$ yields exactly this proposal — hence *preconditioned Crank-Nicolson*.

The practical rule of thumb "design the algorithm in function space first, discretise second" (well-posedness of the *algorithm*, not just the problem) is one of the main takeaways of this whole section.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.6.19</span></p>

The assumption that $\mu_X = \mathcal{N}(0, C)$ in Prop. 5.6.18 is crucial. In practice, we never work with infinite-dimensional distributions. However, the classical Gaussian random walk in Example 5.6.13 is not prior-reversible and its acceptance probability is not well-defined in $H = \mathbb{R}^n$ in the limit as $n \to \infty$. In fact, the step size $s$ to achieve $\bar{\alpha} \approx 0.21$ tends to 0 as $n \to \infty$, so that the proposal distribution degenerates and the convergence of the algorithm becomes very poor.

In contrast, MH algorithms that are well-defined also in the infinite-dimensional limit, such as the pCN algorithm in Proposition 5.6.18, typically lead to a dimension-independent convergence and are therefore suitable also for very high dimensions $n$ of the state space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.6.20</span><span class="math-callout__name">(1D Elliptic Problem: Random Walk vs. pCN)</span></p>

Let us infer the unknown log-diffusion coefficient function $u : [0, 1] \to \mathbb{R}$ in the one-dimensional version of our elliptic model problem

$$-\frac{\mathrm{d}}{\mathrm{d}x}\left( \exp(u(x)) \, \frac{\mathrm{d}p}{\mathrm{d}x}(x) \right) = 0, \qquad p(0) = 0, \quad p(1) = 2,$$

based on 4 noisy observations of the solution $p(x)$ at $x = 0.2, 0.4, 0.6$ and $0.8$.

The prior is chosen to be $U \sim \mu\_U = \mathcal{N}\big(0, (-\Delta\_{\mathrm{D}})^{-1}\big)$, where $\Delta\_{\mathrm{D}}$ denotes the Dirichlet-Laplacian on $(0, 1)$. The additive noise satisfies $E \sim \mathcal{N}(0, \sigma^2 I\_4)$. The prior is discretised using a truncated Karhunen-Loève expansion (KLE) with 50, 100, 200, 400 and 800 terms — in this case a Fourier sine series with i.i.d. Gaussian coefficients. Sample trajectories from the prior are rough mean-zero random functions covering a wide band; sample trajectories from the posterior $\mu\_{U\mid y}$ with noise level $\sigma^2 = 0.01$ and $\sigma^2 = 0.001$ form increasingly narrow bundles around the true coefficient — four point observations of the smooth solution $p$ already pin down the coefficient remarkably well, and the smaller the noise, the tighter the posterior concentration.

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/mcmc_prior_posterior_samples.png' | relative_url }}" alt="Three panels of function samples on the unit interval with the true coefficient, a full sine wave of amplitude 2, drawn in black. Left: 40 prior draws forming a rough, unstructured band around zero that ignores the black curve. Middle: posterior samples for noise level 0.01 forming a bundle that clearly follows the sine shape of the truth. Right: posterior samples for noise level 0.001 forming an even tighter bundle around the truth." loading="lazy">
</figure>

*Example 5.6.20 computed: $40$ draws from the prior $\mathcal{N}\big(0, (-\Delta\_{\mathrm{D}})^{-1}\big)$ (left) and from the posterior (pCN chains, $d = 200$) for noise levels $\sigma^2 = 0.01$ (middle) and $\sigma^2 = 0.001$ (right); the black line is the true log-coefficient $u^\dagger(x) = 2\sin(2\pi x)$. Four point observations of $p$ already force the posterior bundle to follow the shape of $u^\dagger$, and the bundle tightens as the noise decreases. The remaining bias near the extrema is instructive: the data sees $u$ only through the normalised antiderivative of $e^{-u}$ — adding a constant to $u$ leaves $p$ unchanged — so the amplitude is only partially identified and the prior shrinks it towards $0$. Bayesian inversion is doing exactly what it should: certain about what the data determines, prior-driven where it does not.*

Finally, we compare the efficiency of the MH algorithm with two different proposal distributions for increasing dimension $d$ (i.e., more KLE terms in the discretisation of the prior distribution on $u$) in terms of the so-called **integrated autocorrelation time** $\mathrm{IACT}\_F$. Without giving any further details, this is defined as

$$\mathrm{IACT}_F := \frac{\sigma_F^2}{\mathbb{V}_{\mu_{U|y}}\big(F(U)\big)}$$

with $\sigma\_F^2$ as defined in (5.6.7) (with $U\_j$ instead of $X\_j$), i.e., $\mathrm{IACT}\_F$ quantifies how much bigger the MCMC sample size needs to be chosen in comparison to an i.i.d. sample drawn directly from $\mu\_{U\mid y}$. In particular, we consider the Gaussian random walk proposal kernel and the pCN-proposal kernel:

$$Q_s^{\mathrm{RW}}(u, \cdot) := \mathcal{N}\big( u, \, s^2 (-\Delta_{\mathrm{D}})^{-1} \big) \quad \text{and} \quad Q_s^{\mathrm{pCN}}(u, \cdot) := \mathcal{N}\big( \sqrt{1 - s^2}\, u, \, s^2 (-\Delta_{\mathrm{D}})^{-1} \big).$$

We can clearly see that $\mathrm{IACT}\_F \to \infty$ as $d \to \infty$ for the random walk proposals (roughly linearly in $d$), while it remains constant at about $\mathrm{IACT}\_F = 50$ for pCN — the dimension-independence promised by the function-space formulation, observed in practice.

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/mcmc_rw_vs_pcn_iact.png' | relative_url }}" alt="Log-log plot of integrated autocorrelation time against the KLE dimension d from 50 to 800. The random walk Metropolis-Hastings curve climbs steadily from about 90 to about 1300, while the pCN curve stays flat between 50 and 70 across all dimensions." loading="lazy">
</figure>

*The comparison computed for this exact problem ($\sigma^2 = 0.01$, $F(u) = u(0.5)$, chains of $6.5 \cdot 10^4$ steps averaged over three seeds, step sizes adaptively tuned to acceptance rate $\approx 0.23$ during burn-in and then frozen): the $\mathrm{IACT}$ of RW-MH grows from $\approx 90$ at $d = 50$ to $\approx 1300$ at $d = 800$, with the tuned step size shrinking like $d^{-1/2}$ (from $s \approx 0.34$ down to $s \approx 0.09$) exactly as the optimal-scaling theory predicts, while pCN stays flat at $\mathrm{IACT} \approx 50$-$70$ with an essentially $d$-independent step size $s \approx 0.42$. Every RW sample at $d = 800$ is statistically worth twenty times less than a pCN sample — at identical cost per step.*

</div>

#### 5.6.4 More Efficient Proposal Kernels and Multilevel MCMC

As in the context of importance sampling, more efficient proposal kernels can be constructed by using information of the target distribution $\mu$. All the proposals we have seen so far are agnostic about which parts of state space are more probable. Ideally we would like proposals that take this into account, i.e., make it more probable to move to areas where $\mu$ is large. We restrict again to finite dimensions and to $\mu(\mathrm{d}x) = p(x) \, \mathrm{d}x$.

There are a number of **adaptive algorithms** where an appropriate proposal kernel is 'learned' in the initial phase of sampling. One general purpose method that adjusts the covariance of a random walk proposal and combines this with a delayed rejection mechanism that leads to further efficiency enhancements is the so-called **DRAM (Delayed Rejection Adaptive Metropolis)** algorithm of [Haario, Laine, Mira & Saksman, 2006].

Connecting to optimisation, a possible way to include information about $\mu$ is to use gradient information and propose the next move as

$$x' = x_j + \beta \nabla p(x_j).$$

However, this is a deterministic move. We are losing randomness and the ability to explore the state space, as we would converge to a local maximum. So how can we do this properly?

One such approach is the **Metropolis adjusted Langevin algorithm (MALA)** of [Pillai, Stuart & Thiéry, 2012] with proposal kernel

$$Q_\beta^{\mathrm{MALA}}(x, \cdot) = \mathcal{N}\big( x + \beta \nabla \log p(x), \, 2\beta I \big),$$

again with a suitable step size $\beta > 0$. For optimal efficiency, it should be tuned such that the average acceptance rate $\bar{\alpha} \approx 0.574$ here. The background for this sampler is molecular dynamics, more specifically **Langevin dynamics**, described by the stochastic differential equation (SDE)

$$\mathrm{d}X = \nabla \log p(X) \, \mathrm{d}t + \sqrt{2} \, \mathrm{d}W,$$

which is 'driven' by the Wiener process (or Brownian motion) $W$ and has $p$ as its limiting stationary distribution. The MALA proposal is essentially one step of an Euler-Maruyama discretisation applied to the Langevin SDE, i.e.,

$$X' = X_j + \beta \nabla \log p(X_j) + \sqrt{2\beta} \, W_j \quad \text{with} \quad W_j \sim \mathcal{N}(0, I).$$

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Why MALA still needs the accept/reject step)</span></p>

The Langevin SDE has $p$ as its *exact* invariant density — so why not simply simulate it and skip Metropolis-Hastings altogether? Because we cannot simulate the SDE exactly: the Euler-Maruyama step introduces an $O(\beta)$ discretisation error, and the discretised chain is invariant with respect to a *perturbed* distribution $p\_\beta \neq p$ (in unfavourable cases it can even be transient). The MH correction (with the *full*, non-symmetric Hastings ratio — note $q(x, x') \neq q(x', x)$ here, since the drift shifts the mean!) removes this bias **exactly**: the accepted chain targets $p$ for every step size $\beta$. The step size then no longer trades off bias against speed, but only acceptance rate against move size — which is why it can be tuned by the simple $\bar{\alpha} \approx 0.574$ rule. The gradient drift pushes proposals towards high-probability regions, improving the optimal scaling from $s \sim n^{-1/2}$ (random walk) to $\beta \sim n^{-1/3}$, i.e. asymptotically fewer steps per effective sample in high dimensions.

</div>

Other popular approaches are based on **Hamiltonian dynamics**, including a momentum variable — the *Hybrid / Hamiltonian Monte Carlo (HMC)* method of [Duane, Kennedy, Pendleton & Roweth, 1987] and the adaptive *No-U-Turn Sampler (NUTS)* of [Hoffman & Gelman, 2014], which powers modern probabilistic programming systems such as Stan. It is also possible to include 2<sup>nd</sup>-order (Hessian) information, see e.g. the *Riemann manifold Langevin and Hamiltonian Monte Carlo* methods of [Girolami & Calderhead, 2011], the *dimension-independent likelihood-informed MCMC* of [Cui, Law & Marzouk, 2016], or the generalisation of the preconditioned Crank-Nicolson algorithm in [Rudolf & Sprungk, 2018].

Finally, we mention a further, alternative approach that uses a surrogate density $p^\ast \approx p$ to pre-screen proposals, the so-called **surrogate transition method** proposed in [Liu, 2001; Christen & Fox, 2005]. If the surrogate $p^\ast$ is cheap to evaluate, this approach has the important advantage that the potentially expensive target density only needs to be evaluated for proposals that were accepted for $p^\ast$. It proceeds as follows:

1. At state $x$, sample a proposal $x^\ast$ from some proposal density $q^\ast(x, \cdot)$.
2. Set $x' = x^\ast$ with probability

   $$\alpha_1(x, x^\ast) = \min\left( 1, \frac{p^\ast(x^\ast) \, q^\ast(x^\ast, x)}{p^\ast(x) \, q^\ast(x, x^\ast)} \right),$$

   otherwise set $x' = x$.
3. Denote the proposal density associated with this procedure for drawing $x'$ by $q(x, \cdot)$.
4. Accept $x'$ with probability

   $$\alpha_2(x, x') = \min\left( 1, \frac{p(x') \, q(x', x)}{p(x) \, q(x, x')} \right) = \min\left( 1, \frac{p(x') \, p^\ast(x)}{p(x) \, p^\ast(x')} \right),$$

   i.e. $X\_{j+1} = x'$ with probability $\alpha\_2(x, x')$; otherwise stay at $X\_{j+1} = x$.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Why delayed acceptance is exact, and why it is cheap)</span></p>

The two-stage scheme is *not* an approximation: step (iv) is an ordinary MH accept/reject with respect to the **exact** target $p$, applied to the effective proposal $q$ generated by steps (i)-(iii) — so by Proposition 5.6.12 the resulting chain is exactly $\mu$-reversible, whatever the quality of $p^\ast$. The pleasant surprise is the simplification of the Hastings ratio in step (iv): the effective proposal density satisfies $q(x, x') = \alpha\_1(x, x') \, q^\ast(x, x')$ for $x' \neq x$, and inserting this makes all $q^\ast$-factors cancel, leaving $\min\big( 1, \tfrac{p(x') \, p^\ast(x)}{p(x) \, p^\ast(x')} \big)$ — the exact and the surrogate density in a "correction ratio". If $p^\ast \approx p$, this ratio is $\approx 1$: almost everything that survives the cheap first stage is accepted.

The cost accounting is the whole point: the expensive density $p$ (a fine-grid PDE solve, say) is evaluated **only for proposals that already passed the cheap screening** with $p^\ast$ (a coarse-grid solve or reduced-order model). Bad proposals — the majority, for an ambitious step size — are rejected at surrogate cost. The quality of $p^\ast$ affects only the *efficiency*, never the *correctness*.

</div>

The surrogate $p^\ast$ can be, e.g., some reduced order model or the posterior associated with a coarser discretisation ($h^\ast > h$ and/or $s^\ast < s$). This latter choice — combined with the multilevel idea of Section 5.4.2 — leads to the efficient **multilevel Markov chain Monte Carlo** method of [Dodwell, Ketelsen, Scheichl & Teckentrup, *SIAM Review* 2019] and the **multilevel delayed acceptance MCMC** of [Lykkegaard, Dodwell, Fox, Mingas & Scheichl, 2023].

### 5.7 Variational Methods

Contrary to MCMC methods, variational inference is based on optimization instead of sampling. The general idea can be described as follows: Let $\mathcal{H}$ be a **variational family** of probability measures on $\mathbb{R}^n$. To approximate the posterior $\mu_{X\mid y}$, we determine as a surrogate the best approximation within the class $\mathcal{H}$ w.r.t. the KL-divergence

$$\rho^* \in \operatorname{argmin}_{\rho \in \mathcal{H}} D_{\mathrm{KL}}(\rho \| \mu_{X|y}). \tag{5.7.1}$$

Depending on the choice of $\mathcal{H}$, in general such $\rho^\ast$ need not exist or be unique. However, if it does exist, it can be used in place of $\mu_{X\mid y}$ to approximate the quantities we are interested in, such as the conditional mean $\mathbb{E}\_{\mu_{X\mid y}}[X] \approx \mathbb{E}\_{\rho^\ast}[X]$. For this reason the family $\mathcal{H}$ has to be chosen such that expectations $\mathbb{E}\_\rho[f]$ for $\rho \in \mathcal{H}$ are easy and cheap to compute (this is loosely referred to as being "tractable").

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.7.1</span><span class="math-callout__name">(Gaussian Variational Family)</span></p>

Set $\mathcal{H} := \lbrace \mathcal{N}(\mu, \Sigma) : \mu \in \mathbb{R}^m, \Sigma \in \mathbb{R}^{m \times m} \text{ SPD} \rbrace$. Then (5.7.1) corresponds to fitting a Gaussian to the posterior w.r.t. the KL-divergence. Since a Gaussian is uniquely determined through its expectation and covariance, we merely need to determine $\mu \in \mathbb{R}^m$ and $\Sigma \in \mathbb{R}^{m \times m}$. In practice this is done by minimizing $D_{\mathrm{KL}}(\rho \| \mu_{X\mid y})$ with optimization methods such as gradient descent or stochastic gradient descent. Note that (5.7.1) will in general not yield the same result as the Laplace approximation (5.3.4).

</div>

In this section we concentrate on the finite dimensional case and let the parameter $X \in \mathbb{R}^n$, the data $y \in \mathbb{R}^m$, and the posterior $\mu_{X\mid y} \ll \lambda_n$ with density 

$$\pi_{X\mid y}(x) = \frac{\pi_{X,Y}(x,y)}{Z(y)} = \frac{\pi_{Y\mid x}(y)\pi_X(x)}{Z(y)}$$

and normalization constant

$$Z(y) = \int_{\mathbb{R}^n} \pi_{X,Y}(x, y) \, \mathrm{d}x = \int_{\mathbb{R}^n} \pi_{Y|x}(x) \pi_X(x) \, \mathrm{d}x \tag{5.7.2}$$

as in Chapter 4.

#### 5.7.1 ELBO

The normalization constant $Z(y)$ is also referred to as the **model evidence**. Recall that $y \mapsto Z(y) = \pi_Y(y)$ is the marginal density of the data. Assume that $\rho \ll \lambda_n$ for all $\rho \in \mathcal{H}$ and denote $f_\rho(x) = \frac{\mathrm{d}\rho}{\mathrm{d}\lambda_n}(x)$. The **objective function** to be minimized in (5.7.1) then equals

$$D_{\mathrm{KL}}(\rho \| \mu_{X|y}) = \mathbb{E}_\rho\left[\log\left(\frac{f_\rho}{\pi_{X|y}}\right)\right] = \mathbb{E}_\rho[\log(f_\rho)] - \mathbb{E}_\rho[\log(\pi_{X,Y}(\cdot, y))] + \log(Z(y)), \tag{5.7.3}$$

where, as earlier, we use the notation $\mathbb{E}\_\rho[F] = \int F(x) \, \mathrm{d}\rho(x)$. With

$$\mathrm{ELBO}(\rho) := \mathbb{E}_\rho[\log(\pi_{X,Y}(\cdot, y))] - \mathbb{E}_\rho[\log(f_\rho)],$$

the optimization problem (5.7.1) can be reformulated as

$$\operatorname{argmin}_{\rho \in \mathcal{H}} D_{\mathrm{KL}}(\rho \| \mu_{X|y}) = \operatorname{argmax}_{\rho \in \mathcal{H}} \mathrm{ELBO}(\rho),$$

with the equality being an equality of sets in case there are multiple minimizers and maximizers. By Jensen's inequality for concave functions,

$$\mathrm{ELBO}(\rho) = \mathbb{E}_\rho\left[\log\left(\frac{\pi_{X,Y}(\cdot, y)}{f_\rho}\right)\right] \le \log\left(\mathbb{E}_\rho\left[\frac{\pi_{X,Y}(\cdot, y)}{f_\rho}\right]\right) = \log(Z(y)).$$

Therefore $\mathrm{ELBO}(\rho)$ is a lower bound of the logarithm of the model evidence; hence the acronym ELBO (**evidence lower bound**). This could also be deduced from $0 \le D_{\mathrm{KL}}(\rho \| \mu_{X\mid y}) = \log(Z(y)) - \mathrm{ELBO}(\rho)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.7.2</span></p>

In principle another distance or divergence apart from the KL-divergence could be used in (5.7.1), but the KL-divergence has the advantage that the resulting optimization problem can be formulated as maximizing $\mathrm{ELBO}(\rho)$, which is independent of (the in practice unknown constant) $Z(y)$.

</div>

#### 5.7.2 CAVI

In this section we consider coordinate ascent mean-field variational inference (CAVI).

To simplify the optimization problem (5.7.1), one can choose a variational family $\mathcal{H}$ which factorizes over individual variables: $\mathcal{H} = \lbrace \otimes_{j=1}^n \rho_j : \rho_j \in \mathcal{H}\_j \rbrace$ for certain classes $\mathcal{H}\_j$ of probability measures on $\mathbb{R}$. Note that this corresponds to the assumption of the unknown parameters $(X_j)\_{j=1}^n$ being independent. Assuming $\rho_j \ll \lambda$ and setting $f_{\rho_j} := \frac{\mathrm{d}\rho_j}{\mathrm{d}\lambda}$, the density function $f_\rho$ of $\rho = \otimes_{j=1}^n \rho_j$ becomes

$$f_\rho(x_1, \ldots, x_n) = \prod_{j=1}^{n} f_{\rho_j}(x_j).$$

The surrogate $\rho^*$ of $\mu_{X\mid y}$ in (5.7.1) can in this case capture marginal densities of the posterior, but it cannot capture correlation between the different parameters. Due to the type of ansatz, the method is referred to as **mean field variational inference**.

The coordinate ascent algorithm tries to optimize $\mathrm{ELBO}(\rho) = \mathrm{ELBO}(\otimes_{j=1}^n \rho_j)$ by repeatedly iterating through all $j = 1, \ldots, n$, each time only updating (i.e. maximizing in) $\rho_j$. To describe the procedure in more detail let us first introduce the notation

$$\mathbb{E}_{-j}[f](x_j) := \int_{\mathbb{R}^{n-1}} f(x) \, \mathrm{d}\rho_1(x_1) \ldots \mathrm{d}\rho_{j-1}(x_{j-1}) \, \mathrm{d}\rho_{j+1}(x_{j+1}) \ldots \mathrm{d}\rho_n(x_n)$$

for $f : \mathbb{R}^n \to \mathbb{R}$ and $x = (x_1, \ldots, x_n)^\top \in \mathbb{R}^n$. Then $\mathbb{E}_{-j}[f]$ is a function of $x_j$. Fixing $\rho_i$ for all $i \neq j$, we write

$$\rho_j^* := \operatorname{argmax}_{\rho_j \in \mathcal{H}_j} \mathrm{ELBO}(\rho) = \operatorname{argmax}_{\rho_j \in \mathcal{H}_j} \mathbb{E}_\rho[\log(\pi_{X,Y})] - \mathbb{E}_\rho[\log(f_\rho)].$$

If $\mathcal{H}_j$ is chosen as the set of all probability measures on $\mathbb{R}$ which have a density (i.e. are absolutely continuous w.r.t. $\lambda$), then the argmax can be expressed explicitly:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.7.4</span><span class="math-callout__name">(CAVI Update)</span></p>

With the above choice of $\mathcal{H}\_j$ it holds for $f_{\rho_j^\ast} := \frac{\mathrm{d}\rho_j^\ast}{\mathrm{d}\lambda}$ that

$$f_{\rho_j^*}(x_j) \propto \exp\big(\mathbb{E}_{-j}[\log(\pi_{X,Y})]\big)$$

in case $\exp(\mathbb{E}\_{-j}[\log(\pi_{X,Y})]) \in L^1(\mathbb{R})$.

</div>

This leads to Algorithm 2.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 2</span><span class="math-callout__name">(CAVI)</span></p>

**Input:** tolerance, $\pi_{X,Y}$

**while** $\mathrm{ELBO}(\rho) >$ tolerance **do**
&emsp; **for** $j = 1, \ldots, n$ **do**
&emsp;&emsp; set $f_{\rho_j} \propto \exp(\mathbb{E}\_{-j}[\log(\pi_{X,Y})])$
&emsp; **end for**
&emsp; compute $\mathrm{ELBO}(\rho)$
**end while**

</div>

#### 5.7.3 Transport Maps

As mentioned before, the optimization problem (5.7.1) only yields a useful result $\rho^*$, in case the probability measures $\rho \in \mathcal{H}$ are such that they allow for simple computation of quantities like $\mathbb{E}\_{\mu_{X\mid y}}[X] \approx \mathbb{E}\_{\rho^\ast}[X]$. For this reason, variational methods are often applied with somewhat simple variational families such as in Example 5.7.1, which are not able to capture more complex features of the posterior.

Transport maps provide a general approach which is in principle suitable for arbitrarily complex posteriors. Set

$$\mathcal{H} := \lbrace T_\sharp \eta : T \in \mathcal{T} \rbrace, \tag{5.7.5}$$

where $\eta \ll \lambda^n$ is a fixed **reference probability measure** on $\mathbb{R}^n$ (typically $\eta \sim \mathcal{N}(0, I)$), and $\mathcal{T}$ is a family of **transport maps**, which we here assume to be bijective maps $T : \mathbb{R}^n \to \mathbb{R}^n$ such that $T \in C^1$ and also $T^{-1} \in C^1$ (i.e. $T$ is a diffeomorphism). Recall that the pushforward measure is defined as $T_\sharp \eta(A) := \eta(T^{-1}(A))$, and in this section we'll also use the **pullback measure** defined via $T^\sharp \eta(A) := \eta(T(A))$. The optimization problem (5.7.1) can then be equivalently stated as finding

$$T^* := \operatorname{argmin}_{T \in \mathcal{T}} D_{\mathrm{KL}}(T_\sharp \eta \| \mu_{X|y}) = \operatorname{argmin}_{T \in \mathcal{T}} D_{\mathrm{KL}}(\eta \| T^\sharp \mu_{X|y}), \tag{5.7.7}$$

The desired quantity in (5.7.1) is then $\rho^* = T_\sharp^* \eta$.

**Sampling using measure transport.** Note that for any $A \in \mathcal{B}(\mathbb{R}^n)$, a RV $S \sim \eta$ and a bijection $T : \mathbb{R}^n \to \mathbb{R}^n$: $\mathbb{P}[T(S) \in A] = \mathbb{P}[S \in T^{-1}(A)] = \eta(T^{-1}(A)) = T_\sharp \eta(A)$. Thus

$$S \sim \eta \quad \Rightarrow \quad T(S) \sim T_\sharp \eta. \tag{5.7.6}$$

Hence, with the minimizer $T^\ast$ in (5.7.5), an approximation to the conditional mean $\mathbb{E}\_{\mu_{X\mid y}}[X]$ is obtained by the Monte Carlo estimate

$$\mathbb{E}_{\mu_{X|y}}[X] \approx \mathbb{E}_{T_\sharp^* \eta}[X] \approx \frac{1}{N} \sum_{j=1}^{N} T^*(S_j), \qquad S_j \sim \eta.$$

Having computed $T^*$, it is therefore easy to approximate the conditional mean.

The optimization problem (5.7.5) in terms of densities reads: Find

$$\operatorname{argmin}_{T \in \mathcal{T}} \int_{\mathbb{R}^n} \log(f_\eta(x)) - \log(\pi_{X,Y}(T(x), y)) - \log(\det \mathrm{d}T(x)) \, \mathrm{d}\eta(x)$$

where this optimization problem is again independent of the constant $Z(y)$. In practice $\mathcal{T}$ is chosen as some parametrization of possible transport maps, for instance using polynomial expansions or neural networks with a suitable network architecture. The problem is then solved by performing gradient descent (or other optimization techniques) on the approximate objective

$$\frac{1}{N} \sum_{j=1}^{N} \log(f_\eta(S_j)) - \log(\pi_{X,Y}(T(S_j), y)) - \log(\det \mathrm{d}T(S_j))$$

with iid samples $S_j \sim \eta$. This optimization problem is in general nonconvex and highly nontrivial to solve.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 3</span><span class="math-callout__name">(Approximate CM Computation Using Transport)</span></p>

**Input:** $f_\eta$, $\pi_{X,Y}$, $n$

$$\tilde{T} \leftarrow \operatorname{argmin}_{T \in \mathcal{T}} \mathbb{E}_{x \sim \eta}[\log(f_\eta) - \log(\pi_{X,Y}(T(x), y)) - \log(\det \mathrm{d}T(x))]$$

$S_j \sim \eta$ iid for $j = 1, \ldots, n$

**return** $\frac{1}{n} \sum_{j=1}^{n} \tilde{T}(S_j)$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.7.5</span></p>

Note that in the general form (5.7.1), every $\rho \in \mathcal{H}$ needs to be such that we can easily sample from it in order to compute a Monte Carlo approximation. Using the transport maps approach (5.7.5), due to (5.7.6) this automatically holds as long as we can sample from $\eta$.

</div>

**Triangular transports.** We show the existence of transport maps pushing forward a reference measure to a target. We denote in the following by $\mu$ a target measure on $\mathbb{R}^n$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.7.6</span><span class="math-callout__name">(Monotone Transport in 1D)</span></p>

Let $\eta$, $\mu$ be two probability measures on $\mathbb{R}$ with CDFs $F_\eta : \mathbb{R} \to [0, 1]$ and $F_\mu : \mathbb{R} \to [0, 1]$, and let $\eta$ be atomless. Then $T := F_\mu^{[-1]} \circ F_\eta$ is nondecreasing and satisfies $T_\sharp \eta = \mu$.

</div>

The above lemma shows the existence of a transport map pushing forward $\eta$ to $\mu$ for two measures on $\mathbb{R}$. Next, we generalize this construction to the case of two measures on $\mathbb{R}^n$, using marginal and conditional densities. For a vector $x = (x_1, \ldots, x_d)^\top \in \mathbb{R}^n$, we use the notation $\bar{x}\_k := (x_1, \ldots, x_k)^\top \in \mathbb{R}^k$, and define $f^k(\bar{x}\_k) := \int_{\mathbb{R}^{n-k}} f(\bar{x}\_n) \, \mathrm{d}x_{k+1} \ldots \mathrm{d}x_n$ (the marginal density of the first $k$ variables of $\eta$) and $f_{\bar{x}\_{k-1}}^k(x_k) := \frac{f^k(\bar{x}\_k)}{f^{k-1}(\bar{x}\_{k-1})}$ (the conditional density of $x_k$ given $\bar{x}\_{k-1}$). Analogously for $g$.

We next construct $T = (T_1, \ldots, T_n)$. Let $T_1 : \mathbb{R} \to \mathbb{R}$ be such that $(T_1)\_\sharp \eta^1 = \mu^1$ (pushes forward the marginal of $\eta$ in the first variable to the marginal of $\mu$ in the first variable). This exists by Lemma 5.7.6.

Inductively, for each $k = 2, \ldots, d$ and for each $(\bar{x}\_{k-1}) \in \mathbb{R}^{k-1}$ we let $T_k(\bar{x}\_{k-1}, \cdot) : \mathbb{R} \to \mathbb{R}$ be the transport satisfying $(T_k(\bar{x}\_{k-1}, \cdot))\_\sharp \eta_{\bar{x}\_{k-1}}^k = \mu_{T^{k-1}(\bar{x}\_{k-1})}^k$ (pushes forward the conditional marginal of $\eta$ in the $k$th variable conditioned on $\bar{x}_{k-1}$, to the conditional marginal of $\mu$ in the $k$th variable conditioned on $T^{k-1}(\bar{x}\_{k-1})$). This yields a map $T := T^n = (T_1, \ldots, T_n)^\top : \mathbb{R}^n \to \mathbb{R}^n$ which is **triangular** in the sense that the $k$th component $T_k$ depends only on $\bar{x}\_k = (x_1, \ldots, x_k)^\top$:

$$T(x_1, \ldots, x_n) = \begin{pmatrix} T_1(x_1) \\ T_2(x_1, x_2) \\ \vdots \\ T_n(x_1, \ldots, x_n) \end{pmatrix}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.7.8</span><span class="math-callout__name">(Knothe-Rosenblatt Transport)</span></p>

Under the above conditions it holds $T_\sharp \eta = \mu$.

</div>

**Conditional sampling using triangular transports.** For Bayesian inference it is also interesting to consider the case $\mu = \mu_{Y,X}$, i.e. the target is the joint measure of the data and the parameter, and $T_\sharp \eta = \mu_{Y,X}$ for some reference measure $\eta$ on $\mathbb{R}^{m+n}$ with $X \in \mathbb{R}^n$, $Y \in \mathbb{R}^m$. The reason is that, as we saw in the construction of the Knothe-Rosenblatt map, its components push forward conditional densities to conditional densities. Since the posterior is a conditional density, this yields a method to sample from the posterior. We emphasize that this is a specific feature of the triangular Knothe-Rosenblatt map.

To illustrate the idea, consider the simplest case where $m = n = 1$. Suppose that $T : \mathbb{R}^2 \to \mathbb{R}^2$ pushes forward a reference $\eta = \eta_1 \otimes \eta_2$ (e.g. $\eta_j \sim \mathcal{N}(0, 1)$) to the joint $\mu_{Y,X}$. Then $T = (T_1, T_2)$ satisfies $(T_1)\_\sharp \eta_1 = \mu_Y$ and $(T_2(y, \cdot))\_\sharp \eta_2 = \mu_{X\mid T_1(y)}$. Thus for a RV $S \in \mathbb{R}$: $S \sim \eta_2 \Rightarrow T_2(T_1^{-1}(y), S) \sim \mu_{X\mid y}$. In other words, if we have $T$ as in (5.7.10)-(5.7.11) such that $T_\sharp(\eta_1 \otimes \eta_2) = \mu_{Y,X}$, we can use it to construct iid samples from the posterior $\mu_{X\mid y}$ by sending iid samples $S_j \sim \eta_2$ through the map $x \mapsto T_2(T_1^{-1}(y), x)$.

### 5.8 Sequential Monte Carlo Methods & Bayesian Filtering

In this section, we present an outlook to data assimilation problems. In the data assimilation problem we deal with the combination of two information sources:

* **Dynamical system:** We consider a time-dependent physical system described through our mathematical model. In particular, let $Z = (Z_j)\_{j \in \mathbb{N}}$ be a Markov chain describing the dynamical system through

$$Z_{j+1} = H_j(Z_j) + \xi_j, \quad j \in \mathbb{N}, \tag{5.8.1}$$

with $Z_0 \sim \pi_0$ for some probability distribution $\pi_0$ on $\mathbb{R}^n$. The dynamics are driven by the possibly nonlinear mappings $H_j : \mathbb{R}^n \to \mathbb{R}^n$ and perturbed by additive Gaussian noise $\xi = (\xi_i)\_{i \in \mathbb{N}}$ with $\xi_1 \sim \mathcal{N}(0, \Sigma)$, and $Z_0$ and $\xi_0$ are stochastically independent. We refer to (5.8.1) as the **(stochastic) dynamical system** and denote its current state $Z_j$ as **signal**.

* **Observations:** We assume to have access to a time series of **observations** $Y = (Y_i)\_{i \in \mathbb{N}}$ described through the observation model

$$Y_{j+1} = h_{j+1}(Z_{j+1}) + \eta_{j+1}, \quad j \in \mathbb{N}, \tag{5.8.2}$$

where $h_j : \mathbb{R}^n \to \mathbb{R}^K$ are mapping the signal to the observation space $\mathbb{R}^K$, and the noise $\eta = (\eta_i)\_{i \in \mathbb{N}}$ is i.i.d. with $\eta_1 \sim \mathcal{N}(0, \Gamma)$.

We aim to use both the dynamical system as well as the incoming observations to construct sequential estimates of the current signal or even to predict the future signal. We call the task of determining information about the signal $Z$, given the observation $Y$, **data assimilation problem**.

#### 5.8.1 Prediction, Filtering and Smoothing

We assume to have access to the prior information about the unknown signal given by the probability density function $\pi_0$. With the application of the Chapman-Kolmogorov equation for the Markov chain constructed in (5.8.1), we can compute the marginal distribution $\pi_{Z_j}$ of $Z_j$ sequentially by

$$\pi_{Z_{j+1}}(\mathrm{d}z') = \mathbb{P}(Z_{j+1} \in \mathrm{d}z') = \int_{\mathbb{R}^n} \pi_j(\mathrm{d}z' \mid z) \pi_{Z_j}(\mathrm{d}z).$$

Given a realization $y^{[1:N_{\mathrm{obs}}]} = (y_1, \ldots, y_{N_{\mathrm{obs}}})$ of the time series of observations, $N_{\mathrm{obs}} \ge 1$, the data assimilation problem is the computation of the conditional distribution of $Z_j$ given $y^{[1:N_{\mathrm{obs}}]}$:

$$\pi_{Z_j | y^{[1:N_{\mathrm{obs}}]}}(\mathrm{d}z) = \mathbb{P}(Z_j \in \mathrm{d}z \mid Y^{[1:N_{\mathrm{obs}}]} = y^{[1:N_{\mathrm{obs}}]}). \tag{5.8.3}$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.8.2</span><span class="math-callout__name">(Prediction, Filtering, Smoothing)</span></p>

We call the task of computing (5.8.3)

1. **prediction problem** if $j > N_{\mathrm{obs}}$,
2. **filtering problem** if $j = N_{\mathrm{obs}}$,
3. and **smoothing problem** if $j < N_{\mathrm{obs}}$.

Depending on the corresponding case, we denote the distribution in (5.8.3) as **prediction, filtering and smoothing distribution**.

</div>

Through the connection to Bayesian inverse problems, we will focus on filtering problems. The filtering problem splits into two steps:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.8.3</span><span class="math-callout__name">(Prediction and Bayesian Assimilation Steps)</span></p>

Given the filtering distribution $\pi_{Z_j \mid y^{[1:j]}}$, we refer to the **prediction step** as the computation of the marginal distribution of the next state through

$$\pi_{Z_{j+1} | y^{[1:j]}}(\mathrm{d}z) = \mathbb{P}(Z_{j+1} \in \mathrm{d}z \mid Y^{[1:j]} = y^{[1:j]}) = \int_{\mathbb{R}^n} \pi_{j+1}(\mathrm{d}z \mid z') \pi_{Z_j | y^{[1:j]}}(\mathrm{d}z').$$

We call the second step **Bayesian assimilation step**, which is the computation of the filtering distribution $\pi_{Z_{j+1} \mid y^{[1:j+1]}}$ via Bayes' theorem

$$\pi_{Z_{j+1} | y^{[1:j+1]}}(\mathrm{d}z) = \frac{\pi_{Y_{j+1}}(y_{j+1} \mid z) \pi_{Z_{j+1} | y^{[1:j]}}(\mathrm{d}z)}{\int_{\mathbb{R}^n} \pi_{Y_{j+1}}(y_{j+1} \mid \tilde{z}) \pi_{Z_{j+1} | y^{[1:j]}}(\mathrm{d}\tilde{z})}.$$

</div>

Summarizing, given the current filtering distribution $\pi_{Z_j \mid y^{[1:j]}}$, we construct a prior distribution $\pi_{Z_{j+1} \mid y^{[1:j]}}$ using our knowledge of the stochastic dynamical system and update w.r.t. the incoming data $Y_{j+1} = y_{j+1}$ via Bayes' theorem.

#### 5.8.2 Linear Kalman Filter

Under linear and Gaussian assumptions on the underlying stochastic dynamical system and the corresponding observations, the **Kalman filter** solves the filtering problem exactly. We consider the signal described through

$$Z_{j+1} = FZ_j + \xi_j, \quad j \in \mathbb{N} \tag{5.8.4}$$

and the observations

$$Y_{j+1} = AZ_{j+1} + \eta_j, \quad j \in \mathbb{N} \tag{5.8.5}$$

where $F \in \mathcal{L}(\mathbb{R}^n, \mathbb{R}^n)$ and $A \in \mathcal{L}(\mathbb{R}^n, \mathbb{R}^K)$. Furthermore, we assume that the initial distribution is Gaussian, i.e. $\pi_0 = \mathcal{N}(m_0, C_0)$. Since the forward maps are assumed to be linear and the noise to be Gaussian, the filtering distribution remains Gaussian

$$\pi_{Z_j | y^{[1:j]}} = \mathcal{N}(m_j, C_j).$$

Given the initial mean $m_0 \in \mathbb{R}^n$ and symmetric, positive definite covariance $C_0 \in \mathbb{R}^{n \times n}$, the Kalman filter computes the mean $m_j$ and covariance $C_j$ of the filtering distribution recursively.

1. **Prediction step:** Given the mean $m_j$ and covariance $C_j$ of iteration $j$, we first update based on the stochastic dynamical system (5.8.4). Since we have assumed that $\xi_j$ is independent of $Z_j \sim \mathcal{N}(m_j, C_j)$, the prediction step computes

   $$\widehat{m}_{j+1} = Fm_j, \quad \widehat{C}_{j+1} = FC_j F^\top + \Sigma.$$

2. **Bayesian assimilation step:** We set the prior distribution $\pi_Z = \mathcal{N}(\widehat{m}_{j+1}, \widehat{C}\_{j+1})$ and update the mean and the covariance according to Bayes' Theorem (compare Theorem 5.3.1.)

   $$m_{j+1} = \widehat{m}_{j+1} + \widehat{C}_{j+1} A^\top (A\widehat{C}_{j+1} A^\top + \Gamma)^{-1}(y_{j+1} - A\widehat{m}_{j+1})$$

   $$C_{j+1} = \widehat{C}_{j+1} - \widehat{C}_{j+1} A^\top (A\widehat{C}_{j+1} A^\top + \Gamma)^{-1} A\widehat{C}_{j+1} \tag{5.8.6}$$

   Defining the **Kalman gain** $K_j = \widehat{C}\_j A^\top (A\widehat{C}\_j A^\top + \Gamma)^{-1}$, we can write the Bayesian update step as $m_{j+1} = \widehat{m}\_{j+1} + K_{j+1}(y_{j+1} - A\widehat{m}\_{j+1})$ and $C_{j+1} = \widehat{C}\_{j+1} - K_{j+1} A\widehat{C}\_{j+1}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.8.4</span><span class="math-callout__name">(Positive Definiteness of Kalman Covariance)</span></p>

Assume that $Z_0 \sim \mathcal{N}(m_0, C_0)$ for some symmetric and positive definite covariance matrix $C_0 \in \mathbb{R}^{n \times n}$. Then the matrix $C_j$ resulting from (5.8.6) is symmetric and positive definite.

</div>

#### 5.8.3 Extended Kalman Filter

The extended Kalman filter is a generalization of the linear Kalman filter to nonlinear dynamical systems. In order to apply the introduced Kalman filter, we firstly linearize the nonlinear dynamical system and then apply the Kalman filter to the resulting linear system. This method results in a Gaussian approximation to the filtering distribution.

We assume that the signal and the observations are described by $Z_{j+1} = H(Z_j) + \xi_j$ and $Y_{j+1} = AZ_{j+1} + \eta_j$, where $H : \mathbb{R}^n \to \mathbb{R}^n$ is a possibly nonlinear mapping and $A \in \mathcal{L}(\mathbb{R}^n, \mathbb{R}^K)$. Given the initial distribution $\pi_0 = \mathcal{N}(m_0, C_0)$, the extended Kalman filter approximates the filtering distribution by $\mathcal{N}(m_j, C_j)$ as follows

1. **Linearization step:** Given the mean $m_j$ and covariance $C_j$, we build the linearized approximation $Z_{j+1} = F_j Z_j + b_j + \xi_j$, with $F_j := \mathrm{D}H(m_j)$ and $b_j = H(m_j) - F_j m_j$.
2. **Prediction step:** We predict the mean and the covariance by $\widehat{m}\_{j+1} = F_j m_j + b_j$ and $\widehat{C}\_{j+1} = F_j C_j F_j^\top + \Sigma$.
3. **Bayesian assimilation step:** We again set $\pi_Z = \mathcal{N}(\widehat{m}\_{j+1}, \widehat{C}\_{j+1})$ and update the mean and the covariance according to Bayes' Theorem.

#### 5.8.4 Ensemble Kalman Filter

An alternative method to overcome the nonlinearity in the dynamical system is the application of the **ensemble Kalman filter (EnKF)**. The EnKF has been originally introduced by G. Evensen (2003) and can be viewed as a Monte Carlo approximation of the Kalman filter. The basic idea is to use a particle system, initialized by a sample of the prior distribution $Z_0 \sim \pi_0$, which will then be updated according to the Kalman filter. Since we do not use any Gaussian assumptions and approximate the filtering distribution with the particle system empirically, we are now able to apply the EnKF in nonlinear dynamical systems (5.8.1). For simplicity, we again consider a linear observation model (5.8.5).

Given the current particle system $(v_j^{(m)})\_{m=1,\ldots,M}$ of size $M$, we proceed as follows.

1. **Prediction step:** We apply the dynamical system to predict the system's state by

   $$\widehat{v}_{j+1}^{(m)} = H(v_j^{(m)}) + \xi_j^{(m)}, \quad m = 1, \ldots, M,$$

   where $(\xi_j^{(m)})\_{m=1,\ldots,M}$ is an i.i.d. sample of $\mathcal{N}(0, \Sigma)$. The empirical mean and the empirical covariance are given by

   $$\widehat{m}_{j+1} = \frac{1}{M} \sum_{m=1}^{M} \widehat{v}_{j+1}^{(m)}, \quad \widehat{C}_{j+1} = \frac{1}{M} \sum_{m=1}^{M} (\widehat{v}_{j+1}^{(m)} - \widehat{m}_{j+1})(\widehat{v}_{j+1}^{(m)} - \widehat{m}_{j+1})^\top. \tag{5.8.7}$$

2. **Analysis step:** We apply to each particle the linear Kalman filter update corresponding to a Gaussian approximation. The particles are updated by

   $$v_{j+1}^{(m)} = \widehat{v}_{j+1}^{(m)} + K_{j+1}(\tilde{y}_{j+1}^{(m)} - A\widehat{v}_{j+1}^{(m)}),$$

   where $\tilde{y}\_{j+1}^{(m)} = y_j + \eta_{j+1}^{(m)}$, $\eta_{j+1}^{(m)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \Gamma)$ denotes perturbed observation and $K_j = \widehat{C}\_j A^\top (A\widehat{C}\_j A^\top + \Gamma)^{-1}$ is again the Kalman gain. The filtering distribution is approximated empirically by

   $$\pi_{Z_j | y^{1:j}}(\mathrm{y}) \approx \widehat{\pi}_j(\mathrm{y}) = \frac{1}{M} \sum_{m=1}^{M} \delta_{v_j^{(m)}}(\mathrm{y}).$$

One advantage of the EnKF is the application in nonlinear dynamical systems. Furthermore, through the computation of the empirical covariance we save computational costs compared to updating the covariance in each iteration according to (5.8.6).

#### 5.8.5 Particle Filters -- Sequential Monte Carlo Methods

As alternative to the different presented variants of Kalman filters, we briefly introduce the class of **particle filters** which can be seen as sequential Monte Carlo method of the filtering distribution without including Gaussian approximations. The aim of particle filters is to approximate the filtering distribution empirically with a weighted particle system by combining the prediction step with ideas from importance sampling.

Given the current weighted particle system $(w_j^{(m)}, \widehat{Z}\_j^{(m)})\_{m=1,\ldots,M}$ we can divide the update scheme again in a prediction step followed by a Bayesian assimilation step described as follows.

1. **Prediction step:** Given the current state approximations $\widehat{Z}\_j^{(m)}$, we first update the particles based on the stochastic dynamical system by

   $$\widehat{Z}_{j+1}^{(m)} = \widehat{Z}_j^{(m)} + \Delta \cdot b(\widehat{Z}_j^{(m)}) + \xi_{j+1}^{(m)}, \quad \xi_{j+1}^{(m)} \sim N(0, \Delta \cdot RR^\top)$$

   such that the marginal distribution of the state $Z_{j+1}$ can be approximated by $\pi_{Z_{j+1}}(\mathrm{d}z) \approx \sum_{m=1}^{M} w_j^{(m)} \delta_{\widehat{Z}\_{j+1}^{(m)}}(\mathrm{d}z)$.

2. **Bayesian assimilation step:** Following Bayes' Theorem we approximate the filtering distribution by

   $$\pi_{Z_{j+1}|y^{[1:j+1]}}(\mathrm{d}z) \approx \sum_{m=1}^{M} w_{j+1}^{(m)} \delta_{\widehat{Z}_{j+1}^{(m)}}(\mathrm{d}z),$$

   where we have updated and normalized the weights $w_{j+1}^{(m)} = \frac{w_j^{(m)} \pi_{Y_{j+1}}(y_{j+1} \mid \widehat{Z}\_{j+1}^{(m)})}{\sum_{m=1}^{M} w_j^{(m)} \pi_{Y_{j+1}}(y_{j+1} \mid \widehat{Z}\_{j+1}^{(m)})}$.

Given the weighted particle system $(w_{j+1}^{(m)}, \widehat{Z}\_{j+1}^{(m)})\_{m=1,\ldots,M}$ in iteration $j+1$, we are able to approximate expectation values for functionals $F : \mathbb{R}^n \to \mathbb{R}$ of the following kind

$$\mathbb{E}[F(Z_{j+1}) \mid Y^{[1:j]} = y^{[1:j]}] \approx \sum_{m=1}^{M} w_j^{(m)} F(\widehat{Z}_{j+1}^{(m)}),$$

$$\mathbb{E}[F(Z_{j+1}) \mid Y^{[1:j+1]} = y^{[1:j+1]}] \approx \sum_{m=1}^{M} w_{j+1}^{(m)} F(\widehat{Z}_{j+1}^{(m)}).$$

In general, particle filters of this form can be viewed as sequential importance sampling method, where existing consistency results are based on perfect models and $M$ approaching infinity. In practical implementations, the generated weights tend to degenerate for small choices of the number of particles $M$. To overcome this issue, resampling methods based on the effective sample size have been considered in the literature.


## Appendix A: Basic Concepts in Functional Analysis

In this appendix we put together the main concepts from functional analysis that will be needed in this lecture. We recommend the following supplementary references:

* H. Alt, *Funktionalanalysis*, 6. Auflage, Springer, Berlin, 2012.
* W. Rudin, *Functional Analysis*, 2nd ed., Mc-Graw-Hill, New York, 1991.
* D. Werner, *Funktionalanalysis*, 6. Auflage, Springer, Berlin, 2007.

### A.1 Normed Spaces and Bounded Linear Operators

#### A.1.1 Normed Spaces

A map $\lVert \cdot \rVert : X \to [0, \infty)$ is a **norm** on a vector space $X$ (over $\mathbb{K} = \mathbb{R}$), if

1. $\lVert \lambda x \rVert = \lvert \lambda \rvert \lVert x \rVert$ for all $\lambda \in \mathbb{K}$, $x \in X$,
2. $\lVert x + y \rVert \le \lVert x \rVert + \lVert y \rVert$ for all $x, y \in X$,
3. $\lVert x \rVert = 0$ iff $x = 0$.

The pair $(X, \lVert \cdot \rVert)$ is called a **normed space**. Two norms $\lVert \cdot \rVert_\alpha$ and $\lVert \cdot \rVert_\beta$ are **equivalent** if there are constants $c_1, c_2 > 0$ such that $c_1 \lVert x \rVert_\beta \le \lVert x \rVert_\alpha \le c_2 \lVert x \rVert_\beta$ for all $x \in X$. If $\dim(X) < \infty$ all norms on $X$ are equivalent. The constants $c_1, c_2$ depend on the dimension of $X$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.1.1</span><span class="math-callout__name">(Examples of Norms)</span></p>

The following maps are norms on

1. $X = \mathbb{R}^n$, $n \in \mathbb{N}$:

   $$\lVert x \rVert_p = \left( \sum_{j=1}^n |x_j|^p \right)^{1/p}, \quad 1 \le p < \infty, \quad \text{and} \quad \lVert x \rVert_\infty = \max_{j=1,\ldots,n} |x_j|;$$

2. $X = l^p$ $(:= \lbrace (t_n) : t_n \in \mathbb{R}, \, \sum_{n=1}^\infty \lvert t_n \rvert^p < \infty \rbrace)$:

   $$\lVert x \rVert_p = \left( \sum_{j=1}^\infty |x_j|^p \right)^{1/p}, \quad 1 \le p < \infty, \quad \text{and} \quad \lVert x \rVert_\infty = \max_{j \in \mathbb{N}} |x_j|;$$

3. $X = L^p(\Omega)$ $(:= \lbrace f : \Omega \to \mathbb{K} : f \text{ measurable}, \, \int_\Omega \lvert f \rvert^p \, \mathrm{d}\lambda < \infty \rbrace)$ where $\Omega \subset \mathbb{R}^n$:

   $$\lVert f \rVert_p = \left( \int_\Omega |f|^p \, \mathrm{d}\lambda \right)^{1/p}, \quad 1 \le p < \infty, \quad \text{and} \quad \lVert f \rVert_\infty = \operatorname{ess\,sup}_{x \in \Omega} |f(x)|.$$

</div>

A normed space $(X, \|\cdot\|_X)$ with $X \subset Y$ is said to be **continuously embedded** in $(Y, \lVert \cdot \rVert_Y)$, denoted by $X \hookrightarrow Y$, if there is a constant $C > 0$ such that $\lVert x \rVert_Y \le C \lVert x \rVert_X$ for all $x \in X$.

A sequence $(x_n) \subset X$ **converges strongly** in $X$ to $x \in X$, denoted $x_n \to x$ as $n \to \infty$, if $\lim_{n \to \infty} \lVert x_n - x \rVert_X = 0$.

A subset $U \subset X$ is called

* **closed**, if the limit of any convergent sequence $(x_n) \subset U$ lies in $U$;
* **compact**, if any sequence $(x_n) \subset U$ has a convergent subsequence $(x_{n_k})\_{k \ge 1}$ with limit $x \in U$;
* **dense** in $X$, if for any $x \in X$ there exists a sequence $(x_n) \subset U$ with $x_n \to x$.

The union of $U$ with the set of all limits of convergent sequences in $U$ is called the **closure** $\overline{U}$ of $U$. It follows that $U$ is dense in $\overline{U}$.

A normed space $X$ is said to be **complete**, if every Cauchy sequence in $X$ converges. Such a space $X$ is also called a **Banach space**. If $X$ is not complete, we denote by $\overline{X}$ its **completion** (w.r.t. the norm $\lVert \cdot \rVert_X$).

For $x \in X$ and $r > 0$ we define

* the **open ball** $U_r(x) = \lbrace z \in X : \lVert x - z \rVert_X < r \rbrace$ and
* the **closed ball** $B_r(x) = \lbrace z \in X : \lVert x - z \rVert_X \le r \rbrace$.

The closed ball at $0$ with radius $1$ is called the **unit ball** $B_X$ in $X$. Furthermore, the set $U \subset X$ is called

* **open**, if for all $x \in U$ there exists a $r > 0$ such that $U_r(x) \subset U$;
* **bounded**, if there exists an $r > 0$ such that $U$ is contained in the closed ball $B_r(0)$;
* **convex**, if for all $x, y \in U$ and $\lambda \in (0, 1)$ we have $\lambda x + (1 - \lambda) y \in U$.

The complement of an open set in a normed space is closed and vice versa. As a consequence of the norm axioms, all open and closed balls are convex.

#### A.1.2 Bounded Operators

Let $(X, \lVert \cdot \rVert_X)$, $(Y, \lVert \cdot \rVert_Y)$ be normed spaces, $U \subset X$ and $F : U \to Y$ a map. We denote by

* $\mathcal{D}(F) := U$ the **domain** of $F$,
* $\mathcal{N}(F) := \lbrace x \in U : F(x) = 0 \rbrace$ the **kernel** of $F$,
* $\mathcal{R}(F) := \lbrace F(x) \in Y : x \in U \rbrace$ the **range** of $F$.

We say that $F$ is **continuous** in $x \in U$, if for all $\epsilon > 0$ there exists a $\delta > 0$ such that $\lVert F(x) - F(y) \rVert_Y < \epsilon$ for all $z \in U$ with $\lVert x - z \rVert_X < \delta$. We say $F$ is **Lipschitz continuous**, if there exists a $L > 0$ such that $\lVert F(x_1) - F(x_2) \rVert_Y \le L \lVert x_1 - x_2 \rVert_X$ for all $x_1, x_2 \in U$.

A map $F : X \to Y$ is continuous iff $x_n \to x$ implies $F(x_n) \to F(x)$, and **closed**, if for any sequence $x_n \to x$ with $F(x_n) \to y$ it follows that $F(x) = y$.

If $F : X \to Y$ is linear, i.e., $F(\lambda_1 x_1 + \lambda_2 x_2) = \lambda_1 F(x_1) + \lambda_2 F(x_2)$ for all $x_1, x_2 \in X$, $\lambda_1, \lambda_2 \in \mathbb{R}$, then the continuity of $F$ is equivalent to the condition that there exists a constant $C > 0$ such that $\lVert Fx \rVert_Y \le C \lVert x \rVert_X$ for all $x \in X$. For that reason continuous, linear maps are also called **bounded** and we speak about a bounded, linear operator. (To stress this in the following, we will denote such maps with $A$.)

If $Y$ is complete, then the space of all bounded, linear operators from $X$ to $Y$, denoted by $\mathcal{L}(X, Y)$, is a Banach space with the **operator norm**

$$\lVert A \rVert_{\mathcal{L}(X,Y)} = \sup_{x \in X \setminus \lbrace 0 \rbrace} \frac{\lVert Ax \rVert_Y}{\lVert x \rVert_X} = \sup_{\lVert x \rVert_X \le 1} \lVert Ax \rVert_Y.$$

It is equal to the smallest constant $C$ in the boundedness condition above. As for linear operators in $\mathbb{R}^n$ we say that $A$ is

* **injective**, if $\mathcal{N}(A) = \lbrace 0 \rbrace$,
* **surjective**, if $\mathcal{R}(A) = Y$,
* **bijective**, if $A$ is injective and surjective.

If $A \in \mathcal{L}(X, Y)$ is bijective, the inverse $A^{-1} : Y \to X$ is bounded iff there exists a $c > 0$ such that $c \lVert x \rVert_X \le \lVert Ax \rVert_Y$ for all $x \in X$, and $\lVert A^{-1} \rVert_{\mathcal{L}(Y,X)} = c^{-1}$ for the largest possible $c$. Whether this is the case follows from the following fundamental theorem of functional analysis.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.1.2</span><span class="math-callout__name">(Closed Graph Theorem)</span></p>

Let $X, Y$ be Banach spaces. A map $F : X \to Y$ is continuous iff $F$ is closed.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary A.1.3</span></p>

Let $X, Y$ be Banach spaces and $A \in \mathcal{L}(X, Y)$ bijective. Then $A^{-1} : Y \to X$ is continuous.

</div>

A sequence $(A_n) \subset \mathcal{L}(X, Y)$ converges to $A \in \mathcal{L}(X, Y)$

* **pointwise**, if $A_n x \to Ax$ for all $x \in X$ (strong convergence in $Y$);
* **uniformly**, if $A_n \to A$ (strong convergence in $\mathcal{L}(X, Y)$).

Uniform convergence implies pointwise convergence.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.1.4</span><span class="math-callout__name">(Banach-Steinhaus)</span></p>

Let $X$ be a Banach space and $Y$ a normed vector space, and suppose that $(A_i)\_{i \in I} \subset \mathcal{L}(X, Y)$ is a family of pointwise bounded, linear operators, i.e., for all $x \in X$ there exists $M_x > 0$ such that $\sup_{i \in I} \lVert A_i x \rVert \le M_x$. Then

$$\sup_{i \in I} \lVert A_i \rVert_{\mathcal{L}(X,Y)} < \infty.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary A.1.5</span></p>

Let $X, Y$ be Banach spaces and $(A_n) \subset \mathcal{L}(X, Y)$. Then the following three statements are equivalent:

1. $(A_n)$ converges uniformly on compact subsets of $X$,
2. $(A_n)$ converges pointwise on $X$,
3. $(A_n)$ converges pointwise on a dense subset $U \subset X$ and $\sup_{n \in \mathbb{N}} \lVert A_n \rVert_{\mathcal{L}(X,Y)} < \infty$.

Also, if $A_n$ converges pointwise to $A : X \to Y$ then $A$ is bounded.

</div>

### A.2 Hilbert Spaces, Compact Operators and the Spectral Theorem

Inverse problems can be analysed in Banach spaces, but the theory can be presented more comprehensively in Hilbert spaces. It also provides a clearer link to underdetermined or ill-conditioned linear equation systems in $\mathbb{R}^n$, which have been covered, e.g., in introductory numerical analysis courses.

#### A.2.1 Scalar Product and Weak Convergence

Hilbert spaces distinguish themselves from Banach spaces by having one additional structure: a map $\langle \cdot, \cdot \rangle : X \times X \to \mathbb{R}$, called a **scalar product**, with the properties

1. $\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle$ for all $x, y, z \in X$, $\alpha, \beta \in \mathbb{R}$,
2. $\langle x, y \rangle = \langle y, x \rangle$ for all $x, y \in X$,
3. $\langle x, x \rangle \ge 0$ for all $x \in X$, with $\langle x, x \rangle = 0$ iff $x = 0$.

The scalar product induces a norm $\lVert x \rVert_X := \sqrt{\langle x, x \rangle_X}$ that satisfies the Cauchy-Schwarz inequality $\lvert \langle x, y \rangle_X \rvert \le \lVert x \rVert_X \lVert y \rVert_X$. A Banach space with a scalar product $(X, \langle \cdot, \cdot \rangle_X)$ is called a **Hilbert space**.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.2.1</span><span class="math-callout__name">(Examples of Scalar Products)</span></p>

We can define the following scalar products on

1. $X = \mathbb{R}^n$, $n \in \mathbb{N}$:

   $$\langle x, y \rangle = \sum_{j=1}^n x_j y_j \qquad \text{for all } x, y \in X;$$

2. $X = l^2$:

   $$\langle x, y \rangle = \sum_{j=1}^\infty x_j y_j \qquad \text{for all } x, y \in X;$$

3. $X = L^2(\Omega)$, $\Omega \subset \mathbb{R}^n$:

   $$\langle f, g \rangle = \int_\Omega f g \, \mathrm{d}\lambda \qquad \text{for all } f, g \in X.$$

In all cases the scalar product also induces a canonical norm on $X$.

</div>

The scalar product also allows to define a further notion of convergence: a sequence $(x_n) \subset X$ **converges weakly** (in $X$) to $x \in X$ — we write $x_n \rightharpoonup x$ — if $\langle x_n, z \rangle_X \to \langle x, z \rangle_X$ for all $z \in X$. It generalises coordinatewise convergence in $\mathbb{R}^n$. In finite dimensional spaces strong and weak convergence are equivalent. In infinite dimensional spaces, strong convergence implies weak convergence, but the converse is not true. However, if a sequence $(x_n)$ converges weakly to $x \in X$ and in addition $\lVert x_n \rVert_X \to \lVert x \rVert_X$, then $(x_n)$ converges also strongly to $x$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.2.2</span><span class="math-callout__name">(Bolzano-Weierstrass)</span></p>

Every bounded sequence in a Hilbert space has a weakly convergent subsequence.

</div>

Conversely, every weakly convergent sequence is bounded.

Let us now consider bounded, linear operators $A \in \mathcal{L}(X, Y)$ on Hilbert spaces $X, Y$. Of particular interest is the special case $Y = \mathbb{R}$, i.e., the space $\mathcal{L}(X, \mathbb{R})$ of **bounded, linear functionals** on $X$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.2.3</span><span class="math-callout__name">(Riesz-Fischer)</span></p>

Let $X$ be a Hilbert space. For every functional $\lambda \in \mathcal{L}(X, \mathbb{R})$ there exists a unique $z_\lambda \in X$ with $\lVert z_\lambda \rVert_X = \lVert \lambda \rVert_{\mathcal{L}(X, \mathbb{R})}$ such that $\lambda(x) = \langle z_\lambda, x \rangle_X$ for all $x \in X$.

</div>

This theorem allows to define an **adjoint operator** $A^* \in \mathcal{L}(Y, X)$ for every linear operator $A \in \mathcal{L}(X, Y)$ such that $\langle A^* y, x \rangle_X = \langle y, Ax \rangle_Y$ for all $x \in X$, $y \in Y$, and $(A^\ast)^\ast = A$, $\lVert A^\ast \rVert_{\mathcal{L}(Y,X)} = \lVert A \rVert_{\mathcal{L}(X,Y)}$, $\lVert A^\ast A \rVert_{\mathcal{L}(X,X)} = \lVert A \rVert_{\mathcal{L}(X,Y)}^2$. If $A^\ast = A$, then $A$ is called **self-adjoint**.

#### A.2.2 Orthogonality and Orthogonal Systems

Two elements $x, y \in X$ are **orthogonal**, if $\langle x, y \rangle_X = 0$. For any subset $U \subset X$, the **orthogonal complement** $U^\perp := \lbrace x \in X : \langle x, u \rangle_X = 0 \text{ for all } u \in U \rbrace$ is a closed subspace of $X$. In particular, we have $X^\perp = \lbrace 0 \rbrace$ and $U \subset (U^\perp)^\perp$. If $U$ is a closed subspace, then $U = (U^\perp)^\perp$ (and thus also $\lbrace 0 \rbrace^\perp = X$) and there exists an **orthogonal decomposition** $X = U \oplus U^\perp$, i.e., each element $x \in X$ can be uniquely decomposed as $x = u + u_\perp$, $u \in U$, $u_\perp \in U^\perp$.

The assignment $x \mapsto u$ defines the **orthogonal projection** $P_U \in \mathcal{L}(X, X)$ onto $U$. It has the following properties:

1. $P_U$ is self-adjoint;
2. $\lVert P_U \rVert_{\mathcal{L}(X,U)} = 1$ for $U \neq \lbrace 0 \rbrace$;
3. $\mathrm{Id} - P_U = P_{U^\perp}$;
4. $\lVert x - P_U x \rVert_X = \min_{u \in U} \lVert x - u \rVert_X$;
5. $z = P_U x$ iff $z \in U$ and $z - u \in U^\perp$.

If the subspace $U$ is not closed, we only have $(U^\perp)^\perp = \overline{U} \supset U$. Thus, for any $A \in \mathcal{L}(X, Y)$: $\mathcal{R}(A)^\perp = \mathcal{N}(A^\ast)$ and thus $\mathcal{N}(A^\ast)^\perp = \overline{\mathcal{R}(A)}$, and $\mathcal{R}(A^\ast)^\perp = \mathcal{N}(A)$ and thus $\mathcal{N}(A)^\perp = \overline{\mathcal{R}(A^\ast)}$. The kernel of a bounded linear operator is always closed and $A$ is injective iff $\mathcal{R}(A^\ast)$ is dense in $X$.

A set $U \subset X$ consisting of pairwise orthogonal elements is called an **orthogonal system**. If additionally $\langle x, y \rangle_X = \delta_{xy}$ for all $x, y \in U$, we speak of an **orthonormal system**. An orthonormal system is **complete** (called an **orthonormal basis (ONB)**) if there exists no orthonormal system $V \subset X$ with $U \subsetneq V$. Every orthonormal system $U \subset X$ satisfies the **Bessel inequality**

$$\sum_{y \in U} |\langle x, y \rangle_X|^2 \le \lVert x \rVert_X^2 \qquad \text{for all } x \in X, \tag{A.1}$$

with only countably many nonzero terms in the sum. In the case of equality, $U$ is complete and $x = \sum_{y \in U} \langle x, y \rangle_X y$ for all $x \in X$. Every Hilbert space has an ONB. If the ONB is countable, the Hilbert space is called **separable**. In that case, there exists a sequence $(u_n) \subset U$ such that $U = \operatorname{span}(u_n)$. It follows from the Bessel inequality that the sequence $(u_n)$ converges weakly to zero (but not strongly, since $\lVert u_n \rVert_X = 1$).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.2.4</span><span class="math-callout__name">(ONB of $L^2([0,1])$)</span></p>

Let $X = L^2([0, 1])$. An ONB $(u_n)$ for $X$ is given by

$$u_n = \begin{cases} \sqrt{2} \sin(\pi (n+1) x) & n > 0 \text{ odd} \\ \sqrt{2} \cos(\pi n x) & n > 0 \text{ even} \\ 1 & n = 0. \end{cases}$$

</div>

Every closed subspace $U \subset X$ has an ONB $(u_n)$, which can be used to define the orthogonal projection onto $U$ by

$$P_U x = \sum_{j=1}^{\infty} \langle x, u_j \rangle_X u_j.$$

#### A.2.3 Compact Operators and the Spectral Theorem

In the same way as Hilbert spaces are a generalisation of finite dimensional vector spaces, compact operators are the infinite dimensional analogon of matrices.

An operator $A : X \to Y$ is said to be **compact**, if the image of any bounded sequence $(x_n) \subset X$ has a convergent subsequence $(Ax_{n_k})\_{k \ge 1} \subset Y$. Equivalently: $A$ is compact iff $A$ maps weakly convergent sequences in $X$ to strongly convergent sequences in $Y$ (also called **completely continuous**). Compact operators will be denoted by $K$.

Clearly every linear operator is compact if $Y$ is finite dimensional. In particular, the identity operator $\mathrm{Id} : X \to X$ is compact iff $\dim(X) < \infty$. The space $\mathcal{K}(X, Y)$ of all compact operators from $X$ to $Y$ is a closed subspace of $\mathcal{L}(X, Y)$ (and hence a Banach space with the operator norm), and the limit of a sequence of linear operators with finite dimensional range is also compact. If $A, S \in \mathcal{L}(X, Y)$ and at least one of the two operators is compact, then $S \circ A$ is also compact. Finally, $A^*$ is compact iff $A$ is compact (Schauder Fixed-Point Theorem).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.2.5</span><span class="math-callout__name">(Integral Operators)</span></p>

A canonical example for compact operators are **integral operators**. Let $X = Y = L^2([0, 1])$ and, for a given kernel function $k \in L^2([0, 1] \times [0, 1])$, consider the linear operator $K : L^2([0, 1]) \to L^2([0, 1])$ defined by

$$[Kx](t) = \int_0^1 k(s, t) x(s) \, \mathrm{d}s \qquad \text{for almost all } t \in [0, 1].$$

Then $\lVert K \rVert_{\mathcal{L}(X,X)} \le \lVert k \rVert_{L^2([0,1])}$, i.e. $K$ is a bounded operator from $L^2([0, 1])$ to $L^2([0, 1])$. Since the kernel function $k \in L^2([0, 1]^2)$ is measurable, there exists a sequence $(k_n) \subset L^2([0, 1]^2)$ of simple piecewise constant functions such that $k_n \to k$ in $L^2([0, 1]^2)$. The operators $K_n$ with kernel $k_n$ have finite dimensional range, so $K = \lim_{n \to \infty} K_n$ is compact.

For the adjoint operator $K^* \in \mathcal{L}(X, X)$ we have $[K^\ast y] (s) = \int_0^1 k(s, t) y(t) \, \mathrm{d}t$. Hence, an integral operator is self-adjoint iff the kernel function is symmetric, i.e., $k(s, t) = k(t, s)$ for almost all $s, t \in [0, 1]$.

</div>

The analogy between compact operators and matrices is primarily related to the fact that compact linear operators have only countably many eigenvalues. (For bounded linear operators that is not necessarily the case!) We even have the following extension of the Schur decomposition for matrices.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.2.6</span><span class="math-callout__name">(Spectral Theorem)</span></p>

Let $X$ be a Hilbert space and let $K \in \mathcal{K}(X, X)$ be self-adjoint. Then there exists an orthonormal system $(u_n) \subset X$ and a null sequence $(\lambda_n) \subset \mathbb{R} \setminus \lbrace 0 \rbrace$ with

$$Kx = \sum_{n=1}^{\infty} \lambda_n \langle x, u_n \rangle_X u_n \qquad \text{for all } x \in X.$$

The sequence $(u_n)$ forms an ONB for $\overline{\mathcal{R}(K)}$.

</div>

Letting $x = u_n$, we can see that $u_n$ is an eigenvector corresponding to the eigenvalue $\lambda_n$ with $Ku_n = \lambda_n u_n$. Typically, the eigenvalues and the corresponding eigenvectors are ordered such that $\lvert \lambda_1 \rvert \ge \lvert \lambda_2 \rvert \ge \ldots \ge 0$. It follows that $\lVert K \rVert_{\mathcal{L}(X,X)} = \lvert \lambda_1 \rvert$.

## Appendix B: Basic Concepts of Measure Theory

### B.1 $\sigma$-Algebras

In the following $\Omega$ denotes a set, interpreted as the collection of all "elementary events". We write $2^\Omega$ for its power set (the set of all subsets of $\Omega$). A $\sigma$-algebra is a specific subset of $2^\Omega$, on which we will be able to define measures. For $A \subseteq \Omega$ we denote its complement by $A^c := \Omega \setminus A$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.1.1</span><span class="math-callout__name">($\sigma$-Algebra)</span></p>

We call $\mathcal{A} \subseteq 2^\Omega$ a **$\sigma$-algebra** iff

1. $\Omega \in \mathcal{A}$,
2. $A \in \mathcal{A}$ implies $A^c \in \mathcal{A}$,
3. $A_i \in \mathcal{A}$ for all $i \in \mathbb{N}$ implies $\bigcup_{i \in \mathbb{N}} A_i \in \mathcal{A}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.1.2</span><span class="math-callout__name">(Measurable Space)</span></p>

For $\Omega \neq \emptyset$ and a $\sigma$-algebra $\mathcal{A}$ on $\Omega$, the tuple $(\Omega, \mathcal{A})$ is called a **measurable space**. A subset $A \subseteq \Omega$ is called **measurable** iff it belongs to $\mathcal{A}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark B.1.3</span></p>

Properties (i) and (ii) imply $\Omega^c = \emptyset \in \mathcal{A}$, and (iii) and (ii) imply $(\bigcup_{i \in \mathbb{N}} A_i^c)^c = \bigcap_{i \in \mathbb{N}} A_i \in \mathcal{A}$. In particular $\bigcap_{i \in \mathbb{N}} A_i \in \mathcal{A}$ whenever $A_i \in \mathcal{A}$ for all $i \in \mathbb{N}$.

Recall that $(\Omega, \mathcal{T})$ is called a **topological space**, if $\mathcal{T} \subseteq 2^\Omega$ is a topology, i.e. $\mathcal{T}$ is the collection of all "open sets" and satisfies

1. $\emptyset, \Omega \in \mathcal{T}$,
2. $\bigcap_{j=1}^N O_j \in \mathcal{T}$,
3. $\bigcup_{j \in I} O_j \in \mathcal{T}$,

whenever $O_1, \ldots, O_N$ and $(O_j)\_{j \in I}$ belong to $\mathcal{T}$. Here $I$ is an arbitrary index set and need not be countable. On a topological space we can define the Borel $\sigma$-algebra, which is the $\sigma$-algebra generated by the open sets. To introduce it, we need the following result.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition B.1.4</span></p>

Let $\mathcal{E} \subseteq 2^\Omega$ be nonempty. Then

$$\sigma(\mathcal{E}) := \bigcap_{\mathcal{A} \text{ is a } \sigma\text{-algebra s.t. } \mathcal{E} \subseteq \mathcal{A}} \mathcal{A} \tag{B.1}$$

defines a $\sigma$-algebra, called the **$\sigma$-algebra generated by $\mathcal{E}$**.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition B.1.4</summary>

The intersection in (B.1) is not empty since $2^\Omega$ is a $\sigma$-algebra containing $\mathcal{E}$. Let $(\mathcal{A}\_i)\_{i \in I}$ be a family of $\sigma$-algebras ($I$ not necessarily countable). Then it is simple to check that $\bigcap_{i \in I} \mathcal{A}\_i$ (by which we mean $\lbrace A \subseteq \Omega : A \in \mathcal{A}\_i \; \forall i \in I \rbrace$) is again a $\sigma$-algebra, by verifying each item of Def. B.1.1 for this intersection. Hence (B.1) defines a $\sigma$-algebra containing $\mathcal{E}$. Evidently $\sigma(\mathcal{E})$ is the smallest $\sigma$-algebra containing $\mathcal{E}$.

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.1.5</span><span class="math-callout__name">(Borel $\sigma$-Algebra)</span></p>

For a topological space $(\Omega, \mathcal{T})$ we call $\sigma(\mathcal{T})$ the **Borel $\sigma$-algebra** and denote it by $\mathcal{B}(\Omega)$. In case there is no confusion about the topology, we simply say that "$\mathcal{B}$ is the Borel $\sigma$-algebra on $\Omega$". In case $\Omega$ is an open or closed subset of $\mathbb{R}^d$, the Borel $\sigma$-algebra on $\Omega$ is always understood w.r.t. the Euclidean topology on $\mathbb{R}^d$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark B.1.6</span></p>

There exist sets $A \subseteq \mathbb{R}^d$ which do not belong to $\mathcal{B}(\mathbb{R}^d)$, i.e. $\mathcal{B}(\mathbb{R}^d) \neq 2^{\mathbb{R}^d}$. Since complements of open sets are closed (and closed sets are in general not open), the Euclidean topology on $\mathbb{R}^d$ is not a $\sigma$-algebra.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise B.1.7</span></p>

Use Rmk. B.1.6 to show that $\mathcal{B}(\mathbb{R}^d)$ is not a topology.

</div>

We also require the following proposition, which we state without proof (see, e.g., Analysis III). A set $\mathcal{D} \subseteq 2^\Omega$ is called a **Dynkin-system**, iff

* $\Omega \in \mathcal{D}$,
* for $A, B \in \mathcal{D}$ with $A \subset B$ it holds $B \setminus A \in \mathcal{D}$,
* for every countable disjoint pairwise sequence $A_j \in \mathcal{D}$, $j \in \mathbb{N}$, it holds $\bigcup_{j \in \mathbb{N}} A_j \in \mathcal{D}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition B.1.8</span></p>

Let $\mathcal{C} \subseteq 2^\Omega$ satisfy $A \cap B \in \mathcal{C}$ for every $A, B \in \mathcal{C}$. Then the smallest Dynkin-system containing $\mathcal{C}$ exists and is equal to $\sigma(\mathcal{C})$.

</div>

### B.2 Measures

We are now in a position to introduce measures. These are functions assigning nonnegative numbers to each set in $\mathcal{A}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.2.1</span><span class="math-callout__name">(Measure)</span></p>

Let $\mathcal{A}$ be a $\sigma$-algebra on $\Omega \neq \emptyset$. A function $\mu : \mathcal{A} \to [0, \infty]$ is called a **measure** iff

1. $\mu(\emptyset) = 0$,
2. ($\sigma$-additivity) if $A_i \in \mathcal{A}$ for all $i \in \mathbb{N}$ and $A_i \cap A_j = \emptyset$ for $i \neq j$, then

   $$\mu\left( \bigcup_{i \in \mathbb{N}} A_i \right) = \sum_{i \in \mathbb{N}} \mu(A_i).$$

A measure is called **$\sigma$-finite** if there exist $(A_j)\_{j \in \mathbb{N}} \in \mathcal{A}$ such that $\Omega = \bigcup_{j \in \mathbb{N}} A_j$ and $\mu(A_j) < \infty$ for all $j \in \mathbb{N}$. A measure $\mu$ with $\mu(\Omega) = 1$ is called a **probability measure**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.2.2</span><span class="math-callout__name">(Measure Space)</span></p>

For a set $\Omega \neq \emptyset$, a $\sigma$-algebra $\mathcal{A}$ on $\Omega$ and a measure $\mu$ on $\mathcal{A}$ we call the triple $(\Omega, \mathcal{A}, \mu)$ a **measure space**. If $\mathbb{P}$ is a probability measure on $(\Omega, \mathcal{A})$, we call $(\Omega, \mathcal{A}, \mathbb{P})$ a **probability space**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example B.2.3</span></p>

Let $\Omega = \lbrace \omega_1, \ldots, \omega_N \rbrace$ be a finite set and let $0 \le p_j \le 1$ for $j = 1, \ldots, N$ such that $\sum_{j=1}^N p_j = 1$. Set $\mathcal{A} := 2^\Omega$. Then $\mu(A) := \sum_{\omega_j \in A} p_j$ defines a probability measure on $(\Omega, \mathcal{A})$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example B.2.4</span></p>

Let $f : \mathbb{R}^n \to \mathbb{R}$ be nonnegative and integrable with $\int_{\mathbb{R}^n} f(x) \, \mathrm{d}x = 1$. Then

$$\mu(A) := \int_A f(x) \, \mathrm{d}x$$

defines a probability measure on $(\mathbb{R}^n, \mathcal{B})$.

</div>

The following theorem is often useful, as it allows to check for equality of two measures.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.2.5</span></p>

Let $(\Omega, \mathcal{A}, \mu)$ be a $\sigma$-finite measure space. Let $\mathcal{E} \subseteq 2^\Omega$ satisfy $A \cap B \in \mathcal{E}$ for all $A, B \in \mathcal{E}$ as well as $\sigma(\mathcal{E}) = \mathcal{A}$. If there exists a sequence $(E_n)\_{n \in \mathbb{N}}$ with $\Omega = \bigcup_{n \in \mathbb{N}} E_n$, $E_n \subseteq E_{n+1}$ and $\mu(E_n) < \infty$ for all $n$, then $\mu$ is uniquely defined through $\mu(E)$ for all $E \in \mathcal{E}$.

</div>

#### B.2.1 Product Measures

For measurable spaces $(\Omega_j, \mathcal{A}_j)_{j=1}^n$ the $\sigma$-algebra

$$\textstyle\bigotimes_{j=1}^n \mathcal{A}_j := \sigma\left( \lbrace \times_{j=1}^n A_j \,:\, A_j \in \mathcal{A}_j \; \forall j \rbrace \right)$$

is called the **product $\sigma$-algebra** on the space $\times_{j=1}^n \Omega_j$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.2.6</span><span class="math-callout__name">(Product Measure)</span></p>

Let $(\Omega_j, \mathcal{A}_j, \mu_j)$ for $j = 1, \ldots, n$ be a family of $\sigma$-finite measure spaces. Then there exists a unique measure $\mu$ on $(\times_{j=1}^n \Omega_j, \bigotimes_{j=1}^n \mathcal{A}_j)$ such that

$$\mu\left( \times_{j=1}^n A_j \right) = \prod_{j=1}^n \mu_j(A_j) \qquad \forall A_j \in \mathcal{A}_j.$$

We call $\mu$ the **product measure** and use the notation $\mu = \bigotimes_{j=1}^n \mu_j$.

</div>

The product measure can also be constructed for $n = \infty$: Consider the $\sigma$-algebra $\mathcal{A} := \sigma(\mathcal{E})$ generated by the cylindrical sets

$$\mathcal{E} := \lbrace \times_{j \in \mathbb{N}} A_j \,:\, A_j \in \mathcal{B}(\mathbb{R}) \rbrace.$$

Suppose that $(\mu_j)\_{j \in \mathbb{N}}$ is a family of probability measures on $\mathbb{R}$. Then there is a unique measure $\mu$ on $(\mathbb{R}^{\mathbb{N}}, \mathcal{A})$ satisfying $\mu(\times_{j=1}^n A_j \times \times_{i \in \mathbb{N}} \mathbb{R}) = \prod_{j=1}^n \mu_j(A_j)$.

One of the most important measures is the **Lebesgue measure** on the measurable space $(\mathbb{R}, \mathcal{B})$, which satisfies

$$\lambda_1((a, b]) = b - a \qquad \forall b > a. \tag{B.2}$$

Since $\mathcal{E} = \lbrace (a, b] : a < b \rbrace$ generates $\mathcal{B}(\mathbb{R})$, i.e. $\sigma(\mathcal{E}) = \mathcal{B}(\mathbb{R})$, Thm. B.2.5 implies that the Lebesgue measure is unique. Furthermore, by Thm. B.2.6 there is a unique measure $\lambda_d = \bigotimes_{j=1}^d \lambda$ on $(\mathbb{R}^d, \bigotimes_{j=1}^d \mathcal{B}(\mathbb{R}))$ with the property

$$\lambda_d\left( \times_{j=1}^d (a_j, b_j] \right) = \prod_{j=1}^d (b_j - a_j) \qquad \forall a_j < b_j, \tag{B.3}$$

which is again called the Lebesgue measure. Whenever $d$ is clear from the context, we drop the index and simply write $\lambda$ instead of $\lambda_d$. We also mention that $\bigotimes_{j=1}^d \mathcal{B}(\mathbb{R}) = \mathcal{B}(\mathbb{R}^d)$ (exercise).

### B.3 Measurability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.3.1</span><span class="math-callout__name">(Measurability)</span></p>

Let $(\Omega_1, \mathcal{A}\_1)$ and $(\Omega_2, \mathcal{A}\_2)$ be two measurable spaces. A function $f : \Omega_1 \to \Omega_2$ is called **$\mathcal{A}\_1/\mathcal{A}_2$-measurable** iff $f^{-1}(A_2) \in \mathcal{A}\_1$ for all $A_2 \in \mathcal{A}\_2$. If there is no confusion about $\mathcal{A}\_2$ and/or $\mathcal{A}\_1$ we also say that $f$ is **$\mathcal{A}\_1$-measurable** or simply **measurable**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark B.3.2</span></p>

Note that measurability of a function depends only on the $\sigma$-algebras, but no measure needs to be defined. If $\mathcal{A}\_1$ and $\mathcal{A}\_2$ are both the Borel $\sigma$-algebras, then we say that $f : \Omega_1 \to \Omega_2$ is **Borel measurable**. To check for Borel measurability it suffices to consider preimages of open sets; more generally we have the following.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition B.3.3</span></p>

Let $(\Omega_1, \mathcal{A}\_1)$ and $(\Omega_2, \mathcal{A}\_2)$ be two measurable spaces and assume that $\mathcal{A}\_2 = \sigma(\mathcal{E})$ for some $\mathcal{E} \subseteq 2^{\Omega_2}$. A function $f : \Omega_1 \to \Omega_2$ is $\mathcal{A}\_1/\mathcal{A}\_2$-measurable iff $f^{-1}(E) \in \mathcal{A}\_1$ for all $E \in \mathcal{E}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition B.3.3</summary>

Measurability implies that $f^{-1}(E) \in \mathcal{A}_1$ for all $E \in \mathcal{E} \subseteq \sigma(\mathcal{E})$. To show the other direction, define

$$\mathcal{C} := \lbrace B \subseteq \Omega_2 \,:\, f^{-1}(B) \in \mathcal{A}_1 \rbrace.$$

For any $B \in \mathcal{C}$,

$$f^{-1}(B^c) = \lbrace \omega \in \Omega_1 \,:\, f(\omega) \in B^c \rbrace = \lbrace \omega \in \Omega_1 \,:\, f(\omega) \notin B \rbrace = \Omega_1 \setminus f^{-1}(B) = (f^{-1}(B))^c \in \mathcal{A}_1,$$

and thus $B^c \in \mathcal{C}$. Similarly, for all $B_i \in \mathcal{C}$,

$$f^{-1}\left( \bigcup_{i \in \mathbb{N}} B_i \right) = \bigcup_{i \in \mathbb{N}} f^{-1}(B_i),$$

and thus $\bigcup_{i \in \mathbb{N}} B_i \in \mathcal{C}$ whenever $B_i \in \mathcal{C}$ for all $i \in \mathbb{N}$. Hence $\mathcal{C}$ is a $\sigma$-algebra on $\Omega_2$. By assumption every $E \in \mathcal{E}$ belongs to $\mathcal{C}$. Since $\sigma(\mathcal{E})$ is the smallest $\sigma$-algebra containing $\mathcal{E}$ it holds $\mathcal{C} \supseteq \sigma(\mathcal{E})$. Thus $f^{-1}(B) \in \mathcal{A}\_1$ for all $B \in \sigma(\mathcal{E}) = \mathcal{A}\_2$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark B.3.4</span></p>

The previous proposition implies in particular that continuous functions are always Borel-measurable.

</div>

Let $V$ denote a Banach space over the field $\mathbb{R}$ (most results are easily generalized to Banach spaces over $\mathbb{C}$). To give meaning to integrals over $V$-valued functions, we require a stronger notion of measurability. A function $f : \Omega \to V$, defined on the measurable space $(\Omega, \mathcal{A})$, is called **$\mathcal{A}$-simple** iff

$$f(\omega) = \sum_{j=1}^N v_j \mathbb{1}_{A_j}(\omega)$$

for finite $N \in \mathbb{N}$, measurable $A_j \in \mathcal{A}$ with $A_i \cap A_j = \emptyset$ for all $i \neq j$ and $v_j \in V$. Here $\mathbb{1}\_{A_j}(\omega)$ denotes the indicator function, that is $\mathbb{1}\_{A_j}(\omega) = 1$ if $\omega \in A_j$ and $\mathbb{1}\_{A_j}(\omega) = 0$ otherwise.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.3.5</span><span class="math-callout__name">(Strong Measurability)</span></p>

A function $f : \Omega \to V$ is **strongly measurable** iff there exists a sequence $(f_n)\_{n \in \mathbb{N}}$ of $\mathcal{A}$-simple functions such that $\lim_{n \to \infty} f_n = f$ pointwise.

</div>

As the name suggests, strong measurability is in general indeed stronger than measurability. In case $V$ is a separable Banach space, the two notions are in fact equivalent. This follows by Pettis measurability theorem. Recall that $V$ is called **separable** if there exists a countable dense subset of $V$. A function $f : \Omega \to V$ is called **separably valued** if it takes values in a separable subspace $V_0 \subseteq V$. If $V$ is separable, then any $f : \Omega \to V$ is necessarily separably valued.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition B.3.6</span></p>

Let $(\Omega, \mathcal{A})$ be a measurable space and let $f_n : \Omega \to \mathbb{R}$ for $n \in \mathbb{N}$ be a sequence of $\mathcal{A}$-measurable functions. Then

* if $f(\omega) := \sup_{n \in \mathbb{N}} f_n(\omega) \in \mathbb{R}$ for all $\omega \in \Omega$, then $f$ is $\mathcal{A}$-measurable,
* if $f(\omega) := \inf_{n \in \mathbb{N}} f_n(\omega) \in \mathbb{R}$ for all $\omega \in \Omega$, then $f$ is $\mathcal{A}$-measurable,
* if $f(\omega) := \lim_{n \to \infty} f_n(\omega) \in \mathbb{R}$ for all $\omega \in \Omega$, then $f$ is $\mathcal{A}$-measurable.

</div>

The proof is left as an exercise (Hint: use that $\mathcal{E} = \lbrace (a, \infty) : a \in \mathbb{R} \rbrace$ generates $\mathcal{B}(\mathbb{R})$ and write $\lim_n f_n = \sup_{n \in \mathbb{N}} \inf_{m \ge n} f_m$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.3.7</span><span class="math-callout__name">(Pettis Measurability Theorem, First Version)</span></p>

Let $(\Omega, \mathcal{A})$ be a measurable space. For $f : \Omega \to V$ the following are equivalent:

1. $f$ is strongly measurable,
2. $f$ is separably valued and $\langle f, v' \rangle$ is $\mathcal{A}$-measurable for every $v' \in V'$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem B.3.7</summary>

**(i) $\Rightarrow$ (ii):** Let $(f_n)\_{n \in \mathbb{N}}$ be a sequence of $\mathcal{A}$-simple functions converging pointwise to $f$, and let $V_0$ be the closed subspace spanned by the countably many values taken by the functions $(f_n)\_{n \in \mathbb{N}}$. Then $V_0$ is separable and $f : \Omega \to V_0$. Furthermore each $\langle f, v' \rangle : \Omega \to \mathbb{R}$ is $\mathcal{A}$-measurable as the pointwise limit of the $\mathcal{A}$-measurable functions $\langle f_n, v' \rangle$ by Prop. B.3.6.

**(ii) $\Rightarrow$ (i):** Let $V_0$ be a separable subspace of $V$ such that $f : \Omega \to V_0$. First we show that there exists a sequence $(v_n')\_{n \in \mathbb{N}} \subseteq V'$ such that for all $v \in V_0$

$$\lVert v \rVert = \sup_{n \in \mathbb{N}} \lvert \langle v, v_n' \rangle \rvert. \tag{B.4}$$

To this end let $(v_n)\_{n \in \mathbb{N}}$ be a dense sequence in $V_0$. By the Hahn-Banach theorem there exist $v_n' \in V'$ such that $\lVert v_n' \rVert = 1$ and $\lVert v_n \rVert = \langle v_n, v_n' \rangle$. Now, for every $v \in V_0$ and $\varepsilon > 0$ there exists $n \in \mathbb{N}$ so large that $\lVert v - v_n \rVert < \varepsilon$. Then

$$\langle v, v_n' \rangle \ge \langle v_n, v_n' \rangle - \lvert \langle v_n - v, v_n' \rangle \rvert \ge \lVert v_n \rVert - \varepsilon \ge \lVert v \rVert - \lVert v - v_n \rVert - \varepsilon \ge \lVert v \rVert - 2\varepsilon.$$

Also note that for any $n \in \mathbb{N}$, $\lvert \langle v, v_n' \rangle \rvert \le \lVert v \rVert \lVert v_n' \rVert = \lVert v \rVert$. Since $\varepsilon > 0$ was arbitrary, the claim (B.4) follows. By the $\mathcal{A}$-measurability of $\omega \mapsto \langle f(\omega), v_n' \rangle$, for each $v_0 \in V_0$

$$\omega \mapsto \lVert f(\omega) - v_0 \rVert = \sup_{n \in \mathbb{N}} \langle f(\omega) - v_0, v_n' \rangle \quad \text{is } \mathcal{A}\text{-measurable.} \tag{B.5}$$

Next define $s_n : V_0 \to \lbrace v_1, \ldots, v_n \rbrace$ as follows: for all $w \in V_0$ let $k(n, w)$ be the smallest integer in $\lbrace 1, \ldots, n \rbrace$ such that $\lVert w - v_k \rVert = \min_{1 \le j \le n} \lVert w - v_j \rVert$, and set $s_n(w) := v_{k(n, w)}$. By density of $(v_n)\_{n \in \mathbb{N}}$ in $V_0$,

$$\lim_{n \to \infty} \lVert w - s_n(w) \rVert = 0 \qquad \forall w \in V_0.$$

Next, set $f_n(\omega) := s_n(f(\omega))$ for all $\omega \in \Omega$. Then for $1 \le k \le n$

$$
\begin{aligned}
\lbrace \omega \in \Omega : f_n(\omega) = v_k \rbrace = \,
&\lbrace \omega \in \Omega : \lVert f(\omega) - v_k \rVert = \min_{1 \le j \le n} \lVert f(\omega) - v_j \rVert \rbrace \\
&\cap \lbrace \omega \in \Omega : \lVert f(\omega) - v_l \rVert > \min_{1 \le j \le n} \lVert f(\omega) - v_j \rVert \;\; \forall l = 1, \ldots, k-1 \rbrace.
\end{aligned}
$$

The set on the right-hand side is in $\mathcal{A}$ due to (B.5). Since $f_n$ takes values in $\lbrace v_1, \ldots, v_n \rbrace$, we conclude that $f_n$ is $\mathcal{A}$-simple. The proof is finished since for every $\omega \in \Omega$

$$\lim_{n \to \infty} \lVert f_n(\omega) - f(\omega) \rVert = \lim_{n \to \infty} \lVert s_n(f(\omega)) - f(\omega) \rVert = 0.$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary B.3.8</span></p>

The pointwise limit of a sequence of strongly $\mathcal{A}$-measurable functions is strongly $\mathcal{A}$-measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Corollary B.3.8</summary>

Let $\lim_{n \to \infty} f_n = f$ pointwise, where each $f_n$ is strongly $\mathcal{A}$-measurable, and thus takes values in a separable subspace $V_n \subseteq V$. The closure $V_0$ of $\bigcup_{n \in \mathbb{N}} V_n$ is separable, and thus $f$ is separably valued. Moreover, Pettis theorem implies $\langle f_n, v' \rangle : \Omega \to \mathbb{R}$ is measurable for every $n$ and every $v' \in V'$. Now, $\lim_{n \to \infty} \langle f_n, v' \rangle = \langle f, v' \rangle$ for every $v' \in V'$, and since the limit of $\mathbb{R}$-valued $\mathcal{A}$-measurable functions is $\mathcal{A}$-measurable by Prop. B.3.6, we conclude that $\langle f, v' \rangle : \Omega \to \mathbb{R}$ is $\mathcal{A}$-measurable for every $v' \in V'$ so that by Pettis theorem $f$ is strongly measurable.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary B.3.9</span></p>

Let $f : \Omega \to V$ be strongly $\mathcal{A}$-measurable. Let $W$ be another Banach space and let $\phi : V \to W$ be continuous. Then $\phi \circ f : \Omega \to W$ is strongly $\mathcal{A}$-measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Corollary B.3.9</summary>

Let $(f_n)\_{n \in \mathbb{N}}$ be a sequence of simple functions converging pointwise to $f$. Then $\phi \circ f_n$ is a sequence of simple functions converging pointwise to $\phi \circ f$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary B.3.10</span></p>

If $V$ is separable, then measurability implies strong measurability.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Corollary B.3.10</summary>

Since $f : \Omega \to V$ is $\mathcal{A}$-measurable, $\langle f, v' \rangle : \Omega \to \mathbb{R}$ is $\mathcal{A}$-measurable for all $v' \in V'$. Hence Pettis measurability theorem implies the claim.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark B.3.11</span></p>

Cor. B.3.10 shows that "$\mathcal{A}$-measurability and separably valued" implies "strong $\mathcal{A}$-measurability". In fact the two are equivalent (exercise).

</div>

#### B.3.1 Strong $\mu$-Measurability

In this section $(\Omega, \mathcal{A}, \mu)$ is a $\sigma$-finite measure space, that is, $\mu$ is a $\sigma$-finite measure on the measurable space $(\Omega, \mathcal{A})$.

Recall that $f : \Omega \to V$ is $\mu$**-simple** if $f = \sum_{j=1}^n \mathbb{1}\_{A_j} v_j$, where $v_j \in V$ and $A_j \in \mathcal{A}$ such that $\mu(A_j) < \infty$. We say that a property holds $\mu$**-almost everywhere (a.e.)** (or $\mu$-almost surely) if there exists a $\mu$**-null set** $N \in \mathcal{A}$, that is $\mu(N) = 0$, and the property holds on $\Omega \setminus N$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.3.12</span><span class="math-callout__name">(Strong $\mu$-Measurability)</span></p>

A function $f : \Omega \to V$ is **strongly $\mu$-measurable** iff there exists a sequence $(f_n)\_{n \in \mathbb{N}}$ of $\mu$-simple functions converging to $f$ $\mu$-a.e. We call $\tilde{f}$ a $\mu$**-version** of $f$ if $\tilde{f} = f$ $\mu$-a.e. In case there is a $\mu$-version of $f$ that is $\mathcal{A}$-measurable, we say that $f$ is $\mu$**-measurable**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition B.3.13</span></p>

For $f : \Omega \to V$ the following are equivalent:

1. $f$ is strongly $\mu$-measurable,
2. $f$ has a $\mu$-version that is strongly $\mathcal{A}$-measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition B.3.13</summary>

**(i) $\Rightarrow$ (ii):** With $(f_n)\_{n \in \mathbb{N}}$ as in Def. B.3.12 let $N \subseteq \Omega$ be such that $\mu(N) = 0$ and $\lim_{n \to \infty} f_n = f$ pointwise on $\Omega \setminus N$. Then $\mathbb{1}\_{N^c} f_n \to \mathbb{1}\_{N^c} f$ pointwise on $\Omega$. Since $\mathbb{1}\_{N^c} f_n$ are $\mathcal{A}$-simple functions, this shows that $\tilde{f} := \mathbb{1}\_{N^c} f$ is strongly $\mathcal{A}$-measurable, and this function coincides with $f$ $\mu$-a.e.

**(ii) $\Rightarrow$ (i):** Let $\tilde{f}$ be a strongly $\mathcal{A}$-measurable $\mu$-version of $f$ and let $N$ be a $\mu$-null set such that $f = \tilde{f}$ on $N^c$. If $(\tilde{f}_n)\_{n \in \mathbb{N}}$ is a sequence of $\mathcal{A}$-simple functions converging pointwise to $\tilde{f}$, then $\lim_{n \to \infty} \tilde{f}\_n = f$ $\mu$-a.e. Let $\Omega = \bigcup_{n \in \mathbb{N}} A_n$ with $\mu(A_n) < \infty$ for all $n$. Then $f_n := \mathbb{1}\_{A_n} \tilde{f}\_n$ are $\mu$-simple functions and $\lim_{n \to \infty} f_n = f$ $\mu$-a.e.

</details>
</div>

We say that $f : \Omega \to V$ is $\mu$**-separably valued** iff there exists a closed separable subspace $V_0 \subseteq V$ such that $f(\omega) \in V_0$ for $\mu$-a.e. $\omega \in \Omega$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.3.14</span><span class="math-callout__name">(Pettis Measurability Theorem, Second Version)</span></p>

Let $(\Omega, \mathcal{A}, \mu)$ be a $\sigma$-finite measure space. For $f : \Omega \to V$ the following are equivalent:

1. $f$ is strongly $\mu$-measurable,
2. $f$ is $\mu$-separably valued and $\langle f, v' \rangle$ is $\mu$-measurable for every $v' \in V'$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Sketch of proof of Theorem B.3.14</summary>

**(i) $\Rightarrow$ (ii):** By Prop. B.3.13 there exists $\tilde{f}$ such that $f = \tilde{f}$ $\mu$-a.e. and $\tilde{f} : \Omega \to V$ is strongly $\mathcal{A}$-measurable. The statement then follows by Thm. B.3.7.

**(ii) $\Rightarrow$ (i):** This direction can be shown analogous to Thm. B.3.7, with the exception that this time the functions $f_n$ are $\mu$-a.e. equal to functions $\tilde{f}_n$ that are $\mathcal{A}$-simple.

</details>
</div>

Prop. B.3.13 and Corollaries B.3.9 and B.3.10 imply the following two corollaries.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary B.3.15</span></p>

Let $f_n : \Omega \to V$ be a sequence of strongly $\mu$-measurable functions, and let $\lim_{n \to \infty} f_n = f$ $\mu$-a.e. Then $f$ is strongly $\mu$-measurable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary B.3.16</span></p>

Let $f : \Omega \to V$ be strongly $\mu$-measurable and let $W$ be another Banach space. If $\phi : V \to W$ is continuous, then $\phi \circ f : \Omega \to W$ is strongly $\mu$-measurable.

</div>
