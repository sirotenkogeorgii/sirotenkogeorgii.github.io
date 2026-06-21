---
title: Problems from the Numerical Methods for Bayesian Inverse Problems course. Sheet 04
layout: default
noindex: true
tags:
  - inverse-problems
  - bayesian-inference
  - numerical-methods
  - functional-analysis
  - karhunen-loeve-expansion
  - hadamard-product
  - maclaurin-series
  - schur-product-theorem
  - gaussian-random-field
  - random-field
  - covariance-function
  - kernel-function]
  - self-adjoint
  - trace-class-operator
  - eigenfunction
  - mercer-theorem
  - riemann-sum
  - galerkin-method

---

**Table of Contents**
- TOC
{:toc}

## Exercise 4.2 — Gaussian random field I

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intro</span><span class="math-callout__name">(Conventions used throughout)</span></p>


</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.2a</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

Let $V:= H_0^1([0,1])$ and consider the operator $K:= -\frac{d^2}{dx^2}: V\to V'$. Show that $C:=K^{-1}$ is a compact, self-adjoint trace-class operator from $L^2([0,1])\to L^2([0,1])$.

*HINT: Find an orthonormal basis of eigenfunctions of $C$ (or $K$), and write down the spectral decomposition of $C$.*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>What does $H_0^1([0,1])$ space mean</summary>



</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>$C$ is well-defined as an operator $L^2 \to L^2$</summary>



</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Spectral decomposition</summary>



</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Compactness</summary>


</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Trace-class operator</summary>


</details>
</div>


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.2b</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

Let $a:\Omega\to L^2([0,1])$ be a RV with distribution $\mathcal{N}(0,1)$. Compute its Karhunen-Loeve expansion

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.2c</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

Determine the covariance function $c:[0,1]^2\to\mathbb{R}$ of $a$ (i.e. $c(x,y) = \mathbb{E}[a_x a_y]$).

*HINT: (i) For $x \in [0,1]$ compute the Fourier series of $\mathbb{1}\_{[−x,x]} \in L^2([−1,1])$ w.r.t. the basis $w\to\frac{\text{exp}(n\pi iw)}{\sqrt{2}}, n \in \mathbb{Z}$. (ii) Use the fact that the $L^2([−1,1])$-inner product of two functions is equal to the $\ell^2$-inner product of their Fourier coefficients.*

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.2d</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

Plot several (approximate) samples from the RV a by truncating the Karhunen-Loeve expansion after $n\in\lbrace 10,100,500\rbrace$ terms.

</div>

## Exercise 4.3 - Gaussian random field II

Let $D := [0,1]^2$. Define the covariance function

$$c(x,y) := \exp\left(-80\|x-y\|_2^2\right) \qquad \forall x,y \in D$$

and $T_c : L^2(D) \to L^2(D)$ via

$$T_c f(x) = \int_{[0,1]^2} c(x,y)f(y)\,dy.$$

We wish to determine approximations of iid samples from $\mathcal N(0,T_c)$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.3a</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

Show that $c$ is a positive definite function and conclude that $T_c$ is a self-adjoint positive trace-class operator.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>The function $c$ is positive semi-definite function (Schur product theorem, covariance decomposition)</summary>

From the notes, for the function $c: D\times D\to \mathbb{R}$ to be positive (semi-)definite, it must satisfy

$$\sum_{i,j=1}^{n} s_i s_j c(x_i, x_j) \ge 0 \qquad \forall x_i, x_j \in D, \ \forall s_i, s_j \in \mathbb{R}$$

We decompose the covariance function in the following way

$$c(x,y) := \exp\left(-80\|x-y\|_2^2\right) = \exp\left(-80\|x\|_2^2\right)\exp\left(-80\|y\|_2^2\right)\exp\left(160x^\top y\right)$$

Then

$$\sum_{i,j=1}^{n} s_i s_j c(x_i, x_j) = \sum_{i,j=1}^{n} \underbrace{s_i \exp\left(-80\|x_i\|_2^2\right)}_{\tilde{s_i}} \underbrace{s_j \exp\left(-80\|x_j\|_2^2\right)}_{\tilde{s_j}} \exp\left(160x_i^\top x_j\right) = \sum_{i,j=1}^{n} \tilde{s_i} \tilde{s_j} \exp\left(160x_i^\top x_j\right)$$

The idea (or our will) is to **uncouple** the $\exp$ part. The first attempt is to use the expansion of the exponent:

$$\exp\left(160x_i^\top x_j\right) = \sum_{k=0}^{\infty} \frac{160^k}{k!} (x_i^\top x_j)^k$$

Then

$$\sum_{i,j=1}^{n} \tilde{s_i} \tilde{s_j} \sum_{k=0}^{\infty} \frac{160^k}{k!} (x_i^\top x_j)^k = \sum_{k=0}^{\infty} \frac{160^k}{k!} \underbrace{\sum_{i,j=1}^{n} \tilde{s_i} \tilde{s_j}(x_i^\top x_j)^k}_{(\ast)},$$

where $(\ast) = \text{sum}(\tilde{S} \circ X^k)$, i.e. the the sum of the Hadamard product of the matrices $\tilde{S}\_{ij}:=\tilde{s_i} \tilde{s_j}$ and $X^k:=(x_i^\top x_j)^k$, which is Hadamard power ($X^{\circ k}$ to be precise). The sum $\sum_{k=0}^{\infty} \frac{160^k}{k!} \text{sum}(\tilde{S} \circ X^k) \geq 0$, because $\tilde{S}$ is positive semi-definite and $X^k$ is positive semi-definite matrix, the Hadamard product is again positive semi-definite by Schur product theorem and the sum of all elements of a positive semi-definite matrix is always non-negative.

Or more simpler final argument is

$$
\sum_{i,j}\tilde s_i\tilde s_j\,e^{160\,x_i^\top x_j}
=\sum_{k=0}^\infty\frac{160^k}{k!}\underbrace{\sum_{i,j}\tilde s_i\tilde s_j\,(x_i^\top x_j)^k}_{=\ \tilde s^\top G^{\circ k}\tilde s}
=\sum_{k=0}^\infty\frac{160^k}{k!}\,\tilde s^\top G^{\circ k}\tilde s .
$$

Each summand is a quadratic form of the PSD matrix $G^{\circ k}$ evaluated at $\tilde s$, hence $\ge0$, and every prefactor $160^k/k!>0$. Therefore the total is $\ge0$, i.e. $c$ is positive semi-definite.

**Notes on the approach:**

- **Schur is used once, explicitly.** The key fact $G^{\circ k}\succeq0$ *is* the Schur product theorem; in the original draft it was assumed silently. Naming it there is the essential repair.
- **No need for $\tilde S\circ X^k$ plus a sum-of-entries lemma.** With $\tilde S=\tilde s\tilde s^\top$ one has the identity $\operatorname{sum}(\tilde S\circ G^{\circ k})=\tilde s^\top G^{\circ k}\tilde s$, so the quadratic form already delivers $\ge0$ from $G^{\circ k}\succeq0$ alone — one fewer Hadamard product and no separate lemma.
- **Notation.** Use $G^{\circ k}$ for the *entrywise* power; $X^k$ reads as the matrix power $X\cdots X$, a different object.
- **Scope.** This proves positive *semi*-definiteness, which is what (4.5.3) requires. It does **not** give strict positive-definiteness: the $k=1$ term $\tilde s^\top G\,\tilde s=\lVert\sum_i\tilde s_i x_i\rVert^2$ can vanish for $\tilde s\neq0$ once $n>2$ (the nodes are linearly dependent in $\mathbb R^2$). Strictness for distinct nodes needs the feature-map / linear-independence-of-Gaussians argument instead.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>The function $c$ is positive definite function</summary>

The proof above only shows the positive *semi*-definiteness. It does **not** give strict positive-definiteness: the $k=1$ term $\tilde s^\top G\,\tilde s=\lVert\sum_i\tilde s_i x_i\rVert^2$ can vanish for $\tilde s\neq0$ once $n>2$ (the nodes are linearly dependent in $\mathbb R^2$). Strictness for distinct nodes needs the feature-map / linear-independence-of-Gaussians argument instead.

Any symmetric kernel function c(x,y) is PSD if it could be defined by an inner product of *something*: 

c:= \langle \Phi(x), \Phi(y) \rangle_{\mathcal{H}},

where \mathcal{H} is any Hilbert space. We will use \mathcal{H}:=L^2 for simplicity. We make the Ansatz g_x(t) = \exp(-\beta \lVert t-x\rVert^2_2), i.e. we place a Gaussian centered at x in the given D:=[0,1]^2 domain space. The consider the inner product in L^2:

\int_{\mathbb{R}^d} g_x(t) g_y(t) dt = \int_{\mathbb{R}^d} \exp(-\beta(\lVert t - x\rVert^2_2 + \lVert t - y\rVert^2_2)) dt = \exp(\beta \lVert x- y \rVert^2_2) \int_{\mathbb{R}^d} \exp(-4\beta(\lVert t - \frac{x+y}{2}\rVert^2_2)) dt,

where we used the following decomposition

\lVert t-x \rVert^2 + \lVert t-y \rVert^2 = 2 \left\lVert t - \frac{x+y}{2} \right\rVert^2 + 2 \left( \frac{\lVert x-y \rVert}{2} \right)^2.

The remaining integral or the right to \exp(\beta \lVert x- y \rVert^2_2) is just a standard Gaussian bell curve centered at the midpoint \frac{x+y}{2}. The total volume under a standard Gaussian curve over the whole space is always a fixed constant, C_d^{-1} (we use the inverse constant for the purely aesthetic reason). We compute the constant as the Euler-Poisson integral: C_d^{-1} = (\frac{\pi}{4\beta})^{d/2}. Then 
​	
\int_{\mathbb{R}^d} g_x(t) g_y(t) dt = C_d^{-1} \cdot \exp(\beta \lVert x- y \rVert^2_2),

which implies that \Phi(x) = \sqrt{C_d}g_x(t), allowing us to cancel the derived constant C_d^{-1}. From the notes we have that the function c:D\times D \to \mathbb{R} is PSD, if it satisfies

\sum_{i,j=1}^{n} s_i s_j c(x_i, x_j) \ge 0 \qquad \forall x_i, x_j \in D, \ \forall s_i, s_j \in \mathbb{R}

Using the derived inner product

\sum_{i,j=1}^{n} s_i s_j c(x_i, x_j) = C_d\Big\lVert\sum_{i=1}^m s_i\,g_{x_i}\Big\rVert_{L^2}^2\ \ge\ 0,

so c is positive semi-definite in the sense of (4.5.3). It is strictly positive definite for distinct nodes, since distinct Gaussian bumps g_{x_1},\dots,g_{x_m} are linearly independent, so the norm vanishes only when s=0.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>$T_c$ is a self-adjoint and positive operator</summary>

Let f \in L^2(D), then

\langle T_c f, f \rangle_L^2(D) = \int_D (T_cf)(x) f(x) dx = \int_{D} \int_{D} c(x,y)f(y)f(x)dydx > 0,

since its an infinte version of \sum_{i,j=1}^{n} s_i s_j c(x_i, x_j) > 0, which we have already proven.

---

Yes, your proof is structurally correct and captures the exact right intuition! You have successfully identified that the double integral is just the continuous limit of the discrete double sum.

If you submit this in an assignment, the grader will know exactly what you mean. However, to make it perfectly mathematically rigorous (and to protect yourself from a picky grader), there are two minor adjustments you should make.

**1. Change $> 0$ to $\ge 0$**

In functional analysis, an operator $T$ is defined as "positive" if $\langle Tf, f \rangle \ge 0$ for all $f$.

* You must include the "or equal to" because the function $f(x)$ could simply be the zero function ($f(x) = 0$ everywhere), in which case the integral is exactly zero.
* Even if $f(x)$ is not the zero function, limits of strictly positive sequences can equal zero. Therefore, standard practice is to write $\ge 0$.

**2. Formalize the "Infinite Version" Argument**

Saying "it's an infinite version" is excellent physical intuition, but the mathematical bridge between a discrete sum and a continuous integral is the **Riemann Sum**. You can make your argument bulletproof by adding one sentence explaining *why* the integral inherits the positivity of the sum.

Here is how you can polish your proof block for a perfect score:

> **Proof that $T_c$ is a positive operator:**
> Let $f \in L^2(D)$. Then the inner product is:
> 
> $$\langle T_c f, f \rangle_{L^2(D)} = \int_D (T_cf)(x) f(x) \, dx = \int_D \int_D c(x,y)f(y)f(x) \, dy \, dx$$
> 
> 
> By definition, this double integral is the limit of its Riemann sums:
> 
> $$\lim_{\Delta x, \Delta y \to 0} \sum_i \sum_j c(x_i, y_j) f(x_i) f(y_j) \Delta x \Delta y$$
> 
> 
> Because we have already proven that the kernel $c(x,y)$ is positive semi-definite, we know that for any discrete set of points and weights $s_i = f(x_i)\Delta x$, the double sum $\sum_i \sum_j c(x_i, y_j) s_i s_j \ge 0$.
> Since the integral is the limit of non-negative Riemann sums, the integral itself must be non-negative. Therefore:
> 
> $$\langle T_c f, f \rangle_{L^2(D)} \ge 0$$
> 
> 
> 
> and $T_c$ is a positive operator.

**The "Feature Map" Alternative**

As a fun bonus, if you used the Feature Map trick we discussed earlier where $c(x,y) = \langle \Phi(x), \Phi(y) \rangle_{L^2}$, you don't even need the Riemann sum argument. You can just factor the integral directly into a squared length:

$$\int_D \int_D \langle \Phi(x), \Phi(y) \rangle_{L^2} f(y) f(x) \, dy \, dx$$

$$= \left\langle \int_D f(x)\Phi(x) \, dx, \int_D f(y)\Phi(y) \, dy \right\rangle_{L^2}$$

$$= \left\lVert \int_D f(x)\Phi(x) \, dx \right\rVert_{L^2}^2 \ge 0$$

Both ways are perfectly valid. Your approach (the Riemann limit) is exactly how it is traditionally taught when proving that positive definite kernels yield positive operators!

</details>
</div>

### (4.3a, cont.) $T_c$ is self-adjoint and positive — Riemann-sum route

<div class="accordion" markdown="1">
<details markdown="1">
<summary>$T_c$ is a self-adjoint and positive operator (more rigorous)</summary>

**Self-adjoint and bounded.** The kernel $c$ is continuous, symmetric and bounded ($\lvert c\rvert\le1$) on the compact set $D\times D$, so $c\in L^2(D\times D)$ and by Lemma 4.5.5 $T_c$ is a compact, self-adjoint (hence bounded) operator on $L^2(D)$. For $f\in L^2(D)$,

$$
\langle T_cf,f\rangle_{L^2(D)}=\int_D (T_cf)(x)\,f(x)\,\mathrm dx=\int_D\!\int_D c(x,y)\,f(y)f(x)\,\mathrm dy\,\mathrm dx .
$$

We show this is $\ge0$. The idea is exactly that this double integral is the continuum limit of the discrete sums $\sum_{i,j}s_is_j\,c(x_i,x_j)\ge0$ already proven — but the limit must be taken with care, because a general $f\in L^2$ has no pointwise values.

**Step 1: continuous $f$.** Suppose first $f\in C(\overline D)$. Then $g(x,y):=c(x,y)f(x)f(y)$ is continuous on the compact set $D\times D$, hence Riemann integrable. Partition $D$ into cells $Q_1,\dots,Q_M$ with sample points $x_m\in Q_m$, and set $s_m:=f(x_m)\,\lvert Q_m\rvert$. The associated Riemann sum is a discrete positive-definite form:

$$
\sum_{m,m'=1}^M c(x_m,x_{m'})\,f(x_m)f(x_{m'})\,\lvert Q_m\rvert\,\lvert Q_{m'}\rvert
=\sum_{m,m'=1}^M s_m s_{m'}\,c(x_m,x_{m'})\ \ge\ 0,
$$

using the positive semi-definiteness of $c$ established above. As the mesh $\to0$ the left side converges to $\iint_{D\times D} g=\langle T_cf,f\rangle$, and a limit of non-negative numbers is non-negative. Hence $\langle T_cf,f\rangle\ge0$ for all $f\in C(\overline D)$.

**Step 2: density.** $C(\overline D)$ is dense in $L^2(D)$, so any $f\in L^2(D)$ is an $L^2$-limit $f_n\to f$ with $f_n\in C(\overline D)$. The functional $f\mapsto\langle T_cf,f\rangle$ is continuous on $L^2(D)$ because $T_c$ is bounded: writing

$$
\langle T_cf,f\rangle-\langle T_cf_n,f_n\rangle=\langle T_cf,\,f-f_n\rangle+\langle T_c(f-f_n),\,f_n\rangle,
$$

the Cauchy–Schwarz bound gives $\lvert\langle T_cf,f\rangle-\langle T_cf_n,f_n\rangle\rvert\le\lVert T_c\rVert\big(\lVert f\rVert+\lVert f_n\rVert\big)\lVert f-f_n\rVert\to0$. Since $\langle T_cf_n,f_n\rangle\ge0$ for every $n$ by Step 1, passing to the limit gives

$$
\langle T_cf,f\rangle\ \ge\ 0\qquad\text{for all }f\in L^2(D),
$$

i.e. $T_c$ is positive. $\qquad\blacksquare$

<details markdown="1">
<summary>Why the two steps are both needed</summary>

- **"Infinite version of the sum" is not by itself a proof.** For arbitrary $f\in L^2(D)$ the values $f(x_m)$ are undefined (equivalence classes), so neither the discrete form nor the convergence of Riemann sums is automatic. Step 1 restricts to continuous $f$, where both *are* legitimate.
- **Boundedness of $T_c$ is what lets the limit pass through.** It makes $f\mapsto\langle T_cf,f\rangle$ continuous, so the non-negativity proven on the dense set $C(\overline D)$ extends to all of $L^2(D)$.
- **Sign.** This route yields $\ge0$, the positive-semidefinite (operator-positive) statement required. The strict inequality $\langle T_cf,f\rangle>0$ for $f\neq0$ is true but comes from all eigenvalues of $T_c$ being positive, not from this limit.
- **Shortcut.** The feature-map identity $c(x,y)=C_d\langle g_x,g_y\rangle_{L^2}$ gives $\langle T_cf,f\rangle=C_d\lVert\!\int_D f(x)g_x\,\mathrm dx\rVert^2\ge0$ in one Fubini step, valid for all $f\in L^2$ directly and bypassing Steps 1–2.

</details>
</div>

</details>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>$T_c$ is a self-adjoint and positive operator (more general proof)</summary>

**$T_c$ is self-adjoint and positive.** $c$ is continuous, symmetric and bounded ($\lvert c\rvert\le1$, so $c\in L^2(D\times D)$ since $D$ is bounded); by Lemma 4.5.5 $T_c$ is compact and self-adjoint. Positivity follows from the *same* factorization, with no need for Riemann-sum limits: for $f\in L^2(D)$, inserting (4.3a) and applying Fubini (legitimate because $\iint_{D\times D}\lvert f(x)f(y)\rvert\,\mathrm dx\,\mathrm dy=\lVert f\rVert_{L^1(D)}^2<\infty$ and $\int g_xg_y=C_d^{-1}c\le C_d^{-1}$ is bounded),

$$
\langle T_cf,f\rangle=\iint_{D\times D} c(x,y)f(x)f(y)\,\mathrm dx\,\mathrm dy
=C_d\int_{\mathbb R^d}\Big(\underbrace{\int_D f(x)\,g_x(t)\,\mathrm dx}_{=:F(t)}\Big)^{2}\mathrm dt
=C_d\,\lVert F\rVert_{L^2(\mathbb R^d)}^2\ \ge\ 0.
$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>$T_c$ is a trace-class operator</summary>

Since $c$ is continuous, symmetric and positive semi-definite on the compact set $D$, Mercer's theorem applies: $T_c$ is trace-class with

$$
\operatorname{tr}(T_c)=\int_D c(x,x)\,\mathrm dx=\int_D e^{-\beta\lVert x - x\rVert^2}\,\mathrm dx = \int_D e^{0}\,\mathrm dx =\lvert D\rvert=1<\infty .\qquad\blacksquare
$$

</details>
</div>


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.3b</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

To approximate the eigenfunctions of $T_c$, we make the Ansatz

$$\varphi(x) = \sum_{j=1}^n \alpha_j b_j(x)$$

for certain functions $(b_j)\_{j=1}^n \subseteq L^2(D)$ and set

$$V_n := \operatorname{span}\lbrace b_j : j=1,\ldots,n\rbrace \subset L^2(D).$$

Now interpret $T_c : V_n \to V_n$ via

$$T_c\varphi = \sum_{j=1}^n \alpha_j b_j,$$

iff

$$\langle T_c\varphi, v\rangle_{L^2(D)} = \left\langle \sum_{j=1}^n \alpha_j b_j, v\right\rangle_{L^2(D)} \qquad \text{for all } v \in V_n.$$

Determine a matrix $M \in \mathbb R^{n \times n}$ such that

$$\varphi = \sum_j \alpha_j b_j$$

is an eigenfunction of $T_c : V_n \to V_n$ with eigenvalue $\lambda > 0$ iff

$$M\alpha = \lambda \alpha, \qquad \text{where } \alpha = (\alpha_j)_{j=1}^n \in \mathbb R^n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>The bigger picture of the exercise</summary>

Moving from the continuous, infinite-dimensional world of functional analysis into the discrete, finite-dimensional world of linear algebra is one of the biggest conceptual leaps in numerical mathematics.

Let's zoom out and break down exactly what Exercise 4.3b is doing, why you are forced to do it, and how it fits into the overarching goal of your course.

**1. What are the Eigenfunctions of the Operator?**

In standard linear algebra, an eigenvector of a matrix $A$ is a vector $v$ that only gets scaled by $A$, meaning $Av = \lambda v$.

In functional analysis, $T_c$ is an integral operator rather than a matrix, but the concept is identical. An eigenfunction $\varphi(x)$ is a function that, when fed into the integral operator, comes out as the exact same function simply scaled by a constant eigenvalue $\lambda$:

$$T_c \varphi = \lambda \varphi$$

Statistically speaking, the eigenfunctions of a covariance operator $T_c$ represent the "principal components" or the independent modes of spatial variation of your random field. The eigenvalue $\lambda$ tells you how much variance (or "energy") that specific shape contributes to the overall randomness.

**2. Cannot We Compute Them Exactly?**

In Exercise 4.2a, you worked with the Laplacian operator $K = -\frac{d^2}{dx^2}$, and you were able to compute the exact, closed-form eigenfunctions using pen and paper (which turned out to be sine waves).

In Exercise 4.3, however, your operator is built using the squared-exponential (Gaussian) covariance kernel $c(x,y) = \exp(-80\|x-y\|_2^2)$ on the bounded 2D square $D = [0,1]^2$.

**There is no elementary closed-form analytical solution for the eigenfunctions of a Gaussian kernel on a bounded, truncated domain.** Because the integral equation $\int_D c(x,y)\varphi(y)dy = \lambda \varphi(x)$ cannot be solved with standard calculus to yield a neat formula, we absolutely *must* rely on numerical approximations.

**3. The Problem in 4.3b: The Galerkin Projection**

Because we cannot find the true, infinite-dimensional eigenfunctions, Exercise 4.3b asks you to restrict your search to a finite-dimensional subspace $V_n$.

You make a structural guess (the *Ansatz*) that the eigenfunction can be built out of a linear combination of known, well-behaved basis functions. In this case, you are instructed to use $n=100$ 2D Legendre polynomials:

$$\varphi(x) = \sum_{j=1}^n \alpha_j b_j(x)$$

The goal of 4.3b is to project the continuous integral operator $T_c$ down onto this finite basis. By testing this equation against the basis functions, you transform the impossible infinite-dimensional calculus problem into a solvable $n \times n$ matrix equation: $M\alpha = \lambda\alpha$.

Solving for the eigenvectors $\alpha$ of this matrix $M$ gives you the numerical weights needed to construct your approximated continuous eigenfunction $\varphi(x)$.

**4. The Bigger Picture: Why Are We Doing This?**

The immediate goal of this subtask is to complete Exercise 4.3e: drawing actual visual samples from a 2D Gaussian Random Field $\mathcal{N}(0, T_c)$.

However, the ultimate goal of your entire course is solving **Bayesian Inverse Problems** (like inferring an unknown, spatially varying physical parameter inside a PDE from noisy sensor data).

* A computer cannot directly sample or run algorithms on an infinite-dimensional probability distribution.
* By finding the numerical eigenvalues $\lambda_j$ and eigenfunctions $\varphi_j$, you can construct the **Karhunen-Loève (KL) expansion**:

$$a(\omega, x) = m(x) + \sum_{j=1}^\infty \sqrt{\lambda_j} \xi_j(\omega) \varphi_j(x)$$

* Because the eigenvalues of a compact trace-class operator decay to zero, you can safely truncate this infinite sum after $n$ terms (just like the exercise suggests).

**The Master Stroke:** The KL expansion compresses an impossible, infinite-dimensional physical field into a finite vector of independent, standard 1D Gaussian variables $\xi_j \sim \mathcal{N}(0,1)$. Once your random field is parameterized by these finite variables, you can finally feed the problem into a computer to run Markov Chain Monte Carlo (MCMC) algorithms or transport maps to find the posterior distribution.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>$M\alpha$ gives us linear combination of vectors, but $\phi$ is a linear combination of functions. I do not really see how is the matrix related to the problem of finding the eigenvalue.</summary>

This is the exact conceptual hurdle that trips up almost everyone when they first transition from standard linear algebra to functional analysis. You are looking at a column of discrete numbers ($\alpha$) and wondering how it suddenly acts like a continuous wave ($\varphi(x)$).

To bridge this gap, you have to look at it as a translation between two different languages: physical geometry and coordinate maps.

### 1. The Blueprint (The Vector $\alpha$) vs. The House (The Function $\varphi$)

Think of your functional subspace $V_n$ as a physical 3D room, but with $n=100$ dimensions instead of 3.

* The **basis functions** $b_1(x), b_2(x), \dots, b_n(x)$ are the structural pillars (the x, y, z axes) of this room.
* The **function** $\varphi(x)$ is a physical point floating somewhere in that room.
* The **vector** $\alpha = [\alpha_1, \alpha_2, \dots, \alpha_n]^\top$ is just a set of GPS coordinates (a blueprint) that tells you exactly how far to walk along each structural pillar to reach that physical point.

**The vector $\alpha$ is not the function itself. It is the instruction manual for how to build the function out of the basis.**

### 2. The Operator ($T_c$) vs. The Matrix ($M$)

Now, what happens when we apply the continuous operator $T_c$?
The integral operator $T_c$ acts physically on the room. If it eats a function $f(x)$, it spits out a new, distorted function $g(x)$. It physically moves the point in the room to a new location.

But if we already know our structural pillars (the basis functions), we do not need to do the complicated infinite-dimensional calculus every time. We can just write down a matrix $M$ that tracks how those structural pillars get moved.

* $T_c$ is the physical distortion of the space.
* $M$ is the spreadsheet that updates the GPS coordinates.

Applying the continuous operator $T_c$ to the continuous function $\varphi(x)$ gives you the exact same result as multiplying the coordinate vector $\alpha$ by the matrix $M$, and then rebuilding the function from the new coordinates.

### 3. The Eigenvalue Connection

Now, let's tie it all directly to your eigenvalue problem: $M\alpha = \lambda\alpha$.

Suppose you run your numerical solver and find an eigenvector $\alpha = [2, 0, -1]^\top$ with an eigenvalue of $\lambda = 5$.

In the discrete matrix world, $M\alpha = 5\alpha$. This means the matrix transformation did not rotate or skew the vector $\alpha$; it just stretched it out to $[10, 0, -5]^\top$.

Because the vector $\alpha$ is just the blueprint for the function $\varphi(x)$, let's see what happens to the continuous function:

1. **Before the transformation:** $\varphi(x) = 2b_1(x) + 0b_2(x) - 1b_3(x)$
2. **After the transformation:** We build the new function using the scaled coordinates:

$$10b_1(x) + 0b_2(x) - 5b_3(x) = 5 \cdot \Big(2b_1(x) + 0b_2(x) - 1b_3(x)\Big)$$


3. **The Result:** The new function is exactly $5 \cdot \varphi(x)$.

Because $M$ stretching the blueprint $\alpha$ by a factor of 5 perfectly corresponds to $T_c$ stretching the entire function $\varphi(x)$ by a factor of 5, the condition $T_c \varphi = \lambda \varphi$ is completely fulfilled.

By finding the eigenvector $\alpha$ of the finite matrix $M$, you have found the exact coordinates needed to construct the continuous eigenfunction $\varphi(x)$ of the infinite-dimensional operator $T_c$.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>If I understood you correctly, we do not need to compute T_c on some function, because we know its basis functions, like in the Ansatz case of eigenfunction, but then you say that we can only directly compute the update of the alphas of the basis functions after the transformation of T_c. If my understanding of your explanation is correct, then does not it assume that the transformation performed by T_c is linear, since why would be update of the alphas governed by a matrix, which is a linear transformation?</summary>

You have just hit on the absolute most fundamental requirement of this entire mathematical framework. Your intuition is 100% correct.

**Yes, the matrix formulation assumes the transformation is linear, because the operator $T_c$ is, fundamentally, a linear operator.**

If $T_c$ were a non-linear operator—for example, if it squared your function like $T_c(f) = \int c(x,y) f^2(y) \, dy$—the entire matrix shortcut would instantly fall apart. You would not be able to track the updates with a matrix $M$.

Here is exactly why $T_c$ is linear, and how that linearity is the "magic key" that unlocks the matrix blueprint.

### 1. Why is the Operator $T_c$ Linear?

An operator is linear if it obeys two simple rules:

1. **Additivity:** $T(f + g) = T(f) + T(g)$
2. **Homogeneity (Scaling):** $T(\alpha f) = \alpha T(f)$

Look at how $T_c$ is defined. It is an integral:


$$(T_c f)(x) = \int_D c(x,y) f(y) \, dy$$

Because standard integration is inherently a linear process, the operator automatically inherits both rules:

* **Additivity:** $\int c(x,y) [f(y) + g(y)] \, dy = \int c(x,y)f(y) \, dy + \int c(x,y)g(y) \, dy$
* **Scaling:** $\int c(x,y) [\alpha f(y)] \, dy = \alpha \int c(x,y)f(y) \, dy$

Because integrals are linear, integral operators are linear.

### 2. How Linearity Unlocks the Ansatz

When we made the Ansatz that the eigenfunction is a linear combination of basis functions, we wrote:


$$\varphi(x) = \sum_{j=1}^n \alpha_j b_j(x)$$

Now, watch what happens when we apply the linear operator $T_c$ to this function. Because $T_c$ is linear, it passes right through the sum, and it passes right through the constant scalar weights ($\alpha_j$):


$$T_c(\varphi) = T_c \left( \sum_{j=1}^n \alpha_j b_j \right) = \sum_{j=1}^n \alpha_j \Big( T_c b_j \Big)$$

**This equation is the physical bridge between functional analysis and linear algebra.** Notice what just happened: the operator $T_c$ completely ignored the blueprint coordinates ($\alpha_j$) and attached itself *only* to the structural pillars (the basis functions $b_j$).

Because of linearity, to know what $T_c$ does to *any* function in the room, we do not need to evaluate $T_c$ on the whole function. We only need to evaluate $T_c$ on the basis functions. Once we know how the basis functions warp, we just multiply them by the exact same original $\alpha_j$ coordinates.

### 3. The Matrix is Just the Tracker

Because $T_c$ only needs to act on a finite number of basis functions ($n=100$), we can calculate all those interactions exactly once, store those numbers in a grid, and call it a matrix $M$.

From that point on, if you want to know what the continuous, infinite-dimensional operator $T_c$ does to the function $\varphi$, you just matrix-multiply $M \alpha$.

Your brilliant realization is exactly why mathematicians study linear algebra so intensely. If an operator in a continuous, infinite-dimensional space is linear, you can always project it down and perfectly simulate it using a standard, discrete matrix.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>So, do I understand correctly that the whole decomposition of the operator into eigenfunctions theory work only for the linear operators?</summary>

You have absolutely hit the bullseye. **Yes, the entire theory of eigen-decomposition exists exclusively for linear operators.**

If an operator is non-linear, the entire concept of an "eigenfunction" practically ceases to exist, and the ability to decompose a problem into a sum of parts completely collapses.

Here is exactly why this mathematical framework depends entirely on the assumption of linearity.

### The Collapse of the Superposition Principle

The ultimate goal of finding eigenfunctions is to take a difficult function $f(x)$, break it apart into a linear combination of simple eigenfunctions $\varphi_i$, and then apply the operator $T$:


$$f = \sum_i \alpha_i \varphi_i$$

If we want to know what the operator $T$ does to $f$, we plug the sum into $T$:


$$T(f) = T\left( \sum_i \alpha_i \varphi_i \right)$$

Here is the critical moment. If $T$ is linear, we are allowed to pull the sum and the scalars *outside* the operator. This is called the **Principle of Superposition**:


$$T(f) = \sum_i \alpha_i T(\varphi_i)$$

Because $\varphi_i$ are eigenfunctions, we can trivially replace $T(\varphi_i)$ with $\lambda_i \varphi_i$:


$$T(f) = \sum_i \alpha_i \lambda_i \varphi_i$$

**If $T$ is non-linear, that middle step is mathematically illegal.** Imagine a non-linear operator that squares the input: $T(f) = f^2$.
If you try to plug a sum into a square, you get cross-terms: $(A+B)^2 = A^2 + 2AB + B^2$.


$$T(\varphi_1 + \varphi_2) \neq T(\varphi_1) + T(\varphi_2)$$

Because the parts interact with each other (the $2AB$ cross-term), you can no longer analyze the eigenfunctions independently. The "blueprint" of independent basis vectors is destroyed.

### The Broader Consequences in Physics and Math

This boundary between linear and non-linear is the most important dividing line in all of applied mathematics and physics.

* **Why Quantum Mechanics is "Easy":** The Schrödinger equation is governed by a linear operator (the Hamiltonian). Because it is linear, quantum states can exist in superpositions (a sum of eigenfunctions), and we can perfectly predict how they evolve using linear algebra and matrices.
* **Why Fluid Dynamics is "Hard":** The Navier-Stokes equations (which govern weather, turbulence, and airplane aerodynamics) are dominated by non-linear operators. Because they are non-linear, you cannot break the wind into independent "eigen-breezes." The wind currents constantly interact with and mutate each other. There is no simple matrix representation, no Karhunen-Loève expansion, and no exact mathematical solution—which is why forecasting the weather is incredibly difficult.

By recognizing that this entire framework only works because the integral operator $T_c$ is linear, you have successfully grasped the core structural requirement of functional analysis.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**Idea.** "$T_c:V_n\to V_n$" is the $L^2$-orthogonal compression $\widetilde T_c:=P_nT_c|_{V_n}$, where $P_n$ projects onto $V_n=\operatorname{span}\{b_1,\dots,b_n\}$. We proceed in two clean stages: **first** read off the matrix of $\widetilde T_c$ in the basis $(b_j)$ — this carries no eigenvalue — and **then** impose the eigen-relation. Keeping the stages apart is exactly what prevents $\lambda$ from appearing to materialise inside the identity that *defines* the operator.
 
**Stage 1 — represent the compressed operator (no $\lambda$ yet).** Crucially $T_c\varphi$ need not lie in $V_n$. The exercise's identity defines its compression $\widetilde T_c\varphi=P_nT_c\varphi\in V_n$ as the unique element whose coordinate vector $\gamma=(\gamma_1,\dots,\gamma_n)$ satisfies

$$
\Big\langle \textstyle\sum_{j}\gamma_j b_j,\ v\Big\rangle=\langle T_c\varphi,\,v\rangle\qquad\forall v\in V_n .
$$

This is legitimate for two reasons: testing the *full* $T_c\varphi$ against $v\in V_n$ already extracts only its $V_n$-part, since $T_c\varphi-P_nT_c\varphi\perp V_n$; and an element of $V_n$ is pinned down by its inner products against the basis, because the Gram matrix below is invertible. Writing the input as $\varphi=\sum_j\alpha_j b_j$ and testing with $v=b_i$,

$$
\sum_{j}\gamma_j\,\underbrace{\langle b_j,b_i\rangle}_{=:B_{ij}}
=\sum_{j}\alpha_j\,\underbrace{\langle T_c b_j,b_i\rangle}_{=:A_{ij}},
\qquad\text{i.e.}\qquad B\gamma=A\alpha ,
$$

with the (symmetric) **mass / Gram matrix** $B$ and the (symmetric, since $T_c=T_c^*$) **stiffness matrix** $A$,

$$
B_{ij}=\langle b_i,b_j\rangle=\int_D b_i\,b_j ,
\qquad
A_{ij}=\langle T_c b_j,b_i\rangle=\int_D\!\int_D c(x,y)\,b_j(y)\,b_i(x)\,\mathrm dy\,\mathrm dx .
$$

Since $(b_j)$ is linearly independent, $B\succ0$, so $\gamma=B^{-1}A\,\alpha$. In coordinates the compressed operator is therefore the matrix

$$
M:=B^{-1}A,\qquad\text{i.e. the image }\widetilde T_c\varphi\text{ has coordinates }\gamma=M\alpha .
$$

No eigenvalue has entered: this is just the matrix of $\widetilde T_c$.
 
**Stage 2 — impose the eigen-relation (now $\lambda$ enters).** A vector $\varphi=\sum_j\alpha_j b_j$ is a (Galerkin) eigenfunction with eigenvalue $\lambda$ precisely when its image equals $\lambda\varphi$, i.e. when the image coordinates from Stage 1 are

$$
\gamma=\lambda\alpha.
$$

Combining with $\gamma=M\alpha$ gives the requested **standard** eigenproblem

$$
\boxed{\,M\alpha=\lambda\alpha,\qquad M=B^{-1}A.\,}
$$

Equivalently, before inverting $B$, the relation $\gamma=\lambda\alpha$ together with $B\gamma=A\alpha$ is the **generalised** eigenproblem $A\alpha=\lambda B\alpha$ — which is the weak form $\langle T_c\varphi,v\rangle=\lambda\langle\varphi,v\rangle$ for all $v\in V_n$ written out in the basis.
 
The $\lambda$ thus lives entirely in Stage 2, in the eigen-condition $\gamma=\lambda\alpha$; the identity defining $M$ in Stage 1 has none. The only reason a $\lambda$ can sit next to "$\sum_j\alpha_j b_j$" is that, once we name the *eigenfunction's* coordinates $\alpha$, the *image's* coordinates on a true eigenpair are $\gamma=\lambda\alpha$.
 
Finally, $M=B^{-1}A$ is similar to the symmetric $B^{-1/2}AB^{-1/2}$, hence has real eigenvalues; for $\lambda>0$ these are the approximate positive eigenvalues of $T_c$, and $\varphi=\sum_j\alpha_j b_j$ the approximate eigenfunctions.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.3c</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

Write a function to approximate the entries of $M$.

Use the following basis functions: For $j \in \mathbb N_0$ denote by

$$P_j : [-1,1] \to \mathbb R$$

the $j$-th Legendre polynomial. In Python you can use

$$\texttt{scipy.special.legendre}$$

to determine $P_j$. Let

$$\lbrace b_j(x_1,x_2) : j=1,\ldots,100\rbrace = \lbrace P_i(2x_1-1)P_j(2x_2-1) : 0 \leq i,j \leq 9\rbrace.$$

To approximate the occurring integrals over $[0,1]^2 \times [0,1]^2$, use a tensorized Gauss-Lobatto quadrature with $12$ points in each direction, see

$$\texttt{numpy.polynomial.legendre.leggauss}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.3d</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

Compute the eigenvalues/eigenvectors of $M$. Plot the eigenvalues

$$(\lambda_j)_{j=1}^{100}$$

and plot the first $4$ corresponding eigenfunctions in $L^2(D)$, i.e. for the $4$ largest eigenvalues.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.3e</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

Using the approximated Karhunen-Loève expansion, plot $4$ samples from

$$\mathcal N(0,T_c).$$

</div>
