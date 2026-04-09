---
layout: default
title: "18.102: Introduction to Functional Analysis"
date: 2025-01-01
excerpt: "Notes on MIT 18.102 by Casey Rodriguez — normed spaces, Banach spaces, bounded operators, and fundamental theorems."
tags:
  - functional-analysis
  - mathematics
# math: true
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

# 18.102: Introduction to Functional Analysis

## Introduction

In many math courses — calculus, linear algebra — the methods we learn help us solve **equations with finitely many variables**. We might want to find the minimum of a function whose inputs are in $\mathbb{R}^n$, or solve a system of linear equations. But when we encounter ODEs, PDEs, minimization, and other problems where the set of independent variables is no longer finite-dimensional, we need new tools.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Shortest Curve)</span></p>

Finding the shortest possible curve between two points amounts to specifying a **functional** — the input is a function. We need infinitely many real numbers to specify a real-valued function $f : [0, 1] \to \mathbb{R}$.

</div>

Functional analysis helps us solve problems where the vector space is no longer finite-dimensional, and this situation arises very naturally in many concrete problems.

## Normed Spaces

We use a lot of terminology from real analysis and linear algebra. Let us redefine a few key terms.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Space)</span></p>

A **vector space** $V$ over a field $\mathbb{K}$ (which we take to be either $\mathbb{R}$ or $\mathbb{C}$) is a set of vectors equipped with an addition $+ : V \times V \to V$ and scalar multiplication $\cdot : \mathbb{K} \times V \to V$, satisfying the usual axioms: commutativity, associativity, identity and inverse of addition, identity of multiplication, and distributivity.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Basic Vector Spaces)</span></p>

$\mathbb{R}^n$ and $\mathbb{C}^n$ are vector spaces, and so is $C([0, 1])$, the space of continuous functions $[0, 1] \to \mathbb{C}$. The latter is a vector space because the sum of two continuous functions is continuous, and so is a scalar multiple of a continuous function.

</div>

But $C([0, 1])$ is a completely different **size** from $\mathbb{R}^n$ or $\mathbb{C}^n$: it is infinite-dimensional.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Finite- and Infinite-Dimensional)</span></p>

A vector space $V$ is **finite-dimensional** if every linearly independent set is finite. In other words, for all sets $E \subseteq V$ such that

$$\sum_{i=1}^{N} a_i v_i = 0 \implies a_1 = a_2 = \cdots = a_N = 0 \quad \forall\, v_1, \dots, v_N \in E,$$

$E$ has finite cardinality. $V$ is **infinite-dimensional** if it is not finite-dimensional.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($C([0,1])$ is Infinite-Dimensional)</span></p>

The set $E = \lbrace f_n(x) = x^n : n \in \mathbb{Z}_{\ge 0} \rbrace$ is linearly independent but contains infinitely many elements, so $C([0, 1])$ is infinite-dimensional.

</div>

Facts like the Heine–Borel theorem for $\mathbb{R}^n$ become false in infinite-dimensional spaces, so we need to develop more machinery. In analysis, we need a notion of "how close things are." In metric spaces we use metrics; here we define a distance on our vector spaces via norms.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Norm)</span></p>

A **norm** on a vector space $V$ is a function $\lVert \cdot \rVert : V \to [0, \infty)$ satisfying:

1. (Definiteness) $\lVert v \rVert = 0$ if and only if $v = 0$.
2. (Homogeneity) $\lVert \lambda v \rVert = \lvert \lambda \rvert \lVert v \rVert$ for all $v \in V$ and $\lambda \in \mathbb{K}$.
3. (Triangle inequality) $\lVert v_1 + v_2 \rVert \le \lVert v_1 \rVert + \lVert v_2 \rVert$ for all $v_1, v_2 \in V$.

A **seminorm** is a function $\lVert \cdot \rVert : V \to [0, \infty)$ satisfying (2) and (3) but not necessarily (1). A vector space equipped with a norm is called a **normed space**.

</div>

A norm induces a metric, so we can think of our normed space as a metric space. Recall that a **metric** $d : X \times X \to [0, \infty)$ satisfies: 
* (1) $d(x, y) = 0 \iff x = y$, 
* (2) $d(x, y) = d(y, x)$,
* (3) $d(x, y) + d(y, z) \ge d(x, z)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Norm Induces a Metric)</span></p>

Let $\lVert \cdot \rVert$ be a norm on a vector space $V$. Then

$$d(v, w) = \lVert v - w \rVert$$

defines a metric on $V$, called the **metric induced by the norm**.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Property (1) of the norm implies property (1) of metrics, because $d(v, w) = \lVert v - w \rVert = 0 \iff v - w = 0 \iff v = w$.</p>
    <p>For property (2) of the metric, note that $\lVert v - w \rVert = \lVert (-1)(w - v) \rVert = \lvert -1 \rvert \cdot \lVert w - v \rVert = \lVert w - v \rVert$ by homogeneity.</p>
    <p>Property (3) of the metric is implied by property (3) of the norm because $(x - y) + (y - z) = (x - z)$. $\square$</p>
  </details>
</div>

### Examples of Norms

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\ell^p$ Norms on $\mathbb{R}^n$)</span></p>

The **Euclidean norm** on $\mathbb{R}^n$ or $\mathbb{C}^n$ is

$$\lVert x \rVert_2 = \left( \sum_{i=1}^{n} \lvert x_i \rvert^2 \right)^{1/2}.$$

We can also define $\lVert x \rVert_\infty = \max_{1 \le i \le n} \lvert x_i \rvert$ and, more generally, for $1 \le p < \infty$,

$$\lVert x \rVert_p = \left( \sum_{i=1}^{n} \lvert x_i \rvert^p \right)^{1/p}.$$

The "unit balls" $B(0, 1)$ under these norms have different shapes in $\mathbb{R}^2$: a circle for $\lVert \cdot \rVert_2$, a square (rotated 45°) for $\lVert \cdot \rVert_1$, and a square (axis-aligned) for $\lVert \cdot \rVert_\infty$. In general, a large enough $\ell^1$ ball always swallows an $\ell^\infty$ ball of any fixed size, meaning the norms are essentially equivalent — a fact we will prove later.

</div>

### Norms on Function Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($C_\infty(X)$ and the Sup Norm)</span></p>

Let $X$ be a metric space. The vector space of bounded, continuous functions is

$$C_\infty(X) = \lbrace f : X \to \mathbb{C} : f \text{ continuous and bounded} \rbrace.$$

For example, $C_\infty([0, 1]) = C([0, 1])$ because all continuous functions on $[0, 1]$ are bounded.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Sup Norm on $C_\infty(X)$)</span></p>

For any metric space $X$, we can define a norm on $C_\infty(X)$ via

$$\lVert u \rVert_\infty = \sup_{x \in X} \lvert u(x) \rvert.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Properties (1) and (2) of a norm are clear from the definitions. For property (3): if $u, v \in C_\infty(X)$, then for any $x \in X$,</p>
    $$\lvert u(x) + v(x) \rvert \le \lvert u(x) \rvert + \lvert v(x) \rvert \le \lVert u \rVert_\infty + \lVert v \rVert_\infty,$$
    <p>so $\lVert u + v \rVert_\infty = \sup_x \lvert u(x) + v(x) \rvert \le \lVert u \rVert_\infty + \lVert v \rVert_\infty$. $\square$</p>
  </details>
</div>

Convergence $u_n \to u$ in $C_\infty(X)$ means $\lVert u_n - u \rVert_\infty \to 0$, which unpacks to

$$\forall \varepsilon > 0,\; \exists N \in \mathbb{N} : \forall n \ge N,\; \forall x \in X,\; \lvert u_n(x) - u(x) \rvert < \varepsilon.$$

This is precisely the definition of **uniform convergence** on $X$.

### $\ell^p$ Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\ell^p$ Space)</span></p>

The **$\ell^p$ space** is the space of infinite sequences

$$\ell^p = \left\lbrace \lbrace a_j \rbrace_{j=1}^{\infty} : \lVert a \rVert_p < \infty \right\rbrace,$$

where the **$\ell^p$ norm** is

$$\lVert a \rVert_p = \begin{cases} \left( \sum_{j=1}^{\infty} \lvert a_j \rvert^p \right)^{1/p} & 1 \le p < \infty, \\[6pt] \sup_{1 \le j < \infty} \lvert a_j \rvert & p = \infty. \end{cases}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\ell^p$ Membership)</span></p>

The sequence $\left\lbrace \frac{1}{j} \right\rbrace_{j=1}^{\infty}$ is in $\ell^p$ for all $p > 1$ but not in $\ell^1$ (by the usual $p$-series test).

</div>

## Banach Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Banach Space)</span></p>

A normed space is a **Banach space** if it is complete with respect to the metric induced by the norm (i.e., every Cauchy sequence converges).

</div>

We know from real analysis that $\mathbb{Q}$ is not complete — one can construct a sequence of rationals that converges to an irrational. We want our Banach spaces to "fill in the holes," analogously to how $\mathbb{R}$ completes $\mathbb{Q}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Completeness of $\mathbb{R}^n$ and $\mathbb{C}^n$)</span></p>

For any $n \in \mathbb{Z}_{\ge 0}$, $\mathbb{R}^n$ and $\mathbb{C}^n$ are complete with respect to any of the $\lVert \cdot \rVert_p$ norms.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">($C_\infty(X)$ is a Banach Space)</span></p>

For any metric space $X$, the space of bounded, continuous functions on $X$ is complete, and thus $C_\infty(X)$ is a Banach space.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We want to show that every Cauchy sequence $\lbrace u_n \rbrace$ converges to some limit $u$ in $C_\infty(X)$. The strategy (which is the standard approach for proving completeness): take a Cauchy sequence, produce a candidate limit, and show (1) the candidate is in the space and (2) convergence occurs.</p>
    <p><strong>Step 1: Boundedness.</strong> Since $\lbrace u_n \rbrace$ is Cauchy, there exists $N_0$ such that $\lVert u_n - u_m \rVert_\infty < 1$ for all $n, m \ge N_0$. So for $n \ge N_0$,</p>
    $$\lVert u_n \rVert_\infty \le \lVert u_n - u_{N_0} \rVert_\infty + \lVert u_{N_0} \rVert_\infty < 1 + \lVert u_{N_0} \rVert_\infty,$$
    <p>and thus $\lVert u_n \rVert_\infty \le \lVert u_1 \rVert_\infty + \cdots + \lVert u_{N_0} \rVert_\infty + 1 \le B$ for some finite $B$.</p>
    <p><strong>Step 2: Pointwise limit.</strong> For each fixed $x \in X$, $\lvert u_n(x) - u_m(x) \rvert \le \lVert u_n - u_m \rVert_\infty$, so $\lbrace u_n(x) \rbrace$ is a Cauchy sequence in $\mathbb{C}$, which is complete. Hence we define $u(x) = \lim_{n \to \infty} u_n(x)$.</p>
    <p><strong>Step 3: $u$ is bounded.</strong> Since $\lvert u(x) \rvert = \lim_{n \to \infty} \lvert u_n(x) \rvert \le B$, we have $\sup_{x \in X} \lvert u(x) \rvert \le B$.</p>
    <p><strong>Step 4: Uniform convergence.</strong> Fix $\varepsilon > 0$. Since $\lbrace u_n \rbrace$ is Cauchy, there exists $N$ such that $\lVert u_n - u_m \rVert_\infty < \varepsilon/2$ for all $n, m \ge N$. For any $x \in X$ and $n \ge N$,</p>
    $$\lvert u_n(x) - u_m(x) \rvert \le \lVert u_n - u_m \rVert_\infty < \frac{\varepsilon}{2}.$$
    <p>Taking $m \to \infty$, $\lvert u_n(x) - u(x) \rvert \le \varepsilon/2$ for all $x \in X$ and $n \ge N$. Hence $\sup_x \lvert u_n(x) - u(x) \rvert \le \varepsilon/2 < \varepsilon$, so $\lVert u_n - u \rVert_\infty \to 0$.</p>
    <p><strong>Step 5: $u$ is continuous.</strong> Since $u_n \to u$ uniformly and each $u_n$ is continuous, the uniform limit of continuous functions is continuous. Therefore $u \in C_\infty(X)$. $\square$</p>
  </details>
</div>

The same technique can be used to show that the $\ell^p$ spaces are Banach, and also that the space $c_0 = \lbrace a \in \ell^\infty : \lim_{j \to \infty} a_j = 0 \rbrace$ is Banach.

### Summability Characterization of Banach Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Summable and Absolutely Summable Series)</span></p>

Let $\lbrace v_n \rbrace_{n=1}^{\infty}$ be a sequence of points in a normed space $V$. The series $\sum_n v_n$ is **summable** if $\left\lbrace \sum_{m=1}^{n} v_m \right\rbrace_{n=1}^{\infty}$ converges, and $\sum_n v_n$ is **absolutely summable** if $\left\lbrace \sum_{m=1}^{n} \lVert v_m \rVert \right\rbrace_{n=1}^{\infty}$ converges.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Absolutely Summable Implies Cauchy)</span></p>

If $\sum_n v_n$ is absolutely summable, then the sequence of partial sums $\left\lbrace \sum_{m=1}^{n} v_m \right\rbrace_{n=1}^{\infty}$ is Cauchy.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Banach $\iff$ Absolutely Summable Series are Summable)</span></p>

A normed vector space $V$ is a Banach space if and only if every absolutely summable series is summable.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Forward direction.</strong> Suppose $V$ is Banach. Then $V$ is complete, so any absolutely summable series is Cauchy (by the proposition above), and thus convergent (i.e., summable).</p>
    <p><strong>Reverse direction.</strong> Suppose every absolutely summable series is summable. Take any Cauchy sequence $\lbrace v_n \rbrace$. We construct a convergent subsequence by "speeding up" the Cauchy-ness. For each $k \in \mathbb{N}$, choose $N_k$ such that $\lVert v_n - v_m \rVert < 2^{-k}$ for all $n, m \ge N_k$. Define $n_k = N_1 + \cdots + N_k$, so $n_1 < n_2 < \cdots$ and $n_k \ge N_k$. Then</p>
    $$\lVert v_{n_{k+1}} - v_{n_k} \rVert < 2^{-k},$$
    <p>so the telescoping series $\sum_k (v_{n_{k+1}} - v_{n_k})$ is absolutely summable (since $\sum_k 2^{-k} = 1$). By assumption it is summable, so the partial sums</p>
    $$\sum_{k=1}^{m} (v_{n_{k+1}} - v_{n_k}) = v_{n_{m+1}} - v_{n_1}$$
    <p>converge. Thus $\lbrace v_{n_k} \rbrace$ converges. Since a Cauchy sequence with a convergent subsequence must itself converge, $V$ is Banach. $\square$</p>
  </details>
</div>

## Linear Operators

Now that we have characterized our vector spaces, we want to find the analog of **matrices** from linear algebra, which leads us to **operators** and **functionals**.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Integral Operator)</span></p>

Let $K : [0, 1] \times [0, 1] \to \mathbb{C}$ be a continuous function. For any $f \in C([0, 1])$, define

$$Tf(x) = \int_0^1 K(x, y) f(y)\, dy.$$

The map $T$ is basically the inverse of differential operators. We can check that $Tf \in C([0, 1])$, and that $T$ is linear.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Linear Operator)</span></p>

Let $V$ and $W$ be two vector spaces. A map $T : V \to W$ is **linear** if for all $\lambda_1, \lambda_2 \in \mathbb{K}$ and $v_1, v_2 \in V$,

$$T(\lambda_1 v_1 + \lambda_2 v_2) = \lambda_1 T v_1 + \lambda_2 T v_2.$$

We often use the phrase **linear operator** instead of "linear map" or "linear transformation."

</div>

In finite-dimensional vector spaces, all linear transformations are continuous. This is **not** always true when we have a map between two Banach spaces.

### Bounded Operators

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Continuity $\iff$ Boundedness for Linear Operators)</span></p>

Let $V, W$ be two normed vector spaces. A linear operator $T : V \to W$ is continuous if and only if there exists $C > 0$ such that for all $v \in V$,

$$\lVert Tv \rVert_W \le C \lVert v \rVert_V.$$

In this case we say $T$ is a **bounded** linear operator. (This does not mean the image of $T$ is bounded — it means bounded subsets of $V$ are sent to bounded subsets of $W$.)

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Bounded $\Rightarrow$ continuous.</strong> Suppose $\lVert Tv \rVert_W \le C\lVert v \rVert_V$ for all $v$. If $v_n \to v$, then by linearity,</p>
    $$\lVert Tv_n - Tv \rVert_W = \lVert T(v_n - v) \rVert_W \le C \lVert v_n - v \rVert_V \to 0,$$
    <p>so $Tv_n \to Tv$.</p>
    <p><strong>Continuous $\Rightarrow$ bounded.</strong> Suppose $T$ is continuous. The set $T^{-1}(B_W(0, 1)) = \lbrace v \in V : Tv \in B_W(0, 1) \rbrace$ is open in $V$ (by the topological characterization of continuity). Since $T(0) = 0 \in B_W(0, 1)$, there is some $r > 0$ with $B_V(0, r) \subset T^{-1}(B_W(0, 1))$. Take $C = 2/r$. For any $v \in V \setminus \lbrace 0 \rbrace$, the vector $\frac{r}{2\lVert v \rVert_V} v$ has norm $r/2 < r$, so it lies in $B_V(0, r)$, hence</p>
    $$\left\lVert T\!\left(\frac{r}{2\lVert v \rVert_V} v\right) \right\rVert_W < 1 \implies \lVert Tv \rVert_W \le \frac{2}{r} \lVert v \rVert_V. \quad\square$$
  </details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Integral Operator is Bounded)</span></p>

The linear operator $T : C([0, 1]) \to C([0, 1])$ defined by $Tf(x) = \int_0^1 K(x, y) f(y)\, dy$ is bounded. Indeed, for all $x \in [0, 1]$,

$$\lvert Tf(x) \rvert = \left\lvert \int_0^1 K(x, y) f(y)\, dy \right\rvert \le \int_0^1 \lvert K(x, y) \rvert \lvert f(y) \rvert\, dy \le \lVert K \rVert_\infty \lVert f \rVert_\infty,$$

so $\lVert Tf \rVert_\infty \le \lVert K \rVert_\infty \lVert f \rVert_\infty$, and we can take $C = \lVert K \rVert_\infty$. We often call $K$ the **kernel** of the operator.

</div>

### The Space of Bounded Operators and the Operator Norm

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Space of Bounded Linear Operators)</span></p>

Let $V$ and $W$ be two normed spaces. The set of bounded linear operators from $V$ to $W$ is denoted $\mathcal{B}(V, W)$.

</div>

$\mathcal{B}(V, W)$ is a vector space (the sum of two linear operators is linear, etc.), and we can equip it with a norm.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Operator Norm)</span></p>

The **operator norm** of $T \in \mathcal{B}(V, W)$ is

$$\lVert T \rVert = \sup_{\lVert v \rVert = 1,\, v \in V} \lVert Tv \rVert.$$

This is finite because boundedness of $T$ implies $\lVert Tv \rVert \le C\lVert v \rVert = C$ when $\lVert v \rVert = 1$, and the operator norm is the smallest such $C$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Operator Norm is a Norm)</span></p>

The operator norm is a norm, so $\mathcal{B}(V, W)$ is a normed space.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Definiteness.</strong> The zero operator has norm 0. Conversely, if $\lVert T \rVert = 0$, then $Tv = 0$ for all $\lVert v \rVert = 1$, and by rescaling, $Tv' = \lVert v' \rVert T(v'/\lVert v' \rVert) = 0$ for all $v' \ne 0$.</p>
    <p><strong>Homogeneity.</strong> $\lVert \lambda T \rVert = \sup_{\lVert v \rVert=1} \lVert \lambda Tv \rVert = \sup_{\lVert v \rVert=1} \lvert \lambda \rvert \lVert Tv \rVert = \lvert \lambda \rvert \lVert T \rVert$.</p>
    <p><strong>Triangle inequality.</strong> For $S, T \in \mathcal{B}(V, W)$ and $\lVert v \rVert = 1$, $\lVert (S + T)v \rVert = \lVert Sv + Tv \rVert \le \lVert Sv \rVert + \lVert Tv \rVert \le \lVert S \rVert + \lVert T \rVert$. Taking the supremum gives $\lVert S + T \rVert \le \lVert S \rVert + \lVert T \rVert$. $\square$</p>
  </details>
</div>

In general, the operator norm gives a bound: for all $v \in V$,

$$\left\lVert T\!\left(\frac{v}{\lVert v \rVert}\right) \right\rVert \le \lVert T \rVert \implies \lVert Tv \rVert \le \lVert T \rVert \lVert v \rVert.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">($\mathcal{B}(V, W)$ is Banach when $W$ is Banach)</span></p>

If $V$ is a normed vector space and $W$ is a Banach space, then $\mathcal{B}(V, W)$ is a Banach space.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We use the summability characterization (Theorem above). Suppose $\lbrace T_n \rbrace$ is a sequence in $\mathcal{B}(V, W)$ with $C = \sum_n \lVert T_n \rVert < \infty$. We need to show $\sum_n T_n$ is summable.</p>
    <p><strong>Candidate.</strong> For any $v \in V$ and $m \in \mathbb{N}$,</p>
    $$\sum_{n=1}^{m} \lVert T_n v \rVert \le \sum_{n=1}^{m} \lVert T_n \rVert \lVert v \rVert \le C \lVert v \rVert.$$
    <p>Since $T_n v \in W$ and $W$ is Banach, the series $\sum_n T_n v$ is absolutely summable and thus summable. Define $Tv = \lim_{m \to \infty} \sum_{n=1}^{m} T_n v$.</p>
    <p><strong>$T$ is linear.</strong> $T(\lambda_1 v_1 + \lambda_2 v_2) = \lim_m \sum_{n=1}^{m} T_n(\lambda_1 v_1 + \lambda_2 v_2) = \lambda_1 Tv_1 + \lambda_2 Tv_2$.</p>
    <p><strong>$T$ is bounded.</strong> $\lVert Tv \rVert = \lim_m \lVert \sum_{n=1}^{m} T_n v \rVert \le \lim_m \sum_{n=1}^{m} \lVert T_n \rVert \lVert v \rVert = C\lVert v \rVert$.</p>
    <p><strong>Convergence in operator norm.</strong> For $\lVert v \rVert = 1$,</p>
    $$\left\lVert Tv - \sum_{n=1}^{m} T_n v \right\rVert = \left\lVert \lim_{m' \to \infty} \sum_{n=m+1}^{m'} T_n v \right\rVert \le \sum_{n=m+1}^{\infty} \lVert T_n \rVert,$$
    <p>so $\lVert T - \sum_{n=1}^{m} T_n \rVert \le \sum_{n=m+1}^{\infty} \lVert T_n \rVert \to 0$ (tail of a convergent series). $\square$</p>
  </details>
</div>

### Dual Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dual Space and Functional)</span></p>

Let $V$ be a normed space (over $\mathbb{K}$). The **dual space** $V' = \mathcal{B}(V, \mathbb{K})$ is the space of bounded linear maps from $V$ to the scalar field. Since $\mathbb{K} = \mathbb{R}, \mathbb{C}$ are both complete, $V'$ is a Banach space by the theorem above. An element of the dual space $\mathcal{B}(V, \mathbb{K})$ is called a **functional**.

</div>

We can identify the dual space for all $\ell^p$ spaces: it turns out that

$$(\ell^p)' = \ell^{p'},$$

where $p, p'$ satisfy $\frac{1}{p} + \frac{1}{p'} = 1$. So the dual of $\ell^1$ is $\ell^\infty$, and the dual of $\ell^2$ is itself (this is the only $\ell^p$ space for which this is true). However, the dual of $\ell^\infty$ is **not** $\ell^1$.

## Subspaces and Quotients

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subspace)</span></p>

Let $V$ be a vector space. A subset $W \subseteq V$ is a **subspace** of $V$ if for all $w_1, w_2 \in W$ and $\lambda_1, \lambda_2 \in \mathbb{K}$, we have $\lambda_1 w_1 + \lambda_2 w_2 \in W$ (closure under linear combinations).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Closed Subspace $\iff$ Banach Subspace)</span></p>

A subspace $W$ of a Banach space $V$ is Banach (with norm inherited from $V$) if and only if $W$ is a closed subset of $V$ (with respect to the metric induced by the norm).

</div>

<div class="accordion">
  <details>
    <summary>proof sketch</summary>
    <p><strong>$W$ Banach $\Rightarrow$ $W$ closed.</strong> Every sequence in $W$ that converges (to something in $V$) must be Cauchy. Since $W$ is Banach, the sequence converges in $W$. By uniqueness of limits, the limit is in $W$.</p>
    <p><strong>$W$ closed $\Rightarrow$ $W$ Banach.</strong> Any Cauchy sequence in $W$ is also Cauchy in $V$, so it has a limit in $V$. Closedness ensures the limit is in $W$. $\square$</p>
  </details>
</div>

### Quotient Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quotient Space)</span></p>

Let $W \subset V$ be a subspace of $V$. Define the equivalence relation $v \sim v' \iff v - v' \in W$, and let $[v]$ be the equivalence class of $v$. The **quotient space** $V / W$ is the set of all equivalence classes $\lbrace [v] : v \in V \rbrace$.

We typically denote $[v]$ as $v + W$ (coset notation). Addition and scalar multiplication are defined by

$$(v_1 + W) + (v_2 + W) = (v_1 + v_2) + W, \qquad \lambda(v + W) = \lambda v + W.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Seminorm and Quotient)</span></p>

Consider the seminorm which assigns $\sup \lvert f' \rvert$ to a function $f$. This satisfies homogeneity and the triangle inequality, but it is not a norm because the derivative of any constant function is 0. The constant functions form a subspace, and we can "mod out" by that subspace.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Seminorm Quotient)</span></p>

Let $\lVert \cdot \rVert$ be a seminorm on a vector space $V$. If we define $E = \lbrace v \in V : \lVert v \rVert = 0 \rbrace$, then $E$ is a subspace of $V$, and the function on $V / E$ defined by

$$\lVert v + E \rVert_{V/E} = \lVert v \rVert$$

is a well-defined **norm**.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>$E$ is a subspace.</strong> By homogeneity and the triangle inequality, $\lVert \lambda_1 v_1 + \lambda_2 v_2 \rVert \le \lvert \lambda_1 \rvert \lVert v_1 \rVert + \lvert \lambda_2 \rvert \lVert v_2 \rVert = 0$ for $v_1, v_2 \in E$, so $\lambda_1 v_1 + \lambda_2 v_2 \in E$.</p>
    <p><strong>Well-definedness.</strong> If $v + E = v' + E$, then $v = v' + e$ for some $e \in E$. By the triangle inequality, $\lVert v \rVert = \lVert v' + e \rVert \le \lVert v' \rVert + \lVert e \rVert = \lVert v' \rVert$. Swapping $v$ and $v'$ gives $\lVert v' \rVert \le \lVert v \rVert$, so $\lVert v \rVert = \lVert v' \rVert$.</p>
    <p><strong>Norm properties.</strong> Homogeneity and the triangle inequality are inherited from the seminorm. Definiteness holds because $\lVert v + E \rVert_{V/E} = 0 \implies \lVert v \rVert = 0 \implies v \in E \implies v + E = 0 + E$. $\square$</p>
  </details>
</div>

## The Baire Category Theorem and Uniform Boundedness

We now turn to some fundamental named theorems.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Baire Category Theorem)</span></p>

Let $M$ be a complete metric space, and let $\lbrace C_n \rbrace_n$ be a collection of closed subsets of $M$ such that $M = \bigcup_{n \in \mathbb{N}} C_n$. Then at least one of the $C_n$ contains an open ball $B(x, r) = \lbrace y \in M : d(x, y) < r \rbrace$. In other words, at least one $C_n$ has an interior point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Despite the name, this theorem has nothing to do with category theory. A powerful consequence: it can be used to prove that there exists a continuous function which is **nowhere differentiable**. When applying this theorem, the $C_n$ need not be closed — the result then says that one of their closures must contain an open ball. Equivalently, we cannot have all $C_n$ be **nowhere dense**.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Suppose for contradiction that all $C_n$ are nowhere dense (contain no open ball). Since $\bigcup_n C_n = M$ and $M$ contains at least one open ball, $C_1$ cannot contain all of $M$. So there exists $p_1 \in M \setminus C_1$. Since $C_1$ is closed, $M \setminus C_1$ is open, so there exists $\varepsilon_1 > 0$ with $B(p_1, \varepsilon_1) \cap C_1 = \varnothing$.</p>
    <p>Since $B(p_1, \varepsilon_1/3)$ is not contained in $C_2$ (by assumption), there exists $p_2 \in B(p_1, \varepsilon_1/3)$ with $p_2 \notin C_2$, and we can find $\varepsilon_2 < \varepsilon_1/3$ with $B(p_2, \varepsilon_2) \cap C_2 = \varnothing$.</p>
    <p>Continuing inductively, we construct $p_k$ and $\varepsilon_k$ with $\varepsilon_k < \varepsilon_1 / 3^{k-1}$, $p_j \in B(p_{j-1}, \varepsilon_{j-1}/3)$, and $B(p_j, \varepsilon_j) \cap C_j = \varnothing$.</p>
    <p>The sequence $\lbrace p_k \rbrace$ is Cauchy: for all $k, \ell \in \mathbb{N}$,</p>
    $$d(p_k, p_{k+\ell}) < \frac{\varepsilon_k}{3} + \frac{\varepsilon_{k+1}}{3} + \cdots + \frac{\varepsilon_{k+\ell-1}}{3} < \frac{\varepsilon_1}{3^k} + \cdots + \frac{\varepsilon_1}{3^{k+\ell}} < \varepsilon_1 \sum_{m=k}^{\infty} \frac{1}{3^m} = \frac{\varepsilon_1}{2} \cdot 3^{-k+1}.$$
    <p>Since $M$ is complete, $p_k \to p$ for some $p \in M$. Moreover, $d(p_{k+1}, p) \le \varepsilon_{k+1}/2 < \varepsilon_k/6$, and $d(p_k, p) \le \varepsilon_k/3 + \varepsilon_k/6 < \varepsilon_k$. So $p \in B(p_k, \varepsilon_k)$ for each $k$, and hence $p \notin C_k$ for any $k$. But then $p \notin \bigcup_k C_k = M$, which is a contradiction. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Uniform Boundedness Principle)</span></p>

Let $B$ be a Banach space, and let $\lbrace T_n \rbrace$ be a sequence in $\mathcal{B}(B, V)$ (linear operators from $B$ into some normed space $V$). If for all $b \in B$ we have $\sup_n \lVert T_n b \rVert < \infty$ (pointwise boundedness), then $\sup_n \lVert T_n \rVert < \infty$ (the operator norms are bounded).

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Define $C_k = \lbrace b \in B : \lVert b \rVert \le 1,\; \sup_n \lVert T_n b \rVert \le k \rbrace$. Each $C_k$ is closed: if $b_m \to b$ with $b_m \in C_k$, then $\lVert b \rVert = \lim \lVert b_m \rVert \le 1$, and for each fixed $m$, $\lVert T_m b \rVert = \lim_{n \to \infty} \lVert T_m b_n \rVert \le k$.</p>
    <p>Since every $b$ with $\lVert b \rVert \le 1$ has $\sup_n \lVert T_n b \rVert \le k$ for some $k$ (by pointwise boundedness), we have $\lbrace b : \lVert b \rVert \le 1 \rbrace = \bigcup_k C_k$. The closed unit ball is a complete metric space (closed subset of a Banach space), so by Baire's theorem, some $C_k$ contains an open ball $B(b_0, \delta_0)$.</p>
    <p>For any $b \in B$ with $\lVert b \rVert < \delta_0$, both $b_0$ and $b_0 + b$ lie in $B(b_0, \delta_0) \subset C_k$, so</p>
    $$\sup_n \lVert T_n b \rVert = \sup_n \lVert -T_n b_0 + T_n(b_0 + b) \rVert \le \sup_n \lVert T_n b_0 \rVert + \sup_n \lVert T_n(b_0 + b) \rVert \le k + k = 2k.$$
    <p>Rescaling: for any $b \in B$ with $\lVert b \rVert = 1$, the vector $(\delta_0/2) b$ has norm $\delta_0/2 < \delta_0$, so $\sup_n \lVert T_n(\delta_0 b/2) \rVert \le 2k$, giving $\sup_n \lVert T_n b \rVert \le 4k/\delta_0$. Therefore $\sup_n \lVert T_n \rVert \le 4k/\delta_0 < \infty$. $\square$</p>
  </details>
</div>

## The Open Mapping Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Open Mapping Theorem)</span></p>

Let $B_1, B_2$ be two Banach spaces, and let $T \in \mathcal{B}(B_1, B_2)$ be a surjective linear operator. Then $T$ is an **open map**, meaning that for all open subsets $U \subset B_1$, $T(U)$ is open in $B_2$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We first prove a specialized result: the image of the open unit ball $B_1(0, 1) = \lbrace b \in B_1 : \lVert b \rVert < 1 \rbrace$ contains an open ball in $B_2$ centered at 0.</p>
    <p><strong>Step 1: The closure $\overline{T(B(0, 1))}$ contains an open ball.</strong> Since $T$ is surjective, $B_2 = \bigcup_{n \in \mathbb{N}} \overline{T(B(0, n))}$. By Baire's theorem, some $\overline{T(B(0, n_0))}$ contains an open ball. By linearity, this is the same as $n_0 \overline{T(B(0, 1))}$, so $\overline{T(B(0, 1))}$ itself contains an open ball. Restated, there exist $v_0 \in B_2$ and $r > 0$ such that $B(v_0, 4r) \subset \overline{T(B(0, 1))}$.</p>
    <p><strong>Step 2: Show $B(0, r) \subset \overline{T(B(0, 1))}$.</strong> Pick $v_1 = Tu_1 \in T(B(0, 1))$ with $\lVert v_0 - v_1 \rVert < 2r$. Then $B(v_1, 2r) \subset B(v_0, 4r) \subset \overline{T(B(0, 1))}$. For any $\lVert v \rVert < r$, the element $\frac{1}{2}(2v + v_1)$ is in $\frac{1}{2}B(v_1, 2r) = \frac{1}{2}\overline{T(B(0, 1))}$. By linearity this equals $-T(u_1/2) + \overline{T(B(0, 1/2))}$, and since $u_1$ has norm less than 1, this set is contained in $\overline{T(B(0, 1))}$.</p>
    <p><strong>Step 3: Show $B(0, r) \subset T(B(0, 1))$.</strong> Take any $v \in B(0, r/2)$. By the closure property (with $n = 1$), there exists $b_1 \in B(0, 1/2)$ in $B_1$ with $\lVert v - Tb_1 \rVert < r/4$. Taking $n = 2$, there exists $b_2 \in B(0, 1/4)$ with $\lVert v - Tb_1 - Tb_2 \rVert < r/8$. Iterating, we get $\lbrace b_k \rbrace$ with $\lVert b_k \rVert < 2^{-k}$ and $\lVert v - \sum_{k=1}^{n} Tb_k \rVert < 2^{-n-1}r$. The series $\sum b_k$ is absolutely summable, so $b = \sum_{k=1}^{\infty} b_k$ converges in $B_1$ with $\lVert b \rVert < \sum 2^{-k} = 1$. Since $T$ is continuous, $Tb = \sum Tb_k = v$. So $v \in T(B(0, 1))$.</p>
    <p><strong>Step 4: General open sets.</strong> If $U \subset B_1$ is open and $b_2 = Tb_1 \in T(U)$, then by openness of $U$ there exists $\varepsilon > 0$ with $B(b_1, \varepsilon) \subset U$. By the work above, there exists $\delta > 0$ with $B(0, \delta) \subset T(B(0, 1))$. Then $B(b_2, \varepsilon\delta) = b_2 + \varepsilon B(0, \delta) \subset T(b_1 + B(0, \varepsilon)) = T(B(b_1, \varepsilon)) \subset T(U)$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Bounded Inverse Theorem)</span></p>

If $B_1, B_2$ are two Banach spaces and $T \in \mathcal{B}(B_1, B_2)$ is a bijective map, then $T^{-1} \in \mathcal{B}(B_2, B_1)$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>$T^{-1}$ is continuous if and only if for all open $U \subset B_1$, the inverse image of $U$ by $T^{-1}$ (which is $T(U)$) is open. This holds by the Open Mapping Theorem. $\square$</p>
  </details>
</div>

## The Closed Graph Theorem

From the Open Mapping Theorem, we get a topological criterion for continuity of a linear operator.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Product of Banach Spaces)</span></p>

If $B_1, B_2$ are Banach spaces, then $B_1 \times B_2$ with norm $\lVert (b_1, b_2) \rVert = \lVert b_1 \rVert + \lVert b_2 \rVert$ is a Banach space.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Closed Graph Theorem)</span></p>

Let $B_1, B_2$ be two Banach spaces, and let $T : B_1 \to B_2$ be a (not necessarily bounded) linear operator. Then $T \in \mathcal{B}(B_1, B_2)$ if and only if the **graph** of $T$, defined as

$$\Gamma(T) = \lbrace (u, Tu) : u \in B_1 \rbrace,$$

is closed in $B_1 \times B_2$.

</div>

This can sometimes be easier to check than boundedness. Proving that the graph is closed means: given a sequence $u_n \to u$ **and** $Tu_n \to v$, we must show that $v = Tu$ — without explicitly constructing the convergent sequence.

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Forward direction ($T$ bounded $\Rightarrow$ $\Gamma(T)$ closed).</strong> If $(u_n, Tu_n)$ is a sequence in $\Gamma(T)$ with $u_n \to u$ and $Tu_n \to v$, then by continuity $v = \lim Tu_n = T(\lim u_n) = Tu$, so $(u, v) \in \Gamma(T)$.</p>
    <p><strong>Reverse direction ($\Gamma(T)$ closed $\Rightarrow$ $T$ bounded).</strong> Since $\Gamma(T)$ is a closed subspace of the Banach space $B_1 \times B_2$, it is itself a Banach space. Define projection maps $\pi_1 : \Gamma(T) \to B_1$ and $\pi_2 : \Gamma(T) \to B_2$ by $\pi_1(u, Tu) = u$ and $\pi_2(u, Tu) = Tu$. Both are bounded linear operators (since $\lVert \pi_i(u, Tu) \rVert \le \lVert (u, Tu) \rVert$). Furthermore, $\pi_1$ is bijective (each $u$ appears exactly once in the graph), so by the Bounded Inverse Theorem, $\pi_1^{-1} : B_1 \to \Gamma(T)$ is bounded. Now $T = \pi_2 \circ \pi_1^{-1}$ is the composition of two bounded operators, hence bounded. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The Open Mapping Theorem implies the Closed Graph Theorem, and we can also show the converse — the two are logically equivalent.

</div>

## The Hahn-Banach Theorem

Each of the results so far has been trying to answer a question, and the **Hahn-Banach Theorem** asks whether the dual space of a general nontrivial normed space is trivial. We want to know whether there are any normed spaces whose space of functionals $\mathcal{B}(V, \mathbb{K})$ contains only the zero function.

### Zorn's Lemma and Hamel Bases

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial Order)</span></p>

A **partial order** on a set $E$ is a relation $\preceq$ on $E$ with the following properties:

* Reflexivity: $e \preceq e$ for all $e \in E$.
* Antisymmetry: if $e \preceq f$ and $f \preceq e$, then $e = f$.
* Transitivity: if $e \preceq f$ and $f \preceq g$, then $e \preceq g$.

An **upper bound** of a set $D \subset E$ is an element $e \in E$ with $d \preceq e$ for all $d \in D$. A **maximal element** of $E$ is an element $e$ such that $e \preceq f \implies e = f$.

</div>

Notably, in a partial ordering a maximal element does not need to sit "on top" of everything — there can be incomparable elements "to the side."

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Partial Order on Powerset)</span></p>

If $S$ is a set, we can define a partial order on the powerset of $S$ by $E \preceq F$ if $E$ is a subset of $F$. Not all sets can be compared.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Chain)</span></p>

Let $(E, \preceq)$ be a partially ordered set. A set $C \subset E$ is a **chain** if for all $e, f \in C$, we have either $e \preceq f$ or $f \preceq e$ (i.e., all elements in a chain are comparable).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Zorn's Lemma)</span></p>

If every chain in a nonempty partially ordered set $E$ has an upper bound, then $E$ contains a maximal element.

</div>

We take this as an **axiom of set theory**; it can also be used to prove the **Axiom of Choice**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hamel Basis)</span></p>

Let $V$ be a vector space. A **Hamel basis** $H \subset V$ is a linearly independent set such that every element of $V$ is a finite linear combination of elements of $H$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Existence of Hamel Bases)</span></p>

If $V$ is a vector space, then it has a Hamel basis.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let $E$ be the set of linearly independent subsets of $V$, ordered by inclusion $\preceq$. If $C$ is a chain in $E$, define $c = \bigcup_{e \in C} e$. We claim $c$ is linearly independent: given $v_1, \dots, v_n \in c$, pick $e_j \in C$ with $v_j \in e_j$. Since $C$ is a chain, there is some $e_J$ containing all the $e_j$, so $v_1, \dots, v_n \in e_J$, which is linearly independent. Thus $c$ is an upper bound for $C$.</p>
    <p>By Zorn's lemma, $E$ has a maximal element $H$. If $H$ does not span $V$, there exists $v \in V$ not in the span of $H$, so $H \cup \lbrace v \rbrace$ is linearly independent, contradicting maximality. Thus $H$ spans $V$ and is a Hamel basis. $\square$</p>
  </details>
</div>

### Statement and Proof of Hahn-Banach

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hahn-Banach)</span></p>

Let $V$ be a normed space, and let $M \subset V$ be a subspace. If $u : M \to \mathbb{C}$ is a linear map such that $\lvert u(t) \rvert \le C \lVert t \rVert$ for all $t \in M$ (i.e., $u$ is a bounded linear functional on $M$), then there exists a **continuous extension** $U : V \to \mathbb{C}$ (an element of $V' = \mathcal{B}(V, \mathbb{C})$) such that $U\vert_M = u$ and $\lVert U(t) \rVert \le C \lVert t \rVert$ for all $t \in V$ (with the same constant $C$).

</div>

The key intermediate step is extending by one dimension at a time:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(One-Dimensional Extension)</span></p>

Let $V$ be a normed space, $M \subset V$ a subspace, and $u : M \to \mathbb{C}$ linear with $\lvert u(t) \rvert \le C \lVert t \rVert$ for all $t \in M$. If $x \notin M$, then there exists $u' : M' \to \mathbb{C}$ linear on $M' = M + \mathbb{C}x = \lbrace t + ax : t \in M, a \in \mathbb{C} \rbrace$, with $u'\vert_M = u$ and $\lvert u'(t') \rvert \le C \lVert t' \rVert$ for all $t' \in M'$.

</div>

<div class="accordion">
  <details>
    <summary>proof of Hahn-Banach (assuming the lemma)</summary>
    <p>Let $E$ be the set of all continuous extensions of $u$:</p>
    $$E = \lbrace (v, N) : N \text{ subspace of } V,\; M \subset N,\; v \text{ is a continuous extension of } u \text{ to } N \rbrace,$$
    <p>with partial order $(v_1, N_1) \preceq (v_2, N_2)$ if $N_1 \subset N_2$ and $v_2\vert_{N_1} = v_1$. This is nonempty (it contains $(u, M)$).</p>
    <p>For any chain $C = \lbrace (v_i, N_i) : i \in I \rbrace$, define $N = \bigcup_{i \in I} N_i$. This is a subspace, and we define $v : N \to \mathbb{C}$ by $v(t) = v_i(t)$ for any $i$ with $t \in N_i$ (well-defined because $C$ is a chain). Then $(v, N)$ is an upper bound. By Zorn's lemma, $E$ has a maximal element $(U, N)$.</p>
    <p>If $N \ne V$, there exists $x \in V \setminus N$, and the one-dimensional extension lemma gives a continuous extension of $U$ to $N + \mathbb{C}x$, contradicting maximality. So $N = V$. $\square$</p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>proof of the one-dimensional extension lemma</summary>
    <p>The representation $t' = t + ax$ for $t \in M$, $a \in \mathbb{C}$ is unique (if $t + ax = \tilde{t} + \tilde{a}x$, then $(a - \tilde{a})x = \tilde{t} - t \in M$, which forces $a = \tilde{a}$ since $x \notin M$). Define $u'(t + ax) = u(t) + a\lambda$ for some $\lambda \in \mathbb{C}$ to be chosen. This is clearly linear.</p>
    <p>If $C = 0$, we use $\lambda = 0$. Otherwise, assume $C = 1$ (by scaling). We need $\lvert u(t) + a\lambda \rvert \le \lVert t + ax \rVert$ for all $t \in M$, $a \in \mathbb{C}$. For $a \ne 0$, dividing by $\lvert a \rvert$ gives $\lvert u(t/(-a)) - \lambda \rvert \le \lVert t/(-a) - x \rVert$.</p>
    <p><strong>Choosing $\lambda$ (real part).</strong> Let $w(t) = \frac{u(t) + \overline{u(t)}}{2}$ be the real part. Since $w$ is real-valued and $\lvert w(t) \rvert \le \lVert t \rVert$, we get $w(t_1) - w(t_2) = w(t_1 - t_2) \le \lVert t_1 - t_2 \rVert \le \lVert t_1 - x \rVert + \lVert t_2 - x \rVert$. Thus</p>
    $$\sup_{t \in M} w(t) - \lVert t - x \rVert \le \inf_{t \in M} w(t) + \lVert t - x \rVert.$$
    <p>Choose $\alpha \in \mathbb{R}$ between these bounds. Then $\lvert w(t) - \alpha \rvert \le \lVert t - x \rVert$ for all $t \in M$. Repeat the argument with $ix$ instead of $x$ to choose the imaginary part. This defines $u'$ on $M + \mathbb{C}x$ with the desired bound. $\square$</p>
  </details>
</div>

### Consequences of Hahn-Banach

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Existence of Norm-Achieving Functionals)</span></p>

Let $V$ be a normed space. For all $v \in V \setminus \lbrace 0 \rbrace$, there exists $f \in V'$ with $\lVert f \rVert = 1$ and $f(v) = \lVert v \rVert$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Define $u : \mathbb{C}v \to \mathbb{C}$ by $u(\lambda v) = \lambda \lVert v \rVert$. Then $\lvert u(t) \rvert \le \lVert t \rVert$ for all $t \in \mathbb{C}v$, and $u(v) = \lVert v \rVert$. By Hahn-Banach, extend $u$ to $f \in V'$ with $\lVert f(t) \rVert \le \lVert t \rVert$ for all $t \in V$. So $\lVert f \rVert \le 1$, and since $f(v) = \lVert v \rVert$, applying $f$ to $v / \lVert v \rVert$ gives $f(v/\lVert v \rVert) = 1$, so $\lVert f \rVert = 1$. $\square$</p>
  </details>
</div>

### Double Dual and Reflexivity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Double Dual)</span></p>

The **double dual** of a normed space $V$, denoted $V''$, is the dual of $V'$. In other words, $V''$ is the set of bounded linear functionals on the set of bounded linear functionals on $V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Canonical Embedding into $V''$)</span></p>

Fix $v \in V$ and define $T_v : V' \to \mathbb{C}$ by $T_v(v') = v'(v)$ for all $v' \in V'$. Then $T_v$ is an element of the double dual: it is linear in $v'$ (since $v'$ is a functional applied to a fixed $v$), and bounded since $\lvert T_v(v') \rvert = \lvert v'(v) \rvert \le \lVert v' \rVert \cdot \lVert v \rVert$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Isometric Operator)</span></p>

A bounded linear operator $T \in \mathcal{B}(V, W)$ is **isometric** if for all $v \in V$, $\lVert Tv \rVert = \lVert v \rVert$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Canonical Embedding is Isometric)</span></p>

Let $v \in V$, and define $T_v \in V''$ by $T_v(v') = v'(v)$. Then the map $T : V \to V''$ sending $v \mapsto T_v$ is isometric.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We already showed $\lVert T_v \rVert \le \lVert v \rVert$ (so $\lVert T \rVert \le 1$). It suffices to show equality. For $v = 0$ this is clear. For $v \ne 0$, by the existence of norm-achieving functionals, there exists $f \in V'$ with $\lVert f \rVert = 1$ and $f(v) = \lVert v \rVert$. Then</p>
    $$\lVert v \rVert = f(v) = \lvert f(v) \rvert = \lvert T_v(f) \rvert \le \lVert T_v \rVert \cdot \lVert f \rVert = \lVert T_v \rVert,$$
    <p>so $\lVert v \rVert \le \lVert T_v \rVert$, giving $\lVert T_v \rVert = \lVert v \rVert$. $\square$</p>
  </details>
</div>

Isometric operators are one-to-one (the only vector sent to zero is the zero vector). It is natural to ask whether the canonical embedding is also onto:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reflexive Space)</span></p>

A Banach space $V$ is **reflexive** if $V = V''$, in the sense that the map $v \mapsto T_v$ is onto.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Reflexivity of $\ell^p$ Spaces)</span></p>

For all $1 < p < \infty$, $\ell^p$ is reflexive (its dual is $\ell^q$ with $1/p + 1/q = 1$, whose dual is $\ell^p$ again). But $\ell^1$ is not reflexive (its dual $\ell^\infty$ has a much larger dual). The space $c_0$ of sequences converging to 0 is also not reflexive — $(c_0)' = \ell^1$, whose dual is $\ell^\infty \ne c_0$.

</div>

## Lebesgue Measure and Integration

With the general discussion of Banach spaces concluded, we now move to **Lebesgue measure and integration**. We have been talking about $\ell^p$ spaces on sequences; now we want to define $L^p$ spaces on functions in a similar way. Riemann integration is not sufficient — Lebesgue integration has better convergence theorems and is more widely useful.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name"></span></p>

Our goal is to make a new definition of integration that is more general than Riemann integration: it will still be a method of calculating area under a curve, but built up in a way that allows for more powerful formalism. The approach: start with **indicator functions** $1_E$ (equal to 1 on $E$ and 0 otherwise), define the integral of $1_E$ to be the **measure** $m(E)$ of $E$, and build up from there.

</div>

### Desired Properties of a Measure

We want a measure $m$ on subsets of $\mathbb{R}$ with these properties:

1. Defined on all subsets: $m : \mathcal{P}(\mathbb{R}) \to [0, \infty]$.
2. For any interval $I$, $m(I) = \ell(I)$ (the length of $I$).
3. Countable additivity: if $\lbrace E_n \rbrace$ are pairwise disjoint, $m(\bigcup_n E_n) = \sum_n m(E_n)$.
4. Translation invariance: $m(x + E) = m(E)$ for all $x \in \mathbb{R}$.

Unfortunately, **no function** $m : \mathcal{P}(\mathbb{R}) \to [0, \infty]$ satisfies all four simultaneously (this is shown by the **Vitali construction**). The resolution: drop property (1) and define $m$ only on a well-behaved collection of **Lebesgue measurable sets**.

### Outer Measure

The strategy (from Caratheodory): first define an **outer measure** $m^*$ on all subsets, satisfying (2), (4), and "almost (3)," then restrict to measurable sets.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Outer Measure)</span></p>

For any interval $I \subset \mathbb{R}$, let $\ell(I)$ denote its length. For any subset $A \subset \mathbb{R}$, the **outer measure** is

$$m^*(A) = \inf \left\lbrace \sum_n \ell(I_n) : \lbrace I_n \rbrace \text{ countable collection of open intervals with } A \subset \bigcup_n I_n \right\rbrace.$$

We always have $m^*(A) \ge 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Outer Measure of a Point)</span></p>

$m^*(\lbrace 0 \rbrace) = 0$: cover $\lbrace 0 \rbrace$ with $(-\varepsilon/2, \varepsilon/2)$ of length $\varepsilon$ for any $\varepsilon > 0$, and take $\varepsilon \to 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Countable Sets Have Measure Zero)</span></p>

If $A \subset \mathbb{R}$ is countable, then $m^*(A) = 0$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Enumerate $A = \lbrace a_n : n \in \mathbb{N} \rbrace$. For each $n$, let $I_n = (a_n - \varepsilon/2^{n+1}, a_n + \varepsilon/2^{n+1})$, which has length $\varepsilon/2^n$. Then $A \subset \bigcup_n I_n$, so $m^*(A) \le \sum_n \varepsilon/2^n = \varepsilon$. Taking $\varepsilon \to 0$ gives $m^*(A) = 0$. $\square$</p>
  </details>
</div>

In particular, $m^*(\mathbb{Q}) = 0$, even though $\mathbb{Q}$ is dense in $\mathbb{R}$.

### Properties of Outer Measure

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Monotonicity of Outer Measure)</span></p>

If $A \subset B$, then $m^\ast(A) \le m^\ast(B)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Countable Subadditivity)</span></p>

Let $\lbrace A_n \rbrace$ be a countable collection of subsets of $\mathbb{R}$, not necessarily disjoint. Then

$$m^*\!\left(\bigcup_n A_n\right) \le \sum_n m^*(A_n).$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We may assume all $m^*(A_n)$ are finite and $\sum_n m^*(A_n) < \infty$. Fix $\varepsilon > 0$. For each $n$, choose a covering $\lbrace I_{nk} \rbrace_{k \in \mathbb{N}}$ of $A_n$ with $\sum_k \ell(I_{nk}) < m^*(A_n) + \varepsilon/2^n$. Then $\lbrace I_{nk} \rbrace_{n,k}$ covers $\bigcup_n A_n$, so</p>
    $$m^*\!\left(\bigcup_n A_n\right) \le \sum_{n,k} \ell(I_{nk}) = \sum_n \sum_k \ell(I_{nk}) < \sum_n \left(m^*(A_n) + \frac{\varepsilon}{2^n}\right) = \sum_n m^*(A_n) + \varepsilon.$$
    <p>Taking $\varepsilon \to 0$ gives the result. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Outer Measure of Intervals)</span></p>

If $I$ is an interval of $\mathbb{R}$, then $m^*(I) = \ell(I)$.

</div>

<div class="accordion">
  <details>
    <summary>proof sketch</summary>
    <p><strong>$m^*(I) \le \ell(I)$:</strong> For a closed bounded interval $[a, b]$, cover it by $(a - \varepsilon, b + \varepsilon)$, giving $m^*(I) \le \ell(I) + 2\varepsilon$ for any $\varepsilon > 0$.</p>
    <p><strong>$\ell(I) \le m^*(I)$:</strong> Suppose $\lbrace I_n \rbrace$ is a collection of open intervals covering $[a, b]$. By Heine-Borel, a finite subcollection $\lbrace J_1, \dots, J_N \rbrace$ suffices. Rearrange so that $J_i$ and $J_{i+1}$ always overlap. Then $\sum \ell(I_n) \ge \sum_{k=1}^{K} \ell(J_k) \ge b - a = \ell(I)$ by a telescoping argument on the overlapping intervals.</p>
    <p>Other interval types follow by approximation: $[a + \varepsilon, b - \varepsilon] \subset I \subset [a - \varepsilon, b + \varepsilon]$. Infinite intervals have infinite outer measure. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Approximation by Open Sets)</span></p>

For every subset $A \subset \mathbb{R}$ and $\varepsilon > 0$, there exists an open set $O$ such that $A \subset O$ and $m^\ast(A) \le m^\ast(O) \le m^\ast(A) + \varepsilon$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>If $m^*(A) = \infty$, take $O = \mathbb{R}$. Otherwise, let $\lbrace I_n \rbrace$ be open intervals covering $A$ with $\sum \ell(I_n) \le m^*(A) + \varepsilon$. Then $O = \bigcup_n I_n$ is open, $A \subset O$, and by subadditivity $m^*(O) \le \sum m^*(I_n) \le \sum \ell(I_n) \le m^*(A) + \varepsilon$. $\square$</p>
  </details>
</div>

### Lebesgue Measurable Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Measurable)</span></p>

A set $E \subset \mathbb{R}$ is **Lebesgue measurable** if for all $A \subset \mathbb{R}$,

$$m^*(A) = m^*(A \cap E) + m^*(A \cap E^c).$$

In other words, $E$ is well-behaved in that it always cuts any set $A$ into reasonable parts. Since subadditivity gives $m^\ast(A) \le m^\ast(A \cap E) + m^\ast(A \cap E^c)$ for free, measurability amounts to showing the reverse inequality.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Basic Measurability Facts)</span></p>

The empty set $\varnothing$ and $\mathbb{R}$ are measurable. A set $E$ is measurable if and only if $E^c$ is measurable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Zero Outer Measure Implies Measurable)</span></p>

If $m^*(E) = 0$, then $E$ is measurable.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since $A \cap E \subset E$, monotonicity gives $m^*(A \cap E) \le m^*(E) = 0$. So $m^*(A \cap E) + m^*(A \cap E^c) = m^*(A \cap E^c) \le m^*(A)$ (since $A \cap E^c \subset A$). This is the measurability condition. $\square$</p>
  </details>
</div>

Every open set and every closed set is measurable. More generally:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Finite Union of Measurable Sets)</span></p>

If $E_1$ and $E_2$ are measurable sets, then $E_1 \cup E_2$ is measurable.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let $A \subset \mathbb{R}$ be arbitrary. Since $E_2$ is measurable, $m^*(A \cap E_1^c) = m^*(A \cap E_1^c \cap E_2) + m^*(A \cap E_1^c \cap E_2^c)$. Note $E_1^c \cap E_2^c = (E_1 \cup E_2)^c$ by de Morgan, and $A \cap (E_1 \cup E_2) = (A \cap E_1) \cup (A \cap E_2 \cap E_1^c)$. So $m^*(A \cap (E_1 \cup E_2)) \le m^*(A \cap E_1) + m^*(A \cap E_2 \cap E_1^c)$. Using measurability of $E_1$: $m^*(A \cap E_1) + m^*(A \cap E_1^c) = m^*(A)$. Substituting and rearranging gives $m^*(A \cap (E_1 \cup E_2)) + m^*(A \cap (E_1 \cup E_2)^c) \le m^*(A)$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Finite Unions are Measurable)</span></p>

If $E_1, \dots, E_n$ are measurable, then $\bigcup_{k=1}^{n} E_k$ is measurable.

</div>

### Algebras and $\sigma$-Algebras

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Algebra and $\sigma$-Algebra)</span></p>

A nonempty collection $\mathcal{A} \subset \mathcal{P}(\mathbb{R})$ is an **algebra** if for all $E \in \mathcal{A}$, $E^c \in \mathcal{A}$, and for all $E_1, \dots, E_n \in \mathcal{A}$, $\bigcup_{k=1}^{n} E_k \in \mathcal{A}$. An algebra $\mathcal{A}$ is a **$\sigma$-algebra** if additionally for any countable collection $\lbrace E_n \rbrace_{n=1}^{\infty}$ in $\mathcal{A}$, the union $\bigcup_n E_n \in \mathcal{A}$.

</div>

Algebras are closed under complements and finite unions; $\sigma$-algebras are additionally closed under countable unions. By de Morgan's laws, closure under countable unions and complements implies closure under countable intersections.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\sigma$-Algebras)</span></p>

The simplest $\sigma$-algebra is $\mathcal{A} = \lbrace \varnothing, \mathbb{R} \rbrace$, and the largest is $\mathcal{A} = \mathcal{P}(\mathbb{R})$. The **cocountable $\sigma$-algebra** $\mathcal{A} = \lbrace E \subset \mathbb{R} : E \text{ or } E^c \text{ is countable} \rbrace$ is another example.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Borel $\sigma$-Algebra)</span></p>

The intersection of all $\sigma$-algebras containing all open subsets of $\mathbb{R}$,

$$\mathcal{B} = \bigcap_{\mathcal{A} \in \Sigma} \mathcal{A}, \quad \text{where } \Sigma = \lbrace \mathcal{A} : \mathcal{A} \text{ is a } \sigma\text{-algebra containing all open subsets of } \mathbb{R} \rbrace,$$

is the smallest $\sigma$-algebra containing all open subsets of $\mathbb{R}$, called the **Borel $\sigma$-algebra**. It is closed under complements and countable unions.

</div>

### The Measurable Sets Form a $\sigma$-Algebra

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Disjointification)</span></p>

Let $\mathcal{A}$ be an algebra, and let $\lbrace E_n \rbrace$ be a countable collection of elements of $\mathcal{A}$. Then there exists a disjoint countable collection $\lbrace F_n \rbrace$ of elements of $\mathcal{A}$ such that $\bigcup_n E_n = \bigcup_n F_n$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let $G_n = \bigcup_{k=1}^{n} E_k$, so $G_1 \subset G_2 \subset \cdots$ and $\bigcup_n E_n = \bigcup_n G_n$. Define $F_1 = G_1$ and $F_{n+1} = G_{n+1} \setminus G_n$ for $n \ge 1$. The $F_n$ are pairwise disjoint, each $F_n \in \mathcal{A}$ (since $\mathcal{A}$ is closed under finite unions and complements), and $\bigcup_{k=1}^{n} F_k = G_n$, so $\bigcup_n F_n = \bigcup_n G_n = \bigcup_n E_n$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Finite Additivity for Measurable Sets)</span></p>

Let $A \subset \mathbb{R}$, and let $E_1, \dots, E_n$ be disjoint measurable sets. Then

$$m^*\!\left(A \cap \left[\bigcup_{k=1}^{n} E_k\right]\right) = \sum_{k=1}^{n} m^*(A \cap E_k).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Measurable Sets Form a $\sigma$-Algebra)</span></p>

The collection $\mathcal{M}$ of Lebesgue measurable sets is a $\sigma$-algebra.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We already know $\mathcal{M}$ is an algebra. By the disjointification lemma, it suffices to show closure under countable disjoint unions. Let $\lbrace E_n \rbrace$ be pairwise disjoint measurable sets with $E = \bigcup_{n=1}^{\infty} E_n$. We need $m^*(A \cap E) + m^*(A \cap E^c) \le m^*(A)$ for all $A \subset \mathbb{R}$.</p>
    <p>Since $\bigcup_{n=1}^{N} E_n$ is measurable (finite union),</p>
    $$m^*(A) = m^*\!\left(A \cap \bigcup_{n=1}^{N} E_n\right) + m^*\!\left(A \cap \left[\bigcup_{n=1}^{N} E_n\right]^c\right).$$
    <p>Since $\left[\bigcup_{n=1}^{N} E_n\right]^c \supset E^c$, we get $m^*(A) \ge m^*(A \cap \bigcup_{n=1}^{N} E_n) + m^*(A \cap E^c)$. By finite additivity, $m^*(A \cap \bigcup_{n=1}^{N} E_n) = \sum_{n=1}^{N} m^*(A \cap E_n)$. Letting $N \to \infty$:</p>
    $$m^*(A) \ge \sum_{n=1}^{\infty} m^*(A \cap E_n) + m^*(A \cap E^c) \ge m^*\!\left(\bigcup_n (A \cap E_n)\right) + m^*(A \cap E^c) = m^*(A \cap E) + m^*(A \cap E^c),$$
    <p>where the last step uses countable subadditivity. $\square$</p>
  </details>
</div>

## Measurable Functions

The motivation for defining Lebesgue measurable sets is to build towards a theory of integration that surpasses Riemann integration. While Riemann integration chops up the **domain** into intervals, Lebesgue's theory chops up the **range** — looking at the piece of $f$ between two values $y_i$ and $y_{i+1}$, finding the corresponding $x$-values where $f$ intersects those $y$-values, and measuring that set. This is why we care about preimages of closed intervals being measurable.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name"></span></p>

Throughout the discussion of measurable functions and Lebesgue integration, we work with the **extended real numbers** $[-\infty, \infty] = \mathbb{R} \cup \lbrace -\infty, \infty \rbrace$, and we allow functions to take on the values $\pm \infty$. The arithmetic rules are: $x \pm \infty = \pm \infty$ for $x \in \mathbb{R}$, $0(\pm \infty) = 0$, and $x(\pm \infty) = \pm \infty$ for $x > 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Measurable Function)</span></p>

Let $E \subset \mathbb{R}$ be measurable, and let $f : E \to [-\infty, \infty]$ be a function. Then $f$ is **Lebesgue measurable** if for all $\alpha \in \mathbb{R}$, $f^{-1}((\alpha, \infty]) \in \mathcal{M}$ (i.e., the preimage is a measurable set).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Equivalent Conditions for Measurability)</span></p>

Let $E \subset \mathbb{R}$ be measurable and $f : E \to [-\infty, \infty]$. The following are equivalent:

1. For all $\alpha \in \mathbb{R}$, $f^{-1}((\alpha, \infty]) \in \mathcal{M}$.
2. For all $\alpha \in \mathbb{R}$, $f^{-1}([\alpha, \infty]) \in \mathcal{M}$.
3. For all $\alpha \in \mathbb{R}$, $f^{-1}([-\infty, \alpha)) \in \mathcal{M}$.
4. For all $\alpha \in \mathbb{R}$, $f^{-1}([-\infty, \alpha]) \in \mathcal{M}$.

</div>

<div class="accordion">
  <details>
    <summary>proof sketch</summary>
    <p>(1) $\Rightarrow$ (2): $[\alpha, \infty] = \bigcap_n (\alpha - 1/n, \infty]$, so $f^{-1}([\alpha, \infty]) = \bigcap_n f^{-1}((\alpha - 1/n, \infty])$ is a countable intersection of measurable sets.</p>
    <p>(2) $\Rightarrow$ (1): $(\alpha, \infty] = \bigcup_n [\alpha + 1/n, \infty]$, so $f^{-1}((\alpha, \infty])$ is a countable union of measurable sets.</p>
    <p>(2) $\leftrightarrow$ (3): $[-\infty, \alpha) = ([\alpha, \infty])^c$, so $f^{-1}([-\infty, \alpha)) = E \setminus f^{-1}([\alpha, \infty])$, measurable by closure under complements. Similarly for the other equivalences. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Preimages of Borel Sets are Measurable)</span></p>

If $E$ is measurable and $f : E \to \mathbb{R}$ is a measurable function, then for all $F \in \mathcal{B}$ (the Borel $\sigma$-algebra), $f^{-1}(F)$ is measurable.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>The preimage of any open interval $(a, b)$ is measurable: $f^{-1}((a, b)) = f^{-1}([-\infty, b)) \cap f^{-1}((a, \infty])$, and both sets are measurable. Since every open set is a countable union of open intervals, $f^{-1}(U)$ is measurable for all open $U$. The collection $\mathcal{A} = \lbrace F \subset \mathbb{R} : f^{-1}(F) \text{ measurable} \rbrace$ is a $\sigma$-algebra containing all open sets, so $\mathcal{B} \subset \mathcal{A}$. $\square$</p>
  </details>
</div>

If $f : E \to \mathbb{R}$ is measurable, then $f^{-1}(\lbrace \infty \rbrace)$ and $f^{-1}(\lbrace -\infty \rbrace)$ are also measurable (since $f^{-1}(\lbrace \infty \rbrace) = \bigcap_n f^{-1}((n, \infty])$).

### Examples of Measurable Functions

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Continuous Functions are Measurable)</span></p>

If $f : \mathbb{R} \to \mathbb{R}$ is continuous, then $f$ is measurable, because $f^{-1}((\alpha, \infty]) = f^{-1}((\alpha, \infty))$ is the preimage of an open set, hence open and thus measurable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Indicator Functions are Measurable)</span></p>

If $E, F \subset \mathbb{R}$ are two measurable sets, then the indicator function $\chi_F : E \to \mathbb{R}$ defined by $\chi_F(x) = 1$ if $x \in F$ and $\chi_F(x) = 0$ if $x \notin F$ is measurable.

</div>

### Closure Properties of Measurable Functions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Algebra of Measurable Functions)</span></p>

Let $E \subset \mathbb{R}$ be measurable, and suppose $f, g : E \to \mathbb{R}$ are two measurable functions and $c \in \mathbb{R}$. Then $cf$, $f + g$, and $fg$ are all measurable functions.

</div>

<div class="accordion">
  <details>
    <summary>proof sketch</summary>
    <p><strong>Scalar multiple:</strong> $(cf)^{-1}((\alpha, \infty]) = f^{-1}((\alpha/c, \infty])$ (for $c > 0$; similar for $c < 0$; trivial for $c = 0$).</p>
    <p><strong>Sum:</strong> $f(x) + g(x) > \alpha \iff f(x) > r > \alpha - g(x)$ for some rational $r$. So $(f + g)^{-1}((\alpha, \infty]) = \bigcup_{r \in \mathbb{Q}} (f^{-1}((r, \infty]) \cap g^{-1}((\alpha - r, \infty]))$, a countable union of measurable sets.</p>
    <p><strong>Product:</strong> First show $f^2$ is measurable: for $\alpha < 0$, $(f^2)^{-1}((\alpha, \infty]) = E$; for $\alpha \ge 0$, $(f^2)^{-1}((\alpha, \infty]) = f^{-1}((\sqrt{\alpha}, \infty]) \cup f^{-1}([-\infty, -\sqrt{\alpha}))$. Then use $fg = \frac{1}{4}((f + g)^2 - (f - g)^2)$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Closure Under Limits)</span></p>

Let $E \subset \mathbb{R}$ be measurable and $f_n : E \to [-\infty, \infty]$ a sequence of measurable functions. Then the functions 

$$\sup_n f_n, \quad \inf_n f_n, \quad \limsup_{n \to \infty} f_n, \quad \liminf_{n \to \infty} f_n$$

are all measurable.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>For $g_1 = \sup_n f_n$: $x \in g_1^{-1}((\alpha, \infty]) \iff \sup_n f_n(x) > \alpha \iff f_n(x) > \alpha$ for some $n$, so $g_1^{-1}((\alpha, \infty]) = \bigcup_n f_n^{-1}((\alpha, \infty])$, a countable union of measurable sets.</p>
    <p>For $g_2 = \inf_n f_n$: $x \in g_2^{-1}([\alpha, \infty]) \iff \inf_n f_n(x) \ge \alpha \iff f_n(x) \ge \alpha$ for all $n$, so $g_2^{-1}([\alpha, \infty]) = \bigcap_n f_n^{-1}([\alpha, \infty])$.</p>
    <p>For $\limsup$ and $\liminf$: $\limsup_{n \to \infty} f_n = \inf_n [\sup_{k \ge n} f_k]$ and $\liminf_{n \to \infty} f_n = \sup_n [\inf_{k \ge n} f_k]$, compositions of $\sup$ and $\inf$ of measurable functions. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Pointwise Limits of Measurable Functions)</span></p>

If $f_n : E \to [-\infty, \infty]$ are measurable for all $n$ and $\lim_{n \to \infty} f_n(x) = f(x)$ for all $x \in E$, then $f$ is measurable.

</div>

This corollary is **false** for Riemann integration — the pointwise limit of Riemann integrable functions need not be Riemann integrable.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Pointwise Limit Leaving Riemann Integration)</span></p>

Enumerate $\mathbb{Q} \cap [0, 1] = \lbrace r_1, r_2, r_3, \dots \rbrace$. Define $f_n(x) = 1$ if $x \in \lbrace r_1, \dots, r_n \rbrace$ and $f_n(x) = 0$ otherwise. Each $f_n$ is Riemann integrable (piecewise continuous), but $\lim f_n = \chi_{\mathbb{Q} \cap [0,1]}$, which is not Riemann integrable.

</div>

### Almost Everywhere and Measurability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Almost Everywhere)</span></p>

Let $E$ be a measurable set. A statement $P(x)$ **holds almost everywhere (a.e.) on $E$** if

$$m(\lbrace x \in E : P(x) \text{ does not hold} \rbrace) = 0.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Modification on Measure Zero Preserves Measurability)</span></p>

If $f, g : E \to [-\infty, \infty]$ satisfy $f = g$ a.e. on $E$, and $f$ is measurable, then $g$ is measurable.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let $N = \lbrace x \in E : f(x) \ne g(x) \rbrace$ with $m(N) = 0$. For $\alpha \in \mathbb{R}$, $N_\alpha = \lbrace x \in N : g(x) > \alpha \rbrace \subset N$ has $m^*(N_\alpha) \le m^*(N) = 0$, so $N_\alpha$ is measurable. Then $g^{-1}((\alpha, \infty]) = (f^{-1}((\alpha, \infty]) \cap N^c) \cup N_\alpha$. Since $N$ is measurable, $N^c$ is measurable, the intersection with $f^{-1}((\alpha, \infty])$ is measurable, and $N_\alpha$ is measurable, so the union is measurable. $\square$</p>
  </details>
</div>

### Complex-Valued Measurable Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complex-Valued Measurable Function)</span></p>

If $E \subset \mathbb{R}$ is measurable, a complex-valued function $f : E \to \mathbb{C}$ is **measurable** if $\operatorname{Re}(f)$ and $\operatorname{Im}(f)$ (which are both functions $E \to \mathbb{R}$) are measurable.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Algebra of Complex Measurable Functions)</span></p>

If $f, g : E \to \mathbb{C}$ are measurable functions and $\alpha \in \mathbb{C}$, then $\alpha f$, $f + g$, $fg$, $\bar{f}$, and $\lvert f \rvert$ are all measurable.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Pointwise Limits of Complex Measurable Functions)</span></p>

If $f_n : E \to \mathbb{C}$ is measurable for all $n$ and $f_n(x) \to f(x)$ pointwise for all $x \in E$, then $f$ is measurable.

</div>

## Simple Functions and the Lebesgue Integral

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Simple Function)</span></p>

A measurable function $\phi : E \to \mathbb{C}$ is **simple** if $\lvert \phi(E) \rvert$ (the size of the range) is finite. Any simple function can be written as

$$\phi(x) = \sum_{i=1}^{n} a_i \cdot \chi_{A_i}(x),$$

where $a_1, \dots, a_n$ are the distinct values of $\phi$, and $A_i = \phi^{-1}(\lbrace a_i \rbrace)$ are pairwise disjoint measurable sets with $\bigcup_{i=1}^{n} A_i = E$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Closure of Simple Functions)</span></p>

Scalar multiples, linear combinations, and products of simple functions are again simple functions.

</div>

### Approximation by Simple Functions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Simple Function Approximation — Nonnegative Case)</span></p>

If $f : E \to [0, \infty]$ is a nonnegative measurable function, then there exists a sequence of simple functions $\lbrace \phi_n \rbrace$ such that:

* (a) $0 \le \phi_0(x) \le \phi_1(x) \le \cdots \le f(x)$ for all $x \in E$ (pointwise increasing).
* (b) $\lim_{n \to \infty} \phi_n(x) = f(x)$ for all $x \in E$ (pointwise convergence).
* (c) For all $B \ge 0$, $\phi_n \to f$ converges uniformly on $\lbrace x \in E : f(x) \le B \rbrace$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Build $\phi_n$ with resolution $2^{-n}$ and range $[0, 2^n]$. Define</p>
    $$E_n^k = \lbrace x \in E : k 2^{-n} < f(x) \le (k+1)2^{-n} \rbrace \quad (0 \le k \le 2^{2n} - 1), \qquad F_n = f^{-1}((2^n, \infty]),$$
    <p>and set $\phi_n = \sum_{k=0}^{2^{2n}-1} (k 2^{-n}) \chi_{E_n^k} + 2^n \chi_{F_n}$. Then $0 \le \phi_n \le f$ by construction.</p>
    <p><strong>Monotonicity (a):</strong> If $x \in E_n^k$ then $k2^{-n} < f(x) \le (k+1)2^{-n}$, which implies $x \in E_{n+1}^{2k} \cup E_{n+1}^{2k+1}$. In both cases $\phi_{n+1}(x) \ge (2k)2^{-(n+1)} = k2^{-n} = \phi_n(x)$. Similarly for $x \in F_n$.</p>
    <p><strong>Convergence (b) and (c):</strong> On the set $\lbrace x : f(x) \le 2^n \rbrace$, we have $0 \le f(x) - \phi_n(x) \le 2^{-n}$. For any $x$ with $f(x) < \infty$, eventually $f(x) \le 2^n$ for large $n$, giving pointwise convergence. For $f(x) = \infty$, $\phi_n(x) = 2^n \to \infty$. On $\lbrace f(x) \le B \rbrace \subset \lbrace f(x) \le 2^N \rbrace$ for $N$ large enough, the bound $2^{-n}$ gives uniform convergence. $\square$</p>
  </details>
</div>

### Extension to General Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Positive and Negative Parts)</span></p>

Let $f : E \to [-\infty, \infty]$ be a measurable function. The **positive part** and **negative part** of $f$ are

$$f^+(x) = \max(f(x), 0), \qquad f^-(x) = \max(-f(x), 0),$$

so that $f = f^+ - f^-$ and $\lvert f \rvert = f^+ + f^-$. Both $f^+$ and $f^-$ are nonnegative measurable functions.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Simple Function Approximation — General Case)</span></p>

Let $E \subset \mathbb{R}$ be measurable and $f : E \to \mathbb{C}$ be measurable. Then there exists a sequence of simple functions $\lbrace \phi_n \rbrace$ such that $0 \le \lvert \phi_0(x) \rvert \le \lvert \phi_1(x) \rvert \le \cdots \le \lvert f(x) \rvert$, $\lim_{n \to \infty} \phi_n(x) = f(x)$ for all $x \in E$, and $\phi_n \to f$ uniformly on $\lbrace x : \lvert f(x) \rvert \le B \rbrace$ for all $B \ge 0$.

</div>

The idea: split $f$ into real and imaginary parts, then each into positive and negative parts, apply the nonnegative approximation theorem to each, and recombine.

## The Lebesgue Integral

### Integral of Nonnegative Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^+(E)$)</span></p>

If $E \subset \mathbb{R}$ is measurable, define $L^+(E) = \lbrace f : E \to [0, \infty] : f \text{ measurable} \rbrace$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Integral of Simple Functions)</span></p>

For a simple function $\phi \in L^+(E)$ with canonical form $\phi = \sum_{j=1}^{n} a_j \chi_{A_j}$ (where $A_i \cap A_j = \varnothing$ for $i \ne j$ and $\bigcup_j A_j = E$), the **Lebesgue integral** is

$$\int_E \phi = \sum_{j=1}^{n} a_j \, m(A_j) \in [0, \infty].$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Properties of the Integral for Simple Functions)</span></p>

For simple functions $\phi, \psi \in L^+(E)$ and $c \ge 0$:

1. $\int_E c\phi = c \int_E \phi$.
2. $\int_E (\phi + \psi) = \int_E \phi + \int_E \psi$.
3. If $\phi \le \psi$, then $\int_E \phi \le \int_E \psi$.
4. If $F \subset E$ is measurable, then $\int_F \phi = \int_E \chi_F \phi \le \int_E \phi$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Integral of Nonnegative Measurable Functions)</span></p>

For $f \in L^+(E)$, the **Lebesgue integral** of $f$ is

$$\int_E f = \sup \left\lbrace \int_E \phi : \phi \in L^+(E) \text{ simple},\; \phi \le f \right\rbrace.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Integral over Measure-Zero Set)</span></p>

If $E \subset \mathbb{R}$ with $m(E) = 0$, then $\int_E f = 0$ for all $f \in L^+(E)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Basic Properties of the Nonnegative Integral)</span></p>

If $f, g \in L^+(E)$, $c \in [0, \infty)$, and $F \subset E$ is measurable, then: $\int_E cf = c \int_E f$, if $f \le g$ then $\int_E f \le \int_E g$, if $f \le g$ a.e. then $\int_E f \le \int_E g$, and $\int_F f \le \int_E f$.

</div>

### The Monotone Convergence Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Monotone Convergence Theorem)</span></p>

If $\lbrace f_n \rbrace$ is a sequence of nonnegative measurable functions (in $L^+(E)$) such that $f_1 \le f_2 \le \cdots$ pointwise on $E$, and $f_n \to f$ pointwise on $E$ for some $f$ (which is in $L^+(E)$ by closure under limits), then

$$\lim_{n \to \infty} \int_E f_n = \int_E f.$$

</div>

The assumption of pointwise convergence here is much weaker than the uniform convergence we usually need for Riemann integration.

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since $f_n \le f$ for all $n$, $\int_E f_n \le \int_E f$, so $\lim \int_E f_n \le \int_E f$. For the reverse, it suffices to show $\int_E \phi \le \lim \int_E f_n$ for every simple $\phi \le f$.</p>
    <p>Fix $\varepsilon \in (0, 1)$ and let $\phi = \sum_{j=1}^{m} a_j \chi_{A_j}$ be simple with $\phi \le f$. Define $E_n = \lbrace x \in E : f_n(x) \ge (1 - \varepsilon)\phi(x) \rbrace$. Since $f_n \nearrow f \ge \phi$, every $x$ is eventually in some $E_n$, so $\bigcup_{n=1}^{\infty} E_n = E$ and $E_1 \subset E_2 \subset \cdots$.</p>
    <p>Then $\int_E f_n \ge \int_{E_n} f_n \ge (1 - \varepsilon) \int_{E_n} \phi = (1 - \varepsilon) \sum_j a_j m(A_j \cap E_n)$. By continuity of measure ($A_j \cap E_n \nearrow A_j$), taking $n \to \infty$ gives $\lim \int_E f_n \ge (1 - \varepsilon) \sum_j a_j m(A_j) = (1 - \varepsilon) \int_E \phi$. Since $\varepsilon$ is arbitrary, $\lim \int_E f_n \ge \int_E \phi$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Computing Integrals via Simple Approximations)</span></p>

Let $f \in L^+(E)$, and let $\lbrace \phi_n \rbrace$ be a sequence of simple functions with $0 \le \phi_1 \le \phi_2 \le \cdots \le f$ and $\phi_n \to f$ pointwise. Then $\int_E f = \lim_{n \to \infty} \int_E \phi_n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Additivity of the Nonnegative Integral)</span></p>

If $f, g \in L^+(E)$, then $\int_E (f + g) = \int_E f + \int_E g$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Take sequences of simple functions $\lbrace \phi_n \rbrace$ and $\lbrace \psi_n \rbrace$ increasing pointwise to $f$ and $g$ respectively. Then $\phi_n + \psi_n$ increases pointwise to $f + g$, and each $\phi_n + \psi_n$ is simple. By the Monotone Convergence Theorem applied to each:</p>
    $$\int_E (f + g) = \lim_{n \to \infty} \int_E (\phi_n + \psi_n) = \lim_{n \to \infty} \left(\int_E \phi_n + \int_E \psi_n\right) = \int_E f + \int_E g. \quad\square$$
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Countable Additivity of the Integral)</span></p>

Let $\lbrace f_n \rbrace$ be a sequence in $L^+(E)$. Then $\int_E \sum_n f_n = \sum_n \int_E f_n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Zero Integral Characterization)</span></p>

Let $f \in L^+(E)$. Then $\int_E f = 0$ if and only if $f = 0$ almost everywhere on $E$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>($\Rightarrow$)</strong> If $f = 0$ a.e., then $f \le 0$ a.e., so $\int_E f \le \int_E 0 = 0$.</p>
    <p><strong>($\Leftarrow$)</strong> Define $F_n = \lbrace x \in E : f(x) > 1/n \rbrace$ and $F = \lbrace x \in E : f(x) > 0 \rbrace = \bigcup_n F_n$. Then $\frac{1}{n} m(F_n) = \int_{F_n} \frac{1}{n} \le \int_{F_n} f \le \int_E f = 0$, so $m(F_n) = 0$ for all $n$. By continuity of measure, $m(F) = \lim m(F_n) = 0$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Monotone Convergence — a.e. Version)</span></p>

If $\lbrace f_n \rbrace$ is a sequence in $L^+(E)$ with $f_1 \le f_2 \le \cdots$ a.e. on $E$ and $\lim f_n = f$ a.e. on $E$, then $\int_E f = \lim_{n \to \infty} \int_E f_n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Finite Integral Implies Finite a.e.)</span></p>

If $f \in L^+(E)$ and $\int_E f < \infty$, then $\lbrace x \in E : f(x) = \infty \rbrace$ has measure zero.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let $F = \lbrace x \in E : f(x) = \infty \rbrace$. For all $n$, $n \chi_F \le f$, so $n \cdot m(F) = \int_E n \chi_F \le \int_E f < \infty$. Thus $m(F) \le \frac{1}{n} \int_E f \to 0$. $\square$</p>
  </details>
</div>

### Fatou's Lemma

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fatou's Lemma)</span></p>

Let $\lbrace f_n \rbrace$ be a sequence in $L^+(E)$. Then

$$\int_E \liminf_{n \to \infty} f_n \le \liminf_{n \to \infty} \int_E f_n.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We have $\liminf_{n \to \infty} f_n(x) = \sup_{n \ge 1} [\inf_{k \ge n} f_k(x)] = \lim_{n \to \infty} [\inf_{k \ge n} f_k(x)]$, and the sequence $g_n = \inf_{k \ge n} f_k$ is pointwise increasing. By the Monotone Convergence Theorem,</p>
    $$\int_E \liminf f_n = \lim_{n \to \infty} \int_E \left(\inf_{k \ge n} f_k\right).$$
    <p>Since $\inf_{k \ge n} f_k \le f_j$ for all $j \ge n$, we get $\int_E \inf_{k \ge n} f_k \le \int_E f_j$ for all $j \ge n$, hence $\int_E \inf_{k \ge n} f_k \le \inf_{j \ge n} \int_E f_j$. Taking $n \to \infty$: $\int_E \liminf f_n \le \lim_{n \to \infty} \inf_{j \ge n} \int_E f_j = \liminf \int_E f_n$. $\square$</p>
  </details>
</div>

### The General Lebesgue Integral

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Integrable Function)</span></p>

Let $E \subset \mathbb{R}$ be measurable. A measurable function $f : E \to \mathbb{R}$ is **Lebesgue integrable** over $E$ if $\int_E \lvert f \rvert < \infty$.

Since $\lvert f \rvert = f^+ + f^-$, this is equivalent to both $\int_E f^+$ and $\int_E f^-$ being finite.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Integral of a Real-Valued Function)</span></p>

The **Lebesgue integral** of an integrable function $f : E \to \mathbb{R}$ is

$$\int_E f = \int_E f^+ - \int_E f^-.$$

This is well-defined because both terms are finite (so we never subtract $\infty - \infty$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Compact subsets of $\mathbb{R}$ are Borel sets with finite measure, so simple functions with compact support are integrable. Continuous functions on a closed bounded interval $[a, b]$ attain a finite maximum, so $\int_{[a,b]} \lvert f \rvert \le c(b - a) < \infty$: every continuous function on a closed bounded interval is Lebesgue integrable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Properties of the Lebesgue Integral)</span></p>

Suppose $f, g : E \to \mathbb{R}$ are integrable. Then:

1. For all $c \in \mathbb{R}$, $cf$ is integrable and $\int_E cf = c \int_E f$.
2. $f + g$ is integrable and $\int_E (f + g) = \int_E f + \int_E g$.
3. If $A, B$ are disjoint measurable sets, $\int_{A \cup B} f = \int_A f + \int_B f$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Comparison Properties)</span></p>

Suppose $f, g : E \to \mathbb{R}$ are measurable. Then:

1. If $f$ is integrable, $\left\lvert \int_E f \right\rvert \le \int_E \lvert f \rvert$.
2. If $g$ is integrable and $f = g$ a.e., then $f$ is integrable and $\int_E f = \int_E g$.
3. If $f, g$ are integrable and $f \le g$ a.e., then $\int_E f \le \int_E g$.

</div>

### The Dominated Convergence Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Dominated Convergence Theorem)</span></p>

Let $g : E \to [0, \infty)$ be a nonnegative integrable function, and let $\lbrace f_n \rbrace$ be a sequence of real-valued measurable functions such that (1) $\lvert f_n \rvert \le g$ a.e. for all $n$, and (2) $f_n(x) \to f(x)$ pointwise a.e. on $E$. Then

$$\lim_{n \to \infty} \int_E f_n = \int_E f.$$

</div>

This is much stronger than anything available for Riemann integration — we only need pointwise convergence and a dominating integrable function.

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since $\lvert f_n \rvert \le g$ a.e., each $f_n$ is integrable, and $\lvert f \rvert \le g$ a.e. (taking limits), so $f$ is integrable. Also $\lvert \int_E f_n \rvert \le \int_E \lvert f_n \rvert \le \int_E g < \infty$.</p>
    <p>Apply Fatou's lemma to $g - f_n \ge 0$ (a.e.):</p>
    $$\int_E (g - f) = \int_E \liminf(g - f_n) \le \liminf \int_E (g - f_n) = \int_E g - \limsup \int_E f_n.$$
    <p>Since $\int_E g < \infty$, rearranging gives $\limsup \int_E f_n \le \int_E f$.</p>
    <p>Apply Fatou's lemma to $g + f_n \ge 0$ (a.e.):</p>
    $$\int_E (g + f) = \int_E \liminf(g + f_n) \le \liminf \int_E (g + f_n) = \int_E g + \liminf \int_E f_n.$$
    <p>Rearranging: $\int_E f \le \liminf \int_E f_n$. Combining: $\int_E f \le \liminf \int_E f_n \le \limsup \int_E f_n \le \int_E f$, so $\lim \int_E f_n = \int_E f$. $\square$</p>
  </details>
</div>

### Riemann and Lebesgue Integrals Agree

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 126</span></p>

Let $f \in C([a, b])$ for some real numbers $a < b$. Then $\int_{[a,b]} f = \int_a^b f(x)\,dx$: in other words, $f$ is integrable and the Riemann and Lebesgue integrals agree.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since $f \in C([a,b])$ is continuous on a closed and bounded interval, $\lvert f \rvert$ is also continuous and bounded. There exists $B \ge 0$ with $\lvert f \rvert \le B$ on $[a,b]$, so $\int_{[a,b]} \lvert f \rvert \le B \cdot m([a,b]) < \infty$, confirming $f$ is Lebesgue integrable.</p>
    <p>Write $f = f^+ - f^-$ where $f^+ = \frac{f + \lvert f \rvert}{2}$ and $f^- = \frac{\lvert f \rvert - f}{2}$. By linearity it suffices to prove the result for nonnegative $f$.</p>
    <p>Take a sequence of partitions $\underline{x}^n = \lbrace a = x_0^n, x_1^n, \dots, x_{m_n}^n = b \rbrace$ with $\lvert \underline{x}^n \rvert \to 0$. For each $j, n$, let $\xi_j^n \in [x_{j-1}^n, x_j^n]$ achieve the minimum of $f$ on that subinterval (by the Extreme Value Theorem). The lower Riemann sums converge to the Riemann integral:</p>
    $$\lim_{n \to \infty} \sum_{j=1}^{m_n} f(\xi_j^n)(x_j^n - x_{j-1}^n) = \int_a^b f(x)\,dx.$$
    <p>Define $N = \bigcup_{n=1}^{\infty} \underline{x}^n$, a countable set with $m(N) = 0$. The simple functions $f_n = \sum_{j=1}^{m_n} f(\xi_j^n)\chi_{[x_{j-1}^n, x_j^n]}$ satisfy $0 \le f_n(x) \le f(x)$ for all $x \in [a,b] \setminus N$, and $f_n \to f$ pointwise on $[a,b] \setminus N$ (by continuity of $f$ and the partition norms going to zero). All $f_n$ are dominated by the integrable function $f$, so by the Dominated Convergence Theorem:</p>
    $$\int_{[a,b]} f = \lim_{n \to \infty} \int_{[a,b]} f_n = \lim_{n \to \infty} \sum_{j=1}^{m_n} f(\xi_j^n)(x_j^n - x_{j-1}^n) = \int_a^b f(x)\,dx. \quad \square$$
  </details>
</div>

### Complex-Valued Integrable Functions

Everything proved for real integrable functions carries over to complex-valued integrable functions: we define $f : E \to \mathbb{C}$ to be Lebesgue integrable if $\int_E \lvert f \rvert < \infty$, in which case

$$\int_E f = \int_E \operatorname{Re} f + i \int_E \operatorname{Im} f.$$

Results like linearity of the integral and the Dominated Convergence Theorem generalize.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 127</span></p>

If $f : E \to \mathbb{C}$ is integrable, then $\left\lvert \int_E f \right\rvert \le \int_E \lvert f \rvert$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>If $\int_E f = 0$ the result is clear. Otherwise, define $\alpha = \overline{(\int_E f)} / \lvert \int_E f \rvert$, so $\lvert \alpha \rvert = 1$. Then</p>
    $$\left\lvert \int_E f \right\rvert = \alpha \int_E f = \int_E \alpha f = \operatorname{Re} \int_E \alpha f = \int_E \operatorname{Re}(\alpha f) \le \int_E \lvert \operatorname{Re}(\alpha f) \rvert \le \int_E \lvert \alpha f \rvert = \int_E \lvert f \rvert,$$
    <p>using $\operatorname{Re}(z) \le \lvert z \rvert$ and $\lvert \alpha \rvert = 1$. $\square$</p>
  </details>
</div>

## $L^p$ Spaces

We now find the "complete space of integrable functions" that contains the space of continuous functions.

### The $L^p$ Norm

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 128</span><span class="math-callout__name">($L^p$ Norm and Essential Supremum)</span></p>

Let $f : E \to \mathbb{C}$ be a measurable function. For any $1 \le p < \infty$, we define the **$L^p$ norm**

$$\lVert f \rVert_{L^p(E)} = \left( \int_E \lvert f \rvert^p \right)^{1/p}.$$

Furthermore, we define the **$L^\infty$ norm** or **essential supremum** of $f$ as

$$\lVert f \rVert_{L^\infty(E)} = \inf \lbrace M > 0 : m(\lbrace x \in E : \lvert f(x) \rvert > M \rbrace) = 0 \rbrace.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 129</span></p>

If $f : E \to \mathbb{C}$ is measurable, then $\lvert f(x) \rvert \le \lVert f \rVert_{L^\infty(E)}$ almost everywhere on $E$. Also, if $E = [a, b]$ is a closed interval and $f \in C([a, b])$, then $\lVert f \rVert_{L^\infty([a,b])} = \lVert f \rVert_\infty$ is the usual sup norm on bounded continuous functions.

</div>

### Hölder's and Minkowski's Inequalities

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 130</span><span class="math-callout__name">(Hölder's Inequality for $L^p$ Spaces)</span></p>

If $1 \le p \le \infty$ and $\frac{1}{p} + \frac{1}{q} = 1$, and $f, g : E \to \mathbb{C}$ are measurable functions, then

$$\int_E \lvert fg \rvert \le \lVert f \rVert_{L^p(E)} \lVert g \rVert_{L^q(E)}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 131</span><span class="math-callout__name">(Minkowski's Inequality for $L^p$ Spaces)</span></p>

If $1 \le p \le \infty$ and $f, g : E \to \mathbb{C}$ are two measurable functions, then $\lVert f + g \rVert_{L^p(E)} \le \lVert f \rVert_{L^p(E)} + \lVert g \rVert_{L^p(E)}$.

</div>

A similar result also holds for $L^\infty(E)$. We use the shorthand $\lVert \cdot \rVert_p$ for $\lVert \cdot \rVert_{L^p(E)}$ from now on.

### The $L^p$ Space

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 133</span><span class="math-callout__name">($L^p$ Space)</span></p>

For any $1 \le p \le \infty$, we define the **$L^p$ space**

$$L^p(E) = \lbrace f : E \to \mathbb{C} : f \text{ measurable and } \lVert f \rVert_p < \infty \rbrace,$$

where we consider two elements $f, g$ of $L^p(E)$ to be equivalent (the same) if $f = g$ almost everywhere.

</div>

We need the equivalence relation to make the $L^p$ norms actually norms: the space is really a space of equivalence classes $[f] = \lbrace g : E \to \mathbb{C} : \lVert g \rVert_p < \infty \text{ and } g = f \text{ a.e.} \rbrace$, rather than functions. But we still refer to elements as functions (as is custom in mathematics).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 134</span></p>

This might seem like a weird thing to do, but recall that the rational numbers are constructed as equivalence classes of pairs of integers, and we think of $\frac{3}{2}$ as that quantity rather than the set of $(3x, 2x)$ for nonzero integers $x$. What really matters is the properties of the equivalence class, and for our functions in $L^p(E)$, behavior on a set of measure zero does not matter.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 135</span></p>

The space $L^p(E)$ with pointwise addition and natural scalar multiplication operations is a vector space, and it is a normed vector space under $\lVert \cdot \rVert_p$.

</div>

<div class="accordion">
  <details>
    <summary>proof sketch</summary>
    <p>The $L^p$ norm $\lVert \cdot \rVert_p$ is well-defined on equivalence classes: if $f = g$ a.e., then $\lvert f \rvert^p = \lvert g \rvert^p$ a.e., so $\int_E \lvert f \rvert^p = \int_E \lvert g \rvert^p$ and $\lVert f \rVert_p = \lVert g \rVert_p$.</p>
    <p>Scalar multiplication and pointwise addition are well-defined on equivalence classes. For the norm properties: if $\int_E \lvert f \rvert^p = 0$, then $\lvert f \rvert^p = 0$ a.e., so $f = 0$ a.e. (the equivalence class $[0]$). This proves definiteness. Homogeneity and the triangle inequality follow from the definition and Minkowski's inequality, respectively. $\square$</p>
  </details>
</div>

### Properties of $L^p$ Spaces

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 136</span></p>

Let $E \subset \mathbb{R}$ be measurable. Then $f \in L^p(E)$ if and only if

$$\lim_{n \to \infty} \int_{[-n,n] \cap E} \lvert f \rvert^p < \infty.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>The sequence $\lbrace \chi_{[-n,n]} \lvert f \rvert^p \rbrace$ is a pointwise increasing sequence of measurable functions, and for all $x \in E$, $\lim_{n \to \infty} \chi_{[-n,n]}(x) \lvert f(x) \rvert^p = \lvert f(x) \rvert^p$. By the Monotone Convergence Theorem, $\int_E \lvert f \rvert^p = \lim_{n \to \infty} \int_{[-n,n] \cap E} \lvert f \rvert^p$, and thus the two quantities are finite for exactly the same set of $f$s. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 137</span></p>

If $f : \mathbb{R} \to \mathbb{C}$ is measurable and there exist $C \ge 0$ and $q > 1$ such that for almost every $x \in \mathbb{R}$,

$$\lvert f(x) \rvert \ge C(1 + \lvert x \rvert)^{-q},$$

then $f \in L^p(\mathbb{R})$ for all $p \ge 1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 138</span></p>

Let $a < b$ and $1 \le p < \infty$ so that $f \in L^p([a, b])$, and take some $\varepsilon > 0$. Then there exists some $g \in C([a, b])$ such that $g(a) = g(b) = 0$, so that $\lVert f - g \rVert_p < \varepsilon$.

</div>

In other words, the space of continuous functions $C([a, b])$ is dense in $L^p([a, b])$, and it is a proper subset because we can find elements in $L^p$ that are not continuous.

### Riesz–Fischer Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 139</span><span class="math-callout__name">(Riesz–Fischer)</span></p>

For all $1 \le p \le \infty$, $L^p(E)$ is a Banach space.

</div>

<div class="accordion">
  <details>
    <summary>proof (for finite $p$)</summary>
    <p>Recall that a normed space is Banach if and only if every absolutely summable series is summable. Suppose $\lbrace f_k \rbrace$ is a sequence in $L^p(E)$ with $\sum_k \lVert f_k \rVert_p = M < \infty$.</p>
    <p>Define $g_n(x) = \sum_{k=1}^{n} \lvert f_k(x) \rvert$. By the triangle inequality, $\lVert g_n \rVert_p \le \sum_{k=1}^{n} \lVert f_k \rVert_p \le M$. By Fatou's lemma, $\int_E \left( \sum_{k=1}^{\infty} \lvert f_k \rvert \right)^p \le M^p < \infty$, so $\sum_k \lvert f_k(x) \rvert$ is finite a.e.</p>
    <p>Define $f(x) = \sum_k f_k(x)$ where the sum converges absolutely, and $f(x) = 0$ otherwise. Then $\lvert \sum_{k=1}^{n} f_k(x) - f(x) \rvert^p \to 0$ a.e. and $\lvert \sum_{k=1}^{n} f_k - f \rvert^p \le \lvert g \rvert^p$ a.e. where $g = \sum_k \lvert f_k \rvert$ satisfies $\lVert g \rVert_p \le M$ and $\int_E \lvert g \rvert^p < \infty$. By the Dominated Convergence Theorem, $\lim_{n \to \infty} \int_E \lvert \sum_{k=1}^{n} f_k - f \rvert^p = 0$, so $L^p$ is indeed a Banach space. $\square$</p>
  </details>
</div>

Since $C([a, b])$ is dense in $L^p([a, b])$ and the latter is a Banach space, we can think of the $L^p$ space as a **completion** of the continuous functions.

## Hilbert Spaces

From here, we move on to more general topics in functional analysis. Our next topic will be **Hilbert spaces**, which give us the important notions of an inner product, orthogonality, and so on.

### Pre-Hilbert Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 140</span><span class="math-callout__name">(Pre-Hilbert Space)</span></p>

A **pre-Hilbert space** $H$ is a vector space over $\mathbb{C}$ with a **Hermitian inner product**, which is a map $\langle \cdot, \cdot \rangle : H \times H \to \mathbb{C}$ satisfying the following properties:

1. For all $\lambda_1, \lambda_2 \in \mathbb{C}$ and $v_1, v_2, w \in H$, we have $\langle \lambda_1 v_1 + \lambda_2 v_2, w \rangle = \lambda_1 \langle v_1, w \rangle + \lambda_2 \langle v_2, w \rangle$.
2. For all $v, w \in H$, we have $\langle v, w \rangle = \overline{\langle w, v \rangle}$.
3. For all $v \in H$, we have $\langle v, v \rangle \ge 0$, with equality if and only if $v = 0$.

</div>

We should think of pre-Hilbert spaces as **normed vector spaces where the norm comes from an inner product**. The inner product is linear in the first variable but conjugate-linear in the second: $\langle v, \lambda w \rangle = \overline{\lambda} \langle v, w \rangle$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 141</span><span class="math-callout__name">(Norm from Inner Product)</span></p>

Let $H$ be a pre-Hilbert space. Then for any $v \in H$, we define

$$\lVert v \rVert = \langle v, v \rangle^{1/2}.$$

</div>

### Cauchy–Schwarz Inequality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 142</span><span class="math-callout__name">(Cauchy–Schwarz Inequality)</span></p>

Let $H$ be a pre-Hilbert space. For all $u, v \in H$, we have

$$\lvert \langle u, v \rangle \rvert \le \lVert u \rVert \, \lVert v \rVert.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Define $f(t) = \lVert u + tv \rVert^2 = \langle u + tv, u + tv \rangle = \lVert u \rVert^2 + t^2 \lVert v \rVert^2 + 2t \operatorname{Re}(\langle u, v \rangle)$, which is a nonnegative quadratic in $t$. It achieves its minimum at $t_{\min} = \frac{-\operatorname{Re}(\langle u, v \rangle)}{\lVert v \rVert^2}$. Substituting:</p>
    $$0 \le f(t_{\min}) = \lVert u \rVert^2 - \frac{\lvert \operatorname{Re}(\langle u, v \rangle) \rvert^2}{\lVert v \rVert^2},$$
    <p>giving $\lvert \operatorname{Re}(\langle u, v \rangle) \rvert \le \lVert u \rVert \, \lVert v \rVert$. For the full result, suppose $\langle u, v \rangle \ne 0$ and define $\lambda = \overline{\langle u, v \rangle} / \lvert \langle u, v \rangle \rvert$ with $\lvert \lambda \rvert = 1$. Then $\lvert \langle u, v \rangle \rvert = \lambda \langle u, v \rangle = \langle \lambda u, v \rangle = \operatorname{Re} \langle \lambda u, v \rangle \le \lVert \lambda u \rVert \, \lVert v \rVert = \lVert u \rVert \, \lVert v \rVert$, since $\langle \lambda u, \lambda u \rangle = \lambda \overline{\lambda} \langle u, u \rangle = \langle u, u \rangle$. $\square$</p>
  </details>
</div>

### The Norm on a Pre-Hilbert Space

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 143</span></p>

If $H$ is a pre-Hilbert space, then $\lVert \cdot \rVert$ is a norm on $H$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Positive definiteness: $\lVert v \rVert \ge 0$ and $\lVert v \rVert = 0 \iff \langle v, v \rangle = 0 \iff v = 0$. Homogeneity: $\langle \lambda v, \lambda v \rangle = \lambda \overline{\lambda} \langle v, v \rangle$ implies $\lVert \lambda v \rVert = \lvert \lambda \rvert \, \lVert v \rVert$.</p>
    <p>Triangle inequality: $\lVert u + v \rVert^2 = \langle u + v, u + v \rangle = \lVert u \rVert^2 + \lVert v \rVert^2 + 2 \operatorname{Re}(\langle u, v \rangle)$. Since $\operatorname{Re}(z) \le \lvert z \rvert$ and by Cauchy–Schwarz, this is $\le \lVert u \rVert^2 + \lVert v \rVert^2 + 2 \lVert u \rVert \, \lVert v \rVert = (\lVert u \rVert + \lVert v \rVert)^2$. $\square$</p>
  </details>
</div>

### Continuity of the Inner Product

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 144</span><span class="math-callout__name">(Continuity of the Inner Product)</span></p>

If $u_n \to u$ and $v_n \to v$ in a pre-Hilbert space equipped with the norm $\lVert \cdot \rVert$, then $\langle u_n, v_n \rangle \to \langle u, v \rangle$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We bound $\lvert \langle u_n, v_n \rangle - \langle u, v \rangle \rvert = \lvert \langle u_n - u, v_n \rangle + \langle u, v_n - v \rangle \rvert \le \lvert \langle u_n - u, v_n \rangle \rvert + \lvert \langle u, v_n - v \rangle \rvert$. By Cauchy–Schwarz, this is $\le \lVert u_n - u \rVert \cdot \lVert v_n \rVert + \lVert u \rVert \cdot \lVert v_n - v \rVert$. Since $v_n \to v$, $\lVert v_n \rVert$ is bounded, so both terms go to $0$. $\square$</p>
  </details>
</div>

### Hilbert Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 145</span><span class="math-callout__name">(Hilbert Space)</span></p>

A **Hilbert space** is a pre-Hilbert space that is complete with respect to the norm $\lVert \cdot \rVert = \langle \cdot, \cdot \rangle^{1/2}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 146</span><span class="math-callout__name">($\mathbb{C}^n$)</span></p>

The space of $n$-tuples of complex numbers $\mathbb{C}^n$ with inner product $\langle \underline{z}, \underline{w} \rangle = \sum_{j=1}^{n} z_j \overline{w_j}$ is a (finite-dimensional) Hilbert space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 147</span><span class="math-callout__name">($\ell^2$)</span></p>

The space $\ell^2 = \lbrace \underline{a} : \sum_n \lvert a_n \rvert^2 < \infty \rbrace$ is a Hilbert space, where we define

$$\langle \underline{a}, \underline{b} \rangle = \sum_{k=1}^{\infty} a_k \overline{b_k}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 148</span><span class="math-callout__name">($L^2(E)$)</span></p>

Let $E \subset \mathbb{R}$ be measurable. Then $L^2(E)$, the space of measurable functions $f : E \to \mathbb{C}$ with $\int_E \lvert f \rvert^2 < \infty$, is a Hilbert space with inner product

$$\langle f, g \rangle = \int_E f \overline{g}.$$

</div>

The inner product only induces the $\ell_2$ norm (resp. $L^2$ norm). One might ask whether there is an inner product on $\ell^p$ or $L^p$ for $p \ne 2$ that produces the appropriate norm — the answer is **no**, as shown by the following result:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 149</span><span class="math-callout__name">(Parallelogram Law)</span></p>

Let $H$ be a pre-Hilbert space. Then for any $u, v \in H$, we have

$$\lVert u + v \rVert^2 + \lVert u - v \rVert^2 = 2\left( \lVert u \rVert^2 + \lVert v \rVert^2 \right).$$

In addition, if $H$ is a normed vector space satisfying this equality, then $H$ is a pre-Hilbert space.

</div>

One can verify by computation that there are always $u, v$ making this inequality fail if $p \ne 2$ for the $\ell^p$ and $L^p$ spaces.

### Orthogonality and Orthonormal Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 150</span><span class="math-callout__name">(Orthogonal, Orthonormal)</span></p>

Let $H$ be a pre-Hilbert space. Two elements $u, v \in H$ are **orthogonal** if $\langle u, v \rangle = 0$ (also denoted $u \perp v$), and a subset $\lbrace e_\lambda \rbrace_{\lambda \in \Lambda} \subset H$ is **orthonormal** if $\lVert e_\lambda \rVert = 1$ for all $\lambda \in \Lambda$ and for all $\lambda_1 \ne \lambda_2$, $\langle e_{\lambda_1}, e_{\lambda_2} \rangle = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 152</span><span class="math-callout__name">(Standard Basis Vectors)</span></p>

The set $\lbrace (0, 1), (1, 0) \rbrace$ is an orthonormal set in $\mathbb{C}^2$, and $\lbrace (0, 0, 1), (0, 1, 0) \rbrace$ is an orthonormal set in $\mathbb{C}^3$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 153</span><span class="math-callout__name">(Standard Basis in $\ell^2$)</span></p>

Let $\underline{e}\_n$ be the sequence which is $1$ in the $n$th entry and $0$ everywhere else. Then $\lbrace e_n \rbrace_{n \ge 1}$ is an orthonormal subset of $\ell^2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 154</span><span class="math-callout__name">(Fourier Basis in $L^2$)</span></p>

The functions $f_n(x) = \frac{1}{\sqrt{2\pi}} e^{inx}$ (as elements of $L^2([-\pi, \pi])$) form an orthonormal subset of $L^2([-\pi, \pi])$. This is because $\int_{-\pi}^{\pi} e^{imx} \overline{e^{inx}}\,dx = \int_{-\pi}^{\pi} e^{i(m-n)x}\,dx$ is zero unless $m = n$.

</div>

### Bessel's Inequality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 155</span><span class="math-callout__name">(Bessel's Inequality)</span></p>

Let $\lbrace e_n \rbrace$ be a countable (finite or countably infinite) orthonormal subset of a pre-Hilbert space $H$. Then for all $u \in H$,

$$\sum_n \lvert \langle u, e_n \rangle \rvert^2 \le \lVert u \rVert^2.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>For a finite collection $\lbrace e_n \rbrace_{n=1}^{N}$, note that $\left\lVert \sum_{n=1}^{N} \langle u, e_n \rangle e_n \right\rVert^2 = \sum_{n=1}^{N} \lvert \langle u, e_n \rangle \rvert^2$ by orthonormality. Also $\left\langle u, \sum_{n=1}^{N} \langle u, e_n \rangle e_n \right\rangle = \sum_{n=1}^{N} \lvert \langle u, e_n \rangle \rvert^2$.</p>
    <p>Then $0 \le \left\lVert u - \sum_{n=1}^{N} \langle u, e_n \rangle e_n \right\rVert^2 = \lVert u \rVert^2 - \sum_{n=1}^{N} \lvert \langle u, e_n \rangle \rvert^2$, which gives $\sum_{n=1}^{N} \lvert \langle u, e_n \rangle \rvert^2 \le \lVert u \rVert^2$. The infinite case follows by taking $N \to \infty$. $\square$</p>
  </details>
</div>

### Maximal Orthonormal Subsets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 156</span><span class="math-callout__name">(Maximal Orthonormal Subset)</span></p>

An orthonormal subset $\lbrace e_\lambda \rbrace_{\lambda \in \Lambda}$ of a pre-Hilbert space $H$ is **maximal** if the only vector $u \in H$ satisfying $\langle u, e_\lambda \rangle = 0$ for all $\lambda \in \Lambda$ is $u = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 157</span></p>

The $n$ standard basis vectors in $\mathbb{C}^n$ form a maximal orthonormal subset. (A non-example would be any proper subset of that set.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 158</span></p>

The example $\lbrace \underline{e}_n \rbrace$ of sequences from above is a maximal orthonormal subset of $\ell^2$.

</div>

A countably infinite maximal orthonormal subset basically serves the same purpose as an orthonormal basis does in linear algebra, but not every element will be a **finite** linear combination of the orthonormal subset elements (like was possible with a Hamel basis).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 159</span></p>

Every nontrivial pre-Hilbert space has a maximal orthonormal subset.

</div>

This can be proved using Zorn's lemma. A slightly weaker but more constructive result:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 160</span></p>

Every nontrivial **separable** pre-Hilbert space $H$ has a **countable** maximal orthonormal subset.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since $H$ is separable, let $\lbrace v_j \rbrace_{j=1}^{\infty}$ be a countable dense subset with $\lVert v_1 \rVert \ne 0$. We use the <strong>Gram–Schmidt process</strong>: set $e_1 = v_1 / \lVert v_1 \rVert$.</p>
    <p>Inductively, given orthonormal $\lbrace e_1, \dots, e_{m(k)} \rbrace$ spanning $\lbrace v_1, \dots, v_k \rbrace$: if $v_{k+1}$ is already in the span, do nothing. Otherwise, define $w_{k+1} = v_{k+1} - \sum_{j=1}^{m(k)} \langle v_{k+1}, e_j \rangle e_j$ (which is nonzero) and $e_{m(k+1)} = w_{k+1} / \lVert w_{k+1} \rVert$. Orthogonality of $e_{m(k+1)}$ with each $e_\ell$ ($1 \le \ell \le k$) follows from the construction.</p>
    <p>The collection $S = \bigcup_{n=1}^{\infty} \lbrace e_1, \dots, e_{m(n)} \rbrace$ is orthonormal. To show maximality: if $\langle u, e_\ell \rangle = 0$ for all $\ell$, then since $\lbrace v_j \rbrace$ are dense, there exists $v_{j(k)} \to u$. Each $v_{j(k)}$ is in the span of $\lbrace e_1, \dots, e_{m(j(k))} \rbrace$, so $\lVert v_{j(k)} \rVert^2 = \sum_{\ell=1}^{m(j(k))} \lvert \langle v_{j(k)}, e_\ell \rangle \rvert^2 = \sum_{\ell} \lvert \langle v_{j(k)} - u, e_\ell \rangle \rvert^2 \le \lVert v_{j(k)} - u \rVert^2 \to 0$ by Bessel's inequality. So $u = 0$, proving maximality. $\square$</p>
  </details>
</div>

## Orthonormal Bases and Fourier Series

### Orthonormal Bases

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 161</span><span class="math-callout__name">(Orthonormal Basis)</span></p>

Let $H$ be a Hilbert space. An **orthonormal basis** of $H$ is a countable maximal orthonormal subset $\lbrace e_n \rbrace$ of $H$.

</div>

Many examples we have encountered — $\mathbb{C}^n$, $\ell_2$, and $L^2$ — are indeed countable and thus have an orthonormal basis. The reason we call such sets bases, like in linear algebra, is that we can draw an analogy between the two definitions:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 162</span><span class="math-callout__name">(Fourier–Bessel Series)</span></p>

Let $\lbrace e_n \rbrace$ be an orthonormal basis in a Hilbert space $H$. Then for all $u \in H$, we have convergence of the **Fourier–Bessel series**

$$\lim_{m \to \infty} \sum_{n=1}^{m} \langle u, e_n \rangle e_n = \sum_{n=1}^{\infty} \langle u, e_n \rangle e_n = u.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>The sequence of partial sums $\lbrace \sum_{n=1}^{m} \langle u, e_n \rangle e_n \rbrace$ is Cauchy: since $\sum_n \lvert \langle u, e_n \rangle \rvert^2$ converges (bounded by $\lVert u \rVert^2$ via Bessel), for any $\varepsilon > 0$ there exists $M$ so that for $m > \ell \ge M$, $\left\lVert \sum_{n=1}^{m} \langle u, e_n \rangle e_n - \sum_{n=1}^{\ell} \langle u, e_n \rangle e_n \right\rVert^2 = \sum_{n=\ell+1}^{m} \lvert \langle u, e_n \rangle \rvert^2 < \varepsilon^2$.</p>
    <p>Since $H$ is complete, $u' = \lim_{m \to \infty} \sum_{n=1}^{m} \langle u, e_n \rangle e_n$ exists. By continuity of the inner product, $\langle u - u', e_\ell \rangle = \lim_{m \to \infty} \left\langle u - \sum_{n=1}^{m} \langle u, e_n \rangle e_n, e_\ell \right\rangle = \langle u, e_\ell \rangle - \langle u, e_\ell \rangle = 0$ for all $\ell$. By maximality, $u - u' = 0$, so $u = u'$. $\square$</p>
  </details>
</div>

Every separable Hilbert space has an orthonormal basis, and the converse is also true:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 163</span></p>

If a Hilbert space $H$ has an orthonormal basis, then $H$ is separable.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Suppose $\lbrace e_n \rbrace_n$ is an orthonormal basis. Define $S = \bigcup_{m \in \mathbb{N}} \left\lbrace \sum_{n=1}^{m} q_n e_n : q_1, \dots, q_m \in \mathbb{Q} + i\mathbb{Q} \right\rbrace$. This is a countable subset of $H$. By Theorem 162, every element $u$ can be expanded as $\sum_n \langle u, e_n \rangle e_n$, so for any $\varepsilon > 0$ we can take a sufficiently long partial sum and approximate each coefficient with a rational number, yielding an element of $S$ within distance $\varepsilon$ of $u$. Thus $S$ is dense. $\square$</p>
  </details>
</div>

### Parseval's Identity

We can now strengthen Bessel's inequality with our definition of orthonormal basis:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 164</span><span class="math-callout__name">(Parseval's Identity)</span></p>

Let $H$ be a Hilbert space, and let $\lbrace e_n \rbrace$ be a countable orthonormal basis of $H$. Then for all $u \in H$,

$$\sum_n \lvert \langle u, e_n \rangle \rvert^2 = \lVert u \rVert^2.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We know $u = \sum_n \langle u, e_n \rangle e_n$. By continuity of the inner product, $\lVert u \rVert^2 = \lim_{m \to \infty} \left\langle \sum_{n=1}^{m} \langle u, e_n \rangle e_n, \sum_{\ell=1}^{m} \langle u, e_\ell \rangle e_\ell \right\rangle = \lim_{m \to \infty} \sum_{n=1}^{m} \lvert \langle u, e_n \rangle \rvert^2$, where orthonormality picks out only the diagonal terms. $\square$</p>
  </details>
</div>

### Classification of Separable Hilbert Spaces

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 165</span></p>

If $H$ is an infinite-dimensional separable Hilbert space, then $H$ is isometrically isomorphic to $\ell^2$. In other words, there exists a bijective (bounded) linear operator $T : H \to \ell^2$ so that for all $u, v \in H$, $\lVert Tu \rVert_{\ell^2} = \lVert u \rVert_H$ and $\langle Tu, Tv \rangle_{\ell^2} = \langle u, v \rangle_H$.

</div>

<div class="accordion">
  <details>
    <summary>proof sketch</summary>
    <p>Since $H$ is separable, it has an orthonormal basis $\lbrace e_n \rbrace_{n \in \mathbb{N}}$, and by Theorem 162 every $u \in H$ satisfies $u = \sum_{n=1}^{\infty} \langle u, e_n \rangle e_n$. Define $Tu = \lbrace \langle u, e_n \rangle \rbrace_n$. This sequence is in $\ell^2$ by Parseval's identity. The map $T$ is linear in $u$, surjective (every $\ell^2$ sequence defines a Cauchy series $\sum c_n e_n$ in $H$), and injective (two expansions agreeing means the infinite sums are the same). $\square$</p>
  </details>
</div>

### Fourier Series

We now apply the Hilbert space theory in a concrete setting, focusing on **Fourier series**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 166</span></p>

The subset of functions $\left\lbrace \frac{e^{inx}}{\sqrt{2\pi}} \right\rbrace_{n \in \mathbb{Z}}$ is an orthonormal subset of $L^2([-\pi, \pi])$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We have $\langle e^{inx}, e^{imx} \rangle = \int_{-\pi}^{\pi} e^{inx} \overline{e^{imx}}\,dx = \int_{-\pi}^{\pi} e^{i(n-m)x}\,dx$, which equals $2\pi$ when $n = m$ (integrand is $1$) and $0$ when $n \ne m$ (the exponential is $2\pi$-periodic with zero average). Normalizing by $\sqrt{2\pi}$ gives the desired orthonormality. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 167</span><span class="math-callout__name">(Fourier Coefficient)</span></p>

For a function $f \in L^2([-\pi, \pi])$, the **Fourier coefficient** $\hat{f}(n)$ of $f$ is given by

$$\hat{f}(n) = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(t) e^{-int}\,dt,$$

and the $N$th **partial Fourier sum** is

$$S_N f(x) = \sum_{\lvert n \rvert \le N} \hat{f}(n) e^{inx} = \sum_{\lvert n \rvert \le N} \left\langle f, \frac{e^{inx}}{\sqrt{2\pi}} \right\rangle \frac{e^{inx}}{\sqrt{2\pi}}.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 168</span><span class="math-callout__name">(Fourier Series)</span></p>

The **Fourier series** of $f$ is the formal series $\sum_{n \in \mathbb{Z}} \hat{f}(n) e^{inx}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem 169</span></p>

Does the convergence (in $L^2$ norm) $\sum_{n=1}^{\infty} \hat{f} e^{inx} \to f$ hold for all $f \in L^2([-\pi, \pi])$? In other words, does

$$\lVert f - S_N f \rVert_2 = \left( \int_{-\pi}^{\pi} \lvert f(x) - S_N f(x) \rvert^2\,dx \right)^{1/2}$$

converge to $0$ as $N \to \infty$?

</div>

This is equivalent to asking whether $\left\lbrace \frac{e^{inx}}{\sqrt{2\pi}} \right\rbrace$ is a **maximal** orthonormal subset in $L^2([-\pi, \pi])$, i.e., whether $\hat{f}(n) = 0$ for all $n \in \mathbb{Z}$ implies $f = 0$. The answer turns out to be **yes**, but it takes some work to prove.

### The Dirichlet Kernel

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 170</span></p>

For all $f \in L^2([-\pi, \pi])$ and all $N \in \mathbb{Z}\_{\ge 0}$, we have $S_N f(x) = \int_{-\pi}^{\pi} D_N(x - t) f(t)\,dt$, where

$$D_N(x) = \begin{cases} \frac{2N+1}{2\pi} & x = 0, \\[6pt] \frac{\sin\left((N + \frac{1}{2})x\right)}{2\pi \sin \frac{x}{2}} & x \ne 0. \end{cases}$$

The function $D_N$ is called the **Dirichlet kernel**.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We have $S_N f(x) = \sum_{\lvert n \rvert \le N} \left( \frac{1}{2\pi} \int_{-\pi}^{\pi} f(t) e^{-int}\,dt \right) e^{inx} = \int_{-\pi}^{\pi} f(t) \left( \frac{1}{2\pi} \sum_{\lvert n \rvert \le N} e^{in(x-t)} \right) dt$. The sum $D_N(x) = \frac{1}{2\pi} \sum_{\lvert n \rvert \le N} e^{inx} = \frac{1}{2\pi} e^{-iNx} \sum_{n=0}^{2N} e^{inx}$ is a geometric series with ratio $e^{ix}$, evaluating to $\frac{1}{2\pi} e^{-iNx} \frac{1 - e^{i(2N+1)x}}{1 - e^{ix}}$ for $e^{ix} \ne 1$, which simplifies to $\frac{\sin((N + 1/2)x)}{2\pi \sin(x/2)}$ using $\sin x = \frac{e^{ix} - e^{-ix}}{2i}$. $\square$</p>
  </details>
</div>

### Cesaro–Fourier Means and the Fejér Kernel

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 171</span><span class="math-callout__name">(Cesaro–Fourier Mean)</span></p>

Let $f \in L^2([-\pi, \pi])$. The $N$th **Cesaro–Fourier mean** of $f$ is

$$\sigma_N f(x) = \frac{1}{N+1} \sum_{k=0}^{N} S_k f(x).$$

</div>

The strategy is to show $\lVert \sigma_N f - f \rVert_2 \to 0$ as $N \to \infty$, which then implies the desired convergence of Fourier series (because if all Fourier coefficients are zero, then $\sigma_N f = 0$ for all $N$, so $f = 0$).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 172</span></p>

We know from real analysis that the Cesaro means of a sequence of real numbers behave better than the original sequence, but we do not lose any information. In particular, sequences like $\lbrace 1, -1, 1, -1, \dots \rbrace$ do not converge, but their Cesaro means do.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 173</span><span class="math-callout__name">(Fejér Kernel)</span></p>

For all $f \in L^2([-\pi, \pi])$, we have

$$\sigma_N f(x) = \int_{-\pi}^{\pi} K_N(x - t) f(t)\,dt, \quad K_N(x) = \begin{cases} \frac{N+1}{2\pi} & x = 0, \\[6pt] \frac{1}{2\pi(N+1)} \left( \frac{\sin\left(\frac{N+1}{2} x\right)}{\sin \frac{x}{2}} \right)^2 & \text{otherwise}. \end{cases}$$

The function $K_N(x)$ is called the **Fejér kernel**. It has the following properties: **(1)** $K_N(x) \ge 0$ and $K_N(x) = K_N(-x)$ for all $x$, **(2)** $K_N$ is periodic with period $2\pi$, **(3)** $\int_{-\infty}^{\infty} K_N(t)\,dt = 1$, and **(4)** for any $\delta \in (0, \pi)$ and for all $\delta \le \lvert x \rvert \le \pi$, we have $\lvert K_N(x) \rvert \le \frac{1}{2\pi(N+1) \sin^2 \frac{\delta}{2}}$.

</div>

The Fejér kernel grows more and more concentrated at the origin as $N \to \infty$, but the area under the curve is always $1$ (like the Dirac delta function). The key difference from the Dirichlet kernel is that the Fejér kernel is **nonnegative**.

### Fejér's Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 174</span><span class="math-callout__name">(Fejér)</span></p>

Let $f \in C([-\pi, \pi])$ be $2\pi$-periodic (so $f(-\pi) = f(\pi)$). Then $\sigma_N f \to f$ uniformly on $[-\pi, \pi]$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Extend $f$ to all of $\mathbb{R}$ by periodicity. Since $f$ is $2\pi$-periodic and continuous, it is uniformly continuous and bounded ($\lVert f \rVert_\infty < \infty$).</p>
    <p>For any $\varepsilon > 0$, by uniform continuity there exists $\delta > 0$ so that $\lvert y - z \rvert < \delta$ implies $\lvert f(y) - f(z) \rvert < \varepsilon / 2$. Choose $M \in \mathbb{N}$ so that $\frac{2 \lVert f \rVert_\infty}{(N+1) \sin^2(\delta/2)} < \varepsilon / 2$ for all $N \ge M$.</p>
    <p>Using $\int_{-\pi}^{\pi} K_N(t)\,dt = 1$ and $K_N \ge 0$:</p>
    $$\lvert \sigma_N f(x) - f(x) \rvert = \left\lvert \int_{-\pi}^{\pi} K_N(t)(f(x-t) - f(x))\,dt \right\rvert \le \int_{-\pi}^{\pi} K_N(t) \lvert f(x-t) - f(x) \rvert\,dt.$$
    <p>Split into $\lvert t \rvert \le \delta$ (where $\lvert f(x-t) - f(x) \rvert < \varepsilon/2$) and $\delta \le \lvert t \rvert \le \pi$ (where $K_N(t) \le \frac{1}{2\pi(N+1)\sin^2(\delta/2)}$ and $\lvert f(x-t) - f(x) \rvert \le 2\lVert f \rVert_\infty$). Bounding each integral by the total $K_N$-mass gives $< \varepsilon/2 + \varepsilon/2 = \varepsilon$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 175</span></p>

The same proof works if instead of $K_N(x) \ge 0$ we only know $\sup_N \int_{-\pi}^{\pi} \lvert K_N(x) \rvert\,dx < \infty$. But for the Dirichlet kernel, $\int_{-\pi}^{\pi} \lvert D_N(x) \rvert\,dx \sim \log N$, so this condition is not satisfied — having "almost all" of the properties is not enough.

</div>

### Convergence of Cesaro Means in $L^2$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 176</span></p>

For all $f \in L^2([-\pi, \pi])$, we have $\lVert \sigma_N f \rVert_2 \le \lVert f \rVert_2$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>For $2\pi$-periodic $f \in C([-\pi, \pi])$: $\lVert \sigma_N f \rVert_2^2 = \int_{-\pi}^{\pi} \lvert \sigma_N f(x) \rvert^2\,dx$. Write $\sigma_N f(x) = \int_{-\pi}^{\pi} K_N(x - t) f(t)\,dt$ and expand the square. By Fubini's theorem and Cauchy–Schwarz:</p>
    $$\lVert \sigma_N f \rVert_2^2 \le \int_{-\pi}^{\pi} \int_{-\pi}^{\pi} K_N(s) K_N(t) \lVert f(\cdot - s) \rVert_2 \lVert f(\cdot - t) \rVert_2\,ds\,dt = \lVert f \rVert_2^2,$$
    <p>using that $\lVert f(\cdot - s) \rVert_2 = \lVert f \rVert_2$ by periodicity and $\int K_N = 1$. For general $f \in L^2$, approximate by $2\pi$-periodic continuous $f_n$ with $\lVert f_n - f \rVert_2 \to 0$, then $\lVert \sigma_N f \rVert_2 = \lim_{n \to \infty} \lVert \sigma_N f_n \rVert_2 \le \lim_{n \to \infty} \lVert f_n \rVert_2 = \lVert f \rVert_2$. $\square$</p>
  </details>
</div>

### Convergence of Fourier Series in $L^2$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 177</span></p>

For all $f \in L^2$, $\lVert \sigma_N f - f \rVert_2 \to 0$ as $N \to \infty$. Therefore, if $\hat{f}(n) = 0$ for all $n$, then $f = 0$ (since $\sigma_N f = 0$ for all $N$).

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let $f \in L^2([-\pi, \pi])$ and $\varepsilon > 0$. By density of $2\pi$-periodic continuous functions, there exists $g \in C([-\pi, \pi])$ with $\lVert f - g \rVert_2 < \varepsilon / 3$. By Fejér's theorem, $\sigma_N g \to g$ uniformly, so there exists $M$ with $\lvert \sigma_N g(x) - g(x) \rvert < \frac{\varepsilon}{3\sqrt{2\pi}}$ for all $N \ge M$ and all $x$, giving $\lVert \sigma_N g - g \rVert_2 < \varepsilon / 3$.</p>
    <p>By the triangle inequality: $\lVert \sigma_N f - f \rVert_2 \le \lVert \sigma_N(f - g) \rVert_2 + \lVert \sigma_N g - g \rVert_2 + \lVert g - f \rVert_2$. By Proposition 176, the first term is $\le \lVert f - g \rVert_2 < \varepsilon / 3$. Thus $\lVert \sigma_N f - f \rVert_2 < \varepsilon$. $\square$</p>
  </details>
</div>

This shows that the normalized exponentials $\left\lbrace \frac{e^{inx}}{\sqrt{2\pi}} \right\rbrace_{n \in \mathbb{Z}}$ form a maximal orthonormal subset, so the partial Fourier sums of $f$ converge to $f$ in $L^2$. As a deep result, **Carleson's theorem** tells us that in fact $S_N f(x) \to f(x)$ **almost everywhere** for all $f \in L^2$. It is also known that for all $1 < p < \infty$, $\lVert S_N f - f \rVert_p \to 0$, though this is false for $p = 1, \infty$.

## Minimizers, Orthogonal Decomposition, and the Riesz Representation Theorem

From here on, the course returns to general Hilbert space theory and concrete applications.

### Length Minimizers

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 178</span></p>

Let $C$ be a nonempty closed subset of a Hilbert space $H$ which is **convex**, meaning that for all $v_1, v_2 \in C$, we have $tv_1 + (1-t)v_2 \in C$ for all $t \in [0, 1]$. Then there exists a unique element $v \in C$ with $\lVert v \rVert = \inf_{u \in C} \lVert u \rVert$ (this is a length minimizer).

</div>

The convexity condition can alternatively be stated as "the line segment between any two elements of $C$ is contained in $C$." One example of such a set would be $v + W$ for some closed subspace $W$ of $C$ and some $v \in H$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 179</span></p>

The condition that $C$ is closed is required: for example, we can let $C$ be an open disk outside the origin, in which case the minimum norm is not achieved (because it is on the boundary). And convexity is also required — for example, otherwise we could take the complement of an open disk centered at the origin, in which case the minimum norm is achieved on the entire boundary.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let $d = \inf_{u \in C} \lVert u \rVert$. There exists a sequence $\lbrace u_n \rbrace$ in $C$ with $\lVert u_n \rVert \to d$. We claim $\lbrace u_n \rbrace$ is Cauchy. For $\varepsilon > 0$, choose $N$ so that for all $n \ge N$, $2\lVert u_n \rVert^2 < 2d^2 + \varepsilon^2/2$.</p>
    <p>By the parallelogram law: $\lVert u_m - u_n \rVert^2 = 2\lVert u_m \rVert^2 + 2\lVert u_n \rVert^2 - 4\left\lVert \frac{u_n + u_m}{2} \right\rVert^2$. By convexity, $\frac{u_n + u_m}{2} \in C$, so $\left\lVert \frac{u_n + u_m}{2} \right\rVert^2 \ge d^2$. Thus $\lVert u_m - u_n \rVert^2 < \varepsilon^2/2 + \varepsilon^2/2 = \varepsilon^2$ for $n, m \ge N$.</p>
    <p>Since $H$ is complete, $u_n \to v$ for some $v \in H$, and $v \in C$ because $C$ is closed. By continuity of the norm, $\lVert v \rVert = d$. For uniqueness: if $v, \overline{v}$ both have norm $d$, the parallelogram law gives $\lVert v - \overline{v} \rVert^2 \le 2d^2 + 2d^2 - 4d^2 = 0$, so $v = \overline{v}$. $\square$</p>
  </details>
</div>

### Orthogonal Decomposition

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 180</span></p>

Let $H$ be a Hilbert space, and let $W \subset H$ be a subspace. Then the **orthogonal complement**

$$W^\perp = \lbrace u \in H : \langle u, w \rangle = 0 \quad \forall w \in W \rbrace$$

is a closed linear subspace of $H$. Furthermore, if $W$ is closed, then $H = W \oplus W^\perp$; in other words, for all $u \in H$, we can write $u = w + w^\perp$ for some unique $w \in W$ and $w^\perp \in W^\perp$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>$W^\perp$ is a subspace: if $\langle u_1, w \rangle = 0$ and $\langle u_2, w \rangle = 0$ for all $w \in W$, then any linear combination is also orthogonal to all $w \in W$. Also $W \cap W^\perp = \lbrace 0 \rbrace$.</p>
    <p>$W^\perp$ is closed: if $u_n \in W^\perp$ with $u_n \to u$, then $\langle u, w \rangle = \lim \langle u_n, w \rangle = 0$ for all $w \in W$ by continuity of the inner product.</p>
    <p>For the decomposition when $W$ is closed: let $u \in H \setminus W$ and define $C = u + W = \lbrace u + w : w \in W \rbrace$. This set is closed and convex. By Theorem 178, there exists a unique $v \in C$ with $\lVert v \rVert = \inf_{c \in C} \lVert c \rVert$. Since $v \in C$, we have $u - v \in W$, so $u = (u - v) + v$.</p>
    <p>We show $v \in W^\perp$ via a variational argument: for any $w \in W$, $f(t) = \lVert v + tw \rVert^2$ has a minimum at $t = 0$, so $f'(0) = 2\operatorname{Re}\langle v, w \rangle = 0$. Repeating with $itw$ gives $\operatorname{Im}\langle v, w \rangle = 0$. Thus $\langle v, w \rangle = 0$ for all $w \in W$.</p>
    <p>Uniqueness: if $u = w_1 + w_1^\perp = w_2 + w_2^\perp$, then $w_1 - w_2 = w_2^\perp - w_1^\perp \in W \cap W^\perp = \lbrace 0 \rbrace$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 181</span></p>

If $W \subset H$ is a subspace, then $(W^\perp)^\perp$ is the closure $\overline{W}$ of $W$. In particular, if $W$ is closed, then $(W^\perp)^\perp = W$.

</div>

### Projections

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 182</span><span class="math-callout__name">(Projection)</span></p>

Let $P : H \to H$ be a bounded linear operator. Then $P$ is a **projection** if $P^2 = P$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 183</span></p>

Let $H$ be a Hilbert space, and let $W \subset H$ be a closed subspace. Then the map $\Pi_W : H \to H$ sending $v = w + w^\perp$ (for $w \in W$, $w^\perp \in W^\perp$) to $w$ is a projection operator.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Linearity: if $v_1 = w_1 + w_1^\perp$ and $v_2 = w_2 + w_2^\perp$, then $\lambda_1 v_1 + \lambda_2 v_2 = (\lambda_1 w_1 + \lambda_2 w_2) + (\lambda_1 w_1^\perp + \lambda_2 w_2^\perp)$ with each part in $W$ and $W^\perp$ respectively, so $\Pi_W(\lambda_1 v_1 + \lambda_2 v_2) = \lambda_1 \Pi_W(v_1) + \lambda_2 \Pi_W(v_2)$.</p>
    <p>Boundedness: $\lVert v \rVert^2 = \lVert w + w^\perp \rVert^2 = \lVert w \rVert^2 + \lVert w^\perp \rVert^2 \ge \lVert w \rVert^2$, so $\lVert \Pi_W(v) \rVert \le \lVert v \rVert$ and $\lVert \Pi_W \rVert \le 1$.</p>
    <p>Idempotency: $\Pi_W^2(v) = \Pi_W(\Pi_W(v)) = \Pi_W(w) = w = \Pi_W(v)$. $\square$</p>
  </details>
</div>

### Riesz Representation Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 184</span><span class="math-callout__name">(Riesz Representation Theorem)</span></p>

Let $H$ be a Hilbert space. Then for all $f \in H'$, there exists a unique $v \in H$ so that $f(u) = \langle u, v \rangle$ for all $u \in H$.

</div>

In other words, every element of the dual can be realized as an inner product with a fixed vector.

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Uniqueness: if $f(u) = \langle u, v \rangle = \langle u, \tilde{v} \rangle$ for all $u$, then $\langle u, v - \tilde{v} \rangle = 0$ for all $u$, so $v = \tilde{v}$.</p>
    <p>Existence: if $f = 0$, take $v = 0$. Otherwise, let $N = f^{-1}(\lbrace 0 \rbrace)$ be the nullspace of $f$. There exists $u_1 \in H$ with $f(u_1) \ne 0$; set $u_0 = u_1 / f(u_1)$ so that $f(u_0) = 1$. Define $C = \lbrace u \in H : f(u) = 1 \rbrace = f^{-1}(\lbrace 1 \rbrace)$, which is closed and convex. By Theorem 178, there exists $v_0 \in C$ with $\lVert v_0 \rVert = \inf_{u \in C} \lVert u \rVert$.</p>
    <p>We can check that $C = \lbrace v_0 + w : w \in N \rbrace$ and that $v_0 \in N^\perp$ (by the minimization argument from Theorem 180). Set $v = v_0 / \lVert v_0 \rVert^2$. For any $u \in H$, $u = (u - f(u)v_0) + f(u)v_0$ where $u - f(u)v_0 \in N$ and $v_0 \in N^\perp$, so</p>
    $$\langle u, v \rangle = \frac{1}{\lVert v_0 \rVert^2} \left[ \langle u - f(u)v_0, v_0 \rangle + f(u)\langle v_0, v_0 \rangle \right] = f(u). \quad \square$$
  </details>
</div>

## Adjoint Operators

Because we can identify dual spaces of Hilbert spaces with themselves (via the Riesz Representation Theorem), adjoint operators become regular operators on $H$ rather than maps between dual spaces.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 185</span></p>

Let $H$ be a Hilbert space, and let $A : H \to H$ be a bounded linear operator. Then there exists a unique bounded linear operator $A^* : H \to H$, known as the **adjoint** of $A$, satisfying

$$\langle Au, v \rangle = \langle u, A^* v \rangle$$

for all $u, v \in H$. In addition, we have that $\lVert A^* \rVert = \lVert A \rVert$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Uniqueness: if $\langle u, A_1^* v \rangle = \langle u, A_2^* v \rangle$ for all $u, v$, then $A_1^* v = A_2^* v$ for all $v$.</p>
    <p>Existence: for fixed $v \in H$, define $f_v(u) = \langle Au, v \rangle$. This is linear in $u$ and bounded: $\lvert f_v(u) \rvert = \lvert \langle Au, v \rangle \rvert \le \lVert A \rVert \cdot \lVert u \rVert \cdot \lVert v \rVert$, so $f_v \in H'$. By the Riesz Representation Theorem, there exists a unique $A^* v \in H$ with $\langle Au, v \rangle = f_v(u) = \langle u, A^* v \rangle$.</p>
    <p>Linearity of $A^*$: for all $u$, $\langle u, A^*(\lambda_1 v_1 + \lambda_2 v_2) \rangle = \langle Au, \lambda_1 v_1 + \lambda_2 v_2 \rangle = \overline{\lambda_1}\langle Au, v_1 \rangle + \overline{\lambda_2}\langle Au, v_2 \rangle = \langle u, \lambda_1 A^* v_1 + \lambda_2 A^* v_2 \rangle$.</p>
    <p>For $\lVert A^* \rVert = \lVert A \rVert$: first, $\lVert A^* v \rVert^2 = \langle A^* v, A^* v \rangle = \langle AA^* v, v \rangle \le \lVert A \rVert \cdot \lVert A^* v \rVert \cdot \lVert v \rVert$, giving $\lVert A^* v \rVert \le \lVert A \rVert \cdot \lVert v \rVert$, so $\lVert A^* \rVert \le \lVert A \rVert$. Also $(A^*)^* = A$ (since $\langle A^* u, v \rangle = \overline{\langle v, A^* u \rangle} = \overline{\langle Av, u \rangle} = \langle u, Av \rangle$), so $\lVert A \rVert = \lVert (A^*)^* \rVert \le \lVert A^* \rVert$. $\square$</p>
  </details>
</div>

### Examples of Adjoints

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 186</span><span class="math-callout__name">(Matrices on $\mathbb{C}^n$)</span></p>

If $H = \mathbb{C}^n$ and $A$ is represented by a matrix $(A_{ij})$, then the adjoint $A^*$ is the conjugate transpose: $(A^\ast v)\_i = \sum_{j=1}^{n} \overline{A_{ji}} v_j$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 187</span><span class="math-callout__name">(Operators on $\ell^2$)</span></p>

On $\ell^2$, an operator described by a double sequence $\lbrace A_{ij} \rbrace$ with $\sum_{i,j} \lvert A_{ij} \rvert^2 < \infty$ acts as $(A\underline{a})\_i = \sum_{j=1}^{\infty} A_{ij} a_j$. The adjoint is $(A^\ast \underline{b})\_i = \sum_{j=1}^{\infty} \overline{A_{ji}} b_j$ — again, the conjugate transpose.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 188</span><span class="math-callout__name">(Integral Operators on $L^2$)</span></p>

Let $K \in C([0, 1] \times [0, 1])$, and define $A : L^2([0, 1]) \to L^2([0, 1])$ via $Af(x) = \int_0^1 K(x, y) f(y)\,dy$. Then the adjoint is $A^* g(x) = \int_0^1 \overline{K(y, x)} g(y)\,dy$ — flipping the indices and taking a complex conjugate.

</div>

### Range-Nullspace Duality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 189</span></p>

Let $H$ be a Hilbert space, and let $A : H \to H$ be a bounded linear operator. Then

$$(\operatorname{Ran}(A))^\perp = \operatorname{Null}(A^*),$$

where $\operatorname{Ran}(A)$ is the range of $A$ and $\operatorname{Null}(A^\ast)$ is the nullspace of $A^*$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>$v \in \operatorname{Null}(A^*)$ iff $\langle u, A^* v \rangle = 0$ for all $u$, iff $\langle Au, v \rangle = 0$ for all $u$, iff $v \in (\operatorname{Ran}(A))^\perp$. $\square$</p>
  </details>
</div>

In particular, if $\operatorname{Ran}(A)$ is a closed subspace, then surjectivity of $A$ is equivalent to injectivity of $A^\ast$ (an infinite-dimensional version of rank-nullity).

## Compactness in Hilbert Spaces

### Compact Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 190</span><span class="math-callout__name">(Compact)</span></p>

Let $X$ be a metric space. A subset $K \subset X$ is **compact** if every sequence of elements in $K$ has a subsequence converging to an element of $K$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 191</span></p>

By the Pigeonhole Principle, all finite subsets are compact.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 192</span><span class="math-callout__name">(Heine–Borel)</span></p>

A subset $K \subset \mathbb{R}$ (also $\mathbb{R}^n$ and $\mathbb{C}^n$) is compact if and only if $K$ is closed and bounded.

</div>

This does not hold for arbitrary metric spaces or even Banach spaces, and in fact it is still not true for Hilbert spaces:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 193</span></p>

Let $H$ be an infinite-dimensional Hilbert space. Then the closed ball $F = \lbrace u \in H : \lVert u \rVert \le 1 \rbrace$ is closed and bounded but **not** compact.

This is because we can take a countably infinite orthonormal subset $\lbrace e_n \rbrace_{n=1}^{\infty}$ with all $e_n \in F$, but $\lVert e_n - e_k \rVert^2 = 2$ for all $n \ne k$, so no subsequence is Cauchy.

</div>

### Equi-Small Tails

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 194</span><span class="math-callout__name">(Equi-Small Tails)</span></p>

Let $H$ be a Hilbert space. A subset $K \subset H$ has **equi-small tails** with respect to a countable orthonormal subset $\lbrace e_n \rbrace$ if for all $\varepsilon > 0$, there is some $n \ge N$ so that for all $v \in K$, we have

$$\sum_{k > N} \lvert \langle v, e_k \rangle \rvert^2 < \varepsilon^2.$$

</div>

By Bessel's inequality, for any fixed $v$ the tail eventually becomes small. The equi-small tails condition requires this to happen **uniformly** for all $v \in K$ simultaneously.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 195</span></p>

Any finite set $K$ has equi-small tails with respect to any countable orthonormal subset (we can take the maximum of finitely many $N$s).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 196</span></p>

Let $H$ be a Hilbert space, and let $\lbrace v_n \rbrace_n$ be a convergent sequence with $v_n \to v$. If $\lbrace e_k \rbrace$ is a countable orthonormal subset, then $K = \lbrace v_n : n \in \mathbb{N} \rbrace \cup \lbrace v \rbrace$ is compact, and $K$ has equi-small tails with respect to $\lbrace e_k \rbrace$.

</div>

<div class="accordion">
  <details>
    <summary>proof (equi-small tails)</summary>
    <p>Let $\varepsilon > 0$. Choose $M$ so that $\lVert v_n - v \rVert < \varepsilon / 2$ for all $n \ge M$. Choose $N$ large enough so that $\sum_{k > N} \lvert \langle v, e_k \rangle \rvert^2 < \varepsilon^2 / 4$ and also $\sum_{k > N} \lvert \langle v_n, e_k \rangle \rvert^2 < \varepsilon^2 / 4$ for $1 \le n \le M - 1$ (finitely many terms).</p>
    <p>For $n \ge M$: $\left( \sum_{k > N} \lvert \langle v_n, e_k \rangle \rvert^2 \right)^{1/2} \le \left( \sum_{k > N} \lvert \langle v_n - v, e_k \rangle \rvert^2 \right)^{1/2} + \left( \sum_{k > N} \lvert \langle v, e_k \rangle \rvert^2 \right)^{1/2}$ by the $\ell^2$ triangle inequality. By Bessel's inequality, the first term is $\le \lVert v_n - v \rVert < \varepsilon / 2$, and the second is $< \varepsilon / 2$. $\square$</p>
  </details>
</div>

### Characterization of Compact Sets

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 197</span></p>

Let $H$ be a separable Hilbert space, and let $\lbrace e_k \rbrace_k$ be an orthonormal basis of $H$. Then a subset $K \subset H$ is compact if and only if $K$ is closed, bounded, and has equi-small tails with respect to $\lbrace e_k \rbrace$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Forward direction:</strong> If $K$ is compact, it is closed and bounded. For equi-small tails, suppose not: then there exists $\varepsilon_0 > 0$ and for each $N$ some $u_N \in K$ with $\sum_{k > N} \lvert \langle u_N, e_k \rangle \rvert^2 \ge \varepsilon_0^2$. By compactness, $\lbrace u_N \rbrace$ has a convergent subsequence $v_m \to v \in K$. But then $\lbrace v_n : n \in \mathbb{N} \rbrace \cup \lbrace v \rbrace$ does not have equi-small tails, contradicting Theorem 196.</p>
    <p><strong>Backward direction:</strong> Let $\lbrace u_n \rbrace$ be a sequence in $K$. Since $K$ is bounded, $\lVert u_n \rVert \le C$ for all $n$, so $\lvert \langle u_n, e_k \rangle \rvert \le C$. By Bolzano–Weierstrass (applied to $k = 1$), extract a subsequence where the first coefficient converges. Repeat for $k = 2, 3, \dots$ and use a diagonal argument to get a subsequence $v_\ell = u_{n_\ell}$ where $\langle v_\ell, e_k \rangle$ converges for every $k$.</p>
    <p>To show $\lbrace v_\ell \rbrace$ is Cauchy: by equi-small tails, choose $N$ with $\sum_{k > N} \lvert \langle v_\ell, e_k \rangle \rvert^2 < \varepsilon^2 / 16$ for all $\ell$. Then choose $M$ so that for $\ell, m \ge M$, $\sum_{k=1}^{N} \lvert \langle v_\ell, e_k \rangle - \langle v_m, e_k \rangle \rvert^2 < \varepsilon^2 / 4$. Using Parseval and the $\ell^2$ triangle inequality on the tail, $\lVert v_\ell - v_m \rVert < \varepsilon$. Since $K$ is closed, the limit is in $K$. $\square$</p>
  </details>
</div>

We can also characterize compact sets without reference to an orthonormal basis:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 199</span></p>

A subset $K \subset H$ is compact if and only if $K$ is closed, bounded, and for all $\varepsilon > 0$, there exists a finite-dimensional subspace $W \subset H$ so that for all $u \in K$, $\inf_{w \in W} \lVert u - w \rVert < \varepsilon$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 198</span><span class="math-callout__name">(Hilbert Cube)</span></p>

Let $K$ be the set (not subspace) of sequences $\lbrace a_k \rbrace_k$ in $\ell^2$ satisfying $\lvert a_k \rvert \le 2^{-k}$ — this set is known as the **Hilbert cube**, and it is compact.

</div>

## Classes of Operators

From here on, $H$ will be a Hilbert space, and we denote $\mathcal{B}(H, H)$ by $\mathcal{B}(H)$.

### Finite Rank Operators

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 201</span><span class="math-callout__name">(Finite Rank Operator)</span></p>

A bounded linear operator $T \in \mathcal{B}(H)$ is a **finite rank operator** if the range of $T$ is finite-dimensional. We denote this as $T \in \mathcal{R}(H)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 202</span></p>

If $H$ is finite-dimensional, every linear operator is of finite rank. For a more interesting example on $\ell^2$, for any positive integer $n$, the operator $Ta = \left\lbrace \frac{a_1}{1}, \frac{a_2}{2}, \dots, \frac{a_n}{n}, 0, \dots \right\rbrace$ is a finite rank operator (the image is spanned by the first $n$ standard basis vectors).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 203</span></p>

The set $\mathcal{R}(H)$ is a subspace of $\mathcal{B}(H)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 204</span></p>

An operator $T \in \mathcal{B}(H)$ is in $\mathcal{R}(H)$ if and only if there exists an orthonormal set $\lbrace e_k \rbrace_{k=1}^{L}$ and an array of constants $\lbrace c_{ij} \rbrace_{i,j=1}^{L} \subset \mathbb{C}$ such that

$$Tu = \sum_{i,j=1}^{L} c_{ij} \langle u, e_j \rangle e_i.$$

</div>

<div class="accordion">
  <details>
    <summary>proof sketch</summary>
    <p>The backward direction is clear: the range is in the span of $\lbrace e_1, \dots, e_L \rbrace$. For the forward direction, if $T$ is finite rank, find an orthonormal basis $\lbrace \overline{e}_k \rbrace_{k=1}^{N}$ of the range, so $Tu = \sum_k \langle Tu, \overline{e}_k \rangle \overline{e}_k = \sum_k \langle u, T^* \overline{e}_k \rangle \overline{e}_k$. Apply Gram–Schmidt to $\lbrace \overline{e}_1, \dots, \overline{e}_N, T^* \overline{e}_1, \dots, T^* \overline{e}_N \rbrace$ to get a joint orthonormal set $\lbrace e_1, \dots, e_L \rbrace$, and expand to obtain the desired $c_{ij}$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 205</span></p>

If $T \in \mathcal{R}(H)$, then $T^* \in \mathcal{R}(H)$, and for any $A, B \in \mathcal{B}(H)$, $ATB \in \mathcal{R}(H)$.

</div>

In other words, $\mathcal{R}(H)$ is a "star-closed, two-sided ideal in the space of bounded linear operators."

<div class="accordion">
  <details>
    <summary>proof sketch</summary>
    <p>If $Tu = \sum_{i,j} c_{ij} \langle u, e_j \rangle e_i$, then $T^* v = \sum_{i,j} \overline{c_{ji}} \langle v, e_i \rangle e_j$ — the coefficients are the conjugate transpose. For $ATB$: the range of $T$ is finite-dimensional, so the range of $ATB$ is at most that dimension. $\square$</p>
  </details>
</div>

### Compact Operators

The set $\mathcal{R}(H)$ is **not** closed under limits in operator norm:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 206</span></p>

Let $T_n : \ell^2 \to \ell^2$ be defined as $T_n a = \left\lbrace \frac{a_1}{1}, \dots, \frac{a_n}{n}, 0, \dots \right\rbrace$. Each $T_n$ is finite rank, and $\lVert T - T_n \rVert \le \frac{1}{n+1} \to 0$ where $Ta = \left\lbrace \frac{a_1}{1}, \frac{a_2}{2}, \frac{a_3}{3}, \dots \right\rbrace$, but $T$ is **not** of finite rank (since $T(ke_k) = e_k$ for each standard basis vector $e_k$).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 207</span><span class="math-callout__name">(Compact Operator)</span></p>

An operator $K \in \mathcal{B}(H)$ is a **compact operator** if $\overline{K(\lbrace u \in H : \lVert u \rVert \le 1 \rbrace)}$, the closure of the image of the unit ball under $K$, is compact.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 208</span></p>

Illustrative examples of compact operators include $K : \ell^2 \to \ell^2$ sending $a = (a_1, a_2, a_3, \dots)$ to $\left( \frac{a_1}{1}, \frac{a_2}{2}, \frac{a_3}{3}, \dots \right)$, as well as $T : L^2 \to L^2$ sending $f(x)$ to $\int_0^1 K(x, y) f(y)\,dy$ for some continuous function $K : [0, 1] \times [0, 1] \to \mathbb{R}$.

</div>

The integral operator is particularly important because it comes up in solutions to differential equations: for instance, if we take $K(x, y) = (x-1)y$ for $0 \le y \le x \le 1$ and $K(x, y) = x(y-1)$ for $0 \le x \le y \le 1$, then $u(x) = \int_0^1 K(x, y) f(y)\,dy$ satisfies $u'' = f$, $u(0) = u(1) = 0$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 209</span></p>

The identity operator $I$ on $\ell^2$ is **not** compact, because the closed unit ball is not compact in an infinite-dimensional Hilbert space. This argument shows that the identity is never compact for an infinite-dimensional Hilbert space.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 210</span></p>

Let $H$ be a separable Hilbert space. Then a bounded linear operator $T \in \mathcal{B}(H)$ is a compact operator if and only if there exists a sequence $\lbrace T_n \rbrace_n$ of finite rank operators such that $\lVert T - T_n \rVert \to 0$. (In other words, the set of compact operators is the closure $\overline{\mathcal{R}(H)}$.)

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Forward ($T$ compact $\Rightarrow$ limit of finite rank):</strong> Since $H$ is separable with orthonormal basis $\lbrace e_k \rbrace$, the image of the unit ball under $T$ is compact and thus has equi-small tails: for any $\varepsilon > 0$, there exists $N$ with $\sum_{k > N} \lvert \langle Tu, e_k \rangle \rvert^2 < \varepsilon^2$ for all $\lVert u \rVert \le 1$. Define $T_n u = \sum_{k=1}^{n} \langle Tu, e_k \rangle e_k$, which is finite rank. Then $\lVert T_n u - Tu \rVert^2 = \sum_{k > n} \lvert \langle Tu, e_k \rangle \rvert^2 < \varepsilon^2$ for $n \ge N$ and all $\lVert u \rVert \le 1$, giving $\lVert T_n - T \rVert \le \varepsilon$.</p>
    <p><strong>Backward (limit of finite rank $\Rightarrow$ $T$ compact):</strong> We use Theorem 199. The set $\overline{\lbrace Tu : \lVert u \rVert \le 1 \rbrace}$ is closed and bounded (by $\lVert T \rVert$). For any $\varepsilon > 0$, choose $N$ with $\lVert T - T_N \rVert < \varepsilon$ and let $W = \operatorname{Ran}(T_N)$. Then for $\lVert u \rVert \le 1$, $\lVert Tu - T_N u \rVert < \varepsilon$, so $\inf_{w \in W} \lVert Tu - w \rVert < \varepsilon$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 211</span></p>

Let $H$ be a separable Hilbert space, and let $K(H)$ be the set of compact operators on $H$. Then:

1. $K(H)$ is a closed subspace of $\mathcal{B}(H)$.
2. For any $T \in K(H)$, we also have $T^* \in K(H)$.
3. For any $T \in K(H)$ and $A, B \in \mathcal{B}(H)$, we have $ATB \in K(H)$.

</div>

In other words, the set of compact operators is also a star-closed, two-sided ideal in $\mathcal{B}(H)$.

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>(1) is clear because $K(H) = \overline{\mathcal{R}(H)}$. For (2), if $T \in K(H)$ then $T_n \to T$ for finite rank $T_n$, so $T_n^* \to T^*$ (since $\lVert T_n^* - T^* \rVert = \lVert T_n - T \rVert$), and each $T_n^*$ is finite rank. For (3), $AT_n B$ is finite rank for each $n$, and $\lVert AT_n B - ATB \rVert \le \lVert A \rVert \cdot \lVert T_n - T \rVert \cdot \lVert B \rVert \to 0$. $\square$</p>
  </details>
</div>

## Eigenvalues and the Spectrum

### Invertible Operators

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 212</span></p>

Let $T \in \mathcal{B}(H)$ be a bounded linear operator. If $\lVert T \rVert < 1$, then $I - T$ is invertible, and we can compute its inverse to be the absolutely summable series

$$(I - T)^{-1} = \sum_{n=0}^{\infty} T^n.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 213</span></p>

The space of invertible linear operators $GL(H) = \lbrace T \in \mathcal{B}(H) : T \text{ invertible} \rbrace$ is an open subset of $\mathcal{B}(H)$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let $T_0 \in GL(H)$. Any $T$ with $\lVert T - T_0 \rVert < \lVert T_0^{-1} \rVert^{-1}$ is invertible, because $\lVert T_0^{-1}(T - T_0) \rVert \le \lVert T_0^{-1} \rVert \cdot \lVert T - T_0 \rVert < 1$, so $I - T_0^{-1}(T - T_0)$ is invertible by Proposition 212, and thus $T = T_0(I - T_0^{-1}(T_0 - T))$ is invertible. $\square$</p>
  </details>
</div>

### The Spectrum

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 214</span><span class="math-callout__name">(Resolvent Set and Spectrum)</span></p>

Let $A \in \mathcal{B}(H)$ be a bounded linear operator. The **resolvent set** of $A$, denoted $\operatorname{Res}(A)$, is the set $\lbrace \lambda \in \mathbb{C} : A - \lambda I \in GL(H) \rbrace$, and the **spectrum** of $A$, denoted $\operatorname{Spec}(A)$, is the complement $\mathbb{C} \setminus \operatorname{Res}(A)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 215</span></p>

Let $A : \mathbb{C}^2 \to \mathbb{C}^2$ be the matrix $\begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix}$. Then $A - \lambda$ is not invertible exactly when $\lambda = \lambda_1$ or $\lambda = \lambda_2$, so $\operatorname{Spec}(A) = \lbrace \lambda_1, \lambda_2 \rbrace$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 216</span><span class="math-callout__name">(Eigenvalue and Eigenvector)</span></p>

If $A \in \mathcal{B}(H)$ and $A - \lambda$ is not injective, then there exists some $u \in H \setminus \lbrace 0 \rbrace$ with $Au = \lambda u$, and we call $\lambda$ an **eigenvalue** of $A$ and $u$ the associated **eigenvector**.

</div>

In finite dimensions, the spectrum consists entirely of eigenvalues. In infinite dimensions, there can be elements of the spectrum that are not eigenvalues:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 217</span></p>

For the compact operator $T : \ell^2 \to \ell^2$ sending $a \mapsto \left( \frac{a_1}{1}, \frac{a_2}{2}, \frac{a_3}{3}, \dots \right)$, the $n$th basis vector $e_n$ is an eigenvector with eigenvalue $\frac{1}{n}$. But $0$ is also in the spectrum despite not being an eigenvalue: $T$ is injective, but not surjective (its inverse would map $a \mapsto (a_1, 2a_2, 3a_3, \dots)$, which is unbounded).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 218</span></p>

Let $T : L^2([0, 1]) \to L^2([0, 1])$ be defined via $Tf(x) = xf(x)$. Then $T$ has no eigenvalues, but the spectrum is $\operatorname{Spec}(T) = [0, 1]$.

</div>

### Properties of the Spectrum

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 219</span></p>

Let $A \in \mathcal{B}(H)$. Then $\operatorname{Spec}(A)$ is a closed subset of $\mathbb{C}$, and $\operatorname{Spec}(A) \subset \lbrace \lambda \in \mathbb{C} : \lvert \lambda \rvert \le \lVert A \rVert \rbrace$.

</div>

In particular, the spectrum is a **compact** subset of the complex numbers.

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>The resolvent set $\operatorname{Res}(A)$ is open (as a preimage of the open set $GL(H)$ under the continuous map $\lambda \mapsto A - \lambda I$), so $\operatorname{Spec}(A)$ is closed.</p>
    <p>If $\lvert \lambda \rvert > \lVert A \rVert$, then $A - \lambda = -\lambda(I - \frac{1}{\lambda}A)$ is invertible by Proposition 212 (since $\lVert \frac{1}{\lambda}A \rVert < 1$), so $\lambda \in \operatorname{Res}(A)$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 220</span></p>

The spectrum is always nonempty. If it were empty, then for all $u, v \in H$, $f(\lambda) = \langle (A - \lambda)^{-1}u, v \rangle$ would be a continuous, complex-differentiable function on $\mathbb{C}$ with $f(\lambda) \to 0$ as $\lvert \lambda \rvert \to \infty$. By Liouville's theorem, $f \equiv 0$, implying $(A - \lambda)^{-1} = 0$, a contradiction.

</div>

### Self-Adjoint Operators

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 221</span></p>

If we have a self-adjoint operator $A \in \mathcal{B}(H)$, meaning that $A = A^*$, then $\langle Au, u \rangle$ is real for all $u$, and $\lVert A \rVert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle \rvert$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>$\overline{\langle Au, u \rangle} = \langle u, Au \rangle = \langle A^* u, u \rangle = \langle Au, u \rangle$, so $\langle Au, u \rangle$ is real.</p>
    <p>Let $a = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle \rvert$. By Cauchy–Schwarz, $\lvert \langle Au, u \rangle \rvert \le \lVert Au \rVert \le \lVert A \rVert$, so $a \le \lVert A \rVert$. For the reverse: if $Au \ne 0$ with $\lVert u \rVert = 1$, set $v = Au / \lVert Au \rVert$. Then $\lVert Au \rVert = \langle Au, v \rangle = \operatorname{Re}\langle Au, v \rangle$. Using the polarization identity:</p>
    $$\lVert Au \rVert = \frac{1}{4}\left( \langle A(u+v), u+v \rangle - \langle A(u-v), u-v \rangle \right) \le \frac{a}{4}\left( \lVert u+v \rVert^2 + \lVert u-v \rVert^2 \right) = a$$
    <p>by the parallelogram law (since $\lVert u \rVert = \lVert v \rVert = 1$). Thus $\lVert A \rVert \le a$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 222</span></p>

In quantum mechanics, observables (like position, momentum, and so on) are modeled by self-adjoint (unbounded) operators, and the point is that all things measured in nature (the associated eigenvalues) are real. So there are applications of all of our discussions here to physics!

</div>

### Spectrum of Self-Adjoint Operators

Recall that the **resolvent** of an operator $A$ is the set of complex numbers $\lambda$ such that $A - \lambda$ is an element of $GL(H)$ (i.e., $A - \lambda$ is bijective with a bounded inverse), and the **spectrum** $\operatorname{Spec}(A)$ is the complement of the resolvent in $\mathbb{C}$. While the spectrum is just the set of eigenvalues for matrices in a finite-dimensional vector space, there is a more subtle distinction to be made: $\lambda \in \operatorname{Spec}(A)$ is an **eigenvalue** if there is some vector $u$ with $(A - \lambda)u = 0$, so $\lambda$ is in the spectrum because $A - \lambda$ is not injective. But there are other reasons why $\lambda$ might be in the spectrum as well, for instance if the image is not closed.

We have already shown that the spectrum is closed and contained within the ball of radius $\lVert A \rVert$, meaning that it is compact. We also showed that a self-adjoint bounded linear operator $A$ always has $\langle Au, u \rangle$ real, and that $\lVert A \rVert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle \rvert$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 223</span></p>

Let $A = A^* \in \mathcal{B}(H)$ be a self-adjoint operator. Then the spectrum $\operatorname{Spec}(A) \subset [-\lVert A \rVert, \lVert A \rVert]$ is contained within a line segment on the real line, and at least one of $\pm \lVert A \rVert$ is in $\operatorname{Spec}(A)$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We first show that the spectrum is contained in $[-\lVert A \rVert, \lVert A \rVert]$. We already know $\operatorname{Spec}(A) \subset \lbrace \lvert \lambda \rvert \le \lVert A \rVert \rbrace$, so we just need to show $\operatorname{Spec}(A) \subset \mathbb{R}$ (i.e., any complex number with a nonzero imaginary part is in the resolvent). Write $\lambda = s + it$ for $s, t$ real and $t \ne 0$, so that $A - \lambda = \tilde{A} - it$ where $\tilde{A} = A - s$ is another self-adjoint bounded linear operator.</p>
    <p>It suffices to show that $\tilde{A} - it$ is <strong>bijective</strong>. Since $\langle Au, u \rangle$ is real, $\operatorname{Im}(\langle (A - it)u, u \rangle) = \operatorname{Im}(\langle -itu, u \rangle) = -t\lVert u \rVert^2$, so $(A - it)u = 0$ only if $u = 0$. Therefore $A - it$ is injective.</p>
    <p>For surjectivity, notice that $(A - it)^* = A + it$ is also injective by the same argument, so $\operatorname{Range}(A - it)^\perp = \operatorname{Null}((A - it)^*) = \lbrace 0 \rbrace$, and thus $\overline{\operatorname{Range}(A - it)} = \lbrace 0 \rbrace^\perp = H$.</p>
    <p>It remains to show that the range is closed. If $(A - it)u_n \to v$, then from $\lvert t \rvert \cdot \lVert u_n - u_m \rVert^2 = \lvert \operatorname{Im}(\langle (A - it)(u_n - u_m), u_n - u_m \rangle) \rvert \le \lVert (A - it)(u_n) - (A - it)(u_m) \rVert \cdot \lVert u_n - u_m \rVert$, we get $\lVert u_n - u_m \rVert \le \frac{1}{\lvert t \rvert} \lVert (A - it)u_n - (A - it)u_m \rVert$. Since $\lbrace (A - it)u_n \rbrace$ converges, it is Cauchy, so $\lbrace u_n \rbrace$ is also Cauchy. By completeness, $u_n \to u$ and $(A - it)u = v$. $\square$</p>
    <p>For the second property, since $\lVert A \rVert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle \rvert$, there must be a sequence of unit vectors $\lbrace u_n \rbrace$ such that $\lvert \langle Au_n, u_n \rangle \rvert \to \lVert A \rVert$. Since each term is real, there must be a subsequence with $\langle Au_n, u_n \rangle$ converging to $\lVert A \rVert$ or to $-\lVert A \rVert$, which means $\langle (A \mp \lVert A \rVert)u_n, u_n \rangle \to 0$. We claim that $A \mp \lVert A \rVert$ is not invertible: assume for the sake of contradiction that it were invertible, then $1 = \lVert u_n \rVert = \lVert (A \pm \lVert A \rVert)^{-1} (A \mp \lVert A \rVert) u_n \rVert \le \lVert (A \pm \lVert A \rVert)^{-1} \rVert \cdot \lVert (A \mp \lVert A \rVert) u_n \rVert$, but the right-hand side converges to $0$, a contradiction. So $A \mp \lVert A \rVert$ is not bijective, and thus one of $\pm \lVert A \rVert$ must be in the spectrum. $\square$</p>
  </details>
</div>

We can in fact strengthen this bound even more:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 224</span></p>

If $A = A^* \in \mathcal{B}(H)$ is a self-adjoint bounded linear operator, and we define $a_- = \inf_{\lVert u \rVert = 1} \langle Au, u \rangle$ and $a_+ = \sup_{\lVert u \rVert = 1} \langle Au, u \rangle$, then $a_\pm$ are both contained in $\operatorname{Spec}(A)$, which is contained within $[a_-, a_+]$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since $-\lVert A \rVert \le \langle Au, u \rangle \le \lVert A \rVert$ for all $u$, we must have $-\lVert A \rVert \le a_- \le a_+ \le \lVert A \rVert$. By the definition of $a_-, a_+$, there exist two sequences $\lbrace u_n^\pm \rbrace$ of unit vectors so that $\langle Au_n^\pm, u_n^\pm \rangle \to a_\pm$. The same argument as in Theorem 223 shows that $\langle (A - a_\pm) u_n^\mp, u_n^\mp \rangle \to 0$, which implies $a_+$ and $a_-$ are both in the spectrum.</p>
    <p>To show the spectrum is contained within $[a_-, a_+]$, let $b = \frac{a_+ + a_-}{2}$ be their midpoint and $B = A - bI$. Since $b$ is a real number, $B$ is also a bounded self-adjoint operator, so by Theorem 223, $\operatorname{Spec}(B) \subset [-\lVert B \rVert, \lVert B \rVert]$. This means $\operatorname{Spec}(A) \subset [-\lVert B \rVert + b, \lVert B \rVert + b]$. We have $\lVert B \rVert = \sup_{\lVert u \rVert = 1} \lvert \langle Bu, u \rangle \rvert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle - \frac{a_+ + a_-}{2} \rvert$. Since $\langle Au, u \rangle$ always lies in $[a_-, a_+]$, this supremum equals $\frac{a_+ - a_-}{2}$, so $\operatorname{Spec}(A) \subset [-\lVert B \rVert + b, \lVert B \rVert + b] = [a_-, a_+]$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 225</span></p>

Let $A = A^* \in \mathcal{B}(H)$ be a self-adjoint linear operator. Then $\langle Au, u \rangle \ge 0$ for all $u$ if and only if $\operatorname{Spec}(A) \subset [0, \infty)$.

</div>

## Spectral Theory for Self-Adjoint Compact Operators

We'll now move on to the spectral theory for self-adjoint **compact** operators: the short answer is that we essentially see just the eigenvalues, with the exception of zero being a possible accumulation point. And in particular, the spectrum will be countable, and this should make sense because compact operators are the limit of finite rank operators — we don't expect to end up with wildly different behavior in the limit.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 226</span></p>

Let $A \in \mathcal{B}(H)$ be a bounded linear operator. We denote $E_\lambda$ to be the nullspace of $A - \lambda$, or equivalently the set of eigenvectors $\lbrace u \in H : (A - \lambda)u = 0 \rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 227</span></p>

Suppose $A^* = A \in \mathcal{B}(H)$ is a compact self-adjoint operator. Then we have the following:

1. If $\lambda \ne 0$ is an eigenvalue of $A$, then $\lambda \in \mathbb{R}$ and $\dim E_\lambda$ is finite.
2. If $\lambda_1 \ne \lambda_2$ are eigenvalues of $A$, then $E_{\lambda_1}$ and $E_{\lambda_2}$ are orthogonal to each other (every element in $E_{\lambda_1}$ is orthogonal to every element in $E_{\lambda_2}$).
3. The set of nonzero eigenvalues of $A$ is either finite or countably infinite, and if it is countably infinite and given by a sequence $\lbrace \lambda_n \rbrace_n$, then $\lvert \lambda_n \rvert \to 0$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>(1)</strong> Let $\lambda$ be a nonzero eigenvalue. Suppose for contradiction that $E_\lambda$ is infinite-dimensional. By Gram–Schmidt, there exists a countable orthonormal collection $\lbrace u_n \rbrace_n$ of elements of $E_\lambda$. Since $A$ is compact, $\lbrace Au_n \rbrace_n$ must have a convergent subsequence $\lbrace Au_{n_j} \rbrace_j$. But $\lVert Au_{n_j} - Au_{n_k} \rVert^2 = \lVert \lambda u_{n_j} - \lambda u_{n_k} \rVert^2 = \lvert \lambda \rvert^2 \lVert u_{n_j} - u_{n_k} \rVert^2 = 2\lvert \lambda \rvert^2$, so the distance does not go to $0$, a contradiction. Thus $E_\lambda$ is finite-dimensional. To show $\lambda$ is real, pick a unit eigenvector $u$ with $Au = \lambda u$; then $\lambda = \lambda \langle u, u \rangle = \langle \lambda u, u \rangle = \langle Au, u \rangle$, which is real.</p>
    <p><strong>(2)</strong> Suppose $\lambda_1 \ne \lambda_2$ and $u_1 \in E_{\lambda_1}, u_2 \in E_{\lambda_2}$. Then $\lambda_1 \langle u_1, u_2 \rangle = \langle \lambda_1 u_1, u_2 \rangle = \langle Au_1, u_2 \rangle = \langle u_1, Au_2 \rangle = \langle u_1, \lambda_2 u_2 \rangle = \lambda_2 \langle u_1, u_2 \rangle$ (no complex conjugate because eigenvalues are real). Therefore $(\lambda_1 - \lambda_2) \langle u_1, u_2 \rangle = 0$, and since $\lambda_1 - \lambda_2 \ne 0$, we get $\langle u_1, u_2 \rangle = 0$.</p>
    <p><strong>(3)</strong> Let $\Lambda = \lbrace \lambda \ne 0 : \lambda \text{ eigenvalue of } A \rbrace$. We show $\Lambda$ is either finite or countably infinite by showing that if $\lbrace \lambda_n \rbrace_n$ is a sequence of distinct eigenvalues, then $\lambda_n \to 0$. The set $\Lambda_N = \lbrace \lambda \in \Lambda : \lvert \lambda \rvert \ge \frac{1}{N} \rbrace$ is finite for each $N$ (otherwise we could take infinitely many distinct elements in $\Lambda_N$, and that can't converge to $0$), so $\Lambda = \bigcup_{N \in \mathbb{N}} \Lambda_N$ is a countable union of finite sets and thus countable.</p>
    <p>To prove the claim, let $\lbrace u_n \rbrace_n$ be unit eigenvectors of our eigenvalues $\lambda_n$. Then $\lvert \lambda_n \rvert = \lVert \lambda_n u_n \rVert = \lVert Au_n \rVert$, so it suffices to show $\lVert Au_n \rVert \to 0$. Suppose $\lVert Au_n \rVert$ does not converge to $0$. Then there exist $\varepsilon_0 > 0$ and a subsequence $\lbrace Au_{n_j} \rbrace$ with $\lVert Au_{n_j} \rVert \ge \varepsilon_0$. Because $A$ is compact, there is a further convergent subsequence $e_k = u_{n_{j_k}}$ with $\lbrace Ae_k \rbrace_k$ converging. Since $e_k$ and $e_\ell$ are eigenvectors of distinct eigenvalues, they are orthogonal, and therefore $Ae_k$ and $Ae_\ell$ are also orthogonal. If $f = \lim_{k \to \infty} Ae_k$, then $\lVert f \rVert \ge \varepsilon_0$, and $\varepsilon_0^2 \le \lVert f \rVert^2 = \langle f, f \rangle = \lim_{k \to \infty} \langle e_k, Af \rangle$. But $\langle e_k, Af \rangle$ gives the Fourier coefficients of $Af$, and by Bessel's inequality, the sum of their squares is at most $\lVert Af \rVert^2 < \infty$, which contradicts the limit of the Fourier coefficients being at least $\varepsilon_0^2$. $\square$</p>
  </details>
</div>

### The Fredholm Alternative

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 228</span><span class="math-callout__name">(Fredholm Alternative)</span></p>

Let $A = A^* \in \mathcal{B}(H)$ be a self-adjoint compact operator, and let $\lambda \in \mathbb{R} \setminus \lbrace 0 \rbrace$. Then $\operatorname{Range}(A - \lambda)$ is closed, meaning that

$$\operatorname{Range}(A - \lambda) = (\operatorname{Range}(A - \lambda)^\perp)^\perp = \operatorname{Null}(A - \lambda)^\perp.$$

Thus, either $A - \lambda$ is bijective, or the nullspace of $A - \lambda$ (the eigenspace corresponding to $\lambda$) is nontrivial and finite-dimensional.

</div>

This result tells us when we can solve the equality $(A - \lambda)u = f$: we can do so if and only if $f$ is orthogonal to the nullspace of $A - \lambda$. The finite-dimensional part comes from Theorem 227 — it is useful because we can check orthogonality by taking a finite basis of $A - \lambda$'s nullspace.

A further consequence is that because the spectrum of a self-adjoint $A$ is a subset of the reals, we have $\operatorname{Spec}(A) \setminus \lbrace 0 \rbrace = \lbrace \text{eigenvalues of } A \rbrace$, since the nonzero spectrum only fails to be bijective because we have an eigenvector. And because the eigenvalue set is finite or countably infinite, it can only be countably infinite if those eigenvalues converge to zero.

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We need to show that the range of $A - \lambda$ is closed if $\lambda \ne 0$. Suppose we have elements $(A - \lambda)u_n$ that converge to $f \in H$; we need to show $f$ is also in the range of $A - \lambda$.</p>
    <p>The $u_n$'s will not necessarily converge, but we can extract a relevant subsequence. Let $v_n = \Pi_{\operatorname{Null}(A - \lambda)^\perp} u_n$ be the projection onto the orthogonal complement of $\operatorname{Null}(A - \lambda)$. Then $(A - \lambda)u_n = (A - \lambda)v_n$, and the $v_n$ all live in $\operatorname{Null}(A - \lambda)^\perp$.</p>
    <p>We claim $\lbrace v_n \rbrace$ is bounded. Suppose otherwise: there exists a subsequence $\lbrace v_{n_j} \rbrace$ with $\lVert v_{n_j} \rVert \to \infty$. Then $(A - \lambda) \frac{v_{n_j}}{\lVert v_{n_j} \rVert} \to 0$. Because $A$ is compact, there is a further subsequence $\lbrace v_{n_k} \rbrace$ such that $\lbrace A \frac{v_{n_k}}{\lVert v_{n_k} \rVert} \rbrace$ converges. Since $\frac{v_{n_k}}{\lVert v_{n_k} \rVert} = \frac{1}{\lambda}(A \frac{v_{n_k}}{\lVert v_{n_k} \rVert} - (A - \lambda)\frac{v_{n_k}}{\lVert v_{n_k} \rVert})$, both terms on the right converge, so $\frac{v_{n_k}}{\lVert v_{n_k} \rVert}$ converges to some $v$ with $\lVert v \rVert = 1$ and $v \in \operatorname{Null}(A - \lambda)^\perp$. But $(A - \lambda)v = 0$, so $v$ is in both the nullspace and its orthogonal complement, giving $v = 0$ — a contradiction.</p>
    <p>So $\lbrace v_n \rbrace$ is bounded. Because $A$ is compact, $\lbrace (A - \lambda)v_n \rbrace$ is also bounded, and there is a subsequence $\lbrace v_{n_j} \rbrace$ so that $\lbrace Av_{n_j} \rbrace$ converges. Since $v_{n_j} = \frac{1}{\lambda}(Av_{n_j} - (A - \lambda)v_{n_j})$, the subsequence $v_{n_j} \to v$ for some $v \in H$. Then $f = \lim (A - \lambda)v_{n_j} = (A - \lambda)v$, so $f$ is in the range. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 229</span></p>

We did not actually use the fact that $A$ is a self-adjoint operator in the above argument — the fact that $\operatorname{Range}(A - \lambda)$ is closed is still true if $A$ is just a compact operator, but the consequences of that fact only apply for self-adjoint operators.

</div>

### Eigenvalue Existence and the Maximum Principle

We've shown previously that one of $\pm \lVert A \rVert$ must be in the spectrum, and that gives us:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 230</span></p>

Let $A = A^*$ be a nontrivial compact self-adjoint operator. Then $A$ has a nontrivial eigenvalue $\lambda_1$ with $\lvert \lambda_1 \rvert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle \rvert = \lvert \langle Au_1, u_1 \rangle \rvert$, where $u_1$ is a normalized eigenvector (with $\lVert u_1 \rVert = 1$) satisfying $Au_1 = \lambda_1 u_1$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since at least one of $\pm \lVert A \rVert$ is in $\operatorname{Spec}(A)$ (and $\lVert A \rVert \ne 0$ because we have a nontrivial operator), at least one of them will be an eigenvalue by the Fredholm alternative, and we call this $\lambda_1$. The equation for $\lambda_1$ follows from $\lVert A \rVert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle \rvert$, and $\lvert \langle Au_1, u_1 \rangle \rvert$ equals this because being an eigenvalue implies we have an eigenvector. $\square$</p>
  </details>
</div>

It turns out we can keep building up eigenvalues in this way, because eigenvectors of different eigenvalues are orthogonal. This leads us to constructing an orthonormal basis.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 231</span><span class="math-callout__name">(Maximum Principle)</span></p>

Let $A = A^*$ be a self-adjoint compact operator. Then the nonzero eigenvalues of $A$ can be ordered as $\lvert \lambda_1 \rvert \ge \lvert \lambda_2 \rvert \ge \cdots$ (including with multiplicity), such that we have pairwise orthonormal eigenfunctions $\lbrace u_k \rbrace$ for $\lambda_k$, satisfying

$$\lvert \lambda_j \rvert = \sup_{\substack{\lVert u \rVert = 1 \\ u \in \operatorname{Span}(u_1, \dots, u_{j-1})^\perp}} \lvert \langle Au, u \rangle \rvert = \lvert \langle Au_j, u_j \rangle \rvert.$$

Furthermore, we have $\lvert \lambda_j \rvert \to 0$ as $j \to \infty$ if the sequence of nonzero eigenvalues does not terminate.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We construct eigenvalues inductively. First, we find $\lambda_1$ and $u_1$ using Theorem 230. For the inductive step, suppose we have found $\lambda_1, \dots, \lambda_n$ along with orthonormal eigenvectors $u_1, \dots, u_n$ satisfying the equation for $\lvert \lambda_j \rvert$ in the maximum principle.</p>
    <p>If $A$ is finite-rank, then $Au = \sum_{k=1}^n \lambda_k \langle u, u_k \rangle u_k$ and we've found all eigenvalues. Otherwise, define the operator $A_n u = Au - \sum_{k=1}^n \lambda_k \langle u, u_k \rangle u_k$. One can check that $A_n$ is self-adjoint and compact. If $u \in \operatorname{Span}\lbrace u_1, \dots, u_n \rbrace$, then $A_n u = 0$; if $u \in \operatorname{Span}\lbrace u_1, \dots, u_n \rbrace^\perp$, then $A_n u = Au$. Therefore $\operatorname{Range}(A_n) \subset \operatorname{Span}\lbrace u_1, \dots, u_n \rbrace^\perp$.</p>
    <p>Any nonzero eigenvalue of $A_n$ is also a nonzero eigenvalue of $A$. Applying Theorem 230 to $A_n$ gives $\lambda_{n+1}$ with unit eigenvector $u_{n+1}$ orthogonal to $\operatorname{Span}\lbrace u_1, \dots, u_n \rbrace$. We have $\lvert \lambda_{n+1} \rvert = \sup_{\lVert u \rVert = 1} \lvert \langle A_n u, u \rangle \rvert = \sup_{\substack{\lVert u \rVert = 1 \\ u \in \operatorname{Span}(u_1, \dots, u_n)^\perp}} \lvert \langle Au, u \rangle \rvert$. The ordering $\lvert \lambda_{n+1} \rvert \le \lvert \lambda_n \rvert$ is preserved because we are taking the supremum over a smaller set. $\square$</p>
  </details>
</div>

### The Spectral Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 232</span><span class="math-callout__name">(Spectral Theorem)</span></p>

Let $A = A^*$ be a self-adjoint compact operator on a separable Hilbert space $H$. If $\lvert \lambda_1 \rvert \ge \lvert \lambda_2 \rvert \ge \cdots$ are the nonzero eigenvalues of $A$, counted with multiplicity and with corresponding orthonormal eigenvectors $\lbrace u_k \rbrace_k$, then $\lbrace u_k \rbrace_k$ is an orthonormal basis for $\operatorname{Range}(A)$ and also of $\overline{\operatorname{Range}(A)}$, and there is an orthonormal basis $\lbrace f_j \rbrace_j$ of $\operatorname{Null}(A)$ so that $\lbrace u_k \rbrace_k \cup \lbrace f_j \rbrace_j$ form an orthonormal basis of $H$.

</div>

In other words, we can find an orthonormal basis consisting entirely of eigenvectors for our self-adjoint compact operator (since the nullspace corresponds to eigenvectors of eigenvalue $0$).

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>If $A$ is finite rank, the process in the maximum principle terminates at some $n$ with $Au = \sum_{k=1}^n \lambda_k \langle u, u_k \rangle u_k$ and $\operatorname{Range}(A) \subset \operatorname{Span}\lbrace u_1, \dots, u_k \rbrace$, so $\lbrace u_k \rbrace$ form an orthonormal basis for $\operatorname{Range}(A)$ and $\overline{\operatorname{Range}(A)}$.</p>
    <p>Otherwise, we have countably infinitely many nonzero eigenvalues $\lbrace \lambda_k \rbrace_{k=1}^\infty$ with $\lvert \lambda_k \rvert \to 0$. The $u_k$'s form an orthonormal subset of $\operatorname{Range}(A)$. To show it is a basis, we must show that if $f \in \operatorname{Range}(A)$ and $\langle f, u_k \rangle = 0$ for all $k$, then $f = 0$.</p>
    <p>Write $f = Au$ for some $u \in H$. Since $A$ is self-adjoint, $\lambda_k \langle u, u_k \rangle = \langle u, \lambda_k u_k \rangle = \langle u, Au_k \rangle = \langle Au, u_k \rangle = 0$. Therefore $u$ is orthogonal to all $u_k$, so by the maximum principle, $\lVert f \rVert = \lVert Au \rVert = \lVert (A - \sum_{k=1}^n \lambda_k \langle u, u_k \rangle u_k) u \rVert = \lVert A_n u \rVert \le \lvert \lambda_{n+1} \rvert \cdot \lVert u \rVert$. Taking $n \to \infty$ gives $\lVert f \rVert = 0$.</p>
    <p>To show $\lbrace u_k \rbrace$ is also a basis for $\overline{\operatorname{Range}(A)}$, note that $\overline{\operatorname{Range}(A)} \subset \overline{\operatorname{Span}\lbrace u_k \rbrace_k} = \lbrace \sum_k c_k u_k : \sum_k \lvert c_k \rvert^2 < \infty \rbrace$. To complete the orthonormal basis of $H$, we use $\overline{\operatorname{Range}(A)} = (\operatorname{Range}(A)^\perp)^\perp = (\operatorname{Null}(A))^\perp$, so we just need an orthonormal basis of $\operatorname{Null}(A)$, which exists because $H$ is separable. $\square$</p>
  </details>
</div>

## Application to the Dirichlet Problem

In this last section, we apply functional analysis to the Dirichlet problem (understanding ODEs with conditions at the boundary). In an introductory differential equations class, we often state initial conditions by specifying the value and derivatives of a function at a given point, but what we are doing here is slightly different:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem 233</span><span class="math-callout__name">(Dirichlet Problem)</span></p>

Let $V \in C([0, 1])$ be a continuous, real-valued function. We wish to solve the differential equation

$$\begin{cases} -u''(x) + V(x)u(x) = f(x) & \forall x \in [0, 1], \\ u(0) = u(1) = 0. \end{cases}$$

</div>

We can think of this as specifying a "force" $f \in C([0, 1])$ and seeing whether there exists a unique solution $u \in C^2([0, 1])$ to the differential equation above. It turns out the answer is always yes when $V \ge 0$, and that's what we'll show.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 234</span></p>

Let $V \ge 0$. If $f \in C([0, 1])$ and $u_1, u_2 \in C^2([0, 1])$ both satisfy the Dirichlet problem, then $u_1 = u_2$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>If $u = u_1 - u_2$, then $u \in C^2([0, 1])$ and $-u''(x) + V(x)u(x) = 0$ for all $x \in [0, 1]$, with $u(0) = u(1) = 0$. Multiplying by $\overline{u(x)}$ and integrating:</p>
    $$0 = \int_0^1 (-u''(x) + V(x)u(x))\overline{u(x)}\,dx = -\int_0^1 u''(x)\overline{u(x)}\,dx + \int_0^1 V(x)\lvert u(x) \rvert^2\,dx.$$
    <p>Integration by parts on the first term gives $0 = -u'(x)\overline{u(x)}\big\rvert_0^1 + \int_0^1 \lvert u'(x) \rvert^2\,dx + \int_0^1 V(x)\lvert u(x) \rvert^2\,dx$. The boundary term vanishes by the Dirichlet conditions, leaving $0 = \int_0^1 \lvert u'(x) \rvert^2\,dx + \int_0^1 V(x)\lvert u(x) \rvert^2\,dx$. Since $V \ge 0$, both terms are nonnegative, so $u'(x) = 0$ everywhere (since $u$ is continuous), which with $u(0) = 0$ gives $u = 0$, i.e., $u_1 = u_2$. $\square$</p>
  </details>
</div>

Showing existence is more involved, and we start with the easier case $V = 0$. It turns out we can write down the solution explicitly using a self-adjoint compact operator:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 235</span></p>

Define the continuous function $K(x, y) \in C([0, 1] \times [0, 1])$ via

$$K(x, y) = \begin{cases} (x - 1)y & 0 \le y \le x \le 1, \\ (y - 1)x & 0 \le x \le y \le 1. \end{cases}$$

Then if $Af(x) = \int_0^1 K(x, y)f(y)\,dy$, then $A \in \mathcal{B}(L^2([0, 1]))$ is a compact self-adjoint operator, and $Af$ solves the Dirichlet problem with $V = 0$ (meaning $u = Af$ is the unique solution to $-u''(x) = f(x)$, $u(0) = u(1) = 0$).

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let $C = \sup_{[0,1] \times [0,1]} \lvert K(x, y) \rvert$, which is finite because $K$ is continuous. By Cauchy–Schwarz, $\lvert Af(x) \rvert = \lvert \int_0^1 K(x, y)f(y)\,dy \rvert \le C \lVert f \rVert_2$, so $A$ is bounded. Also, $\lvert Af(x) - Af(z) \rvert \le \sup_{y \in [0,1]} \lvert K(x, y) - K(z, y) \rvert \cdot \lVert f \rVert_2$, so by the Arzelà–Ascoli theorem, $A$ is a compact operator on $L^2([0, 1])$.</p>
    <p>Furthermore, $A$ is self-adjoint because for any $f, g \in C([0, 1])$, $\langle Af, g \rangle_2 = \int_0^1 \int_0^1 K(x, y)f(y)\,dy\,\overline{g(x)}\,dx = \int_0^1 f(y) \int_0^1 K(y, x)g(x)\,dx\,dy = \langle f, Ag \rangle$ by Fubini's theorem and using $K(x, y) = K(y, x)$. Since $C([0, 1])$ is dense in $L^2([0, 1])$, self-adjointness extends by density.</p>
    <p>To verify $Af$ solves the Dirichlet problem with $V = 0$, we compute $u(x) = Af(x) = (x - 1)\int_0^x yf(y)\,dy + x\int_x^1 (y - 1)f(y)\,dy$, and by the fundamental theorem of calculus, $u \in C^2([0, 1])$ with $-u'' = f$. Uniqueness follows from Theorem 234. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 236</span></p>

We have $\operatorname{Null}(A) = \lbrace 0 \rbrace$, and the orthonormal eigenvectors for $A$ are given by

$$u_k(x) = \sqrt{2}\sin(k\pi x), \quad k \in \mathbb{N},$$

with associated eigenvalues $\lambda_k = \frac{1}{k^2 \pi^2}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 237</span></p>

As a corollary, the spectral theorem then tells us that $\lbrace \sqrt{2}\sin(k\pi x) \rbrace$ gives us an orthonormal basis of $L^2([0, 1])$, which is a result we can also prove by rescaling our Fourier series result from $L^2([-\pi, \pi])$.

</div>

<div class="accordion">
  <details>
    <summary>proof of Theorem 236</summary>
    <p>First, the nullspace of $A$ is trivial: if $u$ is a polynomial in $[0, 1]$ with $f = -u''$ and $u(0) = u(1) = 0$, then $Af$ is the <strong>unique</strong> solution, so $Af = u$, meaning every polynomial vanishing at $\lbrace 0, 1 \rbrace$ is in the range of $A$. Since such polynomials are dense in $L^2$ (by the Weierstrass approximation theorem), the range of $A$ is dense, so $\overline{\operatorname{Range}(A)} = \operatorname{Null}(A)^\perp$, giving $\operatorname{Null}(A) = \lbrace 0 \rbrace$.</p>
    <p>For eigenvectors: suppose $\lambda \ne 0$ and $Au = \lambda u$. Since $Af$ is always continuous and twice differentiable, $u = \frac{1}{\lambda}Au$ is also twice continuously differentiable, and $u = A(\frac{u}{\lambda}) \implies -u'' = \frac{1}{\lambda}u$ with $u(0) = u(1) = 0$. This is a simple harmonic oscillator: solutions are $u(x) = A\sin(\frac{1}{\sqrt{\lambda}}x) + B\cos(\frac{1}{\sqrt{\lambda}}x)$. The condition $u(0) = 0$ gives $B = 0$, and $u(1) = 0$ requires $\frac{1}{\sqrt{\lambda}} = n\pi$ for some $n \in \mathbb{N}$, so $u(x) = A\sin(k\pi x)$ and $A = \sqrt{2}$ by normalization. $\square$</p>
  </details>
</div>

### Solving the General Dirichlet Problem

Since we now have a basis in which the operator $A$ is diagonal, we can construct $A^{1/2}$ by essentially taking the square roots of all the eigenvalues (so that $A^{1/2}A^{1/2} = A$).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 238</span></p>

Let $f \in L^2([0, 1])$, and suppose that $f(x) = \sum_{k=1}^\infty c_k \sqrt{2}\sin(k\pi x)$, where $c_k = \int_0^1 f(x)\sqrt{2}\sin(k\pi x)\,dx$. Then we define the linear operator $A^{1/2}$ via

$$A^{1/2}f(x) = \sum_{k=1}^\infty \frac{1}{k\pi}\,c_k\,\sqrt{2}\sin(k\pi x).$$

</div>

Here, the reason for $\frac{1}{k\pi}$ in the definition is that we have a $\frac{1}{k^2\pi^2}$ eigenvalue that we want to produce after two iterations of $A^{1/2}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 239</span></p>

The operator $A^{1/2}$ is a compact, self-adjoint operator on $L^2([0, 1])$, and $(A^{1/2})^2 = A$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Boundedness:</strong> $\lVert A^{1/2}f \rVert_2^2 = \sum_{k=1}^\infty \frac{\lvert c_k \rvert^2}{k^2\pi^2} \le \frac{1}{\pi^2}\sum_{k=1}^\infty \lvert c_k \rvert^2 = \frac{1}{\pi^2}\lVert f \rVert_2^2$ by Parseval's identity.</p>
    <p><strong>Self-adjointness:</strong> $\langle A^{1/2}f, g \rangle = \sum_{k=1}^\infty \frac{c_k}{k\pi}\overline{d_k} = \sum_{k=1}^\infty c_k \frac{\overline{d_k}}{k\pi} = \langle f, A^{1/2}g \rangle$.</p>
    <p><strong>$(A^{1/2})^2 = A$:</strong> $A^{1/2}(A^{1/2}f) = A^{1/2}\sum_{k=1}^\infty \frac{c_k}{k\pi}\sqrt{2}\sin(k\pi x) = \sum_{k=1}^\infty \frac{c_k}{k^2\pi^2}\sqrt{2}\sin(k\pi x)$. Since each term is an eigenfunction of $A$, this equals $\sum_{k=1}^\infty c_k A(\sqrt{2}\sin(k\pi x)) = A\sum_{k=1}^\infty c_k \sqrt{2}\sin(k\pi x) = Af$.</p>
    <p><strong>Compactness:</strong> The image of the unit ball $\lbrace A^{1/2}f : \lVert f \rVert_2 \le 1 \rbrace$ has equi-small tails: for any $\varepsilon > 0$, pick $N$ with $\frac{1}{N^2} < \varepsilon^2$, then $\sum_{k > N} \lvert \langle A^{1/2}f, \sqrt{2}\sin(k\pi x) \rangle \rvert^2 = \sum_{k > N} \frac{\lvert c_k \rvert^2}{k^2\pi^2} \le \frac{1}{N^2}\lVert f \rVert_2^2 \le \frac{1}{N^2} < \varepsilon^2$. $\square$</p>
  </details>
</div>

Now that we have $A^{1/2}$, we'll put it to good use:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 240</span></p>

Let $V \in C([0, 1])$ be a real-valued function, and define the multiplication operator $m_V f(x) = V(x)f(x)$. Then $m_V$ is a bounded linear operator and self-adjoint.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 241</span></p>

Let $V \in C([0, 1])$ be a real-valued function. Then $T = A^{1/2}m_V A^{1/2}$ is a self-adjoint compact operator on $L^2([0, 1])$, and $T$ is a bounded operator from $L^2([0, 1])$ to $C([0, 1])$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since $m_V$ and $A^{1/2}$ are compact operators, so is their product. It's self-adjoint because $A^{1/2}$ and $m_V$ are self-adjoint and $(A^{1/2}m_V A^{1/2})^* = A^{1/2}m_V A^{1/2}$ (remembering to reverse the order of operators).</p>
    <p>For the remaining step, we show $A^{1/2}$ is bounded from $L^2([0, 1])$ to $C([0, 1])$: since $A^{1/2}f(x) = \sum_{k=1}^\infty \frac{c_k}{k\pi}\sqrt{2}\sin(k\pi x)$ and $\lvert \frac{c_k}{k\pi}\sqrt{2}\sin(k\pi x) \rvert \le \frac{\lvert c_k \rvert}{k}$, we have $\sum_{k=1}^\infty \frac{\lvert c_k \rvert}{k} \le (\sum_k \frac{1}{k^2})^{1/2}(\sum_k \lvert c_k \rvert^2)^{1/2} < \sqrt{\frac{\pi^2}{6}}\lVert f \rVert_2$ by Cauchy–Schwarz. So the series converges uniformly by the Weierstrass M-test, and $A^{1/2}f \in C([0, 1])$. Furthermore, each term evaluates to $0$ at $x = 0, 1$, so $A^{1/2}f(0) = A^{1/2}f(1) = 0$. $\square$</p>
  </details>
</div>

We now have all the ingredients to solve our problem:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 242</span></p>

Let $V \in C([0, 1])$ be a nonnegative real-valued continuous function, and let $f \in C([0, 1])$. Then there exists a (unique) twice-differentiable solution $u \in C^2([0, 1])$ such that

$$\begin{cases} -u'' + Vu = f & \forall x \in [0, 1], \\ u(0) = u(1) = 0. \end{cases}$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since $-u'' + Vu = f$ can be rewritten as $-u'' = f - Vu$, thinking of $f - Vu$ as a fixed function of $x$ gives $u = A(f - Vu)$, i.e., $(I + AV)u = Af$. To get self-adjoint operators, let $u = A^{1/2}v$; then $(I + A^{1/2}m_V A^{1/2})v = A^{1/2}f$ (using $A^{1/2}A^{1/2} = A$).</p>
    <p>We know $A^{1/2}m_V A^{1/2}$ is a self-adjoint compact operator, so by the Fredholm alternative, $I + A^{1/2}m_V A^{1/2}$ has an inverse if and only if the nullspace is trivial. Suppose $(I + A^{1/2}m_V A^{1/2})g = 0$ for some $g \in L^2$. Then $0 = \langle (I + A^{1/2}m_V A^{1/2})g, g \rangle = \lVert g \rVert_2^2 + \langle m_V A^{1/2}g, A^{1/2}g \rangle = \lVert g \rVert_2^2 + \int_0^1 V\lvert A^{1/2}g \rvert^2\,dx$. Since $V \ge 0$, both terms are nonnegative, so $\lVert g \rVert_2^2 \ge 0$ can only be zero if $g = 0$. Thus $I + A^{1/2}m_V A^{1/2}$ is indeed invertible.</p>
    <p>To finish, define $v = (I + A^{1/2}m_V A^{1/2})^{-1}A^{1/2}f$ and $u = A^{1/2}v$. Then $u + A(Vu) = A^{1/2}v + A^{1/2}(A^{1/2}m_V A^{1/2})v = A^{1/2}(I + (A^{1/2}m_V A^{1/2}))v$, and plugging in $v$ gives $u + AVu = A^{1/2}A^{1/2}f = Af$. Taking two derivatives: $u'' - Vu = -f$, i.e., $-u'' + Vu = f$. By Theorem 241, $u = A^{1/2}v$ satisfies the Dirichlet boundary conditions, and thus we've solved the Dirichlet problem. $\square$</p>
  </details>
</div>
