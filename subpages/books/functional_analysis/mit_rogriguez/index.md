---
layout: default
title: Introduction to Functional Analysis"
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
  .math-figure {
    margin: 1.4rem auto;
    text-align: center;
  }
  .math-figure svg,
  .math-figure img {
    max-width: 100%;
    height: auto;
  }
  .math-figure figcaption {
    font-size: 0.88rem;
    color: var(--text-muted, #555);
    margin-top: 0.4rem;
    font-style: italic;
  }
  .math-figure-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: flex-start;
    gap: 1.2rem;
  }
  .math-figure-row > div {
    flex: 1 1 280px;
    text-align: center;
  }
  .math-figure-row .panel-label {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
  }
</style>

# Introduction to Functional Analysis

**Table of Contents**
- TOC
{:toc}

## Introduction

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Infinitely many independent variables)</span></p>

In many math courses — calculus, linear algebra — the methods we learn help us solve **equations with finitely many variables**. We might want to find the minimum of a function whose inputs are in $\mathbb{R}^n$, or solve a system of linear equations. But when we encounter ODEs, PDEs, minimization, and other problems where the set of independent variables is no longer finite-dimensional, we need new tools.

**Functional analysis** helps us solve problems where the vector space is no longer finite-dimensional, and this situation arises very naturally in many concrete problems.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Shortest Curve)</span></p>

Finding the shortest possible curve between two points amounts to specifying a **functional** — the input is a function. We need infinitely many real numbers to specify a real-valued function $f : [0, 1] \to \mathbb{R}$.

</div>

<figure class="math-figure">
  <svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" width="640" role="img" aria-labelledby="functional-curve-title functional-curve-desc">
    <title id="functional-curve-title">Many curves join two points; assigning each its length is a functional</title>
    <desc id="functional-curve-desc">Two fixed points A and B are joined by several candidate curves drawn in gray and one straight green curve marked as the shortest path. A caption notes that mapping each curve to its length is a functional on an infinite-dimensional space.</desc>
    <rect x="20" y="20" width="600" height="260" rx="10" fill="#fbfcff" stroke="#e3e7f0" />

    <g fill="none" stroke="#b7c0d0" stroke-width="2">
      <path d="M 120,215 Q 350,70 520,120" />
      <path d="M 120,215 C 250,150 420,265 520,120" />
      <path d="M 120,215 Q 300,258 520,120" />
      <path d="M 120,215 C 220,255 330,85 520,120" />
    </g>

    <path d="M 120,215 L 520,120" fill="none" stroke="#1b8f5a" stroke-width="3.5" />

    <g fill="#2c3550">
      <circle cx="120" cy="215" r="6" />
      <circle cx="520" cy="120" r="6" />
    </g>
    <text x="100" y="240" font-size="15" fill="#2c3550">A</text>
    <text x="530" y="112" font-size="15" fill="#2c3550">B</text>

    <text x="288" y="58" font-size="13" fill="#7a8394">candidate paths γ</text>
    <text x="360" y="182" font-size="13" fill="#1b8f5a" font-weight="600">shortest path γ*</text>
  </svg>
  <figcaption>Each path joining $A$ and $B$ is a function $\gamma : [0, 1] \to \mathbb{R}^2$, and assigning it a number — its length $L(\gamma) = \int_0^1 \lvert \gamma'(t) \rvert \, dt$ — is a <strong>functional</strong>. Because specifying an entire curve takes infinitely many real numbers, minimizing $L$ is an optimization over an infinite-dimensional space: exactly the setting functional analysis is built to handle.</figcaption>
</figure>

## Normed Spaces

We use a lot of terminology from real analysis and linear algebra. Let us redefine a few key terms.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Space)</span></p>

A **vector space** $V$ over a field $\mathbb{K}$ (which we take to be either $\mathbb{R}$ or $\mathbb{C}$) is a set of vectors equipped with an addition $+ : V \times V \to V$ and scalar multiplication $\cdot : \mathbb{K} \times V \to V$, satisfying the usual axioms: commutativity, associativity, identity and inverse of addition, identity of multiplication, and distributivity.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Basic Vector Spaces)</span></p>

$\mathbb{R}^n$ and $\mathbb{C}^n$ are vector spaces, and so is $C([0, 1])$, the space of continuous functions $[0, 1] \to \mathbb{C}$. The latter is a vector space because the sum of two continuous functions is continuous, and so is a scalar multiple of a continuous function.

But $C([0, 1])$ is a completely different **size** from $\mathbb{R}^n$ or $\mathbb{C}^n$: it is infinite-dimensional.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Finite- and Infinite-Dimensional)</span></p>

A vector space $V$ is **finite-dimensional** if every linearly independent set is finite. In other words, for all sets $E \subseteq V$ such that

$$\sum_{i=1}^{N} a_i v_i = 0 \implies a_1 = a_2 = \cdots = a_N = 0 \quad \forall\, v_1, \dots, v_N \in E,$$

$E$ has finite cardinality. $V$ is **infinite-dimensional** if it is not finite-dimensional.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($C([0,1])$ is Infinite-Dimensional)</span></p>

The set 

$$E = \lbrace f_n(x) = x^n : n \in \mathbb{Z}_{\ge 0} \rbrace$$

is linearly independent but contains infinitely many elements, so $C([0, 1])$ is infinite-dimensional.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Many theorems fail in infinite-dim – we need generalization)</span></p>

Facts like the Heine–Borel theorem for $\mathbb{R}^n$ become false in infinite-dimensional spaces, so we need to develop more machinery. In analysis, we need a notion of "how close things are." In metric spaces we use metrics; here we define a distance on our vector spaces via norms.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Norm, Normed Space)</span></p>

A **norm** on a vector space $V$ is a function $\lVert \cdot \rVert : V \to [0, \infty)$ satisfying:

1. (Definiteness) $\lVert v \rVert = 0 \iff v = 0$.
2. (Homogeneity) $\lVert \lambda v \rVert = \lvert \lambda \rvert \lVert v \rVert \quad \forall v \in V \quad \forall\lambda \in \mathbb{K}$.
3. (Triangle inequality) $\lVert v_1 + v_2 \rVert \le \lVert v_1 \rVert + \lVert v_2 \rVert \quad \forall v_1, v_2 \in V$.

A vector space equipped with a norm is called a **normed space**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Seminorm)</span></p>

A **seminorm** is a function $\lVert \cdot \rVert : V \to [0, \infty)$ satisfying (2) and (3) but not necessarily (1). A vector space equipped with a norm is called a **normed space**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Metric)</span></p>

A norm induces a metric, so we can think of our normed space as a metric space. Recall that a **metric** $d : X \times X \to [0, \infty)$ satisfies: 
* (1) $d(x, y) = 0 \iff x = y$,
* (2) $d(x, y) = d(y, x)$,
* (3) $d(x, y) + d(y, z) \ge d(x, z)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Norm Induces a Metric)</span></p>

Let $\lVert \cdot \rVert$ be a norm on a vector space $V$. Then

$$d(v, w) := \lVert v - w \rVert$$

defines a metric on $V$, called the **metric induced by the norm**.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

* **Property (1)** of the norm implies property (1) of metrics, because 

  $$d(v, w) = \lVert v - w \rVert = 0 \iff v - w = 0 \iff v = w.$$

* For **property (2)** of the metric, note that 
  
  $$\lVert v - w \rVert = \lVert (-1)(w - v) \rVert = \lvert -1 \rvert \cdot \lVert w - v \rVert = \lVert w - v \rVert$$
  
  by homogeneity.

* **Property (3)** of the metric is implied by property (3) of the norm because 
  
  $$(x - y) + (y - z) = (x - z).$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The Norm Is Lipschitz Continuous)</span></p>

For all $u, v$ in a normed space,

$$\big\lvert \lVert u \rVert - \lVert v \rVert \big\rvert \le \lVert u - v \rVert.$$

In particular $\lVert \cdot \rVert : V \to [0, \infty)$ is continuous: $u\_n \to u$ implies $\lVert u\_n \rVert \to \lVert u \rVert$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

By the triangle inequality, $\lVert u \rVert \le \lVert u - v \rVert + \lVert v \rVert$ and $\lVert v \rVert \le \lVert u - v \rVert + \lVert u \rVert$. $\square$

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

<figure class="math-figure">
  <svg viewBox="0 0 320 320" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Unit balls of l1, l2, and l-infinity norms in R^2">
    <g stroke="#e8e8e8" stroke-width="0.5" fill="none">
      <line x1="60"  y1="20"  x2="60"  y2="300" />
      <line x1="110" y1="20"  x2="110" y2="300" />
      <line x1="210" y1="20"  x2="210" y2="300" />
      <line x1="260" y1="20"  x2="260" y2="300" />
      <line x1="20"  y1="60"  x2="300" y2="60" />
      <line x1="20"  y1="110" x2="300" y2="110" />
      <line x1="20"  y1="210" x2="300" y2="210" />
      <line x1="20"  y1="260" x2="300" y2="260" />
    </g>
    <g stroke="#444" stroke-width="1.2" fill="none">
      <line x1="20"  y1="160" x2="300" y2="160" />
      <line x1="160" y1="20"  x2="160" y2="300" />
    </g>
    <polygon points="300,160 292,156 292,164" fill="#444" />
    <polygon points="160,20 156,28 164,28" fill="#444" />
    <rect x="60" y="60" width="200" height="200" fill="rgba(214,83,54,0.07)" stroke="#d65336" stroke-width="2" />
    <circle cx="160" cy="160" r="100" fill="rgba(44,73,148,0.10)" stroke="#2c4994" stroke-width="2" />
    <polygon points="160,60 260,160 160,260 60,160" fill="rgba(60,120,40,0.10)" stroke="#3d7a26" stroke-width="2" />
    <circle cx="160" cy="160" r="2.5" fill="#222" />
    <text x="305" y="156" font-size="12" fill="#444">x₁</text>
    <text x="166" y="22"  font-size="12" fill="#444">x₂</text>
    <text x="262" y="155" font-size="11" fill="#666">1</text>
    <text x="148" y="60"  font-size="11" fill="#666">1</text>
    <text x="46"  y="155" font-size="11" fill="#666">−1</text>
    <text x="148" y="278" font-size="11" fill="#666">−1</text>
    <text x="178" y="78"  font-size="13" font-weight="600" fill="#3d7a26">‖·‖₁</text>
    <text x="232" y="100" font-size="13" font-weight="600" fill="#2c4994">‖·‖₂</text>
    <text x="262" y="55"  font-size="13" font-weight="600" fill="#d65336">‖·‖∞</text>
  </svg>
  <figcaption>Unit balls $B(0,1)$ in $\mathbb{R}^2$ for the three canonical norms. As $p$ grows, the ball "puffs out" from the diamond ($p=1$) through the circle ($p=2$) toward the square ($p=\infty$), with each one inscribed in or circumscribing the next.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hölder's Inequality)</span></p>

Suppose that $n \in \mathbb{N}$, and let $a_k, b_k \in \mathbb{R}$, $\le k \le n$.

Prove that if $1 < p < \infty$ and

$$\frac{1}{p} + \frac{1}{q} = 1,$$

then

$$\sum_{k=1}^n |a_k b_k| \le \left(\sum_{k=1}^n |a_k|^p\right)^{1/p} \left(\sum_{k=1}^n |b_k|^q\right)^{1/q}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Hint</summary>

Prove that if $A,B > 0$ and $t \in (0,1)$, then

$$A^t B^{1-t} \le tA + (1-t)B$$

by showing the function

$$f(x) := tx + (1-t)B - x^t B^{1-t}, \qquad x > 0,$$

has a minimum at $x = B$.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Recall the proof of general Hölder's Inequality and reduce to only two variables.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Minkowski's Inequality)</span></p>

Suppose that $n \in \mathbb{N}$, and let $a_k, b_k \in \mathbb{R}$, $1 \le k \le n$. For $1 \le p < \infty$,

$$\left(\sum_{k=1}^n |a_k + b_k|^p\right)^{1/p} \le \left(\sum_{k=1}^n |a_k|^p\right)^{1/p} + \left(\sum_{k=1}^n |b_k|^p\right)^{1/p}.$$

Or equivalently,

$$\lVert a + b \rVert_p \leq \lVert a \rVert_p + \lVert b \rVert_p$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Hint</summary>

By the triangle inequality,

$$\sum_{k=1}^n |a_k + b_k|^p \le \sum_{k=1}^n |a_k|\,|a_k+b_k|^{p-1} + \sum_{k=1}^n |b_k|\,|a_k+b_k|^{p-1}.$$

Now apply Hölder’s inequality.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

First turn $\lvert a_k+b_k\rvert^p$ into a product, then apply Hölder to the two resulting sums.

Assume $1\le p<\infty$. For $p=1$, Minkowski is just the ordinary triangle inequality, so let $p>1$. Write $q$ for the conjugate exponent:

$$\frac1p+\frac1q=1,\qquad q=\frac{p}{p-1}.$$

**Start from the $p$-th power of the left-hand side.** Using

$$|a_k+b_k|\le |a_k|+|b_k|,$$

we get

$$
\begin{aligned}
\sum_{k=1}^n |a_k+b_k|^p
&= \sum_{k=1}^n |a_k+b_k|\,|a_k+b_k|^{p-1}\\
&\le \sum_{k=1}^n |a_k|\,|a_k+b_k|^{p-1} + \sum_{k=1}^n |b_k|\,|a_k+b_k|^{p-1}.
\end{aligned}
$$

**Now apply Hölder to each term separately.** For the first sum,

$$
\begin{aligned}
\sum_{k=1}^n |a_k|\,|a_k+b_k|^{p-1}
&\le\left(\sum_{k=1}^n |a_k|^p\right)^{1/p}\left(\sum_{k=1}^n |a_k+b_k|^{(p-1)q}\right)^{1/q}.
\end{aligned}
$$

Since $(p-1)q=p$, this simplifies to

$$\sum_{k=1}^n |a_k|\,|a_k+b_k|^{p-1} \le \left(\sum_{k=1}^n |a_k|^p\right)^{1/p} \left(\sum_{k=1}^n |a_k+b_k|^p\right)^{1/q}.$$

Exactly the same argument gives

$$\sum_{k=1}^n |b_k|\,|a_k+b_k|^{p-1}\le\left(\sum_{k=1}^n |b_k|^p\right)^{1/p}\left(\sum_{k=1}^n |a_k+b_k|^p\right)^{1/q}.$$

Putting these together,

$$\sum_{k=1}^n |a_k+b_k|^p \le \left[ \left(\sum_{k=1}^n |a_k|^p\right)^{1/p} + \left(\sum_{k=1}^n |b_k|^p\right)^{1/p} \right] \left(\sum_{k=1}^n |a_k+b_k|^p\right)^{1/q}.$$

**The last step is just cancellation.** If

$$\sum_{k=1}^n |a_k+b_k|^p=0,$$

there is nothing to prove. Otherwise divide by

$$\left(\sum_{k=1}^n |a_k+b_k|^p\right)^{1/q}.$$

Because

$$1-\frac1q=\frac1p,$$

you obtain

$$\boxed{\left(\sum_{k=1}^n |a_k+b_k|^p\right)^{1/p}\le\left(\sum_{k=1}^n |a_k|^p\right)^{1/p}+\left(\sum_{k=1}^n |b_k|^p\right)^{1/p}}.$$

What you found in your handwritten solution is the standard proof. The central idea is the factorization

$$|a_k+b_k|^p=|a_k+b_k|\,|a_k+b_k|^{p-1},$$

because that creates exactly the product structure Hölder needs.

</details>
</div>

### Norms on Function Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($C_\infty(X)$ and the Sup Norm)</span></p>

Let $X$ be a metric space. The vector space of bounded, continuous functions is

$$C_\infty(X) = \lbrace f : X \to \mathbb{C} : f \text{ continuous and bounded} \rbrace.$$

For example, $C_\infty([0, 1]) = C([0, 1])$ because all continuous functions on $[0, 1]$ are bounded.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Warning on the notation specific for this course)</span></p>

Here two notation are conflated:

1. $C^\infty$: **The Space of Smooth Functions**
2. $C_\infty$ or $C_0$: **The Space of Functions Vanishing at Infinity**
3. $C_b$: **The Space of Continuous Bounded Functions**
4. $C_\infty$: In this course it means "**The Space of Continuous Functions that can be measured by the $\infty$-norm**". It is essentailly common $C_b$.

**A Note on Cross-Notation.** While this notation is perfectly sound and well-defined inside this coursework, keep in mind that it is an author-specific choice. In other other analysis textbooks (like those by Walter Rudin or Gerald Folland), they will use $C_b(X)$ for bounded continuous functions, reserving $C_0(X)$ or $C_\infty(X)$ for functions vanishing at infinity.

* **The Logic Behind the Subscript:** In this specific notation system, the subscript $\infty$ does not stand for "vanishing at infinity". Instead, the subscript directly references the supremum norm (also called the $\infty$-norm, $\Vert\cdot\Vert_\infty$) used to equip the space.
* **The Domain Context:** Because the metric space $X$ is arbitrary (and not necessarily locally compact, like $\mathbb{R}^n$), defining functions that "vanish at infinity" wouldn't make sense on general spaces. Therefore, the textbook/instructor chooses $C_\infty(X)$ to mean "continuous functions that can be measured by the $\infty$-norm". 
* **The Compact Example:** The statement correctly notes that $C_\infty([0, 1]) = C([0, 1])$. Because the interval $[0, 1]$ is compact, the Extreme Value Theorem guarantees that every continuous function on it is automatically bounded.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The Space of Continuous Functions that can be measured by the $\infty$-norm equals to The Space of Continuous Bounded Functions)</span></p>

The space of continuous functions on a topological space $X$ that can be measured by the $\infty$-norm (supremum norm) is exactly the space of continuous bounded functions, denoted as $C_b(X)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**1. Define the Supremum Norm**

$\infty$-norm (or supremum norm) of a function $f: X \to \mathbb{C}$ is defined as:

$$\Vert{}f\Vert{}_\infty = \sup_{x \in X} \vert{}f(x)\vert{}$$ 

**2. Formulate the Condition for Measurement**

For a function to be "measurable" or well-defined under a norm, its norm must yield a finite real number.

$$\Vert{}f\Vert{}_\infty < \infty$$ 

**3. Match the Definition of Boundedness**
By definition, a function $f$ is bounded if there exists a real number $M \ge 0$ such that $\vert{}f(x)\vert{} \le M$ for all $x \in X$.

* If $\Vert{}f\Vert{}\_\infty < \infty$, the function is bounded by $M = \Vert{}f\Vert{}\_\infty$.
* If the function is bounded by $M$, then $\Vert{}f\Vert{}\_\infty \le M < \infty$.

**4. Establish the Vector Space**
Therefore, requiring a continuous function to have a finite $\infty$-norm is mathematically identical to requiring the function to be bounded. Both conditions yield the exact same set of functions.

**Conclusion**

The space of continuous functions with a finite $\infty$-norm is identically the space of continuous bounded functions $C_b(X)$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Sup Norm on $C_\infty(X)$)</span></p>

For any metric space $X$, we can define a norm on $C_\infty(X)$ as

$$\lVert u \rVert_\infty := \sup_{x \in X} \lvert u(x) \rvert.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Properties (1) and (2) of a norm are clear from the definitions. For property (3): if $u, v \in C_\infty(X)$, then for any $x \in X$,</p>
    $$\lvert u(x) + v(x) \rvert \le \lvert u(x) \rvert + \lvert v(x) \rvert \le \lVert u \rVert_\infty + \lVert v \rVert_\infty,$$
    <p>so $\lVert u + v \rVert_\infty = \sup_x \lvert u(x) + v(x) \rvert \le \lVert u \rVert_\infty + \lVert v \rVert_\infty$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniform Convergence)</span></p>

Convergence $u_n \to u$ in $C_\infty(X)$ means $\lVert u_n - u \rVert_\infty \to 0$, which unpacks to

$$\forall \varepsilon > 0,\; \exists N \in \mathbb{N} : \forall n \ge N,\; \forall x \in X\; \lvert u_n(x) - u(x) \rvert < \varepsilon.$$

This is precisely the definition of **uniform convergence** on $X$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Convergence in $C_\infty$ equivalent to the uniform convergence)</span></p>

> Notational remark: here we mean $C_\infty = C_b$ (bounded continuous functions).

Convergence in the $C_b(X)$ space under the $\infty$-norm (supremum norm) is the definition of uniform convergence.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Basically, by definition. Supremum norm imposes global max absolute value constraint. 

</details>
</div>

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

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>For $1 \le p < \infty$, the $p$-th power of the norm of $\lbrace 1/j \rbrace$ is a $p$-series:</p>
    $$\lVert \lbrace 1/j \rbrace \rVert_p^p = \sum_{j=1}^{\infty} \left\lvert \frac{1}{j} \right\rvert^p = \sum_{j=1}^{\infty} \frac{1}{j^p}.$$
    <p>By the integral test, $\sum_{j=1}^{\infty} j^{-p}$ converges if and only if $\int_1^{\infty} x^{-p}\,dx = \frac{1}{p-1}$ is finite, i.e. if and only if $p > 1$. So for every $p > 1$ we have $\lVert \lbrace 1/j \rbrace \rVert_p = \left( \sum_{j=1}^{\infty} j^{-p} \right)^{1/p} < \infty$, hence $\lbrace 1/j \rbrace \in \ell^p$.</p>
    <p>For $p = 1$ the same computation gives the harmonic series</p>
    $$\lVert \lbrace 1/j \rbrace \rVert_1 = \sum_{j=1}^{\infty} \frac{1}{j} = \infty,$$
    <p>which diverges, so $\lbrace 1/j \rbrace \notin \ell^1$. $\square$</p>
  </details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Don't confuse $\ell^p$ and $L^p$</summary>

The primary difference is that $\ell^p$ is a sequence space (discrete) while $L^p$ is a function space (continuous). Both are generalized vector spaces equipped with p-norms, but they operate on fundamentally different underlying measure spaces.

Mathematically, $\ell^p$ is actually a specific, special case of $L^p$ where the domain is the set of natural numbers $\mathbb{N}$ equipped with the discrete counting measure.

**Direct Comparison Overview**

| Feature | $\ell^p$ Space (Sequences) | $L^p$ Space (Functions) |
|---|---|---|
| Element Type | Infinite sequences of numbers, $x = (x_1, x_2, x_3, \dots)$ | Functions over a domain, $f(x)$ |
| Domain Type | Discrete index set (e.g., $\mathbb{N}$ or $\mathbb{Z}$) | Continuous domain (e.g., an interval $[a, b]$ or $\mathbb{R}^n$) |
| Summation Tool | Infinite series ($\sum$) | Lebesgue integral ($\int$) |
| Inclusions $p < q$ | $\ell^p \subset \ell^q$ (Smaller $p$ is more restrictive) | $L^q \subset L^p$ (On finite measure spaces like $[0,1]$) |

**1. The $\ell^p$ Spaces (Discrete)**

The space $\ell^p$ (often spoken as "little $L-p$") consists of all infinite sequences of real or complex numbers whose absolute values raised to the $p$-th power have a finite sum.

* Condition: A sequence $x = (x_n)\_{n=1}^\infty$ belongs to $\ell^p$ if:

  $$\Vert{}x\Vert{}_p = \left( \sum_{n=1}^\infty \vert{}x_n\vert{}^p \right)^{1/p} < \infty$$ 

* Intuition: For the sum to remain finite, the terms $x_n$ must decay to 0 as n → ∞.

**2. The $L^p$ Spaces (Continuous)**

The space $L^p$ (often spoken as "big $L-p$") consists of equivalence classes of measurable functions whose absolute values raised to the $p$-th power have a finite Lebesgue integral over a measure space $(X, \mu)$.

* Condition: A function $f: X \to \mathbb{C}$ belongs to $L^p(X)$ if:

  $$\Vert{}f\Vert{}_p = \left( \int_X \vert{}f(x)\vert{}^p \, d\mu \right)^{1/p} < \infty$$ 

* Intuition: For the integral to remain finite, the function cannot blow up too severely, and if the domain is infinite (like $\mathbb{R}$), it generally must decay toward 0 at infinity. 

**3. Understanding the Reversed Inclusion Properties**

One of the most confusing aspects when learning functional analysis at MIT or similar institutions is how their inclusion behaviors flip completely.

**Why $\ell^p \subset \ell^q$ when $p < q$**

If a sequence converges for a smaller power $p$, its terms must eventually drop below 1. When you raise numbers smaller than 1 to a higher power $q$, they get smaller ($\vert{}x_n\vert{}^q \le \vert{}x_n\vert{}^p$). Therefore, the sum stays finite.

* Example: The sequence $x_n = \frac{1}{n}$ is not in $\ell^1$ (harmonic series diverges), but it is in $\ell^2$ because $\sum \frac{1}{n^2} < \infty$.

**Why $L^q \subset L^p$ on finite domains when $p < q$**

For functions on a bounded domain (like $[0,1]$), the primary threat to a finite integral is a vertical asymptote (the function blowing up to infinity at a point). Higher powers $q$ make singularities blow up much faster, making it harder for the function to integrate cleanly.

* Example: On the domain $(0,1]$, the function $f(x) = \frac{1}{\sqrt{x}}$ is in $L^1$ because $\int_0^1 x^{-1/2} dx = 2$. However, it is not in $L^2$ because $\int_0^1 x^{-1} dx$ blows up logarithmically to infinity.

**Summary of Identity**

Both spaces share the elegant property of being complete normed vector spaces (Banach spaces) for all $1 \leq p \leq \infty$, and they form a Hilbert space exclusively when $p=2$. 

</details>
</div>

## Banach Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Banach Space)</span></p>

A normed space is a **Banach space** if it is complete with respect to the metric induced by the norm (i.e., every Cauchy sequence converges).

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Banach Spaces)</span></p>

We know from real analysis that $\mathbb{Q}$ is not **complete** — one can construct a sequence of rationals that converges to an irrational. We want our Banach spaces to "fill in the holes," analogously to how $\mathbb{R}$ completes $\mathbb{Q}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Completeness of $\mathbb{R}^n$ and $\mathbb{C}^n$)</span></p>

For any $n \in \mathbb{Z}\_{\ge 0}$, $\mathbb{R}^n$ and $\mathbb{C}^n$ are complete with respect to any of the $\lVert \cdot \rVert_p$ norms.

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


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Technique of being Banach Space)</span></p>

The same technique can be used to show that the $\ell^p$ spaces are Banach, and also that the space 

$$c_0 = \lbrace a \in \ell^\infty : \lim_{j \to \infty} a_j = 0 \rbrace$$

is Banach.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">($\ell^p$ spaces are Banach)</span></p>

For $1 \leq p \le \infty$, the space $\ell^p$ is a Banach space (normed and complete).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Normed space.** Closure under scalars is $\lVert ta \rVert\_p = \lvert t \rvert \lVert a \rVert\_p$, and closure under addition follows from the triangle inequality (Minkowski). A useful shortcut for Minkowski: it suffices to know it on $\mathbb{C}^N$ for every $N$. Indeed, then

$$\Big(\sum_{j=1}^N \lvert a_j + b_j \rvert^p\Big)^{1/p} \le \Big(\sum_{j=1}^N \lvert a_j \rvert^p\Big)^{1/p} + \Big(\sum_{j=1}^N \lvert b_j \rvert^p\Big)^{1/p} \le \lVert a \rVert_p + \lVert b \rVert_p,$$

and the left side is nondecreasing in $N$ and bounded, so we may let $N \to \infty$.

**Completeness ($1 \le p < \infty$).** Let $a^{(n)}$ be Cauchy in $\ell^p$. Since the norm is continuous, $\lVert a^{(n)} \rVert\_p$ is Cauchy in $\mathbb{R}$, so $\lVert a^{(n)} \rVert\_p \le A$ for all $n$.

* *Candidate limit.* For each $i$, $\lvert a\_i^{(n)} - a\_i^{(m)} \rvert \le \lVert a^{(n)} - a^{(m)} \rVert\_p$ since $\lVert a^{(n)}-a^{(m)}\rVert_p^p =\sum_{j=1}^{\infty}\lvert a_j^{(n)}-a_j^{(m)}\rvert^p \ge \lvert a_i^{(n)}-a_i^{(m)}\rvert^p$, so $a\_i^{(n)}$ is Cauchy in $\mathbb{C}$; let $a\_i = \lim\_n a\_i^{(n)}$.
* *$a \in \ell^p$.* For each finite $N$, $\sum\_{i \le N} \lvert a\_i^{(n)} \rvert^p \le A^p$; letting $n \to \infty$ (finitely many terms) gives $\sum\_{i \le N} \lvert a\_i \rvert^p \le A^p$, and then $N \to \infty$ gives $\lVert a \rVert\_p \le A$.
* *Convergence.* Given $\varepsilon > 0$ choose $M$ with $\lVert a^{(n)} - a^{(m)} \rVert\_p < \varepsilon/2$ for $n, m > M$. For each finite $N$, $\big(\sum\_{i \le N} \lvert a\_i^{(n)} - a\_i^{(m)} \rvert^p\big)^{1/p} < \varepsilon/2$; let $m \to \infty$ and then $N \to \infty$ to get $\lVert a^{(n)} - a \rVert\_p \le \varepsilon/2 < \varepsilon$ for $n > M$.

The trick throughout is to **back off to finite sums**, where limits can be exchanged freely.

**Completeness ($p = \infty$).** The same argument with $\sup\_{i \le N}$ in place of $\sum\_{i \le N}$ works verbatim (this is also a special case of the completeness of $C\_\infty(X)$ with $X = \mathbb{N}$). $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Unit Sphere of $\ell^p$ Is Closed but Not Compact)</span></p>

Let $1 \le p < \infty$ and $S = \lbrace a \in \ell^p : \lVert a \rVert\_p = 1 \rbrace$.

* $S$ is **closed**: it is the preimage of the closed set $\lbrace 1 \rbrace$ under the continuous map $\lVert \cdot \rVert\_p$.
* $S$ is **not compact**: the standard vectors $e^{(n)}$ (a $1$ in slot $n$, zeros elsewhere) lie in $S$ and satisfy $\lVert e^{(n)} - e^{(m)} \rVert\_p = 2^{1/p}$ for $n \ne m$, so no subsequence is Cauchy, let alone convergent.

This is a fundamental difference from finite dimensions, where the unit sphere is always compact (Heine–Borel). Being closed and bounded is no longer enough.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cauchy's Wrong Theorem)</span></p>

Cauchy once published:

> The pointwise limit of a sequence of continuous functions must always be a continuous function,

which is wrong. But if we make the convergence stronger, then it works. Specifically, if the convergence is uniform, the statement is true:

> The limit of a uniformly convergent sequence of continuous functions is itself continuous.

The space $C_\infty$ is a space of **continuous** functions, equipped with the **supremum norm**. The supremum norm implies uniform convergence by definition, which implies that any Cauchy sequence of continuous function with the uniform norm converges to the continuous function, making the space $C_\infty$ complete.

</div>

### Summability Characterization of Banach Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Summable and Absolutely Summable Series)</span></p>

Let $\lbrace v_n \rbrace_{n=1}^{\infty}$ be a sequence of points in a normed space $V$.

$$\sum_n v_n \quad \text{ is summable} \iff \left\lbrace \sum_{m=1}^{n} v_m \right\rbrace_{n=1}^{\infty} \text{ converges in }V.$$

$$\sum_n v_n \quad \text{ is absolutely summable} \iff \left\lbrace \sum_{m=1}^{n} \lVert v_m \rVert \right\rbrace_{n=1}^{\infty} \text{ converges in }\mathbb{R}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Scalar convergence controls vector-valued partial sums)</span></p>

The important point in the definition above is that absolute summability concerns the real-valued series

$$\sum_{n=1}^\infty \lVert v_n\rVert,$$

whereas summability concerns convergence of the vector-valued partial sums in $V$.

The proposition below shows that convergence of the scalar series of norms forces the vector-valued partial sums to be Cauchy. If $V$ is complete, this further implies that the series $\sum_n v_n$ is summable in $V$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Bridge from scalars to vectors: Absolutely Summable Implies Cauchy)</span></p>

If $\sum_n v_n$ is absolutely summable $\implies$ the sequence of partial sums $\left\lbrace \sum_{m=1}^{n} v_m \right\rbrace_{n=1}^{\infty}$ is Cauchy.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since the goal is two show that the sequence of partial sums $\left\lbrace \sum_{m=1}^{n} v_m \right\rbrace_{n=1}^{\infty}$ is Cauchy given $\sum_n v_n$ is absolutely summable, we must to derive that $\forall \varepsilon > 0 \exists N\in\mathbb{N}: \forall m>n\geq N$:

$$\left\lVert \sum_{j=1}^m v_j - \sum_{i=1}^n v_i\right\rVert \le \varepsilon.$$

The idea:
1. Reduce the proof of being Cauchy sequence for partial sums $\left\lbrace \sum_{m=1}^{n} v_m \right\rbrace_{n=1}^{\infty}$ to the proof of being Cauchy sequence for the partial sums $\sum_{n=1}^\infty \lVert v_n\rVert$.
2. Bound the first partial sums by the second partial sums.

The sequence of the partial sums $\sum_{n=1}^\infty \lVert v_n\rVert$ is Cauchy because it converges in $\mathbb{R}$, which is equivalent to be Cauchy on a real line. Then

$$\left\lVert \sum_{j=1}^m v_j - \sum_{i=1}^n v_i\right\rVert = \left\lVert \sum_{j=n+1}^m v_j \right\rVert \leq \sum_{j=n+1}^m \lVert v_j \rVert = \sum_{j=1}^m \lVert v_j \rVert - \sum_{i=1}^n \lVert v_i \rVert \le \varepsilon.$$

The pairs $(m,n)$ for the first partial sums are chosen the same as for the second partial sums.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Banach $\iff$ Absolutely Summable Series are Summable)</span></p>

A normed vector space $V$ is a Banach space $\iff$ every absolutely summable series is summable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>My proof</summary>

$\boxed{\Rightarrow}$. Consider the absolutely summable series in a normed Banach space. From the proposition, the absolute summable series means that the sequence of partial sums of the vectors is Cauchy, which implies that it convergence, since the space is Banach. Convergent partial sums series of vectors is summable by definition.

$\boxed{\Leftarrow}$.

> Let every asbolutelly summable series is summable in a normed space. From the proposition the sequence of partial sums of this sequence it is Cauchy. From the definition of the summable sequence of partial sums, the sequence if convergent. Then normed+complete vector space implies it is a Banach space. 

The problem with this direction is that it only considers the cauchy sequences, whose norm-series that are absolutely summable. To fix it, we still use the incorrect proof of $\Leftarrow$ direction, but now consider **any** Cauchy sequence $(v_n)\_{n=1}^\infty$ in a normed vector space $V$. The idea is to consider the variation of the Cauchy sequence and show that the variation converges conclude that it is absolutely summable, then from variation restore the initial Cauchy sequence and conclude that it is summable, thus converges. The problem is that the Cauchy sequence does not have finite variation in general, but every Cauchy sequence contains a subsequence with finite variation. We consider such a subsequence $(\tilde{v_n})\_{n=1}^\infty$, setting $\tilde{v}_0 := 0$. Then

$$\left\lbrace \sum_{m=1}^{n} \lVert \tilde{v}_i - \tilde{v}_{i-1} \rVert \right\rbrace_{n=1}^{\infty} \text{ converges in}\mathbb{R}$$

Implying $\sum_n \tilde{v}\_i - \tilde{v}\_{i-1} $ is absolutely summable and, by assumption, summable. Sin
ce it is summable, the sequence 

$$\sum_{i=1}^n \tilde{v}_i - \tilde{v}_{i-1} = \tilde{v}_n$$

converges with $n\to\infty$. It shows the subsequence of the Cauchy sequence has a limit, but any Cauchy sequence that contains a convergent subsequence will converge to that same limit.

</details>
</div>

<div class="accordion">
  <details>
    <summary>Textbook proof</summary>
    <p><strong>Forward direction.</strong> Suppose $V$ is Banach. Then $V$ is complete, so any absolutely summable series is Cauchy (by the proposition above), and thus convergent (i.e., summable).</p>
    <p><strong>Reverse direction.</strong> Suppose every absolutely summable series is summable. Take any Cauchy sequence $\lbrace v_n \rbrace$. We construct a convergent subsequence by "speeding up" the Cauchy-ness. For each $k \in \mathbb{N}$, choose $N_k$ such that $\lVert v_n - v_m \rVert < 2^{-k}$ for all $n, m \ge N_k$. Define $n_k = N_1 + \cdots + N_k$, so $n_1 < n_2 < \cdots$ and $n_k \ge N_k$. Then</p>
    $$\lVert v_{n_{k+1}} - v_{n_k} \rVert < 2^{-k},$$
    <p>so the telescoping series $\sum_k (v_{n_{k+1}} - v_{n_k})$ is absolutely summable (since $\sum_k 2^{-k} = 1$). By assumption it is summable, so the partial sums</p>
    $$\sum_{k=1}^{m} (v_{n_{k+1}} - v_{n_k}) = v_{n_{m+1}} - v_{n_1}$$
    <p>converge. Thus $\lbrace v_{n_k} \rbrace$ converges. Since a Cauchy sequence with a convergent subsequence must itself converge, $V$ is Banach. $\square$</p>
  </details>
</div>

### Completion of a Normed Space

Not every natural normed space is complete. The motivating example of the course:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($C([0,1])$ with the $L^1$ Norm Is Not Complete)</span></p>

On $C([0, 1])$ consider $\lVert u \rVert\_{L^1} = \int\_0^1 \lvert u(x) \rvert\,dx$ (a Riemann integral). Let $u\_n$ be $0$ on $[0, \frac12]$, $1$ on $[\frac12 + \frac1n, 1]$, and linear in between. For $m > n$, $u\_n$ and $u\_m$ differ only on $[\frac12, \frac12 + \frac1n]$, where both lie in $[0,1]$, so $\lVert u\_n - u\_m \rVert\_{L^1} \le \frac1n$: the sequence is Cauchy. Its only candidate limit is the jump function $\mathbf{1}\_{(1/2, 1]}$, which is not continuous: if $u \in C([0,1])$ had $\lVert u\_n - u \rVert\_{L^1} \to 0$, then $u = 0$ on $[0, \frac12]$ and $u = 1$ on $(\frac12, 1]$, contradicting continuity at $\frac12$.

Completing this space is essentially what the Lebesgue integral does. (Riemann integrable functions do not fix this: $\int \lvert u \rvert$ is only a seminorm on them, and even after quotienting the result is still not complete.)

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Completion)</span></p>

A **completion** of a normed space $V$ is a Banach space $B$ together with a linear map $I : V \to B$ such that

1. $\lVert I(v) \rVert\_B = \lVert v \rVert\_V$ for all $v \in V$ (so $I$ is injective), and
2. $I(V)$ is dense in $B$.

If $V$ is already Banach, $B = V$ with $I = \operatorname{Id}$ works.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Every Normed Space Has a Completion)</span></p>

Let $\tilde V$ be the linear space of absolutely summable series $\lbrace u\_k \rbrace$ in $V$ (i.e. $\sum\_k \lVert u\_k \rVert < \infty$), and $S \subset \tilde V$ the subspace of those whose partial sums tend to $0$ in $V$. Then

$$B = \tilde V / S, \qquad \big\lVert \lbrace u_k \rbrace + S \big\rVert_B = \lim_{n \to \infty} \Big\lVert \sum_{k=1}^n u_k \Big\rVert_V, \qquad I(v) = \lbrace v, 0, 0, \dots \rbrace + S$$

is a completion of $V$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

**Norm.** Partial sums of an absolutely summable series are Cauchy, so their norms are Cauchy in $\mathbb{R}$ and the limit exists. Adding an element of $S$ changes the partial sums by something tending to $0$, so $\lVert \cdot \rVert\_B$ is well defined. $\lVert b \rVert\_B = 0$ means the partial sums tend to $0$, i.e. the representative is in $S$, i.e. $b = 0$. Homogeneity is clear, and the triangle inequality passes to the limit from $\lVert \sum\_{k \le n}(u\_k + u\_k^{\prime}) \rVert \le \lVert \sum\_{k \le n} u\_k \rVert + \lVert \sum\_{k \le n} u\_k^{\prime} \rVert$. Clearly $\lVert I(v) \rVert\_B = \lVert v \rVert\_V$. Note also $\lVert \lbrace u\_k \rbrace + S \rVert\_B \le \sum\_k \lVert u\_k \rVert$.

**Density.** Let $b$ be represented by $\lbrace u\_k \rbrace$ and $v\_N = \sum\_{k \le N} u\_k$. The class $I(v\_N) - b$ is represented by $\lbrace v\_N - u\_1, -u\_2, -u\_3, \dots \rbrace$, whose $p$-th partial sum is $-\sum\_{N < k \le p} u\_k$. Hence $\lVert I(v\_N) - b \rVert\_B \le \sum\_{k > N} \lVert u\_k \rVert \to 0$.

**Completeness.** By the summability criterion, it suffices to show that an absolutely summable series $\sum\_n b\_n$ in $B$ converges.

* *Good representatives.* If $\lbrace u\_k \rbrace$ represents $b\_n$, choose $M$ with $\sum\_{k > M} \lVert u\_k \rVert \le 2^{-n-1}$ and regroup the head: $v\_1^{(n)} = \sum\_{k \le M} u\_k$, $v\_k^{(n)} = u\_{M+k-1}$ for $k \ge 2$. The partial sums of the two series differ by $\sum\_{N < k \le N+M-1} u\_k \to 0$, so $\lbrace v\_k^{(n)} \rbrace$ still represents $b\_n$. Moreover, for every $p \ge M$, writing $\sum\_{k \le M} u\_k = \sum\_{k \le p} u\_k - \sum\_{M < k \le p} u\_k$,

  $$\sum_k \lVert v_k^{(n)} \rVert \le \Big\lVert \sum_{k \le p} u_k \Big\rVert + 2\sum_{k > M} \lVert u_k \rVert,$$

  and letting $p \to \infty$ gives $\sum\_k \lVert v\_k^{(n)} \rVert \le \lVert b\_n \rVert\_B + 2^{-n}$.
* *The limit.* Hence $\sum\_{n, k} \lVert v\_k^{(n)} \rVert \le \sum\_n (\lVert b\_n \rVert\_B + 2^{-n}) < \infty$: the **double** series is absolutely summable. Let $\lbrace w\_j \rbrace$ be its enumeration along anti-diagonals $n + k = \text{const}$; it lies in $\tilde V$, so $b := \lbrace w\_j \rbrace + S \in B$.
* *Convergence.* $b - \sum\_{n \le N} b\_n$ is represented by $\lbrace w\_j - \sum\_{n \le N} v\_j^{(n)} \rbrace\_j$. Its partial sums are (sum over a triangle of the array) minus (sum over an $N \times p$ rectangle), so their norms are bounded by the sum of $\lVert v\_k^{(n)} \rVert$ over the symmetric difference: all terms with $n > N$, plus terms with $n \le N$ and $k$ beyond roughly $p - N$. Letting $p \to \infty$, the second part disappears, so
  $$\Big\lVert b - \sum_{n \le N} b_n \Big\rVert_B \le \sum_{n > N} \big(\lVert b_n \rVert_B + 2^{-n}\big) \xrightarrow{N \to \infty} 0. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Absolutely Summable Series)</span></p>

The completion could equally be built from Cauchy sequences modulo null sequences (as $\mathbb{R}$ is built from $\mathbb{Q}$). The absolutely summable series version is used because the very same arguments — regrouping heads of series to make them converge faster, and enumerating double series diagonally — reappear verbatim when the Lebesgue integral is constructed as the completion of the step functions in the $L^1$ norm (Section "The Lebesgue Integral via Step Functions").

</div>

## Linear Operators

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(From Matrices to Linear Operators)</span></p>

Now that we have characterized our vector spaces, we want to find the analog of **matrices** from linear algebra, which leads us to **operators** and **functionals**.

</div>

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Linear Operators are not continuous in general)</span></p>

In finite-dimensional vector spaces, all linear transformations are continuous. This is **not** always true when we have a map between two Banach spaces.

</div>

### Bounded Operators

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Continuous Linear Operator $\iff$ Bounded Linear Operator)</span></p>

Let $V, W$ be two normed vector spaces. A linear operator $T : V \to W$ is continuous if and only if there exists $C > 0$ such that for all $v \in V$,

$$\lVert Tv \rVert_W \le C \lVert v \rVert_V.$$


In this case we say $T$ is a **bounded** linear operator.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>My proof</summary>

$\boxed{\Leftarrow}$. We want to obtain

$$\forall \delta \ge 0 \exists \varepsilon \ge 0: \lVert v - u \rVert_V \le \varepsilon \implies \lVert Tv - Tu \rVert_W \le \delta.$$

By linearity of the operator, $Tv-Tu = T(v-u)$, then for given $\delta$, set $\varepsilon := \delta / C$, 

$$\lVert Tv - Tu \rVert_W = \lVert T(v - u) \rVert_W \leq C\lVert \rVert_V \le C\cdot\frac{\delta}{C} = \delta.$$

$\boxed{\Rightarrow}$. 

Fix **some** $C>0$ to induce a reference ball, which we will use for directions.

$$\forall C \ge 0 \exists \varepsilon \ge 0: \lVert 0 - u \rVert_V = \lVert u \rVert_V \le \varepsilon \implies \lVert T0 - Tu \rVert_W = \lVert Tu \rVert_W \le C.$$

Since the ball is open, closed, without breaking inequality, we pick radius of the reference ball smaller than \varepsilon, $r:= \varepsilon / 2$. Then for every vector $v\in V$, there exists $u$ on a sphere $B(0,r)$: $v = \frac{\lVert v\rVert_V}{r} u$.

$$\frac{\lVert Tv \rVert_W}{\lVert v\rVert_V} = \frac{\lVert T(\frac{\lVert v\rVert_V}{r} u) \rVert_W}{\lVert v\rVert_V} = \lVert v\rVert_V\frac{\lVert Tu \rVert_W}{r\lVert v\rVert_V} = \frac{\lVert Tu \rVert_W}{r} \le \frac{C}{r} = \frac{2C}{\varepsilon}$$

</details>
</div>
<!-- <div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Bounded $\Rightarrow$ continuous.</strong> Suppose $\lVert Tv \rVert_W \le C\lVert v \rVert_V$ for all $v$. If $v_n \to v$, then by linearity,</p>
    $$\lVert Tv_n - Tv \rVert_W = \lVert T(v_n - v) \rVert_W \le C \lVert v_n - v \rVert_V \to 0,$$
    <p>so $Tv_n \to Tv$.</p>
    <p><strong>Continuous $\Rightarrow$ bounded.</strong> Suppose $T$ is continuous. The set $T^{-1}(B_W(0, 1)) = \lbrace v \in V : Tv \in B_W(0, 1) \rbrace$ is open in $V$ (by the topological characterization of continuity). Since $T(0) = 0 \in B_W(0, 1)$, there is some $r > 0$ with $B_V(0, r) \subset T^{-1}(B_W(0, 1))$. Take $C = 2/r$. For any $v \in V \setminus \lbrace 0 \rbrace$, the vector $\frac{r}{2\lVert v \rVert_V} v$ has norm $r/2 < r$, so it lies in $B_V(0, r)$, hence</p>
    $$\left\lVert T\!\left(\frac{r}{2\lVert v \rVert_V} v\right) \right\rVert_W < 1 \implies \lVert Tv \rVert_W \le \frac{2}{r} \lVert v \rVert_V. \quad\square$$
  </details>
</div> -->

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Nullspace of a Bounded Linear Operator is Closed)</span></p>

Let $V,W$ be normed vector spaces and let $T\in\mathcal B(V,W)$. Then the nullspace

$$\mathcal N(T)=\lbrace v\in V:Tv=0\rbrace$$

is a closed linear subspace of $V$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Linearity of $T$ implies that $\mathcal N(T)$ is a linear subspace. Since $T$ is bounded, it is continuous. Therefore

$$\mathcal N(T)=T^{-1}(\lbrace 0\rbrace)$$

is closed, because $\lbrace 0\rbrace\subset W$ is closed and the preimage of a closed set under a continuous map is closed. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Boundedness in Big-O notation)</span></p>

One may write $\lVert Tv\rVert_W=O(\lVert v\rVert_V)$ as $v\to0$, provided the implied constant is uniform in $v$. For linear $T$, this is equivalent to boundedness. However, the explicit inequality $\lVert Tv\rVert_W\le C\lVert v\rVert_V$ is usually preferable because it makes the global uniform bound transparent.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Continuous Linear Operators are Lipschitz Continuous)</span></p>

In fact, the proof shows something stronger: every continuous linear operator $T:V\to W$ is globally Lipschitz. Indeed, there exists $C>0$ such that

$$\|Tv-Tu\|_W = \|T(v-u)\|_W \le C\|v-u\|_V$$

for all $u,v\in V$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bounded Operator Maps Bounded Sets to Bounded Sets)</span></p>

The theorem above does not mean the image of the bounded operator $T$ is bounded — it means bounded subsets of $V$ are sent to bounded subsets of $W$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Integral Operator is Bounded)</span></p>

The linear operator $T : C([0, 1]) \to C([0, 1])$ defined by 

$$Tf(x) = \int_0^1 K(x, y) f(y)\, dy$$

is bounded. Indeed, for all $x \in [0, 1]$,

$$\lvert Tf(x) \rvert = \left\lvert \int_0^1 K(x, y) f(y)\, dy \right\rvert \le \int_0^1 \lvert K(x, y) \rvert \lvert f(y) \rvert\, dy \le \lVert K \rVert_\infty \lVert f \rVert_\infty,$$

so 

$$\lVert Tf \rVert_\infty \le \lVert K \rVert_\infty \lVert f \rVert_\infty,$$

and we can take $C = \lVert K \rVert_\infty$. We often call $K$ the **kernel** of the operator.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Unbounded Linear Operator I)</span></p>

A very clean example is on the space

$$c_{00}:=\lbrace x=(x_1,x_2,\dots):x_n=0\text{ for all but finitely many }n\rbrace,$$

equipped with the $\ell^2$-norm.

Define

$$T:c_{00}\to c_{00},\qquad T(x_1,x_2,\dots)=(x_1,2x_2,3x_3,\dots).$$

It is clearly linear.

**Why is it unbounded?** Take the standard basis vector

$$e_n=(0,\dots,0,1,0,\dots).$$

Then

$$\|e_n\|_2=1, \qquad \|Te_n\|_2=n.$$

Thus, $T$ is linear but unbounded, hence discontinuous.

The mental model is useful: a bounded linear operator is allowed to amplify vectors, but its **amplification factor**

$$\frac{\|Tv\|}{\|v\|}$$

must have a uniform ceiling. Here different coordinate directions $e_n$ are amplified by factors $1,2,3,\ldots$, so there is no ceiling.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Unbounded Linear Operator II: Differentiation Operator)</span></p>

A classical analytic example is differentiation

$$D:C^1([0,1])\to C([0,1]),\qquad Df=f',$$

when $C^1([0,1])$ is equipped only with the sup norm $\|f\|\_\infty$: for $f_n(x)=x^n$,

$$\|f_n\|_\infty=1,\qquad \|f_n'\|_\infty=n.$$

</div>

### The Space of Bounded Operators and the Operator Norm

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Space of Bounded Linear Operators)</span></p>

Let $V$ and $W$ be two normed spaces. The set of bounded linear operators from $V$ to $W$ is denoted $\mathcal{B}(V, W)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\mathcal{B}(V, W)$ is a Vector Space)</span></p>

$\mathcal{B}(V, W)$ is a vector space (the sum of two linear operators is linear, etc.), and we can equip it with a norm.

</div>

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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Operator norm gives a bound)</span></p>

In general, the operator norm gives a bound: for all $v \in V$,

$$\left\lVert T\!\left(\frac{v}{\lVert v \rVert}\right) \right\rVert \le \lVert T \rVert \implies \lVert Tv \rVert \le \lVert T \rVert \lVert v \rVert.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">($\mathcal{B}(V, W)$ is Banach when $W$ is Banach)</span></p>

If $V$ is a normed vector space and $W$ is a Banach space $\implies$ $\mathcal{B}(V, W)$ is a Banach space.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>My proof</summary>

1. $\mathcal{B}(V, W)$ is a normed space with an **operator norm** as a norm.
2. $\mathcal{B}(V, W)$ is a complete vector space:

**GUESS / CANDIDATE CONSTRUCTION.**

Fix a Cauchy sequence $(T_n)\_{n\in\mathbb{N}}$ in $\mathcal{B}(V, W)$ space and **any** vector $v\in V$. Then the sequence $(T_n v)\_{n\in\mathbb{N}}$ in $W$ is Cauchy:

$$\|T_m v - T_n v\| = \|(T_m - T_n)v\| \leq \|(T_m - T_n)\| \cdot \|v|\ \le \varepsilon \|v\|\,$$

and has a limit, since $W$ is Banach. For each fixed $v\in V$, since $(T_n v)\_{n\in\mathbb{N}}$ is Cauchy in $W$, define

$$Tv:= \lim_{n\to\infty} T_n v.$$

We do not need to know in advance that the limit of $(T_n n)\_{n\in\mathbb{N}}$ is the image of $v$ under some already-existing operator $T$. This is precisely how we construct the **candidate** (guess) limit operator. After constructing $T$ do you have to prove that this new map is linear and bounded.

**CANDIDATE LIVES IN OUR SPACE.**

* $T$ is operator since sends every $v\in V$ to some $w\in W$ by a property of $W$ being Banach.

* **$T$ is linear:**

  1. $T0:= \lim_{n\to\infty} T_n 0 = 0.$
  2. $T(\alpha v+ \beta u):= \lim_{n\to\infty} T_n (\alpha v+ \beta u) = \alpha\lim_{n\to\infty} T_n v + \beta\lim_{n\to\infty} T_n u = \alpha Tv + \beta Tu.$

* **$T$ is bounded:**

$$\|Tv\|_W = \|\lim_{n\to\infty} T_n v\|_W = \lim_{n\to\infty} \|T_n v\|_W \leq \lim_{n\to\infty} C\|v\|_V = C\|v\|_V, $$

where $\lVert \lim_{n\to\infty} T_n v\rVert\_W = \lim_{n\to\infty} \lVert T_n v\rVert\_W$ since the norm is continuous and the uniformly bounding constant $C$ comes from the fact that Cauchy sequence in a normed space is bounded, therefore there exists a global for that sequence constant $C>0$ such that $\lVert T_n\rVert < C \forall n\in\mathbb{N}$, thus $\lVert T_n v\rVert \leq \lVert v\rVert\lVert T_n\rVert \le C\lVert v\rVert$.

**CANDIDATE IS A LIMIT IN OUR SPACE.**

Afte we showed that $T\in\mathcal{B}(V, W)$, it is remained to check whether $\lim_{n\to\infty} T_n = T$ in operator norm. For **any** vector $v\in V$ holds

$$\|(T_m - T_n)v\| \to \|(T-T_n)v\|$$

by convergence in $W$ and since for all $m>n\geq N: \lVert (T_m - T_n)v\rVert \le \varepsilon \lVert v \rVert$ for any small $\varepsilon$ and sufficiently large $N$ and, by continuity of a norm,

$$\|(T-T_n)v\| \leq \varepsilon \|v\|.$$

Since it holds for any $v\in V$,

$$\| T - T_n \| = \sup_{\lVert v \rVert = 1,\, v \in V} \lVert (T - T_n)v \rVert \leq \varepsilon$$.

Or another way to see it is to fix $n\in\mathbb{N}$ and then for any $v\in V$:

$$\|(T - T_n)v\| \leq \|(T-T_m)v\| + \|(T_m - T_n)v\|,$$

where the first term goes to zero with $m\to\infty$ by convergence in $W$ and the second term is bounded by $\varepsilon\lVert v\rVert$.

</details>
</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>We use the summability characterization (Theorem above). Suppose $\lbrace T_n \rbrace$ is a sequence in $\mathcal{B}(V, W)$ with $C = \sum_n \lVert T_n \rVert < \infty$ (since is absolutely summable). We need to show $\sum_n T_n$ is summable.</p>
    <p><strong>Candidate.</strong> For any $v \in V$ and $m \in \mathbb{N}$,</p>
    $$\sum_{n=1}^{m} \lVert T_n v \rVert \le \sum_{n=1}^{m} \lVert T_n \rVert \lVert v \rVert \le C \lVert v \rVert.$$
    <p>Since $T_n v \in W$ and $W$ is Banach, the series $\sum_n T_n v$ is absolutely summable (A monotone increasing sequence of partial sums $s_m:=\sum_{n=1}^{m} \lVert T_n v \rVert$, bounded sequence in $\mathbb R$ converges) and thus summable. Define $Tv = \lim_{m \to \infty} \sum_{n=1}^{m} T_n v$.</p>
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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Dual space for all $\ell^p$ spaces)</span></p>

We can identify the dual space for all $\ell^p$ spaces: it turns out that

$$(\ell^p)' = \ell^{p'},$$

where $p, p'$ satisfy $\frac{1}{p} + \frac{1}{p'} = 1$. So the dual of $\ell^1$ is $\ell^\infty$, and the dual of $\ell^2$ is itself (this is the only $\ell^p$ space for which this is true). However, the dual of $\ell^\infty$ is **not** $\ell^1$.

</div>

## Subspaces and Quotients

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subspace)</span></p>

Let $V$ be a vector space. A subset $W \subseteq V$ is a **subspace** of $V$ if for all $w_1, w_2 \in W$ and $\lambda_1, \lambda_2 \in \mathbb{K}$, we have $\lambda_1 w_1 + \lambda_2 w_2 \in W$ (closure under linear combinations).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Closed Subset Of Banach Space $\iff$ Banach Subspace)</span></p>

A subspace $W$ of a Banach space $V$ is Banach (with norm inherited from $V$) $\iff$ $W$ is a closed subset of $V$ (with respect to the metric induced by the norm).

</div>

<div class="accordion">
  <details>
    <summary>proof sketch</summary>
    <p><strong>$W$ Banach $\Rightarrow$ $W$ closed.</strong> Every sequence in $W$ that converges (to something in $V$) must be Cauchy. Since $W$ is Banach, the sequence converges in $W$. By uniqueness of limits, the limit is in $W$.</p>
    <p><strong>$W$ closed $\Rightarrow$ $W$ Banach.</strong> Any Cauchy sequence in $W$ is also Cauchy in $V$, so it has a limit in $V$. Closedness ensures the limit is in $W$. $\square$</p>
  </details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Does not the direction "$W$ closed $\Rightarrow$ $W$ Banach" only prove ​that every closed subset of Banach space is complete?​​</summary>

Yes. Your reading is exactly right: the **metric part of the proof does not need $W$ to be a subspace**.

The more general statement is:

$$\boxed{\text{Every closed subset of a complete metric space is complete.}}$$

So if $V$ is Banach and $W\subseteq V$ is merely closed, then $W$ is complete with respect to the inherited metric.

The reason the proposition assumes that $W$ is a **subspace** is only because the conclusion says that $W$ is **Banach**. A Banach space must first be a normed vector space. If $W$ is just an arbitrary closed subset, it need not be closed under addition or scalar multiplication.

So the proposition is really combining two facts:

$$\boxed{\begin{array}{c}W\subseteq V\text{ closed}\\V\text{ complete}\end{array}\Longrightarrow W\text{ complete as a metric space},}$$

and, separately,

$$\boxed{W\le V\text{ linear subspace}\Longrightarrow W\text{ inherits a normed-vector-space structure}.}$$

Together these imply:

$$W\text{ closed linear subspace of Banach }V \Longrightarrow W\text{ Banach}.$$

So yes: the **subspace assumption is algebraic, not metric**. The proof of completeness itself only uses closedness.

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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Baire Category Theorem)</span></p>

Let $M$ be a complete metric space, and let $\lbrace C_n \rbrace_n$ be a collection of closed subsets of $M$ such that $M = \bigcup_{n \in \mathbb{N}} C_n$. Then at least one of the $C_n$ contains an open ball 

$$B(x, r) = \lbrace y \in M : d(x, y) < r \rbrace.$$

In other words, 

$$\boxed{\text{At least one $C_n$ has an interior point.}}$$

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Despite the name, this theorem has nothing to do with category theory. A powerful consequence: it can be used to prove that there exists a continuous function which is **nowhere differentiable**. When applying this theorem, the $C_n$ need not be closed — the result then says that one of their closures must contain an open ball. Equivalently, we cannot have all $C_n$ be **nowhere dense**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Banach–Steinhaus Theorem: Uniform Boundedness Principle)</span></p>

Let $B$ be a Banach space, $V$ be any normed space, and let $\lbrace T_n \rbrace$ be a sequence in $\mathcal{B}(B, V)$. If for all $b \in B$ we have $\sup_n \lVert T_n b \rVert < \infty$ (pointwise boundedness), then $\sup_n \lVert T_n \rVert < \infty$:

$$\forall b\in B: \ \sup_n \lVert T_n b \rVert < \infty \quad \implies \quad \sup_n \lVert T_n \rVert < \infty$$

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

### Strong Convergence of Operators

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Strong Operator Convergence)</span></p>

Let $X, Y$ be normed spaces. A sequence $A\_n \in \mathcal{B}(X, Y)$ **converges strongly** if $A\_n x$ converges in $Y$ for every $x \in X$; its strong limit is $Ax := \lim\_n A\_n x$.

</div>

Norm convergence $\lVert A\_n - A \rVert \to 0$ means $A\_n x \to Ax$ **uniformly** on the unit ball; strong convergence only asks for it pointwise.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Strong but Not Norm Convergence)</span></p>

On $\ell^2$ let $P\_N$ keep the first $N$ coordinates: $P\_N(a\_1, a\_2, \dots) = (a\_1, \dots, a\_N, 0, \dots)$. For each $a \in \ell^2$, $\lVert a - P\_N a \rVert^2 = \sum\_{k > N}\lvert a\_k \rvert^2 \to 0$, so $P\_N \to I$ strongly. But $(I - P\_N)e\_{N+1} = e\_{N+1}$, so $\lVert I - P\_N \rVert = 1$ for all $N$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Strong Limits Are Bounded)</span></p>

Let $X$ be a Banach space, $Y$ a normed space, and $A\_n \in \mathcal{B}(X, Y)$ strongly convergent with limit $A$. Then $\sup\_n \lVert A\_n \rVert < \infty$, and $A \in \mathcal{B}(X, Y)$ with $\lVert A \rVert \le \liminf\_n \lVert A\_n \rVert$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

For each $x$, the convergent sequence $A\_n x$ is bounded, so the Uniform Boundedness Principle gives $\lVert A\_n \rVert \le C$. The limit $A$ is clearly linear, and $\lVert Ax \rVert = \lim\_n \lVert A\_n x \rVert \le \liminf\_n \lVert A\_n \rVert \lVert x \rVert$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Products Are Strongly Continuous)</span></p>

Let $X$ be a Banach space and $A\_n \to A$, $B\_n \to B$ strongly in $\mathcal{B}(X)$. Then $A\_n B\_n \to AB$ strongly.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

For $x \in X$,

$$A_n B_n x - ABx = A_n(B_n x - Bx) + (A_n - A)Bx.$$

The second term tends to $0$ by strong convergence of $A\_n$ (applied to the fixed vector $Bx$). For the first, the previous proposition gives $\lVert A\_n \rVert \le C$, so $\lVert A\_n(B\_n x - Bx) \rVert \le C\lVert B\_n x - Bx \rVert \to 0$. The uniform bound is essential here: without it, $A\_n$ could amplify the small vectors $B\_n x - Bx$. $\square$

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
    <p>We first prove a specialized result: the image of the open unit ball $B_1(0, 1) = \lbrace b \in B_1 : \lVert b \rVert < 1 \rbrace$ contains an open ball in $B_2$ centered at $0$.</p>
    <p><strong>Step 1: The closure $\overline{T(B(0, 1))}$ contains an open ball.</strong> Since $T$ is surjective, $B_2 = \bigcup_{n \in \mathbb{N}} \overline{T(B(0, n))}$. By Baire's theorem, some $\overline{T(B(0, n_0))}$ contains an open ball. By linearity, this is the same as $n_0 \overline{T(B(0, 1))}$, so $\overline{T(B(0, 1))}$ itself contains an open ball. Restated, there exist $v_0 \in B_2$ and $r > 0$ such that $B(v_0, 4r) \subset \overline{T(B(0, 1))}$.</p>
    <p><strong>Step 2: Show $B(0, r) \subset \overline{T(B(0, 1))}$.</strong> Pick $v_1 = Tu_1 \in T(B(0, 1))$ with $\lVert v_0 - v_1 \rVert < 2r$. Then $B(v_1, 2r) \subset B(v_0, 4r) \subset \overline{T(B(0, 1))}$. For any $\lVert v \rVert < r$, the element $\frac{1}{2}(2v + v_1)$ is in $\frac{1}{2}B(v_1, 2r) = \frac{1}{2}\overline{T(B(0, 1))}$. By linearity this equals $-T(u_1/2) + \overline{T(B(0, 1/2))}$, and since $u_1$ has norm less than 1, this set is contained in $\overline{T(B(0, 1))}$.</p>
    <p><strong>Step 3: Show $B(0, r) \subset T(B(0, 1))$.</strong> Take any $v \in B(0, r/2)$. By the closure property (with $n = 1$), there exists $b_1 \in B(0, 1/2)$ in $B_1$ with $\lVert v - Tb_1 \rVert < r/4$. Taking $n = 2$, there exists $b_2 \in B(0, 1/4)$ with $\lVert v - Tb_1 - Tb_2 \rVert < r/8$. Iterating, we get $\lbrace b_k \rbrace$ with $\lVert b_k \rVert < 2^{-k}$ and $\lVert v - \sum_{k=1}^{n} Tb_k \rVert < 2^{-n-1}r$. The series $\sum b_k$ is absolutely summable, so $b = \sum_{k=1}^{\infty} b_k$ converges in $B_1$ with $\lVert b \rVert < \sum 2^{-k} = 1$. Since $T$ is continuous, $Tb = \sum Tb_k = v$. So $v \in T(B(0, 1))$.</p>
    <p><strong>Step 4: General open sets.</strong> If $U \subset B_1$ is open and $b_2 = Tb_1 \in T(U)$, then by openness of $U$ there exists $\varepsilon > 0$ with $B(b_1, \varepsilon) \subset U$. By the work above, there exists $\delta > 0$ with $B(0, \delta) \subset T(B(0, 1))$. Then $B(b_2, \varepsilon\delta) = b_2 + \varepsilon B(0, \delta) \subset T(b_1 + B(0, \varepsilon)) = T(B(b_1, \varepsilon)) \subset T(U)$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Closed Graph Theorem: Alternative form)</span></p>

Let $X, Y$ be Banach spaces. A map $F : X \to Y$ is continuous iff $F$ is closed.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Here $F$ is linear and "closed" means its graph $\Gamma(F) = \lbrace (x, Fx) : x \in X \rbrace$ is closed in $X \times Y$, so this is the Closed Graph Theorem above; we recall the argument as a commutative diagram. The projections $\pi\_1(x, y) = x$ and $\pi\_2(x, y) = y$ are continuous, and on $\Gamma(F)$ we have

$$F \circ \pi_1 = \pi_2 \quad \text{on } \Gamma(F),$$

since both send $(x, Fx)$ to $Fx$.

$(\Rightarrow)$ If $F$ is continuous and $(x\_n, Fx\_n) \to (x, y)$, then $Fx\_n \to Fx$, so $y = Fx$ and the limit lies in $\Gamma(F)$.

$(\Leftarrow)$ $\Gamma(F)$ is a closed subspace of the Banach space $X \times Y$, hence a Banach space, and $\pi\_1 : \Gamma(F) \to X$ is a continuous linear bijection (each $x$ is the first entry of exactly one pair $(x, Fx)$). By the Bounded Inverse Theorem, $S = \pi\_1^{-1} : X \to \Gamma(F)$ is continuous, hence so is $F = \pi\_2 \circ S$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Bounded Inverse Theorem)</span></p>

Let $X, Y$ be Banach spaces and $A \in \mathcal{L}(X, Y)$ bijective. Then $A^{-1} : Y \to X$ is continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

The inverse map $S = A^{-1} : Y \to X$ is linear. Careful with notation: for $O \subset X$, the preimage of $O$ under $S$ is $S^{-1}(O) = A(O)$. If $O$ is open, then $A(O)$ is open by the Open Mapping Theorem ($A$ is bounded and surjective). So preimages of open sets under $S$ are open, i.e. $S$ is continuous — equivalently bounded. $\square$

</details>
</div>

## The Hahn-Banach Theorem

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Two main perspectives on Hahn-Banach Theorem)</span></p>

The **Hahn-Banach theorem** is a cornerstone of functional analysis that guarantees you can extend a mathematical rule (a linear functional) from a small subspace to an entire vector space without losing its core properties. It essentially ensures that abstract vector spaces have a sufficiently "rich" structure to be analyzed. 

The theorem tells us two main things:

* **The Extension Principle:** If you have a bounded linear function operating on a small part of a larger space, you can always extend it to the entire space while preserving its bound or "norm".
* **The Separation Principle:** In geometric terms, it proves that we can use linear functions (hyperplanes) to separate distinct points or disjoint convex shapes.

The theorem relies on the **Axiom of Choice**, meaning its proof requires assuming it is possible to make an infinite number of choices, even without an explicit rule to make them. 

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Hahn-Banach matters?)</span></p>

In infinite-dimensional spaces, things get incredibly complicated and intuition from 3D space often fails. Hahn-Banach provides the scaffolding needed to:

* **Create Dual Spaces:** It guarantees that every normed vector space has "enough" continuous linear functionals to study the space itself.
* **Define Distances and Norms:** It gives us the ability to rigorously show that if two vectors are distinct, there is a functional that can tell them apart.
* **Optimization and Physics:** The separation theorem is heavily used in convex optimization, economics, and quantum mechanics to find optimal solutions (like pricing models or shortest paths) when constrained by boundaries or limits.

</div>

### Zorn's Lemma and Hamel Bases

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial Order, Upper Bound, Maximal Element)</span></p>

* A **partial order** on a set $E$ is a relation $\preceq$ on $E$ with the following properties:
  * Reflexivity: $e \preceq e$ for all $e \in E$.
  * Antisymmetry: if $e \preceq f$ and $f \preceq e$, then $e = f$.
  * Transitivity: if $e \preceq f$ and $f \preceq g$, then $e \preceq g$.
* An **upper bound** of a set $D \subset E$ is an element $e \in E$ with $d \preceq e$ for all $d \in D$. 
* A **greatest element** of $E$ is an upper bound $e$ of $E$ and $e\in E$.
* A **maximal element** of $E$ is an element $e$ such that $e \preceq f \implies e = f$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Upper Bound vs. Maximal Element vs. Greatest Element)</span></p>

Notably, in a partial ordering a maximal element does not need to sit "on top" of everything — there can be incomparable elements "to the side."

* **Can we have two upper bounds?**
  * Yes. A set $D \subset E$ can have multiple distinct upper bounds. If $e_1$ and $e_2$ are upper bounds of $D$, then

    $$d \preceq e_1 \quad\text{and}\quad d \preceq e_2 \qquad \forall d \in D,$$

    but this does not imply that $e_1$ and $e_2$ are comparable. Even if they are comparable, they need not be equal.
* **Can we have two upper greatest elements?**
  * No, for two different greatest elements $e_1 \neq e_2$: 
    * $d \preceq e_1 \land d \preceq e_2 \forall d \in E \implies e_2 \preceq e_1 \land e_1 \preceq e_2$, 
    * but anyisymetry implies $e_1 = e_2$.
* **Can we have two maximal elements?**
  * Yes. Distinct maximal elements can coexist. They must be incomparable: if $e_1$ and $e_2$ are maximal and $e_1 \preceq e_2$, then maximality of $e_1$ implies $e_1=e_2$.
* **Can an upper bound and a different maximal element coexist?**
  * Yes. Being an upper bound is a property relative to a subset $D \subset E$, whereas being maximal is a property relative to the whole partially ordered set $E$. An upper bound of $D$ need not be maximal in $E$, and a maximal element of $E$ need not be an upper bound of $D$.
  * In the special case $D=E$, an upper bound of $E$ is called a **greatest element**. A greatest element is necessarily maximal and, if it exists, is unique. Conversely, a maximal element need not be a greatest element.

</div>

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

Every vector space $V$ has a Hamel basis.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Let $E$ be the set of linearly independent subsets of $V$, ordered by inclusion $\preceq$. If $C$ is a chain in $E$, define $c = \bigcup_{e \in C} e$. We claim $c$ is linearly independent and thus $c$ is an upper bound for $C$.

By Zorn's lemma, $E$ has a maximal element $H$. If $H$ does not span $V$, there exists $v \in V$ not in the span of $H$, so $H \cup \lbrace v \rbrace$ is linearly independent, contradicting maximality. Thus $H$ spans $V$ and is a Hamel basis. $\square$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Why is every element a finite combination of vectors from the maximal set?</summary>

The missing point is simply what **“span” means**. By definition,

$$\operatorname{span}(H) = \left\lbrace \sum_{i=1}^n a_i h_i : n\in\mathbb N,\; a_i\in\mathbb F,\; h_i\in H \right\rbrace.$$

So the span contains **only finite linear combinations** of elements of $H$. There is no notion of an infinite linear combination in an arbitrary vector space.

So Zorn's lemma itself does not somehow produce finite representations. Finiteness is already built into the algebraic notion of $\operatorname{span}$. This is precisely what distinguishes a Hamel basis from, say, a Schauder basis in a Banach space, where infinite convergent series are allowed.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hamel Bases can be uncountable)</span></p>

A **Hamel basis can be uncountable**, and for many infinite-dimensional spaces it must be uncountable.

The important distinction is:
* the basis set $H$ itself may be uncountable;
* each individual vector $v\in V$ is represented using only finitely many elements of $H$.

For example, a Hamel basis of $\mathbb R$ as a vector space over $\mathbb Q$ is uncountable. Yet every real number is a finite $\mathbb Q$-linear combination of basis elements.

This is one reason Hamel bases in Banach spaces behave very differently from Schauder bases: a Schauder basis is countable and represents vectors by infinite norm-convergent series, whereas a Hamel basis may be huge but uses only finite combinations.

</div>

### Hahn-Banach

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hahn-Banach)</span></p>

Let $V$ be a normed space, and let $M \subset V$ be a subspace. If 

$$u : M \to \mathbb{C} \quad\text{is a linear map with}\quad \lVert u(t) \rVert \le C \lVert t \rVert \quad \forall t \in M,$$

i.e., $u$ is a bounded linear functional on $M$, then there exists a **continuous extension** $U : V \to \mathbb{C}$ (an element of $V' = \mathcal{B}(V, \mathbb{C})$) such that with the same constant $C$

$$U\vert_M = u \quad \text{and}\quad \lVert U(t) \rVert \le C \lVert t \rVert \quad \forall t \in V.$$

</div>

The key intermediate step is extending by one dimension at a time:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(One-Dimensional Extension)</span></p>

Let $V$ be a normed space, $M \subset V$ a subspace, and $u : M \to \mathbb{C}$ linear with $\lVert u(t) \rVert \le C \lVert t \rVert$ for all $t \in M$. If $x \notin M$, then there exists $u' : M' \to \mathbb{C}$ linear on 

$$M' = M + \mathbb{C}x = \lbrace t + ax : t \in M, a \in \mathbb{C} \rbrace,$$

with $u'\vert_M = u$ and $\lVert u'(t') \rVert \le C \lVert t' \rVert$ for all $t' \in M'$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>my proof</summary>

The idea is the following:

1. Start with the mapping $u$ and extend it by defining the value $u'(x)$, while preserving the norm bound.
2. Then for all $y=t+ax\in M'$,

   $$u'(y)=u'(t+ax)=u(t)+au'(x).$$

3. Thus we want

   $$\lVert u'(y)\rVert = |u(t)+au'(x)| \le C\lVert t+ax\rVert = C\lVert y\rVert.$$

* If $C=0$, then $|u(t)|\le 0$ for every $t\in M$, so $u=0$, and the zero extension works.

* Otherwise, assume $C=1$ without loss of generality. Indeed, replace $u$ by $u/C$, construct an extension with bound $1$, and multiply the resulting extension back by $C$.

The problem is now to choose the new value $u'(x)\in\mathbb C$ in such a way that

$$|u(t)+au'(x)|\le \|t+ax\|$$

for every $t\in M$ and every $a\in\mathbb C$.

For $a=0$, this is already true because $u$ satisfies the original bound. So consider $a\neq0$.

Set

$$\tilde t:=\frac{t}{-a}\in M.$$

This is legitimate because $M$ is a complex linear subspace ($\tilde t$ is an equivalent to $t$ vector in a dual sense). Therefore the desired inequality is equivalent to

$$|u(\tilde t)-u'(x)|\le\|\tilde t-x\|.$$

Renaming $\tilde t$ back to $t$, the problem becomes:

> Choose $u'(x)\in\mathbb C$ such that
>
> $$|u(t)-u'(x)|\le\|t-x\|\qquad\forall t\in M.$$

The problem is that the mapping $u$ takes complex values. The straightforward idea would be to control separately

$$|\operatorname{Re}u(t)-\operatorname{Re}u'(x)|$$

and

$$|\operatorname{Im}u(t)-\operatorname{Im}u'(x)|.$$

However, if we later simply add these two estimates, we may obtain an unnecessary factor $2$. Also, complex numbers themselves cannot be ordered, so the supremum/infimum interval argument cannot be applied directly to $u(t)\in\mathbb C$.

Therefore we first isolate the real-valued part of the problem.

Define

$$w(t):=\operatorname{Re}u(t).$$

Then $w:M\to\mathbb R$ is real-linear and

$$|w(t)|\le|u(t)|\le\|t\|.$$

We first ask: how can we choose the value of the real extension at $x$?

Suppose

$$W(x)=\alpha\in\mathbb R.$$

Then for $t\in M$, preserving the norm bound requires

$$|w(t)-\alpha|\le\|t-x\|.$$

Equivalently,

$$w(t)-\|t-x\|\le\alpha\lew(t)+\|t-x\|.$$

Thus every $t\in M$ gives an interval of admissible values for $\alpha$:

$$I_t=\left[w(t)-\|t-x\|,\,w(t)+\|t-x\|\right].$$

We need to show that all these intervals have a common point.

To understand why such a common point exists, consider how much two values of $w$ could differ:

$$
\begin{aligned}
w(t_1)-w(t_2)
&=
w(t_1-t_2)\\
&\le
|w(t_1-t_2)|\\
&\le
\|t_1-t_2\|\\
&\le
\|t_1-x\|+\|t_2-x\|.
\end{aligned}
$$

Therefore

$$
w(t_1)-\|t_1-x\|
\le
w(t_2)+\|t_2-x\|
$$

for arbitrary $t_1,t_2\in M$.

In other words, every lower endpoint of every admissible interval lies below every upper endpoint of every admissible interval.

Hence

$$
\sup_{t\in M}
\left(
w(t)-\|t-x\|
\right)
\le
\inf_{t\in M}
\left(
w(t)+\|t-x\|
\right).
$$

Thus we may choose

$$
\alpha
\in
\left[
\sup_{t\in M}
\left(
w(t)-\|t-x\|
\right),
\,
\inf_{t\in M}
\left(
w(t)+\|t-x\|
\right)
\right].
$$

This gives a real-linear extension

$$
W_1:M+\mathbb Rx\to\mathbb R
$$

satisfying

$$
|W_1(z)|\le \|z\|.
$$

At this point we have chosen the real part of the desired complex extension in the new direction $x$.

However, adding the complex direction $x$ means more than adding only the real direction $x$. As a real vector space,

$$
M+\mathbb Cx
=
M+\mathbb Rx+\mathbb R(ix).
$$

Thus there is one more real direction to add, namely $ix$.

We now repeat exactly the same interval argument, but starting with the already-constructed real-linear functional $W_1$ on

$$
M+\mathbb Rx
$$

and adding the direction $ix$.

This is important: the second choice is not made independently of the first one. We extend the already-defined $W_1$, so compatibility with the value chosen at $x$ is automatically preserved.

The same reasoning produces a real-linear functional

$$
W:M+\mathbb Rx+\mathbb R(ix)\to\mathbb R
$$

such that

$$
|W(z)|\le \|z\|.
$$

But

$$
M+\mathbb Rx+\mathbb R(ix)
=
M+\mathbb Cx
=
M',
$$

so $W$ is now defined on all of $M'$.

Moreover,

$$
W|_M=w=\operatorname{Re}u.
$$

We now use $W$ to reconstruct a complex-valued functional. Define

$$
u'(y)
:=
W(y)-iW(iy),
\qquad y\in M'.
$$

First check that this agrees with $u$ on $M$.

For $t\in M$,

$$
W(t)
=
\operatorname{Re}u(t).
$$

Also, because $u$ is complex-linear,

$$
u(it)=iu(t).
$$

If

$$
u(t)=a+ib,
$$

then

$$
u(it)=ia-b=-b+ia,
$$

so

$$
W(it)
=
\operatorname{Re}u(it)
=
-b
=
-\operatorname{Im}u(t).
$$

Therefore

$$
\begin{aligned}
u'(t)
&=
W(t)-iW(it)\\
&=
\operatorname{Re}u(t)
+i\operatorname{Im}u(t)\\
&=
u(t).
\end{aligned}
$$

Thus $u'\mid_M=u$.

We also need $u'$ to be complex-linear.

Real linearity follows immediately from the real linearity of $W$. Moreover,

$$
\begin{aligned}
u'(iy)
&=
W(iy)-iW(i^2y)\\
&=
W(iy)+iW(y)\\
&=
i\left(W(y)-iW(iy)\right)\\
&=
iu'(y).
\end{aligned}
$$

Real linearity together with

$$
u'(iy)=iu'(y)
$$

implies full complex linearity.

It remains to prove that the norm bound has not increased.

At first glance, from

$$
u'(y)=W(y)-iW(iy)
$$

one might try

$$
|u'(y)|
\le
|W(y)|+|W(iy)|
\le
2\|y\|.
$$

But this introduces the unwanted constant $2$.

To avoid it, the trick is to rotate the complex value $u'(y)$ by a suitable phase so that it becomes a nonnegative real number.

Take arbitrary

$$
y=t+ax\in M'.
$$

Choose $\theta\in\mathbb R$ such that

$$
r
:=
e^{i\theta}u'(y)
=
|u'(y)|
\in\mathbb R_{\ge0}.
$$

Because $u'$ is complex-linear,

$$
r
=
e^{i\theta}u'(y)
=
u'(e^{i\theta}y).
$$

Now define

$$
\tilde y:=e^{i\theta}y.
$$

Since

$$
y=t+ax,
$$

we have

$$
\tilde y
=
e^{i\theta}t+e^{i\theta}ax.
$$

Let

$$
t_r:=e^{i\theta}t,
\qquad
a_r:=e^{i\theta}a.
$$

Then

$$
\tilde y=t_r+a_rx.
$$

We must verify that the rotated vector still lies in $M'$.

Because $M$ is a complex linear subspace,

$$
t_r=e^{i\theta}t\in M.
$$

Also,

$$
a_r=e^{i\theta}a\in\mathbb C.
$$

Therefore

$$
\tilde y=t_r+a_rx\in M'.
$$

So the phase rotation does not take us outside the domain of $u'$.

This is not an arbitrary ambient-space rotation: multiplication by $e^{i\theta}$ is complex scalar multiplication, and every complex subspace is invariant under it.

Now $u'(\tilde y)=r$ is real. Hence

$$
r
=
\operatorname{Re}u'(\tilde y).
$$

But by the definition

$$
u'(z)=W(z)-iW(iz),
$$

we have

$$
\operatorname{Re}u'(z)=W(z).
$$

Therefore

$$
\begin{aligned}
|u'(y)|
&=
r\\
&=
\operatorname{Re}u'(\tilde y)\\
&=
W(\tilde y)\\
&\le
|W(\tilde y)|\\
&\le
\|\tilde y\|.
\end{aligned}
$$

Finally, multiplication by $e^{i\theta}$ preserves the norm:

$$
\|\tilde y\|
=
\|e^{i\theta}y\|
=
|e^{i\theta}|\|y\|
=
\|y\|.
$$

Consequently,

$$
|u'(y)|
\le
\|y\|.
$$

Thus $u'$ is a complex-linear extension of $u$ from $M$ to

$$
M'=M+\mathbb Cx
$$

satisfying the same norm bound.

Undoing the normalization $C=1$, we obtain

$$
|u'(y)|
\le
C\|y\|
\qquad
\forall y\in M'.
$$

Hence the required one-dimensional extension exists.

</details>
</div>

<!-- <div class="accordion">
  <details>
    <summary>proof of the one-dimensional extension lemma</summary>
    <p>The representation $t' = t + ax$ for $t \in M$, $a \in \mathbb{C}$ is unique (if $t + ax = \tilde{t} + \tilde{a}x$, then $(a - \tilde{a})x = \tilde{t} - t \in M$, which forces $a = \tilde{a}$ since $x \notin M$). Define $u'(t + ax) = u(t) + a\lambda$ for some $\lambda \in \mathbb{C}$ to be chosen. This is clearly linear.</p>
    <p>If $C = 0$, we use $\lambda = 0$. Otherwise, assume $C = 1$ (by scaling). We need $\lvert u(t) + a\lambda \rvert \le \lVert t + ax \rVert$ for all $t \in M$, $a \in \mathbb{C}$. For $a \ne 0$, dividing by $\lvert a \rvert$ gives $\lvert u(t/(-a)) - \lambda \rvert \le \lVert t/(-a) - x \rVert$.</p>
    <p><strong>Choosing $\lambda$ (real part).</strong> Let $w(t) = \frac{u(t) + \overline{u(t)}}{2}$ be the real part. Since $w$ is real-valued and $\lvert w(t) \rvert \le \lVert t \rVert$, we get $w(t_1) - w(t_2) = w(t_1 - t_2) \le \lVert t_1 - t_2 \rVert \le \lVert t_1 - x \rVert + \lVert t_2 - x \rVert$. Thus</p>
    $$\sup_{t \in M} (w(t) - \lVert t - x \rVert) \le \inf_{t \in M} (w(t) + \lVert t - x \rVert).$$
    <p>Choose $\alpha \in \mathbb{R}$ between these bounds. Then $\lvert w(t) - \alpha \rvert \le \lVert t - x \rVert$ for all $t \in M$. Repeat the argument with $ix$ instead of $x$ to choose the imaginary part. This defines $u'$ on $M + \mathbb{C}x$ with the desired bound. $\square$</p>
  </details>
</div> -->

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

### Consequences of Hahn-Banach

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Existence of Norm-Achieving Functionals)</span></p>

Let $V$ be a normed space.

$$\forall v \in V \setminus \lbrace 0 \rbrace \ \exists f \in V': \lVert f \rVert = 1 \ \land \ f(v) = \lVert v \rVert.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Define $u : \mathbb{C}v \to \mathbb{C}$ by $u(\lambda v) = \lambda \lVert v \rVert$. Then $\lvert u(t) \rvert \le \lVert t \rVert$ for all $t \in \mathbb{C}v$, and $u(v) = \lVert v \rVert$. By Hahn-Banach, extend $u$ to $f \in V'$ with $\lVert f(t) \rVert \le \lVert t \rVert$ for all $t \in V$. So $\lVert f \rVert \le 1$, and since $f(v) = \lVert v \rVert$, applying $f$ to $v / \lVert v \rVert$ gives $f(v/\lVert v \rVert) = 1$, so $\lVert f \rVert = 1$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Altenrative form of the theorem on Existence of Norm-Achieving Functionals)</span></p>

If $F$ is a closed subspace of $E$, then for all $y^\ast \in F^\ast$ there exists an $x^\ast \in E^\ast$ such that $x^\ast\mid\_F = y^\ast$ and $\lVert x^\ast\rVert = \lVert y^\ast\rVert$. This easily implies that for all $x \in E$ we have

$$\|x\| = \sup_{\|x^*\|\le 1} |\langle x,x^*\rangle|,$$

where by convention is $x^\ast$ is an element of the dual space and $x^\ast(x) = \lvert\langle x,x^\ast\rangle\rvert$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Norming linear subspace)</span></p>

A linear subspace $F$ of $E^\ast$ is called **norming** for a subset $S$ of $E$ if for all $x \in S$ we have

$$\|x\| = \sup_{\substack{x^*\in F\\ \|x^*\|\le 1}} |\langle x,x^*\rangle|.$$

A subspace of $E^\ast$ which is norming for $E$ is simply called **norming**. The following lemma will be used frequently.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dense Sequence)</span></p>

A **dense sequence** in a metric/normed space $E_0$ is a sequence $(x_n)\_{n=1}^\infty$ whose set of values is dense in $E_0$:

$$\overline{\lbrace x_n:n\ge 1\rbrace}=E_0.$$

Equivalently, for every $x\in E_0$ and every $\varepsilon>0$, there exists some $n$ such that

$$\|x-x_n\|<\varepsilon.$$

So the sequence gets arbitrarily close to every point of the space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Dense Sequence)</span></p>

$\mathbb Q\cap[0,1]$ can be enumerated as a sequence $(q_n)$, and that sequence is dense in $[0,1]$.

</div>

In the lemma below, separability of $E_0$ is used exactly here: by definition, a separable metric space has a countable dense subset, which we can enumerate as a sequence $(x_n)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Countable Norming Reduction)</span></p>

If $E_0$ is a separable subspace of $E$ and $F$ is a linear subspace of $E^\ast$ which is norming for $E_0$, then $F$ contains a sequence of unit vectors that is norming for $E_0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Choose a dense sequence $(x_n)\_{n=1}^\infty$ in $E_0$ and choose a sequence of unit vectors $(x_n^\ast)\_{n=1}^\infty$ in $F$ such that

$$|\langle x_n,x_n^*\rangle|\ge(1-\varepsilon_n)\|x_n\|$$

for all $n\ge 1$, where the numbers $0<\varepsilon_n\le 1$ satisfy

$$\lim_{n\to\infty}\varepsilon_n=0.$$

The sequence $(x_n^\ast)\_{n=1}^\infty$ is norming for $E_0$. To see this, fix an arbitrary $x\in E_0$ and let $\delta>0$. Pick $n_0\ge 1$ such that

$$0<\varepsilon_{n_0}\le\delta\qquad\text{and}\qquad\|x-x_{n_0}\|\le\delta.$$

Then,

$$
\begin{aligned}
(1-\delta)\|x\|
&\le
(1-\varepsilon_{n_0})\|x\| \\
&\le
(1-\varepsilon_{n_0})\|x_{n_0}\|
+
(1-\varepsilon_{n_0})\delta \\
&\le
|\langle x_{n_0},x_{n_0}^*\rangle|
+
\delta \\
&\le
|\langle x,x_{n_0}^*\rangle|
+
2\delta.
\end{aligned}
$$

Since $\delta>0$ was arbitrary it follows that

$$\|x\|\le\sup_{n\ge 1}|\langle x,x_n^*\rangle|.$$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Separation of the points by a linear subspace)</span></p>

A linear subspace $F$ of $E^\ast$ is said to **separate the points** of a subset $S$ of $E$ if for every pair $x,y\in S$ with $x \neq y$ there exists an $x^\ast \in F$ with $\langle x,x^\ast\rangle \neq \langle y,x^\ast\rangle$. Clearly, norming subspaces separate points, but the converse need not be true.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lindelöf Property)</span></p>

A metric space $X = (X, d)$ is said to be **Lindelöf** or to have the **Lindelöf property** if every cover of $X$ has a countable subcover.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Separability, Countable Basis, Lindelöf)</span></p>

The following statements about a metric space $X$ are equivalent.

1. $X$ is separable.
2. $X$ has a countable basis.
3. $X$ has the Lindelöf property.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(1)$\Rightarrow$(2):** Let $X$ be separable and let $M$ be a countable dense subset. Set

$$\mathcal{B} = \lbrace \Omega(m, r) \mid m \in M,\; r \text{ rational} \rbrace.$$

$\mathcal{B}$ is obviously countable; we show it is a basis. Let $U$ be open and let $x \in U$. Then there is an $\varepsilon > 0$ such that $\Omega(x, \varepsilon) \subseteq U$. Choose an $m\_x \in M$ and a rational $r\_x$ such that $d(x, m\_x) < \tfrac{1}{3}\varepsilon$ and $\tfrac{1}{3}\varepsilon < r\_x < \tfrac{2}{3}\varepsilon$. Then

$$x \in \Omega(m_x, r_x) \subseteq \Omega(x, \varepsilon) \subseteq U.$$

Indeed, $x \in \Omega(m\_x, r\_x)$ trivially and if $y \in \Omega(m\_x, r\_x)$ then $d(x, y) \le d(x, m\_x) + d(m\_x, y) < \tfrac{1}{3}\varepsilon + \tfrac{2}{3}\varepsilon = \varepsilon$. Thus $U = \bigcup \lbrace \Omega(m\_x, r\_x) \mid x \in U \rbrace$.

**(2)$\Rightarrow$(3):** Let $\mathcal{B}$ be a countable basis and let $\mathcal{U}$ be a cover of $X$. Since $U = \bigcup \lbrace B \mid B \in \mathcal{B},\; B \subseteq U \rbrace$ for each $U \in \mathcal{U}$ we have

$$X = \bigcup \lbrace B \in \mathcal{B} \mid \exists U_B \supseteq B,\; U_B \in U \rbrace.$$

The cover $\mathcal{A} = \lbrace B \in \mathcal{B} \mid \exists U\_B \supseteq B,\; U\_B \in U \rbrace$ is countable and hence so is also the cover $\mathcal{V} = \lbrace U\_B \mid B \in \mathcal{A} \rbrace$.

**(3)$\Rightarrow$(1):** Let $X$ be Lindelöf. For covers $\mathcal{U}\_n = \lbrace \Omega(x, \tfrac{1}{n}) \mid x \in X \rbrace$, choose countable subcovers

$$\Omega(x_{n1}, \tfrac{1}{n}),\; \Omega(x_{n2}, \tfrac{1}{n}),\; \dots,\; \Omega(x_{nk}, \tfrac{1}{n}),\; \dots$$

Then $\lbrace x\_{nk} \mid n = 1, 2, \dots,\; k = 1, 2, \dots \rbrace$ is dense. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

1. In more general **topological spaces**, the equivalence of Theorem above does not hold. The existence of a countable basis implies both separability and the Lindelöf property, but none of the other implications hold in general.
2. A countable basis is inherited by every subspace, so for metric spaces:
   - every subspace of a separable space is separable, and
   - every subspace of a Lindelöf space is Lindelöf.

   The latter statement is somewhat surprising — compare with compactness, which is inherited by closed subspaces only.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Countable Separating Reduction)</span></p>

If $E_0$ is a separable subspace of $E$ and $F$ is a linear subspace of $E^\ast$ which separates the points of $E_0$, then $F$ contains a sequence that separates the points of $E_0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

By the Hahn–Banach theorem, for each $x \in E_0 \setminus \lbrace 0\rbrace$ there exists a vector $x^\ast(x) \in F$ such that

$$\langle x, x^*(x)\rangle \neq 0.$$

Defining

$$V_x := \lbrace y \in E_0 \setminus \lbrace 0\rbrace : \langle y, x^*(x)\rangle \neq 0\rbrace,$$

we obtain an open cover

$$\lbrace V_x\rbrace_{x \in E_0 \setminus \lbrace 0\rbrace}$$

of $E_0 \setminus \lbrace 0\rbrace$.

Since every open cover of a separable metric space admits a countable subcover, it follows that there exists a sequence $(x_n)\_{n=1}^\infty$ in $E_0 \setminus \lbrace 0\rbrace$ such that

$$\lbrace V_{x_n}\rbrace_{n=1}^\infty$$

covers $E_0 \setminus \lbrace 0\rbrace$.

Then the sequence

$$\lbrace x^*(x_n)\rbrace_{n=1}^\infty$$

separates the points of $E_0$: indeed, every $x \in E_0 \setminus \lbrace 0\rbrace$ belongs to some $V\_{x_n}$, which means that

$$\langle x, x^*(x_n)\rangle \neq 0.$$

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

A bounded linear operator $T \in \mathcal{B}(V, W)$ is **isometric** if for all $v \in V$,

$$\lVert Tv \rVert = \lVert v \rVert.$$

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

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation I</span><span class="math-callout__name">(We need Lebesgue integration)</span></p>

With the general discussion of Banach spaces concluded, we now move to **Lebesgue measure and integration**. We have been talking about $\ell^p$ spaces on sequences; now we want to define $L^p$ spaces on functions in a similar way. Riemann integration is not sufficient — Lebesgue integration has better convergence theorems and is more widely useful.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation II</span><span class="math-callout__name"></span></p>

Our goal is to make a new definition of integration that is more general than Riemann integration: it will still be a method of calculating area under a curve, but built up in a way that allows for more powerful formalism. The approach: 
1. Start with **indicator functions** $1_E$ (equal to $1$ on $E$ and $0$ otherwise),
2. Define the integral of $1_E$ to be the **measure** $m(E)$ of $E$,
3. Build up from there.

</div>

### Desired Properties of a Measure

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation III</span><span class="math-callout__name">(Desired Properties of a Measure)</span></p>

We want a measure $m$ on subsets of $\mathbb{R}$ with these properties:

1. Defined on all subsets: $m : \mathcal{P}(\mathbb{R}) \to [0, \infty]$.
2. For any interval $I$, $m(I) = \ell(I)$ (the length of $I$).
3. Countable additivity: if $\lbrace E_n \rbrace$ are pairwise disjoint, $m(\bigcup_n E_n) = \sum_n m(E_n)$.
4. Translation invariance: $m(x + E) = m(E)$ for all $x \in \mathbb{R}$.

Unfortunately, **no function** $m : \mathcal{P}(\mathbb{R}) \to [0, \infty]$ satisfies all four simultaneously (this is shown by the **Vitali construction**). The resolution: drop property (1) and define $m$ only on a well-behaved collection of **Lebesgue measurable sets**.

</div>

### Outer Measure

The strategy (from Caratheodory): first define an **outer measure** $m^*$ on all subsets, satisfying (2), (4), and "almost (3)," then restrict to measurable sets.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Outer Measure)</span></p>

For any interval $I \subset \mathbb{R}$, let $\ell(I)$ denote its length. For any subset $A \subset \mathbb{R}$, the **outer measure** is

$$m^*(A) = \inf \left\lbrace \sum_n \ell(I_n) : \lbrace I_n \rbrace \text{ countable collection of open intervals with } A \subset \bigcup_n I_n \right\rbrace.$$

We always have $m^\ast(A) \ge 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two subtleties of Measure)</span></p>

1. The amount of the intervals $\lbrace I_n\rbrace$ is **countable**, not finite:
   * **Jordan Outer Measure:** Only permits a finite number of rectangles to cover the set.
   * **Lebesgue Outer Measure:** Allows an infinitely countable number of rectangles to cover the set.
2. We use **intervals** to cover $A$, not sets. 

</div>

<figure class="math-figure">
  <svg viewBox="0 0 480 150" xmlns="http://www.w3.org/2000/svg" width="500" aria-label="Outer measure: covering a set A by open intervals">
    <line x1="20" y1="100" x2="460" y2="100" stroke="#444" stroke-width="1.2" />
    <polygon points="460,100 452,96 452,104" fill="#444" />
    <text x="450" y="120" font-size="11" fill="#666">ℝ</text>
    <g fill="#2c4994">
      <circle cx="80"  cy="100" r="3" />
      <circle cx="110" cy="100" r="3" />
      <circle cx="155" cy="100" r="3" />
      <circle cx="175" cy="100" r="3" />
      <circle cx="225" cy="100" r="3" />
      <circle cx="280" cy="100" r="3" />
      <circle cx="305" cy="100" r="3" />
      <circle cx="345" cy="100" r="3" />
      <circle cx="395" cy="100" r="3" />
    </g>
    <text x="232" y="135" font-size="13" font-weight="600" fill="#2c4994">A ⊂ ℝ</text>
    <g stroke="#d65336" stroke-width="2" fill="rgba(214,83,54,0.10)">
      <rect x="65"  y="60" width="60"  height="20" rx="10" />
      <rect x="140" y="60" width="55"  height="20" rx="10" />
      <rect x="205" y="60" width="40"  height="20" rx="10" />
      <rect x="265" y="60" width="55"  height="20" rx="10" />
      <rect x="325" y="60" width="40"  height="20" rx="10" />
      <rect x="380" y="60" width="35"  height="20" rx="10" />
    </g>
    <text x="83"  y="55" font-size="11" fill="#d65336">I₁</text>
    <text x="158" y="55" font-size="11" fill="#d65336">I₂</text>
    <text x="218" y="55" font-size="11" fill="#d65336">I₃</text>
    <text x="283" y="55" font-size="11" fill="#d65336">I₄</text>
    <text x="338" y="55" font-size="11" fill="#d65336">I₅</text>
    <text x="390" y="55" font-size="11" fill="#d65336">I₆</text>
    <g stroke="#444" stroke-width="0.8">
      <line x1="65"  y1="95" x2="65"  y2="105" />
      <line x1="125" y1="95" x2="125" y2="105" />
      <line x1="140" y1="95" x2="140" y2="105" />
      <line x1="195" y1="95" x2="195" y2="105" />
    </g>
    <text x="20" y="35" font-size="11" fill="#666" font-style="italic">m*(A) = inf Σ ℓ(Iₙ) over all such coverings</text>
  </svg>
  <figcaption>Outer measure approximates a set $A$ from above: any countable collection of open intervals covering $A$ gives an upper bound on $m^*(A)$ via the sum of their lengths. We then take the infimum over all such coverings, allowing intervals to overlap, repeat, or shrink onto isolated points.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Every set has an outer measure)</span></p>

$$\boxed{\text{Every set has an outer measure.}}$$

By definition, an outer measure (such as the standard Lebesgue outer measure) is a function assigned to every subset of a given space. Its value can be any non-negative real number or **infinity** $(\infty)$.

While every set has an outer measure, not every set is **measurable**. In classical measure theory, a set is measurable if its outer measure and inner measure are equal. Sets that fail this condition, like the classic **Vitali set**, are referred to as "non-measurable sets". Therefore, the issue is not whether a set has an outer measure, but whether that outer measure behaves like a consistent, additive "volume."

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Outer Measure of a Point)</span></p>

$$m^\ast(\lbrace 0 \rbrace) = 0$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Cover $\lbrace 0 \rbrace$ with $(-\varepsilon/2, \varepsilon/2)$ of length $\varepsilon$ for any $\varepsilon > 0$, and take $\varepsilon \to 0$. The length of the interval convering $\lbrace 0 \rbrace$ is zero (goes to zero).

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Countable Sets Have Measure Zero)</span></p>

$$A \subset \mathbb{R} \quad\text{is countable}\quad \implies m^*(A) = 0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Enumerate 

$$A = \lbrace a_n : n \in \mathbb{N} \rbrace.$$

For each $n$, let 

$$I_n = (a_n - \varepsilon/2^{n+1}, a_n + \varepsilon/2^{n+1}),$$

which has length $\varepsilon/2^n$. Then 

$$A \subset \bigcup_n I_n \implies m^*(A) \le \sum_n \varepsilon/2^n = \varepsilon.$$

Taking $\varepsilon \to 0$ gives $m^\ast(A) = 0$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary </span><span class="math-callout__name">($\mathbb{Q}$ has outer measure zero)</span></p>

$$m^*(\mathbb{Q}) = 0,$$

even though $\mathbb{Q}$ is dense in $\mathbb{R}$.

</div>

### Properties of Outer Measure

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Monotonicity of Outer Measure)</span></p>

$$A \subset B \implies m^\ast(A) \le m^\ast(B).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Countable Subadditivity)</span></p>

$$\lbrace A_n \rbrace_{n\in\mathbb{N}} \subset \mathbb{R} \implies m^*\!\left(\bigcup_n A_n\right) \le \sum_n m^*(A_n),$$

where a countable collection $A_n \rbrace_{n\in\mathbb{N}} \subset \mathbb{R}$ is not necessarily disjoint.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We may assume all $m^\ast(A_n)$ are finite and $\sum_n m^\ast(A_n) < \infty$. Fix $\varepsilon > 0$. For each $n$, choose a covering $\lbrace I_{nk} \rbrace_{k \in \mathbb{N}}$ of $A_n$ with 

$$\sum_k \ell(I_{nk}) < m^\ast(A_n) + \varepsilon/2^n.$$

Then $\lbrace I_{nk} \rbrace_{n,k}$ covers $\bigcup_n A_n$, so
    
$$m^*\!\left(\bigcup_n A_n\right) \le \sum_{n,k} \ell(I_{nk}) = \sum_n \sum_k \ell(I_{nk}) < \sum_n \left(m^*(A_n) + \frac{\varepsilon}{2^n}\right) = \sum_n m^*(A_n) + \varepsilon.$$

Taking $\varepsilon \to 0$ gives the result.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Failure of Equality in Countable Subadditivity for Non-Measurable Sets)</span></p>

Even if the sequence of sets $\lbrace A_n\rbrace$ is perfectly **disjoint**, you can absolutely still end up with a strict inequality:

$$m^*\!\left(\bigcup_n A_n\right) < \sum_n m^*(A_n)$$

Disjointness alone is not enough to guarantee equality. To get that exact equality (which is called **Countable Additivity**), the sets must be disjoint *and* they must be **Lebesgue measurable**.

Here is exactly why that happens, and how it connects back to the core definition of measurability.

Because the Lebesgue outer measure $m^\ast$ is only **subadditive**, it fundamentally overestimates the "size" of highly pathological, non-measurable sets. When non-measurable sets are disjoint, they can be so infinitely tangled together that their combined union actually has a strictly smaller outer measure than the sum of their individual outer measures.

it could be proven that strict inequality happens using the exact logic from Carathéodory's criterion.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Outer Measure of Intervals)</span></p>

$$I \text{ is interval in } \mathbb{R} \implies m^\ast(I) = \ell(I).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

* $m^\ast(I) \le \ell(I)$
 
  For a closed bounded interval $[a, b]$, cover it by $(a - \varepsilon, b + \varepsilon)$, giving $m^\ast(I) \le \ell(I) + 2\varepsilon$ for any $\varepsilon > 0$.

* $\ell(I) \le m^\ast(I)$

  Suppose $\lbrace I_n \rbrace$ is a collection of open intervals covering $[a, b]$. By Heine-Borel, a finite subcollection $\lbrace J_1, \dots, J_N \rbrace$ suffices. Rearrange so that $J_i$ and $J_{i+1}$ always overlap. Then 

  $$\sum \ell(I_n) \ge \sum_{k=1}^{K} \ell(J_k) \ge b - a = \ell(I)$$
  
  by a telescoping argument on the overlapping intervals.

* Other interval types follow by approximation: 
  
  $$[a + \varepsilon, b - \varepsilon] \subset I \subset [a - \varepsilon, b + \varepsilon].$$
  
  Infinite intervals have infinite outer measure.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Approximation by Open Sets)</span></p>

For every subset $A \subset \mathbb{R}$ and $\varepsilon > 0$, there exists an open set $O$ such that $A \subset O$ and 

$$m^\ast(A) \le m^\ast(O) \le m^\ast(A) + \varepsilon$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

* If $m^\ast(A) = \infty$, take $O = \mathbb{R}$. 
* Otherwise, let $\lbrace I_n \rbrace$ be open intervals covering $A$ with the following property (possible, because outer measure is defined as an infimum of the sum of the interval length covering $A$)
  
  $$\sum \ell(I_n) \le m^\ast(A) + \varepsilon.$$
  
  Then $O = \bigcup_n I_n$ is open, $A \subset O$, and by subadditivity 
  
  $$m^\ast(O) \le \sum m^\ast(I_n) \le \sum \ell(I_n) \le m^\ast(A) + \varepsilon.$$

</details>
</div>

### Lebesgue Measurable Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Measurable)</span></p>

A set $E \subset \mathbb{R}$ is **Lebesgue measurable** if for all $A \subset \mathbb{R}$,

$$m^*(A) = \underbrace{m^*(A \cap E)}_{\text{overlap out. measure}} + \underbrace{m^*(A \cap E^c)}_{\text{A's own out. measure}}.$$

In other words, $E$ is well-behaved in that it always cuts any set $A$ into reasonable parts. Since subadditivity gives 

$$m^\ast(A) \le m^\ast(A \cap E) + m^\ast(A \cap E^c)$$

for free, measurability amounts to showing the reverse inequality.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Having Outer Measure $\neq$ Be Measurable)</span></p>

$$\boxed{\text{Measurability of E = Decomposes Outer Measure of all open sets into two parts.}}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Basic Measurability Facts)</span></p>

* The empty set $\varnothing$ and $\mathbb{R}$ are measurable. 
* A set $E$ is measurable if and only if $E^c$ is measurable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Zero Outer Measure Implies Measurable)</span></p>

$$m^\ast(E) = 0 \implies E \quad\text{is measurable}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $A \cap E \subset E$, monotonicity gives 

$$m^\ast(A \cap E) \le m^\ast(E) = 0.$$

So 

$$m^\ast(A \cap E) + m^\ast(A \cap E^c) = m^\ast(A \cap E^c) \le m^\ast(A)$$

since $A \cap E^c \subset A$. This is the measurability condition.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Finite Union of Measurable Sets)</span></p>

$$E_1, E_2 \quad\text{are measurable sets} \implies E_1 \cup E_2 \quad\text{is measurable}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $A \subset \mathbb{R}$ be arbitrary. Since $E_2$ is measurable, 

$$m^*(A \cap E_1^c) = m^*(A \cap E_1^c \cap E_2) + m^*(A \cap E_1^c \cap E_2^c).$$

Note by de Morgan 

$$E_1^c \cap E_2^c = (E_1 \cup E_2)^c$$

$$A \cap (E_1 \cup E_2) = (A \cap E_1) \cup (A \cap E_2 \cap E_1^c).$$

So 

$$m^*(A \cap (E_1 \cup E_2)) \le m^*(A \cap E_1) + m^*(A \cap E_2 \cap E_1^c).$$

Using measurability of $E_1$: 

$$m^*(A \cap E_1) + m^*(A \cap E_1^c) = m^*(A).$$

Substituting and rearranging gives 

$$m^*(A \cap (E_1 \cup E_2)) + m^*(A \cap (E_1 \cup E_2)^c) \le m^*(A)$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Finite Unions are Measurable)</span></p>

$$E_1, \dots, E_n \quad\text{are measurable}\implies \bigcup_{k=1}^{n} E_k \quad\text{is measurable}$$

</div>

So far we only know that *finite* unions of measurable sets are measurable. That **open and closed sets are themselves measurable** — and hence that the Borel $\sigma$-algebra sits inside $\mathcal{M}$ — is proved later, once we show that half-lines are measurable; see the Theorem *(The Borel $\sigma$-Algebra is Measurable)* in the [Lebesgue Measure](#lebesgue-measure) section below.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Not all sets are measurable: Vitali set)</span></p>

The Vitali set ([my post about it](subpages/stuff_i_found_interesting/vitali_set/index.md)) is the classic example of a **non-measurable set of real numbers**. It was constructed by Italian mathematician Giuseppe Vitali in 1905 to prove that it is impossible for the Lebesgue measure to assign a meaningful "length" to every single subset of real numbers without sacrificing basic geometric intuition.

Because assigning any real number or zero to the measure of the Vitali set results in a logical contradiction ($1 \le 0 \le 3$ or $1 \le \infty \le 3$), the Vitali set cannot be assigned an actual measure. It is strictly non-measurable, though its Lebesgue outer measure is known to be greater than zero.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Measurable set is nearly a finite union of open intervals)</span></p>

Let $E \subset \mathbb{R}$, and assume that $m^\ast(E) \le \infty$. Prove that $E$ is measurable if and only if for every $\varepsilon > 0$ there exists a finite union of open intervals $U$ such that $m^\ast(U\Delta E) < \varepsilon$.

*Hint:* To prove the converse direction, let $A \subset\mathbb R$, and prove that for every $\varepsilon > 0$,

$$m^* (A \cap E) + m^*(A \cap E^c) \leq m^*(A) + \varepsilon.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**Direction $\Rightarrow$:**

Assume $E$ is measurable and $m^*(E) < \infty$. Let $\varepsilon > 0$.

**1. Bounding the Overestimate**
By the definition of outer measure (approximation by open sets), there exists a countable collection of open intervals $U_i$ such that $E \subset \bigcup_{i=1}^\infty U_i$ and:

$$m^*\left(\bigcup_{i=1}^\infty U_i\right) < m^*(E) + \frac{\varepsilon}{2}$$

Let $O = \bigcup_{i=1}^\infty U_i$. Because $E$ is Lebesgue measurable, Carathéodory's criterion guarantees that:

$$m^*(O) = m^*(O \cap E) + m^*(O \cap E^c) = m^*(E) + m^*(O \setminus E)$$

Substituting this into our approximation bound yields:

$$m^*(E) + m^*(O \setminus E) < m^*(E) + \frac{\varepsilon}{2}$$

Because $m^*(E) < \infty$, we can subtract it from both sides to obtain:

$$m^*(O \setminus E) < \frac{\varepsilon}{2}$$

**2. Bounding the Tail**
Since $m^\ast(E)$ is finite, $m^\ast(O)$ is also finite. By the continuity of measure from below (or the convergence of the sum of lengths), we can find a finite integer $n$ such that the measure of the remaining tail is small. Let $U = \bigcup_{i=1}^n U_i$. Then $O \setminus U \subset \bigcup_{i=n+1}^\infty U_i$, and we can choose $n$ large enough so that:

$$m^*(O \setminus U) < \frac{\varepsilon}{2}$$

**3. The Symmetric Difference**
We evaluate the symmetric difference $U \Delta E$:

$$m^*(U \Delta E) = m^*((U \setminus E) \cup (E \setminus U)) \le m^*(U \setminus E) + m^*(E \setminus U)$$

We bound each term separately:

* Since $U \subset O$, we have $U \setminus E \subset O \setminus E$. Thus, $m^\ast(U \setminus E) \le m^\ast(O \setminus E) < \frac{\varepsilon}{2}$.
* Since $E \subset O$, any element in $E$ that is not in $U$ must be in $O$ but not in $U$. Thus, $E \setminus U \subset O \setminus U$. This gives $m^\ast(E \setminus U) \le m^\ast(O \setminus U) < \frac{\varepsilon}{2}$.

Summing these bounds gives the final result:

$$m^*(U \Delta E) < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon$$

**Direction $\Leftarrow$:**

By the subadditivity of the outer measure, for any test set $A$:

$$m^*(A) = m^*((A \cap E) \cup (A \cap E^c)) \le m^*(A \cap E) + m^*(A \cap E^c)$$

To prove measurability via Carathéodory's criterion, we only need to prove the opposite inequality to establish equality:

$$m^*(A \cap E) + m^*(A \cap E^c) \le m^*(A)$$

The problem explicitly states we can assume $U$ (a finite union of open intervals) is measurable. Using $U$ as the measurable set and $A$ as the test set in Carathéodory's criterion, we have strict equality:

$$m^*(A) = m^*(A \cap U) + m^*(A \cap U^c)$$

We establish the subset bounds using the error region $U \Delta E$:

1. $A \cap E \subseteq (A \cap U) \cup (A \cap (U \Delta E))$

   $$\implies m^*(A \cap E) \le m^*(A \cap U) + m^*(A \cap (U \Delta E)) \le m^*(A \cap U) + \varepsilon$$

2. $A \cap E^c \subseteq (A \cap U^c) \cup (A \cap (U \Delta E))$

   $$\implies m^*(A \cap E^c) \le m^*(A \cap U^c) + m^*(A \cap (U \Delta E)) \le m^*(A \cap U^c) + \varepsilon$$

Summing these inequalities yields:

$$m^*(A \cap E) + m^*(A \cap E^c) \le m^*(A \cap U) + m^*(A \cap U^c) + 2\varepsilon$$

Substituting the Carathéodory equality for $U$:

$$m^*(A \cap E) + m^*(A \cap E^c) \le m^*(A) + 2\varepsilon$$

Because $\varepsilon > 0$ is arbitrary, as $\varepsilon \to 0$, we obtain:

$$m^*(A \cap E) + m^*(A \cap E^c) \le m^*(A)$$

Thus, $E$ is Lebesgue measurable.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Littlewood’s first principle)</span></p>

The result from the exercise above is known as **Littlewood’s first principle**

$$\boxed{\text{Every measurable set is nearly a finite union of open intervals.}}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Translation invariance of measurable sets)</span></p>

Let $E$ be a measurable set. Prove that for all $x \in \mathbb{R}$, $E + x$ is measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

By Carathéodory's criterion, a set $M$ is Lebesgue measurable if and only if for any arbitrary test set $A \subset \mathbb{R}$, the following holds:

$$m^*(A) = m^*(A \cap M) + m^*(A \cap M^c)$$

Let $A \subset \mathbb{R}$ be an arbitrary test set. Since $E$ is assumed to be measurable, the criterion must hold for the shifted test set $A-x$:

$$m^*(A-x) = m^*((A-x) \cap E) + m^*((A-x) \cap E^c)$$

Because the Lebesgue outer measure is translation-invariant, we immediately know $m^\ast(A-x) = m^\ast(A)$.

Next, we express the intersections on the right-hand side as global translations of sets. A point $y$ belongs to $(A-x) \cap E$ if and only if $y \in A-x$ and $y \in E$. This is equivalent to stating $y+x \in A$ and $y+x \in E+x$. Thus, we can factor the translation out of the intersection:

$$(A-x) \cap E = (A \cap (E+x)) - x$$

Applying the translation invariance of outer measure to this set yields:

$$m^*((A-x) \cap E) = m^*((A \cap (E+x)) - x) = m^*(A \cap (E+x))$$

We apply the same logic to the complement. Since set complements behave cleanly under translation, $(E+x)^c = E^c + x$. Therefore:

$$(A-x) \cap E^c = (A \cap (E^c+x)) - x = (A \cap (E+x)^c) - x$$

Taking the outer measure and applying translation invariance again:

$$m^*((A-x) \cap E^c) = m^*((A \cap (E+x)^c) - x) = m^*(A \cap (E+x)^c)$$

Finally, substituting these translation-invariant measures back into our initial Carathéodory equation gives:

$$m^*(A) = m^*(A \cap (E+x)) + m^*(A \cap (E+x)^c)$$

Because $A$ was entirely arbitrary, this satisfies the criterion. Thus, $E+x$ is measurable.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Scale invariance of measurable sets)</span></p>

Let $E$ be a measurable set. Prove that for all $r > 0$, $rE := \lbrace ry \mid  y \in E\rbrace$ is measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let $A \subset \mathbb{R}$ be an arbitrary test set. Since $E$ is measurable, we can apply Carathéodory's criterion using the scaled test set $\frac{1}{r}A$:

$$m^*\left(\frac{1}{r}A\right) = m^*\left(\frac{1}{r}A \cap E\right) + m^*\left(\frac{1}{r}A \cap E^c\right)$$

Recall that the Lebesgue outer measure scales linearly: for any set $S$ and positive constant $c$, $m^*(cS) = c \cdot m^*(S)$. Therefore, $m^*\left(\frac{1}{r}A\right) = \frac{1}{r}m^*(A)$.

Next, we rewrite the intersections by factoring out the scalar $\frac{1}{r}$. A point $y$ is in $\frac{1}{r}A \cap E$ if and only if $ry \in A$ and $ry \in rE$. This set equivalence can be written as:

$$\frac{1}{r}A \cap E = \frac{1}{r}(A \cap rE)$$

Applying the linear scaling property of outer measure gives:

$$m^*\left(\frac{1}{r}A \cap E\right) = m^*\left(\frac{1}{r}(A \cap rE)\right) = \frac{1}{r}m^*(A \cap rE)$$

Similarly, for the complement, notice that scaling and complements commute: $(rE)^c = r(E^c)$. Therefore:

$$\frac{1}{r}A \cap E^c = \frac{1}{r}(A \cap rE^c) = \frac{1}{r}(A \cap (rE)^c)$$

Taking the outer measure of this complement set yields:

$$m^*\left(\frac{1}{r}A \cap E^c\right) = m^*\left(\frac{1}{r}(A \cap (rE)^c)\right) = \frac{1}{r}m^*(A \cap (rE)^c)$$

Substitute these linearly scaled measures back into our initial Carathéodory equation:

$$\frac{1}{r}m^*(A) = \frac{1}{r}m^*(A \cap rE) + \frac{1}{r}m^*(A \cap (rE)^c)$$

Since $r > 0$, we can safely multiply the entire equation by $r$ to obtain:

$$m^*(A) = m^*(A \cap rE) + m^*(A \cap (rE)^c)$$

Because $A$ was arbitrary, this establishes Carathéodory's criterion. Thus, $rE$ is measurable.

</details>
</div>

### Algebras and $\sigma$-Algebras

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Algebra and $\sigma$-Algebra)</span></p>

* A nonempty collection $\mathcal{A} \subset \mathcal{P}(\mathbb{R})$ is an **algebra** on $\mathbb{R}$ if:
  1. $\mathbb{R} \in \mathcal{A}$;
  2. $E \in \mathcal{A}$, then $E^c := \mathbb{R} \setminus E \in \mathcal{A}$;
  3. $E_1, \dots, E_n \in \mathcal{A}$, then
     
     $$\bigcup_{k=1}^n E_k \in \mathcal{A}.$$

* An algebra $\mathcal{A}$ is a **$\sigma$-algebra** if it is additionally closed under countable unions: whenever $\lbrace E_n\rbrace_{n=1}^{\infty} \subset \mathcal{A}$,
  
  $$\bigcup_{n=1}^{\infty} E_n \in \mathcal{A}.$$

</div>

Algebras are closed under complements and finite unions; $\sigma$-algebras are additionally closed under countable unions. By de Morgan's laws, closure under countable unions and complements implies closure under countable intersections.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\sigma$-Algebras)</span></p>

* The simplest $\sigma$-algebra is 
  
  $$\mathcal{A} = \lbrace \varnothing, \mathbb{R} \rbrace$$

* The largest is 
  
  $$\mathcal{A} = \mathcal{P}(\mathbb{R})$$

* The **cocountable $\sigma$-algebra** 

  $$\mathcal{A} = \lbrace E \subset \mathbb{R} : E \text{ or } E^c \text{ is countable} \rbrace$$

  is another example.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Borel $\sigma$-Algebra)</span></p>

The intersection of all $\sigma$-algebras containing all open subsets of $\mathbb{R}$,

$$\mathcal{B} = \bigcap_{\mathcal{A} \in \Sigma} \mathcal{A}, \quad \text{where } \Sigma = \lbrace \mathcal{A} : \mathcal{A} \text{ is a } \sigma\text{-algebra containing all open subsets of } \mathbb{R} \rbrace,$$

is the smallest $\sigma$-algebra containing all open subsets of $\mathbb{R}$, called the **Borel $\sigma$-algebra**. It is closed under complements and countable unions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why is Borel $\sigma$-algebra defined as an intersection?)</span></p>

> Why is Borel $\sigma$-algebra defined as an intersection of all $\sigma$-algebras containing all real open sets, if we can just define Borel $\sigma$-algebra as a $\sigma$-algebra containing only all real open sets and not any other sets?

Because a **sigma-algebra cannot contain only open sets** unless the open sets are already closed under complements. On $\mathbb R$, they are not.

A sigma-algebra $\mathcal A$ on $\mathbb R$ must satisfy:

$$A\in \mathcal A \implies \mathbb R\setminus A\in \mathcal A,$$

and it must also be closed under countable unions.

So if $\mathcal A$ contains all open sets, then since

$$(0,1)\in \mathcal A,$$

it must also contain

$$\mathbb R\setminus (0,1)=(-\infty,0]\cup[1,\infty),$$

which is **not open**.

**Thus “the sigma-algebra containing only all open sets and no other sets” does not exist.**

The right idea is:

$$\mathcal B(\mathbb R) = \text{the smallest sigma-algebra containing all open subsets of }\mathbb R.$$

To define “smallest” rigorously, we take the intersection of all sigma-algebras containing the open sets:

$$\mathcal B(\mathbb R) := \bigcap_{\substack{\mathcal A\text{ sigma-algebra}\ \mathcal O(\mathbb R)\subseteq \mathcal A}} \mathcal A.$$

This works because an intersection of sigma-algebras is again a sigma-algebra.

So the Borel sigma-algebra contains exactly the sets that are **forced** to be present once we demand:

1. all open sets are measurable;
2. complements are measurable;
3. countable unions are measurable.

For example, once all open sets are included, we automatically get closed sets, because complements of open sets are closed. Then we also get countable unions of closed sets, countable intersections of open sets, countable sets like $\mathbb Q$, and many more.

So the point is:

$$\boxed{\text{Borel sets are not just open sets; they are everything generated from open sets by sigma-algebra operations.}}$$

The intersection definition is just the rigorous way to say “generated by the open sets, and nothing extra unless forced.”

</div>

### The Measurable Sets Form a $\sigma$-Algebra

<div class="math-callout math-callout--proposition" markdown="1">
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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Countable intersections of Lebesgue measurable sets are Lebesgue measurable)</span></p>

**Countable intersections** of Lebesgue measurable sets are Lebesgue measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The reason is that the Lebesgue measurable sets form a **$\sigma$-algebra**. By definition, a $\sigma$-algebra is closed under:

$$\text{complements},\qquad \text{countable unions}.$$

From these two closure properties, countable intersections follow by De Morgan’s law:

$$\bigcap_{n=1}^{\infty} E_n = \left(\bigcup_{n=1}^{\infty} E_n^c\right)^c.$$

If each $E_n$ is Lebesgue measurable, then each complement $E_n^c$ is Lebesgue measurable; the countable union $\bigcup_n E_n^c$ is Lebesgue measurable; and therefore its complement is also Lebesgue measurable.

So in the earlier step,

$$\lbrace f\ge \alpha\rbrace = \bigcap_{n=1}^{\infty}\lbrace f>\alpha-1/n\rbrace,$$

the right-hand side is measurable because it is a **countable intersection** of measurable sets. This is exactly the kind of closure property your notes use when working with Lebesgue measurable sets and preimages of functions. 

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Image of Lebesgue measurable sets forms $\sigma$-algebra)</span></p>

Let $f:\mathbb{R}\to\mathbb{R}$. Prove that the collection of sets

$$\mathcal{A} = \lbrace E \subset \mathbb{R} \mid f^{-1}(E) \text{ is Lebesgue measurable}\rbrace$$

is a $\sigma$-algebra.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

*Interesting observation:* the exercise is solvable without any knowledge of measurable functions.

1. $\emptyset \in \mathcal{A}$
  1. $f$ is a function from $\mathbb{R}$ to $\mathbb{R}$
  2. $\implies$ nothing is mapped to $\emptyset\not\in\mathbb{R}$
  3. $\implies$ $f^{-1}(\emptyset) = \emptyset$, which is Lebesgue measurable by Lemma (Basic Measurability Facts).
  
2. $E \in \mathcal{A} \implies E^c \in \mathcal{A}$
   1. Here it is useful to study the relationship between $(f^{-1}(E))^c$ and $f^{-1}(E^c)$.
   2. $f^{-1}(E)$ is Lebesgue measurable $\implies$ $(f^{-1}(E))^c$ is Lebesgue measurable by Lemma (Basic Measurability Facts).
   3. Because every $x\in (f^{-1}(E))^c$ is mapped to some $f(x)\in E^c$, we obtain $f^{-1}(E^c) = (f^{-1}(E))^c$.
   4. $\implies$ $E^c \in \mathcal{A}$. 

3. Countable $E_1, \dots \in \mathcal{A} \implies \Cup_{i=1} E_i \in \mathcal{A}$
   1. $f^{-1}(\Cup_{i=1} E_i) = \Cup_{i=1} f^{-1}(E_i)$
   2. Each $f^{-1}(E_i)$ is Lebesgue measurable
   3. The countable union of Lebesgue measurable is Lebesgue measurable
   4. $\implies$ $f^{-1}(Cup_{i=1} E_i)$ is Lebesgue measurable
   5. $\implies$ $\Cup_{i=1} E_i \in \mathcal{A}$

Old 2.3:

> Because every $x\in (f^{-1}(E))^c$ is mapped to some $f(x)\in E^c$, we obtain $f^{-1}(E^c) = (f^{-1}(E))^c \cup \lbrace \emptyset \rbrace$
> * Note that the function $f$ is not necessarily surjective, i.e. some values of $E^c$ might have an empty set preimage. 
> * If $f$ is sujective, then $f^{-1}(E^c) = (f^{-1}(E))^c$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Misconceptions in Step 3: $f^{-1}(E^c) = (f^{-1}(E))^c$)</span></p>

In the initial solution, where was a couple of set-theory misconceptions in Step 3 regarding the empty set and surjectivity.
* First, a preimage $f^{-1}(E^c)$ is a collection of elements from the **domain** $\mathbb{R}$. The empty set $\emptyset$ is a set, not a real number, so we wouldn't take the union with the set $\lbrace \emptyset\rbrace$.
* Second, my concern about surjectivity is a common trap, but it actually doesn't affect the preimage here. If there is some value $y \in E^c$ that is **never** hit by the function $f$, its individual preimage is just empty. It simply contributes **nothing** to the overall set $f^{-1}(E^c)$. It doesn't break the equality.

In fact, the relationship 

$$\boxed{f^{-1}(E^c) = (f^{-1}(E))^c}$$ 

is a universal set theory identity that is strictly true for **all** functions, regardless of whether they are surjective!

</div>

### Lebesgue Measure

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Lebesgue Measure)</span></p>

We now know that $\mathcal{M}$ is a $\sigma$-algebra of "well-behaved" sets, but two things are still missing: 
* First, we have not verified that $\mathcal{M}$ actually contains the sets we care about (open sets, closed sets, and everything built from them). 
* Second, the object we have been carrying around, the outer measure $m^\ast$, is only **subadditive** on arbitrary sets; we have not recorded the **exact** additivity and the two convergence properties that make it an honest measure once restricted to $\mathcal{M}$. 

We settle both now.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Half-Lines are Measurable)</span></p>

For every $a \in \mathbb{R}$, the half-line $(a, \infty)$ is Lebesgue measurable.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Fix an arbitrary $A \subset \mathbb{R}$; we may assume $m^\ast(A) < \infty$ (otherwise the measurability inequality is trivial). Write $A_1 = A \cap (a, \infty)$ and $A_2 = A \cap (-\infty, a]$. Given $\varepsilon > 0$, choose open intervals $\lbrace I_n \rbrace$ with $A \subset \bigcup_n I_n$ and $\sum_n \ell(I_n) \le m^\ast(A) + \varepsilon$ (possible by the definition of outer measure as an infimum).</p>
    <p>Split each cover interval at the point $a$: let $I_n' = I_n \cap (a, \infty)$ and $I_n'' = I_n \cap (-\infty, a)$. Each is an interval (or empty), and since they partition $I_n$ up to the single point $a$, their lengths satisfy $\ell(I_n') + \ell(I_n'') = \ell(I_n)$. The collection $\lbrace I_n' \rbrace$ covers $A_1$ and $\lbrace I_n'' \rbrace$ covers $A_2$, so by the definition of outer measure and the Theorem (Countable Subadditivity),</p>
    $$m^\ast(A) \leq m^\ast(A_1) + m^\ast(A_2) \le \sum_n \ell(I_n') + \sum_n \ell(I_n'') = \sum_n \ell(I_n) \le m^\ast(A) + \varepsilon.$$
    <p>Letting $\varepsilon \to 0$ yields $m^\ast(A \cap (a, \infty)) + m^\ast(A \cap (a, \infty)^c) \le m^\ast(A)$, which is exactly the measurability condition for $(a, \infty)$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Borel $\sigma$-Algebra is Measurable)</span></p>

Every open set is Lebesgue measurable. Consequently the Borel $\sigma$-algebra is contained in the Lebesgue $\sigma$-algebra:

$$\mathcal{B} \subset \mathcal{M}.$$

In particular every open, closed, $F_\sigma$, and $G_\delta$ set is measurable.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>By the previous theorem $(a, \infty) \in \mathcal{M}$, and since $\mathcal{M}$ is closed under complements, $(-\infty, a] \in \mathcal{M}$. Then</p>
    $$(-\infty, b) = \bigcup_{n=1}^{\infty} \left(-\infty, \, b - \tfrac{1}{n}\right] \in \mathcal{M}$$
    <p>is a countable union of measurable sets, and</p>
    $$(a, b) = (a, \infty) \cap (-\infty, b) \in \mathcal{M}$$
    <p>is a finite intersection of measurable sets (a $\sigma$-algebra is closed under finite intersections, being closed under complements and countable unions). Every open subset of $\mathbb{R}$ is a countable union of open intervals, so every open set is measurable. Thus $\mathcal{M}$ is a $\sigma$-algebra containing every open set; since the Borel $\sigma$-algebra $\mathcal{B}$ is by definition the *smallest* such $\sigma$-algebra, $\mathcal{B} \subset \mathcal{M}$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Measurable sets are a strictly richer collection than the Borel sets)</span></p>

This is what justifies the claim made earlier that "every open set and every closed set is measurable" — and it means the **measurable sets are a strictly richer collection than the Borel sets**, since $\mathcal{M}$ also contains every set of outer measure zero.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Measure)</span></p>

For a measurable set $E \in \mathcal{M}$, the **Lebesgue measure** of $E$ is its outer measure,

$$m(E) := m^\ast(E).$$

That is, $m$ is the outer measure $m^\ast$ with its domain restricted from all of $\mathcal{P}(\mathbb{R})$ down to $\mathcal{M}$.

</div>

Restricting the symbol $m$ to measurable sets is precisely what buys us the exact additivity that $m^\ast$ fails to have on arbitrary sets.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Monotonicity and Intervals)</span></p>

* If $A, B \in \mathcal{M}$ with $A \subset B$, then $m(A) \le m(B)$. 
* Moreover, every interval $I$ is measurable with $m(I) = \ell(I)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch</summary>

* Monotonicity is inherited directly from the Lemma (Monotonicity of Outer Measure). 
* The interval formula combines the Proposition (Outer Measure of Intervals), which gives $m^\ast(I) = \ell(I)$, with the fact that intervals are Borel and hence measurable.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Countable Additivity of Lebesgue Measure)</span></p>

Let $\lbrace E_n \rbrace$ be a countable collection of **pairwise disjoint** measurable sets. Then

$$m\!\left(\bigcup_{n=1}^{\infty} E_n\right) = \sum_{n=1}^{\infty} m(E_n).$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>The Theorem (Countable Subadditivity) gives "$\le$" for free, so it remains to prove "$\ge$". Applying the Proposition (Finite Additivity for Measurable Sets) with test set $A = \mathbb{R}$ shows that for every $N$,</p>
    $$m\!\left(\bigcup_{n=1}^{N} E_n\right) = \sum_{n=1}^{N} m(E_n).$$
    <p>Since $\bigcup_{n=1}^{N} E_n \subset \bigcup_{n=1}^{\infty} E_n$, monotonicity gives</p>
    $$m\!\left(\bigcup_{n=1}^{\infty} E_n\right) \ge \sum_{n=1}^{N} m(E_n) \qquad \text{for every } N.$$
    <p>Letting $N \to \infty$ yields $m\!\left(\bigcup_{n=1}^{\infty} E_n\right) \ge \sum_{n=1}^{\infty} m(E_n)$, completing the proof. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Lebesgue integral is countably additive)</span></p>

This is exactly the *exact additivity* that we listed among the desired properties of a measure but could not obtain for $m^\ast$ on general sets — disjointness alone was not enough; the sets also had to be measurable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Inclusion-exclusion principle of Lebesgue measure)</span></p>

Prove that if $E$ and $F$ are measurable sets, then

$$m(E \cup F) + m(E \cap F) = m(E) + m(F).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

By breaking the sets down into disjoint components, we can bypass the subadditivity of the outer measure and directly leverage the finite additivity property of the Lebesgue measure.

**Step 1: Define the disjoint components**

Let us decompose $E$ and $F$ into three pairwise disjoint sets based on your logic:

* $B = E \cap F$ (the intersection)
* $A = E \setminus F = E \cap F^c$ (the part of $E$ exclusively outside of $F$)
* $C = F \setminus E = F \cap E^c$ (the part of $F$ exclusively outside of $E$)

**Step 2: Establish measurability of the components**

Because $E$ and $F$ are given as Lebesgue measurable sets, and the collection of Lebesgue measurable sets forms a $\sigma$-algebra, they are closed under complements and finite intersections. Therefore, the sets $A$, $B$, and $C$ are all perfectly measurable.

**Step 3: Express the original sets as disjoint unions**

We can now reconstruct our original sets using disjoint unions of these new components:

* $E = A \cup B$
* $F = C \cup B$
* $E \cup F = A \cup B \cup C$

**Step 4: Apply finite additivity**

Because $A$, $B$, and $C$ are pairwise disjoint and measurable, we can apply the finite additivity property of the Lebesgue measure to our equations from Step 3:

$$m(E) = m(A) + m(B)$$

$$m(F) = m(C) + m(B)$$

$$m(E \cup F) = m(A) + m(B) + m(C)$$

**Step 5: Algebraic substitution**

Now, we evaluate the left side of the target equation by substituting in our expanded measures:

$$m(E \cup F) + m(E \cap F) = (m(A) + m(B) + m(C)) + m(B)$$

Since all measures yield non-negative values in $[0, \infty]$, we can freely rearrange the terms (this algebraic rearrangement remains valid even if any of the sets have an infinite measure):

$$m(E \cup F) + m(E \cap F) = (m(A) + m(B)) + (m(C) + m(B))$$

Finally, substitute $m(E)$ and $m(F)$ back into the right side of the equation:

$$m(E \cup F) + m(E \cap F) = m(E) + m(F)$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Continuity of Measure from Below)</span></p>

Let $\lbrace E_k \rbrace$ be a countable collection of measurable sets that is increasing, $E_1 \subset E_2 \subset \cdots$. Then

$$m\!\left(\bigcup_{k=1}^{\infty} E_k\right) = \lim_{n \to \infty} m(E_n).$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Disjointify the tower: set $F_1 = E_1$ and $F_k = E_k \setminus E_{k-1}$ for $k \ge 2$. Because $\mathcal{M}$ is a $\sigma$-algebra, each $F_k$ is measurable, and the $F_k$ are pairwise disjoint with</p>
    $$\bigcup_{k=1}^{n} F_k = E_n \qquad \text{and} \qquad \bigcup_{k=1}^{\infty} F_k = \bigcup_{k=1}^{\infty} E_k.$$
    <p>By countable additivity applied twice (to the finite and the infinite disjoint unions),</p>
    $$m\!\left(\bigcup_{k=1}^{\infty} E_k\right) = \sum_{k=1}^{\infty} m(F_k) = \lim_{n \to \infty} \sum_{k=1}^{n} m(F_k) = \lim_{n \to \infty} m(E_n). \quad\square$$
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Continuity of measure is essential in measure-theoretic integration)</span></p>

This is the statement invoked (by the name "continuity of measure") in the proofs of the Monotone Convergence Theorem and the Zero-Integral Characterization further below.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Continuity of measure from Above Needs Finite Measure)</span></p>

The dual statement for a **decreasing** tower $E_1 \supset E_2 \supset \cdots$, namely

$$m\left(\bigcap_k E_k\right) = \lim_n m(E_n),$$

is **not** true without the extra hypothesis $m(E_1) < \infty$. The standard counterexample is $E_n = [n, \infty)$: here $\bigcap_n E_n = \varnothing$, so the left side is $0$, yet $m(E_n) = \infty$ for every $n$, so the right side is $\infty$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Continuity from Above)</span></p>

Prove continuity from above for Lebesgue measure: if $\lbrace E_n\rbrace\_{n=1}^{\infty}$ is a countable collection of measurable sets, $m(E_1) < \infty$, and

$$E_1 \supset E_2 \supset \cdots,$$

then

$$m\left(\bigcap_k E_k\right) = \lim_{n \to \infty} m(E_n).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Define the disjoint sequence of sets for the difference between consecutive layers:

$$F_{i} := E_{i-1} \setminus E_i \quad \text{for } i \ge 2.$$

By construction, the sets $F_i$ are pairwise disjoint. The $n$-th set in the sequence can be expressed by subtracting the union of these outer layers from the initial set:

$$E_n = E_1 \setminus \bigcup_{i=2}^n F_i$$

Because $m(E_1) < \infty$ and $\bigcup_{i=2}^n F_i \subset E_1$, the measure of the set difference is strictly well-defined, allowing us to write:

$$m(E_n) = m\left(E_1 \setminus \bigcup_{i=2}^n F_i\right) = m(E_1) - m\left(\bigcup_{i=2}^n F_i\right)$$

Applying this decomposition to the infinite intersection, and invoking countable additivity on the disjoint union of $F_i$, we obtain the final result:

$$\boxed{
\begin{matrix}
m\left(\bigcap_{k=1}^\infty E_k\right) = m\left(E_1 \setminus \bigcup_{i=2}^\infty F_i\right) = m(E_1) - m\left(\bigcup_{i=2}^\infty F_i\right) \\
= m(E_1) - \lim_{n \to \infty} m\left(\bigcup_{i=2}^n F_i\right) = \lim_{n \to \infty} \left( m(E_1) - m\left(\bigcup_{i=2}^n F_i\right) \right) = \lim_{n \to \infty} m(E_n)
\end{matrix}
}$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Justifying the Limit Jump)</span></p>

In the initial draft of the solution, writing the equality 

$$m\left(E_1 \setminus \bigcup_{i=1}^\infty F_i\right) = \lim_{n \to \infty} m\left(E_1 \setminus \bigcup_{i=1}^n F_i\right)$$

in one step was a logical leap because it inherently assumed the very property you were trying to prove: **continuity from above**. You cannot pass a limit through the measure of a decreasing sequence of sets without first proving that the measure operation permits it.

In the revised proof, the limit jump is completely justified because the burden of the limit is shifted from a *decreasing* sequence of sets to an *increasing* sequence of **pairwise disjoint** sets.

By isolating the term $m\left(\bigcup_{i=2}^\infty F_i\right)$, you can invoke the fundamental axiom of **countable additivity** for Lebesgue measure. Countable additivity states that the measure of a countable union of disjoint sets is exactly the infinite series of their individual measures:

$$m\left(\bigcup_{i=2}^\infty F_i\right) = \sum_{i=2}^\infty m(F_i)$$

In real analysis, an infinite series is rigorously defined as the limit of its partial sums. Therefore, you can safely write:

$$\sum_{i=2}^\infty m(F_i) = \lim_{n \to \infty} \sum_{i=2}^n m(F_i) = \lim_{n \to \infty} m\left(\bigcup_{i=2}^n F_i\right)$$

By using the disjoint layers $F_i$, you successfully bypassed the unproven "continuity from above" by relying on the already established "countable additivity" (which is equivalent to continuity from below) to rigorously justify evaluating the limit.

</div>

## Measurable Functions

* ["The Lebesgue integral" by Tao](https://terrytao.wordpress.com/2010/09/19/245a-notes-2-the-lebesgue-integral/)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Split the range, measure the domain)</span></p>

* The motivation for defining Lebesgue measurable sets is to build towards a theory of integration that surpasses Riemann integration.
* While Riemann integration chops up the **domain** into intervals, Lebesgue's theory chops up the **range**:
  1. **splitting** the range into bands such as $[y_i, y_{i+1}]$,
  2. **finding** the corresponding preimage $E_i = f^{-1}([y_i, y_{i+1}])$ in the domain,
  3. **measuring** $m(E_i)$. This is why we care about preimages of closed intervals being measurable.

</div>

<figure class="math-figure">
  <svg viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg" width="760" role="img" aria-labelledby="lebesgue-sketch-title lebesgue-sketch-desc">
    <title id="lebesgue-sketch-title">Range pieces and their preimages in the domain</title>
    <desc id="lebesgue-sketch-desc">A diagram with a vertical range interval and a vertical domain line. Three range pieces have preimages that are respectively a single point, one interval, and two disjoint intervals.</desc>
    <defs>
      <marker id="lebesgue-domain-to-range-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
        <path d="M 0 0 L 8 4 L 0 8 Z" fill="#1b8f5a" />
      </marker>
    </defs>
    <rect x="42" y="34" width="676" height="346" rx="10" fill="#fbfcff" stroke="#e3e7f0" />

    <text x="92" y="62" font-size="16" font-weight="600" fill="#d96b2b">range interval</text>
    <text x="470" y="62" font-size="16" font-weight="600" fill="#1b8f5a">domain line</text>

    <g stroke="#333" stroke-width="2" fill="none">
      <line x1="170" y1="86" x2="170" y2="322" />
      <line x1="540" y1="86" x2="540" y2="322" />
    </g>
    <g stroke="#d96b2b" stroke-width="2.5" fill="none">
      <line x1="156" y1="86" x2="184" y2="86" />
      <line x1="156" y1="322" x2="184" y2="322" />
    </g>
    <g stroke="#1b8f5a" stroke-width="2.5" fill="none">
      <line x1="526" y1="86" x2="554" y2="86" />
      <line x1="526" y1="322" x2="554" y2="322" />
    </g>
    <text x="105" y="91" font-size="15" fill="#d96b2b">fᵢ₊₁</text>
    <text x="123" y="327" font-size="15" fill="#d96b2b">fᵢ</text>
    <text x="552" y="91" font-size="14" fill="#1b8f5a">E</text>

    <g fill="rgba(217,107,43,0.18)" stroke="#d96b2b" stroke-width="1.4">
      <rect x="138" y="104" width="64" height="44" />
      <rect x="138" y="178" width="64" height="54" />
      <rect x="138" y="260" width="64" height="46" />
    </g>
    <text x="155" y="130" font-size="13" fill="#d96b2b">I₁</text>
    <text x="155" y="210" font-size="13" fill="#d96b2b">I₂</text>
    <text x="155" y="288" font-size="13" fill="#d96b2b">I₃</text>
    <text x="84" y="205" font-size="14" fill="#d96b2b" transform="rotate(-90 84 205)">[fᵢ, fᵢ₊₁]</text>

    <g stroke="#ccd4e2" stroke-width="1" stroke-dasharray="4 5">
      <line x1="202" y1="126" x2="530" y2="126" />
      <line x1="202" y1="205" x2="530" y2="205" />
      <line x1="202" y1="283" x2="530" y2="283" />
    </g>

    <g stroke="#1b8f5a" stroke-width="7" stroke-linecap="round" fill="none">
      <line x1="540" y1="188" x2="540" y2="222" />
      <line x1="540" y1="258" x2="540" y2="273" />
      <line x1="540" y1="296" x2="540" y2="315" />
    </g>
    <circle cx="540" cy="126" r="6" fill="#1b8f5a" stroke="#0f6f43" stroke-width="1.4" />

    <g stroke="#1b8f5a" stroke-width="1.6" fill="none" marker-end="url(#lebesgue-domain-to-range-arrow)">
      <path d="M 532 126 C 442 126 304 126 205 126" />
      <path d="M 532 205 C 442 205 304 205 205 205" />
      <path d="M 532 265 C 450 264 307 280 205 283" />
      <path d="M 532 305 C 452 306 310 288 205 283" />
    </g>

    <text x="558" y="131" font-size="12" fill="#1b8f5a">E₁ = f⁻¹(I₁) is one point</text>
    <text x="558" y="209" font-size="12" fill="#1b8f5a">E₂ = f⁻¹(I₂) is an interval</text>
    <text x="558" y="278" font-size="12" fill="#1b8f5a">E₃ = f⁻¹(I₃)</text>
    <text x="558" y="296" font-size="12" fill="#1b8f5a">has two disjoint parts</text>

    <text x="226" y="101" font-size="13" fill="#555">f(E₁) ⊂ I₁</text>
    <text x="226" y="181" font-size="13" fill="#555">f(E₂) ⊂ I₂</text>
    <text x="226" y="259" font-size="13" fill="#555">f(E₃) ⊂ I₃</text>
    <text x="156" y="352" font-size="14" fill="#c0392b">m(f⁻¹([fᵢ,fᵢ₊₁])) measures all these preimage pieces in the domain</text>
  </svg>
  <figcaption>Sketch version of the idea: split the range interval $[f_i,f_{i+1}]$ into pieces $I_k$. Their preimages on the domain line can have different shapes: a point, an interval, or several disjoint pieces. Lebesgue's construction measures those preimages.</figcaption>
</figure>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_measurable_functions_lebesgue_split_find_measure.png' | relative_url }}" alt="Lebesgue split find measure visualization showing a function graph, a highlighted horizontal range band, its preimage in the domain, and the measured level-set pieces" loading="lazy">
  <figcaption>Lebesgue's theory starts with horizontal pieces of the range. For one band $[y_i, y_{i+1}]$, we find the domain set $E_i=f^{-1}([y_i,y_{i+1}])$ and measure $m(E_i)$; sums like $\sum_i y_i\,m(E_i)$ assemble the integral from those measured preimages.</figcaption>
</figure>

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">Riemann: partition the domain</div>
      <svg viewBox="0 0 260 220" xmlns="http://www.w3.org/2000/svg" width="280" aria-label="Riemann partition of the domain">
        <g stroke="#444" stroke-width="1.2" fill="none">
          <line x1="30" y1="180" x2="240" y2="180" />
          <line x1="30" y1="20"  x2="30"  y2="190" />
        </g>
        <polygon points="240,180 232,176 232,184" fill="#444" />
        <polygon points="30,20 26,28 34,28" fill="#444" />
        <text x="244" y="184" font-size="11" fill="#666">x</text>
        <text x="20"  y="22"  font-size="11" fill="#666">y</text>
        <g fill="rgba(44,73,148,0.18)" stroke="#2c4994" stroke-width="1">
          <rect x="30"  y="138" width="30" height="42" />
          <rect x="60"  y="106" width="30" height="74" />
          <rect x="90"  y="80"  width="30" height="100" />
          <rect x="120" y="76"  width="30" height="104" />
          <rect x="150" y="92"  width="30" height="88" />
          <rect x="180" y="118" width="30" height="62" />
          <rect x="210" y="155" width="30" height="25" />
        </g>
        <path d="M 30,150 Q 75,80 135,72 T 240,170" stroke="#222" stroke-width="2" fill="none" />
        <g stroke="#444" stroke-width="0.8" stroke-dasharray="2,2">
          <line x1="60"  y1="180" x2="60"  y2="190" />
          <line x1="90"  y1="180" x2="90"  y2="190" />
          <line x1="120" y1="180" x2="120" y2="190" />
          <line x1="150" y1="180" x2="150" y2="190" />
          <line x1="180" y1="180" x2="180" y2="190" />
          <line x1="210" y1="180" x2="210" y2="190" />
        </g>
        <text x="40"  y="200" font-size="9" fill="#666">x₀</text>
        <text x="70"  y="200" font-size="9" fill="#666">x₁</text>
        <text x="100" y="200" font-size="9" fill="#666">x₂</text>
        <text x="130" y="200" font-size="9" fill="#666">x₃</text>
        <text x="160" y="200" font-size="9" fill="#666">x₄</text>
        <text x="190" y="200" font-size="9" fill="#666">x₅</text>
        <text x="220" y="200" font-size="9" fill="#666">x₆</text>
      </svg>
    </div>
    <div>
      <div class="panel-label">Lebesgue: partition the range</div>
      <svg viewBox="0 0 260 220" xmlns="http://www.w3.org/2000/svg" width="280" aria-label="Lebesgue partition of the range">
        <g stroke="#444" stroke-width="1.2" fill="none">
          <line x1="30" y1="180" x2="240" y2="180" />
          <line x1="30" y1="20"  x2="30"  y2="190" />
        </g>
        <polygon points="240,180 232,176 232,184" fill="#444" />
        <polygon points="30,20 26,28 34,28" fill="#444" />
        <text x="244" y="184" font-size="11" fill="#666">x</text>
        <text x="20"  y="22"  font-size="11" fill="#666">y</text>
        <g fill="rgba(214,83,54,0.18)" stroke="#d65336" stroke-width="1">
          <rect x="30"  y="160" width="210" height="20" />
          <rect x="30"  y="140" width="210" height="20" />
          <rect x="34"  y="120" width="200" height="20" />
          <rect x="42"  y="100" width="184" height="20" />
          <rect x="58"  y="80"  width="140" height="20" />
          <rect x="86"  y="68"  width="74"  height="12" />
        </g>
        <path d="M 30,150 Q 75,80 135,72 T 240,170" stroke="#222" stroke-width="2" fill="none" />
        <g stroke="#444" stroke-width="0.8" stroke-dasharray="2,2">
          <line x1="20" y1="160" x2="30" y2="160" />
          <line x1="20" y1="140" x2="30" y2="140" />
          <line x1="20" y1="120" x2="30" y2="120" />
          <line x1="20" y1="100" x2="30" y2="100" />
          <line x1="20" y1="80"  x2="30" y2="80" />
          <line x1="20" y1="68"  x2="30" y2="68" />
        </g>
        <text x="6" y="163" font-size="9" fill="#666">y₁</text>
        <text x="6" y="143" font-size="9" fill="#666">y₂</text>
        <text x="6" y="123" font-size="9" fill="#666">y₃</text>
        <text x="6" y="103" font-size="9" fill="#666">y₄</text>
        <text x="6" y="83"  font-size="9" fill="#666">y₅</text>
        <text x="6" y="71"  font-size="9" fill="#666">y₆</text>
      </svg>
    </div>
  </div>
  <figcaption>Two strategies for the same area. Riemann sums approximate $\int f$ by $\sum f(\xi_j)\Delta x_j$ over a domain partition — fragile when $f$ wiggles wildly within a strip. Lebesgue sums use a range partition $\sum y_i \cdot m(f^{-1}([y_{i-1}, y_i]))$ — robust because measurability of preimages, not regularity of $f$, is what matters.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name"></span></p>

Throughout the discussion of measurable functions and Lebesgue integration, we work with the **extended real numbers** $[-\infty, \infty] = \mathbb{R} \cup \lbrace -\infty, \infty \rbrace$, and we allow functions to take on the values $\pm \infty$. The arithmetic rules are: 

$$x \pm \infty = \pm \infty \quad \forall x \in \mathbb{R},$$ 

$$0(\pm \infty) = 0,$$

$$x(\pm \infty) = \pm \infty \quad \forall x > 0.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Measurable Function)</span></p>

Let $E \subset \mathbb{R}$ be measurable, and let $f : E \to [-\infty, \infty]$ be a function. Then $f$ is **Lebesgue measurable** if 

$$\forall \alpha \in \mathbb{R}:\quad f^{-1}((\alpha, \infty]) \in \mathcal{M}$$ 

i.e., the preimage is a measurable set.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(This definition is equivalent to the open-set preimage definition.)</span></p>

For an ordinary real-valued function $f:E\to\mathbb R$, saying

$$f^{-1}(O)\in\mathcal M \quad\text{for every open }O\subset\mathbb R$$

is equivalent to saying

$$f^{-1}((\alpha,\infty))\in\mathcal M \quad\text{for every }\alpha\in\mathbb R.$$

Since the course allows $+\infty$, it writes $(\alpha,\infty]$ instead of $(\alpha,\infty)$. That is the only extra complication.

**Why half-lines are enough.**

The reason is that the Borel $\sigma$-algebra on $\mathbb R$ is generated by such half-lines. So if you know all sets

$$\lbrace x\in E:f(x)>\alpha\rbrace$$

are measurable, then you can build preimages of all ordinary open sets by countable unions, intersections, and complements.

For example,

$${f\ge \alpha} = \bigcap_{n=1}^\infty {f>\alpha-1/n},$$

so $\lbrace f\ge \alpha\rbrace$ is measurable. Then

$$\lbrace f<\beta\rbrace = E\setminus \lbrace f\ge \beta\rbrace$$

is measurable. Hence

$$f^{-1}((a,b)) = \lbrace a<f<b \rbrace = \lbrace f>a\rbrace\cap \lbrace f<b\rbrace$$

is measurable. Since every open set $O\subset\mathbb R$ is a countable union of open intervals, $f^{-1}(O)$ is measurable.

Your notes also prove that every open set is Lebesgue measurable and that the Borel $\sigma$-algebra is contained in the Lebesgue $\sigma$-algebra, precisely using the fact that half-lines generate the relevant measurable structure. 

**Why define it this way, then?**

Because the half-line condition is a **minimal test**. Instead of checking all open sets $O\subset\mathbb R$, which is a large class, it is enough to check only the threshold sets

$$\lbrace f>\alpha\rbrace.$$

This is also the natural condition for Lebesgue integration: the Lebesgue integral thinks of a function by slicing its **range** into levels and asking whether the corresponding preimage sets are measurable. Your notes even emphasize this viewpoint: Lebesgue sums use sets like $f^{-1}([y_{i-1},y_i])$, so measurability of preimages is the central object. 

The proposed definition is not wrong. It is the general Borel-measurability definition. The course’s definition is a more economical, order-based version specialized to real or extended-real-valued functions. For $f:E\to\mathbb R$, they are equivalent; for $f:E\to[-\infty,\infty]$, the half-line version is especially convenient because it handles $+\infty$ directly.


</div>

TODO: add a remark on general measurability, ask for the open sets and add visualization

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Equivalent Conditions for Measurability)</span></p>

Let $E \subset \mathbb{R}$ be measurable and $f : E \to [-\infty, \infty]$. The following are equivalent:

1. For all $\alpha \in \mathbb{R}$: $f^{-1}((\alpha, \infty]) \in \mathcal{M}$.
2. For all $\alpha \in \mathbb{R}$: $f^{-1}([\alpha, \infty]) \in \mathcal{M}$.
3. For all $\alpha \in \mathbb{R}$: $f^{-1}([-\infty, \alpha)) \in \mathcal{M}$.
4. For all $\alpha \in \mathbb{R}$: $f^{-1}([-\infty, \alpha]) \in \mathcal{M}$.

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

If $E$ is measurable and $f : E \to \mathbb{R}$ is a measurable function, then 

$$f^{-1}(F) \text{ is measurable } \quad \forall F \in \mathcal{B}.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>The preimage of any open interval $(a, b)$ is measurable: $f^{-1}((a, b)) = f^{-1}([-\infty, b)) \cap f^{-1}((a, \infty])$, and both sets are measurable. Since every open set is a countable union of open intervals, $f^{-1}(U)$ is measurable for all open $U$. The collection $\mathcal{A} = \lbrace F \subset \mathbb{R} : f^{-1}(F) \text{ measurable} \rbrace$ is a $\sigma$-algebra containing all open sets, so $\mathcal{B} \subset \mathcal{A}$. $\square$</p>
  </details>
</div>

TODO: what if the function defined on non-measurable set?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Preimages of Infinities Singletons are Measurable)</span></p>

If $f : E \to \mathbb{R}$ is measurable, then $f^{-1}(\lbrace \infty \rbrace)$ and $f^{-1}(\lbrace -\infty \rbrace)$ are also measurable
* since $f^{-1}(\lbrace \infty \rbrace) = \bigcap_n f^{-1}((n, \infty])$.

</div>

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

Let $E \subset \mathbb{R}$ be measurable, and suppose $f, g : E \to \mathbb{R}$ are two measurable functions and $c \in \mathbb{R}$. Then 
* $cf$, 
* $f + g$,
* $fg$

are all measurable functions.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Scalar multiple:**

$$(cf)^{-1}((\alpha, \infty]) = f^{-1}((\alpha/c, \infty])$$

(for $c > 0$; similar for $c < 0$; trivial for $c = 0$).

**Sum:**

$$f(x) + g(x) > \alpha \iff f(x) > r > \alpha - g(x)$$ 

for some rational $r$. So 

$$(f + g)^{-1}((\alpha, \infty]) = \bigcup_{r \in \mathbb{Q}} (f^{-1}((r, \infty]) \cap g^{-1}((\alpha - r, \infty])),$$

a countable union of measurable sets.

**Product:**

1. First show $f^2$ is measurable: 

   1. for $\alpha < 0$, 

      $$(f^2)^{-1}((\alpha, \infty]) = E.$$

   2. for $\alpha \ge 0$, 
      
      $$(f^2)^{-1}((\alpha, \infty]) = f^{-1}((\sqrt{\alpha}, \infty]) \cup f^{-1}([-\infty, -\sqrt{\alpha})).$$

2. Then use 

   $$fg = \frac{1}{4}((f + g)^2 - (f - g)^2).$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Locality of measurability)</span></p>

Suppose that $E$ and $F$ are measurable sets and $f : E \cup F \to [-\infty,\infty]$. Prove that $f$ is measurable iff its restrictions $f\mid\_E$ and $f\mid\_F$ are measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**Direction $\Rightarrow$:**

Fix $n \in \mathbb{R}$

$$f^{-1}((n, \infty)) = f^{-1}\mid\_E((n, \infty)) \cup f^{-1}\mid\_F((n, \infty)).$$

By assumption, $f^{-1}((n, \infty))$ is a measurable set. $E$ and $F$ are given as measurable sets. Then

$$f^{-1}\mid\_E((n, \infty)) = f^{-1}((n, \infty)) \cap E,$$

$$f^{-1}\mid\_F((n, \infty)) = f^{-1}((n, \infty)) \cap F.$$

Hence, $f^{-1}\mid\_E((n, \infty))$ and $f^{-1}\mid\_F((n, \infty))$ are measurable sets since they are intersection of measurable sets.

**Direction $\Leftarrow$:**

By assumption, for all $n \in \mathbb{R}$ the preimages $f^{-1}\mid\_E((n, \infty))$, $f^{-1}\mid\_F((n, \infty))$ are measurable sets.

Also

$$f^{-1}((n, \infty)) = f^{-1}\mid\_E((n, \infty)) \cup f^{-1}\mid\_F((n, \infty)).$$

Hence $f^{-1}((n, \infty))$ is measurable as a union of measurable sets.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Measurability of zero extensions)</span></p>

Suppose that $E$ is a measurable set and $f : E \to [-\infty,\infty]$. Let $g : \mathbb{R} \to [-\infty,\infty]$ be the extension of $f$ by zero:

$$
g(x)=
\begin{cases}
f(x), & \text{if } x \in E,\\
0, & \text{if } x \in E^c.
\end{cases}
$$

Prove that $f$ is measurable iff $g$ is measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

* Both directions follow from the previous exercise on Locality of measurability.
* In this case $g(x): E \cup E^c \to [-\infty, \infty]$ and $f$ is a measurable restriction on $E$, constant zero function is a measurable restriction on $E^c$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Measurability of the Euclidean norm)</span></p>

Suppose that $E$ is a measurable set, and $u : E \to \mathbb{R}$ and $v : E \to \mathbb{R}$ are measurable. Prove that $(u^2+v^2)^{1/2}$ is measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

By Algebra of Measurable Functions,
* $u,v$ are measurable $\implies$ $u^2,v^2$ are measurable.
* $u^2,v^2$ are measurable $\implies$ $u^2 + v^2$ is measurable
* $u^2 + v^2$ is measurable $\implies$ $(u^2 + v^2)^{1/2}$ is measurable
  * $u^2 + v^2$ is non-negative

  $$(g^{1/2})^{-1}((n, \infty)) = \begin{cases} g^{-1}((n^2, \infty)) & \text{if } n \ge 0 \\ E & \text{if } n < 0 \end{cases}$$

  Because $g$ is a measurable function, $g^{-1}((n^2, \infty))$ is a measurable set. Because $E$ is a measurable set, both piecewise conditions yield a measurable set.

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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Pointwise Limits of Measurable Functions)</span></p>

If $f_n : E \to [-\infty, \infty]$ are measurable for all $n$ and $\lim_{n \to \infty} f_n(x) = f(x)$ for all $x \in E$, then $f$ is measurable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pointwise limit of Riemann integrable functions need not be Riemann integrable)</span></p>

This corollary is **false** for Riemann integration — the pointwise limit of Riemann integrable functions need not be Riemann integrable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Pointwise Limit Leaving Riemann Integration)</span></p>

Enumerate 

$$\mathbb{Q} \cap [0, 1] = \lbrace r_1, r_2, r_3, \dots \rbrace.$$

Define 

$$
f_n(x) = \begin{cases} 
      1, & x \in \lbrace r_1, \dots, r_n \rbrace \\
      0, & \text{otherwise}
   \end{cases}
$$

Each $f_n$ is Riemann integrable (piecewise continuous), but 

$$\lim f_n = \chi_{\mathbb{Q} \cap [0,1]},$$

which is not Riemann integrable.

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Egorov’s theorem / Littlewood’s second principle: Vanishing measure of tail bad sets)</span></p>

Suppose that $E \subset \mathbb{R}$ is a measurable set such that $m(E) < \infty$, and suppose that $f_n : E \to \mathbb{R}$ is measurable for all $n$, $f : E \to \mathbb{R}$ is measurable and $\lim_{n \to \infty} f_n(x) = f(x)$ a.e. on $E$.

(a) For $k,n \in \mathbb{N}$, let

$$F_n(k) = \bigcup_{m=n}^{\infty} \left\lbrace x \in E \mid |f_m(x)-f(x)| \ge k^{-1} \right\rbrace.$$

Prove that

$$\lim_{n \to \infty} m(F_n(k)) = 0.$$

*Hint:* Show that $\bigcap_{n=1}^{\infty} F_n(k)$ is a set of measure zero, and appeal to Problem 1(b).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

By continuity from above for Lebesgue measure, for the sets if $\lbrace F_n \rbrace_{n=1}^\infty$ is a countable collection of measurable sets, $m(F_1) < \infty$ and 

$$F_1 \supset F_2 \supset \cdots,$$

then 

$$m\left(\bigcap_n F_n\right) = \lim_n m(F_n).$$

We verify the pre-requests of the theorem to apply it:

1. $m(F_1(k)) < \infty$:
   * $F\_1(k) = \bigcup\_{m=1}^\infty \lbrace x\in E\mid \lvert f\_m(x) - f(x) \rvert \geq k^{-1} \rbrace$
   * $m(F_1(k)) \leq m(E) < \infty$
2. Nested structure of $F_n(k)$ sets is obvious from the union.

Now show that $\bigcap_{n=1}^\infty F_n(k)$ set has measure zero.

$$\bigcap_{n=1}^\infty \bigcup_{m=n}^\infty \lbrace x\in E \mid \lvert f_m(x) - f(x) \rvert \geq k^{-1} \rbrace \subseteq \lbrace x\in E \mid \limsup_{n\to\infty} \lvert f_n(x) - f(x) \rvert \geq k^{-1} \rbrace.$$

By assumption the latter set has measure zero: $f_n$ converges to $f$ a.e., so the amount of $x\in E$, where the given inequality holds is zero by the definition of a.e. convergence. So

$$0 = m(\bigcap_{n=1}^\infty F_n(k)) = \lim_n m(F_n(k)).$$
 
</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Egorov’s theorem / Littlewood’s second principle: Almost uniform convergence outside a small set)</span></p>

Let $\varepsilon > 0$. By part (a), for $k \in \mathbb{N}$, there exists $n_k \in \mathbb{N}$ such that

$$m(F_{n_k}(k)) < 2^{-k}\varepsilon.$$

Let

$$F = \bigcup_{k=1}^{\infty} F_{n_k}(k).$$

Prove that $m(F) < \varepsilon$ and $f_n \to f$ uniformly on $F^c$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

1. $m(F) < \varepsilon$
   
   $$m(F) \leq \sum_{k=1}^\infty m(F_{n_k}(k)) < \sum_{k=1}^\infty \varepsilon 2^{-k} = \varepsilon.$$
  
2. $f_n \to f$ uniformly on $F^c$

    $$F^c = \bigcap_{k=1}^\infty \bigcap_{m=n_k}^\infty \lbrace x\in E \mid \lvert f_m(x) - f(x) \rvert < k^{-1} \rbrace$$

    * We should be able for every $\varepsilon > 0$ find $n_k\in\mathbb{N}$ s.t. for all $m\geq n_k$ and $x\in F^c \lvert f_m(x) - f(x) \rvert < \varepsilon$.
    * For the given $\varepsilon$ we consider the first $k$ s.t. $k^{-1} < \varepsilon$.
    * The inner intersection gives all $x\in E$ s.t. $m\geq n_k \lvert f_m(x) - f(x) \rvert < k^{-1} < \varepsilon$. The outer intersection guarantees that such $k\in\mathbb{N}$ exists.
    * Because the threshold index $n_k$ depends only on $k$ (which depends only on $\varepsilon$) and is entirely independent of $x$, the convergence $f_n\to f$ is uniform on $F^c$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Littlewood’s second principle)</span></p>

This result is known as **Littlewood’s second principle**: 

$$\boxed{\text{Every convergent sequence of measurable functions is nearly uniformly convergent.}}$$

</div>

### Complex-Valued Measurable Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complex-Valued Measurable Function)</span></p>

If $E \subset \mathbb{R}$ is measurable, a complex-valued function $f : E \to \mathbb{C}$ is **measurable** if $\operatorname{Re}(f)$ and $\operatorname{Im}(f)$ (which are both functions $E \to \mathbb{R}$) are measurable.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Algebra of Complex Measurable Functions)</span></p>

If $f, g : E \to \mathbb{C}$ are measurable functions and $\alpha \in \mathbb{C}$, then 
* $\alpha f$, 
* $f + g$, 
* $fg$, 
* $\bar{f}$,
* $\lvert f \rvert$
  
are all measurable.

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Step functions are nearly continuous off a small set)</span></p>

Let $a < b$. A simple function $\psi : [a,b] \to \mathbb{R}$ with range

$$\psi([a,b]) = \lbrace a_1,\ldots,a_n\rbrace$$

is a **step function** if for all $i = 1,\ldots,n$,

$$\psi^{-1}(\lbrace a_i\rbrace)$$

is a finite union of intervals. This is equivalent to

$$\psi = \sum_{i=1}^n a_i \chi_{U_i},$$

where for all $i$, $U_i \subset [a,b]$ is a finite union of intervals, for all $i \ne j$,

$$U_i \cap U_j = \varnothing$$

and

$$\bigcup_{i=1}^n U_i = [a,b].$$

---

**(a)** Suppose that $\psi : [a,b] \to \mathbb{R}$ is a step function such that

$$\sup_{x \in [a,b]} |\psi(x)| \le B.$$

Let $\epsilon > 0$. Prove that there exists a continuous function

$$g : [a,b] \to \mathbb{R}$$

and a measurable set $E \subset [a,b]$ such that

$$g(a) = g(b) = 0, \qquad  \sup_{x \in [a,b]} |g(x)| \le B, \qquad  m(E) < \epsilon,$$

and

$$\forall x \in E^c,  \qquad  |\psi(x) - g(x)| < \epsilon.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**Key idea.** Since $\psi$ has only finitely many interval boundaries, we replace each jump by a narrow polynomial transition. We include all transition regions in $E$; hence outside $E$, the approximation error is actually zero.

Let $D$ be the finite set consisting of $a,b$ and all endpoints of the intervals appearing in the level sets of $\psi$. Write its distinct elements as

$$a=t_0<t_1<\cdots<t_r=b.$$

Then, for every $j=1,\ldots,r$, the function $\psi$ is constant on $(t_{j-1},t_j)$. Denote this value by $c_j$. Since $\sup\lvert\psi\rvert\le B$,

$$|c_j|\le B \qquad \text{for every }j.$$

Fix an integer $N\ge 1$. Choose $\delta>0$ sufficiently small that

$$(r+1)\delta<\epsilon$$

and

$$3\delta<\min_{1\le j\le r}(t_j-t_{j-1}).$$

The second condition ensures that the transition intervals introduced below do not overlap.

Define the exceptional set

$$E = [a,a+\delta] \cup \bigcup_{j=1}^{r-1}[t_j-\delta,t_j] \cup [b-\delta,b].$$

There are $r+1$ intervals of length $\delta$, so

$$m(E)\le (r+1)\delta<\epsilon.$$

The endpoints $a,b$ and all possible discontinuity points $t_j$ belong to $E$.

We now define $g$. Outside $E$, set

$$g(x):=\psi(x).$$

On the left endpoint transition, define

$$g(x) = c_1\left(\frac{x-a}{\delta}\right)^N, \qquad x\in[a,a+\delta].$$

Thus $g(a)=0$ and $g(a+\delta)=c_1$.

For each internal boundary $t_j$, $j=1,\ldots,r-1$, define

$$g(x) = \left( 1-\left(\frac{x-(t_j-\delta)}{\delta}\right)^N \right)c_j + \left(\frac{x-(t_j-\delta)}{\delta}\right)^N c_{j+1},$$

for $x\in[t_j-\delta,t_j]$. This polynomial transition joins the level $c_j$ to the next level $c_{j+1}$.

Finally, on the right endpoint transition, define

$$g(x) = \left( 1-\left(\frac{x-(b-\delta)}{\delta}\right)^N \right)c_r, \qquad x\in[b-\delta,b].$$

Thus $g(b-\delta)=c_r$ and $g(b)=0$.

**Continuity.** Each transition is polynomial on its interval. At every endpoint of a transition interval, its value agrees with the constant value of $g$ on the adjacent interval. Hence

$$g\in C([a,b]), \qquad g(a)=g(b)=0.$$

**Uniform bound.** On an internal transition interval, put

$$\lambda = \left(\frac{x-(t_j-\delta)}{\delta}\right)^N.$$

Then $0\le\lambda\le1$, and

$$g(x)=(1-\lambda)c_j+\lambda c_{j+1}.$$

Therefore $g(x)$ lies between $c_j$ and $c_{j+1}$, and hence

$$|g(x)| \le \max{|c_j|,|c_{j+1}|} \le B.$$

The same argument applies to the endpoint transitions, which interpolate between $0$ and $c_1$, or between $c_r$ and $0$. Consequently,

$$\sup_{x\in[a,b]}|g(x)|\le B.$$

Finally, by construction,

$$g(x)=\psi(x) \qquad \text{for every }x\in E^c.$$

Thus, more strongly,

$$|\psi(x)-g(x)|=0<\epsilon \qquad \text{for every }x\in E^c.$$

This proves the claim.

**Terminology.** The resulting $g$ is not one polynomial on all of $[a,b]$; it is a continuous piecewise-polynomial function built from truncated and rescaled powers. Also, once the entire transition region is included in $E$, the exponent $N$ does not control $m(E)$; the width $\delta$ does. In fact, $N=1$ already works.

**Remark:** Approximation using the piecewise linear interpolation would also work. 

</details>
</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_step_function_polynomial_transition.png' | relative_url }}" alt="A step function drawn dashed and its continuous approximation drawn solid; each jump is replaced by a steep polynomial transition inside narrow shaded vertical bands forming the exceptional set E, with g vanishing at both endpoints" loading="lazy">
  <figcaption>The construction of the main solution: outside the shaded set $E$ we have $g=\psi$ exactly, while on each band $[t_j-\delta,t_j]$ the polynomial blend $(1-\lambda^N)c_j+\lambda^N c_{j+1}$ (here $N=4$) joins the neighbouring levels continuously; the endpoint bands pull $g$ down to $g(a)=g(b)=0$, and $m(E)\le (r+1)\delta<\epsilon$.</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Simpler solution</summary>

The previous construction is mathematically valid, but the $x^N$ transition and the logarithmic lower bound for $N$ are overkill.

$$\text{step function} \longrightarrow \text{continuous function},$$

followed by

$$\text{simple function} \longrightarrow \text{step function},$$

and then the general Lusin-type approximation for bounded measurable functions. So part (a) is intended as a routine “smooth the finitely many jumps on small intervals” argument, not as a quantitative study of the convergence $x^N\to\chi_{{1}}$. 

**The simplest expected construction is piecewise linear interpolation.**

Let $t_1,\dots,t_r$ be all the interval endpoints at which $\psi$ may change value, together with (a,b). Choose small intervals around these finitely many points whose total length is less than $\epsilon$, and let their union be $E$.

Define $g=\psi$ outside $E$. On every component of $E$:

* linearly interpolate between the constant values of $\psi$ on the two neighboring intervals;
* at (a), interpolate between $0$ and the first value of (\psi);
* at (b), interpolate between the last value of $\psi$ and $0$.

Because every linear interpolation has the form

$$g(x)=(1-\lambda)c+\lambda d, \qquad 0\le \lambda\le 1,$$

where $\lvert c\rvert,\lvert d\rvert \le B$, one immediately gets

$$|g(x)|\le B.$$

Moreover,

$$g=\psi \quad\text{on }E^c,$$

so the error there is exactly zero.

Thus the intended proof is probably about one paragraph. The previous main geometric idea was exactly right; the unnecessary part was introducing $N$ and solving an inequality for it. Since the entire transition region belongs to $E$, the shape and steepness of the transition do not matter. Even $N=1$ works.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Simple Function is nearly Continuous Function)</span></p>

$$\boxed{\text{Every step function is nearly continuous.}}$$

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_step_function_linear_interpolation.png' | relative_url }}" alt="The same step function drawn dashed with a continuous piecewise-linear approximation drawn solid; straight segments cross each jump inside narrow shaded vertical bands forming the exceptional set E, with g vanishing at both endpoints" loading="lazy">
  <figcaption>The simpler construction: on each component of $E$ the values $c_j$ and $c_{j+1}$ are joined by a straight segment $(1-\lambda)c_j+\lambda c_{j+1}$, so $|g|\le B$ is immediate; the shape of the transition is irrelevant because the whole transition region lies inside $E$.</figcaption>
</figure>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/proof_sketch_measure_theory_pol_approx.png' | relative_url }}" alt="123" loading="lazy">
  <figcaption>Sketch of the proof: approximating the step function by a continuous piecewise-polynomial function</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Simple functions are nearly step functions off a small set: Approximation of bounded simple functions by regular functions off small exceptional sets)</span></p>

**(b)** Suppose that $\varphi : [a,b] \to \mathbb{R}$ is a simple function such that

$$\sup_{x \in [a,b]} |\varphi(x)| \le B.$$

Let $\epsilon > 0$. Prove that there exists a step function

$$\psi : [a,b] \to \mathbb{R}$$

and a measurable set $E \subset [a,b]$ such that

$$\sup_{x \in [a,b]} |\psi(x)| \le B, \qquad  m(E) < \epsilon,$$

and

$$\forall x \in E^c, \qquad  |\varphi(x) - \psi(x)| < \epsilon.$$

*Hint: See Littlewood’s first principle.*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

> **Idea:** As a proof sketch can we say that since the simple function is measurable, the preimage of each a_i from the finite range is measurable. According to the first Littlewood's principle, every measurable set is almost (with arbitrarily positive error) a finite union of open intervals. Then applying the same idea on every a_i from the finite range, every simple function is almost a step function, since the step function is a simple function, where every preimage is finite union of intervals. Let the error for each preimage from the first Littlewood's principle will be almost delta, then total error is almost n\cdot delta. Thus taking delta < epsilon/n makes the simple function a step function out of this "bad" set.

Write

$$\varphi=\sum_{i=1}^n a_i\chi_{A_i}, \qquad A_i:=\varphi^{-1}({a_i}).$$

The sets $A_1,\dots,A_n$ are measurable and form a partition of $[a,b]$.

For each $i=1,\dots,n-1$, Littlewood’s first principle gives a finite union of intervals $U_i\subset[a,b]$ such that

$$m(A_i\triangle U_i)<\frac{\epsilon}{n}.$$

Disjointize these sets by defining

$$V_1:=U_1,\qquad V_i:=U_i\setminus\bigcup_{j<i}U_j \quad (2\le i<n),$$

and

$$V_n:=[a,b]\setminus\bigcup_{i=1}^{n-1}V_i.$$

Each $V_i$ is a finite union of intervals, and the sets $V_i$ form a partition of $[a,b]$. Therefore

$$\psi:=\sum_{i=1}^n a_i\chi_{V_i}$$

is a step function. Since $\psi$ takes only values from the range of $\varphi$,

$$\sup_{x\in[a,b]}|\psi(x)|\le B.$$

Set

$$E:=\bigcup_{i=1}^{n-1}(A_i\triangle U_i).$$

Then

$$m(E) \le \sum_{i=1}^{n-1}m(A_i\triangle U_i) <\frac{n-1}{n}\epsilon <\epsilon.$$

Finally, if $x\notin E$, then $x\in U_i$ exactly when $x\in A_i$ for every $i<n$. Hence the disjointization assigns $x$ to the same level set as $\varphi$, so

$$\psi(x)=\varphi(x).$$

Thus, for every $x\in E^c$,

$$|\varphi(x)-\psi(x)|=0<\epsilon.$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Simple Function is nearly Step Function)</span></p>

$$\boxed{\text{Every simple function is nearly a step function.}}$$

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
  </details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Build $\phi_n$ with resolution $2^{-n}$ and range $[0, 2^n]$. Define

$$E_n^k = \lbrace x \in E : k 2^{-n} < f(x) \le (k+1)2^{-n} \rbrace \quad (0 \le k \le 2^{2n} - 1), \qquad F_n = f^{-1}((2^n, \infty]),$$

and set 

$$\phi_n = \sum_{k=0}^{2^{2n}-1} (k 2^{-n}) \chi_{E_n^k} + 2^n \chi_{F_n}$$

Then $0 \le \phi_n \le f$ by construction.

**Monotonicity (a):** If $x \in E_n^k$ then 

$$k2^{-n} < f(x) \le (k+1)2^{-n},$$

which implies 

$$x \in E_{n+1}^{2k} \cup E_{n+1}^{2k+1}.$$

In both cases 

$$\phi_{n+1}(x) \ge (2k)2^{-(n+1)} = k2^{-n} = \phi_n(x).$$

Similarly for $x \in F_n$.

**Convergence (b) and (c):** 

* On the set $\lbrace x : f(x) \le 2^n \rbrace$, we have 

  $$0 \le f(x) - \phi_n(x) \le 2^{-n}$$

  * For any $x$ with $f(x) < \infty$, eventually $f(x) \le 2^n$ for large $n$, giving pointwise convergence. 
  * For $f(x) = \infty$, $\phi_n(x) = 2^n \to \infty$. 

* On 

  $$\lbrace f(x) \le B \rbrace \subset \lbrace f(x) \le 2^N \rbrace$$

  for $N$ large enough, the bound $2^{-n}$ gives uniform convergence.

</details>
</div>

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">Resolution n = 1 (step 2⁻¹)</div>
      <svg viewBox="0 0 260 200" xmlns="http://www.w3.org/2000/svg" width="280" aria-label="Dyadic simple approximation, coarse">
        <g stroke="#e8e8e8" stroke-width="0.7" stroke-dasharray="3,3">
          <line x1="30" y1="145" x2="230" y2="145" />
          <line x1="30" y1="120" x2="230" y2="120" />
          <line x1="30" y1="95"  x2="230" y2="95" />
          <line x1="30" y1="70"  x2="230" y2="70" />
          <line x1="30" y1="45"  x2="230" y2="45" />
        </g>
        <g stroke="#444" stroke-width="1.2" fill="none">
          <line x1="30" y1="170" x2="230" y2="170" />
          <line x1="30" y1="30"  x2="30"  y2="178" />
        </g>
        <polygon points="230,170 222,166 222,174" fill="#444" />
        <polygon points="30,30 26,38 34,38" fill="#444" />
        <text x="234" y="174" font-size="10" fill="#666">x</text>
        <text x="20"  y="32"  font-size="10" fill="#666">y</text>
        <text x="14" y="148" font-size="9" fill="#888">½</text>
        <text x="14" y="123" font-size="9" fill="#888">1</text>
        <text x="11" y="98"  font-size="9" fill="#888">3⁄2</text>
        <text x="14" y="73"  font-size="9" fill="#888">2</text>
        <path d="M 30,170 L 57.5,170 L 57.5,145 L 79.5,145 L 79.5,120 L 101.5,120 L 101.5,95 L 158.5,95 L 158.5,70 L 180.5,70 L 180.5,95 L 202.5,95 L 202.5,120 L 230,120 L 230,170 Z"
              fill="rgba(214,83,54,0.15)" stroke="none" />
        <path d="M 30,145 L 57.5,145 L 57.5,120 L 79.5,120 L 79.5,95 L 101.5,95 L 101.5,70 L 158.5,70 L 158.5,95 L 180.5,95 L 180.5,120 L 202.5,120 L 202.5,145 L 230,145"
              stroke="#d65336" stroke-width="2" fill="none" />
        <path d="M 30,142 C 55,124 80,95 105,67 C 117,57 143,57 155,67 C 180,95 205,124 230,142"
              stroke="#2c4994" stroke-width="2" fill="none" />
        <text x="180" y="42" font-size="11" fill="#2c4994" font-weight="600">f</text>
        <text x="38"  y="160" font-size="10" fill="#d65336" font-weight="600">φ₁</text>
      </svg>
    </div>
    <div>
      <div class="panel-label">Resolution n = 2 (step 2⁻²)</div>
      <svg viewBox="0 0 260 200" xmlns="http://www.w3.org/2000/svg" width="280" aria-label="Dyadic simple approximation, fine">
        <g stroke="#e8e8e8" stroke-width="0.7" stroke-dasharray="3,3">
          <line x1="30" y1="157.5" x2="230" y2="157.5" />
          <line x1="30" y1="145"   x2="230" y2="145" />
          <line x1="30" y1="132.5" x2="230" y2="132.5" />
          <line x1="30" y1="120"   x2="230" y2="120" />
          <line x1="30" y1="107.5" x2="230" y2="107.5" />
          <line x1="30" y1="95"    x2="230" y2="95" />
          <line x1="30" y1="82.5"  x2="230" y2="82.5" />
          <line x1="30" y1="70"    x2="230" y2="70" />
          <line x1="30" y1="57.5"  x2="230" y2="57.5" />
        </g>
        <g stroke="#444" stroke-width="1.2" fill="none">
          <line x1="30" y1="170" x2="230" y2="170" />
          <line x1="30" y1="30"  x2="30"  y2="178" />
        </g>
        <polygon points="230,170 222,166 222,174" fill="#444" />
        <polygon points="30,30 26,38 34,38" fill="#444" />
        <text x="234" y="174" font-size="10" fill="#666">x</text>
        <text x="20"  y="32"  font-size="10" fill="#666">y</text>
        <text x="14" y="148" font-size="9" fill="#888">½</text>
        <text x="14" y="123" font-size="9" fill="#888">1</text>
        <text x="11" y="98"  font-size="9" fill="#888">3⁄2</text>
        <text x="14" y="73"  font-size="9" fill="#888">2</text>
        <path d="M 30,170 L 50,170 L 50,132.5 L 60,132.5 L 60,120 L 70,120 L 70,107.5 L 80,107.5 L 80,95 L 90,95 L 90,82.5 L 110,82.5 L 110,70 L 120,70 L 120,57.5 L 150,57.5 L 150,70 L 160,70 L 160,82.5 L 180,82.5 L 180,95 L 190,95 L 190,107.5 L 200,107.5 L 200,120 L 210,120 L 210,132.5 L 230,132.5 L 230,170 Z"
              fill="rgba(214,83,54,0.15)" stroke="none" />
        <path d="M 30,145 L 50,145 L 50,132.5 L 60,132.5 L 60,120 L 70,120 L 70,107.5 L 80,107.5 L 80,95 L 90,95 L 90,82.5 L 110,82.5 L 110,70 L 120,70 L 120,57.5 L 150,57.5 L 150,70 L 160,70 L 160,82.5 L 180,82.5 L 180,95 L 190,95 L 190,107.5 L 200,107.5 L 200,120 L 210,120 L 210,132.5 L 220,132.5 L 220,145 L 230,145"
              stroke="#d65336" stroke-width="2" fill="none" />
        <path d="M 30,142 C 55,124 80,95 105,67 C 117,57 143,57 155,67 C 180,95 205,124 230,142"
              stroke="#2c4994" stroke-width="2" fill="none" />
        <text x="180" y="42" font-size="11" fill="#2c4994" font-weight="600">f</text>
        <text x="38"  y="160" font-size="10" fill="#d65336" font-weight="600">φ₂</text>
      </svg>
    </div>
  </div>
  <figcaption>Each $\phi_n$ slices the range into bands of height $2^{-n}$ and assigns the lower edge of whichever band $f(x)$ falls into. Refining $n \mapsto n+1$ subdivides every band in two, so $\phi_n \le \phi_{n+1} \le f$ pointwise; on $\{f \le B\}$ the gap $f - \phi_n \le 2^{-n}$ shrinks uniformly. The shaded region is the area under $\phi_n$, which converges to $\int f$ by monotone convergence.</figcaption>
</figure>

### Extension to General Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Positive and Negative Parts)</span></p>

Let $f : E \to [-\infty, \infty]$ be a measurable function. The **positive part** and **negative part** of $f$ are

$$f^+(x) = \max(f(x), 0), \qquad f^-(x) = \max(-f(x), 0),$$

so that $f = f^+ - f^-$ and $\lvert f \rvert = f^+ + f^-$. Both $f^+$ and $f^-$ are nonnegative measurable functions.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Simple Function Approximation — General Case)</span></p>

Let $E \subset \mathbb{R}$ be measurable and $f : E \to \mathbb{C}$ be measurable. Then there exists a sequence of simple functions $\lbrace \phi_n \rbrace$ such that 

$$(i)\quad 0 \le \lvert \phi_0(x) \rvert \le \lvert \phi_1(x) \rvert \le \cdots \le \lvert f(x) \rvert: \quad \lim_{n \to \infty} \phi_n(x) = f(x) \quad \forall x \in E$$

$$(ii)\quad \phi_n \to f \quad \text{uniformly on} \quad \lbrace x : \lvert f(x) \rvert \le B \rbrace \quad \forall B \ge 0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Idea</summary>

Split $f$ into real and imaginary parts, then each into positive and negative parts, apply the nonnegative approximation theorem to each, and recombine.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Lusin Approximation for Bounded Measurable Functions)</span></p>

Let $a < b$. Suppose that $f : [a,b] \to \mathbb{R}$ is a measurable function such that

$$\sup_{x \in [a,b]} |f(x)| \le B.$$

Let $\epsilon > 0$. Prove that there exists a continuous function

$$g : [a,b] \to \mathbb{R}$$

and a measurable set $E \subset [a,b]$ such that

$$g(a) = g(b) = 0, \qquad \sup_{x \in [a,b]} |g(x)| \le B, \qquad m(E) < \epsilon,$$

and

$$\forall x \in E^c, \qquad |f(x) - g(x)| < \epsilon.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution Idea and Different error contributions in each transtion of the proof</summary>

> **Idea:** It is just a compilation of two exercises above + the theorem from the lecture that any measurable function on a measurable set could be approximated by simple functions. Then show measurable function -> simple function -> step function -> continuous function. In this chain errors are accumulated, because each such approximation transition introduces some new error. We bound the error from above using some configurable parameter (in the first transition it is a resolution from the lecture notes, in the second transition it is a number of open intervals in unions and in the third transition it is degrees of the polynomials). he we find such parameters values that the error is below epsilon.

The overall chain is:

$$f \longrightarrow \varphi \longrightarrow \psi \longrightarrow g,$$

where

$$\text{measurable}\to\text{simple}\to\text{step}\to\text{continuous}.$$

But the errors are slightly different from what was described in the idea.

**The first transition produces a vertical approximation error.** Since $f$ is bounded, the simple-function approximation theorem gives a simple $\varphi$ such that

$$\sup_{x\in[a,b]}|f(x)-\varphi(x)|<\eta, \qquad \sup|\varphi|\le B.$$

This convergence is uniform on bounded ranges in your notes. 

**The next two transitions produce exceptional sets.**

* For $\varphi\to\psi$, the parameter is the allowed symmetric-difference measure when approximating the measurable level sets—not the number of intervals.
* For $\psi\to g$, the parameter is the total width of the small neighborhoods around the jump points—not the polynomial degree. Piecewise linear interpolation already works.

Using the stronger versions proved above, choose

$$\sup|f-\varphi|<\epsilon,$$

then choose exceptional sets $E_1,E_2$ with

$$m(E_1)<\frac{\epsilon}{2}, \qquad m(E_2)<\frac{\epsilon}{2},$$

such that

$$\varphi=\psi\quad\text{on }E_1^c, \qquad \psi=g\quad\text{on }E_2^c.$$

Set (E=E_1\cup E_2). Then

$$m(E)<\epsilon,$$

and for $x\in E^c$,

$$|f(x)-g(x)| = |f(x)-\varphi(x)| <\epsilon.$$

So your compilation idea is correct. The main correction is that only the first stage contributes an actual value error; the other two can be arranged to give exact equality outside small exceptional sets.

<!-- </details>
</div> -->

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Littlewood’s third principle)</span></p>

The result from the exercise above is known as **Littlewood’s third principle**

$$\boxed{\text{Every measurable function is nearly continuous.}}$$

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_lusin_nearly_continuous.png' | relative_url }}" alt="A smooth blue curve g on an interval with two narrow orange vertical bands where a red curve f oscillates wildly; outside the bands the two curves coincide, dashed lines mark the bound B and minus B, and g vanishes at both endpoints" loading="lazy">
  <figcaption>Littlewood's third principle in a picture: the measurable $f$ (red) may oscillate wildly, but only inside an exceptional set $E$ (orange bands) with $m(E)<\epsilon$. Outside $E$ it coincides with a continuous $g$ (blue) satisfying $g(a)=g(b)=0$ and $\sup|g|\le B$ — every measurable function is nearly continuous.</figcaption>
</figure>

## The Lebesgue Integral

### Integral of Nonnegative Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^+(E)$)</span></p>

If $E \subset \mathbb{R}$ is measurable, define 

$$L^+(E) = \lbrace f : E \to [0, \infty] : f \text{ measurable} \rbrace$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Integral of Simple Functions)</span></p>

For a simple function $\phi \in L^+(E)$ with canonical form 

$$\phi = \sum_{j=1}^{n} a_j \chi_{A_j},$$

where $A_i \cap A_j = \varnothing$ for $i \ne j$ and $\bigcup_j A_j = E$, the **Lebesgue integral** is

$$\int_E \phi = \sum_{j=1}^{n} a_j \, m(A_j) \in [0, \infty].$$

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_simple_integral_definition.png' | relative_url }}" alt="A simple function taking three values drawn as colored horizontal plateaus over scattered pieces of the interval, with matching colored bars below the axis marking the measure of each level set and a boxed formula summing value times measure" loading="lazy">
  <figcaption>The integral of a simple function $\phi=\sum_j a_j\chi_{A_j}$ in canonical form: each value $a_j$ is weighted by the measure of its preimage $A_j$ (color-coded bars below the axis). Each $A_j$ may be a scattered union of pieces — only $m(A_j)$ enters $\int_E\phi=\sum_j a_j\,m(A_j)$.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Properties of the Integral for Simple Functions)</span></p>

For simple functions $\phi, \psi \in L^+(E)$ and $c \ge 0$:

1. $\int_E c\phi = c \int_E \phi$.
2. $\int_E (\phi + \psi) = \int_E \phi + \int_E \psi$.
3. $\phi \le \psi \implies \int_E \phi \le \int_E \psi$.
4. $F \subset E \text{ is measurable } \implies \int_F \phi = \int_E \chi_F \phi \le \int_E \phi$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Integral of Nonnegative Measurable Functions)</span></p>

For $f \in L^+(E)$, the **Lebesgue integral** of $f$ is

$$\int_E f = \sup \left\lbrace \int_E \phi : \phi \in L^+(E) \text{ simple},\; \phi \le f \right\rbrace.$$

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_nonneg_integral_sup.png' | relative_url }}" alt="Two panels showing the same nonnegative curve: on the left a coarse blue staircase sits below the curve with shaded area 0.94, on the right a much finer green staircase hugs the curve from below with shaded area 1.08" loading="lazy">
  <figcaption>The integral of $f\in L^+(E)$ as a supremum: every simple $\phi\le f$ contributes its integral (shaded area), and refining the minorant pushes $\int_E\phi$ up toward $\int_E f=\sup\{\int_E\phi:\phi\le f\text{ simple}\}$.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Integral over Measure-Zero Set)</span></p>

$$E \subset \mathbb{R}:\ m(E) = 0 \implies \int_E f = 0\quad \forall f \in L^+(E)$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Basic Properties of the Nonnegative Integral)</span></p>

* If $f, g \in L^+(E)$, $c \in [0, \infty)$, and $F \subset E$ is measurable, then 
* $\int_E cf = c \int_E f$, 
* $f \le g \implies \int_E f \le \int_E g$,
* $f \le g \text{ a.e. } \implies \int_E f \le \int_E g \quad\text{ and }\quad \int_F f \le \int_E f$.

</div>

### The Monotone Convergence Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Monotone Convergence Theorem)</span></p>

If $\lbrace f_n \rbrace$ is a sequence of nonnegative measurable functions (in $L^+(E)$) such that $f_1 \le f_2 \le \cdots$ pointwise on $E$, and $f_n \to f$ pointwise on $E$ for some $f$ (which is in $L^+(E)$ by closure under limits), then

$$\lim_{n \to \infty} \int_E f_n = \int_E \lim_{n \to \infty} f_n = \int_E f.$$

</div>

The assumption of pointwise convergence here is much weaker than the uniform convergence we usually need for Riemann integration.

<div class="accordion">
  <details>
    <summary>Proof I</summary>
    <p>Since $f_n \le f$ for all $n$, $\int_E f_n \le \int_E f$, so $\lim \int_E f_n \le \int_E f$. For the reverse, it suffices to show $\int_E \phi \le \lim \int_E f_n$ for every simple $\phi \le f$.</p>
    <p>Fix $\varepsilon \in (0, 1)$ and let $\phi = \sum_{j=1}^{m} a_j \chi_{A_j}$ be simple with $\phi \le f$. Define $E_n = \lbrace x \in E : f_n(x) \ge (1 - \varepsilon)\phi(x) \rbrace$. Since $f_n \nearrow f \ge \phi$, every $x$ is eventually in some $E_n$, so $\bigcup_{n=1}^{\infty} E_n = E$ and $E_1 \subset E_2 \subset \cdots$.</p>
    <p>Then $\int_E f_n \ge \int_{E_n} f_n \ge (1 - \varepsilon) \int_{E_n} \phi = (1 - \varepsilon) \sum_j a_j m(A_j \cap E_n)$. By continuity of measure ($A_j \cap E_n \nearrow A_j$), taking $n \to \infty$ gives $\lim \int_E f_n \ge (1 - \varepsilon) \sum_j a_j m(A_j) = (1 - \varepsilon) \int_E \phi$. Since $\varepsilon$ is arbitrary, $\lim \int_E f_n \ge \int_E \phi$. $\square$</p>
  </details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof II</summary>

Let $E\subseteq\mathbb R$ be measurable, and let

$$0\le f_1\le f_2\le\cdots$$

be measurable functions on $E$ such that

$$f_n(x)\longrightarrow f(x) \qquad\text{for every }x\in E.$$

We prove that

$$\lim_{n\to\infty}\int_E f_n=\int_E f,$$

where the integrals may take the value $+\infty$.

**The key idea is to compare $f_n$ with an arbitrary simple function below $f$, lowered slightly so that $f_n$ eventually reaches it pointwise.**

Since $f_n\le f_{n+1}$, monotonicity of the integral gives

$$\int_E f_n\le \int_E f_{n+1}.$$

Consequently, the sequence of integrals is increasing, and therefore its extended-real limit exists:

$$L:=\lim_{n\to\infty}\int_E f_n =\sup_{n\in\mathbb N}\int_E f_n.$$

**The easy inequality is $L\le \int_E f$.**

Because $f_n\le f$ for every $n$, monotonicity gives

$$\int_E f_n\le \int_E f.$$

Taking the supremum over $n$,

$$L\le \int_E f.$$

It remains to prove the opposite inequality.

**Fix an arbitrary simple test function below $f$.**

Let $\phi\in L^+(E)$ be simple and satisfy

$$0\le \phi\le f.$$

We want to show

$$\int_E\phi\le L.$$

Once this is established for every such $\phi$, taking the supremum over all simple minorants of $f$ will give

$$\int_E f\le L.$$

The difficulty is that $\phi\le f$ does not imply that $\phi\le f_n$ for some $n$. For example, $f_n$ may approach $\phi=f$ strictly from below and never reach it.

**Lower $\phi$ slightly to create a strict gap.**

Fix $c\in(0,1)$, and define

$$E_n:=\lbrace x\in E:f_n(x)\ge c\phi(x)\rbrace.$$

These sets are measurable. Since $f_n\le f_{n+1}$,

$$E_n\subseteq E_{n+1},$$

so $(E_n)$ is increasing.

We claim that

$$\bigcup_{n=1}^{\infty}E_n=E.$$

Indeed, fix $x\in E$.

If $\phi(x)=0$, then

$$f_n(x)\ge0=c\phi(x),$$

so $x\in E_n$ for every $n$.

If $\phi(x)>0$, then because $c<1$,

$$c\phi(x)<\phi(x)\le f(x).$$

Since $f_n(x)\uparrow f(x)$, the increasing numerical sequence $f_n(x)$ must eventually exceed the strictly smaller number $c\phi(x)$. Thus there exists an index $N(x)$ such that

$$f_n(x)\ge c\phi(x) \qquad\text{for all }n\ge N(x).$$

Hence $x\in E_n$ eventually.

Notice that the index $N(x)$ may depend on $x$. We do not claim that there is one $n$ for which $E_n=E$; we only have

$$E_n\uparrow E.$$

**Compare the integrals on $E_n$.**

By the definition of $E_n$,

$$c\phi,\mathbf 1_{E_n}\le f_n$$

on all of $E$. Therefore,

$$c\int_{E_n}\phi = \int_E c\phi,\mathbf 1_{E_n} \le \int_E f_n \le L.$$

Now write the simple function in the form

$$\phi=\sum_{j=1}^{N}a_j\mathbf 1_{A_j}, \qquad a_j>0,$$

where the $A_j$ are measurable and pairwise disjoint. Then

$$\int_{E_n}\phi = \sum_{j=1}^{N}a_j,m(A_j\cap E_n).$$

Since $E_n\uparrow E$,

$$A_j\cap E_n\uparrow A_j,$$

and continuity of measure from below gives

$$m(A_j\cap E_n)\longrightarrow m(A_j).$$

Because the sum is finite,

$$\int_{E_n}\phi\longrightarrow\int_E\phi.$$

Passing to the limit in

$$c\int_{E_n}\phi\le L$$

therefore gives

$$c\int_E\phi\le L.$$

**Return from the lowered test function to the original one.**

The preceding inequality holds for every $c\in(0,1)$. Letting $c\uparrow1$, we obtain

$$\int_E\phi\le L.$$

Thus every actual simple test function $\phi\le f$, not merely its lowered version, satisfies this inequality.

Finally, take the supremum over all nonnegative simple functions $\phi\le f$:

$$\int_E f = \sup_{\substack{\phi\text{ simple}\ 0\le\phi\le f}} \int_E\phi \le L.$$

Combined with the earlier inequality $L\le\int_E f$, this yields

$$\boxed{ \lim_{n\to\infty}\int_E f_n = \int_E f = \int_E\lim_{n\to\infty}f_n}.$$

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_mct_proof_trick.png' | relative_url }}" alt="Two panels with the limit function, an orange staircase test function below it, a dashed lowered copy of the staircase, and a blue curve from the increasing sequence; green bars on the axis mark where the blue curve exceeds the lowered staircase, covering only fragments in the left panel and the whole interval in the right panel" loading="lazy">
  <figcaption>The lowering trick: comparing $f_n$ with a simple $\phi\le f$ directly can fail forever, but the lowered $c\phi$ (dashed) sits strictly below $f$ wherever $\phi>0$, so the sets $E_n=\{f_n\ge c\phi\}$ (green) grow up to $E$. Continuity of measure on the level sets $A_j\cap E_n\uparrow A_j$ then gives $c\int_E\phi\le L$, and $c\uparrow 1$ removes the lowering.</figcaption>
</figure>

**Remark.** The factor $c<1$ is essential because pointwise convergence to $f$ does not imply that $f_n$ ever reaches a test function that equals $f$ somewhere. Lowering the test function creates the strict pointwise gap needed for eventual domination.

</details>
</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_mct_theorem.png' | relative_url }}" alt="Left panel: a family of blue curves increasing pointwise toward a bold black limit curve, with an upward arrow. Right panel: the integrals of the sequence plotted against n climbing monotonically toward a dashed line marking the integral of the limit" loading="lazy">
  <figcaption>Monotone convergence: an increasing sequence $0\le f_1\le f_2\le\cdots$ of measurable functions converging pointwise to $f$ (left) forces the increasing sequence of integrals $\int_E f_n$ to climb exactly to $\int_E f$ (right) — pointwise convergence suffices, no uniformity needed.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The assumption of )</span></p>

The assumption of pointwise convergence here is much weaker than the uniform convergence we usually need for Riemann integration.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Benefits and Caveats)</span></p>

* **Putting the limit under the integral:** the Monotone Convergence Theorem is a fundamental tool in real analysis that strictly allows you to interchange the limit and the integral. It guarantees that the limit of the integrals is equal to the integral of the limit function

**Important Caveats**

* **Decreasing Sequences:** The Monotone Convergence Theorem does not natively work for decreasing sequences unless you modify it (e.g., multiplying by -1), or switch to using the **Dominated Convergence Theorem**.
* **Riemann Integral:** While historically applicable to well-behaved Riemann integrals under strict conditions, the theorem is primarily a cornerstone of Lebesgue integration

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Computing Integrals via Simple Approximations)</span></p>

Let $f \in L^+(E)$, and let $\lbrace \phi_n \rbrace$ be a sequence of simple functions with $0 \le \phi_1 \le \phi_2 \le \cdots \le f$ and $\phi_n \to f$ pointwise. Then 

$$\int_E f = \lim_{n \to \infty} \int_E \phi_n$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Additivity of the Nonnegative Integral)</span></p>

$$f, g \in L^+(E) \implies \int_E (f + g) = \int_E f + \int_E g$$

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

Let $\lbrace f_n \rbrace$ be a sequence in $L^+(E)$. Then 

$$\int_E \sum_n f_n = \sum_n \int_E f_n$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Zero Integral Characterization)</span></p>

Let $f \in L^+(E)$. Then

$$\int_E f = 0 \quad\iff\quad f = 0 \quad\text{almost everywhere on } E.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>($\Rightarrow$)</strong> If $f = 0$ a.e., then $f \le 0$ a.e., so $\int_E f \le \int_E 0 = 0$.</p>
    <p><strong>($\Leftarrow$)</strong> Define $F_n = \lbrace x \in E : f(x) > 1/n \rbrace$ and $F = \lbrace x \in E : f(x) > 0 \rbrace = \bigcup_n F_n$. Then $\frac{1}{n} m(F_n) = \int_{F_n} \frac{1}{n} \le \int_{F_n} f \le \int_E f = 0$, so $m(F_n) = 0$ for all $n$. By continuity of measure, $m(F) = \lim m(F_n) = 0$. $\square$</p>
  </details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>



</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Monotone Convergence — a.e. Version)</span></p>

If $\lbrace f_n \rbrace$ is a sequence in $L^+(E)$ with $f_1 \le f_2 \le \cdots$ a.e. on $E$ and $\lim f_n = f$ a.e. on $E$, then 

$$\int_E f = \lim_{n \to \infty} \int_E f_n$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Finite Integral Implies Finite a.e.)</span></p>

$$f \in L^+(E) \text{ and } \int_E f < \infty \implies \lbrace x \in E : f(x) = \infty \rbrace \quad\text{has measure zero}.$$

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

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_fatou_sliding_bump.png' | relative_url }}" alt="Three unit-area bumps drawn with fading blue color at successive positions sliding to the right with an arrow labeled n to infinity; a bold red line along zero marks the pointwise liminf" loading="lazy">
  <figcaption>Why Fatou's lemma is only an inequality: the sliding bump $f_n$ has $\int f_n=1$ for every $n$, yet $f_n\to 0$ pointwise (red line), so $\int\liminf_n f_n=0<1=\liminf_n\int f_n$ — the mass escapes to infinity and the integral of the liminf never sees it.</figcaption>
</figure>

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

1. For all $c \in \mathbb{R}$, $cf$ is integrable and 
   
   $$\int_E cf = c \int_E f.$$

2. $f + g$ is integrable and 
   
   $$\int_E (f + g) = \int_E f + \int_E g.$$

3. If $A, B$ are disjoint measurable sets, 
   
   $$\int_{A \cup B} f = \int_A f + \int_B f.$$

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

Let $g : E \to [0, \infty)$ be a nonnegative integrable function, and let $\lbrace f_n \rbrace$ be a sequence of real-valued measurable functions such that 

* **(1)** $\lvert f_n \rvert \le g$ a.e. for all $n$,
* **(2)** $f_n(x) \to f(x)$ pointwise a.e. on $E$.

Then

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

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_dct_theorem.png' | relative_url }}" alt="A green dashed envelope curve and its mirror image below zero bound a shaded tube; several blue curves wiggle inside the tube and settle onto a bold black limit curve" loading="lazy">
  <figcaption>Dominated convergence: the integrable envelope $g$ pins the whole sequence inside the tube $\pm g$ (green), so no mass can escape to infinity as in Fatou's sliding-bump example — pointwise convergence $f_n\to f$ a.e. already forces $\int_E f_n\to\int_E f$.</figcaption>
</figure>

### Density of Nice Functions in $L^1([a,b])$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">($L^1$-Approximation by Bounded Measurable Functions / Truncation of $L^1$ Functions)</span></p>

Let $a < b$. Suppose that $f : [a,b] \to \mathbb{R}$ is Lebesgue integrable.

Let $\varepsilon > 0$. Prove that there exists a bounded measurable function $h : [a,b] \to \mathbb{R}$ such that
   
$$\int_a^b |f(x)-h(x)|\,dx < \varepsilon.$$

*Hint:* Let $E_n = f^{-1}([-n,n])$ and $h_n = f\chi_{E_n}$. Prove that

$$\lim_{n\to\infty}\int_a^b |f(x)-h_n(x)|\,dx = 0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution: DCT perspective</summary>

Let

$$E_n=f^{-1}([-n,n]), \qquad h_n=f\chi_{E_n}.$$

Each $E_n$ is measurable, hence $h_n$ is measurable, and $\lvert h_n\rvert\le n$, so $h_n$ is bounded.

Define the function:

$$g_n:=|f-h_n|.$$

For every fixed $x\in[a,b]$, since $f(x)$ is finite, there exists $N$ such that

$$|f(x)|\le N.$$

Hence $x\in E_n$ for every $n\ge N$, and therefore

$$g_n(x)=|f(x)-h_n(x)|=0$$

for every sufficiently large $n$. Thus $g_n(x)\to 0$ pointwise. Also, $g_n(x)\le \lvert f(x)\rvert$. Since $f$ is Lebesgue integrable, then $\lvert f\rvert\in L^1$. Therefore DCT gives

$$\lim_{n\to\infty}\int_a^b g_n(x)dx = \int_a^b0dx = 0.$$

That is,

$$\boxed{\lim_{n\to\infty}\int_a^b|f-h_n|=0}.$$

Hence, for every $\varepsilon>0$, some $h_n$ satisfies

$$\int_a^b|f-h_n|<\varepsilon.$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution: MCT perspective</summary>

Let

$$E_n=f^{-1}([-n,n]), \qquad h_n=f\chi_{E_n}.$$

Each $E_n$ is measurable, hence $h_n$ is measurable, and

$$|h_n|\le n,$$

so $h_n$ is bounded.

Instead of looking at the decreasing error $g_n$, consider the retained part:

$$|h_n|=|f|\chi_{E_n}.$$

Since

$$E_n\subseteq E_{n+1},$$

we have

$$0\le |h_n|\le |h_{n+1}|$$

and

$$|h_n|\to|f|$$

pointwise. Hence MCT gives

$$\lim_{n\to\infty}\int_a^b|h_n| = \int_a^b|f|.$$

For this particular construction we also showed

$$|f-h_n|=|f|-|h_n|.$$

Therefore

$$
\begin{aligned}
\lim_{n\to\infty}\int_a^b|f-h_n|
&= \lim_{n\to\infty} \left( \int_a^b|f|-\int_a^b|h_n| \right)\\
&= \int_a^b|f|-\int_a^b|f|\\
&=0.
\end{aligned}
$$

So DCT studies the **discarded part tending to zero**, while MCT studies the **retained part increasing to the whole function**.

</details>
</div>

<!-- <figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_l1_truncation.png' | relative_url }}" alt="An integrable function with two tall spikes; a bold blue truncated version follows it but drops to zero where the function exceeds the dashed level n, and the spike areas above the cut are shaded red as the error mass" loading="lazy">
  <figcaption>Truncation $h_n=f\chi_{E_n}$ with $E_n=f^{-1}([-n,n])$: $h_n$ agrees with $f$ wherever $|f|\le n$ and vanishes on the spikes. The $L^1$-error $\int_a^b|f-h_n|$ is exactly the shaded mass sitting over $E_n^c=\{|f|>n\}$, and it shrinks to $0$ as $n\to\infty$ because $f$ is integrable.</figcaption>
</figure> -->

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">($L^1$-Approximation by Step Functions / Density of Step Functions in $L^1([a,b])$)</span></p>

Let $a < b$. Suppose that $f : [a,b] \to \mathbb{R}$ is Lebesgue integrable.

Let $\varepsilon > 0$. Prove that there exists a step function $\psi : [a,b] \to \mathbb{R}$ such that

$$\int_a^b |f(x)-\psi(x)|\,dx < \varepsilon.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

This is the accumulated-error argument we discussed.

From the previous part, $f$ can be approximated arbitrarily closely in $L^1$ by a bounded measurable $h_n$.

From the previous approximation results, a bounded measurable $h_n$ can in turn be approximated arbitrarily closely by step functions, through the chain

$$h_n \longrightarrow \text{simple function} \longrightarrow \text{step function}.$$

So let $\varepsilon>0$ and split it as

$$\varepsilon=\varepsilon_1+\varepsilon_2, \qquad \varepsilon_1,\varepsilon_2>0.$$

First choose $n$ sufficiently large that

$$\int_a^b|f-h_n|<\varepsilon_1.$$

Then, keeping this $h_n$ fixed, choose a step function $\phi$ sufficiently close to $h_n$ that

$$\int_a^b|h_n-\phi|<\varepsilon_2.$$

Now use precisely the triangle inequality we discussed:

$$|f-\phi| \le |f-h_n|+|h_n-\phi|.$$

Integrating,

$$
\begin{aligned}
\int_a^b|f-\phi|
&\le
\int_a^b|f-h_n|
+
\int_a^b|h_n-\phi|\
&<
\varepsilon_1+\varepsilon_2\
&=\varepsilon.
\end{aligned}
$$

Thus there exists a step function $\phi$ such that

$$\boxed{\int_a^b|f-\phi|<\varepsilon}.$$

The essential point is exactly what we said earlier: **the approximation errors accumulate, so choose each approximation far enough along that their total is below $\varepsilon$.**

If we expand the hidden middle chain, it is simply

$$f\to h_n\to\varphi_m\to\phi_k$$

and

$$|f-\phi_k| \le |f-h_n|+|h_n-\varphi_m|+|\varphi_m-\phi_k|.$$

Nothing deeper is happening.

</details>
</div>

<!-- <figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_l1_step_approximation.png' | relative_url }}" alt="A smooth bounded curve overlaid by a blue step function on ten intervals; the thin region between the two graphs is shaded orange" loading="lazy">
  <figcaption>Density of step functions in $L^1([a,b])$: the $L^1$-distance $\int_a^b|f-\psi|$ is the shaded area between the graphs. Refining the partition (after truncating $f$ to a bounded $h$ as in part (a)) makes this area smaller than any $\varepsilon$.</figcaption>
</figure> -->

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">($L^1$-Approximation by Continuous Functions Vanishing at the Endpoints / Density of $C_0([a,b])$ in $L^1([a,b])$)</span></p>

Let $a < b$. Suppose that $f : [a,b] \to \mathbb{R}$ is Lebesgue integrable.

Let $\varepsilon > 0$. Prove that there exists a continuous function $g : [a,b] \to \mathbb{R}$ such that $g(a)=g(b)=0$ and

$$\int_a^b |f(x)-g(x)|\,dx < \varepsilon.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

We do not separately work through the construction of $g$; the same accumulated-error argument applies using the previous step-to-continuous approximation result.

Let

$$\varepsilon=\varepsilon_1+\varepsilon_2.$$

By part step the function case, choose a step function $\phi$ such that

$$\int_a^b|f-\phi|<\varepsilon_1.$$

By the previous exercise, choose a continuous $g$ with

$$g(a)=g(b)=0$$

and

$$\int_a^b|\phi-g|<\varepsilon_2.$$

Then

$$|f-g|\le |f-\phi|+|\phi-g|,$$

so

$$
\begin{aligned}
\int_a^b|f-g|
&\le \int_a^b|f-\phi| + \int_a^b|\phi-g|\\
&< \varepsilon_1+\varepsilon_2\\
&=\varepsilon.
\end{aligned}
$$

Hence

$$\boxed{\int_a^b|f-g|<\varepsilon}.$$

So the whole Exercise has the structure we kept coming back to:

$$\boxed{ f \longrightarrow \text{bounded measurable} \longrightarrow \text{step} \longrightarrow\text{continuous}, }$$

with the $L^1$-errors combined using the triangle inequality.

</details>
</div>

<!-- <figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_l1_continuous_approximation.png' | relative_url }}" alt="A dashed step function and a green continuous function that follows it exactly except in narrow red vertical bands around each jump and at both endpoints, where the green curve ramps linearly and vanishes at the interval ends" loading="lazy">
  <figcaption>From a step function $\psi$ (dashed) to a continuous $g$ (green) with $g(a)=g(b)=0$: they differ only inside narrow bands $E$ around the jumps and endpoints, where $|\psi-g|\le 2B$. Hence $\int_a^b|\psi-g|\le 2B\,m(E)<\varepsilon$, and combining with parts (a) and (b) gives density of $C_0([a,b])$ in $L^1$.</figcaption>
</figure> -->

### Riemann and Lebesgue Integrals Agree

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 126</span></p>

Let $f \in C([a, b])$ for some real numbers $a < b$. Then 

$$\int_{[a,b]} f = \int_a^b f(x)\,dx$$

in other words, $f$ is integrable and the Riemann and Lebesgue integrals agree.

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


<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span>(Complex-Valued Integrable Functions)</p>

Everything proved for real integrable functions carries over to complex-valued integrable functions: we define $f : E \to \mathbb{C}$ to be Lebesgue integrable if $\int_E \lvert f \rvert < \infty$, in which case

$$\int_E f = \int_E \operatorname{Re} f + i \int_E \operatorname{Im} f.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span>(Generalized theorems results for complex-valued integrable functions)</p>

Results like linearity of the integral and the Dominated Convergence Theorem generalize.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 127</span></p>

$$f : E \to \mathbb{C} \quad\text{is integrable} \implies \left\lvert \int_E f \right\rvert \le \int_E \lvert f \rvert.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>If $\int_E f = 0$ the result is clear. Otherwise, define $\alpha = \overline{(\int_E f)} / \lvert \int_E f \rvert$, so $\lvert \alpha \rvert = 1$. Then</p>
    $$\left\lvert \int_E f \right\rvert = \alpha \int_E f = \int_E \alpha f = \operatorname{Re} \int_E \alpha f = \int_E \operatorname{Re}(\alpha f) \le \int_E \lvert \operatorname{Re}(\alpha f) \rvert \le \int_E \lvert \alpha f \rvert = \int_E \lvert f \rvert,$$
    <p>using $\operatorname{Re}(z) \le \lvert z \rvert$ and $\lvert \alpha \rvert = 1$. $\square$</p>
  </details>
</div>

### Riemann–Lebesgue Lemma for $L^1$ Functions

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Riemann-Lebesgue Lemma: Vanishing of Fourier Coefficients for Integrable Functions)</span></p>

Suppose that $f : [-\pi,\pi] \to \mathbb{C}$ is Lebesgue integrable. The **Fourier coefficients** $\lbrace \hat f(n)\rbrace\_{n\in\mathbb{Z}}$ of $f$ are defined via

$$\hat f(n) := \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}\,dx, \qquad n\in\mathbb{Z}.$$

Riemann-Lebesgue lemma states:

$$\lim_{|n|\to\infty} |\hat f(n)| = 0. \tag{\(\dagger\)}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

*Hint:* Prove $(\dagger)$ for step functions first. Then use problem 3 and an approximation argument.

**The proof for step functions first.**

According to the hint, we consider the step function $\phi$. The function $\phi$ attains finite number of values from a bounded set $\lvert k_1, \cdots, k_m\rvert$, then let

$$\phi(x) = \sum_j^m k_j [x \in I_j],$$

where $I_j = \phi^{-1}(k_j)$. Then we can write the Fourier coefficients as

$$
\begin{aligned}
\hat \phi(n) 
&= \frac{1}{2\pi}\int_{-\pi}^{\pi} \phi(x)e^{-inx}\,dx \\
&= \frac{1}{2\pi}\int_{-\pi}^{\pi} \sum_j^m k_j [x \in I_j] e^{-inx}\,dx \\
&= \frac{1}{2\pi} \sum_j^m k_j \int_{I_j} e^{-inx}\,dx.
\end{aligned}
$$

To compute the integral in the sum, let $I = [a, b]$. Then 

$$\int_{I} e^{-inx}\,dx = \int_{a}^{b} e^{-inx}\,dx = \dfrac{e^{-ibn} - e^{-ian}}{-in}.$$

Then

$$\left|\int_a^b e^{-inx},dx\right| \le \frac{|e^{-inb}|+|e^{-ina}|}{|n|} = \frac{2}{|n|}.$$

This is the crucial $1/\lvert n\rvert$ decay. Note $\lvert e^{-inx}\rvert = \sqrt{\cos(nx)^2 + \sin(nx)^2} = 1$.

**The proof for general Lebesgue integrable functions.**

We know from previous exercises that for every Lebesgue integrable function $f$ there exists a step function $\phi$ on the same domain (let it be $[a,b]$) such that for any $\varepsilon \gt 0$ we have

$$\int_a^b |f(x)-\phi(x)| \lt \varepsilon.$$

Then 

$$
\begin{aligned}
|\hat f(n)|
&= \left|\frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}\,dx\right| \\
&= \left|\frac{1}{2\pi}\int_{-\pi}^{\pi} (f(x)-\phi(x)+\phi(x)) e^{-inx}\,dx\right| \\
&= \left|\frac{1}{2\pi}\int_{-\pi}^{\pi} (f(x)-\phi(x)) e^{-inx}\,dx + \frac{1}{2\pi}\int_{-\pi}^{\pi} \phi(x) e^{-inx}\,dx\right| \\
&\leq \left|\frac{1}{2\pi}\int_{-\pi}^{\pi} (f(x)-\phi(x)) e^{-inx}\,dx\right| + |\hat \phi(n)|.
\end{aligned}
$$

*Note that these integrals are complex numbers, so there is no order relation $\leq$ here. This is probably the most important correction compared to the previous one.*

We have already proven in the first part that the second term goes to zero. For the first term, we use Hölder inequality, specifically the $L^1$-$L^\infty$ version of it:

$$\left|\int_{-\pi}^{\pi}(f-\phi)e^{-inx},dx\right| \le |f-\phi|_{L^1}, |e^{-inx}|_{L^\infty} = |f-\phi|_{L^1}.$$

And

$$|e^{-inx}|_\infty=1.$$

The norm $\lvert f-\phi\rvert\_{L^1}$ is any close by the $L^1$ approximation.

We still need the final **$\varepsilon$-argument**. Saying that the first term “is arbitrarily close” and the second “goes to zero” expresses the idea, but to prove the limit we should make the order explicit:

Given $\varepsilon>0$,

* first choose a fixed step function $\phi$ so that its $L^1$-approximation error makes the first term $<\varepsilon/2$;
* then, with this $\phi$ fixed, choose $N$ such that for $\lvert n\rvert\ge N$,
  
  $$|\hat\phi(n)|<\varepsilon/2.$$
  
Then the sum is $<\varepsilon$.

</details>
</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_riemann_lebesgue.png' | relative_url }}" alt="Left panel: a three-step function multiplied by a fast cosine, with positive lobes shaded green and negative lobes shaded red showing cancellation on each step. Right panel: a stem plot of Fourier coefficient magnitudes decaying to zero as n grows" loading="lazy">
  <figcaption>Riemann–Lebesgue mechanism: on each constant step of $\psi$ the oscillation $e^{-inx}$ produces adjacent lobes of opposite sign that cancel, leaving only boundary contributions of size $O(1/n)$ (left). Approximating $f\in L^1$ by step functions transfers the decay to $\hat f$: $|\hat f(n)|\to 0$ (right).</figcaption>
</figure>

### Integrals over Subintervals and Improper Integrals

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The Indefinite Integral Is Continuous)</span></p>

Let $f$ be integrable on $[a, b)$. Then $F(x) = \int\_{[x, b)} f$ is continuous on $[a, b]$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

For $x < y$, $\lvert F(x) - F(y) \rvert = \lvert \int f\chi\_{[x, y)} \rvert \le \int \lvert f \rvert \chi\_{[x, y)}$. As $y - x \to 0$ the integrand tends to $0$ pointwise and is dominated by $\lvert f \rvert$, so the right side tends to $0$ by dominated convergence. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Improperly Riemann Integrable but Not Lebesgue Integrable)</span></p>

Let $f(x) = x^{-1}\cos(1/x)$ on $(0, 1]$. The improper Riemann integral exists: substituting $u = 1/x$,

$$\lim_{t \downarrow 0}\int_t^1 \frac{\cos(1/x)}{x}\,dx = \lim_{R \to \infty}\int_1^R \frac{\cos u}{u}\,du,$$

which converges (integrate by parts, or by the alternating series test on the intervals between zeros of $\cos$). But $f$ is **not** Lebesgue integrable: otherwise $\lvert f \rvert$ would be integrable, so $\int\_{[t, 1]} \lvert f \rvert \le \int\_{(0,1]} \lvert f \rvert < \infty$ for all $t$. Yet (by Theorem 126 on each $[t, 1]$)

$$\int_t^1 \frac{\lvert\cos(1/x)\rvert}{x}\,dx = \int_1^{1/t} \frac{\lvert \cos u \rvert}{u}\,du \ge \sum_{1 \le k < \frac{1}{\pi t} - 1} \frac{2}{(k+1)\pi} \xrightarrow{t \to 0} \infty,$$

since $\int\_{k\pi}^{(k+1)\pi}\lvert \cos u \rvert\,du = 2$. The Lebesgue integral is **absolutely** convergent by design: it cannot exploit cancellation the way conditionally convergent improper integrals do.

</div>

### Continuity of Translation in $L^1$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Continuity in the Mean)</span></p>

For $f \in L^1(\mathbb{R})$ and $t \in \mathbb{R}$, let $f\_t(x) = f(x - t)$. Then $f\_t \in L^1(\mathbb{R})$ with $\lVert f\_t \rVert\_1 = \lVert f \rVert\_1$, and

$$\lim_{t \to 0}\int_{\mathbb{R}} \lvert f(x - t) - f(x) \rvert\,dx = 0.$$

Consequently, $t \mapsto f\_t$ is a continuous curve $\mathbb{R} \to L^1(\mathbb{R})$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Translation invariance of Lebesgue measure gives $\int \chi\_{E + t} = \int \chi\_E$, hence $\int \varphi\_t = \int \varphi$ for simple $\varphi$, and by monotone convergence $\int \lvert f\_t \rvert = \int \lvert f \rvert$.

**Step functions first.** For $\varphi = \chi\_{[a, b)}$, $\varphi\_t - \varphi$ is $\pm 1$ on two intervals of length at most $\lvert t \rvert$, so $\lVert \varphi\_t - \varphi \rVert\_1 \le 2\lvert t \rvert$. By the triangle inequality, $\lVert \varphi\_t - \varphi \rVert\_1 \le 2\lvert t \rvert \sum\_i \lvert c\_i \rvert$ for a step function $\varphi = \sum\_i c\_i \chi\_{[a\_i, b\_i)}$.

**Approximation.** Given $\varepsilon > 0$, first truncate: $f\chi\_{[-L, L]} \to f$ in $L^1$ by dominated convergence. Then approximate $f\chi\_{[-L, L]}$ in $L^1([-L, L])$ by a step function (Section "Density of Nice Functions"). This gives a step function $\varphi$ with $\lVert f - \varphi \rVert\_1 < \varepsilon$. Then

$$\lVert f_t - f \rVert_1 \le \lVert (f - \varphi)_t \rVert_1 + \lVert \varphi_t - \varphi \rVert_1 + \lVert \varphi - f \rVert_1 \le 2\varepsilon + \lVert \varphi_t - \varphi \rVert_1 < 3\varepsilon$$

for $\lvert t \rvert$ small. Continuity at any $s$ follows from $\lVert f\_t - f\_s \rVert\_1 = \lVert f\_{t-s} - f \rVert\_1$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($C_c(\mathbb{R})$ Is Dense in $L^1(\mathbb{R})$)</span></p>

The same two-step approximation (truncate to $[-L, L]$, then approximate on $[-L, L]$ by a continuous $g$ with $g(-L) = g(L) = 0$ as in Section "Density of Nice Functions", extended by $0$) shows that continuous functions vanishing outside a compact set are dense in $L^1(\mathbb{R})$. For a single indicator this is explicit: the trapezoid $g\_n$ equal to $1$ on $[a, b]$, $0$ outside $[a - \frac1n, b + \frac1n]$ and linear in between has $\lVert g\_n - \chi\_{[a, b)} \rVert\_1 \le \frac{1}{n}$.

</div>

### Integral Operators from $L^1$ into $C$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Continuous Kernels Smooth Out $L^1$ Functions)</span></p>

Let $G \in C([0,1] \times [0,1])$. For $f \in L^1([0,1])$ define $(Tf)(x) = \int\_{[0,1]} G(x, y) f(y)\,dy$. Then $Tf \in C([0, 1])$, and $T : L^1([0,1]) \to C([0,1])$ is a bounded linear operator with

$$\lVert Tf \rVert_\infty \le \lVert G \rVert_\infty \lVert f \rVert_{L^1}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

For fixed $x$, $y \mapsto G(x, y)f(y)$ is measurable and bounded by $\lVert G \rVert\_\infty \lvert f(y) \rvert$, so $Tf(x)$ is well defined (and unchanged if $f$ is changed on a null set), with $\lvert Tf(x) \rvert \le \lVert G \rVert\_\infty \lVert f \rVert\_1$. By uniform continuity of $G$ on the compact square, for $\delta > 0$ there is $\eta > 0$ with $\sup\_y \lvert G(x, y) - G(x^{\prime}, y) \rvert < \delta$ whenever $\lvert x - x^{\prime} \rvert < \eta$, and then

$$\lvert Tf(x) - Tf(x') \rvert \le \int \lvert G(x, y) - G(x', y) \rvert \lvert f(y) \rvert\,dy \le \delta \lVert f \rVert_1. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Integral as a Functional)</span></p>

$f \mapsto \int f$ is a bounded linear functional on $L^1$ (Proposition 127 gives $\lvert \int f \rvert \le \lVert f \rVert\_1$), and its norm is exactly $1$: equality holds for every $f \ge 0$.

</div>

## The Lebesgue Integral via Step Functions (Melrose's Approach)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Integral First, Measure Later)</span></p>

The chapters above follow the classical route: outer measure $\to$ measurable sets $\to$ measurable functions $\to$ integral. The 2009 version of MIT 18.102 (R. Melrose) goes the other way, following an idea of Mikusiński: **the Lebesgue integral is the completion of the step functions in the $L^1$ norm**, and Lebesgue measure is derived from the integral afterwards. The construction is self-contained and uses only the covering lemma for intervals plus the series tricks from the completion of a normed space. Both routes produce the same space $L^1(\mathbb{R})$.

</div>

Throughout this section an **interval** means $[a, b)$ — then length $b - a = 0$ only for the empty interval, and $[a, b) = [a, t) \cup [t, b)$ for $a < t < b$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Step Functions and Their Integral)</span></p>

A **step function** is $f = \sum\_{i=1}^N c\_i \chi\_{[a\_i, b\_i)}$, $c\_i \in \mathbb{C}$, and its integral is $\int f = \sum\_i c\_i (b\_i - a\_i)$. This does not depend on the presentation (refine to a common disjoint family of intervals). Step functions form a linear space, closed under $\lvert \cdot \rvert$, $\max$, $\min$, and $\lVert f \rVert\_{L^1} = \int \lvert f \rvert$ is a **norm** on it.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Covering Lemma)</span></p>

Let $C\_i = [a\_i, b\_i)$, $i \in \mathbb{N}$, be intervals.

1. If the $C\_i$ are pairwise disjoint and all contained in $[a, b)$, then $\sum\_{i} (b\_i - a\_i) \le b - a$.
2. If $[a, b) \subset \bigcup\_{i} C\_i$, then $\sum\_{i} (b\_i - a\_i) \ge b - a$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

For finitely many intervals both statements follow by inserting all endpoints as division points. (1) then holds for every finite subfamily, hence for the whole series.

(2) needs compactness. Fix $\delta > 0$ and enlarge $C\_i$ to the open interval $(a\_i - 2^{-i}\delta, b\_i)$. These open intervals cover the compact set $[a, b - \delta]$, so finitely many do (Heine–Borel); then the corresponding half-open intervals $[a\_i - 2^{-i}\delta, b\_i)$, $i \le N$, cover $[a, b - \delta)$, and the finite case gives

$$\sum_{i \le N}\big(b_i - a_i + 2^{-i}\delta\big) \ge b - a - \delta \quad\Longrightarrow\quad \sum_i (b_i - a_i) \ge b - a - 2\delta.$$

Let $\delta \to 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Monotonicity Lemma)</span></p>

If $f\_n$ are step functions with $f\_n(x) \downarrow 0$ for every $x \in \mathbb{R}$, then $\int f\_n \to 0$.

More generally, if $g\_n$ are real step functions, nondecreasing in $n$, with $\lim\_n g\_n(x) \in [0, \infty]$ for every $x$, then $\lim\_n \int g\_n \in [0, \infty]$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

The point is that we get convergence of integrals **without uniform convergence**. All $f\_n$ vanish outside the interval $[a, b)$ where $f\_1$ lives, and $f\_n \le A := \max f\_1$. Fix $\varepsilon > 0$ and let $S\_j = \lbrace x \in [a, b) : f\_j(x) \le \varepsilon \rbrace$, a finite union of intervals. The $S\_j$ increase, and $\bigcup\_j S\_j = [a, b)$ because $f\_j(x) \to 0$. So $B\_1 = S\_1$, $B\_j = S\_j \setminus S\_{j-1}$ are disjoint finite unions of intervals with union $[a, b)$, and **both** halves of the covering lemma give $\sum\_j \ell(B\_j) = b - a$. Choose $N$ with $\sum\_{j > N} \ell(B\_j) < \varepsilon$, i.e. $\ell([a, b) \setminus S\_N) < \varepsilon$. For $k \ge N$, $f\_k \le f\_N \le \varepsilon$ on $S\_N$ and $f\_k \le A$ on the rest, so

$$\int f_k \le \varepsilon(b - a) + A\varepsilon.$$

For the general statement, $f\_n = \max(0, -g\_n)$ are step functions decreasing to $0$ pointwise, and $\int g\_n \ge -\int f\_n \to 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Integrable Function, Null Set)</span></p>

A function $f : \mathbb{R} \to \mathbb{C}$ is **Lebesgue integrable**, written $f \in \mathcal{L}^1(\mathbb{R})$, if there is a sequence of step functions $f\_n$ with

$$\sum_n \int \lvert f_n \rvert < \infty \qquad\text{and}\qquad f(x) = \sum_n f_n(x) \quad \text{whenever } \sum_n \lvert f_n(x) \rvert < \infty.$$

(At points where the series does not converge absolutely, $f$ is unconstrained.) A set $E \subset \mathbb{R}$ has **measure zero** if there is a sequence of step functions $f\_n$ with $\sum\_n \int \lvert f\_n \rvert < \infty$ and $\sum\_n \lvert f\_n(x) \rvert = \infty$ for all $x \in E$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Interlacing Trick)</span></p>

Almost every proof in this approach uses one device. Interlacing two absolutely summable series $(f\_1, g\_1, f\_2, g\_2, \dots)$ gives an absolutely summable series that converges absolutely at $x$ iff **both** do. In particular, inserting pairs $(h\_k, -h\_k)$ does not change the sum where it converges, but **shrinks the convergence set** to the points where $\sum\_k \lvert h\_k(x) \rvert < \infty$. So we can always discard a null set from the convergence set at no cost.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Basic Properties of $\mathcal{L}^1$)</span></p>

1. $\mathcal{L}^1(\mathbb{R})$ is a linear space, and if $f \in \mathcal{L}^1$ and $g = f$ off a set of measure zero, then $g \in \mathcal{L}^1$.
2. A countable union of sets of measure zero has measure zero.
3. **The integral is well defined:** $\int f := \sum\_n \int f\_n$ does not depend on the approximating series, and agrees with the step-function integral on step functions.
4. If $f \in \mathcal{L}^1$ then $\lvert f \rvert \in \mathcal{L}^1$, and $\int \lvert f \rvert = \lim\_N \int \lvert \sum\_{n \le N} f\_n \rvert \le \sum\_n \int \lvert f\_n \rvert$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

**(1)** For $f + f^{\prime}$, interlace the two series: the result converges absolutely exactly where both do, and there its sum is $f(x) + f^{\prime}(x)$ (absolutely convergent series can be rearranged). (Simply adding the series termwise would be a mistake: the sum could converge absolutely at points where neither series does.) For $g = f$ off a null set $E$ with witness series $h\_k$, interlace $(f\_k, h\_k, -h\_k)$: it converges absolutely only off $E$, and there to $f = g$.

**(2)** Let $E\_j$ be null with witness series $f^{(j)}\_n$. Dropping finitely many terms keeps divergence on $E\_j$, so we may assume $\sum\_n \int \lvert f\_n^{(j)} \rvert < 2^{-j}$. Any enumeration of the double array is then absolutely summable and diverges absolutely on every $E\_j$.

**(3)** If $f\_n$ and $f^{\prime}\_n$ both represent $f$, interlace $(f\_k, -f^{\prime}\_k)$: it represents the zero function, and $\sum \int$ of it equals $\sum \int f\_k - \sum \int f\_k^{\prime}$. So it suffices to show: **if $h\_n$ represents $0$, then $\sum\_n \int h\_n = 0$.** Fix $N$ and consider the step-function series $g\_1 = \sum\_{j \le N} h\_j$, $g\_k = \lvert h\_{N+k-1} \rvert$ for $k > 1$. Its partial sums $G\_p$ are nondecreasing. Where $\sum \lvert h\_n(x) \rvert = \infty$, $G\_p(x) \to +\infty$; elsewhere $G\_p(x) \ge \sum\_{j \le N+p-1} h\_j(x) \to 0$. By the monotonicity lemma $\lim\_p \int G\_p \ge 0$, i.e.

$$\sum_{j \le N}\int h_j + \sum_{k > N}\int \lvert h_k \rvert \ge 0 \qquad \text{for every } N.$$

Letting $N \to \infty$, the tail vanishes and $\sum\_j \int h\_j \ge 0$; applying this to $-h\_n$ gives equality.

**(4)** Put $g\_1 = \lvert f\_1 \rvert$, $g\_k = \lvert \sum\_{j \le k} f\_j \rvert - \lvert \sum\_{j < k} f\_j \rvert$. Then $\lvert g\_k \rvert \le \lvert f\_k \rvert$, so $\lbrace g\_k \rbrace$ is absolutely summable with partial sums $\lvert \sum\_{j \le N} f\_j \rvert \to \lvert f \rvert$ where $\sum \lvert f\_j \rvert < \infty$. Interlacing with $(f\_k, -f\_k)$ restricts the convergence set to exactly that set, so $\lvert f \rvert \in \mathcal{L}^1$ with the stated integral. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Series of Integrable Functions; $L^1$ Is Complete)</span></p>

Let $f\_n \in \mathcal{L}^1(\mathbb{R})$ with $\sum\_n \int \lvert f\_n \rvert < \infty$. Then $E = \lbrace x : \sum\_n \lvert f\_n(x) \rvert = \infty \rbrace$ has measure zero, any $f$ with $f(x) = \sum\_n f\_n(x)$ off $E$ is integrable, and

$$\int f = \sum_n \int f_n, \qquad \int \Big\lvert f - \sum_{n < N} f_n \Big\rvert \le \sum_{n \ge N}\int \lvert f_n \rvert \to 0.$$

Consequences:

* $\int \lvert f \rvert = 0 \iff f = 0$ off a set of measure zero (**null function**).
* $L^1(\mathbb{R}) = \mathcal{L}^1(\mathbb{R})/\mathcal{N}$, with $\mathcal{N}$ the null functions, is a normed space with $\lVert [f] \rVert = \int \lvert f \rvert$, and it is a **Banach space**.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

This is the completeness argument from the completion of a normed space, made concrete. Represent each $f\_n$ by step functions $f\_{n,j}$ and regroup heads so that $\sum\_j \int \lvert f\_{n,j} \rvert \le \int \lvert f\_n \rvert + 2^{-n+1}$. The double array is then absolutely summable, so enumerating it (say along anti-diagonals) as a single sequence $h\_1, h\_2, \dots$ gives an absolutely summable series of step functions. Let $E^{\prime} = \lbrace x : \sum\_{n,j} \lvert f\_{n,j}(x) \rvert = \infty \rbrace$, a null set by definition (with witness $h\_k$). Off $E^{\prime}$, each $\sum\_j f\_{n,j}(x)$ converges absolutely, hence equals $f\_n(x)$; so $\sum\_n \lvert f\_n(x) \rvert < \infty$ (thus $E \subset E^{\prime}$ is null), and by rearrangement $\sum\_k h\_k(x) = \sum\_n f\_n(x) = f(x)$. So $h\_k$ represents $f$, whence $f \in \mathcal{L}^1$ and $\int f = \sum\_{n,j}\int f\_{n,j} = \sum\_n \int f\_n$. Applying part (4) of the previous theorem to the tail series $\sum\_{n \ge N} f\_n$ gives the estimate.

*Null functions.* If $f$ vanishes off a null set $E$ with witness $h\_k$, the series $(h\_k, -h\_k)$ represents $f$ and has integral $0$. Conversely, if $\int \lvert f \rvert = 0$, apply the theorem to the series with every term equal to $\lvert f \rvert$: it is absolutely summable (all integrals are $0$), so it converges pointwise off a null set — but it diverges wherever $f(x) \ne 0$.

*Completeness.* An absolutely summable series of classes $[f\_n]$ in $L^1$ has, by the estimate, $\lVert [f] - \sum\_{n < N}[f\_n] \rVert \to 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Monotone Convergence, Fatou, Dominated Convergence)</span></p>

* **Monotonicity.** If $f\_j \in \mathcal{L}^1$ is monotone (increasing or decreasing in $j$) and $\int f\_j$ is **bounded**, then $f = \lim\_j f\_j$ exists and is finite off a null set, $f \in \mathcal{L}^1$, and $\int \lvert f - f\_j \rvert \to 0$.
* **Fatou.** If $f\_j \ge 0$ are integrable with $\int f\_j$ bounded, then $\liminf\_j f\_j$ is integrable and $\int \liminf\_j f\_j \le \liminf\_j \int f\_j$.
* **Dominated convergence.** If $\lvert f\_j \rvert \le h \in \mathcal{L}^1$ a.e. and $f\_j \to f$ a.e., then $f \in \mathcal{L}^1$ and $\int f\_j \to \int f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

**Monotonicity.** Say $f\_j$ increases. The integrals increase and are bounded, so they converge, and we can pick a subsequence $g\_i = f\_{n\_i}$ with $\int \lvert g\_i - g\_{i-1} \rvert = \int (g\_i - g\_{i-1}) < 2^{-i}$. The series $g\_1, g\_2 - g\_1, \dots$ is absolutely summable, so by the previous theorem it converges a.e. to an integrable $f$ with $\int f = \lim \int f\_j$. By monotonicity the whole sequence converges wherever the subsequence does, and $\int \lvert f - f\_j \rvert = \int f - \int f\_j \to 0$.

**Fatou.** For fixed $k$, $\min\_{k \le p \le k+n} f\_p$ decreases in $n$ with integrals bounded below by $0$, so by monotonicity $g\_k = \inf\_{p \ge k} f\_p$ is integrable with $\int g\_k \le \inf\_{p \ge k}\int f\_p$. Then $g\_k$ increases in $k$ with $\int g\_k \le \liminf\_j \int f\_j$ bounded, so monotonicity again gives $\int \liminf\_j f\_j = \lim\_k \int g\_k \le \liminf\_j \int f\_j$.

**Dominated convergence.** Apply Fatou to $h - f\_j \ge 0$ and $h + f\_j \ge 0$ (real parts first), exactly as in the measure-theoretic proof: $\limsup \int f\_j \le \int f \le \liminf \int f\_j$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bounded Integrals Replace Positivity)</span></p>

In the measure-theoretic approach, monotone convergence needs $f\_j \ge 0$ but allows $\int f = \infty$. Here there is no notion of a nonnegative function with "infinite integral", so the hypothesis is instead that the integrals are **bounded**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue Measure from the Integral)</span></p>

$f : \mathbb{R} \to \mathbb{C}$ is **locally integrable** if $f\chi\_{[-N, N]} \in \mathcal{L}^1(\mathbb{R})$ for every $N$ (e.g. every continuous function). A set $A \subset \mathbb{R}$ is **measurable** if $\chi\_A$ is locally integrable, and then

$$\mu(A) = \int \chi_A \quad\text{if } \chi_A \in \mathcal{L}^1(\mathbb{R}), \qquad \mu(A) = \infty \text{ otherwise}.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The Measurable Sets Form a $\sigma$-Algebra)</span></p>

The measurable sets contain $\varnothing$ and $\mathbb{R}$, are closed under complements, countable unions and countable intersections, and $\mu$ is countably additive on disjoint families (with values in $[0, \infty]$).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

$\chi\_{\mathbb{R} \setminus A} = 1 - \chi\_A$, and $1$ is locally integrable. For unions, $B\_n = \bigcup\_{k \le n} A\_k$ has $\chi\_{B\_n} = \max\_{k \le n} \chi\_{A\_k}$, and $\chi\_{[-N,N]}\chi\_{B\_n} \uparrow \chi\_{[-N,N]}\chi\_{B}$ with integrals bounded by $2N$, so $\chi\_B$ is locally integrable by monotonicity. Intersections follow by complements. For disjoint $A\_n$, $\chi\_{[-N,N]}\sum\_{n \le m}\chi\_{A\_n}$ increases to $\chi\_{[-N,N]}\chi\_{\bigcup A\_n}$, so $\mu(\bigcup\_n A\_n \cap [-N, N]) = \sum\_n \mu(A\_n \cap [-N, N])$; let $N \to \infty$ (both sides increase). $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(An Unbounded Integrable Function from a Cantor Construction)</span></p>

Remove the middle third $[\frac13, \frac23)$ from $[0, 1)$ to get $C\_1$, remove the middle thirds of the two remaining intervals to get $C\_2$, and so on; $C\_k$ is a union of $2^k$ half-open intervals of total length $(2/3)^k$. Let $f\_k = \chi\_{C\_k}$.

* $\sum\_k \int \lvert f\_k \rvert = \sum\_{k \ge 1} (2/3)^k = 2$: the series is absolutely summable.
* Since $C\_k \supset C\_{k+1}$, $\sum\_k f\_k(x)$ diverges exactly on $E = \bigcap\_k C\_k$ — so this Cantor-type set has **measure zero** by definition.
* The function $f = \sum\_k f\_k$ (set $f = 0$ on $E$) equals $k$ on $C\_k \setminus C\_{k+1}$, so it is integrable with $\int f = 2$ but **unbounded**, hence not Riemann integrable.
* The indicator $g$ of the union of all removed intervals is integrable with $\int g = \sum\_{k \ge 1} \frac13 (\frac23)^{k-1} = 1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Continuous Functions Are Integrable; Riemann = Lebesgue)</span></p>

Let $f \in C([0, 1])$ be real. By uniform continuity choose $N\_n$ (nondecreasing) such that $f$ varies by at most $2^{-n}$ on intervals of length $2^{-N\_n}$, and let $F\_n$ be the step function equal to $\min f$ on each of the $2^{N\_n}$ dyadic subintervals of $[0, 1)$. Then $\lvert f - F\_n \rvert \le 2^{-n}$ on $[0, 1)$, and the telescoping series $F\_1, F\_2 - F\_1, F\_3 - F\_2, \dots$ satisfies $\lvert F\_{n+1} - F\_n \rvert \le 2^{-n}$, so it is absolutely summable and converges (everywhere on $[0,1)$) to $f$. Hence $f\chi\_{[0,1)} \in \mathcal{L}^1$, and $\int f = \lim\_n \int F\_n$ is squeezed between the lower and upper Riemann sums, so it equals the Riemann integral.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Both Constructions Agree; Higher Dimensions)</span></p>

* The two approaches produce the same space $L^1(\mathbb{R})$, the same null sets and the same integral: in both, step functions are dense in $L^1$ and the integral is continuous in the $L^1$ norm, so both are "the" completion of the step functions.
* Replacing intervals by rectangles $[a\_1, b\_1) \times [a\_2, b\_2)$, the covering lemma (proved by subdividing rectangles along all endpoints, and by Heine–Borel for countable covers) and therefore the whole construction extend verbatim to $\mathbb{R}^2$ and $\mathbb{R}^n$.

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

* If $f : E \to \mathbb{C}$ is measurable, then $\lvert f(x) \rvert \le \lVert f \rVert_{L^\infty(E)}$ almost everywhere on $E$.
* If $E = [a, b]$ is a closed interval and $f \in C([a, b])$, then $\lVert f \rVert_{L^\infty([a,b])} = \lVert f \rVert_\infty$ is the usual sup norm on bounded continuous functions.

</div>

### Hölder's and Minkowski's Inequalities

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 130</span><span class="math-callout__name">(Hölder's Inequality for $L^p$ Spaces)</span></p>

If $1 \le p \le \infty$ and $\frac{1}{p} + \frac{1}{q} = 1$, and $f, g : E \to \mathbb{C}$ are measurable functions, then

$$\int_E \lvert fg \rvert \le \lVert f \rVert_{L^p(E)} \lVert g \rVert_{L^q(E)}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 131</span><span class="math-callout__name">(Minkowski's Inequality for $L^p$ Spaces)</span></p>

If $1 \le p \le \infty$ and $f, g : E \to \mathbb{C}$ are two measurable functions, then 

$$\lVert f + g \rVert_{L^p(E)} \le \lVert f \rVert_{L^p(E)} + \lVert g \rVert_{L^p(E)}.$$

</div>

A similar result also holds for $L^\infty(E)$. We use the shorthand $\lVert \cdot \rVert_p$ for $\lVert \cdot \rVert_{L^p(E)}$ from now on.

### The $L^p$ Space

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 133</span><span class="math-callout__name">($L^p$ Space)</span></p>

For any $1 \le p \le \infty$, we define the **$L^p$ space**

$$L^p(E) = \lbrace f : E \to \mathbb{C} : f \text{ measurable and } \lVert f \rVert_p < \infty \rbrace,$$

where we consider two elements $f, g$ of $L^p(E)$ to be equivalent (the same) if $f = g$ almost everywhere.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span>($L^p$ Space as a space of equivalence classes)</p>

We need the equivalence relation to make the $L^p$ norms actually norms: the space is really a space of equivalence classes 

$$[f] = \lbrace g : E \to \mathbb{C} : \lVert g \rVert_p < \infty \text{ and } g = f \text{ a.e.} \rbrace$$

rather than functions. But we still refer to elements as functions (as is custom in mathematics).

</div>

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

### Multiplication Operators on $L^p$ Spaces

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Multiplication Operators on $L^p$)</span></p>

Let $a<b$, $1\le p\le\infty$, and let $g$ be measurable on $[a,b]$. Define

$$Tf = M_gf := gf.$$

Then

$$Tf = M_g\in\mathcal B(L^p([a,b])) \quad\Longleftrightarrow\quad g\in L^\infty([a,b]).$$

In this case,

$$\|T\|_{L^p([a,b])} = \|M_g\|_{L^p([a,b])} = \|g\|_{L^\infty([a,b])}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. $\lVert T \rVert_{L^p([a,b])} = \lVert g\rVert_{L^\infty([a,b])}$.
   1. **Upper bound**

      1. $p \lt \infty$.
      We know that 

      $$\lVert Tf \rVert_{L^p([a,b])} = (\int_a^b \lvert f(x)g(x)\rvert^p dx)^{1/p} \leq (\int_a^b \lvert f(x)\lVert g\rVert_{L^\infty([a,b])} \rvert^p dx)^{1/p} = \lVert g\rVert_{L^\infty([a,b])}(\int_a^b \lvert f(x)\rvert^p dx)^{1/p} = \lVert g\rVert_{L^\infty([a,b])} \lVert f\rVert_{L^p([a,b])}.$$

      Thus, $\lVert Tf \rVert_{L^p([a,b])} \leq \lVert g\rVert_{L^\infty([a,b])} \lVert f\rVert_{L^p([a,b])}$.

      For $\lVert T \rVert_{L^p([a,b])}$ we use $\lVert f\rVert_{L^p([a,b])} = 1$, thus 

      $$\lVert T \rVert_{L^p([a,b])} \leq \lVert g\rVert_{L^\infty([a,b])}.$$
      2. $p = \infty$.

         For $p=\infty$, we use essential supremum directly:
        
         $$\|fg\|_\infty\le\|f\|_\infty\|g\|_\infty.$$
        
   2. **Lower bound**
      To obtain the lower bound on $T$, we construct a specific function $\hat{f}$ with a unit norm. This $\hat{f}$ takes a constant value whenever $g$ attains value $\gt \lVert g \rVert_{L^\infty}-\varepsilon$ for some fixed $\varepsilon$, which could be any value greater than zero, and zero elsewhere. To have norm one, the constant value one 
      
      $$E_\varepsilon := \lbrace x\in [a,b]: \lvert g(x)\rvert \gt \lVert g\rVert_{L^\infty} - \varepsilon \rbrace$$
      
      is $m(E_\varepsilon)^{-1/p}$, where $m$ is a Lebesgue measure. This gives the upper bound:

      $$\lVert T \rVert_{L^p([a,b])} \geq \lVert T\hat{f} \rVert_{L^p([a,b])} \gt \lVert g\rVert_{L^\infty} - \varepsilon.$$

      Since $\varepsilon$ is arbitrarily small, the bound
      
      $$\lVert T \rVert_{L^p([a,b])} \gt \lVert g\rVert_{L^\infty} - \varepsilon$$
      
      yields $\lVert T \rVert_{L^p([a,b])} \geq \lVert g\rVert_{L^\infty}$.

      Combining two bounds we establish $\lVert T \rVert_{L^p([a,b])} = \lVert g\rVert_{L^\infty}$.
  
      For $p$ infinite, the above constructed function $\hat{f}$ still works, since the exponent becomes zero and the constant becomes exactly $1$ on the set $E_\varepsilon$, which satisfies the requirement on a unit norm for a test function.

2. $T \in \mathcal{B}(L^p([a,b]), L^p([a,b]))$. 
   1. $T$ is a bounded linear operator.
      1. $T$ is obviously linear.
      2. $T$ is bounded since $\lVert T \rVert_{L^p([a,b])} = \lVert g\rVert_{L^\infty([a,b])} \lt \infty$.
   2. $Tf \in L^p([a,b])$.
      1. $\forall f\in L^p([a,b]): \quad \lVert Tf \rVert_{L^p([a,b])} \leq \lVert g\rVert_{L^\infty([a,b])} \lVert f\rVert_{L^p([a,b])} \lt \infty$.

</details>
</div>

### Properties of $L^p$ Spaces

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Continuous Embedding of $L^p$ into $L^q$ on Finite Measure Spaces)</span></p>

Let $(X,\mu)$ be a measure space with $0<\mu(X)<\infty$, and let $1\le q\le p\le\infty$. Then

$$L^p(X)\subseteq L^q(X),$$

<!-- and the inclusion is continuous, with -->
<!-- $$\|f\|_{L^q(X)} \le \mu(X)^{1/q-1/p}\|f\|_{L^p(X)}.$$ -->
<!-- Here $1/\infty:=0$. The constant $\mu(X)^{1/q-1/p}$ is **optimal**, with equality attained by nonzero constant functions. -->
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The case $p=q$ is trivial. Suppose $q<p$.

By Hölder's inequality with conjugate exponents $\frac pq$ and $\frac{p}{p-q}$,

$$
\begin{aligned}
\|f\|_{L^q([a,b])}^q
&=\int_a^b |f(x)|^q\,dx\\
&=\int_a^b |f(x)|^q\cdot 1\,dx\\
&\le \||f|^q\|_{L^{p/q}}\|1\|_{L^{p/(p-q)}}\\
&=\|f\|_{L^p}^q(b-a)^{(p-q)/p}<\infty.
\end{aligned}
$$

The first factor is finite because $f\in L^p([a,b])$, and the second is finite because the interval has finite measure.
Consequently,

$$\boxed{\|f\|_{L^q}\le (b-a)^{1/q-1/p}\|f\|_{L^p}.}$$

Thus $L^p([a,b])\subseteq L^q([a,b])$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 136</span></p>

Let $E \subset \mathbb{R}$ be measurable.

$$f \in L^p(E) \quad \iff \quad \lim_{n \to \infty} \int_{[-n,n] \cap E} \lvert f \rvert^p < \infty.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>The sequence $\lbrace \chi_{[-n,n]} \lvert f \rvert^p \rbrace$ is a pointwise increasing sequence of measurable functions, and for all $x \in E$, $\lim_{n \to \infty} \chi_{[-n,n]}(x) \lvert f(x) \rvert^p = \lvert f(x) \rvert^p$. By the Monotone Convergence Theorem, $\int_E \lvert f \rvert^p = \lim_{n \to \infty} \int_{[-n,n] \cap E} \lvert f \rvert^p$, and thus the two quantities are finite for exactly the same set of $f$s. $\square$</p>
  </details>
</div>

### Approximation and Separability

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Density of Compactly Supported Continuous Functions in $L^p$)</span></p>

Let $1\le p<\infty$. Then the space $C_c(\mathbb R)$ of continuous functions with compact support is **dense** in $L^p(\mathbb R)$:

$$\overline{C_c(\mathbb R)}^{\,L^p(\mathbb R)} =L^p(\mathbb R).$$

Equivalently, for every $f\in L^p(\mathbb R)$ and every $\varepsilon>0$, there exists $g\in C_c(\mathbb R)$ such that

$$\|f-g\|_{L^p(\mathbb R)}<\varepsilon.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Part 1: $L^p([a,b])$ is separable</summary>

**Key idea:** Construct a countable dense family by successive approximations:

$$
L^p([a,b])
\xrightarrow{\text{density theorem}}
C([a,b])
\xrightarrow{\text{Weierstrass}}
\mathbb K[x]
\xrightarrow{\text{rational coefficients}}
\mathbb K_{\mathbb Q}[x],
$$

where $\mathbb K=\mathbb R$ or $\mathbb C$, and $\mathbb K_{\mathbb Q}=\mathbb Q$ or $\mathbb Q+i\mathbb Q$, respectively.

1. **Continuous functions are dense in $L^p([a,b])$.**

   By the approximation theorem stated at the beginning of the assignment, for every $f\in L^p([a,b])$ and $\varepsilon>0$, there exists $g\in C([a,b])$ such that 

   $$\|f-g\|_p<\varepsilon.$$

2. **Polynomials are dense in $C([a,b])$ in the $L^p$-norm.**

   By the Weierstrass Approximation Theorem, for any $g\in C([a,b])$ and $\delta>0$, there exists a polynomial $P$ such that

   $$\|g-P\|_\infty<\delta.$$

   Since $[a,b]$ has finite measure,

   $$\|g-P\|_p \le(b-a)^{1/p}\|g-P\|_\infty.$$

   Thus uniform approximation implies $L^p$-approximation.

3. **Polynomials with rational coefficients are dense among all polynomials.**

   Let

   $$P(x)=\sum_{k=0}^m c_kx^k.$$

   Because $\mathbb K_{\mathbb Q}$ is dense in $\mathbb K$, each coefficient $c_k$ can be approximated arbitrarily closely by some $r_k\in\mathbb K_{\mathbb Q}$. Writing $Q(x)=\sum_{k=0}^m r_kx^k$, we have

   $$\|P-Q\|_\infty \le \sum_{k=0}^m |c_k-r_k| \max(1,|a|,|b|)^k.$$

   Since this is a finite sum, it can be made arbitrarily small.

4. **Countability.**
 
   The family

   $$D=\mathbb K_{\mathbb Q}[x] =\bigcup_{m=0}^{\infty} \left\lbrace\sum_{k=0}^m r_kx^k:r_k\in\mathbb K_{\mathbb Q}\right\rbrace$$

   is countable, because it is a countable union of countable sets. Combining steps 1–3 with the triangle inequality shows that $D$ is dense in $L^p([a,b])$. Therefore,

$$\boxed{L^p([a,b])\text{ is separable}.}$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Part 2: Continuous compactly supported functions are dense in $L^p(\mathbb R)$</summary>

Key idea: First truncate the domain, controlling the $L^p$-mass in the tails. Then approximate on the bounded interval and extend by zero.

Fix $f\in L^p(\mathbb R)$ and $\varepsilon>0$.

1. By the lemma, choose $R>0$ sufficiently large that

   $$\|f\|_{L^p(\mathbb R\setminus[-R,R])}<\frac{\varepsilon}{2}.$$

2. By the approximation theorem at the beginning of the assignment, there exists $h\in C([-R,R])$ with

   $$h(-R)=h(R)=0, \qquad \|f-h\|_{L^p([-R,R])}<\frac{\varepsilon}{2}.$$

3. Extend $h$ by zero to $\mathbb R$:

   $$
   g(x)=
   \begin{cases}
   h(x),&x\in[-R,R],\\
   0,&x\notin[-R,R].
   \end{cases}
   $$

   The endpoint conditions ensure $g\in C(\mathbb R)$, and its support is compact. Now,

   $$
   \begin{aligned}
   \|f-g\|_{L^p(\mathbb R)}
   &\le
   \|f\|_{L^p(\mathbb R\setminus[-R,R])}
   +\|f-h\|_{L^p([-R,R])}\\
   &<\frac{\varepsilon}{2}+\frac{\varepsilon}{2}
   =\varepsilon.
   \end{aligned}
   $$

Thus continuous compactly supported functions are dense in $L^p(\mathbb R)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why inequality in error decomposition and not equality)</span></p>

The inequality follows because the $p$-th power of the $L^p$-norm is additive over disjoint measurable sets, whereas the norm itself is generally not additive when $p>1$.

</div>

</details>
</div>

In the proof above we used the following lemma:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Vanishing $L^p$-tails)</span></p>

Let $f\in L^p(\mathbb R)$, and define $f_n=f\mathbf 1_{[-n,n]}$. Then

$$\|f-f_n\|_p\longrightarrow 0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since

$$|f|^p\mathbf 1_{[-n,n]}\uparrow |f|^p$$

almost everywhere, the Monotone Convergence Theorem gives

$$\lim_{n\to\infty}\int_{-n}^n|f(x)|^p\,dx =\int_{\mathbb R}|f(x)|^p\,dx<\infty.$$

Consequently,

$$
\begin{aligned}
\|f-f_n\|_p^p
&=\int_{\mathbb R\setminus[-n,n]}|f(x)|^p\,dx\\
&=\|f\|_p^p-\int_{-n}^n|f(x)|^p\,dx\\
&\longrightarrow0.
\end{aligned}
$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Separability of $L^p$ Spaces)</span></p>

Let $E\subseteq\mathbb R$ be Lebesgue-measurable and $1\le p<\infty$. Then

$$L^p(E)\text{ is separable}.$$

Equivalently, there exists a **countable dense subset** $D\subseteq L^p(E)$ such that

$$\forall f\in L^p(E),\ \forall\varepsilon>0,\ \exists g\in D: \quad \|f-g\|_{L^p(E)}<\varepsilon.$$

In particular, both $L^p([a,b])$ and $L^p(\mathbb R)$ are separable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

(d) \(L^p(\mathbb R)\) is separable
Key idea: Construct a countable dense set by combining countable dense subsets of \(L^p([-n,n])\).
1. Construct a countable candidate.
By (b), each \(L^p([-n,n])\) is separable. Therefore, for every \(n\in\mathbb N\), choose a countable dense subset
\[
D_n\subset L^p([-n,n]).
\]

Identify its elements with their zero extensions to \(\mathbb R\), and define
\[
D:=\bigcup_{n=1}^{\infty}D_n.
\]

Since a countable union of countable sets is countable, \(D\) is countable.
2. Prove density.
Let \(f\in L^p(\mathbb R)\) and \(\varepsilon>0\).
By the lemma from (c), choose \(n\) sufficiently large that
\[
\|f-f\mathbf 1_{[-n,n]}\|_p<\frac{\varepsilon}{2}.
\]

Since \(D_n\) is dense in \(L^p([-n,n])\), there exists \(h\in D_n\) such that
\[
\|f\mathbf 1_{[-n,n]}-h\|_p<\frac{\varepsilon}{2}.
\]

By the triangle inequality,
\[
\begin{aligned}
\|f-h\|_p
&\le
\|f-f\mathbf 1_{[-n,n]}\|_p
+\|f\mathbf 1_{[-n,n]}-h\|_p\\
&<\varepsilon.
\end{aligned}
\]

Thus \(D\) is dense in \(L^p(\mathbb R)\).
Since \(D\) is countable,
\[
\boxed{L^p(\mathbb R)\text{ is separable}.}
\]

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Important proof techniques: (local) dense approximation and tail control)</span></p>

The two reusable proof patterns here:
1. Successive approximation through dense classes,
2. Local approximation plus tail control.
   
They are worth preserving beyond the specific separability result.

</div>


### Mollifiers and Smooth Approximation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mollifiers)</span></p>

A **mollifier** is a function $\varphi\in C^\infty(\mathbb R)$ satisfying:

1. **Symmetry and nonnegativity:**
   
   $$\varphi(-x)=\varphi(x),\qquad \varphi(x)\ge 0.$$

2. **Compact support:**
   
   $$\operatorname{supp}(\varphi)\subseteq[-1,1].$$

3. **Normalization:**
   
   $$\int_{\mathbb R}\varphi(x)\,dx=1.$$

For $\varepsilon>0$, define the **rescaled mollifier**

$$\varphi_\varepsilon(x) :=\frac{1}{\varepsilon}\varphi\left(\frac{x}{\varepsilon}\right).$$

The family $\lbrace \varphi_\varepsilon\rbrace_{\varepsilon>0}$ is called a **family of mollifiers**.

In particular,

$$\operatorname{supp}(\varphi_\varepsilon) \subseteq[-\varepsilon,\varepsilon], \qquad \int_{\mathbb R}\varphi_\varepsilon(x)\,dx=1.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How do mollifiers extend the discussion on approximations)</span></p>

**Mollifiers** naturally extend the discussion of density and approximation, allowing us to approximate functions by smooth functions rather than merely continuous ones. The logical progression becomes

$$\underbrace{C_c(\mathbb R)\text{ dense in }L^p}_{\text{continuous approximation}} \quad\longrightarrow\quad \underbrace{\varphi_\varepsilon*f}_{\text{smooth approximation}}.$$

</div>

### Convergence in $L^p$ and Almost Everywhere Convergence

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Riesz Subsequence Theorem — $L^1$ Version)</span></p>

Let $E \subset \mathbb{R}$ be measurable, and let $h_n \in L^1(E)$, $n \in \mathbb{N}$, such that

$$\int_E |h_n| \to 0$$

as $n \to \infty$. Then there exists a subsequence $\lbrace h_{n_j}\rbrace_j$ such that for almost every $x \in E$,

$$\lim_{j\to\infty} h_{n_j}(x) = 0.$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>TODO:</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relation to Markov/Chebyshev Inequality)</span></p>

The assumption

$$\int_E |h_n| \to 0$$

means precisely that $h_n \to 0$ in $L^1(E)$. Markov's inequality applied to the nonnegative function $\lvert h_n\rvert$ gives, for every $\varepsilon>0$,

$$m\big(\lbrace x\in E:|h_n(x)|>\varepsilon\rbrace\big)\le\frac{1}{\varepsilon}\int_E |h_n|.$$

Hence

$$m\big(\lbrace |h_n|>\varepsilon\rbrace\big)\to 0,$$

so $h_n\to 0$ **in measure**.

Thus Markov's inequality provides the bridge

$$L^1\text{-convergence}\quad\Longrightarrow\quad\text{convergence in measure}.$$

The Riesz subsequence theorem then says that from convergence in measure one can extract a subsequence converging almost everywhere:

$$h_n\to 0\text{ in measure}\quad\Longrightarrow\quad\exists\,h_{n_j}\to 0\text{ a.e.}$$

The inequality above is often also referred to as **Chebyshev's inequality** (or the Markov–Chebyshev inequality); Markov's inequality is the more general name for this form.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span>(Approximation Theorem: Density of Step and Compactly Supported Continuous Functions in $L^p$)</p>

Let $a<b$, $1\le p<\infty$, and $f\in L^p([a,b])$. For all $\epsilon>0$ there exist a step function $\psi\in L^p([a,b])$ and $g\in C([a,b])$ with $g(a)=g(b)=0$ such that

$$\|f-\psi\|_p+\|f-g\|_p<\epsilon.$$

</div>

### Riesz–Fischer Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 139</span><span class="math-callout__name">(Riesz–Fischer)</span></p>

For all $1 \le p \le \infty$, 

$$L^p(E) \quad\text{is a Banach space.}$$

</div>

<div class="accordion">
  <details>
    <summary>Proof for finite $p$</summary>
    <p>Recall that a normed space is Banach if and only if every absolutely summable series is summable. Suppose $\lbrace f_k \rbrace$ is a sequence in $L^p(E)$ with $\sum_k \lVert f_k \rVert_p = M < \infty$.</p>
    <p>Define $g_n(x) = \sum_{k=1}^{n} \lvert f_k(x) \rvert$. By the triangle inequality, $\lVert g_n \rVert_p \le \sum_{k=1}^{n} \lVert f_k \rVert_p \le M$. By Fatou's lemma, $\int_E \left( \sum_{k=1}^{\infty} \lvert f_k \rvert \right)^p \le M^p < \infty$, so $\sum_k \lvert f_k(x) \rvert$ is finite a.e.</p>
    <p>Define $f(x) = \sum_k f_k(x)$ where the sum converges absolutely, and $f(x) = 0$ otherwise. Then $\lvert \sum_{k=1}^{n} f_k(x) - f(x) \rvert^p \to 0$ a.e. and $\lvert \sum_{k=1}^{n} f_k - f \rvert^p \le \lvert g \rvert^p$ a.e. where $g = \sum_k \lvert f_k \rvert$ satisfies $\lVert g \rVert_p \le M$ and $\int_E \lvert g \rvert^p < \infty$. By the Dominated Convergence Theorem, $\lim_{n \to \infty} \int_E \lvert \sum_{k=1}^{n} f_k - f \rvert^p = 0$, so $L^p$ is indeed a Banach space. $\square$</p>
  </details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof for $p=\infty$</summary>

TODO: assignment 7, exercise 2(a).

</details>
</div>

Since $C([a, b])$ is dense in $L^p([a, b])$ and the latter is a Banach space, we can think of the $L^p$ space as a **completion** of the continuous functions.

## Hilbert Spaces

From here, we move on to more general topics in functional analysis. Our next topic will be **Hilbert spaces**, which give us the important notions of an inner product, orthogonality, and so on.

### Pre-Hilbert Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 140</span><span class="math-callout__name">(Pre-Hilbert Space)</span></p>

A **pre-Hilbert space** $H$ is a vector space over $\mathbb{C}$ with a **Hermitian inner product**, which is a map $\langle \cdot, \cdot \rangle : H \times H \to \mathbb{C}$ satisfying the following properties:

1. For all $\lambda_1, \lambda_2 \in \mathbb{C}$ and $v_1, v_2, w \in H$, we have 
   
   $$\langle \lambda_1 v_1 + \lambda_2 v_2, w \rangle = \lambda_1 \langle v_1, w \rangle + \lambda_2 \langle v_2, w \rangle$$

2. For all $v, w \in H$, we have 
   
   $$\langle v, w \rangle = \overline{\langle w, v \rangle}$$

3. For all $v \in H$, we have
   
   $$\langle v, v \rangle \ge 0 \iff v = 0$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Pre-Hilbert Space)</span></p>

We should think of pre-Hilbert spaces as **normed vector spaces where the norm comes from an inner product**. The inner product is linear in the first variable but conjugate-linear in the second: $\langle v, \lambda w \rangle = \overline{\lambda} \langle v, w \rangle$.

</div>

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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Finite-Dimensional Pre-Hilbert Spaces Are $\mathbb{C}^n$)</span></p>

Let $H$ be a pre-Hilbert space of finite dimension $n$. Then $H$ has an orthonormal basis $\lbrace e\_1, \dots, e\_n \rbrace$, every $v \in H$ satisfies $v = \sum\_i \langle v, e\_i \rangle e\_i$, and

$$T : H \to \mathbb{C}^n, \qquad Tv = \big(\langle v, e_1 \rangle, \dots, \langle v, e_n \rangle\big)$$

is a linear isomorphism with $\langle u, v \rangle = \sum\_i (Tu)\_i \overline{(Tv)\_i}$ and $\lVert u \rVert\_H = \lVert Tu \rVert\_{\mathbb{C}^n}$. In particular, **every finite-dimensional pre-Hilbert space is complete**, i.e. a Hilbert space.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

**Gram–Schmidt.** Start from a basis $v\_1, \dots, v\_n$. Put $e\_1 = v\_1/\lVert v\_1 \rVert$ ($v\_1 \ne 0$ by independence) and, inductively,

$$w_{k+1} = v_{k+1} - \sum_{i=1}^{k} \langle v_{k+1}, e_i \rangle e_i, \qquad e_{k+1} = \frac{w_{k+1}}{\lVert w_{k+1} \rVert}.$$

Taking inner products shows $w\_{k+1} \perp e\_1, \dots, e\_k$, and $w\_{k+1} \ne 0$ because $v\_{k+1} \notin \operatorname{Span}\lbrace v\_1, \dots, v\_k \rbrace = \operatorname{Span}\lbrace e\_1, \dots, e\_k \rbrace$.

**Expansion.** $U = v - \sum\_i \langle v, e\_i \rangle e\_i$ is orthogonal to every $e\_j$; writing $U = \sum\_i d\_i e\_i$ gives $d\_j = \langle U, e\_j \rangle = 0$, so $U = 0$.

**Isometric isomorphism.** $T$ is linear, injective by the expansion, and surjective since $u = \sum\_i c\_i e\_i$ has $Tu = (c\_i)$. Then $\langle u, v \rangle = \langle u, \sum\_i \langle v, e\_i \rangle e\_i \rangle = \sum\_i \langle u, e\_i \rangle \overline{\langle v, e\_i \rangle}$; set $u = v$ for the norms.

**Completeness.** $T$ is an isometry onto the complete space $\mathbb{C}^n$, so it maps Cauchy sequences to Cauchy (hence convergent) sequences, and $T^{-1}$ maps convergent sequences back to convergent sequences. $\square$

</details>
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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Polarization Identity)</span></p>

Let $H$ be a complex pre-Hilbert space, with inner product linear in the first argument. Then, for all $u,v\in H$,

$$\langle u,v\rangle =\frac14\left[ \|u+v\|^2-\|u-v\|^2 +i\|u+iv\|^2-i\|u-iv\|^2 \right].$$

In particular, **the inner product is uniquely determined by its induced norm**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jordan–von Neumann Theorem: Characterization of Inner-Product Norms)</span></p>

Let $H$ be a complex normed vector space. The following are equivalent:

1. The norm on $H$ is induced by a Hermitian inner product.
2. The norm satisfies the **parallelogram identity**: for all $u,v\in H$,

   $$\|u+v\|^2+\|u-v\|^2 =2\left(\|u\|^2+\|v\|^2\right).$$

Moreover, when these conditions hold, the inducing inner product is **unique** and is given by the polarization identity:

$$\langle u,v\rangle =\frac14\left[ \|u+v\|^2-\|u-v\|^2 +i\|u+iv\|^2-i\|u-iv\|^2 \right].$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

$(1 \Rightarrow 2)$ Expand $\lVert u \pm v \rVert^2 = \lVert u \rVert^2 \pm 2\operatorname{Re}\langle u, v \rangle + \lVert v \rVert^2$ and add. Expanding all four terms of the polarization identity in the same way shows that any inducing inner product must be given by it, which proves uniqueness.

$(2 \Rightarrow 1)$ Define $\langle u, v \rangle$ by the polarization identity. Write

$$R(u, v) = \tfrac14\big(\lVert u + v \rVert^2 - \lVert u - v \rVert^2\big), \qquad \text{so} \qquad \langle u, v \rangle = R(u, v) + iR(u, iv).$$

* **Norm.** $\langle u, u \rangle = \frac14\big(4\lVert u \rVert^2 + i\lvert 1+i \rvert^2\lVert u \rVert^2 - i\lvert 1-i \rvert^2\lVert u \rVert^2\big) = \lVert u \rVert^2$. This needs no parallelogram law; positivity and definiteness follow.
* **Additivity of $R$ in the first slot.** Apply the parallelogram law to $a = u + v$, $b = u^{\prime} + v$, and to $a = u - v$, $b = u^{\prime} - v$:

  $$\lVert u+v \rVert^2 + \lVert u^{\prime}+v \rVert^2 = \tfrac12\big(\lVert u+u^{\prime}+2v \rVert^2 + \lVert u-u^{\prime} \rVert^2\big), \qquad \lVert u-v \rVert^2 + \lVert u^{\prime}-v \rVert^2 = \tfrac12\big(\lVert u+u^{\prime}-2v \rVert^2 + \lVert u-u^{\prime} \rVert^2\big).$$

  Subtracting gives $R(u, v) + R(u^{\prime}, v) = \frac12 R(u + u^{\prime}, 2v)$. Taking $u^{\prime} = 0$ (note $R(0, v) = 0$) gives $R(u, v) = \frac12 R(u, 2v)$, hence $R(u + u^{\prime}, v) = R(u, v) + R(u^{\prime}, v)$.
* **Real homogeneity.** Additivity gives $R(ku, v) = kR(u, v)$ for $k \in \mathbb{N}$, then for $k \in \mathbb{Z}$ (as $R(-u, v) = -R(u, v)$), then for $r \in \mathbb{Q}$ (write $u = n u^{\prime}$). Since $R$ is continuous in $u$ (the norm is continuous), $R(tu, v) = tR(u, v)$ for all $t \in \mathbb{R}$.
* **Complex homogeneity.** From the definition, $R(iu, v) = \frac14(\lVert u - iv \rVert^2 - \lVert u + iv \rVert^2) = -R(u, iv)$ and $R(iu, iv) = R(u, v)$, so $\langle iu, v \rangle = -R(u, iv) + iR(u, v) = i\langle u, v \rangle$. Together with additivity and real homogeneity, $\langle \cdot, v \rangle$ is complex linear.
* **Hermitian symmetry.** $R(v, u) = R(u, v)$ and $R(v, iu) = -R(u, iv)$, so $\langle v, u \rangle = \overline{\langle u, v \rangle}$. $\square$

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 360 240" xmlns="http://www.w3.org/2000/svg" width="380" aria-label="Parallelogram law: sum of squared diagonals equals twice sum of squared sides">
    <g stroke="#444" stroke-width="0.8" fill="none">
      <line x1="20"  y1="200" x2="340" y2="200" />
      <line x1="60"  y1="20"  x2="60"  y2="220" />
    </g>
    <polygon points="60,200 230,200 290,90 120,90" fill="rgba(44,73,148,0.08)" stroke="#2c4994" stroke-width="1.5" />
    <line x1="60"  y1="200" x2="120" y2="90"  stroke="#2c4994" stroke-width="2.2" />
    <line x1="60"  y1="200" x2="230" y2="200" stroke="#3d7a26" stroke-width="2.2" />
    <line x1="60"  y1="200" x2="290" y2="90"  stroke="#a23ec3" stroke-width="2.2" stroke-dasharray="5,3" />
    <line x1="120" y1="90"  x2="230" y2="200" stroke="#d65336" stroke-width="2.2" stroke-dasharray="5,3" />
    <g fill="#222">
      <circle cx="60"  cy="200" r="3" />
      <circle cx="120" cy="90"  r="3" />
      <circle cx="230" cy="200" r="3" />
      <circle cx="290" cy="90"  r="3" />
    </g>
    <text x="46"  y="216" font-size="12" fill="#222">0</text>
    <text x="108" y="80"  font-size="13" font-weight="600" fill="#2c4994">u</text>
    <text x="234" y="216" font-size="13" font-weight="600" fill="#3d7a26">v</text>
    <text x="295" y="84"  font-size="13" font-weight="600" fill="#a23ec3">u + v</text>
    <text x="160" y="142" font-size="13" font-weight="600" fill="#d65336">u − v</text>
    <text x="20"  y="40" font-size="12" fill="#444" font-style="italic">‖u+v‖² + ‖u−v‖² = 2(‖u‖² + ‖v‖²)</text>
  </svg>
  <figcaption>The two diagonals (purple, orange) of the parallelogram spanned by $u$ and $v$ are $u+v$ and $u-v$. Their squared lengths sum to twice the sum of squared sides — a Euclidean fact that, remarkably, characterizes inner-product spaces among all normed spaces. Whenever this identity fails, no inner product can induce the norm.</figcaption>
</figure>

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

The functions $f_n(x) = \frac{1}{\sqrt{2\pi}} e^{inx}$ (as elements of $L^2([-\pi, \pi])$) form an orthonormal subset of $L^2([-\pi, \pi])$. This is because 

$$\int_{-\pi}^{\pi} e^{imx} \overline{e^{inx}}\,dx = \int_{-\pi}^{\pi} e^{i(m-n)x}\,dx$$

is zero unless $m = n$.

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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Closed Span of an Orthonormal Set and Orthogonal Projection)</span></p>

Let $H$ be a Hilbert space, let $\lbrace e_n\rbrace_{n=1}^\infty$ be a countable orthonormal subset of $H$, and let $W=\operatorname{span}\lbrace e_n\rbrace_{n=1}^\infty$ be the subspace of all **finite** linear combinations of these vectors. Then

$$\overline W = \left\lbrace \sum_{n=1}^{\infty} c_n e_n: (c_n)_{n=1}^{\infty}\in\ell^2 \right\rbrace.$$

Moreover, for every $u\in H$,

$$P_{\overline W}u= \sum_{n=1}^{\infty} \langle u,e_n\rangle e_n$$

is the orthogonal projection of $u$ onto $\overline W$. Equivalently, for every $w\in\overline W$,

$$\left\| u- \sum_{n=1}^{\infty} \langle u,e_n\rangle e_n \right\| \le \|u-w\|,$$

with equality if and only if $w= \sum_{n=1}^{\infty} \langle u,e_n\rangle e_n$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the first statement (using Pythagoras)</summary>

Let

$$W=\operatorname{span}\lbrace e_n:n\ge1\rbrace,$$

where $(e_n)$ is an orthonormal subset of the Hilbert space $H$. We prove that

$$w\in\overline W \iff \exists (c_n)\in\ell^2 \text{ such that } w=\sum_{n=1}^\infty c_ne_n.$$

**Forward direction.** Suppose $w\in\overline W$. Define

$$c_n:=\langle w,e_n\rangle, \qquad s_N:=\sum_{n=1}^N c_ne_n.$$

By Bessel's inequality,

$$\sum_{n=1}^\infty |c_n|^2 = \sum_{n=1}^\infty |\langle w,e_n\rangle|^2 \le \|w\|^2,$$

so $(c_n)\in\ell^2$.

It remains to show $s_N\to w$. Let $\varepsilon>0$. Since $w\in\overline W$, choose

$$v=\sum_{n=1}^M a_ne_n\in W$$

such that

$$\|w-v\|<\varepsilon.$$

For $N\ge M$,

$$w-v=(w-s_N)+(s_N-v).$$

Moreover,

$$w-s_N\perp \operatorname{span}\lbrace e_1,\dots,e_N\rbrace,$$

while $s_N-v\in \operatorname{span}\lbrace e_1,\dots,e_N\rbrace$. Hence, by Pythagoras,

$$\|w-v\|^2 = \|w-s_N\|^2+\|s_N-v\|^2 \ge \|w-s_N\|^2.$$

Therefore

$$\|w-s_N\|\le \|w-v\|<\varepsilon$$

for every $N\ge M$. Thus $s_N\to w$, so

$$w=\sum_{n=1}^\infty c_ne_n.$$

**Reverse direction.** Suppose $(c_n)\in\ell^2$, and let

$$s_N:=\sum_{n=1}^N c_ne_n\in W.$$

For $N>M$, orthonormality gives

$$\|s_N-s_M\|^2=\left\|\sum_{n=M+1}^N c_n e_n\right\|^2=\sum_{n=M+1}^N |c_n|^2.$$

Since $(c_n)\in\ell^2$, the right-hand side tends to $0$, so $(s_N)$ is Cauchy. Because $H$ is complete, $s_N\to w$ for some $w\in H$.

Since every $s_N\in W$, the limit belongs to $\overline W$. Hence

$$w=\sum_{n=1}^\infty c_ne_n\in\overline W.$$

Thus the two conditions are equivalent.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Alternative (approximation by projection-contraction on a dense subspace) proof of the second statement</summary>

The key idea is that the finite-dimensional orthogonal projections $P_N$ are contractions:

$$\lVert P_Nx\rVert \leq \lVert x\rVert$$

**Step 1. Establish the contraction property.**

Define

$$P_Nx:=\sum\_{n=1}^N\langle x,e_n\rangle e_n.$$

By orthonormality,

$$\lVert P_Nx\rVert^2 =\langle x,P_Nx\rangle \leq \lVert x\rVert\lVert P_Nx\rVert.$$

Therefore,

$$\lVert P_Nx\rVert \leq\lVert x\rVert.$$

**Step 2. Approximate $w$ by elements of $W$.**

Let $w\in\overline W$. Given $\varepsilon>0$, choose

$$v=\sum\_{n=1}^M a_ne_n\in W$$

such that $\lVert w-v\rVert<\varepsilon/2$.

For every $N\geq M$, we have $P_Nv=v$. Consequently,

$$\begin{aligned} \lVert w-P_Nw\rVert &=\lVert (w-v)-P_N(w-v)\rVert \\\ &\leq\\|w-v\\|+\\|P_N(w-v)\\|\\\ &\leq2\\|w-v\\|\\\ &<\varepsilon. \end{aligned}$$

Hence

$$P_Nw\longrightarrow w.$$

Moreover, the contraction property implies

$$\sum\_{n=1}^N|\langle w,e_n\rangle|^2 =\|P_Nw\|^2\leq\|w\|^2,$$

so the coefficients belong to $\ell^2$.

**Step 3. The reverse implication.**

This is unchanged: if $(c_n)\in\ell^2$, orthonormality gives

$$\left\|\sum\_{n=M+1}^Nc_ne_n\right\|^2 =\sum\_{n=M+1}^N|c_n|^2\longrightarrow0$$

as $M,N\to\infty$. Completeness of $H$ then gives convergence to an element of $\overline W$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The value of the technique proof of the second proposition)</span></p>

The second proof reveals a useful general principle:

If $P_N$ are uniformly bounded linear operators and $P_Nv\to v$ on a dense subspace, then $P_Nx\to x$ everywhere.

Here $\lVert P_N\rVert \leq1$, and $P_Nv=v$ eventually for every $v\in W$.

Strictly speaking, the reverse implication still uses the norm identity for orthogonal sums, which is itself a form of Pythagoras. But the forward argument avoids the Pythagorean decomposition entirely.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the second statement</summary>

* We notice that $H = \overline{W} \oplus W^\perp \implies u = u_w + u_\perp$, where $u_w \in \overline{W}$ and $u_\perp \in W^\perp$.
* We know that for $u_w \in \overline{W}$ we have expansion $\sum_{n=1}^\infty c_n e_n$, with coefficients $c_n := \langle u_w, e_n\rangle$ and $W = span\lbrace e_n\rbrace_{n=1}^\infty$.
* Then for any $w\in \overline{W}$:
  
  $$\lVert u_\perp + u_w - w \rVert = \lVert u_\perp + \sum_{n=1}^\infty \langle u_w - w, e_n\rangle e_n \rVert$$

  * Since $u_\perp$ and $\langle u_w - w, e_n\rangle e_n \rVert$ live in orthogonal spaces, we can apply Pythagoras:
  
  $$
  \begin{aligned}
  \lVert u_\perp + \sum_{n=1}^\infty \langle u_w - w, e_n\rangle e_n \rVert 
  &= \sqrt{\lVert u_\perp \rVert^2 + \lVert \sum_{n=1}^\infty \langle u_w - w, e_n\rangle e_n \rVert^2} \\
  &\geq \lVert u_\perp \rVert = \lVert u_\perp + u_w - \sum_{n=1}^\infty \langle u_w, e_n\rangle e_n \rVert \\
  &= \lVert u - \sum_{n=1}^\infty \langle u, e_n\rangle e_n \rVert,
  \end{aligned}
  $$

  where we used that $u_w = \sum_{n=1}^\infty \langle u_w, e_n\rangle e_n \rVert$ and $\langle u_\perp, e_n\rangle = 0$ for all $e_n$.

The "if and only if" statement come trivially: 

$$\|u-w\|^2=\|u_\perp\|^2 \iff \|u_w-w\|^2=0 \iff w=u_w,$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The value of the propositon above)</span></p>

That proposition is valuable because it unifies three ideas:

$$\ell^2 \longleftrightarrow \overline{\operatorname{span}\lbrace e_n\rbrace} \longleftrightarrow \text{orthogonal projection}.$$

And it clarifies the distinction between a merely orthonormal set and a full orthonormal basis. If $\lbrace e_n\rbrace$ is a basis of all $H$, then

$$\overline W=H$$

and the projection becomes the identity:

$$u= \sum_{n=1}^{\infty} \langle u,e_n\rangle e_n.$$

That is exactly the special case already appearing in Fourier–Bessel theorem.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Orthogonality of the truncation error)</span></p>

At first sight it may seem surprising that

$$w-s_N \perp \operatorname{span}\lbrace e_1,\dots,e_N\rbrace,$$

because $w\in\overline W$ and one might think that $\overline W$ differs only negligibly from $W$. The point is that the orthogonality is only with respect to the **finite-dimensional truncation space**

$$W_N:=\operatorname{span}\lbrace e_1,\dots,e_N\rbrace,$$

not with respect to the whole space $W$.

The remainder $w-s_N$ may still contain components in the later directions

$$e_{N+1},e_{N+2},\dots,$$

all of which belong to $W$. Thus the error does not represent a component of $w$ lying “outside” $W$; it is simply the part not yet captured by the first $N$ basis directions.

Indeed, if $w-s_N$ were orthogonal to all of $W$, then by continuity it would also be orthogonal to $\overline W$. Since $w-s_N\in\overline W$, this would force $w-s_N=0$.

So the geometric picture is: as $N$ increases, the subspace $W_N$ grows, the error is orthogonal to an increasingly large collection of directions, and its norm is forced to tend to zero.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Best approximation property of Fourier coefficients)</span></p>

For

$$W_N:=\operatorname{span}\lbrace e_1,\dots,e_N\rbrace, \qquad s_N:=\sum_{n=1}^N \langle w,e_n\rangle e_n,$$

the vector $s_N$ is the unique best approximation to $w$ from $W_N$.

Indeed, for any $v\in W_N$,

$$w-v=(w-s_N)+(s_N-v),$$

and the two terms are orthogonal. Hence

$$\|w-v\|^2 = \|w-s_N\|^2+\|s_N-v\|^2 \ge \|w-s_N\|^2.$$

Therefore

$$\|w-s_N\| = \min_{v\in W_N}\|w-v\|.$$

So the Fourier coefficients are not merely convenient coordinates: they are exactly the coefficients that produce the orthogonal projection of $w$ onto $W_N$, hence the best possible approximation using the first $N$ directions.

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

The example $\lbrace \underline{e}\_n \rbrace$ of sequences from above is a maximal orthonormal subset of $\ell^2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

A **countably infinite maximal orthonormal subset** basically serves the same purpose as an **orthonormal basis** does in linear algebra, but not every element will be a **finite** linear combination of the orthonormal subset elements (like was possible with a Hamel basis).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span>(Hamel Basis)</p>

A **Hamel basis** of a vector space $V$ over a field $\mathbb F$ is a subset $B\subseteq V$ such that:

1. $B$ is **linearly independent**, and
2. every $v\in V$ can be written as a **finite** linear combination of elements of $B$.

Equivalently, for every $v\in V$, there exist unique $b_1,\dots,b_n\in B$ and scalars $a_1,\dots,a_n\in\mathbb F$ such that

$$v=\sum_{k=1}^n a_k b_k.$$

The word **finite** is crucial.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span>(Standard vectors is not Hamel Basis in infinite-dimensional Banach Spaces)</p>

For example, in an infinite-dimensional Banach space such as $\ell^2$, the standard vectors $e_1,e_2,\dots$ are not a Hamel basis, because a general element

$$x=(x_1,x_2,\dots)$$

requires an infinite series

$$x=\sum_{n=1}^\infty x_n e_n.$$

They form instead a **Schauder basis**.

So a Hamel basis is the purely algebraic notion of basis, with no topology or convergence involved.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span>(Schauder Basis)</p>

A **Schauder basis** of a normed space $X$ is a sequence $(e_n)\_{n\ge 1}\subset X$ such that every $x\in X$ can be written uniquely as a convergent infinite series

$$x=\sum_{n=1}^\infty a_n e_n,$$

where convergence is with respect to the norm of $X$. Equivalently,

$$\left\|x-\sum_{n=1}^N a_n e_n\right\|\to 0 \qquad\text{as }N\to\infty.$$

The coefficients $a_n$ are uniquely determined by $x$.

The key difference from a Hamel basis is:

* **Hamel basis:** finite linear combinations only.
* **Schauder basis:** infinite norm-convergent series are allowed.

For example, in $\ell^2$, the standard unit vectors

$$e_1=(1,0,0,\dots),\quad e_2=(0,1,0,\dots),\dots$$

form a Schauder basis, because every $x=(x_1,x_2,\dots)\in\ell^2$ satisfies

$$x=\sum_{n=1}^\infty x_n e_n$$

in the $\ell^2$-norm.

One subtle point: not every separable Banach space has a Schauder basis. Separability only guarantees the existence of a countable dense set, which is much weaker.


</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Separable Space)</span></p>

A topological space $X$ is called **separable** if it contains a **countable dense subset**.

That is, there exists a countable set

$$D=\lbrace x_1,x_2,\dots\rbrace\subseteq X$$

such that

$$\overline{D}=X.$$

For a metric space, this means that for every $x\in X$ and every $\varepsilon>0$, there exists some $d\in D$ such that

$$d(x,d)<\varepsilon.$$

So intuitively, a separable space may be uncountable, but it can be approximated arbitrarily well using only countably many points.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span>(Separable Space)</p>

* $\mathbb R$ is separable because $\mathbb Q$ is countable and dense in $\mathbb R$.
* $\ell^2$ is separable because sequences with finite support and rational coordinates form a countable dense subset.

</div>

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

<div class="math-callout math-callout--proposition" markdown="1">
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

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">Proposition 166: eⁱ⁽ⁿ⁻ᵐ⁾ˣ laps the unit circle</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Two unit circles in the complex plane. Left, for n not equal to m, the point e to the i k x runs full laps around the circle, marked by arrowheads, and its mean position is the red centre 0. Right, for n equal to m, the point sits at 1 and its mean is 1.">
        <text x="78" y="24" font-size="10" font-weight="600" fill="#444" text-anchor="middle">n ≠ m</text>
        <g stroke="#bbb" stroke-width="0.8"><line x1="20" y1="104" x2="136" y2="104" /><line x1="78" y1="46" x2="78" y2="162" /></g>
        <g font-size="7.5" fill="#888"><text x="137" y="107">Re</text><text x="81" y="44">Im</text></g>
        <text x="218" y="24" font-size="10" font-weight="600" fill="#444" text-anchor="middle">n = m</text>
        <g stroke="#bbb" stroke-width="0.8"><line x1="160" y1="104" x2="276" y2="104" /><line x1="218" y1="46" x2="218" y2="162" /></g>
        <g font-size="7.5" fill="#888"><text x="277" y="107">Re</text><text x="221" y="44">Im</text></g>
        <circle cx="78" cy="104" r="50" fill="none" stroke="#2c3e94" stroke-width="1.6" />
        <polygon points="69.3,54.8 75.6,50.1 76.8,57.0" fill="#2c3e94" />
        <polygon points="39.7,136.1 32.5,133.0 37.9,128.5" fill="#2c3e94" />
        <polygon points="125.0,121.1 125.9,128.9 119.3,126.5" fill="#2c3e94" />
        <g fill="#2c3e94"><circle cx="116.3" cy="71.9" r="3.5" /></g>
        <text x="121.3" y="67.9" font-size="9" fill="#2c3e94">eⁱᵏˣ</text>
        <g fill="#c62828"><circle cx="78" cy="104" r="3.5" /></g>
        <text x="84" y="117" font-size="8.5" fill="#c62828">mean 0</text>
        <circle cx="218" cy="104" r="50" fill="none" stroke="#bbb" stroke-width="1" stroke-dasharray="3,3" />
        <g fill="#c62828"><circle cx="268" cy="104" r="6.5" opacity="0.35" /></g>
        <g fill="#2c3e94"><circle cx="268" cy="104" r="3.5" /></g>
        <text x="260" y="93" font-size="9" fill="#2c3e94" text-anchor="end">e⁰ = 1</text>
        <text x="260" y="123" font-size="8.5" fill="#c62828" text-anchor="end">mean 1</text>
        <g font-size="8.5" fill="#444" text-anchor="middle">
          <text x="78" y="172">k = n − m ≠ 0: |k| full laps,</text>
          <text x="78" y="184">mean 0 ⇒ ∫ eⁱᵏˣ dx = 0</text>
          <text x="218" y="172">k = 0: the point never moves,</text>
          <text x="218" y="184">mean 1 ⇒ ∫ 1 dx = 2π</text>
        </g>
        <g font-size="9" fill="#444">
          <text x="12" y="203">⟨eⁱⁿˣ, eⁱᵐˣ⟩ = ∫ eⁱ⁽ⁿ⁻ᵐ⁾ˣ dx = 2π · (mean position)</text>
          <text x="12" y="215">so it is 2π when n = m and 0 otherwise</text>
        </g>
      </svg>
    </div>
    <div>
      <div class="panel-label">Real and imaginary parts: whole periods cancel</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Three strips over the interval from minus pi to pi: cos 3x and sin 3x with positive lobes shaded blue and negative lobes red that cancel to integral zero, and the constant one shaded fully with area two pi">
        <line x1="54.0" y1="46" x2="262.0" y2="46" stroke="#bbb" stroke-width="0.8" />
        <g fill="#1565c0" opacity="0.4"><path d="M 54.0,46 L 54.0,46.0 54.5,46.0 55.0,46.0 55.6,46.0 56.1,46.0 56.6,46.0 57.1,46.0 57.6,46.0 58.2,46.0 58.7,46.0 59.2,46.0 59.7,46.0 60.2,46.0 60.8,46.0 61.3,46.0 61.8,46.0 62.3,46.0 62.8,46.0 63.4,46.0 63.9,46.0 64.4,46.0 64.9,46.0 65.4,46.0 66.0,46.0 66.5,46.0 67.0,46.0 67.5,46.0 68.0,46.0 68.6,46.0 69.1,46.0 69.6,46.0 70.1,46.0 70.6,46.0 71.2,46.0 71.7,45.5 72.2,44.7 72.7,44.0 73.2,43.2 73.8,42.5 74.3,41.8 74.8,41.1 75.3,40.3 75.8,39.6 76.4,39.0 76.9,38.3 77.4,37.6 77.9,37.0 78.4,36.4 79.0,35.8 79.5,35.2 80.0,34.7 80.5,34.2 81.0,33.7 81.6,33.2 82.1,32.8 82.6,32.4 83.1,32.0 83.6,31.6 84.2,31.3 84.7,31.0 85.2,30.8 85.7,30.6 86.2,30.4 86.8,30.2 87.3,30.1 87.8,30.0 88.3,30.0 88.8,30.0 89.4,30.0 89.9,30.1 90.4,30.2 90.9,30.3 91.4,30.5 92.0,30.7 92.5,30.9 93.0,31.2 93.5,31.5 94.0,31.9 94.6,32.2 95.1,32.6 95.6,33.1 96.1,33.5 96.6,34.0 97.2,34.5 97.7,35.0 98.2,35.6 98.7,36.2 99.2,36.8 99.8,37.4 100.3,38.1 100.8,38.7 101.3,39.4 101.8,40.1 102.4,40.8 102.9,41.5 103.4,42.3 103.9,43.0 104.4,43.7 105.0,44.5 105.5,45.2 106.0,46.0 106.5,46.0 107.0,46.0 107.6,46.0 108.1,46.0 108.6,46.0 109.1,46.0 109.6,46.0 110.2,46.0 110.7,46.0 111.2,46.0 111.7,46.0 112.2,46.0 112.8,46.0 113.3,46.0 113.8,46.0 114.3,46.0 114.8,46.0 115.4,46.0 115.9,46.0 116.4,46.0 116.9,46.0 117.4,46.0 118.0,46.0 118.5,46.0 119.0,46.0 119.5,46.0 120.0,46.0 120.6,46.0 121.1,46.0 121.6,46.0 122.1,46.0 122.6,46.0 123.2,46.0 123.7,46.0 124.2,46.0 124.7,46.0 125.2,46.0 125.8,46.0 126.3,46.0 126.8,46.0 127.3,46.0 127.8,46.0 128.4,46.0 128.9,46.0 129.4,46.0 129.9,46.0 130.4,46.0 131.0,46.0 131.5,46.0 132.0,46.0 132.5,46.0 133.0,46.0 133.6,46.0 134.1,46.0 134.6,46.0 135.1,46.0 135.6,46.0 136.2,46.0 136.7,46.0 137.2,46.0 137.7,46.0 138.2,46.0 138.8,46.0 139.3,46.0 139.8,46.0 140.3,46.0 140.8,45.7 141.4,45.0 141.9,44.2 142.4,43.5 142.9,42.8 143.4,42.0 144.0,41.3 144.5,40.6 145.0,39.9 145.5,39.2 146.0,38.5 146.6,37.9 147.1,37.2 147.6,36.6 148.1,36.0 148.6,35.4 149.2,34.9 149.7,34.3 150.2,33.8 150.7,33.4 151.2,32.9 151.8,32.5 152.3,32.1 152.8,31.7 153.3,31.4 153.8,31.1 154.4,30.9 154.9,30.6 155.4,30.4 155.9,30.3 156.4,30.2 157.0,30.1 157.5,30.0 158.0,30.0 158.5,30.0 159.0,30.1 159.6,30.2 160.1,30.3 160.6,30.4 161.1,30.6 161.6,30.9 162.2,31.1 162.7,31.4 163.2,31.7 163.7,32.1 164.2,32.5 164.8,32.9 165.3,33.4 165.8,33.8 166.3,34.3 166.8,34.9 167.4,35.4 167.9,36.0 168.4,36.6 168.9,37.2 169.4,37.9 170.0,38.5 170.5,39.2 171.0,39.9 171.5,40.6 172.0,41.3 172.6,42.0 173.1,42.8 173.6,43.5 174.1,44.2 174.6,45.0 175.2,45.7 175.7,46.0 176.2,46.0 176.7,46.0 177.2,46.0 177.8,46.0 178.3,46.0 178.8,46.0 179.3,46.0 179.8,46.0 180.4,46.0 180.9,46.0 181.4,46.0 181.9,46.0 182.4,46.0 183.0,46.0 183.5,46.0 184.0,46.0 184.5,46.0 185.0,46.0 185.6,46.0 186.1,46.0 186.6,46.0 187.1,46.0 187.6,46.0 188.2,46.0 188.7,46.0 189.2,46.0 189.7,46.0 190.2,46.0 190.8,46.0 191.3,46.0 191.8,46.0 192.3,46.0 192.8,46.0 193.4,46.0 193.9,46.0 194.4,46.0 194.9,46.0 195.4,46.0 196.0,46.0 196.5,46.0 197.0,46.0 197.5,46.0 198.0,46.0 198.6,46.0 199.1,46.0 199.6,46.0 200.1,46.0 200.6,46.0 201.2,46.0 201.7,46.0 202.2,46.0 202.7,46.0 203.2,46.0 203.8,46.0 204.3,46.0 204.8,46.0 205.3,46.0 205.8,46.0 206.4,46.0 206.9,46.0 207.4,46.0 207.9,46.0 208.4,46.0 209.0,46.0 209.5,46.0 210.0,46.0 210.5,45.2 211.0,44.5 211.6,43.7 212.1,43.0 212.6,42.3 213.1,41.5 213.6,40.8 214.2,40.1 214.7,39.4 215.2,38.7 215.7,38.1 216.2,37.4 216.8,36.8 217.3,36.2 217.8,35.6 218.3,35.0 218.8,34.5 219.4,34.0 219.9,33.5 220.4,33.1 220.9,32.6 221.4,32.2 222.0,31.9 222.5,31.5 223.0,31.2 223.5,30.9 224.0,30.7 224.6,30.5 225.1,30.3 225.6,30.2 226.1,30.1 226.6,30.0 227.2,30.0 227.7,30.0 228.2,30.0 228.7,30.1 229.2,30.2 229.8,30.4 230.3,30.6 230.8,30.8 231.3,31.0 231.8,31.3 232.4,31.6 232.9,32.0 233.4,32.4 233.9,32.8 234.4,33.2 235.0,33.7 235.5,34.2 236.0,34.7 236.5,35.2 237.0,35.8 237.6,36.4 238.1,37.0 238.6,37.6 239.1,38.3 239.6,39.0 240.2,39.6 240.7,40.3 241.2,41.1 241.7,41.8 242.2,42.5 242.8,43.2 243.3,44.0 243.8,44.7 244.3,45.5 244.8,46.0 245.4,46.0 245.9,46.0 246.4,46.0 246.9,46.0 247.4,46.0 248.0,46.0 248.5,46.0 249.0,46.0 249.5,46.0 250.0,46.0 250.6,46.0 251.1,46.0 251.6,46.0 252.1,46.0 252.6,46.0 253.2,46.0 253.7,46.0 254.2,46.0 254.7,46.0 255.2,46.0 255.8,46.0 256.3,46.0 256.8,46.0 257.3,46.0 257.8,46.0 258.4,46.0 258.9,46.0 259.4,46.0 259.9,46.0 260.4,46.0 261.0,46.0 261.5,46.0 262.0,46.0 L 262.0,46 Z" /></g>
        <g fill="#c62828" opacity="0.4"><path d="M 54.0,46 L 54.0,62.0 54.5,62.0 55.0,61.9 55.6,61.8 56.1,61.7 56.6,61.6 57.1,61.4 57.6,61.1 58.2,60.9 58.7,60.6 59.2,60.3 59.7,59.9 60.2,59.5 60.8,59.1 61.3,58.6 61.8,58.2 62.3,57.7 62.8,57.1 63.4,56.6 63.9,56.0 64.4,55.4 64.9,54.8 65.4,54.1 66.0,53.5 66.5,52.8 67.0,52.1 67.5,51.4 68.0,50.7 68.6,50.0 69.1,49.2 69.6,48.5 70.1,47.8 70.6,47.0 71.2,46.3 71.7,46.0 72.2,46.0 72.7,46.0 73.2,46.0 73.8,46.0 74.3,46.0 74.8,46.0 75.3,46.0 75.8,46.0 76.4,46.0 76.9,46.0 77.4,46.0 77.9,46.0 78.4,46.0 79.0,46.0 79.5,46.0 80.0,46.0 80.5,46.0 81.0,46.0 81.6,46.0 82.1,46.0 82.6,46.0 83.1,46.0 83.6,46.0 84.2,46.0 84.7,46.0 85.2,46.0 85.7,46.0 86.2,46.0 86.8,46.0 87.3,46.0 87.8,46.0 88.3,46.0 88.8,46.0 89.4,46.0 89.9,46.0 90.4,46.0 90.9,46.0 91.4,46.0 92.0,46.0 92.5,46.0 93.0,46.0 93.5,46.0 94.0,46.0 94.6,46.0 95.1,46.0 95.6,46.0 96.1,46.0 96.6,46.0 97.2,46.0 97.7,46.0 98.2,46.0 98.7,46.0 99.2,46.0 99.8,46.0 100.3,46.0 100.8,46.0 101.3,46.0 101.8,46.0 102.4,46.0 102.9,46.0 103.4,46.0 103.9,46.0 104.4,46.0 105.0,46.0 105.5,46.0 106.0,46.0 106.5,46.8 107.0,47.5 107.6,48.3 108.1,49.0 108.6,49.7 109.1,50.5 109.6,51.2 110.2,51.9 110.7,52.6 111.2,53.3 111.7,53.9 112.2,54.6 112.8,55.2 113.3,55.8 113.8,56.4 114.3,57.0 114.8,57.5 115.4,58.0 115.9,58.5 116.4,58.9 116.9,59.4 117.4,59.8 118.0,60.1 118.5,60.5 119.0,60.8 119.5,61.1 120.0,61.3 120.6,61.5 121.1,61.7 121.6,61.8 122.1,61.9 122.6,62.0 123.2,62.0 123.7,62.0 124.2,62.0 124.7,61.9 125.2,61.8 125.8,61.6 126.3,61.4 126.8,61.2 127.3,61.0 127.8,60.7 128.4,60.4 128.9,60.0 129.4,59.6 129.9,59.2 130.4,58.8 131.0,58.3 131.5,57.8 132.0,57.3 132.5,56.8 133.0,56.2 133.6,55.6 134.1,55.0 134.6,54.4 135.1,53.7 135.6,53.0 136.2,52.4 136.7,51.7 137.2,50.9 137.7,50.2 138.2,49.5 138.8,48.8 139.3,48.0 139.8,47.3 140.3,46.5 140.8,46.0 141.4,46.0 141.9,46.0 142.4,46.0 142.9,46.0 143.4,46.0 144.0,46.0 144.5,46.0 145.0,46.0 145.5,46.0 146.0,46.0 146.6,46.0 147.1,46.0 147.6,46.0 148.1,46.0 148.6,46.0 149.2,46.0 149.7,46.0 150.2,46.0 150.7,46.0 151.2,46.0 151.8,46.0 152.3,46.0 152.8,46.0 153.3,46.0 153.8,46.0 154.4,46.0 154.9,46.0 155.4,46.0 155.9,46.0 156.4,46.0 157.0,46.0 157.5,46.0 158.0,46.0 158.5,46.0 159.0,46.0 159.6,46.0 160.1,46.0 160.6,46.0 161.1,46.0 161.6,46.0 162.2,46.0 162.7,46.0 163.2,46.0 163.7,46.0 164.2,46.0 164.8,46.0 165.3,46.0 165.8,46.0 166.3,46.0 166.8,46.0 167.4,46.0 167.9,46.0 168.4,46.0 168.9,46.0 169.4,46.0 170.0,46.0 170.5,46.0 171.0,46.0 171.5,46.0 172.0,46.0 172.6,46.0 173.1,46.0 173.6,46.0 174.1,46.0 174.6,46.0 175.2,46.0 175.7,46.5 176.2,47.3 176.7,48.0 177.2,48.8 177.8,49.5 178.3,50.2 178.8,50.9 179.3,51.7 179.8,52.4 180.4,53.0 180.9,53.7 181.4,54.4 181.9,55.0 182.4,55.6 183.0,56.2 183.5,56.8 184.0,57.3 184.5,57.8 185.0,58.3 185.6,58.8 186.1,59.2 186.6,59.6 187.1,60.0 187.6,60.4 188.2,60.7 188.7,61.0 189.2,61.2 189.7,61.4 190.2,61.6 190.8,61.8 191.3,61.9 191.8,62.0 192.3,62.0 192.8,62.0 193.4,62.0 193.9,61.9 194.4,61.8 194.9,61.7 195.4,61.5 196.0,61.3 196.5,61.1 197.0,60.8 197.5,60.5 198.0,60.1 198.6,59.8 199.1,59.4 199.6,58.9 200.1,58.5 200.6,58.0 201.2,57.5 201.7,57.0 202.2,56.4 202.7,55.8 203.2,55.2 203.8,54.6 204.3,53.9 204.8,53.3 205.3,52.6 205.8,51.9 206.4,51.2 206.9,50.5 207.4,49.7 207.9,49.0 208.4,48.3 209.0,47.5 209.5,46.8 210.0,46.0 210.5,46.0 211.0,46.0 211.6,46.0 212.1,46.0 212.6,46.0 213.1,46.0 213.6,46.0 214.2,46.0 214.7,46.0 215.2,46.0 215.7,46.0 216.2,46.0 216.8,46.0 217.3,46.0 217.8,46.0 218.3,46.0 218.8,46.0 219.4,46.0 219.9,46.0 220.4,46.0 220.9,46.0 221.4,46.0 222.0,46.0 222.5,46.0 223.0,46.0 223.5,46.0 224.0,46.0 224.6,46.0 225.1,46.0 225.6,46.0 226.1,46.0 226.6,46.0 227.2,46.0 227.7,46.0 228.2,46.0 228.7,46.0 229.2,46.0 229.8,46.0 230.3,46.0 230.8,46.0 231.3,46.0 231.8,46.0 232.4,46.0 232.9,46.0 233.4,46.0 233.9,46.0 234.4,46.0 235.0,46.0 235.5,46.0 236.0,46.0 236.5,46.0 237.0,46.0 237.6,46.0 238.1,46.0 238.6,46.0 239.1,46.0 239.6,46.0 240.2,46.0 240.7,46.0 241.2,46.0 241.7,46.0 242.2,46.0 242.8,46.0 243.3,46.0 243.8,46.0 244.3,46.0 244.8,46.3 245.4,47.0 245.9,47.8 246.4,48.5 246.9,49.2 247.4,50.0 248.0,50.7 248.5,51.4 249.0,52.1 249.5,52.8 250.0,53.5 250.6,54.1 251.1,54.8 251.6,55.4 252.1,56.0 252.6,56.6 253.2,57.1 253.7,57.7 254.2,58.2 254.7,58.6 255.2,59.1 255.8,59.5 256.3,59.9 256.8,60.3 257.3,60.6 257.8,60.9 258.4,61.1 258.9,61.4 259.4,61.6 259.9,61.7 260.4,61.8 261.0,61.9 261.5,62.0 262.0,62.0 L 262.0,46 Z" /></g>
        <polyline points="54.0,62.0 54.5,62.0 55.0,61.9 55.6,61.8 56.1,61.7 56.6,61.6 57.1,61.4 57.6,61.1 58.2,60.9 58.7,60.6 59.2,60.3 59.7,59.9 60.2,59.5 60.8,59.1 61.3,58.6 61.8,58.2 62.3,57.7 62.8,57.1 63.4,56.6 63.9,56.0 64.4,55.4 64.9,54.8 65.4,54.1 66.0,53.5 66.5,52.8 67.0,52.1 67.5,51.4 68.0,50.7 68.6,50.0 69.1,49.2 69.6,48.5 70.1,47.8 70.6,47.0 71.2,46.3 71.7,45.5 72.2,44.7 72.7,44.0 73.2,43.2 73.8,42.5 74.3,41.8 74.8,41.1 75.3,40.3 75.8,39.6 76.4,39.0 76.9,38.3 77.4,37.6 77.9,37.0 78.4,36.4 79.0,35.8 79.5,35.2 80.0,34.7 80.5,34.2 81.0,33.7 81.6,33.2 82.1,32.8 82.6,32.4 83.1,32.0 83.6,31.6 84.2,31.3 84.7,31.0 85.2,30.8 85.7,30.6 86.2,30.4 86.8,30.2 87.3,30.1 87.8,30.0 88.3,30.0 88.8,30.0 89.4,30.0 89.9,30.1 90.4,30.2 90.9,30.3 91.4,30.5 92.0,30.7 92.5,30.9 93.0,31.2 93.5,31.5 94.0,31.9 94.6,32.2 95.1,32.6 95.6,33.1 96.1,33.5 96.6,34.0 97.2,34.5 97.7,35.0 98.2,35.6 98.7,36.2 99.2,36.8 99.8,37.4 100.3,38.1 100.8,38.7 101.3,39.4 101.8,40.1 102.4,40.8 102.9,41.5 103.4,42.3 103.9,43.0 104.4,43.7 105.0,44.5 105.5,45.2 106.0,46.0 106.5,46.8 107.0,47.5 107.6,48.3 108.1,49.0 108.6,49.7 109.1,50.5 109.6,51.2 110.2,51.9 110.7,52.6 111.2,53.3 111.7,53.9 112.2,54.6 112.8,55.2 113.3,55.8 113.8,56.4 114.3,57.0 114.8,57.5 115.4,58.0 115.9,58.5 116.4,58.9 116.9,59.4 117.4,59.8 118.0,60.1 118.5,60.5 119.0,60.8 119.5,61.1 120.0,61.3 120.6,61.5 121.1,61.7 121.6,61.8 122.1,61.9 122.6,62.0 123.2,62.0 123.7,62.0 124.2,62.0 124.7,61.9 125.2,61.8 125.8,61.6 126.3,61.4 126.8,61.2 127.3,61.0 127.8,60.7 128.4,60.4 128.9,60.0 129.4,59.6 129.9,59.2 130.4,58.8 131.0,58.3 131.5,57.8 132.0,57.3 132.5,56.8 133.0,56.2 133.6,55.6 134.1,55.0 134.6,54.4 135.1,53.7 135.6,53.0 136.2,52.4 136.7,51.7 137.2,50.9 137.7,50.2 138.2,49.5 138.8,48.8 139.3,48.0 139.8,47.3 140.3,46.5 140.8,45.7 141.4,45.0 141.9,44.2 142.4,43.5 142.9,42.8 143.4,42.0 144.0,41.3 144.5,40.6 145.0,39.9 145.5,39.2 146.0,38.5 146.6,37.9 147.1,37.2 147.6,36.6 148.1,36.0 148.6,35.4 149.2,34.9 149.7,34.3 150.2,33.8 150.7,33.4 151.2,32.9 151.8,32.5 152.3,32.1 152.8,31.7 153.3,31.4 153.8,31.1 154.4,30.9 154.9,30.6 155.4,30.4 155.9,30.3 156.4,30.2 157.0,30.1 157.5,30.0 158.0,30.0 158.5,30.0 159.0,30.1 159.6,30.2 160.1,30.3 160.6,30.4 161.1,30.6 161.6,30.9 162.2,31.1 162.7,31.4 163.2,31.7 163.7,32.1 164.2,32.5 164.8,32.9 165.3,33.4 165.8,33.8 166.3,34.3 166.8,34.9 167.4,35.4 167.9,36.0 168.4,36.6 168.9,37.2 169.4,37.9 170.0,38.5 170.5,39.2 171.0,39.9 171.5,40.6 172.0,41.3 172.6,42.0 173.1,42.8 173.6,43.5 174.1,44.2 174.6,45.0 175.2,45.7 175.7,46.5 176.2,47.3 176.7,48.0 177.2,48.8 177.8,49.5 178.3,50.2 178.8,50.9 179.3,51.7 179.8,52.4 180.4,53.0 180.9,53.7 181.4,54.4 181.9,55.0 182.4,55.6 183.0,56.2 183.5,56.8 184.0,57.3 184.5,57.8 185.0,58.3 185.6,58.8 186.1,59.2 186.6,59.6 187.1,60.0 187.6,60.4 188.2,60.7 188.7,61.0 189.2,61.2 189.7,61.4 190.2,61.6 190.8,61.8 191.3,61.9 191.8,62.0 192.3,62.0 192.8,62.0 193.4,62.0 193.9,61.9 194.4,61.8 194.9,61.7 195.4,61.5 196.0,61.3 196.5,61.1 197.0,60.8 197.5,60.5 198.0,60.1 198.6,59.8 199.1,59.4 199.6,58.9 200.1,58.5 200.6,58.0 201.2,57.5 201.7,57.0 202.2,56.4 202.7,55.8 203.2,55.2 203.8,54.6 204.3,53.9 204.8,53.3 205.3,52.6 205.8,51.9 206.4,51.2 206.9,50.5 207.4,49.7 207.9,49.0 208.4,48.3 209.0,47.5 209.5,46.8 210.0,46.0 210.5,45.2 211.0,44.5 211.6,43.7 212.1,43.0 212.6,42.3 213.1,41.5 213.6,40.8 214.2,40.1 214.7,39.4 215.2,38.7 215.7,38.1 216.2,37.4 216.8,36.8 217.3,36.2 217.8,35.6 218.3,35.0 218.8,34.5 219.4,34.0 219.9,33.5 220.4,33.1 220.9,32.6 221.4,32.2 222.0,31.9 222.5,31.5 223.0,31.2 223.5,30.9 224.0,30.7 224.6,30.5 225.1,30.3 225.6,30.2 226.1,30.1 226.6,30.0 227.2,30.0 227.7,30.0 228.2,30.0 228.7,30.1 229.2,30.2 229.8,30.4 230.3,30.6 230.8,30.8 231.3,31.0 231.8,31.3 232.4,31.6 232.9,32.0 233.4,32.4 233.9,32.8 234.4,33.2 235.0,33.7 235.5,34.2 236.0,34.7 236.5,35.2 237.0,35.8 237.6,36.4 238.1,37.0 238.6,37.6 239.1,38.3 239.6,39.0 240.2,39.6 240.7,40.3 241.2,41.1 241.7,41.8 242.2,42.5 242.8,43.2 243.3,44.0 243.8,44.7 244.3,45.5 244.8,46.3 245.4,47.0 245.9,47.8 246.4,48.5 246.9,49.2 247.4,50.0 248.0,50.7 248.5,51.4 249.0,52.1 249.5,52.8 250.0,53.5 250.6,54.1 251.1,54.8 251.6,55.4 252.1,56.0 252.6,56.6 253.2,57.1 253.7,57.7 254.2,58.2 254.7,58.6 255.2,59.1 255.8,59.5 256.3,59.9 256.8,60.3 257.3,60.6 257.8,60.9 258.4,61.1 258.9,61.4 259.4,61.6 259.9,61.7 260.4,61.8 261.0,61.9 261.5,62.0 262.0,62.0" fill="none" stroke="#1f2430" stroke-width="1" />
        <text x="49.0" y="49" font-size="8.5" fill="#666" text-anchor="end">cos 3x</text>
        <text x="267.0" y="49" font-size="8.5" fill="#1f2430">∫ = 0</text>
        <line x1="54.0" y1="96" x2="262.0" y2="96" stroke="#bbb" stroke-width="0.8" />
        <g fill="#1565c0" opacity="0.4"><path d="M 54.0,96 L 54.0,96.0 54.5,96.0 55.0,96.0 55.6,96.0 56.1,96.0 56.6,96.0 57.1,96.0 57.6,96.0 58.2,96.0 58.7,96.0 59.2,96.0 59.7,96.0 60.2,96.0 60.8,96.0 61.3,96.0 61.8,96.0 62.3,96.0 62.8,96.0 63.4,96.0 63.9,96.0 64.4,96.0 64.9,96.0 65.4,96.0 66.0,96.0 66.5,96.0 67.0,96.0 67.5,96.0 68.0,96.0 68.6,96.0 69.1,96.0 69.6,96.0 70.1,96.0 70.6,96.0 71.2,96.0 71.7,96.0 72.2,96.0 72.7,96.0 73.2,96.0 73.8,96.0 74.3,96.0 74.8,96.0 75.3,96.0 75.8,96.0 76.4,96.0 76.9,96.0 77.4,96.0 77.9,96.0 78.4,96.0 79.0,96.0 79.5,96.0 80.0,96.0 80.5,96.0 81.0,96.0 81.6,96.0 82.1,96.0 82.6,96.0 83.1,96.0 83.6,96.0 84.2,96.0 84.7,96.0 85.2,96.0 85.7,96.0 86.2,96.0 86.8,96.0 87.3,96.0 87.8,96.0 88.3,96.0 88.8,95.7 89.4,95.0 89.9,94.2 90.4,93.5 90.9,92.8 91.4,92.0 92.0,91.3 92.5,90.6 93.0,89.9 93.5,89.2 94.0,88.5 94.6,87.9 95.1,87.2 95.6,86.6 96.1,86.0 96.6,85.4 97.2,84.9 97.7,84.3 98.2,83.8 98.7,83.4 99.2,82.9 99.8,82.5 100.3,82.1 100.8,81.7 101.3,81.4 101.8,81.1 102.4,80.9 102.9,80.6 103.4,80.4 103.9,80.3 104.4,80.2 105.0,80.1 105.5,80.0 106.0,80.0 106.5,80.0 107.0,80.1 107.6,80.2 108.1,80.3 108.6,80.4 109.1,80.6 109.6,80.9 110.2,81.1 110.7,81.4 111.2,81.7 111.7,82.1 112.2,82.5 112.8,82.9 113.3,83.4 113.8,83.8 114.3,84.3 114.8,84.9 115.4,85.4 115.9,86.0 116.4,86.6 116.9,87.2 117.4,87.9 118.0,88.5 118.5,89.2 119.0,89.9 119.5,90.6 120.0,91.3 120.6,92.0 121.1,92.8 121.6,93.5 122.1,94.2 122.6,95.0 123.2,95.7 123.7,96.0 124.2,96.0 124.7,96.0 125.2,96.0 125.8,96.0 126.3,96.0 126.8,96.0 127.3,96.0 127.8,96.0 128.4,96.0 128.9,96.0 129.4,96.0 129.9,96.0 130.4,96.0 131.0,96.0 131.5,96.0 132.0,96.0 132.5,96.0 133.0,96.0 133.6,96.0 134.1,96.0 134.6,96.0 135.1,96.0 135.6,96.0 136.2,96.0 136.7,96.0 137.2,96.0 137.7,96.0 138.2,96.0 138.8,96.0 139.3,96.0 139.8,96.0 140.3,96.0 140.8,96.0 141.4,96.0 141.9,96.0 142.4,96.0 142.9,96.0 143.4,96.0 144.0,96.0 144.5,96.0 145.0,96.0 145.5,96.0 146.0,96.0 146.6,96.0 147.1,96.0 147.6,96.0 148.1,96.0 148.6,96.0 149.2,96.0 149.7,96.0 150.2,96.0 150.7,96.0 151.2,96.0 151.8,96.0 152.3,96.0 152.8,96.0 153.3,96.0 153.8,96.0 154.4,96.0 154.9,96.0 155.4,96.0 155.9,96.0 156.4,96.0 157.0,96.0 157.5,96.0 158.0,96.0 158.5,95.2 159.0,94.5 159.6,93.7 160.1,93.0 160.6,92.3 161.1,91.5 161.6,90.8 162.2,90.1 162.7,89.4 163.2,88.7 163.7,88.1 164.2,87.4 164.8,86.8 165.3,86.2 165.8,85.6 166.3,85.0 166.8,84.5 167.4,84.0 167.9,83.5 168.4,83.1 168.9,82.6 169.4,82.2 170.0,81.9 170.5,81.5 171.0,81.2 171.5,80.9 172.0,80.7 172.6,80.5 173.1,80.3 173.6,80.2 174.1,80.1 174.6,80.0 175.2,80.0 175.7,80.0 176.2,80.0 176.7,80.1 177.2,80.2 177.8,80.4 178.3,80.6 178.8,80.8 179.3,81.0 179.8,81.3 180.4,81.6 180.9,82.0 181.4,82.4 181.9,82.8 182.4,83.2 183.0,83.7 183.5,84.2 184.0,84.7 184.5,85.2 185.0,85.8 185.6,86.4 186.1,87.0 186.6,87.6 187.1,88.3 187.6,89.0 188.2,89.6 188.7,90.3 189.2,91.1 189.7,91.8 190.2,92.5 190.8,93.2 191.3,94.0 191.8,94.7 192.3,95.5 192.8,96.0 193.4,96.0 193.9,96.0 194.4,96.0 194.9,96.0 195.4,96.0 196.0,96.0 196.5,96.0 197.0,96.0 197.5,96.0 198.0,96.0 198.6,96.0 199.1,96.0 199.6,96.0 200.1,96.0 200.6,96.0 201.2,96.0 201.7,96.0 202.2,96.0 202.7,96.0 203.2,96.0 203.8,96.0 204.3,96.0 204.8,96.0 205.3,96.0 205.8,96.0 206.4,96.0 206.9,96.0 207.4,96.0 207.9,96.0 208.4,96.0 209.0,96.0 209.5,96.0 210.0,96.0 210.5,96.0 211.0,96.0 211.6,96.0 212.1,96.0 212.6,96.0 213.1,96.0 213.6,96.0 214.2,96.0 214.7,96.0 215.2,96.0 215.7,96.0 216.2,96.0 216.8,96.0 217.3,96.0 217.8,96.0 218.3,96.0 218.8,96.0 219.4,96.0 219.9,96.0 220.4,96.0 220.9,96.0 221.4,96.0 222.0,96.0 222.5,96.0 223.0,96.0 223.5,96.0 224.0,96.0 224.6,96.0 225.1,96.0 225.6,96.0 226.1,96.0 226.6,96.0 227.2,96.0 227.7,95.5 228.2,94.7 228.7,94.0 229.2,93.2 229.8,92.5 230.3,91.8 230.8,91.1 231.3,90.3 231.8,89.6 232.4,89.0 232.9,88.3 233.4,87.6 233.9,87.0 234.4,86.4 235.0,85.8 235.5,85.2 236.0,84.7 236.5,84.2 237.0,83.7 237.6,83.2 238.1,82.8 238.6,82.4 239.1,82.0 239.6,81.6 240.2,81.3 240.7,81.0 241.2,80.8 241.7,80.6 242.2,80.4 242.8,80.2 243.3,80.1 243.8,80.0 244.3,80.0 244.8,80.0 245.4,80.0 245.9,80.1 246.4,80.2 246.9,80.3 247.4,80.5 248.0,80.7 248.5,80.9 249.0,81.2 249.5,81.5 250.0,81.9 250.6,82.2 251.1,82.6 251.6,83.1 252.1,83.5 252.6,84.0 253.2,84.5 253.7,85.0 254.2,85.6 254.7,86.2 255.2,86.8 255.8,87.4 256.3,88.1 256.8,88.7 257.3,89.4 257.8,90.1 258.4,90.8 258.9,91.5 259.4,92.3 259.9,93.0 260.4,93.7 261.0,94.5 261.5,95.2 262.0,96.0 L 262.0,96 Z" /></g>
        <g fill="#c62828" opacity="0.4"><path d="M 54.0,96 L 54.0,96.0 54.5,96.8 55.0,97.5 55.6,98.3 56.1,99.0 56.6,99.7 57.1,100.5 57.6,101.2 58.2,101.9 58.7,102.6 59.2,103.3 59.7,103.9 60.2,104.6 60.8,105.2 61.3,105.8 61.8,106.4 62.3,107.0 62.8,107.5 63.4,108.0 63.9,108.5 64.4,108.9 64.9,109.4 65.4,109.8 66.0,110.1 66.5,110.5 67.0,110.8 67.5,111.1 68.0,111.3 68.6,111.5 69.1,111.7 69.6,111.8 70.1,111.9 70.6,112.0 71.2,112.0 71.7,112.0 72.2,112.0 72.7,111.9 73.2,111.8 73.8,111.6 74.3,111.4 74.8,111.2 75.3,111.0 75.8,110.7 76.4,110.4 76.9,110.0 77.4,109.6 77.9,109.2 78.4,108.8 79.0,108.3 79.5,107.8 80.0,107.3 80.5,106.8 81.0,106.2 81.6,105.6 82.1,105.0 82.6,104.4 83.1,103.7 83.6,103.0 84.2,102.4 84.7,101.7 85.2,100.9 85.7,100.2 86.2,99.5 86.8,98.8 87.3,98.0 87.8,97.3 88.3,96.5 88.8,96.0 89.4,96.0 89.9,96.0 90.4,96.0 90.9,96.0 91.4,96.0 92.0,96.0 92.5,96.0 93.0,96.0 93.5,96.0 94.0,96.0 94.6,96.0 95.1,96.0 95.6,96.0 96.1,96.0 96.6,96.0 97.2,96.0 97.7,96.0 98.2,96.0 98.7,96.0 99.2,96.0 99.8,96.0 100.3,96.0 100.8,96.0 101.3,96.0 101.8,96.0 102.4,96.0 102.9,96.0 103.4,96.0 103.9,96.0 104.4,96.0 105.0,96.0 105.5,96.0 106.0,96.0 106.5,96.0 107.0,96.0 107.6,96.0 108.1,96.0 108.6,96.0 109.1,96.0 109.6,96.0 110.2,96.0 110.7,96.0 111.2,96.0 111.7,96.0 112.2,96.0 112.8,96.0 113.3,96.0 113.8,96.0 114.3,96.0 114.8,96.0 115.4,96.0 115.9,96.0 116.4,96.0 116.9,96.0 117.4,96.0 118.0,96.0 118.5,96.0 119.0,96.0 119.5,96.0 120.0,96.0 120.6,96.0 121.1,96.0 121.6,96.0 122.1,96.0 122.6,96.0 123.2,96.0 123.7,96.5 124.2,97.3 124.7,98.0 125.2,98.8 125.8,99.5 126.3,100.2 126.8,100.9 127.3,101.7 127.8,102.4 128.4,103.0 128.9,103.7 129.4,104.4 129.9,105.0 130.4,105.6 131.0,106.2 131.5,106.8 132.0,107.3 132.5,107.8 133.0,108.3 133.6,108.8 134.1,109.2 134.6,109.6 135.1,110.0 135.6,110.4 136.2,110.7 136.7,111.0 137.2,111.2 137.7,111.4 138.2,111.6 138.8,111.8 139.3,111.9 139.8,112.0 140.3,112.0 140.8,112.0 141.4,112.0 141.9,111.9 142.4,111.8 142.9,111.7 143.4,111.5 144.0,111.3 144.5,111.1 145.0,110.8 145.5,110.5 146.0,110.1 146.6,109.8 147.1,109.4 147.6,108.9 148.1,108.5 148.6,108.0 149.2,107.5 149.7,107.0 150.2,106.4 150.7,105.8 151.2,105.2 151.8,104.6 152.3,103.9 152.8,103.3 153.3,102.6 153.8,101.9 154.4,101.2 154.9,100.5 155.4,99.7 155.9,99.0 156.4,98.3 157.0,97.5 157.5,96.8 158.0,96.0 158.5,96.0 159.0,96.0 159.6,96.0 160.1,96.0 160.6,96.0 161.1,96.0 161.6,96.0 162.2,96.0 162.7,96.0 163.2,96.0 163.7,96.0 164.2,96.0 164.8,96.0 165.3,96.0 165.8,96.0 166.3,96.0 166.8,96.0 167.4,96.0 167.9,96.0 168.4,96.0 168.9,96.0 169.4,96.0 170.0,96.0 170.5,96.0 171.0,96.0 171.5,96.0 172.0,96.0 172.6,96.0 173.1,96.0 173.6,96.0 174.1,96.0 174.6,96.0 175.2,96.0 175.7,96.0 176.2,96.0 176.7,96.0 177.2,96.0 177.8,96.0 178.3,96.0 178.8,96.0 179.3,96.0 179.8,96.0 180.4,96.0 180.9,96.0 181.4,96.0 181.9,96.0 182.4,96.0 183.0,96.0 183.5,96.0 184.0,96.0 184.5,96.0 185.0,96.0 185.6,96.0 186.1,96.0 186.6,96.0 187.1,96.0 187.6,96.0 188.2,96.0 188.7,96.0 189.2,96.0 189.7,96.0 190.2,96.0 190.8,96.0 191.3,96.0 191.8,96.0 192.3,96.0 192.8,96.3 193.4,97.0 193.9,97.8 194.4,98.5 194.9,99.2 195.4,100.0 196.0,100.7 196.5,101.4 197.0,102.1 197.5,102.8 198.0,103.5 198.6,104.1 199.1,104.8 199.6,105.4 200.1,106.0 200.6,106.6 201.2,107.1 201.7,107.7 202.2,108.2 202.7,108.6 203.2,109.1 203.8,109.5 204.3,109.9 204.8,110.3 205.3,110.6 205.8,110.9 206.4,111.1 206.9,111.4 207.4,111.6 207.9,111.7 208.4,111.8 209.0,111.9 209.5,112.0 210.0,112.0 210.5,112.0 211.0,111.9 211.6,111.8 212.1,111.7 212.6,111.6 213.1,111.4 213.6,111.1 214.2,110.9 214.7,110.6 215.2,110.3 215.7,109.9 216.2,109.5 216.8,109.1 217.3,108.6 217.8,108.2 218.3,107.7 218.8,107.1 219.4,106.6 219.9,106.0 220.4,105.4 220.9,104.8 221.4,104.1 222.0,103.5 222.5,102.8 223.0,102.1 223.5,101.4 224.0,100.7 224.6,100.0 225.1,99.2 225.6,98.5 226.1,97.8 226.6,97.0 227.2,96.3 227.7,96.0 228.2,96.0 228.7,96.0 229.2,96.0 229.8,96.0 230.3,96.0 230.8,96.0 231.3,96.0 231.8,96.0 232.4,96.0 232.9,96.0 233.4,96.0 233.9,96.0 234.4,96.0 235.0,96.0 235.5,96.0 236.0,96.0 236.5,96.0 237.0,96.0 237.6,96.0 238.1,96.0 238.6,96.0 239.1,96.0 239.6,96.0 240.2,96.0 240.7,96.0 241.2,96.0 241.7,96.0 242.2,96.0 242.8,96.0 243.3,96.0 243.8,96.0 244.3,96.0 244.8,96.0 245.4,96.0 245.9,96.0 246.4,96.0 246.9,96.0 247.4,96.0 248.0,96.0 248.5,96.0 249.0,96.0 249.5,96.0 250.0,96.0 250.6,96.0 251.1,96.0 251.6,96.0 252.1,96.0 252.6,96.0 253.2,96.0 253.7,96.0 254.2,96.0 254.7,96.0 255.2,96.0 255.8,96.0 256.3,96.0 256.8,96.0 257.3,96.0 257.8,96.0 258.4,96.0 258.9,96.0 259.4,96.0 259.9,96.0 260.4,96.0 261.0,96.0 261.5,96.0 262.0,96.0 L 262.0,96 Z" /></g>
        <polyline points="54.0,96.0 54.5,96.8 55.0,97.5 55.6,98.3 56.1,99.0 56.6,99.7 57.1,100.5 57.6,101.2 58.2,101.9 58.7,102.6 59.2,103.3 59.7,103.9 60.2,104.6 60.8,105.2 61.3,105.8 61.8,106.4 62.3,107.0 62.8,107.5 63.4,108.0 63.9,108.5 64.4,108.9 64.9,109.4 65.4,109.8 66.0,110.1 66.5,110.5 67.0,110.8 67.5,111.1 68.0,111.3 68.6,111.5 69.1,111.7 69.6,111.8 70.1,111.9 70.6,112.0 71.2,112.0 71.7,112.0 72.2,112.0 72.7,111.9 73.2,111.8 73.8,111.6 74.3,111.4 74.8,111.2 75.3,111.0 75.8,110.7 76.4,110.4 76.9,110.0 77.4,109.6 77.9,109.2 78.4,108.8 79.0,108.3 79.5,107.8 80.0,107.3 80.5,106.8 81.0,106.2 81.6,105.6 82.1,105.0 82.6,104.4 83.1,103.7 83.6,103.0 84.2,102.4 84.7,101.7 85.2,100.9 85.7,100.2 86.2,99.5 86.8,98.8 87.3,98.0 87.8,97.3 88.3,96.5 88.8,95.7 89.4,95.0 89.9,94.2 90.4,93.5 90.9,92.8 91.4,92.0 92.0,91.3 92.5,90.6 93.0,89.9 93.5,89.2 94.0,88.5 94.6,87.9 95.1,87.2 95.6,86.6 96.1,86.0 96.6,85.4 97.2,84.9 97.7,84.3 98.2,83.8 98.7,83.4 99.2,82.9 99.8,82.5 100.3,82.1 100.8,81.7 101.3,81.4 101.8,81.1 102.4,80.9 102.9,80.6 103.4,80.4 103.9,80.3 104.4,80.2 105.0,80.1 105.5,80.0 106.0,80.0 106.5,80.0 107.0,80.1 107.6,80.2 108.1,80.3 108.6,80.4 109.1,80.6 109.6,80.9 110.2,81.1 110.7,81.4 111.2,81.7 111.7,82.1 112.2,82.5 112.8,82.9 113.3,83.4 113.8,83.8 114.3,84.3 114.8,84.9 115.4,85.4 115.9,86.0 116.4,86.6 116.9,87.2 117.4,87.9 118.0,88.5 118.5,89.2 119.0,89.9 119.5,90.6 120.0,91.3 120.6,92.0 121.1,92.8 121.6,93.5 122.1,94.2 122.6,95.0 123.2,95.7 123.7,96.5 124.2,97.3 124.7,98.0 125.2,98.8 125.8,99.5 126.3,100.2 126.8,100.9 127.3,101.7 127.8,102.4 128.4,103.0 128.9,103.7 129.4,104.4 129.9,105.0 130.4,105.6 131.0,106.2 131.5,106.8 132.0,107.3 132.5,107.8 133.0,108.3 133.6,108.8 134.1,109.2 134.6,109.6 135.1,110.0 135.6,110.4 136.2,110.7 136.7,111.0 137.2,111.2 137.7,111.4 138.2,111.6 138.8,111.8 139.3,111.9 139.8,112.0 140.3,112.0 140.8,112.0 141.4,112.0 141.9,111.9 142.4,111.8 142.9,111.7 143.4,111.5 144.0,111.3 144.5,111.1 145.0,110.8 145.5,110.5 146.0,110.1 146.6,109.8 147.1,109.4 147.6,108.9 148.1,108.5 148.6,108.0 149.2,107.5 149.7,107.0 150.2,106.4 150.7,105.8 151.2,105.2 151.8,104.6 152.3,103.9 152.8,103.3 153.3,102.6 153.8,101.9 154.4,101.2 154.9,100.5 155.4,99.7 155.9,99.0 156.4,98.3 157.0,97.5 157.5,96.8 158.0,96.0 158.5,95.2 159.0,94.5 159.6,93.7 160.1,93.0 160.6,92.3 161.1,91.5 161.6,90.8 162.2,90.1 162.7,89.4 163.2,88.7 163.7,88.1 164.2,87.4 164.8,86.8 165.3,86.2 165.8,85.6 166.3,85.0 166.8,84.5 167.4,84.0 167.9,83.5 168.4,83.1 168.9,82.6 169.4,82.2 170.0,81.9 170.5,81.5 171.0,81.2 171.5,80.9 172.0,80.7 172.6,80.5 173.1,80.3 173.6,80.2 174.1,80.1 174.6,80.0 175.2,80.0 175.7,80.0 176.2,80.0 176.7,80.1 177.2,80.2 177.8,80.4 178.3,80.6 178.8,80.8 179.3,81.0 179.8,81.3 180.4,81.6 180.9,82.0 181.4,82.4 181.9,82.8 182.4,83.2 183.0,83.7 183.5,84.2 184.0,84.7 184.5,85.2 185.0,85.8 185.6,86.4 186.1,87.0 186.6,87.6 187.1,88.3 187.6,89.0 188.2,89.6 188.7,90.3 189.2,91.1 189.7,91.8 190.2,92.5 190.8,93.2 191.3,94.0 191.8,94.7 192.3,95.5 192.8,96.3 193.4,97.0 193.9,97.8 194.4,98.5 194.9,99.2 195.4,100.0 196.0,100.7 196.5,101.4 197.0,102.1 197.5,102.8 198.0,103.5 198.6,104.1 199.1,104.8 199.6,105.4 200.1,106.0 200.6,106.6 201.2,107.1 201.7,107.7 202.2,108.2 202.7,108.6 203.2,109.1 203.8,109.5 204.3,109.9 204.8,110.3 205.3,110.6 205.8,110.9 206.4,111.1 206.9,111.4 207.4,111.6 207.9,111.7 208.4,111.8 209.0,111.9 209.5,112.0 210.0,112.0 210.5,112.0 211.0,111.9 211.6,111.8 212.1,111.7 212.6,111.6 213.1,111.4 213.6,111.1 214.2,110.9 214.7,110.6 215.2,110.3 215.7,109.9 216.2,109.5 216.8,109.1 217.3,108.6 217.8,108.2 218.3,107.7 218.8,107.1 219.4,106.6 219.9,106.0 220.4,105.4 220.9,104.8 221.4,104.1 222.0,103.5 222.5,102.8 223.0,102.1 223.5,101.4 224.0,100.7 224.6,100.0 225.1,99.2 225.6,98.5 226.1,97.8 226.6,97.0 227.2,96.3 227.7,95.5 228.2,94.7 228.7,94.0 229.2,93.2 229.8,92.5 230.3,91.8 230.8,91.1 231.3,90.3 231.8,89.6 232.4,89.0 232.9,88.3 233.4,87.6 233.9,87.0 234.4,86.4 235.0,85.8 235.5,85.2 236.0,84.7 236.5,84.2 237.0,83.7 237.6,83.2 238.1,82.8 238.6,82.4 239.1,82.0 239.6,81.6 240.2,81.3 240.7,81.0 241.2,80.8 241.7,80.6 242.2,80.4 242.8,80.2 243.3,80.1 243.8,80.0 244.3,80.0 244.8,80.0 245.4,80.0 245.9,80.1 246.4,80.2 246.9,80.3 247.4,80.5 248.0,80.7 248.5,80.9 249.0,81.2 249.5,81.5 250.0,81.9 250.6,82.2 251.1,82.6 251.6,83.1 252.1,83.5 252.6,84.0 253.2,84.5 253.7,85.0 254.2,85.6 254.7,86.2 255.2,86.8 255.8,87.4 256.3,88.1 256.8,88.7 257.3,89.4 257.8,90.1 258.4,90.8 258.9,91.5 259.4,92.3 259.9,93.0 260.4,93.7 261.0,94.5 261.5,95.2 262.0,96.0" fill="none" stroke="#1f2430" stroke-width="1" />
        <text x="49.0" y="99" font-size="8.5" fill="#666" text-anchor="end">sin 3x</text>
        <text x="267.0" y="99" font-size="8.5" fill="#1f2430">∫ = 0</text>
        <line x1="54.0" y1="150" x2="262.0" y2="150" stroke="#bbb" stroke-width="0.8" />
        <g fill="#1565c0" opacity="0.4"><path d="M 54.0,150 L 54.0,134.0 54.5,134.0 55.0,134.0 55.6,134.0 56.1,134.0 56.6,134.0 57.1,134.0 57.6,134.0 58.2,134.0 58.7,134.0 59.2,134.0 59.7,134.0 60.2,134.0 60.8,134.0 61.3,134.0 61.8,134.0 62.3,134.0 62.8,134.0 63.4,134.0 63.9,134.0 64.4,134.0 64.9,134.0 65.4,134.0 66.0,134.0 66.5,134.0 67.0,134.0 67.5,134.0 68.0,134.0 68.6,134.0 69.1,134.0 69.6,134.0 70.1,134.0 70.6,134.0 71.2,134.0 71.7,134.0 72.2,134.0 72.7,134.0 73.2,134.0 73.8,134.0 74.3,134.0 74.8,134.0 75.3,134.0 75.8,134.0 76.4,134.0 76.9,134.0 77.4,134.0 77.9,134.0 78.4,134.0 79.0,134.0 79.5,134.0 80.0,134.0 80.5,134.0 81.0,134.0 81.6,134.0 82.1,134.0 82.6,134.0 83.1,134.0 83.6,134.0 84.2,134.0 84.7,134.0 85.2,134.0 85.7,134.0 86.2,134.0 86.8,134.0 87.3,134.0 87.8,134.0 88.3,134.0 88.8,134.0 89.4,134.0 89.9,134.0 90.4,134.0 90.9,134.0 91.4,134.0 92.0,134.0 92.5,134.0 93.0,134.0 93.5,134.0 94.0,134.0 94.6,134.0 95.1,134.0 95.6,134.0 96.1,134.0 96.6,134.0 97.2,134.0 97.7,134.0 98.2,134.0 98.7,134.0 99.2,134.0 99.8,134.0 100.3,134.0 100.8,134.0 101.3,134.0 101.8,134.0 102.4,134.0 102.9,134.0 103.4,134.0 103.9,134.0 104.4,134.0 105.0,134.0 105.5,134.0 106.0,134.0 106.5,134.0 107.0,134.0 107.6,134.0 108.1,134.0 108.6,134.0 109.1,134.0 109.6,134.0 110.2,134.0 110.7,134.0 111.2,134.0 111.7,134.0 112.2,134.0 112.8,134.0 113.3,134.0 113.8,134.0 114.3,134.0 114.8,134.0 115.4,134.0 115.9,134.0 116.4,134.0 116.9,134.0 117.4,134.0 118.0,134.0 118.5,134.0 119.0,134.0 119.5,134.0 120.0,134.0 120.6,134.0 121.1,134.0 121.6,134.0 122.1,134.0 122.6,134.0 123.2,134.0 123.7,134.0 124.2,134.0 124.7,134.0 125.2,134.0 125.8,134.0 126.3,134.0 126.8,134.0 127.3,134.0 127.8,134.0 128.4,134.0 128.9,134.0 129.4,134.0 129.9,134.0 130.4,134.0 131.0,134.0 131.5,134.0 132.0,134.0 132.5,134.0 133.0,134.0 133.6,134.0 134.1,134.0 134.6,134.0 135.1,134.0 135.6,134.0 136.2,134.0 136.7,134.0 137.2,134.0 137.7,134.0 138.2,134.0 138.8,134.0 139.3,134.0 139.8,134.0 140.3,134.0 140.8,134.0 141.4,134.0 141.9,134.0 142.4,134.0 142.9,134.0 143.4,134.0 144.0,134.0 144.5,134.0 145.0,134.0 145.5,134.0 146.0,134.0 146.6,134.0 147.1,134.0 147.6,134.0 148.1,134.0 148.6,134.0 149.2,134.0 149.7,134.0 150.2,134.0 150.7,134.0 151.2,134.0 151.8,134.0 152.3,134.0 152.8,134.0 153.3,134.0 153.8,134.0 154.4,134.0 154.9,134.0 155.4,134.0 155.9,134.0 156.4,134.0 157.0,134.0 157.5,134.0 158.0,134.0 158.5,134.0 159.0,134.0 159.6,134.0 160.1,134.0 160.6,134.0 161.1,134.0 161.6,134.0 162.2,134.0 162.7,134.0 163.2,134.0 163.7,134.0 164.2,134.0 164.8,134.0 165.3,134.0 165.8,134.0 166.3,134.0 166.8,134.0 167.4,134.0 167.9,134.0 168.4,134.0 168.9,134.0 169.4,134.0 170.0,134.0 170.5,134.0 171.0,134.0 171.5,134.0 172.0,134.0 172.6,134.0 173.1,134.0 173.6,134.0 174.1,134.0 174.6,134.0 175.2,134.0 175.7,134.0 176.2,134.0 176.7,134.0 177.2,134.0 177.8,134.0 178.3,134.0 178.8,134.0 179.3,134.0 179.8,134.0 180.4,134.0 180.9,134.0 181.4,134.0 181.9,134.0 182.4,134.0 183.0,134.0 183.5,134.0 184.0,134.0 184.5,134.0 185.0,134.0 185.6,134.0 186.1,134.0 186.6,134.0 187.1,134.0 187.6,134.0 188.2,134.0 188.7,134.0 189.2,134.0 189.7,134.0 190.2,134.0 190.8,134.0 191.3,134.0 191.8,134.0 192.3,134.0 192.8,134.0 193.4,134.0 193.9,134.0 194.4,134.0 194.9,134.0 195.4,134.0 196.0,134.0 196.5,134.0 197.0,134.0 197.5,134.0 198.0,134.0 198.6,134.0 199.1,134.0 199.6,134.0 200.1,134.0 200.6,134.0 201.2,134.0 201.7,134.0 202.2,134.0 202.7,134.0 203.2,134.0 203.8,134.0 204.3,134.0 204.8,134.0 205.3,134.0 205.8,134.0 206.4,134.0 206.9,134.0 207.4,134.0 207.9,134.0 208.4,134.0 209.0,134.0 209.5,134.0 210.0,134.0 210.5,134.0 211.0,134.0 211.6,134.0 212.1,134.0 212.6,134.0 213.1,134.0 213.6,134.0 214.2,134.0 214.7,134.0 215.2,134.0 215.7,134.0 216.2,134.0 216.8,134.0 217.3,134.0 217.8,134.0 218.3,134.0 218.8,134.0 219.4,134.0 219.9,134.0 220.4,134.0 220.9,134.0 221.4,134.0 222.0,134.0 222.5,134.0 223.0,134.0 223.5,134.0 224.0,134.0 224.6,134.0 225.1,134.0 225.6,134.0 226.1,134.0 226.6,134.0 227.2,134.0 227.7,134.0 228.2,134.0 228.7,134.0 229.2,134.0 229.8,134.0 230.3,134.0 230.8,134.0 231.3,134.0 231.8,134.0 232.4,134.0 232.9,134.0 233.4,134.0 233.9,134.0 234.4,134.0 235.0,134.0 235.5,134.0 236.0,134.0 236.5,134.0 237.0,134.0 237.6,134.0 238.1,134.0 238.6,134.0 239.1,134.0 239.6,134.0 240.2,134.0 240.7,134.0 241.2,134.0 241.7,134.0 242.2,134.0 242.8,134.0 243.3,134.0 243.8,134.0 244.3,134.0 244.8,134.0 245.4,134.0 245.9,134.0 246.4,134.0 246.9,134.0 247.4,134.0 248.0,134.0 248.5,134.0 249.0,134.0 249.5,134.0 250.0,134.0 250.6,134.0 251.1,134.0 251.6,134.0 252.1,134.0 252.6,134.0 253.2,134.0 253.7,134.0 254.2,134.0 254.7,134.0 255.2,134.0 255.8,134.0 256.3,134.0 256.8,134.0 257.3,134.0 257.8,134.0 258.4,134.0 258.9,134.0 259.4,134.0 259.9,134.0 260.4,134.0 261.0,134.0 261.5,134.0 262.0,134.0 L 262.0,150 Z" /></g>
        <polyline points="54.0,134.0 54.5,134.0 55.0,134.0 55.6,134.0 56.1,134.0 56.6,134.0 57.1,134.0 57.6,134.0 58.2,134.0 58.7,134.0 59.2,134.0 59.7,134.0 60.2,134.0 60.8,134.0 61.3,134.0 61.8,134.0 62.3,134.0 62.8,134.0 63.4,134.0 63.9,134.0 64.4,134.0 64.9,134.0 65.4,134.0 66.0,134.0 66.5,134.0 67.0,134.0 67.5,134.0 68.0,134.0 68.6,134.0 69.1,134.0 69.6,134.0 70.1,134.0 70.6,134.0 71.2,134.0 71.7,134.0 72.2,134.0 72.7,134.0 73.2,134.0 73.8,134.0 74.3,134.0 74.8,134.0 75.3,134.0 75.8,134.0 76.4,134.0 76.9,134.0 77.4,134.0 77.9,134.0 78.4,134.0 79.0,134.0 79.5,134.0 80.0,134.0 80.5,134.0 81.0,134.0 81.6,134.0 82.1,134.0 82.6,134.0 83.1,134.0 83.6,134.0 84.2,134.0 84.7,134.0 85.2,134.0 85.7,134.0 86.2,134.0 86.8,134.0 87.3,134.0 87.8,134.0 88.3,134.0 88.8,134.0 89.4,134.0 89.9,134.0 90.4,134.0 90.9,134.0 91.4,134.0 92.0,134.0 92.5,134.0 93.0,134.0 93.5,134.0 94.0,134.0 94.6,134.0 95.1,134.0 95.6,134.0 96.1,134.0 96.6,134.0 97.2,134.0 97.7,134.0 98.2,134.0 98.7,134.0 99.2,134.0 99.8,134.0 100.3,134.0 100.8,134.0 101.3,134.0 101.8,134.0 102.4,134.0 102.9,134.0 103.4,134.0 103.9,134.0 104.4,134.0 105.0,134.0 105.5,134.0 106.0,134.0 106.5,134.0 107.0,134.0 107.6,134.0 108.1,134.0 108.6,134.0 109.1,134.0 109.6,134.0 110.2,134.0 110.7,134.0 111.2,134.0 111.7,134.0 112.2,134.0 112.8,134.0 113.3,134.0 113.8,134.0 114.3,134.0 114.8,134.0 115.4,134.0 115.9,134.0 116.4,134.0 116.9,134.0 117.4,134.0 118.0,134.0 118.5,134.0 119.0,134.0 119.5,134.0 120.0,134.0 120.6,134.0 121.1,134.0 121.6,134.0 122.1,134.0 122.6,134.0 123.2,134.0 123.7,134.0 124.2,134.0 124.7,134.0 125.2,134.0 125.8,134.0 126.3,134.0 126.8,134.0 127.3,134.0 127.8,134.0 128.4,134.0 128.9,134.0 129.4,134.0 129.9,134.0 130.4,134.0 131.0,134.0 131.5,134.0 132.0,134.0 132.5,134.0 133.0,134.0 133.6,134.0 134.1,134.0 134.6,134.0 135.1,134.0 135.6,134.0 136.2,134.0 136.7,134.0 137.2,134.0 137.7,134.0 138.2,134.0 138.8,134.0 139.3,134.0 139.8,134.0 140.3,134.0 140.8,134.0 141.4,134.0 141.9,134.0 142.4,134.0 142.9,134.0 143.4,134.0 144.0,134.0 144.5,134.0 145.0,134.0 145.5,134.0 146.0,134.0 146.6,134.0 147.1,134.0 147.6,134.0 148.1,134.0 148.6,134.0 149.2,134.0 149.7,134.0 150.2,134.0 150.7,134.0 151.2,134.0 151.8,134.0 152.3,134.0 152.8,134.0 153.3,134.0 153.8,134.0 154.4,134.0 154.9,134.0 155.4,134.0 155.9,134.0 156.4,134.0 157.0,134.0 157.5,134.0 158.0,134.0 158.5,134.0 159.0,134.0 159.6,134.0 160.1,134.0 160.6,134.0 161.1,134.0 161.6,134.0 162.2,134.0 162.7,134.0 163.2,134.0 163.7,134.0 164.2,134.0 164.8,134.0 165.3,134.0 165.8,134.0 166.3,134.0 166.8,134.0 167.4,134.0 167.9,134.0 168.4,134.0 168.9,134.0 169.4,134.0 170.0,134.0 170.5,134.0 171.0,134.0 171.5,134.0 172.0,134.0 172.6,134.0 173.1,134.0 173.6,134.0 174.1,134.0 174.6,134.0 175.2,134.0 175.7,134.0 176.2,134.0 176.7,134.0 177.2,134.0 177.8,134.0 178.3,134.0 178.8,134.0 179.3,134.0 179.8,134.0 180.4,134.0 180.9,134.0 181.4,134.0 181.9,134.0 182.4,134.0 183.0,134.0 183.5,134.0 184.0,134.0 184.5,134.0 185.0,134.0 185.6,134.0 186.1,134.0 186.6,134.0 187.1,134.0 187.6,134.0 188.2,134.0 188.7,134.0 189.2,134.0 189.7,134.0 190.2,134.0 190.8,134.0 191.3,134.0 191.8,134.0 192.3,134.0 192.8,134.0 193.4,134.0 193.9,134.0 194.4,134.0 194.9,134.0 195.4,134.0 196.0,134.0 196.5,134.0 197.0,134.0 197.5,134.0 198.0,134.0 198.6,134.0 199.1,134.0 199.6,134.0 200.1,134.0 200.6,134.0 201.2,134.0 201.7,134.0 202.2,134.0 202.7,134.0 203.2,134.0 203.8,134.0 204.3,134.0 204.8,134.0 205.3,134.0 205.8,134.0 206.4,134.0 206.9,134.0 207.4,134.0 207.9,134.0 208.4,134.0 209.0,134.0 209.5,134.0 210.0,134.0 210.5,134.0 211.0,134.0 211.6,134.0 212.1,134.0 212.6,134.0 213.1,134.0 213.6,134.0 214.2,134.0 214.7,134.0 215.2,134.0 215.7,134.0 216.2,134.0 216.8,134.0 217.3,134.0 217.8,134.0 218.3,134.0 218.8,134.0 219.4,134.0 219.9,134.0 220.4,134.0 220.9,134.0 221.4,134.0 222.0,134.0 222.5,134.0 223.0,134.0 223.5,134.0 224.0,134.0 224.6,134.0 225.1,134.0 225.6,134.0 226.1,134.0 226.6,134.0 227.2,134.0 227.7,134.0 228.2,134.0 228.7,134.0 229.2,134.0 229.8,134.0 230.3,134.0 230.8,134.0 231.3,134.0 231.8,134.0 232.4,134.0 232.9,134.0 233.4,134.0 233.9,134.0 234.4,134.0 235.0,134.0 235.5,134.0 236.0,134.0 236.5,134.0 237.0,134.0 237.6,134.0 238.1,134.0 238.6,134.0 239.1,134.0 239.6,134.0 240.2,134.0 240.7,134.0 241.2,134.0 241.7,134.0 242.2,134.0 242.8,134.0 243.3,134.0 243.8,134.0 244.3,134.0 244.8,134.0 245.4,134.0 245.9,134.0 246.4,134.0 246.9,134.0 247.4,134.0 248.0,134.0 248.5,134.0 249.0,134.0 249.5,134.0 250.0,134.0 250.6,134.0 251.1,134.0 251.6,134.0 252.1,134.0 252.6,134.0 253.2,134.0 253.7,134.0 254.2,134.0 254.7,134.0 255.2,134.0 255.8,134.0 256.3,134.0 256.8,134.0 257.3,134.0 257.8,134.0 258.4,134.0 258.9,134.0 259.4,134.0 259.9,134.0 260.4,134.0 261.0,134.0 261.5,134.0 262.0,134.0" fill="none" stroke="#1f2430" stroke-width="1" />
        <text x="49.0" y="153" font-size="8.5" fill="#666" text-anchor="end">k = 0</text>
        <text x="267.0" y="153" font-size="8.5" fill="#1f2430">∫ = 2π</text>
        <text x="158" y="128" font-size="8.5" fill="#1565c0" text-anchor="middle">area 2π</text>
        <g font-size="8.5" fill="#666" text-anchor="middle">
          <text x="54.0" y="180">−π</text><text x="158" y="180">0</text><text x="262.0" y="180">π</text>
        </g>
        <g font-size="9" fill="#444">
          <text x="12" y="203">[−π, π] holds whole periods, so + and − lobes cancel exactly</text>
          <text x="12" y="215">only k = 0 escapes cancellation: that is the 2π to normalise</text>
        </g>
      </svg>
    </div>
  </div>
  <figcaption>Orthonormality of $e^{inx}/\sqrt{2\pi}$ is one computation seen two ways. The inner product $\langle e^{inx}, e^{imx} \rangle = \int_{-\pi}^{\pi} e^{i(n-m)x}\,dx$ depends only on $k = n - m$. <strong>Left.</strong> As $x$ runs through $[-\pi, \pi]$, the point $e^{ikx}$ runs $\lvert k \rvert$ full laps around the unit circle, and the integral is $2\pi$ times its average position. For $k \ne 0$ the average over full laps is the centre, so the integral is $0$; for $k = 0$ the point never leaves $1$, so the integral is $2\pi$. <strong>Right.</strong> The same fact for the real and imaginary parts: $\cos kx$ and $\sin kx$ complete whole periods on $[-\pi, \pi]$, so their positive and negative lobes cancel exactly, while the constant $1$ has area $2\pi$. Dividing by $\sqrt{2\pi}$ turns $\lVert e^{inx} \rVert^2 = 2\pi$ into $1$, so the family consists of mutually perpendicular unit vectors in $L^2([-\pi, \pi])$. Nothing here is special about exponentials beyond their period: on an interval whose length is not a multiple of $2\pi$ the lobes would not cancel and the same functions would fail to be orthogonal. Orthogonality is what lets each Fourier coefficient of Definition 167 be read off independently of the others, and it is why Parseval's identity (Theorem 164) turns $\lVert f \rVert^2$ into a sum of squares.</figcaption>
</figure>

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation of Fourier Series)</span></p>

The motivating question for Fourier when first studying these objects was **whether or not all continuous functions could be expanded in this Fourier series manner**. Trying to study things on a pointwise convergence level is diﬃcult, but the space we should really be viewing this setup within is the $L^2$ space, and there we’ll be able to get some results. The problem we’re trying to resolve is as follows:

</div>
<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem 169: Is the Fourier Basis Complete in $L^2$</span></p>

Does the convergence (in $L^2$ norm) $\sum_{n=1}^{\infty} \hat{f} e^{inx} \to f$ hold for all $f \in L^2([-\pi, \pi])$? In other words, does

$$\lVert f - S_N f \rVert_2 = \left( \int_{-\pi}^{\pi} \lvert f(x) - S_N f(x) \rvert^2\,dx \right)^{1/2}$$

converge to $0$ as $N \to \infty$?

</div>

This is equivalent to asking whether $\left\lbrace \frac{e^{inx}}{\sqrt{2\pi}} \right\rbrace$ is a **maximal** orthonormal subset in $L^2([-\pi, \pi])$, i.e., whether $\hat{f}(n) = 0$ for all $n \in \mathbb{Z}$ implies $f = 0$. We already know that if we have an orthonormal basis, then we can indeed make this infinite expansion for any element of the space $L^2$. So this rephrasing in terms of the language of Hilbert spaces will help us out here (and we should remember that we require the completeness of $L^2$ to get to this rephrased problem statement). The answer turns out to be **yes**, but it takes some work to prove.

### The Dirichlet Kernel: Representing Fourier partial sums as convolution operators

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 170</span><span class="math-callout__name">(Representing Fourier partial sums as convolution operators: Dirichlet Kernel Representation)</span></p>

For all $f \in L^2([-\pi, \pi])$ and all $N \in \mathbb{Z}\_{\ge 0}$, we have 

$$S_N f(x) = \int_{-\pi}^{\pi} D_N(x - t) f(t)\,dt,$$

where

$$D_N(x) = \begin{cases} \frac{2N+1}{2\pi} & x = 0, \\[6pt] \frac{\sin\left((N + \frac{1}{2})x\right)}{2\pi \sin \frac{x}{2}} & x \ne 0. \end{cases}$$

The function $D_N$ is called the **Dirichlet kernel**. We can check that the function $D_N$ is continuous (and in fact smooth).

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reformulating the strategy)</span></p>

We’ve rephrase our convergence of Fourier series to the statement that “if the Fourier coeﬃcients are all zero, then the function is zero.” And the direction we’re going with this definition here is that if we can show the partial sums $S_Nf$ converge to $f$, then $f$ must be a sum of zeros, but trying to do this with SN directly is our original problem statement! So this “averaged” Cesaro-Fourier mean will be an easier thing to work with, and we’ll try to show that $\sigma_N f \to f$ instead.

So, the strategy is to show $\lVert \sigma_N f - f \rVert_2 \to 0$ as $N \to \infty$, which then implies the desired convergence of Fourier series (because if all Fourier coefficients are zero, then $\sigma_N f = 0$ for all $N$, so $f = 0$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 172</span></p>

We know from real analysis that the Cesaro means of a sequence of real numbers behave better than the original sequence, but we do not lose any information. In particular, sequences like $\lbrace 1, -1, 1, -1, \dots \rbrace$ do not converge, but their Cesaro means do.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 173</span><span class="math-callout__name">(Fejér Kernel)</span></p>

For all $f \in L^2([-\pi, \pi])$, we have

$$\sigma_N f(x) = \int_{-\pi}^{\pi} K_N(x - t) f(t)\,dt, \quad K_N(x) = \begin{cases} \frac{N+1}{2\pi} & x = 0, \\[6pt] \frac{1}{2\pi(N+1)} \left( \frac{\sin\left(\frac{N+1}{2} x\right)}{\sin \frac{x}{2}} \right)^2 & \text{otherwise}. \end{cases}$$

The function $K_N(x)$ is called the **Fejér kernel**. It has the following properties: 
* **(1)** $K_N(x) \ge 0$ and $K_N(x) = K_N(-x)$ for all $x$, 
* **(2)** $K_N$ is periodic with period $2\pi$, 
* **(3)** $\int_{-\infty}^{\infty} K_N(t)\,dt = 1$, 
* **(4)** for any $\delta \in (0, \pi)$ and for all $\delta \le \lvert x \rvert \le \pi$, we have $\lvert K_N(x) \rvert \le \frac{1}{2\pi(N+1) \sin^2 \frac{\delta}{2}}$.

</div>

The Fejér kernel grows more and more concentrated at the origin as $N \to \infty$, but the area under the curve is always $1$ (like the Dirac delta function). The key difference from the Dirichlet kernel is that the Fejér kernel is **nonnegative**.

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">Dirichlet kernel D₅</div>
      <svg viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Dirichlet kernel D_5 over [-pi, pi]">
        <g stroke="#e8e8e8" stroke-width="0.5">
          <line x1="3"   y1="80"  x2="317" y2="80" />
          <line x1="3"   y1="105" x2="317" y2="105" />
          <line x1="3"   y1="155" x2="317" y2="155" />
        </g>
        <line x1="3"   y1="130" x2="317" y2="130" stroke="#444" stroke-width="1" />
        <line x1="160" y1="20"  x2="160" y2="180" stroke="#444" stroke-width="1" />
        <polygon points="317,130 311,127 311,133" fill="#444" />
        <polygon points="160,20 157,26 163,26" fill="#444" />
        <polyline points="3,138 17,130 32,122 46,130 60,139 74,130 89,118 103,130 117,149 131,130 146,74 160,42 174,74 189,130 203,149 217,130 231,118 246,130 260,139 274,130 288,122 303,130 317,138"
                  stroke="#2c4994" stroke-width="2" fill="none" />
        <g fill="#666" font-size="10">
          <text x="0"   y="194">−π</text>
          <text x="153" y="194">0</text>
          <text x="313" y="194">π</text>
          <text x="166" y="46">2N+1⁄2π</text>
        </g>
      </svg>
    </div>
    <div>
      <div class="panel-label">Fejér kernel K₅</div>
      <svg viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Fejér kernel K_5 over [-pi, pi]">
        <g stroke="#e8e8e8" stroke-width="0.5">
          <line x1="3"   y1="105" x2="317" y2="105" />
          <line x1="3"   y1="80"  x2="317" y2="80" />
          <line x1="3"   y1="55"  x2="317" y2="55" />
        </g>
        <line x1="3"   y1="170" x2="317" y2="170" stroke="#444" stroke-width="1" />
        <line x1="160" y1="20"  x2="160" y2="180" stroke="#444" stroke-width="1" />
        <polygon points="317,170 311,167 311,173" fill="#444" />
        <polygon points="160,20 157,26 163,26" fill="#444" />
        <polygon points="3,170 16,169 29,166 42,168 55,170 68,167 81,163 95,167 108,170 121,158 134,119 147,70 160,46 173,70 186,119 199,158 212,170 225,167 238,163 251,167 265,170 278,168 291,166 304,169 317,170 317,170 3,170"
                 fill="rgba(60,120,40,0.10)" stroke="none" />
        <polyline points="3,170 16,169 29,166 42,168 55,170 68,167 81,163 95,167 108,170 121,158 134,119 147,70 160,46 173,70 186,119 199,158 212,170 225,167 238,163 251,167 265,170 278,168 291,166 304,169 317,170"
                  stroke="#3d7a26" stroke-width="2" fill="none" />
        <g fill="#666" font-size="10">
          <text x="0"   y="194">−π</text>
          <text x="153" y="194">0</text>
          <text x="313" y="194">π</text>
          <text x="166" y="50">N+1⁄2π</text>
        </g>
      </svg>
    </div>
  </div>
  <figcaption>Both kernels deliver $\sigma_N f$ or $S_N f$ as a convolution against $f$, but with very different behavior. $D_N$ oscillates with sign changes and tail $L^1$-norm $\sim \log N$, so it is <em>not</em> an approximate identity; $K_N$ is nonnegative, integrates to $1$, and concentrates at the origin — exactly the conditions needed for $\sigma_N f \to f$ uniformly on continuous periodic data (Fejér's theorem).</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why we believe Cesaro means converge to $f$)</span></p>

The reason we might believe that these Cesaro means converge to $f$ is that

$$\sigma_N f(x) = \int_\pi^{-\pi} K_N(x−t)f(t)dt,$$

and $K_N$ is very sharply peaked around $t=x$, so as $N$ gets larger and larger, the main contribution to the integral comes from $f(x) \approx f(t)$ if $f$ is well-behaved enough. So then we end up with

$$\approx f(x) \int_\pi^{-\pi} K_N(x−t)dt= f(x)\cdot 1,$$

since $k_N$ evaluates to the same over any interval of length $2\pi$ by periodicity. So that’s a heuristic motivation for working with the Cesaro means here! (Some of these properties also applied when we did a similar procedure with our partial sums $S_Nf(x)$, but the Dirichlet kernel is not nonnegative – that diﬀerence actually makes a big diﬀerence in the final proof.)

</div>

### Fejér's Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 174</span><span class="math-callout__name">(Fejér's Uniform Convergence Theorem)</span></p>

Let $f \in C([-\pi, \pi])$ be $2\pi$-periodic (so $f(-\pi) = f(\pi)$). Then $\sigma_N f \to f$ uniformly on $[-\pi, \pi]$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Extend $f$ to all of $\mathbb{R}$ by periodicity. Since $f$ is $2\pi$-periodic and continuous, it is uniformly continuous and bounded ($\lVert f \rVert_\infty < \infty$).</p>
    <p>For any $\varepsilon > 0$, by uniform continuity there exists $\delta > 0$ so that $\lvert y - z \rvert < \delta$ implies $\lvert f(y) - f(z) \rvert < \varepsilon / 2$. Choose $M \in \mathbb{N}$ so that $\frac{2 \lVert f \rVert_\infty}{(N+1) \sin^2(\delta/2)} < \varepsilon / 2$ for all $N \ge M$. If $\sin^2(\delta/2)$ is zero, we take $\sigma$ smaller to make $\sin^2(\delta/2)\gt 0$.</p>
    <p>Using $\int_{-\pi}^{\pi} K_N(t)\,dt = 1$ and $K_N \ge 0$:</p>
    $$\lvert \sigma_N f(x) - f(x) \rvert = \left\lvert \int_{-\pi}^{\pi} K_N(t)(f(x-t) - f(x))\,dt \right\rvert \le \int_{-\pi}^{\pi} K_N(t) \lvert f(x-t) - f(x) \rvert\,dt.$$
    <p>Split into $\lvert t \rvert \le \delta$ (where $\lvert f(x-t) - f(x) \rvert < \varepsilon/2$) and $\delta \le \lvert t \rvert \le \pi$ (where $K_N(t) \le \frac{1}{2\pi(N+1)\sin^2(\delta/2)}$ and $\lvert f(x-t) - f(x) \rvert \le 2\lVert f \rVert_\infty$). Bounding each integral by the total $K_N$-mass gives $< \varepsilon/2 + \varepsilon/2 = \varepsilon$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 175</span><span class="math-callout__name">(Why this proof fails for Dirichlet kernel)</span></p>

The same proof works if instead of $K_N(x) \ge 0$ we only know $\sup_N \int_{-\pi}^{\pi} \lvert K_N(x) \rvert\,dx < \infty$. But for the Dirichlet kernel, $\int_{-\pi}^{\pi} \lvert D_N(x) \rvert\,dx \sim \log N$, so this condition is not satisfied — having "almost all" of the properties is not enough.

</div>

### Convergence of Cesaro Means in $L^2$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 176</span><span class="math-callout__name">($L^2$-Contractivity of Fejér Means)</span></p>
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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Density Step)</span></p>

The proof of Proposition 176 and Theorem 177 used and will use that continuous $2\pi$-periodic functions are dense in $L^2([-\pi, \pi])$. One way to see it: by dominated convergence,

$$\lim_{\delta \to 0}\Big(\int_{-\pi}^{-\pi+\delta} \lvert f \rvert^2 + \int_{\pi-\delta}^{\pi} \lvert f \rvert^2\Big) = 0,$$

so functions vanishing near the endpoints are dense; these are approximated by step functions vanishing near the endpoints, and each such step function is an $L^2$-limit of continuous functions vanishing near the endpoints — which are automatically periodic.

</div>

### Convergence of Fourier Series in $L^2$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 177</span><span class="math-callout__name">($L^2$-Convergence of Fejér Means)</span></p>

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Completeness of the Fourier Basis Computes $\zeta(2)$ and $\zeta(4)$)</span></p>

Fourier series on $[0, 2\pi]$ work the same way (translate by $\pi$). For $t \in (0, 2\pi)$ let $f\_t = \mathbf{1}\_{[0, t)}$. Its Fourier coefficients are $\hat f\_t(0) = \frac{t}{2\pi}$ and, for $n \ne 0$,

$$\hat f_t(n) = \frac{1}{2\pi}\int_0^t e^{-inx}\,dx = \frac{1 - e^{-int}}{2\pi i n}, \qquad \lvert \hat f_t(n) \rvert^2 = \frac{2 - 2\cos(nt)}{4\pi^2 n^2}.$$

Completeness of the Fourier basis means Parseval holds, $\lVert f\_t \rVert\_2^2 = 2\pi\sum\_n \lvert \hat f\_t(n) \rvert^2$, i.e. $t = \frac{t^2}{2\pi} + \frac{2}{\pi}\sum\_{n \ge 1}\frac{1 - \cos(nt)}{n^2}$:

$$\sum_{n=1}^\infty \frac{1 - \cos(nt)}{n^2} = \frac{\pi t}{2} - \frac{t^2}{4}, \qquad t \in [0, 2\pi].$$

* At $t = \pi$: $\sum\_{n \text{ odd}} \frac{2}{n^2} = \frac{\pi^2}{4}$, so $\sum\_{n \text{ odd}} \frac{1}{n^2} = \frac{\pi^2}{8}$ and hence $\displaystyle\sum\_{n \ge 1} \frac{1}{n^2} = \frac{4}{3}\cdot\frac{\pi^2}{8} = \frac{\pi^2}{6}$.
* So $g(t) := \sum\_{n \ge 1} \frac{\cos(nt)}{n^2} = \frac{\pi^2}{6} - \frac{\pi t}{2} + \frac{t^2}{4}$ on $[0, 2\pi]$. This is itself a Fourier series with coefficients $\frac{1}{2n^2}$ at $\pm n$; applying Parseval once more,

  $$\int_0^{2\pi}\Big(\frac{\pi^2}{6} - \frac{\pi t}{2} + \frac{t^2}{4}\Big)^2 dt = 2\pi \sum_{n \ne 0} \frac{1}{4n^4} = \pi \sum_{n \ge 1}\frac{1}{n^4}.$$

  The left side equals $\frac{\pi^5}{90}$, giving $\displaystyle\sum\_{n \ge 1}\frac{1}{n^4} = \frac{\pi^4}{90}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A Sine Basis by Odd Extension)</span></p>

The functions $\frac{1}{\sqrt{\pi}}\sin\big(\frac{kx}{2}\big)$, $k \in \mathbb{N}$, form an orthonormal basis of $L^2(0, 2\pi)$.

*Why:* rescaling Theorem 177 from $[-\pi, \pi]$ to $[-2\pi, 2\pi]$ shows that $\frac{1}{\sqrt{4\pi}}e^{ikx/2}$, $k \in \mathbb{Z}$, is an orthonormal basis of $L^2(-2\pi, 2\pi)$. Extend $f \in L^2(0, 2\pi)$ to an **odd** function $\tilde f$ on $(-2\pi, 2\pi)$. Its expansion only involves $e^{ikx/2} - e^{-ikx/2} = 2i\sin(kx/2)$, so $\tilde f$ is an $L^2$-limit of finite combinations of $\sin(kx/2)$; restricting to $(0, 2\pi)$ gives completeness. Orthonormality: $\int_0^{2\pi}\sin\big(\frac{kx}{2}\big)\sin\big(\frac{jx}{2}\big)\,dx = \pi\,\delta\_{kj}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">($L^1$ Functions with Square-Summable Fourier Coefficients Are in $L^2$)</span></p>

Let $f \in L^1([-\pi, \pi])$ have Fourier coefficients 

$$\hat f(n) = \frac{1}{2\pi}\int\_{-\pi}^{\pi} f(x)e^{-inx}\,dx$$

with $\sum\_{n \in \mathbb{Z}} \lvert \hat f(n) \rvert^2 < \infty$. Then $f \in L^2([-\pi, \pi])$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

The coefficients exist since $f e^{-inx} \in L^1$. By square-summability and completeness of $L^2$, $g := \sum\_n \hat f(n) e^{inx}$ converges in $L^2$, and $\hat g(n) = \hat f(n)$ by continuity of the inner product. Since $L^2([-\pi, \pi]) \subset L^1([-\pi, \pi])$ (Cauchy–Schwarz on a finite interval), $h := f - g \in L^1$ has **all Fourier coefficients zero**. It remains to show that this forces $h = 0$ a.e. — the $L^1$ version of completeness.

**Step 1: $\int h\varphi = 0$ for every continuous $2\pi$-periodic $\varphi$.** This holds for trigonometric polynomials, hence for the Fejér means $\sigma\_N \varphi$, and $\lvert \int h(\varphi - \sigma\_N \varphi) \rvert \le \lVert h \rVert\_1 \lVert \varphi - \sigma\_N \varphi \rVert\_\infty \to 0$ by Fejér's theorem.

**Step 2: $\int h s = 0$ for every step function $s$.** The indicator of an interval is a pointwise a.e. limit of continuous periodic functions $\varphi\_n$ with $\lvert \varphi\_n \rvert \le 1$ (piecewise linear ramps), and $h\varphi\_n \to h\mathbf{1}\_I$ by dominated convergence (dominated by $\lvert h \rvert$). Take linear combinations.

**Step 3: $\int \lvert h \rvert = 0$.** Choose step functions $h\_n \to h$ in $L^1$, and let $s\_n = \overline{h\_n}/\lvert h\_n \rvert$ where $h\_n \ne 0$ and $s\_n = 0$ elsewhere — again step functions, with $\lvert s\_n \rvert \le 1$ and $s\_n h\_n = \lvert h\_n \rvert$. From the identity

$$\lvert h \rvert = \big(\lvert h \rvert - \lvert h_n \rvert\big) + s_n(h_n - h) + s_n h$$

and Step 2 (the last term integrates to $0$),

$$\int \lvert h \rvert \le \int \big\lvert \lvert h \rvert - \lvert h_n \rvert \big\rvert + \int \lvert h_n - h \rvert \le 2\lVert h - h_n \rVert_1 \to 0.$$

So $h = 0$ a.e., i.e. $f = g \in L^2$. $\square$

</details>
</div>

### Periodic Sobolev Spaces

Below is a **Fourier (or spectral) definition of periodic Sobolev spaces**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sobolev Space of Order $s$ on the Torus)</span></p>

Let $s\ge 0$, and let $\mathbb T=\mathbb R/(2\pi\mathbb Z)$ denote the one-dimensional torus.

The **Sobolev space of order $s$** is defined by

$$H^s(\mathbb T) := \left\lbrace f\in L^2(\mathbb T): \sum_{n\in\mathbb Z}(1+|n|^2)^s |\widehat f(n)|^2<\infty \right\rbrace,$$

where the Fourier coefficients are

$$\widehat f(n) = \frac{1}{2\pi} \int_{-\pi}^{\pi}f(x)e^{-inx}\,dx.$$

The associated **Sobolev norm** is

$$\|f\|_{H^s(\mathbb T)} = \left( \sum_{n\in\mathbb Z} (1+|n|^2)^s|\widehat f(n)|^2 \right)^{1/2}.$$

The corresponding inner product is

$$\langle f,g\rangle_{H^s(\mathbb T)} = \sum_{n\in\mathbb Z} (1+|n|^2)^s \widehat f(n)\overline{\widehat g(n)}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Useful observation about Fourier (or spectral) definition of periodic Sobolev spaces)</span></p>

With your Fourier normalization, Parseval gives

$$\|f\|_{H^0(\mathbb T)}^2 =\sum_{n\in\mathbb Z}|\widehat f(n)|^2 =\frac{1}{2\pi}\|f\|_{L^2(\mathbb T)}^2.$$

Thus $H^0(\mathbb T)=L^2(\mathbb T)$ as sets, although the norms differ by a constant factor.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relationship to the classical definition of Sobolev Spaces)</span></p>

For an integer $k\ge 0$, the **classical Sobolev** space is defined using **weak derivatives**:

$$H^k(\mathbb T) = \left\lbrace f\in L^2(\mathbb T): D^jf\in L^2(\mathbb T),\quad 0\le j\le k \right\rbrace,$$

where derivatives are understood periodically. The key connection is that differentiation becomes multiplication by $in$ in Fourier space:

$$\widehat{D^jf}(n)=(in)^j\widehat f(n).$$

Consequently, Parseval's identity gives

$$\|D^jf\|_2^2 = 2\pi\sum_{n\in\mathbb Z}n^{2j}|\widehat f(n)|^2,$$

using the conventional normalization $\widehat f(n)=\frac1{2\pi}\int_{-\pi}^{\pi}f(x)e^{-inx}\,dx$.

Thus the Fourier definition and weak-derivative definition are equivalent for integer orders. For noninteger $s$, the Fourier definition provides a natural extension to fractional Sobolev spaces.

</div>

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

<figure class="math-figure">
  <svg viewBox="0 0 360 240" xmlns="http://www.w3.org/2000/svg" width="380" aria-label="Length minimizer in a closed convex set">
    <g stroke="#888" stroke-width="0.6" fill="none">
      <circle cx="100" cy="160" r="40" stroke-dasharray="3,3" />
      <circle cx="100" cy="160" r="80" stroke-dasharray="3,3" />
      <circle cx="100" cy="160" r="120" stroke-dasharray="3,3" />
    </g>
    <path d="M 220,40 Q 290,60 305,140 Q 295,210 215,220 Q 165,210 175,150 Q 188,55 220,40 Z"
          fill="rgba(44,73,148,0.10)" stroke="#2c4994" stroke-width="2" />
    <text x="240" y="118" font-size="14" font-weight="600" fill="#2c4994">C</text>
    <line x1="100" y1="160" x2="180" y2="143" stroke="#d65336" stroke-width="2.5" />
    <circle cx="100" cy="160" r="3" fill="#222" />
    <circle cx="180" cy="143" r="4" fill="#d65336" stroke="#fff" stroke-width="1" />
    <text x="80"  y="178" font-size="12" fill="#222">0</text>
    <text x="187" y="135" font-size="13" font-weight="600" fill="#d65336">v</text>
    <text x="118" y="148" font-size="11" fill="#666" font-style="italic">d</text>
    <g stroke="#3d7a26" stroke-width="1.6" fill="none" stroke-dasharray="2,3">
      <line x1="200" y1="80" x2="260" y2="200" />
    </g>
    <text x="208" y="78" font-size="11" fill="#3d7a26" font-style="italic">tu₁ + (1−t)u₂ ∈ C</text>
  </svg>
  <figcaption>The closed convex set $C$ is "approached" from outside by expanding norm balls; the smallest one that touches $C$ does so at exactly one point $v$. Convexity is what makes the minimizer unique: the parallelogram law together with the midpoint $(u_1+u_2)/2 \in C$ forces $u_1 = u_2$ whenever both achieve the infimum.</figcaption>
</figure>

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

<figure class="math-figure">
  <svg viewBox="0 0 380 240" xmlns="http://www.w3.org/2000/svg" width="400" aria-label="Orthogonal decomposition u = w + w_perp">
    <polygon points="40,180 340,140 340,200 40,240" fill="rgba(44,73,148,0.10)" stroke="#2c4994" stroke-width="1.5" />
    <text x="280" y="220" font-size="14" font-weight="600" fill="#2c4994">W</text>
    <text x="56"  y="195" font-size="11" fill="#666" font-style="italic">closed subspace</text>
    <line x1="160" y1="190" x2="270" y2="60" stroke="#a23ec3" stroke-width="2.5" />
    <line x1="160" y1="190" x2="240" y2="174" stroke="#3d7a26" stroke-width="2.5" />
    <line x1="240" y1="174" x2="270" y2="60"  stroke="#d65336" stroke-width="2.5" />
    <line x1="237" y1="170" x2="225" y2="155" stroke="#666" stroke-width="0.8" />
    <line x1="225" y1="155" x2="244" y2="150" stroke="#666" stroke-width="0.8" />
    <g fill="#222">
      <circle cx="160" cy="190" r="3" />
      <circle cx="240" cy="174" r="3" />
      <circle cx="270" cy="60"  r="3" />
    </g>
    <text x="142" y="208" font-size="12" fill="#222">0</text>
    <text x="275" y="56"  font-size="14" font-weight="600" fill="#a23ec3">u</text>
    <text x="208" y="195" font-size="13" font-weight="600" fill="#3d7a26">w = Π_W u</text>
    <text x="266" y="118" font-size="13" font-weight="600" fill="#d65336">w⊥</text>
    <text x="36"  y="34"  font-size="11" fill="#444" font-style="italic">u = w + w⊥,    ⟨w, w⊥⟩ = 0</text>
    <text x="36"  y="50"  font-size="11" fill="#444" font-style="italic">‖u‖² = ‖w‖² + ‖w⊥‖²</text>
  </svg>
  <figcaption>Every $u \in H$ splits uniquely into a part along the closed subspace $W$ and a part perpendicular to it. The decomposition is realized by the length minimizer in the affine set $u + W$, and the variational condition $\frac{d}{dt}\|v + tw\|^2 \big|_{t=0} = 0$ is exactly what makes $w^\perp \perp W$. The Pythagorean identity $\|u\|^2 = \|w\|^2 + \|w^\perp\|^2$ follows immediately.</figcaption>
</figure>

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

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Uniqueness: if $f(u) = \langle u, v \rangle = \langle u, \tilde{v} \rangle$ for all $u$, then $\langle u, v - \tilde{v} \rangle = 0$ for all $u$, so $v = \tilde{v}$.</p>
    <p>Existence: if $f = 0$, take $v = 0$. Otherwise, let $N = f^{-1}(\lbrace 0 \rbrace)$ be the nullspace of $f$. There exists $u_1 \in H$ with $f(u_1) \ne 0$; set $u_0 = u_1 / f(u_1)$ so that $f(u_0) = 1$. Define $C = \lbrace u \in H : f(u) = 1 \rbrace = f^{-1}(\lbrace 1 \rbrace)$, which is closed and convex. By Theorem 178, there exists $v_0 \in C$ with $\lVert v_0 \rVert = \inf_{u \in C} \lVert u \rVert$.</p>
    <p>We can check that $C = \lbrace v_0 + w : w \in N \rbrace$ and that $v_0 \in N^\perp$ (by the minimization argument from Theorem 180). Set $v = v_0 / \lVert v_0 \rVert^2$. For any $u \in H$, $u = (u - f(u)v_0) + f(u)v_0$ where $u - f(u)v_0 \in N$ and $v_0 \in N^\perp$, so</p>
    $$\langle u, v \rangle = \frac{1}{\lVert v_0 \rVert^2} \left[ \langle u - f(u)v_0, v_0 \rangle + f(u)\langle v_0, v_0 \rangle \right] = f(u). \quad \square$$
  </details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>alternative proof (separable case, via an orthonormal basis)</summary>

Let $\lbrace e\_i \rbrace$ be an orthonormal basis, $\lvert f(u) \rvert \le C\lVert u \rVert$, and set $w\_i = \overline{f(e\_i)}$.

**The $w\_i$ are square-summable.** Put $w\_N = \sum\_{i \le N} w\_i e\_i$, so $\lVert w\_N \rVert^2 = \sum\_{i \le N}\lvert w\_i \rvert^2$. Then

$$f(w_N) = \sum_{i \le N} w_i f(e_i) = \sum_{i \le N} \lvert w_i \rvert^2 = \lVert w_N \rVert^2,$$

and $\lvert f(w\_N) \rvert \le C\lVert w\_N \rVert$ gives $\lVert w\_N \rVert \le C$ for every $N$. Hence $\sum\_i \lvert w\_i \rvert^2 \le C^2$ and $w = \sum\_i w\_i e\_i \in H$.

**$w$ represents $f$.** For $u \in H$, $u\_N = \sum\_{i \le N} \langle u, e\_i \rangle e\_i \to u$, so by continuity of $f$ and of the inner product

$$f(u) = \lim_{N \to \infty} \sum_{i \le N} \langle u, e_i \rangle f(e_i) = \lim_{N \to \infty} \Big\langle u, \sum_{i \le N} w_i e_i \Big\rangle = \langle u, w \rangle.$$

Finally $\lVert f \rVert \le \lVert w \rVert$ by Cauchy–Schwarz, and $f(w) = \lVert w \rVert^2$ gives equality. $\square$

</details>
</div>

In other words, every element of the dual can be realized as an inner product with a fixed vector.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Weighted Sequence Spaces and Their Duals)</span></p>

For $s \in \mathbb{R}$ let

$$h^{s} = \Big\lbrace c : \mathbb{N} \to \mathbb{C} \ :\ \sum_{j=1}^\infty j^{2s}\lvert c_j \rvert^2 < \infty \Big\rbrace, \qquad \langle c, d \rangle_{h^s} = \sum_{j} j^{2s} c_j \overline{d_j}.$$

* $h^s$ is a Hilbert space: the map $(T\_s c)\_j = j^{s}c\_j$ is a linear bijection $h^s \to \ell^2$ with $\lVert c \rVert\_{h^s} = \lVert T\_s c \rVert\_{\ell^2}$, so $h^s$ inherits completeness from $\ell^2$ (no need to redo the $\ell^2$ proof).
* For $s > 0$, $h^s \subset \ell^2 \subset h^{-s}$ with $\lVert c \rVert\_{\ell^2} \le \lVert c \rVert\_{h^s}$: decay of $c\_j$ is a discrete analogue of smoothness (compare $\hat f(n)$ for $f \in H^1$).
* **Duality.** If $T : h^{2} \to \mathbb{C}$ is bounded, Riesz gives $d^{\prime} \in h^{2}$ with $T c = \sum\_j j^4 c\_j \overline{d^{\prime}\_j}$. Setting $d\_j = j^4 \overline{d^{\prime}\_j}$, we get $\sum\_j j^{-4}\lvert d\_j \rvert^2 = \sum\_j j^4 \lvert d^{\prime}\_j \rvert^2 < \infty$, i.e. $d \in h^{-2}$, and

  $$Tc = \sum_{j=1}^\infty c_j d_j.$$

  So, with respect to the **$\ell^2$ pairing** (rather than the $h^2$ inner product), the dual of $h^2$ is $h^{-2}$ — the model for the duality $(H^s)^{\prime} = H^{-s}$ of Sobolev spaces.

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

Let $K \in C([0, 1] \times [0, 1])$, and define $A : L^2([0, 1]) \to L^2([0, 1])$ via 

$$Af(x) = \int_0^1 K(x, y) f(y)\,dy$$

Then the adjoint is 

$$A^* g(x) = \int_0^1 \overline{K(y, x)} g(y)\,dy$$

— flipping the indices and taking a complex conjugate.

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generalization of theorem 189)</span></p>

The Theorem 189 for $A:H\to H$, whereas it is also true for $A:X\to Y$. The identical proof works for $A:X\to Y$, with $A^\ast:Y\to X$. Also 

$$(\bar{\operatorname{Ran}(A)})^\perp = \operatorname{Null}(A^*),$$

because for **any** subset $S$ of a Hilbert space, $S^\perp=\bar{S}^\perp$.

</div>

## Compactness in Hilbert Spaces

### Compact Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 190</span><span class="math-callout__name">(Compact)</span></p>

Let $X$ be a metric space. A subset $K \subset X$ is **compact** if every sequence of elements in $K$ has a subsequence converging to an element of $K$.

</div>

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">Definition 190: a subsequence converges inside K</div>
      <svg viewBox="22 24 296 216" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Sequential compactness: a sequence wanders inside a set K, a highlighted subsequence enters nested shrinking epsilon-balls around a limit point x that lies in K">
        <path d="M 60,110 C 60,55 120,28 185,32 C 250,36 305,70 302,120 C 299,170 245,200 180,198 C 115,196 60,165 60,110 Z" fill="#fff7e0" stroke="#5b6270" stroke-width="1.6" />
        <text x="78" y="116" font-size="15" font-style="italic" fill="#444">K</text>
        <g fill="rgba(21,101,192,0.07)" stroke="#1565c0" stroke-width="1" stroke-dasharray="4,3">
          <circle cx="205" cy="122" r="56" />
          <circle cx="205" cy="122" r="34" />
          <circle cx="205" cy="122" r="18" />
        </g>
        <line x1="198.9" y1="125.5" x2="156.5" y2="150.0" stroke="#1565c0" stroke-width="1" />
        <g fill="#1565c0"><polygon points="156.5,150.0 159.6,145.3 162.1,149.7" /><polygon points="198.9,125.5 193.4,125.8 195.9,130.2" /></g>
        <text x="181" y="135" font-size="10" fill="#1565c0">ε</text>
        <g fill="#888">
          <circle cx="95" cy="150" r="2.6" />
          <circle cx="130" cy="60" r="2.6" />
          <circle cx="275" cy="150" r="2.6" />
          <circle cx="105" cy="95" r="2.6" />
          <circle cx="200" cy="48" r="2.6" />
          <circle cx="150" cy="178" r="2.6" />
          <circle cx="284" cy="112" r="2.6" />
          <circle cx="135" cy="125" r="2.6" />
        </g>
        <g font-size="8.5" fill="#666">
          <text x="90" y="147" text-anchor="end">x₁</text>
          <text x="135" y="57" text-anchor="start">x₃</text>
          <text x="270" y="147" text-anchor="end">x₄</text>
          <text x="100" y="92" text-anchor="end">x₆</text>
          <text x="205" y="45" text-anchor="start">x₇</text>
          <text x="155" y="182" text-anchor="start">x₉</text>
          <text x="279" y="108" text-anchor="end">x₁₀</text>
          <text x="130" y="128" text-anchor="end">x₁₂</text>
        </g>
        <g fill="none" stroke="#2c3e94" stroke-width="1.1">
          <path d="M 150.4,84.0 L 151.2,83.3 L 151.9,82.7 L 152.7,82.2 L 153.5,81.6 L 154.3,81.0 L 155.1,80.5 L 155.9,80.0 L 156.7,79.5 L 157.5,79.0 L 158.4,78.5 L 159.2,78.1 L 160.1,77.6 L 160.9,77.2 L 161.8,76.8 L 162.7,76.4 L 163.5,76.0 L 164.4,75.7 L 165.3,75.3 L 166.2,75.0 L 167.1,74.7 L 168.0,74.4 L 169.0,74.1 L 169.9,73.9 L 170.8,73.6 L 171.8,73.4 L 172.8,73.2 L 173.7,73.0 L 174.7,72.8 L 175.7,72.6 L 176.7,72.5 L 177.7,72.4 L 178.7,72.2 L 179.7,72.1 L 180.7,72.1 L 181.7,72.0 L 182.8,71.9 L 183.8,71.9 L 184.9,71.9 L 185.9,71.9 L 187.0,71.9 L 188.1,71.9 L 189.2,72.0 L 190.2,72.0 L 191.3,72.1 L 192.5,72.2 L 193.6,72.3 L 194.7,72.5 L 195.8,72.6 L 197.0,72.8 L 198.1,72.9 L 199.3,73.1 L 200.4,73.3 L 201.6,73.6 L 202.8,73.8 L 204.0,74.1 L 205.2,74.3 L 206.4,74.6 L 207.6,74.9 L 208.8,75.2 L 210.0,75.6 L 211.2,75.9 L 212.5,76.3 L 213.7,76.7 L 215.0,77.1 L 216.3,77.5" />
          <path d="M 231.8,86.6 L 232.1,87.0 L 232.3,87.5 L 232.6,87.9 L 232.9,88.4 L 233.1,88.9 L 233.4,89.4 L 233.6,89.8 L 233.8,90.3 L 234.1,90.8 L 234.3,91.3 L 234.5,91.8 L 234.6,92.4 L 234.8,92.9 L 235.0,93.4 L 235.2,93.9 L 235.3,94.5 L 235.5,95.0 L 235.6,95.6 L 235.7,96.1 L 235.8,96.7 L 235.9,97.2 L 236.0,97.8 L 236.1,98.4 L 236.2,99.0 L 236.3,99.6 L 236.3,100.2 L 236.4,100.8 L 236.4,101.4 L 236.5,102.0 L 236.5,102.6 L 236.5,103.2 L 236.5,103.8 L 236.5,104.5 L 236.5,105.1 L 236.5,105.8 L 236.5,106.4 L 236.4,107.1 L 236.4,107.7 L 236.3,108.4 L 236.3,109.1 L 236.2,109.8 L 236.1,110.5 L 236.0,111.1 L 235.9,111.8 L 235.8,112.6 L 235.7,113.3 L 235.5,114.0 L 235.4,114.7 L 235.3,115.4 L 235.1,116.2 L 234.9,116.9 L 234.8,117.6 L 234.6,118.4 L 234.4,119.2" />
          <path d="M 225.5,134.5 L 225.3,134.6 L 225.0,134.7 L 224.7,134.8 L 224.4,134.9 L 224.1,135.0 L 223.8,135.1 L 223.5,135.2 L 223.2,135.3 L 222.8,135.4 L 222.5,135.4 L 222.2,135.5 L 221.8,135.5 L 221.5,135.6 L 221.2,135.7 L 220.8,135.7 L 220.5,135.7 L 220.1,135.8 L 219.7,135.8 L 219.4,135.8 L 219.0,135.9 L 218.6,135.9 L 218.3,135.9 L 217.9,135.9 L 217.5,135.9 L 217.1,135.9 L 216.7,135.9 L 216.3,135.9 L 215.9,135.9 L 215.5,135.9 L 215.0,135.9 L 214.6,135.8 L 214.2,135.8 L 213.8,135.8 L 213.3,135.7 L 212.9,135.7" />
        </g>
        <g fill="#2c3e94">
          <polygon points="222.7,79.8 216.0,80.6 218.1,75.0" />
          <polygon points="232.5,125.4 231.4,118.8 237.1,120.5" />
          <polygon points="206.7,134.7 213.1,132.7 212.2,138.6" />
          <circle cx="146.0" cy="88.0" r="3.2" />
          <circle cx="228.0" cy="82.0" r="3.2" />
          <circle cx="230.4" cy="131.2" r="3.2" />
          <circle cx="200.9" cy="133.3" r="3.2" />
        </g>
        <g font-size="8.5" fill="#2c3e94">
          <text x="141" y="85" text-anchor="end">x₂</text>
          <text x="233" y="79" text-anchor="start">x₅</text>
          <text x="236" y="135" text-anchor="start">x₈</text>
          <text x="201" y="148" text-anchor="middle">x₁₁</text>
        </g>
        <g fill="#c62828"><circle cx="205" cy="122" r="4.0" /></g>
        <text x="211" y="129" font-size="10" font-style="italic" font-weight="600" fill="#c62828">x</text>
        <g fill="#888"><circle cx="30" cy="215" r="2.6" /></g>
        <g fill="#2c3e94"><circle cx="30" cy="229" r="3.2" /></g>
        <g font-size="9" fill="#444">
          <text x="38" y="218">the sequence x₁, x₂, x₃, … may wander anywhere in K</text>
          <text x="38" y="232">a subsequence x₂, x₅, x₈, x₁₁, … → x, with the limit x ∈ K</text>
        </g>
      </svg>
    </div>
    <div>
      <div class="panel-label">Finite cover: finitely many ε-balls, for every ε</div>
      <img src="{{ '/assets/images/notes/books/functional_analysis/mit_rogriguez/fa_totally_bounded.png' | relative_url }}" alt="Two gray ellipses representing a totally bounded set; the left is covered by four large dashed epsilon-balls, the right by many small dotted epsilon-balls, with the caption: totally bounded X can be covered with finitely many N(x, epsilon), for any epsilon greater than zero" loading="lazy" width="340">
    </div>
  </div>
  <figcaption>Two faces of compactness in a metric space. <strong>Left (Definition 190).</strong> A sequence in $K$ may wander forever (gray), but one can always extract a subsequence $x_{n_1}, x_{n_2}, \dots$ (blue) that enters ever-smaller $\varepsilon$-balls around some point $x$, and that limit still belongs to $K$. <strong>Right (finite cover).</strong> $K$ is <em>totally bounded</em>: for every $\varepsilon > 0$ it is covered by finitely many balls $N(x, \varepsilon)$; a smaller $\varepsilon$ needs more balls, but never infinitely many. In a metric space these notions line up: $K$ satisfies Definition 190 $\iff$ every open cover of $K$ has a finite subcover $\iff$ $K$ is complete and totally bounded. Completeness is what stops a limit from escaping the set: $(0,1)$ is totally bounded, yet $x_n = 1/n$ has no subsequence converging inside $(0,1)$. Boundedness alone is weaker still: the closed unit ball of an infinite-dimensional Hilbert space is bounded but not totally bounded (an orthonormal sequence has mutual distance $\sqrt{2}$, so no finite $\tfrac{1}{2}$-net exists), which is the mechanism behind Example 193 below.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 191</span></p>

By the Pigeonhole Principle, all finite subsets are compact.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 192</span><span class="math-callout__name">(Heine–Borel)</span></p>

A subset $K \subset \mathbb{R}$ (also $\mathbb{R}^n$ and $\mathbb{C}^n$) is compact $\iff$ $K$ is closed and bounded.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generally, every compact is closed and bounded, but not the reversed)</span></p>

This does not hold for arbitrary metric spaces or even Banach spaces, and in fact it is still not true for Hilbert spaces.

**In any metric space, every compact subset is closed and bounded, but not every closed and bounded subset is compact**

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 193</span></p>

Let $H$ be an infinite-dimensional Hilbert space. Then the closed ball $F = \lbrace u \in H : \lVert u \rVert \le 1 \rbrace$ is closed and bounded but **not** compact.

This is because we can take a countably infinite orthonormal subset $\lbrace e_n \rbrace_{n=1}^{\infty}$ with all $e_n \in F$, but $\lVert e_n - e_k \rVert^2 = 2$ for all $n \ne k$, so no subsequence is Cauchy. (every convergent sequence in a metric space is a Cauchy sequence)

</div>

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">Example 193: eₙ in the unit ball, pairwise √2 apart</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Closed unit ball drawn as a sphere with three orthonormal vectors e1, e2, e3 from the origin to its surface, dashed chords of length root two between them, and a dotted stub hinting at e4, e5 and further orthogonal directions">
        <text x="12" y="16" font-size="10" fill="#444">F = { u : ‖u‖ ≤ 1 }  (closed unit ball)</text>
        <circle cx="120" cy="108" r="76" fill="#edf4ff" stroke="#5b6270" stroke-width="1.6" />
        <ellipse cx="120" cy="108" rx="76" ry="20" fill="none" stroke="#bbb" stroke-width="1" stroke-dasharray="3,3" />
        <g stroke="#b91c1c" stroke-width="1.1" stroke-dasharray="4,3" fill="none">
          <line x1="196" y1="108" x2="120" y2="32" />
          <line x1="120" y1="32" x2="75" y2="148" />
          <line x1="196" y1="108" x2="75" y2="148" />
        </g>
        <g font-size="9" fill="#b91c1c">
          <text x="163" y="67">√2</text>
          <text x="92" y="89" text-anchor="end">√2</text>
          <text x="138" y="143">√2</text>
        </g>
        <path d="M 120,96 L 132,96 L 132,108" fill="none" stroke="#666" stroke-width="1" />
        <g stroke="#2c3e94" stroke-width="1.4">
          <line x1="120" y1="108" x2="196" y2="108" />
          <line x1="120" y1="108" x2="120" y2="32" />
          <line x1="120" y1="108" x2="75" y2="148" />
        </g>
        <line x1="120" y1="108" x2="88" y2="62" stroke="#888" stroke-width="1.2" stroke-dasharray="1.5,2.5" />
        <g fill="#888"><circle cx="88" cy="62" r="2.5" /></g>
        <text x="86" y="54" font-size="8.5" fill="#888" text-anchor="middle">e₄, e₅, …</text>
        <g fill="#444"><circle cx="120" cy="108" r="2.2" /></g>
        <text x="124" y="121" font-size="9" fill="#444">0</text>
        <text x="154" y="104" font-size="8.5" fill="#2c3e94">1</text>
        <g fill="#2c3e94">
          <circle cx="196" cy="108" r="3.5" />
          <circle cx="120" cy="32" r="3.5" />
          <circle cx="75" cy="148" r="3.5" />
        </g>
        <g font-size="10" fill="#2c3e94">
          <text x="203" y="112">e₁</text>
          <text x="127" y="36">e₂</text>
          <text x="72" y="166">e₃</text>
        </g>
        <g font-size="9" fill="#444">
          <text x="12" y="200">‖eₙ − eₘ‖² = ‖eₙ‖² + ‖eₘ‖² = 2   (Pythagoras, n ≠ m)</text>
          <text x="12" y="213">closed ✓   bounded ✓   but no subsequence is Cauchy ✗</text>
        </g>
      </svg>
    </div>
    <div>
      <div class="panel-label">In coordinates: the mass escapes to infinity</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Bar chart of squared coefficients of e_n against the basis index k: a single unit bar at k equals n, drawn for n from 1 to 5 with fading color and an arrow n to infinity, so for any cutoff N the whole mass of later e_n lies in the shaded tail region k greater than N">
        <rect x="100" y="50" width="180" height="110" fill="#fff7e0" />
        <line x1="100" y1="50" x2="100" y2="160" stroke="#a86f00" stroke-width="1" stroke-dasharray="4,3" />
        <g font-size="9" fill="#a86f00" text-anchor="middle">
          <text x="100" y="186">N</text>
          <text x="195" y="186">tail  k &gt; N</text>
        </g>
        <line x1="40.0" y1="160.0" x2="282.0" y2="160.0" stroke="#444" stroke-width="1.2" /><polygon points="288.0,160.0 282.0,163.0 282.0,157.0" fill="#444" />
        <line x1="40.0" y1="160.0" x2="40.0" y2="46.0" stroke="#444" stroke-width="1.2" /><polygon points="40.0,40.0 43.0,46.0 37.0,46.0" fill="#444" />
        <text x="290" y="164" font-size="11" fill="#666">k</text>
        <text x="12" y="34" font-size="9" fill="#666">|⟨eₙ, eₖ⟩|²</text>
        <line x1="36" y1="56" x2="40" y2="56" stroke="#444" stroke-width="1" />
        <text x="32" y="59" font-size="9" fill="#666" text-anchor="end">1</text>
        <g font-size="9" fill="#666" text-anchor="middle">
          <text x="55" y="173">1</text>
          <text x="85" y="173">2</text>
          <text x="115" y="173">3</text>
          <text x="145" y="173">4</text>
          <text x="175" y="173">5</text>
          <text x="205" y="173">6</text>
          <text x="235" y="173">7</text>
          <text x="265" y="173">8</text>
        </g>
        <g fill="#2c3e94">
          <rect x="47" y="56" width="16" height="104" opacity="1" />
          <rect x="77" y="56" width="16" height="104" opacity="0.8" />
          <rect x="107" y="56" width="16" height="104" opacity="0.62" />
          <rect x="137" y="56" width="16" height="104" opacity="0.47" />
          <rect x="167" y="56" width="16" height="104" opacity="0.35" />
        </g>
        <g font-size="8.5" fill="#2c3e94" text-anchor="middle">
          <text x="55" y="51">e₁</text>
          <text x="85" y="51">e₂</text>
          <text x="115" y="51">e₃</text>
          <text x="145" y="51">e₄</text>
          <text x="175" y="51">e₅</text>
        </g>
        <text x="232" y="112" font-size="14" fill="#888">⋯</text>
        <line x1="222.0" y1="98.0" x2="266.0" y2="98.0" stroke="#2c3e94" stroke-width="1.1" /><polygon points="272.0,98.0 266.0,101.0 266.0,95.0" fill="#2c3e94" />
        <text x="247" y="92" font-size="9" fill="#2c3e94" text-anchor="middle">n → ∞</text>
        <g font-size="9" fill="#444">
          <text x="12" y="203">each eₙ with n &gt; N has its whole mass (= 1) in the tail k &gt; N</text>
          <text x="12" y="215">no single N works for all n: the tails are not equi-small</text>
        </g>
      </svg>
    </div>
  </div>
  <figcaption>Example 193 seen two ways. <strong>Left.</strong> In the closed unit ball $F$ the orthonormal vectors $e_1, e_2, e_3, \dots$ all have norm $1$, and by Pythagoras $\lVert e_n - e_m \rVert^2 = \lVert e_n \rVert^2 + \lVert e_m \rVert^2 = 2$ for $n \ne m$. Any subsequence keeps this mutual distance $\sqrt{2}$, so it is never Cauchy, let alone convergent: $F$ is closed and bounded but not compact, and Heine–Borel fails. Only three of the $e_n$ fit into a drawing; in infinite dimensions there is always a fresh direction orthogonal to all the previous ones. <strong>Right.</strong> The same failure in coordinates with respect to the basis $\lbrace e_k \rbrace$ itself, where $\langle e_n, e_k \rangle = \delta_{nk}$. The coefficient mass of $e_n$ is a single unit at position $k = n$, and it drifts to infinity as $n$ grows: for any cutoff $N$, every $e_n$ with $n > N$ has its whole mass in the tail $k > N$. Each individual $e_n$ has a small tail beyond some $N$, but no single $N$ serves all of them at once. This uniformity is exactly what the next definition (equi-small tails) demands, and Theorem 197 shows it is the missing ingredient.</figcaption>
</figure>

### Equi-Small Tails

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 194</span><span class="math-callout__name">(Equi-Small Tails)</span></p>

Let $H$ be a Hilbert space. A subset $K \subset H$ has **equi-small tails** with respect to a countable orthonormal subset $\lbrace e_n \rbrace$ if for all $\varepsilon > 0$, there is some $n \ge N$ so that for all $v \in K$, we have

$$\sum_{k > N} \lvert \langle v, e_k \rangle \rvert^2 < \varepsilon^2.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Small tail for one is trivial, equi-small tail is a group property)</span></p>

We know that the sequence for any given $v$ converges by Bessel’s inequality, so that the inequality above will eventually hold for some $N$ for each $v$. But this equi-small tails requirement is a more “uniform” condition on the rate of convergence – we need to be able to pick an $N$ that works for all $v \in K$ at the same time.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 195</span></p>

Any finite set $K$ has equi-small tails with respect to any countable orthonormal subset (we can take the maximum of finitely many $N$s).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 196</span><span class="math-callout__name">(Equi-Small Tails of Convergent Sequences)</span></p>

Let $H$ be a Hilbert space, and let $\lbrace v_n \rbrace_n$ be a convergent sequence with $v_n \to v$. If $\lbrace e_k \rbrace$ is a countable orthonormal subset, then $K = \lbrace v_n : n \in \mathbb{N} \rbrace \cup \lbrace v \rbrace$ is compact, and $K$ has equi-small tails with respect to $\lbrace e_k \rbrace$.

</div>

<div class="accordion">
  <details>
    <summary>proof (equi-small tails)</summary>
    <p>Let $\varepsilon > 0$. Choose $M$ so that $\lVert v_n - v \rVert < \varepsilon / 2$ for all $n \ge M$. Choose $N$ large enough so that $\sum_{k > N} \lvert \langle v, e_k \rangle \rvert^2 < \varepsilon^2 / 4$ and also $\sum_{k > N} \lvert \langle v_n, e_k \rangle \rvert^2 < \varepsilon^2 / 4$ for $1 \le n \le M - 1$ (finitely many terms).</p>
    <p>For $n \ge M$: $\left( \sum_{k > N} \lvert \langle v_n, e_k \rangle \rvert^2 \right)^{1/2} \le \left( \sum_{k > N} \lvert \langle v_n - v, e_k \rangle \rvert^2 \right)^{1/2} + \left( \sum_{k > N} \lvert \langle v, e_k \rangle \rvert^2 \right)^{1/2}$ by the $\ell^2$ triangle inequality. By Bessel's inequality, the first term is $\le \lVert v_n - v \rVert < \varepsilon / 2$, and the second is $< \varepsilon / 2$. $\square$</p>
  </details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>My proof</summary>

* To prove compactness, consider any open cover of $K$. Some open set $U$ contains $v$. Since $v_n\to v$, all but finitely many $v_n$ belong to $U$. The remaining finitely many points can be covered by finitely many additional sets.
* Now we show the equi-small tails by idea that if some element in a convergent sequence has a small tail, then subsequent elements given they do not differ enough (depends on the tail bound) cannot accumulate enough difference in their tails since the difference is too small. The finite prefix could be then compared to the result with infinite suffix.
* **Find sufficiently small difference suffix of the sequence:**
  * Let $M \in \mathbb{N}$: $\forall m,n > M: \lVert v_m - v_n \rVert \lt \varepsilon / 2$.
* **Find small enough tail element in the suffix:**
  * Let $k>M$ and $N\in\mathbb{N}$: $\lVert v_k^{\gt N} \rVert \lt \varepsilon / 2,$
  * where $v_k^{\gt N}$ is a tail starting $N$.
* For all $l>k$ **triangular inequality**:
  * $\lVert v_l^{\gt N} \rVert = \lVert v_l^{\gt N} - v_k^{\gt N} + v_k^{\gt N} \rVert \leq \lVert v_l^{\gt N} - v_k^{\gt N} \rVert + \lVert v_k^{\gt N} \rVert$.
  * The first term is bounded by the norm of the full elements: $\lVert v_l^{\gt N} - v_k^{\gt N} \rVert \leq \lVert v_l - v_k \rVert \lt \varepsilon / 2$.
  * The second term is bounded by assumption $\lVert v_k^{\gt N} \rVert \lt \varepsilon / 2$.
  * Then $\lVert v_l^{\gt N} \rVert \lt \varepsilon / 2 + \varepsilon / 2 = \varepsilon$.
  * Thus $\lVert v_l^{\gt N} \rVert^2 \lt \varepsilon^2$.
* Given the tail index $N$, we set the final index as $N^\ast := \max(N, N_{1:k})$, where $N_{1:k}$ is the max tail index for the first $k$ elements of the sequence.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Compact sets on a real line)</span></p>

Examples on the real line included the following examples
* closed intervals,
* the set $\lbrace 0\rbrace\cup \lbrace 1 n : n\in\mathbb{N}\rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Significance of the Equi-Small Tails of Convergent Sequences in Hilbert Spaces)</span></p>

This captures the main nontrivial conclusion: **convergence in a Hilbert space implies uniform smallness of the tails of the orthonormal coefficients**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(My proof structure)</span></p>

The proof above has a particularly useful structure:
1. Use the Cauchy property to control an infinite suffix uniformly.
2. Use convergence of the Fourier expansion to control the tail of one representative vector.
3. Handle the remaining finite prefix separately.
4. Combine the two bounds into a single uniform tail estimate.

</div>

### Characterization of Compact Sets

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 197</span><span class="math-callout__name">(Compactness Criterion via Equi-Small Tails)</span></p>

Let $H$ be a separable Hilbert space, and let $\lbrace e_k \rbrace_k$ be an orthonormal basis of $H$. Then a subset $K \subset H$ is compact if and only if $K$ is closed, bounded, and has equi-small tails with respect to $\lbrace e_k \rbrace$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Forward direction:</strong> If $K$ is compact, it is closed and bounded. For equi-small tails, suppose not: then there exists $\varepsilon_0 > 0$ and for each $N$ some $u_N \in K$ with $\sum_{k > N} \lvert \langle u_N, e_k \rangle \rvert^2 \ge \varepsilon_0^2$. By compactness, $\lbrace u_N \rbrace$ (importantly, the whole sequence is formed by such "bad" $u_N$) has a convergent subsequence $v_m \to v \in K$. But then $\lbrace v_n : n \in \mathbb{N} \rbrace \cup \lbrace v \rbrace$ does not have equi-small tails, contradicting Theorem 196.</p>
    <p><strong>Backward direction:</strong> Let $\lbrace u_n \rbrace$ be a sequence in $K$. Since $K$ is bounded, $\lVert u_n \rVert \le C$ for all $n$, so $\lvert \langle u_n, e_k \rangle \rvert \le C$. By Bolzano–Weierstrass (applied to $k = 1$), extract a subsequence where the first coefficient converges. Repeat for $k = 2, 3, \dots$ and use a diagonal argument to get a subsequence $v_\ell = u_{n_\ell}$ where $\langle v_\ell, e_k \rangle$ converges for every $k$.</p>
    <p>To show $\lbrace v_\ell \rbrace$ is Cauchy: (infinite suffix control) by equi-small tails, choose $N$ with $\sum_{k > N} \lvert \langle v_\ell, e_k \rangle \rvert^2 < \varepsilon^2 / 4$ for all $\ell$. (finite prefix control) Then choose $M$ so that for $\ell, m \ge M$, $\sum_{k=1}^{N} \lvert \langle v_\ell, e_k \rangle - \langle v_m, e_k \rangle \rvert^2 < \varepsilon^2 / 2$. Using Parseval and the $\ell^2$ triangle inequality on the tail, $\lVert v_\ell - v_m \rVert < \varepsilon$. Since $K$ is closed, the limit is in $K$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The compactness characterization extends to nonseparable Hilbert spaces, but its natural formulation is through **uniform approximation by finite-dimensional subspaces** (Theorem 199), rather than tails of a countable orthonormal basis.

</div>

We can also characterize compact sets without reference to an orthonormal basis:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 199</span><span class="math-callout__name">(Compactness Criterion via Finite-Dimensional Approximation)</span></p>

A subset $K \subset H$ is compact $\iff$ $K$ is closed, bounded, and for all $\varepsilon > 0$, there exists a finite-dimensional subspace $W \subset H$ so that for all $u \in K$ 

$$\inf_{w \in W} \lVert u - w \rVert < \varepsilon.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two Criteria of Compactness: Equi-Small Tails vs. Finite-Dimensional Approximation)</span></p>

These names emphasize the relationship between the results:
* **Theorem 197:** Compactness characterized through uniform control of Fourier tails.
* **Theorem 199:** The basis-independent generalization, characterized through uniform finite-dimensional approximation.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 198</span><span class="math-callout__name">(Hilbert Cube)</span></p>

Let $K$ be the set (not subspace) of sequences $\lbrace a_k \rbrace_k$ in $\ell^2$ satisfying $\lvert a_k \rvert \le 2^{-k}$ — this set is known as the **Hilbert cube**, and it is compact.

</div>

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">Example 198: the Hilbert cube in three coordinates</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Oblique drawing of a flat box with sides 1, one half and one quarter along the coordinates a1, a2, a3, dimension lines labelling each side, a red sample point a inside, and a note that the remaining sides keep halving">
        <text x="12" y="16" font-size="10" fill="#444">K ∩ span{e₁, e₂, e₃}: a box with sides 1, ½, ¼</text>
        <g stroke="#bbb" stroke-width="1" stroke-dasharray="3,3" fill="none">
          <line x1="70" y1="153" x2="82.4" y2="140.6" />
          <line x1="82.4" y1="140.6" x2="222.4" y2="140.6" />
          <line x1="82.4" y1="140.6" x2="82.4" y2="70.6" />
        </g>
        <polygon points="70,83 210,83 222.4,70.6 82.4,70.6" fill="rgba(44,73,148,0.12)" stroke="#5b6270" stroke-width="1.2" />
        <polygon points="210,83 222.4,70.6 222.4,140.6 210,153" fill="rgba(44,73,148,0.22)" stroke="#5b6270" stroke-width="1.2" />
        <rect x="70" y="83" width="140" height="70" fill="#edf4ff" stroke="#5b6270" stroke-width="1.4" />
        <line x1="75.0" y1="168.0" x2="205.0" y2="168.0" stroke="#666" stroke-width="1" /><polygon points="210.0,168.0 205.0,170.5 205.0,165.5" fill="#666" /><polygon points="70.0,168.0 75.0,165.5 75.0,170.5" fill="#666" />
        <text x="140" y="181" font-size="9" fill="#444" text-anchor="middle">1 = 2·2⁻¹</text>
        <line x1="56.0" y1="148.0" x2="56.0" y2="88.0" stroke="#666" stroke-width="1" /><polygon points="56.0,83.0 58.5,88.0 53.5,88.0" fill="#666" /><polygon points="56.0,153.0 53.5,148.0 58.5,148.0" fill="#666" />
        <text x="50" y="121" font-size="9" fill="#444" text-anchor="end">½ = 2·2⁻²</text>
        <line x1="221.5" y1="155.5" x2="226.9" y2="150.1" stroke="#666" stroke-width="1" /><polygon points="230.4,146.6 228.6,151.9 225.1,148.4" fill="#666" /><polygon points="218.0,159.0 219.8,153.7 223.3,157.2" fill="#666" />
        <text x="236" y="156" font-size="9" fill="#444">¼ = 2·2⁻³</text>
        <g fill="#c62828"><circle cx="150" cy="120" r="3" /></g>
        <text x="156" y="117" font-size="10" font-style="italic" font-weight="600" fill="#c62828">a</text>
        <line x1="34.0" y1="186.0" x2="59.0" y2="186.0" stroke="#666" stroke-width="1" /><polygon points="64.0,186.0 59.0,188.5 59.0,183.5" fill="#666" />
        <line x1="34.0" y1="186.0" x2="34.0" y2="161.0" stroke="#666" stroke-width="1" /><polygon points="34.0,156.0 36.5,161.0 31.5,161.0" fill="#666" />
        <line x1="34.0" y1="186.0" x2="48.5" y2="171.5" stroke="#666" stroke-width="1" /><polygon points="52.0,168.0 50.2,173.3 46.7,169.8" fill="#666" />
        <g font-size="8.5" fill="#666">
          <text x="67" y="189">a₁</text>
          <text x="37" y="156">a₂</text>
          <text x="55" y="167">a₃</text>
        </g>
        <g font-size="9" fill="#444">
          <text x="12" y="203">the sides keep halving: ⅛, 1/16, … in the remaining coordinates,</text>
          <text x="12" y="215">so K is a thin ‘cube’ that is almost finite-dimensional</text>
        </g>
      </svg>
    </div>
    <div>
      <div class="panel-label">Under the envelope 2⁻ᵏ: tails are uniformly small</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Bar chart over the coordinate index k showing the halving envelope two to the minus k as dashed light bars, one element a of the Hilbert cube as dots inside the bars joined by a line, and a shaded tail region beyond N where the envelope is already tiny">
        <rect x="130" y="48" width="150" height="117" fill="#fff7e0" />
        <line x1="130" y1="48" x2="130" y2="165" stroke="#a86f00" stroke-width="1" stroke-dasharray="4,3" />
        <g font-size="9" fill="#a86f00" text-anchor="middle">
          <text x="130" y="191">N</text>
          <text x="210" y="191">tail  k &gt; N</text>
        </g>
        <g fill="#edf4ff" stroke="#1565c0" stroke-width="1" stroke-dasharray="3,2">
          <rect x="46" y="55.0" width="18" height="110.0" />
          <rect x="76" y="110.0" width="18" height="55.0" />
          <rect x="106" y="137.5" width="18" height="27.5" />
          <rect x="136" y="151.2" width="18" height="13.8" />
          <rect x="166" y="158.1" width="18" height="6.9" />
          <rect x="196" y="161.6" width="18" height="3.4" />
          <rect x="226" y="163.3" width="18" height="1.7" />
          <rect x="256" y="164.1" width="18" height="0.9" />
        </g>
        <line x1="40.0" y1="165.0" x2="282.0" y2="165.0" stroke="#444" stroke-width="1.2" /><polygon points="288.0,165.0 282.0,168.0 282.0,162.0" fill="#444" />
        <line x1="40.0" y1="165.0" x2="40.0" y2="46.0" stroke="#444" stroke-width="1.2" /><polygon points="40.0,40.0 43.0,46.0 37.0,46.0" fill="#444" />
        <text x="290" y="169" font-size="11" fill="#666">k</text>
        <text x="14" y="48" font-size="9" fill="#666">|aₖ|</text>
        <g font-size="9" fill="#666" text-anchor="middle">
          <text x="55" y="178">1</text>
          <text x="85" y="178">2</text>
          <text x="115" y="178">3</text>
          <text x="145" y="178">4</text>
          <text x="175" y="178">5</text>
          <text x="205" y="178">6</text>
          <text x="235" y="178">7</text>
          <text x="265" y="178">8</text>
        </g>
        <text x="70" y="60" font-size="9" fill="#1565c0">|aₖ| ≤ 2⁻ᵏ  (envelope)</text>
        <polyline points="55.0,82.5 85.0,140.2 115.0,141.6 145.0,157.4 175.0,158.8 205.0,163.3 235.0,163.8 265.0,164.5" fill="none" stroke="#2c3e94" stroke-width="1" />
        <g fill="#2c3e94">
          <circle cx="55.0" cy="82.5" r="2.8" />
          <circle cx="85.0" cy="140.2" r="2.8" />
          <circle cx="115.0" cy="141.6" r="2.8" />
          <circle cx="145.0" cy="157.4" r="2.8" />
          <circle cx="175.0" cy="158.8" r="2.8" />
          <circle cx="205.0" cy="163.3" r="2.8" />
          <circle cx="235.0" cy="163.8" r="2.8" />
          <circle cx="265.0" cy="164.5" r="2.8" />
        </g>
        <text x="74" y="108" font-size="9" fill="#2c3e94">a ∈ K</text>
        <g font-size="9" fill="#444">
          <text x="12" y="205">tail mass ≤ 4⁻ᴺ/3 for every a ∈ K: one N serves all of K</text>
          <text x="12" y="217">closed + bounded + equi-small tails ⇒ compact (Theorem 197)</text>
        </g>
      </svg>
    </div>
  </div>
  <figcaption>The Hilbert cube $K = \lbrace a \in \ell^2 : \lvert a_k \rvert \le 2^{-k} \rbrace$. <strong>Left.</strong> In the first three coordinates $K$ is a box with sides $1, \tfrac{1}{2}, \tfrac{1}{4}$; each further coordinate halves the side again, so the cube is squeezed ever thinner in the later directions. <strong>Right.</strong> Every $a \in K$ lies under the envelope $2^{-k}$, so its tail is bounded by $\sum_{k > N} 4^{-k} = 4^{-N}/3$, a number that does not depend on $a$: the tails are equi-small (Definition 194). Together with closedness (coordinatewise limits respect $\lvert a_k \rvert \le 2^{-k}$) and boundedness ($\lVert a \rVert^2 \le \sum_k 4^{-k} = \tfrac{1}{3}$), Theorem 197 gives compactness. In the language of Theorem 199, truncating after $N$ coordinates leaves every $a \in K$ within $2^{-N}/\sqrt{3}$ of the finite-dimensional subspace $\operatorname{span}\lbrace e_1, \dots, e_N \rbrace$. Contrast Example 193: there each $e_n$ put its full mass at an ever later coordinate; here the envelope pins the mass to the first few coordinates, which is what makes the cube compact.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Arzela-Ascoli)</span></p>

Let $f_n \in C([a, b])$, $n \in \mathbb{N}$, such that
* the sequence is **uniformly bounded**:
  
  $$\exists B \geq 0\forall n\in\mathbb{N}: \lVert f_n \rVert_\infty \leq B,$$

* the sequence $\lbrace f_n\rbrace_n$ is **equi-continuous**: 
  
  $$\forall \varepsilon > 0 \exists \delta > 0: \lvert x − y\rvert <\delta \implies \forall n\in\mathbb{N}: \lvert f_n(x) − f_n(y)\rvert <\varepsilon.$$

Then there exists a subsequence $\lbrace f_{n_j}\rbrace_j$ converging in $C([a, b])$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>TODO:</p>
  </details>
</div>

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">Uniformly bounded: one band holds every graph</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="A dashed horizontal band between minus B and B over the interval a to b containing four wiggly graphs; a dashed red graph leaves the band through the top, marked as unbounded, escaping to infinity">
        <rect x="30" y="50" width="240" height="120" fill="#edf4ff" stroke="#1565c0" stroke-width="1" stroke-dasharray="4,3" />
        <line x1="30" y1="110" x2="270" y2="110" stroke="#bbb" stroke-width="0.8" />
        <g font-size="9" fill="#1565c0" text-anchor="end"><text x="25" y="54">B</text><text x="25" y="173">−B</text></g>
        <g font-size="9" fill="#666" text-anchor="middle"><text x="30" y="184">a</text><text x="270" y="184">b</text></g>
        <polyline points="30.0,94.2 31.5,91.7 33.0,89.3 34.5,86.9 36.0,84.6 37.5,82.4 39.0,80.2 40.5,78.2 42.0,76.3 43.5,74.5 45.0,72.9 46.5,71.4 48.0,70.1 49.5,68.9 51.0,67.9 52.5,67.1 54.0,66.4 55.5,66.0 57.0,65.7 58.5,65.6 60.0,65.6 61.5,65.9 63.0,66.3 64.5,66.8 66.0,67.5 67.5,68.4 69.0,69.4 70.5,70.5 72.0,71.8 73.5,73.1 75.0,74.5 76.5,76.0 78.0,77.6 79.5,79.2 81.0,80.8 82.5,82.5 84.0,84.2 85.5,85.9 87.0,87.5 88.5,89.2 90.0,90.8 91.5,92.3 93.0,93.8 94.5,95.2 96.0,96.6 97.5,97.8 99.0,99.0 100.5,100.1 102.0,101.1 103.5,102.0 105.0,102.8 106.5,103.5 108.0,104.1 109.5,104.6 111.0,105.0 112.5,105.3 114.0,105.6 115.5,105.8 117.0,105.9 118.5,106.0 120.0,106.0 121.5,106.0 123.0,105.9 124.5,105.8 126.0,105.8 127.5,105.7 129.0,105.7 130.5,105.7 132.0,105.7 133.5,105.8 135.0,105.9 136.5,106.1 138.0,106.4 139.5,106.7 141.0,107.2 142.5,107.7 144.0,108.4 145.5,109.1 147.0,110.0 148.5,111.0 150.0,112.1 151.5,113.3 153.0,114.6 154.5,116.0 156.0,117.5 157.5,119.0 159.0,120.7 160.5,122.5 162.0,124.3 163.5,126.1 165.0,128.1 166.5,130.0 168.0,132.0 169.5,134.0 171.0,135.9 172.5,137.9 174.0,139.8 175.5,141.7 177.0,143.5 178.5,145.3 180.0,146.9 181.5,148.5 183.0,149.9 184.5,151.3 186.0,152.5 187.5,153.5 189.0,154.4 190.5,155.1 192.0,155.7 193.5,156.1 195.0,156.3 196.5,156.3 198.0,156.2 199.5,155.9 201.0,155.3 202.5,154.6 204.0,153.8 205.5,152.7 207.0,151.5 208.5,150.2 210.0,148.7 211.5,147.0 213.0,145.3 214.5,143.4 216.0,141.4 217.5,139.3 219.0,137.1 220.5,134.9 222.0,132.7 223.5,130.3 225.0,128.0 226.5,125.7 228.0,123.4 229.5,121.1 231.0,118.8 232.5,116.6 234.0,114.4 235.5,112.3 237.0,110.3 238.5,108.4 240.0,106.6 241.5,104.9 243.0,103.3 244.5,101.8 246.0,100.5 247.5,99.3 249.0,98.2 250.5,97.2 252.0,96.3 253.5,95.6 255.0,95.0 256.5,94.6 258.0,94.2 259.5,93.9 261.0,93.7 262.5,93.7 264.0,93.6 265.5,93.7 267.0,93.8 268.5,94.0 270.0,94.2" fill="none" stroke="#1f2430" stroke-width="1.4" />
        <polyline points="30.0,137.5 31.5,136.3 33.0,135.2 34.5,134.0 36.0,132.9 37.5,131.7 39.0,130.6 40.5,129.6 42.0,128.6 43.5,127.6 45.0,126.6 46.5,125.7 48.0,124.9 49.5,124.1 51.0,123.4 52.5,122.7 54.0,122.1 55.5,121.5 57.0,121.1 58.5,120.7 60.0,120.3 61.5,120.0 63.0,119.8 64.5,119.7 66.0,119.6 67.5,119.6 69.0,119.7 70.5,119.8 72.0,119.9 73.5,120.2 75.0,120.4 76.5,120.8 78.0,121.1 79.5,121.5 81.0,122.0 82.5,122.5 84.0,123.0 85.5,123.5 87.0,124.0 88.5,124.6 90.0,125.1 91.5,125.7 93.0,126.2 94.5,126.8 96.0,127.3 97.5,127.8 99.0,128.3 100.5,128.7 102.0,129.1 103.5,129.5 105.0,129.8 106.5,130.1 108.0,130.3 109.5,130.5 111.0,130.6 112.5,130.6 114.0,130.6 115.5,130.6 117.0,130.4 118.5,130.2 120.0,129.9 121.5,129.6 123.0,129.2 124.5,128.7 126.0,128.2 127.5,127.6 129.0,126.9 130.5,126.2 132.0,125.4 133.5,124.5 135.0,123.6 136.5,122.7 138.0,121.7 139.5,120.7 141.0,119.6 142.5,118.5 144.0,117.4 145.5,116.2 147.0,115.1 148.5,113.9 150.0,112.8 151.5,111.6 153.0,110.4 154.5,109.3 156.0,108.1 157.5,107.0 159.0,105.9 160.5,104.8 162.0,103.8 163.5,102.8 165.0,101.9 166.5,101.0 168.0,100.1 169.5,99.3 171.0,98.6 172.5,97.9 174.0,97.3 175.5,96.8 177.0,96.3 178.5,95.9 180.0,95.6 181.5,95.3 183.0,95.1 184.5,94.9 186.0,94.9 187.5,94.9 189.0,94.9 190.5,95.0 192.0,95.2 193.5,95.4 195.0,95.7 196.5,96.0 198.0,96.4 199.5,96.8 201.0,97.2 202.5,97.7 204.0,98.2 205.5,98.7 207.0,99.3 208.5,99.8 210.0,100.4 211.5,100.9 213.0,101.5 214.5,102.0 216.0,102.5 217.5,103.0 219.0,103.5 220.5,104.0 222.0,104.4 223.5,104.7 225.0,105.1 226.5,105.3 228.0,105.6 229.5,105.7 231.0,105.8 232.5,105.9 234.0,105.9 235.5,105.8 237.0,105.7 238.5,105.5 240.0,105.2 241.5,104.8 243.0,104.4 244.5,104.0 246.0,103.4 247.5,102.8 249.0,102.1 250.5,101.4 252.0,100.6 253.5,99.8 255.0,98.9 256.5,97.9 258.0,96.9 259.5,95.9 261.0,94.9 262.5,93.8 264.0,92.6 265.5,91.5 267.0,90.3 268.5,89.2 270.0,88.0" fill="none" stroke="#2c3e94" stroke-width="1.1" />
        <polyline points="30.0,110.0 31.5,109.2 33.0,108.5 34.5,107.7 36.0,107.1 37.5,106.4 39.0,105.8 40.5,105.3 42.0,104.8 43.5,104.4 45.0,104.0 46.5,103.6 48.0,103.4 49.5,103.2 51.0,103.1 52.5,103.0 54.0,103.0 55.5,103.0 57.0,103.2 58.5,103.4 60.0,103.7 61.5,104.0 63.0,104.4 64.5,104.9 66.0,105.5 67.5,106.1 69.0,106.8 70.5,107.5 72.0,108.4 73.5,109.2 75.0,110.2 76.5,111.2 78.0,112.2 79.5,113.3 81.0,114.5 82.5,115.6 84.0,116.9 85.5,118.1 87.0,119.4 88.5,120.8 90.0,122.1 91.5,123.5 93.0,124.9 94.5,126.3 96.0,127.7 97.5,129.1 99.0,130.5 100.5,131.9 102.0,133.3 103.5,134.7 105.0,136.1 106.5,137.4 108.0,138.7 109.5,140.0 111.0,141.2 112.5,142.4 114.0,143.6 115.5,144.7 117.0,145.7 118.5,146.7 120.0,147.7 121.5,148.5 123.0,149.4 124.5,150.1 126.0,150.8 127.5,151.4 129.0,151.9 130.5,152.3 132.0,152.7 133.5,153.0 135.0,153.2 136.5,153.3 138.0,153.3 139.5,153.2 141.0,153.1 142.5,152.9 144.0,152.6 145.5,152.2 147.0,151.7 148.5,151.2 150.0,150.5 151.5,149.8 153.0,149.0 154.5,148.2 156.0,147.2 157.5,146.2 159.0,145.2 160.5,144.0 162.0,142.8 163.5,141.6 165.0,140.3 166.5,138.9 168.0,137.5 169.5,136.1 171.0,134.6 172.5,133.1 174.0,131.6 175.5,130.0 177.0,128.5 178.5,126.9 180.0,125.3 181.5,123.7 183.0,122.1 184.5,120.5 186.0,118.9 187.5,117.3 189.0,115.7 190.5,114.2 192.0,112.7 193.5,111.2 195.0,109.7 196.5,108.3 198.0,106.9 199.5,105.6 201.0,104.3 202.5,103.1 204.0,102.0 205.5,100.8 207.0,99.8 208.5,98.8 210.0,97.9 211.5,97.0 213.0,96.2 214.5,95.5 216.0,94.9 217.5,94.3 219.0,93.8 220.5,93.4 222.0,93.1 223.5,92.8 225.0,92.6 226.5,92.5 228.0,92.4 229.5,92.4 231.0,92.5 232.5,92.7 234.0,92.9 235.5,93.2 237.0,93.5 238.5,93.9 240.0,94.4 241.5,94.9 243.0,95.5 244.5,96.1 246.0,96.8 247.5,97.5 249.0,98.2 250.5,99.0 252.0,99.8 253.5,100.6 255.0,101.4 256.5,102.3 258.0,103.1 259.5,104.0 261.0,104.9 262.5,105.8 264.0,106.6 265.5,107.5 267.0,108.4 268.5,109.2 270.0,110.0" fill="none" stroke="#0f9b6c" stroke-width="1.1" />
        <polyline points="30.0,113.9 31.5,113.6 33.0,113.4 34.5,113.2 36.0,113.1 37.5,112.9 39.0,112.8 40.5,112.8 42.0,112.8 43.5,112.8 45.0,112.8 46.5,112.9 48.0,112.9 49.5,113.1 51.0,113.2 52.5,113.4 54.0,113.6 55.5,113.9 57.0,114.2 58.5,114.5 60.0,114.8 61.5,115.2 63.0,115.6 64.5,116.0 66.0,116.4 67.5,116.9 69.0,117.4 70.5,117.9 72.0,118.4 73.5,119.0 75.0,119.5 76.5,120.1 78.0,120.7 79.5,121.4 81.0,122.0 82.5,122.7 84.0,123.3 85.5,124.0 87.0,124.7 88.5,125.4 90.0,126.1 91.5,126.8 93.0,127.5 94.5,128.3 96.0,129.0 97.5,129.7 99.0,130.4 100.5,131.1 102.0,131.9 103.5,132.6 105.0,133.3 106.5,134.0 108.0,134.6 109.5,135.3 111.0,136.0 112.5,136.6 114.0,137.3 115.5,137.9 117.0,138.5 118.5,139.1 120.0,139.7 121.5,140.2 123.0,140.7 124.5,141.3 126.0,141.7 127.5,142.2 129.0,142.6 130.5,143.0 132.0,143.4 133.5,143.8 135.0,144.1 136.5,144.4 138.0,144.7 139.5,144.9 141.0,145.1 142.5,145.3 144.0,145.5 145.5,145.6 147.0,145.7 148.5,145.7 150.0,145.7 151.5,145.7 153.0,145.7 154.5,145.6 156.0,145.5 157.5,145.4 159.0,145.2 160.5,145.0 162.0,144.8 163.5,144.6 165.0,144.3 166.5,144.0 168.0,143.6 169.5,143.3 171.0,142.9 172.5,142.4 174.0,142.0 175.5,141.5 177.0,141.0 178.5,140.5 180.0,140.0 181.5,139.4 183.0,138.8 184.5,138.3 186.0,137.6 187.5,137.0 189.0,136.4 190.5,135.7 192.0,135.0 193.5,134.4 195.0,133.7 196.5,133.0 198.0,132.3 199.5,131.5 201.0,130.8 202.5,130.1 204.0,129.4 205.5,128.7 207.0,127.9 208.5,127.2 210.0,126.5 211.5,125.8 213.0,125.1 214.5,124.4 216.0,123.7 217.5,123.0 219.0,122.4 220.5,121.7 222.0,121.1 223.5,120.5 225.0,119.9 226.5,119.3 228.0,118.7 229.5,118.2 231.0,117.7 232.5,117.2 234.0,116.7 235.5,116.2 237.0,115.8 238.5,115.4 240.0,115.0 241.5,114.7 243.0,114.3 244.5,114.0 246.0,113.8 247.5,113.5 249.0,113.3 250.5,113.2 252.0,113.0 253.5,112.9 255.0,112.8 256.5,112.8 258.0,112.8 259.5,112.8 261.0,112.8 262.5,112.9 264.0,113.0 265.5,113.1 267.0,113.3 268.5,113.5 270.0,113.7" fill="none" stroke="#a86f00" stroke-width="1.1" />
        <polyline points="30.0,99.0 32.4,99.0 34.8,99.0 37.2,98.9 39.6,98.8 42.0,98.7 44.4,98.6 46.8,98.4 49.2,98.2 51.6,98.0 54.0,97.8 56.4,97.5 58.8,97.3 61.2,97.0 63.6,96.6 66.0,96.3 68.4,95.9 70.8,95.5 73.2,95.1 75.6,94.6 78.0,94.2 80.4,93.7 82.8,93.1 85.2,92.6 87.6,92.0 90.0,91.4 92.4,90.8 94.8,90.2 97.2,89.5 99.6,88.8 102.0,88.1 104.4,87.4 106.8,86.6 109.2,85.8 111.6,85.0 114.0,84.2 116.4,83.3 118.8,82.4 121.2,81.5 123.6,80.6 126.0,79.6 128.4,78.7 130.8,77.7 133.2,76.6 135.6,75.6 138.0,74.5 140.4,73.4 142.8,72.3 145.2,71.1 147.6,69.9 150.0,68.8 152.4,67.5 154.8,66.3 157.2,65.0 159.6,63.7 162.0,62.4 164.4,61.1 166.8,59.7 169.2,58.3 171.6,56.9 174.0,55.4 176.4,54.0 178.8,52.5 181.2,51.0 183.6,49.4 186.0,47.9 188.4,46.3 190.8,44.7 193.2,43.0 195.6,41.4 198.0,39.7 200.4,38.0 202.8,36.3 205.2,34.5 207.6,32.7 210.0,30.9 212.4,29.1 214.8,27.3 217.2,25.4 219.6,23.5 222.0,21.6 224.4,19.6 226.8,17.6 229.2,15.6 231.6,13.6 234.0,11.6" fill="none" stroke="#b91c1c" stroke-width="1.2" stroke-dasharray="4,3" />
        <polygon points="234.0,11.6 231.4,17.7 227.5,13.2" fill="#b91c1c" />
        <text x="236" y="40" font-size="9" fill="#b91c1c" text-anchor="end">✗ unbounded: escapes to ∞</text>
        <g font-size="9" fill="#444">
          <text x="12" y="200">one band holds every fₙ at once: the size is controlled</text>
          <text x="12" y="213">✗ without it, fₙ ≡ n runs off to ∞ and nothing can converge</text>
        </g>
      </svg>
    </div>
    <div>
      <div class="panel-label">Equicontinuous: one window tames every graph</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Two smooth graphs with small amber windows of width two delta and height two epsilon placed on them, each window containing its piece of graph; below, a dashed red fast oscillation sin of n x bursts through the top and bottom of such a window">
        <line x1="30" y1="90" x2="270" y2="90" stroke="#bbb" stroke-width="0.8" />
        <polyline points="30.0,69.5 31.5,68.7 33.0,68.0 34.5,67.3 36.0,66.6 37.5,66.0 39.0,65.5 40.5,64.9 42.0,64.5 43.5,64.0 45.0,63.7 46.5,63.3 48.0,63.0 49.5,62.8 51.0,62.6 52.5,62.5 54.0,62.4 55.5,62.4 57.0,62.4 58.5,62.5 60.0,62.6 61.5,62.8 63.0,63.1 64.5,63.4 66.0,63.7 67.5,64.1 69.0,64.5 70.5,65.0 72.0,65.6 73.5,66.1 75.0,66.8 76.5,67.5 78.0,68.2 79.5,68.9 81.0,69.7 82.5,70.6 84.0,71.5 85.5,72.4 87.0,73.3 88.5,74.3 90.0,75.3 91.5,76.4 93.0,77.4 94.5,78.5 96.0,79.6 97.5,80.7 99.0,81.9 100.5,83.0 102.0,84.2 103.5,85.4 105.0,86.6 106.5,87.8 108.0,88.9 109.5,90.1 111.0,91.3 112.5,92.5 114.0,93.7 115.5,94.8 117.0,96.0 118.5,97.1 120.0,98.2 121.5,99.3 123.0,100.4 124.5,101.5 126.0,102.5 127.5,103.5 129.0,104.5 130.5,105.4 132.0,106.3 133.5,107.2 135.0,108.0 136.5,108.8 138.0,109.6 139.5,110.3 141.0,111.0 142.5,111.6 144.0,112.2 145.5,112.7 147.0,113.2 148.5,113.7 150.0,114.1 151.5,114.5 153.0,114.8 154.5,115.0 156.0,115.2 157.5,115.4 159.0,115.5 160.5,115.6 162.0,115.6 163.5,115.6 165.0,115.5 166.5,115.4 168.0,115.3 169.5,115.1 171.0,114.8 172.5,114.5 174.0,114.2 175.5,113.8 177.0,113.4 178.5,113.0 180.0,112.5 181.5,112.0 183.0,111.4 184.5,110.8 186.0,110.2 187.5,109.5 189.0,108.9 190.5,108.2 192.0,107.4 193.5,106.7 195.0,105.9 196.5,105.1 198.0,104.3 199.5,103.5 201.0,102.6 202.5,101.8 204.0,100.9 205.5,100.0 207.0,99.2 208.5,98.3 210.0,97.4 211.5,96.5 213.0,95.6 214.5,94.7 216.0,93.8 217.5,92.9 219.0,92.0 220.5,91.2 222.0,90.3 223.5,89.5 225.0,88.6 226.5,87.8 228.0,87.0 229.5,86.2 231.0,85.4 232.5,84.6 234.0,83.9 235.5,83.1 237.0,82.4 238.5,81.7 240.0,81.1 241.5,80.4 243.0,79.8 244.5,79.2 246.0,78.6 247.5,78.0 249.0,77.5 250.5,77.0 252.0,76.5 253.5,76.1 255.0,75.6 256.5,75.2 258.0,74.8 259.5,74.5 261.0,74.1 262.5,73.8 264.0,73.5 265.5,73.3 267.0,73.1 268.5,72.8 270.0,72.7" fill="none" stroke="#1f2430" stroke-width="1.4" />
        <polyline points="30.0,58.5 31.5,58.5 33.0,58.6 34.5,58.7 36.0,58.8 37.5,59.0 39.0,59.3 40.5,59.6 42.0,59.9 43.5,60.2 45.0,60.6 46.5,61.1 48.0,61.5 49.5,62.0 51.0,62.6 52.5,63.2 54.0,63.8 55.5,64.4 57.0,65.1 58.5,65.8 60.0,66.5 61.5,67.2 63.0,68.0 64.5,68.8 66.0,69.6 67.5,70.4 69.0,71.3 70.5,72.1 72.0,73.0 73.5,73.9 75.0,74.7 76.5,75.6 78.0,76.5 79.5,77.4 81.0,78.3 82.5,79.1 84.0,80.0 85.5,80.9 87.0,81.7 88.5,82.6 90.0,83.4 91.5,84.2 93.0,85.0 94.5,85.8 96.0,86.5 97.5,87.2 99.0,87.9 100.5,88.6 102.0,89.2 103.5,89.8 105.0,90.4 106.5,91.0 108.0,91.5 109.5,91.9 111.0,92.4 112.5,92.8 114.0,93.1 115.5,93.4 117.0,93.7 118.5,94.0 120.0,94.2 121.5,94.3 123.0,94.4 124.5,94.5 126.0,94.5 127.5,94.5 129.0,94.4 130.5,94.3 132.0,94.2 133.5,94.0 135.0,93.7 136.5,93.4 138.0,93.1 139.5,92.8 141.0,92.4 142.5,91.9 144.0,91.5 145.5,91.0 147.0,90.4 148.5,89.8 150.0,89.2 151.5,88.6 153.0,87.9 154.5,87.2 156.0,86.5 157.5,85.8 159.0,85.0 160.5,84.2 162.0,83.4 163.5,82.6 165.0,81.7 166.5,80.9 168.0,80.0 169.5,79.1 171.0,78.3 172.5,77.4 174.0,76.5 175.5,75.6 177.0,74.7 178.5,73.9 180.0,73.0 181.5,72.1 183.0,71.3 184.5,70.4 186.0,69.6 187.5,68.8 189.0,68.0 190.5,67.2 192.0,66.5 193.5,65.8 195.0,65.1 196.5,64.4 198.0,63.8 199.5,63.2 201.0,62.6 202.5,62.0 204.0,61.5 205.5,61.1 207.0,60.6 208.5,60.2 210.0,59.9 211.5,59.6 213.0,59.3 214.5,59.0 216.0,58.8 217.5,58.7 219.0,58.6 220.5,58.5 222.0,58.5 223.5,58.5 225.0,58.6 226.5,58.7 228.0,58.8 229.5,59.0 231.0,59.3 232.5,59.6 234.0,59.9 235.5,60.2 237.0,60.6 238.5,61.1 240.0,61.5 241.5,62.0 243.0,62.6 244.5,63.2 246.0,63.8 247.5,64.4 249.0,65.1 250.5,65.8 252.0,66.5 253.5,67.2 255.0,68.0 256.5,68.8 258.0,69.6 259.5,70.4 261.0,71.3 262.5,72.1 264.0,73.0 265.5,73.9 267.0,74.7 268.5,75.6 270.0,76.5" fill="none" stroke="#2c3e94" stroke-width="1.1" />
        <g fill="rgba(168,111,0,0.13)" stroke="#a86f00" stroke-width="1.1">
          <rect x="59.2" y="51.0" width="28.0" height="30.0" rx="2" />
          <rect x="128.8" y="96.7" width="28.0" height="30.0" rx="2" />
          <rect x="208.0" y="75.3" width="28.0" height="30.0" rx="2" />
          <rect x="167.2" y="57.3" width="28.0" height="30.0" rx="2" />
        </g>
        <line x1="77.2" y1="88.0" x2="83.2" y2="88.0" stroke="#a86f00" stroke-width="1" /><polygon points="87.2,88.0 83.2,90.0 83.2,86.0" fill="#a86f00" /><polygon points="73.2,88.0 77.2,86.0 77.2,90.0" fill="#a86f00" />
        <text x="80.2" y="99.0" font-size="9" fill="#a86f00" text-anchor="middle">δ</text>
        <line x1="94.2" y1="62.0" x2="94.2" y2="55.0" stroke="#a86f00" stroke-width="1" /><polygon points="94.2,51.0 96.2,55.0 92.2,55.0" fill="#a86f00" /><polygon points="94.2,66.0 92.2,62.0 96.2,62.0" fill="#a86f00" />
        <text x="98.2" y="61.5" font-size="9" fill="#a86f00">ε</text>
        <g fill="#a86f00"><circle cx="73.2" cy="66.0" r="2" /></g>
        <text x="268" y="48" font-size="9" fill="#a86f00" text-anchor="end">same window, any fₙ, any x</text>
        <polyline points="160.0,137.9 160.5,137.0 161.0,138.0 161.5,140.7 162.0,145.0 162.5,150.5 163.0,156.6 163.5,162.8 164.0,168.7 164.5,173.5 165.0,177.0 165.5,178.8 166.0,178.7 166.5,176.8 167.0,173.2 167.5,168.2 168.0,162.4 168.5,156.1 169.0,150.0 169.5,144.6 170.0,140.4 170.5,137.8 171.0,137.0 171.5,138.1 172.0,140.9 172.5,145.3 173.0,150.8 173.5,156.9 174.0,163.2 174.5,169.0 175.0,173.8 175.5,177.2 176.0,178.8 176.5,178.7 177.0,176.6 177.5,172.9 178.0,167.9 178.5,162.0 179.0,155.7 179.5,149.7 180.0,144.3 180.5,140.2 181.0,137.7 181.5,137.0 182.0,138.2 182.5,141.1 183.0,145.6 183.5,151.1 184.0,157.3 184.5,163.5 185.0,169.3 185.5,174.0 186.0,177.3 186.5,178.9 187.0,178.6 187.5,176.5 188.0,172.7 188.5,167.6 189.0,161.7 189.5,155.4 190.0,149.3 190.5,144.1 191.0,140.1 191.5,137.6 192.0,137.0 192.5,138.3 193.0,141.3 193.5,145.9 194.0,151.5 194.5,157.6 195.0,163.9 195.5,169.6 196.0,174.2 196.5,177.4 197.0,178.9 197.5,178.5 198.0,176.3 198.5,172.4 199.0,167.3 199.5,161.3 200.0,155.0 200.5,149.0 201.0,143.8 201.5,139.9 202.0,137.5 202.5,137.1 203.0,138.4 203.5,141.6 204.0,146.1 204.5,151.8 205.0,158.0 205.5,164.2 206.0,169.9 206.5,174.4 207.0,177.6 207.5,178.9 208.0,178.5 208.5,176.1 209.0,172.2 209.5,167.0 210.0,161.0 210.5,154.7 211.0,148.7 211.5,143.6 212.0,139.7 212.5,137.5 213.0,137.1 213.5,138.6 214.0,141.8 214.5,146.4 215.0,152.1 215.5,158.4 216.0,164.5 216.5,170.1 217.0,174.7 217.5,177.7 218.0,179.0 218.5,178.4 219.0,175.9 219.5,171.9 220.0,166.7 220.5,160.6 221.0,154.3 221.5,148.4 222.0,143.3 222.5,139.5 223.0,137.4 223.5,137.1 224.0,138.7 224.5,142.0 225.0,146.7 225.5,152.5 226.0,158.7 226.5,164.9 227.0,170.4 227.5,174.9 228.0,177.8 228.5,179.0 229.0,178.3 229.5,175.8 230.0,171.7 230.5,166.3 231.0,160.3 231.5,154.0 232.0,148.1 232.5,143.1 233.0,139.4 233.5,137.3 234.0,137.2 234.5,138.8 235.0,142.2 235.5,147.0 236.0,152.8 236.5,159.1 237.0,165.2 237.5,170.7 238.0,175.1 238.5,177.9 239.0,179.0 239.5,178.2 240.0,175.6 240.5,171.4 241.0,166.0 241.5,159.9 242.0,153.6 242.5,147.8 243.0,142.8 243.5,139.2 244.0,137.3 244.5,137.2 245.0,139.0 245.5,142.5 246.0,147.3 246.5,153.2 247.0,159.4 247.5,165.5 248.0,171.0 248.5,175.3 249.0,178.0 249.5,179.0 250.0,178.1" fill="none" stroke="#b91c1c" stroke-width="1" />
        <rect x="191.0" y="143.0" width="28.0" height="30.0" rx="2" fill="none" stroke="#b91c1c" stroke-width="1.1" stroke-dasharray="3,2" />
        <text x="152" y="162" font-size="9" fill="#b91c1c" text-anchor="end">✗ sin(nx)</text>
        <g font-size="9" fill="#444">
          <text x="12" y="200">one window (width 2δ, height 2ε) fits every fₙ at every x</text>
          <text x="12" y="213">✗ sin(nx) climbs out of any window: the slope escapes to ∞</text>
        </g>
      </svg>
    </div>
  </div>
  <div class="math-figure-row" style="margin-top: 1rem;">
    <div>
      <div class="panel-label">Why it works: finitely many shapes at resolution ε</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="A graph f sampled at grid points x0 to x6 spaced less than delta; a blue polyline joins the samples and a light blue epsilon-tube around the polyline contains the whole graph, inside the band between minus B and B">
        <g stroke="#1565c0" stroke-width="0.8" stroke-dasharray="4,3"><line x1="30" y1="50" x2="270" y2="50" /><line x1="30" y1="170" x2="270" y2="170" /></g>
        <g font-size="9" fill="#1565c0" text-anchor="end"><text x="25" y="54">B</text><text x="25" y="173">−B</text></g>
        <line x1="30" y1="110" x2="270" y2="110" stroke="#bbb" stroke-width="0.8" />
        <g fill="#1565c0" opacity="0.16"><polygon points="30.0,76.2 70.0,65.1 110.0,96.9 150.0,111.7 190.0,133.1 230.0,101.3 270.0,76.2 270.0,100.8 230.0,125.9 190.0,157.6 150.0,136.3 110.0,121.4 70.0,89.7 30.0,100.8" /></g>
        <g stroke="#666" stroke-width="0.8" stroke-dasharray="2,2">
          <line x1="30.0" y1="50" x2="30.0" y2="176" />
          <line x1="70.0" y1="50" x2="70.0" y2="176" />
          <line x1="110.0" y1="50" x2="110.0" y2="176" />
          <line x1="150.0" y1="50" x2="150.0" y2="176" />
          <line x1="190.0" y1="50" x2="190.0" y2="176" />
          <line x1="230.0" y1="50" x2="230.0" y2="176" />
          <line x1="270.0" y1="50" x2="270.0" y2="176" />
        </g>
        <g font-size="8" fill="#666" text-anchor="middle">
          <text x="30.0" y="186">x₀</text>
          <text x="70.0" y="186">x₁</text>
          <text x="110.0" y="186">x₂</text>
          <text x="150.0" y="186">x₃</text>
          <text x="190.0" y="186">x₄</text>
          <text x="230.0" y="186">x₅</text>
          <text x="270.0" y="186">x₆</text>
        </g>
        <polyline points="30.0,88.5 70.0,77.4 110.0,109.2 150.0,124.0 190.0,145.4 230.0,113.6 270.0,88.5" fill="none" stroke="#2c3e94" stroke-width="1.2" />
        <polyline points="30.0,88.5 31.5,86.9 33.0,85.4 34.5,84.0 36.0,82.6 37.5,81.3 39.0,80.0 40.5,78.9 42.0,77.8 43.5,76.8 45.0,75.9 46.5,75.1 48.0,74.4 49.5,73.8 51.0,73.4 52.5,73.0 54.0,72.8 55.5,72.7 57.0,72.7 58.5,72.8 60.0,73.1 61.5,73.4 63.0,73.9 64.5,74.5 66.0,75.1 67.5,75.9 69.0,76.8 70.5,77.7 72.0,78.7 73.5,79.8 75.0,81.0 76.5,82.2 78.0,83.5 79.5,84.8 81.0,86.1 82.5,87.5 84.0,88.9 85.5,90.2 87.0,91.6 88.5,93.0 90.0,94.4 91.5,95.8 93.0,97.1 94.5,98.4 96.0,99.7 97.5,100.9 99.0,102.1 100.5,103.2 102.0,104.3 103.5,105.3 105.0,106.3 106.5,107.2 108.0,108.1 109.5,108.9 111.0,109.7 112.5,110.4 114.0,111.0 115.5,111.7 117.0,112.2 118.5,112.8 120.0,113.3 121.5,113.8 123.0,114.3 124.5,114.7 126.0,115.1 127.5,115.5 129.0,116.0 130.5,116.4 132.0,116.8 133.5,117.3 135.0,117.7 136.5,118.2 138.0,118.7 139.5,119.2 141.0,119.8 142.5,120.4 144.0,121.1 145.5,121.7 147.0,122.5 148.5,123.2 150.0,124.0 151.5,124.9 153.0,125.7 154.5,126.7 156.0,127.6 157.5,128.6 159.0,129.6 160.5,130.6 162.0,131.6 163.5,132.7 165.0,133.7 166.5,134.8 168.0,135.8 169.5,136.8 171.0,137.8 172.5,138.8 174.0,139.7 175.5,140.6 177.0,141.4 178.5,142.2 180.0,142.9 181.5,143.5 183.0,144.0 184.5,144.5 186.0,144.9 187.5,145.1 189.0,145.3 190.5,145.4 192.0,145.3 193.5,145.2 195.0,144.9 196.5,144.5 198.0,144.0 199.5,143.4 201.0,142.7 202.5,141.9 204.0,140.9 205.5,139.9 207.0,138.8 208.5,137.5 210.0,136.2 211.5,134.8 213.0,133.3 214.5,131.8 216.0,130.1 217.5,128.5 219.0,126.8 220.5,125.0 222.0,123.2 223.5,121.4 225.0,119.6 226.5,117.8 228.0,116.0 229.5,114.2 231.0,112.4 232.5,110.7 234.0,109.0 235.5,107.3 237.0,105.7 238.5,104.1 240.0,102.6 241.5,101.2 243.0,99.9 244.5,98.6 246.0,97.4 247.5,96.3 249.0,95.2 250.5,94.3 252.0,93.4 253.5,92.6 255.0,91.9 256.5,91.2 258.0,90.7 259.5,90.2 261.0,89.8 262.5,89.4 264.0,89.1 265.5,88.9 267.0,88.7 268.5,88.6 270.0,88.5" fill="none" stroke="#1f2430" stroke-width="1.4" />
        <g fill="#2c3e94"><circle cx="30.0" cy="88.5" r="3" /><circle cx="70.0" cy="77.4" r="3" /><circle cx="110.0" cy="109.2" r="3" /><circle cx="150.0" cy="124.0" r="3" /><circle cx="190.0" cy="145.4" r="3" /><circle cx="230.0" cy="113.6" r="3" /><circle cx="270.0" cy="88.5" r="3" /></g>
        <text x="262" y="38" font-size="9" fill="#1565c0" text-anchor="end">ε-tube around the polyline</text>
        <text x="115.0" y="101.2" font-size="8.5" fill="#2c3e94">f(x₂)</text>
        <text x="253.2" y="85.7" font-size="10" font-style="italic" fill="#1f2430">f</text>
        <text x="30" y="40" font-size="9" fill="#444">grid spacing &lt; δ</text>
        <g font-size="9" fill="#444">
          <text x="12" y="200">N+1 samples on a δ-grid determine each fₙ to within ε</text>
          <text x="12" y="213">the samples fill a box in ℝᴺ⁺¹: compact by Heine–Borel</text>
        </g>
      </svg>
    </div>
    <div>
      <div class="panel-label">Conclusion: a subsequence settles into an ε-tube</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="A bold limit graph f inside a light green epsilon-tube; four blue graphs of increasing darkness wiggle around it, the faint early ones leaving the tube and the dark later ones staying inside it">
        <line x1="30" y1="110" x2="270" y2="110" stroke="#bbb" stroke-width="0.8" />
        <g fill="#0f9b6c" opacity="0.18"><polygon points="30.0,83.9 31.5,82.7 33.0,81.5 34.5,80.4 36.0,79.3 37.5,78.2 39.0,77.2 40.5,76.2 42.0,75.3 43.5,74.4 45.0,73.5 46.5,72.7 48.0,72.0 49.5,71.3 51.0,70.7 52.5,70.2 54.0,69.7 55.5,69.3 57.0,68.9 58.5,68.6 60.0,68.4 61.5,68.3 63.0,68.2 64.5,68.2 66.0,68.3 67.5,68.4 69.0,68.6 70.5,68.8 72.0,69.2 73.5,69.6 75.0,70.0 76.5,70.5 78.0,71.1 79.5,71.7 81.0,72.4 82.5,73.1 84.0,73.9 85.5,74.7 87.0,75.5 88.5,76.4 90.0,77.4 91.5,78.3 93.0,79.3 94.5,80.3 96.0,81.4 97.5,82.4 99.0,83.5 100.5,84.6 102.0,85.7 103.5,86.8 105.0,87.9 106.5,89.0 108.0,90.1 109.5,91.2 111.0,92.2 112.5,93.3 114.0,94.4 115.5,95.4 117.0,96.4 118.5,97.4 120.0,98.4 121.5,99.3 123.0,100.3 124.5,101.2 126.0,102.0 127.5,102.9 129.0,103.7 130.5,104.5 132.0,105.3 133.5,106.0 135.0,106.7 136.5,107.3 138.0,108.0 139.5,108.6 141.0,109.2 142.5,109.7 144.0,110.3 145.5,110.8 147.0,111.2 148.5,111.7 150.0,112.1 151.5,112.5 153.0,112.9 154.5,113.3 156.0,113.6 157.5,114.0 159.0,114.3 160.5,114.6 162.0,114.9 163.5,115.1 165.0,115.4 166.5,115.7 168.0,115.9 169.5,116.1 171.0,116.4 172.5,116.6 174.0,116.8 175.5,117.0 177.0,117.2 178.5,117.4 180.0,117.6 181.5,117.7 183.0,117.9 184.5,118.1 186.0,118.2 187.5,118.4 189.0,118.5 190.5,118.6 192.0,118.7 193.5,118.8 195.0,118.9 196.5,119.0 198.0,119.0 199.5,119.1 201.0,119.1 202.5,119.1 204.0,119.0 205.5,119.0 207.0,118.9 208.5,118.8 210.0,118.6 211.5,118.5 213.0,118.2 214.5,118.0 216.0,117.7 217.5,117.4 219.0,117.0 220.5,116.6 222.0,116.2 223.5,115.7 225.0,115.2 226.5,114.6 228.0,114.0 229.5,113.4 231.0,112.7 232.5,111.9 234.0,111.2 235.5,110.3 237.0,109.5 238.5,108.6 240.0,107.6 241.5,106.6 243.0,105.6 244.5,104.6 246.0,103.5 247.5,102.3 249.0,101.2 250.5,100.0 252.0,98.8 253.5,97.6 255.0,96.4 256.5,95.1 258.0,93.9 259.5,92.6 261.0,91.4 262.5,90.1 264.0,88.8 265.5,87.6 267.0,86.3 268.5,85.1 270.0,83.9 270.0,107.9 268.5,109.1 267.0,110.3 265.5,111.6 264.0,112.8 262.5,114.1 261.0,115.4 259.5,116.6 258.0,117.9 256.5,119.1 255.0,120.4 253.5,121.6 252.0,122.8 250.5,124.0 249.0,125.2 247.5,126.3 246.0,127.5 244.5,128.6 243.0,129.6 241.5,130.6 240.0,131.6 238.5,132.6 237.0,133.5 235.5,134.3 234.0,135.2 232.5,135.9 231.0,136.7 229.5,137.4 228.0,138.0 226.5,138.6 225.0,139.2 223.5,139.7 222.0,140.2 220.5,140.6 219.0,141.0 217.5,141.4 216.0,141.7 214.5,142.0 213.0,142.2 211.5,142.5 210.0,142.6 208.5,142.8 207.0,142.9 205.5,143.0 204.0,143.0 202.5,143.1 201.0,143.1 199.5,143.1 198.0,143.0 196.5,143.0 195.0,142.9 193.5,142.8 192.0,142.7 190.5,142.6 189.0,142.5 187.5,142.4 186.0,142.2 184.5,142.1 183.0,141.9 181.5,141.7 180.0,141.6 178.5,141.4 177.0,141.2 175.5,141.0 174.0,140.8 172.5,140.6 171.0,140.4 169.5,140.1 168.0,139.9 166.5,139.7 165.0,139.4 163.5,139.1 162.0,138.9 160.5,138.6 159.0,138.3 157.5,138.0 156.0,137.6 154.5,137.3 153.0,136.9 151.5,136.5 150.0,136.1 148.5,135.7 147.0,135.2 145.5,134.8 144.0,134.3 142.5,133.7 141.0,133.2 139.5,132.6 138.0,132.0 136.5,131.3 135.0,130.7 133.5,130.0 132.0,129.3 130.5,128.5 129.0,127.7 127.5,126.9 126.0,126.0 124.5,125.2 123.0,124.3 121.5,123.3 120.0,122.4 118.5,121.4 117.0,120.4 115.5,119.4 114.0,118.4 112.5,117.3 111.0,116.2 109.5,115.2 108.0,114.1 106.5,113.0 105.0,111.9 103.5,110.8 102.0,109.7 100.5,108.6 99.0,107.5 97.5,106.4 96.0,105.4 94.5,104.3 93.0,103.3 91.5,102.3 90.0,101.4 88.5,100.4 87.0,99.5 85.5,98.7 84.0,97.9 82.5,97.1 81.0,96.4 79.5,95.7 78.0,95.1 76.5,94.5 75.0,94.0 73.5,93.6 72.0,93.2 70.5,92.8 69.0,92.6 67.5,92.4 66.0,92.3 64.5,92.2 63.0,92.2 61.5,92.3 60.0,92.4 58.5,92.6 57.0,92.9 55.5,93.3 54.0,93.7 52.5,94.2 51.0,94.7 49.5,95.3 48.0,96.0 46.5,96.7 45.0,97.5 43.5,98.4 42.0,99.3 40.5,100.2 39.0,101.2 37.5,102.2 36.0,103.3 34.5,104.4 33.0,105.5 31.5,106.7 30.0,107.9" /></g>
        <polyline points="30.0,103.6 31.5,104.8 33.0,105.9 34.5,107.0 36.0,107.9 37.5,108.7 39.0,109.4 40.5,109.9 42.0,110.3 43.5,110.4 45.0,110.4 46.5,110.3 48.0,109.9 49.5,109.3 51.0,108.6 52.5,107.7 54.0,106.5 55.5,105.2 57.0,103.8 58.5,102.2 60.0,100.4 61.5,98.6 63.0,96.6 64.5,94.5 66.0,92.4 67.5,90.2 69.0,88.0 70.5,85.8 72.0,83.6 73.5,81.4 75.0,79.3 76.5,77.3 78.0,75.4 79.5,73.6 81.0,72.0 82.5,70.5 84.0,69.3 85.5,68.2 87.0,67.4 88.5,66.7 90.0,66.4 91.5,66.3 93.0,66.4 94.5,66.8 96.0,67.5 97.5,68.4 99.0,69.6 100.5,71.1 102.0,72.8 103.5,74.8 105.0,77.0 106.5,79.4 108.0,82.1 109.5,84.9 111.0,87.9 112.5,91.0 114.0,94.2 115.5,97.6 117.0,101.0 118.5,104.5 120.0,108.0 121.5,111.5 123.0,115.0 124.5,118.4 126.0,121.7 127.5,125.0 129.0,128.1 130.5,131.1 132.0,133.9 133.5,136.5 135.0,138.9 136.5,141.0 138.0,143.0 139.5,144.7 141.0,146.1 142.5,147.3 144.0,148.1 145.5,148.8 147.0,149.1 148.5,149.2 150.0,149.0 151.5,148.5 153.0,147.8 154.5,146.8 156.0,145.6 157.5,144.2 159.0,142.7 160.5,140.9 162.0,139.0 163.5,137.0 165.0,134.8 166.5,132.6 168.0,130.3 169.5,128.0 171.0,125.7 172.5,123.4 174.0,121.1 175.5,118.9 177.0,116.8 178.5,114.8 180.0,113.0 181.5,111.3 183.0,109.7 184.5,108.4 186.0,107.2 187.5,106.3 189.0,105.6 190.5,105.1 192.0,104.9 193.5,104.8 195.0,105.1 196.5,105.5 198.0,106.2 199.5,107.1 201.0,108.2 202.5,109.5 204.0,111.0 205.5,112.7 207.0,114.5 208.5,116.5 210.0,118.5 211.5,120.6 213.0,122.8 214.5,125.1 216.0,127.3 217.5,129.5 219.0,131.7 220.5,133.9 222.0,135.9 223.5,137.8 225.0,139.6 226.5,141.2 228.0,142.6 229.5,143.9 231.0,144.9 232.5,145.6 234.0,146.1 235.5,146.4 237.0,146.4 238.5,146.1 240.0,145.5 241.5,144.6 243.0,143.5 244.5,142.0 246.0,140.3 247.5,138.3 249.0,136.1 250.5,133.6 252.0,130.8 253.5,127.9 255.0,124.8 256.5,121.5 258.0,118.0 259.5,114.4 261.0,110.8 262.5,107.0 264.0,103.2 265.5,99.4 267.0,95.6 268.5,91.9 270.0,88.2" fill="none" stroke="#2c3e94" stroke-width="1" opacity="0.35" />
        <polyline points="30.0,110.7 31.5,109.7 33.0,108.4 34.5,107.0 36.0,105.2 37.5,103.3 39.0,101.1 40.5,98.7 42.0,96.2 43.5,93.6 45.0,90.9 46.5,88.2 48.0,85.4 49.5,82.7 51.0,80.0 52.5,77.5 54.0,75.1 55.5,72.9 57.0,70.9 58.5,69.2 60.0,67.7 61.5,66.6 63.0,65.8 64.5,65.4 66.0,65.3 67.5,65.5 69.0,66.1 70.5,67.0 72.0,68.3 73.5,69.8 75.0,71.7 76.5,73.8 78.0,76.1 79.5,78.6 81.0,81.2 82.5,84.0 84.0,86.8 85.5,89.7 87.0,92.5 88.5,95.3 90.0,98.0 91.5,100.6 93.0,103.0 94.5,105.2 96.0,107.2 97.5,108.9 99.0,110.4 100.5,111.6 102.0,112.5 103.5,113.2 105.0,113.6 106.5,113.7 108.0,113.6 109.5,113.3 111.0,112.7 112.5,112.0 114.0,111.2 115.5,110.2 117.0,109.2 118.5,108.1 120.0,107.1 121.5,106.1 123.0,105.1 124.5,104.3 126.0,103.6 127.5,103.1 129.0,102.7 130.5,102.6 132.0,102.7 133.5,103.1 135.0,103.7 136.5,104.5 138.0,105.7 139.5,107.0 141.0,108.6 142.5,110.4 144.0,112.4 145.5,114.5 147.0,116.8 148.5,119.1 150.0,121.6 151.5,124.0 153.0,126.5 154.5,128.9 156.0,131.2 157.5,133.4 159.0,135.4 160.5,137.2 162.0,138.9 163.5,140.3 165.0,141.4 166.5,142.3 168.0,142.8 169.5,143.1 171.0,143.1 172.5,142.8 174.0,142.3 175.5,141.4 177.0,140.3 178.5,139.1 180.0,137.6 181.5,135.9 183.0,134.2 184.5,132.3 186.0,130.4 187.5,128.5 189.0,126.6 190.5,124.8 192.0,123.1 193.5,121.5 195.0,120.1 196.5,118.8 198.0,117.8 199.5,117.0 201.0,116.4 202.5,116.1 204.0,116.1 205.5,116.3 207.0,116.8 208.5,117.5 210.0,118.4 211.5,119.5 213.0,120.8 214.5,122.2 216.0,123.8 217.5,125.4 219.0,127.1 220.5,128.7 222.0,130.3 223.5,131.9 225.0,133.3 226.5,134.5 228.0,135.6 229.5,136.4 231.0,137.0 232.5,137.3 234.0,137.4 235.5,137.1 237.0,136.4 238.5,135.5 240.0,134.2 241.5,132.7 243.0,130.8 244.5,128.6 246.0,126.2 247.5,123.6 249.0,120.7 250.5,117.7 252.0,114.5 253.5,111.3 255.0,108.0 256.5,104.7 258.0,101.5 259.5,98.3 261.0,95.2 262.5,92.3 264.0,89.6 265.5,87.1 267.0,84.8 268.5,82.8 270.0,81.1" fill="none" stroke="#2c3e94" stroke-width="1" opacity="0.5" />
        <polyline points="30.0,101.5 31.5,99.5 33.0,97.4 34.5,95.2 36.0,92.9 37.5,90.6 39.0,88.4 40.5,86.2 42.0,84.1 43.5,82.1 45.0,80.4 46.5,78.8 48.0,77.5 49.5,76.5 51.0,75.7 52.5,75.2 54.0,75.0 55.5,75.1 57.0,75.4 58.5,76.0 60.0,76.8 61.5,77.7 63.0,78.8 64.5,80.1 66.0,81.4 67.5,82.7 69.0,84.0 70.5,85.3 72.0,86.5 73.5,87.6 75.0,88.6 76.5,89.4 78.0,90.1 79.5,90.6 81.0,90.9 82.5,91.1 84.0,91.2 85.5,91.1 87.0,91.0 88.5,90.7 90.0,90.5 91.5,90.2 93.0,89.9 94.5,89.8 96.0,89.7 97.5,89.8 99.0,90.0 100.5,90.4 102.0,91.0 103.5,91.8 105.0,92.9 106.5,94.1 108.0,95.6 109.5,97.3 111.0,99.1 112.5,101.1 114.0,103.2 115.5,105.4 117.0,107.6 118.5,109.8 120.0,112.0 121.5,114.2 123.0,116.2 124.5,118.0 126.0,119.7 127.5,121.2 129.0,122.4 130.5,123.5 132.0,124.2 133.5,124.8 135.0,125.0 136.5,125.1 138.0,124.9 139.5,124.6 141.0,124.1 142.5,123.5 144.0,122.8 145.5,122.1 147.0,121.3 148.5,120.6 150.0,120.0 151.5,119.5 153.0,119.1 154.5,118.9 156.0,118.8 157.5,119.0 159.0,119.3 160.5,119.9 162.0,120.6 163.5,121.6 165.0,122.7 166.5,123.9 168.0,125.2 169.5,126.6 171.0,128.1 172.5,129.5 174.0,131.0 175.5,132.3 177.0,133.5 178.5,134.6 180.0,135.5 181.5,136.3 183.0,136.8 184.5,137.1 186.0,137.1 187.5,137.0 189.0,136.6 190.5,136.0 192.0,135.3 193.5,134.4 195.0,133.3 196.5,132.2 198.0,131.0 199.5,129.8 201.0,128.7 202.5,127.5 204.0,126.5 205.5,125.6 207.0,124.8 208.5,124.2 210.0,123.7 211.5,123.5 213.0,123.4 214.5,123.5 216.0,123.8 217.5,124.2 219.0,124.7 220.5,125.3 222.0,126.0 223.5,126.8 225.0,127.5 226.5,128.1 228.0,128.7 229.5,129.1 231.0,129.4 232.5,129.5 234.0,129.4 235.5,129.0 237.0,128.4 238.5,127.5 240.0,126.4 241.5,125.0 243.0,123.4 244.5,121.6 246.0,119.6 247.5,117.4 249.0,115.1 250.5,112.7 252.0,110.3 253.5,107.8 255.0,105.5 256.5,103.1 258.0,100.9 259.5,98.9 261.0,97.0 262.5,95.3 264.0,93.9 265.5,92.6 267.0,91.6 268.5,90.8 270.0,90.2" fill="none" stroke="#2c3e94" stroke-width="1" opacity="0.7" />
        <polyline points="30.0,98.2 31.5,97.4 33.0,96.4 34.5,95.4 36.0,94.2 37.5,93.0 39.0,91.6 40.5,90.2 42.0,88.7 43.5,87.2 45.0,85.8 46.5,84.3 48.0,83.0 49.5,81.7 51.0,80.6 52.5,79.7 54.0,78.9 55.5,78.3 57.0,77.9 58.5,77.8 60.0,77.8 61.5,78.0 63.0,78.4 64.5,79.0 66.0,79.7 67.5,80.4 69.0,81.3 70.5,82.1 72.0,83.0 73.5,83.9 75.0,84.7 76.5,85.4 78.0,86.1 79.5,86.7 81.0,87.1 82.5,87.6 84.0,87.9 85.5,88.2 87.0,88.5 88.5,88.7 90.0,89.0 91.5,89.4 93.0,89.8 94.5,90.3 96.0,90.9 97.5,91.6 99.0,92.5 100.5,93.6 102.0,94.8 103.5,96.1 105.0,97.6 106.5,99.2 108.0,100.8 109.5,102.5 111.0,104.2 112.5,105.9 114.0,107.6 115.5,109.2 117.0,110.7 118.5,112.1 120.0,113.3 121.5,114.3 123.0,115.2 124.5,116.0 126.0,116.5 127.5,117.0 129.0,117.3 130.5,117.5 132.0,117.6 133.5,117.7 135.0,117.8 136.5,117.8 138.0,118.0 139.5,118.1 141.0,118.4 142.5,118.8 144.0,119.3 145.5,119.9 147.0,120.6 148.5,121.4 150.0,122.3 151.5,123.2 153.0,124.2 154.5,125.2 156.0,126.2 157.5,127.2 159.0,128.0 160.5,128.8 162.0,129.5 163.5,130.0 165.0,130.4 166.5,130.6 168.0,130.7 169.5,130.7 171.0,130.5 172.5,130.2 174.0,129.8 175.5,129.4 177.0,129.0 178.5,128.5 180.0,128.1 181.5,127.8 183.0,127.5 184.5,127.3 186.0,127.3 187.5,127.4 189.0,127.6 190.5,127.9 192.0,128.4 193.5,128.9 195.0,129.6 196.5,130.2 198.0,130.9 199.5,131.6 201.0,132.2 202.5,132.8 204.0,133.2 205.5,133.6 207.0,133.7 208.5,133.8 210.0,133.6 211.5,133.3 213.0,132.8 214.5,132.2 216.0,131.4 217.5,130.5 219.0,129.5 220.5,128.5 222.0,127.4 223.5,126.3 225.0,125.3 226.5,124.2 228.0,123.3 229.5,122.4 231.0,121.7 232.5,121.0 234.0,120.4 235.5,119.9 237.0,119.5 238.5,119.1 240.0,118.8 241.5,118.4 243.0,118.1 244.5,117.6 246.0,117.1 247.5,116.5 249.0,115.8 250.5,114.9 252.0,113.8 253.5,112.6 255.0,111.2 256.5,109.7 258.0,108.1 259.5,106.4 261.0,104.5 262.5,102.6 264.0,100.7 265.5,98.8 267.0,97.0 268.5,95.2 270.0,93.5" fill="none" stroke="#2c3e94" stroke-width="1" opacity="0.9" />
        <polyline points="30.0,95.9 31.5,94.7 33.0,93.5 34.5,92.4 36.0,91.3 37.5,90.2 39.0,89.2 40.5,88.2 42.0,87.3 43.5,86.4 45.0,85.5 46.5,84.7 48.0,84.0 49.5,83.3 51.0,82.7 52.5,82.2 54.0,81.7 55.5,81.3 57.0,80.9 58.5,80.6 60.0,80.4 61.5,80.3 63.0,80.2 64.5,80.2 66.0,80.3 67.5,80.4 69.0,80.6 70.5,80.8 72.0,81.2 73.5,81.6 75.0,82.0 76.5,82.5 78.0,83.1 79.5,83.7 81.0,84.4 82.5,85.1 84.0,85.9 85.5,86.7 87.0,87.5 88.5,88.4 90.0,89.4 91.5,90.3 93.0,91.3 94.5,92.3 96.0,93.4 97.5,94.4 99.0,95.5 100.5,96.6 102.0,97.7 103.5,98.8 105.0,99.9 106.5,101.0 108.0,102.1 109.5,103.2 111.0,104.2 112.5,105.3 114.0,106.4 115.5,107.4 117.0,108.4 118.5,109.4 120.0,110.4 121.5,111.3 123.0,112.3 124.5,113.2 126.0,114.0 127.5,114.9 129.0,115.7 130.5,116.5 132.0,117.3 133.5,118.0 135.0,118.7 136.5,119.3 138.0,120.0 139.5,120.6 141.0,121.2 142.5,121.7 144.0,122.3 145.5,122.8 147.0,123.2 148.5,123.7 150.0,124.1 151.5,124.5 153.0,124.9 154.5,125.3 156.0,125.6 157.5,126.0 159.0,126.3 160.5,126.6 162.0,126.9 163.5,127.1 165.0,127.4 166.5,127.7 168.0,127.9 169.5,128.1 171.0,128.4 172.5,128.6 174.0,128.8 175.5,129.0 177.0,129.2 178.5,129.4 180.0,129.6 181.5,129.7 183.0,129.9 184.5,130.1 186.0,130.2 187.5,130.4 189.0,130.5 190.5,130.6 192.0,130.7 193.5,130.8 195.0,130.9 196.5,131.0 198.0,131.0 199.5,131.1 201.0,131.1 202.5,131.1 204.0,131.0 205.5,131.0 207.0,130.9 208.5,130.8 210.0,130.6 211.5,130.5 213.0,130.2 214.5,130.0 216.0,129.7 217.5,129.4 219.0,129.0 220.5,128.6 222.0,128.2 223.5,127.7 225.0,127.2 226.5,126.6 228.0,126.0 229.5,125.4 231.0,124.7 232.5,123.9 234.0,123.2 235.5,122.3 237.0,121.5 238.5,120.6 240.0,119.6 241.5,118.6 243.0,117.6 244.5,116.6 246.0,115.5 247.5,114.3 249.0,113.2 250.5,112.0 252.0,110.8 253.5,109.6 255.0,108.4 256.5,107.1 258.0,105.9 259.5,104.6 261.0,103.4 262.5,102.1 264.0,100.8 265.5,99.6 267.0,98.3 268.5,97.1 270.0,95.9" fill="none" stroke="#1f2430" stroke-width="2" />
        <text x="274.0" y="99.9" font-size="10" font-style="italic" fill="#1f2430">f</text>
        <text x="30" y="40" font-size="9" fill="#0f9b6c">ε-tube around the limit f</text>
        <text x="262" y="40" font-size="9" fill="#2c3e94" text-anchor="end">fₙ₁ (faint) … fₙ₄ (dark)</text>
        <g font-size="9" fill="#666" text-anchor="middle"><text x="30" y="184">a</text><text x="270" y="184">b</text></g>
        <g font-size="9" fill="#444">
          <text x="12" y="200">Bolzano–Weierstrass at every grid point + diagonal trick</text>
          <text x="12" y="213">equicontinuity fills the gaps: ‖fₘ − f‖∞ → 0 along it, uniformly</text>
        </g>
      </svg>
    </div>
  </div>
  <figcaption>Arzelà–Ascoli as a picture. A sequence in $C([a, b])$ can fail to have a uniformly convergent subsequence in exactly two ways: the graphs can run off to infinity (size), or they can wiggle ever faster (shape). <strong>Uniform boundedness</strong> forbids the first: one horizontal band $[-B, B]$ holds every $f_n$. <strong>Equicontinuity</strong> forbids the second: one window of width $2\delta$ and height $2\varepsilon$, slid to any point of any graph, always contains the graph, so no $f_n$ can climb by more than $\varepsilon$ within a $\delta$-step. The family $\sin(nx)$, whose oscillation bursts through every such window, is exactly what this hypothesis excludes. <strong>Why these two suffice.</strong> Sample each $f_n$ on a grid of spacing less than $\delta$. By equicontinuity the samples pin the function down to within $\varepsilon$ everywhere, so at resolution $\varepsilon$ the whole family is described by $N + 1$ numbers in $[-B, B]$, a compact box in $\mathbb{R}^{N+1}$: the family is totally bounded in $C([a, b])$. Bolzano–Weierstrass at each grid point and a diagonal argument give a subsequence converging at every grid point, and equicontinuity upgrades that to convergence inside an $\varepsilon$-tube around the limit, that is, in the sup norm. The theorem is the function-space Heine–Borel: <em>bounded</em> controls size and <em>equicontinuous</em> controls the escape to high frequency, exactly as <em>equi-small tails</em> did in Theorem 197. The same family $\sin(nx)$ reappears as an orthonormal sequence in the section on weak convergence below, where it converges weakly but not in norm. The converse holds as well: a compact subset of $C([a, b])$ is uniformly bounded and equicontinuous.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compact vs. Non-Compact Domains)</span></p>

The formulation above assumes that the domain $[a,b]$ is compact. Hence the subsequence obtained by Arzelà--Ascoli converges **uniformly on the whole domain**:

$$\lVert f_{n_j}-f\rVert_\infty = \sup_{x\in[a,b]} |f_{n_j}(x)-f(x)| \longrightarrow 0.$$

For a non-compact domain $X$, one commonly uses a local version of Arzelà--Ascoli. Under suitable local boundedness and equicontinuity assumptions, one obtains a subsequence that converges **uniformly on every compact subset** $K\subset X$:

$$\forall K\subset X \text{ compact}, \qquad \sup_{x\in K}|f_{n_j}(x)-f(x)| \longrightarrow 0.$$

This mode of convergence is called **locally uniform convergence**, or **uniform convergence on compact subsets**.

Thus,

$$\boxed{X\text{ compact} \quad\Longrightarrow\quad \text{locally uniform convergence} = \text{uniform convergence}.}$$

When $X$ is non-compact, locally uniform convergence is generally strictly weaker than uniform convergence on all of $X$.

A common proof strategy on a non-compact domain is to choose an increasing sequence of compact sets

$$K_1\subset K_2\subset\cdots, \qquad \bigcup_{m=1}^{\infty}K_m=X,$$

apply Arzelà--Ascoli successively on each $K_m$, and then use a diagonal subsequence. The resulting subsequence converges uniformly on every $K_m$, and hence locally uniformly on $X$.

In this language, the generalized theorem can be viewed as a compactness criterion in $C_{\mathrm{loc}}(X)$, whose natural topology is uniform convergence on compact subsets.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Local version of Arzelà-Ascoli: Locally Uniform Convergence)</span></p>

Consider the function $f_n(x)=\tanh(x/n)$,  $x\in\mathbb R$. For any fixed compact interval $[-R,R]$,

$$\sup_{|x|\le R}|\tanh(x/n)| \le \tanh(R/n)\to0,$$

so $f_n\to0$ uniformly on every compact subset of $\mathbb R$. But globally,

$$\sup_{x\in\mathbb R}|\tanh(x/n)|=1$$

for every $n$, so convergence is **not uniform on $\mathbb R$**.

</div>

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">fₙ = tanh(x/n): flat on every window, never on ℝ</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Graphs of tanh of x over n for n equal 1, 2, 4, 8, 16 drawn from faint to dark; they flatten inside a shaded compact window from minus R to R but each still climbs to the dashed asymptotes at plus and minus one, with red dots marking where each graph reaches three quarters drifting to the right toward infinity">
        <rect x="120" y="41.8" width="60" height="126.5" fill="#fff7e0" stroke="#a86f00" stroke-width="1" stroke-dasharray="4,3" />
        <text x="150" y="181.2" font-size="8.5" fill="#a86f00" text-anchor="middle">K = [−R, R]</text>
        <g stroke="#b91c1c" stroke-width="0.9" stroke-dasharray="3,3"><line x1="30" y1="50" x2="270" y2="50" /><line x1="30" y1="160" x2="270" y2="160" /></g>
        <g font-size="8.5" fill="#b91c1c"><text x="26" y="53" text-anchor="end">1</text><text x="26" y="163" text-anchor="end">−1</text></g>
        <text x="268" y="44.0" font-size="8.5" fill="#b91c1c" text-anchor="end">sup over ℝ = 1 for every n</text>
        <line x1="28.0" y1="105.0" x2="270.0" y2="105.0" stroke="#444" stroke-width="1" /><polygon points="276.0,105.0 270.0,108.0 270.0,102.0" fill="#444" />
        <line x1="150.0" y1="171.0" x2="150.0" y2="39.5" stroke="#444" stroke-width="1" /><polygon points="150.0,33.5 153.0,39.5 147.0,39.5" fill="#444" />
        <text x="278" y="109" font-size="10" fill="#666">x</text>
        <g stroke="#444" stroke-width="1"><line x1="50" y1="102" x2="50" y2="108" /><line x1="100" y1="102" x2="100" y2="108" /><line x1="200" y1="102" x2="200" y2="108" /><line x1="250" y1="102" x2="250" y2="108" /></g>
        <g font-size="7.5" fill="#666" text-anchor="middle"><text x="50" y="117">-10</text><text x="100" y="117">-5</text><text x="200" y="117">5</text><text x="250" y="117">10</text></g>
        <polyline points="30.0,160.0 31.0,160.0 32.0,160.0 33.0,160.0 34.0,160.0 35.0,160.0 36.0,160.0 37.0,160.0 38.0,160.0 39.0,160.0 40.0,160.0 41.0,160.0 42.0,160.0 43.0,160.0 44.0,160.0 45.0,160.0 46.0,160.0 47.0,160.0 48.0,160.0 49.0,160.0 50.0,160.0 51.0,160.0 52.0,160.0 53.0,160.0 54.0,160.0 55.0,160.0 56.0,160.0 57.0,160.0 58.0,160.0 59.0,160.0 60.0,160.0 61.0,160.0 62.0,160.0 63.0,160.0 64.0,160.0 65.0,160.0 66.0,160.0 67.0,160.0 68.0,160.0 69.0,160.0 70.0,160.0 71.0,160.0 72.0,160.0 73.0,160.0 74.0,160.0 75.0,160.0 76.0,160.0 77.0,160.0 78.0,160.0 79.0,160.0 80.0,160.0 81.0,160.0 82.0,160.0 83.0,160.0 84.0,160.0 85.0,160.0 86.0,160.0 87.0,160.0 88.0,160.0 89.0,160.0 90.0,160.0 91.0,160.0 92.0,160.0 93.0,160.0 94.0,160.0 95.0,160.0 96.0,160.0 97.0,160.0 98.0,160.0 99.0,160.0 100.0,160.0 101.0,160.0 102.0,160.0 103.0,160.0 104.0,160.0 105.0,160.0 106.0,160.0 107.0,160.0 108.0,160.0 109.0,160.0 110.0,160.0 111.0,160.0 112.0,159.9 113.0,159.9 114.0,159.9 115.0,159.9 116.0,159.9 117.0,159.9 118.0,159.8 119.0,159.8 120.0,159.7 121.0,159.7 122.0,159.6 123.0,159.5 124.0,159.4 125.0,159.3 126.0,159.1 127.0,158.9 128.0,158.7 129.0,158.4 130.0,158.0 131.0,157.6 132.0,157.1 133.0,156.4 134.0,155.7 135.0,154.8 136.0,153.7 137.0,152.4 138.0,150.9 139.0,149.0 140.0,146.9 141.0,144.4 142.0,141.5 143.0,138.2 144.0,134.5 145.0,130.4 146.0,125.9 147.0,121.0 148.0,115.9 149.0,110.5 150.0,105.0 151.0,99.5 152.0,94.1 153.0,89.0 154.0,84.1 155.0,79.6 156.0,75.5 157.0,71.8 158.0,68.5 159.0,65.6 160.0,63.1 161.0,61.0 162.0,59.1 163.0,57.6 164.0,56.3 165.0,55.2 166.0,54.3 167.0,53.6 168.0,52.9 169.0,52.4 170.0,52.0 171.0,51.6 172.0,51.3 173.0,51.1 174.0,50.9 175.0,50.7 176.0,50.6 177.0,50.5 178.0,50.4 179.0,50.3 180.0,50.3 181.0,50.2 182.0,50.2 183.0,50.1 184.0,50.1 185.0,50.1 186.0,50.1 187.0,50.1 188.0,50.1 189.0,50.0 190.0,50.0 191.0,50.0 192.0,50.0 193.0,50.0 194.0,50.0 195.0,50.0 196.0,50.0 197.0,50.0 198.0,50.0 199.0,50.0 200.0,50.0 201.0,50.0 202.0,50.0 203.0,50.0 204.0,50.0 205.0,50.0 206.0,50.0 207.0,50.0 208.0,50.0 209.0,50.0 210.0,50.0 211.0,50.0 212.0,50.0 213.0,50.0 214.0,50.0 215.0,50.0 216.0,50.0 217.0,50.0 218.0,50.0 219.0,50.0 220.0,50.0 221.0,50.0 222.0,50.0 223.0,50.0 224.0,50.0 225.0,50.0 226.0,50.0 227.0,50.0 228.0,50.0 229.0,50.0 230.0,50.0 231.0,50.0 232.0,50.0 233.0,50.0 234.0,50.0 235.0,50.0 236.0,50.0 237.0,50.0 238.0,50.0 239.0,50.0 240.0,50.0 241.0,50.0 242.0,50.0 243.0,50.0 244.0,50.0 245.0,50.0 246.0,50.0 247.0,50.0 248.0,50.0 249.0,50.0 250.0,50.0 251.0,50.0 252.0,50.0 253.0,50.0 254.0,50.0 255.0,50.0 256.0,50.0 257.0,50.0 258.0,50.0 259.0,50.0 260.0,50.0 261.0,50.0 262.0,50.0 263.0,50.0 264.0,50.0 265.0,50.0 266.0,50.0 267.0,50.0 268.0,50.0 269.0,50.0 270.0,50.0" fill="none" stroke="#2c3e94" stroke-width="1.3" opacity="0.25" />
        <polyline points="30.0,160.0 31.0,160.0 32.0,160.0 33.0,160.0 34.0,160.0 35.0,160.0 36.0,160.0 37.0,160.0 38.0,160.0 39.0,160.0 40.0,160.0 41.0,160.0 42.0,160.0 43.0,160.0 44.0,160.0 45.0,160.0 46.0,160.0 47.0,160.0 48.0,160.0 49.0,160.0 50.0,160.0 51.0,160.0 52.0,160.0 53.0,160.0 54.0,160.0 55.0,160.0 56.0,160.0 57.0,160.0 58.0,160.0 59.0,160.0 60.0,160.0 61.0,160.0 62.0,160.0 63.0,160.0 64.0,160.0 65.0,160.0 66.0,160.0 67.0,160.0 68.0,160.0 69.0,160.0 70.0,160.0 71.0,160.0 72.0,160.0 73.0,160.0 74.0,159.9 75.0,159.9 76.0,159.9 77.0,159.9 78.0,159.9 79.0,159.9 80.0,159.9 81.0,159.9 82.0,159.9 83.0,159.9 84.0,159.9 85.0,159.8 86.0,159.8 87.0,159.8 88.0,159.8 89.0,159.8 90.0,159.7 91.0,159.7 92.0,159.7 93.0,159.6 94.0,159.6 95.0,159.6 96.0,159.5 97.0,159.5 98.0,159.4 99.0,159.3 100.0,159.3 101.0,159.2 102.0,159.1 103.0,159.0 104.0,158.9 105.0,158.8 106.0,158.7 107.0,158.5 108.0,158.4 109.0,158.2 110.0,158.0 111.0,157.8 112.0,157.6 113.0,157.3 114.0,157.1 115.0,156.8 116.0,156.4 117.0,156.1 118.0,155.7 119.0,155.3 120.0,154.8 121.0,154.3 122.0,153.7 123.0,153.1 124.0,152.4 125.0,151.7 126.0,150.9 127.0,150.0 128.0,149.0 129.0,148.0 130.0,146.9 131.0,145.7 132.0,144.4 133.0,143.0 134.0,141.5 135.0,139.9 136.0,138.2 137.0,136.4 138.0,134.5 139.0,132.5 140.0,130.4 141.0,128.2 142.0,125.9 143.0,123.5 144.0,121.0 145.0,118.5 146.0,115.9 147.0,113.2 148.0,110.5 149.0,107.7 150.0,105.0 151.0,102.3 152.0,99.5 153.0,96.8 154.0,94.1 155.0,91.5 156.0,89.0 157.0,86.5 158.0,84.1 159.0,81.8 160.0,79.6 161.0,77.5 162.0,75.5 163.0,73.6 164.0,71.8 165.0,70.1 166.0,68.5 167.0,67.0 168.0,65.6 169.0,64.3 170.0,63.1 171.0,62.0 172.0,61.0 173.0,60.0 174.0,59.1 175.0,58.3 176.0,57.6 177.0,56.9 178.0,56.3 179.0,55.7 180.0,55.2 181.0,54.7 182.0,54.3 183.0,53.9 184.0,53.6 185.0,53.2 186.0,52.9 187.0,52.7 188.0,52.4 189.0,52.2 190.0,52.0 191.0,51.8 192.0,51.6 193.0,51.5 194.0,51.3 195.0,51.2 196.0,51.1 197.0,51.0 198.0,50.9 199.0,50.8 200.0,50.7 201.0,50.7 202.0,50.6 203.0,50.5 204.0,50.5 205.0,50.4 206.0,50.4 207.0,50.4 208.0,50.3 209.0,50.3 210.0,50.3 211.0,50.2 212.0,50.2 213.0,50.2 214.0,50.2 215.0,50.2 216.0,50.1 217.0,50.1 218.0,50.1 219.0,50.1 220.0,50.1 221.0,50.1 222.0,50.1 223.0,50.1 224.0,50.1 225.0,50.1 226.0,50.1 227.0,50.0 228.0,50.0 229.0,50.0 230.0,50.0 231.0,50.0 232.0,50.0 233.0,50.0 234.0,50.0 235.0,50.0 236.0,50.0 237.0,50.0 238.0,50.0 239.0,50.0 240.0,50.0 241.0,50.0 242.0,50.0 243.0,50.0 244.0,50.0 245.0,50.0 246.0,50.0 247.0,50.0 248.0,50.0 249.0,50.0 250.0,50.0 251.0,50.0 252.0,50.0 253.0,50.0 254.0,50.0 255.0,50.0 256.0,50.0 257.0,50.0 258.0,50.0 259.0,50.0 260.0,50.0 261.0,50.0 262.0,50.0 263.0,50.0 264.0,50.0 265.0,50.0 266.0,50.0 267.0,50.0 268.0,50.0 269.0,50.0 270.0,50.0" fill="none" stroke="#2c3e94" stroke-width="1.3" opacity="0.4" />
        <polyline points="30.0,159.7 31.0,159.7 32.0,159.7 33.0,159.7 34.0,159.7 35.0,159.7 36.0,159.6 37.0,159.6 38.0,159.6 39.0,159.6 40.0,159.6 41.0,159.5 42.0,159.5 43.0,159.5 44.0,159.5 45.0,159.4 46.0,159.4 47.0,159.4 48.0,159.3 49.0,159.3 50.0,159.3 51.0,159.2 52.0,159.2 53.0,159.1 54.0,159.1 55.0,159.1 56.0,159.0 57.0,159.0 58.0,158.9 59.0,158.8 60.0,158.8 61.0,158.7 62.0,158.7 63.0,158.6 64.0,158.5 65.0,158.5 66.0,158.4 67.0,158.3 68.0,158.2 69.0,158.1 70.0,158.0 71.0,157.9 72.0,157.8 73.0,157.7 74.0,157.6 75.0,157.5 76.0,157.3 77.0,157.2 78.0,157.1 79.0,156.9 80.0,156.8 81.0,156.6 82.0,156.4 83.0,156.3 84.0,156.1 85.0,155.9 86.0,155.7 87.0,155.5 88.0,155.3 89.0,155.0 90.0,154.8 91.0,154.5 92.0,154.3 93.0,154.0 94.0,153.7 95.0,153.4 96.0,153.1 97.0,152.7 98.0,152.4 99.0,152.0 100.0,151.7 101.0,151.3 102.0,150.9 103.0,150.4 104.0,150.0 105.0,149.5 106.0,149.0 107.0,148.5 108.0,148.0 109.0,147.5 110.0,146.9 111.0,146.3 112.0,145.7 113.0,145.1 114.0,144.4 115.0,143.7 116.0,143.0 117.0,142.3 118.0,141.5 119.0,140.7 120.0,139.9 121.0,139.1 122.0,138.2 123.0,137.4 124.0,136.4 125.0,135.5 126.0,134.5 127.0,133.5 128.0,132.5 129.0,131.5 130.0,130.4 131.0,129.3 132.0,128.2 133.0,127.1 134.0,125.9 135.0,124.7 136.0,123.5 137.0,122.3 138.0,121.0 139.0,119.8 140.0,118.5 141.0,117.2 142.0,115.9 143.0,114.5 144.0,113.2 145.0,111.8 146.0,110.5 147.0,109.1 148.0,107.7 149.0,106.4 150.0,105.0 151.0,103.6 152.0,102.3 153.0,100.9 154.0,99.5 155.0,98.2 156.0,96.8 157.0,95.5 158.0,94.1 159.0,92.8 160.0,91.5 161.0,90.2 162.0,89.0 163.0,87.7 164.0,86.5 165.0,85.3 166.0,84.1 167.0,82.9 168.0,81.8 169.0,80.7 170.0,79.6 171.0,78.5 172.0,77.5 173.0,76.5 174.0,75.5 175.0,74.5 176.0,73.6 177.0,72.6 178.0,71.8 179.0,70.9 180.0,70.1 181.0,69.3 182.0,68.5 183.0,67.7 184.0,67.0 185.0,66.3 186.0,65.6 187.0,64.9 188.0,64.3 189.0,63.7 190.0,63.1 191.0,62.5 192.0,62.0 193.0,61.5 194.0,61.0 195.0,60.5 196.0,60.0 197.0,59.6 198.0,59.1 199.0,58.7 200.0,58.3 201.0,58.0 202.0,57.6 203.0,57.3 204.0,56.9 205.0,56.6 206.0,56.3 207.0,56.0 208.0,55.7 209.0,55.5 210.0,55.2 211.0,55.0 212.0,54.7 213.0,54.5 214.0,54.3 215.0,54.1 216.0,53.9 217.0,53.7 218.0,53.6 219.0,53.4 220.0,53.2 221.0,53.1 222.0,52.9 223.0,52.8 224.0,52.7 225.0,52.5 226.0,52.4 227.0,52.3 228.0,52.2 229.0,52.1 230.0,52.0 231.0,51.9 232.0,51.8 233.0,51.7 234.0,51.6 235.0,51.5 236.0,51.5 237.0,51.4 238.0,51.3 239.0,51.3 240.0,51.2 241.0,51.2 242.0,51.1 243.0,51.0 244.0,51.0 245.0,50.9 246.0,50.9 247.0,50.9 248.0,50.8 249.0,50.8 250.0,50.7 251.0,50.7 252.0,50.7 253.0,50.6 254.0,50.6 255.0,50.6 256.0,50.5 257.0,50.5 258.0,50.5 259.0,50.5 260.0,50.4 261.0,50.4 262.0,50.4 263.0,50.4 264.0,50.4 265.0,50.3 266.0,50.3 267.0,50.3 268.0,50.3 269.0,50.3 270.0,50.3" fill="none" stroke="#2c3e94" stroke-width="1.3" opacity="0.55" />
        <polyline points="30.0,154.8 31.0,154.7 32.0,154.5 33.0,154.4 34.0,154.3 35.0,154.1 36.0,154.0 37.0,153.8 38.0,153.7 39.0,153.5 40.0,153.4 41.0,153.2 42.0,153.1 43.0,152.9 44.0,152.7 45.0,152.6 46.0,152.4 47.0,152.2 48.0,152.0 49.0,151.8 50.0,151.7 51.0,151.5 52.0,151.3 53.0,151.1 54.0,150.9 55.0,150.6 56.0,150.4 57.0,150.2 58.0,150.0 59.0,149.7 60.0,149.5 61.0,149.3 62.0,149.0 63.0,148.8 64.0,148.5 65.0,148.3 66.0,148.0 67.0,147.7 68.0,147.5 69.0,147.2 70.0,146.9 71.0,146.6 72.0,146.3 73.0,146.0 74.0,145.7 75.0,145.4 76.0,145.1 77.0,144.7 78.0,144.4 79.0,144.1 80.0,143.7 81.0,143.4 82.0,143.0 83.0,142.6 84.0,142.3 85.0,141.9 86.0,141.5 87.0,141.1 88.0,140.7 89.0,140.3 90.0,139.9 91.0,139.5 92.0,139.1 93.0,138.7 94.0,138.2 95.0,137.8 96.0,137.4 97.0,136.9 98.0,136.4 99.0,136.0 100.0,135.5 101.0,135.0 102.0,134.5 103.0,134.0 104.0,133.5 105.0,133.0 106.0,132.5 107.0,132.0 108.0,131.5 109.0,131.0 110.0,130.4 111.0,129.9 112.0,129.3 113.0,128.8 114.0,128.2 115.0,127.6 116.0,127.1 117.0,126.5 118.0,125.9 119.0,125.3 120.0,124.7 121.0,124.1 122.0,123.5 123.0,122.9 124.0,122.3 125.0,121.6 126.0,121.0 127.0,120.4 128.0,119.8 129.0,119.1 130.0,118.5 131.0,117.8 132.0,117.2 133.0,116.5 134.0,115.9 135.0,115.2 136.0,114.5 137.0,113.9 138.0,113.2 139.0,112.5 140.0,111.8 141.0,111.2 142.0,110.5 143.0,109.8 144.0,109.1 145.0,108.4 146.0,107.7 147.0,107.1 148.0,106.4 149.0,105.7 150.0,105.0 151.0,104.3 152.0,103.6 153.0,102.9 154.0,102.3 155.0,101.6 156.0,100.9 157.0,100.2 158.0,99.5 159.0,98.8 160.0,98.2 161.0,97.5 162.0,96.8 163.0,96.1 164.0,95.5 165.0,94.8 166.0,94.1 167.0,93.5 168.0,92.8 169.0,92.2 170.0,91.5 171.0,90.9 172.0,90.2 173.0,89.6 174.0,89.0 175.0,88.4 176.0,87.7 177.0,87.1 178.0,86.5 179.0,85.9 180.0,85.3 181.0,84.7 182.0,84.1 183.0,83.5 184.0,82.9 185.0,82.4 186.0,81.8 187.0,81.2 188.0,80.7 189.0,80.1 190.0,79.6 191.0,79.0 192.0,78.5 193.0,78.0 194.0,77.5 195.0,77.0 196.0,76.5 197.0,76.0 198.0,75.5 199.0,75.0 200.0,74.5 201.0,74.0 202.0,73.6 203.0,73.1 204.0,72.6 205.0,72.2 206.0,71.8 207.0,71.3 208.0,70.9 209.0,70.5 210.0,70.1 211.0,69.7 212.0,69.3 213.0,68.9 214.0,68.5 215.0,68.1 216.0,67.7 217.0,67.4 218.0,67.0 219.0,66.6 220.0,66.3 221.0,65.9 222.0,65.6 223.0,65.3 224.0,64.9 225.0,64.6 226.0,64.3 227.0,64.0 228.0,63.7 229.0,63.4 230.0,63.1 231.0,62.8 232.0,62.5 233.0,62.3 234.0,62.0 235.0,61.7 236.0,61.5 237.0,61.2 238.0,61.0 239.0,60.7 240.0,60.5 241.0,60.3 242.0,60.0 243.0,59.8 244.0,59.6 245.0,59.4 246.0,59.1 247.0,58.9 248.0,58.7 249.0,58.5 250.0,58.3 251.0,58.2 252.0,58.0 253.0,57.8 254.0,57.6 255.0,57.4 256.0,57.3 257.0,57.1 258.0,56.9 259.0,56.8 260.0,56.6 261.0,56.5 262.0,56.3 263.0,56.2 264.0,56.0 265.0,55.9 266.0,55.7 267.0,55.6 268.0,55.5 269.0,55.3 270.0,55.2" fill="none" stroke="#2c3e94" stroke-width="1.3" opacity="0.75" />
        <polyline points="30.0,139.9 31.0,139.7 32.0,139.5 33.0,139.3 34.0,139.1 35.0,138.9 36.0,138.7 37.0,138.5 38.0,138.2 39.0,138.0 40.0,137.8 41.0,137.6 42.0,137.4 43.0,137.1 44.0,136.9 45.0,136.7 46.0,136.4 47.0,136.2 48.0,136.0 49.0,135.7 50.0,135.5 51.0,135.3 52.0,135.0 53.0,134.8 54.0,134.5 55.0,134.3 56.0,134.0 57.0,133.8 58.0,133.5 59.0,133.3 60.0,133.0 61.0,132.8 62.0,132.5 63.0,132.3 64.0,132.0 65.0,131.7 66.0,131.5 67.0,131.2 68.0,131.0 69.0,130.7 70.0,130.4 71.0,130.1 72.0,129.9 73.0,129.6 74.0,129.3 75.0,129.0 76.0,128.8 77.0,128.5 78.0,128.2 79.0,127.9 80.0,127.6 81.0,127.4 82.0,127.1 83.0,126.8 84.0,126.5 85.0,126.2 86.0,125.9 87.0,125.6 88.0,125.3 89.0,125.0 90.0,124.7 91.0,124.4 92.0,124.1 93.0,123.8 94.0,123.5 95.0,123.2 96.0,122.9 97.0,122.6 98.0,122.3 99.0,122.0 100.0,121.6 101.0,121.3 102.0,121.0 103.0,120.7 104.0,120.4 105.0,120.1 106.0,119.8 107.0,119.4 108.0,119.1 109.0,118.8 110.0,118.5 111.0,118.1 112.0,117.8 113.0,117.5 114.0,117.2 115.0,116.8 116.0,116.5 117.0,116.2 118.0,115.9 119.0,115.5 120.0,115.2 121.0,114.9 122.0,114.5 123.0,114.2 124.0,113.9 125.0,113.5 126.0,113.2 127.0,112.9 128.0,112.5 129.0,112.2 130.0,111.8 131.0,111.5 132.0,111.2 133.0,110.8 134.0,110.5 135.0,110.1 136.0,109.8 137.0,109.5 138.0,109.1 139.0,108.8 140.0,108.4 141.0,108.1 142.0,107.7 143.0,107.4 144.0,107.1 145.0,106.7 146.0,106.4 147.0,106.0 148.0,105.7 149.0,105.3 150.0,105.0 151.0,104.7 152.0,104.3 153.0,104.0 154.0,103.6 155.0,103.3 156.0,102.9 157.0,102.6 158.0,102.3 159.0,101.9 160.0,101.6 161.0,101.2 162.0,100.9 163.0,100.5 164.0,100.2 165.0,99.9 166.0,99.5 167.0,99.2 168.0,98.8 169.0,98.5 170.0,98.2 171.0,97.8 172.0,97.5 173.0,97.1 174.0,96.8 175.0,96.5 176.0,96.1 177.0,95.8 178.0,95.5 179.0,95.1 180.0,94.8 181.0,94.5 182.0,94.1 183.0,93.8 184.0,93.5 185.0,93.2 186.0,92.8 187.0,92.5 188.0,92.2 189.0,91.9 190.0,91.5 191.0,91.2 192.0,90.9 193.0,90.6 194.0,90.2 195.0,89.9 196.0,89.6 197.0,89.3 198.0,89.0 199.0,88.7 200.0,88.4 201.0,88.0 202.0,87.7 203.0,87.4 204.0,87.1 205.0,86.8 206.0,86.5 207.0,86.2 208.0,85.9 209.0,85.6 210.0,85.3 211.0,85.0 212.0,84.7 213.0,84.4 214.0,84.1 215.0,83.8 216.0,83.5 217.0,83.2 218.0,82.9 219.0,82.6 220.0,82.4 221.0,82.1 222.0,81.8 223.0,81.5 224.0,81.2 225.0,81.0 226.0,80.7 227.0,80.4 228.0,80.1 229.0,79.9 230.0,79.6 231.0,79.3 232.0,79.0 233.0,78.8 234.0,78.5 235.0,78.3 236.0,78.0 237.0,77.7 238.0,77.5 239.0,77.2 240.0,77.0 241.0,76.7 242.0,76.5 243.0,76.2 244.0,76.0 245.0,75.7 246.0,75.5 247.0,75.2 248.0,75.0 249.0,74.7 250.0,74.5 251.0,74.3 252.0,74.0 253.0,73.8 254.0,73.6 255.0,73.3 256.0,73.1 257.0,72.9 258.0,72.6 259.0,72.4 260.0,72.2 261.0,72.0 262.0,71.8 263.0,71.5 264.0,71.3 265.0,71.1 266.0,70.9 267.0,70.7 268.0,70.5 269.0,70.3 270.0,70.1" fill="none" stroke="#2c3e94" stroke-width="1.3" opacity="1.0" />
        <text x="34" y="94" font-size="8.5" fill="#2c3e94">n = 1, 2, 4, 8, 16 (faint → dark)</text>
        <g fill="#b91c1c"><circle cx="159.7" cy="63.8" r="2.4" /><circle cx="169.5" cy="63.8" r="2.4" /><circle cx="188.9" cy="63.8" r="2.4" /><circle cx="227.8" cy="63.8" r="2.4" /></g>
        <line x1="233.8" y1="63.8" x2="267.0" y2="63.8" stroke="#b91c1c" stroke-width="1" /><polygon points="272.0,63.8 267.0,66.2 267.0,61.2" fill="#b91c1c" />
        <text x="144" y="66.8" font-size="8" fill="#b91c1c" text-anchor="end">level ¾ drifts to ∞</text>
        <line x1="184" y1="105" x2="184" y2="85.3" stroke="#a86f00" stroke-width="1" />
        <g stroke="#a86f00" stroke-width="1"><line x1="181" y1="85.3" x2="187" y2="85.3" /></g>
        <text x="189" y="83.3" font-size="8" fill="#a86f00">sup over K = tanh(R/n)</text>
        <g font-size="9" fill="#444">
          <text x="12" y="200">on K = [−R, R]: sup |fₙ| = tanh(R/n) → 0: locally uniform</text>
          <text x="12" y="213">on ℝ: sup |fₙ| = 1 for every n: not uniform, the step escapes</text>
        </g>
      </svg>
    </div>
    <div>
      <div class="panel-label">Window suprema tanh(R/n) against n</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Chart of the window supremum tanh of R over n against n from 1 to 16 for R equal 1, 3, 10 and 30, each curve decaying to zero later for larger R, and a red constant line at one for the supremum over the whole real line">
        <line x1="40.0" y1="165.0" x2="264.0" y2="165.0" stroke="#444" stroke-width="1" /><polygon points="270.0,165.0 264.0,168.0 264.0,162.0" fill="#444" />
        <line x1="40.0" y1="165.0" x2="40.0" y2="46.0" stroke="#444" stroke-width="1" /><polygon points="40.0,40.0 43.0,46.0 37.0,46.0" fill="#444" />
        <text x="272" y="169" font-size="10" fill="#666">n</text>
        <text x="40" y="36" font-size="8.5" fill="#666" text-anchor="middle">sup |fₙ|</text>
        <g font-size="8" fill="#666" text-anchor="middle"><text x="40.0" y="178">1</text><text x="83.5" y="178">4</text><text x="141.5" y="178">8</text><text x="199.5" y="178">12</text><text x="257.5" y="178">16</text></g>
        <g font-size="8" fill="#666" text-anchor="end"><text x="35" y="168">0</text><text x="35" y="58">1</text></g>
        <line x1="40" y1="55" x2="258" y2="55" stroke="#b91c1c" stroke-width="1.3" />
        <g fill="#b91c1c"><circle cx="40.0" cy="55" r="2.2" /><circle cx="54.5" cy="55" r="2.2" /><circle cx="69.0" cy="55" r="2.2" /><circle cx="83.5" cy="55" r="2.2" /><circle cx="98.0" cy="55" r="2.2" /><circle cx="112.5" cy="55" r="2.2" /><circle cx="127.0" cy="55" r="2.2" /><circle cx="141.5" cy="55" r="2.2" /><circle cx="156.0" cy="55" r="2.2" /><circle cx="170.5" cy="55" r="2.2" /><circle cx="185.0" cy="55" r="2.2" /><circle cx="199.5" cy="55" r="2.2" /><circle cx="214.0" cy="55" r="2.2" /><circle cx="228.5" cy="55" r="2.2" /><circle cx="243.0" cy="55" r="2.2" /><circle cx="257.5" cy="55" r="2.2" /></g>
        <text x="150" y="49" font-size="8.5" fill="#b91c1c" text-anchor="middle">sup over ℝ = 1, every n</text>
        <polyline points="40.0,81.2 54.5,114.2 69.0,129.6 83.5,138.1 98.0,143.3 112.5,146.8 127.0,149.4 141.5,151.3 156.0,152.8 170.5,154.0 185.0,155.0 199.5,155.9 214.0,156.6 228.5,157.2 243.0,157.7 257.5,158.1" fill="none" stroke="#2c3e94" stroke-width="1.2" opacity="0.4" />
        <g fill="#2c3e94" opacity="0.4"><circle cx="40.0" cy="81.2" r="2" /><circle cx="54.5" cy="114.2" r="2" /><circle cx="69.0" cy="129.6" r="2" /><circle cx="83.5" cy="138.1" r="2" /><circle cx="98.0" cy="143.3" r="2" /><circle cx="112.5" cy="146.8" r="2" /><circle cx="127.0" cy="149.4" r="2" /><circle cx="141.5" cy="151.3" r="2" /><circle cx="156.0" cy="152.8" r="2" /><circle cx="170.5" cy="154.0" r="2" /><circle cx="185.0" cy="155.0" r="2" /><circle cx="199.5" cy="155.9" r="2" /><circle cx="214.0" cy="156.6" r="2" /><circle cx="228.5" cy="157.2" r="2" /><circle cx="243.0" cy="157.7" r="2" /><circle cx="257.5" cy="158.1" r="2" /></g>
        <text x="263.5" y="161.1" font-size="8" fill="#2c3e94">R = 1</text>
        <polyline points="40.0,55.5 54.5,65.4 69.0,81.2 83.5,95.1 98.0,105.9 112.5,114.2 127.0,120.5 141.5,125.6 156.0,129.6 170.5,133.0 185.0,135.7 199.5,138.1 214.0,140.1 228.5,141.8 243.0,143.3 257.5,144.6" fill="none" stroke="#2c3e94" stroke-width="1.2" opacity="0.6" />
        <g fill="#2c3e94" opacity="0.6"><circle cx="40.0" cy="55.5" r="2" /><circle cx="54.5" cy="65.4" r="2" /><circle cx="69.0" cy="81.2" r="2" /><circle cx="83.5" cy="95.1" r="2" /><circle cx="98.0" cy="105.9" r="2" /><circle cx="112.5" cy="114.2" r="2" /><circle cx="127.0" cy="120.5" r="2" /><circle cx="141.5" cy="125.6" r="2" /><circle cx="156.0" cy="129.6" r="2" /><circle cx="170.5" cy="133.0" r="2" /><circle cx="185.0" cy="135.7" r="2" /><circle cx="199.5" cy="138.1" r="2" /><circle cx="214.0" cy="140.1" r="2" /><circle cx="228.5" cy="141.8" r="2" /><circle cx="243.0" cy="143.3" r="2" /><circle cx="257.5" cy="144.6" r="2" /></g>
        <text x="263.5" y="147.6" font-size="8" fill="#2c3e94">R = 3</text>
        <polyline points="40.0,55.0 54.5,55.0 69.0,55.3 83.5,56.5 98.0,59.0 112.5,62.6 127.0,66.9 141.5,71.7 156.0,76.5 170.5,81.2 185.0,85.7 199.5,90.0 214.0,93.9 228.5,97.5 243.0,100.9 257.5,104.0" fill="none" stroke="#2c3e94" stroke-width="1.2" opacity="0.8" />
        <g fill="#2c3e94" opacity="0.8"><circle cx="40.0" cy="55.0" r="2" /><circle cx="54.5" cy="55.0" r="2" /><circle cx="69.0" cy="55.3" r="2" /><circle cx="83.5" cy="56.5" r="2" /><circle cx="98.0" cy="59.0" r="2" /><circle cx="112.5" cy="62.6" r="2" /><circle cx="127.0" cy="66.9" r="2" /><circle cx="141.5" cy="71.7" r="2" /><circle cx="156.0" cy="76.5" r="2" /><circle cx="170.5" cy="81.2" r="2" /><circle cx="185.0" cy="85.7" r="2" /><circle cx="199.5" cy="90.0" r="2" /><circle cx="214.0" cy="93.9" r="2" /><circle cx="228.5" cy="97.5" r="2" /><circle cx="243.0" cy="100.9" r="2" /><circle cx="257.5" cy="104.0" r="2" /></g>
        <text x="263.5" y="107.0" font-size="8" fill="#2c3e94">R = 10</text>
        <polyline points="40.0,55.0 54.5,55.0 69.0,55.0 83.5,55.0 98.0,55.0 112.5,55.0 127.0,55.0 141.5,55.1 156.0,55.3 170.5,55.5 185.0,55.9 199.5,56.5 214.0,57.2 228.5,58.0 243.0,59.0 257.5,60.1" fill="none" stroke="#2c3e94" stroke-width="1.2" opacity="1.0" />
        <g fill="#2c3e94" opacity="1.0"><circle cx="40.0" cy="55.0" r="2" /><circle cx="54.5" cy="55.0" r="2" /><circle cx="69.0" cy="55.0" r="2" /><circle cx="83.5" cy="55.0" r="2" /><circle cx="98.0" cy="55.0" r="2" /><circle cx="112.5" cy="55.0" r="2" /><circle cx="127.0" cy="55.0" r="2" /><circle cx="141.5" cy="55.1" r="2" /><circle cx="156.0" cy="55.3" r="2" /><circle cx="170.5" cy="55.5" r="2" /><circle cx="185.0" cy="55.9" r="2" /><circle cx="199.5" cy="56.5" r="2" /><circle cx="214.0" cy="57.2" r="2" /><circle cx="228.5" cy="58.0" r="2" /><circle cx="243.0" cy="59.0" r="2" /><circle cx="257.5" cy="60.1" r="2" /></g>
        <text x="263.5" y="63.1" font-size="8" fill="#2c3e94">R = 30</text>
        <g font-size="9" fill="#444">
          <text x="12" y="200">each window sup tanh(R/n) → 0, but slower for larger R</text>
          <text x="12" y="213">no single N serves every R: locally uniform ≠ uniform on ℝ</text>
        </g>
      </svg>
    </div>
  </div>
  <figcaption>The local example $f_n(x) = \tanh(x/n)$. <strong>Left.</strong> As $n$ grows the graphs flatten, but each one still climbs to $\pm 1$: the step only moves outward, with $f_n$ reaching the level $\tfrac{3}{4}$ at $x \approx 0.97\,n$. Inside any compact window $K = [-R, R]$ the whole graph lies below $\tanh(R/n)$, which tends to $0$, so $f_n \to 0$ uniformly on $K$. On all of $\mathbb{R}$ the supremum stays $1$, so the convergence is not uniform. <strong>Right.</strong> The window suprema $\tanh(R/n)$ against $n$ for several $R$: every curve tends to $0$, but the larger the window the later it starts to drop, and the supremum over $\mathbb{R}$ never drops at all. Uniform convergence on $\mathbb{R}$ would need one $N$ that works for every $R$, and there is none. This is the same escape to infinity as the sliding boxes in the weak-convergence section below: on a non-compact domain, mass can run off to where no fixed window sees it. That is exactly why the local theorem can only promise uniform convergence on each compact $K_m$ of an exhaustion $K_1 \subset K_2 \subset \cdots$ of $\mathbb{R}$, and why on a compact domain the distinction disappears.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Compactness from Uniform $L^2$ Bounds on a Function and its Derivative)</span></p>

Let $\lbrace f_n\rbrace_n$ be a sequence of continuously differentiable functions on $[0,1]$ such that

$$B := \sup_n \left[\lVert f_n\rVert_2 + \lVert f_n'\rVert_2\right] < \infty.$$

Then:

1. For every $n\in\mathbb N$,

   $$\lVert f_n\rVert_\infty \le B.$$

2. For every $n\in\mathbb N$ and $x,y\in[0,1]$,

   $$|f_n(x)-f_n(y)| \le B|x-y|^{1/2}.$$

3. There exists a subsequence $\lbrace f_{n_j}\rbrace\_j$ converging in $L^2([0,1])$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Step 1: Uniform boundedness.**

Prove that for every $x,y\in[0,1]$,

$$|f_n(x)|\le|f_n(y)|+\lVert f_n'\rVert_2.$$

Then integrate with respect to $y\in[0,1]$.

**Step 2: Uniform equicontinuity.**

Using the Fundamental Theorem of Calculus, prove that for every $x,y\in[0,1]$,

$$|f_n(x)-f_n(y)| \le B|x-y|^{1/2}.$$

**Step 3: Compactness.**

Use Steps 1 and 2 together with the Arzelà–Ascoli theorem to extract a uniformly convergent subsequence. Then show that uniform convergence on $[0,1]$ implies convergence in $L^2([0,1])$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sobolev-Space Formulation)</span></p>

The assumption

$$\sup_n\left(\lVert f_n\rVert_{L^2(0,1)}+\lVert f_n'\rVert_{L^2(0,1)}\right)<\infty$$

is precisely the statement that the sequence $\lbrace f_n\rbrace_n$ is bounded in the Sobolev space $H^1(0,1)$, up to the equivalent choice of norm

$$\lVert f\rVert_{H^1}=\left(\lVert f\rVert_{L^2}^2+\lVert f'\rVert_{L^2}^2\right)^{1/2}.$$

Thus the theorem can be reformulated as follows:

> Every bounded sequence in $H^1(0,1)$ admits a subsequence that converges strongly in $L^2(0,1)$.

Equivalently, the embedding

$$H^1(0,1)\Subset L^2(0,1)$$

is compact.

In fact, the proof gives more. In one dimension, every $f\in H^1(0,1)$ has an absolutely continuous representative and

$$f(x)-f(y)=\int_y^x f'(t)\,dt.$$

Hence, by Cauchy--Schwarz,

$$|f(x)-f(y)|\le\lVert f'\rVert_{L^2}|x-y|^{1/2}.$$

Therefore

$$H^1(0,1)\hookrightarrow C^{0,1/2}([0,1])$$

continuously. A bounded subset of $H^1(0,1)$ is consequently uniformly bounded and equicontinuous, so Arzelà--Ascoli yields the stronger compact embedding

$$H^1(0,1)\Subset C([0,1]).$$

The compact embedding into $L^2(0,1)$ then follows immediately from

$$C([0,1])\hookrightarrow L^2(0,1).$$

This is a one-dimensional instance of the Sobolev--Morrey embedding together with the Rellich--Kondrachov compactness theorem.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compactness vs. Sequential compactness: Metric Spaces)</span></p>

**In metric spaces, compactness and sequential compactness are equivalent.** But this equivalence fails in general topological spaces.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(In Metric Space: Compactness $\iff$ Sequential compactness)</span></p>

Let $(X,d)$ be a metric space and $K\subseteq X$. Then

$$K\text{ is compact} \iff K\text{ is sequentially compact}.$$

</div>

Here:
- **Compactness**: Every open cover of $K$ has a finite subcover.
- **Sequential compactness**: Every sequence in $K$ admits a subsequence converging to a point in $K$.

Therefore, in normed spaces equipped with their norm topology, you can use these notions interchangeably.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compactness vs. Sequential compactness: Non-metrizable Space)</span></p>

Because the **weak topology is generally not metrizable** on an infinite-dimensional Banach space.

Consequently, the equivalence between compactness and sequential compactness is no longer automatic.

However, there is a remarkable theorem:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Eberlein–Šmulian Theorem)</span></p>

For a Banach space $X$ and $K\subseteq X$,

$$K\text{ is weakly compact} \iffK\text{ is weakly sequentially compact}.$$

Thus, for weak topologies on Banach spaces, the equivalence is restored despite their possible nonmetrizability.

</div>

**The distinction becomes especially important for weak-$\ast$ topologies:** Banach–Alaoglu guarantees weak-$\ast$ compactness of the dual unit ball, but weak-$\ast$ sequential compactness does not follow in general. Separability of the predual is a standard sufficient condition.

</div>

## Weak Convergence in Hilbert Spaces

The proof of Theorem 197 extracted a subsequence along which **every Fourier coefficient** $\langle u\_n, e\_k \rangle$ converges, and then needed equi-small tails to upgrade this to norm convergence. Without the tail condition we still get something useful: a weaker notion of convergence for which bounded sets **are** (sequentially) compact — an infinite-dimensional substitute for Heine–Borel.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak Convergence)</span></p>

A sequence $\lbrace u\_n \rbrace$ in a Hilbert space $H$ **converges weakly** to $u \in H$, written

$$u_n \rightharpoonup u,$$

if $\langle u\_n, v \rangle \to \langle u, v \rangle$ for every $v \in H$. Equivalently,  $\lvert \langle u\_n - u, v \rangle \rvert \to 0$ for every $v \in H$. To distinguish it, ordinary norm convergence $\lVert u\_n - u \rVert \to 0$ is called **strong** convergence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Uniqueness of Weak Limits)</span></p>

If $u\_n \rightharpoonup u$ and $u\_n \rightharpoonup u^{\prime}$, then $\langle u - u^{\prime}, v \rangle = 0$ for all $v$; taking $v = u - u^{\prime}$ gives $u = u^{\prime}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Strong $\Rightarrow$ Weak)</span></p>

If $u\_n \to u$ in norm, then $u\_n \rightharpoonup u$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

By Cauchy–Schwarz, $\lvert \langle u\_n, v \rangle - \langle u, v \rangle \rvert \le \lVert u\_n - u \rVert \lVert v \rVert \to 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Orthonormal Sequences Converge Weakly to $0$)</span></p>

If $\lbrace e\_n \rbrace$ is an orthonormal sequence, then $e\_n \rightharpoonup 0$: 
* by Bessel's inequality $\sum\_n \lvert \langle e\_n, v \rangle \rvert^2 \le \lVert v \rVert^2 < \infty$, so $\langle e\_n, v \rangle \to 0$ for each $v$. 
* But $\lVert e\_n - e\_m \rVert = \sqrt{2}$ for $n \ne m$, so $e\_n$ does **not** converge strongly. 
* In $L^2([-\pi, \pi])$, this says that the oscillating functions $e^{inx}/\sqrt{2\pi}$ converge weakly to $0$ — a form of the Riemann–Lebesgue lemma. 
* Weak convergence allows "mass" to escape to ever higher frequencies (or to infinity).

</div>

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">Escape to infinity: eₙ = χ[n, n+1) in L²(ℝ)</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="A decaying test function v over the positive axis with unit-height indicator boxes e1, e3, e5, e7 sliding to the right; the shaded overlap of each box with v shrinks to nothing while the boxes keep unit area, and a budget bar below stacks the squared inner products inside the norm of v squared">
        <line x1="24.0" y1="140.0" x2="284.0" y2="140.0" stroke="#444" stroke-width="1.2" /><polygon points="290.0,140.0 284.0,143.0 284.0,137.0" fill="#444" />
        <line x1="24.0" y1="140.0" x2="24.0" y2="54.0" stroke="#444" stroke-width="1.2" /><polygon points="24.0,48.0 27.0,54.0 21.0,54.0" fill="#444" />
        <text x="292" y="144" font-size="11" fill="#666">x</text>
        <line x1="20.0" y1="62.0" x2="24.0" y2="62.0" stroke="#444" stroke-width="1" />
        <text x="18.0" y="65.0" font-size="9" fill="#666" text-anchor="end">1</text>
        <g font-size="8" fill="#666" text-anchor="middle">
          <text x="24.0" y="151">0</text>
          <text x="52.0" y="151">1</text>
          <text x="80.0" y="151">2</text>
          <text x="108.0" y="151">3</text>
          <text x="136.0" y="151">4</text>
          <text x="164.0" y="151">5</text>
          <text x="192.0" y="151">6</text>
          <text x="220.0" y="151">7</text>
          <text x="248.0" y="151">8</text>
          <text x="276.0" y="151">9</text>
        </g>
        <g fill="#c62828" opacity="0.5">
          <path d="M 52.0,140.0 L 52.0,74.7 52.7,74.2 53.4,73.7 54.1,73.2 54.8,72.8 55.5,72.4 56.2,72.1 56.9,71.7 57.6,71.4 58.3,71.2 59.0,70.9 59.7,70.7 60.4,70.5 61.1,70.3 61.8,70.2 62.5,70.1 63.2,70.0 63.9,69.9 64.6,69.9 65.3,69.8 66.0,69.8 66.7,69.8 67.4,69.9 68.1,69.9 68.8,70.0 69.5,70.1 70.2,70.2 70.9,70.3 71.6,70.4 72.3,70.5 73.0,70.7 73.7,70.9 74.4,71.1 75.1,71.3 75.8,71.5 76.5,71.7 77.2,71.9 77.9,72.2 78.6,72.4 79.3,72.7 80.0,73.0 L 80.0,140.0 Z" />
          <path d="M 108.0,140.0 L 108.0,88.4 108.7,88.8 109.4,89.2 110.1,89.7 110.8,90.1 111.5,90.5 112.2,90.9 112.9,91.4 113.6,91.8 114.3,92.2 115.0,92.7 115.7,93.1 116.4,93.5 117.1,93.9 117.8,94.3 118.5,94.8 119.2,95.2 119.9,95.6 120.6,96.0 121.3,96.4 122.0,96.8 122.7,97.2 123.4,97.7 124.1,98.1 124.8,98.5 125.5,98.9 126.2,99.3 126.9,99.7 127.6,100.1 128.3,100.5 129.0,100.9 129.7,101.2 130.4,101.6 131.1,102.0 131.8,102.4 132.5,102.8 133.2,103.2 133.9,103.5 134.6,103.9 135.3,104.3 136.0,104.7 L 136.0,140.0 Z" />
          <path d="M 164.0,140.0 L 164.0,117.3 164.7,117.6 165.4,117.8 166.1,118.1 166.8,118.4 167.5,118.6 168.2,118.9 168.9,119.1 169.6,119.4 170.3,119.6 171.0,119.8 171.7,120.1 172.4,120.3 173.1,120.5 173.8,120.8 174.5,121.0 175.2,121.2 175.9,121.5 176.6,121.7 177.3,121.9 178.0,122.1 178.7,122.3 179.4,122.5 180.1,122.8 180.8,123.0 181.5,123.2 182.2,123.4 182.9,123.6 183.6,123.8 184.3,124.0 185.0,124.2 185.7,124.4 186.4,124.6 187.1,124.8 187.8,124.9 188.5,125.1 189.2,125.3 189.9,125.5 190.6,125.7 191.3,125.8 192.0,126.0 L 192.0,140.0 Z" />
          <path d="M 220.0,140.0 L 220.0,131.6 220.7,131.7 221.4,131.8 222.1,132.0 222.8,132.1 223.5,132.2 224.2,132.3 224.9,132.4 225.6,132.5 226.3,132.6 227.0,132.7 227.7,132.8 228.4,132.9 229.1,132.9 229.8,133.0 230.5,133.1 231.2,133.2 231.9,133.3 232.6,133.4 233.3,133.5 234.0,133.6 234.7,133.7 235.4,133.7 236.1,133.8 236.8,133.9 237.5,134.0 238.2,134.1 238.9,134.1 239.6,134.2 240.3,134.3 241.0,134.4 241.7,134.5 242.4,134.5 243.1,134.6 243.8,134.7 244.5,134.7 245.2,134.8 245.9,134.9 246.6,135.0 247.3,135.0 248.0,135.1 L 248.0,140.0 Z" />
        </g>
        <g fill="#1565c0" stroke="#1565c0" stroke-width="1">
          <rect x="52.0" y="62.0" width="28.0" height="78.0" fill-opacity="0.38" />
          <rect x="108.0" y="62.0" width="28.0" height="78.0" fill-opacity="0.3" />
          <rect x="164.0" y="62.0" width="28.0" height="78.0" fill-opacity="0.22" />
          <rect x="220.0" y="62.0" width="28.0" height="78.0" fill-opacity="0.15" />
        </g>
        <g font-size="8.5" fill="#1565c0" text-anchor="middle">
          <text x="66.0" y="57.0">e₁</text>
          <text x="122.0" y="57.0">e₃</text>
          <text x="178.0" y="57.0">e₅</text>
          <text x="234.0" y="57.0">e₇</text>
        </g>
        <polyline points="24.0,140.0 26.8,128.1 29.6,117.7 32.4,108.8 35.2,101.0 38.0,94.4 40.8,88.9 43.6,84.2 46.4,80.3 49.2,77.2 52.0,74.7 54.8,72.8 57.6,71.4 60.4,70.5 63.2,70.0 66.0,69.8 68.8,70.0 71.6,70.4 74.4,71.1 77.2,71.9 80.0,73.0 82.8,74.1 85.6,75.5 88.4,76.9 91.2,78.4 94.0,79.9 96.8,81.6 99.6,83.2 102.4,84.9 105.2,86.6 108.0,88.4 110.8,90.1 113.6,91.8 116.4,93.5 119.2,95.2 122.0,96.8 124.8,98.5 127.6,100.1 130.4,101.6 133.2,103.2 136.0,104.7 138.8,106.1 141.6,107.5 144.4,108.9 147.2,110.2 150.0,111.5 152.8,112.8 155.6,114.0 158.4,115.1 161.2,116.2 164.0,117.3 166.8,118.4 169.6,119.4 172.4,120.3 175.2,121.2 178.0,122.1 180.8,123.0 183.6,123.8 186.4,124.6 189.2,125.3 192.0,126.0 194.8,126.7 197.6,127.4 200.4,128.0 203.2,128.6 206.0,129.2 208.8,129.7 211.6,130.2 214.4,130.7 217.2,131.2 220.0,131.6 222.8,132.1 225.6,132.5 228.4,132.9 231.2,133.2 234.0,133.6 236.8,133.9 239.6,134.2 242.4,134.5 245.2,134.8 248.0,135.1 250.8,135.3 253.6,135.6 256.4,135.8 259.2,136.0 262.0,136.3 264.8,136.5 267.6,136.7 270.4,136.8 273.2,137.0 276.0,137.2 278.8,137.3 281.6,137.5 284.4,137.6" fill="none" stroke="#1f2430" stroke-width="1.6" />
        <text x="96" y="76" font-size="10" font-style="italic" fill="#1f2430">v</text>
        <line x1="150.0" y1="44.0" x2="256.0" y2="44.0" stroke="#1565c0" stroke-width="1.1" /><polygon points="262.0,44.0 256.0,47.0 256.0,41.0" fill="#1565c0" />
        <text x="206" y="38" font-size="9" fill="#1565c0" text-anchor="middle">n → ∞</text>
        <rect x="24.0" y="163.0" width="176.0" height="8.0" fill="#fff" stroke="#666" stroke-width="1" />
        <g fill="#1565c0" stroke="#fff" stroke-width="0.6">
          <rect x="24.00" y="163.0" width="21.99" height="8.0" fill-opacity="0.95" />
          <rect x="45.99" y="163.0" width="61.15" height="8.0" fill-opacity="0.85" />
          <rect x="107.14" y="163.0" width="46.15" height="8.0" fill-opacity="0.75" />
          <rect x="153.29" y="163.0" width="24.15" height="8.0" fill-opacity="0.65" />
          <rect x="177.44" y="163.0" width="10.60" height="8.0" fill-opacity="0.55" />
          <rect x="188.04" y="163.0" width="4.19" height="8.0" fill-opacity="0.45" />
          <rect x="192.24" y="163.0" width="1.55" height="8.0" fill-opacity="0.35" />
          <rect x="193.78" y="163.0" width="0.54" height="8.0" fill-opacity="0.25" />
          <rect x="194.33" y="163.0" width="0.18" height="8.0" fill-opacity="0.15" />
          <rect x="194.51" y="163.0" width="0.06" height="8.0" fill-opacity="0.15" />
        </g>
        <text x="204.0" y="170.0" font-size="9" fill="#444">‖v‖²</text>
        <g font-size="9" fill="#444">
          <text x="12" y="186">budget bar: |⟨e₀, v⟩|² + |⟨e₁, v⟩|² + ⋯ ≤ ‖v‖²   (Bessel)</text>
          <text x="12" y="200">shaded area ⟨eₙ, v⟩ = ∫ₙⁿ⁺¹ v → 0: a finite budget forces it</text>
          <text x="12" y="213">yet ‖eₙ‖ = 1 always, and boxes never overlap: ‖eₙ − eₘ‖ = √2</text>
        </g>
      </svg>
    </div>
    <div>
      <div class="panel-label">Escape to high frequency: eₙ = sin(nx)/√π</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="340" aria-label="Three strips showing v times sine of n x for n equal 2, 5 and 14 inside the fixed dashed envelope plus and minus v; positive lobes are shaded blue and negative lobes red, they cancel more and more finely, and the printed inner products with v shrink toward zero">
        <line x1="272.0" y1="38.0" x2="272.0" y2="160.0" stroke="#1565c0" stroke-width="1.1" /><polygon points="272.0,166.0 269.0,160.0 275.0,160.0" fill="#1565c0" />
        <text x="272" y="30" font-size="9" fill="#1565c0" text-anchor="middle">n → ∞</text>
        <line x1="36.0" y1="50" x2="250.0" y2="50" stroke="#bbb" stroke-width="0.8" />
        <g fill="#1565c0" opacity="0.4"><path d="M 36.0,50 L 36.0,50.0 36.4,50.0 36.7,50.0 37.1,50.0 37.4,50.0 37.8,50.0 38.1,50.0 38.5,50.0 38.9,50.0 39.2,50.0 39.6,49.9 39.9,49.9 40.3,49.9 40.6,49.9 41.0,49.9 41.4,49.9 41.7,49.9 42.1,49.9 42.4,49.9 42.8,49.9 43.1,49.9 43.5,49.9 43.8,49.9 44.2,49.9 44.6,49.9 44.9,49.8 45.3,49.8 45.6,49.8 46.0,49.8 46.3,49.8 46.7,49.8 47.1,49.8 47.4,49.8 47.8,49.8 48.1,49.8 48.5,49.8 48.8,49.8 49.2,49.8 49.6,49.7 49.9,49.7 50.3,49.7 50.6,49.7 51.0,49.7 51.3,49.7 51.7,49.7 52.0,49.7 52.4,49.7 52.8,49.7 53.1,49.7 53.5,49.7 53.8,49.7 54.2,49.6 54.5,49.6 54.9,49.6 55.3,49.6 55.6,49.6 56.0,49.6 56.3,49.6 56.7,49.6 57.0,49.6 57.4,49.6 57.8,49.6 58.1,49.6 58.5,49.5 58.8,49.5 59.2,49.5 59.5,49.5 59.9,49.5 60.3,49.5 60.6,49.5 61.0,49.5 61.3,49.5 61.7,49.5 62.0,49.5 62.4,49.5 62.8,49.5 63.1,49.5 63.5,49.4 63.8,49.4 64.2,49.4 64.5,49.4 64.9,49.4 65.2,49.4 65.6,49.4 66.0,49.4 66.3,49.4 66.7,49.4 67.0,49.4 67.4,49.4 67.7,49.4 68.1,49.4 68.5,49.4 68.8,49.4 69.2,49.4 69.5,49.4 69.9,49.4 70.2,49.4 70.6,49.4 71.0,49.4 71.3,49.4 71.7,49.4 72.0,49.4 72.4,49.4 72.7,49.4 73.1,49.4 73.5,49.4 73.8,49.4 74.2,49.4 74.5,49.4 74.9,49.4 75.2,49.4 75.6,49.4 75.9,49.4 76.3,49.4 76.7,49.4 77.0,49.4 77.4,49.4 77.7,49.4 78.1,49.4 78.4,49.5 78.8,49.5 79.2,49.5 79.5,49.5 79.9,49.5 80.2,49.5 80.6,49.5 80.9,49.5 81.3,49.5 81.7,49.6 82.0,49.6 82.4,49.6 82.7,49.6 83.1,49.6 83.4,49.6 83.8,49.6 84.2,49.7 84.5,49.7 84.9,49.7 85.2,49.7 85.6,49.7 85.9,49.8 86.3,49.8 86.6,49.8 87.0,49.8 87.4,49.8 87.7,49.9 88.1,49.9 88.4,49.9 88.8,49.9 89.1,50.0 89.5,50.0 89.9,50.0 90.2,50.0 90.6,50.0 90.9,50.0 91.3,50.0 91.6,50.0 92.0,50.0 92.4,50.0 92.7,50.0 93.1,50.0 93.4,50.0 93.8,50.0 94.1,50.0 94.5,50.0 94.8,50.0 95.2,50.0 95.6,50.0 95.9,50.0 96.3,50.0 96.6,50.0 97.0,50.0 97.3,50.0 97.7,50.0 98.1,50.0 98.4,50.0 98.8,50.0 99.1,50.0 99.5,50.0 99.8,50.0 100.2,50.0 100.6,50.0 100.9,50.0 101.3,50.0 101.6,50.0 102.0,50.0 102.3,50.0 102.7,50.0 103.1,50.0 103.4,50.0 103.8,50.0 104.1,50.0 104.5,50.0 104.8,50.0 105.2,50.0 105.5,50.0 105.9,50.0 106.3,50.0 106.6,50.0 107.0,50.0 107.3,50.0 107.7,50.0 108.0,50.0 108.4,50.0 108.8,50.0 109.1,50.0 109.5,50.0 109.8,50.0 110.2,50.0 110.5,50.0 110.9,50.0 111.3,50.0 111.6,50.0 112.0,50.0 112.3,50.0 112.7,50.0 113.0,50.0 113.4,50.0 113.8,50.0 114.1,50.0 114.5,50.0 114.8,50.0 115.2,50.0 115.5,50.0 115.9,50.0 116.2,50.0 116.6,50.0 117.0,50.0 117.3,50.0 117.7,50.0 118.0,50.0 118.4,50.0 118.7,50.0 119.1,50.0 119.5,50.0 119.8,50.0 120.2,50.0 120.5,50.0 120.9,50.0 121.2,50.0 121.6,50.0 122.0,50.0 122.3,50.0 122.7,50.0 123.0,50.0 123.4,50.0 123.7,50.0 124.1,50.0 124.5,50.0 124.8,50.0 125.2,50.0 125.5,50.0 125.9,50.0 126.2,50.0 126.6,50.0 127.0,50.0 127.3,50.0 127.7,50.0 128.0,50.0 128.4,50.0 128.7,50.0 129.1,50.0 129.4,50.0 129.8,50.0 130.2,50.0 130.5,50.0 130.9,50.0 131.2,50.0 131.6,50.0 131.9,50.0 132.3,50.0 132.7,50.0 133.0,50.0 133.4,50.0 133.7,50.0 134.1,50.0 134.4,50.0 134.8,50.0 135.2,50.0 135.5,50.0 135.9,50.0 136.2,50.0 136.6,50.0 136.9,50.0 137.3,50.0 137.6,50.0 138.0,50.0 138.4,50.0 138.7,50.0 139.1,50.0 139.4,50.0 139.8,50.0 140.1,50.0 140.5,50.0 140.9,50.0 141.2,50.0 141.6,50.0 141.9,50.0 142.3,50.0 142.6,50.0 143.0,50.0 143.4,49.8 143.7,49.7 144.1,49.5 144.4,49.4 144.8,49.2 145.1,49.0 145.5,48.8 145.9,48.6 146.2,48.5 146.6,48.3 146.9,48.1 147.3,47.9 147.6,47.7 148.0,47.5 148.4,47.3 148.7,47.1 149.1,46.9 149.4,46.6 149.8,46.4 150.1,46.2 150.5,46.0 150.8,45.7 151.2,45.5 151.6,45.3 151.9,45.1 152.3,44.8 152.6,44.6 153.0,44.3 153.3,44.1 153.7,43.8 154.1,43.6 154.4,43.3 154.8,43.1 155.1,42.8 155.5,42.6 155.8,42.3 156.2,42.0 156.6,41.8 156.9,41.5 157.3,41.2 157.6,41.0 158.0,40.7 158.3,40.4 158.7,40.2 159.1,39.9 159.4,39.6 159.8,39.4 160.1,39.1 160.5,38.8 160.8,38.5 161.2,38.3 161.5,38.0 161.9,37.7 162.3,37.4 162.6,37.2 163.0,36.9 163.3,36.6 163.7,36.4 164.0,36.1 164.4,35.9 164.8,35.6 165.1,35.3 165.5,35.1 165.8,34.8 166.2,34.6 166.5,34.3 166.9,34.1 167.3,33.9 167.6,33.6 168.0,33.4 168.3,33.2 168.7,32.9 169.0,32.7 169.4,32.5 169.8,32.3 170.1,32.1 170.5,32.1 170.8,32.4 171.2,32.6 171.5,32.8 171.9,33.1 172.2,33.3 172.6,33.6 173.0,33.8 173.3,34.1 173.7,34.3 174.0,34.6 174.4,34.9 174.7,35.1 175.1,35.4 175.5,35.7 175.8,35.9 176.2,36.2 176.5,36.5 176.9,36.8 177.2,37.0 177.6,37.3 178.0,37.6 178.3,37.9 178.7,38.2 179.0,38.4 179.4,38.7 179.7,39.0 180.1,39.3 180.4,39.6 180.8,39.8 181.2,40.1 181.5,40.4 181.9,40.7 182.2,41.0 182.6,41.2 182.9,41.5 183.3,41.8 183.7,42.0 184.0,42.3 184.4,42.6 184.7,42.8 185.1,43.1 185.4,43.4 185.8,43.6 186.2,43.9 186.5,44.1 186.9,44.4 187.2,44.6 187.6,44.9 187.9,45.1 188.3,45.4 188.7,45.6 189.0,45.8 189.4,46.1 189.7,46.3 190.1,46.5 190.4,46.8 190.8,47.0 191.2,47.2 191.5,47.4 191.9,47.6 192.2,47.8 192.6,48.0 192.9,48.2 193.3,48.4 193.6,48.6 194.0,48.8 194.4,49.0 194.7,49.2 195.1,49.3 195.4,49.5 195.8,49.7 196.1,49.8 196.5,50.0 196.9,50.0 197.2,50.0 197.6,50.0 197.9,50.0 198.3,50.0 198.6,50.0 199.0,50.0 199.4,50.0 199.7,50.0 200.1,50.0 200.4,50.0 200.8,50.0 201.1,50.0 201.5,50.0 201.8,50.0 202.2,50.0 202.6,50.0 202.9,50.0 203.3,50.0 203.6,50.0 204.0,50.0 204.3,50.0 204.7,50.0 205.1,50.0 205.4,50.0 205.8,50.0 206.1,50.0 206.5,50.0 206.8,50.0 207.2,50.0 207.6,50.0 207.9,50.0 208.3,50.0 208.6,50.0 209.0,50.0 209.3,50.0 209.7,50.0 210.1,50.0 210.4,50.0 210.8,50.0 211.1,50.0 211.5,50.0 211.8,50.0 212.2,50.0 212.5,50.0 212.9,50.0 213.3,50.0 213.6,50.0 214.0,50.0 214.3,50.0 214.7,50.0 215.0,50.0 215.4,50.0 215.8,50.0 216.1,50.0 216.5,50.0 216.8,50.0 217.2,50.0 217.5,50.0 217.9,50.0 218.3,50.0 218.6,50.0 219.0,50.0 219.3,50.0 219.7,50.0 220.0,50.0 220.4,50.0 220.8,50.0 221.1,50.0 221.5,50.0 221.8,50.0 222.2,50.0 222.5,50.0 222.9,50.0 223.2,50.0 223.6,50.0 224.0,50.0 224.3,50.0 224.7,50.0 225.0,50.0 225.4,50.0 225.7,50.0 226.1,50.0 226.5,50.0 226.8,50.0 227.2,50.0 227.5,50.0 227.9,50.0 228.2,50.0 228.6,50.0 229.0,50.0 229.3,50.0 229.7,50.0 230.0,50.0 230.4,50.0 230.7,50.0 231.1,50.0 231.5,50.0 231.8,50.0 232.2,50.0 232.5,50.0 232.9,50.0 233.2,50.0 233.6,50.0 234.0,50.0 234.3,50.0 234.7,50.0 235.0,50.0 235.4,50.0 235.7,50.0 236.1,50.0 236.4,50.0 236.8,50.0 237.2,50.0 237.5,50.0 237.9,50.0 238.2,50.0 238.6,50.0 238.9,50.0 239.3,50.0 239.7,50.0 240.0,50.0 240.4,50.0 240.7,50.0 241.1,50.0 241.4,50.0 241.8,50.0 242.2,50.0 242.5,50.0 242.9,50.0 243.2,50.0 243.6,50.0 243.9,50.0 244.3,50.0 244.7,50.0 245.0,50.0 245.4,50.0 245.7,50.0 246.1,50.0 246.4,50.0 246.8,50.0 247.1,50.0 247.5,50.0 247.9,50.0 248.2,50.0 248.6,50.0 248.9,50.0 249.3,50.0 249.6,50.0 250.0,50.0 L 250.0,50 Z" /></g>
        <g fill="#c62828" opacity="0.4"><path d="M 36.0,50 L 36.0,50.0 36.4,50.0 36.7,50.0 37.1,50.0 37.4,50.0 37.8,50.0 38.1,50.0 38.5,50.0 38.9,50.0 39.2,50.0 39.6,50.0 39.9,50.0 40.3,50.0 40.6,50.0 41.0,50.0 41.4,50.0 41.7,50.0 42.1,50.0 42.4,50.0 42.8,50.0 43.1,50.0 43.5,50.0 43.8,50.0 44.2,50.0 44.6,50.0 44.9,50.0 45.3,50.0 45.6,50.0 46.0,50.0 46.3,50.0 46.7,50.0 47.1,50.0 47.4,50.0 47.8,50.0 48.1,50.0 48.5,50.0 48.8,50.0 49.2,50.0 49.6,50.0 49.9,50.0 50.3,50.0 50.6,50.0 51.0,50.0 51.3,50.0 51.7,50.0 52.0,50.0 52.4,50.0 52.8,50.0 53.1,50.0 53.5,50.0 53.8,50.0 54.2,50.0 54.5,50.0 54.9,50.0 55.3,50.0 55.6,50.0 56.0,50.0 56.3,50.0 56.7,50.0 57.0,50.0 57.4,50.0 57.8,50.0 58.1,50.0 58.5,50.0 58.8,50.0 59.2,50.0 59.5,50.0 59.9,50.0 60.3,50.0 60.6,50.0 61.0,50.0 61.3,50.0 61.7,50.0 62.0,50.0 62.4,50.0 62.8,50.0 63.1,50.0 63.5,50.0 63.8,50.0 64.2,50.0 64.5,50.0 64.9,50.0 65.2,50.0 65.6,50.0 66.0,50.0 66.3,50.0 66.7,50.0 67.0,50.0 67.4,50.0 67.7,50.0 68.1,50.0 68.5,50.0 68.8,50.0 69.2,50.0 69.5,50.0 69.9,50.0 70.2,50.0 70.6,50.0 71.0,50.0 71.3,50.0 71.7,50.0 72.0,50.0 72.4,50.0 72.7,50.0 73.1,50.0 73.5,50.0 73.8,50.0 74.2,50.0 74.5,50.0 74.9,50.0 75.2,50.0 75.6,50.0 75.9,50.0 76.3,50.0 76.7,50.0 77.0,50.0 77.4,50.0 77.7,50.0 78.1,50.0 78.4,50.0 78.8,50.0 79.2,50.0 79.5,50.0 79.9,50.0 80.2,50.0 80.6,50.0 80.9,50.0 81.3,50.0 81.7,50.0 82.0,50.0 82.4,50.0 82.7,50.0 83.1,50.0 83.4,50.0 83.8,50.0 84.2,50.0 84.5,50.0 84.9,50.0 85.2,50.0 85.6,50.0 85.9,50.0 86.3,50.0 86.6,50.0 87.0,50.0 87.4,50.0 87.7,50.0 88.1,50.0 88.4,50.0 88.8,50.0 89.1,50.0 89.5,50.0 89.9,50.0 90.2,50.1 90.6,50.1 90.9,50.1 91.3,50.1 91.6,50.2 92.0,50.2 92.4,50.2 92.7,50.3 93.1,50.3 93.4,50.3 93.8,50.4 94.1,50.4 94.5,50.4 94.8,50.5 95.2,50.5 95.6,50.5 95.9,50.6 96.3,50.6 96.6,50.7 97.0,50.7 97.3,50.7 97.7,50.8 98.1,50.8 98.4,50.9 98.8,50.9 99.1,50.9 99.5,51.0 99.8,51.0 100.2,51.1 100.6,51.1 100.9,51.2 101.3,51.2 101.6,51.3 102.0,51.3 102.3,51.3 102.7,51.4 103.1,51.4 103.4,51.5 103.8,51.5 104.1,51.6 104.5,51.6 104.8,51.7 105.2,51.7 105.5,51.8 105.9,51.8 106.3,51.9 106.6,51.9 107.0,52.0 107.3,52.0 107.7,52.0 108.0,52.1 108.4,52.1 108.8,52.2 109.1,52.2 109.5,52.3 109.8,52.3 110.2,52.4 110.5,52.4 110.9,52.5 111.3,52.5 111.6,52.6 112.0,52.6 112.3,52.6 112.7,52.7 113.0,52.7 113.4,52.8 113.8,52.8 114.1,52.9 114.5,52.9 114.8,52.9 115.2,53.0 115.5,53.0 115.9,53.1 116.2,53.1 116.6,53.1 117.0,53.2 117.3,53.2 117.7,53.2 118.0,53.3 118.4,53.3 118.7,53.3 119.1,53.3 119.5,53.4 119.8,53.4 120.2,53.4 120.5,53.4 120.9,53.5 121.2,53.5 121.6,53.5 122.0,53.5 122.3,53.5 122.7,53.5 123.0,53.6 123.4,53.6 123.7,53.6 124.1,53.6 124.5,53.6 124.8,53.6 125.2,53.6 125.5,53.6 125.9,53.6 126.2,53.6 126.6,53.6 127.0,53.5 127.3,53.5 127.7,53.5 128.0,53.5 128.4,53.5 128.7,53.5 129.1,53.4 129.4,53.4 129.8,53.4 130.2,53.3 130.5,53.3 130.9,53.3 131.2,53.2 131.6,53.2 131.9,53.1 132.3,53.1 132.7,53.0 133.0,53.0 133.4,52.9 133.7,52.8 134.1,52.8 134.4,52.7 134.8,52.6 135.2,52.5 135.5,52.5 135.9,52.4 136.2,52.3 136.6,52.2 136.9,52.1 137.3,52.0 137.6,51.9 138.0,51.8 138.4,51.7 138.7,51.6 139.1,51.5 139.4,51.4 139.8,51.2 140.1,51.1 140.5,51.0 140.9,50.9 141.2,50.7 141.6,50.6 141.9,50.4 142.3,50.3 142.6,50.2 143.0,50.0 143.4,50.0 143.7,50.0 144.1,50.0 144.4,50.0 144.8,50.0 145.1,50.0 145.5,50.0 145.9,50.0 146.2,50.0 146.6,50.0 146.9,50.0 147.3,50.0 147.6,50.0 148.0,50.0 148.4,50.0 148.7,50.0 149.1,50.0 149.4,50.0 149.8,50.0 150.1,50.0 150.5,50.0 150.8,50.0 151.2,50.0 151.6,50.0 151.9,50.0 152.3,50.0 152.6,50.0 153.0,50.0 153.3,50.0 153.7,50.0 154.1,50.0 154.4,50.0 154.8,50.0 155.1,50.0 155.5,50.0 155.8,50.0 156.2,50.0 156.6,50.0 156.9,50.0 157.3,50.0 157.6,50.0 158.0,50.0 158.3,50.0 158.7,50.0 159.1,50.0 159.4,50.0 159.8,50.0 160.1,50.0 160.5,50.0 160.8,50.0 161.2,50.0 161.5,50.0 161.9,50.0 162.3,50.0 162.6,50.0 163.0,50.0 163.3,50.0 163.7,50.0 164.0,50.0 164.4,50.0 164.8,50.0 165.1,50.0 165.5,50.0 165.8,50.0 166.2,50.0 166.5,50.0 166.9,50.0 167.3,50.0 167.6,50.0 168.0,50.0 168.3,50.0 168.7,50.0 169.0,50.0 169.4,50.0 169.8,50.0 170.1,50.0 170.5,50.0 170.8,50.0 171.2,50.0 171.5,50.0 171.9,50.0 172.2,50.0 172.6,50.0 173.0,50.0 173.3,50.0 173.7,50.0 174.0,50.0 174.4,50.0 174.7,50.0 175.1,50.0 175.5,50.0 175.8,50.0 176.2,50.0 176.5,50.0 176.9,50.0 177.2,50.0 177.6,50.0 178.0,50.0 178.3,50.0 178.7,50.0 179.0,50.0 179.4,50.0 179.7,50.0 180.1,50.0 180.4,50.0 180.8,50.0 181.2,50.0 181.5,50.0 181.9,50.0 182.2,50.0 182.6,50.0 182.9,50.0 183.3,50.0 183.7,50.0 184.0,50.0 184.4,50.0 184.7,50.0 185.1,50.0 185.4,50.0 185.8,50.0 186.2,50.0 186.5,50.0 186.9,50.0 187.2,50.0 187.6,50.0 187.9,50.0 188.3,50.0 188.7,50.0 189.0,50.0 189.4,50.0 189.7,50.0 190.1,50.0 190.4,50.0 190.8,50.0 191.2,50.0 191.5,50.0 191.9,50.0 192.2,50.0 192.6,50.0 192.9,50.0 193.3,50.0 193.6,50.0 194.0,50.0 194.4,50.0 194.7,50.0 195.1,50.0 195.4,50.0 195.8,50.0 196.1,50.0 196.5,50.0 196.9,50.2 197.2,50.3 197.6,50.5 197.9,50.6 198.3,50.8 198.6,50.9 199.0,51.0 199.4,51.2 199.7,51.3 200.1,51.4 200.4,51.5 200.8,51.7 201.1,51.8 201.5,51.9 201.8,52.0 202.2,52.1 202.6,52.2 202.9,52.3 203.3,52.4 203.6,52.5 204.0,52.5 204.3,52.6 204.7,52.7 205.1,52.8 205.4,52.9 205.8,52.9 206.1,53.0 206.5,53.1 206.8,53.1 207.2,53.2 207.6,53.2 207.9,53.3 208.3,53.3 208.6,53.4 209.0,53.4 209.3,53.4 209.7,53.5 210.1,53.5 210.4,53.5 210.8,53.6 211.1,53.6 211.5,53.6 211.8,53.6 212.2,53.6 212.5,53.7 212.9,53.7 213.3,53.7 213.6,53.7 214.0,53.7 214.3,53.7 214.7,53.7 215.0,53.7 215.4,53.7 215.8,53.7 216.1,53.7 216.5,53.7 216.8,53.7 217.2,53.6 217.5,53.6 217.9,53.6 218.3,53.6 218.6,53.6 219.0,53.6 219.3,53.5 219.7,53.5 220.0,53.5 220.4,53.5 220.8,53.4 221.1,53.4 221.5,53.4 221.8,53.3 222.2,53.3 222.5,53.3 222.9,53.2 223.2,53.2 223.6,53.2 224.0,53.1 224.3,53.1 224.7,53.0 225.0,53.0 225.4,53.0 225.7,52.9 226.1,52.9 226.5,52.8 226.8,52.8 227.2,52.7 227.5,52.7 227.9,52.6 228.2,52.6 228.6,52.6 229.0,52.5 229.3,52.5 229.7,52.4 230.0,52.4 230.4,52.3 230.7,52.3 231.1,52.2 231.5,52.2 231.8,52.1 232.2,52.1 232.5,52.0 232.9,52.0 233.2,51.9 233.6,51.9 234.0,51.8 234.3,51.8 234.7,51.7 235.0,51.7 235.4,51.6 235.7,51.6 236.1,51.5 236.4,51.5 236.8,51.4 237.2,51.4 237.5,51.3 237.9,51.3 238.2,51.2 238.6,51.2 238.9,51.2 239.3,51.1 239.7,51.1 240.0,51.0 240.4,51.0 240.7,50.9 241.1,50.9 241.4,50.9 241.8,50.8 242.2,50.8 242.5,50.7 242.9,50.7 243.2,50.6 243.6,50.6 243.9,50.6 244.3,50.5 244.7,50.5 245.0,50.5 245.4,50.4 245.7,50.4 246.1,50.3 246.4,50.3 246.8,50.3 247.1,50.2 247.5,50.2 247.9,50.2 248.2,50.1 248.6,50.1 248.9,50.1 249.3,50.1 249.6,50.0 250.0,50.0 L 250.0,50 Z" /></g>
        <polyline points="36.0,49.8 38.1,49.8 40.3,49.7 42.4,49.7 44.6,49.7 46.7,49.7 48.8,49.7 51.0,49.6 53.1,49.6 55.3,49.6 57.4,49.5 59.5,49.5 61.7,49.5 63.8,49.4 66.0,49.4 68.1,49.4 70.2,49.3 72.4,49.3 74.5,49.2 76.7,49.2 78.8,49.1 80.9,49.0 83.1,49.0 85.2,48.9 87.4,48.8 89.5,48.7 91.6,48.6 93.8,48.5 95.9,48.4 98.1,48.3 100.2,48.2 102.3,48.0 104.5,47.9 106.6,47.7 108.8,47.6 110.9,47.4 113.0,47.2 115.2,47.0 117.3,46.8 119.5,46.6 121.6,46.3 123.7,46.1 125.9,45.8 128.0,45.5 130.2,45.1 132.3,44.8 134.4,44.4 136.6,44.0 138.7,43.6 140.9,43.1 143.0,42.6 145.1,42.1 147.3,41.5 149.4,40.9 151.6,40.2 153.7,39.5 155.8,38.8 158.0,37.9 160.1,37.1 162.3,36.1 164.4,35.1 166.5,34.1 168.7,32.9 170.8,32.3 173.0,33.5 175.1,34.6 177.2,35.7 179.4,36.6 181.5,37.5 183.7,38.4 185.8,39.2 187.9,39.9 190.1,40.6 192.2,41.2 194.4,41.8 196.5,42.4 198.6,42.9 200.8,43.4 202.9,43.8 205.1,44.2 207.2,44.6 209.3,45.0 211.5,45.3 213.6,45.6 215.8,45.9 217.9,46.2 220.0,46.5 222.2,46.7 224.3,46.9 226.5,47.1 228.6,47.3 230.7,47.5 232.9,47.7 235.0,47.8 237.2,48.0 239.3,48.1 241.4,48.2 243.6,48.4 245.7,48.5 247.9,48.6 250.0,48.7" fill="none" stroke="#666" stroke-width="0.7" stroke-dasharray="2,2" />
        <polyline points="36.0,50.2 38.1,50.2 40.3,50.3 42.4,50.3 44.6,50.3 46.7,50.3 48.8,50.3 51.0,50.4 53.1,50.4 55.3,50.4 57.4,50.5 59.5,50.5 61.7,50.5 63.8,50.6 66.0,50.6 68.1,50.6 70.2,50.7 72.4,50.7 74.5,50.8 76.7,50.8 78.8,50.9 80.9,51.0 83.1,51.0 85.2,51.1 87.4,51.2 89.5,51.3 91.6,51.4 93.8,51.5 95.9,51.6 98.1,51.7 100.2,51.8 102.3,52.0 104.5,52.1 106.6,52.3 108.8,52.4 110.9,52.6 113.0,52.8 115.2,53.0 117.3,53.2 119.5,53.4 121.6,53.7 123.7,53.9 125.9,54.2 128.0,54.5 130.2,54.9 132.3,55.2 134.4,55.6 136.6,56.0 138.7,56.4 140.9,56.9 143.0,57.4 145.1,57.9 147.3,58.5 149.4,59.1 151.6,59.8 153.7,60.5 155.8,61.2 158.0,62.1 160.1,62.9 162.3,63.9 164.4,64.9 166.5,65.9 168.7,67.1 170.8,67.7 173.0,66.5 175.1,65.4 177.2,64.3 179.4,63.4 181.5,62.5 183.7,61.6 185.8,60.8 187.9,60.1 190.1,59.4 192.2,58.8 194.4,58.2 196.5,57.6 198.6,57.1 200.8,56.6 202.9,56.2 205.1,55.8 207.2,55.4 209.3,55.0 211.5,54.7 213.6,54.4 215.8,54.1 217.9,53.8 220.0,53.5 222.2,53.3 224.3,53.1 226.5,52.9 228.6,52.7 230.7,52.5 232.9,52.3 235.0,52.2 237.2,52.0 239.3,51.9 241.4,51.8 243.6,51.6 245.7,51.5 247.9,51.4 250.0,51.3" fill="none" stroke="#666" stroke-width="0.7" stroke-dasharray="2,2" />
        <polyline points="36.0,50.0 36.4,50.0 36.7,50.0 37.1,50.0 37.4,50.0 37.8,50.0 38.1,50.0 38.5,50.0 38.9,50.0 39.2,50.0 39.6,49.9 39.9,49.9 40.3,49.9 40.6,49.9 41.0,49.9 41.4,49.9 41.7,49.9 42.1,49.9 42.4,49.9 42.8,49.9 43.1,49.9 43.5,49.9 43.8,49.9 44.2,49.9 44.6,49.9 44.9,49.8 45.3,49.8 45.6,49.8 46.0,49.8 46.3,49.8 46.7,49.8 47.1,49.8 47.4,49.8 47.8,49.8 48.1,49.8 48.5,49.8 48.8,49.8 49.2,49.8 49.6,49.7 49.9,49.7 50.3,49.7 50.6,49.7 51.0,49.7 51.3,49.7 51.7,49.7 52.0,49.7 52.4,49.7 52.8,49.7 53.1,49.7 53.5,49.7 53.8,49.7 54.2,49.6 54.5,49.6 54.9,49.6 55.3,49.6 55.6,49.6 56.0,49.6 56.3,49.6 56.7,49.6 57.0,49.6 57.4,49.6 57.8,49.6 58.1,49.6 58.5,49.5 58.8,49.5 59.2,49.5 59.5,49.5 59.9,49.5 60.3,49.5 60.6,49.5 61.0,49.5 61.3,49.5 61.7,49.5 62.0,49.5 62.4,49.5 62.8,49.5 63.1,49.5 63.5,49.4 63.8,49.4 64.2,49.4 64.5,49.4 64.9,49.4 65.2,49.4 65.6,49.4 66.0,49.4 66.3,49.4 66.7,49.4 67.0,49.4 67.4,49.4 67.7,49.4 68.1,49.4 68.5,49.4 68.8,49.4 69.2,49.4 69.5,49.4 69.9,49.4 70.2,49.4 70.6,49.4 71.0,49.4 71.3,49.4 71.7,49.4 72.0,49.4 72.4,49.4 72.7,49.4 73.1,49.4 73.5,49.4 73.8,49.4 74.2,49.4 74.5,49.4 74.9,49.4 75.2,49.4 75.6,49.4 75.9,49.4 76.3,49.4 76.7,49.4 77.0,49.4 77.4,49.4 77.7,49.4 78.1,49.4 78.4,49.5 78.8,49.5 79.2,49.5 79.5,49.5 79.9,49.5 80.2,49.5 80.6,49.5 80.9,49.5 81.3,49.5 81.7,49.6 82.0,49.6 82.4,49.6 82.7,49.6 83.1,49.6 83.4,49.6 83.8,49.6 84.2,49.7 84.5,49.7 84.9,49.7 85.2,49.7 85.6,49.7 85.9,49.8 86.3,49.8 86.6,49.8 87.0,49.8 87.4,49.8 87.7,49.9 88.1,49.9 88.4,49.9 88.8,49.9 89.1,50.0 89.5,50.0 89.9,50.0 90.2,50.1 90.6,50.1 90.9,50.1 91.3,50.1 91.6,50.2 92.0,50.2 92.4,50.2 92.7,50.3 93.1,50.3 93.4,50.3 93.8,50.4 94.1,50.4 94.5,50.4 94.8,50.5 95.2,50.5 95.6,50.5 95.9,50.6 96.3,50.6 96.6,50.7 97.0,50.7 97.3,50.7 97.7,50.8 98.1,50.8 98.4,50.9 98.8,50.9 99.1,50.9 99.5,51.0 99.8,51.0 100.2,51.1 100.6,51.1 100.9,51.2 101.3,51.2 101.6,51.3 102.0,51.3 102.3,51.3 102.7,51.4 103.1,51.4 103.4,51.5 103.8,51.5 104.1,51.6 104.5,51.6 104.8,51.7 105.2,51.7 105.5,51.8 105.9,51.8 106.3,51.9 106.6,51.9 107.0,52.0 107.3,52.0 107.7,52.0 108.0,52.1 108.4,52.1 108.8,52.2 109.1,52.2 109.5,52.3 109.8,52.3 110.2,52.4 110.5,52.4 110.9,52.5 111.3,52.5 111.6,52.6 112.0,52.6 112.3,52.6 112.7,52.7 113.0,52.7 113.4,52.8 113.8,52.8 114.1,52.9 114.5,52.9 114.8,52.9 115.2,53.0 115.5,53.0 115.9,53.1 116.2,53.1 116.6,53.1 117.0,53.2 117.3,53.2 117.7,53.2 118.0,53.3 118.4,53.3 118.7,53.3 119.1,53.3 119.5,53.4 119.8,53.4 120.2,53.4 120.5,53.4 120.9,53.5 121.2,53.5 121.6,53.5 122.0,53.5 122.3,53.5 122.7,53.5 123.0,53.6 123.4,53.6 123.7,53.6 124.1,53.6 124.5,53.6 124.8,53.6 125.2,53.6 125.5,53.6 125.9,53.6 126.2,53.6 126.6,53.6 127.0,53.5 127.3,53.5 127.7,53.5 128.0,53.5 128.4,53.5 128.7,53.5 129.1,53.4 129.4,53.4 129.8,53.4 130.2,53.3 130.5,53.3 130.9,53.3 131.2,53.2 131.6,53.2 131.9,53.1 132.3,53.1 132.7,53.0 133.0,53.0 133.4,52.9 133.7,52.8 134.1,52.8 134.4,52.7 134.8,52.6 135.2,52.5 135.5,52.5 135.9,52.4 136.2,52.3 136.6,52.2 136.9,52.1 137.3,52.0 137.6,51.9 138.0,51.8 138.4,51.7 138.7,51.6 139.1,51.5 139.4,51.4 139.8,51.2 140.1,51.1 140.5,51.0 140.9,50.9 141.2,50.7 141.6,50.6 141.9,50.4 142.3,50.3 142.6,50.2 143.0,50.0 143.4,49.8 143.7,49.7 144.1,49.5 144.4,49.4 144.8,49.2 145.1,49.0 145.5,48.8 145.9,48.6 146.2,48.5 146.6,48.3 146.9,48.1 147.3,47.9 147.6,47.7 148.0,47.5 148.4,47.3 148.7,47.1 149.1,46.9 149.4,46.6 149.8,46.4 150.1,46.2 150.5,46.0 150.8,45.7 151.2,45.5 151.6,45.3 151.9,45.1 152.3,44.8 152.6,44.6 153.0,44.3 153.3,44.1 153.7,43.8 154.1,43.6 154.4,43.3 154.8,43.1 155.1,42.8 155.5,42.6 155.8,42.3 156.2,42.0 156.6,41.8 156.9,41.5 157.3,41.2 157.6,41.0 158.0,40.7 158.3,40.4 158.7,40.2 159.1,39.9 159.4,39.6 159.8,39.4 160.1,39.1 160.5,38.8 160.8,38.5 161.2,38.3 161.5,38.0 161.9,37.7 162.3,37.4 162.6,37.2 163.0,36.9 163.3,36.6 163.7,36.4 164.0,36.1 164.4,35.9 164.8,35.6 165.1,35.3 165.5,35.1 165.8,34.8 166.2,34.6 166.5,34.3 166.9,34.1 167.3,33.9 167.6,33.6 168.0,33.4 168.3,33.2 168.7,32.9 169.0,32.7 169.4,32.5 169.8,32.3 170.1,32.1 170.5,32.1 170.8,32.4 171.2,32.6 171.5,32.8 171.9,33.1 172.2,33.3 172.6,33.6 173.0,33.8 173.3,34.1 173.7,34.3 174.0,34.6 174.4,34.9 174.7,35.1 175.1,35.4 175.5,35.7 175.8,35.9 176.2,36.2 176.5,36.5 176.9,36.8 177.2,37.0 177.6,37.3 178.0,37.6 178.3,37.9 178.7,38.2 179.0,38.4 179.4,38.7 179.7,39.0 180.1,39.3 180.4,39.6 180.8,39.8 181.2,40.1 181.5,40.4 181.9,40.7 182.2,41.0 182.6,41.2 182.9,41.5 183.3,41.8 183.7,42.0 184.0,42.3 184.4,42.6 184.7,42.8 185.1,43.1 185.4,43.4 185.8,43.6 186.2,43.9 186.5,44.1 186.9,44.4 187.2,44.6 187.6,44.9 187.9,45.1 188.3,45.4 188.7,45.6 189.0,45.8 189.4,46.1 189.7,46.3 190.1,46.5 190.4,46.8 190.8,47.0 191.2,47.2 191.5,47.4 191.9,47.6 192.2,47.8 192.6,48.0 192.9,48.2 193.3,48.4 193.6,48.6 194.0,48.8 194.4,49.0 194.7,49.2 195.1,49.3 195.4,49.5 195.8,49.7 196.1,49.8 196.5,50.0 196.9,50.2 197.2,50.3 197.6,50.5 197.9,50.6 198.3,50.8 198.6,50.9 199.0,51.0 199.4,51.2 199.7,51.3 200.1,51.4 200.4,51.5 200.8,51.7 201.1,51.8 201.5,51.9 201.8,52.0 202.2,52.1 202.6,52.2 202.9,52.3 203.3,52.4 203.6,52.5 204.0,52.5 204.3,52.6 204.7,52.7 205.1,52.8 205.4,52.9 205.8,52.9 206.1,53.0 206.5,53.1 206.8,53.1 207.2,53.2 207.6,53.2 207.9,53.3 208.3,53.3 208.6,53.4 209.0,53.4 209.3,53.4 209.7,53.5 210.1,53.5 210.4,53.5 210.8,53.6 211.1,53.6 211.5,53.6 211.8,53.6 212.2,53.6 212.5,53.7 212.9,53.7 213.3,53.7 213.6,53.7 214.0,53.7 214.3,53.7 214.7,53.7 215.0,53.7 215.4,53.7 215.8,53.7 216.1,53.7 216.5,53.7 216.8,53.7 217.2,53.6 217.5,53.6 217.9,53.6 218.3,53.6 218.6,53.6 219.0,53.6 219.3,53.5 219.7,53.5 220.0,53.5 220.4,53.5 220.8,53.4 221.1,53.4 221.5,53.4 221.8,53.3 222.2,53.3 222.5,53.3 222.9,53.2 223.2,53.2 223.6,53.2 224.0,53.1 224.3,53.1 224.7,53.0 225.0,53.0 225.4,53.0 225.7,52.9 226.1,52.9 226.5,52.8 226.8,52.8 227.2,52.7 227.5,52.7 227.9,52.6 228.2,52.6 228.6,52.6 229.0,52.5 229.3,52.5 229.7,52.4 230.0,52.4 230.4,52.3 230.7,52.3 231.1,52.2 231.5,52.2 231.8,52.1 232.2,52.1 232.5,52.0 232.9,52.0 233.2,51.9 233.6,51.9 234.0,51.8 234.3,51.8 234.7,51.7 235.0,51.7 235.4,51.6 235.7,51.6 236.1,51.5 236.4,51.5 236.8,51.4 237.2,51.4 237.5,51.3 237.9,51.3 238.2,51.2 238.6,51.2 238.9,51.2 239.3,51.1 239.7,51.1 240.0,51.0 240.4,51.0 240.7,50.9 241.1,50.9 241.4,50.9 241.8,50.8 242.2,50.8 242.5,50.7 242.9,50.7 243.2,50.6 243.6,50.6 243.9,50.6 244.3,50.5 244.7,50.5 245.0,50.5 245.4,50.4 245.7,50.4 246.1,50.3 246.4,50.3 246.8,50.3 247.1,50.2 247.5,50.2 247.9,50.2 248.2,50.1 248.6,50.1 248.9,50.1 249.3,50.1 249.6,50.0 250.0,50.0" fill="none" stroke="#1f2430" stroke-width="0.8" />
        <text x="31.0" y="53" font-size="8.5" fill="#666" text-anchor="end">n = 2</text>
        <text x="250.0" y="28.0" font-size="8.5" fill="#1f2430" text-anchor="end">⟨e₂, v⟩ ≈ 0.226</text>
        <line x1="36.0" y1="100" x2="250.0" y2="100" stroke="#bbb" stroke-width="0.8" />
        <g fill="#1565c0" opacity="0.4"><path d="M 36.0,100 L 36.0,100.0 36.4,100.0 36.7,100.0 37.1,100.0 37.4,100.0 37.8,100.0 38.1,100.0 38.5,100.0 38.9,100.0 39.2,100.0 39.6,100.0 39.9,100.0 40.3,100.0 40.6,100.0 41.0,100.0 41.4,100.0 41.7,100.0 42.1,100.0 42.4,100.0 42.8,100.0 43.1,100.0 43.5,100.0 43.8,100.0 44.2,100.0 44.6,100.0 44.9,100.0 45.3,100.0 45.6,100.0 46.0,100.0 46.3,100.0 46.7,100.0 47.1,100.0 47.4,100.0 47.8,100.0 48.1,100.0 48.5,100.0 48.8,100.0 49.2,100.0 49.6,100.0 49.9,100.0 50.3,100.0 50.6,100.0 51.0,100.0 51.3,100.0 51.7,100.0 52.0,100.0 52.4,100.0 52.8,100.0 53.1,100.0 53.5,100.0 53.8,100.0 54.2,100.0 54.5,100.0 54.9,100.0 55.3,100.0 55.6,100.0 56.0,100.0 56.3,100.0 56.7,100.0 57.0,100.0 57.4,100.0 57.8,100.0 58.1,100.0 58.5,99.9 58.8,99.9 59.2,99.9 59.5,99.8 59.9,99.8 60.3,99.8 60.6,99.8 61.0,99.7 61.3,99.7 61.7,99.7 62.0,99.7 62.4,99.6 62.8,99.6 63.1,99.6 63.5,99.6 63.8,99.5 64.2,99.5 64.5,99.5 64.9,99.5 65.2,99.5 65.6,99.4 66.0,99.4 66.3,99.4 66.7,99.4 67.0,99.4 67.4,99.4 67.7,99.4 68.1,99.4 68.5,99.4 68.8,99.3 69.2,99.3 69.5,99.3 69.9,99.3 70.2,99.3 70.6,99.3 71.0,99.4 71.3,99.4 71.7,99.4 72.0,99.4 72.4,99.4 72.7,99.4 73.1,99.4 73.5,99.5 73.8,99.5 74.2,99.5 74.5,99.5 74.9,99.6 75.2,99.6 75.6,99.6 75.9,99.7 76.3,99.7 76.7,99.7 77.0,99.8 77.4,99.8 77.7,99.9 78.1,99.9 78.4,100.0 78.8,100.0 79.2,100.0 79.5,100.0 79.9,100.0 80.2,100.0 80.6,100.0 80.9,100.0 81.3,100.0 81.7,100.0 82.0,100.0 82.4,100.0 82.7,100.0 83.1,100.0 83.4,100.0 83.8,100.0 84.2,100.0 84.5,100.0 84.9,100.0 85.2,100.0 85.6,100.0 85.9,100.0 86.3,100.0 86.6,100.0 87.0,100.0 87.4,100.0 87.7,100.0 88.1,100.0 88.4,100.0 88.8,100.0 89.1,100.0 89.5,100.0 89.9,100.0 90.2,100.0 90.6,100.0 90.9,100.0 91.3,100.0 91.6,100.0 92.0,100.0 92.4,100.0 92.7,100.0 93.1,100.0 93.4,100.0 93.8,100.0 94.1,100.0 94.5,100.0 94.8,100.0 95.2,100.0 95.6,100.0 95.9,100.0 96.3,100.0 96.6,100.0 97.0,100.0 97.3,100.0 97.7,100.0 98.1,100.0 98.4,100.0 98.8,100.0 99.1,100.0 99.5,100.0 99.8,100.0 100.2,100.0 100.6,99.9 100.9,99.8 101.3,99.7 101.6,99.6 102.0,99.5 102.3,99.4 102.7,99.3 103.1,99.2 103.4,99.1 103.8,99.0 104.1,98.9 104.5,98.8 104.8,98.7 105.2,98.6 105.5,98.5 105.9,98.4 106.3,98.3 106.6,98.2 107.0,98.1 107.3,98.0 107.7,97.9 108.0,97.8 108.4,97.8 108.8,97.7 109.1,97.6 109.5,97.6 109.8,97.5 110.2,97.5 110.5,97.4 110.9,97.4 111.3,97.4 111.6,97.4 112.0,97.3 112.3,97.3 112.7,97.3 113.0,97.4 113.4,97.4 113.8,97.4 114.1,97.4 114.5,97.5 114.8,97.5 115.2,97.6 115.5,97.7 115.9,97.7 116.2,97.8 116.6,97.9 117.0,98.0 117.3,98.1 117.7,98.2 118.0,98.4 118.4,98.5 118.7,98.6 119.1,98.8 119.5,98.9 119.8,99.1 120.2,99.3 120.5,99.4 120.9,99.6 121.2,99.8 121.6,100.0 122.0,100.0 122.3,100.0 122.7,100.0 123.0,100.0 123.4,100.0 123.7,100.0 124.1,100.0 124.5,100.0 124.8,100.0 125.2,100.0 125.5,100.0 125.9,100.0 126.2,100.0 126.6,100.0 127.0,100.0 127.3,100.0 127.7,100.0 128.0,100.0 128.4,100.0 128.7,100.0 129.1,100.0 129.4,100.0 129.8,100.0 130.2,100.0 130.5,100.0 130.9,100.0 131.2,100.0 131.6,100.0 131.9,100.0 132.3,100.0 132.7,100.0 133.0,100.0 133.4,100.0 133.7,100.0 134.1,100.0 134.4,100.0 134.8,100.0 135.2,100.0 135.5,100.0 135.9,100.0 136.2,100.0 136.6,100.0 136.9,100.0 137.3,100.0 137.6,100.0 138.0,100.0 138.4,100.0 138.7,100.0 139.1,100.0 139.4,100.0 139.8,100.0 140.1,100.0 140.5,100.0 140.9,100.0 141.2,100.0 141.6,100.0 141.9,100.0 142.3,100.0 142.6,100.0 143.0,100.0 143.4,99.6 143.7,99.2 144.1,98.8 144.4,98.4 144.8,98.0 145.1,97.5 145.5,97.1 145.9,96.7 146.2,96.3 146.6,95.8 146.9,95.4 147.3,95.0 147.6,94.6 148.0,94.2 148.4,93.8 148.7,93.4 149.1,93.0 149.4,92.6 149.8,92.3 150.1,91.9 150.5,91.6 150.8,91.3 151.2,91.0 151.6,90.7 151.9,90.4 152.3,90.2 152.6,90.0 153.0,89.8 153.3,89.6 153.7,89.5 154.1,89.4 154.4,89.3 154.8,89.3 155.1,89.2 155.5,89.3 155.8,89.3 156.2,89.4 156.6,89.5 156.9,89.6 157.3,89.8 157.6,90.0 158.0,90.2 158.3,90.5 158.7,90.8 159.1,91.2 159.4,91.5 159.8,92.0 160.1,92.4 160.5,92.9 160.8,93.4 161.2,93.9 161.5,94.5 161.9,95.1 162.3,95.7 162.6,96.4 163.0,97.0 163.3,97.8 163.7,98.5 164.0,99.2 164.4,100.0 164.8,100.0 165.1,100.0 165.5,100.0 165.8,100.0 166.2,100.0 166.5,100.0 166.9,100.0 167.3,100.0 167.6,100.0 168.0,100.0 168.3,100.0 168.7,100.0 169.0,100.0 169.4,100.0 169.8,100.0 170.1,100.0 170.5,100.0 170.8,100.0 171.2,100.0 171.5,100.0 171.9,100.0 172.2,100.0 172.6,100.0 173.0,100.0 173.3,100.0 173.7,100.0 174.0,100.0 174.4,100.0 174.7,100.0 175.1,100.0 175.5,100.0 175.8,100.0 176.2,100.0 176.5,100.0 176.9,100.0 177.2,100.0 177.6,100.0 178.0,100.0 178.3,100.0 178.7,100.0 179.0,100.0 179.4,100.0 179.7,100.0 180.1,100.0 180.4,100.0 180.8,100.0 181.2,100.0 181.5,100.0 181.9,100.0 182.2,100.0 182.6,100.0 182.9,100.0 183.3,100.0 183.7,100.0 184.0,100.0 184.4,100.0 184.7,100.0 185.1,100.0 185.4,100.0 185.8,100.0 186.2,99.4 186.5,98.9 186.9,98.4 187.2,97.8 187.6,97.4 187.9,96.9 188.3,96.4 188.7,96.0 189.0,95.6 189.4,95.2 189.7,94.8 190.1,94.5 190.4,94.1 190.8,93.8 191.2,93.6 191.5,93.3 191.9,93.1 192.2,92.9 192.6,92.7 192.9,92.6 193.3,92.4 193.6,92.3 194.0,92.3 194.4,92.2 194.7,92.2 195.1,92.2 195.4,92.2 195.8,92.2 196.1,92.3 196.5,92.4 196.9,92.5 197.2,92.6 197.6,92.7 197.9,92.9 198.3,93.0 198.6,93.2 199.0,93.4 199.4,93.6 199.7,93.9 200.1,94.1 200.4,94.4 200.8,94.6 201.1,94.9 201.5,95.2 201.8,95.5 202.2,95.8 202.6,96.1 202.9,96.4 203.3,96.7 203.6,97.0 204.0,97.3 204.3,97.6 204.7,97.9 205.1,98.2 205.4,98.5 205.8,98.8 206.1,99.1 206.5,99.4 206.8,99.7 207.2,100.0 207.6,100.0 207.9,100.0 208.3,100.0 208.6,100.0 209.0,100.0 209.3,100.0 209.7,100.0 210.1,100.0 210.4,100.0 210.8,100.0 211.1,100.0 211.5,100.0 211.8,100.0 212.2,100.0 212.5,100.0 212.9,100.0 213.3,100.0 213.6,100.0 214.0,100.0 214.3,100.0 214.7,100.0 215.0,100.0 215.4,100.0 215.8,100.0 216.1,100.0 216.5,100.0 216.8,100.0 217.2,100.0 217.5,100.0 217.9,100.0 218.3,100.0 218.6,100.0 219.0,100.0 219.3,100.0 219.7,100.0 220.0,100.0 220.4,100.0 220.8,100.0 221.1,100.0 221.5,100.0 221.8,100.0 222.2,100.0 222.5,100.0 222.9,100.0 223.2,100.0 223.6,100.0 224.0,100.0 224.3,100.0 224.7,100.0 225.0,100.0 225.4,100.0 225.7,100.0 226.1,100.0 226.5,100.0 226.8,100.0 227.2,100.0 227.5,100.0 227.9,100.0 228.2,100.0 228.6,100.0 229.0,99.9 229.3,99.7 229.7,99.6 230.0,99.5 230.4,99.3 230.7,99.2 231.1,99.1 231.5,99.0 231.8,98.9 232.2,98.8 232.5,98.7 232.9,98.6 233.2,98.5 233.6,98.5 234.0,98.4 234.3,98.3 234.7,98.3 235.0,98.2 235.4,98.2 235.7,98.2 236.1,98.1 236.4,98.1 236.8,98.1 237.2,98.1 237.5,98.1 237.9,98.1 238.2,98.1 238.6,98.1 238.9,98.1 239.3,98.1 239.7,98.1 240.0,98.2 240.4,98.2 240.7,98.2 241.1,98.3 241.4,98.3 241.8,98.4 242.2,98.4 242.5,98.5 242.9,98.5 243.2,98.6 243.6,98.7 243.9,98.7 244.3,98.8 244.7,98.9 245.0,98.9 245.4,99.0 245.7,99.1 246.1,99.2 246.4,99.3 246.8,99.3 247.1,99.4 247.5,99.5 247.9,99.6 248.2,99.6 248.6,99.7 248.9,99.8 249.3,99.9 249.6,99.9 250.0,100.0 L 250.0,100 Z" /></g>
        <g fill="#c62828" opacity="0.4"><path d="M 36.0,100 L 36.0,100.0 36.4,100.0 36.7,100.0 37.1,100.0 37.4,100.0 37.8,100.1 38.1,100.1 38.5,100.1 38.9,100.1 39.2,100.1 39.6,100.1 39.9,100.1 40.3,100.2 40.6,100.2 41.0,100.2 41.4,100.2 41.7,100.2 42.1,100.2 42.4,100.2 42.8,100.2 43.1,100.2 43.5,100.3 43.8,100.3 44.2,100.3 44.6,100.3 44.9,100.3 45.3,100.3 45.6,100.3 46.0,100.3 46.3,100.3 46.7,100.3 47.1,100.3 47.4,100.3 47.8,100.3 48.1,100.3 48.5,100.3 48.8,100.3 49.2,100.3 49.6,100.3 49.9,100.3 50.3,100.3 50.6,100.3 51.0,100.3 51.3,100.3 51.7,100.3 52.0,100.3 52.4,100.3 52.8,100.2 53.1,100.2 53.5,100.2 53.8,100.2 54.2,100.2 54.5,100.2 54.9,100.1 55.3,100.1 55.6,100.1 56.0,100.1 56.3,100.1 56.7,100.0 57.0,100.0 57.4,100.0 57.8,100.0 58.1,100.0 58.5,100.0 58.8,100.0 59.2,100.0 59.5,100.0 59.9,100.0 60.3,100.0 60.6,100.0 61.0,100.0 61.3,100.0 61.7,100.0 62.0,100.0 62.4,100.0 62.8,100.0 63.1,100.0 63.5,100.0 63.8,100.0 64.2,100.0 64.5,100.0 64.9,100.0 65.2,100.0 65.6,100.0 66.0,100.0 66.3,100.0 66.7,100.0 67.0,100.0 67.4,100.0 67.7,100.0 68.1,100.0 68.5,100.0 68.8,100.0 69.2,100.0 69.5,100.0 69.9,100.0 70.2,100.0 70.6,100.0 71.0,100.0 71.3,100.0 71.7,100.0 72.0,100.0 72.4,100.0 72.7,100.0 73.1,100.0 73.5,100.0 73.8,100.0 74.2,100.0 74.5,100.0 74.9,100.0 75.2,100.0 75.6,100.0 75.9,100.0 76.3,100.0 76.7,100.0 77.0,100.0 77.4,100.0 77.7,100.0 78.1,100.0 78.4,100.0 78.8,100.0 79.2,100.0 79.5,100.1 79.9,100.1 80.2,100.2 80.6,100.2 80.9,100.3 81.3,100.4 81.7,100.4 82.0,100.5 82.4,100.5 82.7,100.6 83.1,100.6 83.4,100.7 83.8,100.7 84.2,100.8 84.5,100.8 84.9,100.9 85.2,100.9 85.6,101.0 85.9,101.0 86.3,101.0 86.6,101.1 87.0,101.1 87.4,101.1 87.7,101.2 88.1,101.2 88.4,101.2 88.8,101.3 89.1,101.3 89.5,101.3 89.9,101.3 90.2,101.3 90.6,101.3 90.9,101.3 91.3,101.3 91.6,101.3 92.0,101.3 92.4,101.3 92.7,101.3 93.1,101.3 93.4,101.2 93.8,101.2 94.1,101.2 94.5,101.1 94.8,101.1 95.2,101.0 95.6,101.0 95.9,100.9 96.3,100.9 96.6,100.8 97.0,100.7 97.3,100.7 97.7,100.6 98.1,100.5 98.4,100.4 98.8,100.4 99.1,100.3 99.5,100.2 99.8,100.1 100.2,100.0 100.6,100.0 100.9,100.0 101.3,100.0 101.6,100.0 102.0,100.0 102.3,100.0 102.7,100.0 103.1,100.0 103.4,100.0 103.8,100.0 104.1,100.0 104.5,100.0 104.8,100.0 105.2,100.0 105.5,100.0 105.9,100.0 106.3,100.0 106.6,100.0 107.0,100.0 107.3,100.0 107.7,100.0 108.0,100.0 108.4,100.0 108.8,100.0 109.1,100.0 109.5,100.0 109.8,100.0 110.2,100.0 110.5,100.0 110.9,100.0 111.3,100.0 111.6,100.0 112.0,100.0 112.3,100.0 112.7,100.0 113.0,100.0 113.4,100.0 113.8,100.0 114.1,100.0 114.5,100.0 114.8,100.0 115.2,100.0 115.5,100.0 115.9,100.0 116.2,100.0 116.6,100.0 117.0,100.0 117.3,100.0 117.7,100.0 118.0,100.0 118.4,100.0 118.7,100.0 119.1,100.0 119.5,100.0 119.8,100.0 120.2,100.0 120.5,100.0 120.9,100.0 121.2,100.0 121.6,100.0 122.0,100.2 122.3,100.4 122.7,100.6 123.0,100.8 123.4,101.0 123.7,101.2 124.1,101.4 124.5,101.6 124.8,101.9 125.2,102.1 125.5,102.3 125.9,102.5 126.2,102.7 126.6,102.9 127.0,103.1 127.3,103.3 127.7,103.5 128.0,103.7 128.4,103.9 128.7,104.0 129.1,104.2 129.4,104.3 129.8,104.5 130.2,104.6 130.5,104.8 130.9,104.9 131.2,105.0 131.6,105.1 131.9,105.2 132.3,105.2 132.7,105.3 133.0,105.3 133.4,105.3 133.7,105.3 134.1,105.3 134.4,105.3 134.8,105.3 135.2,105.2 135.5,105.2 135.9,105.1 136.2,105.0 136.6,104.9 136.9,104.7 137.3,104.6 137.6,104.4 138.0,104.2 138.4,104.0 138.7,103.8 139.1,103.5 139.4,103.3 139.8,103.0 140.1,102.7 140.5,102.4 140.9,102.1 141.2,101.8 141.6,101.5 141.9,101.1 142.3,100.8 142.6,100.4 143.0,100.0 143.4,100.0 143.7,100.0 144.1,100.0 144.4,100.0 144.8,100.0 145.1,100.0 145.5,100.0 145.9,100.0 146.2,100.0 146.6,100.0 146.9,100.0 147.3,100.0 147.6,100.0 148.0,100.0 148.4,100.0 148.7,100.0 149.1,100.0 149.4,100.0 149.8,100.0 150.1,100.0 150.5,100.0 150.8,100.0 151.2,100.0 151.6,100.0 151.9,100.0 152.3,100.0 152.6,100.0 153.0,100.0 153.3,100.0 153.7,100.0 154.1,100.0 154.4,100.0 154.8,100.0 155.1,100.0 155.5,100.0 155.8,100.0 156.2,100.0 156.6,100.0 156.9,100.0 157.3,100.0 157.6,100.0 158.0,100.0 158.3,100.0 158.7,100.0 159.1,100.0 159.4,100.0 159.8,100.0 160.1,100.0 160.5,100.0 160.8,100.0 161.2,100.0 161.5,100.0 161.9,100.0 162.3,100.0 162.6,100.0 163.0,100.0 163.3,100.0 163.7,100.0 164.0,100.0 164.4,100.0 164.8,100.8 165.1,101.6 165.5,102.4 165.8,103.2 166.2,104.1 166.5,104.9 166.9,105.8 167.3,106.6 167.6,107.5 168.0,108.4 168.3,109.2 168.7,110.1 169.0,110.9 169.4,111.7 169.8,112.5 170.1,113.3 170.5,113.9 170.8,114.3 171.2,114.6 171.5,114.9 171.9,115.2 172.2,115.4 172.6,115.6 173.0,115.7 173.3,115.7 173.7,115.7 174.0,115.7 174.4,115.6 174.7,115.5 175.1,115.4 175.5,115.2 175.8,114.9 176.2,114.7 176.5,114.3 176.9,114.0 177.2,113.6 177.6,113.2 178.0,112.8 178.3,112.3 178.7,111.8 179.0,111.3 179.4,110.8 179.7,110.3 180.1,109.7 180.4,109.1 180.8,108.5 181.2,107.9 181.5,107.3 181.9,106.7 182.2,106.1 182.6,105.5 182.9,104.8 183.3,104.2 183.7,103.6 184.0,103.0 184.4,102.4 184.7,101.8 185.1,101.2 185.4,100.6 185.8,100.0 186.2,100.0 186.5,100.0 186.9,100.0 187.2,100.0 187.6,100.0 187.9,100.0 188.3,100.0 188.7,100.0 189.0,100.0 189.4,100.0 189.7,100.0 190.1,100.0 190.4,100.0 190.8,100.0 191.2,100.0 191.5,100.0 191.9,100.0 192.2,100.0 192.6,100.0 192.9,100.0 193.3,100.0 193.6,100.0 194.0,100.0 194.4,100.0 194.7,100.0 195.1,100.0 195.4,100.0 195.8,100.0 196.1,100.0 196.5,100.0 196.9,100.0 197.2,100.0 197.6,100.0 197.9,100.0 198.3,100.0 198.6,100.0 199.0,100.0 199.4,100.0 199.7,100.0 200.1,100.0 200.4,100.0 200.8,100.0 201.1,100.0 201.5,100.0 201.8,100.0 202.2,100.0 202.6,100.0 202.9,100.0 203.3,100.0 203.6,100.0 204.0,100.0 204.3,100.0 204.7,100.0 205.1,100.0 205.4,100.0 205.8,100.0 206.1,100.0 206.5,100.0 206.8,100.0 207.2,100.0 207.6,100.3 207.9,100.6 208.3,100.8 208.6,101.1 209.0,101.3 209.3,101.6 209.7,101.8 210.1,102.0 210.4,102.2 210.8,102.4 211.1,102.6 211.5,102.8 211.8,102.9 212.2,103.1 212.5,103.2 212.9,103.3 213.3,103.4 213.6,103.5 214.0,103.6 214.3,103.7 214.7,103.8 215.0,103.8 215.4,103.9 215.8,103.9 216.1,103.9 216.5,103.9 216.8,103.9 217.2,103.9 217.5,103.8 217.9,103.8 218.3,103.8 218.6,103.7 219.0,103.6 219.3,103.6 219.7,103.5 220.0,103.4 220.4,103.3 220.8,103.2 221.1,103.1 221.5,102.9 221.8,102.8 222.2,102.7 222.5,102.5 222.9,102.4 223.2,102.3 223.6,102.1 224.0,102.0 224.3,101.8 224.7,101.7 225.0,101.5 225.4,101.4 225.7,101.2 226.1,101.0 226.5,100.9 226.8,100.7 227.2,100.6 227.5,100.4 227.9,100.3 228.2,100.1 228.6,100.0 229.0,100.0 229.3,100.0 229.7,100.0 230.0,100.0 230.4,100.0 230.7,100.0 231.1,100.0 231.5,100.0 231.8,100.0 232.2,100.0 232.5,100.0 232.9,100.0 233.2,100.0 233.6,100.0 234.0,100.0 234.3,100.0 234.7,100.0 235.0,100.0 235.4,100.0 235.7,100.0 236.1,100.0 236.4,100.0 236.8,100.0 237.2,100.0 237.5,100.0 237.9,100.0 238.2,100.0 238.6,100.0 238.9,100.0 239.3,100.0 239.7,100.0 240.0,100.0 240.4,100.0 240.7,100.0 241.1,100.0 241.4,100.0 241.8,100.0 242.2,100.0 242.5,100.0 242.9,100.0 243.2,100.0 243.6,100.0 243.9,100.0 244.3,100.0 244.7,100.0 245.0,100.0 245.4,100.0 245.7,100.0 246.1,100.0 246.4,100.0 246.8,100.0 247.1,100.0 247.5,100.0 247.9,100.0 248.2,100.0 248.6,100.0 248.9,100.0 249.3,100.0 249.6,100.0 250.0,100.0 L 250.0,100 Z" /></g>
        <polyline points="36.0,99.8 38.1,99.8 40.3,99.7 42.4,99.7 44.6,99.7 46.7,99.7 48.8,99.7 51.0,99.6 53.1,99.6 55.3,99.6 57.4,99.5 59.5,99.5 61.7,99.5 63.8,99.4 66.0,99.4 68.1,99.4 70.2,99.3 72.4,99.3 74.5,99.2 76.7,99.2 78.8,99.1 80.9,99.0 83.1,99.0 85.2,98.9 87.4,98.8 89.5,98.7 91.6,98.6 93.8,98.5 95.9,98.4 98.1,98.3 100.2,98.2 102.3,98.0 104.5,97.9 106.6,97.7 108.8,97.6 110.9,97.4 113.0,97.2 115.2,97.0 117.3,96.8 119.5,96.6 121.6,96.3 123.7,96.1 125.9,95.8 128.0,95.5 130.2,95.1 132.3,94.8 134.4,94.4 136.6,94.0 138.7,93.6 140.9,93.1 143.0,92.6 145.1,92.1 147.3,91.5 149.4,90.9 151.6,90.2 153.7,89.5 155.8,88.8 158.0,87.9 160.1,87.1 162.3,86.1 164.4,85.1 166.5,84.1 168.7,82.9 170.8,82.3 173.0,83.5 175.1,84.6 177.2,85.7 179.4,86.6 181.5,87.5 183.7,88.4 185.8,89.2 187.9,89.9 190.1,90.6 192.2,91.2 194.4,91.8 196.5,92.4 198.6,92.9 200.8,93.4 202.9,93.8 205.1,94.2 207.2,94.6 209.3,95.0 211.5,95.3 213.6,95.6 215.8,95.9 217.9,96.2 220.0,96.5 222.2,96.7 224.3,96.9 226.5,97.1 228.6,97.3 230.7,97.5 232.9,97.7 235.0,97.8 237.2,98.0 239.3,98.1 241.4,98.2 243.6,98.4 245.7,98.5 247.9,98.6 250.0,98.7" fill="none" stroke="#666" stroke-width="0.7" stroke-dasharray="2,2" />
        <polyline points="36.0,100.2 38.1,100.2 40.3,100.3 42.4,100.3 44.6,100.3 46.7,100.3 48.8,100.3 51.0,100.4 53.1,100.4 55.3,100.4 57.4,100.5 59.5,100.5 61.7,100.5 63.8,100.6 66.0,100.6 68.1,100.6 70.2,100.7 72.4,100.7 74.5,100.8 76.7,100.8 78.8,100.9 80.9,101.0 83.1,101.0 85.2,101.1 87.4,101.2 89.5,101.3 91.6,101.4 93.8,101.5 95.9,101.6 98.1,101.7 100.2,101.8 102.3,102.0 104.5,102.1 106.6,102.3 108.8,102.4 110.9,102.6 113.0,102.8 115.2,103.0 117.3,103.2 119.5,103.4 121.6,103.7 123.7,103.9 125.9,104.2 128.0,104.5 130.2,104.9 132.3,105.2 134.4,105.6 136.6,106.0 138.7,106.4 140.9,106.9 143.0,107.4 145.1,107.9 147.3,108.5 149.4,109.1 151.6,109.8 153.7,110.5 155.8,111.2 158.0,112.1 160.1,112.9 162.3,113.9 164.4,114.9 166.5,115.9 168.7,117.1 170.8,117.7 173.0,116.5 175.1,115.4 177.2,114.3 179.4,113.4 181.5,112.5 183.7,111.6 185.8,110.8 187.9,110.1 190.1,109.4 192.2,108.8 194.4,108.2 196.5,107.6 198.6,107.1 200.8,106.6 202.9,106.2 205.1,105.8 207.2,105.4 209.3,105.0 211.5,104.7 213.6,104.4 215.8,104.1 217.9,103.8 220.0,103.5 222.2,103.3 224.3,103.1 226.5,102.9 228.6,102.7 230.7,102.5 232.9,102.3 235.0,102.2 237.2,102.0 239.3,101.9 241.4,101.8 243.6,101.6 245.7,101.5 247.9,101.4 250.0,101.3" fill="none" stroke="#666" stroke-width="0.7" stroke-dasharray="2,2" />
        <polyline points="36.0,100.0 36.4,100.0 36.7,100.0 37.1,100.0 37.4,100.0 37.8,100.1 38.1,100.1 38.5,100.1 38.9,100.1 39.2,100.1 39.6,100.1 39.9,100.1 40.3,100.2 40.6,100.2 41.0,100.2 41.4,100.2 41.7,100.2 42.1,100.2 42.4,100.2 42.8,100.2 43.1,100.2 43.5,100.3 43.8,100.3 44.2,100.3 44.6,100.3 44.9,100.3 45.3,100.3 45.6,100.3 46.0,100.3 46.3,100.3 46.7,100.3 47.1,100.3 47.4,100.3 47.8,100.3 48.1,100.3 48.5,100.3 48.8,100.3 49.2,100.3 49.6,100.3 49.9,100.3 50.3,100.3 50.6,100.3 51.0,100.3 51.3,100.3 51.7,100.3 52.0,100.3 52.4,100.3 52.8,100.2 53.1,100.2 53.5,100.2 53.8,100.2 54.2,100.2 54.5,100.2 54.9,100.1 55.3,100.1 55.6,100.1 56.0,100.1 56.3,100.1 56.7,100.0 57.0,100.0 57.4,100.0 57.8,100.0 58.1,100.0 58.5,99.9 58.8,99.9 59.2,99.9 59.5,99.8 59.9,99.8 60.3,99.8 60.6,99.8 61.0,99.7 61.3,99.7 61.7,99.7 62.0,99.7 62.4,99.6 62.8,99.6 63.1,99.6 63.5,99.6 63.8,99.5 64.2,99.5 64.5,99.5 64.9,99.5 65.2,99.5 65.6,99.4 66.0,99.4 66.3,99.4 66.7,99.4 67.0,99.4 67.4,99.4 67.7,99.4 68.1,99.4 68.5,99.4 68.8,99.3 69.2,99.3 69.5,99.3 69.9,99.3 70.2,99.3 70.6,99.3 71.0,99.4 71.3,99.4 71.7,99.4 72.0,99.4 72.4,99.4 72.7,99.4 73.1,99.4 73.5,99.5 73.8,99.5 74.2,99.5 74.5,99.5 74.9,99.6 75.2,99.6 75.6,99.6 75.9,99.7 76.3,99.7 76.7,99.7 77.0,99.8 77.4,99.8 77.7,99.9 78.1,99.9 78.4,100.0 78.8,100.0 79.2,100.0 79.5,100.1 79.9,100.1 80.2,100.2 80.6,100.2 80.9,100.3 81.3,100.4 81.7,100.4 82.0,100.5 82.4,100.5 82.7,100.6 83.1,100.6 83.4,100.7 83.8,100.7 84.2,100.8 84.5,100.8 84.9,100.9 85.2,100.9 85.6,101.0 85.9,101.0 86.3,101.0 86.6,101.1 87.0,101.1 87.4,101.1 87.7,101.2 88.1,101.2 88.4,101.2 88.8,101.3 89.1,101.3 89.5,101.3 89.9,101.3 90.2,101.3 90.6,101.3 90.9,101.3 91.3,101.3 91.6,101.3 92.0,101.3 92.4,101.3 92.7,101.3 93.1,101.3 93.4,101.2 93.8,101.2 94.1,101.2 94.5,101.1 94.8,101.1 95.2,101.0 95.6,101.0 95.9,100.9 96.3,100.9 96.6,100.8 97.0,100.7 97.3,100.7 97.7,100.6 98.1,100.5 98.4,100.4 98.8,100.4 99.1,100.3 99.5,100.2 99.8,100.1 100.2,100.0 100.6,99.9 100.9,99.8 101.3,99.7 101.6,99.6 102.0,99.5 102.3,99.4 102.7,99.3 103.1,99.2 103.4,99.1 103.8,99.0 104.1,98.9 104.5,98.8 104.8,98.7 105.2,98.6 105.5,98.5 105.9,98.4 106.3,98.3 106.6,98.2 107.0,98.1 107.3,98.0 107.7,97.9 108.0,97.8 108.4,97.8 108.8,97.7 109.1,97.6 109.5,97.6 109.8,97.5 110.2,97.5 110.5,97.4 110.9,97.4 111.3,97.4 111.6,97.4 112.0,97.3 112.3,97.3 112.7,97.3 113.0,97.4 113.4,97.4 113.8,97.4 114.1,97.4 114.5,97.5 114.8,97.5 115.2,97.6 115.5,97.7 115.9,97.7 116.2,97.8 116.6,97.9 117.0,98.0 117.3,98.1 117.7,98.2 118.0,98.4 118.4,98.5 118.7,98.6 119.1,98.8 119.5,98.9 119.8,99.1 120.2,99.3 120.5,99.4 120.9,99.6 121.2,99.8 121.6,100.0 122.0,100.2 122.3,100.4 122.7,100.6 123.0,100.8 123.4,101.0 123.7,101.2 124.1,101.4 124.5,101.6 124.8,101.9 125.2,102.1 125.5,102.3 125.9,102.5 126.2,102.7 126.6,102.9 127.0,103.1 127.3,103.3 127.7,103.5 128.0,103.7 128.4,103.9 128.7,104.0 129.1,104.2 129.4,104.3 129.8,104.5 130.2,104.6 130.5,104.8 130.9,104.9 131.2,105.0 131.6,105.1 131.9,105.2 132.3,105.2 132.7,105.3 133.0,105.3 133.4,105.3 133.7,105.3 134.1,105.3 134.4,105.3 134.8,105.3 135.2,105.2 135.5,105.2 135.9,105.1 136.2,105.0 136.6,104.9 136.9,104.7 137.3,104.6 137.6,104.4 138.0,104.2 138.4,104.0 138.7,103.8 139.1,103.5 139.4,103.3 139.8,103.0 140.1,102.7 140.5,102.4 140.9,102.1 141.2,101.8 141.6,101.5 141.9,101.1 142.3,100.8 142.6,100.4 143.0,100.0 143.4,99.6 143.7,99.2 144.1,98.8 144.4,98.4 144.8,98.0 145.1,97.5 145.5,97.1 145.9,96.7 146.2,96.3 146.6,95.8 146.9,95.4 147.3,95.0 147.6,94.6 148.0,94.2 148.4,93.8 148.7,93.4 149.1,93.0 149.4,92.6 149.8,92.3 150.1,91.9 150.5,91.6 150.8,91.3 151.2,91.0 151.6,90.7 151.9,90.4 152.3,90.2 152.6,90.0 153.0,89.8 153.3,89.6 153.7,89.5 154.1,89.4 154.4,89.3 154.8,89.3 155.1,89.2 155.5,89.3 155.8,89.3 156.2,89.4 156.6,89.5 156.9,89.6 157.3,89.8 157.6,90.0 158.0,90.2 158.3,90.5 158.7,90.8 159.1,91.2 159.4,91.5 159.8,92.0 160.1,92.4 160.5,92.9 160.8,93.4 161.2,93.9 161.5,94.5 161.9,95.1 162.3,95.7 162.6,96.4 163.0,97.0 163.3,97.8 163.7,98.5 164.0,99.2 164.4,100.0 164.8,100.8 165.1,101.6 165.5,102.4 165.8,103.2 166.2,104.1 166.5,104.9 166.9,105.8 167.3,106.6 167.6,107.5 168.0,108.4 168.3,109.2 168.7,110.1 169.0,110.9 169.4,111.7 169.8,112.5 170.1,113.3 170.5,113.9 170.8,114.3 171.2,114.6 171.5,114.9 171.9,115.2 172.2,115.4 172.6,115.6 173.0,115.7 173.3,115.7 173.7,115.7 174.0,115.7 174.4,115.6 174.7,115.5 175.1,115.4 175.5,115.2 175.8,114.9 176.2,114.7 176.5,114.3 176.9,114.0 177.2,113.6 177.6,113.2 178.0,112.8 178.3,112.3 178.7,111.8 179.0,111.3 179.4,110.8 179.7,110.3 180.1,109.7 180.4,109.1 180.8,108.5 181.2,107.9 181.5,107.3 181.9,106.7 182.2,106.1 182.6,105.5 182.9,104.8 183.3,104.2 183.7,103.6 184.0,103.0 184.4,102.4 184.7,101.8 185.1,101.2 185.4,100.6 185.8,100.0 186.2,99.4 186.5,98.9 186.9,98.4 187.2,97.8 187.6,97.4 187.9,96.9 188.3,96.4 188.7,96.0 189.0,95.6 189.4,95.2 189.7,94.8 190.1,94.5 190.4,94.1 190.8,93.8 191.2,93.6 191.5,93.3 191.9,93.1 192.2,92.9 192.6,92.7 192.9,92.6 193.3,92.4 193.6,92.3 194.0,92.3 194.4,92.2 194.7,92.2 195.1,92.2 195.4,92.2 195.8,92.2 196.1,92.3 196.5,92.4 196.9,92.5 197.2,92.6 197.6,92.7 197.9,92.9 198.3,93.0 198.6,93.2 199.0,93.4 199.4,93.6 199.7,93.9 200.1,94.1 200.4,94.4 200.8,94.6 201.1,94.9 201.5,95.2 201.8,95.5 202.2,95.8 202.6,96.1 202.9,96.4 203.3,96.7 203.6,97.0 204.0,97.3 204.3,97.6 204.7,97.9 205.1,98.2 205.4,98.5 205.8,98.8 206.1,99.1 206.5,99.4 206.8,99.7 207.2,100.0 207.6,100.3 207.9,100.6 208.3,100.8 208.6,101.1 209.0,101.3 209.3,101.6 209.7,101.8 210.1,102.0 210.4,102.2 210.8,102.4 211.1,102.6 211.5,102.8 211.8,102.9 212.2,103.1 212.5,103.2 212.9,103.3 213.3,103.4 213.6,103.5 214.0,103.6 214.3,103.7 214.7,103.8 215.0,103.8 215.4,103.9 215.8,103.9 216.1,103.9 216.5,103.9 216.8,103.9 217.2,103.9 217.5,103.8 217.9,103.8 218.3,103.8 218.6,103.7 219.0,103.6 219.3,103.6 219.7,103.5 220.0,103.4 220.4,103.3 220.8,103.2 221.1,103.1 221.5,102.9 221.8,102.8 222.2,102.7 222.5,102.5 222.9,102.4 223.2,102.3 223.6,102.1 224.0,102.0 224.3,101.8 224.7,101.7 225.0,101.5 225.4,101.4 225.7,101.2 226.1,101.0 226.5,100.9 226.8,100.7 227.2,100.6 227.5,100.4 227.9,100.3 228.2,100.1 228.6,100.0 229.0,99.9 229.3,99.7 229.7,99.6 230.0,99.5 230.4,99.3 230.7,99.2 231.1,99.1 231.5,99.0 231.8,98.9 232.2,98.8 232.5,98.7 232.9,98.6 233.2,98.5 233.6,98.5 234.0,98.4 234.3,98.3 234.7,98.3 235.0,98.2 235.4,98.2 235.7,98.2 236.1,98.1 236.4,98.1 236.8,98.1 237.2,98.1 237.5,98.1 237.9,98.1 238.2,98.1 238.6,98.1 238.9,98.1 239.3,98.1 239.7,98.1 240.0,98.2 240.4,98.2 240.7,98.2 241.1,98.3 241.4,98.3 241.8,98.4 242.2,98.4 242.5,98.5 242.9,98.5 243.2,98.6 243.6,98.7 243.9,98.7 244.3,98.8 244.7,98.9 245.0,98.9 245.4,99.0 245.7,99.1 246.1,99.2 246.4,99.3 246.8,99.3 247.1,99.4 247.5,99.5 247.9,99.6 248.2,99.6 248.6,99.7 248.9,99.8 249.3,99.9 249.6,99.9 250.0,100.0" fill="none" stroke="#1f2430" stroke-width="0.8" />
        <text x="31.0" y="103" font-size="8.5" fill="#666" text-anchor="end">n = 5</text>
        <text x="250.0" y="78.0" font-size="8.5" fill="#1f2430" text-anchor="end">⟨e₅, v⟩ ≈ -0.030</text>
        <line x1="36.0" y1="150" x2="250.0" y2="150" stroke="#bbb" stroke-width="0.8" />
        <g fill="#1565c0" opacity="0.4"><path d="M 36.0,150 L 36.0,150.0 36.4,150.0 36.7,149.9 37.1,149.9 37.4,149.9 37.8,149.8 38.1,149.8 38.5,149.8 38.9,149.8 39.2,149.8 39.6,149.7 39.9,149.7 40.3,149.7 40.6,149.8 41.0,149.8 41.4,149.8 41.7,149.8 42.1,149.8 42.4,149.9 42.8,149.9 43.1,149.9 43.5,150.0 43.8,150.0 44.2,150.0 44.6,150.0 44.9,150.0 45.3,150.0 45.6,150.0 46.0,150.0 46.3,150.0 46.7,150.0 47.1,150.0 47.4,150.0 47.8,150.0 48.1,150.0 48.5,150.0 48.8,150.0 49.2,150.0 49.6,150.0 49.9,150.0 50.3,150.0 50.6,150.0 51.0,150.0 51.3,150.0 51.7,149.9 52.0,149.9 52.4,149.8 52.8,149.8 53.1,149.7 53.5,149.7 53.8,149.7 54.2,149.6 54.5,149.6 54.9,149.6 55.3,149.6 55.6,149.6 56.0,149.6 56.3,149.6 56.7,149.6 57.0,149.7 57.4,149.7 57.8,149.8 58.1,149.8 58.5,149.9 58.8,150.0 59.2,150.0 59.5,150.0 59.9,150.0 60.3,150.0 60.6,150.0 61.0,150.0 61.3,150.0 61.7,150.0 62.0,150.0 62.4,150.0 62.8,150.0 63.1,150.0 63.5,150.0 63.8,150.0 64.2,150.0 64.5,150.0 64.9,150.0 65.2,150.0 65.6,150.0 66.0,150.0 66.3,150.0 66.7,150.0 67.0,149.9 67.4,149.8 67.7,149.7 68.1,149.6 68.5,149.5 68.8,149.5 69.2,149.4 69.5,149.4 69.9,149.3 70.2,149.3 70.6,149.3 71.0,149.3 71.3,149.3 71.7,149.4 72.0,149.4 72.4,149.5 72.7,149.6 73.1,149.7 73.5,149.8 73.8,149.9 74.2,150.0 74.5,150.0 74.9,150.0 75.2,150.0 75.6,150.0 75.9,150.0 76.3,150.0 76.7,150.0 77.0,150.0 77.4,150.0 77.7,150.0 78.1,150.0 78.4,150.0 78.8,150.0 79.2,150.0 79.5,150.0 79.9,150.0 80.2,150.0 80.6,150.0 80.9,150.0 81.3,150.0 81.7,150.0 82.0,149.9 82.4,149.8 82.7,149.6 83.1,149.5 83.4,149.4 83.8,149.2 84.2,149.1 84.5,149.0 84.9,149.0 85.2,148.9 85.6,148.9 85.9,148.9 86.3,148.9 86.6,148.9 87.0,149.0 87.4,149.1 87.7,149.2 88.1,149.3 88.4,149.5 88.8,149.6 89.1,149.8 89.5,150.0 89.9,150.0 90.2,150.0 90.6,150.0 90.9,150.0 91.3,150.0 91.6,150.0 92.0,150.0 92.4,150.0 92.7,150.0 93.1,150.0 93.4,150.0 93.8,150.0 94.1,150.0 94.5,150.0 94.8,150.0 95.2,150.0 95.6,150.0 95.9,150.0 96.3,150.0 96.6,150.0 97.0,150.0 97.3,149.9 97.7,149.6 98.1,149.4 98.4,149.1 98.8,148.9 99.1,148.7 99.5,148.5 99.8,148.4 100.2,148.3 100.6,148.2 100.9,148.1 101.3,148.1 101.6,148.2 102.0,148.2 102.3,148.3 102.7,148.5 103.1,148.7 103.4,148.9 103.8,149.2 104.1,149.4 104.5,149.7 104.8,150.0 105.2,150.0 105.5,150.0 105.9,150.0 106.3,150.0 106.6,150.0 107.0,150.0 107.3,150.0 107.7,150.0 108.0,150.0 108.4,150.0 108.8,150.0 109.1,150.0 109.5,150.0 109.8,150.0 110.2,150.0 110.5,150.0 110.9,150.0 111.3,150.0 111.6,150.0 112.0,150.0 112.3,150.0 112.7,149.7 113.0,149.3 113.4,148.9 113.8,148.5 114.1,148.2 114.5,147.8 114.8,147.5 115.2,147.3 115.5,147.1 115.9,147.0 116.2,146.9 116.6,146.9 117.0,147.0 117.3,147.1 117.7,147.3 118.0,147.6 118.4,147.9 118.7,148.3 119.1,148.7 119.5,149.1 119.8,149.6 120.2,150.0 120.5,150.0 120.9,150.0 121.2,150.0 121.6,150.0 122.0,150.0 122.3,150.0 122.7,150.0 123.0,150.0 123.4,150.0 123.7,150.0 124.1,150.0 124.5,150.0 124.8,150.0 125.2,150.0 125.5,150.0 125.9,150.0 126.2,150.0 126.6,150.0 127.0,150.0 127.3,150.0 127.7,150.0 128.0,149.4 128.4,148.8 128.7,148.1 129.1,147.5 129.4,146.9 129.8,146.4 130.2,145.9 130.5,145.5 130.9,145.2 131.2,145.0 131.6,144.9 131.9,144.9 132.3,145.0 132.7,145.3 133.0,145.6 133.4,146.1 133.7,146.6 134.1,147.2 134.4,147.9 134.8,148.7 135.2,149.5 135.5,150.0 135.9,150.0 136.2,150.0 136.6,150.0 136.9,150.0 137.3,150.0 137.6,150.0 138.0,150.0 138.4,150.0 138.7,150.0 139.1,150.0 139.4,150.0 139.8,150.0 140.1,150.0 140.5,150.0 140.9,150.0 141.2,150.0 141.6,150.0 141.9,150.0 142.3,150.0 142.6,150.0 143.0,150.0 143.4,148.9 143.7,147.8 144.1,146.7 144.4,145.7 144.8,144.8 145.1,143.9 145.5,143.1 145.9,142.5 146.2,142.0 146.6,141.7 146.9,141.6 147.3,141.6 147.6,141.9 148.0,142.3 148.4,142.9 148.7,143.6 149.1,144.5 149.4,145.6 149.8,146.8 150.1,148.1 150.5,149.4 150.8,150.0 151.2,150.0 151.6,150.0 151.9,150.0 152.3,150.0 152.6,150.0 153.0,150.0 153.3,150.0 153.7,150.0 154.1,150.0 154.4,150.0 154.8,150.0 155.1,150.0 155.5,150.0 155.8,150.0 156.2,150.0 156.6,150.0 156.9,150.0 157.3,150.0 157.6,150.0 158.0,150.0 158.3,149.7 158.7,147.9 159.1,146.1 159.4,144.4 159.8,142.7 160.1,141.1 160.5,139.7 160.8,138.5 161.2,137.5 161.5,136.8 161.9,136.3 162.3,136.2 162.6,136.3 163.0,136.7 163.3,137.4 163.7,138.4 164.0,139.7 164.4,141.3 164.8,143.0 165.1,145.0 165.5,147.1 165.8,149.3 166.2,150.0 166.5,150.0 166.9,150.0 167.3,150.0 167.6,150.0 168.0,150.0 168.3,150.0 168.7,150.0 169.0,150.0 169.4,150.0 169.8,150.0 170.1,150.0 170.5,150.0 170.8,150.0 171.2,150.0 171.5,150.0 171.9,150.0 172.2,150.0 172.6,150.0 173.0,150.0 173.3,150.0 173.7,149.3 174.0,147.0 174.4,144.8 174.7,142.8 175.1,141.0 175.5,139.4 175.8,138.0 176.2,137.0 176.5,136.3 176.9,135.8 177.2,135.7 177.6,135.9 178.0,136.4 178.3,137.1 178.7,138.2 179.0,139.4 179.4,140.9 179.7,142.5 180.1,144.2 180.4,146.0 180.8,147.9 181.2,149.7 181.5,150.0 181.9,150.0 182.2,150.0 182.6,150.0 182.9,150.0 183.3,150.0 183.7,150.0 184.0,150.0 184.4,150.0 184.7,150.0 185.1,150.0 185.4,150.0 185.8,150.0 186.2,150.0 186.5,150.0 186.9,150.0 187.2,150.0 187.6,150.0 187.9,150.0 188.3,150.0 188.7,150.0 189.0,149.4 189.4,148.0 189.7,146.7 190.1,145.5 190.4,144.4 190.8,143.4 191.2,142.6 191.5,142.0 191.9,141.6 192.2,141.4 192.6,141.3 192.9,141.5 193.3,141.8 193.6,142.3 194.0,142.9 194.4,143.7 194.7,144.6 195.1,145.6 195.4,146.6 195.8,147.7 196.1,148.9 196.5,150.0 196.9,150.0 197.2,150.0 197.6,150.0 197.9,150.0 198.3,150.0 198.6,150.0 199.0,150.0 199.4,150.0 199.7,150.0 200.1,150.0 200.4,150.0 200.8,150.0 201.1,150.0 201.5,150.0 201.8,150.0 202.2,150.0 202.6,150.0 202.9,150.0 203.3,150.0 203.6,150.0 204.0,150.0 204.3,149.5 204.7,148.7 205.1,147.9 205.4,147.1 205.8,146.5 206.1,145.9 206.5,145.5 206.8,145.1 207.2,144.9 207.6,144.7 207.9,144.7 208.3,144.8 208.6,145.0 209.0,145.4 209.3,145.8 209.7,146.2 210.1,146.8 210.4,147.4 210.8,148.0 211.1,148.7 211.5,149.4 211.8,150.0 212.2,150.0 212.5,150.0 212.9,150.0 213.3,150.0 213.6,150.0 214.0,150.0 214.3,150.0 214.7,150.0 215.0,150.0 215.4,150.0 215.8,150.0 216.1,150.0 216.5,150.0 216.8,150.0 217.2,150.0 217.5,150.0 217.9,150.0 218.3,150.0 218.6,150.0 219.0,150.0 219.3,150.0 219.7,149.6 220.0,149.1 220.4,148.6 220.8,148.2 221.1,147.8 221.5,147.5 221.8,147.2 222.2,147.0 222.5,146.9 222.9,146.8 223.2,146.8 223.6,146.9 224.0,147.0 224.3,147.2 224.7,147.5 225.0,147.8 225.4,148.1 225.7,148.5 226.1,148.9 226.5,149.3 226.8,149.7 227.2,150.0 227.5,150.0 227.9,150.0 228.2,150.0 228.6,150.0 229.0,150.0 229.3,150.0 229.7,150.0 230.0,150.0 230.4,150.0 230.7,150.0 231.1,150.0 231.5,150.0 231.8,150.0 232.2,150.0 232.5,150.0 232.9,150.0 233.2,150.0 233.6,150.0 234.0,150.0 234.3,150.0 234.7,150.0 235.0,149.7 235.4,149.4 235.7,149.1 236.1,148.9 236.4,148.6 236.8,148.4 237.2,148.3 237.5,148.2 237.9,148.1 238.2,148.1 238.6,148.1 238.9,148.1 239.3,148.2 239.7,148.3 240.0,148.5 240.4,148.7 240.7,148.9 241.1,149.1 241.4,149.4 241.8,149.6 242.2,149.9 242.5,150.0 242.9,150.0 243.2,150.0 243.6,150.0 243.9,150.0 244.3,150.0 244.7,150.0 245.0,150.0 245.4,150.0 245.7,150.0 246.1,150.0 246.4,150.0 246.8,150.0 247.1,150.0 247.5,150.0 247.9,150.0 248.2,150.0 248.6,150.0 248.9,150.0 249.3,150.0 249.6,150.0 250.0,150.0 L 250.0,150 Z" /></g>
        <g fill="#c62828" opacity="0.4"><path d="M 36.0,150 L 36.0,150.0 36.4,150.0 36.7,150.0 37.1,150.0 37.4,150.0 37.8,150.0 38.1,150.0 38.5,150.0 38.9,150.0 39.2,150.0 39.6,150.0 39.9,150.0 40.3,150.0 40.6,150.0 41.0,150.0 41.4,150.0 41.7,150.0 42.1,150.0 42.4,150.0 42.8,150.0 43.1,150.0 43.5,150.0 43.8,150.0 44.2,150.1 44.6,150.1 44.9,150.2 45.3,150.2 45.6,150.2 46.0,150.3 46.3,150.3 46.7,150.3 47.1,150.3 47.4,150.3 47.8,150.3 48.1,150.3 48.5,150.3 48.8,150.3 49.2,150.3 49.6,150.2 49.9,150.2 50.3,150.1 50.6,150.1 51.0,150.0 51.3,150.0 51.7,150.0 52.0,150.0 52.4,150.0 52.8,150.0 53.1,150.0 53.5,150.0 53.8,150.0 54.2,150.0 54.5,150.0 54.9,150.0 55.3,150.0 55.6,150.0 56.0,150.0 56.3,150.0 56.7,150.0 57.0,150.0 57.4,150.0 57.8,150.0 58.1,150.0 58.5,150.0 58.8,150.0 59.2,150.1 59.5,150.1 59.9,150.2 60.3,150.3 60.6,150.3 61.0,150.4 61.3,150.4 61.7,150.5 62.0,150.5 62.4,150.5 62.8,150.5 63.1,150.5 63.5,150.5 63.8,150.5 64.2,150.5 64.5,150.4 64.9,150.4 65.2,150.3 65.6,150.2 66.0,150.1 66.3,150.1 66.7,150.0 67.0,150.0 67.4,150.0 67.7,150.0 68.1,150.0 68.5,150.0 68.8,150.0 69.2,150.0 69.5,150.0 69.9,150.0 70.2,150.0 70.6,150.0 71.0,150.0 71.3,150.0 71.7,150.0 72.0,150.0 72.4,150.0 72.7,150.0 73.1,150.0 73.5,150.0 73.8,150.0 74.2,150.0 74.5,150.1 74.9,150.2 75.2,150.3 75.6,150.4 75.9,150.5 76.3,150.6 76.7,150.7 77.0,150.8 77.4,150.8 77.7,150.9 78.1,150.9 78.4,150.9 78.8,150.9 79.2,150.8 79.5,150.8 79.9,150.7 80.2,150.6 80.6,150.5 80.9,150.4 81.3,150.2 81.7,150.1 82.0,150.0 82.4,150.0 82.7,150.0 83.1,150.0 83.4,150.0 83.8,150.0 84.2,150.0 84.5,150.0 84.9,150.0 85.2,150.0 85.6,150.0 85.9,150.0 86.3,150.0 86.6,150.0 87.0,150.0 87.4,150.0 87.7,150.0 88.1,150.0 88.4,150.0 88.8,150.0 89.1,150.0 89.5,150.0 89.9,150.2 90.2,150.4 90.6,150.6 90.9,150.7 91.3,150.9 91.6,151.1 92.0,151.2 92.4,151.3 92.7,151.4 93.1,151.4 93.4,151.5 93.8,151.5 94.1,151.4 94.5,151.3 94.8,151.2 95.2,151.1 95.6,151.0 95.9,150.8 96.3,150.6 96.6,150.3 97.0,150.1 97.3,150.0 97.7,150.0 98.1,150.0 98.4,150.0 98.8,150.0 99.1,150.0 99.5,150.0 99.8,150.0 100.2,150.0 100.6,150.0 100.9,150.0 101.3,150.0 101.6,150.0 102.0,150.0 102.3,150.0 102.7,150.0 103.1,150.0 103.4,150.0 103.8,150.0 104.1,150.0 104.5,150.0 104.8,150.0 105.2,150.4 105.5,150.7 105.9,151.0 106.3,151.3 106.6,151.5 107.0,151.8 107.3,152.0 107.7,152.2 108.0,152.3 108.4,152.4 108.8,152.4 109.1,152.4 109.5,152.3 109.8,152.2 110.2,152.0 110.5,151.8 110.9,151.5 111.3,151.2 111.6,150.9 112.0,150.5 112.3,150.1 112.7,150.0 113.0,150.0 113.4,150.0 113.8,150.0 114.1,150.0 114.5,150.0 114.8,150.0 115.2,150.0 115.5,150.0 115.9,150.0 116.2,150.0 116.6,150.0 117.0,150.0 117.3,150.0 117.7,150.0 118.0,150.0 118.4,150.0 118.7,150.0 119.1,150.0 119.5,150.0 119.8,150.0 120.2,150.1 120.5,150.7 120.9,151.2 121.2,151.7 121.6,152.2 122.0,152.6 122.3,153.0 122.7,153.3 123.0,153.6 123.4,153.8 123.7,153.9 124.1,154.0 124.5,153.9 124.8,153.8 125.2,153.6 125.5,153.3 125.9,152.9 126.2,152.4 126.6,151.9 127.0,151.4 127.3,150.7 127.7,150.1 128.0,150.0 128.4,150.0 128.7,150.0 129.1,150.0 129.4,150.0 129.8,150.0 130.2,150.0 130.5,150.0 130.9,150.0 131.2,150.0 131.6,150.0 131.9,150.0 132.3,150.0 132.7,150.0 133.0,150.0 133.4,150.0 133.7,150.0 134.1,150.0 134.4,150.0 134.8,150.0 135.2,150.0 135.5,150.4 135.9,151.2 136.2,152.1 136.6,152.9 136.9,153.7 137.3,154.4 137.6,155.0 138.0,155.6 138.4,156.0 138.7,156.3 139.1,156.5 139.4,156.6 139.8,156.5 140.1,156.2 140.5,155.8 140.9,155.3 141.2,154.7 141.6,153.9 141.9,153.0 142.3,152.1 142.6,151.1 143.0,150.0 143.4,150.0 143.7,150.0 144.1,150.0 144.4,150.0 144.8,150.0 145.1,150.0 145.5,150.0 145.9,150.0 146.2,150.0 146.6,150.0 146.9,150.0 147.3,150.0 147.6,150.0 148.0,150.0 148.4,150.0 148.7,150.0 149.1,150.0 149.4,150.0 149.8,150.0 150.1,150.0 150.5,150.0 150.8,150.8 151.2,152.2 151.6,153.6 151.9,154.9 152.3,156.2 152.6,157.4 153.0,158.4 153.3,159.3 153.7,160.0 154.1,160.5 154.4,160.7 154.8,160.8 155.1,160.6 155.5,160.2 155.8,159.5 156.2,158.6 156.6,157.5 156.9,156.2 157.3,154.8 157.6,153.2 158.0,151.5 158.3,150.0 158.7,150.0 159.1,150.0 159.4,150.0 159.8,150.0 160.1,150.0 160.5,150.0 160.8,150.0 161.2,150.0 161.5,150.0 161.9,150.0 162.3,150.0 162.6,150.0 163.0,150.0 163.3,150.0 163.7,150.0 164.0,150.0 164.4,150.0 164.8,150.0 165.1,150.0 165.5,150.0 165.8,150.0 166.2,151.6 166.5,154.0 166.9,156.3 167.3,158.5 167.6,160.5 168.0,162.4 168.3,164.1 168.7,165.5 169.0,166.6 169.4,167.3 169.8,167.7 170.1,167.7 170.5,167.1 170.8,166.0 171.2,164.5 171.5,162.8 171.9,160.9 172.2,158.7 172.6,156.5 173.0,154.1 173.3,151.7 173.7,150.0 174.0,150.0 174.4,150.0 174.7,150.0 175.1,150.0 175.5,150.0 175.8,150.0 176.2,150.0 176.5,150.0 176.9,150.0 177.2,150.0 177.6,150.0 178.0,150.0 178.3,150.0 178.7,150.0 179.0,150.0 179.4,150.0 179.7,150.0 180.1,150.0 180.4,150.0 180.8,150.0 181.2,150.0 181.5,151.6 181.9,153.3 182.2,155.0 182.6,156.4 182.9,157.8 183.3,158.9 183.7,159.8 184.0,160.5 184.4,160.9 184.7,161.1 185.1,161.1 185.4,160.8 185.8,160.3 186.2,159.6 186.5,158.7 186.9,157.6 187.2,156.4 187.6,155.1 187.9,153.7 188.3,152.3 188.7,150.8 189.0,150.0 189.4,150.0 189.7,150.0 190.1,150.0 190.4,150.0 190.8,150.0 191.2,150.0 191.5,150.0 191.9,150.0 192.2,150.0 192.6,150.0 192.9,150.0 193.3,150.0 193.6,150.0 194.0,150.0 194.4,150.0 194.7,150.0 195.1,150.0 195.4,150.0 195.8,150.0 196.1,150.0 196.5,150.0 196.9,151.1 197.2,152.2 197.6,153.1 197.9,154.0 198.3,154.8 198.6,155.5 199.0,156.0 199.4,156.4 199.7,156.7 200.1,156.8 200.4,156.7 200.8,156.5 201.1,156.2 201.5,155.8 201.8,155.2 202.2,154.5 202.6,153.8 202.9,153.0 203.3,152.1 203.6,151.3 204.0,150.4 204.3,150.0 204.7,150.0 205.1,150.0 205.4,150.0 205.8,150.0 206.1,150.0 206.5,150.0 206.8,150.0 207.2,150.0 207.6,150.0 207.9,150.0 208.3,150.0 208.6,150.0 209.0,150.0 209.3,150.0 209.7,150.0 210.1,150.0 210.4,150.0 210.8,150.0 211.1,150.0 211.5,150.0 211.8,150.1 212.2,150.8 212.5,151.4 212.9,152.0 213.3,152.5 213.6,153.0 214.0,153.4 214.3,153.7 214.7,153.9 215.0,154.1 215.4,154.1 215.8,154.1 216.1,153.9 216.5,153.7 216.8,153.5 217.2,153.1 217.5,152.7 217.9,152.2 218.3,151.7 218.6,151.2 219.0,150.7 219.3,150.2 219.7,150.0 220.0,150.0 220.4,150.0 220.8,150.0 221.1,150.0 221.5,150.0 221.8,150.0 222.2,150.0 222.5,150.0 222.9,150.0 223.2,150.0 223.6,150.0 224.0,150.0 224.3,150.0 224.7,150.0 225.0,150.0 225.4,150.0 225.7,150.0 226.1,150.0 226.5,150.0 226.8,150.0 227.2,150.1 227.5,150.5 227.9,150.9 228.2,151.3 228.6,151.6 229.0,151.9 229.3,152.1 229.7,152.3 230.0,152.4 230.4,152.5 230.7,152.5 231.1,152.5 231.5,152.4 231.8,152.2 232.2,152.1 232.5,151.8 232.9,151.6 233.2,151.3 233.6,151.0 234.0,150.7 234.3,150.4 234.7,150.0 235.0,150.0 235.4,150.0 235.7,150.0 236.1,150.0 236.4,150.0 236.8,150.0 237.2,150.0 237.5,150.0 237.9,150.0 238.2,150.0 238.6,150.0 238.9,150.0 239.3,150.0 239.7,150.0 240.0,150.0 240.4,150.0 240.7,150.0 241.1,150.0 241.4,150.0 241.8,150.0 242.2,150.0 242.5,150.1 242.9,150.4 243.2,150.6 243.6,150.8 243.9,151.0 244.3,151.1 244.7,151.3 245.0,151.4 245.4,151.5 245.7,151.5 246.1,151.5 246.4,151.5 246.8,151.4 247.1,151.4 247.5,151.2 247.9,151.1 248.2,150.9 248.6,150.8 248.9,150.6 249.3,150.4 249.6,150.2 250.0,150.0 L 250.0,150 Z" /></g>
        <polyline points="36.0,149.8 38.1,149.8 40.3,149.7 42.4,149.7 44.6,149.7 46.7,149.7 48.8,149.7 51.0,149.6 53.1,149.6 55.3,149.6 57.4,149.5 59.5,149.5 61.7,149.5 63.8,149.4 66.0,149.4 68.1,149.4 70.2,149.3 72.4,149.3 74.5,149.2 76.7,149.2 78.8,149.1 80.9,149.0 83.1,149.0 85.2,148.9 87.4,148.8 89.5,148.7 91.6,148.6 93.8,148.5 95.9,148.4 98.1,148.3 100.2,148.2 102.3,148.0 104.5,147.9 106.6,147.7 108.8,147.6 110.9,147.4 113.0,147.2 115.2,147.0 117.3,146.8 119.5,146.6 121.6,146.3 123.7,146.1 125.9,145.8 128.0,145.5 130.2,145.1 132.3,144.8 134.4,144.4 136.6,144.0 138.7,143.6 140.9,143.1 143.0,142.6 145.1,142.1 147.3,141.5 149.4,140.9 151.6,140.2 153.7,139.5 155.8,138.8 158.0,137.9 160.1,137.1 162.3,136.1 164.4,135.1 166.5,134.1 168.7,132.9 170.8,132.3 173.0,133.5 175.1,134.6 177.2,135.7 179.4,136.6 181.5,137.5 183.7,138.4 185.8,139.2 187.9,139.9 190.1,140.6 192.2,141.2 194.4,141.8 196.5,142.4 198.6,142.9 200.8,143.4 202.9,143.8 205.1,144.2 207.2,144.6 209.3,145.0 211.5,145.3 213.6,145.6 215.8,145.9 217.9,146.2 220.0,146.5 222.2,146.7 224.3,146.9 226.5,147.1 228.6,147.3 230.7,147.5 232.9,147.7 235.0,147.8 237.2,148.0 239.3,148.1 241.4,148.2 243.6,148.4 245.7,148.5 247.9,148.6 250.0,148.7" fill="none" stroke="#666" stroke-width="0.7" stroke-dasharray="2,2" />
        <polyline points="36.0,150.2 38.1,150.2 40.3,150.3 42.4,150.3 44.6,150.3 46.7,150.3 48.8,150.3 51.0,150.4 53.1,150.4 55.3,150.4 57.4,150.5 59.5,150.5 61.7,150.5 63.8,150.6 66.0,150.6 68.1,150.6 70.2,150.7 72.4,150.7 74.5,150.8 76.7,150.8 78.8,150.9 80.9,151.0 83.1,151.0 85.2,151.1 87.4,151.2 89.5,151.3 91.6,151.4 93.8,151.5 95.9,151.6 98.1,151.7 100.2,151.8 102.3,152.0 104.5,152.1 106.6,152.3 108.8,152.4 110.9,152.6 113.0,152.8 115.2,153.0 117.3,153.2 119.5,153.4 121.6,153.7 123.7,153.9 125.9,154.2 128.0,154.5 130.2,154.9 132.3,155.2 134.4,155.6 136.6,156.0 138.7,156.4 140.9,156.9 143.0,157.4 145.1,157.9 147.3,158.5 149.4,159.1 151.6,159.8 153.7,160.5 155.8,161.2 158.0,162.1 160.1,162.9 162.3,163.9 164.4,164.9 166.5,165.9 168.7,167.1 170.8,167.7 173.0,166.5 175.1,165.4 177.2,164.3 179.4,163.4 181.5,162.5 183.7,161.6 185.8,160.8 187.9,160.1 190.1,159.4 192.2,158.8 194.4,158.2 196.5,157.6 198.6,157.1 200.8,156.6 202.9,156.2 205.1,155.8 207.2,155.4 209.3,155.0 211.5,154.7 213.6,154.4 215.8,154.1 217.9,153.8 220.0,153.5 222.2,153.3 224.3,153.1 226.5,152.9 228.6,152.7 230.7,152.5 232.9,152.3 235.0,152.2 237.2,152.0 239.3,151.9 241.4,151.8 243.6,151.6 245.7,151.5 247.9,151.4 250.0,151.3" fill="none" stroke="#666" stroke-width="0.7" stroke-dasharray="2,2" />
        <polyline points="36.0,150.0 36.4,150.0 36.7,149.9 37.1,149.9 37.4,149.9 37.8,149.8 38.1,149.8 38.5,149.8 38.9,149.8 39.2,149.8 39.6,149.7 39.9,149.7 40.3,149.7 40.6,149.8 41.0,149.8 41.4,149.8 41.7,149.8 42.1,149.8 42.4,149.9 42.8,149.9 43.1,149.9 43.5,150.0 43.8,150.0 44.2,150.1 44.6,150.1 44.9,150.2 45.3,150.2 45.6,150.2 46.0,150.3 46.3,150.3 46.7,150.3 47.1,150.3 47.4,150.3 47.8,150.3 48.1,150.3 48.5,150.3 48.8,150.3 49.2,150.3 49.6,150.2 49.9,150.2 50.3,150.1 50.6,150.1 51.0,150.0 51.3,150.0 51.7,149.9 52.0,149.9 52.4,149.8 52.8,149.8 53.1,149.7 53.5,149.7 53.8,149.7 54.2,149.6 54.5,149.6 54.9,149.6 55.3,149.6 55.6,149.6 56.0,149.6 56.3,149.6 56.7,149.6 57.0,149.7 57.4,149.7 57.8,149.8 58.1,149.8 58.5,149.9 58.8,150.0 59.2,150.1 59.5,150.1 59.9,150.2 60.3,150.3 60.6,150.3 61.0,150.4 61.3,150.4 61.7,150.5 62.0,150.5 62.4,150.5 62.8,150.5 63.1,150.5 63.5,150.5 63.8,150.5 64.2,150.5 64.5,150.4 64.9,150.4 65.2,150.3 65.6,150.2 66.0,150.1 66.3,150.1 66.7,150.0 67.0,149.9 67.4,149.8 67.7,149.7 68.1,149.6 68.5,149.5 68.8,149.5 69.2,149.4 69.5,149.4 69.9,149.3 70.2,149.3 70.6,149.3 71.0,149.3 71.3,149.3 71.7,149.4 72.0,149.4 72.4,149.5 72.7,149.6 73.1,149.7 73.5,149.8 73.8,149.9 74.2,150.0 74.5,150.1 74.9,150.2 75.2,150.3 75.6,150.4 75.9,150.5 76.3,150.6 76.7,150.7 77.0,150.8 77.4,150.8 77.7,150.9 78.1,150.9 78.4,150.9 78.8,150.9 79.2,150.8 79.5,150.8 79.9,150.7 80.2,150.6 80.6,150.5 80.9,150.4 81.3,150.2 81.7,150.1 82.0,149.9 82.4,149.8 82.7,149.6 83.1,149.5 83.4,149.4 83.8,149.2 84.2,149.1 84.5,149.0 84.9,149.0 85.2,148.9 85.6,148.9 85.9,148.9 86.3,148.9 86.6,148.9 87.0,149.0 87.4,149.1 87.7,149.2 88.1,149.3 88.4,149.5 88.8,149.6 89.1,149.8 89.5,150.0 89.9,150.2 90.2,150.4 90.6,150.6 90.9,150.7 91.3,150.9 91.6,151.1 92.0,151.2 92.4,151.3 92.7,151.4 93.1,151.4 93.4,151.5 93.8,151.5 94.1,151.4 94.5,151.3 94.8,151.2 95.2,151.1 95.6,151.0 95.9,150.8 96.3,150.6 96.6,150.3 97.0,150.1 97.3,149.9 97.7,149.6 98.1,149.4 98.4,149.1 98.8,148.9 99.1,148.7 99.5,148.5 99.8,148.4 100.2,148.3 100.6,148.2 100.9,148.1 101.3,148.1 101.6,148.2 102.0,148.2 102.3,148.3 102.7,148.5 103.1,148.7 103.4,148.9 103.8,149.2 104.1,149.4 104.5,149.7 104.8,150.0 105.2,150.4 105.5,150.7 105.9,151.0 106.3,151.3 106.6,151.5 107.0,151.8 107.3,152.0 107.7,152.2 108.0,152.3 108.4,152.4 108.8,152.4 109.1,152.4 109.5,152.3 109.8,152.2 110.2,152.0 110.5,151.8 110.9,151.5 111.3,151.2 111.6,150.9 112.0,150.5 112.3,150.1 112.7,149.7 113.0,149.3 113.4,148.9 113.8,148.5 114.1,148.2 114.5,147.8 114.8,147.5 115.2,147.3 115.5,147.1 115.9,147.0 116.2,146.9 116.6,146.9 117.0,147.0 117.3,147.1 117.7,147.3 118.0,147.6 118.4,147.9 118.7,148.3 119.1,148.7 119.5,149.1 119.8,149.6 120.2,150.1 120.5,150.7 120.9,151.2 121.2,151.7 121.6,152.2 122.0,152.6 122.3,153.0 122.7,153.3 123.0,153.6 123.4,153.8 123.7,153.9 124.1,154.0 124.5,153.9 124.8,153.8 125.2,153.6 125.5,153.3 125.9,152.9 126.2,152.4 126.6,151.9 127.0,151.4 127.3,150.7 127.7,150.1 128.0,149.4 128.4,148.8 128.7,148.1 129.1,147.5 129.4,146.9 129.8,146.4 130.2,145.9 130.5,145.5 130.9,145.2 131.2,145.0 131.6,144.9 131.9,144.9 132.3,145.0 132.7,145.3 133.0,145.6 133.4,146.1 133.7,146.6 134.1,147.2 134.4,147.9 134.8,148.7 135.2,149.5 135.5,150.4 135.9,151.2 136.2,152.1 136.6,152.9 136.9,153.7 137.3,154.4 137.6,155.0 138.0,155.6 138.4,156.0 138.7,156.3 139.1,156.5 139.4,156.6 139.8,156.5 140.1,156.2 140.5,155.8 140.9,155.3 141.2,154.7 141.6,153.9 141.9,153.0 142.3,152.1 142.6,151.1 143.0,150.0 143.4,148.9 143.7,147.8 144.1,146.7 144.4,145.7 144.8,144.8 145.1,143.9 145.5,143.1 145.9,142.5 146.2,142.0 146.6,141.7 146.9,141.6 147.3,141.6 147.6,141.9 148.0,142.3 148.4,142.9 148.7,143.6 149.1,144.5 149.4,145.6 149.8,146.8 150.1,148.1 150.5,149.4 150.8,150.8 151.2,152.2 151.6,153.6 151.9,154.9 152.3,156.2 152.6,157.4 153.0,158.4 153.3,159.3 153.7,160.0 154.1,160.5 154.4,160.7 154.8,160.8 155.1,160.6 155.5,160.2 155.8,159.5 156.2,158.6 156.6,157.5 156.9,156.2 157.3,154.8 157.6,153.2 158.0,151.5 158.3,149.7 158.7,147.9 159.1,146.1 159.4,144.4 159.8,142.7 160.1,141.1 160.5,139.7 160.8,138.5 161.2,137.5 161.5,136.8 161.9,136.3 162.3,136.2 162.6,136.3 163.0,136.7 163.3,137.4 163.7,138.4 164.0,139.7 164.4,141.3 164.8,143.0 165.1,145.0 165.5,147.1 165.8,149.3 166.2,151.6 166.5,154.0 166.9,156.3 167.3,158.5 167.6,160.5 168.0,162.4 168.3,164.1 168.7,165.5 169.0,166.6 169.4,167.3 169.8,167.7 170.1,167.7 170.5,167.1 170.8,166.0 171.2,164.5 171.5,162.8 171.9,160.9 172.2,158.7 172.6,156.5 173.0,154.1 173.3,151.7 173.7,149.3 174.0,147.0 174.4,144.8 174.7,142.8 175.1,141.0 175.5,139.4 175.8,138.0 176.2,137.0 176.5,136.3 176.9,135.8 177.2,135.7 177.6,135.9 178.0,136.4 178.3,137.1 178.7,138.2 179.0,139.4 179.4,140.9 179.7,142.5 180.1,144.2 180.4,146.0 180.8,147.9 181.2,149.7 181.5,151.6 181.9,153.3 182.2,155.0 182.6,156.4 182.9,157.8 183.3,158.9 183.7,159.8 184.0,160.5 184.4,160.9 184.7,161.1 185.1,161.1 185.4,160.8 185.8,160.3 186.2,159.6 186.5,158.7 186.9,157.6 187.2,156.4 187.6,155.1 187.9,153.7 188.3,152.3 188.7,150.8 189.0,149.4 189.4,148.0 189.7,146.7 190.1,145.5 190.4,144.4 190.8,143.4 191.2,142.6 191.5,142.0 191.9,141.6 192.2,141.4 192.6,141.3 192.9,141.5 193.3,141.8 193.6,142.3 194.0,142.9 194.4,143.7 194.7,144.6 195.1,145.6 195.4,146.6 195.8,147.7 196.1,148.9 196.5,150.0 196.9,151.1 197.2,152.2 197.6,153.1 197.9,154.0 198.3,154.8 198.6,155.5 199.0,156.0 199.4,156.4 199.7,156.7 200.1,156.8 200.4,156.7 200.8,156.5 201.1,156.2 201.5,155.8 201.8,155.2 202.2,154.5 202.6,153.8 202.9,153.0 203.3,152.1 203.6,151.3 204.0,150.4 204.3,149.5 204.7,148.7 205.1,147.9 205.4,147.1 205.8,146.5 206.1,145.9 206.5,145.5 206.8,145.1 207.2,144.9 207.6,144.7 207.9,144.7 208.3,144.8 208.6,145.0 209.0,145.4 209.3,145.8 209.7,146.2 210.1,146.8 210.4,147.4 210.8,148.0 211.1,148.7 211.5,149.4 211.8,150.1 212.2,150.8 212.5,151.4 212.9,152.0 213.3,152.5 213.6,153.0 214.0,153.4 214.3,153.7 214.7,153.9 215.0,154.1 215.4,154.1 215.8,154.1 216.1,153.9 216.5,153.7 216.8,153.5 217.2,153.1 217.5,152.7 217.9,152.2 218.3,151.7 218.6,151.2 219.0,150.7 219.3,150.2 219.7,149.6 220.0,149.1 220.4,148.6 220.8,148.2 221.1,147.8 221.5,147.5 221.8,147.2 222.2,147.0 222.5,146.9 222.9,146.8 223.2,146.8 223.6,146.9 224.0,147.0 224.3,147.2 224.7,147.5 225.0,147.8 225.4,148.1 225.7,148.5 226.1,148.9 226.5,149.3 226.8,149.7 227.2,150.1 227.5,150.5 227.9,150.9 228.2,151.3 228.6,151.6 229.0,151.9 229.3,152.1 229.7,152.3 230.0,152.4 230.4,152.5 230.7,152.5 231.1,152.5 231.5,152.4 231.8,152.2 232.2,152.1 232.5,151.8 232.9,151.6 233.2,151.3 233.6,151.0 234.0,150.7 234.3,150.4 234.7,150.0 235.0,149.7 235.4,149.4 235.7,149.1 236.1,148.9 236.4,148.6 236.8,148.4 237.2,148.3 237.5,148.2 237.9,148.1 238.2,148.1 238.6,148.1 238.9,148.1 239.3,148.2 239.7,148.3 240.0,148.5 240.4,148.7 240.7,148.9 241.1,149.1 241.4,149.4 241.8,149.6 242.2,149.9 242.5,150.1 242.9,150.4 243.2,150.6 243.6,150.8 243.9,151.0 244.3,151.1 244.7,151.3 245.0,151.4 245.4,151.5 245.7,151.5 246.1,151.5 246.4,151.5 246.8,151.4 247.1,151.4 247.5,151.2 247.9,151.1 248.2,150.9 248.6,150.8 248.9,150.6 249.3,150.4 249.6,150.2 250.0,150.0" fill="none" stroke="#1f2430" stroke-width="0.8" />
        <text x="31.0" y="153" font-size="8.5" fill="#666" text-anchor="end">n = 14</text>
        <text x="250.0" y="128.0" font-size="8.5" fill="#1f2430" text-anchor="end">⟨e₁₄, v⟩ ≈ -0.009</text>
        <text x="120" y="38" font-size="8.5" fill="#666" text-anchor="end">±v (envelope)</text>
        <g font-size="8.5" fill="#666" text-anchor="middle">
          <text x="36.0" y="182">−π</text><text x="143" y="182">0</text><text x="250.0" y="182">π</text>
        </g>
        <g font-size="9" fill="#444">
          <text x="12" y="200">finer and finer cancellation: ⟨eₙ, v⟩ = ∫ v(x) sin(nx) dx / √π → 0</text>
          <text x="12" y="213">‖eₙ‖ = 1 always: the mass has moved to high frequency</text>
        </g>
      </svg>
    </div>
  </div>
  <figcaption>Weak convergence $e_n \rightharpoonup 0$ cannot be drawn as vectors in a ball: every $e_n$ has norm $1$ and $\lVert e_n - e_m \rVert = \sqrt{2}$, yet the limit is $0$, which no sequence on a finite-dimensional sphere can do (there, weak and strong convergence coincide). What can be drawn is what a fixed test vector $v$ sees. <strong>Left.</strong> In $L^2(\mathbb{R})$ the indicators $e_n = \chi_{[n, n+1)}$ are orthonormal. The pairing $\langle e_n, v \rangle = \int_n^{n+1} v$ is the shaded area, and it shrinks to $0$ as the box slides out to where $v$ has no mass left, while the box itself keeps area $1$ and never overlaps another box. The budget bar stacks the squares $\lvert \langle e_n, v \rangle \rvert^2$: Bessel's inequality caps their sum by $\lVert v \rVert^2$, which is why the terms must tend to $0$. <strong>Right.</strong> In $L^2([-\pi, \pi])$ the functions $e_n = \sin(nx)/\sqrt{\pi}$ are orthonormal, and $\langle e_n, v \rangle$ is the signed area of $v(x)\sin(nx)$: the positive and negative lobes cancel ever more finely under the fixed envelope $\pm v$, which is the Riemann–Lebesgue lemma. In both pictures the norm does not survive the limit: $\lVert e_n \rVert = 1$ for all $n$ but $\lVert 0 \rVert = 0$. The mass is not lost, it has escaped, to infinity on the left and to high frequency on the right, beyond the reach of any fixed $v$.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Weakly Cauchy Sequences Are Bounded and Weakly Convergent)</span></p>

Let $\lbrace u\_n \rbrace$ be a sequence in a Hilbert space $H$ such that $\langle u\_n, v \rangle$ converges in $\mathbb{C}$ (equivalently, is Cauchy) for every $v \in H$. Then

1. $\sup\_n \lVert u\_n \rVert < \infty$, and
2. there is $u \in H$ with $u\_n \rightharpoonup u$.

In particular, every weakly convergent sequence is bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>hint</summary>

1. The weakly convergent sequence is bounded, since induces a family of uniformly bounded linear operators $(T_n=\langle \cdot, u_n\rangle)$ with the same norms $\lVert u_n \rVert$, where boundednes comes from the convergence for any input and further application of Uniform Boundedness Principle.
2. The sequence $(T_n=\langle \cdot, u_n\rangle)$ of linear bounded operators converge to a linear bounded operator $T\in H^\ast$. By the Riesz Representation Theorem there is $u \in H$ with $T(v) = \langle v, u \rangle$ for all $v$.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Consider the bounded linear functionals $T\_n(v) = \langle v, u\_n \rangle$ (the order is reversed to make them linear rather than conjugate-linear). By Cauchy–Schwarz and choosing a unit test vector $v = \frac{u}{\lVert u\_n \rVert}$ for non-zero $u_n$, we have $\lVert T\_n \rVert = \lVert u\_n \rVert$.

**(1)** For each fixed $v$, the sequence $T\_n(v)$ converges, hence is bounded. By the Uniform Boundedness Principle, $\lVert u\_n \rVert = \lVert T\_n \rVert \le C$ for all $n$.

**(2)** Define $T(v) = \lim\_{n \to \infty} \langle v, u\_n \rangle$. This is linear, and $\lvert T(v) \rvert \le C\lVert v \rVert$, so $T \in H^{\prime}$. Linearity follows from the linearity of each $T_n$ and preservation of linear combinations under limits. Boundedness follows from the uniform operator-norm bound $\lVert T_n\rVert\le C$, established by the Uniform Boundedness Principle, and continuity of the absolute value. By the Riesz Representation Theorem there is $u \in H$ with $T(v) = \langle v, u \rangle$ for all $v$. Conjugating, $\langle u\_n, v \rangle \to \langle u, v \rangle$ for all $v$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Coefficient Convergence Is Not Enough)</span></p>

* Let $\lbrace e\_j \rbrace$ be an orthonormal basis and $u\_n = n e\_n$. 
* For each fixed $j$, $\langle u\_n, e\_j \rangle = 0$ as soon as $n > j$, 
  * so every coefficient converges (to $0$). 
* But $\lVert u\_n \rVert = n$ is unbounded,
  * so by the theorem above $u\_n$ is **not** weakly convergent.
  
Concretely, for $v = \sum\_j j^{-1} e\_j$ we get $\langle u\_n, v \rangle = 1$, while $\langle 0, v \rangle = 0$; and for $v = \sum\_j j^{-3/4} e\_j \in H$, $\langle u\_n, v \rangle = n^{1/4} \to \infty$.

</div>

<figure class="math-figure">
  <div class="math-figure-row">
    <div>
      <div class="panel-label">uₙ = n eₙ: every column → 0, yet ⟨uₙ, v⟩ = 1</div>
      <svg viewBox="0 0 300 220" xmlns="http://www.w3.org/2000/svg" width="380" aria-label="Table of the coefficients of u n against e k: rows are the vectors u n equal n times e n, columns the basis coordinates; the only nonzero entry of each row is the diagonal entry n drawn in a growing box, so every column shaded in cream ends in zeros, the norm readout column grows without bound, and pairing each row with the green test weights one over k gives the constant one in the red readout column">
        <rect x="95" y="30" width="22" height="170" fill="#fff7e0" />
        <rect x="47" y="119" width="241" height="22" fill="#edf4ff" />
        <path d="M 51,50 H 47 V 200 H 51" fill="none" stroke="#888" stroke-width="1.1" />
        <path d="M 207,50 H 211 V 200 H 207" fill="none" stroke="#888" stroke-width="1.1" />
        <text x="40" y="40" font-size="8" fill="#888" text-anchor="end">⟨uₙ, eₖ⟩</text>
        <g font-size="9" fill="#666" text-anchor="middle"><text x="58" y="40">e₁</text><text x="82" y="40">e₂</text><text x="106" y="40">e₃</text><text x="130" y="40">e₄</text><text x="154" y="40">e₅</text><text x="178" y="40">e₆</text><text x="200" y="40">⋯</text></g>
        <text x="236" y="40" font-size="9" fill="#1565c0" text-anchor="middle" font-weight="600">‖uₙ‖</text>
        <text x="272" y="40" font-size="9" fill="#b91c1c" text-anchor="middle" font-weight="600">⟨uₙ, v⟩</text>
        <text x="40" y="22" font-size="9" fill="#0f9b6c" text-anchor="end" font-weight="600">v</text>
        <g font-size="9" fill="#0f9b6c" text-anchor="middle"><text x="58" y="22">1</text><text x="82" y="22">½</text><text x="106" y="22">⅓</text><text x="130" y="22">¼</text><text x="154" y="22">⅕</text><text x="178" y="22">⅙</text><text x="200" y="22">⋯</text></g>
        <text x="272" y="22" font-size="8" fill="#0f9b6c" text-anchor="middle">vₖ = 1/k</text>
        <text x="40" y="67" font-size="9" fill="#444" text-anchor="end">u₁</text>
        <g font-size="9" fill="#444" opacity="0.38" text-anchor="middle"><text x="82" y="67">0</text><text x="106" y="67">0</text><text x="130" y="67">0</text><text x="154" y="67">0</text><text x="178" y="67">0</text><text x="200" y="67">⋯</text></g>
        <rect x="52.7" y="58.7" width="10.6" height="10.6" fill="#edf4ff" stroke="#1565c0" stroke-width="0.9" />
        <text x="58" y="67.2" font-size="9" fill="#1565c0" text-anchor="middle" font-weight="700">1</text>
        <text x="236" y="67" font-size="9" fill="#1565c0" text-anchor="middle">1</text>
        <text x="272" y="67" font-size="9" fill="#b91c1c" text-anchor="middle" font-weight="700">1</text>
        <text x="40" y="89" font-size="9" fill="#444" text-anchor="end">u₂</text>
        <g font-size="9" fill="#444" opacity="0.38" text-anchor="middle"><text x="58" y="89">0</text><text x="106" y="89">0</text><text x="130" y="89">0</text><text x="154" y="89">0</text><text x="178" y="89">0</text><text x="200" y="89">⋯</text></g>
        <rect x="75.9" y="79.9" width="12.2" height="12.2" fill="#edf4ff" stroke="#1565c0" stroke-width="0.9" />
        <text x="82" y="89.2" font-size="9" fill="#1565c0" text-anchor="middle" font-weight="700">2</text>
        <text x="236" y="89" font-size="9" fill="#1565c0" text-anchor="middle">2</text>
        <text x="272" y="89" font-size="9" fill="#b91c1c" text-anchor="middle" font-weight="700">1</text>
        <text x="40" y="111" font-size="9" fill="#444" text-anchor="end">u₃</text>
        <g font-size="9" fill="#444" opacity="0.38" text-anchor="middle"><text x="58" y="111">0</text><text x="82" y="111">0</text><text x="130" y="111">0</text><text x="154" y="111">0</text><text x="178" y="111">0</text><text x="200" y="111">⋯</text></g>
        <rect x="99.1" y="101.1" width="13.8" height="13.8" fill="#edf4ff" stroke="#1565c0" stroke-width="0.9" />
        <text x="106" y="111.2" font-size="9" fill="#1565c0" text-anchor="middle" font-weight="700">3</text>
        <text x="236" y="111" font-size="9" fill="#1565c0" text-anchor="middle">3</text>
        <text x="272" y="111" font-size="9" fill="#b91c1c" text-anchor="middle" font-weight="700">1</text>
        <text x="40" y="133" font-size="9" fill="#444" text-anchor="end">u₄</text>
        <g font-size="9" fill="#444" opacity="0.38" text-anchor="middle"><text x="58" y="133">0</text><text x="82" y="133">0</text><text x="106" y="133">0</text><text x="154" y="133">0</text><text x="178" y="133">0</text><text x="200" y="133">⋯</text></g>
        <rect x="122.3" y="122.3" width="15.4" height="15.4" fill="#edf4ff" stroke="#1565c0" stroke-width="0.9" />
        <text x="130" y="133.2" font-size="9" fill="#1565c0" text-anchor="middle" font-weight="700">4</text>
        <text x="236" y="133" font-size="9" fill="#1565c0" text-anchor="middle">4</text>
        <text x="272" y="133" font-size="9" fill="#b91c1c" text-anchor="middle" font-weight="700">1</text>
        <text x="40" y="155" font-size="9" fill="#444" text-anchor="end">u₅</text>
        <g font-size="9" fill="#444" opacity="0.38" text-anchor="middle"><text x="58" y="155">0</text><text x="82" y="155">0</text><text x="106" y="155">0</text><text x="130" y="155">0</text><text x="178" y="155">0</text><text x="200" y="155">⋯</text></g>
        <rect x="145.5" y="143.5" width="17.0" height="17.0" fill="#edf4ff" stroke="#1565c0" stroke-width="0.9" />
        <text x="154" y="155.2" font-size="9" fill="#1565c0" text-anchor="middle" font-weight="700">5</text>
        <text x="236" y="155" font-size="9" fill="#1565c0" text-anchor="middle">5</text>
        <text x="272" y="155" font-size="9" fill="#b91c1c" text-anchor="middle" font-weight="700">1</text>
        <text x="40" y="177" font-size="9" fill="#444" text-anchor="end">u₆</text>
        <g font-size="9" fill="#444" opacity="0.38" text-anchor="middle"><text x="58" y="177">0</text><text x="82" y="177">0</text><text x="106" y="177">0</text><text x="130" y="177">0</text><text x="154" y="177">0</text><text x="200" y="177">⋯</text></g>
        <rect x="168.7" y="164.7" width="18.6" height="18.6" fill="#edf4ff" stroke="#1565c0" stroke-width="0.9" />
        <text x="178" y="177.2" font-size="9" fill="#1565c0" text-anchor="middle" font-weight="700">6</text>
        <text x="236" y="177" font-size="9" fill="#1565c0" text-anchor="middle">6</text>
        <text x="272" y="177" font-size="9" fill="#b91c1c" text-anchor="middle" font-weight="700">1</text>
        <g font-size="10" fill="#888" text-anchor="middle"><text x="37" y="195">⋮</text><text x="200" y="195">⋱</text><text x="236" y="195">⋮</text><text x="272" y="195">⋮</text></g>
        <text x="106" y="214" font-size="8.5" fill="#a86f00" text-anchor="middle">each column → 0</text>
        <text x="236" y="214" font-size="8.5" fill="#1565c0" text-anchor="middle">→ ∞</text>
        <text x="272" y="214" font-size="8.5" fill="#b91c1c" text-anchor="middle">n·(1/n) = 1</text>
      </svg>
    </div>
  </div>
  <figcaption>The coefficient table of $u_n = n e_n$: row $n$ is the vector $u_n$, column $k$ the coordinate $\langle u_n, e_k \rangle$, and the only non-zero entry of a row is the diagonal one. <strong>Down a column</strong> (shaded) one reads the coefficient sequence $\langle u_n, e_k \rangle$ for fixed $k$: a single entry $k$ and then zeros for ever, so every coefficient converges to $0$. This is all that coordinatewise convergence sees. <strong>Along the diagonal</strong> one reads $\lVert u_n \rVert = n$: the boxes grow without bound, so by the theorem above there can be no weak limit. <strong>Across a row</strong> one reads the pairing with a test vector: with the weights $v = \sum_k e_k / k$ (green; in $H$ since $\sum_k k^{-2} < \infty$) the row $n$ meets the weight $1/n$ exactly where it carries its spike, and $\langle u_n, v \rangle = n \cdot \tfrac{1}{n} = 1$ for every $n$. The weights decay as slowly as $\ell^2$ allows, which is slow enough to catch a spike growing like $n$; with $v = \sum_k k^{-3/4} e_k$ one even gets $\langle u_n, v \rangle = n^{1/4} \to \infty$. Compare the orthonormal sequence $e_n$ from the example above: its table has every diagonal entry equal to $1$, the same $v$ gives $\langle e_n, v \rangle = 1/n \to 0$, and indeed $e_n \rightharpoonup 0$. Weak convergence must survive every $v \in H$, not only the coordinates one checks one at a time, and boundedness is exactly what rules out this escape: that is the content of the proposition that follows.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Can we take any test vector $v$?)</span></p>

**The restriction on the admissible test vectors is what makes weak convergence possible.**

* If we allowed arbitrary formal infinite linear combinations as test vectors, the orthonormal sequence might be distinguishable from zero using an inadmissible $v$. But the Hilbert space structure restricts us to square-summable coefficients.

* For example, consider a sequence of orthonormal vectors $(e_n)$ that converges weakly to $0$ by the results before, and take a fixed test vector $v = \sum_i^\infty n\ce_n$. Then $\langle e_n, v\rangle$ does not weak converge to $0$. So, test vectors must have finite norm since the contradicting $v$ blows up in norm. However, in the definition of weak convergence we only require $v \in H$, which implies that Hilbert space contains only finite norm vectors.

* There is also a subtle connection to the dual space: weak convergence tests against **continuous** linear functionals (linear operator is continuous $\iif$ linear operator is bounded). The proposed $v$ would formally define

  $$T(x)=\langle x,v\rangle=\sum_{n=1}^{\infty}n\langle x,e_n\rangle,$$

  wherever this expression makes sense, but it does not define a bounded linear functional on all of $H$ since $\lVert T \rVert = \lVert v \rVert = \infty$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Weak Convergence = Bounded + Coefficientwise Convergence)</span></p>

Let $H$ be a separable Hilbert space with orthonormal basis $\lbrace e\_k \rbrace$. A sequence $\lbrace u\_n \rbrace$ converges weakly if and only if it is bounded and $\langle u\_n, e\_k \rangle$ converges for each $k$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

$(\Rightarrow)$ Boundedness is the theorem above, and coefficient convergence is the definition with $v = e\_k$.

$(\Leftarrow)$ Let $\lVert u\_n \rVert \le C$ and $\langle u\_n, e\_k \rangle \to c\_k$. For each $p$, by Bessel,

$$\sum_{k \le p} \lvert c_k \rvert^2 = \lim_{n \to \infty} \sum_{k \le p} \lvert \langle u_n, e_k \rangle \rvert^2 \le C^2,$$

so $\lbrace c\_k \rbrace \in \ell^2$ and $u := \sum\_k c\_k e\_k \in H$ is well defined, with $\langle u\_n, e\_k \rangle \to \langle u, e\_k \rangle$ for each $k$. By linearity, $\langle u\_n, w \rangle \to \langle u, w \rangle$ for every finite linear combination $w$ of the $e\_k$. For general $v$, let $v\_p = \sum\_{k \le p} \langle v, e\_k \rangle e\_k$; then

$$\lvert \langle u_n, v \rangle - \langle u, v \rangle \rvert \le \lvert \langle u_n - u, v_p \rangle \rvert + \lvert \langle u_n - u, v - v_p \rangle \rvert \le \lvert \langle u_n - u, v_p \rangle \rvert + 2C \lVert v - v_p \rVert.$$

Choose $p$ to make the last term small (independently of $n$), then $n$ large to make the first term small. $\square$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>My proof</summary>


$(\Rightarrow)$ If $\lbrace u_n\rbrace$ converges weakly, then it converges not only coordinstewise, but for every $v\in H$. The boundedness follows from the theorem before the proposition.

$(\Leftarrow)$ The idea is to extract a convergent subsequence by boundedness, then to show that the limit holds for the whole sequence by coordinatewise convergence. By boundedness we extract a subsequence $u_{n_l}$ that converges pointwise to some value $v^k = lim_{l\to\infty} \langle u_{n_l}, e_k\rangle$. The we construct a vector $v$ with nth coordinate being $v^k$ and check that it belongs to a Hilbert space, i.e. has finite norm induced by an inner product. We write,

$$\sum_{k=1}^\infty \lvert \langle v, ek \rangle \rvert^2 = \sum_{k=1}^\infty \lvert \lim_{l\to\infty} \langle unl, ek \rangle \leq \lim_{l\to\infty} \sum_{k=1}^\infty \lvert \langle unl, ek \rangle \rvert^2 = \lim_{l\to\infty} \lVert unl \rVert^2 \leq C^2, $$

where $C$ is a uniform bound from assumption and we exchange the limit and sum since $\lvert \langle u_{n_l}, e_k \rangle \rvert^2 \geq 0$ and we apply Fatou's Lemma (or it could be shown by truncation of the sum and MCT) Thus $\lVert v \rVert \leq C and v \in H$. Since the whole sequence converges pointwise, we have $\langle u_n, e_k\rangle \to \langle v, e_k\rangle$.

Now we want to show that it holds for any $f=\sum_{k=1}^\infty \langle f, e_k \rangle e_k \in H$. However, we cannot exchange the limit in the sum since summands might have negative signs and sum is not monotone:

$$\lim_{n\to\infty} \sum_{k=1}^\infty \langle f, ek \rangle \langle un, ek \rangle$$

We truncate the sum up to summand $N$, show it converges there with any error, then $N\to\infty$:

$$\lim_{n\to\infty} \sum_{k=1}^N \langle f, ek \rangle \langle un, ek \rangle = \sum_{k=1}^N \lim_{n\to\infty} \langle f, ek \rangle \langle un, ek \rangle = \sum_{k=1}^N \langle f, ek \rangle \langle v, ek \rangle.$$

The crucial missing idea is to estimate the tail using Cauchy–Schwarz. Let $P_Nf$ be the prefix sum up to the nth summand. Then we can decompose the function $f$ into the prefix and tail

$$f=P_Nf+(f-P_Nf).$$

Then

$$\begin{aligned} |\langle u_n-v,f\rangle| &\le |\langle u_n-v,P_Nf\rangle|\\ &\quad+ |\langle u_n-v,f-P_Nf\rangle|\\ &\le |\langle u_n-v,P_Nf\rangle| +2C\|f-P_Nf\|. \end{aligned}$$

Now the two terms are controlled independently:
- For fixed $N$, the first term tends to zero as $n\to\infty$, by coordinatewise convergence.
- The second term tends to zero as $N\to\infty$, uniformly in $n$, because $P_Nf\to f$ in norm.
Therefore,

$$\limsup_{n\to\infty} |\langle u_n-v,f\rangle| \le 2C\|f-P_Nf\|.$$

Letting $N\to\infty$ gives

$$\langle u_n,f\rangle\to\langle v,f\rangle \qquad\forall f\in H.$$

Thus $u_n\rightharpoonup v$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Weak Sequential Compactness of Bounded Sets)</span></p>

In a separable Hilbert space, every bounded sequence has a weakly convergent subsequence.

$$\sup_n\|u_n\|<\infty \implies \exists (u_{n_k}),u\in H: u_{n_k}\rightharpoonup u.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Let $\lVert u\_n \rVert \le C$ and fix an orthonormal basis $\lbrace e\_k \rbrace$. Each coefficient sequence satisfies $\lvert \langle u\_n, e\_k \rangle \rvert \le C$. As in the proof of Theorem 197, use Bolzano–Weierstrass to pass to a subsequence along which $\langle u\_n, e\_1 \rangle$ converges, then to a further subsequence along which $\langle u\_n, e\_2 \rangle$ converges, and so on; the **diagonal** subsequence is eventually a subsequence of each of these, so all of its coefficients converge. It is bounded, so by the previous proposition it converges weakly. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">("Heine–Borel" for the Weak Topology)</span></p>

One can read this as: *the closed unit ball of a separable Hilbert space is weakly (sequentially) compact*, even though it is never norm-compact in infinite dimensions (Example 193). This is the Hilbert-space case of the Banach–Alaoglu theorem. It is the basic tool for producing solutions of variational problems: take a minimizing sequence, extract a weakly convergent subsequence, and then show that the weak limit is a minimizer — for which the next fact is essential.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Weak Lower Semicontinuity of the Norm)</span></p>

$$u_n \rightharpoonup u \implies \lVert u \rVert \le \liminf_{n \to \infty} \lVert u_n \rVert.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Taking $v = u$ in the definition, $\lVert u \rVert^2 = \lim\_n \langle u\_n, u \rangle \le \liminf\_n \lVert u\_n \rVert \lVert u \rVert$ by Cauchy–Schwarz. If $u \ne 0$, divide by $\lVert u \rVert$. (Alternatively, in the separable case: $\sum\_{k \le p} \lvert \langle u, e\_k \rangle \rvert^2 = \lim\_n \sum\_{k \le p} \lvert \langle u\_n, e\_k \rangle \rvert^2 \le \liminf\_n \lVert u\_n \rVert^2$ for each $p$, and let $p \to \infty$.) $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Drop in norm of a weak limit)</span></p>

The inequality can be strict — for an orthonormal sequence, $\lVert 0 \rVert = 0 < 1 = \lVert e\_n \rVert$. The norm can only **drop** in a weak limit, and the next result says that this drop is exactly the obstruction to strong convergence.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Weak Convergence + Convergence of Norms = Strong Convergence)</span></p>

Let $v\_n \rightharpoonup v$. Then

$$v\_n \to v \text{ strongly } \iff \lVert v\_n \rVert \to \lVert v \rVert.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

$(\Rightarrow)$ is continuity of the norm. $(\Leftarrow)$ Expand

$$\lVert v_n - v \rVert^2 = \lVert v_n \rVert^2 - 2\operatorname{Re}\langle v_n, v \rangle + \lVert v \rVert^2 \longrightarrow \lVert v \rVert^2 - 2\lVert v \rVert^2 + \lVert v \rVert^2 = 0,$$

using $\langle v\_n, v \rangle \to \langle v, v \rangle$ by weak convergence. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Compactness via Weak Convergence)</span></p>

Let $H$ be a separable Hilbert space. A subset $K \subset H$ is compact if and only if $K$ is closed, bounded, and every sequence in $K$ which converges weakly in $H$ converges strongly.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

$(\Rightarrow)$ Compact sets are closed and bounded. Let $u\_n \in K$ with $u\_n \rightharpoonup u$. Every subsequence has, by compactness, a further subsequence converging strongly to some $w \in K$; strong limits are weak limits, so $w = u$. Since every subsequence has a sub-subsequence converging to the same $u$, the whole sequence converges strongly to $u$.

$(\Leftarrow)$ Given a sequence in $K$, it is bounded, so it has a weakly convergent subsequence; by hypothesis this subsequence converges strongly, and its limit lies in $K$ because $K$ is closed. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Three Characterizations of Compact Sets)</span></p>

For $K \subset H$ closed and bounded in a separable Hilbert space, the following are equivalent to compactness:

* equi-small tails with respect to one (equivalently every) orthonormal basis (Theorem 197);
* finite-dimensional approximability (Theorem 199);
* weakly convergent sequences in $K$ converge strongly (the theorem above).

The same trichotomy reappears for operators: $K$ is compact iff it is a norm limit of finite rank operators iff it maps weakly convergent sequences to strongly convergent ones (Section "Compact Operators").

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weak Convergence via Dual Pairing)</span></p>

The definition of weak convergence via inner products is specific to Hilbert spaces. More generally, for a normed space $X$ with continuous dual $X^\ast$, **weak convergence is defined through the dual pairing**:

$$x_n \rightharpoonup x \iff \forall f\in X^*,\quad \underbrace{\langle x_n,f\rangle_{X,X^*}}_{f(x_n)} \longrightarrow \underbrace{\langle x,f\rangle_{X,X^*}}_{f(x)}.$$

In a Hilbert space $H$, the **Riesz representation theorem** identifies $H^\ast$ with $H$: every continuous linear functional $f\in H^\ast$ has a unique representation

$$f(u)=\langle u,v_f\rangle_H, \qquad v_f\in H,$$

where the inner product is linear in the first argument. Consequently, the general definition reduces to

$$u_n\rightharpoonup u \iff \forall v\in H,\quad \langle u_n,v\rangle_H\to\langle u,v\rangle_H.$$

Thus, **weak convergence via inner products is a special case of weak convergence via dual pairing**. The latter applies to arbitrary normed spaces, even when no inner product exists. **Summary:** The definition of weak convergence in this notes works only for Hilbert space since every continuous linear functional is representable by an inner product via Riesz representation theorem. In more general spaces we define weak convergence using continuous linear functionals $f$ from a dual space.

</div>

## Classes of Operators

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name">($\mathcal{B}(H, H)$ = $\mathcal{B}(H)$)</span></p>

From here on, $H$ will be a Hilbert space, and we denote $\mathcal{B}(H, H)$ by $\mathcal{B}(H)$.

</div>

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

A more economical normal form: if $\lbrace f\_1, \dots, f\_N \rbrace$ is an orthonormal basis of $\operatorname{Range}(T)$, then $Tu = \sum\_i \langle Tu, f\_i \rangle f\_i = \sum\_i \langle u, T^\ast f\_i \rangle f\_i$, i.e.

$$Tu = \sum_{i=1}^N \langle u, e_i \rangle f_i, \qquad e_i = T^\ast f_i, \qquad\text{and then}\qquad T^\ast v = \sum_{i=1}^N \langle v, f_i \rangle e_i.$$

The roles of the $e\_i$ and the $f\_i$ are simply interchanged when passing to the adjoint.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Row Rank = Column Rank)</span></p>

For every finite rank operator $T$ on a Hilbert space, $\dim \operatorname{Range}(T) = \dim \operatorname{Range}(T^\ast)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Write $Tu = \sum\_{i=1}^N \langle u, e\_i \rangle f\_i$ with $\lbrace f\_i \rbrace$ an orthonormal basis of $\operatorname{Range}(T)$, $N = \dim \operatorname{Range}(T)$. The $e\_i$ are linearly independent: if, say, $e\_N = \sum\_{j < N} c\_j e\_j$, then

$$Tu = \sum_{j < N} \langle u, e_j \rangle (f_j + \overline{c_j} f_N),$$

so $\operatorname{Range}(T)$ would have dimension at most $N - 1$. Since $T^\ast v = \sum\_i \langle v, f\_i \rangle e\_i$ and $v = f\_j$ gives $T^\ast f\_j = e\_j$, the range of $T^\ast$ is $\operatorname{Span}\lbrace e\_i \rbrace$, of dimension $N$. $\square$

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

Illustrative examples of compact operators include $K : \ell^2 \to \ell^2$ sending 

$$a = (a_1, a_2, a_3, \dots) \mapsto \left( \frac{a_1}{1}, \frac{a_2}{2}, \frac{a_3}{3}, \dots \right),$$

as well as $T : L^2 \to L^2$ 

$$f(x) \mapsto \int_0^1 K(x, y) f(y)\,dy$$

for some continuous function $K : [0, 1] \times [0, 1] \to \mathbb{R}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span>(Importance of integral operators)</p>

The **integral operator** is particularly important because it comes up in solutions to differential equations: for instance, if we take 

1. $K(x, y) = (x-1)y$ for $0 \le y \le x \le 1$
2. $K(x, y) = x(y-1)$ for $0 \le x \le y \le 1$
   
then

$$u(x) = \int_0^1 K(x, y) f(y)\,dy$$

satisfies 

* $u'' = f$, 
* $u(0) = u(1) = 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Integral Operators with Continuous Kernels are Compact)</span></p>

Let $K \in C([a,b]\times[a,b])$, and define the integral operator

$$Tf(x) = \int_a^b K(x,y)f(y)\,dy, \qquad f\in L^2([a,b]).$$

Then $T\in\mathcal B(L^2([a,b])),$ and $T$ is compact.

Equivalently, for every bounded sequence $\lbrace f_n\rbrace_n$ in $L^2([a,b])$, the sequence $\lbrace Tf_n\rbrace_n$ has a subsequence converging in $L^2([a,b])$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We first show that $T$ is bounded.

Since $K$ is continuous on the compact set $[a,b]\times[a,b]$, it is bounded. Let

$$M:=\sup_{(x,y)\in[a,b]^2}|K(x,y)|<\infty.$$

For every $x\in[a,b]$, by Cauchy--Schwarz,

$$
\begin{aligned}
|Tf(x)|
&=
\left|
\int_a^b K(x,y)f(y)\,dy
\right|\\
&\le
\left(
\int_a^b |K(x,y)|^2\,dy
\right)^{1/2}
\lVert f\rVert_2\\
&\le
M\sqrt{b-a}\,\lVert f\rVert_2.
\end{aligned}
$$

Therefore

$$\lVert Tf\rVert_\infty \le M\sqrt{b-a}\,\lVert f\rVert_2.$$

Hence

$$\lVert Tf\rVert_2 \le\sqrt{b-a}\,\lVert Tf\rVert_\infty\le M(b-a)\lVert f\rVert_2,$$

so $T$ is bounded.

Now let $\lbrace f_n\rbrace_n$ be a bounded sequence in $L^2([a,b])$. Thus there exists $C>0$ such that

$$\lVert f_n\rVert_2\le C$$

for all $n$.

The previous estimate gives

$$\lVert Tf_n\rVert_\infty \le M\sqrt{b-a}\,C,$$

so $\lbrace Tf_n\rbrace_n$ is uniformly bounded.

Next, for $x,z\in[a,b]$,

$$
\begin{aligned}
|Tf_n(x)-Tf_n(z)|
&=
\left|
\int_a^b
\bigl(K(x,y)-K(z,y)\bigr)f_n(y)\,dy
\right|\\
&\le
\left(
\int_a^b
|K(x,y)-K(z,y)|^2\,dy
\right)^{1/2}
\lVert f_n\rVert_2\\
&\le
C\sqrt{b-a}\,
\sup_{y\in[a,b]}
|K(x,y)-K(z,y)|.
\end{aligned}
$$

Because $K$ is continuous on the compact set $[a,b]^2$, it is uniformly continuous. Hence the right-hand side tends to $0$ as $\lvert x-z\rvert\to0$, uniformly in $n$.

Therefore $\lbrace Tf_n\rbrace_n$ is equicontinuous.

By the Arzelà--Ascoli theorem, there exists a subsequence $\lbrace Tf_{n_j}\rbrace_j$ converging uniformly on $[a,b]$.

Since uniform convergence implies $L^2$ convergence on a finite interval,

$$\lVert Tf_{n_j}-g\rVert_2 \le \sqrt{b-a}\, \lVert Tf_{n_j}-g\rVert_\infty \to0.$$

Thus every bounded sequence in $L^2([a,b])$ has an image subsequence converging in $L^2([a,b])$.

Hence $T$ is compact.

</details>
</div>

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
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Compact = Weakly Convergent $\mapsto$ Strongly Convergent)</span></p>

Let $H$ be a separable Hilbert space. An operator $K \in \mathcal{B}(H)$ is compact if and only if for every weakly convergent sequence $u\_n \rightharpoonup u$ in $H$, the image sequence converges in norm: $Ku\_n \to Ku$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Any bounded operator preserves weak convergence: $\langle Ku\_n, v \rangle = \langle u\_n, K^\ast v \rangle \to \langle u, K^\ast v \rangle = \langle Ku, v \rangle$.

$(\Rightarrow)$ Let $u\_n \rightharpoonup u$. Weakly convergent sequences are bounded, $\lVert u\_n \rVert \le C$, so $Ku\_n \in C \cdot \overline{K(\lbrace \lVert u \rVert \le 1 \rbrace)}$, a compact set. Hence every subsequence of $Ku\_n$ has a further subsequence converging in norm to some $w$; norm convergence implies weak convergence, and $Ku\_n \rightharpoonup Ku$, so $w = Ku$ by uniqueness of weak limits. A sequence in a metric space all of whose subsequences have sub-subsequences converging to the **same** limit $Ku$ converges to $Ku$ (otherwise some subsequence stays $\varepsilon$-far from $Ku$).

$(\Leftarrow)$ Let $Ku\_n$ be a sequence in $K(\lbrace \lVert u \rVert \le 1 \rbrace)$. The bounded sequence $u\_n$ has a weakly convergent subsequence $u\_{n\_j} \rightharpoonup u$, and then $Ku\_{n\_j} \to Ku$ in norm. So the image of the unit ball has compact closure. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Finite Rank Operators Upgrade Weak to Strong Convergence)</span></p>

In particular, if $A$ has finite-dimensional range and $v\_n \rightharpoonup v$, then $Av\_n \to Av$. This is also easy to see directly: with an orthonormal basis $f\_1, \dots, f\_N$ of $\operatorname{Range}(A)$, $Av\_n = \sum\_i \langle v\_n, A^\ast f\_i \rangle f\_i$, and each of the finitely many coefficients converges. This is the mechanism behind the backward direction of Theorem 210.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Finite-Rank Operators Map Weak Convergence to Strong Convergence)</span></p>

Let $H_1,H_2$ be Hilbert spaces and $A\in\mathcal{B}(H_1,H_2)$ have finite-dimensional range. Then

$$u_n\rightharpoonup u \quad\Longrightarrow\quad \|Au_n-Au\|_{H_2}\longrightarrow 0.$$

In particular, every finite-rank operator is completely continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

TODO:

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Integral Operators via Finite Rank Approximation)</span></p>

Theorem 210 gives a second proof that an integral operator $Tf(x) = \int\_a^b K(x, y) f(y)\,dy$ with $K \in C([a, b]^2)$ is compact, without Arzelà–Ascoli:

* $x \mapsto K(x, \cdot)$ is continuous $[a, b] \to L^2([a, b])$ (uniform continuity of $K$), so its image is a compact subset of $L^2$ and hence has **equi-small tails** with respect to any orthonormal basis $\lbrace \varphi\_k \rbrace$ (Theorem 197).
* Truncating the expansion of $K(x, \cdot)$ at $k \le n$ gives kernels $K\_n(x, y) = \sum\_{k \le n} a\_k(x) \varphi\_k(y)$ with $\sup\_x \lVert K(x, \cdot) - K\_n(x, \cdot) \rVert\_2 \to 0$, i.e. the Fourier series converges **uniformly in $x$**.
* The corresponding operators $T\_n$ have range in $\operatorname{Span}\lbrace a\_1, \dots, a\_n \rbrace$, so they have finite rank, and by Cauchy–Schwarz $\lVert T - T\_n \rVert \le \sqrt{b - a}\,\sup\_x \lVert K(x, \cdot) - K\_n(x, \cdot) \rVert\_2 \to 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 211</span></p>

Let $H$ be a separable Hilbert space, and let $K(H)$ be the set of compact operators on $H$. Then:

1. $K(H)$ is a closed subspace of $\mathcal{B}(H)$.
2. For any $T \in K(H)$, we also have $T^\ast \in K(H)$.
3. For any $T \in K(H)$ and $A, B \in \mathcal{B}(H)$, we have $ATB \in K(H)$.

</div>

In other words, the set of compact operators is also a star-closed, two-sided ideal in $\mathcal{B}(H)$.

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>(1) is clear because $K(H) = \overline{\mathcal{R}(H)}$. For (2), if $T \in K(H)$ then $T_n \to T$ for finite rank $T_n$, so $T_n^* \to T^*$ (since $\lVert T_n^* - T^* \rVert = \lVert T_n - T \rVert$), and each $T_n^*$ is finite rank. For (3), $AT_n B$ is finite rank for each $n$, and $\lVert AT_n B - ATB \rVert \le \lVert A \rVert \cdot \lVert T_n - T \rVert \cdot \lVert B \rVert \to 0$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Ideals and the Calkin Algebra)</span></p>

The ideal property is the analogue of normality of a subgroup: it is exactly what makes the quotient $\mathcal{B}(H)/K(H)$ again an algebra (products of cosets are well defined). Because $K(H)$ is closed, the quotient is also a Banach space; it is called the **Calkin algebra**. Compact operators are "small" in the sense that they become zero there — which is why $I - K$ behaves like the identity (see the Fredholm theory below).

</div>

## Eigenvalues and the Spectrum

### Invertible Operators

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 212</span></p>

Let $T \in \mathcal{B}(H)$ be a bounded linear operator. If $\lVert T \rVert < 1$, then $I - T$ is invertible, and we can compute its inverse to be the absolutely summable series

$$(I - T)^{-1} = \sum_{n=0}^{\infty} T^n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof (Neumann series)</summary>

Since $\lVert T^n \rVert \le \lVert T \rVert^n$ and $\lVert T \rVert < 1$, the series $\sum\_{n \ge 0} T^n$ is absolutely summable in the Banach space $\mathcal{B}(H)$, hence converges to some $B \in \mathcal{B}(H)$. By continuity of composition in operator norm,

$$TB = \lim_{N \to \infty} \sum_{n=0}^{N} T^{n+1} = B - I, \qquad BT = B - I,$$

so $(I - T)B = B(I - T) = I$, and $B$ is a two-sided (hence the) inverse of $I - T$. $\square$

</details>
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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($GL(H)$ Is a Group; Invertible = Bijective)</span></p>

$GL(H)$ is a group under composition: $(G\_1 G\_2)^{-1} = G\_2^{-1} G\_1^{-1}$. By the Bounded Inverse Theorem, an operator $A \in \mathcal{B}(H)$ lies in $GL(H)$ as soon as it is **bijective** — boundedness of the inverse is automatic. The proof above gives the explicit radius: the ball of radius $\lVert G^{-1} \rVert^{-1}$ around $G \in GL(H)$ consists of invertible operators.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Shift Operator)</span></p>

Let $\lbrace e\_j \rbrace\_{j \ge 1}$ be an orthonormal basis of $H$ (e.g. $H = \ell^2$). The **(right) shift**

$$S\Big(\sum_{j=1}^\infty c_j e_j\Big) = \sum_{j=1}^\infty c_j e_{j+1}$$

is an isometry ($\lVert Su \rVert = \lVert u \rVert$, so $\lVert S \rVert = 1$), hence injective, but it is **not surjective**: its range is exactly the closed subspace

$$H_1 = \lbrace u \in H : \langle u, e_1 \rangle = 0 \rbrace = \lbrace e_1 \rbrace^\perp.$$

So $S$ is invertible as a map $H \to H\_1$, but not as a map $H \to H$. In finite dimensions an injective linear map from a space to itself is automatically surjective; the shift shows this fails in infinite dimensions. Its adjoint is the left shift $S^\ast e\_1 = 0$, $S^\ast e\_{j+1} = e\_j$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">($GL(H)$ Is Not Dense)</span></p>

Let $S$ be the shift and $B \in \mathcal{B}(H)$ with $\lVert B \rVert < 1$. Then $S + B$ is not invertible; in fact $e\_1 \notin \operatorname{Range}(S + B)$. Consequently, when $\dim H = \infty$, the open ball of radius $1$ around $S$ contains no invertible operator, so $GL(H)$ is open but **not dense** in $\mathcal{B}(H)$ — in sharp contrast with matrices, where invertible matrices are dense.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Let $\Pi\_1$ be the orthogonal projection onto $H\_1 = \lbrace e\_1 \rbrace^\perp$ and $B^{\prime} = \Pi\_1 B$, i.e. $B^{\prime}u = Bu - \langle Bu, e\_1 \rangle e\_1$, a bounded operator $H \to H\_1$ with $\lVert B^{\prime} \rVert \le \lVert B \rVert < 1$.

Regard $S : H \to H\_1$ as a bijective isometry, with inverse $S^{-1} : H\_1 \to H$ of norm $1$. Then

$$S + B^{\prime} = S\,(I + S^{-1}B^{\prime}) : H \to H_1,$$

and $\lVert S^{-1}B^{\prime} \rVert < 1$, so by the Neumann series $S + B^{\prime}$ is a bijection $H \to H\_1$; in particular it is injective.

Now suppose $(S + B)u = e\_1$. Since $\operatorname{Range}(S) \subset H\_1$, applying $\Pi\_1$ gives $(S + B^{\prime})u = \Pi\_1 e\_1 = 0$, so $u = 0$ by injectivity — but then $(S + B)u = 0 \ne e\_1$. Contradiction. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Unitary Group)</span></p>

The **unitary group** of $H$ is the subgroup

$$\mathcal{U}(H) = \lbrace U \in GL(H) : U^{-1} = U^\ast \rbrace \subset GL(H).$$

Its elements are exactly the isometric isomorphisms of $H$ onto itself:

$$\langle Uu, Uv \rangle = \langle u, v \rangle, \qquad \lVert Uu \rVert = \lVert u \rVert \qquad \forall u, v \in H.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($\mathcal{U}(H)$ vs. $U(n)$)</span></p>

* An isometry need not be unitary: the shift $S$ preserves inner products but is not surjective, so $S \notin \mathcal{U}(H)$ ($S^\ast S = I$ but $SS^\ast \ne I$).
* For a separable infinite-dimensional $H$, $\mathcal{U}(H)$ (with the norm topology) is **contractible** — it has no interesting topology at all (Kuiper's theorem; stated without proof). The finite unitary groups $U(n)$ have very rich topology, so $\mathcal{U}(H)$ does not look like a "limit" of the $U(n)$.

</div>

### The Spectrum

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 214</span><span class="math-callout__name">(Resolvent Set and Spectrum)</span></p>

Let $A \in \mathcal{B}(H)$ be a bounded linear operator. The **resolvent set** of $A$, denoted $\operatorname{Res}(A)$, is the set $\lbrace \lambda \in \mathbb{C} : A - \lambda I \in GL(H) \rbrace$, and the **spectrum** of $A$, denoted $\operatorname{Spec}(A)$, is the complement $\mathbb{C} \setminus \operatorname{Res}(A)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 215</span></p>

Let $A : \mathbb{C}^2 \to \mathbb{C}^2$ be the matrix $\begin{bmatrix} \lambda_1 & 0 \\\ 0 & \lambda_2 \end{bmatrix}$. Then $A - \lambda$ is not invertible exactly when $\lambda = \lambda_1$ or $\lambda = \lambda_2$, so $\operatorname{Spec}(A) = \lbrace \lambda_1, \lambda_2 \rbrace$.

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

If we have a self-adjoint operator $A \in \mathcal{B}(H)$, meaning that $A = A^\ast$, then $\langle Au, u \rangle$ is real for all $u$, and $\lVert A \rVert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle \rvert$.

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

<figure class="math-figure">
  <svg viewBox="0 0 460 220" xmlns="http://www.w3.org/2000/svg" width="480" aria-label="Spectrum of a self-adjoint operator on the complex plane">
    <g stroke="#bbb" stroke-width="0.5" fill="none">
      <circle cx="230" cy="110" r="150" stroke-dasharray="4,4" />
    </g>
    <text x="380" y="40" font-size="11" fill="#888" font-style="italic">‖A‖-disk in ℂ</text>
    <line x1="40"  y1="110" x2="420" y2="110" stroke="#444" stroke-width="1.2" />
    <line x1="230" y1="20"  x2="230" y2="200" stroke="#444" stroke-width="1.2" />
    <polygon points="420,110 412,106 412,114" fill="#444" />
    <polygon points="230,20 226,28 234,28" fill="#444" />
    <text x="425" y="114" font-size="11" fill="#666">Re λ</text>
    <text x="236" y="20"  font-size="11" fill="#666">Im λ</text>
    <line x1="80" y1="105" x2="80" y2="115" stroke="#444" stroke-width="1.2" />
    <line x1="380" y1="105" x2="380" y2="115" stroke="#444" stroke-width="1.2" />
    <text x="56"  y="135" font-size="11" fill="#666">−‖A‖</text>
    <text x="370" y="135" font-size="11" fill="#666">‖A‖</text>
    <line x1="125" y1="108" x2="320" y2="108" stroke="#2c4994" stroke-width="6" stroke-linecap="round" opacity="0.35" />
    <line x1="125" y1="110" x2="320" y2="110" stroke="#2c4994" stroke-width="3" stroke-linecap="round" />
    <g fill="#d65336">
      <circle cx="125" cy="110" r="4" />
      <circle cx="320" cy="110" r="4" />
    </g>
    <g fill="#2c4994">
      <circle cx="160" cy="110" r="3.5" />
      <circle cx="200" cy="110" r="3.5" />
      <circle cx="248" cy="110" r="3.5" />
      <circle cx="285" cy="110" r="3.5" />
    </g>
    <text x="100" y="100" font-size="12" fill="#d65336" font-weight="600">a₋</text>
    <text x="320" y="100" font-size="12" fill="#d65336" font-weight="600">a₊</text>
    <text x="160" y="146" font-size="11" fill="#2c4994">λ₁</text>
    <text x="200" y="146" font-size="11" fill="#2c4994">λ₂</text>
    <text x="248" y="146" font-size="11" fill="#2c4994">λ₃</text>
    <text x="285" y="146" font-size="11" fill="#2c4994">λ₄</text>
    <text x="40" y="180" font-size="11" fill="#444" font-style="italic">a₋ = inf ⟨Au,u⟩,    a₊ = sup ⟨Au,u⟩,    Spec(A) ⊂ [a₋, a₊] ⊂ ℝ</text>
  </svg>
  <figcaption>For a self-adjoint operator the entire spectrum collapses onto the real axis (the imaginary part of $\langle (A-it)u, u\rangle$ is $-t\|u\|^2$, blocking complex eigenvalues), and is further pinched into the numerical range $[a_-, a_+]$. Both endpoints are themselves spectral — they are realized by sequences of unit vectors driving $\langle Au,u\rangle$ to the supremum or infimum.</figcaption>
</figure>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since $-\lVert A \rVert \le \langle Au, u \rangle \le \lVert A \rVert$ for all $u$, we must have $-\lVert A \rVert \le a_- \le a_+ \le \lVert A \rVert$. By the definition of $a_-, a_+$, there exist two sequences $\lbrace u_n^\pm \rbrace$ of unit vectors so that $\langle Au_n^\pm, u_n^\pm \rangle \to a_\pm$. The same argument as in Theorem 223 shows that $\langle (A - a_\pm) u_n^\mp, u_n^\mp \rangle \to 0$, which implies $a_+$ and $a_-$ are both in the spectrum.</p>
    <p>To show the spectrum is contained within $[a_-, a_+]$, let $b = \frac{a_+ + a_-}{2}$ be their midpoint and $B = A - bI$. Since $b$ is a real number, $B$ is also a bounded self-adjoint operator, so by Theorem 223, $\operatorname{Spec}(B) \subset [-\lVert B \rVert, \lVert B \rVert]$. This means $\operatorname{Spec}(A) \subset [-\lVert B \rVert + b, \lVert B \rVert + b]$. We have $\lVert B \rVert = \sup_{\lVert u \rVert = 1} \lvert \langle Bu, u \rangle \rvert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle - \frac{a_+ + a_-}{2} \rvert$. Since $\langle Au, u \rangle$ always lies in $[a_-, a_+]$, this supremum equals $\frac{a_+ - a_-}{2}$, so $\operatorname{Spec}(A) \subset [-\lVert B \rVert + b, \lVert B \rVert + b] = [a_-, a_+]$. $\square$</p>
  </details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
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

<figure class="math-figure">
  <svg viewBox="0 0 480 160" xmlns="http://www.w3.org/2000/svg" width="500" aria-label="Eigenvalues of a compact self-adjoint operator accumulating at 0">
    <line x1="20"  y1="100" x2="460" y2="100" stroke="#444" stroke-width="1.2" />
    <polygon points="460,100 452,96 452,104" fill="#444" />
    <text x="450" y="120" font-size="11" fill="#666">ℝ</text>
    <line x1="240" y1="92" x2="240" y2="108" stroke="#444" stroke-width="1.2" />
    <text x="234" y="125" font-size="11" fill="#222">0</text>
    <g fill="#2c4994">
      <circle cx="80"  cy="100" r="10" />
      <circle cx="135" cy="100" r="7.5" />
      <circle cx="178" cy="100" r="5.6" />
      <circle cx="208" cy="100" r="4.2" />
      <circle cx="225" cy="100" r="3.1" />
      <circle cx="234" cy="100" r="2.3" />
    </g>
    <g fill="#d65336">
      <circle cx="400" cy="100" r="10" />
      <circle cx="345" cy="100" r="7.5" />
      <circle cx="302" cy="100" r="5.6" />
      <circle cx="272" cy="100" r="4.2" />
      <circle cx="255" cy="100" r="3.1" />
      <circle cx="246" cy="100" r="2.3" />
    </g>
    <text x="68"  y="86"  font-size="11" fill="#2c4994">λ₁</text>
    <text x="123" y="86"  font-size="11" fill="#2c4994">λ₂</text>
    <text x="167" y="86"  font-size="10" fill="#2c4994">λ₃</text>
    <text x="396" y="86"  font-size="11" fill="#d65336">μ₁</text>
    <text x="338" y="86"  font-size="11" fill="#d65336">μ₂</text>
    <text x="294" y="86"  font-size="10" fill="#d65336">μ₃</text>
    <line x1="60"  y1="138" x2="240" y2="138" stroke="#666" stroke-width="0.8" />
    <text x="100" y="152" font-size="10" fill="#666" font-style="italic">|λₙ| → 0</text>
    <line x1="240" y1="138" x2="420" y2="138" stroke="#666" stroke-width="0.8" />
    <text x="305" y="152" font-size="10" fill="#666" font-style="italic">|μₙ| → 0</text>
  </svg>
  <figcaption>For a compact self-adjoint operator the nonzero spectrum is purely a sequence (or finite list) of real eigenvalues, with $0$ as the only possible accumulation point. The size of each disk indicates $|\lambda_n|$; eigenvectors of distinct eigenvalues are orthogonal, and once compactness forces $\|Au_n\| = |\lambda_n|$ to be small, no infinite subset can stay separated from $0$.</figcaption>
</figure>

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

### Fredholm Operators and $I - K$ for Non-Self-Adjoint Compact $K$

The remark above can be pushed much further: for **any** compact $K$ (self-adjoint or not), $I - K$ behaves like a square matrix. Since $\lambda$ is an eigenvalue of $K$ iff $(I - \lambda^{-1}K)u = 0$ has a nontrivial solution, and $\lambda^{-1}K$ is again compact, this is the right object for the spectral theory of compact operators.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fredholm Operator and Index)</span></p>

A bounded operator $F \in \mathcal{B}(H)$ is **Fredholm** if

1. $\operatorname{Null}(F)$ is finite-dimensional,
2. $\operatorname{Range}(F)$ is closed, and
3. $\operatorname{Range}(F)^\perp$ is finite-dimensional.

Its **index** is the integer

$$\operatorname{ind}(F) = \dim \operatorname{Null}(F) - \dim \operatorname{Range}(F)^\perp.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conditions 2 and 3 Are Independent)</span></p>

Since $W^\perp = \overline{W}^\perp$, condition 3 says nothing about closedness. Multiplication by $x$ on $L^2([0,1])$ (Example 218) is injective and has **dense** range (every $L^2$ function vanishing on some $[0, \varepsilon)$ is in the range, and these are dense), so $\operatorname{Range}^\perp = \lbrace 0 \rbrace$ — yet the range is not closed, since the operator is not invertible.

By Theorem 189, $\operatorname{Range}(F)^\perp = \operatorname{Null}(F^\ast)$, so condition 3 is the same as finite-dimensionality of $\operatorname{Null}(F^\ast)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Shift Has Index $-1$)</span></p>

The right shift $S$ on $\ell^2$ (see Section "Invertible Operators") is injective with closed range $\lbrace e_1 \rbrace^\perp$. So $S$ is Fredholm with $\operatorname{ind}(S) = 0 - 1 = -1$. The theorem below shows this can never happen for $I - K$ with $K$ compact.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Subspaces Containing a Closed Finite-Codimensional Subspace)</span></p>

Let $V \subset H$ be a linear subspace containing a closed subspace $W$ of finite codimension, i.e. there are $e_1, \dots, e_N \in H$ with $H = W + \operatorname{Span}\lbrace e_1, \dots, e_N \rbrace$. Then $V$ is closed.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Projecting $H = W + \operatorname{Span}\lbrace e\_i \rbrace$ onto $W^\perp$ shows $W^\perp = \operatorname{Span}\lbrace \Pi\_{W^\perp} e\_i \rbrace$ is finite-dimensional. For $v \in V$ write $v = w + w^\perp$ with $w \in W \subset V$; then $w^\perp = v - w \in V$. Hence

$$V = W \oplus E, \qquad E := V \cap W^\perp,$$

with $E$ finite-dimensional, hence closed. If $v\_n = w\_n + e\_n \to v$ with $w\_n \in W$, $e\_n \in E$, then by continuity of the orthogonal projections $w\_n = \Pi\_W v\_n \to \Pi\_W v \in W$ and $e\_n = \Pi\_{W^\perp} v\_n \to \Pi\_{W^\perp} v \in E$. So $v \in W \oplus E = V$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">($I - K$ Is Fredholm of Index $0$)</span></p>

Let $K \in \mathcal{B}(H)$ be compact on a separable Hilbert space. Then $I - K$ is Fredholm:

$$\dim \operatorname{Null}(I - K) < \infty, \qquad \operatorname{Range}(I - K) \text{ is closed}, \qquad \dim \operatorname{Range}(I - K)^\perp < \infty,$$

and moreover

$$\dim \operatorname{Null}(I - K) = \dim \operatorname{Range}(I - K)^\perp = \dim \operatorname{Null}(I - K^\ast),$$

i.e. $\operatorname{ind}(I - K) = 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

**Step 1: finite rank $K = T$.** Write $Tu = \sum\_{i=1}^N \langle u, e\_i \rangle f\_i$ with $\lbrace f\_i \rbrace$ a basis of $\operatorname{Range}(T)$ (Lemma "Row Rank = Column Rank" below shows the $e\_i$ are then linearly independent).

* $\operatorname{Null}(I - T) = \lbrace u : u = Tu \rbrace \subset \operatorname{Range}(T)$ is finite-dimensional.
* If $Tu = 0$ then $(I - T)u = u$, so $\operatorname{Range}(I - T) \supset \operatorname{Null}(T) = \lbrace e\_1, \dots, e\_N \rbrace^\perp$, a closed subspace of finite codimension. By the Lemma, $\operatorname{Range}(I - T)$ is closed.
* $\operatorname{Range}(I - T)^\perp = \operatorname{Null}(I - T^\ast)$ is finite-dimensional by the first bullet applied to the finite rank operator $T^\ast$.

**Step 2: general compact $K$.** By Theorem 210, choose a finite rank $T$ with $\lVert K - T \rVert < \frac{1}{2}$ and set $B = K - T$. Then $I - B$ is invertible (Proposition 212), and

$$I - K = (I - B) - T = (I - B)(I - T^{\prime}) = (I - T^{\prime\prime})(I - B),$$

$$T^{\prime} = (I - B)^{-1}T, \qquad T^{\prime\prime} = T(I - B)^{-1},$$

both of finite rank by the ideal property. Since $I - B$ is a bijection,

* $\operatorname{Null}(I - K) = \operatorname{Null}(I - T^{\prime})$ is finite-dimensional,
* $\operatorname{Range}(I - K) = \operatorname{Range}(I - T^{\prime\prime})$ is closed,
* $\operatorname{Range}(I - K)^\perp = \operatorname{Null}(I - K^\ast)$ is finite-dimensional because $K^\ast$ is compact (Theorem 211).

**Step 3: index zero.** Taking adjoints in $I - K = (I - B)(I - T^{\prime})$ gives $I - K^\ast = (I - T^{\prime\ast})(I - B)^\ast$ with $(I - B)^\ast$ invertible, so

$$\dim \operatorname{Null}(I - K) = \dim \operatorname{Null}(I - T^{\prime}), \qquad \dim \operatorname{Null}(I - K^\ast) = \dim \operatorname{Null}(I - T^{\prime\ast}).$$

So it suffices to treat finite rank $T u = \sum\_{i} \langle u, e\_i \rangle f\_i$, for which $T^\ast v = \sum\_i \langle v, f\_i \rangle e\_i$. Let $W = \operatorname{Span}\lbrace e\_i, f\_i \rbrace$, a finite-dimensional space. Both $T$ and $T^\ast$ map $H$ into $W$, so any solution of $u = Tu$ or $u = T^\ast u$ lies in $W$. Let $R = (I - T)\vert\_W : W \to W$; for $w, w^{\prime} \in W$ we have $\langle Rw, w^{\prime} \rangle = \langle w, (I - T^\ast)w^{\prime} \rangle$, so $R^\ast = (I - T^\ast)\vert\_W$. Therefore

$$\dim \operatorname{Null}(I - T) = \dim \operatorname{Null}(R) = \dim W - \operatorname{rank}(R) = \dim W - \operatorname{rank}(R^\ast) = \dim \operatorname{Null}(R^\ast) = \dim \operatorname{Null}(I - T^\ast),$$

using rank-nullity and row rank $=$ column rank for the square matrix of $R$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Fredholm Alternative for Compact Operators)</span></p>

Let $K$ be compact. Then $I - K$ is injective if and only if it is surjective (in which case it is invertible). In general, $(I - K)u = f$ is solvable iff $f \perp \operatorname{Null}(I - K^\ast)$, a finite-dimensional space with the same dimension as the space of solutions of $(I - K)u = 0$. If $K = K^\ast$, this is exactly Theorem 228 with $\lambda = 1$.

</div>

### Eigenvalue Existence and the Maximum Principle

We've shown previously that one of $\pm \lVert A \rVert$ must be in the spectrum, and that gives us:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 230</span></p>

Let $A = A^*$ be a nontrivial compact self-adjoint operator. Then $A$ has a nontrivial eigenvalue $\lambda_1$ with 

$$\lvert \lambda_1 \rvert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle \rvert = \lvert \langle Au_1, u_1 \rangle \rvert,$$

where $u_1$ is a normalized eigenvector (with $\lVert u_1 \rVert = 1$) satisfying $Au_1 = \lambda_1 u_1$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Since at least one of $\pm \lVert A \rVert$ is in $\operatorname{Spec}(A)$ (and $\lVert A \rVert \ne 0$ because we have a nontrivial operator), at least one of them will be an eigenvalue by the Fredholm alternative, and we call this $\lambda_1$. The equation for $\lambda_1$ follows from $\lVert A \rVert = \sup_{\lVert u \rVert = 1} \lvert \langle Au, u \rangle \rvert$, and $\lvert \langle Au_1, u_1 \rangle \rvert$ equals this because being an eigenvalue implies we have an eigenvector. $\square$</p>
  </details>
</div>

There is a second, more hands-on route to Theorem 230 that avoids the spectrum altogether and mirrors the finite-dimensional proof via maximizing the Rayleigh quotient. Compactness of $A$ replaces compactness of the unit sphere.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Extrema of the Quadratic Form Are Attained at Eigenvectors)</span></p>

Let $A = A^\ast$ be compact on a separable Hilbert space, and let

$$F(u) = \langle Au, u \rangle, \qquad F : \lbrace u \in H : \lVert u \rVert = 1 \rbrace \to \mathbb{R}.$$

If $\Lambda_+ = \sup F > 0$, then the supremum is attained at some unit vector $u_+$, and $Au_+ = \Lambda_+ u_+$. Similarly, if $\Lambda_- = \inf F < 0$, it is attained at an eigenvector with eigenvalue $\Lambda_-$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

$F$ is real (Theorem 221) and bounded by $\lVert A \rVert$. Take unit vectors $u\_n$ with $F(u\_n) \to \Lambda\_+$. By weak sequential compactness of the unit ball (Section "Weak Convergence in Hilbert Spaces"), pass to a subsequence with $u\_n \rightharpoonup u\_+$, so $\lVert u\_+ \rVert \le 1$. Since $A$ is compact it maps weakly convergent sequences to strongly convergent ones, so $Au\_n \to Au\_+$, and

$$\lvert F(u_n) - \langle Au_+, u_+ \rangle \rvert \le \lvert \langle Au_n - Au_+, u_n \rangle \rvert + \lvert \langle Au_+, u_n - u_+ \rangle \rvert \le \lVert Au_n - Au_+ \rVert + \lvert \langle u_n - u_+, Au_+ \rangle \rvert \to 0.$$

So $\langle Au\_+, u\_+ \rangle = \Lambda\_+ > 0$; in particular $u\_+ \ne 0$, and if $\lVert u\_+ \rVert < 1$ then $F(u\_+/\lVert u\_+ \rVert) = \Lambda\_+/\lVert u\_+ \rVert^2 > \Lambda\_+$, impossible. Hence $\lVert u\_+ \rVert = 1$ and the sup is a max.

For the eigenvector property, let $v \perp u\_+$ with $\lVert v \rVert = 1$. The curve $\theta \mapsto \cos\theta\,u\_+ + \sin\theta\,v$ stays on the unit sphere, and

$$F(\cos\theta\,u_+ + \sin\theta\,v) = \cos^2\theta\,\Lambda_+ + \sin(2\theta)\operatorname{Re}\langle Au_+, v \rangle + \sin^2\theta\,F(v)$$

is maximized at $\theta = 0$, so its derivative there, $2\operatorname{Re}\langle Au\_+, v \rangle$, vanishes. Replacing $v$ by $iv$ kills the imaginary part too: $\langle Au\_+, v \rangle = 0$ for all $v \perp u\_+$. Thus $Au\_+ \in (u\_+^\perp)^\perp = \operatorname{Span}\lbrace u\_+ \rbrace$, say $Au\_+ = c\,u\_+$, and $c = \langle Au\_+, u\_+ \rangle = \Lambda\_+$. The infimum case follows by applying this to $-A$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Variational Proof of the Spectral Theorem)</span></p>

If $F \equiv 0$ then $A = 0$ (Theorem 221: $\lVert A \rVert = \sup \lvert F \rvert$; alternatively, recover $\langle Au, v \rangle$ from $F$ by polarization). So a nonzero compact self-adjoint $A$ always has an eigenvector from the Lemma, with $\lvert \lambda \rvert = \max(\Lambda_+, -\Lambda_-) = \lVert A \rVert$. Iterating on $H_N = \operatorname{Null}(A)^\perp \cap \lbrace u_1, \dots, u_N \rbrace^\perp$, which $A$ maps into itself because $\langle Au, u_j \rangle = \lambda_j \langle u, u_j \rangle = 0$, produces the eigenvalues $\lvert \lambda_1 \rvert \ge \lvert \lambda_2 \rvert \ge \cdots$ of the Maximum Principle below. The process only stops if $A \equiv 0$ on some $H_N$, which forces $H_N = \lbrace 0 \rbrace$ (as $H_N \perp \operatorname{Null}(A)$). If it never stops, then $\lvert \lambda_j \rvert \to 0$ (an eigenvalue $\lambda \ne 0$ has a finite-dimensional eigenspace), and the $u_j$ have dense span in $\operatorname{Null}(A)^\perp$: otherwise $A$ would be nonzero on the orthocomplement $H^{\prime}$ of their span inside $\operatorname{Null}(A)^\perp$, and the Lemma would give an eigenvector in $H^{\prime} \subset H_N$ with eigenvalue $\lambda \ne 0$; once $\lvert \lambda_{N+1} \rvert < \lvert \lambda \rvert$, this contradicts the maximality used to choose $\lambda_{N+1}$.

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

<figure class="math-figure">
  <svg viewBox="0 0 380 220" xmlns="http://www.w3.org/2000/svg" width="400" aria-label="First three Dirichlet eigenfunctions on [0,1]">
    <g stroke="#e8e8e8" stroke-width="0.5">
      <line x1="30" y1="30"  x2="350" y2="30" />
      <line x1="30" y1="170" x2="350" y2="170" />
    </g>
    <line x1="30" y1="100" x2="350" y2="100" stroke="#444" stroke-width="1.2" />
    <line x1="30" y1="20"  x2="30"  y2="180" stroke="#444" stroke-width="1.2" />
    <polygon points="350,100 342,96 342,104" fill="#444" />
    <polygon points="30,20 26,28 34,28" fill="#444" />
    <text x="354" y="104" font-size="11" fill="#666">x</text>
    <text x="22"  y="22"  font-size="11" fill="#666">y</text>
    <text x="14"  y="34"  font-size="10" fill="#888">1</text>
    <text x="12"  y="174" font-size="10" fill="#888">−1</text>
    <text x="34"  y="116" font-size="10" fill="#666">0</text>
    <text x="346" y="116" font-size="10" fill="#666">1</text>
    <line x1="350" y1="95" x2="350" y2="105" stroke="#444" stroke-width="1.2" />
    <polyline points="30,100 62,78 94,59 126,43 158,33 190,30 222,33 254,43 286,59 318,78 350,100"
              stroke="#2c4994" stroke-width="2.2" fill="none" />
    <polyline points="30,100 62,59 94,33 126,33 158,59 190,100 222,141 254,167 286,167 318,141 350,100"
              stroke="#3d7a26" stroke-width="2.2" fill="none" />
    <polyline points="30,100 62,43 94,33 126,78 158,141 190,170 222,141 254,78 286,33 318,43 350,100"
              stroke="#d65336" stroke-width="2.2" fill="none" />
    <g fill="#222">
      <circle cx="30"  cy="100" r="3" />
      <circle cx="350" cy="100" r="3" />
    </g>
    <g font-size="11" font-weight="600">
      <rect x="60"  y="194" width="14" height="3" fill="#2c4994" />
      <text x="79"  y="200" fill="#2c4994">u₁,  λ₁ = 1/π²</text>
      <rect x="170" y="194" width="14" height="3" fill="#3d7a26" />
      <text x="189" y="200" fill="#3d7a26">u₂,  λ₂ = 1/4π²</text>
      <rect x="285" y="194" width="14" height="3" fill="#d65336" />
      <text x="304" y="200" fill="#d65336">u₃,  λ₃ = 1/9π²</text>
    </g>
  </svg>
  <figcaption>The first three orthonormal eigenfunctions $u_k(x) = \sqrt{2}\sin(k\pi x)$ of the Green-function operator $A$ on $L^2([0,1])$. Each vanishes at the endpoints (Dirichlet condition), oscillates with $k$ half-periods on $[0,1]$, and corresponds to a smaller eigenvalue $\lambda_k = 1/(k\pi)^2 \to 0$ — the spectral theorem then declares $\{u_k\}$ a Hilbert basis of $L^2([0,1])$.</figcaption>
</figure>

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

## Application: The Periodic Schrödinger Operator

The Dirichlet problem used the Green operator $A$ of $-\frac{d^2}{dx^2}$ with zero boundary values. The same machinery handles **periodic** boundary conditions, where $-\frac{d^2}{dx^2}$ itself is not invertible (constants are killed), so we shift it by $1$ first. This is the content of a problem set in the 2009 version of the course.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Functions on the Circle)</span></p>

Let $\mathbb{S} = \lbrace z \in \mathbb{C} : \lvert z \rvert = 1 \rbrace$. Via $x \mapsto e^{ix}$, continuous functions on $\mathbb{S}$ are the same as continuous $2\pi$-periodic functions on $\mathbb{R}$:

$$C^0(\mathbb{S}) \cong \lbrace u \in C(\mathbb{R}) : u(x + 2\pi) = u(x) \ \forall x \rbrace,$$

and $L^2(\mathbb{S})$ is identified with $L^2(0, 2\pi)$ (a $2\pi$-periodic, locally integrable function modulo periodic null functions is determined by its restriction to $(0, 2\pi)$). The inclusion $C^0(\mathbb{S}) \hookrightarrow L^2(\mathbb{S})$ is dense, and the functions $e^{ikx}/\sqrt{2\pi}$, $k \in \mathbb{Z}$, form an orthonormal basis of $L^2(\mathbb{S})$ by Theorem 177.

</div>

We want to solve, for a real-valued $V \in C^0(\mathbb{S})$ and $f \in C^0(\mathbb{S})$,

$$-u''(x) + V(x)u(x) = f(x), \qquad x \in \mathbb{R}, \quad u \in C^2 \text{ and } 2\pi\text{-periodic}.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Green Operator of $-\frac{d^2}{dx^2} + 1$ on the Circle)</span></p>

Let $G$ be the $2\pi$-periodic function with

$$G(s) = \frac{\cosh(\pi - s)}{2\sinh \pi}, \qquad s \in [0, 2\pi],$$

and define $Sf(x) = \int_0^{2\pi} G(x - y) f(y)\,dy$. Then:

1. For every $f \in C^0(\mathbb{S})$, $u = Sf$ is the **unique** $2\pi$-periodic $C^2$ solution of $-u'' + u = f$.
2. $S$ extends to a compact self-adjoint operator on $L^2(\mathbb{S})$, and
   $$S(e^{ikx}) = \frac{1}{1 + k^2}\,e^{ikx}, \qquad k \in \mathbb{Z}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

**Uniqueness.** If $-w'' + w = 0$ with $w$ periodic and $C^2$, multiply by $\overline{w}$ and integrate over a period. Integrating by parts, the boundary terms cancel by periodicity, leaving $\int_0^{2\pi} (\lvert w^{\prime} \rvert^2 + \lvert w \rvert^2) = 0$, so $w = 0$.

**Existence.** $G$ is smooth on $(0, 2\pi)$ with $G'' = G$ there, $G$ is continuous and periodic (since $G(0) = G(2\pi)$), and its derivative jumps by $-1$ at $s = 0$:

$$G^{\prime}(0^+) - G^{\prime}(2\pi^-) = -\tfrac{1}{2} - \tfrac{1}{2} = -1.$$

Splitting $\int_0^{2\pi} G(x-y)f(y)\,dy$ at $y = x$ and differentiating twice under the integral sign, the jump produces exactly $-u'' + u = f$; periodicity of $u$ is inherited from $G$. (This is what one gets by factorizing $-\frac{d^2}{dx^2} + 1 = -(\frac{d}{dx} + 1)(\frac{d}{dx} - 1)$, solving the two first-order equations with integrating factors $e^{\mp x}$, and then adding $c_1 e^x + c_2 e^{-x}$ to enforce periodicity.)

**Compact self-adjoint.** The kernel $(x, y) \mapsto G(x - y)$ is continuous, real, and symmetric (because $G(s) = G(2\pi - s)$), so $S$ is a compact self-adjoint operator on $L^2$ (Theorem on integral operators with continuous kernels, and Example 188).

**Eigenvalues.** $e^{ikx}/(1+k^2)$ is a periodic $C^2$ solution of $-u'' + u = e^{ikx}$, so by uniqueness it equals $S(e^{ikx})$. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Square Root $F = S^{1/2}$)</span></p>

Define $F$ on $L^2(\mathbb{S})$ diagonally in the Fourier basis:

$$F\Big(\sum_{k \in \mathbb{Z}} c_k e^{ikx}\Big) = \sum_{k \in \mathbb{Z}} \frac{c_k}{\sqrt{1 + k^2}}\,e^{ikx}.$$

Then $F$ is compact, self-adjoint, injective, $F^2 = S$, and $F$ maps $L^2(\mathbb{S})$ boundedly into $C^0(\mathbb{S})$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Compactness: the truncations to $\lvert k \rvert \le n$ are finite rank and converge to $F$ in norm (the error has norm $(1 + n^2)^{-1/2}$), so Theorem 210 applies. Self-adjointness and injectivity are clear from the real, nonzero diagonal entries, and $F^2 = S$ because both are diagonal in the same basis with entries $(1+k^2)^{-1}$. Finally, by Cauchy–Schwarz,

$$\sum_k \frac{\lvert c_k \rvert}{\sqrt{1 + k^2}} \le \Big(\sum_k \frac{1}{1 + k^2}\Big)^{1/2}\Big(\sum_k \lvert c_k \rvert^2\Big)^{1/2},$$

so the series for $Fv$ converges uniformly (Weierstrass M-test) and $\lVert Fv \rVert\_\infty \le C\lVert v \rVert\_2$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fredholm Alternative for the Periodic Schrödinger Operator)</span></p>

Let $V$ be real-valued, continuous, and $2\pi$-periodic. Then exactly one of the following holds:

1. For every $f \in C^0(\mathbb{S})$, the equation $-u'' + Vu = f$ has a **unique** $2\pi$-periodic $C^2$ solution.
2. The space $N$ of $2\pi$-periodic $C^2$ solutions of the homogeneous equation $-w'' + Vw = 0$ is **finite-dimensional and nonzero**, and $-u'' + Vu = f$ has a periodic $C^2$ solution if and only if
   $$\int_0^{2\pi} f\,\overline{w}\,dx = 0 \qquad \text{for all } w \in N.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>proof</summary>

Let $m$ be multiplication by $V - 1$ (bounded, self-adjoint since $V$ is real), and set

$$T = F\,m\,F,$$

a compact (compact $\circ$ bounded $\circ$ bounded) self-adjoint operator on $L^2(\mathbb{S})$.

**Step 1 (ODE $\Rightarrow$ integral equation).** If $u$ is a periodic $C^2$ solution, then $-u'' + u = f - (V-1)u$, so by uniqueness in the Proposition, $u = S(f - (V-1)u) = F\big(Ff - F(V-1)u\big)$. Thus $u = Fv$ with $v := Ff - F m u$, and substituting $u = Fv$ gives

$$v + Tv = Ff.$$

**Step 2 (integral equation $\Rightarrow$ ODE).** Conversely, let $v \in L^2$ solve $v + Tv = Ff$ and put $u = Fv$, which is continuous since $F : L^2 \to C^0$. Applying $F$ gives $u + S((V-1)u) = Sf$, i.e. $u = S\big(f - (V-1)u\big)$. The argument $f - (V-1)u$ is continuous, so by the Proposition $u$ is periodic, $C^2$, and $-u'' + u = f - (V-1)u$, i.e. $-u'' + Vu = f$.

**Step 3 (the homogeneous equation).** Taking $f = 0$ in Steps 1–2, $v \mapsto Fv$ is a bijection $\operatorname{Null}(I + T) \to N$ (injective because $F$ is). By Theorem 227 applied to the eigenvalue $-1$ of $T$, $\dim N = \dim \operatorname{Null}(I + T) < \infty$.

**Step 4 (Fredholm alternative).** By Theorem 228 (with $\lambda = -1$), $\operatorname{Range}(I + T) = \operatorname{Null}(I + T)^\perp$.

* If $N = \lbrace 0 \rbrace$, then $I + T$ is bijective, so $v + Tv = Ff$ is solvable for every $f$ and Step 2 gives a solution $u$; two solutions differ by an element of $N = \lbrace 0 \rbrace$.
* Otherwise, by Steps 1–2 the ODE is solvable iff $Ff \in \operatorname{Range}(I + T)$, iff $\langle Ff, v \rangle = 0$ for all $v \in \operatorname{Null}(I + T)$. Since $F$ is self-adjoint, $\langle Ff, v \rangle = \langle f, Fv \rangle$, and $Fv$ runs over all of $N$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Shift by $1$, and Eigenfunctions)</span></p>

* There is nothing special about adding $1$: any positive constant works. Adding $0$ fails because constants solve $-u'' = 0$, so $-\frac{d^2}{dx^2}$ has no inverse on periodic functions.
* Unlike the Dirichlet case with $V \ge 0$, here $V$ is an arbitrary real potential, so alternative 2 genuinely occurs (e.g. $V = 0$, where $N$ = constants and the condition is $\int_0^{2\pi} f = 0$).
* When $N = \lbrace 0 \rbrace$, the operator $R = F(I + T)^{-1}F$ is a compact self-adjoint inverse of $Q = -\frac{d^2}{dx^2} + V$, and the $C^2$ periodic eigenfunctions $Qu = \tau u$ are exactly the solutions of $Ru = \tau^{-1}u$. The spectral theorem for $R$ then yields an orthonormal basis of $L^2(\mathbb{S})$ consisting of eigenfunctions of the Schrödinger operator.

</div>
