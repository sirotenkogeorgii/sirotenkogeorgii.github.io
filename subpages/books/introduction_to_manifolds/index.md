---
layout: default
title: An Introduction to Manifolds
date: 2025-03-11
excerpt: Notes on Loring Tu's "An Introduction to Manifolds" — smooth functions, tangent vectors, differential forms, and the foundations of manifold theory.
tags:
  - differential-geometry
  - manifolds
  - mathematics
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

# An Introduction to Manifolds

# Chapter 1 — Euclidean Spaces

The Euclidean space $\mathbb{R}^n$ is the prototype of all manifolds. Not only is it the simplest, but locally every manifold looks like $\mathbb{R}^n$. Euclidean space is special in having a set of standard global coordinates — this is both a blessing (all constructions can be carried out explicitly) and a handicap (it is often not obvious which concepts are intrinsic, i.e., independent of coordinates). Since a manifold in general does not have standard coordinates, only coordinate-independent concepts will make sense on a manifold.

The goal of this chapter is to recast calculus on $\mathbb{R}^n$ in a coordinate-free way suitable for generalization to manifolds: tangent vectors as derivations on functions, and differential forms via alternating multilinear functions on a vector space.

## §1 Smooth Functions on a Euclidean Space

### 1.1 $C^\infty$ Versus Analytic Functions

Write the coordinates on $\mathbb{R}^n$ as $x^1, \dots, x^n$ and let $p = (p^1, \dots, p^n)$ be a point in an open set $U$ in $\mathbb{R}^n$. In keeping with the conventions of differential geometry, the indices on coordinates are *superscripts*, not subscripts.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.1</span><span class="math-callout__name">($C^k$ and $C^\infty$ Functions)</span></p>

Let $k$ be a nonnegative integer. A real-valued function $f \colon U \to \mathbb{R}$ is said to be $C^k$ at $p \in U$ if its partial derivatives

$$\frac{\partial^j f}{\partial x^{i_1} \cdots \partial x^{i_j}}$$

of all orders $j \le k$ exist and are continuous at $p$. The function $f$ is $C^\infty$ at $p$ if it is $C^k$ for all $k \ge 0$; in other words, its partial derivatives of all orders exist and are continuous at $p$.

A vector-valued function $f \colon U \to \mathbb{R}^m$ is $C^k$ at $p$ if all of its component functions $f^1, \dots, f^m$ are $C^k$ at $p$. We say that $f \colon U \to \mathbb{R}^m$ is $C^k$ *on* $U$ if it is $C^k$ at every point in $U$. We treat the terms "$C^\infty$" and "smooth" as synonymous.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2</span></p>

**(i)** A $C^0$ function on $U$ is simply a continuous function on $U$.

**(ii)** Let $f \colon \mathbb{R} \to \mathbb{R}$ be $f(x) = x^{1/3}$. Then

$$f'(x) = \begin{cases} \frac{1}{3} x^{-2/3} & \text{for } x \neq 0, \\ \text{undefined} & \text{for } x = 0. \end{cases}$$

Thus $f$ is $C^0$ but not $C^1$ at $x = 0$.

**(iii)** Let $g \colon \mathbb{R} \to \mathbb{R}$ be defined by

$$g(x) = \int_0^x f(t)\,dt = \int_0^x t^{1/3}\,dt = \frac{3}{4} x^{4/3}.$$

Then $g'(x) = f(x) = x^{1/3}$, so $g(x)$ is $C^1$ but not $C^2$ at $x = 0$. In the same way one can construct a function that is $C^k$ but not $C^{k+1}$ at a given point.

**(iv)** The polynomial, sine, cosine, and exponential functions on the real line are all $C^\infty$.

</div>

A **neighborhood** of a point in $\mathbb{R}^n$ is an open set containing the point. The function $f$ is **real-analytic** at $p$ if in some neighborhood of $p$ it is equal to its Taylor series at $p$:

$$f(x) = f(p) + \sum_i \frac{\partial f}{\partial x^i}(p)(x^i - p^i) + \frac{1}{2!} \sum_{i,j} \frac{\partial^2 f}{\partial x^i \partial x^j}(p)(x^i - p^i)(x^j - p^j) + \cdots$$

A real-analytic function is necessarily $C^\infty$, because a convergent power series can be differentiated term by term. However, a $C^\infty$ function need not be real-analytic.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3</span><span class="math-callout__name">(A $C^\infty$ function very flat at $0$)</span></p>

Define $f(x)$ on $\mathbb{R}$ by

$$f(x) = \begin{cases} e^{-1/x} & \text{for } x > 0, \\ 0 & \text{for } x \le 0. \end{cases}$$

By induction, one can show that $f$ is $C^\infty$ on $\mathbb{R}$ and that the derivatives $f^{(k)}(0)$ are equal to $0$ for all $k \ge 0$. The Taylor series of this function at the origin is identically zero in any neighborhood of the origin, since all derivatives $f^{(k)}(0)$ equal $0$. Therefore, $f(x)$ cannot be equal to its Taylor series and $f(x)$ is not real-analytic at $0$.

</div>

### 1.2 Taylor's Theorem with Remainder

Although a $C^\infty$ function need not be equal to its Taylor series, there is a Taylor's theorem with remainder for $C^\infty$ functions that is often good enough for our purposes.

We say that a subset $S$ of $\mathbb{R}^n$ is **star-shaped** with respect to a point $p$ in $S$ if for every $x$ in $S$, the line segment from $p$ to $x$ lies in $S$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1.4</span><span class="math-callout__name">(Taylor's theorem with remainder)</span></p>

Let $f$ be a $C^\infty$ function on an open subset $U$ of $\mathbb{R}^n$ star-shaped with respect to a point $p = (p^1, \dots, p^n)$ in $U$. Then there are functions $g_1(x), \dots, g_n(x) \in C^\infty(U)$ such that

$$f(x) = f(p) + \sum_{i=1}^n (x^i - p^i)\, g_i(x), \quad g_i(p) = \frac{\partial f}{\partial x^i}(p).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since $U$ is star-shaped with respect to $p$, for any $x$ in $U$ the line segment $p + t(x - p)$, $0 \le t \le 1$, lies in $U$. So $f(p + t(x - p))$ is defined for $0 \le t \le 1$. By the chain rule,

$$\frac{d}{dt} f(p + t(x - p)) = \sum_i (x^i - p^i) \frac{\partial f}{\partial x^i}(p + t(x - p)).$$

Integrating both sides with respect to $t$ from $0$ to $1$:

$$f(x) - f(p) = \sum_i (x^i - p^i) \int_0^1 \frac{\partial f}{\partial x^i}(p + t(x - p))\,dt.$$

Let

$$g_i(x) = \int_0^1 \frac{\partial f}{\partial x^i}(p + t(x - p))\,dt.$$

Then $g_i(x)$ is $C^\infty$ and $g_i(p) = \int_0^1 \frac{\partial f}{\partial x^i}(p)\,dt = \frac{\partial f}{\partial x^i}(p)$.

</details>
</div>

In case $n = 1$ and $p = 0$, the lemma says that $f(x) = f(0) + x\,g_1(x)$ for some $C^\infty$ function $g_1(x)$. Applying the lemma repeatedly gives

$$f(x) = f(0) + g_1(0)\,x + g_2(0)\,x^2 + \cdots + g_i(0)\,x^i + g_{i+1}(x)\,x^{i+1},$$

where $g_k(0) = \frac{1}{k!} f^{(k)}(0)$ for $k = 1, 2, \dots, i$. So this is a polynomial expansion of $f(x)$ whose terms up to the last term agree with the Taylor series of $f(x)$ at $0$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Being star-shaped is not such a restrictive condition, since any open ball $B(p, \varepsilon) = \lbrace x \in \mathbb{R}^n \mid \|x - p\| < \varepsilon \rbrace$ is star-shaped with respect to $p$. If $f$ is a $C^\infty$ function defined on an open set $U$ containing $p$, then there is an $\varepsilon > 0$ such that $p \in B(p, \varepsilon) \subset U$. When its domain is restricted to $B(p, \varepsilon)$, the function $f$ is defined on a star-shaped neighborhood of $p$ and Taylor's theorem with remainder applies.

</div>

**Notation.** It is customary to write the standard coordinates on $\mathbb{R}^2$ as $x, y$, and the standard coordinates on $\mathbb{R}^3$ as $x, y, z$.

## §2 Tangent Vectors in $\mathbb{R}^n$ as Derivations

In elementary calculus we normally represent a vector at a point $p$ in $\mathbb{R}^3$ algebraically as a column of numbers or geometrically as an arrow emanating from $p$. Such a definition presupposes that the surface is embedded in a Euclidean space. Our goal in this section is to find a characterization of tangent vectors in $\mathbb{R}^n$ that will generalize to manifolds.

### 2.1 The Directional Derivative

In calculus we visualize the tangent space $T_p(\mathbb{R}^n)$ at $p$ in $\mathbb{R}^n$ as the vector space of all arrows emanating from $p$. By the correspondence between arrows and column vectors, the vector space $\mathbb{R}^n$ can be identified with this column space. To distinguish between points and vectors, we write a point in $\mathbb{R}^n$ as $p = (p^1, \dots, p^n)$ and a vector in the tangent space $T_p(\mathbb{R}^n)$ as

$$v = \begin{bmatrix} v^1 \\ \vdots \\ v^n \end{bmatrix} \quad \text{or} \quad \langle v^1, \dots, v^n \rangle.$$

We usually denote the standard basis for $\mathbb{R}^n$ or $T_p(\mathbb{R}^n)$ by $e_1, \dots, e_n$. Then $v = \sum v^i e_i$ for some $v^i \in \mathbb{R}$. Elements of $T_p(\mathbb{R}^n)$ are called **tangent vectors** (or simply **vectors**) at $p$ in $\mathbb{R}^n$.

The line through a point $p = (p^1, \dots, p^n)$ with direction $v = \langle v^1, \dots, v^n \rangle$ in $\mathbb{R}^n$ has parametrization $c(t) = (p^1 + tv^1, \dots, p^n + tv^n)$. If $f$ is $C^\infty$ in a neighborhood of $p$ in $\mathbb{R}^n$ and $v$ is a tangent vector at $p$, the **directional derivative** of $f$ in the direction $v$ at $p$ is defined to be

$$D_v f = \lim_{t \to 0} \frac{f(c(t)) - f(p)}{t} = \frac{d}{dt}\bigg|_{t=0} f(c(t)).$$

By the chain rule,

$$D_v f = \sum_{i=1}^n \frac{dc^i}{dt}(0) \frac{\partial f}{\partial x^i}(p) = \sum_{i=1}^n v^i \frac{\partial f}{\partial x^i}(p).$$

In the notation $D_v f$, it is understood that the partial derivatives are to be evaluated at $p$, since $v$ is a vector at $p$. So $D_v f$ is a number, not a function. We write

$$D_v = \sum v^i \left.\frac{\partial}{\partial x^i}\right|_p$$

for the map that sends a function $f$ to the number $D_v f$.

The association $v \mapsto D_v$ of the directional derivative $D_v$ to a tangent vector $v$ offers a way to characterize tangent vectors as certain operators on functions.

### 2.2 Germs of Functions

A **relation** on a set $S$ is a subset $R$ of $S \times S$. Given $x, y$ in $S$, we write $x \sim y$ if and only if $(x, y) \in R$. The relation $R$ is an **equivalence relation** if it satisfies reflexivity ($x \sim x$), symmetry ($x \sim y \Rightarrow y \sim x$), and transitivity ($x \sim y,\, y \sim z \Rightarrow x \sim z$) for all $x, y, z \in S$.

As long as two functions agree on some neighborhood of a point $p$, they will have the same directional derivatives at $p$. This suggests that we introduce an equivalence relation on the $C^\infty$ functions defined in some neighborhood of $p$. Consider the set of all pairs $(f, U)$, where $U$ is a neighborhood of $p$ and $f \colon U \to \mathbb{R}$ is a $C^\infty$ function. We say that $(f, U)$ is *equivalent* to $(g, V)$ if there is an open set $W \subset U \cap V$ containing $p$ such that $f = g$ when restricted to $W$. The equivalence class of $(f, U)$ is called the **germ** of $f$ at $p$. We write $C_p^\infty(\mathbb{R}^n)$, or simply $C_p^\infty$, for the set of all germs of $C^\infty$ functions on $\mathbb{R}^n$ at $p$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

The functions $f(x) = \frac{1}{1-x}$ with domain $\mathbb{R} - \lbrace 1 \rbrace$ and $g(x) = 1 + x + x^2 + x^3 + \cdots$ with domain the open interval $]-1, 1[$ have the same germ at any point $p$ in the open interval $]-1, 1[$.

</div>

An **algebra** over a field $K$ is a vector space $A$ over $K$ with a multiplication map $\mu \colon A \times A \to A$, usually written $\mu(a, b) = a \cdot b$, satisfying associativity, distributivity, and homogeneity. Equivalently, an algebra over a field $K$ is a ring $A$ (with or without multiplicative identity) that is also a vector space over $K$ such that the ring multiplication satisfies the homogeneity condition. The addition and multiplication of functions induce corresponding operations on $C_p^\infty$, making it into an algebra over $\mathbb{R}$.

A map $L \colon V \to W$ between vector spaces over a field $K$ is called a **linear map** or a **linear operator** if for any $r \in K$ and $u, v \in V$:
- $L(u + v) = L(u) + L(v)$,
- $L(rv) = rL(v)$.

If $A$ and $A'$ are algebras over a field $K$, then an **algebra homomorphism** is a linear map $L \colon A \to A'$ that preserves the algebra multiplication: $L(ab) = L(a)L(b)$ for all $a, b \in A$.

### 2.3 Derivations at a Point

For each tangent vector $v$ at a point $p$ in $\mathbb{R}^n$, the directional derivative at $p$ gives a map of real vector spaces $D_v \colon C_p^\infty \to \mathbb{R}$. By the formula for the directional derivative, $D_v$ is $\mathbb{R}$-linear and satisfies the **Leibniz rule**

$$D_v(fg) = (D_v f)\,g(p) + f(p)\,D_v g,$$

precisely because the partial derivatives $\partial / \partial x^i |_p$ have these properties.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Derivation at a Point)</span></p>

In general, any linear map $D \colon C_p^\infty \to \mathbb{R}$ satisfying the Leibniz rule

$$D(fg) = (Df)\,g(p) + f(p)\,Dg$$

is called a **derivation at $p$** or a **point-derivation** of $C_p^\infty$. Denote the set of all derivations at $p$ by $\mathcal{D}_p(\mathbb{R}^n)$. This set is in fact a real vector space, since the sum of two derivations at $p$ and a scalar multiple of a derivation at $p$ are again derivations at $p$.

</div>

Thus far, we know that directional derivatives at $p$ are all derivations at $p$, so there is a map

$$\phi \colon T_p(\mathbb{R}^n) \to \mathcal{D}_p(\mathbb{R}^n), \quad v \mapsto D_v = \sum v^i \left.\frac{\partial}{\partial x^i}\right|_p.$$

Since $\phi$ is clearly linear in $v$, the map $\phi$ is a linear map of vector spaces.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.1</span></p>

If $D$ is a point-derivation of $C_p^\infty$, then $D(c) = 0$ for any constant function $c$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since we do not know whether every derivation at $p$ is a directional derivative, we need to prove this using only the defining properties of a derivation at $p$. By $\mathbb{R}$-linearity, $D(c) = c\,D(1)$. So it suffices to prove that $D(1) = 0$. By the Leibniz rule,

$$D(1) = D(1 \cdot 1) = D(1) \cdot 1 + 1 \cdot D(1) = 2\,D(1).$$

Subtracting $D(1)$ from both sides gives $0 = D(1)$.

</details>
</div>

The **Kronecker delta** $\delta_j^i$ is a useful notation that we frequently call upon:

$$\delta_j^i = \begin{cases} 1 & \text{if } i = j, \\ 0 & \text{if } i \neq j. \end{cases}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2</span></p>

The linear map $\phi \colon T_p(\mathbb{R}^n) \to \mathcal{D}_p(\mathbb{R}^n)$ defined by $v \mapsto D_v$ is an isomorphism of vector spaces.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**Injectivity.** Suppose $D_v = 0$ for $v \in T_p(\mathbb{R}^n)$. Applying $D_v$ to the coordinate function $x^j$ gives

$$0 = D_v(x^j) = \sum_i v^i \left.\frac{\partial}{\partial x^i}\right|_p x^j = \sum_i v^i \delta_j^i = v^j.$$

Hence $v = 0$ and $\phi$ is injective.

**Surjectivity.** Let $D$ be a derivation at $p$ and let $(f, V)$ be a representative of a germ in $C_p^\infty$. Making $V$ smaller if necessary, we may assume that $V$ is an open ball, hence star-shaped. By Taylor's theorem with remainder (Lemma 1.4), there are $C^\infty$ functions $g_i(x)$ in a neighborhood of $p$ such that

$$f(x) = f(p) + \sum (x^i - p^i)\,g_i(x), \quad g_i(p) = \frac{\partial f}{\partial x^i}(p).$$

Applying $D$ to both sides and noting that $D(f(p)) = 0$ and $D(p^i) = 0$ by Lemma 2.1, we get by the Leibniz rule

$$Df(x) = \sum (Dx^i)\,g_i(p) + \sum (p^i - p^i)\,Dg_i(x) = \sum (Dx^i) \frac{\partial f}{\partial x^i}(p).$$

This proves that $D = D_v$ for $v = \langle Dx^1, \dots, Dx^n \rangle$.

</details>
</div>

This theorem shows that one may identify the tangent vectors at $p$ with the derivations at $p$. Under the vector space isomorphism $T_p(\mathbb{R}^n) \simeq \mathcal{D}_p(\mathbb{R}^n)$, the standard basis $e_1, \dots, e_n$ for $T_p(\mathbb{R}^n)$ corresponds to the set $\lbrace \partial/\partial x^1|_p, \dots, \partial/\partial x^n|_p \rbrace$ of partial derivatives. From now on, we will make this identification and write a tangent vector $v = \langle v^1, \dots, v^n \rangle = \sum v^i e_i$ as

$$v = \sum v^i \left.\frac{\partial}{\partial x^i}\right|_p.$$

The vector space $\mathcal{D}_p(\mathbb{R}^n)$ of derivations at $p$, although not as geometric as arrows, turns out to be more suitable for generalization to manifolds.

### 2.4 Vector Fields

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Field)</span></p>

A **vector field** $X$ on an open subset $U$ of $\mathbb{R}^n$ is a function that assigns to each point $p$ in $U$ a tangent vector $X_p$ in $T_p(\mathbb{R}^n)$. Since $T_p(\mathbb{R}^n)$ has basis $\lbrace \partial/\partial x^i|_p \rbrace$, the vector $X_p$ is a linear combination

$$X_p = \sum a^i(p) \left.\frac{\partial}{\partial x^i}\right|_p, \quad p \in U, \quad a^i(p) \in \mathbb{R}.$$

Omitting $p$, we may write $X = \sum a^i \,\partial/\partial x^i$, where the $a^i$ are now functions on $U$. We say that the vector field $X$ is $C^\infty$ *on* $U$ if the coefficient functions $a^i$ are all $C^\infty$ on $U$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.3</span></p>

On $\mathbb{R}^2 - \lbrace \mathbf{0} \rbrace$, let $p = (x, y)$. Then

$$X = \frac{-y}{\sqrt{x^2 + y^2}} \frac{\partial}{\partial x} + \frac{x}{\sqrt{x^2 + y^2}} \frac{\partial}{\partial y} = \left\langle \frac{-y}{\sqrt{x^2 + y^2}},\, \frac{x}{\sqrt{x^2 + y^2}} \right\rangle$$

is a vector field on $\mathbb{R}^2 - \lbrace \mathbf{0} \rbrace$. The vector field $Y = x\,\partial/\partial x - y\,\partial/\partial y = \langle x, -y \rangle$ is a vector field on $\mathbb{R}^2$.

</div>

One can identify vector fields on $U$ with column vectors of $C^\infty$ functions on $U$:

$$X = \sum a^i \frac{\partial}{\partial x^i} \quad \longleftrightarrow \quad \begin{bmatrix} a^1 \\ \vdots \\ a^n \end{bmatrix}.$$

This is the same identification as before, but now we are allowing the point $p$ to move in $U$.

The ring of $C^\infty$ functions on an open set $U$ is commonly denoted by $C^\infty(U)$ or $\mathcal{F}(U)$. Multiplication of vector fields by functions on $U$ is defined pointwise:

$$(fX)_p = f(p)\,X_p, \quad p \in U.$$

If $X = \sum a^i \,\partial/\partial x^i$ is a $C^\infty$ vector field and $f$ is a $C^\infty$ function on $U$, then $fX = \sum (fa^i)\,\partial/\partial x^i$ is a $C^\infty$ vector field on $U$. Thus, the set of all $C^\infty$ vector fields on $U$, denoted by $\mathfrak{X}(U)$, is not only a vector space over $\mathbb{R}$, but also a **module** over the ring $C^\infty(U)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.4</span><span class="math-callout__name">($R$-module)</span></p>

If $R$ is a commutative ring with identity, then a (left) **$R$-module** is an abelian group $A$ with a scalar multiplication map $\mu \colon R \times A \to A$, usually written $\mu(r, a) = ra$, such that for all $r, s \in R$ and $a, b \in A$:

- (associativity) $(rs)a = r(sa)$,
- (identity) if $1$ is the multiplicative identity in $R$, then $1a = a$,
- (distributivity) $(r + s)a = ra + sa$, $\;r(a + b) = ra + rb$.

If $R$ is a field, then an $R$-module is precisely a vector space over $R$. In this sense, a module generalizes a vector space by allowing scalars in a ring rather than a field.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.5</span><span class="math-callout__name">($R$-module Homomorphism)</span></p>

Let $A$ and $A'$ be $R$-modules. An **$R$-module homomorphism** from $A$ to $A'$ is a map $f \colon A \to A'$ that preserves both addition and scalar multiplication: for all $a, b \in A$ and $r \in R$,

- $f(a + b) = f(a) + f(b)$,
- $f(ra) = r\,f(a)$.

</div>

### 2.5 Vector Fields as Derivations

If $X$ is a $C^\infty$ vector field on an open subset $U$ of $\mathbb{R}^n$ and $f$ is a $C^\infty$ function on $U$, we define a new function $Xf$ on $U$ by

$$(Xf)(p) = X_p f \quad \text{for any } p \in U.$$

Writing $X = \sum a^i \,\partial/\partial x^i$, we get $Xf = \sum a^i \,\partial f/\partial x^i$, which shows that $Xf$ is a $C^\infty$ function on $U$. Thus, a $C^\infty$ vector field $X$ gives rise to an $\mathbb{R}$-linear map

$$C^\infty(U) \to C^\infty(U), \quad f \mapsto Xf.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.6</span><span class="math-callout__name">(Leibniz rule for a vector field)</span></p>

If $X$ is a $C^\infty$ vector field and $f$ and $g$ are $C^\infty$ functions on an open subset $U$ of $\mathbb{R}^n$, then $X(fg)$ satisfies the product rule (Leibniz rule):

$$X(fg) = (Xf)\,g + f\,Xg.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

At each point $p \in U$, the vector $X_p$ satisfies the Leibniz rule:

$$X_p(fg) = (X_p f)\,g(p) + f(p)\,X_p g.$$

As $p$ varies over $U$, this becomes an equality of functions: $X(fg) = (Xf)\,g + f\,Xg$.

</details>
</div>

If $A$ is an algebra over a field $K$, a **derivation** of $A$ is a $K$-linear map $D \colon A \to A$ such that $D(ab) = (Da)b + a\,Db$ for all $a, b \in A$.

The set of all derivations of $A$ is closed under addition and scalar multiplication and forms a vector space, denoted by $\operatorname{Der}(A)$. As noted above, a $C^\infty$ vector field on an open set $U$ gives rise to a derivation of the algebra $C^\infty(U)$. We therefore have a map

$$\varphi \colon \mathfrak{X}(U) \to \operatorname{Der}(C^\infty(U)), \quad X \mapsto (f \mapsto Xf).$$

Just as the tangent vectors at a point $p$ can be identified with the point-derivations of $C_p^\infty$, so the vector fields on an open set $U$ can be identified with the derivations of the algebra $C^\infty(U)$; i.e., the map $\varphi$ is an isomorphism of vector spaces.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Note that a derivation at $p$ is not a derivation of the algebra $C_p^\infty$. A derivation at $p$ is a map from $C_p^\infty$ to $\mathbb{R}$, while a derivation of the algebra $C_p^\infty$ is a map from $C_p^\infty$ to $C_p^\infty$.

</div>

## §3 The Exterior Algebra of Multicovectors

A basic principle in manifold theory is the linearization principle: every manifold can be locally approximated by its tangent space at a point. Instead of working with tangent vectors directly, it turns out to be more fruitful to adopt the dual point of view and work with linear functions on a tangent space. Once one admits linear functions, it is a small step to consider multilinear functions — functions of several arguments, linear in each. Among multilinear functions, certain ones such as the determinant and the cross product have an *antisymmetric* or *alternating* property: they change sign if two arguments are switched. The alternating multilinear functions with $k$ arguments on a vector space are called **multicovectors of degree $k$**, or **$k$-covectors** for short.

### 3.1 Dual Space

If $V$ and $W$ are real vector spaces, we denote by $\operatorname{Hom}(V, W)$ the vector space of all linear maps $f \colon V \to W$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dual Space)</span></p>

The **dual space** $V^\vee$ of $V$ is the vector space of all real-valued linear functions on $V$:

$$V^\vee = \operatorname{Hom}(V, \mathbb{R}).$$

The elements of $V^\vee$ are called **covectors** or **1-covectors** on $V$.

</div>

Assume $V$ to be a finite-dimensional vector space. Let $e_1, \dots, e_n$ be a basis for $V$. Then every $v$ in $V$ is uniquely a linear combination $v = \sum v^i e_i$ with $v^i \in \mathbb{R}$. Let $\alpha^i \colon V \to \mathbb{R}$ be the linear function that picks out the $i$th coordinate, $\alpha^i(v) = v^i$. Note that $\alpha^i$ is characterized by

$$\alpha^i(e_j) = \delta_j^i = \begin{cases} 1 & \text{for } i = j, \\ 0 & \text{for } i \neq j. \end{cases}$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.1</span></p>

The functions $\alpha^1, \dots, \alpha^n$ form a basis for $V^\vee$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

We first prove that $\alpha^1, \dots, \alpha^n$ span $V^\vee$. If $f \in V^\vee$ and $v = \sum v^i e_i \in V$, then

$$f(v) = \sum v^i f(e_i) = \sum f(e_i)\,\alpha^i(v).$$

Hence $f = \sum f(e_i)\,\alpha^i$, which shows that $\alpha^1, \dots, \alpha^n$ span $V^\vee$.

To show linear independence, suppose $\sum c_i \alpha^i = 0$ for some $c_i \in \mathbb{R}$. Applying both sides to the vector $e_j$ gives

$$0 = \sum_i c_i \alpha^i(e_j) = \sum_i c_i \delta_j^i = c_j, \quad j = 1, \dots, n.$$

Hence $\alpha^1, \dots, \alpha^n$ are linearly independent.

</details>
</div>

This basis $\alpha^1, \dots, \alpha^n$ for $V^\vee$ is said to be **dual** to the basis $e_1, \dots, e_n$ for $V$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.2</span></p>

The dual space $V^\vee$ of a finite-dimensional vector space $V$ has the same dimension as $V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.3</span><span class="math-callout__name">(Coordinate functions)</span></p>

With respect to a basis $e_1, \dots, e_n$ for a vector space $V$, every $v \in V$ can be written uniquely as a linear combination $v = \sum b^j(v)\,e_j$, where $b^j(v) \in \mathbb{R}$. Let $\alpha^1, \dots, \alpha^n$ be the basis of $V^\vee$ dual to $e_1, \dots, e_n$. Then

$$\alpha^i(v) = \alpha^i\!\left(\sum_j b^j(v)\,e_j\right) = \sum_j b^j(v)\,\alpha^i(e_j) = \sum_j b^j(v)\,\delta_j^i = b^i(v).$$

Thus, the dual basis to $e_1, \dots, e_n$ is precisely the set of coordinate functions $b^1, \dots, b^n$ with respect to the basis $e_1, \dots, e_n$.

</div>

### 3.2 Permutations

Fix a positive integer $k$. A **permutation** of the set $A = \lbrace 1, \dots, k \rbrace$ is a bijection $\sigma \colon A \to A$. The **cyclic permutation** $(a_1\; a_2 \;\cdots\; a_r)$, where the $a_i$ are distinct, is the permutation $\sigma$ such that $\sigma(a_1) = a_2$, $\sigma(a_2) = a_3$, $\dots$, $\sigma(a_{r-1}) = a_r$, $\sigma(a_r) = a_1$, and $\sigma$ fixes all other elements. It is also called a **cycle of length $r$** or an **$r$-cycle**. A **transposition** is a $2$-cycle $(a\; b)$ that interchanges $a$ and $b$.

Two cycles $(a_1 \cdots a_r)$ and $(b_1 \cdots b_s)$ are **disjoint** if $\lbrace a_1, \dots, a_r \rbrace$ and $\lbrace b_1, \dots, b_s \rbrace$ have no elements in common. The **product** $\tau\sigma$ of two permutations $\tau$ and $\sigma$ of $A$ is the composition $\tau \circ \sigma \colon A \to A$; first apply $\sigma$, then $\tau$. Any permutation can be written as a product of disjoint cycles.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.4</span></p>

The permutation $\sigma \colon \lbrace 1,2,3,4,5 \rbrace \to \lbrace 1,2,3,4,5 \rbrace$ mapping $1,2,3,4,5$ to $2,4,5,1,3$ in that order has matrix

$$\sigma = \begin{bmatrix} 1 & 2 & 3 & 4 & 5 \\ 2 & 4 & 5 & 1 & 3 \end{bmatrix}.$$

As a product of disjoint cycles, $\sigma = (1\;2\;4)(3\;5)$.

</div>

Let $S_k$ be the group of all permutations of the set $\lbrace 1, \dots, k \rbrace$. A permutation is **even** or **odd** depending on whether it is the product of an even or an odd number of transpositions. The **sign** of a permutation $\sigma$, denoted by $\operatorname{sgn}(\sigma)$ or $\operatorname{sgn}\sigma$, is defined to be $+1$ or $-1$ depending on whether the permutation is even or odd. The sign of a permutation satisfies

$$\operatorname{sgn}(\sigma\tau) = \operatorname{sgn}(\sigma)\operatorname{sgn}(\tau)$$

for $\sigma, \tau \in S_k$.

An **inversion** in a permutation $\sigma$ is an ordered pair $(\sigma(i), \sigma(j))$ such that $i < j$ but $\sigma(i) > \sigma(j)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.8</span></p>

A permutation is even if and only if it has an even number of inversions.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

We will obtain the identity permutation $\mathbb{1}$ by multiplying $\sigma$ on the left by a number of transpositions. This can be achieved in $k$ steps:

**(i)** First, look for the number $1$ among $\sigma(1), \sigma(2), \dots, \sigma(k)$. Every number preceding $1$ in this list gives rise to an inversion. If $1 = \sigma(i)$, then $(\sigma(1), 1), \dots, (\sigma(i-1), 1)$ are inversions of $\sigma$. Now move $1$ to the beginning of the list across the $i - 1$ elements $\sigma(1), \dots, \sigma(i-1)$. This requires multiplying $\sigma$ on the left by $i - 1$ transpositions. Note that the number of transpositions is the number of inversions ending in $1$.

**(ii)** Next look for the number $2$ in the list $1, \sigma(1), \dots, \sigma(i-1), \sigma(i+1), \dots, \sigma(k)$. Every number other than $1$ preceding $2$ in this list gives rise to an inversion $(\sigma(m), 2)$. Suppose there are $i_2$ such numbers. Then there are $i_2$ inversions ending in $2$. In moving $2$ to its natural position, we multiply by $i_2$ transpositions.

Repeating this procedure, we see that for each $j = 1, \dots, k$, the number of transpositions required to move $j$ to its natural position is the same as the number of inversions ending in $j$. In the end we achieve the identity permutation. Therefore, $\operatorname{sgn}(\sigma) = (-1)^{\#\text{inversions in } \sigma}$.

</details>
</div>

### 3.3 Multilinear Functions

Denote by $V^k = V \times \cdots \times V$ the Cartesian product of $k$ copies of a real vector space $V$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($k$-linear Function)</span></p>

A function $f \colon V^k \to \mathbb{R}$ is **$k$-linear** if it is linear in each of its $k$ arguments:

$$f(\dots, av + bw, \dots) = a\,f(\dots, v, \dots) + b\,f(\dots, w, \dots)$$

for all $a, b \in \mathbb{R}$ and $v, w \in V$. Instead of $2$-linear and $3$-linear, it is customary to say "bilinear" and "trilinear." A $k$-linear function on $V$ is also called a **$k$-tensor** on $V$. We will denote the vector space of all $k$-tensors on $V$ by $L_k(V)$. If $f$ is a $k$-tensor on $V$, we also call $k$ the **degree** of $f$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.9</span><span class="math-callout__name">(Dot product on $\mathbb{R}^n$)</span></p>

With respect to the standard basis $e_1, \dots, e_n$ for $\mathbb{R}^n$, the **dot product**, defined by

$$f(v, w) = v \bullet w = \sum_i v^i w^i, \quad \text{where } v = \sum v^i e_i,\; w = \sum w^i e_i,$$

is bilinear. The determinant $f(v_1, \dots, v_n) = \det[v_1 \;\cdots\; v_n]$, viewed as a function of the $n$ column vectors $v_1, \dots, v_n$ in $\mathbb{R}^n$, is $n$-linear.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.10</span><span class="math-callout__name">(Symmetric and Alternating)</span></p>

A $k$-linear function $f \colon V^k \to \mathbb{R}$ is **symmetric** if

$$f\!\left(v_{\sigma(1)}, \dots, v_{\sigma(k)}\right) = f(v_1, \dots, v_k)$$

for all permutations $\sigma \in S_k$; it is **alternating** if

$$f\!\left(v_{\sigma(1)}, \dots, v_{\sigma(k)}\right) = (\operatorname{sgn}\sigma)\,f(v_1, \dots, v_k)$$

for all $\sigma \in S_k$.

</div>

**Examples.**
- The dot product $f(v, w) = v \bullet w$ on $\mathbb{R}^n$ is symmetric.
- The determinant $f(v_1, \dots, v_n) = \det[v_1 \;\cdots\; v_n]$ on $\mathbb{R}^n$ is alternating.
- The cross product $v \times w$ on $\mathbb{R}^3$ is alternating.
- For any two linear functions $f, g \colon V \to \mathbb{R}$ on a vector space $V$, the function $f \wedge g \colon V \times V \to \mathbb{R}$ defined by $(f \wedge g)(u, v) = f(u)\,g(v) - f(v)\,g(u)$ is alternating. This is a special case of the wedge product.

We are especially interested in the space $A_k(V)$ of all alternating $k$-linear functions on a vector space $V$ for $k > 0$. These are also called **alternating $k$-tensors**, **$k$-covectors**, or **multicovectors of degree $k$** on $V$. For $k = 0$, we define a $0$-covector to be a constant, so that $A_0(V)$ is the vector space $\mathbb{R}$. A $1$-covector is simply a covector.

### 3.4 The Permutation Action on Multilinear Functions

If $f$ is a $k$-linear function on a vector space $V$ and $\sigma$ is a permutation in $S_k$, we define a new $k$-linear function $\sigma f$ by

$$(\sigma f)(v_1, \dots, v_k) = f\!\left(v_{\sigma(1)}, \dots, v_{\sigma(k)}\right).$$

Thus, $f$ is symmetric if and only if $\sigma f = f$ for all $\sigma \in S_k$, and $f$ is alternating if and only if $\sigma f = (\operatorname{sgn}\sigma)\,f$ for all $\sigma \in S_k$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.11</span></p>

If $\sigma, \tau \in S_k$ and $f$ is a $k$-linear function on $V$, then $\tau(\sigma f) = (\tau\sigma)f$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For $v_1, \dots, v_k \in V$,

$$\tau(\sigma f)(v_1, \dots, v_k) = (\sigma f)(v_{\tau(1)}, \dots, v_{\tau(k)})$$

$$= f(v_{\tau(\sigma(1))}, \dots, v_{\tau(\sigma(k))}) = (\tau\sigma)f(v_1, \dots, v_k).$$

</details>
</div>

In general, if $G$ is a group and $X$ is a set, a map $G \times X \to X$, $(\sigma, x) \mapsto \sigma \cdot x$, is called a **left action** of $G$ on $X$ if $e \cdot x = x$ and $\tau \cdot (\sigma \cdot x) = (\tau\sigma) \cdot x$ for all $\tau, \sigma \in G$ and $x \in X$. The **orbit** of an element $x \in X$ is $Gx := \lbrace \sigma \cdot x \in X \mid \sigma \in G \rbrace$.

By Lemma 3.11, we have defined a left action of $S_k$ on the vector space $L_k(V)$ of $k$-linear functions on $V$. Note that each permutation acts as a linear function on $L_k(V)$ since $\sigma f$ is $\mathbb{R}$-linear in $f$.

### 3.5 The Symmetrizing and Alternating Operators

Given any $k$-linear function $f$ on a vector space $V$, there is a way to make a symmetric $k$-linear function $Sf$ from it:

$$(Sf)(v_1, \dots, v_k) = \sum_{\sigma \in S_k} f\!\left(v_{\sigma(1)}, \dots, v_{\sigma(k)}\right),$$

or in shorthand, $Sf = \sum_{\sigma \in S_k} \sigma f$.

Similarly, there is a way to make an alternating $k$-linear function from $f$. Define

$$Af = \sum_{\sigma \in S_k} (\operatorname{sgn}\sigma)\,\sigma f.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.12</span></p>

If $f$ is a $k$-linear function on a vector space $V$, then

- (i) the $k$-linear function $Sf$ is symmetric, and
- (ii) the $k$-linear function $Af$ is alternating.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

We prove (ii) only. For $\tau \in S_k$,

$$\tau(Af) = \sum_{\sigma \in S_k} (\operatorname{sgn}\sigma)\,\tau(\sigma f) = \sum_{\sigma \in S_k} (\operatorname{sgn}\sigma)\,(\tau\sigma)f$$

$$= (\operatorname{sgn}\tau) \sum_{\sigma \in S_k} (\operatorname{sgn}\tau\sigma)\,(\tau\sigma)f = (\operatorname{sgn}\tau)\,Af,$$

since as $\sigma$ runs through all permutations in $S_k$, so does $\tau\sigma$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.14</span></p>

If $f$ is an alternating $k$-linear function on a vector space $V$, then $Af = (k!)\,f$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since alternating $f$ we have $\sigma f = (\operatorname{sgn}\sigma)\,f$, and $\operatorname{sgn}\sigma$ is $\pm 1$, we must have

$$Af = \sum_{\sigma \in S_k} (\operatorname{sgn}\sigma)\,\sigma f = \sum_{\sigma \in S_k} (\operatorname{sgn}\sigma)(\operatorname{sgn}\sigma)\,f = (k!)\,f.$$

</details>
</div>

### 3.6 The Tensor Product

Let $f$ be a $k$-linear function and $g$ an $\ell$-linear function on a vector space $V$. Their **tensor product** is the $(k + \ell)$-linear function $f \otimes g$ defined by

$$(f \otimes g)(v_1, \dots, v_{k+\ell}) = f(v_1, \dots, v_k)\,g(v_{k+1}, \dots, v_{k+\ell}).$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.16</span><span class="math-callout__name">(Bilinear maps)</span></p>

Let $e_1, \dots, e_n$ be a basis for a vector space $V$, $\alpha^1, \dots, \alpha^n$ the dual basis in $V^\vee$, and $\langle\,,\,\rangle \colon V \times V \to \mathbb{R}$ a bilinear map on $V$. Set $g_{ij} = \langle e_i, e_j \rangle \in \mathbb{R}$. If $v = \sum v^i e_i$ and $w = \sum w^j e_j$, then by bilinearity,

$$\langle v, w \rangle = \sum v^i w^j g_{ij} = \sum g_{ij}\,\alpha^i(v)\,\alpha^j(w) = \sum g_{ij}\,(\alpha^i \otimes \alpha^j)(v, w).$$

Hence $\langle\,,\,\rangle = \sum g_{ij}\,\alpha^i \otimes \alpha^j$. This notation is often used in differential geometry to describe an inner product on a vector space.

</div>

The tensor product of multilinear functions is associative: $(f \otimes g) \otimes h = f \otimes (g \otimes h)$.

### 3.7 The Wedge Product

If two multilinear functions $f$ and $g$ on a vector space $V$ are alternating, then we would like to have a product that is alternating as well. This motivates the definition of the **wedge product**, also called the **exterior product**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Wedge Product)</span></p>

For $f \in A_k(V)$ and $g \in A_\ell(V)$, the **wedge product** $f \wedge g$ is defined by

$$f \wedge g = \frac{1}{k!\,\ell!}\,A(f \otimes g),$$

or explicitly,

$$(f \wedge g)(v_1, \dots, v_{k+\ell}) = \frac{1}{k!\,\ell!} \sum_{\sigma \in S_{k+\ell}} (\operatorname{sgn}\sigma)\,f\!\left(v_{\sigma(1)}, \dots, v_{\sigma(k)}\right) g\!\left(v_{\sigma(k+1)}, \dots, v_{\sigma(k+\ell)}\right).$$

By Proposition 3.12, $f \wedge g$ is alternating.

</div>

When $k = 0$, the element $f \in A_0(V)$ is simply a constant $c$. In this case, the wedge product $c \wedge g$ is scalar multiplication: $c \wedge g = cg$ for $c \in \mathbb{R}$ and $g \in A_\ell(V)$.

The coefficient $1/(k!\,\ell!)$ compensates for repetitions in the sum. One way to avoid redundancies is to stipulate that in the sum, $\sigma(1), \dots, \sigma(k)$ be in ascending order and $\sigma(k+1), \dots, \sigma(k+\ell)$ also be in ascending order. We call a permutation $\sigma \in S_{k+\ell}$ a *$(k, \ell)$-shuffle* if

$$\sigma(1) < \cdots < \sigma(k) \quad \text{and} \quad \sigma(k+1) < \cdots < \sigma(k+\ell).$$

Using shuffles, we may rewrite the wedge product as

$$(f \wedge g)(v_1, \dots, v_{k+\ell}) = \sum_{(k,\ell)\text{-shuffles } \sigma} (\operatorname{sgn}\sigma)\,f\!\left(v_{\sigma(1)}, \dots, v_{\sigma(k)}\right) g\!\left(v_{\sigma(k+1)}, \dots, v_{\sigma(k+\ell)}\right).$$

Written this way, the definition of $(f \wedge g)(v_1, \dots, v_{k+\ell})$ is a sum of $\binom{k+\ell}{k}$ terms, instead of $(k+\ell)!$ terms.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.19</span><span class="math-callout__name">(Wedge product of two covectors)</span></p>

If $f$ and $g$ are covectors on a vector space $V$ and $v_1, v_2 \in V$, then by the shuffle formula,

$$(f \wedge g)(v_1, v_2) = f(v_1)\,g(v_2) - f(v_2)\,g(v_1).$$

</div>

### 3.8 Anticommutativity of the Wedge Product

It follows directly from the definition of the wedge product that $f \wedge g$ is bilinear in $f$ and in $g$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.21</span><span class="math-callout__name">(Anticommutativity)</span></p>

The wedge product is **anticommutative**: if $f \in A_k(V)$ and $g \in A_\ell(V)$, then

$$f \wedge g = (-1)^{k\ell}\, g \wedge f.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Define $\tau \in S_{k+\ell}$ to be the permutation

$$\tau = \begin{bmatrix} 1 & \cdots & \ell & \ell+1 & \cdots & \ell+k \\ k+1 & \cdots & k+\ell & 1 & \cdots & k \end{bmatrix}.$$

Then $\operatorname{sgn}\tau = (-1)^{k\ell}$. Starting from the definition,

$$A(f \otimes g)(v_1, \dots, v_{k+\ell}) = \sum_{\sigma \in S_{k+\ell}} (\operatorname{sgn}\sigma)\,f(v_{\sigma(1)}, \dots, v_{\sigma(k)})\,g(v_{\sigma(k+1)}, \dots, v_{\sigma(k+\ell)}).$$

Making the substitution $\sigma \mapsto \sigma\tau$ and using $\operatorname{sgn}(\sigma\tau) = (\operatorname{sgn}\sigma)(\operatorname{sgn}\tau)$, one obtains $A(f \otimes g) = (\operatorname{sgn}\tau)\,A(g \otimes f)$. Dividing by $k!\,\ell!$ gives $f \wedge g = (-1)^{k\ell}\,g \wedge f$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.23</span></p>

If $f$ is a multicovector of odd degree on $V$, then $f \wedge f = 0$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $k$ be the degree of $f$. By anticommutativity, $f \wedge f = (-1)^{k^2}\,f \wedge f = -f \wedge f$, since $k$ is odd. Hence $2\,f \wedge f = 0$. Dividing by $2$ gives $f \wedge f = 0$.

</details>
</div>

### 3.9 Associativity of the Wedge Product

To prove associativity of the wedge product, we first prove a lemma on the alternating operator $A$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.24</span></p>

Suppose $f$ is a $k$-linear function and $g$ an $\ell$-linear function on a vector space $V$. Then

- (i) $A(A(f) \otimes g) = k!\,A(f \otimes g)$, and
- (ii) $A(f \otimes A(g)) = \ell!\,A(f \otimes g)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.25</span><span class="math-callout__name">(Associativity of the wedge product)</span></p>

Let $V$ be a real vector space and $f, g, h$ alternating multilinear functions on $V$ of degrees $k, \ell, m$, respectively. Then

$$(f \wedge g) \wedge h = f \wedge (g \wedge h).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By the definition of the wedge product,

$$(f \wedge g) \wedge h = \frac{1}{(k+\ell)!\,m!}\,A((f \wedge g) \otimes h) = \frac{1}{(k+\ell)!\,m!} \cdot \frac{1}{k!\,\ell!}\,A(A(f \otimes g) \otimes h).$$

By Lemma 3.24(i), $A(A(f \otimes g) \otimes h) = (k+\ell)!\,A((f \otimes g) \otimes h)$, so

$$(f \wedge g) \wedge h = \frac{1}{k!\,\ell!\,m!}\,A((f \otimes g) \otimes h).$$

Similarly, $f \wedge (g \wedge h) = \frac{1}{k!\,\ell!\,m!}\,A(f \otimes (g \otimes h))$. Since the tensor product is associative, $(f \wedge g) \wedge h = f \wedge (g \wedge h)$.

</details>
</div>

By associativity, we can omit the parentheses and write simply $f \wedge g \wedge h$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.26</span></p>

Under the hypotheses of the proposition, if $f_i \in A_{d_i}(V)$, then

$$f_1 \wedge \cdots \wedge f_r = \frac{1}{(d_1)!\cdots(d_r)!}\,A(f_1 \otimes \cdots \otimes f_r).$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.27</span><span class="math-callout__name">(Wedge product of 1-covectors)</span></p>

If $\alpha^1, \dots, \alpha^k$ are linear functions on a vector space $V$ and $v_1, \dots, v_k \in V$, then

$$(\alpha^1 \wedge \cdots \wedge \alpha^k)(v_1, \dots, v_k) = \det[\alpha^i(v_j)].$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By Corollary 3.26 with each $d_i = 1$,

$$(\alpha^1 \wedge \cdots \wedge \alpha^k)(v_1, \dots, v_k) = A(\alpha^1 \otimes \cdots \otimes \alpha^k)(v_1, \dots, v_k)$$

$$= \sum_{\sigma \in S_k} (\operatorname{sgn}\sigma)\,\alpha^1(v_{\sigma(1)}) \cdots \alpha^k(v_{\sigma(k)}) = \det[\alpha^i(v_j)].$$

</details>
</div>

### 3.10 A Basis for $k$-Covectors

Let $e_1, \dots, e_n$ be a basis for a real vector space $V$, and let $\alpha^1, \dots, \alpha^n$ be the dual basis for $V^\vee$. Introduce the multi-index notation $I = (i_1, \dots, i_k)$ and write $e_I$ for $(e_{i_1}, \dots, e_{i_k})$ and $\alpha^I$ for $\alpha^{i_1} \wedge \cdots \wedge \alpha^{i_k}$.

A $k$-linear function $f$ on $V$ is completely determined by its values on all $k$-tuples $(e_{i_1}, \dots, e_{i_k})$. If $f$ is alternating, then it is completely determined by its values on $(e_{i_1}, \dots, e_{i_k})$ with $1 \le i_1 < \cdots < i_k \le n$; that is, it suffices to consider $e_I$ with $I$ in **strictly ascending** order.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.28</span></p>

Let $e_1, \dots, e_n$ be a basis for a vector space $V$ and let $\alpha^1, \dots, \alpha^n$ be its dual basis in $V^\vee$. If $I = (1 \le i_1 < \cdots < i_k \le n)$ and $J = (1 \le j_1 < \cdots < j_k \le n)$ are strictly ascending multi-indices of length $k$, then

$$\alpha^I(e_J) = \delta_J^I = \begin{cases} 1 & \text{for } I = J, \\ 0 & \text{for } I \neq J. \end{cases}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.29</span></p>

The alternating $k$-linear functions $\alpha^I$, $I = (i_1 < \cdots < i_k)$, form a basis for the space $A_k(V)$ of alternating $k$-linear functions on $V$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**Linear independence.** Suppose $\sum_I c_I \alpha^I = 0$, $c_I \in \mathbb{R}$, and $I$ runs over all strictly ascending multi-indices of length $k$. Applying both sides to $e_J$, $J = (j_1 < \cdots < j_k)$, we get by Lemma 3.28,

$$0 = \sum_I c_I \alpha^I(e_J) = \sum_I c_I \delta_J^I = c_J,$$

since among all strictly ascending multi-indices $I$ of length $k$, there is only one equal to $J$. This proves that the $\alpha^I$ are linearly independent.

**Spanning.** To show that the $\alpha^I$ span $A_k(V)$, let $f \in A_k(V)$. We claim that $f = \sum_I f(e_I)\,\alpha^I$, where $I$ runs over all strictly ascending multi-indices of length $k$. Let $g = \sum_I f(e_I)\,\alpha^I$. By $k$-linearity and the alternating property, if two $k$-covectors agree on all $e_J$, where $J = (j_1 < \cdots < j_k)$, then they are equal. But

$$g(e_J) = \sum_I f(e_I)\,\alpha^I(e_J) = \sum_I f(e_I)\,\delta_J^I = f(e_J).$$

Therefore, $f = g = \sum_I f(e_I)\,\alpha^I$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.30</span></p>

If the vector space $V$ has dimension $n$, then the vector space $A_k(V)$ of $k$-covectors on $V$ has dimension $\binom{n}{k}$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

A strictly ascending multi-index $I = (i_1 < \cdots < i_k)$ is obtained by choosing a subset of $k$ numbers from $1, \dots, n$. This can be done in $\binom{n}{k}$ ways.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.31</span></p>

If $k > \dim V$, then $A_k(V) = 0$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

In $\alpha^{i_1} \wedge \cdots \wedge \alpha^{i_k}$, at least two of the factors must be the same, say $\alpha^j = \alpha^\ell = \alpha$. Because $\alpha$ is a $1$-covector, $\alpha \wedge \alpha = 0$ by Corollary 3.23, so $\alpha^{i_1} \wedge \cdots \wedge \alpha^{i_k} = 0$.

</details>
</div>

#### The Exterior Algebra

An algebra $A$ over a field $K$ is said to be **graded** if it can be written as a direct sum $A = \bigoplus_{k=0}^\infty A^k$ of vector spaces over $K$ such that the multiplication map sends $A^k \times A^\ell$ to $A^{k+\ell}$. The notation $A = \bigoplus_{k=0}^\infty A^k$ means that each nonzero element of $A$ is uniquely a *finite* sum $a = a_{i_1} + \cdots + a_{i_m}$, where $a_{i_j} \neq 0 \in A^{i_j}$.

A graded algebra $A = \bigoplus_{k=0}^\infty A^k$ is said to be **anticommutative** or **graded commutative** if for all $a \in A^k$ and $b \in A^\ell$,

$$ab = (-1)^{k\ell}\,ba.$$

For a finite-dimensional vector space $V$, say of dimension $n$, define

$$A_*(V) = \bigoplus_{k=0}^\infty A_k(V) = \bigoplus_{k=0}^n A_k(V).$$

With the wedge product of multicovectors as multiplication, $A_*(V)$ becomes an anticommutative graded algebra, called the **exterior algebra** or the **Grassmann algebra** of multicovectors on the vector space $V$.

# Chapter 4 — Differential Forms on $\mathbb{R}^n$

Just as a vector field assigns a tangent vector to each point of an open subset $U$ of $\mathbb{R}^n$, so dually a differential $k$-form assigns a $k$-covector on the tangent space to each point of $U$. The wedge product of differential forms is defined pointwise as the wedge product of multicovectors. Since differential forms exist on an open set, not just at a single point, there is a notion of differentiation for differential forms. In fact, there is a unique one, called the *exterior derivative*, characterized by three natural properties. Although we define it using the standard coordinates of $\mathbb{R}^n$, the exterior derivative turns out to be independent of coordinates, and is therefore intrinsic to a manifold.

Differential forms extend Grassmann's exterior algebra from the tangent space at a point globally to an entire manifold. It is the ultimate abstract extension to a manifold of the gradient, curl, and divergence of vector calculus in $\mathbb{R}^3$.

## §4.1 Differential 1-Forms and the Differential of a Function

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cotangent Space)</span></p>

The **cotangent space** to $\mathbb{R}^n$ at $p$, denoted by $T_p^*(\mathbb{R}^n)$ or $T_p^*\mathbb{R}^n$, is defined to be the dual space $(T_p\mathbb{R}^n)^\vee$ of the tangent space $T_p(\mathbb{R}^n)$. Thus, an element of the cotangent space $T_p^*(\mathbb{R}^n)$ is a covector or a linear functional on the tangent space $T_p(\mathbb{R}^n)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential 1-Form)</span></p>

In parallel with the definition of a vector field, a **covector field** or a **differential 1-form** on an open subset $U$ of $\mathbb{R}^n$ is a function $\omega$ that assigns to each point $p$ in $U$ a covector $\omega_p \in T_p^*(\mathbb{R}^n)$:

$$\omega \colon U \to \bigcup_{p \in U} T_p^*(\mathbb{R}^n), \qquad p \mapsto \omega_p \in T_p^*(\mathbb{R}^n).$$

Note that the sets $T_p^*(\mathbb{R}^n)$ in the union are all disjoint. We call a differential 1-form a **1-form** for short.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential of a Function)</span></p>

From any $C^\infty$ function $f \colon U \to \mathbb{R}$, we can construct a 1-form $df$, called the **differential** of $f$, as follows. For $p \in U$ and $X_p \in T_pU$, define

$$(df)_p(X_p) = X_p f.$$

The directional derivative of a function in the direction of a tangent vector at a point $p$ sets up a bilinear pairing

$$T_p(\mathbb{R}^n) \times C_p^\infty(\mathbb{R}^n) \to \mathbb{R}, \qquad (X_p, f) \mapsto \langle X_p, f \rangle = X_p f.$$

One may think of a tangent vector as a function on the second argument of this pairing: $\langle X_p, \cdot \rangle$. The differential $(df)_p$ at $p$ is a function on the first argument of the pairing: $(df)_p = \langle \cdot, f \rangle$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.1</span></p>

If $x^1, \dots, x^n$ are the standard coordinates on $\mathbb{R}^n$, then at each point $p \in \mathbb{R}^n$, $\lbrace (dx^1)_p, \dots, (dx^n)_p \rbrace$ is the basis for the cotangent space $T_p^*(\mathbb{R}^n)$ dual to the basis $\lbrace \partial/\partial x^1\vert_p, \dots, \partial/\partial x^n\vert_p \rbrace$ for the tangent space $T_p(\mathbb{R}^n)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By definition,

$$(dx^i)_p\!\left(\frac{\partial}{\partial x^j}\bigg\vert_p\right) = \frac{\partial}{\partial x^j}\bigg\vert_p\, x^i = \delta_j^i. \qquad \square$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.2</span><span class="math-callout__name">(The Differential in Terms of Coordinates)</span></p>

If $f \colon U \to \mathbb{R}$ is a $C^\infty$ function on an open set $U$ in $\mathbb{R}^n$, then

$$df = \sum_i \frac{\partial f}{\partial x^i}\,dx^i.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By Proposition 4.1, at each point $p$ in $U$,

$$(df)_p = \sum_i a_i(p)\,(dx^i)_p$$

for some real numbers $a_i(p)$ depending on $p$. Thus, $df = \sum a_i\,dx^i$ for some real functions $a_i$ on $U$. To find $a_j$, apply both sides to the coordinate vector field $\partial/\partial x^j$:

$$df\!\left(\frac{\partial}{\partial x^j}\right) = \sum_i a_i\,dx^i\!\left(\frac{\partial}{\partial x^j}\right) = \sum_i a_i \delta_j^i = a_j.$$

On the other hand, by the definition of the differential,

$$df\!\left(\frac{\partial}{\partial x^j}\right)\bigg\vert_p = \frac{\partial f}{\partial x^j}. \qquad \square$$

</details>
</div>

Equation (4.1) shows that if $f$ is a $C^\infty$ function, then the 1-form $df$ is also $C^\infty$.

If $\omega$ is a 1-form on an open subset $U$ of $\mathbb{R}^n$, then by Proposition 4.1, at each point $p$ in $U$, $\omega$ can be written as a linear combination

$$\omega_p = \sum_i a_i(p)\,(dx^i)_p,$$

for some $a_i(p) \in \mathbb{R}$. As $p$ varies over $U$, the coefficients $a_i$ become functions on $U$, and we may write $\omega = \sum a_i\,dx^i$. The covector field $\omega$ is said to be $C^\infty$ on $U$ if the coefficient functions $a_i$ are all $C^\infty$ on $U$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Differential 1-Forms and Tangent Vectors)</span></p>

Differential 1-forms arise naturally even if one is interested only in tangent vectors. Every tangent vector $X_p \in T_p(\mathbb{R}^n)$ is a linear combination of the standard basis vectors:

$$X_p = \sum_i b^i(X_p)\,\frac{\partial}{\partial x^i}\bigg\vert_p.$$

At each point $p \in \mathbb{R}^n$, we have $b^i(X_p) = (dx^i)_p(X_p)$. Hence, the coefficient $b^i$ of a vector at $p$ with respect to the standard basis $\partial/\partial x^1\vert_p, \dots, \partial/\partial x^n\vert_p$ is none other than the dual covector $dx^i\vert_p$ on $\mathbb{R}^n$. As $p$ varies, $b^i = dx^i$.

</div>

## §4.2 Differential $k$-Forms

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential $k$-Form)</span></p>

More generally, a **differential form $\omega$ of degree $k$** or a **$k$-form** on an open subset $U$ of $\mathbb{R}^n$ is a function that assigns to each point $p$ in $U$ an alternating $k$-linear function on the tangent space $T_p(\mathbb{R}^n)$, i.e., $\omega_p \in A_k(T_p\mathbb{R}^n)$. Since $A_1(T_p\mathbb{R}^n) = T_p^*(\mathbb{R}^n)$, the definition of a $k$-form generalizes that of a 1-form in Subsection 4.1.

</div>

By Proposition 3.29, a basis for $A_k(T_p\mathbb{R}^n)$ is

$$dx_p^I = dx_p^{i_1} \wedge \cdots \wedge dx_p^{i_k}, \quad 1 \le i_1 < \cdots < i_k \le n.$$

Therefore, at each point $p$ in $U$, $\omega_p$ is a linear combination

$$\omega_p = \sum_I a_I(p)\,dx_p^I, \quad 1 \le i_1 < \cdots < i_k \le n,$$

and a $k$-form on $U$ is a linear combination

$$\omega = \sum_I a_I\,dx^I,$$

with function coefficients $a_I \colon U \to \mathbb{R}$. We say that a $k$-form $\omega$ is $C^\infty$ on $U$ if all the coefficients $a_I$ are $C^\infty$ functions on $U$.

Denote by $\Omega^k(U)$ the vector space of $C^\infty$ $k$-forms on $U$. A 0-form on $U$ assigns to each point $p$ in $U$ an element of $A_0(T_p\mathbb{R}^n) = \mathbb{R}$. Thus, a 0-form on $U$ is simply a function on $U$, and $\Omega^0(U) = C^\infty(U)$.

There are no nonzero differential forms of degree $> n$ on an open subset of $\mathbb{R}^n$. This is because if $\deg dx^I > n$, then in the expression $dx^I$ at least two of the 1-forms $dx^{i_\alpha}$ must be the same, forcing $dx^I = 0$.

The **wedge product** of a $k$-form $\omega$ and an $\ell$-form $\tau$ on an open set $U$ is defined pointwise:

$$(\omega \wedge \tau)_p = \omega_p \wedge \tau_p, \quad p \in U.$$

In terms of coordinates, if $\omega = \sum_I a_I\,dx^I$ and $\tau = \sum_J b_J\,dx^J$, then

$$\omega \wedge \tau = \sum_{I,J} (a_I b_J)\,dx^I \wedge dx^J.$$

If $I$ and $J$ are not disjoint on the right-hand side, then $dx^I \wedge dx^J = 0$. Hence, the sum is actually over disjoint multi-indices:

$$\omega \wedge \tau = \sum_{\substack{I,J \text{ disjoint}}} (a_I b_J)\,dx^I \wedge dx^J,$$

which shows that the wedge product of two $C^\infty$ forms is $C^\infty$. So the wedge product is a bilinear map

$$\wedge \colon \Omega^k(U) \times \Omega^\ell(U) \to \Omega^{k+\ell}(U).$$

By Propositions 3.21 and 3.25, the wedge product of differential forms is anticommutative and associative.

In case one of the factors has degree 0, say $k = 0$, the wedge product

$$\wedge \colon \Omega^0(U) \times \Omega^\ell(U) \to \Omega^\ell(U)$$

is the pointwise multiplication of a $C^\infty$ $\ell$-form by a $C^\infty$ function:

$$(f \wedge \omega)_p = f(p) \wedge \omega_p = f(p)\,\omega_p,$$

since the wedge product with a 0-covector is scalar multiplication. Thus, if $f \in C^\infty(U)$ and $\omega \in \Omega^\ell(U)$, then $f \wedge \omega = f\omega$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Differential Forms on $\mathbb{R}^3$)</span></p>

Let $x, y, z$ be the coordinates on $\mathbb{R}^3$. The $C^\infty$ 1-forms are

$$f\,dx + g\,dy + h\,dz,$$

where $f, g, h$ range over all $C^\infty$ functions on $\mathbb{R}^3$. The $C^\infty$ 2-forms are

$$f\,dy \wedge dz + g\,dz \wedge dx + h\,dx \wedge dy$$

and the $C^\infty$ 3-forms are

$$f\,dx \wedge dy \wedge dz.$$

</div>

With the wedge product as multiplication and the degree of a form as the grading, the direct sum $\Omega^*(U) = \bigoplus_{k=0}^n \Omega^k(U)$ becomes an anticommutative graded algebra over $\mathbb{R}$. Since one can multiply $C^\infty$ $k$-forms by $C^\infty$ functions, the set $\Omega^k(U)$ of $C^\infty$ $k$-forms on $U$ is both a vector space over $\mathbb{R}$ and a module over $C^\infty(U)$, and so the direct sum $\Omega^*(U) = \bigoplus_{k=0}^n \Omega^k(U)$ is also a module over the ring $C^\infty(U)$ of $C^\infty$ functions.

## §4.3 Differential Forms as Multilinear Functions on Vector Fields

If $\omega$ is a $C^\infty$ 1-form and $X$ is a $C^\infty$ vector field on an open set $U$ in $\mathbb{R}^n$, we define a function $\omega(X)$ on $U$ by the formula

$$\omega(X)_p = \omega_p(X_p), \quad p \in U.$$

Written out in coordinates, if $\omega = \sum a_i\,dx^i$ and $X = \sum b^j\,\partial/\partial x^j$ for some $a_i, b^j \in C^\infty(U)$, then

$$\omega(X) = \left(\sum a_i\,dx^i\right)\!\left(\sum b^j\,\frac{\partial}{\partial x^j}\right) = \sum_i a_i b^i,$$

which shows that $\omega(X)$ is $C^\infty$ on $U$. Thus, a $C^\infty$ 1-form on $U$ gives rise to a map from $\mathfrak{X}(U)$ to $C^\infty(U)$.

This function is actually linear over the ring $C^\infty(U)$; i.e., if $f \in C^\infty(U)$, then $\omega(fX) = f\omega(X)$. To show this, it suffices to evaluate $\omega(fX)$ at an arbitrary point $p \in U$:

$$(\omega(fX))_p = \omega_p(f(p)X_p) = f(p)\omega_p(X_p) = (f\omega(X))_p.$$

Let $\mathcal{F}(U) = C^\infty(U)$. In this notation, a 1-form $\omega$ on $U$ gives rise to an $\mathcal{F}(U)$-linear map $\mathfrak{X}(U) \to \mathcal{F}(U)$, $X \mapsto \omega(X)$. Similarly, a $k$-form $\omega$ on $U$ gives rise to a $k$-linear map over $\mathcal{F}(U)$,

$$\underbrace{\mathfrak{X}(U) \times \cdots \times \mathfrak{X}(U)}_{k \text{ times}} \to \mathcal{F}(U), \qquad (X_1, \dots, X_k) \mapsto \omega(X_1, \dots, X_k).$$

## §4.4 The Exterior Derivative

To define the *exterior derivative* of a $C^\infty$ $k$-form on an open subset $U$ of $\mathbb{R}^n$, we first define it on 0-forms: the exterior derivative of a $C^\infty$ function $f \in C^\infty(U)$ is defined to be its differential $df \in \Omega^1(U)$; in terms of coordinates, Proposition 4.2 gives

$$df = \sum_i \frac{\partial f}{\partial x^i}\,dx^i.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.5</span><span class="math-callout__name">(Exterior Derivative)</span></p>

For $k \ge 1$, if $\omega = \sum_I a_I\,dx^I \in \Omega^k(U)$, then

$$d\omega = \sum_I da_I \wedge dx^I = \sum_I \left(\sum_j \frac{\partial a_I}{\partial x^j}\,dx^j\right) \wedge dx^I \in \Omega^{k+1}(U).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exterior Derivative of a 1-Form on $\mathbb{R}^2$)</span></p>

Let $\omega$ be the 1-form $f\,dx + g\,dy$ on $\mathbb{R}^2$, where $f$ and $g$ are $C^\infty$ functions. To simplify the notation, write $f_x = \partial f/\partial x$, $f_y = \partial f/\partial y$, $g_x = \partial g/\partial x$, $g_y = \partial g/\partial y$. Then

$$d\omega = df \wedge dx + dg \wedge dy = (f_x\,dx + f_y\,dy) \wedge dx + (g_x\,dx + g_y\,dy) \wedge dy = (g_x - f_y)\,dx \wedge dy.$$

In this computation $dy \wedge dx = -dx \wedge dy$ and $dx \wedge dx = dy \wedge dy = 0$ by the anticommutative property of the wedge product.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.6</span><span class="math-callout__name">(Antiderivation)</span></p>

Let $A = \bigoplus_{k=0}^\infty A^k$ be a graded algebra over a field $K$. An **antiderivation** of the graded algebra $A$ is a $K$-linear map $D \colon A \to A$ such that for $a \in A^k$ and $b \in A^\ell$,

$$D(ab) = (Da)b + (-1)^k a\,Db.$$

If there is an integer $m$ such that the antiderivation $D$ sends $A^k$ to $A^{k+m}$ for all $k$, then we say that it is an antiderivation of **degree $m$**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.7</span></p>

The exterior differentiation $d \colon \Omega^*(U) \to \Omega^*(U)$ is an antiderivation of degree 1:

**(i)** $d(\omega \wedge \tau) = (d\omega) \wedge \tau + (-1)^{\deg \omega}\,\omega \wedge d\tau.$

**(ii)** $d^2 = 0.$

**(iii)** If $f \in C^\infty(U)$ and $X \in \mathfrak{X}(U)$, then $(df)(X) = Xf.$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**(i)** Since both sides are linear in $\omega$ and in $\tau$, it suffices to check the equality for $\omega = f\,dx^I$ and $\tau = g\,dx^J$. Then

$$d(\omega \wedge \tau) = d(fg\,dx^I \wedge dx^J) = \sum_i \frac{\partial(fg)}{\partial x^i}\,dx^i \wedge dx^I \wedge dx^J = \sum_i \frac{\partial f}{\partial x^i}\,g\,dx^i \wedge dx^I \wedge dx^J + \sum_i f\,\frac{\partial g}{\partial x^i}\,dx^i \wedge dx^I \wedge dx^J.$$

In the second sum, moving the 1-form $(\partial g/\partial x^i)\,dx^i$ across the $k$-form $dx^I$ results in the sign $(-1)^k$ by anticommutativity. Hence,

$$d(\omega \wedge \tau) = d\omega \wedge \tau + (-1)^k\,\omega \wedge d\tau.$$

**(ii)** By the $\mathbb{R}$-linearity of $d$, it suffices to show that $d^2\omega = 0$ for $\omega = f\,dx^I$. We compute:

$$d^2(f\,dx^I) = d\!\left(\sum_i \frac{\partial f}{\partial x^i}\,dx^i \wedge dx^I\right) = \sum_{i,j} \frac{\partial^2 f}{\partial x^j \partial x^i}\,dx^j \wedge dx^i \wedge dx^I.$$

In this sum if $i = j$, then $dx^i \wedge dx^i = 0$; if $i \neq j$, then $\partial^2 f/\partial x^j \partial x^i$ is symmetric in $i$ and $j$, but $dx^j \wedge dx^i$ is alternating in $i$ and $j$, so the terms with $i \neq j$ pair up and cancel each other. Therefore, $d^2(f\,dx^I) = 0$.

**(iii)** This is simply the definition of the exterior derivative of a function as the differential of the function. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.8</span><span class="math-callout__name">(Characterization of the Exterior Derivative)</span></p>

The three properties of Proposition 4.7 uniquely characterize exterior differentiation on an open set $U$ in $\mathbb{R}^n$; that is, if $D \colon \Omega^*(U) \to \Omega^*(U)$ is (i) an antiderivation of degree 1 such that (ii) $D^2 = 0$ and (iii) $(Df)(X) = Xf$ for $f \in C^\infty(U)$ and $X \in \mathfrak{X}(U)$, then $D = d$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since every $k$-form on $U$ is a sum of terms such as $f\,dx^{i_1} \wedge \cdots \wedge dx^{i_k}$, by linearity it suffices to show that $D = d$ on a $k$-form of this type. By (iii), $Df = df$ on $C^\infty$ functions. It follows that $Ddx^i = DDx^i = 0$ by (ii). A simple induction on $k$, using the antiderivation property of $D$, proves that for all $k$ and all multi-indices $I$ of length $k$,

$$D(dx^I) = D(dx^{i_1} \wedge \cdots \wedge dx^{i_k}) = 0.$$

Finally, for every $k$-form $f\,dx^I$,

$$D(f\,dx^I) = (Df) \wedge dx^I + f\,D(dx^I) = (df) \wedge dx^I = d(f\,dx^I).$$

Hence, $D = d$ on $\Omega^*(U)$. $\square$

</details>
</div>

## §4.5 Closed Forms and Exact Forms

A $k$-form $\omega$ on $U$ is **closed** if $d\omega = 0$; it is **exact** if there is a $(k-1)$-form $\tau$ such that $\omega = d\tau$ on $U$. Since $d(d\tau) = 0$, every exact form is closed. In the next section we will discuss the meaning of closed and exact forms in the context of vector calculus on $\mathbb{R}^3$.

A collection of vector spaces $\lbrace V^k \rbrace_{k=0}^\infty$ with linear maps $d_k \colon V^k \to V^{k+1}$ such that $d_{k+1} \circ d_k = 0$ is called a **differential complex** or a **cochain complex**. For any open subset $U$ of $\mathbb{R}^n$, the exterior derivative $d$ makes the vector space $\Omega^*(U)$ of $C^\infty$ forms on $U$ into a cochain complex, called the **de Rham complex** of $U$:

$$0 \to \Omega^0(U) \xrightarrow{d} \Omega^1(U) \xrightarrow{d} \Omega^2(U) \to \cdots.$$

The closed forms are precisely the elements of the kernel of $d$, and the exact forms are the elements of the image of $d$.

## §4.6 Applications to Vector Calculus

The theory of differential forms unifies many theorems in vector calculus on $\mathbb{R}^3$. By a **vector-valued function** on an open subset $U$ of $\mathbb{R}^3$, we mean a function $\mathbf{F} = \langle P, Q, R \rangle \colon U \to \mathbb{R}^3$. Such a function assigns to $p \in U$ a vector $\mathbf{F}_p \in \mathbb{R}^3 \simeq T_p(\mathbb{R}^3)$. Hence, a vector-valued function on $U$ is a vector field on $U$.

Recall the three operators gradient, curl, and divergence on scalar- and vector-valued functions on $U$:

$$\lbrace\text{scalar func.}\rbrace \xrightarrow{\text{grad}} \lbrace\text{vector func.}\rbrace \xrightarrow{\text{curl}} \lbrace\text{vector func.}\rbrace \xrightarrow{\text{div}} \lbrace\text{scalar func.}\rbrace$$

Since every 1-form on $U$ is a linear combination with function coefficients of $dx$, $dy$, and $dz$, we can identify 1-forms with vector fields on $U$ via

$$P\,dx + Q\,dy + R\,dz \longleftrightarrow \begin{bmatrix} P \\ Q \\ R \end{bmatrix}.$$

Similarly, 2-forms on $U$ can also be identified with vector fields on $U$:

$$P\,dy \wedge dz + Q\,dz \wedge dx + R\,dx \wedge dy \longleftrightarrow \begin{bmatrix} P \\ Q \\ R \end{bmatrix},$$

and 3-forms on $U$ can be identified with functions on $U$:

$$f\,dx \wedge dy \wedge dz \longleftrightarrow f.$$

In terms of these identifications, the exterior derivative of a 0-form $f$ is

$$df = \frac{\partial f}{\partial x}\,dx + \frac{\partial f}{\partial y}\,dy + \frac{\partial f}{\partial z}\,dz \longleftrightarrow \begin{bmatrix} \partial f/\partial x \\ \partial f/\partial y \\ \partial f/\partial z \end{bmatrix} = \operatorname{grad} f;$$

the exterior derivative of a 1-form is

$$d(P\,dx + Q\,dy + R\,dz) = (R_y - Q_z)\,dy \wedge dz - (R_x - P_z)\,dz \wedge dx + (Q_x - P_y)\,dx \wedge dy,$$

which corresponds to

$$\operatorname{curl}\begin{bmatrix} P \\ Q \\ R \end{bmatrix} = \begin{bmatrix} R_y - Q_z \\ -(R_x - P_z) \\ Q_x - P_y \end{bmatrix};$$

the exterior derivative of a 2-form is

$$d(P\,dy \wedge dz + Q\,dz \wedge dx + R\,dx \wedge dy) = (P_x + Q_y + R_z)\,dx \wedge dy \wedge dz,$$

which corresponds to

$$\operatorname{div}\begin{bmatrix} P \\ Q \\ R \end{bmatrix} = P_x + Q_y + R_z.$$

Thus, after appropriate identifications, the exterior derivatives $d$ on 0-forms, 1-forms, and 2-forms are simply the three operators grad, curl, and div. In summary, on an open subset $U$ of $\mathbb{R}^3$, there are identifications

$$\Omega^0(U) \xrightarrow{d} \Omega^1(U) \xrightarrow{d} \Omega^2(U) \xrightarrow{d} \Omega^3(U)$$

$$\simeq \downarrow \qquad \simeq \downarrow \qquad \simeq \downarrow \qquad \simeq \downarrow$$

$$C^\infty(U) \xrightarrow{\text{grad}} \mathfrak{X}(U) \xrightarrow{\text{curl}} \mathfrak{X}(U) \xrightarrow{\text{div}} C^\infty(U).$$

Under these identifications, a vector field $\langle P, Q, R \rangle$ on $\mathbb{R}^3$ is the gradient of a $C^\infty$ function $f$ if and only if the corresponding 1-form $P\,dx + Q\,dy + R\,dz$ is $df$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A</span></p>

$\operatorname{curl}(\operatorname{grad} f) = \mathbf{0}.$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition B</span></p>

$\operatorname{div}(\operatorname{curl}\,\mathbf{F}) = 0.$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition C</span></p>

On $\mathbb{R}^3$, a vector field $\mathbf{F}$ is the gradient of some scalar function $f$ if and only if $\operatorname{curl}\,\mathbf{F} = \mathbf{0}$.

</div>

Propositions A and B express the property $d^2 = 0$ of the exterior derivative on open subsets of $\mathbb{R}^3$; these are easy computations. Proposition C expresses the fact that a 1-form on $\mathbb{R}^3$ is exact if and only if it is closed. Proposition C need not be true on a region other than $\mathbb{R}^3$, as the following well-known example from calculus shows.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A Closed but Not Exact 1-Form)</span></p>

If $U = \mathbb{R}^3 - \lbrace z\text{-axis}\rbrace$, and $\mathbf{F}$ is the vector field

$$\mathbf{F} = \left\langle \frac{-y}{x^2 + y^2},\;\frac{x}{x^2 + y^2},\;0 \right\rangle$$

on $\mathbb{R}^3$, then $\operatorname{curl}\,\mathbf{F} = \mathbf{0}$, but $\mathbf{F}$ is not the gradient of any $C^\infty$ function on $U$. The reason is that if $\mathbf{F}$ were the gradient of a $C^\infty$ function $f$ on $U$, then by the fundamental theorem for line integrals, the line integral

$$\int_C -\frac{y}{x^2 + y^2}\,dx + \frac{x}{x^2 + y^2}\,dy$$

over any closed curve $C$ would be zero. However, on the unit circle $C$ in the $(x,y)$-plane, with $x = \cos t$ and $y = \sin t$ for $0 \le t \le 2\pi$, this integral is

$$\int_0^{2\pi} -(\sin t)\,d\cos t + (\cos t)\,d\sin t = 2\pi.$$

In terms of differential forms, the 1-form

$$\omega = \frac{-y}{x^2 + y^2}\,dx + \frac{x}{x^2 + y^2}\,dy$$

is closed but not exact on $U$.

</div>

It turns out that whether Proposition C is true for a region $U$ depends only on the topology of $U$. One measure of the failure of a closed $k$-form to be exact is the quotient vector space

$$H^k(U) := \frac{\lbrace\text{closed } k\text{-forms on } U\rbrace}{\lbrace\text{exact } k\text{-forms on } U\rbrace},$$

called the $k$th **de Rham cohomology** of $U$.

The generalization of Proposition C to any differential form on $\mathbb{R}^n$ is called the **Poincaré lemma**: for $k \ge 1$, every closed $k$-form on $\mathbb{R}^n$ is exact. This is of course equivalent to the vanishing of the $k$th de Rham cohomology $H^k(\mathbb{R}^n)$ for $k \ge 1$.

The theory of differential forms allows us to generalize vector calculus from $\mathbb{R}^3$ to $\mathbb{R}^n$ and indeed to a manifold of any dimension. The general Stokes theorem for a manifold subsumes and unifies the fundamental theorem for line integrals, Green's theorem in the plane, the classical Stokes theorem for a surface in $\mathbb{R}^3$, and the divergence theorem.

## §4.7 Convention on Subscripts and Superscripts

In differential geometry it is customary to index vector fields with subscripts $e_1, \dots, e_n$, and differential forms with superscripts $\omega^1, \dots, \omega^n$. Being 0-forms, coordinate functions take superscripts: $x^1, \dots, x^n$. Their differentials, being 1-forms, should also have superscripts, and indeed they do: $dx^1, \dots, dx^n$. Coordinate vector fields $\partial/\partial x^1, \dots, \partial/\partial x^n$ are considered to have subscripts because the $i$ in $\partial/\partial x^i$, although a superscript for $x^i$, is in the lower half of the fraction.

Coefficient functions can have superscripts or subscripts depending on whether they are the coefficient functions of a vector field or of a differential form. For a vector field $X = \sum a^i e_i$, the coefficient functions $a^i$ have superscripts; the idea is that the superscript in $a^i$ "cancels out" the subscript in $e_i$. For the same reason, the coefficient functions $b_j$ in a differential form $\omega = \sum b_j\,dx^j$ have subscripts.

The beauty of this convention is that there is a "conservation of indices" on the two sides of an equality sign. For example, if $X = \sum a^i\,\partial/\partial x^i$, then

$$a^i = (dx^i)(X).$$

Here both sides have a net superscript $i$. As another example, if $\omega = \sum b_j\,dx^j$, then

$$\omega(X) = \left(\sum b_j\,dx^j\right)\!\left(\sum a^i\,\frac{\partial}{\partial x^i}\right) = \sum_i b_i a^i;$$

after cancellation of superscripts and subscripts, both sides of the equality sign have zero net index. This convention is a useful mnemonic aid in some of the transformation formulas of differential geometry.

# Chapter 2 — Manifolds

Intuitively, a manifold is a generalization of curves and surfaces to higher dimensions. It is locally Euclidean in that every point has a neighborhood, called a chart, homeomorphic to an open subset of $\mathbb{R}^n$. The coordinates on a chart allow one to carry out computations as though in a Euclidean space, so that many concepts from $\mathbb{R}^n$, such as differentiability, point-derivations, tangent spaces, and differential forms, carry over to a manifold.

In this chapter we give the basic definitions and properties of a smooth manifold and of smooth maps between manifolds. Initially, the only way to verify that a space is a manifold is to exhibit a collection of $C^\infty$ compatible charts covering the space. In Section 7 we describe a set of sufficient conditions under which a quotient topological space becomes a manifold, giving us a second way to construct manifolds.

## §5 Manifolds

### 5.1 Topological Manifolds

We first recall a few definitions from point-set topology. A topological space is **second countable** if it has a countable basis. A **neighborhood** of a point $p$ in a topological space $M$ is any open set containing $p$. An **open cover** of $M$ is a collection $\lbrace U_\alpha \rbrace_{\alpha \in A}$ of open sets in $M$ whose union $\bigcup_{\alpha \in A} U_\alpha$ is $M$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1</span><span class="math-callout__name">(Locally Euclidean Space)</span></p>

A topological space $M$ is **locally Euclidean of dimension $n$** if every point $p$ in $M$ has a neighborhood $U$ such that there is a homeomorphism $\phi$ from $U$ onto an open subset of $\mathbb{R}^n$. We call the pair $(U, \phi)$ a **chart**, $U$ a **coordinate neighborhood** or a **coordinate open set**, and $\phi$ a **coordinate map** or a **coordinate system** on $U$. We say that a chart $(U, \phi)$ is **centered** at $p \in U$ if $\phi(p) = 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.2</span><span class="math-callout__name">(Topological Manifold)</span></p>

A **topological manifold** is a Hausdorff, second countable, locally Euclidean space. It is said to be of **dimension $n$** if it is locally Euclidean of dimension $n$.

</div>

For the dimension of a topological manifold to be well defined, we need to know that for $n \neq m$ an open subset of $\mathbb{R}^n$ is not homeomorphic to an open subset of $\mathbb{R}^m$. This fact, called **invariance of dimension**, is indeed true, but is not easy to prove directly. We will not pursue this point, since we are mainly interested in *smooth* manifolds, for which the analogous result is easy to prove (Corollary 8.7).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Euclidean Space)</span></p>

The Euclidean space $\mathbb{R}^n$ is covered by a single chart $(\mathbb{R}^n, \mathbb{1}_{\mathbb{R}^n})$, where $\mathbb{1}_{\mathbb{R}^n} \colon \mathbb{R}^n \to \mathbb{R}^n$ is the identity map. It is the prime example of a topological manifold. Every open subset of $\mathbb{R}^n$ is also a topological manifold, with chart $(U, \mathbb{1}_U)$.

</div>

Recall that the Hausdorff condition and second countability are "hereditary properties": a subspace of a Hausdorff space is Hausdorff, and a subspace of a second-countable space is second countable. So any subspace of $\mathbb{R}^n$ is automatically Hausdorff and second countable.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.3</span><span class="math-callout__name">(A Cusp)</span></p>

The graph of $y = x^{2/3}$ in $\mathbb{R}^2$ is a topological manifold. By virtue of being a subspace of $\mathbb{R}^2$, it is Hausdorff and second countable. It is locally Euclidean, because it is homeomorphic to $\mathbb{R}$ via $(x, x^{2/3}) \mapsto x$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.4</span><span class="math-callout__name">(A Cross)</span></p>

The cross in $\mathbb{R}^2$ with the subspace topology is not locally Euclidean at the intersection point $p$, and so cannot be a topological manifold.

Suppose the cross is locally Euclidean of dimension $n$ at the point $p$. Then $p$ has a neighborhood $U$ homeomorphic to an open ball $B := B(0, \varepsilon) \subset \mathbb{R}^n$ with $p$ mapping to 0. The homeomorphism $U \to B$ restricts to a homeomorphism $U - \lbrace p \rbrace \to B - \lbrace 0 \rbrace$. Now $B - \lbrace 0 \rbrace$ is either connected if $n \ge 2$ or has two connected components if $n = 1$. Since $U - \lbrace p \rbrace$ has four connected components, there can be no homeomorphism from $U - \lbrace p \rbrace$ to $B - \lbrace 0 \rbrace$. This contradiction proves that the cross is not locally Euclidean at $p$.

</div>

### 5.2 Compatible Charts

Suppose $(U, \phi \colon U \to \mathbb{R}^n)$ and $(V, \psi \colon V \to \mathbb{R}^n)$ are two charts of a topological manifold. Since $U \cap V$ is open in $U$ and $\phi \colon U \to \mathbb{R}^n$ is a homeomorphism onto an open subset of $\mathbb{R}^n$, the image $\phi(U \cap V)$ will also be an open subset of $\mathbb{R}^n$. Similarly, $\psi(U \cap V)$ is an open subset of $\mathbb{R}^n$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.5</span><span class="math-callout__name">($C^\infty$-Compatible Charts)</span></p>

Two charts $(U, \phi \colon U \to \mathbb{R}^n)$, $(V, \psi \colon V \to \mathbb{R}^n)$ of a topological manifold are **$C^\infty$-compatible** if the two maps

$$\phi \circ \psi^{-1} \colon \psi(U \cap V) \to \phi(U \cap V), \qquad \psi \circ \phi^{-1} \colon \phi(U \cap V) \to \psi(U \cap V)$$

are $C^\infty$. These two maps are called the **transition functions** between the charts. If $U \cap V$ is empty, then the two charts are automatically $C^\infty$-compatible.

</div>

Since we are interested only in $C^\infty$-compatible charts, we often omit mention of "$C^\infty$" and speak simply of compatible charts.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.6</span><span class="math-callout__name">($C^\infty$ Atlas)</span></p>

A **$C^\infty$ atlas** or simply an **atlas** on a locally Euclidean space $M$ is a collection $\mathfrak{U} = \lbrace (U_\alpha, \phi_\alpha) \rbrace$ of pairwise $C^\infty$-compatible charts that cover $M$, i.e., such that $M = \bigcup_\alpha U_\alpha$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.7</span><span class="math-callout__name">(A $C^\infty$ Atlas on a Circle)</span></p>

The unit circle $S^1$ in the complex plane $\mathbb{C}$ may be described as the set of points $\lbrace e^{it} \in \mathbb{C} \mid 0 \le t \le 2\pi \rbrace$. Let $U_1$ and $U_2$ be the two open subsets of $S^1$:

$$U_1 = \lbrace e^{it} \in \mathbb{C} \mid {-\pi} < t < \pi \rbrace, \qquad U_2 = \lbrace e^{it} \in \mathbb{C} \mid 0 < t < 2\pi \rbrace,$$

and define $\phi_\alpha \colon U_\alpha \to \mathbb{R}$ for $\alpha = 1, 2$ by $\phi_1(e^{it}) = t$ for $-\pi < t < \pi$ and $\phi_2(e^{it}) = t$ for $0 < t < 2\pi$. Both $\phi_1$ and $\phi_2$ are branches of the complex log function $(1/i)\log z$ and are homeomorphisms onto their respective images. The intersection $U_1 \cap U_2$ consists of two connected components,

$$A = \lbrace e^{it} \mid {-\pi} < t < 0 \rbrace, \qquad B = \lbrace e^{it} \mid 0 < t < \pi \rbrace.$$

The transition functions are

$$(\phi_2 \circ \phi_1^{-1})(t) = \begin{cases} t + 2\pi & \text{for } t \in ]{-\pi}, 0[, \\ t & \text{for } t \in ]0, \pi[, \end{cases}$$

$$(\phi_1 \circ \phi_2^{-1})(t) = \begin{cases} t - 2\pi & \text{for } t \in ]\pi, 2\pi[, \\ t & \text{for } t \in ]0, \pi[. \end{cases}$$

Therefore, $(U_1, \phi_1)$ and $(U_2, \phi_2)$ are $C^\infty$-compatible charts and form a $C^\infty$ atlas on $S^1$.

</div>

Although the $C^\infty$ compatibility of charts is clearly reflexive and symmetric, it is not transitive. We say that a chart $(V, \psi)$ is **compatible with an atlas** $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ if it is compatible with all the charts $(U_\alpha, \phi_\alpha)$ of the atlas.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.8</span></p>

Let $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ be an atlas on a locally Euclidean space. If two charts $(V, \psi)$ and $(W, \sigma)$ are both compatible with the atlas $\lbrace (U_\alpha, \phi_\alpha) \rbrace$, then they are compatible with each other.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $p \in V \cap W$. We need to show that $\sigma \circ \psi^{-1}$ is $C^\infty$ at $\psi(p)$. Since $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ is an atlas for $M$, $p \in U_\alpha$ for some $\alpha$. Then $p$ is in the triple intersection $V \cap W \cap U_\alpha$.

By the remark above, $\sigma \circ \psi^{-1} = (\sigma \circ \phi_\alpha^{-1}) \circ (\phi_\alpha \circ \psi^{-1})$ is $C^\infty$ on $\psi(V \cap W \cap U_\alpha)$, hence at $\psi(p)$. Since $p$ was an arbitrary point of $V \cap W$, this proves that $\sigma \circ \psi^{-1}$ is $C^\infty$ on $\psi(V \cap W)$. Similarly, $\psi \circ \sigma^{-1}$ is $C^\infty$ on $\sigma(V \cap W)$. $\square$

</details>
</div>

### 5.3 Smooth Manifolds

An atlas $\mathfrak{M}$ on a locally Euclidean space is said to be **maximal** if it is not contained in a larger atlas; in other words, if $\mathfrak{U}$ is any other atlas containing $\mathfrak{M}$, then $\mathfrak{U} = \mathfrak{M}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.9</span><span class="math-callout__name">(Smooth Manifold)</span></p>

A **smooth** or **$C^\infty$ manifold** is a topological manifold $M$ together with a maximal atlas. The maximal atlas is also called a **differentiable structure** on $M$. A manifold is said to have dimension $n$ if all of its connected components have dimension $n$. A 1-dimensional manifold is also called a **curve**, a 2-dimensional manifold a **surface**, and an $n$-dimensional manifold an **$n$-manifold**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.10</span></p>

Any atlas $\mathfrak{U} = \lbrace (U_\alpha, \phi_\alpha) \rbrace$ on a locally Euclidean space is contained in a unique maximal atlas.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Adjoin to the atlas $\mathfrak{U}$ all charts $(V_i, \psi_i)$ that are compatible with $\mathfrak{U}$. By Lemma 5.8 the charts $(V_i, \psi_i)$ are compatible with one another. So the enlarged collection of charts is an atlas. Any chart compatible with the original atlas $\mathfrak{U}$ and so by construction belongs to the new atlas. This proves that the new atlas is maximal.

Let $\mathfrak{M}$ be the maximal atlas that we have just constructed. If $\mathfrak{M}'$ is another maximal atlas containing $\mathfrak{U}$, then all the charts in $\mathfrak{M}'$ are compatible with $\mathfrak{U}$ and so by construction must belong to $\mathfrak{M}$. This proves that $\mathfrak{M}' \subset \mathfrak{M}$. Since both are maximal, $\mathfrak{M}' = \mathfrak{M}$. Therefore, the maximal atlas containing $\mathfrak{U}$ is unique. $\square$

</details>
</div>

In summary, to show that a topological space $M$ is a $C^\infty$ manifold, it suffices to check that

(i) $M$ is Hausdorff and second countable,

(ii) $M$ has a $C^\infty$ atlas (not necessarily maximal).

From now on, a "manifold" will mean a $C^\infty$ manifold. We use the terms "smooth" and "$C^\infty$" interchangeably. In the context of manifolds, we denote the standard coordinates on $\mathbb{R}^n$ by $r^1, \dots, r^n$. If $(U, \phi \colon U \to \mathbb{R}^n)$ is a chart of a manifold, we let $x^i = r^i \circ \phi$ be the $i$th component of $\phi$ and write $\phi = (x^1, \dots, x^n)$ and $(U, \phi) = (U, x^1, \dots, x^n)$. Thus, for $p \in U$, $(x^1(p), \dots, x^n(p))$ is a point in $\mathbb{R}^n$. The functions $x^1, \dots, x^n$ are called **coordinates** or **local coordinates** on the open set $U$. By a **chart** $(U, \phi)$ **about** $p$ in a manifold $M$, we will mean a chart in the differentiable structure of $M$ such that $p \in U$.

### 5.4 Examples of Smooth Manifolds

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.11</span><span class="math-callout__name">(Euclidean Space)</span></p>

The Euclidean space $\mathbb{R}^n$ is a smooth manifold with a single chart $(\mathbb{R}^n, r^1, \dots, r^n)$, where $r^1, \dots, r^n$ are the standard coordinates on $\mathbb{R}^n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.12</span><span class="math-callout__name">(Open Subset of a Manifold)</span></p>

Any open subset $V$ of a manifold $M$ is also a manifold. If $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ is an atlas for $M$, then $\lbrace (U_\alpha \cap V, \phi_\alpha\vert_{U_\alpha \cap V}) \rbrace$ is an atlas for $V$, where $\phi_\alpha\vert_{U_\alpha \cap V} \colon U_\alpha \cap V \to \mathbb{R}^n$ denotes the restriction of $\phi_\alpha$ to the subset $U_\alpha \cap V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.13</span><span class="math-callout__name">(Manifolds of Dimension Zero)</span></p>

In a manifold of dimension zero, every singleton subset is homeomorphic to $\mathbb{R}^0$ and so is open. Thus, a zero-dimensional manifold is a discrete set. By second countability, this discrete set must be countable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.14</span><span class="math-callout__name">(Graph of a Smooth Function)</span></p>

For a subset $A \subset \mathbb{R}^n$ and a function $f \colon A \to \mathbb{R}^m$, the **graph** of $f$ is defined to be the subset

$$\Gamma(f) = \lbrace (x, f(x)) \in A \times \mathbb{R}^m \rbrace.$$

If $U$ is an open subset of $\mathbb{R}^n$ and $f \colon U \to \mathbb{R}^n$ is $C^\infty$, then the two maps

$$\phi \colon \Gamma(f) \to U, \quad (x, f(x)) \mapsto x, \qquad (1, f) \colon U \to \Gamma(f), \quad x \mapsto (x, f(x)),$$

are continuous and inverse to each other, and so are homeomorphisms. The graph $\Gamma(f)$ of a $C^\infty$ function $f \colon U \to \mathbb{R}^m$ has an atlas with a single chart $(\Gamma(f), \phi)$, and is therefore a $C^\infty$ manifold. This shows that many of the familiar surfaces of calculus, for example an elliptic paraboloid or a hyperbolic paraboloid, are manifolds.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.15</span><span class="math-callout__name">(General Linear Groups)</span></p>

For any two positive integers $m$ and $n$ let $\mathbb{R}^{m \times n}$ be the vector space of all $m \times n$ matrices. Since $\mathbb{R}^{m \times n}$ is isomorphic to $\mathbb{R}^{mn}$, we give it the topology of $\mathbb{R}^{mn}$. The **general linear group** $\operatorname{GL}(n, \mathbb{R})$ is by definition

$$\operatorname{GL}(n, \mathbb{R}) := \lbrace A \in \mathbb{R}^{n \times n} \mid \det A \neq 0 \rbrace = \det^{-1}(\mathbb{R} - \lbrace 0 \rbrace).$$

Since the determinant function $\det \colon \mathbb{R}^{n \times n} \to \mathbb{R}$ is continuous, $\operatorname{GL}(n, \mathbb{R})$ is an open subset of $\mathbb{R}^{n \times n} \simeq \mathbb{R}^{n^2}$ and is therefore a manifold.

The **complex general linear group** $\operatorname{GL}(n, \mathbb{C})$ is defined to be the group of nonsingular $n \times n$ complex matrices. Since an $n \times n$ matrix $A$ is nonsingular if and only if $\det A \neq 0$, $\operatorname{GL}(n, \mathbb{C})$ is an open subset of $\mathbb{C}^{n \times n} \simeq \mathbb{R}^{2n^2}$, the vector space of $n \times n$ complex matrices. By the same reasoning as in the real case, $\operatorname{GL}(n, \mathbb{C})$ is a manifold of dimension $2n^2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.16</span><span class="math-callout__name">(Unit Circle in the $(x,y)$-Plane)</span></p>

In Example 5.7 we found a $C^\infty$ atlas with two charts on the unit circle $S^1$ in the complex plane $\mathbb{C}$. It follows that $S^1$ is a manifold. We now view $S^1$ as the unit circle in the real plane $\mathbb{R}^2$ with defining equation $x^2 + y^2 = 1$, and describe a $C^\infty$ atlas with four charts on it.

We can cover $S^1$ with four open sets: the upper and lower semicircles $U_1, U_2$, and the right and left semicircles $U_3, U_4$. On $U_1$ and $U_2$, the coordinate function $x$ is a homeomorphism onto the open interval $]{-1}, 1[$. Thus, $\phi_i(x, y) = x$ for $i = 1, 2$. Similarly, on $U_3$ and $U_4$, $y$ is a homeomorphism onto the open interval $]{-1}, 1[$, and so $\phi_i(x, y) = y$ for $i = 3, 4$.

It is easy to check that on every nonempty pairwise intersection $U_\alpha \cap U_\beta$, $\phi_\beta \circ \phi_\alpha^{-1}$ is $C^\infty$. For example, on $U_1 \cap U_3$,

$$(\phi_3 \circ \phi_1^{-1})(x) = \phi_3\!\left(x, \sqrt{1 - x^2}\right) = \sqrt{1 - x^2},$$

which is $C^\infty$. Thus, $\lbrace (U_i, \phi_i) \rbrace_{i=1}^4$ is a $C^\infty$ atlas on $S^1$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.18</span><span class="math-callout__name">(Atlas for a Product Manifold)</span></p>

If $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ and $\lbrace (V_i, \psi_i) \rbrace$ are $C^\infty$ atlases for the manifolds $M$ and $N$ of dimensions $m$ and $n$, respectively, then the collection

$$\lbrace (U_\alpha \times V_i, \phi_\alpha \times \psi_i \colon U_\alpha \times V_i \to \mathbb{R}^m \times \mathbb{R}^n) \rbrace$$

of charts is a $C^\infty$ atlas on $M \times N$. Therefore, $M \times N$ is a $C^\infty$ manifold of dimension $m + n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Product Manifolds)</span></p>

It follows from Proposition 5.18 that the infinite cylinder $S^1 \times \mathbb{R}$ and the torus $S^1 \times S^1$ are manifolds. Since $M \times N \times P = (M \times N) \times P$ is the successive product of pairs of spaces, if $M$, $N$, and $P$ are manifolds, then so is $M \times N \times P$. Thus, the $n$-dimensional torus $S^1 \times \cdots \times S^1$ ($n$ times) is a manifold.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Standard $n$-Sphere)</span></p>

Let $S^n$ be the unit sphere

$$(x^1)^2 + (x^2)^2 + \cdots + (x^{n+1})^2 = 1$$

in $\mathbb{R}^{n+1}$. It is easy to write down a $C^\infty$ atlas on $S^n$, showing that $S^n$ has a differentiable structure. The manifold $S^n$ with this differentiable structure is called the **standard $n$-sphere**.

One of the most surprising achievements in topology was John Milnor's discovery in 1956 of exotic 7-spheres, smooth manifolds homeomorphic but not diffeomorphic to the standard 7-sphere. In dimensions $< 4$ every topological manifold has a unique differentiable structure and in dimensions $> 4$ every compact topological manifold has a finite number of differentiable structures. Dimension 4 is a mystery. It is not known whether $S^4$ has a finite or infinite number of differentiable structures. The statement that $S^4$ has a unique differentiable structure is called the **smooth Poincaré conjecture**.

</div>

## §6 Smooth Maps on a Manifold

Now that we have defined smooth manifolds, it is time to consider maps between them. Using coordinate charts, one can transfer the notion of smooth maps from Euclidean spaces to manifolds. By the $C^\infty$ compatibility of charts in an atlas, the smoothness of a map turns out to be independent of the choice of charts and is therefore well defined.

### 6.1 Smooth Functions on a Manifold

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.1</span><span class="math-callout__name">(Smooth Function on a Manifold)</span></p>

Let $M$ be a smooth manifold of dimension $n$. A function $f \colon M \to \mathbb{R}$ is said to be **$C^\infty$ or smooth at a point $p$** in $M$ if there is a chart $(U, \phi)$ about $p$ in $M$ such that $f \circ \phi^{-1}$, a function defined on the open subset $\phi(U)$ of $\mathbb{R}^n$, is $C^\infty$ at $\phi(p)$. The function $f$ is said to be **$C^\infty$ on $M$** if it is $C^\infty$ at every point of $M$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.2</span></p>

The definition of the smoothness of $f$ at a point is independent of the chart $(U, \phi)$, for if $f \circ \phi^{-1}$ is $C^\infty$ at $\phi(p)$ and $(V, \psi)$ is any other chart about $p$ in $M$, then on $\psi(U \cap V)$,

$$f \circ \psi^{-1} = (f \circ \phi^{-1}) \circ (\phi \circ \psi^{-1}),$$

which is $C^\infty$ at $\psi(p)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.3</span><span class="math-callout__name">(Smoothness of a Real-Valued Function)</span></p>

Let $M$ be a manifold of dimension $n$, and $f \colon M \to \mathbb{R}$ a real-valued function on $M$. The following are equivalent:

**(i)** The function $f \colon M \to \mathbb{R}$ is $C^\infty$.

**(ii)** The manifold $M$ has an atlas such that for every chart $(U, \phi)$ in the atlas, $f \circ \phi^{-1} \colon \mathbb{R}^n \supset \phi(U) \to \mathbb{R}$ is $C^\infty$.

**(iii)** For every chart $(V, \psi)$ on $M$, the function $f \circ \psi^{-1} \colon \mathbb{R}^n \supset \psi(V) \to \mathbb{R}$ is $C^\infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

We prove the proposition as a cyclic chain of implications.

(ii) $\Rightarrow$ (i): This follows directly from the definition of a $C^\infty$ function, since by (ii) every point $p \in M$ has a coordinate neighborhood $(U, \phi)$ such that $f \circ \phi^{-1}$ is $C^\infty$ at $\phi(p)$.

(i) $\Rightarrow$ (iii): Let $(V, \psi)$ be an arbitrary chart on $M$ and let $p \in V$. By Remark 6.2, $f \circ \psi^{-1}$ is $C^\infty$ at $\psi(p)$. Since $p$ was an arbitrary point of $V$, $f \circ \psi^{-1}$ is $C^\infty$ on $\psi(V)$.

(iii) $\Rightarrow$ (ii): Obvious. $\square$

</details>
</div>

The smoothness conditions of Proposition 6.3 will be a recurrent motif throughout the book: to prove the smoothness of an object, it is sufficient that a smoothness criterion hold on the charts of some atlas. Once the object is shown to be smooth, it then follows that the same smoothness criterion holds on *every* chart on the manifold.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.4</span><span class="math-callout__name">(Pullback)</span></p>

Let $F \colon N \to M$ be a map and $h$ a function on $M$. The **pullback** of $h$ by $F$, denoted by $F^*h$, is the composite function $h \circ F$.

</div>

In this terminology, a function $f$ on $M$ is $C^\infty$ on a chart $(U, \phi)$ if and only if its pullback $(\phi^{-1})^*f$ by $\phi^{-1}$ is $C^\infty$ on the subset $\phi(U)$ of Euclidean space.

### 6.2 Smooth Maps Between Manifolds

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.5</span><span class="math-callout__name">(Smooth Map Between Manifolds)</span></p>

Let $N$ and $M$ be manifolds of dimension $n$ and $m$, respectively. A continuous map $F \colon N \to M$ is **$C^\infty$ at a point $p$** in $N$ if there are charts $(V, \psi)$ about $F(p)$ in $M$ and $(U, \phi)$ about $p$ in $N$ such that the composition $\psi \circ F \circ \phi^{-1}$, a map from the open subset $\phi(F^{-1}(V) \cap U)$ of $\mathbb{R}^n$ to $\mathbb{R}^m$, is $C^\infty$ at $\phi(p)$. The continuous map $F \colon N \to M$ is said to be **$C^\infty$** if it is $C^\infty$ at every point of $N$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.7</span></p>

Suppose $F \colon N \to M$ is $C^\infty$ at $p \in N$. If $(U, \phi)$ is any chart about $p$ in $N$ and $(V, \psi)$ is any chart about $F(p)$ in $M$, then $\psi \circ F \circ \phi^{-1}$ is $C^\infty$ at $\phi(p)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.8</span><span class="math-callout__name">(Smoothness of a Map in Terms of Charts)</span></p>

Let $N$ and $M$ be smooth manifolds, and $F \colon N \to M$ a continuous map. The following are equivalent:

**(i)** The map $F \colon N \to M$ is $C^\infty$.

**(ii)** There are atlases $\mathfrak{U}$ for $N$ and $\mathfrak{V}$ for $M$ such that for every chart $(U, \phi)$ in $\mathfrak{U}$ and $(V, \psi)$ in $\mathfrak{V}$, the map

$$\psi \circ F \circ \phi^{-1} \colon \phi(U \cap F^{-1}(V)) \to \mathbb{R}^m$$

is $C^\infty$.

**(iii)** For every chart $(U, \phi)$ on $N$ and every chart $(V, \psi)$ on $M$, the map

$$\psi \circ F \circ \phi^{-1} \colon \phi(U \cap F^{-1}(V)) \to \mathbb{R}^m$$

is $C^\infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.9</span><span class="math-callout__name">(Composition of $C^\infty$ Maps)</span></p>

If $F \colon N \to M$ and $G \colon M \to P$ are $C^\infty$ maps of manifolds, then the composite $G \circ F \colon N \to P$ is $C^\infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $(U, \phi)$, $(V, \psi)$, and $(W, \sigma)$ be charts on $N$, $M$, and $P$ respectively. Then

$$\sigma \circ (G \circ F) \circ \phi^{-1} = (\sigma \circ G \circ \psi^{-1}) \circ (\psi \circ F \circ \phi^{-1}).$$

Since $F$ and $G$ are $C^\infty$, by Proposition 6.8(i)$\Rightarrow$(iii), $\sigma \circ G \circ \psi^{-1}$ and $\psi \circ F \circ \phi^{-1}$ are $C^\infty$. As a composite of $C^\infty$ maps of open subsets of Euclidean spaces, $\sigma \circ (G \circ F) \circ \phi^{-1}$ is $C^\infty$. By Proposition 6.8(iii)$\Rightarrow$(i), $G \circ F$ is $C^\infty$. $\square$

</details>
</div>

### 6.3 Diffeomorphisms

A **diffeomorphism** of manifolds is a bijective $C^\infty$ map $F \colon N \to M$ whose inverse $F^{-1}$ is also $C^\infty$. According to the next two propositions, coordinate maps are diffeomorphisms, and conversely, every diffeomorphism of an open subset of a manifold with an open subset of a Euclidean space can serve as a coordinate map.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.10</span></p>

If $(U, \phi)$ is a chart on a manifold $M$ of dimension $n$, then the coordinate map $\phi \colon U \to \phi(U) \subset \mathbb{R}^n$ is a diffeomorphism.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.11</span></p>

Let $U$ be an open subset of a manifold $M$ of dimension $n$. If $F \colon U \to F(U) \subset \mathbb{R}^n$ is a diffeomorphism onto an open subset of $\mathbb{R}^n$, then $(U, F)$ is a chart in the differentiable structure of $M$.

</div>

### 6.4 Smoothness in Terms of Components

In this subsection we derive a criterion that reduces the smoothness of a map to the smoothness of real-valued functions on open sets.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.12</span><span class="math-callout__name">(Smoothness of a Vector-Valued Function)</span></p>

Let $N$ be a manifold and $F \colon N \to \mathbb{R}^m$ a continuous map. The following are equivalent:

**(i)** The map $F \colon N \to \mathbb{R}^m$ is $C^\infty$.

**(ii)** The manifold $N$ has an atlas such that for every chart $(U, \phi)$ in the atlas, the map $F \circ \phi^{-1} \colon \phi(U) \to \mathbb{R}^m$ is $C^\infty$.

**(iii)** For every chart $(U, \phi)$ on $N$, the map $F \circ \phi^{-1} \colon \phi(U) \to \mathbb{R}^m$ is $C^\infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.13</span><span class="math-callout__name">(Smoothness in Terms of Components)</span></p>

Let $N$ be a manifold. A vector-valued function $F \colon N \to \mathbb{R}^m$ is $C^\infty$ if and only if its component functions $F^1, \dots, F^m \colon N \to \mathbb{R}$ are all $C^\infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.15</span><span class="math-callout__name">(Smoothness of a Map in Terms of Vector-Valued Functions)</span></p>

Let $F \colon N \to M$ be a continuous map between two manifolds of dimensions $n$ and $m$ respectively. The following are equivalent:

**(i)** The map $F \colon N \to M$ is $C^\infty$.

**(ii)** The manifold $M$ has an atlas such that for every chart $(V, \psi) = (V, y^1, \dots, y^m)$ in the atlas, the vector-valued function $\psi \circ F \colon F^{-1}(V) \to \mathbb{R}^m$ is $C^\infty$.

**(iii)** For every chart $(V, \psi) = (V, y^1, \dots, y^m)$ on $M$, the vector-valued function $\psi \circ F \colon F^{-1}(V) \to \mathbb{R}^m$ is $C^\infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.16</span><span class="math-callout__name">(Smoothness of a Map in Terms of Components)</span></p>

Let $F \colon N \to M$ be a continuous map between two manifolds of dimensions $n$ and $m$ respectively. The following are equivalent:

**(i)** The map $F \colon N \to M$ is $C^\infty$.

**(ii)** The manifold $M$ has an atlas such that for every chart $(V, \psi) = (V, y^1, \dots, y^m)$ in the atlas, the components $y^i \circ F \colon F^{-1}(V) \to \mathbb{R}$ of $F$ relative to the chart are all $C^\infty$.

**(iii)** For every chart $(V, \psi) = (V, y^1, \dots, y^m)$ on $M$, the components $y^i \circ F \colon F^{-1}(V) \to \mathbb{R}$ of $F$ relative to the chart are all $C^\infty$.

</div>

### 6.5 Examples of Smooth Maps

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.17</span><span class="math-callout__name">(Smoothness of a Projection Map)</span></p>

Let $M$ and $N$ be manifolds and $\pi \colon M \times N \to M$, $\pi(p, q) = p$ the projection to the first factor. Then $\pi$ is a $C^\infty$ map.

Let $(p, q)$ be an arbitrary point of $M \times N$. Suppose $(U, \phi) = (U, x^1, \dots, x^m)$ and $(V, \psi) = (V, y^1, \dots, y^n)$ are coordinate neighborhoods of $p$ and $q$ in $M$ and $N$ respectively. By Proposition 5.18, $(U \times V, \phi \times \psi) = (U \times V, x^1, \dots, x^m, y^1, \dots, y^n)$ is a coordinate neighborhood of $(p, q)$. Then

$$(\phi \circ \pi \circ (\phi \times \psi)^{-1})(a^1, \dots, a^m, b^1, \dots, b^n) = (a^1, \dots, a^m),$$

which is a $C^\infty$ map from $(\phi \times \psi)(U \times V)$ in $\mathbb{R}^{m+n}$ to $\phi(U)$ in $\mathbb{R}^m$, so $\pi$ is $C^\infty$ at $(p, q)$. Since $(p, q)$ was an arbitrary point in $M \times N$, $\pi$ is $C^\infty$ on $M \times N$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.19</span><span class="math-callout__name">(Restriction of $C^\infty$ Functions to $S^1$)</span></p>

The unit circle $S^1$ defined by $x^2 + y^2 = 1$ in $\mathbb{R}^2$ is a $C^\infty$ manifold. A $C^\infty$ function $f(x, y)$ on $\mathbb{R}^2$ restricts to a $C^\infty$ function on $S^1$.

To see this, denote a point in $S^1$ as $p = (a, b)$ and use $x, y$ to mean the standard coordinate functions on $\mathbb{R}^2$. We can show that $x$ and $y$ restrict to $C^\infty$ functions on $S^1$ using the atlas $\lbrace (U_i, \phi_i) \rbrace_{i=1}^4$ from Example 5.16. For example, on $U_3$:

$$(x \circ \phi_3^{-1})(b) = x\!\left(\sqrt{1 - b^2},\,b\right) = \sqrt{1 - b^2},$$

which is $C^\infty$. Since $x$ is $C^\infty$ on the four open sets $U_1, U_2, U_3, U_4$ which cover $S^1$, $x$ is $C^\infty$ on $S^1$. The proof that $y$ is $C^\infty$ on $S^1$ is similar.

</div>

### 6.6 Partial Derivatives

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial Derivative on a Manifold)</span></p>

On a manifold $M$ of dimension $n$, let $(U, \phi)$ be a chart and $f$ a $C^\infty$ function. As a function into $\mathbb{R}^n$, $\phi$ has $n$ components $x^1, \dots, x^n$. For $p \in U$, we define the **partial derivative** $\partial f/\partial x^i$ of $f$ with respect to $x^i$ at $p$ to be

$$\frac{\partial}{\partial x^i}\bigg\vert_p f := \frac{\partial f}{\partial x^i}(p) := \frac{\partial(f \circ \phi^{-1})}{\partial r^i}(\phi(p)) := \frac{\partial}{\partial r^i}\bigg\vert_{\phi(p)} (f \circ \phi^{-1}).$$

As functions on $\phi(U)$,

$$\frac{\partial f}{\partial x^i} \circ \phi^{-1} = \frac{\partial(f \circ \phi^{-1})}{\partial r^i}.$$

</div>

The partial derivative $\partial f/\partial x^i$ is $C^\infty$ on $U$ because its pullback $(\partial f/\partial x^i) \circ \phi^{-1}$ is $C^\infty$ on $\phi(U)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.22</span></p>

Suppose $(U, x^1, \dots, x^n)$ is a chart on a manifold. Then $\partial x^i/\partial x^j = \delta_j^i$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

At a point $p \in U$, by the definition of $\partial/\partial x^j\vert_p$,

$$\frac{\partial x^i}{\partial x^j}(p) = \frac{\partial(x^i \circ \phi^{-1})}{\partial r^j}(\phi(p)) = \frac{\partial(r^i \circ \phi \circ \phi^{-1})}{\partial r^j}(\phi(p)) = \frac{\partial r^i}{\partial r^j}(\phi(p)) = \delta_j^i. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.23</span><span class="math-callout__name">(Jacobian Matrix and Jacobian Determinant)</span></p>

Let $F \colon N \to M$ be a smooth map, and let $(U, \phi) = (U, x^1, \dots, x^n)$ and $(V, \psi) = (V, y^1, \dots, y^m)$ be charts on $N$ and $M$ respectively such that $F(U) \subset V$. Denote by

$$F^j := y^j \circ F = r^j \circ \psi \circ F \colon U \to \mathbb{R}$$

the $j$th component of $F$ in the chart $(V, \psi)$. Then the matrix $[\partial F^j/\partial x^i]$ is called the **Jacobian matrix** of $F$ relative to the charts $(U, \phi)$ and $(V, \psi)$. In case $N$ and $M$ have the same dimension, the determinant $\det[\partial F^j/\partial x^i]$ is called the **Jacobian determinant** of $F$ relative to the two charts.

</div>

### 6.7 The Inverse Function Theorem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Local Diffeomorphism)</span></p>

By Proposition 6.11, any diffeomorphism $F \colon U \to F(U) \subset \mathbb{R}^n$ of an open subset $U$ of a manifold may be thought of as a coordinate system on $U$. We say that a $C^\infty$ map $F \colon N \to M$ is **locally invertible** or a **local diffeomorphism** at $p \in N$ if $p$ has a neighborhood $U$ on which $F\vert_U \colon U \to F(U)$ is a diffeomorphism.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.25</span><span class="math-callout__name">(Inverse Function Theorem for $\mathbb{R}^n$)</span></p>

Let $F \colon W \to \mathbb{R}^n$ be a $C^\infty$ map defined on an open subset $W$ of $\mathbb{R}^n$. For any point $p$ in $W$, the map $F$ is locally invertible at $p$ if and only if the Jacobian determinant $\det[\partial F^i/\partial r^j(p)]$ is not zero.

</div>

Because the inverse function theorem for $\mathbb{R}^n$ is a local result, it easily translates to manifolds.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.26</span><span class="math-callout__name">(Inverse Function Theorem for Manifolds)</span></p>

Let $F \colon N \to M$ be a $C^\infty$ map between two manifolds of the same dimension, and $p \in N$. Suppose for some charts $(U, \phi) = (U, x^1, \dots, x^n)$ about $p$ in $N$ and $(V, \psi) = (V, y^1, \dots, y^n)$ about $F(p)$ in $M$, $F(U) \subset V$. Set $F^i = y^i \circ F$. Then $F$ is locally invertible at $p$ if and only if its Jacobian determinant $\det[\partial F^i/\partial x^j(p)]$ is nonzero.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since $F^i = y^i \circ F = r^i \circ \psi \circ F$, the Jacobian matrix of $F$ relative to the charts $(U, \phi)$ and $(V, \psi)$ is

$$\left[\frac{\partial F^i}{\partial x^j}(p)\right] = \left[\frac{\partial(r^i \circ \psi \circ F \circ \phi^{-1})}{\partial r^j}(\phi(p))\right],$$

which is precisely the Jacobian matrix at $\phi(p)$ of the map $\psi \circ F \circ \phi^{-1} \colon \mathbb{R}^n \supset \phi(U) \to \psi(V) \subset \mathbb{R}^n$ between two open subsets of $\mathbb{R}^n$. By the inverse function theorem for $\mathbb{R}^n$,

$$\det\!\left[\frac{\partial F^i}{\partial x^j}(p)\right] \neq 0$$

if and only if $\psi \circ F \circ \phi^{-1}$ is locally invertible at $\phi(p)$. Since $\psi$ and $\phi$ are diffeomorphisms (Proposition 6.10), this last statement is equivalent to the local invertibility of $F$ at $p$. $\square$

</details>
</div>

We usually apply the inverse function theorem in the following form.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 6.27</span></p>

Let $N$ be a manifold of dimension $n$. A set of $n$ smooth functions $F^1, \dots, F^n$ defined on a coordinate neighborhood $(U, x^1, \dots, x^n)$ of a point $p \in N$ forms a coordinate system about $p$ if and only if the Jacobian determinant $\det[\partial F^i/\partial x^j(p)]$ is nonzero.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $F = (F^1, \dots, F^n) \colon U \to \mathbb{R}^n$. Then

$\det[\partial F^i/\partial x^j(p)] \neq 0$

$\iff$ $F \colon U \to \mathbb{R}^n$ is locally invertible at $p$ (by the inverse function theorem)

$\iff$ there is a neighborhood $W$ of $p$ in $N$ such that $F \colon W \to F(W)$ is a diffeomorphism (by the definition of local invertibility)

$\iff$ $(W, F^1, \dots, F^n)$ is a coordinate chart about $p$ in the differentiable structure of $N$ (by Proposition 6.11). $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Local Coordinate System)</span></p>

Find all points in $\mathbb{R}^2$ in a neighborhood of which the functions $x^2 + y^2 - 1, y$ can serve as a local coordinate system.

Define $F \colon \mathbb{R}^2 \to \mathbb{R}^2$ by $F(x, y) = (x^2 + y^2 - 1, y)$. The map $F$ can serve as a coordinate map in a neighborhood of $p$ if and only if it is a local diffeomorphism at $p$. The Jacobian determinant of $F$ is

$$\frac{\partial(F^1, F^2)}{\partial(x, y)} = \det\begin{bmatrix} 2x & 2y \\ 0 & 1 \end{bmatrix} = 2x.$$

By the inverse function theorem, $F$ is a local diffeomorphism at $p = (x, y)$ if and only if $x \neq 0$. Thus, $F$ can serve as a coordinate system at any point $p$ not on the $y$-axis.

</div>

### Lie Groups

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.20</span><span class="math-callout__name">(Lie Group)</span></p>

A **Lie group** is a $C^\infty$ manifold $G$ having a group structure such that the multiplication map

$$\mu \colon G \times G \to G$$

and the inverse map

$$\iota \colon G \to G, \quad \iota(x) = x^{-1},$$

are both $C^\infty$.

Similarly, a **topological group** is a topological space having a group structure such that the multiplication and inverse maps are both continuous. Note that a topological group is required to be a topological space, but not a topological manifold.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Examples of Lie Groups)</span></p>

* The Euclidean space $\mathbb{R}^n$ is a Lie group under addition.
* The set $\mathbb{C}^\times$ of nonzero complex numbers is a Lie group under multiplication.
* The unit circle $S^1$ in $\mathbb{C}^\times$ is a Lie group under multiplication.
* The Cartesian product $G_1 \times G_2$ of two Lie groups $(G_1, \mu_1)$ and $(G_2, \mu_2)$ is a Lie group under coordinatewise multiplication $\mu_1 \times \mu_2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.21</span><span class="math-callout__name">(General Linear Group as a Lie Group)</span></p>

In Example 5.15 we defined the general linear group

$$\operatorname{GL}(n, \mathbb{R}) = \lbrace A = [a_{ij}] \in \mathbb{R}^{n \times n} \mid \det A \neq 0 \rbrace.$$

As an open subset of $\mathbb{R}^{n \times n}$, it is a manifold. Since the $(i,j)$-entry of the product of two matrices $A$ and $B$ in $\operatorname{GL}(n, \mathbb{R})$,

$$(AB)_{ij} = \sum_{k=1}^n a_{ik} b_{kj},$$

is a polynomial in the coordinates of $A$ and $B$, matrix multiplication $\mu \colon \operatorname{GL}(n, \mathbb{R}) \times \operatorname{GL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R})$ is a $C^\infty$ map.

By Cramer's rule, the $(i,j)$-entry of $A^{-1}$ is

$$(A^{-1})_{ij} = \frac{1}{\det A} \cdot (-1)^{i+j}((j,i)\text{-minor of } A),$$

which is a $C^\infty$ function of the $a_{ij}$'s provided $\det A \neq 0$. Therefore, the inverse map $\iota \colon \operatorname{GL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R})$ is also $C^\infty$. This proves that $\operatorname{GL}(n, \mathbb{R})$ is a Lie group.

</div>

## §7 Quotients

Gluing the edges of a malleable square is one way to create new surfaces. For example, gluing together the top and bottom edges of a square gives a cylinder; gluing together the boundaries of the cylinder with matching orientations gives a torus. This gluing process is called an *identification* or a *quotient construction*.

The quotient construction is a process of simplification. Starting with an equivalence relation on a set, we identify each equivalence class to a point. Mathematics abounds in quotient constructions, for example, the quotient group, quotient ring, or quotient vector space in algebra. If the original set is a topological space, it is always possible to give the quotient set a topology so that the natural projection map becomes continuous. However, even if the original space is a manifold, a quotient space is often not a manifold. The main results of this section give conditions under which a quotient space remains second countable and Hausdorff. We then study real projective space as an example of a quotient manifold.

### 7.1 The Quotient Topology

Recall that an equivalence relation on a set $S$ is a reflexive, symmetric, and transitive relation. The *equivalence class* $[x]$ of $x \in S$ is the set of all elements in $S$ equivalent to $x$. An equivalence relation on $S$ partitions $S$ into disjoint equivalence classes. We denote the set of equivalence classes by $S/{\sim}$ and call this set the *quotient* of $S$ by the equivalence relation $\sim$. There is a natural *projection map* $\pi \colon S \to S/{\sim}$ that sends $x \in S$ to its equivalence class $[x]$.

Assume now that $S$ is a topological space. We define a topology on $S/{\sim}$ by declaring a set $U$ in $S/{\sim}$ to be *open* if and only if $\pi^{-1}(U)$ is open in $S$. Clearly, both the empty set $\varnothing$ and the entire quotient $S/{\sim}$ are open. Further, since

$$\pi^{-1}\!\left(\bigcup_\alpha U_\alpha\right) = \bigcup_\alpha \pi^{-1}(U_\alpha)$$

and

$$\pi^{-1}\!\left(\bigcap_i U_i\right) = \bigcap_i \pi^{-1}(U_i),$$

the collection of open sets in $S/{\sim}$ is closed under arbitrary unions and finite intersections, and is therefore a topology. It is called the **quotient topology** on $S/{\sim}$. With this topology, $S/{\sim}$ is called the **quotient space** of $S$ by the equivalence relation $\sim$. The projection map $\pi \colon S \to S/{\sim}$ is automatically continuous, because the inverse image of an open set in $S/{\sim}$ is by definition open in $S$.

### 7.2 Continuity of a Map on a Quotient

Let $\sim$ be an equivalence relation on the topological space $S$ and give $S/{\sim}$ the quotient topology. Suppose a function $f \colon S \to Y$ from $S$ to another topological space $Y$ is constant on each equivalence class. Then it induces a map $\bar{f} \colon S/{\sim} \to Y$ by

$$\bar{f}([p]) = f(p) \quad \text{for } p \in S.$$

In other words, there is a commutative diagram: $f = \bar{f} \circ \pi$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.1</span></p>

The induced map $\bar{f} \colon S/{\sim} \to Y$ is continuous if and only if the map $f \colon S \to Y$ is continuous.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$(\Rightarrow)$ If $\bar{f}$ is continuous, then as the composite $\bar{f} \circ \pi$ of continuous functions, $f$ is also continuous.

$(\Leftarrow)$ Suppose $f$ is continuous. Let $V$ be open in $Y$. Then $f^{-1}(V) = \pi^{-1}(\bar{f}^{-1}(V))$ is open in $S$. By the definition of quotient topology, $\bar{f}^{-1}(V)$ is open in $S/{\sim}$. Since $V$ was arbitrary, $\bar{f} \colon S/{\sim} \to Y$ is continuous. $\square$

</details>
</div>

This proposition gives a useful criterion for checking whether a function $\bar{f}$ on a quotient space $S/{\sim}$ is continuous: simply lift the function $\bar{f}$ to $f := \bar{f} \circ \pi$ on $S$ and check the continuity of the lifted map $f$ on $S$.

### 7.3 Identification of a Subset to a Point

If $A$ is a subspace of a topological space $S$, we can define a relation $\sim$ on $S$ by declaring $x \sim x$ for all $x \in S$ (so the relation is reflexive) and $x \sim y$ for all $x, y \in A$. This is an equivalence relation on $S$. We say that the quotient space $S/{\sim}$ is obtained from $S$ by **identifying $A$ to a point**.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.2</span></p>

Let $I$ be the unit interval $[0, 1]$ and $I/{\sim}$ the quotient space obtained from $I$ by identifying the two points $\lbrace 0, 1 \rbrace$ to a point. Denote by $S^1$ the unit circle in the complex plane. The function $f \colon I \to S^1$, $f(x) = \exp(2\pi i x)$, assumes the same value at $0$ and $1$, and so induces a function $\bar{f} \colon I/{\sim} \to S^1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.3</span></p>

The function $\bar{f} \colon I/{\sim} \to S^1$ is a homeomorphism.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since $f$ is continuous, $\bar{f}$ is also continuous by Proposition 7.1. Clearly, $\bar{f}$ is a bijection. As the continuous image of the compact set $I$, the quotient $I/{\sim}$ is compact. Thus, $\bar{f}$ is a continuous bijection from the compact space $I/{\sim}$ to the Hausdorff space $S^1$. By Corollary A.36, $\bar{f}$ is a homeomorphism. $\square$

</details>
</div>

### 7.4 A Necessary Condition for a Hausdorff Quotient

The quotient construction does not in general preserve the Hausdorff property or second countability. Indeed, since every singleton set in a Hausdorff space is closed, if $\pi \colon S \to S/{\sim}$ is the projection and the quotient $S/{\sim}$ is Hausdorff, then for any $p \in S$, its image $\lbrace \pi(p) \rbrace$ is closed in $S/{\sim}$. By the continuity of $\pi$, the inverse image $\pi^{-1}(\lbrace \pi(p) \rbrace) = [p]$ is closed in $S$. This gives a necessary condition for a quotient space to be Hausdorff.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.4</span></p>

If the quotient space $S/{\sim}$ is Hausdorff, then the equivalence class $[p]$ of any point $p$ in $S$ is closed in $S$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Define an equivalence relation $\sim$ on $\mathbb{R}$ by identifying the open interval $]0, \infty[$ to a point. Then the quotient space $\mathbb{R}/{\sim}$ is not Hausdorff because the equivalence class $]0, \infty[$ of $\sim$ in $\mathbb{R}$ corresponding to the point $]0, \infty[$ in $\mathbb{R}/{\sim}$ is not a closed subset of $\mathbb{R}$.

</div>

### 7.5 Open Equivalence Relations

In this section we derive conditions under which a quotient space is Hausdorff or second countable. Recall that a map $f \colon X \to Y$ of topological spaces is *open* if the image of any open set under $f$ is open.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.5</span></p>

An equivalence relation $\sim$ on a topological space $S$ is said to be **open** if the projection map $\pi \colon S \to S/{\sim}$ is open.

</div>

In other words, the equivalence relation $\sim$ on $S$ is open if and only if for every open set $U$ in $S$, the set

$$\pi^{-1}(\pi(U)) = \bigcup_{x \in U} [x]$$

of all points equivalent to some point of $U$ is open.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.6</span></p>

The projection map to a quotient space is in general not open. For example, let $\sim$ be the equivalence relation on $\mathbb{R}$ that identifies the two points $1$ and $-1$, and $\pi \colon \mathbb{R} \to \mathbb{R}/{\sim}$ the projection map. The map $\pi$ is open if and only if for every open set $V$ in $\mathbb{R}$, its image $\pi(V)$ is open in $\mathbb{R}/{\sim}$, which by the definition of the quotient topology means that $\pi^{-1}(\pi(V))$ is open in $\mathbb{R}$. Now let $V$ be the open interval $]-2, 0[$. Then

$$\pi^{-1}(\pi(V)) = {]{-2}, 0[} \cup \lbrace 1 \rbrace,$$

which is not open in $\mathbb{R}$. Therefore, the projection map $\pi \colon \mathbb{R} \to \mathbb{R}/{\sim}$ is not an open map.

</div>

Given an equivalence relation $\sim$ on $S$, let $R$ be the subset of $S \times S$ that defines the relation

$$R = \lbrace (x, y) \in S \times S \mid x \sim y \rbrace.$$

We call $R$ the **graph** of the equivalence relation $\sim$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.7</span></p>

Suppose $\sim$ is an open equivalence relation on a topological space $S$. Then the quotient space $S/{\sim}$ is Hausdorff if and only if the graph $R$ of $\sim$ is closed in $S \times S$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

There is a sequence of equivalent statements:

$R$ is closed in $S \times S$

$\iff$ $(S \times S) - R$ is open in $S \times S$

$\iff$ for every $(x, y) \in S \times S - R$, there is a basic open set $U \times V$ containing $(x, y)$ such that $(U \times V) \cap R = \varnothing$

$\iff$ for every pair $x \not\sim y$ in $S$, there exist neighborhoods $U$ of $x$ and $V$ of $y$ in $S$ such that no element of $U$ is equivalent to an element of $V$

$\iff$ for any two points $[x] \neq [y]$ in $S/{\sim}$, there exist neighborhoods $U$ of $x$ and $V$ of $y$ in $S$ such that $\pi(U) \cap \pi(V) = \varnothing$ in $S/{\sim}$. $(\ast)$

We now show that this last statement $(\ast)$ is equivalent to $S/{\sim}$ being Hausdorff. First assume $(\ast)$. Since $\sim$ is an open equivalence relation, $\pi(U)$ and $\pi(V)$ are disjoint open sets in $S/{\sim}$ containing $[x]$ and $[y]$ respectively. Therefore, $S/{\sim}$ is Hausdorff.

Conversely, suppose $S/{\sim}$ is Hausdorff. Let $[x] \neq [y]$ in $S/{\sim}$. Then there exist disjoint open sets $A$ and $B$ in $S/{\sim}$ such that $[x] \in A$ and $[y] \in B$. By the surjectivity of $\pi$, we have $A = \pi(\pi^{-1}A)$ and $B = \pi(\pi^{-1}B)$ (see Problem 7.1). Let $U = \pi^{-1}A$ and $V = \pi^{-1}B$. Then $x \in U$, $y \in V$, and $A = \pi(U)$ and $B = \pi(V)$ are disjoint open sets in $S/{\sim}$. $\square$

</details>
</div>

If the equivalence relation $\sim$ is equality, then the quotient space $S/{\sim}$ is $S$ itself and the graph $R$ of $\sim$ is simply the diagonal $\Delta = \lbrace (x, x) \in S \times S \rbrace$. In this case, Theorem 7.7 becomes the following well-known characterization of a Hausdorff space by its diagonal.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 7.8</span></p>

A topological space $S$ is Hausdorff if and only if the diagonal $\Delta$ in $S \times S$ is closed.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.9</span></p>

Let $\sim$ be an open equivalence relation on a topological space $S$ with projection $\pi \colon S \to S/{\sim}$. If $\mathcal{B} = \lbrace B_\alpha \rbrace$ is a basis for $S$, then its image $\lbrace \pi(B_\alpha) \rbrace$ under $\pi$ is a basis for $S/{\sim}$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since $\pi$ is an open map, $\lbrace \pi(B_\alpha) \rbrace$ is a collection of open sets in $S/{\sim}$. Let $W$ be an open set in $S/{\sim}$ and $[x] \in W$, $x \in S$. Then $x \in \pi^{-1}(W)$. Since $\pi^{-1}(W)$ is open, there is a basic open set $B \in \mathcal{B}$ such that $x \in B \subset \pi^{-1}(W)$. Then $[x] = \pi(x) \in \pi(B) \subset W$, which proves that $\lbrace \pi(B_\alpha) \rbrace$ is a basis for $S/{\sim}$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 7.10</span></p>

If $\sim$ is an open equivalence relation on a second-countable space $S$, then the quotient space $S/{\sim}$ is second countable.

</div>

### 7.6 Real Projective Space

Define an equivalence relation on $\mathbb{R}^{n+1} - \lbrace 0 \rbrace$ by

$$x \sim y \iff y = tx \text{ for some nonzero real number } t,$$

where $x, y \in \mathbb{R}^{n+1} - \lbrace 0 \rbrace$. The **real projective space** $\mathbb{R}P^n$ is the quotient space of $\mathbb{R}^{n+1} - \lbrace 0 \rbrace$ by this equivalence relation. We denote the equivalence class of a point $(a^0, \dots, a^n) \in \mathbb{R}^{n+1} - \lbrace 0 \rbrace$ by $[a^0, \dots, a^n]$ and let $\pi \colon \mathbb{R}^{n+1} - \lbrace 0 \rbrace \to \mathbb{R}P^n$ be the projection. We call $[a^0, \dots, a^n]$ **homogeneous coordinates** on $\mathbb{R}P^n$.

Geometrically, two nonzero points in $\mathbb{R}^{n+1}$ are equivalent if and only if they lie on the same line through the origin, so $\mathbb{R}P^n$ can be interpreted as the set of all lines through the origin in $\mathbb{R}^{n+1}$. Each line through the origin in $\mathbb{R}^{n+1}$ meets the unit sphere $S^n$ in a pair of antipodal points. This suggests that we define an equivalence relation $\sim$ on $S^n$ by identifying antipodal points:

$$x \sim y \iff x = \pm y, \quad x, y \in S^n.$$

We then have a bijection $\mathbb{R}P^n \leftrightarrow S^n/{\sim}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.12</span><span class="math-callout__name">(The real projective line $\mathbb{R}P^1$)</span></p>

Each line through the origin in $\mathbb{R}^2$ meets the unit circle in a pair of antipodal points. By Exercise 7.11, $\mathbb{R}P^1$ is homeomorphic to the quotient $S^1/{\sim}$, which is in turn homeomorphic to the closed upper semicircle with the two endpoints identified. Thus, $\mathbb{R}P^1$ is homeomorphic to $S^1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.13</span><span class="math-callout__name">(The real projective plane $\mathbb{R}P^2$)</span></p>

By Exercise 7.11, there is a homeomorphism $\mathbb{R}P^2 \simeq S^2 / \lbrace \text{antipodal points} \rbrace = S^2/{\sim}$. For points not on the equator, each pair of antipodal points contains a unique point in the upper hemisphere. Thus, there is a bijection between $S^2/{\sim}$ and the quotient of the closed upper hemisphere in which each pair of antipodal points on the equator is identified.

Let $H^2$ be the closed upper hemisphere

$$H^2 = \lbrace (x, y, z) \in \mathbb{R}^3 \mid x^2 + y^2 + z^2 = 1,\; z \ge 0 \rbrace$$

and let $D^2$ be the closed unit disk

$$D^2 = \lbrace (x, y) \in \mathbb{R}^2 \mid x^2 + y^2 \le 1 \rbrace.$$

These two spaces are homeomorphic to each other via the continuous map $\varphi \colon H^2 \to D^2$, $\varphi(x, y, z) = (x, y)$, and its inverse $\psi \colon D^2 \to H^2$, $\psi(x, y) = (x, y, \sqrt{1 - x^2 - y^2})$.

On $H^2$, define an equivalence relation $\sim$ by identifying the antipodal points on the equator: $(x, y, 0) \sim (-x, -y, 0)$ for $x^2 + y^2 = 1$. On $D^2$, define an equivalence relation $\sim$ by identifying the antipodal points on the boundary circle: $(x, y) \sim (-x, -y)$ for $x^2 + y^2 = 1$. Then $\varphi$ and $\psi$ induce homeomorphisms $\bar{\varphi} \colon H^2/{\sim} \to D^2/{\sim}$ and $\bar{\psi} \colon D^2/{\sim} \to H^2/{\sim}$.

In summary, there is a sequence of homeomorphisms

$$\mathbb{R}P^2 \xrightarrow{\sim} S^2/{\sim} \xrightarrow{\sim} H^2/{\sim} \xrightarrow{\sim} D^2/{\sim}$$

that identifies the real projective plane as the quotient of the closed disk $D^2$ with the antipodal points on its boundary identified.

The real projective plane $\mathbb{R}P^2$ cannot be embedded as a submanifold of $\mathbb{R}^3$. However, if we allow self-intersection, then we can map $\mathbb{R}P^2$ into $\mathbb{R}^3$ as a cross-cap. This map is not one-to-one.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.14</span></p>

The equivalence relation $\sim$ on $\mathbb{R}^{n+1} - \lbrace 0 \rbrace$ in the definition of $\mathbb{R}P^n$ is an open equivalence relation.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For an open set $U \subset \mathbb{R}^{n+1} - \lbrace 0 \rbrace$, the image $\pi(U)$ is open in $\mathbb{R}P^n$ if and only if $\pi^{-1}(\pi(U))$ is open in $\mathbb{R}^{n+1} - \lbrace 0 \rbrace$. But $\pi^{-1}(\pi(U))$ consists of all nonzero scalar multiples of points of $U$; that is,

$$\pi^{-1}(\pi(U)) = \bigcup_{t \in \mathbb{R}^\times} tU = \bigcup_{t \in \mathbb{R}^\times} \lbrace tp \mid p \in U \rbrace.$$

Since multiplication by $t \in \mathbb{R}^\times$ is a homeomorphism of $\mathbb{R}^{n+1} - \lbrace 0 \rbrace$, the set $tU$ is open for any $t$. Therefore, their union $\bigcup_{t \in \mathbb{R}^\times} tU = \pi^{-1}(\pi(U))$ is also open. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 7.15</span></p>

The real projective space $\mathbb{R}P^n$ is second countable.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Apply Corollary 7.10. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.16</span></p>

The real projective space $\mathbb{R}P^n$ is Hausdorff.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $S = \mathbb{R}^{n+1} - \lbrace 0 \rbrace$ and consider the set

$$R = \lbrace (x, y) \in S \times S \mid y = tx \text{ for some } t \in \mathbb{R}^\times \rbrace.$$

If we write $x$ and $y$ as column vectors, then $[x\ y]$ is an $(n+1) \times 2$ matrix, and $R$ may be characterized as the set of matrices $[x\ y]$ in $S \times S$ of rank $\le 1$. By a standard fact from linear algebra, $\operatorname{rk}[x\ y] \le 1$ is equivalent to the vanishing of all $2 \times 2$ minors of $[x\ y]$ (see Problem B.1). As the zero set of finitely many polynomials, $R$ is a closed subset of $S \times S$. Since $\sim$ is an open equivalence relation on $S$, and $R$ is closed in $S \times S$, by Theorem 7.7 the quotient $S/{\sim} \simeq \mathbb{R}P^n$ is Hausdorff. $\square$

</details>
</div>

### 7.7 The Standard $C^\infty$ Atlas on a Real Projective Space

Let $[a^0, \dots, a^n]$ be homogeneous coordinates on the projective space $\mathbb{R}P^n$. Although $a^0$ is not a well-defined function on $\mathbb{R}P^n$, the condition $a^0 \neq 0$ is independent of the choice of a representative for $[a^0, \dots, a^n]$. Hence, the condition $a^0 \neq 0$ makes sense on $\mathbb{R}P^n$, and we may define

$$U_0 := \lbrace [a^0, \dots, a^n] \in \mathbb{R}P^n \mid a^0 \neq 0 \rbrace.$$

Similarly, for each $i = 1, \dots, n$, let

$$U_i := \lbrace [a^0, \dots, a^n] \in \mathbb{R}P^n \mid a^i \neq 0 \rbrace.$$

Define $\phi_0 \colon U_0 \to \mathbb{R}^n$ by

$$[a^0, \dots, a^n] \mapsto \left(\frac{a^1}{a^0}, \dots, \frac{a^n}{a^0}\right).$$

This map has a continuous inverse $(b^1, \dots, b^n) \mapsto [1, b^1, \dots, b^n]$ and is therefore a homeomorphism. Similarly, there are homeomorphisms for each $i = 1, \dots, n$:

$$\phi_i \colon U_i \to \mathbb{R}^n, \quad [a^0, \dots, a^n] \mapsto \left(\frac{a^0}{a^i}, \dots, \widehat{\frac{a^i}{a^i}}, \dots, \frac{a^n}{a^i}\right),$$

where the caret sign $\widehat{\phantom{x}}$ over $a^i/a^i$ means that that entry is to be omitted. This proves that $\mathbb{R}P^n$ is locally Euclidean with the $(U_i, \phi_i)$ as charts.

On the intersection $U_0 \cap U_1$, we have $a^0 \neq 0$ and $a^1 \neq 0$, and there are two coordinate systems. We refer to the coordinate functions on $U_0$ as $x^1, \dots, x^n$, and the coordinate functions on $U_1$ as $y^1, \dots, y^n$. On $U_0$,

$$x^i = \frac{a^i}{a^0}, \quad i = 1, \dots, n,$$

and on $U_1$,

$$y^1 = \frac{a^0}{a^1}, \quad y^2 = \frac{a^2}{a^1}, \quad \dots, \quad y^n = \frac{a^n}{a^1}.$$

Then on $U_0 \cap U_1$,

$$y^1 = \frac{1}{x^1}, \quad y^2 = \frac{x^2}{x^1}, \quad y^3 = \frac{x^3}{x^1}, \quad \dots, \quad y^n = \frac{x^n}{x^1},$$

so

$$(\phi_1 \circ \phi_0^{-1})(x) = \left(\frac{1}{x^1}, \frac{x^2}{x^1}, \frac{x^3}{x^1}, \dots, \frac{x^n}{x^1}\right).$$

This is a $C^\infty$ function because $x^1 \neq 0$ on $\phi_0(U_0 \cap U_1)$. On any other $U_i \cap U_j$ an analogous formula holds. Therefore, the collection $\lbrace (U_i, \phi_i) \rbrace_{i=0,\dots,n}$ is a $C^\infty$ atlas for $\mathbb{R}P^n$, called the **standard atlas**. This concludes the proof that $\mathbb{R}P^n$ is a $C^\infty$ manifold.

# Chapter 3 — The Tangent Space

By definition, the tangent space to a manifold at a point is the vector space of derivations at the point. A smooth map of manifolds induces a linear map, called its *differential*, of tangent spaces at corresponding points. In local coordinates, the differential is represented by the Jacobian matrix of partial derivatives of the map. In this sense, the differential of a map between manifolds is a generalization of the derivative of a map between Euclidean spaces.

A basic principle in manifold theory is the linearization principle, according to which a manifold can be approximated near a point by its tangent space at the point, and a smooth map can be approximated by the differential of the map. In this way, one turns a topological problem into a linear problem.

Using the differential, we classify maps having maximal rank at a point into immersions and submersions at the point, depending on whether the differential is injective or surjective there. A point where the differential is surjective is a *regular point* of the map. The regular level set theorem states that a level set all of whose points are regular is a regular submanifold, i.e., a subset that locally looks like a coordinate $k$-plane in $\mathbb{R}^n$. This theorem gives a powerful tool for proving that a topological space is a manifold.

## §8 The Tangent Space

In Section 2 we saw that for any point $p$ in an open set $U$ in $\mathbb{R}^n$ there are two equivalent ways to define a tangent vector at $p$: as an arrow, represented by a column vector, or as a point-derivation of $C_p^\infty$, the algebra of germs of $C^\infty$ functions at $p$.

Both definitions generalize to a manifold. In the arrow approach, one defines a tangent vector at $p$ in a manifold $M$ by first choosing a chart $(U, \phi)$ at $p$ and then decreeing a tangent vector at $p$ to be an arrow at $\phi(p)$ in $\phi(U)$. This approach, while more visual, is complicated to work with, since a different chart $(V, \psi)$ at $p$ would give rise to a different set of tangent vectors at $p$ and one would have to decide how to identify the arrows at $\phi(p)$ in $U$ with the arrows at $\psi(p)$ in $\psi(V)$.

The cleanest and most intrinsic definition of a tangent vector at $p$ in $M$ is as a point-derivation, and this is the approach we adopt.

### 8.1 The Tangent Space at a Point

Just as for $\mathbb{R}^n$, we define a *germ* of a $C^\infty$ function at $p$ in $M$ to be an equivalence class of $C^\infty$ functions defined in a neighborhood of $p$ in $M$, two such functions being equivalent if they agree on some, possibly smaller, neighborhood of $p$. The set of germs of $C^\infty$ real-valued functions at $p$ in $M$ is denoted by $C_p^\infty(M)$. The addition and multiplication of functions make $C_p^\infty(M)$ into a ring; with scalar multiplication by real numbers, $C_p^\infty(M)$ becomes an algebra over $\mathbb{R}$.

Generalizing a derivation at a point in $\mathbb{R}^n$, we define a *derivation at a point* in a manifold $M$, or a *point-derivation* of $C_p^\infty(M)$, to be a linear map $D \colon C_p^\infty(M) \to \mathbb{R}$ such that

$$D(fg) = (Df)g(p) + f(p)Dg.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.1</span><span class="math-callout__name">(Tangent Vector)</span></p>

A **tangent vector** at a point $p$ in a manifold $M$ is a derivation at $p$.

</div>

Just as for $\mathbb{R}^n$, the tangent vectors at $p$ form a vector space $T_p(M)$, called the **tangent space** of $M$ at $p$. We also write $T_pM$ instead of $T_p(M)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.2</span><span class="math-callout__name">(Tangent space to an open subset)</span></p>

If $U$ is an open set containing $p$ in $M$, then the algebra $C_p^\infty(U)$ of germs of $C^\infty$ functions in $U$ at $p$ is the same as $C_p^\infty(M)$. Hence, $T_pU = T_pM$.

</div>

Given a coordinate neighborhood $(U, \phi) = (U, x^1, \dots, x^n)$ about a point $p$ in a manifold $M$, we recall the definition of the partial derivatives $\partial/\partial x^i$ first introduced in Section 6. Let $r^1, \dots, r^n$ be the standard coordinates on $\mathbb{R}^n$. Then $x^i = r^i \circ \phi \colon U \to \mathbb{R}$. If $f$ is a smooth function in a neighborhood of $p$, we set

$$\frac{\partial}{\partial x^i}\bigg|_p f = \frac{\partial}{\partial r^i}\bigg|_{\phi(p)} (f \circ \phi^{-1}) \in \mathbb{R}.$$

It is easily checked that $\partial/\partial x^i|_p$ satisfies the derivation property and so is a tangent vector at $p$.

When $M$ is one-dimensional and $t$ is a local coordinate, it is customary to write $d/dt|_p$ instead of $\partial/\partial t|_p$ for the coordinate vector at the point $p$.

### 8.2 The Differential of a Map

Let $F \colon N \to M$ be a $C^\infty$ map between two manifolds. At each point $p \in N$, the map $F$ induces a linear map of tangent spaces, called its **differential at $p$**,

$$F_{*} \colon T_pN \to T_{F(p)}M$$

as follows. If $X_p \in T_pN$, then $F_*(X_p)$ is the tangent vector in $T_{F(p)}M$ defined by

$$(F_*(X_p))f = X_p(f \circ F) \in \mathbb{R} \quad \text{for } f \in C_{F(p)}^\infty(M). \tag{8.1}$$

Here $f$ is a germ at $F(p)$, represented by a $C^\infty$ function in a neighborhood of $F(p)$. Since (8.1) is independent of the representative of the germ, in practice we can be cavalier about the distinction between a germ and a representative function for the germ.

To make the dependence on $p$ explicit we sometimes write $F_{*,p}$ instead of $F_*$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.4</span><span class="math-callout__name">(Differential of a map between Euclidean spaces)</span></p>

Suppose $F \colon \mathbb{R}^n \to \mathbb{R}^m$ is smooth and $p$ is a point in $\mathbb{R}^n$. Let $x^1, \dots, x^n$ be the coordinates on $\mathbb{R}^n$ and $y^1, \dots, y^m$ the coordinates on $\mathbb{R}^m$. Then the tangent vectors $\partial/\partial x^1|_p, \dots, \partial/\partial x^n|_p$ form a basis for the tangent space $T_p(\mathbb{R}^n)$ and $\partial/\partial y^1|_{F(p)}, \dots, \partial/\partial y^m|_{F(p)}$ form a basis for the tangent space $T_{F(p)}(\mathbb{R}^m)$. The linear map $F_* \colon T_p(\mathbb{R}^n) \to T_{F(p)}(\mathbb{R}^m)$ is described by a matrix $[a_j^i]$ relative to these two bases:

$$F_*\left(\frac{\partial}{\partial x^j}\bigg|_p\right) = \sum_k a_j^k \frac{\partial}{\partial y^k}\bigg|_{F(p)}, \quad a_j^k \in \mathbb{R}. \tag{8.2}$$

Let $F^i = y^i \circ F$ be the $i$th component of $F$. We can find $a_j^i$ by evaluating both sides of (8.2) on $y^i$:

$$\text{RHS} = \sum_k a_j^k \frac{\partial}{\partial y^k}\bigg|_{F(p)} y^i = \sum_k a_j^k \delta_k^i = a_j^i,$$

$$\text{LHS} = F_*\left(\frac{\partial}{\partial x^j}\bigg|_p\right) y^i = \frac{\partial}{\partial x^j}\bigg|_p (y^i \circ F) = \frac{\partial F^i}{\partial x^j}(p).$$

So the matrix of $F_*$ relative to the bases $\lbrace \partial/\partial x^j|_p \rbrace$ and $\lbrace \partial/\partial y^i|_{F(p)} \rbrace$ is $[\partial F^i / \partial x^j(p)]$. This is precisely the Jacobian matrix of the derivative of $F$ at $p$. Thus, the differential of a map between manifolds generalizes the derivative of a map between Euclidean spaces.

</div>

### 8.3 The Chain Rule

Let $F \colon N \to M$ and $G \colon M \to P$ be smooth maps of manifolds, and $p \in N$. The differentials of $F$ at $p$ and $G$ at $F(p)$ are linear maps

$$T_pN \xrightarrow{F_{*,p}} T_{F(p)}M \xrightarrow{G_{*,F(p)}} T_{G(F(p))}P.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.5</span><span class="math-callout__name">(The chain rule)</span></p>

If $F \colon N \to M$ and $G \colon M \to P$ are smooth maps of manifolds and $p \in N$, then

$$(G \circ F)_{*,p} = G_{*,F(p)} \circ F_{*,p}.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $X_p \in T_pN$ and let $f$ be a smooth function at $G(F(p))$ in $P$. Then

$$((G \circ F)_* X_p) f = X_p(f \circ G \circ F)$$

and

$$((G_* \circ F_*) X_p) f = (G_*(F_* X_p)) f = (F_* X_p)(f \circ G) = X_p(f \circ G \circ F). \quad \square$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Differential of the identity)</span></p>

The differential of the identity map $\mathbb{1}_M \colon M \to M$ at any point $p$ in $M$ is the identity map $\mathbb{1}_{T_pM} \colon T_pM \to T_pM$, because $((\mathbb{1}_M)_* X_p) f = X_p(f \circ \mathbb{1}_M) = X_p f$ for any $X_p \in T_pM$ and $f \in C_p^\infty(M)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 8.6</span></p>

If $F \colon N \to M$ is a diffeomorphism of manifolds and $p \in N$, then $F_* \colon T_pN \to T_{F(p)}M$ is an isomorphism of vector spaces.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

To say that $F$ is a diffeomorphism means that it has a differentiable inverse $G \colon M \to N$ such that $G \circ F = \mathbb{1}_N$ and $F \circ G = \mathbb{1}_M$. By the chain rule,

$$(G \circ F)_* = G_* \circ F_* = (\mathbb{1}_N)_* = \mathbb{1}_{T_pN},$$

$$(F \circ G)_* = F_* \circ G_* = (\mathbb{1}_M)_* = \mathbb{1}_{T_{F(p)}M}.$$

Hence, $F_*$ and $G_*$ are isomorphisms. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 8.7</span><span class="math-callout__name">(Invariance of dimension)</span></p>

If an open set $U \subset \mathbb{R}^n$ is diffeomorphic to an open set $V \subset \mathbb{R}^m$, then $n = m$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $F \colon U \to V$ be a diffeomorphism and let $p \in U$. By Corollary 8.6, $F_{*,p} \colon T_pU \to T_{F(p)}V$ is an isomorphism of vector spaces. Since there are vector space isomorphisms $T_pU \simeq \mathbb{R}^n$ and $T_{F(p)}V \simeq \mathbb{R}^m$, we must have that $n = m$. $\square$

</details>
</div>

### 8.4 Bases for the Tangent Space at a Point

As usual, we denote by $r^1, \dots, r^n$ the standard coordinates on $\mathbb{R}^n$, and if $(U, \phi)$ is a chart about a point $p$ in a manifold $M$ of dimension $n$, we set $x^i = r^i \circ \phi$. Since $\phi \colon U \to \mathbb{R}^n$ is a diffeomorphism onto its image (Proposition 6.10), by Corollary 8.6 the differential

$$\phi_* \colon T_pM \to T_{\phi(p)}\mathbb{R}^n$$

is a vector space isomorphism. In particular, the tangent space $T_pM$ has the same dimension $n$ as the manifold $M$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.8</span></p>

Let $(U, \phi)$ be a chart and $x^i = r^i \circ \phi$ the coordinate functions. Then

$$\phi_*\left(\frac{\partial}{\partial x^i}\bigg|_p\right) = \frac{\partial}{\partial r^i}\bigg|_{\phi(p)}.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For any $f \in C_{\phi(p)}^\infty(\mathbb{R}^n)$,

$$\phi_*\left(\frac{\partial}{\partial x^i}\bigg|_p\right) f = \frac{\partial}{\partial x^i}\bigg|_p (f \circ \phi) = \frac{\partial}{\partial r^i}\bigg|_{\phi(p)} (f \circ \phi \circ \phi^{-1}) = \frac{\partial}{\partial r^i}\bigg|_{\phi(p)} f. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.9</span></p>

If $(U, \phi) = (U, x^1, \dots, x^n)$ is a chart containing $p$, then the tangent space $T_pM$ has basis

$$\frac{\partial}{\partial x^1}\bigg|_p, \dots, \frac{\partial}{\partial x^n}\bigg|_p.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

An isomorphism of vector spaces carries a basis to a basis. By Proposition 8.8 the isomorphism $\phi_* \colon T_pM \to T_{\phi(p)}(\mathbb{R}^n)$ maps $\partial/\partial x^1|_p, \dots, \partial/\partial x^n|_p$ to $\partial/\partial r^1|_{\phi(p)}, \dots, \partial/\partial r^n|_{\phi(p)}$, which is a basis for $T_{\phi(p)}(\mathbb{R}^n)$. Therefore, $\partial/\partial x^1|_p, \dots, \partial/\partial x^n|_p$ is a basis for $T_pM$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.10</span><span class="math-callout__name">(Transition matrix for coordinate vectors)</span></p>

Suppose $(U, x^1, \dots, x^n)$ and $(V, y^1, \dots, y^n)$ are two coordinate charts on a manifold $M$. Then

$$\frac{\partial}{\partial x^j} = \sum_i \frac{\partial y^i}{\partial x^j} \frac{\partial}{\partial y^i}$$

on $U \cap V$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

At each point $p \in U \cap V$, the sets $\lbrace \partial/\partial x^j|_p \rbrace$ and $\lbrace \partial/\partial y^i|_p \rbrace$ are both bases for the tangent space $T_pM$, so there is a matrix $[a_j^i(p)]$ of real numbers such that

$$\frac{\partial}{\partial x^j} = \sum_k a_j^k \frac{\partial}{\partial y^k}.$$

Applying both sides of the equation to $y^i$, we get

$$\frac{\partial y^i}{\partial x^j} = \sum_k a_j^k \frac{\partial y^i}{\partial y^k} = \sum_k a_j^k \delta_k^i = a_j^i. \quad \square$$

</details>
</div>

### 8.5 A Local Expression for the Differential

Given a smooth map $F \colon N \to M$ of manifolds and $p \in N$, let $(U, x^1, \dots, x^n)$ be a chart about $p$ in $N$ and let $(V, y^1, \dots, y^m)$ be a chart about $F(p)$ in $M$. We will find a local expression for the differential $F_{*,p} \colon T_pN \to T_{F(p)}M$ relative to the two charts.

By Proposition 8.9, $\lbrace \partial/\partial x^j|_p \rbrace_{j=1}^n$ is a basis for $T_pN$ and $\lbrace \partial/\partial y^i|_{F(p)} \rbrace_{i=1}^m$ is a basis for $T_{F(p)}M$. Therefore, the differential $F_* = F_{*,p}$ is completely determined by the numbers $a_j^i$ such that

$$F_*\left(\frac{\partial}{\partial x^j}\bigg|_p\right) = \sum_{k=1}^m a_j^k \frac{\partial}{\partial y^k}\bigg|_{F(p)}, \quad j = 1, \dots, n.$$

Applying both sides to $y^i$, we find that

$$a_j^i = F_*\left(\frac{\partial}{\partial x^j}\bigg|_p\right) y^i = \frac{\partial}{\partial x^j}\bigg|_p (y^i \circ F) = \frac{\partial F^i}{\partial x^j}(p).$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.11</span></p>

Given a smooth map $F \colon N \to M$ of manifolds and a point $p \in N$, let $(U, x^1, \dots, x^n)$ and $(V, y^1, \dots, y^m)$ be coordinate charts about $p$ in $N$ and $F(p)$ in $M$ respectively. Relative to the bases $\lbrace \partial/\partial x^j|_p \rbrace$ for $T_pN$ and $\lbrace \partial/\partial y^i|_{F(p)} \rbrace$ for $T_{F(p)}M$, the differential $F_{*,p} \colon T_pN \to T_{F(p)}M$ is represented by the matrix $[\partial F^i / \partial x^j(p)]$, where $F^i = y^i \circ F$ is the $i$th component of $F$.

</div>

This proposition is in the spirit of the "arrow" approach to tangent vectors. Here each tangent vector in $T_pN$ is represented by a column vector relative to the basis $\lbrace \partial/\partial x^j|_p \rbrace$, and the differential $F_{*,p}$ is represented by a matrix.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.12</span><span class="math-callout__name">(Inverse function theorem)</span></p>

In terms of the differential, the inverse function theorem for manifolds (Theorem 6.26) has a coordinate-free description: a $C^\infty$ map $F \colon N \to M$ between two manifolds of the same dimension is locally invertible at a point $p \in N$ if and only if its differential $F_{*,p} \colon T_pN \to T_{F(p)}M$ at $p$ is an isomorphism.

</div>

### 8.6 Curves in a Manifold

A **smooth curve** in a manifold $M$ is by definition a smooth map $c \colon ]a, b[ \to M$ from some open interval $]a, b[$ into $M$. Usually we assume $0 \in ]a, b[$ and say that $c$ is a *curve starting at $p$* if $c(0) = p$. The **velocity vector** $c'(t_0)$ of the curve $c$ at time $t_0 \in ]a, b[$ is defined to be

$$c'(t_0) := c_*\left(\frac{d}{dt}\bigg|_{t_0}\right) \in T_{c(t_0)}M.$$

We also say that $c'(t_0)$ is the velocity of $c$ at the point $c(t_0)$. Alternative notations for $c'(t_0)$ are $\frac{dc}{dt}(t_0)$ and $\frac{d}{dt}\big|_{t_0} c$.

Every smooth curve $c$ at $p$ in a manifold $M$ gives rise to a tangent vector $c'(0)$ in $T_pM$. Conversely, one can show that every tangent vector $X_p \in T_pM$ is the velocity vector of some curve at $p$, as follows.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.15</span><span class="math-callout__name">(Velocity of a curve in local coordinates)</span></p>

Let $c \colon ]a, b[ \to M$ be a smooth curve, and let $(U, x^1, \dots, x^n)$ be a coordinate chart about $c(t)$. Write $c^i = x^i \circ c$ for the $i$th component of $c$ in the chart. Then $c'(t)$ is given by

$$c'(t) = \sum_{i=1}^n \dot{c}^i(t) \frac{\partial}{\partial x^i}\bigg|_{c(t)}.$$

Thus, relative to the basis $\lbrace \partial/\partial x^i|_p \rbrace$ for $T_{c(t)}M$, the velocity $c'(t)$ is represented by the column vector $[\dot{c}^1(t), \dots, \dot{c}^n(t)]^T$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.16</span><span class="math-callout__name">(Existence of a curve with a given initial vector)</span></p>

For any point $p$ in a manifold $M$ and any tangent vector $X_p \in T_pM$, there are $\varepsilon > 0$ and a smooth curve $c \colon ]-\varepsilon, \varepsilon[ \to M$ such that $c(0) = p$ and $c'(0) = X_p$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $(U, \phi) = (U, x^1, \dots, x^n)$ be a chart centered at $p$; i.e., $\phi(p) = \mathbf{0} \in \mathbb{R}^n$. Suppose $X_p = \sum a^i \partial/\partial x^i|_p$ at $p$. Let $r^1, \dots, r^n$ be the standard coordinates on $\mathbb{R}^n$. Then $x^i = r^i \circ \phi$. To find a curve $c$ at $p$ with $c'(0) = X_p$, start with a curve $\alpha$ in $\mathbb{R}^n$ with $\alpha(0) = \mathbf{0}$ and $\alpha'(0) = \sum a^i \partial/\partial r^i|_{\mathbf{0}}$. We then map $\alpha$ to $M$ via $\phi^{-1}$. By Proposition 8.15, the simplest such $\alpha$ is

$$\alpha(t) = (a^1 t, \dots, a^n t), \quad t \in ]-\varepsilon, \varepsilon[,$$

where $\varepsilon$ is sufficiently small that $\alpha(t)$ lies in $\phi(U)$. Define $c = \phi^{-1} \circ \alpha \colon ]-\varepsilon, \varepsilon[ \to M$. Then $c(0) = \phi^{-1}(\alpha(0)) = \phi^{-1}(\mathbf{0}) = p$, and by Proposition 8.8,

$$c'(0) = (\phi^{-1})_* \alpha_*\left(\frac{d}{dt}\bigg|_{t=0}\right) = (\phi^{-1})_*\left(\sum a^i \frac{\partial}{\partial r^i}\bigg|_{\mathbf{0}}\right) = \sum a^i \frac{\partial}{\partial x^i}\bigg|_p = X_p. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.17</span></p>

Suppose $X_p$ is a tangent vector at a point $p$ of a manifold $M$ and $f \in C_p^\infty(M)$. If $c \colon ]-\varepsilon, \varepsilon[ \to M$ is a smooth curve starting at $p$ with $c'(0) = X_p$, then

$$X_p f = \frac{d}{dt}\bigg|_0 (f \circ c).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By the definitions of $c'(0)$ and $c_*$,

$$X_p f = c'(0) f = c_*\left(\frac{d}{dt}\bigg|_0\right) f = \frac{d}{dt}\bigg|_0 (f \circ c). \quad \square$$

</details>
</div>

In Definition 8.1 we defined a tangent vector at a point $p$ of a manifold abstractly as a derivation at $p$. Using curves, we can now interpret a tangent vector geometrically as a directional derivative.

### 8.7 Computing the Differential Using Curves

We have introduced two ways of computing the differential of a smooth map, in terms of derivations at a point (equation (8.1)) and in terms of local coordinates (Proposition 8.11). The next proposition gives still another way of computing the differential $F_{*,p}$, this time using curves.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.18</span></p>

Let $F \colon N \to M$ be a smooth map of manifolds, $p \in N$, and $X_p \in T_pN$. If $c$ is a smooth curve starting at $p$ in $N$ with velocity $X_p$ at $p$, then

$$F_{*,p}(X_p) = \frac{d}{dt}\bigg|_0 (F \circ c)(t).$$

In other words, $F_{*,p}(X_p)$ is the velocity vector of the image curve $F \circ c$ at $F(p)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By hypothesis, $c(0) = p$ and $c'(0) = X_p$. Then

$$F_{*,p}(X_p) = F_{*,p}(c'(0)) = (F_{*,p} \circ c_{*,0})\left(\frac{d}{dt}\bigg|_0\right) = (F \circ c)_{*,0}\left(\frac{d}{dt}\bigg|_0\right) = \frac{d}{dt}\bigg|_0 (F \circ c)(t). \quad \square$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.19</span><span class="math-callout__name">(Differential of left multiplication)</span></p>

If $g$ is a matrix in the general linear group $\operatorname{GL}(n, \mathbb{R})$, let $\ell_g \colon \operatorname{GL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R})$ be left multiplication by $g$; thus, $\ell_g(B) = gB$ for any $B \in \operatorname{GL}(n, \mathbb{R})$. Since $\operatorname{GL}(n, \mathbb{R})$ is an open subset of the vector space $\mathbb{R}^{n \times n}$, the tangent space $T_g(\operatorname{GL}(n, \mathbb{R}))$ can be identified with $\mathbb{R}^{n \times n}$. Show that with this identification the differential $(\ell_g)_{*,I} \colon T_I(\operatorname{GL}(n, \mathbb{R})) \to T_g(\operatorname{GL}(n, \mathbb{R}))$ is also left multiplication by $g$.

*Solution.* Let $X \in T_I(\operatorname{GL}(n, \mathbb{R})) = \mathbb{R}^{n \times n}$. To compute $(\ell_g)_{*,I}(X)$, choose a curve $c(t)$ in $\operatorname{GL}(n, \mathbb{R})$ with $c(0) = I$ and $c'(0) = X$. Then $\ell_g(c(t)) = gc(t)$ is simply matrix multiplication. By Proposition 8.18,

$$(\ell_g)_{*,I}(X) = \frac{d}{dt}\bigg|_{t=0} \ell_g(c(t)) = \frac{d}{dt}\bigg|_{t=0} gc(t) = gc'(0) = gX. \quad \square$$

</div>

### 8.8 Immersions and Submersions

Just as the derivative of a map between Euclidean spaces is a linear map that best approximates the given map at a point, so the differential at a point serves the same purpose for a $C^\infty$ map between manifolds. Two cases are especially important.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Immersion and Submersion)</span></p>

A $C^\infty$ map $F \colon N \to M$ is said to be an **immersion at $p$** $\in N$ if its differential $F_{*,p} \colon T_pN \to T_{F(p)}M$ is injective, and a **submersion at $p$** if $F_{*,p}$ is surjective. We call $F$ an **immersion** if it is an immersion at every $p \in N$ and a **submersion** if it is a submersion at every $p \in N$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.20</span></p>

Suppose $N$ and $M$ are manifolds of dimensions $n$ and $m$ respectively. Then $\dim T_pN = n$ and $\dim T_{F(p)}M = m$. The injectivity of the differential $F_{*,p} \colon T_pN \to T_{F(p)}M$ implies immediately that $n \le m$. Similarly, the surjectivity of the differential $F_{*,p}$ implies that $n \ge m$. Thus, if $F \colon N \to M$ is an immersion at a point of $N$, then $n \le m$ and if $F$ is a submersion at a point of $N$, then $n \ge m$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.21</span></p>

The prototype of an immersion is the inclusion of $\mathbb{R}^n$ in a higher-dimensional $\mathbb{R}^m$:

$$i(x^1, \dots, x^n) = (x^1, \dots, x^n, 0, \dots, 0).$$

The prototype of a submersion is the projection of $\mathbb{R}^n$ onto a lower-dimensional $\mathbb{R}^m$:

$$\pi(x^1, \dots, x^m, x^{m+1}, \dots, x^n) = (x^1, \dots, x^m).$$

If $U$ is an open subset of a manifold $M$, then the inclusion $i \colon U \to M$ is both an immersion and a submersion. This example shows in particular that a submersion need not be onto.

</div>

### 8.9 Rank, and Critical and Regular Points

The *rank* of a linear transformation $L \colon V \to W$ between finite-dimensional vector spaces is the dimension of the image $L(V)$ as a subspace of $W$, while the *rank* of a matrix $A$ is the dimension of its column space. If $L$ is represented by a matrix $A$ relative to a basis for $V$ and a basis for $W$, then the rank of $L$ is the same as the rank of $A$, because the image $L(V)$ is simply the column space of $A$.

Now consider a smooth map $F \colon N \to M$ of manifolds. Its *rank* at a point $p$ in $N$, denoted by $\operatorname{rk} F(p)$, is defined as the rank of the differential $F_{*,p} \colon T_pN \to T_{F(p)}M$. Relative to coordinate neighborhoods $(U, x^1, \dots, x^n)$ at $p$ and $(V, y^1, \dots, y^m)$ at $F(p)$, the differential is represented by the Jacobian matrix $[\partial F^i / \partial x^j(p)]$ (Proposition 8.11), so

$$\operatorname{rk} F(p) = \operatorname{rk}\left[\frac{\partial F^i}{\partial x^j}(p)\right].$$

Since the differential of a map is independent of coordinate charts, so is the rank of a Jacobian matrix.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.22</span><span class="math-callout__name">(Critical and Regular Points)</span></p>

A point $p$ in $N$ is a **critical point** of $F$ if the differential $F_{*,p} \colon T_pN \to T_{F(p)}M$ fails to be surjective. It is a **regular point** of $F$ if the differential $F_{*,p}$ is surjective. In other words, $p$ is a regular point of the map $F$ if and only if $F$ is a submersion at $p$.

A point in $M$ is a **critical value** if it is the image of a critical point; otherwise it is a **regular value**.

</div>

Two aspects of this definition merit elaboration:

1. We do *not* define a regular value to be the image of a regular point. In fact, a regular value need not be in the image of $F$ at all. Any point of $M$ not in the image of $F$ is automatically a regular value because it is not the image of a critical point.
2. A point $c$ in $M$ is a critical value if and only if *some* point in the preimage $F^{-1}(\lbrace c \rbrace)$ is a critical point. A point $c$ in the image of $F$ is a regular value if and only if *every* point in the preimage $F^{-1}(\lbrace c \rbrace)$ is a regular point.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.23</span></p>

For a real-valued function $f \colon M \to \mathbb{R}$, a point $p$ in $M$ is a critical point if and only if relative to some chart $(U, x^1, \dots, x^n)$ containing $p$, all the partial derivatives satisfy

$$\frac{\partial f}{\partial x^j}(p) = 0, \quad j = 1, \dots, n.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By Proposition 8.11 the differential $f_{*,p} \colon T_pM \to T_{f(p)}\mathbb{R} \simeq \mathbb{R}$ is represented by the matrix

$$\left[\frac{\partial f}{\partial x^1}(p) \quad \cdots \quad \frac{\partial f}{\partial x^n}(p)\right].$$

Since the image of $f_{*,p}$ is a linear subspace of $\mathbb{R}$, it is either zero-dimensional or one-dimensional. In other words, $f_{*,p}$ is either the zero map or a surjective map. Therefore, $f_{*,p}$ fails to be surjective if and only if all the partial derivatives $\partial f / \partial x^j(p)$ are zero. $\square$

</details>
</div>

## §9 Submanifolds

We now have two ways of showing that a given topological space is a manifold:

(a) by checking directly that the space is Hausdorff, second countable, and has a $C^\infty$ atlas;
(b) by exhibiting it as an appropriate quotient space. Section 7 lists some conditions under which a quotient space is a manifold.

In this section we introduce the concept of a *regular submanifold* of a manifold, a subset that is locally defined by the vanishing of some of the coordinate functions. Using the inverse function theorem, we derive a criterion, called the *regular level set theorem*, that can often be used to show that a level set of a $C^\infty$ map of manifolds is a regular submanifold and therefore a manifold.

### 9.1 Submanifolds

The $xy$-plane in $\mathbb{R}^3$ is the prototype of a *regular submanifold* of a manifold. It is defined by the vanishing of the coordinate function $z$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.1</span><span class="math-callout__name">(Regular Submanifold)</span></p>

A subset $S$ of a manifold $N$ of dimension $n$ is a **regular submanifold** of dimension $k$ if for every $p \in S$ there is a coordinate neighborhood $(U, \phi) = (U, x^1, \dots, x^n)$ of $p$ in the maximal atlas of $N$ such that $U \cap S$ is defined by the vanishing of $n - k$ of the coordinate functions. By renumbering the coordinates, we may assume that these $n - k$ coordinate functions are $x^{k+1}, \dots, x^n$.

</div>

We call such a chart $(U, \phi)$ an **adapted chart** relative to $S$. On $U \cap S$, $\phi = (x^1, \dots, x^k, 0, \dots, 0)$. Let

$$\phi_S \colon U \cap S \to \mathbb{R}^k$$

be the restriction of the first $k$ components of $\phi$ to $U \cap S$, that is, $\phi_S = (x^1, \dots, x^k)$. Note that $(U \cap S, \phi_S)$ is a chart for $S$ in the subspace topology.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.2</span><span class="math-callout__name">(Codimension)</span></p>

If $S$ is a regular submanifold of dimension $k$ in a manifold $N$ of dimension $n$, then $n - k$ is said to be the **codimension** of $S$ in $N$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

In the definition of a regular submanifold, the dimension $k$ of the submanifold may be equal to $n$, the dimension of the manifold. In this case, $U \cap S = U$. Therefore, an open subset of a manifold is a regular submanifold of the same dimension.

There are other types of submanifolds, but unless otherwise specified, by a "submanifold" we will always mean a "regular submanifold."

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Interval on the $x$-axis)</span></p>

The interval $S := ]-1, 1[$ on the $x$-axis is a regular submanifold of the $xy$-plane. As an adapted chart, we can take the open square $U = ]-1, 1[ \times ]-1, 1[$ with coordinates $x, y$. Then $U \cap S$ is precisely the zero set of $y$ on $U$.

Note that if $V = ]-2, 0[ \times ]-1, 1[$, then $(V, x, y)$ is not an adapted chart relative to $S$, since $V \cap S$ is the open interval $]-1, 0[$ on the $x$-axis, while the zero set of $y$ on $V$ is the open interval $]-2, 0[$ on the $x$-axis.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.3</span><span class="math-callout__name">(Topologist's sine curve)</span></p>

Let $\Gamma$ be the graph of the function $f(x) = \sin(1/x)$ on the interval $]0, 1[$, and let $S$ be the union of $\Gamma$ and the open interval $I = \lbrace (0, y) \in \mathbb{R}^2 \mid -1 < y < 1 \rbrace$. The closure of $\Gamma$ in $\mathbb{R}^2$ is called the *topologist's sine curve*.

The subset $S$ of $\mathbb{R}^2$ is not a regular submanifold for the following reason: if $p$ is in the interval $I$, then there is no adapted chart containing $p$, since any sufficiently small neighborhood $U$ of $p$ in $\mathbb{R}^2$ intersects $S$ in infinitely many components.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.4</span></p>

Let $S$ be a regular submanifold of $N$ and $\mathfrak{U} = \lbrace (U, \phi) \rbrace$ a collection of compatible adapted charts of $N$ that covers $S$. Then $\lbrace (U \cap S, \phi_S) \rbrace$ is an atlas for $S$. Therefore, a regular submanifold is itself a manifold. If $N$ has dimension $n$ and $S$ is locally defined by the vanishing of $n - k$ coordinates, then $\dim S = k$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $(U, \phi) = (U, x^1, \dots, x^n)$ and $(V, \psi) = (V, y^1, \dots, y^n)$ be two adapted charts in the given collection. Assume that they intersect. As remarked in Definition 9.1, in any adapted chart relative to a submanifold $S$ it is possible to renumber the coordinates so that the last $n - k$ coordinates vanish on points of $S$. Then for $p \in U \cap V \cap S$,

$$\phi(p) = (x^1, \dots, x^k, 0, \dots, 0) \quad \text{and} \quad \psi(p) = (y^1, \dots, y^k, 0, \dots, 0),$$

so $\phi_S(p) = (x^1, \dots, x^k)$ and $\psi_S(p) = (y^1, \dots, y^k)$. Therefore,

$$(\psi_S \circ \phi_S^{-1})(x^1, \dots, x^k) = (y^1, \dots, y^k).$$

Since $y^1, \dots, y^k$ are $C^\infty$ functions of $x^1, \dots, x^k$ (because $\psi \circ \phi^{-1}(x^1, \dots, x^k, 0, \dots, 0)$ is $C^\infty$), the transition function $\psi_S \circ \phi_S^{-1}$ is also $C^\infty$. Hence, any two charts in $\lbrace (U \cap S, \phi_S) \rbrace$ are $C^\infty$ compatible. Since $\lbrace U \cap S \rbrace_{U \in \mathfrak{U}}$ covers $S$, the collection $\lbrace (U \cap S, \phi_S) \rbrace$ is a $C^\infty$ atlas on $S$. $\square$

</details>
</div>

### 9.2 Level Sets of a Function

A **level set** of a map $F \colon N \to M$ is a subset

$$F^{-1}(\lbrace c \rbrace) = \lbrace p \in N \mid F(p) = c \rbrace$$

for some $c \in M$. The usual notation for a level set is $F^{-1}(c)$, rather than the more correct $F^{-1}(\lbrace c \rbrace)$. The value $c \in M$ is called the **level** of the level set $F^{-1}(c)$. If $F \colon N \to \mathbb{R}^m$, then $Z(F) := F^{-1}(\mathbf{0})$ is the **zero set** of $F$. Recall that $c$ is a regular value of $F$ if and only if either $c$ is not in the image of $F$ or at every point $p \in F^{-1}(c)$, the differential $F_{*,p} \colon T_pN \to T_{F(p)}M$ is surjective. The inverse image $F^{-1}(c)$ of a regular value $c$ is called a **regular level set**. If the zero set $F^{-1}(\mathbf{0})$ is a regular level set of $F \colon N \to \mathbb{R}^m$, it is called a **regular zero set**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.5</span></p>

If a regular level set $F^{-1}(c)$ is nonempty, say $p \in F^{-1}(c)$, then the map $F \colon N \to M$ is a submersion at $p$. By Remark 8.20, $\dim N \ge \dim M$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.6</span><span class="math-callout__name">(The 2-sphere in $\mathbb{R}^3$)</span></p>

The unit 2-sphere

$$S^2 = \lbrace (x, y, z) \in \mathbb{R}^3 \mid x^2 + y^2 + z^2 = 1 \rbrace$$

is the level set $g^{-1}(1)$ of level $1$ of the function $g(x, y, z) = x^2 + y^2 + z^2$. We will use the inverse function theorem to find adapted charts of $\mathbb{R}^3$ that cover $S^2$. To express $S^2$ as a zero set, we rewrite its defining equation as

$$f(x, y, z) = x^2 + y^2 + z^2 - 1 = 0.$$

Then $S^2 = f^{-1}(0)$. Since $\partial f / \partial x = 2x$, $\partial f / \partial y = 2y$, $\partial f / \partial z = 2z$, the only critical point of $f$ is $(0, 0, 0)$, which does not lie on the sphere $S^2$. Thus, all points on the sphere are regular points of $f$ and $0$ is a regular value of $f$.

Let $p$ be a point of $S^2$ at which $(\partial f / \partial x)(p) = 2x(p) \neq 0$. By Corollary 6.27 of the inverse function theorem, there is a neighborhood $U_p$ of $p$ in $\mathbb{R}^3$ such that $(U_p, f, y, z)$ is a chart in the atlas of $\mathbb{R}^3$. In this chart, the set $U_p \cap S^2$ is defined by the vanishing of the first coordinate $f$. Thus, $(U_p, f, y, z)$ is an adapted chart relative to $S^2$, and $(U_p \cap S^2, y, z)$ is a chart for $S^2$.

Similarly, if $(\partial f / \partial y)(p) \neq 0$, there is an adapted chart $(V_p, x, f, z)$ containing $p$. If $(\partial f / \partial z)(p) \neq 0$, there is an adapted chart $(W_p, x, y, f)$ containing $p$. Since for every $p \in S^2$, at least one of the partial derivatives $\partial f / \partial x(p)$, $\partial f / \partial y(p)$, $\partial f / \partial z(p)$ is nonzero, as $p$ varies over all points of the sphere we obtain a collection of adapted charts of $\mathbb{R}^3$ covering $S^2$. Therefore, $S^2$ is a regular submanifold of $\mathbb{R}^3$. By Proposition 9.4, $S^2$ is a manifold of dimension $2$.

</div>

This is an important example because one can generalize its proof almost verbatim to prove that if the zero set of a function $f \colon N \to \mathbb{R}$ is a regular level set, then it is a regular submanifold of $N$. The idea is that in a coordinate chart $(U, x^1, \dots, x^n)$ if a partial derivative $\partial f / \partial x^i(p)$ is nonzero, then we can replace the coordinate $x^i$ by $f$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.7</span></p>

Every regular level set $g^{-1}(c)$ of a $C^\infty$ function $g \colon N \to \mathbb{R}$ can be expressed as the regular zero set $f^{-1}(0)$ of the function $f = g - c$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.8</span></p>

Let $g \colon N \to \mathbb{R}$ be a $C^\infty$ function on the manifold $N$. Then a nonempty regular level set $S = g^{-1}(c)$ is a regular submanifold of $N$ of codimension $1$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $f = g - c$. By the preceding lemma, $S$ equals $f^{-1}(0)$ and is a regular level set of $f$. Let $p \in S$. Since $p$ is a regular point of $f$, relative to any chart $(U, x^1, \dots, x^n)$ about $p$, $(\partial f / \partial x^i)(p) \neq 0$ for some $i$. By renumbering $x^1, \dots, x^n$, we may assume that $(\partial f / \partial x^1)(p) \neq 0$.

The Jacobian matrix of the $C^\infty$ map $(f, x^2, \dots, x^n) \colon U \to \mathbb{R}^n$ at $p$ is

$$\begin{bmatrix} \frac{\partial f}{\partial x^1} & \ast & \cdots & \ast \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix},$$

and the Jacobian determinant $\partial(f, x^2, \dots, x^n) / \partial(x^1, x^2, \dots, x^n)$ at $p$ is $\partial f / \partial x^1(p) \neq 0$. By the inverse function theorem (Corollary 6.27), there is a neighborhood $U_p$ of $p$ on which $f, x^2, \dots, x^n$ form a coordinate system. Relative to the chart $(U_p, f, x^2, \dots, x^n)$, the level set $U_p \cap S$ is defined by setting the first coordinate $f$ equal to $0$, so $(U_p, f, x^2, \dots, x^n)$ is an adapted chart relative to $S$. Since this is true about every point $p \in S$, $S$ is a regular submanifold of $N$ of dimension $n - 1$. $\square$

</details>
</div>

### 9.3 The Regular Level Set Theorem

The next step is to extend Theorem 9.8 to a regular level set of a map between smooth manifolds. This very useful theorem does not seem to have an agreed-upon name in the literature. It is known variously as the implicit function theorem, the preimage theorem, and the regular level set theorem. We call it the regular level set theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.9</span><span class="math-callout__name">(Regular level set theorem)</span></p>

Let $F \colon N \to M$ be a $C^\infty$ map of manifolds, with $\dim N = n$ and $\dim M = m$. Then a nonempty regular level set $F^{-1}(c)$, where $c \in M$, is a regular submanifold of $N$ of dimension equal to $n - m$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Choose a chart $(V, \psi) = (V, y^1, \dots, y^m)$ of $M$ centered at $c$, i.e., such that $\psi(c) = \mathbf{0}$ in $\mathbb{R}^m$. Then $F^{-1}(V)$ is an open set in $N$ that contains $F^{-1}(c)$. Moreover, in $F^{-1}(V)$, $F^{-1}(c) = (\psi \circ F)^{-1}(\mathbf{0})$. So the level set $F^{-1}(c)$ is the zero set of $\psi \circ F$. If $F^i = y^i \circ F = r^i \circ (\psi \circ F)$, then $F^{-1}(c)$ is also the common zero set of the functions $F^1, \dots, F^m$ on $F^{-1}(V)$.

Because the regular level set is assumed nonempty, $n \ge m$ (Remark 9.5). Fix a point $p \in F^{-1}(c)$ and let $(U, \phi) = (U, x^1, \dots, x^n)$ be a coordinate neighborhood of $p$ in $N$ contained in $F^{-1}(V)$. Since $F^{-1}(c)$ is a regular level set, $p \in F^{-1}(c)$ is a regular point of $F$. Therefore, the $m \times n$ Jacobian matrix $[\partial F^i / \partial x^j(p)]$ has rank $m$. By renumbering the $F^i$ and $x^j$'s, we may assume that the first $m \times m$ block $[\partial F^i / \partial x^j(p)]_{1 \le i, j \le m}$ is nonsingular.

Replace the first $m$ coordinate functions $x^1, \dots, x^m$ of the chart $(U, \phi)$ by $F^1, \dots, F^m$. We claim that there is a neighborhood $U_p$ of $p$ such that $(U_p, F^1, \dots, F^m, x^{m+1}, \dots, x^n)$ is a chart in the atlas of $N$. It suffices to compute its Jacobian matrix at $p$:

$$\begin{bmatrix} \frac{\partial F^i}{\partial x^j} & \frac{\partial F^i}{\partial x^\beta} \\ \frac{\partial x^\alpha}{\partial x^j} & \frac{\partial x^\alpha}{\partial x^\beta} \end{bmatrix} = \begin{bmatrix} \frac{\partial F^i}{\partial x^j} & \ast \\ 0 & I \end{bmatrix},$$

where $1 \le i, j \le m$ and $m + 1 \le \alpha, \beta \le n$. Since this matrix has determinant $\det[\partial F^i / \partial x^j(p)]_{1 \le i, j \le m} \neq 0$, the inverse function theorem in the form of Corollary 6.27 implies the claim.

In the chart $(U_p, F^1, \dots, F^m, x^{m+1}, \dots, x^n)$, the set $S := f^{-1}(c)$ is obtained by setting the first $m$ coordinate functions $F^1, \dots, F^m$ equal to $0$. So $(U_p, F^1, \dots, F^m, x^{m+1}, \dots, x^n)$ is an adapted chart relative to $S$. Since this is true about every point $p \in S$, $S$ is a regular submanifold of $N$ of dimension $n - m$. $\square$

</details>
</div>

The proof of the regular level set theorem gives the following useful lemma.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.10</span></p>

Let $F \colon N \to \mathbb{R}^m$ be a $C^\infty$ map on a manifold $N$ of dimension $n$ and let $S$ be the level set $F^{-1}(\mathbf{0})$. If relative to some coordinate chart $(U, x^1, \dots, x^n)$ about $p \in S$, the Jacobian determinant $\partial(F^1, \dots, F^m) / \partial(x^{j_1}, \dots, x^{j_m})(p)$ is nonzero, then in some neighborhood of $p$ one may replace $x^{j_1}, \dots, x^{j_m}$ by $F^1, \dots, F^m$ to obtain an adapted chart for $N$ relative to $S$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The regular level set theorem gives a sufficient but not necessary condition for a level set to be a regular submanifold. For example, if $f \colon \mathbb{R}^2 \to \mathbb{R}$ is the map $f(x, y) = y^2$, then the zero set $Z(f) = Z(y^2)$ is the $x$-axis, a regular submanifold of $\mathbb{R}^2$. However, since $\partial f / \partial x = 0$ and $\partial f / \partial y = 2y = 0$ on the $x$-axis, every point in $Z(f)$ is a critical point of $f$. Thus, although $Z(f)$ is a regular submanifold of $\mathbb{R}^2$, it is not a regular level set of $f$.

</div>

### 9.4 Examples of Regular Submanifolds

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.11</span><span class="math-callout__name">(Hypersurface)</span></p>

Show that the solution set $S$ of $x^3 + y^3 + z^3 = 1$ in $\mathbb{R}^3$ is a manifold of dimension $2$.

*Solution.* Let $f(x, y, z) = x^3 + y^3 + z^3$. Then $S = f^{-1}(1)$. Since $\partial f / \partial x = 3x^2$, $\partial f / \partial y = 3y^2$, and $\partial f / \partial z = 3z^2$, the only critical point of $f$ is $(0, 0, 0)$, which is not in $S$. Thus, $1$ is a regular value of $f \colon \mathbb{R}^3 \to \mathbb{R}$. By the regular level set theorem (Theorem 9.9), $S$ is a regular submanifold of $\mathbb{R}^3$ of dimension $2$. So $S$ is a manifold (Proposition 9.4). $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.12</span><span class="math-callout__name">(Solution set of two polynomial equations)</span></p>

Decide whether the subset $S$ of $\mathbb{R}^3$ defined by the two equations

$$x^3 + y^3 + z^3 = 1, \quad x + y + z = 0$$

is a regular submanifold of $\mathbb{R}^3$.

*Solution.* Define $F \colon \mathbb{R}^3 \to \mathbb{R}^2$ by $(u, v) = F(x, y, z) = (x^3 + y^3 + z^3,\; x + y + z)$. Then $S$ is the level set $F^{-1}(1, 0)$. The Jacobian matrix of $F$ is

$$J(F) = \begin{bmatrix} 3x^2 & 3y^2 & 3z^2 \\ 1 & 1 & 1 \end{bmatrix}.$$

The critical points of $F$ are the points $(x, y, z)$ where the matrix $J(F)$ has rank $< 2$. That is precisely where all $2 \times 2$ minors of $J(F)$ are zero:

$$\begin{vmatrix} 3x^2 & 3y^2 \\ 1 & 1 \end{vmatrix} = 0, \quad \begin{vmatrix} 3x^2 & 3z^2 \\ 1 & 1 \end{vmatrix} = 0.$$

Solving, we get $y = \pm x$ and $z = \pm x$. Since $x + y + z = 0$ on $S$, this implies that $(x, y, z) = (0, 0, 0)$. Since $(0, 0, 0)$ does not satisfy the first equation $x^3 + y^3 + z^3 = 1$, there are no critical points of $F$ on $S$. By the regular level set theorem, $S$ is a regular submanifold of $\mathbb{R}^3$ of dimension $1$. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.13</span><span class="math-callout__name">(Special linear group)</span></p>

As a set, the **special linear group** $\operatorname{SL}(n, \mathbb{R})$ is the subset of $\operatorname{GL}(n, \mathbb{R})$ consisting of matrices of determinant $1$. Since

$$\det(AB) = (\det A)(\det B) \quad \text{and} \quad \det(A^{-1}) = \frac{1}{\det A},$$

$\operatorname{SL}(n, \mathbb{R})$ is a subgroup of $\operatorname{GL}(n, \mathbb{R})$. To show that it is a regular submanifold, we let $f \colon \operatorname{GL}(n, \mathbb{R}) \to \mathbb{R}$ be the determinant map $f(A) = \det A$, and apply the regular level set theorem to $f^{-1}(1) = \operatorname{SL}(n, \mathbb{R})$. We need to check that $1$ is a regular value of $f$.

Let $a_{ij}$, $1 \le i \le n$, $1 \le j \le n$, be the standard coordinates on $\mathbb{R}^{n \times n}$, and let $S_{ij}$ denote the submatrix of $A = [a_{ij}] \in \mathbb{R}^{n \times n}$ obtained by deleting its $i$th row and $j$th column. Then $m_{ij} := \det S_{ij}$ is the $(i, j)$-minor of $A$. From linear algebra we have

$$f(A) = \det A = (-1)^{i+1} a_{i1} m_{i1} + (-1)^{i+2} a_{i2} m_{i2} + \cdots + (-1)^{i+n} a_{in} m_{in}$$

for any row $i$. Therefore $\partial f / \partial a_{ij} = (-1)^{i+j} m_{ij}$.

Hence, a matrix $A \in \operatorname{GL}(n, \mathbb{R})$ is a critical point of $f$ if and only if all the $(n-1) \times (n-1)$ minors $m_{ij}$ of $A$ are zero. By the cofactor expansion, such a matrix $A$ has determinant $0$. Since every matrix in $\operatorname{SL}(n, \mathbb{R})$ has determinant $1$, all the matrices in $\operatorname{SL}(n, \mathbb{R})$ are regular points of the determinant function. By the regular level set theorem (Theorem 9.9), $\operatorname{SL}(n, \mathbb{R})$ is a regular submanifold of $\operatorname{GL}(n, \mathbb{R})$ of codimension $1$; i.e.,

$$\dim \operatorname{SL}(n, \mathbb{R}) = \dim \operatorname{GL}(n, \mathbb{R}) - 1 = n^2 - 1.$$

</div>

# Chapter 4 — Lie Groups and Lie Algebras

## §10 Categories and Functors

Many of the problems in mathematics share common features. For example, in topology one is interested in knowing whether two topological spaces are homeomorphic; in group theory, whether two groups are isomorphic. This has given rise to the theory of categories and functors, which tries to clarify the structural similarities among different areas of mathematics.

A category is essentially a collection of objects and arrows between objects. These arrows, called morphisms, satisfy the abstract properties of maps and are often structure-preserving maps. Smooth manifolds and smooth maps form a category, and so do vector spaces and linear maps. A functor from one category to another preserves the identity morphism and the composition of morphisms. It provides a way to simplify problems in the first category, for the target category of a functor is usually simpler than the original category.

### 10.1 Categories

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.0</span><span class="math-callout__name">(Category)</span></p>

A **category** consists of a collection of elements, called *objects*, and for any two objects $A$ and $B$, a set $\operatorname{Mor}(A, B)$ of elements, called *morphisms* from $A$ to $B$, such that given any morphism $f \in \operatorname{Mor}(A, B)$ and any morphism $g \in \operatorname{Mor}(B, C)$, the *composite* $g \circ f \in \operatorname{Mor}(A, C)$ is defined. Furthermore, the composition of morphisms is required to satisfy two properties:

1. **Identity axiom:** for each object $A$, there is an identity morphism $\mathbb{1}_A \in \operatorname{Mor}(A, A)$ such that for any $f \in \operatorname{Mor}(A, B)$ and $g \in \operatorname{Mor}(B, A)$,

$$f \circ \mathbb{1}_A = f \quad \text{and} \quad \mathbb{1}_A \circ g = g.$$

2. **Associative axiom:** for $f \in \operatorname{Mor}(A, B)$, $g \in \operatorname{Mor}(B, C)$, and $h \in \operatorname{Mor}(C, D)$,

$$h \circ (g \circ f) = (h \circ g) \circ f.$$

If $f \in \operatorname{Mor}(A, B)$, we often write $f \colon A \to B$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Groups and group homomorphisms)</span></p>

The collection of groups and group homomorphisms forms a category in which the objects are groups and for any two groups $A$ and $B$, $\operatorname{Mor}(A, B)$ is the set of group homomorphisms from $A$ to $B$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Real vector spaces and linear maps)</span></p>

The collection of all vector spaces over $\mathbb{R}$ and $\mathbb{R}$-linear maps forms a category in which the objects are real vector spaces and for any two real vector spaces $V$ and $W$, $\operatorname{Mor}(V, W)$ is the set $\operatorname{Hom}(V, W)$ of linear maps from $V$ to $W$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Continuous category)</span></p>

The collection of all topological spaces together with continuous maps between them is called the **continuous category**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Smooth category)</span></p>

The collection of smooth manifolds together with smooth maps between them is called the **smooth category**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Pointed manifolds)</span></p>

We call a pair $(M, q)$, where $M$ is a manifold and $q$ a point in $M$, a **pointed manifold**. Given any two such pairs $(N, p)$ and $(M, q)$, let $\operatorname{Mor}\bigl((N, p), (M, q)\bigr)$ be the set of all smooth maps $F \colon N \to M$ such that $F(p) = q$. This gives rise to the **category of pointed manifolds**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.1</span><span class="math-callout__name">(Isomorphic objects)</span></p>

Two objects $A$ and $B$ in a category are said to be **isomorphic** if there are morphisms $f \colon A \to B$ and $g \colon B \to A$ such that

$$g \circ f = \mathbb{1}_A \quad \text{and} \quad f \circ g = \mathbb{1}_B.$$

In this case both $f$ and $g$ are called **isomorphisms**.

</div>

The usual notation for an isomorphism is "$\simeq$". Thus, $A \simeq B$ can mean, for example, a group isomorphism, a vector space isomorphism, a homeomorphism, or a diffeomorphism, depending on the category and the context.

### 10.2 Functors

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.2</span><span class="math-callout__name">(Covariant functor)</span></p>

A *(covariant) functor* $\mathcal{F}$ from one category $\mathcal{C}$ to another category $\mathcal{D}$ is a map that associates to each object $A$ in $\mathcal{C}$ an object $\mathcal{F}(A)$ in $\mathcal{D}$ and to each morphism $f \colon A \to B$ a morphism $\mathcal{F}(f) \colon \mathcal{F}(A) \to \mathcal{F}(B)$ such that

1. $\mathcal{F}(\mathbb{1}_A) = \mathbb{1}_{\mathcal{F}(A)}$,
2. $\mathcal{F}(f \circ g) = \mathcal{F}(f) \circ \mathcal{F}(g)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tangent space as a functor)</span></p>

The tangent space construction is a functor from the category of pointed manifolds to the category of vector spaces. To each pointed manifold $(N, p)$ we associate the tangent space $T_pN$ and to each smooth map $f \colon (N, p) \to (M, f(p))$ we associate the differential $f_{*, p} \colon T_pN \to T_{f(p)}M$.

The functorial property (i) holds because if $\mathbb{1} \colon N \to N$ is the identity map, then its differential $\mathbb{1}_{*, p} \colon T_pN \to T_pN$ is also the identity map. The functorial property (ii) holds because in this context it is the chain rule:

$$(g \circ f)_{*, p} = g_{*, f(p)} \circ f_{*, p}.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10.3</span><span class="math-callout__name">(Functors preserve isomorphisms)</span></p>

Let $\mathcal{F} \colon \mathcal{C} \to \mathcal{D}$ be a functor from a category $\mathcal{C}$ to a category $\mathcal{D}$. If $f \colon A \to B$ is an isomorphism in $\mathcal{C}$, then $\mathcal{F}(f) \colon \mathcal{F}(A) \to \mathcal{F}(B)$ is an isomorphism in $\mathcal{D}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Invariance of dimension)</span></p>

Note that we can now recast Corollaries 8.6 and 8.7 in a more functorial form. Suppose $f \colon N \to M$ is a diffeomorphism. Then $(N, p)$ and $(M, f(p))$ are isomorphic objects in the category of pointed manifolds. By Proposition 10.3, the tangent spaces $T_pN$ and $T_{f(p)}M$ must be isomorphic as vector spaces and therefore have the same dimension. It follows that the dimension of a manifold is invariant under a diffeomorphism.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.4</span><span class="math-callout__name">(Contravariant functor)</span></p>

A **contravariant functor** $\mathcal{F}$ from one category $\mathcal{C}$ to another category $\mathcal{D}$ is a map that associates to each object $A$ in $\mathcal{C}$ an object $\mathcal{F}(A)$ in $\mathcal{D}$ and to each morphism $f \colon A \to B$ a morphism $\mathcal{F}(f) \colon \mathcal{F}(B) \to \mathcal{F}(A)$ such that

1. $\mathcal{F}(\mathbb{1}_A) = \mathbb{1}_{\mathcal{F}(A)}$;
2. $\mathcal{F}(f \circ g) = \mathcal{F}(g) \circ \mathcal{F}(f)$. (Note the reversal of order.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Pullback of smooth functions)</span></p>

Smooth functions on a manifold give rise to a contravariant functor that associates to each manifold $M$ the algebra $\mathcal{F}(M) = C^\infty(M)$ of $C^\infty$ functions on $M$ and to each smooth map $F \colon N \to M$ of manifolds the pullback map $\mathcal{F}(F) = F^* \colon C^\infty(M) \to C^\infty(N)$, where $F^*(h) = h \circ F$ for $h \in C^\infty(M)$. It is easy to verify that the pullback satisfies the two functorial properties:

1. $(\mathbb{1}_M)^* = \mathbb{1}_{C^\infty(M)}$,
2. if $F \colon N \to M$ and $G \colon M \to P$ are $C^\infty$ maps, then $(G \circ F)^* = F^* \circ G^* \colon C^\infty(P) \to C^\infty(N)$.

</div>

### 10.3 The Dual Functor and the Multicovector Functor

Let $V$ be a real vector space. Recall that its dual space $V^\vee$ is the vector space of all *linear functionals* on $V$, i.e., linear functions $\alpha \colon V \to \mathbb{R}$. We also write

$$V^\vee = \operatorname{Hom}(V, \mathbb{R}).$$

If $V$ is a finite-dimensional vector space with basis $\lbrace e_1, \dots, e_n \rbrace$, then by Proposition 3.1 its dual space $V^\vee$ has as a basis the collection of linear functionals $\lbrace \alpha^1, \dots, \alpha^n \rbrace$ defined by

$$\alpha^i(e_j) = \delta^i_j, \quad 1 \le i, j \le n.$$

A linear map $L \colon V \to W$ of vector spaces induces a linear map $L^\vee$, called the **dual** of $L$, as follows. To every linear functional $\alpha \colon W \to \mathbb{R}$, the dual map $L^\vee$ associates the linear functional

$$V \xrightarrow{L} W \xrightarrow{\alpha} \mathbb{R}.$$

Thus, the dual map $L^\vee \colon W^\vee \to V^\vee$ is given by

$$L^\vee(\alpha) = \alpha \circ L \quad \text{for } \alpha \in W^\vee.$$

Note that the dual of $L$ reverses the direction of the arrow.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10.5</span><span class="math-callout__name">(Functorial properties of the dual)</span></p>

Suppose $V$, $W$, and $S$ are real vector spaces.

1. If $\mathbb{1}_V \colon V \to V$ is the identity map on $V$, then $\mathbb{1}_V^\vee \colon V^\vee \to V^\vee$ is the identity map on $V^\vee$.
2. If $f \colon V \to W$ and $g \colon W \to S$ are linear maps, then $(g \circ f)^\vee = f^\vee \circ g^\vee$.

</div>

According to this proposition, the dual construction $\mathcal{F} \colon () \mapsto ()^\vee$ is a contravariant functor from the category of vector spaces to itself: for $V$ a real vector space, $\mathcal{F}(V) = V^\vee$ and for $f \in \operatorname{Hom}(V, W)$, $\mathcal{F}(f) = f^\vee \in \operatorname{Hom}(W^\vee, V^\vee)$. Consequently, if $f \colon V \to W$ is an isomorphism, then so is its dual $f^\vee \colon W^\vee \to V^\vee$ (cf. Proposition 10.3).

Fix a positive integer $k$. For any linear map $L \colon V \to W$ of vector spaces, define the **pullback map** $L^* \colon A_k(W) \to A_k(V)$ to be

$$(L^* f)(v_1, \dots, v_k) = f(L(v_1), \dots, L(v_k))$$

for $f \in A_k(W)$ and $v_1, \dots, v_k \in V$. From the definition, it is easy to see that $L^*$ is a linear map: $L^*(af + bg) = aL^*f + bL^*g$ for $a, b \in \mathbb{R}$ and $f, g \in A_k(W)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10.6</span><span class="math-callout__name">(Functorial properties of the pullback of covectors)</span></p>

The pullback of covectors by a linear map satisfies the two functorial properties:

1. If $\mathbb{1}_V \colon V \to V$ is the identity map on $V$, then $\mathbb{1}_V^* = \mathbb{1}_{A_k(V)}$, the identity map on $A_k(V)$.
2. If $K \colon U \to V$ and $L \colon V \to W$ are linear maps of vector spaces, then

$$(L \circ K)^* = K^* \circ L^* \colon A_k(W) \to A_k(U).$$

</div>

To each vector space $V$, we associate the vector space $A_k(V)$ of all $k$-covectors on $V$, and to each linear map $L \colon V \to W$ of vector spaces, we associate the pullback $A_k(L) = L^* \colon A_k(W) \to A_k(V)$. Then $A_k(\ )$ is a contravariant functor from the category of vector spaces and linear maps to itself.

When $k = 1$, for any vector space $V$, the space $A_1(V)$ is the dual space, and for any linear map $L \colon V \to W$, the pullback map $A_1(L) = L^*$ is the dual map $L^\vee \colon W^\vee \to V^\vee$. Thus, the multicovector functor $A_k(\ )$ generalizes the dual functor $()^\vee$.

## §11 The Rank of a Smooth Map

In this section we analyze the local structure of a smooth map through its rank. Recall that the rank of a smooth map $f \colon N \to M$ at a point $p \in N$ is the rank of its differential at $p$. Two cases are of special interest: that in which the map $f$ has maximal rank at a point and that in which it has constant rank in a neighborhood. Let $n = \dim N$ and $m = \dim M$. In case $f \colon N \to M$ has maximal rank at $p$, there are three not mutually exclusive possibilities:

1. If $n = m$, then by the inverse function theorem, $f$ is a local diffeomorphism at $p$.
2. If $n \le m$, then the maximal rank is $n$ and $f$ is an *immersion* at $p$.
3. If $n \ge m$, then the maximal rank is $m$ and $f$ is a *submersion* at $p$.

Because manifolds are locally Euclidean, theorems on the rank of a smooth map between Euclidean spaces (Appendix B) translate easily to theorems about manifolds. This leads to the constant rank theorem for manifolds, which gives a simple normal form for a smooth map having constant rank on an open set (Theorem 11.1). As an immediate consequence, we obtain a criterion for a level set to be a regular submanifold, which, following [25], we call the constant-rank level set theorem. The constant rank theorem specializes to the immersion theorem and the submersion theorem, giving simple normal forms for an immersion and a submersion. The regular level set theorem, which we encountered in Subsection 9.3, is now seen to be a consequence of the submersion theorem and a special case of the constant-rank level set theorem.

### 11.1 Constant Rank Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.1</span><span class="math-callout__name">(Constant rank theorem)</span></p>

Let $N$ and $M$ be manifolds of dimensions $n$ and $m$ respectively. Suppose $f \colon N \to M$ has constant rank $k$ in a neighborhood of a point $p$ in $N$. Then there are charts $(U, \phi)$ centered at $p$ in $N$ and $(V, \psi)$ centered at $f(p)$ in $M$ such that for $(r^1, \dots, r^n)$ in $\phi(U)$,

$$(\psi \circ f \circ \phi^{-1})(r^1, \dots, r^n) = (r^1, \dots, r^k, 0, \dots, 0).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Choose a chart $(\bar{U}, \bar{\phi})$ about $p$ in $N$ and $(\bar{V}, \bar{\psi})$ about $f(p)$ in $M$. Then $\bar{\psi} \circ f \circ \bar{\phi}^{-1}$ is a map between open subsets of Euclidean spaces. Because $\bar{\phi}$ and $\bar{\psi}$ are diffeomorphisms, $\bar{\psi} \circ f \circ \bar{\phi}^{-1}$ has the same constant rank $k$ as $f$ in a neighborhood of $\bar{\phi}(p)$ in $\mathbb{R}^n$. By the constant rank theorem for Euclidean spaces (Theorem B.4) there are a diffeomorphism $G$ of a neighborhood of $\bar{\phi}(p)$ in $\mathbb{R}^n$ and a diffeomorphism $F$ of a neighborhood of $(\bar{\psi} \circ f)(p)$ in $\mathbb{R}^m$ such that

$$(F \circ \bar{\psi} \circ f \circ \bar{\phi}^{-1} \circ G^{-1})(r^1, \dots, r^n) = (r^1, \dots, r^k, 0, \dots, 0).$$

Set $\phi = G \circ \bar{\phi}$ and $\psi = F \circ \bar{\psi}$. $\square$

</details>
</div>

In the constant rank theorem, it is possible that the normal form (11.1) for the function $f$ has no zeros at all: if the rank $k$ equals $m$, then

$$(\psi \circ f \circ \phi^{-1})(r^1, \dots, r^n) = (r^1, \dots, r^m).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.2</span><span class="math-callout__name">(Constant-rank level set theorem)</span></p>

Let $f \colon N \to M$ be a $C^\infty$ map of manifolds and $c \in M$. If $f$ has constant rank $k$ in a neighborhood of the level set $f^{-1}(c)$ in $N$, then $f^{-1}(c)$ is a regular submanifold of $N$ of codimension $k$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $p$ be an arbitrary point in $f^{-1}(c)$. By the constant rank theorem there are a coordinate chart $(U, \phi) = (U, x^1, \dots, x^n)$ centered at $p \in N$ and a coordinate chart $(V, \psi) = (V, y^1, \dots, y^m)$ centered at $f(p) = c \in M$ such that

$$(\psi \circ f \circ \phi^{-1})(r^1, \dots, r^n) = (r^1, \dots, r^k, 0, \dots, 0) \in \mathbb{R}^m.$$

This shows that the level set $(\psi \circ f \circ \phi^{-1})^{-1}(0)$ is defined by the vanishing of the coordinates $r^1, \dots, r^k$.

The image of the level set $f^{-1}(c)$ under $\phi$ is the level set $(\psi \circ f \circ \phi^{-1})^{-1}(0)$, since

$$\phi(f^{-1}(c)) = \phi\bigl(f^{-1}(\psi^{-1}(0))\bigr) = (\psi \circ f \circ \phi^{-1})^{-1}(0).$$

Thus, the level set $f^{-1}(c)$ in $U$ is defined by the vanishing of the coordinate functions $x^1, \dots, x^k$, where $x^i = r^i \circ \phi$. This proves that $f^{-1}(c)$ is a regular submanifold of $N$ of codimension $k$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.3</span><span class="math-callout__name">(Orthogonal group)</span></p>

The **orthogonal group** $\operatorname{O}(n)$ is defined to be the subgroup of $\operatorname{GL}(n, \mathbb{R})$ consisting of matrices $A$ such that $A^T A = I$, the $n \times n$ identity matrix. Using the constant rank theorem, prove that $\operatorname{O}(n)$ is a regular submanifold of $\operatorname{GL}(n, \mathbb{R})$.

*Solution.* Define $f \colon \operatorname{GL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R})$ by $f(A) = A^T A$. Then $\operatorname{O}(n)$ is the level set $f^{-1}(I)$. For any two matrices $A, B \in \operatorname{GL}(n, \mathbb{R})$, there is a unique matrix $C \in \operatorname{GL}(n, \mathbb{R})$ such that $B = AC$. Denote by $\ell_C$ and $r_C \colon \operatorname{GL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R})$ the left and right multiplication by $C$, respectively. Since

$$f(AC) = (AC)^T AC = C^T A^T AC = C^T f(A) C,$$

we have

$$(f \circ r_C)(A) = (\ell_{C^T} \circ r_C \circ f)(A).$$

Since this is true for all $A \in \operatorname{GL}(n, \mathbb{R})$,

$$f \circ r_C = \ell_{C^T} \circ r_C \circ f.$$

By the chain rule,

$$f_{*, AC} \circ (r_C)_{*, A} = (\ell_{C^T})_{*, A^T AC} \circ (r_C)_{*, A^T A} \circ f_{*, A}.$$

Since left and right multiplications are diffeomorphisms, their differentials are isomorphisms. Composition with an isomorphism does not change the rank of a linear map. Hence,

$$\operatorname{rk} f_{*, AC} = \operatorname{rk} f_{*, A}.$$

Since $AC$ and $A$ are two arbitrary points of $\operatorname{GL}(n, \mathbb{R})$, this proves that the differential of $f$ has constant rank on $\operatorname{GL}(n, \mathbb{R})$. By the constant-rank level set theorem, the orthogonal group $\operatorname{O}(n) = f^{-1}(I)$ is a regular submanifold of $\operatorname{GL}(n, \mathbb{R})$.

</div>

### 11.2 The Immersion and Submersion Theorems

In this subsection we explain why immersions and submersions have constant rank. The constant rank theorem gives local normal forms for immersions and submersions, called the immersion theorem and the submersion theorem respectively.

Consider a $C^\infty$ map $f \colon N \to M$. Let $(U, \phi) = (U, x^1, \dots, x^n)$ be a chart about $p$ in $N$ and $(V, \psi) = (V, y^1, \dots, y^m)$ a chart about $f(p)$ in $M$. Write $f^i = y^i \circ f$ for the $i$th component of $f$ in the chart $(V, y^1, \dots, y^m)$. Relative to the charts $(U, \phi)$ and $(V, \psi)$, the linear map $f_{*, p}$ is represented by the matrix $[\partial f^i / \partial x^j(p)]$ (Proposition 8.11). Hence,

$$f_{*, p} \text{ is injective} \iff n \le m \text{ and } \operatorname{rk}[\partial f^i / \partial x^j(p)] = n,$$

$$f_{*, p} \text{ is surjective} \iff n \ge m \text{ and } \operatorname{rk}[\partial f^i / \partial x^j(p)] = m.$$

The rank of a matrix is the number of linearly independent rows of the matrix; it is also the number of linearly independent columns. Thus, the maximum possible rank of an $m \times n$ matrix is the minimum of $m$ and $n$. It follows that being an immersion or a submersion at $p$ is equivalent to the maximality of $\operatorname{rk}[\partial f^i / \partial x^j(p)]$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.4</span><span class="math-callout__name">(Maximal rank implies constant rank)</span></p>

Let $N$ and $M$ be manifolds of dimensions $n$ and $m$ respectively. If a $C^\infty$ map $f \colon N \to M$ is an immersion at a point $p \in N$, then it has constant rank $n$ in a neighborhood of $p$. If a $C^\infty$ map $f \colon N \to M$ is a submersion at a point $p \in N$, then it has constant rank $m$ in a neighborhood of $p$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

While maximal rank at a point implies constant rank in a neighborhood, the converse is not true. The map $f \colon \mathbb{R}^2 \to \mathbb{R}^3$, $f(x, y) = (x, 0, 0)$, has constant rank $1$, but does not have maximal rank at any point.

</div>

By Proposition 11.4, the following theorems are simply special cases of the constant rank theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.5</span><span class="math-callout__name">(Immersion and submersion theorems)</span></p>

Let $N$ and $M$ be manifolds of dimensions $n$ and $m$ respectively.

**(i) (Immersion theorem)** Suppose $f \colon N \to M$ is an immersion at $p \in N$. Then there are charts $(U, \phi)$ centered at $p$ in $N$ and $(V, \psi)$ centered at $f(p)$ in $M$ such that in a neighborhood of $\phi(p)$,

$$(\psi \circ f \circ \phi^{-1})(r^1, \dots, r^n) = (r^1, \dots, r^n, 0, \dots, 0).$$

**(ii) (Submersion theorem)** Suppose $f \colon N \to M$ is a submersion at $p$ in $N$. Then there are charts $(U, \phi)$ centered at $p$ in $N$ and $(V, \psi)$ centered at $f(p)$ in $M$ such that in a neighborhood of $\phi(p)$,

$$(\psi \circ f \circ \phi^{-1})(r^1, \dots, r^m, r^{m+1}, \dots, r^n) = (r^1, \dots, r^m).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 11.6</span></p>

A submersion $f \colon N \to M$ of manifolds is an open map.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $W$ be an open subset of $N$. We need to show that its image $f(W)$ is open in $M$. Choose a point $f(p)$ in $f(W)$, with $p \in W$. By the submersion theorem, $f$ is locally a projection. Since a projection is an open map (Problem A.7), there is an open neighborhood $U$ of $p$ in $W$ such that $f(U)$ is open in $M$. Clearly,

$$f(p) \in f(U) \subset f(W).$$

Since $f(p) \in f(W)$ was arbitrary, $f(W)$ is open in $M$. $\square$

</details>
</div>

The regular level set theorem (Theorem 9.9) is an easy corollary of the submersion theorem. Indeed, for a $C^\infty$ map $f \colon N \to M$ of manifolds, a level set $f^{-1}(c)$ is regular if and only if $f$ is a submersion at every point $p \in f^{-1}(c)$.

### 11.3 Images of Smooth Maps

By the regular level set theorem, the *preimage* of a regular value of a smooth map is a manifold. The *image* of a smooth map, on the other hand, does not generally have a nice structure. Using the immersion theorem we derive conditions under which the image of a smooth map is a manifold.

The following are all examples of $C^\infty$ maps $f \colon N \to M$, with $N = \mathbb{R}$ and $M = \mathbb{R}^2$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.7</span></p>

$f(t) = (t^2, t^3)$.

This $f$ is one-to-one, because $t \mapsto t^3$ is one-to-one. Since $f'(0) = (0, 0)$, the differential $f_{*, 0} \colon T_0 \mathbb{R} \to T_{(0,0)} \mathbb{R}^2$ is the zero map and hence not injective; so $f$ is not an immersion at $0$. Its image is the cuspidal cubic $y^2 = x^3$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.8</span></p>

$f(t) = (t^2 - 1, t^3 - t)$.

Since the equation $f'(t) = (2t, 3t^2 - 1) = (0, 0)$ has no solution in $t$, this map $f$ is an immersion. It is not one-to-one, because it maps both $t = 1$ and $t = -1$ to the origin. To find an equation for the image $f(N)$, let $x = t^2 - 1$ and $y = t^3 - t$. Then $y = t(t^2 - 1) = tx$; so

$$y^2 = t^2 x^2 = (x + 1)x^2.$$

Thus the image of $f$ is the nodal cubic $y^2 = x^2(x + 1)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.9</span></p>

The map $f$ in Figure 11.4 is a one-to-one immersion but its image, with the subspace topology induced from $\mathbb{R}^2$, is not homeomorphic to the domain $\mathbb{R}$, because there are points near $f(p)$ in the image that correspond to points in $\mathbb{R}$ far away from $p$. More precisely, if $U$ is an interval about $p$ as shown, there is no neighborhood $V$ of $f(p)$ in $f(N)$ such that $f^{-1}(V) \subset U$; hence, $f^{-1}$ is not continuous.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 11.11</span><span class="math-callout__name">(Embedding)</span></p>

A $C^\infty$ map $f \colon N \to M$ is called an **embedding** if

1. it is a one-to-one immersion and
2. the image $f(N)$ with the subspace topology is homeomorphic to $N$ under $f$.

(The phrase "one-to-one" in this definition is redundant, since a homeomorphism is necessarily one-to-one.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Immersed submanifold vs. regular submanifold)</span></p>

Unfortunately, there is quite a bit of terminological confusion in the literature concerning the use of the word "submanifold." Many authors give the image $f(N)$ of a one-to-one immersion $f \colon N \to M$ not the subspace topology, but the topology inherited from $f$; i.e., a subset $f(U)$ of $f(N)$ is said to be open if and only if $U$ is open in $N$. With this topology, $f(N)$ is by definition homeomorphic to $N$. These authors define a submanifold to be the image of any one-to-one immersion with the topology and differentiable structure inherited from $f$. Such a set is sometimes called an **immersed submanifold** of $M$. If the underlying set of an immersed submanifold is given the subspace topology, then the resulting space need not be a manifold at all!

For us, a submanifold without any qualifying adjective is always a **regular submanifold**. To recapitulate, a regular submanifold of a manifold $M$ is a subset $S$ of $M$ with the subspace topology such that every point of $S$ has a neighborhood $U \cap S$ defined by the vanishing of coordinate functions on $U$, where $U$ is a chart in $M$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.12</span><span class="math-callout__name">(The figure-eight)</span></p>

The figure-eight is the image of a one-to-one immersion

$$f(t) = (\cos t, \sin 2t), \quad -\pi/2 < t < 3\pi/2.$$

As such, it is an immersed submanifold of $\mathbb{R}^2$, with a topology and manifold structure induced from the open interval $]-\pi/2, 3\pi/2[$ by $f$. Because of the presence of a cross at the origin, it cannot be a regular submanifold of $\mathbb{R}^2$. In fact, with the subspace topology of $\mathbb{R}^2$, the figure-eight is not even a manifold.

The figure-eight is also the image of the one-to-one immersion

$$g(t) = (\cos t, -\sin 2t), \quad -\pi/2 < t < 3\pi/2.$$

The maps $f$ and $g$ induce distinct immersed submanifold structures on the figure-eight.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.13</span><span class="math-callout__name">(Image of an embedding)</span></p>

If $f \colon N \to M$ is an embedding, then its image $f(N)$ is a regular submanifold of $M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $p \in N$. By the immersion theorem (Theorem 11.5), there are local coordinates $(U, x^1, \dots, x^n)$ near $p$ and $(V, y^1, \dots, y^m)$ near $f(p)$ such that $f \colon U \to V$ has the form

$$(x^1, \dots, x^n) \mapsto (x^1, \dots, x^n, 0, \dots, 0).$$

Thus, $f(U)$ is defined in $V$ by the vanishing of the coordinates $y^{n+1}, \dots, y^m$. This alone does not prove that $f(N)$ is a regular submanifold, since $V \cap f(N)$ may be larger than $f(U)$. (Think about Examples 11.9 and 11.10.) We need to show that in some neighborhood of $f(p)$ in $V$, the set $f(N)$ is defined by the vanishing of $m - n$ coordinates.

Since $f(N)$ with the subspace topology is homeomorphic to $N$, the image $f(U)$ is open in $f(N)$. By the definition of the subspace topology, there is an open set $V'$ in $M$ such that $V' \cap f(N) = f(U)$. In $V \cap V'$,

$$V \cap V' \cap f(N) = V \cap f(U) = f(U),$$

and $f(U)$ is defined by $y^{n+1}, \dots, y^m$. Thus, $(V \cap V', y^1, \dots, y^m)$ is an adapted chart containing $f(p)$ for $f(N)$. Since $f(p)$ is an arbitrary point of $f(N)$, this proves that $f(N)$ is a regular submanifold of $M$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.14</span></p>

If $N$ is a regular submanifold of $M$, then the inclusion $i \colon N \to M$, $i(p) = p$, is an embedding.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since a regular submanifold has the subspace topology and $i(N)$ also has the subspace topology, $i \colon N \to i(N)$ is a homeomorphism. It remains to show that $i \colon N \to M$ is an immersion.

Let $p \in N$. Choose an adapted chart $(V, y^1, \dots, y^n, y^{n+1}, \dots, y^m)$ for $M$ about $p$ such that $V \cap N$ is the zero set of $y^{n+1}, \dots, y^m$. Relative to the charts $(V \cap N, y^1, \dots, y^n)$ for $N$ and $(V, y^1, \dots, y^m)$ for $M$, the inclusion $i$ is given by

$$(y^1, \dots, y^n) \mapsto (y^1, \dots, y^n, 0, \dots, 0),$$

which shows that $i$ is an immersion. $\square$

</details>
</div>

In the literature the image of an embedding is often called an **embedded submanifold**. Theorems 11.13 and 11.14 show that an embedded submanifold and a regular submanifold are one and the same thing.

### 11.4 Smooth Maps into a Submanifold

Suppose $f \colon N \to M$ is a $C^\infty$ map whose image $f(N)$ lies in a subset $S \subset M$. If $S$ is a manifold, is the induced map $\tilde{f} \colon N \to S$ also $C^\infty$? This question is more subtle than it looks, because the answer depends on whether $S$ is a regular submanifold or an immersed submanifold.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Smooth maps and immersed submanifolds)</span></p>

Consider the one-to-one immersions $f$ and $g \colon I \to \mathbb{R}^2$ in Example 11.12, where $I$ is the open interval $]-\pi/2, 3\pi/2[$. Let $S$ be the figure-eight in $\mathbb{R}^2$ with the immersed submanifold structure induced from $g$. Because the image of $f \colon I \to \mathbb{R}^2$ lies in $S$, the $C^\infty$ map $f$ induces a map $\tilde{f} \colon I \to S$.

The open interval from $A$ to $B$ in Figure 11.6 is an open neighborhood of the origin $0$ in $S$. Its inverse image under $\tilde{f}$ contains the point $\pi/2$ as an isolated point and is therefore not open. This shows that although $f \colon I \to \mathbb{R}^2$ is $C^\infty$, the induced map $\tilde{f} \colon I \to S$ is not continuous and therefore not $C^\infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.15</span><span class="math-callout__name">(Smooth maps into a regular submanifold)</span></p>

Suppose $f \colon N \to M$ is $C^\infty$ and the image of $f$ lies in a subset $S$ of $M$. If $S$ is a regular submanifold of $M$, then the induced map $\tilde{f} \colon N \to S$ is $C^\infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $p \in N$. Denote the dimensions of $N$, $M$, and $S$ by $n$, $m$, and $s$, respectively. By hypothesis, $f(p) \in S \subset M$. Since $S$ is a regular submanifold of $M$, there is an adapted coordinate chart $(V, \psi) = (V, y^1, \dots, y^m)$ for $M$ about $f(p)$ such that $S \cap V$ is the zero set of $y^{s+1}, \dots, y^m$, with coordinate map $\psi_S = (y^1, \dots, y^s)$. By the continuity of $f$, it is possible to choose a neighborhood $U$ of $p$ with $f(U) \subset V$. Then $f(U) \subset V \cap S$, so that for $q \in U$,

$$(\psi \circ f)(q) = (y^1(f(q)), \dots, y^s(f(q)), 0, \dots, 0).$$

It follows that on $U$,

$$\psi_S \circ \tilde{f} = (y^1 \circ f, \dots, y^s \circ f).$$

Since $y^1 \circ f, \dots, y^s \circ f$ are $C^\infty$ on $U$, by Proposition 6.16, $\tilde{f}$ is $C^\infty$ on $U$ and hence at $p$. Since $p$ was an arbitrary point of $N$, the map $\tilde{f} \colon N \to S$ is $C^\infty$. $\square$

</details>
</div>

### 11.5 The Tangent Plane to a Surface in $\mathbb{R}^3$

Suppose $f(x^1, x^2, x^3)$ is a real-valued function on $\mathbb{R}^3$ with no critical points on its zero set $N = f^{-1}(0)$. By the regular level set theorem, $N$ is a regular submanifold of $\mathbb{R}^3$. By Theorem 11.14 the inclusion $i \colon N \to \mathbb{R}^3$ is an embedding, so at any point $p$ in $N$, $i_{*, p} \colon T_pN \to T_p\mathbb{R}^3$ is injective. We may therefore think of the tangent plane $T_pN$ as a plane in $T_p\mathbb{R}^3 \simeq \mathbb{R}^3$. We would like to find the equation of this plane.

Suppose $v = \sum v^i \,\partial/\partial x^i|_p$ is a vector in $T_pN$. Under the linear isomorphism $T_p\mathbb{R}^3 \simeq \mathbb{R}^3$, we identify $v$ with the vector $\langle v^1, v^2, v^3 \rangle$ in $\mathbb{R}^3$. Let $c(t)$ be a curve lying in $N$ with $c(0) = p$ and $c'(0) = \langle v^1, v^2, v^3 \rangle$. Since $c(t)$ lies in $N$, $f(c(t)) = 0$ for all $t$. By the chain rule,

$$0 = \frac{d}{dt} f(c(t)) = \sum_{i=1}^3 \frac{\partial f}{\partial x^i}(c(t))(c^i)'(t).$$

At $t = 0$,

$$0 = \sum_{i=1}^3 \frac{\partial f}{\partial x^i}(p) v^i.$$

Since the vector $v = \langle v^1, v^2, v^3 \rangle$ represents the arrow from the point $p = (p^1, p^2, p^3)$ to $x = (x^1, x^2, x^3)$ in the tangent plane, one usually makes the substitution $v^i = x^i - p^i$. Thus the tangent plane to $N$ at $p$ is defined by the equation

$$\sum_{i=1}^3 \frac{\partial f}{\partial x^i}(p)(x^i - p^i) = 0.$$

One interpretation of this equation is that the gradient vector $\langle \partial f/\partial x^1(p), \partial f/\partial x^2(p), \partial f/\partial x^3(p) \rangle$ of $f$ at $p$ is normal to any vector in the tangent plane.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.17</span><span class="math-callout__name">(Tangent plane to a sphere)</span></p>

Let $f(x, y, z) = x^2 + y^2 + z^2 - 1$. To get the equation of the tangent plane to the unit sphere $S^2 = f^{-1}(0)$ in $\mathbb{R}^3$ at $(a, b, c) \in S^2$, we compute

$$\frac{\partial f}{\partial x} = 2x, \quad \frac{\partial f}{\partial y} = 2y, \quad \frac{\partial f}{\partial z} = 2z.$$

At $p = (a, b, c)$,

$$\frac{\partial f}{\partial x}(p) = 2a, \quad \frac{\partial f}{\partial y}(p) = 2b, \quad \frac{\partial f}{\partial z}(p) = 2c.$$

By the equation of the tangent plane, the tangent plane to the sphere at $(a, b, c)$ is

$$2a(x - a) + 2b(y - b) + 2c(z - c) = 0,$$

or

$$ax + by + cz = 1,$$

since $a^2 + b^2 + c^2 = 1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.16</span><span class="math-callout__name">(Multiplication map of $\operatorname{SL}(n, \mathbb{R})$)</span></p>

The multiplication map

$$\mu \colon \operatorname{GL}(n, \mathbb{R}) \times \operatorname{GL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R}), \quad (A, B) \mapsto AB$$

is clearly $C^\infty$ because

$$(AB)_{ij} = \sum_{k=1}^n a_{ik} b_{kj}$$

is a polynomial and hence a $C^\infty$ function of the coordinates $a_{ik}$ and $b_{kj}$. However, one cannot conclude in the same way that the multiplication map

$$\bar{\mu} \colon \operatorname{SL}(n, \mathbb{R}) \times \operatorname{SL}(n, \mathbb{R}) \to \operatorname{SL}(n, \mathbb{R})$$

is $C^\infty$. This is because $\lbrace a_{ij} \rbrace_{1 \le i, j \le n}$ is not a coordinate system on $\operatorname{SL}(n, \mathbb{R})$; there is one coordinate too many (see Problem 11.6).

Since $\operatorname{SL}(n, \mathbb{R}) \times \operatorname{SL}(n, \mathbb{R})$ is a regular submanifold of $\operatorname{GL}(n, \mathbb{R}) \times \operatorname{GL}(n, \mathbb{R})$, the inclusion map

$$i \colon \operatorname{SL}(n, \mathbb{R}) \times \operatorname{SL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R}) \times \operatorname{GL}(n, \mathbb{R})$$

is $C^\infty$ by Theorem 11.14; therefore, the composition

$$\mu \circ i \colon \operatorname{SL}(n, \mathbb{R}) \times \operatorname{SL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R})$$

is also $C^\infty$. Because the image of $\mu \circ i$ lies in $\operatorname{SL}(n, \mathbb{R})$, and $\operatorname{SL}(n, \mathbb{R})$ is a regular submanifold of $\operatorname{GL}(n, \mathbb{R})$ (see Example 9.13), by Theorem 11.15 the induced map

$$\bar{\mu} \colon \operatorname{SL}(n, \mathbb{R}) \times \operatorname{SL}(n, \mathbb{R}) \to \operatorname{SL}(n, \mathbb{R})$$

is $C^\infty$.

</div>

## §12 The Tangent Bundle

A smooth vector bundle over a smooth manifold $M$ is a smoothly varying family of vector spaces, parametrized by $M$, that locally looks like a product. The collection of tangent spaces to a manifold has the structure of a vector bundle over the manifold, called the *tangent bundle*. A smooth map between two manifolds induces, via its differential at each point, a bundle map of the corresponding tangent bundles. Thus, the tangent bundle construction is a functor from the category of smooth manifolds to the category of vector bundles.

For us in this book the importance of the vector bundle point of view comes from its role in unifying concepts. A *section* of a vector bundle $\pi \colon E \to M$ is a map from $M$ to $E$ that maps each point of $M$ into the fiber of the bundle over the point. As we shall see, both vector fields and differential forms on a manifold are sections of vector bundles over the manifold.

### 12.1 The Topology of the Tangent Bundle

Let $M$ be a smooth manifold. Recall that at each point $p \in M$, the tangent space $T_pM$ is the vector space of all point-derivations of $C_p^\infty(M)$, the algebra of germs of $C^\infty$ functions at $p$. The **tangent bundle** of $M$ is the union of all the tangent spaces of $M$:

$$TM = \bigsqcup_{p \in M} T_pM.$$

There is a natural map $\pi \colon TM \to M$ given by $\pi(v) = p$ if $v \in T_pM$. We sometimes write a tangent vector $v \in T_pM$ as a pair $(p, v)$, to make explicit the point $p \in M$ at which $v$ is a tangent vector.

As defined, $TM$ is a set, with no topology or manifold structure. We will make it into a smooth manifold and show that it is a $C^\infty$ vector bundle over $M$.

If $(U, \phi) = (U, x^1, \dots, x^n)$ is a coordinate chart on $M$, let

$$TU = \bigcup_{p \in U} T_pU = \bigcup_{p \in U} T_pM.$$

(We saw in Remark 8.2 that $T_pU = T_pM$.) At a point $p \in U$, a basis for $T_pM$ is the set of coordinate vectors $\partial/\partial x^1|_p, \dots, \partial/\partial x^n|_p$, so a tangent vector $v \in T_pM$ is uniquely a linear combination

$$v = \sum_i c^i \frac{\partial}{\partial x^i}\bigg|_p.$$

In this expression, the coefficients $c^i = c^i(v)$ depend on $v$ and so are functions on $TU$. Let $\tilde{x}^i = x^i \circ \pi$ and define the map $\tilde{\phi} \colon TU \to \phi(U) \times \mathbb{R}^n$ by

$$v \mapsto (x^1(p), \dots, x^n(p), c^1(v), \dots, c^n(v)) = (\tilde{x}^1, \dots, \tilde{x}^n, c^1, \dots, c^n)(v).$$

Then $\tilde{\phi}$ has inverse

$$(\phi(p), c^1, \dots, c^n) \mapsto \sum c^i \frac{\partial}{\partial x^i}\bigg|_p$$

and is therefore a bijection. This means we can use $\tilde{\phi}$ to transfer the topology of $\phi(U) \times \mathbb{R}^n$ to $TU$: a set $A$ in $TU$ is open if and only if $\tilde{\phi}(A)$ is open in $\phi(U) \times \mathbb{R}^n$, where $\phi(U) \times \mathbb{R}^n$ is given its standard topology as an open subset of $\mathbb{R}^{2n}$. So another way to describe $\tilde{\phi}$ is $\tilde{\phi} = (\phi \circ \pi, \phi_*)$.

Let $\mathcal{B}$ be the collection of all open subsets of $T(U_\alpha)$ as $U_\alpha$ runs over all coordinate open sets in $M$:

$$\mathcal{B} = \bigcup_\alpha \lbrace A \mid A \text{ open in } T(U_\alpha),\; U_\alpha \text{ a coordinate open set in } M \rbrace.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 12.1</span></p>

**(i)** For any manifold $M$, the set $TM$ is the union of all $A \in \mathcal{B}$.

**(ii)** Let $U$ and $V$ be coordinate open sets in a manifold $M$. If $A$ is open in $TU$ and $B$ is open in $TV$, then $A \cap B$ is open in $T(U \cap V)$.

</div>

It follows from this lemma that the collection $\mathcal{B}$ satisfies the conditions of Proposition A.8 for a collection of subsets to be a basis for some topology on $TM$. We give the tangent bundle $TM$ the topology generated by the basis $\mathcal{B}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 12.2</span></p>

A manifold $M$ has a countable basis consisting of coordinate open sets.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12.3</span></p>

The tangent bundle $TM$ of a manifold $M$ is second countable.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $\lbrace U_i \rbrace_{i=1}^\infty$ be a countable basis for $M$ consisting of coordinate open sets. Let $\phi_i$ be the coordinate map on $U_i$. Since $TU_i$ is homeomorphic to the open subset $\phi_i(U_i) \times \mathbb{R}^n$ of $\mathbb{R}^{2n}$ and any subset of a Euclidean space is second countable (Example A.13 and Proposition A.14), $TU_i$ is second countable. For each $i$, choose a countable basis $\lbrace B_{i,j} \rbrace_{j=1}^n$ for $TU_i$. Then $\lbrace B_{i,j} \rbrace_{i,j=1}^n$ is a countable basis for the tangent bundle. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12.4</span></p>

The tangent bundle $TM$ of a manifold $M$ is Hausdorff.

</div>

### 12.2 The Manifold Structure on the Tangent Bundle

Next we show that if $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ is a $C^\infty$ atlas for $M$, then $\lbrace (TU_\alpha, \tilde{\phi}_\alpha) \rbrace$ is a $C^\infty$ atlas for the tangent bundle $TM$, where $\tilde{\phi}_\alpha$ is the map on $TU_\alpha$ induced by $\phi_\alpha$ as in (12.1). It is clear that $TM = \bigcup_\alpha TU_\alpha$. It remains to check that on $(TU_\alpha) \cap (TU_\beta)$, $\tilde{\phi}_\alpha$ and $\tilde{\phi}_\beta$ are $C^\infty$ compatible.

Recall that if $(U, x^1, \dots, x^n)$, $(V, y^1, \dots, y^n)$ are two charts on $M$, then for any $p \in U \cap V$ there are two bases singled out for the tangent space $T_pM$: $\lbrace \partial/\partial x^j|_p \rbrace_{j=1}^n$ and $\lbrace \partial/\partial y^i|_p \rbrace_{i=1}^n$. So any tangent vector $v \in T_pM$ has two descriptions:

$$v = \sum_j a^j \frac{\partial}{\partial x^j}\bigg|_p = \sum_i b^i \frac{\partial}{\partial y^i}\bigg|_p.$$

By applying both sides to $x^k$ and $y^k$ respectively, we find

$$a^k = \sum_i b^i \frac{\partial x^k}{\partial y^i}, \qquad b^k = \sum_j a^j \frac{\partial y^k}{\partial x^j}.$$

Returning to the atlas $\lbrace (U_\alpha, \phi_\alpha) \rbrace$, we write $U_{\alpha\beta} = U_\alpha \cap U_\beta$, $\phi_\alpha = (x^1, \dots, x^n)$ and $\phi_\beta = (y^1, \dots, y^n)$. Then

$$\tilde{\phi}_\beta \circ \tilde{\phi}_\alpha^{-1} \colon \phi_\alpha(U_{\alpha\beta}) \times \mathbb{R}^n \to \phi_\beta(U_{\alpha\beta}) \times \mathbb{R}^n$$

is given by

$$\bigl(\phi_\alpha(p), a^1, \dots, a^n\bigr) \mapsto \left((\phi_\beta \circ \phi_\alpha^{-1})(\phi_\alpha(p)),\, b^1, \dots, b^n\right),$$

where by (12.3) and Example 6.24,

$$b^i = \sum_j a^j \frac{\partial y^i}{\partial x^j}(p) = \sum_j a^j \frac{\partial(\phi_\beta \circ \phi_\alpha^{-1})^i}{\partial r^j}(\phi_\alpha(p)).$$

By the definition of an atlas, $\phi_\beta \circ \phi_\alpha^{-1}$ is $C^\infty$. Therefore, $\tilde{\phi}_\beta \circ \tilde{\phi}_\alpha^{-1}$ is $C^\infty$. This completes the proof that the tangent bundle $TM$ is a $C^\infty$ manifold, with $\lbrace (TU_\alpha, \tilde{\phi}_\alpha) \rbrace$ as a $C^\infty$ atlas.

### 12.3 Vector Bundles

On the tangent bundle $TM$ of a smooth manifold $M$, the natural projection map $\pi \colon TM \to M$, $\pi(p, v) = p$ makes $TM$ into a $C^\infty$ *vector bundle* over $M$, which we now define.

Given any map $\pi \colon E \to M$, we call the inverse image $\pi^{-1}(p) := \pi^{-1}(\lbrace p \rbrace)$ of a point $p \in M$ the **fiber** at $p$. The fiber at $p$ is often written $E_p$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.5</span><span class="math-callout__name">($C^\infty$ vector bundle)</span></p>

A surjective smooth map $\pi \colon E \to M$ of manifolds is said to be **locally trivial of rank $r$** if

1. each fiber $\pi^{-1}(p)$ has the structure of a vector space of dimension $r$;
2. for each $p \in M$, there are an open neighborhood $U$ of $p$ and a fiber-preserving diffeomorphism $\phi \colon \pi^{-1}(U) \to U \times \mathbb{R}^r$ such that for every $q \in U$ the restriction

$$\phi|_{\pi^{-1}(q)} \colon \pi^{-1}(q) \to \lbrace q \rbrace \times \mathbb{R}^r$$

is a vector space isomorphism. Such an open set $U$ is called a **trivializing open set** for $E$, and $\phi$ is called a **trivialization** of $E$ over $U$.

The collection $\lbrace (U, \phi) \rbrace$, with $\lbrace U \rbrace$ an open cover of $M$, is called a **local trivialization** for $E$, and $\lbrace U \rbrace$ is called a **trivializing open cover** of $M$ for $E$.

A **$C^\infty$ vector bundle of rank $r$** is a triple $(E, M, \pi)$ consisting of manifolds $E$ and $M$ and a surjective smooth map $\pi \colon E \to M$ that is locally trivial of rank $r$. The manifold $E$ is called the **total space** of the vector bundle and $M$ the **base space**. By abuse of language, we say that $E$ is a *vector bundle over $M$*.

</div>

For any regular submanifold $S \subset M$, the triple $(\pi^{-1}S, S, \pi|_{\pi^{-1}S})$ is a $C^\infty$ vector bundle over $S$, called the **restriction** of $E$ to $S$. We will often write the restriction as $E|_S$ instead of $\pi^{-1}S$.

Properly speaking, the tangent bundle of a manifold $M$ is a triple $(TM, M, \pi)$, and $TM$ is the total space of the tangent bundle. In common usage, $TM$ is often referred to as the tangent bundle.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 12.6</span><span class="math-callout__name">(Product bundle)</span></p>

Given a manifold $M$, let $\pi \colon M \times \mathbb{R}^r \to M$ be the projection to the first factor. Then $M \times \mathbb{R}^r \to M$ is a vector bundle of rank $r$, called the **product bundle** of rank $r$ over $M$. The vector space structure on the fiber $\pi^{-1}(p) = \lbrace (p, v) \mid v \in \mathbb{R}^r \rbrace$ is the obvious one:

$$(p, u) + (p, v) = (p, u + v), \quad b \cdot (p, v) = (p, bv) \text{ for } b \in \mathbb{R}.$$

A local trivialization on $M \times R$ is given by the identity map $\mathbb{1}_{M \times \mathbb{R}} \colon M \times \mathbb{R} \to M \times \mathbb{R}$. The infinite cylinder $S^1 \times \mathbb{R}$ is the product bundle of rank $1$ over the circle.

</div>

Let $\pi \colon E \to M$ be a $C^\infty$ vector bundle. Suppose $(U, \psi) = (U, x^1, \dots, x^n)$ is a chart on $M$ and

$$\phi \colon E|_U \xrightarrow{\sim} U \times \mathbb{R}^r, \quad \phi(e) = (\pi(e), c^1(e), \dots, c^r(e)),$$

is a trivialization of $E$ over $U$. Then

$$(\psi \times \mathbb{1}) \circ \phi = (x^1, \dots, x^n, c^1, \dots, c^r) \colon E|_U \xrightarrow{\sim} \psi(U) \times \mathbb{R}^r \subset \mathbb{R}^n \times \mathbb{R}^r$$

is a diffeomorphism of $E|_U$ onto its image and so is a chart on $E$. We call $x^1, \dots, x^n$ the **base coordinates** and $c^1, \dots, c^r$ the **fiber coordinates** of the chart $(E|_U, (\psi \times \mathbb{1}) \circ \phi)$ on $E$.

**Bundle maps.** Let $\pi_E \colon E \to M$, $\pi_F \colon F \to N$ be two vector bundles, possibly of different ranks. A **bundle map** from $E$ to $F$ is a pair of maps $(f, \tilde{f})$, $f \colon M \to N$ and $\tilde{f} \colon E \to F$, such that

1. the diagram

$$E \xrightarrow{\tilde{f}} F$$

$$\pi_E \downarrow \qquad \downarrow \pi_F$$

$$M \xrightarrow{f} N$$

is commutative, meaning $\pi_F \circ \tilde{f} = f \circ \pi_E$;

2. $\tilde{f}$ is linear on each fiber; i.e., for each $p \in M$, $\tilde{f} \colon E_p \to F_{f(p)}$ is a linear map of vector spaces.

The collection of all vector bundles together with bundle maps between them forms a category.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tangent bundle as a functor)</span></p>

A smooth map $f \colon N \to M$ of manifolds induces a bundle map $(f, \tilde{f})$, where $\tilde{f} \colon TN \to TM$ is given by

$$\tilde{f}(p, v) = (f(p), f_{*}(v)) \in \lbrace f(p) \rbrace \times T_{f(p)}M \subset TM$$

for all $v \in T_pN$. This gives rise to a covariant functor $T$ from the category of smooth manifolds and smooth maps to the category of vector bundles and bundle maps: to each manifold $M$, we associate its tangent bundle $T(M)$, and to each $C^\infty$ map $f \colon N \to M$ of manifolds, we associate the bundle map $T(f) = (f \colon N \to M, \tilde{f} \colon T(N) \to T(M))$.

</div>

If $E$ and $F$ are two vector bundles over the same manifold $M$, then a bundle map from $E$ to $F$ *over $M$* is a bundle map in which the base map is the identity $\mathbb{1}_M$. For a fixed manifold $M$, we can also consider the category of all $C^\infty$ vector bundles over $M$ and $C^\infty$ bundle maps over $M$. In this category it makes sense to speak of an isomorphism of vector bundles *over $M$*. Any vector bundle over $M$ isomorphic over $M$ to the product bundle $M \times \mathbb{R}^r$ is called a **trivial bundle**.

### 12.4 Smooth Sections

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Section of a vector bundle)</span></p>

A **section** of a vector bundle $\pi \colon E \to M$ is a map $s \colon M \to E$ such that $\pi \circ s = \mathbb{1}_M$, the identity map on $M$. This condition means precisely that for each $p$ in $M$, $s$ maps $p$ into the fiber $E_p$ above $p$. We say that a section is **smooth** if it is smooth as a map from $M$ to $E$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.7</span><span class="math-callout__name">(Vector field on a manifold)</span></p>

A **vector field** $X$ on a manifold $M$ is a function that assigns a tangent vector $X_p \in T_pM$ to each point $p \in M$. In terms of the tangent bundle, a vector field on $M$ is simply a section of the tangent bundle $\pi \colon TM \to M$ and the vector field is **smooth** if it is smooth as a map from $M$ to $TM$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 12.8</span></p>

The formula

$$X_{(x,y)} = -y \frac{\partial}{\partial x} + x \frac{\partial}{\partial y} = \begin{bmatrix} -y \\ x \end{bmatrix}$$

defines a smooth vector field on $\mathbb{R}^2$ (cf. Example 2.3).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12.9</span><span class="math-callout__name">(Smoothness of sections)</span></p>

Let $s$ and $t$ be $C^\infty$ sections of a $C^\infty$ vector bundle $\pi \colon E \to M$ and let $f$ be a $C^\infty$ real-valued function on $M$. Then

**(i)** the sum $s + t \colon M \to E$ defined by

$$(s + t)(p) = s(p) + t(p) \in E_p, \quad p \in M,$$

is a $C^\infty$ section of $E$.

**(ii)** the product $fs \colon M \to E$ defined by

$$(fs)(p) = f(p)s(p) \in E_p, \quad p \in M,$$

is a $C^\infty$ section of $E$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**(i)** It is clear that $s + t$ is a section of $E$. To show that it is $C^\infty$, fix a point $p \in M$ and let $V$ be a trivializing open set for $E$ containing $p$, with $C^\infty$ trivialization

$$\phi \colon \pi^{-1}(V) \to V \times \mathbb{R}^r.$$

Suppose

$$(\phi \circ s)(q) = (q, a^1(q), \dots, a^r(q))$$

and

$$(\phi \circ t)(q) = (q, b^1(q), \dots, b^r(q))$$

for $q \in V$. Because $s$ and $t$ are $C^\infty$ maps, $a^i$ and $b^i$ are $C^\infty$ functions on $V$ (Proposition 6.16). Since $\phi$ is linear on each fiber,

$$(\phi \circ (s + t))(q) = (q, a^1(q) + b^1(q), \dots, a^r(q) + b^r(q)), \quad q \in V.$$

This proves that $s + t$ is a $C^\infty$ map on $V$ and hence at $p$. Since $p$ is an arbitrary point of $M$, the section $s + t$ is $C^\infty$ on $M$.

**(ii)** We omit the proof, since it is similar to that of (i). $\square$

</details>
</div>

Denote the set of all $C^\infty$ sections of $E$ by $\Gamma(E)$. The proposition shows that $\Gamma(E)$ is not only a vector space over $\mathbb{R}$, but also a module over the ring $C^\infty(M)$ of $C^\infty$ functions on $M$. For any open subset $U \subset M$, one can also consider the vector space $\Gamma(U, E)$ of $C^\infty$ sections of $E$ over $U$. Then $\Gamma(U, E)$ is both a vector space over $\mathbb{R}$ and a $C^\infty(U)$-module. Note that $\Gamma(M, E) = \Gamma(E)$. To contrast with sections over a proper subset $U$, a section over the entire manifold $M$ is called a **global section**.

### 12.5 Smooth Frames

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Frame)</span></p>

A **frame** for a vector bundle $\pi \colon E \to M$ over an open set $U$ is a collection of sections $s_1, \dots, s_r$ of $E$ over $U$ such that at each point $p \in U$, the elements $s_1(p), \dots, s_r(p)$ form a basis for the fiber $E_p := \pi^{-1}(p)$. A frame $s_1, \dots, s_r$ is said to be **smooth** or **$C^\infty$** if $s_1, \dots, s_r$ are $C^\infty$ as sections of $E$ over $U$. A frame for the tangent bundle $TM \to M$ over an open set $U$ is simply called a **frame on $U$**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Standard frame on $\mathbb{R}^3$)</span></p>

The collection of vector fields $\partial/\partial x, \partial/\partial y, \partial/\partial z$ is a smooth frame on $\mathbb{R}^3$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Frame of a product bundle)</span></p>

Let $M$ be a manifold and $e_1, \dots, e_r$ the standard basis for $\mathbb{R}^r$. Define $\bar{e}_i \colon M \to M \times \mathbb{R}^r$ by $\bar{e}_i(p) = (p, e_i)$. Then $\bar{e}_1, \dots, \bar{e}_r$ is a $C^\infty$ frame for the product bundle $M \times \mathbb{R}^r \to M$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 12.10</span><span class="math-callout__name">(The frame of a trivialization)</span></p>

Let $\pi \colon E \to M$ be a smooth vector bundle of rank $r$. If $\phi \colon E|_U \xrightarrow{\sim} U \times \mathbb{R}^r$ is a trivialization of $E$ over an open set $U$, then $\phi^{-1}$ carries the $C^\infty$ frame $\bar{e}_1, \dots, \bar{e}_r$ of the product bundle $U \times \mathbb{R}^r$ to a $C^\infty$ frame $t_1, \dots, t_r$ for $E$ over $U$:

$$t_i(p) = \phi^{-1}(\bar{e}_i(p)) = \phi^{-1}(p, e_i), \quad p \in U.$$

We call $t_1, \dots, t_r$ the $C^\infty$ **frame over $U$ of the trivialization** $\phi$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 12.11</span></p>

Let $\phi \colon E|_U \to U \times \mathbb{R}^r$ be a trivialization over an open set $U$ of a $C^\infty$ vector bundle $E \to M$, and $t_1, \dots, t_r$ the $C^\infty$ frame over $U$ of the trivialization. Then a section $s = \sum b^i t_i$ of $E$ over $U$ is $C^\infty$ if and only if its coefficients $b^i$ relative to the frame $t_1, \dots, t_r$ are $C^\infty$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12.12</span><span class="math-callout__name">(Characterization of $C^\infty$ sections)</span></p>

Let $\pi \colon E \to M$ be a $C^\infty$ vector bundle and $U$ an open subset of $M$. Suppose $s_1, \dots, s_r$ is a $C^\infty$ frame for $E$ over $U$. Then a section $s = \sum c^j s_j$ of $E$ over $U$ is $C^\infty$ if and only if the coefficients $c^j$ are $C^\infty$ functions on $U$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

If $s_1, \dots, s_r$ is the frame of a trivialization of $E$ over $U$, then the proposition is Lemma 12.11. We prove the proposition in general by reducing it to this case. One direction is quite easy. If the $c^j$'s are $C^\infty$ functions on $U$, then $s = \sum c^j s_j$ is a $C^\infty$ section on $U$ by Proposition 12.9.

Conversely, suppose $s = \sum c^j s_j$ is a $C^\infty$ section of $E$ over $U$. Fix a point $p \in U$ and choose a trivializing open set $V \subset U$ for $E$ containing $p$, with $C^\infty$ trivialization $\phi \colon \pi^{-1}(V) \to V \times \mathbb{R}^r$. Let $t_1, \dots, t_r$ be the $C^\infty$ frame of the trivialization $\phi$ (Example 12.10). If we write $s$ and $s_j$ in terms of the frame $t_1, \dots, t_r$, say $s = \sum b^i t_i$ and $s_j = \sum a_j^i t_i$, the coefficients $b^i$, $a_j^i$ will all be $C^\infty$ functions on $V$ by Lemma 12.11. Next, express $s = \sum c^j s_j$ in terms of the $t_i$'s:

$$\sum b^i t_i = s = \sum c^j s_j = \sum_{i,j} c^j a_j^i t_i.$$

Comparing the coefficients of $t_i$ gives $b^i = \sum_j c^j a_j^i$. In matrix notation,

$$b = Ac.$$

At each point of $V$, being the transition matrix between two bases, the matrix $A$ is invertible. By Cramer's rule, $A^{-1}$ is a matrix of $C^\infty$ functions on $V$ (see Example 6.21). Hence, $c = A^{-1}b$ is a column vector of $C^\infty$ functions on $V$. This proves that $c^1, \dots, c^r$ are $C^\infty$ functions at $p \in U$. Since $p$ is an arbitrary point of $U$, the coefficients $c^j$ are $C^\infty$ functions on $U$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 12.13</span></p>

If one replaces "smooth" by "continuous" throughout, the discussion in this subsection remains valid in the continuous category.

</div>

## §13 Bump Functions and Partitions of Unity

A partition of unity on a manifold is a collection of nonnegative functions that sum to $1$. Usually one demands in addition that the partition of unity be *subordinate* to an open cover $\lbrace U_\alpha \rbrace_{\alpha \in A}$. What this means is that the partition of unity $\lbrace \rho_\alpha \rbrace_{\alpha \in A}$ is indexed by the same set as the open cover $\lbrace U_\alpha \rbrace_{\alpha \in A}$ and for each $\alpha$ in the index $A$, the support of $\rho_\alpha$ is contained in $U_\alpha$. In particular, $\rho_\alpha$ vanishes outside $U_\alpha$.

The existence of a $C^\infty$ partition of unity is one of the most important technical tools in the theory of $C^\infty$ manifolds. It is the single feature that makes the behavior of $C^\infty$ manifolds so different from that of real-analytic or complex manifolds. In this section we construct $C^\infty$ bump functions on any manifold and prove the existence of a $C^\infty$ partition of unity on a compact manifold. The proof of the existence of a $C^\infty$ partition of unity on a general manifold is more technical and is postponed to Appendix C.

A partition of unity is used in two ways: (1) to decompose a global object on a manifold into a locally finite sum of local objects on the open sets $U_\alpha$ of an open cover, and (2) to patch together local objects on the open sets $U_\alpha$ into a global object on the manifold. Thus, a partition of unity serves as a bridge between global and local analysis on a manifold.

### 13.1 $C^\infty$ Bump Functions

Recall that $\mathbb{R}^\times$ denotes the set of nonzero real numbers. The **support** of a real-valued function $f$ on a manifold $M$ is defined to be the closure in $M$ of the subset on which $f \neq 0$:

$$\operatorname{supp} f = \operatorname{cl}_M(f^{-1}(\mathbb{R}^\times)) = \text{closure of } \lbrace q \in M \mid f(q) \neq 0 \rbrace \text{ in } M.$$

Let $q$ be a point in $M$, and $U$ a neighborhood of $q$. By a **bump function at $q$ supported in $U$** we mean any continuous nonnegative function $\rho$ on $M$ that is $1$ in a neighborhood of $q$ with $\operatorname{supp} \rho \subset U$.

The only bump functions of interest to us are $C^\infty$ bump functions. While the continuity of a function can often be seen by inspection, the smoothness of a function always requires a formula. Our goal in this subsection is to find a formula for a $C^\infty$ bump function.

The main challenge in building a smooth bump function from $f$ is to construct a smooth version of a step function, that is, a $C^\infty$ function $g \colon \mathbb{R} \to \mathbb{R}$ with

$$g(t) = \begin{cases} 0 & \text{for } t \le 0, \\ 1 & \text{for } t \ge 1. \end{cases}$$

In Example 1.3 we introduced the $C^\infty$ function

$$f(t) = \begin{cases} e^{-1/t} & \text{for } t > 0, \\ 0 & \text{for } t \le 0. \end{cases}$$

We seek $g(t)$ by dividing $f(t)$ by a positive function $\ell(t)$, for the quotient $f(t)/\ell(t)$ will then be zero for $t \le 0$. The denominator $\ell(t)$ should be a positive function that agrees with $f(t)$ for $t \ge 1$, for then $f(t)/\ell(t)$ will be identically $1$ for $t \ge 1$. The simplest way to construct such an $\ell(t)$ is to add to $f(t)$ a nonnegative function that vanishes for $t \ge 1$. One such nonnegative function is $f(1 - t)$. This suggests that we take $\ell(t) = f(t) + f(1 - t)$ and consider

$$g(t) = \frac{f(t)}{f(t) + f(1 - t)}.$$

One can verify that $f(t) + f(1 - t)$ is never zero, so $g(t)$ is $C^\infty$ for all $t$. Moreover, $g$ is identically zero for $t \le 0$ and identically $1$ for $t \ge 1$. Thus, $g$ is a $C^\infty$ step function with the desired properties.

Given two positive real numbers $a < b$, we make a linear change of variables to map $[a^2, b^2]$ to $[0, 1]$:

$$x \mapsto \frac{x - a^2}{b^2 - a^2}.$$

Let

$$h(x) = g\!\left(\frac{x - a^2}{b^2 - a^2}\right).$$

Then $h \colon \mathbb{R} \to [0, 1]$ is a $C^\infty$ step function such that

$$h(x) = \begin{cases} 0 & \text{for } x \le a^2, \\ 1 & \text{for } x \ge b^2. \end{cases}$$

Replace $x$ by $x^2$ to make the function symmetric in $x$: $k(x) = h(x^2)$. Finally, set

$$\rho(x) = 1 - k(x) = 1 - g\!\left(\frac{x^2 - a^2}{b^2 - a^2}\right).$$

This $\rho(x)$ is a $C^\infty$ bump function at $0$ in $\mathbb{R}$ that is identically $1$ on $[-a, a]$ and has support in $[-b, b]$. For any $q \in \mathbb{R}$, $\rho(x - q)$ is a $C^\infty$ bump function at $q$.

It is easy to extend the construction of a bump function from $\mathbb{R}$ to $\mathbb{R}^n$. To get a $C^\infty$ bump function at $\mathbf{0}$ in $\mathbb{R}^n$ that is $1$ on the closed ball $\overline{B}(\mathbf{0}, a)$ and has support in the closed ball $\overline{B}(\mathbf{0}, b)$, set

$$\sigma(x) = \rho(\|x\|) = 1 - g\!\left(\frac{\|x\|^2 - a^2}{b^2 - a^2}\right).$$

As a composition of $C^\infty$ functions, $\sigma$ is $C^\infty$. To get a $C^\infty$ bump function at $q$ in $\mathbb{R}^n$, take $\sigma(x - q)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13.2</span><span class="math-callout__name">($C^\infty$ extension of a function)</span></p>

Suppose $f$ is a $C^\infty$ function defined on a neighborhood $U$ of a point $p$ in a manifold $M$. Then there is a $C^\infty$ function $\tilde{f}$ on $M$ that agrees with $f$ in some possibly smaller neighborhood of $p$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Choose a $C^\infty$ bump function $\rho \colon M \to \mathbb{R}$ supported in $U$ that is identically $1$ in a neighborhood $V$ of $p$. Define

$$\tilde{f}(q) = \begin{cases} \rho(q) f(q) & \text{for } q \text{ in } U, \\ 0 & \text{for } q \text{ not in } U. \end{cases}$$

As the product of two $C^\infty$ functions on $U$, $\tilde{f}$ is $C^\infty$ on $U$. If $q \notin U$, then $q \notin \operatorname{supp} \rho$, and so there is an open set containing $q$ on which $\tilde{f}$ is $0$, since $\operatorname{supp} \rho$ is closed. Therefore, $\tilde{f}$ is also $C^\infty$ at every point $q \notin U$.

Finally, since $\rho \equiv 1$ on $V$, the function $\tilde{f}$ agrees with $f$ on $V$. $\square$

</details>
</div>

### 13.2 Partitions of Unity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Locally finite collection)</span></p>

A collection $\lbrace A_\alpha \rbrace$ of subsets of a topological space $S$ is said to be **locally finite** if every point $q$ in $S$ has a neighborhood that meets only finitely many of the sets $A_\alpha$. In particular, every $q$ in $S$ is contained in only finitely many of the $A_\alpha$'s.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 13.3</span><span class="math-callout__name">(An open cover that is not locally finite)</span></p>

Let $U_{r,n}$ be the open interval $]r - \frac{1}{n}, r + \frac{1}{n}[$ on the real line $\mathbb{R}$. The open cover $\lbrace U_{r,n} \mid r \in \mathbb{Q}, n \in \mathbb{Z}^+ \rbrace$ of $\mathbb{R}$ is not locally finite.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 13.4</span><span class="math-callout__name">($C^\infty$ partition of unity)</span></p>

A **$C^\infty$ partition of unity** on a manifold is a collection of nonnegative $C^\infty$ functions $\lbrace \rho_\alpha \colon M \to \mathbb{R} \rbrace_{\alpha \in A}$ such that

1. the collection of supports, $\lbrace \operatorname{supp} \rho_\alpha \rbrace_{\alpha \in A}$, is locally finite,
2. $\sum \rho_\alpha = 1$.

Given an open cover $\lbrace U_\alpha \rbrace_{\alpha \in A}$ of $M$, we say that a partition of unity $\lbrace \rho_\alpha \rbrace$ is **subordinate to the open cover** $\lbrace U_\alpha \rbrace$ if $\operatorname{supp} \rho_\alpha \subset U_\alpha$ for every $\alpha \in A$.

</div>

Since the collection of supports, $\lbrace \operatorname{supp} \rho_\alpha \rbrace_{\alpha \in A}$, is locally finite (condition (i)), every point $q$ lies in only finitely many of the sets $\operatorname{supp} \rho_\alpha$. Hence $\rho_\alpha(q) \neq 0$ for only finitely many $\alpha$. It follows that the sum in (ii) is a finite sum at every point.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Locally finite sum)</span></p>

Suppose $\lbrace f_\alpha \rbrace_{\alpha \in A}$ is a collection of $C^\infty$ functions on a manifold $M$ such that the collection of its supports, $\lbrace \operatorname{supp} f_\alpha \rbrace_{\alpha \in A}$, is locally finite. Then every point $q$ in $M$ has a neighborhood $W_q$ that intersects $\operatorname{supp} f_\alpha$ for only finitely many $\alpha$. Thus, on $W_q$ the sum $\sum_{\alpha \in A} f_\alpha$ is actually a finite sum. This shows that the function $f = \sum f_\alpha$ is well defined and $C^\infty$ on the manifold $M$. We call such a sum a **locally finite sum**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Partition of unity on $\mathbb{R}$)</span></p>

Let $U$ and $V$ be the open intervals $]-\infty, 2[$ and $]-1, \infty[$ in $\mathbb{R}$ respectively, and let $\rho_V$ be a $C^\infty$ function with graph as in Figure 13.9, for example the function $g(t)$ in (13.1). Define $\rho_U = 1 - \rho_V$. Then $\operatorname{supp} \rho_V \subset V$ and $\operatorname{supp} \rho_U \subset U$. Thus, $\lbrace \rho_U, \rho_V \rbrace$ is a partition of unity subordinate to the open cover $\lbrace U, V \rbrace$.

</div>

### 13.3 Existence of a Partition of Unity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 13.5</span></p>

If $\rho_1, \dots, \rho_m$ are real-valued functions on a manifold $M$, then

$$\operatorname{supp}\!\left(\sum \rho_i\right) \subset \bigcup \operatorname{supp} \rho_i.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13.6</span><span class="math-callout__name">(Partition of unity on a compact manifold)</span></p>

Let $M$ be a compact manifold and $\lbrace U_\alpha \rbrace_{\alpha \in A}$ an open cover of $M$. There exists a $C^\infty$ partition of unity $\lbrace \rho_\alpha \rbrace_{\alpha \in A}$ subordinate to $\lbrace U_\alpha \rbrace_{\alpha \in A}$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For each $q \in M$, find an open set $U_\alpha$ containing $q$ from the given cover and let $\psi_q$ be a $C^\infty$ bump function at $q$ supported in $U_\alpha$ (Exercise 13.1, p. 144). Because $\psi_q(q) > 0$, there is a neighborhood $W_q$ of $q$ on which $\psi_q > 0$. By the compactness of $M$, the open cover $\lbrace W_q \mid q \in M \rbrace$ has a finite subcover, $\lbrace W_{q_1}, \dots, W_{q_m} \rbrace$. Let $\psi_{q_1}, \dots, \psi_{q_m}$ be the corresponding bump functions. Then $\psi := \sum \psi_{q_i}$ is positive at every point $q$ in $M$ because $q \in W_{q_i}$ for some $i$. Define

$$\varphi_i = \frac{\psi_{q_i}}{\psi}, \quad i = 1, \dots, m.$$

Clearly, $\sum \varphi_i = 1$. Moreover, since $\psi > 0$, $\varphi_i(q) \neq 0$ if and only if $\psi_{q_i}(q) \neq 0$, so

$$\operatorname{supp} \varphi_i = \operatorname{supp} \psi_{q_i} \subset U_\alpha$$

for some $\alpha \in A$. This shows that $\lbrace \varphi_i \rbrace$ is a partition of unity such that for every $i$, $\operatorname{supp} \varphi_i \subset U_\alpha$ for some $\alpha \in A$.

The next step is to make the index set of the partition of unity the same as that of the open cover. For each $i = 1, \dots, m$, choose $\tau(i) \in A$ to be an index such that $\operatorname{supp} \varphi_i \subset U_{\tau(i)}$. We group the collection of functions $\lbrace \varphi_i \rbrace$ into subcollections according to $\tau(i)$ and define for each $\alpha \in A$,

$$\rho_\alpha = \sum_{\tau(i) = \alpha} \varphi_i;$$

if there is no $i$ for which $\tau(i) = \alpha$, the sum above is empty and we define $\rho_\alpha = 0$. Then

$$\sum_{\alpha \in A} \rho_\alpha = \sum_{\alpha \in A} \sum_{\tau(i) = \alpha} \varphi_i = \sum_{i=1}^m \varphi_i = 1.$$

Moreover, by Lemma 13.5,

$$\operatorname{supp} \rho_\alpha \subset \bigcup_{\tau(i) = \alpha} \operatorname{supp} \varphi_i \subset U_\alpha.$$

So $\lbrace \rho_\alpha \rbrace$ is a partition of unity subordinate to $\lbrace U_\alpha \rbrace$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13.7</span><span class="math-callout__name">(Existence of a $C^\infty$ partition of unity)</span></p>

Let $\lbrace U_\alpha \rbrace_{\alpha \in A}$ be an open cover of a manifold $M$.

**(i)** There is a $C^\infty$ partition of unity $\lbrace \varphi_k \rbrace_{k=1}^\infty$ with every $\varphi_k$ having compact support such that for each $k$, $\operatorname{supp} \varphi_k \subset U_\alpha$ for some $\alpha \in A$.

**(ii)** If we do not require compact support, then there is a $C^\infty$ partition of unity $\lbrace \rho_\alpha \rbrace$ subordinate to $\lbrace U_\alpha \rbrace$.

</div>

# Chapter 4 — Lie Groups and Lie Algebras

## §14 Vector Fields

A vector field $X$ on a manifold $M$ is the assignment of a tangent vector $X_p \in T_pM$ to each point $p \in M$. More formally, a vector field on $M$ is a section of the tangent bundle $TM$ of $M$. It is natural to define a vector field as smooth if it is smooth as a section of the tangent bundle.

Vector fields abound in nature, for example the velocity vector field of a fluid flow, the electric field of a charge, the gravitational field of a mass, and so on. The fluid flow model is in fact quite general, for as we will see shortly, every smooth vector field may be viewed locally as the velocity vector field of a fluid flow. The path traced out by a point under this flow is called an **integral curve** of the vector field. Integral curves are curves whose velocity vector field is the restriction of the given vector field to the curve. Finding the equation of an integral curve is equivalent to solving a system of first-order ordinary differential equations (ODE). Thus, the theory of ODE guarantees the existence of integral curves.

The set $\mathfrak{X}(M)$ of all $C^\infty$ vector fields on a manifold $M$ clearly has the structure of a vector space. We introduce a bracket operation $[\,,\,]$ that makes it into a Lie algebra. Because vector fields do not push forward under smooth maps, the Lie algebra $\mathfrak{X}(M)$ does not give rise to a functor on the category of smooth manifolds. Nonetheless, there is a notion of *related vector fields* that allows us to compare vector fields on two manifolds under a smooth map.

### 14.1 Smoothness of a Vector Field

In Definition 12.7 we defined a vector field $X$ on a manifold $M$ to be *smooth* if the map $X \colon M \to TM$ is smooth as a section of the tangent bundle $\pi \colon TM \to M$. In a coordinate chart $(U, \phi) = (U, x^1, \dots, x^n)$ on $M$, the value of the vector field $X$ at $p \in U$ is a linear combination

$$X_p = \sum a^i(p) \frac{\partial}{\partial x^i}\bigg\vert_p.$$

As $p$ varies in $U$, the coefficients $a^i$ become functions on $U$.

As we learned in Subsections 12.1 and 12.2, the chart $(U, \phi) = (U, x^1, \dots, x^n)$ on the manifold $M$ induces a chart

$$(TU, \tilde{\phi}) = (TU, \tilde{x}^1, \dots, \tilde{x}^n, c^1, \dots, c^n)$$

on the tangent bundle $TM$, where $\tilde{x}^i = \pi^* x^i = x^i \circ \pi$ and the $c^i$ are defined by

$$v = \sum c^i(v) \frac{\partial}{\partial x^i}\bigg\vert_p, \quad v \in T_pM.$$

Comparing coefficients in

$$X_p = \sum a^i(p) \frac{\partial}{\partial x^i}\bigg\vert_p = \sum c^i(X_p) \frac{\partial}{\partial x^i}\bigg\vert_p, \quad p \in U,$$

we get $a^i = c^i \circ X$ as functions on $U$. Being coordinates, the $c^i$ are smooth functions on $TU$. Thus, if $X$ is smooth and $(U, x^1, \dots, x^n)$ is any chart on $M$, then the coefficients $a^i$ of $X = \sum a^i \,\partial/\partial x^i$ relative to the frame $\partial/\partial x^i$ are smooth on $U$.

The converse is also true, as indicated in the following lemma.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 14.1</span><span class="math-callout__name">(Smoothness of a vector field on a chart)</span></p>

Let $(U, \phi) = (U, x^1, \dots, x^n)$ be a chart on a manifold $M$. A vector field $X = \sum a^i \,\partial/\partial x^i$ on $U$ is smooth if and only if the coefficient functions $a^i$ are all smooth on $U$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

This lemma is a special case of Proposition 12.12, with $E$ the tangent bundle of $M$ and $s_i$ the coordinate vector field $\partial/\partial x^i$.

Because we have an explicit description of the manifold structure on the tangent bundle $TM$, a direct proof of the lemma is also possible. Since $\tilde{\phi} \colon TU \to U \times \mathbb{R}^n$ is a diffeomorphism, $X \colon U \to TU$ is smooth if and only if $\tilde{\phi} \circ X \colon U \to U \times \mathbb{R}^n$ is smooth. For $p \in U$,

$$(\tilde{\phi} \circ X)(p) = \tilde{\phi}(X_p) = \bigl(x^1(p), \dots, x^n(p), c^1(X_p), \dots, c^n(X_p)\bigr) = \bigl(x^1(p), \dots, x^n(p), a^1(p), \dots, a^n(p)\bigr).$$

As coordinate functions, $x^1, \dots, x^n$ are $C^\infty$ on $U$. Therefore, by Proposition 6.13, $\tilde{\phi} \circ X$ is smooth if and only if all the $a^i$ are smooth on $U$. $\square$

</details>
</div>

This lemma leads to a characterization of the smoothness of a vector field on a manifold in terms of the coefficients of the vector field relative to coordinate frames.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.2</span><span class="math-callout__name">(Smoothness of a vector field in terms of coefficients)</span></p>

Let $X$ be a vector field on a manifold $M$. The following are equivalent:

**(i)** The vector field $X$ is smooth on $M$.

**(ii)** The manifold $M$ has an atlas such that on any chart $(U, \phi) = (U, x^1, \dots, x^n)$ of the atlas, the coefficients $a^i$ of $X = \sum a^i \,\partial/\partial x^i$ relative to the frame $\partial/\partial x^i$ are all smooth.

**(iii)** On any chart $(U, \phi) = (U, x^1, \dots, x^n)$ on the manifold $M$, the coefficients $a^i$ of $X = \sum a^i \,\partial/\partial x^i$ relative to the frame $\partial/\partial x^i$ are all smooth.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**(ii) $\Rightarrow$ (i):** Assume (ii). By the preceding lemma, $X$ is smooth on every chart $(U, \phi)$ of an atlas of $M$. Thus, $X$ is smooth on $M$.

**(i) $\Rightarrow$ (iii):** A smooth vector field on $M$ is smooth on every chart $(U, \phi)$ on $M$. The preceding lemma then implies (iii).

**(iii) $\Rightarrow$ (ii):** Obvious. $\square$

</details>
</div>

Just as in Subsection 2.5, a vector field $X$ on a manifold $M$ induces a linear map on the algebra $C^\infty(M)$ of $C^\infty$ functions on $M$: for $f \in C^\infty(M)$, define $Xf$ to be the function

$$(Xf)(p) = X_p f, \quad p \in M.$$

In terms of its action as an operator on $C^\infty$ functions, there is still another characterization of a smooth vector field.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.3</span><span class="math-callout__name">(Smoothness of a vector field in terms of functions)</span></p>

A vector field $X$ on $M$ is smooth if and only if for every smooth function $f$ on $M$, the function $Xf$ is smooth on $M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$(\Rightarrow)$ Suppose $X$ is smooth and $f \in C^\infty(M)$. By Proposition 14.2, on any chart $(U, x^1, \dots, x^n)$ on $M$, the coefficients $a^i$ of the vector field $X = \sum a^i \,\partial/\partial x^i$ are $C^\infty$. It follows that $Xf = \sum a^i \,\partial f/\partial x^i$ is $C^\infty$ on $U$. Since $M$ can be covered by charts, $Xf$ is $C^\infty$ on $M$.

$(\Leftarrow)$ Let $(U, x^1, \dots, x^n)$ be any chart on $M$. Suppose $X = \sum a^i \,\partial/\partial x^i$ on $U$ and $p \in U$. By Proposition 13.2, for $k = 1, \dots, n$, each $x^k$ can be extended to a $C^\infty$ function $\tilde{x}^k$ on $M$ that agrees with $x^k$ in a neighborhood $V$ of $p$ in $U$. Therefore, on $V$,

$$X\tilde{x}^k = \left(\sum a^i \frac{\partial}{\partial x^i}\right)\tilde{x}^k = \left(\sum a^i \frac{\partial}{\partial x^i}\right)x^k = a^k.$$

This proves that $a^k$ is $C^\infty$ at $p$. Since $p$ is an arbitrary point in $U$, the function $a^k$ is $C^\infty$ on $U$. By the smoothness criterion of Proposition 14.2, $X$ is smooth. $\square$

</details>
</div>

By Proposition 14.3, we may view a $C^\infty$ vector field $X$ as a linear operator $X \colon C^\infty(M) \to C^\infty(M)$ on the algebra of $C^\infty$ functions on $M$. As in Proposition 2.6, this linear operator $X \colon C^\infty(M) \to C^\infty(M)$ is a derivation: for all $f, g \in C^\infty(M)$,

$$X(fg) = (Xf)g + f(Xg).$$

In the following we think of $C^\infty$ vector fields on $M$ alternately as $C^\infty$ sections of the tangent bundle $TM$ and as derivations on the algebra $C^\infty(M)$ of $C^\infty$ functions. In fact, it can be shown that these two descriptions of $C^\infty$ vector fields are equivalent (Problem 19.12).

Proposition 13.2 on $C^\infty$ extensions of functions has an analogue for vector fields.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.4</span><span class="math-callout__name">($C^\infty$ extension of a vector field)</span></p>

Suppose $X$ is a $C^\infty$ vector field defined on a neighborhood $U$ of a point $p$ in a manifold $M$. Then there is a $C^\infty$ vector field $\tilde{X}$ on $M$ that agrees with $X$ on some possibly smaller neighborhood of $p$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Choose a $C^\infty$ bump function $\rho \colon M \to \mathbb{R}$ supported in $U$ that is identically $1$ in a neighborhood $V$ of $p$ (Figure 13.8). Define

$$\tilde{X}(q) = \begin{cases} \rho(q) X_q & \text{for } q \text{ in } U, \\ 0 & \text{for } q \text{ not in } U. \end{cases}$$

The rest of the proof is the same as in Proposition 13.2. $\square$

</details>
</div>

### 14.2 Integral Curves

In Example 12.8, it appears that through each point in the plane one can draw a circle whose velocity at any point is the given vector field at that point. Such a curve is an example of an **integral curve** of the vector field, which we now define.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.5</span><span class="math-callout__name">(Integral curve)</span></p>

Let $X$ be a $C^\infty$ vector field on a manifold $M$, and $p \in M$. An **integral curve** of $X$ is a smooth curve $c \colon \,]a, b[\, \to M$ such that $c'(t) = X_{c(t)}$ for all $t \in \,]a, b[\,$. Usually we assume that the open interval $]a, b[$ contains $0$. In this case, if $c(0) = p$, then we say that $c$ is an integral curve *starting at* $p$ and call $p$ the **initial point** of $c$. To show the dependence of such an integral curve on the initial point $p$, we also write $c_t(p)$ instead of $c(t)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.6</span><span class="math-callout__name">(Maximal integral curve)</span></p>

An integral curve is **maximal** if its domain cannot be extended to a larger interval.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Integral curve of $X_{(x,y)} = \langle -y, x \rangle$ on $\mathbb{R}^2$)</span></p>

Recall the vector field $X_{(x,y)} = \langle -y, x \rangle$ on $\mathbb{R}^2$ (Figure 12.4). We find an integral curve $c(t)$ of $X$ starting at the point $(1, 0) \in \mathbb{R}^2$. The condition for $c(t) = (x(t), y(t))$ to be an integral curve is $c'(t) = X_{c(t)}$, or

$$\begin{bmatrix} \dot{x}(t) \\ \dot{y}(t) \end{bmatrix} = \begin{bmatrix} -y(t) \\ x(t) \end{bmatrix},$$

so we need to solve the system of first-order ordinary differential equations

$$\dot{x} = -y, \qquad \dot{y} = x,$$

with initial condition $(x(0), y(0)) = (1, 0)$. From $\dot{x} = -y$, we get $y = -\dot{x}$, so $\dot{y} = -\ddot{x}$. Substituting into $\dot{y} = x$ gives $\ddot{x} = -x$. The general solution is

$$x = A\cos t + B\sin t, \qquad y = -\dot{x} = A\sin t - B\cos t.$$

The initial condition forces $A = 1$, $B = 0$, so the integral curve starting at $(1, 0)$ is $c(t) = (\cos t, \sin t)$, which parametrizes the unit circle.

More generally, if the initial point of the integral curve, corresponding to $t = 0$, is $p = (x_0, y_0)$, then $A = x_0$, $B = -y_0$, and the general solution is

$$x = x_0 \cos t - y_0 \sin t, \qquad y = x_0 \sin t + y_0 \cos t, \quad t \in \mathbb{R}.$$

This can be written in matrix notation as

$$c(t) = \begin{bmatrix} x(t) \\ y(t) \end{bmatrix} = \begin{bmatrix} \cos t & -\sin t \\ \sin t & \cos t \end{bmatrix} \begin{bmatrix} x_0 \\ y_0 \end{bmatrix},$$

which shows that the integral curve of $X$ starting at $p$ can be obtained by rotating the point $p$ counterclockwise about the origin through an angle $t$. Notice that

$$c_s(c_t(p)) = c_{s+t}(p),$$

since a rotation through an angle $t$ followed by a rotation through an angle $s$ is the same as a rotation through the angle $s + t$. For each $t \in \mathbb{R}$, $c_t \colon \mathbb{R}^2 \to \mathbb{R}^2$ is a diffeomorphism with inverse $c_{-t}$.

</div>

Let $\operatorname{Diff}(M)$ be the group of diffeomorphisms of a manifold $M$ with itself, the group operation being composition. A homomorphism $c \colon \mathbb{R} \to \operatorname{Diff}(M)$ is called a **one-parameter group of diffeomorphisms** of $M$. In this example the integral curves of the vector field $X_{(x,y)} = \langle -y, x \rangle$ on $\mathbb{R}^2$ give rise to a one-parameter group of diffeomorphisms of $\mathbb{R}^2$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Maximal integral curve of $x^2\,d/dx$)</span></p>

Let $X$ be the vector field $x^2\,d/dx$ on the real line $\mathbb{R}$. We find the maximal integral curve of $X$ starting at $x = 2$.

*Solution.* Denote the integral curve by $x(t)$. Then

$$x'(t) = X_{x(t)} \iff \dot{x}(t)\frac{d}{dx} = x^2 \frac{d}{dx},$$

so $x(t)$ satisfies the differential equation

$$\frac{dx}{dt} = x^2, \quad x(0) = 2.$$

By separation of variables: $dx/x^2 = dt$. Integrating both sides gives

$$-\frac{1}{x} = t + C, \quad \text{or} \quad x = -\frac{1}{t + C},$$

for some constant $C$. The initial condition $x(0) = 2$ forces $C = -1/2$. Hence, $x(t) = 2/(1 - 2t)$. The maximal interval containing $0$ on which $x(t)$ is defined is $]-\infty, 1/2[$.

From this example we see that it may not be possible to extend the domain of definition of an integral curve to the entire real line.

</div>

### 14.3 Local Flows

The two examples in the preceding section illustrate the fact that locally, finding an integral curve of a vector field amounts to solving a system of first-order ordinary differential equations with initial conditions. In general, if $X$ is a smooth vector field on a manifold $M$, to find an integral curve $c(t)$ of $X$ starting at $p$, we first choose a coordinate chart $(U, \phi) = (U, x^1, \dots, x^n)$ about $p$. In terms of the local coordinates,

$$X_{c(t)} = \sum a^i(c(t)) \frac{\partial}{\partial x^i}\bigg\vert_{c(t)},$$

and by Proposition 8.15,

$$c'(t) = \sum \dot{c}^i(t) \frac{\partial}{\partial x^i}\bigg\vert_{c(t)},$$

where $c^i(t) = x^i \circ c(t)$ is the $i$th component of $c(t)$ in the chart $(U, \phi)$. The condition $c'(t) = X_{c(t)}$ is thus equivalent to

$$\dot{c}^i(t) = a^i(c(t)) \quad \text{for } i = 1, \dots, n.$$

This is a system of ordinary differential equations (ODE); the initial condition $c(0) = p$ translates to $(c^1(0), \dots, c^n(0)) = (p^1, \dots, p^n)$. By an existence and uniqueness theorem from the theory of ODE, such a system always has a unique solution in the following sense.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.7</span><span class="math-callout__name">(Existence and uniqueness of ODE solutions)</span></p>

Let $V$ be an open subset of $\mathbb{R}^n$, $p_0$ a point in $V$, and $f \colon V \to \mathbb{R}^n$ a $C^\infty$ function. Then the differential equation

$$dy/dt = f(y), \quad y(0) = p_0,$$

has a unique $C^\infty$ solution $y \colon \,]a(p_0), b(p_0)[\, \to V$, where $]a(p_0), b(p_0)[$ is the maximal open interval containing $0$ on which $y$ is defined.

</div>

The uniqueness of the solution means that if $z \colon \,]\delta, \varepsilon[\, \to V$ satisfies the same differential equation $dz/dt = f(z)$, $z(0) = p_0$, then the domain of definition $]\delta, \varepsilon[$ of $z$ is a subset of $]a(p_0), b(p_0)[$ and $z(t) = y(t)$ on the interval $]\delta, \varepsilon[$.

For a vector field $X$ on a chart $U$ of a manifold and a point $p \in U$, this theorem guarantees the existence and uniqueness of a maximal integral curve starting at $p$.

Next we would like to study the dependence of an integral curve on its initial point. Again we study the problem locally on $\mathbb{R}^n$. The function $y$ will now be a function of two arguments $t$ and $q$, and the condition for $y$ to be an integral curve starting at the point $q$ is

$$\frac{\partial y}{\partial t}(t, q) = f(y(t, q)), \quad y(0, q) = q.$$

The following theorem from the theory of ODE guarantees the smooth dependence of the solution on the initial point.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.8</span><span class="math-callout__name">(Smooth dependence on initial conditions)</span></p>

Let $V$ be an open subset of $\mathbb{R}^n$ and $f \colon V \to \mathbb{R}^n$ a $C^\infty$ function on $V$. For each point $p_0 \in V$, there are a neighborhood $W$ of $p_0$ in $V$, a number $\varepsilon > 0$, and a $C^\infty$ function

$$y \colon \,]-\varepsilon, \varepsilon[\, \times W \to V$$

such that

$$\frac{\partial y}{\partial t}(t, q) = f(y(t, q)), \quad y(0, q) = q$$

for all $(t, q) \in \,]-\varepsilon, \varepsilon[\, \times W$.

</div>

It follows from Theorem 14.8 and (14.8) that if $X$ is any $C^\infty$ vector field on a chart $U$ and $p \in U$, then there are a neighborhood $W$ of $p$ in $U$, an $\varepsilon > 0$, and a $C^\infty$ map

$$F \colon \,]-\varepsilon, \varepsilon[\, \times W \to U$$

such that for each $q \in W$, the function $F(t, q)$ is an integral curve of $X$ starting at $q$. In particular, $F(0, q) = q$. We usually write $F_t(q)$ for $F(t, q)$.

Suppose $s, t$ in the interval $]-\varepsilon, \varepsilon[$ are such that both $F_t(F_s(q))$ and $F_{t+s}(q)$ are defined. Then both $F_t(F_s(q))$ and $F_{t+s}(q)$ as functions of $t$ are integral curves of $X$ with initial point $F_s(q)$, which is the point corresponding to $t = 0$. By the uniqueness of the integral curve starting at a point,

$$F_t(F_s(q)) = F_{t+s}(q).$$

The map $F$ is called a **local flow generated by the vector field** $X$. For each $q \in U$, the function $F_t(q)$ of $t$ is called a **flow line** of the local flow. Each flow line is an integral curve of $X$. If a local flow $F$ is defined on $\mathbb{R} \times M$, then it is called a **global flow**. Every smooth vector field has a local flow about any point, but not necessarily a global flow. A vector field having a global flow is called a **complete vector field**. If $F$ is a global flow, then for every $t \in \mathbb{R}$,

$$F_t \circ F_{-t} = F_{-t} \circ F_t = F_0 = \mathbb{1}_M,$$

so $F_t \colon M \to M$ is a diffeomorphism. Thus, a global flow on $M$ gives rise to a one-parameter group of diffeomorphisms of $M$.

This discussion suggests the following definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.9</span><span class="math-callout__name">(Local flow)</span></p>

A **local flow** about a point $p$ in an open set $U$ of a manifold is a $C^\infty$ function

$$F \colon \,]-\varepsilon, \varepsilon[\, \times W \to U,$$

where $\varepsilon$ is a positive real number and $W$ is a neighborhood of $p$ in $U$, such that writing $F_t(q) = F(t, q)$, we have

**(i)** $F_0(q) = q$ for all $q \in W$,

**(ii)** $F_t(F_s(q)) = F_{t+s}(q)$ whenever both sides are defined.

</div>

If $F(t, q)$ is a local flow of the vector field $X$ on $U$, then

$$F(0, q) = q \quad \text{and} \quad \frac{\partial F}{\partial t}(0, q) = X_{F(0,q)} = X_q.$$

Thus, one can recover the vector field from its flow.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Recovering the vector field from its flow)</span></p>

The function $F \colon \mathbb{R} \times \mathbb{R}^2 \to \mathbb{R}^2$,

$$F\!\left(t, \begin{bmatrix} x \\ y \end{bmatrix}\right) = \begin{bmatrix} \cos t & -\sin t \\ \sin t & \cos t \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix},$$

is the global flow on $\mathbb{R}^2$ generated by the vector field

$$X_{(x,y)} = \frac{\partial F}{\partial t}\bigl(t, (x, y)\bigr)\bigg\vert_{t=0} = \begin{bmatrix} -\sin t & -\cos t \\ \cos t & -\sin t \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}\bigg\vert_{t=0} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} -y \\ x \end{bmatrix} = -y\frac{\partial}{\partial x} + x\frac{\partial}{\partial y}.$$

This is Example 12.8 again.

</div>

### 14.4 The Lie Bracket

Suppose $X$ and $Y$ are smooth vector fields on an open subset $U$ of a manifold $M$. We view $X$ and $Y$ as derivations on $C^\infty(U)$. For a $C^\infty$ function $f$ on $U$, by Proposition 14.3 the function $Yf$ is $C^\infty$ on $U$, and the function $(XY)f := X(Yf)$ is also $C^\infty$ on $U$. Moreover, because $X$ and $Y$ are both $\mathbb{R}$-linear maps from $C^\infty(U)$ to $C^\infty(U)$, the map $XY \colon C^\infty(U) \to C^\infty(U)$ is $\mathbb{R}$-linear. However, $XY$ does not satisfy the derivation property: if $f, g \in C^\infty(U)$, then

$$XY(fg) = X\bigl((Yf)g + fYg\bigr) = (XYf)g + (Yf)(Xg) + (Xf)(Yg) + f(XYg).$$

Looking more closely at this formula, we see that the two extra terms $(Yf)(Xg)$ and $(Xf)(Yg)$ that make $XY$ not a derivation are symmetric in $X$ and $Y$. Thus, if we compute $YX(fg)$ as well and subtract it from $XY(fg)$, the extra terms will disappear, and $XY - YX$ will be a derivation of $C^\infty(U)$.

Given two smooth vector fields $X$ and $Y$ on $U$ and $p \in U$, we define their **Lie bracket** $[X, Y]$ at $p$ to be

$$[X, Y]_p f = (X_p Y - Y_p X) f$$

for any germ $f$ of a $C^\infty$ function at $p$. By the same calculation as above, but now evaluated at $p$, it is easy to check that $[X, Y]_p$ is a derivation of $C_p^\infty(U)$ and is therefore a tangent vector at $p$ (Definition 8.1). As $p$ varies over $U$, $[X, Y]$ becomes a vector field on $U$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.10</span></p>

If $X$ and $Y$ are smooth vector fields on $M$, then the vector field $[X, Y]$ is also smooth on $M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By Proposition 14.3 it suffices to check that if $f$ is a $C^\infty$ function on $M$, then so is $[X, Y]f$. But

$$[X, Y]f = (XY - YX)f,$$

which is clearly $C^\infty$, since both $X$ and $Y$ are. $\square$

</details>
</div>

From this proposition, we see that the Lie bracket provides a product operation on the vector space $\mathfrak{X}(M)$ of all smooth vector fields on $M$. Clearly,

$$[Y, X] = -[X, Y].$$

**Exercise 14.11 (Jacobi identity).** Check the *Jacobi identity*:

$$\sum_{\text{cyclic}} [X, [Y, Z]] = 0.$$

This notation means that one permutes $X, Y, Z$ cyclically and one takes the sum of the resulting terms. Written out,

$$\sum_{\text{cyclic}} [X, [Y, Z]] = [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]].$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.12</span><span class="math-callout__name">(Lie algebra)</span></p>

Let $K$ be a field. A **Lie algebra** over $K$ is a vector space $V$ over $K$ together with a product $[\,,\,] \colon V \times V \to V$, called the **bracket**, satisfying the following properties: for all $a, b \in K$ and $X, Y, Z \in V$,

**(i)** (bilinearity) $[aX + bY, Z] = a[X, Z] + b[Y, Z]$, $\quad [Z, aX + bY] = a[Z, X] + b[Z, Y]$,

**(ii)** (anticommutativity) $[Y, X] = -[X, Y]$,

**(iii)** (Jacobi identity) $\sum_{\text{cyclic}} [X, [Y, Z]] = 0$.

</div>

In practice, we will be concerned only with *real Lie algebras*, i.e., Lie algebras over $\mathbb{R}$. Unless otherwise specified, a Lie algebra in this book means a real Lie algebra.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Abelian Lie algebra)</span></p>

On any vector space $V$, define $[X, Y] = 0$ for all $X, Y \in V$. With this bracket, $V$ becomes a Lie algebra, called an **abelian Lie algebra**.

</div>

Our definition of an algebra in Subsection 2.2 requires that the product be associative. An abelian Lie algebra is trivially associative, but in general the bracket of a Lie algebra need not be associative. So despite its name, a Lie algebra is in general not an algebra.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Lie algebra of vector fields)</span></p>

If $M$ is a manifold, then the vector space $\mathfrak{X}(M)$ of $C^\infty$ vector fields on $M$ is a real Lie algebra with the Lie bracket $[\,,\,]$ as the bracket.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Lie algebra of matrices)</span></p>

Let $K^{n \times n}$ be the vector space of all $n \times n$ matrices over a field $K$. Define for $X, Y \in K^{n \times n}$,

$$[X, Y] = XY - YX,$$

where $XY$ is the matrix product of $X$ and $Y$. With this bracket, $K^{n \times n}$ becomes a Lie algebra. The bilinearity and anticommutativity of $[\,,\,]$ are immediate, while the Jacobi identity follows from the same computation as in Exercise 14.11.

</div>

More generally, if $A$ is any algebra over a field $K$, then the product

$$[x, y] = xy - yx, \quad x, y \in A,$$

makes $A$ into a Lie algebra over $K$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.13</span><span class="math-callout__name">(Derivation of a Lie algebra)</span></p>

A **derivation** of a Lie algebra $V$ over a field $K$ is a $K$-linear map $D \colon V \to V$ satisfying the product rule

$$D[Y, Z] = [DY, Z] + [Y, DZ] \quad \text{for } Y, Z \in V.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The adjoint is a derivation)</span></p>

Let $V$ be a Lie algebra over a field $K$. For each $X$ in $V$, define $\operatorname{ad}_X \colon V \to V$ by

$$\operatorname{ad}_X(Y) = [X, Y].$$

We may rewrite the Jacobi identity in the form

$$[X, [Y, Z]] = [[X, Y], Z] + [Y, [X, Z]]$$

or

$$\operatorname{ad}_X [Y, Z] = [\operatorname{ad}_X Y, Z] + [Y, \operatorname{ad}_X Z],$$

which shows that $\operatorname{ad}_X \colon V \to V$ is a derivation of $V$.

</div>

### 14.5 The Pushforward of Vector Fields

Let $F \colon N \to M$ be a smooth map of manifolds and let $F_{*} \colon T_pN \to T_{F(p)}M$ be its differential at a point $p$ in $N$. If $X_p \in T_pN$, we call $F_*(X_p)$ the **pushforward** of the vector $X_p$ at $p$. This notion does not extend in general to vector fields, since if $X$ is a vector field on $N$ and $z = F(p) = F(q)$ for two distinct points $p, q \in N$, then $X_p$ and $X_q$ are both pushed forward to tangent vectors at $z \in M$, but there is no reason why $F_*(X_p)$ and $F_*(X_q)$ should be equal.

In one important special case, the pushforward $F_*X$ of any vector field $X$ on $N$ always makes sense, namely, when $F \colon N \to M$ is a diffeomorphism. In this case, since $F$ is injective, there is no ambiguity about the meaning of $(F_*X)_{F(p)} = F_{*,p}(X_p)$, and since $F$ is surjective, $F_*X$ is defined everywhere on $M$.

### 14.6 Related Vector Fields

Under a $C^\infty$ map $F \colon N \to M$, although in general a vector field on $N$ cannot be pushed forward to a vector field on $M$, there is nonetheless a useful notion of *related vector fields*, which we now define.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.14</span><span class="math-callout__name">($F$-related vector fields)</span></p>

Let $F \colon N \to M$ be a smooth map of manifolds. A vector field $X$ on $N$ is **$F$-related** to a vector field $\tilde{X}$ on $M$ if for all $p \in N$,

$$F_{*,p}(X_p) = \tilde{X}_{F(p)}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 14.15</span><span class="math-callout__name">(Pushforward by a diffeomorphism)</span></p>

If $F \colon N \to M$ is a diffeomorphism and $X$ is a vector field on $N$, then the pushforward $F_*X$ is defined. By definition, the vector field $X$ on $N$ is $F$-related to the vector field $F_*X$ on $M$. In Subsection 16.5, we will see examples of vector fields related by a map $F$ that is not a diffeomorphism.

</div>

We may reformulate the condition for $F$-relatedness as follows.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.16</span></p>

Let $F \colon N \to M$ be a smooth map of manifolds. A vector field $X$ on $N$ and a vector field $\tilde{X}$ on $M$ are $F$-related if and only if for all $g \in C^\infty(M)$,

$$X(g \circ F) = (\tilde{X}g) \circ F.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$(\Rightarrow)$ Suppose $X$ and $\tilde{X}$ on $M$ are $F$-related. By (14.11), for any $g \in C^\infty(M)$ and $p \in N$,

$$F_{*,p}(X_p)g = \tilde{X}_{F(p)}g \quad \text{(definition of $F$-relatedness)},$$

$$X_p(g \circ F) = (\tilde{X}g)(F(p)) \quad \text{(definitions of $F_*$ and $\tilde{X}g$)},$$

$$(X(g \circ F))(p) = (\tilde{X}g)(F(p)).$$

Since this is true for all $p \in N$,

$$X(g \circ F) = (\tilde{X}g) \circ F.$$

$(\Leftarrow)$ Reversing the set of equations above proves the converse. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.17</span></p>

Let $F \colon N \to M$ be a smooth map of manifolds. If the $C^\infty$ vector fields $X$ and $Y$ on $N$ are $F$-related to the $C^\infty$ vector fields $\tilde{X}$ and $\tilde{Y}$, respectively, on $M$, then the Lie bracket $[X, Y]$ on $N$ is $F$-related to the Lie bracket $[\tilde{X}, \tilde{Y}]$ on $M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For any $g \in C^\infty(M)$,

$$[X, Y](g \circ F) = XY(g \circ F) - YX(g \circ F) \quad \text{(definition of $[X, Y]$)}$$

$$= X\bigl((\tilde{Y}g) \circ F\bigr) - Y\bigl((\tilde{X}g) \circ F\bigr) \quad \text{(Proposition 14.16)}$$

$$= (\tilde{X}\tilde{Y}g) \circ F - (\tilde{Y}\tilde{X}g) \circ F \quad \text{(Proposition 14.16)}$$

$$= ((\tilde{X}\tilde{Y} - \tilde{Y}\tilde{X})g) \circ F$$

$$= ([\tilde{X}, \tilde{Y}]g) \circ F.$$

By Proposition 14.16 again, this proves that $[X, Y]$ on $N$ and $[\tilde{X}, \tilde{Y}]$ on $M$ are $F$-related. $\square$

</details>
</div>

## §15 Lie Groups

We begin with several examples of matrix groups, subgroups of the general linear group over a field. The goal is to exhibit a variety of methods for showing that a group is a Lie group and for computing the dimension of a Lie group. These examples become templates for investigating other matrix groups. A powerful tool, which we state but do not prove, is the closed subgroup theorem. According to this theorem, an abstract subgroup that is a closed subset of a Lie group is itself a Lie group. In many instances, the closed subgroup theorem is the easiest way to prove that a group is a Lie group.

The matrix exponential gives rise to curves in a matrix group with a given initial vector. It is useful in computing the differential of a map on a matrix group. As an example, we compute the differential of the determinant map on the general linear group over $\mathbb{R}$.

### 15.1 Examples of Lie Groups

We recall here the definition of a Lie group, which first appeared in Subsection 6.5.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 15.1</span><span class="math-callout__name">(Lie group)</span></p>

A **Lie group** is a $C^\infty$ manifold $G$ that is also a group such that the two group operations, multiplication

$$\mu \colon G \times G \to G, \quad \mu(a, b) = ab,$$

and inverse

$$\iota \colon G \to G, \quad \iota(a) = a^{-1},$$

are $C^\infty$.

</div>

For $a \in G$, denote by $\ell_a \colon G \to G$, $\ell_a(x) = \mu(a, x) = ax$, the operation of **left multiplication** by $a$, and by $r_a \colon G \to G$, $r_a(x) = xa$, the operation of **right multiplication** by $a$. We also call left and right multiplications *left* and *right translations*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 15.3</span><span class="math-callout__name">(Lie group homomorphism)</span></p>

A map $F \colon H \to G$ between two Lie groups $H$ and $G$ is a **Lie group homomorphism** if it is a $C^\infty$ map and a group homomorphism.

</div>

The group homomorphism condition means that for all $h, x \in H$,

$$F(hx) = F(h)F(x).$$

This may be rewritten in functional notation as

$$F \circ \ell_h = \ell_{F(h)} \circ F \quad \text{for all } h \in H.$$

Let $e_H$ and $e_G$ be the identity elements of $H$ and $G$, respectively. Taking $h$ and $x$ in the equation above to be $e_H$, it follows that $F(e_H) = e_G$. So a group homomorphism always maps the identity to the identity.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 15.4</span><span class="math-callout__name">(General linear group)</span></p>

In Example 6.21, we showed that the general linear group

$$\operatorname{GL}(n, \mathbb{R}) = \lbrace A \in \mathbb{R}^{n \times n} \mid \det A \neq 0 \rbrace$$

is a Lie group.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 15.5</span><span class="math-callout__name">(Special linear group)</span></p>

The special linear group $\operatorname{SL}(n, \mathbb{R})$ is the subgroup of $\operatorname{GL}(n, \mathbb{R})$ consisting of matrices of determinant $1$. By Example 9.13, $\operatorname{SL}(n, \mathbb{R})$ is a regular submanifold of dimension $n^2 - 1$ of $\operatorname{GL}(n, \mathbb{R})$. By Example 11.16, the multiplication map

$$\bar{\mu} \colon \operatorname{SL}(n, \mathbb{R}) \times \operatorname{SL}(n, \mathbb{R}) \to \operatorname{SL}(n, \mathbb{R})$$

is $C^\infty$. To see that the inverse map

$$\iota \colon \operatorname{SL}(n, \mathbb{R}) \to \operatorname{SL}(n, \mathbb{R})$$

is $C^\infty$, let $i \colon \operatorname{SL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R})$ be the inclusion map and $\iota \colon \operatorname{GL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R})$ the inverse map of $\operatorname{GL}(n, \mathbb{R})$. As the composite of two $C^\infty$ maps,

$$\iota \circ i \colon \operatorname{SL}(n, \mathbb{R}) \xrightarrow{i} \operatorname{GL}(n, \mathbb{R}) \xrightarrow{\iota} \operatorname{GL}(n, \mathbb{R})$$

is a $C^\infty$ map. Since its image is contained in the regular submanifold $\operatorname{SL}(n, \mathbb{R})$, the induced map $\iota \colon \operatorname{SL}(n, \mathbb{R}) \to \operatorname{SL}(n, \mathbb{R})$ is $C^\infty$ by Theorem 11.15. Thus, $\operatorname{SL}(n, \mathbb{R})$ is a Lie group.

An entirely analogous argument proves that the complex special linear group $\operatorname{SL}(n, \mathbb{C})$ is also a Lie group.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 15.6</span><span class="math-callout__name">(Orthogonal group)</span></p>

Recall that the orthogonal group $\operatorname{O}(n)$ is the subgroup of $\operatorname{GL}(n, \mathbb{R})$ consisting of all matrices $A$ satisfying $A^T A = I$. Thus, $\operatorname{O}(n)$ is the inverse image of $I$ under the map $f(A) = A^T A$.

In Example 11.3 we showed that $f \colon \operatorname{GL}(n, \mathbb{R}) \to \operatorname{GL}(n, \mathbb{R})$ has constant rank. By the constant-rank level set theorem, $\operatorname{O}(n)$ is a regular submanifold of $\operatorname{GL}(n, \mathbb{R})$.

One drawback of this approach is that it does not tell us what the rank of $f$ is, and so the dimension of $\operatorname{O}(n)$ remains unknown. We will instead apply the regular level set theorem to prove that $\operatorname{O}(n)$ is a regular submanifold of $\operatorname{GL}(n, \mathbb{R})$, which will at the same time determine the dimension of $\operatorname{O}(n)$.

Since $A^T A$ is a symmetric matrix, the image of $f$ lies in $S_n$, the vector space of all $n \times n$ real symmetric matrices. The space $S_n$ is a proper subspace of $\mathbb{R}^{n \times n}$ as soon as $n \ge 2$.

**Exercise 15.7 (Space of symmetric matrices).** The vector space $S_n$ of $n \times n$ real symmetric matrices has dimension $(n^2 + n)/2$.

Consider the map $f \colon \operatorname{GL}(n, \mathbb{R}) \to S_n$, $f(A) = A^T A$. The tangent space of $S_n$ at any point is canonically isomorphic to $S_n$ itself, because $S_n$ is a vector space. Thus, the image of the differential

$$f_{*,A} \colon T_A(\operatorname{GL}(n, \mathbb{R})) \to T_{f(A)}(S_n) \simeq S_n$$

lies in $S_n$. For any matrix $X \in \mathbb{R}^{n \times n}$, there is a curve $c(t)$ in $\operatorname{GL}(n, \mathbb{R})$ with $c(0) = A$ and $c'(0) = X$ (Proposition 8.16). By Proposition 8.18,

$$f_{*,A}(X) = \frac{d}{dt} f(c(t))\bigg\vert_{t=0} = \frac{d}{dt} c(t)^T c(t)\bigg\vert_{t=0} = (c'(t)^T c(t) + c(t)^T c'(t))\vert_{t=0} = X^T A + A^T X.$$

The surjectivity of $f_{*,A}$ becomes the following question: if $A \in \operatorname{O}(n)$ and $B$ is any symmetric matrix in $S_n$, does there exist an $n \times n$ matrix $X$ such that

$$X^T A + A^T X = B?$$

Note that since $(X^T A)^T = A^T X$, it is enough to solve

$$A^T X = \frac{1}{2} B,$$

for then $X^T A + A^T X = \frac{1}{2} B^T + \frac{1}{2} B = B$. Equation (15.3) clearly has a solution: $X = \frac{1}{2}(A^T)^{-1} B$. So $f_{*,A} \colon T_A \operatorname{GL}(n, \mathbb{R}) \to S_n$ is surjective for all $A \in \operatorname{O}(n)$, and $\operatorname{O}(n)$ is a regular level set of $f$. By the regular level set theorem, $\operatorname{O}(n)$ is a regular submanifold of $\operatorname{GL}(n, \mathbb{R})$ of dimension

$$\dim \operatorname{O}(n) = n^2 - \dim S_n = n^2 - \frac{n^2 + n}{2} = \frac{n^2 - n}{2}.$$

</div>

### 15.2 Lie Subgroups

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 15.8</span><span class="math-callout__name">(Lie subgroup)</span></p>

A **Lie subgroup** of a Lie group $G$ is (i) an abstract subgroup $H$ that is (ii) an *immersed* submanifold via the inclusion map such that (iii) the group operations on $H$ are $C^\infty$.

</div>

An "abstract subgroup" simply means a subgroup in the algebraic sense, in contrast to a "Lie subgroup." The group operations on the subgroup $H$ are the restrictions of the multiplication map $\mu$ and the inverse map $\iota$ from $G$ to $H$. For an explanation of why a Lie subgroup is defined to be an immersed submanifold instead of a regular submanifold, see Remark 16.15. Because a Lie subgroup is an immersed submanifold, it need not have the relative topology. However, being an immersion, the inclusion map $i \colon H \hookrightarrow G$ of a Lie subgroup $H$ is of course $C^\infty$. It follows that the composite

$$\mu \circ (i \times i) \colon H \times H \to G \times G \to G$$

is $C^\infty$. If $H$ were defined to be a regular submanifold of $G$, then by Theorem 11.15, because $H$ is a regular submanifold of $G$, the multiplication map $H \times H \to H$ and similarly the inverse map $H \to H$ would automatically be $C^\infty$, and condition (iii) in the definition of a Lie subgroup would be redundant. Since a Lie subgroup is defined to be an immersed submanifold, it is necessary to impose condition (iii) on the group operations on $H$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 15.9</span><span class="math-callout__name">(Lines with irrational slope in a torus)</span></p>

Let $G$ be the torus $\mathbb{R}^2/\mathbb{Z}^2$ and $L$ a line through the origin in $\mathbb{R}^2$. The torus can also be represented by the unit square with the opposite edges identified. The image $H$ of $L$ under the projection $\pi \colon \mathbb{R}^2 \to \mathbb{R}^2/\mathbb{Z}^2$ is a closed curve if and only if the line $L$ goes through another lattice point, say $(m, n) \in \mathbb{Z}^2$. This is the case if and only if the slope of $L$ is $n/m$, a rational number or $\infty$; then $H$ is the image of finitely many line segments on the unit square. It is a closed curve diffeomorphic to a circle and is a regular submanifold of $\mathbb{R}^2/\mathbb{Z}^2$ (Figure 15.1).

If the slope of $L$ is irrational, then its image $H$ on the torus will never close up. In this case the restriction to $L$ of the projection map, $f = \pi\vert_L \colon L \to \mathbb{R}^2/\mathbb{Z}^2$, is a one-to-one immersion. We give $H$ the topology and manifold structure induced from $f$. It can be shown that $H$ is a dense subset of the torus. Thus, $H$ is an immersed submanifold but not a regular submanifold of the torus $\mathbb{R}^2/\mathbb{Z}^2$.

Whatever the slope of $L$, its image $H$ in $\mathbb{R}^2/\mathbb{Z}^2$ is an abstract subgroup of the torus, an immersed submanifold, and a Lie group. Therefore, $H$ is a Lie subgroup of the torus.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15.11</span></p>

If $H$ is an abstract subgroup and a regular submanifold of a Lie group $G$, then it is a Lie subgroup of $G$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since a regular submanifold is the image of an embedding (Theorem 11.14), it is also an immersed submanifold.

Let $\mu \colon G \times G \to G$ be the multiplication map on $G$. Since $H$ is an immersed submanifold of $G$, the inclusion map $i \colon H \hookrightarrow G$ is $C^\infty$. Hence, the inclusion map $i \times i \colon H \times H \hookrightarrow G \times G$ is $C^\infty$, and the composition $\mu \circ (i \times i) \colon H \times H \to G$ is $C^\infty$. By Theorem 11.15, because $H$ is a regular submanifold of $G$, the induced map $\bar{\mu} \colon H \times H \to H$ is $C^\infty$.

The smoothness of the inverse map $\iota \colon H \to H$ can be deduced from the smoothness of $\iota \colon G \to G$ just as in Example 15.5. $\square$

</details>
</div>

A subgroup $H$ as in Proposition 15.11 is called an **embedded Lie subgroup**, because the inclusion map $i \colon H \to G$ of a regular submanifold is an embedding (Theorem 11.14).

We showed in Examples 15.5 and 15.6 that the subgroups $\operatorname{SL}(n, \mathbb{R})$ and $\operatorname{O}(n)$ of $\operatorname{GL}(n, \mathbb{R})$ are both regular submanifolds. By Proposition 15.11 they are embedded Lie subgroups.

We state without proof an important theorem about Lie subgroups. If $G$ is a Lie group, then an abstract subgroup that is a closed subset in the topology of $G$ is called a **closed subgroup**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 15.12</span><span class="math-callout__name">(Closed subgroup theorem)</span></p>

A closed subgroup of a Lie group is an embedded Lie subgroup.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples</span></p>

**(i)** A line with irrational slope in the torus $\mathbb{R}^2/\mathbb{Z}^2$ is not a closed subgroup, since it is not the whole torus, but being dense, its closure is.

**(ii)** The special linear group $\operatorname{SL}(n, \mathbb{R})$ and the orthogonal group $\operatorname{O}(n)$ are the zero sets of polynomial equations on $\operatorname{GL}(n, \mathbb{R})$. As such, they are closed subsets of $\operatorname{GL}(n, \mathbb{R})$. By the closed subgroup theorem, $\operatorname{SL}(n, \mathbb{R})$ and $\operatorname{O}(n)$ are embedded Lie subgroups of $\operatorname{GL}(n, \mathbb{R})$.

</div>

### 15.3 The Matrix Exponential

To compute the differential of a map on a subgroup of $\operatorname{GL}(n, \mathbb{R})$, we need a curve of nonsingular matrices. Because the matrix exponential is always nonsingular, it is uniquely suited for this purpose.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Norm on a vector space)</span></p>

A **norm** on a vector space $V$ is a real-valued function $\lVert \cdot \rVert \colon V \to \mathbb{R}$ satisfying the following three properties: for all $r \in \mathbb{R}$ and $v, w \in V$,

**(i)** (positive-definiteness) $\lVert v \rVert \ge 0$ with equality if and only if $v = 0$,

**(ii)** (positive homogeneity) $\lVert rv \rVert = \lvert r \rvert \,\lVert v \rVert$,

**(iii)** (subadditivity) $\lVert v + w \rVert \le \lVert v \rVert + \lVert w \rVert$.

</div>

A vector space $V$ together with a norm $\lVert \cdot \rVert$ is called a **normed vector space**. The vector space $\mathbb{R}^{n \times n} \simeq \mathbb{R}^{n^2}$ of all $n \times n$ real matrices can be given the Euclidean norm: for $X = [x_{ij}] \in \mathbb{R}^{n \times n}$,

$$\lVert X \rVert = \left(\sum x_{ij}^2\right)^{1/2}.$$

The **matrix exponential** $e^X$ of a matrix $X \in \mathbb{R}^{n \times n}$ is defined by the same formula as the exponential of a real number:

$$e^X = I + X + \frac{1}{2!} X^2 + \frac{1}{3!} X^3 + \cdots,$$

where $I$ is the $n \times n$ identity matrix. For this formula to make sense, we need to show that the series on the right converges in the normed vector space $\mathbb{R}^{n \times n} \simeq \mathbb{R}^{n^2}$.

A **normed algebra** $V$ is a normed vector space that is also an algebra over $\mathbb{R}$ satisfying the submultiplicative property: for all $v, w \in V$, $\lVert vw \rVert \le \lVert v \rVert \,\lVert w \rVert$. Matrix multiplication makes the normed vector space $\mathbb{R}^{n \times n}$ into a normed algebra.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15.13</span></p>

For $X, Y \in \mathbb{R}^{n \times n}$, $\lVert XY \rVert \le \lVert X \rVert \,\lVert Y \rVert$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Write $X = [x_{ij}]$ and $Y = [y_{ij}]$ and fix a pair of subscripts $(i, j)$. By the Cauchy–Schwarz inequality,

$$(XY)_{ij}^2 = \left(\sum_k x_{ik} y_{kj}\right)^2 \le \left(\sum_k x_{ik}^2\right)\left(\sum_k y_{kj}^2\right) = a_i b_j,$$

where we set $a_i = \sum_k x_{ik}^2$ and $b_j = \sum_k y_{kj}^2$. Then

$$\lVert XY \rVert^2 = \sum_{i,j} (XY)_{ij}^2 \le \sum_{i,j} a_i b_j = \left(\sum_i a_i\right)\left(\sum_j b_j\right) = \left(\sum_{i,k} x_{ik}^2\right)\left(\sum_{j,k} y_{kj}^2\right) = \lVert X \rVert^2 \,\lVert Y \rVert^2. \quad \square$$

</details>
</div>

In a normed algebra, multiplication distributes over a finite sum. When the sum is infinite as in a convergent series, the distributivity of multiplication over the sum requires a proof.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15.14</span></p>

Let $V$ be a normed algebra.

**(i)** If $a \in V$ and $s_m$ is a sequence in $V$ that converges to $s$, then $as_m$ converges to $as$.

**(ii)** If $a \in V$ and $\sum_{k=0}^\infty b_k$ is a convergent series in $V$, then $a \sum_k b_k = \sum_k ab_k$.

</div>

In a normed vector space $V$ a series $\sum a_k$ is said to **converge absolutely** if the series $\sum \lVert a_k \rVert$ of norms converges in $\mathbb{R}$. The normed vector space $V$ is said to be **complete** if every Cauchy sequence in $V$ converges to a point in $V$. For example, $\mathbb{R}^{n \times n}$ is a complete normed vector space. It is easy to show that in a complete normed vector space, absolute convergence implies convergence. Thus, to show that a series $\sum Y_k$ of matrices converges, it is enough to show that the series $\sum \lVert Y_k \rVert$ of real numbers converges.

For any $X \in \mathbb{R}^{n \times n}$ and $k > 0$, repeated applications of Proposition 15.13 give $\lVert X^k \rVert \le \lVert X \rVert^k$. So the series $\sum_{k=0}^\infty \lVert X^k / k! \rVert$ is bounded term by term by the convergent series

$$\sqrt{n} + \lVert X \rVert + \frac{1}{2!} \lVert X \rVert^2 + \frac{1}{3!} \lVert X \rVert^3 + \cdots = (\sqrt{n} - 1) + e^{\lVert X \rVert}.$$

By the comparison test for series of real numbers, the series $\sum_{k=0}^\infty \lVert X^k / k! \rVert$ converges. Therefore, the series $e^X = \sum_{k=0}^\infty X^k / k!$ converges absolutely for any $n \times n$ matrix $X$.

Unlike the exponential of real numbers, when $A$ and $B$ are $n \times n$ matrices with $n > 1$, it is not necessarily true that $e^{A+B} = e^A e^B$.

**Exercise 15.16 (Exponentials of commuting matrices).** Prove that if $A$ and $B$ are commuting $n \times n$ matrices, then $e^A e^B = e^{A+B}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15.17</span></p>

For $X \in \mathbb{R}^{n \times n}$,

$$\frac{d}{dt} e^{tX} = X e^{tX} = e^{tX} X.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Because each $(i, j)$-entry of the series for the exponential function $e^{tX}$ is a power series in $t$, it is possible to differentiate term by term. Hence,

$$\frac{d}{dt} e^{tX} = \frac{d}{dt}\left(I + tX + \frac{1}{2!} t^2 X^2 + \frac{1}{3!} t^3 X^3 + \cdots\right) = X + tX^2 + \frac{1}{2!} t^2 X^3 + \cdots$$

$$= X\left(I + tX + \frac{1}{2!} t^2 X^2 + \cdots\right) = X e^{tX} \quad \text{(Proposition 15.14(ii))}.$$

In the second equality above, one could have factored out $X$ as the second factor:

$$\frac{d}{dt} e^{tX} = X + tX^2 + \frac{1}{2!} t^2 X^3 + \cdots = \left(I + tX + \frac{1}{2!} t^2 X^2 + \cdots\right)X = e^{tX} X. \quad \square$$

</details>
</div>

The definition of the matrix exponential $e^X$ makes sense even if $X$ is a complex matrix. All the arguments so far carry over word for word; one merely has to replace the Euclidean norm $\lVert X \rVert^2 = \sum x_{ij}^2$ by the Hermitian norm $\lVert X \rVert^2 = \sum \lvert x_{ij} \rvert^2$, where $\lvert x_{ij} \rvert$ is the modulus of a complex number $x_{ij}$.

### 15.4 The Trace of a Matrix

Define the **trace** of an $n \times n$ matrix $X$ to be the sum of its diagonal entries:

$$\operatorname{tr}(X) = \sum_{i=1}^n x_{ii}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 15.18</span></p>

**(i)** For any two matrices $X, Y \in \mathbb{R}^{n \times n}$, $\operatorname{tr}(XY) = \operatorname{tr}(YX)$.

**(ii)** For $X \in \mathbb{R}^{n \times n}$ and $A \in \operatorname{GL}(n, \mathbb{R})$, $\operatorname{tr}(AXA^{-1}) = \operatorname{tr}(X)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**(i)**

$$\operatorname{tr}(XY) = \sum_i (XY)_{ii} = \sum_i \sum_k x_{ik} y_{ki},$$

$$\operatorname{tr}(YX) = \sum_k (YX)_{kk} = \sum_k \sum_i y_{ki} x_{ik}.$$

**(ii)** Set $B = XA^{-1}$ in (i). $\square$

</details>
</div>

The eigenvalues of an $n \times n$ matrix $X$ are the roots of the polynomial equation $\det(\lambda I - X) = 0$. Over the field of complex numbers, which is algebraically closed, such an equation necessarily has $n$ roots, counted with multiplicity. Thus, the advantage of allowing complex numbers is that every $n \times n$ matrix, real or complex, has $n$ complex eigenvalues, counted with multiplicity, whereas a real matrix need not have any real eigenvalue.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

The real matrix

$$\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

has no real eigenvalues. It has two complex eigenvalues, $\pm i$.

</div>

Two facts about eigenvalues are immediate from the definitions:

**(i)** Two similar matrices $X$ and $AXA^{-1}$ have the same eigenvalues, because

$$\det(\lambda I - AXA^{-1}) = \det\bigl(A(\lambda I - X)A^{-1}\bigr) = \det(\lambda I - X).$$

**(ii)** The eigenvalues of a triangular matrix are its diagonal entries, because

$$\det\left(\lambda I - \begin{bmatrix} \lambda_1 & * \\ & \ddots & \\ 0 & & \lambda_n \end{bmatrix}\right) = \prod_{i=1}^n (\lambda - \lambda_i).$$

By a theorem from algebra, any complex square matrix $X$ can be triangularized; more precisely, there exists a nonsingular complex square matrix $A$ such that $AXA^{-1}$ is upper triangular. Since the eigenvalues $\lambda_1, \dots, \lambda_n$ of $X$ are the same as the eigenvalues of $AXA^{-1}$, the triangular matrix $AXA^{-1}$ must have the eigenvalues of $X$ along its diagonal.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15.19</span></p>

The trace of a matrix, real or complex, is equal to the sum of its complex eigenvalues.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Suppose $X$ has complex eigenvalues $\lambda_1, \dots, \lambda_n$. Then there exists a nonsingular matrix $A \in \operatorname{GL}(n, \mathbb{C})$ such that

$$AXA^{-1} = \begin{bmatrix} \lambda_1 & * \\ & \ddots & \\ 0 & & \lambda_n \end{bmatrix}.$$

By Lemma 15.18, $\operatorname{tr}(X) = \operatorname{tr}(AXA^{-1}) = \sum \lambda_i$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15.20</span></p>

For any $X \in \mathbb{R}^{n \times n}$, $\det(e^X) = e^{\operatorname{tr} X}$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**Case 1.** Assume that $X$ is upper triangular:

$$X = \begin{bmatrix} \lambda_1 & * \\ & \ddots & \\ 0 & & \lambda_n \end{bmatrix}.$$

Then

$$e^X = \sum \frac{1}{k!} X^k = \sum \frac{1}{k!} \begin{bmatrix} \lambda_1^k & * \\ & \ddots & \\ 0 & & \lambda_n^k \end{bmatrix} = \begin{bmatrix} e^{\lambda_1} & * \\ & \ddots & \\ 0 & & e^{\lambda_n} \end{bmatrix}.$$

Hence, $\det e^X = \prod e^{\lambda_i} = e^{\sum \lambda_i} = e^{\operatorname{tr} X}$.

**Case 2.** Given a general matrix $X$, with eigenvalues $\lambda_1, \dots, \lambda_n$, we can find a nonsingular complex matrix $A$ such that

$$AXA^{-1} = \begin{bmatrix} \lambda_1 & * \\ & \ddots & \\ 0 & & \lambda_n \end{bmatrix},$$

an upper triangular matrix. Then

$$e^{AXA^{-1}} = I + AXA^{-1} + \frac{1}{2!}(AXA^{-1})^2 + \frac{1}{3!}(AXA^{-1})^3 + \cdots$$

$$= I + AXA^{-1} + A\left(\frac{1}{2!}X^2\right)A^{-1} + A\left(\frac{1}{3!}X^3\right)A^{-1} + \cdots = Ae^X A^{-1}$$

(by Proposition 15.14(ii)). Hence,

$$\det e^X = \det(Ae^X A^{-1}) = \det(e^{AXA^{-1}}) = e^{\operatorname{tr}(AXA^{-1})} = e^{\operatorname{tr} X}$$

by Case 1 (since $AXA^{-1}$ is upper triangular) and by Lemma 15.18. $\square$

</details>
</div>

It follows from this proposition that the matrix exponential is always nonsingular, because $\det(e^X) = e^{\operatorname{tr} X}$ is never zero. This is one reason why the matrix exponential is so useful: for it allows us to write down explicitly a curve in $\operatorname{GL}(n, \mathbb{R})$ with a given initial point and a given initial velocity. For example, $c(t) = e^{tX} \colon \mathbb{R} \to \operatorname{GL}(n, \mathbb{R})$ is a curve in $\operatorname{GL}(n, \mathbb{R})$ with initial point $I$ and initial velocity $X$, since

$$c(0) = e^{0X} = e^0 = I \quad \text{and} \quad c'(0) = \frac{d}{dt} e^{tX}\bigg\vert_{t=0} = Xe^{tX}\big\vert_{t=0} = X.$$

Similarly, $c(t) = Ae^{tX} \colon \mathbb{R} \to \operatorname{GL}(n, \mathbb{R})$ is a curve in $\operatorname{GL}(n, \mathbb{R})$ with initial point $A$ and initial velocity $AX$.

### 15.5 The Differential of det at the Identity

Let $\det \colon \operatorname{GL}(n, \mathbb{R}) \to \mathbb{R}$ be the determinant map. The tangent space $T_I \operatorname{GL}(n, \mathbb{R})$ to $\operatorname{GL}(n, \mathbb{R})$ at the identity matrix $I$ is the vector space $\mathbb{R}^{n \times n}$ and the tangent space $T_1 \mathbb{R}$ to $\mathbb{R}$ at $1$ is $\mathbb{R}$. So

$$\det_{*,I} \colon \mathbb{R}^{n \times n} \to \mathbb{R}.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15.21</span></p>

For any $X \in \mathbb{R}^{n \times n}$, $\det_{*,I}(X) = \operatorname{tr} X$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

We use a curve at $I$ to compute the differential (Proposition 8.18). As a curve $c(t)$ with $c(0) = I$ and $c'(0) = X$, choose the matrix exponential $c(t) = e^{tX}$. Then

$$\det_{*,I}(X) = \frac{d}{dt} \det(e^{tX})\bigg\vert_{t=0} = \frac{d}{dt} e^{t \operatorname{tr} X}\bigg\vert_{t=0} = (\operatorname{tr} X)\,e^{t \operatorname{tr} X}\big\vert_{t=0} = \operatorname{tr} X. \quad \square$$

</details>
</div>

# Chapter 16 — Lie Algebras

In a Lie group $G$, because left translation by an element $g \in G$ is a diffeomorphism that maps a neighborhood of the identity to a neighborhood of $g$, all the local information about the group is concentrated in a neighborhood of the identity, and the tangent space at the identity assumes a special importance.

Moreover, one can give the tangent space $T_e G$ a Lie bracket $[\;,\;]$, so that in addition to being a vector space, it becomes a Lie algebra, called the *Lie algebra* of the Lie group. This Lie algebra encodes in it much information about the Lie group. The goal of this section is to define the Lie algebra structure on $T_e G$ and to identify the Lie algebras of a few classical groups.

The Lie bracket on $T_g G$ is defined using a canonical isomorphism between the tangent space at the identity and the vector space of left-invariant vector fields on $G$. With respect to this Lie bracket, the differential of a Lie group homomorphism becomes a Lie algebra homomorphism. We thus obtain a functor from the category of Lie groups and Lie group homomorphisms to the category of Lie algebras and Lie algebra homomorphisms.

## §16.1 Tangent Space at the Identity of a Lie Group

Because of the existence of a multiplication, a Lie group is a very special kind of manifold. In Exercise 15.2, we learned that for any $g \in G$, left translation $\ell_g \colon G \to G$ by $g$ is a diffeomorphism with inverse $\ell_{g^{-1}}$. The diffeomorphism $\ell_g$ takes the identity element $e$ to the element $g$ and induces an isomorphism of tangent spaces

$$\ell_{g*} = (\ell_g)_{*,e} \colon T_e(G) \to T_g(G).$$

Thus, if we can describe the tangent space $T_e(G)$ at the identity, then $\ell_{g*} T_e(G)$ will give a description of the tangent space $T_g(G)$ at any point $g \in G$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 16.1</span><span class="math-callout__name">(The tangent space to $\operatorname{GL}(n, \mathbb{R})$ at $I$)</span></p>

In Example 8.19, we identified the tangent space $\operatorname{GL}(n, \mathbb{R})$ at any point $g \in \operatorname{GL}(n, \mathbb{R})$ as $\mathbb{R}^{n \times n}$, the vector space of all $n \times n$ real matrices. We also identified the isomorphism $\ell_{g*} \colon T_I(\operatorname{GL}(n, \mathbb{R})) \to T_g(\operatorname{GL}(n, \mathbb{R}))$ as left multiplication by $g \colon X \mapsto gX$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 16.2</span><span class="math-callout__name">(The tangent space to $\operatorname{SL}(n, \mathbb{R})$ at $I$)</span></p>

We begin by finding a condition that a tangent vector $X$ in $T_I(\operatorname{SL}(n, \mathbb{R}))$ must satisfy. By Proposition 8.16 there is a curve $c \colon ]-\varepsilon, \varepsilon[ \to \operatorname{SL}(n, \mathbb{R})$ with $c(0) = I$ and $c'(0) = X$. Being in $\operatorname{SL}(n, \mathbb{R})$, this curve satisfies

$$\det c(t) = 1$$

for all $t$ in the domain $]-\varepsilon, \varepsilon[$. We now differentiate both sides with respect to $t$ and evaluate at $t = 0$. On the left-hand side, we have

$$\frac{d}{dt} \det(c(t))\bigg\vert_{t=0} = (\det \circ\, c)_* \left(\frac{d}{dt}\bigg\vert_0\right) = \det_{*,I}\!\left(c_* \frac{d}{dt}\bigg\vert_0\right) = \det_{*,I}(c'(0)) = \det_{*,I}(X) = \operatorname{tr}(X)$$

by Proposition 15.21. Thus,

$$\operatorname{tr}(X) = \frac{d}{dt} 1\bigg\vert_{t=0} = 0.$$

So the tangent space $T_I(\operatorname{SL}(n, \mathbb{R}))$ is contained in the subspace $V$ of $\mathbb{R}^{n \times n}$ defined by

$$V = \lbrace X \in \mathbb{R}^{n \times n} \mid \operatorname{tr} X = 0 \rbrace.$$

Since $\dim V = n^2 - 1 = \dim T_I(\operatorname{SL}(n, \mathbb{R}))$, the two spaces must be equal.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.3</span></p>

The tangent space $T_I(\operatorname{SL}(n, \mathbb{R}))$ at the identity of the special linear group $\operatorname{SL}(n, \mathbb{R})$ is the subspace of $\mathbb{R}^{n \times n}$ consisting of all $n \times n$ matrices of trace $0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 16.4</span><span class="math-callout__name">(The tangent space to $\operatorname{O}(n)$ at $I$)</span></p>

Let $X$ be a tangent vector to the orthogonal group $\operatorname{O}(n)$ at the identity $I$. Choose a curve $c(t)$ in $\operatorname{O}(n)$ defined on a small interval containing $0$ such that $c(0) = I$ and $c'(0) = X$. Since $c(t)$ is in $\operatorname{O}(n)$,

$$c(t)^T c(t) = I.$$

Differentiating both sides with respect to $t$ using the matrix product rule (Problem 15.2) gives

$$c'(t)^T c(t) + c(t)^T c'(t) = 0.$$

Evaluating at $t = 0$ gives

$$X^T + X = 0.$$

Thus, $X$ is a skew-symmetric matrix.

Let $K_n$ be the space of all $n \times n$ real skew-symmetric matrices. For example, for $n = 3$, these are matrices of the form

$$\begin{bmatrix} 0 & a & b \\ -a & 0 & c \\ -b & -c & 0 \end{bmatrix}, \quad \text{where } a, b, c \in \mathbb{R}.$$

The diagonal entries of such a matrix are all $0$ and the entries below the diagonal are determined by those above the diagonal. So

$$\dim K_n = \frac{n^2 - n}{2} = \frac{1}{2}(n^2 - n).$$

We have shown that

$$T_I(\operatorname{O}(n)) \subset K_n. \tag{16.1}$$

By an earlier computation (see (15.4)),

$$\dim T_I(\operatorname{O}(n)) = \dim \operatorname{O}(n) = \frac{n^2 - n}{2}.$$

Since the two vector spaces in (16.1) have the same dimension, equality holds.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.5</span></p>

The tangent space $T_I(\operatorname{O}(n))$ of the orthogonal group $\operatorname{O}(n)$ at the identity is the subspace of $\mathbb{R}^{n \times n}$ consisting of all $n \times n$ skew-symmetric matrices.

</div>

## §16.2 Left-Invariant Vector Fields on a Lie Group

Let $X$ be a vector field on a Lie group $G$. We do not assume $X$ to be $C^\infty$. For any $g \in G$, because left multiplication $\ell_g \colon G \to G$ is a diffeomorphism, the pushforward $\ell_{g*} X$ is a well-defined vector field on $G$. We say that the vector field $X$ is **left-invariant** if

$$\ell_{g*} X = X$$

for every $g \in G$; this means for any $h \in G$,

$$\ell_{g*}(X_h) = X_{gh}.$$

In other words, a vector field $X$ is left-invariant if and only if it is $\ell_g$-related to itself for all $g \in G$.

Clearly, a left-invariant vector field $X$ is completely determined by its value $X_e$ at the identity, since

$$X_g = \ell_{g*}(X_e). \tag{16.2}$$

Conversely, given a tangent vector $A \in T_e(G)$ we can define a vector field $\tilde{A}$ on $G$ by (16.2): $(\tilde{A})_g = \ell_{g*} A$. So defined, the vector field $\tilde{A}$ is left-invariant, since

$$\ell_{g*}(\tilde{A}_h) = \ell_{g*} \ell_{h*} A = (\ell_g \circ \ell_h)_* A = (\ell_{gh})_*(A) = \tilde{A}_{gh}.$$

We call $\tilde{A}$ the **left-invariant vector field on $G$ generated by** $A \in T_e G$. Let $L(G)$ be the vector space of all left-invariant vector fields on $G$. Then there is a one-to-one correspondence

$$T_e(G) \leftrightarrow L(G), \tag{16.3}$$

$$X_e \leftarrow X,$$

$$A \mapsto \tilde{A}.$$

It is easy to show that this correspondence is in fact a vector space isomorphism.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 16.6</span><span class="math-callout__name">(Left-invariant vector fields on $\mathbb{R}$)</span></p>

On the Lie group $\mathbb{R}$, the group operation is addition and the identity element is $0$. So "left multiplication" $\ell_g$ is actually addition:

$$\ell_g(x) = g + x.$$

Let us compute $\ell_{g*}(d/dx\vert_0)$. Since $\ell_{g*}(d/dx\vert_0)$ is a tangent vector at $g$, it is a scalar multiple of $d/dx\vert_g$:

$$\ell_{g*}\!\left(\frac{d}{dx}\bigg\vert_0\right) = a\,\frac{d}{dx}\bigg\vert_g. \tag{16.4}$$

To evaluate $a$, apply both sides of (16.4) to the function $f(x) = x$:

$$a = a\,\frac{d}{dx}\bigg\vert_g f = \ell_{g*}\!\left(\frac{d}{dx}\bigg\vert_0\right) f = \frac{d}{dx}\bigg\vert_0 f \circ \ell_g = \frac{d}{dx}\bigg\vert_0 (g + x) = 1.$$

Thus,

$$\ell_{g*}\!\left(\frac{d}{dx}\bigg\vert_0\right) = \frac{d}{dx}\bigg\vert_g.$$

This shows that $d/dx$ is a left-invariant vector field on $\mathbb{R}$. Therefore, the left-invariant vector fields on $\mathbb{R}$ are constant multiples of $d/dx$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 16.7</span><span class="math-callout__name">(Left-invariant vector fields on $\operatorname{GL}(n, \mathbb{R})$)</span></p>

Since $\operatorname{GL}(n, \mathbb{R})$ is an open subset of $\mathbb{R}^{n \times n}$, at any $g \in \operatorname{GL}(n, \mathbb{R})$ there is a canonical identification of the tangent space $T_g(\operatorname{GL}(n, \mathbb{R}))$ with $\mathbb{R}^{n \times n}$, under which a tangent vector corresponds to an $n \times n$ matrix:

$$\sum a_{ij} \frac{\partial}{\partial x_{ij}}\bigg\vert_g \longleftrightarrow [a_{ij}]. \tag{16.5}$$

We use the same letter $B$ to denote alternately a tangent vector $B = \sum b_{ij}\,\partial/\partial x_{ij}\vert_I \in T_I(\operatorname{GL}(n, \mathbb{R}))$ at the identity and a matrix $B = [b_{ij}]$. Let $B = \sum b_{ij}\,\partial/\partial x_{ij}\vert_I \in T_I(\operatorname{GL}(n, \mathbb{R}))$ and let $\tilde{B}$ be the left-invariant vector field on $\operatorname{GL}(n, \mathbb{R})$ generated by $B$. By Example 8.19,

$$\tilde{B}_g = (\ell_g)_* B \longleftrightarrow gB$$

under the identification (16.5). In terms of the standard basis $\partial/\partial x_{ij}\vert_g$,

$$\tilde{B}_g = \sum_{i,j} (gB)_{ij}\,\frac{\partial}{\partial x_{ij}}\bigg\vert_g = \sum_{i,j}\!\left(\sum_k g_{ik} b_{kj}\right)\frac{\partial}{\partial x_{ij}}\bigg\vert_g.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.8</span></p>

Any left-invariant vector field $X$ on a Lie group $G$ is $C^\infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By Proposition 14.3 it suffices to show that for any $C^\infty$ function $f$ on $G$, the function $Xf$ is also $C^\infty$. Choose a $C^\infty$ curve $c \colon I \to G$ defined on some interval $I$ containing $0$ such that $c(0) = e$ and $c'(0) = X_e$. If $g \in G$, then $gc(t)$ is a curve starting at $g$ with initial vector $X_g$, since $gc(0) = ge = g$ and

$$(gc)'(0) = \ell_{g*} c'(0) = \ell_{g*} X_e = X_g.$$

By Proposition 8.17,

$$(Xf)(g) = X_g f = \frac{d}{dt}\bigg\vert_{t=0} f(gc(t)).$$

Now the function $f(gc(t))$ is a composition of $C^\infty$ functions

$$G \times I \xrightarrow{1 \times c} G \times G \xrightarrow{\mu} G \xrightarrow{f} \mathbb{R},$$

$$(g, t) \longmapsto (g, c(t)) \mapsto gc(t) \mapsto f(gc(t));$$

as such, it is $C^\infty$. Its derivative with respect to $t$,

$$F(g, t) := \frac{d}{dt} f(gc(t)),$$

is therefore also $C^\infty$. Since $(Xf)(g)$ is a composition of $C^\infty$ functions,

$$G \to G \times I \xrightarrow{F} \mathbb{R},$$

$$g \mapsto (g, 0) \mapsto F(g, 0) = \frac{d}{dt}\bigg\vert_{t=0} f(gc(t)),$$

it is a $C^\infty$ function on $G$. This proves that $X$ is a $C^\infty$ vector field on $G$. $\square$

</details>
</div>

It follows from this proposition that the vector space $L(G)$ of left-invariant vector fields on $G$ is a subspace of the vector space $\mathfrak{X}(G)$ of all $C^\infty$ vector fields on $G$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.9</span></p>

If $X$ and $Y$ are left-invariant vector fields on $G$, then so is $[X, Y]$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For any $g$ in $G$, $X$ is $\ell_g$-related to itself, and $Y$ is $\ell_g$-related to itself. By Proposition 14.17, $[X, Y]$ is $\ell_g$-related to itself. $\square$

</details>
</div>

## §16.3 The Lie Algebra of a Lie Group

Recall that a *Lie algebra* is a vector space $\mathfrak{g}$ together with a *bracket*, i.e., an anticommutative bilinear map $[\;,\;] \colon \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g}$ that satisfies the Jacobi identity (Definition 14.12). A *Lie subalgebra* of a Lie algebra $\mathfrak{g}$ is a vector subspace $\mathfrak{h} \subset \mathfrak{g}$ that is closed under the bracket $[\;,\;]$. By Proposition 16.9, the space $L(G)$ of left-invariant vector fields on a Lie group $G$ is closed under the Lie bracket $[\;,\;]$ and is therefore a Lie subalgebra of the Lie algebra $\mathfrak{X}(G)$ of all $C^\infty$ vector fields on $G$.

As we will see in the next few subsections, the linear isomorphism $\varphi \colon T_e G \simeq L(G)$ in (16.3) is mutually beneficial to the two vector spaces, for each space has something that the other one lacks. The vector space $L(G)$ has a natural Lie algebra structure given by the Lie bracket of vector fields, while the tangent space at the identity has a natural notion of pushforward, given by the differential of a Lie group homomorphism. The linear isomorphism $\varphi \colon T_e G \simeq L(G)$ allows us to define a Lie bracket on $T_e G$ and to push forward left-invariant vector fields under a Lie group homomorphism.

We begin with the Lie bracket on $T_e G$. Given $A, B \in T_e G$, we first map them via $\varphi$ to the left-invariant vector fields $\tilde{A}, \tilde{B}$, take the Lie bracket $[\tilde{A}, \tilde{B}] = \tilde{A}\tilde{B} - \tilde{B}\tilde{A}$, and then map it back to $T_e G$ via $\varphi^{-1}$. Thus, the definition of the Lie bracket $[A, B] \in T_e G$ should be

$$[A, B] = [\tilde{A}, \tilde{B}]_e. \tag{16.6}$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.10</span></p>

If $A, B \in T_e G$ and $\tilde{A}, \tilde{B}$ are the left-invariant vector fields they generate, then

$$[\tilde{A}, \tilde{B}] = [A, B]^{\sim}.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Applying $(\;)^{\sim}$ to both sides of (16.6) gives

$$[A, B]^{\sim} = ([\tilde{A}, \tilde{B}]_e)^{\sim} = [\tilde{A}, \tilde{B}],$$

since $(\;)^{\sim}$ and $(\;)_e$ are inverse to each other. $\square$

</details>
</div>

With the Lie bracket $[\;,\;]$, the tangent space $T_e(G)$ becomes a Lie algebra, called the **Lie algebra** of the Lie group $G$. As a Lie algebra, $T_e(G)$ is usually denoted by $\mathfrak{g}$.

## §16.4 The Lie Bracket on $\mathfrak{gl}(n, \mathbb{R})$

For the general linear group $\operatorname{GL}(n, \mathbb{R})$, the tangent space at the identity $I$ can be identified with the vector space $\mathbb{R}^{n \times n}$ of all $n \times n$ real matrices. We identified a tangent vector in $T_I(\operatorname{GL}(n, \mathbb{R}))$ with a matrix $A \in \mathbb{R}^{n \times n}$ via

$$\sum a_{ij}\,\frac{\partial}{\partial x_{ij}}\bigg\vert_I \longleftrightarrow [a_{ij}]. \tag{16.7}$$

The tangent space $T_I \operatorname{GL}(n, \mathbb{R})$ with its Lie algebra structure is denoted by $\mathfrak{gl}(n, \mathbb{R})$. Let $\tilde{A}$ be the left-invariant vector field on $\operatorname{GL}(n, \mathbb{R})$ generated by $A$. Then on the Lie algebra $\mathfrak{gl}(n, \mathbb{R})$ we have the Lie bracket $[A, B] = [\tilde{A}, \tilde{B}]_I$ coming from the Lie bracket of left-invariant vector fields. In the next proposition, we identify the Lie bracket in terms of matrices.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.11</span></p>

Let

$$A = \sum a_{ij}\,\frac{\partial}{\partial x_{ij}}\bigg\vert_I, \qquad B = \sum b_{ij}\,\frac{\partial}{\partial x_{ij}}\bigg\vert_I \in T_I(\operatorname{GL}(n, \mathbb{R})).$$

If

$$[A, B] = [\tilde{A}, \tilde{B}]_I = \sum c_{ij}\,\frac{\partial}{\partial x_{ij}}\bigg\vert_I, \tag{16.8}$$

then

$$c_{ij} = \sum_k a_{ik} b_{kj} - b_{ik} a_{kj}.$$

Thus, if derivations are identified with matrices via (16.7), then

$$[A, B] = AB - BA.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Applying both sides of (16.8) to $x_{ij}$, we get

$$c_{ij} = [\tilde{A}, \tilde{B}]_I x_{ij} = \tilde{A}_I \tilde{B} x_{ij} - \tilde{B}_I \tilde{A} x_{ij}$$

so it is necessary to find a formula for the function $\tilde{B} x_{ij}$.

In Example 16.7 we found that the left-invariant vector field $\tilde{B}$ on $\operatorname{GL}(n, \mathbb{R})$ is given by

$$\tilde{B}_g = \sum_{i,j} (gB)_{ij}\,\frac{\partial}{\partial x_{ij}}\bigg\vert_g \quad \text{at } g \in \operatorname{GL}(n, \mathbb{R}).$$

Hence,

$$\tilde{B}_g x_{ij} = (gB)_{ij} = \sum_k g_{ik} b_{kj} = \sum_k b_{kj} x_{ik}(g).$$

Since this formula holds for all $g \in \operatorname{GL}(n, \mathbb{R})$, the function $\tilde{B} x_{ij}$ is

$$\tilde{B} x_{ij} = \sum_k b_{kj} x_{ik}.$$

It follows that

$$\tilde{A}\tilde{B} x_{ij} = \sum_{p,q} a_{pq}\,\frac{\partial}{\partial x_{pq}}\bigg\vert_I\!\left(\sum_k b_{kj} x_{ik}\right) = \sum_{p,q,k} a_{pq} b_{kj} \delta_{ip} \delta_{qk} = \sum_k a_{ik} b_{kj} = (AB)_{ij}.$$

Interchanging $A$ and $B$ gives

$$\tilde{B}\tilde{A} x_{ij} = \sum_k b_{ik} a_{kj} = (BA)_{ij}.$$

Therefore,

$$c_{ij} = \sum_k a_{ik} b_{kj} - b_{ik} a_{kj} = (AB - BA)_{ij}. \quad \square$$

</details>
</div>

## §16.5 The Pushforward of Left-Invariant Vector Fields

As we noted in Subsection 14.5, if $F \colon N \to M$ is a $C^\infty$ map of manifolds and $X$ is a $C^\infty$ vector field on $N$, the pushforward $F_* X$ is in general not defined except when $F$ is a diffeomorphism. In the case of Lie groups, however, because of the correspondence between left-invariant vector fields and tangent vectors at the identity, it is possible to push forward left-invariant vector fields under a Lie group homomorphism.

Let $F \colon H \to G$ be a Lie group homomorphism. A left-invariant vector field $X$ on $H$ is generated by its value $A = X_e \in T_e H$ at the identity, so that $X = \tilde{A}$. Since a Lie group homomorphism $F \colon H \to G$ maps the identity of $H$ to the identity of $G$, its differential $F_{*,e}$ at the identity is a linear map from $T_e H$ to $T_e G$. The diagrams

$$T_e H \xrightarrow{F_{*,e}} T_e G \qquad\qquad A \longmapsto F_{*,e} A$$

$$\downarrow\simeq \qquad\quad \downarrow\simeq \qquad\qquad\qquad\quad \downarrow \qquad\qquad\quad \downarrow$$

$$L(H) \dashrightarrow L(G), \qquad\qquad \tilde{A} \longmapsto (F_{*,e} A)^{\sim}$$

show clearly the existence of an induced linear map $F_* \colon L(H) \to L(G)$ on left-invariant vector fields as well as a way to define it.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 16.12</span></p>

Let $F \colon H \to G$ be a Lie group homomorphism. Define $F_* \colon L(H) \to L(G)$ by

$$F_*(\tilde{A}) = (F_{*,e} A)^{\sim}$$

for all $A \in T_e H$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.13</span></p>

If $F \colon H \to G$ is a Lie group homomorphism and $X$ is a left-invariant vector field on $H$, then the left-invariant vector field $F_* X$ on $G$ is $F$-related to the left-invariant vector field $X$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For each $h \in H$, we need to verify that

$$F_{*,h}(X_h) = (F_* X)_{F(h)}. \tag{16.9}$$

The left-hand side of (16.9) is

$$F_{*,h}(X_h) = F_{*,h}(\ell_{h*,e} X_e) = (F \circ \ell_h)_{*,e}(X_e),$$

while the right-hand side of (16.9) is

$$(F_* X)_{F(h)} = (F_{*,e} X_e)^{\sim}_{F(h)} = \ell_{F(h)*} F_{*,e}(X_e) = (\ell_{F(h)} \circ F)_{*,e}(X_e).$$

Since $F$ is a Lie group homomorphism, we have $F \circ \ell_h = \ell_{F(h)} \circ F$, so the two sides of (16.9) are equal. $\square$

</details>
</div>

If $F \colon H \to G$ is a Lie group homomorphism and $X$ is a left-invariant vector field on $H$, we will call $F_* X$ the **pushforward of $X$ under $F$**.

## §16.6 The Differential as a Lie Algebra Homomorphism

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.14</span></p>

If $F \colon H \to G$ is a Lie group homomorphism, then its differential at the identity,

$$F_* = F_{*,e} \colon T_e H \to T_e G,$$

is a Lie algebra homomorphism, i.e., a linear map such that for all $A, B \in T_e H$,

$$F_*[A, B] = [F_* A, F_* B].$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By Proposition 16.13, the vector field $F_* \tilde{A}$ on $G$ is $F$-related to the vector field $\tilde{A}$ on $H$. Hence, the bracket $[F_* \tilde{A}, F_* \tilde{B}]$ on $G$ is $F$-related to the bracket $[\tilde{A}, \tilde{B}]$ on $H$ (Proposition 14.17). This means that

$$F_*\!\left([\tilde{A}, \tilde{B}]_e\right) = [F_* \tilde{A}, F_* \tilde{B}]_{F(e)} = [F_* \tilde{A}, F_* \tilde{B}]_e.$$

The left-hand side of this equality is $F_*[A, B]$, while the right-hand side is

$$[F_* \tilde{A}, F_* \tilde{B}]_e = [(F_* A)^{\sim}, (F_* B)^{\sim}]_e = [F_* A, F_* B].$$

Equating the two sides gives

$$F_*[A, B] = [F_* A, F_* B]. \quad \square$$

</details>
</div>

Suppose $H$ is a Lie subgroup of a Lie group $G$, with inclusion map $i \colon H \to G$. Since $i$ is an immersion, its differential

$$i_* \colon T_e H \to T_e G$$

is injective. To distinguish the Lie bracket on $T_e H$ from the Lie bracket on $T_e G$, we temporarily attach subscripts $T_e H$ and $T_e G$ to the two Lie brackets respectively. By Proposition 16.14, for $X, Y \in T_e H$,

$$i_*([X, Y]_{T_e H}) = [i_* X, i_* Y]_{T_e G}. \tag{16.10}$$

This shows that if $T_e H$ is identified with a subspace of $T_e G$ via $i_*$, then the bracket on $T_e H$ is the restriction of the bracket on $T_e G$ to $T_e H$. Thus, the Lie algebra of a Lie subgroup $H$ may be identified with a Lie subalgebra of the Lie algebra of $G$.

In general, the Lie algebras of the classical groups are denoted by gothic letters. For example, the Lie algebras of $\operatorname{GL}(n, \mathbb{R})$, $\operatorname{SL}(n, \mathbb{R})$, $\operatorname{O}(n)$, and $\operatorname{U}(n)$ are denoted by $\mathfrak{gl}(n, \mathbb{R})$, $\mathfrak{sl}(n, \mathbb{R})$, $\mathfrak{o}(n)$, and $\mathfrak{u}(n)$, respectively. By (16.10) and Proposition 16.11, the Lie algebra structures on $\mathfrak{sl}(n, \mathbb{R})$, $\mathfrak{o}(n)$, and $\mathfrak{u}(n)$ are given by

$$[A, B] = AB - BA,$$

as on $\mathfrak{gl}(n, \mathbb{R})$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 16.15</span></p>

A fundamental theorem in Lie group theory asserts the existence of a one-to-one correspondence between the connected Lie subgroups of a Lie group $G$ and the Lie subalgebras of its Lie algebra $\mathfrak{g}$. For the torus $\mathbb{R}^2 / \mathbb{Z}^2$, the Lie algebra $\mathfrak{g}$ has $\mathbb{R}^2$ as the underlying vector space and the one-dimensional Lie subalgebras are all the lines through the origin. Each line through the origin in $\mathbb{R}^2$ is a subgroup of $\mathbb{R}^2$ under addition. Its image under the quotient map $\mathbb{R}^2 \to \mathbb{R}^2 / \mathbb{Z}^2$ is a subgroup of the torus $\mathbb{R}^2 / \mathbb{Z}^2$. If a line has rational slope, then its image is a regular submanifold of the torus. If a line has irrational slope, then its image is only an immersed submanifold of the torus. According to the correspondence theorem just quoted, the one-dimensional connected Lie subgroups of the torus are the images of all the lines through the origin.

Note that if a Lie subgroup had been defined as a subgroup that is also a *regular* submanifold, then one would have to exclude all the lines with irrational slopes as Lie subgroups of the torus, and it would not be possible to have a one-to-one correspondence between the connected subgroups of a Lie group and the Lie subalgebras of its Lie algebra. It is because of our desire for such a correspondence that a Lie subgroup of a Lie group is defined to be a subgroup that is also an *immersed* submanifold.

</div>

# Chapter 5 — Differential Forms

Differential forms are generalizations of real-valued functions on a manifold. Instead of assigning to each point of the manifold a number, a differential $k$-form assigns to each point a $k$-covector on its tangent space. For $k = 0$ and $1$, differential $k$-forms are functions and covector fields respectively.

Differential forms play a crucial role in manifold theory. First and foremost, they are intrinsic objects associated to any manifold, and so can be used to construct diffeomorphism invariants of a manifold. In contrast to vector fields, which are also intrinsic to a manifold, differential forms have a far richer algebraic structure. Due to the existence of the wedge product, a grading, and the exterior derivative, the set of smooth forms on a manifold is both a graded algebra and a differential complex. Such an algebraic structure is called a *differential graded algebra*. Moreover, the differential complex of smooth forms on a manifold can be pulled back under a smooth map, making the complex into a contravariant functor called the *de Rham complex* of the manifold.

## §17 Differential 1-Forms

### 17.1 The Differential of a Function

Let $M$ be a smooth manifold and $p$ a point in $M$. The **cotangent space** of $M$ at $p$, denoted by $T_p^*(M)$ or $T_p^* M$, is defined to be the dual space of the tangent space $T_p M$:

$$T_p^* M = (T_p M)^\vee = \operatorname{Hom}(T_p M, \mathbb{R}).$$

An element of the cotangent space $T_p^* M$ is called a **covector** at $p$. Thus, a covector $\omega_p$ at $p$ is a linear function

$$\omega_p \colon T_p M \to \mathbb{R}.$$

A **covector field**, a **differential 1-form**, or more simply a **1-form** on $M$, is a function $\omega$ that assigns to each point $p$ in $M$ a covector $\omega_p$ at $p$. In this sense it is dual to a vector field on $M$, which assigns to each point in $M$ a tangent vector at $p$. There are many reasons for the great utility of differential forms in manifold theory, among which is the fact that they can be pulled back under a map. This is in contrast to vector fields, which in general cannot be pushed forward under a map.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 17.1</span><span class="math-callout__name">(Differential of a function)</span></p>

If $f$ is a $C^\infty$ real-valued function on a manifold $M$, its **differential** is defined to be the 1-form $df$ on $M$ such that for any $p \in M$ and $X_p \in T_p M$,

$$(df)_p(X_p) = X_p f.$$

Instead of $(df)_p$, we also write $df\vert_p$ for the value of the 1-form $df$ at $p$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.2</span></p>

If $f \colon M \to \mathbb{R}$ is a $C^\infty$ function, then for $p \in M$ and $X_p \in T_p M$,

$$f_*(X_p) = (df)_p(X_p)\,\frac{d}{dt}\bigg\vert_{f(p)}.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since $f_*(X_p) \in T_{f(p)} \mathbb{R}$, there is a real number $a$ such that

$$f_*(X_p) = a\,\frac{d}{dt}\bigg\vert_{f(p)}. \tag{17.1}$$

To evaluate $a$, apply both sides of (17.1) to $t$:

$$a = f_*(X_p)(t) = X_p(t \circ f) = X_p f = (df)_p(X_p). \quad \square$$

</details>
</div>

This proposition shows that under the canonical identification of the tangent space $T_{f(p)} \mathbb{R}$ with $\mathbb{R}$ via $a\,d/dt\vert_{f(p)} \longleftrightarrow a$, the pushforward $f_*$ is the same as $df$. For this reason, we are justified in calling both of them the **differential** of $f$. In terms of the differential $df$, a $C^\infty$ function $f \colon M \to \mathbb{R}$ has a critical point at $p \in M$ if and only if $(df)_p = 0$.

### 17.2 Local Expression for a Differential 1-Form

Let $(U, \phi) = (U, x^1, \dots, x^n)$ be a coordinate chart on a manifold $M$. Then the differentials $dx^1, \dots, dx^n$ are 1-forms on $U$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.3</span></p>

At each point $p \in U$, the covectors $(dx^1)_p, \dots, (dx^n)_p$ form a basis for the cotangent space $T_p^* M$ dual to the basis $\partial/\partial x^1\vert_p, \dots, \partial/\partial x^n\vert_p$ for the tangent space $T_p M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The proof is just like that in the Euclidean case (Proposition 4.1):

$$(dx^i)_p\!\left(\frac{\partial}{\partial x^j}\bigg\vert_p\right) = \frac{\partial}{\partial x^j}\bigg\vert_p x^i = \delta^i_j. \quad \square$$

</details>
</div>

Thus, every 1-form $\omega$ on $U$ can be written as a linear combination

$$\omega = \sum a_i\,dx^i,$$

where the coefficients $a_i$ are functions on $U$. In particular, if $f$ is a $C^\infty$ function on $M$, then the restriction of the 1-form $df$ to $U$ must be a linear combination $df = \sum a_i\,dx^i$. To find $a_j$, we apply the usual trick of evaluating both sides on $\partial/\partial x^j$:

$$(df)\!\left(\frac{\partial}{\partial x^j}\right) = \sum_i a_i\,dx^i\!\left(\frac{\partial}{\partial x^j}\right) \implies \frac{\partial f}{\partial x^j} = \sum_i a_i \delta^i_j = a_j.$$

This gives a local expression for $df$:

$$df = \sum \frac{\partial f}{\partial x^i}\,dx^i. \tag{17.2}$$

### 17.3 The Cotangent Bundle

The underlying set of the **cotangent bundle** $T^* M$ of a manifold $M$ is the union of the cotangent spaces at all the points of $M$:

$$T^* M := \bigcup_{p \in M} T_p^* M. \tag{17.3}$$

Just as in the case of the tangent bundle, the union (17.3) is a disjoint union and there is a natural map $\pi \colon T^* M \to M$ given by $\pi(\alpha) = p$ if $\alpha \in T_p^* M$. Mimicking the construction of the tangent bundle, we give $T^* M$ a topology as follows. If $(U, \phi) = (U, x^1, \dots, x^n)$ is a chart on $M$ and $p \in U$, then each $\alpha \in T_p^* M$ can be written uniquely as a linear combination

$$\alpha = \sum c_i(\alpha)\,dx^i\vert_p.$$

This gives rise to a bijection

$$\tilde{\phi} \colon T^* U \to \phi(U) \times \mathbb{R}^n, \tag{17.4}$$

$$\alpha \mapsto (\phi(p), c_1(\alpha), \dots, c_n(\alpha)) = (\phi \circ \pi, c_1, \dots, c_n)(\alpha).$$

Using this bijection, we can transfer the topology of $\phi(U) \times \mathbb{R}^n$ to $T^* U$, giving $T^* M$ the structure of a $C^\infty$ manifold and a vector bundle of rank $n$ over $M$. Properly speaking, the **cotangent bundle** of a manifold $M$ is the triple $(T^* M, M, \pi)$, while $T^* M$ and $M$ are the *total space* and the *base space* of the cotangent bundle respectively, but by abuse of language, it is customary to call $T^* M$ the cotangent bundle of $M$.

In terms of the cotangent bundle, a 1-form on $M$ is simply a section of the cotangent bundle $T^* M$; i.e., it is a map $\omega \colon M \to T^* M$ such that $\pi \circ \omega = \mathbb{1}_M$, the identity map on $M$. We say that a 1-form $\omega$ is $C^\infty$ if it is $C^\infty$ as a map $M \to T^* M$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 17.4</span><span class="math-callout__name">(Liouville form on the cotangent bundle)</span></p>

If a manifold $M$ has dimension $n$, then the total space $T^* M$ of its cotangent bundle $\pi \colon T^* M \to M$ is a manifold of dimension $2n$. Remarkably, on $T^* M$ there is a 1-form $\lambda$, called the **Liouville form** (or the **Poincaré form** in some books), defined independently of charts as follows. A point in $T^* M$ is a covector $\omega_p \in T_p^* M$ at some point $p \in M$. If $X_{\omega_p}$ is a tangent vector to $T^* M$ at $\omega_p$, then the pushforward $\pi_*\!\left(X_{\omega_p}\right)$ is a tangent vector to $M$ at $p$. Therefore, one can pair up $\omega_p$ and $\pi_*\!\left(X_{\omega_p}\right)$ to obtain a real number $\omega_p\!\left(\pi_*\!\left(X_{\omega_p}\right)\right)$. Define

$$\lambda_{\omega_p}\!\left(X_{\omega_p}\right) = \omega_p\!\left(\pi_*\!\left(X_{\omega_p}\right)\right).$$

</div>

### 17.4 Characterization of $C^\infty$ 1-Forms

We define a 1-form $\omega$ on a manifold $M$ to be **smooth** if $\omega \colon M \to T^* M$ is smooth as a section of the cotangent bundle $\pi \colon T^* M \to M$. The set of all smooth 1-forms on $M$ has the structure of a vector space, denoted by $\Omega^1(M)$. In a coordinate chart $(U, \phi) = (U, x^1, \dots, x^n)$ on $M$, the value of the 1-form $\omega$ at $p \in U$ is a linear combination

$$\omega_p = \sum a_i(p)\,dx^i\vert_p.$$

As $p$ varies in $U$, the coefficients $a_i$ become functions on $U$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 17.5</span></p>

Let $(U, \phi) = (U, x^1, \dots, x^n)$ be a chart on a manifold $M$. A 1-form $\omega = \sum a_i\,dx^i$ on $U$ is smooth if and only if the coefficient functions $a_i$ are all smooth.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.6</span><span class="math-callout__name">(Smoothness of a 1-form in terms of coefficients)</span></p>

Let $\omega$ be a 1-form on a manifold $M$. The following are equivalent:

**(i)** The 1-form $\omega$ is smooth on $M$.

**(ii)** The manifold $M$ has an atlas such that on any chart $(U, x^1, \dots, x^n)$ of the atlas, the coefficients $a_i$ of $\omega = \sum a_i\,dx^i$ relative to the frame $dx^i$ are all smooth.

**(iii)** On any chart $(U, x^1, \dots, x^n)$ on the manifold, the coefficients $a_i$ of $\omega = \sum a_i\,dx^i$ relative to the frame $dx^i$ are all smooth.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 17.7</span></p>

If $f$ is a $C^\infty$ function on a manifold $M$, then its differential $df$ is a $C^\infty$ 1-form on $M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

On any chart $(U, x^1, \dots, x^n)$ on $M$, the equality $df = \sum (\partial f / \partial x^i)\,dx^i$ holds. Since the coefficients $\partial f / \partial x^i$ are all $C^\infty$, by Proposition 17.6(iii), the 1-form $df$ is $C^\infty$. $\square$

</details>
</div>

If $\omega$ is a 1-form and $X$ is a vector field on a manifold $M$, we define a function $\omega(X)$ on $M$ by the formula

$$\omega(X)_p = \omega_p(X_p) \in \mathbb{R}, \quad p \in M.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.8</span><span class="math-callout__name">(Linearity of a 1-form over functions)</span></p>

Let $\omega$ be a 1-form on a manifold $M$. If $f$ is a function and $X$ is a vector field on $M$, then $\omega(fX) = f\,\omega(X)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

At each point $p \in M$,

$$\omega(fX)_p = \omega_p\!\left(f(p) X_p\right) = f(p)\,\omega_p(X_p) = (f\,\omega(X))_p,$$

because $\omega(X)$ is defined pointwise, and at each point, $\omega_p$ is $\mathbb{R}$-linear in its argument. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.9</span><span class="math-callout__name">(Smoothness of a 1-form in terms of vector fields)</span></p>

A 1-form $\omega$ on a manifold $M$ is $C^\infty$ if and only if for every $C^\infty$ vector field $X$ on $M$, the function $\omega(X)$ is $C^\infty$ on $M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$(\Rightarrow)$ Suppose $\omega$ is a $C^\infty$ 1-form and $X$ is a $C^\infty$ vector field on $M$. On any chart $(U, x^1, \dots, x^n)$ on $M$, by Propositions 14.2 and 17.6, $\omega = \sum a_i\,dx^i$ and $X = \sum b^j\,\partial/\partial x^j$ for $C^\infty$ functions $a_i, b^j$. By the linearity of 1-forms over functions (Proposition 17.8),

$$\omega(X) = \left(\sum a_i\,dx^i\right)\!\left(\sum b^j\,\frac{\partial}{\partial x^j}\right) = \sum_{i,j} a_i b^j \delta^i_j = \sum_i a_i b^i,$$

a $C^\infty$ function on $U$. Since $U$ is an arbitrary chart on $M$, the function $\omega(X)$ is $C^\infty$ on $M$.

$(\Leftarrow)$ Suppose $\omega$ is a 1-form on $M$ such that $\omega(X)$ is $C^\infty$ for every $C^\infty$ vector field $X$ on $M$. Given $p \in M$, choose a coordinate neighborhood $(U, x^1, \dots, x^n)$ about $p$. Then $\omega = \sum a_i\,dx^i$ on $U$ for some functions $a_i$. Fix an integer $j$, $1 \le j \le n$. By Proposition 14.4, we can extend the $C^\infty$ vector field $X = \partial/\partial x^j$ on $U$ to a $C^\infty$ vector field $\tilde{X}$ on $M$ that agrees with $\partial/\partial x^j$ in a neighborhood $V_p^j$ of $p$ in $U$. Restricted to the open set $V_p^j$,

$$\omega(\tilde{X}) = \left(\sum a_i\,dx^i\right)\!\left(\frac{\partial}{\partial x^j}\right) = a_j.$$

This proves that $a_j$ is $C^\infty$ on the coordinate chart $(V_p^j, x^1, \dots, x^n)$. On the intersection $V_p := \bigcap_j V_p^j$, all $a_j$ are $C^\infty$. By Lemma 17.5, the 1-form $\omega$ is $C^\infty$ on $V_p$. So for each $p \in M$, we have found a coordinate neighborhood $V_p$ on which $\omega$ is $C^\infty$. It follows that $\omega$ is a $C^\infty$ map from $M$ to $T^* M$. $\square$

</details>
</div>

Let $\mathcal{F} = C^\infty(M)$ be the ring of all $C^\infty$ functions on $M$. By Proposition 17.9, a 1-form $\omega$ on $M$ defines a map $\mathfrak{X}(M) \to \mathcal{F}$, $X \mapsto \omega(X)$. According to Proposition 17.8, this map is both $\mathbb{R}$-linear and $\mathcal{F}$-linear.

### 17.5 Pullback of 1-Forms

If $F \colon N \to M$ is a $C^\infty$ map of manifolds, then at each point $p \in N$ the differential

$$F_{*,p} \colon T_p N \to T_{F(p)} M$$

is a linear map that pushes forward vectors at $p$ from $N$ to $M$. The **codifferential**, i.e., the dual of the differential,

$$(F_{*,p})^\vee \colon T_{F(p)}^* M \to T_p^* N,$$

reverses the arrow and pulls back a covector at $F(p)$ from $M$ to $N$. Another notation for the codifferential is $F^* = (F_{*,p})^\vee$. By the definition of the dual, if $\omega_{F(p)} \in T_{F(p)}^* M$ is a covector at $F(p)$ and $X_p \in T_p N$ is a tangent vector at $p$, then

$$F^*\!\left(\omega_{F(p)}\right)(X_p) = \left((F_{*,p})^\vee\,\omega_{F(p)}\right)(X_p) = \omega_{F(p)}(F_{*,p} X_p).$$

We call $F^*\!\left(\omega_{F(p)}\right)$ the **pullback** of the covector $\omega_{F(p)}$ by $F$. Thus, the pullback of covectors is simply the codifferential.

Unlike vector fields, which in general cannot be pushed forward under a $C^\infty$ map, every covector field can be pulled back by a $C^\infty$ map. If $\omega$ is a 1-form on $M$, its **pullback** $F^* \omega$ is the 1-form on $N$ defined pointwise by

$$(F^* \omega)_p = F^*\!\left(\omega_{F(p)}\right), \quad p \in N.$$

This means that

$$(F^* \omega)_p(X_p) = \omega_{F(p)}(F_{*,p} X_p)$$

for all $X_p \in T_p N$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.10</span><span class="math-callout__name">(Commutation of the pullback with the differential)</span></p>

Let $F \colon N \to M$ be a $C^\infty$ map of manifolds. For any $h \in C^\infty(M)$, $F^*(dh) = d(F^* h)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

It suffices to check that for any point $p \in N$ and any tangent vector $X_p \in T_p N$,

$$(F^* dh)_p(X_p) = (dF^* h)_p(X_p). \tag{17.5}$$

The left-hand side of (17.5) is

$$(F^* dh)_p(X_p) = (dh)_{F(p)}(F_*(X_p)) = (F_*(X_p))h = X_p(h \circ F).$$

The right-hand side of (17.5) is

$$(dF^* h)_p(X_p) = X_p(F^* h) = X_p(h \circ F). \quad \square$$

</details>
</div>

Pullback of functions and 1-forms respects addition and scalar multiplication.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.11</span><span class="math-callout__name">(Pullback of a sum and a product)</span></p>

Let $F \colon N \to M$ be a $C^\infty$ map of manifolds. Suppose $\omega, \tau \in \Omega^1(M)$ and $g \in C^\infty(M)$. Then

**(i)** $F^*(\omega + \tau) = F^* \omega + F^* \tau$,

**(ii)** $F^*(g\omega) = (F^* g)(F^* \omega)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.12</span><span class="math-callout__name">(Pullback of a $C^\infty$ 1-form)</span></p>

The pullback $F^* \omega$ of a $C^\infty$ 1-form $\omega$ on $M$ under a $C^\infty$ map $F \colon N \to M$ is $C^\infty$ on $N$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Given $p \in N$, choose a chart $(V, \psi) = (V, y^1, \dots, y^n)$ in $M$ about $F(p)$. By the continuity of $F$, there is a chart $(U, \phi) = (U, x^1, \dots, x^n)$ about $p$ in $N$ such that $F(U) \subset V$. On $V$, $\omega = \sum a_i\,dy^i$ for some $a_i \in C^\infty(V)$. On $U$,

$$F^* \omega = \sum (F^* a_i) F^*(dy^i) = \sum (F^* a_i)\,d F^* y^i = \sum (a_i \circ F)\,d(y^i \circ F) = \sum_{i,j} (a_i \circ F)\,\frac{\partial F^i}{\partial x^j}\,dx^j$$

using Proposition 17.11, Proposition 17.10, and equation (17.2). Since the coefficients $(a_i \circ F)\,\partial F^i / \partial x^j$ are all $C^\infty$, by Proposition 17.5 the 1-form $F^* \omega$ is $C^\infty$ on $U$ and therefore at $p$. Since $p$ was an arbitrary point in $N$, the pullback $F^* \omega$ is $C^\infty$ on $N$. $\square$

</details>
</div>

### 17.6 Restriction of 1-Forms to an Immersed Submanifold

Let $S \subset M$ be an immersed submanifold and $i \colon S \hookrightarrow M$ the inclusion map. At any $p \in S$, since the differential $i_* \colon T_p S \to T_p M$ is injective, one may view the tangent space $T_p S$ as a subspace of $T_p M$. If $\omega$ is a 1-form on $M$, then the **restriction** of $\omega$ to $S$ is the 1-form $\omega\vert_S$ defined by

$$(\omega\vert_S)_p(v) = \omega_p(v) \quad \text{for all } p \in S \text{ and } v \in T_p S.$$

Thus, the restriction $\omega\vert_S$ is the same as $\omega$ except that its domain has been restricted from $M$ to $S$ and for each $p \in S$, the domain of $(\omega\vert_S)_p$ has been restricted from $T_p M$ to $T_p S$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.14</span></p>

If $i \colon S \hookrightarrow M$ is the inclusion map of an immersed submanifold $S$ and $\omega$ is a 1-form on $M$, then $i^* \omega = \omega\vert_S$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For $p \in S$ and $v \in T_p S$,

$$(i^* \omega)_p(v) = \omega_{i(p)}(i_* v) = \omega_p(v) = (\omega\vert_S)_p(v),$$

since both $i$ and $i_*$ are inclusions. $\square$

</details>
</div>

To avoid too cumbersome a notation, we sometimes simply write $\omega$ to mean $\omega\vert_S$, relying on the context to make clear that it is the restriction of $\omega$ to $S$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 17.15</span><span class="math-callout__name">(A 1-form on the circle)</span></p>

The velocity vector field of the unit circle $c(t) = (x, y) = (\cos t, \sin t)$ in $\mathbb{R}^2$ is

$$c'(t) = (-\sin t, \cos t) = (-y, x).$$

Thus,

$$X = -y\,\frac{\partial}{\partial x} + x\,\frac{\partial}{\partial y}$$

is a $C^\infty$ vector field on the unit circle $S^1$. What this notation means is that if $x, y$ are the standard coordinates on $\mathbb{R}^2$ and $i \colon S^1 \hookrightarrow \mathbb{R}^2$ is the inclusion map, then at a point $p = (x, y) \in S^1$, one has $i_* X_p = -y\,\partial/\partial x\vert_p + x\,\partial/\partial y\vert_p$, where $\partial/\partial x\vert_p$ and $\partial/\partial y\vert_p$ are tangent vectors at $p$ in $\mathbb{R}^2$.

Find a 1-form $\omega = a\,dx + b\,dy$ on $S^1$ such that $\omega(X) \equiv 1$. Here $\omega$ is viewed as the restriction to $S^1$ of the 1-form $a\,dx + b\,dy$ on $\mathbb{R}^2$. We calculate in $\mathbb{R}^2$, where $dx, dy$ are dual to $\partial/\partial x, \partial/\partial y$:

$$\omega(X) = (a\,dx + b\,dy)\!\left(-y\,\frac{\partial}{\partial x} + x\,\frac{\partial}{\partial y}\right) = -ay + bx = 1. \tag{17.6}$$

Since $x^2 + y^2 = 1$ on $S^1$, $a = -y$ and $b = x$ is a solution to (17.6). So $\omega = -y\,dx + x\,dy$ is one such 1-form. Since $\omega(X) \equiv 1$, the form $\omega$ is nowhere vanishing on the circle.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 17.16</span><span class="math-callout__name">(Pullback of a 1-form)</span></p>

Let $h \colon \mathbb{R} \to S^1 \subset \mathbb{R}^2$ be given by $h(t) = (x, y) = (\cos t, \sin t)$. If $\omega$ is the 1-form $-y\,dx + x\,dy$ on $S^1$, compute the pullback $h^* \omega$.

**Solution.**

$$h^*(-y\,dx + x\,dy) = -(h^* y)\,d(h^* x) + (h^* x)\,d(h^* y)$$

$$= -(\sin t)\,d(\cos t) + (\cos t)\,d(\sin t)$$

$$= \sin^2 t\,dt + \cos^2 t\,dt = dt.$$

</div>

## §18 Differential $k$-Forms

We now generalize the construction of 1-forms on a manifold to $k$-forms. After defining $k$-forms on a manifold, we show that locally they look no different from $k$-forms on $\mathbb{R}^n$. In parallel to the construction of the tangent and cotangent bundles on a manifold, we construct the $k$th exterior power $\bigwedge^k(T^* M)$ of the cotangent bundle. A differential $k$-form is seen to be a section of the bundle $\bigwedge^k(T^* M)$. The pullback and the wedge product of differential forms are defined pointwise. As examples of differential forms, we consider left-invariant forms on a Lie group.

### 18.1 Differential Forms

Recall that a **$k$-tensor** on a vector space $V$ is a $k$-linear function

$$f \colon V \times \cdots \times V \to \mathbb{R}.$$

The $k$-tensor $f$ is **alternating** if for any permutation $\sigma \in S_k$,

$$f(v_{\sigma(1)}, \dots, v_{\sigma(k)}) = (\operatorname{sgn} \sigma)\,f(v_1, \dots, v_k). \tag{18.1}$$

When $k = 1$, the only element of the permutation group $S_1$ is the identity permutation. So for 1-tensors the condition (18.1) is vacuous and all 1-tensors are alternating (and symmetric too). An alternating $k$-tensor on $V$ is also called a **$k$-covector** on $V$.

For any vector space $V$, denote by $A_k(V)$ the vector space of alternating $k$-tensors on $V$. Another common notation for the space $A_k(V)$ is $\bigwedge^k(V^\vee)$. Thus,

$$\bigwedge^0(V^\vee) = A_0(V) = \mathbb{R},$$

$$\bigwedge^1(V^\vee) = A_1(V) = V^\vee,$$

$$\bigwedge^2(V^\vee) = A_2(V), \quad \text{and so on.}$$

We apply the functor $A_k(\;)$ to the tangent space $T_p M$ of a manifold $M$ at a point $p$. The vector space $A_k(T_p M)$, usually denoted by $\bigwedge^k(T_p^* M)$, is the space of all alternating $k$-tensors on the tangent space $T_p M$. A **$k$-covector field** on $M$ is a function $\omega$ that assigns to each point $p \in M$ a $k$-covector $\omega_p \in \bigwedge^k(T_p^* M)$. A $k$-covector field is also called a **differential $k$-form**, a **differential form of degree $k$**, or simply a **$k$-form**. A **top form** on a manifold is a differential form whose degree is the dimension of the manifold.

If $\omega$ is a $k$-form on a manifold $M$ and $X_1, \dots, X_k$ are vector fields on $M$, then $\omega(X_1, \dots, X_k)$ is the function on $M$ defined by

$$(\omega(X_1, \dots, X_k))(p) = \omega_p\!\left((X_1)_p, \dots, (X_k)_p\right).$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.1</span><span class="math-callout__name">(Multilinearity of a form over functions)</span></p>

Let $\omega$ be a $k$-form on a manifold $M$. For any vector fields $X_1, \dots, X_k$ and any function $h$ on $M$,

$$\omega(X_1, \dots, hX_i, \dots, X_k) = h\,\omega(X_1, \dots, X_i, \dots, X_k).$$

</div>

### 18.2 Local Expression for a $k$-Form

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 18.2</span></p>

Let $(U, x^1, \dots, x^n)$ be a coordinate chart on a manifold. At each point $p \in U$, a basis for the tangent space $T_p U$ is

$$\frac{\partial}{\partial x^1}\bigg\vert_p, \dots, \frac{\partial}{\partial x^n}\bigg\vert_p.$$

As we saw in Proposition 17.3, the dual basis for the cotangent space $T_p^* U$ is $(dx^1)_p, \dots, (dx^n)_p$. By Proposition 3.29 a basis for the alternating $k$-tensors in $\bigwedge^k(T_p^* U)$ is

$$(dx^{i_1})_p \wedge \cdots \wedge (dx^{i_k})_p, \quad 1 \le i_1 < \cdots < i_k \le n.$$

If $\omega$ is a $k$-form on $\mathbb{R}^n$, then at each point $p \in \mathbb{R}^n$, $\omega_p$ is a linear combination

$$\omega_p = \sum a_{i_1 \cdots i_k}(p)\,(dx^{i_1})_p \wedge \cdots \wedge (dx^{i_k})_p.$$

Omitting the point $p$, we write

$$\omega = \sum a_{i_1 \cdots i_k}\,dx^{i_1} \wedge \cdots \wedge dx^{i_k}.$$

To simplify the notation, we let

$$\mathcal{J}_{k,n} = \lbrace I = (i_1, \dots, i_k) \mid 1 \le i_1 < \cdots < i_k \le n \rbrace$$

be the set of all strictly ascending multi-indices between $1$ and $n$ of length $k$, and write

$$\omega = \sum_{I \in \mathcal{J}_{k,n}} a_I\,dx^I,$$

where $dx^I$ stands for $dx^{i_1} \wedge \cdots \wedge dx^{i_k}$.

</div>

By Example 18.2, on a coordinate chart $(U, x^1, \dots, x^n)$ of a manifold $M$, a $k$-form on $U$ is a linear combination $\omega = \sum a_I\,dx^I$, where $I \in \mathcal{J}_{k,n}$ and the $a_I$ are functions on $U$. As a shorthand, we write $\partial_i = \partial/\partial x^i$ for the $i$th coordinate vector field. Evaluating pointwise as in Lemma 3.28, we obtain the following equality on $U$ for $I, J \in \mathcal{J}_{k,n}$:

$$dx^I(\partial_{j_1}, \dots, \partial_{j_k}) = \delta_J^I = \begin{cases} 1 & \text{for } I = J, \\ 0 & \text{for } I \neq J. \end{cases} \tag{18.2}$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.3</span><span class="math-callout__name">(A wedge of differentials in local coordinates)</span></p>

Let $(U, x^1, \dots, x^n)$ be a chart on a manifold and $f^1, \dots, f^k$ smooth functions on $U$. Then

$$df^1 \wedge \cdots \wedge df^k = \sum_{I \in \mathcal{J}_{k,n}} \frac{\partial(f^1, \dots, f^k)}{\partial(x^{i_1}, \dots, x^{i_k})}\,dx^{i_1} \wedge \cdots \wedge dx^{i_k}.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

On $U$,

$$df^1 \wedge \cdots \wedge df^k = \sum_{J \in \mathcal{J}_{k,n}} c_J\,dx^{j_1} \wedge \cdots \wedge dx^{j_k} \tag{18.3}$$

for some functions $c_J$. By the definition of the differential, $df^i(\partial/\partial x^j) = \partial f^i / \partial x^j$. Applying both sides of (18.3) to the list of coordinate vectors $\partial_{i_1}, \dots, \partial_{i_k}$, we get

$$\text{LHS} = (df^1 \wedge \cdots \wedge df^k)(\partial_{i_1}, \dots, \partial_{i_k}) = \det\!\left[\frac{\partial f^i}{\partial x^j}\right] = \frac{\partial(f^1, \dots, f^k)}{\partial(x^{i_1}, \dots, x^{i_k})}$$

by Proposition 3.27, and

$$\text{RHS} = \sum_J c_J\,dx^J(\partial_{i_1}, \dots, \partial_{i_k}) = \sum_J c_J \delta^J_I = c_I$$

by Lemma 18.2. Hence, $c_I = \partial(f^1, \dots, f^k)/\partial(x^{i_1}, \dots, x^{i_k})$. $\square$

</details>
</div>

Two cases of Proposition 18.3 are of special interest:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 18.4</span></p>

Let $(U, x^1, \dots, x^n)$ be a chart on a manifold, and let $f, f^1, \dots, f^n$ be $C^\infty$ functions on $U$. Then

**(i)** (1-forms) $df = \sum (\partial f / \partial x^i)\,dx^i$,

**(ii)** (top forms) $df^1 \wedge \cdots \wedge df^n = \det[\partial f^j / \partial x^i]\,dx^1 \wedge \cdots \wedge dx^n$.

</div>

If $(U, x^1, \dots, x^n)$ and $(V, y^1, \dots, y^n)$ are two overlapping charts on a manifold, then on the intersection $U \cap V$, Proposition 18.3 becomes the transition formula for $k$-forms:

$$dy^I = \sum_J \frac{\partial(y^{i_1}, \dots, y^{i_k})}{\partial(x^{j_1}, \dots, x^{j_k})}\,dx^J.$$

### 18.3 The Bundle Point of View

Let $M$ be a manifold of dimension $n$. To better understand differential forms, we mimic the construction of the tangent and cotangent bundles and form the set

$$\bigwedge^k(T^* M) := \bigcup_{p \in M} \bigwedge^k(T_p^* M) = \bigcup_{p \in M} A_k(T_p M).$$

This set is called the **$k$th exterior power** of the cotangent bundle. There is a projection map $\pi \colon \bigwedge^k(T^* M) \to M$ given by $\pi(\alpha) = p$ if $\alpha \in \bigwedge^k(T_p^* M)$. If $(U, \phi)$ is a coordinate chart on $M$, then there is a bijection

$$\bigwedge^k(T^* U) = \bigcup_{p \in U} \bigwedge^k(T_p^* U) \simeq \phi(U) \times \mathbb{R}^{\binom{n}{k}},$$

$$\alpha \in \bigwedge^k(T_p^* U) \mapsto (\phi(p), \lbrace c_I(\alpha) \rbrace_I),$$

where $\alpha = \sum c_I(\alpha)\,dx^I\vert_p$ and $I = (1 \le i_1 < \cdots < i_k \le n)$. In this way we can give $\bigwedge^k(T^* M)$ a topology and even a differentiable structure. The upshot is that the projection map $\pi \colon \bigwedge^k(T^* M) \to M$ is a $C^\infty$ vector bundle of rank $\binom{n}{k}$ and that a differential $k$-form is simply a section of this bundle. As one might expect, we define a $k$-form to be $C^\infty$ if it is $C^\infty$ as a section of the bundle $\pi \colon \bigwedge^k(T^* M) \to M$.

**Notation.** If $E \to M$ is a $C^\infty$ vector bundle, then the vector space of $C^\infty$ sections of $E$ is denoted by $\Gamma(E)$ or $\Gamma(M, E)$. The vector space of all $C^\infty$ $k$-forms on $M$ is usually denoted by $\Omega^k(M)$. Thus,

$$\Omega^k(M) = \Gamma\!\left(\bigwedge^k(T^* M)\right) = \Gamma\!\left(M, \bigwedge^k(T^* M)\right).$$

We defined the 0-tensors and the 0-covectors to be the constants, that is, $L_0(V) = A_0(V) = \mathbb{R}$. Therefore, the bundle $\bigwedge^0(T^* M)$ is simply $M \times \mathbb{R}$ and a 0-form on $M$ is a function on $M$. A $C^\infty$ 0-form on $M$ is thus the same as a $C^\infty$ function on $M$. In our new notation,

$$\Omega^0(M) = \Gamma\!\left(\bigwedge^0(T^* M)\right) = \Gamma(M \times \mathbb{R}) = C^\infty(M).$$

### 18.4 Smooth $k$-Forms

There are several equivalent characterizations of a smooth $k$-form. Since the proofs are similar to those for 1-forms (Lemma 17.5 and Propositions 17.6 and 17.9), we omit them.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 18.6</span><span class="math-callout__name">(Smoothness of a $k$-form on a chart)</span></p>

Let $(U, x^1, \dots, x^n)$ be a chart on a manifold $M$. A $k$-form $\omega = \sum a_I\,dx^I$ on $U$ is smooth if and only if the coefficient functions $a_I$ are all smooth on $U$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.7</span><span class="math-callout__name">(Characterization of a smooth $k$-form)</span></p>

Let $\omega$ be a $k$-form on a manifold $M$. The following are equivalent:

**(i)** The $k$-form $\omega$ is $C^\infty$ on $M$.

**(ii)** The manifold $M$ has an atlas such that on every chart $(U, \phi) = (U, x^1, \dots, x^n)$ in the atlas, the coefficients $a_I$ of $\omega = \sum a_I\,dx^I$ relative to the coordinate frame $\lbrace dx^I \rbrace_{I \in \mathcal{J}_{k,n}}$ are all $C^\infty$.

**(iii)** On every chart $(U, \phi) = (U, x^1, \dots, x^n)$ on $M$, the coefficients $a_I$ of $\omega = \sum a_I\,dx^I$ relative to the coordinate frame $\lbrace dx^I \rbrace_{I \in \mathcal{J}_{k,n}}$ are all $C^\infty$.

**(iv)** For any $k$ smooth vector fields $X_1, \dots, X_k$ on $M$, the function $\omega(X_1, \dots, X_k)$ is $C^\infty$ on $M$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.8</span><span class="math-callout__name">($C^\infty$ extension of a form)</span></p>

Suppose $\tau$ is a $C^\infty$ differential form defined on a neighborhood $U$ of a point $p$ in a manifold $M$. Then there is a $C^\infty$ form $\tilde{\tau}$ on $M$ that agrees with $\tau$ on a possibly smaller neighborhood of $p$.

</div>

### 18.5 Pullback of $k$-Forms

We have defined the pullback of 0-forms and 1-forms under a $C^\infty$ map $F \colon N \to M$. For a $C^\infty$ 0-form on $M$, i.e., a $C^\infty$ function on $M$, the pullback $F^* f$ is simply the composition $F^*(f) = f \circ F \in \Omega^0(N)$.

To generalize the pullback to $k$-forms for all $k \ge 1$, we first recall the pullback of $k$-covectors from Subsection 10.3. A linear map $L \colon V \to W$ of vector spaces induces a pullback map $L^* \colon A_k(W) \to A_k(V)$ by

$$(L^* \alpha)(v_1, \dots, v_k) = \alpha(L(v_1), \dots, L(v_k))$$

for $\alpha \in A_k(W)$ and $v_1, \dots, v_k \in V$.

Now suppose $F \colon N \to M$ is a $C^\infty$ map of manifolds. At each point $p \in N$, the differential $F_{*,p} \colon T_p N \to T_{F(p)} M$ is a linear map of tangent spaces, and so by the preceding paragraph there is a pullback map

$$(F_{*,p})^* \colon A_k(T_{F(p)} M) \to A_k(T_p N).$$

This ugly notation is usually simplified to $F^*$. Thus, if $\omega_{F(p)}$ is a $k$-covector at $F(p)$ in $M$, then its **pullback** $F^*\!\left(\omega_{F(p)}\right)$ is the $k$-covector at $p$ in $N$ given by

$$F^*\!\left(\omega_{F(p)}\right)(v_1, \dots, v_k) = \omega_{F(p)}(F_{*,p} v_1, \dots, F_{*,p} v_k), \quad v_i \in T_p N.$$

Finally, if $\omega$ is a $k$-form on $M$, then its **pullback** $F^* \omega$ is the $k$-form on $N$ defined pointwise by $(F^* \omega)_p = F^*\!\left(\omega_{F(p)}\right)$ for all $p \in N$. Equivalently,

$$(F^* \omega)_p(v_1, \dots, v_k) = \omega_{F(p)}(F_{*,p} v_1, \dots, F_{*,p} v_k), \quad v_i \in T_p N. \tag{18.4}$$

When $k = 1$, this formula specializes to the definition of the pullback of a 1-form in Subsection 17.5. The pullback of a $k$-form (18.4) can be viewed as a composition

$$T_p N \times \cdots \times T_p N \xrightarrow{F_* \times \cdots \times F_*} T_{F(p)} M \times \cdots \times T_{F(p)} M \xrightarrow{\omega_{F(p)}} \mathbb{R}.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.9</span><span class="math-callout__name">(Linearity of the pullback)</span></p>

Let $F \colon N \to M$ be a $C^\infty$ map. If $\omega, \tau$ are $k$-forms on $M$ and $a$ is a real number, then

**(i)** $F^*(\omega + \tau) = F^* \omega + F^* \tau$;

**(ii)** $F^*(a\omega) = a F^* \omega$.

</div>

### 18.6 The Wedge Product

We learned in Section 3 that if $\alpha$ and $\beta$ are alternating tensors of degree $k$ and $\ell$ respectively on a vector space $V$, then their wedge product $\alpha \wedge \beta$ is the alternating $(k + \ell)$-tensor on $V$ defined by

$$(\alpha \wedge \beta)(v_1, \dots, v_{k+\ell}) = \sum (\operatorname{sgn} \sigma)\,\alpha(v_{\sigma(1)}, \dots, v_{\sigma(k)})\,\beta(v_{\sigma(k+1)}, \dots, v_{\sigma(k+\ell)}),$$

where $v_i \in V$ and $\sigma$ runs over all $(k, \ell)$-shuffles of $1, \dots, k + \ell$. For example, if $\alpha$ and $\beta$ are 1-covectors, then

$$(\alpha \wedge \beta)(v_1, v_2) = \alpha(v_1)\,\beta(v_2) - \alpha(v_2)\,\beta(v_1).$$

The wedge product extends pointwise to differential forms on a manifold: for a $k$-form $\omega$ and an $\ell$-form $\tau$ on $M$, define their **wedge product** $\omega \wedge \tau$ to be the $(k + \ell)$-form on $M$ such that

$$(\omega \wedge \tau)_p = \omega_p \wedge \tau_p$$

at all $p \in M$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.10</span></p>

If $\omega$ and $\tau$ are $C^\infty$ forms on $M$, then $\omega \wedge \tau$ is also $C^\infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $(U, x^1, \dots, x^n)$ be a chart on $M$. On $U$,

$$\omega = \sum a_I\,dx^I, \quad \tau = \sum b_J\,dx^J$$

for $C^\infty$ functions $a_I, b_J$ on $U$. Their wedge product on $U$ is

$$\omega \wedge \tau = \left(\sum a_I\,dx^I\right) \wedge \left(\sum b_J\,dx^J\right) = \sum a_I b_J\,dx^I \wedge dx^J.$$

In this sum, $dx^I \wedge dx^J = 0$ if $I$ and $J$ have an index in common. If $I$ and $J$ are disjoint, then $dx^I \wedge dx^J = \pm dx^K$, where $K = I \cup J$ but reordered as an increasing sequence. Thus,

$$\omega \wedge \tau = \sum_K \!\left(\sum_{\substack{I, J = K \\ I, J \text{ disjoint}}} \pm a_I b_J\right) dx^K.$$

Since the coefficients of $dx^K$ are $C^\infty$ on $U$, by Proposition 18.7, $\omega \wedge \tau$ is $C^\infty$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.11</span><span class="math-callout__name">(Pullback of a wedge product)</span></p>

If $F \colon N \to M$ is a $C^\infty$ map of manifolds and $\omega$ and $\tau$ are differential forms on $M$, then

$$F^*(\omega \wedge \tau) = F^* \omega \wedge F^* \tau.$$

</div>

Define the vector space $\Omega^*(M)$ of $C^\infty$ differential forms on a manifold $M$ of dimension $n$ to be the direct sum

$$\Omega^*(M) = \bigoplus_{k=0}^n \Omega^k(M).$$

What this means is that each element of $\Omega^*(M)$ is uniquely a sum $\sum_{k=0}^n \omega_k$, where $\omega_k \in \Omega^k(M)$. With the wedge product, the vector space $\Omega^*(M)$ becomes a graded algebra, the grading being the degree of differential forms.

### 18.7 Differential Forms on a Circle

Consider the map

$$h \colon \mathbb{R} \to S^1, \quad h(t) = (\cos t, \sin t).$$

Since the derivative $h'(t) = (-\sin t, \cos t)$ is nonzero for all $t$, the map $h \colon \mathbb{R} \to S^1$ is a submersion. By Problem 18.8, the pullback map $h^* \colon \Omega^*(S^1) \to \Omega^*(\mathbb{R})$ on smooth differential forms is injective. This will allow us to identify the differential forms on $S^1$ with a subspace of differential forms on $\mathbb{R}$.

Let $\omega = -y\,dx + x\,dy$ be the nowhere-vanishing form on $S^1$ from Example 17.15. In Example 17.16, we showed that $h^* \omega = dt$. Since $\omega$ is nowhere vanishing, it is a frame for the cotangent bundle $T^* S^1$ over $S^1$, and every $C^\infty$ 1-form $\alpha$ on $S^1$ can be written as $\alpha = f\omega$ for some function $f$ on $S^1$. The function $f$ is $C^\infty$. Its pullback $\tilde{f} := h^* f$ is a $C^\infty$ function on $\mathbb{R}$. Since pulling back preserves multiplication (Proposition 18.11),

$$h^* \alpha = (h^* f)(h^* \omega) = \tilde{f}\,dt. \tag{18.5}$$

We say that a function $g$ or a 1-form $g\,dt$ on $\mathbb{R}$ is **periodic of period** $a$ if $g(t + a) = g(t)$ for all $t \in \mathbb{R}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.12</span></p>

For $k = 0, 1$, under the pullback map $h^* \colon \Omega^*(S^1) \to \Omega^*(\mathbb{R})$, smooth $k$-forms on $S^1$ are identified with smooth periodic $k$-forms of period $2\pi$ on $\mathbb{R}$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

If $f \in \Omega^0(S^1)$, then since $h \colon \mathbb{R} \to S^1$ is periodic of period $2\pi$, the pullback $h^* f = f \circ h \in \Omega^0(\mathbb{R})$ is periodic of period $2\pi$.

Conversely, suppose $\tilde{f} \in \Omega^0(\mathbb{R})$ is periodic of period $2\pi$. For $p \in S^1$, let $s$ be the $C^\infty$ inverse in a neighborhood $U$ of $p$ of the local diffeomorphism $h$ and define $f = \tilde{f} \circ s$ on $U$. To show that $f$ is well defined, let $s_1$ and $s_2$ be two inverses of $h$ over $U$. By the periodic properties of sine and cosine, $s_1 = s_2 + 2\pi n$ for some $n \in \mathbb{Z}$. Because $\tilde{f}$ is periodic of period $2\pi$, we have $\tilde{f} \circ s_1 = \tilde{f} \circ s_2$. This proves that $f$ is well defined on $U$.

As $p$ varies over $S^1$, we obtain a well-defined $C^\infty$ function $f$ on $S^1$ such that $\tilde{f} = h^* f$.

As for 1-forms, note that $\Omega^1(S^1) = \Omega^0(S^1)\omega$ and $\Omega^1(\mathbb{R}) = \Omega^0(\mathbb{R})\,dt$. The pullback $h^* \colon \Omega^1(S^1) \to \Omega^1(\mathbb{R})$ is given by $h^*(f\omega) = (h^* f)\,dt$, so the image of $h^*$ consists precisely of the $C^\infty$ periodic 1-forms of period $2\pi$. $\square$

</details>
</div>

### 18.8 Invariant Forms on a Lie Group

Just as there are left-invariant vector fields on a Lie group $G$, so also are there left-invariant differential forms. For $g \in G$, let $\ell_g \colon G \to G$ be left multiplication by $g$. A $k$-form $\omega$ on $G$ is said to be **left-invariant** if $\ell_g^* \omega = \omega$ for all $g \in G$. This means that for all $g, x \in G$,

$$\ell_g^*(\omega_{gx}) = \omega_x.$$

Thus, a left-invariant $k$-form is uniquely determined by its value at the identity, since for any $g \in G$,

$$\omega_g = \ell_{g^{-1}}^*(\omega_e). \tag{18.6}$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 18.13</span><span class="math-callout__name">(A left-invariant 1-form on $S^1$)</span></p>

By Problem 17.3, $\omega = -y\,dx + x\,dy$ is a left-invariant 1-form on $S^1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.14</span></p>

Every left-invariant $k$-form $\omega$ on a Lie group $G$ is $C^\infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By Proposition 18.7(iii), it suffices to prove that for any $k$ smooth vector fields $X_1, \dots, X_k$ on $G$, the function $\omega(X_1, \dots, X_k)$ is $C^\infty$ on $G$. Let $(Y_1)_e, \dots, (Y_n)_e$ be a basis for the tangent space $T_e G$ and $Y_1, \dots, Y_n$ the left-invariant vector fields they generate. Then $Y_1, \dots, Y_n$ is a $C^\infty$ frame on $G$ (Proposition 16.8). Each $X_j$ can be written as a linear combination $X_j = \sum a^i_j Y_i$. By Proposition 12.12, the functions $a^i_j$ are $C^\infty$. Hence, to prove that $\omega$ is $C^\infty$, it suffices to show that $\omega(Y_{i_1}, \dots, Y_{i_k})$ is $C^\infty$ for the left-invariant vector fields $Y_{i_1}, \dots, Y_{i_k}$. But

$$(\omega(Y_{i_1}, \dots, Y_{i_k}))(g) = \omega_g\!\left((Y_{i_1})_g, \dots, (Y_{i_k})_g\right) = (\ell_{g^{-1}}^*(\omega_e))\!\left(\ell_{g*}(Y_{i_1})_e, \dots, \ell_{g*}(Y_{i_k})_e\right) = \omega_e\!\left((Y_{i_1})_e, \dots, (Y_{i_k})_e\right),$$

which is a constant, independent of $g$. Being a constant function, $\omega(Y_{i_1}, \dots, Y_{i_k})$ is $C^\infty$ on $G$. $\square$

</details>
</div>

It follows from this proposition that the vector space $L(G)$ of left-invariant vector fields on $G$ is a subspace of the vector space $\mathfrak{X}(G)$ of all $C^\infty$ vector fields on $G$.

Similarly, a $k$-form $\omega$ on $G$ is said to be **right-invariant** if $r_g^* \omega = \omega$ for all $g \in G$. The analogue of Proposition 18.14, that every right-invariant form on a Lie group is $C^\infty$, is proven in the same way.

Let $\Omega^k(G)^G$ denote the vector space of left-invariant $k$-forms on $G$. The linear map

$$\Omega^k(G)^G \to \bigwedge^k(\mathfrak{g}^\vee), \quad \omega \mapsto \omega_e,$$

has an inverse defined by (18.6) and is therefore an isomorphism. It follows that $\dim \Omega^k(G)^G = \binom{n}{k}$.

## §19 The Exterior Derivative

In contrast to undergraduate calculus, where the basic objects of study are functions, the basic objects in calculus on manifolds are differential forms. Our program now is to learn how to integrate and differentiate differential forms.

Recall that an **antiderivation** on a graded algebra $A = \bigoplus_{k=0}^{\infty} A^k$ is an $\mathbb{R}$-linear map $D \colon A \to A$ such that

$$D(\omega \cdot \tau) = (D\omega) \cdot \tau + (-1)^k \omega \cdot D\tau$$

for $\omega \in A^k$ and $\tau \in A^\ell$. In the graded algebra $A$, an element of $A^k$ is called a **homogeneous element** of degree $k$. The antiderivation is of **degree** $m$ if

$$\deg D\omega = \deg \omega + m$$

for all homogeneous elements $\omega \in A$.

Let $M$ be a manifold and $\Omega^*(M)$ the graded algebra of $C^\infty$ differential forms on $M$. On the graded algebra $\Omega^*(M)$ there is a uniquely and intrinsically defined antiderivation called the **exterior derivative**. The process of applying the exterior derivative is called **exterior differentiation**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 19.1</span><span class="math-callout__name">(Exterior Derivative)</span></p>

An **exterior derivative** on a manifold $M$ is an $\mathbb{R}$-linear map

$$D \colon \Omega^*(M) \to \Omega^*(M)$$

such that

1. $D$ is an antiderivation of degree 1,
2. $D \circ D = 0$,
3. if $f$ is a $C^\infty$ function and $X$ a $C^\infty$ vector field on $M$, then $(Df)(X) = Xf$.

</div>

Condition (iii) says that on 0-forms an exterior derivative agrees with the differential $df$ of a function $f$. Hence, by (17.2), on a coordinate chart $(U, x^1, \dots, x^n)$,

$$Df = df = \sum \frac{\partial f}{\partial x^i}\,dx^i.$$

### 19.1 Exterior Derivative on a Coordinate Chart

We showed in Subsection 4.4 the existence and uniqueness of an exterior derivative on an open subset of $\mathbb{R}^n$. The same proof carries over to any coordinate chart on a manifold.

More precisely, suppose $(U, x^1, \dots, x^n)$ is a coordinate chart on a manifold $M$. Then any $k$-form $\omega$ on $U$ is uniquely a linear combination

$$\omega = \sum a_I\,dx^I, \quad a_I \in C^\infty(U).$$

If $D$ is an exterior derivative on $U$, then

$$D\omega = \sum (Da_I) \wedge dx^I + \sum a_I\,D\,dx^I \quad \text{(by (i))}$$

$$= \sum (Da_I) \wedge dx^I \quad \text{(by (iii) and (ii), } Dd = D^2 = 0\text{)}$$

$$= \sum_I \sum_j \frac{\partial a_I}{\partial x^j}\,dx^j \wedge dx^I \quad \text{(by (iii))}. \tag{19.1}$$

Hence, if an exterior derivative $D$ exists on $U$, then it is uniquely defined by (19.1). To show existence, we define $D$ by the formula (19.1). The proof that $D$ satisfies (i), (ii), and (iii) is the same as in the case of $\mathbb{R}^n$ in Proposition 4.7. We will denote the unique exterior derivative on a chart $(U, \phi)$ by $d_U$.

Like the derivative of a function on $\mathbb{R}^n$, an antiderivation $D$ on $\Omega^*(M)$ has the property that for a $k$-form $\omega$, the value of $D\omega$ at a point $p$ depends only on the values of $\omega$ in a neighborhood of $p$. To explain this, we make a digression on local operators.

### 19.2 Local Operators

An endomorphism of a vector space $W$ is often called an **operator** on $W$. For example, if $W = C^\infty(\mathbb{R})$ is the vector space of $C^\infty$ functions on $\mathbb{R}$, then the derivative $d/dx$ is an operator on $W$:

$$\frac{d}{dx}f(x) = f'(x).$$

The derivative has the property that the value of $f'(x)$ at a point $p$ depends only on the values of $f$ in a small neighborhood of $p$. More precisely, if $f = g$ on an open set $U$ in $\mathbb{R}$, then $f' = g'$ on $U$. We say that the derivative is a **local operator** on $C^\infty(\mathbb{R})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 19.2</span><span class="math-callout__name">(Local Operator)</span></p>

An operator $D \colon \Omega^*(M) \to \Omega^*(M)$ is said to be **local** if for all $k \ge 0$, whenever a $k$-form $\omega \in \Omega^k(M)$ restricts to 0 on an open set $U$ in $M$, then $D\omega \equiv 0$ on $U$.

</div>

Here by restricting to 0 on $U$, we mean that $\omega_p = 0$ at every point $p$ in $U$, and the symbol "$\equiv 0$" means "is identically zero": $(D\omega)_p = 0$ at every point $p$ in $U$. An equivalent criterion for an operator $D$ to be local is that for all $k \ge 0$, whenever two $k$-forms $\omega, \tau \in \Omega^k(M)$ agree on an open set $U$, then $D\omega \equiv D\tau$ on $U$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Integral Operator)</span></p>

Define the integral operator

$$I \colon C^\infty([a,b]) \to C^\infty([a,b])$$

by

$$I(f) = \int_a^b f(t)\,dt.$$

Here $I(f)$ is a number, which we view as a constant function on $[a,b]$. The integral is not a local operator, since the value of $I(f)$ at any point $p$ depends on the values of $f$ over the entire interval $[a,b]$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 19.3</span></p>

Any antiderivation $D$ on $\Omega^*(M)$ is a local operator.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Suppose $\omega \in \Omega^k(M)$ and $\omega \equiv 0$ on an open subset $U$. Let $p$ be an arbitrary point in $U$. It suffices to prove that $(D\omega)_p = 0$.

Choose a $C^\infty$ bump function $f$ at $p$ supported in $U$. In particular, $f \equiv 1$ in a neighborhood of $p$ in $U$. Then $f\omega \equiv 0$ on $M$, since if a point $q$ is in $U$, then $\omega_q = 0$, and if $q$ is not in $U$, then $f(q) = 0$. Applying the antiderivation property of $D$ to $f\omega$, we get

$$0 = D(0) = D(f\omega) = (Df) \wedge \omega + (-1)^0 f \wedge (D\omega).$$

Evaluating the right-hand side at $p$, noting that $\omega_p = 0$ and $f(p) = 1$, gives $0 = (D\omega)_p$. $\square$

</details>
</div>

*Remark.* The same proof shows that a derivation on $\Omega^*(M)$ is also a local operator.

### 19.3 Existence of an Exterior Derivative on a Manifold

To define an exterior derivative on a manifold $M$, let $\omega$ be a $k$-form on $M$ and $p \in M$. Choose a chart $(U, x^1, \dots, x^n)$ about $p$. Suppose $\omega = \sum a_I\,dx^I$ on $U$. In Subsection 19.1 we showed the existence of an exterior derivative $d_U$ on $U$ with the property

$$d_U \omega = \sum da_I \wedge dx^I \quad \text{on } U. \tag{19.2}$$

Define $(d\omega)_p = (d_U \omega)_p$. We now show that $(d_U \omega)_p$ is independent of the chart $U$ containing $p$. If $(V, y^1, \dots, y^n)$ is another chart about $p$ and $\omega = \sum b_J\,dy^J$ on $V$, then on $U \cap V$,

$$\sum a_I\,dx^I = \sum b_J\,dy^J.$$

On $U \cap V$ there is a unique exterior derivative $d_{U \cap V} \colon \Omega^*(U \cap V) \to \Omega^*(U \cap V)$. By the properties of the exterior derivative, on $U \cap V$,

$$\sum da_I \wedge dx^I = \sum db_J \wedge dy^J.$$

In particular,

$$\left(\sum da_I \wedge dx^I\right)_p = \left(\sum db_J \wedge dy^J\right)_p.$$

Thus, $(d\omega)_p = (d_U \omega)_p$ is well defined, independently of the chart $(U, x^1, \dots, x^n)$. As $p$ varies over all points of $M$, this defines an operator

$$d \colon \Omega^*(M) \to \Omega^*(M).$$

To check properties (i), (ii), and (iii), it suffices to check them at each point $p \in M$. As in Subsection 19.1, the verification reduces to the same calculation as for the exterior derivative on $\mathbb{R}^n$ in Proposition 4.7.

### 19.4 Uniqueness of the Exterior Derivative

Suppose $D \colon \Omega^*(M) \to \Omega^*(M)$ is an exterior derivative. We will show that $D$ coincides with the exterior derivative $d$ defined in Subsection 19.3.

If $f$ is a $C^\infty$ function and $X$ a $C^\infty$ vector field on $M$, then by condition (iii) of Definition 19.1,

$$(Df)(X) = Xf = (df)(X).$$

Therefore, $Df = df$ on functions $f \in \Omega^0(M)$.

Next consider a wedge product of exact 1-forms $df^1 \wedge \cdots \wedge df^k$:

$$D(df^1 \wedge \cdots \wedge df^k)$$

$$= D(Df^1 \wedge \cdots \wedge Df^k) \quad (\text{because } Df^i = df^i)$$

$$= \sum_{i=1}^{k} (-1)^{i-1} Df^1 \wedge \cdots \wedge DDf^i \wedge \cdots \wedge Df^k \quad (D \text{ is an antiderivation})$$

$$= 0 \quad (D^2 = 0).$$

Finally, we show that $D$ agrees with $d$ on any $k$-form $\omega \in \Omega^k(M)$. Fix $p \in M$. Choose a chart $(U, x^1, \dots, x^n)$ about $p$ and suppose $\omega = \sum a_I\,dx^I$ on $U$. Extend the functions $a_I, x^1, \dots, x^n$ on $U$ to $C^\infty$ functions $\tilde{a}_I, \tilde{x}^1, \dots, \tilde{x}^n$ on $M$ that agree with $a_I, x^1, \dots, x^n$ on a neighborhood $V$ of $p$ (by Proposition 18.8). Define

$$\tilde{\omega} = \sum \tilde{a}_I\,d\tilde{x}^I \in \Omega^k(M).$$

Then $\omega \equiv \tilde{\omega}$ on $V$. Since $D$ is a local operator,

$$D\omega = D\tilde{\omega} \quad \text{on } V.$$

Thus,

$$(D\omega)_p = (D\tilde{\omega})_p = \left(D \sum \tilde{a}_I\,d\tilde{x}^I\right)_p$$

$$= \left(\sum D\tilde{a}_I \wedge d\tilde{x}^I + \sum \tilde{a}_I \wedge D\,d\tilde{x}^I\right)_p$$

$$= \left(\sum d\tilde{a}_I \wedge d\tilde{x}^I\right)_p \quad (\text{because } D\,d\tilde{x}^I = DD\tilde{x}^I = 0)$$

$$= \left(\sum da_I \wedge dx^I\right)_p \quad (\text{since } D \text{ is a local operator})$$

$$= (d\omega)_p.$$

We have proven the following theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19.4</span></p>

On any manifold $M$ there exists a unique exterior derivative $d \colon \Omega^*(M) \to \Omega^*(M)$ characterized uniquely by the three properties of Definition 19.1.

</div>

### 19.5 Exterior Differentiation Under a Pullback

The pullback of differential forms commutes with the exterior derivative. This fact, together with Proposition 18.11 that the pullback preserves the wedge product, is a cornerstone of calculations involving the pullback. Using these two properties, we will finally be in a position to prove that the pullback of a $C^\infty$ form under a $C^\infty$ map is $C^\infty$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 19.5</span><span class="math-callout__name">(Commutation of the pullback with $d$)</span></p>

Let $F \colon N \to M$ be a smooth map of manifolds. If $\omega \in \Omega^k(M)$, then $dF^*\omega = F^*d\omega$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The case $k = 0$, when $\omega$ is a $C^\infty$ function on $M$, is Proposition 17.10. Next consider the case $k \ge 1$. It suffices to verify $dF^*\omega = F^*d\omega$ at an arbitrary point $p \in N$. This reduces the proof to a local computation, i.e., computation in a coordinate chart. If $(V, y^1, \dots, y^m)$ is a chart on $M$ about $F(p)$, then on $V$,

$$\omega = \sum a_I\,dy^{i_1} \wedge \cdots \wedge dy^{i_k}, \quad I = (i_1 < \cdots < i_k),$$

for some $C^\infty$ functions $a_I$ on $V$ and

$$F^*\omega = \sum (F^*a_I)\,F^*dy^{i_1} \wedge \cdots \wedge F^*dy^{i_k} \quad \text{(Proposition 18.11)}$$

$$= \sum (a_I \circ F)\,dF^{i_1} \wedge \cdots \wedge dF^{i_k} \quad (F^*dy^i = dF^*y^i = d(y^i \circ F) = dF^i).$$

So

$$dF^*\omega = \sum d(a_I \circ F) \wedge dF^{i_1} \wedge \cdots \wedge dF^{i_k}.$$

On the other hand,

$$F^*d\omega = F^*\left(\sum da_I \wedge dy^{i_1} \wedge \cdots \wedge dy^{i_k}\right)$$

$$= \sum F^*da_I \wedge F^*dy^{i_1} \wedge \cdots \wedge F^*dy^{i_k}$$

$$= \sum d(F^*a_I) \wedge dF^{i_1} \wedge \cdots \wedge dF^{i_k} \quad (\text{by the case } k = 0)$$

$$= \sum d(a_I \circ F) \wedge dF^{i_1} \wedge \cdots \wedge dF^{i_k}.$$

Therefore, $dF^*\omega = F^*d\omega$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 19.6</span></p>

If $U$ is an open subset of a manifold $M$ and $\omega \in \Omega^k(M)$, then $(d\omega)\vert_U = d(\omega\vert_U)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $i \colon U \hookrightarrow M$ be the inclusion map. Then $\omega\vert_U = i^*\omega$, so the corollary is simply a restatement of the commutativity of $d$ with $i^*$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Pullback under polar coordinates)</span></p>

Let $U$ be the open set $]0,\infty[ \times ]0, 2\pi[$ in the $(r,\theta)$-plane $\mathbb{R}^2$. Define $F \colon U \subset \mathbb{R}^2 \to \mathbb{R}^2$ by

$$F(r, \theta) = (r\cos\theta,\, r\sin\theta).$$

If $x, y$ are the standard coordinates on the target $\mathbb{R}^2$, compute the pullback $F^*(dx \wedge dy)$.

**Solution.** We first compute $F^*dx$:

$$F^*dx = dF^*x = d(x \circ F) = d(r\cos\theta) = (\cos\theta)\,dr - r\sin\theta\,d\theta.$$

Similarly,

$$F^*dy = dF^*y = d(r\sin\theta) = (\sin\theta)\,dr + r\cos\theta\,d\theta.$$

Since the pullback commutes with the wedge product (Proposition 18.11),

$$F^*(dx \wedge dy) = (F^*dx) \wedge (F^*dy)$$

$$= \bigl((\cos\theta)\,dr - r\sin\theta\,d\theta\bigr) \wedge \bigl((\sin\theta)\,dr + r\cos\theta\,d\theta\bigr)$$

$$= (r\cos^2\theta + r\sin^2\theta)\,dr \wedge d\theta \quad (\text{because } d\theta \wedge dr = -dr \wedge d\theta)$$

$$= r\,dr \wedge d\theta.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 19.7</span></p>

If $F \colon N \to M$ is a $C^\infty$ map of manifolds and $\omega$ is a $C^\infty$ $k$-form on $M$, then $F^*\omega$ is a $C^\infty$ $k$-form on $N$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

It is enough to show that every point in $N$ has a neighborhood on which $F^*\omega$ is $C^\infty$. Fix $p \in N$ and choose a chart $(V, y^1, \dots, y^m)$ on $M$ about $F(p)$. Let $F^i = y^i \circ F$ be the $i$th coordinate of the map $F$ in this chart. By the continuity of $F$, there is a chart $(U, x^1, \dots, x^n)$ on $N$ about $p$ such that $F(U) \subset V$. Because $\omega$ is $C^\infty$, on $V$,

$$\omega = \sum_I a_I\,dy^{i_1} \wedge \cdots \wedge dy^{i_k}$$

for some $C^\infty$ functions $a_I \in C^\infty(V)$ (Proposition 18.7(i)$\Rightarrow$(ii)). By properties of the pullback,

$$F^*\omega = \sum (F^*a_I)\,F^*dy^{i_1} \wedge \cdots \wedge F^*dy^{i_k} \quad \text{(Propositions 18.9 and 18.11)}$$

$$= \sum (a_I \circ F)\,dF^{i_1} \wedge \cdots \wedge dF^{i_k} \quad \text{(Proposition 19.5)}.$$

Since the $a_I \circ F$ and $\partial(F^{i_1}, \dots, F^{i_k})/\partial(x^{j_1}, \dots, x^{j_k})$ are all $C^\infty$, $F^*\omega$ is $C^\infty$ by Proposition 18.7(iii)$\Rightarrow$(i). $\square$

</details>
</div>

In summary, if $F \colon N \to M$ is a $C^\infty$ map of manifolds, then the pullback map $F^* \colon \Omega^*(M) \to \Omega^*(N)$ is a morphism of differential graded algebras, i.e., a degree-preserving algebra homomorphism that commutes with the differential.

### 19.6 Restriction of $k$-Forms to a Submanifold

The restriction of a $k$-form to an immersed submanifold is just like the restriction of a 1-form, but with $k$ arguments. Let $S$ be a regular submanifold of a manifold $M$. If $\omega$ is a $k$-form on $M$, then the **restriction** of $\omega$ to $S$ is the $k$-form $\omega\vert_S$ on $S$ defined by

$$(\omega\vert_S)_p(v_1, \dots, v_k) = \omega_p(v_1, \dots, v_k)$$

for $v_1, \dots, v_k \in T_p S \subset T_p M$. Thus, $(\omega\vert_S)_p$ is obtained from $\omega_p$ by restricting the domain of $\omega_p$ to $T_p S \times \cdots \times T_p S$ ($k$ times). As in Proposition 17.14, the restriction of $k$-forms is the same as the pullback under the inclusion map $i \colon S \hookrightarrow M$.

A nonzero form on $M$ may restrict to the zero form on a submanifold $S$. For example, if $S$ is a smooth curve in $\mathbb{R}^2$ defined by the nonconstant function $f(x,y)$, then $df = (\partial f / \partial x)\,dx + (\partial f / \partial y)\,dy$ is a nonzero 1-form on $\mathbb{R}^2$, but since $f$ is identically zero on $S$, the differential $df$ is also identically zero on $S$. Thus, $(df)\vert_S \equiv 0$.

One should distinguish between a *nonzero* form and a *nowhere-zero* or *nowhere-vanishing* form. For example, $x\,dy$ is a nonzero form on $\mathbb{R}^2$, meaning that it is not identically zero. However, it is not nowhere-zero, because it vanishes on the $y$-axis. On the other hand, $dx$ and $dy$ are nowhere-zero 1-forms on $\mathbb{R}^2$.

**Notation.** Since pullback and exterior differentiation commute, $(df)\vert_S = d(f\vert_S)$, so one may write $df\vert_S$ to mean either expression.

### 19.7 A Nowhere-Vanishing 1-Form on the Circle

In Example 17.15 we found a nowhere-vanishing 1-form $-y\,dx + x\,dy$ on the unit circle. As an application of the exterior derivative, we will construct in a different way a nowhere-vanishing 1-form on the circle. One advantage of the new method is that it generalizes to the construction of a nowhere-vanishing top form on a **smooth hypersurface** in $\mathbb{R}^{n+1}$, a regular level set of a smooth function $f \colon \mathbb{R}^{n+1} \to \mathbb{R}$. As we will see in Section 21, the existence of a nowhere-vanishing top form is intimately related to orientations on a manifold.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 19.8</span><span class="math-callout__name">(Nowhere-vanishing 1-form on $S^1$)</span></p>

Let $S^1$ be the unit circle defined by $x^2 + y^2 = 1$ in $\mathbb{R}^2$. The 1-form $dx$ restricts from $\mathbb{R}^2$ to a 1-form on $S^1$. At each point $p \in S^1$, the domain of $(dx\vert_{S^1})_p$ is $T_p(S^1)$ instead of $T_p(\mathbb{R}^2)$:

$$(dx\vert_{S^1})_p \colon T_p(S^1) \to \mathbb{R}.$$

At $p = (1, 0)$, a basis for the tangent space $T_p(S^1)$ is $\partial/\partial y$. Since

$$(dx)_p\!\left(\frac{\partial}{\partial y}\right) = 0,$$

we see that although $dx$ is a nowhere-vanishing 1-form on $\mathbb{R}^2$, it vanishes at $(1, 0)$ when restricted to $S^1$.

To find a nowhere-vanishing 1-form on $S^1$, we take the exterior derivative of both sides of the equation

$$x^2 + y^2 = 1.$$

Using the antiderivation property of $d$, we get

$$2x\,dx + 2y\,dy = 0. \tag{19.3}$$

Of course, this equation is valid only at a point $(x,y) \in S^1$. Let

$$U_x = \lbrace (x,y) \in S^1 \mid x \ne 0 \rbrace \quad \text{and} \quad U_y = \lbrace (x,y) \in S^1 \mid y \ne 0 \rbrace.$$

By (19.3), on $U_x \cap U_y$,

$$\frac{dy}{x} = -\frac{dx}{y}.$$

Define a 1-form $\omega$ on $S^1$ by

$$\omega = \begin{cases} \dfrac{dy}{x} & \text{on } U_x, \\[6pt] -\dfrac{dx}{y} & \text{on } U_y. \end{cases} \tag{19.4}$$

Since these two 1-forms agree on $U_x \cap U_y$, $\omega$ is a well-defined 1-form on $S^1 = U_x \cup U_y$.

To show that $\omega$ is $C^\infty$ and nowhere-vanishing, we need charts. Let

$$U_x^+ = \lbrace (x, y) \in S^1 \mid x > 0 \rbrace.$$

On $U_x^+$, $y$ is a local coordinate, so $dy$ is a basis for the cotangent space $T_p^*(S^1)$ at each point $p \in U_x^+$. Since $\omega = dy/x$ on $U_x^+$, $\omega$ is $C^\infty$ and nowhere zero on $U_x^+$. A similar argument applies to $dy/x$ on $U_x^-$ and $-dx/y$ on $U_y^+$ and $U_y^-$. Hence, $\omega$ is $C^\infty$ and nowhere vanishing on $S^1$.

</div>

## §20 The Lie Derivative and Interior Multiplication

The only portion of this section necessary for the remainder of the book is Subsection 20.4 on interior multiplication. The rest may be omitted on first reading.

The construction of exterior differentiation in Section 19 is local and depends on a choice of coordinates: if $\omega = \sum a_I\,dx^I$, then

$$d\omega = \sum \frac{\partial a_I}{\partial x^j}\,dx^j \wedge dx^I.$$

It turns out, however, that this $d$ is in fact global and intrinsic to the manifold, i.e., independent of the choice of local coordinates. Indeed, for a $C^\infty$ 1-form $\omega$ and $C^\infty$ vector fields $X, Y$ on a manifold $M$, one has the formula

$$(d\omega)(X, Y) = X\omega(Y) - Y\omega(X) - \omega([X, Y]).$$

In this section we will derive a global intrinsic formula like this for the exterior derivative of a $k$-form. The proof uses the Lie derivative and interior multiplication, two other intrinsic operations on a manifold. The Lie derivative is a way of differentiating a vector field or a differential form on a manifold along another vector field. For any vector field $X$ on a manifold, the interior multiplication $\iota_X$ is an antiderivation of degree $-1$ on differential forms. Being intrinsic operators on a manifold, both the Lie derivative and interior multiplication are important in their own right in differential topology and geometry.

### 20.1 Families of Vector Fields and Differential Forms

A collection $\lbrace X_t \rbrace$ or $\lbrace \omega_t \rbrace$ of vector fields or differential forms on a manifold is said to be a **1-parameter family** if the parameter $t$ runs over some subset of the real line. Let $I$ be an open interval in $\mathbb{R}$ and let $M$ be a manifold. Suppose $\lbrace X_t \rbrace$ is a 1-parameter family of vector fields on $M$ defined for all $t \in I$ except at $t_0 \in I$. We say that the **limit** $\lim_{t \to t_0} X_t$ exists if every point $p \in M$ has a coordinate neighborhood $(U, x^1, \dots, x^n)$ on which $X_t\vert_p = \sum a^i(t, p)\,\partial/\partial x^i\vert_p$ and $\lim_{t \to t_0} a^i(t, p)$ exists for all $i$. In this case, we set

$$\lim_{t \to t_0} X_t\big\vert_p = \sum_{i=1}^{n} \lim_{t \to t_0} a^i(t, p)\,\frac{\partial}{\partial x^i}\bigg\vert_p. \tag{20.1}$$

A 1-parameter family $\lbrace X_t \rbrace_{t \in I}$ of smooth vector fields on $M$ is said to **depend smoothly** on $t$ if every point in $M$ has a coordinate neighborhood $(U, x^1, \dots, x^n)$ on which

$$(X_t)_p = \sum a^i(t, p)\,\frac{\partial}{\partial x^i}\bigg\vert_p, \qquad (t, p) \in I \times U, \tag{20.2}$$

for some $C^\infty$ functions $a^i$ on $I \times U$. In this case we also say that $\lbrace X_t \rbrace_{t \in I}$ is a **smooth family of vector fields** on $M$.

For a smooth family of vector fields on $M$, one can define its derivative with respect to $t$ at $t = t_0$ by

$$\left(\frac{d}{dt}\bigg\vert_{t=t_0} X_t\right)_p = \sum \frac{\partial a^i}{\partial t}(t_0, p)\,\frac{\partial}{\partial x^i}\bigg\vert_p \tag{20.3}$$

for $(t_0, p) \in I \times U$. It is easy to check that this definition is independent of the chart $(U, x^1, \dots, x^n)$ containing $p$ (Problem 20.3). Clearly, the derivative $d/dt\vert_{t=t_0} X_t$ is a smooth vector field on $M$.

Similarly, a 1-parameter family $\lbrace \omega_t \rbrace_{t \in I}$ of smooth $k$-forms on $M$ is said to **depend smoothly** on $t$ if every point of $M$ has a coordinate neighborhood $(U, x^1, \dots, x^n)$ on which

$$(\omega_t)_p = \sum b_J(t, p)\,dx^J\big\vert_p, \qquad (t, p) \in I \times U,$$

for some $C^\infty$ functions $b_J$ on $I \times U$. We also call such a family $\lbrace \omega_t \rbrace_{t \in I}$ a **smooth family of $k$-forms** on $M$ and define its derivative with respect to $t$ to be

$$\left(\frac{d}{dt}\bigg\vert_{t=t_0} \omega_t\right)_p = \sum \frac{\partial b_J}{\partial t}(t_0, p)\,dx^J\big\vert_p.$$

As for vector fields, this definition is independent of the chart and defines a $C^\infty$ $k$-form $d/dt\vert_{t=t_0}\,\omega_t$ on $M$.

**Notation.** We write $d/dt$ for the derivative of a smooth family of vector fields or differential forms, but $\partial/\partial t$ for the partial derivative of a function of several variables.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 20.1</span><span class="math-callout__name">(Product rule for $d/dt$)</span></p>

If $\lbrace \omega_t \rbrace$ and $\lbrace \tau_t \rbrace$ are smooth families of $k$-forms and $\ell$-forms respectively on a manifold $M$, then

$$\frac{d}{dt}(\omega_t \wedge \tau_t) = \left(\frac{d}{dt}\,\omega_t\right) \wedge \tau_t + \omega_t \wedge \frac{d}{dt}\,\tau_t.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 20.2</span><span class="math-callout__name">(Commutation of $d/dt\vert_{t=t_0}$ with $d$)</span></p>

If $\lbrace \omega_t \rbrace_{t \in I}$ is a smooth family of differential forms on a manifold $M$, then

$$\frac{d}{dt}\bigg\vert_{t=t_0} d\omega_t = d\left(\frac{d}{dt}\bigg\vert_{t=t_0} \omega_t\right).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

It is enough to check the equality at an arbitrary point $p \in M$. Let $(U, x^1, \dots, x^n)$ be a neighborhood of $p$ such that $\omega = \sum_J b_J\,dx^J$ for some $C^\infty$ functions $b_J$ on $I \times U$. On $U$,

$$\frac{d}{dt}(d\omega_t) = \frac{d}{dt} \sum_{j,J} \frac{\partial b_J}{\partial x^j}\,dx^j \wedge dx^J \quad \text{(note that there is no } dt \text{ term)}$$

$$= \sum_{j,J} \frac{\partial}{\partial x^j}\!\left(\frac{\partial b_J}{\partial t}\right) dx^j \wedge dx^J \quad (\text{since } b_J \text{ is } C^\infty)$$

$$= d\!\left(\sum_J \frac{\partial b_J}{\partial t}\,dx^J\right) = d\!\left(\frac{d}{dt}\,\omega_t\right).$$

Evaluation at $t = t_0$ commutes with $d$, because $d$ involves only partial derivatives with respect to the $x^i$ variables. $\square$

</details>
</div>

### 20.2 The Lie Derivative of a Vector Field

In a first course on calculus, one defines the derivative of a real-valued function $f$ on $\mathbb{R}$ at a point $p \in \mathbb{R}$ as

$$f'(p) = \lim_{t \to 0} \frac{f(p+t) - f(p)}{t}.$$

The problem in generalizing this definition to the derivative of a vector field $Y$ on a manifold $M$ is that at two nearby points $p$ and $q$ in $M$, the tangent vectors $Y_p$ and $Y_q$ are in different vector spaces $T_pM$ and $T_qM$ and so it is not possible to compare them by subtracting one from the other. One way to get around this difficulty is to use the local flow of another vector field $X$ to transport $Y_q$ to the tangent space $T_pM$ at $p$. This leads to the definition of the Lie derivative of a vector field.

Recall from Subsection 14.3 that for any smooth vector field $X$ on $M$ and point $p$ in $M$, there is a neighborhood $U$ of $p$ on which the vector field $X$ has a **local flow**; this means that there exist a real number $\varepsilon > 0$ and a map

$$\varphi \colon ]-\varepsilon, \varepsilon[ \times U \to M$$

such that if we set $\varphi_t(q) = \varphi(t, q)$, then

$$\frac{\partial}{\partial t}\varphi_t(q) = X_{\varphi_t(q)}, \quad \varphi_0(q) = q \quad \text{for } q \in U. \tag{20.5}$$

In other words, for each $q$ in $U$, the curve $\varphi_t(q)$ is an integral curve of $X$ with initial point $q$. By definition, $\varphi_0 \colon U \to U$ is the identity map. The local flow satisfies the property

$$\varphi_s \circ \varphi_t = \varphi_{s+t}$$

whenever both sides are defined (see (14.10)). Consequently, for each $t$ the map $\varphi_t \colon U \to \varphi_t(U)$ is a diffeomorphism onto its image, with a $C^\infty$ inverse $\varphi_{-t}$:

$$\varphi_{-t} \circ \varphi_t = \varphi_0 = \mathbb{1}, \quad \varphi_t \circ \varphi_{-t} = \varphi_0 = \mathbb{1}.$$

Let $Y$ be a $C^\infty$ vector field on $M$. To compare the values of $Y$ at $\varphi_t(p)$ and at $p$, we use the diffeomorphism $\varphi_{-t} \colon \varphi_t(U) \to U$ to push $Y_{\varphi_t(p)}$ into $T_pM$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 20.3</span><span class="math-callout__name">(Lie Derivative of a Vector Field)</span></p>

For $X, Y \in \mathfrak{X}(M)$ and $p \in M$, let $\varphi \colon ]-\varepsilon, \varepsilon[ \times U \to M$ be a local flow of $X$ on a neighborhood $U$ of $p$ and define the **Lie derivative** $\mathcal{L}_X Y$ of $Y$ with respect to $X$ at $p$ to be the vector

$$(\mathcal{L}_X Y)_p = \lim_{t \to 0} \frac{\varphi_{-t*}\!\left(Y_{\varphi_t(p)}\right) - Y_p}{t} = \lim_{t \to 0} \frac{(\varphi_{-t*}Y)_p - Y_p}{t} = \frac{d}{dt}\bigg\vert_{t=0} (\varphi_{-t*}Y)_p.$$

</div>

In this definition the limit is taken in the finite-dimensional vector space $T_pM$. For the derivative to exist, it suffices that $\lbrace \varphi_{-t*}Y \rbrace$ be a smooth family of vector fields on $M$. To show the smoothness of the family $\lbrace \varphi_{-t*}Y \rbrace$, we write out $\varphi_{-t*}Y$ in local coordinates $x^1, \dots, x^n$ in a chart. Let $\varphi_t^i$ and $\varphi^i$ be the $i$th components of $\varphi_t$ and $\varphi$ respectively. Then

$$(\varphi_t)^i(p) = \varphi^i(t, p) = (x^i \circ \varphi)(t, p).$$

By Proposition 8.11, relative to the frame $\lbrace \partial/\partial x^j \rbrace$, the differential $\varphi_{t*}$ at $p$ is represented by the Jacobian matrix $[\partial(\varphi_t)^i / \partial x^j(p)] = [\partial\varphi^i / \partial x^j(t, p)]$. This means that

$$\varphi_{t*}\!\left(\frac{\partial}{\partial x^j}\bigg\vert_p\right) = \sum_i \frac{\partial \varphi^i}{\partial x^j}(t, p)\,\frac{\partial}{\partial x^i}\bigg\vert_{\varphi_t(p)}.$$

Thus, if $Y = \sum b^j\,\partial/\partial x^j$, then

$$\varphi_{-t*}\!\left(Y_{\varphi_t(p)}\right) = \sum_j b^j(\varphi(t, p))\,\varphi_{-t*}\!\left(\frac{\partial}{\partial x^j}\bigg\vert_{\varphi_t(p)}\right)$$

$$= \sum_{i,j} b^j(\varphi(t, p))\,\frac{\partial \varphi^i}{\partial x^j}(-t, p)\,\frac{\partial}{\partial x^i}\bigg\vert_p. \tag{20.6}$$

When $X$ and $Y$ are $C^\infty$ vector fields on $M$, both $\varphi^i$ and $b^j$ are $C^\infty$ functions. The formula (20.6) then shows that $\lbrace \varphi_{-t*}Y \rbrace$ is a smooth family of vector fields on $M$. It follows that the Lie derivative $\mathcal{L}_X Y$ exists and is given in local coordinates by

$$(\mathcal{L}_X Y)_p = \frac{d}{dt}\bigg\vert_{t=0} \varphi_{-t*}\!\left(Y_{\varphi_t(p)}\right)$$

$$= \sum_{i,j} \frac{\partial}{\partial t}\bigg\vert_{t=0} \left(b^j(\varphi(t, p))\,\frac{\partial \varphi^i}{\partial x^j}(-t, p)\right) \frac{\partial}{\partial x^i}\bigg\vert_p. \tag{20.7}$$

It turns out that the Lie derivative of a vector field gives nothing new.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 20.4</span></p>

If $X$ and $Y$ are $C^\infty$ vector fields on a manifold $M$, then the Lie derivative $\mathcal{L}_X Y$ coincides with the Lie bracket $[X, Y]$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

It suffices to check the equality $\mathcal{L}_X Y = [X, Y]$ at every point. To do this, we expand both sides in local coordinates. Suppose a local flow for $X$ is $\varphi \colon ]-\varepsilon, \varepsilon[ \times U \to M$, where $U$ is a coordinate chart with coordinates $x^1, \dots, x^n$. Let $X = \sum a^i\,\partial/\partial x^i$ and $Y = \sum b^j\,\partial/\partial x^j$ on $U$. The condition (20.5) that $\varphi_t(p)$ be an integral curve of $X$ translates into the equations

$$\frac{\partial \varphi^i}{\partial t}(t, p) = a^i(\varphi(t, p)), \qquad i = 1, \dots, n, \quad (t, p) \in ]-\varepsilon, \varepsilon[ \times U.$$

At $t = 0$, $\partial\varphi^i/\partial t(0, p) = a^i(\varphi(0, p)) = a^i(p)$. Since $\varphi(0, p) = p$, the identity map has Jacobian equal to the identity matrix. Thus,

$$\frac{\partial \varphi^i}{\partial x^j}(0, p) = \delta^i_j, \quad \text{the Kronecker delta.}$$

So (20.8) simplifies to

$$\mathcal{L}_X Y = \sum_{i,k} \left(a^k\,\frac{\partial b^i}{\partial x^k} - b^k\,\frac{\partial a^i}{\partial x^k}\right)\frac{\partial}{\partial x^i} = [X, Y].$$

$\square$

</details>
</div>

Although the Lie derivative of a vector field gives us nothing new, in conjunction with the Lie derivative of differential forms it turns out to be a tool of great utility, for example, in the proof of the global formula for the exterior derivative in Theorem 20.14.

### 20.3 The Lie Derivative of a Differential Form

Let $X$ be a smooth vector field and $\omega$ a smooth $k$-form on a manifold $M$. Fix a point $p \in M$ and let $\varphi_t \colon U \to M$ be a flow of $X$ in a neighborhood $U$ of $p$. The definition of the Lie derivative of a differential form is similar to that of the Lie derivative of a vector field. However, instead of pushing a vector at $\varphi_t(p)$ to $p$ via $(\varphi_{-t})_*$, we now pull the $k$-covector $\omega_{\varphi_t(p)}$ back to $p$ via $\varphi_t^*$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 20.5</span><span class="math-callout__name">(Lie Derivative of a Differential Form)</span></p>

For $X$ a smooth vector field and $\omega$ a smooth $k$-form on a manifold $M$, the **Lie derivative** $\mathcal{L}_X\,\omega$ at $p \in M$ is

$$(\mathcal{L}_X\,\omega)_p = \lim_{t \to 0} \frac{\varphi_t^*\!\left(\omega_{\varphi_t(p)}\right) - \omega_p}{t} = \lim_{t \to 0} \frac{(\varphi_t^*\,\omega)_p - \omega_p}{t} = \frac{d}{dt}\bigg\vert_{t=0} (\varphi_t^*\,\omega)_p.$$

</div>

By an argument similar to that for the existence of the Lie derivative $\mathcal{L}_X Y$ in Section 20.2, one shows that $\lbrace \varphi_t^*\omega \rbrace$ is a smooth family of $k$-forms on $M$ by writing it out in local coordinates. The existence of $(\mathcal{L}_X\,\omega)_p$ follows.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 20.6</span></p>

If $f$ is a $C^\infty$ function and $X$ a $C^\infty$ vector field on $M$, then $\mathcal{L}_X f = Xf$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Fix a point $p$ in $M$ and let $\varphi_t \colon U \to M$ be a local flow of $X$ as above. Then

$$(\mathcal{L}_X f)_p = \frac{d}{dt}\bigg\vert_{t=0} (\varphi_t^* f)_p = \frac{d}{dt}\bigg\vert_{t=0} (f \circ \varphi_t)(p) = X_p f,$$

since $\varphi_t(p)$ is a curve through $p$ with initial vector $X_p$ (Proposition 8.17). $\square$

</details>
</div>

### 20.4 Interior Multiplication

We first define interior multiplication on a vector space. If $\beta$ is a $k$-covector on a vector space $V$ and $v \in V$, for $k \ge 2$ the **interior multiplication** or **contraction** of $\beta$ with $v$ is the $(k-1)$-covector $\iota_v \beta$ defined by

$$(\iota_v \beta)(v_2, \dots, v_k) = \beta(v, v_2, \dots, v_k), \quad v_2, \dots, v_k \in V.$$

We define $\iota_v \beta = \beta(v) \in \mathbb{R}$ for a 1-covector $\beta$ on $V$ and $\iota_v \beta = 0$ for a 0-covector $\beta$ (a constant) on $V$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 20.7</span></p>

For 1-covectors $\alpha^1, \dots, \alpha^k$ on a vector space $V$ and $v \in V$,

$$\iota_v(\alpha^1 \wedge \cdots \wedge \alpha^k) = \sum_{i=1}^{k} (-1)^{i-1} \alpha^i(v)\,\alpha^1 \wedge \cdots \wedge \widehat{\alpha^i} \wedge \cdots \wedge \alpha^k,$$

where the caret $\widehat{\phantom{x}}$ over $\alpha^i$ means that $\alpha^i$ is omitted from the wedge product.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$$\left(\iota_v\!\left(\alpha^1 \wedge \cdots \wedge \alpha^k\right)\right)(v_2, \dots, v_k) = \left(\alpha^1 \wedge \cdots \wedge \alpha^k\right)(v, v_2, \dots, v_k)$$

$$= \det\begin{bmatrix} \alpha^1(v) & \alpha^1(v_2) & \cdots & \alpha^1(v_k) \\ \alpha^2(v) & \alpha^2(v_2) & \cdots & \alpha^2(v_k) \\ \vdots & \vdots & \ddots & \vdots \\ \alpha^k(v) & \alpha^k(v_2) & \cdots & \alpha^k(v_k) \end{bmatrix} \quad \text{(Proposition 3.27)}$$

$$= \sum_{i=1}^{k} (-1)^{i+1} \alpha^i(v)\det[\alpha^\ell(v_j)]_{1 \le \ell \le k, \ell \ne i \atop 2 \le j \le k} \quad \text{(expansion along first column)}$$

$$= \sum_{i=1}^{k} (-1)^{i+1} \alpha^i(v)\left(\alpha^1 \wedge \cdots \wedge \widehat{\alpha^i} \wedge \cdots \wedge \alpha^k\right)(v_2, \dots, v_k) \quad \text{(Proposition 3.27).}$$

$\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 20.8</span></p>

For $v$ in a vector space $V$, let $\iota_v \colon \bigwedge^*(V^\vee) \to \bigwedge^{*-1}(V^\vee)$ be interior multiplication by $v$. Then

1. $\iota_v \circ \iota_v = 0$,
2. for $\beta \in \bigwedge^k(V^\vee)$ and $\gamma \in \bigwedge^\ell(V^\vee)$,

$$\iota_v(\beta \wedge \gamma) = (\iota_v \beta) \wedge \gamma + (-1)^k \beta \wedge \iota_v \gamma.$$

In other words, $\iota_v$ is an antiderivation of degree $-1$ whose square is zero.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**(i)** Let $\beta \in \bigwedge^k(V^\vee)$. By the definition of interior multiplication,

$$(\iota_v(\iota_v \beta))(v_3, \dots, v_k) = (\iota_v \beta)(v, v_3, \dots, v_k) = \beta(v, v, v_3, \dots, v_k) = 0,$$

because $\beta$ is alternating and there is a repeated variable $v$ among its arguments.

**(ii)** Since both sides of the equation are linear in $\beta$ and in $\gamma$, we may assume that $\beta = \alpha^1 \wedge \cdots \wedge \alpha^k$ and $\gamma = \alpha^{k+1} \wedge \cdots \wedge \alpha^{k+\ell}$, where the $\alpha^i$ are all 1-covectors. Then

$$\iota_v(\beta \wedge \gamma) = \iota_v(\alpha^1 \wedge \cdots \wedge \alpha^{k+\ell})$$

$$= \left(\sum_{i=1}^{k} (-1)^{i-1} \alpha^i(v)\,\alpha^1 \wedge \cdots \wedge \widehat{\alpha^i} \wedge \cdots \wedge \alpha^k\right) \wedge \alpha^{k+1} \wedge \cdots \wedge \alpha^{k+\ell}$$

$$+ (-1)^k \alpha^1 \wedge \cdots \wedge \alpha^k \wedge \sum_{i=1}^{k} (-1)^{i+1} \alpha^{k+i}(v)\,\alpha^{k+1} \wedge \cdots \wedge \widehat{\alpha^{k+i}} \wedge \cdots \wedge \alpha^{k+\ell}$$

$$\text{(by Proposition 20.7)}$$

$$= (\iota_v \beta) \wedge \gamma + (-1)^k \beta \wedge \iota_v \gamma. \quad \square$$

</details>
</div>

Interior multiplication on a manifold is defined pointwise. If $X$ is a smooth vector field on $M$ and $\omega \in \Omega^k(M)$, then $\iota_X\,\omega$ is the $(k-1)$-form defined by $(\iota_X\,\omega)_p = \iota_{X_p}\,\omega_p$ for all $p \in M$. The form $\iota_X\,\omega$ on $M$ is smooth because for any smooth vector fields $X_2, \dots, X_k$ on $M$,

$$(\iota_X\,\omega)(X_2, \dots, X_k) = \omega(X, X_2, \dots, X_k)$$

is a smooth function on $M$ (Proposition 18.7(iii)$\Rightarrow$(i)). Of course, $\iota_X\,\omega = \omega(X)$ for a 1-form $\omega$ and $\iota_X f = 0$ for a function $f$ on $M$. By the properties of interior multiplication at each point $p \in M$ (Proposition 20.8), the map $\iota_X \colon \Omega^*(M) \to \Omega^*(M)$ is an antiderivation of degree $-1$ such that $\iota_X \circ \iota_X = 0$.

Let $\mathcal{F}$ be the ring $C^\infty(M)$ of $C^\infty$ functions on the manifold $M$. Because $\iota_X\,\omega$ is a point operator — that is, its value at $p$ depends only on $X_p$ and $\omega_p$ — it is $\mathcal{F}$-linear in either argument. This means that $\iota_X\,\omega$ is additive in each argument and moreover, for any $f \in \mathcal{F}$,

1. $\iota_{fX}\,\omega = f\,\iota_X\,\omega$;
2. $\iota_X(f\omega) = f\,\iota_X\,\omega$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 20.9</span><span class="math-callout__name">(Interior multiplication on $\mathbb{R}^2$)</span></p>

Let $X = x\,\partial/\partial x + y\,\partial/\partial y$ be the radial vector field and $\alpha = dx \wedge dy$ the area 2-form on the plane $\mathbb{R}^2$. Compute the contraction $\iota_X\,\alpha$.

**Solution.** We first compute $\iota_X\,dx$ and $\iota_X\,dy$:

$$\iota_X\,dx = dx\!\left(x\frac{\partial}{\partial x} + y\frac{\partial}{\partial y}\right) = x,$$

$$\iota_X\,dy = dy\!\left(x\frac{\partial}{\partial x} + y\frac{\partial}{\partial y}\right) = y.$$

By the antiderivation property of $\iota_X$,

$$\iota_X\,\alpha = \iota_X(dx \wedge dy) = (\iota_X\,dx)\,dy - dx\,(\iota_X\,dy) = x\,dy - y\,dx,$$

which restricts to the nowhere-vanishing 1-form $\omega$ on the circle $S^1$ in Example 17.15.

</div>

### 20.5 Properties of the Lie Derivative

In this section we state and prove several basic properties of the Lie derivative. We also relate the Lie derivative to two other intrinsic operators on differential forms on a manifold: the exterior derivative and interior multiplication. The interplay of these three operators results in some surprising formulas.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 20.10</span></p>

Assume $X$ to be a $C^\infty$ vector field on a manifold $M$.

**(i)** The Lie derivative $\mathcal{L}_X \colon \Omega^*(M) \to \Omega^*(M)$ is a derivation: it is an $\mathbb{R}$-linear map and if $\omega \in \Omega^k(M)$ and $\tau \in \Omega^\ell(M)$, then

$$\mathcal{L}_X(\omega \wedge \tau) = (\mathcal{L}_X\,\omega) \wedge \tau + \omega \wedge (\mathcal{L}_X\,\tau).$$

**(ii)** The Lie derivative $\mathcal{L}_X$ commutes with the exterior derivative $d$.

**(iii)** (Cartan homotopy formula) $\mathcal{L}_X = d\iota_X + \iota_X d$.

**(iv)** ("Product" formula) For $\omega \in \Omega^k(M)$ and $Y_1, \dots, Y_k \in \mathfrak{X}(M)$,

$$\mathcal{L}_X\!\left(\omega(Y_1, \dots, Y_k)\right) = (\mathcal{L}_X\,\omega)(Y_1, \dots, Y_k) + \sum_{i=1}^{k} \omega(Y_1, \dots, \mathcal{L}_X Y_i, \dots, Y_k).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**(i)** Since the Lie derivative $\mathcal{L}_X$ is $d/dt$ of a vector-valued function of $t$, the derivation property of $\mathcal{L}_X$ is really just the product rule for $d/dt$ (Proposition 20.1). More precisely,

$$(\mathcal{L}_X(\omega \wedge \tau))_p = \frac{d}{dt}\bigg\vert_{t=0} (\varphi_t^*(\omega \wedge \tau))_p = \frac{d}{dt}\bigg\vert_{t=0} (\varphi_t^*\omega)_p \wedge (\varphi_t^*\tau)_p$$

$$= \left(\frac{d}{dt}\bigg\vert_{t=0} (\varphi_t^*\omega)_p\right) \wedge \tau_p + \omega_p \wedge \frac{d}{dt}\bigg\vert_{t=0} (\varphi_t^*\tau)_p$$

$$= (\mathcal{L}_X\,\omega)_p \wedge \tau_p + \omega_p \wedge (\mathcal{L}_X\,\tau)_p.$$

**(ii)**

$$\mathcal{L}_X\,d\omega = \frac{d}{dt}\bigg\vert_{t=0} \varphi_t^*\,d\omega = \frac{d}{dt}\bigg\vert_{t=0} d\varphi_t^*\omega = d\!\left(\frac{d}{dt}\bigg\vert_{t=0} \varphi_t^*\omega\right) = d\mathcal{L}_X\,\omega.$$

Here we used that $d$ commutes with pullback, and that $d/dt\vert_{t=0}$ commutes with $d$ (Proposition 20.2).

**(iii)** We make two observations that reduce the problem to a simple case. First, for any $\omega \in \Omega^k(M)$, to prove the equality $\mathcal{L}_X\,\omega = (d\iota_X + \iota_X d)\omega$ it suffices to check it at any point $p$, which is a local problem. In a coordinate neighborhood $(U, x^1, \dots, x^n)$ about $p$, we may assume by linearity that $\omega$ is a wedge product $\omega = f\,dx^{i_1} \wedge \cdots \wedge dx^{i_k}$.

Second, on the left-hand side of the Cartan homotopy formula, by (i) and (ii), $\mathcal{L}_X$ is a derivation that commutes with $d$. On the right-hand side, since $d$ and $\iota_X$ are antiderivations, $d\iota_X + \iota_X d$ is a derivation by Problem 4.7. It clearly commutes with $d$. Thus, both sides of the Cartan homotopy formula are derivations that commute with $d$. Consequently, if the formula holds for two differential forms $\omega$ and $\tau$, then it holds for the wedge product $\omega \wedge \tau$ as well as for $d\omega$. These observations reduce the verification of (iii) to checking

$$\mathcal{L}_X f = (d\iota_X + \iota_X d)f \quad \text{for } f \in C^\infty(U).$$

This is quite easy:

$$(d\iota_X + \iota_X d)f = \iota_X\,df = (df)(X) = Xf = \mathcal{L}_X f \quad \text{(Proposition 20.6).}$$

**(iv)** In Theorem 20.10(iv), $\mathcal{L}_X(\omega(Y_1, \dots, Y_k)) = X(\omega(Y_1, \dots, Y_k))$ by Proposition 20.6 and $\mathcal{L}_X Y_i = [X, Y_i]$ by Theorem 20.4. $\square$

</details>
</div>

*Remark.* Unlike interior multiplication, the Lie derivative $\mathcal{L}_X\,\omega$ is not $\mathcal{F}$-linear in either argument. By the derivation property of the Lie derivative (Theorem 20.10(i)),

$$\mathcal{L}_X(f\omega) = (\mathcal{L}_X f)\omega + f\mathcal{L}_X\,\omega = (Xf)\omega + f\mathcal{L}_X\,\omega.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 20.11</span><span class="math-callout__name">(The Lie derivative on a circle)</span></p>

Let $\omega$ be the 1-form $-y\,dx + x\,dy$ and let $X$ be the tangent vector field $-y\,\partial/\partial x + x\,\partial/\partial y$ on the unit circle $S^1$ from Example 17.15. Compute the Lie derivative $\mathcal{L}_X\,\omega$.

**Solution.** By Proposition 20.6,

$$\mathcal{L}_X(x) = Xx = \left(-y\frac{\partial}{\partial x} + x\frac{\partial}{\partial y}\right)x = -y,$$

$$\mathcal{L}_X(y) = Xy = \left(-y\frac{\partial}{\partial x} + x\frac{\partial}{\partial y}\right)y = x.$$

Next we compute $\mathcal{L}_X(-y\,dx)$:

$$\mathcal{L}_X(-y\,dx) = -(\mathcal{L}_X y)\,dx - y\,\mathcal{L}_X\,dx \quad (\mathcal{L}_X \text{ is a derivation})$$

$$= -(\mathcal{L}_X y)\,dx - y\,d\mathcal{L}_X x \quad (\mathcal{L}_X \text{ commutes with } d)$$

$$= -x\,dx + y\,dy.$$

Similarly, $\mathcal{L}_X(x\,dy) = -y\,dy + x\,dx$. Hence, $\mathcal{L}_X\,\omega = \mathcal{L}_X(-y\,dx + x\,dy) = 0$.

</div>

### 20.6 Global Formulas for the Lie and Exterior Derivatives

The definition of the Lie derivative $\mathcal{L}_X\,\omega$ is local, since it makes sense only in a neighborhood of a point. The product formula in Theorem 20.10(iv), however, gives a global formula for the Lie derivative.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 20.12</span><span class="math-callout__name">(Global formula for the Lie derivative)</span></p>

For a smooth $k$-form $\omega$ and smooth vector fields $X, Y_1, \dots, Y_k$ on a manifold $M$,

$$(\mathcal{L}_X\,\omega)(Y_1, \dots, Y_k) = X\!\left(\omega(Y_1, \dots, Y_k)\right) - \sum_{i=1}^{k} \omega(Y_1, \dots, [X, Y_i], \dots, Y_k).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

In Theorem 20.10(iv), $\mathcal{L}_X(\omega(Y_1, \dots, Y_k)) = X(\omega(Y_1, \dots, Y_k))$ by Proposition 20.6 and $\mathcal{L}_X Y_i = [X, Y_i]$ by Theorem 20.4. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 20.13</span></p>

If $\omega$ is a $C^\infty$ 1-form and $X$ and $Y$ are $C^\infty$ vector fields on a manifold $M$, then

$$d\omega(X, Y) = X\omega(Y) - Y\omega(X) - \omega([X, Y]).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

It is enough to check the formula in a chart $(U, x^1, \dots, x^n)$, so we may assume $\omega = \sum a_i\,dx^i$. Since both sides of the equation are $\mathbb{R}$-linear in $\omega$, we may further assume that $\omega = f\,dg$, where $f, g \in C^\infty(U)$. In this case, $d\omega = d(f\,dg) = df \wedge dg$ and

$$d\omega(X, Y) = df(X)\,dg(Y) - df(Y)\,dg(X) = (Xf)Yg - (Yf)Xg,$$

$$X\omega(Y) = X(f\,dg(Y)) = X(fYg) = (Xf)Yg + fXYg,$$

$$Y\omega(X) = Y(f\,dg(X)) = Y(fXg) = (Yf)Xg + fYXg,$$

$$\omega([X, Y]) = f\,dg([X, Y]) = f(XY - YX)g.$$

It follows that

$$X\omega(Y) - Y\omega(X) - \omega([X, Y]) = (Xf)Yg - (Yf)Xg = d\omega(X, Y). \quad \square$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 20.14</span><span class="math-callout__name">(Global formula for the exterior derivative)</span></p>

Assume $k \ge 1$. For a smooth $k$-form $\omega$ and smooth vector fields $Y_0, Y_1, \dots, Y_k$ on a manifold $M$,

$$(d\omega)(Y_0, Y_1, \dots, Y_k) = \sum_{i=0}^{k} (-1)^i Y_i\,\omega(Y_0, \dots, \widehat{Y}_i, \dots, Y_k) + \sum_{0 \le i < j \le k} (-1)^{i+j} \omega([Y_i, Y_j], Y_0, \dots, \widehat{Y}_i, \dots, \widehat{Y}_j, \dots, Y_k).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

When $k = 1$, the formula is proven in Proposition 20.13.

Assuming the formula for forms of degree $k - 1$, we can prove it by induction for a form $\omega$ of degree $k$. By the definition of $\iota_{Y_0}$ and Cartan's homotopy formula (Theorem 20.10(iii)),

$$(d\omega)(Y_0, Y_1, \dots, Y_k) = (\iota_{Y_0}\,d\omega)(Y_1, \dots, Y_k)$$

$$= (\mathcal{L}_{Y_0}\,\omega)(Y_1, \dots, Y_k) - (d\iota_{Y_0}\,\omega)(Y_1, \dots, Y_k).$$

The first term of this expression can be computed using the global formula for the Lie derivative $\mathcal{L}_{Y_0}\,\omega$, while the second term can be computed using the global formula for $d$ of a form of degree $k - 1$. We leave it as an exercise (Problem 20.6). $\square$

</details>
</div>

# Chapter 6 — Integration

On a manifold one integrates not functions as in calculus on $\mathbb{R}^n$ but differential forms. There are actually two theories of integration on manifolds: one in which the integration is over a submanifold, and the other in which the integration is over what is called a *singular chain*. For integration over a manifold to be well defined, the manifold needs to be oriented.

## §21 Orientations

It is a familiar fact from vector calculus that line and surface integrals depend on the orientation of the curve or surface over which the integration takes place: reversing the orientation changes the sign of the integral. The goal of this section is to define orientation for $n$-dimensional manifolds and to investigate various equivalent characterizations of orientation.

An orientation of a finite-dimensional real vector space is simply an equivalence class of ordered bases, two ordered bases being equivalent if and only if their transition matrix has positive determinant. By its alternating nature, a multicovector of top degree turns out to represent perfectly an orientation of a vector space.

An orientation on a manifold is a choice of an orientation for each tangent space satisfying a continuity condition. Globalizing $n$-covectors over a manifold, we obtain differential $n$-forms. An orientation on an $n$-manifold can also be given by an equivalence class of $C^\infty$ nowhere-vanishing $n$-forms, two such forms being equivalent if and only if one is a multiple of the other by a positive function. Finally, a third way to represent an orientation on a manifold is through an *oriented atlas*, an atlas in which any two overlapping charts are related by a transition function with everywhere positive Jacobian determinant.

### 21.1 Orientations of a Vector Space

On $\mathbb{R}^1$ an orientation is one of two directions. On $\mathbb{R}^2$ an orientation is either counterclockwise or clockwise. On $\mathbb{R}^3$ an orientation is either right-handed or left-handed. The right-handed orientation of $\mathbb{R}^3$ is the choice of a Cartesian coordinate system such that if you hold out your right hand with the index finger curling from $e_1$ in the $x$-axis to $e_2$ in the $y$-axis, then your thumb points in the direction of $e_3$ in the $z$-axis.

How should one define an orientation for $\mathbb{R}^4$, $\mathbb{R}^5$, and beyond? Let $e_1, \dots, e_n$ be the standard basis for $\mathbb{R}^n$. For $\mathbb{R}^1$ an orientation could be given by either $e_1$ or $-e_1$. For $\mathbb{R}^2$ the counterclockwise orientation is $(e_1, e_2)$, while the clockwise orientation is $(e_2, e_1)$. For $\mathbb{R}^3$ the right-handed orientation is $(e_1, e_2, e_3)$, and the left-handed orientation is $(e_2, e_1, e_3)$.

For any two ordered bases $(u_1, u_2)$ and $(v_1, v_2)$ for $\mathbb{R}^2$, there is a unique nonsingular $2 \times 2$ matrix $A = [a^i_j]$ such that

$$u_j = \sum_{i=1}^{2} v_i a^i_j, \quad j = 1, 2,$$

called the **change-of-basis matrix** from $(v_1, v_2)$ to $(u_1, u_2)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Equivalent Ordered Bases and Orientation)</span></p>

Two ordered bases $u = [u_1 \cdots u_n]$ and $v = [v_1 \cdots v_n]$ of a vector space $V$ are said to be **equivalent**, written $u \sim v$, if $u = vA$ for an $n \times n$ matrix $A$ with positive determinant. An **orientation** of $V$ is an equivalence class of ordered bases. Any finite-dimensional vector space has two orientations. If $\mu$ is an orientation of a finite-dimensional vector space $V$, we denote the other orientation by $-\mu$ and call it the **opposite** of the orientation $\mu$.

</div>

**Notation.** A basis for a vector space is normally written $v_1, \dots, v_n$, without parentheses, brackets, or braces. If it is an *ordered* basis, then we enclose it in parentheses: $(v_1, \dots, v_n)$. In matrix notation, we also write an ordered basis as a row vector $[v_1 \cdots v_n]$. An orientation is an equivalence class of ordered bases, so the notation is $[(v_1, \dots, v_n)]$, where the brackets now stand for equivalence class.

The zero-dimensional vector space $\lbrace 0 \rbrace$ is a special case because it does not have a basis. We define an orientation on $\lbrace 0 \rbrace$ to be one of the two signs $+$ and $-$.

### 21.2 Orientations and $n$-Covectors

Instead of using an ordered basis, we can also use an $n$-covector to specify an orientation of an $n$-dimensional vector space $V$. This approach to orientations is based on the fact that the space $\bigwedge^n(V^\vee)$ of $n$-covectors on $V$ is one-dimensional.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 21.1</span></p>

Let $u_1, \dots, u_n$ and $v_1, \dots, v_n$ be vectors in a vector space $V$. Suppose

$$u_j = \sum_{i=1}^n v_i a^i_j, \quad j = 1, \dots, n,$$

for a matrix $A = [a^i_j]$ of real numbers. If $\beta$ is an $n$-covector on $V$, then

$$\beta(u_1, \dots, u_n) = (\det A)\,\beta(v_1, \dots, v_n).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By hypothesis, $u_j = \sum_i v_i a^i_j$. Since $\beta$ is $n$-linear,

$$\beta(u_1, \dots, u_n) = \beta\!\left(\sum v_{i_1} a^{i_1}_1, \dots, \sum v_{i_n} a^{i_n}_n\right) = \sum a^{i_1}_1 \cdots a^{i_n}_n\,\beta(v_{i_1}, \dots, v_{i_n}).$$

For $\beta(v_{i_1}, \dots, v_{i_n})$ to be nonzero, the subscripts $i_1, \dots, i_n$ must all be distinct. An ordered $n$-tuple $I = (i_1, \dots, i_n)$ with distinct components corresponds to a permutation $\sigma_I$ of $1, \dots, n$ with $\sigma_I(j) = i_j$ for $j = 1, \dots, n$. Since $\beta$ is an alternating $n$-tensor,

$$\beta(v_{i_1}, \dots, v_{i_n}) = (\operatorname{sgn} \sigma_I)\,\beta(v_1, \dots, v_n).$$

Thus,

$$\beta(u_1, \dots, u_n) = \sum_{\sigma_I \in S_n} (\operatorname{sgn} \sigma_I)\,a^{i_1}_1 \cdots a^{i_n}_n\,\beta(v_1, \dots, v_n) = (\det A)\,\beta(v_1, \dots, v_n). \quad \square$$

</details>
</div>

As a corollary, if $u_1, \dots, u_n$ and $v_1, \dots, v_n$ are ordered bases of a vector space $V$, then

$$\beta(u_1, \dots, u_n) \text{ and } \beta(v_1, \dots, v_n) \text{ have the same sign}$$

$$\iff \quad \det A > 0$$

$$\iff \quad u_1, \dots, u_n \text{ and } v_1, \dots, v_n \text{ are equivalent ordered bases.}$$

We say that the $n$-covector $\beta$ **determines** or **specifies** the orientation $(v_1, \dots, v_n)$ if $\beta(v_1, \dots, v_n) > 0$. Two $n$-covectors $\beta$ and $\beta'$ on $V$ determine the same orientation if and only if $\beta = a\beta'$ for some positive real number $a$. We define an equivalence relation on the nonzero $n$-covectors on the $n$-dimensional vector space $V$ by setting

$$\beta \sim \beta' \quad \iff \quad \beta = a\beta' \text{ for some } a > 0.$$

Thus, in addition to an equivalence class of ordered bases, an orientation of $V$ is also given by an equivalence class of nonzero $n$-covectors on $V$.

A linear isomorphism $\bigwedge^n(V^\vee) \simeq \mathbb{R}$ identifies the set of nonzero $n$-covectors on $V$ with $\mathbb{R} - \lbrace 0 \rbrace$, which has two connected components. Two nonzero $n$-covectors $\beta$ and $\beta'$ on $V$ are in the same component if and only if $\beta = a\beta'$ for some real number $a > 0$. Thus, each connected component of $\bigwedge^n(V^\vee) - \lbrace 0 \rbrace$ determines an orientation of $V$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Let $e_1, e_2$ be the standard basis for $\mathbb{R}^2$ and $\alpha^1, \alpha^2$ its dual basis. Then the 2-covector $\alpha^1 \wedge \alpha^2$ determines the counterclockwise orientation of $\mathbb{R}^2$, since

$$(\alpha^1 \wedge \alpha^2)(e_1, e_2) = 1 > 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Let $\partial/\partial x|_p, \partial/\partial y|_p$ be the standard basis for the tangent space $T_p(\mathbb{R}^2)$, and $(dx)_p, (dy)_p$ its dual basis. Then $(dx \wedge dy)_p$ determines the counterclockwise orientation of $T_p(\mathbb{R}^2)$.

</div>

### 21.3 Orientations on a Manifold

Recall that every vector space of dimension $n$ has two orientations, corresponding to the two equivalence classes of ordered bases or the two equivalence classes of nonzero $n$-covectors. To orient a manifold $M$, we orient the tangent space at each point in $M$, but of course this has to be done in a "coherent" way so that the orientation does not change abruptly anywhere.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Frame, Global and Local Frames)</span></p>

A **frame** on an open set $U \subset M$ is an $n$-tuple $(X_1, \dots, X_n)$ of possibly discontinuous vector fields on $U$ such that at every point $p \in U$, the $n$-tuple $(X_{1,p}, \dots, X_{n,p})$ of vectors is an ordered basis for the tangent space $T_pM$. A **global frame** is a frame defined on the entire manifold $M$, while a **local frame** about $p \in M$ is a frame defined on some neighborhood of $p$.

</div>

We introduce an equivalence relation on frames on $U$:

$$(X_1, \dots, X_n) \sim (Y_1, \dots, Y_n) \quad \iff \quad (X_{1,p}, \dots, X_{n,p}) \sim (Y_{1,p}, \dots, Y_{n,p}) \text{ for all } p \in U.$$

In other words, if $Y_j = \sum_i a^i_j X_i$, then two frames $(X_1, \dots, X_n)$ and $(Y_1, \dots, Y_n)$ are equivalent if and only if the change-of-basis matrix $A = [a^i_j]$ has positive determinant at every point in $U$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pointwise and Continuous Orientation)</span></p>

A **pointwise orientation** on a manifold $M$ assigns to each $p \in M$ an orientation $\mu_p$ of the tangent space $T_pM$. In terms of frames, a pointwise orientation on $M$ is simply an equivalence class of possibly discontinuous frames on $M$.

A pointwise orientation $\mu$ on $M$ is said to be **continuous** at $p \in M$ if $p$ has a neighborhood $U$ on which $\mu$ is represented by a **continuous frame**; i.e., there exist continuous vector fields $Y_1, \dots, Y_n$ on $U$ such that $\mu_q = [(Y_{1,q}, \dots, Y_{n,q})]$ for all $q \in U$.

The pointwise orientation $\mu$ is **continuous on $M$** if it is continuous at every point $p \in M$. Note that a continuous pointwise orientation need not be represented by a continuous global frame; it suffices that it be locally representable by a continuous local frame.

A continuous pointwise orientation on $M$ is called an **orientation** on $M$. A manifold is said to be **orientable** if it has an orientation. A manifold together with an orientation is said to be **oriented**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

The Euclidean space $\mathbb{R}^n$ is orientable with orientation given by the continuous global frame $(\partial/\partial r^1, \dots, \partial/\partial r^n)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 21.2</span><span class="math-callout__name">(The open Möbius band)</span></p>

Let $R$ be the rectangle

$$R = \lbrace (x, y) \in \mathbb{R}^2 \mid 0 \le x \le 1, \; -1 < y < 1 \rbrace.$$

The open Möbius band $M$ is the quotient of the rectangle $R$ by the equivalence relation generated by

$$(0, y) \sim (1, -y).$$

Suppose the Möbius band $M$ is orientable. An orientation on $M$ restricts to an orientation on $U$. For the sake of definiteness, we first assume the orientation on $U$ to be given by $e_1, e_2$. By continuity the orientations at the points $(0, 0)$ and $(1, 0)$ are also given by $e_1, e_2$. But under the identification $(0, y) \sim (1, -y)$, the ordered basis $e_1, e_2$ at $(1, 0)$ maps to $e_1, -e_2$ at $(0, 0)$. Thus, at $(0, 0)$ the orientation has to be given by both $e_1, e_2$ and $e_1, -e_2$, a contradiction. Assuming the orientation to be given by $e_2, e_1$ also leads to a contradiction. This proves that the Möbius band is not orientable.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 21.3</span></p>

A connected orientable manifold $M$ has exactly two orientations.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $\mu$ and $\nu$ be two orientations on $M$. At any point $p \in M$, $\mu_p$ and $\nu_p$ are orientations of $T_pM$. They either are the same or are opposite orientations. Define a function $f \colon M \to \lbrace \pm 1 \rbrace$ by

$$f(p) = \begin{cases} 1 & \text{if } \mu_p = \nu_p, \\ -1 & \text{if } \mu_p = -\nu_p. \end{cases}$$

Fix a point $p \in M$. By continuity, there exists a connected neighborhood $U$ of $p$ on which $\mu = [(X_1, \dots, X_n)]$ and $\nu = [(Y_1, \dots, Y_n)]$ for some continuous vector fields $X_i$ and $Y_j$ on $U$. Then there is a matrix-valued function $A = [a^i_j] \colon U \to \mathrm{GL}(n, \mathbb{R})$ such that $Y_j = \sum_i a^i_j X_i$. The entries $a^i_j$ are continuous, so $\det A \colon U \to \mathbb{R}^\times$ is continuous also. By the intermediate value theorem, the continuous nowhere-vanishing function $\det A$ on the connected set $U$ is everywhere positive or everywhere negative. Hence, $\mu = \nu$ or $\mu = -\nu$ on $U$. This proves that $f \colon M \to \lbrace \pm 1 \rbrace$ is locally constant. Since a locally constant function on a connected set is constant, $\mu = \nu$ or $\mu = -\nu$ on $M$. $\square$

</details>
</div>

### 21.4 Orientations and Differential Forms

While the definition of an orientation on a manifold as a continuous pointwise orientation is geometrically intuitive, in practice it is easier to manipulate the nowhere-vanishing top forms that specify a pointwise orientation. In this section we show that the continuity condition on pointwise orientations translates to a $C^\infty$ condition on nowhere-vanishing top forms.

If $f$ is a real-valued function on a set $M$, we use the notation $f > 0$ to mean that $f$ is everywhere positive on $M$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 21.4</span></p>

A pointwise orientation $[(X_1, \dots, X_n)]$ on a manifold $M$ is continuous if and only if each point $p \in M$ has a coordinate neighborhood $(U, x^1, \dots, x^n)$ on which the function $(dx^1 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n)$ is everywhere positive.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$(\Rightarrow)$ Assume that the pointwise orientation $\mu = [(X_1, \dots, X_n)]$ on $M$ is continuous. This does not mean that the global frame $(X_1, \dots, X_n)$ is continuous. What it means is that every point $p \in M$ has a neighborhood $W$ on which $\mu$ is represented by a continuous frame $(Y_1, \dots, Y_n)$. Choose a connected coordinate neighborhood $(U, x^1, \dots, x^n)$ of $p$ contained in $W$ and let $\partial_i = \partial/\partial x^i$. Then $Y_j = \sum_i b^i_j \partial_i$ for a continuous matrix function $[b^i_j] \colon U \to \mathrm{GL}(n, \mathbb{R})$, the change-of-basis matrix at each point. By Lemma 21.1,

$$(dx^1 \wedge \cdots \wedge dx^n)(Y_1, \dots, Y_n) = (\det[b^i_j])(dx^1 \wedge \cdots \wedge dx^n)(\partial_1, \dots, \partial_n) = \det[b^i_j],$$

which is never zero, because $[b^i_j]$ is nonsingular. As a continuous nowhere-vanishing real-valued function on a connected set, $(dx^1 \wedge \cdots \wedge dx^n)(Y_1, \dots, Y_n)$ is everywhere positive or everywhere negative on $U$. If it is negative, then by setting $\tilde{x}^1 = -x^1$, we have on the chart $(U, \tilde{x}^1, x^2, \dots, x^n)$ that

$$(d\tilde{x}^1 \wedge dx^2 \wedge \cdots \wedge dx^n)(Y_1, \dots, Y_n) > 0.$$

Renaming $\tilde{x}^1$ as $x^1$, we may assume that $(dx^1 \wedge \cdots \wedge dx^n)(Y_1, \dots, Y_n)$ is always positive. Since $\mu = [(X_1, \dots, X_n)] = [(Y_1, \dots, Y_n)]$ on $U$, the change-of-basis matrix $C = [c^i_j]$ such that $X_j = \sum_i c^i_j Y_i$ has positive determinant. By Lemma 21.1 again,

$$(dx^1 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n) = (\det C)(dx^1 \wedge \cdots \wedge dx^n)(Y_1, \dots, Y_n) > 0.$$

$(\Leftarrow)$ Suppose each point $p \in M$ has a coordinate neighborhood $(U, x^1, \dots, x^n)$ on which $(dx^1 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n) > 0$. Write $X_j = \sum_i a^i_j \partial_i$ on $U$. By Lemma 21.1,

$$(dx^1 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n) = \det[a^i_j] > 0.$$

Therefore, on $U$, $[(X_1, \dots, X_n)] = [(\partial_1, \dots, \partial_n)]$, which proves that the pointwise orientation $\mu$ is continuous at $p$. Since $p$ was arbitrary, $\mu$ is continuous on $M$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21.5</span></p>

A manifold $M$ of dimension $n$ is orientable if and only if there exists a $C^\infty$ nowhere-vanishing $n$-form on $M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$(\Rightarrow)$ Suppose $[(X_1, \dots, X_n)]$ is an orientation on $M$. By Lemma 21.4, each point $p$ has a coordinate neighborhood $(U, x^1, \dots, x^n)$ on which

$$(dx^1 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n) > 0.$$

Let $\lbrace (U_\alpha, x^1_\alpha, \dots, x^n_\alpha) \rbrace$ be a collection of these charts that covers $M$, and let $\lbrace \rho_\alpha \rbrace$ be a $C^\infty$ partition of unity subordinate to the open cover $\lbrace U_\alpha \rbrace$. Being a locally finite sum, the $n$-form $\omega = \sum_\alpha \rho_\alpha\,dx^1_\alpha \wedge \cdots \wedge dx^n_\alpha$ is well defined and $C^\infty$ on $M$. Fix $p \in M$. Since $\rho_\alpha(p) \ge 0$ for all $\alpha$ and $\rho_\alpha(p) > 0$ for at least one $\alpha$, by the orientation condition,

$$\omega_p(X_{1,p}, \dots, X_{n,p}) = \sum_\alpha \rho_\alpha(p)(dx^1_\alpha \wedge \cdots \wedge dx^n_\alpha)_p(X_{1,p}, \dots, X_{n,p}) > 0.$$

Therefore, $\omega$ is a $C^\infty$ nowhere-vanishing $n$-form on $M$.

$(\Leftarrow)$ Suppose $\omega$ is a $C^\infty$ nowhere-vanishing $n$-form on $M$. At each point $p \in M$, choose an ordered basis $(X_{1,p}, \dots, X_{n,p})$ for $T_pM$ such that $\omega_p(X_{1,p}, \dots, X_{n,p}) > 0$. Fix $p \in M$ and let $(U, x^1, \dots, x^n)$ be a connected coordinate neighborhood of $p$. On $U$, $\omega = f\,dx^1 \wedge \cdots \wedge dx^n$ for a $C^\infty$ nowhere-vanishing function $f$. Being continuous and nowhere vanishing on a connected set, $f$ is everywhere positive or everywhere negative on $U$. If $f > 0$, then on the chart $(U, x^1, \dots, x^n)$,

$$(dx^1 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n) > 0.$$

If $f < 0$, then on the chart $(U, -x^1, x^2, \dots, x^n)$,

$$(d(-x^1) \wedge dx^2 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n) > 0.$$

In either case, by Lemma 21.4, $\mu = [(X_1, \dots, X_n)]$ is a continuous pointwise orientation on $M$. $\square$

</details>
</div>

If $\omega$ and $\omega'$ are two nowhere-vanishing $C^\infty$ $n$-forms on a manifold $M$ of dimension $n$, then $\omega = f\omega'$ for some nowhere-vanishing function $f$ on $M$. On a *connected* manifold $M$, such a function $f$ is either everywhere positive or everywhere negative. In this way the nowhere-vanishing $C^\infty$ $n$-forms on a connected orientable manifold $M$ are partitioned into two equivalence classes by the equivalence relation

$$\omega \sim \omega' \quad \iff \quad \omega = f\omega' \text{ with } f > 0.$$

To each orientation $\mu = [(X_1, \dots, X_n)]$ on a connected orientable manifold $M$, we associate the equivalence class of a $C^\infty$ nowhere-vanishing $n$-form $\omega$ on $M$ such that $\omega(X_1, \dots, X_n) > 0$. (Such an $\omega$ exists by the proof of Theorem 21.5.) If $\mu \mapsto [\omega]$, then $-\mu \mapsto [-\omega]$. On a connected orientable manifold, this sets up a one-to-one correspondence

$$\lbrace \text{orientations on } M \rbrace \quad \longleftrightarrow \quad \left\lbrace \begin{array}{c} \text{equivalence classes of} \\ C^\infty \text{ nowhere-vanishing} \\ n\text{-forms on } M \end{array} \right\rbrace,$$

each side being a set of two elements.

If $\omega$ is a $C^\infty$ nowhere-vanishing $n$-form such that $\omega(X_1, \dots, X_n) > 0$, we say that $\omega$ **determines** or **specifies** the orientation $[(X_1, \dots, X_n)]$ and we call $\omega$ an **orientation form** on $M$. An oriented manifold can be described by a pair $(M, [\omega])$, where $[\omega]$ is the equivalence class of an orientation form on $M$. For example, unless otherwise specified, $\mathbb{R}^n$ is oriented by $dx^1 \wedge \cdots \wedge dx^n$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21.7</span><span class="math-callout__name">(Orientations on zero-dimensional manifolds)</span></p>

A connected manifold of dimension 0 is a point. The equivalence class of a nowhere-vanishing 0-form on a point is either $[-1]$ or $[1]$. Hence, a connected zero-dimensional manifold is always orientable. Its two orientations are specified by the two numbers $\pm 1$. A general zero-dimensional manifold $M$ is a countable discrete set of points, and an orientation on $M$ is given by a function that assigns to each point of $M$ either $1$ or $-1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 21.6</span><span class="math-callout__name">(Orientability of a regular zero set)</span></p>

By the regular level set theorem, if $0$ is a regular value of a $C^\infty$ function $f(x, y, z)$ on $\mathbb{R}^3$, then the zero set $f^{-1}(0)$ is a $C^\infty$ manifold. One can construct a nowhere-vanishing 2-form on the regular zero set of a $C^\infty$ function. It then follows from Theorem 21.5 that the regular zero set of a $C^\infty$ function on $\mathbb{R}^3$ is orientable.

As an example, the unit sphere $S^2$ in $\mathbb{R}^3$ is orientable. As another example, since an open Möbius band is not orientable, it cannot be realized as the regular zero set of a $C^\infty$ function on $\mathbb{R}^3$.

</div>

A diffeomorphism $F \colon (N, [\omega_N]) \to (M, [\omega_M])$ of oriented manifolds is said to be **orientation-preserving** if $[F^*\omega_M] = [\omega_N]$; it is **orientation-reversing** if $[F^*\omega_M] = [-\omega_N]$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 21.8</span></p>

Let $U$ and $V$ be open subsets of $\mathbb{R}^n$, both with the standard orientation inherited from $\mathbb{R}^n$. A diffeomorphism $F \colon U \to V$ is orientation-preserving if and only if the Jacobian determinant $\det[\partial F^i / \partial x^j]$ is everywhere positive on $U$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $x^1, \dots, x^n$ and $y^1, \dots, y^n$ be the standard coordinates on $U \subset \mathbb{R}^n$ and $V \subset \mathbb{R}^n$. Then

$$F^*(dy^1 \wedge \cdots \wedge dy^n) = d(F^*y^1) \wedge \cdots \wedge d(F^*y^n)$$

$$= d(y^1 \circ F) \wedge \cdots \wedge d(y^n \circ F)$$

$$= dF^1 \wedge \cdots \wedge dF^n$$

$$= \det\!\left[\frac{\partial F^i}{\partial x^j}\right] dx^1 \wedge \cdots \wedge dx^n.$$

Thus, $F$ is orientation-preserving if and only if $\det[\partial F^i / \partial x^j]$ is everywhere positive on $U$. $\square$

</details>
</div>

### 21.5 Orientations and Atlases

Using the characterization of an orientation-preserving diffeomorphism by the sign of its Jacobian determinant, we can describe orientability of manifolds in terms of atlases.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 21.9</span></p>

An atlas on $M$ is said to be **oriented** if for any two overlapping charts $(U, x^1, \dots, x^n)$ and $(V, y^1, \dots, y^n)$ of the atlas, the Jacobian determinant $\det[\partial y^i / \partial x^j]$ is everywhere positive on $U \cap V$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21.10</span></p>

A manifold $M$ is orientable if and only if it has an oriented atlas.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$(\Rightarrow)$ Let $\mu = [(X_1, \dots, X_n)]$ be an orientation on the manifold $M$. By Lemma 21.4, each point $p \in M$ has a coordinate neighborhood $(U, x^1, \dots, x^n)$ on which

$$(dx^1 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n) > 0.$$

We claim that the collection $\mathfrak{U} = \lbrace (U, x^1, \dots, x^n) \rbrace$ of these charts is an oriented atlas. If $(U, x^1, \dots, x^n)$ and $(V, y^1, \dots, y^n)$ are two overlapping charts from $\mathfrak{U}$, then on $U \cap V$,

$$(dx^1 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n) > 0 \quad \text{and} \quad (dy^1 \wedge \cdots \wedge dy^n)(X_1, \dots, X_n) > 0.$$

Since $dy^1 \wedge \cdots \wedge dy^n = (\det[\partial y^i / \partial x^j])\,dx^1 \wedge \cdots \wedge dx^n$, it follows from the two inequalities that $\det[\partial y^i / \partial x^j] > 0$ on $U \cap V$. Therefore, $\mathfrak{U}$ is an oriented atlas.

$(\Leftarrow)$ Suppose $\lbrace (U, x^1, \dots, x^n) \rbrace$ is an oriented atlas. For each $p \in (U, x^1, \dots, x^n)$, define $\mu_p$ to be the equivalence class of the ordered basis $(\partial/\partial x^1|_p, \dots, \partial/\partial x^n|_p)$ for $T_pM$. If two charts $(U, x^1, \dots, x^n)$ and $(V, y^1, \dots, y^n)$ in the oriented atlas contain $p$, then by the orientability of the atlas, $\det[\partial y^i / \partial x^j] > 0$, so that $(\partial/\partial x^1|_p, \dots, \partial/\partial x^n|_p)$ is equivalent to $(\partial/\partial y^1|_p, \dots, \partial/\partial y^n|_p)$. This proves that $\mu$ is a well-defined pointwise orientation on $M$. It is continuous because every point $p$ has a coordinate neighborhood $(U, x^1, \dots, x^n)$ on which $\mu = [(\partial/\partial x^1, \dots, \partial/\partial x^n)]$ is represented by a continuous frame. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 21.11</span><span class="math-callout__name">(Equivalent Oriented Atlases)</span></p>

Two oriented atlases $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ and $\lbrace (V_\beta, \psi_\beta) \rbrace$ on a manifold $M$ are said to be **equivalent** if the transition functions

$$\phi_\alpha \circ \psi_\beta^{-1} \colon \psi_\beta(U_\alpha \cap V_\beta) \to \phi_\alpha(U_\alpha \cap V_\beta)$$

have positive Jacobian determinant for all $\alpha, \beta$.

</div>

In the proof of Theorem 21.10, an oriented atlas $\lbrace (U, x^1, \dots, x^n) \rbrace$ on a manifold $M$ determines an orientation $U \ni p \mapsto [(\partial/\partial x^1|_p, \dots, \partial/\partial x^n|_p)]$ on $M$, and conversely, an orientation $[(X_1, \dots, X_n)]$ on $M$ gives rise to an oriented atlas $\lbrace (U, x^1, \dots, x^n) \rbrace$ on $M$ such that $(dx^1 \wedge \cdots \wedge dx^n)(X_1, \dots, X_n) > 0$ on $U$. The two induced maps

$$\left\lbrace \begin{array}{c} \text{equivalence classes of} \\ \text{oriented atlases on } M \end{array} \right\rbrace \implies \lbrace \text{orientations on } M \rbrace$$

are well defined and inverse to each other. Therefore, one can also specify an orientation on an orientable manifold by an equivalence class of oriented atlases.

For an oriented manifold $M$, we denote by $-M$ the same manifold but with the opposite orientation. If $\lbrace (U, \phi) \rbrace = \lbrace (U, x^1, x^2, \dots, x^n) \rbrace$ is an oriented atlas specifying the orientation of $M$, then an oriented atlas specifying the orientation of $-M$ is $\lbrace (U, \hat{\phi}) \rbrace = \lbrace (U, -x^1, x^2, \dots, x^n) \rbrace$.

## §22 Manifolds with Boundary

The prototype of a manifold with boundary is the *closed upper half-space*

$$\mathcal{H}^n = \lbrace (x^1, \dots, x^n) \in \mathbb{R}^n \mid x^n \ge 0 \rbrace,$$

with the subspace topology inherited from $\mathbb{R}^n$. The points $(x^1, \dots, x^n)$ in $\mathcal{H}^n$ with $x^n > 0$ are called the **interior points** of $\mathcal{H}^n$, and the points with $x^n = 0$ are called the **boundary points** of $\mathcal{H}^n$. These two sets are denoted by $(\mathcal{H}^n)^\circ$ and $\partial(\mathcal{H}^n)$, respectively.

We require that $\mathcal{H}^n$ include the boundary in order for it to serve as a model for manifolds with boundary.

If $M$ is a manifold with boundary, then its boundary $\partial M$ turns out to be a manifold of dimension one less without boundary. Moreover, an orientation on $M$ induces an orientation on $\partial M$. The choice of the induced orientation on the boundary is a matter of convention, guided by the desire to make Stokes's theorem sign-free. Of the various ways to describe the boundary orientation, two stand out for their simplicity: (1) contraction of an orientation form on $M$ with an outward-pointing vector field on $\partial M$ and (2) "outward vector first."

### 22.1 Smooth Invariance of Domain in $\mathbb{R}^n$

To discuss $C^\infty$ functions on a manifold with boundary, we need to extend the definition of a $C^\infty$ function to allow nonopen domains.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 22.1</span><span class="math-callout__name">(Smooth function on an arbitrary subset)</span></p>

Let $S \subset \mathbb{R}^n$ be an arbitrary subset. A function $f \colon S \to \mathbb{R}^m$ is **smooth at a point** $p$ in $S$ if there exist a neighborhood $U$ of $p$ in $\mathbb{R}^n$ and a $C^\infty$ function $\tilde{f} \colon U \to \mathbb{R}^m$ such that $\tilde{f} = f$ on $U \cap S$. The function is **smooth on $S$** if it is smooth at each point of $S$.

</div>

With this definition it now makes sense to speak of an arbitrary subset $S \subset \mathbb{R}^n$ being diffeomorphic to an arbitrary subset $T \subset \mathbb{R}^m$; this will be the case if and only if there are smooth maps $f \colon S \to T \subset \mathbb{R}^m$ and $g \colon T \to S \subset \mathbb{R}^n$ that are inverse to each other.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.3</span><span class="math-callout__name">(Smooth invariance of domain)</span></p>

Let $U \subset \mathbb{R}^n$ be an open subset, $S \subset \mathbb{R}^n$ an arbitrary subset, and $f \colon U \to S$ a diffeomorphism. Then $S$ is open in $\mathbb{R}^n$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $f(p)$ be an arbitrary point in $S$, with $p \in U$. Since $f \colon U \to S$ is a diffeomorphism, there are an open set $V \subset \mathbb{R}^n$ containing $S$ and a $C^\infty$ map $g \colon V \to \mathbb{R}^n$ such that $g|_S = f^{-1}$. Thus,

$$U \xrightarrow{f} V \xrightarrow{g} \mathbb{R}^n$$

satisfies $g \circ f = \mathbb{1}_U \colon U \to U \subset \mathbb{R}^n$, the identity map on $U$. By the chain rule,

$$g_{*,f(p)} \circ f_{*,p} = \mathbb{1}_{T_pU} \colon T_pU \to T_pU \simeq T_p(\mathbb{R}^n),$$

the identity on the tangent space $T_pU$. Hence, $f_{*,p}$ is injective. Since $U$ and $V$ have the same dimension, $f_{*,p} \colon T_pU \to T_{f(p)}V$ is invertible. By the inverse function theorem, $f$ is locally invertible at $p$. This means that there are open neighborhoods $U_p$ of $p$ in $U$ and $V_{f(p)}$ of $f(p)$ in $V$ such that $f \colon U_p \to V_{f(p)}$ is a diffeomorphism. It follows that

$$f(p) \in V_{f(p)} = f(U_p) \subset f(U) = S.$$

Since $V$ is open in $\mathbb{R}^n$ and $V_{f(p)}$ is open in $V$, the set $V_{f(p)}$ is open in $\mathbb{R}^n$. By the local criterion for openness, $S$ is open in $\mathbb{R}^n$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 22.4</span></p>

Let $U$ and $V$ be open subsets of the upper half-space $\mathcal{H}^n$ and $f \colon U \to V$ a diffeomorphism. Then $f$ maps interior points to interior points and boundary points to boundary points.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $p \in U$ be an interior point. Then $p$ is contained in an open ball $B$, which is actually open in $\mathbb{R}^n$ (not just in $\mathcal{H}^n$). By smooth invariance of domain, $f(B) \subset (\mathcal{H}^n)^\circ$. Since $f(p) \in f(B)$, $f(p)$ is an interior point of $\mathcal{H}^n$.

If $p$ is a boundary point in $U \cap \partial\mathcal{H}^n$, then $f^{-1}(f(p)) = p$ is a boundary point. Since $f^{-1} \colon V \to U$ is a diffeomorphism, by what has just been proven, $f(p)$ cannot be an interior point. Thus, $f(p)$ is a boundary point. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.5</span></p>

Replacing Euclidean spaces by manifolds throughout this subsection, one can prove in exactly the same way smooth invariance of domain for manifolds: if there is a diffeomorphism between an open subset $U$ of an $n$-dimensional manifold $N$ and an arbitrary subset $S$ of another $n$-dimensional manifold $M$, then $S$ is open in $M$.

</div>

### 22.2 Manifolds with Boundary

In the upper half-space $\mathcal{H}^n$ one may distinguish two kinds of open subsets, depending on whether the set is disjoint from the boundary or intersects the boundary. Charts on a manifold are homeomorphic to only the first kind of open sets.

A manifold with boundary generalizes the definition of a manifold by allowing both kinds of open sets. We say that a topological space $M$ is *locally $\mathcal{H}^n$* if every point $p \in M$ has a neighborhood $U$ homeomorphic to an open subset of $\mathcal{H}^n$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 22.6</span><span class="math-callout__name">(Topological manifold with boundary)</span></p>

A **topological $n$-manifold with boundary** is a second countable, Hausdorff topological space that is locally $\mathcal{H}^n$.

</div>

Let $M$ be a topological $n$-manifold with boundary. For $n \ge 2$, a **chart** on $M$ is defined to be a pair $(U, \phi)$ consisting of an open set $U$ in $M$ and a homeomorphism

$$\phi \colon U \to \phi(U) \subset \mathcal{H}^n$$

of $U$ with an open subset $\phi(U)$ of $\mathcal{H}^n$. When $n = 1$, we need to allow two local models: the *right half-line* $\mathcal{H}^1$ and the *left half-line*

$$\mathcal{L}^1 := \lbrace x \in \mathbb{R} \mid x \le 0 \rbrace.$$

A chart $(U, \phi)$ in dimension 1 consists of an open set $U$ in $M$ and a homeomorphism $\phi$ of $U$ with an open subset of $\mathcal{H}^1$ or $\mathcal{L}^1$. With this convention, if $(U, x^1, x^2, \dots, x^n)$ is a chart of an $n$-dimensional manifold with boundary, then so is $(U, -x^1, x^2, \dots, x^n)$ for any $n \ge 1$.

A collection $\lbrace (U, \phi) \rbrace$ of charts is a $C^\infty$ **atlas** if for any two charts $(U, \phi)$ and $(V, \psi)$, the transition map

$$\psi \circ \phi^{-1} \colon \phi(U \cap V) \to \psi(U \cap V) \subset \mathcal{H}^n$$

is a diffeomorphism. A $C^\infty$ **manifold with boundary** is a topological manifold with boundary together with a maximal $C^\infty$ atlas.

A point $p$ of $M$ is called an **interior point** if in some chart $(U, \phi)$, the point $\phi(p)$ is an interior point of $\mathcal{H}^n$. Similarly, $p$ is a **boundary point** of $M$ if $\phi(p)$ is a boundary point of $\mathcal{H}^n$. These concepts are well defined, independent of the charts, because if $(V, \psi)$ is another chart, then the diffeomorphism $\psi \circ \phi^{-1}$ maps $\phi(p)$ to $\psi(p)$, and so by Proposition 22.4, $\phi(p)$ and $\psi(p)$ are either both interior points or both boundary points. The set of boundary points of $M$ is denoted by $\partial M$.

Most of the concepts introduced for a manifold extend word for word to a manifold with boundary, the only difference being that now a chart can be either of two types and the local model is $\mathcal{H}^n$ (or $\mathcal{L}^1$). For example, a function $f \colon M \to \mathbb{R}$ is $C^\infty$ at a boundary point $p \in \partial M$ if there is a chart $(U, \phi)$ about $p$ such that $f \circ \phi^{-1}$ is $C^\infty$ at $\phi(p) \in \mathcal{H}^n$. This in turn means that $f \circ \phi^{-1}$ has a $C^\infty$ extension to a neighborhood of $\phi(p)$ in $\mathbb{R}^n$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 22.7</span><span class="math-callout__name">(Topological boundary versus manifold boundary)</span></p>

Let $A$ be the open unit disk in $\mathbb{R}^2$:

$$A = \lbrace x \in \mathbb{R}^2 \mid \|x\| < 1 \rbrace.$$

Then its topological boundary $\mathrm{bd}(A)$ in $\mathbb{R}^2$ is the unit circle, while its manifold boundary $\partial A$ is the empty set.

If $B$ is the closed unit disk in $\mathbb{R}^2$, then its topological boundary $\mathrm{bd}(B)$ and its manifold boundary $\partial B$ coincide; both are the unit circle.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 22.8</span><span class="math-callout__name">(Topological interior versus manifold interior)</span></p>

Let $S$ be the upper half-plane $\mathcal{H}^2$ and let $D$ be the subset

$$D = \lbrace (x, y) \in \mathcal{H}^2 \mid y \le 1 \rbrace.$$

The topological interior of $D$ is the set

$$\mathrm{int}(D) = \lbrace (x, y) \in \mathcal{H}^2 \mid 0 \le y < 1 \rbrace,$$

containing the $x$-axis, while the manifold interior of $D$ is the set

$$D^\circ = \lbrace (x, y) \in \mathcal{H}^2 \mid 0 < y < 1 \rbrace,$$

not containing the $x$-axis.

To indicate the dependence of the topological interior of a set $A$ on its ambient space $S$, we might denote it by $\mathrm{int}_S(A)$ instead of $\mathrm{int}(A)$. Then the topological interior $\mathrm{int}_{\mathcal{H}^2}(D)$ of $D$ in $\mathcal{H}^2$ is as above, but the topological interior $\mathrm{int}_{\mathbb{R}^2}(D)$ of $D$ in $\mathbb{R}^2$ coincides with $D^\circ$.

</div>

### 22.3 The Boundary of a Manifold with Boundary

Let $M$ be a manifold of dimension $n$ with boundary $\partial M$. If $(U, \phi)$ is a chart on $M$, we denote by $\phi' = \phi|_{U \cap \partial M}$ the restriction of the coordinate map $\phi$ to the boundary. Since $\phi$ maps boundary points to boundary points,

$$\phi' \colon U \cap \partial M \to \partial\mathcal{H}^n = \mathbb{R}^{n-1}.$$

Moreover, if $(U, \phi)$ and $(V, \psi)$ are two charts on $M$, then

$$\psi' \circ (\phi')^{-1} \colon \phi'(U \cap V \cap \partial M) \to \psi'(U \cap V \cap \partial M)$$

is $C^\infty$. Thus, an atlas $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ for $M$ induces an atlas $\lbrace (U_\alpha \cap \partial M, \phi_\alpha|_{U_\alpha \cap \partial M}) \rbrace$ for $\partial M$, making $\partial M$ into a manifold of dimension $n - 1$ without boundary.

### 22.4 Tangent Vectors, Differential Forms, and Orientations

Let $M$ be a manifold with boundary and let $p \in \partial M$. As in Subsection 2.2, two $C^\infty$ functions $f \colon U \to \mathbb{R}$ and $g \colon V \to \mathbb{R}$ defined on neighborhoods $U$ and $V$ of $p$ in $M$ are said to be *equivalent* if they agree on some neighborhood $W$ of $p$ contained in $U \cap V$. A **germ** of $C^\infty$ functions at $p$ is an equivalence class of such functions. With the usual addition, multiplication, and scalar multiplication of germs, the set $C_p^\infty(M)$ of germs of $C^\infty$ functions at $p$ is an $\mathbb{R}$-algebra. The **tangent space** $T_pM$ at $p$ is then defined to be the vector space of all point-derivations on the algebra $C_p^\infty(M)$.

For example, for $p$ in the boundary of the upper half-plane $\mathcal{H}^2$, $\partial/\partial x|_p$ and $\partial/\partial y|_p$ are both derivations on $C_p^\infty(\mathcal{H}^2)$. The tangent space $T_p(\mathcal{H}^2)$ is represented by a 2-dimensional vector space with the origin at $p$. Since $\partial/\partial y|_p$ is a tangent vector to $\mathcal{H}^2$ at $p$, its negative $-\partial/\partial y|_p$ is also a tangent vector at $p$, although there is no curve through $p$ in $\mathcal{H}^2$ with initial velocity $-\partial/\partial y|_p$.

The **cotangent space** $T_p^*M$ is defined to be the dual of the tangent space:

$$T_p^*M = \mathrm{Hom}(T_pM, \mathbb{R}).$$

**Differential $k$-forms** on $M$ are defined as before, as sections of the vector bundle $\bigwedge^k(T^*M)$. A differential $k$-form is $C^\infty$ if it is $C^\infty$ as a section of the vector bundle $\bigwedge^k(T^*M)$. For example, $dx \wedge dy$ is a $C^\infty$ 2-form on $\mathcal{H}^2$.

An **orientation** on an $n$-manifold $M$ with boundary is again a continuous pointwise orientation on $M$. The discussion in Section 21 on orientations goes through word for word for manifolds with boundary. Thus, the orientability of a manifold with boundary is equivalent to the existence of a $C^\infty$ nowhere-vanishing top form and to the existence of an oriented atlas. At one point in the proof of Lemma 21.4, it was necessary to replace the chart $(U, x^1, x^2, \dots, x^n)$ by $(U, -x^1, x^2, \dots, x^n)$. This would not have been possible for $n = 1$ if we had not allowed the left half-line $\mathcal{L}^1$ as a local model in the definition of a chart on a 1-dimensional manifold with boundary.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 22.9</span></p>

The closed interval $[0, 1]$ is a $C^\infty$ manifold with boundary. It has an atlas with two charts $(U_1, \phi_1)$ and $(U_2, \phi_2)$, where $U_1 = [0, 1[$, $\phi_1(x) = x$, and $U_2 = {]0, 1]}$, $\phi_2(x) = 1 - x$. With $d/dx$ as a continuous pointwise orientation, $[0, 1]$ is an oriented manifold with boundary. However, $\lbrace (U_1, \phi_1), (U_2, \phi_2) \rbrace$ is not an oriented atlas, because the Jacobian determinant of the transition function $(\phi_2 \circ \phi_1^{-1})(x) = 1 - x$ is negative. If we change the sign of $\phi_2$, then $\lbrace (U_1, \phi_1), (U_2, -\phi_2) \rbrace$ is an oriented atlas. Note that $-\phi_2(x) = x - 1$ maps $]0, 1]$ into the left half-line $\mathcal{L}^1 \subset \mathbb{R}$. If we had allowed only $\mathcal{H}^1$ as a local model for a 1-dimensional manifold with boundary, the closed interval $[0, 1]$ would not have an oriented atlas.

</div>

### 22.5 Outward-Pointing Vector Fields

Let $M$ be a manifold with boundary and $p \in \partial M$. We say that a tangent vector $X_p \in T_p(M)$ is **inward-pointing** if $X_p \notin T_p(\partial M)$ and there are a positive real number $\varepsilon$ and a curve $c \colon [0, \varepsilon[ \to M$ such that $c(0) = p$, $c(]0, \varepsilon[) \subset M^\circ$, and $c'(0) = X_p$. A vector $X_p \in T_p(M)$ is **outward-pointing** if $-X_p$ is inward-pointing.

For example, on the upper half-plane $\mathcal{H}^2$, the vector $\partial/\partial y|_p$ is inward-pointing and the vector $-\partial/\partial y|_p$ is outward-pointing at a point $p$ on the $x$-axis.

A **vector field along $\partial M$** is a function $X$ that assigns to each point $p$ in $\partial M$ a vector $X_p$ in the tangent space $T_pM$ (as opposed to $T_p(\partial M)$). In a coordinate neighborhood $(U, x^1, \dots, x^n)$ of $p$ in $M$, such a vector field $X$ can be written as a linear combination

$$X_q = \sum_i a^i(q) \left.\frac{\partial}{\partial x^i}\right|_q, \quad q \in \partial M.$$

The vector field $X$ along $\partial M$ is said to be **smooth** at $p \in M$ if there exists a coordinate neighborhood of $p$ for which the functions $a^i$ on $\partial M$ are $C^\infty$ at $p$; it is said to be **smooth** if it is smooth at every point $p$. In terms of local coordinates, a vector $X_p$ is outward-pointing if and only if $a^n(p) < 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 22.10</span></p>

On a manifold $M$ with boundary $\partial M$, there is a smooth outward-pointing vector field along $\partial M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Cover $\partial M$ with coordinate open sets $(U_\alpha, x^1_\alpha, \dots, x^n_\alpha)$ in $M$. On each $U_\alpha$ the vector field $X_\alpha = -\partial/\partial x^n_\alpha$ along $U_\alpha \cap \partial M$ is smooth and outward-pointing. Choose a partition of unity $\lbrace \rho_\alpha \rbrace_{\alpha \in A}$ on $\partial M$ subordinate to the open cover $\lbrace U_\alpha \cap \partial M \rbrace_{\alpha \in A}$. Then one can check that $X := \sum \rho_\alpha X_\alpha$ is a smooth outward-pointing vector field along $\partial M$. $\square$

</details>
</div>

### 22.6 Boundary Orientation

In this section we show that the boundary of an orientable manifold $M$ with boundary is an orientable manifold (without boundary, by Subsection 22.3). We will designate one of the orientations on the boundary as the boundary orientation. It is easily described in terms of an orientation form or of a pointwise orientation on $\partial M$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 22.11</span></p>

Let $M$ be an oriented $n$-manifold with boundary. If $\omega$ is an orientation form on $M$ and $X$ is a smooth outward-pointing vector field on $\partial M$, then $\iota_X \omega$ is a smooth nowhere-vanishing $(n-1)$-form on $\partial M$. Hence, $\partial M$ is orientable.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since $\omega$ and $X$ are both smooth on $\partial M$, so is the contraction $\iota_X \omega$ (Subsection 20.4). We will now prove by contradiction that $\iota_X \omega$ is nowhere-vanishing on $\partial M$. Suppose $\iota_X \omega$ vanishes at some $p \in \partial M$. This means that $(\iota_X \omega)_p(v_1, \dots, v_{n-1}) = 0$ for all $v_1, \dots, v_{n-1} \in T_p(\partial M)$. Let $e_1, \dots, e_{n-1}$ be a basis for $T_p(\partial M)$. Then $X_p, e_1, \dots, e_{n-1}$ is a basis for $T_pM$, and

$$\omega_p(X_p, e_1, \dots, e_{n-1}) = (\iota_X \omega)_p(e_1, \dots, e_{n-1}) = 0.$$

By Problem 3.9, $\omega_p \equiv 0$ on $T_pM$, a contradiction. Therefore, $\iota_X \omega$ is nowhere vanishing on $\partial M$. By Theorem 21.5, $\partial M$ is orientable. $\square$

</details>
</div>

In the notation of the preceding proposition, we define the **boundary orientation** on $\partial M$ to be the orientation with orientation form $\iota_X \omega$. For the boundary orientation to be well defined, we need to check that it is independent of the choice of the orientation form $\omega$ and of the outward-pointing vector field $X$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 22.12</span></p>

Suppose $M$ is an oriented $n$-manifold with boundary. Let $p$ be a point of the boundary $\partial M$ and let $X_p$ be an outward-pointing vector in $T_pM$. An ordered basis $(v_1, \dots, v_{n-1})$ for $T_p(\partial M)$ represents the boundary orientation at $p$ if and only if the ordered basis $(X_p, v_1, \dots, v_{n-1})$ for $T_pM$ represents the orientation on $M$ at $p$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For $p$ in $\partial M$, let $(v_1, \dots, v_{n-1})$ be an ordered basis for the tangent space $T_p(\partial M)$. Then

$$(v_1, \dots, v_{n-1}) \text{ represents the boundary orientation on } \partial M \text{ at } p$$

$$\iff \quad (\iota_{X_p} \omega_p)(v_1, \dots, v_{n-1}) > 0$$

$$\iff \quad \omega_p(X_p, v_1, \dots, v_{n-1}) > 0$$

$$\iff \quad (X_p, v_1, \dots, v_{n-1}) \text{ represents the orientation on } M \text{ at } p. \quad \square$$

</details>
</div>

To make this rule easier to remember, we summarize it under the rubric **"outward vector first."**

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 22.13</span><span class="math-callout__name">(The boundary orientation on $\partial\mathcal{H}^n$)</span></p>

An orientation form for the standard orientation on the upper half-space $\mathcal{H}^n$ is $\omega = dx^1 \wedge \cdots \wedge dx^n$. A smooth outward-pointing vector field on $\partial\mathcal{H}^n$ is $-\partial/\partial x^n$. By definition, an orientation form for the boundary orientation on $\partial\mathcal{H}^n$ is given by the contraction

$$\iota_{-\partial/\partial x^n}(\omega) = -\iota_{\partial/\partial x^n}(dx^1 \wedge \cdots \wedge dx^{n-1} \wedge dx^n)$$

$$= -(-1)^{n-1} dx^1 \wedge \cdots \wedge dx^{n-1} \wedge \iota_{\partial/\partial x^n}(dx^n)$$

$$= (-1)^n dx^1 \wedge \cdots \wedge dx^{n-1}.$$

Thus, the boundary orientation on $\partial\mathcal{H}^1 = \lbrace 0 \rbrace$ is given by $-1$, the boundary orientation on $\partial\mathcal{H}^2$, given by $dx^1$, is the usual orientation on the real line $\mathbb{R}$, and the boundary orientation on $\partial\mathcal{H}^3$, given by $-dx^1 \wedge dx^2$, is the clockwise orientation in the $(x^1, x^2)$-plane $\mathbb{R}^2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Boundary orientation on a closed interval)</span></p>

The closed interval $[a, b]$ in the real line with coordinate $x$ has a standard orientation given by the vector field $d/dx$, with orientation form $dx$. At the right endpoint $b$, an outward vector is $d/dx$. Hence, the boundary orientation at $b$ is given by $\iota_{d/dx}(dx) = +1$. Similarly, the boundary orientation at the left endpoint $a$ is given by $\iota_{-d/dx}(dx) = -1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Boundary orientation on an immersed curve)</span></p>

Suppose $c \colon [a, b] \to M$ is a $C^\infty$ immersion whose image is a 1-dimensional manifold $C$ with boundary. An orientation on $[a, b]$ induces an orientation on $C$ via the differential $c_{*,p} \colon T_p([a, b]) \to T_pC$ at each point $p \in [a, b]$. In a situation like this, we give $C$ the orientation induced from the standard orientation on $[a, b]$. The boundary orientation on the boundary of $C$ is given by $+1$ at the endpoint $c(b)$ and $-1$ at the initial point $c(a)$.

</div>

# Chapter 5 — Integration

## §23 Integration on Manifolds

In this chapter we first recall Riemann integration for a function over a closed rectangle in Euclidean space. By Lebesgue's theorem, this theory can be extended to integrals over domains of integration, bounded subsets of $\mathbb{R}^n$ whose boundary has measure zero.

The integral of an $n$-form with compact support in an open set of $\mathbb{R}^n$ is defined to be the Riemann integral of the corresponding function. Using a partition of unity, we define the integral of an $n$-form with compact support on a manifold by writing the form as a sum of forms each with compact support in a coordinate chart. We then prove the general Stokes theorem for an oriented manifold and show how it generalizes the fundamental theorem for line integrals as well as Green's theorem from calculus.

### 23.1 The Riemann Integral of a Function on $\mathbb{R}^n$

A **closed rectangle** in $\mathbb{R}^n$ is a Cartesian product $R = [a^1, b^1] \times \cdots \times [a^n, b^n]$ of closed intervals in $\mathbb{R}$, where $a^i, b^i \in \mathbb{R}$. Let $f \colon R \to \mathbb{R}$ be a bounded function defined on a closed rectangle $R$. The **volume** $\operatorname{vol}(R)$ of the closed rectangle $R$ is defined to be

$$\operatorname{vol}(R) := \prod_{i=1}^n (b_i - a_i).$$

A **partition** of the closed interval $[a, b]$ is a set of real numbers $\lbrace p_0, \dots, p_n \rbrace$ such that

$$a = p_0 < p_1 < \cdots < p_n = b.$$

A **partition** of the rectangle $R$ is a collection $P = \lbrace P_1, \dots, P_n \rbrace$, where each $P_i$ is a partition of $[a^i, b^i]$. The partition $P$ divides the rectangle $R$ into closed subrectangles, which we denote by $R_j$.

We define the **lower sum** and the **upper sum** of $f$ with respect to the partition $P$ to be

$$L(f, P) := \sum (\inf_{R_j} f) \operatorname{vol}(R_j), \quad U(f, P) := \sum (\sup_{R_j} f) \operatorname{vol}(R_j),$$

where each sum runs over all subrectangles of the partition $P$. For any partition $P$, clearly $L(f, P) \le U(f, P)$.

A partition $P' = \lbrace P'_1, \dots, P'_n \rbrace$ is a **refinement** of the partition $P = \lbrace P_1, \dots, P_n \rbrace$ if $P_i \subset P'_i$ for all $i = 1, \dots, n$. If $P'$ is a refinement of $P$, then each subrectangle $R_j$ of $P$ is subdivided into subrectangles $R'_{jk}$ of $P'$, and it is easily seen that

$$L(f, P) \le L(f, P'), \quad U(f, P') \le U(f, P).$$

Any two partitions $P$ and $P'$ of the rectangle $R$ have a common refinement $Q = \lbrace Q_1, \dots, Q_n \rbrace$ with $Q_i = P_i \cup P'_i$, and so

$$L(f, P) \le L(f, Q) \le U(f, Q) \le U(f, P').$$

It follows that the supremum of the lower sum $L(f, P)$ over all partitions $P$ of $R$ is less than or equal to the infimum of the upper sum $U(f, P)$ over all partitions $P$ of $R$. We define these two numbers to be the **lower integral** $\underline{\int}_R f$ and the **upper integral** $\overline{\int}_R f$, respectively:

$$\underline{\int}_R f := \sup_P L(f, P), \quad \overline{\int}_R f := \inf_P L(f, P).$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 23.1</span><span class="math-callout__name">(Riemann Integrable)</span></p>

Let $R$ be a closed rectangle in $\mathbb{R}^n$. A bounded function $f \colon R \to \mathbb{R}$ is said to be **Riemann integrable** if $\underline{\int}_R f = \overline{\int}_R f$; in this case, the Riemann integral of $f$ is this common value, denoted by $\int_R f(x)\,dx^1 \cdots dx^n$, where $x^1, \dots, x^n$ are the standard coordinates on $\mathbb{R}^n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

When we speak of a rectangle $[a^1, b^1] \times \cdots \times [a^n, b^n]$ in $\mathbb{R}^n$, we have already tacitly chosen $n$ coordinates axes, with coordinates $x^1, \dots, x^n$. Thus, the definition of a Riemann integral depends on the coordinates $x^1, \dots, x^n$.

</div>

If $f \colon A \subset \mathbb{R}^n \to \mathbb{R}$, then the **extension of $f$ by zero** is the function $\tilde{f} \colon \mathbb{R}^n \to \mathbb{R}$ such that

$$\tilde{f}(x) = \begin{cases} f(x) & \text{for } x \in A, \\ 0 & \text{for } x \notin A. \end{cases}$$

Now suppose $f \colon A \to \mathbb{R}$ is a bounded function on a bounded set $A$ in $\mathbb{R}^n$. Enclose $A$ in a closed rectangle $R$ and define the **Riemann integral of $f$ over $A$** to be

$$\int_A f(x)\,dx^1 \cdots dx^n = \int_R \tilde{f}(x)\,dx^1 \cdots dx^n$$

if the right-hand side exists.

The **volume** $\operatorname{vol}(A)$ of a subset $A \subset \mathbb{R}^n$ is defined to be the integral $\int_A 1\,dx^1 \cdots dx^n$ if the integral exists. This concept generalizes the volume of a closed rectangle.

### 23.2 Integrability Conditions

In this section we describe some conditions under which a function defined on an open subset of $\mathbb{R}^n$ is Riemann integrable.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 23.2</span><span class="math-callout__name">(Measure Zero)</span></p>

A set $A \subset \mathbb{R}^n$ is said to have **measure zero** if for every $\varepsilon > 0$, there is a countable cover $\lbrace R_i \rbrace_{i=1}^\infty$ of $A$ by closed rectangles $R_i$ such that $\sum_{i=1}^\infty \operatorname{vol}(R_i) < \varepsilon$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 23.3</span><span class="math-callout__name">(Lebesgue's theorem)</span></p>

A bounded function $f \colon A \to \mathbb{R}$ on a bounded subset $A \subset \mathbb{R}^n$ is Riemann integrable if and only if the set $\operatorname{Disc}(\tilde{f})$ of discontinuities of the extended function $\tilde{f}$ has measure zero.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 23.4</span></p>

If a continuous function $f \colon U \to \mathbb{R}$ defined on an open subset $U$ of $\mathbb{R}^n$ has compact support, then $f$ is Riemann integrable on $U$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Being continuous on a compact set, the function $f$ is bounded. Being compact, the set $\operatorname{supp} f$ is closed and bounded in $\mathbb{R}^n$. We claim that the extension $\tilde{f}$ is continuous.

Since $\tilde{f}$ agrees with $f$ on $U$, the extended function $\tilde{f}$ is continuous on $U$. It remains to show that $\tilde{f}$ is continuous on the complement of $U$ in $\mathbb{R}^n$ as well. If $p \notin U$, then $p \notin \operatorname{supp} f$. Since $\operatorname{supp} f$ is a closed subset of $\mathbb{R}^n$, there is an open ball $B$ containing $p$ and disjoint from $\operatorname{supp} f$. On this open ball, $\tilde{f} \equiv 0$, which implies that $\tilde{f}$ is continuous at $p \notin U$. Thus, $\tilde{f}$ is continuous on $\mathbb{R}^n$. By Lebesgue's theorem, $f$ is Riemann integrable on $U$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 23.5</span></p>

The continuous function $f \colon \,]-1, 1[\, \to \mathbb{R}$, $f(x) = \tan(\pi x / 2)$, is defined on an open subset of finite length in $\mathbb{R}$, but is not bounded. The support of $f$ is the open interval $]-1, 1[$, which is not compact. Thus, the function $f$ does not satisfy the hypotheses of either Lebesgue's theorem or Proposition 23.4. Note that it is not Riemann integrable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The support of a real-valued function is the closure *in its domain* of the subset where the function is not zero. In Example 23.5, the support of $f$ is the open interval $]-1, 1[$, not the closed interval $[-1, 1]$, because the domain of $f$ is $]-1, 1[$, not $\mathbb{R}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 23.6</span><span class="math-callout__name">(Domain of Integration)</span></p>

A subset $A \subset \mathbb{R}^n$ is called a **domain of integration** if it is bounded and its topological boundary $\operatorname{bd}(A)$ is a set of measure zero.

</div>

Familiar plane figures such as triangles, rectangles, and circular disks are all domains of integration in $\mathbb{R}^2$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 23.7</span></p>

Every bounded continuous function $f$ defined on a domain of integration $A$ in $\mathbb{R}^n$ is Riemann integrable over $A$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $\tilde{f} \colon \mathbb{R}^n \to \mathbb{R}$ be the extension of $f$ by zero. Since $f$ is continuous on $A$, the extension $\tilde{f}$ is necessarily continuous at all interior points of $A$. Clearly, $\tilde{f}$ is continuous at all exterior points of $A$ also, because every exterior point has a neighborhood contained entirely in $\mathbb{R}^n - A$, on which $\tilde{f}$ is identically zero. Therefore, the set $\operatorname{Disc}(\tilde{f})$ of discontinuities of $\tilde{f}$ is a subset of $\operatorname{bd}(A)$, a set of measure zero. By Lebesgue's theorem, $f$ is Riemann integrable on $A$. $\square$

</details>
</div>

### 23.3 The Integral of an $n$-Form on $\mathbb{R}^n$

Once a set of coordinates $x^1, \dots, x^n$ has been fixed on $\mathbb{R}^n$, $n$-forms on $\mathbb{R}^n$ can be identified with functions on $\mathbb{R}^n$, since every $n$-form on $\mathbb{R}^n$ can be written as $\omega = f(x)\,dx^1 \wedge \cdots \wedge dx^n$ for a unique function $f(x)$ on $\mathbb{R}^n$. In this way the theory of Riemann integration of functions on $\mathbb{R}^n$ carries over to $n$-forms on $\mathbb{R}^n$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 23.8</span><span class="math-callout__name">(Integral of an $n$-Form)</span></p>

Let $\omega = f(x)\,dx^1 \wedge \cdots \wedge dx^n$ be a $C^\infty$ $n$-form on an open subset $U \subset \mathbb{R}^n$, with standard coordinates $x^1, \dots, x^n$. Its **integral** over a subset $A \subset U$ is defined to be the Riemann integral of $f(x)$:

$$\int_A \omega = \int_A f(x)\,dx^1 \wedge \cdots \wedge dx^n := \int_A f(x)\,dx^1 \cdots dx^n,$$

if the Riemann integral exists.

</div>

In this definition the $n$-form must be written in the order $dx^1 \wedge \cdots \wedge dx^n$. To integrate, for example, $\tau = f(x)\,dx^2 \wedge dx^1$ over $A \subset \mathbb{R}^2$, one would write

$$\int_A \tau = \int_A -f(x)\,dx^1 \wedge dx^2 = -\int_A f(x)\,dx^1\,dx^2.$$

**Change of variables for $n$-forms.** Let us see how the integral of an $n$-form $\omega = f\,dx^1 \wedge \cdots \wedge dx^n$ on an open subset $U \subset \mathbb{R}^n$ transforms under a change of variables. A change of variables on $U$ is given by a diffeomorphism $T \colon \mathbb{R}^n \supset V \to U \subset \mathbb{R}^n$. Let $x^1, \dots, x^n$ be the standard coordinates on $U$ and $y^1, \dots, y^n$ the standard coordinates on $V$. Then $T^i := x^i \circ T = T^*(x^i)$ is the $i$th component of $T$. Denote by $J(T)$ the Jacobian matrix $[\partial T^i / \partial y^j]$. By Corollary 18.4(ii),

$$dT^1 \wedge \cdots \wedge dT^n = \det(J(T))\,dy^1 \wedge \cdots \wedge dy^n.$$

Hence,

$$\int_V T^*\omega = \int_V (T^*f)\,T^*dx^1 \wedge \cdots \wedge T^*dx^n = \int_V (f \circ T)\,dT^1 \wedge \cdots \wedge dT^n$$

$$= \int_V (f \circ T) \det(J(T))\,dy^1 \wedge \cdots \wedge dy^n = \int_V (f \circ T) \det(J(T))\,dy^1 \cdots dy^n.$$

On the other hand, the change-of-variables formula from advanced calculus gives

$$\int_U \omega = \int_U f\,dx^1 \cdots dx^n = \int_V (f \circ T)\,|\det(J(T))|\,dy^1 \cdots dy^n,$$

with an absolute-value sign around the Jacobian determinant. These two expressions differ by the sign of $\det(J(T))$. Hence,

$$\int_V T^*\omega = \pm \int_U \omega,$$

depending on whether the Jacobian determinant $\det(J(T))$ is positive or negative.

By Proposition 21.8, a diffeomorphism $T \colon \mathbb{R}^n \supset V \to U \subset \mathbb{R}^n$ is orientation-preserving if and only if its Jacobian determinant $\det(J(T))$ is everywhere positive on $V$. The equation above shows that the integral of a differential form is not invariant under all diffeomorphisms of $V$ with $U$, but only under orientation-preserving diffeomorphisms.

### 23.4 Integral of a Differential Form over a Manifold

Integration of an $n$-form on $\mathbb{R}^n$ is not so different from integration of a function. Our approach to integration over a general manifold has several distinguishing features:

1. The manifold must be oriented (in fact, $\mathbb{R}^n$ has a standard orientation).
2. On a manifold of dimension $n$, one can integrate only $n$-forms, not functions.
3. The $n$-forms must have compact support.

Let $M$ be an oriented manifold of dimension $n$, with an oriented atlas $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ giving the orientation of $M$. Denote by $\Omega_c^k(M)$ the vector space of $C^\infty$ $k$-forms with compact support on $M$. Suppose $\lbrace (U, \phi) \rbrace$ is a chart in this atlas. If $\omega \in \Omega_c^n(U)$ is an $n$-form with compact support on $U$, then because $\phi \colon U \to \phi(U)$ is a diffeomorphism, $(\phi^{-1})^*\omega$ is an $n$-form with compact support on the open subset $\phi(U) \subset \mathbb{R}^n$. We define the integral of $\omega$ on $U$ to be

$$\int_U \omega := \int_{\phi(U)} (\phi^{-1})^*\omega.$$

If $(U, \psi)$ is another chart in the oriented atlas with the same $U$, then $\phi \circ \psi^{-1} \colon \psi(U) \to \phi(U)$ is an orientation-preserving diffeomorphism, and so

$$\int_{\phi(U)} (\phi^{-1})^*\omega = \int_{\psi(U)} (\phi \circ \psi^{-1})^*(\phi^{-1})^*\omega = \int_{\psi(U)} (\psi^{-1})^*\omega.$$

Thus, the integral $\int_U \omega$ on a chart $U$ of the atlas is well defined, independent of the choice of coordinates on $U$. By the linearity of the integral on $\mathbb{R}^n$, if $\omega, \tau \in \Omega_c^n(U)$, then

$$\int_U \omega + \tau = \int_U \omega + \int_U \tau.$$

Now let $\omega \in \Omega_c^n(M)$. Choose a partition of unity $\lbrace \rho_\alpha \rbrace$ subordinate to the open cover $\lbrace U_\alpha \rbrace$. Because $\omega$ has compact support and a partition of unity has locally finite supports, all except finitely many $\rho_\alpha \omega$ are identically zero by Problem 18.6. In particular,

$$\omega = \sum_\alpha \rho_\alpha \omega$$

is a *finite* sum. Since by Problem 18.4(b),

$$\operatorname{supp}(\rho_\alpha \omega) \subset \operatorname{supp} \rho_\alpha \cap \operatorname{supp} \omega,$$

$\operatorname{supp}(\rho_\alpha \omega)$ is a closed subset of the compact set $\operatorname{supp} \omega$. Hence, $\operatorname{supp}(\rho_\alpha \omega)$ is compact. Since $\rho_\alpha \omega$ is an $n$-form with compact support in the chart $U_\alpha$, its integral $\int_{U_\alpha} \rho_\alpha \omega$ is defined. Therefore, we can define the integral of $\omega$ over $M$ to be the finite sum

$$\int_M \omega := \sum_\alpha \int_{U_\alpha} \rho_\alpha \omega.$$

For this integral to be well defined, we must show that it is independent of the choices of oriented atlas and partition of unity. Let $\lbrace V_\beta \rbrace$ be another oriented atlas of $M$ specifying the orientation of $M$, and $\lbrace \chi_\beta \rbrace$ a partition of unity subordinate to $\lbrace V_\beta \rbrace$. Then

$$\sum_\alpha \int_{U_\alpha} \rho_\alpha \omega = \sum_\alpha \int_{U_\alpha} \rho_\alpha \sum_\beta \chi_\beta \omega = \sum_\alpha \sum_\beta \int_{U_\alpha} \rho_\alpha \chi_\beta \omega = \sum_\alpha \sum_\beta \int_{U_\alpha \cap V_\beta} \rho_\alpha \chi_\beta \omega,$$

where the last line follows from the fact that the support of $\rho_\alpha \chi_\beta$ is contained in $U_\alpha \cap V_\beta$. By symmetry, $\sum_\beta \int_{V_\beta} \chi_\beta \omega$ is equal to the same sum. Hence,

$$\sum_\alpha \int_{U_\alpha} \rho_\alpha \omega = \sum_\beta \int_{V_\beta} \chi_\beta \omega,$$

proving that the integral is well defined.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 23.9</span></p>

Let $\omega$ be an $n$-form with compact support on an oriented manifold $M$ of dimension $n$. If $-M$ denotes the same manifold but with the opposite orientation, then $\int_{-M} \omega = -\int_M \omega$.

</div>

Thus, reversing the orientation of $M$ reverses the sign of an integral over $M$.

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By the definition of an integral (using the integral on charts and the partition of unity), it is enough to show that for every chart $(U, \phi) = (U, x^1, \dots, x^n)$ and differential form $\tau \in \Omega_c^n(U)$, if $(U, \tilde{\phi}) = (U, -x^1, x^2, \dots, x^n)$ is the chart with the opposite orientation, then

$$\int_{\tilde{\phi}(U)} (\tilde{\phi}^{-1})^*\tau = -\int_{\phi(U)} (\phi^{-1})^*\tau.$$

Let $r^1, \dots, r^n$ be the standard coordinates on $\mathbb{R}^n$. Then $x^i = r^i \circ \phi$ and $r^i = x^i \circ \phi^{-1}$. With $\tilde{\phi}$, the only difference is that when $i = 1$,

$$-x^1 = r^1 \circ \tilde{\phi} \quad \text{and} \quad r^1 = -x^1 \circ \tilde{\phi}^{-1}.$$

Suppose $\tau = f\,dx^1 \wedge \cdots \wedge dx^n$ on $U$. Then

$$(\tilde{\phi}^{-1})^*\tau = (f \circ \tilde{\phi}^{-1})\,d(x^1 \circ \tilde{\phi}^{-1}) \wedge d(x^2 \circ \tilde{\phi}^{-1}) \wedge \cdots \wedge d(x^n \circ \tilde{\phi}^{-1})$$

$$= -(f \circ \tilde{\phi}^{-1})\,dr^1 \wedge dr^2 \wedge \cdots \wedge dr^n.$$

Similarly,

$$(\phi^{-1})^*\tau = (f \circ \phi^{-1})\,dr^1 \wedge dr^2 \wedge \cdots \wedge dr^n.$$

Since $\phi \circ \tilde{\phi}^{-1} \colon \tilde{\phi}(U) \to \phi(U)$ is given by

$$(\phi \circ \tilde{\phi}^{-1})(a^1, a^2, \dots, a^n) = (-a^1, a^2, \dots, a^n),$$

the absolute value of its Jacobian determinant is $|J(\phi \circ \tilde{\phi}^{-1})| = |-1| = 1$. Therefore,

$$\int_{\tilde{\phi}(U)} (\tilde{\phi}^{-1})^*\tau = -\int_{\tilde{\phi}(U)} (f \circ \tilde{\phi}^{-1})\,dr^1 \cdots dr^n$$

$$= -\int_{\tilde{\phi}(U)} (f \circ \phi^{-1}) \circ (\phi \circ \tilde{\phi}^{-1})\,|J(\phi \circ \tilde{\phi}^{-1})|\,dr^1 \cdots dr^n$$

$$= -\int_{\phi(U)} (f \circ \phi^{-1})\,dr^1 \cdots dr^n = -\int_{\phi(U)} (\phi^{-1})^*\tau. \quad \square$$

</details>
</div>

The treatment of integration above can be extended almost word for word to oriented manifolds with boundary. It has the virtue of simplicity and is of great utility in proving theorems. However, it is not practical for actual computation of integrals; an $n$-form multiplied by a partition of unity can rarely be integrated as a closed expression. To calculate explicitly integrals over an oriented $n$-manifold $M$, it is best to consider integrals over a parametrized set.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 23.10</span><span class="math-callout__name">(Parametrized Set)</span></p>

A **parametrized set** in an oriented $n$-manifold $M$ is a subset $A$ together with a $C^\infty$ map $F \colon D \to A$ from a compact domain of integration $D \subset \mathbb{R}^n$ to $M$ such that $A = F(D)$ and $F$ restricts to an orientation-preserving diffeomorphism from $\operatorname{int}(D)$ to $F(\operatorname{int}(D))$. Note that by smooth invariance of domain for manifolds (Remark 22.5), $F(\operatorname{int}(D))$ is an open subset of $M$. The $C^\infty$ map $F \colon D \to A$ is called a **parametrization** of $A$.

</div>

If $A$ is a parametrized set in $M$ with parametrization $F \colon D \to A$ and $\omega$ is a $C^\infty$ $n$-form on $M$, not necessarily with compact support, then we define $\int_A \omega$ to be $\int_D F^*\omega$. It can be shown that the definition of $\int_A \omega$ is independent of the parametrization and that in case $A$ is a manifold, it agrees with the earlier definition of integration over a manifold.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 23.11</span><span class="math-callout__name">(Integral over a sphere)</span></p>

In spherical coordinates, $\rho$ is the distance $\sqrt{x^2 + y^2 + z^2}$ of the point $(x, y, z) \in \mathbb{R}^3$ to the origin, $\varphi$ is the angle that the vector $\langle x, y, z \rangle$ makes with the positive $z$-axis, and $\theta$ is the angle that the vector $\langle x, y \rangle$ in the $(x, y)$-plane makes with the positive $x$-axis. Let $\omega$ be the 2-form on the unit sphere $S^2$ in $\mathbb{R}^3$ given by

$$\omega = \begin{cases} \dfrac{dy \wedge dz}{x} & \text{for } x \neq 0, \\[6pt] \dfrac{dz \wedge dx}{y} & \text{for } y \neq 0, \\[6pt] \dfrac{dx \wedge dy}{z} & \text{for } z \neq 0. \end{cases}$$

Up to a factor of 2, the form $\omega$ is the 2-form on $S^2$ from Problem 19.11(b). In Riemannian geometry, it is shown that $\omega$ is the area form of the sphere $S^2$ with respect to the Euclidean metric. Therefore, the integral $\int_{S^2} \omega$ is the surface area of the sphere.

**Solution.** The sphere $S^2$ has a parametrization by spherical coordinates:

$$F(\varphi, \theta) = (\sin\varphi \cos\theta, \sin\varphi \sin\theta, \cos\varphi)$$

on $D = \lbrace (\varphi, \theta) \in \mathbb{R}^2 \mid 0 \le \varphi \le \pi,\; 0 \le \theta \le 2\pi \rbrace$. Since

$$F^*x = \sin\varphi \cos\theta, \quad F^*y = \sin\varphi \sin\theta, \quad F^*z = \cos\varphi,$$

we have

$$F^*dy = \cos\varphi \sin\theta\,d\varphi + \sin\varphi \cos\theta\,d\theta$$

and

$$F^*dz = -\sin\varphi\,d\varphi,$$

so for $x \neq 0$,

$$F^*\omega = \frac{F^*dy \wedge F^*dz}{F^*x} = \sin\varphi\,d\varphi \wedge d\theta.$$

For $y \neq 0$ and $z \neq 0$, similar calculations show that $F^*\omega$ is given by the same formula. Therefore, $F^*\omega = \sin\varphi\,d\varphi \wedge d\theta$ everywhere on $D$, and

$$\int_{S^2} \omega = \int_D F^*\omega = \int_0^{2\pi} \int_0^\pi \sin\varphi\,d\varphi\,d\theta = 2\pi \left[-\cos\varphi\right]_0^\pi = 4\pi. \quad \square$$

</div>

**Integration over a zero-dimensional manifold.** The discussion of integration so far assumes implicitly that the manifold $M$ has dimension $n \ge 1$. We now treat integration over a zero-dimensional manifold. A compact oriented manifold $M$ of dimension 0 is a finite collection of points, each point oriented by $+1$ or $-1$. We write this as $M = \sum p_i - \sum q_j$. The integral of a 0-form $f \colon M \to \mathbb{R}$ is defined to be the sum

$$\int_M f = \sum f(p_i) - \sum f(q_j).$$

### 23.5 Stokes's Theorem

Let $M$ be an oriented manifold of dimension $n$ with boundary $\partial M$ and let $i \colon \partial M \hookrightarrow M$ be the inclusion map. If $\omega$ is an $(n-1)$-form on $M$, it is customary to write $\int_{\partial M} \omega$ instead of $\int_{\partial M} i^*\omega$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 23.12</span><span class="math-callout__name">(Stokes's theorem)</span></p>

For any smooth $(n-1)$-form $\omega$ with compact support on the oriented $n$-dimensional manifold $M$,

$$\int_M d\omega = \int_{\partial M} \omega.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Choose an atlas $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ for $M$ in which each $U_\alpha$ is diffeomorphic to either $\mathbb{R}^n$ or $\mathcal{H}^n$ via an orientation-preserving diffeomorphism. This is possible since any open disk is diffeomorphic to $\mathbb{R}^n$ and any half-disk containing its boundary diameter is diffeomorphic to $\mathcal{H}^n$ (see Problem 1.5). Let $\lbrace \rho_\alpha \rbrace$ be a $C^\infty$ partition of unity subordinate to $\lbrace U_\alpha \rbrace$. As we showed in the preceding section, the $(n-1)$-form $\rho_\alpha \omega$ has compact support in $U_\alpha$.

Suppose Stokes's theorem holds for $\mathbb{R}^n$ and for $\mathcal{H}^n$. Then it holds for all the charts $U_\alpha$ in our atlas, which are diffeomorphic to $\mathbb{R}^n$ or $\mathcal{H}^n$. Also, note that

$$(\partial M) \cap U_\alpha = \partial U_\alpha.$$

Therefore,

$$\int_{\partial M} \omega = \int_{\partial M} \sum_\alpha \rho_\alpha \omega = \sum_\alpha \int_{\partial M} \rho_\alpha \omega = \sum_\alpha \int_{\partial U_\alpha} \rho_\alpha \omega$$

$$= \sum_\alpha \int_{U_\alpha} d(\rho_\alpha \omega) \quad \text{(Stokes's theorem for } U_\alpha\text{)}$$

$$= \sum_\alpha \int_M d(\rho_\alpha \omega) \quad (\operatorname{supp} d(\rho_\alpha \omega) \subset U_\alpha)$$

$$= \int_M d\!\left(\sum_\alpha \rho_\alpha \omega\right) = \int_M d\omega.$$

Thus, it suffices to prove Stokes's theorem for $\mathbb{R}^n$ and for $\mathcal{H}^n$. We will give a proof only for $\mathcal{H}^2$, since the general case is similar (see Problem 23.4).

**Proof of Stokes's theorem for the upper half-plane $\mathcal{H}^2$.** Let $x, y$ be the coordinates on $\mathcal{H}^2$. Then the standard orientation on $\mathcal{H}^2$ is given by $dx \wedge dy$, and the boundary orientation on $\partial\mathcal{H}^2$ is given by $\iota_{-\partial/\partial y}(dx \wedge dy) = dx$.

The form $\omega$ is a linear combination

$$\omega = f(x, y)\,dx + g(x, y)\,dy$$

for $C^\infty$ functions $f, g$ with compact support in $\mathcal{H}^2$. Since the supports of $f$ and $g$ are compact, we may choose a real number $a > 0$ large enough that the supports of $f$ and $g$ are contained in the interior of the square $[-a, a] \times [0, a]$. Then

$$d\omega = \left(\frac{\partial g}{\partial x} - \frac{\partial f}{\partial y}\right) dx \wedge dy = (g_x - f_y)\,dx \wedge dy,$$

and

$$\int_{\mathcal{H}^2} d\omega = \int_{\mathcal{H}^2} g_x\,dx\,dy - \int_{\mathcal{H}^2} f_y\,dx\,dy = \int_0^a \int_{-a}^a g_x\,dx\,dy - \int_{-a}^a \int_0^a f_y\,dy\,dx.$$

In this expression,

$$\int_{-a}^a g_x(x, y)\,dx = g(x, y)\Big|_{x=-a}^a = 0$$

because $\operatorname{supp} g$ lies in the interior of $[-a, a] \times [0, a]$. Similarly,

$$\int_0^a f_y(x, y)\,dy = f(x, y)\Big|_{y=0}^a = -f(x, 0)$$

because $f(x, a) = 0$. Thus,

$$\int_{\mathcal{H}^2} d\omega = \int_{-a}^a f(x, 0)\,dx.$$

On the other hand, $\partial\mathcal{H}^2$ is the $x$-axis and $dy = 0$ on $\partial\mathcal{H}^2$. It follows from the expression for $\omega$ that $\omega = f(x, 0)\,dx$ when restricted to $\partial\mathcal{H}^2$ and

$$\int_{\partial\mathcal{H}^2} \omega = \int_{-a}^a f(x, 0)\,dx.$$

This proves Stokes's theorem for the upper half-plane. $\square$

</details>
</div>

### 23.6 Line Integrals and Green's Theorem

We will now show how Stokes's theorem for a manifold unifies some of the theorems of vector calculus on $\mathbb{R}^2$ and $\mathbb{R}^3$. Recall the calculus notation $\mathbf{F} \cdot d\mathbf{r} = P\,dx + Q\,dy + R\,dz$ for $\mathbf{F} = \langle P, Q, R \rangle$ and $\mathbf{r} = (x, y, z)$. As in calculus, we assume in this section that functions, vector fields, and regions of integration have sufficient smoothness or regularity properties so that all the integrals are defined.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 23.13</span><span class="math-callout__name">(Fundamental theorem for line integrals)</span></p>

Let $C$ be a curve in $\mathbb{R}^3$, parametrized by $\mathbf{r}(t) = (x(t), y(t), z(t))$, $a \le t \le b$, and let $\mathbf{F}$ be a vector field on $\mathbb{R}^3$. If $\mathbf{F} = \operatorname{grad} f$ for some scalar function $f$, then

$$\int_C \mathbf{F} \cdot d\mathbf{r} = f(\mathbf{r}(b)) - f(\mathbf{r}(a)).$$

</div>

Suppose in Stokes's theorem we take $M$ to be a curve $C$ with parametrization $\mathbf{r}(t)$, $a \le t \le b$, and $\omega$ to be the function $f$ on $C$. Then

$$\int_C d\omega = \int_C df = \int_C \frac{\partial f}{\partial x}\,dx + \frac{\partial f}{\partial y}\,dy + \frac{\partial f}{\partial z}\,dz = \int_C \operatorname{grad} f \cdot d\mathbf{r}$$

and

$$\int_{\partial C} \omega = f\Big|_{\mathbf{r}(a)}^{\mathbf{r}(b)} = f(\mathbf{r}(b)) - f(\mathbf{r}(a)).$$

In this case Stokes's theorem specializes to the fundamental theorem for line integrals.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 23.14</span><span class="math-callout__name">(Green's theorem)</span></p>

If $D$ is a plane region with boundary $\partial D$, and $P$ and $Q$ are $C^\infty$ functions on $D$, then

$$\int_{\partial D} P\,dx + Q\,dy = \int_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA.$$

</div>

In this statement, $dA$ is the usual calculus notation for $dx\,dy$. To obtain Green's theorem, let $M$ be a plane region $D$ with boundary $\partial D$ and let $\omega$ be the 1-form $P\,dx + Q\,dy$ on $D$. Then

$$\int_{\partial D} \omega = \int_{\partial D} P\,dx + Q\,dy$$

and

$$\int_D d\omega = \int_D P_y\,dy \wedge dx + Q_x\,dx \wedge dy = \int_D (Q_x - P_y)\,dx \wedge dy = \int_D (Q_x - P_y)\,dA.$$

In this case Stokes's theorem is Green's theorem in the plane.

# Chapter 7 — De Rham Theory

By the fundamental theorem for line integrals (Theorem 23.13), if a smooth vector field $\mathbf{F}$ is the gradient of a scalar function $f$, then for any two points $p$ and $q$ in $\mathbb{R}^3$, the line integral $\int_C \mathbf{F} \cdot d\mathbf{r}$ over a curve $C$ from $p$ to $q$ is independent of the curve. In this case, $\int_C \mathbf{F} \cdot d\mathbf{r} = f(q) - f(p)$. Similarly, by the classical Stokes theorem for a surface, the surface integral of a smooth vector field $\mathbf{F}$ over an oriented surface $S$ with boundary $C$ in $\mathbb{R}^3$ can be evaluated as an integral over the curve $C$ if $\mathbf{F}$ is the curl of another vector field. By the correspondence of Section 4.6 between vector fields and differential forms, these questions translate into whether a differential form $\omega$ on $\mathbb{R}^3$ is exact.

Poincaré proved in 1887 that for $k = 1, 2, 3$, a $k$-form on $\mathbb{R}^n$ is exact if and only if it is closed, a lemma that now bears his name. Vito Volterra published in 1889 the first complete proof of the Poincaré lemma for all $k$.

Whether every closed form on a manifold is exact depends on the topology of the manifold. For example, on $\mathbb{R}^2$ every closed $k$-form is exact for $k > 0$, but on the punctured plane $\mathbb{R}^2 - \lbrace (0,0) \rbrace$ there are closed 1-forms that are not exact. The extent to which closed forms are not exact is measured by the de Rham cohomology, possibly the most important diffeomorphism invariant of a manifold.

## §24 De Rham Cohomology

In this section we define de Rham cohomology, prove some of its basic properties, and compute two elementary examples: the de Rham cohomology vector spaces of the real line and of the unit circle.

### 24.1 De Rham Cohomology

In Section 4.6 we established a one-to-one correspondence between vector fields and differential 1-forms on an open subset of $\mathbb{R}^3$. There is a similar correspondence on an open subset of any $\mathbb{R}^n$. For $\mathbb{R}^2$, it is as follows:

$$\mathbf{F} = \langle P, Q \rangle \longleftrightarrow \omega = P\,dx + Q\,dy,$$

$$\operatorname{grad} f = \langle f_x, f_y \rangle \longleftrightarrow df = f_x\,dx + f_y\,dy,$$

$$Q_x - P_y = 0 \longleftrightarrow d\omega = (Q_x - P_y)\,dx \wedge dy = 0.$$

In terms of differential forms the question becomes the following: if the 1-form $\omega = P\,dx + Q\,dy$ is closed on $U$, is it exact? The answer to this question is sometimes yes and sometimes no, depending on the topology of $U$.

Just as for an open subset of $\mathbb{R}^n$, a differential form $\omega$ on a manifold $M$ is said to be **closed** if $d\omega = 0$, and **exact** if $\omega = d\tau$ for some form $\tau$ of degree one less. Since $d^2 = 0$, every exact form is closed. In general, not every closed form is exact.

Let $Z^k(M)$ be the vector space of all closed $k$-forms and $B^k(M)$ the vector space of all exact $k$-forms on the manifold $M$. Because every exact form is closed, $B^k(M)$ is a subspace of $Z^k(M)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(De Rham Cohomology)</span></p>

The quotient vector space $H^k(M) := Z^k(M) / B^k(M)$ measures the extent to which closed $k$-forms fail to be exact, and is called the **de Rham cohomology** of $M$ in degree $k$. The quotient vector space construction introduces an equivalence relation on $Z^k(M)$:

$$\omega' \sim \omega \quad \text{in } Z^k(M) \quad \text{iff} \quad \omega' - \omega \in B^k(M).$$

The equivalence class of a closed form $\omega$ is called its **cohomology class** and denoted by $[\omega]$. Two closed forms $\omega$ and $\omega'$ determine the same cohomology class if and only if they differ by an exact form:

$$\omega' = \omega + d\tau.$$

In this case we say that the two closed forms $\omega$ and $\omega'$ are **cohomologous**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 24.1</span></p>

If the manifold $M$ has $r$ connected components, then its de Rham cohomology in degree $0$ is $H^0(M) = \mathbb{R}^r$. An element of $H^0(M)$ is specified by an ordered $r$-tuple of real numbers, each real number representing a constant function on a connected component of $M$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since there are no nonzero exact 0-forms,

$$H^0(M) = Z^0(M) = \lbrace \text{closed 0-forms} \rbrace.$$

Suppose $f$ is a closed 0-form on $M$; i.e., $f$ is a $C^\infty$ function on $M$ such that $df = 0$. On any chart $(U, x^1, \dots, x^n)$,

$$df = \sum \frac{\partial f}{\partial x^i}\,dx^i.$$

Thus, $df = 0$ on $U$ if and only if all the partial derivatives $\partial f / \partial x^i$ vanish identically on $U$. This in turn is equivalent to $f$ being locally constant on $U$. Hence, the closed 0-forms on $M$ are precisely the locally constant functions on $M$. Such a function must be constant on each connected component of $M$. If $M$ has $r$ connected components, then a locally constant function on $M$ can be specified by an ordered set of $r$ real numbers. Thus, $Z^0(M) = \mathbb{R}^r$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 24.2</span></p>

On a manifold $M$ of dimension $n$, the de Rham cohomology $H^k(M)$ vanishes for $k > n$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

At any point $p \in M$, the tangent space $T_pM$ is a vector space of dimension $n$. If $\omega$ is a $k$-form on $M$, then $\omega_p \in A_k(T_pM)$, the space of alternating $k$-linear functions on $T_pM$. By Corollary 3.31, if $k > n$, then $A_k(T_pM) = 0$. Hence, for $k > n$, the only $k$-form on $M$ is the zero form. $\square$

</details>
</div>

### 24.2 Examples of de Rham Cohomology

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 24.3</span><span class="math-callout__name">(De Rham cohomology of the real line)</span></p>

Since the real line $\mathbb{R}^1$ is connected, by Proposition 24.1,

$$H^0(\mathbb{R}^1) = \mathbb{R}.$$

For dimensional reasons, on $\mathbb{R}^1$ there are no nonzero 2-forms. This implies that every 1-form on $\mathbb{R}^1$ is closed. A 1-form $f(x)\,dx$ on $\mathbb{R}^1$ is exact if and only if there is a $C^\infty$ function $g(x)$ on $\mathbb{R}^1$ such that

$$f(x)\,dx = dg = g'(x)\,dx,$$

where $g'(x)$ is the calculus derivative of $g$ with respect to $x$. Such a function $g(x)$ is simply an antiderivative of $f(x)$, for example

$$g(x) = \int_0^x f(t)\,dt.$$

This proves that every 1-form on $\mathbb{R}^1$ is exact. Therefore, $H^1(\mathbb{R}^1) = 0$. In combination with Proposition 24.2, we have

$$H^k(\mathbb{R}^1) = \begin{cases} \mathbb{R} & \text{for } k = 0, \\ 0 & \text{for } k \ge 1. \end{cases}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 24.4</span><span class="math-callout__name">(De Rham cohomology of a circle)</span></p>

Let $S^1$ be the unit circle in the $xy$-plane. By Proposition 24.1, $H^0(S^1) = \mathbb{R}$, and because $S^1$ is one-dimensional, $H^k(S^1) = 0$ for all $k \ge 2$. It remains to compute $H^1(S^1)$.

Recall from Subsection 18.7 the map $h \colon \mathbb{R} \to S^1$, $h(t) = (\cos t, \sin t)$. Let $i \colon [0, 2\pi] \to \mathbb{R}$ be the inclusion map. Restricting the domain of $h$ to $[0, 2\pi]$ gives a parametrization $F := h \circ i \colon [0, 2\pi] \to S^1$ of the circle. In Examples 17.15 and 17.16, we found a nowhere-vanishing 1-form $\omega = -y\,dx + x\,dy$ on $S^1$ and showed that $F^*\omega = i^*h^*\omega = i^*dt = dt$. Thus,

$$\int_{S^1} \omega = \int_{F([0,2\pi])} \omega = \int_{[0,2\pi]} F^*\omega = \int_0^{2\pi} dt = 2\pi.$$

Since the circle has dimension 1, all 1-forms on $S^1$ are closed, so $\Omega^1(S^1) = Z^1(S^1)$. The integration of 1-forms on $S^1$ defines a linear map

$$\varphi \colon Z^1(S^1) = \Omega^1(S^1) \to \mathbb{R}, \quad \varphi(\alpha) = \int_{S^1} \alpha.$$

Because $\varphi(\omega) = 2\pi \neq 0$, the linear map $\varphi$ is onto.

By Stokes's theorem, the exact 1-forms on $S^1$ are in $\ker \varphi$. Conversely, we will show that all 1-forms in $\ker \varphi$ are exact. Suppose $\alpha = f\omega$ is a smooth 1-form on $S^1$ such that $\varphi(\alpha) = 0$. Let $\tilde{f} = h^*f = f \circ h \in \Omega^0(\mathbb{R})$. Then $\tilde{f}$ is periodic of period $2\pi$ and $\int_0^{2\pi} \tilde{f}(u)\,du = 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 24.5</span></p>

Suppose $\tilde{f}$ is a $C^\infty$ periodic function of period $2\pi$ on $\mathbb{R}$ and $\int_0^{2\pi} \tilde{f}(u)\,du = 0$. Then $\tilde{f}\,dt = d\tilde{g}$ for a $C^\infty$ periodic function $\tilde{g}$ of period $2\pi$ on $\mathbb{R}$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Define $\tilde{g} \in \Omega^0(\mathbb{R})$ by

$$\tilde{g}(t) = \int_0^t \tilde{f}(u)\,du.$$

Since $\int_0^{2\pi} \tilde{f}(u)\,du = 0$ and $\tilde{f}$ is periodic of period $2\pi$,

$$\tilde{g}(t + 2\pi) = \int_0^{2\pi} \tilde{f}(u)\,du + \int_{2\pi}^{t+2\pi} \tilde{f}(u)\,du = 0 + \int_0^t \tilde{f}(u)\,du = \tilde{g}(t).$$

Hence, $\tilde{g}(t)$ is also periodic of period $2\pi$ on $\mathbb{R}$. Moreover,

$$d\tilde{g} = \tilde{g}'(t)\,dt = \tilde{f}(t)\,dt. \quad \square$$

</details>
</div>

Let $\tilde{g}$ be the periodic function of period $2\pi$ on $\mathbb{R}$ from Lemma 24.5. By Proposition 18.12, $\tilde{g} = h^*g$ for some $C^\infty$ function $g$ on $S^1$. It follows that

$$d\tilde{g} = dh^*g = h^*(dg).$$

On the other hand,

$$\tilde{f}(t)\,dt = (h^*f)(h^*\omega) = h^*(f\omega) = h^*\alpha.$$

Since $h^* \colon \Omega^1(S^1) \to \Omega^1(\mathbb{R})$ is injective, $\alpha = dg$. This proves that the kernel of $\varphi$ consists of exact forms. Therefore, integration induces an isomorphism

$$H^1(S^1) = \frac{Z^1(S^1)}{B^1(S^1)} \xrightarrow{\;\sim\;} \mathbb{R}.$$

</div>

### 24.3 Diffeomorphism Invariance

For any smooth map $F \colon N \to M$ of manifolds, there is a pullback map $F^* \colon \Omega^*(M) \to \Omega^*(N)$ of differential forms. Moreover, the pullback $F^*$ commutes with the exterior derivative $d$ (Proposition 19.5).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 24.6</span></p>

The pullback map $F^*$ sends closed forms to closed forms, and sends exact forms to exact forms.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Suppose $\omega$ is closed. By the commutativity of $F^*$ with $d$,

$$dF^*\omega = F^*d\omega = 0.$$

Hence, $F^*\omega$ is also closed.

Next suppose $\omega = d\tau$ is exact. Then

$$F^*\omega = F^*d\tau = dF^*\tau.$$

Hence, $F^*\omega$ is exact. $\square$

</details>
</div>

It follows that $F^*$ induces a linear map of quotient spaces, denoted by $F^\sharp$:

$$F^\sharp \colon \frac{Z^k(M)}{B^k(M)} \to \frac{Z^k(N)}{B^k(N)}, \quad F^\sharp([\omega]) = [F^*(\omega)].$$

This is a map in cohomology,

$$F^\sharp \colon H^k(M) \to H^k(N),$$

called the **pullback map in cohomology**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.7</span></p>

The functorial properties of the pullback map $F^*$ on differential forms easily yield the same functorial properties for the induced map in cohomology:

1. If $\mathbb{1}_M \colon M \to M$ is the identity map, then $\mathbb{1}_M^\sharp \colon H^k(M) \to H^k(M)$ is also the identity map.
2. If $F \colon N \to M$ and $G \colon M \to P$ are smooth maps, then $(G \circ F)^\sharp = F^\sharp \circ G^\sharp$.

It follows from (i) and (ii) that $(H^k(\,\cdot\,), F^\sharp)$ is a contravariant functor from the category of $C^\infty$ manifolds and $C^\infty$ maps to the category of vector spaces and linear maps. By Proposition 10.3, if $F \colon N \to M$ is a diffeomorphism of manifolds, then $F^\sharp \colon H^k(M) \to H^k(N)$ is an isomorphism of vector spaces.

</div>

In fact, the usual notation for the induced map in cohomology is $F^*$, the same as for the pullback map on differential forms. Unless there is a possibility of confusion, henceforth we will follow this convention. It is usually clear from the context whether $F^*$ is a map in cohomology or on forms.

### 24.4 The Ring Structure on de Rham Cohomology

The wedge product of differential forms on a manifold $M$ gives the vector space $\Omega^*(M)$ of differential forms a product structure. This product structure induces a product structure in cohomology: if $[\omega] \in H^k(M)$ and $[\tau] \in H^\ell(M)$, define

$$[\omega] \wedge [\tau] = [\omega \wedge \tau] \in H^{k+\ell}(M).$$

For the product to be well defined, we need to check three things about closed forms $\omega$ and $\tau$:

1. The wedge product $\omega \wedge \tau$ is a closed form.
2. The class $[\omega \wedge \tau]$ is independent of the choice of representative for $[\tau]$. In other words, if $\tau$ is replaced by a cohomologous form $\tau' = \tau + d\sigma$, then $\omega \wedge d\sigma$ is exact.
3. The class $[\omega \wedge \tau]$ is independent of the choice of representative for $[\omega]$.

These all follow from the antiderivation property of $d$. For example, in (i), since $\omega$ and $\tau$ are closed,

$$d(\omega \wedge \tau) = (d\omega) \wedge \tau + (-1)^k \omega \wedge d\tau = 0.$$

In (ii),

$$d(\omega \wedge \sigma) = (d\omega) \wedge \sigma + (-1)^k \omega \wedge d\sigma = (-1)^k \omega \wedge d\sigma \quad (\text{since } d\omega = 0),$$

which shows that $\omega \wedge d\sigma$ is exact. Item (iii) is analogous to (ii), with the roles of $\omega$ and $\tau$ reversed.

If $M$ is a manifold of dimension $n$, we set

$$H^*(M) = \bigoplus_{k=0}^n H^k(M).$$

Elements of $H^*(M)$ can be added and multiplied in the same way that one would add or multiply polynomials, except here multiplication is the wedge product. It is easy to check that under addition and multiplication, $H^*(M)$ satisfies all the properties of a ring, called the **cohomology ring** of $M$. The ring $H^*(M)$ has a natural grading by the degree of a closed form. It is an **anticommutative graded ring**: for all $a \in H^k(M)$ and $b \in H^\ell(M)$,

$$a \cdot b = (-1)^{k\ell}\, b \cdot a.$$

Since $H^*(M)$ is also a real vector space, it is in fact an anticommutative graded algebra over $\mathbb{R}$.

Suppose $F \colon N \to M$ is a $C^\infty$ map of manifolds. Because $F^*(\omega \wedge \tau) = F^*\omega \wedge F^*\tau$ for differential forms $\omega$ and $\tau$ on $M$ (Proposition 18.11), the linear map $F^* \colon H^*(M) \to H^*(N)$ is a ring homomorphism. By Remark 24.7, if $N \to M$ is a diffeomorphism, then the pullback $F^* \colon H^*(M) \to H^*(N)$ is a ring isomorphism.

To sum up, de Rham cohomology gives a contravariant functor from the category of $C^\infty$ manifolds to the category of anticommutative graded rings. If $M$ and $N$ are diffeomorphic manifolds, then $H^*(M)$ and $H^*(N)$ are isomorphic as anticommutative graded rings. In this way the de Rham cohomology becomes a powerful diffeomorphism invariant of $C^\infty$ manifolds.

## §25 The Long Exact Sequence in Cohomology

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cochain Complex)</span></p>

A **cochain complex** $\mathcal{C}$ is a collection of vector spaces $\lbrace C^k \rbrace_{k \in \mathbb{Z}}$ together with a sequence of linear maps $d_k \colon C^k \to C^{k+1}$,

$$\cdots \to C^{-1} \xrightarrow{d_{-1}} C^0 \xrightarrow{d_0} C^1 \xrightarrow{d_1} C^2 \xrightarrow{d_2} \cdots,$$

such that

$$d_k \circ d_{k-1} = 0$$

for all $k$. We call the collection of linear maps $\lbrace d_k \rbrace$ the **differential** of the cochain complex $\mathcal{C}$.

</div>

The vector space $\Omega^*(M)$ of differential forms on a manifold $M$ together with the exterior derivative $d$ is a cochain complex, the **de Rham complex** of $M$:

$$0 \to \Omega^0(M) \xrightarrow{d} \Omega^1(M) \xrightarrow{d} \Omega^2(M) \xrightarrow{d} \cdots, \quad d \circ d = 0.$$

It turns out that many of the results on the de Rham cohomology of a manifold depend not on the topological properties of the manifold, but on the algebraic properties of the de Rham complex. To better understand de Rham cohomology, it is useful to isolate these algebraic properties. In this section we investigate the properties of a cochain complex that constitute the beginning of a subject known as *homological algebra*.

### 25.1 Exact Sequences

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 25.1</span><span class="math-callout__name">(Exact Sequence)</span></p>

A sequence of homomorphisms of vector spaces

$$A \xrightarrow{f} B \xrightarrow{g} C$$

is said to be **exact at $B$** if $\operatorname{im} f = \ker g$. A sequence of homomorphisms

$$A^0 \xrightarrow{f_0} A^1 \xrightarrow{f_1} A^2 \xrightarrow{f_2} \cdots \xrightarrow{f_{n-1}} A^n$$

that is exact at every term except the first and the last is simply said to be an **exact sequence**. A five-term exact sequence of the form

$$0 \to A \to B \to C \to 0$$

is said to be **short exact**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

**(i)** When $A = 0$, the sequence $0 \xrightarrow{f} B \xrightarrow{g} C$ is exact if and only if $\ker g = \operatorname{im} f = 0$, so that $g$ is injective.

**(ii)** Similarly, when $C = 0$, the sequence $A \xrightarrow{f} B \xrightarrow{g} 0$ is exact if and only if $\operatorname{im} f = \ker g = B$, so that $f$ is surjective.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 25.2</span><span class="math-callout__name">(A three-term exact sequence)</span></p>

Suppose

$$A \xrightarrow{f} B \xrightarrow{g} C$$

is an exact sequence. Then

1. the map $f$ is surjective if and only if $g$ is the zero map;
2. the map $g$ is injective if and only if $f$ is the zero map.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 25.3</span><span class="math-callout__name">(A four-term exact sequence)</span></p>

**(i)** The four-term sequence $0 \to A \xrightarrow{f} B \to 0$ of vector spaces is exact if and only if $f \colon A \to B$ is an isomorphism.

**(ii)** If

$$A \xrightarrow{f} B \to C \to 0$$

is an exact sequence of vector spaces, then there is a linear isomorphism

$$C \simeq \operatorname{coker} f := \frac{B}{\operatorname{im} f}.$$

</div>

### 25.2 Cohomology of Cochain Complexes

If $\mathcal{C}$ is a cochain complex, then by the condition $d_k \circ d_{k-1} = 0$,

$$\operatorname{im} d_{k-1} \subset \ker d_k.$$

We can therefore form the quotient vector space

$$H^k(\mathcal{C}) := \frac{\ker d_k}{\operatorname{im} d_{k-1}},$$

which is called the **$k$th cohomology vector space** of the cochain complex $\mathcal{C}$. It is a measure of the extent to which the cochain complex $\mathcal{C}$ fails to be exact at $C^k$.

Elements of the vector space $C^k$ are called **cochains of degree $k$** or **$k$-cochains** for short. A $k$-cochain in $\ker d_k$ is called a **$k$-cocycle** and a $k$-cochain in $\operatorname{im} d_{k-1}$ is called a **$k$-coboundary**. The equivalence class $[c] \in H^k(\mathcal{C})$ of a $k$-cocycle $c \in \ker d_k$ is called its **cohomology class**. We denote the subspaces of $k$-cocycles and $k$-coboundaries of $\mathcal{C}$ by $Z^k(\mathcal{C})$ and $B^k(\mathcal{C})$ respectively. The letter $Z$ for cocycles comes from *Zyklen*, the German word for cycles.

In the de Rham complex, a cocycle is a closed form and a coboundary is an exact form.

If $\mathcal{A}$ and $\mathcal{B}$ are two cochain complexes with differentials $d$ and $d'$ respectively, a **cochain map** $\varphi \colon \mathcal{A} \to \mathcal{B}$ is a collection of linear maps $\varphi_k \colon A^k \to B^k$, one for each $k$, that commute with $d$ and $d'$:

$$d' \circ \varphi_k = \varphi_{k+1} \circ d.$$

A cochain map $\varphi \colon \mathcal{A} \to \mathcal{B}$ naturally induces a linear map in cohomology

$$\varphi^* \colon H^k(\mathcal{A}) \to H^k(\mathcal{B})$$

by

$$\varphi^*[a] = [\varphi(a)].$$

This is well defined because a cochain map takes cocycles to cocycles and coboundaries to coboundaries:

1. For $a \in Z^k(\mathcal{A})$: $d'(\varphi(a)) = \varphi(da) = 0$.
2. For $a' \in A^{k-1}$: $\varphi(da') = d'(\varphi(a'))$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 25.4</span></p>

**(i)** For a smooth map $F \colon N \to M$ of manifolds, the pullback map $F^* \colon \Omega^*(M) \to \Omega^*(N)$ on differential forms is a cochain map, because $F^*$ commutes with $d$ (Proposition 19.5). By the discussion above, there is an induced map $F^* \colon H^k(M) \to H^k(N)$ in cohomology, as we saw once before, after Lemma 24.6.

**(ii)** If $X$ is a $C^\infty$ vector field on a manifold $M$, then the Lie derivative $\mathcal{L}_X \colon \Omega^*(M) \to \Omega^*(M)$ commutes with $d$ (Theorem 20.10(ii)). By the above, $\mathcal{L}_X$ induces a linear map $\mathcal{L}_X^* \colon H^*(M) \to H^*(M)$ in cohomology.

</div>

### 25.3 The Connecting Homomorphism

A sequence of cochain complexes

$$0 \to \mathcal{A} \xrightarrow{i} \mathcal{B} \xrightarrow{j} \mathcal{C} \to 0$$

is **short exact** if $i$ and $j$ are cochain maps and for each $k$,

$$0 \to A^k \xrightarrow{i_k} B^k \xrightarrow{j_k} C^k \to 0$$

is a short exact sequence of vector spaces. We usually omit subscripts on cochain maps, writing simply $i$, $j$ instead of $i_k$, $j_k$.

Given a short exact sequence as above, we can construct a linear map $d^* \colon H^k(\mathcal{C}) \to H^{k+1}(\mathcal{A})$, called the **connecting homomorphism**, as follows. Consider the short exact sequences in dimensions $k$ and $k+1$:

Start with $[c] \in H^k(\mathcal{C})$. Since $j \colon B^k \to C^k$ is onto, there is an element $b \in B^k$ such that $j(b) = c$. Then $db \in B^{k+1}$ is in $\ker j$ because

$$j\,db = d\,jb = dc = 0 \quad (\text{because } c \text{ is a cocycle}).$$

By the exactness of the sequence in degree $k+1$, $\ker j = \operatorname{im} i$. This implies that $db = i(a)$ for some $a$ in $A^{k+1}$. Once $b$ is chosen, this $a$ is unique because $i$ is injective. The injectivity of $i$ also implies that $da = 0$, since

$$i(da) = d(ia) = ddb = 0.$$

Therefore, $a$ is a cocycle and defines a cohomology class $[a]$. We set

$$d^*[c] = [a] \in H^{k+1}(\mathcal{A}).$$

The recipe for defining the connecting homomorphism $d^*$ is best remembered as a zig-zag diagram: $a \xleftarrow{i} db$, $b \xrightarrow{j} c$, where $a \xleftarrow{i} db$ means that $a$ maps to $db$ under an injection and $b \xrightarrow{j} c$ means that $b$ maps to $c$ under a surjection.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 25.5</span><span class="math-callout__name">(Connecting homomorphism)</span></p>

Show that the connecting homomorphism

$$d^* \colon H^k(\mathcal{C}) \to H^{k+1}(\mathcal{A})$$

is a well-defined linear map.

</div>

### 25.4 The Zig-Zag Lemma

The zig-zag lemma produces a long exact sequence in cohomology from a short exact sequence of cochain complexes. It is most useful when some of the terms in the long exact sequence are known to be zero, for then by exactness, the adjacent maps will be injections, surjections, or even isomorphisms. For example, if the cohomology of one of the three cochain complexes is zero, then the cohomology vector spaces of the other two cochain complexes will be isomorphic.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 25.6</span><span class="math-callout__name">(The zig-zag lemma)</span></p>

A short exact sequence of cochain complexes

$$0 \to \mathcal{A} \xrightarrow{i} \mathcal{B} \xrightarrow{j} \mathcal{C} \to 0$$

gives rise to a long exact sequence in cohomology:

$$\cdots \to H^k(\mathcal{A}) \xrightarrow{i^*} H^k(\mathcal{B}) \xrightarrow{j^*} H^k(\mathcal{C}) \xrightarrow{d^*} H^{k+1}(\mathcal{A}) \xrightarrow{i^*} \cdots,$$

where $i^*$ and $j^*$ are the maps in cohomology induced from the cochain maps $i$ and $j$, and $d^*$ is the connecting homomorphism.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (exactness at $H^k(\mathcal{C})$)</summary>

To prove the theorem one needs to check exactness at $H^k(\mathcal{A})$, $H^k(\mathcal{B})$, and $H^k(\mathcal{C})$ for each $k$. As an example, we prove exactness at $H^k(\mathcal{C})$.

**Claim.** $\operatorname{im} j^* \subset \ker d^*$.

*Proof.* Let $[b] \in H^k(\mathcal{B})$. Then

$$d^*\,j^*[b] = d^*[j(b)].$$

In the recipe for $d^*$, we can choose the element in $B^k$ that maps to $j(b)$ to be $b$. Then $db \in B^{k+1}$. Because $b$ is a cocycle, $db = 0$. Following the zig-zag diagram, since $i(0) = 0 = db$, we must have $d^*[j(b)] = [0]$. So $j^*[b] \in \ker d^*$. $\square$

**Claim.** $\ker d^* \subset \operatorname{im} j^*$.

*Proof.* Suppose $d^*[c] = [a] = 0$, where $[c] \in H^k(\mathcal{C})$. This means that $a = da'$ for some $a' \in A^k$. The calculation of $d^*[c]$ can be represented by the zig-zag diagram: $a \xleftarrow{i} db$, $b \xrightarrow{j} c$, where $b$ is an element in $B^k$ with $j(b) = c$ and $i(a) = db$. Then $b - i(a')$ is a cocycle in $B^k$ that maps to $c$ under $j$:

$$d(b - i(a')) = db - di(a') = db - id(a') = db - ia = 0,$$

$$j(b - i(a')) = j(b) - ji(a') = j(b) = c.$$

Therefore, $j^*[b - i(a')] = [c]$. So $[c] \in \operatorname{im} j^*$. $\square$

These two claims together imply the exactness of the cohomology sequence at $H^k(\mathcal{C})$. The exactness at $H^k(\mathcal{A})$ and at $H^k(\mathcal{B})$ is left as an exercise (Problem 25.3).

</details>
</div>

## §26 The Mayer–Vietoris Sequence

As the example of the cohomology of $\mathbb{R}^1$ illustrates, calculating the de Rham cohomology of a manifold amounts to solving a canonically given system of differential equations on the manifold and, in case it is not solvable, to finding obstructions to its solvability. This is usually quite difficult to do directly. We introduce here one of the most useful tools in the calculation of de Rham cohomology, the Mayer–Vietoris sequence.

### 26.1 The Mayer–Vietoris Sequence

Let $\lbrace U, V \rbrace$ be an open cover of a manifold $M$, and let $i_U \colon U \to M$, $i_V(p) = p$, be the inclusion map. Then the pullback

$$i_U^* \colon \Omega^k(M) \to \Omega^k(U)$$

is the restriction map that restricts the domain of a $k$-form on $M$ to $U$: $i_U^* \omega = \omega\vert_U$. There are four inclusion maps forming a commutative diagram among $U \cap V$, $U$, $V$, and $M$.

By restricting a $k$-form from $M$ to $U$ and to $V$, we get a homomorphism of vector spaces

$$i \colon \Omega^k(M) \to \Omega^k(U) \oplus \Omega^k(V),$$

$$\sigma \mapsto (i_U^* \sigma, i_V^* \sigma) = (\sigma\vert_U, \sigma\vert_V).$$

Define the **difference map**

$$j \colon \Omega^k(U) \oplus \Omega^k(V) \to \Omega^k(U \cap V)$$

by

$$j(\omega, \tau) = j_V^* \tau - j_U^* \omega = \tau\vert_{U \cap V} - \omega\vert_{U \cap V}.$$

If $U \cap V$ is empty, we define $\Omega^k(U \cap V) = 0$. In this case $j$ is simply the zero map. We call $i$ the **restriction map** and $j$ the **difference map**. Since the direct sum $\Omega^*(U) \oplus \Omega^*(V)$ is the de Rham complex $\Omega^*(U \amalg V)$ of the disjoint union $U \amalg V$, the exterior derivative $d$ on $\Omega^*(U) \oplus \Omega^*(V)$ is given by $d(\omega, \tau) = (d\omega, d\tau)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 26.1</span></p>

Both the restriction map $i$ and the difference map $j$ commute with the exterior derivative $d$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

This is a consequence of the commutativity of $d$ with the pullback. For $\sigma \in \Omega^k(M)$,

$$di\sigma = d(i_U^* \sigma, i_V^* \sigma) = (di_U^* \sigma, di_V^* \sigma) = (i_U^* d\sigma, i_V^* d\sigma) = id\sigma.$$

For $(\omega, \tau) \in \Omega^k(U) \oplus \Omega^k(V)$,

$$dj(\omega, \tau) = d(j_V^* \tau - j_U^* \omega) = j_V^* d\tau - j_U^* d\omega = jd(\omega, \tau).$$

Thus, $i$ and $j$ are cochain maps. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 26.2</span></p>

For each integer $k \ge 0$, the sequence

$$0 \to \Omega^k(M) \xrightarrow{i} \Omega^k(U) \oplus \Omega^k(V) \xrightarrow{j} \Omega^k(U \cap V) \to 0$$

is exact.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof sketch (surjectivity of $j$)</summary>

Exactness at the first two terms $\Omega^k(M)$ and $\Omega^k(U) \oplus \Omega^k(V)$ is straightforward. To prove surjectivity of $j$, let $\lbrace \rho_U, \rho_V \rbrace$ be a partition of unity subordinate to the open cover $\lbrace U, V \rbrace$. For $\omega \in \Omega^k(U \cap V)$, define $\omega_U$ to be the extension by zero of $\rho_V \omega$ from $U \cap V$ to $U$, and $\omega_V$ to be the extension by zero of $\rho_U \omega$ from $U \cap V$ to $V$. On $U \cap V$, $(-\omega_U, \omega_V)$ restricts to $(-\rho_V \omega, \rho_U \omega)$. Hence $j$ maps $(-\omega_U, \omega_V) \in \Omega^k(U) \oplus \Omega^k(V)$ to

$$\rho_V \omega - (-\rho_U \omega) = \omega \in \Omega^k(U \cap V).$$

This shows that $j$ is surjective and the sequence is exact at $\Omega^k(U \cap V)$. $\square$

</details>
</div>

It follows from Proposition 26.2 that the sequence of cochain complexes

$$0 \to \Omega^*(M) \xrightarrow{i} \Omega^*(U) \oplus \Omega^*(V) \xrightarrow{j} \Omega^*(U \cap V) \to 0$$

is short exact. By the zig-zag lemma (Theorem 25.6), this short exact sequence of cochain complexes gives rise to a long exact sequence in cohomology, called the **Mayer–Vietoris sequence**:

$$\cdots \to H^k(M) \xrightarrow{i^*} H^k(U) \oplus H^k(V) \xrightarrow{j^*} H^k(U \cap V) \xrightarrow{d^*} H^{k+1}(M) \xrightarrow{i^*} \cdots$$

In this sequence $i^*$ and $j^*$ are induced from $i$ and $j$:

$$i^*[\sigma] = [i(\sigma)] = ([\sigma\vert_U], [\sigma\vert_V]) \in H^k(U) \oplus H^k(V),$$

$$j^*([\omega], [\tau]) = [j(\omega, \tau)] = [\tau\vert_{U \cap V} - \omega\vert_{U \cap V}] \in H^k(U \cap V).$$

The connecting homomorphism $d^* \colon H^k(U \cap V) \to H^{k+1}(M)$ is obtained in three steps:

1. Starting with a closed $k$-form $\zeta \in \Omega^k(U \cap V)$ and using a partition of unity $\lbrace \rho_U, \rho_V \rbrace$ subordinate to $\lbrace U, V \rbrace$, one can extend $\rho_U \zeta$ by zero from $U \cap V$ to a $k$-form $\zeta_V$ on $V$ and extend $\rho_V \zeta$ by zero from $U \cap V$ to a $k$-form $\zeta_U$ on $U$. Then $j(-\zeta_U, \zeta_V) = \zeta$.

2. The commutativity of the square for $d$ and $j$ shows that the pair $(-d\zeta_U, d\zeta_V)$ maps to $0$ under $j$. More formally, since $jd = dj$ and since $\zeta$ is a cocycle, $j(-d\zeta_U, d\zeta_V) = dj(-\zeta_U, \zeta_V) = d\zeta = 0$. It follows that the $(k+1)$-forms $-d\zeta_U$ on $U$ and $d\zeta_V$ on $V$ agree on $U \cap V$.

3. Therefore, $-d\zeta_U$ on $U$ and $d\zeta_V$ on $V$ patch together to give a global $(k+1)$-form $\alpha$ on $M$. Diagram-chasing shows that $\alpha$ is closed. We set $d^*[\zeta] = [\alpha] \in H^{k+1}(M)$.

Because $\Omega^k(M) = 0$ for $k \le -1$, the Mayer–Vietoris sequence starts with

$$0 \to H^0(M) \to H^0(U) \oplus H^0(V) \to H^0(U \cap V) \to \cdots$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 26.4</span></p>

In the Mayer–Vietoris sequence, if $U$, $V$, and $U \cap V$ are connected and nonempty, then

(i) $0 \to H^0(M) \to H^0(U) \oplus H^0(V) \to H^0(U \cap V) \to 0$ is exact ($M$ is connected);

(ii) we may start the Mayer–Vietoris sequence with

$$0 \to H^1(M) \xrightarrow{i^*} H^1(U) \oplus H^1(V) \xrightarrow{j^*} H^1(U \cap V) \to \cdots.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**(i)** The connectedness of $M$ follows from a lemma in point-set topology. On a nonempty connected open set, $H^0$ is simply the vector space of constant functions. By the definition of $j^*$, the map

$$j^* \colon H^0(U) \oplus H^0(V) \to H^0(U \cap V)$$

is given by $(u, v) \mapsto v - u$, $u, v \in \mathbb{R}$. This is clearly surjective. The surjectivity of $j^*$ implies that $\operatorname{im} j^* = H^0(U \cap V) = \ker d^*$, from which we conclude that $d^* \colon H^0(U \cap V) \to H^1(M)$ is the zero map. Thus the Mayer–Vietoris sequence starts with

$$0 \to H^0(M) \xrightarrow{i^*} \mathbb{R} \oplus \mathbb{R} \xrightarrow{j^*} \mathbb{R} \xrightarrow{d^*} 0.$$

This short exact sequence shows that $H^0(M) \simeq \operatorname{im} i^* = \ker j^*$. Since $\ker j^* = \lbrace (u, u) \mid v - u = 0 \rbrace \simeq \mathbb{R}$, we have $H^0(M) \simeq \mathbb{R}$, which proves that $M$ is connected.

**(ii)** From (i) we know that $d^* \colon H^0(U \cap V) \to H^1(M)$ is the zero map. Thus, in the Mayer–Vietoris sequence, the sequence of two maps $H^0(U \cap V) \xrightarrow{d^*} H^1(M) \xrightarrow{i^*} H^1(U) \oplus H^1(V)$ may be replaced by $0 \to H^1(M) \xrightarrow{i^*} H^1(U) \oplus H^1(V)$ without affecting exactness. $\square$

</details>
</div>

### 26.2 The Cohomology of the Circle

In Example 24.4 we showed that integration of 1-forms induces an isomorphism of $H^1(S^1)$ with $\mathbb{R}$. Here we apply the Mayer–Vietoris sequence to give an alternative computation of the cohomology of the circle.

Cover the circle with two open arcs $U$ and $V$. The intersection $U \cap V$ is the disjoint union of two open arcs, which we call $A$ and $B$. Since an open arc is diffeomorphic to an open interval and hence to the real line $\mathbb{R}^1$, the cohomology rings of $U$ and $V$ are isomorphic to that of $\mathbb{R}^1$, and the cohomology ring of $U \cap V$ to that of the disjoint union $\mathbb{R}^1 \amalg \mathbb{R}^1$. They fit into the Mayer–Vietoris sequence, arranged in tabular form:

|  | $S^1$ | $U \amalg V$ | $U \cap V$ |
|---|---|---|---|
| $H^2$ | $\to 0$ | $\to 0$ | $\to 0$ |
| $H^1$ | $\xrightarrow{d^*} H^1(S^1)$ | $\to 0$ | $\to 0$ |
| $H^0$ | $0 \to \mathbb{R}$ | $\xrightarrow{i^*} \mathbb{R} \oplus \mathbb{R}$ | $\xrightarrow{j^*} \mathbb{R} \oplus \mathbb{R}$ |

From the exact sequence

$$0 \to \mathbb{R} \xrightarrow{i^*} \mathbb{R} \oplus \mathbb{R} \xrightarrow{j^*} \mathbb{R} \oplus \mathbb{R} \xrightarrow{d^*} H^1(S^1) \to 0$$

and Problem 26.2, we conclude that $\dim H^1(S^1) = 1$. Hence, the cohomology of the circle is

$$H^k(S^1) = \begin{cases} \mathbb{R} & \text{for } k = 0, 1, \\ 0 & \text{otherwise.} \end{cases}$$

By analyzing the maps in the Mayer–Vietoris sequence, it is possible to write down an explicit generator for $H^1(S^1)$. An element of $H^0(U) \oplus H^0(V)$ is an ordered pair $(u, v) \in \mathbb{R} \oplus \mathbb{R}$, representing a constant function $u$ on $U$ and a constant function $v$ on $V$. An element of $H^0(U \cap V) = H^0(A) \oplus H^0(B)$ is an ordered pair $(a, b) \in \mathbb{R} \oplus \mathbb{R}$, representing a constant function $a$ on $A$ and a constant function $b$ on $B$. The induced map $j^*$ is given by

$$j^*(u, v) = (v - u, v - u).$$

The image of $j^*$ is the diagonal $\Delta = \lbrace (a, a) \in \mathbb{R}^2 \rbrace$. Since $H^1(S^1)$ is isomorphic to $\mathbb{R}$, a generator of $H^1(S^1)$ is simply a nonzero element. Moreover, since $d^* \colon H^0(U \cap V) \to H^1(S^1)$ is surjective and $\ker d^* = \operatorname{im} j^* = \Delta$, such a nonzero element in $H^1(S^1)$ is the image under $d^*$ of an element $(a, b) \in H^0(U \cap V) \simeq \mathbb{R}^2$ with $a \ne b$.

Starting with $(a, b) = (1, 0) \in H^0(U \cap V)$, which corresponds to a function $f$ with value 1 on $A$ and 0 on $B$, the connecting homomorphism $d^*$ produces a generator of $H^1(S^1)$: it is a bump 1-form on $S^1$ supported in $A$.

The explicit description of the map $j^*$ also gives another way to compute $H^1(S^1)$, using the first isomorphism theorem:

$$H^1(S^1) = \operatorname{im} d^* \simeq \frac{\mathbb{R} \oplus \mathbb{R}}{\ker d^*} = \frac{\mathbb{R} \oplus \mathbb{R}}{\operatorname{im} j^*} \simeq \frac{\mathbb{R}^2}{\mathbb{R}} \simeq \mathbb{R}.$$

### 26.3 The Euler Characteristic

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Euler Characteristic)</span></p>

If the cohomology vector space $H^k(M)$ of an $n$-manifold $M$ is finite-dimensional for every $k$, we define its **Euler characteristic** to be the alternating sum

$$\chi(M) = \sum_{k=0}^{n} (-1)^k \dim H^k(M).$$

</div>

As a corollary of the Mayer–Vietoris sequence, the Euler characteristic of $U \cup V$ is computable from those of $U$, $V$, and $U \cap V$:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 26.5</span><span class="math-callout__name">(Euler characteristics in terms of an open cover)</span></p>

Suppose a manifold $M$ has an open cover $\lbrace U, V \rbrace$ and the spaces $M$, $U$, $V$, and $U \cap V$ all have finite-dimensional cohomology. By applying Problem 26.2 to the Mayer–Vietoris sequence, prove that

$$\chi(M) - (\chi(U) + \chi(V)) + \chi(U \cap V) = 0.$$

</div>

## §27 Homotopy Invariance

The homotopy axiom is a powerful tool for computing de Rham cohomology. While homotopy is normally defined in the continuous category, since we are primarily interested in smooth manifolds and smooth maps, our notion of homotopy will be *smooth homotopy*. It differs from the usual homotopy in topology only in that all the maps are assumed to be smooth.

### 27.1 Smooth Homotopy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Smooth Homotopy)</span></p>

Let $M$ and $N$ be manifolds. Two $C^\infty$ maps $f, g \colon M \to N$ are **(smoothly) homotopic** if there is a $C^\infty$ map

$$F \colon M \times \mathbb{R} \to N$$

such that $F(x, 0) = f(x)$ and $F(x, 1) = g(x)$ for all $x \in M$. The map $F$ is called a **homotopy** from $f$ to $g$. A homotopy $F$ from $f$ to $g$ can be viewed as a smoothly varying family of maps $\lbrace f_t \colon M \to N \mid t \in \mathbb{R} \rbrace$, where $f_t(x) = F(x, t)$, $x \in M$, such that $f_0 = f$ and $f_1 = g$. If $f$ and $g$ are homotopic, we write $f \sim g$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 27.1</span><span class="math-callout__name">(Straight-line homotopy)</span></p>

Let $f$ and $g$ be $C^\infty$ maps from a manifold $M$ to $\mathbb{R}^n$. Define $F \colon M \times \mathbb{R} \to \mathbb{R}^n$ by

$$F(x, t) = f(x) + t(g(x) - f(x)) = (1-t)f(x) + tg(x).$$

Then $F$ is a homotopy from $f$ to $g$, called the **straight-line homotopy** from $f$ to $g$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 27.2</span><span class="math-callout__name">(Homotopy)</span></p>

Let $M$ and $N$ be manifolds. Prove that homotopy is an equivalence relation on the set of all $C^\infty$ maps from $M$ to $N$.

</div>

### 27.2 Homotopy Type

As usual, $\mathbb{1}_M$ denotes the identity map on a manifold $M$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 27.3</span><span class="math-callout__name">(Homotopy Equivalence)</span></p>

A map $f \colon M \to N$ is a **homotopy equivalence** if it has a **homotopy inverse**, i.e., a map $g \colon N \to M$ such that $g \circ f$ is homotopic to the identity $\mathbb{1}_M$ on $M$ and $f \circ g$ is homotopic to $\mathbb{1}_N$ on $N$:

$$g \circ f \sim \mathbb{1}_M \quad \text{and} \quad f \circ g \sim \mathbb{1}_N.$$

In this case we say that $M$ is **homotopy equivalent** to $N$, or that $M$ and $N$ have the same **homotopy type**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 27.4</span><span class="math-callout__name">(Homotopy type of the punctured plane)</span></p>

Let $i \colon S^1 \to \mathbb{R}^2 - \lbrace \mathbf{0} \rbrace$ be the inclusion map and let $r \colon \mathbb{R}^2 - \lbrace \mathbf{0} \rbrace \to S^1$ be the map $r(x) = x / \lVert x \rVert$. Then $r \circ i$ is the identity map on $S^1$.

We claim that $i \circ r \colon \mathbb{R}^2 - \lbrace \mathbf{0} \rbrace \to \mathbb{R}^2 - \lbrace \mathbf{0} \rbrace$ is homotopic to the identity map. Set

$$F(x,t) = (1-t)^2 x + t^2 \frac{x}{\lVert x \rVert} = \left((1-t)^2 + \frac{t^2}{\lVert x \rVert}\right) x.$$

Then $F(x, t) = 0$ iff $(1-t)^2 = 0$ and $t^2 / \lVert x \rVert = 0$, i.e., $t = 1 = 0$, a contradiction. Therefore $F$ provides a homotopy between the identity map on $\mathbb{R}^2 - \lbrace \mathbf{0} \rbrace$ and $i \circ r$. It follows that $r$ and $i$ are homotopy inverse to each other, and $\mathbb{R}^2 - \lbrace \mathbf{0} \rbrace$ and $S^1$ have the same homotopy type.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 27.5</span></p>

A manifold is **contractible** if it has the homotopy type of a point.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 27.6</span><span class="math-callout__name">(The Euclidean space $\mathbb{R}^n$ is contractible)</span></p>

Let $p$ be a point in $\mathbb{R}^n$, $i \colon \lbrace p \rbrace \to \mathbb{R}^n$ the inclusion map, and $r \colon \mathbb{R}^n \to \lbrace p \rbrace$ the constant map. Then $r \circ i = \mathbb{1}_{\lbrace p \rbrace}$, the identity map on $\lbrace p \rbrace$. The straight-line homotopy provides a homotopy between the constant map $i \circ r \colon \mathbb{R}^n \to \mathbb{R}^n$ and the identity map on $\mathbb{R}^n$:

$$F(x, t) = (1-t)x + t\,r(x) = (1-t)x + tp.$$

Hence, the Euclidean space $\mathbb{R}^n$ and the set $\lbrace p \rbrace$ have the same homotopy type.

</div>

### 27.3 Deformation Retractions

Let $S$ be a submanifold of a manifold $M$, with $i \colon S \to M$ the inclusion map.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 27.7</span><span class="math-callout__name">(Retraction)</span></p>

A **retraction** from $M$ to $S$ is a map $r \colon M \to S$ that restricts to the identity map on $S$; in other words, $r \circ i = \mathbb{1}_S$. If there is a retraction from $M$ to $S$, we say that $S$ is a **retract** of $M$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 27.8</span><span class="math-callout__name">(Deformation Retraction)</span></p>

A **deformation retraction** from $M$ to $S$ is a map $F \colon M \times \mathbb{R} \to M$ such that for all $x \in M$,

- (i) $F(x, 0) = x$,
- (ii) there is a retraction $r \colon M \to S$ such that $F(x, 1) = r(x)$,
- (iii) for all $s \in S$ and $t \in \mathbb{R}$, $F(s, t) = s$.

If there is a deformation retraction from $M$ to $S$, we say that $S$ is a **deformation retract** of $M$.

</div>

Setting $f_t(x) = F(x, t)$, we can think of a deformation retraction as a family of maps $f_t \colon M \to M$ such that $f_0$ is the identity on $M$, $f_1(x) = r(x)$ for some retraction $r \colon M \to S$, and for every $t$ the map $f_t$ restricts to the identity on $S$. Thus, a deformation retraction is a homotopy between the identity map $\mathbb{1}_M$ and $i \circ r$ for a retraction $r \colon M \to S$ such that this homotopy leaves $S$ fixed for all time $t$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 27.9</span></p>

If $S \subset M$ is a deformation retract of $M$, then $S$ and $M$ have the same homotopy type.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $F \colon M \times \mathbb{R} \to M$ be a deformation retraction and let $r(x) = f_1(x) = F(x, 1)$ be the retraction. Because $r$ is a retraction, the composite $S \xrightarrow{i} M \xrightarrow{r} S$ is the identity on $S$. The deformation retraction provides a homotopy $f_1 = i \circ r \sim f_0 = \mathbb{1}_M$. Therefore, $r \colon M \to S$ is a homotopy equivalence, with homotopy inverse $i \colon S \to M$. $\square$

</details>
</div>

### 27.4 The Homotopy Axiom for de Rham Cohomology

We state here the homotopy axiom and derive a few consequences. The proof will be given in Section 29.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 27.10</span><span class="math-callout__name">(Homotopy axiom for de Rham cohomology)</span></p>

Homotopic maps $f_0, f_1 \colon M \to N$ induce the same map $f_0^* = f_1^* \colon H^*(N) \to H^*(M)$ in cohomology.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 27.11</span></p>

If $f \colon M \to N$ is a homotopy equivalence, then the induced map in cohomology

$$f^* \colon H^*(N) \to H^*(M)$$

is an isomorphism.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $g \colon N \to M$ be a homotopy inverse to $f$. Then $g \circ f \sim \mathbb{1}_M$ and $f \circ g \sim \mathbb{1}_N$. By the homotopy axiom,

$$(g \circ f)^* = \mathbb{1}_{H^*(M)}, \quad (f \circ g)^* = \mathbb{1}_{H^*(N)}.$$

By functoriality,

$$f^* \circ g^* = \mathbb{1}_{H^*(M)}, \quad g^* \circ f^* = \mathbb{1}_{H^*(N)}.$$

Therefore, $f^*$ is an isomorphism in cohomology. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 27.12</span></p>

Suppose $S$ is a submanifold of a manifold $M$ and $F$ is a deformation retraction from $M$ to $S$. Let $r \colon M \to S$ be the retraction $r(x) = F(x, 1)$. Then $r$ induces an isomorphism in cohomology

$$r^* \colon H^*(S) \xrightarrow{\sim} H^*(M).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The proof of Proposition 27.9 shows that a retraction $r \colon M \to S$ is a homotopy equivalence. Apply Corollary 27.11. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 27.13</span><span class="math-callout__name">(Poincaré lemma)</span></p>

Since $\mathbb{R}^n$ has the homotopy type of a point, the cohomology of $\mathbb{R}^n$ is

$$H^k(\mathbb{R}^n) = \begin{cases} \mathbb{R} & \text{for } k = 0, \\ 0 & \text{for } k > 0. \end{cases}$$

</div>

More generally, any contractible manifold will have the same cohomology as a point.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 27.14</span><span class="math-callout__name">(Cohomology of a punctured plane)</span></p>

For any $p \in \mathbb{R}^2$, the translation $x \mapsto x - p$ is a diffeomorphism of $\mathbb{R}^2 - \lbrace p \rbrace$ with $\mathbb{R}^2 - \lbrace 0 \rbrace$. Because the punctured plane $\mathbb{R}^2 - \lbrace 0 \rbrace$ and the circle $S^1$ have the same homotopy type (Example 27.4), they have isomorphic cohomology. Hence, $H^k(\mathbb{R}^2 - \lbrace p \rbrace) \simeq H^k(S^1)$ for all $k \ge 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Cohomology of the Möbius band)</span></p>

The central circle of an open Möbius band $M$ is a deformation retract of $M$. Thus, the open Möbius band has the homotopy type of a circle. By the homotopy axiom,

$$H^k(M) = H^k(S^1) = \begin{cases} \mathbb{R} & \text{for } k = 0, 1, \\ 0 & \text{for } k > 1. \end{cases}$$

</div>

## §28 Computation of de Rham Cohomology

With the tools developed so far, we can compute the cohomology of many manifolds. This section is a compendium of some examples.

### 28.1 Cohomology Vector Space of a Torus

Cover a torus $M$ with two open subsets $U$ and $V$. Both $U$ and $V$ are diffeomorphic to a cylinder and therefore have the homotopy type of a circle (Problem 27.3). Similarly, the intersection $U \cap V$ is the disjoint union of two cylinders $A$ and $B$ and has the homotopy type of a disjoint union of two circles. Our knowledge of the cohomology of a circle allows us to fill in many terms in the Mayer–Vietoris sequence:

|  | $M$ | $U \amalg V$ | $U \cap V$ |
|---|---|---|---|
| $H^2$ | $\xrightarrow{d_1^*} H^2(M)$ | $\to 0$ | $\to 0$ |
| $H^1$ | $\xrightarrow{d_0^*} H^1(M)$ | $\xrightarrow{i^*} \mathbb{R} \oplus \mathbb{R}$ | $\xrightarrow{\beta} \mathbb{R} \oplus \mathbb{R}$ |
| $H^0$ | $0 \to \mathbb{R}$ | $\xrightarrow{} \mathbb{R} \oplus \mathbb{R}$ | $\xrightarrow{\alpha} \mathbb{R} \oplus \mathbb{R}$ |

Let $j_U \colon U \cap V \to U$ and $j_V \colon U \cap V \to V$ be the inclusion maps. Recall that $H^0$ of a connected manifold is the vector space of constant functions. If $a \in H^0(U)$ is the constant function with value $a$ on $U$, then $j_U^* a = a\vert_{U \cap V} \in H^0(U \cap V)$ is the constant function with value $a$ on each component of $U \cap V$, that is, $j_U^* a = (a, a)$. Therefore, for $(a, b) \in H^0(U) \oplus H^0(V)$,

$$\alpha(a, b) = b\vert_{U \cap V} - a\vert_{U \cap V} = (b, b) - (a, a) = (b - a, b - a).$$

Similarly, let us describe the map $\beta \colon H^1(U) \oplus H^1(V) \to H^1(U \cap V) = H^1(A) \oplus H^1(B)$. Since $A$ is a deformation retract of $U$, the restriction $H^*(U) \to H^*(A)$ is an isomorphism, so if $\omega_U$ generates $H^1(U)$, then $j_U^* \omega_U$ is a generator of $H^1$ on $A$ and on $B$. Identifying $H^1(U \cap V)$ with $\mathbb{R} \oplus \mathbb{R}$, we write $j_U^* \omega_U = (1, 1)$. Let $\omega_V$ be a generator of $H^1(V)$. The pair of real numbers $(a, b) \in H^1(U) \oplus H^1(V) \simeq \mathbb{R} \oplus \mathbb{R}$ stands for $(a\omega_U, b\omega_V)$. Then

$$\beta(a, b) = j_V^*(b\omega_V) - j_U^*(a\omega_U) = (b, b) - (a, a) = (b - a, b - a).$$

By the exactness of the Mayer–Vietoris sequence,

$$H^2(M) = \operatorname{im} d_1^* \simeq H^1(U \cap V) / \ker d_1^* = H^1(U \cap V) / \operatorname{im} \beta \simeq (\mathbb{R} \oplus \mathbb{R}) / \mathbb{R} \simeq \mathbb{R}.$$

Applying Problem 26.2 (alternating sum of dimensions) to the Mayer–Vietoris sequence, we get

$$1 - 2 + 2 - \dim H^1(M) + 2 - 2 + \dim H^2(M) = 0.$$

Since $\dim H^2(M) = 1$, this gives $\dim H^1(M) = 2$.

As a check, we can also compute $H^1(M)$ from the Mayer–Vietoris sequence using our knowledge of the maps $\alpha$ and $\beta$:

$$H^1(M) \simeq \ker i^* \oplus \operatorname{im} i^* \simeq \operatorname{im} d_0^* \oplus \ker \beta$$

$$\simeq (H^0(U \cap V) / \ker d_0^*) \oplus \ker \beta \simeq ((\mathbb{R} \oplus \mathbb{R}) / \operatorname{im} \alpha) \oplus \mathbb{R} \simeq \mathbb{R} \oplus \mathbb{R}.$$

### 28.2 The Cohomology Ring of a Torus

A torus is the quotient of $\mathbb{R}^2$ by the integer lattice $\Lambda = \mathbb{Z}^2$. The quotient map

$$\pi \colon \mathbb{R}^2 \to \mathbb{R}^2 / \Lambda$$

induces a pullback map on differential forms $\pi^* \colon \Omega^*(\mathbb{R}^2 / \Lambda) \to \Omega^*(\mathbb{R}^2)$. Since $\pi$ is a local diffeomorphism, its differential $\pi_*$ is an isomorphism at each point. By Problem 18.8, $\pi^*$ is an injection.

For $\lambda \in \Lambda$, define $\ell_\lambda \colon \mathbb{R}^2 \to \mathbb{R}^2$ to be translation by $\lambda$: $\ell_\lambda(q) = q + \lambda$, $q \in \mathbb{R}^2$. A differential form $\tilde{\omega}$ on $\mathbb{R}^2$ is said to be **invariant under translation by $\lambda \in \Lambda$** if $\ell_\lambda^* \tilde{\omega} = \tilde{\omega}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 28.1</span></p>

The image of the injection $\pi^* \colon \Omega^*(\mathbb{R}^2 / \Lambda) \to \Omega^*(\mathbb{R}^2)$ is the subspace of differential forms on $\mathbb{R}^2$ invariant under translations by elements of $\Lambda$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For all $q \in \mathbb{R}^2$, $(\pi \circ \ell_\lambda)(q) = \pi(q + \lambda) = \pi(q)$, so $\pi \circ \ell_\lambda = \pi$. By the functoriality of the pullback, $\pi^* = \ell_\lambda^* \circ \pi^*$. Thus, for any $\omega \in \Omega^k(\mathbb{R}^2 / \Lambda)$, $\pi^* \omega$ is invariant under all translations $\ell_\lambda$, $\lambda \in \Lambda$.

Conversely, suppose $\tilde{\omega} \in \Omega^k(\mathbb{R}^2)$ is invariant under translations $\ell_\lambda$ for all $\lambda \in \Lambda$. For $p \in \mathbb{R}^2 / \Lambda$ and $v_1, \ldots, v_k \in T_p(\mathbb{R}^2 / \Lambda)$, define

$$\omega_p(v_1, \ldots, v_k) = \tilde{\omega}_{\bar{p}}(\tilde{v}_1, \ldots, \tilde{v}_k)$$

for any $\bar{p} \in \pi^{-1}(p)$ and $\tilde{v}_1, \ldots, \tilde{v}_k \in T_{\bar{p}}\mathbb{R}^2$ such that $\pi_* \tilde{v}_i = v_i$. Since $\pi_*$ is an isomorphism, $\tilde{v}_1, \ldots, \tilde{v}_k$ are unique. The invariance of $\tilde{\omega}$ ensures that $\omega_p$ is independent of the choice of $\bar{p}$. Hence, $\tilde{\omega} = \pi^* \omega$. $\square$

</details>
</div>

Let $x, y$ be the standard coordinates on $\mathbb{R}^2$. Since for any $\lambda \in \Lambda$, $\ell_\lambda^*(dx) = d(\ell_\lambda^* x) = d(x + \lambda) = dx$, by Proposition 28.1 the 1-form $dx$ on $\mathbb{R}^2$ is $\pi^*$ of a 1-form $\alpha$ on the torus $\mathbb{R}^2 / \Lambda$. Similarly, $dy$ is $\pi^*$ of a 1-form $\beta$ on the torus.

Note that $\pi^*(d\alpha) = d(\pi^* \alpha) = d(dx) = 0$. Since $\pi^*$ is injective, $d\alpha = 0$. Similarly, $d\beta = 0$. Thus, both $\alpha$ and $\beta$ are closed 1-forms on the torus.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 28.2</span></p>

Let $M$ be the torus $\mathbb{R}^2 / \mathbb{Z}^2$. A basis for the cohomology vector space $H^*(M)$ is represented by the forms $1$, $\alpha$, $\beta$, $\alpha \wedge \beta$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $I$ be the closed interval $[0, 1]$, and $i \colon I^2 \hookrightarrow \mathbb{R}^2$ the inclusion map of the closed unit square. The composite map $F = \pi \circ i \colon I^2 \hookrightarrow \mathbb{R}^2 \to \mathbb{R}^2 / \mathbb{Z}^2$ represents the torus $M$ as a parametrized set. Then $F^* \alpha = i^*(\pi^* \alpha) = i^* dx$, the restriction of $dx$ to the square $I^2$. Similarly, $F^* \beta = i^* dy$.

As an integral over a parametrized set,

$$\int_M \alpha \wedge \beta = \int_{F(I^2)} \alpha \wedge \beta = \int_{I^2} F^*(\alpha \wedge \beta) = \int_{I^2} dx \wedge dy = \int_0^1 \int_0^1 dx\,dy = 1.$$

Thus, the closed 2-form $\alpha \wedge \beta$ represents a nonzero cohomology class on $M$. Since $H^2(M) = \mathbb{R}$ by the computation of Subsection 28.1, the cohomology class $[\alpha \wedge \beta]$ is a basis for $H^2(M)$.

Next we show that the cohomology classes of the closed 1-forms $\alpha$, $\beta$ on $M$ constitute a basis for $H^1(M)$. Let $i_1, i_2 \colon I \to \mathbb{R}^2$ be given by $i_1(t) = (t, 0)$, $i_2(t) = (0, t)$. Define two closed curves $C_1$, $C_2$ in $M = \mathbb{R}^2 / \mathbb{Z}^2$ as the images of the maps $c_k = \pi \circ i_k$ ($k = 1, 2$). Moreover,

$$c_1^* \alpha = (\pi \circ i_1)^* \alpha = i_1^* \pi^* \alpha = i_1^* dx = d(i_1^* x) = dt,$$

$$c_1^* \beta = (\pi \circ i_1)^* \beta = i_1^* \pi^* \beta = i_1^* dy = d(i_1^* y) = 0.$$

Similarly, $c_2^* \alpha = 0$ and $c_2^* \beta = dt$. Therefore,

$$\int_{C_1} \alpha = \int_I c_1^* \alpha = \int_0^1 dt = 1, \quad \int_{C_1} \beta = \int_I c_1^* \beta = \int_0^1 0 = 0.$$

In the same way, $\int_{C_2} \alpha = 0$ and $\int_{C_2} \beta = 1$.

Because $\int_{C_1} \alpha \ne 0$ and $\int_{C_2} \beta \ne 0$, neither $\alpha$ nor $\beta$ is exact on $M$. Furthermore, the cohomology classes $[\alpha]$ and $[\beta]$ are linearly independent, for if $[\alpha]$ were a multiple of $[\beta]$, then $\int_{C_1} \alpha$ would have to be a nonzero multiple of $\int_{C_1} \beta = 0$. By Subsection 28.1, $H^1(M)$ is two-dimensional. Hence, $[\alpha]$, $[\beta]$ is a basis for $H^1(M)$. $\square$

</details>
</div>

The ring structure of $H^*(M)$ is clear from this proposition. Abstractly it is the **exterior algebra** on two generators $a$ and $b$ of degree 1:

$$\bigwedge(a, b) := \mathbb{R}[a, b] / (a^2, b^2, ab + ba), \quad \deg a = 1, \quad \deg b = 1.$$

### 28.3 The Cohomology of a Surface of Genus $g$

Using the Mayer–Vietoris sequence to compute the cohomology of a manifold often leads to ambiguities, because there may be several unknown terms in the sequence. We can resolve these ambiguities if we can describe explicitly the maps occurring in the sequence.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 28.3</span></p>

Suppose $p$ is a point in a compact oriented surface $M$ without boundary, and $i \colon C \to M - \lbrace p \rbrace$ is the inclusion of a small circle around the puncture. Then the restriction map

$$i^* \colon H^1(M - \lbrace p \rbrace) \to H^1(C)$$

is the zero map.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

An element $[\omega] \in H^1(M - \lbrace p \rbrace)$ is represented by a closed 1-form $\omega$ on $M - \lbrace p \rbrace$. If $D$ is the open disk in $M$ bounded by the curve $C$, then $M - D$ is a compact oriented surface with boundary $C$. By Stokes's theorem,

$$\int_C i^* \omega = \int_{\partial(M - D)} i^* \omega = \int_{M - D} d\omega = 0,$$

because $d\omega = 0$. Hence, $i^* \colon H^1(M - \lbrace p \rbrace) \to H^1(C)$ is the zero map. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 28.4</span></p>

Let $M$ be a torus, $p$ a point in $M$, and $A$ the punctured torus $M - \lbrace p \rbrace$. The cohomology of $A$ is

$$H^k(A) = \begin{cases} \mathbb{R} & \text{for } k = 0, \\ \mathbb{R}^2 & \text{for } k = 1, \\ 0 & \text{for } k > 1. \end{cases}$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Cover $M$ with two open sets, $A$ and a disk $U$ containing $p$. Since $A$, $U$, and $A \cap U$ are all connected, we may start the Mayer–Vietoris sequence with the $H^1$ term (Proposition 26.4(ii)). With $H^1(M)$ known from Section 28.1, the Mayer–Vietoris sequence begins with

|  | $M$ | $U \amalg A$ | $U \cap A \sim S^1$ |
|---|---|---|---|
| $H^2$ | $\xrightarrow{d_1^*} \mathbb{R}$ | $\to H^2(A)$ | $\to 0$ |
| $H^1$ | $0 \to \mathbb{R} \oplus \mathbb{R}$ | $\xrightarrow{\beta} H^1(A)$ | $\xrightarrow{\alpha} H^1(S^1)$ |

Because $H^1(U) = 0$, the map $\alpha \colon H^1(A) \to H^1(S^1)$ is simply the restriction map $i^*$. By Lemma 28.3, $\alpha = i^* = 0$. Hence,

$$H^1(A) = \ker \alpha = \operatorname{im} \beta \simeq H^1(M) \simeq \mathbb{R} \oplus \mathbb{R}$$

and there is an exact sequence of linear maps

$$0 \to H^1(S^1) \xrightarrow{d_1^*} \mathbb{R} \to H^2(A) \to 0.$$

Since $H^1(S^1) \simeq \mathbb{R}$, it follows that $H^2(A) = 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 28.5</span></p>

The cohomology of a compact orientable surface $\Sigma_2$ of genus 2 is

$$H^k(\Sigma_2) = \begin{cases} \mathbb{R} & \text{for } k = 0, 2, \\ \mathbb{R}^4 & \text{for } k = 1, \\ 0 & \text{for } k > 2. \end{cases}$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Cover $\Sigma_2$ with two open sets $U$ and $V$ such that each is a punctured torus and $U \cap V$ has the homotopy type of $S^1$. Since $U$, $V$, and $U \cap V$ are all connected, the Mayer–Vietoris sequence begins with

|  | $M$ | $U \amalg V$ | $U \cap V \sim S^1$ |
|---|---|---|---|
| $H^2$ | $\to H^2(\Sigma_2)$ | $\to 0$ | $\to 0$ |
| $H^1$ | $0 \to H^1(\Sigma_2)$ | $\to \mathbb{R}^2 \oplus \mathbb{R}^2$ | $\xrightarrow{\alpha} \mathbb{R}$ |

The map $\alpha \colon H^1(U) \oplus H^1(V) \to H^1(S^1)$ is the difference map. By Lemma 28.3, $j_U^* = j_V^* = 0$, so $\alpha = 0$. It then follows from the exactness of the Mayer–Vietoris sequence that

$$H^1(\Sigma_2) \simeq H^1(U) \oplus H^1(V) \simeq \mathbb{R}^4$$

and $H^2(\Sigma_2) \simeq H^1(S^1) \simeq \mathbb{R}$. $\square$

</details>
</div>

A genus-2 surface $\Sigma_2$ can be obtained as the quotient space of an octagon with its edges identified. By cutting $\Sigma_2$ along a circle $e$, the two halves $A$ and $B$ are each a torus minus an open disk, and can each be represented as a pentagon before identification. When $A$ and $B$ are glued together along $e$, we obtain the octagon.

By Lemma 28.3, if $p \in \Sigma_2$ and $i \colon C \to \Sigma_2 - \lbrace p \rbrace$ is a small circle around $p$ in $\Sigma_2$, then the restriction map $i^* \colon H^1(\Sigma_2 - \lbrace p \rbrace) \to H^1(C)$ is the zero map. This allows us to compute inductively the cohomology of a compact orientable surface $\Sigma_g$ of genus $g$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 28.6</span><span class="math-callout__name">(Surface of genus 3)</span></p>

Compute the cohomology vector space of $\Sigma_2 - \lbrace p \rbrace$ and then compute the cohomology vector space of a compact orientable surface $\Sigma_3$ of genus 3.

</div>

## §29 Proof of Homotopy Invariance

In this section we prove the homotopy invariance of de Rham cohomology. If $f \colon M \to N$ is a $C^\infty$ map, the pullback maps on differential forms and on cohomology classes are normally both denoted by $f^*$. To avoid confusion in this proof, we denote the pullback of forms by

$$f^* \colon \Omega^k(N) \to \Omega^k(M)$$

and the induced map in cohomology by

$$f^\# \colon H^k(N) \to H^k(M).$$

The relation between these two maps is $f^\#[\omega] = [f^* \omega]$ for $[\omega] \in H^k(N)$.

### 29.1 Reduction to Two Sections

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 29.1</span><span class="math-callout__name">(Homotopy axiom for de Rham cohomology)</span></p>

Two smoothly homotopic maps $f, g \colon M \to N$ of manifolds induce the same map in cohomology:

$$f^\# = g^\# \colon H^k(N) \to H^k(M).$$

</div>

We first reduce the problem to two special maps $i_0$ and $i_1 \colon M \to M \times \mathbb{R}$, the 0-section and the 1-section, respectively, of the product line bundle $M \times \mathbb{R} \to M$:

$$i_0(x) = (x, 0), \quad i_1(x) = (x, 1).$$

Suppose $F \colon M \times \mathbb{R} \to N$ is a smooth homotopy from $f$ to $g$, so that $F(x, 0) = f(x)$ and $F(x, 1) = g(x)$. Then $F \circ i_0 = f$ and $F \circ i_1 = g$. By the functoriality of the pullback,

$$f^\# = i_0^\# \circ F^\#, \quad g^\# = i_1^\# \circ F^\#.$$

This reduces proving homotopy invariance to the special case $i_0^\# = i_1^\#$.

The two maps $i_0, i_1 \colon M \to M \times \mathbb{R}$ are obviously smoothly homotopic via the identity map $\mathbb{1}_{M \times \mathbb{R}} \colon M \times \mathbb{R} \to M \times \mathbb{R}$.

### 29.2 Cochain Homotopies

The usual method for showing that two cochain maps $\varphi, \psi \colon \mathcal{A} \to \mathcal{B}$ induce the same map in cohomology is to find a linear map $K \colon \mathcal{A} \to \mathcal{B}$ of degree $-1$ such that

$$\varphi - \psi = d \circ K + K \circ d.$$

Such a map $K$ is called a **cochain homotopy** from $\varphi$ to $\psi$. Note that $K$ is not assumed to be a cochain map. If $a$ is a cocycle in $\mathcal{A}$, then

$$\varphi(a) - \psi(a) = dKa + Kda = dKa$$

is a coboundary, so that in cohomology

$$\varphi^\#[a] = [\varphi(a)] = [\psi(a)] = \psi^\#[a].$$

Thus, the existence of a cochain homotopy between $\varphi$ and $\psi$ implies that the induced maps $\varphi^\#$ and $\psi^\#$ in cohomology are equal.

### 29.3 Differential Forms on $M \times \mathbb{R}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Recall that a sum $\sum_\alpha \omega_\alpha$ of $C^\infty$ differential forms on a manifold $M$ is said to be **locally finite** if the collection $\lbrace \operatorname{supp} \omega_\alpha \rbrace$ of supports is locally finite: every point $p$ in $M$ has a neighborhood $V_p$ such that $V_p$ intersects only finitely many of the sets $\operatorname{supp} \omega_\alpha$.

</div>

Let $\pi \colon M \times \mathbb{R} \to M$ be the projection to the first factor. Every $C^\infty$ differential form on $M \times \mathbb{R}$ is a locally finite sum of the following two types of forms:

- **(I)** $f(x, t)\, \pi^* \eta$,
- **(II)** $f(x, t)\, dt \wedge \pi^* \eta$,

where $f(x, t)$ is a $C^\infty$ function on $M \times \mathbb{R}$ and $\eta$ is a $C^\infty$ form on $M$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 29.2</span></p>

Let $U$ be an open subset of a manifold $M$. If a smooth $k$-form $\tau \in \Omega^k(U)$ defined on $U$ has support in a closed subset of $M$ contained in $U$, then $\tau$ can be extended by zero to a smooth $k$-form on $M$.

</div>

### 29.4 A Cochain Homotopy Between $i_0^*$ and $i_1^*$

Fix an atlas $\lbrace (U_\alpha, \phi_\alpha) \rbrace$ for $M$, a $C^\infty$ partition of unity $\lbrace \rho_\alpha \rbrace$ subordinate to $\lbrace U_\alpha \rbrace$, and a collection $\lbrace g_\alpha \rbrace$ of $C^\infty$ functions on $M$ such that $g_\alpha \equiv 1$ on $\operatorname{supp} \rho_\alpha$ and $\operatorname{supp} g_\alpha \subset U_\alpha$. Let $\omega \in \Omega^k(M \times \mathbb{R})$ and let $\omega_\alpha = (\pi^* \rho_\alpha) \omega$. Since $\sum \pi^* \rho_\alpha = 1$,

$$\omega = \sum_\alpha (\pi^* \rho_\alpha) \omega = \sum_\alpha \omega_\alpha.$$

Each $\omega_\alpha$ can be uniquely decomposed into type-I and type-II forms on $\pi^{-1}U_\alpha$, using the local coordinates $\pi^* x^1, \ldots, \pi^* x^n, t$. Extending by zero using $g_\alpha$, this gives a globally defined decomposition.

Define the linear operator

$$K \colon \Omega^*(M \times \mathbb{R}) \to \Omega^{*-1}(M)$$

by the following rules:

- (i) On type-I forms: $K(f \pi^* \eta) = 0$;
- (ii) On type-II forms: $K(f\, dt \wedge \pi^* \eta) = \left(\int_0^1 f(x, t)\, dt\right) \eta$;
- (iii) $K$ is linear over locally finite sums.

Thus,

$$K(\omega) = K\!\left(\sum_\alpha \omega_\alpha\right) = \sum_{\alpha, J} \left(\int_0^1 b_J^\alpha(x, t)\, dt\right) g_\alpha\, dx^J_\alpha.$$

### 29.5 Verification of Cochain Homotopy

We check in this subsection that

$$d \circ K + K \circ d = i_1^* - i_0^*.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 29.3</span></p>

(i) The exterior derivative $d$ is $\mathbb{R}$-linear over locally finite sums.

(ii) Pullback by a $C^\infty$ map is $\mathbb{R}$-linear over locally finite sums.

</div>

By linearity of $K$, $d$, $i_0^*$, and $i_1^*$ over locally finite sums, it suffices to check the equality $d \circ K + K \circ d = i_1^* - i_0^*$ on a coordinate open set $(U \times \mathbb{R}, \pi^* x^1, \ldots, \pi^* x^n, t)$ on $M \times \mathbb{R}$.

**On type-I forms** $f \pi^* \eta$:

$$Kd(f \pi^* \eta) = K\!\left(\frac{\partial f}{\partial t}\, dt \wedge \pi^* \eta + \sum_i \frac{\partial f}{\partial x^i}\, \pi^* dx^i \wedge \pi^* \eta + f\, \pi^* d\eta\right).$$

In the sum on the right-hand side, the second and third terms are type-I forms; they map to 0 under $K$. Thus,

$$Kd(f \pi^* \eta) = K\!\left(\frac{\partial f}{\partial t}\, dt \wedge \pi^* \eta\right) = \left(\int_0^1 \frac{\partial f}{\partial t}\, dt\right) \eta = (f(x, 1) - f(x, 0))\,\eta = (i_1^* - i_0^*)(f \pi^* \eta).$$

Since $dK(f \pi^* \eta) = d(0) = 0$, on type-I forms, $d \circ K + K \circ d = i_1^* - i_0^*$.

**On type-II forms** $f\, dt \wedge \pi^* \eta$:

$$i_1^*(f(x, t)\, dt \wedge \pi^* \eta) = 0$$

because $i_1^* dt = d(i_1^* t) = d(1) = 0$. Similarly, $i_0^*$ also vanishes on type-II forms. Therefore, $d \circ K + K \circ d = 0 = i_1^* - i_0^*$ on type-II forms.

One verifies separately that $dK + Kd = 0$ on type-II forms by a direct computation using the Leibniz rule for $d$ and the fact that differentiation under the integral sign is permissible for $C^\infty$ functions.

This completes the proof that $K$ is a cochain homotopy between $i_0^*$ and $i_1^*$. The existence of the cochain homotopy $K$ proves that the induced maps in cohomology $i_0^\#$ and $i_1^\#$ are equal. As we pointed out in Section 29.1,

$$f^\# = i_0^\# \circ F^\# = i_1^\# \circ F^\# = g^\#.$$

This proves the homotopy axiom for de Rham cohomology (Theorem 29.1). $\square$

# Appendices

# §A Point-Set Topology

Point-set topology, also called "general topology," is concerned with properties that remain invariant under homeomorphisms (continuous maps having continuous inverses). The basic development in the subject took place in the late nineteenth and early twentieth centuries. This appendix is a collection of basic results from point-set topology that are used throughout the book.

## A.1 Topological Spaces

The prototype of a topological space is the Euclidean space $\mathbb{R}^n$. However, Euclidean space comes with many additional structures, such as a metric, coordinates, an inner product, and an orientation, that are extraneous to its topology. The idea behind the definition of a topological space is to discard all those properties of $\mathbb{R}^n$ that have nothing to do with continuous maps, thereby distilling the notion of continuity to its very essence.

In advanced calculus one learns several characterizations of a continuous map, among which is the following: a map $f$ from an open subset of $\mathbb{R}^n$ to $\mathbb{R}^m$ is continuous if and only if the inverse image $f^{-1}(V)$ of any open set $V$ in $\mathbb{R}^m$ is open in $\mathbb{R}^n$. This shows that continuity can be defined solely in terms of open sets.

Recall that in $\mathbb{R}^n$ the *distance* between two points $p$ and $q$ is given by

$$d(p, q) = \left[\sum_{i=1}^n (p^i - q^i)^2\right]^{1/2},$$

and the *open ball* $B(p, r)$ with center $p \in \mathbb{R}^n$ and radius $r > 0$ is the set

$$B(p, r) = \lbrace x \in \mathbb{R}^n \mid d(x, p) < r \rbrace.$$

A set $U$ in $\mathbb{R}^n$ is said to be *open* if for every $p$ in $U$, there is an open ball $B(p, r)$ with center $p$ and radius $r$ such that $B(p, r) \subset U$. It is clear that the union of an arbitrary collection $\lbrace U_\alpha \rbrace$ of open sets is open, but the same need not be true of the intersection of infinitely many open sets.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

The intervals $]-1/n, 1/n[$, $n = 1, 2, 3, \ldots$, are all open in $\mathbb{R}^1$, but their intersection $\bigcap_{n=1}^\infty ]-1/n, 1/n[$ is the singleton set $\lbrace 0 \rbrace$, which is not open.

</div>

What is true is that the intersection of a *finite* collection of open sets in $\mathbb{R}^n$ is open. This leads to the definition of a topology on a set.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.1</span><span class="math-callout__name">(Topology)</span></p>

A **topology** on a set $S$ is a collection $\mathcal{T}$ of subsets containing both the empty set $\varnothing$ and the set $S$ such that $\mathcal{T}$ is closed under arbitrary unions and finite intersections; i.e., if $U_\alpha \in \mathcal{T}$ for all $\alpha$ in an index set A, then $\bigcup_{\alpha \in A} U_\alpha \in \mathcal{T}$ and if $U_1, \ldots, U_n \in \mathcal{T}$, then $\bigcap_{i=1}^n U_i \in \mathcal{T}$.

</div>

The elements of $\mathcal{T}$ are called **open sets** and the pair $(S, \mathcal{T})$ is called a **topological space**. A **neighborhood** of a point $p$ in $S$ is an open set $U$ containing $p$. If $\mathcal{T}_1$ and $\mathcal{T}_2$ are two topologies on a set $S$ and $\mathcal{T}_1 \subset \mathcal{T}_2$, then we say that $\mathcal{T}_1$ is **coarser** than $\mathcal{T}_1$, or that $\mathcal{T}_2$ is **finer** than $\mathcal{T}_1$. A coarser topology has fewer open sets; conversely, a finer topology has more open sets.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Standard topology on $\mathbb{R}^n$)</span></p>

The open subsets of $\mathbb{R}^n$ as we understand them in advanced calculus form a topology on $\mathbb{R}^n$, the **standard topology** of $\mathbb{R}^n$. In this topology a set $U$ is open in $\mathbb{R}^n$ if and only if for every $p \in U$, there is an open ball $B(p, \varepsilon)$ with center $p$ and radius $\varepsilon$ contained in $U$. Unless stated otherwise, $\mathbb{R}^n$ will always have its standard topology.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Trivial and discrete topologies)</span></p>

For any set $S$, the collection $\mathcal{T} = \lbrace \varnothing, S \rbrace$ consisting of the empty set $\varnothing$ and the entire set $S$ is a topology on $S$, sometimes called the **trivial** or **indiscrete topology**. It is the coarsest topology on a set.

For any set $S$, let $\mathcal{T}$ be the collection of all subsets of $S$. Then $\mathcal{T}$ is a topology on $S$, called the **discrete topology**. A **singleton set** is a set with a single element. The discrete topology can also be characterized as the topology in which every singleton subset $\lbrace p \rbrace$ is open. A topological space having the discrete topology is called a **discrete space**. The discrete topology is the finest topology on a set.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma A.2</span><span class="math-callout__name">(Local criterion for openness)</span></p>

Let $S$ be a topological space. A subset $A$ is open in $S$ if and only if for every $p \in A$, there is an open set $V$ such that $p \in V \subset A$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$(\Rightarrow)$ If $A$ is open, we can take $V = A$.

$(\Leftarrow)$ Suppose for every $p \in A$ there is an open set $V_p$ such that $p \in V_p \subset A$. Then

$$A \subset \bigcup_{p \in A} V_p \subset A,$$

so that equality $A = \bigcup_{p \in A} V_p$ holds. As a union of open sets, $A$ is open. $\square$

</details>
</div>

The complement of an open set is called a **closed set**. By de Morgan's laws from set theory, arbitrary intersections and finite unions of closed sets are closed. One may also specify a topology by describing all the closed sets.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

When we say that a topology is *closed* under arbitrary union and finite intersection, the word "closed" has a different meaning from that of a "closed subset."

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.3</span><span class="math-callout__name">(Finite-complement topology on $\mathbb{R}^1$)</span></p>

Let $\mathcal{T}$ be the collection of subsets of $\mathbb{R}^1$ consisting of the empty set $\varnothing$, the line $\mathbb{R}^1$ itself, and the complements of finite sets. Suppose $F_\alpha$ and $F_i$ are finite subsets of $\mathbb{R}^1$ for $\alpha \in$ some index set A and $i = 1, \ldots, n$. By de Morgan's laws,

$$\bigcup_\alpha (\mathbb{R}^1 - F_\alpha) = \mathbb{R}^1 - \bigcap_\alpha F_\alpha \quad \text{and} \quad \bigcap_{i=1}^n (\mathbb{R}^1 - F_i) = \mathbb{R}^1 - \bigcup_{i=1}^n F_i.$$

Since the arbitrary intersection $\bigcap_{\alpha \in A} F_\alpha$ and the finite union $\bigcup_{i=1}^n F_i$ are both finite, $\mathcal{T}$ is closed under arbitrary unions and finite intersections. Thus, $\mathcal{T}$ defines a topology on $\mathbb{R}^1$, called the **finite-complement topology**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.4</span><span class="math-callout__name">(Zariski topology)</span></p>

One well-known topology is the **Zariski topology** from algebraic geometry. Let $K$ be a field and let $S$ be the vector space $K^n$. Define a subset of $K^n$ to be **Zariski closed** if it is the zero set $Z(f_1, \ldots, f_r)$ of finitely many polynomials $f_1, \ldots, f_r$ on $K^n$.

Let $I = (f_1, \ldots, f_r)$ be the ideal generated by $f_1, \ldots, f_r$ in the polynomial ring $K[x_1, \ldots, x_n]$. Then $Z(f_1, \ldots, f_r) = Z(I)$, the zero set of *all* the polynomials in the ideal $I$. For ideals $I_\alpha$, $I$, and $J$ in $K[x_1, \ldots, x_n]$:

$$\bigcap_\alpha Z(I_\alpha) = Z\!\left(\sum_\alpha I_\alpha\right) \quad \text{and} \quad Z(I) \cup Z(J) = Z(IJ).$$

The complement of a Zariski-closed subset of $K^n$ is said to be **Zariski open**. Both the empty set and $K^n$ are Zariski open. It follows that the Zariski-open subsets of $K^n$ form a topology on $K^n$, called the **Zariski topology** on $K^n$. On $\mathbb{R}^1$ (a finite set), the Zariski topology on $\mathbb{R}^1$ is precisely the finite-complement topology of Example A.3.

</div>

## A.2 Subspace Topology

Let $(S, \mathcal{T})$ be a topological space and $A$ a subset of $S$. Define $\mathcal{T}_A$ to be the collection of subsets

$$\mathcal{T}_A = \lbrace U \cap A \mid U \in \mathcal{T} \rbrace.$$

By the distributive property of union and intersection,

$$\bigcup_\alpha (U_\alpha \cap A) = \left(\bigcup_\alpha U_\alpha\right) \cap A \quad \text{and} \quad \bigcap_i (U_i \cap A) = \left(\bigcap_i U_i\right) \cap A,$$

which shows that $\mathcal{T}_A$ is closed under arbitrary unions and finite intersections. Moreover, $\varnothing, A \in \mathcal{T}_A$. So $\mathcal{T}_A$ is a topology on $A$, called the **subspace topology** or the **relative topology** of $A$ in $S$, and elements of $\mathcal{T}_A$ are said to be **open in $A$**. To emphasize the fact that an open set $U$ in $A$ need not be open in $S$, we also say that $U$ is *open relative to* $A$ or *relatively open in* $A$. The subset $A$ of $S$ with the subspace topology $\mathcal{T}_A$ is called a **subspace**.

If $A$ is an open subset of a topological space $S$, then a subset of $A$ is relatively open in $A$ if and only if it is open in $S$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Consider the subset $A = [0, 1]$ of $\mathbb{R}^1$. In the subspace topology, the half-open interval $[0, 1/2[$ is open relative to $A$, because

$$\bigl[0, \tfrac{1}{2}\bigr[ \;=\; \bigl]-\tfrac{1}{2}, \tfrac{1}{2}\bigr[ \;\cap\; A.$$

</div>

## A.3 Bases

It is generally difficult to describe directly all the open sets in a topology $\mathcal{T}$. What one can usually do is to describe a subcollection $\mathcal{B}$ of $\mathcal{T}$ such that any open set is expressible as a union of sets in $\mathcal{B}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.6</span><span class="math-callout__name">(Basis for a topology)</span></p>

A subcollection $\mathcal{B}$ of a topology $\mathcal{T}$ on a topological space $S$ is a **basis** for the topology $\mathcal{T}$ if given an open set $U$ and point $p$ in $U$, there is an open set $B \in \mathcal{B}$ such that $p \in B \subset U$. We also say that $\mathcal{B}$ **generates** the topology $\mathcal{T}$ or that $\mathcal{B}$ is a **basis for the topological space** $S$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

The collection of all open balls $B(p, r)$ in $\mathbb{R}^n$, with $p \in \mathbb{R}^n$ and $r$ a positive real number, is a basis for the standard topology of $\mathbb{R}^n$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.7</span></p>

A collection $\mathcal{B}$ of open sets of $S$ is a basis if and only if every open set in $S$ is a union of sets in $\mathcal{B}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$(\Rightarrow)$ Suppose $\mathcal{B}$ is a basis and $U$ is an open set in $S$. For every $p \in U$, there is a basic open set $B_p \in \mathcal{B}$ such that $p \in B_p \subset U$. Therefore, $U = \bigcup_{p \in U} B_p$.

$(\Leftarrow)$ Suppose every open set in $S$ is a union of open sets in $\mathcal{B}$. Given an open set $U$ and a point $p$ in $U$, since $U = \bigcup_{B_\alpha \in \mathcal{B}} B_\alpha$, there is a $B_\alpha \in \mathcal{B}$ such that $p \in B_\alpha \subset U$. Hence, $\mathcal{B}$ is a basis. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.8</span></p>

A collection $\mathcal{B}$ of subsets of a set $S$ is a basis for some topology $\mathcal{T}$ on $S$ if and only if

- (i) $S$ is the union of all the sets in $\mathcal{B}$, and
- (ii) given any two sets $B_1$ and $B_2 \in \mathcal{B}$ and a point $p \in B_1 \cap B_2$, there is a set $B \in \mathcal{B}$ such that $p \in B \subset B_1 \cap B_2$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$(\Rightarrow)$ (i) follows from Proposition A.7.

(ii) If $\mathcal{B}$ is a basis, then $B_1$ and $B_2$ are open sets and hence so is $B_1 \cap B_2$. By the definition of a basis, there is a $B \in \mathcal{B}$ such that $p \in B \subset B_1 \cap B_2$.

$(\Leftarrow)$ Define $\mathcal{T}$ to be the collection consisting of all sets that are unions of sets in $\mathcal{B}$. Then the empty set $\varnothing$ and the set $S$ are in $\mathcal{T}$ and $\mathcal{T}$ is clearly closed under arbitrary union. To show that $\mathcal{T}$ is closed under finite intersection, let $U = \bigcup_\mu B_\mu$ and $V = \bigcup_\nu B_\nu$ be in $\mathcal{T}$, where $B_\mu, B_\nu \in \mathcal{B}$. Then

$$U \cap V = \left(\bigcup_\mu B_\mu\right) \cap \left(\bigcup_\nu B_\nu\right) = \bigcup_{\mu, \nu} (B_\mu \cap B_\nu).$$

Thus, any $p$ in $U \cap V$ is in $B_\mu \cap B_\nu$ for some $\mu, \nu$. By (ii) there is a set $B_p$ in $\mathcal{B}$ such that $p \in B_p \subset B_\mu \cap B_\nu$. Therefore,

$$U \cap V = \bigcup_{p \in U \cap V} B_p \in \mathcal{T}. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.9</span></p>

Let $\mathcal{B} = \lbrace B_\alpha \rbrace$ be a basis for a topological space $S$, and $A$ a subspace of $S$. Then $\lbrace B_\alpha \cap A \rbrace$ is a basis for $A$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $U'$ be any open set in $A$ and $p \in U'$. By the definition of subspace topology, $U' = U \cap A$ for some open set $U$ in $S$. Since $p \in U \cap A \subset U$, there is a basic open set $B_\alpha$ such that $p \in B_\alpha \subset U$. Then

$$p \in B_\alpha \cap A \subset U \cap A = U',$$

which proves that the collection $\lbrace B_\alpha \cap A \mid B_\alpha \in \mathcal{B} \rbrace$ is a basis for $A$. $\square$

</details>
</div>

## A.4 First and Second Countability

First and second countability of a topological space have to do with the countability of a basis. We say that a point in $\mathbb{R}^n$ is **rational** if all of its coordinates are rational numbers. Let $\mathbb{Q}$ be the set of rational numbers and $\mathbb{Q}^+$ the set of positive rational numbers. From real analysis, it is well known that every open interval in $\mathbb{R}$ contains a rational number.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma A.10</span></p>

Every open set in $\mathbb{R}^n$ contains a rational point.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.11</span></p>

The collection $\mathcal{B}_{\mathrm{rat}}$ of all open balls in $\mathbb{R}^n$ with rational centers and rational radii is a basis for $\mathbb{R}^n$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Given an open set $U$ in $\mathbb{R}^n$ and point $p$ in $U$, there is an open ball $B(p, r')$ with positive real radius $r'$ such that $p \in B(p, r') \subset U$. Take a rational number $r$ in $]0, r'[$. Then $p \in B(p, r) \subset U$. By Lemma A.10, there is a rational point $q$ in the smaller ball $B(p, r/2)$. We claim that

$$p \in B(q, r/2) \subset B(p, r).$$

Since $d(p, q) < r/2$, we have $p \in B(q, r/2)$. Next, if $x \in B(q, r/2)$, then by the triangle inequality,

$$d(x, p) \le d(x, q) + d(q, p) < \frac{r}{2} + \frac{r}{2} = r.$$

So $x \in B(p, r)$. This proves the claim. Because $p \in B(q, r/2) \subset U$, the collection $\mathcal{B}_{\mathrm{rat}}$ of open balls with rational centers and rational radii is a basis for $\mathbb{R}^n$. $\square$

</details>
</div>

Both of the sets $\mathbb{Q}$ and $\mathbb{Q}^+$ are countable. Since the centers of the balls in $\mathcal{B}_{\mathrm{rat}}$ are indexed by $\mathbb{Q}^n$, a countable set, and the radii are indexed by $\mathbb{Q}^+$, also a countable set, the collection $\mathcal{B}_{\mathrm{rat}}$ is countable.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.12</span><span class="math-callout__name">(Second countable)</span></p>

A topological space is said to be **second countable** if it has a countable basis.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.13</span></p>

Proposition A.11 shows that $\mathbb{R}^n$ with its standard topology is second countable. With the discrete topology, $\mathbb{R}^n$ would not be second countable. More generally, any uncountable set with the discrete topology is not second countable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.14</span></p>

A subspace $A$ of a second-countable space $S$ is second countable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By Proposition A.9, if $\mathcal{B} = \lbrace B_i \rbrace$ is a countable basis for $S$, then $\mathcal{B}_A := \lbrace B_i \cap A \rbrace$ is a countable basis for $A$. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.15</span><span class="math-callout__name">(Basis of neighborhoods, first countable)</span></p>

Let $S$ be a topological space and $p$ a point in $S$. A **basis of neighborhoods at $p$** or a **neighborhood basis at $p$** is a collection $\mathcal{B} = \lbrace B_\alpha \rbrace$ of neighborhoods of $p$ such that for any neighborhood $U$ of $p$, there is a $B_\alpha \in \mathcal{B}$ such that $p \in B_\alpha \subset U$. A topological space $S$ is **first countable** if it has a countable basis of neighborhoods at every point $p \in S$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

For $p \in \mathbb{R}^n$, let $B(p, 1/n)$ be the open ball of center $p$ and radius $1/n$ in $\mathbb{R}^n$. Then $\lbrace B(p, 1/n) \rbrace_{n=1}^\infty$ is a neighborhood basis at $p$. Thus, $\mathbb{R}^n$ is first countable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

An uncountable discrete space is first countable but not second countable. Every second-countable space is first countable.

</div>

Suppose $p$ is a point in a first-countable topological space and $\lbrace V_i \rbrace_{i=1}^\infty$ is a countable neighborhood basis at $p$. By taking $U_i = V_1 \cap \cdots \cap V_i$, we obtain a countable descending sequence

$$U_1 \supset U_2 \supset U_3 \supset \cdots$$

that is also a neighborhood basis at $p$. Thus, in the definition of first countability, we may assume that at every point the countable neighborhood basis at the point is a descending sequence of open sets.

## A.5 Separation Axioms

There are various separation axioms for a topological space. The only ones we will need are the Hausdorff condition and normality.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.16</span><span class="math-callout__name">(Hausdorff and normal)</span></p>

A topological space $S$ is **Hausdorff** if given any two distinct points $x, y$ in $S$, there exist disjoint open sets $U, V$ such that $x \in U$ and $y \in V$. A Hausdorff space is **normal** if given any two disjoint closed sets $F, G$ in $S$, there exist disjoint open sets $U, V$ such that $F \subset U$ and $G \subset V$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.17</span></p>

Every singleton set (a one-point set) in a Hausdorff space $S$ is closed.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $x \in S$. For any $y \in S - \lbrace x \rbrace$, by the Hausdorff condition there exist an open set $U \ni x$ and an open set $V \ni y$ such that $U$ and $V$ are disjoint. In particular,

$$y \in V \subset S - U \subset S - \lbrace x \rbrace.$$

By the local criterion for openness (Lemma A.2), $S - \lbrace x \rbrace$ is open. Therefore, $\lbrace x \rbrace$ is closed. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

The Euclidean space $\mathbb{R}^n$ is Hausdorff, for given distinct points $x, y$ in $\mathbb{R}^n$, if $\varepsilon = \frac{1}{2} d(x, y)$, then the open balls $B(x, \varepsilon)$ and $B(y, \varepsilon)$ will be disjoint.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.18</span><span class="math-callout__name">(Zariski topology)</span></p>

Let $S = K^n$ be a vector space of dimension $n$ over a field $K$, endowed with the Zariski topology. Every open set $U$ in $S$ is of the form $S - Z(I)$, where $I$ is an ideal in $K[x_1, \ldots, x_n]$. The open set $U$ is nonempty if and only if $I$ is not the zero ideal. In the Zariski topology any two nonempty open sets intersect: if $U = S - Z(I)$ and $V = S - Z(J)$ are nonempty, then $I$ and $J$ are nonzero ideals and

$$U \cap V = S - (Z(I) \cup Z(J)) = S - Z(IJ),$$

which is nonempty because $IJ$ is not the zero ideal. Therefore, $K^n$ with the Zariski topology is not Hausdorff.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.19</span></p>

Any subspace $A$ of a Hausdorff space $S$ is Hausdorff.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $x$ and $y$ be distinct points in $A$. Since $S$ is Hausdorff, there exist disjoint neighborhoods $U$ and $V$ of $x$ and $y$ respectively in $S$. Then $U \cap A$ and $V \cap A$ are disjoint neighborhoods of $x$ and $y$ respectively in $A$. $\square$

</details>
</div>

## A.6 Product Topology

The **Cartesian product** of two sets $A$ and $B$ is the set $A \times B$ of all ordered pairs $(a, b)$ with $a \in A$ and $b \in B$. Given two topological spaces $X$ and $Y$, consider the collection $\mathcal{B}$ of subsets of $X \times Y$ of the form $U \times V$, with $U$ open in $X$ and $V$ open in $Y$. We will call elements of $\mathcal{B}$ **basic open sets** in $X \times Y$. If $U_1 \times V_1$ and $U_2 \times V_2$ are in $\mathcal{B}$, then

$$(U_1 \times V_1) \cap (U_2 \times V_2) = (U_1 \cap U_2) \times (V_1 \cap V_2),$$

which is also in $\mathcal{B}$. From this, it follows easily that $\mathcal{B}$ satisfies the conditions of Proposition A.8 for a basis and generates a topology on $X \times Y$, called the **product topology**. Unless noted otherwise, this will always be the topology we assign to the product of two topological spaces.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.20</span></p>

Let $\lbrace U_i \rbrace$ and $\lbrace V_j \rbrace$ be bases for the topological spaces $X$ and $Y$, respectively. Then $\lbrace U_i \times V_j \rbrace$ is a basis for $X \times Y$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Given an open set $W$ in $X \times Y$ and point $(x, y) \in W$, we can find a basic open set $U \times V$ in $X \times Y$ such that $(x, y) \in U \times V \subset W$. Since $U$ is open in $X$ and $\lbrace U_i \rbrace$ is a basis for $X$,

$$x \in U_i \subset U$$

for some $U_i$. Similarly,

$$y \in V_j \subset V$$

for some $V_j$. Therefore,

$$(x, y) \in U_i \times V_j \subset U \times V \subset W.$$

By the definition of a basis, $\lbrace U_i \times V_j \rbrace$ is a basis for $X \times Y$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary A.21</span></p>

The product of two second-countable spaces is second countable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.22</span></p>

The product of two Hausdorff spaces $X$ and $Y$ is Hausdorff.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Given two distinct points $(x_1, y_1), (x_2, y_2)$ in $X \times Y$, without loss of generality we may assume that $x_1 \neq x_2$. Since $X$ is Hausdorff, there exist disjoint open sets $U_1, U_2$ in $X$ such that $x_1 \in U_1$ and $x_2 \in U_2$. Then $U_1 \times Y$ and $U_2 \times Y$ are disjoint neighborhoods of $(x_1, y_1)$ and $(x_2, y_2)$, so $X \times Y$ is Hausdorff. $\square$

</details>
</div>

The product topology can be generalized to the product of an arbitrary collection $\lbrace X_\alpha \rbrace_{\alpha \in A}$ of topological spaces. The projection maps $\pi_{\alpha_i} \colon \prod_\alpha X_\alpha \to X_{\alpha_i}$, $\pi_{\alpha_i}(\prod x_\alpha) = x_{\alpha_i}$ should all be continuous. Thus, for each open set $U_{\alpha_i}$ in $X_{\alpha_i}$, the inverse image $\pi_{\alpha_i}^{-1}(U_{\alpha_i})$ should be open in $\prod_\alpha X_\alpha$. By the properties of open sets, a finite intersection $\bigcap_{i=1}^l \pi_{\alpha_i}^{-1}(U_{\alpha_i})$ should also be open. Such a finite intersection is a set of the form $\prod_{\alpha \in A} U_\alpha$, where $U_\alpha$ is open in $X_\alpha$ and $U_\alpha = X_\alpha$ for all but finitely many $\alpha \in A$. We define the **product topology** on the Cartesian product $\prod_{\alpha \in A} X_\alpha$ to be the topology with basis consisting of sets of this form. The product topology is the coarsest topology on $\prod_\alpha X_\alpha$ such that all the projection maps $\pi_{\alpha_i} \colon \prod_\alpha X_\alpha \to X_{\alpha_i}$ are continuous.

## A.7 Continuity

Let $f \colon X \to Y$ be a function of topological spaces. Mimicking the definition from advanced calculus, we say that $f$ is **continuous at a point $p$** in $X$ if for every neighborhood $V$ of $f(p)$ in $Y$, there is a neighborhood $U$ of $p$ in $X$ such that $f(U) \subset V$. We say that $f$ is **continuous on $X$** if it is continuous at every point of $X$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.23</span><span class="math-callout__name">(Continuity in terms of open sets)</span></p>

A function $f \colon X \to Y$ is continuous if and only if the inverse image of any open set is open.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$(\Rightarrow)$ Suppose $V$ is open in $Y$. To show that $f^{-1}(V)$ is open in $X$, let $p \in f^{-1}(V)$. Then $f(p) \in V$. Since $f$ is assumed to be continuous at $p$, there is a neighborhood $U$ of $p$ such that $f(U) \subset V$. Therefore, $p \in U \subset f^{-1}(V)$. By the local criterion for openness (Lemma A.2), $f^{-1}(V)$ is open in $X$.

$(\Leftarrow)$ Let $p$ be a point in $X$, and $V$ a neighborhood of $f(p)$ in $Y$. By hypothesis, $f^{-1}(V)$ is open in $X$. Since $f(p) \in V$, $p \in f^{-1}(V)$. Then $U = f^{-1}(V)$ is a neighborhood of $p$ such that $f(U) = f(f^{-1}(V)) \subset V$, so $f$ is continuous at $p$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.24</span><span class="math-callout__name">(Continuity of an inclusion map)</span></p>

If $A$ is a subspace of $X$, then the inclusion map $i \colon A \to X$, $i(a) = a$, is continuous. If $U$ is open in $X$, then $i^{-1}(U) = U \cap A$, which is open in the subspace topology of $A$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.25</span><span class="math-callout__name">(Continuity of a projection map)</span></p>

The projection $\pi \colon X \times Y \to X$, $\pi(x, y) = x$, is continuous. Let $U$ be open in $X$. Then $\pi^{-1}(U) = U \times Y$, which is open in the product topology on $X \times Y$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.26</span><span class="math-callout__name">(Composition of continuous maps)</span></p>

The composition of continuous maps is continuous: if $f \colon X \to Y$ and $g \colon Y \to Z$ are continuous, then $g \circ f \colon X \to Z$ is continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $V$ be an open subset of $Z$. Then

$$(g \circ f)^{-1}(V) = f^{-1}(g^{-1}(V)),$$

because for any $x \in X$,

$$x \in (g \circ f)^{-1}(V) \iff g(f(x)) \in V \iff f(x) \in g^{-1}(V) \iff x \in f^{-1}(g^{-1}(V)).$$

By Proposition A.23, since $g$ is continuous, $g^{-1}(V)$ is open in $Y$. Similarly, since $f$ is continuous, $f^{-1}(g^{-1}(V))$ is open in $X$. By Proposition A.23 again, $g \circ f \colon X \to Z$ is continuous. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary A.27</span></p>

The restriction $f\vert_A$ of a continuous function $f \colon X \to Y$ to a subspace $A$ is continuous.

</div>

If $A$ is a subspace of $X$ and $f \colon X \to Y$ is a function, the **restriction** of $f$ to $A$,

$$f\vert_A \colon A \to Y,$$

is defined by $(f\vert_A)(a) = f(a)$. With $i \colon A \to X$ being the inclusion map, the restriction $f\vert_A$ is the composite $f \circ i$. Since both $f$ and $i$ are continuous (Example A.24) and the composition of continuous functions is continuous (Proposition A.26), we have the corollary.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.28</span><span class="math-callout__name">(Continuity in terms of closed sets)</span></p>

A function $f \colon X \to Y$ is continuous if and only if the inverse image of any closed set is closed.

</div>

A map $f \colon X \to Y$ is said to be **open** if the image of every open set in $X$ is open in $Y$; similarly, $f \colon X \to Y$ is said to be **closed** if the image of every closed set in $X$ is closed in $Y$.

If $f \colon X \to Y$ is a bijection, then its inverse map $f^{-1} \colon Y \to X$ is defined. In this context, for any subset $V \subset Y$, the notation $f^{-1}(V)$ a priori has two meanings. It can mean either the inverse image of $V$ under the map $f$,

$$f^{-1}(V) = \lbrace x \in X \mid f(x) \in V \rbrace,$$

or the image of $V$ under the map $f^{-1}$,

$$f^{-1}(V) = \lbrace f^{-1}(y) \in X \mid y \in V \rbrace.$$

Fortunately, because $y = f(x)$ if and only if $x = f^{-1}(y)$, these two meanings coincide.

A continuous bijection $f \colon X \to Y$ whose inverse is also continuous is called a **homeomorphism**.

## A.8 Compactness

While its definition may not be intuitive, the notion of compactness is of central importance in topology. Let $S$ be a topological space. A collection $\lbrace U_\alpha \rbrace$ of open subsets of $S$ is said to **cover** $S$ or to be an **open cover** of $S$ if $S \subset \bigcup_\alpha U_\alpha$. Of course, because $S$ is the ambient space, this condition is equivalent to $S = \bigcup_\alpha U_\alpha$. A **subcover** of an open cover is a subcollection whose union still contains $S$. The topological space $S$ is said to be **compact** if every open cover of $S$ has a finite subcover.

With the subspace topology, a subset $A$ of a topological space $S$ is a topological space in its own right. The subspace $A$ can be covered by open sets in $A$ or by open sets in $S$. An **open cover of $A$ in $S$** is a collection $\lbrace U_\alpha \rbrace$ of open sets in $S$ that covers $A$. In this terminology, $A$ is compact if and only if every open cover of $A$ in $A$ has a finite subcover.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.29</span></p>

A subspace $A$ of a topological space $S$ is compact if and only if every open cover of $A$ in $S$ has a finite subcover.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$(\Rightarrow)$ Assume $A$ compact and let $\lbrace U_\alpha \rbrace$ be an open cover of $A$ in $S$. This means that $A \subset \bigcup_\alpha U_\alpha$. Hence,

$$A \subset \left(\bigcup_\alpha U_\alpha\right) \cap A = \bigcup_\alpha (U_\alpha \cap A).$$

Since $A$ is compact, the open cover $\lbrace U_\alpha \cap A \rbrace$ has a finite subcover $\lbrace U_{\alpha_i} \cap A \rbrace_{i=1}^r$. Thus,

$$A \subset \bigcup_{i=1}^r (U_{\alpha_i} \cap A) \subset \bigcup_{i=1}^r U_{\alpha_i},$$

which means that $\lbrace U_{\alpha_i} \rbrace_{i=1}^r$ is a finite subcover of $\lbrace U_\alpha \rbrace$.

$(\Leftarrow)$ Suppose every open cover of $A$ in $S$ has a finite subcover, and let $\lbrace V_\alpha \rbrace$ be an open cover of $A$ in $A$. Then each $V_\alpha$ is equal to $U_\alpha \cap A$ for some open set $U_\alpha$ in $S$. Since

$$A \subset \bigcup_\alpha V_\alpha \subset \bigcup_\alpha U_\alpha,$$

by hypothesis there are finitely many sets $U_{\alpha_i}$ such that $A \subset \bigcup_i U_{\alpha_i}$. Hence,

$$A \subset \left(\bigcup_i U_{\alpha_i}\right) \cap A = \bigcup_i (U_{\alpha_i} \cap A) = \bigcup_i V_{\alpha_i}.$$

So $\lbrace V_{\alpha_i} \rbrace$ is a finite subcover of $\lbrace V_\alpha \rbrace$ that covers $A$. Therefore, $A$ is compact. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.30</span></p>

A closed subset $F$ of a compact topological space $S$ is compact.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\lbrace U_\alpha \rbrace$ be an open cover of $F$ in $S$. The collection $\lbrace U_\alpha, S - F \rbrace$ is then an open cover of $S$. By the compactness of $S$, there is a finite subcover $\lbrace U_{\alpha_i}, S - F \rbrace$ that covers $S$, so $F \subset \bigcup_i U_{\alpha_i}$. This proves that $F$ is compact. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.31</span></p>

In a Hausdorff space $S$, it is possible to separate a compact subset $K$ and a point $p$ not in $K$ by disjoint open sets; i.e., there exist an open set $U \supset K$ and an open set $V \ni p$ such that $U \cap V = \varnothing$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the Hausdorff property, for every $x \in K$, there are disjoint open sets $U_x \ni x$ and $V_x \ni p$. The collection $\lbrace U_x \rbrace_{x \in K}$ is a cover of $K$ by open subsets of $S$. Since $K$ is compact, it has a finite subcover $\lbrace U_{x_i} \rbrace$.

Let $U = \bigcup_i U_{x_i}$ and $V = \bigcap_i V_{x_i}$. Then $U$ is an open set of $S$ containing $K$. Being the intersection of finitely many open sets containing $p$, $V$ is an open set containing $p$. Moreover, the set

$$U \cap V = \bigcup_i (U_{x_i} \cap V)$$

is empty, since each $U_{x_i} \cap V$ is contained in $U_{x_i} \cap V_{x_i}$, which is empty. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.32</span></p>

Every compact subset $K$ of a Hausdorff space $S$ is closed.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By Proposition A.31, for every point $p$ in $S - K$, there is an open set $V$ such that $p \in V \subset S - K$. This proves that $S - K$ is open. Hence, $K$ is closed. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.34</span></p>

The image of a compact set under a continuous map is compact.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $f \colon X \to Y$ be a continuous map and $K$ a compact subset of $X$. Suppose $\lbrace U_\alpha \rbrace$ is a cover of $f(K)$ by open subsets of $Y$. Since $f$ is continuous, the inverse images $f^{-1}(U_\alpha)$ are all open. Moreover,

$$K \subset f^{-1}(f(K)) \subset f^{-1}\!\left(\bigcup_\alpha U_\alpha\right) = \bigcup_\alpha f^{-1}(U_\alpha).$$

So $\lbrace f^{-1}(U_\alpha) \rbrace$ is an open cover of $K$ in $X$. By the compactness of $K$, there is a finite subcollection $\lbrace f^{-1}(U_{\alpha_i}) \rbrace$ such that

$$K \subset \bigcup_i f^{-1}(U_{\alpha_i}) = f^{-1}\!\left(\bigcup_i U_{\alpha_i}\right).$$

Then $f(K) \subset \bigcup_i U_{\alpha_i}$. Thus, $f(K)$ is compact. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.35</span></p>

A continuous map $f \colon X \to Y$ from a compact space $X$ to a Hausdorff space $Y$ is a closed map.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $F$ be a closed subset of the compact space $X$. By Proposition A.30, $F$ is compact. As the image of a compact set under a continuous map, $f(F)$ is compact in $Y$ (Proposition A.34). As a compact subset of the Hausdorff space $Y$, $f(F)$ is closed (Proposition A.32). $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary A.36</span></p>

A continuous bijection $f \colon X \to Y$ from a compact space $X$ to a Hausdorff space $Y$ is a homeomorphism.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By Proposition A.28, to show that $f^{-1} \colon Y \to X$ is continuous, it suffices to prove that for every closed set $F$ in $X$, the set $(f^{-1})^{-1}(F) = f(F)$ is closed in $Y$, i.e., that $f$ is a closed map. The corollary then follows from Proposition A.35. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.38</span><span class="math-callout__name">(The Tychonoff theorem)</span></p>

The product of any collection of compact spaces is compact in the product topology.

</div>

## A.9 Boundedness in $\mathbb{R}^n$

A subset $A$ of $\mathbb{R}^n$ is said to be **bounded** if it is contained in some open ball $B(p, r)$; otherwise, it is **unbounded**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.39</span></p>

A compact subset of $\mathbb{R}^n$ is bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $A$ were an unbounded subset of $\mathbb{R}^n$, then the collection $\lbrace B(0, i) \rbrace_{i=1}^\infty$ of open balls with radius increasing to infinity would be an open cover of $A$ in $\mathbb{R}^n$ that does not have a finite subcover. $\square$

</details>
</div>

By Propositions A.39 and A.32, a compact subset of $\mathbb{R}^n$ is closed and bounded. The converse is also true.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.40</span><span class="math-callout__name">(The Heine–Borel theorem)</span></p>

A subset of $\mathbb{R}^n$ is compact if and only if it is closed and bounded.

</div>

## A.10 Connectedness

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.41</span><span class="math-callout__name">(Connectedness)</span></p>

A topological space $S$ is **disconnected** if it is the union $S = U \cup V$ of two disjoint nonempty open subsets $U$ and $V$. It is **connected** if it is not disconnected. A subset $A$ of $S$ is **disconnected** if it is disconnected in the subspace topology.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.42</span></p>

A subset $A$ of a topological space $S$ is disconnected if and only if there are open sets $U$ and $V$ in $S$ such that

- (i) $U \cap A \neq \varnothing$, $V \cap A \neq \varnothing$,
- (ii) $U \cap V \cap A = \varnothing$,
- (iii) $A \subset U \cup V$.

A pair of open sets in $S$ with these properties is called a **separation** of $A$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.43</span></p>

The image of a connected space $X$ under a continuous map $f \colon X \to Y$ is connected.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose $f(X)$ is not connected. Then there is a separation $\lbrace U, V \rbrace$ of $f(X)$ in $Y$. By the continuity of $f$, both $f^{-1}(U)$ and $f^{-1}(V)$ are open in $X$. We claim that $\lbrace f^{-1}(U), f^{-1}(V) \rbrace$ is a separation of $X$.

- (i) Since $U \cap f(X) \neq \varnothing$, the open set $f^{-1}(U)$ is nonempty.
- (ii) If $x \in f^{-1}(U) \cap f^{-1}(V)$, then $f(x) \in U \cap V \cap f(X) = \varnothing$, a contradiction. Hence, $f^{-1}(U) \cap f^{-1}(V)$ is empty.
- (iii) Since $f(X) \subset U \cup V$, we have $X \subset f^{-1}(U \cup V) = f^{-1}(U) \cup f^{-1}(V)$.

The existence of a separation of $X$ contradicts the connectedness of $X$. This contradiction proves that $f(X)$ is connected. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.44</span></p>

In a topological space $S$, the union of a collection of connected subsets $A_\alpha$ having a point $p$ in common is connected.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose $\bigcup_\alpha A_\alpha = U \cup V$, where $U$ and $V$ are disjoint open subsets of $\bigcup_\alpha A_\alpha$. The point $p \in \bigcup_\alpha A_\alpha$ belongs to $U$ or $V$. Assume without loss of generality that $p \in U$.

For each $\alpha$,

$$A_\alpha = A_\alpha \cap (U \cup V) = (A_\alpha \cap U) \cup (A_\alpha \cap V).$$

The two open sets $A_\alpha \cap U$ and $A_\alpha \cap V$ of $A_\alpha$ are clearly disjoint. Since $p \in A_\alpha \cap U$, $A_\alpha \cap U$ is nonempty. By the connectedness of $A_\alpha$, $A_\alpha \cap V$ must be empty for all $\alpha$. Hence,

$$V = \left(\bigcup_\alpha A_\alpha\right) \cap V = \bigcup_\alpha (A_\alpha \cap V)$$

is empty. So $\bigcup_\alpha A_\alpha$ must be connected. $\square$

</details>
</div>

## A.11 Connected Components

Let $x$ be a point in a topological space $S$. By Proposition A.44, the union $C_x$ of all connected subsets of $S$ containing $x$ is connected. It is called the **connected component** of $S$ containing $x$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.45</span></p>

Let $C_x$ be a connected component of a topological space $S$. Then a connected subset $A$ of $S$ is either disjoint from $C_x$ or is contained entirely in $C_x$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $A$ and $C_x$ have a point in common, then by Proposition A.44, $A \cup C_x$ is a connected set containing $x$. Hence, $A \cup C_x \subset C_x$, which implies that $A \subset C_x$. $\square$

</details>
</div>

Accordingly, the connected component $C_x$ is the largest connected subset of $S$ containing $x$ in the sense that it contains every connected subset of $S$ containing $x$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary A.46</span></p>

For any two points $x, y$ in a topological space $S$, the connected components $C_x$ and $C_y$ either are disjoint or coincide.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $C_x$ and $C_y$ are not disjoint, then by Proposition A.45, they are contained in each other. In this case, $C_x = C_y$. $\square$

</details>
</div>

As a consequence of Corollary A.46, the connected components of $S$ partition $S$ into disjoint subsets.

## A.12 Closure

Let $S$ be a topological space and $A$ a subset of $S$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.47</span><span class="math-callout__name">(Closure)</span></p>

The **closure** of $A$ in $S$, denoted by $\overline{A}$, $\operatorname{cl}(A)$, or $\operatorname{cl}_S(A)$, is defined to be the intersection of all the closed sets containing $A$.

</div>

The advantage of the bar notation $\overline{A}$ is its simplicity, while the advantage of the $\operatorname{cl}_S(A)$ notation is its indication of the ambient space $S$. If $A \subset B \subset S$, then the closure of $A$ in $B$ and the closure of $A$ in $S$ need not be the same. In this case, it is useful to have the notations $\operatorname{cl}_B(A)$ and $\operatorname{cl}_M(A)$ for the two closures.

As an intersection of closed sets, $\overline{A}$ is a closed set. It is the smallest closed set containing $A$ in the sense that any closed set containing $A$ contains $\overline{A}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

The closure of the open disk $B(\mathbf{0}, r)$ in $\mathbb{R}^2$ is the closed disk

$$\overline{B(\mathbf{0}, r)} = \lbrace p \in \mathbb{R}^2 \mid d(p, \mathbf{0}) \le r \rbrace.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.48</span><span class="math-callout__name">(Local characterization of closure)</span></p>

Let $A$ be a subset of a topological space $S$. A point $p \in S$ is in the closure $\operatorname{cl}(A)$ if and only if every neighborhood of $p$ contains a point of $A$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We will prove the proposition in the form of its contrapositive:

$$p \notin \operatorname{cl}(A) \iff \text{there is a neighborhood of } p \text{ disjoint from } A.$$

$(\Rightarrow)$ Suppose

$$p \notin \operatorname{cl}(A) = \bigcap \lbrace F \text{ closed in } S \mid F \supset A \rbrace.$$

Then $p \notin$ some closed set $F$ containing $A$. It follows that $p \in S - F$, an open set disjoint from $A$.

$(\Leftarrow)$ Suppose $p \in$ an open set $U$ disjoint from $A$. Then the complement $F := S - U$ is a closed set containing $A$ and not containing $p$. Therefore, $p \notin \operatorname{cl}(A)$. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.49</span><span class="math-callout__name">(Accumulation point, limit point)</span></p>

A point $p$ in $S$ is an **accumulation point** of $A$ if every neighborhood of $p$ in $S$ contains a point of $A$ other than $p$. The set of all accumulation points of $A$ is denoted by $\operatorname{ac}(A)$.

If $U$ is a neighborhood of $p$ in $S$, we call $U - \lbrace p \rbrace$ a **deleted neighborhood** of $p$. An equivalent condition for $p$ to be an accumulation point of $A$ is to require that every deleted neighborhood of $p$ in $S$ contain a point of $A$. In some books an accumulation point is called a **limit point**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

If $A = [0, 1[ \cup \lbrace 2 \rbrace$ in $\mathbb{R}^1$, then the closure of $A$ is $[0, 1] \cup \lbrace 2 \rbrace$, but the set of accumulation points of $A$ is only the closed interval $[0, 1]$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.50</span></p>

Let $A$ be a subset of a topological space $S$. Then

$$\operatorname{cl}(A) = A \cup \operatorname{ac}(A).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$(\supset)$ By definition, $A \subset \operatorname{cl}(A)$. By the local characterization of closure (Proposition A.48), $\operatorname{ac}(A) \subset \operatorname{cl}(A)$. Hence, $A \cup \operatorname{ac}(A) \subset \operatorname{cl}(A)$.

$(\subset)$ Suppose $p \in \operatorname{cl}(A)$. Either $p \in A$ or $p \notin A$. If $p \in A$, then $p \in A \cup \operatorname{ac}(A)$. Suppose $p \notin A$. By Proposition A.48, every neighborhood of $p$ contains a point of $A$, which cannot be $p$, since $p \notin A$. Therefore, every deleted neighborhood of $p$ contains a point of $A$. In this case,

$$p \in \operatorname{ac}(A) \subset A \cup \operatorname{ac}(A).$$

So $\operatorname{cl}(A) \subset A \cup \operatorname{ac}(A)$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.51</span></p>

A set $A$ is closed if and only if $A = \overline{A}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$(\Leftarrow)$ If $A = \overline{A}$, then $A$ is closed because $\overline{A}$ is closed.

$(\Rightarrow)$ Suppose $A$ is closed. Then $A$ is a closed set containing $A$, so that $\overline{A} \subset A$. Because $A \subset \overline{A}$, equality holds. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.52</span></p>

If $A \subset B$ in a topological space $S$, then $\overline{A} \subset \overline{B}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $\overline{B}$ contains $B$, it also contains $A$. As a closed subset of $S$ containing $A$, it contains $\overline{A}$ by definition. $\square$

</details>
</div>

## A.13 Convergence

Let $S$ be a topological space. A **sequence** in $S$ is a map from the set $\mathbb{Z}^+$ of positive integers to $S$. We write a sequence as $\langle x_i \rangle$ or $x_1, x_2, x_3, \ldots$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.54</span><span class="math-callout__name">(Convergence)</span></p>

The sequence $\langle x_i \rangle$ **converges** to $p$ if for every neighborhood $U$ of $p$, there is a positive integer $N$ such that for all $i \ge N$, $x_i \in U$. In this case we say that $p$ is a **limit** of the sequence $\langle x_i \rangle$ and write $x_i \to p$ or $\lim_{i \to \infty} x_i = p$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.55</span><span class="math-callout__name">(Uniqueness of the limit)</span></p>

In a Hausdorff space $S$, if a sequence $\langle x_i \rangle$ converges to $p$ and to $q$, then $p = q$.

</div>

Thus, in a Hausdorff space we may speak of *the* limit of a convergent sequence.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition A.56</span><span class="math-callout__name">(The sequence lemma)</span></p>

Let $S$ be a topological space and $A$ a subset of $S$. If there is a sequence $\langle a_i \rangle$ in $A$ that converges to $p$, then $p \in \operatorname{cl}(A)$. The converse is true if $S$ is first countable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$(\Rightarrow)$ Suppose $a_i \to p$, where $a_i \in A$ for all $i$. By the definition of convergence, every neighborhood $U$ of $p$ contains all but finitely many of the points $a_i$. In particular, $U$ contains a point in $A$. By the local characterization of closure (Proposition A.48), $p \in \operatorname{cl}(A)$.

$(\Leftarrow)$ Suppose $p \in \operatorname{cl}(A)$. Since $S$ is first countable, we can find a countable basis of neighborhoods $\lbrace U_n \rbrace$ at $p$ such that

$$U_1 \supset U_2 \supset \cdots.$$

By the local characterization of closure, in each $U_i$ there is a point $a_i \in A$. We claim that the sequence $\langle a_i \rangle$ converges to $p$. If $U$ is any neighborhood of $p$, then by the definition of a basis of neighborhoods at $p$, there is a $U_N$ such that $p \in U_N \subset U$. For all $i \ge N$, we then have

$$U_i \subset U_N \subset U.$$

Therefore, for all $i \ge N$,

$$a_i \in U_i \subset U.$$

This proves that $\langle a_i \rangle$ converges to $p$. $\square$

</details>
</div>

# §B The Inverse Function Theorem on $\mathbb{R}^n$ and Related Results

This appendix reviews three logically equivalent theorems from real analysis: the inverse function theorem, the implicit function theorem, and the constant rank theorem, which describe the local behavior of a $C^\infty$ map from $\mathbb{R}^n$ to $\mathbb{R}^m$. We will assume the inverse function theorem and from it deduce the other two in the simplest cases. In Section 11 these theorems are applied to manifolds in order to clarify the local behavior of a $C^\infty$ map when the map has maximal rank at a point or constant rank in a neighborhood.

## B.1 The Inverse Function Theorem

A $C^\infty$ map $f \colon U \to \mathbb{R}^n$ defined on an open subset $U$ of $\mathbb{R}^n$ is **locally invertible** or a **local diffeomorphism** at a point $p$ in $U$ if $f$ has a $C^\infty$ inverse in some neighborhood of $p$. The inverse function theorem gives a criterion for a map to be locally invertible. We call the matrix $Jf = [\partial f^i / \partial x^j]$ of partial derivatives of $f$ the **Jacobian matrix** of $f$ and its determinant $\det[\partial f^i / \partial x^j]$ the **Jacobian determinant** of $f$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.1</span><span class="math-callout__name">(Inverse function theorem)</span></p>

Let $f \colon U \to \mathbb{R}^n$ be a $C^\infty$ map defined on an open subset $U$ of $\mathbb{R}^n$. At any point $p$ in $U$, the map $f$ is invertible in some neighborhood of $p$ if and only if the Jacobian determinant $\det[\partial f^i / \partial x^j(p)]$ is not zero.

</div>

Although the inverse function theorem apparently reduces the invertibility of $f$ on an open set to a single number at $p$, because the Jacobian determinant is a continuous function, the nonvanishing of the Jacobian determinant at $p$ is equivalent to its nonvanishing in a neighborhood of $p$.

Since the linear map represented by the Jacobian matrix $Jf(p)$ is the best linear approximation to $f$ at $p$, it is plausible that $f$ is invertible in a neighborhood of $p$ if and only if $Jf(p)$ is also, i.e., if and only if $\det(Jf(p)) \neq 0$.

## B.2 The Implicit Function Theorem

In an equation such as $f(x, y) = 0$, it is often impossible to solve explicitly for one of the variables in terms of the other. If we can show the existence of a function $y = h(x)$, which we may or may not be able to write down explicitly, such that $f(x, h(x)) = 0$, then we say that $f(x, y) = 0$ can be solved **implicitly** for $y$ in terms of $x$. The implicit function theorem provides a sufficient condition on a system of equations $f^i(x^1, \ldots, x^n) = 0$, $i = 1, \ldots, m$, under which *locally* a set of variables can be solved implicitly as $C^\infty$ functions of the other variables.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Consider the equation

$$f(x, y) = x^2 + y^2 - 1 = 0.$$

The solution set is the unit circle in the $xy$-plane. From the picture we see that in a neighborhood of any point other than $(\pm 1, 0)$, $y$ is a function of $x$. Indeed,

$$y = \pm \sqrt{1 - x^2},$$

and either function is $C^\infty$ as long as $x \neq \pm 1$. At $(\pm 1, 0)$, there is no neighborhood on which $y$ is a function of $x$.

On a smooth curve $f(x, y) = 0$ in $\mathbb{R}^2$,

$y$ can be expressed as a function of $x$ in a neighborhood of a point $(a, b)$

$\iff$ the tangent line to $f(x, y) = 0$ at $(a, b)$ is not vertical

$\iff$ the normal vector $\operatorname{grad} f := \langle f_x, f_y \rangle$ to $f(x, y) = 0$ at $(a, b)$ is not horizontal

$\iff$ $f_y(a, b) \neq 0$.

</div>

The implicit function theorem generalizes this condition to higher dimensions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.2</span><span class="math-callout__name">(Implicit function theorem)</span></p>

Let $U$ be an open subset in $\mathbb{R}^n \times \mathbb{R}^m$ and $f \colon U \to \mathbb{R}^m$ a $C^\infty$ map. Write $(x, y) = (x^1, \ldots, x^n, y^1, \ldots, y^m)$ for a point in $U$. At a point $(a, b) \in U$ where $f(a, b) = 0$ and the determinant $\det[\partial f^i / \partial y^j(a, b)]$ is nonzero, there exist a neighborhood $A \times B$ of $(a, b)$ in $U$ and a unique function $h \colon A \to B$ such that in $A \times B \subset U \subset \mathbb{R}^n \times \mathbb{R}^m$,

$$f(x, y) = 0 \iff y = h(x).$$

Moreover, $h$ is $C^\infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (for $n = m = 1$)</summary>

Since $f(x, y)$ is a map from an open set $U$ in $\mathbb{R}^{n+m}$ to $\mathbb{R}^m$, it is natural to extend $f$ to a map $F \colon U \to \mathbb{R}^{n+m}$ by adjoining $x$ to it as the first $n$ components:

$$F(x, y) = (u, v) = (x, f(x, y)).$$

For $n = m = 1$, the Jacobian matrix of $F$ is

$$JF = \begin{bmatrix} 1 & 0 \\ \partial f / \partial x & \partial f / \partial y \end{bmatrix}.$$

At the point $(a, b)$,

$$\det JF(a, b) = \frac{\partial f}{\partial y}(a, b) \neq 0.$$

By the inverse function theorem, there are neighborhoods $U_1$ of $(a, b)$ and $V_1$ of $F(a, b) = (a, 0)$ in $\mathbb{R}^2$ such that $F \colon U_1 \to V_1$ is a diffeomorphism with $C^\infty$ inverse $F^{-1}$. Since $F \colon U_1 \to V_1$ is defined by

$$u = x, \quad v = f(x, y),$$

the inverse map $F^{-1} \colon V_1 \to U_1$ must be of the form

$$x = u, \quad y = g(u, v)$$

for some $C^\infty$ function $g \colon V_1 \to \mathbb{R}$. Thus, $F^{-1}(u, v) = (u, g(u, v))$.

The two compositions $F^{-1} \circ F$ and $F \circ F^{-1}$ give

$$y = g(x, f(x, y)) \quad \text{for all } (x, y) \in U_1,$$
$$v = f(u, g(u, v)) \quad \text{for all } (u, v) \in V_1.$$

If $f(x, y) = 0$, then $y = g(x, 0)$. This suggests that we define $h(x) = g(x, 0)$ for all $x$ such that $(x, 0) \in V_1$. Since $g$ is $C^\infty$ by the inverse function theorem, $h$ is also $C^\infty$.

**Claim.** For $(x, y) \in U_1$ such that $(x, 0) \in V_1$,

$$f(x, y) = 0 \iff y = h(x).$$

*Proof of Claim.* $(\Rightarrow)$ If $f(x, y) = 0$, then $y = g(x, f(x, y)) = g(x, 0) = h(x)$.

$(\Leftarrow)$ If $y = h(x)$ and in $(u, v) = (x, 0)$, then $0 = f(x, g(x, 0)) = f(x, h(x)) = f(x, y)$. $\square$

By the claim, in some neighborhood of $(a, b) \in U_1$, the zero set of $f(x, y)$ is precisely the graph of $h$. To find a product neighborhood of $(a, b)$ as in the statement of the theorem, let $A_1 \times B$ be a neighborhood of $(a, b)$ contained in $U_1$ and let $A = h^{-1}(B) \cap A_1$. Since $h$ is continuous, $A$ is open in the domain of $h$ and hence in $\mathbb{R}^1$. Then $h(A) \subset B$, and

$$A \times B \subset A_1 \times B \subset U_1, \quad \text{and} \quad A \times \lbrace 0 \rbrace \subset V_1.$$

By the claim, for $(x, y) \in A \times B$,

$$f(x, y) = 0 \iff y = h(x).$$

The uniqueness of $h$ follows from the equation $y = g(x, f(x, y)) = g(x, 0) = h(x)$. $\square$

</details>
</div>

Replacing a partial derivative such as $\partial f / \partial y$ with a Jacobian matrix $[\partial f^i / \partial y^j]$, we can prove the general case of the implicit function theorem in exactly the same way. Of course, in the theorem $y^1, \ldots, y^m$ need not be the last $m$ coordinates in $\mathbb{R}^{n+m}$; they can be any set of $m$ coordinates in $\mathbb{R}^{n+m}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.3</span></p>

The implicit function theorem is equivalent to the inverse function theorem.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have already shown, at least for one typical case, that the inverse function theorem implies the implicit function theorem. We now prove the reverse implication.

Assume the implicit function theorem, and let $f \colon U \to \mathbb{R}^n$ be a $C^\infty$ map defined on an open subset $U$ of $\mathbb{R}^n$ such that at some point $p \in U$, the Jacobian determinant $\det[\partial f^i / \partial x^j(p)]$ is nonzero. Finding a local inverse for $y = f(x)$ near $p$ amounts to solving the equation

$$g(x, y) = f(x) - y = 0$$

for $x$ in terms of $y$ near $(p, f(p))$. Note that $\partial g^i / \partial x^j = \partial f^i / \partial x^j$. Hence,

$$\det\!\left[\frac{\partial g^i}{\partial x^j}(p, f(p))\right] = \det\!\left[\frac{\partial f^i}{\partial x^j}(p)\right] \neq 0.$$

By the implicit function theorem, there is a $C^\infty$ function $x = h(y)$ defined in a neighborhood of $f(p)$ in $\mathbb{R}^n$ such that

$$g(x, y) = f(x) - y = 0 \iff x = h(y).$$

Thus, $y = f(x)$,

$$x = h(y) = h(f(x)).$$

Since $y = f(x)$,

$$x = h(y) = h(f(x)).$$

Therefore, $f$ and $h$ are inverse functions defined near $p$ and $f(p)$ respectively. $\square$

</details>
</div>

## B.3 Constant Rank Theorem

Every $C^\infty$ map $f \colon U \to \mathbb{R}^m$ on an open set $U$ of $\mathbb{R}^n$ has a **rank** at each point $p$ in $U$, namely the rank of its Jacobian matrix $[\partial f^i / \partial x^j(p)]$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.4</span><span class="math-callout__name">(Constant rank theorem)</span></p>

If $f \colon \mathbb{R}^n \supset U \to \mathbb{R}^m$ has constant rank $k$ in a neighborhood of a point $p \in U$, then after a suitable change of coordinates near $p$ in $U$ and $f(p)$ in $\mathbb{R}^m$, the map $f$ assumes the form

$$(x^1, \ldots, x^n) \mapsto (x^1, \ldots, x^k, 0, \ldots, 0).$$

More precisely, there are a diffeomorphism $G$ of a neighborhood of $p$ in $U$ sending $p$ to the origin in $\mathbb{R}^n$ and a diffeomorphism $F$ of a neighborhood of $f(p)$ in $\mathbb{R}^m$ sending $f(p)$ to the origin in $\mathbb{R}^m$ such that

$$(F \circ f \circ G)^{-1}(x^1, \ldots, x^n) = (x^1, \ldots, x^k, 0, \ldots, 0).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (for $n = m = 2$, $k = 1$)</summary>

Suppose $f = (f^1, f^2) \colon \mathbb{R}^2 \supset U \to \mathbb{R}^2$ has constant rank 1 in a neighborhood of $p \in U$. By reordering the functions $f^1, f^2$ or the variables $x, y$, we may assume that $\partial f^1 / \partial x(p) \neq 0$. Define $G \colon U \to \mathbb{R}^2$ by

$$G(x, y) = (u, v) = (f^1(x, y), y).$$

The Jacobian matrix of $G$ is

$$JG = \begin{bmatrix} \partial f^1 / \partial x & \partial f^1 / \partial y \\ 0 & 1 \end{bmatrix}.$$

Since $\det JG(p) = \partial f^1 / \partial x(p) \neq 0$, by the inverse function theorem there are neighborhoods $U_1$ of $p \in \mathbb{R}^2$ and $V_1$ of $G(p) \in \mathbb{R}^2$ such that $G \colon U_1 \to V_1$ is a diffeomorphism. By making $U_1$ sufficiently small, we may assume that $f$ has constant rank 1 on $U_1$.

On $V_1$,

$$(u, v) = (G \circ G^{-1})(u, v) = (f^1 \circ G^{-1}, y \circ G^{-1})(u, v).$$

Comparing the first components gives $u = (f^1 \circ G^{-1})(u, v)$. Hence,

$$(f \circ G^{-1})(u, v) = (f^1 \circ G^{-1}, f^2 \circ G^{-1})(u, v) = (u, h(u, v)),$$

where we set $h = f^2 \circ G^{-1}$.

Because $G^{-1}$ is a diffeomorphism and $f$ has constant rank 1 on $U_1$, the composite $f \circ G^{-1}$ has constant rank 1 on $V_1$. Its Jacobian matrix is

$$J(f \circ G^{-1}) = \begin{bmatrix} 1 & 0 \\ \partial h / \partial u & \partial h / \partial v \end{bmatrix}.$$

For this matrix to have constant rank 1, $\partial h / \partial v$ must be identically zero on $V_1$. (Here we are using the fact that $f$ has rank $\le 1$ in a neighborhood of $p$.) Thus, $h$ is a function of $u$ alone and we may write

$$(f \circ G^{-1})(u, v) = (u, h(u)).$$

Finally, let $F \colon \mathbb{R}^2 \to \mathbb{R}^2$ be the change of coordinates $F(x, y) = (x, y - h(x))$. Then

$$(F \circ f \circ G^{-1})(u, v) = F(u, h(u)) = (u, h(u) - h(u)) = (u, 0). \quad \square$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example B.5</span></p>

If a $C^\infty$ map $f \colon \mathbb{R}^n \supset U \to \mathbb{R}^n$ defined on an open subset $U$ of $\mathbb{R}^n$ has nonzero Jacobian determinant $\det(Jf(p))$ at a point $p \in U$, then by continuity it has nonzero Jacobian determinant in a neighborhood of $p$. Therefore, it has constant rank $n$ in a neighborhood of $p$.

</div>

# Appendix C — Existence of a Partition of Unity in General

This appendix contains a proof of Theorem 13.7 on the existence of a $C^\infty$ partition of unity on a general manifold.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma C.1</span></p>

Every manifold $M$ has a countable basis all of whose elements have compact closure.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Start with a countable basis $\mathcal{B}$ for $M$ and consider the subcollection $\mathcal{S}$ of elements in $\mathcal{B}$ that have compact closure. We claim that $\mathcal{S}$ is again a basis.

Given an open subset $U \subset M$ and point $p \in U$, choose a neighborhood $V$ of $p$ such that $V \subset U$ and $V$ has compact closure. This is always possible since $M$ is locally Euclidean. Since $\mathcal{B}$ is a basis, there is an open set $B \in \mathcal{B}$ such that

$$p \in B \subset V \subset U.$$

Then $\overline{B} \subset \overline{V}$. Because $\overline{V}$ is compact, so is the closed subset $\overline{B}$. Hence, $B \in \mathcal{S}$. Since for any open set $U$ and any $p \in U$, we have found a set $B \in \mathcal{S}$ such that $p \in B \subset U$, the collection $\mathcal{S}$ of open sets is a basis. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition C.2</span></p>

Every manifold $M$ has a countable increasing sequence of subsets

$$V_1 \subset \overline{V_1} \subset V_2 \subset \overline{V_2} \subset \cdots,$$

with each $V_i$ open and $\overline{V_i}$ compact, such that $M$ is the union of the $V_i$'s.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By Lemma C.1, $M$ has a countable basis $\lbrace B_i \rbrace_{i=1}^\infty$ with each $\overline{B_i}$ compact. Any basis of $M$ of course covers $M$. Set $V_1 = B_1$. By compactness, $\overline{V_1}$ is covered by finitely many of the $B_i$'s. Define $i_1$ to be the smallest integer $\ge 2$ such that

$$\overline{V_1} \subset B_1 \cup B_2 \cup \cdots \cup B_{i_1}.$$

Suppose open sets $V_1, \ldots, V_m$ have been defined, each with compact closure. As before, by compactness, $\overline{V_m}$ is covered by finitely many of the $B_i$'s. If $i_m$ is the smallest integer $\ge m + 1$ and $\ge i_{m-1}$ such that

$$\overline{V_m} \subset B_1 \cup B_2 \cup \cdots \cup B_{i_m},$$

then we set

$$V_{m+1} = B_1 \cup B_2 \cup \cdots \cup B_{i_m}.$$

Since $\overline{V_{m+1}} \subset \overline{B_1} \cup \overline{B_2} \cup \cdots \cup \overline{B_{i_m}}$ is a closed subset of a compact set, $V_{m+1}$ is compact. Since $i_m \ge m + 1$, $B_{m+1} \subset V_{m+1}$. Thus,

$$M = \bigcup B_i \subset \bigcup V_i \subset M.$$

This proves that $M = \bigcup_{i=1}^\infty V_i$. $\square$

</details>
</div>

Define $V_0$ to be the empty set. For each $i \ge 1$, because $\overline{V_{i+1}} - V_i$ is a closed subset of the compact $\overline{V_{i+1}}$, it is compact. Moreover, it is contained in the open set $V_{i+2} - \overline{V_{i-1}}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13.7</span><span class="math-callout__name">(Existence of a $C^\infty$ partition of unity)</span></p>

Let $\lbrace U_\alpha \rbrace_{\alpha \in A}$ be an open cover of a manifold $M$.

**(i)** There is a $C^\infty$ partition of unity $\lbrace \varphi_k \rbrace_{k=1}^\infty$ with every $\varphi_k$ having compact support such that for each $k$, $\operatorname{supp} \varphi_k \subset U_\alpha$ for some $\alpha \in A$.

**(ii)** If we do not require compact support, then there is a $C^\infty$ partition of unity $\lbrace \rho_\alpha \rbrace$ subordinate to $\lbrace U_\alpha \rbrace$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(i)** Let $\lbrace V_i \rbrace_{i=0}^\infty$ be an open cover of $M$ as in Proposition C.2, with $V_0$ the empty set. The idea of the proof is quite simple. For each $i$, we find finitely many smooth bump functions $\psi_j^i$ on $M$, each with compact support in the open set $V_{i+2} - \overline{V_{i-1}}$ as well as in some $U_\alpha$, such that their sum $\sum_j \psi_j^i$ is positive on the compact set $\overline{V_{i+1}} - V_i$. The collection $\lbrace \operatorname{supp} \psi_j^i \rbrace$ of supports over all $i, j$ will be locally finite. Since the compact sets $\overline{V_{i+1}} - V_i$ cover $M$, the locally finite sum $\psi = \sum_{i,j} \psi_j^i$ will be positive on $M$. Then $\lbrace \psi_j^i / \psi \rbrace$ is a $C^\infty$ partition of unity satisfying the conditions in (i).

More precisely: fix an integer $i \ge 1$. For each $p$ in the compact set $\overline{V_{i+1}} - V_i$, choose an open set $U_\alpha$ containing $p$ from the open cover $\lbrace U_\alpha \rbrace$. Then $p$ is in the open set $U_\alpha \cap (V_{i+2} - \overline{V_{i-1}})$. Let $\psi_p$ be a $C^\infty$ bump function on $M$ that is positive on a neighborhood $W_p$ of $p$ and has support in $U_\alpha \cap (V_{i+2} - \overline{V_{i-1}})$. Since $\operatorname{supp} \psi_p$ is a closed set contained in the compact set $\overline{V_{i+2}}$, it is compact.

The collection $\lbrace W_p \mid p \in \overline{V_{i+1}} - V_i \rbrace$ is an open cover of the compact set $\overline{V_{i+1}} - V_i$, and so there is a finite subcover $\lbrace W_{p_1}, \ldots, W_{p_m} \rbrace$, with associated bump functions $\psi_{p_1}, \ldots, \psi_{p_m}$. We relabel them as $\psi_1^i, \ldots, \psi_{m(i)}^i$ such that:

1. $\psi_j^i > 0$ on $W_j^i$ for $j = 1, \ldots, m(i)$;
2. $W_1^i, \ldots, W_{m(i)}^i$ cover the compact set $\overline{V_{i+1}} - V_i$;
3. $\operatorname{supp} \psi_j^i \subset U_{\alpha_j} \cap (V_{i+2} - \overline{V_{i-1}})$ for some $\alpha_j \in A$;
4. $\operatorname{supp} \psi_j^i$ is compact.

As $i$ runs from 1 to $\infty$, we obtain countably many bump functions $\lbrace \psi_j^i \rbrace$. The collection of their supports, $\lbrace \operatorname{supp} \psi_j^i \rbrace$, is locally finite, since only finitely many of these sets intersect any $V_i$. Indeed, since

$$\operatorname{supp} \psi_j^i \subset V_{\ell+2} - \overline{V_{\ell-1}}$$

for all $\ell$, as soon as $\ell \ge i + 1$,

$$\left(\operatorname{supp} \psi_j^i\right) \cap V_i = \text{the empty set } \varnothing.$$

Any point $p \in M$ is in the compact set $\overline{V_{i+1}} - V_i$ for some $i$, and therefore $p \in W_j^i$ for some $(i, j)$. For this $(i, j)$, $\psi_j^i(p) > 0$. Hence, the sum $\psi := \sum_{i,j} \psi_j^i$ is locally finite and everywhere positive on $M$. We now relabel the countable set $\lbrace \psi_j^i \rbrace$ as $\lbrace \psi_1, \psi_2, \psi_3, \ldots \rbrace$. Define

$$\varphi_k = \frac{\psi_k}{\psi}.$$

Then $\sum \varphi_k = 1$ and

$$\operatorname{supp} \varphi_k = \operatorname{supp} \psi_k \subset U_\alpha$$

for some $\alpha \in A$. So $\lbrace \varphi_k \rbrace$ is a partition of unity with compact support such that for each $k$, $\operatorname{supp} \varphi_k \subset U_\alpha$ for some $\alpha \in A$.

**(ii)** For each $k = 1, 2, \ldots$, let $\tau(k)$ be an index in $A$ such that

$$\operatorname{supp} \varphi_k \subset U_{\tau(k)}$$

as in the preceding paragraph. Group the collection $\lbrace \varphi_k \rbrace$ according to $\tau(k)$ and define

$$\rho_\alpha = \sum_{\tau(k) = \alpha} \varphi_k$$

if there is a $k$ with $\tau(k) = \alpha$; otherwise, set $\rho_\alpha = 0$. Then

$$\sum_{\alpha \in A} \rho_\alpha = \sum_{\alpha \in A} \sum_{\tau(k)=\alpha} \varphi_k = \sum_{k=1}^\infty \varphi_k = 1.$$

By Problem 13.7,

$$\operatorname{supp} \rho_\alpha \subset \bigcup_{\tau(k) = \alpha} \operatorname{supp} \varphi_k \subset U_\alpha.$$

Hence, $\lbrace \rho_\alpha \rbrace$ is a $C^\infty$ partition of unity subordinate to $\lbrace U_\alpha \rbrace$. $\square$

</details>
</div>

# Appendix D — Linear Algebra

This appendix gathers together a few facts from linear algebra used throughout the book, especially in Sections 24 and 25. The quotient vector space is a construction in which one reduces a vector space to a smaller space by identifying a subspace to zero. For a linear map $f \colon V \to W$ of vector spaces, the first isomorphism theorem of linear algebra gives an isomorphism between the quotient space $V / \ker f$ and the image of $f$. We also discuss the direct sum and the direct product of a family of vector spaces, as well as the distinction between an internal and an external direct sum.

## D.1 Quotient Vector Spaces

If $V$ is a vector space and $W$ is a subspace of $V$, a **coset** of $W$ in $V$ is a subset of the form

$$v + W = \lbrace v + w \mid w \in W \rbrace$$

for some $v \in V$.

Two cosets $v + W$ and $v' + W$ are equal if and only if $v' = v + w$ for some $w \in W$, or equivalently, if and only if $v' - v \in W$. This introduces an equivalence relation on $V$:

$$v \sim v' \iff v' - v \in W \iff v + W = v' + W.$$

A coset of $W$ in $V$ is simply an equivalence class under this equivalence relation. Any element of $v + W$ is called a **representative** of the coset $v + W$.

The set $V/W$ of all cosets of $W$ in $V$ is again a vector space, with addition and scalar multiplication defined by

$$(u + W) + (v + W) = (u + v) + W, \quad r(v + W) = rv + W$$

for $u, v \in V$ and $r \in \mathbb{R}$. We call $V/W$ the **quotient vector space** or the **quotient space** of $V$ by $W$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example D.1</span></p>

For $V = \mathbb{R}^2$ and $W$ a line through the origin in $\mathbb{R}^2$, a coset of $W$ in $\mathbb{R}^2$ is a line in $\mathbb{R}^2$ parallel to $W$. (For the purpose of this discussion, two lines in $\mathbb{R}^2$ are *parallel* if and only if they coincide or fail to intersect. This definition differs from the usual one in plane geometry in allowing a line to be parallel to itself.) The quotient space $\mathbb{R}^2/W$ is the collection of lines in $\mathbb{R}^2$ parallel to $W$.

</div>

## D.2 Linear Transformations

Let $V$ and $W$ be vector spaces over $\mathbb{R}$. A map $f \colon V \to W$ is called a **linear transformation**, a **vector space homomorphism**, a **linear operator**, or a **linear map** over $\mathbb{R}$ if for all $u, v \in V$ and $r \in \mathbb{R}$,

$$f(u + v) = f(u) + f(v), \quad f(ru) = rf(u).$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example D.2</span></p>

Let $V = \mathbb{R}^2$ and $W$ a line through the origin in $\mathbb{R}^2$ as in Example D.1. If $L$ is a line through the origin not parallel to $W$, then $L$ will intersect each line in $\mathbb{R}^2$ parallel to $W$ in one and only one point. This one-to-one correspondence

$$L \to \mathbb{R}^2/W, \quad v \mapsto v + W,$$

preserves addition and scalar multiplication, and so is an isomorphism of vector spaces. Thus, in this example the quotient space $\mathbb{R}^2/W$ can be identified with the line $L$.

</div>

If $f \colon V \to W$ is a linear transformation, the **kernel** of $f$ is the set

$$\ker f = \lbrace v \in V \mid f(v) = 0 \rbrace$$

and the **image** of $f$ is the set

$$\operatorname{im} f = \lbrace f(v) \in W \mid v \in V \rbrace.$$

The kernel of $f$ is a subspace of $V$ and the image of $f$ is a subspace of $W$. Hence, one can form the quotient spaces $V / \ker f$ and $W / \operatorname{im} f$. This latter space, $W / \operatorname{im} f$, denoted by $\operatorname{coker} f$, is called the **cokernel** of the linear map $f \colon V \to W$.

For now, denote by $K$ the kernel of $f$. The linear map $f \colon V \to W$ induces a linear map $\tilde{f} \colon V/K \to \operatorname{im} f$, by

$$\tilde{f}(v + K) = f(v).$$

It is easy to check that $\tilde{f}$ is linear and bijective. This gives the following fundamental result of linear algebra.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem D.3</span><span class="math-callout__name">(The first isomorphism theorem)</span></p>

Let $f \colon V \to W$ be a homomorphism of vector spaces. Then $f$ induces an isomorphism

$$\tilde{f} \colon \frac{V}{\ker f} \xrightarrow{\;\sim\;} \operatorname{im} f.$$

</div>

## D.3 Direct Product and Direct Sum

Let $\lbrace V_\alpha \rbrace_{\alpha \in I}$ be a family of real vector spaces. The **direct product** $\prod_\alpha V_\alpha$ is the set of all sequences $(v_\alpha)$ with $v_\alpha \in V_\alpha$ for all $\alpha \in I$, and the **direct sum** $\bigoplus_\alpha V_\alpha$ is the subset of the direct product $\prod_\alpha V_\alpha$ consisting of sequences $(v_\alpha)$ such that $v_\alpha = 0$ for all but finitely many $\alpha \in I$. Under componentwise addition and scalar multiplication,

$$(v_\alpha) + (w_\alpha) = (v_\alpha + w_\alpha), \quad r(v_\alpha) = (rv_\alpha), \quad r \in \mathbb{R},$$

both the direct product $\prod_\alpha V_\alpha$ and the direct sum $\bigoplus_\alpha V_\alpha$ are real vector spaces. When the index set $I$ is finite, the direct sum coincides with the direct product. In particular, for two vector spaces $A$ and $B$,

$$A \oplus B = A \times B = \lbrace (a, b) \mid a \in A \text{ and } b \in B \rbrace.$$

The **sum** of two subspaces $A$ and $B$ of a vector space $V$ is the subspace

$$A + B = \lbrace a + b \in V \mid a \in A,\; b \in B \rbrace.$$

If $A \cap B = \lbrace 0 \rbrace$, this sum is called an **internal direct sum** and written $A \oplus_i B$. In an internal direct sum $A \oplus_i B$, every element has a representation as $a + b$ for a unique $a \in A$ and a unique $b \in B$. Indeed, if $a + b = a' + b' \in A \oplus_i B$, then

$$a - a' = b' - b \in A \cap B = \lbrace 0 \rbrace.$$

Hence, $a = a'$ and $b = b'$.

In contrast to the internal direct sum $A \oplus_i B$, the direct sum $A \oplus B$ is called the **external direct sum**. In fact, the two notions are isomorphic: the natural map

$$\varphi \colon A \oplus B \to A \oplus_i B, \quad (a, b) \mapsto a + b$$

is easily seen to be a linear isomorphism. For this reason, in the literature the internal direct sum is normally denoted by $A \oplus B$, just like the external direct sum.

If $V = A \oplus_i B$, then $A$ is called a **complementary subspace** to $B$ in $V$. In Example D.2, the line $L$ is a complementary subspace to $W$, and we may identify the quotient vector space $\mathbb{R}^2/W$ with any complementary subspace to $W$.

In general, if $W$ is a subspace of a vector space $V$ and $W'$ is a complementary subspace to $W$, then there is a linear map

$$\varphi \colon W' \to V/W, \quad w' \mapsto w' + W.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise D.4</span></p>

Show that $\varphi \colon W' \to V/W$ is an isomorphism of vector spaces.

</div>

Thus, the quotient space $V/W$ may be identified with any complementary subspace to $W$ in $V$. This identification is not canonical, for there are many complementary subspaces to a given subspace $W$ and there is no reason to single out any one of them. However, when $V$ has an inner product $\langle\;,\;\rangle$, one can single out a canonical complementary subspace, the **orthogonal complement** of $W$:

$$W^\perp = \lbrace v \in V \mid \langle v, w \rangle = 0 \text{ for all } w \in W \rbrace.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise D.5</span></p>

Check that $W^\perp$ is a complementary subspace to $W$.

</div>

In this case, there is a canonical identification $W^\perp \xrightarrow{\sim} V/W$.

Let $f \colon V \to W$ be a linear map of finite-dimensional vector spaces. It follows from the first isomorphism theorem and Problem D.1 that

$$\dim V - \dim(\ker f) = \dim(\operatorname{im} f).$$

Since the dimension is the only isomorphism invariant of a vector space, we therefore have the following corollary of the first isomorphism theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary D.6</span></p>

If $f \colon V \to W$ is a linear map of finite-dimensional vector spaces, then there is a vector space isomorphism

$$V \simeq \ker f \oplus \operatorname{im} f.$$

(The right-hand side is an external direct sum because $\ker f$ and $\operatorname{im} f$ are not subspaces of the same vector space.)

</div>

### Problems

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise D.1</span><span class="math-callout__name">(Dimension of a quotient vector space)</span></p>

Prove that if $w_1, \ldots, w_m$ is a basis for $W$ that extends to a basis $w_1, \ldots, w_m, v_1, \ldots, v_n$ for $V$, then $v_1 + W, \ldots, v_n + W$ is a basis for $V/W$. Therefore,

$$\dim V/W = \dim V - \dim W.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise D.2</span><span class="math-callout__name">(Dimension of a direct sum)</span></p>

Prove that if $a_1, \ldots, a_m$ is a basis for a vector space $A$ and $b_1, \ldots, b_n$ is a basis for a vector space $B$, then $(a_i, 0), (0, b_j)$, $i = 1, \ldots, m$, $j = 1, \ldots, n$, is a basis for the direct sum $A \oplus B$. Therefore,

$$\dim A \oplus B = \dim A + \dim B.$$

</div>

# Appendix E — Quaternions and the Symplectic Group

First described by William Rowan Hamilton in 1843, **quaternions** are elements of the form

$$q = a + \mathbf{i}b + \mathbf{j}c + \mathbf{k}d, \quad a, b, c, d \in \mathbb{R},$$

that add componentwise and multiply according to the distributive property and the rules

$$\mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = -1,$$

$$\mathbf{i}\mathbf{j} = \mathbf{k},\; \mathbf{j}\mathbf{k} = \mathbf{i},\; \mathbf{k}\mathbf{i} = \mathbf{j},$$

$$\mathbf{i}\mathbf{j} = -\mathbf{j}\mathbf{i},\; \mathbf{j}\mathbf{k} = -\mathbf{k}\mathbf{j},\; \mathbf{k}\mathbf{i} = -\mathbf{i}\mathbf{k}.$$

A mnemonic for the three rules $\mathbf{i}\mathbf{j} = \mathbf{k}$, $\mathbf{j}\mathbf{k} = \mathbf{i}$, $\mathbf{k}\mathbf{i} = \mathbf{j}$ is that in going clockwise around the cycle $\mathbf{i} \to \mathbf{j} \to \mathbf{k} \to \mathbf{i}$, the product of two successive elements is the next one. Under addition and multiplication, the quaternions satisfy all the properties of a field except the commutative property for multiplication. Such an algebraic structure is called a **skew field** or a **division ring**. In honor of Hamilton, the usual notation for the skew field of quaternions is $\mathbb{H}$.

A division ring that is also an algebra over a field $K$ is called a **division algebra** over $K$. The real and complex fields $\mathbb{R}$ and $\mathbb{C}$ are commutative division algebras over $\mathbb{R}$. By a theorem of Ferdinand Georg Frobenius (1878), the skew field $\mathbb{H}$ of quaternions has the distinction of being the only (associative) division algebra over $\mathbb{R}$ other than $\mathbb{R}$ and $\mathbb{C}$.

One can define vector spaces and formulate linear algebra over a skew field, just as one would for vector spaces over a field. The only difference is that over a skew field it is essential to keep careful track of the order of multiplication. A vector space over $\mathbb{H}$ is called a **quaternionic** vector space. We denote by $\mathbb{H}^n$ the quaternionic vector space of $n$-tuples of quaternions.

## E.1 Representation of Linear Maps by Matrices

Relative to given bases, a linear map of vector spaces over a skew field will also be represented by a matrix. Since maps are written on the left of their arguments as in $f(x)$, we will choose our convention so that a linear map $f$ corresponds to left multiplication by a matrix. In order for a vector in $\mathbb{H}^n$ to be multiplied on the left by a matrix, the elements of $\mathbb{H}^n$ must be column vectors, and for left multiplication by a matrix to be a linear map, scalar multiplication on $\mathbb{H}^n$ should be on the right.

Let $K$ be a skew field and let $V$ and $W$ be vector spaces over $K$, with scalar multiplication on the right. A map $f \colon V \to W$ is **linear** over $K$ or **$K$-linear** if for all $x, y \in V$ and $q \in K$,

$$f(x + y) = f(x) + f(y), \quad f(xq) = f(x)q.$$

An **endomorphism** or a **linear transformation** of a vector space $V$ over $K$ is a $K$-linear map from $V$ to itself. The endomorphisms of $V$ over $K$ form an algebra over $K$, denoted by $\operatorname{End}_K(V)$. An endomorphism $f \colon V \to V$ is **invertible** if it has a two-sided inverse, i.e., a linear map $g \colon V \to V$ such that $f \circ g = g \circ f = 1_V$. An invertible endomorphism of $V$ is also called an **automorphism** of $V$. The **general linear group** $\operatorname{GL}(V)$ is by definition the group of all automorphisms of the vector space $V$. When $V = K^n$, we also write $\operatorname{GL}(n, K)$ for $\operatorname{GL}(V)$.

Let $e_i$ be the column vector with 1 in the $i$th row and 0 everywhere else. The set $e_1, \ldots, e_n$ is called the **standard basis** for $K^n$. If $f \colon K^n \to K^n$ is $K$-linear, then

$$f(e_j) = \sum_i e_i a_j^i$$

for some matrix $A = [a_j^i] \in K^{n \times n}$, called the **matrix** of $f$ (relative to the standard basis). Here $a_j^i$ is the entry in the $i$th row and $j$th column of the matrix $A$. For $x = \sum_j e_j x^j \in K^n$,

$$f(x) = \sum_j f(e_j)x^j = \sum_{i,j} e_i a_j^i x^j.$$

Hence, the $i$th component of the column vector $f(x)$ is

$$(f(x))^i = \sum_j a_j^i x^j.$$

In matrix notation, $f(x) = Ax$.

If $g \colon K^n \to K^n$ is another linear map and $g(e_j) = \sum_i e_i b_j^i$, then

$$(f \circ g)(e_j) = f\!\left(\sum_k e_k b_j^k\right) = \sum_k f(e_k) b_j^k = \sum_{l,k} e_l a_k^l b_j^k.$$

Thus, if $A = [a_j^i]$ and $B = [b_j^i]$ are the matrices representing $f$ and $g$ respectively, then the matrix product $AB$ is the matrix representing the composite $f \circ g$. Therefore, there is an algebra isomorphism

$$\operatorname{End}_K(K^n) \xrightarrow{\;\sim\;} K^{n \times n}$$

between endomorphisms of $K^n$ and $n \times n$ matrices over $K$. Under this isomorphism, the group $\operatorname{GL}(n, K)$ corresponds to the group of all invertible $n \times n$ matrices over $K$.

## E.2 Quaternionic Conjugation

The **conjugate** of a quaternion $q = a + \mathbf{i}b + \mathbf{j}c + \mathbf{k}d$ is defined to be

$$\bar{q} = a - \mathbf{i}b - \mathbf{j}c - \mathbf{k}d.$$

It is easily shown that conjugation is an **antihomomorphism** from the ring $\mathbb{H}$ to itself: it preserves addition, but under multiplication,

$$\overline{pq} = \bar{q}\bar{p} \quad \text{for } p, q \in \mathbb{H}.$$

The **conjugate** of a matrix $A = [a_j^i] \in \mathbb{H}^{m \times n}$ is $\bar{A} = [\overline{a_j^i}]$, obtained by conjugating each entry of $A$. The **transpose** $A^T$ of the matrix $A$ is the matrix whose $(i, j)$-entry is the $(j, i)$-entry of $A$. In contrast to the case for complex matrices, when $A$ and $B$ are quaternion matrices, in general

$$\overline{AB} \neq \bar{A}\bar{B}, \quad \overline{AB} \neq \bar{B}\bar{A}, \quad \text{and} \quad (AB)^T \neq B^T A^T.$$

However, it is true that

$$\overline{AB}^T = \bar{B}^T \bar{A}^T,$$

as one sees by a direct computation.

## E.3 Quaternionic Inner Product

The **quaternionic inner product** on $\mathbb{H}^n$ is defined to be

$$\langle x, y \rangle = \sum_i \overline{x^i} y^i = \bar{x}^T y, \quad x, y \in \mathbb{H}^n,$$

with conjugation on the first argument $x = \langle x^1, \ldots, x^n \rangle$. For any $q \in \mathbb{H}$,

$$\langle xq, y \rangle = \bar{q} \langle x, y \rangle \quad \text{and} \quad \langle x, yq \rangle = \langle x, y \rangle q.$$

If conjugation were on the second argument, then the inner product would not have the correct linearity property with respect to scalar multiplication on the right.

For quaternionic vector spaces $V$ and $W$, we say that a map $f \colon V \times W \to \mathbb{H}$ is **sesquilinear** over $\mathbb{H}$ if it is conjugate-linear on the left in the first argument and linear on the right in the second argument: for all $v \in V$, $w \in W$, and $q \in \mathbb{H}$,

$$f(vq, w) = \bar{q} f(v, w), \quad f(v, wq) = f(v, w)q.$$

In this terminology, the quaternionic inner product is sesquilinear over $\mathbb{H}$.

## E.4 Representations of Quaternions by Complex Numbers

A quaternion can be identified with a pair of complex numbers:

$$q = a + \mathbf{i}b + \mathbf{j}c + \mathbf{k}d = (a + \mathbf{i}b) + \mathbf{j}(c - \mathbf{i}d) = u + \mathbf{j}v \longleftrightarrow (u, v).$$

Thus, $\mathbb{H}$ is a vector space over $\mathbb{C}$ with basis $1, \mathbf{j}$, and $\mathbb{H}^n$ is a vector space over $\mathbb{C}$ with basis $e_1, \ldots, e_n, \mathbf{j}e_1, \ldots, \mathbf{j}e_n$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition E.1</span></p>

Let $q$ be a quaternion and let $u, v$ be complex numbers.

**(i)** If $q = u + \mathbf{j}v$, then $\bar{q} = \bar{u} - \mathbf{j}v$.

**(ii)** $\mathbf{j}u\mathbf{j}^{-1} = \bar{u}$.

</div>

By Proposition E.1(ii), for any complex vector $v \in \mathbb{C}^n$, one has $\mathbf{j}v = \bar{v}\mathbf{j}$. Although elements of $\mathbb{H}^n$ should be written as $u + \mathbf{j}v$, not as $u + v\mathbf{j}$, so that the map $\mathbb{H}^n \to \mathbb{C}^{2n}$, $u + \mathbf{j}v \mapsto (u, v)$, will be a complex vector space isomorphism.

For any quaternion $q = u + \mathbf{j}v$, left multiplication $\ell_q \colon \mathbb{H} \to \mathbb{H}$ by $q$ is $\mathbb{H}$-linear and a fortiori $\mathbb{C}$-linear. Since

$$\ell_q(1) = u + \mathbf{j}v, \quad \ell_q(\mathbf{j}) = (u + \mathbf{j}v)\mathbf{j} = -\bar{v} + \mathbf{j}\bar{u},$$

the matrix of $\ell_q$ as a $\mathbb{C}$-linear map relative to the basis $1, \mathbf{j}$ for $\mathbb{H}$ over $\mathbb{C}$ is the $2 \times 2$ complex matrix $\begin{bmatrix} u & -\bar{v} \\ v & \bar{u} \end{bmatrix}$. The map $\mathbb{H} \to \operatorname{End}_\mathbb{C}(\mathbb{C}^2)$, $q \mapsto \ell_q$ is an injective algebra homomorphism over $\mathbb{R}$, giving rise to a representation of the quaternions by $2 \times 2$ complex matrices.

## E.5 Quaternionic Inner Product in Terms of Complex Components

Let $x = x_1 + \mathbf{j}x_2$ and $y = y_1 + \mathbf{j}y_2$ be in $\mathbb{H}^n$, with $x_1, x_2, y_1, y_2 \in \mathbb{C}^n$. By Proposition E.1,

$$\langle x, y \rangle = \bar{x}^T y = (\bar{x}_1^T - \mathbf{j}x_2^T)(y_1 + \mathbf{j}y_2)$$

$$= (\bar{x}_1^T y_1 + \bar{x}_2^T y_2) + \mathbf{j}(\bar{x}_1^T y_2 - \bar{x}_2^T y_1).$$

Let

$$\langle x, y \rangle_1 = \bar{x}_1^T y_1 + \bar{x}_2^T y_2 = \sum_{i=1}^n \bar{x}_1^i y_1^i + \bar{x}_2^i y_2^i$$

and

$$\langle x, y \rangle_2 = x_1^T y_2 - x_2^T y_1 = \sum_{i=1}^n x_1^i y_2^i - x_2^i y_1^i.$$

So the quaternionic inner product $\langle\;,\;\rangle$ is the sum of a Hermitian inner product and $\mathbf{j}$ times a skew-symmetric bilinear form on $\mathbb{C}^{2n}$:

$$\langle\;,\;\rangle = \langle\;,\;\rangle_1 + \mathbf{j}\langle\;,\;\rangle_2.$$

Let $x = x_1 + \mathbf{j}x_2 \in \mathbb{H}^n$. By skew-symmetry, $\langle x, x \rangle_2 = 0$, so that

$$\langle x, x \rangle = \langle x, x \rangle_1 = \|x_1\|^2 + \|x_2\|^2 \ge 0.$$

The **norm** of a quaternionic vector $x = x_1 + \mathbf{j}x_2$ is defined to be

$$\|x\| = \sqrt{\langle x, x \rangle} = \sqrt{\|x_1\|^2 + \|x_2\|^2}.$$

In particular, the norm of a quaternion $q = a + \mathbf{i}b + \mathbf{j}c + \mathbf{k}d$ is

$$\|q\| = \sqrt{a^2 + b^2 + c^2 + d^2}.$$

## E.6 $\mathbb{H}$-Linearity in Terms of Complex Numbers

Recall that an $\mathbb{H}$-linear map of quaternionic vector spaces is a map that is additive and commutes with right multiplication $r_q$ for any quaternion $q$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition E.2</span></p>

Let $V$ be a quaternionic vector space. A map $f \colon V \to V$ is $\mathbb{H}$-linear if and only if it is $\mathbb{C}$-linear and $f \circ r_\mathbf{j} = r_\mathbf{j} \circ f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$(\Rightarrow)$ Clear.

$(\Leftarrow)$ Suppose $f$ is $\mathbb{C}$-linear and $f$ commutes with $r_\mathbf{j}$. By $\mathbb{C}$-linearity, $f$ is additive and commutes with $r_u$ for any complex number $u$. Any $q \in \mathbb{H}$ can be written as $q = u + \mathbf{j}v$ for some $u, v \in \mathbb{C}$; moreover, $r_q = r_{u+\mathbf{j}v} = r_u + r_v \circ r_\mathbf{j}$ (note the order reversal in $r_{\mathbf{j}v} = r_v \circ r_\mathbf{j}$). Since $f$ is additive and commutes with $r_u$, $r_v$, and $r_\mathbf{j}$, it commutes with $r_q$ for any $q \in \mathbb{H}$. Therefore, $f$ is $\mathbb{H}$-linear. $\square$

</details>
</div>

Because the map $r_\mathbf{j} \colon \mathbb{H}^n \to \mathbb{H}^n$ is neither $\mathbb{H}$-linear nor $\mathbb{C}$-linear, it cannot be represented by left multiplication by a complex matrix. If $q = u + \mathbf{j}v \in \mathbb{H}^n$, where $u, v \in \mathbb{C}^n$, then

$$r_\mathbf{j}(q) = q\mathbf{j} = (u + \mathbf{j}v)\mathbf{j} = -\bar{v} + \mathbf{j}\bar{u}.$$

In matrix notation,

$$r_\mathbf{j}\!\left(\begin{bmatrix} u \\ v \end{bmatrix}\right) = \begin{bmatrix} -\bar{v} \\ \bar{u} \end{bmatrix} = c\!\left(\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} u \\ v \end{bmatrix}\right) = -c\!\left(J\begin{bmatrix} u \\ v \end{bmatrix}\right),$$

where $c$ denotes complex conjugation and $J$ is the $2 \times 2$ matrix $\begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$.

## E.7 Symplectic Group

Let $V$ be a vector space over a skew field $K$ with conjugation, and let $B \colon V \times V \to K$ be a bilinear or sesquilinear function over $K$. Such a function is often called a bilinear or sesquilinear **form** over $K$. A $K$-linear automorphism $f \colon V \to V$ is said to **preserve** the form $B$ if

$$B(f(x), f(y)) = B(x, y) \quad \text{for all } x, y \in V.$$

The set of these automorphisms is a subgroup of the general linear group $\operatorname{GL}(V)$.

When $K$ is the skew field $\mathbb{R}$, $\mathbb{C}$, or $\mathbb{H}$, and $B$ is the Euclidean, Hermitian, or quaternionic inner product respectively on $K^n$, the subgroup of $\operatorname{GL}(n, K)$ consisting of automorphisms of $K^n$ preserving each of these inner products is called the **orthogonal**, **unitary**, or **symplectic group** and denoted by $\mathrm{O}(n)$, $\mathrm{U}(n)$, or $\mathrm{Sp}(n)$ respectively. Naturally, the automorphisms in these three groups are called **orthogonal**, **unitary**, or **symplectic** automorphisms.

In particular, the **symplectic group** is the group of automorphisms $f$ of $\mathbb{H}^n$ such that

$$\langle f(x), f(y) \rangle = \langle x, y \rangle \quad \text{for all } x, y \in \mathbb{H}^n.$$

In terms of matrices, if $A$ is the quaternionic matrix of such an $f$, then

$$\langle f(x), f(y) \rangle = \overline{Ax}^T Ay = \bar{x}^T \bar{A}^T Ay = \bar{x}^T y \quad \text{for all } x, y \in \mathbb{H}^n.$$

Therefore, $f \in \mathrm{Sp}(n)$ if and only if its matrix $A$ satisfies $\bar{A}^T A = I$.

Because $\mathbb{H}^n = \mathbb{C}^n \oplus \mathbf{j}\mathbb{C}^n$ is isomorphic to $\mathbb{C}^{2n}$ as a complex vector space and an $\mathbb{H}$-linear map is necessarily $\mathbb{C}$-linear, the group $\operatorname{GL}(n, \mathbb{H})$ is isomorphic to a subgroup of $\operatorname{GL}(2n, \mathbb{C})$ (see Problem E.2).

The **complex symplectic group** $\mathrm{Sp}(2n, \mathbb{C})$ is the subgroup of $\operatorname{GL}(2n, \mathbb{C})$ consisting of automorphisms of $\mathbb{C}^{2n}$ preserving the skew-symmetric bilinear form $B \colon \mathbb{C}^{2n} \times \mathbb{C}^{2n} \to \mathbb{C}$,

$$B(x, y) = \sum_{i=1}^n x^i y^{n+i} - x^{n+i} y^i = x^T Jy, \quad J = \begin{bmatrix} 0 & I_n \\ -I_n & 0 \end{bmatrix},$$

where $I_n$ is the $n \times n$ identity matrix. If $f \colon \mathbb{C}^{2n} \to \mathbb{C}^{2n}$ is given by $f(x) = Ax$, then

$$f \in \mathrm{Sp}(2n, \mathbb{C}) \iff A^T J A = J.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem E.3</span></p>

Under the injection $\operatorname{GL}(n, \mathbb{H}) \hookrightarrow \operatorname{GL}(2n, \mathbb{C})$, the symplectic group $\mathrm{Sp}(n)$ maps isomorphically to the intersection $\mathrm{U}(2n) \cap \mathrm{Sp}(2n, \mathbb{C})$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$f \in \mathrm{Sp}(n)$

$\iff f \colon \mathbb{H}^n \to \mathbb{H}^n$ is $\mathbb{H}$-linear and preserves the quaternionic inner product

$\iff f \colon \mathbb{C}^{2n} \to \mathbb{C}^{2n}$ is $\mathbb{C}$-linear, $f \circ r_\mathbf{j} = r_\mathbf{j} \circ f$, and $f$ preserves the Hermitian inner product and the standard skew-symmetric bilinear form on $\mathbb{C}^{2n}$ (by Proposition E.2 and Section E.5)

$\iff f \circ r_\mathbf{j} = r_\mathbf{j} \circ f$ and $f \in \mathrm{U}(2n) \cap \mathrm{Sp}(2n, \mathbb{C})$.

We will now show that if $f \in \mathrm{U}(2n)$, then the condition $f \circ r_\mathbf{j} = r_\mathbf{j} \circ f$ is equivalent to $f \in \mathrm{Sp}(2n, \mathbb{C})$. Let $f \in \mathrm{U}(2n)$ and let $A$ be the matrix of $f$ relative to the standard basis in $\mathbb{C}^{2n}$. Then

$$(f \circ r_\mathbf{j})(x) = (r_\mathbf{j} \circ f)(x) \text{ for all } x \in \mathbb{C}^{2n}$$

$$\iff -Ac(Jx) = -c(JAx) \text{ for all } x \in \mathbb{C}^{2n} \quad (\text{by (E.1)})$$

$$\iff c(\bar{A}Jx) = c(JAx) \text{ for all } x \in \mathbb{C}^{2n}$$

$$\iff \bar{A}Jx = JAx \text{ for all } x \in \mathbb{C}^{2n}$$

$$\iff J = \bar{A}^{-1}JA$$

$$\iff J = A^T JA \quad (\text{since } A \in \mathrm{U}(2n))$$

$$\iff f \in \mathrm{Sp}(2n, \mathbb{C}).$$

Therefore, the condition $f \circ r_\mathbf{j} = r_\mathbf{j} \circ f$ is redundant if $f \in \mathrm{U}(2n) \cap \mathrm{Sp}(2n, \mathbb{C})$. By the first paragraph of this proof, there is a group isomorphism $\mathrm{Sp}(n) \simeq \mathrm{U}(2n) \cap \mathrm{Sp}(2n, \mathbb{C})$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span></p>

Under the algebra isomorphisms $\operatorname{End}_\mathbb{H}(\mathbb{H}) \simeq \mathbb{H}$, elements of $\mathrm{Sp}(1)$ correspond to quaternions $q = a + \mathbf{i}b + \mathbf{j}c + \mathbf{k}d$ such that

$$\bar{q}q = a^2 + b^2 + c^2 + d^2 = 1.$$

These are precisely quaternions of norm 1. Therefore, under the chain of real vector space isomorphisms $\operatorname{End}_\mathbb{H}(\mathbb{H}) \simeq \mathbb{H} \simeq \mathbb{R}^4$, the group $\mathrm{Sp}(1)$ maps to $S^3$, the unit 3-sphere in $\mathbb{R}^4$.

</div>

### Problems

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise E.1</span><span class="math-callout__name">(Quaternionic conjugation)</span></p>

Prove Proposition E.1.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise E.2</span><span class="math-callout__name">(Complex representation of an $\mathbb{H}$-linear map)</span></p>

Suppose an $\mathbb{H}$-linear map $f \colon \mathbb{H}^n \to \mathbb{H}^n$ is represented relative to the standard basis $e_1, \ldots, e_n$ by the matrix $A = u + \mathbf{j}v \in \mathbb{H}^{n \times n}$, where $u, v \in \mathbb{C}^{n \times n}$. Show that as a $\mathbb{C}$-linear map, $f \colon \mathbb{H}^n \to \mathbb{H}^n$ is represented relative to the basis $e_1, \ldots, e_n, \mathbf{j}e_1, \ldots, \mathbf{j}e_n$ by the matrix $\begin{bmatrix} u & -\bar{v} \\ v & \bar{u} \end{bmatrix}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise E.3</span><span class="math-callout__name">(Symplectic and unitary groups of small dimension)</span></p>

For a field $K$, the *special linear group* $\mathrm{SL}(n, K)$ is the subgroup of $\operatorname{GL}(n, K)$ consisting of all automorphisms of $K^n$ of determinant 1, and the *special unitary group* $\mathrm{SU}(n)$ is the subgroup of $\mathrm{U}(n)$ consisting of unitary automorphisms of $\mathbb{C}^n$ of determinant 1. Prove the following identifications or group isomorphisms.

**(a)** $\mathrm{Sp}(2, \mathbb{C}) = \mathrm{SL}(2, \mathbb{C})$.

**(b)** $\mathrm{Sp}(1) \simeq \mathrm{SU}(2)$. *(Hint: Use Theorem E.3 and part (a).)*

**(c)**

$$\mathrm{SU}(2) \simeq \left\lbrace \begin{bmatrix} u & -\bar{v} \\ v & \bar{u} \end{bmatrix} \in \mathbb{C}^{2 \times 2} \;\middle|\; u\bar{u} + v\bar{v} = 1 \right\rbrace.$$

*(Hint: Use part (b) and the representation of quaternions by $2 \times 2$ complex matrices in Subsection E.4.)*

</div>

# Solutions to Selected Exercises Within the Text

This section collects solutions to selected exercises that appear inline throughout the text.

## Chapter 3 — Alternating $k$-Linear Functions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 3.6</span><span class="math-callout__name">(Inversions)</span></p>

As a matrix, $\tau = \begin{bmatrix} 1 & 2 & 3 & 4 & 5 \\ 1 & 3 & 4 & 5 & 1 \end{bmatrix}$. Scanning the second row, we see that $\tau$ has four inversions: $(2,1)$, $(3,1)$, $(4,1)$, $(5,1)$.

Alternatively, one can count the number of inversions in the permutation $\tau$. There are $k$ inversions starting with $k+1$, namely, $(k+1,1), (k+1,2), \ldots, (k+1,k)$. Indeed, for each $i = 1, \ldots, \ell$, there are $k$ inversions starting with $k + i$. Hence, the total number of inversions in $\tau$ is $k\ell$. By Proposition 3.8, $\operatorname{sgn}(\tau) = (-1)^{k\ell}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 3.13</span><span class="math-callout__name">(Symmetrizing operator)</span></p>

A $k$-linear function $h \colon V \to \mathbb{R}$ is symmetric if and only if $\tau h = h$ for all $\tau \in S_k$. Now

$$\tau(Sf) = \tau \sum_{\sigma \in S_k} \sigma f = \sum_{\sigma \in S_k} (\tau\sigma)f.$$

As $\sigma$ runs over all elements of the permutation group $S_k$, so does $\tau\sigma$. Hence,

$$\sum_{\sigma \in S_k} (\tau\sigma) f = \sum_{\tau\sigma \in S_k} (\tau\sigma) f = Sf.$$

This proves that $\tau(Sf) = Sf$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 3.15</span><span class="math-callout__name">(Alternating operator)</span></p>

$$f(v_1, v_2, v_3) - f(v_1, v_3, v_2) + f(v_2, v_3, v_1) - f(v_2, v_1, v_3) + f(v_3, v_1, v_2) - f(v_3, v_2, v_1).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 3.20</span><span class="math-callout__name">(Wedge product of two 2-covectors)</span></p>

$$(f \wedge g)(v_1, v_2, v_3, v_4)$$

$$= f(v_1, v_2)g(v_3, v_4) - f(v_1, v_3)g(v_2, v_4) + f(v_1, v_4)g(v_2, v_3)$$

$$\quad + f(v_2, v_3)g(v_1, v_4) - f(v_2, v_4)g(v_1, v_3) + f(v_3, v_4)g(v_1, v_2).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 3.22</span><span class="math-callout__name">(Sign of a permutation)</span></p>

We can achieve the permutation $\tau$ from the initial configuration $1, 2, \ldots, k + \ell$ in $k$ steps.

1. First, move the element $k$ to the very end across the $\ell$ elements $k+1, \ldots, k+\ell$. This requires $\ell$ transpositions.
2. Next, move the element $k-1$ across the $\ell$ elements $k+1, \ldots, k+\ell$.
3. Then move the element $k-2$ across the same $\ell$ elements, and so on.

Each of the $k$ steps requires $\ell$ transpositions. In the end we achieve $\tau$ from the identity using $\ell k$ transpositions.

</div>

## Chapter 4 — Differential Forms on $\mathbb{R}^n$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 4.3</span><span class="math-callout__name">(A basis for 3-covectors)</span></p>

By Proposition 3.29, a basis for $A_3(T_p(\mathbb{R}^4))$ is

$$\left(dx^1 \wedge dx^2 \wedge dx^3\right)_p, \quad \left(dx^1 \wedge dx^2 \wedge dx^4\right)_p, \quad \left(dx^1 \wedge dx^3 \wedge dx^4\right)_p, \quad \left(dx^2 \wedge dx^3 \wedge dx^4\right)_p.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 4.4</span><span class="math-callout__name">(Wedge product of a 2-form with a 1-form)</span></p>

The $(2,1)$-shuffles are $(1 < 2, 3)$, $(1 < 3, 2)$, $(2 < 3, 1)$, with respective signs $+, -, +$. By equation (3.6),

$$(\omega \wedge \tau)(X, Y, Z) = \omega(X, Y)\tau(Z) - \omega(X, Z)\tau(Y) + \omega(Y, Z)\tau(X).$$

</div>

## Chapters 6–8 — Smooth Maps, Quotients, and the Tangent Space

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 6.14</span><span class="math-callout__name">(Smoothness of a map to a circle)</span></p>

Without further justification, the fact that both $\cos t$ and $\sin t$ are $C^\infty$ proves only the smoothness of $(\cos t, \sin t)$ as a map from $\mathbb{R}$ to $\mathbb{R}^2$. To show that $F \colon \mathbb{R} \to S^1$ is $C^\infty$, we need to cover $S^1$ with charts $(U_i, \phi_i)$ and examine in turn each $\phi_i \circ F \colon F^{-1}(U_i) \to \mathbb{R}$. Let $\lbrace (U_i, \phi_i) \mid i = 1, \ldots, 4 \rbrace$ be the atlas of Example 5.16. On $F^{-1}(U_1)$, $\phi_1 \circ F(t) = (x \circ F)(t) = \cos t$ is $C^\infty$. On $F^{-1}(U_3)$, $\phi_3 \circ F(t) = \sin t$ is $C^\infty$. Similar computations on $F^{-1}(U_2)$ and $F^{-1}(U_4)$ prove the smoothness of $F$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 7.11</span><span class="math-callout__name">(Real projective space as a quotient of a sphere)</span></p>

Define $\bar{f} \colon \mathbb{R}P^n \to S^n/\!\sim$ by $\bar{f}([x]) = \left[\frac{x}{\|x\|}\right] \in S^n/\!\sim$. This map is well defined because $\bar{f}([x]) = \left[\frac{tx}{\|tx\|}\right] = \left[\pm\frac{x}{\|x\|}\right] = \left[\frac{x}{\|x\|}\right]$. Note that if $\pi_1 \colon \mathbb{R}^{n+1} - \lbrace 0 \rbrace \to \mathbb{R}P^n$ and $\pi_2 \colon S^n \to S^n/\!\sim$ are the projection maps, then there is a commutative diagram and $\bar{f}$ is continuous because $\pi_2 \circ f$ is continuous.

Next define $g \colon S^n \to \mathbb{R}^{n+1} - \lbrace 0 \rbrace$ by $g(x) = x$. This map induces a map $\bar{g} \colon S^n/\!\sim \hookrightarrow \mathbb{R}P^n$, $\bar{g}([x]) = [x]$. Then $\bar{g} \circ \bar{f}([x]) = [x]$ and $\bar{f} \circ \bar{g}([x]) = [x]$, so $\bar{f}$ and $\bar{g}$ are inverses to each other.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 8.7*</span><span class="math-callout__name">(Tangent space to a product)</span></p>

If $(U, \phi) = (U, x^1, \ldots, x^m)$ and $(V, \psi) = (V, y^1, \ldots, y^n)$ are charts about $p$ in $M$ and $q$ in $N$ respectively, then by Proposition 5.18, a chart about $(p, q)$ in $M \times N$ is

$$(U \times V, \phi \times \psi) = (U \times V, \bar{x}^1, \ldots, \bar{x}^m, \bar{y}^1, \ldots, \bar{y}^n),$$

where $\bar{x}^i = \pi_1^* x^i$ and $\bar{y}^i = \pi_2^* y^i$. A basis for $T_{(p,q)}(M \times N)$ is

$$\frac{\partial}{\partial \bar{x}^1}\bigg|_{(p,q)}, \ldots, \frac{\partial}{\partial \bar{x}^m}\bigg|_{(p,q)}, \quad \frac{\partial}{\partial \bar{y}^1}\bigg|_{(p,q)}, \ldots, \frac{\partial}{\partial \bar{y}^n}\bigg|_{(p,q)}.$$

By (8.7.1) and (8.7.2), the linear map $(\pi_{1*}, \pi_{2*})$ maps a basis of $T_{(p,q)}(M \times N)$ to a basis of $T_p M \times T_q N$. It is therefore an isomorphism.

</div>

# Hints and Solutions to Selected End-of-Section Problems

This section collects hints and selected complete solutions (marked with *) to end-of-section problems.

## Chapter 1 — Euclidean Spaces

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 1.2*</span><span class="math-callout__name">(A $C^\infty$ function very flat at $0$)</span></p>

**(a)** Assume $x > 0$. For $k = 1$, $f'(x) = (1/x^2)e^{-1/x}$. With $p_2(y) = y^2$, this verifies the claim. Now suppose $f^{(k)}(x) = p_{2k}(1/x) e^{-1/x}$. By the product rule and the chain rule,

$$f^{(k+1)}(x) = p_{2k-1}\!\left(\tfrac{1}{x}\right) \cdot \left(-\tfrac{1}{x^2}\right) e^{-1/x} + p_{2k}\!\left(\tfrac{1}{x}\right) \cdot \tfrac{1}{x^2} e^{-1/x} = p_{2k+2}\!\left(\tfrac{1}{x}\right) e^{-1/x},$$

where $q_n(y)$ and $p_n(y)$ are polynomials of degree $n$ in $y$. By induction, the claim is true for all $k \ge 1$. It is trivially true for $k = 0$ also.

**(b)** For $x > 0$, the formula in (a) shows that $f(x)$ is $C^\infty$. For $x < 0$, $f(x) \equiv 0$, which is trivially $C^\infty$. It remains to show that $f^{(k)}(x)$ is defined and continuous at $x = 0$ for all $k$. Suppose $f^{(k)}(0) = 0$. By the definition of the derivative,

$$f^{(k+1)}(0) = \lim_{x \to 0} \frac{f^{(k)}(x) - f^{(k)}(0)}{x} = \lim_{x \to 0} \frac{f^{(k)}(x)}{x}.$$

The limit from the left is clearly 0. So it suffices to compute the limit from the right:

$$\lim_{x \to 0^+} \frac{f^{(k)}(x)}{x} = \lim_{x \to 0^+} \frac{p_{2k}(1/x) e^{-1/x}}{x} = \lim_{x \to 0^+} p_{2k+1}\!\left(\tfrac{1}{x}\right) e^{-1/x}$$

$$= \lim_{y \to \infty} \frac{p_{2k+1}(y)}{e^y} \quad \left(\text{replacing } \tfrac{1}{x} \text{ by } y\right).$$

Applying l'Hôpital's rule $2k+1$ times, we reduce this limit to 0. Hence, $f^{(k+1)}(0) = 0$. By induction, $f^{(k)}(0) = 0$ for all $k \ge 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 1.6*</span><span class="math-callout__name">(Taylor's theorem with remainder to order 2)</span></p>

In Problem 1.6, set $x = t$ and $y = tu$. By Taylor's theorem with remainder, there exist $C^\infty$ functions $g_1, g_2$ such that

$$f(x, y) = f(\mathbf{0}) + xg_1(x, y) + yg_2(x, y).$$

Applying the theorem again, but to $g_1$ and $g_2$, we obtain

$$g_1(x, y) = g_1(\mathbf{0}) + xg_{11}(x, y) + yg_{12}(x, y),$$

$$g_2(x, y) = g_2(\mathbf{0}) + xg_{21}(x, y) + yg_{22}(x, y).$$

Since $g_1(\mathbf{0}) = \partial f / \partial x(\mathbf{0})$ and $g_2(\mathbf{0}) = \partial f / \partial y(\mathbf{0})$, substituting (1.6.2) and (1.6.3) into (1.6.1) gives the result.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 1.7*</span><span class="math-callout__name">(A function with a removable singularity)</span></p>

In Problem 1.6, set $x = t$ and $y = tu$. We obtain

$$f(t, tu) = f(\mathbf{0}) + t \frac{\partial f}{\partial x}(\mathbf{0}) + tu \frac{\partial f}{\partial y}(\mathbf{0}) + t^2(\cdots),$$

where

$$(\cdots) = g_{11}(t, tu) + ug_{12}(t, tu) + u^2 g_{22}(t, tu)$$

is a $C^\infty$ function of $t$ and $u$. Since $f(\mathbf{0}) = \partial f / \partial x(\mathbf{0}) = \partial f / \partial y(\mathbf{0}) = 0$,

$$\frac{f(t, tu)}{t} = t(\cdots),$$

which is clearly $C^\infty$ in $t, u$ and agrees with $g$ when $t = 0$.

</div>

## Chapter 3 — Alternating $k$-Linear Functions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 3.10*</span><span class="math-callout__name">(Linear independence of covectors)</span></p>

$(\Rightarrow)$ If $\alpha^1, \ldots, \alpha^k$ are linearly dependent, then one of them is a linear combination of the others. Without loss of generality, we may assume that

$$\alpha^k = \sum_{j=1}^{k-1} c_j \alpha^j.$$

In the wedge product $\alpha^1 \wedge \cdots \wedge \alpha^{k-1} \wedge (\sum_{j=1}^{k-1} c_j \alpha^j)$, every term has a repeated $\alpha^i$. Hence, $\alpha^1 \wedge \cdots \wedge \alpha^k = 0$.

$(\Leftarrow)$ Suppose $\alpha^1, \ldots, \alpha^k$ are linearly independent. Then they can be extended to a basis $\alpha^1, \ldots, \alpha^k, \ldots, \alpha^n$ for $V^\vee$. Let $v_1, \ldots, v_n$ be the dual basis for $V$. By Proposition 3.27,

$$(\alpha^1 \wedge \cdots \wedge \alpha^k)(v_1, \ldots, v_k) = \det[\alpha^i(v_j)] = \det[\delta_j^i] = 1.$$

Hence, $\alpha^1 \wedge \cdots \wedge \alpha^k \neq 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 3.11*</span><span class="math-callout__name">(Exterior multiplication)</span></p>

$(\Leftarrow)$ Clear because $\alpha \wedge \alpha = 0$.

$(\Rightarrow)$ Suppose $\alpha \wedge \gamma = 0$. Extend $\alpha$ to a basis $\alpha^1, \ldots, \alpha^n$ for $V^\vee$, with $\alpha^1 = \alpha$. Write $\gamma = \sum c_J \alpha^J$, where $J$ runs over all strictly ascending multi-indices $1 \le j_1 < \cdots < j_k \le n$. In the sum $\alpha \wedge \gamma = \sum c_J \alpha \wedge \alpha^J$, all the terms $\alpha \wedge \alpha^J$ with $j_1 = 1$ vanish, since $\alpha = \alpha^1$.

Since $\lbrace \alpha \wedge \alpha^J \rbrace_{j_1 \neq 1}$ is a subset of a basis for $A_{k+1}(V)$, it is linearly independent, and so all $c_J$ are 0 if $j_1 \neq 1$. Thus,

$$\gamma = \sum_{j_1 = 1} c_J \alpha^J = \alpha \wedge \left(\sum_{j_1 = 1} c_J \alpha^{j_2} \wedge \cdots \wedge \alpha^{j_k}\right).$$

</div>

## Chapter 7 — Quotient Spaces and Projective Space

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 7.4*</span><span class="math-callout__name">(Quotient space of a sphere with antipodal points)</span></p>

**(a)** Let $U$ be an open subset of $S^n$. Then $\pi^{-1}(\pi(U)) = U \cup a(U)$, where $a \colon S^n \to S^n$, $a(x) = -x$, is the antipodal map. Since the antipodal map is a homeomorphism, $a(U)$ is open, and hence $\pi^{-1}(\pi(U))$ is an open set. By the definition of quotient topology, $\pi(U)$ is open. This proves that $\pi$ is an open map.

**(b)** The graph $R$ of the equivalence relation $\sim$ is

$$R = \lbrace (x, x) \in S^n \times S^n \rbrace \cup \lbrace (x, -x) \in S^n \times S^n \rbrace = \Delta \cup (\mathbb{1} \times a)(\Delta).$$

By Corollary 7.8, because $S^n$ is Hausdorff, the diagonal $\Delta$ in $S^n \times S^n$ is closed. Since $\mathbb{1} \times a \colon S^n \times S^n \to S^n \times S^n$, $(x, y) \mapsto (x, -y)$ is a homeomorphism, $(\mathbb{1} \times a)(\Delta)$ is also closed. As a union of two closed sets, $R$ is closed in $S^n \times S^n$. By Theorem 7.7, $S^n/\!\sim$ is Hausdorff.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 7.9*</span><span class="math-callout__name">(Compactness of real projective space)</span></p>

By Exercise 7.11 there is a continuous surjective map $\pi \colon S^n \to \mathbb{R}P^n$. Since the sphere $S^n$ is compact, and the continuous image of a compact set is compact (Proposition A.34), $\mathbb{R}P^n$ is compact.

</div>

## Chapter 8 — The Tangent Space

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 8.1*</span><span class="math-callout__name">(Differential of a map)</span></p>

To determine the coefficient $a$ in $F_*(\partial/\partial x) = a\,\partial/\partial u + b\,\partial/\partial v + c\,\partial/\partial w$, we apply both sides to $u$ to get

$$a = F_*\!\left(\frac{\partial}{\partial x}\right) u = \left(a \frac{\partial}{\partial u} + b \frac{\partial}{\partial v} + c \frac{\partial}{\partial w}\right) u = a.$$

Hence,

$$a = F_*\!\left(\frac{\partial}{\partial x}\right) u = \frac{\partial}{\partial x}(u \circ F) = \frac{\partial}{\partial x}(x) = 1.$$

Similarly, $b = F_*(\partial/\partial x)\,v = \frac{\partial}{\partial x}(v \circ F) = \frac{\partial}{\partial x}(y) = 0$ and $c = F_*(\partial/\partial x)\,w = \frac{\partial}{\partial x}(w \circ F) = \frac{\partial}{\partial x}(xy) = y$.

So $F_*(\partial/\partial x) = \partial/\partial u + y\,\partial/\partial w$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 8.9*</span><span class="math-callout__name">(Transforming vectors to coordinate vectors)</span></p>

Let $(V, y^1, \ldots, y^n)$ be a chart about $p$. Suppose $(X_j)_p = \sum_i a_j^i\, \partial/\partial y^i\big|_p$. Since $(X_1)_p, \ldots, (X_n)_p$ are linearly independent, the matrix $A = [a_j^i]$ is nonsingular.

Define a new coordinate system $x^1, \ldots, x^n$ by

$$y^i = \sum_{j=1}^n a_j^i x^j \quad \text{for } i = 1, \ldots, n.$$

By the chain rule,

$$\frac{\partial}{\partial x^j} = \sum_i \frac{\partial y^i}{\partial x^j} \frac{\partial}{\partial y^i} = \sum_i a_j^i \frac{\partial}{\partial y^i}.$$

At the point $p$, $\frac{\partial}{\partial x^j}\big|_p = \sum_i a_j^i \frac{\partial}{\partial y^i}\big|_p = (X_j)_p$.

</div>

## Chapter 9 — Submanifolds

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 9.4*</span><span class="math-callout__name">(Regular submanifolds)</span></p>

Let $p \in S$. By hypothesis there is an open set $U$ in $\mathbb{R}^2$ such that on $U \cap S$ one of the coordinates is a $C^\infty$ function of the other. Without loss of generality, we assume that $y = f(x)$ for some $C^\infty$ function $f \colon A \subset \mathbb{R} \to B \subset \mathbb{R}$, where $A$ and $B$ are open sets in $\mathbb{R}$ and $V := A \times B \subset U$. Let $F \colon V \to \mathbb{R}^2$ be given by $F(x, y) = (x, y - f(x))$. Since $F$ is a diffeomorphism onto its image, it can be used as a coordinate map. In the chart $(V, x, y - f(x))$, $V \cap S$ is defined by the vanishing of the coordinate $y - f(x)$. This proves that $S$ is a regular submanifold of $\mathbb{R}^2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 9.10*</span><span class="math-callout__name">(The transversality theorem)</span></p>

**(a)** $f^{-1}(U) \cap f^{-1}(S) = f^{-1}(U \cap S) = f^{-1}(g^{-1}(0)) = (g \circ f)^{-1}(0)$.

**(b)** Let $p \in f^{-1}(U) \cap f^{-1}(S)$. Then $f(p) \in U \cap S$. Because $S$ is a fiber of $g$, the pushforward $g_*(T_{f(p)}S)$ equals 0. Because $g \colon U \to \mathbb{R}^k$ is a projection, $g_*(T_{f(p)}M) = T_0(\mathbb{R}^k)$. Applying $g_*$ to the transversality equation (9.4), we get

$$g_* f_*(T_p N) = g_*(T_{f(p)}M) = T_0(\mathbb{R}^k).$$

Hence, $g \circ f \colon f^{-1}(U) \to \mathbb{R}^k$ is a submersion at $p$. Since $p$ is an arbitrary point of $f^{-1}(U) \cap f^{-1}(S) = (g \circ f)^{-1}(0)$, this set is a regular level set of $g \circ f$.

**(c)** By the regular level set theorem, $f^{-1}(U) \cap f^{-1}(S)$ is a regular submanifold of $f^{-1}(U) \subset N$. Thus every point $p \in f^{-1}(S)$ has an adapted chart relative to $f^{-1}(S)$ in $N$.

</div>

## Chapter 11 — The Rank of a Smooth Map

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 11.3*</span><span class="math-callout__name">(Critical points of a smooth map on a compact manifold)</span></p>

**First Proof.** Suppose $f \colon N \to \mathbb{R}^m$ has no critical point. Then it is a submersion. The projection to the first factor, $\pi \colon \mathbb{R}^m \to \mathbb{R}$, is also a submersion. It follows that the composite $\pi \circ f \colon N \to \mathbb{R}$ is a submersion. This contradicts the fact that as a continuous function from a compact manifold to $\mathbb{R}$, the function $\pi \circ f$ has a maximum and hence a critical point (see Problem 8.10).

**Second Proof.** Suppose $f \colon N \to \mathbb{R}^m$ has no critical point. Then it is a submersion. Since a submersion is an open map (Corollary 11.6), the image $f(N)$ is open in $\mathbb{R}^m$. But the continuous image of a compact set is compact and a compact subset of $\mathbb{R}^m$ is closed and bounded. Hence, $f(N)$ is a nonempty closed proper subset of $\mathbb{R}^m$ that is both open and closed. This is a contradiction, because being connected, $\mathbb{R}^m$ cannot have a nonempty proper subset that is both open and closed.

</div>

## Chapter 12 — The Tangent Bundle

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 12.1*</span><span class="math-callout__name">(The Hausdorff condition on the tangent bundle)</span></p>

Let $(p, v)$ and $(q, w)$ be distinct points of the tangent bundle $TM$.

**Case 1:** $p \neq q$. Because $M$ is Hausdorff, $p$ and $q$ can be separated by disjoint neighborhoods $U$ and $V$. Then $TU$ and $TV$ are disjoint open subsets of $TM$ containing $(p, v)$ and $(q, w)$, respectively.

**Case 2:** $p = q$. Let $(U, \phi)$ be a coordinate neighborhood of $p$. Then $(p, v)$ and $(p, w)$ are distinct points in the open set $TU$. Since $TU$ is homeomorphic to the open subset $\phi(U) \times \mathbb{R}^n$ of $\mathbb{R}^{2n}$, and any subspace of a Hausdorff space is Hausdorff, $TU$ is Hausdorff. Therefore, $(p, v)$ and $(p, w)$ can be separated by disjoint open sets in $TU$.

</div>

## Chapter 13 — Bump Functions and Partitions of Unity

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 13.1*</span><span class="math-callout__name">(Support of a finite sum)</span></p>

Let $A$ be the set where $\sum \rho_i$ is not zero and $A_i$ the set where $\rho_i(x) \neq 0$:

$$A = \lbrace x \in M \mid \textstyle\sum \rho_i(x) \neq 0 \rbrace, \quad A_i = \lbrace x \in M \mid \rho_i(x) \neq 0 \rbrace.$$

If $\sum \rho_i(x) \neq 0$, then at least one $\rho_i(x)$ must be nonzero. This implies that $A \subset \bigcup A_i$. Taking the closure of both sides gives $\operatorname{cl}(A) \subset \bigcup A_i$. For a finite union, $\overline{\bigcup A_i} = \bigcup \overline{A_i}$ (Exercise A.53). Hence,

$$\operatorname{supp}\!\left(\textstyle\sum \rho_i\right) = \operatorname{cl}(A) \subset \bigcup A_i = \bigcup \overline{A_i} = \bigcup \operatorname{supp} \rho_i.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 13.2*</span><span class="math-callout__name">(Locally finite family and compact set)</span></p>

For each $p \in K$, let $W_p$ be a neighborhood of $p$ that intersects only finitely many of the sets $A_\alpha$. The collection $\lbrace W_p \rbrace_{p \in K}$ is an open cover of $K$. By compactness, $K$ has a finite subcover $\lbrace W_{p_1}, \ldots, W_{p_r} \rbrace$. Since each $W_{p_i}$ intersects only finitely many of the $A_\alpha$, the finite union $W := \bigcup_{i=1}^r W_{p_i}$ intersects only finitely many of the $A_\alpha$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 13.7*</span><span class="math-callout__name">(Closure of a locally finite union)</span></p>

$(\supset)$ Since $A_\alpha \subset \bigcup A_\alpha$, taking the closure of both sides gives $\overline{A_\alpha} \subset \overline{\bigcup A_\alpha}$. Hence, $\bigcup \overline{A_\alpha} \subset \overline{\bigcup A_\alpha}$.

$(\subset)$ Instead of proving $\overline{\bigcup A_\alpha} \subset \bigcup \overline{A_\alpha}$, we will prove the contrapositive: if $p \notin \bigcup \overline{A_\alpha}$, then $p \notin \overline{\bigcup A_\alpha}$. Suppose $p \notin \bigcup \overline{A_\alpha}$. By local finiteness, $p$ has a neighborhood $W$ that meets only finitely many of the $A_\alpha$'s, say $A_{\alpha_1}, \ldots, A_{\alpha_n}$.

Since $p \notin \overline{A_\alpha}$ for any $\alpha$, $p \notin \bigcup_{i=1}^n \overline{A_{\alpha_i}}$. Note that $W$ is disjoint from $A_\alpha$ for all $\alpha \neq \alpha_i$, so $W - \bigcup_{i=1}^n \overline{A_{\alpha_i}}$ is disjoint from $A_\alpha$ for all $\alpha$. Because $\bigcup_{i=1}^n \overline{A_{\alpha_i}}$ is closed, $W - \bigcup_{i=1}^n \overline{A_{\alpha_i}}$ is an open set containing $p$ disjoint from $\bigcup A_\alpha$. By the local characterization of closure (Proposition A.48), $p \notin \overline{\bigcup A_\alpha}$. Hence, $\overline{\bigcup A_\alpha} \subset \bigcup \overline{A_\alpha}$.

</div>

## Chapter 14 — Vector Fields

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 14.1*</span><span class="math-callout__name">(Equality of vector fields)</span></p>

The implication in the direction $(\Rightarrow)$ is obvious. For the converse, let $p \in M$. To show that $X_p = Y_p$, it suffices to show that $X_p[h] = Y_p[h]$ for any germ $[h]$ of $C^\infty$ functions in $C_p^\infty(M)$.

Suppose $h \colon U \to \mathbb{R}$ is a $C^\infty$ function that represents the germ $[h]$. We can extend it to a $C^\infty$ function $\tilde{h} \colon M \to \mathbb{R}$ by multiplying it by a $C^\infty$ bump function supported in $U$ that is identically 1 in a neighborhood of $p$. By hypothesis, $X\tilde{h} = Y\tilde{h}$.

Because $\tilde{h} = h$ in a neighborhood of $p$, we have $X_p h = X_p \tilde{h}$ and $Y_p h = Y_p \tilde{h}$. It follows from (14.1.1) that $X_p h = Y_p h$. Thus, $X_p = Y_p$. Since $p$ is an arbitrary point of $M$, the two vector fields $X$ and $Y$ are equal.

</div>

## Chapter 15 — Lie Groups

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 15.4*</span><span class="math-callout__name">(Open subgroup of a connected Lie group)</span></p>

For any $g \in G$, left multiplication $\ell_g \colon G \to G$ by $g$ maps the subgroup $H$ to the left coset $gH$. Since $H$ is open and $\ell_g$ is a homeomorphism, the coset $gH$ is open. Thus, the set of cosets $gH$, $g \in G$, partitions $G$ into a disjoint union of open subsets. Since $G$ is connected, there can be only one coset. Therefore, $H = G$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 15.7*</span><span class="math-callout__name">(Differential of the determinant map)</span></p>

Let $c(t) = Ae^{tX}$. Then $c(0) = A$ and $c'(0) = AX$. Using the curve $c(t)$ to calculate the differential yields

$$\det_{A*}(AX) = \frac{d}{dt}\bigg|_{t=0} \det(c(t)) = \frac{d}{dt}\bigg|_{t=0} (\det A) \det e^{tX}$$

$$= (\det A) \frac{d}{dt}\bigg|_{t=0} e^{t\,\operatorname{tr} X} = (\det A) \operatorname{tr} X.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 15.8*</span><span class="math-callout__name">(Special linear group)</span></p>

If $\det A = 1$, then Exercise 15.7 gives

$$\det_{*A}(AX) = \operatorname{tr} X.$$

Since $\operatorname{tr} X$ can assume any real value, $\det_{*A} \colon T_A \operatorname{GL}(n, \mathbb{R}) \to \mathbb{R}$ is surjective for all $A \in \det^{-1}(1)$. Hence, 1 is a regular value of $\det$.

</div>

## Chapter 16 — Lie Algebras

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 16.10*</span><span class="math-callout__name">(The pushforward of left-invariant vector fields)</span></p>

Under the isomorphisms $\varphi_H \colon T_e H \xrightarrow{\sim} L(H)$ and $\varphi_G \colon T_e G \xrightarrow{\sim} L(G)$, the Lie brackets and the pushforward maps correspond. Thus, this problem follows from Proposition 16.14 by the correspondence.

A more formal proof goes as follows. Since $X$ and $Y$ are left-invariant vector fields, $X = \tilde{A}$ and $Y = \tilde{B}$ for $A = X_e$ and $B = Y_e \in T_e H$. Then

$$F_*[X, Y] = F_*[\tilde{A}, \tilde{B}] = F_*(F_*[A, B])^{\sim} \quad \text{(Proposition 16.10)}$$

$$= (F_*[A, B])^{\sim} = [F_* A, F_* B]^{\sim} \quad \text{(Proposition 16.14)}$$

$$= [(F_* A)^{\sim}, (F_* B)^{\sim}] \quad \text{(Proposition 16.10)}$$

$$= [F_* \tilde{A}, F_* \tilde{B}]$$

$$= [F_* X, F_* Y].$$

</div>

## Chapters 18–20 — Differential Forms on Manifolds

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 18.8*</span><span class="math-callout__name">(Pullback by a surjective submersion)</span></p>

The fact that $\pi^* \colon \Omega^*(M) \to \Omega^*(\tilde{M})$ is an algebra homomorphism follows from Propositions 18.9 and 18.11.

Suppose $\omega \in \Omega^k(M)$ is a $k$-form on $M$ for which $\pi^* \omega = 0$ in $\Omega^k(\tilde{M})$. To show that $\omega = 0$, pick an arbitrary point $p \in M$, and arbitrary vectors $v_1, \ldots, v_k \in T_p M$. Since $\pi$ is surjective, there is a point $\tilde{p} \in \tilde{M}$ that maps to $p$. Since $\pi$ is a submersion at $\tilde{p}$, there exist $\tilde{v}_1, \ldots, \tilde{v}_k \in T_{\tilde{p}} \tilde{M}$ such that $\pi_{*\tilde{p}} \tilde{v}_i = v_i$. Then

$$0 = (\pi^* \omega)_{\tilde{p}}(\tilde{v}_1, \ldots, \tilde{v}_k) = \omega_{\pi(\tilde{p})}(\pi_* \tilde{v}_1, \ldots, \pi_* \tilde{v}_k) = \omega_p(v_1, \ldots, v_k).$$

Since $p \in M$ and $v_1, \ldots, v_k \in T_p M$ are arbitrary, this proves that $\omega = 0$. Therefore, $\pi^* \colon \Omega^*(M) \to \Omega^*(\tilde{M})$ is injective.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 19.5*</span><span class="math-callout__name">(Coordinates and differential forms)</span></p>

Let $(V, x^1, \ldots, x^n)$ be a chart about $p$. By Corollary 18.4(ii),

$$df^1 \wedge \cdots \wedge df^n = \det\!\left[\frac{\partial f^i}{\partial x^j}\right] dx^1 \wedge \cdots \wedge dx^n.$$

So $(df^1 \wedge \cdots \wedge df^n)_p \neq 0$ if and only if $\det[\partial f^i / \partial x^j(p)] \neq 0$. By the inverse function theorem, this condition is equivalent to the existence of a neighborhood $W$ on which the map $F := (f^1, \ldots, f^n) \colon W \to \mathbb{R}^n$ is a $C^\infty$ diffeomorphism onto its image. In other words, $(W, f^1, \ldots, f^n)$ is a chart.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 20.6</span><span class="math-callout__name">(Global formula for the exterior derivative)</span></p>

By Theorem 20.12,

$$(\mathcal{L}_{Y_0} \omega)(Y_1, \ldots, Y_k) = Y_0\big(\omega(Y_1, \ldots, Y_k)\big) - \sum_{j=1}^k \omega([Y_0, Y_j], Y_1, \ldots, \hat{Y}_j, \ldots, Y_k).$$

By the induction hypothesis, Theorem 20.14 is true for $(k-1)$-forms. Hence,

$$(d\iota_{Y_0} \omega)(Y_1, \ldots, Y_k) = -\sum_{i=1}^k (-1)^{i-1} Y_i\big((\iota_{Y_0}\omega)(Y_1, \ldots, \hat{Y}_i, \ldots, Y_k)\big) - \sum_{1 \le i < j \le k} (-1)^{i+j} (\iota_{Y_0} \omega)([Y_i, Y_j], Y_1, \ldots, \hat{Y}_i, \ldots, \hat{Y}_j, \ldots, Y_k).$$

This expands to

$$= \sum_{i=1}^k (-1)^i Y_i\big(\omega(Y_0, Y_1, \ldots, \hat{Y}_i, \ldots, Y_k)\big) + \sum_{1 \le i < j \le k} (-1)^{i+j} \omega([Y_i, Y_j], Y_0, Y_1, \ldots, \hat{Y}_i, \ldots, \hat{Y}_j, \ldots, Y_k).$$

Adding (20.6.1) and (20.6.2) gives

$$\sum_{i=0}^k (-1)^i Y_i\big(\omega(Y_0, Y_1, \ldots, \hat{Y}_i, \ldots, Y_k)\big) + \sum_{0 \le i < j \le k} (-1)^{i+j} \omega([Y_i, Y_j], Y_0, Y_1, \ldots, \hat{Y}_i, \ldots, \hat{Y}_j, \ldots, Y_k),$$

which simplifies to the right-hand side of Theorem 20.14.

</div>

## Chapter 22 — Manifolds with Boundary and Stokes's Theorem

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 22.3*</span><span class="math-callout__name">(Inward-pointing vectors at the boundary)</span></p>

$(\Leftarrow)$ Suppose $(U, \phi) = (U, x^1, \ldots, x^n)$ is a chart for $M$ centered at $p$ such that $X_p = \sum a^i \partial/\partial x^i\big|_p$ with $a^n > 0$. Then the curve $c(t) = \phi^{-1}(a^1 t, \ldots, a^n t)$ in $M$ satisfies

$$c(0) = p, \quad c([0, \varepsilon[) \subset M^\circ, \quad \text{and} \quad c'(0) = X_p.$$

So $X_p$ is inward-pointing.

$(\Rightarrow)$ Suppose $X_p$ is inward-pointing. Then $X_p \notin T_p(\partial M)$ and there is a curve $c \colon [0, \varepsilon[ \to M$ such that (22.3.1) holds. Let $(U, \phi) = (U, x^1, \ldots, x^n)$ be a chart centered at $p$. On $U \cap M$, we have $x^n \ge 0$. If $(\phi \circ c)(t) = (c^1(t), \ldots, c^n(t))$, then $c^n(0) = 0$ and $c^n(t) > 0$ for $t > 0$. Therefore, the derivative of $c^n$ at $t = 0$ is

$$\dot{c}^n(0) = \lim_{t \to 0^+} \frac{c^n(t)}{t} \ge 0.$$

Since $X_p = \sum_{i=1}^n \dot{c}^i(0)\,\partial/\partial x^i\big|_p$, the coefficient of $\partial/\partial x^n\big|_p$ in $X_p$ is $\dot{c}^n(0)$. In fact, $\dot{c}^n(0) > 0$ because if $\dot{c}^n(0)$ were 0, then $X_p$ would be tangent to $\partial M$ at $p$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 23.4*</span><span class="math-callout__name">(Stokes's theorem for $\mathbb{R}^n$ and for $\mathcal{H}^n$)</span></p>

An $(n-1)$-form $\omega$ with compact support on $\mathbb{R}^n$ or $\mathcal{H}^n$ is a linear combination

$$\omega = \sum_{i=1}^n f_i\, dx^1 \wedge \cdots \wedge \widehat{dx^i} \wedge \cdots \wedge dx^n.$$

Since both sides of Stokes's theorem are $\mathbb{R}$-linear in $\omega$, it suffices to check the theorem for just one term of the sum. So we may assume $\omega = f\,dx^1 \wedge \cdots \wedge \widehat{dx^i} \wedge \cdots \wedge dx^n$, where $f$ is a $C^\infty$ function with compact support. Then

$$d\omega = (-1)^{i-1} \frac{\partial f}{\partial x^i}\,dx^1 \wedge \cdots \wedge dx^i \wedge \cdots \wedge dx^n.$$

**Stokes's theorem for $\mathbb{R}^n$:** By Fubini's theorem, one can first integrate with respect to $x^i$:

$$\int_{\mathbb{R}^n} d\omega = (-1)^{i-1} \int_{\mathbb{R}^{n-1}} \left(\int_{-a}^a \frac{\partial f}{\partial x^i}\,dx^i\right) dx^1 \cdots \widehat{dx^i} \cdots dx^n.$$

But $\int_{-a}^a \frac{\partial f}{\partial x^i}\,dx^i = f(\ldots, a, x^{i+1}, \ldots) - f(\ldots, -a, x^{i+1}, \ldots) = 0 - 0 = 0$, because the support of $f$ lies in the interior of $[-a, a]^n$. Hence, $\int_{\mathbb{R}^n} d\omega = 0$. The right-hand side of Stokes's theorem is $\int_{\partial\mathbb{R}^n} \omega = \int_\varnothing \omega = 0$, because $\mathbb{R}^n$ has empty boundary.

**Stokes's theorem for $\mathcal{H}^n$:**

*Case 1:* $i \neq n$. The computation proceeds as for $\mathbb{R}^n$, giving $\int_{\mathcal{H}^n} d\omega = 0$. As for $\int_{\partial\mathcal{H}^n} \omega$, note that $\partial\mathcal{H}^n$ is defined by the equation $x^n = 0$. On $\partial\mathcal{H}^n$, the 1-form $dx^n$ is identically zero. Since $i \neq n$, $\omega = f\,dx^1 \wedge \cdots \wedge \widehat{dx^i} \wedge \cdots \wedge dx^n \equiv 0$ on $\partial\mathcal{H}^n$. So $\int_{\partial\mathcal{H}^n} \omega = 0$. Thus, Stokes's theorem holds in this case.

*Case 2:* $i = n$.

$$\int_{\mathcal{H}^n} d\omega = (-1)^{n-1} \int_{\mathbb{R}^{n-1}} \left(\int_0^\infty \frac{\partial f}{\partial x^n}\,dx^n\right) dx^1 \cdots dx^{n-1}.$$

In this integral,

$$\int_0^\infty \frac{\partial f}{\partial x^n}\,dx^n = f(x^1, \ldots, x^{n-1}, a) - f(x^1, \ldots, x^{n-1}, 0) = -f(x^1, \ldots, x^{n-1}, 0).$$

Hence,

$$\int_{\mathcal{H}^n} d\omega = (-1)^n \int_{\mathbb{R}^{n-1}} f(x^1, \ldots, x^{n-1}, 0)\,dx^1 \cdots dx^{n-1} = \int_{\partial\mathcal{H}^n} \omega$$

because $(-1)^n \mathbb{R}^{n-1}$ is precisely $\partial\mathcal{H}^n$ with its boundary orientation. So Stokes's theorem also holds in this case.

</div>

## Chapters 25–28 — Cohomology

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution 25.4*</span><span class="math-callout__name">(The snake lemma)</span></p>

If we view each column of the given commutative diagram as a cochain complex, then the diagram is a short exact sequence of cochain complexes

$$0 \to \mathcal{A} \to \mathcal{B} \to \mathcal{C} \to 0.$$

By the zig-zag lemma, it gives rise to a long exact sequence in cohomology. In the long exact sequence, $H^0(\mathcal{A}) = \ker \alpha$, $H^1(\mathcal{A}) = A^1 / \operatorname{im} \alpha = \operatorname{coker} \alpha$, and similarly for $\mathcal{B}$ and $\mathcal{C}$.

</div>

## Appendix A — Point-Set Topology

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution A.13*</span><span class="math-callout__name">(The Lindelöf condition)</span></p>

Let $\lbrace B_i \rbrace_{i \in I}$ be a countable basis and $\lbrace U_\alpha \rbrace_{\alpha \in A}$ an open cover of the topological space $S$. For every $p \in U_\alpha$, there exists a $B_i$ such that

$$p \in B_i \subset U_\alpha.$$

Since this $B_i$ depends on $p$ and $\alpha$, we write $i = i(p, \alpha)$. Thus,

$$p \in B_{i(p,\alpha)} \subset U_\alpha.$$

Now let $J$ be the set of all indices $j \in I$ such that $j = i(p, \alpha)$ for some $p$ and some $\alpha$. Then $\bigcup_{j \in J} B_j = S$ because every $p$ in $S$ is contained in some $B_{i(p,\alpha)} = B_j$.

For each $j \in J$, choose an $\alpha(j)$ such that $B_j \subset U_{\alpha(j)}$. Then $S = \bigcup_j B_j \subset \bigcup_j U_{\alpha(j)}$. So $\lbrace U_{\alpha(j)} \rbrace_{j \in J}$ is a countable subcover of $\lbrace U_\alpha \rbrace_{\alpha \in A}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution A.33</span><span class="math-callout__name">(Compact Hausdorff space)</span></p>

Let $S$ be a compact Hausdorff space, and $A$, $B$ two closed subsets of $S$. By Proposition A.30, $A$ and $B$ are compact. By Proposition A.31, for any $a \in A$ there are disjoint open sets $U_a \ni a$ and $V_a \supset B$. Since $A$ is compact, the open cover $\lbrace U_a \rbrace_{a \in A}$ for $A$ has a finite subcover $\lbrace U_{a_i} \rbrace_{i=1}^n$. Let $U = \bigcup_{i=1}^n U_{a_i}$ and $V = \bigcap_{i=1}^n V_{a_i}$. Then $A \subset U$ and $B \subset V$. The open sets $U$ and $V$ are disjoint because if $x \in U \cap V$, then $x \in U_{a_i}$ for some $i$ and $x \in V_{a_i}$ for the same $i$, contradicting the fact that $U_{a_i} \cap V_{a_i} = \varnothing$.

</div>

## Appendix B — The Rank of a Matrix

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution B.1*</span><span class="math-callout__name">(The rank of a matrix)</span></p>

$(\Rightarrow)$ Suppose $\operatorname{rk} A \ge k$. Then one can find $k$ linearly independent columns, which we call $a_1, \ldots, a_k$. Since the $m \times k$ matrix $[a_1 \cdots a_k]$ has rank $k$, it has $k$ linearly independent rows $b^1, \ldots, b^k$. The matrix $B$ whose rows are $b^1, \ldots, b^k$ is a $k \times k$ submatrix of $A$, and $\operatorname{rk} B = k$. In other words, $B$ is a nonsingular $k \times k$ submatrix of $A$.

$(\Leftarrow)$ Suppose $A$ has a nonsingular $k \times k$ submatrix $B$. Let $a_1, \ldots, a_k$ be the columns of $A$ such that the submatrix $[a_1 \cdots a_k]$ contains $B$. Since $[a_1 \cdots a_k]$ has $k$ linearly independent rows, it also has $k$ linearly independent columns. Thus, $\operatorname{rk} A \ge k$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Solution B.4*</span><span class="math-callout__name">(Degeneracy loci and maximal-rank locus of a map)</span></p>

**(a)** Let $D_r$ be the subset of $\mathbb{R}^{m \times n}$ consisting of matrices of rank at most $r$. The degeneracy locus of rank $r$ of the map $F \colon S \to \mathbb{R}^{m \times n}$ may be described as

$$D_r(F) = \lbrace x \in S \mid F(x) \in D_r \rbrace = F^{-1}(D_r).$$

Since $D_r$ is a closed subset of $\mathbb{R}^{m \times n}$ (Problem B.2) and $F$ is continuous, $F^{-1}(D_r)$ is a closed subset of $S$.

**(b)** Let $D_{\max}$ be the subset of $\mathbb{R}^{m \times n}$ consisting of all matrices of maximal rank. Then $D_{\max}(F) = F^{-1}(D_{\max})$. Since $D_{\max}$ is open in $\mathbb{R}^{m \times n}$ (Problem B.3) and $F$ is continuous, $F^{-1}(D_{\max})$ is open in $S$.

</div>

# List of Notations

A reference of symbols and notations used throughout the book, organized by topic.

## Euclidean Spaces and Calculus

| Notation | Meaning |
| --- | --- |
| $\mathbb{R}^n$ | Euclidean space of dimension $n$ |
| $p = (p^1, \ldots, p^n)$ | point in $\mathbb{R}^n$ |
| $C^\infty$ | smooth or infinitely differentiable |
| $\partial f / \partial x^i$ | partial derivative with respect to $x^i$ |
| $f^{(k)}(x)$ | the $k$th derivative of $f(x)$ |
| $B(p, r)$ | open ball with center $p$ and radius $r$ |
| $\overline{B}(p, r)$ | closed ball with center $p$ and radius $r$ |
| $T_p(\mathbb{R}^n)$ or $T_p\mathbb{R}^n$ | tangent space to $\mathbb{R}^n$ at $p$ |
| $D_v f$ | directional derivative of $f$ in the direction of $v$ at $p$ |
| $C_p^\infty$ | algebra of germs of $C^\infty$ functions at $p$ in $\mathbb{R}^n$ |
| $\mathcal{D}_p(\mathbb{R}^n)$ | vector space of derivations at $p$ in $\mathbb{R}^n$ |
| $\delta_j^i$ | Kronecker delta |

## Multilinear Algebra and Differential Forms

| Notation | Meaning |
| --- | --- |
| $\operatorname{Hom}(V, W)$ | space of linear maps $f \colon V \to W$ |
| $V^\vee = \operatorname{Hom}(V, \mathbb{R})$ | dual of a vector space $V$ |
| $L_k(V)$ | vector space of $k$-linear functions on $V$ |
| $A_k(V)$ | vector space of alternating $k$-linear functions on $V$ |
| $S_k$ | group of permutations of $k$ objects |
| $\operatorname{sgn}(\sigma)$ | sign of a permutation $\sigma$ |
| $Sf$ | symmetrizing operator applied to $f$ |
| $Af$ | alternating operator applied to $f$ |
| $f \otimes g$ | tensor product of multilinear functions $f$ and $g$ |
| $f \wedge g$ | wedge product of multicovectors $f$ and $g$ |
| $\bigwedge(V)$ | exterior algebra of a vector space $V$ |
| $I = (i_1, \ldots, i_k)$ | multi-index |
| $\alpha^I$ | $k$-covector $\alpha^{i_1} \wedge \cdots \wedge \alpha^{i_k}$ |
| $\Omega^k(U)$ | vector space of $C^\infty$ $k$-forms on $U$ |
| $\Omega^*(U)$ | direct sum $\bigoplus_{k=0}^n \Omega^k(U)$ |
| $d\omega$ | exterior derivative of $\omega$ |
| $\operatorname{grad} f$ | gradient of a function $f$ |
| $\operatorname{curl} \mathbf{F}$ | curl of a vector field $\mathbf{F}$ |
| $\operatorname{div} \mathbf{F}$ | divergence of a vector field $\mathbf{F}$ |
| $H^k(U)$ | $k$th de Rham cohomology of $U$ |

## Manifolds and Smooth Maps

| Notation | Meaning |
| --- | --- |
| $(U, \phi)$ | chart or coordinate open set |
| $\mathfrak{U} = \lbrace (U_\alpha, \phi_\alpha) \rbrace$ | atlas |
| $M \times N$ | product manifold |
| $S^n$ | unit sphere in $\mathbb{R}^{n+1}$ |
| $\mathbb{R}P^n$ | real projective space of dimension $n$ |
| $G(k, n)$ | Grassmannian of $k$-planes in $\mathbb{R}^n$ |
| $\operatorname{rk} A$ | rank of a matrix $A$ |
| $\Gamma(f)$ | graph of $f$ |
| $\operatorname{GL}(n, K)$ | general linear group over a field $K$ |
| $\operatorname{SL}(n, K)$ | special linear group over a field $K$ |
| $J(f) = [\partial F^i / \partial x^j]$ | Jacobian matrix |
| $F^* h$ | pullback of a function $h$ by a map $F$ |

## Tangent Spaces and Vector Fields

| Notation | Meaning |
| --- | --- |
| $T_p(M)$ or $T_pM$ | tangent space to $M$ at $p$ |
| $\partial/\partial x^i\big\|_p$ | coordinate tangent vector at $p$ |
| $F_{*,p}$ or $F_*$ | differential of $F$ at $p$ |
| $c(t)$ | curve in a manifold |
| $c'(t) := c_*(d/dt\big\|_{t_0})$ | velocity vector of a curve |
| $TM$ | tangent bundle |
| $X_p$ | tangent vector at $p$ |
| $\Gamma(U, E)$ | vector space of $C^\infty$ sections of $E$ over $U$ |
| $[X, Y]$ | Lie bracket of vector fields |
| $\mathfrak{X}(M)$ | Lie algebra of $C^\infty$ vector fields on $M$ |
| $c_t(p)$ | integral curve through $p$ |
| $F_t(q) = F(t, q)$ | local flow |

## Lie Groups and Lie Algebras

| Notation | Meaning |
| --- | --- |
| $\mu \colon G \times G \to G$ | multiplication on a Lie group |
| $\iota \colon G \to G$ | inverse map of a Lie group |
| $\ell_g$, $r_g$ | left and right multiplication by $g$ |
| $S^1$ | unit circle in $\mathbb{C}^\times$ |
| $\exp(X)$ or $e^X$ | exponential of a matrix $X$ |
| $\operatorname{tr}(X)$ | trace |
| $Z(G)$ | center of a group $G$ |
| $\mathrm{SO}(n)$, $\mathrm{U}(n)$, $\mathrm{SU}(n)$ | special orthogonal, unitary, special unitary groups |
| $\mathrm{Sp}(n)$ | compact symplectic group |
| $L(G)$ | algebra of left-invariant vector fields on $G$ |
| $\mathfrak{g}$ | Lie algebra |
| $\mathfrak{gl}(n, \mathbb{R})$, $\mathfrak{sl}(n, \mathbb{R})$ | Lie algebras of $\operatorname{GL}(n,\mathbb{R})$ and $\operatorname{SL}(n,\mathbb{R})$ |
| $\mathfrak{o}(n)$, $\mathfrak{u}(n)$ | Lie algebras of $\mathrm{O}(n)$ and $\mathrm{U}(n)$ |

## Differential Forms on Manifolds

| Notation | Meaning |
| --- | --- |
| $T_p^*(M)$ or $T_p^*M$ | cotangent space at $p$ |
| $T^*M$ | cotangent bundle |
| $F^* \colon T_{F(p)}^* M \to T_p^* N$ | codifferential |
| $\bigwedge^k(V^\vee) = A_k(V)$ | $k$-covectors on $V$ |
| $\omega_p$ | value of a differential form $\omega$ at $p$ |
| $\bigwedge^k(T^*M)$ | $k$th exterior power of the cotangent bundle |
| $\Omega^k(G)^G$ | left-invariant $k$-forms on a Lie group $G$ |
| $\operatorname{supp} \omega$ | support of a $k$-form |
| $F^* \omega$ | pullback of a differential form $\omega$ by $F$ |
| $\omega\big\|_S$ | restriction of $\omega$ to a submanifold $S$ |
| $\mathcal{L}_X Y$ | Lie derivative of $Y$ along $X$ |
| $\mathcal{L}_X \omega$ | Lie derivative of $\omega$ along $X$ |
| $\iota_v \omega$ | interior multiplication of $\omega$ by $v$ |

## Orientation and Integration

| Notation | Meaning |
| --- | --- |
| $(v_1, \ldots, v_n)$ | ordered basis |
| $(M, [\omega])$ | oriented manifold with orientation $[\omega]$ |
| $-M$ | opposite orientation |
| $\mathcal{H}^n$ | closed upper half-space |
| $M^\circ$ | interior of a manifold with boundary |
| $\partial M$ | boundary of a manifold with boundary |
| $\int_U \omega$ | integral of a differential form $\omega$ over $U$ |
| $\Omega_c^k(M)$ | $C^\infty$ $k$-forms with compact support on $M$ |

## Cohomology

| Notation | Meaning |
| --- | --- |
| $Z^k(M)$ | closed $k$-forms on $M$ |
| $B^k(M)$ | exact $k$-forms on $M$ |
| $H^k(M)$ | de Rham cohomology in degree $k$ |
| $[\omega]$ | cohomology class of $\omega$ |
| $H^*(M)$ | the cohomology ring $\bigoplus_{k=0}^n H^k(M)$ |
| $\mathcal{C} = (\lbrace C^k \rbrace_{k \in \mathbb{Z}}, d)$ | cochain complex |
| $(\Omega^*(M), d)$ | de Rham complex |
| $H^k(\mathcal{C})$ | $k$th cohomology of $\mathcal{C}$ |
| $Z^k(\mathcal{C})$, $B^k(\mathcal{C})$ | $k$-cocycles, $k$-coboundaries |
| $d^* \colon H^k(\mathcal{C}) \to H^{k+1}(\mathcal{A})$ | connecting homomorphism |
| $\chi(M)$ | Euler characteristic of $M$ |
| $f \sim g$ | $f$ is homotopic to $g$ |

## Topology

| Notation | Meaning |
| --- | --- |
| $(S, \mathcal{T})$ | a set $S$ with a topology $\mathcal{T}$ |
| $\overline{A}$, $\operatorname{cl}(A)$ | closure of a set $A$ |
| $\operatorname{int}(A)$ | topological interior of $A$ |
| $\operatorname{bd}(A)$ | topological boundary of $A$ |
| $C_x$ | connected component containing $x$ |
| $\operatorname{ac}(A)$ | accumulation points of $A$ |

## Linear Algebra (Appendix D)

| Notation | Meaning |
| --- | --- |
| $\ker f$ | kernel of a homomorphism $f$ |
| $\operatorname{im} f$ | image of a map $f$ |
| $\operatorname{coker} f$ | cokernel of a homomorphism $f$ |
| $v + W$ | coset of a subspace $W$ |
| $V/W$ | quotient vector space of $V$ by $W$ |
| $\prod_\alpha V_\alpha$, $A \times B$ | direct product |
| $\bigoplus_\alpha V_\alpha$, $A \oplus B$ | direct sum |
| $A \oplus_i B$ | internal direct sum |
| $W^\perp$ | orthogonal complement of $W$ |

## Quaternions and Symplectic Group (Appendix E)

| Notation | Meaning |
| --- | --- |
| $\mathbb{H}$ | skew field of quaternions |
| $\operatorname{End}_K(V)$ | algebra of endomorphisms of $V$ over $K$ |
| $\bar{q}$ | conjugate of a quaternion $q$ |
| $\langle x, y \rangle = \bar{x}^T y$ | quaternionic inner product |
| $\mathrm{Sp}(n)$ | compact symplectic group |
| $\mathrm{Sp}(2n, \mathbb{C})$ | complex symplectic group |
| $J = \begin{bmatrix} 0 & I_n \\ -I_n & 0 \end{bmatrix}$ | standard symplectic matrix |
