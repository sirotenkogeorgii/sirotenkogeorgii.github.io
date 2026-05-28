---
layout: default
title: Optimal Transport Old and New
date: 2024-11-01
---

# Optimal Transport Old and New

## Conventions

This chapter collects the main notational conventions used throughout the book. It covers axioms, sets and structures, metric and topological notions, calculus, probability measures, and notation specific to optimal transport.

### Axioms

The book uses the classical axioms of set theory, but not the full axiom of choice — only the classical axiom of "countable dependent choice".

### Sets and Structures

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Basic Set Theory)</span></p>

* $\mathrm{Id}$ is the identity mapping, whatever the space.
* If $A$ is a set, $1\_A$ is the **indicator function** of $A$: $1\_A(x) = 1$ if $x \in A$, and $0$ otherwise. If $F$ is a formula, $1\_F$ is the indicator function of the set defined by $F$.
* If $f$ and $g$ are two functions, $(f, g)$ is the function $x \longmapsto (f(x), g(x))$. The composition $f \circ g$ will often be denoted by $f(g)$.
* $\mathbb{N}$ is the set of *positive* integers: $\mathbb{N} = \lbrace 1, 2, 3, \ldots \rbrace$. A sequence is written $(x\_k)\_{k \in \mathbb{N}}$, or simply $(x\_k)$.
* $\mathbb{R}$ is the set of real numbers. When writing $\mathbb{R}^n$ it is implicitly assumed that $n$ is a positive integer.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Euclidean and Matrix Conventions)</span></p>

* The Euclidean scalar product between two vectors $a$ and $b$ in $\mathbb{R}^n$ is denoted interchangeably by $a \cdot b$ or $\langle a, b \rangle$.
* The Euclidean norm will be denoted simply by $\lvert \cdot \rvert$, independently of the dimension $n$.
* $M\_n(\mathbb{R})$ is the space of real $n \times n$ matrices, and $I\_n$ the $n \times n$ identity matrix.
* The **trace** of a matrix $M$ is denoted by $\mathrm{tr}\, M$, its **determinant** by $\det M$, its **adjoint** by $M^\ast$, and its **Hilbert–Schmidt norm** $\sqrt{\mathrm{tr}(M^\ast M)}$ by $\lVert M \rVert\_{\mathrm{HS}}$ (or just $\lVert M \rVert$).

</div>

### Riemannian Manifolds

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Riemannian Manifolds)</span></p>

Unless otherwise stated, Riemannian manifolds appearing in the text are **finite-dimensional, smooth and complete**.

* If a Riemannian manifold $M$ is given, $n$ denotes its dimension, $d$ the geodesic distance on $M$, and $\mathrm{vol}$ the volume ($= n$-dimensional Hausdorff) measure on $M$.
* The **tangent space** at $x$ is denoted by $T\_x M$, and the **tangent bundle** by $TM$.
* The norm on $T\_x M$ will most of the time be denoted by $\lvert \cdot \rvert$, as in $\mathbb{R}^n$, without explicit mention of the point $x$. The symbol $\lVert \cdot \rVert$ is reserved for special norms or functional norms.
* If $S$ is a set without smooth structure, $T\_x S$ instead denotes the **tangent cone** to $S$ at $x$.
* If $Q$ is a quadratic form defined on $\mathbb{R}^n$ or on the tangent bundle of a manifold, its value on a (tangent) vector $v$ will be denoted by $\langle Q \cdot v, v \rangle$, or simply $Q(v)$.

</div>

### Metric Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Metric Space Notions)</span></p>

* The **closure** of a set $A$ in a metric space is denoted by $\overline{A}$ (the set of all limits of sequences with values in $A$).
* The **open ball** of radius $r$ and center $x$ in a metric space $\mathcal{X}$ is denoted interchangeably by $B(x, r)$ or $B\_r(x)$. The **closed ball** is denoted by $B[x, r]$ or $B\_{r]}(x)$.
* The **diameter** of a metric space $\mathcal{X}$ is denoted by $\mathrm{diam}(\mathcal{X})$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Compactness)</span></p>

* A metric space $\mathcal{X}$ is **locally compact** if every point $x \in \mathcal{X}$ admits a compact neighborhood.
* A metric space $\mathcal{X}$ is **boundedly compact** if every closed and bounded subset of $\mathcal{X}$ is compact.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lipschitz Maps)</span></p>

A map $f$ between metric spaces $(\mathcal{X}, d)$ and $(\mathcal{X}', d')$ is said to be **$C$-Lipschitz** if

$$d'(f(x), f(y)) \le C\, d(x, y) \quad \text{for all } x, y \in \mathcal{X}.$$

The best admissible constant $C$ is then denoted by $\lVert f \rVert\_{\mathrm{Lip}}$.

A map is said to be **locally Lipschitz** if it is Lipschitz on bounded sets, *not necessarily compact* (so it makes sense to speak of a locally Lipschitz map defined almost everywhere).

</div>

### Curves and Geodesics

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Curves in Metric Spaces)</span></p>

A **curve** in a space $\mathcal{X}$ is a continuous map defined on an interval of $\mathbb{R}$, valued in $\mathcal{X}$. The words "curve" and "path" are synonymous. The **time-$t$ evaluation map** $e\_t$ is defined by $e\_t(\gamma) = \gamma\_t = \gamma(t)$.

If $\gamma$ is a curve defined from an interval of $\mathbb{R}$ into a metric space, its **length** is denoted by $\mathcal{L}(\gamma)$, and its **speed** by $\lvert \dot{\gamma} \rvert$.

Usually geodesics will be *minimizing, constant-speed* geodesic curves. If $\mathcal{X}$ is a metric space, $\Gamma(\mathcal{X})$ stands for the space of all geodesics $\gamma : [0, 1] \to \mathcal{X}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Barycenters)</span></p>

Given $x\_0$ and $x\_1$ in a metric space, $[x\_0, x\_1]\_t$ denotes the set of all $t$-barycenters of $x\_0$ and $x\_1$. If $A\_0$ and $A\_1$ are two sets, then $[A\_0, A\_1]\_t$ stands for the set of all $[x\_0, x\_1]\_t$ with $(x\_0, x\_1) \in A\_0 \times A\_1$.

</div>

### Function Spaces

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Function Spaces)</span></p>

* $C(\mathcal{X})$ is the space of continuous functions $\mathcal{X} \to \mathbb{R}$.
* $C\_b(\mathcal{X})$ is the space of **bounded** continuous functions $\mathcal{X} \to \mathbb{R}$.
* $C\_0(\mathcal{X})$ is the space of continuous functions $\mathcal{X} \to \mathbb{R}$ converging to $0$ at infinity.
* All of the above are equipped with the norm of uniform convergence $\lVert \varphi \rVert\_\infty = \sup \lvert \varphi \rvert$.
* $C\_b^k(\mathcal{X})$ is the space of $k$-times continuously differentiable functions $u : \mathcal{X} \to \mathbb{R}$ such that all partial derivatives of $u$ up to order $k$ are bounded; equipped with the norm $\sup \lVert \partial u \rVert\_{C\_b}$.
* $C\_c^k(\mathcal{X})$ is the space of $k$-times continuously differentiable functions with **compact support**.
* $L^p$ is the Lebesgue space of exponent $p$; the space and the measure will often be implicit.
* When the target space is not $\mathbb{R}$ but some other space $\mathcal{Y}$, the notation is transformed in the obvious way: $C(\mathcal{X}; \mathcal{Y})$, etc.

</div>

### Calculus

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Derivatives)</span></p>

* The derivative of a function $u = u(t)$, defined on an interval of $\mathbb{R}$ and valued in $\mathbb{R}^n$ or in a smooth manifold, is denoted by $u'$, or more often by $\dot{u}$.
* The notation $d^+ u / dt$ stands for the **upper right-derivative**: $d^+ u / dt = \limsup\_{s \downarrow 0} [u(t+s) - u(t)] / s$.
* If $u$ is a function of several variables, the partial derivative with respect to the variable $t$ is denoted by $\partial\_t u$, or $\partial u / \partial t$. *The notation $u\_t$ does not stand for $\partial\_t u$, but for $u(t)$.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Differential Operators)</span></p>

* The **gradient** operator is denoted by $\mathrm{grad}$ or simply $\nabla$.
* The **divergence** operator by $\mathrm{div}$ or $\nabla \cdot$.
* The **Laplace** operator by $\Delta$.
* The **Hessian** operator by $\mathrm{Hess}$ or $\nabla^2$ (so $\nabla^2$ *does not* stand for the Laplace operator).
* The notation is the same in $\mathbb{R}^n$ or in a Riemannian manifold. $\Delta$ is the divergence of the gradient, so it is typically a **nonpositive** operator.
* The value of the gradient of $f$ at point $x$ is denoted by $\nabla\_x f$ or $\nabla f(x)$. The notation $\widetilde{\nabla}$ stands for the approximate gradient.
* If $T$ is a map $\mathbb{R}^n \to \mathbb{R}^n$, $\nabla T$ stands for the **Jacobian matrix** of $T$, i.e. the matrix of all partial derivatives $(\partial T\_i / \partial x\_j)$ for $1 \le i, j \le n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Measures and Differential Operators)</span></p>

All differential operators are applied not only to (smooth) functions but also to measures, by duality. For instance, the **Laplacian of a measure** $\mu$ is defined via the identity

$$\int \zeta \, d(\Delta \mu) = \int (\Delta \zeta) \, d\mu \quad (\zeta \in C_c^2).$$

The notation is consistent in the sense that $\Delta(f\, \mathrm{vol}) = (\Delta f)\, \mathrm{vol}$. Similarly, one takes the divergence of a vector-valued measure, etc.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Asymptotic and Miscellaneous)</span></p>

* $f = o(g)$ means $f/g \longrightarrow 0$ (in an asymptotic regime that should be clear from the context), while $f = O(g)$ means that $f/g$ is bounded.
* $\log$ stands for the natural logarithm with base $e$.
* The **positive and negative parts** of $x \in \mathbb{R}$ are defined by $x\_+ = \max(x, 0)$ and $x\_- = \max(-x, 0)$; both are nonnegative, and $\lvert x \rvert = x\_+ + x\_-$.
* The notation $a \wedge b$ will sometimes be used for $\min(a, b)$.

</div>

### Probability Measures

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Basic Measure Theory)</span></p>

* $\delta\_x$ is the **Dirac mass** at point $x$.
* All measures considered in the text are Borel measures on **Polish spaces** (complete, separable metric spaces), equipped with their Borel $\sigma$-algebra.
* A measure is **finite** if it has finite mass, and **locally finite** if it attributes finite mass to compact sets.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Spaces of Measures)</span></p>

* $P(\mathcal{X})$ — the space of Borel probability measures on $\mathcal{X}$.
* $M\_+(\mathcal{X})$ — the space of finite Borel measures.
* $M(\mathcal{X})$ — the space of signed finite Borel measures.
* The **total variation** of $\mu$ is denoted by $\lVert \mu \rVert\_{\mathrm{TV}}$.
* The integral of a function $f$ with respect to a probability measure $\mu$ is denoted interchangeably by $\int f(x) \, d\mu(x)$ or $\int f(x) \, \mu(dx)$ or $\int f \, d\mu$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Negligible Sets and Support)</span></p>

* If $\mu$ is a Borel measure on a topological space $\mathcal{X}$, a set $N$ is said to be **$\mu$-negligible** if $N$ is included in a Borel set of zero $\mu$-measure. Then $\mu$ is said to be **concentrated** on a set $C$ if $\mathcal{X} \setminus C$ is negligible (equivalently $\mu[\mathcal{X} \setminus C] = 0$).
* If $\mu$ is a Borel measure, its **support** $\mathrm{Spt}\, \mu$ is the smallest *closed* set on which it is concentrated.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Push-forward Measure)</span></p>

If $\mu$ is a Borel measure on $\mathcal{X}$, and $T$ is a Borel map $\mathcal{X} \to \mathcal{Y}$, then $T\_\# \mu$ stands for the **image measure** (or push-forward) of $\mu$ by $T$: It is a Borel measure on $\mathcal{Y}$, defined by

$$(T_\# \mu)[A] = \mu[T^{-1}(A)].$$

The law of a random variable $X$ defined on a probability space $(\Omega, \mathbb{P})$ is denoted by $\mathrm{law}(X)$; this is the same as $X\_\# \mathbb{P}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weak Topology)</span></p>

The **weak topology** on $P(\mathcal{X})$ (or topology of weak convergence, or narrow topology) is induced by convergence against $C\_b(\mathcal{X})$, i.e. bounded continuous test functions. If $\mathcal{X}$ is Polish, then $P(\mathcal{X})$ itself is Polish. Unless explicitly stated, the weak-$\ast$ topology of measures (induced by $C\_0(\mathcal{X})$ or $C\_c(\mathcal{X})$) is not used.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Marginals and Conditional Laws)</span></p>

* If $\pi(dx\, dy)$ is a probability measure in two variables $x \in \mathcal{X}$ and $y \in \mathcal{Y}$, its **marginal** (or projection) on $\mathcal{X}$ (resp. $\mathcal{Y}$) is the measure $X\_\# \pi$ (resp. $Y\_\# \pi$), where $X(x,y) = x$, $Y(x,y) = y$.
* If $(x,y)$ is random with law $(x,y) = \pi$, then the **conditional law** of $x$ given $y$ is denoted by $\pi(dx \mid y)$; this is a measurable function $\mathcal{Y} \to P(\mathcal{X})$, obtained by disintegrating $\pi$ along its $y$-marginal. The conditional law of $y$ given $x$ is denoted by $\pi(dy \mid x)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Absolute Continuity)</span></p>

A measure $\mu$ is said to be **absolutely continuous** with respect to a measure $\nu$ if there exists a measurable function $f$ such that $\mu = f\, \nu$.

</div>

### Notation Specific to Optimal Transport

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Couplings and Cost)</span></p>

* If $\mu \in P(\mathcal{X})$ and $\nu \in P(\mathcal{Y})$ are given, then $\Pi(\mu, \nu)$ is the set of all joint probability measures on $\mathcal{X} \times \mathcal{Y}$ whose marginals are $\mu$ and $\nu$.
* $\mathcal{C}(\mu, \nu)$ is the **optimal (total) cost** between $\mu$ and $\nu$. It implicitly depends on the choice of a cost function $c(x,y)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Wasserstein Distance and Space)</span></p>

For any $p \in [1, +\infty)$, $W\_p$ is the **Wasserstein distance** of order $p$, and $P\_p(\mathcal{X})$ is the **Wasserstein space** of order $p$, i.e. the set of probability measures with finite moments of order $p$, equipped with the distance $W\_p$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Absolutely Continuous Measures in Transport)</span></p>

* $P\_c(\mathcal{X})$ is the set of probability measures on $\mathcal{X}$ with **compact support**.
* If a reference measure $\nu$ on $\mathcal{X}$ is specified, then $P^{\mathrm{ac}}(\mathcal{X})$ (resp. $P\_p^{\mathrm{ac}}(\mathcal{X})$, $P\_c^{\mathrm{ac}}(\mathcal{X})$) stands for those elements of $P(\mathcal{X})$ (resp. $P\_p(\mathcal{X})$, $P\_c(\mathcal{X})$) which are **absolutely continuous** with respect to $\nu$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Displacement Convexity)</span></p>

$\mathcal{DC}\_N$ is the **displacement convexity class** of order $N$ ($N$ plays the role of a dimension); this is a family of convex functions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Functionals and Operators)</span></p>

* $U\_\nu$ is a functional defined on $P(\mathcal{X})$; it depends on a convex function $U$ and a reference measure $\nu$ on $\mathcal{X}$.
* $U\_{\pi,\nu}^\beta$ is another functional on $P(\mathcal{X})$, which involves not only a convex function $U$ and a reference measure $\nu$, but also a coupling $\pi$ and a distortion coefficient $\beta$, which is a nonnegative function on $\mathcal{X} \times \mathcal{X}$.
* The $\Gamma$ and $\Gamma\_2$ operators are quadratic differential operators associated with a diffusion operator.
* $\beta\_t^{(K,N)}$ is the notation for the **distortion coefficients** that play a prominent role.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Curvature-Dimension Condition)</span></p>

$\mathrm{CD}(K, N)$ means "**curvature-dimension condition** $(K, N)$", which morally means that the Ricci curvature is bounded below by $Kg$ ($K$ a real number, $g$ the Riemannian metric) and the dimension is bounded above by $N$ (a real number which is not less than 1).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Cost Symmetry and Coupling Reversal)</span></p>

* If $c(x,y)$ is a cost function then $\check{c}(y,x) = c(x,y)$.
* Similarly, if $\pi(dx\, dy)$ is a coupling, then $\check{\pi}$ is the coupling obtained by swapping variables: $\check{\pi}(dy\, dx) = \pi(dx\, dy)$, or more rigorously, $\check{\pi} = S\_\# \pi$, where $S(x, y) = (y, x)$.

</div>

## Chapter 1: Couplings and Changes of Variables

Couplings are well-known in all branches of probability theory and will occur repeatedly throughout the book. This chapter provides basic reminders and addresses a few technical issues.

### Couplings

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.1</span><span class="math-callout__name">(Coupling)</span></p>

Let $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$ be two probability spaces. **Coupling** $\mu$ and $\nu$ means constructing two random variables $X$ and $Y$ on some probability space $(\Omega, \mathbb{P})$, such that $\mathrm{law}(X) = \mu$, $\mathrm{law}(Y) = \nu$. The couple $(X, Y)$ is called a **coupling** of $(\mu, \nu)$. By abuse of language, the law of $(X, Y)$ is also called a coupling of $(\mu, \nu)$.

</div>

Without loss of generality one may choose $\Omega = \mathcal{X} \times \mathcal{Y}$. In a more measure-theoretic formulation, coupling $\mu$ and $\nu$ means constructing a measure $\pi$ on $\mathcal{X} \times \mathcal{Y}$ such that $\pi$ admits $\mu$ and $\nu$ as **marginals** on $\mathcal{X}$ and $\mathcal{Y}$ respectively. The following three statements are equivalent ways to rephrase the marginal condition:

* $(\mathrm{proj}\_\mathcal{X})\_\# \pi = \mu$, $(\mathrm{proj}\_\mathcal{Y})\_\# \pi = \nu$, where $\mathrm{proj}\_\mathcal{X}$ and $\mathrm{proj}\_\mathcal{Y}$ stand for the projection maps $(x,y) \longmapsto x$ and $(x,y) \longmapsto y$;
* For all measurable sets $A \subset \mathcal{X}$, $B \subset \mathcal{Y}$, one has $\pi[A \times \mathcal{Y}] = \mu[A]$, $\pi[\mathcal{X} \times B] = \nu[B]$;
* For all integrable (resp. nonnegative) measurable functions $\varphi, \psi$ on $\mathcal{X}, \mathcal{Y}$,

$$\int_{\mathcal{X} \times \mathcal{Y}} \bigl(\varphi(x) + \psi(y)\bigr) \, d\pi(x,y) = \int_{\mathcal{X}} \varphi \, d\mu + \int_{\mathcal{Y}} \psi \, d\nu.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Trivial Coupling)</span></p>

Couplings always exist: at least there is the **trivial coupling**, in which $X$ and $Y$ are **independent** (their joint law is the tensor product $\mu \otimes \nu$). This can hardly be called a coupling, since the value of $X$ gives no information about the value of $Y$. The other extreme is when all the information about $Y$ is contained in the value of $X$, i.e. $Y$ is just a function of $X$.

</div>

### Deterministic Couplings

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.2</span><span class="math-callout__name">(Deterministic Coupling)</span></p>

With the notation of Definition 1.1, a coupling $(X, Y)$ is said to be **deterministic** if there exists a measurable function $T : \mathcal{X} \to \mathcal{Y}$ such that $Y = T(X)$.

</div>

To say that $(X, Y)$ is a deterministic coupling of $\mu$ and $\nu$ is strictly equivalent to any one of the four statements below:

* $(X, Y)$ is a coupling of $\mu$ and $\nu$ whose law $\pi$ is concentrated on the *graph* of a measurable function $T : \mathcal{X} \to \mathcal{Y}$;
* $X$ has law $\mu$ and $Y = T(X)$, where $T\_\# \mu = \nu$;
* $X$ has law $\mu$ and $Y = T(X)$, where $T$ is a **change of variables** from $\mu$ to $\nu$: for all $\nu$-integrable (resp. nonnegative measurable) functions $\varphi$,

$$\int_{\mathcal{Y}} \varphi(y) \, d\nu(y) = \int_{\mathcal{X}} \varphi\bigl(T(x)\bigr) \, d\mu(x);$$

* $\pi = (\mathrm{Id}, T)\_\# \mu$.

The map $T$ appearing in all these statements is the same and is uniquely defined $\mu$-almost surely. It is common to call $T$ the **transport map**: informally, $T$ transports the mass represented by $\mu$ to the mass represented by $\nu$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Existence of Deterministic Couplings)</span></p>

Unlike couplings, deterministic couplings do not always exist: just think of the case when $\mu$ is a Dirac mass and $\nu$ is not. But there may also be infinitely many deterministic couplings between two given probability measures.

</div>

### Some Famous Couplings

Here are some of the most famous couplings used in mathematics.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Measurable Isomorphism)</span></p>

Let $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$ be two Polish (i.e. complete, separable, metric) probability spaces without atom (i.e. no single point carries a positive mass). Then there exists a (nonunique) measurable bijection $T : \mathcal{X} \to \mathcal{Y}$ such that $T\_\# \mu = \nu$, $(T^{-1})\_\# \nu = \mu$. In that sense, all atomless Polish probability spaces are isomorphic, e.g. isomorphic to $\mathcal{Y} = [0, 1]$ equipped with the Lebesgue measure. In practice, however, the map $T$ is very singular, and the author advises to never use it.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Moser Mapping)</span></p>

Let $\mathcal{X}$ be a smooth compact Riemannian manifold with volume $\mathrm{vol}$, and let $f, g$ be Lipschitz continuous positive probability densities on $\mathcal{X}$; then there exists a deterministic coupling of $\mu = f\, \mathrm{vol}$ and $\nu = g\, \mathrm{vol}$, constructed by resolution of an elliptic equation. There is a somewhat explicit representation of the transport map $T$, and it is as smooth as can be: if $f, g$ are $C^{k,\alpha}$ then $T$ is $C^{k+1,\alpha}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Increasing Rearrangement on $\mathbb{R}$)</span></p>

Let $\mu$, $\nu$ be two probability measures on $\mathbb{R}$; define their cumulative distribution functions by

$$F(x) = \int_{-\infty}^x d\mu, \qquad G(y) = \int_{-\infty}^y d\nu.$$

Define their right-continuous inverses by

$$F^{-1}(t) = \inf\lbrace x \in \mathbb{R};\; F(x) > t \rbrace, \qquad G^{-1}(t) = \inf\lbrace y \in \mathbb{R};\; G(y) > t \rbrace,$$

and set $T = G^{-1} \circ F$. If $\mu$ does not have atoms, then $T\_\# \mu = \nu$. This rearrangement is quite simple, explicit, as smooth as can be, and enjoys good geometric properties.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Knothe–Rosenblatt Rearrangement in $\mathbb{R}^n$)</span></p>

Let $\mu$ and $\nu$ be two probability measures on $\mathbb{R}^n$, such that $\mu$ is absolutely continuous with respect to Lebesgue measure. Define a coupling of $\mu$ and $\nu$ as follows:

*Step 1:* Take the marginal on the first variable: this gives probability measures $\mu\_1(dx\_1)$, $\nu\_1(dy\_1)$ on $\mathbb{R}$, with $\mu\_1$ being atomless. Then define $y\_1 = T\_1(x\_1)$ by the formula for the increasing rearrangement of $\mu\_1$ into $\nu\_1$.

*Step 2:* Take the marginal on the first two variables and disintegrate with respect to the first variable. This gives $\mu\_2(dx\_1\, dx\_2) = \mu\_1(dx\_1)\, \mu\_2(dx\_2 \mid x\_1)$, $\nu\_2(dy\_1\, dy\_2) = \nu\_1(dy\_1)\, \nu\_2(dy\_2 \mid y\_1)$. For each given $y\_1 \in \mathbb{R}$, set $y\_1 = T\_1(x\_1)$, and define $y\_2 = T\_2(x\_2; x\_1)$ by the increasing rearrangement of $\mu\_2(dx\_2 \mid x\_1)$ into $\nu\_2(dy\_2 \mid y\_1)$.

Repeat the construction, adding variables one after the other. After $n$ steps, this produces a map $y = T(x)$ which transports $\mu$ to $\nu$. The Jacobian matrix of $T$ is (by construction) upper triangular with positive entries on the diagonal.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Holley Coupling on a Lattice)</span></p>

Let $\mu$ and $\nu$ be two discrete probabilities on a finite lattice $\Lambda$, say $\lbrace 0, 1 \rbrace^N$, equipped with the natural partial ordering ($x \le y$ if $x\_n \le y\_n$ for all $n$). Assume that

$$\forall x, y \in \Lambda, \qquad \mu[\inf(x,y)]\; \nu[\sup(x,y)] \ge \mu[x]\; \nu[y].$$

Then there exists a coupling $(X, Y)$ of $(\mu, \nu)$ with $X \le Y$. This appears in connection with the FKG (Fortuin–Kasteleyn–Ginibre) inequalities and intuitively says that $\nu$ puts more mass on large values than $\mu$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Other Famous Couplings)</span></p>

* **Probabilistic representation formulas** for solutions of PDEs: there are hundreds of them, representing solutions of diffusion, transport or jump processes as the laws of various deterministic or stochastic processes.
* The **exact coupling** of two stochastic processes, or Markov chains: two realizations are started at initial time, and when they happen to be in the same state at some time, they are merged (follow the same path from then on). For independent starts this is the **classical coupling**. Variants include the **Ornstein coupling**, the **$\varepsilon$-coupling**, and the **shift-coupling**.

</div>

### Optimal Coupling and the Monge–Kantorovich Problem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Monge–Kantorovich Minimization Problem)</span></p>

Given a **cost function** $c(x,y)$ on $\mathcal{X} \times \mathcal{Y}$ (interpreted as the work needed to move one unit of mass from $x$ to $y$), the **Monge–Kantorovich minimization problem** is

$$\inf \int_{\mathcal{X} \times \mathcal{Y}} c(x,y) \, d\pi(x,y),$$

where the infimum runs over all joint probability measures $\pi$ on $\mathcal{X} \times \mathcal{Y}$ with marginals $\mu$ and $\nu$. Such joint measures are called **transference plans** (or transport plans); those achieving the infimum are called **optimal transference plans**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Total Variation as Optimal Transport)</span></p>

Even the apparently trivial choice $c(x,y) = 1\_{x \ne y}$ has a probabilistic interpretation of total variation:

$$\lVert \mu - \nu \rVert_{TV} = 2 \inf\lbrace \mathbb{E}\, 1_{X \ne Y};\; \mathrm{law}(X) = \mu,\; \mathrm{law}(Y) = \nu \rbrace.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Monge Problem)</span></p>

The search of deterministic optimal couplings (Monge couplings) is called the **Monge problem**. A solution yields a plan to transport the mass at minimal cost, associating to each point $x$ a single point $y$ ("*No mass shall be split*"). To guarantee existence, two kinds of assumptions are natural: first, $c$ should "vary enough"; second, $\mu$ should enjoy some regularity (at least Dirac masses should be ruled out). A typical result: if $c(x,y) = \lvert x - y \rvert^2$ in the Euclidean space, $\mu$ is absolutely continuous with respect to Lebesgue measure, and $\mu, \nu$ have finite moments of order 2, then there is a unique optimal Monge coupling between $\mu$ and $\nu$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Optimal Couplings)</span></p>

Optimal couplings enjoy several nice properties:

1. They naturally arise in many problems from economics, physics, PDEs or geometry (the increasing rearrangement and the Holley coupling are particular cases of optimal transport);
2. They are quite stable with respect to perturbations;
3. They encode good geometric information, if the cost function $c$ is defined in terms of the underlying geometry;
4. They exist in smooth as well as nonsmooth settings;
5. They come with a rich structure: an **optimal cost** functional, a **dual variational problem**, and under adequate structure conditions, a continuous **interpolation**.

On the negative side, optimal transport is in general not smooth. There are known counterexamples limiting the regularity one can expect, even for very nice cost functions.

</div>

### Gluing

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Gluing Lemma)</span></p>

Let $(\mathcal{X}\_i, \mu\_i)$, $i = 1, 2, 3$, be Polish probability spaces. If $(X\_1, X\_2)$ is a coupling of $(\mu\_1, \mu\_2)$ and $(Y\_2, Y\_3)$ is a coupling of $(\mu\_2, \mu\_3)$, then one can construct a triple of random variables $(Z\_1, Z\_2, Z\_3)$ such that $(Z\_1, Z\_2)$ has the same law as $(X\_1, X\_2)$ and $(Z\_2, Z\_3)$ has the same law as $(Y\_2, Y\_3)$.

</div>

The idea is simple: if $\pi\_{12}$ is the law of $(X\_1, X\_2)$ on $\mathcal{X}\_1 \times \mathcal{X}\_2$ and $\pi\_{23}$ is the law of $(X\_2, X\_3)$ on $\mathcal{X}\_2 \times \mathcal{X}\_3$, one just has to *glue* $\pi\_{12}$ and $\pi\_{23}$ along their common marginal $\mu\_2$. Disintegrate $\pi\_{12}$ and $\pi\_{23}$ as

$$\pi_{12}(dx_1\, dx_2) = \pi_{12}(dx_1 \mid x_2)\, \mu_2(dx_2), \qquad \pi_{23}(dx_2\, dx_3) = \pi_{23}(dx_3 \mid x_2)\, \mu_2(dx_2),$$

and then reconstruct $\pi\_{123}$ as

$$\pi_{123}(dx_1\, dx_2\, dx_3) = \pi_{12}(dx_1 \mid x_2)\, \mu_2(dx_2)\, \pi_{23}(dx_3 \mid x_2).$$

### Change of Variables Formula

When writing the formula for change of variables (say in $\mathbb{R}^n$ or on a Riemannian manifold), a Jacobian term appears. The change of variables should be *injective* and somewhat smooth.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Change of Variables Formula)</span></p>

Let $M$ be an $n$-dimensional Riemannian manifold with a $C^1$ metric, let $\mu\_0$, $\mu\_1$ be two probability measures on $M$, and let $T : M \to M$ be a measurable function such that $T\_\# \mu\_0 = \mu\_1$. Let $\nu$ be a reference measure, of the form $\nu(dx) = e^{-V(x)} \mathrm{vol}(dx)$, where $V$ is continuous and $\mathrm{vol}$ is the volume (or $n$-dimensional Hausdorff) measure. Further assume that

* (i) $\mu\_0(dx) = \rho\_0(x)\, \nu(dx)$ and $\mu\_1(dy) = \rho\_1(y)\, \nu(dy)$;
* (ii) $T$ is injective;
* (iii) $T$ is locally Lipschitz.

Then, $\mu\_0$-almost surely,

$$\rho_0(x) = \rho_1(T(x))\, \mathcal{J}_T(x),$$

where $\mathcal{J}\_T(x)$ is the **Jacobian determinant** of $T$ at $x$, defined by

$$\mathcal{J}_T(x) := \lim_{\varepsilon \downarrow 0} \frac{\nu[T(B_\varepsilon(x))]}{\nu[B_\varepsilon(x)]}.$$

The same holds if $T$ is only defined on the complement of a $\mu\_0$-negligible set.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1.3</span><span class="math-callout__name">(Jacobian Determinant)</span></p>

When $\nu$ is just the volume measure, $\mathcal{J}\_T$ coincides with the usual Jacobian determinant, which in the case $M = \mathbb{R}^n$ is the absolute value of the determinant of the Jacobian matrix $\nabla T$. Since $V$ is continuous, it is almost immediate to deduce the statement with an arbitrary $V$ from the statement with $V = 0$ (this amounts to multiplying $\rho\_0(x)$ by $e^{V(x)}$, $\rho\_1(y)$ by $e^{V(y)}$, $\mathcal{J}\_T(x)$ by $e^{V(x) - V(T(x))}$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1.4</span><span class="math-callout__name">(Approximate Differentiability)</span></p>

There is a more general framework beyond differentiability, namely the property of **approximate differentiability**. A function $T$ on an $n$-dimensional Riemannian manifold is said to be approximately differentiable at $x$ if there exists a function $\widetilde{T}$, differentiable at $x$, such that the set $\lbrace \widetilde{T} \ne T \rbrace$ has zero density at $x$:

$$\lim_{r \to 0} \frac{\mathrm{vol}\bigl[\lbrace x \in B_r(x);\; T(x) \ne \widetilde{T}(x) \rbrace\bigr]}{\mathrm{vol}[B_r(x)]} = 0.$$

An approximately differentiable map can be replaced, up to neglecting a small set, by a Lipschitz map. The change of variables formula still holds when assumption (iii) is replaced by (iii') $T$ is approximately differentiable.

</div>

### Conservation of Mass Formula

The single most important theorem of change of variables arising in continuum physics comes from the **conservation of mass** formula.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Conservation of Mass — PDE Form)</span></p>

If $\rho = \rho(t, x)$ is the density of a system of particles at time $t$ and position $x$, and $\xi = \xi(t, x)$ is the velocity field, then the conservation of mass equation reads

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho\, \xi) = 0.$$

More generally, working with particle densities $\mu\_t(dx)$ (not necessarily absolutely continuous), this becomes

$$\frac{\partial \mu}{\partial t} + \nabla \cdot (\mu\, \xi) = 0,$$

where the time-derivative is taken in the weak sense and the divergence operator is defined by duality against continuously differentiable functions with compact support:

$$\int_M \varphi\, \nabla \cdot (\mu\, \xi) = -\int_M (\xi \cdot \nabla \varphi) \, d\mu.$$

</div>

This is an **Eulerian** description of the physical world (unknowns are fields). The next theorem links it with the **Lagrangian** description, in which everything is expressed in terms of particle trajectories, which are integral curves of the velocity field:

$$\xi\bigl(t, T_t(x)\bigr) = \frac{d}{dt} T_t(x).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Mass Conservation Formula)</span></p>

Let $M$ be a $C^1$ manifold, $T \in (0, +\infty]$ and let $\xi(t, x)$ be a (measurable) velocity field on $[0, T) \times M$. Let $(\mu\_t)\_{0 \le t < T}$ be a time-dependent family of probability measures on $M$ (continuous in time for the weak topology), such that

$$\int_0^T \int_M \lvert \xi(t, x) \rvert \, \mu_t(dx)\, dt < +\infty.$$

Then, the following two statements are equivalent:

* (i) $\mu = \mu\_t(dx)$ is a weak solution of the linear (transport) partial differential equation $\partial\_t \mu + \nabla\_x \cdot (\mu\, \xi) = 0$ on $[0, T) \times M$;
* (ii) $\mu\_t$ is the law at time $t$ of a random solution $T\_t(x)$ of $\xi(t, T\_t(x)) = \frac{d}{dt} T\_t(x)$.

If moreover $\xi$ is locally Lipschitz, then $(T\_t)\_{0 \le t < T}$ defines a deterministic flow, and statement (ii) can be rewritten as (ii') $\mu\_t = (T\_t)\_\# \mu\_0$.

</div>

### Diffusion Formula

The final tool in this chapter is related to Itô's formula. The natural assumptions on the phase space involve *Ricci curvature*.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Diffusion Theorem)</span></p>

Let $M$ be a Riemannian manifold with a $C^2$ metric, such that the Ricci curvature tensor of $M$ is uniformly bounded below, and let $\sigma(t, x) : T\_x M \to T\_x M$ be a twice differentiable linear mapping on each tangent space. Let $X\_t$ stand for the solution of the stochastic differential equation

$$dX_t = \sqrt{2}\, \sigma(t, X_t)\, dB_t \qquad (0 \le t < T).$$

Then the following two statements are equivalent:

* (i) $\mu = \mu\_t(dx)$ is a weak solution of the linear (diffusion) partial differential equation

$$\partial_t \mu = \nabla_x \cdot \bigl((\sigma \sigma^*) \nabla_x \mu\bigr)$$

on $M \times [0, T)$, where $\sigma^\ast$ stands for the transpose of $\sigma$;

* (ii) $\mu\_t = \mathrm{law}(X\_t)$ for all $t \in [0, T)$, where $X\_t$ solves the SDE above.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.5</span><span class="math-callout__name">(Heat Equation and Brownian Motion)</span></p>

In $\mathbb{R}^n$, the solution of the heat equation with initial datum $\delta\_0$ is the law of $X\_t = \sqrt{2}\, B\_t$ (Brownian motion sped up by a factor $\sqrt{2}$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1.6</span><span class="math-callout__name">(Ricci Curvature Criterion)</span></p>

There is a finer criterion for the diffusion equation to hold true: it is sufficient that the Ricci curvature at point $x$ be bounded below by $-C\, d(x\_0, x)^2 g\_x$ as $x \to \infty$, where $g\_x$ is the metric at point $x$ and $x\_0$ is an arbitrary reference point. The exponent 2 here is sharp.

</div>

### Appendix: Moser's Coupling

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Moser's Construction)</span></p>

Let $M$ be a smooth $n$-dimensional Riemannian manifold, equipped with a reference probability measure $\nu(dx) = e^{-V(x)} \mathrm{vol}(dx)$, where $V \in C^1(M)$. Let $\mu\_0 = \rho\_0\, \nu$, $\mu\_1 = \rho\_1\, \nu$ be two probability measures on $M$; assume $\rho\_0, \rho\_1$ are bounded below by a constant $K > 0$ and are locally Lipschitz. The key idea is to solve the equation

$$(\Delta - \nabla V \cdot \nabla)\, u = \rho_0 - \rho_1$$

for some $u \in C^{1,1}\_{\mathrm{loc}}(M)$. Then, define a locally Lipschitz vector field

$$\xi(t, x) = \frac{\nabla u(x)}{(1-t)\, \rho_0(x) + t\, \rho_1(x)},$$

with associated flow $(T\_t(x))\_{0 \le t \le 1}$, and a family of probability measures

$$\mu_t = (1-t)\, \mu_0 + t\, \mu_1.$$

Then $\mu\_t$ satisfies the conservation of mass formula, so $\mu\_t = (T\_t)\_\# \mu\_0$. In particular, $T\_1$ pushes $\mu\_0$ forward to $\mu\_1$.

When $M$ is compact and $V = 0$, and $\rho\_0, \rho\_1$ are Lipschitz continuous and positive, the solution $u$ of $\Delta u = \rho\_0 - \rho\_1$ will be of class $C^{2,\alpha}$ for all $\alpha \in (0, 1)$, and $\nabla u$ will be $C^1$ (in fact $C^{1,\alpha}$).

</div>

## Chapter 2: Three Examples of Coupling Techniques

This chapter presents three applications of coupling methods: convergence of the Langevin process, Euclidean isoperimetry, and Caffarelli's log-concave perturbation theorem. The proofs vary greatly in difficulty.

### Convergence of the Langevin Process

Consider a particle subject to a force induced by a potential $V \in C^1(\mathbb{R}^n)$, friction, and random white noise. If $X\_t$ is the position, $m$ the mass, $\lambda$ the friction coefficient, $k$ the Boltzmann constant and $T$ the temperature, Newton's equation becomes

$$m \frac{d^2 X_t}{dt^2} = -\nabla V(X_t) - \lambda\, m \frac{dX_t}{dt} + \sqrt{kT}\, \frac{dB_t}{dt}.$$

In the overdamped limit (high friction, slow motion), the acceleration term is negligible and the equation simplifies to the **Langevin process**:

$$\frac{dX_t}{dt} = -\nabla V(X_t) + \sqrt{2}\, \frac{dB_t}{dt}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Convergence of the Langevin Process via Coupling)</span></p>

Assume $V$ is uniformly convex, i.e. there exists $K > 0$ such that $\nabla^2 V \ge K I\_n$. Consider two solutions of the Langevin equation driven by the *same* Brownian motion $B\_t$:

$$\frac{dX_t}{dt} = -\nabla V(X_t) + \sqrt{2}\, \frac{dB_t}{dt}, \qquad \frac{dY_t}{dt} = -\nabla V(Y_t) + \sqrt{2}\, \frac{dB_t}{dt}.$$

Setting $\alpha\_t := X\_t - Y\_t$, one finds that $\alpha\_t$ is continuously differentiable and

$$\frac{d}{dt} \frac{\lvert \alpha_t \rvert^2}{2} = -\bigl\langle \nabla V(X_t) - \nabla V(Y_t),\; X_t - Y_t \bigr\rangle \le -K\, \lvert \alpha_t \rvert^2.$$

By Gronwall's lemma,

$$\mathbb{E}\, \lvert X_t - Y_t \rvert^2 \le 2\bigl(\mathbb{E}\, \lvert X_0 \rvert^2 + \mathbb{E}\, \lvert Y_0 \rvert^2\bigr)\, e^{-2Kt}.$$

In particular, $X\_t - Y\_t \to 0$ almost surely (independent of the distribution of $Y\_0$), and $\mu\_t := \mathrm{law}(X\_t)$ converges weakly to the stationary distribution

$$\nu(dy) = \frac{e^{-V(y)}\, dy}{Z}, \qquad Z = \int e^{-V}.$$

The convergence is exponentially fast.

</div>

### Euclidean Isoperimetry

Among all subsets of $\mathbb{R}^n$ with given surface, which one has the largest volume? The answer is the ball: this is the **Euclidean isoperimetric inequality**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euclidean Isoperimetric Inequality)</span></p>

For any bounded open set $\Omega \subset \mathbb{R}^n$ with Lipschitz boundary $\partial \Omega$, and any ball $B$,

$$\frac{\lvert \partial \Omega \rvert}{\lvert \Omega \rvert^{\frac{n}{n-1}}} \ge \frac{\lvert \partial B \rvert}{\lvert B \rvert^{\frac{n}{n-1}}}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof by Coupling)</span></p>

*Sketch of proof.* Let $B$ be a ball such that $\lvert \partial B \rvert = \lvert \partial \Omega \rvert$. Consider $X$ uniformly distributed in $\Omega$ and $Y$ uniformly distributed in $B$. Introduce the Knothe–Rosenblatt coupling $Y = T(X)$, so that $\nabla T(x)$ is triangular with nonnegative diagonal entries. Since the law of $X$ (resp. $Y$) has uniform density $1/\lvert \Omega \rvert$ (resp. $1/\lvert B \rvert$), the change of variables formula yields

$$\forall x \in \Omega \qquad \frac{1}{\lvert \Omega \rvert} = \bigl(\det \nabla T(x)\bigr)\, \frac{1}{\lvert B \rvert}.$$

Since $\nabla T$ is triangular, $\det(\nabla T) = \prod \lambda\_i$ and $\nabla \cdot T = \sum \lambda\_i$, where $(\lambda\_i)$ are the eigenvalues. By the arithmetic-geometric inequality, $(\prod \lambda\_i)^{1/n} \le (\sum \lambda\_i)/n$, so

$$\frac{1}{\lvert \Omega \rvert^{1/n}} \le \frac{(\nabla \cdot T)(x)}{n\, \lvert B \rvert^{1/n}}.$$

Integrating over $\Omega$ and applying the divergence theorem,

$$\lvert \Omega \rvert^{1 - 1/n} \le \frac{1}{n\, \lvert B \rvert^{1/n}} \int_{\partial \Omega} (T \cdot \sigma)\, d\mathcal{H}^{n-1},$$

where $\sigma$ is the unit outer normal to $\Omega$ and $\mathcal{H}^{n-1}$ is the $(n-1)$-dimensional Hausdorff measure. Since $T$ is valued in $B$, $\lvert T \cdot \sigma \rvert \le 1$, giving $\lvert \Omega \rvert^{1-1/n} \le \lvert \partial \Omega \rvert / (n\, \lvert B \rvert^{1/n})$. Since $\lvert \partial \Omega \rvert = \lvert \partial B \rvert = n\lvert B \rvert$, the right-hand side is $\lvert B \rvert^{1-1/n}$, so the volume of $\Omega$ is bounded by the volume of $B$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Open Problem 2.1</span></p>

Can one devise an optimal coupling between sets (in the sense of a coupling between the uniform probability measures on these sets) in such a way that the total cost of the coupling decreases under some evolution converging to balls, such as mean curvature motion?

</div>

### Caffarelli's Log-Concave Perturbation Theorem

The idea is to "transport" functional inequalities from a model space to another space. Let $F, G, H, J, L$ be nonnegative continuous functions on $\mathbb{R}$, with $H$ and $J$ nondecreasing, and let $\ell \in \mathbb{R}$. For a given measure $\mu$ on $\mathbb{R}^n$, let $\lambda[\mu]$ be the largest $\lambda \ge 0$ such that, for all Lipschitz functions $h : \mathbb{R}^n \to \mathbb{R}$,

$$\int_{\mathbb{R}^n} L(h)\, d\mu = \ell \implies F\!\left(\int_{\mathbb{R}^n} G(h)\, d\mu\right) \le \frac{1}{\lambda}\, H\!\left(\int_{\mathbb{R}^n} J(\lvert \nabla h \rvert)\, d\mu\right).$$

Functional inequalities of this form are variants of Sobolev inequalities.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Caffarelli's Log-Concave Perturbation Theorem)</span></p>

If $d\mu / d\gamma$ is log-concave (where $\gamma$ is the standard Gaussian measure), then there exists a 1-Lipschitz change of variables from $\gamma$ to $\mu$. That is, there exists a deterministic coupling $(X, Y = \mathcal{C}(X))$ of $(\gamma, \mu)$ such that $\lvert \mathcal{C}(x) - \mathcal{C}(y) \rvert \le \lvert x - y \rvert$, or equivalently $\lvert \nabla \mathcal{C} \rvert \le 1$ (almost everywhere).

It follows in particular that

$$\lvert \nabla(h \circ \mathcal{C}) \rvert \le \lvert (\nabla h) \circ \mathcal{C} \rvert$$

for whatever function $h$. As a consequence, if $\mu = e^{-v} \gamma$ with $v$ convex, then

$$\lambda[\mu] \ge \lambda[\gamma].$$

In words: *functional inequalities can only be improved by log-concave perturbation of the Gaussian distribution*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Caffarelli's Theorem)</span></p>

The existence of the map $\mathcal{C}$ implies the inequality: by the change of variables formula,

$$\int G(h)\, d\mu = \int G(h \circ \mathcal{C})\, d\gamma, \qquad \int L(h)\, d\mu = \int L(h \circ \mathcal{C})\, d\gamma;$$

and by the 1-Lipschitz property and the nondecreasing nature of $J$,

$$\int J(\lvert \nabla h \rvert)\, d\mu = \int J(\lvert \nabla h \circ \mathcal{C} \rvert)\, d\gamma \ge \int J(\lvert \nabla(h \circ \mathcal{C}) \rvert)\, d\gamma.$$

Thus, inequality (for the functional involving $F, G, H, J, L$) is indeed "transported" from $(\mathbb{R}^n, \gamma)$ to $(\mathbb{R}^n, \mu)$.

</div>

## Chapter 3: The Founding Fathers of Optimal Transport

This chapter provides a historical overview of how optimal transport was born and reborn several times, from Monge in the 18th century to the modern era.

### Gaspard Monge (1746–1818)

The field of optimal transport was born at the end of the eighteenth century, by way of the French geometer Gaspard Monge. Born in 1746 under the French Ancient Régime, Monge was admitted to a military training school despite his modest origin due to his outstanding skills. He invented descriptive geometry on his own, was appointed professor at age 22 (with the understanding that his theory would remain a military secret), and later became one of the most ardent warrior scientists of the French Revolution and one of Napoleon's closest friends.

In 1781 he published *Mémoire sur la théorie des déblais et des remblais* ("déblai" = material extracted from the earth; "remblai" = material input into a new construction). The problem: given a certain amount of soil to extract and transport to places where it should be incorporated in a construction, determine the assignment that minimizes the total transport cost. Monge assumed the cost of transporting one unit of mass along a certain distance was the product of the mass by the distance.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Monge's Problem as Optimal Coupling)</span></p>

Monge's problem can be recast in an economic perspective: consider bakeries producing loaves that should be transported to cafés. The amounts produced and consumed are modeled as probability measures (a "density of production" and a "density of consumption") on a space equipped with its natural metric. The problem is to find where each unit of bread should go, minimizing total transport cost. So Monge's problem is really the search of an optimal coupling — and to be more precise, a *deterministic* optimal coupling.

Monge studied the problem in three dimensions for a continuous distribution of mass and made the important geometric observation that transport should go along straight lines orthogonal to a family of surfaces. This led him to the discovery of *lines of curvature*, a concept that was a great contribution to the geometry of surfaces.

</div>

### Leonid Kantorovich (1912–1986)

Much later, Monge's problem was rediscovered by the Russian mathematician Leonid Vitaliyevich Kantorovich. Born in 1912, Kantorovich was a prodigy who earned a first-class researcher reputation at 18 and became a professor at the same age as Monge. He worked in many areas of mathematics, with a strong taste for applications in economics and theoretical computer science.

In 1938, a laboratory consulted him for the solution of an optimization problem, which he recognized as representative of a whole class of linear problems. This led him to develop the tools of **linear programming**, which later became prominent in economics. His most important works were delayed and suppressed by Soviet authorities due to their economic content. In 1975 he was awarded the Nobel Prize for economics, jointly with Tjalling Koopmans.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Kantorovich's Key Contributions)</span></p>

In the case of optimal coupling, Kantorovich stated and proved, by means of functional analytical tools, a **duality theorem** that would play a crucial role later. He also devised a convenient notion of distance between probability measures: the distance between two measures should be the optimal transport cost from one to the other, if the cost is chosen as the distance function. This distance is nowadays called the **Kantorovich–Rubinstein distance** (also known as the Wasserstein distance), and has proven to be particularly flexible and useful.

It was only several years after his main results that Kantorovich made the connection with Monge's work. The problem of optimal coupling has since been called the **Monge–Kantorovich problem**.

</div>

### The Modern Era (Late 1980s Onward)

Throughout the second half of the twentieth century, optimal coupling techniques and variants of the Kantorovich–Rubinstein distance were used by statisticians and probabilists. Noticeable contributions from the seventies are due to Roland Dobrushin (distances in the study of particle systems) and Hiroshi Tanaka (the Boltzmann equation). By the mid-eighties, specialists like Svetlozar Rachev and Ludger Rüschendorf possessed a large library of tools and applications.

At the end of the eighties, three directions of research emerged independently and almost simultaneously, completely reshaping the field:

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Three Revolutions in Optimal Transport)</span></p>

1. **John Mather** and Lagrangian dynamical systems: Mather found it convenient to study not just action-minimizing curves, but action-minimizing stationary *measures* in phase space. These **Mather's measures** solve a variational problem which is in effect a Monge–Kantorovich problem. Under certain conditions, Mather proved that certain action-minimizing measures are automatically concentrated on Lipschitz graphs — intimately related to the construction of a deterministic optimal coupling.

2. **Yann Brenier** and incompressible fluid mechanics: Brenier needed to construct an operator acting like the projection on the set of measure-preserving mappings. He achieved this by introducing an optimal coupling, revealing an unexpected link between optimal transport and fluid mechanics. By pointing out the relation with **Monge–Ampère equations**, he attracted the attention of the PDE community.

3. **Mike Cullen** and meteorology: Cullen's group of meteorologists worked on **semi-geostrophic equations** used in the modeling of atmospheric fronts. They showed that a famous change of variables due to Brian Hoskins could be re-interpreted as an optimal coupling problem, identifying the minimization property as a *stability* condition. This demonstrated that optimal transport could arise naturally in PDEs which seemed to have nothing to do with it.

All three contributions emphasized that *important information can be gained by a qualitative description of optimal transport*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Felix Otto's Differential Viewpoint)</span></p>

An important conceptual step was accomplished by **Felix Otto**, who discovered an appealing formalism introducing a *differential* point of view in optimal transport theory. This opened the way to a more geometric description of the space of probability measures, and connected optimal transport to the theory of diffusion equations, leading to a rich interplay of geometry, functional analysis and PDEs.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Applications of Optimal Transport)</span></p>

Nowadays optimal transport has been applied to diverse topics including:

* Meteorology and fluid mechanics
* Modeling of sandpiles and compression molding
* Image processing and shape recognition
* Design of reflector antennas (cost function $c(x,y) = -\log(1 - x \cdot y)$ on $S^2$)
* Optimal design of lenses and refraction problems
* Irrigation and network design
* City planning
* Electrodynamic equations of Maxwell and string theory
* Reconstruction of the "conditions of the initial Universe"
* Kinetic theory of granular media

Many generalizations and variants have also been studied, including the **optimal matching**, the **optimal transshipment**, the **optimal transport of a fraction of the mass**, and the **optimal coupling with more than two prescribed marginals**.

</div>

## Part I: Qualitative Description of Optimal Transport

The first part of the book is devoted to the description and characterization of optimal transport under certain regularity assumptions on the measures and the cost function.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Overview</span><span class="math-callout__name">(Part I Structure)</span></p>

* **Chapters 4–5:** General theorems about optimal transport plans, in particular the **Kantorovich duality theorem**. Emphasis on $c$-cyclically monotone maps; very general assumptions on the cost function and spaces.
* **Chapter 6:** Natural distance functions on spaces of probability measures derived from the Monge–Kantorovich problem, choosing the cost function as a power of the distance (**Wasserstein distances**).
* **Chapter 7:** Time-dependent version of the Monge–Kantorovich problem, leading to **displacement interpolation** between probability measures. The natural assumption is that the cost function derives from a Lagrangian action.
* **Chapter 8:** Regularity properties of the displacement interpolant, recovered by a strategy due to Mather under smoothness and convexity assumptions.
* **Chapters 9–10:** Existence of deterministic optimal couplings and characterization of the associated transport maps, under regularity and convexity assumptions. The Change of Variables Formula is treated in Chapter 11.
* **Chapter 12:** Regularity of the transport map (which is in general not smooth).
* **Chapter 13:** Synthesis of the main results of Part I. A good understanding of this chapter is sufficient to proceed to Part II.

</div>

## Chapter 4: Basic Properties

This chapter establishes the fundamental properties of optimal transport plans: existence, lower semicontinuity, tightness, restriction, convexity, and a first look at the structure of optimal plans.

### Existence

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.1</span><span class="math-callout__name">(Existence of an Optimal Coupling)</span></p>

Let $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$ be two Polish probability spaces; let $a : \mathcal{X} \to \mathbb{R} \cup \lbrace -\infty \rbrace$ and $b : \mathcal{Y} \to \mathbb{R} \cup \lbrace -\infty \rbrace$ be two upper semicontinuous functions such that $a \in L^1(\mu)$, $b \in L^1(\nu)$. Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ be a lower semicontinuous cost function, such that $c(x,y) \ge a(x) + b(y)$ for all $x, y$. Then there is a coupling of $(\mu, \nu)$ which minimizes the total cost $\mathbb{E}\, c(X, Y)$ among all possible couplings $(X, Y)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.2</span><span class="math-callout__name">(Well-definedness of Cost)</span></p>

The lower bound assumption on $c$ guarantees that the expected cost $\mathbb{E}\, c(X, Y)$ is well-defined in $\mathbb{R} \cup \lbrace +\infty \rbrace$. In most cases of applications — but not all — one may choose $a = 0$, $b = 0$.

</div>

The proof relies on basic variational arguments involving the topology of weak convergence (imposed by bounded continuous test functions). Two key properties are needed: (a) lower semicontinuity, (b) compactness.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Prokhorov's Theorem)</span></p>

If $\mathcal{X}$ is a Polish space, then a set $\mathcal{P} \subset P(\mathcal{X})$ is precompact for the weak topology if and only if it is **tight**, i.e. for any $\varepsilon > 0$ there is a compact set $K\_\varepsilon$ such that $\mu[\mathcal{X} \setminus K\_\varepsilon] \le \varepsilon$ for all $\mu \in \mathcal{P}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.3</span><span class="math-callout__name">(Lower Semicontinuity of the Cost Functional)</span></p>

Let $\mathcal{X}$ and $\mathcal{Y}$ be two Polish spaces, and $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ a lower semicontinuous cost function. Let $h : \mathcal{X} \times \mathcal{Y} \to \mathbb{R} \cup \lbrace -\infty \rbrace$ be an upper semicontinuous function such that $c \ge h$. Let $(\pi\_k)\_{k \in \mathbb{N}}$ be a sequence of probability measures on $\mathcal{X} \times \mathcal{Y}$, converging weakly to some $\pi \in P(\mathcal{X} \times \mathcal{Y})$, in such a way that $h \in L^1(\pi\_k)$, $h \in L^1(\pi)$, and

$$\int_{\mathcal{X} \times \mathcal{Y}} h\, d\pi_k \xrightarrow{k \to \infty} \int_{\mathcal{X} \times \mathcal{Y}} h\, d\pi.$$

Then

$$\int_{\mathcal{X} \times \mathcal{Y}} c\, d\pi \le \liminf_{k \to \infty} \int_{\mathcal{X} \times \mathcal{Y}} c\, d\pi_k.$$

In particular, if $c$ is nonnegative, then $F : \pi \mapsto \int c\, d\pi$ is lower semicontinuous on $P(\mathcal{X} \times \mathcal{Y})$ equipped with the topology of weak convergence.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.4</span><span class="math-callout__name">(Tightness of Transference Plans)</span></p>

Let $\mathcal{X}$ and $\mathcal{Y}$ be two Polish spaces. Let $\mathcal{P} \subset P(\mathcal{X})$ and $\mathcal{Q} \subset P(\mathcal{Y})$ be tight subsets of $P(\mathcal{X})$ and $P(\mathcal{Y})$ respectively. Then the set $\Pi(\mathcal{P}, \mathcal{Q})$ of all transference plans whose marginals lie in $\mathcal{P}$ and $\mathcal{Q}$ respectively, is itself tight in $P(\mathcal{X} \times \mathcal{Y})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 4.1)</span></p>

Since $\mathcal{X}$ is Polish, $\lbrace \mu \rbrace$ is tight in $P(\mathcal{X})$; similarly $\lbrace \nu \rbrace$ is tight in $P(\mathcal{Y})$. By Lemma 4.4, $\Pi(\mu, \nu)$ is tight in $P(\mathcal{X} \times \mathcal{Y})$, and by Prokhorov's theorem it has a compact closure. Passing to the limit in the equation for marginals shows that $\Pi(\mu, \nu)$ is closed, hence compact.

Then let $(\pi\_k)$ be a sequence with $\int c\, d\pi\_k$ converging to the infimum transport cost. Extract a subsequence converging weakly to some $\pi \in \Pi(\mu, \nu)$. Lemma 4.3 gives $\int c\, d\pi \le \liminf \int c\, d\pi\_k$, so $\pi$ is minimizing.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.5</span><span class="math-callout__name">(Finiteness of Optimal Cost)</span></p>

The existence theorem does not imply that the optimal cost is finite. It might be that *all* transport plans lead to an infinite total cost, i.e. $\int c\, d\pi = +\infty$ for all $\pi \in \Pi(\mu, \nu)$. A simple condition to rule this out is

$$\int c(x, y)\, d\mu(x)\, d\nu(y) < +\infty,$$

which guarantees that at least the independent coupling has finite total cost. A stronger assumption is $c(x, y) \le c\_\mathcal{X}(x) + c\_\mathcal{Y}(y)$ with $(c\_\mathcal{X}, c\_\mathcal{Y}) \in L^1(\mu) \times L^1(\nu)$, which implies that *any* coupling has finite total cost.

</div>

### Restriction Property

The second good property of optimal couplings is that *any sub-coupling is still optimal*: if you have an optimal transport plan, then any induced sub-plan (transferring part of the initial mass to part of the final mass) has to be optimal too — otherwise you would be able to lower the cost of the sub-plan and consequently the cost of the whole plan.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.6</span><span class="math-callout__name">(Optimality is Inherited by Restriction)</span></p>

Let $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$ be two Polish spaces, $a \in L^1(\mu)$, $b \in L^1(\nu)$, let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ be a measurable cost function such that $c(x,y) \ge a(x) + b(y)$ for all $x, y$; and let $C(\mu, \nu)$ be the optimal transport cost from $\mu$ to $\nu$. Assume that $C(\mu, \nu) < +\infty$ and let $\pi \in \Pi(\mu, \nu)$ be an optimal transport plan. Let $\widetilde{\pi}$ be a nonnegative measure on $\mathcal{X} \times \mathcal{Y}$, such that $\widetilde{\pi} \le \pi$ and $\widetilde{\pi}[\mathcal{X} \times \mathcal{Y}] > 0$. Then the probability measure

$$\pi' := \frac{\widetilde{\pi}}{\widetilde{\pi}[\mathcal{X} \times \mathcal{Y}]}$$

is an optimal transference plan between its marginals $\mu'$ and $\nu'$.

Moreover, if $\pi$ is the *unique* optimal transference plan between $\mu$ and $\nu$, then also $\pi'$ is the unique optimal transference plan between $\mu'$ and $\nu'$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.7</span><span class="math-callout__name">(Restriction to a Measurable Set)</span></p>

If $(X, Y)$ is an optimal coupling of $(\mu, \nu)$, and $\mathcal{Z} \subset \mathcal{X} \times \mathcal{Y}$ is such that $\mathbb{P}[(X, Y) \in \mathcal{Z}] > 0$, then the pair $(X, Y)$ conditioned to lie in $\mathcal{Z}$ is an optimal coupling of $(\mu', \nu')$, where $\mu'$ is the law of $X$ conditioned by "$(X, Y) \in \mathcal{Z}$", and $\nu'$ is the law of $Y$ conditioned by the same event.

</div>

### Convexity Properties

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.8</span><span class="math-callout__name">(Convexity of the Optimal Cost)</span></p>

Let $\mathcal{X}$ and $\mathcal{Y}$ be two Polish spaces, let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ be a lower semicontinuous function, and let $C$ be the associated optimal transport cost functional on $P(\mathcal{X}) \times P(\mathcal{Y})$. Let $(\Theta, \lambda)$ be a probability space, and let $\mu\_\theta$, $\nu\_\theta$ be two measurable functions defined on $\Theta$, with values in $P(\mathcal{X})$ and $P(\mathcal{Y})$ respectively. Assume that $c(x,y) \ge a(x) + b(y)$, where $a \in L^1(d\mu\_\theta\, d\lambda(\theta))$, $b \in L^1(d\nu\_\theta\, d\lambda(\theta))$. Then

$$C\!\left(\int_\Theta \mu_\theta\, \lambda(d\theta),\; \int_\Theta \nu_\theta\, \lambda(d\theta)\right) \le \int_\Theta C(\mu_\theta, \nu_\theta)\, \lambda(d\theta).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of Convexity)</span></p>

In words: the cost of transporting a mixture is at most the mixture of the costs. The proof constructs $\pi := \int \pi\_\theta\, \lambda(d\theta)$, where each $\pi\_\theta$ is an optimal plan for $(\mu\_\theta, \nu\_\theta)$. Then $\pi$ has the correct marginals $\mu = \int \mu\_\theta\, d\lambda$ and $\nu = \int \nu\_\theta\, d\lambda$, and its cost $\int c\, d\pi = \int C(\mu\_\theta, \nu\_\theta)\, d\lambda$ provides an upper bound for $C(\mu, \nu)$.

</div>

### Description of Optimal Plans

Obtaining more precise information about minimizers is much harder. Key questions include:

* Is the optimal coupling unique? Smooth in some sense?
* Is there a *Monge coupling*, i.e. a deterministic optimal coupling?
* Is there a geometrical way to characterize optimal couplings? Can one check in practice that a certain coupling is optimal?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Monge vs. Kantorovich)</span></p>

Why not apply the same compactness reasoning (as in Theorem 4.1) to prove existence of a Monge minimizer? The problem is that the set of deterministic couplings is in general *not* compact; in fact, this set is often dense in the larger space of all couplings. So the *value* of the infimum in the Monge problem coincides with the value of the minimum in the Kantorovich problem, but there is no a priori reason to expect the existence of a Monge minimizer.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.9</span><span class="math-callout__name">(Non-existence of Monge Minimizer)</span></p>

Let $\mathcal{X} = \mathcal{Y} = \mathbb{R}^2$, $c(x,y) = \lvert x - y \rvert^2$, let $\mu$ be $\mathcal{H}^1$ restricted to $\lbrace 0 \rbrace \times [-1, 1]$, and let $\nu$ be $(1/2)\, \mathcal{H}^1$ restricted to $\lbrace -1, 1 \rbrace \times [-1, 1]$, where $\mathcal{H}^1$ is the one-dimensional Hausdorff measure. Then there is a unique optimal transport, which for each point $(0, a)$ sends one half of the mass to $(-1, a)$ and the other half to $(1, a)$. This is *not* a Monge transport (mass is split), but it can be approximated by (nonoptimal) deterministic transports.

</div>

## Chapter 5: Cyclical Monotonicity and Kantorovich Duality

This chapter introduces two basic concepts in the theory of optimal transport: a geometric property called **cyclical monotonicity**, and the **Kantorovich dual problem**, which is another face of the original Monge–Kantorovich problem. The main result is Theorem 5.10.

### Definitions and Heuristics

#### Cyclical Monotonicity

The concept is motivated by the bakery analogy from Chapter 3. If you have a transference plan that sends bread from bakeries $x\_i$ to cafés $y\_i$, and you try to improve it by rerouting mass along a cycle $(x\_1, y\_1), \ldots, (x\_N, y\_N)$ (redirecting the basket from $x\_i$ to $y\_{i+1}$ instead of $y\_i$), then the new plan is strictly better if and only if

$$c(x_1, y_2) + c(x_2, y_3) + \ldots + c(x_N, y_1) < c(x_1, y_1) + c(x_2, y_2) + \ldots + c(x_N, y_N).$$

If you can find such cycles, the plan is certainly not optimal. If you cannot, the plan cannot be improved by cyclic rerouting.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1</span><span class="math-callout__name">($c$-Cyclical Monotonicity)</span></p>

Let $\mathcal{X}, \mathcal{Y}$ be arbitrary sets, and $c : \mathcal{X} \times \mathcal{Y} \to (-\infty, +\infty]$ be a function. A subset $\Gamma \subset \mathcal{X} \times \mathcal{Y}$ is said to be **$c$-cyclically monotone** if, for any $N \in \mathbb{N}$, and any family $(x\_1, y\_1), \ldots, (x\_N, y\_N)$ of points in $\Gamma$, the inequality holds:

$$\sum_{i=1}^N c(x_i, y_i) \le \sum_{i=1}^N c(x_i, y_{i+1})$$

(with the convention $y\_{N+1} = y\_1$). A transference plan is said to be **$c$-cyclically monotone** if it is concentrated on a $c$-cyclically monotone set.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation)</span></p>

Informally, a $c$-cyclically monotone plan is one that *cannot be improved* by rerouting mass along any cycle. It is intuitively obvious that an optimal plan should be $c$-cyclically monotone; the converse is much less obvious (maybe the plan can be improved by radically changing it), but it holds true under mild conditions.

</div>

#### The Dual Kantorovich Problem

While the central notion in the original Monge–Kantorovich problem is *cost*, in the dual problem it is *price*. Imagine that a company offers to handle all transportation, buying bread at bakeries and selling it to cafés. Let $\psi(x)$ be the price at which bread is bought at bakery $x$, and $\phi(y)$ the price at which it is sold at café $y$. The consortium pays $\phi(y) - \psi(x)$ instead of $c(x,y)$.

To be competitive, prices must satisfy $\phi(y) - \psi(x) \le c(x,y)$ for all $(x, y)$. The company's problem is to **maximize profits**, leading to the **dual Kantorovich problem**:

$$\sup\left\lbrace \int_{\mathcal{Y}} \phi(y)\, d\nu(y) - \int_{\mathcal{X}} \psi(x)\, d\mu(x);\quad \phi(y) - \psi(x) \le c(x,y) \right\rbrace.$$

The weak duality inequality always holds:

$$\sup_{\phi - \psi \le c} \left\lbrace \int_{\mathcal{Y}} \phi\, d\nu - \int_{\mathcal{X}} \psi\, d\mu \right\rbrace \le \inf_{\pi \in \Pi(\mu,\nu)} \left\lbrace \int_{\mathcal{X} \times \mathcal{Y}} c(x,y)\, d\pi(x,y) \right\rbrace.$$

If equality holds and is achieved on both sides, then the pair $(\psi, \phi)$ is optimal in the dual, and $\pi$ is optimal in the primal.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Tight Prices)</span></p>

A pair of price functions $(\psi, \phi)$ is called **competitive** if $\phi(y) - \psi(x) \le c(x,y)$. It is called **tight** if

$$\phi(y) = \inf_x \bigl(\psi(x) + c(x,y)\bigr), \qquad \psi(x) = \sup_y \bigl(\phi(y) - c(x,y)\bigr).$$

One can always improve a competitive pair by tightening it. After one iteration the process is stationary, so it makes sense to restrict attention to tight pairs. From the tight pair equations, $\phi$ can be reconstructed from $\psi$, so $\psi$ becomes the only unknown — but it cannot be arbitrary: $\psi$ must be $c$-convex.

</div>

### $c$-Convexity and $c$-Transforms

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.2</span><span class="math-callout__name">($c$-Convexity)</span></p>

Let $\mathcal{X}, \mathcal{Y}$ be sets, and $c : \mathcal{X} \times \mathcal{Y} \to (-\infty, +\infty]$. A function $\psi : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ is said to be **$c$-convex** if it is not identically $+\infty$, and there exists $\zeta : \mathcal{Y} \to \mathbb{R} \cup \lbrace \pm\infty \rbrace$ such that

$$\forall x \in \mathcal{X} \qquad \psi(x) = \sup_{y \in \mathcal{Y}} \bigl(\zeta(y) - c(x,y)\bigr).$$

Then its **$c$-transform** is the function $\psi^c$ defined by

$$\forall y \in \mathcal{Y} \qquad \psi^c(y) = \inf_{x \in \mathcal{X}} \bigl(\psi(x) + c(x,y)\bigr),$$

and its **$c$-subdifferential** is the $c$-cyclically monotone set

$$\partial_c \psi := \left\lbrace (x,y) \in \mathcal{X} \times \mathcal{Y};\quad \psi^c(y) - \psi(x) = c(x,y) \right\rbrace.$$

The functions $\psi$ and $\psi^c$ are said to be **$c$-conjugate**. The $c$-subdifferential of $\psi$ at point $x$ is

$$\partial_c \psi(x) = \left\lbrace y \in \mathcal{Y};\quad (x,y) \in \partial_c \psi \right\rbrace,$$

or equivalently $y \in \partial\_c \psi(x)$ iff $\psi(x) + c(x,y) \le \psi(z) + c(z,y)$ for all $z \in \mathcal{X}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.7</span><span class="math-callout__name">($c$-Concavity)</span></p>

A function $\phi : \mathcal{Y} \to \mathbb{R} \cup \lbrace -\infty \rbrace$ is said to be **$c$-concave** if it is not identically $-\infty$, and there exists $\psi : \mathcal{X} \to \mathbb{R} \cup \lbrace \pm\infty \rbrace$ such that $\phi = \psi^c$. Then its $c$-transform is

$$\phi^c(x) = \sup_{y \in \mathcal{Y}} \bigl(\phi(y) - c(x,y)\bigr),$$

and its $c$-superdifferential is $\partial^c \phi := \lbrace (x,y) \subset \mathcal{X} \times \mathcal{Y};\; \phi(y) - \phi^c(x) = c(x,y) \rbrace$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Particular Case 5.3</span><span class="math-callout__name">($c(x,y) = -x \cdot y$)</span></p>

If $c(x,y) = -x \cdot y$ on $\mathbb{R}^n \times \mathbb{R}^n$, then the $c$-transform coincides with the usual **Legendre transform**, and $c$-convexity is just plain convexity (plus lower semicontinuity) on $\mathbb{R}^n$. One can think of $c(x,y) = -x \cdot y$ as basically the same as $c(x,y) = \lvert x - y \rvert^2 / 2$, since the "interaction" between positions $x$ and $y$ is the same for both costs.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Particular Case 5.4</span><span class="math-callout__name">($c = d$, a distance)</span></p>

If $c = d$ is a distance on some metric space $\mathcal{X}$, then a $c$-convex function is just a **1-Lipschitz** function, and it is its own $c$-transform. More generally, if $c$ satisfies the triangle inequality $c(x,z) \le c(x,y) + c(y,z)$, then $\psi$ is $c$-convex if and only if $\psi(y) - \psi(x) \le c(x,y)$ for all $x, y$; and then $\psi^c = \psi$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.8</span><span class="math-callout__name">(Alternative Characterization of $c$-Convexity)</span></p>

For any function $\psi : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$, let its **$c$-convexification** be defined by $\psi^{cc} = (\psi^c)^c$. More explicitly,

$$\psi^{cc}(x) = \sup_{y \in \mathcal{Y}} \inf_{\widetilde{x} \in \mathcal{X}} \bigl(\psi(\widetilde{x}) + c(\widetilde{x}, y) - c(x, y)\bigr).$$

Then $\psi$ is $c$-convex if and only if $\psi^{cc} = \psi$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.9</span><span class="math-callout__name">(Generalized Legendre Duality)</span></p>

Proposition 5.8 is a generalized version of the Legendre duality in convex analysis (to recover the usual Legendre duality, take $c(x,y) = -x \cdot y$ in $\mathbb{R}^n \times \mathbb{R}^n$). As a general fact, for any $\phi : \mathcal{Y} \to \mathbb{R} \cup \lbrace -\infty \rbrace$, one has $\phi^{ccc} = \phi^c$.

</div>

### Kantorovich Duality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.10</span><span class="math-callout__name">(Kantorovich Duality)</span></p>

Let $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$ be two Polish probability spaces and let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ be a lower semicontinuous cost function, such that $c(x,y) \ge a(x) + b(y)$ for some real-valued upper semicontinuous functions $a \in L^1(\mu)$ and $b \in L^1(\nu)$. Then

**(i)** There is duality:

$$\min_{\pi \in \Pi(\mu,\nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x,y)\, d\pi(x,y) = \sup_{\substack{(\psi,\phi) \in L^1(\mu) \times L^1(\nu) \\ \phi - \psi \le c}} \left(\int_{\mathcal{Y}} \phi\, d\nu - \int_{\mathcal{X}} \psi\, d\mu\right)$$

$$= \sup_{\psi \in L^1(\mu)} \left(\int_{\mathcal{Y}} \psi^c\, d\nu - \int_{\mathcal{X}} \psi\, d\mu\right),$$

and in the latter expressions one might as well impose that $\psi$ be $c$-convex and $\phi$ be $c$-concave.

**(ii)** If $c$ is real-valued and the optimal cost $C(\mu, \nu) = \inf\_{\pi \in \Pi(\mu,\nu)} \int c\, d\pi$ is finite, then there is a measurable $c$-cyclically monotone set $\Gamma \subset \mathcal{X} \times \mathcal{Y}$ such that for any $\pi \in \Pi(\mu, \nu)$ and any $c$-convex $\psi \in L^1(\mu)$, the following five statements are equivalent:

* (a) $\pi$ is optimal;
* (b) $\pi$ is $c$-cyclically monotone;
* (c) There is a $c$-convex $\psi$ such that, $\pi$-almost surely, $\psi^c(y) - \psi(x) = c(x,y)$;
* (d) There exist $\psi : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ and $\phi : \mathcal{Y} \to \mathbb{R} \cup \lbrace -\infty \rbrace$, such that $\phi(y) - \psi(x) \le c(x,y)$ for all $(x,y)$, with equality $\pi$-almost surely;
* (e) $\pi$ is concentrated on $\Gamma$.

If in addition $a$, $b$ and $c$ are continuous, then there is a closed $c$-cyclically monotone set $\Gamma$ such that: $\pi$ is optimal in the Kantorovich problem if and only if $\pi[\Gamma] = 1$; $\psi$ is optimal in the dual Kantorovich problem if and only if $\Gamma \subset \partial\_c \psi$.

**(iii)** If $c$ is real-valued, $C(\mu,\nu) < +\infty$, and $c(x,y) \le c\_\mathcal{X}(x) + c\_\mathcal{Y}(y)$ with $(c\_\mathcal{X}, c\_\mathcal{Y}) \in L^1(\mu) \times L^1(\nu)$, then both the primal and dual Kantorovich problems have solutions, and the sup in the duality becomes a max.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.11</span><span class="math-callout__name">(Continuous vs. Discontinuous Cost)</span></p>

When the cost $c$ is continuous, the support of $\pi$ is $c$-cyclically monotone. But for a discontinuous cost function, $\pi$ might be concentrated on a (nonclosed) $c$-cyclically monotone set while the support of $\pi$ is not $c$-cyclically monotone. So "concentrated on" and "supported in" are not exchangeable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.12</span><span class="math-callout__name">(The Set $\Gamma$)</span></p>

The set $\Gamma$ appearing in (ii)(e) is the same for all optimal $\pi$'s; it depends only on $\mu$ and $\nu$. This set is in general not unique. If $c$ is continuous and $\Gamma$ is imposed to be closed, then one can define a smallest $\Gamma$ (the closure of the union of all supports of optimal $\pi$'s) and a largest $\Gamma$ (the intersection of all $c$-subdifferentials $\partial\_c \psi$ where $\psi$ is such that some optimal $\pi$ is supported in $\partial\_c \psi$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.13</span><span class="math-callout__name">(Practical Optimality Test)</span></p>

A useful practical consequence of Theorem 5.10: Given a transference plan $\pi$, if you can cook up a pair of competitive prices $(\psi, \phi)$ such that $\phi(y) - \psi(x) = c(x,y)$ throughout the support of $\pi$, then you know that $\pi$ is optimal. This also shows that optimal transference plans satisfy a special condition: if you fix an optimal pair $(\psi, \phi)$, then mass arriving at $y$ can come from $x$ only if $c(x,y) = \phi(y) - \psi(x) = \psi^c(y) - \psi(x)$, meaning $x \in \mathrm{Arg\,min}\_{x' \in \mathcal{X}} \bigl(\psi(x') + c(x', y)\bigr)$.

In the bakery analogy: a café accepts bread from a bakery only if the combined cost of buying the bread there and transporting it is lowest among all bakeries.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.14</span><span class="math-callout__name">(Weakened Assumptions)</span></p>

The assumption $c \le c\_\mathcal{X} + c\_\mathcal{Y}$ in (iii) can be weakened to

$$\int_{\mathcal{X} \times \mathcal{Y}} c(x,y)\, d\mu(x)\, d\nu(y) < +\infty,$$

or even to the condition that $\mu\bigl[\lbrace x;\; \int\_\mathcal{Y} c(x,y)\, d\nu(y) < +\infty \rbrace\bigr] > 0$ and $\nu\bigl[\lbrace y;\; \int\_\mathcal{X} c(x,y)\, d\mu(x) < +\infty \rbrace\bigr] > 0$.

</div>

#### Particular Cases of Kantorovich Duality

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Kantorovich–Rubinstein Formula)</span></p>

When $c(x,y) = d(x,y)$ is a distance on a Polish space $\mathcal{X}$, and $\mu, \nu \in P\_1(\mathcal{X})$, then

$$\inf \mathbb{E}\, d(X, Y) = \sup \left\lbrace \int_\mathcal{X} \psi\, d\mu - \int_\mathcal{Y} \psi\, d\nu \right\rbrace,$$

where the infimum is over all couplings $(X, Y)$ of $(\mu, \nu)$, and the supremum is over all **1-Lipschitz** functions $\psi$. This is the **Kantorovich–Rubinstein formula**; it holds true as soon as the supremum in the left-hand side is finite.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Duality for $c(x,y) = -x \cdot y$)</span></p>

For $c(x,y) = -x \cdot y$ in $\mathbb{R}^n \times \mathbb{R}^n$, if $x \mapsto \lvert x \rvert^2 \in L^1(\mu)$ and $y \mapsto \lvert y \rvert^2 \in L^1(\nu)$, then

$$\sup \mathbb{E}(X \cdot Y) = \inf \left\lbrace \int_\mathcal{X} \varphi\, d\mu + \int_\mathcal{Y} \varphi^*\, d\nu \right\rbrace,$$

where the supremum is over all couplings $(X, Y)$ of $(\mu, \nu)$, the infimum is over all (lower semicontinuous) convex functions $\varphi$ on $\mathbb{R}^n$, and $\varphi^\ast$ is the usual Legendre transform. The problem is to **maximize the correlation** of the random variables $X$ and $Y$.

</div>

### Proof Structure of Theorem 5.10

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Strategy)</span></p>

The proof of Theorem 5.10 is divided into five main steps:

**Step 1** (Discrete case): If $\mu = (1/n)\sum \delta\_{x\_i}$, $\nu = (1/n)\sum \delta\_{y\_j}$, then the Monge–Kantorovich problem reduces to minimizing a linear function on the compact set $[0,1]^{n \times n}$ of bistochastic arrays. A minimizer exists and its support is $c$-cyclically monotone (otherwise a cyclic perturbation would lower the cost).

**Step 2** (Continuous cost): Approximate $\mu, \nu$ by empirical measures $\mu\_n, \nu\_n$ via the law of large numbers. For each $n$, a cyclically monotone $\pi\_n$ exists (Step 1). By tightness and Prokhorov's theorem, a subsequence converges weakly to some $\pi \in \Pi(\mu, \nu)$, and $\pi$ is $c$-cyclically monotone.

**Step 3** (Construction of $\psi$): Given a $c$-cyclically monotone transference plan $\pi$ with support $\Gamma$, pick $(x\_0, y\_0) \in \Gamma$ and define

$$\psi(x) := \sup_{m \in \mathbb{N}} \sup \left\lbrace [c(x_0, y_0) - c(x_1, y_0)] + \ldots + [c(x_m, y_m) - c(x, y_m)];\; (x_i, y_i) \in \Gamma \right\rbrace.$$

Then $\psi$ is $c$-convex (with $\psi(x\_0) = 0$), and $\partial\_c \psi$ contains $\Gamma$.

**Step 4** (Bounded continuous cost): Using Steps 2 and 3, construct $\pi$ and $(\psi, \phi = \psi^c)$ with $\phi(y) - \psi(x) = c(x,y)$ on $\mathrm{Spt}\, \pi$. Since $\psi, \phi$ are bounded, integrate via the marginal condition to get duality.

**Step 5** (Lower semicontinuous cost): Approximate $c$ from below by bounded, uniformly continuous $c\_k$. Apply Step 4 to each $c\_k$, then pass to the limit using monotone convergence.

The proof of Part (ii) establishes the chain (a) $\Rightarrow$ (b) $\Rightarrow$ (c) $\Rightarrow$ (d) $\Rightarrow$ (a) $\Rightarrow$ (e) $\Rightarrow$ (b).

</div>

### Restriction of the Duality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.18</span><span class="math-callout__name">(Restriction of $c$-Convexity)</span></p>

Let $\mathcal{X}, \mathcal{Y}$ be two sets and $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R} \cup \lbrace +\infty \rbrace$. Let $\mathcal{X}' \subset \mathcal{X}$, $\mathcal{Y}' \subset \mathcal{Y}$ and let $c'$ be the restriction of $c$ to $\mathcal{X}' \times \mathcal{Y}'$. Let $\psi : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ be a $c$-convex function. Then there is a $c'$-convex function $\psi' : \mathcal{X}' \to \mathbb{R} \cup \lbrace +\infty \rbrace$ such that $\psi' \le \psi$ on $\mathcal{X}'$, $\psi'$ coincides with $\psi$ on $\mathrm{proj}\_\mathcal{X}((\partial\_c \psi) \cap (\mathcal{X}' \times \mathcal{Y}'))$, and $\partial\_c \psi \cap (\mathcal{X}' \times \mathcal{Y}') \subset \partial\_{c'} \psi'$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.19</span><span class="math-callout__name">(Restriction for the Kantorovich Duality)</span></p>

Let $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$ be two Polish probability spaces, let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be a lower semicontinuous cost function with $c(x,y) \ge a(x) + b(y)$. Assume the optimal transport cost $C(\mu, \nu)$ is finite. Let $\pi$ be an optimal transference plan, and let $\psi$ be a $c$-convex function such that $\pi$ is concentrated on $\partial\_c \psi$. Let $\widetilde{\pi} \le \pi$ with $\widetilde{\pi}[\mathcal{X} \times \mathcal{Y}] > 0$; let $\pi' := \widetilde{\pi}/\widetilde{\pi}[\mathcal{X} \times \mathcal{Y}]$. Let $\mathcal{X}' \supset \mathrm{Spt}\, \mu'$ and $\mathcal{Y}' \supset \mathrm{Spt}\, \nu'$ be closed sets, and let $c'$ be the restriction of $c$ to $\mathcal{X}' \times \mathcal{Y}'$. Then there is a $c'$-convex function $\psi'$ such that:

* (a) $\psi'$ coincides with $\psi$ on $\mathrm{proj}\_\mathcal{X}((\partial\_c \psi) \cap (\mathcal{X}' \times \mathcal{Y}'))$, which has full $\mu'$-measure;
* (b) $\pi'$ is concentrated on $\partial\_{c'} \psi'$;
* (c) $\psi'$ solves the dual Kantorovich problem between $(\mathcal{X}', \mu')$ and $(\mathcal{Y}', \nu')$ with cost $c'$.

In short: "it is always possible to restrict the cost function."

</div>

### Application: Stability

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.20</span><span class="math-callout__name">(Stability of Optimal Transport)</span></p>

Let $\mathcal{X}$ and $\mathcal{Y}$ be Polish spaces, and let $c \in C(\mathcal{X} \times \mathcal{Y})$ be a real-valued continuous cost function, $\inf c > -\infty$. Let $(c\_k)\_{k \in \mathbb{N}}$ be a sequence of continuous cost functions converging uniformly to $c$. Let $(\mu\_k)$ and $(\nu\_k)$ be sequences of probability measures converging weakly to $\mu$ and $\nu$ respectively. For each $k$, let $\pi\_k$ be an optimal transference plan between $\mu\_k$ and $\nu\_k$. If $\int c\_k\, d\pi\_k < +\infty$ for all $k$, then, up to extraction of a subsequence, $\pi\_k$ converges weakly to some $c$-cyclically monotone $\pi \in \Pi(\mu, \nu)$.

If moreover $\liminf\_{k} \int c\_k\, d\pi\_k < +\infty$, then $C(\mu, \nu)$ is finite and $\pi$ is an optimal transference plan.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.21</span><span class="math-callout__name">(Compactness of the Set of Optimal Plans)</span></p>

Let $\mathcal{X}$ and $\mathcal{Y}$ be Polish spaces, and let $c(x,y)$ be a real-valued continuous cost function, $\inf c > -\infty$. Let $\mathcal{K}$ and $\mathcal{L}$ be two compact subsets of $P(\mathcal{X})$ and $P(\mathcal{Y})$ respectively. Then the set of optimal transference plans $\pi$ whose marginals belong to $\mathcal{K}$ and $\mathcal{L}$ is itself compact in $P(\mathcal{X} \times \mathcal{Y})$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.22</span><span class="math-callout__name">(Measurable Selection of Optimal Plans)</span></p>

Let $\mathcal{X}$, $\mathcal{Y}$ be Polish spaces and $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ a continuous cost function, $\inf c > -\infty$. Let $\Omega$ be a measurable space and $\omega \longmapsto (\mu\_\omega, \nu\_\omega)$ a measurable function $\Omega \to P(\mathcal{X}) \times P(\mathcal{Y})$. Then there is a measurable choice $\omega \longmapsto \pi\_\omega$ such that for each $\omega$, $\pi\_\omega$ is an optimal transference plan between $\mu\_\omega$ and $\nu\_\omega$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.23</span><span class="math-callout__name">(Stability of the Transport Map)</span></p>

Let $\mathcal{X}$ be a locally compact Polish space, $(\mathcal{Y}, d)$ another Polish space. Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be lower semicontinuous with $\inf c > -\infty$, and let $c\_k \to c$ uniformly. Let $\mu \in P(\mathcal{X})$ and $\nu\_k \to \nu$ weakly. Assume that each $\pi\_k := (\mathrm{Id}, T\_k)\_\# \mu$ is an optimal transference plan for cost $c\_k$, having finite total transport cost. Further, assume that $\pi := (\mathrm{Id}, T)\_\# \mu$ is the *unique* optimal transference plan for cost $c$, with finite cost. Then $T\_k$ converges to $T$ **in probability**:

$$\forall \varepsilon > 0 \qquad \mu\bigl[\lbrace x \in \mathcal{X};\; d(T_k(x), T(x)) \ge \varepsilon \rbrace\bigr] \xrightarrow{k \to \infty} 0.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.24</span></p>

An important assumption in Corollary 5.23 is the **uniqueness** of the optimal transport map $T$.

</div>

### Application: Dual Formulation of Transport Inequalities

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.26</span><span class="math-callout__name">(Dual Transport Inequalities)</span></p>

Let $\mathcal{X}$, $\mathcal{Y}$ be two Polish spaces, $\nu$ a given probability measure on $\mathcal{Y}$. Let $F : P(\mathcal{X}) \to \mathbb{R} \cup \lbrace +\infty \rbrace$ be a convex lower semicontinuous function, and $\Lambda$ its Legendre transform on $C\_b(\mathcal{X})$:

$$F(\mu) = \sup_{\varphi \in C_b(\mathcal{X})} \left(\int_\mathcal{X} \varphi\, d\mu - \Lambda(\varphi)\right), \qquad \Lambda(\varphi) = \sup_{\mu \in P(\mathcal{X})} \left(\int_\mathcal{X} \varphi\, d\mu - F(\mu)\right).$$

Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ be a lower semicontinuous cost function, $\inf c > -\infty$. Then the following two statements are equivalent:

* (i) $\forall \mu \in P(\mathcal{X}), \qquad C(\mu, \nu) \le F(\mu)$;
* (ii) $\forall \phi \in C\_b(\mathcal{Y}), \qquad \Lambda\!\left(\int\_\mathcal{Y} \phi\, d\nu - \phi^c\right) \le 0$, where $\phi^c(x) := \sup\_y [\phi(y) - c(x,y)]$.

Moreover, if $\Phi : \mathbb{R} \to \mathbb{R}$ is a nondecreasing function with $\Phi(0) = 0$, then:

* (i') $\forall \mu \in P(\mathcal{X}), \qquad \Phi(C(\mu, \nu)) \le F(\mu)$;
* (ii') $\forall \phi \in C\_b(\mathcal{Y}),\; \forall t \ge 0, \qquad \Lambda\!\left(t \int\_\mathcal{Y} \phi\, d\nu - t\, \phi^c - \Phi^\ast(t)\right) \le 0$,

where $\Phi^\ast$ is the Legendre transform of $\Phi$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.29</span><span class="math-callout__name">(Kullback–Leibler Information)</span></p>

The most famous example occurs when $\mathcal{X} = \mathcal{Y}$ and $F(\mu) = H\_\nu(\mu) = \int \rho \log \rho\, d\nu$ is the Kullback–Leibler information (relative entropy) of $\mu$ with respect to $\nu$, where $\rho = d\mu/d\nu$ (and $F(\mu) = +\infty$ if $\mu$ is not absolutely continuous with respect to $\nu$). Then

$$\Lambda(\varphi) = \log\!\left(\int e^\varphi\, d\nu\right).$$

So the two functional inequalities

$$\forall \mu \in P(\mathcal{X}), \quad C(\mu, \nu) \le H_\nu(\mu) \qquad \text{and} \qquad \forall \phi \in C_b(\mathcal{Y}), \quad e^{\int \phi\, d\nu} \le \int e^{\phi^c}\, d\nu$$

are equivalent.

</div>

### Application: Solvability of the Monge Problem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.30</span><span class="math-callout__name">(Criterion for Solvability of the Monge Problem)</span></p>

Let $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$ be two Polish probability spaces, and let $a \in L^1(\mu)$, $b \in L^1(\nu)$ be two real-valued upper semicontinuous functions. Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be a lower semicontinuous cost function such that $c(x,y) \ge a(x) + b(y)$ for all $x, y$. Let $C(\mu, \nu)$ be the optimal total transport cost between $\mu$ and $\nu$. If

* (i) $C(\mu, \nu) < +\infty$;
* (ii) for any $c$-convex function $\psi : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$, the set of $x \in \mathcal{X}$ such that $\partial\_c \psi(x)$ contains more than one element is $\mu$-negligible;

then there is a unique (in law) optimal coupling $(X, Y)$ of $(\mu, \nu)$, and it is **deterministic**. It is characterized by the existence of a $c$-convex function $\psi$ such that, almost surely, $Y$ belongs to $\partial\_c \psi(X)$. In particular, the Monge problem with initial measure $\mu$ and final measure $\nu$ admits a unique solution.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 5.30)</span></p>

The proof is almost immediate from Theorem 5.10(ii). There is a $c$-convex $\psi$ and a measurable $\Gamma \subset \partial\_c \psi$ such that any optimal plan $\pi$ is concentrated on $\Gamma$. By assumption (ii), outside a $\mu$-negligible set $Z$, $\partial\_c \psi(x)$ contains exactly one element. So for $x \in \mathrm{proj}\_\mathcal{X}(\Gamma) \setminus Z$, there is exactly one $y$ with $(x, y) \in \Gamma$, defining $T(x) = y$. Any optimal coupling must then be concentrated on the graph of $T$, i.e. $\pi = (\mathrm{Id}, T)\_\# \mu$ is the unique Monge transport.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.25</span><span class="math-callout__name">(Stability Counterexample)</span></p>

If the measure $\mu$ in Corollary 5.23 is replaced by a sequence $(\mu\_k)$ converging weakly to $\mu$, then the maps $T\_k$ and $T$ may be far away from each other, even $\mu\_k$-almost surely. Example: $\mathcal{X} = \mathcal{Y} = \mathbb{R}$, $\mu\_k = \delta\_{1/k}$, $\mu = \delta\_0$, $\nu\_k = \nu = \delta\_0$, $T\_k(x) = 0$, $T(x) = 1\_{x \ne 0}$.

</div>

### Bibliographical Notes on Chapter 5

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to Weak KAM Theory and Mather/Aubry Sets)</span></p>

A common convention in the literature takes the pair $(-\psi, \phi)$ as the unknown, which makes some formulas more symmetric: the $c$-transform becomes $\varphi^c(y) = \inf\_x [c(x,y) - \varphi(x)]$ and then $\psi^c(x) = \inf\_y [c(x,y) - \psi(y)]$, the same formula going back and forth between functions of $x$ and functions of $y$.

In weak KAM theory, $\mathcal{X} = \mathcal{Y}$ is a Riemannian manifold $M$; a Lagrangian cost function is given on the tangent bundle $TM$; and $c = c(x,y)$ is the minimum action to go from $x$ to $y$. Since in general $c(x,x) \ne 0$, it is meaningful to consider the optimal transport cost $C(\mu, \mu)$ of a measure $\mu$ to itself. If $M$ is compact, there exists a $\overline{\mu}$ minimizing $C(\mu, \mu)$. Theorem 5.10 associates to the optimal transport problem between $\overline{\mu}$ and itself two distinguished closed $c$-cyclically monotone sets $\Gamma\_{\min} \subset \Gamma\_{\max} \subset M \times M$. These sets can be identified with subsets of $TM$ via the embedding (initial position, final position) $\longmapsto$ (initial position, initial velocity). Under this identification, $\Gamma\_{\min}$ and $\Gamma\_{\max}$ are called the **Mather** and **Aubry** sets respectively; they carry valuable information about the underlying dynamics.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History of Kantorovich Duality)</span></p>

The Kantorovich duality theorem was proven by Kantorovich himself on a compact space in his famous 1942 note (even before he realized the connection with Monge's problem). Later, Kantorovich noted that for $c(x,y) = \lvert x - y \rvert$ in $\mathbb{R}^n$, the duality implies that transport pathlines are orthogonal to the surfaces $\lbrace \psi = \mathrm{constant} \rbrace$, where $\psi$ is the Kantorovich potential — thus recovering Monge's celebrated original observation.

In 1958, Kantorovich and Rubinstein made the duality more explicit for $c(x,y) = d(x,y)$. The statement was later generalized by Dudley, with alternative arguments by Neveu (useful for nonseparable spaces). Many contributors including Rüschendorf, Fernique, Szulga, Kellerer, and Feyel contributed to the problem. Modern treatments most often use variants of the Hahn–Banach theorem.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalence of Optimality and Cyclical Monotonicity)</span></p>

The equivalence between optimality (of a transference plan) and cyclical monotonicity, for quite general cost functions and probability measures, was a widely open problem until recently. The current state of the art:

* The equivalence is **false** for a general lower semicontinuous cost function with possibly infinite values (counterexample by Ambrosio and Pratelli);
* The equivalence is **true** for a *continuous* cost function with possibly infinite values (Pratelli);
* The equivalence is **true** for a *real-valued* lower semicontinuous cost function (Schachermayer and Teichmann); actually it suffices for the cost to be lower semicontinuous and real-valued $(\mu \otimes \nu)$-almost everywhere;
* More generally, the equivalence is true as soon as $c$ is measurable and $\lbrace c = \infty \rbrace$ is the union of a closed set and a $(\mu \otimes \nu)$-negligible Borel set (Beiglböck, Goldstern, Maresch and Schachermayer).

Schachermayer and Teichmann suggested that the correct notion is not cyclical monotonicity but a variant called "**strong cyclical monotonicity condition**". A striking theorem of Beiglböck et al. shows that **robust optimality** is always equivalent to strong $c$-monotonicity.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Infinite-Valued and $L^\infty$ Cost Functions)</span></p>

In most applications, the cost function is continuous and often rather simple. However, cost functions that achieve the value $+\infty$ are sometimes useful, as in the "secondary variational problem" of Ambrosio and Pratelli, and in the optimal transport in Wiener space (Feyel and Üstünel), where $c(x,y)$ is the square norm of $x - y$ in the Cameron–Martin space.

If one uses the cost function $\lvert x - y \rvert^p$ in $\mathbb{R}^n$ and lets $p \to \infty$, the $c$-cyclical monotonicity condition becomes $\sup \lvert x\_i - y\_i \rvert \le \sup \lvert x\_i - y\_{i+1} \rvert$, giving a different flavor to the analysis.

</div>

## Chapter 6: The Wasserstein Distances

From the Monge–Kantorovich problem one can derive natural distance functions on spaces of probability measures, by choosing the cost function as a power of the distance. This chapter establishes the main properties of these distances.

### Definition and Basic Properties

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.1</span><span class="math-callout__name">(Wasserstein Distances)</span></p>

Let $(\mathcal{X}, d)$ be a Polish metric space, and let $p \in [1, \infty)$. For any two probability measures $\mu, \nu$ on $\mathcal{X}$, the **Wasserstein distance of order $p$** between $\mu$ and $\nu$ is defined by

$$W_p(\mu, \nu) = \left(\inf_{\pi \in \Pi(\mu,\nu)} \int_\mathcal{X} d(x,y)^p\, d\pi(x,y)\right)^{1/p} = \inf\left\lbrace \bigl[\mathbb{E}\, d(X,Y)^p\bigr]^{1/p};\; \mathrm{law}(X) = \mu,\; \mathrm{law}(Y) = \nu \right\rbrace.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Particular Case 6.2</span><span class="math-callout__name">(Kantorovich–Rubinstein Distance)</span></p>

The distance $W\_1$ is also commonly called the **Kantorovich–Rubinstein distance** (although it would be more proper to reserve the terminology Kantorovich–Rubinstein for the *norm* which extends $W\_1$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3</span></p>

$W\_p(\delta\_x, \delta\_y) = d(x,y)$. In this example, the distance does not depend on $p$; but this is not the rule.

</div>

At the present level of generality, $W\_p$ is still not a distance in the strict sense, because it might take the value $+\infty$; but otherwise it does satisfy the axioms of a distance.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof that $W\_p$ Satisfies Distance Axioms)</span></p>

* **Symmetry:** $W\_p(\mu, \nu) = W\_p(\nu, \mu)$ is clear.
* **Triangle inequality:** Let $\mu\_1, \mu\_2, \mu\_3$ be three probability measures on $\mathcal{X}$. Let $(X\_1, X\_2)$ be an optimal coupling of $(\mu\_1, \mu\_2)$ and $(Z\_2, Z\_3)$ an optimal coupling of $(\mu\_2, \mu\_3)$ for the cost $c = d^p$. By the Gluing Lemma, there exist $(X\_1', X\_2', X\_3')$ with $\mathrm{law}(X\_1', X\_2') = \mathrm{law}(X\_1, X\_2)$ and $\mathrm{law}(X\_2', X\_3') = \mathrm{law}(Z\_2, Z\_3)$. Then by Minkowski's inequality in $L^p(\mathbb{P})$:

$$W_p(\mu_1, \mu_3) \le \bigl(\mathbb{E}\, d(X_1', X_3')^p\bigr)^{1/p} \le \bigl(\mathbb{E}\, d(X_1', X_2')^p\bigr)^{1/p} + \bigl(\mathbb{E}\, d(X_2', X_3')^p\bigr)^{1/p} = W_p(\mu_1, \mu_2) + W_p(\mu_2, \mu_3).$$

* **Separation:** If $W\_p(\mu, \nu) = 0$, then the optimal plan is concentrated on the diagonal $\lbrace y = x \rbrace$, so $\nu = \mathrm{Id}\_\# \mu = \mu$.

</div>

### Wasserstein Space

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.4</span><span class="math-callout__name">(Wasserstein Space)</span></p>

With the same conventions as in Definition 6.1, the **Wasserstein space of order $p$** is defined as

$$P_p(\mathcal{X}) := \left\lbrace \mu \in P(\mathcal{X});\quad \int_\mathcal{X} d(x_0, x)^p\, \mu(dx) < +\infty \right\rbrace,$$

where $x\_0 \in \mathcal{X}$ is arbitrary. This space does not depend on the choice of $x\_0$. Then $W\_p$ defines a (finite) distance on $P\_p(\mathcal{X})$.

</div>

In words, the Wasserstein space is the space of probability measures with a **finite moment of order $p$**. That $W\_p$ is finite on $P\_p$ follows from the inequality $d(x,y)^p \le 2^{p-1}[d(x,x\_0)^p + d(x\_0,y)^p]$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.5</span><span class="math-callout__name">(Duality Formula for $W\_1$)</span></p>

Theorem 5.10(i) and Particular Case 5.4 together lead to the useful **duality formula for the Kantorovich–Rubinstein distance**: For any $\mu, \nu$ in $P\_1(\mathcal{X})$,

$$W_1(\mu, \nu) = \sup_{\lVert \psi \rVert_{\mathrm{Lip}} \le 1} \left\lbrace \int_\mathcal{X} \psi\, d\mu - \int_\mathcal{X} \psi\, d\nu \right\rbrace.$$

Among many applications: if $f$ is a probability density with respect to $\mu$ then

$$\left(\int f\, d\mu\right)\left(\int g\, d\mu\right) - \int (fg)\, d\mu \le \lVert g \rVert_{\mathrm{Lip}}\, W_1(f\, \mu, \mu).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.6</span><span class="math-callout__name">(Ordering of Wasserstein Distances)</span></p>

A simple application of Hölder's inequality shows that

$$p \le q \implies W_p \le W_q.$$

In particular, $W\_1$ is the weakest of all. The most useful exponents are $p = 1$ and $p = 2$. As a general rule, $W\_1$ is more flexible and easier to bound, while $W\_2$ better reflects geometric features (at least for problems with a Riemannian flavor), is better adapted when there is more structure, and scales better with the dimension. Results in $W\_2$ distance are usually stronger and more difficult to establish than results in $W\_1$ distance.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.7</span><span class="math-callout__name">(Reverse Inequalities)</span></p>

Under adequate regularity assumptions on the cost function and the probability measures, it is possible to control $W\_p$ in terms of $W\_q$ even for $q < p$; these reverse inequalities express a certain rigidity property of optimal transport maps that comes from $c$-cyclical monotonicity.

</div>

### Convergence in Wasserstein Sense

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.8</span><span class="math-callout__name">(Weak Convergence in $P\_p$)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, $p \in [1, \infty)$. Let $(\mu\_k)\_{k \in \mathbb{N}}$ be a sequence of measures in $P\_p(\mathcal{X})$ and let $\mu$ be another element of $P\_p(\mathcal{X})$. Then $(\mu\_k)$ is said to **converge weakly in $P\_p(\mathcal{X})$** if any one of the following equivalent properties is satisfied (for some, and then any, $x\_0 \in \mathcal{X}$):

* (i) $\mu\_k \longrightarrow \mu$ weakly and $\int d(x\_0, x)^p\, d\mu\_k(x) \longrightarrow \int d(x\_0, x)^p\, d\mu(x)$;
* (ii) $\mu\_k \longrightarrow \mu$ weakly and $\limsup\_{k \to \infty} \int d(x\_0, x)^p\, d\mu\_k(x) \le \int d(x\_0, x)^p\, d\mu(x)$;
* (iii) $\mu\_k \longrightarrow \mu$ weakly and $\lim\_{R \to \infty} \limsup\_{k \to \infty} \int\_{d(x\_0, x) \ge R} d(x\_0, x)^p\, d\mu\_k(x) = 0$;
* (iv) For all continuous functions $\varphi$ with $\lvert \varphi(x) \rvert \le C\bigl(1 + d(x\_0, x)^p\bigr)$, one has $\int \varphi\, d\mu\_k \longrightarrow \int \varphi\, d\mu$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.9</span><span class="math-callout__name">($W\_p$ Metrizes $P\_p$)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, and $p \in [1, \infty)$; then the Wasserstein distance $W\_p$ **metrizes the weak convergence** in $P\_p(\mathcal{X})$. In other words, if $(\mu\_k)$ is a sequence of measures in $P\_p(\mathcal{X})$ and $\mu$ is another measure in $P(\mathcal{X})$, then the statements

$$\mu_k \text{ converges weakly in } P_p(\mathcal{X}) \text{ to } \mu \qquad \text{and} \qquad W_p(\mu_k, \mu) \longrightarrow 0$$

are equivalent.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.10</span></p>

As a consequence of Theorem 6.9, convergence in the $p$-Wasserstein space implies convergence of the moments of order $p$. There is a stronger statement: the map $\mu \longmapsto \bigl(\int d(x\_0, x)^p\, \mu(dx)\bigr)^{1/p}$ is 1-Lipschitz with respect to $W\_p$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 6.11</span><span class="math-callout__name">(Continuity of $W\_p$)</span></p>

If $(\mathcal{X}, d)$ is a Polish space, and $p \in [1, \infty)$, then $W\_p$ is continuous on $P\_p(\mathcal{X})$. More explicitly, if $\mu\_k$ (resp. $\nu\_k$) converges to $\mu$ (resp. $\nu$) weakly in $P\_p(\mathcal{X})$ as $k \to \infty$, then

$$W_p(\mu_k, \nu_k) \longrightarrow W_p(\mu, \nu).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.12</span><span class="math-callout__name">(Lower Semicontinuity)</span></p>

If the convergences are only usual weak convergences (not in $P\_p$), then one can only conclude that $W\_p(\mu, \nu) \le \liminf W\_p(\mu\_k, \nu\_k)$: the Wasserstein distance is **lower semicontinuous** on $P(\mathcal{X})$, just like the optimal transport cost for any lower semicontinuous nonnegative cost function $c$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 6.13</span><span class="math-callout__name">(Metrizability of the Weak Topology)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space. If $\widehat{d}$ is a bounded distance inducing the same topology as $d$ (such as $\widehat{d} = d/(1+d)$), then the convergence in Wasserstein sense for the distance $\widehat{d}$ is equivalent to the usual **weak convergence** of probability measures in $P(\mathcal{X})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Other Probability Metrics)</span></p>

The fact that Wasserstein distances metrize weak convergence sounds good, but there are many ways to metrize weak convergence. Here are some of the most popular alternatives:

* **Lévy–Prokhorov distance:** $d\_P(\mu, \nu) = \inf\lbrace \varepsilon > 0;\; \exists X, Y,\; \inf \mathbb{P}[d(X,Y) > \varepsilon] \le \varepsilon \rbrace$;
* **Bounded Lipschitz distance** (Fortet–Mourier): $d\_{bL}(\mu, \nu) = \sup\lbrace \int \varphi\, d\mu - \int \varphi\, d\nu;\; \lVert \varphi \rVert\_\infty + \lVert \varphi \rVert\_{\mathrm{Lip}} \le 1 \rbrace$;
* **Weak-$\ast$ distance** (on a locally compact metric space): $d\_{w\ast}(\mu, \nu) = \sum\_{k \in \mathbb{N}} 2^{-k} \bigl\lvert \int \varphi\_k\, d\mu - \int \varphi\_k\, d\nu \bigr\rvert$, where $(\varphi\_k)$ is dense in $C\_0(\mathcal{X})$;
* **Toscani distance** (on $P\_2(\mathbb{R}^n)$): $d\_T(\mu, \nu) = \sup\_{\xi \in \mathbb{R}^n \setminus \lbrace 0 \rbrace} \frac{\lvert \int e^{-ix \cdot \xi}\, d\mu(x) - \int e^{-ix \cdot \xi}\, d\nu(x) \rvert}{\lvert \xi \rvert^2}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Wasserstein Distances?)</span></p>

Several reasons to prefer Wasserstein distances:

1. They are **rather strong**, especially in the way they handle large distances in $\mathcal{X}$; it is not difficult to combine convergence in Wasserstein distance with some smoothness bound to get convergence in stronger distances.
2. Their definition via optimal transport makes them convenient for problems where optimal transport is naturally involved, such as many PDE problems.
3. They have a **rich duality** (especially for $p = 1$, via the Kantorovich–Rubinstein formula).
4. Being defined by an infimum, they are **easy to bound from above**: any coupling between $\mu$ and $\nu$ yields an upper bound.
5. They **incorporate geometry**: the mapping $x \longmapsto \delta\_x$ is an *isometric embedding* of $\mathcal{X}$ into $P\_p(\mathcal{X})$, and there are much deeper links.

</div>

### Proof of Theorem 6.9

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6.14</span><span class="math-callout__name">(Cauchy Sequences in $W\_p$ are Tight)</span></p>

Let $\mathcal{X}$ be a Polish space, let $p \ge 1$ and let $(\mu\_k)\_{k \in \mathbb{N}}$ be a Cauchy sequence in $(P\_p(\mathcal{X}), W\_p)$. Then $(\mu\_k)$ is **tight**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 6.9)</span></p>

*($W\_p \to 0$ implies weak convergence in $P\_p$):* By Lemma 6.14, $(\mu\_k)$ is tight, so by Prokhorov it has a weakly convergent subsequence. By lower semicontinuity (Lemma 4.3), the limit must be $\mu$, and the whole sequence converges. The moment convergence follows from the triangle inequality in $L^p$.

*(Weak convergence in $P\_p$ implies $W\_p \to 0$):* Let $\pi\_k$ be an optimal plan between $\mu\_k$ and $\mu$. By tightness (Lemma 4.4), extract a subsequence $\pi\_k \to \pi$ weakly. By Theorem 5.20, $\pi$ is an optimal coupling of $\mu$ and $\mu$, hence the trivial coupling $(\mathrm{Id}, \mathrm{Id})\_\# \mu$. Then a truncation argument using the decomposition $d(x,y)^p = [d(x,y) \wedge R]^p + [d(x,y)^p - R^p]\_+$ shows that $W\_p(\mu\_k, \mu)^p \to 0$.

</div>

### Control by Total Variation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Total Variation as Optimal Transport)</span></p>

The total variation has a classical probabilistic representation:

$$\lVert \mu - \nu \rVert_{TV} = 2 \inf \mathbb{P}[X \ne Y],$$

where the infimum is over all couplings $(X, Y)$ of $(\mu, \nu)$. This is a very particular case of Kantorovich duality for the cost function $1\_{x \ne y}$. A control in Wasserstein distance should be weaker than a control in total variation (since total variation does not account for large distances), but one can control $W\_p$ by **weighted** total variation.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.15</span><span class="math-callout__name">(Wasserstein Distance Controlled by Weighted Total Variation)</span></p>

Let $\mu$ and $\nu$ be two probability measures on a Polish space $(\mathcal{X}, d)$. Let $p \in [1, \infty)$ and $x\_0 \in \mathcal{X}$. Then

$$W_p(\mu, \nu) \le 2^{1/p} \left(\int d(x_0, x)^p\, d\lvert \mu - \nu \rvert(x)\right)^{1/p'}, \qquad \frac{1}{p} + \frac{1}{p'} = 1.$$

In particular, for $p = 1$, if the diameter of $\mathcal{X}$ is bounded by $D$, this gives $W\_1(\mu, \nu) \le D\, \lVert \mu - \nu \rVert\_{TV}$.

</div>

### Topological Properties of the Wasserstein Space

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.18</span><span class="math-callout__name">(Topology of the Wasserstein Space)</span></p>

Let $\mathcal{X}$ be a complete separable metric space and $p \in [1, \infty)$. Then the Wasserstein space $P\_p(\mathcal{X})$, metrized by $W\_p$, is also a **complete separable metric space**. In short: *the Wasserstein space over a Polish space is itself a Polish space*. Moreover, any probability measure can be approximated by a sequence of probability measures with **finite support**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.19</span><span class="math-callout__name">(Compactness)</span></p>

If $\mathcal{X}$ is compact, then $P\_p(\mathcal{X})$ is also compact; but if $\mathcal{X}$ is only locally compact, then $P\_p(\mathcal{X})$ is *not* locally compact.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 6.18)</span></p>

**Separability:** Let $\mathcal{D}$ be a dense sequence in $\mathcal{X}$, and let $\mathcal{P}$ be the space of probability measures $\sum a\_j \delta\_{x\_j}$ with rational coefficients $a\_j$ and finitely many elements $x\_j \in \mathcal{D}$. Then $\mathcal{P}$ is dense in $P\_p(\mathcal{X})$. The key step is to show that any $\mu \in P\_p(\mathcal{X})$ can be approximated by a finite combination of Dirac masses: cover a compact set $K$ (carrying most of the mass) by finitely many balls, map each ball to a single point, and use Theorem 6.15 to show the error is small.

**Completeness:** Let $(\mu\_k)$ be Cauchy in $(P\_p(\mathcal{X}), W\_p)$. By Lemma 6.14, it is tight, so by Prokhorov's theorem it admits a subsequence converging weakly to some $\mu$. Then $\mu \in P\_p(\mathcal{X})$ and $W\_p(\mu, \mu\_{\ell'}) \to 0$ by lower semicontinuity. Since a Cauchy sequence with a convergent subsequence converges, the whole sequence converges.

</div>

### Bibliographical Notes on Chapter 6

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Terminology and History)</span></p>

The terminology "Wasserstein distance" (apparently introduced by Dobrushin) is debatable, since these distances were discovered and rediscovered by several authors throughout the twentieth century, including (chronologically) Gini, Kantorovich, Wasserstein, Mallows, and Tanaka (among others: Salvemini, Dall'Aglio, Hoeffding, Fréchet, Rubinstein, Ornstein). Rüschendorf advocates the denomination "minimal $L^p$-metric"; Vershik stands in favor of "Kantorovich distance". Nonetheless, the terminology "Wasserstein distance" has become extremely successful and is now standard.

In image processing, the $W\_1$ distance is also known as the **"Earth Mover's distance"**.

The Kantorovich–Rubinstein norm provides an explicit isometric embedding of any Polish space in a Banach space. As pointed out by Vershik, it can be intrinsically characterized as the maximal norm $\lVert \cdot \rVert$ on $M(\mathcal{X})$ which is "compatible" with the distance: $\lVert \delta\_x - \delta\_y \rVert = d(x,y)$ for all $x, y \in \mathcal{X}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Applications of Wasserstein Distances)</span></p>

Applications of Wasserstein distances are extremely diverse:

* Comparing color distributions in images
* Statistics, limit theorems, and approximation of probability measures
* Rates of fluctuations of empirical measures: the average $W\_1$ distance between two independent copies of the empirical measure behaves like $(\int \rho^{1-1/d}) / N^{1-1/d}$ for $d \ge 3$
* Propagation of chaos and mean behavior of large particle systems (going back to Dobrushin)
* Mixing and convergence for Markov chains (contraction property)
* Boltzmann equations ($W\_2$ is contracting along solutions, proved by Tanaka)
* Large or infinite dimension: stochastic PDEs, hydrodynamic limits
* Ricci curvature and diffusion equations (Part II of the book)
* Uncertainty principle in quantum physics ($W\_1$, suggested by Werner)
* Classification of metric spaces and "linearly rigid" spaces (Vershik)

</div>

## Chapter 7: Displacement Interpolation

This chapter discusses a **time-dependent** version of optimal transport leading to a *continuous* displacement of measures. The main additional structure assumption is that the cost is associated with an **action** — a way to measure the cost of displacement along a continuous curve. The main result is Theorem 7.21.

### Deterministic Interpolation via Action-Minimizing Curves

The cost function between an initial point $x$ and a final point $y$ is obtained by minimizing the action among paths going from $x$ to $y$:

$$c(x,y) = \inf\left\lbrace \mathcal{A}(\gamma);\quad \gamma_0 = x,\; \gamma_1 = y;\quad \gamma \in \mathcal{C} \right\rbrace,$$

where $\mathcal{C}$ is a class of continuous curves and $\mathcal{A}$ is an action functional.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lagrangian Action)</span></p>

On a smooth Riemannian manifold $M$, the action is classically given by the time-integral of a **Lagrangian** along the path:

$$\mathcal{A}(\gamma) = \int_0^1 L(\gamma_t, \dot{\gamma}_t, t)\, dt,$$

where $L$ is defined on $TM \times [0,1]$ (the tangent bundle times the time interval). Typically $L(x, v, t) = \lvert v \rvert^2 / 2 - V(x)$, where $V$ is a potential. Minimizers with given endpoints satisfy Newton's equation $d^2 x / dt^2 = -\nabla V(x)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Absolutely Continuous Curves)</span></p>

If $(\mathcal{X}, d)$ is a metric space, a continuous curve $\gamma : [0,1] \to \mathcal{X}$ is said to be **absolutely continuous** if there exists a function $\ell \in L^1([0,1]; dt)$ such that for all intermediate times $t\_0 < t\_1$ in $[0,1]$,

$$d(\gamma_{t_0}, \gamma_{t_1}) \le \int_{t_0}^{t_1} \ell(t)\, dt.$$

More generally, it is absolutely continuous of order $p$ if the formula holds with some $\ell \in L^p([0,1]; dt)$.

</div>

#### Key Examples of Action-Minimizing Curves

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.1</span><span class="math-callout__name">($L(x,v,t) = \lvert v \rvert$, Euclidean Length)</span></p>

In $\mathcal{X} = \mathbb{R}^n$, with $L(x,v,t) = \lvert v \rvert$, the action is the length functional. The cost is $c(x,y) = \lvert x - y \rvert$ (Euclidean distance). Minimizing curves are straight lines, with arbitrary parametrization.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.4</span><span class="math-callout__name">($L(x,v,t) = \lvert v \rvert^p$ on a Riemannian Manifold)</span></p>

Let $\mathcal{X} = M$ be a smooth Riemannian manifold, $TM$ its tangent bundle, and $L(x,v,t) = \lvert v \rvert^p$, $p \ge 1$. Then the cost function is $c(x,y) = d(x,y)^p$, where $d$ is the geodesic distance.

* If $p > 1$, minimizing curves satisfy zero acceleration $\ddot{\gamma}\_t = 0$ (covariant derivative), have constant speed, and are called **minimizing, constant-speed geodesics**.
* If $p = 1$, minimizing curves are geodesic curves parametrized in an arbitrary way.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.5</span><span class="math-callout__name">(General Lagrangian on a Riemannian Manifold)</span></p>

For a general Lagrangian $L(x,v,t)$, strictly convex in $v$, on a smooth Riemannian manifold:

* Minimizing curves satisfy the **Euler–Lagrange equation**: $\frac{d}{dt}\bigl[(\nabla\_v L)(\gamma\_t, \dot{\gamma}\_t, t)\bigr] = (\nabla\_x L)(\gamma\_t, \dot{\gamma}\_t, t)$.
* If $L$ is strictly convex and superlinear ($L(x,v,t)/\lvert v \rvert \to +\infty$ as $\lvert v \rvert \to \infty$), then $v \mapsto \nabla\_v L$ is invertible and the Euler–Lagrange equation becomes a differential equation on the new unknown $\nabla\_v L(\gamma, \dot{\gamma}, t)$.
* The **Hamiltonian** $H(x, p, t) := \sup\_{v \in T\_x M} (p \cdot v - L(x,v,t))$ provides access to the rich world of Hamiltonian dynamics.

</div>

### Classical Conditions on a Lagrangian

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.6</span><span class="math-callout__name">(Classical Conditions on a Lagrangian Function)</span></p>

Let $M$ be a smooth, complete Riemannian manifold, and $L(x,v,t)$ a Lagrangian on $TM \times [0,1]$. It is said that $L$ satisfies the **classical conditions** if:

* (a) $L$ is $C^1$ in all variables;
* (b) $L$ is a strictly convex superlinear function of $v$, in the sense of (7.7);
* (c) There are constants $K, C > 0$ such that $L(x,v,t) \ge K \lvert v \rvert - C$ for all $(x,v,t) \in TM \times [0,1]$;
* (d) There is a well-defined locally Lipschitz flow associated to the Euler–Lagrange equation: each action-minimizing curve $\gamma : [0,1] \to M$ belongs to $C^1([0,1]; M)$ and satisfies $(\gamma(t), \dot{\gamma}(t)) = \phi\_t(\gamma(t\_0), \dot{\gamma}(t\_0), t\_0)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.7</span></p>

Assumption (d) above is automatically satisfied if $L$ is of class $C^2$, $\nabla\_v^2 L > 0$ everywhere, and $L$ does not depend on $t$.

</div>

### Length Spaces and Geodesic Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Length Space and Geodesic Space)</span></p>

The **length** of a continuous curve $\gamma$ in a metric space $(\mathcal{X}, d)$ is

$$\mathcal{L}(\gamma) = \sup_{N \in \mathbb{N}} \sup_{0 = t_0 < t_1 < \ldots < t_N = 1} \bigl[d(\gamma_{t_0}, \gamma_{t_1}) + \cdots + d(\gamma_{t_{N-1}}, \gamma_{t_N})\bigr].$$

A metric space $(\mathcal{X}, d)$ is a **length space** if for any two $x, y \in \mathcal{X}$,

$$d(x, y) = \inf_{\gamma \in C([0,1]; \mathcal{X})} \left\lbrace \mathcal{L}(\gamma);\quad \gamma_0 = x,\; \gamma_1 = y \right\rbrace.$$

If in addition $\mathcal{X}$ is complete and locally compact, then the infimum is a minimum, and the space is called a **strictly intrinsic length space** or **geodesic space**. Such spaces play an important role in modern nonsmooth geometry.

</div>

### Abstract Lagrangian Action

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.11</span><span class="math-callout__name">(Lagrangian Action)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, and let $t\_i, t\_f \in \mathbb{R}$. A **Lagrangian action** $(\mathcal{A})^{t\_i, t\_f}$ on $\mathcal{X}$ is a family of lower semicontinuous functionals $\mathcal{A}^{s,t}$ on $C([s,t]; \mathcal{X})$ ($t\_i \le s < t \le t\_f$), and cost functions $c^{s,t}$ on $\mathcal{X} \times \mathcal{X}$, such that:

* (i) **Additivity:** $t\_i \le t\_1 < t\_2 < t\_3 \le t\_f \implies \mathcal{A}^{t\_1, t\_2} + \mathcal{A}^{t\_2, t\_3} = \mathcal{A}^{t\_1, t\_3}$;
* (ii) **Minimality:** $\forall x, y \in \mathcal{X}$, $c^{s,t}(x,y) = \inf\lbrace \mathcal{A}^{s,t}(\gamma);\; \gamma\_s = x,\; \gamma\_t = y \rbrace$;
* (iii) **Reconstruction:** For any curve $(\gamma\_t)\_{t\_i \le t \le t\_f}$,

$$\mathcal{A}^{t_i, t_f}(\gamma) = \sup_{N \in \mathbb{N}} \sup_{t_i = t_0 \le t_1 \le \ldots \le t_N = t_f} \bigl[c^{t_0, t_1}(\gamma_{t_0}, \gamma_{t_1}) + \cdots + c^{t_{N-1}, t_N}(\gamma_{t_{N-1}}, \gamma_{t_N})\bigr].$$

A curve $\gamma : [t\_i, t\_f] \to \mathcal{X}$ is **action-minimizing** if it minimizes $\mathcal{A}$ among all curves with the same endpoints.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.10</span><span class="math-callout__name">(Power-Law Lagrangians)</span></p>

For $L(x, \lvert v \rvert, t) = \lvert v \rvert^p$, the cost function is $c^{s,t}(x,y) = d(x,y)^p / (t-s)^{p-1}$. The cost depends on $s, t$ only through multiplication by a constant. In particular, minimizing curves are independent of $s$ and $t$, up to reparametrization.

</div>

### Coercive Actions and Their Properties

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.13</span><span class="math-callout__name">(Coercive Action)</span></p>

Let $(\mathcal{A})^{0,1}$ be a Lagrangian action on a Polish space $\mathcal{X}$, with associated cost functions $(c^{s,t})\_{0 \le s < t \le 1}$. The action is called **coercive** if:

* (i) It is bounded below: $\inf\_{s < t} \inf\_\gamma \mathcal{A}^{s,t}(\gamma) > -\infty$;
* (ii) For any two times $s < t$, and any two nonempty compact sets $K\_s, K\_t \subset \mathcal{X}$ such that $c^{s,t}(x,y) < +\infty$ for all $x \in K\_s$, $y \in K\_t$, the set $\Gamma^{s,t}\_{K\_s \to K\_t}$ of minimizing paths starting from $K\_s$ at time $s$ and ending in $K\_t$ at time $t$ is compact and nonempty.

In particular, minimizing curves between any two fixed points $x, y$ with $c(x,y) < +\infty$ always exist and form a compact set.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.16</span><span class="math-callout__name">(Properties of Lagrangian Actions)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space and $(\mathcal{A})^{0,1}$ a coercive Lagrangian action on $\mathcal{X}$. Then:

* (i) For all intermediate times $s < t$, $c^{s,t}$ is lower semicontinuous on $\mathcal{X} \times \mathcal{X}$, with values in $\mathbb{R} \cup \lbrace +\infty \rbrace$.
* (ii) **Restriction:** If a curve $\gamma$ on $[s,t]$ is a minimizer of $\mathcal{A}^{s,t}$, then its restriction to $[s', t'] \subset [s,t]$ is also a minimizer for $\mathcal{A}^{s', t'}$.
* (iii) **Dynamic programming:** For all times $t\_1 < t\_2 < t\_3$ in $[0,1]$, and all $x\_1, x\_3 \in \mathcal{X}$,

$$c^{t_1, t_3}(x_1, x_3) = \inf_{x_2 \in \mathcal{X}} \bigl(c^{t_1, t_2}(x_1, x_2) + c^{t_2, t_3}(x_2, x_3)\bigr),$$

and if the infimum is achieved at some $x\_2$, then there is a minimizing curve from $x\_1$ at time $t\_1$ to $x\_3$ at time $t\_3$ passing through $x\_2$ at time $t\_2$.

* (iv) **Characterization:** A curve $\gamma$ is a minimizer of $\mathcal{A}$ if and only if, for all intermediate times $t\_1 < t\_2 < t\_3$ in $[0,1]$,

$$c^{t_1, t_3}(\gamma_{t_1}, \gamma_{t_3}) = c^{t_1, t_2}(\gamma_{t_1}, \gamma_{t_2}) + c^{t_2, t_3}(\gamma_{t_2}, \gamma_{t_3}).$$

* (v) If the cost functions $c^{s,t}$ are continuous, then the set $\Gamma$ of all action-minimizing curves is closed in the topology of uniform convergence.
* (vi) For all times $s < t$, there exists a Borel map $S\_{s \to t} : \mathcal{X} \times \mathcal{X} \to C([s,t]; \mathcal{X})$, such that for all $x, y \in \mathcal{X}$, $S(x,y)$ is a minimizing curve from $x$ to $y$.

</div>

### Interpolation of Random Variables: Displacement Interpolation

Action-minimizing curves provide a framework to interpolate between points. The idea is to *lift* this to an interpolation between *probability measures*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.19</span><span class="math-callout__name">(Dynamical Coupling)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space. A **dynamical transference plan** $\Pi$ is a probability measure on the space $C([0,1]; \mathcal{X})$. A **dynamical coupling** of two probability measures $\mu\_0, \mu\_1 \in P(\mathcal{X})$ is a random curve $\gamma : [0,1] \to \mathcal{X}$ such that $\mathrm{law}(\gamma\_0) = \mu\_0$, $\mathrm{law}(\gamma\_1) = \mu\_1$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.20</span><span class="math-callout__name">(Dynamical Optimal Coupling)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, $(\mathcal{A})^{0,1}$ a Lagrangian action on $\mathcal{X}$, $c$ the associated cost, and $\Gamma$ the set of action-minimizing curves. A **dynamical optimal transference plan** is a probability measure $\Pi$ on $\Gamma$ such that

$$\pi_{0,1} := (e_0, e_1)\_\# \Pi$$

is an optimal transference plan between $\mu\_0$ and $\mu\_1$. Equivalently, $\Pi$ is the law of a random action-minimizing curve whose endpoints constitute an optimal coupling of $(\mu\_0, \mu\_1)$.

The procedure of defining $\mu\_t := (e\_t)\_\# \Pi$ is called **displacement interpolation**, by opposition to the standard linear interpolation $\mu\_t = (1-t)\mu\_0 + t\mu\_1$. Note that there is a priori no uniqueness.

</div>

### The Main Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.21</span><span class="math-callout__name">(Displacement Interpolation)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, and $(\mathcal{A})^{0,1}$ a coercive Lagrangian action on $\mathcal{X}$, with continuous cost functions $c^{s,t}$. Let $C^{s,t}(\mu, \nu)$ denote the optimal transport cost between $\mu$ and $\nu$ for the cost $c^{s,t}$; write $c = c^{0,1}$ and $C = C^{0,1}$. Let $\mu\_0$ and $\mu\_1$ be any two probability measures on $\mathcal{X}$ such that $C(\mu\_0, \mu\_1)$ is finite. Then, given a continuous path $(\mu\_t)\_{0 \le t \le 1}$, the following properties are equivalent:

* **(i)** For each $t \in [0,1]$, $\mu\_t$ is the law of $\gamma\_t$, where $(\gamma\_t)\_{0 \le t \le 1}$ is a dynamical optimal coupling of $(\mu\_0, \mu\_1)$;
* **(ii)** For any three intermediate times $t\_1 < t\_2 < t\_3$ in $[0,1]$,

$$C^{t_1, t_2}(\mu_{t_1}, \mu_{t_2}) + C^{t_2, t_3}(\mu_{t_2}, \mu_{t_3}) = C^{t_1, t_3}(\mu_{t_1}, \mu_{t_3});$$

* **(iii)** The path $(\mu\_t)\_{0 \le t \le 1}$ is a minimizing curve for the coercive action functional defined on $P(\mathcal{X})$ by

$$\mathbb{A}^{s,t}(\mu) = \sup_{N \in \mathbb{N}} \sup_{s = t_0 < t_1 < \ldots < t_N = t} \sum_{i=0}^{N-1} C^{t_i, t_{i+1}}(\mu_{t_i}, \mu_{t_{i+1}}) = \inf_\gamma \mathbb{E}\, \mathcal{A}^{s,t}(\gamma),$$

where the last infimum is over all random curves $\gamma : [s,t] \to \mathcal{X}$ such that $\mathrm{law}(\gamma\_\tau) = \mu\_\tau$ for $s \le \tau \le t$.

In that case $(\mu\_t)\_{0 \le t \le 1}$ is said to be a **displacement interpolation** between $\mu\_0$ and $\mu\_1$. There always exists at least one such curve.

Finally, if $\mathcal{K}\_0$ and $\mathcal{K}\_1$ are two compact subsets of $P(\mathcal{X})$ with $C^{0,1}(\mu\_0, \mu\_1) < +\infty$ for all $\mu\_0 \in \mathcal{K}\_0$, $\mu\_1 \in \mathcal{K}\_1$, then the set of dynamical optimal transference plans $\Pi$ with $(e\_0)\_\# \Pi \in \mathcal{K}\_0$ and $(e\_1)\_\# \Pi \in \mathcal{K}\_1$ is **compact**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation)</span></p>

The equivalence (i) $\Leftrightarrow$ (ii) says that a displacement interpolation is characterized by the property that the optimal transport cost between any two intermediate times *adds up*. The equivalence (i) $\Leftrightarrow$ (iii) says: **"A geodesic in the space of laws is the law of a geodesic."**

</div>

### Displacement Interpolation as Geodesics

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 7.22</span><span class="math-callout__name">(Displacement Interpolation as Geodesics)</span></p>

Let $(\mathcal{X}, d)$ be a complete separable, locally compact length space. Let $p > 1$ and let $P\_p(\mathcal{X})$ be the space of probability measures on $\mathcal{X}$ with finite moment of order $p$, metrized by $W\_p$. Then, given any two $\mu\_0, \mu\_1 \in P\_p(\mathcal{X})$, and a continuous curve $(\mu\_t)\_{0 \le t \le 1}$ valued in $P(\mathcal{X})$, the following are equivalent:

* (i) $\mu\_t$ is the law of $\gamma\_t$, where $\gamma$ is a random (minimizing, constant-speed) geodesic such that $(\gamma\_0, \gamma\_1)$ is an optimal coupling;
* (ii) $(\mu\_t)\_{0 \le t \le 1}$ is a **geodesic** curve in the space $P\_p(\mathcal{X})$.

Moreover, if $\mu\_0$ and $\mu\_1$ are given, there always exists at least one such curve. More generally, the set of geodesic curves $(\mu\_t)$ with $\mu\_0 \in \mathcal{K}\_0$ and $\mu\_1 \in \mathcal{K}\_1$ (compact subsets of $P\_p(\mathcal{X})$) is compact and nonempty.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 7.23</span><span class="math-callout__name">(Uniqueness of Displacement Interpolation)</span></p>

With the same assumptions as in Theorem 7.21, if:

* (a) there is a unique optimal transport plan $\pi$ between $\mu\_0$ and $\mu\_1$;
* (b) $\pi(dx\_0\, dx\_1)$-almost surely, $x\_0$ and $x\_1$ are joined by a unique minimizing curve;

then there is a **unique** displacement interpolation $(\mu\_t)\_{0 \le t \le 1}$ joining $\mu\_0$ to $\mu\_1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.24</span></p>

In Corollary 7.22, $\mathcal{A}^{s,t}(\gamma) = \int\_s^t \lvert \dot{\gamma}\_\tau \rvert^p\, d\tau$. Then action-minimizing curves in $\mathcal{X}$ are the same whatever the value of $p > 1$. Yet geodesics in $P\_p(\mathcal{X})$ are not the same for different values of $p$, because a coupling which is optimal for a certain value of $p$ might not be optimal for another.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.26</span><span class="math-callout__name">(Open Questions)</span></p>

Theorem 7.21 leaves open several natural questions, all of which will be answered affirmatively in the sequel under suitable regularity assumptions:

* Is there a differential equation for geodesic curves $(\mu\_t)\_{0 \le t \le 1}$? (Related to defining a tangent space in the space of measures.)
* Is there a more explicit formula for the action on the space of probability measures, e.g. $\int\_0^1 \mathbb{L}(\mu\_t, \dot{\mu}\_t, t)\, dt$ for some Lagrangian $\mathbb{L}$?
* Are geodesic paths nonbranching? (Does the velocity at initial time uniquely determine the final measure $\mu\_1$?)
* Can one identify simple conditions for the existence of a *unique* geodesic path between two given probability measures?

</div>

### Lipschitz Continuity of Moments

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.29</span><span class="math-callout__name">($W\_p$-Lipschitz Continuity of $p$-Moments)</span></p>

Let $(\mathcal{X}, d)$ be a locally compact Polish length space, let $p \ge 1$ and $\mu, \nu \in P\_p(\mathcal{X})$. Then for any $\varphi \in \mathrm{Lip}(\mathcal{X}; \mathbb{R}\_+)$,

$$\left\lvert \left(\int \varphi(x)^p\, \mu(dx)\right)^{1/p} - \left(\int \varphi(y)^p\, \nu(dy)\right)^{1/p} \right\rvert \le \lVert \varphi \rVert_{\mathrm{Lip}}\, W_p(\mu, \nu).$$

</div>

### Interpolation Between Intermediate Times and Restriction

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.30</span><span class="math-callout__name">(Interpolation from Intermediate Times and Restriction)</span></p>

Let $\mathcal{X}$ be a Polish space equipped with a coercive action $(\mathcal{A})$ on $C([0,1]; \mathcal{X})$. Let $\Pi \in P(C([0,1]; \mathcal{X}))$ be a dynamical optimal transport plan associated with a finite total cost. For any $0 \le t\_0 < t\_1 \le 1$, define the time-restriction of $\Pi$ to $[t\_0, t\_1]$ as $\Pi^{t\_0, t\_1} := (r\_{t\_0, t\_1})\_\# \Pi$. Then:

* (i) $\Pi^{t\_0, t\_1}$ is a dynamical optimal coupling for the action $(\mathcal{A})^{t\_0, t\_1}$.
* (ii) **Restriction:** If $\widetilde{\Pi}$ is a measure on $C([t\_0, t\_1]; \mathcal{X})$ with $\widetilde{\Pi} \le \Pi^{t\_0, t\_1}$ and $\widetilde{\Pi}[C([t\_0, t\_1]; \mathcal{X})] > 0$, then $\Pi' := \widetilde{\Pi} / \widetilde{\Pi}[C([t\_0, t\_1]; \mathcal{X})]$ is a dynamical optimal coupling between its marginals, and $(\mu\_t')\_{t\_0 \le t \le t\_1}$ is a displacement interpolation.
* (iii) **Uniqueness propagation:** If action-minimizing curves are uniquely and measurably determined by their restriction to a nontrivial time-interval, and $(t\_0, t\_1) \ne (0,1)$, then $\Pi'$ in (ii) is the *unique* dynamical optimal coupling between $\mu'\_{t\_0}$ and $\mu'\_{t\_1}$, and $(\mu\_t')$ is the unique displacement interpolation.
* (iv) **No crossing:** Under the assumptions of (iii), for any $t \in (0,1)$, $(\Pi \otimes \Pi)$-almost surely, $[\gamma\_t = \widetilde{\gamma}\_t] \implies [\gamma = \widetilde{\gamma}]$. The curves seen by $\Pi$ **cannot cross** at intermediate times.
* (v) Under the same assumptions, there is a measurable map $F\_t : \mathcal{X} \to \Gamma(\mathcal{X})$ such that, $\Pi(d\gamma)$-almost surely, $F\_t(\gamma\_t) = \gamma$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 7.32</span><span class="math-callout__name">(Nonbranching is Inherited by the Wasserstein Space)</span></p>

Let $(\mathcal{X}, d)$ be a complete separable, locally compact length space and let $p \in (1, \infty)$. Assume that $\mathcal{X}$ is **nonbranching**, in the sense that a geodesic $\gamma : [0,1] \to \mathcal{X}$ is uniquely determined by its restriction to a nontrivial time-interval. Then also the Wasserstein space $P\_p(\mathcal{X})$ is nonbranching. Conversely, if $P\_p(\mathcal{X})$ is nonbranching, then $\mathcal{X}$ is nonbranching.

</div>

### Interpolation of Prices

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.33</span><span class="math-callout__name">(Hamilton–Jacobi–Hopf–Lax–Oleinik Evolution Semigroup)</span></p>

Let $\mathcal{X}$ be a metric space and $(\mathcal{A})^{0,1}$ a coercive Lagrangian action on $\mathcal{X}$, with cost functions $(c^{s,t})\_{0 \le s < t \le 1}$. For any two functions $\psi : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$, $\phi : \mathcal{X} \to \mathbb{R} \cup \lbrace -\infty \rbrace$, and any two times $0 \le s < t \le 1$, define

$$H_+^{s,t} \psi(y) = \inf_{x \in \mathcal{X}} \bigl(\psi(x) + c^{s,t}(x,y)\bigr); \qquad H_-^{t,s} \phi(x) = \sup_{y \in \mathcal{X}} \bigl(\phi(y) - c^{s,t}(x,y)\bigr).$$

The family of operators $(H\_+^{s,t})\_{t > s}$ (resp. $(H\_-^{s,t})\_{s < t}$) is called the **forward** (resp. **backward**) **Hamilton–Jacobi** (or Hopf–Lax, or Lax–Oleinik) **semigroup**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.34</span><span class="math-callout__name">(Elementary Properties of Hamilton–Jacobi Semigroups)</span></p>

With the notation of Definition 7.33:

* (i) $H\_+^{s,t}$ and $H\_-^{s,t}$ are **order-preserving**: $\psi \le \overline{\psi} \implies H\_\pm^{s,t} \psi \le H\_\pm^{s,t} \overline{\psi}$.
* (ii) **Semigroup property:** Whenever $t\_1 < t\_2 < t\_3$ are three intermediate times in $[0,1]$,

$$H_+^{t_2, t_3} H_+^{t_1, t_2} = H_+^{t_1, t_3}; \qquad H_-^{t_2, t_1} H_-^{t_3, t_2} = H_-^{t_3, t_1}.$$

* (iii) Whenever $s < t$ are two times in $[0,1]$,

$$H_-^{t,s} H_+^{s,t} \le \mathrm{Id}; \qquad H_+^{s,t} H_-^{t,s} \ge \mathrm{Id}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hamilton–Jacobi PDE)</span></p>

On a smooth Riemannian manifold, when the action is given by a Lagrangian $L(x,v,t)$, strictly convex and superlinear in $v$, the function $S\_+(x, t) := H\_+^{0,t} \psi\_0$ solves the **Hamilton–Jacobi equation**

$$\frac{\partial S_+}{\partial t}(x, t) + H\bigl(x, \nabla S_+(x, t), t\bigr) = 0,$$

where $H = L^\ast$ is the **Hamiltonian** (Legendre transform of $L$ in the velocity variable). This equation bridges between a Lagrangian description of action-minimizing curves and an Eulerian description. From $S\_+(x, t)$ one can reconstruct a velocity field $v(x, t) = \nabla\_p H(x, \nabla S\_+(x, t), t)$ whose integral curves are minimizing. But $S\_+$ is not in general differentiable everywhere, so the equation must be interpreted in a suitable sense (viscosity sense).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.35</span><span class="math-callout__name">(Quadratic Cost on a Riemannian Manifold)</span></p>

On a Riemannian manifold $M$ with $L(x,v,t) = \lvert v \rvert^2 / 2$, the Hamiltonian is $H(x,p,t) = \lvert p \rvert^2 / 2$. If $S\_0$ is a given Lipschitz function and $S\_+(t, x) := H\_+^{0,t} S\_0$, then

$$\frac{\partial S_+}{\partial t} + \frac{\lvert \nabla^- S_+ \rvert^2}{2} = 0, \qquad \lvert \nabla^- f \rvert(x) := \limsup_{y \to x} \frac{[f(y) - f(x)]\_-}{d(x,y)}.$$

Conversely, using the backward semigroup, $S\_-(x, t) := H\_-^{t,1} \phi\_1$ satisfies

$$\frac{\partial S_-}{\partial t} + \frac{\lvert \nabla^+ S_- \rvert^2}{2} = 0, \qquad \lvert \nabla^+ f \rvert(x) := \limsup_{y \to x} \frac{[f(y) - f(x)]\_+}{d(x,y)}.$$

</div>

### Interpolation of Prices

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.36</span><span class="math-callout__name">(Interpolation of Prices)</span></p>

With the same assumptions and notation as in Definition 7.33, let $\mu\_0$, $\mu\_1$ be two probability measures on $\mathcal{X}$ such that $C^{0,1}(\mu\_0, \mu\_1) < +\infty$, and let $(\psi\_0, \phi\_1)$ be a pair of $c^{0,1}$-conjugate functions such that any optimal plan $\pi\_{0,1}$ between $\mu\_0$ and $\mu\_1$ has its support included in $\partial\_c \psi$. Further, let $(\mu\_t)\_{0 \le t \le 1}$ be a displacement interpolation between $\mu\_0$ and $\mu\_1$. Whenever $s < t$ are two intermediate times in $[0,1]$, define

$$\psi_s := H_+^{0,s} \psi_0, \qquad \phi_t := H_-^{1,t} \phi_1.$$

Then $(\psi\_s, \phi\_t)$ is optimal in the dual Kantorovich problem associated to $(\mu\_s, \mu\_t)$ and cost $c^{s,t}$. In particular,

$$C^{s,t}(\mu_s, \mu_t) = \int \phi_t\, d\mu_t - \int \psi_s\, d\mu_s,$$

and $\phi\_t(y) - \psi\_s(x) \le c^{s,t}(x,y)$, with equality $\pi\_{s,t}(dx\, dy)$-almost surely, where $\pi\_{s,t}$ is any optimal transference plan between $\mu\_s$ and $\mu\_t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.37</span></p>

In the limit case $s \to t$, the above results become $\phi\_t \le \psi\_t$ and $\phi\_t = \psi\_t$ $\mu\_t$-almost surely. But it is not true in general that $\phi\_t = \psi\_t$ everywhere in $\mathcal{X}$. However, $\psi\_1 = \phi\_1$ holds everywhere as a consequence of the definitions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 7.39</span><span class="math-callout__name">(Displacement Interpolation of Geometric Objects)</span></p>

For a quadratic Lagrangian:

* (i) The displacement interpolation between two balls in Euclidean space is always a ball, whose radius increases linearly in time.
* (ii) More generally, the displacement interpolation between two ellipsoids is always an ellipsoid.
* (iii) But the displacement interpolation between two general sets is in general not a set.
* (iv) The displacement interpolation between two spherical caps on the sphere is in general not a spherical cap.
* (v) The displacement interpolation between two antipodal spherical caps on the sphere is unique, while the displacement interpolation between two antipodal points can be realized in infinitely many ways.

</div>

### Appendix: Paths in Metric Structures

This appendix provides a crash course in Riemannian geometry and its nonsmooth generalizations, as needed for the theory of displacement interpolation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Riemannian Manifold — Basic Concepts)</span></p>

A (finite-dimensional, smooth) Riemannian manifold is a manifold $M$ equipped with a **Riemannian metric** $g$: a scalar product on each tangent space $T\_x M$, varying smoothly with $x$. The **length** of a smooth path $\gamma : [0,1] \to M$ is

$$\mathcal{L}(\gamma) = \int_0^1 \lvert \dot{\gamma}_t \rvert\, dt,$$

and the **geodesic distance** between two points $x$ and $y$ is

$$d(x, y) = \inf\left\lbrace \mathcal{L}(\gamma);\quad \gamma_0 = x,\; \gamma_1 = y \right\rbrace.$$

Any one of the three objects (metric, length, distance) determines the other two. In particular, the metric can be recovered from the distance via $\lvert \dot{\gamma}\_0 \rvert = \lim\_{t \downarrow 0} d(\gamma\_0, \gamma\_t)/t$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Covariant Derivative and Parallel Transport)</span></p>

There is in general no canonical way to identify tangent spaces $T\_x M$ and $T\_y M$ if $x \ne y$. But there is a canonical way to identify $T\_{\gamma\_0} M$ and $T\_{\gamma\_t} M$ as $t$ varies continuously along a curve $\gamma$: this is called **parallel transport** (or Levi-Civita transport).

The **covariant derivative** of a vector field $\xi$ along $\gamma$ is defined by

$$\dot{\xi}(t_0) = \frac{d}{dt}\bigg\rvert_{t=t_0} \theta_{t \to t_0}(\xi(\gamma_t)),$$

where $\theta\_{t \to t\_0}$ is the parallel transport from $T\_{\gamma\_t} M$ to $T\_{\gamma\_{t\_0}} M$ along $\gamma$. This is denoted $\nabla\_{\dot{\gamma}} \xi$, or $D\xi/Dt$, or $d\xi/dt$. In $\mathbb{R}^n$, $\nabla\_{\dot{\gamma}} \xi$ coincides with $(\dot{\gamma} \cdot \nabla)\xi$. In coordinates, using **Christoffel symbols** $\Gamma\_{ij}^k$:

$$\left(\frac{D\xi}{Dt}\right)^k = \frac{d\xi^k}{dt} + \sum_{ij} \Gamma_{ij}^k\, \dot{\gamma}^i\, \xi^j.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Euler–Lagrange Equation and Geodesics)</span></p>

Let $L(x, v, t)$ be a smooth Lagrangian on $TM \times [0,1]$. The **first variation formula** gives:

$$d\mathcal{A}(\gamma) \cdot h = \int_0^1 \left(\nabla_x L - \frac{d}{dt}(\nabla_v L)\right)(\gamma_t, \dot{\gamma}_t, t) \cdot h(t)\, dt + (\nabla_v L)(\gamma_1, \dot{\gamma}_1, 1) \cdot h(1) - (\nabla_v L)(\gamma_0, \dot{\gamma}_0, 0) \cdot h(0).$$

For curves with fixed endpoints ($h(0) = h(1) = 0$), the equation for minimizers is the **Euler–Lagrange equation**:

$$\frac{d}{dt} \nabla_v L = \nabla_x L, \qquad \text{i.e.} \quad \frac{d}{dt}\bigl(\nabla_v L(\gamma_t, \dot{\gamma}_t, t)\bigr) = \nabla_x L(\gamma_t, \dot{\gamma}_t, t).$$

For $L(x, v, t) = \lvert v \rvert^2 / 2$, this reduces to $\nabla\_{\dot{\gamma}} \dot{\gamma} = 0$ (zero acceleration). Curves satisfying this equation are called **geodesics**; in coordinates, $\ddot{\gamma}^k + \sum\_{ij} \Gamma\_{ij}^k \dot{\gamma}^i \dot{\gamma}^j = 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Exponential Map)</span></p>

If $x \in M$ and $v \in T\_x M$, the **exponential map** $\exp\_x v$ is defined as $\gamma(1)$, where $\gamma : [0,1] \to M$ is the unique constant-speed geodesic starting from $\gamma(0) = x$ with velocity $\dot{\gamma}(0) = v$. The exponential map is a convenient notation to handle "all" geodesics of a Riemannian manifold at the same time.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Minimizing Geodesics)</span></p>

A curve which minimizes the action between its endpoints is called a **minimizing geodesic**. The Hopf–Rinow theorem guarantees that if $M$ (seen as a metric space) is complete, then any two points in $M$ are joined by at least one minimal geodesic. Geodesics on a Riemannian manifold enjoy the following properties:

* **Nonbranching:** Two geodesics defined on $[0, t]$ that coincide on $[0, t']$ for some $t' > 0$ must coincide on the whole of $[0, t]$ (consequence of the Cauchy–Lipschitz theorem).
* **Locally unique:** For any given $x$, there is $r\_x > 0$ such that any $y$ in $B\_{r\_x}(x)$ can be connected to $x$ by a single geodesic, and $y \mapsto \dot{\gamma}(0)$ is a diffeomorphism.
* **Almost everywhere unique:** For any $x$, the set of points $y$ that can be connected to $x$ by several minimizing geodesics is of zero measure (because $d^2(x, \cdot)$ is locally semiconcave, hence differentiable almost everywhere).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Finsler and Length Spaces)</span></p>

**Finsler structures** generalize Riemannian manifolds: one has a norm (not necessarily from a scalar product) on each tangent space $T\_x M$. **Length spaces** go further: no differentiable structure is required, only a length $\mathcal{L}$ and a distance $d$ which are compatible. A **geodesic space** is a complete, locally compact length space (generalization of Hopf–Rinow). The main practical differences from Riemannian manifolds are: (i) no equation for geodesic curves, (ii) geodesics may branch, (iii) no guarantee of local uniqueness, (iv) no canonical notion of dimension or reference measure, (v) geodesics need not be almost everywhere unique. Nevertheless, there is a theory of differential analysis on nonsmooth geodesic spaces, mainly when there are lower bounds on the sectional curvature (Alexandrov spaces).

</div>

### Bibliographical Notes on Chapter 7

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Benamou–Brenier Formula)</span></p>

The concept and denomination of **displacement interpolation** was introduced by McCann in the particular case of the quadratic cost in Euclidean space. Brenier then understood that this procedure could be recast as an action minimization problem in the space of measures. In Brenier's formulation, the action is defined by

$$\mathbb{A}(\mu) = \inf_{v(t,x)} \left\lbrace \int_0^1 \int \lvert v(t, x) \rvert^2\, d\mu_t(x)\, dt;\quad \frac{\partial \mu}{\partial t} + \nabla \cdot (v\mu) = 0 \right\rbrace,$$

and then one has the **Benamou–Brenier formula**:

$$W_2(\mu_0, \mu_1)^2 = \inf \mathbb{A}(\mu),$$

where the infimum is taken among all paths $(\mu\_t)\_{0 \le t \le 1}$ satisfying suitable regularity conditions. This formula was one of Benamou and Brenier's motivations for devising new numerical methods for optimal transport.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Nelson's Stochastic Mechanics)</span></p>

A remarkable precursor to displacement interpolation is **Nelson's theory of stochastic mechanics**, in which quantum effects are explained by stochastic fluctuations. Nelson considered the action minimization problem $\inf \mathbb{E} \int\_0^1 \lvert \dot{X}\_t \rvert^2\, dt$ where the infimum is over all random paths $(X\_t)$ with $\mathrm{law}(X\_0) = \mu\_0$, $\mathrm{law}(X\_1) = \mu\_1$, and $X\_t$ solving a stochastic differential equation $dX\_t = \sigma\, dB\_t + \xi(t, X\_t)\, dt$ (minimization is also over all drifts $\xi$). Nelson made the incredible discovery that minimizers produce solutions of the free Schrödinger equation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Further Developments)</span></p>

* It was **Otto** who first explicitly reformulated the Benamou–Brenier formula as the equation for a geodesic distance on a Riemannian-like structure on the space of measures.
* **Ambrosio, Gigli and Savaré** pointed out that for the geodesic property, it is simpler to use the metric notion of geodesic in a length space.
* The displacement interpolation for more general cost functions arising from a smooth Lagrangian was constructed by **Bernard and Buffoni**, who first introduced the Property (ii) in Theorem 7.21 and made explicit the link with the **Mather minimization problem**.
* **Lisini** obtained representation theorems for general absolutely continuous paths in the Wasserstein space $P\_p(\mathcal{X})$ ($p > 1$) on a general Polish space, removing the assumption of local compactness.
* Important applications of Lagrangian cost functions of the form $L(x,v,t) = \lvert v \rvert^2/2 - U(t,x)$ arise in incompressible fluid mechanics (where $U$ is the pressure field, studied by **Ambrosio and Figalli**) and in the theory of **Ricci flow** (where $U$ is the scalar curvature evolving according to Ricci flow, studied by **Topping** and **Lott** in relation to **Perelman's $\mathcal{L}$-distance**).

</div>

## Chapter 8: The Monge–Mather Shortening Principle

This chapter develops a powerful regularity tool for optimal transport: the observation that optimal transport curves (action-minimizing paths paired by an optimal coupling) **cannot cross** at intermediate times. This principle, originating in Monge's work for the Euclidean distance cost and generalized by Mather for Lagrangian cost functions, has deep consequences for the regularity of displacement interpolations.

### Monge's Observation (Euclidean Distance Cost)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Monge's Non-Crossing Principle)</span></p>

For the transport cost $c(x,y) = \lvert x - y \rvert$ in the Euclidean plane, consider two pairs $(x\_1, y\_1)$, $(x\_2, y\_2)$ in the support of an optimal coupling $\pi$. Then either all four points are collinear, or the line segments $[x\_1, y\_1]$ and $[x\_2, y\_2]$ do not cross (except possibly at their endpoints). If the lines crossed at an interior point, then by the triangle inequality

$$\lvert x_1 - y_2 \rvert + \lvert x_2 - y_1 \rvert < \lvert x_1 - y_1 \rvert + \lvert x_2 - y_2 \rvert,$$

contradicting the $c$-cyclical monotonicity of $\mathrm{Spt}\, \pi$. In other words: given two crossing line segments, one can **shorten** the total length by rerouting.

</div>

### Quadratic Cost Function

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-Crossing for $c(x,y) = \lvert x - y \rvert^2$)</span></p>

For the quadratic cost $c(x,y) = \lvert x - y \rvert^2$ in $\mathbb{R}^n$, Monge's argument about non-crossing of line segments does not apply directly (the cost does not satisfy the triangle inequality, so pathlines *can* cross). However, the *time-dependent* curves $\gamma\_1(t) = (1-t)x\_1 + t y\_1$ and $\gamma\_2(t) = (1-t)x\_2 + t y\_2$ **cannot meet at intermediate times**. Indeed, by cyclical monotonicity:

$$\lvert x_1 - y_1 \rvert^2 + \lvert x_2 - y_2 \rvert^2 \le \lvert x_1 - y_2 \rvert^2 + \lvert x_2 - y_1 \rvert^2.$$

A direct computation yields the identity:

$$\lvert \gamma_1(t) - \gamma_2(t) \rvert^2 = (1-t)^2 \lvert x_1 - x_2 \rvert^2 + t^2 \lvert y_1 - y_2 \rvert^2 + t(1-t)\bigl(\lvert x_1 - y_2 \rvert^2 + \lvert x_2 - y_1 \rvert^2 - \lvert x_1 - y_1 \rvert^2 - \lvert x_2 - y_2 \rvert^2\bigr).$$

The last term is nonnegative by cyclical monotonicity, so the distance $\lvert \gamma\_1(t) - \gamma\_2(t) \rvert$ vanishes at some $t\_0 \in (0,1)$ only if $x\_1 - y\_1 = x\_2 - y\_2$, i.e. the two curves are identical.

Moreover, for any $t\_0 \in (0,1)$, the uniform distance between the *whole paths* is controlled by the distance at time $t\_0$:

$$\sup_{0 \le t \le 1} \lvert \gamma_1(t) - \gamma_2(t) \rvert \le \max\!\left(\frac{1}{t_0},\; \frac{1}{1-t_0}\right) \lvert \gamma_1(t_0) - \gamma_2(t_0) \rvert.$$

</div>

### Mather's Shortening Lemma (General Statement)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.1</span><span class="math-callout__name">(Mather's Shortening Lemma)</span></p>

Let $M$ be a smooth Riemannian manifold equipped with its geodesic distance $d$, and let $c(x,y)$ be a cost function on $M \times M$, defined by a Lagrangian $L(x,v,t)$ on $TM \times [0,1]$. Let $x\_1, x\_2, y\_1, y\_2$ be four points on $M$ such that

$$c(x_1, y_1) + c(x_2, y_2) \le c(x_1, y_2) + c(x_2, y_1).$$

Further, let $\gamma\_1$ and $\gamma\_2$ be two action-minimizing curves respectively joining $x\_1$ to $y\_1$ and $x\_2$ to $y\_2$. Let $V$ be a bounded neighborhood of the graphs of $\gamma\_1$ and $\gamma\_2$ in $M \times [0,1]$, and $S$ a strict upper bound on the maximal speed along these curves. Define

$$\mathcal{V} := \bigcup_{(x,t) \in V} \bigl(x, B_S(0), t\bigr) \subset TM \times [0,1].$$

Assume that:

* (i) minimizing curves for $L$ are solutions of a Lipschitz flow, in the sense of Definition 7.6(d);
* (ii) $L$ is of class $C^{1,\alpha}$ in $\mathcal{V}$ with respect to the variables $x$ and $v$, for some $\alpha \in (0, 1]$;
* (iii) $L$ is $(2+\kappa)$-convex in $\mathcal{V}$, with respect to the $v$ variable.

Then, for any $t\_0 \in (0,1)$, there is a constant $C\_{t\_0} = C(L, \mathcal{V}, t\_0)$ and a positive exponent $\beta = \beta(\alpha, \kappa)$ such that

$$\sup_{0 \le t \le 1} d\bigl(\gamma_1(t), \gamma_2(t)\bigr) \le C_{t_0}\, d\bigl(\gamma_1(t_0), \gamma_2(t_0)\bigr)^\beta.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 8.2</span><span class="math-callout__name">(Mather's Shortening Lemma — $C^2$ Case)</span></p>

Let $M$ be a smooth Riemannian manifold and let $L = L(x,v,t)$ be a $C^2$ Lagrangian on $TM \times [0,1]$, satisfying the classical conditions of Definition 7.6, together with $\nabla\_v^2 L > 0$. Let $c$ be the cost function associated to $L$, and let $d$ be the geodesic distance on $M$. Then, for any compact $K \subset M$ there is a constant $C\_K$ such that, whenever $x\_1, y\_1, x\_2, y\_2$ are four points in $K$ with

$$c(x_1, y_1) + c(x_2, y_2) \le c(x_1, y_2) + c(x_2, y_1),$$

and $\gamma\_1$, $\gamma\_2$ are action-minimizing curves joining respectively $x\_1$ to $y\_1$ and $x\_2$ to $y\_2$, then for any $t\_0 \in (0,1)$,

$$\sup_{0 \le t \le 1} d\bigl(\gamma_1(t), \gamma_2(t)\bigr) \le \frac{C_K}{\min(t_0, 1-t_0)}\, d\bigl(\gamma_1(t_0), \gamma_2(t_0)\bigr).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation)</span></p>

The short version: the distance between $\gamma\_1$ and $\gamma\_2$ is controlled, **uniformly in $t$**, by the distance at **any** intermediate time $t\_0 \in (0,1)$. In particular, the initial and final distance between these curves is controlled by their distance at any intermediate time. (But the *final* distance is not controlled by the *initial* distance!) These are quantitative versions of the qualitative statement that the two curves, if distinct, cannot cross except at initial or final time.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.3</span><span class="math-callout__name">($c(x,y) = d(x,y)^2$)</span></p>

The cost function $c(x,y) = d(x,y)^2$ corresponds to the Lagrangian $L(x,v,t) = \lvert v \rvert^2$, which obviously satisfies the assumptions of Corollary 8.2. In that case the exponent $\beta = 1$ is admissible. It is natural to expect that the constant $C\_K$ can be controlled in terms of just a **lower bound on the sectional curvature** of $M$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.4</span><span class="math-callout__name">($c(x,y) = d(x,y)^{1+\alpha}$, $0 < \alpha < 1$)</span></p>

The cost function $c(x,y) = d(x,y)^{1+\alpha}$ does not satisfy the assumptions of Corollary 8.2 (the associated Lagrangian $L = \lvert v \rvert^{1+\alpha}$ is not smooth). But Assumption (i) in Theorem 8.1 is still satisfied (minimizing curves are geodesics). By tracking exponents, $\beta = (1+\alpha)/(3-\alpha)$. However, by exploiting the homogeneity of the power function, one can prove that $\beta = 1$ is also admissible for all $\alpha \in (0,1)$ (only the constant deteriorates as $\alpha \downarrow 0$).

</div>

### Applications to Optimal Transport

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.5</span><span class="math-callout__name">(Transport from Intermediate Times is Locally Lipschitz)</span></p>

On a Riemannian manifold $M$, let $c$ be a cost function satisfying the assumptions of Corollary 8.2, let $K$ be a compact subset of $M$, and let $\Pi$ be a dynamical optimal transport supported in $K$. Then $\Pi$ is supported on a set of geodesics $S$ such that for any two $\gamma, \widetilde{\gamma} \in S$,

$$\sup_{0 \le t \le 1} d\bigl(\gamma(t), \widetilde{\gamma}(t)\bigr) \le C_K(t_0)\, d\bigl(\gamma(t_0), \widetilde{\gamma}(t_0)\bigr).$$

In particular, if $(\mu\_t)\_{0 \le t \le 1}$ is a displacement interpolation between any two compactly supported probability measures on $M$, and $t\_0 \in (0,1)$ is given, then for any $t \in [0,1]$ the map

$$T_{t_0 \to t} : \gamma(t_0) \longmapsto \gamma(t)$$

is well-defined $\mu\_{t\_0}$-almost surely and **Lipschitz continuous** on its domain; and it is in fact the **unique** solution of the Monge problem between $\mu\_{t\_0}$ and $\mu\_t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Singular Initial Time)</span></p>

The map $\gamma(0) \to \gamma(t)$ (from initial to intermediate time) need not be well-defined: starting from a Dirac mass $\mu\_0 = \delta\_{x\_0}$, the displacement interpolation is $\mu\_t = \mathrm{law}(t X)$, and the map $\gamma(0) \to \gamma(1/2)$ is not well-defined since all geodesics start at the same point $x\_0$. But $\gamma(1/2) \to \gamma(t)$ is always well-defined and Lipschitz for any $t\_0 \in (0,1)$.

</div>

### Preservation of Absolute Continuity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.7</span><span class="math-callout__name">(Absolute Continuity of Displacement Interpolation)</span></p>

Let $M$ be a Riemannian manifold, and let $L(x,v,t)$ be a $C^2$ Lagrangian on $TM \times [0,1]$, satisfying the classical conditions of Definition 7.6, with $\nabla\_v^2 L > 0$; let $c$ be the associated cost function. Let $\mu\_0$ and $\mu\_1$ be two probability measures on $M$ such that the optimal cost $C(\mu\_0, \mu\_1)$ is finite, and let $(\mu\_t)\_{0 \le t \le 1}$ be a displacement interpolation between $\mu\_0$ and $\mu\_1$. If either $\mu\_0$ or $\mu\_1$ is **absolutely continuous** with respect to the volume on $M$, then also $\mu\_t$ is absolutely continuous for all $t \in (0,1)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Idea of Theorem 8.7)</span></p>

Assume $\mu\_1$ is absolutely continuous. For compactly supported measures, Theorem 8.5 gives a Lipschitz map $T$ solving the Monge problem between $\mu\_{t\_0}$ and $\mu\_1$. If $N$ is a set of zero volume, then $\mu\_{t\_0}[N] \le \mu\_{t\_0}[T^{-1}(T(N))] = (T\_\# \mu\_{t\_0})[T(N)] = \mu\_1[T(N)]$, and $\mathrm{vol}[T(N)] \le \lVert T \rVert\_{\mathrm{Lip}}^n\, \mathrm{vol}[N] = 0$, so $\mu\_{t\_0}[N] = 0$ since $\mu\_1$ is absolutely continuous. The general case (not compactly supported) follows from a restriction argument using Theorem 7.30(ii).

</div>

### Proof of Mather's Estimates

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Idea of the Proof of Theorem 8.1)</span></p>

Assume $\gamma\_1$ and $\gamma\_2$ cross at a point $m\_0$ at time $t\_0$. Close to $m\_0$, these two curves look like two straight lines crossing each other, with respective velocities $v\_1$ and $v\_2$. Cut the curves on the time-interval $[t\_0 - \tau, t\_0 + \tau]$ and introduce "deviations" (shortcuts) that join the first curve to the second and vice versa (like a plumber installing a bypass). This amounts to replacing (on a short interval) two curves with approximate velocities $v\_1$ and $v\_2$ by two curves with approximate velocities $(v\_1 + v\_2)/2$. By strict convexity of the Lagrangian, the modification in action is approximately

$$(2\tau)\left(2L\!\left(m_0, \frac{v_1 + v_2}{2}, t_0\right) - \bigl[L(m_0, v_1, t_0) + L(m_0, v_2, t_0)\bigr]\right),$$

which is negative if $v\_1 \ne v\_2$, contradicting optimality. So $v\_1 = v\_2$, and since the curves satisfy the same second-order ODE with the same initial conditions, they must coincide. Making this quantitative yields the bound on $\lvert V\_1 - V\_2 \rvert$ in terms of $\lvert X\_1 - X\_2 \rvert$, and then Cauchy–Lipschitz propagation gives the full estimate.

</div>

### Complement: Ruling Out Focalization by Shortening

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cut Locus and Focal Points)</span></p>

Let $\gamma$ be a minimizing geodesic on a Riemannian manifold $M$, and let $t\_c$ be the largest time such that $\gamma$ restricted to $[0, t]$ is minimizing for all $t < t\_c$. The point $\gamma(t\_c)$ is said to be a **cut point** of $\gamma\_0$ along $\gamma$. The set of all cut points, as the geodesic varies, constitutes the **cut locus** of $x\_0$.

Two points $x\_0$ and $x'$ are said to be **focal** (or **conjugate**) if $x' = \exp\_{x\_0}(t' v\_0)$ and the differential $d\_{v\_0} \exp\_{x\_0}(t' \cdot)$ is *not invertible*. Intuitively, this means trajectories have a tendency to "concentrate" at time $t'$ along certain preferred directions.

The cut locus of a point $x$ can be separated into two sets: (a) those points $y$ joined to $x$ by at least two distinct minimizing geodesics; (b) those points $y$ joined by a unique minimizing geodesic, but which are focal points.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem 8.8</span><span class="math-callout__name">(Focalization is Impossible Before the Cut Locus)</span></p>

Under the assumptions of Corollary 8.2 with $L(x,v,t) = \lvert v \rvert^2$, let $\gamma : [0,1] \to M$ be a minimizing geodesic starting from $x\_0$. Then starting from $x\_0$, **focalization is impossible** at $\gamma(t\_\ast)$ if $0 < t\_\ast < 1$ (i.e. before the cut locus). In other words, the differential $d\_{v\_0} X\_t(x\_0, \cdot)$ is invertible, and its inverse is of size $O((1 - t\_\ast)^{-1})$ as a function of $t\_\ast$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.9</span><span class="math-callout__name">(Cut Locus and Focal Points on $S^2$)</span></p>

On the sphere $S^2$, the north pole $N$ has only one cut point (the south pole $S$), which is also its only focal point. If one deforms the sphere in a neighborhood of $\gamma[0,1]$ so as to create a shortcut from $N$ to $\gamma(1/2)$, then $S$ will no longer be a cut point along $\gamma$ (though it may still be a cut point along some other geodesic). On the other hand, $S$ will still be the only focal point along $\gamma$.

</div>

### Introduction to Mather's Theory

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.11</span><span class="math-callout__name">(Lipschitz Graph Theorem)</span></p>

Let $M$ be a compact Riemannian manifold, let $L = L(x,v,t)$ be a Lagrangian function on $TM \times \mathbb{R}$, and $T > 0$, such that

* (a) $L$ is $T$-periodic in the $t$ variable, i.e. $L(x,v,t+T) = L(x,v,t)$;
* (b) $L$ is of class $C^2$ in all variables;
* (c) $\nabla\_v^2 L$ is (strictly) positive everywhere, and $L$ is superlinear in $v$.

Define the action by $\mathcal{A}^{s,t}(\gamma) = \int\_s^t L(\gamma\_\tau, \dot{\gamma}\_\tau, \tau)\, d\tau$, let $c^{s,t}$ be the associated cost function on $M \times M$, and $C^{s,t}$ the corresponding optimal cost functional on $P(M) \times P(M)$. Let $\overline{\mu}$ be a probability measure solving the minimization problem

$$\inf_{\mu \in P(\mathcal{X})} C^{0,T}(\mu, \mu),$$

and let $(\mu\_t)\_{0 \le t \le T}$ be a displacement interpolation between $\mu\_0 = \overline{\mu}$ and $\mu\_T = \overline{\mu}$. Extend $(\mu\_t)$ into a $T$-periodic curve $\mathbb{R} \to P(M)$ defined for all times. Then

* (i) For all $t\_0 < t\_1$, the curve $(\mu\_t)\_{t\_0 \le t \le t\_1}$ still defines a displacement interpolation;
* (ii) The optimal transport cost $C^{t, t+T}(\mu\_t, \mu\_t)$ is independent of $t$;
* (iii) For any $t\_0 \in \mathbb{R}$, and for any $k \in \mathbb{N}$, $\mu\_{t\_0}$ is a minimizer for $C^{t\_0, t\_0 + kT}(\mu, \mu)$.

Moreover, there is a random curve $(\gamma\_t)\_{t \in \mathbb{R}}$ such that

* (iv) For all $t \in \mathbb{R}$, $\mathrm{law}(\gamma\_t) = \mu\_t$;
* (v) For any $t\_0 < t\_1$, the curve $(\gamma\_t)\_{t\_0 \le t \le t\_1}$ is action-minimizing;
* (vi) The map $\gamma\_0 \to \dot{\gamma}\_0$ is well-defined and **Lipschitz**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to Mather's Theory)</span></p>

Theorem 8.11 is a reformulation, in the language of optimal transport, of Mather's celebrated result on the existence of action-minimizing measures in Lagrangian dynamics. The property (vi) states that the support of $\overline{\mu}$ is a **Lipschitz graph** in phase space $TM$ (hence the name). Mather's measures are probability measures $\overline{\mu}$ on $TM$ that minimize the average action among all invariant probability measures. The shortening principle is the key tool to prove the Lipschitz graph property.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.13</span><span class="math-callout__name">(Stationary Measures)</span></p>

If $L$ does not depend on $t$, then one can apply Theorem 8.11 for any $T = 2^{-\ell}$, and then use a compactness argument to construct a constant curve $(\mu\_t)\_{t \in \mathbb{R}}$ satisfying Properties (i)–(vi). In particular, $\mu\_0$ is a **stationary measure** for the Lagrangian system.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.14</span></p>

Let $M$ be a compact Riemannian manifold, and let $L(x,v,t) = \lvert v \rvert^2/2 - V(x)$, where $V$ has a unique maximum $x\_0$. Then Mather's procedure selects the probability measure $\delta\_{x\_0}$ and the stationary curve $\gamma \equiv x\_0$ (which is an unstable equilibrium).

</div>

### Mather Critical Value, Mather Set, and Aubry Set

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.15</span><span class="math-callout__name">(Useful Transport Quantities Describing a Lagrangian System)</span></p>

For each displacement interpolation $(\mu\_t)\_{t \ge 0}$ as in Theorem 8.11, define:

* (i) The **Mather critical value** as the opposite of the mean optimal transport cost:

$$-M = \overline{c} := \frac{1}{T}\, C^{0,T}(\mu, \mu) = \frac{1}{kT}\, C^{0,kT}(\mu, \mu).$$

* (ii) The **Mather set** as the closure of the union of all supports $V\_\# \mu\_0$, where $(\mu\_t)\_{t \ge 0}$ is a displacement interpolation as in Theorem 8.11 and $V$ is the Lipschitz map $\gamma\_0 \to (\gamma\_0, \dot{\gamma}\_0)$.
* (iii) The **Aubry set** as the set of all $(\gamma\_0, \dot{\gamma}\_0)$ such that there is a solution $(\phi, \psi)$ of the dual problem (8.36) satisfying $H\_+^{0,T} \psi(\gamma\_1) - \psi(\gamma\_0) = c^{0,T}(\gamma\_0, \gamma\_1)$.

Up to the change of variables $(\gamma\_0, \dot{\gamma}\_0) \to (\gamma\_0, \gamma\_1)$, the Mather and Aubry sets are just the same as $\Gamma\_{\min}$ and $\Gamma\_{\max}$ appearing in the bibliographical notes of Chapter 5.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.16</span><span class="math-callout__name">(One-Dimensional Pendulum)</span></p>

For a one-dimensional pendulum: for small values of the total energy, the pendulum oscillates with small periodic motions (arcs of circle in physical space). For large values, it describes complete revolutions. At a critical intermediate energy, the trajectory consists in going from the vertical upward position (at time $-\infty$) to the vertical upward position again (at time $+\infty$), exploring all intermediate angles — these are "revolutions of infinite period", and they are globally action-minimizing. When the rotation number $\xi = 0$, the Mather problem selects $\delta\_{x\_0}$ (the unstable equilibrium), and the Mather and Aubry sets are reduced to $\lbrace (x\_0, x\_0) \rbrace$. At a certain critical value of $\xi$, the Mather and Aubry sets differ: the Aubry set (viewed in $(x,v)$ variables) is the union of the two revolutions of infinite period.

</div>

### Connection to Weak KAM Theory

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.17</span><span class="math-callout__name">(Mather Critical Value and Stationary Hamilton–Jacobi Equation)</span></p>

With the same notation as in Theorem 8.11, assume that the Lagrangian $L$ does not depend on $t$, and let $\psi$ be a Lipschitz function on $M$ such that $H\_+^{0,t} \psi = \psi + c\, t$ for all times $t \ge 0$; that is, $\psi$ is left invariant by the forward Hamilton–Jacobi semigroup, except for the addition of a constant which varies linearly in time. Then, necessarily $c = \overline{c}$, and the pair $(\psi, H\_+^{0,T} \psi) = (\psi, \psi + \overline{c}\, T)$ is optimal in the dual Kantorovich problem with cost function $c^{0,T}$, and initial and final measures equal to $\overline{\mu}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.18</span></p>

The equation $H\_+^{0,1} \psi = \psi + c\, t$ is a way to reformulate the stationary Hamilton–Jacobi equation $H(x, \nabla \psi(x)) + c = 0$. Theorem 8.17 does not guarantee the *existence* of such stationary solutions; it just states that *if* they exist, then the constant $c$ is uniquely determined and can be related to a Monge–Kantorovich problem. In weak KAM theory, one then establishes the existence by independent means.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.19</span></p>

The constant $-\overline{c}$ (which coincides with Mather's critical value) is often called the **effective Hamiltonian** of the system.

</div>

### Possible Extensions of Mather's Estimates

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sectional Curvature and Alexandrov Spaces)</span></p>

Mather's estimates are related to the local behavior of geodesics and to the convexity properties of the square distance function $d^2(x\_0, \cdot)$. Both features are captured by lower bounds on the **sectional curvature** of the manifold. There is a generalized notion of sectional curvature bounds due to Alexandrov, which makes sense in a general metric space without smoothness. Metric spaces satisfying these bounds are called **Alexandrov spaces**, and it is natural to conjecture that Mather's shortening lemma extends to this setting.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Open Problem 8.21</span></p>

Let $(\mathcal{X}, d)$ be an Alexandrov space with curvature bounded below by $K \in \mathbb{R}$, and let $x\_1, x\_2, y\_1, y\_2$ be four points in $\mathcal{X}$ such that

$$d(x_1, y_1)^2 + d(x_2, y_2)^2 \le d(x_1, y_2)^2 + d(x_2, y_1)^2.$$

Let $\gamma\_1$ and $\gamma\_2$ be two constant-speed geodesics respectively joining $x\_1$ to $y\_1$ and $x\_2$ to $y\_2$. Then, for any $t\_0 \in (0,1)$, there is a constant $C\_{t\_0}$ depending only on $K$, $t\_0$, and an upper bound on all the distances involved, such that

$$\sup_{0 \le t \le 1} d\bigl(\gamma_1(t), \gamma_2(t)\bigr) \le C_{t_0}\, d\bigl(\gamma_1(t_0), \gamma_2(t_0)\bigr).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.22</span><span class="math-callout__name">(A Rough Nonsmooth Shortening Lemma)</span></p>

Let $(\mathcal{X}, d)$ be a metric space, and let $\gamma\_1, \gamma\_2$ be two constant-speed, minimizing geodesics such that

$$d\bigl(\gamma_1(0), \gamma_1(1)\bigr)^2 + d\bigl(\gamma_2(0), \gamma_2(1)\bigr)^2 \le d\bigl(\gamma_1(0), \gamma_2(1)\bigr)^2 + d\bigl(\gamma_2(0), \gamma_1(1)\bigr)^2.$$

Let $L\_1$ and $L\_2$ stand for the respective lengths of $\gamma\_1$ and $\gamma\_2$, and let $D$ be a bound on the diameter of $(\gamma\_1 \cup \gamma\_2)([0,1])$. Then

$$\lvert L_1 - L_2 \rvert \le \frac{C \sqrt{D}}{\sqrt{t_0(1-t_0)}}\, \sqrt{d\bigl(\gamma_1(t_0), \gamma_2(t_0)\bigr)},$$

for some numeric constant $C$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of Theorem 8.22)</span></p>

In a general metric space without curvature bounds, there may be branching geodesics, so a bound on the distance at one intermediate time cannot control the distance at all times. But one can still exploit the constant-speed property: if geodesics in a displacement interpolation pass near each other at some intermediate time, then their **lengths** have to be approximately equal. This is a much rougher statement than the smooth version, but it holds in complete generality.

</div>

### Appendix: Lipschitz Estimates for Power Cost Functions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.23</span><span class="math-callout__name">(Shortening Lemma for Power Cost Functions)</span></p>

Let $\alpha \in (0,1)$, and let $x\_1, y\_1, x\_2, y\_2$ be four points in $\mathbb{R}^n$ such that

$$\lvert x_1 - y_1 \rvert^{1+\alpha} + \lvert x_2 - y_2 \rvert^{1+\alpha} \le \lvert x_1 - y_2 \rvert^{1+\alpha} + \lvert x_2 - y_1 \rvert^{1+\alpha}.$$

Let $\gamma\_1(t) = (1-t)x\_1 + t y\_1$, $\gamma\_2(t) = (1-t)x\_2 + t y\_2$. Then, for any $t\_0 \in (0,1)$ there is a constant $K = K(\alpha, t\_0) > 0$ such that

$$\lvert \gamma_1(t_0) - \gamma_2(t_0) \rvert \ge K \sup_{0 \le t \le 1} \lvert \gamma_1(t) - \gamma_2(t) \rvert.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.24</span></p>

The constant $K(\alpha, t)$ goes to 0 as $\alpha \downarrow 0$. When $\alpha = 0$ (i.e. the distance cost $c(x,y) = \lvert x - y \rvert$), the conclusion is false: just think of the case when $x\_1, y\_1, x\_2, y\_2$ are aligned. But this is the only case in which the conclusion fails, so it might be that a modified statement still holds true.

</div>

### Bibliographical Notes on Chapter 8

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History of the Shortening Principle)</span></p>

Monge's observation about the impossibility of crossing appears in his seminal 1781 memoir. It applies whenever the cost function satisfies a triangle inequality (this is called the **Monge–Mañé problem** by Bernard and Buffoni).

At the end of the seventies, **Aubry** discovered a noncrossing lemma in a different setting (Frenkel–Kantorova model from solid-state physics); together with Le Daeron, he demonstrated the power of this principle. Their method provided an alternative proof of **Mather's** results about quasiperiodic orbits. The relations between the methods of Aubry and Mather constitute the **Aubry–Mather theory**. **Moser** showed that the theory of twist diffeomorphisms could be embedded in strictly convex Lagrangian systems.

Around 1990, **Mather** made two crucial contributions: (1) introducing minimizing *measures* rather than minimizing curves; (2) a *quantitative* version of the noncrossing argument for a general class of strictly convex Lagrangian functions — what is called here **Mather's shortening lemma**, the key ingredient in his fundamental **Lipschitz graph theorem**.

The noncrossing property was independently rediscovered in optimal transport by **McCann** (qualitative version) for the quadratic cost. Quantitative results about absolute continuity of the displacement interpolant were generalized by **Cordero-Erausquin, McCann and Schmuckenschläger** for Riemannian manifolds.

Results similar to Theorems 8.5 and 8.7 are also proven by **Bernard and Buffoni** via Hamilton–Jacobi equations, exploiting the automatic semiconcavity of solutions for positive times. **Figalli and Juillet** obtained a result similar to Theorem 8.7 on degenerate Riemannian structures such as the Heisenberg group or Alexandrov spaces, using the uniqueness of Wasserstein geodesics and the measure contraction property — notably the Monge–Mather shortening lemma does *not* hold in this setting.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weak KAM Theory)</span></p>

The acronym "KAM" stands for **Kolmogorov, Arnold and Moser**. Classical KAM theory deals with stability of perturbed integrable Hamiltonian systems. **Weak KAM theory** is much more recent, developed in particular by **Fathi**. The existence of a stationary solution of the Hamilton–Jacobi equation (Theorem 8.17) can be found in Fathi's book. From its very beginning, weak KAM theory has been associated with viscosity solutions of Hamilton–Jacobi equations. Aubry sets are also related to the $C^1$ regularity of Hamilton–Jacobi equations.

There is an alternative presentation of Mather's problem due to **Mañé**: the unknown is a probability measure $\mu(dx\, dv)$ on the tangent bundle $TM$, stationary in the sense that $\nabla\_x \cdot (v\, \mu) = 0$ (a stationary kinetic transport equation), and minimizing the action $\int L(x, v)\, \mu(dx\, dv)$.

</div>

## Chapter 9: Solution of the Monge Problem I — Global Approach

This chapter investigates the solvability of the Monge problem for a Lagrangian cost function. Recall from Theorem 5.30 that it is sufficient to identify conditions under which the initial measure $\mu$ does not see the set of points where the $c$-subdifferential of a $c$-convex function $\psi$ is multivalued. The approach here uses Assumption (C) on the cost function and combines it with Mather's shortening principle; it works out for a particular class of cost functions including the quadratic cost in Euclidean space.

### Assumption (C): Connectedness of $c$-Subdifferentials

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption (C)</span></p>

For any $c$-convex function $\psi$ and any $x \in M$, the $c$-subdifferential $\partial\_c \psi(x)$ is **pathwise connected**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.1</span><span class="math-callout__name">($c(x,y) = -x \cdot y$)</span></p>

Consider the cost function $c(x,y) = -x \cdot y$ in $\mathbb{R}^n$. If $y\_0$ and $y\_1$ belong to $\partial\_c \psi(x)$, then $\psi(x) + y\_0 \cdot (z - x) \le \psi(z)$ and $\psi(x) + y\_1 \cdot (z - x) \le \psi(z)$ for all $z$. Setting $y\_t := (1-t)y\_0 + t y\_1$, the same holds for $y\_t$: the entire line segment $(y\_t)\_{0 \le t \le 1}$ lies in $\partial\_c \psi(x)$. The same computation applies to $c(x,y) = \lvert x - y \rvert^2 / 2$, or to any cost function of the form $a(x) - x \cdot y + b(y)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Known Examples Satisfying Assumption (C))</span></p>

There are not so many known examples satisfying Assumption (C). The short list includes:

* $c(x,y) = -x \cdot y$ (or $\lvert x - y \rvert^2/2$) on $\mathbb{R}^n \times \mathbb{R}^n$;
* $c(x,y) = \sqrt{1 + \lvert x - y \rvert^2}$ on $\mathbb{R}^n \times \mathbb{R}^n$, or more generally $c(x,y) = (1 + \lvert x - y \rvert^2)^{p/2}$ ($1 < p < 2$) on $B\_R(0) \times B\_R(0) \subset \mathbb{R}^n \times \mathbb{R}^n$, where $R = 1/\sqrt{p-1}$;
* $c(x,y) = d(x,y)^2$ on $S^{n-1} \times S^{n-1}$ (the squared geodesic distance on the sphere).

If $\mathcal{Y}$ is a nonconvex subset of $\mathbb{R}^n$, then Assumption (C) can be violated even by the quadratic cost function. Assumption (C) is *not* a generic condition, and its verification is subtle.

</div>

### Conditions for Single-Valued Subdifferentials

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.2</span><span class="math-callout__name">(Conditions for Single-Valued Subdifferentials)</span></p>

Let $M$ be a smooth $n$-dimensional Riemannian manifold, and $c$ a real-valued cost function, bounded below, deriving from a Lagrangian $L(x,v,t)$ on $TM \times [0,1]$ satisfying the classical conditions of Definition 7.6 and such that:

* (i) Assumption (C) is satisfied.
* (ii) The conclusion of Theorem 8.1 (Mather's shortening lemma) holds true for $t\_0 = 1/2$ with an exponent $\beta > 1 - (1/n)$, and a uniform constant. More explicitly: whenever $x\_1, x\_2, y\_1, y\_2$ are four points in $M$ satisfying $c(x\_1, y\_1) + c(x\_2, y\_2) \le c(x\_1, y\_2) + c(x\_2, y\_1)$, and $\gamma\_1$, $\gamma\_2$ are two action-minimizing curves with $\gamma\_1(0) = x\_1$, $\gamma\_1(1) = y\_1$, $\gamma\_2(0) = x\_2$, $\gamma\_2(1) = y\_2$, then

$$\sup_{0 \le t \le 1} d\bigl(\gamma_1(t), \gamma_2(t)\bigr) \le C\, d\bigl(\gamma_1(1/2), \gamma_2(1/2)\bigr)^\beta.$$

Then, for any $c$-convex function $\psi$, there is a set $Z \subset M$ of **Hausdorff dimension at most $(n-1)/\beta < n$** (and therefore of zero $n$-dimensional measure), such that the $c$-subdifferential $\partial\_c \psi(x)$ contains at most one element if $x \notin Z$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Idea of Theorem 9.2)</span></p>

The key idea is to use Mather's shortening lemma to show that the map $F : \gamma(1/2) \longmapsto x = \gamma(0)$ (from midpoints of action-minimizing curves to initial points) is well-defined and **Hölder-$\beta$** continuous. If $\partial\_c \psi(x)$ contains two distinct elements $y\_0, y\_1$, then by Assumption (C) there is a continuous path $(y\_t)\_{0 \le t \le 1}$ in $\partial\_c \psi(x)$. This produces a nontrivial continuous path of midpoints $(m\_t) = (\gamma\_t(1/2))$ which must intersect a countable set $D$ of codimension $n-1$ (hyperplane sections). Since $F(m\_t) = x$ for all $t$, we get $x \in F(D)$. Since $F$ is $\beta$-Hölder, $\dim\_H F(D) \le (n-1)/\beta$, so $Z \subset F(D)$ has the required small dimension.

</div>

### Solution of the Monge Problem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.3</span><span class="math-callout__name">(Solution of the Monge Problem, I)</span></p>

Let $M$ be a Riemannian manifold, let $c$ be a cost function on $M \times M$, with associated cost functional $C$, and let $\mu, \nu$ be two probability measures on $M$. Assume that:

* (i) $C(\mu, \nu) < +\infty$;
* (ii) the assumptions of Theorem 9.2 are satisfied;
* (iii) $\mu$ gives zero probability to sets of dimension at most $(n-1)/\beta$.

Then there is a **unique** (in law) optimal coupling $(x, y)$ of $\mu$ and $\nu$; it is **deterministic**, and characterized (among all couplings of $(\mu, \nu)$) by the existence of a $c$-convex function $\psi$ such that

$$y \in \partial_c \psi(x) \qquad \text{almost surely.}$$

Equivalently, there is a unique optimal transport plan $\pi$; it is deterministic, and characterized by the existence of a $c$-convex $\psi$ such that $\mathrm{Spt}\, \pi \subset \partial\_c \psi$.

</div>

### The Quadratic Cost: First Result

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.4</span><span class="math-callout__name">(Monge Problem for Quadratic Cost, First Result)</span></p>

Let $c(x,y) = \lvert x - y \rvert^2$ in $\mathbb{R}^n$. Let $\mu$, $\nu$ be two probability measures on $\mathbb{R}^n$ such that

$$\int \lvert x \rvert^2\, d\mu(x) + \int \lvert y \rvert^2\, d\nu(y) < +\infty$$

and $\mu$ does not give mass to sets of dimension at most $n - 1$. (This is true in particular if $\mu$ is absolutely continuous with respect to the Lebesgue measure.) Then there is a **unique** (in law) optimal coupling $(x, y)$ of $\mu$ and $\nu$; it is **deterministic**, and characterized, among all couplings of $(\mu, \nu)$, by the existence of a lower semicontinuous convex function $\psi$ such that

$$y \in \partial \psi(x) \qquad \text{almost surely.}$$

In other words, there is a unique optimal transference plan $\pi$; it is a Monge transport plan, and it is characterized by the existence of a lower semicontinuous convex function $\psi$ whose subdifferential contains $\mathrm{Spt}\, \pi$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reduction to $c(x,y) = -x \cdot y$)</span></p>

For the quadratic cost, $c$-convexity reduces to plain convexity (plus lower semicontinuity), and the $c$-subdifferential is just the usual subdifferential $\partial \psi$. Under the assumption of finite second moments, the optimal transport cost for $c(x,y) = \lvert x - y \rvert^2$ is the same as for $c(x,y) = -x \cdot y$ up to the additive constant $\int (\lvert x \rvert^2 + \lvert y \rvert^2)\, d\pi$, which is independent of $\pi$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.5</span><span class="math-callout__name">(Optimality of the Dimension Assumption)</span></p>

The assumption that $\mu$ does not give mass to sets of dimension at most $n-1$ is optimal for the existence of a Monge coupling. Example: $\mu = \mathcal{H}^1\rvert\_{[0,1] \times \lbrace 0 \rbrace}$ (one-dimensional Hausdorff measure on a segment in $\mathbb{R}^2$) and $\nu = (1/2)\, \mathcal{H}^1\rvert\_{[0,1] \times \lbrace -1, +1 \rbrace}$. Then there is a unique optimal coupling, but it is *not* a Monge coupling (mass must be split). The dimension assumption is also optimal for uniqueness: if $\mu$ and $\nu$ are supported on orthogonal subspaces of $\mathbb{R}^n$, then *any* transference plan is optimal.

</div>

### Limitations and Non-Connectedness

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Limitations of the Global Approach)</span></p>

The approach via Assumption (C) suffers from two main drawbacks:

1. It seems to be limited to a small number of examples of cost functions.
2. The verification of Assumption (C) is subtle.

The next chapter will investigate a more pedestrian, local approach which applies in much greater generality.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.6</span><span class="math-callout__name">(Non-Connectedness of the $c$-Subdifferential)</span></p>

Let $p > 2$ and let $c(x,y) = \lvert x - y \rvert^p$ on $\mathbb{R}^2 \times \mathbb{R}^2$. Then there is a $c$-convex function $\psi : \mathbb{R}^2 \to \mathbb{R}$ such that $\partial\_c \psi(0)$ is **not connected**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of Proposition 9.6)</span></p>

This shows that Assumption (C) fails for the cost $c(x,y) = \lvert x - y \rvert^p$ when $p > 2$. The construction works by "surelevating" a function at the origin to force $0 \notin \partial\_c \psi(0)$, thereby disconnecting the subdifferential. This is an obstruction specific to exponents $p > 2$; for $1 < p \le 2$ the story is different (and more delicate).

</div>

### Bibliographical Notes on Chapter 9

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History of Theorem 9.4)</span></p>

The paternity of Theorem 9.4 is shared by **Brenier** and **Rachev–Rüschendorf**; it builds upon earlier work by **Knott and Smith**, who already knew that an optimal coupling lying entirely in the subdifferential of a convex function would be optimal. **Brenier** rewrote the result as a beautiful **polar factorization theorem**: any vector-valued $L^2$ function can be uniquely decomposed as the composition of the gradient of a convex function with a measure-preserving map.

**McCann** extended Theorem 9.4 by removing the assumption of bounded second moments and even the assumption of finite transport cost: *whenever $\mu$ does not charge sets of dimension $n-1$, there exists a unique coupling of $(\mu, \nu)$ which takes the form $y = \nabla \Psi(x)$, where $\Psi$ is a lower semicontinuous convex function*. The tricky part is uniqueness; this will be proven in the next chapter (Theorem 10.42, Corollary 10.44).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Assumption (C) and the Ma–Trudinger–Wang Condition)</span></p>

**Ma, Trudinger and X.-J. Wang** were the first to seriously study Assumption (C); they had the intuition that it was connected to a certain **fourth-order differential condition** on the cost function which plays a key role in the *smoothness* of optimal transport. Later **Trudinger and Wang**, and **Loeper**, showed that the above-mentioned differential condition is, under adequate geometric and regularity assumptions, essentially equivalent to Assumption (C). These issues will be discussed in more detail in Chapter 12.

**Loeper** discovered that the squared geodesic distance on $S^{n-1}$ satisfies Assumption (C); a simplified argument was devised by **von Nessi**. By combining this with Theorems 8.1, 9.2 and 5.30, one obtains the unique solvability of the Monge problem for the quadratic distance on the sphere.

</div>

## Chapter 10: Solution of the Monge Problem II — Local Approach

In the previous chapter, the solvability of the Monge problem was established via a "global" topological argument (connectedness of $c$-subdifferentials). Since this strategy works only in certain particular cases, this chapter explores a different method based on **local** properties of $c$-convex functions. The key insight is that the global question "*Is the $c$-subdifferential of $\psi$ at $x$ single-valued?*" can often be replaced by the more tractable local question "*Is $\psi$ differentiable at $x$?*" The emphasis is on tangent vectors and gradients rather than on points in the $c$-subdifferential.

### A Heuristic Argument

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Twist Condition)</span></p>

If $\psi$ is a $c$-convex function and $y \in \partial\_c \psi(x)$, then (from the definition of $c$-subdifferential) $\psi(x) - \psi(\widetilde{x}) \le c(\widetilde{x}, y) - c(x, y)$ for all $\widetilde{x}$. If both $\psi$ and $c(\cdot, y)$ are differentiable at $x$, passing to the limit as $\widetilde{x} \to x$ gives:

$$\nabla \psi(x) + \nabla_x c(x, y) = 0.$$

If $x$ is given, this is an equation for $y$. If the map $y \longmapsto \nabla\_x c(x, y)$ is **injective**, then this equation has at most one solution, and $y$ is uniquely determined. This injectivity property is a classical condition in dynamical systems, sometimes referred to as a **twist condition**.

So the strategy is: (1) show that $\psi$ is differentiable at $x$ (which is a local regularity question); (2) use the twist condition to deduce that $\partial\_c \psi(x)$ contains at most one element. Three potential objections: (a) $\psi$ might not be differentiable; (b) injectivity of $\nabla\_x c$ might be hard to check; (c) $c(x, y)$ might not be differentiable at $(x, y)$. All three will be addressed using nonsmooth analysis.

</div>

### Differentiability and Approximate Differentiability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.1</span><span class="math-callout__name">(Differentiability)</span></p>

Let $U \subset \mathbb{R}^n$ be an open set. A function $f : U \to \mathbb{R}$ is said to be **differentiable** at $x \in U$ if there exists a vector $p \in \mathbb{R}^n$ such that

$$f(z) = f(x) + \langle p, z - x \rangle + o(\lvert z - x \rvert) \qquad \text{as } z \to x.$$

The vector $p$ is uniquely determined; it is the **gradient** $\nabla f(x)$. On a Riemannian manifold $M$, the same definition applies with $p \in T\_x M$ and $f(\exp\_x w) = f(x) + \langle p, w \rangle + o(\lvert w \rvert)$ as $w \to 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.2</span><span class="math-callout__name">(Approximate Differentiability)</span></p>

Let $U$ be an open set of a smooth Riemannian manifold $M$, and let $f : U \to \mathbb{R} \cup \lbrace \pm\infty \rbrace$ be a measurable function. Then $f$ is said to be **approximately differentiable** at $x \in U$ if there is a measurable function $\widetilde{f} : U \to \mathbb{R}$, differentiable at $x$, such that the set $\lbrace \widetilde{f} = f \rbrace$ has **density 1** at $x$:

$$\lim_{r \to 0} \frac{\mathrm{vol}\bigl[\lbrace z \in B_r(x);\; f(z) = \widetilde{f}(z) \rbrace\bigr]}{\mathrm{vol}[B_r(x)]} = 1.$$

Then the **approximate gradient** $\widetilde{\nabla} f(x) := \nabla \widetilde{f}(x)$ is well-defined (independent of the choice of $\widetilde{f}$).

</div>

### Regularity in a Nonsmooth World

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.3</span><span class="math-callout__name">(Lipschitz Continuity)</span></p>

A function $f : U \to \mathbb{R}$ (where $U \subset \mathbb{R}^n$ is open) is **Lipschitz** if there exists $L < \infty$ such that $\lvert f(z) - f(x) \rvert \le L \lvert z - x \rvert$ for all $x, z \in U$. It is **locally Lipschitz** if for any $x\_0 \in U$, there is a neighborhood of $x\_0$ in which $f$ is Lipschitz. On a Riemannian manifold, $f$ is locally Lipschitz if it is Lipschitz on any compact subset of $U$, equipped with the geodesic distance.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.5</span><span class="math-callout__name">(Subdifferentiability, Superdifferentiability)</span></p>

Let $U$ be an open subset of $\mathbb{R}^n$, and $f : U \to \mathbb{R}$ a function.

* (i) $f$ is said to be **subdifferentiable** at $x$, with subgradient $p$, if

$$f(z) \ge f(x) + \langle p, z - x \rangle + o(\lvert z - x \rvert).$$

The convex set of all subgradients at $x$ is denoted by $\nabla^- f(x)$.

* (ii) $f$ is said to be **uniformly subdifferentiable** in $U$ if there is a continuous function $\omega : \mathbb{R}\_+ \to \mathbb{R}\_+$, with $\omega(r) = o(r)$ as $r \to 0$, and for all $x \in U$ there exists $p \in \mathbb{R}^n$ such that

$$f(z) \ge f(x) + \langle p, z - x \rangle - \omega(\lvert z - x \rvert).$$

* (iii) $f$ is **locally (uniformly) subdifferentiable** if each point admits a neighborhood on which $f$ is uniformly subdifferentiable.

Corresponding notions of **superdifferentiability** and **supergradients** are obtained by reversing the inequality signs. The convex set of supergradients at $x$ is $\nabla^+ f(x)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10.7</span><span class="math-callout__name">(Sub- and Superdifferentiability Imply Differentiability)</span></p>

Let $U$ be an open set of a smooth Riemannian manifold $M$, and let $f : U \to \mathbb{R}$ be a function. Then $f$ is differentiable at $x$ if and only if it is both subdifferentiable and superdifferentiable there; and then

$$\nabla^- f(x) = \nabla^+ f(x) = \lbrace \nabla f(x) \rbrace.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.8</span><span class="math-callout__name">(Regularity and Differentiability Almost Everywhere)</span></p>

Let $U$ be an open subset of a smooth Riemannian manifold $M$, and let $f : U \to \mathbb{R}$ be a function. Let $n$ be the dimension of $M$. Then:

* (i) If $f$ is continuous, then it is subdifferentiable on a dense subset of $U$, and also superdifferentiable on a dense subset of $U$.
* (ii) If $f$ is locally Lipschitz, then it is differentiable **almost everywhere** (with respect to the volume measure). This is **Rademacher's theorem**.
* (iii) If $f$ is locally subdifferentiable (resp. locally superdifferentiable), then it is locally Lipschitz and differentiable out of a countably $(n-1)$-**rectifiable** set. Moreover, the set of differentiability points coincides with the set of points where there is a unique subgradient (resp. supergradient). Finally, $\nabla f$ is continuous on its domain of definition.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.9</span></p>

Statement (ii) is known as **Rademacher's theorem**. The conclusion in statement (iii) is stronger than differentiability almost everywhere, since an $(n-1)$-rectifiable set has dimension $n-1$ and is therefore negligible. In fact, the local subdifferentiability property is stronger than the local Lipschitz property.

</div>

### Semiconvexity and Semiconcavity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.10</span><span class="math-callout__name">(Semiconvexity)</span></p>

Let $U$ be an open set of a smooth Riemannian manifold and let $\omega : \mathbb{R}\_+ \to \mathbb{R}\_+$ be continuous, with $\omega(r) = o(r)$ as $r \to 0$. A function $f : U \to \mathbb{R} \cup \lbrace +\infty \rbrace$ is said to be **semiconvex with modulus $\omega$** if, for any constant-speed geodesic path $(\gamma\_t)\_{0 \le t \le 1}$ whose image is included in $U$,

$$f(\gamma_t) \le (1-t) f(\gamma_0) + t f(\gamma_1) + t(1-t)\, \omega\bigl(d(\gamma_0, \gamma_1)\bigr).$$

It is **locally semiconvex** if for each $x\_0 \in U$ there is a neighborhood $V$ of $x\_0$ in $U$ such that the above holds for $\gamma\_0, \gamma\_1 \in V$. **Semiconcavity** and **local semiconcavity** are defined by reversing the inequality.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.11</span></p>

In $\mathbb{R}^n$, semiconvexity with modulus $\omega$ means $f\bigl((1-t)x + ty\bigr) \le (1-t)f(x) + tf(y) + t(1-t)\, \omega(\lvert x - y \rvert)$. When $\omega = 0$ this is plain convexity. When $\omega(r) = Cr^2/2$, there is a differential characterization: $f : \mathbb{R}^n \to \mathbb{R}$ is semiconvex with modulus $Cr^2/2$ if and only if $\nabla^2 f \ge -C I\_n$ (in the distributional sense).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10.12</span><span class="math-callout__name">(Local Equivalence of Semiconvexity and Subdifferentiability)</span></p>

Let $M$ be a smooth complete Riemannian manifold. Then:

* (i) If $\psi : M \to \mathbb{R} \cup \lbrace +\infty \rbrace$ is locally semiconvex, then it is locally subdifferentiable in the interior of its domain $D := \psi^{-1}(\mathbb{R})$; and $\partial D$ is countably $(n-1)$-rectifiable.
* (ii) Conversely, if $U$ is an open subset of $M$ and $\psi : U \to \mathbb{R}$ is locally subdifferentiable, then it is also locally semiconvex.

Similar statements hold with "subdifferentiable" replaced by "superdifferentiable" and "semiconvex" replaced by "semiconcave".

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.13</span></p>

Proposition 10.12 implies that *local semiconvexity* and *local subdifferentiability* are basically the same thing. But there is also a global version of semiconvexity. Since the concept is not invariant by diffeomorphism (unless it is an isometry), proofs on Riemannian manifolds require more care.

</div>

### Assumptions on the Cost Function

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Assumptions on the Cost Function)</span></p>

Let $M$ be a smooth complete connected Riemannian manifold, $\mathcal{X}$ a closed subset of $M$, $\mathcal{Y}$ an arbitrary Polish space, and $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ a continuous cost function. The following assumptions will be imposed on $c$ as a function of $x$, for $x$ in the interior of $\mathcal{X}$:

* **(Super)** $c(x,y)$ is everywhere superdifferentiable as a function of $x$, for all $y$.
* **(Twist)** On its domain of definition, $\nabla\_x c(x, \cdot)$ is injective: if $x, y, y'$ are such that $\nabla\_x c(x,y) = \nabla\_x c(x,y')$, then $y = y'$.
* **(Lip)** $c(x,y)$ is locally Lipschitz as a function of $x$, uniformly in $y$.
* **(SC)** $c(x,y)$ is locally semiconcave as a function of $x$, uniformly in $y$.
* **(locLip)** $c(x,y)$ is locally Lipschitz as a function of $x$, locally in $y$.
* **(locSC)** $c(x,y)$ is locally semiconcave as a function of $x$, locally in $y$.
* **(H$\infty$)$\_1$** For any $x$ and any measurable set $S$ whose tangent cone $T\_x S$ is not contained in a half-space, there is a finite collection $z\_1, \ldots, z\_k \in S$ and a small ball $B$ around $x$ such that for any $y$ outside a compact set, $\inf\_{w \in B} c(w,y) \ge \inf\_{1 \le j \le k} c(z\_j, y)$.
* **(H$\infty$)$\_2$** For any $x$ and any neighborhood $U$ of $x$ there is a small ball $B$ containing $x$ such that $\lim\_{y \to \infty} \sup\_{w \in B} \inf\_{z \in U} [c(z,y) - c(w,y)] = -\infty$.

Write **(H$\infty$)** for the combination of **(H$\infty$)$\_1$** and **(H$\infty$)$\_2$**. Note that **(locSC)** implies **(Super)**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10.15</span><span class="math-callout__name">(Properties of Lagrangian Cost Functions)</span></p>

On a smooth Riemannian manifold $M$, let $c(x,y)$ be a cost function associated with a $C^1$ Lagrangian $L(x,v,t)$. Assume that any $x, y \in M$ can be joined by at least one $C^1$ minimizing curve. Then:

* (i) For any $(x,y) \in M \times M$, and any $C^1$ minimizing curve $\gamma$ connecting $x$ to $y$, the tangent vector $-\nabla\_v L(x, \dot{\gamma}\_0, 0) \in T\_x M$ is a supergradient for $c(\cdot, y)$ at $x$; in particular, $c$ is superdifferentiable at $(x,y)$ as a function of $x$.
* (ii) If $L$ is strictly convex as a function of $v$, and minimizing curves are uniquely determined by their initial position and velocity, then $c$ satisfies a twist condition: if $c$ is differentiable at $(x,y)$ as a function of $x$, then $y$ is uniquely determined by $x$ and $\nabla\_x c(x,y)$. Moreover, $\nabla\_x c(x,y) + \nabla\_v L(x, \dot{\gamma}(0), 0) = 0$.
* (iii) If $L$ has the property that for any two compact sets $K\_0$ and $K\_1$, the velocities of minimizing curves starting in $K\_0$ and ending in $K\_1$ are uniformly bounded, then $c$ is locally Lipschitz and locally semiconcave as a function of $x$, locally in $y$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.16</span></p>

For $L(x,v,t) = \lvert v \rvert^2$, we have $\nabla\_v L = 2v$; so part (i) says that $-2v\_0$ is a supergradient of $d(\cdot, y)^2$ at $x$, where $v\_0$ is the velocity used to go from $x$ to $y$. In Euclidean space this gives $\nabla\_x(\lvert x - y \rvert^2) = 2(x - y) = -2(y - x)$. Part (ii) says that if $d(x,y)^2$ is differentiable as a function of $x$, then $x$ and $y$ are connected by a unique minimizing geodesic.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.20</span></p>

As a particular case, **(H$\infty$)$\_1$** holds true if $c = c(x-y)$ is radially symmetric and strictly increasing as a function of $\lvert x - y \rvert$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.22</span></p>

If $(M, g)$ is a Riemannian manifold with **nonnegative sectional curvature**, then $\nabla\_x^2 (d(x\_0, x)^2/2) \le g\_x$, and it follows that $c(x,y) = d(x,y)^2$ is semiconcave with a modulus $\omega(r) = r^2$. This condition is quite restrictive, but there does not seem to be any good alternative geometric condition implying the semiconcavity of $d(x,y)^2$ uniformly in $x$ and $y$.

</div>

### Differentiability of $c$-Convex Functions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.24</span><span class="math-callout__name">($c$-Subdifferentiability of $c$-Convex Functions)</span></p>

Let Assumption **(H$\infty$)** be satisfied. Let $\psi : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ be a $c$-convex function, and let $\Omega$ be the interior (in $M$) of its domain $\psi^{-1}(\mathbb{R})$. Then, $\psi^{-1}(\mathbb{R}) \setminus \Omega$ is a set of dimension at most $n-1$. Moreover, $\psi$ is locally bounded and $c$-subdifferentiable everywhere in $\Omega$. Finally, if $K \subset \Omega$ is compact, then $\partial\_c \psi(K)$ is itself compact.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.25</span><span class="math-callout__name">(Subdifferentiability of $c$-Convex Functions)</span></p>

Assume that **(Super)** is satisfied. Let $\psi$ be a $c$-convex function, and let $x$ be an interior point of $\mathcal{X}$ (in $M$) such that $\partial\_c \psi(x) \ne \emptyset$. Then $\psi$ is subdifferentiable at $x$. In short:

$$\partial_c \psi(x) \ne \emptyset \implies \nabla^- \psi(x) \ne \emptyset.$$

More precisely, for any $y \in \partial\_c \psi(x)$, one has $-\nabla\_x^+ c(x,y) \subset \nabla^- \psi(x)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.26</span><span class="math-callout__name">(Differentiability of $c$-Convex Functions)</span></p>

Assume that **(Super)** and **(Twist)** are satisfied, and let $\psi$ be a $c$-convex function. Then:

* (i) If **(Lip)** is satisfied, then $\psi$ is locally Lipschitz and differentiable in $\mathcal{X}$, apart from a set of zero volume; the same is true if **(locLip)** and **(H$\infty$)** are satisfied.
* (ii) If **(SC)** is satisfied, then $\psi$ is locally semiconvex and differentiable in the interior (in $M$) of its domain, apart from a set of dimension at most $n - 1$; and the boundary of the domain of $\psi$ is also of dimension at most $n-1$. The same is true if **(locSC)** and **(H$\infty$)** are satisfied.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.27</span><span class="math-callout__name">(Picture of Differentiability)</span></p>

Theorems 10.24–10.26, and (in the Lagrangian case) Proposition 10.15 provide a good picture of differentiability points of $c$-convex functions: Let $c$ satisfy **(Twist)**, **(Super)** and **(H$\infty$)**, and let $x$ be in the interior of the domain of a $c$-convex function $\psi$. If $\psi$ is differentiable at $x$ then $\partial\_c \psi(x)$ consists of just one point $y$, and $\nabla \psi(x) = -\nabla\_x c(x,y)$. In the Lagrangian case this also coincides with $\nabla\_v L(x, v, 0)$, where $v$ is the initial velocity of the unique action-minimizing curve joining $x$ to $y$. If $\psi$ is not differentiable at $x$, one can use the local semiconvexity of $\psi$ to show that $\nabla^- \psi(x)$ is included in the closed convex hull of $-\nabla\_x^+ c(x, \partial\_c \psi(x))$. There is in general no reason why $-\nabla\_x^+ c(x, \partial\_c \psi(x))$ would be convex; we shall come back to this issue in Chapter 12.

</div>

### Applications to the Monge Problem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.28</span><span class="math-callout__name">(Solution of the Monge Problem II)</span></p>

Let $M$ be a Riemannian manifold, $\mathcal{X}$ a closed subset of $M$, with $\dim(\partial \mathcal{X}) \le n-1$, and $\mathcal{Y}$ an arbitrary Polish space. Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be a continuous cost function, bounded below, and let $\mu \in P(\mathcal{X})$, $\nu \in P(\mathcal{Y})$, such that the optimal cost $C(\mu, \nu)$ is finite. Assume that:

* (i) $c$ is superdifferentiable everywhere (Assumption **(Super)**);
* (ii) $\nabla\_x c(x, \cdot)$ is injective where defined (Assumption **(Twist)**);
* (iii) any $c$-convex function is differentiable $\mu$-almost surely on its domain of $c$-subdifferentiability.

Then there exists a unique (in law) optimal coupling $(x, y)$ of $(\mu, \nu)$; it is **deterministic**, and there is a $c$-convex function $\psi$ such that

$$\nabla \psi(x) + \nabla_x c(x, y) = 0 \qquad \text{almost surely.}$$

In other words, there is a unique transport map $T$ solving the Monge problem, and $\nabla \psi(x) + \nabla\_x c(x, T(x)) = 0$, $\mu(dx)$-almost surely.

If moreover **(H$\infty$)** is satisfied, then:

* (a) Equation (10.20) **characterizes** the optimal coupling;
* (b) Let $Z$ be the set of points where $\psi$ is differentiable; then one can define a continuous map $x \to T(x)$ on $Z$ by the equation $T(x) \in \partial\_c \psi(x)$, and $\mathrm{Spt}\, \nu = \overline{T(\mathrm{Spt}\, \mu)}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.32</span></p>

If in Theorem 10.28 the cost $c$ derives from a $C^1$ Lagrangian $L(x,v,t)$, strictly convex in $v$, such that minimizing curves are uniquely determined by their initial velocity, then Proposition 10.15(ii) implies: *Almost surely, $x$ is joined to $y$ by a unique minimizing curve*. For instance, if $c(x,y) = d(x,y)^2$, the optimal transference plan $\pi$ will be concentrated on the set of $(x,y) \in M \times M$ such that $x$ and $y$ are joined by a unique geodesic.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.33</span><span class="math-callout__name">(How to Check Assumption (iii))</span></p>

Assumption (iii) can be realized in several ways: if $c$ is Lipschitz on $\mathcal{X} \times \mathcal{Y}$ and $\mu$ is absolutely continuous; or if $c$ is locally Lipschitz and $\mu, \nu$ are compactly supported and $\mu$ is absolutely continuous; or if $c$ is locally semiconcave and satisfies **(H$\infty$)** and $\mu$ does not charge sets of dimension $n-1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.34</span></p>

All the assumptions of Theorem 10.28 are satisfied if $\mathcal{X} = M = \mathcal{Y}$ is compact and the Lagrangian $L$ is $C^2$ and satisfies the classical conditions of Definition 7.6.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.35</span></p>

All the assumptions of Theorem 10.28 are satisfied if $\mathcal{X} = M = \mathcal{Y} = \mathbb{R}^n$, $c$ is a $C^1$ strictly convex function with a bounded Hessian and $\mu$ does not charge sets of dimension $n-1$. Indeed, $\nabla\_x c$ will be injective by strict convexity; and $c$ will be uniformly semiconcave with a modulus $Cr^2$, so Theorem 10.26 guarantees that $c$-convex functions are differentiable everywhere apart from a set of dimension at most $n-1$.

</div>

### Removing the Conditions at Infinity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.38</span><span class="math-callout__name">(Solution of the Monge Problem Without Conditions at Infinity)</span></p>

Let $M$ be a Riemannian manifold and $\mathcal{Y}$ an arbitrary Polish space. Let $c : M \times \mathcal{Y} \to \mathbb{R}$ be a continuous cost function, bounded below, and let $\mu \in P(M)$, $\nu \in P(\mathcal{Y})$, such that the optimal cost $C(\mu, \nu)$ is finite. Assume that:

* (i) $c$ is superdifferentiable everywhere (Assumption **(Super)**);
* (ii) $\nabla\_x c(x, \cdot)$ is injective (Assumption **(Twist)**);
* (iii) for any closed ball $B = B\_r(x\_0)$ and any compact set $K \subset \mathcal{Y}$, the function $c' := c\rvert\_{B \times K}$ is such that any $c'$-convex function on $B \times K$ is differentiable $\mu$-almost surely;
* (iv) $\mu$ is absolutely continuous with respect to the volume measure.

Then there exists a unique (in law) optimal coupling $(x,y)$ of $(\mu, \nu)$; it is **deterministic**, and satisfies the equation

$$\widetilde{\nabla} \psi(x) + \nabla_x c(x, y) = 0 \qquad \text{almost surely},$$

where $\widetilde{\nabla} \psi$ is the **approximate gradient** of $\psi$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.41</span><span class="math-callout__name">(Solution of the Monge Problem for the Square Distance)</span></p>

Let $M$ be a smooth Riemannian manifold, and $c(x,y) = d(x,y)^2$. Let $\mu, \nu$ be two probability measures on $M$, such that the optimal cost between $\mu$ and $\nu$ is finite. If $\mu$ is absolutely continuous, then there is a unique solution of the Monge problem between $\mu$ and $\nu$, and it can be written as

$$y = T(x) = \exp_x\bigl(\widetilde{\nabla} \psi(x)\bigr),$$

where $\psi$ is some $d^2/2$-convex function. The approximate gradient can be replaced by a true gradient if any one of the following conditions is satisfied:

* (a) $\mu$ and $\nu$ are compactly supported;
* (b) $M$ has nonnegative sectional curvature;
* (c) $\nu$ is compactly supported and $M$ has asymptotically nonnegative curvature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Particular Case 10.45</span></p>

If $M = \mathbb{R}^n$, formula (10.36) becomes $y = x + \nabla \psi(x) = \nabla\bigl(\lvert \cdot \rvert^2/2 + \psi\bigr)(x)$, where $\lvert \cdot \rvert^2/2 + \psi$ is convex lower semicontinuous, and we are back to Theorem 9.4.

</div>

### Removing the Assumption of Finite Cost

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.42</span><span class="math-callout__name">(Solution of the Monge Problem with Possibly Infinite Total Cost)</span></p>

Let $\mathcal{X}$ be a closed subset of a Riemannian manifold $M$ such that $\dim(\partial \mathcal{X}) \le n-1$, and let $\mathcal{Y}$ be an arbitrary Polish space. Let $c : M \times \mathcal{Y} \to \mathbb{R}$ be a continuous cost function, bounded below, and let $\mu \in P(\mathcal{X})$, $\nu \in P(\mathcal{Y})$. Assume that:

* (i) $c$ is locally semiconcave (Assumption **(locSC)**);
* (ii) $\nabla\_x c(x, \cdot)$ is injective (Assumption **(Twist)**);
* (iii) $\mu$ does not give mass to sets of dimension at most $n-1$.

Then there exists a unique (in law) coupling $(x, y)$ of $(\mu, \nu)$ such that $\pi = \mathrm{law}(x, y)$ is $c$-cyclically monotone; moreover this coupling is **deterministic**. The measure $\pi$ is called the **generalized optimal transference plan**. Furthermore, there is a $c$-convex function $\psi : M \to \mathbb{R} \cup \lbrace +\infty \rbrace$ such that $\pi[\partial\_c \psi] = 1$.

* If Assumption (iii) is reinforced to (iii') $\mu$ is absolutely continuous, then $\widetilde{\nabla} \psi(x) + \nabla\_x c(x, y) = 0$ $\pi$-almost surely.
* If Assumption (iii) is left as it is, but one adds (iv) the cost satisfies **(H$\infty$)** or **(SC)**, then $\nabla \psi(x) + \nabla\_x c(x, y) = 0$ $\pi$-almost surely, and this **characterizes** the generalized optimal transference plan. One can define a continuous map $T(x) \in \partial\_c \psi(x)$ on the set of differentiability points of $\psi$, and then $\mathrm{Spt}\, \nu = \overline{T(\mathrm{Spt}\, \mu)}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 10.44</span><span class="math-callout__name">(Generalized Monge Problem for the Square Distance)</span></p>

Let $M$ be a smooth Riemannian manifold, and let $c(x,y) = d(x,y)^2$. Let $\mu, \nu$ be two probability measures on $M$.

* If $\mu$ gives zero mass to sets of dimension at most $n-1$, then there is a unique transport map $T$ solving the generalized Monge problem between $\mu$ and $\nu$.
* If $\mu$ is absolutely continuous, then this solution can be written $y = T(x) = \exp\_x(\widetilde{\nabla} \psi(x))$, where $\psi$ is some $d^2/2$-convex function.
* If $M$ has nonnegative sectional curvature, or $\nu$ is compactly supported and $M$ satisfies asymptotically nonnegative curvature, then the approximate gradient can be replaced by a true gradient.

</div>

### First Appendix: Geometric Measure Theory

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.46</span><span class="math-callout__name">(Tangent Cone)</span></p>

If $S$ is an arbitrary subset of $\mathbb{R}^n$, and $x \in \overline{S}$, then the **tangent cone** $T\_x S$ to $S$ at $x$ is defined as

$$T_x S := \left\lbrace \lim_{k \to \infty} \frac{x_k - x}{t_k};\quad x_k \in S,\; x_k \to x,\; t_k > 0,\; t_k \to 0 \right\rbrace.$$

The dimension of this cone is the dimension of the vector space that it generates.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.47</span><span class="math-callout__name">(Countable Rectifiability)</span></p>

Let $S$ be a subset of $\mathbb{R}^n$, and let $d \in [0, n]$ be an integer. Then $S$ is said to be **countably $d$-rectifiable** if $S \subset \bigcup\_{k \in \mathbb{N}} f\_k(D\_k)$, where each $f\_k$ is Lipschitz on a measurable subset $D\_k$ of $\mathbb{R}^d$. In particular, $S$ has Hausdorff dimension at most $d$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.48</span><span class="math-callout__name">(Sufficient Conditions for Countable Rectifiability)</span></p>

* (i) Let $S$ be a measurable set in $\mathbb{R}^n$, such that $T\_x S$ has dimension at most $d$ for all $x \in S$. Then $S$ is countably $d$-rectifiable.
* (ii) Let $S$ be a measurable set in $\mathbb{R}^n$, such that $T\_x S$ is included in a half-space, for each $x \in \partial S$. Then $\partial S$ is countably $(n-1)$-rectifiable.

</div>

### Second Appendix: Nonsmooth Implicit Function Theorem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.49</span><span class="math-callout__name">(Clarke Subdifferential)</span></p>

Let $f$ be a continuous real-valued function defined on an open subset $U$ of a Riemannian manifold. For each $x \in U$, define $\partial f(x)$ as the **convex hull of all limits** of sequences $\nabla f(x\_k)$, where all $x\_k$ are differentiability points of $f$ and $x\_k \to x$. In short:

$$\partial f(x) = \overline{\mathrm{Conv}} \left\lbrace \lim_{x_k \to x} \nabla f(x_k) \right\rbrace.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.50</span><span class="math-callout__name">(Nonsmooth Implicit Function Theorem)</span></p>

Let $(f\_i)\_{1 \le i \le m}$ be real-valued Lipschitz functions defined in an open set $U$ of an $n$-dimensional Riemannian manifold, and let $x\_0 \in U$ be such that:

* (a) $\sum f\_i(x\_0) = 0$;
* (b) $0 \notin \sum \partial f\_i(x\_0)$.

Then $\lbrace \sum f\_i = 0 \rbrace$ is an $(n-1)$-dimensional **Lipschitz graph** around $x\_0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.51</span></p>

Let $\psi$ be a convex continuous function defined around some point $x\_0 \in \mathbb{R}^n$, and let $p \in \mathbb{R}^n$ such that $p$ does not belong to the Clarke differential of $\psi$ at $x\_0$; then $0$ does not belong to the Clarke differential of $\widetilde{\psi} : x \longmapsto \psi(x) - \psi(x\_0) - p \cdot (x - x\_0) + \lvert x - x\_0 \rvert^2$ at $x\_0$, and Theorem 10.50 obviously implies the existence of $x \ne x\_0$ such that $\widetilde{\psi}(x) = 0$, in particular $\psi(x) < \psi(x\_0) + p \cdot (x - x\_0)$. So $p$ does not belong to the subdifferential of $\psi$ at $x\_0$. In other words, the subdifferential is included in the Clarke differential. The other inclusion is obvious, so both notions coincide. This justifies a posteriori the notation $\partial \psi$ used in Definition 10.49. Similarly, for any locally semiconvex function $\psi$ defined in a neighborhood of $x$, $\partial \psi(x) = \nabla^- \psi(x)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 10.52</span><span class="math-callout__name">(Implicit Function Theorem for Two Subdifferentiable Functions)</span></p>

Let $\psi$ and $\widetilde{\psi}$ be two locally subdifferentiable functions defined in an open set $U$ of an $n$-dimensional Riemannian manifold $M$, and let $x\_0 \in U$ be such that $\psi$, $\widetilde{\psi}$ are differentiable at $x\_0$, and

$$\psi(x_0) = \widetilde{\psi}(x_0); \qquad \nabla \psi(x_0) \ne \nabla \widetilde{\psi}(x_0).$$

Then there is a neighborhood $V$ of $x\_0$ such that $\lbrace \psi = \widetilde{\psi} \rbrace \cap V$ is an $(n-1)$-dimensional Lipschitz graph; in particular, it has Hausdorff dimension exactly $n - 1$.

</div>

### Third Appendix: Curvature and the Hessian of the Squared Distance

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hessian of the Squared Distance)</span></p>

In $\mathbb{R}^n$, there is the simple formula $\nabla\_x^2 (\lvert x - y \rvert^2 / 2) = I\_n$. On a general Riemannian manifold $M$, there is no such simple formula; the Hessian $\nabla\_x^2 (d(x,y)^2/2)$ might not even be defined (it can take eigenvalues $-\infty$ if $x$ and $y$ are conjugate points). However, one can still estimate $\nabla\_x^2 (d(x,y)^2/2)$ **from above**, and thus derive semiconcavity estimates for $d^2/2$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Hessian Bound via Sectional Curvature)</span></p>

Let $x$ and $y$ be any two points in $M$, and let $\gamma$ be a minimizing geodesic joining $y$ to $x$, parametrized by arc length. Let $H(t)$ stand for the Hessian operator of $x \to d(x,y)^2/2$ at $x = \gamma(t)$. Define $h(t) = \langle H(t) \cdot e\_i(t), e\_i(t) \rangle$, where $e\_i(t)$ is a parallel transport of an orthonormal vector along $\gamma$. If $k(t) = \langle \nabla\_x^2 d(y,x) \cdot e\_i(t), e\_i(t) \rangle\_{\gamma(t)}$ and $\sigma(t)$ is the sectional curvature of the plane generated by $\dot{\gamma}(t)$ and $e\_i(t)$, then $h(t) = t\, k(t)$ and the key differential inequality is:

$$t\, \dot{h}(t) - h(t) + h(t)^2 \le -t^2\, \sigma(t).$$

From this inequality follow the two comparison results used in Theorems 10.41 and Corollary 10.44:

**(a)** If all **sectional curvatures of $M$ are nonnegative**, then $\dot{h} \le 0$, so $h$ remains bounded above by $1$ for all times:

$$\text{nonneg. sectional curvature} \implies \nabla_x^2 \!\left(\frac{d(x,y)^2}{2}\right) \le \mathrm{Id}\_{T_x M}.$$

This means $x \to d(x,y)^2/2$ is semiconcave with modulus $\omega(r) = r^2/2$.

**(b)** If $M$ has **asymptotically nonnegative curvature**, i.e. all sectional curvatures at point $x$ are bounded below by $-C/d(x\_0, x)^2$, then

$$\forall y \in K, \qquad \nabla_x^2 \!\left(\frac{d(x,y)^2}{2}\right) \le C(K)\, \mathrm{Id}\_{T_x M},$$

where $K$ is any compact subset of $M$. So $x \to d(x,y)^2/2$ is semiconcave with modulus $\omega(r) = C(K)\, r^2/2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.53</span></p>

The previous results apply to any compact manifold, or any manifold obtained from $\mathbb{R}^n$ by modification on a compact set. But they do *not* apply to the hyperbolic space $\mathbb{H}^n$: if $y$ is any given point in $\mathbb{H}^n$, then $x \to d(y,x)^2$ is not uniformly semiconcave as $x \to \infty$. (In a model of $\mathbb{H}^2$ as the unit disk with distance $d(r, \theta) = \log((1+r)/(1-r))$, the Hessian coefficient is $1 + r\, d(r)$, which diverges logarithmically as $r \to 1$.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.54</span></p>

The exponent 2 appearing in the definition of "asymptotically nonnegative curvature" ($\sigma\_x \ge -C/d(x\_0, x)^2$) is optimal: for any $p < 2$ it is possible to construct manifolds satisfying $\sigma\_x \ge -C/d(x\_0, x)^p$ on which $d(x\_0, \cdot)^2$ is not uniformly semiconcave.

</div>

### Bibliographical Notes on Chapter 10

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History of the Local Approach)</span></p>

The key ideas in this chapter were first used for the quadratic cost in Euclidean space by **Brenier**, **Rachev and Rüschendorf**. The existence of solutions to the Monge problem and the differentiability of $c$-convex functions for strictly superlinear convex cost functions in $\mathbb{R}^n$ was investigated by **Rüschendorf** (the formula $\nabla \psi(x) + \nabla\_x c(x, y) = 0$ seems to appear there for the first time), **Smith and Knott**, **Gangbo and McCann**. The latter authors get rid of all moment assumptions by avoiding the explicit use of Kantorovich duality.

The terminology of the **twist condition** comes from dynamical systems, in particular the study of twist diffeomorphisms (Bangert, Moser). There are cases of interest where the twist condition is satisfied even without a Lagrangian structure: the symmetrized Bregman cost function $c(x,y) = \langle \nabla \phi(x) - \nabla \phi(y), x - y \rangle$ where $\phi$ is strictly convex; and costs of the form $c(x,y) = \lvert x - y \rvert^2 + \lvert f(x) - g(y) \rvert^2$ where $f, g$ are convex.

**McCann** proved Theorem 10.41 when $M$ is a compact Riemannian manifold and $\mu$ is absolutely continuous — this was the first optimal transport theorem on a Riemannian manifold (save for the special case of the torus by Cordero-Erausquin). Later **Bernard and Buffoni** extended McCann's results to more general Lagrangian cost functions, importing tools from Mather's minimization problem. **Fang and Shao** rewrote McCann's theorem in the formalism of Lie groups. **Feyel and Üstünel** derived unique solvability of the Monge problem in the Wiener space. **Ambrosio and Rigot** adapted the proof to degenerate (sub-Riemannian) situations such as the Heisenberg group. **Bertrand** extended it to Alexandrov spaces.

The tricky proof of Theorem 10.42 takes its roots in **Alexandrov's uniqueness theorem** for graphs of prescribed Gauss curvature. **McCann** understood that Alexandrov's strategy could be revisited to yield uniqueness of a cyclically monotone transport in $\mathbb{R}^n$ without the assumption of finite total cost. The extension to Riemannian manifolds was performed by **Figalli**.

The use of **approximate differentials** as in Theorem 10.38 was initiated by **Ambrosio and collaborators** for strictly convex costs in $\mathbb{R}^n$. The adaptation to Riemannian manifolds is due to **Fathi and Figalli**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Technical Ingredients)</span></p>

* **Rademacher's theorem** (1918): almost everywhere differentiability of Lipschitz functions; the proof presented here is due to Christensen.
* The book by **Cannarsa and Sinestrari** is an excellent reference for semiconvexity and subdifferentiability in $\mathbb{R}^n$, as well as the links with Hamilton–Jacobi equations.
* The **Besicovich density theorem** is an alternative to the more classical Lebesgue density theorem (based on Vitali's covering lemma), which requires the doubling property. It works in $\mathbb{R}^n$ (or Riemannian manifolds by localization).
* The **nonsmooth implicit function theorem** (Theorem 10.50) seems to be folklore in nonsmooth real analysis; the core of the proof was explained to the author by Fathi. Corollary 10.52 was discovered or rediscovered by McCann in the case where $\psi$ and $\widetilde{\psi}$ are convex functions in $\mathbb{R}^n$.
* The case when the cost function is the distance ($c(x,y) = d(x,y)$) is *not* covered by Theorem 10.28, nor by any of the theorems in this chapter. This case is quite tricky; the treatment by Bernard and Buffoni is appealing for its links to dynamical system tools.

</div>

## Chapter 11: The Jacobian Equation

Transport is but a change of variables, and in many problems involving changes of variables, it is useful to write the **Jacobian equation** $f(x) = g(T(x))\, \mathcal{J}\_T(x)$, where $f$ and $g$ are the respective densities of the probability measures $\mu$ and $\nu$, and $\mathcal{J}\_T(x) = \lvert \det(\nabla T(x)) \rvert$ is the absolute value of the Jacobian determinant.

Two important things must be verified before writing this equation: $T$ should be **injective** on its domain of definition, and it should possess minimal **regularity**. The transport map $T$ might fail to be even continuous (as shown in Chapter 12), so there are three strategies:

1. Only use the Jacobian equation in situations where the optimal map is smooth (rare; discussed in Chapter 12).
2. Only use it for the optimal map between $\mu\_{t\_0}$ and $\mu\_t$, where $(\mu\_t)$ is a compactly supported displacement interpolation and $t\_0 \in (0,1)$ is fixed. Then by Theorem 8.5, the transport map is essentially Lipschitz. **This is the strategy used most often.**
3. Apply a more sophisticated change of variables theorem covering possibly discontinuous maps of bounded variation, or approximately differentiable maps (Theorem 11.1).

### The General Jacobian Equation

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.1</span><span class="math-callout__name">(Jacobian Equation)</span></p>

Let $M$ be a Riemannian manifold, let $f \in L^1(M)$ be a nonnegative integrable function on $M$, and let $T : M \to M$ be a Borel map. Define $\mu(dx) = f(x)\, \mathrm{vol}(dx)$ and $\nu := T\_\# \mu$. Assume that:

* (i) There exists a measurable set $\Sigma \subset M$, such that $f = 0$ almost everywhere outside of $\Sigma$, and $T$ is injective on $\Sigma$;
* (ii) $T$ is approximately differentiable almost everywhere on $\Sigma$.

Let $\widetilde{\nabla} T$ be the approximate gradient of $T$, and let $\mathcal{J}\_T$ be defined almost everywhere on $\Sigma$ by $\mathcal{J}\_T(x) := \lvert \det(\widetilde{\nabla} T(x)) \rvert$. Then $\nu$ is absolutely continuous with respect to the volume measure if and only if $\mathcal{J}\_T > 0$ almost everywhere. In that case $\nu$ is concentrated on $T(\Sigma)$, and its density is determined by the equation

$$f(x) = g(T(x))\, \mathcal{J}_T(x).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bounded Variation)</span></p>

Theorem 11.1 establishes the Jacobian equation as soon as, say, the optimal transport has locally bounded variation. In that case the map $T$ is almost everywhere differentiable, and its gradient coincides with the absolutely continuous part of the distributional gradient $\nabla\_{\mathcal{D}'} T$. The property of bounded variation is obviously satisfied for the quadratic cost in Euclidean space, since the second derivative of a convex function is a nonnegative measure.

</div>

### Application: The Quadratic Cost in $\mathbb{R}^n$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.2</span><span class="math-callout__name">(Quadratic Cost in $\mathbb{R}^n$)</span></p>

Consider two probability measures $\mu\_0$ and $\mu\_1$ on $\mathbb{R}^n$, with finite second moments, both absolutely continuous with respective densities $f\_0$ and $f\_1$. The unique optimal transport map takes the form $T(x) = \nabla \Psi(x)$ for some lower semicontinuous convex function $\Psi$. There is a unique displacement interpolation $(\mu\_t)\_{0 \le t \le 1}$, defined by

$$\mu_t = (T_t)\_\# \mu_0, \qquad T_t(x) = (1-t)x + t\, T(x) = (1-t)x + t\, \nabla \Psi(x).$$

By Theorem 8.7, each $\mu\_t$ is absolutely continuous; let $f\_t$ be its density. The map $\nabla T = \nabla^2 \Psi$ is of locally bounded variation (the Alexandrov Hessian of $\Psi$), so Theorem 11.1 yields, $\mu\_0(dx)$-almost surely,

$$f_0(x) = f_1(\nabla \Psi(x))\, \det(\nabla^2 \Psi(x)).$$

More generally, for any $t \in [0,1]$:

$$f_0(x) = f_t(T_t(x))\, \det\bigl((1-t)I_n + t \nabla^2 \Psi(x)\bigr).$$

If $T\_{t\_0 \to t} = T\_t \circ T\_{t\_0}^{-1}$ is the transport map between $\mu\_{t\_0}$ and $\mu\_t$ (which by Theorem 8.5 is Lipschitz for $t\_0 \in (0,1)$), then:

$$f_{t_0}(x) = f_t(T_{t_0 \to t}(x))\, \det(\nabla T_{t_0 \to t}(x)).$$

</div>

### The Change of Variables Theorem for Lagrangian Costs

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.3</span><span class="math-callout__name">(Change of Variables)</span></p>

Let $M$ be a Riemannian manifold, and $c(x,y)$ a cost function deriving from a $C^2$ Lagrangian $L(x,v,t)$ on $TM \times [0,1]$, where $L$ satisfies the classical conditions of Definition 7.6, together with $\nabla\_v^2 L > 0$. Let $(\mu\_t)\_{0 \le t \le 1}$ be a displacement interpolation, such that each $\mu\_t$ is absolutely continuous and has density $f\_t$. Let $t\_0 \in (0,1)$, and $t \in [0,1]$; further, let $T\_{t\_0 \to t}$ be the ($\mu\_{t\_0}$-almost surely) unique optimal transport from $\mu\_{t\_0}$ to $\mu\_t$, and let $\mathcal{J}\_{t\_0 \to t}$ be the associated Jacobian determinant. Let $F$ be a nonnegative measurable function on $M \times \mathbb{R}\_+$ such that $[f\_t(y) = 0] \implies F(y, f\_t(y)) = 0$. Then,

$$\int_M F(y, f_t(y))\, \mathrm{vol}(dy) = \int_M F\!\left(T_{t_0 \to t}(x),\; \frac{f_{t_0}(x)}{\mathcal{J}\_{t_0 \to t}(x)}\right) \mathcal{J}\_{t_0 \to t}(x)\, \mathrm{vol}(dx).$$

Furthermore, $\mu\_{t\_0}(dx)$-almost surely, $\mathcal{J}\_{t\_0 \to t}(x) > 0$ for all $t \in [0,1]$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Positivity of the Jacobian)</span></p>

The positivity of $\mathcal{J}\_{t\_0 \to t}$ for all $t$ (not just $t = 1$) is important. It follows from the factorization $T\_{t\_0 \to t} = F\_3 \circ F\_2 \circ F\_1$, where $F\_1 : \gamma(t\_0) \to (\gamma(0), \gamma(t\_0))$, $F\_2 : (\gamma(0), \gamma(t\_0)) \to (\gamma(0), \dot{\gamma}(0))$, and $F\_3 : (\gamma(0), \dot{\gamma}(0)) \to \gamma(t)$. Both $F\_2$ and $F\_3$ have positive Jacobian determinant (at least for $t < 1$); and $F\_1$ has positive Jacobian whenever $x$ is chosen such that $F\_1$ already has a positive Jacobian determinant. By Problem 8.8, focalization is impossible before the cut locus.

</div>

### Bibliographical Notes on Chapter 11

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History)</span></p>

In the context of optimal transport, the change of variables formula (11.1) was first proven by **McCann**. His argument is based on Lebesgue's density theory, and takes advantage of **Alexandrov's theorem** (a convex function admits a Taylor expansion to order 2 at almost each point of its domain). McCann's argument is reproduced in the book. Along with **Cordero-Erausquin and Schmuckenschläger**, McCann later generalized his result to Riemannian manifolds. Then **Cordero-Erausquin** treated the case of strictly convex cost functions in $\mathbb{R}^n$.

**Ambrosio** pointed out that those results could be retrieved within the general framework of push-forward by approximately differentiable mappings. This viewpoint also applies to nonsmooth cost functions such as $\lvert x - y \rvert^p$. It is a general feature of optimal transport with strictly convex cost in $\mathbb{R}^n$ that the Jacobian matrix $\nabla T$, even if not necessarily nonnegative symmetric, is diagonalizable with nonnegative eigenvalues.

Changes of variables of the form $y = \exp\_x(\nabla \psi(x))$ (where $\psi$ is not necessarily $d^2/2$-convex) have been used in a remarkable paper by **Cabré** to investigate qualitative properties of nondivergent elliptic equations on Riemannian manifolds with nonnegative sectional curvature. For the Harnack inequality, Cabré's method was extended to nonneg. *Ricci* curvature by **S. Kim**.

</div>

## Chapter 12: Smoothness

The smoothness of the optimal transport map may give information about its qualitative behavior and simplify computations. What characterizes the optimal transport map $T$ is the existence of a $c$-convex $\psi$ such that $\nabla \psi(x) + \nabla\_x c(x, y) = 0$; so it is natural to search for a closed equation on $\psi$.

### The PDE of Optimal Transport

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Derivation of the Generalized Monge–Ampère Equation)</span></p>

Working formally (assuming smoothness), differentiate $\nabla \psi(x) + \nabla\_x c(x, T(x)) = 0$ once more with respect to $x$:

$$\nabla^2 \psi(x) + \nabla_{xx}^2 c(x, T(x)) + \nabla_{xy}^2 c(x, T(x)) \cdot \nabla T(x) = 0,$$

which can be rewritten as

$$\nabla^2 \psi(x) + \nabla_{xx}^2 c(x, T(x)) = -\nabla_{xy}^2 c(x, T(x)) \cdot \nabla T(x).$$

The left-hand side is the Hessian of the function $x' \mapsto c(x', T(x)) + \psi(x')$, evaluated at the minimum $x' = x$; hence it is a **nonnegative symmetric** operator. Taking absolute values of determinants and using $f(x) = g(T(x))\, \mathcal{J}\_T(x)$, one arrives at the basic **partial differential equation of optimal transport**:

$$\det\bigl(\nabla^2 \psi(x) + \nabla_{xx}^2 c(x, T(x))\bigr) = \bigl\lvert \det \nabla_{xy}^2 c(x, T(x)) \bigr\rvert\; \frac{f(x)}{g(T(x))},$$

where $T(x) = (\nabla\_x c)^{-1}(x, -\nabla \psi(x))$. This is a closed equation for $\psi$, sometimes called the **generalized Monge–Ampère equation**. For the quadratic cost $c(x,y) = -x \cdot y$ in $\mathbb{R}^n$, it reduces to

$$\det \nabla^2 \psi(x) = \frac{f(x)}{g(\nabla \psi(x))},$$

which is the classical **Monge–Ampère equation**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimal Transport is in General Not Smooth)</span></p>

One sad conclusion: **optimal transport is in general not smooth** — even worse, smoothness requires nonlocal conditions which are probably impossible to check effectively on a generic Riemannian manifold. It is actually a striking feature of optimal transport that the theory can be pushed very far with so little regularity available.

</div>

### Caffarelli's Counterexample

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 12.2</span></p>

Let $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$ be any two Polish probability spaces, let $T$ be a continuous map $\mathcal{X} \to \mathcal{Y}$, and let $\pi = (\mathrm{Id}, T)\_\# \mu$ be the associated transport map. Then, for each $x \in \mathrm{Spt}\, \mu$, the pair $(x, T(x))$ belongs to the support of $\pi$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.3</span><span class="math-callout__name">(Caffarelli's Counterexample: Discontinuous Optimal Transport)</span></p>

There are smooth compactly supported probability densities $f$ and $g$ on $\mathbb{R}^n$, such that the supports of $f$ and $g$ are smooth and connected, $f$ and $g$ are (strictly) positive in the interior of their respective supports, and yet the optimal transport between $\mu(dx) = f(x)\, dx$ and $\nu(dy) = g(y)\, dy$, for the cost $c(x,y) = \lvert x - y \rvert^2$, is **discontinuous**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The "Dumb-bells" Construction)</span></p>

The proof uses a "dumb-bells" construction: let $f$ be the indicator of the unit ball $B$ in $\mathbb{R}^2$, and let $g = g\_\varepsilon$ be the indicator of a set $C\_\varepsilon$ obtained by separating a ball into two halves $B\_1$ and $B\_2$ (distance 2) and building a thin bridge of width $O(\varepsilon)$ between them. While $g\_\varepsilon$ can be obtained from $f$ by a continuous deformation (like playing with clay), for $\varepsilon$ small enough the optimal transport *cannot* be continuous. The argument uses the stability of optimal transport (Corollary 5.23): as $\varepsilon \downarrow 0$, $T\_\varepsilon \to T$ in probability, where $T$ (the limiting transport to $B\_1 \cup B\_2$) splits the top region $S$ of the ball into $S\_-$ and $S\_+$. If $T\_\varepsilon$ were continuous, the image of $S$ would be connected, preventing this splitting — a contradiction with cyclical monotonicity.

</div>

### Loeper's Counterexample

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.4</span><span class="math-callout__name">(Loeper's Counterexample: Discontinuous Transport on a Surface)</span></p>

There is a smooth compact Riemannian surface $S$, and there are smooth positive probability densities $f$ and $g$ on $S$, such that the optimal transport between $\mu(dx) = f(x)\, \mathrm{vol}(dx)$ and $\nu(dy) = g(y)\, \mathrm{vol}(dy)$, with a cost function equal to the square of the geodesic distance on $S$, is **discontinuous**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 12.5</span></p>

The obstruction has nothing to do with the lack of smoothness of the squared distance. Counterexamples of the same type exist for very smooth cost functions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 12.6</span></p>

As we shall see in Theorem 12.44, the surface $S$ in Theorem 12.4 could be replaced by any compact Riemannian manifold admitting a **negative sectional curvature** at some point. In that sense there is no hope for general regularity results outside the world of nonnegative sectional curvature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Idea of Theorem 12.4 — Horse Saddle)</span></p>

Let $S$ be a compact surface in $\mathbb{R}^3$, invariant under $x \to -x$ and $y \to -y$, which near the origin $O$ coincides with the horse saddle $z = x^2 - y^2$. Near $O$, $S$ has strictly negative curvature. Choose four points $A\_\pm = (\pm x\_0, 0, x\_0^2)$ and $B\_\pm = (0, \pm y\_0, -y\_0^2)$ close to $O$. On a negatively curved surface, Pythagoras's identity in a triangle with a square angle is modified in favor of the diagonal: $d(O, A\_\pm)^2 + d(O, B\_\pm)^2 < d(A\_\pm, B\_\pm)^2$. Using this and a symmetry argument, one shows that either $T$ or its inverse $\widetilde{T}$ must be discontinuous: if both were continuous, by symmetry $T(O) = O$, but then the transport scheme $(A \to B, O \to O)$ can be improved to $(A \to O, O \to B)$ using the negative-curvature Pythagoras, contradicting $c$-cyclical monotonicity.

</div>

### Smoothness and Assumption (C)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.7</span><span class="math-callout__name">(Smoothness Needs Assumption (C))</span></p>

Let $\mathcal{X}$ (resp. $\mathcal{Y}$) be the closure of a bounded open set in a smooth Riemannian manifold $M$ (resp. $N$), equipped with its volume measure. Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be a continuous cost function. Assume that there are a $c$-convex function $\psi : \mathcal{X} \to \mathbb{R}$, and a point $\overline{x} \in \mathcal{X}$, such that $\partial\_c \psi(\overline{x})$ is **disconnected**. Then there exist $C^\infty$ positive probability densities $f$ on $\mathcal{X}$ and $g$ on $\mathcal{Y}$, such that the optimal transport map $T$ between $\mu = f\, \mathrm{vol}$ and $\nu = g\, \mathrm{vol}$ is **discontinuous**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation)</span></p>

In other words, Assumption (C) (connectedness of $c$-subdifferentials, introduced in Chapter 9) is more or less **necessary** for the regularity of optimal transport. This means that the conditions identified in Chapter 9 as sufficient for solvability of the Monge problem are also essentially necessary for smoothness. Thus the search for regularity of optimal transport inevitably leads back to the geometry of $c$-subdifferentials.

</div>

### Regular Cost Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.10</span><span class="math-callout__name">($c$-Segment)</span></p>

A continuous curve $(y\_t)\_{0 \le t \le 1}$ in $\mathcal{Y}$ is said to be a **$c$-segment with base $\overline{x}$** if (a) $(\overline{x}, y\_t) \in \mathrm{Dom}(\nabla\_x c)$ for all $t$; (b) there are $p\_0, p\_1 \in T\_{\overline{x}} M$ such that $\nabla\_x c(\overline{x}, y\_t) + p\_t = 0$, where $p\_t = (1-t)\, p\_0 + t\, p\_1$. In other words, a $c$-segment is the image of a usual segment by the map $(\nabla\_x c(\overline{x}, \cdot))^{-1}$. It is uniquely determined by its base point $\overline{x}$ and its endpoints $y\_0, y\_1$; denoted $[y\_0, y\_1]\_{\overline{x}}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.11</span><span class="math-callout__name">($c$-Convexity of a Set)</span></p>

A set $C \subset \mathcal{Y}$ is said to be **$c$-convex with respect to $\overline{x} \in \mathcal{X}$** if for any two points $y\_0, y\_1 \in C$ the $c$-segment $[y\_0, y\_1]\_{\overline{x}}$ is entirely contained in $C$. More generally, $C$ is $c$-convex with respect to a subset $\widetilde{X}$ of $\mathcal{X}$ if $C$ is $c$-convex with respect to any $x \in \widetilde{X}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 12.12</span></p>

When $\mathcal{X} = \mathcal{Y} = \mathbb{R}^n$ and $c(x,y) = -x \cdot y$ (or $x \cdot y$), $c$-convexity is just plain convexity. If $\mathcal{X} = \mathcal{Y} = S^{n-1}$ and $c(x,y) = d(x,y)^2/2$ then $\mathrm{Dom}(\nabla\_x c(\overline{x}, \cdot)) = S^{n-1} \setminus \lbrace -\overline{x} \rbrace$ (the cut locus is the antipodal point), and $\nabla\_x c(\overline{x}, S^{n-1} \setminus \lbrace -\overline{x} \rbrace) = B(0, \pi) \subset T\_{\overline{x}} M$ is a convex set. So for any point $\overline{x}$, $S^{n-1}$ minus the cut locus of $\overline{x}$ is $c$-convex with respect to $\overline{x}$. An equivalent statement is that $S^{n-1}$ is $d^2/2$-convex with respect to itself.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.14</span><span class="math-callout__name">(Regular Cost Function)</span></p>

A cost $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ is said to be **regular** if for any $\overline{x}$ in the interior of $\mathcal{X}$ and for any $c$-convex function $\psi : \mathcal{X} \to \mathbb{R}$, the set $\partial\_c \psi(\overline{x}) \cap \mathrm{Dom}'(\nabla\_x c(\overline{x}, \cdot))$ is $c$-convex with respect to $\overline{x}$.

The cost $c$ is said to be **strictly regular** if moreover, for any nontrivial $c$-segment $(y\_t) = [y\_0, y\_1]\_{\overline{x}}$ in $\partial\_c \psi(\overline{x})$ and for any $t \in (0,1)$, $\overline{x}$ is the only contact point of $\psi^c$ at $y\_t$, i.e. the only $x \in \mathcal{X}$ such that $y\_t \in \partial\_c \psi(x)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12.15</span><span class="math-callout__name">(Reformulation of Regularity)</span></p>

Let $\mathcal{X}$ be a closed subset of a Riemannian manifold $M$ and $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ a continuous cost function satisfying **(Twist)**. Then:

* (i) $c$ is regular if and only if for any $(\overline{x}, y\_0), (\overline{x}, y\_1) \in \mathrm{Dom}'(\nabla\_x c)$, the $c$-segment $[y\_0, y\_1]\_{\overline{x}}$ is well-defined, and for any $t \in [0,1]$,

$$-c(x, y_t) + c(\overline{x}, y_t) \le \max\bigl(-c(x, y_0) + c(\overline{x}, y_0),\; -c(x, y_1) + c(\overline{x}, y_1)\bigr)$$

(with strict inequality if $c$ is strictly regular, $y\_0 \ne y\_1$, $t \in (0,1)$ and $\overline{x} \ne x$).

* (ii) If $c$ is a regular cost function satisfying **(locSC)** and **(H$\infty$)**, then for any $c$-convex $\psi : \mathcal{X} \to \mathbb{R}$ and any $\overline{x}$ in the interior of $\mathcal{X}$ such that $\partial\_c \psi(\overline{x}) \subset \mathrm{Dom}'(\nabla\_x c(\overline{x}, \cdot))$, one has $\nabla\_c^- \psi(\overline{x}) = \nabla^- \psi(\overline{x})$, where $\nabla\_c^- \psi(\overline{x}) := -\nabla\_x c(\overline{x}, \partial\_c \psi(\overline{x}))$.

* (iii) If $c$ satisfies **(locSC)** and $\mathrm{Dom}'(\nabla\_x c)$ is totally $c$-convex, then $c$ is regular if and only if for any $c$-convex $\psi : \mathcal{X} \to \mathbb{R}$ and any $\overline{x}$ in the interior of $\mathcal{X}$, $\partial\_c \psi(\overline{x}) \cap \mathrm{Dom}'(\nabla\_x c(\overline{x}, \cdot))$ is **connected**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 12.19</span></p>

Proposition 12.15(iii) shows that, modulo issues about the domain of differentiability, the regularity property is morally equivalent to Assumption (C) in Chapter 9.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.20</span><span class="math-callout__name">(Nonregularity Implies Nondensity of Differentiable $c$-Convex Functions)</span></p>

Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ satisfy **(Twist)**, **(locSC)** and **(H$\infty$)**. Let $C$ be a totally $c$-convex set contained in $\mathrm{Dom}'(\nabla\_x c)$, such that (a) $c$ is not regular in $C$; and (b) for any $(\overline{x}, y\_0), (\overline{x}, y\_1)$ in $C$, $\partial\_c \psi\_{\overline{x}, y\_0, y\_1}(\overline{x}) \subset \mathrm{Dom}'(\nabla\_x c(\overline{x}, \cdot))$. Then for some $(\overline{x}, y\_0), (\overline{x}, y\_1)$ in $C$ the $c$-convex function $\psi = \psi\_{\overline{x}, y\_0, y\_1}$ **cannot** be the locally uniform limit of differentiable $c$-convex functions $\psi\_k$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 12.21</span><span class="math-callout__name">(Nonsmoothness of the Kantorovich Potential)</span></p>

With the same assumptions as in Theorem 12.20, if $\mathcal{Y}$ is a closed subset of a Riemannian manifold, then there are smooth positive probability densities $f$ on $\mathcal{X}$ and $g$ on $\mathcal{Y}$, such that the associated Kantorovich potential $\psi$ is **not differentiable**.

</div>

### The Ma–Trudinger–Wang Condition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Strong Twist Condition (STwist))</span></p>

$\mathrm{Dom}'(\nabla\_x c)$ is an open set on which $c$ is smooth, $\nabla\_x c$ is one-to-one, and the **mixed Hessian** $\nabla\_{x,y}^2 c$ is **nonsingular**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.29</span><span class="math-callout__name">($c$-Exponential)</span></p>

Let $c$ be a cost function satisfying **(Twist)**. Define the **$c$-exponential** map on the image of $-\nabla\_x c$ by the formula $c\text{-}\exp\_x(p) = (\nabla\_x c)^{-1}(x, -p)$. In other words, $c\text{-}\exp\_x(p)$ is the unique $y$ such that $\nabla\_x c(x, y) + p = 0$. When $c(x,y) = d(x,y)^2/2$ on a complete Riemannian manifold, one recovers the usual Riemannian exponential map.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.26</span><span class="math-callout__name">($c$-Second Fundamental Form)</span></p>

Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ satisfy **(STwist)**. Let $\Omega \subset \mathcal{Y}$ be open (in the ambient manifold) with $C^2$ boundary $\partial \Omega$ contained in the interior of $\mathcal{Y}$. Let $(x, y) \in \mathrm{Dom}'(\nabla\_x c)$, with $y \in \partial \Omega$, and let $n$ be the outward unit normal vector to $\partial \Omega$. Define the quadratic form $\mathbb{I}\_c(x, y)$ on $T\_y \Omega$ by the formula

$$\mathbb{I}_c(x, y)(\xi) = \sum_{ijk\ell} c_{i,k}\, \partial_j\bigl(c^{k,\ell}\, n_\ell\bigr)\, \xi^i\, \xi^j.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.27</span><span class="math-callout__name">(Ma–Trudinger–Wang Tensor, or $c$-Curvature Operator)</span></p>

Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ satisfy **(STwist)**. For any $(x, y) \in \mathrm{Dom}'(\nabla\_x c)$, define a quadrilinear form $\mathfrak{S}\_c(x, y)$ on the space of bivectors $(\xi, \eta) \in T\_x M \times T\_y N$ satisfying $\langle \nabla\_{x,y}^2 c(x,y) \cdot \xi, \eta \rangle = 0$, by the formula

$$\mathfrak{S}_c(x, y)(\xi, \eta) = \frac{3}{2} \sum_{ijk\ell rs} \bigl(c_{ij,r}\, c^{r,s}\, c_{s,k\ell} - c_{ij,k\ell}\bigr)\, \xi^i\, \xi^j\, \eta^k\, \eta^\ell.$$

Equivalently, using the $c$-exponential and setting $p = -\nabla\_x c(x, y)$:

$$\mathfrak{S}_c(x, y)(\xi, \eta) = -\frac{3}{2}\, \frac{d^2}{ds^2}\, \frac{d^2}{dt^2}\, c\bigl(\exp_x(t\xi),\; c\text{-}\exp_x(p + s\eta)\bigr).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Particular Case 12.30</span><span class="math-callout__name">(Loeper's Identity)</span></p>

If $\mathcal{X} = \mathcal{Y} = M$ is a smooth complete Riemannian manifold, $c(x,y) = d(x,y)^2/2$, and $\xi, \eta$ are two unit orthogonal vectors in $T\_x M$, then

$$\mathfrak{S}_c(x, x)(\xi, \eta) = \sigma_x(P)$$

is the **sectional curvature** of $M$ at $x$ along the plane $P$ generated by $\xi$ and $\eta$. Thus the $c$-curvature is a nonlocal generalization of sectional curvature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 12.32</span></p>

$\mathfrak{S}\_c$ is not a standard curvature-type operator, for at least two reasons. First it involves derivatives of order greater than 2. Second, it is **nonlocal** in a strong sense: for $c(x,y) = d(x,y)^2/2$ on a Riemannian manifold, a change of the metric $g$ can affect the value of $\mathfrak{S}\_c(x,y)$, even if the metric is left unchanged in a neighborhood of both $x$ and $y$ and even in a neighborhood of the geodesics joining $x$ to $y$! This is because geodesic distance is itself a highly nonlocal notion.

</div>

### Main Theorems on Regularity via $\mathfrak{S}\_c$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.35</span><span class="math-callout__name">(Differential Formulation of $c$-Convexity)</span></p>

Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be a cost function satisfying **(STwist)**. Let $x \in \mathcal{X}$ and let $C$ be a connected open subset of $\mathcal{Y}$ with $C^2$ boundary. Let $x \in \mathcal{X}$, such that $\lbrace x \rbrace \times \overline{C} \subset \mathrm{Dom}'(\nabla\_x c)$. Then $\overline{C}$ is $c$-convex with respect to $x$ if and only if $\mathbb{I}\_c(x, y) \ge 0$ for all $y \in \partial C$.

If moreover $\mathbb{I}\_c(x, y) > 0$ for all $y \in \partial C$ then $C$ is strictly $c$-convex with respect to $x$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.36</span><span class="math-callout__name">(Differential Formulation of Regularity)</span></p>

Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be a cost function satisfying **(STwist)**, and let $D$ be a totally $c$-convex open subset of $\mathrm{Dom}'(\nabla\_x c)$. Then:

* (i) If $c$ is regular in $D$, then $\mathfrak{S}\_c(x, y) \ge 0$ for all $(x, y) \in D$.
* (ii) Conversely, if $\mathfrak{S}\_c(x, y) \ge 0$ (resp. $> 0$) for all $(x,y) \in D$, $\check{c}$ satisfies **(STwist)**, $\check{D}$ is totally $\check{c}$-convex, and $c$ satisfies **(Cut$^{n-1}$)** on $D$, then $c$ is regular (resp. strictly regular) in $D$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 12.37</span></p>

For Theorem 12.36 to hold, it is important that no condition be imposed on the sign of $\mathfrak{S}\_c(x,y) \cdot (\xi, \eta)$ when $\xi$ and $\eta$ are not "orthogonal" ($\langle \nabla\_{x,y}^2 c \cdot \xi, \eta \rangle \ne 0$). For instance, $c(x,y) = \sqrt{1 + \lvert x - y \rvert^2}$ is regular, even though $\mathfrak{S}\_c(x,y) \cdot (\xi, \xi) < 0$ for $\xi = x - y$ and $\lvert x - y \rvert$ large enough.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary of Regularity Conditions)</span></p>

The chain of implications relating the various notions:

$$\mathfrak{S}_c \ge 0 \;\; (+\text{STwist, Cut}^{n-1}) \;\Longrightarrow\; c \text{ is regular} \;\Longleftrightarrow\; \text{Assumption (C)} \;\Longrightarrow\; \text{smoothness is possible.}$$

Conversely: regularity of $c$ implies $\mathfrak{S}\_c \ge 0$. So the condition $\mathfrak{S}\_c \ge 0$ is essentially necessary *and* sufficient for regularity of optimal transport. In particular, for the squared distance cost on a Riemannian manifold, $\mathfrak{S}\_c(x,x) = \sigma\_x$ (sectional curvature), so nonnegative sectional curvature is a necessary condition for regularity.

</div>

### Negative Curvature Implies Discontinuity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.44</span><span class="math-callout__name">(Negative Sectional Curvature Implies Discontinuous Transport)</span></p>

Let $M$ be a smooth compact Riemannian manifold such that the sectional curvature is strictly negative at some point. Then there are smooth positive probability densities $f$ and $g$ on $M$ such that the optimal transport map $T$ from $f\, \mathrm{vol}$ to $g\, \mathrm{vol}$, with cost $c(x,y) = d(x,y)^2$, is **discontinuous**.

The same conclusion holds true under the weaker assumption that $\mathfrak{S}\_c(\overline{x}, \overline{y}) \cdot (\xi, \eta) < 0$ for some $(\overline{x}, \overline{y}) \in M \times M$ such that $\overline{y}$ does not belong to the cut locus of $\overline{x}$ and $\nabla\_{xy}^2 c(\overline{x}, \overline{y}) \cdot (\xi, \eta) = 0$.

</div>

### Differential Formulation of $c$-Convexity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.46</span><span class="math-callout__name">(Differential Criterion for $c$-Convexity)</span></p>

Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be a cost function such that $c$ and $\check{c}$ satisfy **(STwist)**, and let $D$ be a totally $c$-convex closed subset of $\mathrm{Dom}'(\nabla\_x c)$ such that $\check{D}$ is totally $\check{c}$-convex and $\mathfrak{S}\_c \ge 0$ on $D$. Let $\mathcal{X}' = \mathrm{proj}\_\mathcal{X}(D)$ and let $\psi \in C^2(\mathcal{X}'; \mathbb{R})$ (meaning that $\psi$ is twice continuously differentiable on $\mathcal{X}'$, up to the boundary). If for any $x \in \mathcal{X}'$ there is $y \in \mathcal{Y}$ such that $(x, y) \in D$ and

$$\begin{cases} \nabla \psi(x) + \nabla_x c(x, y) = 0 \\\\ \nabla^2 \psi(x) + \nabla_x^2 c(x, y) \ge 0, \end{cases}$$

then $\psi$ is $c$-convex on $\mathcal{X}'$ (or more rigorously, $c'$-convex, where $c'$ is the restriction of $c$ to $D$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 12.47</span></p>

In view of the generalized Monge–Ampère equation at the beginning of this chapter, condition (12.28) is a **necessary and sufficient** condition for $c$-convexity, up to issues about the smoothness of $\psi$ and the domain of differentiability of $c$. The set of $y$'s appearing in (12.28) is not required to be the whole of $\mathrm{proj}\_\mathcal{Y}(D)$, so in practice one may often enlarge $D$ in the $y$ variable before applying Theorem 12.46.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 12.48</span></p>

Theorem 12.46 shows that if $c$ is a regular cost, then (up to issues about the domain of definition) $c$-convexity is a **local** notion.

</div>

### Control of the Gradient via $c$-Convexity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.49</span><span class="math-callout__name">(Control of $c$-Subdifferential by $c$-Convexity of Target)</span></p>

Let $\mathcal{X}$, $\mathcal{Y}$, $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$, $\mu \in P(\mathcal{X})$, $\nu \in P(\mathcal{Y})$ and $\psi : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ satisfy the same assumptions as in Theorem 10.28 (including **(H$\infty$)**). Let $\Omega \subset \mathcal{X}$ be an open set such that $\mathrm{Spt}\, \mu = \overline{\Omega}$, and let $C \subset \mathcal{Y}$ be a closed set such that $\mathrm{Spt}\, \nu \subset C$. Assume that:

* (a) $\Omega \times C \subset \mathrm{Dom}'(\nabla\_x c)$;
* (b) $C$ is $c$-convex with respect to $\Omega$.

Then $\partial\_c \psi(\Omega) \subset C$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Importance of Theorem 12.49)</span></p>

The property $\partial\_c \psi(\Omega) \subset C$ is the key to get good control of the localization of the gradient of the solution to the generalized Monge–Ampère equation. It is needed to approximate $\psi$ by smooth solutions: without control on $\partial\_c \psi(x)$ when $\psi$ is not differentiable at $x$, the approximating $y = T(x)$ might escape the support of $\nu$, causing trouble. The $c$-convexity of the target $C$ is what guarantees this control.

</div>

### Smoothness Results

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conditions for Smoothness)</span></p>

After the counterexamples (Theorems 12.3, 12.4, 12.7, Corollary 12.21, Theorem 12.44), a good regularity theory can be developed once the previously discussed obstructions are avoided by:

* suitable assumptions of **convexity of the domains** (or $c$-convexity);
* suitable assumptions of **regularity of the cost function** ($\mathfrak{S}\_c \ge 0$).

These results constitute a chapter in the theory of Monge–Ampère-type equations, more precisely for the **second boundary value problem** (the boundary condition is not of Dirichlet type; instead, the image of the source domain by the transport map should be the target domain). Typically a convexity-type assumption on the target will be needed for **local** regularity results, while **global** regularity (up to the boundary) will request convexity of both domains. The main problem is to get $C^2$ estimates on $\psi$; once these are secured, the equation becomes "uniformly elliptic", and higher regularity follows from the well-developed machinery of fully nonlinear second-order PDEs (Schauder estimates).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.50</span><span class="math-callout__name">(Caffarelli's Regularity Theory)</span></p>

Let $c(x,y) = \lvert x - y \rvert^2$ in $\mathbb{R}^n \times \mathbb{R}^n$, and let $\Omega, \Lambda$ be connected bounded open subsets of $\mathbb{R}^n$. Let $f, g$ be probability densities on $\Omega$ and $\Lambda$ respectively, with $f$ and $g$ bounded from above and below. Let $\psi : \Omega \to \mathbb{R}$ be the unique (up to an additive constant) Kantorovich potential associated with $\mu(dx) = f(x)\, dx$ and $\nu(dy) = g(y)\, dy$, and the cost $c$. Then:

* (i) If $\Lambda$ is convex, then $\psi \in C^{1,\beta}(\Omega)$ for some $\beta \in (0,1)$.
* (ii) If $\Lambda$ is convex, $f \in C^{0,\alpha}(\Omega)$, $g \in C^{0,\alpha}(\Lambda)$ for some $\alpha \in (0,1)$, then $\psi \in C^{2,\alpha}(\Omega)$; moreover, for any $k \in \mathbb{N}$ and $\alpha \in (0,1)$,

$$f \in C^{k,\alpha}(\Omega),\; g \in C^{k,\alpha}(\Lambda) \implies \psi \in C^{k+2,\alpha}(\Omega).$$

* (iii) If $\Lambda$ and $\Omega$ are $C^2$ and uniformly convex, $f \in C^{0,\alpha}(\overline{\Omega})$ and $g \in C^{0,\alpha}(\overline{\Lambda})$ for some $\alpha \in (0,1)$, then $\psi \in C^{2,\alpha}(\overline{\Omega})$; more generally,

$$f \in C^{k,\alpha}(\overline{\Omega}),\; g \in C^{k,\alpha}(\overline{\Lambda}),\; \Omega, \Lambda \in C^{k+2} \implies \psi \in C^{k+2,\alpha}(\overline{\Omega}).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.51</span><span class="math-callout__name">(Urbas–Trudinger–Wang Regularity Theory)</span></p>

Let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be a smooth cost function satisfying **(STwist)** and $\mathfrak{S}\_c \ge 0$ in the interior of $\mathcal{X} \times \mathcal{Y}$. Let $\Omega \subset \mathcal{X}$ and $\Lambda \subset \mathcal{Y}$ be $C^2$-smooth connected open sets and let $f \in C(\overline{\Omega})$, $g \in C(\overline{\Lambda})$ be positive probability densities. Let $\psi$ be the unique Kantorovich potential associated with $\mu(dx) = f(x)\, dx$ and $\nu(dy) = g(y)\, dy$, and the cost $c$. If (a) $\Lambda$ is uniformly $c$-convex with respect to $\Omega$, and $\Omega$ uniformly $\check{c}$-convex with respect to $\Lambda$, (b) $f \in C^{1,1}(\overline{\Omega})$, $g \in C^{1,1}(\overline{\Lambda})$, and (c) $\Lambda$ and $\Omega$ are of class $C^{3,1}$, then $\psi \in C^{3,\beta}(\overline{\Omega})$ for all $\beta \in (0,1)$.

If moreover for some $k \in \mathbb{N}$ and $\alpha \in (0,1)$ we have $f \in C^{k,\alpha}(\overline{\Omega})$, $g \in C^{k,\alpha}(\overline{\Lambda})$ and $\Omega, \Lambda$ are of class $C^{k+2,\alpha}$, then $\psi \in C^{k+2,\alpha}(\overline{\Omega})$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.52</span><span class="math-callout__name">(Loeper–Ma–Trudinger–Wang Regularity Theory)</span></p>

Let $\mathcal{X}$, $\mathcal{Y}$ be closures of bounded connected open sets in $\mathbb{R}^n$, and $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ a smooth cost function satisfying **(STwist)** and $\mathfrak{S}\_c \ge \lambda\, \mathrm{Id}$, $\lambda > 0$, in the interior of $\mathcal{X} \times \mathcal{Y}$. Let $\Omega \subset \mathcal{X}$ and $\Lambda \subset \mathcal{Y}$ be two connected open sets, let $\mu \in P(\Omega)$ such that $d\mu/dx > 0$ almost everywhere in $\Omega$, and let $g$ be a probability density on $\Lambda$, bounded from above and below. Let $\psi$ be the unique Kantorovich potential associated with $\mu$, $\nu(dy) = g(y)\, dy$, and the cost $c$. Then:

* (i) If $\Lambda$ is $c$-convex with respect to $\Omega$ and $\exists\, m > n-1$, $\exists\, C > 0$, $\forall x \in \Omega$, $\forall r > 0$, $\mu[B\_r(x)] \le C\, r^m$, then $\psi \in C^{1,\beta}(\Omega)$ for some $\beta \in (0,1)$.
* (ii) If $\Lambda$ is uniformly $c$-convex with respect to $\Omega$ and $g \in C^{1,1}(\Lambda)$, then $\psi \in C^{3,\beta}(\Omega)$ for all $\beta \in (0,1)$. If moreover for some $k \in \mathbb{N}$, $\alpha \in (0,1)$ we have $f \in C^{k,\alpha}(\Omega)$, $g \in C^{k,\alpha}(\Lambda)$, then $\psi \in C^{k+2,\alpha}(\Omega)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 12.54</span></p>

The first part of Theorem 12.52 shows that a uniformly regular cost function behaves better, in certain ways, than the square Euclidean norm. For instance, the condition in Theorem 12.52(i) is automatically satisfied if $\mu(dx) = f(x)\, dx$, $f \in L^p$ for $p > n$; but it also allows $\mu$ to be a singular measure. (Such estimates are not even true for the linear Laplace equation!)

</div>

### Interior A Priori Estimates

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.56</span><span class="math-callout__name">(Caffarelli's Interior A Priori Estimates)</span></p>

Let $\Omega \subset \mathbb{R}^n$ be open and let $\psi : \Omega \to \mathbb{R}$ be a smooth convex function satisfying the Monge–Ampère equation $\det(\nabla^2 \psi(x)) = F(x, \nabla \psi(x))$ in $\Omega$. Let $\kappa\_\Omega(\psi)$ stand for the modulus of (strict) convexity of $\psi$ in $\Omega$. Then for any open subdomain $\Omega'$ such that $\overline{\Omega'} \subset \Omega$, one has the a priori estimates (for some $\beta \in (0,1)$, for all $\alpha \in (0,1)$, for all $k \in \mathbb{N}$):

$$\lVert \psi \rVert\_{C^{1,\beta}(\Omega')} \le C\bigl(\Omega, \Omega', \lVert F \rVert\_{L^\infty(\Omega)}, \lVert \nabla \psi \rVert\_{L^\infty(\Omega)}, \kappa_\Omega(\psi)\bigr);$$

$$\lVert \psi \rVert\_{C^{2,\alpha}(\Omega')} \le C\bigl(\alpha, \Omega, \Omega', \lVert F \rVert\_{C^{0,\alpha}(\Omega)}, \lVert \nabla \psi \rVert\_{L^\infty(\Omega)}, \kappa_\Omega(\psi)\bigr);$$

$$\lVert \psi \rVert\_{C^{k+2,\alpha}(\Omega')} \le C\bigl(k, \alpha, \Omega, \Omega', \lVert F \rVert\_{C^{k,\alpha}(\Omega)}, \lVert \nabla \psi \rVert\_{L^\infty(\Omega)}\bigr).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.57</span><span class="math-callout__name">(Loeper–Ma–Trudinger–Wang Interior A Priori Estimates)</span></p>

Let $\mathcal{X}$, $\mathcal{Y}$ be closures of bounded open sets in $\mathbb{R}^n$, and let $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ be a smooth cost function satisfying **(STwist)** and **uniformly regular** ($\mathfrak{S}\_c \ge \lambda\, \mathrm{Id}$). Let $\Omega \subset \mathcal{X}$ be a bounded open set and let $\psi : \Omega \to \mathbb{R}$ be a smooth $c$-convex solution of the generalized Monge–Ampère equation. Let $\Lambda \subset \mathcal{Y}$ be a strict neighborhood of $\lbrace (\nabla\_x c)^{-1}(x, -\nabla \psi(x));\; x \in \Omega \rbrace$, $c$-convex with respect to $\Omega$. Then for any open subset $\Omega' \subset \Omega$ such that $\overline{\Omega'} \subset \Omega$, one has the a priori estimates (for some $\beta \in (0,1)$, for all $\alpha \in (0,1)$, for all $k \ge 2$):

$$\lVert \psi \rVert\_{C^{1,\beta}(\Omega')} \le C\bigl(\Omega, \Omega', c\rvert\_{\Omega \times \Lambda}, \lVert F \rVert\_{L^\infty(\Omega)}, \lVert \nabla \psi \rVert\_{L^\infty(\Omega)}\bigr);$$

$$\lVert \psi \rVert\_{C^{3,\alpha}(\Omega')} \le C\bigl(\alpha, \Omega, \Omega', c\rvert\_{\Omega \times \Lambda}, \lVert F \rVert\_{C^{1,1}(\Omega)}, \lVert \nabla \psi \rVert\_{L^\infty(\Omega)}\bigr);$$

$$\lVert \psi \rVert\_{C^{k+2,\alpha}(\Omega')} \le C\bigl(k, \alpha, \Omega, \Omega', c\rvert\_{\Omega \times \Lambda}, \lVert F \rVert\_{C^{k,\alpha}(\Omega)}, \lVert \nabla \psi \rVert\_{L^\infty(\Omega)}\bigr).$$

</div>

### Smoothness of Optimal Transport on $S^{n-1}$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12.58</span><span class="math-callout__name">(Smoothness of Optimal Transport on $S^{n-1}$)</span></p>

Let $S^{n-1}$ be the unit Euclidean sphere in $\mathbb{R}^n$, equipped with its volume measure, and let $d$ be the geodesic distance on $S^{n-1}$. Let $f$ and $g$ be $C^{1,1}$ positive probability densities on $S^{n-1}$. Let $\psi$ be the unique (up to an additive constant) Kantorovich potential associated with the transport of $\mu(dx) = f(x)\, \mathrm{vol}(dx)$ to $\nu(dy) = g(y)\, \mathrm{vol}(dy)$ with cost $c(x,y) = d(x,y)^2$, and let $T$ be the optimal transport map. Then $\psi \in C^{3,\beta}(S^{n-1})$ for all $\beta \in (0,1)$, and in particular $T \in C^{2,\beta}(S^{n-1}, S^{n-1})$.

If moreover $f, g$ lie in $C^{k,\alpha}(S^{n-1})$ for some $k \in \mathbb{N}$, $\alpha \in (0,1)$, then $\psi \in C^{k+2,\alpha}(S^{n-1})$ and $T \in C^{k+1,\alpha}(S^{n-1}, S^{n-1})$. (In particular, if $f$ and $g$ are positive and $C^\infty$ then $\psi$ and $T$ are $C^\infty$.)

</div>

### Bibliographical Notes on Chapter 12

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History of Regularity Theory)</span></p>

The denomination **Monge–Ampère equation** is used for any equation resembling $\det(\nabla^2 \psi) = f$; the link between the Monge problem and Monge–Ampère equations was made by **Knott and Smith**, then popularized by **Brenier**. Weak solutions constructed via optimal transport are often called *Brenier solutions*. **Caffarelli** showed that for a convex target, Brenier's notion is equivalent to the older concepts of Alexandrov solution and viscosity solution. The modern regularity theory was pioneered by **Alexandrov** and **Pogorelov**.

The pioneering papers for the Monge–Ampère second boundary value problem are due to **Delanoë** (dimension 2), then **Caffarelli**, **Urbas**, and **X.-J. Wang** (arbitrary dimension). At this point **Loeper** made three crucial contributions: (1) the very strong estimates in Theorem 12.52(i) under the strict MTW condition **(A3s)**; (2) the geometric interpretation of the MTW condition as the regularity property (Definition 12.14), related to sectional curvature (Particular Case 12.30); (3) the proof that $\mathfrak{S}\_c \ge 0$ is *mandatory* to derive regularity (Theorem 12.21).

The converse implication in Theorem 12.36 was proven independently by **Trudinger and Wang** on the one hand, and by **Y.-H. Kim and McCann** on the other. Kim and McCann developed a framework identifying the MTW tensor as the sectional curvature tensor of the pseudo-Riemannian metric $\partial^2 c / \partial x\, \partial y$ on the product manifold.

**Loeper** proved that the squared distance on the sphere is a uniformly regular cost, and combined all the above elements to derive Theorem 12.58. The cut locus is also a major issue in the study of perturbation of these smoothness results; a stability problem first formulated in [572] is solved by **Figalli and Rifford** near $S^2$.

</div>

## Chapter 13: Qualitative Picture

This chapter synthesizes the whole picture of optimal transport on a smooth Riemannian manifold $M$. A good understanding of this chapter is sufficient to attack Part II. Three settings are considered: (i) general $C^2$ Lagrangian on a compact manifold; (ii) $L = \lvert v \rvert^2/2$ on a complete manifold (cost $c = d^2/2$); (iii) $L = \lvert v \rvert^2/2$ in $\mathbb{R}^n$ (cost $c = \lvert x - y \rvert^2/2$).

### Recap of the Main Results

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Summary of Optimal Transport — Nine Key Facts)</span></p>

**1. Existence.** There always exists an optimal coupling $(x\_0, x\_1)$ with law $\pi$, a displacement interpolation $(\mu\_t)$, and a random minimizing curve $\gamma$ with law $\Pi$, such that $\mathrm{law}(\gamma\_t) = \mu\_t$ and $\mathrm{law}(\gamma\_0, \gamma\_1) = \pi$. Each $\gamma$ solves the Euler–Lagrange equation. Two trajectories in $\mathrm{Spt}\, \Pi$ may intersect at $t = 0$ or $t = 1$, but **never at intermediate times**.

**2. Absolute continuity.** If either $\mu\_0$ or $\mu\_1$ is absolutely continuous, then so is $\mu\_t$ for all $t \in (0,1)$.

**3. Unique deterministic coupling.** If $\mu\_0$ is absolutely continuous, the optimal coupling is unique, deterministic ($x\_1 = T(x\_0)$), and characterized by $\nabla \psi(x\_0) = -\nabla\_x c(x\_0, x\_1) = \nabla\_v L(x\_0, \dot{\gamma}\_0, 0)$, where $\psi$ is $c$-convex.
* Quadratic cost on $M$: $T\_t(x) = \exp\_x(t\, \nabla \psi(x))$.
* Quadratic cost in $\mathbb{R}^n$: $T\_t(x) = (1-t)x + t\, \nabla \Psi(x)$, where $\Psi = \lvert \cdot \rvert^2/2 + \psi$ is convex l.s.c.

**4. Uniqueness of displacement interpolation.** $(\mu\_t)$ is unique.

**5. Dual prices.** $\phi(y) = \inf\_x [\psi(x) + c(x,y)]$ and $\psi(x) = \sup\_y [\phi(y) - c(x,y)]$, with $\phi(x\_1) - \psi(x\_0) = c(x\_0, x\_1)$ a.s.

**6.** Two minimizing curves may meet at $t = 0$ or $t = 1$ on a set of dimension at most $n - 1$.

**7. Hamilton–Jacobi evolution.** Replacing $\mu\_0$ by $\mu\_t$, $\psi$ is replaced by $\psi\_t(y) = \inf\_x [\psi\_0(x) + c^{0,t}(x,y)]$, a viscosity solution of $\partial\_t \psi\_t + L^\ast(x, \nabla \psi\_t, t) = 0$.

**8. Transport map formula.** $T\_t(x)$ is the solution at time $t$ of the Euler–Lagrange equation starting from $x$ with velocity $v\_0(x) = (\nabla\_v L(x, \cdot, 0))^{-1}(\nabla \psi(x))$.

**9. Optimal cost formula.** $\int \psi\_{t\_1}\, d\mu\_{t\_1} - \int \psi\_{t\_0}\, d\mu\_{t\_0} = C^{t\_0, t\_1}(\mu\_{t\_0}, \mu\_{t\_1})$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Open Problem 13.1</span></p>

If the initial and final densities $\rho\_0$ and $\rho\_1$ are positive everywhere, does this imply that the intermediate densities $\rho\_t$ are also positive? Otherwise, can one identify simple sufficient conditions for the density of the displacement interpolant to be positive everywhere?

</div>

### Standard Approximation Procedure

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13.2</span><span class="math-callout__name">(Standard Approximation Scheme)</span></p>

Let $M$, $c$, $\mu\_0$, $\mu\_1$, $\pi$, $(\mu\_t)$, $\Pi$ be as above. Let $(K\_\ell)$ be compact sets in $\Gamma$ with $\Pi[\cup K\_\ell] = 1$. Define $\Pi\_\ell := 1\_{K\_\ell} \Pi / \Pi[K\_\ell]$. Then each $(\mu\_{t,\ell})$ is a compactly supported displacement interpolation, $\pi\_\ell$ is optimal, and the monotone convergences $Z\_\ell \pi\_\ell \uparrow \pi$, $Z\_\ell \mu\_{t,\ell} \uparrow \mu\_t$, $Z\_\ell \Pi\_\ell \uparrow \Pi$ hold. If $\mu\_0$ is absolutely continuous, then $\mu\_{0,\ell}$ is absolutely continuous, $\pi\_\ell$ is deterministic, and $T\_{t\_0 \to t, \ell} = T\_{t\_0 \to t}$ for $t\_0 \in (0,1)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13.4</span><span class="math-callout__name">(Regularization of Singular Transport Problems)</span></p>

Let $\mu\_0$ and $\mu\_1$ be two probability measures on $M$ with finite optimal transport cost, and $\pi$ an optimal transference plan. Then there are sequences $(\mu\_0^k)$, $(\mu\_1^k)$ and $(\pi^k)$ such that each $\pi^k$ is an optimal transference plan between $\mu\_0^k$ and $\mu\_1^k$ (each having a smooth, compactly supported density); and $\mu\_0^k \to \mu\_0$, $\mu\_1^k \to \mu\_1$, $\pi^k \to \pi$ weakly.

</div>

### Equations of Displacement Interpolation

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Equations of Displacement Interpolation)</span></p>

**General Lagrangian:**

$$\begin{cases} \partial_t \mu_t + \nabla \cdot (\xi_t\, \mu_t) = 0; \quad \nabla_v L(x, \xi_t(x), t) = \nabla \psi_t(x); \\\\ \psi_0 \text{ is } c\text{-convex}; \quad \partial_t \psi_t + L^*(x, \nabla \psi_t(x), t) = 0. \end{cases}$$

**Quadratic cost on a Riemannian manifold:**

$$\begin{cases} \partial_t \mu_t + \nabla \cdot (\nabla \psi_t\, \mu_t) = 0; \quad \psi_0 \text{ is } d^2/2\text{-convex}; \quad \partial_t \psi_t + \lvert \nabla \psi_t \rvert^2 / 2 = 0. \end{cases}$$

**Quadratic cost in $\mathbb{R}^n$:** The same, with $\lvert x \rvert^2/2 + \psi\_0(x)$ convex l.s.c. Apart from the initial datum, this is the **pressureless Euler equation** for a potential velocity field.

</div>

### Quadratic Cost Function and $d^2/2$-Convexity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13.5</span><span class="math-callout__name">($C^2$-Small Functions are $d^2/2$-Convex)</span></p>

Let $M$ be a Riemannian manifold, and let $K$ be a compact subset of $M$. Then, there is $\varepsilon > 0$ such that any function $\psi \in C\_c^2(M)$ satisfying $\mathrm{Spt}(\psi) \subset K$ and $\lVert \psi \rVert\_{C\_b^2} \le \varepsilon$ is $d^2/2$-convex. In $\mathbb{R}^n$, $\psi$ is $d^2/2$-convex if $\nabla^2 \psi \ge -I\_n$ (no need for compact support).

</div>

### The Structure of $P\_2(M)$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Otto's Riemannian Structure on $P\_2(M)$)</span></p>

A striking discovery by **Otto**: the differentiable structure on $M$ induces a kind of differentiable structure on $P\_2(M)$. The idea: the path $(\mu\_t)$ is determined by the initial velocity field $\nabla \psi$, which plays the role of an "initial velocity" for the curve in $P\_2(M)$. If $\mu$ is absolutely continuous, the **tangent cone** $T\_\mu P\_2(M)$ can be identified isometrically with the closed vector space generated by $d^2/2$-convex functions $\psi$, equipped with $\lVert \nabla \psi \rVert\_{L^2(\mu; TM)}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13.8</span><span class="math-callout__name">(Representation of Lipschitz Paths in $P\_2(M)$)</span></p>

Let $M$ be a smooth complete Riemannian manifold, let $(\mu\_t)\_{0 \le t \le 1}$ be a Lipschitz-continuous path in $P\_2(M)$: $W\_2(\mu\_s, \mu\_t) \le L\, \lvert t - s \rvert$. For any $t \in [0,1]$, let $H\_t := \overline{\mathrm{Vect}(\lbrace \nabla \psi;\; \psi \in C\_c^1(M) \rbrace)}^{L^2(\mu\_t; TM)}$. Then there exists a measurable vector field $\xi\_t(x) \in L^\infty(dt; L^2(d\mu\_t(x)))$, $\mu\_t(dx)\, dt$-a.e. unique, such that $\xi\_t \in H\_t$ for all $t$ (the velocity field is truly tangent along the path), and

$$\partial_t \mu_t + \nabla \cdot (\xi_t\, \mu_t) = 0$$

in the weak sense. Conversely, if $(\mu\_t)$ satisfies this equation for some vector field with $L^2(\mu\_t)$-norm bounded by $L$ a.s. in $t$, then $(\mu\_t)$ is Lipschitz with $\lVert \dot{\mu} \rVert \le L$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Riemannian-Like Formulas)</span></p>

$$W_2(\mu_0, \mu_1)^2 = \inf \int_0^1 \lVert \dot{\mu}_t \rVert^2\_{T\_{\mu_t} P_2}\, dt, \qquad \lVert \dot{\mu} \rVert^2\_{T_\mu P_2} = \inf \left\lbrace \int \lvert v \rvert^2\, d\mu;\; \dot{\mu} + \nabla \cdot (v\mu) = 0 \right\rbrace.$$

This has the formal structure of a Riemannian distance: $P\_2(M)$ can be viewed as an **infinite-dimensional Riemannian manifold** whose metric tensor at $\mu$ is the $L^2(\mu)$ norm on gradient vector fields.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 13.10</span><span class="math-callout__name">(Physical Interpretation)</span></p>

Imagine observing the infinitesimal evolution of the density of particles moving in a continuum, without knowing the actual velocities. Among all velocity fields compatible with the observed density evolution (solutions of the continuity equation), select the one with **minimum kinetic energy**. This energy is (up to a factor $1/2$) the square norm of your infinitesimal evolution in $P\_2(M)$.

</div>

### Bibliographical Notes on Chapter 13

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History)</span></p>

The formula (13.8) appears in **Cordero-Erausquin, McCann and Schmuckenschläger** and has the consequence that on a Riemannian manifold, optimal transport starting from an absolutely continuous measure **almost never hits the cut locus**.

**Otto** took the conceptual step of viewing $P\_2(M)$ formally as an infinite-dimensional Riemannian manifold. For some time this was used as a purely formal but powerful heuristic. Rigorous constructions were performed by **Ambrosio, Gigli and Savaré** (in $\mathbb{R}^n$); a more geometric treatment is due to **Lott**, who established "explicit" formulas for the Riemannian connection and curvature in $P\_2^{\mathrm{ac}}(M)$.

The pressureless Euler equations describe the evolution of "sticky particles." **Khesin** suggested the problem of characterizing the "time of the first shock" for a geodesic in $P\_2(\mathbb{R}^n)$.

</div>

## Part II: Optimal Transport and Riemannian Geometry

This second part explores Riemannian geometry through optimal transport. The geometry of a manifold influences the qualitative properties of optimal transport; this can be quantified by the effect of **Ricci curvature bounds** on the convexity of certain well-chosen functionals along displacement interpolation.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Overview</span><span class="math-callout__name">(Part II Structure)</span></p>

* **Chapter 14:** Self-contained exposition of **Ricci curvature** (Jacobian determinant of the exponential map, Bochner's formula).
* **Chapter 15:** Formal differential calculus on the Wasserstein space (Otto's calculus).
* **Chapters 16–17:** Relations between **displacement convexity** and **Ricci curvature** — Ricci bounds imply displacement convexity, and conversely.
* **Chapters 18–22:** Classical properties of Riemannian manifolds derived from Ricci curvature via displacement convexity: volume growth, Sobolev inequalities, concentration inequalities, Poincaré inequalities.
* **Chapter 23:** **Gradient flows** in the Wasserstein space (recovering the heat equation).
* **Chapters 24–25:** Functional inequalities applied to gradient flows, and conversely.

**Convention:** Throughout Part II, a "Riemannian manifold" is a *smooth, complete connected finite-dimensional Riemannian manifold distinct from a point, equipped with a smooth metric tensor*.

</div>

## Chapter 14: Ricci Curvature

This chapter provides a crash course on Ricci curvature, sufficient for the rest of the book. In practice, Ricci curvature appears from two complementary points of view: (a) estimates of the **Jacobian determinant of the exponential map**; (b) **Bochner's formula**. Both will be needed.

### Curvature: Overview

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Curvature Hierarchy)</span></p>

The most popular curvatures on a Riemannian manifold $(M, g)$ are:

* **Sectional curvature** $\sigma$: for each point $x$ and each plane $P \subset T\_x M$, $\sigma\_x(P)$ is a number. The most precise: knowledge of all sectional curvatures is equivalent to knowledge of the full Riemann curvature tensor.
* **Ricci curvature** $\mathrm{Ric}$: for each point $x$, $\mathrm{Ric}\_x$ is a quadratic form on $T\_x M$. If $e$ is a unit vector and $(e, e\_2, \ldots, e\_n)$ an orthonormal basis, then $\mathrm{Ric}\_x(e, e) = \sum\_{j=2}^n \sigma\_x(P\_j)$, where $P\_j$ is the plane spanned by $\lbrace e, e\_j \rbrace$.
* **Scalar curvature** $S$: for each point $x$, $S\_x$ is a number (the trace of $\mathrm{Ric}\_x$).

A control on sectional curvature is stronger than a control on Ricci, which is stronger than a control on scalar curvature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gauss Curvature and Geodesic Spreading)</span></p>

For a surface (dimension 2), all three notions reduce to the **Gauss curvature**. The intrinsic nature of Gauss curvature (Gauss's *Theorema Egregium*) means it can be computed purely from the metric, without knowing the embedding.

Curvature is intimately related to the **local behavior of geodesics**: positive curvature makes geodesics converge, negative curvature makes them diverge. If two geodesics start from $x$ with unit speed and respective velocities $v, w$ at angle $\theta$, the distance $\delta(t)$ between them at time $t$ satisfies

$$\delta(t) = \sqrt{2(1-\cos\theta)}\, t \left(1 - \frac{\kappa_x \cos^2(\theta/2)}{6}\, t^2 + O(t^4)\right),$$

where $\kappa\_x$ is the Gauss curvature at $x$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Constant Curvature Spaces)</span></p>

* $S^n(R)$ (sphere of radius $R$): constant sectional curvature $1/R^2$.
* $\mathbb{R}^n$ (Euclidean space): curvature $0$.
* $\mathbb{H}^n(R)$ (hyperbolic space): constant sectional curvature $-1/R^2$.

These three families are the only simply connected Riemannian manifolds with constant sectional curvature, and they play the role of comparison spaces.

</div>

### Preliminary: Second-Order Differentiation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hessian and Laplacian on Riemannian Manifolds)</span></p>

The **Hessian operator** of a function $f$ on a Riemannian manifold $M$ at point $x$ is the linear operator $\nabla^2 f(x) : T\_x M \to T\_x M$ defined by

$$\nabla^2 f \cdot v = \nabla_v(\nabla f).$$

Equivalently, for a geodesic $(\gamma\_t)$ with $\gamma\_0 = x$ and $\dot{\gamma}\_0 = v$:

$$f(\gamma_t) = f(x) + t\, \langle \nabla f(x), v \rangle + \frac{t^2}{2}\, \langle \nabla^2 f(x) \cdot v, v \rangle + o(t^2).$$

If $f \in C^2(M)$, then $\nabla^2 f(x)$ is a symmetric operator. The **Laplacian** (Laplace–Beltrami operator) is

$$\Delta f(x) = \mathrm{tr}(\nabla^2 f(x)) = \nabla \cdot (\nabla f).$$

In coordinates: $\Delta f = (\det g)^{-1/2} \sum\_{ij} \partial\_i\bigl((\det g)^{1/2}\, g^{ij}\, \partial\_j f\bigr)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.1</span><span class="math-callout__name">(Alexandrov's Second Differentiability Theorem)</span></p>

Let $M$ be a Riemannian manifold equipped with its volume measure, let $U$ be an open subset of $M$, and let $\psi : U \to \mathbb{R}$ be locally semiconvex with a quadratic modulus of semiconvexity. Then, for almost every $x \in U$, $\psi$ is differentiable at $x$ and there exists a symmetric operator $A : T\_x M \to T\_x M$ (the **Hessian** $\nabla^2 \psi(x)$) such that:

* (i) For any $v \in T\_x M$, $\nabla\_v(\nabla \psi)(x) = A v$;
* (ii) $\psi(\exp\_x v) = \psi(x) + \langle \nabla \psi(x), v \rangle + \frac{\langle A \cdot v, v \rangle}{2} + o(\lvert v \rvert^2)$ as $v \to 0$.

The trace of $A$ is the Laplacian $\Delta \psi(x)$, which coincides with the density of the absolutely continuous part of the distributional Laplacian; the singular part is a nonneg. measure.

</div>

### The Jacobian Determinant of the Exponential Map

Let $\xi$ be a vector field on $M$, $T = \exp \xi$ (i.e. $T(x) = \exp\_x \xi(x)$), and $T\_t = \exp(t\xi)$. Define $\mathbf{J}(t, x) = (J\_1(t,x), \ldots, J\_n(t,x))$ where $J\_i(t,x) := \frac{d}{d\delta}\rvert\_{\delta=0} T\_t(x + \delta e\_i)$ are **Jacobi fields** along the geodesic $\gamma(t) = T\_t(x)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jacobi Fields and Jacobi Equation)</span></p>

The Jacobi fields $J\_i$ satisfy the **Jacobi equation**:

$$\ddot{J}(t) + R(t)\, J(t) = 0,$$

where $R(t)$ is the matrix with entries $R\_{ij}(t) = \langle \mathrm{Riem}\_{\gamma(t)}(\dot{\gamma}(t), e\_i(t))\, \dot{\gamma}(t),\; e\_j(t) \rangle\_{\gamma(t)}$, and $\mathbf{E}(t) = (e\_1(t), \ldots, e\_n(t))$ is the parallel transport of an initial orthonormal basis along $\gamma$. The initial conditions are

$$J(0) = I_n, \qquad \dot{J}(0) = \nabla \xi(x).$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jacobian, Distortion, and Relative Velocity)</span></p>

The **Jacobian determinant** is $\mathcal{J}(t, x) = \det J(t, x)$. The **relative velocity matrix** is

$$U(t) := \dot{J}(t)\, J(t)^{-1}.$$

Differentiating (14.6) gives the **matrix Riccati equation**:

$$\dot{U} + U^2 + R = 0.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fundamental Differential Inequality Involving Ricci Curvature)</span></p>

Let $\xi = \nabla \psi$ where $\psi$ is locally semiconvex with quadratic modulus. Then $U(t, x) = \nabla^2 \psi(x)$ is **symmetric** (at $t = 0$), and by the Cauchy–Lipschitz uniqueness, $U(t, x)$ remains symmetric for all $t$. Taking the trace of the Riccati equation and applying the Cauchy–Schwarz inequality $\mathrm{tr}(U^2) \ge (\mathrm{tr}\, U)^2 / n$:

$$\frac{d}{dt}(\mathrm{tr}\, U) + \frac{(\mathrm{tr}\, U)^2}{n} + \mathrm{Ric}(\dot{\gamma}) \le 0.$$

This can be rewritten in several equivalent forms. Defining:
* $\mathcal{D}(t) := \mathcal{J}(t)^{1/n}$ (coefficient of **mean distortion**);
* $\ell(t) := -\log \mathcal{J}(t)$;

the inequality becomes

$$\frac{\ddot{\mathcal{D}}}{\mathcal{D}} \le -\frac{\mathrm{Ric}(\dot{\gamma})}{n}, \qquad \text{or equivalently} \qquad \ddot{\ell} \ge \frac{\dot{\ell}^2}{n} + \mathrm{Ric}(\dot{\gamma}).$$

</div>

### Taking Out the Direction of Motion

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Parallel and Orthogonal Decomposition)</span></p>

Since $R(t)\, \dot{\gamma}(t) = 0$, curvature is not felt in the direction of motion. Decompose the Jacobian as $\mathcal{J} = \mathcal{J}\_{//}\, \mathcal{J}\_\perp$, where $\mathcal{J}\_{//}(t) = \exp(\int\_0^t u\_{//}(s)\, ds)$ and $\mathcal{J}\_\perp$ is the orthogonal part. Define orthogonal distortions $\mathcal{D}\_\perp = \mathcal{J}\_\perp^{1/(n-1)}$ and $\ell\_\perp = -\log \mathcal{J}\_\perp$. Then:

* **Parallel direction:** $\ddot{\mathcal{J}}\_{//} \le 0$ (always concave, independent of curvature).
* **Orthogonal direction:** $\ddot{\ell}\_\perp \ge \frac{\dot{\ell}\_\perp^2}{n-1} + \mathrm{Ric}(\dot{\gamma})$, and $\frac{\ddot{\mathcal{D}}\_\perp}{\mathcal{D}\_\perp} \le -\frac{\mathrm{Ric}(\dot{\gamma})}{n-1}$.

The basic inequalities for $\ell\_\perp$ and $\ell\_{//}$ are the same as for $\ell$, but with the exponent $n$ replaced by $n-1$ in the case of $\ell\_\perp$ and $1$ in the case of $\ell\_{//}$; and $\mathrm{Ric}(\dot{\gamma})$ replaced by $0$ in the case of $\ell\_{//}$.

</div>

### Positivity of the Jacobian

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Blowup and Bonnet–Myers)</span></p>

The Jacobian $\mathcal{J}(t)$ may vanish at some time (the exponential map becomes singular). Since $\ell$ solves a Riccati-type equation $\ddot{\ell} \ge \dot{\ell}^2/(n-1) + K$ (when $\mathrm{Ric} \ge K g$), the function $\ell$ must blow up in finite time $T = \pi\sqrt{(n-1)/K}$. This implies the **Bonnet–Myers theorem**: the diameter of $M$ cannot be larger than $\pi\sqrt{(n-1)/K}$ if $\mathrm{Ric} \ge K g$ with $K > 0$.

However, in the case of **optimal transport** ($\xi = \nabla \psi$ with $\psi$ being $d^2/2$-convex), the Jacobian **cannot vanish at intermediate times** for almost all initial points. This is a consequence of the non-crossing property (Problem 8.8, Theorem 11.3).

</div>

### Bochner's Formula

The Lagrangian viewpoint follows geodesic paths $\gamma(t)$, keeping the memory of the initial position. The **Eulerian** viewpoint focuses on the velocity field $\xi = \xi(t, x)$, where $\dot{\gamma}(t) = \xi(t, \gamma(t))$. The pressureless Euler equation is $\partial \xi / \partial t + \nabla\_\xi \xi = 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bochner–Weitzenböck–Lichnerowicz Formula)</span></p>

For any smooth ($C^2$) vector field $\xi$ on a Riemannian manifold $M$:

$$-\nabla \cdot (\nabla_\xi \xi) + \xi \cdot \nabla(\nabla \cdot \xi) + \mathrm{tr}\,(\nabla \xi)^2 + \mathrm{Ric}(\xi) = 0.$$

If $\nabla \xi$ is symmetric (i.e. $\xi = \nabla \psi$ for some function $\psi$), this simplifies to:

$$-\Delta \frac{\lvert \nabla \psi \rvert^2}{2} + \nabla \psi \cdot \nabla(\Delta \psi) + \lVert \nabla^2 \psi \rVert_{\mathrm{HS}}^2 + \mathrm{Ric}(\nabla \psi) = 0.$$

Using the Cauchy–Schwarz inequality $\lVert \nabla^2 \psi \rVert\_{\mathrm{HS}}^2 \ge (\Delta \psi)^2/n$:

$$\Delta \frac{\lvert \nabla \psi \rvert^2}{2} - \nabla \psi \cdot \nabla(\Delta \psi) \ge \frac{(\Delta \psi)^2}{n} + \mathrm{Ric}(\nabla \psi).$$

This inequality is strictly equivalent (modulo regularity) to the Lagrangian Jacobian inequality (14.13)/(14.14)/(14.15).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 14.5</span><span class="math-callout__name">(Hamilton–Jacobi Equation)</span></p>

With the ansatz $\xi = \nabla \psi$, the pressureless Euler equation reduces to the **Hamilton–Jacobi equation** $\partial\_t \psi + \lvert \nabla \psi \rvert^2/2 = 0$, whose solution is given by the Hopf–Lax formula $\psi(t, x) = \inf\_{y \in M} [\psi(y) + d(x,y)^2/(2t)]$. The geodesic curves $\gamma$ starting with $\dot{\gamma}(0) = \nabla \psi(x)$ are the **characteristic curves** of this equation.

</div>

### Analytic and Geometric Consequences of Ricci Curvature Bounds

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Classical Consequences of $\mathrm{Ric} \ge K$)</span></p>

Assume $M$ is an $n$-dimensional Riemannian manifold with $\mathrm{Ric} \ge K$. Then:

**1. Volume growth (Bishop–Gromov inequality).** The ratio $\mathrm{vol}[B\_r(x)] / V(r)$ is nonincreasing in $r$, where $V(r)$ is the volume of a ball of radius $r$ in the model space of constant Ricci curvature $K$ and dimension $n$. The surface area function of the model space is

$$S(r) = c_{n,K} \begin{cases} \sin^{n-1}\!\bigl(\sqrt{K/(n-1)}\, r\bigr) & \text{if } K > 0 \\\\ r^{n-1} & \text{if } K = 0 \\\\ \sinh^{n-1}\!\bigl(\sqrt{\lvert K \rvert/(n-1)}\, r\bigr) & \text{if } K < 0. \end{cases}$$

**2. Bonnet–Myers theorem.** If $K > 0$, then $M$ is compact and $\mathrm{diam}(M) \le \pi\sqrt{(n-1)/K}$.

**3. Spectral gap.** If $K > 0$, the first nonzero eigenvalue $\lambda\_1$ of $-\Delta$ satisfies $\lambda\_1 \ge nK/(n-1)$, with equality for the model sphere.

**4. Sharp Sobolev inequalities.** If $K > 0$ and $n \ge 3$, then $\lVert f \rVert\_{L^{2^\ast}(\mu)}^2 \le \lVert f \rVert\_{L^2(\mu)}^2 + \frac{4}{Kn(n-2)} \lVert \nabla f \rVert\_{L^2(\mu)}^2$, where $2^\ast = 2n/(n-2)$ and $\mu = \mathrm{vol}/\mathrm{vol}[M]$.

**5. Heat kernel bounds (Li–Yau estimates).** If $K \ge 0$, then $p\_t(x,y) \le \frac{C}{\mathrm{vol}[B\_{\sqrt{t}}(x)]} \exp(-d(x,y)^2/(2Ct))$.

**6. Ricci flow.** Hamilton's equation $\partial g/\partial t = -2\,\mathrm{Ric}(g)$ is a "heat flow" in the space of metrics. Perelman used it to prove the Poincaré conjecture, showing that a compact simply connected 3-manifold with positive Ricci curvature is diffeomorphic to $S^3$.

**7. Splitting theorem (Cheeger–Gromoll).** If $\mathrm{Ric} \ge 0$ and $M$ contains a line (a geodesic minimizing for all time $t \in \mathbb{R}$), then $M \cong \mathbb{R} \times M'$.

</div>

### Change of Reference Measure and Effective Dimension

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Ricci Tensor $\mathrm{Ric}\_{N,\nu}$)</span></p>

Let $\nu(dx) = e^{-V(x)}\, \mathrm{vol}(dx)$ be a reference measure on $M$ (with $V \in C^2$). The **generalized Ricci tensor** with effective dimension $N \ge n$ is

$$\mathrm{Ric}\_{N,\nu} := \mathrm{Ric} + \nabla^2 V - \frac{\nabla V \otimes \nabla V}{N - n},$$

where $(\nabla V \otimes \nabla V)\_x(v) = (\nabla V(x) \cdot v)^2$. Conventions: if $N = n$ then $\mathrm{Ric}\_{n,\mathrm{vol}} = \mathrm{Ric}$ (and $V$ must be $0$); if $N = \infty$ then $\mathrm{Ric}\_{\infty,\nu} = \mathrm{Ric} + \nabla^2 V$.

The Jacobian determinant with respect to $\nu$ becomes $\mathcal{J}(t,x) = e^{-V(T\_t(x))+V(x)} \mathcal{J}\_0(t,x)$, and the fundamental differential inequality generalizes to

$$\ddot{\ell} \ge \frac{\dot{\ell}^2}{N} + \mathrm{Ric}\_{N,\nu}(\dot{\gamma}), \qquad -N\frac{\ddot{\mathcal{D}}}{\mathcal{D}} \ge \mathrm{Ric}\_{N,\nu}(\dot{\gamma}).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gaussian Measure)</span></p>

The most famous example is the **Gaussian measure** $\gamma^{(n)}(dx) = e^{-\lvert x \rvert^2}/(2\pi)^{n/2}\, dx$ on $\mathbb{R}^n$. The effective dimension of $(\mathbb{R}^n, \gamma^{(n)})$ is infinite, in a certain sense, whatever $n$. Most theorems about the Gaussian measure can be written the same in dimension 1 or in dimension $n$, or even in infinite dimension.

</div>

### Generalized Bochner Formula and $\Gamma\_2$ Formalism

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\Gamma$ and $\Gamma\_2$ Operators)</span></p>

Given the **modified Laplace operator** $L := \Delta - \nabla V \cdot \nabla$, define:

* The **$\Gamma$ operator** (*carré du champ*): $\Gamma(f, g) = \frac{1}{2}[L(fg) - f\, Lg - g\, Lf] = \nabla f \cdot \nabla g$.
* The **$\Gamma\_2$ operator** (*carré du champ itéré*): $\Gamma\_2(f, g) = \frac{1}{2}[L\Gamma(fg) - \Gamma(f, Lg) - \Gamma(g, Lf)]$.

The key formula is $\Gamma\_2(\psi) := \Gamma\_2(\psi, \psi) = L\frac{\lvert \nabla \psi \rvert^2}{2} - \nabla \psi \cdot \nabla(L\psi)$, which gives

$$\Gamma_2(\psi) = \frac{(L\psi)^2}{N} + \mathrm{Ric}\_{N,\nu}(\nabla \psi) + \left(\lVert \nabla^2 \psi - \frac{\Delta \psi}{n}\, I_n \rVert_{\mathrm{HS}}^2 + \frac{n}{N(N-n)} \left[\frac{N-n}{n}\, \Delta \psi + \nabla V \cdot \nabla \psi\right]^2\right).$$

</div>

### Curvature-Dimension Bounds

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.8</span><span class="math-callout__name">($\mathrm{CD}(K, N)$ Curvature-Dimension Bound)</span></p>

Let $M$ be a Riemannian manifold of dimension $n$, and let $K \in \mathbb{R}$, $N \in [n, \infty]$. Then the following conditions are all equivalent if they are required to hold for arbitrary data; when they are fulfilled, $M$ is said to satisfy the $\mathrm{CD}(K, N)$ **curvature-dimension bound**:

* (i) $\mathrm{Ric}\_{N,\nu} \ge K$;
* (ii) $\Gamma\_2(\psi) \ge \frac{(L\psi)^2}{N} + K\, \lvert \nabla \psi \rvert^2$ (for all smooth $\psi$);
* (iii) $\ddot{\ell} \ge \frac{\dot{\ell}^2}{N} + K\, \lvert \dot{\gamma} \rvert^2$ (for all semiconvex $\psi$, a.e. $x$, all $t \in (0,1)$).

If $N < \infty$, these are also equivalent to:

* (iv) $\ddot{\mathcal{D}} + \frac{K \lvert \dot{\gamma} \rvert^2}{N}\, \mathcal{D} \le 0$.

Moreover, the corresponding "orthogonal" versions (ii'), (iii'), (iv') hold with $N$ replaced by $N-1$ in the exponent and $\mathrm{Ric}\_{N,\nu}$ on the right.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 14.10</span><span class="math-callout__name">(One-Dimensional $\mathrm{CD}(K, N)$ Model Spaces)</span></p>

* (a) $K > 0$, $1 < N < \infty$: $M = (-\sqrt{(N-1)/K}\, \pi/2,\; \sqrt{(N-1)/K}\, \pi/2)$, $\nu(dx) = \cos^{N-1}(\sqrt{K/(N-1)}\, x)\, dx$.
* (b) $K < 0$, $1 \le N < \infty$: $M = \mathbb{R}$, $\nu(dx) = \cosh^{N-1}(\sqrt{\lvert K \rvert/(N-1)}\, x)\, dx$.
* (c) $K = 0$, $N \in [1, \infty)$: $M = (0, +\infty)$, $\nu(dx) = x^{N-1}\, dx$.
* (d) $K \in \mathbb{R}$, $N = \infty$: $M = \mathbb{R}$, $\nu(dx) = e^{-Kx^2/2}\, dx$. This satisfies $\mathrm{CD}(K, \infty)$.

</div>

### From Differential to Integral Curvature-Dimension Bounds

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.11</span><span class="math-callout__name">(Integral Reformulation of $\mathrm{CD}(K, N)$)</span></p>

Let $M$ be a Riemannian manifold with a reference measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, and let $d$ be the geodesic distance. Then $M$ satisfies $\mathrm{CD}(K, N)$ if and only if the following inequality holds (for any semiconvex $\psi$, almost any $x$, as soon as $\mathcal{J}(t,x)$ does not vanish for $t \in (0,1)$):

**For $N < \infty$:**

$$\mathcal{D}(t, x) \ge \tau_{K,N}^{(1-t)}\, \mathcal{D}(0, x) + \tau_{K,N}^{(t)}\, \mathcal{D}(1, x),$$

where $\alpha = \sqrt{\lvert K \rvert/N}\, d(x, y)$ and

$$\tau_{K,N}^{(t)} = \begin{cases} \sin(t\alpha)/\sin\alpha & \text{if } K > 0 \\\\ t & \text{if } K = 0 \\\\ \sinh(t\alpha)/\sinh\alpha & \text{if } K < 0. \end{cases}$$

**For $N = \infty$:**

$$\ell(t, x) \le (1-t)\, \ell(0, x) + t\, \ell(1, x) - \frac{Kt(1-t)}{2}\, d(x,y)^2.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.12</span><span class="math-callout__name">($\mathrm{CD}(K, N)$ with Direction of Motion Taken Out)</span></p>

Under the same assumptions as Theorem 14.11, $M$ satisfies $\mathrm{CD}(K, N)$ if and only if the improved inequality holds:

$$\mathcal{D}(t, x) \ge \tau_{K,N}^{(1-t)}\, \mathcal{D}(0, x) + \tau_{K,N}^{(t)}\, \mathcal{D}(1, x),$$

where now $\alpha = \sqrt{\lvert K \rvert/(N-1)}\, d(x,y)$ and

$$\tau_{K,N}^{(t)} = t^{1/N} \left(\frac{\sin(t\alpha)}{\sin\alpha}\right)^{1-1/N} \quad (K > 0), \qquad t \quad (K = 0), \qquad t^{1/N} \left(\frac{\sinh(t\alpha)}{\sinh\alpha}\right)^{1-1/N} \quad (K < 0).$$

This is stronger than Theorem 14.11 (note $N-1$ in place of $N$ in the trigonometric factor). When $N < \infty$ and $K > 0$, this contains the Bonnet–Myers bound $d(x,y) \le \pi\sqrt{(N-1)/K}$.

</div>

### Third Appendix: Jacobi Fields Forever

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jacobi Matrix)</span></p>

Let $R : [0,1] \to M\_n(\mathbb{R})$ be a continuous map valued in symmetric matrices. A **Jacobi matrix** is a matrix-valued function $J(t)$ solving

$$\ddot{J}(t) + R(t)\, J(t) = 0.$$

If $J$ is a Jacobi matrix and $A$ is a constant matrix, then $JA$ is still a Jacobi matrix. The space of Jacobi fields $\mathcal{U}$ is a $2n$-dimensional vector space, isomorphic to $\mathbb{R}^{2n}$ via $u \longmapsto (u(0), \dot{u}(0))$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.29</span><span class="math-callout__name">(Jacobi Matrices Have Symmetric Logarithmic Derivatives)</span></p>

Let $J$ be a Jacobi matrix such that $J(0)$ is invertible and $\dot{J}(0)\, J(0)^{-1}$ is symmetric. Let $t\_\ast$ be the largest time such that $J(t)$ is invertible for $t < t\_\ast$. Then $\dot{J}(t)\, J(t)^{-1}$ is symmetric for all $t \in [0, t\_\ast)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.30</span><span class="math-callout__name">(Cosymmetrization of Jacobi Matrices)</span></p>

Let $J\_0^1$ and $J\_1^0$ be defined by $J\_0^1(0) = I\_n$, $\dot{J}\_0^1(0) = 0$, $J\_1^0(0) = 0$, $\dot{J}\_1^0(0) = I\_n$. Any Jacobi matrix can be written as $J(t) = J\_0^1(t)\, J(0) + J\_1^0(t)\, \dot{J}(0)$. Assuming $J\_1^0(t)$ is invertible for all $t \in (0,1]$:

* (a) $S(t) := J\_1^0(t)^{-1} J\_0^1(t)$ is symmetric positive for all $t \in (0,1]$, and strictly decreasing in $t$.
* (b) There is a unique pair $(J^{1,0}, J^{0,1})$ with $J^{1,0}(0) = I\_n$, $J^{1,0}(1) = 0$, $J^{0,1}(0) = 0$, $J^{0,1}(1) = I\_n$; moreover $\dot{J}^{1,0}(0)$ and $\dot{J}^{0,1}(1)$ are symmetric.
* (c) For $K(t) = t\, J\_1^0(t)^{-1}$: the matrices $K(t)(J^{1,0}(t)\, J(0))\, J(0)^{-1}$ and $K(t)(J^{0,1}(t)\, J(1))\, J(0)^{-1}$ are symmetric. Moreover, $\det K(t) > 0$ for all $t \in [0,1)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.31</span><span class="math-callout__name">(Jacobi Matrices with Positive Determinant)</span></p>

Let $S(t)$ and $K(t)$ be the matrices from Proposition 14.30. Let $J$ be a Jacobi matrix with $J(0) = I\_n$ and $\dot{J}(0)$ symmetric. Then the following are equivalent:

* (i) $\dot{J}(0) + S(1) > 0$;
* (ii) $K(t)\, J^{0,1}(t)\, J(1) > 0$ for all $t \in (0,1)$;
* (iii) $K(t)\, J(t) > 0$ for all $t \in [0,1]$;
* (iv) $\det J(t) > 0$ for all $t \in [0,1]$.

**Geometric interpretation:** If $\psi$ is $d^2/2$-convex, then $\nabla^2 \psi(x) + \nabla\_x^2 [d(\cdot, \exp\_x \nabla \psi(x))^2/2] \ge 0$. In the Jacobi field language, this corresponds to condition (i): $\dot{J}(0) + S(1) \ge 0$. So Proposition 14.31 implies that the Jacobian of the optimal transport map remains positive for $0 < t < 1$, recovering the last part of Theorem 11.3.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 14.32</span><span class="math-callout__name">(Symplectic Interpretation)</span></p>

The proof of symmetry of $S(t)$ uses a "symplectic" argument: if $\mathbb{R}^n \times \mathbb{R}^n$ is equipped with its natural symplectic form $\omega((u, \dot{u}), (v, \dot{v})) = \langle \dot{u}, v \rangle - \langle \dot{v}, u \rangle$, then the flow $(u(s), \dot{u}(s)) \longmapsto (u(t), \dot{u}(t))$ preserves $\omega$. The subspaces $\mathcal{U}\_0 = \lbrace u(0) = 0 \rbrace$ and $\dot{\mathcal{U}}\_0 = \lbrace \dot{u}(0) = 0 \rbrace$ are **Lagrangian** (dimension $n$, $\omega$ vanishes on each). Then $\mathcal{U}\_t = \lbrace u(t) = 0 \rbrace$ is also Lagrangian, and writing it as a graph in $\mathcal{U}\_0 \times \dot{\mathcal{U}}\_0$ gives a symmetric operator.

</div>

### Bibliographical Notes on Chapter 14

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History)</span></p>

Recommended textbooks for Riemannian geometry: do Carmo, Gallot–Hulin–Lafontaine, and Chavel. The differential inequalities relating the Jacobian of the exponential map to the Ricci curvature appear in many sources (e.g. Chavel, Section 3.4), usually in conjunction with volume comparison principles (Heintze–Kärcher, Lévy–Gromov, Bishop–Gromov). Their adaptation to the nonsmooth context of semiconvex functions was achieved by **Cordero-Erausquin, McCann and Schmuckenschläger**, and more recently by various other authors.

Bochner's formula can be found in Gallot–Hulin–Lafontaine (Proposition 4.15) or Petersen (Proposition 3.3 (3)). Another derivation uses the square distance function $d(x\_0, x)^2$, which is the solution of the Hamilton–Jacobi equation at time 1 with initial datum 0 at $x\_0$ and $+\infty$ elsewhere.

The one-dimensional $\mathrm{CD}(K, N)$ model spaces (Examples 14.10) have been discussed by **Bakry and Qian** in relation with spectral gap estimates. The practical importance of separating out the direction of motion is implicit in **Cordero-Erausquin, McCann and Schmuckenschläger**, but was highlighted by **Sturm**. Alexandrov's second differentiability theorem was proven in 1942. The proof given in the First Appendix follows the classical argument of Evans–Gariepy, modified to resemble the proof of Rademacher's theorem.

</div>

## Chapter 15: Otto Calculus

This chapter presents **Otto's formal differential calculus** on the Wasserstein space $P\_2(M)$: rules for computing gradients, Hessians, and related quantities of functionals defined on probability measures. The treatment is purely formal (no rigorous justification attempted), but the formulas are powerful heuristic tools and will be justified by other means (Lagrangian formalism) in subsequent chapters.

### Setup: Internal Energy Functionals

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Internal Energy Functional)</span></p>

Let $\nu(dx) = e^{-V(x)}\, \mathrm{vol}(dx)$ be a reference measure on a Riemannian manifold $M$, with $V : M \to \mathbb{R}$ smooth. Let $U : \mathbb{R}\_+ \to \mathbb{R}$ be twice differentiable (at least on $(0, +\infty)$), with $U(0) = 0$. The **internal energy functional** is

$$U_\nu(\mu) := \int_M U(\rho(x))\, d\nu(x), \qquad \mu = \rho\, \nu.$$

Associated to $U$ are the **pressure** $p(\rho) = \rho\, U'(\rho) - U(\rho)$ and the **iterated pressure** $p\_2(\rho) = \rho\, p'(\rho) - p(\rho)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 15.1</span></p>

* $U(\rho) = U^{(m)}(\rho) = (\rho^m - \rho)/(m-1)$, $m \ne 1$: $p(\rho) = \rho^m$, $p\_2(\rho) = (m-1)\rho^m$.
* $U(\rho) = U^{(1)}(\rho) = \rho \log \rho$ (limit $m \to 1$): $p(\rho) = \rho$, $p\_2(\rho) = 0$.

The modified Laplace operator is $L = \Delta - \nabla V \cdot \nabla$, with $\Gamma\_2(\psi) = L(\lvert \nabla \psi \rvert^2/2) - \nabla \psi \cdot \nabla(L\psi) = \lVert \nabla^2 \psi \rVert\_{\mathrm{HS}}^2 + (\mathrm{Ric} + \nabla^2 V)(\nabla \psi)$.

</div>

### Gradient Formula

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Formula 15.2</span><span class="math-callout__name">(Gradient Formula in Wasserstein Space)</span></p>

Let $\mu$ be absolutely continuous with respect to $\nu$. Then:

$$\mathrm{grad}\_\mu\, U_\nu = -\nabla \cdot \bigl(\mu\, \nabla U'(\rho)\bigr) = -\nabla \cdot \bigl(e^{-V} \nabla p(\rho)\bigr)\, \mathrm{vol} = -(L\, p(\rho))\, \nu.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 15.4</span><span class="math-callout__name">(Boltzmann Entropy)</span></p>

For $H(\mu) = \int \rho \log \rho\, d\mathrm{vol}$ (with $\nu = \mathrm{vol}$): $\mathrm{grad}\_\mu H = -\Delta \mu$. Thus **the gradient of Boltzmann's entropy is the Laplace operator**. More generally, for $H\_\nu(\mu) = \int \rho \log \rho\, d\nu$: $\mathrm{grad}\_\mu H\_\nu = -(L\rho)\, \nu$, i.e. the gradient of the relative entropy is the **distorted Laplace operator** $L$.

</div>

### Hessian Formula

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Formula 15.7</span><span class="math-callout__name">(Hessian Formula in Wasserstein Space)</span></p>

Let $\mu$ be absolutely continuous with respect to $\nu$, and let $\dot{\mu} = -\nabla \cdot (\mu\, \nabla \psi)$ be a tangent vector at $\mu$. Then:

$$\mathrm{Hess}\_\mu\, U_\nu(\dot{\mu}) = \int_M \Gamma_2(\psi)\, p(\rho)\, d\nu + \int_M (L\psi)^2\, p_2(\rho)\, d\nu.$$

More explicitly:

$$\mathrm{Hess}\_\mu\, U_\nu(\dot{\mu}) = \int_M \bigl[\lVert \nabla^2 \psi \rVert_{\mathrm{HS}}^2 + (\mathrm{Ric} + \nabla^2 V)(\nabla \psi)\bigr]\, p(\rho)\, d\nu + \int_M (-\Delta \psi + \nabla V \cdot \nabla \psi)^2\, p_2(\rho)\, d\nu.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 15.9</span></p>

For $U(\rho) = (\rho^m - \rho)/(m-1)$: $\mathrm{Hess}\_\mu H\_\nu^{(m)}(\dot{\mu}) = \int [\lVert \nabla^2 \psi \rVert\_{\mathrm{HS}}^2 + (\mathrm{Ric} + \nabla^2 V)(\nabla \psi) + (m-1)(\Delta \psi - \nabla V \cdot \nabla \psi)^2]\, \rho^{m-1}\, d\mu$.

In the limit $m = 1$ ($U = \rho \log \rho$): $\mathrm{Hess}\_\mu H\_\nu(\dot{\mu}) = \int [\lVert \nabla^2 \psi \rVert\_{\mathrm{HS}}^2 + \mathrm{Ric}\_{\infty,\nu}(\nabla \psi)]\, d\mu$.

</div>

### General Gradient Formula and Open Problems

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General Gradient Formula)</span></p>

For an arbitrary functional $\mathcal{F}$ on $P\_2(M)$:

$$\mathrm{grad}\_\mu \mathcal{F} = -\nabla \cdot \!\left(\mu\, \nabla \frac{\delta \mathcal{F}}{\delta \mu}\right), \qquad \phi = \frac{\delta \mathcal{F}}{\delta \mu},$$

where $\delta \mathcal{F}/\delta \mu$ is the first variation defined by $\frac{d}{dt} \mathcal{F}(\mu\_t) = \int (\delta \mathcal{F}/\delta \mu)\, \partial\_t \mu\_t$. For $\mathcal{F}(\mu) = \int F(x, \rho(x), \nabla \rho(x))\, d\nu(x)$: $(\delta \mathcal{F}/\delta \mu)(x) = (\partial\_\rho F)(x, \rho, \nabla \rho) - (\nabla\_x - \nabla V) \cdot (\nabla\_p F)(x, \rho, \nabla \rho)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Think Eulerian, Prove Lagrangian)</span></p>

The Otto calculus is best used as a **heuristic tool**: compute formally in the Eulerian framework (gradients, Hessians in $P\_2(M)$), then prove the results rigorously by translating everything into Lagrangian language (second derivatives along geodesics, Jacobian determinants). This strategy works because "there are no shocks" in optimal transport — trajectories do not meet until possibly the final time (Chapter 8).

</div>

### Bibliographical Notes on Chapter 15

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History)</span></p>

**Otto's** seminal paper studied the formal Riemannian structure of the Wasserstein space, with applications to the porous medium equation. Then **Otto and Villani** considered $U(\rho) = \rho \log \rho$ on a manifold and computed the Hessian, yielding the first published work where Ricci curvature appeared in relation to optimal transport. Functionals of interaction type $E(\mu) = \int W(x-y)\, \mu(dx)\, \mu(dy)$ were later studied by **Carrillo, McCann and Villani**. The pressure interpretation $p(\rho) = \rho U'(\rho) - U(\rho)$ was explained by McCann.

The $H$ functional $H\_\nu(\mu) = \int \rho \log \rho\, d\nu$ is well-known in statistical physics (Boltzmann, 1870s), information theory (Kullback–Leibler information, Shannon's entropy), and statistics (Sanov's theorem). The **Fisher information** $I(\mu) = \int \lvert \nabla \rho \rvert^2/\rho$ also appears naturally in this context.

Open questions: Is there a Jacobi equation in $P\_2(M)$? Christoffel symbols? A Laplace operator? A volume element? **Lott** partly answered some of these by establishing formulas for the Riemannian connection and curvature in $P^\infty(M)$ (smooth positive densities). **Gigli** gave a rigorous construction of parallel transport along curves in $P\_2(\mathbb{R}^n)$. For $p \ne 2$, $P\_p(M)$ has a Finsler structure, and Otto calculus is much less developed.

</div>

## Chapter 16: Displacement Convexity I

Convexity in a metric space means convexity along geodesics. This chapter introduces **displacement convexity** — convexity along geodesics in the Wasserstein space $P\_2(M)$ — and uses Otto's calculus (Chapter 15) to guess conditions under which the internal energy functional $U\_\nu$ is displacement convex. These guesses will be rigorously proven in the next chapter.

### Convexity in Geodesic Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 16.1</span><span class="math-callout__name">(Geodesic Convexity)</span></p>

Let $(\mathcal{X}, d)$ be a complete geodesic space. A function $F : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ is said to be **geodesically convex** (or just convex) if for any constant-speed geodesic path $(\gamma\_t)\_{0 \le t \le 1}$ valued in $\mathcal{X}$,

$$\forall t \in [0,1] \qquad F(\gamma_t) \le (1-t)\, F(\gamma_0) + t\, F(\gamma_1).$$

It is **weakly convex** if for any $x\_0, x\_1 \in \mathcal{X}$ there exists *at least one* geodesic $(\gamma\_t)$ with $\gamma\_0 = x\_0$, $\gamma\_1 = x\_1$ such that the inequality holds.

</div>

### Reminders on Convexity: Differential and Integral Conditions

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.2</span><span class="math-callout__name">(Convexity and Lower Hessian Bounds)</span></p>

Let $(M, g)$ be a Riemannian manifold, $\Lambda = \Lambda(x, v)$ a continuous quadratic form on $TM$ with $\lambda[\gamma] := \inf\_{0 \le t \le 1} \Lambda(\gamma\_t, \dot{\gamma}\_t)/\lvert \dot{\gamma}\_t \rvert^2 > -\infty$. For any $F \in C^2(M)$, the following are equivalent:

* (i) $\nabla^2 F \ge \Lambda$;
* (ii) For any constant-speed minimizing geodesic $\gamma : [0,1] \to M$:

$$F(\gamma_t) \le (1-t)\, F(\gamma_0) + t\, F(\gamma_1) - \int_0^1 \Lambda(\gamma_s, \dot{\gamma}_s)\, G(s,t)\, ds;$$

* (iii) $F(\gamma\_1) \ge F(\gamma\_0) + \langle \nabla F(\gamma\_0), \dot{\gamma}\_0 \rangle + \int\_0^1 \Lambda(\gamma\_t, \dot{\gamma}\_t)\, (1-t)\, dt$;
* (iv) $\langle \nabla F(\gamma\_1), \dot{\gamma}\_1 \rangle - \langle \nabla F(\gamma\_0), \dot{\gamma}\_0 \rangle \ge \int\_0^1 \Lambda(\gamma\_t, \dot{\gamma}\_t)\, dt$;

where $G(s,t) = \min(s(1-t), t(1-s))$ is the one-dimensional Green function. The equivalence is preserved if (ii), (iii), (iv) are weakened to (ii'), (iii'), (iv') by replacing $\int\_0^1 \Lambda\, G\, ds$ by $\lambda[\gamma]\, t(1-t)\, d(\gamma\_0, \gamma\_1)^2/2$, etc.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 16.4</span><span class="math-callout__name">($\Lambda$-Convexity)</span></p>

A function $F : M \to \mathbb{R} \cup \lbrace +\infty \rbrace$ is said to be **$\Lambda$-convex** if Property (ii) in Proposition 16.2 holds. When $\Lambda = \lambda g$ for a constant $\lambda \in \mathbb{R}$, $F$ is said to be **$\lambda$-convex**: $F(\gamma\_t) \le (1-t)F(\gamma\_0) + tF(\gamma\_1) - \frac{\lambda\, t(1-t)}{2}\, d(\gamma\_0, \gamma\_1)^2$. In particular, 0-convexity is plain convexity.

</div>

### Displacement Convexity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 16.5</span><span class="math-callout__name">(Displacement Convexity)</span></p>

A functional $F : P\_2^{\mathrm{ac}}(M) \to \mathbb{R} \cup \lbrace +\infty \rbrace$ is said to be:

* **displacement convex** if, whenever $(\mu\_t)\_{0 \le t \le 1}$ is a (constant-speed, minimizing) geodesic in $P\_2^{\mathrm{ac}}(M)$: $F(\mu\_t) \le (1-t)\, F(\mu\_0) + t\, F(\mu\_1)$;

* **$\lambda$-displacement convex** if: $F(\mu\_t) \le (1-t)\, F(\mu\_0) + t\, F(\mu\_1) - \frac{\lambda\, t(1-t)}{2}\, W\_2(\mu\_0, \mu\_1)^2$;

* **$\Lambda$-displacement convex** if, with $(\psi\_t)$ the associated Hamilton–Jacobi solution: $F(\mu\_t) \le (1-t)\, F(\mu\_0) + t\, F(\mu\_1) - \int\_0^1 \Lambda(\mu\_s, \widetilde{\nabla} \psi\_s)\, G(s,t)\, ds$.

$\Lambda$-displacement convexity reduces to $\lambda$-displacement convexity when $\Lambda(\mu, v) = \lambda\, \lVert v \rVert\_{L^2(\mu)}^2$, which in turn reduces to plain displacement convexity when $\lambda = 0$.

</div>

### Displacement Convexity from Curvature-Dimension Bounds

Using the Hessian formula from Chapter 15 (Formula 15.7), combined with the $\mathrm{CD}(K, N)$ bound (Theorem 14.8), one formally derives sufficient conditions for displacement convexity.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Heuristic Derivation)</span></p>

From $\mathrm{Hess}\_\mu\, U\_\nu(\dot{\mu}) = \int \Gamma\_2(\psi)\, p(\rho)\, d\nu + \int (L\psi)^2\, p\_2(\rho)\, d\nu$ and $\Gamma\_2(\psi) \ge (L\psi)^2/N + K\, \lvert \nabla \psi \rvert^2$ (from $\mathrm{CD}(K,N)$), one gets

$$\mathrm{Hess}\_\mu\, U_\nu(\dot{\mu}) \ge K \int \lvert \nabla \psi \rvert^2\, p(\rho)\, d\nu + \int (L\psi)^2 \left[p_2 + \frac{p}{N}\right](\rho)\, d\nu.$$

This is nonneg. (yielding displacement convexity) if $p\_2 + p/N \ge 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Displacement Convexity Class $\mathcal{DC}\_N$)</span></p>

The set of all continuous, convex functions $U : \mathbb{R}\_+ \to \mathbb{R}$ with $U(0) = 0$, twice continuously differentiable on $(0,+\infty)$, and satisfying

$$p_2(\rho) + \frac{p(\rho)}{N} \ge 0 \qquad \forall \rho > 0$$

is called the **displacement convexity class of dimension $N$**, denoted $\mathcal{DC}\_N$. A typical representative achieving equality is

$$U_N(\rho) = \begin{cases} -N(\rho^{1-1/N} - \rho) & \text{if } 1 < N < \infty, \\\\ \rho \log \rho & \text{if } N = \infty. \end{cases}$$

The associated functionals are denoted $H\_{N,\nu}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Guess 16.6</span></p>

Let $M$ be a Riemannian manifold satisfying $\mathrm{CD}(K, N)$ for some $K \in \mathbb{R}$, $N \in (1, \infty]$, and let $U$ satisfy $p\_2 + p/N \ge 0$. Then $U\_\nu$ is $K \Lambda\_U$-displacement convex, where $\Lambda\_U(\mu, \dot{\mu}) = \int \lvert \nabla \psi \rvert^2\, p(\rho)\, d\nu$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Guess 16.7</span><span class="math-callout__name">(Converse)</span></p>

If for each $x\_0 \in M$, $H\_{N,\nu}$ is $K \Lambda\_U$-displacement convex when applied to probability measures supported in a small neighborhood of $x\_0$, then $M$ satisfies $\mathrm{CD}(K, N)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 16.8</span><span class="math-callout__name">($\mathrm{CD}(0, \infty)$ and Boltzmann Entropy)</span></p>

$\mathrm{CD}(0, \infty)$ with $\nu = \mathrm{vol}$ just means $\mathrm{Ric} \ge 0$, and $U \in \mathcal{DC}\_\infty$ means $p\_2 \ge 0$. The typical case is $U(\rho) = \rho \log \rho$, giving $H(\mu) = \int \rho \log \rho\, d\mathrm{vol}$. The Otto calculus suggests the following equivalences:

* (i) $\mathrm{Ric} \ge 0$;
* (ii) If $U \in \mathcal{DC}\_\infty$ then $U\_{\mathrm{vol}}$ is displacement convex;
* (iii) $H$ is displacement convex;
* (iii') $H$ is locally displacement convex (on measures supported in small neighborhoods).

More generally, $\mathrm{Ric} \ge Kg$ should be equivalent to the $K$-displacement convexity of $H$.

</div>

### A Fluid Mechanics Feeling for Ricci Curvature

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two Physical Experiments to Detect Curvature)</span></p>

**The light source test (relativistic physicist):** Take a small light source and try to determine its volume from a distant position. If you systematically overestimate the volume, you live in a nonnegatively Ricci-curved space (recall the distortion coefficients from Definition 14.17).

**The lazy gas experiment (fluid mechanics physicist):** Take a perfect gas of noninteracting particles and ask it to move from one prescribed density to another using a path of least action. Measure the entropy $S = -\int \rho \log \rho$ at each time, and check that it always lies **above** the line joining initial and final entropies. If so, you live in a nonnegatively Ricci-curved space. (In positive curvature, geodesics converge at intermediate times, so particles spread out more, lowering the density and increasing the entropy.)

</div>

### Bibliographical Notes on Chapter 16

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History)</span></p>

The concept and terminology of **displacement convexity** were introduced by **McCann** in the mid-nineties. He identified the condition $p\_2 + p/N \ge 0$ as the basic criterion for convexity in $P\_2(\mathbb{R}^n)$. The application of Otto calculus to displacement convexity goes back to **Otto** and **Otto–Villani**, where it was conjectured that nonneg. Ricci curvature would imply displacement convexity of $H$. Ricci curvature has a well-known role in general relativity (Einstein's equations); the fluid mechanics analogies for curvature appear in **Cordero-Erausquin, McCann and Schmuckenschläger**. **Lott** recently showed that convexity of $t \mapsto \int \rho\_t \log \rho\_t\, d\nu + N\, t \log t$ along displacement interpolation characterizes $\mathrm{CD}(0, N)$.

</div>

## Chapter 17: Displacement Convexity II

This chapter provides a rigorous Lagrangian justification of the conjectures formulated in Chapter 16, and leads to new curvature-dimension criteria based on "distorted displacement convexity". The main results are Theorems 17.15 and 17.37.

### Displacement Convexity Classes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 17.1</span><span class="math-callout__name">(Displacement Convexity Classes $\mathcal{DC}\_N$)</span></p>

Let $N$ be a real parameter in $[1, \infty]$. The class $\mathcal{DC}\_N$ is the set of continuous convex functions $U : \mathbb{R}\_+ \to \mathbb{R}$, twice continuously differentiable on $(0, +\infty)$, such that $U(0) = 0$, and satisfying any one of the following equivalent conditions:

* (i) $p\_2(r) + p(r)/N \ge 0$, where $p(r) = rU'(r) - U(r)$, $p\_2(r) = rp'(r) - p(r)$;
* (ii) $p(r)/r^{1-1/N}$ is a nondecreasing function of $r$;
* (iii) The function $u(\delta) := \begin{cases} \delta^N U(\delta^{-N}) & \text{if } N < \infty \\\\ e^\delta U(e^{-\delta}) & \text{if } N = \infty \end{cases}$ is convex.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 17.3</span></p>

$\mathcal{DC}\_{N'} \subset \mathcal{DC}\_N$ if $N' \ge N$. So $\mathcal{DC}\_\infty$ is the smallest class and $\mathcal{DC}\_1$ the largest (conditions (i)–(iii) are void for $N = 1$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 17.6</span></p>

* (i) $U(r) = r^\alpha$, $\alpha \ge 1$: belongs to all $\mathcal{DC}\_N$.
* (ii) $U(r) = -r^\alpha$, $\alpha < 1$: belongs to $\mathcal{DC}\_N$ iff $N \le (1-\alpha)^{-1}$ (i.e. $\alpha \ge 1 - 1/N$). The function $-r^{1-1/N}$ is the minimal representative of $\mathcal{DC}\_N$.
* (iii) $U\_\infty(r) = r \log r$ belongs to $\mathcal{DC}\_\infty$ (limit of $U\_N(r) = -N(r^{1-1/N} - r)$ as $N \to \infty$).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17.7</span><span class="math-callout__name">(Behavior of Functions in $\mathcal{DC}\_N$)</span></p>

* (i) For $N < \infty$: any growth rate can be achieved at infinity ($U(r)/r \to +\infty$ as $r \to \infty$).
* (ii) For $N = \infty$: either $U$ is linear, or $U(r) \ge a\, r \log r + b\, r$ for some $a > 0$, $b \in \mathbb{R}$.
* (iii) If $p(r\_0) > 0$ then $p'(r) \ge K r^{-1/N}$ for $r \ge r\_0$; if $p(r\_0) = 0$ then $U$ is linear on $[0, r\_0]$.
* (iv)–(vii) Any $U \in \mathcal{DC}\_N$ can be approximated (monotonically from below or above) by smooth functions $U\_\ell \in \mathcal{DC}\_N$ that are linear near $0$ and asymptotically like $-a\, r^{1-1/N} + b\, r$ (or $a\, r \log r + b\, r$ if $N = \infty$) at infinity. Moreover, $U''\_\ell \le C U''$ for some constant $C$ independent of $\ell$.

</div>

### Domain of the Functionals $U\_\nu$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 17.8</span><span class="math-callout__name">(Moment Conditions Make Sense of $U\_\nu(\mu)$)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, $\nu$ a reference Borel measure, $N \in [1, \infty]$. Assume there exist $x\_0 \in \mathcal{X}$ and $p \in [2, +\infty)$ such that

$$\begin{cases} \int_\mathcal{X} \frac{d\nu(x)}{[1 + d(x_0, x)]^{p(N-1)}} < +\infty & \text{if } N < \infty, \\\\ \exists\, c > 0:\; \int_M e^{-c\, d(x_0, x)^p}\, d\nu(x) < +\infty & \text{if } N = \infty. \end{cases}$$

Then, for any $U \in \mathcal{DC}\_N$, the formula $U\_\nu(\mu) = \int U(\rho)\, d\nu$ ($\mu = \rho\, \nu$) unambiguously defines a functional $U\_\nu : P\_p^{\mathrm{ac}}(\mathcal{X}) \to \mathbb{R} \cup \lbrace +\infty \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 17.12</span></p>

If $\mathcal{X}$ is a length space (e.g. a Riemannian manifold), then $P\_p(\mathcal{X})$ is a geodesically convex subset of $P\_q(\mathcal{X})$ for any $q \in (1, +\infty)$. Also $P\_p^{\mathrm{ac}}(\mathcal{X})$ is geodesically convex in $P\_2(\mathcal{X})$. So it makes sense to study convexity of $U\_\nu$ along geodesics of $P\_2(M)$, even if $U\_\nu$ is only defined on $P\_p^{\mathrm{ac}}(M)$.

</div>

### Displacement Convexity from Curvature-Dimension Bounds

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 17.13</span><span class="math-callout__name">(Local Displacement Convexity)</span></p>

Let $M$ be a Riemannian manifold and $F : P\_2^{\mathrm{ac}}(M) \to \mathbb{R} \cup \lbrace +\infty \rbrace$. Then $F$ is **locally displacement convex** if for any $x\_0 \in M$ there is $r > 0$ such that $F(\mu\_t) \le (1-t)\, F(\mu\_0) + t\, F(\mu\_1)$ for all measures $\mu\_t$ supported in $B\_r(x\_0)$. "Local" refers to the topology of the base space $M$, not of the Wasserstein space.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 17.15</span><span class="math-callout__name">(CD Bounds Read Off from Displacement Convexity)</span></p>

Let $M$ be a Riemannian manifold with geodesic distance $d$ and reference measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$. Let $K \in \mathbb{R}$ and $N \in (1, \infty]$. Let $p \in [2, +\infty) \cup \lbrace c \rbrace$ satisfy the assumptions of Theorem 17.8. Then the following three conditions are equivalent:

**(i)** $M$ satisfies the curvature-dimension criterion $\mathrm{CD}(K, N)$;

**(ii)** For each $U \in \mathcal{DC}\_N$, the functional $U\_\nu$ is $\Lambda\_{N,U}$-displacement convex on $P\_p^{\mathrm{ac}}(M)$, where $\Lambda\_{N,U} = K\_{N,U}\, \Lambda\_N$, with

$$\Lambda_N(\mu, v) = \int_M \lvert v(x) \rvert^2\, \rho(x)^{1-1/N}\, d\nu(x), \qquad K_{N,U} = \inf_{r > 0} \frac{K\, p(r)}{r^{1-1/N}};$$

**(iii)** $H\_{N,\nu}$ is locally $K\Lambda\_N$-displacement convex;

and then necessarily $N \ge n$, with equality possible only if $V$ is constant.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 17.19</span><span class="math-callout__name">($\mathrm{CD}(K, \infty)$ and $\mathrm{CD}(0, N)$ via Optimal Transport)</span></p>

Let $M$ be a Riemannian manifold, $K \in \mathbb{R}$ and $N \in (1, \infty]$. Then:

* (a) $\mathrm{Ric} \ge Kg$ if and only if Boltzmann's $H$ functional is $K$-displacement convex on $P\_c^{\mathrm{ac}}(M)$;
* (b) $M$ has nonneg. Ricci curvature and dimension $\le N$ if and only if $H\_{N,\mathrm{vol}}$ is displacement convex on $P\_c^{\mathrm{ac}}(M)$.

</div>

### Core of the Proof of Theorem 17.15

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Idea: (i) $\Rightarrow$ (ii))</span></p>

The key argument (for $K = 0$, compactly supported measures). Let $(\mu\_t)$ be a Wasserstein geodesic with $\mu\_t = (T\_t)\_\# \mu\_0$. Fix $t\_0 \in (0,1)$; the transport $T\_{t\_0 \to t}$ has Jacobian $\mathcal{J}\_{t\_0 \to t}$, and by Theorem 11.3, $U\_\nu(\mu\_t) = \int U(\rho\_{t\_0}/\mathcal{J}\_{t\_0 \to t})\, \mathcal{J}\_{t\_0 \to t}/\rho\_{t\_0}\, \rho\_{t\_0}\, d\nu$. Define $\delta\_{t\_0}(t, x) = (\mathcal{J}\_{t\_0 \to t}(x)/\rho\_{t\_0}(x))^{1/N}$ (which coincides, up to a factor, with the mean distortion $\mathcal{D}(t)$). Since $U \in \mathcal{DC}\_N$, the associated function $u(\delta) = \delta^N U(\delta^{-N})$ is convex. Since $\mathrm{CD}(K, N)$ implies $\ddot{\mathcal{D}} + (K\lvert \dot{\gamma} \rvert^2/N)\mathcal{D} \le 0$, the argument of $u$ is concave in $t$. The composition of a convex nonincreasing function with a concave function is convex; this gives the displacement convexity of $U\_\nu$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Idea: (iii) $\Rightarrow$ (i))</span></p>

If $H\_{N,\nu}$ is locally $K\Lambda\_N$-displacement convex, consider $\mu\_0$ supported near a point $x\_0$ and $\mu\_1 = \delta\_{y}$. The optimal transport is $T(x) = \exp\_x(\nabla \psi(x))$, which is close to the exponential map. The Jacobian is $\det J^{0,1}(t)/t^n = \overline{\beta}\_t(x, y)$ (the distortion coefficient). By localizing the displacement convexity inequality to measures concentrated around $x\_0$ and sending the measures to Dirac masses, one recovers $\overline{\beta} \ge \beta^{(K,N)}$, which by Theorem 14.21 implies $\mathrm{Ric}\_{N,\nu} \ge K$.

</div>

### Ricci Curvature Bounds from Distorted Displacement Convexity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 17.25</span><span class="math-callout__name">(Distorted $U\_\nu$ Functional)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space with Borel reference measure $\nu$. Let $U$ be convex with $U(0) = 0$, let $\pi(dy \mid x)$ be conditional probability measures, and $\beta : \mathcal{X} \times \mathcal{X} \to (0, +\infty]$. The **distorted $U\_\nu$ functional** is

$$U_{\pi,\nu}^\beta(\mu) = \int_{\mathcal{X} \times \mathcal{X}} U\!\left(\frac{\rho(x)}{\beta(x,y)}\right) \beta(x,y)\, \pi(dy \mid x)\, \nu(dx), \qquad \mu = \rho\, \nu.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 17.34</span><span class="math-callout__name">(Distorted Displacement Convexity)</span></p>

$U\_\nu$ is **displacement convex with distortion $(\beta\_t)$** if for any geodesic $(\mu\_t)$ in the domain of $U\_\nu$:

$$U_\nu(\mu_t) \le (1-t)\, U_{\pi,\nu}^{\beta_{1-t}}(\mu_0) + t\, U_{\check{\pi},\nu}^{\beta_t}(\mu_1),$$

where $\pi$ is the optimal plan between $\mu\_0$ and $\mu\_1$, and $\check{\pi} = S\_\# \pi$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 17.37</span><span class="math-callout__name">(CD Bounds from Distorted Displacement Convexity)</span></p>

Let $M$ be a Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$. Let $K \in \mathbb{R}$, $N \in (1, \infty]$, and $\beta\_t^{(K,N)}$ the reference distortion coefficients. Then the following are equivalent:

* **(i)** $\mathrm{CD}(K, N)$;
* **(ii)** For each $U \in \mathcal{DC}\_N$, $U\_\nu$ is displacement convex on $P\_p^{\mathrm{ac}}(M)$ with distortion $(\beta\_t^{(K,N)})$;
* **(iii)** $H\_{N,\nu}$ is locally displacement convex with distortion $(\beta\_t^{(K,N)})$;

and then $N \ge n$, with equality only if $V$ is constant.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 17.41</span><span class="math-callout__name">(One-Dimensional CD and Displacement Convexity)</span></p>

$\mathrm{CD}(K, 1)$ is equivalent to: for each $U \in \mathcal{DC}\_1$, $U\_\nu$ is displacement convex on $P\_c^{\mathrm{ac}}(M)$ with distortion $(\beta\_t^{(K,1)})$. Then necessarily $n = 1$, $V$ is constant and $K \le 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 17.42</span><span class="math-callout__name">(Intrinsic Displacement Convexity)</span></p>

Let $M$ have dimension $n \ge 2$, $\beta\_t(x,y)$ continuous positive on $[0,1] \times M \times M$, $\nu = \mathrm{vol}$, $N = n$. Then the following are equivalent:

* (i) $\beta \le \overline{\beta}$ (the actual distortion coefficients of $M$);
* (ii) For any $U \in \mathcal{DC}\_n$, $U\_\nu$ is displacement convex with distortion $(\beta\_t)$;
* (iii) $H\_n$ is locally displacement convex with distortion $(\beta\_t)$.

</div>

### Bibliographical Notes on Chapter 17

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History)</span></p>

The $\mathcal{DC}\_N$ classes go back to **McCann's** PhD thesis. **McCann** established displacement convexity in $P\_2(\mathbb{R}^n)$ using concavity of $\det^{1/n}$ and the change of variables formula. **Cordero-Erausquin, McCann and Schmuckenschläger** extended this to Riemannian manifolds via distortion estimates. **Sturm and von Renesse** first showed that displacement convexity of $H$ *characterizes* nonneg. Ricci curvature. This was generalized by **Lott–Villani** and **Sturm** to $N \le \infty$.

**Sturm** realized the importance of *distorted* displacement convexity and proved Theorem 17.37 for $U = U\_N$. The general formulation was worked out by **Lott and Villani**. **Ohta** extended these results to Finsler geometries. **Lott** showed $\mathrm{CD}(0, N)$ is characterized by convexity of $t \mapsto t\, H\_\nu(\mu\_t) + N\, t \log t$. The **Cheng–Toponogov theorem** ($\mathrm{Ric} \ge K > 0$, $\mathrm{diam} = D\_{K,N}$ implies sphere) can be proved via displacement convexity (explained by Lott).

</div>

## Chapter 18: Volume Control

This chapter derives volume estimates (doubling property, Brunn–Minkowski inequality, Bishop–Gromov inequality) from curvature-dimension bounds using optimal transport arguments.

### Doubling Property

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 18.1</span><span class="math-callout__name">(Doubling Property)</span></p>

Let $(\mathcal{X}, d)$ be a metric space, $\nu$ a nonzero Borel measure on $\mathcal{X}$. The measure $\nu$ is **doubling** if there is a constant $D$ such that $\nu[B\_{2r}(x)] \le D\, \nu[B\_r(x)]$ for all $x \in \mathcal{X}$, $r > 0$. It is **locally doubling** if for any closed ball $B[z, R]$, there is $D = D(z, R)$ such that $\nu[B\_{2r}(x)] \le D\, \nu[B\_r(x)]$ for all $x \in B[z, R]$, $r \in (0, R)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18.4</span><span class="math-callout__name">(Doubling Measures Have Full Support)</span></p>

If $(\mathcal{X}, d)$ is a metric space with a locally doubling measure $\nu$, then $\mathrm{Spt}\, \nu = \mathcal{X}$.

</div>

### Distorted Brunn–Minkowski Inequality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 18.5</span><span class="math-callout__name">(Distorted Brunn–Minkowski Inequality)</span></p>

Let $M$ be a complete Riemannian manifold with reference measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying $\mathrm{CD}(K, N)$. Let $A\_0, A\_1$ be two nonempty compact subsets, and $t \in (0,1)$. Then:

**If $N < \infty$:**

$$\nu\bigl[[A_0, A_1]_t\bigr]^{1/N} \ge (1-t) \left[\inf_{(x_0, x_1) \in A_0 \times A_1} \beta_{1-t}^{(K,N)}(x_0, x_1)^{1/N}\right] \nu[A_0]^{1/N} + t \left[\inf_{(x_0, x_1) \in A_0 \times A_1} \beta_t^{(K,N)}(x_0, x_1)^{1/N}\right] \nu[A_1]^{1/N}.$$

**If $N = \infty$:**

$$\log \frac{1}{\nu[[A_0, A_1]_t]} \le (1-t) \log \frac{1}{\nu[A_0]} + t \log \frac{1}{\nu[A_1]} - \frac{Kt(1-t)}{2} \sup_{x_0 \in A_0,\, x_1 \in A_1} d(x_0, x_1)^2.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 18.6</span><span class="math-callout__name">(Brunn–Minkowski in Nonneg. Curvature)</span></p>

If $M$ satisfies $\mathrm{CD}(0, N)$, $N \in (1, +\infty)$, then

$$\nu\bigl[[A_0, A_1]_t\bigr]^{1/N} \ge (1-t)\, \nu[A_0]^{1/N} + t\, \nu[A_1]^{1/N}.$$

When $M = \mathbb{R}^n$, $N = n$, this is the classical **Brunn–Minkowski inequality**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Idea)</span></p>

Take $\mu\_0$ uniform on $A\_0$ and $\mu\_1$ uniform on $A\_1$. The displacement interpolation $\mu\_t$ is supported in $A\_t = [A\_0, A\_1]\_t$. Apply the distorted displacement convexity inequality (Theorem 17.37) with $U = U\_N$, then use Jensen's inequality ($\int \rho\_t^{1-1/N}\, d\nu \le \nu[A\_t]^{1/N}$) to conclude.

</div>

### Bishop–Gromov Inequality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 18.8</span><span class="math-callout__name">(Bishop–Gromov Inequality)</span></p>

Let $M$ be a Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$, satisfying $\mathrm{CD}(K, N)$ for some $K \in \mathbb{R}$, $1 < N < \infty$. Let

$$s^{(K,N)}(t) = \begin{cases} \left(\sin\sqrt{K/(N-1)}\, t\right)^{N-1} & \text{if } K > 0, \\\\ t^{N-1} & \text{if } K = 0, \\\\ \left(\sinh\sqrt{\lvert K \rvert/(N-1)}\, t\right)^{N-1} & \text{if } K < 0. \end{cases}$$

Then, for any $x \in M$, the ratio $\nu[B\_r(x)] / \int\_0^r s^{(K,N)}(t)\, dt$ is a **nonincreasing** function of $r$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Bishop–Gromov)</span></p>

For $K = 0$: apply Corollary 18.6 with $A\_0 = \lbrace x \rbrace$ and $A\_1 = B\_r(x)$. Since $\nu[\lbrace x \rbrace] = 0$, one gets $\nu[B\_s(x)]^{1/N} \ge (s/r)\, \nu[B\_r(x)]^{1/N}$ for $s \le r$, which gives $\nu[B\_r(x)]/r^N$ nonincreasing. For $K \ne 0$: use Theorem 18.5 with $A\_0 = \lbrace x \rbrace$, $A\_1 = B\_{r+\varepsilon}(x) \setminus B\_r(x)$, and a comparison argument (Lemma 18.9) to pass from the annular estimates to the monotonicity of $\nu[B\_r]/\int\_0^r s^{(K,N)}$.

</div>

### Doubling Property from CD Bounds

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 18.11</span><span class="math-callout__name">($\mathrm{CD}(K, N)$ Implies Doubling)</span></p>

Let $M$ be a Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$ satisfying $\mathrm{CD}(K, N)$, $1 < N < \infty$. Then $\nu$ is doubling with a constant $C$ that is:

* **uniform** and no more than $2^N$ if $K \ge 0$;
* **locally uniform** and no more than $2^N\, D(K, N, R)$ if $K < 0$, where

$$D(K, N, R) = \left[\cosh\!\left(2\sqrt{\frac{\lvert K \rvert}{N-1}}\, R\right)\right]^{N-1},$$

when restricted to a ball $B[z, R]$.

</div>

### Dimension-Free Bounds

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 18.12</span><span class="math-callout__name">(Dimension-Free Control on Ball Growth)</span></p>

Let $M$ be a Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$, satisfying $\mathrm{CD}(K, \infty)$ for some $K \in \mathbb{R}$. Then, for any $\delta > 0$, there is a constant $C = C(K\_-, \delta, \nu[B\_\delta(x\_0)], \nu[B\_{2\delta}(x\_0)])$ such that for all $r \ge \delta$:

$$\nu[B_r(x_0)] \le e^{Cr}\, e^{(K_-)\, r^2/2};$$

$$\nu[B_{r+\delta}(x_0) \setminus B_r(x_0)] \le e^{Cr}\, e^{-K\, r^2/2} \qquad \text{if } K > 0.$$

In particular, if $K' < K$ then $\int e^{K' d(x\_0, x)^2/2}\, \nu(dx) < +\infty$.

</div>

### Bibliographical Notes on Chapter 18

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History)</span></p>

The Brunn–Minkowski inequality in $\mathbb{R}^n$ dates back to the nineteenth century (Brunn, Minkowski, Lusternik). **McCann** observed that optimal transport provides a convenient reparametrization for proving it. At the end of the nineties, it was unclear how to generalize to curved spaces. **Cordero-Erausquin** guessed the Prékopa–Leindler inequality on the sphere using optimal transport. **Cordero-Erausquin, McCann and Schmuckenschläger** developed the rigorous tools and established Prékopa–Leindler inequalities in curved geometry (volume as reference measure). **Sturm** adapted the proof of Theorem 18.5 for general reference measures and proved the Bishop–Gromov inequality for $K \ne 0$. **Ohta** further generalized to Finsler geometries. The proof of the Bishop–Gromov inequality for $K = 0$ is from Lott–Villani. Lemma 18.9 is apparently due to Gromov.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 14.15</span><span class="math-callout__name">(Sectional vs. Ricci Curvature Bounds)</span></p>

There is a similar (and more well-known) formulation for lower *sectional* curvature bounds: define $\mathcal{L}\_M(t, \delta, L) = \inf\lbrace d(\exp\_x(tv), \exp\_x(tw));\; \lvert v \rvert = \lvert w \rvert = \delta;\; d(\exp\_x v, \exp\_x w) = L \rbrace$. Then $M$ has sectional curvature $\ge \kappa$ if and only if $\mathcal{L}\_M \ge \mathcal{L}^{(\kappa)}$ (the function of the model space $S^2(1/\sqrt{\kappa})$, $\mathbb{R}^2$, or $\mathbb{H}^2(1/\sqrt{\lvert \kappa \rvert})$). The key comparison: *sectional curvature bounds measure the rate of separation of geodesics in terms of distances, while Ricci curvature bounds do it in terms of Jacobian determinants.*

</div>

### Distortion Coefficients

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.17</span><span class="math-callout__name">(Distortion Coefficients)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C(M)$, and let $x$ and $y$ be any two points in $M$. The **distortion coefficient** $\overline{\beta}\_t(x, y)$ between $x$ and $y$ at time $t \in (0,1)$ is defined as follows:

* If $x$ and $y$ are joined by a unique geodesic $\gamma$:

$$\overline{\beta}_t(x, y) = \lim_{r \downarrow 0} \frac{\nu\bigl[[x, B_r(y)]_t\bigr]}{\nu[B_{tr}(y)]} = \lim_{r \downarrow 0} \frac{\nu\bigl[[x, B_r(y)]_t\bigr]}{t^n\, \nu[B_r(y)]}.$$

* If $x$ and $y$ are joined by several minimizing geodesics: $\overline{\beta}\_t(x, y) = \inf\_\gamma \limsup\_{s \to 1^-} \overline{\beta}\_t(x, \gamma\_s)$.
* $\overline{\beta}\_1(x,y) \equiv 1$; $\overline{\beta}\_0(x,y) := \liminf\_{t \to 0^+} \overline{\beta}\_t(x,y)$.

**Heuristic interpretation:** Standing at point $x$ and observing a device located at $y$, light rays traveling along geodesics are distorted by curvature. The coefficient $\overline{\beta}\_0(x,y)$ tells you by how much you overestimate the volume of the device: $> 1$ in positive curvature, $< 1$ in negative curvature.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.18</span><span class="math-callout__name">(Computation of Distortion Coefficients)</span></p>

$\overline{\beta}\_t(x, y) = \inf\_\gamma \overline{\beta}\_t^{[\gamma]}(x, y)$, where the infimum is over all minimizing geodesics $\gamma$ from $x$ to $y$. If $x, y$ are not conjugate along $\gamma$, let $\mathbf{J}^{0,1}(t)$ be the unique matrix of Jacobi fields satisfying $\mathbf{J}^{0,1}(0) = 0$, $\mathbf{J}^{0,1}(1) = \mathbf{E}$ (the parallel orthonormal basis). Then:

$$\overline{\beta}_t^{[\gamma]}(x, y) = \frac{\det \mathbf{J}^{0,1}(t)}{t^n}.$$

If $x, y$ are conjugate along $\gamma$, then $\overline{\beta}\_t^{[\gamma]}(x,y) = +\infty$ for $0 \le t < 1$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.19</span><span class="math-callout__name">(Reference Distortion Coefficients)</span></p>

Given $K \in \mathbb{R}$, $N \in [1, \infty]$ and $t \in [0,1]$, and two points $x, y$ in a metric space $(\mathcal{X}, d)$, define $\beta\_t^{(K,N)}(x, y)$ as follows:

* If $0 < t \le 1$ and $1 < N < \infty$:

$$\beta_t^{(K,N)}(x, y) = \begin{cases} +\infty & \text{if } K > 0 \text{ and } \alpha > \pi, \\\\ \left(\frac{\sin(t\alpha)}{t \sin \alpha}\right)^{N-1} & \text{if } K > 0 \text{ and } \alpha \in [0, \pi], \\\\ 1 & \text{if } K = 0, \\\\ \left(\frac{\sinh(t\alpha)}{t \sinh \alpha}\right)^{N-1} & \text{if } K < 0, \end{cases}$$

where $\alpha = \sqrt{\lvert K \rvert/(N-1)}\, d(x, y)$.

* Limit cases: $\beta\_t^{(K,1)}(x,y) = \begin{cases} +\infty & K > 0 \\\\ 1 & K \le 0 \end{cases}$; $\quad \beta\_t^{(K,\infty)}(x,y) = e^{\frac{K}{6}(1-t^2)\, d(x,y)^2}$.

* $\beta\_0^{(K,N)}(x,y) = 1$.

If $\mathcal{X}$ is the model $\mathrm{CD}(K, N)$ space, then $\beta^{(K,N)}$ is just the distortion coefficient on $\mathcal{X}$. The coefficient $\beta\_t^{(K,N)}$ is nondecreasing in $K$ and nonincreasing in $N$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.20</span><span class="math-callout__name">(Distortion Coefficients and Concavity of Jacobian Determinant)</span></p>

Let $M$ be a Riemannian manifold of dimension $n$, and let $x, y$ be any two points in $M$. If $(\beta\_t(x,y))\_{0 \le t \le 1}$ and $(\beta\_t(y,x))\_{0 \le t \le 1}$ are two families of nonneg. coefficients, the following are equivalent:

**(a)** $\forall t \in [0,1]$, $\beta\_t(x,y) \le \overline{\beta}\_t(x,y)$; $\beta\_t(y,x) \le \overline{\beta}\_t(y,x)$.

**(b)** For any $N \ge n$, any geodesic $\gamma$ from $x$ to $y$, any $t\_0 \in [0,1]$, any initial vector field $\xi$ around $x\_0 = \gamma(t\_0)$ with $\nabla \xi(x\_0)$ symmetric, if $\mathcal{J}(s)$ is the Jacobian of $\exp((s - t\_0)\xi)$ (not vanishing for $0 < s < 1$), then for all $t \in [0,1]$:

$$\mathcal{J}(t)^{1/N} \ge (1-t)\, \beta_{1-t}(y, x)^{1/N}\, \mathcal{J}(0)^{1/N} + t\, \beta_t(x, y)^{1/N}\, \mathcal{J}(1)^{1/N} \quad (N < \infty);$$

$$\log \mathcal{J}(t) \ge (1-t) \log \mathcal{J}(0) + t \log \mathcal{J}(1) + [(1-t) \log \beta_{1-t}(y,x) + t \log \beta_t(x,y)] \quad (N = \infty).$$

**(c)** Property (b) holds true for $N = n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.21</span><span class="math-callout__name">(Ricci Curvature Bounds via Distortion Coefficients)</span></p>

Let $M$ be a Riemannian manifold of dimension $n$, equipped with its volume measure. Then the following are equivalent:

* (a) $\mathrm{Ric} \ge K$;
* (b) $\overline{\beta} \ge \beta^{(K,n)}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 14.24</span><span class="math-callout__name">(Self-Improvement Property)</span></p>

If $(M, \nu)$ satisfies $\mathrm{CD}(K, N)$ then (14.65) still holds with $\beta = \beta^{(K,N)}$. But in view of Theorems 14.11 and 14.12, for any manifold $M$ of dimension $n$ the following two conditions are equivalent for $K > 0$:

* (i) $\forall x, y \in M$, $\forall t \in [0,1]$: $\overline{\beta}\_t(x,y) \ge \left(\frac{\sin(t\sqrt{K/n}\, d(x,y))}{t\sin(\sqrt{K/n}\, d(x,y))}\right)^n$;
* (ii) $\forall x, y \in M$, $\forall t \in [0,1]$: $\overline{\beta}\_t(x,y) \ge \left(\frac{\sin(t\sqrt{K/(n-1)}\, d(x,y))}{t\sin(\sqrt{K/(n-1)}\, d(x,y))}\right)^{n-1}$.

This self-improvement property implies restrictions on the possible behavior of $\overline{\beta}$.

</div>

### First Appendix: Second Differentiability of Convex Functions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.25</span><span class="math-callout__name">(Alexandrov's Second Differentiability Theorem)</span></p>

Let $\varphi : \mathbb{R}^n \to \mathbb{R}$ be a convex function. Then, for Lebesgue-almost every $x \in \mathbb{R}^n$, $\varphi$ is differentiable at $x$ and there exists a symmetric operator $A : \mathbb{R}^n \to \mathbb{R}^n$, characterized by any one of the following equivalent properties:

* (i) $\nabla \varphi(x + v) = \nabla \varphi(x) + Av + o(\lvert v \rvert)$ as $v \to 0$;
* (i') $\partial \varphi(x + v) = \nabla \varphi(x) + Av + o(\lvert v \rvert)$ as $v \to 0$;
* (ii) $\varphi(x + v) = \varphi(x) + \nabla \varphi(x) \cdot v + \frac{\langle Av, v \rangle}{2} + o(\lvert v \rvert^2)$ as $v \to 0$;
* (ii') $\forall v \in \mathbb{R}^n$, $\varphi(x + tv) = \varphi(x) + t\, \nabla \varphi(x) \cdot v + t^2\, \frac{\langle Av, v \rangle}{2} + o(t^2)$ as $t \to 0$.

The operator $A$ is the **Hessian** $\nabla^2 \varphi(x)$. Its trace $\Delta \varphi(x) = \mathrm{tr}(\nabla^2 \varphi(x))$ is the density of the absolutely continuous part of the distributional Laplacian of $\varphi$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Strategy)</span></p>

The proof reduces to $\mathbb{R}^n$ by diffeomorphism invariance of semiconvexity. In dimension 1 the result is trivial ($\varphi'$ is nondecreasing, hence differentiable a.e.). In the Lipschitz gradient case, the proof uses Rademacher + dominated convergence + convolution. The general case uses a regularization argument together with the Lebesgue density theorem: the key difficulty is to show that the singular part of the distributional Hessian $\mu\_v := (1/2)\langle \nabla^2 \varphi \cdot v, v \rangle$ does not contribute at Lebesgue points.

</div>

### Second Appendix: Elementary Comparison Arguments

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.28</span><span class="math-callout__name">(One-Dimensional Comparison Inequalities)</span></p>

Let $\Lambda \in \mathbb{R}$, and $f \in C([0,1]) \cap C^2(0,1)$, $f \ge 0$. Then the following are equivalent:

* (i) $\ddot{f} + \Lambda f \le 0$ in $(0,1)$;
* (ii) If $\Lambda < \pi^2$ then for all $t\_0, t\_1 \in [0,1]$,

$$f\bigl((1-\lambda)t_0 + \lambda t_1\bigr) \ge \tau^{(1-\lambda)}(\lvert t_0 - t_1 \rvert)\, f(t_0) + \tau^{(\lambda)}(\lvert t_0 - t_1 \rvert)\, f(t_1),$$

where $\tau^{(\lambda)}(\theta) = \sin(\lambda \theta \sqrt{\Lambda}) / \sin(\theta \sqrt{\Lambda})$ if $0 < \Lambda < \pi^2$; $= \lambda$ if $\Lambda = 0$; $= \sinh(\lambda \theta \sqrt{-\Lambda}) / \sinh(\theta \sqrt{-\Lambda})$ if $\Lambda < 0$.

If $\Lambda = \pi^2$ then $f(t) = c \sin(\pi t)$ for some $c \ge 0$; if $\Lambda > \pi^2$ then $f = 0$.

</div>

### Chapter 18: Supplementary Details

#### Remarks on the Doubling Property

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 18.2</span><span class="math-callout__name">(Equivalence of Local Doubling Formulations)</span></p>

It is equivalent to say that a measure $\nu$ is locally doubling, or that its restriction to any ball $B[z, R]$ (considered as a metric space) is doubling.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 18.3</span><span class="math-callout__name">(Open vs. Closed Balls)</span></p>

It does not really matter whether the definition of doubling is formulated in terms of open or closed balls; at worst this changes the value of the constant $D$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sharp Spines and Curvature-Dimension Bounds)</span></p>

It is a standard fact in Riemannian geometry that doubling constants may be estimated, at least locally, in terms of curvature-dimension bounds. These estimates express the fact that the manifold does not contain *sharp spines*. A Riemannian manifold obviously has this property (being locally diffeomorphic to $\mathbb{R}^n$), but curvature-dimension bounds quantify this in terms of intrinsic geometry, without reference to charts.

</div>

#### The Classical Brunn–Minkowski Inequality and $t$-Barycenters

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Classical Brunn–Minkowski Inequality)</span></p>

Whenever $A\_0$ and $A\_1$ are two nonempty compact subsets of $\mathbb{R}^n$,

$$\lvert A_0 + A_1 \rvert^{1/n} \ge \lvert A_0 \rvert^{1/n} + \lvert A_1 \rvert^{1/n},$$

where $\lvert \cdot \rvert$ stands for Lebesgue measure, and $A\_0 + A\_1$ is the set of all vectors of the form $a\_0 + a\_1$ with $a\_0 \in A\_0$ and $a\_1 \in A\_1$. This inequality contains the Euclidean isoperimetric inequality as a limit case (take $A\_1 = \varepsilon B(0,1)$ and let $\varepsilon \to 0$).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($t$-Barycenters)</span></p>

If $A\_0$ and $A\_1$ are two nonempty compact subsets of a Riemannian manifold $M$, the set $[A\_0, A\_1]\_t$ stands for the set of all **$t$-barycenters** of $A\_0$ and $A\_1$: the set of all $y \in M$ that can be written as $\gamma\_t$, where $\gamma$ is a minimizing, constant-speed geodesic with $\gamma\_0 \in A\_0$ and $\gamma\_1 \in A\_1$. Equivalently, $[A\_0, A\_1]\_t$ is the set of all $y$ such that there exists $(x\_0, x\_1) \in A\_0 \times A\_1$ with $d(x\_0, y)/d(y, x\_1) = t/(1-t)$.

In $\mathbb{R}^n$, of course $[A\_0, A\_1]\_t = (1-t)\, A\_0 + t\, A\_1$.

</div>

#### Remark on Corollary 18.6

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 18.7</span><span class="math-callout__name">(Reduction to Classical Brunn–Minkowski)</span></p>

When $M = \mathbb{R}^n$, $N = n$, inequality (18.6) reduces to

$$\lvert (1-t) A_0 + t A_1 \rvert^{1/n} \ge (1-t)\, \lvert A_0 \rvert^{1/n} + t\, \lvert A_1 \rvert^{1/n},$$

where $\lvert \cdot \rvert$ stands for $n$-dimensional Lebesgue measure. By homogeneity, this is equivalent to the classical Brunn–Minkowski inequality.

</div>

#### Detailed Proof of Theorem 18.5

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Detailed Proof of Theorem 18.5, Case $N < \infty$)</span></p>

By regularity of $\nu$ and an approximation argument, it suffices to treat $\nu[A\_0] > 0$ and $\nu[A\_1] > 0$. Define $\mu\_0 = \rho\_0\, \nu$ and $\mu\_1 = \rho\_1\, \nu$ where

$$\rho_0 = \frac{1_{A_0}}{\nu[A_0]}, \qquad \rho_1 = \frac{1_{A_1}}{\nu[A_1]}.$$

Let $(\mu\_t)\_{0 \le t \le 1}$ be the unique displacement interpolation between $\mu\_0$ and $\mu\_1$ for the cost $d(x,y)^2$. Since $M$ satisfies the curvature-dimension bound $\mathrm{CD}(K, N)$, Theorem 17.37 applied with $U(r) = U\_N(r) = -N(r^{1-1/N} - r)$ implies

$$\int_M U_N(\rho_t(x))\, \nu(dx) \le (1-t) \int_M U_N\!\left(\frac{\rho_0(x_0)}{\beta_{1-t}(x_0, x_1)}\right) \beta_{1-t}(x_0, x_1)\, \pi(dx_0, x_1)\, \nu(dx_0) + t \int_M U_N\!\left(\frac{\rho_1(x_1)}{\beta_t(x_0, x_1)}\right) \beta_t(x_0, x_1)\, \pi(dx_0, x_1)\, \nu(dx_1),$$

where $\pi$ is the optimal coupling and $\beta\_t$ is shorthand for $\beta\_t^{(K,N)}$. After substitution and simplification using the explicit form of $U\_N$, this leads to

$$\int_M \rho_t(x)^{1-1/N}\, \nu(dx) \ge (1-t) \int_M \rho_0(x)^{-1/N}\, \beta_{1-t}(x_0, x_1)^{1/N}\, \pi(dx_0\, dx_1) + t \int_M \rho_1(x)^{-1/N}\, \beta_t(x_0, x_1)^{1/N}\, \pi(dx_0\, dx_1).$$

Since $\pi$ is supported in $A\_0 \times A\_1$ with marginals $\rho\_0\, \nu$ and $\rho\_1\, \nu$, bounding the right-hand side from below gives:

$$(1-t)\, \beta_{1-t}^{1/N} \int_M \rho_0(x_0)^{1-1/N}\, d\nu(x_0) + t\, \beta_t^{1/N} \int_M \rho_1(x_1)^{1-1/N}\, d\nu(x_1),$$

where $\beta\_t$ stands for the minimum of $\beta\_t(x\_0, x\_1)$ over all pairs in $A\_0 \times A\_1$. By explicit computation, $\int\_M \rho\_i^{1-1/N}\, d\nu = \nu[A\_i]^{1/N}$ for $i = 0, 1$. So it suffices to show $\int\_M \rho\_t^{1-1/N}\, d\nu \le \nu[[A\_0, A\_1]\_t]^{1/N}$.

Since $\mu\_t$ is supported in $A\_t = [A\_0, A\_1]\_t$ and $\rho\_t$ is a probability density on that set, Jensen's inequality gives:

$$\int_{A_t} \rho_t^{1-1/N}\, d\nu = \nu[A_t] \int_{A_t} \rho_t^{1-1/N}\, \frac{d\nu}{\nu[A_t]} \le \nu[A_t] \left(\int_{A_t} \rho_t\, \frac{d\nu}{\nu[A_t]}\right)^{1-1/N} = \nu[A_t]^{1/N}.$$

The case $N = \infty$ follows similar lines, using $K$-displacement convexity of $H\_\nu$ and the convexity of $r \longmapsto r \log r$.

</div>

#### Lemma 18.9

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 18.9</span><span class="math-callout__name">(Integral Comparison Lemma)</span></p>

Let $a < b$ in $\mathbb{R} \cup \lbrace +\infty \rbrace$, let $g : (a, b) \to \mathbb{R}\_+$ be a positive continuous function, integrable at $a$, and let $G(r) = \int\_a^r g(s)\, ds$. Let $F : [a, b) \to \mathbb{R}\_+$ be a nondecreasing measurable function satisfying $F(a) = 0$, and let $f(r) = d^+ F / dr$ be its upper derivative. If $f/g$ is nonincreasing then also $F/G$ is nonincreasing.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Lemma 18.9)</span></p>

Let $h = f/g$; by assumption $h$ is nonincreasing. For $x \ge x\_0 > a$, $f(x) \le g(x)\, h(x\_0)$, so $F$ is locally Lipschitz and $F(y) - F(x) = \int\_x^y f(t)\, dt$. The key step: for $a \le t \le x \le t' \le y$, $h(t) \le h(t')$ fails (since $h$ is nonincreasing), so $f(t) \int\_x^y g \le \int\_x^y g \int\_a^x f$. After careful rearrangement:

$$\frac{\int_a^x f}{\int_a^x g} \le \frac{\int_x^y f}{\int_x^y g},$$

which gives $F/G$ nonincreasing.

</div>

#### Proof of Theorem 18.8 (Bishop–Gromov)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Detailed Proof of Bishop–Gromov, Case $K \ne 0$)</span></p>

It suffices to show that

$$\frac{d^+/dr\; \nu[B_r]}{s^{(K,N)}(r)} \quad \text{is nonincreasing,}$$

where $B\_r = B\_{r]}(x)$. By Lemma 18.9, this implies the nonincreasing property of $\nu[B\_r] / \int\_0^r s^{(K,N)}(t)\, dt$.

Apply Theorem 18.5 with $A\_0 = \lbrace x \rbrace$ and $A\_1 = B\_{r+\varepsilon} \setminus B\_r$; then for $t \in (0,1)$, $[A\_0, A\_1]\_t \subset B\_{t(r+\varepsilon)} \setminus B\_{tr}$. For $K \ge 0$, the distortion coefficient satisfies

$$\beta_t^{(K,N)}(x_0, x_1) \ge \left(\frac{\sin\bigl(t\sqrt{K/(N-1)}\,(r+\varepsilon)\bigr)}{t\sin\bigl(\sqrt{K/(N-1)}\,(r+\varepsilon)\bigr)}\right)^{N-1}.$$

Applying the Brunn–Minkowski inequality (18.4) and letting $\phi(r) = \nu[B\_r]$:

$$\frac{\phi(tr + t\varepsilon) - \phi(tr)}{\varepsilon\, s^{(K,N)}(t(r+\varepsilon))} \ge \frac{\phi(r+\varepsilon) - \phi(r)}{\varepsilon\, s^{(K,N)}(r+\varepsilon)}.$$

In the limit $\varepsilon \to 0$, this yields $\phi'(tr) / s^{(K,N)}(tr) \ge \phi'(r) / s^{(K,N)}(r)$ for any $t \in [0,1]$, confirming $\phi'/s^{(K,N)}$ is nonincreasing.

</div>

#### Volume Ratio Comparison

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Volume Ratio Comparison)</span></p>

The Bishop–Gromov inequality is more precise than the doubling property. For $0 < s < r$:

$$\nu[B_r(x)] \ge \left(\frac{V(s)}{V(r)}\right) \nu[B_r(x)],$$

where $V(r)$ is the volume of $B\_r(x)$ in the model space of constant sectional curvature $K/(N-1)$ and dimension $N$. This implies that $\nu[B\_r(x)]$ is a continuous function of $r$, a fact obvious for Riemannian manifolds but for which the Bishop–Gromov inequality provides an explicit modulus of continuity.

</div>

#### Exercise 18.10

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 18.10</span><span class="math-callout__name">(Alternative Proof of Bishop–Gromov for $\mathrm{CD}(0, N)$)</span></p>

Give an alternative proof of the Bishop–Gromov inequality for $\mathrm{CD}(0, N)$ Riemannian manifolds, using the convexity of $t \longmapsto t\, U\_\nu(\mu\_t) + N\, t \log t$, for $U \in \mathcal{DC}\_N$, when $(\mu\_t)\_{0 \le t \le 1}$ is a displacement interpolation.

</div>

#### Proof of Theorem 18.12 (Dimension-Free Bounds)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 18.12)</span></p>

Write $B\_r$ for $B\_{r]}(x\_0)$. Apply the $N = \infty$ case of Theorem 18.5 with $A\_0 = B\_\delta$, $A\_1 = B\_r$, and $t = \delta/(2r) \le 1/2$. For any minimizing geodesic $\gamma$ from $A\_0$ to $A\_1$, $d(\gamma\_0, \gamma\_1) \le r + \delta$, so

$$d(x_0, \gamma_t) \le d(x_0, \gamma_0) + d(\gamma_0, \gamma_t) \le \delta + t(r+\delta) \le \delta + 2t r \le 2\delta.$$

Thus $[A\_0, A\_1]\_t \subset B\_{2\delta}$, and by (18.5):

$$\log \frac{1}{\nu[B_{2\delta}]} \le \left(1 - \frac{\delta}{2r}\right) \log \frac{1}{\nu[B_\delta]} + \frac{\delta}{2r} \log \frac{1}{\nu[B_r]} + \frac{K_-}{2} \cdot \frac{\delta}{2r}\left(1 - \frac{\delta}{2r}\right)(r+\delta)^2.$$

This implies an estimate of the form $\nu[B\_r] \le \exp(a + br + c/r + K\_- r^2/2)$ where $a, b, c$ depend only on $\delta$, $\nu[B\_\delta]$ and $\nu[B\_{2\delta}]$, giving (18.11).

The proof of (18.12) is similar, with $A\_0 = B\_\delta$, $A\_1 = B\_{r+\delta} \setminus B\_r$, $t = \delta/(3r)$.

For (18.13) when $K > 0$: take $\delta = 1$ and write

$$\int_{\mathcal{X}} e^{K' d(x_0, x)^2/2}\, \nu(dx) \le e^{K'/2}\, \nu[B_1] + \sum_{k \ge 1} e^{K'(k+1)^2/2}\, \nu[B_{k+1} \setminus B_k] \le e^{K'/2}\, \nu[B_1] + C \sum_{k \ge 1} e^{C(k+1)}\, e^{K'(k+1)^2/2}\, e^{-K k^2/2} < +\infty,$$

since $K' < K$.

</div>

## Chapter 19: Density Control and Local Regularity

This chapter addresses the following local regularity problem: given an estimate on a ball $B\_r(x\_0)$, deduce a better estimate on a smaller ball $B\_{r/2}(x\_0)$. The key ingredients, following the methods of De Giorgi, Nash, and Moser, are:

* a **doubling inequality** for the reference volume measure;
* a **local Poincaré inequality**, controlling the deviation of a function on a smaller ball by the integral of its gradient on a larger ball.

The strategy of this chapter is to derive both from curvature-dimension bounds via optimal transport, going through **pointwise bounds** on the density of the displacement interpolant.

### Local Poincaré Inequality

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 19.1</span><span class="math-callout__name">(Local Poincaré Inequality)</span></p>

Let $(\mathcal{X}, d)$ be a Polish metric space and $\nu$ a Borel measure on $\mathcal{X}$. The measure $\nu$ satisfies a **local Poincaré inequality** with constant $C$ if, for any Lipschitz function $u$, any point $x\_0 \in \mathcal{X}$ and any radius $r > 0$,

$$\fint_{B_r(x_0)} \bigl\lvert u(x) - \langle u \rangle_{B_r(x_0)} \bigr\rvert\, d\nu(x) \le C\, r \fint_{B_{2r}(x_0)} \lvert \nabla u(x) \rvert\, d\nu(x),$$

where $\fint\_B = (\nu[B])^{-1} \int\_B$ is the averaged integral and $\langle u \rangle\_B = \fint\_B u\, d\nu$ is the average of $u$ on $B$.

The measure $\nu$ satisfies a local Poincaré inequality with constant $C$ **in** a Borel subset $B$ if the above holds under the additional restriction that $B\_{2r}(x\_0) \subset B$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 19.2</span><span class="math-callout__name">(Gradient in Nonsmooth Contexts)</span></p>

The definition of $\lvert \nabla u \rvert$ in a nonsmooth context will be discussed later (Chapter 20). In this chapter, only Riemannian manifolds are considered, where the gradient has its usual meaning.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 19.3</span><span class="math-callout__name">(Local vs. Global)</span></p>

The word "local" in Definition 19.1 means that the inequality is interested in averages *around some point* $x\_0$. This contrasts with the "global" Poincaré inequalities (Chapter 21), in which averages are over the whole space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Uniform vs. Local Constants)</span></p>

Sometimes $\nu$ is said to satisfy a *uniform* local Poincaré inequality to stress that $C$ is independent of $x\_0$ and $r$. For most applications, it suffices that the inequality holds in the neighborhood of any point $x\_0$, i.e., with constant $C = C(R)$ on each ball $B(z, R)$.

Just as the doubling inequality, the local Poincaré inequality can be ruined by sharp spines; Ricci curvature bounds prevent this, providing quantitative Poincaré constants that are uniform in nonnegative curvature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two Proof Strategies)</span></p>

There are at least two ways to prove pointwise bounds on the displacement interpolant density $\rho\_t$:

1. **Jacobian estimates** (Chapter 14): combine the Jacobian equation for the interpolant density (Chapter 11) with Ricci curvature bounds on the Jacobian determinant. This is formally simpler.
2. **Displacement convexity** (Chapter 17): use the displacement convexity of internal energy functionals, combined with stability of optimal transport under restriction (Theorem 4.6). This is more indirect but more robust, and generalizes to nonsmooth settings (Chapter 30).

In either approach, a pointwise bound on $\rho\_t(x)$ is obtained by considering integral bounds on a very small ball $B\_\delta(x)$ and letting $\delta \to 0$.

</div>

### Pointwise Estimates on the Interpolant Density

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19.4</span><span class="math-callout__name">($\mathrm{CD}(K, N)$ Implies Pointwise Bounds on Displacement Interpolants)</span></p>

Let $M$ be a Riemannian manifold with reference measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying $\mathrm{CD}(K, N)$ for some $K \in \mathbb{R}$, $N \in (1, \infty]$. Let $\mu\_0 = \rho\_0\, \nu$ and $\mu\_1 = \rho\_1\, \nu$ be two probability measures in $P\_p^{\mathrm{ac}}(M)$, $p \in [2, +\infty) \cup \lbrace c \rbrace$. Let $(\mu\_t)\_{0 \le t \le 1}$ be the unique displacement interpolation, and $\rho\_t$ the density of $\mu\_t$ w.r.t. $\nu$. Then for any $t \in (0,1)$:

* **If $N < \infty$:**

$$\rho_t(x) \le \sup_{x \in [x_0, x_1]_t} \left((1-t) \left(\frac{\rho_0(x_0)}{\beta_{1-t}^{(K,N)}(x_0, x_1)}\right)^{-1/N} + t \left(\frac{\rho_1(x_1)}{\beta_t^{(K,N)}(x_0, x_1)}\right)^{-1/N}\right)^{-N},$$

with the convention $((1-t)\, a^{-1/N} + t\, b^{-1/N})^{-N} = 0$ if either $a$ or $b$ is $0$.

* **If $N = \infty$:**

$$\rho_t(x) \le \sup_{x \in [x_0, x_1]_t} \rho_0(x_0)^{1-t}\, \rho_1(x_1)^t \exp\!\left(-\frac{Kt(1-t)}{2}\, d(x_0, x_1)^2\right).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 19.5</span><span class="math-callout__name">(Preservation of Uniform Bounds in Nonnegative Curvature)</span></p>

With the same notation as in Theorem 19.4, if $K \ge 0$ then

$$\lVert \rho_t \rVert_{L^\infty(\nu)} \le \max\!\bigl(\lVert \rho_0 \rVert_{L^\infty(\nu)},\, \lVert \rho_1 \rVert_{L^\infty(\nu)}\bigr).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 19.4 via Displacement Convexity)</span></p>

Consider the case $N < \infty$. Let $t \in (0,1)$ and $y \in M$ with $\delta > 0$. The goal is to bound $\mathbb{P}[\gamma\_t \in B\_\delta(y)] = \mu\_t[B\_\delta(y)]$ from above.

**Step 1 (Restriction).** Condition $\gamma$ on the event $\gamma\_t \in B\_\delta(y)$. This yields a conditioned geodesic $\gamma'$ with law $\Pi' = (1\_{\mathcal{Z}}\, \Pi) / \Pi[\mathcal{Z}]$, where $\mathcal{Z} = \lbrace \gamma \in \Gamma(M) : \gamma\_t \in B\_\delta(y) \rbrace$. The resulting marginals $\mu\_s' = (e\_s)\_\# \Pi'$ satisfy $\mu\_s' \le \mu\_s / \mu\_t[B\_\delta(y)]$, so the density $\rho\_s' \le \rho\_s / \mu\_t[B\_\delta(y)]$. By Theorem 4.6 (restriction), $(\gamma\_0', \gamma\_1')$ is an optimal coupling, and $(\mu\_s')\_{0 \le s \le 1}$ is a displacement interpolation.

**Step 2 (Displacement convexity).** Apply Theorem 17.37 with $U(r) = -r^{1-1/N}$ to get

$$\int_M (\rho_t')^{1-1/N}\, d\nu \ge (1-t) \int (\rho_0')^{-1/N}\, \beta_{1-t}^{1/N}\, \pi'(dx_0\, dx_1) + t \int (\rho_1')^{-1/N}\, \beta_t^{1/N}\, \pi'(dx_0\, dx_1).$$

**Step 3 (Jensen + Lebesgue density).** Since $\mu\_t'$ is supported in $B\_\delta(y)$, Jensen's inequality gives $\int (\rho\_t')^{1-1/N}\, d\nu \le \nu[B\_\delta(y)]^{1/N}$. The right-hand side is bounded below using $F(x) = \inf\_{x \in [x\_0, x\_1]\_t}[(1-t)(\rho\_0(x\_0))^{-1/N} \beta\_{1-t}^{1/N} + t\, (\rho\_1(x\_1))^{-1/N} \beta\_t^{1/N}]$. Letting $\delta \to 0$ and using Lebesgue's density theorem yields $\rho\_t(y)^{-1/N} \ge F(y)$, i.e. $\rho\_t(y) \le F(y)^{-N}$.

</div>

### Jacobian Bounds Revisited

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19.6</span><span class="math-callout__name">(Jacobian Bounds Revisited)</span></p>

Let $M$ be a Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying $\mathrm{CD}(K, N)$ for some $K \in \mathbb{R}$, $N \in (1, \infty)$. Let $z\_0 \in M$ and $B$ be a bounded set of positive measure. Let $(\mu\_t^{z\_0})\_{0 \le t \le 1}$ be the displacement interpolation joining $\mu\_0 = \delta\_{z\_0}$ to $\mu\_1 = (1\_B\, \nu)/\nu[B]$. Then the density $\rho\_t^{z\_0}$ of $\mu\_t^{z\_0}$ satisfies

$$\rho_t^{z_0}(x) \le \frac{C(K, N, R)}{t^N\, \nu[B]},$$

where

$$C(K, N, R) = \exp\!\left(-\sqrt{(N-1)\, K_-}\; R\right), \qquad K_- = \max(-K, 0),$$

and $R$ is an upper bound on the distances between $z\_0$ and elements of $B$.

In particular, if $K \ge 0$, then $\rho\_t^{z\_0}(x) \le 1/(t^N\, \nu[B])$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 19.7</span><span class="math-callout__name">(Classical Jacobian Estimate)</span></p>

Theorem 19.6 is a classical estimate in Riemannian geometry, often stated as a bound on the Jacobian of the map $(s, \xi) \longmapsto \exp\_x(s\xi)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 19.6)</span></p>

Let $\mu\_1 = (1\_B\, \nu)/\nu[B]$ and $\mu\_0 = \delta\_{z\_0}$. Consider the displacement interpolation $(\mu\_t)\_{0 \le t \le 1}$ and its reparametrized version $(\mu\_t')$ with $t' = t\_0 + (1-t\_0)\, t$, so that $\mu\_0' = \mu\_{t\_0}$ and $\mu\_1' = \mu\_1$. Theorem 19.4 gives a bound on $\rho\_{t'}$, and since $\rho\_1 = 1\_B/\nu[B]$ and $x\_0 \in [z\_0, B]\_{t\_0}$, one gets

$$\rho_{t'}(x) \le \sup \frac{\rho_1(x_1)}{t^N\, \beta_t(x_0, x_1)} = \sup \frac{1}{t^N\, \nu[B]\, \beta_t(x_0, x_1)}.$$

Letting $t\_0 \to 0$ with $t'$ fixed, the supremum of $\beta\_t(x\_0, x\_1)^{-1}$ converges to $S(0, z\_0, B) \le C(K, N, R)$ by an elementary estimate on the distortion coefficients.

</div>

### Intrinsic Pointwise Bounds

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19.8</span><span class="math-callout__name">(Intrinsic Pointwise Bounds on the Displacement Interpolant)</span></p>

Let $M$ be an $n$-dimensional Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, and let $\overline{\beta}$ be the associated distortion coefficients (Definition 14.17). Let $\mu\_0, \mu\_1$ be two absolutely continuous probability measures on $M$, $(\mu\_t)\_{0 \le t \le 1}$ the unique generalized displacement interpolation, and $\rho\_t$ the density of $\mu\_t$. Then

$$\rho_t(x) \le \sup_{x \in [x_0, x_1]_t} \left((1-t) \left(\frac{\rho_0(x_0)}{\overline{\beta}_{1-t}(x_0, x_1)}\right)^{-1/n} + t \left(\frac{\rho_1(x_1)}{\overline{\beta}_t(x_0, x_1)}\right)^{-1/n}\right)^{-n},$$

with the same convention as in Theorem 19.4. This bound uses the *intrinsic* distortion coefficients $\overline{\beta}$ of the manifold, without any reference to a choice of $K$ and $N$.

</div>

### Democratic Condition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 19.9</span><span class="math-callout__name">(Democratic Condition)</span></p>

A Borel measure $\nu$ on a geodesic space $(\mathcal{X}, d)$ satisfies the **democratic condition** $\mathrm{Dm}(C)$ for some constant $C > 0$ if the following property holds: For any closed ball $B$ in $\mathcal{X}$ there is a random geodesic $\gamma$ such that $\gamma\_0$ and $\gamma\_1$ are independent and distributed uniformly in $B$, and the time-integral of the density of $\gamma\_t$ (with respect to $\nu$) never exceeds $C/\nu[B]$.

More explicitly, if $\mu\_t$ stands for the law of $\gamma\_t$, then

$$\int_0^1 \mu_t\, dt \le C\, \frac{\nu}{\nu[B]}.$$

The condition is **uniform** if $C$ is independent of $B = B[x, r]$, and **locally uniform** if it is independent of $B$ as long as $B[x, 2r] \subseteq B[z, R]$ for a large fixed ball.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19.10</span><span class="math-callout__name">($\mathrm{CD}(K, N)$ Implies Democratic Condition)</span></p>

Let $M$ be a Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying $\mathrm{CD}(K, N)$ for some $K \in \mathbb{R}$, $N \in (1, \infty)$. Then $\nu$ satisfies a locally uniform democratic condition with admissible constant $2^N\, C(K, N, R)$ in a large ball $B[z, R]$, where $C(K, N, R)$ is defined in Theorem 19.6.

In particular, if $K \ge 0$, then $\nu$ satisfies the uniform democratic condition $\mathrm{Dm}(2^N)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 19.10)</span></p>

Let $B$ be a ball of radius $r$ and $\mu = (1\_B\, \nu)/\nu[B]$. For any $x\_0$, by Theorem 19.6 the density of $\mu\_t^{x\_0}$ (the displacement interpolation from $\delta\_{x\_0}$ to $\mu$) satisfies $\rho\_t(x) \le C(K, N, R)/(t^N\, \nu[B])$. Hence $\mu\_t^{x\_0} \le [C(K, N, R)/(t^N\, \nu[B])]\, \nu$.

Now $\mu\_t = \mathrm{law}(\gamma\_t) = \int\_M \mu\_t^{x\_0}\, d\mu(x\_0)$, so $\mu\_t \le [C(K,N,R)/(t^N\, \nu[B])]\, \nu$. By time-reversal symmetry, $\mu\_t \le [C(K,N,R)/((1-t)^N\, \nu[B])]\, \nu$ as well. Combining: $\rho\_t(x) \le C(K,N,R)\, \min(1/t^N,\, 1/(1-t)^N) / \nu[B] \le 2^N\, C(K,N,R)/\nu[B]$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 19.11</span><span class="math-callout__name">(Improvement with $L^p$ Bounds)</span></p>

The bounds in Theorem 19.10 can be improved: if $\mu = \rho\, \nu$ is an arbitrary absolutely continuous probability measure, then there exists a random geodesic $\gamma$ with $\mathrm{law}(\gamma\_0, \gamma\_1) = \mu \otimes \mu$, $\mathrm{law}(\gamma\_t) = \mu\_t$ admitting a density $\rho\_t$ with respect to $\nu$, and

$$\lVert \rho_t \rVert_{L^p(\nu)} \le C(K, N, R)^{1/p'}\, \min\!\left(\frac{1}{t^{N/p'}},\, \frac{1}{(1-t)^{N/p'}}\right) \lVert \rho \rVert_{L^p(\nu)}$$

for all $p \in (1, \infty)$, where $p' = p/(p-1)$ is the conjugate exponent.

</div>

### Local Poincaré Inequality from CD Bounds

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19.13</span><span class="math-callout__name">(Doubling + Democratic Imply Local Poincaré)</span></p>

Let $(\mathcal{X}, d)$ be a length space with reference measure $\nu$ satisfying a doubling condition with constant $D$ and a democratic condition with constant $C$. Then $\nu$ satisfies a local Poincaré inequality with constant $P = 2\, C\, D$.

If the doubling and democratic conditions hold inside a ball $B(z, R)$ with constants $C = C(z, R)$ and $D = D(z, R)$ respectively, then $\nu$ satisfies a local Poincaré inequality in $B(z, R)$ with constant $P(z, R) = 2\, C(z, R)\, D(z, R)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 19.14</span><span class="math-callout__name">($\mathrm{CD}(K, N)$ Implies Local Poincaré)</span></p>

Let $M$ be a Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying $\mathrm{CD}(K, N)$ for some $K \in \mathbb{R}$, $N \in (1, \infty)$. Then $\nu$ satisfies a local Poincaré inequality with constant $P(K, N, R) = 2^{2N+1}\, C(K, N, R)\, D(K, N, R)$, inside any ball $B[z, R]$, where $C(K, N, R)$ and $D(K, N, R)$ are defined in (19.11) and (18.10).

In particular, if $K \ge 0$ then $\nu$ satisfies a local Poincaré inequality on the whole of $M$ with constant $2^{2N+1}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 19.13)</span></p>

Let $x\_0$ be a given point, $r > 0$, $B = B\_{r]}(x\_0)$, $2B = B\_{2r]}(x\_0)$, and $\mu = (1\_B\, \nu)/\nu[B]$. For any $y\_0 \in M$:

$$u(y_0) - \langle u \rangle_B = \int_M \bigl(u(y_0) - u(y_1)\bigr)\, d\mu(y_1).$$

So $\fint\_B \lvert u - \langle u \rangle\_B \rvert\, d\nu \le \int\_{B \times B} \lvert u(y\_0) - u(y\_1) \rvert\, d\mu(y\_0)\, d\mu(y\_1)$.

Estimate $\lvert u(y\_0) - u(y\_1) \rvert$ along a geodesic $\gamma$ of length $\le 2r$: $\lvert u(y\_0) - u(y\_1) \rvert \le 2r \int\_0^1 \lvert \nabla u \rvert(\gamma(t))\, dt$. By the democratic condition, there exists a random geodesic with $\mathrm{law}(\gamma\_0, \gamma\_1) = \mu \otimes \mu$ and $\int\_0^1 \mu\_t\, dt \le C\, \nu/\nu[B]$. Integrating against the law of $\gamma$:

$$\fint_B \lvert u - \langle u \rangle_B \rvert\, d\nu \le 2r \int_0^1 \int_M \lvert \nabla u \rvert\, d\mu_t\, dt \le \frac{2Cr}{\nu[B]} \int_{2B} \lvert \nabla u \rvert\, d\nu,$$

since geodesics joining points of $B$ stay inside $2B$. By the doubling property, $1/\nu[B] \le D/\nu[2B]$, giving $P = 2CD$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 19.15</span><span class="math-callout__name">(Refined Poincaré Inequality)</span></p>

With almost the same proof, one obtains the following refinement of the local Poincaré inequality:

$$\int_{B[x,r]} \frac{\lvert u(x) - u(y) \rvert}{d(x,y)}\, d\nu(x)\, d\nu(y) \le P(K, N, R) \int_{B[x, 2r]} \lvert \nabla u \rvert(x)\, d\nu(x).$$

</div>

### Back to Brunn–Minkowski and Prékopa–Leindler Inequalities

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pointwise Bounds Imply Brunn–Minkowski)</span></p>

Theorem 19.4 directly re-proves the distorted Brunn–Minkowski inequality (Theorem 18.5). Let $\mu\_0$ be $\nu$ conditioned on $A\_0$ (with density $\rho\_0 = 1\_{A\_0}/\nu[A\_0]$), and similarly $\mu\_1$ conditioned on $A\_1$. Then $\rho\_t$ is supported in $A\_t = [A\_0, A\_1]\_t$, and Theorem 19.4 gives

$$\rho_t(x)^{-1/N} \ge (1-t) \left[\inf \beta_{1-t}(x_0, x_1)^{1/N}\right] \nu[A_0]^{1/N} + t \left[\inf \beta_t(x_0, x_1)^{1/N}\right] \nu[A_1]^{1/N}.$$

Integrating $\rho\_t(x)^{1-1/N}\, d\nu(x)$ and applying Jensen's inequality (as in the proof of Theorem 18.5) yields the inequality.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19.16</span><span class="math-callout__name">(Prékopa–Leindler Inequality, Dimension-Free)</span></p>

With the same notation as in Theorem 19.4, assume that $(M, \nu)$ satisfies $\mathrm{CD}(K, \infty)$. Let $t \in (0,1)$, and let $f$, $g$, $h$ be three nonnegative functions such that

$$h(x) \ge \sup_{x \in [x_0, x_1]_t} f(x_0)^{1-t}\, g(x_1)^t \exp\!\left(-\frac{Kt(1-t)}{2}\, d(x_0, x_1)^2\right)$$

for all $x \in M$. Then

$$\int h\, d\nu \ge \left(\int f\, d\nu\right)^{1-t} \left(\int g\, d\nu\right)^t.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 19.17</span><span class="math-callout__name">(Square-Exponential Moments from Prékopa–Leindler)</span></p>

Let $(M, \nu)$ satisfy $\mathrm{CD}(K, \infty)$ with $K > 0$, and let $A \subset M$ be a compact set with $\nu[A] > 0$. Apply the Prékopa–Leindler inequality with $t = 1/2$, $f = 1\_A$, $g = \exp(K\, d(x, A)^2/4)$ and $h = 1$. This shows that

$$\int_M e^{K\, d(x, A)^2/4}\, d\nu(x) < +\infty,$$

recovering the square-exponential moment bound of Theorem 18.12.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19.18</span><span class="math-callout__name">(Finite-Dimension Distorted Prékopa–Leindler Inequality)</span></p>

With the same notation as in Theorem 19.4, assume that $(M, \nu)$ satisfies $\mathrm{CD}(K, N)$ for some $K \in \mathbb{R}$, $N \in (1, \infty)$. Let $f$, $g$, $h$ be three nonneg. functions on $M$ satisfying

$$h(x) \ge \sup_{x \in [x_0, x_1]_t} \mathcal{M}_t^q\!\left(\frac{f(x_0)}{\beta_{1-t}^{(K,N)}(x_0, x_1)},\; \frac{g(x_1)}{\beta_t^{(K,N)}(x_0, x_1)}\right), \qquad q \ge -\frac{1}{N},$$

where $\mathcal{M}\_t^q(a,b) := \bigl[(1-t)\, a^q + t\, b^q\bigr]^{1/q}$ (with $\mathcal{M}\_t^q(a,b) = 0$ if $a$ or $b$ is $0$, and $\mathcal{M}\_t^{-\infty}(a,b) = \min(a,b)$). Then

$$\int h\, d\nu \ge \mathcal{M}_t^{q/(1+Nq)}\!\left(\int f\, d\nu,\; \int g\, d\nu\right).$$

</div>

### Bibliographical Notes on Chapter 19

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

The main historical references on interior regularity estimates are **De Giorgi**, **Nash**, and **Moser**. It is now accepted that the two key ingredients underlying their methods are a doubling inequality and a local Poincaré inequality.

Local Poincaré inequalities admit many variants (sometimes called "weak" Poincaré inequalities). The inequality (19.1) is of type $(1,1)$ and implies the other main variants. Applied to the whole space, it is equivalent to **Cheeger's isoperimetric inequality**: $\nu[\Omega] \le 1/2 \implies \lvert \partial\Omega \rvert\_\nu \ge K\, \nu[\Omega]$.

The "intrinsic" bounds (Theorem 19.8) go back to Cordero-Erausquin, McCann, Schmuckenschläger (compactly supported case). The restriction strategy used to prove Theorems 19.4 and 19.8 is an amplification of the transport-based proof of Theorem 19.6 from **Lott–Villani**.

The democratic condition $\mathrm{Dm}(C)$ was explicitly introduced in Lott–Villani but is somehow implicit in earlier works (Cheeger–Colding). The general proof strategy behind Theorem 19.13 is rather classical, in the context of Riemannian manifolds, groups, or graphs.

The classical **Prékopa–Leindler inequality** in Euclidean space goes back to Prékopa and Rinott; see Gardner for references and its role in the Brunn–Minkowski theory. Although in principle equivalent to the Brunn–Minkowski inequality, it is sometimes more useful (e.g., Maurey's concentration inequalities). **Bobkov–Ledoux** showed how to use it to derive logarithmic Sobolev inequalities (Chapter 21). The Prékopa–Leindler inequality on manifolds (Theorem 19.16) was established by **Cordero-Erausquin, McCann, and Schmuckenschläger**.

</div>

## Chapter 20: Infinitesimal Displacement Convexity

The goal of this chapter is to translate displacement convexity inequalities of the form "the graph of a convex function lies below the chord" into inequalities of the form "the graph of a convex function lies above the tangent" — just as in statements (ii) and (iii) of Proposition 16.2. This corresponds to taking the limit $t \to 0$ in the convexity inequality. The main results are the **HWI inequality** (Corollary 20.13) and the **distorted HWI inequality** (Theorem 20.10).

### Gradient Norms on Metric Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient Norm and Descending Gradient Norm)</span></p>

On a nonsmooth length space, even though there is no natural notion for the gradient $\nabla f$ of a function $f$, there are natural definitions for the *norm* of the gradient $\lvert \nabla f \rvert$:

$$\lvert \nabla f \rvert(x) := \limsup_{y \to x} \frac{\lvert f(y) - f(x) \rvert}{d(x, y)}.$$

A slightly finer notion is the **descending gradient norm**:

$$\lvert \nabla^- f \rvert(x) := \limsup_{y \to x} \frac{[f(y) - f(x)]_-}{d(x, y)},$$

where $a\_- = \max(-a, 0)$. Obviously $\lvert \nabla^- f \rvert \le \lvert \nabla f \rvert$, both notions coincide with the usual one if $f$ is differentiable, and $\lvert \nabla^- f \rvert(x) = 0$ automatically if $x$ is a local minimum of $f$.

</div>

### Time-Derivative of the Energy

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 20.1</span><span class="math-callout__name">(Differentiating an Energy Along Optimal Transport)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a locally compact, complete geodesic space with a locally finite measure $\nu$. Let $U : [0, +\infty) \to \mathbb{R}$ be continuous convex, twice differentiable on $(0, +\infty)$. Let $(\mu\_t)\_{0 \le t \le 1}$ be a geodesic in $P\_2(\mathcal{X})$, with each $\mu\_t$ absolutely continuous w.r.t. $\nu$, with density $\rho\_t$, and $U(\rho\_t)\_-$ $\nu$-integrable for all $t$. Further assume that $\rho\_0$ is Lipschitz continuous, $U(\rho\_0)$, $\rho\_0\, U'(\rho\_0)$ are $\nu$-integrable, and $U'$ is Lipschitz continuous on $\rho\_0(\mathcal{X})$. Then

$$\liminf_{t \downarrow 0} \frac{U_\nu(\mu_t) - U_\nu(\mu_0)}{t} \ge -\int_{\mathcal{X}} U''(\rho_0(x_0))\, \lvert \nabla^- \rho_0 \rvert(x_0)\, d(x_0, x_1)\, \pi(dx_0\, dx_1),$$

where $\pi$ is an optimal coupling of $(\mu\_0, \mu\_1)$ associated with the geodesic path $(\mu\_t)\_{0 \le t \le 1}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 20.2</span><span class="math-callout__name">(Technical Assumptions)</span></p>

The assumption on the negative part $U(\rho\_t)\_-$ being integrable ensures that $U\_\nu(\mu\_t)$ is well-defined in $\mathbb{R} \cup \lbrace +\infty \rbrace$. The assumption about $U'$ being Lipschitz on $\rho\_0(\mathcal{X})$ means in practice that either $U$ is twice differentiable at the origin, or $\rho\_0$ is bounded away from $0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 20.3</span><span class="math-callout__name">(Probabilistic Reformulation)</span></p>

Let $\gamma$ be a random geodesic such that $\mu\_t = \mathrm{law}(\gamma\_t)$. Then Theorem 20.1 can be restated as:

$$\liminf_{t \downarrow 0} \frac{U_\nu(\mu_t) - U_\nu(\mu_0)}{t} \ge -\mathbb{E}\bigl[U''(\rho_0(\gamma_0))\, \lvert \nabla^- \rho_0 \rvert(\gamma_0)\, d(\gamma_0, \gamma_1)\bigr].$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 20.1)</span></p>

By convexity of $U$: $U(\rho\_t) - U(\rho\_0) \ge U'(\rho\_0)(\rho\_t - \rho\_0)$. Integrating against $\nu$:

$$U_\nu(\mu_t) - U_\nu(\mu_0) \ge \int U'(\rho_0)\, d\mu_t - \int U'(\rho_0)\, d\mu_0.$$

Writing $\mu\_t = \mathrm{law}(\gamma\_t)$: the right-hand side equals $\mathbb{E}[U'(\rho\_0(\gamma\_t)) - U'(\rho\_0(\gamma\_0))]$. Since $U'$ is nondecreasing:

$$U'(\rho_0(\gamma_t)) - U'(\rho_0(\gamma_0)) \ge [U'(\rho_0(\gamma_t)) - U'(\rho_0(\gamma_0))]\, 1_{\rho_0(\gamma_0) > \rho_0(\gamma_t)}.$$

After dividing by $t$ and using $d(\gamma\_0, \gamma\_t) = t\, d(\gamma\_0, \gamma\_1)$, passing to the limit via Fatou's lemma yields the result with the difference quotient of $U'$ converging to $U''$ and the ratio $(\rho\_0(\gamma\_t) - \rho\_0(\gamma\_0)) / d(\gamma\_0, \gamma\_t)$ bounded by $-\lvert \nabla^- \rho\_0 \rvert(\gamma\_0)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 20.4</span><span class="math-callout__name">(Riemannian Refinement)</span></p>

When $\mathcal{X}$ is a Riemannian manifold of dimension $n$, $\nu = e^{-V}\, \mathrm{vol}$, and $\mu$ is compactly supported, there is a more precise result: if $\psi$ is such that $T = \exp(\nabla \psi)$ is the optimal transport map from $\mu\_0$ to $\mu\_1$, and $L\psi = \Delta\psi - \nabla V \cdot \nabla\psi$, then

$$\lim_{t \to 0} \frac{U_\nu(\mu_t) - U_\nu(\mu_0)}{t} = -\int p(\rho_0)(L\psi)\, d\nu,$$

where $p(r) = r\, U'(r) - U(r)$. By integration by parts, $-\int p(\rho\_0)\, L\psi\, d\nu \ge \int \rho\_0\, U''(\rho\_0)\, \nabla\rho\_0 \cdot \nabla\psi\, d\nu$, which is an upper bound for the right-hand side of (20.3) since $\lvert \nabla\psi(x\_0) \rvert = d(x\_0, x\_1)$.

</div>

### HWI Inequalities

The name "HWI" comes from the three quantities involved:
* the **H**-functional (Boltzmann entropy): $H\_\nu(\mu) = \int \rho \log \rho\, d\nu$;
* the **W**asserstein distance of order 2, $W\_2$;
* the Fisher **I**nformation: $I\_\nu(\mu) = \int \lvert \nabla\rho \rvert^2 / \rho\, d\nu$.

The HWI inequality is obtained by combining the time-derivative bound (Theorem 20.1) with displacement convexity. For example, if $M$ satisfies $\mathrm{CD}(0, \infty)$ so that $H\_\nu$ is displacement convex, then the convexity inequality $(H\_\nu(\mu\_t) - H\_\nu(\mu\_0))/t \le H\_\nu(\mu\_1) - H\_\nu(\mu\_0)$ combined with the time-derivative bound and Cauchy–Schwarz gives

$$H_\nu(\mu_0) - H_\nu(\mu_1) \le W_2(\mu_0, \mu_1)\, \sqrt{I_\nu(\mu_0)}.$$

#### Generalized Fisher Information

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 20.6</span><span class="math-callout__name">(Generalized Fisher Information)</span></p>

Let $U$ be a continuous convex function $\mathbb{R}\_+ \to \mathbb{R}$, twice continuously differentiable on $(0, +\infty)$. Let $M$ be a Riemannian manifold with Borel reference measure $\nu$. For $\mu \in P^{\mathrm{ac}}(M)$ with locally Lipschitz density $\rho$, define the **generalized Fisher information**:

$$I_{U,\nu}(\mu) = \int \rho\, U''(\rho)^2\, \lvert \nabla\rho \rvert^2\, d\nu = \int \frac{\lvert \nabla p(\rho) \rvert^2}{\rho}\, d\nu = \int \rho\, \lvert \nabla U'(\rho) \rvert^2\, d\nu,$$

where $p(r) = r\, U'(r) - U(r)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classical Fisher Information)</span></p>

When $U(r) = r \log r$ (Boltzmann entropy), then $p(r) = r$, $U''(r) = 1/r$, and

$$I_\nu(\mu) = \int \frac{\lvert \nabla\rho \rvert^2}{\rho}\, d\nu.$$

This is the classical **Fisher information**, introduced by Fisher as part of his theory of "efficient statistics". It plays a crucial role in the Cramér–Rao inequality, in the asymptotic variance of the maximum likelihood estimate, and in large deviations of heat-like equations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 20.8</span><span class="math-callout__name">(Chain Rule Identity)</span></p>

The identity in Definition 20.6 comes from the chain rule:

$$\nabla p(\rho) = p'(\rho)\, \nabla\rho = \rho\, U''(\rho)\, \nabla\rho = \rho\, \nabla U'(\rho).$$

One can also replace $\lvert \nabla\rho \rvert$ by $\lvert \nabla^- \rho \rvert$ and $\lvert \nabla p(\rho) \rvert$ by $\lvert \nabla^- p(\rho) \rvert$ since a locally Lipschitz function is differentiable almost everywhere.

</div>

#### Distortion Coefficients at $t = 0$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Distortion Coefficients at $t = 0$ and $t = 1$)</span></p>

For the HWI inequality, we need the reference distortion coefficients $\beta\_t^{(K,N)}$ and their derivatives evaluated at $t = 0$ and $t = 1$. Write $\beta(x\_0, x\_1) = \beta\_0^{(K,N)}(x\_0, x\_1)$ and $\beta'(x\_0, x\_1) = (\beta\_1^{(K,N)})'(x\_0, x\_1)$, where the prime denotes differentiation with respect to $t$. With $\alpha = \sqrt{\lvert K \rvert/(N-1)}\, d(x\_0, x\_1)$:

$$\beta(x_0, x_1) = \begin{cases} (\alpha / \sin\alpha)^{N-1} > 1 & \text{if } K > 0, \\\\ 1 & \text{if } K = 0, \\\\ (\alpha / \sinh\alpha)^{N-1} < 1 & \text{if } K < 0, \end{cases}$$

$$\beta'(x_0, x_1) = \begin{cases} -(N-1)(1 - \alpha/\tan\alpha) < 0 & \text{if } K > 0, \\\\ 0 & \text{if } K = 0, \\\\ (N-1)(\alpha/\tanh\alpha - 1) > 0 & \text{if } K < 0. \end{cases}$$

As $\alpha \to 0$ (i.e. $d(x\_0, x\_1) \to 0$ or $N \to \infty$): $\beta \simeq 1 - \frac{K}{6}\, d(x\_0, x\_1)^2$ and $\beta' \simeq -\frac{K}{3}\, d(x\_0, x\_1)^2$.

</div>

#### The Distorted HWI Inequality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 20.10</span><span class="math-callout__name">(Distorted HWI Inequality)</span></p>

Let $M$ be a Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying $\mathrm{CD}(K, N)$ for some $K \in \mathbb{R}$, $N \in (1, \infty]$. Let $U \in \mathcal{DC}\_N$ and let $p(r) = r\, U'(r) - U(r)$. Let $\mu\_0 = \rho\_0\, \nu$ and $\mu\_1 = \rho\_1\, \nu$ be two absolutely continuous probability measures satisfying suitable integrability and regularity conditions. Then:

$$\int_M U(\rho_0)\, d\nu \le \int_{M \times M} U\!\left(\frac{\rho_1(x_1)}{\beta(x_0, x_1)}\right) \beta(x_0, x_1)\, \pi(dx_0 \lvert x_1)\, \nu(dx_1) + \int_{M \times M} p(\rho_0(x_0))\, \beta'(x_0, x_1)\, \pi(dx_1 \lvert x_0)\, \nu(dx_0) + \int_{M \times M} U''(\rho_0(x_0))\, \lvert \nabla\rho_0(x_0) \rvert\, d(x_0, x_1)\, \pi(dx_0\, dx_1),$$

where $\pi$ is the unique optimal coupling, $\beta = \beta\_0^{(K,N)}$, and $\beta' = (\beta\_1^{(K,N)})'$ are defined in (20.9)–(20.10).

In particular:

**(i)** If $K = 0$ and $U\_\nu(\mu\_1) < +\infty$:

$$U_\nu(\mu_0) - U_\nu(\mu_1) \le \int U''(\rho_0(x_0))\, \lvert \nabla\rho_0(x_0) \rvert\, d(x_0, x_1)\, \pi(dx_0\, dx_1) \le W_2(\mu_0, \mu_1)\, \sqrt{I_{U,\nu}(\mu_0)}.$$

**(ii)** If $N = \infty$ and $U\_\nu(\mu\_1) < +\infty$:

$$U_\nu(\mu_0) - U_\nu(\mu_1) \le W_2(\mu_0, \mu_1)\, \sqrt{I_{U,\nu}(\mu_0)} - K_{\infty, U}\, \frac{W_2(\mu_0, \mu_1)^2}{2},$$

where $K\_{\infty, U}$ is defined in (17.10).

**(iii)** If $N < \infty$, $K \ge 0$ and $U\_\nu(\mu\_1) < +\infty$:

$$U_\nu(\mu_0) - U_\nu(\mu_1) \le W_2(\mu_0, \mu_1)\, \sqrt{I_{U,\nu}(\mu_0)} - K\, \lambda_{N,U}\, \max\!\bigl(\lVert \rho_0 \rVert_{L^\infty},\, \lVert \rho_1 \rVert_{L^\infty}\bigr)^{-1/N}\, \frac{W_2(\mu_0, \mu_1)^2}{2},$$

where $\lambda\_{N,U} = \lim\_{r \to 0} p(r) / r^{1-1/N}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 20.12</span><span class="math-callout__name">(Extension to Negative Curvature)</span></p>

Theorem 20.10(iii) extends to negative curvature modulo the following changes: replace $\lim\_{r \to 0}$ in (20.16) by $\lim\_{r \to \infty}$; and $\max(\lVert \rho\_0 \rVert\_{L^\infty}, \lVert \rho\_1 \rVert\_{L^\infty})^{-1/N}$ in (20.15) by $\max(\lVert 1/\rho\_0 \rVert\_{L^\infty}, \lVert 1/\rho\_1 \rVert\_{L^\infty})^{1/N}$. This result is not easy to derive by plain displacement convexity alone.

</div>

#### The HWI Inequality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 20.13</span><span class="math-callout__name">(HWI Inequalities)</span></p>

Let $M$ be a Riemannian manifold with $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying $\mathrm{CD}(K, \infty)$ for some $K \in \mathbb{R}$. Then:

**(i)** Let $p \in [2, +\infty) \cup \lbrace c \rbrace$ satisfy (17.30) for $N = \infty$, and let $\mu\_0 = \rho\_0\, \nu$, $\mu\_1 = \rho\_1\, \nu$ be two probability measures in $P\_p^{\mathrm{ac}}(M)$, with $H\_\nu(\mu\_1) < +\infty$ and $\rho\_0$ Lipschitz. Then

$$H_\nu(\mu_0) - H_\nu(\mu_1) \le W_2(\mu_0, \mu_1)\, \sqrt{I_\nu(\mu_0)} - K\, \frac{W_2(\mu_0, \mu_1)^2}{2}.$$

**(ii)** If $\nu \in P\_2(M)$ then for any $\mu \in P\_2(M)$,

$$H_\nu(\mu) \le W_2(\mu, \nu)\, \sqrt{I_\nu(\mu)} - K\, \frac{W_2(\mu, \nu)^2}{2}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 20.14</span><span class="math-callout__name">(HWI as Interpolation Inequality)</span></p>

The HWI inequality plays the role of a nonlinear interpolation inequality: it shows that the Kullback information $H$ is controlled by a bit of the Fisher information $I$ (which is stronger, involving smoothness) and the Wasserstein distance $W\_2$ (which is weaker). A related "linear" interpolation inequality is $\lVert h \rVert\_{L^2} \le \sqrt{\lVert h \rVert\_{H^{-1}}\, \lVert h \rVert\_{H^1}}$, where $H^1$ is the Sobolev space and $H^{-1}$ its dual.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Corollary 20.13)</span></p>

Statement (i) follows from Theorem 20.10 by choosing $N = \infty$ and $U(r) = r \log r$. Statement (ii) is obtained by approximation: find a sequence of Lipschitz densities $\rho\_{0,k} \to \rho\_0$ such that $H\_\nu(\rho\_{0,k}\, \nu) \to H\_\nu(\mu)$, $W\_2(\rho\_{0,k}\, \nu, \nu) \to W\_2(\mu, \nu)$, and $I\_\nu(\rho\_{0,k}\, \nu) \to I\_\nu(\mu)$.

</div>

### Proof of Theorem 20.10

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Strategy for Theorem 20.10)</span></p>

The proof proceeds in three steps:

**Step 1 (Nice $U$ and $\beta$).** Assume $U$ is Lipschitz (if $N < \infty$) or $U(r) = O(r \log(2+r))$ (if $N = \infty$), and $\beta$, $\beta'$ are bounded. Start from the distorted displacement convexity inequality (Theorem 17.37):

$$\int U(\rho_t)\, d\nu \le (1-t) \int U\!\left(\frac{\rho_0}{\beta_{1-t}}\right) \beta_{1-t}\, \pi\, \nu + t \int U\!\left(\frac{\rho_1}{\beta_t}\right) \beta_t\, \pi\, \nu.$$

Rearrange this into four terms and pass to the limit $t \to 0$. The first two terms converge to $\int U(\rho\_0)\, d\nu$ and $\int U(\rho\_1/\beta)\, \beta\, \pi\, \nu$ by monotone convergence. The third term contributes $\int p(\rho\_0)\, \beta'\, \pi\, \nu$ (using the convexity of $b \mapsto U(r/b)$ and the monotone convergence theorem). The fourth term contributes $\int U''(\rho\_0)\, \lvert \nabla\rho\_0 \rvert\, d(x\_0, x\_1)\, \pi$ by Theorem 20.1.

**Step 2 (Relaxation on $U$).** Approximate $U$ by a sequence $(U\_\ell)$ that coincides with $U$ on $[\ell^{-1}, \ell]$ and has better regularity properties. Pass to the limit using monotone/dominated convergence.

**Step 3 (Relaxation on $\beta$).** When $K > 0$ and $\mathrm{diam}(M) = D\_{K,N} = \pi\sqrt{(N-1)/K}$, the coefficients $\beta$, $\beta'$ may be unbounded. Replace $N$ by $N' > N$ (for which $\beta^{(K,N')}$, $(\beta^{(K,N')})' $ are bounded), establish (20.12) with $N'$, and pass to the limit as $N' \downarrow N$. This works because $\beta^{(K,N')}$ is increasing in $N'$, so $U(\rho\_1/\beta^{(K,N')})\, \beta^{(K,N')}$ is decreasing, and $(\beta^{(K,N')})\_1'$ is decreasing.

The particular cases (i)–(iii) are then obtained by specializing $K$, $N$ and applying Cauchy–Schwarz.

</div>

### Bibliographical Notes on Chapter 20

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

**Fisher information** was introduced by Fisher as part of his theory of "efficient statistics". It plays a crucial role in the Cramér–Rao inequality, determines the asymptotic variance of the maximum likelihood estimate, and appears in the rate function for large deviations of solutions of heat-like equations. The Boltzmann–Gibbs–Shannon–Kullback information ($H$) and the Fisher information ($I$) play leading roles in information theory and in statistical mechanics/kinetic theory.

The **HWI inequality** was established in the joint work of **Otto and Villani**; it obviously extends to any reasonable $K$-displacement convex functional. A precursor inequality was studied by Otto. It has been applied to PDEs, uniqueness for spin systems (Gao–Wu), and to study convergence rates of nonlinear PDEs by combining Fisher information bounds with Wasserstein convergence estimates.

The HWI inequality is also interesting as an "infinite-dimensional" interpolation inequality, with applications to the study of the limit behavior of entropy in hydrodynamic limits.

Alternative derivations of the HWI inequality are due to **Cordero-Erausquin** and **Bobkov, Gentil, and Ledoux**. The first systematic studies of HWI-type inequalities in the case $N < \infty$ are due to **Lott and Villani**. The elementary inequalities (20.32) and (20.34) are used to derive the Lichnerowicz spectral gap inequality (Theorem 21.20 in Chapter 21).

Theorem 20.1 applies to nonsmooth spaces, which will be important in Part III. The argument is taken from the joint work with **Lott**.

</div>

## Chapter 21: Isoperimetric-Type Inequalities

Several inequalities with isoperimetric content can be retrieved by considering the above-tangent formulation of displacement convexity. The idea is heuristic: assume the initial measure is the normalized indicator function of some set $A$. Think of the functional $U\_\nu$ as the internal energy of some fluid initially confined in $A$. In a displacement interpolation, some mass flows out of $A$, leading to a variation of the energy related to the surface of $A$. By controlling the decrease of energy, one should eventually gain control of the surface of $A$.

The functional nature of this approach makes it possible to replace the set $A$ by an arbitrary probability measure $\mu = \rho\, \nu$. Then what plays the role of the "surface" of $A$ is some integral expression involving $\nabla \rho$. Any inequality expressing the domination of an integral expression of $\rho$ by an integral expression of $\rho$ and $\nabla \rho$ will be loosely referred to as a **Sobolev-type**, or **isoperimetric-type** inequality.

### Logarithmic Sobolev Inequalities

A probability measure $\nu$ on a Riemannian manifold is said to satisfy a logarithmic Sobolev inequality if the functional $H\_\nu$ is dominated by (a constant multiple of) the functional $I\_\nu$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 21.1</span><span class="math-callout__name">(Logarithmic Sobolev Inequality)</span></p>

Let $M$ be a Riemannian manifold, and $\nu$ a probability measure on $M$. It is said that $\nu$ satisfies a **logarithmic Sobolev inequality with constant $\lambda$** if, for any probability measure $\mu = \rho\, \nu$ with $\rho$ Lipschitz, one has

$$H_\nu(\mu) \le \frac{1}{2\lambda}\, I_\nu(\mu). \tag{21.1}$$

Explicitly, inequality (21.1) means

$$\int \rho \log \rho\, d\nu \le \frac{1}{2\lambda} \int \frac{\lvert \nabla \rho \rvert^2}{\rho}\, d\nu. \tag{21.2}$$

Equivalently, for any function $u$ (regular enough) one should have

$$\int u^2 \log(u^2)\, d\nu - \left(\int u^2\, d\nu\right) \log\left(\int u^2\, d\nu\right) \le \frac{2}{\lambda} \int \lvert \nabla u \rvert^2\, d\nu. \tag{21.3}$$

To go from (21.2) to (21.3), set $\rho = u^2 / (\int u^2\, d\nu)$ and notice that $\nabla \lvert u \rvert \le \lvert \nabla u \rvert$.

</div>

The Lipschitz regularity of $\rho$ allows one to define $\lvert \nabla \rho \rvert$ pointwise, for instance by means of (20.1). Everywhere in this chapter, $\lvert \nabla \rho \rvert$ may also be replaced by the quantity $\lvert \nabla^- \rho \rvert$ appearing in (20.2); in fact both expressions coincide almost everywhere if $u$ is Lipschitz.

This restriction of Lipschitz continuity is unnecessary, and can be relaxed with a bit of work. For instance, if $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, then one can use distribution theory to show that the quantity $\int \lvert \nabla \rho \rvert^2 / \rho\, d\nu$ is well-defined in $[0, +\infty]$, and then (21.1) makes sense.

Logarithmic Sobolev inequalities are **dimension-free** Sobolev-type inequalities: the dimension of the space does not appear explicitly in (21.3). This is one reason why these inequalities are extremely popular in various branches of statistical mechanics, mathematical statistics, quantum field theory, and more generally the study of phenomena in high or infinite dimension. They are also used in geometry and partial differential equations, including Perelman's work on the Ricci flow and the Poincaré conjecture.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21.2</span><span class="math-callout__name">(Bakry–Émery Theorem)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference probability measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying a curvature assumption $\mathrm{CD}(K, \infty)$ for some $K > 0$. Then $\nu$ satisfies a logarithmic Sobolev inequality with constant $K$, i.e.

$$H_\nu \le \frac{I_\nu}{2K}. \tag{21.4}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 21.3</span><span class="math-callout__name">(Stam–Gross Logarithmic Sobolev Inequality)</span></p>

For the Gaussian measure $\gamma(dx) = (2\pi)^{-n/2} e^{-\lvert x \rvert^2/2}$ in $\mathbb{R}^n$, one has

$$H_\gamma \le \frac{I_\gamma}{2}, \tag{21.5}$$

independently of the dimension. This is the **Stam–Gross logarithmic Sobolev inequality**. By scaling, for any $K > 0$ the measure $\gamma\_K(dx) = (2\pi/K)^{-n/2} e^{-K\lvert x \rvert^2 / 2}\, dx$ satisfies a logarithmic Sobolev inequality with constant $K$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21.4</span></p>

More generally, if $V \in C^2(\mathbb{R}^n)$ and $\nabla^2 V \ge K I\_n$, Theorem 21.2 shows that $\nu(dx) = e^{-V(x)}\, dx$ satisfies a logarithmic Sobolev inequality with constant $K$. When $V(x) = K\lvert x \rvert^2 / 2$ the constant $K$ is optimal in (21.4).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21.5</span><span class="math-callout__name">(Perturbation)</span></p>

The curvature assumption $\mathrm{CD}(K, \infty)$ is quite restrictive; however, there are known perturbation theorems which immediately extend the range of application of Theorem 21.2. For instance, if $\nu$ satisfies a logarithmic Sobolev inequality, $v$ is a bounded function and $\widetilde{\nu} = e^{-v}\, \nu / Z$ is another probability measure obtained from $\nu$ by multiplication by $e^{-v}$, then also $\widetilde{\nu}$ satisfies a logarithmic Sobolev inequality (**Holley–Stroock perturbation theorem**). The same is true if $v$ is unbounded, but satisfies $\int e^{\alpha \lvert \nabla v \rvert^2}\, d\nu < \infty$ for $\alpha$ large enough.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 21.2)</span></p>

By Theorem 18.12, $\nu$ admits square-exponential moments, in particular it lies in $P\_2(M)$. Then from Corollary 20.13(ii) and the inequality $ab \le Ka^2/2 + b^2/(2K)$,

$$H_\nu(\mu) \le W_2(\mu, \nu)\, \sqrt{I_\nu(\mu)} - \frac{K\, W_2(\mu, \nu)^2}{2} \le \frac{I_\nu(\mu)}{2K}. \qquad \square$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Open Problem 21.6</span></p>

For Riemannian manifolds satisfying $\mathrm{CD}(K, N)$ with $N < \infty$, the optimal constant in the logarithmic Sobolev inequality is not $K$ but $KN/(N-1)$. Can this be proven by a transport argument?

</div>

### Sobolev–$L^\infty$ Interpolation Inequalities

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21.7</span><span class="math-callout__name">(Sobolev–$L^\infty$ Interpolation Inequalities)</span></p>

Let $M$ be a Riemannian manifold, equipped with a reference probability measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying a $\mathrm{CD}(K, N)$ curvature-dimension bound for some $K > 0$, $N \in (1, \infty]$. Further, let $U \in \mathcal{DC}\_N$. Then, for any Lipschitz-continuous probability density $\rho$, if $\mu = \rho\, \nu$, one has the inequality

$$0 \le U_\nu(\mu) - U_\nu(\nu) \le \frac{(\sup \rho)^{1/N}}{2K\lambda}\, I_{U,\nu}(\mu), \tag{21.6}$$

where

$$\lambda = \lim_{r \to 0} \left(\frac{p(r)}{r^{1 - 1/N}}\right).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 21.7)</span></p>

The proof of the inequality on the right-hand side of (21.6) is the same as for Theorem 21.2, using Theorem 20.10(iii). The inequality on the left-hand side is a consequence of Jensen's inequality: $U\_\nu(\mu) = \int U(\rho)\, d\nu \ge U(\int \rho\, d\nu) = U(1) = U\_\nu(\nu)$. $\square$

</div>

### Sobolev Inequalities

Sobolev inequalities are one among several classes of functional inequalities with isoperimetric content; they are extremely popular in the theory of partial differential equations. They look like logarithmic Sobolev inequalities, but with powers instead of logarithms, and they take dimension into account explicitly.

The most basic Sobolev inequality is in Euclidean space: If $u$ is a function on $\mathbb{R}^n$ such that $\nabla u \in L^p(\mathbb{R}^n)$ ($1 \le p < n$) and $u$ vanishes at infinity, then $u$ automatically lies in $L^{p^\star}(\mathbb{R}^n)$ where $p^\star = (np)/(n - p) > p$. More quantitatively, there is a constant $S = S(n, p)$ such that

$$\lVert u \rVert_{L^{p^\star}(\mathbb{R}^n)} \le S\, \lVert \nabla u \rVert_{L^p(\mathbb{R}^n)}.$$

There are also very many variants for a function $u$ defined on a set $\Omega$ that might be a reasonable open subset of either $\mathbb{R}^n$ or a Riemannian manifold $M$. One can also quote the **Gagliardo–Nirenberg interpolation inequalities**, which typically take the form

$$\lVert u \rVert_{L^{p^\star}} \le G\, \lVert \nabla u \rVert_{L^p}^{1-\theta}\, \lVert u \rVert_{L^q}^{\theta}, \qquad 1 \le p < n, \quad 1 \le q < p^\star, \quad 0 \le \theta \le 1,$$

with some restrictions on the exponents.

In a Riemannian setting, there is a famous family of Sobolev inequalities obtained from the curvature-dimension bound $\mathrm{CD}(K, N)$ with $K > 0$ and $2 < N < \infty$:

$$1 \le q \le \frac{2N}{N-2} \implies \frac{c}{q-2} \left[\left(\int \lvert u \rvert^q\, d\nu\right)^{2/q} - \int \lvert u \rvert^2\, d\nu\right] \le \int \lvert \nabla u \rvert^2\, d\nu, \quad c = \frac{NK}{N-1}. \tag{21.7}$$

When $q \to 2$, (21.7) reduces to Bakry–Émery's logarithmic Sobolev inequality. The other most interesting case is when $q$ coincides with the critical exponent $2^\star = (2N)/(N-2)$, and then (21.7) becomes

$$\lVert u \rVert^2_{L^{2N/(N-2)}(M)} \le \lVert u \rVert^2_{L^2(M)} + \left(\frac{4}{N-2}\right)\left(\frac{N-1}{KN}\right)\, \lVert \nabla u \rVert^2_{L^2(M)}. \tag{21.8}$$

There is no loss of generality in assuming $u \ge 0$. Setting $\rho = u^{2N/(N-2)}$ and $\mu := \rho\, \nu$, inequality (21.8) transforms into

$$H_{N/2,\nu}(\mu) = -\frac{N}{2} \int (\rho^{1 - 2/N} - \rho)\, d\nu \le \frac{1}{2K} \int \frac{\lvert \nabla \rho \rvert^2}{\rho} \left(\frac{(N-1)(N-2)}{N^2}\, \rho^{-2/N}\right) d\nu. \tag{21.9}$$

This has the merit of showing very clearly how the limit $N \to \infty$ leads to the logarithmic Sobolev inequality $H\_{\infty,\nu}(\mu) \le (1/2K) \int (\lvert \nabla \rho \rvert^2 / \rho)\, d\nu$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21.8</span></p>

It is possible that (21.8) implies (21.6) if $U = U\_N$. This would follow from the inequality

$$H_{N,\nu} \le \left(\frac{N-1}{N-2}\right) (\sup \rho)^{1/N}\, H_{N/2,\nu},$$

which should not be difficult to prove, or disprove.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21.9</span><span class="math-callout__name">(Sobolev Inequalities from $\mathrm{CD}(K, N)$)</span></p>

Let $M$ be a Riemannian manifold, equipped with a reference measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying a curvature-dimension inequality $\mathrm{CD}(K, N)$ for some $K > 0$, $1 < N < \infty$. Then, for any probability density $\rho$, Lipschitz continuous and strictly positive, and $\mu = \rho\, \nu$, one has

$$H_{N,\nu}(\mu) = -N \int_M (\rho^{1 - 1/N} - \rho)\, d\nu \le \int_M \Theta^{(N,K)}(\rho, \lvert \nabla \rho \rvert)\, d\nu, \tag{21.10}$$

where

$$\Theta^{(N,K)}(r, g) = r \sup_{0 \le \alpha \le \pi} \left(\frac{N-1}{N} \cdot \frac{g}{r^{1+1/N}} \sqrt{\frac{N-1}{K}}\, \alpha + N\left(1 - \left(\frac{\alpha}{\sin \alpha}\right)^{1 - 1/N}\right) + (N-1)\left(\frac{\alpha}{\tan \alpha} - 1\right) r^{-1/N}\right). \tag{21.11}$$

As a consequence,

$$H_{N,\nu}(\mu) \le \frac{1}{2K} \int_M \frac{\lvert \nabla \rho \rvert^2}{\rho} \left(\left(\frac{N-1}{N}\right)^2 \frac{\rho^{-2/N}}{\frac{1}{3} + \frac{2}{3}\, \rho^{-1/N}}\right) d\nu. \tag{21.12}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21.10</span></p>

By taking the limit as $N \to \infty$ in (21.12), one recovers again the logarithmic Sobolev inequality of Bakry and Émery, with the sharp constant. For fixed $N$, the exponents appearing in (21.12) are sharp: For large $\rho$, the integrand in the right-hand side behaves like $\lvert \nabla \rho \rvert^2 \rho^{-(1+2/N)} = c\_N \lvert \nabla \rho^{1/2^\star} \rvert^2$, so the critical Sobolev exponent $2^\star$ does govern this inequality. On the other hand, the *constants* appearing in (21.12) are definitely not sharp.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Open Problems 21.11</span></p>

Is inequality (21.10) stronger, weaker, or not comparable to inequality (21.9)? Does (21.12) follow from (21.9)? Can one find a transport argument leading to (21.9)?

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 21.9)</span></p>

Start from Theorem 20.10 and choose $U(r) = -N(r^{1-1/N} - r)$. After some straightforward calculations, one obtains

$$H_{N,\nu}(\mu) \le \int_M \theta^{(N,K)}(\rho, \lvert \nabla \rho \rvert, \alpha)\, d\nu,$$

where $\alpha = \sqrt{K/(N-1)}\, d(x\_0, x\_1) \in [0, \pi]$, and $\theta^{(N,K)}$ is an explicit function such that

$$\Theta^{(N,K)}(r, g) = \sup_{\alpha \in [0,\pi]} \theta^{(N,K)}(r, g, \alpha).$$

This is sufficient to prove (21.10). To go from (21.10) to (21.12), one can use the elementary inequalities (20.32) and (20.34) and compute the supremum explicitly. $\square$

</div>

### Sobolev Inequalities in $\mathbb{R}^n$

Now consider the case of the Euclidean space $\mathbb{R}^n$, equipped with the Lebesgue measure. Sharp Sobolev inequalities can be obtained by a transport approach, taking advantage of the scaling properties in $\mathbb{R}^n$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21.12</span><span class="math-callout__name">(Sobolev Inequalities in $\mathbb{R}^n$)</span></p>

Whenever $u$ is a Lipschitz, compactly supported function on $\mathbb{R}^n$, then

$$\lVert u \rVert_{L^{p^\star}(\mathbb{R}^n)} \le S_n(p)\, \lVert \nabla u \rVert_{L^p(\mathbb{R}^n)}, \qquad 1 \le p < n, \quad p^\star = \frac{np}{n - p}, \tag{21.13}$$

where the constant $S\_n(p)$ is given by

$$S_n(p) = \inf \left\lbrace \frac{p(n-1)}{n(n-p)} \cdot \frac{\left(\int \lvert g \rvert\right)^{1/p'} \left(\int \lvert y \rvert^{p'} \lvert g(y) \rvert\, dy\right)^{1/p'}}{\int \lvert g \rvert^{1 - 1/n}} \right\rbrace, \quad p' = \frac{p}{p-1},$$

and the infimum is taken over all functions $g \in L^1(\mathbb{R}^n)$, not identically $0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21.13</span></p>

The assumption of Lipschitz continuity for $u$ can be removed. Actually, inequality (21.13) holds true as soon as $u$ is locally integrable and vanishes at infinity, in the sense that the Lebesgue measure of $\lbrace \lvert u \rvert \ge r \rbrace$ is finite for any $r > 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21.14</span></p>

The constant $S\_n(p)$ in (21.13) is optimal.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 21.12)</span></p>

Choose $M = \mathbb{R}^n$, $\nu = $ Lebesgue measure, and apply Theorem 20.10 with $K = 0$, $N = n$, and $\mu\_0 = \rho\_0\, \nu$, $\mu\_1 = \rho\_1\, \nu$, both compactly supported. By formula (20.14) in Theorem 20.10(i),

$$H_{n,\nu}(\mu_0) - H_{n,\nu}(\mu_1) \le \left(1 - \frac{1}{n}\right) \int_{\mathbb{R}^n \times \mathbb{R}^n} \rho_0(x_0)^{-(1+1/n)}\, \lvert \nabla \rho_0 \rvert(x_0)\, d(x_0, x_1)\, \pi(dx_0\, dx_1).$$

Then Hölder's inequality and the marginal property of $\pi$ imply

$$H_{n,\nu}(\mu_0) - H_{n,\nu}(\mu_1) \le \left(1 - \frac{1}{n}\right) \left(\int_{\mathbb{R}^n} \rho_0^{-p(1+1/n)}\, \lvert \nabla \rho_0 \rvert^p\, d\mu_0\right)^{1/p} \left(\int_{\mathbb{R}^n \times \mathbb{R}^n} d(x_0, x_1)^{p'}\, \pi(dx_0\, dx_1)\right)^{1/p'}.$$

This can be rewritten as

$$n \int \rho_1^{1-1/n}\, d\nu \le n \int \rho_0^{1-1/n}\, d\nu + \left(1 - \frac{1}{n}\right) \left(\rho_0^{-p(1+1/n)}\, \lvert \nabla \rho_0 \rvert^p\, d\mu_0\right)^{1/p}\, W_{p'}(\mu_0, \mu_1). \tag{21.14}$$

Now use a homogeneity argument. Fix $\rho\_1$ and $\rho\_0$ as above, and define $\rho\_0^{(\lambda)}(x) = \lambda^n\, \rho\_0(\lambda x)$. Passing to the limit as $\lambda \to \infty$, the probability measure $\mu\_0^{(\lambda)} = \rho\_0^{(\lambda)}\, \nu$ converges weakly to the Dirac mass $\delta\_0$ at the origin, so

$$W_{p'}(\mu_0^{(\lambda)}, \mu_1) \longrightarrow W_{p'}(\delta_0, \mu_1) = \left(\int \lvert y \rvert^{p'}\, d\mu_1(y)\right)^{1/p'}.$$

After passing to the limit, one obtains

$$n \int \rho_1^{1-1/n}\, d\nu \le \left(\int \rho_0^{-p(1+1/n)}\, \lvert \nabla \rho_0 \rvert^p\, d\mu_0\right)^{1/p} \left(\int \lvert y \rvert^{p'}\, d\mu_1(y)\right)^{1/p'}. \tag{21.15}$$

Let us change unknowns and define $\rho\_0 = u^{1/p^\star}$, $\rho\_1 = g$; inequality (21.15) then becomes

$$1 \le \frac{p\,(n-1)}{n\,(n-p)} \left(\frac{\left(\int \lvert y \rvert^{p'}\, g(y)\, dy\right)^{1/p'}}{\int g^{(1-1/n)}}\right) \lVert \nabla u \rVert_{L^p},$$

where $u$ and $g$ are only required to satisfy $\int u^{p^\star} = 1$, $\int g = 1$. The inequality (21.13) follows by homogeneity again. $\square$

</div>

### $L^1$-Sobolev Inequalities from $\mathrm{CD}(K, N)$

To conclude this section, we assume $\mathrm{Ric}\_{N,\nu} \ge K < 0$ and derive Sobolev inequalities for compactly supported functions. The limit case $p = 1$, $p^\star = n/(n-1)$, implies the general inequality for $p < n$ via Hölder's inequality.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21.15</span><span class="math-callout__name">($\mathrm{CD}(K, N)$ Implies $L^1$-Sobolev Inequalities)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference measure $\nu = e^{-V}\, \mathrm{vol}$, satisfying a curvature-dimension bound $\mathrm{CD}(K, N)$ for some $K < 0$, $N \in (1, \infty)$. Then, for any ball $\mathcal{B} = B(z, R)$, $R \ge 1$, there are constants $A$ and $B$, only depending on a lower bound on $K$, and upper bounds on $N$ and $R$, such that for any Lipschitz function $u$ supported in $\mathcal{B}$,

$$\lVert u \rVert_{L^{N/(N-1)}} \le A\, \lVert \nabla u \rVert_{L^1} + B\, \lVert u \rVert_{L^1}. \tag{21.16}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 21.15)</span></p>

Inequality (21.16) remains unchanged if $\nu$ is multiplied by a positive constant. So we may assume, without loss of generality, that $\nu[B(z, R)] = 1$. Formula (20.12) in Theorem 20.10 implies

$$N - \int \rho_0^{1-1/N}\, d\nu \le N - \int \rho_1(x_1)^{1-1/N}\, \beta(x_0, x_1)^{1/N}\, \pi(dx_0 \mid x_1)\, \nu(dx_1) + \int \rho_0(x_0)^{1-1/N}\, \beta'(x_0, x_1)\, \pi(dx_1 \mid x_0)\, \nu(dx_0) + \frac{1}{N} \int \rho_0(x_0)^{-1-1/N}\, \lvert \nabla \rho_0 \rvert(x_0)\, d(x_0, x_1)\, \pi(dx_0\, dx_1). \tag{21.17}$$

Choose $\rho\_1 = 1\_{B(z,R)} / \nu[B(z, R)]$ (the normalized indicator function of the ball). The coefficients $\beta$ and $\beta'$ in (21.17) belong to $B(z, R)$, so they remain bounded by explicit functions of $N$, $K$ and $R$, while $d(x\_0, x\_1)$ remains bounded by $2R$. So there are constants $\delta(K, N, R) > 0$ and $\overline{C}(K, N, R)$ such that

$$-\int \rho_0^{1-1/N}\, d\nu \le -\delta(K, N, R)\, \nu[\mathcal{B}]^{1/N} + \overline{C}(K, N, R) \left[\int \rho_0^{1-1/N} + \int \rho_0^{-1/N}\, \lvert \nabla \rho_0 \rvert\right]. \tag{21.18}$$

Recall that $\nu[\mathcal{B}] = 1$; then after the change of unknowns $\rho\_0 = u^{N/(N-1)}$, inequality (21.18) implies

$$1 \le S(K, N, R)\, \left[\lVert \nabla u \rVert_{L^1(M)} + \lVert u \rVert_{L^1(M)}\right],$$

for some explicit constant $S = (\overline{C} + 1)/\delta$. This holds true under the constraint $1 = \int \rho = \int u^{N/(N-1)}$, and then inequality (21.16) follows by homogeneity. $\square$

</div>

### Isoperimetric Inequalities

Isoperimetric inequalities are sometimes obtained as limits of Sobolev inequalities applied to indicator functions. The most classical example is the equivalence between the Euclidean isoperimetric inequality

$$\frac{\lvert \partial A \rvert}{\lvert A \rvert^{(n-1)/n}} \ge \frac{\lvert \partial B^n \rvert}{\lvert B^n \rvert^{(n-1)/n}}$$

and the *optimal* Sobolev inequality $\lVert u \rVert\_{L^{n/(n-1)}(\mathbb{R}^n)} \le S\_n(1)\, \lVert \nabla u \rVert\_{L^1(\mathbb{R}^n)}$.

There is a proof of the optimal Sobolev inequality in $\mathbb{R}^n$ based on transport, which leads to a proof of the Euclidean isoperimetry. There is also a more direct transport-based proof of isoperimetry, as explained in Chapter 2.

Besides the Euclidean one, the most famous isoperimetric inequality in differential geometry is certainly the **Lévy–Gromov inequality**, which states that if $A$ is a reasonable set in a manifold $(M, g)$ with dimension $n$ and Ricci curvature bounded below by $K > 0$, then

$$\frac{\lvert \partial A \rvert}{\lvert M \rvert} \ge \frac{\lvert \partial B \rvert}{\lvert S \rvert},$$

where $B$ is a spherical cap in the model sphere $S$ (that is, the sphere with dimension $N$ and Ricci curvature $K$) such that $\lvert B \rvert / \lvert S \rvert = \lvert A \rvert / \lvert M \rvert$. In other words, isoperimetry in $M$ is at least as strong as isoperimetry in the model sphere.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Open Problem 21.16</span></p>

Find a transport-based, soft proof of the Lévy–Gromov isoperimetric inequality.

</div>

The same question can be asked for the **Gaussian isoperimetry**, which is the infinite-dimensional version of the Lévy–Gromov inequality. In this case however there are softer approaches based on functional versions.

### Poincaré Inequalities

Poincaré inequalities are related to Sobolev inequalities, and often appear as limit cases. Here only *global* Poincaré inequalities are considered, which are rather different from the local inequalities considered in Chapter 19.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 21.17</span><span class="math-callout__name">(Poincaré Inequality)</span></p>

Let $M$ be a Riemannian manifold, and $\nu$ a probability measure on $M$. It is said that $\nu$ satisfies a **Poincaré inequality with constant $\lambda$** if, for any $u \in L^2(\mu)$ with $u$ Lipschitz, one has

$$\lVert u - \langle u \rangle \rVert^2_{L^2(\nu)} \le \frac{1}{\lambda}\, \lVert \nabla u \rVert^2_{L^2(\nu)}, \qquad \langle u \rangle = \int u\, d\nu. \tag{21.19}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21.18</span></p>

Throughout Part II, it is always assumed that $\nu$ is absolutely continuous with respect to the volume measure. This implies that Lipschitz functions are $\nu$-almost everywhere differentiable.

</div>

Inequality (21.19) can be reformulated as

$$\left[\int u\, d\nu = 0\right] \implies \lVert u \rVert^2_{L^2} \le \frac{\lVert \nabla u \rVert^2_{L^2}}{\lambda}.$$

This makes the formal connection with the logarithmic Sobolev inequality very natural. (The Poincaré inequality is obtained as the limit of the logarithmic Sobolev inequality when one sets $\mu = (1 + \varepsilon u)\, \nu$ and lets $\varepsilon \to 0$.)

Like Sobolev inequalities, Poincaré inequalities express the domination of a function by its gradient; but unlike Sobolev inequalities, they do not include any gain of integrability. Poincaré inequalities have spectral content, since the best constant $\lambda$ can be interpreted as the spectral gap for the Laplace operator on $M$. There is no Poincaré inequality on $\mathbb{R}^n$ equipped with the Lebesgue measure (the usual "flat" Laplace operator does not have a spectral gap), but there is a Poincaré inequality on any compact Riemannian manifold equipped with its volume measure.

Poincaré inequalities are implied by logarithmic Sobolev inequalities, but the converse is false.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 21.19</span><span class="math-callout__name">(Exponential Measure)</span></p>

Let $\nu(dx) = e^{-\lvert x \rvert}\, dx$ be the exponential measure on $[0, +\infty)$. Then $\nu$ satisfies a Poincaré inequality (with constant 1). On the other hand, it does not satisfy any logarithmic Sobolev inequality. The same conclusions hold true for the double-sided exponential measure $\nu(dx) = e^{-\lvert x \rvert}\, dx / 2$ on $\mathbb{R}$. More generally, the measure $\nu\_\beta(dx) = e^{-\lvert x \rvert^\beta}\, dx / Z\_\beta$ satisfies a Poincaré inequality if and only if $\beta \ge 1$, and a logarithmic Sobolev inequality if and only if $\beta \ge 2$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21.20</span><span class="math-callout__name">(Lichnerowicz's Spectral Gap Inequality)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference Borel measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying a curvature-dimension condition $\mathrm{CD}(K, N)$ for some $K > 0$, $N \in (1, \infty]$. Then $\nu$ satisfies a Poincaré inequality with constant $KN/(N-1)$.

</div>

In other words, if $\mathrm{CD}(K, N)$ holds true, then for any Lipschitz function $f$ on $M$ with $\int f\, d\nu = 0$, one has

$$\left[\int f\, d\nu = 0\right] \implies \int f^2\, d\nu \le \frac{N-1}{KN} \int \lvert \nabla f \rvert^2\, d\nu. \tag{21.20}$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 21.21</span></p>

Let $L = \Delta - \nabla V \cdot \nabla$, then (21.20) means that $L$ admits a spectral gap of size at least $KN/(N-1)$:

$$\lambda_1(-L) \ge \frac{KN}{N-1}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 21.20)</span></p>

**Case $N < \infty$.** Apply (21.12) with $\mu = (1 + \varepsilon f)\, \nu$, where $\varepsilon$ is a small positive number, $f$ is Lipschitz and $\int f\, d\nu = 0$. Since $M$ has a finite diameter, $f$ is bounded, so $\mu$ is a probability measure for $\varepsilon$ small enough. Then, by standard Taylor expansion of the logarithm function,

$$H_{N,\nu}(\mu) = \varepsilon \int f\, d\nu + \varepsilon^2 \left(\frac{N-1}{N}\right) \int \frac{f^2}{2}\, d\nu + o(\varepsilon^2),$$

and the first term vanishes by assumption. Similarly,

$$\int \frac{\lvert \nabla \rho \rvert^2}{\rho} \left(\frac{\rho^{-2/N}}{\frac{1}{3} + \frac{2}{3}\, \rho^{-1/N}}\right) = \varepsilon^2 \int \lvert \nabla f \rvert^2\, d\nu + o(\varepsilon^2).$$

So (21.12) implies

$$\frac{N-1}{N} \cdot \frac{1}{2} \int f^2\, d\nu \le \frac{1}{2K} \left(\frac{N-1}{N}\right)^2 \int \lvert \nabla f \rvert^2\, d\nu,$$

and then inequality (21.20) follows.

**Case $N = \infty$.** Start from inequality (21.4) and apply a similar reasoning. (It is in fact a well-known property that a logarithmic Sobolev inequality with constant $K$ implies a Poincaré inequality with constant $K$.) $\square$

</div>

### Bibliographical Notes on Chapter 21

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

Popular sources dealing with classical isoperimetric inequalities are the book by **Burago and Zalgaller** and the survey by **Osserman**. The subject is related to Poincaré inequalities through **Cheeger's isoperimetric inequality** (Theorem 19.32). **Talagrand** put forward the use of isoperimetric inequalities in product spaces as part of his work on concentration of measure.

There are entire books devoted to **logarithmic Sobolev inequalities**; this subject goes back at least to **Nelson** and **Gross**, in relation to hypercontractivity and quantum field theory, with roots in earlier works by **Stam**, **Federbush** and **Bonami**. The 1992 survey by Gross, the Saint-Flour course by **Bakry** and the book by **Royer** are classical references. Applications to concentration of measure and deviation inequalities can be found in **Ledoux's** synthesis works.

The first and most famous logarithmic Sobolev inequality is the one that holds true for the Gaussian reference measure in $\mathbb{R}^n$ (equation (21.5)). **Stam** established (in the late fifties) an inequality equivalent to the logarithmic Sobolev inequality, found fifteen years later by **Gross**. At present, there are more than fifteen known proofs of (21.5).

The **Bakry–Émery theorem** (Theorem 21.2) was proven by a semigroup method which was later reinterpreted as a gradient flow argument (Chapter 25). The proof given here is essentially the one from the joint work of **Otto and Villani**. The normalized volume measure on a compact Riemannian manifold always satisfies a logarithmic Sobolev inequality, as **Rothaus** showed long ago. **Saloff-Coste** proved a partial converse: If $M$ has finite volume and Ricci curvature bounded below by $K$, and the normalized volume measure satisfies a logarithmic Sobolev inequality with constant $\lambda > 0$, then $M$ is compact and there is an explicit upper bound on its diameter:

$$\mathrm{diam}(M) \le C \sqrt{\dim(M)} \max\!\left(\frac{1}{\sqrt{\lambda}}, \frac{K_-}{\sqrt{\lambda}}\right). \tag{21.21}$$

The **Holley–Stroock perturbation theorem** for logarithmic Sobolev inequalities was proven in [478]. The criterion $\int e^{\alpha \lvert \nabla v \rvert^2}\, d\nu < \infty$ for $\alpha$ large enough is due to **Aida**. **F.-Y. Wang** showed that logarithmic Sobolev inequalities follow from curvature-dimension together with square-exponential moments, and **Barthe and Kolesnikov** derived more general results in the same spirit.

Logarithmic Sobolev inequalities in $\mathbb{R}^n$ for the measure $e^{-V(x)}\, dx$ require a sort of quadratic growth of the potential $V$, while Poincaré inequalities require a sort of linear growth. It is natural to ask what happens in between, when $V(x)$ behaves like $\lvert x \rvert^\beta$ with $1 < \beta < 2$. This subject has been studied by **Latała and Oleszkiewicz**, **Barthe, Cattiaux and Roberto**, and **Gentil, Guillin and Miclo**. Modified logarithmic Sobolev inequalities will be studied later, in Chapter 22.

The use of transport methods to study isoperimetric inequalities in $\mathbb{R}^n$ goes back at least to **Knothe**. **Gromov** revived the interest in Knothe's approach by using it to prove the isoperimetric inequality in $\mathbb{R}^n$. Recently, the method was put to a higher degree of sophistication by **Cordero-Erausquin, Nazaret and Villani**, recovering general optimal Sobolev inequalities in $\mathbb{R}^n$ together with some families of optimal **Gagliardo–Nirenberg** inequalities.

The proof of Theorem 21.9 is taken from a collaboration with **Lott**. **Demange** obtained a derivation of (21.9) which will be explained later in Chapter 25. The **Lévy–Gromov inequality** was first conjectured by Lévy and proven by Gromov. Functional versions of the Lévy–Gromov inequality are available, while **Bobkov, Bakry and Ledoux** have done striking work on the infinite-dimensional version (**Gaussian isoperimetry**).

The **Lichnerowicz spectral gap theorem** is usually encountered as a simple application of the Bochner formula; the above proof of Theorem 21.20 is a variant of the one in the joint work with **Lott**. It has the advantage, for the purpose of these notes, of being based on optimal transport.

</div>

## Chapter 22: Concentration Inequalities

The theory of concentration of measure is a collection of results, tools and recipes built on the idea that if a set $A$ is given in a metric probability space $(\mathcal{X}, d, \mathbb{P})$, then the enlargement $A^r := \lbrace x;\ d(x, A) \le r \rbrace$ might acquire a very high probability as $r$ increases. There is an equivalent statement that Lipschitz functions $\mathcal{X} \to \mathbb{R}$ are "almost constant" in the sense that they have a very small probability of deviating from some typical quantity, for instance their mean value. This theory was founded by Lévy and later developed by many authors, in particular V. Milman, Gromov and Talagrand.

To understand the relation between the two sides of concentration (sets and functions), it is most natural to think in terms of **median**, rather than mean value. By definition, a real number $m\_f$ is a median of the random variable $f : \mathcal{X} \to \mathbb{R}$ if

$$\mathbb{P}[f \ge m_f] \ge \frac{1}{2}, \qquad \mathbb{P}[f \le m_f] \ge \frac{1}{2}.$$

Then the two statements

- (a) $\forall A \subset \mathcal{X},\ \forall r \ge 0$, $\quad \mathbb{P}[A] \ge 1/2 \implies \mathbb{P}[A^r] \ge 1 - \psi(r)$
- (b) $\forall f \in \mathrm{Lip}(\mathcal{X}),\ \forall r \ge 0$, $\quad \mathbb{P}[f > m\_f + r] \le \psi(r / \lVert f \rVert\_{\mathrm{Lip}})$

are equivalent. Indeed, to pass from (a) to (b), first reduce to the case $\lVert f \rVert\_{\mathrm{Lip}} = 1$ and let $A = \lbrace f \le m\_f \rbrace$; conversely, to pass from (b) to (a), let $f = d(\,\cdot\,, A)$ and note that $0$ is a median of $f$.

The typical and most emblematic example of concentration of measure occurs in the Gaussian probability space $(\mathbb{R}^n, \gamma)$:

$$\gamma[A] \ge \frac{1}{2} \implies \forall r \ge 0, \quad \gamma[A^r] \ge 1 - e^{-\frac{r^2}{2}}. \tag{22.1}$$

Here is the translation in terms of Lipschitz functions: If $X$ is a Gaussian random variable with law $\gamma$, then for all Lipschitz functions $f : \mathbb{R}^n \to \mathbb{R}$,

$$\forall r \ge 0, \quad \mathbb{P}\left[f(X) \ge \mathbb{E}\, f(X) + r\right] \le \exp\!\left(-\frac{r^2}{2\, \lVert f \rVert_{\mathrm{Lip}}^2}\right). \tag{22.2}$$

Another famous example is the unit sphere $S^N$: if $\sigma^N$ stands for the normalized volume on $S^N$, then the formulas above can be replaced by

$$\sigma^N[A] \ge \frac{1}{2} \implies \sigma^N[A^r] \ge 1 - e^{-\frac{(N-1)}{2}\, r^2},$$

$$\mathbb{P}\left[f(X) \ge \mathbb{E}\, f(X) + r\right] \le \exp\!\left(-\frac{(N-1)\, r^2}{2\, \lVert f \rVert_{\mathrm{Lip}}^2}\right).$$

In this example the phenomenon of concentration of measure becomes more and more important as the dimension increases to infinity.

This chapter reviews the links between optimal transport and concentration, focusing on certain **transport inequalities**. The main results are Theorems 22.10 (characterization of Gaussian concentration), 22.14 (concentration via Ricci curvature bounds), 22.17 (concentration via logarithmic Sobolev inequalities), 22.22 (concentration via Talagrand inequalities) and 22.25 (concentration via Poincaré inequalities).

### Optimal Transport and Concentration

As first understood by Marton, there is a simple and robust functional approach to concentration inequalities based on optimal transport. One can encode information about the concentration of measure with respect to some reference measure $\nu$, by functional inequalities of the form

$$\forall \mu \in P(\mathcal{X}), \quad C(\mu, \nu) \le \mathcal{E}_\nu(\mu), \tag{22.3}$$

where $C(\mu, \nu)$ is the optimal transport cost between $\mu$ and $\nu$, and $\mathcal{E}\_\nu$ is some local nonlinear functional ("energy") of $\mu$, involving for instance the integral of a function of the density of $\mu$ with respect to $\nu$.

This principle may be heuristically understood as follows. To any measurable set $A$, associate the conditional measure $\mu\_A = (1\_A / \nu[A])\, \nu$. If the measure of $A$ is not too small, then the associated energy $\mathcal{E}\_\nu(\mu\_A)$ will not be too high, and by (22.3) the optimal transport cost $C(\mu\_A, \nu)$ will not be too high either. In that sense, the whole space $\mathcal{X}$ can be considered as a "small enlargement" of just $A$.

Here is a fluid mechanics analogy: imagine $\mu$ as the density of a fluid. The term on the right-hand side of (22.3) measures how difficult it is to prepare $\mu$, for instance to confine it within a set $A$ (this has to do with the measure of $A$); while the term on the left-hand side says how difficult it is for the fluid to invade the whole space, after it has been prepared initially with density $\mu$.

The most important class of functional inequalities of the type (22.3) occurs when the cost function is of the type $c(x, y) = d(x, y)^p$, and the "energy" functional is the square root of Boltzmann's $H$ functional,

$$H_\nu(\mu) = \int \rho \log \rho\, d\nu, \quad \mu = \rho\, \nu,$$

with the understanding that $H\_\nu(\mu) = +\infty$ if $\mu$ is not absolutely continuous with respect to $\nu$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 22.1</span><span class="math-callout__name">($T\_p$ Inequality)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space and let $p \in [1, \infty)$. Let $\nu$ be a reference probability measure in $P\_p(\mathcal{X})$, and let $\lambda > 0$. It is said that $\nu$ satisfies a $T\_p$ **inequality with constant** $\lambda$ if

$$\forall \mu \in P_p(\mathcal{X}), \qquad W_p(\mu, \nu) \le \sqrt{\frac{2\, H_\nu(\mu)}{\lambda}}. \tag{22.4}$$

These inequalities are often called transportation-cost inequalities, or **Talagrand inequalities**, although the latter denomination is sometimes restricted to the case $p = 2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.2</span></p>

Since $W\_p \le W\_q$ for $p \le q$, the $T\_p$ inequalities become stronger and stronger as $p$ increases. The inequalities $T\_1$ and $T\_2$ have deserved most attention. It is an experimental fact that $T\_1$ is more handy and flexible, while $T\_2$ has more geometric content, and behaves better in large dimensions (see for instance Corollary 22.6 below).

</div>

There are two important facts to know about $T\_p$ inequalities when $p$ varies in the range $[1, 2]$: they admit a *dual formulation*, and they *tensorize*. These properties are described in the two propositions below.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 22.3</span><span class="math-callout__name">(Dual Formulation of $T\_p$)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, $p \in [1, 2]$ and $\nu \in P\_p(\mathcal{X})$. Then the following two statements are equivalent:

- (a) $\nu$ satisfies $T\_p(\lambda)$;
- (b) For any $\varphi \in C\_b(\mathcal{X})$,

$$\begin{cases} \forall t \ge 0 \quad \displaystyle\int e^{\lambda t\, \inf_{y \in \mathcal{X}}\!\left[\varphi(y) + \frac{d(x,y)^p}{p}\right]} \nu(dx) \le e^{\lambda\!\left(\frac{1}{p} - \frac{1}{2}\right) t^{\frac{2}{2-p}}}\, e^{\lambda t \int \varphi\, d\nu} & (p < 2) \\[10pt] \displaystyle\int e^{\lambda\, \inf_{y \in \mathcal{X}}\!\left[\varphi(y) + \frac{d(x,y)^2}{2}\right]} \nu(dx) \le e^{\lambda \int \varphi\, d\nu} & (p = 2). \end{cases} \tag{22.5}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Particular Case 22.4</span><span class="math-callout__name">(Dual Formulation of $T\_1$)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space and $\nu \in P\_1(\mathcal{X})$, then the following two statements are equivalent:

- (a) $\nu$ satisfies $T\_1(\lambda)$;
- (b) For any $\varphi \in C\_b(\mathcal{X})$,

$$\forall t \ge 0 \qquad \int e^{t\, \inf_{y \in \mathcal{X}}\!\left[\varphi(y) + d(x,y)\right]}\, \nu(dx) \le e^{\frac{t^2}{2\lambda}}\, e^{t \int \varphi\, d\nu}. \tag{22.6}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 22.5</span><span class="math-callout__name">(Tensorization of $T\_p$)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, $p \in [1, 2]$ and let $\nu \in P\_p(\mathcal{X})$ be a reference probability measure satisfying an inequality $T\_p(\lambda)$. Then for any $N \in \mathbb{N}$, the measure $\nu^{\otimes N}$ satisfies an inequality $T\_p(N^{1 - \frac{2}{p}}\, \lambda)$ on $(\mathcal{X}^N, d\_p, \nu^{\otimes N})$, where the product distance $d\_p$ is defined by

$$d_p\!\left((x_1, \ldots, x_N);\, (y_1, \ldots, y_N)\right) = \left(\sum_{i=1}^{N} d(x_i, y_i)^p\right)^{\!\frac{1}{p}}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 22.6</span><span class="math-callout__name">($T\_2$ Inequalities Tensorize Exactly)</span></p>

If $\nu$ satisfies $T\_2(\lambda)$, then also $\mu^{\otimes N}$ satisfies $T\_2(\lambda)$ on $(\mathcal{X}^N, d\_2, \nu^{\otimes N})$, for any $N \in \mathbb{N}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Proposition 22.3)</span></p>

The proof is obtained as a consequence of Theorem 5.26. Recall the Legendre representation of the $H$-functional: For any $\lambda > 0$,

$$\frac{H_\nu(\mu)}{\lambda} = \sup_{\varphi \in C_b(\mathcal{X})} \left[\int \varphi\, d\mu - \frac{1}{\lambda} \log\!\left(\int_\mathcal{X} e^{\lambda\varphi}\, d\nu\right)\right],$$

$$\frac{1}{\lambda} \log\!\left(\int_\mathcal{X} e^{\lambda\varphi}\, d\nu\right) = \sup_{\mu \in P(\mathcal{X})} \left[\int \varphi\, d\mu - \frac{H_\nu(\mu)}{\lambda}\right]. \tag{22.7}$$

For the case $p = 2$: apply Theorem 5.26 with $c(x,y) = d(x,y)^2/2$, $F(\mu) = (1/\lambda) H\_\nu(\mu)$, $\Lambda(\varphi) = (1/\lambda) \log(\int e^{\lambda\varphi}\, d\nu)$. The conclusion is that $\nu$ satisfies $T\_2(\lambda)$ if and only if

$$\forall \phi \in C_b(\mathcal{X}), \quad \log \int \exp\!\left(\lambda \int \phi\, d\nu - \lambda\, \phi^c\right) d\nu \le 0,$$

i.e. $\int e^{-\lambda\, \phi^c}\, d\nu \le e^{-\lambda \int \phi\, d\nu}$, where $\phi^c(x) := \sup\_y\!\left(\phi(y) - d(x,y)^2/2\right)$. Upon changing $\phi$ for $\varphi = -\phi$, this is the desired result.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Proposition 22.5)</span></p>

The proof is reminiscent of the strategy used to construct the Knothe–Rosenblatt coupling. First choose an optimal coupling (for the cost function $c = d^p$) between $\mu\_1(dx\_1)$ to $\nu(dy\_1)$, call it $\pi\_1(dx\_1\, dy\_1)$. Then for each $x\_1$, choose an optimal coupling between $\mu\_2(dx\_2 \lvert x\_1)$ and $\nu(dy\_2)$, call it $\pi\_2(dx\_2\, dy\_2 \lvert x\_1)$, and so on. Glue these plans together to get a coupling

$$\pi(dx\, dy) = \pi_1(dx_1\, dy_1)\, \pi_2(dx_2\, dy_2 \lvert x^1) \cdots \pi_N(dx_N\, dy_N \lvert x^{N-1}).$$

By the definition of $d\_p$, and since each $\pi(\,\cdot\, \lvert x^{i-1})$ is an optimal transference plan between its marginals, one obtains the key estimate

$$W_p(\mu, \nu^{\otimes N})^p \le \sum_{i=1}^{N} \int W_p\!\left(\mu_i(\,\cdot\, \lvert x^{i-1}),\, \nu\right)^p\, \mu^{i-1}(dx^{i-1}). \tag{22.9}$$

By assumption, $\nu$ satisfies $T\_p(\lambda)$, so the right-hand side is bounded by $\sum\_i \int (2/\lambda\, H\_\nu(\mu\_i(\,\cdot\, \lvert x^{i-1})))^{p/2}\, \mu^{i-1}(dx^{i-1})$. Since $p \le 2$, one can apply Hölder's inequality and the **additivity of entropy** (Lemma 22.8 below) to conclude.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.7</span></p>

The same proof shows that the inequality $\forall \mu \in P(\mathcal{X}),\ C(\mu, \nu) \le H\_\nu(\mu)$ implies $\forall \mu \in P(\mathcal{X}^N),\ C^N(\mu, \nu) \le H\_{\nu^{\otimes N}}(\mu)$, where $C^N$ is the optimal transport cost associated with the cost function $c^N(x, y) = \sum c(x\_i, y\_i)$ on $\mathcal{X}^N$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 22.8</span><span class="math-callout__name">(Additivity of Entropy)</span></p>

Let $N \in \mathbb{N}$, let $\mathcal{X}\_1, \ldots, \mathcal{X}\_N$ be Polish spaces, $\nu\_i \in P(\mathcal{X}\_i)$ $(1 \le i \le N)$, $\mathcal{X} = \prod \mathcal{X}\_i$, $\nu = \bigotimes \nu\_i$, and $\mu \in P(\mathcal{X})$. Then, with the same notation as in the beginning of the proof of Proposition 22.5,

$$H_\nu(\mu) = \sum_{1 \le i \le N} \int H_{\nu_i}\!\left(\mu_i(dx_i \lvert x^{i-1})\right)\, \mu^{i-1}(dx^{i-1}). \tag{22.13}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Lemma 22.8)</span></p>

By induction, it suffices to treat the case $N = 2$. Let $\rho = \rho(x\_1, x\_2)$ be the density of $\mu$ with respect to $\nu\_1 \otimes \nu\_2$. The measure $\mu\_1(dx\_1)$ has density $\int \rho(x\_1, x\_2)\, \nu\_2(dx\_2)$, while the conditional measure $\mu\_2(dx\_2 \lvert x\_1)$ has density $\rho(x\_1, x\_2) / (\int \rho(x\_1, x\_2')\, \nu\_2(dx\_2'))$. From this and the additive property of the logarithm, one deduces

$$\int H_{\nu_2}\!\left(\mu_2(\,\cdot\, \lvert x_1)\right)\, \mu_1(dx_1) = H_\nu(\mu) - H_{\nu_1}(\mu_1).$$

</div>

### Gaussian Concentration and $T\_1$ Inequality

Gaussian concentration is a loose terminology meaning that some reference measure enjoys properties of concentration which are similar to those of the Gaussian measure. In this section we see that a certain form of Gaussian concentration is *equivalent* to a $T\_1$ inequality. Once again, this principle holds in very general metric spaces.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.10</span><span class="math-callout__name">(Gaussian Concentration)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, equipped with a reference probability measure $\nu$. Then the following properties are all equivalent:

- (i) $\nu$ lies in $P\_1(\mathcal{X})$ and satisfies a $T\_1$ inequality;
- (ii) There is $\lambda > 0$ such that for any $\varphi \in C\_b(\mathcal{X})$,

$$\forall t \ge 0 \qquad \int e^{t\, \inf_{y \in \mathcal{X}}\!\left[\varphi(y) + d(x,y)\right]}\, \nu(dx) \le e^{\frac{t^2}{2\lambda}}\, e^{t \int \varphi\, d\nu}.$$

- (iii) There is a constant $C > 0$ such that for any Borel set $A \subset \mathcal{X}$,

$$\nu[A] \ge \frac{1}{2} \implies \forall r > 0, \quad \nu[A^r] \ge 1 - e^{-C\, r^2}.$$

- (iv) There is a constant $C > 0$ such that

$$\forall f \in L^1(\nu) \cap \mathrm{Lip}(\mathcal{X}),\ \forall \varepsilon > 0, \quad \nu\!\left[\left\lbrace x \in \mathcal{X};\ f(x) \ge \int f\, d\nu + \varepsilon \right\rbrace\right] \le \exp\!\left(-C \frac{\varepsilon^2}{\lVert f \rVert_{\mathrm{Lip}}^2}\right);$$

- (v) There is a constant $C > 0$ such that

$$\forall f \in L^1(\nu) \cap \mathrm{Lip}(\mathcal{X}),\ \forall \varepsilon > 0,\ \forall N \in \mathbb{N},$$

$$\nu^{\otimes N}\!\left[\left\lbrace x \in \mathcal{X}^N;\ \frac{1}{N}\sum_{i=1}^{N} f(x_i) \ge \int f\, d\nu + \varepsilon \right\rbrace\right] \le \exp\!\left(-C \frac{N\, \varepsilon^2}{\lVert f \rVert_{\mathrm{Lip}}^2}\right);$$

- (vi) There is a constant $C > 0$ such that

$$\forall f \in \mathrm{Lip}(\mathcal{X}),\ \forall\, m_f = \text{median of } f,\ \forall \varepsilon > 0,$$

$$\nu\!\left[\left\lbrace x \in \mathcal{X};\ f(x) \ge m_f + \varepsilon \right\rbrace\right] \le \exp\!\left(-C \frac{\varepsilon^2}{\lVert f \rVert_{\mathrm{Lip}}^2}\right);$$

- (vii) For any $x\_0 \in \mathcal{X}$ there is a constant $a > 0$ such that

$$\int e^{a\, d(x_0, x)^2}\, \nu(dx) < +\infty;$$

- (viii) There exists $a > 0$ such that

$$\int e^{a\, d(x, y)^2}\, \nu(dx)\, \nu(dy) < +\infty;$$

- (ix) There exist $x\_0 \in \mathcal{X}$ and $a > 0$ such that

$$\int e^{a\, d(x_0, x)^2}\, \nu(dx) < +\infty.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.11</span></p>

One should not overestimate the power of Theorem 22.10. The simple (too simple?) criterion (ix) behaves badly in large dimensions, and in practice might lead to terrible constants at the level of (iii). In particular, this theorem alone is unable to recover dimension-free concentration inequalities such as (22.1) or (22.2). Statement (v) is dimension-independent, but limited to particular observables of the form $(1/N) \sum f(x\_i)$. Here we see some limitations of the $T\_1$ inequality.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.10)</span></p>

The proof establishes (i) $\Rightarrow$ (ii) $\Rightarrow$ (iv) $\Rightarrow$ (vii), (i) $\Rightarrow$ (v) $\Rightarrow$ (iv), (i) $\Rightarrow$ (iii) $\Rightarrow$ (vi) $\Rightarrow$ (vii) $\Rightarrow$ (viii) $\Rightarrow$ (ix) $\Rightarrow$ (i).

**(i) $\Rightarrow$ (ii):** This was already seen in Particular Case 22.4.

**(ii) $\Rightarrow$ (iv):** It suffices to treat the case $\lVert f \rVert\_{\mathrm{Lip}} = 1$ (replace $\varepsilon$ by $\varepsilon / \lVert f \rVert\_{\mathrm{Lip}}$ and $f$ by $f / \lVert f \rVert\_{\mathrm{Lip}}$). Then if $f$ is $1$-Lipschitz, $\inf\_{y \in \mathcal{X}} [f(y) + d(x,y)] = f(x)$, so (ii) implies

$$\int e^{t\, f(x)}\, \nu(dx) \le e^{\frac{t^2}{2\lambda}}\, e^{t \int f\, d\nu}.$$

With the shorthand $\langle f \rangle = \int f\, d\nu$, this is the same as $\int e^{t\,(f - \langle f \rangle)}\, d\nu \le e^{t^2/(2\lambda)}$. Then by the exponential Chebyshev inequality,

$$\nu\!\left[\lbrace f - \langle f \rangle \ge \varepsilon \rbrace\right] \le e^{-t\varepsilon}\, \int e^{t\,(f - \langle f \rangle)}\, d\nu \le e^{-t\varepsilon}\, e^{t^2/(2\lambda)};$$

and (iv) is obtained by optimizing in $t$. ($C = \lambda/2$ does the job.)

**(i) $\Rightarrow$ (v):** If $\nu$ satisfies $T\_1(\lambda)$, by Proposition 22.5 $\nu^{\otimes N}$ satisfies $T\_1(\lambda/N)$ on $\mathcal{X}^N$ equipped with the distance $d\_1(x,y) = \sum d(x\_i, y\_i)$. Let $F : \mathcal{X}^N \to \mathbb{R}$ be defined by $F(x) = \frac{1}{N} \sum\_{i=1}^{N} f(x\_i)$. If $f$ is Lipschitz then $\lVert F \rVert\_{\mathrm{Lip}} = \lVert f \rVert\_{\mathrm{Lip}} / N$. Applying (iv) with $\mathcal{X}$ replaced by $\mathcal{X}^N$ and $f$ replaced by $F$, we obtain the result.

**(i) $\Rightarrow$ (iii):** Assume that $\forall \mu \in P\_1(\mathcal{X}),\ W\_1(\mu, \nu) \le C \sqrt{H\_\nu(\mu)}$. Choose $A$ with $\nu[A] \ge 1/2$, and $\mu = (1\_A\, \nu) / \nu[A]$. If $\nu[A^r] = 1$ there is nothing to prove, otherwise let $\widetilde{\mu} = (1\_{\mathcal{X} \setminus A^r}\, \nu) / \nu[\mathcal{X} \setminus A^r]$. By immediate computation,

$$H_\nu(\mu) = \log \frac{1}{\nu[A]} \le \log 2, \qquad H_\nu(\widetilde{\mu}) = \log\!\left(\frac{1}{1 - \nu[A^r]}\right).$$

By the triangle inequality for $W\_1$,

$$W_1(\mu, \widetilde{\mu}) \le W_1(\mu, \nu) + W_1(\widetilde{\mu}, \nu) \le C\sqrt{\log 2} + C\sqrt{\log\!\left(\frac{1}{1 - \nu[A^r]}\right)}. \tag{22.15}$$

On the other hand, it is obvious that $W\_1(\mu, \widetilde{\mu}) \ge r$ (all the mass has to go from $A$ to $\mathcal{X} \setminus A^r$, so each unit of mass should travel a distance at least $r$). So (22.15) implies

$$r \le C\sqrt{\log 2} + C\sqrt{\log\!\left(\frac{1}{1 - \nu[A^r]}\right)},$$

therefore

$$\nu[A^r] \ge 1 - \exp\!\left[-\left(\frac{r}{C} - \sqrt{\log 2}\right)^2\right].$$

This establishes a bound of the type $\nu[A^r] \ge 1 - a\, e^{-C\, r^2}$, and (iii) follows.

**(ix) $\Rightarrow$ (i):** If $\nu$ satisfies (ix), then obviously $\nu \in P\_1(\mathcal{X})$. To prove that $\nu$ satisfies $T\_1$, one establishes the **weighted Csiszár–Kullback–Pinsker inequality**

$$\left\lVert d(x_0,\, \cdot\,)\, (\mu - \nu)\right\rVert_{TV} \le \frac{\sqrt{2}}{a}\left(1 + \log \int_\mathcal{X} e^{a\, d(x_0, x)^2}\, d\nu(x)\right)^{1/2}\, \sqrt{H_\nu(\mu)}. \tag{22.16}$$

Inequality (22.16) implies a $T\_1$ inequality, since Theorem 6.15 yields $W\_1(\mu, \nu) \le \left\lVert d(x\_0,\, \cdot\,)\, (\mu - \nu)\right\rVert\_{TV}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.12</span><span class="math-callout__name">(CKP Inequality)</span></p>

In the particular case $\varphi = 1$, one can replace the inequality (22.21) by just $\int d\mu = 1$; then instead of (22.23) we obtain

$$\lVert \mu - \nu \rVert_{TV} \le \sqrt{2\, H_\nu(\mu)}. \tag{22.25}$$

This is the classical **Csiszár–Kullback–Pinsker** (CKP) inequality, with the sharp constant $\sqrt{2}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.13</span></p>

If $\nu$ satisfies $T\_2(\lambda)$, then also $\nu^{\otimes N}$ satisfies $T\_2(\lambda)$, independently of $N$; so one might hope to improve the concentration inequality appearing in Theorem 22.10(v). But now the space $\mathcal{X}^N$ should be equipped with the $d\_2$ distance, for which the function $F : x \to (1/N) \sum f(x\_i)$ is only $\sqrt{N}$-Lipschitz! In the end, $T\_2$ does not lead to any improvement of Theorem 22.10(v). This is not in contradiction with the fact that $T\_2$ is significantly stronger than $T\_1$ (as we shall see in the sequel); it just shows that we cannot tell the difference when we consider observables of the particular form $(1/N) \sum \varphi(x\_i)$. If one is interested in more complicated observables (such as nonlinear functionals, or suprema as in Example 22.36 below) the difference between $T\_1$ and $T\_2$ might become considerable.

</div>

### Talagrand Inequalities from Ricci Curvature Bounds

In the previous section the focus was on $T\_1$ inequalities; in this and the next two sections we shall consider the stronger $T\_2$ inequalities (Talagrand inequalities). The most simple criterion for $T\_2$ to hold is expressed in terms of *Ricci curvature bounds*:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.14</span><span class="math-callout__name">($\mathrm{CD}(K, \infty)$ implies $T\_2(K)$)</span></p>

Let $M$ be a Riemannian manifold, equipped with a reference probability measure $\nu = e^{-V}\, \mathrm{vol} \in C^2(M)$, satisfying a $\mathrm{CD}(K, \infty)$ curvature-dimension bound for some $K > 0$. Then $\nu$ belongs to $P\_2(M)$ and satisfies the Talagrand inequality $T\_2(K)$. In particular, $\nu$ satisfies Gaussian concentration bounds.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 22.14)</span></p>

It follows by Theorem 18.12 that $\nu$ lies in $P\_2(M)$; then the inequality $T\_2(K)$ comes from Corollary 20.13(i) with $\mu\_0 = \nu$ and $\mu\_1 = \mu$. Since $T\_2(K)$ implies $T\_1(K)$, Theorem 22.10 shows that $\nu$ satisfies Gaussian concentration bounds.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 22.15</span></p>

The standard Gaussian $\gamma$ on $\mathbb{R}^N$ satisfies $\mathrm{CD}(1, \infty)$, and therefore $T\_2(1)$ too. This is independent of $N$.

</div>

The links between Talagrand inequalities and dimension free concentration bounds will be considered further in Theorem 22.22.

### Relation with log Sobolev and Poincaré Inequalities

So far we learnt that logarithmic Sobolev inequalities follow from curvature bounds, and that Talagrand inequalities also result from the same bounds. We also learnt from Chapter 21 that logarithmic Sobolev inequalities imply Poincaré inequalities. Actually, Talagrand inequalities are *intermediate* between these two inequalities: a logarithmic Sobolev inequality implies a Talagrand inequality, which in turn implies a Poincaré inequality. In some sense however, Talagrand is closer to logarithmic Sobolev than to Poincaré: For instance, in nonnegative curvature, the validity of the Talagrand inequality is equivalent to the validity of the logarithmic Sobolev inequality — up to a degradation of the constant by a factor $1/4$.

To establish these properties, we shall use, for the first time in this course, a **semigroup argument**. As discovered by Bobkov, Gentil and Ledoux, it is indeed convenient to consider inequality (22.5) from a dynamical point of view, with the help of the (forward) Hamilton–Jacobi semigroup defined as in Chapter 7 by

$$\begin{cases} H_0\, \varphi = \varphi, \\[4pt] (H_t\, \varphi)(x) = \inf_{y \in M}\!\left[\varphi(y) + \frac{d(x,y)^2}{2t}\right] \qquad (t > 0,\ x \in M). \end{cases} \tag{22.26}$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 22.16</span><span class="math-callout__name">(Properties of the Quadratic Hamilton–Jacobi Semigroup)</span></p>

Let $f$ be a bounded continuous function on a Riemannian manifold $M$. Then:

- (i) For any $s, t \ge 0$, $H\_t\, H\_s\, f = H\_{t+s}\, f$.
- (ii) For any $x \in M$, $\inf f \le (H\_t f)(x) \le f(x)$; moreover, the infimum over $M$ in (22.26) can be restricted to the ball $B[x, \sqrt{Ct}]$, where $C := 2(\sup f - \inf f)$.
- (iii) For any $t > 0$, $H\_t f$ is Lipschitz and locally semiconcave (with a quadratic modulus of semiconcavity) on $M$.
- (iv) For any $x \in M$, $H\_t f(x)$ is a nonincreasing function of $t$, converging monotonically to $f(x)$ as $t \to 0$. In particular, $\lim\_{t \to 0} H\_t f = f$, locally uniformly.
- (v) For any $t \ge 0$, $s > 0$, $x \in M$,

$$\frac{\lvert H_{t+s} f(x) - H_t f(x) \rvert}{s} \le \frac{\lVert H_t f \rVert_{\mathrm{Lip}(B[x, \sqrt{Cs}])}^2}{2}.$$

- (vi) For any $x \in M$ and $t \ge 0$,

$$\liminf_{s \to 0^+} \frac{(H_{t+s} f)(x) - (H_t f)(x)}{s} \ge -\frac{\lvert \nabla^- H_t f(x) \rvert^2}{2}. \tag{22.27}$$

- (vii) For any $x \in M$ and $t > 0$,

$$\lim_{s \to 0^+} \frac{(H_{t+s} f)(x) - (H_t f)(x)}{s} = -\frac{\lvert \nabla^- H_t f(x) \rvert^2}{2}. \tag{22.28}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.17</span><span class="math-callout__name">(Logarithmic Sobolev $\Rightarrow$ $T\_2$ $\Rightarrow$ Poincaré)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference probability measure $\nu \in P\_2(M)$. Then:

- (i) If $\nu$ satisfies a logarithmic Sobolev inequality with some constant $K > 0$, then it also satisfies a Talagrand inequality with constant $K$.
- (ii) If $\nu$ satisfies a Talagrand inequality with some constant $K > 0$, then it also satisfies a Poincaré inequality with constant $K$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.18</span></p>

Theorem 22.17 has the important advantage over Theorem 22.14 that logarithmic Sobolev inequalities are somewhat easy to perturb (recall Remark 21.5), while there are few known perturbation criteria for $T\_2$. One of them is as follows: if $\nu$ satisfies $T\_2$ and $\widetilde{\nu} = e^{-v}\, \nu$ with $v$ bounded, then there is a constant $C$ such that

$$\forall \mu \in P_2(M), \quad W_2(\mu, \nu) \le C\!\left(\sqrt{H_\nu(\mu)} + H_\nu(\mu)^{\frac{1}{4}}\right). \tag{22.29}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.19</span></p>

Part (ii) of Theorem 22.17 shows that the $T\_2$ inequality on a Riemannian manifold contains spectral information, and imposes qualitative restrictions on measures satisfying $T\_2$. For instance, the support of such a measure needs to be connected. (Otherwise take $u = a$ on one connected component, $u = b$ on another, $u = 0$ elsewhere, where $a$ and $b$ are two constants chosen in such a way that $\int u\, d\nu = 0$. Then $\int \lvert \nabla u \rvert^2\, d\nu = 0$, while $\int u^2\, d\nu > 0$.) This remark shows that $T\_2$ does not result from just decay estimates, in contrast with $T\_1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.17, Part (i))</span></p>

Let $\nu$ satisfy a logarithmic Sobolev inequality with constant $K > 0$. By the dual reformulation of $T\_2(K)$ (Proposition 22.3 for $p = 2$), it is sufficient to show

$$\forall g \in C_b(M), \qquad \int_M e^{K(Hg)}\, d\nu \le e^{\int_M g\, d\nu}, \tag{22.30}$$

where $(Hg)(x) = \inf\_{y \in M} \left[g(y) + d(x,y)^2/2\right]$. Define

$$\phi(t) = \frac{1}{Kt} \log\!\left(\int_M e^{Kt\, H_t g}\, d\nu\right). \tag{22.31}$$

The proof amounts to showing $\phi$ is nonincreasing. Since $g$ is bounded, Proposition 22.16(ii) implies that $H\_t g$ is bounded, uniformly in $t$. Then $e^{Kt\, H\_t g} = 1 + Kt \int\_M H\_t g\, d\nu + O(t^2)$, so $\phi(t) = \int\_M H\_t g\, d\nu + O(t)$. By Proposition 22.16(iv), $H\_t g$ converges pointwise to $g$ as $t \to 0^+$; then by the dominated convergence theorem, $\lim\_{t \to 0^+} \phi(t) = \int\_M g\, d\nu$.

So it all amounts to showing that $\phi(1) \le \lim\_{t \to 0^+} \phi(t)$, and this will obviously be true if $\phi(t)$ is nonincreasing in $t$. Computing the right-derivative $d^+ \phi / dt$ and using the Hamilton–Jacobi equation (Proposition 22.16(vii)) together with the logarithmic Sobolev inequality with constant $K$, one shows that the quantity inside square brackets is nonpositive. So $\phi$ is nonincreasing and the proof is complete.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.17, Part (ii))</span></p>

Let $h : M \to \mathbb{R}$ be a bounded Lipschitz function satisfying $\int\_M h\, d\nu = 0$. Introduce $\psi(t) = \int\_M e^{Kt\, H\_t h}\, d\nu$. From the dual formulation of Talagrand's inequality (Proposition 22.3 for $p = 2$), $\psi(t)$ is bounded above by $\exp(Kt \int\_M h\, d\nu) = 1$; hence $\psi$ has a maximum at $t = 0$. Combining this with $\int h\, d\nu = 0$, we find

$$0 \le \limsup_{t \to 0^+}\!\left(\frac{1 - \psi(t)}{Kt^2}\right) = \limsup_{t \to 0^+} \int_M \!\left(\frac{1 + Kth - e^{Kt\, H_t h}}{Kt^2}\right) d\nu.$$

By the boundedness of $H\_t h$ and Proposition 22.16(iv),

$$e^{Kt\, H_t h} = 1 + Kt\, H_t h + \frac{K^2 t^2}{2}\, (H_t h)^2 + O(t^3) = 1 + Kt\, H_t h + \frac{K^2 t^2}{2}\, h^2 + o(t^2).$$

So the right-hand side of the limsup equals $\limsup\_{t \to 0^+} \int\_M \!\left(\frac{h - H\_t h}{t}\right) d\nu - \frac{K}{2} \int\_M h^2\, d\nu$. By Proposition 22.16(v), $(h - H\_t h)/t$ is bounded; so we can apply Fatou's lemma. Then Proposition 22.16(vi) implies that $\int\_M \limsup\_{t \to 0^+}\!\left(\frac{h - H\_t h}{t}\right) d\nu \le \int\_M \frac{\lvert \nabla^- h \rvert^2}{2}\, d\nu$. All in all, the right-hand side can be bounded above by

$$\frac{1}{2} \int_M \lvert \nabla^- h \rvert^2\, d\nu - \frac{K}{2} \int_M h^2\, d\nu. \tag{22.44}$$

So (22.44) is always nonnegative, which concludes the proof of the Poincaré inequality.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Talagrand and Poincaré via Otto's Calculus)</span></p>

Use Otto's calculus to show that, at least formally,

$$\lVert h \rVert_{H^{-1}(\nu)} = \lim_{\varepsilon \to 0} \frac{W_2\!\left((1 + \varepsilon h)\, \nu,\, \nu\right)}{\varepsilon},$$

where $h$ is smooth and bounded (and compactly supported, if you wish), $\int h\, d\nu = 0$, and the dual Sobolev norm $H^{-1}(\nu)$ is defined by

$$\lVert h \rVert_{H^{-1}(\nu)} = \sup_{h \neq 0} \frac{\lVert h \rVert_{L^2(\nu)}}{\lVert \nabla h \rVert_{L^2(\nu)}} = \left\lVert \nabla(L^{-1} h)\right\rVert_{L^2(\nu)},$$

where as before $L = \Delta - \nabla V \cdot \nabla$. Deduce that, at least formally, the Talagrand inequality reduces, in the limit when $\mu = (1 + \varepsilon h)\, \nu$ and $\varepsilon \to 0$, to the **dual Poincaré inequality**

$$\left[\int h\, d\nu = 0\right] \implies \lVert h \rVert_{H^{-1}(\nu)} \le \frac{\lVert h \rVert_{L^2(\nu)}}{\sqrt{K}}.$$

</div>

To close this section, we show that the Talagrand inequality does imply a logarithmic Sobolev inequality under strong enough curvature assumptions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.21</span><span class="math-callout__name">($T\_2$ Sometimes Implies Log Sobolev)</span></p>

Let $M$ be a Riemannian manifold and let $\nu = e^{-V}\, \mathrm{vol} \in P\_2(M)$ be a reference measure on $M$, $V \in C^2(M)$. Assume that $\nu$ satisfies a Talagrand inequality $T\_2(\lambda)$, and a curvature-dimension inequality $\mathrm{CD}(K, \infty)$ for some $K > -\lambda$. Then $\nu$ also satisfies a logarithmic Sobolev inequality with constant

$$\widetilde{\lambda} = \max\!\left[\frac{\lambda}{4}\!\left(1 + \frac{K}{\lambda}\right)^{\!2},\, K\right].$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.21)</span></p>

From the assumptions and Corollary 20.13(ii), the nonnegative quantities $H = H\_\nu(\mu)$, $W = W\_2(\mu, \nu)$ and $I = I\_\nu(\mu)$ satisfy the inequalities

$$H \le W\sqrt{I} - \frac{\lambda\, W^2}{2}, \qquad W \le \sqrt{\frac{2H}{K}}.$$

It follows by an elementary calculation that $H \lesssim I/(2\widetilde{\lambda})$, so $\nu$ satisfies a logarithmic Sobolev inequality with constant $\widetilde{\lambda}$.

</div>

### Talagrand Inequalities and Gaussian Concentration

We already saw in Theorem 22.10 that the $T\_1$ inequality implies Gaussian concentration bounds. Now we shall see that the stronger $T\_2$ inequality implies *dimension free* concentration bounds; roughly speaking, this means that $\nu^{\otimes N}$ satisfies concentration inequalities with constants independent of $N$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.22</span><span class="math-callout__name">($T\_2$ and Dimension Free Gaussian Concentration)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space, equipped with a reference probability measure $\nu$. Then the following properties are equivalent:

- (i) $\nu$ lies in $P\_2(\mathcal{X})$ and satisfies a $T\_2$ inequality.
- (ii) There is a constant $C > 0$ such that for any $N \in \mathbb{N}$ and any Borel set $A \subset \mathcal{X}^N$,

$$\nu[A] \ge \frac{1}{2} \implies \forall r > 0, \quad \nu^{\otimes N}[A^r] \ge 1 - e^{-C\, r^2};$$

here the enlargement $A^r$ is defined with the $d\_2$ distance,

$$d_2\!\left((x_1, \ldots, x_N),\, (y_1, \ldots, y_N)\right) = \left(\sum_{i=1}^{N} d(x_i, y_i)^2\right)^{\!\frac{1}{2}}.$$

- (iii) There is a constant $C > 0$ such that for any $N \in \mathbb{N}$ and any $f \in \mathrm{Lip}(\mathcal{X}^N, d\_2) \cap L^1(\nu^{\otimes N})$,

$$\nu^{\otimes N}\!\left[\left\lbrace x \in \mathcal{X}^N;\ f(x) \ge m + r \right\rbrace\right] \le e^{-C\, \frac{r^2}{\lVert f \rVert_{\mathrm{Lip}}^2}},$$

where $m$ is a median (resp. the mean value) of $f$ with respect to the measure $\nu^{\otimes N}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.23</span></p>

The dependence of the constants can be made more precise: If $\nu$ satisfies $T\_2(K)$, then there are $a, r\_0 > 0$ such that (with obvious notation) $\nu^{\otimes N}[A^r] \ge 1 - a\, e^{-K(r - r\_0)^2/2}$ and $\nu^{\otimes N}[\lbrace f \ge m + r \rbrace] \le a\, e^{-K(r - r\_0)^2/2}$, for all $N \in \mathbb{N}$ and $r \ge r\_0$. Conversely, these inequalities imply $T\_2(K)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.22)</span></p>

**(i) $\Rightarrow$ (ii):** If (i) is satisfied, then by Proposition 22.5, $\nu^{\otimes N}$ satisfies $T\_2$ (and therefore $T\_1$) with a uniform constant. Then we can repeat the proof of (i) $\Rightarrow$ (iii) in Theorem 22.10, with $\nu$ replaced by $\nu^{\otimes N}$. Since the constants obtained in the end are independent of $N$, this proves (ii).

**(ii) $\Rightarrow$ (iii):** Follows the same lines as in Theorem 22.10.

**(iii) $\Rightarrow$ (i):** This is more subtle. For any $x \in \mathcal{X}^N$, define the empirical measure $\widehat{\mu}\_x^N = \frac{1}{N} \sum\_{i=1}^{N} \delta\_{x\_i} \in P(\mathcal{X})$, and let $f\_N(x) = W\_2(\widehat{\mu}\_x^N, \nu)$. By the triangle inequality for $W\_2$ and Theorem 4.8, for any $x, y \in \mathcal{X}^N$,

$$\lvert f_N(x) - f_N(y) \rvert^2 \le W_2\!\left(\frac{1}{N}\sum \delta_{x_i},\, \frac{1}{N}\sum \delta_{y_i}\right)^{\!2} \le \frac{1}{N}\sum_{i=1}^{N} d(x_i, y_i)^2;$$

so $f\_N$ is $(1/\sqrt{N})$-Lipschitz in distance $d\_2$. By (iii), $\nu^{\otimes N}[f\_N \ge m\_N + r] \le e^{-C\, N\, r^2}$, where $m\_N$ is either a median or the mean value of $f\_N$.

By Varadarajan's theorem, $\widehat{\mu}\_x^N$ converges weakly to $\nu$ for all $x$ outside of a $\nu^{\otimes N}$-negligible set. By the strong law of large numbers, $\int \varphi\, d\widehat{\mu}\_x^N = (1/N) \sum d(x\_0, x\_i)^2$ converges to $\int d(x\_0, x)^2\, d\nu(x) = \int \varphi\, d\nu$. Combining with Theorem 6.9 we see that $W\_2(\widehat{\mu}\_x^N, \nu) \to 0$ as $N \to \infty$, for $\nu^{\otimes N}$-almost all sequences. By Lebesgue's dominated convergence theorem, for any $t > 0$ we have $\nu^{\otimes N}[W\_2(\widehat{\mu}^N, \nu) \ge t] \to 0$; this implies that any sequence $(m\_N)$ of medians of $f\_N$ converges to zero.

Since $m\_N \to 0$, (22.45) implies

$$\liminf_{N \to \infty}\!\left(-\frac{1}{N} \log \nu^{\otimes N}\!\left[\left\lbrace x;\ W_2(\widehat{\mu}_x^N, \nu) \ge r \right\rbrace\right]\right) \ge C\, r^2. \tag{22.46}$$

The left-hand side is a large deviation estimate for the empirical measure of independent identically distributed samples; since the set $W\_2(\mu, \nu) > r$ is open in the topology of $P\_2(\mathcal{X})$, a suitable version of **Sanov's theorem** yields

$$\limsup_{N \to \infty}\!\left(-\frac{1}{N} \log \nu^{\otimes N}\!\left[\left\lbrace x;\ W_2(\widehat{\mu}_x^N, \nu) \ge r \right\rbrace\right]\right) \le \inf\!\left\lbrace H_\nu(\mu);\ \nu \in P_2(\mathcal{X});\ W_2(\mu, \nu) > r \right\rbrace. \tag{22.47}$$

Combining (22.46) and (22.47) gives $\inf\!\left\lbrace H\_\nu(\mu);\ W\_2(\mu, \nu) > r \right\rbrace \ge C\, r^2$, which is equivalent to $W\_2(\mu, \nu) \le \sqrt{H\_\nu(\mu)/C}$, hence (i).

</div>

### Poincaré Inequalities and Quadratic-Linear Transport Cost

So far we have encountered transport inequalities involving the quadratic cost function $c(x, y) = d(x, y)^2$, and the linear cost function $c(x, y) = d(x, y)$. Remarkably, Poincaré inequalities can be recast in terms of transport cost inequalities for a cost function which behaves quadratically for small distances, and linearly for large distances. As discovered by Bobkov and Ledoux, they can also be rewritten as **modified logarithmic Sobolev inequalities**, which are just usual logarithmic Sobolev inequalities, except that there is a Lipschitz constraint on the logarithm of the density of the measure. These two reformulations of Poincaré inequalities will be discussed below.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 22.24</span><span class="math-callout__name">(Quadratic-Linear Cost)</span></p>

Let $(\mathcal{X}, d)$ be a metric space. The **quadratic-linear cost** $c\_{q\ell}$ on $\mathcal{X}$ is defined by

$$c_{q\ell}(x, y) = \begin{cases} d(x, y)^2 & \text{if } d(x, y) \le 1; \\ d(x, y) & \text{if } d(x, y) > 1. \end{cases}$$

In a compact notation, $c\_{q\ell}(x, y) = \min(d(x, y)^2, d(x, y))$. The optimal total cost associated with $c\_{q\ell}$ will be denoted by $C\_{q\ell}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.25</span><span class="math-callout__name">(Reformulations of Poincaré Inequalities)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference probability measure $\nu = e^{-V}\, \mathrm{vol}$. Then the following statements are equivalent:

- (i) $\nu$ satisfies a Poincaré inequality;
- (ii) There are constants $c, K > 0$ such that for any Lipschitz probability density $\rho$,

$$\lvert \nabla \log \rho \rvert \le c \implies H_\nu(\mu) \le \frac{I_\nu(\mu)}{K}, \qquad \mu = \rho\, \nu; \tag{22.48}$$

- (iii) $\nu \in P\_1(M)$ and there is a constant $C > 0$ such that

$$\forall \mu \in P_1(M), \qquad C_{q\ell}(\mu, \nu) \le C\, H_\nu(\mu). \tag{22.49}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.26</span></p>

The equivalence between (i) and (ii) remains true when the Riemannian manifold $M$ is replaced by a general metric space. On the other hand, the equivalence with (iii) uses at least a little bit of the Riemannian structure (say, a local Poincaré inequality, a local doubling property and a length property).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.27</span></p>

The equivalence between (i), (ii) and (iii) can be made more precise. As the proof will show, if $\nu$ satisfies a Poincaré inequality with constant $\lambda$, then for any $c < 2\sqrt{\lambda}$ there is an explicit constant $K = K(c) > 0$ such that (22.48) holds true; and the $K(c)$ converges to $\lambda$ as $c \to 0$. Conversely, if for each $c > 0$ we call $K(c)$ the best constant in (22.48), then $\nu$ satisfies a Poincaré inequality with constant $\lambda = \lim\_{c \to 0} K(c)$. Also, in (ii) $\Rightarrow$ (iii) one can choose $C = \max(4/K,\, 2/c)$, while in (iii) $\Rightarrow$ (i) the Poincaré constant can be taken equal to $C^{-1}$.

</div>

Theorem 22.25 will be obtained by two ingredients: The first one is the Hamilton–Jacobi semigroup with a nonquadratic Lagrangian; the second one is a generalization of Theorem 22.17, stated below.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.28</span><span class="math-callout__name">(From Generalized Log Sobolev to Transport to Generalized Poincaré)</span></p>

Let $M$ be a Riemannian manifold equipped with its geodesic distance $d$ and with a reference probability measure $\nu = e^{-V}\, \mathrm{vol} \in P\_2(M)$. Let $L : \mathbb{R}\_+ \to \mathbb{R}\_+$ be a strictly increasing convex function such that $L(0) = 0$ and $L''$ is bounded above; let $c\_L(x, y) = L(d(x, y))$ and let $C\_L$ be the optimal transport cost associated with the cost function $c\_L$. Further, assume that $L(r) \le C(1 + r)^p$ for some $p \in [1, 2]$ and some $C > 0$. Then:

- (i) Further, assume that $L^\ast(ts) \le t^2 L^\ast(s)$ for all $t \in [0, 1]$, $s \ge 0$. If there is $\lambda \in (0, 1]$ such that $\nu$ satisfies the generalized logarithmic Sobolev inequality with constant $\lambda$:

For any $\mu = \rho\, \nu \in P(M)$ such that $\log \rho \in \mathrm{Lip}(M)$,

$$H_\nu(\mu) \le \frac{1}{\lambda} \int L^*\!\left(\lvert \nabla \log \rho \rvert\right) d\mu; \tag{22.50}$$

then $\nu$ also satisfies the following transport inequality:

$$\forall \mu \in P_p(M), \qquad C_L(\mu, \nu) \le \frac{H_\nu(\mu)}{\lambda}. \tag{22.51}$$

- (ii) If $\nu$ satisfies (22.51), then it also satisfies the generalized Poincaré inequality with constant $\lambda$:

$$\forall f \in \mathrm{Lip}(M),\ \lVert f \rVert_{\mathrm{Lip}} \le L'(\infty),$$

$$\int f\, d\nu = 0 \implies \int f^2\, d\nu \le \frac{2}{\lambda} \int L^*(\lvert \nabla f \rvert)\, d\nu.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.25)</span></p>

**(i) $\Rightarrow$ (ii):** Let $f = \log \rho - \int (\log \rho)\, d\nu$; so $\int f\, d\nu = 0$ and the assumption in (ii) reads $\lvert \nabla f \rvert \le c$. Moreover, with $a = \int (\log \rho)\, d\nu$ and $X = \int e^f\, d\nu$,

$$I_\nu(\mu) = e^a \int \lvert \nabla f \rvert^2 e^f\, d\nu;$$

$$H_\nu(\mu) = \int (f + a) e^{f+a}\, d\nu - \left(\int e^{f+a}\, d\nu\right) \log\!\left(\int e^{f+a}\, d\nu\right) = e^a\!\left(\int f e^f\, d\nu - \int e^f\, d\nu + 1\right).$$

So it is sufficient to prove $\lvert \nabla f \rvert \le c \implies \int (f e^f - e^f + 1)\, d\nu \le \frac{1}{K} \int \lvert \nabla f \rvert^2 e^f\, d\nu$. In the sequel, $c$ is any constant satisfying $0 < c < 2\sqrt{\lambda}$. Inequality (22.53) will be proven by two auxiliary inequalities:

$$\int f^2\, d\nu \le e^{c\sqrt{5/\lambda}} \int f^2 e^{-\lvert f \rvert}\, d\nu; \tag{22.54}$$

$$\int f^2 e^f\, d\nu \le \frac{1}{\lambda}\!\left(\frac{2\sqrt{\lambda} + c}{2\sqrt{\lambda} - c}\right)^{\!2} \int \lvert \nabla f \rvert^2 e^f\, d\nu. \tag{22.55}$$

For (22.54): use the elementary inequality $2\lvert f \rvert^3 \le \delta f^2 + \delta^{-1} f^4$ ($\delta > 0$), integrate, and apply the Poincaré inequality to bound $\int (f^2)^2\, d\nu - (\int f^2\, d\nu)^2 \le (1/\lambda) \int \lvert \nabla(f^2) \rvert^2\, d\nu = (4/\lambda) \int f^2 \lvert \nabla f \rvert^2\, d\nu \le (4c^2/\lambda) \int f^2\, d\nu$. The choice $\delta = \sqrt{5c^2/\lambda}$ yields $\int \lvert f \rvert^3\, d\nu \le c\sqrt{5/\lambda}\, \int f^2\, d\nu$. By Jensen's inequality with the convex function $x \to e^{-\lvert x \rvert}$ and the probability measure $\sigma = f^2\, \nu / (\int f^2\, d\nu)$, we get $\int f^2 e^{-\lvert f \rvert}\, d\nu \ge e^{-\int \lvert f \rvert^3\, d\nu / \int f^2\, d\nu}\, \int f^2\, d\nu$, which gives (22.54).

For (22.55): use the Poincaré inequality and the chain rule, combined with the Cauchy–Schwarz inequality, to bound $(\int f e^{f/2}\, d\nu)^2$ and $\int f^2 e^f\, d\nu - (\int f e^{f/2}\, d\nu)^2$ separately, then combine. The constraint $c^2/(4\lambda) < 1$ is crucial.

**(ii) $\Rightarrow$ (iii):** Let $\nu$ satisfy a modified logarithmic Sobolev inequality as in (22.48). Then let $L(s) = cs^2/2$ for $0 \le s \le 1$, $L(s) = c(s - 1/2)$ for $s > 1$. The function $L$ so defined is convex, strictly increasing and $L'' \le c$. Its Legendre transform $L^\ast$ is quadratic on $[0, c]$ and identically $+\infty$ on $(c, +\infty)$. So (22.48) can be rewritten

$$H_\nu(\mu) \le \frac{2c}{K} \int L^*(\lvert \nabla \log \rho \rvert)\, d\mu.$$

Since $L^\ast(tr) \le t^2 L^\ast(r)$ for all $t \in [0, 1]$, $r \ge 0$, we can apply Theorem 22.28(i) to deduce the modified transport inequality $C\_L(\mu, \nu) \le \max(2c/K,\, 1)\, H\_\nu(\mu)$. This implies (iii) since $C\_{q\ell} \le (2/c)\, C\_L$.

**(iii) $\Rightarrow$ (i):** If $\nu$ satisfies (iii), then it also satisfies $C\_L(\mu, \nu) \le C\, H\_\nu(\mu)$, where $L$ is as before (with $c = 1$); as a consequence, it satisfies the generalized Poincaré inequality of Theorem 22.28(ii). Pick up any Lipschitz function $f$ and apply this inequality to $\varepsilon f$, where $\varepsilon$ is small enough that $\varepsilon \lVert f \rVert\_{\mathrm{Lip}} < 1$; the result is $\int f\, d\nu = 0 \implies \varepsilon^2 \int f^2\, d\nu \le (2C) \int L^\ast(\varepsilon \lvert \nabla^- f \rvert)\, d\nu$. Since $L^\ast$ is quadratic on $[0, 1]$, factors $\varepsilon^2$ cancel out on both sides, and we are back with the usual Poincaré inequality.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.30</span><span class="math-callout__name">(Measure Concentration from Poincaré Inequality)</span></p>

Let $M$ be a Riemannian manifold equipped with its geodesic distance, and with a reference probability measure $\nu = e^{-V}\, \mathrm{vol}$. Assume that $\nu$ satisfies a Poincaré inequality with constant $\lambda$. Then there is a constant $C = C(\lambda) > 0$ such that for any Borel set $A$,

$$\forall r \ge 0, \qquad \nu[A^r] \ge 1 - \frac{e^{-C\, \min(r,\, r^2)}}{\nu[A]}. \tag{22.61}$$

Moreover, for any $f \in \mathrm{Lip}(M)$ (resp. $\mathrm{Lip}(M) \cap L^1(\nu)$),

$$\nu\!\left[\left\lbrace x;\ f(x) \ge m + r \right\rbrace\right] \le e^{-C\, \min\!\left(\frac{r}{\lVert f \rVert_{\mathrm{Lip}}},\, \frac{r^2}{\lVert f \rVert_{\mathrm{Lip}}^2}\right)}, \tag{22.62}$$

where $m$ is a median (resp. the mean value) of $f$ with respect to $\nu$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.30)</span></p>

The proof of (22.61) is similar to the implication (i) $\Rightarrow$ (iii) in Theorem 22.10. Define $B = M \setminus A^r$, and let $\nu\_A = (1\_A)\, \nu / \nu[A]$, $\nu\_B = (1\_B)\, \nu / \nu[B]$. Obviously, $C\_{q\ell}(\nu\_A, \nu\_B) \ge \min(r, r^2)$. The inequality $\min(a + b, (a + b)^2) \le 4[\min(a, a^2) + \min(b, b^2)]$ makes it possible to adapt the proof of the triangle inequality for $W\_1$ and get $C\_{q\ell}(\nu\_A, \nu\_B) \le 4[C\_{q\ell}(\nu\_A, \nu) + C\_{q\ell}(\nu\_B, \nu)]$. Thus $\min(r, r^2) \le 4[C\_{q\ell}(\nu\_A, \nu) + C\_{q\ell}(\nu\_B, \nu)]$. By Theorem 22.28, $\nu$ satisfies (22.49), so there is $C > 0$ such that

$$\min(r, r^2) \le C\!\left(H_\nu(\nu_A) + H_\nu(\nu_B)\right) = C\!\left(\log \frac{1}{\nu[A]} + \log \frac{1}{1 - \nu[A^r]}\right),$$

and (22.61) follows immediately. Then (22.62) is obtained by arguments similar to those used before in the proof of Theorem 22.10.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 22.31</span></p>

The exponential measure $\nu(dx) = (1/2) e^{-\lvert x \rvert}\, dx$ does not admit Gaussian tails, so it fails to satisfy properties of Gaussian concentration expressed in Theorem 22.10. However, it does satisfy a Poincaré inequality. So (22.61), (22.62) hold true for this measure.

</div>

Now, consider the problem of concentration of measure in a *product space*, say $(M^N, \nu^{\otimes N})$, where $\nu$ satisfies a Poincaré inequality. We may equip $M^N$ with the metric $d\_2(x, y) = \sqrt{\sum\_i d(x\_i, y\_i)^2}$; then $\mu^{\otimes N}$ will satisfy a Poincaré inequality with the same constant as $\nu$, and we may apply Theorem 22.30 to study concentration in $(M^N, d\_2, \nu^{\otimes N})$.

However, there is a more interesting approach, due to Talagrand, in which one uses both the distance $d\_2$ and the distance $d\_1(x, y) = \sum\_i d(x\_i, y\_i)$. Here is the procedure: Given a Borel set $A \subset M^N$, first enlarge it by $r$ in distance $d\_2$ (that is, consider all points which lie at a distance less than $r$ from $A$); then enlarge the result by $r^2$ in distance $d\_1$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.32</span><span class="math-callout__name">(Product Measure Concentration from Poincaré Inequality)</span></p>

Let $M$ be a Riemannian manifold equipped with its geodesic distance $d$ and a reference probability measure $\nu = e^{-V}\, \mathrm{vol}$. Assume that $\nu$ satisfies a Poincaré inequality with constant $\lambda$. Then there is a constant $C = C(\lambda)$ such that for any $N \in \mathbb{N}$, and for any Borel set $A \subset M^N$,

$$\nu^{\otimes N}[A] \ge \frac{1}{2} \implies \nu^{\otimes N}\!\left[(A^{r;d_2})^{r^2;d_1}\right] \ge 1 - e^{-C\, r^2}. \tag{22.63}$$

Here $A^{r;d}$ stands for the enlargement of $A$ by $r$ in distance $d$, and $\lVert f \rVert\_{\mathrm{Lip}(\mathcal{X}, d)}$ stands for the Lipschitz norm of $f$ on $\mathcal{X}$ with respect to the distance $d$.

Moreover, for any $f \in \mathrm{Lip}(M^N, d\_1) \cap \mathrm{Lip}(M^N, d\_2) \cap L^1(\nu^{\otimes N})$,

$$\nu^{\otimes N}\!\left[\left\lbrace x;\ f(x) \ge m + r \right\rbrace\right] \le e^{-C\, \min\!\left(\frac{r}{\lVert f \rVert_{\mathrm{Lip}(M^N, d_1)}},\, \frac{r^2}{\lVert f \rVert_{\mathrm{Lip}(M^N, d_2)}^2}\right)}, \tag{22.64}$$

where $m$ is a median (resp. the mean value) of $f$ with respect to the measure $\nu^{\otimes N}$.

Conversely, if (22.63) or (22.64) holds true, then $\nu$ satisfies a Poincaré inequality.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.32)</span></p>

Let us assume that $\nu$ satisfies a Poincaré inequality, and prove (22.63). By Theorem 22.25, $\nu$ satisfies a transport-cost inequality of the form $\forall \mu \in P\_1(M),\ C\_{q\ell}(\mu, \nu) \le C\, H\_\nu(\mu)$. On $M^N$ define the cost $c(x, y) = \sum c\_{q\ell}(x\_i, y\_i)$, and let $\overline{C}$ be the associated optimal cost functional. By Remark 22.7, $\nu^{\otimes N}$ satisfies an inequality of the form $\forall \mu \in P\_1(M^N),\ \overline{C}(\mu, \nu^{\otimes N}) \le C\, H\_{\nu^{\otimes N}}(\mu)$.

Let $A$ be a Borel set of $M^N$ with $\nu^{\otimes N}[A] \ge 1/2$, and let $r > 0$ be given. Let $B = M^N \setminus (A^{r;d\_2})^{r^2;d\_1}$. Let $\nu\_B$ be obtained by conditioning $\nu$ on $B$. Consider the problem of transporting $\nu\_B$ to $\nu$ optimally, with the cost $c$. At least a portion $\nu^{\otimes N}[A] \ge 1/2$ of the mass has to go to from $B$ to $A$, so $\overline{C}(\nu\_B, \nu^{\otimes N}) \ge \frac{1}{2} \inf\_{x \in A,\, y \in B} c(x, y) =: \frac{1}{2}\, c(A, B)$. On the other hand, by (22.65), $\overline{C}(\nu\_B, \nu^{\otimes N}) \le C\, H\_{\nu^{\otimes N}}(\mu) = C \log \frac{1}{\nu[B]}$.

The key geometric step is to show that $c(A, B) \ge r^2$. Indeed, let $x \in A$ and $y \in M^N$ such that $c(x, y) < r^2$; we must show $y \in (A^{r;d\_2})^{r^2;d\_1}$. For each $i$, define $z\_i = x\_i$ if $d(x\_i, y\_i) > 1$, $z\_i = y\_i$ otherwise. Then $d\_2(x, z)^2 = \sum\_{d(x\_i,y\_i) \le 1} d(x\_i, y\_i)^2 \le \sum\_i c\_{q\ell}(x\_i, y\_i) = c(x, y) < r^2$, so $z \in A^{r;d\_2}$. Similarly, $d\_1(z, y) = \sum\_{d(x\_i,y\_i) > 1} d(x\_i, y\_i) \le \sum\_i c\_{q\ell}(x\_i, y\_i) = c(x, y) < r^2$, so $y$ lies at distance at most $r^2$ from $z$ in distance $d\_1$. This concludes the proof.

The converse direction uses the empirical measure argument as in the proof of Theorem 22.22.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 22.33</span></p>

Let $\nu(dx)$ be the exponential measure $e^{-\lvert x \rvert}\, dx/2$ on $\mathbb{R}$, then $\nu^{\otimes N}(dx) = (1/2^N) e^{-\sum \lvert x\_i \rvert}\, \prod dx\_i$ on $\mathbb{R}^N$. Theorem 22.32 shows that for every Borel set $A \subset \mathbb{R}^N$ with $\nu^{\otimes N}[A] \ge 1/2$ and any $\delta > 0$,

$$\nu^{\otimes N}\!\left[A + B_r^{d_2} + B_{r^2}^{d_1}\right] \ge 1 - e^{-c\, r^2} \tag{22.67}$$

where $B\_r^d$ stands for the ball of center $0$ and radius $r$ in $\mathbb{R}^N$ for the distance $d$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.34</span></p>

Strange as this may seem, inequality (22.67) contains (up to numerical constants) the Gaussian concentration of the Gaussian measure! Indeed, let $T : \mathbb{R} \to \mathbb{R}$ be the increasing rearrangement of the exponential measure $\nu$ onto the one-dimensional Gaussian measure $\gamma$ (so $T\_\# \nu = \gamma$, $(T^{-1})\_\# \gamma = \nu$). An explicit computation shows that

$$\lvert T(x) - T(y) \rvert \le C \min\!\left(\lvert x - y \rvert,\, \sqrt{\lvert x - y \rvert}\right) \tag{22.68}$$

for some numeric constant $C$. Let $T\_N(x\_1, \ldots, x\_N) = (T(x\_1), \ldots, T(x\_N))$; obviously $(T\_N)\_\#(\nu^{\otimes N}) = \gamma^{\otimes N}$, $(T\_N)\_\#^{-1}(\gamma^{\otimes N}) = \nu^{\otimes N}$. By (22.68), if $C' = \sqrt{8C}$, then $T\_N\!\left(T\_N^{-1}(A) + B\_r^{d\_2} + B\_{r^2}^{d\_1}\right) \subset A + B\_{C' r}^{d\_2}$. As a consequence, if $A \subset \mathbb{R}^N$ is any Borel set satisfying $\gamma^{\otimes N}[A] \ge 1/2$, then $\gamma^{\otimes N}[A^{C' r}] \ge 1 - e^{-c\, r^2}$ for some numeric constant $c > 0$. This is precisely the Gaussian concentration property as it appears in Theorem 22.10(iii) — in a dimension-free form.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.35</span></p>

In certain situations, (22.67) provides sharper concentration properties for the Gaussian measure, than the usual Gaussian concentration bounds. This might look paradoxical, but can be explained by the fact that Gaussian concentration considers *arbitrary* sets $A$, while in many problems one is led to study the concentration of measure around certain very particular sets, for instance with a "cubic" structure; then inequality (22.67) might be very efficient.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 22.36</span></p>

Let $A = \lbrace x \in \mathbb{R}^N;\ \max \lvert x\_i \rvert \le m \rbrace$ be the centered cube of side $2m$, where $m = m(N) \to \infty$ is chosen in such a way that $\gamma^{\otimes N}[A] \ge 1/2$. (It is a classical fact that $m = O(\sqrt{\log N})$ will do, but we don't need that information.) If $r \ge 1$ is small with respect to $m$, then the enlargement of the cube is dominated by the behavior of $T$ close to $T^{-1}(m)$. Since $T(x)$ behaves approximately like $\sqrt{x}$ for large values of $x$, $T^{-1}(m)$ is of the order $m^2$; and close to $m^2$ the Lipschitz norm of $T$ is $O(1/m)$. Then the computation before can be sharpened into

$$T_N\!\left(T_N^{-1}(A) + B_r^{d_2} + B_{r^2}^{d_1}\right) \subset A + B_{C'\, r^2/m}^{d_2};$$

so the concentration of measure can be felt with enlargements by a distance of the order of $r^2/m \ll r$.

</div>

### Dimension-Dependent Inequalities

There is no well-identified analog of Talagrand inequalities that would take advantage of the finiteness of the dimension to provide sharper concentration inequalities. In this section we suggest some natural possibilities, focusing on positive curvature for simplicity; so $M$ will be compact. This compactness assumption is not a serious restriction: If $(M, e^{-V}\, \mathrm{vol})$ is any Riemannian manifold satisfying a $\mathrm{CD}(K, N)$ inequality for some $K \in \mathbb{R}$, $N < \infty$, and $e^{-V}\, \mathrm{vol}$ satisfies a Talagrand inequality, then it can be shown that $M$ is compact.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.37</span><span class="math-callout__name">(Finite-Dimensional Transport-Energy Inequalities)</span></p>

Let $M$ be a Riemannian manifold equipped with a probability measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying a curvature-dimension bound $\mathrm{CD}(K, N)$ for some $K > 0$, $N \in (1, \infty)$. Then, for any $\mu = \rho\, \nu \in P\_2(M)$,

$$\int_M \left[N\!\left(\frac{\alpha}{\sin \alpha}\right)^{\!1 - \frac{1}{N}} \rho(x_0)^{-\frac{1}{N}} - (N - 1) \frac{\alpha}{\tan \alpha}\right] \pi(dx_0\, dx_1) \le 1, \tag{22.69}$$

where $\alpha(x\_0, x\_1) = \sqrt{K/(N-1)}\, d(x\_0, x\_1)$, and $\pi$ is the unique optimal coupling between $\mu$ and $\nu$. Equivalently,

$$\int_M \left[N\!\left(\frac{\alpha}{\sin \alpha}\right)^{\!1 - \frac{1}{N}} - (N-1) \frac{\alpha}{\tan \alpha} - 1\right] \pi(dx_0\, dx_1) \le \int \left(\frac{\alpha}{\sin \alpha}\right)^{\!1 - \frac{1}{N}} \left[(N-1)\rho - N\rho^{1 - \frac{1}{N}} + 1\right] d\nu. \tag{22.70}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.38</span></p>

The function $(N-1)r - Nr^{1 - 1/N} + 1$ is nonnegative, and so is the integrand in the right-hand side of (22.70). If the coefficient $\alpha / \sin \alpha$ above would be replaced by $1$, then the right-hand side of (22.70) would be just $\int [(N-1)\rho - N\rho^{1-1/N} + 1]\, d\nu = H\_{N,\nu}(\rho)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 22.39</span><span class="math-callout__name">(Further Finite-Dimensional Transport-Energy Inequalities)</span></p>

With the same assumptions and notation as in Theorem 22.37, the following inequalities hold true:

$\forall p \in (1, \infty)$, $\qquad H\_{Np,\nu}(\mu) \ge$

$$\int \left[(Np - 1) - (N-1) \frac{\alpha}{\tan \alpha} - N(p-1)\!\left(\frac{\sin \alpha}{\alpha}\right)^{\!\frac{1}{p-1}\!\left(1 - \frac{1}{N}\right)}\right] d\pi; \tag{22.71}$$

$$2 H_{N,\nu}(\mu) - \int \rho^{1 - \frac{1}{N}} \log \rho\, d\nu \ge \int \left[(2N - 1) - (N-1) \frac{\alpha}{\tan \alpha} - N \exp\!\left(1 - \left(\frac{\alpha}{\sin \alpha}\right)^{\!1 - \frac{1}{N}}\right)\right] d\pi; \tag{22.72}$$

$$H_{\infty,\nu}(\mu) \ge (N-1) \int \left(1 - \frac{\alpha}{\tan \alpha} + \log \frac{\alpha}{\sin \alpha}\right) d\pi. \tag{22.73}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.37)</span></p>

Apply Theorem 20.10 with $U(r) = -Nr^{1-1/N}$, $\rho\_0 = 1$, $\rho\_1 = \rho$: This yields the inequality

$$-N \le -\int N\, \rho^{1-\frac{1}{N}}\, \left(\frac{\alpha}{\sin \alpha}\right)^{1-\frac{1}{N}} d\pi(\,\cdot\, \lvert \,\cdot\,)\, d\nu - \int (N-1)\!\left(1 - \frac{\alpha}{\tan \alpha}\right) d\pi\, d\nu,$$

which since $\pi$ has marginals $\rho\, \nu$ and $\nu$, is the same as (22.69).

To derive (22.70) from (22.69), it is sufficient to check that $\int NQ\, d\pi = \int Q[(N-1)\rho + 1]\, d\nu$, where $Q = (\alpha/\sin \alpha)^{1-1/N}$. But this is immediate because $Q$ is a symmetric function of $x\_0$ and $x\_1$, and $\pi$ has marginals $\rho\, \nu$ and $\nu$.

</div>

All the inequalities appearing in Corollary 22.39 can be seen as refinements of the Talagrand inequality appearing in Theorem 22.14; concentration inequalities derived from them take into account, for instance, the fact that the distance between any two points can never exceed $\pi\sqrt{(N-1)/K}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.42</span></p>

If one applies the same procedure to (22.71), one recovers a constant $K(Np)/(Np - 1)$, which reduces to the correct constant only in the limit $p \to 1$. As for inequality (22.73), it leads to just $K$ (which would be the limit $p \to \infty$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.43</span></p>

Since the Talagrand inequality implies a Poincaré inequality without any loss in the constants, and the optimal constant in the Poincaré inequality is $KN/(N-1)$, it is natural to ask whether this is also the optimal constant in the Talagrand inequality. The answer is affirmative, in view of Theorem 22.17, since the logarithmic Sobolev inequality also holds true with the same constant. But I don't know of any transport proof of this!

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Open Problem 22.44</span></p>

*Find a direct transport argument to prove that the curvature-dimension $\mathrm{CD}(K, N)$ with $K > 0$ and $N < \infty$ implies $T\_2(\widetilde{K})$ with $\widetilde{K} = KN/(N-1)$, rather than just $T\_2(K)$.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Open Problem 22.45</span></p>

*In the Euclidean case, is there a particular variant of the Talagrand inequality which takes advantage of the homogeneity under dilations, just as the usual Sobolev inequality in $\mathbb{R}^n$? Is it useful?*

</div>

### Recap

The main results of this chapter can be summarized by just a few diagrams:

**Relations between functional inequalities:** By combining Theorems 21.2, 22.17, 22.10 and elementary inequalities, one has

$$\mathrm{CD}(K, \infty) \implies (\mathrm{LS}) \implies (T_2) \implies (\mathrm{P}) \implies (\exp_1)$$

$$(T_1) \iff (\exp_2) \implies (\exp_1)$$

All these symbols designate properties of the reference measure $\nu$: (LS) stands for logarithmic Sobolev inequality, (P) for Poincaré inequality, $\exp\_2$ means that $\nu$ has a finite square-exponential moment, and $\exp\_1$ that it has a finite exponential moment.

**Reformulations of Poincaré inequality:** Theorem 22.25 can be visualized as

$$(\mathrm{P}) \iff (\mathrm{LSLL}) \iff (T_{q\ell})$$

where (LSLL) means logarithmic Sobolev for log-Lipschitz functions, and $(T\_{q\ell})$ designates the transportation-cost inequality involving the quadratic-linear cost.

**Concentration properties via functional inequalities:** The three main such results proven in this chapter are

- $(T\_1) \iff$ *Gaussian concentration* (Theorem 22.10)
- $(T\_2) \iff$ *dimension free* Gaussian concentration (Theorem 22.22)
- $(\mathrm{P}) \iff$ dimension free *exponential* concentration (Theorem 22.32)

### Appendix: Properties of the Hamilton–Jacobi Semigroup

This Appendix is devoted to the proof of Theorem 22.46 below, which was used in the proof of Theorem 22.28 (and also in the proof of Theorem 22.17 via Proposition 22.16). It says that if a nice convex Lagrangian $L(\lvert v \rvert)$ is given on a Riemannian manifold, then the solution $f(t, x)$ of the associated Hamilton–Jacobi semigroup satisfies (a) certain regularity properties which go beyond differentiability; (b) the *pointwise* differential equation $\frac{\partial f}{\partial t} + L^\ast(\lvert \nabla^- f(x) \rvert) = 0$, where $\lvert \nabla^- f(x) \rvert$ is defined by (20.2).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22.46</span><span class="math-callout__name">(Properties of the Hamilton–Jacobi Semigroup on a Manifold)</span></p>

Let $L : \mathbb{R}\_+ \to \mathbb{R}\_+$ be a strictly increasing, locally semiconcave, convex continuous function with $L(0) = 0$. Let $M$ be a Riemannian manifold equipped with its geodesic distance $d$. For any $f \in C\_b(M)$, define the evolution $(H\_t f)\_{t \ge 0}$ by

$$\begin{cases} H_0 f = f \\[4pt] (H_t f)(x) = \inf_{y \in M}\!\left[f(y) + t\, L\!\left(\frac{d(x, y)}{t}\right)\right] \qquad (t > 0,\ x \in M). \end{cases} \tag{22.74}$$

Then:

- (i) For any $s, t \ge 0$, $H\_s\, H\_t\, f = H\_{t+s}\, f$.
- (ii) For any $x \in M$, $\inf f \le (H\_t f)(x) \le f(x)$; moreover, for any $t > 0$ the infimum over $M$ in (22.74) can be restricted to the closed ball $B[x, R(f, t)]$, where $R(f, t) = t\, L^{-1}\!\left(\frac{\sup f - \inf f}{t}\right)$.
- (iii) For any $t > 0$, $H\_t f$ is Lipschitz and locally semiconcave on $M$; moreover $\lVert H\_t f \rVert\_{\mathrm{Lip}} \le L'(\infty)$.
- (iv) For any $t > 0$, $H\_{t+s} f$ is nonincreasing in $s$, and converges monotonically and locally uniformly to $H\_t f$ as $s \to 0$; this conclusion extends to $t = 0$ if $\lVert f \rVert\_{\mathrm{Lip}} \le L'(\infty)$.
- (v) For any $t \ge 0$, $s > 0$, $x \in M$,

$$\frac{\lvert H_{t+s} f(x) - H_t f(x) \rvert}{s} \le L^*\!\left(\lVert H_t f \rVert_{\mathrm{Lip}(B[x, R(f, s)])}\right).$$

- (vi) For any $x \in M$ and $t > 0$,

$$\liminf_{s \downarrow 0} \frac{(H_{t+s} f)(x) - (H_t f)(x)}{s} \ge -L^*\!\left(\lvert \nabla^- H_t f \rvert(x)\right);$$

this conclusion extends to $t = 0$ if $\lVert f \rVert\_{\mathrm{Lip}} \le L'(\infty)$.

- (vii) For any $x \in M$ and $t > 0$,

$$\lim_{s \downarrow 0} \frac{(H_{t+s} f)(x) - (H_t f)(x)}{s} = -L^*\!\left(\lvert \nabla^- H_t f \rvert(x)\right);$$

this conclusion extends to $t = 0$ if $\lVert f \rVert\_{\mathrm{Lip}} \le L'(\infty)$ and $f$ is locally semiconcave.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.47</span></p>

If $L'(\infty) < +\infty$ then in general $H\_t f$ is *not* continuous as a function of $t$ at $t = 0$. This can be seen by the fact that $\lVert H\_t f \rVert\_{\mathrm{Lip}} \le L'(\infty)$ for all $t > 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 22.48</span></p>

There is no measure theory in Theorem 22.46, and conclusions hold for *all* (not just almost all) $x \in M$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 22.46)</span></p>

First, note that the inverse $L^{-1}$ of $L$ is well-defined $\mathbb{R}\_+ \to \mathbb{R}\_+$ since $L$ is strictly increasing and goes to $+\infty$ at infinity. Also $L'(\infty) = \lim\_{r \to \infty}(L(r)/r)$ is well-defined in $(0, +\infty]$. Further, note that $L^\ast(p) = \sup\_{r \ge 0} [p\, r - L(r)]$ is a convex nondecreasing function of $p$, satisfying $L^\ast(0) = 0$.

**(i):** Let $x, y, z \in M$ and $t, s > 0$. Since $L$ is increasing and convex,

$$L\!\left(\frac{d(x, y)}{t + s}\right) \le L\!\left(\frac{d(x, z) + d(z, y)}{t + s}\right) \le \frac{t}{t+s}\, L\!\left(\frac{d(x, z)}{t}\right) + \frac{s}{t+s}\, L\!\left(\frac{d(z, y)}{s}\right),$$

with equality if $d(x, z)/t = d(z, y)/s$, i.e. if $z$ is an $s/(t+s)$-barycenter of $x$ and $y$. So $(t+s)\, L\!\left(\frac{d(x+y)}{t+s}\right) = \inf\_{z \in M}\!\left[t\, L\!\left(\frac{d(x,z)}{t}\right) + s\, L\!\left(\frac{d(z,y)}{s}\right)\right]$. This implies (i).

**(ii):** The lower bound in (ii) is obvious since $L \ge 0$, and the upper bound follows from the choice $y = x$ in (22.74). Moreover, if $d(x, y) > R(f, t)$, then $f(x) + t\, L(d(x, y)/t) > (\inf f) + t\, L(R(f, t)/t) = (\inf f) + (\sup f - \inf f) = \sup f$; so the infimum in (22.74) may be restricted to those $y \in M$ such that $d(x, y) \le R(f, t)$.

**(iii):** When $y$ varies in $B[x, R(f, t)]$, the function $t\, L(d(x, y)/t)$ remains $C$-Lipschitz, where $C = L'(R(f, t)/t)$. So $H\_t f$ is an infimum of uniformly Lipschitz functions, and is therefore Lipschitz. It is obvious that $C \le L'(\infty)$.

To prove that $H\_t f$ is locally semiconcave for $t > 0$: Let $(\gamma\_t)\_{0 \le t \le 1}$ be a minimizing geodesic in $M$, then for $\lambda \in [0, 1]$, $H\_t f(\gamma\_\lambda) - (1 - \lambda)\, H\_t f(\gamma\_0) - \lambda\, H\_t f(\gamma\_1) \ge t\, \inf\_{z \in B}\!\left[L(d(z, \gamma\_\lambda)/t) - (1-\lambda)\, L(d(z, \gamma\_0)/t) - \lambda\, L(d(z, \gamma\_1)/t)\right]$. When $z$ varies in a compact set $B$, the distance function $d(z, \cdot)$ is uniformly semiconcave (with a quadratic modulus) on the compact set $K$; let $F = L(\cdot/t)$, restricted to a large interval where $d(z, \cdot)$ takes values; and let $\varphi = d(z, \cdot)$, restricted to $K$. Since $F$ is semiconcave increasing and $\varphi$ is semiconcave Lipschitz, their composition $F \circ \varphi$ is semiconcave, and the modulus of semiconcavity is uniform in $z$. So there is $C = C(K)$ such that the infimum is $\ge -C\, \lambda(1 - \lambda)\, d(\gamma\_0, \gamma\_1)^2$. This shows that $H\_t f$ is locally semiconcave.

**(iv):** Let $g = H\_t f$. It is clear that $H\_s g$ is a nonincreasing function of $s$ since $s\, L(d(x, y)/s)$ is itself a nonincreasing function of $s$. Two cases:

- **Case 1: $L'(\infty) = +\infty$.** Then $\lim\_{s \to 0} R(g, s) = (\sup g - \inf g)/L'(\infty) = 0$. For any $x \in M$, $g(x) \ge H\_s g(x) = \inf\_{d(x,y) \le R(g,s)}\!\left[g(y) + s\, L(d(x, y)/s)\right] \ge \inf\_{d(x,y) \le R(g,s)} g(y)$, and this converges to $g(x)$ as $s \to 0$, locally uniformly in $x$.

- **Case 2: $L'(\infty) < +\infty$.** Then $\lim\_{s \to 0} R(g, s) > 0$. Since $\lVert g \rVert\_{\mathrm{Lip}} \le L'(\infty)$, we have $g(y) \ge g(x) - L'(\infty)\, d(x, y)$, so $g(x) \ge H\_s g(x) \ge g(x) + [s\, L(R(g, s)/s) - L'(\infty)\, R(g, s)]$. As $s \to 0$, the expression inside square brackets goes to $0$, and $H\_s g(x)$ converges uniformly to $g(x)$.

**(v):** Again let $g = H\_t f$, then $0 \le g(x) - H\_s g(x) = \sup\_{d(x,y) \le R(g,s)}\!\left[g(x) - g(y) - s\, L(d(x, y)/s)\right] \le s\, L^\ast\!\left(\sup\_{d(x,y) \le R(g,s)} \frac{[g(y) - g(x)]\_-}{d(x, y)}\right)$, where I have used the inequality $p\, r \le L(r) + L^\ast(p)$. Moreover, if $L'(\infty) = +\infty$, then $L^\ast$ is continuous on $\mathbb{R}\_+$, so by the definition of $\lvert \nabla^- g \rvert$ and the fact that $R(g, s) \to 0$,

$$\limsup_{s \downarrow 0} \frac{g(x) - H_s g(x)}{s} \le L^*\!\left(\lim_{s \downarrow 0} \sup_{d(x,y) \le R(g,s)} \frac{[g(y) - g(x)]_-}{d(x, y)}\right) = L^*\!\left(\lvert \nabla^- g(x) \rvert\right),$$

which proves (v) in the case $L'(\infty) = +\infty$.

If $L'(\infty) < +\infty$, then of course $\lvert \nabla^- g(x) \rvert \le L'(\infty)$. Two subcases: if $\lvert \nabla^- g(x) \rvert = L'(\infty)$, the same argument as before shows $(g(x) - H\_s g(x))/s \le L^\ast(\lVert g \rVert\_{\mathrm{Lip}}) \le L^\ast(L'(\infty)) = L^\ast(\lvert \nabla^- g(x) \rvert)$. If $\lvert \nabla^- g(x) \rvert < L'(\infty)$, there is a function $\alpha = \alpha(s) \to 0$ as $s \to 0$, such that the infimum defining $H\_s g(x)$ may be restricted to $d(x, y) \le \alpha(s)$ (by Lemma 22.49 below). Then the same limiting argument as in the case $L'(\infty) = +\infty$ works.

**(vi) and (vii):** These are the most delicate parts. The key idea for (vii) is: let $g = H\_t f$; as we already know, $\lVert g \rVert\_{\mathrm{Lip}} \le L'(\infty)$ and $g$ is locally semiconcave. The problem is to show $\liminf\_{s \downarrow 0} (g(x) - H\_s g(x))/s \ge L^\ast(\lvert \nabla^- g(x) \rvert)$. For $s$ small enough, $R(g, s) > s\, q$ (where $q \in \partial L(\lvert \nabla^- g(x) \rvert)$). This implies

$$\frac{g(x) - H_s g(x)}{s} \ge \frac{1}{s} \sup_{d(x,y) = s\, q}\!\left[g(x) - g(y) - s\, L\!\left(\frac{d(x,y)}{s}\right)\right] = \sup_{d(x,y) = s\, q}\!\left[\left(\frac{g(x) - g(y)}{d(x,y)}\right) q - L(q)\right].$$

Let $\psi(r) = \sup\_{d(x,y) = r}\!\left[\frac{g(x) - g(y)}{d(x,y)}\right]$. If it can be shown that $\psi(r) \xrightarrow{r \to 0} \lvert \nabla^- g(x) \rvert$, then we can pass to the limit in the above and recover $\liminf\_{s \downarrow 0} (g(x) - H\_s g(x))/s \ge \lvert \nabla^- g(x) \rvert\, q - L(q) = L^\ast(\lvert \nabla^- g(x) \rvert)$.

The convergence $\psi(r) \to \lvert \nabla^- g(x) \rvert$ as $r \to 0$ is where the semiconcavity of $g$ is used. Let $S\_r$ denote the sphere of center $x$ and radius $r$. If $r$ is small enough, for any $z \in S\_r$ there is a unique geodesic joining $x$ to $z$, and the exponential map induces a bijection between $S\_{r'}$ and $S\_r$, for any $r' \in (0, r]$. Let $\lambda = r'/r \in (0, 1]$; for any $y \in S\_{r'}$ we can find a unique geodesic $\gamma$ such that $\gamma\_0 = x$, $\gamma\_\lambda = y$, $\gamma\_1 \in S\_r$. By semiconcavity, there is a constant $C = C(x, r)$ such that $g(\gamma\_\lambda) - (1 - \lambda)\, g(\gamma\_0) - \lambda\, g(\gamma\_1) \ge -C\, \lambda(1 - \lambda)\, d(\gamma\_0, \gamma\_1)^2$. This can be rewritten

$$\frac{g(x) - g(\gamma_1)}{d(x, \gamma_1)} - \frac{g(x) - g(\gamma_\lambda)}{\lambda\, d(x, \gamma_1)} \ge -C\, (r - r').$$

So $d(x, y) = r' \implies \psi(r) - \frac{g(x) - g(y)}{d(x, y)} \ge -C\, (r - r')$. By taking the supremum over $y$, we conclude that $\psi(r) - \psi(r') \ge -C\, (r - r')$. In particular $\psi(r) + C\, r$ is a nondecreasing function of $r$, so $\psi$ has a limit as $r \to 0$. This concludes the proof.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 22.49</span></p>

Let $M$ be a Riemannian manifold (or more generally, a geodesic space), and let $L, R$ be positive numbers. If $g : M \to \mathbb{R}$ is $L$-Lipschitz and $\lvert \nabla^- g(x) \rvert < L$ for some $x \in M$, then there is $\delta > 0$ such that for any $y \in B[x, R]$,

$$g(x) - g(y) \le (L - \delta)\, d(x, y).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Lemma 22.49)</span></p>

By assumption, $\limsup\_{y \to x} \frac{[g(y) - g(x)]\_-}{d(x, y)} < L$. So there are $r > 0$, $\eta > 0$ such that if $d(x, z) \le r$ then $g(x) \le g(z) + (L - \eta)\, d(x, z)$. Let $y \in B[z, R]$ and let $\gamma$ be a geodesic joining $\gamma(0) = x$ to $\gamma(1) = y$; let $z = \gamma(r/R)$. Then $d(x, z) = (r/R)\, d(x, y) \le r$, so (22.83) holds true. As a consequence,

$$g(x) - g(y) = [g(x) - g(z)] + [g(z) - g(y)] \le (L - \eta)\, d(x, z) + L\, d(z, y) = L\, d(x, y) - \eta\, d(x, z) \le \left(L - \eta\, \frac{r}{R}\right) d(x, y),$$

which proves the lemma with $\delta = \eta\, r / R$.

</div>

### Bibliographical Notes on Chapter 22

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

Most of the literature described below is reviewed with more detail in the synthesis works of **Ledoux**. Selected applications of the concentration of measure to various parts of mathematics (Banach space theory, fine study of Brownian motion, combinatorics, percolation, spin glass systems, random matrices, etc.) are briefly developed in these works. The role of $T\_p$ inequalities in that theory is also discussed there.

**Lévy** is often quoted as the founding father of concentration theory. His work might have been forgotten without the determination of **V. Milman** to make it known. The modern period of concentration of measure starts with a work by Milman himself on the so-called **Dvoretzy theorem**.

The **Lévy–Gromov isoperimetric inequality** is a way to get sharp concentration estimates from Ricci curvature bounds. **Gromov** has further worked on the links between Ricci curvature and concentration. Also **Talagrand** made decisive contributions to the theory of concentration of measure, mainly in product spaces. **Dembo** showed how to recover several of Talagrand's results in an elegant way by means of information-theoretical inequalities.

$T\_p$ inequalities have been studied for themselves at least since the beginning of the nineties; the **Csiszár–Kullback–Pinsker** inequality can be considered as their ancestor from the sixties. It is easy to show that transport inequalities are stable under weak convergence. They are also stable under pushforward.

**Proposition 22.3** was studied by **Rachev** and **Bobkov and Götze**, in the cases $p = 1$ and $p = 2$. These duality formulas were later systematically exploited by **Bobkov, Gentil and Ledoux**. The tensorization argument used in Proposition 22.5 goes back to **Marton**. As for Lemma 22.8, it is as old as information theory, since **Shannon** used it to motivate the introduction of entropy in this context. After Marton's work, this tensorization technique has been adapted to various situations, such as weakly dependent Markov chains. Relations with the so-called **Dobrushin(–Shlosman) mixing condition** appear in some of these works; in fact, the original version of the mixing condition was formulated in terms of optimal transport!

It is also **Marton** who introduced the simple argument by which $T\_p$ inequalities lead to concentration inequalities (implication (i) $\Rightarrow$ (iii) in Theorem 22.10), and which has since then been reproduced in nearly all introductions to the subject. She used it mainly with the so-called **Hamming distance**: $d((x\_i)\_{1 \le i \le n}, (y\_i)\_{1 \le i \le n}) = \sum 1\_{x\_i \neq y\_i}$.

There are alternative functional approaches to the concentration of measure: via logarithmic Sobolev inequalities; and via **Brunn–Minkowski**, **Prékopa–Leindler**, or isoperimetric inequalities. For instance, (19.27) immediately implies $\nu[A^r] \ge 1 - e^{-Kr^2/4} / \nu[A]$. This kind of inequality goes back to **Gromov and V. Milman**, who also studied concentration from Poincaré inequalities. The relations between Poincaré inequalities and concentration are reviewed by **E. Milman**, who also shows that a weak concentration estimate, together with the $\mathrm{CD}(0, \infty)$ criterion, implies Poincaré (and even the stronger **Cheeger isoperimetric inequality**).

**Theorem 22.10** has been obtained by patching together results due to **Bobkov and Götze**, **Djellout, Guillin and Wu**, and **Bolley and myself**, together with a few arguments from folklore. There is an alternative proof of (ix) $\Rightarrow$ (i) based on the following fact, well-known to specialists: *Let $X$ be a centered real random variable such that $\mathbb{E}\, e^{X^2} < \infty$, then the Laplace transform of $X$ is bounded above by a Gaussian Laplace transform.*

The **CKP inequality** (22.25) was found independently by **Pinsker**, **Kullback** and **Csiszár**. The approach used in Remark 22.12 is taken from work with **Bolley**; it takes inspiration from an argument which I heard in a graduate course by **Talagrand**.

**Weighted** CKP inequalities such as (22.16) were introduced in the author's paper with **Bolley**; then **Gozlan and Léonard** studied similar inequalities from the point of view of the theory of large deviations.

**Talagrand** proved Theorem 22.14 when $\nu$ is the Gaussian measure in $\mathbb{R}^n$, using a change of variables in the one-dimensional case, and then a tensorization argument (Corollary 22.6). This strategy was developed by **Blower** who proved Theorem 22.14 when $M = \mathbb{R}^n$, $\nu(dx) = e^{-V(x)}\, dx$, $\nabla^2 V \ge K > 0$; see also **Cordero-Erausquin**. Generalizations to nonquadratic costs appear in the literature. Also **Kolesnikov** made systematic use of this approach in infinite-dimensional situations and for various classes of inequalities.

**Otto and I** found an alternative approach to Theorem 22.14, via the HWI inequality (which at the time had been established only in $\mathbb{R}^n$). The proof which I have used in this chapter is the same as the proof in that work, modulo the extension of the HWI inequality to general Riemannian manifolds.

**Theorem 22.17** (log Sobolev implies $T\_2$ implies Poincaré) was first proven by **Otto and myself**; the Otto calculus had first been used to get an idea of the result. Our proof relied on a heat semigroup argument, which will be explained later in Chapter 25. The "dual" strategy which I have used in this chapter, based on the **Hamilton–Jacobi semigroup**, is due to **Bobkov, Gentil and Ledoux**. In their original work it was assumed that the Ricci curvature of the manifold $M$ is bounded below, and this assumption was removed later. The methods were pushed to treat nonquadratic cost functions. Infimum convolutions in the style of the Hopf–Lax formula also play a role, in relation with logarithmic or plain Sobolev inequalities. Much later, **Gozlan** found a third proof of Theorem 22.17, based on Theorem 22.22 (which is itself based on Sanov's theorem).

The remarkable result according to which dimension free Gaussian concentration bounds are *equivalent* to $T\_2$ inequality (Theorem 22.22) is due to **Gozlan**; the proof of (iii) $\Rightarrow$ (i) in Theorem 22.22 is extracted from this paper. **Gozlan's** argument relies on **Sanov's theorem** in large deviation theory; this classical result states that the rate of deviation of the empirical measure of independent identically distributed samples is the (Kullback) information with respect to their common law; in other words, under adequate conditions, $-\frac{1}{N} \log \nu^{\otimes N}[\widehat{\mu}\_x^N \in A] \simeq \inf\!\left\lbrace H\_\nu(\mu);\ \mu \in A \right\rbrace$.

**Varadarajan's theorem** (law of large numbers for empirical measures) was already used in the proof of Theorem 5.10; it is anyway implied by Sanov's theorem.

Theorem 22.10 shows that $T\_1$ is quite well understood, but many questions remain open about the more interesting $T\_2$ inequality. One of the most natural is the following: given a probability measure $\nu$ satisfying $T\_2$, and a bounded function $v$, does $e^{-v}\, \nu / (\int e^{-v}\, d\nu)$ also satisfy a $T\_2$ inequality? For the moment, the only partial result in this direction is (22.29).

If one considers probability measures of the form $e^{-V(x)}\, dx$ with $V(x)$ behaving like $\lvert x \rvert^\beta$ for large $\lvert x \rvert$, then the critical exponents for concentration-type inequalities are the same as we already discussed for isoperimetric-type inequalities: If $\beta \ge 2$ there is the $T\_2$ inequality, while for $\beta = 1$ there is the transport inequality with linear-quadratic cost function. What happens for intermediate values of $\beta$ has been investigated by **Gentil, Guillin and Miclo**, by means of modified logarithmic Sobolev inequalities in the style of **Bobkov and Ledoux**.

It was shown that (Talagrand) $\Rightarrow$ (log Sobolev) in $\mathbb{R}^n$, if the reference measure $\nu$ is log concave (with respect to the Lebesgue measure). It was natural to conjecture that the same argument would work under an assumption of nonnegative curvature (say $\mathrm{CD}(0, \infty)$); **Theorem 22.21** shows that such is indeed the case.

It is only recently that **Cattiaux and Guillin** produced a counterexample on the real line, showing that the $T\_2$ inequality does not necessarily imply a log Sobolev inequality. Their counterexample takes the form $d\nu = e^{-V}\, dx$, where $V$ oscillates rather wildly at infinity, in particular $V''$ is not bounded below. More precisely, their potential looks like $V(x) = \lvert x \rvert^3 + 3x^2 \sin^2 x + \lvert x \rvert^\beta$ as $x \to +\infty$; then $\nu$ satisfies a logarithmic Sobolev inequality only if $\beta \ge 5/2$, but a $T\_2$ inequality as soon as $\beta > 2$. Counterexamples with $V''$ bounded below have still not yet been found.

Even more recently, **Gozlan** exhibited a characterization of $T\_2$ and other transport inequalities on $\mathbb{R}$, for certain classes of measures. He even identified situations where it is useful to deduce logarithmic Sobolev inequalities from $T\_2$ inequalities. **Gentil, Guillin and Miclo** considered transport inequalities on $\mathbb{R}$ for log-concave probability measures.

Theorem 22.14 admits an almost obvious generalization: if $\mathcal{F}$ is uniformly $K$-displacement convex and has a minimum at $\nu$, then $\frac{K\, W\_2(\mu, \nu)^2}{2} \le \mathcal{F}(\mu) - \mathcal{F}(\nu)$. Such inequalities have been studied and have proven useful in the study of certain partial differential equations.

Optimal transport inequalities in *infinite dimension* have started to receive a lot of attention recently, for instance on the **Wiener space**. A major technical difficulty is that the natural distance in this problem, the so-called **Cameron–Martin distance**, takes the value $+\infty$ "most of the time". **Gentil** established the $T\_2$ inequality for the Wiener measure by using the logarithmic Sobolev inequality on the Wiener space, and adapting the proof of Theorem 22.17(i) to that setting. **Feyel and Üstünel** on the one hand, and **Djellout, Guillin and Wu** on the other, suggested a more direct approach based on **Girsanov's formula**. **Fang and Shao** also extended Theorem 22.17 (Logarithmic Sobolev implies Talagrand inequality) to an infinite-dimensional setting, via the study of the Hamilton–Jacobi semigroup in infinite dimension. Very recently, **Fang and Shao** used Talagrand inequalities to obtain results of unique existence of optimal transport in the Wiener space over a Lie group.

The equivalence between Poincaré inequalities and modified transport inequalities, as expressed in Theorem 22.25, has a long history. **Talagrand** had identified concentration properties satisfied by the exponential measure, or a product of exponential measures. He established a precise version of (22.67): $\nu^{\otimes N}\!\left[A + 6\sqrt{r}\, B\_1^{d\_2} + 9r\, B\_1^{d\_1}\right] \ge 1 - e^{-r} / \nu^{\otimes N}[A]$. Then **Maurey** found a simple approach to concentration inequalities for the product exponential measure. Later, **Talagrand** made the connection with transport inequalities for the quadratic-linear cost. **Bobkov and Ledoux** introduced modified logarithmic Sobolev inequalities, and showed their equivalence with Poincaré inequalities. Finally, **Bobkov, Gentil and Ledoux** understood how to deduce quadratic-linear transport inequalities from modified logarithmic Sobolev inequalities, thanks to the Hamilton–Jacobi semigroup.

The treatment of dimension-dependent Talagrand-type inequalities in the last section is inspired from a joint work with **Lott**. That topic had been addressed before, with different tools, by **Gentil**. The study of Hamilton–Jacobi equations is an old topic; Theorem 22.46 (behavior of solutions of Hamilton–Jacobi equations) has been obtained by generalizing the proof of Proposition 22.16 as it appears in work with **Lott**. When $L'(\infty) = +\infty$, the proof is basically the same, while there are a few additional technical difficulties if $L'(\infty) < +\infty$. In fact Proposition 22.16 was established in a more general context, namely when $M$ is a finite-dimensional Alexandrov space with (sectional) curvature locally bounded below. The same extension probably holds for Theorem 22.46, although part (vii) would require a bit more thinking because the inequalities defining Alexandrov spaces are in terms of the squared distance, not the distance. **Bobkov and Ledoux** recently established closely related results for the quadratic Hamilton–Jacobi equation in a finite-dimensional Banach space.

**Further applications of $T\_p$ inequalities.** Relations of $T\_p$ inequalities with the so-called **slicing problem** are discussed in the literature. These inequalities are also useful to study the **propagation of chaos** or the mean behavior of particle systems.

As already noticed before, the functional $H\_\nu$ appears in **Sanov's theorem** as the rate function for the deviations of the empirical mean of independent samples; this explains why $T\_p$ inequalities are handy tools for a quantitative study of concentration of the empirical measure associated with certain particle systems. The links with large deviation theory were further explored. If one is interested in the concentration of *time averages*, then one should replace the Kullback information $H\_\nu$ by the **Fisher information** $I\_\nu$, as was understood by **Donsker and Varadhan**. As a matter of fact, **Guillin, Léonard, Wu and Yao** have established that the functional inequality

$$\alpha\!\left(W_1(\mu, \nu)\right) \le I_\nu(\mu),$$

where $\alpha$ is an increasing function, $\alpha(0) = 0$, is equivalent to the concentration inequality

$$\mathbb{P}\!\left[\frac{1}{t} \int_0^t \varphi(X_s)\, ds > \int \varphi\, d\nu + \varepsilon\right] \le \left\lVert \frac{d\mu}{d\nu} \right\rVert_{L^2(\nu)} e^{-t\, \alpha\!\left(\varepsilon / \lVert \varphi \rVert_{\mathrm{Lip}}\right)},$$

where $(X\_s)\_{s \ge 0}$ is the symmetric diffusion process with invariant measure $\nu$, $\mu = \mathrm{law}(X\_0)$, and $\varphi$ is an arbitrary Lipschitz function.

</div>

## Chapter 23: Gradient Flows I

Take a Riemannian manifold $M$ and a function $\Phi : M \to \mathbb{R}$, which for the sake of this exposition will be assumed to be continuously differentiable. The **gradient** of $\Phi$, denoted by $\nabla \Phi$, is the vector field defined by the equation

$$d_x \Phi \cdot v = \langle \nabla_x \Phi, v \rangle_x,$$

where $v$ is an arbitrary vector in the tangent space $T\_x M$, $d\_x \Phi$ stands for the differential of $\Phi$ at $x$, and $\langle \cdot, \cdot \rangle\_x$ is the scalar product on $T\_x M$. In other words, if $(\gamma\_t)\_{-\varepsilon < t < \varepsilon}$ is a smooth path in $M$, with $\gamma\_0 = x$, then

$$\left.\frac{d}{dt}\right|_{t=0} \gamma_t = v \implies \left.\frac{d}{dt}\right|_{t=0} \Phi(\gamma_t) = \langle \nabla_x \Phi, v \rangle_x.$$

If $\lvert v \rvert$ is given, then in order to make the latter derivative as large as possible, the best choice is to take $v$ colinear to $\nabla\_x \Phi$. In that sense $\nabla\_x \Phi$ indicates the *direction in which $\Phi$ increases most rapidly*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient Flow)</span></p>

The **gradient flow** associated to $\Phi$ is the flow induced by the differential equation

$$\frac{dX}{dt} = -\operatorname{grad}_X \Phi.$$

One may think of it heuristically as a flow which *makes $\Phi$ decrease as fast as possible*. An important consequence of the definition of gradient flow is the following neat formula for the time-derivative of the energy:

$$\frac{d}{dt}\, \Phi(X(t)) = -\left\lvert \operatorname{grad}_{X(t)} \Phi \right\rvert^2.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gradient Flows and Damped Hamiltonian Systems)</span></p>

Gradient flows (as Hamiltonian flows) are everywhere in physics and mathematics. In mechanics, they often describe the behavior of damped Hamiltonian systems, in an asymptotic regime in which dissipative effects play such an important role, that the effects of forcing and dissipation compensate each other. The basic example one should think of is

$$\ddot{X} = -\lambda\, \operatorname{grad}_X \Phi - \lambda\, \dot{X}$$

(acceleration $=$ forcing $-$ friction), in the limit $\lambda \to +\infty$ (which means strong friction).

</div>

### Gradient Flows in Wasserstein Space

Around the end of the nineties, **Jordan, Kinderlehrer and Otto** made the important discovery that *a number of well-known partial differential equations can be reformulated as gradient flows in the Wasserstein space*. The most emblematic example is that of the **heat equation**,

$$\partial_t \rho = \Delta \rho,$$

say in Euclidean space for simplicity. It is classical that this equation can be seen as a gradient flow, for instance for the quadratic functional $\Phi(\rho) = \int \lvert \nabla \rho \rvert^2\, dx$ in $L^2(\mathbb{R}^n)$. But the **Jordan–Kinderlehrer–Otto formulation** describes the heat equation as a gradient flow *in the space of probability measures*, and with a natural "information-theoretical" content. In this new approach, the functional $\Phi$ is the negative of the entropy: $\Phi(\rho) = \int \rho \log \rho\, dx$.

To better understand this point of view, Otto developed a set of calculation rules which I dubbed "Otto calculus" in Chapter 15. In this chapter, the author describes in which *rigorous* sense one can say that certain equations are gradient flows in the Wasserstein space.

### Reformulations of Gradient Flows

There are several ways to reformulate gradient flows in a weak sense, so as to obtain definitions that are general (for nonsmooth energies, or nonsmooth spaces), and stable (under some limit process). They usually require a convexity-type assumption on the energy $\Phi$. Recall the definitions of $\lvert \nabla^- \Phi \rvert$ and $\nabla^- \Phi$ (or $\partial \Phi$):

$$\lvert \nabla^- \Phi(x) \rvert = \limsup_{y \to x} \frac{[\Phi(y) - \Phi(x)]_-}{d(x, y)};$$

$$\nabla^- \Phi(x) = \left\lbrace v \in T_x M;\ \forall w \in T_x M,\ \Phi\!\left(\exp_x(\varepsilon w)\right) \ge \Phi(x) + \varepsilon \langle v, w \rangle + o(\varepsilon) \right\rbrace.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 23.1</span><span class="math-callout__name">(Reformulations of Gradient Flows)</span></p>

Let $M$ be a Riemannian manifold. Let $\Lambda = \Lambda(x, v)$ be a quadratic form on $TM$, satisfying (16.7), and let $\Phi$ be a differentiable function $M \to \mathbb{R}$, $\Lambda$-convex in the sense of Proposition 16.2. Let $X : (t\_1, t\_2) \to M$ be a continuous path, and let $t \in (t\_1, t\_2)$ be a time where $X$ is differentiable. Then the following statements are equivalent:

**(i)** $\dot{X}(t) = -\operatorname{grad}\_{X(t)} \Phi$;

**(ii)** $\displaystyle \frac{\lvert \dot{X}(t) \rvert^2 + \lvert \nabla^- \Phi(X(t)) \rvert^2}{2} = -\frac{d}{dt} \Phi(X(t))$;

**(iii)** $-\dot{X}(t) \in \nabla^- \Phi(X(t))$;

**(iv)** For any $y \in M$, and any geodesic $(\gamma\_s)\_{0 \le s \le 1}$ joining $\gamma\_0 = X(t)$ to $\gamma\_1 = y$,

$$\frac{d^+}{dt}\!\left(\frac{d(X(t), y)^2}{2}\right) \le \left.\frac{d^+}{ds}\right|_{s=0} \Phi(\gamma_s) := \limsup_{s \downarrow 0} \frac{\Phi(\gamma_s) - \Phi(\gamma_0)}{s};$$

**(v)** For any $y \in M$, and any geodesic $(\gamma\_s)\_{0 \le s \le 1}$ joining $\gamma\_0 = X(t)$ to $\gamma\_1 = y$,

$$\frac{d^+}{dt}\!\left(\frac{d(X(t), y)^2}{2}\right) \le \Phi(y) - \Phi(X(t)) - \int_0^1 \Lambda(\gamma_s, \dot{\gamma}_s)\,(1 - s)\, ds;$$

**(vi)** For any $y \in M$, and any geodesic $(\gamma\_s)\_{0 \le s \le 1}$ joining $\gamma\_0 = X(t)$ to $\gamma\_1 = y$,

$$\frac{d^+}{dt}\!\left(\frac{d(X(t), y)^2}{2}\right) \le \Phi(y) - \Phi(X(t)) - \lambda[\gamma]\, \frac{d(X(t), y)^2}{2},$$

where $\lambda[\gamma]$ is defined by (16.7).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.2</span></p>

As the proof will show, the equivalence between (iii), (iv), (v) and (vi) does not require the differentiability of $\Phi$; it is sufficient that $\Phi$ be valued in $\mathbb{R} \cup \lbrace +\infty \rbrace$ and $\Phi(X(t)) < +\infty$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.3</span></p>

The most well-known case is when $\Lambda = 0$ ($\Phi$ is convex), and then (v) becomes just

$$\frac{d^+}{dt}\!\left(\frac{d(X(t), y)^2}{2}\right) \le \Phi(y) - \Phi(X(t)).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.4</span></p>

Statements (i) to (iii) do not explicitly depend on $\Lambda$, so here the assumption of $\Lambda$-convexity is not essential. But as soon as one wants to generalize Proposition 23.1 by dropping some smoothness assumptions, it might be important to know that $\Phi$ is $\Lambda$-convex for some $\Lambda$. Note that in formulations (iv) to (vi), one can always replace $\Lambda$ by $\Lambda' \le \Lambda$, and the equivalence still holds true, independently of the choice of $\Lambda'$! In particular, if $\Lambda(x, v) \ge \lambda\, \lvert v \rvert^2$ for some $\lambda \in \mathbb{R}$, i.e. when $\Phi$ is $\lambda$-convex in the sense of Definition 16.4, one may replace $\Lambda(x, v)$ by $\lambda\, \lvert v \rvert^2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.5</span></p>

If one wants to use Proposition 23.1 to characterize a curve $(X(t))$ as a gradient flow, the natural regularity assumption is that $X$ be an absolutely continuous function of $t$, in the sense of (7.5). This will imply the existence of the derivative $\dot{X}(t)$ for almost all $t$, and in addition this will guarantee that the values of $X$ are uniquely determined by $X(0)$ and the values of $\dot{X}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interest of the Different Formulations)</span></p>

Property (ii) involves speeds (norms of velocities) rather than velocities; this is interesting also in a nonsmooth setting, where the speed might be well-defined even though the velocity is not. Property (iii) has the advantage of being formulated in terms of subgradients (or subdifferentials), which are often well-defined even if $\Phi$ is not differentiable (for instance if $\Phi$ is semiconvex), and quite stable. Properties (iv) to (vi) are quite handy to study gradient flows in abstract metric spaces. As a matter of fact, formulation (iv) will be used to define gradient flows in the Wasserstein space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Proposition 23.1)</span></p>

**(i) $\Leftrightarrow$ (ii):** By the chain-rule, and Cauchy–Schwarz and Young's inequalities,

$$-\frac{d}{dt}\, \Phi(X(t)) = \left\langle -\nabla\Phi(X(t)),\, \dot{X}(t) \right\rangle \le \lvert \nabla\Phi(X(t)) \rvert\, \lvert \dot{X}(t) \rvert \le \frac{\lvert \nabla\Phi(X(t)) \rvert^2 + \lvert \dot{X}(t) \rvert^2}{2},$$

with equality if and only if $\nabla\Phi(X(t))$ and $\dot{X}(t)$ have the same norm and opposite directions.

**(i) $\Leftrightarrow$ (iii):** If $\Phi$ is differentiable then $\lvert \nabla^- \Phi(x) \rvert = \lvert \nabla \Phi(x) \rvert$ and $\nabla^- \Phi(x) = \lbrace \nabla \Phi(x) \rbrace$.

**(iii) $\Rightarrow$ (iv):** Let $y$ be given and let $\gamma$ be a geodesic path joining $\gamma(0) = X(t)$ to $\gamma(1) = y$. By the formula of first variation (7.29),

$$\frac{d^+}{dt}\!\left(\frac{d(X(t), y)^2}{2}\right) \le -\langle \dot{\gamma}(0), \dot{X}(t) \rangle_{X(t)}.$$

On the other hand, $\gamma\_s = \exp(sw)$, where $w = \dot{\gamma}(0)$, so if (iii) holds true then as $s \to 0$,

$$\frac{\Phi(\gamma_s) - \Phi(\gamma_0)}{s} \ge \langle -\dot{X}(t), \dot{\gamma}_0 \rangle + o(1).$$

Consequently, $(d^+/dt)(d(X(t), y)^2/2) \le \liminf\_{s \downarrow 0} (\Phi(\gamma\_s) - \Phi(\gamma\_0))/s$, which obviously implies (iv).

**(iv) $\Rightarrow$ (v):** Since $\Phi$ is $\Lambda$-convex, $(\Phi(\gamma\_s) - \Phi(\gamma\_0))/s \le \Phi(\gamma\_1) - \Phi(\gamma\_0) - \int\_0^1 \Lambda(\gamma\_\tau, \dot{\gamma}\_\tau)\,(1 - \tau)\, d\tau$, so (iv) implies (v).

**(v) $\Rightarrow$ (vi):** Trivial since $\int\_0^1 \Lambda(\gamma\_s, \dot{\gamma}\_s)\,(1 - s)\, ds \ge \lambda[\gamma]\, d(\gamma\_0, \gamma\_1)^2/2$.

**(vi) $\Rightarrow$ (iii):** Let $t$ be given, $w \in T\_{X(t)} M$, $y = \exp\_{X(t)}(\varepsilon w)$, and $\gamma(s) = \exp\_{X(t)}(s\varepsilon w)$. Then $(d/dt)(d(X(t), y)^2) = -\langle \dot{\gamma}(0), \dot{X}(t) \rangle = -\langle \varepsilon w, \dot{X}(t) \rangle$. So (v) implies $-\langle \varepsilon w, \dot{X}(t) \rangle \le \Phi(\exp\_{X(t)} \varepsilon w) - \Phi(X(t)) - \lambda[\gamma]\, \varepsilon^2 \lvert w \rvert^2/2$. As a consequence, $\Phi(\exp\_{X(t)} \varepsilon w) \ge \Phi(X(t)) + \varepsilon \langle w, -\dot{X}(t) \rangle + o(\varepsilon)$, which precisely means that $-\dot{X}(t) \in \nabla^- \Phi(X(t))$.

</div>

### Gradient Flows in Metric Spaces

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation 23.6</span><span class="math-callout__name">(Locally Absolutely Continuous Paths)</span></p>

Let $(\mathcal{X}, d)$ be a metric space and $T \in (0, +\infty]$. Denote by $\mathrm{AC}\_{\mathrm{loc}}((0, T); \mathcal{X})$ the set of paths $\gamma : (0, T) \to \mathcal{X}$ such that there is a measurable function $\ell : (0, T) \to \mathbb{R}\_+$ satisfying $d(\gamma\_s, \gamma\_t) \le \int\_s^t \ell(\tau)\, d\tau$ for all $s < t$ in $(0, T)$, and

$$0 < t_1 < t_2 < T \implies \int_{t_1}^{t_2} \ell(\tau)\, d\tau < +\infty.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 23.7</span><span class="math-callout__name">(Gradient Flows in a Geodesic Space)</span></p>

Let $(\mathcal{X}, d)$ be a geodesic space and let $\Phi : \mathcal{X} \to \mathbb{R} \cup \lbrace +\infty \rbrace$. Let $T \in (0, +\infty]$ and let $X \in C([0, T); \mathcal{X}) \cap \mathrm{AC}\_{\mathrm{loc}}((0, T); \mathcal{X})$. Then $X$ is said to be a **trajectory of the gradient flow** associated with the energy $\Phi$ if **(a)** $\Phi(X(t)) < +\infty$ for all $t > 0$; and **(b)** for any $y \in \mathcal{X}$ and for almost any $t > 0$, there is a geodesic $(\gamma\_s)\_{0 \le s \le 1}$ joining $\gamma\_0 = X(t)$ to $\gamma\_1 = y$, such that

$$\frac{d^+}{dt}\!\left(\frac{d(X(t), y)^2}{2}\right) \le \left.\frac{d^+}{ds}\right|_{s=0} \Phi(\gamma_s).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.8</span></p>

If $\Phi$ is $\lambda$-convex, then property (b) in the previous definition implies

$$\frac{d^+}{dt}\!\left(\frac{d(X(t), y)^2}{2}\right) \le \Phi(y) - \Phi(X(t)) - \lambda\, \frac{d(X(t), y)^2}{2}.$$

The proof is the same as for the implication (iv) $\Rightarrow$ (v) in Proposition 23.1. One could have used this inequality to define gradient flows in metric spaces, at least for $\lambda$-convex functions; but Definition 23.7 is more general.

Proposition 23.1 guarantees that the concept of abstract gradient flow coincides with the usual one when $\mathcal{X}$ is a Riemannian manifold equipped with its geodesic distance. In the sequel, Definition 23.7 will be applied in the Wasserstein space $P\_2(M)$, where $M$ is a Riemannian manifold (sometimes with additional geometric assumptions). To avoid complications the definition will in fact be used in $P\_2^{\mathrm{ac}}(M)$, that is, restricting to absolutely continuous probability measures. This might look a bit dangerous, because $\mathcal{X} = P\_2^{\mathrm{ac}}(M)$ is not complete, but after all it is a geodesic space in its own right, as a geodesically convex subset of $P\_2(M)$ (recall Theorem 8.7), and completeness is not needed.

</div>

To go on with this program, two things are needed:
- compute the (upper) derivative of the distance function;
- compute the subdifferential of a given energy functional.

### Derivative of the Wasserstein Distance

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 23.9</span><span class="math-callout__name">(Derivative of the Wasserstein Distance)</span></p>

Let $M$ be a Riemannian manifold, and $[t\_1, t\_2) \subset \mathbb{R}$. Let $(\mu\_t)$ and $(\widehat{\mu}\_t)$ be two weakly continuous curves $[t\_1, t\_2) \to P(M)$. Assume that $\mu\_t, \widehat{\mu}\_t \in P\_2^{\mathrm{ac}}(M)$ for all $t \in (t\_1, t\_2)$, and that $\mu\_t$, $\widehat{\mu}\_t$ solve the continuity equations

$$\frac{\partial \mu_t}{\partial t} + \nabla \cdot (\xi_t\, \mu_t) = 0, \qquad \frac{\partial \widehat{\mu}_t}{\partial t} + \nabla \cdot (\widehat{\xi}_t\, \widehat{\mu}_t) = 0,$$

where $\xi\_t = \xi\_t(x)$, $\widehat{\xi}\_t = \widehat{\xi}\_t(x)$ are locally Lipschitz vector fields and

$$\int_{t_1}^{t_2}\!\left(\int_M \lvert \xi_t \rvert^2\, d\mu_t + \int_M \lvert \widehat{\xi} \rvert^2\, d\widehat{\mu}_t\right) dt < +\infty.$$

Then $t \to \mu\_t$ and $t \to \widehat{\mu}\_t$ are Hölder-$1/2$ continuous and absolutely continuous. Moreover, for almost any $t \in (t\_1, t\_2)$,

$$\frac{d}{dt}\!\left(\frac{W_2(\mu_t, \widehat{\mu}_t)^2}{2}\right) = -\int_M \langle \widetilde{\nabla} \psi_t, \xi_t \rangle\, d\mu_t - \int_M \langle \widetilde{\nabla} \widehat{\psi}_t, \widehat{\xi}_t \rangle\, d\widehat{\mu}_t,$$

where $\psi\_t$, $\widehat{\psi}\_t$ are $(d^2/2)$-convex functions such that

$$\exp(\widetilde{\nabla} \psi_t)_\# \mu_t = \widehat{\mu}_t, \qquad \exp(\widetilde{\nabla} \widehat{\psi}_t)_\# \widehat{\mu}_t = \mu_t.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.10</span></p>

Recall that Theorem 10.41 gives a list of a few conditions under which the approximate gradient $\widetilde{\nabla}$ can be replaced by the usual gradient $\nabla$ in the formulas above.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.12</span></p>

For the purpose of this chapter, the superdifferentiability of the Wasserstein distance would be enough. However, the plain differentiability will be useful later on.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 23.9)</span></p>

Without loss of generality assume that $\tau = \lvert t\_1 - t\_2 \rvert$ is finite.

A crucial ingredient in the proof is the **flow** associated with the velocity fields $\xi$ and $\widehat{\xi}$. If $t$ and $s$ both belong to $[t\_1, t\_2]$, define the characteristics (or flow, or trajectory map) $T\_{t \to s} : M \to M$ associated with $\xi$ by the differential equation

$$\begin{cases} T_{t \to t}(x) = x; \\ \frac{d}{ds} T_{t \to s}(x) = \xi_s(T_{t \to s}\, x). \end{cases}$$

(If $\xi\_s(x)$ is the velocity field at time $s$ and position $x$, then $T\_{t \to s}(x)$ is the position at time $s$ of a particle which was at time $t$ at $x$ and then followed the flow.) By the formula of conservation of mass, for all $t, s \in [t\_1, t\_2]$, $\mu\_s = (T\_{t \to s})\_\# \mu\_t$.

The idea is to compose the transport $T\_{t \to s}$ with some optimal transport; this will not result in an optimal transport, but at least it will provide bounds on the Wasserstein distance.

Since $\gamma$ solves $\dot{\gamma}\_t = \xi\_t(\gamma\_t)$, $d(\gamma\_s, \gamma\_t) \le \int \lvert \xi\_\tau(\gamma\_\tau) \rvert\, d\tau$. Since $(\gamma\_s, \gamma\_t)$ is a coupling of $(\mu\_s, \mu\_t)$, it follows that

$$W_2(\mu_s, \mu_t) \le \sqrt{\lvert s - t \rvert} \sqrt{\int_s^t \left(\int \lvert \xi_\tau \rvert^2\, d\mu_\tau\right) d\tau}.$$

This shows at the same time that $t \to \mu\_t$ is Hölder-$1/2$ continuous, and that it is absolutely continuous: $W\_2(\mu\_s, \mu\_t) \le \int\_s^t \ell(\tau)\, d\tau$ with $\ell(\tau) = \frac{1}{2}\!\left(1 + \int \lvert \xi\_\tau \rvert^2\, d\mu\_\tau\right)$.

The rest of the proof is decomposed into four steps.

**Step 1:** *$t \to W\_2(\mu\_t, \sigma)$ is superdifferentiable at each Lebesgue point of $t \to \int \lvert \xi\_t \rvert^2\, d\mu\_t$.*

In this step, the path $\widehat{\mu}\_t$ will be constant and equal to some fixed $\sigma \in P\_2^{\mathrm{ac}}(M)$. Let $\psi\_t$ be a $d^2/2$-convex function such that $\exp(\widetilde{\nabla} \psi\_t)\_\# \mu\_t = \sigma$. One shall show that

$$\frac{W_2(\mu_{t+s}, \sigma)^2}{2} \le \frac{W_2(\mu_t, \sigma)^2}{2} - s \int \langle \widetilde{\nabla} \psi_t, \xi_t \rangle\, d\mu_t + o(s).$$

Let $T = \exp(\widetilde{\nabla} \widehat{\psi})$ be the optimal (Monge) transport $\sigma \to \mu\_t$. Then $W\_2(\mu\_t, \sigma)^2/2 = \frac{1}{2} \int d(x, T(x))^2\, d\sigma(x)$. For any $s > 0$ small enough, $(T\_{t \to t+s})\_\# \mu\_t = \mu\_{t+s}$; so $T\_{t \to t+s} \circ T$ is a transport $\sigma \to \mu\_{t+s}$. By definition of the Wasserstein distance, $W\_2(\mu\_{t+s}, \sigma)^2/2 \le \frac{1}{2} \int d(x, T\_{t \to t+s} \circ T(x))^2\, d\sigma(x)$. The maps $\exp(\widetilde{\nabla} \psi)$ and $\exp(\widetilde{\nabla} \widehat{\psi})$ are inverse to each other in the almost sure sense. So for $\sigma(dx)$-almost all $x$, there is a minimizing geodesic connecting $T(x)$ to $x$ with initial velocity $\widetilde{\nabla} \psi(T(x))$; then by the formula of first variation,

$$\limsup_{s \downarrow 0}\!\left[\frac{d(x, T_{t \to t+s} \circ T(x))^2 - d(x, T(x))^2}{2s}\right] \le -\langle \xi_t(T(x)),\, \widetilde{\nabla} \psi(T(x)) \rangle.$$

So if we can pass to the $\limsup$ as $s \to 0$ under the integral sign, it will follow that

$$\frac{d^+}{dt}\!\left(\frac{W_2(\mu_t, \sigma)^2}{2}\right) \le -\int_M \langle \xi_t(T(x)),\, \widetilde{\nabla} \psi(T(x)) \rangle\, d\sigma(x) = -\int \langle \xi_t(y),\, \widetilde{\nabla} \psi(y) \rangle\, d\mu_t(y),$$

and this will establish (23.7). The passage to the limit is justified by a domination argument using Fatou's lemma and the integrability conditions.

**Step 2:** *If $\xi$ grows at most linearly, then differentiability holds for all $t$.*

In this step one assumes that there are $z \in M$ and $C > 0$ such that for all $x \in M$ and $t \in (t\_1, t\_2)$, $\lvert \xi\_t(x) \rvert \le C\, (1 + d(z, x))$. Under this assumption, $t \to W\_2(\mu\_t, \sigma)^2$ is differentiable on the whole of $(t\_1, t\_2)$, and

$$\frac{d}{dt}\!\left(\frac{W_2(\mu_t, \sigma)^2}{2}\right) = -\int_M \langle \widetilde{\nabla} \psi_t, \xi_t \rangle\, d\mu_t.$$

From the flow estimates, there is a constant $C$ such that for all $y \in M$ and $t, s \in (t\_1, t\_2)$:

$$\begin{cases} d(z, T_{t \to s}(x)) \le C\, (1 + d(z, x)); \\ d(x, T_{t \to s}(x)) \le C\, \lvert s - t \rvert\, (1 + d(z, x)). \end{cases}$$

The second moment of $\mu\_t$ is bounded by a constant independent of $t$, and the Lipschitz continuity $W\_2(\mu\_t, \mu\_s) \le C\, \lvert t - s \rvert$ follows. To prove superdifferentiability of $W\_2(\mu\_t, \sigma)^2$, by Step 1 it is sufficient to check the continuity of $t \to \int \lvert \xi\_t \rvert^2\, d\mu\_t$, which follows by dominated convergence. The proof of subdifferentiability is more delicate and involves extracting subsequences of optimal transport maps and applying Fatou's lemma with careful tail estimates.

**Step 3:** *Doubling of variables.*

Now let $\xi\_t$, $\widehat{\xi}\_t$ satisfy the same assumptions as in Step 2. By Step 2, $s \to W\_2(\mu\_s, \widehat{\mu}\_t)$ and $t \to W\_2(\mu\_t, \widehat{\mu}\_s)$ are differentiable for all $s, t$. To conclude the differentiability of $t \to W\_2(\mu\_t, \widehat{\mu}\_t)$, use Lemma 23.28 in the Appendix, provided that $s \to W\_2(\mu\_s, \widehat{\mu}\_t)$ is (locally) absolutely continuous in $s$, uniformly in $t$. This will result from the triangle inequality:

$$W_2(\mu_s, \widehat{\mu}_t)^2 - W_2(\mu_{s'}, \widehat{\mu}_t)^2 \le \left[W_2(\mu_s, \sigma) + W_2(\mu_{s'}, \sigma) + 2\, W_2(\widehat{\mu}_t, \sigma)\right] W_2(\mu_s, \mu_{s'}),$$

where $\sigma$ is any arbitrary element of $P\_2(M)$. The quantity inside square brackets is bounded and the path $(\mu\_s)$ is Lipschitz in $W\_2$ distance; so in fact $W\_2(\mu\_s, \widehat{\mu}\_t)^2 - W\_2(\mu\_{s'}, \widehat{\mu}\_t)^2 \le C\, \lvert s - s' \rvert$ for some constant $C$.

**Step 4:** *Integral reformulation and restriction argument.*

In this last step the proof of Theorem 23.9 is completed for general vector fields. Let $\xi\_t$, $\widehat{\xi}\_t$ satisfy the assumptions of the theorem. Let $z$ be a fixed point in $M$; consider the increasing sequence of events $A\_k = \lbrace \sup\_t d(z, \gamma\_t) \le k \rbrace$. For $k$ large enough the event $A\_k$ has positive probability and it makes sense to condition $\gamma$ by it. Then let $\mu\_{t,k}$ be the law of this conditioned path, evaluated at time $t$: explicitly, $\mu\_{t,k} = (e\_t)\_\# \Pi\_k$, where $\Pi\_k(d\gamma) = 1\_{\gamma \in A\_k}\, \Pi(d\gamma) / \Pi[A\_k]$. Then $Z\_k := \Pi[A\_k] \uparrow 1$, $Z\_k\, \mu\_{t,k} \uparrow \mu\_t$ as $k \to \infty$.

For each $k$, $\mu\_{t,k}$ solves the *same* continuity equation as $\mu\_t$: $\partial \mu\_{t,k}/\partial t + \nabla \cdot (\xi\_t\, \mu\_{t,k}) = 0$. But by definition $\mu\_{t,k}$ is concentrated on the ball $B(z, k)$, so we may replace $\xi\_t$ by $\xi\_{t,k} = \xi\, \chi\_k$, where $\chi\_k$ is a smooth cutoff function, $0 \le \chi\_k \le 1$, $\chi\_k = 1$ on $B[z, k]$, $\chi\_k = 0$ outside of $B[z, 2k]$. Since $\xi\_{t,k}$ and $\widehat{\xi}\_{t,k}$ are compactly supported, we may apply the result of Step 3: for all $t \in (t\_1, t\_2)$,

$$\frac{d}{dt}\!\left(\frac{W_2(\mu_{t,k}, \widehat{\mu}_{t,k})^2}{2}\right) = -\int \langle \widetilde{\nabla} \psi_{t,k}, \xi_{t,k} \rangle\, d\mu_{t,k} - \int \langle \widetilde{\nabla} \widehat{\psi}_{t,k}, \widehat{\xi}_{t,k} \rangle\, d\widehat{\mu}_{t,k},$$

where $\exp(\widetilde{\nabla} \psi\_{t,k})$ and $\exp(\widetilde{\nabla} \widehat{\psi}\_{t,k})$ are the optimal transports $\mu\_{t,k} \to \widehat{\mu}\_{t,k}$ and $\widehat{\mu}\_{t,k} \to \mu\_{t,k}$. Since $t \to \mu\_{t,k}$ and $t \to \widehat{\mu}\_{t,k}$ are Lipschitz paths, also $W\_2(\mu\_{t,k}, \widehat{\mu}\_{t,k})$ is a Lipschitz function of $t$, so (23.19) integrates up to the desired formula. The passage to the limit $k \to \infty$ completes the proof.

</div>

### Subdifferential of Energy Functionals

The next problem to be addressed is the differentiation of an energy functional $U\_\nu$, along a path in the Wasserstein space $P\_2(M)$, or rather in $P\_2^{\mathrm{ac}}(M)$. This problem is easy to solve formally by means of Otto calculus, but the rigorous justification is definitely not trivial, especially when $M$ is noncompact. The proof uses Alexandrov's second differentiability theorem (Theorem 14.1), some elements of distribution theory, and many technical tricks. We denote by $W\_{\mathrm{loc}}^{1,1}(M)$ the space of functions $f$ which are locally integrable in $M$ and whose distributional gradient $\nabla f$ is defined by a locally integrable function. Recall Convention 17.10.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 23.14</span><span class="math-callout__name">(Computation of Subdifferentials in Wasserstein Space)</span></p>

Let $M$ be a Riemannian manifold, equipped with a reference measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying a curvature-dimension bound $\mathrm{CD}(K, N)$ for some $K \in \mathbb{R}$, $N \in (1, \infty]$. Let $U \in \mathcal{DC}\_N$, $p(r) = r\, U'(r) - U(r)$, let $\mu$ and $\sigma$ belong to $P\_2^{\mathrm{ac}}(M)$, and let $\rho$ be the density of $\mu$ with respect to $\nu$. Let $\psi$ be a $d^2/2$-convex function such that $T = \exp(\widetilde{\nabla} \psi)$ is the unique Monge transport $\mu \to \sigma$, and for $t \in [0, 1]$ let $\mu\_t = (\exp(t\widetilde{\nabla} \psi))\_\# \mu$. Assume that:

**(i)** $p(\rho) \in W\_{\mathrm{loc}}^{1,1}$;

**(ii)** $U\_\nu(\mu) < +\infty$;

**(iii)** $K\_{N,U} > -\infty$, where $K\_{N,U}$ is defined in (17.10).

If $M$ is noncompact, further assume that:

**(iv)** $I\_{U,\nu}(\mu) := \int \frac{\lvert \nabla p(\rho) \rvert^2}{\rho}\, d\nu < +\infty$; and

**(v)** $\mu, \sigma \in P\_p^{\mathrm{ac}}(M)$, where $p \in [2, +\infty) \cup \lbrace c \rbrace$ satisfies (17.5).

If $M$ is noncompact, $N < \infty$ and $K < 0$, reinforce (v) into:

**(v')** $\mu, \sigma \in P\_q^{\mathrm{ac}}(M)$, where $q \in ((2N)/(N-1), +\infty) \cup \lbrace c \rbrace$ satisfies $\exists\, \delta > 0$; $\int \frac{\nu(dx)}{(1 + d(x\_0, x))^{q(N-1) - 2N - \delta}} < +\infty$.

Then

$$\liminf_{t \downarrow 0}\, \frac{U_\nu(\mu_t) - U_\nu(\mu)}{t} \ge \int \langle \widetilde{\nabla} \psi, \nabla p(\rho) \rangle\, d\nu;$$

and

$$U_\nu(\sigma) \ge U_\nu(\mu) + \int \langle \widetilde{\nabla} \psi, \nabla p(\rho) \rangle\, d\nu + K_{N,U} \int_0^1\!\left(\int \lvert \widetilde{\nabla} \psi_t(x) \rvert^2\, \rho_t(x)^{1 - \frac{1}{N}}\, \nu(dx)\right)(1 - t)\, dt.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Particular Case: Displacement Convexity of $H$)</span></p>

If $N = \infty$ and $U(r) = r \log r$, Formula (23.26) becomes

$$H_\nu(\sigma) \ge H_\nu(\mu) + \int_M \langle \widetilde{\nabla} \psi, \nabla \rho \rangle\, d\nu + K\, \frac{W_2(\mu, \sigma)^2}{2}.$$

By the Cauchy–Schwarz inequality, this implies

$$H_\nu(\sigma) \ge H_\nu(\mu) - W_2(\mu, \sigma)\, \sqrt{I_\nu(\mu)} + K\, \frac{W_2(\mu, \sigma)^2}{2}.$$

In this sense (23.27) is a precise version of the **HWI inequality** appearing in Corollary 20.13.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.17</span></p>

If $K\_{N,U} = -\infty$ (i.e. $K < 0$ and $p(r)/r^{1-1/N} \to +\infty$ as $r \to \infty$) then (23.26) remains obviously true but I don't know about (23.25).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.18</span></p>

As soon as $\rho \in W\_{\mathrm{loc}}^{1,1}(M)$ we can write $\nabla p(\rho) = p'(\rho)\, \nabla \rho = \rho\, U''(\rho)\, \nabla \rho = \rho\, \nabla U'(\rho)$, so (23.25) becomes

$$\liminf_{t \downarrow 0}\, \frac{U_\nu(\mu_t) - U_\nu(\mu)}{t} \ge \int \langle \widetilde{\nabla} \psi, \nabla U'(\rho) \rangle\, d\mu,$$

which is the result that one would have formally guessed by using Otto's calculus.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 23.14)</span></p>

The complete proof is quite tricky and is divided into seven steps. Here we summarize the key ideas.

**Step 1: Computation of the $\liminf$ in the compactly supported case.** Assume $\mu$ and $\sigma$ are compactly supported, and *compute* the lower derivative:

$$\liminf_{t \downarrow 0}\, \frac{U_\nu(\mu_t) - U_\nu(\mu_0)}{t} = -\int p(\rho)\, (L\psi)\, d\nu,$$

where the function $L\psi$ is obtained from the measure $L\psi$ (understood in the sense of Alexandrov) by keeping only the absolutely continuous part (with respect to the volume measure).

The argument starts with a change of variables: $U\_\nu(\mu\_t) = \int U(\rho\_t(\exp\_x t\nabla\psi(x)))\, \mathcal{J}\_{0 \to t}(x)\, d\nu(x)$, where $\mathcal{J}\_{0 \to t}$ is the Jacobian determinant associated with the map $\exp(t\nabla\psi)$. By Theorem 14.1, for $\mu$-almost all $x$, $\det(d\_x \exp(t\nabla\psi)) = 1 + t\, \Delta\psi(x) + o(t)$ as $t \to 0$, so $\mathcal{J}\_{0 \to t}(x) = 1 + t\, (L\psi)(x) + o(t)$, where $(L\psi)(x) = \Delta\psi - \nabla V \cdot \nabla\psi$. This leads to the formula $(U\_\nu(\mu\_t) - U\_\nu(\mu))/t = \int w(t, x)\, \mu(dx)$, and passing to the limit yields (23.29).

For $K \ge 0$, $u(t, x)$ is a convex function of $t$ and $w(t, x)$ is nonincreasing as $t \downarrow 0$, so (23.33) follows from the monotone convergence theorem. For $K < 0$, a modified convexity argument with a correction term $R(t, x)$ is used; by Jensen's inequality, $R(t\_k, x)/t\_k \to 0$ as $t \to 0$, so the correction is negligible.

**Step 2: Extension of $\nabla\psi$.** The function $\psi$ might only be finite outside $\operatorname{Spt} \mu$, which might cause problems. In this step, $\psi$ is extended to a function which is finite everywhere on $M$. Let $\phi$ be the $\widetilde{c}$-transform of $\widetilde{\psi}$: then $\widetilde{\psi}(x) = \sup\_{y \in \operatorname{Spt} \sigma}\!\left(\phi(y) - d(x, y)^2/2\right)$. Since $\operatorname{Spt} \sigma$ is bounded, the extended function $\widetilde{\psi}$ is locally Lipschitz, locally semiconvex, $d^2/2$-convex, and $\nabla\widetilde{\psi}$ coincides $\mu$-almost surely with $\nabla\psi$.

**Step 3: Integration by parts.** Show that $-\int p(\rho)\, L\psi\, d\nu \ge \int \langle \nabla\psi, \nabla p(\rho) \rangle\, d\nu$, where $L\psi = \Delta\psi - \nabla V \cdot \nabla\psi$ is understood in the sense of Alexandrov (Theorem 14.1) or equivalently as the absolutely continuous part of the distribution $L\psi$. Since $\rho$ is compactly supported, $p(\rho) \in W^{1,1}(M)$. By regularization with a $C^\infty$ mollifier, construct $\zeta\_k \to p(\rho)$ in $L^1$ with $\nabla\zeta\_k \to \nabla p(\rho)$ in $L^1$. Since $\Delta\psi$ is bounded below on $W$ (where $\psi$ is Lipschitz), Fatou's lemma applies to show $\int p(\rho)\, (L\psi)\, d\nu \le \liminf \int \zeta\_k\, (L\psi)\, d\nu$.

**Step 4: Integral reformulation.** Take advantage of the displacement convexity properties of $U\_\nu$ to reformulate the differential condition $\liminf\_{t \downarrow 0} (U\_\nu(\mu\_t) - U\_\nu(\mu\_0))/t \ge \int \langle \nabla\psi, \nabla p(\rho\_0) \rangle\, d\nu$ into the integral condition

$$U_\nu(\mu_1) \ge U_\nu(\mu_0) + \int \langle \nabla\psi, \nabla p(\rho_0) \rangle\, d\nu + K_{N,U} \int_0^1\!\left(\int \rho_t(x)^{1-\frac{1}{N}} \lvert \nabla\psi_t(x) \rvert^2\, \nu(dx)\right)(1 - t)\, dt.$$

The strategy is the same as in the proof of (iv) $\Rightarrow$ (v) in Proposition 23.1. By Theorem 17.15, for any $t \in (0, 1)$, $U\_\nu(\mu\_t) \le (1 - t)\, U\_\nu(\mu\_0) + t\, U\_\nu(\mu\_1) - K\_{N,U} \int\_0^1 (\int \rho\_s^{1-1/N} \lvert \nabla\psi\_s \rvert^2\, \nu(dx))\, G(s, t)\, ds$. Subtracting $U\_\nu(\mu\_0)$ and dividing by $t$, then passing to the limit using Steps 1 and 2, establishes the integral formulation. This also completes the proof for compactly supported measures.

**Step 5: Removal of compactness assumption, for nice pressure laws.** This step extends (23.47) to the noncompact case, under some additional regularity assumptions on $U$ (namely, $p$ is assumed Lipschitz and satisfies conditions (23.52)–(23.54)). Two cases are distinguished:

- **Case 1: $\operatorname{Spt}(\mu\_1)$ is compact.** Use a modification of the standard approximation scheme: set $\rho\_{0,k} = \chi\_k\, \rho\_0 / Z\_k$ and $\mu\_{0,k} = \rho\_{0,k}\, \nu$. Apply the result of Step 4 with $\mu\_t$ replaced by $\mu\_{t,k}$ and $U$ replaced by $U\_k = U(Z\_k\, \cdot)$. Pass to the limit in all four terms of the resulting inequality. The key difficulty is the third term $\int \langle \nabla\psi\_k, \nabla p\_k(\rho\_{0,k}) \rangle\, d\nu$: use the chain-rule formula $\nabla p(\chi\_k\, \rho\_0) = p'(\chi\_k\, \rho\_0)\, (\rho\_0\, \nabla\chi\_k + \chi\_k\, \nabla\rho\_0)$, control each piece using conditions (23.52)–(23.54), and apply the dominated convergence theorem.

- **Case 2: $\operatorname{Spt}(\mu\_1)$ is not compact.** Use smooth truncation instead. Define $\mu\_{0,k}$ and $\mu\_{1,k}$ by truncation with cutoff functions $\chi\_{\ell(k)}$ and $\chi\_k$ respectively. The geodesic curves $(\mu\_{t,k})$ converge (up to subsequences) to $(\mu\_t)$ by Corollary 7.22. Apply Step 4 to the truncated measures and pass to the limit term by term. The fourth term requires the most delicate analysis, involving three cases (I, II, III) depending on $N$ and $K$: when $N = \infty$, use Proposition 17.24(i); when $N < \infty$ and $K \ge 0$, the term is nonnegative; when $N < \infty$ and $K < 0$, use a locally uniform bound on $\lvert \nabla\psi\_{t,k} \rvert$ (established via cyclical monotonicity and the "no-crossing property"), combined with a concavity argument using Theorem 29.20 to establish (23.66).

**Step 6: Extension to general pressure laws.** If $p$ does not satisfy the regularity assumptions of Step 5, write $U$ as the monotone limit of $U\_\ell$ (using Proposition 17.7), where each $U\_\ell$ satisfies the assumptions for Step 5 to go through, and $U\_\ell'' \le C\, U''$, $p\_\ell' \le C\, p'$. Apply Step 5 with $U\_\ell$ and $p\_\ell$, then pass to the limit $\ell \to \infty$. The key subtlety is showing $\int \langle \widetilde{\nabla}\psi, \nabla p\_\ell(\rho\_0) \rangle\, d\nu \to \int \langle \widetilde{\nabla}\psi, \nabla p(\rho\_0) \rangle\, d\nu$: one first checks that $\nabla p\_\ell(\rho\_0)$ converges almost surely to $\nabla p(\rho\_0)$, and then uses domination by an integrable function via $\lvert \widetilde{\nabla}\psi \rvert\, \lvert \nabla p(\rho\_0) \rvert \le W\_2(\mu\_0, \mu\_1)\, \sqrt{I\_{U,\nu}(\mu\_0)}$.

**Step 7: Differential reformulation and conclusion.** Convert back from the integral formulation to the desired differential inequality, distinguishing three cases:

- *Case (I): $N < \infty$ and $K \ge 0$.* Start from $U\_\nu(\mu\_1) \ge U\_\nu(\mu\_0) + \int \langle \widetilde{\nabla}\psi, \nabla p(\rho\_0) \rangle\, d\nu$, which is a priori weaker than (23.26). Improve this by applying (23.76) with $\mu\_t$ replacing $\mu\_1$ and $t\psi$ replacing $\psi$, then passing to the $\liminf$ as $t \to 0$, yielding (23.77). Then use Theorem 17.15 combined with (23.77) to recover (23.25).

- *Case (II): $N = \infty$ and $K < 0$.* Start from $U\_\nu(\mu\_1) \ge U\_\nu(\mu\_0) + \int \langle \widetilde{\nabla}\psi, \nabla p(\rho\_0) \rangle\, d\nu + K\, W\_2(\mu\_0, \mu\_1)^2/2$. Substitute $\mu\_1$ for $\mu\_t$ and $\psi$ by $t\psi$ to get $U\_\nu(\mu\_t) \ge U\_\nu(\mu\_0) + t \int \langle \widetilde{\nabla}\psi, \nabla p(\rho\_0) \rangle\, d\nu + t^2\, K\, W\_2(\mu\_0, \mu\_1)^2/2$. Divide by $t$ and pass to the $\liminf$.

- *Case (III): $N < \infty$ and $K < 0$.* Start from the full formula with the integral correction term. Substitute $\mu\_1$ by $\mu\_t$, $\rho\_s$ by $\rho\_{st}$, and $\psi\_s$ by $t\psi\_{st}$, yielding $U\_\nu(\mu\_t) \ge U\_\nu(\mu\_0) + t \int \langle \widetilde{\nabla}\psi, \nabla p(\rho\_0) \rangle\, d\nu + t^2\, (K\_{N,U}/2)\, (\sup\_{0 \le \tau \le t} \int \rho\_\tau^{1-1/N} \lvert \widetilde{\nabla}\psi\_\tau \rvert^2\, \nu(dx))$. By Proposition 17.24(ii), the expression inside brackets is uniformly bounded for $t \le 1/2$, so $U\_\nu(\mu\_t) \ge U\_\nu(\mu\_0) + t \int \langle \widetilde{\nabla}\psi, \nabla p(\rho\_0) \rangle\, d\nu - O(t^2)$, and we can conclude as before.

</div>

### Diffusion Equations as Gradient Flows

Now we are equipped to identify certain nonlinear diffusive equations as gradient flows in the Wasserstein space.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 23.19</span><span class="math-callout__name">(Diffusion Equations as Gradient Flows in the Wasserstein Space)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference measure $\nu = e^{-V}\, \mathrm{vol}$, $V \in C^2(M)$, satisfying a $\mathrm{CD}(K, N)$ curvature-dimension bound for some $K \in \mathbb{R}$, $N \in (1, \infty]$. Let $L = \Delta - \nabla V \cdot \nabla$. Let $U$ be a nonlinearity in $\mathcal{DC}\_N$, such that $U \in C^3(0, +\infty)$; and let $p(r) = r\, U'(r) - U(r)$. Let $\rho = \rho\_t(x)$ be a smooth ($C^1$ in $t$, $C^2$ in $x$) positive solution of the partial differential equation

$$\frac{\partial \rho_t}{\partial t} = L\, p(\rho_t),$$

and let $\mu\_t = \rho\_t\, \nu$. Assume that $U\_\nu(\mu\_t) < +\infty$ for all $t > 0$; and that for all $0 < t\_1 < t\_2$, $\int\_{t\_1}^{t\_2} I\_{U,\nu}(\mu\_t)\, dt < +\infty$.

- If $K < 0$, further assume that $p(r) = O(r^{1-1/N})$ as $r \to \infty$.
- If $M$ is noncompact, further assume, with the same notation as in Theorem 23.14, that $\mu\_t \in P\_p^{\mathrm{ac}}(M)$.
- If $M$ is noncompact, $K < 0$ and $N < \infty$, reinforce the latter assumption into $\mu\_t \in P\_q^{\mathrm{ac}}(M)$, where $q$ is as in Theorem 23.14.

Then $(\mu\_t)\_{t \ge 0}$ is a trajectory of the gradient flow associated with the energy functional $U\_\nu$ in $P\_2^{\mathrm{ac}}(M)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Meaning of Theorem 23.19)</span></p>

This theorem gives a rigorous meaning to the informal statement that "the diffusion equation $\partial\_t \rho = L\, p(\rho)$ is the gradient flow of $U\_\nu$ in Wasserstein space." The most important particular case is the **heat equation** $\partial\_t \rho = \Delta \rho - \nabla V \cdot \nabla \rho$ (i.e. $U(r) = r \log r$, $p(r) = r$), which is the gradient flow of the **Boltzmann entropy** $H\_\nu(\mu) = \int \rho \log \rho\, d\nu$.

To verify Definition 23.7, one needs to check conditions (a) and (b). Condition (a) follows from the assumption $U\_\nu(\mu\_t) < +\infty$. For condition (b), let $\sigma \in P\_2^{\mathrm{ac}}(M)$ be arbitrary and let $\psi$ be the $d^2/2$-convex function realizing the optimal transport $\mu\_t \to \sigma$. The continuity equation $\partial\_t \mu\_t + \nabla \cdot (\xi\_t\, \mu\_t) = 0$ holds with velocity field $\xi\_t = -\nabla p(\rho\_t)/\rho\_t$ (which follows from $\partial\_t \rho = L\, p(\rho)$). Combining Theorem 23.9 (derivative of $W\_2$) with Theorem 23.14 (subdifferential of $U\_\nu$) yields the desired gradient flow inequality.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.20</span></p>

If $(\rho\_t)$ is reasonably well-behaved at infinity (a fortiori if $M$ is compact), then $t \to U\_\nu(\mu\_t)$ is nonincreasing (see Theorem 24.2(ii) in the next chapter). Thus the assumption $U\_\nu(\mu\_t) < +\infty$ is satisfied as soon as $U\_\nu(\mu\_0) < +\infty$. However, it is interesting to cover also cases where $U\_\nu(\mu\_0) = +\infty$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.21</span></p>

The finiteness of $I\_{U,\nu}(\mu\_t) = \int \rho\_t\, \lvert \nabla U'(\rho\_t) \rvert^2\, d\nu$ is a reinforcement of the condition $p(\rho) \in W\_{\mathrm{loc}}^{1,1}(M)$, since $\int \lvert \nabla p(\rho) \rvert\, d\nu \le \sqrt{I\_{U,\nu}(\mu\_t)}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 23.22</span></p>

- If $M$ is compact, any smooth positive solution of $\partial\_t \rho = \Delta \rho$ can be seen as a trajectory of the gradient flow associated with the energy $H(\mu) = \int \rho \log \rho$.
- Similarly, any smooth positive solution of $\partial\_t \rho = \Delta \rho + \nabla \cdot (\rho\, \nabla V)$ can be seen as a trajectory of the gradient flow associated with the energy $F(\mu) = \int \rho \log \rho + \int \rho\, V$. (With respect to the previous example, this amounts to changing the reference measure $\mathrm{vol}$ into $e^{-V}\, \mathrm{vol}$.)
- If $M$ has dimension $n$, any smooth positive solution of $\partial\_t \rho = \Delta \rho^m$, $m \ge 1 - 1/n$, can be seen as a trajectory of the gradient flow associated with the energy $E(\mu) = (m - 1)^{-1} \int \rho^m$.
- All these statements can be generalized to noncompact manifolds, under adequate global smoothness and decay assumptions. For instance, any smooth positive solution of $\partial\_t \rho = \Delta \rho$ in $\mathbb{R}^n$, with $\int \rho\_0(x)\, \lvert x \rvert^2\, dx < +\infty$, is a trajectory of the gradient flow associated with the $H$ functional.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 23.19)</span></p>

First note that the assumptions of the theorem imply $K\_{N,U} > -\infty$.

Because $U$ is $C^3$ on $(0, +\infty)$, the function $U'(\rho)$ is $C^2$, so $\xi\_t(x) := -\nabla U'(\rho\_t(x))$ is a $C^1$ vector field. Then (23.79) can be rewritten as $\partial \mu\_t / \partial t + \nabla \cdot (\xi\_t\, \mu\_t) = 0$. Let $\sigma \in P\_2^{\mathrm{ac}}(M)$. By Theorem 23.9, the definition of $\xi$ and the identity $\rho\, U''(\rho) = p'(\rho)$, for almost any $t$,

$$\frac{d}{dt}\!\left(\frac{W_2(\mu_t, \sigma)^2}{2}\right) = -\int \langle \widetilde{\nabla} \psi_t, \xi_t \rangle\, d\mu_t = \int \langle \widetilde{\nabla} \psi_t, \nabla U'(\rho_t) \rangle\, d\mu_t = \int \langle \widetilde{\nabla} \psi_t, \nabla p(\rho_t) \rangle\, d\nu,$$

where $\exp(\widetilde{\nabla} \psi\_t)$ is the Monge transport $\mu\_t \to \sigma$.

Let $(\mu^{(s)})$ be the displacement interpolation joining $\mu^{(0)} = \mu\_t$ to $\mu^{(1)} = \sigma$. By Theorem 23.14,

$$\liminf_{s \downarrow 0}\, \frac{U_\nu(\mu^{(s)}) - U_\nu(\mu^{(0)})}{s} \ge \int \langle \widetilde{\nabla} \psi_t, \nabla p(\rho_t) \rangle\, d\nu.$$

The combination of these two formulas implies

$$\frac{d^+}{dt}\!\left(\frac{W_2(\mu_t, \sigma)^2}{2}\right) \le \limsup_{s \downarrow 0}\, \frac{U_\nu(\mu^{(s)}) - U_\nu(\mu^{(0)})}{s},$$

and the conclusion follows from Definition 23.7.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 23.23</span><span class="math-callout__name">(Heat Equation as a Gradient Flow)</span></p>

Let $M$ be a compact Riemannian manifold curvature, let $V \in C^2(M)$, and let $L = \Delta - \nabla V \cdot \nabla$. Let $\mu\_0 \in P\_2(M)$, and let $\mu\_t = \rho\_t\, \nu$ solve

$$\frac{\partial \rho_t}{\partial t} = L\, \rho_t.$$

Then $(\mu\_t)\_{t \ge 0}$ is a trajectory of the gradient flow associated with the energy functional

$$H_\nu(\mu) = \int \rho \log \rho\, d\nu, \qquad \mu = \rho\, \nu$$

in the Wasserstein space $P\_2^{\mathrm{ac}}(M)$.

In particular, the gradient flow associated with $H\_{\mathrm{vol}}$ is the standard heat equation $\partial \rho / \partial t = \Delta \rho$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.25</span></p>

The heat equation can be seen as a gradient flow in various ways. For instance, take the basic heat equation in $\mathbb{R}^n$, in the form $\partial\_t u = \Delta u$: then it can be interpreted as the gradient flow of the functional $E(u) = (1/2) \int \lvert \nabla u \rvert^2$ for the usual Hilbert structure imposed by the $L^2$ norm; or as the gradient flow of the functional $E(u) = \int u^2$ for the Hilbert structure induced by the $H^{-1}$ norm (say on the subspace $\int u = 0$). But the interesting new feature coming from Theorem 23.19 is that now the heat equation can be seen as the gradient flow of a nice functional which has statistical (or thermodynamical) meaning; and in such a way that it is naturally set in the space of probability measures.

</div>

### Stability

A good point of our weak formulation of gradient flows is that it comes with stability estimates almost for free.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 23.26</span><span class="math-callout__name">(Stability of Gradient Flows in the Wasserstein Space)</span></p>

Let $\mu\_t$, $\widehat{\mu}\_t$ be two solutions of (23.79), satisfying the assumptions of Theorem 23.19 with either $K \ge 0$ or $N = \infty$. Let $\lambda = K\_{\infty, U}$ if $N = \infty$; and $\lambda = 0$ if $N < \infty$ and $K \ge 0$. Then, for all $t \ge 0$,

$$W_2(\mu_t, \widehat{\mu}_t) \le e^{-\lambda t}\, W_2(\mu_0, \widehat{\mu}_0).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 23.26)</span></p>

By Theorem 23.9, for almost any $t$,

$$\frac{d}{dt}\!\left(\frac{W_2(\mu_t, \widehat{\mu}_t)^2}{2}\right) = \int \langle \widetilde{\nabla} \psi_t, \nabla U'(\rho_t) \rangle\, d\mu_t + \int \langle \widetilde{\nabla} \widehat{\psi}_t, \nabla U'(\widehat{\rho}_t) \rangle\, d\widehat{\mu}_t,$$

where $\exp(\widetilde{\nabla} \psi\_t)$ (resp. $\exp(\widetilde{\nabla} \widehat{\psi}\_t)$) is the optimal transport $\mu\_t \to \widehat{\mu}\_t$ (resp. $\widehat{\mu}\_t \to \mu\_t$).

By the chain-rule and Theorem 23.14,

$$\int \langle \widetilde{\nabla} \psi_t, \nabla U'(\rho_t) \rangle\, d\mu_t = \int \langle \widetilde{\nabla} \psi_t, \nabla p(\rho_t) \rangle\, d\nu \le U_\nu(\widehat{\mu}_t) - U_\nu(\mu_t) - \lambda\, \frac{W_2(\mu_t, \widehat{\mu}_t)^2}{2}.$$

Similarly, $\int \langle \widetilde{\nabla} \widehat{\psi}\_t, \nabla U'(\widehat{\rho}\_t) \rangle\, d\widehat{\mu}\_t \le U\_\nu(\mu\_t) - U\_\nu(\widehat{\mu}\_t) - \lambda\, W\_2(\widehat{\mu}\_t, \mu\_t)^2/2$.

Combining these three inequalities: $(d/dt)(W\_2(\mu\_t, \widehat{\mu}\_t)^2/2) \le -2\lambda\, (W\_2(\mu\_t, \widehat{\mu}\_t)^2/2)$. Then the result follows from **Gronwall's lemma**.

</div>

### General Theory and Time-Discretization

There is a general theory of gradient flows in metric spaces, based for instance on Definition 23.7, or other variants appearing in Proposition 23.1. Motivations for these developments come from both pure and applied mathematics. This theory was pushed to a high degree of sophistication by many researchers, in particular **De Giorgi** and his school. A key role is played by discrete-time **approximation schemes**, the simplest of which can be stated as follows:

1. Choose your initial datum $X\_0$;
2. Choose a time step $\tau$, which in the end will decrease to 0;
3. Let $X\_1^{(\tau)}$ be a minimizer of $X \longmapsto \Phi(X) + \frac{d(X\_0, X)^2}{2\tau}$; then define inductively $X\_{k+1}^{(\tau)}$ as a minimizer of $X \longmapsto \Phi(X) + \frac{d(X\_k^{(\tau)}, X)^2}{2\tau}$.
4. Pass to the limit in $X\_k^{(\tau)}$ as $\tau \to 0$, $k\tau \to t$, hopefully recover a function $X(t)$ which is the value of the gradient flow at time $t$.

Such schemes sometimes provide an excellent way to construct the gradient flow, and they may be useful in numerical simulations. They also give a more precise formulation of the statement according to which gradient flows make the energy decrease "as fast as possible".

The time-discretization procedure also suggests a better understanding of the gradient flow in Wasserstein distance. Consider, as in Theorem 23.19, the partial differential equation $\partial \rho / \partial t = L\, p(\rho)$. Suppose you know the density $\rho(t)$ at some time $t$, and look for the density $\rho(t + dt)$ at a later time, where $dt$ is infinitesimally small. To do this, minimize the quantity

$$U_\nu(\mu_{t+dt}) - U_\nu(\mu_t) + \frac{W_2(\mu_t, \mu_{t+dt})^2}{2\, dt}.$$

By using the interpretation of the Wasserstein distance between infinitesimally close probability measures, this can also be rewritten as

$$\frac{W_2(\mu_t, \mu_{t+dt})^2}{dt} \simeq \inf\!\left\lbrace \int \lvert v \rvert^2\, d\mu_t;\ \frac{\partial \mu}{\partial t} + \nabla \cdot (\mu v) = 0 \right\rbrace.$$

To go from $\mu(t)$ to $\mu(t + dt)$, what you have to do is *find a velocity field $v$ inducing an infinitesimal variation $d\mu = -\nabla \cdot (\mu v)\, dt$*, so as to minimize the infinitesimal quantity

$$dU_\nu + K\, dt,$$

where $U\_\nu(\mu) = \int U(\rho)\, d\nu$, and $K$ is the kinetic energy $(1/2) \int \lvert v \rvert^2\, d\mu$ (so $K\, dt$ is the infinitesimal action). For the heat equation $\partial \rho / \partial t = \Delta \rho$, $\nu = \mathrm{vol}$, $U\_\nu(\mu) = \int \rho \log \rho\, d\nu$, we are back to the example discussed at the beginning of this chapter, and (23.85) can be rewritten as an "infinitesimal variation of free energy", $K\, dt - dS$, with $S$ standing for the entropy.

There is an important moral here: Behind many *nonequilibrium* equations of statistical mechanics, there is a variational principle involving entropy and energy, or functionals alike — just as in equilibrium statistical mechanics.

### Appendix: A Lemma about Doubling Variables

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 23.28</span><span class="math-callout__name">(Differentiation through Doubling of Variables)</span></p>

Let $F = F(s, t)$ be a function $[0, T] \times [0, T] \to \mathbb{R}$, locally absolutely continuous in $s$, uniformly in $t$; and locally absolutely continuous in $t$, uniformly in $s$. Then $t \to F(t, t)$ is absolutely continuous, and for almost all $t\_0$,

$$\left.\frac{d}{dt}\right|_{t=t_0} F(t, t) \le \limsup_{t \downarrow t_0}\!\left(\frac{F(t, t_0) - F(t_0, t_0)}{t - t_0}\right) + \limsup_{t \downarrow t_0}\!\left(\frac{F(t_0, t) - F(t_0, t_0)}{t - t_0}\right).$$

If moreover $F(t\_0, \cdot)$ and $F(\cdot, t\_0)$ are differentiable at all times, for almost all $t\_0$, then the inequality can be reinforced into the equality

$$\left.\frac{d}{dt}\right|_{t=t_0} F(t, t) = \left.\frac{d}{dt}\right|_{t=t_0} F(t, t_0) + \left.\frac{d}{dt}\right|_{t=t_0} F(t_0, t).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 23.29</span></p>

Lemma 23.28 does not allow us to conclude the equality if it is only known that for any $t\_0$, $F(t, t\_0)$ and $F(t\_0, t)$ are differentiable almost everywhere as functions of $t$. Indeed, it might be a priori that differentiability fails precisely at $t = t\_0$, for all $t\_0$.

</div>

### Bibliographical Notes on Chapter 23

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

Historically, the development of the theory of abstract gradient flows was initiated by **De Giorgi** and coworkers on the basis of the time-discretized variational scheme; and by **Bénilan** on the basis of the variational inequalities involving the square distance, as in Proposition 23.1(iv)–(vi). The latter approach has the advantage of incorporating stability and uniqueness as a built-in feature, while the former is more efficient in establishing existence. **Bénilan** introduced his method in the setting of Banach spaces, but it applies just as well to abstract metric spaces. Both approaches work in the Wasserstein space. **De Giorgi** also introduced the formulation in Proposition 23.1(ii), which is an alternative "intrinsic" definition for gradient flows in metric spaces.

Currently, *the* reference for abstract gradient flows is the monograph by **Ambrosio, Gigli and Savaré**. More than half of the book is devoted to gradient flows in the space of probability measures on $\mathbb{R}^n$ (or a separable Hilbert space). The results presented in this chapter extend some of the results to $P\_2(M)$, where $M$ is a Riemannian manifold, sometimes at the price of less precise conclusions. In another direction of generalization, **Fang, Shao and Sturm** have considered gradient flows in $P\_2(\mathcal{W})$, where $\mathcal{W}$ is an abstract Wiener space.

Other treatments of gradient flows in nonsmooth structures, under various curvature assumptions, are due to **Perelman and Petrunin**, **Lytchak**, **Ohta** and **Savaré**; the first two references are concerned with Alexandrov spaces, while the latter two deal with so-called 2-uniform spaces. The assumption of 2-uniform smoothness is relevant for optimal transport, since the Wasserstein space over a Riemannian manifold is not an Alexandrov space in general (except for nonnegative curvature).

The differentiability of the Wasserstein distance in $P\_2^{\mathrm{ac}}(\mathbb{R}^n)$, and in fact in $P\_p^{\mathrm{ac}}(\mathbb{R}^n)$ (for $1 < p < \infty$), is proven in **Ambrosio, Gigli and Savaré**. The assumption of absolute continuity of the probability measures is not crucial for the superdifferentiability (actually there is no such assumption for that case). For the subdifferentiability, this assumption is only used to guarantee the uniqueness of the transference plan.

There is a lot to say about the genesis of **Theorem 23.14**, which can be considered as a refinement of Theorem 20.1. The exact computation of Step 1 appears in the literature for particular functions $U$, and for general functions $U$ in **Sturm**; all these references only consider $M = \mathbb{R}^n$. The procedure of extension of $\nabla\psi$ (Step 2) appears in **McCann** (in the particular case of convex functions). The integration by parts of Step 3 appears in many papers.

It is interesting to compare the two strategies used in the extension from compact to noncompact situations, in Theorem 17.15 on the one hand, and in Theorem 23.14 on the other. In the former case, one could use the standard approximation scheme of Proposition 13.2, with an excellent control of the displacement interpolation and the optimal transport. But for Theorem 23.14, this seems to be impossible because of the need to control the smoothness of the approximation of $\rho\_0$; as a consequence, passing to the limit is more delicate.

The interpretation of the linear **Fokker–Planck equation** $\partial\_t \rho = \Delta \rho + \nabla \cdot (\rho\, \nabla V)$ as the limit of a discretized scheme goes back to the pioneering work of **Jordan, Kinderlehrer and Otto**. In that sense the Fokker–Planck equation can be considered as the abstract gradient flow corresponding to the free energy $\Phi(\rho) = \int \rho \log \rho + \int \rho\, V$. The proof is based on the three main estimates which are more or less at the basis of the whole theory of abstract gradient flows: If $\tau$ is the time step, and $X\_k^{(\tau)}$ the position at step $k$ of the discretized system, then $\Phi(X\_n^{(\tau)}) = O(1)$; $\sum d(X\_j^{(\tau)}, X\_{j+1}^{(\tau)})^2/(2\tau) = O(1)$; $\tau \sum \lVert \operatorname{grad} \Phi(X\_j^{(\tau)}) \rVert^2 = O(1)$.

**Otto** applied the same method to various classes of nonlinear diffusion equations, including porous medium and fast diffusion equations, and parabolic $p$-Laplace type equations, but also more exotic models. In his work about porous medium equations, Otto also made two important conceptual contributions: First, he introduced the abstract formalism allowing him to interpret these equations as gradient flows, directly at the continuous level (without going through the time-discretization). Secondly, he showed that certain features of the porous medium equations (qualitative behavior, rates of convergence to equilibrium) were best seen via the new gradient flow interpretation. Otto's approach was developed by various authors, including **Carrillo, McCann and myself**, **Ambrosio, Gigli and Savaré**, and others. As an example of recent application, **Carrillo and Calvez** applied the same methodology to a one-dimensional variant of the **Keller–Segel chemotaxis** model.

The setting adopted in the literature is the following: Let $E$ denote an energy functional of the form

$$E(\mu) = \int_{\mathbb{R}^n} U(\rho(x))\, dx + \int_{\mathbb{R}^n} V(x)\, d\mu(x) + \frac{1}{2} \int_{\mathbb{R}^n \times \mathbb{R}^n} W(x - y)\, d\mu(x)\, d\mu(y),$$

where $\rho$ is the density of $\mu$, and $U(0) = 0$; then under certain regularity assumptions, the associated gradient flow with respect to the 2-Wasserstein distance $W\_2$ is

$$\frac{\partial \rho}{\partial t} = \Delta p(\rho) + \nabla \cdot (\rho\, \nabla V) + \nabla \cdot (\rho\, \nabla(\rho * W)),$$

where as usual $p(r) = r\, U'(r) - U(r)$. (When $p(r) = r$, the above equation is a special case of **McKean–Vlasov equation**.) Such equations arise in a number of physical models. As an interesting particular case, the logarithmic interaction in dimension 2 gives rise to a form of the **Keller–Segel model** for chemotaxis.

Other interesting gradient flows are obtained by choosing for the energy functional:

- the **Fisher information** $I(\mu) = \int \lvert \nabla \rho \rvert^2 / \rho$; then the resulting fourth-order, nonlinear partial differential equation is a **quantum drift-diffusion equation**, which also appears in the modeling of interfaces in spin systems. The gradient flow interpretation of this equation was recently studied rigorously by **Gianazza, Savaré and Toscani**.

</div>

## Chapter 24: Gradient Flows II — Qualitative Properties

Consider a Riemannian manifold $M$, equipped with a reference measure $\nu = e^{-V}\, \mathrm{vol}$, and a partial differential equation of the form

$$\frac{\partial \rho}{\partial t} = L\, p(\rho),$$

where $p(r) = r\, U'(r) - U(r)$, $U$ is a given nonlinearity, the unknown $\rho = \rho(t, x)$ is a probability density on $M$ and $L = \Delta - \nabla V \cdot \nabla$.

Theorem 23.19 provides an interpretation of (24.1) as a gradient flow in the Wasserstein space $P\_2(M)$. What do we gain from that information? A first possible answer is a new physical insight. Another one is a set of recipes and estimates associated with gradient flows; this is what shall be illustrated in this chapter.

Conventions used throughout:
- $M$ is a complete Riemannian manifold, $d$ its geodesic distance and $\mathrm{vol}$ its volume;
- $\nu = e^{-V}\, \mathrm{vol}$ is a reference measure on $M$;
- $L = \Delta - \nabla V \cdot \nabla$ is a linear differential operator admitting $\nu$ as invariant measure;
- $U$ is a convex nonlinearity with $U(0) = 0$; typically $U$ will belong to some $\mathcal{DC}\_N$ class;
- $p(r) = r\, U'(r) - U(r)$ is the pressure function associated to $U$;
- $\mu\_t = \rho\_t\, \nu$ is the solution of the PDE $\partial\_t \rho\_t = L\, p(\rho\_t)$;
- $U\_\nu(\mu) = \int U(\rho)\, d\nu$; $\quad I\_{U,\nu}(\mu) = \int \rho\, \lvert \nabla U'(\rho) \rvert^2\, d\nu = \int \frac{\lvert \nabla p(\rho) \rvert^2}{\rho}\, d\nu$.

### Calculation Rules

Having put equation (24.1) in gradient flow form, one may use Otto's calculus to shortcut certain formal computations, and quickly get relevant results, without risks of computational errors. When it comes to rigorous justification, regularity issues should be addressed. For the most important of these gradient flows, such as the heat, Fokker–Planck or porous medium equations, these regularity issues are nowadays under good control.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 24.1</span></p>

Consider a power law nonlinearity $U(r) = r^m$, $m > 0$. For $m > 1$ the resulting equation (24.1) is called a **porous medium equation**, and for $m < 1$ a **fast diffusion equation**. These equations are usually studied under the restriction $m > 1 - (2/n)$, because for $m \le 1 - (2/n)$ the solution might fail to exist (there is in general loss of mass at infinity in finite time, or even in no time, at least if $M = \mathbb{R}^n$).

If $M$ is compact and $\rho\_0$ is positive, then there is a unique $C^\infty$, positive solution. For $m > 1$, if $\rho\_0$ vanishes somewhere, the solution in general fails to have $C^\infty$ regularity at the boundary of the support of $\rho$. For $m < 1$, adequate decay conditions at infinity are needed.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 24.2</span><span class="math-callout__name">(Computations for Gradient Flow Diffusion Equations)</span></p>

Let $\rho = \rho(t, x)$ be a solution of (24.1) defined and continuous on $[0, T) \times M$. Further, let $A$ be a convex nonlinearity, $C^2$ on $(0, +\infty)$. Assume that:

**(a)** $\rho$ is bounded and positive on $[0, \theta) \times M$, for any $\theta < T$;

**(b)** $\rho$ is $C^3$ in the $x$ variable and $C^1$ in the $t$ variable on $(0, T) \times M$;

**(c)** $U$ is $C^4$ on $(0, T) \times M$;

**(d)** $V$ is $C^4$ on $M$;

**(e)** For any $t > 0$, $\exists\, \delta > 0$: $\sup\_{\lvert s - t \rvert < \delta} \frac{1}{\lvert t - s \rvert}\!\left(\lvert \rho\_t - \rho\_s \rvert + \lvert U(\rho\_t) - U(\rho\_s) \rvert + \lvert L\, U'(\rho\_t)\, p(\rho\_t) - L\, U'(\rho\_s)\, p(\rho\_s) \rvert\right) \in L^1(d\nu)$;

**(f)** $\rho$, $p(\rho)$, $Lp(\rho)$, $p\_2(\rho)$, $\nabla p\_2(\rho)$, $U'(\rho)$, $\nabla U'(\rho)$, $LU'(\rho)$, $\nabla LU'(\rho)$, $L\lvert \nabla U'(\rho) \rvert^2$, $L(\nabla U'(\rho)\, \nabla LU'(\rho))$ and $e^{-V}$ satisfy adequate growth/decay conditions at infinity.

Then the following formulas hold true:

**(i)** $\forall t > 0$, $\quad \frac{d}{dt} \int A(\rho\_t)\, d\nu = -\int p'(\rho\_t)\, A''(\rho\_t)\, \lvert \nabla \rho\_t \rvert^2\, d\nu$;

**(ii)** $\forall t > 0$, $\quad \frac{d}{dt}\, U\_\nu(\mu\_t) = -I\_{U,\nu}(\mu\_t)$;

**(iii)** $\forall t > 0$,

$$\frac{d}{dt}\, I_{U,\nu}(\mu_t) = -2 \int_M\!\left[\lVert \nabla^2 U'(\rho_t) \rVert_{\mathrm{HS}}^2 + \left(\mathrm{Ric} + \nabla^2 V\right)(\nabla U'(\rho_t))\right] p(\rho_t)\, d\nu + \int_M \left(L\, U'(\rho_t)\right)^2 p_2(\rho_t)\, d\nu;$$

**(iv)** $\forall \sigma \in P\_2^{\mathrm{ac}}(M)$, $\quad \left\lvert \frac{d}{dt}\, W\_2(\sigma, \mu\_t) \right\rvert \le \sqrt{I\_{U,\nu}(\mu\_t)}$ for almost all $t > 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Particular Case 24.3: $U(r) = r \log r$)</span></p>

When $U(r) = r \log r$, Formula (ii) becomes a famous identity: *the Fisher information is the time-derivative of the entropy along the heat semigroup*. (What I call entropy is not $H\_\nu$ but $-H\_\nu$; this agrees with the physicists' convention.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Formal Proof of Theorem 24.2)</span></p>

By Formula 15.2 (Otto calculus),

$$\frac{d}{dt} \int A(\rho_t)\, d\nu = -\langle \operatorname{grad}_{\mu_t} A_\nu,\, \operatorname{grad}_{\mu_t} U_\nu \rangle = -\int \rho_t\, \nabla A'(\rho_t) \cdot \nabla U'(\rho_t)\, d\nu = -\int p'(\rho_t)\, A''(\rho_t)\, \lvert \nabla \rho_t \rvert^2\, d\nu.$$

This leads to formula (i). The choice $A = U$ gives

$$\frac{d}{dt} \int U(\rho_t)\, d\nu = -\lVert \operatorname{grad}_{\mu_t} U_\nu \rVert^2 = -\int \rho_t\, \lvert \nabla U'(\rho_t) \rvert^2\, d\nu = -I_{U,\nu}(\mu_t),$$

which is (ii). Next, we can differentiate the previous expression once again along the gradient flow $\dot{\mu} = -\operatorname{grad} U\_\nu(\mu)$:

$$\frac{d}{dt} \lVert \operatorname{grad}_{\mu_t} U_\nu \rVert^2 = -2\, \langle \operatorname{Hess}_{\mu_t} U_\nu \cdot \operatorname{grad}_{\mu_t} U_\nu,\, \operatorname{grad}_{\mu_t} U_\nu \rangle,$$

and then (iii) follows from Formula 15.7. As for (iv), this is just a particular case of the general formula $\lvert (d/dt)\, d(X\_0, \gamma(t)) \rvert \le \lvert \dot{\gamma}(t) \rvert\_{\gamma(t)}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rigorous Proof of Theorem 24.2)</span></p>

A crucial observation is that (24.1) can be rewritten $\partial\_t \rho = \nabla\_\nu \cdot (\rho\_t\, \nabla U'(\rho\_t))$, where $\nabla\_\nu \cdot$ stands for the negative of the adjoint of the gradient operator in $L^2(\nu)$. (Explicitly: $\nabla\_\nu \cdot u = \nabla \cdot u - \nabla V \cdot u$.) Then the proofs of (i) and (ii) are obtained by just repeating the arguments by which Formula 15.2 was established. This is a succession of differentiations under the integral symbol, chain-rules and integrations by parts:

$$\frac{d}{dt} \int A(\rho_t)\, d\nu = \int A'(\rho_t)\, (\partial_t \rho_t)\, d\nu = \int A'(\rho_t)\, \nabla_\nu \cdot (\rho_t\, \nabla U'(\rho_t))\, d\nu = -\int \nabla A'(\rho_t)\, \rho_t\, \nabla U'(\rho_t)\, d\nu,$$

and then the rest of the computation is the same as before.

The justification of (iii) is more tricky. First write $-\int \rho\, \lvert \nabla U'(\rho) \rvert^2\, d\nu = \int U'(\rho)\, Lp(\rho)\, d\nu = \int LU'(\rho)\, p(\rho)\, d\nu$, where the self-adjointness of $L$ with respect to $\nu$ was used. Then differentiate $\int LU'(\rho\_t)\, p(\rho\_t)\, d\nu$ and use $\partial\_t U'(\rho\_t) = U''(\rho\_t)\, \partial\_t \rho\_t = \lvert \nabla U'(\rho\_t) \rvert^2 + \rho\_t\, U''(\rho\_t)\, LU'(\rho\_t)$. The expression appearing is exactly *twice* the expression in (15.18), up to the replacement of $\psi$ by $-U'(\rho\_t)$. At this point, it suffices to repeat the computations leading from (15.18) to (15.20), and to apply Bochner's formula in the form (15.6)–(15.7).

The proof of (iv) is simple: By Theorem 23.9, $d(W\_2(\mu\_t, \sigma)^2/2)/dt = -\int \langle \nabla p(\rho\_t), \widetilde{\nabla}\psi \rangle\, d\nu$. By Cauchy–Schwarz, $\lvert d^+W\_2(\mu\_t, \sigma)^2/(2\, dt) \rvert \le \sqrt{I\_{U,\nu}(\mu\_t)}\, W\_2(\mu\_t, \sigma)$, so $\lvert d^+ W\_2(\mu\_t, \sigma)/dt \rvert \le \sqrt{I\_{U,\nu}(\mu\_t)}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 24.5</span><span class="math-callout__name">(Integrated Regularity for Gradient Flows)</span></p>

With the same assumptions as in Theorem 24.2, one has $U\_\nu(\mu\_t) \le U\_\nu(\mu\_0)$ and

$$\int_0^{+\infty}\!\left[\limsup_{s \downarrow 0}\, \frac{W_2(\mu_t, \mu_{t+s})}{s}\right]^2 dt \le \int_0^{+\infty} I_{U,\nu}(\mu_t)\, dt \le U_\nu(\mu_0) - (\inf U_\nu).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.6</span></p>

If $U\_\nu$ is bounded below, this corollary yields exactly the regularity which is a priori required in Theorem 23.19. It also shows that $t \to \mu\_t$ belongs to $\mathrm{AC}\_2((0, +\infty); P\_2(M))$ (absolute continuity of order 2) in the sense that there is $\ell \in L^2(dt)$ such that $W\_2(\mu\_t, \mu\_s) \le \int\_s^t \ell(\tau)\, d\tau$. Finally, the bound $\int\_0^\infty I\_{U,\nu}(\mu\_t)\, dt < +\infty$ is the assumption of Theorem 23.9.

</div>

### Large-Time Behavior

Otto's calculus, described in Chapter 15, was first developed to estimate rates of equilibration for certain nonlinear diffusion equations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 24.7</span><span class="math-callout__name">(Equilibration in Positive Curvature)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference measure $\nu = e^{-V}$, $V \in C^4(M)$, satisfying a curvature-dimension bound $\mathrm{CD}(K, N)$ for some $K > 0$, $N \in (1, \infty]$, and let $U \in \mathcal{DC}\_N$. Then:

**(i) (Exponential convergence to equilibrium)** Any smooth solution $(\mu\_t)\_{t \ge 0}$ of (24.1) satisfies the following estimates:

$$(a) \quad [U_\nu(\mu_t) - U_\nu(\nu)] \le e^{-2K\lambda t}\, [U_\nu(\mu_0) - U_\nu(\nu)]$$

$$(b) \quad I_{U,\nu}(\mu_t) \le e^{-2K\lambda t}\, I_{U,\nu}(\mu_0)$$

$$(c) \quad W_2(\mu_t, \nu) \le e^{-K\lambda t}\, W_2(\mu_0, \nu),$$

where

$$\lambda := \left(\lim_{r \to 0}\, \frac{p(r)}{r^{1 - \frac{1}{N}}}\right)\!\left(\sup_{x \in M} \rho_0(x)\right)^{-\frac{1}{N}}.$$

In particular, $\lambda$ is independent of $\rho\_0$ if $N = \infty$.

**(ii) (Exponential contraction)** Any two smooth solutions $(\mu\_t)\_{t \ge 0}$ and $(\widetilde{\mu}\_t)\_{t \ge 0}$ of (24.1) satisfy

$$W_2(\mu_t, \widetilde{\mu}_t) \le e^{-K\lambda t}\, W_2(\mu_0, \widetilde{\mu}_0),$$

where

$$\lambda := \left(\lim_{r \to 0}\, \frac{p(r)}{r^{1 - \frac{1}{N}}}\right)\!\left[\max\!\left(\sup_{x \in M} \rho_0(x),\, \sup_{x \in M} \rho_1(x)\right)\right]^{-\frac{1}{N}}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 24.8</span></p>

Smooth solutions of the Fokker–Planck equation $\partial \rho / \partial t = L\, \rho$ converge to equilibrium at least as fast as $O(e^{-Kt})$, in $W\_2$ distance, in the entropy sense (i.e. in the sense of the convergence of $\sqrt{H\_\nu(\mu)}$ to $0$), and in the Fisher information sense.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.9</span></p>

At least formally, these properties are in fact *general properties of gradient flows*: Let $F$ be a function defined on a geodesically convex subset of a Riemannian manifold $(M, g)$; $\operatorname{Hess} F \ge \lambda\, g$, $\lambda > 0$; $X\_\infty$ is the minimizer of $F$; and $X$, $\widetilde{X}$ are two trajectories of the gradient flow associated with $F$. Then we have three neat estimates: (a) $[F(X(t)) - F(X\_\infty)] \le e^{-\lambda t}\, [F(X(0)) - F(X\_\infty)]$; (b) $\lvert \nabla F(X(t)) \rvert \le e^{-\lambda t}\, \lvert \nabla F(X(0)) \rvert$; (c) $d(X(t), \widetilde{X}(t)) \le e^{-\lambda t}\, d(X(0), \widetilde{X}(0))$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.10</span></p>

The rate of decay $O(e^{-\lambda t})$ is optimal for (24.9) if dimension is not taken into account; but if $N$ is finite, the optimal rate of decay is $O(e^{-\lambda t})$ with $\lambda = KN/(N - 1)$. The method presented in this chapter is not clever enough to catch this sharp rate.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.12</span></p>

If $N < \infty$, Theorem 24.7 proves convergence to equilibrium with a rate that depends on the initial datum. However, if the solution $(\rho\_t)\_{t \ge 0}$ satisfies *uniform* smoothness bounds, it is often possible to reinforce the statement $\rho\_t \xrightarrow{L^1} 1$ into $\rho\_t \xrightarrow{L^\infty} 1$. Then we can choose $\rho\_T$ as new initial datum, and get

$$t \ge T \implies U_\nu(\mu_t) \le e^{-K\lambda_T\, (t-T)}\, U_\nu(\mu_T) \le e^{-K\lambda_T\, (t-T)}\, U_\nu(\mu_0),$$

where $\lambda\_T = (\lim p(r)/r^{1-1/N})\, (\sup \rho\_T)^{-1/N} \longrightarrow \lambda\_\infty = (\lim p(r)/r^{1-1/N})$ as $T \to \infty$. It follows that $\mu\_t$ converges to $\nu$ as $O(e^{-K\widetilde{\lambda}\, t})$ for any $\widetilde{\lambda} > \lambda$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 24.7)</span></p>

Let $H(t) = U\_\nu(\mu\_t)$. Theorem 24.2(ii) reads $H'(t) = -I\_{U,\nu}(\mu\_t)$. Let $\lambda\_0 := \lim\_{r \to 0} p(r)/r^{1-1/N}$. The (modified) Sobolev inequality of Theorem 21.7 implies $U\_\nu(\mu\_t) \le (\sup \rho\_t)^{1/N}/(2K\lambda\_0)\, I\_{U,\nu}(\mu\_t)$. Thus $(d/dt)\, H(t) \le -2K\lambda\_0\, (\sup \rho\_t)^{-1/N}\, H(t)$.

Theorem 24.2(i) with $A(r) = r^p$, $p \ge 2$, gives $(d/dt) \int \rho^p\, d\nu = -p(p-1) \int \rho\, U''(\rho)\, \rho^{p-2}\, \lvert \nabla \rho \rvert^2\, d\nu \le 0$. So $\lVert \rho\_t \rVert\_{L^p}$ is a nonincreasing function of $t$, and therefore $\forall t \ge 0$, $\lVert \rho\_t \rVert\_{L^p(\nu)} \le \lVert \rho\_0 \rVert\_{L^p(\nu)}$. Passing to the limit as $p \to \infty$ yields $\forall t \ge 0$, $\sup \rho\_t \le \sup \rho\_0$. Plugging this back into (24.10), we get $(d/dt)\, H(t) \le -2K\lambda\_0\, (\sup \rho\_0)^{-1/N}\, H(t) = -2K\lambda\, H(t)$, and then (24.5)(a) follows.

Next, if $U \in \mathcal{DC}\_N$ and $\mathrm{CD}(K, N)$ is enforced, we can write, as in (16.13), $-(1/2)\, (d/dt)\, I\_{U,\nu}(\mu\_t) = \int \Gamma\_2(U'(\rho\_t))\, p(\rho\_t)\, d\nu + \int (LU'(\rho\_t))^2\, [p\_2 + p/N](\rho\_t)\, d\nu \ge K \int \lvert \nabla U'(\rho\_t) \rvert^2\, p(\rho\_t)\, d\nu \ge K\lambda\_0\, (\sup \rho\_t)^{-1/N}\, I\_{U,\nu}(\mu\_t)$. This implies (24.5)(b).

For (24.7), the strategy is the same as for Theorem 23.26. The assumption $K \ge 0$ implies $\sup \rho\_t \le \sup \rho\_0$ and $\sup \widetilde{\rho}\_t \le \sup \widetilde{\rho}\_0$ (by (24.11)). If $(\mu^{(s)} = \rho^{(s)}\, \nu)\_{0 \le s \le 1}$ is the displacement interpolation between $\rho\_t\, \nu$ and $\widetilde{\rho}\_t\, \nu$, then by displacement convexity $\sup \rho^{(s)} \le \max(\sup \rho\_0, \sup \widetilde{\rho}\_0)$. Apply Theorem 23.14 with $\widetilde{\mu}\_t$ replaced by $\mu\_t$ and vice versa, combine with Theorem 23.9 to get $(d^+/dt)\, W\_2(\mu\_t, \widetilde{\mu}\_t)^2 \le -2K\lambda\, W\_2(\mu\_t, \widetilde{\mu}\_t)^2$, and the desired result follows.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.14</span></p>

In the particular case $\widetilde{\mu}\_t = \nu$, inequality (24.13) and Theorem 23.9 imply the following: Let $M$ satisfy a $\mathrm{CD}(K, N)$ condition with $K \ge 0$, let $U \in \mathcal{DC}\_N$ with $U(1) = 0$, let $\mu = \rho\, \nu \in P\_2^{\mathrm{ac}}(M)$, and let $\lambda := (\lim\_{r \to 0} p(r)/r^{1-1/N})\, (\sup \rho\_0(x))^{-1/N}$; if $(\mu\_t)\_{0 \le t \le 1}$ is a smooth solution of the gradient flow $\partial\_t \rho\_t = L\, p(\rho\_t)$ starting from $\mu\_0 = \mu$, then for almost all $t$,

$$-\left.\frac{d^+}{dt}\right|_{t=0}\!\left(\frac{W_2(\mu_t, \nu)^2}{2}\right) \ge U_\nu(\mu) + \frac{K\lambda\, W_2(\mu, \nu)^2}{2}.$$

So (24.17) is a reinforcement of the Talagrand-type inequality $U\_\nu(\mu) \ge K\lambda\, W\_2(\mu, \nu)^2/2$.

</div>

### Short-Time Behavior

A popular and useful topic in the study of diffusion processes consists in establishing **regularization estimates** in short time. Typically, a certain functional used to quantify the regularity of the solution (for instance, the supremum of the unknown or some Lebesgue or Sobolev norm) is shown to be bounded like $O(t^{-\kappa})$ for some characteristic exponent $\kappa$, independent of the initial datum (or depending only on certain weak estimates on the initial datum), when $t > 0$ is small enough.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 24.16</span><span class="math-callout__name">(Short-Time Regularization for Gradient Flows)</span></p>

Let $M$ be a Riemannian manifold satisfying a curvature-dimension bound $\mathrm{CD}(K, \infty)$, $K \in \mathbb{R}$; let $\nu = e^{-V}\, \mathrm{vol} \in P\_2(M)$, with $V \in C^4(M)$, and let $U \in \mathcal{DC}\_\infty$ with $U(1) = 0$. Further, let $(\mu\_t)\_{t \ge 0}$ be a smooth solution of (24.1). Then:

**(i)** If $K \ge 0$ then for any $t \ge 0$,

$$t^2\, I_{U,\nu}(\mu_t) + 2t\, U_\nu(\mu_t) + W_2(\mu_t, \nu)^2 \le W_2(\mu_0, \nu)^2.$$

In particular,

$$U_\nu(\mu_t) \le \frac{W_2(\mu_0, \nu)^2}{2t}, \qquad I_{U,\nu}(\mu_t) \le \frac{W_2(\mu_0, \nu)^2}{t^2}.$$

**(ii)** If $K \ge 0$ and $t \ge s > 0$, then

$$W_2(\mu_s, \mu_t) \le \min\!\left(\sqrt{2\, U_\nu(\mu_s)}\, \sqrt{\lvert t - s \rvert},\, \sqrt{I_{U,\nu}(\mu_s)}\, \lvert t - s \rvert\right) \le W_2(\mu_0, \nu)\, \min\!\left(\frac{\sqrt{\lvert t - s \rvert}}{\sqrt{s}},\, \frac{\lvert t - s \rvert}{s}\right).$$

**(iii)** If $K < 0$, the previous conclusions become

$$U_\nu(\mu_t) \le \frac{e^{2Ct}\, W_2(\mu_0, \nu)^2}{2t}; \qquad I_{U,\nu}(\mu_t) \le \frac{e^{2Ct}\, W_2(\mu_0, \nu)^2}{t^2};$$

$$W_2(\mu_s, \mu_t) \le e^{Ct}\, \min\!\left(\sqrt{2\, U_\nu(\mu_s)}\, \sqrt{\lvert t - s \rvert},\, \sqrt{I_{U,\nu}(\mu_s)}\, \lvert t - s \rvert\right) \le e^{2Ct}\, W_2(\mu_0, \nu)\, \min\!\left(\frac{\sqrt{\lvert t - s \rvert}}{\sqrt{s}},\, \frac{\lvert t - s \rvert}{s}\right),$$

with $C = -K$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Particular Case 24.17: $U(\rho) = \rho \log \rho$)</span></p>

When $U(\rho) = \rho \log \rho$, inequalities (24.18) and (24.19) become

$$H_\nu(\mu_t) \le \frac{W_2(\mu_0, \nu)^2}{2t}, \qquad I_\nu(\mu_t) \le \frac{W_2(\mu_0, \nu)^2}{t^2}.$$

Under a $\mathrm{CD}(K, \infty)$ bound ($K < 0$) there is an additional factor $e^{-2Kt}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.18</span></p>

Theorem 24.16 should be thought of as an *a priori estimate*. If life is not unfair, one can then remove the assumption of smoothness by a density argument, and transform (24.18), (24.19) into genuine regularization estimates. This is true at least for the Particular Case 24.17.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.19</span></p>

Inequalities (24.20) and (24.21) establish the following estimates: The curve $(\mu\_t)\_{t \ge 0}$, viewed as a function of time $t$, is Hölder-$1/2$ close to $t = 0$, and Lipschitz away from $t = 0$, if $U\_\nu(\mu\_0)$ is finite. If $I\_{U,\nu}(\mu\_0)$ is finite, then the curve is Lipschitz all along.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.20</span></p>

Theorem 24.7 gave upper bounds on $U\_\nu(\mu\_t) - U\_\nu(\nu)$ like $O(e^{-\kappa t})$, with a constant depending on $U\_\nu(\mu\_0)$. But now we can combine Theorem 24.7 with Theorem 24.16 to get an exponential decay with a constant that does not depend on $U\_\nu(\mu\_0)$, but only on $W\_2(\mu\_0, \nu)$. By approximation, this will lead to results of convergence that do not need the finiteness of $U\_\nu(\mu\_0)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 24.16)</span></p>

First note that $U(1) = 0$ implies $U\_\nu(\mu) \ge 0$ and $U\_\nu(\nu) = 0$.

Let $t > 0$ be given, and let $\exp(\widetilde{\nabla}\psi)$ be the optimal transport between $\mu\_t$ and $\nu$, where $\psi$ is $d^2/2$-convex. Since $U\_\nu(\nu) = 0$ and $K \ge 0$, Theorem 23.14 implies $U\_\nu(\mu\_t) + \int \langle \widetilde{\nabla}\psi, \nabla p(\rho\_t) \rangle\, d\nu \le 0$. On the other hand, by Theorem 23.9, for almost all $t$, $(d^+/dt)\, W\_2(\mu\_t, \nu)^2 \le 2 \int \langle \widetilde{\nabla}\psi, \nabla p(\rho\_t) \rangle\, d\nu$. Combining: $(d^+/dt)\, W\_2(\mu\_t, \nu)^2 \le -2\, U\_\nu(\mu\_t)$.

Now introduce $\psi(t) := a(t)\, I\_{U,\nu}(\mu\_t) + b(t)\, U\_\nu(\mu\_t) + c(t)\, W\_2(\mu\_t, \nu)^2$, where $a(t)$, $b(t)$, $c(t)$ will be determined later. Because of the nonnegative curvature assumption, $I\_{U,\nu}(\mu\_t)$ is nonincreasing with time. Combining this with (24.25) and Theorem 24.2(ii), we get $d^+\psi/dt \le [a'(t) - b(t)]\, I\_{U,\nu}(\mu\_t) + [b'(t) - 2c(t)]\, U\_\nu(\mu\_t) + c'(t)\, W\_2(\mu\_t, \nu)^2$. If we choose $a(t) \equiv t^2$, $b(t) \equiv 2t$, $c(t) \equiv 1$, then $\psi$ has to be nonincreasing as a function of $t$, and this implies (i).

For (ii), by Theorem 24.2(iv), $(d^+/dt)\, W\_2(\mu\_s, \mu\_t) \le \sqrt{I\_{U,\nu}(\mu\_t)} \le \sqrt{I\_{U,\nu}(\mu\_s)}$, so $W\_2(\mu\_s, \mu\_t) \le \sqrt{I\_{U,\nu}(\mu\_s)}\, \lvert t - s \rvert$. On the other hand, by Theorems 23.9 and 23.14 (with $K = 0$, $\sigma$ replaced by $\mu\_t$ and $\mu$ replaced by $\mu\_s$), $(d^+/dt)\, W\_2(\mu\_s, \mu\_t)^2 \le 2\, [U\_\nu(\mu\_s) - U\_\nu(\mu\_t)] \le 2\, U\_\nu(\mu\_s)$. So $W\_2(\mu\_s, \mu\_t)^2 \le 2\, U\_\nu(\mu\_s)\, \lvert t - s \rvert$. Combining with (i) yields (ii).

The proof of (iii) is similar with modifications: $dI\_{U,\nu}(\mu\_t)/dt \le (-2K)\, I\_{U,\nu}(\mu\_t)$; $(d^+/dt)\, W\_2(\mu\_t, \nu)^2 \le -2\, U\_\nu(\mu\_t) + (-2K)\, W\_2(\mu\_t, \nu)^2$; $\psi(t) := e^{2Kt}\, (t^2\, I\_{U,\nu}(\mu\_t) + 2t\, U\_\nu(\mu\_t) + W\_2(\mu\_t, \nu)^2)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24.23</span></p>

There are many known regularization results in short time, for certain of the gradient flows considered in this chapter. The two most famous examples are:

- the **Li–Yau estimates**, which give lower bounds on $\Delta \log \rho\_t$, for a solution of the heat equation on a Riemannian manifold, under certain curvature-dimension conditions. For instance, if $M$ satisfies $\mathrm{CD}(0, N)$, then $\Delta \log \rho\_t \ge -N/(2t)$.

- the **Aronson–Bénilan estimates**, which give lower bounds on $\Delta \rho\_t^{m-1}$ for solutions of the nonlinear diffusion equation $\partial\_t \rho = \Delta \rho^m$ in $\mathbb{R}^n$, where $1 - 2/n < m < 1$: $\frac{m}{m - 1}\, \Delta(\rho\_t^{m-1}) \ge -n/(\lambda\, t)$, $\lambda = 2 - n(1 - m)$.

There is an obvious similarity between these two estimates, and both can be interpreted as a lower bound on the rate of divergence of the vector field which drives particles in the gradient flow interpretation of these PDEs.

</div>

### Bibliographical Notes on Chapter 24

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

**Otto** advocated the use of his formalism both for the purpose of finding new schemes of proof, and for giving a new understanding of certain results.

What I call the **Fokker–Planck equation** is $\partial \mu / \partial t = \Delta \mu + \nabla \cdot (\mu\_t\, \nabla V)$. This is in fact an equation on *measures*. It can be recast as an equation on *functions* (densities): $\partial \rho / \partial t = \Delta \rho - \nabla V \cdot \nabla \rho$. From the point of view of stochastic processes, $\mu\_t$ can be thought of as law$(X\_t)$, where $X\_t$ is the stochastic process defined by $dX\_t = \sqrt{2}\, dB\_t - \nabla V(X\_t)\, dt$. In the particular case when $V$ is a quadratic potential in $\mathbb{R}^n$, the evolution equation for $\rho\_t$ is often called the **Ornstein–Uhlenbeck equation**.

The observation that the Fisher information $I\_\nu$ is the time-derivative of the entropy functional $-H\_\nu$ along the heat semigroup seems to first appear in a famous paper by **Stam** at the end of the fifties (Stam gives credit to **de Bruijn** for that remark). The generalization appearing in Theorem 24.2(ii) has been discovered and rediscovered by many authors.

**Theorem 24.2(iii)** goes back to **Bakry and Émery** for the case $U(r) = r \log r$. After many successive generalizations, the statement as I wrote it was formally derived in the author's work. The argument given in this chapter is the first rigorous one to be written down in detail (modulo the technical justifications of the integrations by parts).

**Theorem 24.2(iv)** was proven by **Otto and myself** for $\sigma = \mu\_0$.

It has been known since the mid-seventies that logarithmic Sobolev inequalities yield rates of convergence to equilibrium for heat-like equations, and that these estimates are independent of the dimension. Around the mid-nineties, **Toscani** introduced the logarithmic Sobolev inequality in kinetic theory. The links between logarithmic Sobolev inequalities and Fokker–Planck equations were re-investigated by the kinetic theory community.

Around 2000, it was discovered independently by **Otto**, **Carrillo and Toscani** and **Del Pino and Dolbeault** that the same "information-theoretical" tools could be used for nonlinear equations of the form $\partial \rho / \partial t = \Delta \rho^m$ in $\mathbb{R}^n$. Such equations are called **porous medium equations** for $m > 1$, and **fast diffusion equations** for $m < 1$. There is a well-known scaling, due to **Barenblatt**, which transforms (24.28) into $\partial \rho / \partial t = \Delta \rho^m + \nabla\_x \cdot (\rho\, x)$. The extra drift term acts like the confinement by a quadratic potential, and this is in effect equivalent to imposing a curvature condition $\mathrm{CD}(K, \infty)$ ($K > 0$). This explains why there is an approach based on generalized logarithmic Sobolev inequalities, quite similar to the proof of Theorem 24.7.

The setting of the more general energy functional $E(\mu) = \int U(\rho)\, dx + \int V\, d\mu + \frac{1}{2} \int W(x - y)\, d\mu\, d\mu$ leads to the gradient flow $\partial \rho / \partial t = \sigma\, \Delta \rho + \nabla \cdot (\rho\, \nabla V) + \nabla \cdot (\rho\, \nabla(\rho \ast \nabla W))$, where $\sigma \in \mathbb{R}\_+$ and $W = W(x - y)$ is some interaction potential on $\mathbb{R}^n$. These equations (a particular instance of **McKean–Vlasov equations**) appeared in the modeling of granular media. The study of exponential convergence for (24.30) leads to interesting issues, with criteria for exponential convergence in terms of the convexity of $V$ and $W$.

**Demange** studied the fast diffusion equation $\partial\_t \rho = \Delta \rho^{1-1/N}$ on a Riemannian manifold, under a curvature-dimension condition $\mathrm{CD}(K, N)$. He used the Sobolev inequality to obtain a differential inequality and deduced an estimate of the form $H\_{N/2}(\mu\_t) = O(e^{-(\lambda\_N + \varepsilon)\, t})$, where $\lambda\_N$ is the presumably optimal rate that one would obtain without the $(\sup \rho)$ term, and $\varepsilon > 0$ is arbitrarily small.

</div>

## Chapter 25: Gradient Flows III — Functional Inequalities

In the preceding chapter certain functional inequalities were used to provide quantitative information about the behavior of solutions to certain partial differential equations. In the present chapter, conversely, the behavior of solutions to certain partial differential equations will help establish certain functional inequalities.

For the kind of inequalities that will be encountered in this chapter, this principle has been explored in depth since the mid-eighties, starting with **Bakry and Émery's** heat semigroup proof of Theorem 21.2. Nowadays, one can prove this theorem by more direct means (as done in Chapter 21); nevertheless, the heat semigroup argument is still of interest, and not only for historical reasons. Indeed it has been the basis for many generalizations, some of which are still out of reach of alternative methods.

Optimal transport appears in this game from two different perspectives. On the one hand, several inequalities involving optimal transport have been proven by diffusion semigroup methods. On the other hand, optimal transport has provided a re-interpretation of these methods, since several diffusion equations can be understood as gradient flows with respect to a structure induced by optimal transport. This interpretation has led to a more synthetic and geometric picture of the field; and Otto's calculus has provided a way to shortcut some intricate computations.

There are limitations to this point of view. It is usually okay to interpret in terms of optimal transport a proof involving functions in $\mathcal{DC}\_\infty$ under a curvature-dimension assumption $\mathrm{CD}(K, \infty)$. Such is also the case for a proof involving functions in $\mathcal{DC}\_N$ under a curvature-dimension assumption $\mathrm{CD}(K, N)$. But to get the correct constants for an inequality involving functions in $\mathcal{DC}\_N$ under a condition $\mathrm{CD}(K, N')$, $N' < N$, may be much more of a problem.

Three examples are discussed: an alternative proof of Theorem 21.2 (log Sobolev from Bakry–Émery), a proof of the optimal Sobolev inequality (21.8) under a $\mathrm{CD}(K, N)$ condition (recently discovered by Demange), and an alternative proof of Theorem 22.17 (log Sobolev implies Talagrand).

Conventions as in previous chapters: $U$ is a nonlinearity belonging to some displacement convexity class, $p(r) = r\, U'(r) - U(r)$, $\nu = e^{-V}\, \mathrm{vol}$ is a reference measure, $L$ is the associated Laplace-type operator admitting $\nu$ as invariant measure. Moreover,

$$U_\nu(\mu) = \int U(\rho)\, d\nu, \qquad I_{U,\nu}(\mu) = \int \rho\, \lvert \nabla U'(\rho) \rvert^2\, d\nu = \int \frac{\lvert \nabla p(\rho) \rvert^2}{\rho}\, d\nu,$$

$$H_{N,\nu}(\mu) = -N \int (\rho^{1-\frac{1}{N}} - \rho)\, d\nu, \qquad I_{N,\nu}(\mu) = \left(1 - \frac{1}{N}\right)^2 \int \rho^{-1-\frac{2}{N}}\, \lvert \nabla \rho \rvert^2\, d\nu,$$

$$H_{\infty,\nu}(\mu) = H_\nu(\mu) = \int \rho \log \rho\, d\nu, \qquad I_{\infty,\nu}(\mu) = I_\nu(\mu) = \int \frac{\lvert \nabla \rho \rvert^2}{\rho}\, d\nu,$$

where $\rho$ always stands for the density of $\mu$ with respect to $\nu$.

### Logarithmic Sobolev Inequalities Revisited

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 25.1</span><span class="math-callout__name">(Infinite-Dimensional Sobolev Inequalities from Ricci Curvature)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference measure $\nu$ satisfying a curvature-dimension bound $\mathrm{CD}(K, \infty)$ for some $K > 0$, and let $U \in \mathcal{DC}\_\infty$. Further, let $\lambda := \lim\_{r \to 0} p(r)/r$. Then, for all $\mu \in P\_2^{\mathrm{ac}}(M)$,

$$U_\nu(\mu) - U_\nu(\nu) \le \frac{I_{U,\nu}(\mu)}{2K\lambda}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Particular Case 25.2: Bakry–Émery Theorem Again)</span></p>

If $(M, \nu)$ satisfies $\mathrm{CD}(K, \infty)$ for some $K > 0$, then the following logarithmic Sobolev inequality holds true:

$$\forall \mu \in P^{\mathrm{ac}}(M), \qquad H_\nu(\mu) \le \frac{I_\nu(\mu)}{2K}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 25.1)</span></p>

By using Theorem 17.7(vii) and an approximation argument, we may assume that $\rho$ is smooth, that $U$ is smooth on $(0, +\infty)$, that the solution $(\rho\_t)\_{t \ge 0}$ of the gradient flow $\partial \rho / \partial t = L\, p(\rho\_t)$ starting from $\rho\_0 = \rho$ is smooth, that $U\_\nu(\mu\_0)$ is finite, and that $t \to U\_\nu(\mu\_t)$ is continuous at $t = 0$.

For notational simplicity, let $H(t) := U\_\nu(\mu\_t)$ and $I(t) := I\_{U,\nu}(\mu\_t)$. From Theorems 24.2(ii) and 24.7(i)(b),

$$\frac{dH(t)}{dt} = -I(t), \qquad I(t) \le I(0)\, e^{-2K\lambda t}.$$

By Theorem 24.7(i)(a), $H(t) \to 0$ as $t \to \infty$. So

$$H(0) = \int_0^{+\infty} I(t)\, dt \le I(0) \int_0^{+\infty} e^{-2K\lambda t}\, dt = \frac{I(0)}{2K\lambda},$$

which is the desired result.

</div>

### Sobolev Inequalities Revisited

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 25.3</span><span class="math-callout__name">(Generalized Sobolev Inequalities under Ricci Curvature Bounds)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference measure $\nu = e^{-V}$, $V \in C^2(M)$, satisfying a curvature-dimension bound $\mathrm{CD}(K, N)$ for some $K > 0$, $N \in [1, \infty)$. Let $U \in \mathcal{DC}\_N$ with $U'' > 0$ on $(0, +\infty)$, and let $A \in C(\mathbb{R}\_+) \cap C^2((0, +\infty))$ be such that $A(0) = A(1) = 0$ and $A''(r) = r^{-\frac{1}{N}}\, U''(r)$. Then, for any probability density $\rho$ on $M$,

$$\int_M A(\rho)\, d\nu \le \frac{1}{2K\lambda} \int_M \rho\, \lvert \nabla U'(\rho) \rvert^2\, d\nu,$$

where $\lambda = \lim\_{r \downarrow 0} p(r)/r^{1-\frac{1}{N}}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 25.4</span></p>

For a given $U$, there might not necessarily exist a suitable $A$. For instance, if $U = U\_N$, it is only for $N > 2$ that we can construct $A$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Particular Case 25.5: Sobolev Inequalities)</span></p>

Whenever $N > 2$, let

$$U(r) = U_N(r) = -N\, (r^{1-\frac{1}{N}} - r), \qquad A(r) = -\frac{N(N-1)}{2(N-2)}\, (r^{1-\frac{2}{N}} - r);$$

then (25.1) reads

$$H_{\frac{N}{2},\nu}(\mu) \le \frac{1}{2K}\!\left(\frac{N-2}{N-1}\right) I_{N,\nu}(\mu),$$

which can also be rewritten in the form of (21.9) or (21.8).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 25.3)</span></p>

By density, we may assume that the density $\rho\_0$ of $\mu$ is smooth; we may also assume that $A$ and $U$ are smooth on $(0, +\infty)$ (recall Proposition 17.7(vii)). Let $(\rho\_t)\_{t \ge 0}$ be the solution of the gradient flow equation $\partial \rho / \partial t = \nabla \cdot (\rho\, \nabla U'(\rho))$, and as usual $\mu\_t = \rho\_t\, \nu$. It can be shown that $\rho\_t$ is uniformly bounded below by a positive number as $t \to \infty$.

By Theorem 24.2(iii),

$$\frac{d}{dt}\, I_{U,\nu}(\mu_t) \le -2K\lambda \int_M \rho_t^{1-\frac{1}{N}}\, \lvert \nabla U'(\rho_t) \rvert^2\, d\nu.$$

On the other hand, from the assumption $A''(r) = r^{-\frac{1}{N}}\, U''(r)$, we get $\nabla A'(\rho) = \rho^{-\frac{1}{N}}\, \nabla U'(\rho)$. So Theorem 24.2(i) implies

$$\frac{d}{dt} \int A(\rho_t)\, d\nu = -\int_M \rho_t^{1-\frac{1}{N}}\, \lvert \nabla U'(\rho_t) \rvert^2\, d\nu.$$

The combination of (25.3) and (25.4) leads to

$$-\frac{d}{dt}\, A_\nu(\mu_t) \le -\left(\frac{1}{2K\lambda}\right)\frac{d}{dt}\, I_{U,\nu}(\mu_t).$$

As $t \to \infty$, $I\_{U,\nu}(\mu\_t)$ and $U\_\nu(\mu\_t)$ converge to 0 (Theorem 24.7(i)). Since $\rho\_t$ is uniformly bounded below and $U''$ is uniformly positive on the range of $\rho\_t$, this implies that $\rho\_t \to 1$ in $L^1(\nu)$, and also that $A\_\nu(\mu\_t) \to 0$. Then one can integrate both sides of (25.5) from $t = 0$ to $t = \infty$, and recover

$$A_\nu(\mu_0) \le \left(\frac{1}{2K\lambda}\right) I_{U,\nu}(\mu_0),$$

as desired.

</div>

### From Log Sobolev to Talagrand, Revisited

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 25.6</span><span class="math-callout__name">(From Sobolev-Type Inequalities to Concentration Inequalities)</span></p>

Let $M$ be a Riemannian manifold equipped with a reference probability measure $\nu = e^{-V}\, \mathrm{vol} \in P\_2^{\mathrm{ac}}(M)$, $V \in C^2(M)$. Let $U \in \mathcal{DC}\_\infty$. Assume that for any $\mu \in P\_2^{\mathrm{ac}}(M)$, holds the inequality

$$U_\nu(\mu) - U_\nu(\nu) \le \frac{1}{2K}\, I_{U,\nu}(\mu).$$

Further assume that the Cauchy problem associated with the gradient flow $\partial\_t \rho = L\, p(\rho)$ admits smooth solutions for smooth initial data. Then, for any $\mu \in P\_2^{\mathrm{ac}}(M)$, holds the inequality

$$\frac{W_2(\mu, \nu)^2}{2} \le \frac{U_\nu(\mu) - U_\nu(\nu)}{K}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Particular Case 25.7: From Log Sobolev to Talagrand)</span></p>

If the reference measure $\nu$ on $M$ satisfies a logarithmic Sobolev inequality with constant $K$, and a curvature-dimension bound $\mathrm{CD}(K', \infty)$ for some $K' \in \mathbb{R}$, then it also satisfies a **Talagrand inequality** with constant $K$:

$$\forall \mu \in P_2^{\mathrm{ac}}(M), \qquad W_2(\mu, \nu) \le \sqrt{\frac{2\, H_\nu(\mu)}{K}}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 25.6)</span></p>

By a density argument, we may assume that $\mu$ has a smooth density $\mu\_0$, and let $(\mu\_t)\_{t \ge 0}$ evolve according to the gradient flow (25.2). By Theorem 24.2(ii), $(d/dt)\, U\_\nu(\mu\_t) = -I\_{U,\nu}(\mu\_t)$. In particular, $(d/dt)\, U\_\nu(\mu\_t) \le -2K\, U\_\nu(\mu\_t)$, so $U\_\nu(\mu\_t) \to 0$ as $t \to \infty$ (exponentially fast).

By Theorem 24.2(iv), for almost all $t$, $(d^+/dt)\, W\_2(\mu\_0, \mu\_t) \le \sqrt{I\_{U,\nu}(\mu\_t)}$. On the other hand, by assumption,

$$\sqrt{I_{U,\nu}(\mu_t)} \le \frac{I_{U,\nu}(\mu_t)}{\sqrt{2K\, U_\nu(\mu_t)}} = -\frac{d}{dt}\, \sqrt{\frac{2\, U_\nu(\mu_t)}{K}}.$$

From (24.4) and (25.8),

$$\frac{d^+}{dt}\, W_2(\mu_0, \mu_t) \le -\frac{d}{dt}\, \sqrt{\frac{2\, U_\nu(\mu_t)}{K}}.$$

Stated otherwise: If $\psi(t) := W\_2(\mu\_0, \mu\_t) + \sqrt{2\, U\_\nu(\mu\_t)/K}$, then $d^+\psi/dt \le 0$, i.e. $\psi$ is nonincreasing as a function of $t$, and so $\lim\_{t \to \infty} \psi(t) \le \psi(0)$.

Let us now check that $\mu\_t$ converges weakly to $\nu$. Inequality (25.9) implies that $W\_2(\mu\_0, \mu\_t)$ remains bounded as $t \to \infty$; so $\int d(z, x)\, \mu\_t(dx) \le \int d(z, x)\, \mu\_0(dx) + W\_1(\mu\_0, \mu\_t)$ is also uniformly bounded, and $\lbrace \mu\_t \rbrace$ is tight as $t \to \infty$. Up to extraction of a sequence of times, $\mu\_t$ converges weakly to some measure $\widetilde{\mu}$. On the other hand, the functional inequality (25.6) forces $U''$ to be positive on $(0, +\infty)$, and then the convergence $U\_\nu(\mu\_t) \to 0 = U\_\nu(\nu)$ is easily seen to imply $\rho\_t \xrightarrow{} 1$ almost surely. This combined with the weak convergence of $\mu$ to $\widetilde{\mu}$ imposes $\widetilde{\mu} = \nu$; so $\mu\_t$ does converge weakly to $\nu$. As a consequence,

$$W_2(\mu_0, \nu) \le \liminf_{t \to \infty}\, W_2(\mu_0, \mu_t) = \liminf_{t \to \infty}\, \psi(t) \le \psi(0) = \sqrt{(2\, U_\nu(\mu_0))/K},$$

which proves the claim.

</div>

### Appendix: Comparison of Proofs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gradient Flow vs. Displacement Convexity Proofs)</span></p>

The proofs in this chapter were based on gradient flows of displacement convex functionals, while proofs in Chapters 21 and 22 were more directly based on displacement interpolation. How do these two strategies compare to each other?

From a formal point of view, they are not so different. Take the case of the heat equation $\partial \rho / \partial t = \Delta \rho$, or equivalently $\partial \rho / \partial t + \nabla \cdot (\rho\, \nabla(-\log \rho)) = 0$. The evolution of $\rho$ is determined by the "vector field" $\rho \to (-\log \rho)$ in the space of probability densities. Rescale time and the vector field itself: $\varphi\_\varepsilon(t, x) = -\varepsilon \log \rho(\varepsilon t / 2, x)$. Then $\varphi\_\varepsilon$ satisfies $\partial \varphi\_\varepsilon / \partial t + \lvert \nabla \varphi\_\varepsilon \rvert^2/2 = (\varepsilon/2)\, \Delta \varphi\_\varepsilon$. Passing to the limit as $\varepsilon \to 0$, one gets, at least formally, the **Hamilton–Jacobi equation** $\partial \varphi / \partial t + \lvert \nabla \varphi \rvert^2/2 = 0$, which is in some sense the equation driving displacement interpolation.

There is a general principle here: After suitable rescaling, the velocity field associated with a gradient flow resembles the velocity field of a geodesic flow. Thus we may expect the gradient flow strategy to be more precise than the displacement convexity strategy. This is also what the use of Otto's calculus suggests: Proofs based on gradient flows need a control of $\operatorname{Hess} U\_\nu$ only in the direction $\operatorname{grad} U\_\nu$, while proofs based on displacement convexity need a control of $\operatorname{Hess} U\_\nu$ in all directions. This might explain why there is at present no displacement convexity analogue of Demange's proof of the Sobolev inequality (so far only weaker inequalities with nonsharp constants have been obtained).

On the other hand, proofs based on displacement convexity are usually rather simpler, and more robust than proofs based on gradient flows: no issues about the regularity of the semigroup, no subtle interplay between the Hessian of the functional and the "direction of evolution".

In the end we can put some of the main functional inequalities discussed in these notes in a nice array. Below, "LSI" stands for "Logarithmic Sobolev inequality"; "T" for "Talagrand inequality"; and "Sob$\_2$" for the Sobolev inequality with exponent 2. So $\mathrm{LSI}(K)$, $\mathrm{T}(K)$, $\mathrm{HWI}(K)$ and $\mathrm{Sob}\_2(K, N)$ respectively stand for (21.4), (22.4) (with $p = 2$), (20.17) and (21.8).

| Theorem | Gradient flow proof | Displ. convexity proof |
| --- | --- | --- |
| $\mathrm{CD}(K, \infty) \Rightarrow \mathrm{LSI}(K)$ | Bakry–Émery | Otto–Villani |
| $\mathrm{LSI}(K) \Rightarrow \mathrm{T}(K)$ | Otto–Villani | Bobkov–Gentil–Ledoux |
| $\mathrm{CD}(K, \infty) \Rightarrow \mathrm{HWI}(K)$ | Bobkov–Gentil–Ledoux | Otto–Villani |
| $\mathrm{CD}(K, N) \Rightarrow \mathrm{Sob}\_2(K, N)$ | Demange | ?? |

</div>

### Bibliographical Notes on Chapter 25

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

**Stam** used a heat semigroup argument to prove an inequality which is equivalent to the Gaussian logarithmic Sobolev inequality in dimension 1. His argument was not completely rigorous because of regularity issues, but can be repaired.

The proof of **Theorem 25.1** in this chapter follows the strategy by **Bakry and Émery**, who were only interested in the Particular Case 25.2. These authors used a set of calculus rules which has been dubbed the "$\Gamma\_2$ calculus". They were not very careful about regularity issues, and for that reason the original proof probably cannot be considered as completely rigorous (in particular for noncompact manifolds). However, recently **Demange** carried out complete proofs for much more delicate situations, so there is no reason to doubt that the Bakry–Émery argument can be made fully rigorous. Also, when the manifold is $\mathbb{R}^n$ equipped with a reference density $e^{-V}$, the proof was carefully rewritten by **Arnold, Markowich, Toscani and Unterreiter**, in the language of partial differential equations.

The **Bakry–Émery strategy** was applied independently by **Otto** and by **Carrillo and Toscani** to study the asymptotic behavior of porous medium equations. Since then, many authors have applied it to various classes of nonlinear equations.

The interpretation of the Bakry–Émery proof as a gradient flow argument was developed in the author's paper with **Otto**. This interpretation was of much help when we considered more complicated nonlinear situations.

**Theorem 25.3** is due to **Demange**. Demange did not only treat the inequality (21.9), but also the whole family (21.7). A disturbing remark is that for many members of this family, several distinct gradient flows can be used to yield the same functional inequality. Demange also discussed other criteria than $U \in \mathcal{DC}\_N$, allowing for finer results if, say, $U \in \mathcal{DC}\_N$ but the curvature-dimension bound is $\mathrm{CD}(K, N')$ for some $N' < N$; at this point he uses formulas of change of variables for $\Gamma\_2$ operators.

The proof of **Theorem 25.6** was implemented in the author's joint work with **Otto**. The proof there is (hopefully!) complete, but only the Particular Case 25.7 (the most important) was considered. Later, **Biane and Voiculescu** adapted the argument to free probability theory, deriving a noncommutative analog of the Talagrand inequality. **F.-Y. Wang**, and **Cattiaux and Guillin** have worked out several other variants and applications.

The observation that the Hamilton–Jacobi equation can be obtained from the heat equation after proper rescaling is quite old, and it is now a classical exercise in the theory of viscosity solutions. **Bobkov, Gentil and Ledoux** observed that this could constitute a bridge between the two main existing strategies for logarithmic Sobolev inequalities. Links with the theory of large deviations have been investigated.

</div>

## Part III: Synthetic Treatment of Ricci Curvature

In Chapter 17 it was proven that lower Ricci curvature bounds influence displacement convexity properties of certain classes of functionals, and moreover that these properties **characterize** lower Ricci curvature bounds. One may therefore "transform the theorem into a definition" and express the property "Ricci curvature is bounded below by $K$" in terms of certain displacement convexity properties. This approach is **synthetic**: it does not rely on analytic computations (of the Ricci tensor), but rather on qualitative properties of certain objects which play an important role in geometric arguments.

This point of view has the advantage of applying to nonsmooth spaces, just as lower (or upper) sectional curvature bounds can be defined in nonsmooth metric spaces by Alexandrov's method. An important difference, however, is that the notion of generalized Ricci curvature will be defined not only in terms of distances, but also in terms of reference measures. The basic object will therefore not be a metric space, but a **metric-measure space** — a metric space equipped with a reference measure.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convention for Part III)</span></p>

Throughout Part III, geodesics are *constant-speed, minimizing geodesics*.

</div>

**Overview of Part III:**

- **Chapters 26 and 27** are preparatory. Chapter 26 illustrates the meaning of the word "synthetic" via a simple example about convex functions. Chapter 27 covers convergence of metric-measure spaces.
- **Chapter 28** considers optimal transport in possibly nonsmooth spaces and establishes various stability properties of optimal transport under convergence of metric-measure spaces.
- **Chapter 29** presents a synthetic definition of the curvature-dimension condition $\mathrm{CD}(K, N)$ in a nonsmooth context and proves that it is stable. A geometric consequence: if a Riemannian manifold is the limit of a sequence of $\mathrm{CD}(K, N)$ Riemannian manifolds, then it, too, satisfies $\mathrm{CD}(K, N)$.
- **Chapter 30** presents the state of the art concerning the qualitative geometric and analytic properties enjoyed by metric-measure spaces satisfying curvature-dimension conditions, with complete proofs.

## Chapter 26: Analytic and Synthetic Points of View

### Two Definitions of Convexity

The present chapter is devoted to a simple pedagogical illustration of the opposition between the "analytic" and "synthetic" points of view. Consider the following two definitions for convexity on $\mathbb{R}^n$:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Convexity — Analytic)</span></p>

A convex function is a function $\varphi$ which is twice continuously differentiable, and whose Hessian $\nabla\_x^2 \varphi$ is nonnegative at each $x \in \mathbb{R}^n$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Convexity — Synthetic)</span></p>

A convex function is a function $\varphi$ such that for all $x, y \in \mathbb{R}^n$ and $\lambda \in [0, 1]$,

$$\varphi\bigl((1 - \lambda)x + \lambda y\bigr) \le (1 - \lambda)\, \varphi(x) + \lambda\, \varphi(y).$$

</div>

### Comparison of the Two Definitions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Five Key Observations)</span></p>

**1. Generality.** When applied to $C^2$ functions, both definitions coincide, but the synthetic definition is obviously *more general*. It is expressed without any reference to second differentiability; for instance, $\varphi(x) = \lvert x \rvert$ satisfies the synthetic definition but not the analytic one.

**2. Stability.** The synthetic definition is *more stable*. Take a sequence $(\varphi\_k)\_{k \in \mathbb{N}}$ of convex functions converging to some other function $\varphi$. To pass to the limit in the analytic definition, one would need very strong convergence (say in $C^2(\mathbb{R}^n)$). On the other hand, one can pass to the limit in the synthetic definition assuming only pointwise convergence. So the synthetic definition is much easier to "pass to the limit in" — even if the limit is known to be smooth.

**3. Starting point.** The synthetic definition is also a better *starting point* for studying properties of convex functions. Throughout these notes, when convexity was used, it was via the synthetic definition, not the analytic one.

**4. Practical verification.** On the other hand, if one is given a particular function (by its explicit analytic expression) and asked whether it is convex, the analytic definition is often more workable: just *compute* the second derivative and check its sign. This will work most easily for the huge majority of candidate convex functions encountered in practice.

**5. Local vs. global.** The analytic definition is naturally *local*, while the synthetic definition is global (and probably this is related to the fact that it is so difficult to check). In particular, the analytic definition involves an object (the second derivative) which can quantify the "strength of convexity" *at each point*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analytic vs. Synthetic — Summary)</span></p>

The analytic definition (based on the computation of certain objects) and the synthetic definition (based on certain qualitative properties which are the basis for proofs) exhibit a pattern that is typical in geometry:

- **Synthetic definitions** tend to be more general and more stable; they should be usable directly to prove interesting results. On the other hand, they may be difficult to check in practice, and they are usually less precise (and less "local") than analytic definitions.
- **Analytic definitions** are conceptually simple and easy to verify via computation, but may sometimes lead to very difficult computations. The synthetic approach is often lighter but requires better intuition and clever elementary arguments.

</div>

### Curvature: Analytic and Synthetic

In Riemannian geometry, curvature is traditionally defined by a purely analytic approach: from the Riemannian scalar product one can compute several functions called sectional curvature, Ricci curvature, scalar curvature, etc. For any $x \in M$, the sectional curvature at point $x$ is a *function* which associates to each 2-dimensional plane $P \subset T\_x M$ a number $\sigma\_x(P)$, expressed in terms of a basis of $P$ and a certain combination of derivatives of the metric at $x$. Intuitively, $\sigma\_x(P)$ measures the speed of convergence of geodesics that start at $x$, with velocities spanning the plane $P$. A space is said to have nonnegative sectional curvature if $\sigma\_x(P)$ is nonnegative, for all $P$ and for all $x$.

### Alexandrov Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Alexandrov Space)</span></p>

A geodesic space $(\mathcal{X}, d)$ is said to have **Alexandrov curvature bounded below by $K$**, or to be an **Alexandrov space** with curvature bounded below by $K$, if triangles in $\mathcal{X}$ are no more "skinny" than reference triangles drawn on the model space with constant curvature $K$.

More precisely: if $xyz$ is a triangle in $\mathcal{X}$ and $x\_0 y\_0 z\_0$ is a triangle drawn on the model space with $d(x\_0, y\_0) = d(x, y)$, $d(y\_0, z\_0) = d(y, z)$, $d(z\_0, x\_0) = d(z, x)$, $x'$ is a midpoint between $y$ and $z$, and $x'\_0$ a midpoint between $y\_0$ and $z\_0$, then one should have $d(x\_0, x'\_0) \le d(x, x')$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Model Spaces for Curvature $K$)</span></p>

The model spaces with constant curvature $K$ are:

- the sphere $S^2(1/\sqrt{K})$ with radius $R = 1/\sqrt{K}$, if $K > 0$;
- the plane $\mathbb{R}^2$, if $K = 0$;
- the hyperbolic space $\mathbb{H}(1/\sqrt{\lvert K \rvert})$ with "hyperbolic radius" $R = 1/\sqrt{\lvert K \rvert}$, if $K < 0$; this can be realized as the half-plane $\mathbb{R} \times (0, +\infty)$, equipped with the metric $g\_{(x,y)}(dx\, dy) = (dx^2 + dy^2)/(\lvert K \rvert y^2)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Properties of Alexandrov Spaces)</span></p>

The Alexandrov definition shares exactly the same advantages and drawbacks as the synthetic definition of convexity:

- It is **equivalent** to the analytic definition when applied to a smooth Riemannian manifold.
- It is **more general**, since it applies, e.g., to a cone (the two-dimensional cone embedded in $\mathbb{R}^3$, constructed over a circular basis).
- It is **more stable**: in particular, it passes to the limit under **Gromov–Hausdorff convergence**.
- It is useful as a **starting point** for many properties involving sectional curvature.
- It is in general **difficult to check** directly, and there is no associated notion of *curvature* (in "Alexandrov space of nonnegative curvature", the words "nonnegative" and "curvature" do not make sense independently of each other).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Synthetic Sectional vs. Ricci Curvature)</span></p>

The generalized notion of *sectional* curvature bounds (Alexandrov spaces) has been extensively explored, and quite strong results have been obtained concerning the geometric and analytic implications of such bounds. But until recently, the synthetic treatment of lower *Ricci* curvature bounds stood as an open problem. The thesis developed in the rest of these notes is that **optimal transport provides a solution to this problem**.

</div>

### Bibliographical Notes on Chapter 26

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bibliographical Notes)</span></p>

Alexandrov spaces are also called **CAT** spaces, in honor of Cartan, Alexandrov and Toponogov. The terminology of CAT space is often restricted to Alexandrov spaces with *upper* sectional bounds. So a $\mathrm{CAT}(K)$ space typically means an Alexandrov space with "sectional curvature bounded above by $K$". In the sequel, only lower curvature bounds are considered.

Good sources for Alexandrov spaces include the book by **Burago, Burago and Ivanov** and the synthesis paper by **Burago, Gromov and Perelman**. There is also a notion of "approximate" Alexandrov spaces, called $\mathrm{CAT}\_\delta(K)$ spaces, in which a fixed "resolution error" $\delta$ is allowed in the defining inequalities. (In the case of upper curvature bounds, this notion has applications to the theory of hyperbolic groups.) Such spaces are not necessarily geodesic, not even length spaces; they can be discrete. A pair of points $(x\_0, x\_1)$ will not necessarily admit a midpoint, but there will be a $\delta$-approximate midpoint (that is, $m$ such that $\lvert d(x\_0, m) - d(x\_0, x\_1)/2 \rvert \le \delta$, $\lvert d(x\_1, m) - d(x\_0, x\_1)/2 \rvert \le \delta$).

The open problem of developing a satisfactory synthetic treatment of Ricci curvature bounds was discussed by **Gromov**, and more recently by **Cheeger and Colding**.

</div>

### The Wasserstein Space over Alexandrov Spaces

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Alexandrov Structure of $P\_2(\mathcal{X})$)</span></p>

It was shown by **Lott and Villani** that if $M$ is a compact Riemannian manifold then $M$ has nonnegative sectional curvature if and only if $P\_2(M)$ is an Alexandrov space with nonnegative curvature. Independently, **Sturm** proved the more general result: $\mathcal{X}$ is an Alexandrov space with nonnegative curvature if and only if $P\_2(\mathcal{X})$ is an Alexandrov space with nonnegative curvature.

All this suggested that the notion of Alexandrov curvature matched well with optimal transport. However, at the same time, Sturm showed that if $\mathcal{X}$ is not nonnegatively curved, then $P\_2(\mathcal{X})$ cannot be an Alexandrov space (morally, the curvature takes all values in $(-\infty, +\infty)$ at Dirac masses). To circumvent this obstacle, **Ohta** suggested replacing the Alexandrov property by a weaker condition known as **2-uniform convexity**, used in Banach space theory. **Savaré** came up independently with a similar idea.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2-Uniform Convexity)</span></p>

A geodesic space $(\mathcal{X}, d)$ is **2-uniform** with a constant $S \ge 1$ if, given any three points $x, y, z \in \mathcal{X}$, and a minimizing geodesic $\gamma$ joining $y$ to $z$, one has

$$\forall t \in [0, 1], \qquad d(x, \gamma_t)^2 \ge (1 - t)\, d(x, y)^2 + t\, d(x, z)^2 - S^2\, t(1-t)\, d(y, z)^2.$$

When $S = 1$ this is exactly the inequality defining nonnegative Alexandrov curvature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Results of Ohta and Savaré)</span></p>

**Ohta** showed that (a) any Alexandrov space with curvature bounded below is locally 2-uniform; (b) $\mathcal{X}$ is 2-uniformly smooth with constant $S$ if and only if $P\_2(\mathcal{X})$ is 2-uniformly smooth with the same constant $S$. He further uses the 2-uniform smoothness to study the structure of tangent cones in $P\_2(\mathcal{X})$. Both **Ohta** and **Savaré** use these inequalities to construct gradient flows in the Wasserstein space over Alexandrov spaces.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Open Problems on $P\_2(M)$)</span></p>

Even when $\mathcal{X}$ is a smooth manifold with nonnegative curvature, many technical issues remain open about the structure of $P\_2(M)$ as an Alexandrov space. For instance, the notion of the tangent cone used in the above-mentioned works is the one involving the space of directions, but does it coincide with the notion derived from rescaling and Gromov–Hausdorff convergence? How should one define a regular point of $P\_2(M)$: as an absolutely continuous measure, or an absolutely continuous measure with positive density? Do these measures form a totally convex set? Do singular measures form a small set in some sense? Can one define and use quasi-geodesics in $P\_2(M)$?

</div>

## Chapter 27: Convergence of Metric-Measure Spaces

The central question in this chapter is: What does it mean to say that a metric-measure space $(\mathcal{X}, d\_\mathcal{X}, \nu\_\mathcal{X})$ is "close" to another metric-measure space $(\mathcal{Y}, d\_\mathcal{Y}, \nu\_\mathcal{Y})$? The answer should be as "intrinsic" as possible, depending only on the metric-measure properties of $\mathcal{X}$ and $\mathcal{Y}$.

### Hausdorff Topology

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hausdorff Distance)</span></p>

If $\mathcal{X}$ and $\mathcal{Y}$ are two compact subsets of a metric space $(\mathcal{Z}, d)$, their **Hausdorff distance** is

$$d_H(\mathcal{X}, \mathcal{Y}) = \max\Bigl(\sup_{x \in \mathcal{X}} d(x, \mathcal{Y}),\; \sup_{y \in \mathcal{Y}} d(y, \mathcal{X})\Bigr),$$

where $d(a, B) = \inf\lbrace d(a,b);\; b \in B \rbrace$ is the distance from the point $a$ to the set $B$.

The statement "$d\_H(\mathcal{X}, \mathcal{Y}) \le r$" means: if we *inflate* (enlarge) $\mathcal{Y}$ by a distance $r$, the resulting set covers $\mathcal{X}$; and conversely if we inflate $\mathcal{X}$ by a distance $r$, the resulting set covers $\mathcal{Y}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analogy with Prokhorov Distance)</span></p>

The Hausdorff distance can be thought of as a set-theoretical analog of the Prokhorov distance between probability measures. This becomes apparent when rewriting them in parallel:

$$d_H(A,B) = \inf\bigl\lbrace r > 0;\; A \subset B^{r]} \text{ and } B \subset A^{r]}\bigr\rbrace,$$

$$d_P(\mu, \nu) = \inf\bigl\lbrace r > 0;\; \forall C,\; \mu[C] \le \nu[C^{r]}] + r \text{ and } \nu[C] \le \mu[C^{r]}] + r\bigr\rbrace,$$

where $C^{r]}$ is the set of all points whose distance to $C$ is no more than $r$.

</div>

### Correspondences

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Correspondence)</span></p>

A **correspondence** between two sets $\mathcal{X}$ and $\mathcal{Y}$ is a subset $\mathcal{R}$ of $\mathcal{X} \times \mathcal{Y}$ such that each $x \in \mathcal{X}$ is in correspondence with at least one $y$, and each $y \in \mathcal{Y}$ is in correspondence with at least one $x$. If $(x, y) \in \mathcal{R}$, then $x$ and $y$ are said to be in correspondence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Couplings vs. Correspondences)</span></p>

While the Prokhorov distance can be defined in terms of **couplings**, the Hausdorff distance can be defined in terms of **correspondences**. The two formulas are strikingly similar:

$$d_P(\mu, \nu) = \inf\bigl\lbrace r > 0;\; \exists \text{ coupling } (X,Y) \text{ of } (\mu,\nu);\; \mathbb{P}[d(X,Y) > r] \le r\bigr\rbrace;$$

$$d_H(\mathcal{X}, \mathcal{Y}) = \inf\bigl\lbrace r > 0;\; \exists \text{ correspondence } \mathcal{R} \text{ in } \mathcal{X} \times \mathcal{Y};\; \forall (x,y) \in \mathcal{R},\; d(x,y) \le r\bigr\rbrace.$$

Like their probabilistic counterparts, correspondences can be **glued** together: if $\mathcal{R}\_{12}$ is a correspondence between $\mathcal{X}\_1$ and $\mathcal{X}\_2$, and $\mathcal{R}\_{23}$ is a correspondence between $\mathcal{X}\_2$ and $\mathcal{X}\_3$, one may define a correspondence $\mathcal{R}\_{13} = \mathcal{R}\_{23} \circ \mathcal{R}\_{12}$ between $\mathcal{X}\_1$ and $\mathcal{X}\_3$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hausdorff Distance Is a Metric)</span></p>

The Hausdorff distance $d\_H$ is indeed a distance on compact subsets of a given metric space $\mathcal{Z}$:

1. It is obviously symmetric.
2. Because it is defined on compact (hence bounded) sets, the infimum is a nonnegative finite number.
3. If $d\_H(\mathcal{X}, \mathcal{Y}) = 0$, then any $x \in \mathcal{X}$ satisfies $d(x, \mathcal{Y}) = 0$, and since $\mathcal{Y}$ is closed, this implies $\mathcal{X} = \mathcal{Y}$.
4. The triangle inequality follows easily from gluing correspondences.

One may define the metric space $\mathcal{H}(\mathcal{Z})$ as the space of all compact subsets of $\mathcal{Z}$, equipped with the Hausdorff distance. If $\mathcal{Z}$ is compact, then $\mathcal{H}(\mathcal{Z})$ is also a compact metric space.

</div>

### The Gromov–Hausdorff Distance

The Hausdorff distance only compares subsets of a given underlying space. To compare *different* metric spaces with possibly nothing in common, one works with *isometry classes*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Isometry)</span></p>

If $(\mathcal{X}, d)$ and $(\mathcal{X}', d')$ are two metric spaces, a map $f : \mathcal{X} \to \mathcal{X}'$ is called an **isometry** if:

- (a) it preserves distances: for all $x, y \in \mathcal{X}$, $d'(f(x), f(y)) = d(x, y)$;
- (b) it is surjective: for any $x' \in \mathcal{X}'$ there is $x \in \mathcal{X}$ with $f(x) = x'$.

An isometry is automatically injective, hence a bijection, and $f^{-1}$ is also an isometry. Two metric spaces are said to be **isometric** if there exists an isometry between them. An **isometry class** $\overline{\mathcal{X}}$ is the set of all metric spaces isometric to a given space $\mathcal{X}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gromov–Hausdorff Distance)</span></p>

If $(\mathcal{X}, d\_\mathcal{X})$ and $(\mathcal{Y}, d\_\mathcal{Y})$ are two compact metric spaces, the **Gromov–Hausdorff distance** is

$$d_{GH}(\mathcal{X}, \mathcal{Y}) = \inf\; d_H(\mathcal{X}', \mathcal{Y}'),$$

where the infimum is taken over all *isometric embeddings* $\mathcal{X}', \mathcal{Y}'$ of $\mathcal{X}$ and $\mathcal{Y}$ into a common metric space $\mathcal{Z}$; that is, $\mathcal{X}'$ is isometric to $\mathcal{X}$, $\mathcal{Y}'$ is isometric to $\mathcal{Y}$, and both $\mathcal{X}'$ and $\mathcal{Y}'$ are subspaces of $\mathcal{Z}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Metric Coupling)</span></p>

When one chooses $\mathcal{Z} = \mathcal{X}' \cup \mathcal{Y}'$, the pair $(\mathcal{X}', \mathcal{Y}')$ is called a **metric coupling** of the abstract spaces $(\mathcal{X}, \mathcal{Y})$. The metric on $\mathcal{X}', \mathcal{Y}'$ has to be the metric induced by $\mathcal{Z}$.

</div>

### Representation by Semi-Distances

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(27.1 — Metric Couplings as Semi-Distances)</span></p>

Let $(\mathcal{X}, d\_\mathcal{X})$ and $(\mathcal{Y}, d\_\mathcal{Y})$ be two disjoint metric spaces, and let $\mathcal{X} \sqcup \mathcal{Y}$ be their union. Then:

**(i)** Let $(\mathcal{X}', \mathcal{Y}')$ be a metric coupling of $\mathcal{X}$ and $\mathcal{Y}$; let $f : \mathcal{X} \to \mathcal{X}'$ and $g : \mathcal{Y} \to \mathcal{Y}'$ be isometries, and let $(\mathcal{Z}, d\_\mathcal{Z})$ be the ambient metric space. Then

$$d(a, b) = \begin{cases} d_\mathcal{X}(a,b) & \text{if } a, b \in \mathcal{X} \\\\ d_\mathcal{Y}(a,b) & \text{if } a, b \in \mathcal{Y} \\\\ d_\mathcal{Z}(f(a), g(b)) & \text{if } a \in \mathcal{X},\; b \in \mathcal{Y} \\\\ d_\mathcal{Z}(g(a), f(b)) & \text{if } a \in \mathcal{Y},\; b \in \mathcal{X} \end{cases}$$

is a semi-distance on $\mathcal{X} \sqcup \mathcal{Y}$, whose restriction to $\mathcal{X} \times \mathcal{X}$ (resp. $\mathcal{Y} \times \mathcal{Y}$) coincides with $d\_\mathcal{X}$ (resp. $d\_\mathcal{Y}$).

**(ii)** Conversely, let $d$ be a semi-distance on $\mathcal{X} \sqcup \mathcal{Y}$ whose restriction to $\mathcal{X} \times \mathcal{X}$ (resp. $\mathcal{Y} \times \mathcal{Y}$) coincides with $d\_\mathcal{X}$ (resp. $d\_\mathcal{Y}$). Define the equivalence relation $x \mathcal{R}\, x' \iff d(x, x') = 0$ and set $\mathcal{Z} = (\mathcal{X} \sqcup \mathcal{Y})/d := (\mathcal{X} \sqcup \mathcal{Y})/\mathcal{R}$, with $d\_\mathcal{Z}(\overline{a}, \overline{b}) = d(a,b)$. Then $x \to \overline{x}$ is an isometric embedding of $\mathcal{X}$ into $(\mathcal{Z}, d\_\mathcal{Z})$, and similarly for $\mathcal{Y}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(27.2 — Metric Gluing Lemma)</span></p>

Let $(\mathcal{X}\_1, d\_1)$, $(\mathcal{X}\_2, d\_2)$, $(\mathcal{X}\_3, d\_3)$ be three abstract compact metric spaces. If $(\mathcal{X}'\_1, \mathcal{X}'\_2)$ is a metric coupling of $(\mathcal{X}\_1, \mathcal{X}\_2)$ and $(\mathcal{X}''\_2, \mathcal{X}''\_3)$ is a metric coupling of $(\mathcal{X}\_2, \mathcal{X}\_3)$, then there is a triple of metric spaces $(\widetilde{\mathcal{X}}\_1, \widetilde{\mathcal{X}}\_2, \widetilde{\mathcal{X}}\_3)$, all subspaces of a common metric space $(\mathcal{Z}, d\_\mathcal{Z})$, such that $(\widetilde{\mathcal{X}}\_1, \widetilde{\mathcal{X}}\_2)$ is isometric (as a coupling) to $(\mathcal{X}'\_1, \mathcal{X}'\_2)$, and $(\widetilde{\mathcal{X}}\_2, \widetilde{\mathcal{X}}\_3)$ is isometric to $(\mathcal{X}''\_2, \mathcal{X}''\_3)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Theorem 27.2)</span></p>

By Proposition 27.1, the metric coupling $(\mathcal{X}'\_1, \mathcal{X}'\_2)$ may be thought of as a semi-distance $d\_{12}$ on $\mathcal{X}\_1 \sqcup \mathcal{X}\_2$; similarly, $(\mathcal{X}''\_2, \mathcal{X}''\_3)$ as a semi-distance $d\_{23}$ on $\mathcal{X}\_2 \sqcup \mathcal{X}\_3$. For $x\_1 \in \mathcal{X}\_1$ and $x\_3 \in \mathcal{X}\_3$, define

$$d_{13}(x_1, x_3) = \inf_{x_2 \in \mathcal{X}_2}\bigl[d_{12}(x_1, x_2) + d_{23}(x_2, x_3)\bigr].$$

This induces a semi-distance on $\mathcal{X}\_1 \sqcup \mathcal{X}\_2 \sqcup \mathcal{X}\_3$, and one defines $\mathcal{Z} = (\mathcal{X}\_1 \sqcup \mathcal{X}\_2 \sqcup \mathcal{X}\_3)/d$ as in Proposition 27.1.

</div>

### Representation by Approximate Isometries

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Distortion of a Correspondence)</span></p>

Given a correspondence $\mathcal{R} \subset \mathcal{X} \times \mathcal{Y}$, its **distortion** is defined by

$$\operatorname{dis}(\mathcal{R}) = \sup_{(x,y),(x',y') \in \mathcal{R}} \bigl\lvert d_\mathcal{Y}(y, y') - d_\mathcal{X}(x, x') \bigr\rvert.$$

Then it can be shown that

$$d_{GH}(\mathcal{X}, \mathcal{Y}) = \frac{1}{2}\;\inf\;\operatorname{dis}(\mathcal{R}),$$

where the infimum is over all correspondences $\mathcal{R}$ between $\mathcal{X}$ and $\mathcal{Y}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\varepsilon$-Isometry / Approximate Isometry)</span></p>

An **$\varepsilon$-isometry** between $(\mathcal{X}, d\_\mathcal{X})$ and $(\mathcal{Y}, d\_\mathcal{Y})$ is a map $f : \mathcal{X} \to \mathcal{Y}$ that is "almost an isometry":

- (a') it almost preserves distances: for all $x, x'$ in $\mathcal{X}$, $\bigl\lvert d(f(x), f(x')) - d(x, x') \bigr\rvert \le \varepsilon$;
- (b') it is almost surjective: $\forall y \in \mathcal{Y}$, $\exists\, x \in \mathcal{X}$; $d(f(x), y) \le \varepsilon$.

In particular, $d\_H(f(\mathcal{X}), \mathcal{Y}) \le \varepsilon$.

Heuristically, an $\varepsilon$-isometry is a map that you can't distinguish from an isometry if you are short-sighted, i.e. if you measure all distances with a possible error of about $\varepsilon$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gromov–Hausdorff Distance via $\varepsilon$-Isometries)</span></p>

It can be shown that

$$\frac{2}{3}\, d_{GH}(\mathcal{X}, \mathcal{Y}) \;\le\; \inf\bigl\lbrace \varepsilon;\; \exists f\; \varepsilon\text{-isometry } \mathcal{X} \to \mathcal{Y}\bigr\rbrace \;\le\; 2\, d_{GH}(\mathcal{X}, \mathcal{Y}).$$

Moreover, any approximate isometry admits an **approximate inverse**: if $f$ is an $\varepsilon$-isometry $\mathcal{X} \to \mathcal{Y}$, then there is a $(4\varepsilon)$-isometry $f' : \mathcal{Y} \to \mathcal{X}$ such that for all $x \in \mathcal{X}$, $y \in \mathcal{Y}$,

$$d_\mathcal{X}\bigl(f' \circ f(x),\, x\bigr) \le 3\varepsilon, \qquad d_\mathcal{Y}\bigl(f \circ f'(y),\, y\bigr) \le \varepsilon.$$

Such $f'$ is called an **$\varepsilon$-inverse** of $f$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(27.4 — Approximate Isometries Converge to Isometries)</span></p>

Let $\mathcal{X}$ and $\mathcal{Y}$ be two compact metric spaces, and for each $k \in \mathbb{N}$ let $f\_k$ be an $\varepsilon\_k$-isometry, where $\varepsilon\_k \to 0$. Then, up to extraction of a subsequence, $f\_k$ converges to an isometry.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Lemma 27.4)</span></p>

Introduce a dense subset $S$ of $\mathcal{X}$. For each $x \in S$, the sequence $(f\_k(x))$ is valued in the compact set $\mathcal{Y}$, so up to extraction it converges to some $f(x) \in \mathcal{Y}$. By a diagonal extraction, we may assume $f\_k(x) \to f(x)$ for all $x \in S$. Passing to the limit in the inequality satisfied by $f\_k$, we see that $f$ is distance-preserving. By uniform continuity, $f$ extends to a distance-preserving map $\mathcal{X} \to \mathcal{Y}$. Similarly, there is a distance-preserving map $g : \mathcal{Y} \to \mathcal{X}$ obtained as a limit of approximate inverses. Since $\mathcal{X}$ is compact, $g \circ f$ is a bijection, so both $f$ and $g$ are isometries.

</div>

### The Gromov–Hausdorff Space

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($d\_{GH}$ Is a Honest Distance)</span></p>

The Gromov–Hausdorff distance $d\_{GH}$ is a true distance on the set of isometry classes of compact metric spaces:

1. It is obviously symmetric.
2. $d\_{GH}(\mathcal{X}, \mathcal{Y})$ is always finite (equip $\mathcal{X} \sqcup \mathcal{Y}$ with $d(x,y) = D > 0$ for $(x,y) \in \mathcal{X} \times \mathcal{Y}$, choosing $D$ large enough).
3. $d\_{GH}(\mathcal{X}, \mathcal{X}) = 0$. Conversely, if $d\_{GH}(\mathcal{X}, \mathcal{Y}) = 0$, then by Lemma 27.4, $\mathcal{X}$ and $\mathcal{Y}$ are isometric.
4. The triangle inequality follows from the metric gluing lemma — just as the triangle inequality for the Wasserstein distance was a consequence of the probabilistic gluing lemma.

The set $(\mathcal{GH}, d\_{GH})$ of all classes of isometry of compact metric spaces, equipped with the Gromov–Hausdorff distance, is itself a complete separable metric space. An explicit countable dense subset is provided by the family of all finite subsets of $\mathbb{N}$ with rational-valued distances.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(27.6 — Gromov–Hausdorff Convergence)</span></p>

Let $(\mathcal{X}\_k)\_{k \in \mathbb{N}}$ be a sequence of compact metric spaces, and let $\mathcal{X}$ be a compact metric space. Then $\mathcal{X}\_k$ **converges to $\mathcal{X}$ in the Gromov–Hausdorff topology** if any one of the following three equivalent statements is satisfied:

- (i) $d\_{GH}(\mathcal{X}\_k, \mathcal{X}) \longrightarrow 0$;
- (ii) There exist correspondences $\mathcal{R}\_k$ between $\mathcal{X}\_k$ and $\mathcal{X}$ such that $\operatorname{dis} \mathcal{R}\_k \longrightarrow 0$;
- (iii) There exist $\varepsilon\_k$-isometries $f\_k : \mathcal{X}\_k \to \mathcal{X}$, for some sequence $\varepsilon\_k \to 0$.

This convergence is denoted $\mathcal{X}\_k \xrightarrow{GH} \mathcal{X}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(27.7 — Mr. Magoo Topology)</span></p>

Two spaces are close in Gromov–Hausdorff topology if they look the same to a short-sighted person. Lott coined the expression *Mr. Magoo topology* to convey this idea.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(27.8 — Discontinuous Approximate Isometries)</span></p>

It is important to allow the approximate isometries to be discontinuous. Two spaces $\mathcal{X}$ and $\mathcal{Y}$ may be very close in Gromov–Hausdorff topology although there is no continuous map $\mathcal{X} \to \mathcal{Y}$. (For example, a balloon with a very small handle is very close to a balloon without a handle.) However, a celebrated convergence theorem by Gromov shows that such behavior is ruled out by bounds on the curvature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Passing Geometric Statements to the Limit)</span></p>

The Gromov–Hausdorff topology enjoys the nice property that *any geometric statement which can be expressed in terms of the distances between a finite number of points automatically passes to the limit*. For example, "Any pair $(x, y)$ of points in $\mathcal{X}$ admits a midpoint" (which characterizes a geodesic space under a completeness assumption) only involves configurations of three points, so it passes to the limit. Then geodesics can be reconstructed from successive midpoints.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(27.9 — Convergence of Geodesic Spaces)</span></p>

Let $(\mathcal{X}\_k)\_{k \in \mathbb{N}}$ be a sequence of compact geodesic spaces converging to $\mathcal{X}$ in Gromov–Hausdorff topology; then $\mathcal{X}$ is a geodesic space. Moreover, if $f\_k$ is an $\varepsilon\_k$-isometry $\mathcal{X}\_k \to \mathcal{X}$, and $\gamma\_k$ is a geodesic curve in $\mathcal{X}\_k$ such that $f\_k \circ \gamma\_k$ converges to some curve $\gamma$ in $\mathcal{X}$, then $\gamma$ is a geodesic.

</div>

### Gromov–Hausdorff Topology and Nets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\varepsilon$-Net and Total Boundedness)</span></p>

Given $\varepsilon > 0$, a set $\mathcal{N}$ in a metric space $(\mathcal{X}, d)$ is called an **$\varepsilon$-net** (in $\mathcal{X}$) if the enlargement $S^{\varepsilon]}$ covers $\mathcal{X}$; in other words, for any $x \in \mathcal{X}$ there is $y \in \mathcal{N}$ such that $d(x, y) \le \varepsilon$.

A metric space $\mathcal{X}$ is said to be **totally bounded** if for any $\varepsilon > 0$ it can be covered by a finite number of balls of radius $\varepsilon$. The function $\varepsilon \longmapsto N(\varepsilon)$ (the minimal number of such balls) is a "modulus of total boundedness".

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(GH Convergence Reduces to $\varepsilon$-Nets)</span></p>

If $\mathcal{X}$ is compact, it admits finite $\varepsilon$-nets for all $\varepsilon > 0$, so it can be approximated in Gromov–Hausdorff topology by a sequence of finite sets. In fact, $\mathcal{X}\_n \longrightarrow \mathcal{X}$ in the Gromov–Hausdorff topology if and only if for any $\varepsilon > 0$ there exists a finite $\varepsilon$-net $\lbrace x\_1, \ldots, x\_k \rbrace$ in $\mathcal{X}$, and for $n$ large enough there is an $\varepsilon$-net $\lbrace x\_1^{(n)}, \ldots, x\_k^{(n)} \rbrace$ in $\mathcal{X}\_n$, and for all $j \le k$, $x\_j^{(n)} \longrightarrow x\_j$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(27.10 — Compactness Criterion in Gromov–Hausdorff Topology)</span></p>

A family $\mathcal{F}$ of compact metric spaces is **precompact** in the Gromov–Hausdorff topology if and only if it is **uniformly totally bounded**, in the sense that for any $\varepsilon > 0$ there is $N = N(\varepsilon)$ such that any $\mathcal{X} \in \mathcal{F}$ contains an $\varepsilon$-net of cardinality at most $N$.

</div>

### Noncompact Spaces

For noncompact spaces, the Gromov–Hausdorff distance may be infinite or of limited use (just as uniform convergence is too strong for noncompact spaces, replaced by *locally uniform* convergence). The idea is to exhaust $\mathcal{X}$ by compact sets $K^{(\ell)}$ such that each $K^{(\ell)}$ is a Gromov–Hausdorff limit of corresponding compact sets $K\_k^{(\ell)} \subset \mathcal{X}\_k$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(27.11 — Local Gromov–Hausdorff Convergence)</span></p>

Let $(\mathcal{X}\_k)\_{k \in \mathbb{N}}$ be a family of Polish spaces, and let $\mathcal{X}$ be another Polish space. It is said that $\mathcal{X}\_k$ converges to $\mathcal{X}$ in the **local Gromov–Hausdorff topology** if there are nondecreasing sequences of compact sets $(K\_k^{(\ell)})\_{\ell \in \mathbb{N}}$ in each $\mathcal{X}\_k$, and $(K^{(\ell)})\_{\ell \in \mathbb{N}}$ in $\mathcal{X}$, such that:

- (i) $\bigcup\_\ell K^{(\ell)}$ is dense in $\mathcal{X}$;
- (ii) for each fixed $\ell$, $K\_k^{(\ell)}$ converges to $K^{(\ell)}$ in Gromov–Hausdorff sense as $k \to \infty$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(27.12 — Geodesic Local Gromov–Hausdorff Convergence)</span></p>

Let $(\mathcal{X}\_k)\_{k \in \mathbb{N}}$ be a family of geodesic Polish spaces, and let $\mathcal{X}$ be a Polish space. $\mathcal{X}\_k$ converges to $\mathcal{X}$ in the **geodesic local Gromov–Hausdorff topology** if in addition to conditions (i) and (ii) of Definition 27.11:

- (iii) For each $\ell \in \mathbb{N}$, there exists $\ell'$ such that all geodesics starting and ending in $K\_k^{(\ell)}$ have their image contained in $K\_k^{(\ell')}$.

Then $\mathcal{X}$ is automatically a geodesic space.

</div>

### Pointed Metric Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pointed Metric Space)</span></p>

A **pointed metric space** consists of a triple $(\mathcal{X}, d, \star)$, where $(\mathcal{X}, d)$ is a metric space and $\star$ is some point in $\mathcal{X}$ (the "reference point" or "base point"). A **pointed isometry** between $(\mathcal{X}, \star\_\mathcal{X})$ and $(\mathcal{Y}, \star\_\mathcal{Y})$ is an isometry sending $\star\_\mathcal{X}$ to $\star\_\mathcal{Y}$.

For a geodesic space, being boundedly compact (all closed balls are compact) is equivalent to being locally compact. In the sequel, the basic regularity assumption when considering pointed Gromov–Hausdorff convergence will be **local compactness**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(27.13 — Pointed Gromov–Hausdorff Convergence)</span></p>

Let $(\mathcal{X}\_k, \star\_k)$ be a sequence of pointed locally compact geodesic Polish spaces, and let $(\mathcal{X}, \star)$ be a pointed locally compact Polish space. Then $\mathcal{X}\_k$ converges to $\mathcal{X}$ in the **pointed Gromov–Hausdorff topology** if any one of the following equivalent statements is satisfied:

- (i) There is a sequence $R\_k \to \infty$ such that $d\_{pGH}\bigl(B[\star\_k, R\_k],\, B[\star, R\_k]\bigr) \longrightarrow 0$;
- (ii) There is a sequence $R\_k \to \infty$, and there are pointed correspondences $\mathcal{R}\_k$ between $B[\star\_k, R\_k]$ and $B[\star, R\_k]$ such that $\operatorname{dis}(\mathcal{R}\_k) \longrightarrow 0$;
- (iii) There are sequences $R\_k \to \infty$ and $\varepsilon\_k \to 0$, and pointed $\varepsilon\_k$-isometries $f\_k : B[\star\_k, R\_k] \to B[\star, R\_k]$ with $\varepsilon\_k \to 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(27.15 — Blow-Up)</span></p>

Let $M$ be a Riemannian manifold of dimension $n$, and $x$ a point in $M$. For each $k$, consider the pointed metric space $\mathcal{X}\_k = (M, k\, d, x)$, where $kd$ is the original geodesic distance on $M$ dilated by a factor $k$. Then $\mathcal{X}\_k$ converges in the pointed Gromov–Hausdorff topology to the tangent space $T\_x M$, pointed at 0 and equipped with the metric $g\_x$ (a Euclidean space). This is true as soon as $M$ is just differentiable at $x$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(27.16 — Tangent Cone)</span></p>

More generally, if $\mathcal{X}$ is a given metric space and $x$ is a point in $\mathcal{X}$, one can define the rescaled pointed spaces $\mathcal{X}\_k = (\mathcal{X}, k\, d, x)$; if this sequence converges in the pointed Gromov–Hausdorff topology to some metric space $\mathcal{Y}$, then $\mathcal{Y}$ is said to be the **tangent space**, or **tangent cone**, to $\mathcal{X}$ at $x$.

In many cases, the tangent cone coincides with the **metric cone** built on some length space $\Sigma$ (the space of tangent directions). By definition, the metric cone over $(B, d)$ is obtained by considering $B \times [0, \infty)$, gluing together all the points in the fiber $B \times \lbrace 0 \rbrace$, and equipping the resulting space with the cone metric: $d\_c((x,t),(y,s)) = \sqrt{t^2 + s^2 - 2ts \cos d(x,y)}$ when $d(x,y) \le \pi$, and $d\_c((x,t),(y,s)) = t + s$ when $d(x,y) > \pi$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(27.17 — $\ell^p$ Spaces Converge to $\ell^\infty$)</span></p>

For any $p \in [1, \infty)$, define the $\ell^p$ norm on $\mathbb{R}^n$ by $\lVert x \rVert\_{\ell^p} = (\sum \lvert x\_i \rvert^p)^{1/p}$; and let $\mathcal{X}\_p$ be the space $\mathbb{R}^n$ equipped with the $\ell^p$ norm, pointed at 0. Then as $p \to \infty$, $\mathcal{X}\_p$ converges in the pointed Gromov–Hausdorff topology to $\mathcal{X}\_\infty$, which is $\mathbb{R}^n$ equipped with the $\ell^\infty$ norm, $\lVert x \rVert\_{\ell^\infty} = \sup \lvert x\_i \rvert$.

In $\mathcal{X}\_p$, geodesics are segments of the form $(1-t)a + tb$ (nonbranching, unique). In contrast, geodesics in $\mathcal{X}\_\infty$ are branching and definitely nonunique (any two distinct points can be joined by uncountably many geodesic paths). This shows that neither the nonbranching property nor the uniqueness of geodesics are preserved under Gromov–Hausdorff convergence.

</div>

### Functional Analysis on Gromov–Hausdorff Converging Sequences

Many theorems about metric spaces still hold true, after appropriate modification, for *converging sequences* of metric spaces. In particular, analogues of Ascoli's theorem and Prokhorov's theorem can be formulated.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(27.20 — Ascoli Theorem in GH Converging Sequences)</span></p>

Let $(\mathcal{X}\_k)\_{k \in \mathbb{N}}$ be a sequence of compact metric spaces, converging in the Gromov–Hausdorff topology to some compact metric space $\mathcal{X}$, by means of $\varepsilon\_k$-approximations $f\_k : \mathcal{X}\_k \to \mathcal{X}$, admitting approximate inverses $f'\_k$; and let $(\mathcal{Y}\_k)\_{k \in \mathbb{N}}$ be another sequence of compact metric spaces converging to $\mathcal{Y}$, by means of $\varepsilon\_k$-approximations $g\_k : \mathcal{Y}\_k \to \mathcal{Y}$. Let $(\alpha\_k)$ be a sequence of maps $\mathcal{X}\_k \to \mathcal{Y}\_k$ that are **asymptotically equicontinuous**, in the sense that for every $\varepsilon > 0$, there are $\delta = \delta(\varepsilon) > 0$ and $N = N(\varepsilon) \in \mathbb{N}$ so that for all $k \ge N$,

$$d_{\mathcal{X}_k}(x_k, x'_k) \le \delta \implies d_{\mathcal{Y}_k}\bigl(\alpha_k(x_k), \alpha_k(x'_k)\bigr) \le \varepsilon.$$

Then after passing to a subsequence, the maps $g\_k \circ \alpha\_k \circ f'\_k : \mathcal{X} \to \mathcal{Y}$ converge uniformly to a continuous map $\alpha : \mathcal{X} \to \mathcal{Y}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(27.22 — Prokhorov Theorem in GH Converging Sequences)</span></p>

Let $(\mathcal{X}\_k)\_{k \in \mathbb{N}}$ be a sequence of compact metric spaces, converging in the Gromov–Hausdorff topology to some compact metric space $\mathcal{X}$, by means of $\varepsilon\_k$-approximations $f\_k : \mathcal{X}\_k \to \mathcal{X}$. For each $k$, let $\mu\_k$ be a probability measure on $\mathcal{X}\_k$. Then, after extraction of a subsequence, $(f\_k)\_\# \mu\_k$ converges in the weak topology to a probability measure $\mu$ on $\mathcal{X}$ as $k \to \infty$.

This statement extends to Polish spaces converging by means of local Gromov–Hausdorff approximations, provided that the probability measures $\mu\_k$ are uniformly tight with respect to the sequences $(K\_k^{(\ell)})$ appearing in the definition of local Gromov–Hausdorff approximation.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(27.24 — Compactness of Locally Finite Measures)</span></p>

Let $(\mathcal{X}\_k, d\_k, \star\_k)\_{k \in \mathbb{N}}$ be a sequence of pointed locally compact Polish spaces converging in the pointed Gromov–Hausdorff topology to some pointed locally compact Polish space $(\mathcal{X}, d, \star)$, by means of pointed $\varepsilon\_k$-isometries $f\_k$ with $\varepsilon\_k \to 0$. For each $k \in \mathbb{N}$, let $\nu\_k$ be a locally finite Borel measure on $\mathcal{X}\_k$. Assume that for each $R > 0$, there is a finite constant $M = M(R)$ such that

$$\forall k \in \mathbb{N}, \qquad \nu_k[B_{R]}(\star_k)] \le M.$$

Then, there is a locally finite measure $\nu$ such that, up to extraction of a subsequence,

$$(f_k)\_\# \nu_k \xrightarrow[k \to \infty]{} \nu$$

in the weak-$\ast$ topology (convergence against compactly supported continuous test functions).

</div>

### Adding the Measure

Now let us switch from metric spaces to **metric-measure spaces**, which are triples $(\mathcal{X}, d, \nu)$, where $d$ is a distance on $\mathcal{X}$ and $\nu$ a Borel measure on $\mathcal{X}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Measure-Preserving Isometry)</span></p>

Two metric-measure spaces $(\mathcal{X}, d, \nu)$ and $(\mathcal{X}', d', \nu')$ are **isomorphic** if there exists a measurable bijection $f : \mathcal{X} \to \mathcal{X}'$ such that $f$ is an isometry and $f$ preserves the measure: $f\_\# \nu = \nu'$. Such a map is called a **measure-preserving isometry**, and its inverse $f^{-1}$ is automatically a measure-preserving isometry.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two Philosophies for Comparing Metric-Measure Spaces)</span></p>

There is a nontrivial choice to be made:

**(a)** Two metric-measure spaces should be declared close only if they are close in terms of **both** the metric and the measure. In this case, one identifies $(\mathcal{X}, d, \nu)$ and $(\mathcal{X}', d', \nu')$ only if they are isomorphic as metric-measure spaces.

**(b)** Only the measure is relevant; one should disregard sets of zero or small measure when estimating how far two metric-measure spaces are. In this case, $(\mathcal{X}, d, \nu)$ and $(\mathcal{X}', d', \nu')$ are the same if there is a measure-preserving isometry between $\operatorname{Spt} \nu$ and $\operatorname{Spt} \nu'$.

For example, a balloon with a thin spike converging to a balloon without a spike: the limit has a spike in view (a), but no spike in view (b), because the spike carries no measure.

</div>

### Distances between Metric-Measure Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gromov–Hausdorff–Prokhorov Distance)</span></p>

For compact probability spaces $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$, the **Gromov–Hausdorff–Prokhorov distance** is

$$d_{GHP}(\mathcal{X}, \mathcal{Y}) = \inf\bigl\lbrace d_H(\mathcal{X}', \mathcal{Y}') + d_P(\nu_{\mathcal{X}'}, \nu_{\mathcal{Y}'})\bigr\rbrace,$$

where the infimum is taken over all measure-preserving isometric embeddings $f : (\mathcal{X}, \nu\_\mathcal{X}) \to (\mathcal{X}', \nu\_{\mathcal{X}'})$ and $g : (\mathcal{Y}, \nu\_\mathcal{Y}) \to (\mathcal{Y}', \nu\_{\mathcal{Y}'})$ into a *common* metric space $\mathcal{Z}$. This corresponds to philosophy (a).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gromov–Prokhorov Distance)</span></p>

In philosophy (b), one uses the **Gromov–Prokhorov distance**:

$$d_{GP}(\mathcal{X}, \mathcal{Y}) = \inf\; d_P(\nu_{\mathcal{X}'}, \nu_{\mathcal{Y}'}),$$

where the infimum is only over isometries (the metric structure of $\mathcal{X}$ has disappeared since the infimum is only over isometries).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gromov–Hausdorff–Wasserstein and Gromov–Wasserstein Distances)</span></p>

One can also replace the Prokhorov distance by the Wasserstein distance of order $p$, yielding the **Gromov–Hausdorff–Wasserstein distance of order $p$**:

$$d_{GHW_p}(\mathcal{X}, \mathcal{Y}) = \inf\bigl\lbrace d_H(\mathcal{X}', \mathcal{Y}') + W_p(\nu_{\mathcal{X}'}, \nu_{\mathcal{Y}'})\bigr\rbrace,$$

and the **Gromov–Wasserstein distance of order $p$**:

$$d_{GW_p}(\mathcal{X}, \mathcal{Y}) = \inf\; W_p(\nu_{\mathcal{X}'}, \nu_{\mathcal{Y}'}).$$

</div>

### Convergence and Doubling Property

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Doubling Resolves the Ambiguity)</span></p>

One should be cautious about which notion of convergence is used. However, whenever *doubling estimates* are available (in the sense of Definition 18.1), they basically rule out the discrepancy between philosophies (a) and (b). The idea is that doubling prevents the formation of sharp spikes (as in Figure 27.4). When doubling is not available, point of view (b) is more in line with the work of Gromov on *concentration of measure* and Sturm on *displacement convexity*. Nevertheless, in the sequel, philosophy (a) is adopted, as it provides a more precise notion of convergence.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(27.26 — Doubling Lets Metric and Metric-Measure Approaches Coincide)</span></p>

Let $(\mathcal{X}, \mu)$ and $(\mathcal{Y}, \nu)$ be two compact Polish probability spaces with diameter at most $R$. Assume both $\mu$ and $\nu$ are doubling with a constant $D$. Then

$$d_{GP}(\mathcal{X}, \mathcal{Y}) \le d_{GHP}(\mathcal{X}, \mathcal{Y}) \le \Phi_{R,D}\bigl(d_{GP}(\mathcal{X}, \mathcal{Y})\bigr),$$

where $\Phi\_{R,D}(\delta) = \max\bigl(8\delta,\; R\,(16\delta)^{1/\log\_2 D}\bigr) + \delta$ is a function that goes to 0 as $\delta \to 0$, at a rate controlled in terms of just upper bounds on $R$ and $D$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(27.28 — $d\_{GP}$ Convergence and Doubling Imply $d\_{GHP}$ Convergence)</span></p>

Let $(\mathcal{X}\_k, d\_k, \nu\_k)$ be a family of Polish probability spaces satisfying a uniform doubling condition, uniformly bounded, and converging to $(\mathcal{X}, d, \nu)$ in the Gromov–Prokhorov sense. Then $(\mathcal{X}\_k, d\_k, \nu\_k)$ also converges in the Gromov–Hausdorff–Prokhorov sense to $(\mathcal{X}, d, \nu)$. In particular, $(\mathcal{X}\_k, d\_k)$ converges to $(\mathcal{X}, d)$ in the Gromov–Hausdorff sense.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(27.29 — Doubling Implies Uniform Total Boundedness)</span></p>

Let $(\mathcal{X}, d)$ be a Polish space with diameter bounded above by $R$, equipped with a finite (nonzero) $D$-doubling measure $\nu$. Then for any $\varepsilon > 0$ there is a number $N = N(\varepsilon)$, only depending on $R$, $D$ and $\varepsilon$, such that $\mathcal{X}$ can be covered with $N$ balls of radius $\varepsilon$.

</div>

### Measured Gromov–Hausdorff Topology

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(27.30 — Measured Gromov–Hausdorff Topology)</span></p>

Let $(\mathcal{X}\_k, d\_k, \nu\_k)\_{k \in \mathbb{N}}$ and $(\mathcal{X}, d, \nu)$ be compact metric spaces equipped with finite nonzero measures. It is said that $\mathcal{X}\_k$ converges to $\mathcal{X}$ in the **measured Gromov–Hausdorff topology** if there are measurable $\varepsilon\_k$-isometries $f\_k : \mathcal{X}\_k \to \mathcal{X}$ such that $\varepsilon\_k \to 0$ and

$$(f_k)\_\# \nu_k \xrightarrow[k \to \infty]{} \nu$$

in the weak topology of measures.

If $(\mathcal{X}\_k, d\_k, \nu\_k)$ and $(\mathcal{X}, d, \nu)$ are Polish spaces (not necessarily compact) with $\sigma$-finite measures, then $\mathcal{X}\_k$ converges to $\mathcal{X}$ in the **local measured Gromov–Hausdorff topology** if there are nondecreasing sequences of compact sets $(K\_k^{(\ell)})\_{\ell \in \mathbb{N}}$ and $(K^{(\ell)})\_{\ell \in \mathbb{N}}$, with $\bigcup\_\ell K^{(\ell)}$ dense in $\mathcal{X}$, and for each $\ell$, the space $K\_k^{(\ell)}$ (as a subspace of $\mathcal{X}\_k$) converges in the measured Gromov–Hausdorff topology to $K^{(\ell)}$.

If the spaces are pointed, $(\mathcal{X}\_k, d\_k, \nu\_k, \star\_k)$ and $(\mathcal{X}, d, \nu, \star)$ are locally compact pointed Polish spaces with locally finite measures, then $\mathcal{X}\_k$ converges to $\mathcal{X}$ in the **pointed measured Gromov–Hausdorff topology** if there are sequences $R\_k \to \infty$ and $\varepsilon\_k \to 0$, and measurable pointed $\varepsilon\_k$-isometries $B[\star\_k, R\_k] \to B[\star, R\_k]$, such that $(f\_k)\_\# \nu\_k \to \nu$ in the weak-$\ast$ topology.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(27.32 — Compactness in Measured Gromov–Hausdorff Topology)</span></p>

**(i)** Let $R > 0$, $D > 0$, and $0 < m \le M$ be finite positive constants, and let $\mathcal{F}$ be a family of compact metric-measure spaces such that (a) for each $(\mathcal{X}, d, \nu) \in \mathcal{F}$ the diameter of $(\mathcal{X}, d)$ is bounded above by $2R$; (b) the measure $\nu$ has a doubling constant bounded above by $D$; and (c) $m \le \nu[\mathcal{X}] \le M$. Then $\mathcal{F}$ is **precompact** in the measured Gromov–Hausdorff topology. In particular, any weak cluster space $(\mathcal{X}\_\infty, d\_\infty, \nu\_\infty)$ satisfies $\operatorname{Spt} \nu\_\infty = \mathcal{X}\_\infty$.

**(ii)** Let $\mathcal{F}$ be a family of locally compact pointed Polish metric-measure spaces. Assume that for each $R$, there is a constant $D = D(R)$ such that for each $(\mathcal{X}, d, \nu, \star) \in \mathcal{F}$ the measure $\nu$ is $D$-doubling on the ball $B\_{R]}(\star)$. Further, assume the existence of $m, M > 0$ such that $m \le \nu[B\_1](\star)] \le M$ for all $(\mathcal{X}, d, \nu) \in \mathcal{F}$. Then $\mathcal{F}$ is **precompact** in the pointed measured Gromov–Hausdorff topology. In particular, any weak cluster space satisfies $\operatorname{Spt} \nu\_\infty = \mathcal{X}\_\infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(27.34 — Gromov's Precompactness Theorem)</span></p>

Let $K \in \mathbb{R}$, $N \in (1, \infty]$ and $D \in (0, +\infty)$. Let $\mathcal{M}(K, N, D)$ be the set of Riemannian manifolds $(M, g)$ such that $\dim(M) \le N$, $\operatorname{Ric}\_M \ge K\, g$ and $\operatorname{diam}(M) \le D$, equipped with their geodesic distance and their volume measure. Then $\mathcal{M}(K, N, D)$ is **precompact** in the measured Gromov–Hausdorff topology.

</div>

### Bibliographical Notes on Chapter 27

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

**Gromov's** influential book is one of the founding texts for the Gromov–Hausdorff topology. The presentation in this chapter mainly follows the pedagogical book of **Burago, Burago and Ivanov**. Other classical sources include the book by **Petersen** and the survey by **Fukaya**.

Gromov's "box" metric $\square\_1$ is an interesting example of a metric-measure distance: if $d$ and $d'$ are two metrics on a given probability space $\mathcal{X}$, define $\square\_1(d, d')$ as the infimum of $\varepsilon > 0$ such that $\lvert d - d' \rvert \le \varepsilon$ outside a set of measure at most $\varepsilon$ in $\mathcal{X} \times \mathcal{X}$. **Sturm** made a detailed study of $d\_{GW\_2}$ (denoted by $\mathbf{D}$) and advocated it as a natural distance between classes of equivalence of probability spaces in the context of optimal transport.

The approach of caring about both the metric and the measure was introduced by **Fukaya**. This is the approach used by **Lott and Villani** in their study of displacement convexity in geodesic spaces.

The pointed Gromov–Hausdorff topology is presented in **Burago, Burago and Ivanov**; it has become very popular as a way to study tangent spaces in the absence of smoothness. A celebrated theorem of **Gromov** (after precursors by Shikata) states: If $M$ is an $n$-dimensional compact Riemannian manifold, and $(M\_k)$ is a sequence of $n$-dimensional compact Riemannian manifolds converging to $M$, with uniform upper and lower bounds on the sectional curvatures, and a volume which is uniformly bounded below, then $M\_k$ is diffeomorphic to $M$ for $k$ large enough.

</div>

## Chapter 28: Stability of Optimal Transport

This chapter is devoted to the following theme: Consider a family of geodesic spaces $\mathcal{X}\_k$ which converges to some geodesic space $\mathcal{X}$; does this imply that certain basic objects in the theory of optimal transport on $\mathcal{X}\_k$ "pass to the limit"? The answer is affirmative: one of the main results is that the Wasserstein space $P\_2(\mathcal{X}\_k)$ converges, in (local) Gromov–Hausdorff sense, to the Wasserstein space $P\_2(\mathcal{X})$. The chapter also considers the stability of dynamical optimal transference plans, displacement interpolation, kinetic energy, and related objects. Compact spaces are considered first, serving as the basis for the subsequent treatment of noncompact spaces.

### Optimal Transport in a Nonsmooth Setting

Most of the objects that were introduced and studied in the context of optimal transport on Riemannian manifolds still make sense on a general metric-measure length space $(\mathcal{X}, d, \nu)$, satisfying certain regularity assumptions. Assume that $(\mathcal{X}, d)$ is a **locally compact, complete separable geodesic space** equipped with a $\sigma$-finite reference Borel measure $\nu$. From general properties of such spaces, plus the results in Chapters 6 and 7:

- The cost function $c(x, y) = d(x, y)^2$ is associated with the coercive Lagrangian action $\mathcal{A}(\gamma) = \mathcal{L}(\gamma)^2$, and minimizers are constant-speed, minimizing geodesics, the collection of which is denoted by $\Gamma(\mathcal{X})$.
- For any given $\mu\_0, \mu\_1$ in $P\_2(\mathcal{X})$, the optimal total cost $C(\mu\_0, \mu\_1)$ is finite and there exists at least one optimal transference plan $\pi \in P(\mathcal{X} \times \mathcal{X})$ with marginals $\mu\_0$ and $\mu\_1$.
- The 2-Wasserstein space $P\_2(\mathcal{X})$, equipped with the 2-Wasserstein distance, is a complete separable geodesic space.
- A displacement interpolation $(\mu\_t)\_{0 \le t \le 1}$ can be defined either as a geodesic in $P\_2(\mathcal{X})$, or as $(e\_t)\_\# \Pi$, where $e\_t$ is the evaluation at time $t$, and $\Pi$ is a dynamical optimal transference plan, i.e. the law of a random geodesic whose endpoints form an optimal coupling of $(\mu\_0, \mu\_1)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Absence of Explicit Descriptions)</span></p>

An important difference from the Riemannian setting is the absence of any "explicit" description of optimal couplings in terms of $d^2/2$-convex maps $\psi$. Expressions involving $\nabla \psi$ will not a priori make sense. However, one can make sense of $\lvert \nabla \psi \rvert$ by identifying it with the *length* $\mathcal{L}(\gamma)$ of the geodesic $\gamma$ joining $x = \gamma(0)$ to $y = \gamma(1)$. For instance,

$$\int \rho_0(x) \lvert \nabla \psi(x) \rvert^2\, d\nu(x) = \int d\bigl(x, \exp_x \nabla \psi(x)\bigr)^2\, d\mu_0(x) = W_2(\mu_0, \mu_1)^2.$$

</div>

### Kinetic Energy and Speed

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(28.1 — Kinetic Energy)</span></p>

Let $\mathcal{X}$ be a locally compact Polish geodesic space, and let $\Pi \in P(\Gamma(\mathcal{X}))$ be a dynamical transference plan. For each $t \in (0, 1)$ define the associated **kinetic energy** $\varepsilon\_t(dx)$ by the formula

$$\varepsilon_t = (e_t)\_\#\!\left(\frac{\mathcal{L}^2}{2}\, \Pi\right).$$

If $\varepsilon\_t$ is absolutely continuous with respect to $\mu\_t$, define the **speed field** $\lvert v \rvert(t, x)$ by

$$\lvert v \rvert(t, x) = \sqrt{2\,\frac{d\varepsilon_t}{d\mu_t}}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(28.2 — Boundedness of Speed)</span></p>

If $\mathcal{X}$ is compact then $\varepsilon\_t \le C\mu\_t$ with $C = (\operatorname{diam} \mathcal{X})^2/2$; so $\lvert v \rvert$ is well-defined (up to modification on a set of zero $\mu\_t$-measure) and almost surely bounded by $\sqrt{2C} = \operatorname{diam}(\mathcal{X})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(28.3 — Interpretation of the Speed Field)</span></p>

If $\gamma$ is a geodesic curve, then $\mathcal{L}(\gamma) = \lvert \dot{\gamma} \rvert(t)$ for all $t \in (0, 1)$. Assume $\mathcal{X}$ is a Riemannian manifold $M$, and geodesics in the support of $\Pi$ do not cross at intermediate times (which is the case if $\Pi$ is an *optimal* dynamical transference plan, by Chapter 8). Then for each $t \in (0, 1)$ and $x \in M$ there is at most one geodesic $\gamma = \gamma^{x,t}$ such that $\gamma(t) = x$. So

$$\varepsilon_t(dx) = \left(\frac{\lvert \dot{\gamma}^{x,t}(t) \rvert^2}{2}\right) \mu_t(dx),$$

and $\lvert v \rvert(t, x)$ is really $\lvert \dot{\gamma}^{x,t} \rvert$, i.e. the speed at time $t$ and position $x$. This is consistent with the usual notions of kinetic energy and speed field (speed = norm of the velocity).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(28.4 — Particular Case: Riemannian Manifold)</span></p>

Let $M$ be a Riemannian manifold, let $\mu\_0, \mu\_1 \in P\_2(M)$ with $\mu\_0$ absolutely continuous with respect to the volume measure. Let $\psi$ be a $d^2/2$-convex function such that $\exp(\nabla \psi)$ is the optimal transport from $\mu\_0$ to $\mu\_1$, and let $\psi\_t$ be obtained by solving the forward Hamilton–Jacobi equation $\partial\_t \psi\_t + \lvert \nabla \psi\_t \rvert^2/2 = 0$ starting from $\psi\_0 = \psi$. Then the speed $\lvert v \rvert(t, x)$ coincides, $\mu\_t$-almost surely, with $\lvert \nabla \psi\_t(x) \rvert$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(28.5 — Regularity of the Speed Field)</span></p>

Let $(\mathcal{X}, d)$ be a compact geodesic space, let $\Pi \in P(\Gamma(\mathcal{X}))$ be a dynamical optimal transference plan, let $(\mu\_t)\_{0 \le t \le 1}$ be the associated displacement interpolation, and $\lvert v \rvert = \lvert v \rvert(t, x)$ the associated speed field. Then, for each $t \in (0, 1)$ one can modify $\lvert v \rvert(t, \cdot)$ on a $\mu\_t$-negligible set in such a way that for all $x, y \in \mathcal{X}$,

$$\bigl\lvert\, \lvert v \rvert(t, x) - \lvert v \rvert(t, y) \bigr\rvert \;\le\; \frac{C\sqrt{\operatorname{diam}(\mathcal{X})}}{\sqrt{t(1-t)}}\; \sqrt{d(x, y)},$$

where $C$ is a numeric constant. In particular, $\lvert v \rvert(t, \cdot)$ is **Hölder-$1/2$**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Theorem 28.5)</span></p>

Let $t$ be a fixed time in $(0,1)$. Let $\gamma\_1$ and $\gamma\_2$ be two minimizing geodesics in the support of $\Pi$, and let $x = \gamma\_1(t)$, $y = \gamma\_2(t)$. Then by Theorem 8.22 (the Monge–Mather shortening principle),

$$\bigl\lvert \mathcal{L}(\gamma_1) - \mathcal{L}(\gamma_2) \bigr\rvert \le \frac{C\sqrt{\operatorname{diam}(\mathcal{X})}}{\sqrt{t(1-t)}}\; \sqrt{d(x, y)}.$$

Let $\mathcal{X}\_t$ be the union of all $\gamma(t)$ for $\gamma$ in the support of $\Pi$. For a given $x \in \mathcal{X}\_t$, there might be several geodesics $\gamma$ passing through $x$, but (by the above) they all have the same length; define $\lvert v \rvert(t, x)$ to be that length. To extend $\lvert v \rvert(t, x)$ to the whole of $\mathcal{X}$, one adapts the proof of a well-known extension theorem for Lipschitz functions by defining $w(x) := \inf\_{y \in \mathcal{X}\_t}\bigl[H\sqrt{d(x, y)} + \lvert v \rvert(t, y)\bigr]$, which preserves the Hölder-$1/2$ estimate.

</div>

### Convergence of the Wasserstein Space

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(28.6 — If $\mathcal{X}\_k$ Converges Then $P\_2(\mathcal{X}\_k)$ Also)</span></p>

Let $(\mathcal{X}\_k)\_{k \in \mathbb{N}}$ and $\mathcal{X}$ be compact metric spaces such that

$$\mathcal{X}_k \xrightarrow{GH} \mathcal{X}.$$

Then

$$P_2(\mathcal{X}_k) \xrightarrow{GH} P_2(\mathcal{X}).$$

Moreover, if $f\_k : \mathcal{X}\_k \to \mathcal{X}$ are approximate isometries, then the maps $(f\_k)\_\# : P\_2(\mathcal{X}\_k) \to P\_2(\mathcal{X})$, defined by $(f\_k)\_\#(\mu) = (f\_k)\_\# \mu$, are approximate isometries too.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(28.7 — If $f$ Is an Approximate Isometry Then $f\_\#$ Also)</span></p>

Let $f : (\mathcal{X}\_1, d\_1) \to (\mathcal{X}\_2, d\_2)$ be an $\varepsilon$-isometry between two Polish spaces. Then the map $f\_\#$ is a $\widetilde{\varepsilon}$-isometry between $P\_2(\mathcal{X}\_1)$ and $P\_2(\mathcal{X}\_2)$, where

$$\widetilde{\varepsilon} = 6\varepsilon + 2\sqrt{\varepsilon\,(2\operatorname{diam}(\mathcal{X}_2) + \varepsilon)} \;\le\; 8\bigl(\varepsilon + \sqrt{\varepsilon\,\operatorname{diam}(\mathcal{X}_2)}\bigr).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Proposition 28.7)</span></p>

Let $f'$ be an $\varepsilon$-inverse for $f$ (a $(4\varepsilon)$-isometry satisfying (27.4)). Given $\mu\_1$ and $\mu'\_1$ in $P\_2(\mathcal{X}\_1)$, let $\pi\_1$ be an optimal transference plan between them. Define $\pi\_2 := (f, f)\_\# \pi\_1$.

**$f\_\#$ does not increase distances much:**

$$W_2(f\_\# \mu_1, f\_\# \mu'_1) \le W_2(\mu_1, \mu'_1) + \sqrt{\varepsilon\,(2\operatorname{diam}(\mathcal{X}_2) + \varepsilon)}.$$

**$f\_\#$ does not decrease distances much:** Exchanging the roles of $\mathcal{X}\_1$ and $\mathcal{X}\_2$ and applying the map $f'$, one gets a complementary bound. Combining yields that $f\_\#$ distorts distances by at most $\widetilde{\varepsilon}$.

**$f\_\#$ is approximately surjective:** For any $\mu\_2 \in P\_2(\mathcal{X}\_2)$, $W\_2(\mu\_2, f\_\#(f'\_\# \mu\_2)) \le \varepsilon$.

</div>

### Compactness of Dynamical Transference Plans and Related Objects

The main technical challenge is that $\varepsilon$-isometries, being in general discontinuous, do not map geodesic paths into continuous paths. To handle this, one embeds the space of measurable paths $[0, 1] \to \mathcal{X}$ into the even much larger space of probability measures on $[0, 1] \times \mathcal{X}$, via the identification

$$\gamma \longmapsto \overline{\gamma} = (\operatorname{Id}, \gamma)\_\# \lambda,$$

where $\lambda$ is the Lebesgue measure on $[0, 1]$. In loose notation, $\overline{\gamma}(dt\, dx) = \delta\_{x = \gamma(t)}\, dt$. One thinks of the injection $i : \Gamma \to P([0, 1] \times \mathcal{X})$ defined by $i(\gamma) = \overline{\gamma}$ as an "inclusion"; any $\Pi \in P(\Gamma)$ can be identified with its push-forward $i\_\# \Pi \in P(P([0, 1] \times \mathcal{X}))$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(28.9 — Optimal Transport Is Stable Under GH Convergence)</span></p>

Let $(\mathcal{X}\_k, d\_k)\_{k \in \mathbb{N}}$ and $(\mathcal{X}, d)$ be compact geodesic spaces such that $\mathcal{X}\_k$ converges in the Gromov–Hausdorff topology as $k \to \infty$, by means of approximate isometries $f\_k : \mathcal{X}\_k \to \mathcal{X}$. For each $k \in \mathbb{N}$, let $\Pi\_k$ be a Borel probability measure on $\Gamma(\mathcal{X}\_k)$; further, let $\pi\_k = (e\_0, e\_1)\_\# \Pi\_k$, $\mu\_{k,t} = (e\_t)\_\# \Pi\_k$, and $\varepsilon\_{k,t} = (e\_t)\_\# [(\mathcal{L}^2/2)\, \Pi\_k]$. Then, after extraction of a subsequence (still denoted with index $k$ for simplicity), there is a dynamical transference plan $\Pi$ on $\mathcal{X}$, with associated transference plan $\pi(dx\, dy)$, measure-valued path $(\mu\_t(dx))\_{0 \le t \le 1}$, and kinetic energy $\varepsilon\_t(dx)$, such that:

- **(i)** $\lim\_{k \to \infty} (f\_k \circ)\_\# \Pi\_k = \Pi$ in the weak topology on $P(P([0,1] \times \mathcal{X}))$;
- **(ii)** $\lim\_{k \to \infty} (f\_k, f\_k)\_\# \pi\_k = \pi$ in $P(\mathcal{X} \times \mathcal{X})$;
- **(iii)** $\lim\_{k \to \infty} (f\_k)\_\# \mu\_{k,t} = \mu\_t$ in $P\_2(\mathcal{X})$ uniformly in $t$; more explicitly, $\lim\_{k \to \infty} \sup\_{t \in [0,1]} W\_2(\mu\_{k,t}, \mu\_t) = 0$;
- **(iv)** $\lim\_{k \to \infty} (f\_k)\_\# \varepsilon\_{k,t} = \varepsilon\_t$, in the weak topology of measures, for each $t \in (0, 1)$.

Assume further that each $\Pi\_k$ is an **optimal** dynamical transference plan, for the square distance cost function. Then:

- **(v)** For each $t \in (0, 1)$, there is a choice of the speed fields $\lvert v\_k \rvert$ associated with the plans $\Pi\_k$, such that $\lim\_{k \to \infty} \lvert v\_k \rvert \circ f'\_k = \lvert v \rvert$, in the **uniform topology**;
- **(vi)** The limit $\Pi$ is an optimal dynamical transference plan, so $\pi$ is an optimal transference plan and $(\mu\_t)\_{0 \le t \le 1}$ is a displacement interpolation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(28.10 — Meaning of $f\_k \circ$)</span></p>

In part (i), $f\_k \circ$ is the map $\gamma \to f\_k \circ \gamma$, which maps continuous paths $[0,1] \to \mathcal{X}\_k$ into measurable maps $[0,1] \to \mathcal{X}$ (identified to probability measures on $[0,1] \times \mathcal{X}$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Theorem 28.9)</span></p>

The proof proceeds in two main steps:

**Step 1: Compactness.** Since $[0,1] \times \mathcal{X}$ is a compact metric space, so is $P([0,1] \times \mathcal{X})$ and $P(P([0,1] \times \mathcal{X}))$. After extraction of a subsequence, $(f\_k \circ)\_\# \Pi\_k$ converges to some $\Pi$. Similarly, $(f\_k, f\_k)\_\# \pi\_k$ converges to some $\pi$. Since $\mathcal{X}$ is bounded and $\mathcal{X}\_k \to \mathcal{X}$, there is a uniform bound $C$ on the diameters. All geodesics $\gamma\_k \in \Gamma(\mathcal{X}\_k)$ have lengths bounded by $C$, and $d(\gamma\_k(s), \gamma\_k(t)) \le C\lvert s - t \rvert$ for all $s, t$. By the Ascoli theorem in Gromov–Hausdorff converging sequences (Proposition 27.20), the family $((f\_k)\_\# \mu\_{k,t})$ converges uniformly to a continuous curve $(\mu\_t) \in C([0,1]; P(\mathcal{X}))$. The kinetic energy measures $\varepsilon\_{k,t}$ have uniformly bounded total mass, so $(f\_k)\_\# \varepsilon\_{k,t}$ converges weakly to some measure $\varepsilon\_t$.

**Step 2: Passing to the limit.** To conclude properties (a)–(d) (that $\Pi$ is concentrated on $\Gamma(\mathcal{X})$, $\pi = (e\_0, e\_1)\_\# \Pi$, $\mu\_t = (e\_t)\_\# \Pi$, and $\varepsilon\_t = (e\_t)\_\# (\mathcal{L}^2\, \Pi)/2$), one uses a mollification technique. The condition $\mathcal{L}(\gamma) = d(\gamma(0), \gamma(1))$ (which characterizes geodesics) is mollified: for $\delta \in (0, 1/2)$ define "tent hat" functions $\varphi^\delta$ which converge weakly to $\delta\_0$ as $\delta \to 0$, and mollified length functionals $\mathcal{L}^\delta\_{t\_0 \to s\_0}$. One shows that certain closed sets $\Gamma\_{\varepsilon, \delta}(\mathcal{X})$ (defined by the mollified conditions) satisfy $\bigcap\_{\varepsilon, \delta} \Gamma\_{\varepsilon, \delta}(\mathcal{X}) = \Gamma(\mathcal{X})$. Since $(f\_k \circ)\_\# \Pi\_k \in P(\Gamma\_{\varepsilon, \delta}(\mathcal{X}))$ for $k$ large enough, passing to the limit yields $\Pi \in P(\Gamma(\mathcal{X}))$.

For the convergence of transference plans (b) and marginals (c), one applies a similar mollification argument with test functions. For (v), one uses Theorem 28.5 to get uniform Hölder-$1/2$ estimates on $\lvert v\_k \rvert$, and then the Ascoli theorem (Proposition 27.20) gives uniform convergence of $\lvert v\_k \rvert \circ f'\_k$ to $\lvert v \rvert$.

For (vi), since $\pi = \lim (f\_k, f\_k)\_\# \pi\_k$ and $f\_k$ is an approximate isometry, the transport cost satisfies

$$\int d(x_0, x_1)^2\, d\pi(x_0, x_1) = \lim_{k \to \infty} \int d_k(x_0, x_1)^2\, d\pi_k(x_0, x_1) = \lim_{k \to \infty} W_2(\mu_{0,k}, \mu_{1,k})^2 = W_2(\mu_0, \mu_1)^2,$$

using Theorem 28.6 ($(f\_k)\_\#$ is an approximate isometry on $P\_2$) and the continuity of $W\_2$ under weak convergence. So $\pi$ is an optimal transference plan and $\Pi$ is an optimal dynamical transference plan.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(28.11 — Characterization of Geodesics via Probability Measures)</span></p>

Let $(\mathcal{X}, d)$ be a compact geodesic space. Let $\sigma$ be a probability measure on $[0, 1] \times \mathcal{X}$ satisfying

$$\mathcal{L}^\delta_{t_0 \to s_0}(\sigma) \le C(\lvert s_0 - t_0 \rvert + \delta)$$

for all $t\_0, s\_0$ and all $\delta > 0$. Then there is a Lipschitz curve $\gamma : [0, 1] \to \mathcal{X}$ such that $\sigma(dt\, dx) = \overline{\gamma}(dt\, dx) = \delta\_{x = \gamma(t)}\, dt$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Lemma 28.11)</span></p>

First disintegrate $\sigma$ with respect to its first marginal $\lambda$: there is a family $(\nu\_t)\_{0 \le t \le 1}$ such that $\sigma(dt\, dx) = \nu\_t(dx)\, dt$. The goal is to show that, up to modification on a negligible set, $\nu\_t = \delta\_{\gamma(t)}$ where $\gamma$ is Lipschitz.

**Step 1 (Almost-everywhere Lipschitz continuity).** Integrating the bound on $\mathcal{L}^\delta\_{t\_0 \to s\_0}$ against an arbitrary nonnegative continuous function $\beta$ and passing $\delta \to 0$ yields $W\_1(\nu\_t, \nu\_s) \le C\lvert t - s \rvert$ for almost all $(t, s)$.

**Step 2 (True Lipschitz continuity).** Define mollified measures $\nu\_t^\varepsilon = \frac{1}{2\varepsilon}\int\_{-\varepsilon}^{\varepsilon} \nu\_{t+\tau}\, d\tau$. Then $W\_1(\nu\_t^\varepsilon, \nu\_s^\varepsilon) \le C\lvert t - s \rvert + O(\varepsilon)$. As $\varepsilon \to 0$, $\nu\_t^\varepsilon$ converges weakly to $\nu\_t$ for almost all $t$. Since $(P(\mathcal{X}), W\_1)$ is a complete metric space, one can redefine $\nu\_t$ on a negligible set so that $W\_1(\nu\_t, \nu\_s) \le C\lvert t - s \rvert$ for all $t, s \in [0, 1]$.

**Step 3 (Conclusion).** Setting $s\_0 = t\_0$ in the bound $\mathcal{L}^\delta\_{t\_0 \to s\_0}(\sigma) \le C(\lvert s\_0 - t\_0 \rvert + \delta)$ and taking $\delta \to 0$ yields $\int d(x, y)\, d\nu\_{t\_0}(x)\, d\nu\_{t\_0}(y) = 0$. This is possible only if $\nu\_{t\_0}$ is a Dirac measure. Hence for any $t\_0 \in [0, 1]$ there is $\gamma(t\_0) \in \mathcal{X}$ such that $\nu\_{t\_0} = \delta\_{\gamma(t\_0)}$. Then $d(\gamma(t), \gamma(s)) = W\_1(\nu\_t, \nu\_s) \le C\lvert t - s \rvert$, so $\gamma$ is Lipschitz continuous.

</div>

### Noncompact Spaces

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(28.13 — Pointed Convergence of $\mathcal{X}\_k$ Implies Local Convergence of $P\_2(\mathcal{X}\_k)$)</span></p>

Let $(\mathcal{X}\_k, d\_k, \star\_k)$ be a sequence of locally compact geodesic Polish spaces converging in the pointed Gromov–Hausdorff topology to some locally compact Polish space $(\mathcal{X}, d, \star)$. Then $P\_2(\mathcal{X}\_k)$ converges to $P\_2(\mathcal{X})$ in the geodesic local Gromov–Hausdorff topology.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(28.14 — Basepoint for $P\_2(\mathcal{X})$)</span></p>

If a basepoint $\star$ is given in $\mathcal{X}$, there is a natural choice of basepoint for $P\_2(\mathcal{X})$, namely $\delta\_\star$. However, $P\_2(\mathcal{X})$ is in general *not* locally compact, and it does not make sense to consider the *pointed* convergence of $P\_2(\mathcal{X}\_k)$ to $P\_2(\mathcal{X})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(28.15 — Extension to Geodesic Local GH Convergence)</span></p>

Theorem 28.13 admits the following extension: If $(\mathcal{X}\_k, d\_k)$ converges to $(\mathcal{X}, d)$ in the geodesic local Gromov–Hausdorff topology, then also $P\_2(\mathcal{X}\_k)$ converges to $P\_2(\mathcal{X})$ in the geodesic local Gromov–Hausdorff topology.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 28.13)</span></p>

Let $R\_\ell \to \infty$ be a given increasing sequence of positive numbers. Define

$$K^{(\ell)} = P_2\bigl(B_{R_\ell]}(\star)\bigr) \subset P_2(\mathcal{X}), \qquad K_k^{(\ell)} = P_2\bigl(B_{R_\ell]}(\star_k)\bigr) \subset P_2(\mathcal{X}_k).$$

Since $B\_{R\_\ell]}(\star)$ is compact, $K^{(\ell)}$ is compact too. The union of all $K^{(\ell)}$ is dense in $P\_2(\mathcal{X})$ (by Corollary of Theorem 6.18). For each $\ell$, there is a sequence $(f\_k)$ such that each $f\_k$ is an $\varepsilon\_k$-isometry $B\_{R\_\ell]}(\star\_k) \to B\_{R\_\ell]}(\star)$. From Proposition 28.7, $(f\_k)\_\#$ is a $\widetilde{\varepsilon}\_{k,\ell}$-isometry $K\_k^{(\ell)} \to K^{(\ell)}$, with $\widetilde{\varepsilon}\_{k,\ell} \le 8(\varepsilon\_k + \sqrt{2R\_\ell \varepsilon\_k}) \to 0$ as $k \to \infty$. This verifies all conditions of Definition 27.11 (and 27.12 for the geodesic version, choosing $R\_{\ell'} \ge 2R\_\ell$).

</div>

### Bibliographical Notes on Chapter 28

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

**Theorem 28.6** is taken from Lott and Villani. **Theorem 28.13** is an adaptation of the same reference. **Theorem 28.9** is new (a part of this theorem was included in a preliminary version of Lott–Villani, and later removed).

The discussion about push-forwarding dynamical transference plans is somewhat subtle. The point of view adopted in this chapter is: when an approximate isometry $f$ is given between two spaces, use it to push-forward a dynamical transference plan $\Pi$ via $(f \circ)\_\# \Pi$. The advantage is that this is the same map that push-forwards the measure and the dynamical plan. The drawback is that the resulting object $(f \circ)\_\# \Pi$ is **not** a dynamical transference plan — it may not even be supported on continuous paths. This leads to the technical approach of embedding into probability measures on probability measures.

An alternative strategy: given two spaces $\mathcal{X}$ and $\mathcal{Y}$, with an approximate isometry $f : \mathcal{X} \to \mathcal{Y}$, and a dynamical transference plan $\Pi$ on $\Gamma(\mathcal{X})$, define a **true** dynamical transference plan on $\Gamma(\mathcal{Y})$ which is a good approximation of $(f \circ)\_\# \Pi$, by constructing a recipe that associates to any geodesic $\gamma$ in $\mathcal{X}$ a geodesic $S(\gamma)$ in $\mathcal{Y}$ that is "close enough" to $f \circ \gamma$. This simpler strategy was successfully implemented in the final version of Lott–Villani.

The study of the kinetic energy measure and the speed field was motivated by regularity estimates on the speed, which come from a direction of research related to the proof of Theorem 23.14.

</div>

## Chapter 29: Weak Ricci Curvature Bounds I — Definition and Stability

In Chapter 14 several reformulations of the $\mathrm{CD}(K, N)$ curvature-dimension bound for a smooth manifold $(M, g)$ equipped with a reference measure $\nu$ whose density (with respect to the volume element) is smooth were discussed. For instance, for $N < \infty$: for any $C^2$ function $\psi : M \to \mathbb{R}$, let $\mathcal{J}(t, \cdot)$ be the Jacobian determinant of $T\_t : x \longmapsto \exp\_x(t\nabla\psi(x))$, and let $\mathcal{D}(t, x) = \mathcal{J}(t, x)^{1/N}$; then, with the notation of Theorem 14.11,

$$\mathcal{D}(t, x) \ge \tau_{K,N}^{(1-t)}\, \mathcal{D}(0, x) + \tau_{K,N}^{(t)}\, \mathcal{D}(1, x).$$

How to generalize this definition so that it makes sense in a possibly nonsmooth metric-measure space? This is nontrivial since (i) there might be no good notion of gradient, and (ii) there might be no good notion of exponential map either.

The only approach that yields acceptable results so far is the one based on **displacement convexity**. Recall from Chapters 16 and 17 two displacement convexity inequalities that characterize $\mathrm{CD}(K, N)$: Let $\mu\_0$ and $\mu\_1$ be two compactly supported absolutely continuous probability measures, let $\pi = (\mathrm{Id}, \exp \nabla\psi)\_\# \mu\_0$ be the optimal coupling of $(\mu\_0, \mu\_1)$, let $(\rho\_t\, \nu)\_{0 \le t \le 1}$ be the displacement interpolation between $\mu\_0 = \rho\_0\, \nu$ and $\mu\_1 = \rho\_1\, \nu$; let $(v\_t)\_{0 \le t \le 1}$ be the associated velocity field; then for any $U \in \mathcal{DC}\_N$, $t \in [0, 1]$,

$$\int U(\rho_t)\, d\nu \le (1-t) \int_{M \times M} U\!\left(\frac{\rho_0(x_0)}{\beta_{1-t}^{(K,N)}(x_0, x_1)}\right) \beta_{1-t}^{(K,N)}(x_0, x_1)\, \pi(dx_1 \vert x_0)\, \nu(dx_0) + t \int_{M \times M} U\!\left(\frac{\rho_1(x_1)}{\beta_t^{(K,N)}(x_0, x_1)}\right) \beta_t^{(K,N)}(x_0, x_1)\, \pi(dx_0 \vert x_1)\, \nu(dx_1).$$

Here $G(s, t)$ is the one-dimensional Green function (from (16.6)), $K\_{N,U}$ is defined by (17.10), and the distortion coefficients $\beta\_t^{(K,N)}$ are those appearing in (14.61).

When $K = 0$, both inequalities reduce to just

$$\int U(\rho_t)\, d\nu \le (1-t) \int U(\rho_0)\, d\nu + t \int U(\rho_1)\, d\nu.$$

### Issues in the Nonsmooth Extension

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two Key Issues)</span></p>

**(i) Nonuniqueness of the displacement interpolation.** There is a priori no reason to expect uniqueness of the displacement interpolation in a nonsmooth context. Rather than requiring distorted displacement convexity along *every* geodesic in Wasserstein space, we shall only impose a **weak displacement convexity** property: For any $\mu\_0$ and $\mu\_1$ there should be *some* geodesic $(\mu\_t)\_{0 \le t \le 1}$ along which the inequality holds.

To appreciate the difference: "F is convex along each geodesic $(\gamma\_t)$" and "For any $x\_0, x\_1$, there is a geodesic $(\gamma\_t)$ joining $x\_0$ to $x\_1$ such that $F(\gamma\_t) \le (1-t)F(\gamma\_0) + tF(\gamma\_1)$" are *not* equivalent in general (they become equivalent under some regularity assumption, e.g. if any two close enough points are joined by a unique geodesic).

**(ii) Treatment of the singular part.** Even if $\mu\_0$ and $\mu\_1$ are absolutely continuous with respect to $\nu$, there is no guarantee that the Wasserstein interpolant $\mu\_t$ will also be absolutely continuous. Also for stability issues it will be useful to work with singular measures, since $P\_2^{\mathrm{ac}}(\mathcal{X}, \nu)$ is not closed under weak convergence.

</div>

### Integral Functionals of Singular Measures

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(29.1 — Integral Functionals for Singular Measures)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a locally compact metric-measure space, where $\nu$ is locally finite; let $U : \mathbb{R}\_+ \to \mathbb{R}$ be a continuous convex function with $U(0) = 0$, and let $\mu$ be a measure on $\mathcal{X}$, compactly supported. Let

$$\mu = \rho\, \nu + \mu_s$$

be the Lebesgue decomposition of $\mu$ into absolutely continuous and singular parts. Then:

**(i)** Define the integral functional $U\_\nu$, with nonlinearity $U$ and reference measure $\nu$, by

$$U_\nu(\mu) := \int_\mathcal{X} U(\rho(x))\, \nu(dx) \;+\; U'(\infty)\, \mu_s[\mathcal{X}],$$

where $U'(\infty) := \lim\_{r \to \infty} U(r)/r = \lim\_{r \to \infty} U'(r) \in \mathbb{R} \cup \lbrace +\infty \rbrace$.

**(ii)** If $x \to \pi(dy \vert x)$ is a family of probability measures on $\mathcal{X}$, indexed by $x \in \mathcal{X}$, and $\beta$ is a measurable function $\mathcal{X} \times \mathcal{X} \to (0, +\infty]$, define the integral functional $U\_{\pi,\nu}^\beta$ with nonlinearity $U$, reference measure $\nu$, coupling $\pi$ and distortion coefficient $\beta$, by

$$U_{\pi,\nu}^\beta(\mu) := \int_{\mathcal{X} \times \mathcal{X}} U\!\left(\frac{\rho(x)}{\beta(x,y)}\right) \beta(x,y)\, \pi(dy \vert x)\, \nu(dx) \;+\; U'(\infty)\, \mu_s[\mathcal{X}].$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(29.2–29.4 — Properties of $U\_{\pi,\nu}^\beta$)</span></p>

- $U\_{\pi,\nu}^\beta$ reduces to $U\_\nu$ when $\beta \equiv 1$ (i.e. when there is no distortion).
- One often identifies $\pi$ with the probability measure $\pi(dx\, dy) = \mu(dx)\, \pi(dy \vert x)$ on $\mathcal{X} \times \mathcal{X}$.
- The new definition of $U\_\nu$ takes care of singularities of $\mu$ via the term $U'(\infty)\, \mu\_s[\mathcal{X}]$. The idea is that the singular part of $\mu$ has "infinite density", and the asymptotic slope $U'(\infty)$ captures the contribution of $U$ at infinity. Subtleties linked to the behavior at infinity will be addressed in the next chapter.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(29.6 — Rewriting of the Distorted $U\_\nu$ Functional)</span></p>

With the notation of Definition 29.1, when $\mu\_s = 0$:

$$U_{\pi,\nu}^\beta(\mu) = \int_{\mathcal{X} \times \mathcal{X}} U\!\left(\frac{\rho(x)}{\beta(x,y)}\right) \frac{\beta(x,y)}{\rho(x)}\; \pi(dx\, dy) = \int_{\mathcal{X} \times \mathcal{X}} v\!\left(\frac{\rho(x)}{\beta(x,y)}\right) \pi(dx\, dy),$$

where $v(r) = U(r)/r$, with the conventions $U(0)/0 = U'(0) \in [-\infty, +\infty)$, $U(\infty)/\infty = U'(\infty) \in (-\infty, +\infty]$, and $\rho = 0$ on $\operatorname{Spt} \mu\_s$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(29.7 — Rescaled Subadditivity of the Distorted $U\_\nu$ Functionals)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a locally compact metric-measure space, where $\nu$ is locally finite, and let $\beta$ be a positive measurable function on $\mathcal{X} \times \mathcal{X}$. Let $U$ be a continuous convex function with $U(0) = 0$. Let $\mu\_1, \ldots, \mu\_k$ be probability measures on $\mathcal{X}$, let $\pi\_1, \ldots, \pi\_k$ be probability measures on $\mathcal{X} \times \mathcal{X}$, and let $Z\_1, \ldots, Z\_k$ be positive numbers with $\sum Z\_j = 1$. Then, with the notation $U\_a(r) = a^{-1} U(ar)$, one has

$$U_{\sum_j Z_j \pi_j, \nu}^\beta\!\left(\sum Z_j \mu_j\right) \ge \sum_j Z_j\, (U_{Z_j})_{\pi_j, \nu}^\beta(\mu_j),$$

with equality if the measures $\mu\_k$ are singular with respect to each other.

</div>

### Synthetic Definition of the Curvature-Dimension Bound

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(29.8 — Weak Curvature-Dimension Condition)</span></p>

Let $K \in \mathbb{R}$ and $N \in [1, \infty]$. A locally compact, complete $\sigma$-finite metric-measure geodesic space $(\mathcal{X}, d, \nu)$ is said to satisfy a **weak $\mathrm{CD}(K, N)$ condition**, or to be a **weak $\mathrm{CD}(K, N)$ space**, if the following is satisfied:

Whenever $\mu\_0, \mu\_1$ are two compactly supported probability measures with $\operatorname{Spt} \mu\_0, \operatorname{Spt} \mu\_1 \subset \operatorname{Spt} \nu$, there exist a displacement interpolation $(\mu\_t)\_{0 \le t \le 1}$ and an associated optimal coupling $\pi$ of $(\mu\_0, \mu\_1)$ such that, **for all $U \in \mathcal{DC}\_N$** and for all $t \in [0, 1]$,

$$U_\nu(\mu_t) \le (1-t)\, U_{\pi,\nu}^{\beta_{1-t}^{(K,N)}}(\mu_0) \;+\; t\, U_{\check{\pi},\nu}^{\beta_t^{(K,N)}}(\mu_1).$$

Here $\check{\pi} = S\_\# \pi$ (obtained from $\pi$ by "exchanging $x$ and $y$", i.e. $S(x,y) = (y,x)$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of Weak $\mathrm{CD}(K,N)$)</span></p>

Roughly speaking, the weak $\mathrm{CD}(K, N)$ condition states that the functionals $U\_\nu$ are "jointly" weakly displacement convex with distortion coefficients $(\beta\_t^{(K,N)})$, for all $U \in \mathcal{DC}\_N$. This is a property of the triple $(\mathcal{X}, d, \nu)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(29.9 — Smooth Weak $\mathrm{CD}(K,N)$ Spaces Are $\mathrm{CD}(K,N)$ Manifolds)</span></p>

Let $(M, g)$ be a smooth Riemannian manifold, equipped with its geodesic distance $d$, its volume measure $\mathrm{vol}$, and a reference measure $\nu = e^{-V}\, \mathrm{vol}$, where $V \in C^2(M)$. Then, $(M, d, \nu)$ is a weak $\mathrm{CD}(K, N)$ space **if and only if** $(M, g, \nu)$ satisfies the $\mathrm{CD}(K, N)$ curvature-dimension bound; or equivalently, if the modified Ricci tensor satisfies $\operatorname{Ric}\_{N,\nu} \ge K g$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Comments on Definition 29.8)</span></p>

- In Definition 29.8, displacement convexity inequalities are imposed along *some* Wasserstein geodesic, because such geodesics might not be unique. There are two reasons for nonuniqueness: lack of smoothness of the space $\mathcal{X}$, and the possibility that $\mu\_0, \mu\_1$ might be singular. However, it will turn out later (Theorem 30.32) that displacement convexity inequalities hold along *all* Wasserstein geodesics if $\mathcal{X}$ is nonbranching.
- The classical $\mathrm{CD}(K, N)$ condition becomes more stringent as $K$ increases and as $N$ decreases. The same is true in the nonsmooth setting.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(29.10 — Consistency of the $\mathrm{CD}(K,N)$ Conditions)</span></p>

The weak condition $\mathrm{CD}(K, N)$ becomes more and more stringent as $K$ increases, and as $N$ decreases.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(29.11 — Bonnet–Myers Diameter Bound for Weak $\mathrm{CD}(K,N)$ Spaces)</span></p>

If $(\mathcal{X}, d, \nu)$ is a weak $\mathrm{CD}(K, N)$ space with $K > 0$ and $N < \infty$, then

$$\operatorname{diam}(\operatorname{Spt} \nu) \le D_{K,N} := \pi\sqrt{\frac{N-1}{K}}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Proposition 29.11)</span></p>

In the case $K > 0$ and $N < \infty$, the coefficient $\beta\_t^{(K,N)}(x,y)$ takes the value $+\infty$ if $0 < t < 1$ and $d(x,y) \ge D\_{K,N}$. With this convention, Definition 29.8 implies that the diameter of the support of $\nu$ is automatically bounded above by $D\_{K,N}$.

Take $x\_0, x\_1 \in \operatorname{Spt} \nu$ with $d(x\_0, x\_1) > D\_{K,N}$ and choose $r > 0$ small enough. Take $\rho\_0 = 1\_{B\_r(x\_0)}/\nu[B\_r(x\_0)]$ and $\rho\_1 = 1\_{B\_r(x\_1)}/\nu[B\_r(x\_1)]$. Then the $\beta\_t$ coefficients in the right-hand side of (29.11) are identically $+\infty$, and the measures have no singular part; so the inequality becomes $U\_\nu(\mu\_t) \le U'(0)$. Now choose $U(r) = -r^{1-1/N}$: then $U'(0) = -\infty$, so $U\_\nu(\mu\_t) = -\infty$. On the other hand, by Jensen's inequality, $U\_\nu(\mu\_t) \ge -\nu[S]^{1/N}$ where $S$ is the support of $\mu\_t$; so $U\_\nu(\mu\_t)$ cannot be $-\infty$. Contradiction.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(29.12 — Sufficient Condition to Be a Weak $\mathrm{CD}(K,N)$ Space)</span></p>

In Definition 29.8, it is equivalent to require that the inequality holds for all $U \in \mathcal{DC}\_N$, or just for those $U \in \mathcal{DC}\_N$ which are nonnegative and satisfy:

- $U$ is Lipschitz, if $N < \infty$;
- $U$ is locally Lipschitz and $U(r) = a\, r \log r + b\, r$ for $r$ large enough, if $N = \infty$ (with $a \ge 0$, $b \in \mathbb{R}$).

</div>

### Examples of Weak $\mathrm{CD}(K, N)$ Spaces

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(29.13 — Euclidean Space with Log-Concave Measure)</span></p>

Let $V$ be a continuous function $\mathbb{R}^n \to \mathbb{R}$ with $\int e^{-V(x)}\, dx < \infty$, let $\nu(dx) = e^{-V(x)}\, dx$, and let $d\_2$ be the Euclidean distance. Then the space $(\mathbb{R}^n, d\_2, \nu)$ satisfies the usual $\mathrm{CD}(K, \infty)$ condition if $V$ is $C^2$ and $\nabla^2 V \ge K\, I\_n$. It satisfies the **weak $\mathrm{CD}(K, \infty)$ condition** *without any regularity assumption on $V$*, as soon as $\nabla^2 V \ge K\, I\_n$ in the sense of distributions, which means that $V$ is $K$-convex. In particular, if $V$ is merely convex, then $(\mathbb{R}^n, d\_2, \nu)$ satisfies the weak $\mathrm{CD}(0, \infty)$ condition.

To see this, note that if $\mu(dx) = \rho(x)\, dx$, then $H\_\nu(\mu) = \int \rho(x) \log \rho(x)\, dx + \int \rho(x)\, V(x)\, dx = H(\mu) + \int V\, d\mu$; the first term is always displacement convex, and the second is displacement convex if $V$ is convex (simple exercise).

Conversely, if $V$ is not convex, then $(\mathbb{R}^n, d\_2, e^{-V(x)}\, dx)$ **cannot** be a weak $\mathrm{CD}(0, \infty)$ space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(29.15 — Quotient Manifold by a Compact Lie Group)</span></p>

Let $M$ be a smooth compact $n$-dimensional Riemannian manifold with nonnegative Ricci curvature, and let $G$ be a compact Lie group acting isometrically on $M$. Then $\mathcal{X} = M/G$ with the quotient distance $d(x, y) = \inf\lbrace d\_M(x', y');\; q(x') = x,\, q(y') = y \rbrace$ and the measure $\nu = q\_\# \mathrm{vol}\_M$ is a weak $\mathrm{CD}(0, n)$ space. The resulting space will in general not be a manifold (there will typically be singularities at fixed points of the group action).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(29.16 — Normed Spaces)</span></p>

It will be shown in the concluding chapter that $(\mathbb{R}^n, \lVert \cdot \rVert, \lambda\_n)$ is a weak $\mathrm{CD}(K, N)$ space, where $\lVert \cdot \rVert$ is *any* norm on $\mathbb{R}^n$ and $\lambda\_n$ is the $n$-dimensional Lebesgue measure. This example proves that a weak $\mathrm{CD}(K, N)$ space may be "strongly" branching (recall the discussion in Example 27.17).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(29.17 — Infinite Product of Circles)</span></p>

Let $\mathcal{X} = \prod\_{i=1}^\infty T\_i$, where $T\_i = \mathbb{R}/(\varepsilon\_i \mathbb{Z})$ is equipped with the usual distance $d\_i$ and the normalized Lebesgue measure $\lambda\_i$, and $\varepsilon\_i = 2\operatorname{diam}(T\_i)$ is some positive number. If $\sum \varepsilon\_i^2 < +\infty$ then the product distance $d = \sqrt{\sum d\_i^2}$ turns $\mathcal{X}$ into a compact metric space. Equip $\mathcal{X}$ with the product measure $\nu = \prod \lambda\_i$; then $(\mathcal{X}, d, \nu)$ is a weak $\mathrm{CD}(0, \infty)$ space. (Indeed, it is the measured Gromov–Hausdorff limit of $\mathcal{X} = \prod\_{j=1}^k T\_j$ which is $\mathrm{CD}(0, k)$, hence $\mathrm{CD}(0, \infty)$; and it will be shown in Theorem 29.24 that the $\mathrm{CD}(0, \infty)$ property is stable under measured Gromov–Hausdorff limits.)

</div>

### Continuity Properties of the Functionals $U\_\nu$ and $U\_{\pi,\nu}^\beta$

The remaining part of the chapter is devoted to a proof of stability for the weak $\mathrm{CD}(K, N)$ property.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Functionals on the Space of Measures)</span></p>

It will be convenient to consider $U\_\nu$ and $U\_{\pi,\nu}^\beta$ as defined on the whole vector space $M(\mathcal{X})$ of finite Borel measures on $\mathcal{X}$, with the convention that their value is $+\infty$ if $\mu$ is not nonnegative; then $U\_\nu$ and $U\_{\pi,\nu}^\beta$ are true convex functionals on $M(\mathcal{X})$.

It will be convenient to study the functionals $U\_\nu$ by means of their **Legendre representation**. Generally speaking, the Legendre representation of a convex functional $\Phi$ defined on a vector space $E$ is an identity of the form

$$\Phi(x) = \sup\bigl\lbrace \langle \Lambda, x \rangle - \Psi(\Lambda) \bigr\rbrace,$$

where $\Lambda$ varies over a certain subset of $E^\ast$, and $\Psi$ is a convex functional of $\Lambda$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(29.18 — Legendre Transform of a Real-Valued Convex Function)</span></p>

Let $U : \mathbb{R}\_+ \to \mathbb{R}$ be a continuous convex function with $U(0) = 0$; its **Legendre transform** is defined on $\mathbb{R}$ by

$$U^\ast(p) = \sup_{r \in \mathbb{R}_+} \bigl[p\, r - U(r)\bigr].$$

It is easy to check that $U^\ast$ is a convex function, taking the value $-U(0) = 0$ on $(-\infty, U'(0)]$ and $+\infty$ on $(U'(\infty), +\infty)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(29.19 — Legendre Representation of $U\_\nu$)</span></p>

Let $U : \mathbb{R}\_+ \to \mathbb{R}$ be a continuous convex function with $U(0) = 0$, let $\mathcal{X}$ be a compact metric space, equipped with a finite reference measure $\nu$. Then, whenever $\mu$ is a finite measure on $\mathcal{X}$,

**(i)** $\displaystyle U\_\nu(\mu) = \sup\left\lbrace \int\_\mathcal{X} \varphi\, d\mu - \int\_\mathcal{X} U^\ast(\varphi)\, d\nu;\;\; \varphi \in L^\infty(\mathcal{X});\;\; \varphi \le U'(\infty)\right\rbrace$.

**(ii)** $\displaystyle U\_\nu(\mu) = \sup\left\lbrace \int\_\mathcal{X} \varphi\, d\mu - \int\_\mathcal{X} U^\ast(\varphi)\, d\nu;\;\; \varphi \in C(\mathcal{X}),\;\; U'\!\left(\tfrac{1}{M}\right) \le \varphi \le U'(M);\;\; M \in \mathbb{N}\right\rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(29.20 — Continuity and Contraction Properties of $U\_\nu$ and $U\_{\pi,\nu}^\beta$)</span></p>

Let $(\mathcal{X}, d)$ be a compact metric space, equipped with a finite measure $\nu$. Let $U : \mathbb{R}\_+ \to \mathbb{R}\_+$ be a convex continuous function with $U(0) = 0$. Further, let $\beta(x, y)$ be a continuous positive function on $\mathcal{X} \times \mathcal{X}$. Then, with the notation of Definition 29.1:

**(i)** $U\_\nu(\mu)$ is a **weakly lower semicontinuous** function of both $\mu$ and $\nu$ in $M\_+(\mathcal{X})$. More explicitly, if $\mu\_k \to \mu$ and $\nu\_k \to \nu$ in the weak topology of convergence against bounded continuous functions, then

$$U_\nu(\mu) \le \liminf_{k \to \infty}\; U_{\nu_k}(\mu_k).$$

**(ii)** $U\_\nu$ satisfies a **contraction principle** in both $\mu$ and $\nu$: if $\mathcal{Y}$ is another compact space and $f : \mathcal{X} \to \mathcal{Y}$ is any measurable function, then

$$U_{f\_\# \nu}(f\_\# \mu) \le U_\nu(\mu).$$

**(iii)** If $U$ "grows at most polynomially", in the sense that $r\, U'(r) \le C\,(U(r)\_+ + r)$ for all $r > 0$, then for any probability measure $\mu \in P(\mathcal{X})$ with $\operatorname{Spt} \mu \subset \operatorname{Spt} \nu$, there is a sequence $(\mu\_k)\_{k \in \mathbb{N}}$ of probability measures converging weakly to $\mu$, such that each $\mu\_k$ has a continuous density, and for any sequence $(\pi\_k)\_{k \in \mathbb{N}}$ converging weakly to $\pi$ in $P(\mathcal{X} \times \mathcal{X})$, such that $\pi\_k$ admits $\mu\_k$ as first marginal and $\operatorname{Spt} \pi\_k \subset (\operatorname{Spt} \nu) \times (\operatorname{Spt} \nu)$,

$$\limsup_{k \to \infty}\; U_{\pi_k, \nu}^\beta(\mu_k) \le U_{\pi, \nu}^\beta(\mu).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary of Theorem 29.20)</span></p>

The three key properties on which the main results of this chapter rest:

1. $U\_\nu(\mu)$ is **lower semicontinuous** in $(\mu, \nu)$ — this handles the left-hand side of displacement convexity inequalities when passing to the limit.
2. $U\_\nu(\mu)$ is **never increased by push-forward** — this provides the contraction principle needed when transferring inequalities between spaces via approximate isometries.
3. $\mu$ can be **regularized** (approximated by measures with continuous densities) in such a way that $U\_{\pi,\nu}^\beta(\mu)$ is **upper semicontinuous** along the approximation — this handles the right-hand side of displacement convexity inequalities.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Theorem 29.20)</span></p>

**(i) Lower semicontinuity.** By Proposition 29.19(ii), $U\_\nu(\mu)$ can be rewritten as $\sup\_{(\varphi, \psi) \in \mathcal{U}} \lbrace \int \varphi\, d\mu + \int \psi\, d\nu \rbrace$ where $\mathcal{U}$ is a certain subset of $C(\mathcal{X}) \times C(\mathcal{X})$. In particular, $U\_\nu(\mu)$ is a supremum of continuous functions of $(\mu, \nu)$; hence lower semicontinuous.

**(ii) Contraction.** For any $\varphi \le U'(\infty)$ bounded, $\int (\varphi \circ f)\, d\mu - \int U^\ast(\varphi \circ f)\, d\nu = \int \varphi\, d(f\_\# \mu) - \int U^\ast(\varphi)\, d(f\_\# \nu)$. Since $\varphi \circ f \le U'(\infty)$ as well, taking suprema yields $U\_\nu(\mu) \ge U\_{f\_\# \nu}(f\_\# \mu)$.

**(iii) Upper semicontinuity of the distorted functional.** This is the most delicate part. The proof proceeds in 9 steps:

- **Steps 1–4:** Reduce to $\operatorname{Spt} \nu = \mathcal{X}$, $U'(0) > -\infty$, $U \ge 0$, and handle the singular part. The regularization uses kernels $K\_\varepsilon(x, y)$ (symmetric, continuous, nonnegative functions supported on $\lbrace d(x,y) \le \varepsilon \rbrace$ and integrating to 1 against $\nu$). One defines $\mu\_\varepsilon = \rho\_\varepsilon\, \nu$ where $\rho\_\varepsilon(x) = \int K\_\varepsilon(x,y)\, \mu(dy)$. By Jensen's inequality, $U\_\nu(\mu\_\varepsilon) \le U\_\nu(\mu)$ for all $\varepsilon > 0$ (for the $\beta \equiv 1$ case).
- **Steps 5–6:** Approximate $\beta$ by its convolution $\beta\_\varepsilon$ (which converges uniformly to $\beta$), and use the key convexity inequality: $\beta\, U(\rho/\beta) = \sup\_{p \in \mathbb{R}}[p\rho - \beta\, U^\ast(p)]$, so $\beta\, U(\rho/\beta)$ is jointly convex in $(\beta, \rho)$. By Jensen's inequality applied to the averaged quantities $\beta\_\varepsilon, \rho\_\varepsilon$, the distorted functional with $(\beta\_\varepsilon, \rho\_\varepsilon)$ is bounded by the original one.
- **Steps 7–9:** Replace the non-continuous function $f(x,y) = \beta(x,y)\, U(\rho(x)/\beta(x,y))$ by $g(x,y) = f(x,y)/\rho(x)$, which belongs to $L^1((\mathcal{X}, \mu); C(\mathcal{X}))$. Approximate $g$ by continuous functions $\Psi\_k$ (via Lemma 29.36 in the Second Appendix), and use duality to "transfer the regularization to the test function". This yields the convergence $\int f\, d\omega\_\varepsilon \to \int f\, d\omega$ where $\omega\_\varepsilon$ and $\omega$ are certain measures derived from $\pi\_\varepsilon$ and $\pi$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(29.23 — Another Sufficient Condition to Be a Weak $\mathrm{CD}(K,N)$ Space)</span></p>

In Definition 29.8 it is equivalent to require the inequality for all probability measures $\mu\_0, \mu\_1$; or only when $\mu\_0, \mu\_1$ are absolutely continuous with continuous densities.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Corollary 29.23)</span></p>

Assume $(\mathcal{X}, d, \nu)$ satisfies the assumptions of Definition 29.8, except that $\mu\_0, \mu\_1$ are required to be absolutely continuous. Let $\mu\_0, \mu\_1$ be two possibly singular probability measures with $\operatorname{Spt} \mu\_0, \operatorname{Spt} \mu\_1 \subset \operatorname{Spt} \nu$. By Theorem 29.20(iii), there are sequences of probability measures $\mu\_{k,0} \to \mu\_0$ and $\mu\_{k,1} \to \mu\_1$, all absolutely continuous and with continuous densities, and for any sequence $\pi\_k \in \Pi(\mu\_{k,0}, \mu\_{k,1})$ converging weakly to $\pi$,

$$\limsup_{k \to \infty} U_{\pi_k, \nu}^{\beta_{1-t}^{(K,N)}}(\mu_{k,0}) \le U_{\pi, \nu}^{\beta_{1-t}^{(K,N)}}(\mu_0).$$

For each $k$ there is a displacement interpolation $(\mu\_{k,t})$ and an associated optimal coupling $\pi\_k$ satisfying the CD inequality. By Theorem 28.9, up to extraction, $\mu\_{k,t} \to \mu\_t$ in $P\_2(\mathcal{X})$ for each $t$ and $\pi\_k \to \pi$. Then by Theorem 29.20(i), $U\_\nu(\mu\_t) \le \liminf U\_\nu(\mu\_{k,t})$. Combining yields the desired inequality. (The case when $\beta\_t^{(K,N)}$ is not continuous is handled via Proposition 29.10 by taking $N' > N$ and letting $N' \downarrow N$.)

</div>

### Stability of Ricci Bounds

The main result of this chapter: *The weak curvature-dimension bound $\mathrm{CD}(K, N)$ passes to the limit* under measured Gromov–Hausdorff convergence. The compact case implies the general statement.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(29.24 — Stability of Weak $\mathrm{CD}(K,N)$ Under MGH)</span></p>

Let $(\mathcal{X}\_k, d\_k, \nu\_k)\_{k \in \mathbb{N}}$ be a sequence of compact metric-measure geodesic spaces converging in the measured Gromov–Hausdorff topology to a compact metric-measure space $(\mathcal{X}, d, \nu)$. Let $K \in \mathbb{R}$ and $N \in [1, \infty]$. If each $(\mathcal{X}\_k, d\_k, \nu\_k)$ satisfies the weak curvature-dimension condition $\mathrm{CD}(K, N)$, then also $(\mathcal{X}, d, \nu)$ satisfies $\mathrm{CD}(K, N)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(29.25 — Stability of Weak $\mathrm{CD}(K,N)$ Under Pointed MGH)</span></p>

Let $(\mathcal{X}\_k, d\_k, \nu\_k)\_{k \in \mathbb{N}}$ be a sequence of locally compact, complete, separable $\sigma$-finite metric-measure geodesic spaces converging in the pointed measured Gromov–Hausdorff topology to a locally compact, complete separable $\sigma$-finite metric-measure space $(\mathcal{X}, d, \nu)$. Let $K \in \mathbb{R}$ and $N \in [1, \infty]$. If each $(\mathcal{X}\_k, d\_k, \nu\_k)$ satisfies the weak curvature-dimension condition $\mathrm{CD}(K, N)$, then also $(\mathcal{X}, d, \nu)$ satisfies $\mathrm{CD}(K, N)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Theorem 29.24)</span></p>

From the characterization of measured Gromov–Hausdorff convergence, there are measurable $f\_k : \mathcal{X}\_k \to \mathcal{X}$ such that (i) $f\_k$ is an $\varepsilon\_k$-isometry with $\varepsilon\_k \to 0$; (ii) $(f\_k)\_\# \nu\_k \to \nu$ weakly.

Let $\rho\_0, \rho\_1$ be two probability densities on $(\mathcal{X}, \nu)$; let $\mu\_0 = \rho\_0\, \nu$, $\mu\_1 = \rho\_1\, \nu$. Let $\varepsilon = (\varepsilon\_m)$ be a sequence going to 0; for each $t\_0 \in \lbrace 0, 1 \rbrace$, let $(\rho\_{\varepsilon, t\_0})$ be a sequence of continuous probability densities satisfying the conclusion of Theorem 29.20(iii), and let $\mu\_{\varepsilon, t\_0} = \rho\_{\varepsilon, t\_0}\, \nu$. Define

$$\mu_{\varepsilon, t_0}^k := \frac{(\rho_{\varepsilon, t_0} \circ f_k)\, \nu_k}{Z_{\varepsilon, t_0}^k}, \qquad Z_{\varepsilon, t_0}^k = \int (\rho_{\varepsilon, t_0} \circ f_k)\, d\nu_k.$$

Since $\rho\_{\varepsilon, t\_0}$ is continuous and $(f\_k)\_\# \nu\_k \to \nu$, one has $Z\_{\varepsilon, t\_0}^k \to 1$ and $(f\_k)\_\# \mu\_{\varepsilon, t\_0}^k \to \mu\_{\varepsilon, t\_0}$ weakly.

Since each $(\mathcal{X}\_k, d\_k, \nu\_k)$ satisfies $\mathrm{CD}(K, N)$, there is a Wasserstein geodesic $(\mu\_{\varepsilon, t}^k)\_{0 \le t \le 1}$ joining $\mu\_{\varepsilon, 0}^k$ to $\mu\_{\varepsilon, 1}^k$ with an associated optimal coupling $\pi\_\varepsilon^k$ and dynamical plan $\Pi\_\varepsilon^k$, satisfying the CD inequality. By Theorem 28.9, up to extraction in $k$, there is a dynamical optimal transference plan $\Pi\_\varepsilon$ on $\Gamma(\mathcal{X})$ with marginal geodesic $(\mu\_{\varepsilon, t})$ and coupling $\pi\_\varepsilon$ such that $\sup\_{t} W\_2(\mu\_{\varepsilon, t}^k, \mu\_{\varepsilon, t}) \to 0$. By Ascoli's theorem, a further extraction in $\varepsilon$ gives $\sup\_{t} W\_2(\mu\_{\varepsilon, t}, \mu\_t) \to 0$ where $(\mu\_t)$ is a Wasserstein geodesic from $\mu\_0$ to $\mu\_1$.

**Left-hand side:** By the joint lower semicontinuity (Theorem 29.20(i)) and the contraction property (Theorem 29.20(ii)):

$$U_\nu(\mu_t) \le \liminf_{\varepsilon \to 0}\; U_\nu(\mu_{\varepsilon, t}) \le \liminf_{\varepsilon \to 0}\; \liminf_{k \to \infty}\; U_{\nu_k}(\mu_{\varepsilon, t}^k).$$

**Right-hand side:** Since $\beta(x, y) = \beta\_{1-t}^{(K,N)}(x, y)$ is only a function of the distance $d(x, y)$, and $\lim\_{k \to \infty} \sup \lvert d\_k(x, y) - d(f\_k(x), f\_k(y)) \rvert = 0$, the integrands on $\mathcal{X}\_k$ converge uniformly to those on $\mathcal{X}$. Combined with the weak convergence $(f\_k, f\_k)\_\# \pi\_\varepsilon^k \to \pi\_\varepsilon$ and the upper semicontinuity from Theorem 29.20(iii):

$$\limsup_{\varepsilon \downarrow 0}\; \limsup_{k \to \infty}\; U_{\pi_\varepsilon^k, \nu_k}^{\beta_{1-t}^{(K,N)}}(\mu_{\varepsilon, 0}^k) \le U_{\pi, \nu}^{\beta_{1-t}^{(K,N)}}(\mu_0).$$

Combining left and right yields the desired CD inequality.

When $\beta\_t^{(K,N)}$ is not continuous (which occurs only if $N = 1$ or $\operatorname{diam}(\mathcal{X}) = D\_{K,N}$), one uses Proposition 29.10 to replace $N$ by $N' > N$, derive the inequality with bounded coefficients $\beta\_t^{(K,N')}$, and then let $N' \downarrow N$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(29.27 — Stability of Distorted Displacement Convexity)</span></p>

What the previous proof really shows is that under certain assumptions the property of distorted displacement convexity is stable under measured Gromov–Hausdorff convergence. The usual displacement convexity is a particular case (take $\beta\_t \equiv 1$).

</div>

### An Application in Riemannian Geometry

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(29.28 — Smooth MGH Limits of $\mathrm{CD}(K,N)$ Manifolds Are $\mathrm{CD}(K,N)$)</span></p>

Let $K \in \mathbb{R}$ and $N \in [1, \infty]$. If a sequence $(M\_k)\_{k \in \mathbb{N}}$ of smooth $\mathrm{CD}(K, N)$ Riemannian manifolds converges to some smooth manifold $M$ in the (pointed) measured Gromov–Hausdorff topology, then the limit also satisfies the $\mathrm{CD}(K, N)$ curvature-dimension bound.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(29.29 — Interest of Theorem 29.28)</span></p>

The proof follows at once from Theorems 29.9 and 29.25. All the interest of this theorem lies in the fact that measured Gromov–Hausdorff convergence is a very weak notion of convergence, which does **not** imply the convergence of the Ricci tensor. Note that the statement of Theorem 29.28 does not involve the definition of weak $\mathrm{CD}(K, N)$ spaces, nor any reference to optimal transport — yet its proof crucially depends on both.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(29.30 — No Analog for Upper Bounds)</span></p>

There is no analog of Theorem 29.28 for *upper* bounds on the Ricci curvature. In fact, any compact Riemannian manifold $(M, g)$ can be approximated by a sequence $(M, g\_k)\_{k \in \mathbb{N}}$ with (arbitrarily large) negative Ricci curvature.

</div>

### The Space of $\mathrm{CD}(K, N)$ Spaces

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Closedness of Weak $\mathrm{CD}(K,N)$)</span></p>

Theorem 29.24 can be summarized as: *The space of all compact metric-measure geodesic spaces satisfying a weak $\mathrm{CD}(K, N)$ bound is closed under measured Gromov–Hausdorff convergence.*

In connection with this, recall Gromov's precompactness theorem (Corollary 27.34): Given $K \in \mathbb{R}$, $N < \infty$ and $D < \infty$, the set $\mathcal{M}(K, N, D)$ of all smooth compact manifolds with dimension bounded above by $N$, Ricci curvature bounded below by $K$ and diameter bounded above by $D$ is *precompact* in the Gromov–Hausdorff topology. Then Theorem 29.24 implies that any element of the closure of $\mathcal{M}(K, N, D)$ is a compact metric-measure geodesic space satisfying $\mathrm{CD}(K, N)$, in the weak sense of Definition 29.8.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(29.31 — Collapse and Non-Trivial Reference Measures)</span></p>

Even if the limit space is smooth, it might have reference measure $\nu = e^{-\Psi}\, \mathrm{vol}$ for some nonconstant $\Psi$. Such phenomena occur in examples of **collapse** — when the dimension of the limit manifold is strictly less than the dimension of the manifolds in the converging sequence. This shows that basically any reference measure can be obtained as a limit of volume measures of higher-dimensional manifolds; it is a strong motivation to replace the class of Riemannian manifolds by the class of metric-measure spaces.

Let $(M, g)$ be a compact $n$-dimensional Riemannian manifold, let $V$ be any $C^2$ function on $M$, and let $\nu(dx) = e^{-V(x)}\, d\mathrm{vol}(x)$. Let $S^2$ stand for the usual 2-dimensional sphere with its usual metric $\sigma$. For $\varepsilon \in (0, 1)$, define $M\_\varepsilon$ to be the $e^{-V}$-warped product of $(M, g)$ by $\varepsilon\, S^2$: the $(n+2)$-dimensional manifold $M \times S^2$ equipped with the metric $g\_\varepsilon(dx, ds) = g(dx) + \varepsilon^2\, e^{-V(x)}\, \sigma(ds)$. As $\varepsilon \to 0$, $M\_\varepsilon$ collapses to $M$; more precisely $(M\_\varepsilon, g\_\varepsilon)$, seen as a metric-measure space, converges in the measured Gromov–Hausdorff sense to $(M, d, \nu)$. Moreover, if $\operatorname{Ric}\_{n+2, \nu} \ge K$, then $M\_\varepsilon$ has Ricci curvature bounded below by $K\_\varepsilon$, where $K\_\varepsilon \to K$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(29.32 — Compactness of the Space of Weak $\mathrm{CD}(K,N)$ Spaces)</span></p>

**(i)** Let $K \in \mathbb{R}$, $N < \infty$, $D < \infty$, and $0 < m \le M < \infty$. Let $\mathcal{CDD}(K, N, D, m, M)$ be the space of all compact metric-measure geodesic spaces $(\mathcal{X}, d, \nu)$ satisfying the weak curvature-dimension bound $\mathrm{CD}(K, N)$ of Definition 29.8, together with $\operatorname{diam}(\mathcal{X}, d) \le D$, $m \le \nu[\mathcal{X}] \le M$, and $\operatorname{Spt} \nu = \mathcal{X}$. Then $\mathcal{CDD}(K, N, D, m, M)$ is **compact** in the measured Gromov–Hausdorff topology.

**(ii)** Let $K \in \mathbb{R}$, $N < \infty$, $0 < m \le M < \infty$. Let $p\mathcal{CDD}(K, N, m, M)$ be the space of all pointed locally compact Polish metric-measure geodesic spaces $(\mathcal{X}, d, \nu, \star)$ satisfying the weak $\mathrm{CD}(K, N)$ bound of Definition 29.8, together with $m \le \nu[B\_1(\mathcal{X})] \le M$, and $\operatorname{Spt} \nu = \mathcal{X}$. Then $p\mathcal{CDD}(K, N, m, M)$ is **compact** in the measured Gromov–Hausdorff topology.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(29.33 — Density of Smooth Manifolds)</span></p>

It is a natural question whether smooth Riemannian manifolds, equipped with their geodesic distance and their volume measure (multiplied by a positive constant), form a dense set in $\mathcal{CDD}(K, N, D, m, M)$. The answer is negative, as will be discussed in the concluding chapter.

</div>

### First Appendix: Regularization in Metric-Measure Spaces

Regularization by convolution is a fundamental tool in real analysis. It is still available, to some extent, in metric-measure spaces.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(29.34 — Regularizing Kernels)</span></p>

Let $(\mathcal{X}, d)$ be a boundedly compact metric space equipped with a locally finite measure $\nu$, and let $\mathcal{Y}$ be a compact subset of $\mathcal{X}$. A $(\mathcal{Y}, \nu)$-**regularizing kernel** is a family of nonnegative continuous symmetric functions $(K\_\varepsilon)\_{\varepsilon > 0}$ on $\mathcal{X} \times \mathcal{X}$, such that:

- (i) $\forall x \in \mathcal{Y}$, $\int\_\mathcal{X} K\_\varepsilon(x, y)\, \nu(dy) = 1$;
- (ii) $d(x, y) > \varepsilon \implies K\_\varepsilon(x, y) = 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Construction and Properties of Regularizing Kernels)</span></p>

For any compact subset $\mathcal{Y}$ of $\operatorname{Spt} \nu$, there is a $(\mathcal{Y}, \nu)$-regularizing kernel. The construction proceeds as follows: Cover $\mathcal{Y}$ by a finite number of balls $B(x\_i, \varepsilon/2)$. Introduce a continuous subordinate partition of unity $(\phi\_i)\_{i \in I}$ satisfying $0 \le \phi\_i \le 1$, $\operatorname{Spt}(\phi\_i) \subset B(x\_i, \varepsilon/2)$, $\sum\_i \phi\_i = 1$ on $\mathcal{Y}$; only keep those $\phi\_i$ such that $\operatorname{Spt} \phi\_i \cap \mathcal{Y} \ne \emptyset$ and $\int \phi\_i\, d\nu > 0$. Then define

$$K_\varepsilon(x, y) := \sum_i \frac{\phi_i(x)\, \phi_i(y)}{\int \phi_i\, d\nu}.$$

For any $x \in \mathcal{Y}$, $\int K\_\varepsilon(x, y)\, \nu(dy) = \sum\_i \phi\_i(x) = 1$. Also $\phi\_i(x)\, \phi\_i(y)$ can be nonzero only if $x$ and $y$ both belong to $B(x\_i, \varepsilon/2)$, which implies $d(x, y) < \varepsilon$.

As soon as $\mu$ is a finite measure on $\mathcal{X}$, one may define a **continuous** function $K\_\varepsilon \mu$ on $\mathcal{X}$ by

$$(K_\varepsilon \mu)(x) := \int_\mathcal{X} K_\varepsilon(x, y)\, \mu(dy).$$

The linear operator $K\_\varepsilon : \mu \to (K\_\varepsilon \mu)\nu$ is mass-preserving and defines a (nonstrict) contraction operator on $M(\mathcal{Y})$. Moreover, as $\varepsilon \to 0$:

- If $f \in C(\mathcal{X})$, then $K\_\varepsilon f$ converges **uniformly** to $f$ on $\mathcal{Y}$.
- If $\mu$ is a finite measure supported in $\mathcal{Y}$, then $(K\_\varepsilon \mu)\nu$ converges **weakly** to $\mu$.
- If $f \in L^1(\mathcal{Y})$, then $K\_\varepsilon f$ converges to $f$ in $L^1(\mathcal{Y})$. More precisely, $\int\_{\mathcal{Y} \times \mathcal{Y}} \lvert f(x) - f(y) \rvert\, K\_\varepsilon(x, y)\, \nu(dx)\, \nu(dy) \to 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(29.35 — Doubling Measures and Kernel Bounds)</span></p>

If the measure $\nu$ is (locally) doubling, then one can ask more of the kernel $(K\_\varepsilon)$. Indeed, by Vitali's covering lemma, one can ensure that the balls $B(x\_i, \varepsilon/10)$ are disjoint. If $(\phi\_i)$ is a partition of unity associated to the covering $(B(x\_i, \varepsilon/2))$, necessarily $\phi\_i$ is identically 1 on $B(x\_i, \varepsilon/2)$, so $\int \phi\_i\, d\nu \ge \nu[B(x\_i, \varepsilon/10)] \ge C\, \nu[B(x\_i, \varepsilon)]$, where $C$ depends on the doubling constant. This gives the uniform bound:

$$K_\varepsilon(x, y) \le \frac{C}{\nu[B_\varepsilon(x)]}.$$

Together with the doubling property of $\nu$ and classical Lebesgue density theory, this guarantees that for any $f \in L^1(\mathcal{Y})$, $K\_\varepsilon f$ converges to $f$ not only in $L^1(\mathcal{Y})$ but also **almost everywhere**.

</div>

### Second Appendix: Separability of $L^1(\mathcal{X}; C(\mathcal{Y}))$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(29.36 — Separability of $L^1(C)$)</span></p>

Let $(\mathcal{X}, d)$ be a compact metric space equipped with a finite Borel measure $\mu$, let $\mathcal{Y}$ be another compact metric space, and let $f$ be a measurable function $\mathcal{X} \times \mathcal{Y} \to \mathbb{R}$, such that:

- (i) $f(x, \cdot)$ is continuous for all $x$;
- (ii) $\int\_\mathcal{X} \sup\_y \lvert f(x, y) \rvert\, d\mu(x) < +\infty$.

Then for any $\varepsilon > 0$ there is $\Psi \in C(\mathcal{X} \times \mathcal{Y})$ such that

$$\int_\mathcal{X} \sup_{y \in \mathcal{Y}} \bigl\lvert f(x, y) - \Psi(x, y) \bigr\rvert\, d\mu(x) \le \varepsilon.$$

Moreover, if a (possibly empty) compact subset $K$ of $\mathcal{X}$, and a function $h \in C(K)$ are given, such that $f(x, y) = h(x)$ for all $x \in K$, then it is possible to impose that $\Psi(x, y) = h(x)$ for all $x \in K$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Lemma 29.36)</span></p>

When $\mathcal{Y}$ is a single point, this reduces to the density of $C(\mathcal{X})$ in $L^1(\mathcal{X}, \mu)$, which is classical. For the general case: let $K$ be a compact subset of $\mathcal{X}$, $h \in C(K)$ with $f = h$ on $K$. Let $\psi \in C\_c(\mathcal{X} \setminus K)$ be such that $\lVert \psi - f \rVert\_{L^1(\mathcal{X} \setminus K, \mu)} \le \varepsilon$. Use continuous functions $\chi, \eta$ on $\mathcal{X}$, valued in $[0, 1]$, with $\chi + \eta = 1$, $\chi$ supported in a small open set $O\_\varepsilon \supset K$ and $\eta$ supported in $\mathcal{X} \setminus K$. Set $\Psi = h\chi + \psi\eta$. Then $\lVert \Psi - f \rVert\_{L^1(\mathcal{X})} \le 3\varepsilon$.

For general $\mathcal{Y}$: cover $\mathcal{Y}$ by balls $B\_\delta(y\_\ell)$, take a subordinate partition of unity $(\zeta\_\ell)$. For each $\ell$, $f(\cdot, y\_\ell)$ is $\mu$-integrable, so approximate it by $\psi\_\ell \in C(\mathcal{X})$ with $\psi\_\ell = h$ on $K$. Define $\Psi(x, y) := \sum\_\ell \psi\_\ell(x)\, \zeta\_\ell(y)$. Then

$$\int_\mathcal{X} \sup_y \lvert f(x, y) - \Psi(x, y) \rvert\, d\mu(x) \le \int_\mathcal{X} m_\delta(x)\, \mu(dx) + L(\delta)\, \eta,$$

where $m\_\delta(x) = \sup\lbrace \lvert f(x, z) - f(x, z') \rvert;\; d(z, z') \le \delta \rbrace$. Since $f(x, \cdot)$ is continuous, $m\_\delta(x) \searrow 0$ as $\delta \to 0$, so by monotone convergence the integral goes to 0. Choosing $\delta$ small enough then $\eta$ small enough concludes the proof.

</div>

### Bibliographical Notes on Chapter 29

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes — Genesis of Definition 29.8)</span></p>

Definition 29.8 comes after a series of particular cases and/or variants studied by **Lott and Villani** on the one hand, and **Sturm** on the other. In a first step, Lott and Villani treated $\mathrm{CD}(K, \infty)$ and $\mathrm{CD}(0, N)$, while Sturm independently treated $\mathrm{CD}(K, \infty)$. These cases can be handled with just displacement convexity. Then it took some time before **Sturm** came up with the brilliant idea to use *distorted displacement* as the basis of the definition of $\mathrm{CD}(K, N)$ for $N < \infty$ and $K \ne 0$.

There are slight variations in the definitions across these works. In the case $K = 0$, Definition 29.8 is exactly the definition used by Lott–Villani. In the case $N = \infty$, the definition in Lott–Villani was about the same as Definition 29.8, but based on inequality (29.2) instead of (29.3). Sturm also used a similar definition, but preferred to impose weak displacement convexity only for the Boltzmann $H$ functional, i.e. $U(r) = r \log r$, not for the whole class $\mathcal{DC}\_\infty$. For the general $\mathrm{CD}(K, N)$ criterion, Sturm's original definition has three differences from Definition 29.8: (a) the basic inequality is imposed only for functions of the form $-r^{1-1/N'}$ with $N' \ge N$, not all $U \in \mathcal{DC}\_N$; (b) the displacement interpolation and coupling are not required to be related via a dynamical optimal transference plan; (c) $\mu\_0, \mu\_1$ are imposed to be absolutely continuous rather than just having support in $\operatorname{Spt} \nu$.

Sturm proved the stability of his definition under a variant of measured Gromov–Hausdorff convergence (provided that one stays away from the limit Bonnet–Myers diameter). Then Lott and Villani briefly sketched a proof of stability for their modified definition. Details appear in the present notes for the first time, in particular the proof of upper semicontinuity of $U\_{\pi,\nu}^\beta(\mu)$ under regularization (Theorem 29.20(iii)).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Alternative Definitions — Rejected Approaches)</span></p>

Several tentative definitions have been rejected for various reasons:

**(i) Convexity along all geodesics.** Imposing the displacement convexity inequality along *all* displacement interpolations in Definition 29.8, rather than along *some*. This concept is not stable under measured Gromov–Hausdorff convergence.

**(ii) Pointwise displacement convexity.** Replace the *integrated* displacement convexity inequalities by *pointwise* inequalities. This notion is a priori stronger and there is no evidence it should be stable.

**(iii) Use inequality (29.2) instead of (29.3).** In the case $K < 0$, this inequality is stable (due to convexity of $-r^{1-1/N}$ and a priori regularity of the speed field from Theorem 28.5). For $K > 0$ there is no reason to expect stability. Moreover, basing the definition on (29.2) seems to make it very difficult, if not impossible, to derive sharp geometric inequalities (Bishop–Gromov, Bonnet–Myers).

**(iv) Measure contraction property (MCP).** Define $\mathrm{MCP}(K, N)$ using a conditional probability $P^{(t)}(x, y; dz)$ on the set of $t$-barycenters. This approach has two drawbacks: it does not extend to $N = \infty$, and on a Riemannian manifold $\mathrm{MCP}(K, N)$ does *not* imply $\mathrm{CD}(K, N)$ unless $N$ coincides with the true dimension. On the other hand, $\mathrm{CD}(K, N)$ implies $\mathrm{MCP}(K, N)$ in a nonbranching space. The MCP property is known to hold for finite-dimensional Alexandrov spaces. An interesting application is the analysis on the Heisenberg group, which does not satisfy any $\mathrm{CD}(K, N)$ bound but still satisfies $\mathrm{MCP}(0, 5)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Other Approaches to Synthetic Ricci Bounds)</span></p>

**Lott** noticed that (for a Riemannian manifold) at least $\mathrm{CD}(0, N)$ bounds can be formulated in terms of displacement convexity of certain functionals explicitly involving the time variable. For instance, $\mathrm{CD}(0, N)$ is equivalent to the convexity of $t \to t\, U\_\nu(\mu\_t) + N\, t \log t$ on $[0, 1]$, along displacement interpolation, for all $U \in \mathcal{DC}\_\infty$; rather than convexity of $U\_\nu(\mu\_t)$ for all $U \in \mathcal{DC}\_N$.

**Bonciocat and Sturm** modified the definition of weak $\mathrm{CD}(K, \infty)$ spaces to allow for a fixed "resolution error" $\delta$ in the measurement of distances, defining "$\delta$-approximately $\mathrm{CD}(K, \infty)$" spaces. This extends the scope to discrete spaces and even non-length spaces; any weak $\mathrm{CD}(K, \infty)$ space is a limit of $\delta$-approximate $\mathrm{CD}(K, \infty)$ discrete spaces as $\delta \to 0$.

**Joulin** and **Ollivier** independently suggested defining the infimum of the Ricci curvature as the best constant $K$ in the contraction inequality

$$W_1(P_t \delta_x, P_t \delta_y) \le e^{-Kt}\, d(x, y),$$

where $P\_t$ is the heat semigroup on probability measures. In a Riemannian setting, **Sturm and von Renesse** justified this definition by showing that one recovers the usual Ricci curvature bound. This point of view is natural when the problem includes a distinguished Markov kernel; in particular, it leads to the possibility of treating discrete spaces equipped with a random walk. **Ollivier** has demonstrated the geometric interest of this notion on an impressive list of examples, mostly in discrete spaces, and has derived geometric consequences such as concentration of measure.

**Kontsevich and Soibelman** proposed a completely different approach to Ricci bounds in metric-measure spaces, in relation to Quantum Field Theory, mirror symmetry and heat kernels.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Notes on the Examples)</span></p>

- **Example 29.13** (Euclidean space with log-concave measure) can be generalized: If $\nu$ is a finite measure on $\mathbb{R}^n$ such that $H\_\nu$ is displacement convex, then $\nu$ takes the form $e^{-V}\, \mathcal{H}^k$, where $V$ is lower semicontinuous and $\mathcal{H}^k$ is the $k$-dimensional Hausdorff measure, $k = \dim(\operatorname{Spt} \nu)$. The result extends to infinite-dimensional separable Hilbert spaces: $H\_\nu$ is displacement convex if and only if $\nu$ is log-concave.
- **Example 29.15** (quotient manifold by a compact Lie group) was treated by Lott–Villani: the quotient of a $\mathrm{CD}(K, N)$ Riemannian manifold by a compact Lie group action is still a weak $\mathrm{CD}(K, N)$ space (when $K = 0$ or $N = \infty$; the same theorem is certainly true for all values of $K$ and $N$).
- **Example 29.31** (collapse and warped products) was explained to the author by Lott. This example shows that a lower bound on the Ricci curvature is not enough to ensure the continuity of the Hausdorff measure under measured Gromov–Hausdorff convergence — such a phenomenon is necessarily linked with *collapsing* (loss of dimension).

</div>

## Chapter 30: Weak Ricci Curvature Bounds II — Geometric and Analytic Properties

In the previous chapter, the concept of weak curvature-dimension bound was introduced (extending the classical $\mathrm{CD}(K, N)$ from smooth Riemannian manifolds to metric-measure geodesic spaces) and shown to be stable under measured Gromov–Hausdorff convergence. This last chapter presents the state of the art concerning the geometric and analytic properties of weak $\mathrm{CD}(K, N)$ spaces.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convention for Chapter 30)</span></p>

Throughout the sequel, a "weak $\mathrm{CD}(K, N)$ space" is a locally compact, complete separable geodesic space $(\mathcal{X}, d)$ equipped with a locally finite Borel measure $\nu$, satisfying a weak $\mathrm{CD}(K, N)$ condition as in Definition 29.8.

</div>

### Elementary Properties

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(30.1 — Elementary Consequences of Weak $\mathrm{CD}(K,N)$ Bounds)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, N)$ space. Then:

**(i)** If $\mathcal{X}'$ is a totally convex closed subset of $\mathcal{X}$, then $\mathcal{X}'$ inherits from $(\mathcal{X}, d, \nu)$ a natural structure of metric-measure geodesic space, and $\mathcal{X}'$ is also a weak $\mathrm{CD}(K, N)$ space.

**(ii)** For any $\alpha > 0$, $(\mathcal{X}, d, \alpha\nu)$ is a weak $\mathrm{CD}(K, N)$ space.

**(iii)** For any $\lambda > 0$, $(\mathcal{X}, \lambda d, \nu)$ is a weak $\mathrm{CD}(\lambda^{-2}K, N)$ space.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.2 — Restriction of the $\mathrm{CD}(K,N)$ Property to the Support)</span></p>

A metric-measure space $(\mathcal{X}, d, \nu)$ is a weak $\mathrm{CD}(K, N)$ space if and only if $(\operatorname{Spt} \nu, d, \nu)$ is itself a weak $\mathrm{CD}(K, N)$ space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.3 — Support Reduction and Convergence)</span></p>

Theorem 30.2 allows us to systematically reduce to the case $\operatorname{Spt} \nu = \mathcal{X}$ when studying properties of weak $\mathrm{CD}(K, N)$ spaces. The reason it is useful to allow $\mathcal{X}$ to be larger than $\operatorname{Spt} \nu$ is for *convergence* issues: a sequence $(\mathcal{X}\_k, d\_k, \nu\_k)$ with $\operatorname{Spt} \nu\_k = \mathcal{X}\_k$ may converge to $(\mathcal{X}, d, \nu)$ with $\operatorname{Spt} \nu$ strictly smaller than $\mathcal{X}$. This "reduction of support" is impossible if $N < \infty$ (by Theorem 29.32), but can occur in the case $N = \infty$ (e.g. $\mathcal{X}\_k = (\mathbb{R}^n, \lvert \cdot \rvert)$ with sharply peaked Gaussian measures converging to a Dirac mass).

</div>

### Displacement Convexity

The definition of weak $\mathrm{CD}(K, N)$ spaces is based upon displacement convexity inequalities, but these are only required to hold under the assumption that $\mu\_0$ and $\mu\_1$ are compactly supported. To exploit the full strength of displacement convexity, it is important to get rid of this restriction.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.4 — Domain of Definition of $U\_\nu$ and $U\_{\pi,\nu}^\beta$ on Noncompact Spaces)</span></p>

Let $(\mathcal{X}, d)$ be a boundedly compact Polish space, equipped with a locally finite measure $\nu$, and let $z$ be any point in $\mathcal{X}$. Let $K \in \mathbb{R}$, $N \in [1, \infty]$, and $U \in \mathcal{DC}\_N$. For any measure $\mu$ on $\mathcal{X}$, with Lebesgue decomposition $\mu = \rho\, \nu + \mu\_s$, let $\pi(dy \vert x)$ be a family of conditional probability measures on $\mathcal{X}$, and let $\pi(dx\, dy) = \mu(dx)\, \pi(dy \vert x)$. Assume that

$$\int_{\mathcal{X} \times \mathcal{X}} d(x,y)^2\, \pi(dx\, dy) < +\infty; \qquad \int_\mathcal{X} d(z, x)^p\, \mu(dx) < +\infty,$$

where $p \ge 2$ is such that condition (30.4) holds (a growth condition on $\nu$ at infinity). Then for any $t \in [0, 1]$, the expressions $U\_\nu(\mu)$ and $U\_{\pi,\nu}^{\beta\_t^{(K,N)}}(\mu)$ make sense in $\mathbb{R} \cup \lbrace \pm\infty \rbrace$ and can be taken as generalized definitions of the functionals appearing in Definition 29.1.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.5 — Displacement Convexity Inequalities in Weak $\mathrm{CD}(K,N)$ Spaces)</span></p>

Let $N \in [1, \infty]$, let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, N)$ space, and let $p \in [2, +\infty) \cup \lbrace c \rbrace$ satisfy condition (30.4). Let $\mu\_0$ and $\mu\_1$ be two probability measures in $P\_p(\mathcal{X})$, whose supports are included in $\operatorname{Spt} \nu$. Then there exists a Wasserstein geodesic $(\mu\_t)\_{0 \le t \le 1}$, and an associated optimal coupling $\pi$ of $(\mu\_0, \mu\_1)$ such that, for all $U \in \mathcal{DC}\_N$ and for all $t \in [0, 1]$,

$$U_\nu(\mu_t) \le (1 - t)\, U_{\pi,\nu}^{\beta_{1-t}^{(K,N)}}(\mu_0) \;+\; t\, U_{\check{\pi},\nu}^{\beta_t^{(K,N)}}(\mu_1).$$

Furthermore, if $N = \infty$, one also has

$$U_\nu(\mu_t) \le (1-t)\, U_\nu(\mu_0) + t\, U_\nu(\mu_1) - \frac{\lambda(K, U)\, t(1-t)}{2}\; W_2(\mu_0, \mu_1)^2,$$

where

$$\lambda(K, U) = \inf_{r > 0} \frac{K\, p(r)}{r} = \begin{cases} K\, p'(0) & \text{if } K > 0 \\\\ 0 & \text{if } K = 0 \\\\ K\, p'(\infty) & \text{if } K < 0. \end{cases}$$

These inequalities are the starting point for *all* subsequent inequalities in this chapter.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.6 — Lower Semicontinuity of $U\_\nu$ on Noncompact Spaces)</span></p>

Let $(\mathcal{X}, d)$ be a boundedly compact Polish space, equipped with a locally finite measure $\nu$ such that $\operatorname{Spt} \nu = \mathcal{X}$. Let $U : \mathbb{R}\_+ \to \mathbb{R}$ be a continuous convex function, with $U(0) = 0$, $U(r) \ge -c\, r$ for some $c \in \mathbb{R}$. Then:

**(i)** For any $\mu \in M\_+(\mathcal{X})$ and any sequence $(\mu\_k)$ converging weakly to $\mu$ in $M\_+(\mathcal{X})$,

$$U_\nu(\mu) \le \liminf_{k \to \infty}\; U_\nu(\mu_k).$$

**(ii)** Assume further that $\mu \in P\_2(\mathcal{X})$, and let $\beta(x, y)$ be a positive measurable function on $\mathcal{X} \times \mathcal{X}$, with $\lvert \log \beta(x, y) \rvert = O(d(x, y)^2)$. Then there is a sequence $(\mu\_k)$ of compactly supported probability measures on $\mathcal{X}$ such that $\mu\_k \to \mu$ in $P\_2(\mathcal{X})$, and for any sequence of probability measures $(\pi\_k)$ such that $\pi\_k$ has first marginal $\mu\_k$ and second marginal $\mu\_{k,1}$, with $\pi\_k \to \pi$ weakly, $\int d(x,y)^2\, \pi\_k \to \int d(x,y)^2\, \pi$, and $\mu\_{k,1} \to \mu\_1$ in $P\_2(\mathcal{X})$, one has

$$\lim_{k \to \infty} U_{\pi_k, \nu}^\beta(\mu_k) = U_{\pi, \nu}^\beta(\mu).$$

</div>

### Brunn–Minkowski Inequality

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.7 — Brunn–Minkowski Inequality in Weak $\mathrm{CD}(K,N)$ Spaces)</span></p>

Let $K \in \mathbb{R}$ and $N \in [1, \infty]$. Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, N)$ space, let $A\_0$, $A\_1$ be two compact subsets of $\operatorname{Spt} \nu$, and let $t \in (0, 1)$. Then:

- If $N < \infty$,

$$\nu\bigl[[A_0, A_1]_t\bigr]^{1/N} \ge (1-t) \left[\inf_{(x_0, x_1) \in A_0 \times A_1} \beta_{1-t}^{(K,N)}(x_0, x_1)^{1/N}\right] \nu[A_0]^{1/N} + t \left[\inf_{(x_0, x_1) \in A_0 \times A_1} \beta_t^{(K,N)}(x_0, x_1)^{1/N}\right] \nu[A_1]^{1/N}.$$

In particular, if $N < \infty$ and $K \ge 0$, then $\nu[[A\_0, A\_1]\_t]^{1/N} \ge (1-t)\, \nu[A\_0]^{1/N} + t\, \nu[A\_1]^{1/N}$.

- If $N = \infty$, then

$$\log \frac{1}{\nu[[A_0, A_1]_t]} \le (1-t) \log \frac{1}{\nu[A_0]} + t \log \frac{1}{\nu[A_1]} - \frac{K\, t(1-t)}{2} \sup_{x_0 \in A_0, x_1 \in A_1} d(x_0, x_1)^2.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.8 — Sets Not in the Support)</span></p>

The result fails if $A\_0$, $A\_1$ are not assumed to lie in the support of $\nu$. (Take $\nu = \delta\_{x\_0}$, $x\_1 \ne x\_0$, and $A\_0 = \lbrace x\_0 \rbrace$, $A\_1 = \lbrace x\_1 \rbrace$.)

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(30.9 — Nonatomicity of the Support)</span></p>

Let $K \in \mathbb{R}$ and $N \in [1, \infty]$. If $(\mathcal{X}, d, \nu)$ is a weak $\mathrm{CD}(K, N)$ space, then either $\nu$ is a Dirac mass, or $\nu$ has no atom.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(30.10 — Exhaustion by Intermediate Points)</span></p>

Let $K \in \mathbb{R}$ and $N \in [1, \infty)$. Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, N)$ space, let $A$ be a compact subset of $\operatorname{Spt} \nu$, and let $x \in A$. Then

$$\nu\bigl[[x, A]_t\bigr] \xrightarrow{t \to 1} \nu[A].$$

</div>

### Bishop–Gromov Inequalities

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.11 — Bishop–Gromov Inequality in Metric-Measure Spaces)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, N)$ space and let $x\_0 \in \operatorname{Spt} \nu$. Then, for any $r > 0$, $\nu[B[x\_0, r]] = \nu[B(x\_0, r)]$. Moreover,

- If $N < \infty$, then

$$\frac{\nu[B_r(x_0)]}{\int_0^r s^{(K,N)}(t)\, dt}$$

is a nonincreasing function of $r$, where $s^{(K,N)}$ is defined as in Theorem 18.8.

- If $N = \infty$, then for any $\delta > 0$ there exists a constant $C = C(K\_-, \delta, \nu[B\_\delta(x\_0)], \nu[B\_{2\delta}(x\_0)])$ such that for all $r \ge \delta$,

$$\nu[B_r(x_0)] \le e^{Cr}\, e^{(K_-)^{r^2/2}};$$

$$\nu[B_{r+\delta}(x_0) \setminus B_r(x_0)] \le e^{Cr}\, e^{-K\, r^2/2} \qquad \text{if } K > 0.$$

In particular, if $K' < K$ then $\int e^{(K'/2)\, d(x\_0, x)^2}\, \nu(dx) < +\infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(30.12 — Measure of Small Balls in Weak $\mathrm{CD}(K,N)$ Spaces)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, N)$ space and let $z \in \operatorname{Spt} \nu$. Then for any $R > 0$ there is a constant $c = c(K, N, R)$ such that if $B(x\_0, r) \subset B(z, R)$ then

$$\nu[B(x_0, r)] \ge \bigl(c\, \nu[B(z, R)]\bigr)\, r^N.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(30.13 — Dimension of Weak $\mathrm{CD}(K,N)$ Spaces)</span></p>

If $\mathcal{X}$ is a weak $\mathrm{CD}(K, N)$ space with $K \in \mathbb{R}$ and $N \in [1, \infty)$, then the Hausdorff dimension of $\operatorname{Spt} \nu$ is at most $N$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(30.14 — Weak $\mathrm{CD}(K,N)$ Spaces Are Locally Doubling)</span></p>

If $\mathcal{X}$ is a weak $\mathrm{CD}(K, N)$ space with $K \in \mathbb{R}$, $N < \infty$, $\operatorname{Spt} \nu = \mathcal{X}$, then $(\mathcal{X}, d, \nu)$ is $C$-doubling on each ball $B(z, R)$, with a constant $C$ depending only on $K$, $N$ and $R$. In particular if $\operatorname{diam}(\mathcal{X}) \le D$ then $(\mathcal{X}, d, \nu)$ is $C$-doubling with a constant $C = C(K, N, D)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.15–30.16)</span></p>

- Corollary 30.14, combined with the general theory of Gromov–Hausdorff convergence (Chapter 27), implies the compactness Theorem 29.32.
- It is natural to ask whether the equality $N = \dim(\mathcal{X})$ in Corollary 30.13 forces $\nu$ to be proportional to the $N$-dimensional Hausdorff measure.

</div>

### Uniqueness of Geodesics

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.17 — Unique Geodesics in Nonbranching $\mathrm{CD}(K,N)$ Spaces)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a nonbranching weak $\mathrm{CD}(K, N)$ space with $K \in \mathbb{R}$ and $N \in [1, \infty)$. Then for $\nu \otimes \nu$-almost any $(x, y) \in \mathcal{X} \times \mathcal{X}$, there is a **unique** geodesic joining $x$ to $y$. More precisely, for any $x \in \operatorname{Spt} \nu$, the set of points $y \in \operatorname{Spt} \nu$ which can be joined to $x$ by several geodesics has zero $\nu$-measure.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.18)</span></p>

The restriction $N < \infty$ seems natural, but a counterexample for $N = \infty$ is not known.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Theorem 30.17)</span></p>

By Theorem 30.2 we may assume $\operatorname{Spt} \nu = \mathcal{X}$. Let $x \in \mathcal{X}$, $r > 0$, $A = B\_r(x)$ and $A\_t = [x, B\_r(x)]\_t \subset B\_{tr}(x)$. For any $z \in A\_t$, there is a geodesic $\gamma$ joining $x$ to some $y \in Z$, with $\gamma(t) = z$. Assume there is a distinct geodesic $\widetilde{\gamma}$ also joining $x$ to $z$; up to rescaling of time, $\widetilde{\gamma}(0) = x$, $\widetilde{\gamma}(t) = z$. Then the concatenation of $\widetilde{\gamma}$ on $[0, t]$ and $\gamma$ on $[t, 1]$ is a geodesic that coincides with $\gamma$ on the nontrivial interval $[t, 1]$. Since $\mathcal{X}$ is nonbranching, this is impossible — so there is only one geodesic from $x$ to $z$.

Let $Z := \bigcup\_{0 < t < 1} A\_t$. The preceding reasoning shows that for any $z \in Z$ there is a unique geodesic from $x$ to $z$. The sets $A\_t$ are nondecreasing in $t$, so $\nu[Z] = \lim\_{t \to 1} \nu[[x, A]\_t] = \nu[A]$ by Corollary 30.10. So for any $k$, the set $Z\_k$ of points in $B\_k(x)$ that can be joined to $x$ by several geodesics has $\nu$-measure zero. The set of *all* such points is contained in $\bigcup\_k Z\_k$, and therefore also has $\nu$-measure zero.

</div>

### Regularity of the Interpolant

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.19 — Regularity of Interpolants in Weak $\mathrm{CD}(K,N)$ Spaces)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, N)$ space with $K \in \mathbb{R}$ and $N \in [1, \infty)$. Further, let $\mu\_0, \mu\_1$ be two probability measures in $P\_2(\mathcal{X})$ with $\operatorname{Spt} \mu\_0 \subset \operatorname{Spt} \nu$, $\operatorname{Spt} \mu\_1 \subset \operatorname{Spt} \nu$. Then:

**(i)** Assume that both $\mu\_0$ and $\mu\_1$ are absolutely continuous with respect to $\nu$; if $K < 0$, further assume that they are compactly supported. Let $(\mu\_t)\_{0 \le t \le 1}$ be a Wasserstein geodesic satisfying the displacement convexity inequalities of Theorem 30.5. Then also $\mu\_t$ is absolutely continuous, for all $t \in [0, 1]$.

**(ii)** If either $\mu\_0$ or $\mu\_1$ is absolutely continuous, and $t\_0 \in (0, 1)$ is given, then one can find a Wasserstein geodesic joining $\mu\_0$ to $\mu\_1$ such that $\mu\_{t\_0}$ is also absolutely continuous.

**(iii)** If either $\mu\_0$ or $\mu\_1$ is not purely singular, then one can find a Wasserstein geodesic joining $\mu\_0$ to $\mu\_1$ such that for any $t \in [0, 1]$, $\mu\_t$ is not purely singular.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.20 — Uniform Bound on the Interpolant in Nonnegative Curvature)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(0, \infty)$ space and let $\mu\_0, \mu\_1 \in P^{\mathrm{ac}}(\mathcal{X})$, with bounded respective densities $\rho\_0, \rho\_1$. Let $(\mu\_t)\_{0 \le t \le 1}$ be a Wasserstein geodesic satisfying the displacement convexity inequalities of Theorem 30.5. Then the density $\rho\_t$ of $\mu\_t$ is bounded by

$$\lVert \rho_t \rVert_{L^\infty(\nu)} \le \max\!\bigl(\lVert \rho_0 \rVert_{L^\infty(\nu)},\; \lVert \rho_1 \rVert_{L^\infty(\nu)}\bigr).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.21 — Role of the Joint Inequality)</span></p>

The proof of Theorem 30.20 exploits the fact that in the definition of weak $\mathrm{CD}(K, N)$ spaces, the displacement convexity inequality is required to hold *for all members* of $\mathcal{DC}\_N$ and *along a common Wasserstein geodesic*. Using $U(r) = r^p$ for all $p \ge 1$ and then passing to the limit $p \to \infty$ gives the $L^\infty$ bound.

</div>

### HWI and Logarithmic Sobolev Inequalities

There is a generalized notion of **Fisher information** in a metric-measure space $(\mathcal{X}, d, \nu)$:

$$I_\nu(\mu) = \int \frac{\lvert \nabla^- \rho \rvert^2}{\rho}\, d\nu, \qquad \mu = \rho\, \nu,$$

where $\lvert \nabla^- \rho \rvert$ is the descending slope defined by (20.2).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.22 — HWI and Log Sobolev Inequalities in Weak $\mathrm{CD}(K,\infty)$ Spaces)</span></p>

Let $K \in \mathbb{R}$ and let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, \infty)$ space. Further, let $\mu\_0$ and $\mu\_1$ be two probability measures in $P\_2(\mathcal{X})$, such that $\mu\_0 = \rho\_0\, \nu$ with $\rho\_0$ Lipschitz. Then

$$H_\nu(\mu_0) \le H_\nu(\mu_1) + W_2(\mu_0, \mu_1)\, \sqrt{I_\nu(\mu_0)} - \frac{K\, W_2(\mu_0, \mu_1)^2}{2}.$$

In particular, if $\nu$ is a probability measure and $K > 0$, taking $\mu\_1 = \nu$ gives the **logarithmic Sobolev inequality**:

$$H_\nu(\mu_0) \le \frac{1}{2K}\; I_\nu(\mu_0).$$

</div>

### Sobolev Inequalities

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.23 — Sobolev Inequality in Weak $\mathrm{CD}(K,N)$ Spaces)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, N)$ space, where $K < 0$ and $N \in [1, \infty)$. Then, for any $R > 0$ there exist constants $A = A(K, N, R)$ and $B = B(K, N, R)$ such that for any Lipschitz function $u$ supported in a ball $B(z, R)$,

$$\lVert u \rVert_{L^{N/(N-1)}(\nu)} \le A\, \lVert \nabla^- u \rVert_{L^1(\nu)} + B\, \lVert u \rVert_{L^1(\nu)}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.24 — Sharp Sobolev Inequalities)</span></p>

It is not known whether weak $\mathrm{CD}(K, N)$ spaces with $K > 0$ and $N < \infty$ satisfy sharp Sobolev inequalities such as (21.8).

</div>

### Diameter Control

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bonnet–Myers and Diameter Bounds)</span></p>

Recall from Proposition 29.11 that a weak $\mathrm{CD}(K, N)$ space with $K > 0$ and $N < \infty$ satisfies the Bonnet–Myers diameter bound $\operatorname{diam}(\operatorname{Spt} \nu) \le \pi\sqrt{(N-1)/K}$.

Slightly weaker conclusions can also be obtained under a priori weaker assumptions: For instance, if $\mathcal{X}$ is at the same time a weak $\mathrm{CD}(0, N)$ space and a weak $\mathrm{CD}(K, \infty)$ space, then there is a universal constant $C$ such that $\operatorname{diam}(\operatorname{Spt} \nu) \le C\sqrt{(N-1)/K}$.

</div>

### Poincaré Inequalities

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.25 — Global Poincaré Inequalities in Weak $\mathrm{CD}(K,N)$ Spaces)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, N)$ space with $K > 0$ and $N \in (1, \infty]$. Then, for any Lipschitz function $f : \operatorname{Spt} \nu \to \mathbb{R}$,

$$\int f\, d\nu = 0 \implies \int f^2\, d\nu \le \left(\frac{N-1}{NK}\right) \int \lvert \nabla^- f \rvert^2\, d\nu,$$

with the convention that $(N-1)/N = 1$ if $N = \infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.26 — Local Poincaré Inequalities in Nonbranching $\mathrm{CD}(K,N)$ Spaces)</span></p>

Let $K \in \mathbb{R}$, $N \in [1, \infty)$, and let $(\mathcal{X}, d, \nu)$ be a nonbranching weak $\mathrm{CD}(K, N)$ space. Let $u : \operatorname{Spt} \nu \to \mathbb{R}$ be a Lipschitz function, and let $x\_0 \in \operatorname{Spt} \nu$. For any $R > 0$, if $r \le R$ then

$$\fint_{B_r(x_0)} \left\lvert u(x) - \langle u \rangle_{B_r(x_0)} \right\rvert\, d\nu(x) \le P(K, N, R)\, r \fint_{B_{2r}(x_0)} \lvert \nabla u \rvert(x)\, d\nu(x),$$

where $\fint\_B$ stands for the averaged integral over $B$; $\langle u \rangle\_B$ for the average of $u$ on $B$; $P(K, N, R) = 2^{2N+1}\, C(K, N, R)\, D(K, N, R)$; $C(K, N, R)$, $D(K, N, R)$ are defined by (19.11) and (18.10) respectively. In particular, if $K \ge 0$ then $P(K, N, R) = 2^{2N+1}$ is admissible; so $\nu$ satisfies a uniform local Poincaré inequality. Moreover, (30.33) still holds if $\lvert \nabla u \rvert$ is replaced by any **upper gradient** of $u$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.27)</span></p>

It would be desirable to eliminate the nonbranching condition, since it is not always satisfied by weak $\mathrm{CD}(K, N)$ spaces, and rather unnatural in the theory of local Poincaré inequalities.

</div>

### Talagrand Inequalities

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.28 — Talagrand Inequalities and Weak Curvature Bounds)</span></p>

**(i)** Let $(\mathcal{X}, d, \nu)$ be a weak $\mathrm{CD}(K, \infty)$ space with $K > 0$. Then $\nu$ lies in $P\_2(\mathcal{X})$ and satisfies the Talagrand inequality $T\_2(K)$.

**(ii)** Let $(\mathcal{X}, d, \nu)$ be a locally compact Polish geodesic space equipped with a locally doubling measure $\nu$, satisfying a local Poincaré inequality. If $\nu$ satisfies a logarithmic Sobolev inequality for some constant $K > 0$, then $\nu$ lies in $P\_2(\mathcal{X})$ and satisfies the Talagrand inequality $T\_2(K)$.

**(iii)** Let $(\mathcal{X}, d, \nu)$ be a locally compact Polish geodesic space. If $\nu$ satisfies a Talagrand inequality $T\_2(K)$ for some $K > 0$, then it also satisfies a global Poincaré inequality with constant $K$.

**(iv)** Let $(\mathcal{X}, d, \nu)$ be a locally compact Polish geodesic space equipped with a locally doubling measure $\nu$, satisfying a local Poincaré inequality. If $\nu$ satisfies a global Poincaré inequality, then it also satisfies a modified logarithmic Sobolev inequality and a quadratic-linear transportation inequality as in Theorem 22.25.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.29)</span></p>

In view of Corollary 30.14 and Theorem 30.26, the regularity assumptions required in (ii) are satisfied if $(\mathcal{X}, d, \nu)$ is a nonbranching weak $\mathrm{CD}(K', N')$ space for some $K' \in \mathbb{R}$, $N' < \infty$; note that the values of $K'$ and $N'$ do not play any role in the conclusion.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.30 — Hamilton–Jacobi Semigroup in Metric Spaces)</span></p>

Let $L : \mathbb{R}\_+ \to \mathbb{R}\_+$ be a strictly increasing, locally semiconcave, convex continuous function such that $L(0) = 0$. Let $(\mathcal{X}, d)$ be a locally compact Polish geodesic space equipped with a reference measure $\nu$, locally doubling and satisfying a local Poincaré inequality. For any $f \in C\_b(\mathcal{X})$, define the evolution $(H\_t f)\_{t \ge 0}$ by

$$H_0 f = f; \qquad (H_t f)(x) = \inf_{y \in \mathcal{X}} \left[f(y) + t\, L\!\left(\frac{d(x, y)}{t}\right)\right] \quad (t > 0).$$

Then Properties (i)–(vi) of Theorem 22.46 remain true, up to the replacement of $M$ by $\mathcal{X}$. Moreover, the following weakened version of (vii) holds:

**(vii')** For $\nu$-almost any $x \in \mathcal{X}$ and any $t > 0$,

$$\lim_{s \downarrow 0} \frac{(H_{t+s} f)(x) - (H_t f)(x)}{s} = -L^\ast\!\bigl(\lvert \nabla^- H_t f \rvert\bigr);$$

this conclusion extends to $t = 0$ if $\lVert f \rVert\_{\mathrm{Lip}} \le L'(\infty)$ and $f$ is locally Lipschitz.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.31 — Dimensional Talagrand Inequalities)</span></p>

There are also dimensional versions of Talagrand inequalities available; for instance the analog of Theorem 22.37 holds true in weak $\mathrm{CD}(K, N)$ spaces with $K > 0$ and $N < \infty$.

</div>

### Equivalence of Definitions in Nonbranching Spaces

In the definition of weak $\mathrm{CD}(K, N)$ spaces we chose to impose the displacement convexity inequality for all $U \in \mathcal{DC}\_N$, but only along *some* displacement interpolation. In *nonbranching* metric-measure spaces, the choice really does not matter. It is equivalent:

- to require the inequality for any $U \in \mathcal{DC}\_N$; or just for $U = U\_N$, where $U\_N(r) = -Nr^{1-1/N}$ if $1 < N < \infty$, and $U\_\infty(r) = r \log r$;
- to require the inequality for compactly supported, absolutely continuous measures $\mu\_0, \mu\_1$; or for any two probability measures with suitable moment conditions;
- to require the inequality along *some* displacement interpolation, or along *any* displacement interpolation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.32 — Equivalent Definitions of $\mathrm{CD}(K,N)$ in Nonbranching Spaces)</span></p>

Let $(\mathcal{X}, d, \nu)$ be a nonbranching locally compact Polish geodesic space equipped with a locally finite measure $\nu$. Let $K \in \mathbb{R}$, $N \in (1, \infty]$, and let $p \in [2, +\infty) \cup \lbrace c \rbrace$ satisfy the assumptions of Theorem 30.4. Then the following three properties are equivalent:

**(i)** $(\mathcal{X}, d, \nu)$ is a weak $\mathrm{CD}(K, N)$ space, in the sense of Definition 29.8;

**(ii)** For any two compactly supported continuous probability densities $\rho\_0$ and $\rho\_1$, there is a displacement interpolation $(\mu\_t)\_{0 \le t \le 1}$ joining $\mu\_0 = \rho\_0\, \nu$ to $\mu\_1 = \rho\_1\, \nu$, and an associated optimal plan $\pi$, such that for all $t \in [0, 1]$,

$$H_{N,\nu}(\mu_t) \le (1-t)\, H_{N,\pi,\nu}^{\beta_{1-t}^{(K,N)}}(\mu_0) \;+\; t\, H_{N,\check{\pi},\nu}^{\beta_t^{(K,N)}}(\mu_1).$$

**(iii)** For **any** displacement interpolation $(\mu\_t)\_{0 \le t \le 1}$ with $\mu\_0, \mu\_1 \in P\_p(\mathcal{X})$, for **any** associated transport plan $\pi$, for **any** $U \in \mathcal{DC}\_N$ and for **any** $t \in [0, 1]$,

$$U_\nu(\mu_t) \le (1-t)\, U_{\pi,\nu}^{\beta_{1-t}^{(K,N)}}(\mu_0) \;+\; t\, U_{\check{\pi},\nu}^{\beta_t^{(K,N)}}(\mu_1).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.33)</span></p>

In the case $N = 1$, (30.35) does not make sense, but the equivalence (i) $\Rightarrow$ (iii) still holds. This can be seen by working in dimension $N' > 1$ and letting $N' \downarrow 1$, as in the proof of Theorem 17.41.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of Theorem 30.32)</span></p>

Clearly (iii) $\Rightarrow$ (i) $\Rightarrow$ (ii). The core of the proof is showing (ii) $\Rightarrow$ (iii). The proof proceeds in four steps:

**Step 1:** Assume $\mu\_0, \mu\_1$ are compactly supported, $\mu\_t$ is absolutely continuous. Use the nonbranching property: for any subplan $0 \le \widetilde{\Pi} \le \Pi$, $\Pi' := \widetilde{\Pi}/\widetilde{\Pi}[\Gamma]$ is the **unique** dynamical optimal transference plan between its endpoint measures. Condition the dynamical plan $\Pi$ on the event $\gamma\_t \in B\_\delta(y)$, apply the CD inequality to the resulting conditional plan $\Pi'$, and use the Lebesgue density theorem (valid since $\nu$ is locally doubling by the Bishop–Gromov inequality) to pass to the limit $\delta \to 0$. This yields the **pointwise** inequality: $\Pi(d\gamma)$-almost surely,

$$\frac{1}{\rho_t(\gamma_t)^{1/N}} \ge (1-t)\!\left(\frac{\beta_{1-t}(\gamma_0, \gamma_1)}{\rho_0(\gamma_0)}\right)^{1/N} + t\!\left(\frac{\beta_t(\gamma_0, \gamma_1)}{\rho_1(\gamma_1)}\right)^{1/N}.$$

**Step 2:** Extend to the case $\mu\_0, \mu\_1$ compactly supported and $\mu\_t$ absolutely continuous. Use the nonbranching property and Theorem 7.30(iii): the restricted plan $\Pi^{0,1-\varepsilon}$ is the only dynamical optimal plan between $\mu\_0$ and $\mu\_{1-\varepsilon}$. Apply Step 1 on intervals $[0, 1-\varepsilon]$ and $[t, 1]$ independently, then combine the resulting inequalities and pass to the limit $\varepsilon \to 0$.

**Step 3:** Drop the assumption of compact support. This is possible since (from Step 2) all the geometric consequences of $\mathrm{CD}(K, N)$ are now available (Bishop–Gromov, regularity of the interpolant, Poincaré, etc.), even without compact support.

**Step 4:** Handle the case when $\mu\_t$ is not absolutely continuous. Decompose $\Pi = (1-m)\, \Pi^{(a)} + m\, \Pi^{(s)}$ according to whether $\gamma\_t$ lands in the absolutely continuous or singular part of $\mu\_t$. Apply the inequality with the rescaled nonlinearity $U\_m(r) = U((1-m)r)/Z\_\ell$ to the absolutely continuous part. For the singular part, use Theorem 30.19(iii) to show that $\mu\_0^{(s)}$ is purely singular, and the definition of $U\_\nu$ to handle the $U'(\infty)\, \mu\_s[\mathcal{X}]$ terms.

For $N = \infty$, the argument of Step 1 changes since $\nu$ is not a priori locally doubling. The key observation is that $e^{-f \circ F\_t}$ is upper semicontinuous (where $F\_t$ is the time-$t$ evaluation map, which is continuous on $\operatorname{Spt} \Pi$). The Lebesgue density argument is replaced by a direct pointwise bound using this upper semicontinuity. Densities are then approximated by nondecreasing sequences of upper semicontinuous functions.

</div>

### Locality

Locality is one of the most fundamental properties one may expect from any notion of curvature. In the setting of weak $\mathrm{CD}(K, N)$ spaces, the locality problem may be loosely formulated as: *If $(\mathcal{X}, d, \nu)$ is weakly $\mathrm{CD}(K, N)$ in the neighborhood of any of its points, then $(\mathcal{X}, d, \nu)$ should be a weakly $\mathrm{CD}(K, N)$ space.*

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conjecture 30.34 — Local-to-Global $\mathrm{CD}(K,N)$ Property Along a Path)</span></p>

Let $\theta \in (0, 1)$ and $\alpha \in [0, \pi]$. Let $f : [0, 1] \to \mathbb{R}\_+$ be a measurable function such that for all $\lambda \in [0, 1]$, $t, t' \in [0, 1]$, the inequality

$$f\!\bigl((1-\lambda)t + \lambda t'\bigr) \ge (1-\lambda)\!\left(\frac{\sin\bigl((1-\lambda)\, \alpha \lvert t - t' \rvert\bigr)}{(1-\lambda) \sin(\alpha \lvert t - t' \rvert)}\right)^\theta f(t) + \lambda\!\left(\frac{\sin\bigl(\lambda\, \alpha \lvert t - t' \rvert\bigr)}{\lambda \sin(\alpha \lvert t - t' \rvert)}\right)^\theta f(t')$$

holds true as soon as $\lvert t - t' \rvert$ is small enough. Then (30.56) automatically holds true for all $t, t' \in [0, 1]$.

If $K = 0$ (resp. $N = \infty$), inequality (30.43) (resp. (30.51)) satisfies a local-to-global principle. In the other cases, this is open.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(30.35 — Local $\mathrm{CD}(K,N)$ Space)</span></p>

Let $K \in \mathbb{R}$ and $N \in [1, \infty]$. A locally compact Polish geodesic space $(\mathcal{X}, d)$ equipped with a locally finite measure $\nu$ is said to be a **local weak $\mathrm{CD}(K, N)$ space** if for any $x\_0 \in \mathcal{X}$ there is $r > 0$ such that whenever $\mu\_0, \mu\_1$ are two probability measures supported in $B\_r(x\_0) \cap \operatorname{Spt} \nu$, there is a displacement interpolation $(\mu\_t)\_{0 \le t \le 1}$ joining $\mu\_0$ to $\mu\_1$, and an associated optimal coupling $\pi$, such that for all $t \in [0, 1]$ and for all $U \in \mathcal{DC}\_N$,

$$U_\nu(\mu_t) \le (1-t)\, U_{\pi,\nu}^{\beta_{1-t}^{(K,N)}}(\mu_0) + t\, U_{\check{\pi},\nu}^{\beta_t^{(K,N)}}(\mu_1).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.37 — From Local to Global $\mathrm{CD}(K,N)$)</span></p>

Let $K \in \mathbb{R}$, $N \in [1, \infty)$, and let $(\mathcal{X}, d, \nu)$ be a nonbranching local weak $\mathrm{CD}(K, N)$ space with $\operatorname{Spt} \nu = \mathcal{X}$. If $K = 0$, then $\mathcal{X}$ is also a weak $\mathrm{CD}(K, N)$ space. The same is true for all values of $K$ if Conjecture 30.34 has an affirmative answer.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(30.38–30.39)</span></p>

- If the assumption $\operatorname{Spt} \nu = \mathcal{X}$ is dropped then the result becomes trivially false. (Counterexample: $\mathcal{X} = \mathbb{R}^3$ with the 2-dimensional Lebesgue measure concentrated on parallel planes at integer altitudes is a local weak $\mathrm{CD}(0, 2)$ space but not a weak $\mathrm{CD}(0, 2)$ space.)
- It is not known if the nonbranching condition can be removed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Finite-Dimensional and Infinite-Dimensional Points)</span></p>

A point $x$ in a metric-measure space $(\mathcal{X}, d, \nu)$ is called **finite-dimensional** if there is a small ball $B\_r(x)$ in which the criterion for $\mathrm{CD}(K', N')$ is satisfied, where $K' \in \mathbb{R}$ and $N' < \infty$. A point is called **infinite-dimensional** if it is not finite-dimensional. Example 29.17 (infinite product of circles) is "genuinely infinite-dimensional" — none of its points is finite-dimensional.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(30.42 — From Local to Global $\mathrm{CD}(K,\infty)$)</span></p>

Let $K \in \mathbb{R}$ and let $(\mathcal{X}, d, \nu)$ be a local weak $\mathrm{CD}(K, \infty)$ space with $\operatorname{Spt} \nu = \mathcal{X}$. Assume that $\mathcal{X}$ is nonbranching and that there is a totally convex measurable subset $\mathcal{Y}$ of $\mathcal{X}$ such that all points in $\mathcal{Y}$ are finite-dimensional and $\nu[\mathcal{X} \setminus \mathcal{Y}] = 0$. Then $(\mathcal{X}, d, \nu)$ is a weak $\mathrm{CD}(K, \infty)$ space.

</div>

### Appendix: Localization in Measure Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(30.44 — Cutoff Functions)</span></p>

Let $(\mathcal{X}, d)$ be a boundedly compact metric space, and let $\star$ be an arbitrary base point. For any $R > 0$, let $B\_R = B[\star, R]$. A **$\star$-cutoff** is a family of nonnegative continuous functions $(\chi\_R)\_{R > 0}$ such that $1\_{B\_R} \le \chi\_R \le 1\_{B\_{R+1}}$ for all $R$.

The existence of a $\star$-cutoff follows from Urysohn's lemma. If $\mu$ is any finite measure on $\mathcal{X}$, then $\chi\_R \mu$ converges to $\mu$ in total variation norm; moreover, for any $R > 0$, the truncation operator $T\_R : \mu \to \chi\_R \mu$ is a (nonstrict) contraction. As a particular case, if $\nu$ is any measure on $\mathcal{X}$ and $f \in L^1(\mathcal{X}, \nu)$, then $\chi\_R f$ converges to $f$ in $L^1(\nu)$. A consequence is the density of $C\_c(\mathcal{X})$ in $L^1(\mathcal{X}, \nu)$, as soon as $\nu$ is locally finite.

</div>

### Bibliographical Notes on Chapter 30

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Notes)</span></p>

Most of the material in this chapter comes from papers by **Lott and Villani** and by **Sturm**. Prior to these, there had been an important series of papers by **Cheeger and Colding**, about the structure of measured Gromov–Hausdorff limits of sequences of Riemannian manifolds satisfying a uniform $\mathrm{CD}(K, N)$ bound. Some of these results can be re-interpreted in the present framework, but many others remain open (generalized splitting theorem, mutual absolute continuity of admissible reference measures, continuity of the volume in absence of collapsing).

**Theorem 30.2** is taken from Lott–Villani, as well as Corollary 30.9, Theorems 30.22 and 30.23, and the first part of Theorem 30.28. **Theorem 30.7**, Corollary 30.10, Theorems 30.11 and 30.17 are due to **Sturm**. Part (i) of Theorem 30.19 was proven by Lott–Villani in the case $K = 0$; part (ii) follows a scheme of proof communicated by Sturm. **Theorem 30.20** is well-known in a Euclidean context and used in several recent works about optimal transport.

The Poincaré inequalities appearing in Theorems 30.25 and 30.26 (in the case $K = 0$) are due to Lott–Villani. The concept of upper gradient was put forward by **Heinonen and Koskela**; it played a key role in **Cheeger's** construction of a differentiable structure on metric spaces satisfying a doubling condition and a local Poincaré inequality. Independently, there were several simultaneous treatments of local Poincaré inequalities under weak $\mathrm{CD}(K, N)$ conditions, by **Sturm** on the one hand, and **von Renesse** on the other. The proofs have common features with the proof by Cheeger and Colding, but the argument by Cheeger and Colding uses a "segment inequality" which has not been adapted to metric-measure spaces; Lott–Villani instead used the concept of "democratic condition" from Theorem 19.13.

**Theorem 30.32** was proved by the author specifically for these notes, but a very close statement was also obtained shortly after and independently by **Sturm**, at least for absolutely continuous measures. The treatment of singular measures (Step 4 of the proof) grew out of a joint work with **Figalli**. An alternative "Eulerian" approach to displacement convexity for singular measures was implemented by **Daneri and Savaré**.

In Alexandrov spaces, the locality of "curvature is bounded below by $\kappa$" is called **Toponogov's theorem**; in full generality it is due to **Perelman**.

The conditional locality of $\mathrm{CD}(K, \infty)$ in nonbranching spaces (Theorem 30.42) was first proven by **Sturm**, with a different argument. Theorem 30.37 is new as far as the author knows.

**On the quotient problem:** Lott and Villani proved that the quotient of a $\mathrm{CD}(K, N)$ space by a compact Lie group of isometries is itself $\mathrm{CD}(K, N)$, under assumptions (a) $\mathcal{X}$ and $G$ are compact; (b) $K = 0$ or $N = \infty$; (c) any two absolutely continuous probability measures are joined by a unique displacement interpolation which is absolutely continuous for all times. Theorem 30.32 guarantees that there is no difference if $\mathcal{X}$ is nonbranching.

**On the lifting problem:** One might think that locality (Theorems 30.37 or 30.42) implies that the universal covering inherits $\mathrm{CD}(K, N)$, but even when locality has been established, the existence of the universal covering is not obvious. It requires $\mathcal{X}$ to be locally contractible (which is known for Alexandrov spaces, but not for weak $\mathrm{CD}(K, N)$ spaces).

The thesis developed in these notes is that **optimal transport has suddenly become an important actor** in the theory of analysis on metric spaces.

</div>

## Conclusions and Open Problems

These notes have presented a consistent picture of the theory of optimal transport, with a *dynamical*, *probabilistic* and *geometric* point of view, insisting on the notions of displacement interpolation, probabilistic representation, and curvature effects.

### Open Problems in Qualitative Optimal Transport (Part I)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Open Issues in Part I)</span></p>

The qualitative description of optimal transport (Part I) is now more or less under control. Even the smoothness of the transport map in curved geometries starts to be better understood, thanks to recent works of Loeper, Ma, Trudinger and Wang (Chapter 12). Among issues which seem to be of interest:

- Find relevant examples of cost functions with nonnegative, or positive $c$-curvature (Definition 12.27), and theorems guaranteeing that the optimal transport does not approach singularities of the cost function — so that the smoothness of the transport map can be established.
- Get a precise description of the singularities of the optimal transport map when the latter is not smooth.
- Further analyze the displacement interpolation on singular spaces, maybe via nonsmooth generalizations of Mather's estimates (as in Open Problem 8.21).

</div>

### Open Problems in Riemannian Geometry (Part II)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Open Issues in Part II)</span></p>

For the applications of optimal transport to Riemannian geometry, a consistent picture is emerging (Part II). The main regularity problems seem to be under control, but there remain several challenging "structural" problems:

- How can one best understand the relation between plain displacement convexity and distorted displacement convexity, as described in Chapter 17? Is there an Eulerian counterpart of the latter concept? (Open Problems 17.38 and 17.39.)
- Optimal transport seems to work well to establish sharp geometric inequalities when the "natural dimension" of the inequality coincides with the dimension bound; but so far it has failed to establish sharp logarithmic Sobolev or Talagrand inequalities (infinite-dimensional) under a $\mathrm{CD}(K, N)$ condition for $N < \infty$ (Open Problems 21.6 and 22.44). The sharp $L^2$-Sobolev inequality (21.9) has also escaped investigations based on optimal transport (Open Problems 21.11). Can one find a more precise strategy to attack such problems? Can one mimick the changes of variables in the $\Gamma\_2$ formalism, which are at the basis of the derivation of such sharp inequalities?
- Are there interesting **examples of displacement convex functionals** apart from those of the form $\int\_M U(\rho)\, d\nu + \int\_{M^k} V\, d\mu^{\otimes k}$? It is frustrating that so few examples are known, in contrast with the enormous amount of plainly convex functionals that one can construct. (Open Problem 15.11.)
- Is there a transport-based proof of the famous **Lévy–Gromov isoperimetric inequalities** (Open Problem 21.16), that would not involve so much "hard analysis"? Such a proof could hopefully be adapted to nonsmooth spaces such as weak $\mathrm{CD}(K, N)$ spaces.
- **Caffarelli's log concave perturbation theorem** (Chapter 2) is another riddle. It can be restated as: *If the Euclidean space $(\mathbb{R}^n, d\_2)$ is equipped with a probability measure $\nu$ that makes it a $\mathrm{CD}(K, \infty)$ space, then $\nu$ can be realized as a 1-Lipschitz push-forward of the reference Gaussian measure with curvature $K$.* There is a strong analogy with the Lévy–Gromov isoperimetric inequality; is there a common framework?
- The geometric theory has been developed almost exclusively for the **quadratic** cost function ($p = 2$); working with other exponents (or radically different Lagrangian cost functions) might lead to new geometric territories, as illustrated by Ohta's work in Finsler geometry. (Open Problem 15.12.)

</div>

### Open Problems in Synthetic Ricci Curvature (Part III)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Three Missing Pieces of the Puzzle)</span></p>

The emerging theory of weak Ricci curvature lower bounds in metric-measure spaces (Part III) has grown very fast and is starting to be rather well-developed; however, some challenging issues remain. Three missing pieces of the puzzle:

- A **globalization theorem** that would play the role of the Toponogov–Perelman theorem for Alexandrov spaces with a lower bound on the curvature. This result should state that a weak local $\mathrm{CD}(K, N)$ space is automatically a weak $\mathrm{CD}(K, N)$ space. Theorem 30.37 shows this is true at least if $K = 0$, $N < \infty$ and $\mathcal{X}$ is nonbranching; if Conjecture 30.34 turns out to be true, the same result will be available for all values of $K$.
- The **compatibility with the theory of Alexandrov spaces** (with lower curvature bounds). Since Alexandrov bounds are weak sectional curvature bounds, they should in principle be able to control weak Ricci curvature bounds. The natural question: Let $(\mathcal{X}, d)$ be a finite-dimensional Alexandrov space with dimension $n$ and curvature bounded below by $\kappa$, and let $\mathcal{H}^n$ be the $n$-dimensional Hausdorff measure on $\mathcal{X}$; is $(\mathcal{X}, d, \mathcal{H}^n)$ a weak $\mathrm{CD}((n-1)\kappa, n)$ space?
- A thorough discussion of the **branching** problem: Find examples of weak $\mathrm{CD}(K, N)$ spaces that are branching; that are singular but nonbranching; identify simple regularity conditions that prevent branching; etc. It is also of interest to enquire whether the nonbranching assumption can be dispensed with in Theorems 30.26 and 30.37 (Remarks 30.27 and 30.39).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Further Structural Questions)</span></p>

More generally, one would like to have more information about the structure of weak $\mathrm{CD}(K, N)$ spaces, at least when $N$ is finite. In Alexandrov spaces, some rather strong structure theorems have been established by Perelman and others; it is natural to ask whether similar results hold true for weak $\mathrm{CD}(K, N)$ spaces.

Another relevant problem is to check the compatibility of the $\mathrm{CD}(K, N)$ condition with the operations of **quotient** by Lie group actions, and **lifting** to the universal covering. As explained in the bibliographical notes of Chapter 30, only partial results are known.

Besides these issues, it seems important to find further **examples** of weak $\mathrm{CD}(K, N)$ spaces, apart from those presented in Chapter 29 (mostly constructed as limits or quotients of manifolds).

</div>

### Normed Spaces as Weak $\mathrm{CD}(0, n)$ Spaces

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Normed Spaces Are Weak $\mathrm{CD}(0,n)$ Spaces)</span></p>

Let $\lVert \cdot \rVert$ be a norm on $\mathbb{R}^n$ (considered as a distance on $\mathbb{R}^n \times \mathbb{R}^n$), and let $\lambda\_n$ be the $n$-dimensional Lebesgue measure. Then the metric-measure space $(\mathbb{R}^n, \lVert \cdot \rVert, \lambda\_n)$ is a weak $\mathrm{CD}(0, n)$ space in the sense of Definition 29.8.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Significance of the Normed Space Theorem)</span></p>

This result is motivating and a bit shocking:

**(a)** As pointed out by **Lott**, if $\lVert \cdot \rVert$ is not Euclidean, then the metric-measure space $(\mathbb{R}^n, \lVert \cdot \rVert, \lambda\_n)$ cannot be realized as a limit of smooth Riemannian manifolds with a uniform $\mathrm{CD}(0, N)$ bound, because it fails to satisfy the *splitting principle*. This means that smooth $\mathrm{CD}(K, N)$ manifolds are **not** dense in the spaces $\mathcal{CDD}(K, N, D, m, M)$ introduced in Theorem 29.32.

**(b)** If $\lVert \cdot \rVert$ is not the Euclidean norm, the resulting metric space is in general not an Alexandrov space, and it can be extremely branching. For instance, if $\lVert \cdot \rVert$ is the $\ell\_\infty$ norm, then any two distinct points are joined by an uncountable infinity of geodesics. Since $(\mathbb{R}^n, \lVert \cdot \rVert\_{\ell\_\infty}, \lambda\_n)$ is the (pointed) limit of the nonbranching spaces $(\mathbb{R}^n, \lVert \cdot \rVert\_{\ell\_p}, \lambda\_n)$ as $p \to \infty$, this shows that weak $\mathrm{CD}(K, N)$ bounds do not prevent branching in measured Gromov–Hausdorff limits.

**(c)** On the other hand, the study of optimal Sobolev inequalities in $\mathbb{R}^n$ (by the author together with Nazaret and Cordero-Erausquin) shows that optimal Sobolev inequalities basically do not depend on the choice of the norm. In a Riemannian context, Sobolev inequalities strongly depend on Ricci curvature bounds; so it is not absurd to decide that $\mathbb{R}^n$ is a weak $\mathrm{CD}(0, n)$ space independently of the norm. **Shin-ichi Ohta** has developed this point of view by studying curvature-dimension conditions in certain classes of Finsler spaces.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequence: Strong $\mathrm{CD}(K,N)$ Is Not Stable)</span></p>

In the approximation argument above, the spaces $(\mathbb{R}^n, N\_k, \lambda\_n)$ satisfy the property that the displacement interpolation between any two absolutely continuous, compactly supported probability measures is unique. The limit space $(\mathbb{R}^n, N, \lambda\_n)$ does not necessarily satisfy this (e.g. $N = \lVert \cdot \rVert\_{\ell\_\infty}$ has an enormous number of displacement interpolations, most of which do not satisfy the displacement convexity inequalities). This shows that if in Definition 29.8 one requires the inequality (29.11) to hold true for *any* Wasserstein geodesic, rather than for *some* Wasserstein geodesic, then the resulting $\mathrm{CD}(K, N)$ property is **not stable** under measured Gromov–Hausdorff convergence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sketch of Proof of the Normed Space Theorem)</span></p>

First consider the case when $N = \lVert \cdot \rVert$ is a uniformly convex, smooth norm, in the sense that $\lambda\, I\_n \le \nabla^2 N^2 \le \Lambda\, I\_n$ for some positive constants $\lambda$ and $\Lambda$. Then the cost function $c(x, y) = N(x - y)^2$ is both strictly convex and $C^{1,1}$, i.e. uniformly semiconcave. By Theorem 10.28 (recall Example 10.35), if $\mu\_0$ and $\mu\_1$ are compactly supported and absolutely continuous, then there is a unique optimal transport, and it takes the form $T(x) = x - \nabla(N^2)^\ast(-\nabla\psi(x))$, where $\psi$ is a $c$-convex function.

Let $\theta(x) = \nabla(N^2)^\ast(-\nabla\psi(x))$. The Jacobian matrix $\nabla\theta$, although not symmetric, is pointwise diagonalizable, with eigenvalues bounded above by 1. It follows that $t \to \det(I\_n - t\nabla\theta)^{1/n}$ is a concave function of $t$, and one can reproduce the proof of displacement convexity for $U\_{\lambda\_n}$, as soon as $U \in \mathcal{DC}\_n$.

This shows that $(\mathbb{R}^n, N, \lambda\_n)$ satisfies the $\mathrm{CD}(0, n)$ displacement convexity inequalities when $N$ is smooth and uniformly convex. Now if $N$ is arbitrary, it can be approximated by a sequence $(N\_k)$ of smooth uniformly convex norms, in such a way that $(\mathbb{R}^n, N, \lambda\_n, 0)$ is the pointed measured Gromov–Hausdorff limit of $(\mathbb{R}^n, N\_k, \lambda\_n, 0)$ as $k \to \infty$. Then the general conclusion follows by stability of the weak $\mathrm{CD}(0, n)$ criterion (Theorem 29.24).

</div>

### Open Problems in Gradient Flows and PDEs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gradient Flows and Further Directions)</span></p>

The interpretation of dissipative equations as gradient flows with respect to optimal transport (Chapters 23–25) also leads to fascinating issues:

**(a) Heat flow on weak $\mathrm{CD}(K, N)$ spaces.** Can one define a reasonably well-behaved **heat flow** on weak $\mathrm{CD}(K, N)$ spaces by taking the gradient flow for Boltzmann's $H$ functional? The theory of gradient flows in abstract metric spaces has been pushed very far by **Ambrosio, Gigli and Savaré**. **Ohta** and independently **Savaré** recently made progress in this direction by constructing gradient flows in the Wasserstein space over a finite-dimensional Alexandrov space (or over more general spaces satisfying weak regularity assumptions). Savaré uses an elegant argument, based on properties of Wasserstein distances and entropy, to prove the linearity of this semigroup, and other properties as well (positivity, contraction in $W\_p$ for $1 \le p \le 2$, contraction in $L^p$, regularizing effect). As noted by **Sturm**, the gradient flow of the $H$ functional in $P\_2((\mathbb{R}^N, \lVert \cdot \rVert))$ yields a *nonlinear* evolution. The fact that this equation has the same fundamental solution as the Euclidean one is an argument to believe that this is a natural notion of heat equation on non-Euclidean $\mathbb{R}^N$.

**(b) Extension to Hamiltonian and dissipative Hamiltonian equations.** As explained in the bibliographical notes of Chapter 23, there has been some recent work by **Ambrosio, Gangbo** and others, but the situation is still far from clear.

**(c) Semi-geostrophic system.** The semi-geostrophic system can formally be written as a Hamiltonian flow where the Hamiltonian function is the square Wasserstein distance with respect to some uniform reference measure. The rigorous qualitative understanding of this system is one of the most exciting open problems in theoretical fluid mechanics.

**(d) Geometry of $P\_2(\mathcal{X})$.** There is a neat statement according to which $P\_2(\mathcal{X})$ is nonnegatively curved (in the sense of Alexandrov) if and only if $\mathcal{X}$ itself is nonnegatively curved. But there is no similar statement for nonzero lower bounds on the curvature. It is not clear what exactly is "the right" structure on, say, $P\_2(\mathbb{R}^n)$. Another relevant open problem is whether there is a natural "volume" measure on $P\_2(M)$.

**(e) Generalized geodesics.** Ambrosio, Gigli and Savaré define "generalized geodesics" in $P\_2(\mathbb{R}^n)$ by considering the law of $(1-t)X\_0 + tX\_1$, where $(X\_0, Z)$ and $(X\_1, Z)$ are optimal couplings. These have intriguing properties (they satisfy characteristic displacement interpolation inequalities and provide curves of "nonpositive curvature" for error estimates). It is natural to further investigate these objects, which are reminiscent of $c$-segments (Chapter 12).

**(f) Numerical analysis.** The numerical analysis of optimal transport has a long and complex history (old simplex algorithm, Bertsekas's auction algorithm, schemes based on Monge–Ampère equations, continuous schemes based on PDEs). This subject deserves a systematic study on its own.

</div>
