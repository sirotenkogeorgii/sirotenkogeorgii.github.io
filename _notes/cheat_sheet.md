---
layout: default
title: cheat sheet
date: 2026-05-13
---

# Math Cheat Sheet

> Personal lookup file. The point is not rigor but **the core idea**: what the object is, what it is for, and the one or two facts that make it show up everywhere. Entries are added as I bump into them.

**Table of Contents**
- TOC
{:toc}

---

<!-- Weak-star Convergence
Uniform Convergence
Hamel Basis
Schauder Basis
L^p_{\text{per}}
Convoultion
Deconvolution
Fourier Analysis
Fourier Multipliers
Fourier Coefficients
Fourier Series
Fourier Decomposition
Fourier Transform
Discrete Fourier Transform
Bounded Linear Operator
Convolution Theorem
Fubini's Theorem
Diagonal Operator -->

Fourier basis
Lusin’s theorem
Radon Measure
Knott-Smith optimality criterion
Covariance Operator
Manifold
Riemannian Manifold
Brenier’s Theorem
Fenchel Inequality
Rademacher’s Theorem
Hausdorff dimension
Semigroup
Seminorm
Basel Problem
Rellich–Kondrachov Theorem
Kolmogorov–Riesz Theorem
Riesz's Lemma
"Riesz's lemma gives a clean characterization: a normed space is finite-dimensional iff its closed unit ball is compact. So "closed and bounded $\implies$ compact" is exactly the finite-dimensional case.
Kakutani's Reflexivity Theorem
Finite rank approximation of a compact operator
Picard-condition
\ell^2 space and L^2 space difference
Inverse DFT
Discrete Convolution Theorem
Spectral Decomposition
Double Convexification Trick


> The existence theory of Monge problem was not fully understood until 1995


<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Questions</span><span class="math-callout__name"></span></p>

* Question: Example of linear functions that are not continuous in infinite dimensional spaces
* Question: Is being complete always with respect to some norm / metric? Could be complete space not normed?
* Question: Is metric space more general that normed space?
* Question: When weak convergence implies strong convergence?
* Question: Why does topological boundary has the same notation as differential
* Question: What is the difference between support and domain?
* Question: Does a matric induce topology? Any topology?
* Question: Why are duals important?
* Question: Why is topology formulated in terms of open sets?
* Question: Convolution is a Fourier Transform? No, convolution is not a Fourier Transform, but they are profoundly linked through the convolution theorem, which states that convolution in the time domain is equivalent to point-wise multiplication in the frequency domain.
* Question: In infinite dimensions, compactness is much more restrictive — you typically need extra structure (equicontinuity for Arzelà–Ascoli in C(K), tightness in measure-theoretic settings, decay conditions for subsets of ℓ², etc.).

</div>

## Foundational spaces

### Vector space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(vector space)</span></p>

A **vector space** over a field $\mathbb{K}$ (always $\mathbb{R}$ or $\mathbb{C}$ for us) is a set $V$ equipped with addition $V \times V \to V$ and scalar multiplication $\mathbb{K} \times V \to V$ such that addition is a commutative group operation and scalar multiplication is bilinear and associative (with $1 \cdot v = v$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(what a bare vector space gives you)</span></p>

Linearity, subspaces, spans, linear independence, bases (via Zorn), quotient spaces, linear maps, kernels and images, the rank–nullity theorem — but **no notion of size, distance, limit, or continuity** until you add a topology / norm / metric / inner product. The whole point of normed, Banach, Hilbert, and topological vector spaces is to attach exactly enough analytic structure to do calculus.

* **Topological vector space (TVS)** — add a topology making $+$ and scalar multiplication continuous. Locally convex TVS is the right setting for distributions and weak topologies.
* **Normed space** — add a norm $\lVert \cdot \rVert$; then a metric $d(x,y) = \lVert x - y \rVert$ and a topology.
* **Banach** — normed and complete.
* **Hilbert** — Banach where the norm comes from an inner product.

</div>

### Hamel basis

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hamel basis)</span></p>

A **Hamel basis** of a vector space $V$ is a subset $B \subseteq V$ such that every $v \in V$ has a unique representation as a **finite** linear combination

$$
v \;=\; \sum_{i=1}^N c_i b_i, \qquad b_i \in B, \;\; c_i \in \mathbb{K},
$$

i.e. $B$ is linearly independent and spans $V$ in the purely algebraic sense.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(facts — and why nobody uses them in infinite dimensions)</span></p>

* Every vector space has a Hamel basis (Zorn's lemma) and any two have the same cardinality (the **algebraic dimension**).
* For an infinite-dimensional Banach space, a Hamel basis is **uncountable** — concretely, $\ge \lvert\mathbb{R}\rvert$ (Baire category: a countable Hamel basis would write $X$ as a countable union of finite-dimensional, hence nowhere-dense, subspaces).
* So Hamel bases exist but are useless for analysis: you can never write an interesting element of $L^p$ as a *finite* combination of basis vectors. They appear in pathological constructions (discontinuous linear functionals, non-measurable additive functions, Vitali sets via $\mathbb{R}/\mathbb{Q}$).
* In functional analysis, "basis" defaults to **Schauder basis**, not Hamel.

</div>

### Schauder basis

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Schauder basis)</span></p>

A sequence $(e_n)\_{n \in \mathbb{N}}$ in a Banach space $X$ is a **Schauder basis** if every $x \in X$ has a unique representation as a **norm-convergent** series

$$
x \;=\; \sum_{n=1}^\infty c_n(x) \, e_n,
$$

with scalars $c_n(x) \in \mathbb{K}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why this is the right notion)</span></p>

* Each coordinate functional $c_n : X \to \mathbb{K}$ is automatically **continuous** (by Banach–Steinhaus / closed-graph) — the "basis projections" $P_N x = \sum_{n \le N} c_n(x) e_n$ are uniformly bounded.
* A Banach space with a Schauder basis is **separable** (rational finite combinations are dense). The converse is false: Per Enflo (1973) built a separable Banach space with **no** Schauder basis.
* Standard examples: $(e_n)$ in $\ell^p$ ($1 \le p < \infty$) and $c_0$; trigonometric system $\lbrace e^{ikx}\rbrace$ in $L^p(\mathbb{T})$ for $1 < p < \infty$ (fails at $p = 1, \infty$); Haar / wavelet bases in $L^p$ for $1 \le p < \infty$.
* In a **Hilbert** space, an orthonormal basis is automatically a Schauder basis whose coefficients are $c_n(x) = \langle x, e_n\rangle$. The series converges in norm and Parseval holds.

</div>

### Banach space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Banach space)</span></p>

A **Banach space** is a normed vector space $(X, \lVert \cdot \rVert)$ that is **complete** with respect to the metric $d(x,y) = \lVert x - y \rVert$ — every Cauchy sequence converges in $X$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why completeness)</span></p>

Completeness is what makes the limit-passage arguments of analysis legal:

* **Existence by Cauchy criterion** — Picard iteration, fixed-point arguments, series $\sum x_n$ with $\sum \lVert x_n \rVert < \infty$ converging in $X$.
* The **trinity**: open mapping theorem, closed graph theorem, uniform boundedness principle — all rely on Baire category, which needs completeness.
* The **Hahn–Banach** extension theorem does **not** need completeness, but the dual $X^\ast$ is automatically a Banach space anyway (it inherits completeness from the scalar field).

Examples: $L^p$, $C(K)$ ($K$ compact), $W^{k,p}$, $\ell^p$, $C^{k,\alpha}$. Non-examples: $C^\infty$ with its natural topology (Fréchet, not Banach), $C_c^\infty$ (LF-space), polynomial spaces (incomplete in any reasonable norm).

</div>

### Hilbert space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hilbert space)</span></p>

A **Hilbert space** is a vector space $H$ over $\mathbb{R}$ or $\mathbb{C}$ equipped with an **inner product** $\langle \cdot, \cdot \rangle$ such that the induced norm $\lVert x \rVert = \sqrt{\langle x, x \rangle}$ makes $H$ a Banach space (complete).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(what the inner product buys)</span></p>

Hilbert is "$\mathbb{R}^n$-like" infinite dimensional analysis:

* **Geometry**: orthogonality, orthogonal projection onto closed convex sets, orthogonal complement $V \oplus V^\perp = H$.
* **Bases**: every separable Hilbert space has a countable orthonormal basis; Parseval expresses every $x$ as $\sum \langle x, e_n \rangle e_n$.
* **Self-duality**: $H^\ast \cong H$ via Riesz representation — every continuous linear functional is "inner-product against a vector."
* **Reflexive and uniformly convex** automatically, so bounded sequences have weakly convergent subsequences (Banach–Alaoglu + reflexivity).

Canonical examples: $L^2(\Omega)$, $\ell^2$, Sobolev $H^k = W^{k,2}$, reproducing kernel Hilbert spaces.

</div>

---

## Function spaces

### $L^p$ space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^p$ space)</span></p>

For a measure space $(\Omega, \mathcal{A}, \mu)$ and $1 \le p < \infty$,

$$
L^p(\Omega) = \Big\lbrace f : \Omega \to \mathbb{R} \text{ measurable} \;\Big|\; \int_\Omega \lvert f \rvert^p \, d\mu < \infty \Big\rbrace, \qquad \|f\|_{L^p} = \Big(\int_\Omega \lvert f \rvert^p \, d\mu\Big)^{1/p},
$$

modulo equality $\mu$-a.e. For $p=\infty$, $\|f\|_{L^\infty} = \operatorname*{ess\,sup} \lvert f \rvert$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why $L^p$ everywhere)</span></p>

* Banach for all $1 \le p \le \infty$; **Hilbert** when $p=2$ (inner product $\int fg$).
* **Reflexive** and **separable** for $1 < p < \infty$ (on $\sigma$-finite measures). $L^1$ is not reflexive; $L^\infty$ is not separable.
* **Dual:** $(L^p)^\ast \cong L^q$ with $\tfrac{1}{p}+\tfrac{1}{q}=1$ for $1 \le p < \infty$ — this is Riesz representation in $L^p$.
* Natural home for energies of the form $\int \lvert f\rvert^p$ and for almost every theorem in analysis/PDE.

</div>

### $L^p_{\text{per}}$ — periodic $L^p$ spaces

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^p_{\text{per}}$ / $L^p(\mathbb{T}^n)$)</span></p>

For $1 \le p \le \infty$ and a period $T > 0$, $L^p_{\text{per}}([0,T])$ is the space of (equivalence classes of) measurable $T$-periodic functions $f : \mathbb{R} \to \mathbb{C}$ with $f\!\restriction\_{[0,T]} \in L^p([0,T])$, normed by the $L^p([0,T])$ norm. Equivalently — and most cleanly — it is $L^p(\mathbb{T}^n)$ with $\mathbb{T}^n = \mathbb{R}^n / T\mathbb{Z}^n$ the $n$-torus.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why we package periodic functions this way)</span></p>

* Identifying the domain with the **torus** $\mathbb{T}^n$ makes it **compact without boundary**: no integration-by-parts boundary terms, no need to cut off near $\partial \Omega$, every smooth periodic function is in $L^p_{\text{per}}$ automatically.
* The natural setting for **Fourier series**: $\lbrace e^{2\pi i k\cdot x / T}\rbrace_{k \in \mathbb{Z}^n}$ is an orthonormal basis of $L^2\_{\text{per}}$ (Parseval), and Sobolev spaces $H^s\_{\text{per}}$ can be defined directly through decay of Fourier coefficients $\lvert k\rvert^s \widehat f(k) \in \ell^2$.
* Standard solution / test space for **periodic PDE** (Schrödinger, KdV, wave, Navier–Stokes on $\mathbb{T}^n$) — periodicity replaces boundary conditions, so one gets a self-contained problem on a compact manifold.

</div>

### $C_c^\infty$ space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($C_c^\infty$ — test functions)</span></p>

$C_c^\infty(\Omega)$ is the space of smooth functions $\varphi : \Omega \to \mathbb{R}$ whose **support** $\overline{\lbrace x : \varphi(x) \ne 0\rbrace}$ is a compact subset of $\Omega$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role)</span></p>

The "probes" of distribution theory. Two crucial facts:

* **Density**: $C_c^\infty(\Omega)$ is dense in $L^p(\Omega)$ for $1 \le p < \infty$ — prove things on $C_c^\infty$ and extend.
* **Free differentiation**: integrate by parts as many times as you want, which is exactly what defines weak derivatives, weak solutions, and Sobolev spaces.

</div>

### Test functions and the space $\mathcal{D}$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(test functions)</span></p>

A **test function** on an open $\Omega \subseteq \mathbb{R}^n$ is an element of $C_c^\infty(\Omega)$ — a smooth function with compact support inside $\Omega$. The space of test functions, denoted $\mathcal{D}(\Omega) = C_c^\infty(\Omega)$, is equipped with the **inductive-limit (LF) topology**: a sequence $\varphi_n \to \varphi$ in $\mathcal{D}(\Omega)$ iff

1. there exists a single compact $K \subseteq \Omega$ with $\mathrm{supp}(\varphi_n) \subseteq K$ for all $n$, and
2. $D^\alpha \varphi_n \to D^\alpha \varphi$ **uniformly** on $K$ for every multi-index $\alpha$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role — probes for distributions)</span></p>

Test functions are *deliberately* the most restrictive reasonable class: smooth, compactly supported, with a stringent convergence. The point is that the **dual space** $\mathcal{D}'(\Omega) = \mathcal{D}(\Omega)^\ast$ — the continuous linear functionals — is then enormous and contains every "generalized function" one wants:

* Every $L^1\_{\mathrm{loc}}$ function $f$ defines a distribution $\varphi \mapsto \int f \varphi$.
* Every Radon measure $\mu$: $\varphi \mapsto \int \varphi \, d\mu$, in particular the Dirac $\delta_a : \varphi \mapsto \varphi(a)$.
* Derivatives are defined by transposition: $\langle D^\alpha T, \varphi \rangle := (-1)^{\lvert \alpha\rvert} \langle T, D^\alpha \varphi \rangle$ — so every distribution is differentiable to all orders, which is the whole point.

The Schwartz space $\mathcal{S}(\mathbb{R}^n)$ (rapidly decreasing smooth functions) is a related, slightly larger test class; its dual $\mathcal{S}'$ contains the **tempered distributions** on which the Fourier transform is well-defined.

</div>

### Hölder space $C^{k,\alpha}$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hölder space)</span></p>

For $k \in \mathbb{N}\_0$ and $\alpha \in (0,1]$, $C^{k,\alpha}(\overline{\Omega})$ is the set of $C^k$ functions whose $k$-th derivatives are **$\alpha$-Hölder continuous**:

$$
[D^k f]_\alpha := \sup_{x \ne y} \frac{\lvert D^k f(x) - D^k f(y) \rvert}{\lvert x - y \rvert^\alpha} < \infty,
$$

with norm $\|f\|\_{C^{k,\alpha}} = \|f\|\_{C^k} + [D^k f]\_\alpha$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(usage)</span></p>

A quantitative "smoothness up to a fractional order" — interpolates between $C^k$ and $C^{k+1}$. The Banach norm structure is what makes elliptic regularity (Schauder estimates) clean: data in $C^{k,\alpha}$ gives a solution in $C^{k+2,\alpha}$, i.e. you **gain two derivatives in the same Hölder scale**.

</div>

### Sobolev spaces $W^{k,p}$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sobolev space)</span></p>

For $k \in \mathbb{N}_0$ and $1 \le p \le \infty$,

$$
W^{k,p}(\Omega) = \Big\lbrace f \in L^p(\Omega) \;\Big|\; D^\alpha f \in L^p(\Omega) \text{ for all } \lvert\alpha\rvert \le k\Big\rbrace,
$$

where $D^\alpha f$ is the **weak** derivative, with norm

$$
\|f\|_{W^{k,p}} = \Big(\sum_{\lvert\alpha\rvert \le k}\|D^\alpha f\|_{L^p}^p\Big)^{1/p}.
$$

Hilbert case $H^k := W^{k,2}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(the two facts you reuse)</span></p>

* **Sobolev embedding**: enough derivatives in $L^p$ buys higher integrability or continuity. On $\mathbb{R}^n$ with $p < n$, $W^{1,p} \hookrightarrow L^{p^\ast}$ with $p^\ast = \tfrac{np}{n-p}$. If $kp > n$, $W^{k,p} \hookrightarrow C^{0,\alpha}$.
* **Rellich–Kondrachov**: on bounded $\Omega$ the embeddings are **compact**, which is how you extract limits in nonlinear PDE.

</div>

---

## Measures and measurability

### Lebesgue measure (and why it matters)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue measure)</span></p>

The unique translation-invariant, $\sigma$-additive measure on the Lebesgue $\sigma$-algebra of $\mathbb{R}^n$ that assigns $[0,1]^n$ measure $1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why this one and not another)</span></p>

There are many other measures (counting, Dirac, Gaussian, Hausdorff, ...) but Lebesgue is *the* default for analysis because:

* It is rich enough that $L^1(\mathbb{R}^n)$ is **complete** — the Riemann integral space is not.
* Convergence theorems (monotone, dominated, Fatou) are clean.
* It is compatible with translations / Fourier / convolutions.

</div>

### Lebesgue outer measure

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lebesgue outer measure)</span></p>

For $E \subseteq \mathbb{R}^n$,

$$
\lambda^\ast(E) \;=\; \inf \Big\lbrace \sum_{i=1}^\infty \mathrm{vol}(Q_i) \;:\; E \subseteq \bigcup_{i=1}^\infty Q_i, \; Q_i \text{ open box}\Big\rbrace,
$$

i.e. the infimum of the total volume of any **countable** cover of $E$ by open boxes.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role — the construction step before Lebesgue)</span></p>

* Defined for **every** subset of $\mathbb{R}^n$, but only **countably subadditive** ($\lambda^\ast(\bigcup E_i) \le \sum \lambda^\ast(E_i)$), not countably additive on arbitrary sets — that is exactly why $\lambda^\ast$ alone is not a measure.
* The "abstract outer measure" recipe generalizes: any non-negative, monotone, countably subadditive set function $\mu^\ast$ on $2^X$ with $\mu^\ast(\emptyset) = 0$ is called an outer measure. Hausdorff outer measure, Carathéodory outer measure, and the outer measure built from a premeasure on an algebra are the same recipe in disguise.
* The fix to recover countable additivity is to restrict $\lambda^\ast$ to a smaller $\sigma$-algebra — the **Carathéodory measurable** sets — and that gives Lebesgue measure.

</div>

### Carathéodory's theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Carathéodory extension / measurability criterion)</span></p>

Let $\mu^\ast$ be an outer measure on a set $X$. A set $A \subseteq X$ is **Carathéodory-measurable** if it "splits" every test set additively:

$$
\mu^\ast(E) \;=\; \mu^\ast(E \cap A) + \mu^\ast(E \setminus A) \qquad \forall E \subseteq X.
$$

The collection $\mathcal{M}$ of measurable sets is a **$\sigma$-algebra**, and the restriction $\mu := \mu^\ast \restriction\_{\mathcal{M}}$ is a **complete countably additive measure**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why this is the right definition)</span></p>

* This is the **construction theorem** that turns an outer measure into an honest measure. Apply it to $\lambda^\ast$ on $\mathbb{R}^n$ → Lebesgue measure. Apply it to the Hausdorff $s$-outer measure → Hausdorff $\mathcal{H}^s$. Apply it to a premeasure on an algebra → Carathéodory–Hahn extension to a $\sigma$-algebra.
* The "splits every set" criterion looks ad hoc but is exactly the minimal axiom that forces countable additivity to hold on the restricted family.
* The resulting measure $\mu$ is automatically **complete**: every subset of a $\mu$-null set is measurable with measure $0$. This is why Lebesgue measure is complete while the Borel measure (Lebesgue restricted to $\mathcal{B}(\mathbb{R}^n)$) is not.

</div>

### Jordan measure

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jordan measure)</span></p>

A bounded set $E \subseteq \mathbb{R}^n$ is **Jordan measurable** if its inner and outer Jordan content (approximating $E$ from inside/outside by finite unions of axis-aligned boxes) agree. The common value is its Jordan measure.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why Lebesgue replaced it)</span></p>

Equivalent to Riemann integrability of the indicator. Limitation: only **finitely additive**, so $\mathbb{Q} \cap [0,1]$ has no Jordan measure although it is "obviously" small. Lebesgue extends Jordan by allowing **countable** covers, which closes the measurable sets under all the operations one actually wants.

</div>

### Borel sets and the Borel $\sigma$-algebra

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Borel $\sigma$-algebra)</span></p>

For a topological space $X$, the **Borel $\sigma$-algebra** $\mathcal{B}(X)$ is the smallest $\sigma$-algebra containing the open sets. Its elements are **Borel sets**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(importance)</span></p>

* Continuous functions are automatically Borel-measurable, so Borel measurability is the natural compatibility between topology and measure.
* Every reasonable measure (Lebesgue, Gaussian, Hausdorff, push-forwards under continuous maps) is at minimum Borel.

</div>

### Hausdorff measure

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hausdorff measure)</span></p>

For $s \ge 0$ and a metric space $X$, the **$s$-dimensional Hausdorff (outer) measure** of $E \subseteq X$ is

$$
\mathcal{H}^s(E) \;=\; \lim_{\delta \to 0^+} \inf \Big\lbrace \sum_i \mathrm{diam}(U_i)^s \;:\; E \subseteq \bigcup_i U_i, \;\; \mathrm{diam}(U_i) \le \delta \Big\rbrace.
$$

The infimum runs over all countable covers of $E$ by sets of diameter $\le \delta$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role and Hausdorff dimension)</span></p>

* Generalizes **length / area / volume** to any real exponent $s$, including non-integer.
* On $\mathbb{R}^n$, $\mathcal{H}^n$ equals a constant multiple of Lebesgue measure; $\mathcal{H}^{n-1}$ is the "surface area" of smooth $(n-1)$-dimensional subsets.
* For each $E$ there is a unique critical value
    
    $$
    \dim_H(E) \;:=\; \inf\lbrace s : \mathcal{H}^s(E) = 0\rbrace \;=\; \sup\lbrace s : \mathcal{H}^s(E) = +\infty\rbrace,
    $$
    
    the **Hausdorff dimension** of $E$ — exactly the $s$ at which $\mathcal{H}^s$ jumps from $\infty$ to $0$.
* Standard tool in geometric measure theory and fractal geometry; in PDE it measures the singular sets (e.g. dimension of the singular set in minimal-surface or Navier–Stokes regularity theory).

</div>

### $\mu$-measurability vs $\mu$-strong-measurability

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weak vs strong measurability)</span></p>

For $f : \Omega \to X$ valued in a Banach space $X$:

* **$\mu$-measurable** (weakly measurable): $x^* \circ f$ is measurable for every $x^* \in X^*$.
* **$\mu$-strongly measurable**: $f$ is the pointwise a.e. limit of **simple functions** (finite-valued, measurable).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pettis)</span></p>

**Pettis's theorem**: if $X$ is separable (or $f$ has separable range a.e.), the two notions coincide. Strong measurability is the right hypothesis for defining the Bochner integral.

</div>

### Bochner integral

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bochner integral)</span></p>

For $f : \Omega \to X$ strongly measurable into a Banach space, 

$$f \text{ is Bochner integrable } \quad\iff\quad \int_\Omega \|f(x)\|\_X \, d\mu(x) < \infty$$

The integral $\int_\Omega f \, d\mu \in X$ is then the $X$-norm limit of integrals of simple-function approximants.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why care)</span></p>

Almost every property of the scalar Lebesgue integral (dominated convergence, Fubini, FTC for absolutely continuous paths) carries over. This makes expressions like $\int_0^T u(t)\, dt$ for a function of time taking values in a function space (PDE, SDE, optimal control) actually well-defined.

</div>

---

## Convergence

### Strong convergence (general)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(strong convergence)</span></p>

In a normed space $X$: 

$$x_n \to x \quad \iff \quad \lVert x_n - x\rVert_X \to 0$$

I.e. convergence in the **norm topology**.

</div>

### Weak convergence (general)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weak convergence)</span></p>

In a Banach space $X$: 

$$x_n \rightharpoonup x \quad \iff \quad f(x_n) \to f(x) \quad \forall f \in X^\ast$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(facts)</span></p>

Intuition: pair against every continuous probe and check pointwise convergence of the pairings. Strictly weaker than strong unless $\dim X < \infty$.

* Weakly convergent sequences are **bounded** (Banach–Steinhaus).
* Norm is **lower semicontinuous** under weak limits: $\|x\| \le \liminf \|x_n\|$ (mass can be lost in the limit but not gained).
* In a reflexive space, bounded $\Rightarrow$ has a weakly convergent subsequence (Eberlein–Šmulian / Banach–Alaoglu).

</div>

### Weak-$\ast$ convergence

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weak-$\ast$ convergence)</span></p>

In the dual $X^\ast$ of a normed space $X$,

$$
x_n^\ast \;\xrightharpoonup{\ast}\; x^\ast \quad\iff\quad x_n^\ast(x) \;\to\; x^\ast(x) \qquad \forall x \in X.
$$

I.e. pointwise convergence of functionals as ordinary scalar-valued functions on $X$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(weak vs weak-$\ast$, and why this is the right notion in $X^\ast$)</span></p>

* In $X^\ast$, weak convergence tests against all of $X^{\ast\ast}$ while weak-$\ast$ tests only against the canonical image $X \hookrightarrow X^{\ast\ast}$. They coincide **iff $X$ is reflexive**.
* **Banach–Alaoglu**: the closed unit ball $B_{X^\ast}$ is weak-$\ast$ **compact**. If $X$ is separable, $B_{X^\ast}$ is weak-$\ast$ metrizable, so bounded sequences have weak-$\ast$ convergent **subsequences**.
* The standard mode of compactness for measures, $L^\infty$-functions, and dual data: a bounded sequence in $L^\infty$ has a weak-$\ast$ subsequential limit; a tight family of probability measures has a weak-$\ast$ limit (this is exactly the [[weak-topology-of-measures entry]] and Prokhorov).
* **Standard example**: $\delta_n \xrightharpoonup{\ast} 0$ in $C_0(\mathbb{R})^\ast$ — the Dirac at $n$ tests against $\varphi \in C_0$ as $\varphi(n) \to 0$. Total variation is $1$ for every $n$, yet the limit is $0$: weak-$\ast$ convergence allows mass to escape to infinity (not visible to $C_0$).

</div>

### Strong convergence in $L^p$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(strong $L^p$ convergence)</span></p>

$$f_n \to f \in L^p \iff \|f_n - f\|_{L^p} \to 0$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Implies a.e. convergence along a subsequence (and that subsequence is dominated when $p<\infty$ on finite-measure spaces).

</div>

### Weak convergence in $L^p$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weak $L^p$ convergence, $1 \le p < \infty$)</span></p>

$f_n \rightharpoonup f$ in $L^p$ iff for every $g \in L^q$ (with $\tfrac1p + \tfrac1q = 1$),

$$
\int f_n \, g \, d\mu \;\longrightarrow\; \int f \, g \, d\mu.
$$

For $p = \infty$ one typically uses **weak-$*$** convergence: test against $g \in L^1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(oscillation has weak limit zero)</span></p>

$f_n(x) = \sin(nx) \rightharpoonup 0$ in $L^2([0, 2\pi])$ by Riemann–Lebesgue, but $\lVert f_n\rVert_{L^2} = \sqrt{\pi}$ for all $n$ — so the convergence is **not** strong. The mass is preserved; it just oscillates faster and faster and averages out against any test function.

</div>

### Uniform convergence

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(uniform convergence)</span></p>

$f_n \to f$ **uniformly** on $X$ if

$$
\sup_{x \in X} \lvert f_n(x) - f(x)\rvert \;\xrightarrow{n \to \infty}\; 0.
$$

Equivalently, convergence in the sup-norm topology on $B(X)$ (bounded functions).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(what it preserves and what it does not)</span></p>

* **Preserves continuity** — a uniform limit of continuous functions is continuous. This is the cleanest way to build continuous functions: e.g. Weierstrass M-test for series of continuous functions.
* **Preserves integrals** on bounded sets: $\int f_n \to \int f$ on any set of finite measure.
* **Does not preserve differentiability** by itself: even with each $f_n \in C^\infty$, the limit can fail to be differentiable (Weierstrass's nowhere-differentiable example is a uniform limit of trig polynomials).
* The strongest of the standard convergence modes for functions: uniform $\Rightarrow$ locally uniform $\Rightarrow$ pointwise; uniform $\Rightarrow$ $L^p$ on sets of finite measure.

</div>

### Locally uniform convergence

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(locally uniform convergence)</span></p>

$f_n \to f$ locally uniformly on $X$ if every $x \in X$ has a neighborhood on which $f_n \to f$ **uniformly**. On a locally compact $X$, equivalent to uniform convergence on **compact subsets** (the compact-open topology).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Strong enough to preserve continuity, flexible enough to handle non-compact domains (e.g. $\mathbb{R}$, holomorphic functions on a domain).

</div>

---

## Topologies on function spaces

### Dense subset

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(dense subset)</span></p>

A subset $D \subseteq X$ of a topological space is **dense** if its closure equals $X$:

$$
\overline{D} = X \quad\Longleftrightarrow\quad \text{every non-empty open } U \subseteq X \text{ satisfies } U \cap D \ne \emptyset.
$$

In a metric space, equivalently: every $x \in X$ is the limit of a sequence from $D$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why density is the universal proof tactic)</span></p>

The standard analysis move: **prove a statement on a dense subset, extend by continuity / limit-passage to the whole space.** Examples that all reduce to this:

* $C_c^\infty(\Omega)$ dense in $L^p$ ($1 \le p < \infty$) — verify $L^p$-inequalities and weak derivatives on smooth probes, extend.
* Simple functions dense in $L^p$ — used to define and bound the Bochner integral.
* Polynomials dense in $C([a,b])$ (Stone–Weierstrass) — approximation theory.
* **Separable** $=$ has a countable dense subset; this is exactly what makes weak-$\ast$ topologies metrizable on bounded sets.

</div>

### Topological boundary

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(topological boundary)</span></p>

For $A \subseteq X$ in a topological space, the **topological boundary** is

$$
\partial A \;=\; \overline{A} \cap \overline{X \setminus A} \;=\; \overline{A} \setminus \mathrm{int}(A).
$$

Equivalently, $x \in \partial A$ iff every open neighborhood of $x$ meets both $A$ and its complement.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(facts and notation collision)</span></p>

* $\partial A$ is always **closed**; $\overline{A} = \mathrm{int}(A) \sqcup \partial A$ and $X = \mathrm{int}(A) \sqcup \partial A \sqcup \mathrm{int}(X \setminus A)$.
* $A$ is **closed** iff $\partial A \subseteq A$; **open** iff $\partial A \cap A = \emptyset$; **clopen** iff $\partial A = \emptyset$.
* In $\mathbb{R}^n$, $\partial$ of a smooth domain is its manifold boundary (the two notions coincide for open sets with smooth boundary); the divergence theorem, trace theorems, and boundary-value problems all live on $\partial \Omega$.
* The symbol $\partial$ collides with the notation for partial derivatives ($\partial_x f$) and for the boundary operator of a chain complex; in algebraic topology this is intentional — $\partial \partial = 0$ for chains is the discrete analogue of "the boundary of a region is itself closed."

</div>

### Coarsest topology

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(coarser / coarsest)</span></p>

Of two topologies $\tau_1, \tau_2$ on the same set $X$, $\tau_1$ is **coarser** than $\tau_2$ (equivalently, $\tau_2$ is **finer** than $\tau_1$) if $\tau_1 \subseteq \tau_2$ — fewer open sets in $\tau_1$. The **coarsest topology** satisfying a given property is the intersection of all topologies on $X$ enjoying that property.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(what "coarser" buys you)</span></p>

Fewer open sets means:

* **More compact** subsets (compactness asks every open cover to admit a finite subcover — fewer covers to defeat).
* **Fewer** continuous functions $X \to Y$ (preimages of open sets are scarcer).
* **More** continuous functions $Y \to X$ (the target has fewer open sets to test against).

Most constructions in analysis isolate the **coarsest** topology making some prescribed family of maps continuous — that is the initial topology, and it specializes to subspace, product, weak, and weak-$\ast$ topologies.

</div>

### Initial topology

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Initial Topology)</span></p>

Given a set $X$, a family of topological spaces $(Y_i, \tau_i)\_{i \in I}$, and a family of maps $f_i: X \to Y_i$, the **initial topology** on $X$ induced by the family $\lbrace f_i\rbrace\_{i \in I}$ is the coarsest (smallest) topology on $X$ that makes every $f_i$ continuous.

Concretely, it is generated by the subbase

$$\mathcal{S} = \lbrace f_i^{-1}(U) : i \in I,\ U \in \tau_i\rbrace.$$

That is, open sets in $X$ are arbitrary unions of finite intersections of preimages $f_i^{-1}(U)$. Any topology making all the $f_i$ continuous must contain $\mathcal{S}$, and the topology generated by $\mathcal{S}$ is the smallest such — hence the name "initial."

**Universal property.** This is the cleanest characterization, and it's what makes the construction useful. A map $g: Z \to X$ from any topological space $Z$ is continuous if and only if $f_i \circ g: Z \to Y_i$ is continuous for every $i \in I$.

In categorical terms, the initial topology is the unique topology on $X$ for which this universal property holds.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__namExample">(Intiial topologies)</span></p>

Here are four worked examples, going from the simplest to the most useful in your current functional-analysis context.

**1. Subspace topology on the circle.** Let $X = S^1 = \lbrace(x,y) \in \mathbb{R}^2 : x^2 + y^2 = 1\rbrace$ and let $\iota: S^1 \hookrightarrow \mathbb{R}^2$ be inclusion. The initial topology with respect to $\lbrace\iota\rbrace$ has as a subbase

$$\lbrace\iota^{-1}(U) : U \subseteq \mathbb{R}^2 \text{ open}\rbrace = \lbrace U \cap S^1 : U \subseteq \mathbb{R}^2 \text{ open}\rbrace,$$

which is already closed under finite intersections, so this is the topology itself. An open arc on $S^1$, say $\lbrace(\cos\theta, \sin\theta) : \theta \in (0, \pi/2)\rbrace$, is obtained as $B \cap S^1$ where $B$ is an open half-plane in $\mathbb{R}^2$.

**2. Product topology on $\mathbb{R}^{\mathbb{N}}$.** Consider $X = \mathbb{R}^{\mathbb{N}}$ (sequences of reals) with the projections $\pi_n: X \to \mathbb{R}$, $\pi_n(x) = x_n$. A subbasic open set is

$$\pi_n^{-1}(U) = \lbrace x \in \mathbb{R}^{\mathbb{N}} : x_n \in U\rbrace, \qquad U \subseteq \mathbb{R} \text{ open}.$$

Basic open sets are finite intersections, restricting only finitely many coordinates:

$$\lbrace x : x_{n_1} \in U_1, \ldots, x_{n_k} \in U_k\rbrace.$$

A sequence $(x^{(m)})\_{m \geq 1}$ in $X$ converges to $x$ in this topology iff $x_n^{(m)} \to x_n$ for every $n$ — i.e. coordinatewise convergence. Crucially, sets like $\prod_n (-1/n, 1/n)$ (the "box") are *not* open here, because they restrict infinitely many coordinates. This is exactly the reason the product topology is initial: it puts in the minimum needed to keep projections continuous, no more.

**3. Weak topology and the canonical example.** Take $E = \ell^2$ with its dual $E^\ast \cong \ell^2$ via $\varphi_y(x) = \sum_n x_n \overline{y_n}$. The weak topology $\sigma(\ell^2, \ell^2)$ is the initial topology generated by all $\varphi_y$, $y \in \ell^2$. A subbasic weak neighborhood of $0$ has the form

$$V_{y, \varepsilon} = \lbrace x \in \ell^2 : |\langle x, y \rangle| < \varepsilon\rbrace.$$

Now consider the standard basis $(e_n)$. For any fixed $y \in \ell^2$,

$$\langle e_n, y \rangle = \overline{y_n} \to 0 \quad \text{as } n \to \infty$$

because $\sum \|y_n\|^2 < \infty$. Hence $e_n \rightharpoonup 0$ weakly. But $\|e_n\| = 1$ for all $n$, so $(e_n)$ does *not* converge to $0$ in norm. This separation between weak and norm convergence is the whole reason the weak topology is interesting — and it is precisely what "initial topology" buys you: enough open sets to make every $\varphi_y$ continuous, but no more.

**4. Pointwise convergence as initial topology.** Let $X$ be any set and let $F = \mathbb{R}^X$ be the set of real-valued functions on $X$. For each $x \in X$, define the evaluation $\mathrm{ev}\_x: F \to \mathbb{R}$, $\mathrm{ev}\_x(f) = f(x)$. The initial topology on $F$ induced by $\lbrace\mathrm{ev}\_x\rbrace\_{x \in X}$ is the *topology of pointwise convergence*: a net $f_\alpha \to f$ in this topology iff $f_\alpha(x) \to f(x)$ for every $x$. Notice this is literally the product topology on $\mathbb{R}^X$ under the identification $F = \prod_{x \in X} \mathbb{R}$, with $\mathrm{ev}\_x = \pi_x$. Example 2 is the special case $X = \mathbb{N}$.

The pattern across all four is identical: pick the maps you want to be continuous, take preimages of open sets, generate a topology. The universal property then guarantees this is *the* right topology for studying maps *into* $X$ — continuity from any source $Z$ reduces to continuity of the compositions.

**Universal property.** A map $g : Z \to X$ into a set carrying an initial topology is continuous **iff** every composition $f_i \circ g$ is continuous. Continuity into $X$ reduces to checking it "coordinatewise" — which is exactly why initial topologies are practical.

</div>

### Norm topology

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(norm topology)</span></p>

On a normed vector space $(X, \lVert \cdot \rVert)$, the **norm topology** is the topology induced by the metric $d(x,y) = \lVert x - y \rVert$. A subbase of opens is the family of open balls

$$
B_r(x_0) \;=\; \lbrace x \in X : \lVert x - x_0 \rVert < r\rbrace, \qquad x_0 \in X, \; r > 0.
$$

Sequential convergence in this topology is **strong convergence**: $x_n \to x$ iff $\lVert x_n - x\rVert \to 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role and contrast)</span></p>

* The default, "strongest" topology on a normed space — every continuous linear functional is automatically continuous in the norm topology, but the converse is false (the weak topology is coarser).
* Makes $X$ a metric space, hence first-countable and Hausdorff. Sequences capture closure and continuity.
* In **infinite-dimensional** spaces the norm topology is too strong for compactness: closed bounded balls fail to be compact (this is essentially the Riesz lemma). The whole point of introducing the weak and weak-$\ast$ topologies is to recover compactness by **coarsening** the topology.
* On $C(K)$ with the sup-norm, the norm topology coincides with the uniform topology — they are different names for the same thing once you pick the sup-norm.

</div>

### Uniform topology

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniform topology)</span></p>

Induced by the sup-norm $\|f\|\_\infty = \sup_x \lvert f(x)\rvert$. Convergence is uniform convergence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Strongest of the three below; preserves continuity, integrals, derivatives (up to uniform convergence of derivatives), etc.

</div>

### Pointwise topology

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pointwise topology)</span></p>

The product topology on $\mathbb{R}^X$: 

$$f_n \to f \quad\iff\quad f_n(x) \to f(x) \quad \forall x$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Very weak — limits can be horribly discontinuous (e.g. $x^n$ on $[0,1]$). Useful for very general convergence statements (e.g. Helly's selection theorem, monotone limits).

</div>

### Weak topology

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weak topology on a Banach space)</span></p>

The coarsest topology on $X$ making every $x^* \in X^*$ continuous. A subbase of open sets is

$$
\lbrace x : \lvert x^*(x) - x^*(x_0) \rvert < \varepsilon \rbrace, \quad x^* \in X^*, \; \varepsilon > 0, \; x_0 \in X.
$$

The associated sequential convergence is weak convergence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The point: bounded sets are **much closer to compact** here than in the norm topology — that is exactly Banach–Alaoglu / Eberlein–Šmulian.

</div>

### Weak topology of measures

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weak / narrow topology on $\mathcal{P}(X)$)</span></p>

Let $X$ be a topological space (typically Polish) and $\mathcal{P}(X)$ the set of Borel probability measures on $X$. The **weak topology of measures** (also called the **narrow** topology) is the coarsest topology on $\mathcal{P}(X)$ making the map

$$
\mu \;\longmapsto\; \int_X f \, d\mu
$$

continuous for every $f \in C_b(X)$ (bounded continuous). Sequential convergence: $\mu_n \rightharpoonup \mu$ iff $\int f \, d\mu_n \to \int f \, d\mu$ for all $f \in C_b(X)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(probabilists' "weak" $=$ analysts' "weak-$\ast$")</span></p>

* Despite the name, from the Banach-space viewpoint this is a **weak-$\ast$** topology: $\mathcal{P}(X) \subseteq C_b(X)^\ast$ via $\mu \mapsto (f \mapsto \int f\, d\mu)$, and the weak topology of measures is the restriction of the weak-$\ast$ topology to $\mathcal{P}(X)$.
* **Compactness theorem (Prokhorov)**: on a Polish $X$, a family $\mathcal{M} \subseteq \mathcal{P}(X)$ is relatively compact in this topology iff it is **tight** — see [[the Prokhorov entry]].
* **Portmanteau theorem** characterizes $\mu_n \rightharpoonup \mu$ by half a dozen equivalent conditions: convergence of $\int f\, d\mu_n$ for all bounded continuous $f$ / for all bounded uniformly continuous $f$; $\liminf \mu_n(G) \ge \mu(G)$ for open $G$; $\limsup \mu_n(F) \le \mu(F)$ for closed $F$; $\mu_n(A) \to \mu(A)$ for every "$\mu$-continuity set" $A$ (a Borel $A$ with $\mu(\partial A) = 0$).
* On Polish $X$ this topology is itself **Polish** (metrizable by Lévy–Prokhorov or bounded-Lipschitz metric), so almost everything carries through as for sequences.

</div>

---

## Compactness flavors

### Compact support

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(compact support)</span></p>

A function $f$ on $X$ has compact support if 

$$\operatorname{supp}(f) := \overline{\lbrace x : f(x) \ne 0\rbrace}$$

is compact.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Combined with smoothness this is the $C_c^\infty$ class — the test functions.

</div>

### Precompact (relatively compact) sets

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(precompact set)</span></p>

$A \subseteq X$ is **precompact** (relatively compact) if $\overline{A}$ is compact. In a metric space, equivalent to: every sequence in $A$ has a convergent subsequence (limit possibly outside $A$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

This is what one actually verifies in compactness arguments — you don't need $A$ itself closed.

</div>

### Locally compact spaces

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(locally compact)</span></p>

A topological space $X$ is **locally compact** if every point has a compact neighborhood.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(finite vs infinite dimension)</span></p>

$\mathbb{R}^n$ is locally compact. **Infinite-dimensional Banach spaces are not** (closed bounded balls are not norm-compact). This is exactly why infinite-dimensional analysis is hard and weak topologies become essential. Locally compact Hausdorff (LCH) is the setting for the Riesz representation theorem for measures and for $C_0(X)$ / $C_c(X)$.

</div>

### Weakly compact sets

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weakly compact)</span></p>

A set is **weakly compact** if it is compact in the weak topology.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(in reflexive spaces)</span></p>

In a **reflexive** Banach space, weakly compact $=$ closed and bounded (Banach–Alaoglu + reflexivity). This is how you compactify in $L^p$ for $1 < p < \infty$: you give up strong compactness, but bounded sequences have weak limit points for free.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(sequential version — Eberlein–Šmulian)</span></p>

In general topological spaces compact $\ne$ sequentially compact (compactness gives convergent **nets**, not necessarily subsequences), and the weak topology is not metrizable. A priori "weakly compact" therefore gives nets only.

**Eberlein–Šmulian** erases that gap in Banach spaces: a subset $A$ of a Banach space is weakly compact iff weakly sequentially compact iff relatively weakly countably compact. So every bounded sequence in a reflexive space **really does** have a weakly convergent **subsequence** — this is what you actually use in PDE / calc-of-variations.

Don't confuse with **weak-$\ast$ compactness** in $X^\ast$: that is governed by Banach–Alaoglu (the closed unit ball of $X^\ast$ is always weak-$\ast$ compact) and becomes sequential exactly when $X$ is separable (the ball is then metrizable in the weak-$\ast$ topology).

</div>

### Equicontinuous family of functions

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(equicontinuity)</span></p>

A family $\mathcal{F} \subseteq C(X)$ is **equicontinuous at $x_0$** if for every $\varepsilon > 0$ there exists a **single** $\delta > 0$ (independent of $f \in \mathcal{F}$) with

$$
\lvert f(x) - f(x_0) \rvert < \varepsilon \quad \text{for all } f \in \mathcal{F}, \; \lvert x - x_0 \rvert < \delta.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

"Uniformly continuous, uniformly across the family." Half of the hypothesis of Arzelà–Ascoli.

</div>

### Lower semicontinuous functions

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(lower semicontinuity)</span></p>

$f : X \to \mathbb{R} \cup \lbrace +\infty\rbrace$ is **lower semicontinuous (l.s.c.)** at $x_0$ if

$$
\liminf_{x \to x_0} f(x) \ge f(x_0).
$$

Equivalently, $\lbrace f \le c\rbrace$ is closed for every $c \in \mathbb{R}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why it matters for optimization)</span></p>

The right class for **minimization**: a l.s.c. function attains its infimum on a compact set (Weierstrass). Norms are l.s.c. **for weak convergence** — which is why one minimizes energies by passing to weak limits.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(four canonical plots)</span></p>

<figure>
  <img src="{{ 'assets/images/notes/cheat-sheet/lsc_examples.png' | relative_url }}" alt="Four canonical lower-semicontinuous examples and one non-example" loading="lazy">
  <figcaption>Rule of thumb at every discontinuity: $f$ is l.s.c. iff the filled dot lies on the <em>lower</em> branch. The empty circle marks the value the limit approaches but $f$ does not take.</figcaption>
</figure>

* **(a) Step, l.s.c.** — $f(x) = 0$ for $x \le 0$, $f(x) = 1$ for $x > 0$. At $x_0 = 0$: $f(0) = 0$ and $\liminf\_{x \to 0} f(x) = 0$. The inequality $\liminf \ge f(x_0)$ holds with equality. The value lands on the **lower** branch.
* **(b) Step, NOT l.s.c.** — same graph but with $f(0) = 1$. Now $\liminf\_{x \to 0} f(x) = 0 < 1 = f(0)$, so l.s.c. fails. This $f$ is **upper** semicontinuous instead. Same picture, dot on the upper branch.
* **(c) Isolated dip, l.s.c.** — $f(x) = x^2$ for $x \ne 0$ and $f(0) = -0.6$. The value at $0$ is *strictly below* the surrounding limit ($\liminf = 0$). l.s.c. allows this: a function can dip down at isolated points but never spike up.
* **(d) Convex-analysis indicator $\delta_{[-1,1]}$, l.s.c.** — $0$ on $[-1,1]$, $+\infty$ off it. The sublevel sets are $\lbrace \delta\_C \le c\rbrace = [-1,1]$ for $c \ge 0$ and $\emptyset$ for $c < 0$, both closed. This is *the* canonical extended-real-valued l.s.c. function in convex analysis — it lets you encode the constraint "$x \in C$" inside an unconstrained minimization.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(epigraph characterization)</span></p>

<figure>
  <img src="{{ 'assets/images/notes/cheat-sheet/lsc_epigraph.png' | relative_url }}" alt="Epigraph characterization of lower semicontinuity" loading="lazy">
  <figcaption>$f$ is l.s.c. iff its epigraph $\mathrm{epi}(f) = \lbrace (x,t) : t \ge f(x)\rbrace$ is a closed subset of $X \times \mathbb{R}$.</figcaption>
</figure>

The cleanest reformulation: **l.s.c. = closed epigraph**. At a downward jump (filled dot on the lower side), the epigraph picks up the vertical line all the way down to that lower value, and the set stays closed. If instead the filled dot were on the upper side (the u.s.c. case), the epigraph would be missing exactly that vertical segment between the two branches and would no longer be closed.

This is the version of l.s.c. you actually use in convex analysis and the calculus of variations — closedness of the epigraph is preserved under pointwise suprema, infimal convolutions, perspectives, etc., which is why **a supremum of any family of continuous (or l.s.c.) functions is l.s.c.**, while suprema of u.s.c. functions are generally not u.s.c.

</div>

---

## "Niceness" properties of spaces

### Hausdorff space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hausdorff / $T_2$)</span></p>

A topological space $X$ is **Hausdorff** ($T_2$) if for any two distinct points $x \ne y$ there exist disjoint open sets $U \ni x$ and $V \ni y$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(what Hausdorff buys you)</span></p>

Almost every space met in analysis is Hausdorff — metric spaces, normed and topological vector spaces, manifolds, the analytic side of schemes. It is silently assumed because without it the most basic facts fail:

* **Limits are unique** (sequences, nets, filters).
* **Compact sets are closed**; finite sets, in particular singletons, are closed.
* The diagonal $\Delta_X = \lbrace (x,x) : x \in X\rbrace$ is closed in $X \times X$.
* Two continuous maps into a Hausdorff space that agree on a dense subset are equal.

Common non-Hausdorff settings: the Zariski topology on $\mathrm{Spec}(R)$, quotient topologies that fail to separate orbits, the cofinite topology on infinite sets.

</div>

### Separable spaces

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(separable)</span></p>

A topological space is **separable** if it contains a **countable dense** subset.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(who is and isn't)</span></p>

* Separable: $L^p$ ($1 \le p < \infty$) on $\sigma$-finite measures, $C(K)$ for $K$ compact metric, Sobolev $W^{k,p}$ for $p < \infty$, every Hilbert space with countable orthonormal basis.
* **Not** separable: $L^\infty$, $\ell^\infty$, $C_b(\mathbb{R})$.
* Buys you: countable bases, approximation by finite-dimensional projections, **metrizability of weak/weak-$*$ topologies on bounded sets**, well-posed Borel probability.

</div>

### Reflexive spaces

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(reflexive)</span></p>

A Banach space $X$ is **reflexive** if the canonical embedding 

$$J : X \to X^{\ast\ast}$$

$$J(x)(x^\ast) = x^\ast(x)$$

is **surjective** (and hence an isometric isomorphism).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(payoff)</span></p>

Closed bounded sets are **weakly compact**, so bounded sequences have weakly convergent subsequences.

* Examples: Hilbert spaces, $L^p$ ($1<p<\infty$), $W^{k,p}$ ($1<p<\infty$).
* Non-examples: $L^1$, $L^\infty$, $C(K)$.

</div>

### Metrizable space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(metrizable)</span></p>

A topological space $(X, \tau)$ is **metrizable** if there exists a metric $d$ on $X$ inducing $\tau$ — the open sets of $\tau$ are exactly the unions of open balls of $d$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(what metrizability buys you)</span></p>

A metrizable topology is "as nice as $\mathbb{R}$":

* **Sequences capture the topology** — closure $=$ sequential closure, continuity $=$ sequential continuity.
* The whole metric toolbox: Cauchy criterion, completion, compactness via total boundedness, uniform continuity, Lipschitz maps.
* **Urysohn's metrization theorem**: $X$ is metrizable iff it is regular, Hausdorff, and second-countable.
* Many "abstract" topologies of analysis are metrizable **on bounded subsets** even when not globally — e.g. the weak-$\ast$ topology on $B_{X^\ast}$ when $X$ is separable. This is what makes Banach–Alaoglu sequential in practice.

</div>

### Completely metrizable space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(completely metrizable)</span></p>

A topological space $X$ is **completely metrizable** if there exists a metric $d$ inducing its topology under which $(X, d)$ is **complete** (every Cauchy sequence converges).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(topological, not metric)</span></p>

* The property is intrinsic to the **topology**, not the chosen metric. Example: $(0,1)$ with the usual metric is not complete, but it is completely metrizable — it is homeomorphic to $\mathbb{R}$, which is complete.
* Equivalently (Alexandrov / Čech): $X$ is completely metrizable iff $X$ is a $G_\delta$ subset of some compact metric space.
* Buys you **Baire category**: every completely metrizable space is a Baire space. This is the backbone of the open mapping theorem, the closed graph theorem, the uniform boundedness principle, and most generic-point arguments.
* **Completely metrizable + separable = Polish** — the natural arena for probability theory.

</div>

### Polish spaces

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Polish space)</span></p>

A topological space that is **separable and completely metrizable** (admits a complete metric inducing its topology).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role in probability)</span></p>

The natural arena for probability on function-valued / infinite-dimensional objects:

* Regular conditional probabilities exist.
* Borel measures are automatically **tight** (Ulam).
* Prokhorov characterizes weak compactness via tightness.

Examples: $\mathbb{R}^n$, $C([0,T])$, every separable Banach space, $\mathcal{D}'$ (with its standard topology).

</div>

---

## Probability and stochastic processes

### Filtration

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(filtration)</span></p>

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space and $I \subseteq \mathbb{R}$ an index set ("time"). A **filtration** is a family

$$
\mathbb{F} := (\mathcal{F}_t)_{t \in I}
$$

of sub-$\sigma$-algebras of $\mathcal{F}$ that is **monotone**:

$$
\mathcal{F}_s \subseteq \mathcal{F}_t \subseteq \mathcal{F} \qquad \text{whenever } s \le t.
$$

The filtered probability space is then $(\Omega, \mathcal{F}, \mathbb{F}, \mathbb{P})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(intuition)</span></p>

$\mathcal{F}\_t$ is the **information available up to time $t$**: the collection of events whose occurrence/non-occurrence can be decided using only what has happened by time $t$. The monotonicity condition says "information is never forgotten."

Vocabulary you will see attached to a filtration:

* A process $X = (X\_t)\_{t \in I}$ is **$\mathbb{F}$-adapted** if $X\_t$ is $\mathcal{F}\_t$-measurable for every $t$ — its value at time $t$ depends only on past information.
* A random time $\tau : \Omega \to I \cup \lbrace +\infty\rbrace$ is a **stopping time** w.r.t. $\mathbb{F}$ if $\lbrace\tau \le t\rbrace \in \mathcal{F}\_t$ for every $t$ — you can decide whether $\tau$ has occurred from the information at time $t$.
* The **usual conditions**: $\mathbb{F}$ is right-continuous ($\mathcal{F}\_t = \bigcap\_{s > t}\mathcal{F}\_s$) and $\mathcal{F}\_0$ contains all $\mathbb{P}$-null sets. This is the technical hygiene assumed in continuous-time stochastic analysis (Itô calculus, martingales).

</div>

### Natural / canonical filtration

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(natural filtration of a process)</span></p>

Let $X = (X\_t)\_{t \in I}$ be a stochastic process on $(\Omega, \mathcal{F}, \mathbb{P})$. The **natural** (or **canonical**) filtration generated by $X$ is

$$
\mathcal{F}_t^X := \sigma\!\big(X_s : s \in I, \; s \le t\big),
$$

i.e. the smallest sub-$\sigma$-algebra of $\mathcal{F}$ making every past value $X\_s$, $s \le t$, measurable. Set $\mathbb{F}^X := (\mathcal{F}\_t^X)\_{t \in I}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why this filtration)</span></p>

$\mathcal{F}\_t^X$ encodes exactly **the information you obtain by observing $X$ up to time $t$**, and nothing else. It is the **minimal** filtration to which $X$ is adapted — for any other filtration $\mathbb{G}$ with $X$ $\mathbb{G}$-adapted, $\mathcal{F}\_t^X \subseteq \mathcal{G}\_t$.

Two reasons you constantly meet it:

* When no filtration is specified (e.g. "the Markov property", "$X$ is a martingale") the implicit default is $\mathbb{F}^X$.
* Constructing $\mathbb{F}^X$ is how you upgrade a bare process $X$ into a filtered setup where you can talk about adaptedness, stopping times, conditional expectations $\mathbb{E}[\cdot \mid \mathcal{F}\_t^X]$, and the Markov / martingale properties.

Caveat: $\mathbb{F}^X$ generally does **not** satisfy the usual conditions out of the box — you typically augment it with $\mathbb{P}$-null sets and pass to its right-continuous version to obtain the **augmented natural filtration** before doing serious stochastic analysis.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(natural vs non-natural filtration of the same process)</span></p>

Take i.i.d. fair $\pm 1$ steps $X_1, X_2, \dots$ and the simple symmetric random walk $S_n = X_1 + \cdots + X_n$. Assume the underlying $(\Omega, \mathcal{F}, \mathbb{P})$ is rich enough to also carry an **independent** fair coin $U \in \lbrace 0, 1\rbrace$ with $U \perp (X_k)\_{k \ge 1}$.

**Natural filtration of $S$.**

$$
\mathcal{F}_n^S \;=\; \sigma(S_1, \dots, S_n) \;=\; \sigma(X_1, \dots, X_n).
$$

At time $n$, $\mathcal{F}\_n^S$ knows exactly the first $n$ steps — nothing earlier and nothing later. $S$ is adapted, and $S$ is a martingale:

$$
\mathbb{E}\!\left[S_{n+1} \mid \mathcal{F}_n^S\right] \;=\; S_n + \mathbb{E}[X_{n+1}] \;=\; S_n.
$$

**Two non-natural filtrations of the *same* process.** Both are different from $\mathbb{F}^S$, both contain $\mathbb{F}^S$, both keep $S$ adapted — but they differ in *what kind* of extra information they carry.

* **(A) Look-ahead:** $\mathcal{G}\_n := \mathcal{F}\_{n+1}^S = \sigma(X_1, \dots, X_{n+1})$.

    Strictly finer than $\mathbb{F}^S$: the event $\lbrace X\_{n+1} = +1\rbrace$ lies in $\mathcal{G}\_n$ but not in $\mathcal{F}\_n^S$. The walk $S$ is still $\mathbb{G}$-adapted ($S_n$ depends only on $X_1, \dots, X_n \subseteq \mathcal{G}_n$), but the **martingale property is destroyed**:
    
    $$
    \mathbb{E}\!\left[S_{n+1} \mid \mathcal{G}_n\right] \;=\; S_n + X_{n+1} \;=\; S_{n+1} \;\ne\; S_n.
    $$
    
    The "filtration" leaks future information into the present.

* **(B) Independent augmentation:** $\mathcal{H}\_n := \mathcal{F}\_n^S \vee \sigma(U)$.

    Also strictly finer than $\mathbb{F}^S$, but in an *orthogonal* direction: the event $\lbrace U = 1\rbrace$ lies in $\mathcal{H}\_0$ already, while $\mathcal{F}\_0^S = \lbrace\emptyset, \Omega\rbrace$ is trivial. Because $U \perp X\_{n+1}$, the martingale property **survives**:
    
    $$
    \mathbb{E}\!\left[S_{n+1} \mid \mathcal{H}_n\right] \;=\; S_n + \mathbb{E}[X_{n+1} \mid U] \;=\; S_n.
    $$

**Reading the contrast.** $\mathbb{F}^S$ is the **minimal** filtration adapting $S$. $\mathbb{H}$ adds "harmless" extra information independent of the process — it behaves identically to $\mathbb{F}^S$ for every property of $S$ that is sensitive only to the conditional distributions of $X$ given the past (martingale property, Markov property, stopping-time decompositions for $S$ alone). $\mathbb{G}$ adds "dangerous" extra information about the future of the very process you are filtering — it formally adapts $S$ but corrupts every theorem whose statement uses conditional expectations given the past, because the "past" now includes some of the future.

This is exactly why one bothers to single out the natural filtration: in statements like "$S$ is an $\mathbb{F}$-martingale" the filtration matters, and the default $\mathbb{F}^S$ is the unique smallest choice that makes such a statement non-vacuous and non-falsified-by-leaks.

</div>

### Tightness / tight measure

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(tight family of measures)</span></p>

A family $\mathcal{M}$ of *Borel probability measures* on a topological space $X$ is **tight** if for every $\varepsilon > 0$ there exists a **compact** $K \subseteq X$ with

$$
\mu(K) \;\ge\; 1 - \varepsilon \qquad \text{for every } \mu \in \mathcal{M}.
$$

A single Borel measure $\mu$ is **tight** (a.k.a. inner regular by compacts) if for every Borel $A$ and every $\varepsilon > 0$ there is a compact $K \subseteq A$ with $\mu(A \setminus K) < \varepsilon$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(intuition and role)</span></p>

"**No mass escapes to infinity**" — every measure in the family puts almost all of its mass on a single compact set, uniformly in the family.

* On a **Polish** space, every individual Borel probability measure is automatically tight (**Ulam's theorem**) — so "tightness" is really a property of *families*.
* **Prokhorov's theorem**: on a Polish space, a family is **relatively compact in the weak topology of measures** iff it is **tight**. Tightness is the probabilistic Banach–Alaoglu.
* Standard non-tight example: $\mu_n = \delta_n$ on $\mathbb{R}$. Any fixed compact $K$ has $\mu_n(K) = 0$ for all $n$ large enough — the mass leaks to $+\infty$, no weakly convergent subsequence exists.
* In practice, tightness is verified by a **uniform moment bound** ($\sup_\mu \int \lvert x \rvert^p \, d\mu < \infty$ implies tightness on $\mathbb{R}^n$ via Markov), or by an Aldous-type criterion on path space.

</div>

---

## The headline theorems

### Riesz representation theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Riesz — Hilbert version)</span></p>

Let $H$ be a Hilbert space. For every continuous linear functional $\varphi \in H^\ast$ there is a **unique** $y \in H$ with

$$
\varphi(x) = \langle x, y \rangle \quad \forall x \in H,
$$

and $\|\varphi\|\_{H^\ast} = \|y\|\_H$. So $H^\ast \cong H$ canonically.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Riesz — measure version)</span></p>

Let $X$ be locally compact Hausdorff. Every positive linear functional $\Lambda : C_c(X) \to \mathbb{R}$ is given by integration against a **unique** regular Borel measure $\mu$:

$$
\Lambda(f) = \int_X f \, d\mu.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The bridge from "abstract functional" to "concrete object" (vector or measure). Measures **are** the continuous linear functionals on the test-function space.

</div>

### Hahn–Banach theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hahn–Banach — extension form)</span></p>

Let $X$ be a real vector space, $p : X \to \mathbb{R}$ a **sublinear** functional ($p(x+y) \le p(x)+p(y)$, $p(\lambda x) = \lambda p(x)$ for $\lambda \ge 0$), and $f : Y \to \mathbb{R}$ a linear functional on a subspace $Y \subseteq X$ with $f(y) \le p(y)$ on $Y$. Then $f$ extends to a linear functional $\widetilde f : X \to \mathbb{R}$ with $\widetilde f(x) \le p(x)$ for all $x \in X$.

In particular: any bounded linear functional on a subspace of a normed space extends to all of $X$ with the **same norm**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hahn–Banach — geometric / separation form)</span></p>

Let $X$ be a locally convex TVS and $A, B \subseteq X$ disjoint, non-empty, convex.

* If $A$ is **open**, there is a continuous linear $f$ and $\alpha \in \mathbb{R}$ with $f(a) < \alpha \le f(b)$ for all $a \in A$, $b \in B$.
* If $A$ is **compact** and $B$ is **closed**, the separation can be made **strict**: $f(a) < \alpha < \beta < f(b)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role — duals are rich, geometry is convex)</span></p>

* Hahn–Banach is what makes the dual $X^\ast$ **non-trivial** and rich enough to separate points: for any $x \ne 0$ there is $x^\ast \in X^\ast$ with $x^\ast(x) = \lVert x\rVert$. Without H–B, $X^\ast$ could in principle be $\lbrace 0\rbrace$.
* Underpins the entire theory of weak topologies, biduals, and reflexivity (the canonical map $X \hookrightarrow X^{\ast\ast}$ is well-defined and isometric thanks to H–B).
* The separation form is the analytic engine behind **convex duality**: characterization of closed convex sets as intersections of half-spaces, Mazur's theorem, the bipolar theorem, supporting hyperplanes, Fenchel duality.
* Does **not** require completeness; it requires (some form of) the axiom of choice (Zorn).

</div>

### Parseval's identity

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Parseval)</span></p>

For a Hilbert space $H$ with orthonormal basis $\lbrace e_n\rbrace\_{n \in \mathbb{N}}$ and any $x \in H$,

$$
\|x\|_H^2 = \sum_{n} \lvert \langle x, e_n \rangle \rvert^2.
$$

Concretely for Fourier series on $[0, 2\pi]$: $\int_0^{2\pi} \lvert f \rvert^2 \, dx = 2\pi \sum_{n \in \mathbb{Z}} \lvert \hat f_n \rvert^2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Says the Fourier transform is an **isometry** $L^2 \to \ell^2$ (or $L^2 \to L^2$ on $\mathbb{R}^n$).

</div>

### Hölder's inequality

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hölder)</span></p>

Let $(\Omega, \mu)$ be a measure space and $p, q \in [1, \infty]$ with $\tfrac{1}{p} + \tfrac{1}{q} = 1$ (**conjugate exponents**, with the convention $1/\infty = 0$). For measurable $f, g$,

$$
\Big\lvert \int_\Omega f \, g \, d\mu \Big\rvert \;\le\; \int_\Omega \lvert f g\rvert \, d\mu \;\le\; \lVert f \rVert_{L^p} \, \lVert g \rVert_{L^q}.
$$

In particular $f g \in L^1$ whenever $f \in L^p$ and $g \in L^q$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role and generalizations)</span></p>

* The reason the **dual pairing** $(f, g) \mapsto \int f g$ is well defined for $f \in L^p$, $g \in L^q$ — and the reason $(L^p)^\ast \cong L^q$ for $1 \le p < \infty$.
* $p = q = 2$ is the **Cauchy–Schwarz** inequality.
* **Generalized Hölder**: if $\tfrac{1}{r} = \tfrac{1}{p_1} + \cdots + \tfrac{1}{p_k}$, then $\lVert f_1 \cdots f_k\rVert_{L^r} \le \prod_i \lVert f_i\rVert_{L^{p_i}}$.
* **Equality case**: requires $\lvert f \rvert^p$ and $\lvert g\rvert^q$ to be proportional a.e. — the analyst's analogue of "vectors are parallel."

</div>

### Minkowski inequality

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Minkowski)</span></p>

For $1 \le p \le \infty$ and $f, g \in L^p(\Omega, \mu)$,

$$
\lVert f + g \rVert_{L^p} \;\le\; \lVert f \rVert_{L^p} + \lVert g \rVert_{L^p}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role and integral form)</span></p>

* This is exactly the **triangle inequality** for the $L^p$ norm — the statement that $\lVert \cdot \rVert_{L^p}$ is a genuine norm (for $p \ge 1$).
* Proof is via Hölder applied to $\lvert f + g\rvert^p = \lvert f + g\rvert \cdot \lvert f+g\rvert^{p-1}$.
* Fails for $0 < p < 1$ — there $L^p$ is only a quasi-Banach space and the right inequality is $\lVert f + g\rVert^p \le \lVert f\rVert^p + \lVert g\rVert^p$.
* **Integral / Minkowski's integral inequality**: $\bigl\lVert \int F(\cdot, y)\, d\nu(y) \bigr\rVert_{L^p_x} \le \int \lVert F(\cdot, y)\rVert_{L^p_x} \, d\nu(y)$ — "$L^p$-norm of an integral $\le$ integral of $L^p$-norms." This is the form used to control convolutions and to swap $L^p$ with auxiliary integrations.

</div>

### Young's inequality

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Young — for products)</span></p>

For $a, b \ge 0$ and conjugate exponents $\tfrac{1}{p} + \tfrac{1}{q} = 1$ with $1 < p, q < \infty$,

$$
a b \;\le\; \frac{a^p}{p} + \frac{b^q}{q}.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Young's convolution inequality)</span></p>

For $f \in L^p(\mathbb{R}^n)$, $g \in L^q(\mathbb{R}^n)$ with $\tfrac{1}{p} + \tfrac{1}{q} = 1 + \tfrac{1}{r}$ and $1 \le p, q, r \le \infty$,

$$
\lVert f \ast g \rVert_{L^r} \;\le\; \lVert f \rVert_{L^p} \, \lVert g \rVert_{L^q}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role — the proof of Hölder and the analysis of convolutions)</span></p>

* The pointwise form $ab \le a^p/p + b^q/q$ is the **standard one-line proof** of Hölder: bound $\lvert f g\rvert / (\lVert f\rVert_p \lVert g\rVert_q)$ by $\tfrac{1}{p}(\lvert f\rvert/\lVert f\rVert_p)^p + \tfrac{1}{q}(\lvert g\rvert/\lVert g\rVert_q)^q$ and integrate.
* The convolution form is how you control mollifications, heat-kernel smoothing, regularization arguments — the staple of PDE estimates.
* Special case $p = 1$: $\lVert f \ast g\rVert_{L^q} \le \lVert f\rVert_{L^1} \lVert g\rVert_{L^q}$ — convolving with an $L^1$ function preserves $L^q$ (this is why mollifying with an $L^1$ kernel gives bounded operators on every $L^q$).

</div>

### Riesz–Fischer theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Riesz–Fischer)</span></p>

For every $1 \le p \le \infty$ and every measure space $(\Omega, \mathcal{A}, \mu)$, $L^p(\Omega, \mu)$ is **complete** — i.e. it is a Banach space.

Equivalently, in $L^2$ form: if $\lbrace e_n\rbrace$ is an orthonormal basis of a separable Hilbert space $H$ and $(c_n) \in \ell^2$, then $\sum c_n e_n$ converges in $H$. So $\ell^2 \to H$, $(c_n) \mapsto \sum c_n e_n$, is an **isometric isomorphism**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why it matters)</span></p>

* This is **the** theorem that justifies treating $L^p$ as a Banach space — the missing ingredient that the Riemann-integrable functions did not have.
* Proof template: from a Cauchy sequence $(f_n)$ extract a fast subsequence $(f_{n_k})$ with $\sum \lVert f_{n_{k+1}} - f_{n_k}\rVert_p < \infty$; Minkowski + monotone convergence give a pointwise a.e. limit; show it is the $L^p$ limit.
* The $L^2$ form is what makes Fourier series legal: every $\ell^2$ sequence of Fourier coefficients comes from an honest $L^2$ function.

</div>

### Arzelà–Ascoli theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Arzelà–Ascoli)</span></p>

Let $K$ be a compact metric space. A family $\mathcal{F} \subseteq C(K)$ is **precompact in the uniform topology** iff

1. $\mathcal{F}$ is **pointwise bounded**, and
2. $\mathcal{F}$ is **equicontinuous**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

How you build compactness in $C(K)$. Used for: existence of solutions to ODEs (Peano), normal families of holomorphic functions, regularity arguments in PDE.

</div>

### Tonelli's theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Tonelli)</span></p>

Let $(\Omega_1, \mathcal{A}_1, \mu_1)$, $(\Omega_2, \mathcal{A}_2, \mu_2)$ be $\sigma$-finite measure spaces, and $f : \Omega_1 \times \Omega_2 \to [0, \infty]$ measurable. Then

$$
\int_{\Omega_1 \times \Omega_2} f \, d(\mu_1 \otimes \mu_2) \;=\; \int_{\Omega_1} \!\!\int_{\Omega_2} f \, d\mu_2 \, d\mu_1 \;=\; \int_{\Omega_2} \!\!\int_{\Omega_1} f \, d\mu_1 \, d\mu_2,
$$

with **no integrability assumption**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(vs Fubini)</span></p>

Tonelli says: "if it is $\ge 0$, you can swap the order, full stop." Fubini is the same statement under the extra assumption $f \in L^1(\mu_1 \otimes \mu_2)$ and is what you need for signed $f$. Standard combo: use Tonelli on $\lvert f \rvert$ to check $f \in L^1$, then apply Fubini.

</div>

### Fubini's theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fubini)</span></p>

Let $(\Omega_1, \mathcal{A}_1, \mu_1)$, $(\Omega_2, \mathcal{A}_2, \mu_2)$ be $\sigma$-finite measure spaces and $f \in L^1(\mu_1 \otimes \mu_2)$. Then for $\mu_1$-a.e. $x \in \Omega_1$ 
1. the slice $f(x, \cdot)$ is in $L^1(\mu_2)$,
2. the map $x \mapsto \int f(x, y)\, d\mu_2(y)$ is in $L^1(\mu_1)$,

3. the integral could split and exchange the order
   
   $$
   \int_{\Omega_1 \times \Omega_2} f \, d(\mu_1 \otimes \mu_2) \;=\; \int_{\Omega_1}\!\!\int_{\Omega_2} f(x, y) \, d\mu_2(y) \, d\mu_1(x) \;=\; \int_{\Omega_2}\!\!\int_{\Omega_1} f(x, y) \, d\mu_1(x) \, d\mu_2(y).
   $$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role and the standard combo)</span></p>

* Fubini = "you can swap the order of integration **once** $f \in L^1$." This is the *signed* (or complex) analogue of Tonelli.
* The standard workflow when staring at an iterated integral with sign changes: apply **Tonelli** to $\lvert f\rvert$ to check $\int\int \lvert f\rvert < \infty$; once that is in hand, **Fubini** applies and you can swap freely.
* Both hypotheses are sharp. The classic counterexample $f(x, y) = (x^2 - y^2)/(x^2 + y^2)^2$ on $(0,1)^2$ has both iterated integrals well-defined but **unequal** ($+\pi/4$ vs $-\pi/4$) — and indeed $\int\int \lvert f\rvert = \infty$, so Fubini's hypothesis fails.
* On $\sigma$-finite measure spaces, the product $\sigma$-algebra and measure are unambiguously defined. Without $\sigma$-finiteness one needs the Bourbaki / Radon measure framework instead.

</div>

### Monotone convergence theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(MCT — Beppo Levi)</span></p>

Let $(f_n)$ be a sequence of $[0, \infty]$-valued measurable functions on $(\Omega, \mathcal{A}, \mu)$ with $f_n \uparrow f$ pointwise a.e. (or equivalently $f_n \le f_{n+1}$ for every $n$). Then

$$
\int_\Omega f_n \, d\mu \;\uparrow\; \int_\Omega f \, d\mu.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role — the foundation)</span></p>

* The **first** of the three integration limit theorems, and effectively the **definition step** of the Lebesgue integral itself: the integral of a non-negative measurable function is defined as $\sup$ of integrals over simple-function approximations, and MCT is what makes that definition well-behaved.
* Standard consequence: countable additivity of the integral, $\int \sum_n f_n = \sum_n \int f_n$ for non-negative $f_n$. This is how one swaps $\sum$ and $\int$ without justification when everything is non-negative.
* No domination hypothesis is needed — non-negativity and monotonicity do all the work.

</div>

### Fatou's lemma

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fatou)</span></p>

For $[0, \infty]$-valued measurable $f_n$,

$$
\int_\Omega \liminf_{n \to \infty} f_n \, d\mu \;\le\; \liminf_{n \to \infty} \int_\Omega f_n \, d\mu.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(intuition — mass can escape but not appear)</span></p>

* "Mass can be lost in the limit but not created." Equality fails in general — escaping bumps $f_n = \chi_{[n,n+1]}$ on $\mathbb{R}$ have $\int f_n = 1$ for every $n$ while $\liminf f_n = 0$ pointwise, so $\int \liminf = 0 < 1 = \liminf \int$.
* No monotonicity, no domination — only non-negativity. This is what makes Fatou the **all-purpose** lower-semicontinuity tool for integrals: $\int \lvert \cdot\rvert^p$ is lower semicontinuous under a.e. convergence.
* In disguise this is exactly why norms on $L^p$ are weakly lower semicontinuous: a weak limit cannot have more mass than the sequence.

</div>

### Dominated convergence theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(DCT — Lebesgue)</span></p>

Let $f_n : \Omega \to \mathbb{R}$ (or $\mathbb{C}$) be measurable with $f_n \to f$ pointwise a.e., and suppose there exists a **dominating function** $g \in L^1(\mu)$ with

$$
\lvert f_n(x) \rvert \;\le\; g(x) \qquad \text{for a.e. } x, \text{ every } n.
$$

Then $f \in L^1$ and

$$
\int_\Omega f_n \, d\mu \;\longrightarrow\; \int_\Omega f \, d\mu, \qquad \lVert f_n - f\rVert_{L^1} \to 0.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role — the workhorse limit theorem for signed integrands)</span></p>

* The **all-purpose** swap-limit-and-integral theorem in analysis. Whenever you want $\lim \int = \int \lim$ for signed or complex-valued sequences, the first thing you reach for is a uniform dominant.
* DCT also yields strong $L^p$ convergence ($1 \le p < \infty$) under domination of $\lvert f_n\rvert^p$ by an $L^1$ function — so it powers convergence in $L^p$, continuity of integrals in parameters, differentiation under the integral, $\partial_t \int = \int \partial_t$, etc.
* Three-cornered "trinity" — MCT, Fatou, DCT — are logically interderivable: prove one, the other two follow. MCT is cleanest to set up; Fatou is the most flexible; DCT is the one you actually quote 90% of the time.
* **Vitali's convergence theorem** replaces "dominated" with "uniformly integrable + tight," which is sometimes the hypothesis you can actually verify (e.g. weak convergence in $L^1$).

</div>

### Banach–Alaoglu theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Banach–Alaoglu)</span></p>

Let $X$ be a normed space. The closed unit ball $B_{X^*}$ of the dual is **compact in the weak-$*$ topology**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(sequential version)</span></p>

If $X$ is **separable**, $B_{X^\ast}$ is metrizable in the weak-$\ast$ topology, hence **sequentially compact**: every bounded sequence in $X^\ast$ has a weak-$\ast$ convergent subsequence. The workhorse for extracting limits in infinite dimensions (limits of measures, of $L^\infty$ data, of approximate solutions of PDE).

</div>

### Mazur's theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Mazur — convex hulls and weak closure)</span></p>

Let $X$ be a Banach space (or, more generally, a locally convex TVS) and $A \subseteq X$ a **convex** subset. Then

$$
\overline{A}^{\,\text{norm}} \;=\; \overline{A}^{\,\text{weak}}.
$$

I.e. a convex set is norm-closed iff weakly closed.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Mazur's lemma)</span></p>

If $x_n \rightharpoonup x$ weakly in a Banach space $X$, then there exists a sequence of **convex combinations**

$$
y_N \;=\; \sum_{k=N}^{M(N)} \lambda_k^N x_k, \qquad \lambda_k^N \ge 0, \;\; \sum_k \lambda_k^N = 1,
$$

with $y_N \to x$ in **norm**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why it matters — convex functions are weakly l.s.c.)</span></p>

* Direct consequence of Hahn–Banach separation: a norm-closed convex set is the intersection of closed half-spaces, each of which is weakly closed.
* The standard application: **convex, norm-continuous functions $J : X \to \mathbb{R}$ are automatically weakly lower semicontinuous** (their sublevel sets are convex and norm-closed, hence weakly closed). This is exactly the hypothesis needed for the direct method of the calculus of variations — bounded minimizing sequences have weak limits, $J$ does not jump up under them, so the limit is a minimizer.
* Mazur's lemma is what lets you "upgrade" a weak limit to a strong limit when you have enough convexity — for instance, to pass to the limit in nonlinear convex functionals along a minimizing sequence.

</div>

### Prokhorov's theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Prokhorov)</span></p>

Let $X$ be a Polish space. A family $\mathcal{M}$ of Borel probability measures on $X$ is **relatively compact in the weak topology of measures** iff it is **tight**: for every $\varepsilon > 0$ there exists a compact $K \subseteq X$ with

$$
\mu(K) \ge 1 - \varepsilon \quad \text{for every } \mu \in \mathcal{M}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The probabilistic analogue of Banach–Alaoglu, and the standard route to proving existence of weak limits of random measures / laws of processes (invariance principles, scaling limits, stochastic PDE).

</div>

---

## Weak calculus, weak solutions

### Weak derivatives

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weak derivative)</span></p>

For $f \in L^1\_{\text{loc}}(\Omega)$ and multi-index $\alpha$, a function $g \in L^1\_{\text{loc}}(\Omega)$ is the **weak $\alpha$-th derivative** of $f$, written $g = D^\alpha f$, if

$$
\int_\Omega f \, D^\alpha \varphi \, dx \;=\; (-1)^{\lvert \alpha \rvert} \int_\Omega g \, \varphi \, dx \qquad \forall \varphi \in C_c^\infty(\Omega).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

You **define** derivatives by what they do under integration by parts against test functions. Coincides with the classical derivative when $f$ is smooth, but extends to objects like $\lvert x \rvert$, kink functions, or merely $L^p$ data.

</div>

### Weak solutions

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weak solution)</span></p>

A function that satisfies a PDE only after pairing with test functions and integrating by parts.

**Example.** $u \in H^1_0(\Omega)$ is a weak solution of $-\Delta u = f$ if

$$
\int_\Omega \nabla u \cdot \nabla \varphi \, dx \;=\; \int_\Omega f \, \varphi \, dx \qquad \forall \varphi \in C_c^\infty(\Omega).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why use them)</span></p>

* The natural energy space is a **Hilbert space**, so existence follows from Lax–Milgram or direct minimization.
* You do not need $u$ to be twice differentiable a priori.
* **Regularity theory** then upgrades the weak solution to a classical one whenever the data is nice enough.

</div>

---

## Linear operators

### Bounded linear operator

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(bounded linear operator)</span></p>

A linear map $T : X \to Y$ between normed spaces is **bounded** if its **operator norm**

$$
\lVert T \rVert \;=\; \sup_{\lVert x\rVert_X \le 1} \lVert T x \rVert_Y \;=\; \sup_{x \ne 0} \frac{\lVert Tx\rVert_Y}{\lVert x\rVert_X}
$$

is finite. The space of bounded linear operators $X \to Y$ is denoted $\mathcal{B}(X, Y)$ (or $\mathcal{L}(X, Y)$), with $\mathcal{B}(X) := \mathcal{B}(X, X)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(bounded = continuous, the linearity miracle)</span></p>

* For linear maps between normed spaces, **bounded $\iff$ continuous $\iff$ continuous at $0$**. This equivalence is specific to linearity — generic continuous functions can be unbounded.
* $\mathcal{B}(X, Y)$ is a normed space; if $Y$ is **Banach**, so is $\mathcal{B}(X, Y)$. In particular $X^\ast = \mathcal{B}(X, \mathbb{K})$ is always Banach (the scalar field is complete).
* **Unbounded operators** (differentiation on $L^2$, multiplication by $x$ on $L^2(\mathbb{R})$) are not continuous and are only densely defined on a domain $D(T) \subsetneq X$ — the whole theory of unbounded self-adjoint operators (spectral theorem, semigroups, Stone's theorem) is the price you pay for working with PDEs.
* The three Banach-space pillars of $\mathcal{B}(X, Y)$ — **uniform boundedness, open mapping, closed graph** — all follow from Baire category and depend on $X$, $Y$ being Banach.

</div>

### Diagonal operator

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(diagonal operator)</span></p>

On $\ell^p$ (or any sequence space with the standard basis $\lbrace e_n\rbrace$) and a scalar sequence $\lambda = (\lambda_n)\_{n \in \mathbb{N}}$, the **diagonal operator** $T_\lambda$ is the coordinatewise multiplication

$$
T_\lambda : (x_n) \;\longmapsto\; (\lambda_n x_n), \qquad T_\lambda e_n = \lambda_n e_n.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(the universal model of self-adjoint operators)</span></p>

* $T_\lambda$ is bounded on $\ell^p$ iff $(\lambda_n) \in \ell^\infty$, and then $\lVert T_\lambda\rVert = \sup_n \lvert \lambda_n\rvert$.
* Spectrum: $\sigma(T_\lambda) = \overline{\lbrace \lambda_n\rbrace}$. The point spectrum is $\lbrace \lambda_n\rbrace$, with $e_n$ as the eigenvector for $\lambda_n$.
* **Operator-ideal characterization:** $T_\lambda$ is compact iff $\lambda_n \to 0$; Hilbert–Schmidt iff $(\lambda_n) \in \ell^2$; trace class iff $(\lambda_n) \in \ell^1$.
* The **spectral theorem** for a self-adjoint bounded operator $A$ on a Hilbert space says exactly: $A$ is **unitarily equivalent to a multiplication-by-$\lambda(\cdot)$ operator** on $L^2$ of its spectrum (the "diagonalization"). So "diagonal operator on a basis" is the canonical model of self-adjointness.
* Fourier multipliers $T_m$ (see below) are diagonal operators on the Fourier basis: they multiply each frequency component $\widehat f(\xi)$ by $m(\xi)$.

</div>

---

## Fourier and harmonic analysis

### Fourier analysis (overview)

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name">(what Fourier analysis is)</span></p>

The general project: decompose a function $f$ as a superposition of **simple waves** (eigenfunctions of translation / differentiation) and study $f$ via its **frequency content** $\widehat f$.

By domain — the three standard incarnations:

| Domain | Decomposition | Frequency index |
|---|---|---|
| $\mathbb{T}^n = \mathbb{R}^n / \mathbb{Z}^n$ (compact, periodic) | **Fourier series** | discrete $k \in \mathbb{Z}^n$ |
| $\mathbb{R}^n$ (non-compact) | **Fourier transform** | continuous $\xi \in \mathbb{R}^n$ |
| $\mathbb{Z}_N$ (finite cyclic) | **Discrete Fourier transform** (DFT) | finite $k \in \mathbb{Z}_N$ |

In each case the transform **diagonalizes the translation operator** $T_a f(x) = f(x - a)$, and hence diagonalizes every constant-coefficient differential operator. On the Fourier side, $\partial_j$ becomes "multiply by $2\pi i \xi_j$" — differential equations turn into algebraic ones.

</div>

### Fourier coefficients

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fourier coefficients on $\mathbb{T}^n$)</span></p>

For $f \in L^1(\mathbb{T}^n)$ (period $2\pi$ in each variable),

$$
\widehat f(k) \;=\; \frac{1}{(2\pi)^n} \int_{\mathbb{T}^n} f(x) \, e^{-i k \cdot x} \, dx, \qquad k \in \mathbb{Z}^n.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(facts — boundedness, Plancherel, smoothness ↔ decay)</span></p>

* $\lvert \widehat f(k)\rvert \le \lVert f\rVert_{L^1}$, so $\widehat f \in \ell^\infty$.
* **Riemann–Lebesgue lemma**: $\widehat f(k) \to 0$ as $\lvert k\rvert \to \infty$ for $f \in L^1$.
* **Plancherel / Parseval**: $f \in L^2(\mathbb{T}^n) \iff (\widehat f(k)) \in \ell^2(\mathbb{Z}^n)$, with $\lVert f\rVert_{L^2}^2 = \sum_k \lvert \widehat f(k)\rvert^2$.
* **Smoothness $\iff$ decay**: $f \in C^\infty(\mathbb{T}^n)$ iff $\widehat f(k)$ decays **faster than any polynomial** in $\lvert k\rvert$. Sobolev $H^s$ is exactly $\sum_k (1 + \lvert k\rvert^2)^s \lvert \widehat f(k)\rvert^2 < \infty$ — Sobolev spaces on the torus are *defined* via Fourier decay.

</div>

### Fourier series

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fourier series)</span></p>

For $f \in L^2(\mathbb{T}^n)$,

$$
f(x) \;=\; \sum_{k \in \mathbb{Z}^n} \widehat f(k) \, e^{i k \cdot x},
$$

with convergence in the $L^2$ norm. The partial sums $S_N f(x) = \sum_{\lvert k\rvert \le N} \widehat f(k) e^{ik\cdot x}$ are the orthogonal projections onto trig polynomials of degree $\le N$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(modes of convergence — Carleson, Hunt, Kolmogorov)</span></p>

* In $L^2$: $S_N f \to f$ **always** (Parseval / Riesz–Fischer applied to the o.n. basis $\lbrace (2\pi)^{-n/2} e^{ik\cdot x}\rbrace$).
* In $L^p$ for $1 < p < \infty$: $S_N f \to f$ in $L^p$ — this is a non-trivial theorem (M. Riesz, via boundedness of the Hilbert transform / conjugate function on $L^p$). **Fails at $p = 1$ and $p = \infty$.**
* **Pointwise**: Carleson (1966) — $S_N f \to f$ a.e. for $f \in L^2(\mathbb{T})$. Hunt (1968) extends to $L^p$ with $1 < p \le \infty$. **Kolmogorov (1923)**: there exists $f \in L^1$ with $S_N f \to \infty$ everywhere.
* **Uniform** convergence requires absolute summability $\sum \lvert \widehat f(k)\rvert < \infty$ (e.g. $f \in C^{1+\varepsilon}$ on $\mathbb{T}$ via Bernstein).

</div>

### Fourier decomposition

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name">(Fourier decomposition — the umbrella term)</span></p>

"Fourier decomposition" is a loose synonym for **expanding a function in an orthonormal basis of eigenfunctions of a self-adjoint operator** — usually the Laplacian. The three concrete cases above (series, transform, DFT) are special cases corresponding to $-\Delta$ on $\mathbb{T}^n$, $\mathbb{R}^n$, $\mathbb{Z}_N^n$ respectively. The eigenfunctions are the **characters of the underlying abelian group**: $e^{ik\cdot x}$ on $\mathbb{T}^n$, $e^{i\xi \cdot x}$ on $\mathbb{R}^n$, $e^{2\pi i kn/N}$ on $\mathbb{Z}_N$.

Generalizations under the same umbrella:

* Expansions in **spherical harmonics** on $S^{n-1}$ (eigenfunctions of $-\Delta_{S^{n-1}}$).
* **Hermite expansions** on $\mathbb{R}^n$ (eigenfunctions of the quantum harmonic oscillator $-\Delta + \lvert x\rvert^2$).
* **Wavelet decomposition** — replaces frequency with frequency $\times$ location.
* **Peter–Weyl** on a compact Lie group, **Pontryagin duality** on a locally compact abelian group.

In each case the "Fourier transform" is the change of basis from the spatial side to the spectral side, and it diagonalizes the operator the basis comes from.

</div>

### Fourier transform

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fourier transform on $\mathbb{R}^n$)</span></p>

For $f \in L^1(\mathbb{R}^n)$,

$$
\widehat f(\xi) \;=\; \int_{\mathbb{R}^n} f(x) \, e^{-2\pi i \xi \cdot x} \, dx \qquad (\xi \in \mathbb{R}^n),
$$

with formal inverse $f(x) = \int_{\mathbb{R}^n} \widehat f(\xi) e^{+2\pi i \xi \cdot x} \, d\xi$. (Conventions vary; the factor $e^{-i\xi \cdot x}$ with normalization $(2\pi)^{-n/2}$ or $(2\pi)^{-n}$ is also standard.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Plancherel, Schwartz, tempered distributions)</span></p>

* **Hausdorff–Young**: $\widehat f \in L^q$ for $f \in L^p$, $1 \le p \le 2$, $\tfrac{1}{p} + \tfrac{1}{q} = 1$, with $\lVert \widehat f\rVert_q \le \lVert f\rVert_p$. Does **not** extend to $p > 2$.
* **Plancherel**: $\mathcal{F}$ extends uniquely from $L^1 \cap L^2$ to an **isometric isomorphism** $L^2(\mathbb{R}^n) \to L^2(\mathbb{R}^n)$.
* **Diagonalization of differentiation**: 
  * $\widehat{\partial_j f}(\xi) = 2\pi i \xi_j \widehat f(\xi)$, 
  * $\widehat{x_j f}(\xi) = -\tfrac{1}{2\pi i} \partial_{\xi_j} \widehat f(\xi)$. 
  * So $\mathcal{F}$ swaps differentiation $\leftrightarrow$ multiplication-by-coordinate.
* **Schwartz space $\mathcal{S}$**: rapidly decreasing $C^\infty$ functions. $\mathcal{F} : \mathcal{S} \to \mathcal{S}$ is a topological isomorphism with $\mathcal{F}^4 = \mathrm{id}$ (with the symmetric convention). By duality, $\mathcal{F}$ acts on **tempered distributions** $\mathcal{S}'$ — so $\widehat{\delta} = 1$, $\widehat{1} = \delta$, $\widehat{e^{2\pi i a \cdot x}} = \delta_a$, etc.

</div>

### Discrete Fourier transform (DFT)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(DFT)</span></p>

For $f = (f_0, f_1, \dots, f_{N-1}) \in \mathbb{C}^N$,

$$
\widehat f(k) \;=\; \sum_{n=0}^{N-1} f_n \, e^{-2\pi i k n / N}, \qquad k = 0, 1, \dots, N-1,
$$

with inverse $f_n = \tfrac{1}{N} \sum_{k=0}^{N-1} \widehat f(k) e^{+2\pi i k n / N}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role — change of basis and the FFT)</span></p>

* The DFT is the change of basis on $\mathbb{C}^N$ from the standard basis to the basis of **characters** $\chi_k(n) = e^{2\pi i k n / N}$ of the cyclic group $\mathbb{Z}\_N$. Up to $1/\sqrt N$ scaling it is a **unitary** matrix.
* The **Fast Fourier Transform (FFT)** computes the DFT in $O(N \log N)$ operations (Cooley–Tukey, 1965). This is what makes Fourier methods practical for signals, polynomial multiplication, and numerical PDE on periodic grids.
* Same algebraic structure as the continuous Fourier transform: $\widehat{f \ast g} = \widehat f \cdot \widehat g$ where $\ast$ is **cyclic convolution** on $\mathbb{Z}\_N$.
* Discretization of a periodic signal $f$ on $[0, T]$ at $N$ equispaced samples gives the DFT as a numerical approximation to its Fourier series — with the famous **aliasing** caveat: frequencies above the **Nyquist** $N/2$ alias to lower frequencies.

</div>

### Fourier multipliers

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fourier multiplier)</span></p>

A **Fourier multiplier** is a linear operator $T_m$ defined on a suitable function space by

$$
\widehat{T_m f}(\xi) \;=\; m(\xi) \, \widehat f(\xi),
$$

for a fixed function $m : \mathbb{R}^n \to \mathbb{C}$ called the **symbol** (or the **multiplier function**).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(canonical examples and boundedness on $L^p$)</span></p>

* **Constant-coefficient differential operators are multipliers**: $\partial_j \leftrightarrow 2\pi i \xi_j$, $-\Delta \leftrightarrow 4\pi^2 \lvert \xi\rvert^2$, the fractional Laplacian $(-\Delta)^{s/2} \leftrightarrow (2\pi \lvert \xi\rvert)^s$.
* **Hilbert transform** on $\mathbb{R}$: $m(\xi) = -i \, \mathrm{sgn}(\xi)$. **Riesz transforms** on $\mathbb{R}^n$: $m_j(\xi) = -i \xi_j / \lvert \xi\rvert$. These are the building blocks of Calderón–Zygmund theory.
* **$L^2$ boundedness** is trivial: $T_m$ is bounded on $L^2$ iff $m \in L^\infty$, with $\lVert T_m\rVert_{L^2 \to L^2} = \lVert m\rVert_{L^\infty}$.
* **$L^p$ boundedness ($p \ne 2$) is subtle**: Marcinkiewicz / Mihlin / Hörmander multiplier theorems give sufficient smoothness/decay conditions on $m$ (e.g. $\lvert \partial^\alpha m(\xi)\rvert \lesssim \lvert \xi\rvert^{-\lvert \alpha\rvert}$ for $\lvert \alpha\rvert \le \lfloor n/2\rfloor + 1$).
* **Pseudo-differential operators** generalize multipliers to $m = m(x, \xi)$ — symbols depending on both space and frequency — and form the variable-coefficient counterpart used in microlocal analysis.

</div>

### Convolution

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(convolution)</span></p>

For $f, g$ measurable on $\mathbb{R}^n$ (whenever the integral exists),

$$
(f \ast g)(x) \;=\; \int_{\mathbb{R}^n} f(y) \, g(x - y) \, dy \;=\; \int_{\mathbb{R}^n} f(x - y) \, g(y) \, dy.
$$

Commutative, associative, distributive over addition.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role — smoothing, Banach algebra, translation invariance)</span></p>

* **Existence**: Young's inequality $\lVert f \ast g\rVert_{L^r} \le \lVert f\rVert_{L^p} \lVert g\rVert_{L^q}$ for $\tfrac{1}{p} + \tfrac{1}{q} = 1 + \tfrac{1}{r}$ — in particular $L^1 \ast L^p \subseteq L^p$ for every $p$.
* **$L^1$ is a Banach algebra under $\ast$**, but with **no unit**: the would-be identity $\delta$ lives in $\mathcal{D}'$, not in $L^1$.
* **Smoothing**: $f \ast g$ inherits the regularity of the smoother factor. Convolving with a mollifier $\rho_\varepsilon \in C_c^\infty$ produces $f_\varepsilon = f \ast \rho_\varepsilon \in C^\infty$ approximating $f$ in $L^p$ — this is the standard tool for density of $C_c^\infty$ in $L^p$, and for regularization of weak solutions.
* **Translation invariance**: $T_a (f \ast g) = (T_a f) \ast g$, and conversely the bounded translation-invariant operators on $L^p(\mathbb{R}^n)$ are **exactly the Fourier multipliers**, which by the convolution theorem are exactly convolution-with-a-distribution operators. So
    
    $$
    \text{convolution operators} \;=\; \text{translation-invariant operators} \;=\; \text{Fourier multipliers}.
    $$
* Periodic / discrete versions: convolution on $\mathbb{T}^n$ (integrate over the torus), cyclic convolution on $\mathbb{Z}\_N$, convolution on any locally compact group.

</div>

### Convolution theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(convolution theorem)</span></p>

For $f, g \in L^1(\mathbb{R}^n)$,

$$
\widehat{f \ast g} \;=\; \widehat f \cdot \widehat g.
$$

Conversely, under appropriate hypotheses (e.g. $f, g \in L^1 \cap L^2$),

$$
\widehat{f \cdot g} \;=\; \widehat f \ast \widehat g.
$$

The same identity holds in the periodic, discrete, and tempered-distribution settings (replacing $\ast$ by the appropriate convolution and $\widehat{\cdot}$ by the corresponding Fourier map).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why this single identity does so much work)</span></p>

* **It is the reason Fourier analysis is useful for PDE / signal processing**: a constant-coefficient linear PDE $L u = f$ becomes $\widehat L(\xi) \widehat u(\xi) = \widehat f(\xi)$, solved algebraically as $\widehat u = \widehat f / \widehat L$, and $u = \mathcal{F}^{-1}(\widehat f / \widehat L) = f \ast G$ where $G = \mathcal{F}^{-1}(1/\widehat L)$ is the **fundamental solution**.
* It identifies convolution operators with Fourier multipliers (see the [[Fourier-multipliers entry]]).
* The product side is what makes the **uncertainty principle** clean: localization of $f$ implies spreading of $\widehat f$, because pointwise products on one side correspond to convolutions on the other.

</div>

### Deconvolution

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(deconvolution problem)</span></p>

Given an observed signal $g = f \ast k$ and a **known** kernel $k$, the **deconvolution problem** is to recover $f$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(ill-posedness and regularization)</span></p>

* On the Fourier side the problem is algebraic: $\widehat g = \widehat f \cdot \widehat k$ gives $\widehat f = \widehat g / \widehat k$. But division blows up at zeros of $\widehat k$, and is unstable wherever $\widehat k$ is small.
* Smoothing kernels (Gaussian, mollifier, box average) have **rapidly decaying $\widehat k$** — so naive inversion massively amplifies high frequencies, and any noise in $g$ is amplified catastrophically. This is the **archetypal ill-posed inverse problem**.
* Standard fixes:
    * **Tikhonov regularization**: replace $\widehat g / \widehat k$ by $\overline{\widehat k}\, \widehat g / (\lvert \widehat k\rvert^2 + \alpha)$ for small $\alpha > 0$. Trades bias for stability.
    * **Wiener filter**: chooses $\alpha = \alpha(\xi)$ optimally from the signal-to-noise power spectrum — minimum mean-square-error linear estimator.
    * **Sparsity / total-variation priors**: replace least-squares fit by $\arg\min \tfrac{1}{2}\lVert k \ast f - g\rVert^2 + \lambda \lVert f\rVert_{\mathrm{TV}}$ (or $\ell^1$ on a wavelet basis) — modern image deblurring, compressed sensing.
* Application domains: image deblurring, seismic inversion, microscopy, single-particle cryo-EM, channel equalization in communications.

</div>

---

## Miscellaneous tools

### Universal property

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(universal property)</span></p>

A category-theoretic specification of an object by **how it interacts with every other object via maps**, not by an internal construction. Concretely: an object $U$ together with a "universal arrow" such that every morphism of a prescribed type factors **uniquely** through $U$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(tensor product)</span></p>

The tensor product $V \otimes W$ is defined as the universal object through which every bilinear map $V \times W \to Z$ factors uniquely as a linear map $V \otimes W \to Z$. Whatever construction you pick (formal symbols modulo relations, free abelian group quotient, ...) you get the same object up to canonical isomorphism — that is the *point* of the universal property.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why care)</span></p>

* Uniqueness up to canonical isomorphism is automatic.
* Morphisms out of (or into) the universal object are free — you just exploit universality.
* It is the right way to formalize "gluing / quotienting / completing in the canonical way."

</div>

### Spectral theory

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(spectrum)</span></p>

For a bounded operator $T : X \to X$ on a Banach space,

$$
\sigma(T) = \lbrace \lambda \in \mathbb{C} : T - \lambda I \text{ is not invertible} \rbrace.
$$

It splits into:

* **Point spectrum** $\sigma_p(T)$: $\lambda$ is an eigenvalue.
* **Continuous spectrum** $\sigma_c(T)$: $T - \lambda I$ is injective with dense range, but not surjective.
* **Residual spectrum** $\sigma_r(T)$: $T - \lambda I$ is injective but the range is not dense.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(spectral theorem, self-adjoint case)</span></p>

Every self-adjoint operator $T$ on a Hilbert space $H$ is **unitarily equivalent to a multiplication operator** $f \mapsto m \cdot f$ on some $L^2(X, \mu)$, or equivalently

$$
T = \int_{\mathbb{R}} \lambda \, dE(\lambda)
$$

for a projection-valued measure $E$ on $\mathbb{R}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why it matters)</span></p>

Generalizes "diagonalize the matrix" to infinite dimensions. This is what makes **functions of operators** $f(T) = \int f(\lambda) \, dE(\lambda)$ well-defined, and it underlies quantum mechanics, semigroup theory, and the analytic side of PDE.

</div>

## Sort it out

### Minimizing Sequences

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Minimizing Sequence)</span></p>

A sequence of admissible functions $\lbrace y\_n \rbrace$ is called a **minimizing sequence** for the functional $J[y]$ if $\lim\_{n \to \infty} J[y\_n] = \mu$, where $\mu = \inf\_y J[y]$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(If Minimizing Sequence Converges)</span></p>

If the minimizing sequence $\lbrace y\_n \rbrace$ converges to a limit function $\hat{y}$, and if interchanging the functional and the limit is justified, then

$$J[\hat{y}] = \lim_{n \to \infty} J[y_n] = \mu,$$

and $\hat{y}$ solves the variational problem. The functions of the minimizing sequence serve as approximate solutions.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Solving variational problems with minimizing sequences (called direct method))</span></p>

Thus, solving a variational problem by the direct method requires three steps:

1. Construct a minimizing sequence $\lbrace y\_n \rbrace$.
2. Show that $\lbrace y\_n \rbrace$ has a limit function $\hat{y}$.
3. Justify passing the functional through the limit.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lower Semicontinuity and Minimizing Sequences)</span></p>

Even when a minimizing sequence does converge in the $\mathscr{C}$-norm (i.e., uniformly), passing the functional through the limit is nontrivial because typical functionals in the calculus of variations are not continuous in the $\mathscr{C}$-norm. However, the interchange is still valid under a weaker condition: it suffices for $J[y]$ to be **lower semicontinuous**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lower Semicontinuity and Minimizing Sequences)</span></p>

If $\lbrace y\_n \rbrace$ is a minimizing sequence of the functional $J[y]$ with limit function $\hat{y}$, and if $J[y]$ is lower semicontinuous at $\hat{y}$, then

$$J[\hat{y}] = \lim_{n \to \infty} J[y_n].$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Sketch of proof</summary>

On one hand, $J[\hat{y}] \geqslant \lim\_{n\to\infty} J[y\_n] = \inf J[y]$, since $\hat{y}$ is admissible. On the other hand, for every $\varepsilon > 0$ and $n$ large enough, lower semicontinuity gives $J[y\_n] - J[\hat{y}] > -\varepsilon$, so letting $n \to \infty$ yields $J[\hat{y}] \leqslant \lim\_{n\to\infty} J[y\_n]$. Combining both inequalities gives equality.

</details>
</div>


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Spaces of Measures: OTOaN)</span></p>

* $P(\mathcal{X})$ — the space of Borel probability measures on $\mathcal{X}$.
* $M\_+(\mathcal{X})$ — the space of finite Borel measures.
* $M(\mathcal{X})$ — the space of signed finite Borel measures.
* The **total variation** of $\mu$ is denoted by $\lVert \mu \rVert\_{\mathrm{TV}}$.
* The integral of a function $f$ with respect to a probability measure $\mu$ is denoted interchangeably by $\int f(x) \, d\mu(x)$ or $\int f(x) \, \mu(dx)$ or $\int f \, d\mu$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weak Topology)</span></p>

The **weak topology** on $P(\mathcal{X})$ (or topology of weak convergence, or narrow topology) is induced by convergence against $C\_b(\mathcal{X})$, i.e. bounded continuous test functions. If $\mathcal{X}$ is Polish, then $P(\mathcal{X})$ itself is Polish. Unless explicitly stated, the weak-$\ast$ topology of measures (induced by $C\_0(\mathcal{X})$ or $C\_c(\mathcal{X})$) is not used.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Negligible Sets and Support)</span></p>

* If $\mu$ is a Borel measure on a topological space $\mathcal{X}$, a set $N$ is said to be **$\mu$-negligible** if $N$ is included in a Borel set of zero $\mu$-measure. Then $\mu$ is said to be **concentrated** on a set $C$ if $\mathcal{X} \setminus C$ is negligible (equivalently $\mu[\mathcal{X} \setminus C] = 0$).
* If $\mu$ is a Borel measure, its **support** $\mathrm{Spt}\, \mu$ is the smallest *closed* set on which it is concentrated.

</div>

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



---

In a Banach space, weak convergence is a mode of convergence where a sequence stabilizes under the action of all continuous linear functionals. This concept is central to functional analysis and partial differential equations, bypassing the rigid restrictions of norm topology in infinite dimensions. [1, 2, 3, 4, 5] 
## 1. Mathematical Definition
Let $X$ be a real or complex Banach space and $X^*$ its continuous dual space. A sequence $(x_n)_{n=1}^\infty$ in $X$ converges weakly to an element $x \in X$ (denoted $x_n \rightharpoonup x$) if: [1, 6, 7, 8, 9] 
$$\lim_{n \to \infty} f(x_n) = f(x) \quad \text{for all } f \in X^*$$ 
This indicates convergence in the [weak topology](https://en.wikipedia.org/wiki/Weak_topology) of $X$, which is the coarsest topology rendering all elements of $X^*$ continuous. [10, 11] 
## 2. Weak vs. Strong Convergence

* Strong Convergence: Implies $\lim_{n\to\infty} \Vert{}x_n - x\Vert{}_X = 0$.
* Implication: Strong convergence always forces weak convergence.
* The Converse: Fails in infinite-dimensional spaces. For example, the standard orthonormal basis $(e_n)$ in the Hilbert space $\ell^2$ satisfies $e_n \rightharpoonup 0$ but $\Vert{}e_n\Vert{} = 1$.
* Equivalence: Strong and weak convergence coincide if and only if $X$ is finite-dimensional. [1, 12, 13, 14, 15, 16, 17, 18] 

## 3. Fundamental Properties

* Uniqueness: The weak limit of a sequence is unique.
* Boundedness: Every weakly convergent sequence is bounded in norm: $\sup_n \Vert{}x_n\Vert{} < \infty$.
* Norm Lower Semicontinuity: The limit satisfies the inequality $\Vert{}x\Vert{} \le \liminf_{n \to \infty} \Vert{}x_n\Vert{}$.
* Linear Operators: If $T: X \to Y$ is a bounded linear operator and $x_n \rightharpoonup x$, then $T x_n \rightharpoonup T x$ in $Y$.
* Compact Operators: If $T: X \to Y$ is a compact operator and $x_n \rightharpoonup x$, then $T x_n \to T x$ strongly in $Y$. [11, 19, 20, 21, 22] 

## 4. Key Theorems## Mazur's Lemma
If $x_n \rightharpoonup x$, there exists a sequence of convex combinations $y_n = \sum_{i=1}^{N(n)} \alpha_{n,i} x_i$ (where $\sum_i \alpha_{n,i} = 1, \alpha_{n,i} \ge 0$) that converges strongly to $x$. As a corollary, any norm-closed, convex set is weakly closed. [23, 24] 
## Eberlein–Šmulyan Theorem
A subset of a Banach space is weakly compact if and only if it is weakly sequentially compact. [25, 26] 
## Banach–Alaoglu Theorem & Reflexivity
The closed unit ball of $X$ is weakly compact if and only if $X$ is a reflexive Banach space (i.e., $X \cong X^{**}$). Consequently, every bounded sequence in a reflexive space contains a weakly convergent subsequence. [3, 11, 27, 28, 29] 
## 5. Special Space Phenomena

* Schur's Property: In $\ell^1$, every weakly convergent sequence converges in norm.
* Radon–Riesz Property: In uniformly convex Banach spaces (such as Hilbert spaces or $L^p$ for $1 < p < \infty$), if $x_n \rightharpoonup x$ and $\Vert{}x_n\Vert{} \to \Vert{}x\Vert{}$, then $x_n \to x$ strongly. [6, 13, 30] 

If you would like to explore deeper, please share:

* A specific Banach space you are studying (e.g., $L^p$, $C[a,b]$, $\ell^\infty$)
* Whether you need to analyze weak vs. weak- convergence*
* A particular problem or proof context you are trying to resolve [19, 31, 32] 

---

We care about weak topologies because they solve a fundamental crisis in infinite-dimensional analysis: the failure of compactness in the standard norm topology. By removing open sets, weak topologies make it easier for sequences to converge and for sets to be compact, which is the cornerstone for solving Partial Differential Equations (PDEs), optimization problems, and quantum mechanics. [1, 2, 3, 4] 
------------------------------
## 1. Regaining Compactness (The Biggest Reason)

* The Problem: In infinite-dimensional spaces (like function spaces), the standard norm topology is "too large" (has too many open sets). As a result, the closed unit ball is never compact. Without compactness, we cannot guarantee that bounded sequences have convergent subsequences.
* The Solution: Weak topologies strip away open sets, making it easier for sets to become compact. Under the [Banach-Alaoglu Theorem](https://en.wikipedia.org/wiki/Weak_topology), the closed unit ball becomes compact in the weak* topology. [2, 5, 6, 7, 8] 

## 2. Solving Optimization and PDEs (The Direct Method)
Weak topologies power the Direct Method in the Calculus of Variations to prove that solutions to physical equations actually exist: [1, 9, 10] 

   1. Minimize: You want to minimize an energy functional (e.g., finding the shape of a soap film or a physical system at rest).
   2. Bound: You generate a "minimizing sequence" of functions and prove they are bounded in norm.
   3. Converge: Because the norm ball is not compact, you cannot extract a norm-convergent subsequence. You switch to the weak topology, where the Banach-Alaoglu Theorem guarantees a weakly convergent subsequence.
   4. Solve: You prove this weak limit is the actual solution to your PDE. [1, 2, 5, 6, 11] 

## 3. Modeling Real-World Convergence Modes
Standard norm topologies often fail to capture how functions behave in practice. Weak topologies naturally represent essential mathematical and physical modes of convergence: [2] 

* Pointwise Convergence: The topology of pointwise convergence is fundamentally a weak topology.
* Quantum Mechanics: Physical observables are linear functionals. We cannot physically measure "norm convergence" of a quantum state, but we can measure weak convergence, which represents the convergence of expected measurement outcomes.
* Probability Theory: The weak* topology allows for the "weak convergence of measures" (like the [Prokhorov's Theorem](https://terrytao.wordpress.com/2009/02/21/245b-notes-11-the-strong-and-weak-topologies/)), allowing mathematicians to talk about probability distributions approaching a limit. [2, 12, 13, 14, 15] 

## 4. Preservation of Convex Geometry
Even though weak topologies have fewer open sets than norm topologies, they perfectly preserve the structure of closed convex sets. By the Hahn-Banach Theorem, a convex set is closed in the norm topology if and only if it is closed in the weak topology. This allows mathematicians to switch effortlessly between the geometric convenience of norms and the compactness of weak topologies. [2, 16] 
------------------------------
To help visualize how weak topologies relate to other structures in functional analysis, here is a quick summary:

| Topology Feature [2, 5, 6, 7, 8, 17] | Norm (Strong) Topology | Weak / Weak* Topology |
|---|---|---|
| Number of Open Sets | Massive (Harder to be compact) | Minimal (Easier to be compact) |
| Unit Ball Compactness | Only if finite-dimensional | Always Compact (via Banach-Alaoglu) |
| Convergence Requirement | Elements must get physically close in distance | Projections/measurements must converge |

If you'd like, I can walk you through a concrete example (like why the shifting wave $f_n(x) = \sin(nx)$ converges weakly to $0$ but doesn't converge in norm) or explain the exact difference between weak and weak* topologies. Turn the page by letting me know what you'd like to explore next!

[1] [https://mathoverflow.net](https://mathoverflow.net/questions/239648/applications-of-the-weak-and-weak-topologies-to-pdes)
[2] [https://terrytao.wordpress.com](https://terrytao.wordpress.com/2009/02/21/245b-notes-11-the-strong-and-weak-topologies/comment-page-1/)
[3] [https://link.springer.com](https://link.springer.com/content/pdf/10.1007/978-3-031-81791-5_6.pdf)
[4] [https://www.vaia.com](https://www.vaia.com/en-us/textbooks/math/functional-analysis-a-first-course-1-edition/chapter-8/problem-12-let-x-be-a-normed-linear-space-show-that-the-weak/)
[5] [https://www.reddit.com](https://www.reddit.com/r/math/comments/1ch2hiq/good_expository_resource_to_build_intuition_on/)
[6] [https://www.reddit.com](https://www.reddit.com/r/math/comments/127d5t/weak_topology/)
[7] [https://www.math.uni-bonn.de](https://www.math.uni-bonn.de/ag/ana/SoSe2025/V5B1/WeakTopology.pdf)
[8] [https://en.wikipedia.org](https://en.wikipedia.org/wiki/Weak_topology)
[9] [https://arxiv.org](https://arxiv.org/pdf/1601.00390)
[10] [https://math.stackexchange.com](https://math.stackexchange.com/questions/4503837/why-find-weak-solutions-of-pde-and-how-to-go-from-weak-to-strong)
[11] [https://www.latimes.com](https://www.latimes.com/archives/la-xpm-1988-11-24-vw-209-story.html)
[12] [https://perso.crans.org](https://perso.crans.org/lecomte/Math/WeakTopologies.pdf)
[13] [https://math.stackexchange.com](https://math.stackexchange.com/questions/410698/how-is-the-weak-star-topology-useful)
[14] [https://terrytao.wordpress.com](https://terrytao.wordpress.com/2009/02/21/245b-notes-11-the-strong-and-weak-topologies/)
[15] [https://link.springer.com](https://link.springer.com/chapter/10.1007/978-3-031-97239-3_10)
[16] [https://www.sciencedirect.com](https://www.sciencedirect.com/topics/mathematics/weak-topology-sigma)
[17] [https://math.stackexchange.com](https://math.stackexchange.com/questions/4290243/some-intuition-about-weak-topology)

---

Is topology induced by a metric?

The short answer is no, not all topologies are induced by a metric.
While every metric space naturally generates a topology, many critical topologies in advanced mathematics—including most weak topologies on infinite-dimensional spaces—cannot be produced by any metric.
------------------------------
## 1. The Core Split: Metrizable vs. Non-Metrizable

* Metrizable Topologies: A topology is called "metrizable" if you can define a distance function (metric) $d(x,y)$ that perfectly generates all the open sets of that topology. Examples include Euclidean space $\mathbb{R}^n$ and Hilbert spaces.
* Non-Metrizable Topologies: These are spaces where open sets are defined purely by set relationships, and it is mathematically impossible to construct a distance function that matches those open sets.

## 2. Why Weak Topologies Are Usually Not Metrizable
If an infinite-dimensional space has a weak topology, it is never metrizable. This creates a striking paradox:

* The Problem with Open Sets: In a metric space, every point has a tiny "ball" around it ($B_\epsilon(x)$) that fits inside an open set.
* The Weak Topology Reality: In an infinite-dimensional weak topology, every open neighborhood of the origin is "huge"—it must contain an entire infinite-dimensional closed subspace. Because it contains straight lines extending to infinity, you can never fit a bounded metric ball inside it.

## 3. The Sequential Loophole (Separable Spaces)
While the full weak topology on an infinite-dimensional space is not metrizable, mathematicians use a vital workaround when dealing with separable spaces (spaces with a countable dense subset):

* The Weak Compact Ball:* If a normed space is separable, the [Banach-Alaoglu Theorem](https://en.wikipedia.org/wiki/Weak_topology) guarantees that the closed unit ball is compact and metrizable in the weak* topology.
* Why this matters: This allows you to use regular sequences ($x_n \to x$) to solve PDEs and optimization problems on the unit ball, rather than abstract topological nets.

## 4. How to Test if a Topology is Metrizable
Topologists use strict structural properties to determine if a space can handle a metric. A topology cannot be induced by a metric if it fails any of these conditions:

* First Countability: Every point must have a countable base of neighborhoods (e.g., balls of radius $1/n$). Many weak topologies lack this.
* Hausdorff Property ($T_2$): Distinct points must have disjoint open neighborhoods. (Some weak topologies fail this if the linear functionals cannot separate points).
* Urysohn's Metrization Theorem: A topological space is metrizable if and only if it is regular ($T_3$) and has a countable base of open sets.

------------------------------
To help clarify how metrics and topologies interact, let us look at the hierarchy of spaces:

[ All Topological Spaces ]
         │
         ▼
[ Hausdorff Spaces (Points can be separated) ]
         │
         ▼
[ Metrizable Spaces (Can be given a metric d(x,y)) ]
         │
         ▼
[ Normed Vector Spaces (Can be given a length ||x||) ]

To help you apply this to your work, let me know if you would like to:

* See the proof of why an infinite-dimensional weak neighborhood must contain a line.
* Explore how weak convergence is turned into a metric on the unit ball using a countable dense set.
* Understand the difference between sequences and nets in non-metrizable spaces.

---

Give an example of Non-Metrizable Topologies

The most famous and intuitive examples of non-metrizable topologies come from standard calculus, set theory, and functional analysis.
------------------------------
## 1. The Space of All Real Functions ($\mathbb{R}^{\mathbb{R}}$) [1] 
This is the space of all functions $f: \mathbb{R} \to \mathbb{R}$ equipped with the topology of pointwise convergence.

* The Open Sets: A sequence of functions $f_n$ converges to $f$ if and only if $f_n(x) \to f(x)$ for every individual point $x$.
* Why it fails a metric: To track pointwise convergence across uncountably many points, you would need uncountably many independent distance constraints. A metric can only compress distances into a single real number, which fails to capture this uncountably infinite freedom. It lacks a [countable neighborhood base](https://en.wikipedia.org/wiki/Weak_topology) (violates First Countability). [2, 3, 4] 

## 2. The Weak Topology on $\ell^2$ (Infinite-Dimensional Hilbert Space)
Take the space of all square-summable sequences $\ell^2$ (like $(1, \frac{1}{2}, \frac{1}{4}, \dots)$) under its weak topology.

* The Open Sets: Open sets are defined by testing sequences against linear functionals. A neighborhood of zero looks like an infinite cylinder that is restricted in a few directions but completely open and flat in infinitely many other directions.
* Why it fails a metric: Every open neighborhood of the origin contains a full, infinitely long straight line (a closed subspace). In a metric space, if you move far enough down a line, you must eventually leave a bounded metric ball. Because weak open neighborhoods contain entire lines, no bounded metric ball can ever fit inside them. [5] 

## 3. The Cocountable Topology
Let $X$ be an uncountable set (like the real numbers $\mathbb{R}$). Define the open sets to be the empty set plus any set whose complement is countable. [6, 7] 

* The Behavior: In this topology, the only way a sequence $x_n$ can converge to a point $x$ is if the sequence becomes completely constant ($x_n = x$) after a certain index.
* Why it fails a metric: This topology completely breaks the relationship between sequences and closure. You can easily find a subset $A$ and a point $x$ that is physically touching $A$ (in its closure), but no sequence of elements inside $A$ can ever reach $x$. Metric spaces require that if a point is in the closure, a sequence must be able to reach it. [8, 9, 10] 

## 4. The Zariski Topology (Algebraic Geometry)
Used heavily in geometry and algebra, the open sets on $\mathbb{R}^n$ (or $\mathbb{C}^n$) are defined as the complements of the zero-sets of polynomials.

* The Behavior: Open sets are massive. For example, on the real line $\mathbb{R}$, an open set is any set that just misses a finite number of points.
* Why it fails a metric: Any two non-empty open sets must intersect. This means it is completely impossible to separate two distinct points into their own isolated neighborhoods. It fails the Hausdorff property ($T_2$), which is a mandatory requirement for any metric space. [11, 12, 13, 14, 15] 

------------------------------
## Summary Checklist: Why They Fail

| Example Space [16, 17, 18] | What Property Breaks Down? |
|---|---|
| Pointwise Function Space | Too large to have a countable base of neighborhoods. |
| Weak Topology ($\ell^2$) | Neighborhoods contain infinite lines, breaking metric boundedness. |
| Cocountable Topology | Sequences fail to detect when points are touching a set. |
| Zariski Topology | Cannot separate two points with distinct open sets (Non-Hausdorff). |

If you want to dive deeper, let me know if I should:

* Write out the formal neighborhood construction for the $\ell^2$ weak topology.
* Explain why nets and filters are used to fix sequences in these spaces.
* Show how the Zariski topology breaks the Hausdorff condition with a concrete algebraic example. [19, 20, 21, 22] 


[1] [https://home.fau.edu](https://home.fau.edu/wmcgove1/web/Papers/rtop.pdf)
[2] [https://planetmath.org](https://planetmath.org/examplesofcompactspaces)
[3] [https://www.vaia.com](https://www.vaia.com/en-us/textbooks/math/functional-analysis-and-infinite-dimensional-geometry-0-edition/chapter-3/problem-51-let-x-be-an-infinite-dimensional-banach-space-sho/)
[4] [https://math.stackexchange.com](https://math.stackexchange.com/questions/3535073/monotone-convergence-theorem-for-sequences-in-a-completely-metrizable-totally-or)
[5] [https://math.stackexchange.com](https://math.stackexchange.com/questions/1354159/the-topological-space-leftx-2x-right-is-metrizable)
[6] [https://www.tandfonline.com](https://www.tandfonline.com/doi/abs/10.1080/10511970.2021.1872750)
[7] [https://en.citizendium.org](https://en.citizendium.org/wiki/Cocountable_topology)
[8] [https://malayajournal.org](http://malayajournal.org/articles/MJM0S200373.pdf)
[9] [https://dbfin.com](https://dbfin.com/topology/munkres/chapter-2/section-18-continuous-functions/problem-13-solution/)
[10] [https://www.vaia.com](https://www.vaia.com/en-us/textbooks/math/elementary-real-analysis-1-edition/chapter-13/problem-18-a-metric-space-x-is-said-to-be-absolutely-closed-/)
[11] [https://mathoverflow.net](https://mathoverflow.net/questions/192127/what-is-the-meaning-of-non-hausdorff-spaces-in-algebraic-geometry)
[12] [https://prepp.in](https://prepp.in/question/let-mathbb-r-denote-the-set-of-all-real-numbers-co-6967f6d627723f4257376b8a)
[13] [https://askfilo.com](https://askfilo.com/user-question-answers-smart-solutions/example-of-a-topological-space-that-is-t1-but-not-t2-provide-3336353732353036)
[14] [https://askfilo.com](https://askfilo.com/user-question-answers-smart-solutions/show-that-every-metric-space-is-3135333533353935)
[15] [https://www.studysmarter.co.uk](https://www.studysmarter.co.uk/explanations/math/pure-maths/zariski-topology/)
[16] [https://math.stackexchange.com](https://math.stackexchange.com/questions/1363870/is-every-hausdorff-space-metric)
[17] [https://www.researchgate.net](https://www.researchgate.net/post/What-problem-will-occur-if-we-not-suppose-the-topological-space-is-Hausdorff-and-second-countable)
[18] [https://www.studysmarter.co.uk](https://www.studysmarter.co.uk/explanations/math/pure-maths/zariski-topology/)
[19] [https://www.mdpi.com](https://www.mdpi.com/2227-7390/11/3/663)
[20] [https://mathoverflow.net](https://mathoverflow.net/questions/5957/what-is-a-metric-space)
[21] [https://www.cs.uoi.gr](https://www.cs.uoi.gr/~nikos/EDBT26.pdf)
[22] [https://old.maa.org](https://old.maa.org/press/maa-reviews/topology-point-set-and-geometric)

---

Give an example of "In infinite-dimensional spaces (like function spaces), the standard norm topology is "too large" (has too many open sets). As a result, the closed unit ball is never compact."

The classic example demonstrating this failure of compactness is the standard infinite-dimensional Hilbert space $\ell^2$ (the space of all square-summable sequences) using the standard basis vectors [3].
------------------------------
## The Setup
Consider the Hilbert space $\ell^2$, where every element is an infinite sequence of numbers $x = (x_1, x_2, x_3, \dots)$ such that the norm is finite:
$$\Vert{}x\Vert{} = \sqrt{\sum_{i=1}^{\infty} x_i^2} < \infty$$ 
The closed unit ball $B$ in this space is the set of all sequences whose norm is less than or equal to 1:
$$B = \lbrace x \in \ell^2 : \Vert{}x\Vert{} \le 1 \rbrace$$ 
## The Sequence Counterexample
To prove that the closed unit ball $B$ is not compact, we only need to find a sequence entirely inside $B$ that has no convergent subsequence (sequential compactness).
Let us use the standard infinite coordinate basis vectors $e_n$:

* $e_1 = (1, 0, 0, 0, \dots)$
* $e_2 = (0, 1, 0, 0, \dots)$
* $e_3 = (0, 0, 1, 0, \dots)$
* $e_n = (0, 0, \dots, 0, 1, 0, \dots)$ (where the $1$ is in the $n$-th position)

## Why It Fails to Converge in the Norm Topology

   1. Inside the Ball: For every single vector in this sequence, the norm is exactly one ($\Vert{}e_n\Vert{} = 1$). Therefore, the entire sequence lives on the surface of our closed unit ball $B$.
   2. The Distance Between Elements: Let us calculate the norm distance between any two distinct basis vectors $e_n$ and $e_m$ (where $n \neq m$):
   $$\Vert{}e_n - e_m\Vert{} = \sqrt{(1)^2 + (-1)^2} = \sqrt{2}$$ 
   3. No Subsequence Can Converge: Every single element in this infinite sequence sits at a rigid distance of exactly $\sqrt{2}$ from every other element.
   4. The Open Set Problem: Because the norm topology is "too large," it can easily construct tiny, isolated open balls of radius $0.5$ around every single $e_n$. None of these tiny balls overlap. The sequence is completely scattered; it cannot bunch up, cluster, or find a limit point.

Because you cannot extract a convergent subsequence from $\lbrace e_n\rbrace $, the closed unit ball $B$ is not compact under the norm topology.
------------------------------
## The Contrast: How the Weak Topology Fixes This
If you switch from the norm topology to the weak topology, the exact same sequence $\lbrace e_n\rbrace $ behaves completely differently:

* The Test: In the weak topology, convergence is judged by passing the vectors through linear functionals (which, in $\ell^2$, means taking the dot product with any fixed vector $y \in \ell^2$).
* The Result: For any fixed sequence $y = (y_1, y_2, y_3, \dots)$, the dot product is $\langle e_n, y \rangle = y_n$. Since the sum of $y_n^2$ is finite, the individual terms $y_n$ must fade to 0 as $n$ goes to infinity.
* Convergence: Therefore, $\langle e_n, y \rangle \to 0$ for every single test vector. The sequence $e_n$ converges weakly to 0.

By stripping away the aggressive open sets of the norm topology, the weak topology allows these scattered basis vectors to "collapse" down to the origin, restoring the compactness of the unit ball [3].
------------------------------
If you want to explore further, let me know if I should:

* Show you how this same phenomenon looks in continuous function spaces (like $C[0,1]$ or $L^2[0,1]$ using sine waves).
* Provide the explicit open cover proof of non-compactness instead of using sequences.
* Explain how the Riesz Lemma uses this concept to prove that no infinite-dimensional normed space can have a compact unit ball.


---

You said space of all functions equipped with the topology of pointwise convergence. Could you explain to me, what does it mean a topology pointwise convergence? Is not topology a pair of sets and a subset of all subsets?

You are completely correct about the definition of a topology! A topology is strictly a pair $(X, \tau)$, where $X$ is a set and $\tau$ is a collection of subsets (the open sets) satisfying specific axioms.
When mathematicians say "the topology of pointwise convergence," they are using a common shorthand. They mean: "The specific collection of open sets $\tau$ chosen such that a sequence of functions converges using those open sets if and only if it converges pointwise."
Here is exactly how that collection of open sets is constructed.
------------------------------
## 1. The Underlying Set ($X$)
Our underlying set $X$ is the space of all functions from $\mathbb{R}$ to $\mathbb{R}$, often written as $\mathbb{R}^\mathbb{R}$. An individual "point" in this space is an entire function, $f(x)$.
## 2. Defining the "Basic" Open Sets
To build the collection $\tau$, we define the foundational open neighborhoods. In this topology, a single basic open set is determined by checking a function at only a finite number of inputs.
Let $f$ be a target function. Pick a finite set of inputs $\lbrace x_1, x_2, \dots, x_k\rbrace $ and a small error margin $\epsilon > 0$. The basic open neighborhood around $f$ is the set of all functions $g$ that stay close to $f$ at only those specific inputs:
$$U(f; x_1, \dots, x_k; \epsilon) = \lbrace  g \in \mathbb{R}^\mathbb{R} : \vert{}g(x_i) - f(x_i)\vert{} < \epsilon \text{ for all } i = 1, \dots, k \rbrace $$ 
## 3. Building the Full Topology ($\tau$)
The actual topology $\tau$ is the collection of all possible arbitrary unions of these basic open sets $U$. This collection perfectly satisfies the required topological axioms:

* The empty set and the entire space $\mathbb{R}^\mathbb{R}$ are in $\tau$.
* Any union of these sets is in $\tau$.
* Any finite intersection of these sets is in $\tau$.

## 4. Why is it called "Pointwise Convergence"?
We name the collection $\tau$ after how sequences behave inside it.
In any topological space, a sequence of points $f_n$ converges to $f$ if, for every open set containing $f$, the sequence eventually gets trapped inside that open set.
Because our open sets only test a finite number of points at a time, forcing $f_n$ to enter every open set containing $f$ is mathematically identical to demanding that:
$$\lim_{n \to \infty} f_n(x) = f(x) \quad \text{for every individual } x \in \mathbb{R}$$ 
This is the exact definition of pointwise convergence from calculus.
------------------------------
## Summary of the Dual Perspectives

| The Abstract Definition | The Functional Shorthand |
|---|---|
| The Set $X$ | The set of all functions $f: \mathbb{R} \to \mathbb{R}$. |
| The Topology $\tau$ | All unions of finite-point "tunnel" constraints $U$. |
| The Practical Result | Convergence in $\tau$ means tracking points one by one. |

If you want to see how this works mechanically, let me know if I should:

* Show you how to prove a finite intersection of these basic open sets stays open.
* Explain why this is identical to the Product Topology via Tychonoff's theorem.
* Contrast this collection of open sets with the collection used for uniform convergence.


---

So, the set tau is usually induced by a single property, sets of properties or handcrafted?

In practice, topologies are almost never "handcrafted" by listing individual open sets one by one. Because topologies can contain uncountably many sets, mathematicians generate the collection $\tau$ using systematic machinery driven by a specific structural property or goal.
The collection $\tau$ is usually induced in one of three systematic ways:
------------------------------
## 1. Induced by an "Evaluating Property" (Maps & Functions)
This is how the topology of pointwise convergence and all weak topologies are built. You have a collection of functions, and you want to force those functions to be continuous.

* The Goal: You demand that a specific property or measurement (like evaluating a function at a point $x$, or taking a dot product) behaves nicely.
* The Mechanism: You take the absolute bare minimum number of sets required to make those measurements continuous, and then you close the collection under unions and finite intersections.
* Analogy: Imagine setting up security checkpoints. You don't build a whole wall; you just declare that passing through specific gates must be safe, and the rest of the layout adapts automatically.

## 2. Induced by a "Distance Property" (Metrics & Norms)
This is the most common way to generate a topology in early mathematics.

* The Goal: You want to capture the structural property of "closeness" using a single real-number distance formula $d(x,y)$.
* The Mechanism: You define basic open balls $B_\epsilon(x) = \lbrace y : d(x,y) < \epsilon\rbrace$. Your topology $\tau$ is automatically generated by taking every possible union of these balls.

## 3. Induced by an "Algebraic Property" (Structures & Operations)
Sometimes, the open sets are completely dictated by the algebraic equations of the space itself.

* The Goal: You want geometric shapes (like curves, surfaces, or solutions to equations) to act as the fundamental rigid structures.
* The Mechanism: In the Zariski Topology, you declare that the "closed sets" are exactly the zero-sets of polynomial equations. The open sets ($\tau$) are then automatically defined as the complements of those algebraic shapes.

------------------------------
## Summary of How $\tau$ is Created
Mathematicians use a "Base" or a "Subbase"—a small, highly structured rulebook—and let the axioms of topology do the heavy lifting to generate the massive set $\tau$:

| Generation Method | The "Property" Dictating the Rulebook | Resulting Topology $\tau$ |
|---|---|---|
| Functional Demand | "Evaluation at a finite number of points must be continuous." | Pointwise / Weak Topology |
| Geometric Metric | "Every point within distance $\epsilon$ must be included." | Metric / Strong Topology |
| Algebraic Constraint | "Complements of polynomial solutions must be open." | Zariski Topology |

If you are interested, we can look at the exact recipe (called the Subbase Theorem) that turns a raw set of properties into a valid topology, or look at how weak topologies specifically pick their property. What would you like to explore next?

---

In topology tau is a set of open sets or just a set of subsets with specific properties?

In topology, $\tau$ is both: it is a specific set of subsets that satisfies three precise structural properties, and by definition, the members of that set are called "open sets." [1, 2] 
## 1. Axiomatic Definition
Let $X$ be a non-empty set. A topology $\tau$ on $X$ is a collection of subsets of $X$ (meaning $\tau \subseteq \mathcal{P}(X)$, the power set) that satisfies exactly three axioms: [3, 4, 5] 

   1. Trivial Sets: The empty set $\emptyset$ and the whole space $X$ must belong to $\tau$.
   2. Arbitrary Unions: The union of any family (finite or infinite) of sets in $\tau$ must belong to $\tau$.
   3. Finite Intersections: The intersection of any finite number of sets in $\tau$ must belong to $\tau$. [6, 7, 8, 9, 10] 

## 2. The Link to "Open Sets"
The term "open set" has no independent meaning in general topology. [11] 

* A subset $U \subseteq X$ is defined to be open if and only if $U \in \tau$.
* You cannot determine if a set is open until you choose a specific topology $\tau$. [12, 13] 

## 3. Example: One Set, Different Topologies
The same underlying set $X = \lbrace 1, 2\rbrace $ can be given different collections of subsets, creating different topologies: [14] 

* Discrete Topology: $\tau_1 = \lbrace \emptyset, \lbrace 1\rbrace , \lbrace 2\rbrace , \lbrace 1,2\rbrace \rbrace $. Here, the singletons $\lbrace 1\rbrace $ and $\lbrace 2\rbrace $ are open sets.
* Indiscrete Topology: $\tau_2 = \lbrace \emptyset, \lbrace 1,2\rbrace \rbrace $. Here, the singletons are not open sets.
* Sierpiński Topology: $\tau_3 = \lbrace \emptyset, \lbrace 1\rbrace , \lbrace 1,2\rbrace \rbrace $. Here, $\lbrace 1\rbrace $ is open, but $\lbrace 2\rbrace $ is not. [15, 16, 17, 18, 19] 

All three collections $\tau_1, \tau_2, \tau_3$ are valid topologies because they each satisfy the three axioms, defining different rules for what counts as an "open set" on the exact same space. [20] 
To ground this in your prior context, if you want to explore how these abstract rules apply to analysis, please let me know if you would like to look at:

* How the weak topology on a Banach space is generated by linear functionals.
* The difference between metric open sets (using $\epsilon$-balls) and general topological open sets.
* How closed sets or neighborhoods are defined using $\tau$. [21, 22, 23, 24] 


[1] [https://www.linkedin.com](https://www.linkedin.com/pulse/introduction-topology-jack-stoppenbach)
[2] [https://terezatizkova.substack.com](https://terezatizkova.substack.com/p/what-is-topology-and-why-you-should)
[3] [https://mathonline.wikidot.com](http://mathonline.wikidot.com/the-fundamentals-of-topological-spaces)
[4] [https://www.researchgate.net](https://www.researchgate.net/publication/354625570_Introduction_Topology)
[5] [https://link.springer.com](https://link.springer.com/article/10.1007/s40995-021-01223-z)
[6] [https://www.math.fau.de](https://www.math.fau.de/wp-content/uploads/2024/01/topo.pdf)
[7] [https://mathonline.wikidot.com](http://mathonline.wikidot.com/nested-topologies)
[8] [https://mathonline.wikidot.com](http://mathonline.wikidot.com/the-fundamentals-of-topological-spaces)
[9] [https://en.wikipedia.org](https://en.wikipedia.org/wiki/Topological_space)
[10] [https://mathonline.wikidot.com](http://mathonline.wikidot.com/the-fundamentals-of-topological-spaces)
[11] [https://www.hiroleetanaka.com](https://www.hiroleetanaka.com/pdfs/2020-fall-4330-topology-notes-13-quotient-spaces.pdf)
[12] [https://en.wikipedia.org](https://en.wikipedia.org/wiki/Fr%C3%A9chet_space)
[13] [https://math.stackexchange.com](https://math.stackexchange.com/questions/413033/little-problem-about-open-set-in-the-definition-of-topology)
[14] [https://math.stackexchange.com](https://math.stackexchange.com/questions/1833269/what-does-it-mean-to-have-a-different-topology)
[15] [https://math.stackexchange.com](https://math.stackexchange.com/questions/3973137/what-is-an-open-set-in-a-topological-space)
[16] [https://math.stackexchange.com](https://math.stackexchange.com/questions/94456/product-topology-of-discrete-sets)
[17] [https://math.stackexchange.com](https://math.stackexchange.com/questions/1794473/constructing-topology-on-bbbz)
[18] [https://rip94550.wordpress.com](https://rip94550.wordpress.com/2008/11/21/topology-separation-axioms/)
[19] [https://www.researchgate.net](https://www.researchgate.net/topic/General-Topology)
[20] [https://arxiv.org](https://arxiv.org/abs/2201.10977)
[21] [https://mathoverflow.net](https://mathoverflow.net/questions/338910/properties-of-the-topology-of-sequential-convergence-tau-textseq)
[22] [https://planetmath.org](https://planetmath.org/openset)
[23] [https://fiveable.me](https://fiveable.me/thinking-like-a-mathematician/unit-8/topological-spaces/study-guide/M0f4Xg7YRAbdhnks)
[24] [https://link.springer.com](https://link.springer.com/article/10.1007/s00012-023-00815-7)

---

Building intuition for infinite-dimensional spaces requires moving from geometric visualization to algebraic and analytic thinking.
## 📐 The Limit of Geometry
Visual intuition fails because human brains evolve for three dimensions. [1] 

* No visualization: You cannot picture infinite perpendicular axes.
* The "Exploding" Sphere: In high dimensions, a hypersphere's volume concentrates entirely near its surface.
* The "Spiky" Cube: The corners of a high-dimensional cube stretch far away from the center.
* Orthogonal Escape: You can always find a new direction perpendicular to all existing directions. [2, 3, 4, 5, 6] 

## 🧠 Shift to Functional Intuition
Stop thinking of points as positions. Think of points as functions.

* Points are graphs: A single "point" in an infinite space is an entire wave or curve.
* Coordinates are values: Instead of $x, y, z$ coordinates, a function has coordinates at every single input $t$.
* Distance is difference: The distance between two points is the area between their curves.
* Basis elements are notes: Just as 3D vectors use $\hat{i}, \hat{j}, \hat{k}$, function spaces use base shapes like sine and cosine waves. [7, 8, 9, 10] 

## 🎒 Core Conceptual Tools

* Sequences: Think of the space as an endless list of numbers $(x_1, x_2, x_3, \dots)$.
* Constraints: To keep things manageable, we restrict the space (e.g., the sum of squares must be finite).
* Linear Algebra: Concepts like lines, planes, and projections still work exactly the same way.
* Topology: Nearby functions look similar when plotted together. [11, 12] 

## ⚠️ Common Intuitive Traps

* Bounded sets: Closed and bounded sets are not automatically compact.
* Unit spheres: The surface of a unit sphere contains infinitely many points that are far apart from each other.
* Convergence: A sequence of functions can close in on a target in one definition of distance, but fail in another. [13, 14, 15] 

[1] [https://www.preposterousuniverse.com](https://www.preposterousuniverse.com/blog/2014/12/16/guest-post-chip-sebens-on-the-many-interacting-worlds-approach-to-quantum-mechanics/comment-page-2/)
[2] [https://www.reddit.com](https://www.reddit.com/r/learnmath/comments/1drm5xi/sparsity_of_points_in_highdimensions_curse_of/)
[3] [https://cdn.aaai.org](https://cdn.aaai.org/ojs/9855/9855-13-13383-1-2-20201228.pdf)
[4] [https://www.reddit.com](https://www.reddit.com/r/askscience/comments/1wqfqi/is_there_a_reason_why_the_unit_sphere_has_the/)
[5] [https://cdn.aaai.org](https://cdn.aaai.org/ojs/9855/9855-13-13383-1-2-20201228.pdf)
[6] [https://medium.datadriveninvestor.com](https://medium.datadriveninvestor.com/how-to-travel-in-high-dimensional-spaces-ee06f271022b)
[7] [https://www.reddit.com](https://www.reddit.com/r/PhilosophyofScience/comments/sfuljs/how_could_space_time_be_emergent/)
[8] [https://chalkdustmagazine.com](https://chalkdustmagazine.com/features/a-symmetric-universe/)
[9] [https://www.facebook.com](https://www.facebook.com/groups/quantumphysicsnews/posts/3401997113405755/)
[10] [https://www.reddit.com](https://www.reddit.com/r/learnmath/comments/wyfzy/what_is_an_intuitive_understanding_of_vector/)
[11] [https://math.stackexchange.com](https://math.stackexchange.com/questions/63650/dimension-of-continuous-functions)
[12] [https://math.stackexchange.com](https://math.stackexchange.com/questions/2042005/infinite-sums-in-vector-spaces)
[13] [https://faculty.etsu.edu](https://faculty.etsu.edu/gardnerr/vs/maa01.pdf)
[14] [https://www.reddit.com](https://www.reddit.com/r/askscience/comments/45wb50/has_the_banachtarski_paradox_been_generalized_to/)
[15] [https://www.reddit.com](https://www.reddit.com/r/AskScienceDiscussion/comments/qtxqcp/actual_space_with_imaginary_axes/)

---

> Is not Parseval's identity just a Bessel inequality, where the orthonormal subset forms a basis?

Yes — **that is exactly the right intuition**, with one important nuance.

For an orthonormal subset ((e_i)_{i \in I}) in a Hilbert space (H), **Bessel’s inequality** says

[
\sum_{i \in I} |\langle x,e_i\rangle|^2 \le |x|^2.
]

It says: the squared lengths of the “shadows” of (x) onto the orthonormal directions cannot exceed the full squared length of (x).

Now, if ((e_i)_{i \in I}) is not just an orthonormal subset but a **complete orthonormal basis** of (H), then no part of (x) is left outside those directions. Therefore the inequality becomes equality:

[
\sum_{i \in I} |\langle x,e_i\rangle|^2 = |x|^2.
]

That equality is **Parseval’s identity**.

So yes:

[
\boxed{\text{Parseval's identity = Bessel's inequality with a complete orthonormal basis.}}
]

The geometric reason is this. For any orthonormal family, you can decompose (x) as

[
x = \sum_i \langle x,e_i\rangle e_i + r,
]

where (r) is the part of (x) orthogonal to all the chosen (e_i)'s. Then

[
|x|^2
=====

\sum_i |\langle x,e_i\rangle|^2 + |r|^2.
]

Since (|r|^2 \ge 0), you get Bessel’s inequality.

If the orthonormal family is complete, then there is no leftover orthogonal part:

[
r = 0.
]

So you get Parseval:

[
|x|^2
=====

\sum_i |\langle x,e_i\rangle|^2.
]

One subtlety: here “basis” means **orthonormal/Hilbert basis**, meaning its closed linear span is all of (H). It does **not** mean a Hamel basis, where every vector is a finite linear combination. In infinite-dimensional Hilbert spaces, orthonormal expansions are usually infinite series.

> you mean orthonormal basis, which expansion converges for every x?

Yes. By **orthonormal basis** here I mean a **complete orthonormal system** ((e_i)_{i\in I}) such that every (x\in H) can be recovered as the Hilbert-space limit

[
x=\sum_{i\in I} \langle x,e_i\rangle e_i.
]

So in a separable Hilbert space, with a countable orthonormal basis ((e_n)_{n\ge 1}), this means

[
x = \lim_{N\to\infty} \sum_{n=1}^N \langle x,e_n\rangle e_n.
]

The convergence is in the **norm of (H)**:

[
\left|x-\sum_{n=1}^N \langle x,e_n\rangle e_n\right|\to 0.
]

Then Parseval says

[
|x|^2=\sum_{n=1}^\infty |\langle x,e_n\rangle|^2.
]

So yes: **an orthonormal basis is an orthonormal family whose Fourier expansion converges to every vector (x\in H)**.

The important distinction is:

[
\text{orthonormal set}
\quad \Rightarrow \quad
\sum_n |\langle x,e_n\rangle|^2 \le |x|^2
]

but

[
\text{complete orthonormal basis}
\quad \Rightarrow \quad
x=\sum_n \langle x,e_n\rangle e_n
\quad \text{and} \quad
\sum_n |\langle x,e_n\rangle|^2=|x|^2.
]

So the missing condition is **completeness**: the closed span of the (e_n)'s is all of (H),

[
\overline{\operatorname{span}{e_n:n\in\mathbb N}}=H.
]

That is exactly what guarantees that the expansion converges for every (x).

---

> Different views on continuous maps and why do we have different views (because of different settings)

A **continuous map** (or function) is a mapping between mathematical spaces where small changes in the input result in small changes in the output. Depending on the context, it is formally defined in one of three primary ways:

## 1. The Topological Definition (Most General)
In general topology, a function $f: X \rightarrow Y$ between two topological spaces is continuous if and only if the preimage of every open set in $Y$ is an open set in $X$.

* That is, for every open set $U \subseteq Y$, the set $f^{-1}(U) = \lbrace x \in X \mid f(x) \in U\rbrace$ is open.

## 2. The Metric Space Definition ($\epsilon-\delta$)

In metric spaces (where a precise distance $d$ can be measured), a map $f: X \rightarrow Y$ is continuous if for every point $x \in X$ and every $\epsilon > 0$, there exists a $\delta > 0$ such that:

* If $d(x, x_0) < \delta$, then $d(f(x), f(x_0)) < \epsilon$.
(Intuitively, you can keep the output within a tiny "target" distance $\epsilon$ by restricting the input to a small enough "zone" $\delta$.)

## 3. The Real Analysis Definition (Limits)

For everyday functions of real numbers ($f: \mathbb{R} \rightarrow \mathbb{R}$), a map is continuous at a specific point $c$ if the limit of the function as $x$ approaches $c$ is exactly equal to the function's value at $c$:

* $\lim_{x \to c} f(x) = f(c)$