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

## Function spaces

### $L^p$ space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^p$ space)</span></p>

For a measure space $(\Omega, \mathcal{A}, \mu)$ and $1 \le p < \infty$,

$$
L^p(\Omega) = \Big\{ f : \Omega \to \mathbb{R} \text{ measurable} \;\Big|\; \int_\Omega \lvert f \rvert^p \, d\mu < \infty \Big\}, \qquad \|f\|_{L^p} = \Big(\int_\Omega \lvert f \rvert^p \, d\mu\Big)^{1/p},
$$

modulo equality $\mu$-a.e. For $p=\infty$, $\|f\|_{L^\infty} = \operatorname*{ess\,sup} \lvert f \rvert$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why $L^p$ everywhere)</span></p>

* Banach for all $1 \le p \le \infty$; **Hilbert** when $p=2$ (inner product $\int fg$).
* **Reflexive** and **separable** for $1 < p < \infty$ (on $\sigma$-finite measures). $L^1$ is not reflexive; $L^\infty$ is not separable.
* **Dual:** $(L^p)^* \cong L^q$ with $\tfrac{1}{p}+\tfrac{1}{q}=1$ for $1 \le p < \infty$ — this is Riesz representation in $L^p$.
* Natural home for energies of the form $\int \lvert f\rvert^p$ and for almost every theorem in analysis/PDE.

</div>

### $C_c^\infty$ space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($C_c^\infty$ — test functions)</span></p>

$C_c^\infty(\Omega)$ is the space of smooth functions $\varphi : \Omega \to \mathbb{R}$ whose **support** $\overline{\{x : \varphi(x) \ne 0\}}$ is a compact subset of $\Omega$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(role)</span></p>

The "probes" of distribution theory. Two crucial facts:

* **Density**: $C_c^\infty(\Omega)$ is dense in $L^p(\Omega)$ for $1 \le p < \infty$ — prove things on $C_c^\infty$ and extend.
* **Free differentiation**: integrate by parts as many times as you want, which is exactly what defines weak derivatives, weak solutions, and Sobolev spaces.

</div>

### Hölder space $C^{k,\alpha}$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hölder space)</span></p>

For $k \in \mathbb{N}_0$ and $\alpha \in (0,1]$, $C^{k,\alpha}(\overline{\Omega})$ is the set of $C^k$ functions whose $k$-th derivatives are **$\alpha$-Hölder continuous**:

$$
[D^k f]_\alpha := \sup_{x \ne y} \frac{\lvert D^k f(x) - D^k f(y) \rvert}{\lvert x - y \rvert^\alpha} < \infty,
$$

with norm $\|f\|_{C^{k,\alpha}} = \|f\|_{C^k} + [D^k f]_\alpha$.

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
W^{k,p}(\Omega) = \Big\{ f \in L^p(\Omega) \;\Big|\; D^\alpha f \in L^p(\Omega) \text{ for all } \lvert\alpha\rvert \le k\Big\},
$$

where $D^\alpha f$ is the **weak** derivative, with norm

$$
\|f\|_{W^{k,p}} = \Big(\sum_{\lvert\alpha\rvert \le k}\|D^\alpha f\|_{L^p}^p\Big)^{1/p}.
$$

Hilbert case $H^k := W^{k,2}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(the two facts you reuse)</span></p>

* **Sobolev embedding**: enough derivatives in $L^p$ buys higher integrability or continuity. On $\mathbb{R}^n$ with $p < n$, $W^{1,p} \hookrightarrow L^{p^*}$ with $p^* = \tfrac{np}{n-p}$. If $kp > n$, $W^{k,p} \hookrightarrow C^{0,\alpha}$.
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

For $f : \Omega \to X$ strongly measurable into a Banach space, $f$ is **Bochner integrable** iff $\int_\Omega \|f(x)\|_X \, d\mu(x) < \infty$. The integral $\int_\Omega f \, d\mu \in X$ is then the $X$-norm limit of integrals of simple-function approximants.

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

In a normed space $X$: $x_n \to x$ if $\|x_n - x\|_X \to 0$. I.e. convergence in the **norm topology**.

</div>

### Weak convergence (general)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(weak convergence)</span></p>

In a Banach space $X$: $x_n \rightharpoonup x$ if $x^*(x_n) \to x^*(x)$ for every $x^* \in X^*$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(facts)</span></p>

Intuition: pair against every continuous probe and check pointwise convergence of the pairings. Strictly weaker than strong unless $\dim X < \infty$.

* Weakly convergent sequences are **bounded** (Banach–Steinhaus).
* Norm is **lower semicontinuous** under weak limits: $\|x\| \le \liminf \|x_n\|$ (mass can be lost in the limit but not gained).
* In a reflexive space, bounded $\Rightarrow$ has a weakly convergent subsequence (Eberlein–Šmulian / Banach–Alaoglu).

</div>

### Strong convergence in $L^p$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(strong $L^p$ convergence)</span></p>

$f_n \to f$ in $L^p$ iff $\|f_n - f\|_{L^p} \to 0$.

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

$f_n(x) = \sin(nx) \rightharpoonup 0$ in $L^2([0, 2\pi])$ by Riemann–Lebesgue, but $\|f_n\|_{L^2} = \sqrt{\pi}$ for all $n$ — so the convergence is **not** strong. The mass is preserved; it just oscillates faster and faster and averages out against any test function.

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

### Uniform topology

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(uniform topology)</span></p>

Induced by the sup-norm $\|f\|_\infty = \sup_x \lvert f(x)\rvert$. Convergence is uniform convergence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Strongest of the three below; preserves continuity, integrals, derivatives (up to uniform convergence of derivatives), etc.

</div>

### Pointwise topology

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(pointwise topology)</span></p>

The product topology on $\mathbb{R}^X$: $f_n \to f$ iff $f_n(x) \to f(x)$ for every $x$.

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
\{ x : \lvert x^*(x) - x^*(x_0) \rvert < \varepsilon \}, \quad x^* \in X^*, \; \varepsilon > 0, \; x_0 \in X.
$$

The associated sequential convergence is weak convergence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

The point: bounded sets are **much closer to compact** here than in the norm topology — that is exactly Banach–Alaoglu / Eberlein–Šmulian.

</div>

---

## Compactness flavors

### Compact support

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(compact support)</span></p>

A function $f$ on $X$ has compact support if $\operatorname{supp}(f) := \overline{\{x : f(x) \ne 0\}}$ is compact.

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

$f : X \to \mathbb{R} \cup \{+\infty\}$ is **lower semicontinuous (l.s.c.)** at $x_0$ if

$$
\liminf_{x \to x_0} f(x) \ge f(x_0).
$$

Equivalently, $\{f \le c\}$ is closed for every $c \in \mathbb{R}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why it matters for optimization)</span></p>

The right class for **minimization**: a l.s.c. function attains its infimum on a compact set (Weierstrass). Norms are l.s.c. **for weak convergence** — which is why one minimizes energies by passing to weak limits.

</div>

---

## "Niceness" properties of spaces

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

A Banach space $X$ is **reflexive** if the canonical embedding $J : X \to X^{**}$, $J(x)(x^*) = x^*(x)$, is **surjective** (and hence an isometric isomorphism).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(payoff)</span></p>

Closed bounded sets are **weakly compact**, so bounded sequences have weakly convergent subsequences.

* Examples: Hilbert spaces, $L^p$ ($1<p<\infty$), $W^{k,p}$ ($1<p<\infty$).
* Non-examples: $L^1$, $L^\infty$, $C(K)$.

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

## The headline theorems

### Riesz representation theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Riesz — Hilbert version)</span></p>

Let $H$ be a Hilbert space. For every continuous linear functional $\varphi \in H^*$ there is a **unique** $y \in H$ with

$$
\varphi(x) = \langle x, y \rangle \quad \forall x \in H,
$$

and $\|\varphi\|_{H^*} = \|y\|_H$. So $H^* \cong H$ canonically.

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

### Parseval's identity

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Parseval)</span></p>

For a Hilbert space $H$ with orthonormal basis $\{e_n\}_{n \in \mathbb{N}}$ and any $x \in H$,

$$
\|x\|_H^2 = \sum_{n} \lvert \langle x, e_n \rangle \rvert^2.
$$

Concretely for Fourier series on $[0, 2\pi]$: $\int_0^{2\pi} \lvert f \rvert^2 \, dx = 2\pi \sum_{n \in \mathbb{Z}} \lvert \hat f_n \rvert^2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span></p>

Says the Fourier transform is an **isometry** $L^2 \to \ell^2$ (or $L^2 \to L^2$ on $\mathbb{R}^n$).

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

### Banach–Alaoglu theorem

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Banach–Alaoglu)</span></p>

Let $X$ be a normed space. The closed unit ball $B_{X^*}$ of the dual is **compact in the weak-$*$ topology**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(sequential version)</span></p>

If $X$ is **separable**, $B_{X^*}$ is metrizable in the weak-$*$ topology, hence **sequentially compact**: every bounded sequence in $X^*$ has a weak-$*$ convergent subsequence. The workhorse for extracting limits in infinite dimensions (limits of measures, of $L^\infty$ data, of approximate solutions of PDE).

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

For $f \in L^1_{\text{loc}}(\Omega)$ and multi-index $\alpha$, a function $g \in L^1_{\text{loc}}(\Omega)$ is the **weak $\alpha$-th derivative** of $f$, written $g = D^\alpha f$, if

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
\sigma(T) = \{ \lambda \in \mathbb{C} : T - \lambda I \text{ is not invertible} \}.
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
