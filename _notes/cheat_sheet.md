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
L^p(\Omega) = \Big\lbrace f : \Omega \to \mathbb{R} \text{ measurable} \;\Big|\; \int_\Omega \lvert f \rvert^p \, d\mu < \infty \Big\rbrace, \qquad \|f\|_{L^p} = \Big(\int_\Omega \lvert f \rvert^p \, d\mu\Big)^{1/p},
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

$C_c^\infty(\Omega)$ is the space of smooth functions $\varphi : \Omega \to \mathbb{R}$ whose **support** $\overline{\lbrace x : \varphi(x) \ne 0\rbrace}$ is a compact subset of $\Omega$.

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

For $f : \Omega \to X$ strongly measurable into a Banach space, $f$ is **Bochner integrable** iff $\int_\Omega \|f(x)\|\_X \, d\mu(x) < \infty$. The integral $\int_\Omega f \, d\mu \in X$ is then the $X$-norm limit of integrals of simple-function approximants.

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

In a Banach space $X$: $x_n \rightharpoonup x$ if $x^\ast(x_n) \to x^\ast(x)$ for every $x^\ast \in X^\ast$.

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
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(initial topology)</span></p>

Given a family of maps $\lbrace f_i : X \to Y_i\rbrace_{i \in I}$ with each $Y_i$ already topologised, the **initial topology** on $X$ induced by the family is the **coarsest** topology making every $f_i$ continuous. Explicitly, it is generated by the subbase

$$
\lbrace f_i^{-1}(U) \;:\; i \in I, \; U \subseteq Y_i \text{ open} \rbrace.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(examples and universal property)</span></p>

The initial-topology recipe specializes to almost every "abstract" topology in functional analysis:

* **Subspace** topology — single inclusion $X \hookrightarrow Y$.
* **Product** topology on $\prod_j X_j$ — the family of projections $\pi_j$.
* **Weak topology** on a Banach space $X$ — the family $X^\ast$ of continuous linear functionals.
* **Weak-$\ast$ topology** on $X^\ast$ — the evaluations $\hat x : X^\ast \to \mathbb{R}$, $x^\ast \mapsto x^\ast(x)$, for $x \in X$.
* Topology on $\mathcal{D}'$ — pairings against test functions.

**Universal property.** A map $g : Z \to X$ into a set carrying an initial topology is continuous **iff** every composition $f_i \circ g$ is continuous. Continuity into $X$ reduces to checking it "coordinatewise" — which is exactly why initial topologies are practical.

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

---

## Compactness flavors

### Compact support

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(compact support)</span></p>

A function $f$ on $X$ has compact support if 

$\operatorname{supp}(f) := \overline{\lbrace x : f(x) \ne 0\rbrace}$ 

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

$$J : X \to X^{\ast\ast}$, $J(x)(x^\ast) = x^\ast(x)$$

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

### Tightness / tight measure

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(tight family of measures)</span></p>

A family $\mathcal{M}$ of Borel probability measures on a topological space $X$ is **tight** if for every $\varepsilon > 0$ there exists a **compact** $K \subseteq X$ with

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

If $X$ is **separable**, $B_{X^\ast}$ is metrizable in the weak-$\ast$ topology, hence **sequentially compact**: every bounded sequence in $X^\ast$ has a weak-$\ast$ convergent subsequence. The workhorse for extracting limits in infinite dimensions (limits of measures, of $L^\infty$ data, of approximate solutions of PDE).

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
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Solving variational problems with minimizing sequences)</span></p>

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
