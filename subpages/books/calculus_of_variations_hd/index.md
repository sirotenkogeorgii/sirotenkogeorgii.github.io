---
layout: default
title: "Calculus of Variations"
date: 2026-10-25
excerpt: "Lecture notes on 'Calculus of Variations' (Heidelberg, WS 2026/27, Tim Laux after Hans Knüpfer) — the variational viewpoint, Carathéodory integrands and growth conditions, the fundamental lemma, Gâteaux vs. Fréchet differentiability, Euler–Lagrange equations with Dirichlet and natural boundary conditions, second variation and rank-1 convexity, Null Lagrangians and the determinant, weak continuity of minors, and Brouwer's fixed point theorem."
tags:
  - calculus-of-variations
  - euler-lagrange
  - gateaux-derivative
  - frechet-derivative
  - null-lagrangian
  - rank-one-convexity
  - weak-convergence
  - brouwer-fixed-point
  - sobolev-spaces
  - analysis
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
</style>

# Calculus of Variations

Notes based on the lecture course *Calculus of Variations* held by Prof. Dr. Tim Laux at Heidelberg University in the winter term 2026/27, following the script written by Prof. Dr. Hans Knüpfer (winter 2025/26). The numbering of definitions, theorems and equations follows the script.

**Table of Contents**
- TOC
{:toc}

## Chapter 1: Introduction

### 1.1 Motivation and Literature

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(What the calculus of variations is about)</span></p>

The calculus of variations provides a mathematical framework for studying systems that **minimize or extremize certain quantities**, typically energies or actions. Many physical and geometric problems can be formulated in this way: equilibrium states in mechanics correspond to minima of elastic energy, stationary heat distributions minimize the Dirichlet integral, and light rays or particle trajectories follow paths determined by a principle of least action.

We assume that the state of the system can be described by a function $u:\Omega\to\mathbb R^m$ and consider the problem

$$
I[u]\;\to\;\min \tag{1.1}
$$

for a functional $I:\mathcal A\to[0,\infty)$ and an admissible class of functions $u\in\mathcal A$.

An analogous problem is known from elementary calculus (curve sketching), where we study local minima of a function $f:[a,b]\to\mathbb R$. A necessary condition for a local minimum is the equation $f^{\prime}(x)=0$. Analogously, for the functional in (1.1) we obtain

$$
\delta I[u,\varphi]=0 \tag{1.2}
$$

for suitable test functions $\varphi$. The equation (1.2) is typically a **partial differential equation**; it is called the **Euler–Lagrange equation** (EL equation).

* In the **direct method** of the calculus of variations we study the minimization problem (1.1).
* In the **indirect method** of the calculus of variations we study the equation (1.2).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Questions addressed in this course)</span></p>

In the first part of the lecture:

* **Existence** of a minimizer: does there exist $u\in\mathcal A$ with $I[u]=\inf\_{v\in\mathcal A} I[v]$?
* **Uniqueness** of the minimizer: under which assumptions is the minimizer unique?
* **Properties** of the minimizer (regularity, structure).

In the second part:

* **Generalized notions of solutions**: if the functional $I$ does not admit a minimizer, does there exist a solution to a *relaxed* problem $I\_{\mathrm{rel}}$?
* **Convergence of minimization problems**: for a sequence of functionals $I\_\varepsilon$, what is an appropriate notion of convergence $I\_\varepsilon\to I$? Can we derive simplified or macroscopic problems? (This is $\Gamma$-convergence.)
* **Free boundary problems**: in particular the Stefan problem, describing a system that evolves toward a state of minimal energy.

</div>

<div class="math-callout math-callout--example" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">1.1 (Variational problems)</span></p>

1. **Dirichlet type integrals.** For $u:\Omega\to\mathbb R$ and $p\ge 1$ we consider

   $$
   I[u]=\int_\Omega \lvert\nabla u\rvert^p\,\mathrm{d}x .
   $$

   For $p=2$ this is the **Dirichlet integral**. If $u$ describes the temperature distribution in $\Omega$, then its equilibrium distribution minimizes the Dirichlet functional. The Dirichlet integral also appears in many other contexts, e.g. as the electrostatic energy of a charge distribution.

2. **Geodesics.** Let $M\subset\mathbb R^n$ be a manifold and $p,q\in M$. The shortest curve $\gamma:[0,1]\to M$ connecting $p$ and $q$ minimizes the length functional

   $$
   I[\gamma]=\int_0^1 \lvert\dot\gamma(t)\rvert\,\mathrm{d}t .
   $$

3. **Minimal surfaces.** The surface area of the graph of a function $u:\Omega\to\mathbb R$ is

   $$
   I[u]=\int_\Omega \sqrt{1+\lvert\nabla u\rvert^2}\,\mathrm{d}x .
   $$

   The minimal surface minimizes the area for given boundary data.

4. **Action functionals.** Models in physics are often formulated using the principle of least action. The action functional of a (field-theoretic) harmonic oscillator is

   $$
   S[\phi]=\int_{\Omega\times(0,T)} \phi_t^2-\lvert\nabla\phi\rvert^2\,\mathrm{d}x\,\mathrm{d}t .
   $$

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_brachistochrone.png' | relative_url }}" alt="Four curves joining A to a lower point B with their sliding times; the cycloid is fastest" loading="lazy">
  <figcaption>The brachistochrone (Figure 2 of the script): among all slides from $A$ to $B$, a bead under gravity arrives fastest along a cycloid. Travel time is a functional $T[y]=\int \sqrt{(1+y'^2)/(2gy)}\,\mathrm{d}x$ of the whole curve $y$, and the straight line, a circular arc and a "drop first, then run flat" profile are all slower. This is the archetypal variational problem: the unknown is a function, and the quantity to minimize is an integral of it.</figcaption>
</figure>

<div class="math-callout math-callout--example" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">1.2 (Elastic body)</span></p>

Consider an elastic body in the reference configuration $\Omega\subset\mathbb R^n$, deformed by a displacement field $u:\Omega\to\mathbb R^n$. The energy of a *homogeneous* elastic body is a function of the local deformation of the crystal lattice, which is encoded in the deformation gradient $\nabla u$. The energy density is thus of the form $f(\nabla u)$, where $f$ is the elastic energy density. More generally, for an *inhomogeneous* elastic body the energy has the form

$$
I[u]=\int_\Omega f(x,u,\nabla u)\,\mathrm{d}x .
$$

This functional includes all the previous examples.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Literature</span><span class="math-callout__name">(Recommended selection)</span></p>

1. B. Dacorogna, *Direct Methods in the Calculus of Variations*, 2nd ed., Springer 2008.
2. I. Fonseca, G. Leoni, *Modern Methods in the Calculus of Variations: $L^p$ Spaces*, Springer 2007.
3. M. Giaquinta, S. Hildebrandt, *Calculus of Variations I*, Springer 1996.
4. S. Müller, *Variational Models for Microstructure and Phase Transitions*, lecture notes.
5. F. Rindler, *Calculus of Variations*, Springer 2018.
6. M. Struwe, *Variational Methods*, 4th ed., Springer 2008.

</div>

### 1.2 Carathéodory functions and growth condition

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setting</span><span class="math-callout__name">(Integral functionals and admissible classes)</span></p>

If the energy density at $x$ is a *local* function that depends only on $u(x)$, $Du(x)$ and $x$, we obtain a functional of the form

$$
I[u]=\int_\Omega f(x,u,Du)\,\mathrm{d}x \tag{1.3}
$$

for a given function $f:\Omega\times\mathbb R^m\times\mathbb R^{m\times n}\to\mathbb R$, $f=f(x,z,\xi)$. In the following we mainly consider functionals of this form. We usually assume that $\Omega$ is open and bounded, and we write $\Omega\Subset\mathbb R^n$ for a set that is compactly contained in $\mathbb R^n$ (in particular bounded). We write

$$
W^{1,p}_g(\Omega)^m=\bigl\lbrace u\in W^{1,p}(\Omega)^m:\ u|_{\partial\Omega}=g\bigr\rbrace
$$

with the abbreviation $W^{1,p}(\Omega)^m=W^{1,p}(\Omega,\mathbb R^m)$. Our goal is to find a minimizer of (1.3) in a class of possible states $\mathcal A$; a natural choice is $\mathcal A=W^{1,p}\_g(\Omega)^m$ for a given function $g\in L^p(\partial\Omega)^m$.

**Question.** Under which assumptions on $f$ is the expression (1.3) well-defined for $u\in\mathcal A$ and some $p\ge 1$? We need a *regularity* assumption (so that $x\mapsto f(x,u(x),Du(x))$ is measurable) and a *growth* condition (so that the integral is finite).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name">(Lebesgue vs. Borel measurability)</span></p>

A function $f:\mathbb R^n\to\mathbb R^m$ is

* **Lebesgue measurable** if $f^{-1}(B)$ is Lebesgue measurable for every Borel set $B$,
* **Borel (measurable)** if $f^{-1}(B)$ is a Borel set for every Borel set $B$.

By definition every Borel function is Lebesgue measurable. The composition of two Lebesgue measurable functions need **not** be Lebesgue measurable, because the preimage of a Lebesgue measurable set under a Lebesgue measurable function is not necessarily Lebesgue measurable. It makes sense to use the Borel $\sigma$-algebra on the *target* space, since this is the minimal regularity needed to define integrals.

We write $\overline{\mathbb R}\_\pm:=\mathbb R\cup\lbrace\pm\infty\rbrace$ and $\overline{\mathbb R}:=\mathbb R\cup\lbrace+\infty\rbrace\cup\lbrace-\infty\rbrace$. If not specified otherwise, functions take values in $\mathbb R$.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">1.3 (Carathéodory function and measurability)</span></p>

Let $\Omega\subset\mathbb R^n$ be Lebesgue measurable and let $f:\Omega\times\mathbb R^d\to\mathbb R\cup\lbrace+\infty\rbrace$ be a **Carathéodory function**, i.e.

1. $f(\cdot,\xi)$ is Lebesgue measurable for all $\xi\in\mathbb R^d$, and
2. $f(x,\cdot)$ is continuous for almost all $x\in\Omega$.

Then $g:\Omega\to\mathbb R\cup\lbrace+\infty\rbrace$, $g:=f(\cdot,u)$, is Lebesgue measurable whenever $u:\Omega\to\mathbb R^d$ is Lebesgue measurable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 1.3</summary>

Since $u$ is measurable, there is a sequence of simple measurable functions $u\_k=\sum\_{j}\xi\_{k,j}\mathbf 1\_{E\_{k,j}}$ (finite sums, $E\_{k,j}$ measurable, disjoint and covering $\Omega$) with $u\_k\to u$ a.e. The functions

$$
g_k(x):=f(x,u_k(x))=\sum_j \mathbf 1_{E_{k,j}}(x)\,f(x,\xi_{k,j})
$$

are measurable by (i): on each piece $E\_{k,j}$ the function $g\_k$ coincides with the measurable function $f(\cdot,\xi\_{k,j})$. By (ii), $g\_k(x)\to f(x,u(x))=g(x)$ for a.e. $x$ (namely at every $x$ where $u\_k(x)\to u(x)$ and $f(x,\cdot)$ is continuous). Because the pointwise a.e. limit of measurable functions is measurable, $g$ is measurable. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the two conditions split the way they do)</span></p>

The proof shows exactly what each half of "Carathéodory" is for: measurability in $x$ is only ever tested on *simple* functions $u\_k$ (finitely many fixed values $\xi$), and continuity in $\xi$ is what carries this measurability through the limit $u\_k\to u$. Neither condition alone suffices: a function that is measurable in each variable separately need not be jointly measurable, and then $x\mapsto f(x,u(x))$ can fail to be measurable even for $u=\mathrm{id}$. Take $f=\mathbf 1\_{\lbrace(x,x):\,x\in N\rbrace}$ with $N\subset\mathbb R$ non-measurable: every section $f(\cdot,\xi)$ and $f(x,\cdot)$ is measurable (an indicator of at most one point), but $f(x,x)=\mathbf 1\_N(x)$.

*Alternatively* one can consider **normal integrands**: functions $f:\Omega\times\mathbb R^d\to\overline{\mathbb R}$ such that $f(\cdot,\xi)$ is Borel measurable and $f(x,\cdot)$ is lower semicontinuous (Dacorogna, Def. 3.3).

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Standing assumptions</span><span class="math-callout__name">(Growth condition and regularity)</span></p>

The integral (1.3) is well-defined on $W^{1,p}(\Omega)^m$ under the following conditions:

1. **Growth condition.** There is a constant $C>0$ such that for all $(x,z,\xi)\in\Omega\times\mathbb R^m\times\mathbb R^{m\times n}$

   $$
   \lvert f(x,z,\xi)\rvert\le C\bigl(1+\lvert z\rvert^p+\lvert\xi\rvert^p\bigr).
   $$

2. **Carathéodory.** $f(\cdot,z,\xi)$ is measurable for all $(z,\xi)\in\mathbb R^m\times\mathbb R^{m\times n}$ and $f(x,\cdot,\cdot)$ is continuous for a.e. $x\in\Omega$.

Indeed, with the identification $\mathbb R^m\times\mathbb R^{m\times n}=\mathbb R^d$, $f$ is a Carathéodory function as in Lemma 1.3, so $x\mapsto f(x,u(x),Du(x))$ is measurable for measurable $u$. From the growth condition we get for $u\in W^{1,p}(\Omega)^m$

$$
\int_\Omega \lvert f(x,u,Du)\rvert\,\mathrm{d}x\le C\int_\Omega \bigl(1+\lvert u\rvert^p+\lvert Du\rvert^p\bigr)\mathrm{d}x<\infty .
$$

In the following we generally assume, unless otherwise stated,

$$
f\in C^\infty(\Omega\times\mathbb R^m\times\mathbb R^{m\times n}).
$$

We write $\mathcal D(\Omega):=C\_c^\infty(\Omega)$ for the space of test functions.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_growth_envelopes.png' | relative_url }}" alt="Three integrands from Example 1.1 plotted against the gradient variable together with dashed p-growth envelopes" loading="lazy">
  <figcaption>The growth condition pins the exponent $p$ of the Sobolev space on which $I$ is finite. The minimal-surface integrand $\sqrt{1+\xi^2}$ grows linearly ($p=1$), the Dirichlet integrand quadratically ($p=2$), and the double-well integrand $(\xi^2-1)^2$ of Example 2.8 quartically ($p=4$): each solid curve stays below its dashed envelope $1+\lvert\xi\rvert^p$, and no smaller $p$ would do.</figcaption>
</figure>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">1.4 (Fundamental lemma, Du Bois-Reymond)</span></p>

Let $\Omega\subset\mathbb R^n$ be open and $v\in L^1(\Omega)$.

1. **Fundamental lemma of the calculus of variations.**

   $$
   \int_\Omega v\,\varphi\,\mathrm{d}x=0\quad\forall\varphi\in\mathcal D(\Omega)\qquad\Longrightarrow\qquad v=0\ \text{a.e.}
   $$

2. **Lemma of Du Bois-Reymond.** Let $\Omega$ be connected. Then

   $$
   \int_\Omega v\,\nabla\varphi\,\mathrm{d}x=0\in\mathbb R^n\quad\forall\varphi\in\mathcal D(\Omega)\qquad\Longrightarrow\qquad v=\mathrm{const}\ \text{a.e.},
   $$

   i.e. a function whose weak gradient vanishes is constant.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_fundamental_lemma.png' | relative_url }}" alt="Left: a function v and a bump test function supported where v is negative, the shaded product has nonzero integral. Right: a step function v and the derivative of a bump straddling the jump, the shaded product has nonzero integral." loading="lazy">
  <figcaption>Test functions as probes. (a) If $v\neq 0$ on a set of positive measure, a bump $\varphi$ concentrated there makes $\int v\varphi\neq 0$, so "$\int v\varphi=0$ for all bumps" forces $v=0$ a.e. (b) In Du Bois-Reymond's lemma the probes are derivatives of bumps, i.e. dipoles. A dipole across a jump of $v$ at $x_0$ returns $\int v\varphi'=-[v]\varphi(x_0)\neq 0$: dipoles are blind to constants but detect every change of $v$.</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 1.4</summary>

**(i).** For $m>0$ define the truncation $v\_m\in L^1\cap L^\infty(\Omega)$ by

$$
v_m(x):=\begin{cases} v(x) & \text{if } \lvert v(x)\rvert\le m,\\ \pm m & \text{if } \pm v(x)\ge m.\end{cases}
$$

It suffices to show $v\_m=0$ a.e. for every $m>0$, since $v\_m\to v$ pointwise. Choose $\varphi\_k\in\mathcal D(\Omega)$ with $\varphi\_k\to v\_m$ a.e. and $\lvert\varphi\_k\rvert\le m$ (for instance mollify $v\_m\mathbf 1\_{\Omega\_k}$ for an exhaustion $\Omega\_k\Subset\Omega$: mollification preserves the bound $\lvert\cdot\rvert\le m$, and a subsequence of the $L^1$-convergent mollifications converges a.e.). Then $\lvert v\varphi\_k\rvert\le m\lvert v\rvert\in L^1(\Omega)$, so by dominated convergence

$$
0=\int_\Omega v\,\varphi_k\,\mathrm{d}x\;\longrightarrow\;\int_\Omega v\,v_m\,\mathrm{d}x\;\ge\;\int_\Omega v_m^2\,\mathrm{d}x ,
$$

where the last inequality holds pointwise: $v\,v\_m=\lvert v\rvert\min(\lvert v\rvert,m)\ge\min(\lvert v\rvert,m)^2=v\_m^2$. Hence $v\_m=0$ a.e.

**(ii).** Let $\Omega^{\prime}\Subset\Omega$ be open and connected, and let $\eta\_\varepsilon$ be a standard (symmetric) mollifier with $\varepsilon<\operatorname{dist}(\Omega^{\prime},\partial\Omega)$. For $\varphi\in\mathcal D(\Omega^{\prime})$ the function $\eta\_\varepsilon\ast\varphi$ belongs to $\mathcal D(\Omega)$, so we may use it in the hypothesis; moving the mollifier across the integral (symmetry of $\eta\_\varepsilon$) and integrating by parts gives, for each component $i$,

$$
0=\int_\Omega v\,\partial_i(\eta_\varepsilon\ast\varphi)\,\mathrm{d}x=\int_\Omega v\,(\eta_\varepsilon\ast\partial_i\varphi)\,\mathrm{d}x=\int_{\Omega'} (\eta_\varepsilon\ast v)\,\partial_i\varphi\,\mathrm{d}x=-\int_{\Omega'} \partial_i(\eta_\varepsilon\ast v)\,\varphi\,\mathrm{d}x\qquad\forall\varphi\in\mathcal D(\Omega').
$$

Since $\eta\_\varepsilon\ast v$ is smooth on $\Omega^{\prime}$, part (i) gives $\nabla(\eta\_\varepsilon\ast v)=0$ in $\Omega^{\prime}$, hence $\eta\_\varepsilon\ast v=\mathrm{const}=:c\_\varepsilon$ on the connected set $\Omega^{\prime}$. As $\varepsilon\to0$, $\eta\_\varepsilon\ast v\to v$ in $L^1(\Omega^{\prime})$, so the constants $c\_\varepsilon$ converge and $v=\lim c\_\varepsilon$ a.e. on $\Omega^{\prime}$. Finally, any two points of the connected open set $\Omega$ lie in a common connected $\Omega^{\prime}\Subset\Omega$ (a thin tube around a path joining them), so the constant is the same throughout $\Omega$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On the proofs, and uniqueness of weak derivatives)</span></p>

* In the proof of (i) the point is the **uniform bound** $\lvert\varphi\_k\rvert\le m$: an approximation $\varphi\_k\to v\_m$ in $L^2$ alone would not license dominated convergence for $\int v\varphi\_k$ when $v$ is merely in $L^1$. (The script's version of this step is written for $\int v\_m\varphi\_k$ with an $L^2$ approximation; the truncation plus the $L^\infty$ bound is what makes it rigorous.) The mollifier argument of (ii), applied to $\varphi=\eta\_\varepsilon(x-\cdot)$, gives an alternative one-line proof of (i): $\eta\_\varepsilon\ast v=0$ on $\Omega^{\prime}$ and $\eta\_\varepsilon\ast v\to v$ in $L^1(\Omega^{\prime})$.
* Part (ii) is used in one space dimension in Example 2.8 ($\int v\varphi^{\prime}=0$ for all $\varphi$ forces $v$ to be constant), and in general to conclude that a function with vanishing weak gradient is constant on a connected domain.
* The fundamental lemma implies in particular that **weak derivatives are unique**: if $w\_1,w\_2$ are both weak derivatives of $u$, then $\int (w\_1-w\_2)\varphi=0$ for all $\varphi\in\mathcal D(\Omega)$, hence $w\_1=w\_2$ a.e.

</div>

## Chapter 2: Indirect Method

In this chapter we consider indirect methods for variational problems for functions of general dimension: instead of minimizing $I$ directly we study its critical points, i.e. the Euler–Lagrange equation.

### 2.1 Differentiability and first variation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setting</span><span class="math-callout__name">(The admissible class as an affine subspace)</span></p>

We want to define the derivative of functionals $I:\mathcal A\to\mathbb R$. We have in mind the case $\mathcal A=W^{1,p}\_g(\Omega)^m\subset W^{1,p}(\Omega)^m$. Note that $\mathcal A$ is an **affine subspace** of the real normed vector space $X=W^{1,p}(\Omega)^m$: if $u\_g\in\mathcal A$ is any fixed element then

$$
\mathcal A=u_g+X_0,\qquad X_0:=W^{1,p}_0(\Omega)^m ,
$$

so the "tangent space" of $\mathcal A$ at any of its points is $X\_0$. The derivative of $I$ at $u\in\mathcal A$ will be a **linear functional** on this tangent space. We recall that a linear operator $A:X\to Y$ between normed spaces is continuous if and only if $\lVert Ax\rVert\_Y\le C\lVert x\rVert\_X$ for some $C<\infty$.

In this section $\mathcal A\subset X$ is an affine subspace of a normed vector space $X$, which may be infinite-dimensional, and we write $X\_0$ for the subspace parallel to $\mathcal A$ (so that $u+tv\in\mathcal A$ for $u\in\mathcal A$, $v\in X\_0$). For $\mathcal A=X$ one has $X\_0=X$.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_affine_admissible_class.png' | relative_url }}" alt="Left: the affine subspace A parallel to the subspace X_0 in the plane X, a point u on A, a tangent direction v along A and a direction w leaving A. Right: functions on the unit interval with fixed boundary values; the variation tv vanishes at both ends." loading="lazy">
  <figcaption>(a) The admissible class $\mathcal A=u_g+X_0$ is a translate of the subspace $X_0=W^{1,p}_0(\Omega)^m$. Moving from $u\in\mathcal A$ in a direction $v\in X_0$ stays in $\mathcal A$; moving in a direction $w\notin X_0$ leaves it, so only directions in $X_0$ may be used to differentiate $I$ on $\mathcal A$. (b) Concretely, in one dimension: $u$ and $u+tv$ share the boundary data $g$ because the variation $v$ vanishes on $\partial\Omega$.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.1 (Differentiability)</span></p>

Let $\mathcal U\subset\mathcal A$ be open, $I:\mathcal U\to\mathbb R$ and $u\in\mathcal U$.

1. The **first variation** of $I$ at $u$ in direction $v\in X\_0$ is defined by (if it exists)

   $$\delta I[u,v]:=\frac{\mathrm{d}}{\mathrm{d}t}\Big|_{t=0} I[u+tv].$$

   Variations of order $k\in\mathbb N$ are defined iteratively: 
   
   $$\delta^2 I[u;v,w]:=\frac{\mathrm{d}}{\mathrm{d}t}\big\rvert_{t=0}\,\delta I[u+tw,v],$$
   
2. $I$ is **Gâteaux differentiable** at $u$ if $\delta I[u,v]$ exists for every $v\in X\_0$ and $I^{\prime}[u]:=\delta I[u,\cdot]$ defines a continuous *linear* functional on $X\_0$.

3. $I$ is **Fréchet differentiable** at $u$ if it is Gâteaux differentiable at $u$ and

   $$I[u+v]=I[u]+I'[u]v+o(\lVert v\rVert)\qquad\text{as } \lVert v\rVert\to0,\ v\in X_0.$$

   The space $C^k(\mathcal U)$ is the space of all $k$-times continuously Fréchet differentiable functions on $\mathcal U$.

When all variations exist as Fréchet derivatives, the first variation $\delta I[u,v]$ is linear in $v$ and the second variation $\delta^2 I[u;v,w]$ is bilinear (and symmetric) in $(v,w)$.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_differentiability_ladder.png' | relative_url }}" alt="Three panels: slopes of I along lines through u in several directions; the directional derivative as a function of the direction angle, a sinusoid for a Gateaux differentiable function versus cos cubed for a counterexample; a heat map of a function that is Gateaux but not Frechet differentiable, with a ridge along y equals x cubed." loading="lazy">
  <figcaption>The three levels of differentiability, illustrated on $X=\mathbb R^2$. (a) The first variation only looks along lines: $\delta I[u,v]$ is the slope at $t=0$ of $t\mapsto I[u+tv]$, one number per direction. (b) Gâteaux differentiability asks these slopes to assemble into a linear functional of $v$. Restricted to unit directions $v=(\cos\theta,\sin\theta)$, a linear functional is a pure sinusoid in $\theta$ (green); for $h(x,y)=x^3/(x^2+y^2)$ every directional derivative at $0$ exists, but $\delta h[0,v]=\cos^3\theta$ (red) is not linear, so $h$ is not Gâteaux differentiable. (c) Fréchet differentiability asks the remainder $I[u+v]-I[u]-I'[u]v$ to be $o(\lVert v\rVert)$ uniformly in the direction. For $g(x,y)=x^3y/(x^6+y^2)$ all directional derivatives at $0$ vanish, so $g$ is Gâteaux differentiable with $g'(0)=0$, yet $g\equiv\tfrac12$ along the ridge $y=x^3$: rays through the origin never see the ridge, a curve approaching the origin does. $g$ is not even continuous at $0$, let alone Fréchet differentiable.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The hierarchy, and what each level buys)</span></p>

$$
\text{Fréchet}\ \Longrightarrow\ \text{Gâteaux}\ \Longrightarrow\ \text{all first variations exist},
$$

and none of the arrows can be reversed (the figure above shows the two gaps). 
* The first variation is always homogeneous of degree one, $\delta I[u,\lambda v]=\lambda\,\delta I[u,v]$ for $\lambda\in\mathbb R$, but need not be additive in $v$; linearity is an extra requirement, which is why Definition 2.1(ii) demands it. 
* In the calculus of variations one mostly works with the weakest notion: 
  * the **Euler–Lagrange equation** (Theorem 2.3) only needs the first variation in every direction. 
  * Fréchet differentiability is what one needs for the **chain rule** and for **Taylor expansions** with a remainder that is uniform in the direction, as in Theorem 2.4(ii).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Fréchet vs. Gâteaux differentiability)</span></p>

Let $X$ be a Banach space and let $F:X\to\mathbb R$ be Fréchet differentiable at $x\in X$. 
1. Show that $F$ is continuous at $x$. 
2. Is Gâteaux differentiability at $x$ a sufficient condition for continuity at $x$? Give a reason for your answer.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

1. **Frechet differentiability at $u$ implies continuity at $u$.**

   Using Heine's sequential continuity definition, which coincides with the Cauchy's continuity definition on general metric spaces, and definition of Frechet differentiability,

   $$\lim_{v\to 0} F(u+v) = \limg_{v\to 0} \Bigl[ F(u) + F'(u)v + o(\lVert v \rVert)\Bigr] = F(u) \quad \forall v \in X.$$

   Thus $F$ is continuous at $u$.

2. **Gateaux differentiability at $u$ is not sufficient for continuity at $u$.**

   * No, consider the function of two variables $f(x,y)=1$ if $y=x^2 \neq 0$ and $0$ otherwise.
   * The function jumps on parabola / $\lbrace 0\rbrace$ and consider the zero point.
   * The line through the origin touches true parabola at at most two points: origin itself (always) and another point (at most once).
   * Consider ant fixed line through the origin that touches parabola at $y_0$.
   * Then the for all $t\in(-1, 1)$, the function $t \mapsto 0+ty_0 \equiv 0$, implying Gateaux differentiability at $0$.
   * The function $f$ is obviosuly discontinuous at $0$:
     * Consider the sequence $(x, x^2)$.
     * Then $\lim_{x\to 0} f(x,x^2) = 1 \neq 0 = f(0,0)$.

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.2 (Extrema, critical points)</span></p>

Let $\mathcal U\subset\mathcal A$ be open, $I:\mathcal U\to\mathbb R$ and $u\in\mathcal U$. Then $I$ has a

1. **global minimum** at $u$ if $I[u]\le I[v]$ for all $v\in\mathcal U$,
2. **local minimum** at $u$ if $I[u]\le I[v]$ for all $v\in\mathcal V$, for some neighborhood $\mathcal V$ of $u$.

Analogously we speak of global and local maxima and extrema.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_extrema.png' | relative_url }}" alt="Graph of cos(3 pi x)/x on the interval from 0.1 to 1.1 with its global maximum at the left endpoint, a global minimum, a local maximum and a local minimum marked" loading="lazy">
  <figcaption>Extrema of $f(x)=\cos(3\pi x)/x$ on $\tfrac1{10}\le x\le\tfrac{11}{10}$ (Figure 3 of the script). The global maximum sits at the boundary of the domain, where $f'\neq 0$: the necessary condition $f'=0$ of Theorem 2.3 applies only to extrema in the interior of an open set $\mathcal U$. This is the one-dimensional model for everything that follows, with $f$ replaced by a functional $I$ on an infinite-dimensional $\mathcal A$.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.3 (Euler–Lagrange equation)</span></p>

Let $\mathcal U\subset\mathcal A$ be open, $I:\mathcal U\to\mathbb R$ and $u\in\mathcal U$. If $I$ has a local extremum at $u$ and is Gâteaux differentiable at $u$, then

$$I'[u]=0 . \tag{2.1}$$

If (2.1) is well-defined and holds, we say that $u$ is a **critical point** of $I$. The equation (2.1) is also called the **Euler–Lagrange equation**.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.3</summary>

Fix $v\in X\_0$. Since $\mathcal U$ is open there is $\varepsilon\_0>0$ such that $u+tv\in\mathcal U$ for $\lvert t\rvert<\varepsilon\_0$, and by assumption the mapping $f:(-\varepsilon\_0,\varepsilon\_0)\to\mathbb R$, $f(t)=I[u+tv]$, is differentiable at $t=0$. Moreover, $f$ has a local extremum at $t=0$, so by the necessary condition for extrema of functions of one variable, $f^{\prime}(0)=\delta I[u,v]=0$. As $v$ was arbitrary, $I^{\prime}[u]=0$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(EL equation vs. general extremum equation)</span></p>

The Euler–Lagrange equation is **specific to variational problems (the calculus of variations)**, rather than a general equation for local extrema in standard multivariate calculus.

**Key Facts About the Euler–Lagrange Equation:**
* **Variational Origin:** It is the central equation of the calculus of variations, used to find functions (paths or fields) that make an action functional stationary (where the first variation equals zero).
* **Not Always a Minimum:** It only yields a stationary point, which can be a local minimum, a local maximum, or a saddle point (much like setting a single-variable derivative $f'(x) = 0$ in standard calculus does not guarantee a minimum). Additional tests, such as the second-variation or convexity conditions, are required to prove whether a solution is actually a minimum.
* **General Optimization vs. Variational Problems:** 
  * For standard finite-dimensional calculus (finding a local minimum of a function $f(x)$ on $\mathbb{R}^{n}$), the condition is $\nabla f(x) = 0$ and checking that the Hessian matrix is positive definite.
  * The Euler–Lagrange equation generalizes this idea of a derivative to infinite-dimensional function spaces (functionals), rather than serving as a general tool for every local minimum problem (such as constrained nonlinear programming or discrete optimization).
* **The Role of Euler–Lagrange:** It serves as a necessary condition (analogous to setting a derivative to zero) for a function to make a functional stationary (yielding a local minimum, local maximum, or saddle point)

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.4 (Minimizer and second variation)</span></p>

Let $X$ be a normed vector space, $\mathcal U\subset X$ open, $I\in C^2(\mathcal U)$ and $u\in\mathcal U$ a critical point of $I$.

1. If $u$ is a local minimizer of $I$, then $\delta^2 I[u;v,v]\ge 0$ for all $v\in X$.
2. If $\delta^2 I[u;v,v]\ge c\lVert v\rVert^2$ for all $v\in X$ and some $c>0$, then $u$ is a (strict) local minimizer.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.4</summary>

Let $\varepsilon>0$ with $B\_\varepsilon(u)\subset\mathcal U$ and let $h\in X$ with $\lVert h\rVert<\varepsilon$. By assumption $f(t):=I[u+th]$ defines a function $f\in C^2$ on the open interval $(-\varepsilon/\lVert h\rVert,\varepsilon/\lVert h\rVert)\supset[-1,1]$ with $f^{\prime}(0)=\delta I[u,h]=0$ and $f^{\prime\prime}(t)=\delta^2 I[u+th;h,h]$. The proof of (i) and (ii) then proceeds as in the one-dimensional case, via Taylor's formula with the Lagrange remainder: there is $\theta\in(0,1)$ with

$$
I[u+h]-I[u]=f(1)-f(0)=f'(0)+\tfrac12 f''(\theta)=\tfrac12\,\delta^2 I[u+\theta h;h,h]. \tag{T}
$$

**(i).** Suppose $\delta^2 I[u;v,v]<0$ for some $v\in X$. Apply (T) to $h=tv$: by continuity of the second derivative, $\delta^2 I[u+\theta tv;v,v]<0$ for $\lvert t\rvert$ small, so $I[u+tv]-I[u]=\tfrac{t^2}2\,\delta^2 I[u+\theta tv;v,v]<0$ for all small $t\neq0$, contradicting the local minimality of $u$.

**(ii).** Since $I\in C^2$, the map $w\mapsto\delta^2 I[w;\cdot,\cdot]$ is continuous into the bounded bilinear forms, so there is $\rho>0$ such that $\lvert\delta^2 I[w;h,h]-\delta^2 I[u;h,h]\rvert\le\tfrac c2\lVert h\rVert^2$ whenever $\lVert w-u\rVert<\rho$. For $\lVert h\rVert<\rho$, (T) gives

$$
I[u+h]-I[u]=\tfrac12\,\delta^2 I[u+\theta h;h,h]\ \ge\ \tfrac12\bigl(c-\tfrac c2\bigr)\lVert h\rVert^2=\tfrac c4\lVert h\rVert^2>0\qquad(h\neq0),
$$

so $u$ is a strict local minimizer. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why (ii) needs a uniform constant $c$ in infinite dimensions)</span></p>

In finite dimensions, $\delta^2 I[u;v,v]>0$ for all $v\neq0$ already implies $\delta^2 I[u;v,v]\ge c\lVert v\rVert^2$ (compactness of the unit sphere), hence a strict local minimum. In infinite dimensions this fails. On $X=\ell^2$ consider

$$
I[x]=\sum_{k\ge1}\Bigl(\frac{x_k^2}{k}-x_k^4\Bigr),\qquad \delta^2 I[0;v,v]=2\sum_k \frac{v_k^2}{k}>0\quad\text{for } v\neq0,
$$

so $0$ is a critical point with positive definite second variation. Yet along the coordinate directions $I[te\_k]=t^2/k-t^4<0$ as soon as $t^2>1/k$, and these $te\_k$ come arbitrarily close to $0$ as $k\to\infty$: $0$ is **not** a local minimizer. The infimum of $\delta^2 I[0;v,v]$ over the unit sphere is $0$, i.e. no $c>0$ exists. In the integral functionals below, the uniform constant corresponds to *coercivity* of the second variation, typically $\delta^2 I[u;\varphi,\varphi]\ge c\lVert\nabla\varphi\rVert\_{L^2}^2$.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_second_variation_coercivity.png' | relative_url }}" alt="The one-dimensional sections t squared over k minus t to the fourth for k equal to 1, 2, 5 and 20; the region where the section is positive shrinks to zero width as k grows" loading="lazy">
  <figcaption>The sections $t\mapsto I[te_k]=t^2/k-t^4$ of the $\ell^2$ example. Each section has a strict local minimum at $t=0$ (the second variation is positive in every direction), but its basin $\lvert t\rvert<(2k)^{-1/2}$, marked by the ticks, shrinks to nothing as $k\to\infty$. Positivity in every direction without a uniform lower bound $c\lVert v\rVert^2$ does not give a local minimum.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setting</span><span class="math-callout__name">(Integral functionals and growth conditions for the derivatives)</span></p>

In the following we consider, for $\Omega\Subset\mathbb R^n$, functionals $I:\mathcal A\to\mathbb R$ of the form

$$
I[u]=\int_\Omega f(x,u,Du)\,\mathrm{d}x, \tag{2.2}
$$

with $\mathcal A=W^{1,p}\_g(\Omega)^m$ or $\mathcal A=W^{1,p}(\Omega)^m$. We assume that $f\in C^1$ satisfies the growth condition

$$
\lvert f(x,z,\xi)\rvert\le C\bigl(1+\lvert z\rvert^p+\lvert\xi\rvert^p\bigr), \tag{2.3}
$$

so that $I$ is well-defined and finite on $W^{1,p}(\Omega)^m$. To compute the Euler–Lagrange equation we need one more growth condition, now on the first derivatives: for all $(x,z,\xi)\in\Omega\times\mathbb R^m\times\mathbb R^{m\times n}$

$$
\lvert\nabla_z f(x,z,\xi)\rvert+\lvert D_\xi f(x,z,\xi)\rvert\le C\bigl(1+\lvert z\rvert^{p-1}+\lvert\xi\rvert^{p-1}\bigr). \tag{2.4}
$$

Here $\nabla\_z f\in\mathbb R^m$ and $D\_\xi f\in\mathbb R^{m\times n}$ denote the partial derivatives with respect to $z$ and $\xi$, and $A:B=\operatorname{tr}(A^{\!\top}B)=A\_{\alpha i}B\_{\alpha i}$ is the Frobenius product. In general the Euler–Lagrange equation forms a **system of $m$ second-order partial differential equations**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.5 (Euler–Lagrange equation I)</span></p>

Let $\Omega\Subset\mathbb R^n$ be open and let $f\in C^2(\Omega\times\mathbb R^m\times\mathbb R^{m\times n})$ satisfy the growth conditions (2.3) and (2.4). Let $\mathcal A=W^{1,p}\_g(\Omega)^m$. Then for any critical point $u\in\mathcal A$ of (2.2) the Euler–Lagrange equation holds in the distributional (weak) sense,

$$
\int_\Omega \nabla_z f(\cdot,u,Du)\cdot\varphi+D_\xi f(\cdot,u,Du):D\varphi\,\mathrm{d}x=0\qquad\forall\varphi\in\mathcal D(\Omega)^m . \tag{2.5}
$$

If additionally $u\in C^2(\Omega)^m$, then

$$
\begin{cases}\nabla_z f(\cdot,u,Du)-\operatorname{div}_x\bigl(D_\xi f(\cdot,u,Du)\bigr)=0 & \text{in }\Omega,\\ u=g & \text{on }\partial\Omega .\end{cases} \tag{2.6}
$$

The boundary condition in (2.6) is called a **Dirichlet boundary condition**.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.5</summary>

Let $u\in\mathcal A$ be a critical point and $\varphi\in\mathcal D(\Omega)^m$. Then $u+\varepsilon\varphi\in\mathcal A$ for all $\varepsilon\in\mathbb R$, and

$$
\begin{aligned}
\delta I[u,\varphi]
&=\frac{\mathrm{d}}{\mathrm{d}\varepsilon}\Big|_{\varepsilon=0}\int_\Omega f(\cdot,u+\varepsilon\varphi,Du+\varepsilon D\varphi)\,\mathrm{d}x \\
&=\int_\Omega \frac{\mathrm{d}}{\mathrm{d}\varepsilon}\Big|_{\varepsilon=0} f(\cdot,u+\varepsilon\varphi,Du+\varepsilon D\varphi)\,\mathrm{d}x \\
&=\int_\Omega \partial_{z_\alpha}f(\cdot,u,Du)\,\varphi_\alpha+\partial_{\xi_{\alpha i}}f(\cdot,u,Du)\,\partial_i\varphi_\alpha\,\mathrm{d}x.
\end{aligned}
$$

The interchange of differentiation and integration is justified by dominated convergence: by the fundamental theorem of calculus the difference quotients are

$$
\begin{aligned}
&\frac{f(\cdot,u+\varepsilon\varphi,Du+\varepsilon D\varphi)-f(\cdot,u,Du)}{\varepsilon} \\
&= \int_0^1 \nabla_z f(\cdot,u+s\varepsilon\varphi,Du+s\varepsilon D\varphi)\cdot\varphi+D_\xi f(\cdot,u+s\varepsilon\varphi,Du+s\varepsilon D\varphi):D\varphi\,\mathrm{d}s ,
\end{aligned}
$$

and by (2.4) and $\lvert\varepsilon\rvert\le1$ the integrand is bounded by $C(1+\lvert u\rvert^{p-1}+\lvert Du\rvert^{p-1}+\lvert\varphi\rvert^{p-1}+\lvert D\varphi\rvert^{p-1})(\lvert\varphi\rvert+\lvert D\varphi\rvert)$, which lies in $L^1(\Omega)$ by Hölder's inequality with the exponents $\frac p{p-1}$ and $p$ (and because $\varphi$ is bounded). Since $u$ is a critical point, $\delta I[u,\varphi]=0$, which is (2.5).

For $u\in C^2(\Omega)^m$ the field $D\_\xi f(\cdot,u,Du)$ is $C^1$ in $\Omega$ (here $f\in C^2$ is used) and we may integrate by parts on the compact support of $\varphi$; no boundary terms appear because $\varphi=0$ near $\partial\Omega$:

$$
0=\delta I[u,\varphi]=\int_\Omega \Bigl[\partial_{z_\alpha}f(\cdot,u,Du)-\partial_i\bigl(\partial_{\xi_{\alpha i}}f(\cdot,u,Du)\bigr)\Bigr]\varphi_\alpha\,\mathrm{d}x .
$$

Since $\varphi\in\mathcal D(\Omega)^m$ is arbitrary, the fundamental lemma (Lemma 1.4) yields the first line of (2.6) pointwise; the second line is the definition of $\mathcal A$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Componentwise form, divergence of a matrix field)</span></p>

The divergence of a matrix field is applied **row-wise**, $(\operatorname{div}A)\_\alpha=\partial\_j A\_{\alpha j}$ (for dimensional reasons this is the only possibility when $m\neq n$). With Einstein's summation convention, summing over repeated indices $j\in\lbrace1,\dots,n\rbrace$ and $\alpha\in\lbrace1,\dots,m\rbrace$, the system (2.6) reads

$$
-\partial_j\Bigl(\partial_{\xi_{\alpha j}}f\bigl(x,u(x),Du(x)\bigr)\Bigr)+\partial_{z_\alpha}f\bigl(x,u(x),Du(x)\bigr)=0\qquad\text{for all } x\in\Omega,\ \alpha=1,\dots,m .
$$

If the critical point $u$ is sufficiently regular, then by the fundamental lemma the Euler–Lagrange equation holds in the classical sense; for a general $u\in W^{1,p}$ only the weak form (2.5) is available, and it is the weak form that carries all the information.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The Dirichlet energy with a source term)</span></p>

Let $\Omega\subset\mathbb R^d$ be an open and bounded set, $X:=W^{1,2}(\Omega)$ and $f\in L^2(\Omega)$. Furthermore, let $E:X\to\mathbb R$ be given by

$$
E[u]:=\int_\Omega\Bigl(\frac12\lvert\nabla u\rvert^2-f(x)\,u(x)\Bigr)\mathrm{d}x\qquad\text{for all } u\in X .
$$

* **(a)** Let $v\in X$. Compute the first variation of $E$ in direction $v$.
* **(b)** Show that $E$ is Gâteaux differentiable at every $u\in X$.
* **(c)** Show that $E$ is Fréchet differentiable at every $u\in X$.
* **(d)** Show that $\delta E:X\to X^\ast$ is continuous.
* **(e)** Identify the Euler–Lagrange equation for $E\|\_{W\_0^{1,2}(\Omega)}:W\_0^{1,2}(\Omega)\to\mathbb R$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

TODO:

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.6 (Euler–Lagrange equation II, natural boundary conditions)</span></p>

Let $\mathcal A=W^{1,p}(\Omega)^m$ (no condition on the boundary) and let $\Omega$ have $C^1$ boundary with outer normal $\nu$. Under the hypotheses of Theorem 2.5 the Euler–Lagrange equation for a critical point $u\in\mathcal A$ takes the form

$$
\int_\Omega \nabla_z f(\cdot,u,Du)\cdot\varphi+D_\xi f(\cdot,u,Du):D\varphi\,\mathrm{d}x=0\qquad\forall\varphi\in C^\infty(\overline\Omega)^m .
$$

If additionally $u\in C^2(\overline\Omega)^m$, then

$$
\begin{cases}\nabla_z f(\cdot,u,Du)-\operatorname{div}_x\bigl(D_\xi f(\cdot,u,Du)\bigr)=0 & \text{in }\Omega,\\ D_\xi f(\cdot,u,Du)\,\nu=0 & \text{on }\partial\Omega .\end{cases} \tag{2.7}
$$

The boundary condition in (2.7) is called a **Neumann** or **natural boundary condition**. If no condition is imposed on the boundary (or on part of it), one also speaks of **free boundary conditions**.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.6</summary>

Let $\varphi\in C^\infty(\overline\Omega)^m$. Now $u+\varepsilon\varphi\in\mathcal A=W^{1,p}(\Omega)^m$ for every $\varepsilon$, so exactly as in the previous proof

$$
0=\int_\Omega \nabla_z f(\cdot,u,Du)\cdot\varphi+D_\xi f(\cdot,u,Du):D\varphi\,\mathrm{d}x .
$$

Assume now $u\in C^2(\overline\Omega)^m$, so that integration by parts is permitted. For a $C^1$ matrix field $A$ and a vector field $\varphi$ one has the product rule

$$
\operatorname{div}\bigl(A^{\!\top}\varphi\bigr)=\partial_j(A_{\alpha j}\varphi_\alpha)=(\operatorname{div}A)\cdot\varphi+A:D\varphi ,
$$

which with $A=D\_\xi f(\cdot,u,Du)$ and the divergence theorem gives

$$
0=\int_\Omega \Bigl(\nabla_z f(\cdot,u,Du)-\operatorname{div}\bigl(D_\xi f(\cdot,u,Du)\bigr)\Bigr)\cdot\varphi\,\mathrm{d}x+\int_{\partial\Omega}\bigl(D_\xi f(\cdot,u,Du)\,\nu\bigr)\cdot\varphi\,\mathrm{d}S .
$$

First choose $\varphi\in\mathcal D(\Omega)^m$: the boundary integral vanishes and the fundamental lemma gives the first equation of (2.7). Knowing this, the volume integral vanishes for *every* $\varphi\in C^\infty(\overline\Omega)^m$, so the boundary integral vanishes for every $\varphi$; since the restrictions $\varphi\|\_{\partial\Omega}$ can be chosen arbitrarily (within smooth functions on $\partial\Omega$) and $D\_\xi f(\cdot,u,Du)\nu$ is continuous, the second equation of (2.7) follows. $\square$

</details>
</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_boundary_conditions.png' | relative_url }}" alt="Two panels showing the minimizer of the integral of one half u prime squared minus u, once with both ends pinned to zero and once with the right end free; competitors have larger energy, and at the free end the minimizer has horizontal tangent" loading="lazy">
  <figcaption>The same integrand $f(z,\xi)=\tfrac12\xi^2-z$ on $\Omega=(0,1)$, two admissible classes. (a) With both endpoints prescribed, the Euler–Lagrange equation $-u''=1$ comes with the Dirichlet data $u(0)=u(1)=0$, and competitors $u+t\varphi$ must vanish at both ends. (b) If the right endpoint is left free, the test functions may be nonzero at $x=1$, and the extra boundary term in the first variation forces the natural condition $D_\xi f\,\nu=u'(1)=0$: the minimizer "chooses" a horizontal tangent at the free end by itself. In both cases every competitor has strictly larger energy than the minimizer (the values of $I$ are listed in the legends).</figcaption>
</figure>

<div class="math-callout math-callout--example" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">2.7 (Dirichlet integral)</span></p>

Let $\Omega\Subset\mathbb R^n$ be smooth, $m=1$, and

$$
I[u]=\frac12\int_\Omega \lvert\nabla u\rvert^2\,\mathrm{d}x .
$$

Then a critical point $u\in C^2(\overline\Omega)$ satisfies the boundary value problem

$$
\begin{cases}\Delta u=0 & \text{in }\Omega\\ u=g & \text{on }\partial\Omega\end{cases}\quad\text{if }\mathcal A=W^{1,2}_g(\Omega),\qquad\qquad
\begin{cases}\Delta u=0 & \text{in }\Omega\\ \partial_\nu u=0 & \text{on }\partial\Omega\end{cases}\quad\text{if }\mathcal A=W^{1,2}(\Omega).
$$

*Proof.* With the notation of Theorems 2.5 and 2.6, $f(\xi)=\frac12\lvert\xi\rvert^2$, so $\nabla\_z f=0$ and $D\_\xi f(\xi)=\xi$, and $f$ satisfies the growth conditions with $p=2$, so the integral is well-defined on $W^{1,2}(\Omega)$. The Euler–Lagrange equation $-\operatorname{div}(\nabla u)=0$ is Laplace's equation, and the natural boundary condition $D\_\xi f(\nabla u)\cdot\nu=\nabla u\cdot\nu=0$ is the homogeneous Neumann condition. For a general critical point $u\in W^{1,2}$ the equations hold in the weak sense. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Euler–Lagrange equations of three classical functionals)</span></p>

Compute (without worrying about technicalities like interchanging limits and regularity of the functions) the first variation and the induced partial differential equation of the following functionals:

$$
I_1[\gamma]:=\int_0^1\lvert\dot\gamma(t)\rvert\,\mathrm{d}t\qquad\text{where }\gamma:[0,1]\to M\text{ and }M\subset\mathbb R^d\text{ a manifold}, \tag{1}
$$

$$
I_2[u]:=\int_\Omega\sqrt{1-\lvert\nabla u\rvert^2}\,\mathrm{d}x\qquad\text{where }u:\Omega\to\mathbb R\text{ and }\Omega\subset\mathbb R^d\text{ an open and bounded set}, \tag{2}
$$

$$
I_3[u]:=\int_0^T\!\!\int_\Omega-\frac12\lvert\partial_tu\rvert^2+\frac12\lvert\nabla_xu\rvert^2\,\mathrm{d}x\,\mathrm{d}t\qquad\text{where }u:(0,T)\times\Omega\to\mathbb R . \tag{3}
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

TODO:

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setting</span><span class="math-callout__name">(The one-dimensional case)</span></p>

Next we consider the one-dimensional case $n=1$, i.e. $I:\mathcal A\to\mathbb R$ with

$$
I[u]=\int_a^b f(x,u,u')\,\mathrm{d}x,\qquad W^{1,p}_g([a,b])=\bigl\lbrace u\in W^{1,p}([a,b]):\ u=g\text{ on }\lbrace a,b\rbrace\bigr\rbrace ,
$$

with $\mathcal A=W^{1,p}\_g([a,b])^m$ or $\mathcal A=W^{1,p}([a,b])^m$. The following example shows that the **convexity of $f$** plays an important role for existence, uniqueness and regularity of minimizers.

</div>

<div class="math-callout math-callout--example" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">2.8 (Non-convex integrand I)</span></p>

Let $I:W^{1,4}\_0([-1,1])\to\mathbb R$ be given by

$$
I[u]=\int_{-1}^1 f(u')\,\mathrm{d}x=\int_{-1}^1 \bigl(u'^2-1\bigr)^2\,\mathrm{d}x,\qquad f(\xi)=(\xi^2-1)^2 .
$$

The Euler–Lagrange equation is

$$
\frac{\mathrm{d}}{\mathrm{d}x}\Bigl(4\,(u'^2-1)\,u'\Bigr)=0\qquad\text{for a.e. } x\in(-1,1).
$$

Then the following holds:

1. **No uniqueness.** Every function $u\in W^{1,4}\_0([-1,1])$ with $\lvert u^{\prime}\rvert=1$ a.e. is a minimizer, because $I\ge0$ and $I[u]=0$ for such $u$. There are uncountably many of them (sawtooth functions with arbitrary corner positions).
2. **No regularity.** No minimizer belongs to $C^2([-1,1])$, even though $f$ is smooth. In fact no minimizer is even $C^1$: a $C^1$ function with $\lvert u^{\prime}\rvert=1$ everywhere has $u^{\prime}\equiv1$ or $u^{\prime}\equiv-1$ by continuity, so $u(1)-u(-1)=\pm2\neq0$, contradicting the boundary condition.

**Question.** What is the essential property of $f$ which leads to the above issues?

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_double_well.png' | relative_url }}" alt="Left: the double-well integrand with its two wells at plus and minus one and the concave bump at zero. Right: three sawtooth functions with slopes plus or minus one, all with zero energy, and the zero function, which solves the Euler-Lagrange equation but has energy two" loading="lazy">
  <figcaption>Example 2.8. (a) The integrand $f(\xi)=(\xi^2-1)^2$ is a double well: the chord between the wells $\xi=\pm1$ lies below the graph, so $f$ is not convex, and $f''(0)=-4<0$. (b) Any sawtooth with slopes $\pm1$ and zero boundary values has $I[u]=0=\min I$, so there are uncountably many minimizers and none is $C^1$. The only $C^2$ solution of the Euler–Lagrange equation with the right boundary values is $u\equiv0$ (its slope must be a constant root of $4(s^2-1)s=c$, and the boundary values force the constant to be $0$), and this critical point has $I[0]=2$: it is not a minimizer, and indeed $\delta^2 I[0;\varphi,\varphi]=-4\int\varphi'^2<0$.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reading the Euler–Lagrange equation of Example 2.8 with Du Bois-Reymond)</span></p>

The weak form (2.5) reads $\int\_{-1}^1 4(u^{\prime 2}-1)u^{\prime}\,\varphi^{\prime}\,\mathrm{d}x=0$ for all $\varphi\in\mathcal D((-1,1))$. By Lemma 1.4(ii) this is equivalent to $4(u^{\prime 2}-1)u^{\prime}=c$ a.e. for some constant $c$: the derivative $u^{\prime}$ takes values in the (at most three) roots of a cubic. The minimizers with $\lvert u^{\prime}\rvert=1$ realize $c=0$ with $u^{\prime}$ jumping between the two roots $\pm1$, which a continuous $u^{\prime}$ cannot do. The non-convexity of $f$ is what makes the cubic $\xi\mapsto f^{\prime}(\xi)$ non-monotone, i.e. what allows several roots. We will see that the answer to the Question above is exactly the **non-convexity of the integrand**; the next theorem makes the connection precise through the second variation.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption</span><span class="math-callout__name">(Growth condition for the second derivatives)</span></p>

To study the second variation of $I$ we need a growth condition for the second derivatives of the integrand: for all $(x,z,\xi)\in\Omega\times\mathbb R^m\times\mathbb R^{m\times n}$

$$
\lvert D_z^2 f(x,z,\xi)\rvert+\lvert D_\xi^2 f(x,z,\xi)\rvert+\lvert D^2_{z,\xi}f(x,z,\xi)\rvert\le C\bigl(1+\lvert z\rvert^{p-2}+\lvert\xi\rvert^{p-2}\bigr). \tag{2.8}
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.9 (Second variation, Legendre–Hadamard condition)</span></p>

Let $\Omega\Subset\mathbb R^n$ be open and let $f\in C^3(\Omega\times\mathbb R^m\times\mathbb R^{m\times n})$ satisfy the growth conditions (2.3), (2.4) and (2.8). Let $\mathcal A=W^{1,p}\_g(\Omega)^m$. Then for any minimizer $u\in\mathcal A$ of (2.2) we have

$$
\int_\Omega \varphi\cdot D_z^2 f(X)\,\varphi+2\,\varphi\cdot\nabla_z D_\xi f(X):D\varphi+D_\xi^2 f(X):(D\varphi\otimes D\varphi)\,\mathrm{d}x\ \ge\ 0 \tag{2.9}
$$

for all $\varphi\in\mathcal D(\Omega)^m$, where $X:=(x,u(x),Du(x))$. In particular,

$$
D_\xi^2 f(X):\bigl((a\otimes b)\otimes(a\otimes b)\bigr)\ \ge\ 0\qquad\forall a\in\mathbb R^m,\ b\in\mathbb R^n,\ \text{a.e. } x\in\Omega .
$$

Here we use the notation

* $\varphi\cdot\nabla\_z D\_\xi f(X):D\varphi=\varphi\_\alpha\,\partial\_{z\_\alpha}\partial\_{\xi\_{\beta j}}f(X)\,\partial\_j\varphi\_\beta$,
* $D\_\xi^2 f(X):(A\otimes A)=\partial^2\_{\xi\_{\alpha i}\xi\_{\beta j}}f(X)\,A\_{\alpha i}A\_{\beta j}$ for $A\in\mathbb R^{m\times n}$,
* $(a\otimes b)\_{\alpha i}=a\_\alpha b\_i$, the rank-one matrix with image $\operatorname{span}\langle a\rangle$ and kernel $b^\perp$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.9</summary>

**Step 1: the second variation.** For $\varphi\in\mathcal D(\Omega)^m$ and $t\in\mathbb R$ we have, as in the proof of Theorem 2.5,

$$
\frac{\mathrm{d}}{\mathrm{d}t}I[u+t\varphi]=\int_\Omega \partial_{z_\alpha}f(x,u+t\varphi,\nabla u+t\nabla\varphi)\,\varphi_\alpha+\partial_{\xi_{\alpha i}}f(x,u+t\varphi,\nabla u+t\nabla\varphi)\,\partial_i\varphi_\alpha\,\mathrm{d}x .
$$

Differentiating once more and evaluating at $t=0$ (with $f:=f(x,u,\nabla u)$) gives

$$
\frac{\mathrm{d}^2}{\mathrm{d}t^2}\Big|_{t=0} I[u+t\varphi]=\int_\Omega \partial^2_{z_\alpha z_\beta}f\,\varphi_\alpha\varphi_\beta+2\,\partial_{z_\alpha}\partial_{\xi_{\beta j}}f\,\varphi_\alpha\partial_j\varphi_\beta+\partial^2_{\xi_{\alpha i}\xi_{\beta j}}f\,\partial_i\varphi_\alpha\partial_j\varphi_\beta\,\mathrm{d}x . \tag{2.10}
$$

With the growth condition (2.8) the integral is well-defined and induces a continuous bilinear form (the justification of the differentiation under the integral is as before, now using (2.8) and Hölder). From Theorem 2.4(i) applied to the minimizer $u$ we get $\delta^2 I[u;\varphi,\varphi]\ge0$, i.e. (2.9).

**Step 2: a smart test function.** The idea is now to choose a test function in (2.9) that extracts *pointwise* information about $f$ evaluated on the solution. For $\varepsilon>0$, $\psi\in\mathcal D(\Omega)$, $b\in\mathbb R^n$, $a\in\mathbb R^m$ define the family $\varphi^{(\varepsilon)}\in\mathcal D(\Omega)^m$ by

$$
\varphi^{(\varepsilon)}(x):=\varepsilon\,\psi(x)\,a\,\sin\Bigl(\frac{x\cdot b}{\varepsilon}\Bigr).
$$

The cut-off $\psi$ ensures the correct (zero) boundary data; otherwise this is a highly oscillatory function, reminiscent of the sawtooth minimizers of the previous example. By definition $\lVert\varphi^{(\varepsilon)}\rVert\_{L^\infty}\to0$ as $\varepsilon\to0$, while

$$
\partial_i\varphi^{(\varepsilon)}_\alpha(x)=\psi(x)\,a_\alpha b_i\cos\Bigl(\frac{x\cdot b}{\varepsilon}\Bigr)+\varepsilon\,\partial_i\psi(x)\,a_\alpha\sin\Bigl(\frac{x\cdot b}{\varepsilon}\Bigr),
$$

so that $\lVert D\varphi^{(\varepsilon)}\rVert\_{L^\infty}\le C$ uniformly in $\varepsilon$. Inserting $\varphi^{(\varepsilon)}$ into (2.10), the first two terms vanish in the limit (one factor tends to $0$ uniformly, the other factor is bounded, and the coefficients $D\_z^2f(X)$, $\nabla\_zD\_\xi f(X)$ are in $L^1$ by (2.8)), so

$$
0\le\liminf_{\varepsilon\to0}\int_\Omega \partial^2_{z_\alpha z_\beta}f\,\varphi^{(\varepsilon)}_\alpha\varphi^{(\varepsilon)}_\beta+2\,\partial_{z_\alpha}\partial_{\xi_{\beta j}}f\,\varphi^{(\varepsilon)}_\alpha\partial_j\varphi^{(\varepsilon)}_\beta+\partial^2_{\xi_{\alpha i}\xi_{\beta j}}f\,\partial_i\varphi^{(\varepsilon)}_\alpha\partial_j\varphi^{(\varepsilon)}_\beta\,\mathrm{d}x
=\liminf_{\varepsilon\to0}\int_\Omega \partial^2_{\xi_{\alpha i}\xi_{\beta j}}f\,\partial_i\varphi^{(\varepsilon)}_\alpha\partial_j\varphi^{(\varepsilon)}_\beta\,\mathrm{d}x .
$$

**Step 3: the weak limit of the oscillating product.** We need the weak limit of $\partial\_i\varphi^{(\varepsilon)}\_\alpha\partial\_j\varphi^{(\varepsilon)}\_\beta$, which is given by the mean value of $\cos^2$:

$$
\partial_i\varphi^{(\varepsilon)}_\alpha\,\partial_j\varphi^{(\varepsilon)}_\beta=\psi(x)^2\,a_\alpha a_\beta b_i b_j\cos^2\Bigl(\frac{x\cdot b}{\varepsilon}\Bigr)+O(\varepsilon)\ \overset{\ast}{\rightharpoonup}\ \frac12\,\psi^2\,(a\otimes b)\otimes(a\otimes b)\qquad\text{in } L^\infty(\Omega),
$$

because a bounded periodic function converges weakly-$\ast$ to its mean (Riemann–Lebesgue). Since $\partial^2\_{\xi\xi}f(X)\in L^1(\Omega)$, weak-$\ast$ convergence of the test factor in $L^\infty$ suffices to pass to the limit, and we obtain

$$
0\le\frac12\int_\Omega \psi(x)^2\,D_\xi^2 f(X):\bigl((a\otimes b)\otimes(a\otimes b)\bigr)\,\mathrm{d}x\qquad\forall\psi\in\mathcal D(\Omega).
$$

Since $\psi^2$ can approximate the indicator function of any set $\lbrace D\_\xi^2 f(X):((a\otimes b)\otimes(a\otimes b))<-\delta\rbrace\cap K$, $K\Subset\Omega$, the integrand must be $\ge0$ a.e., which is the claim. $\square$

</details>
</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_oscillating_test_function.png' | relative_url }}" alt="Left: the oscillating test function for two values of epsilon inside the envelope plus or minus epsilon psi; the amplitude shrinks while the slopes stay of order one. Right: the squared slope, its local average, and the weak limit one half psi squared." loading="lazy">
  <figcaption>The test function of Theorem 2.9 in one dimension, $\varphi^{(\varepsilon)}=\varepsilon\psi\sin(x/\varepsilon)$. (a) Its amplitude is $O(\varepsilon)$, so all terms of (2.10) containing $\varphi^{(\varepsilon)}$ itself disappear, but its slope $\psi\cos(x/\varepsilon)+O(\varepsilon)$ stays of order one. (b) The product of slopes oscillates like $\psi^2\cos^2(x/\varepsilon)$; integrated against any $L^1$ coefficient only its local mean $\tfrac12\psi^2$ survives. Choosing $\psi$ concentrated near a point localizes the inequality and yields the pointwise Legendre–Hadamard condition. This is the first appearance of a theme that runs through the whole course: fine-scale oscillations and their weak limits.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(Growth conditions)</span></p>

Check that the growth conditions (2.3), (2.4), (2.8) are the right conditions to ensure that all the integrals above are finite and that differentiation and integration can be exchanged. (Hint: for (2.8) one needs $p\ge2$ so that $\lvert Du\rvert^{p-2}\in L^{p/(p-2)}\subset L^1(\Omega)$; the test functions and their gradients are bounded.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rank-one directions and convexity along lines)</span></p>

For $F=F(\xi)$ and $\phi(t)=F(\xi+tA)$ we have

$$
\frac{\mathrm{d}^2}{\mathrm{d}t^2}\Big|_{t=0}\phi(t)=D_\xi^2F(\xi):(A\otimes A).
$$

Matrices of the form $a\otimes b$ have a one-dimensional image: they are precisely the **rank-one matrices**, $A(\mathbb R^n)=\operatorname{span}\langle a\rangle$ and $\ker A=b^\perp$. Indeed,

$$
\bigl\lbrace a\otimes b:\ a\in\mathbb R^m,\ b\in\mathbb R^n\bigr\rbrace=\bigl\lbrace A\in\mathbb R^{m\times n}:\ \operatorname{rank}A=1\bigr\rbrace .
$$

The condition $D\_\xi^2F(\xi):(A\otimes A)\ge0$ for all rank-one matrices $A$ and all $\xi$ hence means that the restriction of $F$ to **lines with rank-one tangent directions** is convex. Theorem 2.9 gives this condition only at the single point $\xi=Du(x)$: at a minimizer, for a.e. $x$, the second derivative of $\xi\mapsto f(x,u(x),\xi)$ at $\xi=Du(x)$ is non-negative in every rank-one direction (the **Legendre–Hadamard condition**), i.e. $f$ is rank-one convex *to second order at $Du(x)$*. We therefore define:

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.10 (Rank-1 convex and rank-1 affine functions)</span></p>

1. A function $f:\mathbb R^{m\times n}\to\mathbb R\cup\lbrace+\infty\rbrace$ is called **rank-1 convex** if

   $$
   f\bigl(\theta A+(1-\theta)B\bigr)\le\theta f(A)+(1-\theta)f(B)\qquad\text{whenever } \operatorname{rank}(A-B)=1,\ \theta\in[0,1]. \tag{2.11}
   $$

2. The function is called **rank-1 affine** if equality holds in (2.11) (for all such $A,B,\theta$).

If $m=1$ or $n=1$ then *every* nonzero matrix has rank one, so rank-1 convexity and convexity coincide.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_rank_one_cone.png' | relative_url }}" alt="Left: a double cone in three-dimensional space representing the rank-one symmetric two by two matrices, with a rank-one direction on the cone and the identity direction inside it. Right: the determinant along the rank-one line is a straight line, along the identity direction a parabola." loading="lazy">
  <figcaption>Rank-one directions are a thin set. (a) In the three-dimensional space of symmetric $2\times2$ matrices $\xi$ (entries $\xi_{11}$, $\xi_{12}=\xi_{21}$, $\xi_{22}$), with coordinates $p=(\xi_{11}+\xi_{22})/2$, $q=(\xi_{11}-\xi_{22})/2$, $b=\xi_{12}$, the rank-one matrices form the double cone $\det\xi=p^2-q^2-b^2=0$. Rank-1 convexity only constrains $f$ along lines whose direction lies on this cone (such as $A=a\otimes a$), not along directions like $B=\mathrm{Id}$ inside it. (b) The determinant is the model example: $\det(\xi+tA)=\det\xi+t\,\mathrm{cof}\,\xi:A+t^2\det A$ is affine in $t$ exactly when $\det A=0$, so $\det$ is rank-1 affine but, being quadratic along rank-2 directions, neither convex nor concave.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Summary</span><span class="math-callout__name">(What we have learned from the second variation)</span></p>

* If $m=1$ or $n=1$, rank-1 convexity is the same as convexity. In this case, for any minimizer $u$ of the functional $I$, the function $\xi\mapsto f(x,u(x),\xi)$ is locally convex at $\xi=Du(x)$ with respect to variations in the $\xi$ direction (in the sense of the second-order condition $D^2\_\xi f(X)\ge0$).
* If $m>1$ and $n>1$, the suitable generalization of convexity in this context is **rank-1 convexity** (the Legendre–Hadamard condition). The relation between rank-1 convexity, the intermediate notion of *quasiconvexity* (convex $\Rightarrow$ quasiconvex $\Rightarrow$ rank-1 convex) and the lower semicontinuity of $I$ is the subject of the direct method in the next chapter.

</div>

### 2.2 Weak continuity and Brouwer's Fixed Point Theorem

We continue to investigate the relation between the form of the integrand $f$ and the minimization of the functional $I$. This leads to the peculiar notion of **Null Lagrangians**: integrands for which *every* function solves the Euler–Lagrange equation. This notion allows us to learn more about fine-scale oscillations and their relation to weak convergence, and it yields a proof of **Brouwer's fixed point theorem**.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setting</span></p>

We again consider functionals $I:\mathcal A\to\mathbb R$ of the form

$$
I[u]=\int_\Omega f(x,u,Du)\,\mathrm{d}x ,
$$

with $\mathcal A=W^{1,p}\_g(\Omega)^m$ or $W^{1,p}(\Omega)^m$, where $\Omega\Subset\mathbb R^n$ is smooth with outer normal $\nu$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">2.11 (Affine integrands)</span></p>

Let $\Omega\Subset\mathbb R^n$ with $\partial\Omega\in C^1$ and outer normal $\nu$, and let $f:\mathbb R^{m\times n}\to\mathbb R$ be affine, $f(\xi):=A:\xi+c$ with $A\in\mathbb R^{m\times n}$, $c\in\mathbb R$. Since $A:Du=\partial\_j(A\_{\alpha j}u\_\alpha)=\operatorname{div}(A^{\!\top}u)$, the divergence theorem gives for $u\in W^{1,1}\_g(\Omega)^m$

$$
I[u]=\int_\Omega A:Du+c\,\mathrm{d}x=\int_\Omega \operatorname{div}(A^{\!\top}u)\,\mathrm{d}x+c\lvert\Omega\rvert=\int_{\partial\Omega} g\cdot A\nu\,\mathrm{d}S+c\lvert\Omega\rvert .
$$

Thus for affine $f=f(\xi)$ the integral $I$ is **constant on $W^{1,1}\_g(\Omega)^m$**: it depends only on the boundary values.

</div>

We want to understand more generally the class of integrands for which the integral depends only on the boundary values. We have learned that the integrand is necessarily rank-1 convex in $\xi$ in the neighborhood of a minimizer; to avoid issues of non-integrability the "boundary-values-only" property is best formulated in terms of the Euler–Lagrange equation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.12 (Null Lagrangian)</span></p>

A function $f\in C^1(\Omega\times\mathbb R^m\times\mathbb R^{m\times n})$ is a **Null Lagrangian** if

$$
\nabla_z f(\cdot,u,Du)=\operatorname{div}_x\bigl(D_\xi f(\cdot,u,Du)\bigr)\quad\text{in }\Omega\qquad\forall u\in C^2(\overline\Omega)^m .
$$

A Null Lagrangian is hence an integrand $f$ such that **every** sufficiently smooth function $u$ solves the Euler–Lagrange equation.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">2.13 (Null Lagrangians depend only on boundary values)</span></p>

Let $\Omega\Subset\mathbb R^n$. For $f\in C^2$ the following statements are equivalent:

1. $f$ is a Null Lagrangian,
2. $I[u]=I[v]$ for all $u,v\in C^2(\overline\Omega)^m$ with $u=v$ on $\partial\Omega$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 2.13</summary>

**(i) $\Rightarrow$ (ii).** Let $u,v\in C^2(\overline\Omega)^m$ with $u=v$ on $\partial\Omega$ and set $w\_t:=(1-t)u+tv$ and $h(t):=I[w\_t]$. Differentiating under the integral and integrating by parts (the boundary term vanishes since $v-u=0$ on $\partial\Omega$),

$$
h'(t)=\int_\Omega \nabla_z f(\cdot,w_t,Dw_t)\cdot(v-u)+D_\xi f(\cdot,w_t,Dw_t):D(v-u)\,\mathrm{d}x
=\int_\Omega \Bigl(\nabla_z f(\cdot,w_t,Dw_t)-\operatorname{div}D_\xi f(\cdot,w_t,Dw_t)\Bigr)\cdot(v-u)\,\mathrm{d}x=0,
$$

because $w\_t\in C^2(\overline\Omega)^m$ solves the Euler–Lagrange equation. Hence $I[u]=h(0)=h(1)=I[v]$.

**(ii) $\Rightarrow$ (i).** Let $u\in C^2(\overline\Omega)^m$. For any $t\in\mathbb R$ and $\varphi\in\mathcal D(\Omega)^m$, the function $u+t\varphi$ has the same boundary values as $u$, so by (ii) $t\mapsto I[u+t\varphi]$ is constant and $\delta I[u,\varphi]=\frac{\mathrm{d}}{\mathrm{d}t}\big\|\_{t=0}I[u+t\varphi]=0$: $u$ is a minimizer (and maximizer) with respect to its own boundary values. Since $\varphi$ is arbitrary, $u$ satisfies the weak Euler–Lagrange equation, and since $u\in C^2$ also the strong one, by Theorem 2.5. $\square$

</details>
</div>

We now ask ourselves whether there are Null Lagrangians $f=f(\xi)$ which are not affine.

<div class="math-callout math-callout--example" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">2.14 (The $2\times2$ determinant is a Null Lagrangian)</span></p>

Let $f(\xi)=\det\xi$ for $m=n=2$ and $u\in C^2(\overline\Omega)^2$. We calculate, using that second derivatives commute,

$$
\det Du=\partial_1u_1\,\partial_2u_2-\partial_1u_2\,\partial_2u_1=\partial_1\bigl(u_1\partial_2u_2\bigr)-\partial_2\bigl(u_1\partial_1u_2\bigr)=\operatorname{div}_x\begin{pmatrix}u_1\partial_2u_2\\ -u_1\partial_1u_2\end{pmatrix}.
$$

The $2\times2$ determinant can hence be written in **divergence form**. In a more symmetric way,

$$
\det Du=\frac12\operatorname{div}_x\left[\begin{pmatrix}u_1\partial_2u_2\\ -u_1\partial_1u_2\end{pmatrix}+\begin{pmatrix}-u_2\partial_2u_1\\ u_2\partial_1u_1\end{pmatrix}\right]
=\frac12\operatorname{div}_x\left[\begin{pmatrix}\partial_2u_2 & -\partial_2u_1\\ -\partial_1u_2 & \partial_1u_1\end{pmatrix}\begin{pmatrix}u_1\\ u_2\end{pmatrix}\right].
$$

From the divergence structure we obtain that the integral of the determinant is a boundary integral:

$$
\int_\Omega \det Du\,\mathrm{d}x=\frac12\int_{\partial\Omega}\left[\begin{pmatrix}\partial_2u_2 & -\partial_2u_1\\ -\partial_1u_2 & \partial_1u_1\end{pmatrix}\begin{pmatrix}u_1\\ u_2\end{pmatrix}\right]\cdot\nu\,\mathrm{d}S .
$$

The matrix appearing here is the **transpose of the cofactor matrix** of $Du$ (see below), so that the integrand on the right is $\tfrac12\,u\cdot\operatorname{cof}(Du)\,\nu$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cofactor matrix)</span></p>

The **cofactor matrix** of $A\in\mathbb R^{n\times n}$ is given in terms of its coefficients by

$$
(\operatorname{cof}A)_{ij}:=(-1)^{i+j}\det A^{ij},
$$

where the matrix $A^{ij}\in\mathbb R^{(n-1)\times(n-1)}$ is obtained by deleting the $i$-th row and $j$-th column of $A$. (The transpose $(\operatorname{cof}A)^{\!\top}$ is the adjugate matrix.) For $n=2$:

$$
\operatorname{cof}\begin{pmatrix}a&b\\ c&d\end{pmatrix}=\begin{pmatrix}d&-c\\ -b&a\end{pmatrix}.
$$

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">2.15 (Cofactor matrix and determinant)</span></p>

For $A\in\mathbb R^{n\times n}$ we have

1. $A^{\!\top}\operatorname{cof}A=(\det A)\,\mathrm{Id}$ (and likewise $(\operatorname{cof}A)A^{\!\top}=(\det A)\,\mathrm{Id}$),
2. $\dfrac{\partial(\det A)}{\partial A\_{\alpha j}}=(\operatorname{cof}A)\_{\alpha j}$,
3. $\dfrac{\partial^2(\det A)}{\partial A\_{\alpha j}\partial A\_{\beta k}}=-\dfrac{\partial^2(\det A)}{\partial A\_{\alpha k}\partial A\_{\beta j}}$ (antisymmetry when swapping $(j,k)$ for fixed $\alpha,\beta$).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 2.15</summary>

**(i)** is a result of linear algebra (the adjugate formula $A\,\operatorname{adj}A=\operatorname{adj}A\,A=(\det A)\,\mathrm{Id}$ with $\operatorname{adj}A=(\operatorname{cof}A)^{\!\top}$).

**(ii).** From (i), reading off the $(\alpha,\alpha)$ entry of $(\operatorname{cof}A)A^{\!\top}=(\det A)\mathrm{Id}$, we get for any fixed $\alpha$ (no summation over $\alpha$) the Laplace expansion along the $\alpha$-th row,

$$
\det A=\sum_{i=1}^n A_{\alpha i}\,(\operatorname{cof}A)_{\alpha i}. \tag{2.12}
$$

None of the coefficients $(\operatorname{cof}A)\_{\alpha i}$ depends on the entries $A\_{\alpha j}$ of the $\alpha$-th row (that row is deleted in the defining minors). Identity (ii) follows by differentiating (2.12) with respect to $A\_{\alpha j}$.

**(iii)** follows from the fact that the determinant is an alternating multilinear form (it is an exercise in the script). Explicitly, from the Leibniz formula $\det A=\sum\_\sigma\operatorname{sgn}\sigma\prod\_i A\_{i\sigma(i)}$,

$$
\frac{\partial^2\det A}{\partial A_{\alpha j}\partial A_{\beta k}}=\sum_{\sigma:\ \sigma(\alpha)=j,\ \sigma(\beta)=k}\operatorname{sgn}\sigma\prod_{i\neq\alpha,\beta}A_{i\sigma(i)}\qquad(\alpha\neq\beta,\ j\neq k),
$$

and composing $\sigma$ with the transposition $(j\,k)$ is a bijection between the permutations with $(\sigma(\alpha),\sigma(\beta))=(j,k)$ and those with $(\sigma(\alpha),\sigma(\beta))=(k,j)$ which flips the sign; for $\alpha=\beta$ or $j=k$ both sides vanish. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Geometry</span><span class="math-callout__name">(Volumes and surface elements)</span></p>

A matrix $A$ sends a vector $v$ to $Av$ and scales **volumes** by the factor $\det A$. The entries of $\operatorname{cof}A$ describe the transformation of **surface elements**: if a surface patch has area element $\mathrm{d}S$ and unit normal $\nu$, its image under $A$ has area element and normal given by $\operatorname{cof}(A)\,\nu\,\mathrm{d}S$ (Nanson's formula). This is why $\operatorname{cof}(Du)\nu$ appears in the boundary integral of Example 2.14, and why that boundary integral only sees $u$ on $\partial\Omega$: the surface element of $u(\partial\Omega)$ depends only on tangential derivatives of $u$ along $\partial\Omega$.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_cofactor_geometry.png' | relative_url }}" alt="Left: the unit square with unit outward normals on its four edges. Right: its image under a linear map A, a parallelogram of area det A, with the vectors cof(A) times nu attached to the image edges; each is normal to its edge and as long as the edge." loading="lazy">
  <figcaption>Lemma 2.15 in pictures. A linear map $A$ scales area by $\det A$ (the square of area $1$ becomes a parallelogram of area $\det A$). Each edge, with unit normal $\nu$ and length $1$, is mapped to an edge whose scaled normal is $\mathrm{cof}(A)\,\nu$: this vector is perpendicular to the image edge and its length equals the image edge's length. Volumes transform with $\det$, oriented surface elements with $\mathrm{cof}$.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.16 (The determinant is a Null Lagrangian)</span></p>

The determinant $\det:\mathbb R^{n\times n}\to\mathbb R$ is a Null Lagrangian. In particular, for $u\in C^2(\overline\Omega)^n$ we have

1. $\operatorname{div}(\operatorname{cof}Du)=0$ (**Piola identity**),
2. $\det Du=\dfrac1n\operatorname{div}\bigl((\operatorname{cof}Du)^{\!\top}u\bigr)$ (**divergence structure**),

where $\operatorname{div}$ of a matrix field acts row-wise.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.16</summary>

**(i).** With $f(\xi):=\det\xi$ we have $D\_\xi f(\xi)=\operatorname{cof}\xi$ by Lemma 2.15(ii) and $\nabla\_z f=0$, so the Euler–Lagrange equation of $f$ is exactly the Piola identity (i). Since the divergence of a matrix is defined row-wise, we need to show, with $\partial\_j=\partial\_{x\_j}$,

$$
\partial_j(\operatorname{cof}Du)_{\alpha j}=0\quad\text{in }\Omega\qquad\text{for each }\alpha\in\lbrace1,\dots,n\rbrace .
$$

Using Lemma 2.15(ii) and the chain rule with $F:=Du$,

$$
\partial_j(\operatorname{cof}F)_{\alpha j}=\partial_{x_j}\Bigl(\frac{\partial\det F}{\partial F_{\alpha j}}\Bigr)=\frac{\partial^2(\det F)}{\partial F_{\alpha j}\partial F_{\beta k}}\,\partial_jF_{\beta k},
$$

hence

$$
\sum_{j=1}^n\partial_j(\operatorname{cof}F)_{\alpha j}=\sum_{\beta=1}^n\sum_{j,k=1}^n\frac{\partial^2(\det F)}{\partial F_{\alpha j}\partial F_{\beta k}}\,\partial_jF_{\beta k}.
$$

Since $F\_{\beta k}=\partial\_ku\_\beta$, we have $\partial\_jF\_{\beta k}=\partial^2\_{jk}u\_\beta=\partial^2\_{kj}u\_\beta=\partial\_kF\_{\beta j}$: the second factor in the double sum is **symmetric** in $(j,k)$. By Lemma 2.15(iii) the first factor is **antisymmetric** in $(j,k)$. Hence the double sum vanishes, which concludes the proof of (i).

**(ii).** Using the product rule $\operatorname{div}(M^{\!\top}u)=(\operatorname{div}M)\cdot u+M:Du$ (as in the proof of Theorem 2.6) with $M=\operatorname{cof}Du$ and the Piola identity,

$$
\operatorname{div}\bigl((\operatorname{cof}Du)^{\!\top}u\bigr)=\bigl(\operatorname{div}\operatorname{cof}Du\bigr)\cdot u+\operatorname{cof}Du:Du\overset{\text{(i)}}{=}\operatorname{cof}Du:Du .
$$

Using the adjugate identity of Lemma 2.15(i) and the trace notation $A:B=\operatorname{tr}(A^{\!\top}B)$,

$$
\operatorname{cof}Du:Du=\operatorname{tr}\bigl((\operatorname{cof}Du)^{\!\top}Du\bigr)=\operatorname{tr}\bigl((Du)^{\!\top}\operatorname{cof}Du\bigr)=\operatorname{tr}\bigl((\det Du)\,\mathrm{Id}\bigr)=n\det Du .
$$

Combining the two equalities yields the claim. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On the transpose in (ii))</span></p>

The script writes (ii) as $\det Du=\frac1n\operatorname{div}((\operatorname{cof}Du)u)$. With the row-wise divergence used throughout, the product rule produces $\operatorname{div}(M^{\!\top}u)$, and the $2\times2$ computation of Example 2.14 confirms that the transpose is needed: there the matrix multiplying $u$ is

$$
\begin{pmatrix}\partial_2u_2&-\partial_2u_1\\ -\partial_1u_2&\partial_1u_1\end{pmatrix}=(\operatorname{cof}Du)^{\!\top}.
$$

In index notation (ii) reads $n\det Du=\partial\_j\bigl((\operatorname{cof}Du)\_{\alpha j}u\_\alpha\bigr)$, which is the form in which it is usually quoted. The determinant is an example of a Null Lagrangian which is **not affine**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(Why should determinants be Null Lagrangians?)</span></p>

Can you explain *geometrically* why determinants should be Null Lagrangians?

*Answer.* $\det Du(x)$ is the (signed) volume scaling factor of $u$ at $x$, so $\int\_\Omega\det Du\,\mathrm{d}x$ is the signed volume of the image $u(\Omega)$, counted with multiplicity (the degree of $u$). If $u$ folds $\Omega$ over itself, the folded parts are counted with opposite signs and cancel, and what remains is the volume enclosed by the image of the boundary $u(\partial\Omega)$, which only depends on $u\|\_{\partial\Omega}$. Analytically this is exactly the divergence structure (ii): $\int\_\Omega\det Du=\frac1n\int\_{\partial\Omega}u\cdot\operatorname{cof}(Du)\nu\,\mathrm{d}S$, the "$\frac1n\int x\cdot\nu\,\mathrm{d}S$" formula for the volume enclosed by the surface $u(\partial\Omega)$.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_det_null_lagrangian.png' | relative_url }}" alt="Two deformed polar grids on the unit disk; both maps equal the identity on the boundary circle but deform the interior differently, one with a visible fold; both have integral of det equal to pi" loading="lazy">
  <figcaption>Two maps $u,v:\overline{B_1}\to\mathbb R^2$ that are the identity on the circle $\partial B_1$ but deform the interior differently (the images of a polar grid are shown; $u$ even folds the disk over itself near the upper left, where $\det Du<0$). Numerically $\int_{B_1}\det Du\,\mathrm{d}x=\int_{B_1}\det Dv\,\mathrm{d}x=\pi=\lvert B_1\rvert$: the integral of the determinant sees only the boundary values, as Lemma 2.13 and Theorem 2.16 predict. Folds contribute negative area that is exactly compensated by the triple covering next to them.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Minors)</span></p>

For a matrix $F\in\mathbb R^{m\times n}$ we call every determinant of a square submatrix of $F$ a **minor**. We denote the vector of all minors by $M(F)$. For $n=m=2$ we have $M(F)=(F,\det F)\in\mathbb R^5$; for $n=m=3$ we have $M(F)=(F,\operatorname{cof}F,\det F)\in\mathbb R^{19}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.17 (Characterization of Null Lagrangians)</span></p>

For $f\in C^2(\mathbb R^{m\times n})$ the following statements are equivalent:

1. $f$ is a Null Lagrangian,
2. $f$ is rank-1 affine,
3. $f$ is the sum of a constant and a linear combination of minors of its matrix argument.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.17</summary>

**(i) $\Rightarrow$ (ii).** Let $f$ be a Null Lagrangian. For $\xi\in\mathbb R^{m\times n}$ the linear map $u(x):=\xi x$ is, by Lemma 2.13, a minimizer of $I$ for its own boundary data on any $\Omega\Subset\mathbb R^n$. Since $-f$ is also a Null Lagrangian, $u$ is also a maximizer. By Theorem 2.9 applied to $f$ and to $-f$ (growth conditions are not needed here: $Du=\xi$ and the test functions have bounded gradients, so all integrands stay in a compact set), for $\phi(t):=f(\xi+tA)$ we therefore have

$$
\phi''(0)=D_\xi^2 f(\xi):(A\otimes A)=0
$$

for every rank-one matrix $A\in\mathbb R^{m\times n}$ and every $\xi$. Hence $t\mapsto f(\xi+tA)$ has vanishing second derivative everywhere, i.e. $f$ is affine along every rank-one line: $f$ is rank-1 affine.

**(ii) $\Rightarrow$ (iii) (sketch).** We follow the classical argument (see Dacorogna, *Direct Methods*, Chapter 5, "quasiaffine functions"). For simplicity assume $m=n$. For $A\in\mathbb R^{n\times n}$ let $A^i$ be the matrix that keeps the $i$-th column of $A$ and has zeros elsewhere, so $A=\sum\_i A^i$ and each $A^i$ has rank $\le1$. Since $f$ is affine along the rank-one direction $A^1$ (exactly, not just to second order), $f(A)=f(A-A^1)+\partial\_{A^1}f(A-A^1):A^1$, where $\partial\_{A^1}f$ denotes the derivative in the direction of the first column. The derivatives $\partial\_{A^k}f$ are again rank-1 affine (a function affine in two variables separately is of the form $\alpha+\beta s+\gamma t+\delta st$, so its partial derivatives are again affine in the other variable). With $A^{12}=A^1+A^2$ we therefore obtain

$$
f(A)=f(A-A^{12})+\partial_{A^2}f(A-A^{12}):A^2+\partial_{A^1}f(A-A^{12}):A^1+\partial^2_{A^1A^2}f(A-A^{12}):(A^1\otimes A^2),
$$

and iterating over all $n$ columns,

$$
f(A)=f(0)+\sum_{i=1}^n\partial_{A^i}f(0):A^i+\dots+\partial^n_{A^1\cdots A^n}f(0):(A^1\otimes\dots\otimes A^n),
$$

a polynomial of degree $\le n$ in the entries of $A$ in which every column appears at most linearly. The term of highest order,

$$
\phi_n(A^1,\dots,A^n):=\partial^n_{A^1\cdots A^n}f(0):(A^1\otimes\dots\otimes A^n),
$$

is $n$-linear in the columns. We claim that $\phi\_n$ is proportional to the determinant, $\phi\_n(A)=c\det A$; for this it suffices to show that $\phi\_n$ is alternating, i.e. $\phi\_n(A)=0$ whenever $A$ does not have full rank. Without loss of generality the last column is a combination of the others, $a\_n=\sum\_{i<n}\lambda\_ia\_i$. By linearity in the last slot, $\phi\_n(a\_1,\dots,a\_n)=\sum\_{i<n}\lambda\_i\,\phi\_n(a\_1,\dots,a\_{n-1},a\_i)$, and each term contains the mixed second derivative of $f$ in the two directions $a\_i\otimes e\_i$ and $a\_i\otimes e\_n$. Their sum $a\_i\otimes(e\_i+e\_n)$ is again rank one, so $f$ is affine along it, and expanding $0=D^2f:\bigl((a\_i\otimes(e\_i+e\_n))^{\otimes2}\bigr)$ shows that the mixed second derivative vanishes (the two pure second derivatives vanish since both directions are rank one). Hence $\phi\_n(A)=0$ for rank-deficient $A$, and $\phi\_n=c\det$. Similarly one shows that the terms of lower order are linear combinations of minors.

**(iii) $\Rightarrow$ (i).** By linearity it suffices to show that each minor is a Null Lagrangian. Write $x=(y,z)$ with $y\in\mathbb R^k$, $z\in\mathbb R^{n-k}$, and $u=(v,w)$ with $v\in\mathbb R^k$ (after renumbering coordinates and components); without loss of generality consider the minor $f(Du)=\det D\_yv$. By Fubini's theorem,

$$
I[u]=\int_\Omega \det D_yv\,\mathrm{d}x=\int_{\mathbb R^{n-k}}\int_{\Omega_z}\det D_yv\,\mathrm{d}y\,\mathrm{d}z ,
$$

where for $z\in\mathbb R^{n-k}$ the slice $\Omega\_z\subset\mathbb R^k$ is the set of $y$ with $(y,z)\in\Omega$. By Theorem 2.16 in $\mathbb R^k$ (divergence structure and the Piola identity $\operatorname{div}\_y(\operatorname{cof}D\_yv)=0$), the inner integral is a boundary integral over $\partial\Omega\_z$, hence depends only on $v\|\_{\partial\Omega\_z}$; therefore the full integral depends only on the trace $u\|\_{\partial\Omega}$. By Lemma 2.13, $\det D\_yv$ is a Null Lagrangian. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weak convergence and nonlinearity)</span></p>

Weak convergence is in general **not compatible with nonlinearity**: from $u\_k\rightharpoonup u$ it does not follow that $f(u\_k)\rightharpoonup f(u)$ when $f$ is nonlinear. The standard example is $u\_k(x)=\sin(kx)\rightharpoonup0$ in $L^2(0,2\pi)$ while $u\_k^2=\sin^2(kx)\rightharpoonup\frac12\neq0$: the oscillations average out in $u\_k$ but not in $u\_k^2$ (compare the proof of Theorem 2.9). The composition with the **determinant**, however, preserves weak continuity, because by Theorem 2.16 the determinant of a gradient is a divergence, and divergences only see boundary values, which converge by compactness:

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.18 (Weak continuity of the determinant)</span></p>

Let $\Omega\Subset\mathbb R^n$ be open and $q>n$. Then

$$
u_k\rightharpoonup u\ \text{ in } W^{1,q}(\Omega)^n\qquad\Longrightarrow\qquad \det Du_k\rightharpoonup\det Du\ \text{ in } L^{q/n}(\Omega).
$$

An analogous continuity result holds for all minors.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_weak_continuity_det.png' | relative_url }}" alt="Left: a square grid deformed by a shear wave of small amplitude and high frequency. Right: along the first coordinate, the squared gradient norm oscillates between 2 and 3 with weak limit 2.5 while the limit map has squared gradient norm 2; the determinant is identically 1 and equals the determinant of the limit." loading="lazy">
  <figcaption>Theorem 2.18 on an example. The shear waves $u_k(x)=x+\tfrac1k\sin(kx_1)e_2$ converge uniformly to the identity, and $Du_k=\mathrm{Id}+\cos(kx_1)\,e_2\otimes e_1\rightharpoonup\mathrm{Id}$ weakly but not strongly. The nonlinear quantity $\lvert Du_k\rvert^2=2+\cos^2(kx_1)$ has weak limit $2.5\neq\lvert D\,\mathrm{id}\rvert^2=2$. But $\det Du_k\equiv1=\det D\,\mathrm{id}$: the oscillation is a rank-one perturbation, invisible to the rank-1 affine function $\det$. This is the mechanism behind the theorem. (Minors are in fact the only functions of the gradient with this property, a converse due to Ball that is not proved in this course.)</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.18</summary>

We prove the statement for the determinant; the proof carries over to all minors by restricting to the relevant coordinates and components (as in the proof of Theorem 2.17). Since weakly convergent sequences are bounded, and $\Omega$ is bounded so that $L^q\subset L^n$, we have the estimates

$$
\lVert Du_k\rVert_{L^n(\Omega)}\le C_\Omega\lVert Du_k\rVert_{L^q(\Omega)}\le C,\qquad \lVert\det Du_k\rVert_{L^{q/n}(\Omega)}\lesssim\lVert Du_k\rVert^n_{L^q(\Omega)}\le C. \tag{2.13}
$$

By the Rellich–Kondrachov theorem (exercise) the weak convergence in $W^{1,q}$ upgrades to strong convergence of the functions,

$$
u_k\to u\qquad\text{in } L^q(\Omega).
$$

We have to show that

$$
\int_\Omega \varphi\det Du_k\,\mathrm{d}x\to\int_\Omega \varphi\det Du\,\mathrm{d}x\qquad\forall\varphi\in\mathcal D(\Omega). \tag{2.14}
$$

In view of the uniform bound (2.13) in $L^{q/n}$, and since $q/n>1$ so that the dual exponent is finite, it suffices to show (2.14) for a dense subset of $L^{(q/n)^{\prime}}(\Omega)$, e.g. for simple functions built from open sets; by linearity it suffices to consider characteristic functions, i.e. to show

$$
\int_V \det Du_k\,\mathrm{d}x\to\int_V\det Du\,\mathrm{d}x\qquad\text{for all open } V\Subset\Omega .
$$

If $u\_k$ and $u$ had the same boundary data on $\partial V$, the statement would follow immediately from the fact that the determinant is a Null Lagrangian (Lemma 2.13, extended to $W^{1,q}$ by density). In general we must **correct the boundary values**: for a cut-off $\eta\in\mathcal D(V)$ with $0\le\eta\le1$ define

$$
v_k:=u+\eta\,(u_k-u).
$$

Since $v\_k=u$ on $\partial V$, we get $\int\_V\det Dv\_k=\int\_V\det Du$ and therefore

$$
\Bigl|\int_V\det Du_k\,\mathrm{d}x-\int_V\det Du\,\mathrm{d}x\Bigr|=\Bigl|\int_V\det Du_k-\det Dv_k\,\mathrm{d}x\Bigr|=:I_k .
$$

From the estimate (exercise; it follows from multilinearity of the determinant)

$$
\lvert\det G-\det F\rvert\lesssim\bigl(\lvert F\rvert^{n-1}+\lvert G\rvert^{n-1}\bigr)\lvert F-G\rvert
$$

we obtain, using Hölder's inequality with the dual exponents $\frac n{n-1}$ and $n$,

$$
I_k\lesssim\bigl(\lVert Du_k\rVert^{n-1}_{L^n}+\lVert Dv_k\rVert^{n-1}_{L^n}\bigr)\lVert Dv_k-Du_k\rVert_{L^n}\le C\,\lVert D(v_k-u_k)\rVert_{L^n(V)},
$$

where $C$ can be chosen independently of $\eta$: indeed $Dv\_k=Du+\eta(Du\_k-Du)+(u\_k-u)\otimes\nabla\eta$, and for fixed $\eta$ the last term tends to $0$ in $L^n$ by the strong convergence $u\_k\to u$ in $L^q$, so $\limsup\_k\lVert Dv\_k\rVert\_{L^n}\le\lVert Du\rVert\_{L^n}+\sup\_k\lVert Du\_k-Du\rVert\_{L^n}$, a bound that does not involve $\eta$. Now $v\_k-u\_k=(1-\eta)(u-u\_k)$ and hence

$$
\lvert D(v_k-u_k)\rvert\le(1-\eta)\lvert Du-Du_k\rvert+\lvert\nabla\eta\rvert\,\lvert u-u_k\rvert .
$$

By Hölder's inequality with the exponents $\frac qn$ and $\frac q{q-n}$, i.e. $\frac1n=\frac1s+\frac1q$ with $s:=\frac{nq}{q-n}<\infty$,

$$
\limsup_{k\to\infty}I_k\le C\limsup_{k\to\infty}\Bigl(\lVert1-\eta\rVert_{L^s(V)}\lVert Du-Du_k\rVert_{L^q}+\lVert\nabla\eta\rVert_{L^s}\lVert u-u_k\rVert_{L^q}\Bigr)\le C\lVert1-\eta\rVert_{L^s(V)},
$$

where the first term is bounded by the uniform $L^q$ bound on the gradients and the second term tends to $0$ by the strong convergence $u\_k\to u$ in $L^q$. Since $s<\infty$, we can make $\lVert1-\eta\rVert\_{L^s(V)}$ arbitrarily small by choosing the cut-off $\eta$ equal to $1$ on most of $V$. Hence $I\_k\to0$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Where does the proof use $q>n$?)</span></p>

Check where the proof does not work if $q=n$. (Then $s=\infty$ and $\lVert1-\eta\rVert\_{L^\infty(V)}=1$ for every cut-off $\eta\in\mathcal D(V)$, so the final step collapses. Indeed, for $q=n$ the statement is false in general: weak limits of $\det Du\_k$ can concentrate into measures.)

</div>

As a consequence of the fact that the determinant is a Null Lagrangian we can derive Brouwer's fixed point theorem. We first prove the following auxiliary lemma, the **no-retraction theorem**. Here $B\_1\subset\mathbb R^n$ is the open unit ball and $\overline{B\_1}$ its closure.

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">2.19 (No retraction of the ball onto its boundary)</span></p>

There is no continuous function $w:\overline{B\_1}\to\partial B\_1$ with $w(x)=x$ for all $x\in\partial B\_1$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Lemma 2.19</summary>

**Step 1: There is no smooth map $w:\overline{B\_1}\to\partial B\_1$ with $w(x)=x$ for all $x\in\partial B\_1$.** We argue indirectly and assume that such a function $w$ exists. Let $v(x)=x$ be the identity map. From Theorem 2.16 and Lemma 2.13, and since $v=w$ on $\partial B\_1$, it follows that

$$
\int_{B_1}\det Dw\,\mathrm{d}x=\int_{B_1}\det Dv\,\mathrm{d}x=\lvert B_1\rvert\neq0 . \tag{2.15}
$$

On the other hand, from $\lvert w\rvert^2=1$ we obtain by differentiating $2\,w\cdot\partial\_iw=0$ for all $i\in\lbrace1,\dots,n\rbrace$, i.e. $(Dw)^{\!\top}w=0$. Since $w\neq0$, the matrix $(Dw)^{\!\top}$ has a nontrivial kernel, so $Dw$ is not invertible and $\det Dw=0$ at every $x\in B\_1$. This contradicts (2.15).

**Step 2: There is no continuous such $w$ either.** We argue indirectly and assume that a continuous $w$ as in the statement exists. We extend $w$ to $\mathbb R^n\setminus B\_1$ by $w(x)=x$; the extension is continuous since $w=\mathrm{id}$ on $\partial B\_1$, and satisfies $\lvert w\rvert\ge1$ everywhere. Let $\eta\_\varepsilon$ be a standard family of symmetric mollifiers and define $w\_\varepsilon:=\eta\_\varepsilon\ast w$. For $\varepsilon<1$ we have $w\_\varepsilon(x)=x$ on $\partial B\_2$, since there the mollifier only sees the identity map, and mollifying the identity with a symmetric kernel returns the identity. Since $w\_\varepsilon\to w$ uniformly on compact sets as $\varepsilon\to0$ and $\lvert w\rvert\ge1$, for $\varepsilon$ sufficiently small we have $\lvert w\_\varepsilon\rvert\ge\frac12$ on $\overline{B\_2}$. For such $\varepsilon<1$ define $\tilde w:\overline{B\_1}\to\partial B\_1$ by

$$
\tilde w(x):=\frac{w_\varepsilon(2x)}{\lvert w_\varepsilon(2x)\rvert}.
$$

Then $\tilde w$ is smooth, takes values in $\partial B\_1$, and for $x\in\partial B\_1$ we have $w\_\varepsilon(2x)=2x$ and thus $\tilde w(x)=x$. So $\tilde w$ satisfies the assumptions of Step 1, a contradiction. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.20 (Brouwer's fixed point theorem)</span></p>

Every continuous map $u:\overline{B\_1}\to\overline{B\_1}$ has a fixed point.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 2.20</summary>

Suppose there is $u\in C^0(\overline{B\_1};\overline{B\_1})$ without a fixed point, i.e. $u(x)\neq x$ for all $x\in\overline{B\_1}$. Since $\overline{B\_1}$ is compact and $x\mapsto\lvert u(x)-x\rvert$ is continuous and positive, there exists $\varepsilon>0$ with $\lvert u(x)-x\rvert>\varepsilon$ for all $x\in\overline{B\_1}$. Let $L\_x$, given by $y(t)=u(x)+t\,(x-u(x))$ for $t\ge0$, be the half-line which starts at $u(x)$ and passes through $x$. We define $w(x)$ as the point where $L\_x$ **exits** the closed ball, i.e. $w(x)=u(x)+t(x)\,(x-u(x))$ where $t(x)\ge1$ is the larger root of the quadratic equation $\lvert u(x)+t(x-u(x))\rvert^2=1$. This root depends continuously on $x$ because the direction $x-u(x)$ never degenerates, so $w:\overline{B\_1}\to\partial B\_1$ is continuous. In particular, for $x\in\partial B\_1$ the segment from $u(x)$ to $x$ lies in the ball and $x$ itself is on the sphere, so $t(x)=1$ and $w(x)=x$. This is a continuous retraction of $\overline{B\_1}$ onto $\partial B\_1$, a contradiction to Lemma 2.19. $\square$

</details>
</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/calculus_of_variations_hd/cov_brouwer_retraction.png' | relative_url }}" alt="Left: the unit disk with three points x, their images u(x) and the exit points w(x) of the rays from u(x) through x; a boundary point is fixed by w. Right: the determinant of the near-retraction w delta as a function of the radius for three values of delta; it is a tall step of height delta to the minus two on the small disk and zero outside, with constant integral." loading="lazy">
  <figcaption>Brouwer via Null Lagrangians. (a) If $u$ had no fixed point, following the ray from $u(x)$ through $x$ until it leaves the ball would define a continuous retraction $w$ of $\overline{B_1}$ onto $\partial B_1$, with $w(x)=x$ on the sphere. (b) Why no such retraction exists: for any map that is the identity on $\partial B_1$, $\int_{B_1}\det Dw\,\mathrm{d}x=\lvert B_1\rvert$ is fixed by the boundary values, so the enclosed area cannot be destroyed. The maps $w_\delta(x)=x/\max(\lvert x\rvert,\delta)$ squash the annulus $\delta\le\lvert x\rvert\le1$ onto the circle, where $\det Dw_\delta=0$, but the whole area $\pi$ is then concentrated on the small disk $\lvert x\rvert<\delta$, where $\det Dw_\delta=\delta^{-2}$. A genuine retraction would need $\det Dw\equiv0$, which is incompatible with the conserved integral.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Summary</span><span class="math-callout__name">(The thread of Chapter 2)</span></p>

* **Differentiating a functional** (Definition 2.1) is done along lines $u+tv$ in the tangent space $X\_0$ of the admissible class; the first variation gives the **Euler–Lagrange equation** (Theorems 2.3, 2.5, 2.6), whose weak form is the fundamental object and whose boundary terms produce the natural boundary conditions.
* **The second variation** at a minimizer is non-negative (Theorem 2.4); tested with fine oscillations it localizes to the **Legendre–Hadamard condition** (Theorem 2.9): convexity of the integrand along rank-one directions. For $m=1$ or $n=1$ this is plain convexity, whose absence destroys uniqueness and regularity (Example 2.8).
* **Null Lagrangians** are the integrands whose integral depends only on boundary values (Lemma 2.13); they are exactly the rank-1 affine functions, i.e. the linear combinations of **minors** (Theorem 2.17). Being divergences, minors are **weakly continuous** (Theorem 2.18; by a converse not proved here, they are the only nonlinear functions of the gradient with this property), and the invariance of $\int\det Du$ under interior changes proves **Brouwer's fixed point theorem** (Theorem 2.20).

</div>
