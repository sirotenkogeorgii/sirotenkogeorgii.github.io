---
layout: default
title: "Introduction to Geometric Deep Learning"
date: 2026-04-19
excerpt: "Notes on 'Introduction to Geometric Deep Learning' (SS 2026) — starting with the appendix on mathematical background: Euclidean structure, orthogonal inverses, pseudo-inverses, tensor products and generalized inverses."
tags:
  - geometric-deep-learning
  - linear-algebra
  - tensor-products
  - pseudo-inverse
---

<!-- <style>
  .accordion summary {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    background-color: var(--accent-soft, #f5f6ff);
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--accent-strong, #2c3e94);
    border-radius: 0.25rem;
  }
</style> -->

# Introduction to Geometric Deep Learning

**Table of Contents**
- TOC
{:toc}

## Problems

[Selected Problems](/subpages/books/geometric_deep_learning_hd/problems/)

## Appendix A. Mathematical Background

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">A.1 (Summation convention)</span></p>

Sums are taken "automatically" over repeated indices, without the $\sum$ symbol, whenever the summands and their range make sense in the context.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Summation convention)</span></p>

$$
\ell(x, x') \equiv \langle L x, x'\rangle \equiv (Lx)(x') \equiv (Lx)_i\,(x')^i \qquad \text{(summation convention)}.
$$

</div>

### A.1. Basic Euclidean Set-Up

We supply finite-dimensional real vector spaces $\mathcal{X}$ and $\mathcal{Y}$ with **scalar products** $\ell$ and $m$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dual spaces)</span></p>

The **dual spaces** of $\mathcal{X}$ and $\mathcal{Y}$ are denoted by

$$
\widecheck{\mathcal{X}} = \mathcal{L}(\mathcal{X}, \mathbb{R}), \qquad \widecheck{\mathcal{Y}} = \mathcal{L}(\mathcal{Y}, \mathbb{R}). \tag{A.1}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Duality mappings and pairings)</span></p>

The **duality mappings** and associated **duality pairings** are

$$
\begin{aligned}
L \in \mathcal{L}(\mathcal{X}, \widecheck{\mathcal{X}}): & \quad \langle L x, x' \rangle := \ell(x, x'), \quad \forall\, x, x' \in \mathcal{X}, &&\text{(A.2a)} \\
M \in \mathcal{L}(\mathcal{Y}, \widecheck{\mathcal{Y}}): & \quad \langle M y, y' \rangle := m(y, y'), \quad \forall\, y, y' \in \mathcal{Y}. &&\text{(A.2b)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Induced norms)</span></p>

The norms induced by those scalar products are

$$
\lambda(x) := \sqrt{\ell(x, x)}, \qquad \mu(y) := \sqrt{m(y, y)}. \tag{A.3}
$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_scalar_product_geometry.png' | relative_url }}" alt="Three side-by-side panels of R² with the same canonical basis δ₁, δ₂ but three different scalar products: a circle for G = I, a horizontally-stretched ellipse for G = diag(1, 4), and a tilted ellipse for an oblique G with off-diagonal entries. Inset heatmaps show each Gramian." loading="lazy">
  <figcaption>The same vector space $\mathcal X = \mathbb R^2$ acquires entirely different geometries depending on which scalar product $\ell$ we install on it. Concretely, the unit ball $\lbrace x : \ell(x,x) \le 1\rbrace $ is the level-1 set of the quadratic form $x^\top G x$, where $G = (\ell(\delta_i, \delta_j))_{ij}$ is the Gramian. For $G = I$ it is the round disc; for $G = \mathrm{diag}(1, 4)$ it is squashed along $\delta_2$ (lengths in that direction are now twice as expensive); for the oblique $G$ it tilts away from the axes because $\delta_1, \delta_2$ are no longer $\ell$-orthogonal. The basis arrows themselves do not move — only the geometry painted on top of them.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Gramian representation)</span></p>

For a basis $(e_i)\_{i \in [d]} \subset \mathcal{X}$, one has (cf. Remark A.1)

$$
\ell(x, x') = \ell(x^ie_i, x^je_j) = x^ix^j\ell(e_i,e_j). \tag{A.4}
$$

The Gramian matrix $(\ell(e_i,e_j))\_{i, j \in [d]}$ is symmetric, positive definite, and invertible. It represents the operator $L$ in the basis $(e_i)\_{i \in [d]}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dual scalar product and dual norm)</span></p>

The **dual scalar product** on $\widecheck{\mathcal{X}}$ is given by

$$
\widecheck{\ell}(p, p') = \ell(L^{-1}p, L^{-1}p') = \langle p, L^{-1} p' \rangle, \qquad p, p' \in \widecheck{\mathcal{X}}, \tag{A.5a}
$$

with corresponding norm

$$
\widecheck{\lambda}(p) := \sqrt{\widecheck{\ell}(p, p)}. \tag{A.5b}
$$

#TODO: what is the motivation for dual inner product?

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Canonical Euclidean basis)</span></p>

The canonical Euclidean basis $(\delta_i)\_{i \in [d]} \subset \lbrace 0,1 \rbrace^d$ is

$$
\delta_i = (0, \dots, 0, \underbrace{1}_{i\text{-th}}, 0, \dots, 0)^{\top}, \tag{A.6}
$$

and the corresponding coordinate vectors are often arbitrary and do not relate to the application at hand. Scalar products can then be used to adjust the geometry.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">A.2 (Making bases orthonormal)</span></p>

1. Let $(e_i)_{i \in [d]} \subset \mathcal{X}$ be any basis. Define (cf. Remark A.1)

   $$
   B \in \mathcal{L}(\mathbb{R}^d, \mathcal{X}), \qquad B\, v := v^i e_i \in \mathcal{X}, \quad v \in \mathbb{R}^d. \tag{A.7}
   $$

2. Then the basis $(e_i)_{i \in [d]}$ is **orthonormal for the scalar product** on $\mathcal{X}$ iff

   $$
   \ell(x, x') := \langle B^{-1}x, B^{-1}x' \rangle, \tag{A.8}
   $$

   where $\langle \cdot, \cdot \rangle$ denotes the **canonical scalar product** on $\mathbb{R}^d$. 

3. The **duality mapping** is

   $$
   L = (B \widecheck{B})^{-1} \tag{A.9}
   $$

   with the **transposed operator** (and the identification $\widecheck{\mathbb{R}}^d \cong \mathbb{R}^d$):

   $$
   \widecheck{B} \in \mathcal{L}(\widecheck{\mathcal{X}}, \mathbb{R}^d). \tag{A.10}
   $$

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_orthonormalization.png' | relative_url }}" alt="Three panels: left, R^d with canonical inner product and standard basis δ₁, δ₂ inside a circular unit ball; middle, the space X with an oblique basis e₁, e₂ inside the same circle, where e₁ and e₂ are not orthonormal; right, the same space X with an elliptical unit ball aligned to e₁, e₂, where they become orthonormal under the induced metric." loading="lazy">
  <figcaption>Lemma A.2 in three pictures. <em>Left:</em> $\mathbb R^d$ with the canonical inner product, where the standard basis $(\delta_i)$ is automatically orthonormal — its unit ball is the round disc. <em>Middle:</em> transport the $\delta_i$ via $B$ to land at the (oblique) basis $e_1, e_2 \in \mathcal X$; under the canonical inner product on $\mathcal X$, the $e_i$ are not orthonormal — they are not orthogonal ($\langle e_1, e_2\rangle \ne 0$) and their lengths differ from $1$. <em>Right:</em> swap the metric for the pulled-back inner product $\ell(x, x') := \langle B^{-1}x, B^{-1}x'\rangle$. Its unit ball is the image $B(\text{round disc})$ — an ellipse whose principal directions are exactly the $e_i$, restoring orthonormality.</figcaption>
</figure>

### A.1.2. Orthogonal Left- and Right-Inverses

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">A.3 (Orthogonal Left- and Right-Inverses)</span></p>

**(a)** Let $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be *injective*. Then

$$
A^{-} := (\widecheck{A} M A)^{-1} \widecheck{A} M \in \mathcal{L}(\mathcal{Y}, \mathcal{X}) \tag{A.11}
$$

is the **orthogonal left-inverse** of $A$. In particular,

$$
\begin{aligned}
A^{-} Ax &=x  && \forall x\in \mathcal{X}, &&\text{(A.12a)} \\
A A^{-} &= \Pi_{\mathrm{rge}(A)} &&\text{(orthogonal projection onto the range of } A\text{).} &&\text{(A.12b)}
\end{aligned}
$$

**(b)** Let $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be *surjective*. Then

$$
A^{+} := L^{-1} \widecheck{A} (A L^{-1} \widecheck{A})^{-1} \in \mathcal{L}(\mathcal{Y}, \mathcal{X}) \tag{A.13}
$$

is the **orthogonal right-inverse** of $A$. In particular,

$$
\begin{aligned}
A A^{+}y &= y && \forall y\in \mathcal{Y}, &&\text{(A.14a)} \\
I - A^{+} A &= \Pi_{\ker(A)} &&\text{(orthogonal projection onto the kernel (nullspace) of } A\text{).} &&\text{(A.14b)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">A.4 (Properties of orthogonal partial inverses)</span></p>

**(a)** Let $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be *injective*. Then

$$
\begin{aligned}
\widecheck{A^{-}} &= \widecheck{A}^{+}, &&\text{(A.15a)} \\
A &= (A^{+})^{-}, &&\text{(A.15b)} \\
(\widecheck{A} M A)^{-1} &= A^{-} M^{-1} (\widecheck{A})^{-1}, &&\text{(A.15c)} \\
(A V)^{-} &= V^{-1} A^{-} &&\text{if } V \in \mathcal{L}(\mathcal{Z}, \mathcal{X}) \text{ is invertible.} &&\text{(A.15d)}
\end{aligned}
$$

**(b)** Let $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be *surjective*. Then

$$
\begin{aligned}
\widecheck{A^{+}} &= \widecheck{A}^{-}, &&\text{(A.16a)} \\
A &= (A^{+})^{-}, &&\text{(A.16b)} \\
(A L^{-1} \widecheck{A})^{-1} &= \widecheck{(A^{+})} L A^{+}, &&\text{(A.16c)} \\
(V A)^{+} &= A^{+} V^{-1} &&\text{if } V \in \mathcal{L}(\mathcal{Y}, \mathcal{Z}) \text{ is invertible.} &&\text{(A.16d)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">A.5 (Pseudo-inverse)</span></p>

Let

$$
C = B A \in \mathcal{L}(\mathcal{X}, \mathcal{Y}) \tag{A.17}
$$

with *injective* factor $B$ and *surjective* factor $A$. Then

$$
C^{\dagger} := A^{+} B^{-} \in \mathcal{L}(\mathcal{Y}, \mathcal{X}) \tag{A.18}
$$

is called the **pseudo-inverse** of $C$. For a given $y \in \mathcal{Y}$, the vector

$$
\overline{x} = C^{\dagger} y \tag{A.19}
$$

has *minimal* norm $\lambda(\overline{x})$ among all points $x \in \mathcal{X}$ mapped by $C$ to the *closest* point to $y$ in $\mathrm{rge}(C)$. One has

$$
\begin{aligned}
C^{\dagger} &= C^{-} &&\text{if } C \text{ injective}, &&\text{(A.20a)} \\
C^{\dagger} &= C^{+} &&\text{if } C \text{ surjective}, &&\text{(A.20b)} \\
C^{\dagger} &= C^{-1} &&\text{if } C \text{ bijective}. &&\text{(A.20c)}
\end{aligned}
$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_pseudoinverse_minnorm.png' | relative_url }}" alt="Two panels for the pseudo-inverse picture. Left: the source space X with ker(C) drawn as a dashed line, ker(C)^perp as a solid line, and an affine line of preimages of ŷ parallel to ker(C); the special preimage x̄ = C†y sits where this affine line meets ker(C)^perp, closer to the origin than the other (grey) candidate preimages. Right: the target space Y with rge(C) as a line, a target point y above it, and ŷ = CC†y as the orthogonal projection of y onto rge(C); the residual y − ŷ is shown perpendicular to rge(C)." loading="lazy">
  <figcaption>The pseudo-inverse $C^\dagger = A^+ B^-$ from Definition A.5 read right-to-left. <em>Right:</em> a target $y \in \mathcal Y$ that may not lie in $\mathrm{rge}(C)$. The orthogonal projection $\hat y = CC^\dagger y$ is the closest point of $\mathrm{rge}(C)$ to $y$ in the $m$-metric; the residual $y - \hat y$ is $m$-orthogonal to $\mathrm{rge}(C)$. <em>Left:</em> the preimage $C^{-1}(\hat y)$ is an affine line — a coset of $\ker(C)$. All grey points map to $\hat y$, but $\bar x = C^\dagger y$ is the unique element of that coset closest to the origin in $\mathcal X$, equivalently the unique element lying in $\ker(C)^\perp = \mathrm{rge}(C^\dagger)$. So $C^\dagger$ realises in a single linear map the two-step recipe: <em>project to the range, then take the minimum-norm preimage.</em></figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">A.6 (Properties of the pseudo-inverse)</span></p>

The pseudo-inverse (A.18) satisfies

$$
\begin{aligned}
C^{\dagger} C C^{\dagger} &= C^{\dagger}, &&\text{(A.21a)} \\
C C^{\dagger} C &= C, &&\text{(A.21b)} \\
C^{\dagger} C &= \Pi_{\ker(C)^{\perp}}, &&\text{(A.21c)} \\
C C^{\dagger} &= \Pi_{\mathrm{rge}(C)}, &&\text{(A.21d)} \\
(C^{\dagger})^{\dagger} &= C, &&\text{(A.21e)} \\
\widecheck{C^{\dagger}} &= \widecheck{C}^{\dagger}, &&\text{(A.21f)} \\
(C_2 C_1)^{\dagger} &= C_1^{\dagger} C_2^{\dagger}. &&\text{(A.21g)}
\end{aligned}
$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_penrose_conditions.png' | relative_url }}" alt="Two-by-two grid of small diagrams illustrating the four Penrose–Moore conditions for a fixed rank-1 example C: R² → R². Top-left (a): a target vector y and its projection ŷ = CC†y onto rge(C). Top-right (b): a vector Cx already in rge(C). Bottom-left (c): a vector x in source space and its projection C†Cx onto ker(C)^perp. Bottom-right (d): a vector y in target space and its projection CC†y onto rge(C)." loading="lazy">
  <figcaption>The four Penrose–Moore identities (A.21a–d) read off geometrically for a fixed rank-1 example $C : \mathcal X \to \mathcal Y$. <strong>(a)</strong> $C^\dagger$ ignores the $\mathrm{rge}(C)^\perp$-component of its input, so feeding it $y$ versus its projection $\hat y = CC^\dagger y$ produces the same answer — hence $C^\dagger CC^\dagger = C^\dagger$. <strong>(b)</strong> $Cx$ already lives in $\mathrm{rge}(C)$, where $CC^\dagger$ acts as the identity, so $CC^\dagger Cx = Cx$. <strong>(c)</strong> $C^\dagger C$ is the orthogonal projection in $\mathcal X$ onto $\ker(C)^\perp$ — it strips off the kernel component and keeps the rest. <strong>(d)</strong> $CC^\dagger$ is the orthogonal projection in $\mathcal Y$ onto $\mathrm{rge}(C)$. Together (c) and (d) capture the symmetry that makes $C^\dagger$ unique among all generalized inverses.</figcaption>
</figure>

### A.1.3. Tensor-Products and Generalized Inverses

*Mathematical background: Section A.2.1.*

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Scalar product on $\widecheck{\mathcal{X}} \otimes \mathcal{Y}$)</span></p>

In the situation (A.2), (A.3), **the space $\widecheck{\mathcal{X}} \otimes \mathcal{Y} \cong \mathcal{L}(\mathcal{X}, \mathcal{Y})$ is equipped with the scalar product** (cf. Remark A.1)

$$
(\widecheck{\ell} \otimes m)(U, V) := \langle \widecheck{f}^i, U e_j\rangle\, \langle \widecheck{f}^i, V e_j\rangle, \qquad \forall\, U, V \in \mathcal{L}(\mathcal{X}, \mathcal{Y}), \tag{A.22}
$$

where $(e_i)\_{i \in [d]} \subset \mathcal{X}$ and $(f_i)\_{i \in [d]} \subset \mathcal{Y}$ are $\ell$-orthonormal and $m$-orthonormal bases:

$$
\langle \widecheck{e}^i, e_j\rangle = \ell(e_i, e_j) = \delta^i_j, \qquad \langle \widecheck{f}^i, f_j\rangle = m(f_i, f_j) = \delta^i_j, \qquad i, j \in [d]. \tag{A.23}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Basis-independence of (A.22))</span></p>

The expression defining the inner product (A.25) does *not* depend on the choice of these bases, since

$$
\begin{aligned}
(\widecheck{\ell} \otimes m)(U, V) &= \langle \widecheck{f}^i, U e_j\rangle\, \langle \widecheck{f}^i, V e_j\rangle &&\text{(A.24a)} \\
&= \widecheck{f}^i(U e_j)\, \widecheck{f}^i(V e_j)\, \underbrace{\delta^i_i}_{=\,1} \;=\; \widecheck{f}^i(U e_j)\, \widecheck{f}^i(V e_j)\, \underbrace{\langle M f_i, f_i\rangle}_{=\,\delta^i_i} &&\text{(A.24b)} \\
&= \langle M\, \underbrace{\widecheck{f}^i(U e_j)\, f_i}_{=\,U e_j},\; \underbrace{\widecheck{f}^i(V e_j)\, f_i}_{=\,V e_j}\rangle \;=\; \langle M U e_j, V e_j\rangle &&\text{(A.24c)} \\
&= m(U e_j, V e_j). &&\text{(A.24d)}
\end{aligned}
$$

Similarly, using $\delta^j_k = \langle \widecheck{e}^j, e_k\rangle = \langle \widecheck{e}^j, L^{-1}\widecheck{e}^k\rangle$,

$$
\begin{aligned}
(\widecheck{\ell} \otimes m)(U, V) &= \langle \widecheck{f}^i, U e_j\rangle\, \langle \widecheck{f}^i, V e_j\rangle &&\text{(A.24e)} \\
&= \langle \widecheck{U}\widecheck{f}^i, e_j\rangle\, \langle \widecheck{V}\widecheck{f}^i, e_j\rangle\, \underbrace{\delta^j_j}_{=\,1} \;=\; \langle \widecheck{U}\widecheck{f}^i, e_j\rangle\, \langle \widecheck{V}\widecheck{f}^i, e_j\rangle\, \underbrace{\langle \widecheck{e}^j, L^{-1}\widecheck{e}^j\rangle}_{=\,\delta^j_j} &&\text{(A.24f)} \\
&= \langle\, \underbrace{\langle \widecheck{U}\widecheck{f}^i, e_j\rangle\, \widecheck{e}^j}_{=\,\widecheck{U}\widecheck{f}^i},\; L^{-1}\, \underbrace{\langle \widecheck{V}\widecheck{f}^i, e_j\rangle\, \widecheck{e}^j}_{=\,\widecheck{V}\widecheck{f}^i}\rangle \;=\; \langle \widecheck{U}\widecheck{f}^i, L^{-1}\widecheck{V}\widecheck{f}^i\rangle &&\text{(A.24g)} \\
&= \widecheck{\ell}(\widecheck{U}\widecheck{f}^i, \widecheck{V}\widecheck{f}^i). &&\text{(A.24h)}
\end{aligned}
$$

Thus, the definition (A.22) is equivalent to the expressions

$$
\begin{aligned}
(\widecheck{\ell} \otimes m)(U, V) &= \langle \widecheck{f}^i, U e_j\rangle\, \langle \widecheck{f}^i, V e_j\rangle &&\text{(A.25a)} \\
&= m(U e_j, V e_j) \;=\; \widecheck{\ell}(\widecheck{U}\widecheck{f}^i, \widecheck{V}\widecheck{f}^i) &&\text{(A.25b)}
\end{aligned}
$$

and independent of the choice of the bases.

</div>

<!-- <figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_tensor_scalar_product.png' | relative_url }}" alt="A single R² panel with four arrows from the origin: two blue arrows Ue₁ and Ue₂ representing U applied to the orthonormal basis vectors, and two orange arrows Ve₁ and Ve₂ representing V applied to the same basis. Annotations show the per-index inner products m(Ue_j, Ve_j) and their sum equal to (ℓ̌ ⊗ m)(U, V)." loading="lazy">
  <figcaption>How the tensor scalar product (A.22) / (A.25b) really works. Pick an $\ell$-orthonormal frame $(e_j)$ of $\mathcal X$ and apply $U$ and $V$ to each frame vector — this produces two "image frames" $(Ue_j), (Ve_j) \subset \mathcal Y$ (here in $\mathbb R^2$ with $e_1, e_2$ canonical). The scalar product of $U$ and $V$ as elements of $\mathcal L(\mathcal X, \mathcal Y) \cong \widecheck{\mathcal X} \otimes \mathcal Y$ is then the sum of the pointwise $m$-inner products $m(Ue_j, Ve_j)$, evaluated index-by-index. The basis-independence proposition (A.24) says this sum does not depend on which orthonormal frame we picked — a clean way to see why $\widecheck\ell \otimes m$ is intrinsically defined on operators rather than on matrices.</figcaption>
</figure> -->

Recall from (A.63) that $\mathcal{L}(\widecheck{\mathcal{X}}, \widecheck{\mathcal{Y}}) = \mathcal{L}(\mathcal{X}, \mathcal{Y})^{\vee}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">A.7 (Scalar product of tensor products: duality mapping)</span></p>

The duality mapping corresponding to the scalar product (A.25) is

$$
L^{-1} \otimes M \in \mathcal{L}\bigl(\mathcal{L}(\mathcal{X}, \mathcal{Y}),\, \mathcal{L}(\widecheck{\mathcal{X}}, \widecheck{\mathcal{Y}})\bigr). \tag{A.26}
$$

We denote the norm on $\mathcal{L}(\mathcal{X}, \mathcal{Y})$ induced by the scalar product (A.25) by

$$
(\widecheck{\lambda} \otimes \mu)(U) := \sqrt{(\widecheck{\ell} \otimes m)(U, U)}. \tag{A.27}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">A.8 (Orthogonal right-inverse)</span></p>

Let $A \in \mathcal{L}(\mathcal{X}', \mathcal{X})$ be injective and $B \in \mathcal{L}(\mathcal{Y}, \mathcal{Y}')$ be surjective. Then $\widecheck{A} \otimes B$ is surjective with the orthogonal right-inverse

$$
(\widecheck{A} \otimes B)^{+} = \widecheck{A}^{+} \otimes B^{+} \in \mathcal{L}\bigl(\mathcal{L}(\mathcal{X}', \mathcal{Y}'),\, \mathcal{L}(\mathcal{X}, \mathcal{Y})\bigr). \tag{A.28}
$$

Furthermore, the minimal $(\widecheck{\lambda} \otimes \mu)$-norm solution to the equation

$$
\begin{aligned}
B W A &= W' \in \mathcal{L}(\mathcal{X}', \mathcal{Y}') \quad \text{is} &&\text{(A.29a)} \\
W &= B^{+} W' A^{-} \in \mathcal{L}(\mathcal{X}, \mathcal{Y}). &&\text{(A.29b)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">A.9</span></p>

Assume $A \in \mathcal{L}(\mathcal{X}', \mathcal{X})$ is injective. Then the minimal $(\widecheck{\lambda} \otimes \mu)$-norm solution to the equation

$$
\begin{aligned}
W A &= W' \in \mathcal{L}(\mathcal{X}', \mathcal{Y}) \quad \text{is} &&\text{(A.30a)} \\
W &= W' A^{-} \in \mathcal{L}(\mathcal{X}, \mathcal{Y}), &&\text{(A.30b)}
\end{aligned}
$$

which does not depend on the scalar product $m$ on $\mathcal{Y}$.

</div>

### A.2. Multilinear Functions, Tensors on a Vector Space

### A.2.1. General Functions and Tensors, Duality, Scalar Products

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name"></span></p>

Let $E, F$ be vector spaces of $\mathbb{R}$. Let $\widecheck{E}$ be the dual space of $E$. Throughout this section, if $(e_i)\_{i \in [n]}$ is a basis of $E$ and $(\widecheck{e}^i)\_{i \in [n]}$ is a dual basis of $\widecheck{E}$, then

$$
\langle \widecheck{e}^i, e_j\rangle = \delta^i_j = \begin{cases} 1 & \text{if } i = j, \\ 0 & \text{otherwise.} \end{cases} \qquad \textbf{(Kronecker delta)} \tag{A.31}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($F$-valued Tensors)</span></p>

The space of **$F$-valued tensors**

$$
T^{p,q}(E, F), \qquad p, q \in \mathbb{N} \tag{A.32a}
$$

contains multilinear functions of the form

$$
f: \widecheck{E} \times \overset{p}{\cdots} \times \widecheck{E} \times E \times \overset{q}{\cdots} \times E \to F. \tag{A.32b}
$$

We set

$$
\begin{aligned}
T^{0,0}(E, F) &:= F, &&\text{(A.33a)} \\
T^{p,q}(E) &:= T^{p,q}(E, \mathbb{R}) &&\text{(A.33b)}
\end{aligned}
$$

with elements

$$
T^{p,q}(E) \ni u_1 \otimes \cdots \otimes u_p \otimes \widecheck{v}^1 \otimes \cdots \otimes \widecheck{v}^q, \qquad u_1, \dots, u_p \in E,\ \widecheck{v}^1, \dots, \widecheck{v}^q \in \widecheck{E} \tag{A.34a}
$$

defined by

$$
(u_1 \otimes \cdots \otimes \widecheck{v}^q)(\widecheck{x}^1, \dots, \widecheck{x}^p, y_1, \dots, y_q) = \widecheck{x}^1(u_1) \cdots \widecheck{x}^p(u_p)\, \widecheck{v}^1(y_1) \cdots \widecheck{v}^q(y_q). \tag{A.34b}
$$

</div>

Let $e_1, \dots, e_n \in E$ be a basis with dual basis $\widecheck{e}^1, \dots, \widecheck{e}^n$. Then any function $f \in T^{p,q}(E)$ can be uniquely specified as

$$
f = \sum f^{i_1 \cdots i_p}_{j_1 \cdots j_q}\, e_{i_1} \otimes \cdots \otimes e_{i_p} \otimes \widecheck{e}^{j_1} \otimes \cdots \otimes \widecheck{e}^{j_q} \tag{A.35}
$$

with coefficients

$$
f^{i_1 \cdots i_p}_{j_1 \cdots j_q} = f(\widecheck{e}^{i_1}, \dots, \widecheck{e}^{i_p}, e_{j_1}, \dots, e_{j_q}) \in \mathbb{R}. \tag{A.36}
$$

Here the sum ranges over all $p$-tuples $1 \leq i_1, \dots, i_p \leq n$ and all $q$-tuples $1 \leq j_1, \dots, j_q \leq n$ with $n = \dim(E)$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">A.10 (Tensors, Duality)</span></p>

**(1)** The coefficients (A.36) of a multilinear function $f \in T^{2,2}(E)$, with $n = \dim(E) = 2$, form a $2 \times 2 \times 2 \times 2$ array, here plotted after reshaping it to matrix form

$$
\begin{pmatrix}
\begin{pmatrix} f^{1,1}_{1,1} & f^{1,1}_{1,2} \\ f^{1,1}_{2,1} & f^{1,1}_{2,2} \end{pmatrix} & \begin{pmatrix} f^{1,2}_{1,1} & f^{1,2}_{1,2} \\ f^{1,2}_{2,1} & f^{1,2}_{2,2} \end{pmatrix} \\[8pt]
\begin{pmatrix} f^{2,1}_{1,1} & f^{2,1}_{1,2} \\ f^{2,1}_{2,1} & f^{2,1}_{2,2} \end{pmatrix} & \begin{pmatrix} f^{2,2}_{1,1} & f^{2,2}_{1,2} \\ f^{2,2}_{2,1} & f^{2,2}_{2,2} \end{pmatrix}
\end{pmatrix}
$$

**(2)** Let $(e_i)\_{i \in [n]}$ be a basis of $E$ and let $(f_j)\_{j \in [m]}$ be a basis of $F$. The bilinear form

$$
\langle \widecheck{x}, x \rangle := \widecheck{x}(x) \tag{A.37}
$$

on $\widecheck{E} \times E$ is also called **duality pairing** or **duality product**. The $j$th component of $x \in E$ in this basis is given by

$$
x^j = \langle \widecheck{e}^j, x\rangle, \qquad j \in [n]. \tag{A.38}
$$

The $j$th component of $\widecheck{x} \in \widecheck{E}$ in the dual basis is given by

$$
\widecheck{x}_j = \langle \widecheck{x}, e_j\rangle. \tag{A.39}
$$

Hence (cf. Remark A.1)

$$
\langle \widecheck{x}, x \rangle = \langle \widecheck{x}_i \widecheck{e}^i,\, x^j e_j\rangle = \widecheck{x}_i x^j \langle \widecheck{e}^i, e_j\rangle = \widecheck{x}_i x^j \delta^i_j = \widecheck{x}_i x^i. \tag{A.40}
$$

**(3)** Let $y \in F$ and the mapping $f_y \in T^{1,1}(E, F)$ be given by

$$
f_y(\widecheck{x}, x) = \langle \widecheck{x}, x\rangle\, y. \tag{A.41}
$$

This expression is also bilinear in $\widecheck{x}, y$. Hence we may define

$$
\boxed{\,x \mapsto (\widecheck{x} \otimes y)(x) := \widecheck{x}(x)\, y\,}. \tag{A.42}
$$

Thus

$$
\boxed{\,\widecheck{x} \otimes y \in \mathcal{L}(E, F)\,} \tag{A.43}
$$

becomes an element of the vector space of *linear* maps from $E$ to $F$ and we may identify

$$
\boxed{\,\widecheck{E} \otimes F \cong \mathcal{L}(E, F)\,}. \tag{A.44}
$$

Likewise

$$
E \otimes F \cong \mathcal{L}(\widecheck{E}, F) \tag{A.45}
$$

and

$$
\widecheck{E}_1 \otimes \cdots \otimes \widecheck{E}_q \otimes F \cong T^{0,q}(E, F) \tag{A.46}
$$

is the space of all multilinear functions

$$
f: E_1 \otimes \cdots \otimes E_q \to F. \tag{A.47}
$$

We write

$$
\widecheck{E}^{\otimes 2} := \widecheck{E} \otimes \widecheck{E} := \widecheck{E} \otimes \widecheck{E} \otimes \mathbb{R} \tag{A.48}
$$

for the space of all **real-valued bilinear forms** on $E$ and similarly

$$
\widecheck{E}^{\otimes q} := \widecheck{E} \otimes \overset{q}{\cdots} \otimes \widecheck{E} \qquad \text{for } 2 \leq q \in \mathbb{N}. \tag{A.49}
$$

**(4)** Let $(e_i)\_{i \in [n]}$ be a basis of $E$ and let $(f_j)\_{j \in [m]}$ be a basis of $F$. The mappings

$$
(\widecheck{e}^j \otimes f_i)_{\substack{i \in [m] \\ j \in [n]}} \tag{A.50}
$$

form a basis of the space $\mathcal{L}(E, F)$. The transformation of the basis $(e_j)\_{j \in [n]}$ determines the components of the matrix $W$ which represents the transformation in $\mathcal{L}(E, F)$ in the basis (A.50),

$$
\begin{aligned}
F \ni Wx &= \langle \widecheck{f}^i, Wx\rangle f_i = \langle \widecheck{f}^i, W(x^j e_j)\rangle f_i = \langle \widecheck{f}^i, W(\langle \widecheck{e}^j, x\rangle e_j)\rangle f_i &&\text{(A.51a)} \\
&= \langle \widecheck{f}^i, W e_j\rangle\, \langle \widecheck{e}^j, x\rangle\, f_i = \underbrace{\langle \widecheck{f}^i, W e_j\rangle}_{=:\,W^i_j}\, (\widecheck{e}^j \otimes f_i)(x) &&\text{(A.51b)} \\
&= (W^i_j\, \widecheck{e}^j \otimes f_i)(x). &&\text{(A.51c)}
\end{aligned}
$$

**(5)** A vector $x \in E$ is transformed by the operator (A.43) to

$$
(\widecheck{x} \otimes y)(x) = (\widecheck{x}_j \widecheck{e}^j) \otimes (y^i f_i)(x) = (y^i \widecheck{x}_j)(\widecheck{e}^j \otimes f_i)(x) = (W^i_j\, \widecheck{e}^j \otimes f_i)(x). \tag{A.52}
$$

Thus, in terms of the coordinates and matrix-vector notation, the matrix representing this linear mapping is

$$
W = (W^i_j)_{\substack{i \in [m] \\ j \in [n]}} = y\, \widecheck{x}^{\top}. \tag{A.53}
$$

**(6)** The **transpose**

$$
(\widecheck{x} \otimes y)^{\vee} \in \mathcal{L}(\widecheck{F}, \widecheck{E}) \tag{A.54}
$$

maps a vector $\widecheck{y} \in \mathcal{F}$ to

$$
\boxed{\,(\widecheck{x} \otimes y)^{\vee}(\widecheck{y}) = \langle y, \widecheck{y}\rangle\, \widecheck{x}\,} \tag{A.55}
$$

since

$$
\begin{aligned}
\langle \widecheck{y}, (\widecheck{x} \otimes y)(x)\rangle &= \langle \widecheck{y}, \widecheck{x}(x) y\rangle = \langle \widecheck{y}, \langle \widecheck{x}, x\rangle y\rangle = \langle\langle \widecheck{y}, y\rangle\, \widecheck{x},\, x\rangle &&\text{(A.56a)} \\
&= \langle (\widecheck{x} \otimes y)^{\vee}(\widecheck{y}),\, x\rangle. &&\text{(A.56b)}
\end{aligned}
$$

Consequently, because

$$
(\widecheck{x} \otimes y)^{\vee}(\widecheck{y}) = \langle y, \widecheck{y}\rangle\, \widecheck{x} = (y \otimes \widecheck{x})(\widecheck{y}) \tag{A.57}
$$

we identify

$$
\boxed{\,(\widecheck{x} \otimes y)^{\vee} \cong y \otimes \widecheck{x}\,} \in \mathcal{L}(\widecheck{F}, \widecheck{E}). \tag{A.58}
$$

The matrix representation of this map is the transposed matrix (A.53)

$$
W^{\top} = \widecheck{x}\, y^{\top}. \tag{A.59}
$$

</div>

<!-- <figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_22tensor_reshape.png' | relative_url }}" alt="A 4×4 heatmap divided by heavy lines into a 2×2 grid of 2×2 blocks. Each cell shows a numeric value and an index label f^{i₁i₂}_{j₁j₂}. The outer block rows and columns are labelled by i₁ and i₂; the inner cell rows and columns by j₁ and j₂." loading="lazy">
  <figcaption>The $(2,2)$-tensor $f \in T^{2,2}(E)$ from Example A.10(1) made tangible. With $\dim E = 2$, the coefficients $f^{i_1 i_2}_{j_1 j_2}$ form a $2 \times 2 \times 2 \times 2$ array of $16$ real numbers — too many dimensions to display directly. The trick used in Example A.10(1) is to read the upper indices $(i_1, i_2)$ as the address of an outer $2 \times 2$ <em>block</em>, and the lower indices $(j_1, j_2)$ as the address of a <em>cell</em> within that block. The four coloured outer blocks (separated by the heavy grid lines) are themselves $2 \times 2$ matrices indexed by $(j_1, j_2)$. This nested layout is exactly the matrix-of-matrices arrangement printed in the original definition.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_outer_product_rank1.png' | relative_url }}" alt="Three panels left-to-right. Left: input space E with the covector x̌ drawn as a family of equally-spaced parallel level lines, and an input vector x crossing several of them. Middle: an arrow indicating the scalar value ⟨x̌, x⟩ multiplies the direction y. Right: output space F showing the direction y (faded) and the scaled output ⟨x̌, x⟩ y along the same line. A small heatmap on the far right shows the rank-1 matrix W = y x̌^⊤." loading="lazy">
  <figcaption>The outer product $\widecheck x \otimes y$ acting on a vector $x \in E$, in three steps. <em>Left:</em> the covector $\widecheck x \in \widecheck E$ is drawn as the family of its level lines $\lbrace x : \langle\widecheck x, x\rangle = c\rbrace $ — equally-spaced parallel lines in $E$, with the dark line being the kernel ($c = 0$). The number $\langle\widecheck x, x\rangle$ counts how many level lines $x$ has crossed past zero. <em>Middle:</em> that scalar then scales the fixed direction $y \in F$ to produce the output $\langle\widecheck x, x\rangle\, y$, always lying along the same ray through $y$. <em>Right:</em> in matrix form (A.53), $W = y\,\widecheck x^\top$ is a rank-$1$ matrix — every column is a scalar multiple of $y$, so the image of $E$ collapses onto the one-dimensional subspace $\mathbb R\cdot y \subset F$.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_transpose_duality.png' | relative_url }}" alt="Two-row layout. Top: side-by-side schematic diagrams. Left shows an arrow E → F labelled x̌ ⊗ y; right shows an arrow F̌ → Ě labelled (x̌ ⊗ y)^v ≅ y ⊗ x̌. Bottom: two heatmaps — left, the 2×3 matrix W = y x̌^⊤; right, the 3×2 transposed matrix W^⊤ = x̌ y^⊤." loading="lazy">
  <figcaption>The transpose identity $(\widecheck x \otimes y)^\vee \cong y \otimes \widecheck x$ from (A.55)–(A.58). <em>Top:</em> the original map sends $x \in E$ to $\langle\widecheck x, x\rangle\,y \in F$ along the direction $y$; its transpose sends $\widecheck y \in \widecheck F$ to $\langle y, \widecheck y\rangle\,\widecheck x \in \widecheck E$ — the roles of "vector direction" and "covector measurement" swap between the two spaces. <em>Bottom:</em> the same statement at the matrix level — taking the transpose of $W = y\,\widecheck x^\top \in \mathbb R^{m\times n}$ gives $W^\top = \widecheck x\,y^\top \in \mathbb R^{n \times m}$, with rows and columns simply swapped.</figcaption>
</figure> -->

We focus on *linear* mappings

$$
\mathcal{A}: \mathcal{L}(E, F) \to G, \tag{A.60}
$$

that is, on elements $\mathcal{A} \in \mathcal{L}(\mathcal{L}(E, F), G)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">A.11 (Isomorphism)</span></p>

One has the isomorphism

$$
\mathcal{L}(\mathcal{L}(E, F), G) \cong \mathcal{L}(\widecheck{E}, \mathcal{L}(F, G)) \tag{A.61}
$$

which is identified by the equation

$$
G \ni \underbrace{\mathcal{A}(\overbrace{\widecheck{x} \otimes y}^{\in\, \mathcal{L}(E, F)})}_{\text{l.h.s.}} := \underbrace{\mathcal{A}(\widecheck{x})(y)}_{\text{r.h.s.}} \in G, \qquad \forall\, \widecheck{x} \in \widecheck{E},\ \forall\, y \in F. \tag{A.62}
$$

Note that when $G = \mathbb{R}$, then (A.61) characterizes **dual spaces** of linear operators

$$
(\widecheck{E} \otimes F)^{\vee} \cong \boxed{\,\mathcal{L}(E, F)^{\vee} \cong \mathcal{L}(\widecheck{E}, \widecheck{F})\,} \cong E \otimes \widecheck{F}. \tag{A.63}
$$

</div>

<!-- <figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_isomorphism_currying.png' | relative_url }}" alt="Two side-by-side block diagrams. Left: a single arrow from a large box labelled L(E, F) (containing the element W = x̌ ⊗ y) to a smaller box labelled G, with the arrow labelled 𝒜 and W ↦ 𝒜(W). Right: a chain of three boxes — Ě, then L(F, G), then G — connected by two arrows; the first labelled 𝒜 with x̌ ↦, the second labelled eval_y, and the middle box noting y ↦ 𝒜(x̌)(y)." loading="lazy">
  <figcaption>Theorem A.11 in pictures: the same operator $\mathcal A$ admits two equivalent readings. <em>Left:</em> as a single map $\mathcal L(E, F) \to G$, eating an entire linear map $W = \widecheck x \otimes y$ and returning an element of $G$. <em>Right:</em> in curried form, $\mathcal A$ is a map $\widecheck E \to \mathcal L(F, G)$: feed it just the covector part $\widecheck x$, get back another linear map $\mathcal A(\widecheck x) : F \to G$, then evaluate that on the vector part $y$. The defining identity $\mathcal A(\widecheck x \otimes y) = \mathcal A(\widecheck x)(y)$ from (A.62) is precisely the assertion that these two readings produce the same answer, so the two function spaces are isomorphic via uncurrying / currying.</figcaption>
</figure> -->

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">A.12 (Mapping $\mathcal{A} \in \mathcal{L}(\mathcal{L}(E, F), G)$)</span></p>

Let $x \in E$ and $B \in \mathcal{L}(F, G)$ and set

$$
\mathcal{A} = x \otimes B:\ W \in \mathcal{L}(E, F) \mapsto \mathcal{A}(W) = (x \otimes B)(W) := BWx \in G. \tag{A.64}
$$

On the one hand, this definition clearly reveals $\mathcal{A} = x \otimes B \in \mathcal{L}(\mathcal{L}(E, F), G)$ which *linearly* maps $W$ to an element of $G$. On the other hand, by the isomorphism (A.61), $\mathcal{A} = x \otimes B \in \mathcal{L}(\widecheck{E}, \mathcal{L}(F, G))$ which becomes explicit by the right-hand side of (A.62): representing the action of $W$ by $W = \widecheck{x} \otimes y$, one has

$$
\mathcal{A}(W) = (x \otimes B)(\widecheck{x} \otimes y) \stackrel{(A.64)}{=} B(\widecheck{x} \otimes y)x = \langle \widecheck{x}, x\rangle\, By = \underbrace{\mathcal{A}(\widecheck{x})}_{\in\, \mathcal{L}(F, G)}(y) \in G. \tag{A.65}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">A.13 (Dual spaces of Linear Operators)</span></p>

Based on the identification (A.63), one has $x \otimes \widecheck{y} \in \mathcal{L}(E, F)^{\vee}$ and

$$
\begin{aligned}
\langle x \otimes \widecheck{y},\, W \rangle &= \langle \widecheck{y}, Wx \rangle, &&\text{(A.66a)} \\
\langle x \otimes \widecheck{y},\, \widecheck{x} \otimes y \rangle &= \langle \widecheck{x}, x \rangle\, \langle \widecheck{y}, y \rangle. &&\text{(A.66b)}
\end{aligned}
$$

If $(e_j)\_{j \in [n]}, (\widecheck{e}^j)\_{j \in [n]}$ and $(f_i)\_{i \in [m]}, (\widecheck{f}^i)\_{i \in [m]}$ are dual bases of $E, \widecheck{E}$ and $F, \widecheck{F}$, then

$$
(\widecheck{e}^j \otimes f_i)_{\substack{i \in [m] \\ j \in [n]}} \qquad \text{and} \qquad (e_l \otimes \widecheck{f}^k)_{\substack{k \in [m] \\ l \in [n]}} \tag{A.67}
$$

are dual bases of $\mathcal{L}(E, F)$ and $\mathcal{L}(\widecheck{E}, \widecheck{F})$. Independent of the choice of the dual bases, the duality product for $\mathcal{L}(E, F)$ and $\mathcal{L}(\widecheck{E}, \widecheck{F}) \cong \mathcal{L}(E, F)^{\vee}$ is given with $U \in \mathcal{L}(E, F),\ \widecheck{V} \in \mathcal{L}(\widecheck{E}, \widecheck{F})$ by (recall the summation convention, Remark A.1)

$$
\begin{aligned}
\langle \widecheck{V}, U \rangle &:= \langle \widecheck{V} \widecheck{e}^j, f_i\rangle\, \langle \widecheck{f}^i, U e_j\rangle &&\text{(A.68a)} \\
&= \langle \widecheck{V} \widecheck{e}^j,\, U e_j\rangle = \langle \widecheck{U} \widecheck{f}^i,\, V f_i\rangle, &&\text{(A.68b)}
\end{aligned}
$$

with the transposed mappings $\widecheck{U} \in \mathcal{L}(\widecheck{F}, \widecheck{E}),\ V \in \mathcal{L}(F, E)$.

</div>

We generalize the mapping $\mathcal{A}$ in (A.64) to **tensor products of linear operators**. Given pairs of vector spaces $E, E'$ and $F, F'$, consider

$$
A \in \mathcal{L}(E', E) \qquad \text{and} \qquad B \in \mathcal{L}(F, F') \tag{A.69a}
$$

and the *linear* mapping

$$
\boxed{\,\widecheck{A} \otimes B \in \mathcal{L}(\mathcal{L}(E, F),\, \mathcal{L}(E', F'))\,}, \qquad \boxed{\,(\widecheck{A} \otimes B)(W) := BWA\,}. \tag{A.69b}
$$

This reduces to (A.64) by choosing $G = F' \cong \mathcal{L}(\mathbb{R}, F')$, i.e. $E' = \mathbb{R}$, and $A = x: E' = \mathbb{R} \ni \lambda \mapsto \lambda x \in E$. For the special case $W = \widecheck{x} \otimes y \in \mathcal{L}(E, F)$, (A.69) generalizes the equation $(x \otimes B)(\widecheck{x} \otimes y) = \langle \widecheck{x}, x\rangle\, By$ as given by (A.65) to

$$
(\widecheck{A} \otimes B)(\widecheck{x} \otimes y) = \underbrace{\widecheck{A}\widecheck{x}}_{\in\, \widecheck{E}'} \otimes\, By \in \mathcal{L}(E', F'). \tag{A.70}
$$

<!-- <figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_tensor_product_operators.png' | relative_url }}" alt="A four-stage horizontal pipeline: rounded boxes labelled E', E, F, F' connected by arrows labelled A (purple), W (red), and B (purple) in turn. Below, the formula (Ǎ ⊗ B)(W) := BWA ∈ L(E', F') and the action on outer products (Ǎ ⊗ B)(x̌ ⊗ y) = (Ǎx̌) ⊗ (By)." loading="lazy">
  <figcaption>The tensor product of operators $\widecheck A \otimes B$ from (A.69) as a four-stage pipeline. Given $W \in \mathcal L(E, F)$ — an arbitrary linear map between the "inner" spaces — the operator $\widecheck A \otimes B$ pre-composes with $A$ on the input side and post-composes with $B$ on the output side, producing $BWA \in \mathcal L(E', F')$. The pipeline view makes clear why $\widecheck A \otimes B$ acts on the entire space of operators $\mathcal L(E, F)$ rather than on individual vectors. On rank-$1$ operators it factors as in (A.70): $(\widecheck A \otimes B)(\widecheck x \otimes y) = (\widecheck A\widecheck x) \otimes (By)$, with $A$ pulling back the covector and $B$ pushing forward the vector.</figcaption>
</figure> -->

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">A.14 (Transposed tensor products of Linear Operators)</span></p>

The transpose of the linear operator (A.69) is

$$
\boxed{\,(\widecheck{A} \otimes B)^{\vee} = A \otimes \widecheck{B} \in \mathcal{L}(\mathcal{L}(\widecheck{E}', \widecheck{F}'),\, \mathcal{L}(\widecheck{E}, \widecheck{F}))\,}. \tag{A.71}
$$

Now assume (A.69) and in addition 

$$A' \in \mathcal{L}(E'', E'), \qquad B' \in \mathcal{L}(F', F'')$$

to be given. Then, with 

$$\widecheck{A'} \otimes B' \in \mathcal{L}(\mathcal{L}(E', F'),\, \mathcal{L}(E'', F''))$$

one has

$$
\begin{aligned}
\boxed{\,(\widecheck{A'} \otimes B')(\widecheck{A} \otimes B)\,} &= (\underbrace{AA'}_{\in\, \mathcal{L}(E'', E)})^{\vee} \otimes (\underbrace{B'B}_{\in\, \mathcal{L}(F, F'')}) &&\text{(A.72a)} \\
&\boxed{\,= \widecheck{(A'A)} \otimes (B'B) \in \mathcal{L}(\mathcal{L}(E, F),\, \mathcal{L}(E'', F''))\,}. &&\text{(A.72b)}
\end{aligned}
$$

In addition, one has:

$$
\begin{aligned}
A \text{ and } B \text{ are invertible} &\;\Longrightarrow\; (\widecheck{A} \otimes B)^{-1} = \widecheck{A^{-1}} \otimes B^{-1} &&\text{(A.73a)} \\
A \text{ left-invertible (injective)},\ B \text{ right-invertible (surjective)} &\;\Longrightarrow\; \widecheck{A} \otimes B \text{ right-invertible (surjective)} &&\text{(A.73c)} \\
A \text{ right-invertible (surjective)},\ B \text{ left-invertible (injective)} &\;\Longrightarrow\; \widecheck{A} \otimes B \text{ left-invertible (injective)} &&\text{(A.73e)} \\
A \text{ and } B \text{ are projections} &\;\Longrightarrow\; \widecheck{A} \otimes B \text{ is a projection.} &&\text{(A.73f)}
\end{aligned}
$$

</div>

## 1. Multilayer Networks on Euclidean Spaces, Associative Memories

### 1.1. Minimizing Quadratic Functionals over Affine Subspaces

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Minimizer of a quadratic functional over an affine subspace)</span></p>

This section applies the Euclidean-space machinery from Appendix A.1. The basic object is a constrained least-distance problem: given $u \in \mathcal{X}$, $y \in \mathcal{Y}$, and a linear map $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$, minimize the quadratic distance to $u$ over the affine subspace of points satisfying $Ax = y$:

$$
\min_{x:\, Ax = y} \frac{1}{2}\lambda(x-u)^2. \tag{1.1}
$$

Here $\lambda$ is the norm induced by the scalar product on $\mathcal{X}$, and $L : \mathcal{X} \to \widecheck{\mathcal{X}}$ is the corresponding duality mapping from Appendix A.1.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">1.1 (Minimizer of a quadratic functional over an affine subspace)</span></p>

Assume that $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ is **surjective**. Then the unique solution of (1.1) is

$$
\begin{aligned}
x^* &= u - L^{-1}\widecheck{A}\widecheck{y}^* &&\text{(1.2a)} \\
&= u - A^+(Au-y), &&\text{(1.2b)}
\end{aligned}
$$

where the optimal multiplier vector is

$$
\widecheck{y}^* = (A L^{-1} \widecheck{A})^{-1}(Au-y). \tag{1.2c}
$$

Equivalently, $\widecheck{y}^*$ solves the dual problem

$$
\min_{\widecheck{y} \in \widecheck{\mathcal{Y}}}
\left\lbrace 
\frac{1}{2}\widecheck{\lambda}\!\left(\widecheck{A}\widecheck{y} - Lu\right)^2
+ \langle \widecheck{y}, y\rangle
\right\rbrace . \tag{1.2d}
$$

TODO: What is the interpretation of the result of the proposition?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $A$ is surjective, the affine subspace $\lbrace x \in \mathcal{X} : Ax = y\rbrace$ is non-empty. The functional $\frac{1}{2}\lambda(x-u)^2$ is strictly convex, so its restriction to this affine subspace has at most one minimizer.

Introduce the Lagrangian

$$
\mathscr{L}(x,\widecheck{y})
= \frac{1}{2}\lambda(x-u)^2
+ \langle \widecheck{y}, Ax-y\rangle .
$$

**Note:** we use $\langle \widecheck{y}, Ax-y\rangle$, because it is duality pairing (covector+vector). In ACO we used (vector+vector).

The first variation in the $x$-variable gives, for all $h \in \mathcal{X}$,

$$
0
= \left.\frac{d}{dt}\right|_{t=0}\mathscr{L}(x+th,\widecheck{y})
= \langle L(x-u), h\rangle + \langle \widecheck{A}\widecheck{y}, h\rangle .
$$

Hence

$$
L(x-u) + \widecheck{A}\widecheck{y}=0,
\qquad
x = u - L^{-1}\widecheck{A}\widecheck{y}. \tag{1.2e}
$$

Imposing the constraint $Ax=y$ gives

$$
A L^{-1}\widecheck{A}\widecheck{y} = Au-y. \tag{1.2f}
$$

**It is the moment, where whe the strength of the argument increases significantly:** 

> the operator $A L^{-1}\widecheck{A} : \widecheck{\mathcal{Y}} \to \mathcal{Y}$ is invertible. 

Indeed, if $\widecheck{y}\ne 0$, then surjectivity of $A$ implies $\widecheck{A}\widecheck{y}\ne 0$, and therefore

$$
\langle \widecheck{y}, A L^{-1}\widecheck{A}\widecheck{y}\rangle
= \langle \widecheck{A}\widecheck{y}, L^{-1}\widecheck{A}\widecheck{y}\rangle
= \widecheck{\lambda}(\widecheck{A}\widecheck{y})^2
> 0.
$$

which means for non-zero $\widecheck{y}$ is never in the kernel of $A L^{-1}\widecheck{A}$. Thus

$$
\widecheck{y}^* = (A L^{-1}\widecheck{A})^{-1}(Au-y),
$$

and substitution into $x = u - L^{-1}\widecheck{A}\widecheck{y}$ gives (1.2a). By the definition of the orthogonal right-inverse (A.13),

$$
L^{-1}\widecheck{A}(A L^{-1}\widecheck{A})^{-1} = A^+,
$$

so (1.2b) follows.

It remains to check the dual formulation. Let

$$
\Phi(\widecheck{y})
=
\frac{1}{2}\widecheck{\lambda}(\widecheck{A}\widecheck{y}-Lu)^2
+ \langle \widecheck{y}, y\rangle .
$$

For every $\widecheck{z}\in \widecheck{\mathcal{Y}}$,

$$
\begin{aligned}
D\Phi(\widecheck{y})[\widecheck{z}]
&= \left\langle \widecheck{A}\widecheck{z}, L^{-1}(\widecheck{A}\widecheck{y}-Lu)\right\rangle
+ \langle \widecheck{z}, y\rangle \\
&= \left\langle \widecheck{z},
A L^{-1}\widecheck{A}\widecheck{y} - Au + y
\right\rangle .
\end{aligned}
$$

The critical point condition is exactly (1.2f). Since the quadratic part is strictly convex by the same positivity argument above, this critical point is the unique minimizer of (1.2d), namely $\widecheck{y}^\ast$.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Note on the Proof I: Will the lagrangian solution give us the optimal solution of the initial problem?</summary>

Introduce the Lagrangian

$$
\mathscr{L}(x,\widecheck{y})
= \frac{1}{2}\lambda(x-u)^2
+ \langle \widecheck{y}, Ax-y\rangle .
$$

In the present problem the Lagrange equations are not merely necessary conditions. Since the objective is strictly convex and the constraint set $\lbrace x:Ax=y\rbrace$ is affine, any feasible $x^\ast$ for which there exists $\widecheck{y}^\ast\in\widecheck{\mathcal Y}$ satisfying

$$L(x^*-u)+\widecheck A\widecheck y^*=0$$

is automatically the unique global minimizer. Indeed, for every feasible $z$, one has $A(z-x^\ast)=0$, and hence

$$
\begin{aligned}
\frac12\lambda(z-u)^2
&=
\frac12\lambda(x^*-u)^2
+
\langle L(x^*-u),z-x^*\rangle
+
\frac12\lambda(z-x^*)^2 \\
&=
\frac12\lambda(x^*-u)^2
-
\langle \widecheck A\widecheck y^*,z-x^*\rangle
+
\frac12\lambda(z-x^*)^2 \\
&=
\frac12\lambda(x^*-u)^2
-
\langle \widecheck y^*,A(z-x^*)\rangle
+
\frac12\lambda(z-x^*)^2 \\
&=
\frac12\lambda(x^*-u)^2
+
\frac12\lambda(z-x^*)^2.
\end{aligned}
$$

Therefore $x^\ast$ is the unique minimizer.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Note on the Proof II: but what is the global optimum of the strictly convex function not in the feasible set? Could not we in this case have multiple minimizers (which are not globally optimal)?</summary>

Yes, the **unconstrained** global optimum may be outside the feasible set. That is exactly why we need a **constrained** optimum.

For your functional

$$f(x)=\frac12\lambda(x-u)^2$$,

the unconstrained global minimizer is clearly $x=u$. If $Au=y$, then $u$ is feasible and the constrained minimizer is just $x^\ast=u$. But if $Au\neq y$, then $u$ is not allowed. The constrained problem is then:

$$\min_{x\in C} f(x), \qquad C:=\lbrace x\in\mathcal X:Ax=y\rbrace.$$

So the question becomes:

> Among the feasible points $x\in C$, which one is closest to $u$?

That is a different minimization problem. Its global optimum is the **projection of $u$ onto the affine subspace $C$**.

---

**Strict convexity still prevents multiple constrained minimizers.**

The feasible set $C=\lbrace x:Ax=y\rbrace$ is affine, hence convex. Since $f$ is strictly convex on $\mathcal X$, its restriction to $C$ is also strictly convex.

So if $x_1,x_2\in C$, $x_1\neq x_2$, and both were constrained minimizers, then their midpoint $\frac{x_1+x_2}{2}\in C$ would also be feasible. But strict convexity gives

$$f\left(\frac{x_1+x_2}{2}\right) < \frac12 f(x_1)+\frac12 f(x_2).$$

If both $x_1$ and $x_2$ had the same minimal value, this would produce an even smaller feasible value, contradiction.

Therefore there can be **at most one constrained minimizer**.

---

**There are no non-global local minimizers either.**

For a convex function on a convex feasible set, any local minimizer is already global. Indeed, suppose $x\in C$ is a local minimizer but not a global minimizer. Then there exists $z\in C$ with $f(z)<f(x)$. Because $C$ is convex, the whole segment $x_t=(1-t)x+tz,\qquad t\in[0,1],$ lies in $C$. By convexity,

$$f(x_t) \leq (1-t)f(x)+tf(z) < f(x)$$

for every $t>0$. For small $t$, $x_t$ is arbitrarily close to $x$, contradicting local minimality.

So in this affine-constrained strictly convex problem, you cannot have “bad” local constrained minimizers.

---

**What the Lagrange multiplier condition says geometrically.**

If $u\notin C$, then the constrained minimizer $x^\ast$ does **not** satisfy $Df(x^\ast)=0.$ That would only happen at $x^\ast=u$. Instead, it satisfies

$$Df(x^*)[h]=0 \qquad \text{for every feasible direction } h\in\ker A.$$

Equivalently, $L(x^\ast-u)$ is orthogonal, in the duality-pairing sense, to all feasible directions. That means $x^\ast-u$ is normal to the affine constraint set.

The Lagrange multiplier equation $L(x^\ast-u)+\widecheck A\widecheck y^\ast=0$ expresses exactly this: the gradient of the objective lies in the normal space generated by the constraint.

So the unconstrained minimizer $u$ may be infeasible, but the constrained minimizer is still unique and global because the feasible set is convex and the objective is strictly convex.

</details>
</div>

The proposition says that the nearest feasible point is obtained by correcting $u$ in the directions controlled by $A$. Since $A$ is surjective, the affine constraint $Ax=y$ is always feasible, and the pseudo-inverse $A^+$ gives the minimum-norm correction that changes $Au$ into $y$.

In the sequel, Proposition 1.1 is reused in three scenarios. Each scenario keeps the same optimization template, but changes the concrete choices of the spaces $\mathcal{X}$ and $\mathcal{Y}$ and of the duality mapping $L$.

### 1.2. Single Linear Associative Memory

We begin with the simplest neural network architecture: a single linear map trained on input-output pairs. Given data, the operator is selected as the minimizer of a constrained quadratic functional: it must map the training inputs to the prescribed outputs while staying as close as possible to a reference operator. This lets the map **recall** outputs associated with known inputs and generalize, to some degree, to perturbed inputs. This function is called **associative memory**.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Single Linear Associative Memory)</span></p>

The mathematical background is the operator-space scalar product (A.25) and induced norm (A.27) on $\mathcal{L}(\mathcal{X}, \mathcal{Y})$. Let

$$
\mathcal{D}_n = \lbrace (x_1,y_1), \ldots, (x_n,y_n)\rbrace  \subset \mathcal{X} \times \mathcal{Y} \tag{1.3}
$$

be a collection of input-output pairs. We want to determine a linear map

$$
W \in \mathcal{L}(\mathcal{X}, \mathcal{Y}) \tag{1.4a}
$$

such that

$$
W x_k = y_k, \qquad \forall\, k \in [n]. \tag{1.4b}
$$

The Gramian matrix of the input data is

$$
G = (G_{kl})_{k,l \in [n]}, \qquad G_{kl} := \ell(x_k,x_l), \tag{1.5a}
$$

and its inverse is denoted by

$$
G^{-1} = (G^{kl})_{k,l \in [n]}, \qquad G_{jk}G^{kl} = \delta^l_j, \qquad j,l \in [n]. \tag{1.5b}
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">1.2 (Linear associative memory)</span></p>

Assume that the input vectors $(x_k)\_{k \in [n]}$ are linearly independent, and let $U \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be given. Then there exists a unique operator

$$
W_{\mathcal{D}} := U - G^{kl}(Lx_l) \otimes (Ux_k - y_k) \in \mathcal{L}(\mathcal{X}, \mathcal{Y}) \tag{1.6}
$$

satisfying the interpolation constraints (1.4b) and minimizing the distance

$$
(\widecheck{\lambda} \otimes \mu)(W-U)
$$

from $W$ to $U$. If, in addition, the input vectors $(x_k)\_{k \in [n]}$ are $\ell$-orthogonal, then

$$
W_{\mathcal{D}} = U - (Lx_k) \otimes \frac{Ux_k-y_k}{\lambda(x_k)^2}. \tag{1.7}
$$

If $\mathcal{D} = \mathcal{D}\_n = \mathcal{D}\_{n-1} \cup \lbrace (x_n,y_n)\rbrace$ and $U = W_{\mathcal{D}\_{n-1}}$ satisfies the constraints (1.4b) for $\mathcal{D}\_{n-1}$, then

$$
W_{\mathcal{D}_n} = W_{\mathcal{D}_{n-1}} - G^{nl}(Lx_l) \otimes (W_{\mathcal{D}_{n-1}}x_n - y_n) \qquad \text{(with } n \text{ fixed).} \tag{1.8}
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $E = \mathbb{R}^n$ with canonical basis $(\delta_k)\_{k \in [n]}$, and define

$$
A \in \mathcal{L}(E,\mathcal{X}), \qquad A\delta_k := x_k,
\qquad
Y \in \mathcal{L}(E,\mathcal{Y}), \qquad Y\delta_k := y_k .
$$

The interpolation constraints are equivalent to

$$
WA = Y.
$$

Since the vectors $(x_k)\_{k \in [n]}$ are linearly independent, $A$ is injective. Thus Corollary A.9 applies to the correction $V := W-U$: among all $V$ satisfying

$$
VA = Y-UA,
$$

the unique minimal $(\widecheck{\lambda}\otimes\mu)$-norm correction is

$$
V = (Y-UA)A^{-}.
$$

The orthogonal left-inverse of $A$ is

$$
A^{-} = (\widecheck{A}LA)^{-1}\widecheck{A}L.
$$

In the basis $(\delta_k)$, the operator $\widecheck{A}LA$ has matrix

$$
(\widecheck{A}LA)_{kl}
= \langle Lx_l,x_k\rangle
= \ell(x_l,x_k)
= G_{kl},
$$

and therefore

$$
A^{-}x
= G^{kl}\langle Lx_l,x\rangle\,\delta_k .
$$

Consequently,

$$
\begin{aligned}
Vx
&= (Y-UA)A^{-}x \\
&= G^{kl}\langle Lx_l,x\rangle\,(y_k-Ux_k),
\end{aligned}
$$

or, as an operator,

$$
V = -G^{kl}(Lx_l)\otimes(Ux_k-y_k).
$$

Hence

$$
W = U+V
= U - G^{kl}(Lx_l)\otimes(Ux_k-y_k),
$$

which is (1.6). Since it was obtained from the minimal-norm solution of $VA=Y-UA$, it satisfies $WA=Y$, i.e. $Wx_k=y_k$ for all $k$, and is the unique minimizer of $(\widecheck{\lambda}\otimes\mu)(W-U)$ under these constraints.

If the input vectors are $\ell$-orthogonal, then $G_{kl}=0$ for $k\ne l$ and $G_{kk}=\lambda(x_k)^2$. Hence

$$
G^{kl}(Lx_l)\otimes(Ux_k-y_k)
= (Lx_k)\otimes\frac{Ux_k-y_k}{\lambda(x_k)^2},
$$

which gives (1.7).

Finally, suppose $\mathcal{D}\_n=\mathcal{D}\_{n-1}\cup\lbrace(x_n,y_n)\rbrace$ and take $U=W_{\mathcal{D}\_{n-1}}$. Then $Ux_k=y_k$ for all $k<n$, so in (1.6) all residuals $Ux_k-y_k$ vanish except the one with $k=n$. Therefore

$$
W_{\mathcal{D}_n}
= W_{\mathcal{D}_{n-1}}
- G^{nl}(Lx_l)\otimes(W_{\mathcal{D}_{n-1}}x_n-y_n),
$$

which is (1.8).

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">1.3 (Structure of the learning rule, generalization)</span></p>

**(a)** The variational problem determines $W_{\mathcal{D}}$ as a correction of the reference operator $U$. Consequently, formula (1.7) specifies how $W_{\mathcal{D}}$ can be decomposed into an iterative learning rule (1.8), which extends to storing novel input-output pairs in the associative memory.

**(b)** The iterative learning rule (1.8) corrects a given operator by adjusting the wiring from inputs to outputs: it correlates all input vectors with the residual vector of the current input-output pair. This mirrors mechanisms known from the research literature on natural neural systems, often called **Hebbian learning**.

**(c)** Although the architecture is only a single linear layer, there are still additional degrees of freedom for learning. In particular, the scalar product of the input space determines its geometry and therefore affects how the network generalizes to unseen data.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">1.4 (Linear Associative Memory)</span></p>

We apply Theorem 1.2 to the data depicted in Figure 1.1(a) and (b). From a few thousand samples of the digit $0$, with $\dim = 784$, representative vectors were selected by unsupervised clustering. The goal is to compute a data-specific inner product $\ell(x,x')$ according to Lemma A.2.

Panel (c) shows the transformed input vectors

$$
W_{\mathcal{D}}x_k \approx y_k, \qquad k \in [n].
$$

The approximation is not very good, but improves substantially after binarization.

The example suggests three lessons:

* Treating discrete, here binary, data as real-valued data is not a good idea.
* Almost linearly dependent input vectors cause numerically ill-conditioned evaluations of data-specific inner products. Regularization is needed for the numerical computation, and the predicted outputs are sensitive to spurious numerical errors, as revealed by the varying backgrounds in panel (c).
* Panel (d) suggests that respecting the discrete nature of the data during inference and learning should improve prediction.

</div>

### 1.3. Two Auto-Associative Settings, Linear Auto-Associative Memory

Section 1.2 treated the case where the *operator itself* is the unknown that is trained on input-output pairs. We now turn to a complementary family of networks: the operator class is *fixed in form*, but the operator depends on a **control vector** $u \in \mathcal{U}$ that is to be learned. The same data-interpolation principle of Proposition 1.1 will still apply, only the parameter space changes.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Affine-in-control network family)</span></p>

The network is the parametric mapping

$$
F : \mathcal{X} \times \mathcal{U} \to \mathcal{Y}, \qquad F(x,u) = c(x) + G(x)\,u, \tag{1.9a}
$$

assuming that

$$
c : \mathcal{X} \to \mathcal{Y} \qquad \text{is continuous,} \tag{1.9b}
$$

$$
G : \mathcal{X} \to \mathcal{L}(\mathcal{U}, \mathcal{Y}) \qquad \text{is continuous,} \tag{1.9c}
$$

$$
G(x) \in \mathcal{L}(\mathcal{U}, \mathcal{Y}) \qquad \text{is surjective}, \quad \forall x \in \mathcal{X}. \tag{1.9d}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation: control-theoretic view)</span></p>

The structure (1.9) is motivated by the control theory of *nonlinear* systems, where more generally *dynamical* rather than *static* systems like (1.9) -- which depend **affinely** on the control input vector $u$ -- form a (more) tractable class of systems. Three structural observations:

* The network mapping $F$ given by (1.9a) is **nonlinear** in the input $x$. Linearity is reserved only for the dependence on the control $u$; this is precisely what makes the learning problem analytically tractable.
* The maps $c$ and $G$ may involve **component-wise activation functions**, provided the assumptions (1.9b)-(1.9d) are satisfied. The nonlinearity is therefore not just decorative -- it can absorb the usual elementwise activations one would put inside a layer.
* The network realized by $F$ is **adaptive**, since both the offset $c(x)$ and the input-to-output coupling $G(x)$ depend on $x$. Different inputs are routed through different linear maps acting on the control.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Control-input space)</span></p>

We complement the basic Euclidean set-up of Appendix A.1.1 by the finite-dimensional

$$
\mathcal{U} \qquad \text{vector space of \textbf{control inputs}}, \tag{1.10}
$$

supplied with the scalar product and duality mapping

$$
\nu(u, u') := \langle Nu, u' \rangle, \qquad N \in \mathcal{L}(\mathcal{U}, \widecheck{\mathcal{U}}). \tag{1.11}
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Learning problem with control inputs)</span></p>

Given a training set

$$
\mathcal{D}_n = \lbrace (x_1, y_1), \ldots, (x_n, y_n) \rbrace \subset \mathcal{X} \times \mathcal{Y}, \tag{1.12}
$$

we wish to find a single control vector $u_{\mathcal{D}} \in \mathcal{U}$ such that

$$
F(x_k, u_{\mathcal{D}}) = y_k, \qquad \forall\, k \in [n]. \tag{1.13}
$$

</div>

Thus the entire dataset is to be explained by **one** control vector $u_{\mathcal{D}} \in \mathcal{U}$ which, plugged into the affine-in-control family (1.9a), reproduces every prescribed output. By assumption (1.9d), the transposed mappings

$$
\widecheck{G}(x) := \widetilde{G(x)} \in \mathcal{L}\!\left(\widecheck{\mathcal{Y}}, \underbrace{\mathrm{rge}(\widecheck{G}(x))}_{\subseteq\, \widecheck{\mathcal{U}}}\right) = \mathcal{L}\!\left(\widecheck{\mathcal{Y}}, \underbrace{(\ker G(x))^{\perp}}_{\subseteq\, \widecheck{\mathcal{U}}}\right) \tag{1.14}
$$

are one-to-one -- a duality reflection of $G(x)$ being surjective. To ensure that the per-pattern learning increments are non-interfering, we impose the following structural condition on the training inputs.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orthogonality condition)</span></p>

$$
G(x_k)\, N^{-1}\, \widecheck{G}(x_{k'}) = 0, \qquad \forall\, k \neq k' \in [n]. \tag{1.15}
$$

Equivalently, the subspaces $\mathrm{rge}(\widecheck{G}(x_k))$, $k \in [n]$, are **mutually orthogonal** in $\widecheck{\mathcal{U}}$ with respect to the inner product induced by $N^{-1}$.

</div>

The following result specifies how the control vector $u\_{\mathcal{D}}$ in (1.13) can be iteratively determined by processing in turn each pair of training patterns in $\mathcal{D}_n$. Recall Definition A.3(b).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">1.5 (Learning algorithm)</span></p>

Assume (1.9) and (1.15). Then $u_{\mathcal{D}}$ solving (1.13) is determined by

$$
u_{\mathcal{D}} = u_n, \tag{1.16a}
$$

$$
u_k = u_{k-1} - G(x_k)^{+} \big( c(x_k) + G(x_k)\, u_{k-1} - y_k \big), \qquad k \in [n], \tag{1.16b}
$$

$$
u_0 = 0, \tag{1.16c}
$$

with the orthogonal right-inverse

$$
G(x)^{+} = N^{-1}\, \widecheck{G}(x)\, \big( G(x)\, N^{-1}\, \widecheck{G}(x) \big)^{-1}. \tag{1.16d}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reading the iterates (1.16))</span></p>

The update (1.16b) is exactly the constrained-least-distance correction of Proposition 1.1, transported from the operator setting of Section 1.2 to the control-vector setting. Three things are worth flagging:

* **The residual.** The quantity $c(x_k) + G(x_k)\, u_{k-1} - y_k = F(x_k, u_{k-1}) - y_k$ is the prediction error at step $k$ when the network uses the current control $u_{k-1}$. The iterate moves $u_{k-1}$ in the direction that *kills this residual* on the next forward pass at $x_k$.
* **The right-inverse $G(x_k)^{+}$.** Surjectivity of $G(x_k)$ (assumption (1.9d)) makes the operator $G(x)\, N^{-1}\, \widecheck{G}(x) \in \mathcal{L}(\widecheck{\mathcal{Y}}, \mathcal{Y})$ invertible, so (1.16d) is well-defined. The map $G(x_k)^{+}$ is exactly the minimum-$\nu$-norm correction that achieves $G(x_k)\,(u_k - u_{k-1}) = -(F(x_k, u_{k-1}) - y_k)$, in direct analogy with (1.2b).
* **Why orthogonality.** The condition (1.15) guarantees that the correction made at step $k$ lies in a subspace of $\mathcal{U}$ that is $\nu$-orthogonal to all *previous* correction directions. As a result, fitting pattern $k$ does not destroy the fits obtained for patterns $1, \ldots, k-1$, so the single sweep (1.16b) is enough -- no outer loop is required.

</div>

### 1.4. Extension to Latent Spaces

We now generalize the affine-in-control family (1.9) of §1.3 by interposing two **latent spaces** $\mathcal{Z}_1, \mathcal{Z}_2$ between input and output, and by replacing the fixed control vector $u \in \mathcal{U}$ with an input-dependent latent vector $W\psi(x) \in \mathcal{Z}_2$. The free parameter to be learned is now the linear operator $W$ between the two latent spaces, while the nonlinear embeddings $c, \psi, \Phi$ are taken as given.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Latent-space network family)</span></p>

The network is the parametric mapping

$$
F : \mathcal{X} \to \mathcal{Y}, \qquad F(x) = y_1 + y_2 := c(x) + \Phi(x)\, W\, \psi(x), \tag{1.17a}
$$

assuming that

$$
c : \mathcal{X} \to \mathcal{Y} \qquad \text{is continuous,} \tag{1.17b}
$$

$$
\psi : \mathcal{X} \to \mathcal{Z}_1 \qquad \text{is continuous,} \tag{1.17c}
$$

$$
\Phi : \mathcal{X} \to \mathcal{L}(\mathcal{Z}_2, \mathcal{Y}) \qquad \text{is continuous,} \tag{1.17d}
$$

$$
\forall x \in \mathcal{X}, \qquad \Phi(x) \in \mathcal{L}(\mathcal{Z}_2, \mathcal{Y}) \quad \text{is surjective,} \tag{1.17e}
$$

and, for a given data set (1.12), that the input vectors transformed to the first latent space

$$
\bigl(\psi(x)\bigr)_{x \in \mathcal{D}_n} \qquad \text{are mutually orthonormal,} \tag{1.17f}
$$

and that the subsequent transformation to the second latent space

$$
W \in \mathcal{L}(\mathcal{Z}_1, \mathcal{Z}_2) \tag{1.17g}
$$

has been determined ('trained') as specified below.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reading (1.17a))</span></p>

Note the similarity of the network (1.17a) to (1.9a), with the control input $u$ replaced by a more expressive linear mapping $W$ between two latent spaces $\mathcal{Z}_1, \mathcal{Z}_2$. The input flows through the architecture as

$$
x \;\xrightarrow{\psi}\; \mathcal{Z}_1 \;\xrightarrow{W}\; \mathcal{Z}_2 \;\xrightarrow{\Phi(x)}\; \mathcal{Y}, \qquad \text{plus the offset } c(x).
$$

* $\psi$ is a fixed nonlinear **embedding** of the input into a first latent space. The orthonormality condition (1.17f) plays the same role as the orthogonality condition (1.15) of §1.3: it guarantees that the per-pattern updates do not interfere.
* $W$ is the linear operator to be **learned** from data, playing the role that the control vector $u$ played in (1.9a) -- but now living in the operator space $\mathcal{L}(\mathcal{Z}_1, \mathcal{Z}_2)$, hence carrying many more parameters.
* $\Phi(x)$ is an input-adaptive linear **read-out** mapping the second latent space back to the output. Its dependence on $x$ is what makes the family strictly more expressive than (1.9a): even though the trainable piece $W$ is linear, it is sandwiched between $x$-dependent maps $\psi$ and $\Phi(x)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">1.6 (Learning algorithm)</span></p>

Assume (1.17b)-(1.17f). Then the linear latent-layer operator

$$
W_{\mathcal{D}} = W_n \tag{1.18}
$$

that achieves the input-output data correspondence (1.13) for the network mapping (1.17a) is determined by

$$
W_k = W_{k-1} - \bigl( (L_1 \psi(x_k)) \otimes \Phi(x_k)^{+} \bigr) \bigl( c(x_k) + \Phi(x_k)\, W_{k-1}\, \psi(x_k) - y_k \bigr), \qquad W_0 = 0. \tag{1.19}
$$

Here $L_1 : \mathcal{Z}_1 \to \widecheck{\mathcal{Z}}_1$ denotes the duality mapping on the first latent space, and $\Phi(x_k)^{+}$ is the orthogonal right-inverse of $\Phi(x_k)$, defined as in (1.16d).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Three theorems, one template)</span></p>

The mathematical similarity of the approaches leading to the increasingly expressive learning rules and corresponding architectures (**Theorems 1.2, 1.5 and 1.6**) is by design. Each theorem instantiates Proposition 1.1 in a different choice of parameter space:

* **Theorem 1.2** -- the unknown is the operator $W \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ itself, regularized toward a reference operator $U$.
* **Theorem 1.5** -- the unknown is a control vector $u_{\mathcal{D}} \in \mathcal{U}$ entering the network affinely through $F(x,u) = c(x) + G(x)\,u$.
* **Theorem 1.6** -- the unknown is a latent linear map $W \in \mathcal{L}(\mathcal{Z}_1, \mathcal{Z}_2)$ acting between two embedded spaces, with input-dependent read-out $\Phi(x)$.

In particular, since nonlinearities ('*activation functions*') only act componentwise, *linear analysis and algebra on tensor spaces* suffice to determine the linear layer operators in closed form and to learn them through a sequential process.

The residual $c(x_k) + \Phi(x_k)\, W_{k-1}\, \psi(x_k) - y_k = F\_{W_{k-1}}(x_k) - y_k$ is exactly the prediction error of the current network at pattern $k$. The tensor factor $(L_1 \psi(x_k)) \otimes \Phi(x_k)^{+}$ converts that $\mathcal{Y}$-valued error into a rank-one operator in $\mathcal{L}(\mathcal{Z}_1, \mathcal{Z}_2)$: pull the error back through $\Phi(x_k)^{+}$ to land in $\mathcal{Z}_2$, and tensor it with the dual covector $L_1 \psi(x_k) \in \widecheck{\mathcal{Z}}_1$ so the resulting operator activates only along the direction $\psi(x_k)$.

</div>

The class of networks (1.17) already contains many degrees of freedom that can be additionally learned from data, besides the linear latent-space transformation $W \in \mathcal{L}(\mathcal{Z}_1, \mathcal{Z}_2)$: the nonlinear functions $c, \psi$ and the linear-operator-valued function $\Phi$ that adapts to the input. Using the general approach above, this architecture can be straightforwardly extended to the composition of multiple such transformations -- '**stacking layers**'.

### 1.5. Continuous Linear Associative Memories

In this section we study a linear *continuous-time* version of the class of networks from Section 1.4. The discrete sweep (1.19) over training patterns is replaced by the **flow of a linear ODE**, and the trained operator is no longer determined by an iterative correction but by the closed-form solution of the underlying dynamical system.

#### 1.5.1. Linear Time-Invariant Associative Memories

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Linear time-invariant dynamical system)</span></p>

We consider linear continuous-time *time-invariant* dynamical systems

$$
\dot{x}(t) = A\, x(t) + f(t), \qquad x(t_0) = x_0. \tag{1.20}
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">1.7 (Variation of constants formula)</span></p>

The solution of (1.20) is given by

$$
x(t) = e^{A(t - t_0)}\, x_0 + \int_{t_0}^{t} e^{A(t - \tau)}\, f(\tau)\, d\tau, \tag{1.21}
$$

where

$$
\exp_m(A) = e^A := \sum_{k=0}^{\infty} \frac{1}{k!}\, A^k \tag{1.22}
$$

is the **matrix exponential**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">1.8 (Time-variant systems, transition matrix)</span></p>

The solution formula (1.21) is an important special case of the more general *variation of constants formula*

$$
x(t) = \Phi(t, t_0)\, x_0 + \int_{t_0}^{t} \Phi(t, \tau)\, f(\tau)\, d\tau \tag{1.23a}
$$

that applies to time-*variant* systems

$$
\dot{x}(t) = A(t)\, x(t) + f(t), \qquad x(t_0) = x_0, \tag{1.23b}
$$

where $\Phi(t, t_0)$ is the **transition matrix** for the *homogeneous* time-variant system

$$
\dot{x}(t) = A(t)\, x(t), \qquad x(t_0) = x_0. \tag{1.24}
$$

The transition matrix expresses, in closed form, the solution

$$
x(t) = \Phi(t, t_0)\, x_0, \qquad t \geq t_0, \tag{1.25}
$$

of (1.24) that passes through $x_0$ at time $t_0$. It satisfies the *composition rule*

$$
\Phi(t, t_0) = \Phi(t, t_1)\, \Phi(t_1, t_0), \qquad t \geq t_1 \geq t_0. \tag{1.26}
$$

Since the computation of $\Phi$ is intractable in almost all cases of interest, we consider numerically the time-invariant case only. The time-variant case is tractable only in a discrete-time fashion, as outlined at the end of Section 1.4. Even in the time-invariant case, the computation of the transition matrix

$$
\Phi(t, t_0) = \exp_m\!\bigl( A(t - t_0) \bigr) \tag{1.27}
$$

is numerically involved for large dimensions $d$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">1.9 (Adjoint transformation, commutator)</span></p>

We define the mappings

$$
\mathrm{ad} : \mathbb{R}^{d \times d} \to \mathcal{L}(\mathbb{R}^{d \times d}, \mathbb{R}^{d \times d}), \qquad \mathrm{ad}_A = \mathrm{ad}(A), \qquad \mathrm{ad}_A^0 := \mathrm{id}_{\mathbb{R}^{d \times d}}, \tag{1.28a}
$$

$$
\mathrm{ad}_A(B) = \mathrm{ad}(A)\, B := [A, B], \qquad [A, B] := AB - BA \quad \textbf{(commutator).} \tag{1.28b}
$$

The actual role of $\mathrm{ad}(\cdot)$ (adjoint representation of a Lie algebra) will be introduced later.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">1.10 (Matrix exponential: properties)</span></p>

The matrix exponential (1.22) has the following properties. For any $A \in \mathbb{C}^{d \times d}$ (and in particular for any $A \in \mathbb{R}^{d \times d}$):

$$
e^{0} = I, \tag{1.29a}
$$

$$
(e^A)^{\ast} = e^{A^{\ast}}, \qquad A^{\ast} := \overline{A}^{\top} = \overline{A^{\top}}, \tag{1.29b}
$$

$$
(e^A)^{-1} = e^{A^{-1}}, \qquad \text{if } A \in \mathrm{GL}(d; \mathbb{C}), \tag{1.29c}
$$

$$
e^{(\alpha + \beta) A} = e^{\alpha A}\, e^{\beta A}, \qquad \forall\, \alpha, \beta \in \mathbb{C}, \tag{1.29d}
$$

$$
AB = BA \quad \Longrightarrow \quad e^{A + B} = e^{A}\, e^{B} = e^{B}\, e^{A}, \tag{1.29e}
$$

$$
e^{B A B^{-1}} = B\, e^{A}\, B^{-1}, \qquad \text{if } B \in \mathrm{GL}(d; \mathbb{C}), \tag{1.29f}
$$

$$
\frac{d}{dt}\, e^{tA} = A\, e^{tA} = e^{tA}\, A, \tag{1.29g}
$$

$$
e^{A}\, e^{B} = e^{C}, \qquad \text{where} \tag{1.29h}
$$

$$
C = A + B + \frac{1}{2}[A, B] + \frac{1}{12}\bigl( [[A, B], A] + [[B, A], B] \bigr) + \cdots \qquad \textbf{(BCH formula)}\ \text{(Baker, Campbell, Hausdorff).} \tag{1.29i}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">1.11 (Two matrix-valued matrix functions)</span></p>

For any $A \in \mathbb{R}^{d \times d}$, one defines

$$
\varphi(A) := \sum_{k=0}^{\infty} \frac{1}{(k+1)!}\, A^{k}, \tag{1.30a}
$$

which is the matrix-valued matrix function generated by the analytical function

$$
\varphi(z) = \frac{e^{z} - 1}{z} = \int_{0}^{1} e^{(1 - t) z}\, dt. \tag{1.30b}
$$

The series defining $\varphi$ converges for any matrix argument $A$ and in turn defines

$$
\psi : \mathbb{R}^{d \times d} \times \mathbb{R}^{d \times d} \to \mathbb{R}^{d \times d}, \qquad \psi_{A}(B) := \varphi(\mathrm{ad}_{A})\, B = \sum_{k=0}^{\infty} \frac{1}{(k+1)!}\, \mathrm{ad}_{A}^{k}\, B \tag{1.31a}
$$

$$
\phantom{\psi_{A}(B)} = B + \frac{1}{2}[A, B] + \frac{1}{6}\, [A, [A, B]] + \cdots \tag{1.31b}
$$

This series converges for any $A$.

</div>

The significance of the function (1.30) is due to the special case of a *constant* vector in (1.20),

$$
f(t) = f, \qquad \forall\, t, \tag{1.32}
$$

in which case the variation of constants formula (1.21) takes the form

$$
x(t) = t\, \varphi(t A)\, (f + A x_0) + x_0. \tag{1.33}
$$

The matrix-valued function $\varphi(\cdot)$ also defines the matrix-valued function (1.31) which in turn yields an expression for the matrix exponential (Proposition 1.13).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">1.12 (Formula for computing $\varphi(A)$)</span></p>

A simple (yet not efficient) way for evaluating the matrix function $\varphi$ is to embed the argument as a block of a larger matrix and to compute the matrix exponential:

$$
B = \begin{pmatrix} A & I \\ 0 & 0 \end{pmatrix} \qquad \Longrightarrow \qquad \exp_{m}(B) = \begin{pmatrix} e^{A} & \varphi(A) \\ 0 & I \end{pmatrix}. \tag{1.34}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">1.13 (Differential of the matrix exponential)</span></p>

One has

$$
d\!\exp_{m}(A)\, B = \psi_{A}(B)\, \exp_{m}(A). \tag{1.35}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">1.14 (Formula for computing $d\!\exp_{m}(A) B$)</span></p>

Similar to the simple method for evaluating the matrix function $\varphi$ (Remark 1.12), the differential of the matrix exponential can be computed as the matrix exponential of a larger matrix:

$$
\exp_{m}\!\begin{pmatrix} A & B \\ 0 & A \end{pmatrix} = \begin{pmatrix} e^{A} & d\!\exp_{m}(A)\, B \\ 0 & e^{A} \end{pmatrix}. \tag{1.36}
$$

</div>

The following result is relevant for computing the inverse $(d\!\exp_{m}(A))^{-1}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">1.15 (Inverse of $\psi_{A}$)</span></p>

Suppose the eigenvalues $\lambda_j = \lambda_j(\mathrm{ad}_A)$ of the linear operator $\mathrm{ad}_A$ do not take values in the set $\lbrace i 2\pi z : z \in \mathbb{Z} \setminus \lbrace 0 \rbrace \rbrace$. If $\|A\|_2 < \pi$, then one has

$$
\psi_{A}^{-1}(B) = \sum_{k=0}^{\infty} \frac{B_{k}}{k!}\, \mathrm{ad}_{A}^{k}\, B \tag{1.37a}
$$

$$
\phantom{\psi_{A}^{-1}(B)} = B - \frac{1}{2}[A, B] + \frac{1}{12}\, [A, [A, B]] \mp \cdots \tag{1.37b}
$$

where the **Bernoulli numbers** $B_{k}$ are defined by

$$
\frac{z}{e^{z} - 1} = 1 - \frac{z}{2} + \frac{z^{2}}{12} - \frac{z^{4}}{720} + \cdots = \sum_{k=0}^{\infty} \frac{B_{k}}{k!}\, z^{k}. \tag{1.37c}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">1.16 (Adjoint equation)</span></p>

Given a linear *homogeneous* dynamical system in $x \in \mathcal{X}$, another linear homogeneous dynamical system in $p \in \widecheck{\mathcal{X}}$ is said to be the *adjoint equation* if, for any initial point, the duality product

$$
\langle p(t),\, x(t) \rangle_{\widecheck{\mathcal{X}}, \mathcal{X}} \tag{1.38}
$$

is **constant**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">1.17 (Adjoint equation)</span></p>

Let $\Phi(T, t_0)$ be the transition matrix for the homogeneous time-variant linear system (1.24) with $t \in [t_0, T]$. Then the solution to the *adjoint equation*

$$
\dot{p}(t) = -\widetilde{A}(t)\, p(t), \qquad p_{T} := p(T), \tag{1.39a}
$$

to be integrated *backwards* in time, is given by

$$
p(t) = \widetilde{\Phi}(T, t)\, p_{T}, \qquad t \in [t_0, T]. \tag{1.39b}
$$

</div>

Recall also in this context that for time-*invariant* systems, the transition matrix is given by (1.27).

#### 1.5.2. Continuous-Time Linear Associative Memory

Recall once more the set-up of Section A.1.1, the scalar product (A.25) and the induced norm (A.27) on the space $\mathcal{L}(\mathcal{X}, \mathcal{Y})$.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Time-variant linear dynamical system with latent control)</span></p>

We consider the time-variant linear dynamical system

$$
\dot{x}(t) = A(t)\, x(t) + B(t)\, W(t)\, y(t), \qquad x(0) = x_{0}, \qquad t \in [0, T], \tag{1.40a}
$$

with input and output spaces $\mathcal{X}, \mathcal{Y}$, a latent space $\mathcal{Z}$ and the operators

$$
A(t) \in \mathcal{L}(\mathcal{X}, \mathcal{X}), \qquad B(t) \in \mathcal{L}(\mathcal{Z}, \mathcal{X}), \qquad W(t) \in \mathcal{L}(\mathcal{Y}, \mathcal{Z}). \tag{1.40b}
$$

</div>

Note that the application of any numerical integration scheme yields a network with corresponding parameters indexed by the discrete points of time. By (1.23), the solution is given by

$$
x(T) = \Phi(T, 0)\, x_{0} + \int_{0}^{T} \Phi(T, t)\, B(t)\, W(t)\, y(t)\, dt. \tag{1.41}
$$

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Learning problem -- continuous-time)</span></p>

Given a target state $x_{T}^{\ast}$, we consider the problem to determine the function $t \mapsto W(t)$ such that

$$
x(T) = x_{T}^{\ast}. \tag{1.42}
$$

Since there are many trajectories $t \mapsto x(t)$ which achieve this, we adopt the strategy from the previous sections and look for a minimal-norm solution in the space

$$
L^{2}\bigl( 0, T;\, \mathcal{L}(\mathcal{Y}, \mathcal{Z}) \bigr), \tag{1.43}
$$

equipped with the scalar product

$$
\langle U, V \rangle_{[0, T]} := \int_{0}^{T} (\widetilde{\ell} \otimes m)\bigl( U(t),\, V(t) \bigr)\, dt \tag{1.44a}
$$

and the corresponding duality mapping

$$
\mathcal{J} := U(\cdot) \;\mapsto\; (L^{-1} \otimes M)\bigl( U(\cdot) \bigr). \tag{1.44b}
$$

</div>

By (1.41), any operator-valued function $U(\cdot)$ achieving (1.42) has to satisfy the equation

$$
\int_{0}^{T} \Phi(T, t)\, B(t)\, U(t)\, y(t)\, dt = x_{T}^{\ast} - \Phi(T, 0)\, x_{0}. \tag{1.45}
$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">1.18 (Solution to the learning problem)</span></p>

Suppose that the operator

$$
\int_{0}^{T} \lambda\bigl( y(t) \bigr)^{2}\, \Phi(T, t)\, B(t)\, M^{-1}\, \widetilde{B}(t)\, \widetilde{\Phi}(T, t)\, dt \tag{1.46}
$$

is invertible and that $p_{T} \in \mathcal{X}$ solves the equation

$$
\left( \int_{0}^{T} \lambda\bigl( y(t) \bigr)^{2}\, \Phi(T, t)\, B(t)\, M^{-1}\, \widetilde{B}(t)\, \widetilde{\Phi}(T, t)\, dt \right) p_{T} = x_{T}^{\ast} - \Phi(T, 0)\, x_{0}. \tag{1.47}
$$

Then the linear operator-valued function $t \mapsto W(t)$ of minimal norm $\|W\|_{[0, T]}$ mapping $x_0$ to $x_T^{\ast}$ via (1.41) is given by

$$
W(t) = \bigl( L\, y(t) \bigr) \otimes \bigl( M^{-1}\, \widetilde{B}(t)\, p(t) \bigr), \tag{1.48}
$$

where $p(\cdot)$ solves the adjoint problem

$$
\dot{p}(t) = -\widetilde{A}(t)\, p(t), \qquad p(T) = p_{T}. \tag{1.49}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">1.19 (Time-invariant systems)</span></p>

For the simpler case of a time-*invariant* system matrix $A$,

$$
\dot{x}(t) = A\, x(t) + B(t)\, W(t)\, y(t), \tag{1.50}
$$

the result of Theorem 1.18 directly applies without any change, except for the transition matrix corresponding to $A$ which then takes the simpler form (1.27).

On the other hand, even if both $B(t) = B$ and $y(t) = y$ were constant too, the function $W(\cdot)$ would not be constant due to the solution $p(\cdot)$ of the adjoint equation (1.48), which is the **continuous-time analogue of gradient backpropagation**.

</div>

## 2. Geometry on Smooth Manifolds

### 2.1. Geometry of a Manifold

We compare a Euclidean space, represented by $\mathbb{R}^N$, with a smooth manifold $\mathcal{M}$. Like any Euclidean space, $\mathbb{R}^N$ is equipped *per point* with three pieces of geometric structure that we shall now isolate, so that — in the rest of this chapter — we can generalize them, one at a time, to the manifold setting.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(The three Euclidean structures on $\mathbb{R}^N$)</span></p>

For every point $p \in \mathbb{R}^N$:

- **with identification** of the tangent space at $p$ with the ambient vector space itself,

$$
T_{p}\mathbb{R}^{N} \;\cong\; \mathbb{R}^{N}, \tag{2.1}
$$

- **with the standard inner product** on the tangent space,

$$
T_{p}\mathbb{R}^{N} \ni v, w \;\mapsto\; v^{\top} w \;\in\; \mathbb{R}, \qquad \forall\, v, w \in T_{p}\mathbb{R}^{N}, \tag{2.2}
$$

- **and with length** measured along the straight-line path

$$
\gamma_{a, b}(t) \;=\; a + t\,(b - a), \qquad t \in [0, 1], \qquad a, b \in \mathbb{R}^{N}, \tag{2.3}
$$

connecting any two points $a, b \in \mathbb{R}^{N}$.

</div>

Our goal is to generalize this concrete scenario by incorporating more generic geometry, in a gradual manner. To this end — and to organise the plan of this chapter — it will be helpful to consider the differences (and similarities) between geometric quantities in $\mathbb{R}^N$ and on $\mathcal{M}$, gradually.

To keep the discussion concrete, two running examples will accompany us throughout the chapter: one *flat* and Abelian, the other *curved* and non-Abelian.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Two running Lie groups)</span></p>

We consider the two Lie groups

- $\mathbb{R}^{N}$ with the *additive group* structure, and
- $\mathrm{GL}^{+}(N)$ of *invertible matrices of order $N$ with positive determinant*.

Like $\mathbb{R}^{N}$, the matrix Lie group $\mathrm{GL}^{+}(N)$ is connected and forms its own *embedded* manifold inside $\mathbb{R}^{N \times N}$, but unlike $\mathbb{R}^{N}$, the group $\mathrm{GL}^{+}(N)$ is **non-Abelian**.

</div>

We now make precise what we mean by *(Riemannian) metric* on a smooth manifold. The basic idea is that, since the tangent spaces $T_{p}\mathcal{M}$ at different points $p \in \mathcal{M}$ are *a priori* unrelated vector spaces, we must equip each one of them with its own inner product — and do so *smoothly* in $p$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.1 (Riemannian metric, Riemannian manifold)</span></p>

A **(Riemannian) metric** on a smooth manifold $\mathcal{M}$ assigns to each point $p \in \mathcal{M}$ a (positive definite) symmetric bilinear form

$$
g_{p} : T_{p}\mathcal{M} \times T_{p}\mathcal{M} \to \mathbb{R}, \qquad g_{p}(v, w) := \langle v, w\rangle_{T_{p}\mathcal{M}}, \qquad \forall\, v, w \in T_{p}\mathcal{M}, \tag{2.4}
$$

equivalently identified with a homomorphism $g_{p} \in \mathcal{L}\bigl(T_{p}\mathcal{M},\, T_{p}^{\ast}\mathcal{M}\bigr)$. Therefore, every (Riemannian) metric induces a norm

$$
\|v\|_{T_{p}\mathcal{M}} := \sqrt{g_{p}(v, v)}, \qquad \forall\, v \in T_{p}\mathcal{M}, \tag{2.5}
$$

such that the mapping

$$
G : \mathcal{M} \ni p \;\mapsto\; g_{p} \;\in\; \mathcal{L}\bigl(T\mathcal{M},\, T^{\ast}\mathcal{M}\bigr) \tag{2.6}
$$

is a homomorphism of vector bundles. The pair $(\mathcal{M}, g)$ is then called a **Riemannian manifold**, and $g$ is its **(Riemannian) metric (tensor)**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reading the three layers of Definition 2.1)</span></p>

Definition 2.1 packs three layers of structure into a single object — it is worth peeling them apart:

- **Pointwise layer.** At each fixed $p \in \mathcal{M}$, $g_{p}$ is just a scalar product on the finite-dimensional vector space $T_{p}\mathcal{M}$, in exactly the sense of Appendix A.1. So locally, *every* tangent space looks like the Euclidean set-up we have been working with so far.
- **Algebraic layer.** Via the duality mapping construction (cf. (A.2)), the bilinear form $g_{p}$ is the same datum as the linear isomorphism $T_{p}\mathcal{M} \to T_{p}^{\ast}\mathcal{M}$ — this is precisely what the codomain $\mathcal{L}(T_{p}\mathcal{M}, T_{p}^{\ast}\mathcal{M})$ in (2.6) is recording.
- **Global / bundle layer.** Letting $p$ vary, $G$ in (2.6) is a section of the bundle of bilinear forms over $\mathcal{M}$. The qualifier *vector-bundle homomorphism* enforces that $g_{p}$ varies *smoothly* with $p$ — without this, one could choose a wildly discontinuous metric at every point and the resulting geometry would be useless.

</div>

We now apply Definition 2.1 to our first running example, the Euclidean space $\mathbb{R}^N$ regarded as a Riemannian manifold.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\mathbb{R}^{N}$ as Riemannian manifold)</span></p>

With respect to the geometry of $\mathbb{R}^{N}$ as a Riemannian manifold, the metric is

$$
g_{p}(v, w) \;:=\; \langle v, w\rangle_{T_{p}\mathbb{R}^{N}} \;=\; v^{\top} w, \qquad \forall\, v, w \in T_{p}\mathbb{R}^{N} \cong \mathbb{R}^{N}, \tag{2.7}
$$

i.e. the standard inner product (2.2) is the *same* at every point $p$ under the canonical identification (2.1). Informally, $G \in \mathcal{L}(T\mathbb{R}^{N}, T^{\ast}\mathbb{R}^{N})$ is constant along $\mathbb{R}^{N}$ — the curvature carried by $G$ vanishes, which is what makes $\mathbb{R}^{N}$ *flat* among Riemannian manifolds.

</div>

Having fixed the *pointwise* geometric data via $g$, we now bring in the second protagonist: **Lie groups acting on $\mathcal{M}$**. The reason is that, in the running examples, the manifolds $\mathbb{R}^{N}$ and $\mathrm{GL}^{+}(N)$ each act on themselves (by translation and by left-multiplication respectively), and combining the two will yield the rigid-motion group of $\mathbb{R}^{N}$. We single out the conditions on a smooth action $\Phi : G \times \mathcal{M} \to \mathcal{M}$, $\Phi(g, x) =: g \cdot x$, which together guarantee that the quotient $\mathcal{M} / G$ is again a manifold.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Free, transitive, and proper actions)</span></p>

Let $G$ be a Lie group acting smoothly on a smooth manifold $\mathcal{M}$. The action is called:

- **free**, if the *stabilizer (isotropy) group*

$$
G_{x} := \lbrace g \in G : g \cdot x = x \rbrace = \lbrace e \rbrace, \qquad \forall\, x \in \mathcal{M}, \tag{2.10}
$$

is *trivial* at every point, i.e. $G$ acts without fixed points. Informally: every element $g \in G \setminus \lbrace e \rbrace$ moves every point of $\mathcal{M}$.

- **transitive**, if any two points of $\mathcal{M}$ can be connected by some $g \in G$, equivalently, if $\mathcal{M}$ consists of a single *orbit*

$$
G \cdot x := \lbrace g \cdot x : g \in G \rbrace \subset \mathcal{M} \qquad (\text{orbit of } x \text{ under } G), \tag{2.11}
$$

namely $G \cdot x = \mathcal{M}$ for some (equivalently every) $x \in \mathcal{M}$.

- **proper**, if each orbit $G \cdot x$ is a closed subset of $\mathcal{M}$ and each isotropy group $G_{x}$ is compact. Equivalently, the map $G \times \mathcal{M} \ni (g, x) \mapsto (g \cdot x, x) \in \mathcal{M} \times \mathcal{M}$ is a proper map of topological spaces.

</div>

One associates with the action $\Phi$ of $G$ on $\mathcal{M}$ the

$$
\mathcal{M} / G, \qquad (\text{orbit space}) \tag{2.12}
$$

that is, the set of orbits $\lbrace G \cdot x : x \in \mathcal{M} \rbrace$ equipped with the quotient topology. In general, $\mathcal{M} / G$ may be a poorly behaved topological space; the next theorem identifies the right hypotheses to guarantee that it is again a smooth manifold.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.1 (Quotient Manifold Theorem)</span></p>

Suppose a Lie group $G$ acts smoothly, freely and properly on a smooth manifold $\mathcal{M}$. Then the orbit space $\mathcal{M} / G$ is a topological manifold of dimension $\dim \mathcal{M} - \dim G$ and has a unique smooth structure with respect to which the

$$
\pi : \mathcal{M} \to \mathcal{M} / G \qquad (\text{quotient map, canonical projection}) \tag{2.13}
$$

is a smooth submersion.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the three hypotheses of Theorem 2.1?)</span></p>

The hypotheses on the action $\Phi$ of $G$ on $\mathcal{M}$ play three complementary roles:

- **smooth** makes the action compatible with the differentiable structure on both factors, so that any quotient structure inherits *some* differential geometry rather than only a topology;
- **free** prevents non-trivial isotropies, so that distinct orbits do not collapse at fixed points — this is what makes the orbit space *Hausdorff* and ultimately a manifold;
- **proper** is the global control: it ensures orbits do not "accumulate" in $\mathcal{M}$ (each orbit is closed) and that isotropies do not blow up to non-compact subgroups, so that $\mathcal{M} / G$ remains *second countable* and locally Euclidean.

The dimension formula $\dim(\mathcal{M} / G) = \dim \mathcal{M} - \dim G$ is the manifold analogue of the orbit-stabilizer relation in finite group theory.

</div>

We next define an operation which constructs a new Lie group from two given Lie groups. Recall that a Lie group is both a group and a smooth manifold; the construction we now introduce is what turns the running pair $\bigl(\mathbb{R}^{N}, \mathrm{GL}^{+}(N)\bigr)$ — together with the natural matrix-on-vector action — into a single rigid-motion group.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.2 (Semidirect product)</span></p>

Let $H$ and $N$ be Lie groups and let $\theta : H \times N \to N$ be a smooth left action of $H$ on $N$ by automorphisms — that is, for each fixed $h \in H$, the map $\theta_{h} : N \to N$ is a Lie-group automorphism of $N$. Then the **semidirect product** of $H$ and $N$ is the Lie group

$$
N \rtimes_{\theta} H, \qquad (\text{semidirect product}) \tag{2.14}
$$

defined as the smooth manifold $N \times H$ endowed with the product (group operation)

$$
(n, h) \cdot (n', h') := \bigl(n \cdot \theta_{h}(n'),\; h \cdot h'\bigr). \tag{2.15}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reading the semidirect product formula)</span></p>

The formula (2.15) splits cleanly into two parts:

- **The $H$-coordinate** multiplies independently of $N$: $(h, h') \mapsto h \cdot h'$. So $H$ embeds as a closed subgroup of $N \rtimes_{\theta} H$ via $h \mapsto (e_{N}, h)$.
- **The $N$-coordinate** would have been $n \cdot n'$ in the direct product, but in the *semi*-direct product it is *twisted* by $\theta_{h}$ acting on $n'$. Setting $\theta \equiv \mathrm{id}_{N}$ recovers the direct product $N \times H$ as a special case — the prefix "semi" measures precisely the failure of $H$ to commute with $N$.

This twisting is exactly what we need to encode that, when a rotation $h$ is performed *after* a translation $n'$, the translation looks rotated.[^auto]

[^auto]: A map $\varphi$ between two groups $H, K$ is called an **isomorphism** (resp. **automorphism** if $H = K$) if it is a bijective group homomorphism.

</div>

We apply this operation to the two running Lie groups and obtain a *single* Lie group $G$ that acts on $\mathcal{M} = \mathbb{R}^{N}$ by rigid motions. This will serve as a simple yet representative scenario which exemplifies generic geometric objects on $\mathcal{M}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Special Euclidean group $E^{+}(N)$)</span></p>

We set $H := \mathrm{SO}(N)$ and $N := \mathbb{R}^{N}$. The group $H$ acts on $N$ by automorphisms in the obvious way (a rotation matrix times a vector). Applying Definition 2.2 yields the

$$
E^{+}(N) := \mathbb{R}^{N} \rtimes_{\theta} \mathrm{SO}(N), \qquad (\text{special Euclidean group}) \tag{2.16}
$$

which represents all *rigid motions* of $\mathbb{R}^{N}$ (translations composed with proper rotations). The multiplication is given by

$$
(b, A) \cdot (b', A') := \bigl(b + A\, b',\; A\, A'\bigr), \tag{2.17}
$$

and the action of $G = E^{+}(N)$ on $\mathcal{M} = \mathbb{R}^{N}$ is given as

$$
\Phi_{(b, A)}(p) := A\, p + b, \qquad b \in \mathbb{R}^{N},\; A \in \mathrm{SO}(N),\; p \in \mathbb{R}^{N}. \tag{2.18}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reading $E^{+}(N)$ against the three conditions)</span></p>

Why is $E^{+}(N)$ a useful test case? Because, against the three properties of Definition above:

- the action (2.18) of $E^{+}(N)$ on $\mathbb{R}^{N}$ is **transitive** — any point can be sent to any other by a translation alone, so $G \cdot p = \mathbb{R}^{N}$;
- it is **not free**, because for any non-zero rotation $A \in \mathrm{SO}(N)$ with $A p + b = p$ (e.g. fixing the axis of $A$) the stabilizer $G_{p}$ is non-trivial;
- but it is **proper**, since $\mathrm{SO}(N)$ is compact and translations are proper.

Hence $\mathbb{R}^{N} \cong E^{+}(N) / \mathrm{SO}(N)$ — exhibiting $\mathbb{R}^{N}$ as a *homogeneous space* under the rigid-motion group, the simplest setting in which the Quotient Manifold Theorem 2.1 applies to the stabilizer of a point.

</div>

We now repackage the special Euclidean group — denoted from here on by $\mathrm{SE}(N) \equiv E^{+}(N)$ — as a closed matrix Lie group sitting inside $\mathrm{GL}^{+}(N + 1)$. The trick is to identify a point $x \in \mathbb{R}^{N}$ with its *homogeneous coordinates*

$$
x \;\mapsto\; \binom{x}{1} \;\in\; \mathbb{R}^{N} \times \lbrace 1 \rbrace \;\subset\; \mathbb{R}^{N + 1},
$$

which trades the *affine* action $p \mapsto A p + b$ on $\mathbb{R}^{N}$ for an honest *linear* representation on the hyperplane $\lbrace x_{N+1} = 1 \rbrace$ — i.e. a chart of projective space $\mathbb{RP}^{N}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Matrix representation of $\mathrm{SE}(N)$)</span></p>

Under the homogeneous-coordinate identification above, one has the linear representation

$$
\mathrm{SE}(N) \;\cong\; \left\lbrace
\begin{pmatrix} A & b \\ 0 & 1 \end{pmatrix}
\;:\; A \in \mathrm{SO}(N),\; b \in \mathbb{R}^{N}
\right\rbrace, \tag{2.19}
$$

with the multiplication law (2.17) reproduced by ordinary matrix multiplication of these block matrices.

</div>

We specialize to $N = 3$, where $\mathrm{SE}(3) = G$ is the rigid-motion group of physical space. Its Lie algebra $\mathfrak{se}(3)$ — the tangent space at the identity, which we shall meet formally in the next section — has the analogous block description

$$
\mathfrak{se}(3) \;=\; \left\lbrace
\begin{pmatrix} \widehat{\omega} & v \\ 0 & 0 \end{pmatrix}
\;:\; \omega, v \in \mathbb{R}^{3}
\right\rbrace, \tag{2.20}
$$

where the **hat map** $\widehat{(\cdot)} : \mathbb{R}^{3} \to \mathfrak{so}(3)$ sends a vector to a skew-symmetric matrix:

$$
\widehat{\omega} \;=\;
\begin{pmatrix}
0 & -\omega_{3} & \omega_{2} \\
\omega_{3} & 0 & -\omega_{1} \\
-\omega_{2} & \omega_{1} & 0
\end{pmatrix}, \qquad \omega \in \mathbb{R}^{3}. \tag{2.21}
$$

The hat map is characterized intrinsically by the identity

$$
\widehat{\omega}\, v \;=\; \omega \times v, \qquad \forall\, v \in \mathbb{R}^{3}, \quad (\text{cross product in } \mathbb{R}^{3}) \tag{2.22}
$$

so that $\mathfrak{se}(3) \cong \mathbb{R}^{6}$ canonically via $(\omega, v) \leftrightarrow \bigl(\begin{smallmatrix}\widehat{\omega} & v \\ 0 & 0\end{smallmatrix}\bigr)$ — three parameters $\omega$ for the rotational part, three for the translational part $v$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Rodrigues' formula)</span></p>

The rotation of $\mathbb{R}^{3}$ by an angle $\alpha \in [0, 2\pi)$ around a unit axis $n \in S^{2} \subset \mathbb{R}^{3}$ is given by

$$
\begin{aligned}
\exp(\alpha\, \widehat{n}) &\;=\; \cos(\alpha)\, I_{3} \;+\; \sin(\alpha)\, \widehat{n} \;+\; \bigl(1 - \cos(\alpha)\bigr)\, n\, n^{\top} &&\text{(2.23a)} \\
&\;=\; I_{3} \;+\; \sin(\alpha)\, \widehat{n} \;+\; \bigl(1 - \cos(\alpha)\bigr)\, \widehat{n}^{\,2}, \qquad (\text{Rodrigues' formula}) &&\text{(2.23b)}
\end{aligned}
$$

where the unit axis and angle are recovered from a general $\omega \in \mathbb{R}^{3}$ via

$$
n \;=\; \omega \,/\, \lVert \omega \rVert \;\in\; S^{2}, \qquad \alpha \;=\; \lVert \omega \rVert. \tag{2.24}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why two forms of (2.23)?)</span></p>

Forms (2.23a) and (2.23b) are equivalent via the identity $n n^{\top} = I_{3} + \widehat{n}^{\,2}$ valid for unit $n$ (a direct computation using $\widehat{n}\,v = n \times v$ and the BAC-CAB rule). The two presentations emphasize different geometries:

- **(2.23a)** decomposes a rotation into its action on the rotation axis (the $n n^{\top}$ projector fixes vectors along $n$) and on the orthogonal plane (rotated by $\cos\alpha\, I + \sin\alpha\, \widehat{n}$).
- **(2.23b)** is the Taylor-series form obtained by exponentiating $\alpha\,\widehat{n}$ and using $\widehat{n}^{\,3} = -\widehat{n}$ to collapse the series into three terms. This is the form that generalises cleanly to other matrix Lie groups.

</div>

With $\mathrm{SE}(N)$ realised concretely, we now have all the language we need to encode the picture *"$\mathbb{R}^{N}$ together with the action by rigid motions"* in a single intrinsic definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.3 (Homogeneous space, $G$-space)</span></p>

A smooth manifold $\mathcal{M}$ endowed with a *transitive* smooth action $\Phi$ by a Lie group $G$ is called a **homogeneous space**, or a $G$**-space**. In particular, $\mathbb{R}^{N}$ is a homogeneous $\mathrm{SE}(N)$-space.

</div>

This is a specific instance of the following general situation, characterized by the next two theorems. We first introduce some further notation around **cosets**, which will let us describe $\mathcal{M}$ as a quotient $G / H$ in terms of an isotropy subgroup.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Left and right cosets)</span></p>

Let $H \subseteq G$ be a Lie subgroup. Fix $g \in G$ and consider the

$$
g H \;:=\; \lbrace g \cdot h : h \in H \rbrace \;\subseteq\; G, \qquad (\text{left coset of } H \text{ in } G) \tag{2.25}
$$

$$
H g \;:=\; \lbrace h \cdot g : h \in H \rbrace \;\subseteq\; G, \qquad (\text{right coset of } H \text{ in } G) \tag{2.26}
$$

Two left (resp. right) cosets are either disjoint or coincide. In particular, the cosets *partition* $G$.

</div>

We write

$$
G / H \;:=\; \lbrace g H : g \in G \rbrace \tag{2.27}
$$

for the **left coset space**, i.e. the set of left cosets of $H$ in $G$, equipped with the quotient topology induced by the canonical projection $g \mapsto g H$. When $H$ is *closed* in $G$, the next theorem upgrades this set-theoretic quotient to a bona fide smooth manifold.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.2 (Homogeneous Space Construction Theorem)</span></p>

Let $H \subseteq G$ be a *closed* Lie subgroup of $G$. The left coset space $G / H$ is a topological manifold of dimension $\dim G - \dim H$, and has a unique smooth structure such that the

$$
\pi : G \to G / H \qquad (\text{canonical projection, quotient map}) \tag{2.28}
$$

is a smooth submersion. The left action of $G$ on $G / H$ given by

$$
g \cdot g' H \;:=\; (g g') H, \qquad g, g' \in G \qquad (\text{natural action}) \tag{2.29}
$$

turns $G / H$ into a homogeneous $G$-space.

</div>

The following theorem provides the converse: whenever one has a smooth manifold $\mathcal{M}$ with a transitive $G$-action, then $\mathcal{M}$ — regarded as a homogeneous $G$-space — can be **equivariantly** identified with a coset space of the form (2.28), namely $G / G_{x}$ where $G_{x}$ is the isotropy group of any chosen base point $x$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">2.3 (Homogeneous Space Characterization Theorem)</span></p>

Let $G$ be a Lie group, let $\mathcal{M}$ be a homogeneous $G$-space, and let $x \in \mathcal{M}$ be any point. The isotropy group $G_{x} \subseteq G$ from (2.10) is a closed subgroup of $G$ and the map

$$
F : G / G_{x} \to \mathcal{M}, \qquad F(g G_{x}) := g \cdot x, \qquad g \in G, \tag{2.30a}
$$

is an *equivariant* diffeomorphism, i.e.

$$
F\bigl( g \cdot (g_{0} G_{x}) \bigr) \;=\; g \cdot F(g_{0} G_{x}), \qquad g, g_{0} \in G. \tag{2.30b}
$$

</div>

In view of the last statement, we recall the following general definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">2.4 (Equivariant maps, intertwining maps)</span></p>

Let $\mathcal{M}, \mathcal{N}$ be smooth manifolds, each carrying a smooth action $\Phi$ of a Lie group $G$. A map $F : \mathcal{M} \to \mathcal{N}$ is said to be **equivariant** with respect to the given $G$-actions if

$$
F\bigl( \Phi_{g}(x) \bigr) \;=\; \Phi_{g}\bigl( F(x) \bigr), \qquad g \in G,\; x \in \mathcal{M}, \qquad (\text{for left actions}) \tag{2.31a}
$$

$$
F\bigl( \Phi_{g}(x) \bigr) \;=\; \Phi_{g^{-1}}\bigl( F(x) \bigr), \qquad g \in G,\; x \in \mathcal{M}, \qquad (\text{for right actions}) \tag{2.31b}
$$

Either equation says that the diagram

$$
\begin{array}{ccc}
\mathcal{M} & \xrightarrow{\;\;\Phi_{g}\;\;} & \mathcal{M} \\
\phantom{F}\Big\downarrow F & & \Big\downarrow F\phantom{F} \\
\mathcal{N} & \xrightarrow{\;\;\Phi_{g}\;\;} & \mathcal{N}
\end{array}
$$

commutes. One also says that $F$ **intertwines** $\Phi$ on $\mathcal{M}$ and $\mathcal{N}$.

</div>

In the situation of Theorem 2.3, the roles of $\mathcal{M}, \mathcal{N}$ in Definition 2.4 are played by $G / G_{x}$ and $\mathcal{M}$ respectively, and the equivariance condition (2.30b) is exactly (2.31a) — the natural action (2.29) on the source corresponds to the original action $\Phi$ on the target.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How Theorems 2.2 and 2.3 fit together)</span></p>

The two theorems are converses of each other and together yield a *bijection* between two perspectives on a homogeneous space:

- **Theorem 2.2 (construction):** start with the data of a closed subgroup $H \subseteq G$, and *produce* a homogeneous $G$-space $G / H$.
- **Theorem 2.3 (characterization):** start with an *abstract* homogeneous $G$-space $\mathcal{M}$, pick a base point $x \in \mathcal{M}$, and *recover* the closed subgroup $G_{x} \subseteq G$ such that $\mathcal{M} \cong G / G_{x}$ equivariantly.

The choice of base point $x$ in Theorem 2.3 is harmless: two different choices give *conjugate* isotropy subgroups (since $G_{g \cdot x} = g G_{x} g^{-1}$), so the resulting coset space is the same up to a canonical diffeomorphism.

</div>

We return to the running example (2.16). For $\mathcal{M} = \mathbb{R}^{N}$ and $G = \mathrm{SE}(N)$, we have to identify the isotropy $G_{x}$ in order to apply Theorem 2.3. Choosing $x = 0$ as the base point of $\mathbb{R}^{N}$, the elements $(b, A) \in \mathrm{SE}(N)$ fixing the origin are exactly those with $b = 0$, so

$$
G_{0} \;=\; \mathrm{SO}(N) \;\subset\; \mathrm{SE}(N). \tag{2.32}
$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\mathbb{R}^{N} \cong \mathrm{SE}(N)/\mathrm{SO}(N)$ as a homogeneous $G$-space)</span></p>

We unfold the map $F$ of Theorem 2.3 in this case. With $x_{0} := 0 \in \mathbb{R}^{N} = \mathcal{M}$, $g_{0} := (b_{0}, A_{0}) \in \mathrm{SE}(N)$ and $h := (0, A') \in \mathrm{SO}(N) = G_{0}$, the map (2.30a) becomes

$$
F\bigl( g_{0}\, G_{0} \bigr) \;=\; \Phi_{g_{0}}(x_{0}) \;=\; A_{0} \cdot 0 + b_{0} \;=\; b_{0}, \tag{2.33a}
$$

so each coset $(b_{0}, A_{0})\, \mathrm{SO}(N)$ is uniquely labelled by its translation part $b_{0} \in \mathbb{R}^{N}$ — exactly what Theorem 2.3 predicts. Equivariance (2.30b) reduces here to the direct computation

$$
F\bigl( g \cdot g_{0} G_{0} \bigr) \;=\; \Phi_{g}\bigl( F(g_{0} G_{0}) \bigr) \;=\; \Phi_{(b, A)}(b_{0}) \;=\; b + A\, b_{0}, \qquad g = (b, A) \in \mathrm{SE}(N), \tag{2.33b}
$$

which is consistent both with the rigid-motion action (2.18) on $\mathbb{R}^{N}$ and with the group law (2.17) on $\mathrm{SE}(N)$.

</div>
