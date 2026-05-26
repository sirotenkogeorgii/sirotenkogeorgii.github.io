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

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_tensor_scalar_product.png' | relative_url }}" alt="A single R² panel with four arrows from the origin: two blue arrows Ue₁ and Ue₂ representing U applied to the orthonormal basis vectors, and two orange arrows Ve₁ and Ve₂ representing V applied to the same basis. Annotations show the per-index inner products m(Ue_j, Ve_j) and their sum equal to (ℓ̌ ⊗ m)(U, V)." loading="lazy">
  <figcaption>How the tensor scalar product (A.22) / (A.25b) really works. Pick an $\ell$-orthonormal frame $(e_j)$ of $\mathcal X$ and apply $U$ and $V$ to each frame vector — this produces two "image frames" $(Ue_j), (Ve_j) \subset \mathcal Y$ (here in $\mathbb R^2$ with $e_1, e_2$ canonical). The scalar product of $U$ and $V$ as elements of $\mathcal L(\mathcal X, \mathcal Y) \cong \widecheck{\mathcal X} \otimes \mathcal Y$ is then the sum of the pointwise $m$-inner products $m(Ue_j, Ve_j)$, evaluated index-by-index. The basis-independence proposition (A.24) says this sum does not depend on which orthonormal frame we picked — a clean way to see why $\widecheck\ell \otimes m$ is intrinsically defined on operators rather than on matrices.</figcaption>
</figure>

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

<figure>
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
</figure>

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

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_isomorphism_currying.png' | relative_url }}" alt="Two side-by-side block diagrams. Left: a single arrow from a large box labelled L(E, F) (containing the element W = x̌ ⊗ y) to a smaller box labelled G, with the arrow labelled 𝒜 and W ↦ 𝒜(W). Right: a chain of three boxes — Ě, then L(F, G), then G — connected by two arrows; the first labelled 𝒜 with x̌ ↦, the second labelled eval_y, and the middle box noting y ↦ 𝒜(x̌)(y)." loading="lazy">
  <figcaption>Theorem A.11 in pictures: the same operator $\mathcal A$ admits two equivalent readings. <em>Left:</em> as a single map $\mathcal L(E, F) \to G$, eating an entire linear map $W = \widecheck x \otimes y$ and returning an element of $G$. <em>Right:</em> in curried form, $\mathcal A$ is a map $\widecheck E \to \mathcal L(F, G)$: feed it just the covector part $\widecheck x$, get back another linear map $\mathcal A(\widecheck x) : F \to G$, then evaluate that on the vector part $y$. The defining identity $\mathcal A(\widecheck x \otimes y) = \mathcal A(\widecheck x)(y)$ from (A.62) is precisely the assertion that these two readings produce the same answer, so the two function spaces are isomorphic via uncurrying / currying.</figcaption>
</figure>

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

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/gdl_a_tensor_product_operators.png' | relative_url }}" alt="A four-stage horizontal pipeline: rounded boxes labelled E', E, F, F' connected by arrows labelled A (purple), W (red), and B (purple) in turn. Below, the formula (Ǎ ⊗ B)(W) := BWA ∈ L(E', F') and the action on outer products (Ǎ ⊗ B)(x̌ ⊗ y) = (Ǎx̌) ⊗ (By)." loading="lazy">
  <figcaption>The tensor product of operators $\widecheck A \otimes B$ from (A.69) as a four-stage pipeline. Given $W \in \mathcal L(E, F)$ — an arbitrary linear map between the "inner" spaces — the operator $\widecheck A \otimes B$ pre-composes with $A$ on the input side and post-composes with $B$ on the output side, producing $BWA \in \mathcal L(E', F')$. The pipeline view makes clear why $\widecheck A \otimes B$ acts on the entire space of operators $\mathcal L(E, F)$ rather than on individual vectors. On rank-$1$ operators it factors as in (A.70): $(\widecheck A \otimes B)(\widecheck x \otimes y) = (\widecheck A\widecheck x) \otimes (By)$, with $A$ pulling back the covector and $B$ pushing forward the vector.</figcaption>
</figure>

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

The operator $A L^{-1}\widecheck{A} : \widecheck{\mathcal{Y}} \to \mathcal{Y}$ is invertible. Indeed, if $\widecheck{y}\ne 0$, then surjectivity of $A$ implies $\widecheck{A}\widecheck{y}\ne 0$, and therefore

$$
\langle \widecheck{y}, A L^{-1}\widecheck{A}\widecheck{y}\rangle
= \langle \widecheck{A}\widecheck{y}, L^{-1}\widecheck{A}\widecheck{y}\rangle
= \widecheck{\lambda}(\widecheck{A}\widecheck{y})^2
> 0.
$$

Thus

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

The critical point condition is exactly (1.2f). Since the quadratic part is strictly convex by the same positivity argument above, this critical point is the unique minimizer of (1.2d), namely $\widecheck{y}^*$.

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

Assume that the input vectors $(x_k)_{k \in [n]}$ are linearly independent, and let $U \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be given. Then there exists a unique operator

$$
W_{\mathcal{D}} := U - G^{kl}(Lx_l) \otimes (Ux_k - y_k) \in \mathcal{L}(\mathcal{X}, \mathcal{Y}) \tag{1.6}
$$

satisfying the interpolation constraints (1.4b) and minimizing the distance

$$
(\widecheck{\lambda} \otimes \mu)(W-U)
$$

from $W$ to $U$. If, in addition, the input vectors $(x_k)_{k \in [n]}$ are $\ell$-orthogonal, then

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

Let $E = \mathbb{R}^n$ with canonical basis $(\delta_k)_{k \in [n]}$, and define

$$
A \in \mathcal{L}(E,\mathcal{X}), \qquad A\delta_k := x_k,
\qquad
Y \in \mathcal{L}(E,\mathcal{Y}), \qquad Y\delta_k := y_k .
$$

The interpolation constraints are equivalent to

$$
WA = Y.
$$

Since the vectors $(x_k)_{k \in [n]}$ are linearly independent, $A$ is injective. Thus Corollary A.9 applies to the correction $V := W-U$: among all $V$ satisfying

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

Finally, suppose $\mathcal{D}_n=\mathcal{D}_{n-1}\cup\lbrace(x_n,y_n)\rbrace$ and take $U=W_{\mathcal{D}_{n-1}}$. Then $Ux_k=y_k$ for all $k<n$, so in (1.6) all residuals $Ux_k-y_k$ vanish except the one with $k=n$. Therefore

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
