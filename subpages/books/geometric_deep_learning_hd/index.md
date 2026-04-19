---
layout: note
title: "Introduction to Geometric Deep Learning"
date: 2026-04-19
excerpt: "Notes on 'Introduction to Geometric Deep Learning' (SS 2026) — starting with the appendix on mathematical background: Euclidean structure, orthogonal inverses, pseudo-inverses, tensor products and generalized inverses."
tags:
  - geometric-deep-learning
  - linear-algebra
  - tensor-products
  - pseudo-inverse
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

# Introduction to Geometric Deep Learning

## Appendix A. Mathematical Background

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">A.1 (Summation convention)</span></p>

Sums are taken "automatically" over repeated indices, without the $\sum$ symbol, whenever the summands and their range make sense in the context.

</div>

### A.1. Basic Euclidean Set-Up

We supply finite-dimensional real vector spaces $\mathcal{X}$ and $\mathcal{Y}$ with scalar products $\ell$ and $m$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dual spaces)</span></p>

The **dual spaces** of $\mathcal{X}$ and $\mathcal{Y}$ are denoted by

$$
\widetilde{\mathcal{X}} = \mathcal{L}(\mathcal{X}, \mathbb{R}), \qquad \widetilde{\mathcal{Y}} = \mathcal{L}(\mathcal{Y}, \mathbb{R}). \tag{A.1}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Duality mappings and pairings)</span></p>

The **duality mappings** and associated **duality pairings** are

$$
\begin{aligned}
L \in \mathcal{L}(\mathcal{X}, \widetilde{\mathcal{X}}): & \quad \langle L x, x' \rangle = \ell(x, x'), \quad \forall\, x, x' \in \mathcal{X}, &&\text{(A.2a)} \\
M \in \mathcal{L}(\mathcal{Y}, \widetilde{\mathcal{Y}}): & \quad \langle M y, y' \rangle = m(y, y'), \quad \forall\, y, y' \in \mathcal{Y}. &&\text{(A.2b)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Induced norms)</span></p>

The norms induced by the scalar products are

$$
\lambda(x) := \sqrt{\ell(x, x)}, \qquad \mu(y) := \sqrt{m(y, y)}. \tag{A.3}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Gramian representation)</span></p>

For a basis $(x_i)_{1 \le i \le n}$ of $\mathcal{X}$, one has $\det L > 0$ with

$$
L_{ij} = \ell(x_i, x_j) = \langle x_i, L x_j \rangle. \tag{A.4}
$$

The Gramian matrix $(L_{ij})_{1 \le i, j \le n}$ is symmetric, positive definite, and invertible. It represents the operator $L$ in the basis $(x_i)_{1 \le i \le n}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dual scalar product and dual norm)</span></p>

The **dual scalar product** on $\widetilde{\mathcal{X}}$ is given by

$$
\widetilde{\ell}(p, p') = L^{-1}_{ij}\, p_i\, p'_j = \langle p, L^{-1} p' \rangle, \qquad p, p' \in \widetilde{\mathcal{X}}, \tag{A.5}
$$

with corresponding norm

$$
\widetilde{\lambda}(p) := \sqrt{\widetilde{\ell}(p, p)}. \tag{A.6}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Canonical Euclidean basis)</span></p>

The canonical Euclidean basis $(b_k)_{1 \le k \le n} \subset \mathbb{R}^n$ is

$$
b_k = (0, \dots, 0, \underbrace{1}_{k\text{-th}}, 0, \dots, 0)^{\top}, \tag{A.7}
$$

and the corresponding coordinate vectors are often arbitrary and do not relate to the application at hand. Scalar products can then be used to adjust the geometry.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">A.2 (Making bases orthonormal)</span></p>

Let $(x_i)_{1 \le i \le n} \subset \mathcal{X}$ be any basis. Define (cf. Remark A.1)

$$
B \in \mathcal{L}(\mathbb{R}^n, \mathcal{X}), \qquad B\, b_k := x_k, \quad k \in \{1, \dots, n\}. \tag{A.8}
$$

Then the basis $(x_i)_{1 \le i \le n}$ is orthonormal for the scalar product on $\mathcal{X}$ iff

$$
\ell(x_i, x_j) = \langle x_i, L x_j \rangle = \delta_{ij}, \tag{A.9}
$$

where $\langle \cdot, \cdot \rangle$ denotes the canonical scalar product on $\mathbb{R}^n$. The duality mapping is

$$
L = (B B^{\top})^{-1} \tag{A.10}
$$

with the transposed operator (and the identification $\widetilde{\mathbb{R}}^n \cong \mathbb{R}^n$):

$$
B^{\top} \in \mathcal{L}(\widetilde{\mathcal{X}}, \mathbb{R}^n). \tag{A.11}
$$

</div>

### A.1.2. Orthogonal Left- and Right-Inverses

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">A.3 (Orthogonal Left- and Right-Inverses)</span></p>

**(a)** Let $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be injective. Then

$$
A^{-} := (A^{\top} M A)^{-1} A^{\top} M \in \mathcal{L}(\mathcal{Y}, \mathcal{X}) \tag{A.11}
$$

is the **orthogonal left-inverse** of $A$. In particular,

$$
\begin{aligned}
A^{-} A &= I_{\mathcal{X}}, &&\text{(A.12a)} \\
A A^{-} &= \Pi_{\mathrm{rge}(A)} &&\text{(orthogonal projection onto the range of } A\text{).} &&\text{(A.12b)}
\end{aligned}
$$

**(b)** Let $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be surjective. Then

$$
A^{+} := L^{-1} A^{\top} (A L^{-1} A^{\top})^{-1} \in \mathcal{L}(\mathcal{Y}, \mathcal{X}) \tag{A.13}
$$

is the **orthogonal right-inverse** of $A$. In particular,

$$
\begin{aligned}
A A^{+} &= I_{\mathcal{Y}}, &&\text{(A.14a)} \\
I - A^{+} A &= \Pi_{\ker(A)} &&\text{(orthogonal projection onto the kernel (nullspace) of } A\text{).} &&\text{(A.14b)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">A.4 (Properties of orthogonal partial inverses)</span></p>

**(a)** Let $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be injective. Then

$$
\begin{aligned}
\widetilde{A^{-}} &= \widetilde{A}^{+}, &&\text{(A.15a)} \\
A^{--} &= A, &&\text{(A.15b)} \\
(A^{\top} M A)^{-1} &= A^{-} M^{-1} (A^{\top})^{-1}, &&\text{(A.15c)} \\
(A V)^{-} &= V^{-1} A^{-} &&\text{if } V \in \mathcal{L}(\mathcal{X}, \mathcal{X}) \text{ is invertible.} &&\text{(A.15d)}
\end{aligned}
$$

**(b)** Let $A \in \mathcal{L}(\mathcal{X}, \mathcal{Y})$ be surjective. Then

$$
\begin{aligned}
\widetilde{A^{+}} &= \widetilde{A}^{-}, &&\text{(A.16a)} \\
A^{++} &= A, &&\text{(A.16b)} \\
(A L^{-1} A^{\top})^{-1} &= (A^{+})^{\top} L A^{+}, &&\text{(A.16c)} \\
(V A)^{+} &= A^{+} V^{-1} &&\text{if } V \in \mathcal{L}(\mathcal{Y}, \mathcal{Y}) \text{ is invertible.} &&\text{(A.16d)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">A.5 (Pseudo-inverse)</span></p>

Let

$$
C = B A \in \mathcal{L}(\mathcal{X}, \mathcal{Y}) \tag{A.17}
$$

with injective factor $B$ and surjective factor $A$. Then

$$
C^{\dagger} := A^{+} B^{-} \in \mathcal{L}(\mathcal{Y}, \mathcal{X}) \tag{A.18}
$$

is called the **pseudo-inverse** of $C$. For a given $y \in \mathcal{Y}$, the vector

$$
\widehat{x} = C^{\dagger} y \tag{A.19}
$$

has minimal norm $\lambda(\widehat{x})$ among all points $x \in \mathcal{X}$ mapped by $C$ to the closest point to $y$ in $\mathrm{rge}(C)$. One has

$$
\begin{aligned}
C^{\dagger} &= A^{+} B^{-}, &&\text{(A.20a)} \\
C^{\dagger} &= A^{+} B^{-1} &&\text{if } B \text{ bijective}, &&\text{(A.20b)} \\
C^{\dagger} &= A^{-1} B^{-} &&\text{if } A \text{ bijective}. &&\text{(A.20c)}
\end{aligned}
$$

</div>

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
\widetilde{C^{\dagger}} &= \widetilde{C}^{\dagger}, &&\text{(A.21f)} \\
(C_2 C_1)^{\dagger} &= C_1^{\dagger} C_2^{\dagger}. &&\text{(A.21g)}
\end{aligned}
$$

</div>

### A.1.3. Tensor-Products and Generalized Inverses

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Scalar product on $\mathcal{X} \otimes \mathcal{Y}$)</span></p>

In the situation (A.2), (A.3), the space $\mathcal{X} \otimes \mathcal{Y} \cong \mathcal{L}(\widetilde{\mathcal{X}}, \mathcal{Y})$ is equipped with the scalar product (cf. Remark A.1)

$$
(\ell \otimes m)(V, V') := \langle V_i, V'_j \rangle_{\mathcal{Y}}\, L^{ij}, \qquad V, V' \in \mathcal{L}(\widetilde{\mathcal{X}}, \mathcal{Y}), \tag{A.22}
$$

where $(x_i)_{1 \le i \le n} \subset \mathcal{X}$ is any basis of $\mathcal{X}$ and $V_i = V(\widetilde{x}_i)$ denote the images of the corresponding co-basis.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Basis-independence of (A.22))</span></p>

The expression defining the inner product (A.22) does not depend on the choice of these bases, since

$$
\begin{aligned}
(\ell \otimes m)(V, V') &= \langle V_i, V'_j \rangle_{\mathcal{Y}}\, L^{ij} = \langle V(L^{-1} \widetilde{x}_i), V'(\widetilde{x}_j) \rangle_{\mathcal{Y}} &&\text{(A.23a)} \\
&= \langle V, V' \rangle_{\mathcal{L}(\widetilde{\mathcal{X}}, \mathcal{Y})} &&\text{(A.23b)} \\
&= (M \otimes L^{-1})(V, V'). &&\text{(A.23c)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Companion scalar products)</span></p>

Similarly, using $V_i = V(x_i) \in \widetilde{\mathcal{Y}}$,

$$
\begin{aligned}
(\widetilde{\ell} \otimes \widetilde{m})(P, P') &= \langle P_i, P'_j \rangle_{\widetilde{\mathcal{Y}}}\, (L^{-1})_{ij} = \langle P, P' \rangle_{\mathcal{L}(\mathcal{X}, \widetilde{\mathcal{Y}})}, &&\text{(A.24a)} \\
(\ell \otimes \widetilde{m})(Q, Q') &= \langle Q_i, Q'_j \rangle_{\widetilde{\mathcal{Y}}}\, L^{ij}, &&\text{(A.24b)} \\
(\widetilde{\ell} \otimes m)(W, W') &= \langle W_i, W'_j \rangle_{\mathcal{Y}}\, (L^{-1})_{ij}. &&\text{(A.24c)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Compact form of (A.22))</span></p>

The definition (A.22) is equivalent to the expression

$$
(\ell \otimes m)(V, V') = \langle V, (L^{-1} \otimes M)\, V' \rangle_{\mathcal{L}(\widetilde{\mathcal{X}}, \mathcal{Y})}, \tag{A.25}
$$

independent of the choice of the bases. Recall from (A.6) that $\widetilde{\mathcal{L}}(\mathcal{X}, \mathcal{Y}) = \mathcal{L}(\mathcal{X}, \mathcal{Y})^{\vee}$.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">A.7 (Scalar product of tensor products: duality mapping)</span></p>

The duality mapping corresponding to the scalar product (A.25) is

$$
L^{-1} \otimes M \in \mathcal{L}\bigl(\mathcal{L}(\mathcal{X}, \mathcal{Y}),\, \mathcal{L}(\widetilde{\mathcal{X}}, \widetilde{\mathcal{Y}})\bigr). \tag{A.26}
$$

We denote the norm on $\mathcal{L}(\mathcal{X}, \mathcal{Y})$ induced by the scalar product (A.25) by

$$
(\widetilde{\lambda} \otimes \mu)(U) := \sqrt{(\widetilde{\ell} \otimes m)(U, U)}. \tag{A.27}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">A.8 (Orthogonal right-inverse)</span></p>

Let $A \in \mathcal{L}(\mathcal{X}', \mathcal{X})$ be injective and $B \in \mathcal{L}(\mathcal{Y}, \mathcal{Y}')$ be surjective. Then $\widetilde{A} \otimes B$ is surjective with the orthogonal right-inverse

$$
(\widetilde{A} \otimes B)^{+} = \widetilde{A}^{+} \otimes B^{+} \in \mathcal{L}\bigl(\mathcal{L}(\mathcal{X}', \mathcal{Y}'),\, \mathcal{L}(\mathcal{X}, \mathcal{Y})\bigr). \tag{A.28}
$$

Furthermore, the minimal $(\widetilde{\lambda} \otimes \mu)$-norm solution to the equation

$$
\begin{aligned}
B W A &= W' \in \mathcal{L}(\mathcal{X}', \mathcal{Y}') \quad \text{is} &&\text{(A.29a)} \\
W &= B^{+} W' A^{-} \in \mathcal{L}(\mathcal{X}, \mathcal{Y}). &&\text{(A.29b)}
\end{aligned}
$$

</div>

<div class="math-callout math-callout--corollary" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">A.9</span></p>

Assume $A \in \mathcal{L}(\mathcal{X}', \mathcal{X})$ is injective. Then the minimal $(\widetilde{\lambda} \otimes \mu)$-norm solution to the equation

$$
\begin{aligned}
W A &= W' \in \mathcal{L}(\mathcal{X}', \mathcal{Y}) \quad \text{is} &&\text{(A.30a)} \\
W &= W' A^{-} \in \mathcal{L}(\mathcal{X}, \mathcal{Y}), &&\text{(A.30b)}
\end{aligned}
$$

which does not depend on the scalar product $m$ on $\mathcal{Y}$.

</div>
