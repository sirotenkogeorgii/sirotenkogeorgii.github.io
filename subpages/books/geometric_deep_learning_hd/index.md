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

## Appendix A. Mathematical Background

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">A.1 (Summation convention)</span></p>

Sums are taken "automatically" over repeated indices, without the $\sum$ symbol, whenever the summands and their range make sense in the context.

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

Let $E, F$ be vector spaces of $\mathbb{R}$. Let $\widecheck{E}$ be the dual space of $E$. Throughout this section, if $(e_i)\_{i \in [n]}$ is a basis of $E$ and $(\widecheck{e}^i)\_{i \in [n]}$ is a dual basis of $\widecheck{E}$, then

$$
\langle \widecheck{e}^i, e_j\rangle = \delta^i_j = \begin{cases} 1 & \text{if } i = j, \\ 0 & \text{otherwise.} \end{cases} \qquad \textbf{(Kronecker delta)} \tag{A.31}
$$

The space of $F$-valued tensors

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

Let $e_1, \dots, e_n \in E$ be a basis with dual basis $\widecheck{e}^1, \dots, \widecheck{e}^n$. Then any function $f \in T^{p,q}(E)$ can be uniquely specified as

$$
f = \sum f^{i_1 \cdots i_p}_{j_1 \cdots j_q}\, e_{i_1} \otimes \cdots \otimes e_{i_p} \otimes \widecheck{e}^{j_1} \otimes \cdots \otimes \widecheck{e}^{j_q} \tag{A.35}
$$

with coefficients

$$
f^{i_1 \cdots i_p}_{j_1 \cdots j_q} = f(\widecheck{e}^{i_1}, \dots, \widecheck{e}^{i_p}, e_{j_1}, \dots, e_{j_q}) \in \mathbb{R}. \tag{A.36}
$$

Here the sum ranges over all $p$-tuples $1 \leq i_1, \dots, i_p \leq n$ and all $q$-tuples $1 \leq j_1, \dots, j_q \leq n$ with $n = \dim(E)$.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">A.10 (tensors, duality)</span></p>

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
(\widecheck{e}^j \otimes f_i)\_{\substack{i \in [m] \\ j \in [n]}} \tag{A.50}
$$

form a basis of the space $\mathcal{L}(E, F)$. The transformation of the basis $(e_j)\_{j \in [n]}$ determines the components of the matrix $W$ which represents the transformation in $\mathcal{L}(E, F)$ in the basis (A.50),

$$
\begin{aligned}
F \ni Wx &= \langle \widecheck{f}^i, Wx\rangle f_i = \langle \widecheck{f}^i, W(x^j e_j)\rangle f_i = \langle \widecheck{f}^i, W(\langle \widecheck{e}^j, x\rangle e_j)\rangle f_i &&\text{(A.51a)} \\
&= \langle \widecheck{f}^i, W e_j\rangle\, \langle \widecheck{e}^j, x\rangle\, f_i = \underbrace{\langle \widecheck{f}^i, W e_j\rangle}\_{=:\,W^i_j}\, (\widecheck{e}^j \otimes f_i)(x) &&\text{(A.51b)} \\
&= (W^i_j\, \widecheck{e}^j \otimes f_i)(x). &&\text{(A.51c)}
\end{aligned}
$$

**(5)** A vector $x \in E$ is transformed by the operator (A.43) to

$$
(\widecheck{x} \otimes y)(x) = (\widecheck{x}_j \widecheck{e}^j) \otimes (y^i f_i)(x) = (y^i \widecheck{x}_j)(\widecheck{e}^j \otimes f_i)(x) = (W^i_j\, \widecheck{e}^j \otimes f_i)(x). \tag{A.52}
$$

Thus, in terms of the coordinates and matrix-vector notation, the matrix representing this linear mapping is

$$
W = (W^i_j)\_{\substack{i \in [m] \\ j \in [n]}} = y\, \widecheck{x}^{\top}. \tag{A.53}
$$

**(6)** The **transpose**

$$
(\widecheck{x} \otimes y)^{\vee} \in \mathcal{L}(\widecheck{F}, \widecheck{E}) \tag{A.54}
$$

maps a vector $\widecheck{y} \in \widecheck{F}$ to

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

We focus on *linear* mappings

$$
\mathcal{A}: \mathcal{L}(E, F) \to G, \tag{A.60}
$$

that is, on elements $\mathcal{A} \in \mathcal{L}(\mathcal{L}(E, F), G)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">A.11 (isomorphism)</span></p>

One has the isomorphism

$$
\mathcal{L}(\mathcal{L}(E, F), G) \cong \mathcal{L}(\widecheck{E}, \mathcal{L}(F, G)) \tag{A.61}
$$

which is identified by the equation

$$
G \ni \underbrace{\mathcal{A}(\overbrace{\widecheck{x} \otimes y}^{\in\, \mathcal{L}(E, F)})}\_{\text{l.h.s.}} := \underbrace{\mathcal{A}(\widecheck{x})(y)}\_{\text{r.h.s.}} \in G, \qquad \forall\, \widecheck{x} \in \widecheck{E},\ \forall\, y \in F. \tag{A.62}
$$

Note that when $G = \mathbb{R}$, then (A.61) characterizes **dual spaces** of linear operators

$$
(\widecheck{E} \otimes F)^{\vee} \cong \boxed{\,\mathcal{L}(E, F)^{\vee} \cong \mathcal{L}(\widecheck{E}, \widecheck{F})\,} \cong E \otimes \widecheck{F}. \tag{A.63}
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">A.12 (mapping $\mathcal{A} \in \mathcal{L}(\mathcal{L}(E, F), G)$)</span></p>

Let $x \in E$ and $B \in \mathcal{L}(F, G)$ and set

$$
\mathcal{A} = x \otimes B:\ W \in \mathcal{L}(E, F) \mapsto \mathcal{A}(W) = (x \otimes B)(W) := BWx \in G. \tag{A.64}
$$

On the one hand, this definition clearly reveals $\mathcal{A} = x \otimes B \in \mathcal{L}(\mathcal{L}(E, F), G)$ which *linearly* maps $W$ to an element of $G$. On the other hand, by the isomorphism (A.61), $\mathcal{A} = x \otimes B \in \mathcal{L}(\widecheck{E}, \mathcal{L}(F, G))$ which becomes explicit by the right-hand side of (A.62): representing the action of $W$ by $W = \widecheck{x} \otimes y$, one has

$$
\mathcal{A}(W) = (x \otimes B)(\widecheck{x} \otimes y) \stackrel{(A.64)}{=} B(\widecheck{x} \otimes y)x = \langle \widecheck{x}, x\rangle\, By = \underbrace{\mathcal{A}(\widecheck{x})}\_{\in\, \mathcal{L}(F, G)}(y) \in G. \tag{A.65}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">A.13 (dual spaces of linear operators)</span></p>

Based on the identification (A.63), one has $x \otimes \widecheck{y} \in \mathcal{L}(E, F)^{\vee}$ and

$$
\begin{aligned}
\langle x \otimes \widecheck{y},\, W \rangle &= \langle \widecheck{y}, Wx \rangle, &&\text{(A.66a)} \\
\langle x \otimes \widecheck{y},\, \widecheck{x} \otimes y \rangle &= \langle \widecheck{x}, x \rangle\, \langle \widecheck{y}, y \rangle. &&\text{(A.66b)}
\end{aligned}
$$

If $(e_j)\_{j \in [n]}, (\widecheck{e}^j)\_{j \in [n]}$ and $(f_i)\_{i \in [m]}, (\widecheck{f}^i)\_{i \in [m]}$ are dual bases of $E, \widecheck{E}$ and $F, \widecheck{F}$, then

$$
(\widecheck{e}^j \otimes f_i)\_{\substack{i \in [m] \\ j \in [n]}} \qquad \text{and} \qquad (e_l \otimes \widecheck{f}^k)\_{\substack{k \in [m] \\ l \in [n]}} \tag{A.67}
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
(\widecheck{A} \otimes B)(\widecheck{x} \otimes y) = \underbrace{\widecheck{A}\widecheck{x}}\_{\in\, \widecheck{E}'} \otimes\, By \in \mathcal{L}(E', F'). \tag{A.70}
$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">A.14 (transposed tensor products of linear operators)</span></p>

The transpose of the linear operator (A.69) is

$$
\boxed{\,(\widecheck{A} \otimes B)^{\vee} = A \otimes \widecheck{B} \in \mathcal{L}(\mathcal{L}(\widecheck{E}', \widecheck{F}'),\, \mathcal{L}(\widecheck{E}, \widecheck{F}))\,}. \tag{A.71}
$$

Now assume (A.69) and in addition $A' \in \mathcal{L}(E'', E')$ and $B' \in \mathcal{L}(F', F'')$ to be given. Then, with $\widecheck{A'} \otimes B' \in \mathcal{L}(\mathcal{L}(E', F'),\, \mathcal{L}(E'', F''))$, one has

$$
\begin{aligned}
\boxed{\,(\widecheck{A'} \otimes B')(\widecheck{A} \otimes B)\,} &= (\underbrace{AA'}\_{\in\, \mathcal{L}(E'', E)})^{\vee} \otimes (\underbrace{B'B}\_{\in\, \mathcal{L}(F, F'')}) &&\text{(A.72a)} \\
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
