---
layout: default
title: "Representation Theory: A First Course — Harris & Fulton"
date: 2026-03-19
excerpt: Notes on representations of finite groups, characters, and their applications from Fulton & Harris.
tags:
  - representation-theory
  - algebra
  - group-theory
---

**Table of Contents**
- TOC
{:toc}

# Part I: Finite Groups

Representation theory is very much a 20th-century subject. In the 19th century, groups were generally understood as subsets of permutations or as automorphisms $\mathrm{GL}(V)$ of a vector space. Only in the 20th century did the notion of an abstract group make it possible to distinguish between properties of the abstract group and properties of a particular realization as a subgroup of a permutation group or $\mathrm{GL}(V)$.

The study of "group theory" factors into two parts:
1. The study of the structure of abstract groups (e.g., classification of simple groups).
2. The companion question: given a group $G$, how can we describe all the ways in which $G$ may be embedded in (or mapped to) a linear group $\mathrm{GL}(V)$? This is the subject matter of **representation theory**.

The first six lectures are devoted to finite groups. Many techniques developed here carry over to Lie groups.

---

## Lecture 1: Representations of Finite Groups

### 1.1 Definitions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Representation)</span></p>

A **representation** of a finite group $G$ on a finite-dimensional complex vector space $V$ is a homomorphism $\rho\colon G \to \mathrm{GL}(V)$ of $G$ to the group of automorphisms of $V$. We say that such a map gives $V$ the structure of a **$G$-module**. The dimension of $V$ is called the **degree** of $\rho$.

When there is little ambiguity about the map $\rho$, we sometimes call $V$ itself a representation of $G$; we will often suppress the symbol $\rho$ and write $g \cdot v$ or $gv$ for $\rho(g)(v)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Map of Representations, $G$-linear Map)</span></p>

A **map** $\varphi$ between two representations $V$ and $W$ of $G$ is a vector space map $\varphi\colon V \to W$ such that the diagram

$$\varphi \circ g = g \circ \varphi \quad \text{for every } g \in G$$

commutes (i.e., $\varphi(gv) = g\varphi(v)$ for all $g \in G$, $v \in V$). This is called a **$G$-linear map**. We can then define $\ker \varphi$, $\operatorname{Im} \varphi$, and $\operatorname{Coker} \varphi$, which are also $G$-modules.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subrepresentation, Irreducible Representation)</span></p>

A **subrepresentation** of a representation $V$ is a vector subspace $W$ of $V$ which is invariant under $G$. A representation $V$ is called **irreducible** if there is no proper nonzero invariant subspace $W$ of $V$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Direct Sum, Tensor Product, Exterior and Symmetric Powers)</span></p>

If $V$ and $W$ are representations, the **direct sum** $V \oplus W$ and the **tensor product** $V \otimes W$ are also representations, the latter via

$$g(v \otimes w) = gv \otimes gw.$$

For a representation $V$, the $n$th tensor power $V^{\otimes n}$ is again a representation of $G$ by this rule, and the **exterior powers** $\bigwedge^n(V)$ and **symmetric powers** $\mathrm{Sym}^n(V)$ are subrepresentations of it.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dual Representation)</span></p>

The **dual** $V^* = \mathrm{Hom}(V, \mathbb{C})$ of $V$ is also a representation. We want the two representations of $G$ to respect the natural pairing $\langle\, ,\, \rangle$ between $V^*$ and $V$, so that if $\rho\colon G \to \mathrm{GL}(V)$ is a representation and $\rho^*\colon G \to \mathrm{GL}(V^*)$ is the dual, we should have

$$\langle \rho^*(g)(v^*),\, \rho(g)(v) \rangle = \langle v^*,\, v \rangle$$

for all $g \in G$, $v \in V$, and $v^* \in V^*$. This forces us to define the dual representation by

$$\rho^*(g) = {}^t\!\rho(g^{-1})\colon V^* \to V^*.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hom as a Representation)</span></p>

If $V$ and $W$ are representations, then $\mathrm{Hom}(V, W)$ is also a representation, via the identification $\mathrm{Hom}(V, W) = V^* \otimes W$. Unraveling this, if we view an element of $\mathrm{Hom}(V, W)$ as a linear map $\varphi$ from $V$ to $W$, we have

$$(g\varphi)(v) = g\varphi(g^{-1}v)$$

for all $v \in V$. The space of $G$-linear maps between two representations $V$ and $W$ of $G$ is the subspace $\mathrm{Hom}(V, W)^G$ of elements of $\mathrm{Hom}(V, W)$ fixed under the action of $G$, often denoted $\mathrm{Hom}_G(V, W)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Identities for Representations)</span></p>

The usual identities for vector spaces are also true for representations, e.g.,

$$V \otimes (U \oplus W) = (V \otimes U) \oplus (V \otimes W),$$

$$\bigwedge^k(V \oplus W) = \bigoplus_{a+b=k} \bigwedge^a V \otimes \bigwedge^b W,$$

$$\bigwedge^k(V^*) = \bigwedge^k(V)^*.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Permutation Representation)</span></p>

If $X$ is any finite set and $G$ acts on the left on $X$, i.e., $G \to \mathrm{Aut}(X)$ is a homomorphism to the permutation group of $X$, there is an associated **permutation representation**: let $V$ be the vector space with basis $\lbrace e_x \colon x \in X \rbrace$, and let $G$ act on $V$ by

$$g \cdot \sum a_x e_x = \sum a_x e_{gx}.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regular Representation)</span></p>

The **regular representation**, denoted $R_G$ or $R$, corresponds to the left action of $G$ on itself. Alternatively, $R$ is the space of complex-valued functions on $G$, where an element $g \in G$ acts on a function $\alpha$ by $(g\alpha)(h) = \alpha(g^{-1}h)$.

</div>

### 1.2 Complete Reducibility; Schur's Lemma

A representation is **indecomposable** if it cannot be expressed as a direct sum of others. Happily, the situation is as nice as it could possibly be: a representation is atomic in this sense if and only if it is irreducible, and every representation is the direct sum of irreducibles, in a suitable sense uniquely so.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(1.5 — Complementary Subrepresentation)</span></p>

If $W$ is a subrepresentation of a representation $V$ of a finite group $G$, then there is a complementary invariant subspace $W'$ of $V$, so that $V = W \oplus W'$.

**Proof sketch.** One can introduce a (positive definite) Hermitian inner product $H$ on $V$ which is preserved by each $g \in G$ (i.e., $H(gv, gw) = H(v, w)$ for all $v, w \in V$ and $g \in G$). Indeed, if $H_0$ is any Hermitian product on $V$, one gets such an $H$ by averaging over $G$:

$$H(v, w) = \sum_{g \in G} H_0(gv, gw).$$

Then the perpendicular subspace $W^\perp$ is complementary to $W$ in $V$.

Alternatively, one can choose an arbitrary subspace $U$ complementary to $W$, let $\pi_0\colon V \to W$ be the projection given by the direct sum decomposition $V = W \oplus U$, and average $\pi_0$ over $G$: that is, take

$$\pi(v) = \sum_{g \in G} g(\pi_0(g^{-1}v)).$$

This will be a $G$-linear map from $V$ onto $W$, which is multiplication by $\lvert G \rvert$ on $W$; its kernel will therefore be a subspace of $V$ invariant under $G$ and complementary to $W$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(1.6 — Complete Reducibility)</span></p>

Any representation is a direct sum of irreducible representations.

This property is called **complete reducibility**, or **semisimplicity**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Failure of Complete Reducibility)</span></p>

For continuous representations, the circle $S^1$, or any compact group, has the property of complete reducibility; integration over the group (with respect to an invariant measure on the group) plays the role of averaging. The (additive) group $\mathbb{R}$ does **not** have this property: the representation

$$a \mapsto \begin{pmatrix} 1 & a \\ 0 & 1 \end{pmatrix}$$

leaves the $x$-axis fixed, but there is no complementary invariant subspace. Other Lie groups such as $\mathrm{SL}_n(\mathbb{C})$ are semisimple in this sense. Note also that this argument would fail if the vector space $V$ was over a field of finite characteristic.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(1.7 — Schur's Lemma)</span></p>

If $V$ and $W$ are irreducible representations of $G$ and $\varphi\colon V \to W$ is a $G$-module homomorphism, then

1. Either $\varphi$ is an isomorphism, or $\varphi = 0$.
2. If $V = W$, then $\varphi = \lambda \cdot I$ for some $\lambda \in \mathbb{C}$, $I$ the identity.

**Proof.** The first claim follows from the fact that $\ker \varphi$ and $\operatorname{Im} \varphi$ are invariant subspaces. For the second, since $\mathbb{C}$ is algebraically closed, $\varphi$ must have an eigenvalue $\lambda$, i.e., for some $\lambda \in \mathbb{C}$, $\varphi - \lambda I$ has a nonzero kernel. By (1), then, we must have $\varphi - \lambda I = 0$, so $\varphi = \lambda I$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(1.8 — Unique Decomposition into Irreducibles)</span></p>

For any representation $V$ of a finite group $G$, there is a decomposition

$$V = V_1^{\oplus a_1} \oplus \cdots \oplus V_k^{\oplus a_k},$$

where the $V_i$ are distinct irreducible representations. The decomposition of $V$ into a direct sum of the $k$ factors is unique, as are the $V_i$ that occur and their multiplicities $a_i$.

**Proof.** It follows from Schur's lemma that if $W$ is another decomposition of $G$, with a decomposition $W = \bigoplus W_j^{\oplus b_j}$, and $\varphi\colon V \to W$ is a map of representations, then $\varphi$ must map the factor $V_i^{\oplus a_i}$ into that factor $W_j^{\oplus b_j}$ for which $W_j \cong V_i$; when applied to the identity map of $V$ to $V$, the stated uniqueness follows. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Goals of Representation Theory)</span></p>

Our goals in analyzing the representations of any group $G$ will therefore be:

1. **Describe all the irreducible representations of $G$.**
2. **Find techniques for giving the direct sum decomposition** $V = a_1 V_1 + \cdots + a_k V_k$, and in particular determining the multiplicities $a_i$ of an arbitrary representation $V$.
3. **Plethysm:** Describe the decompositions, with multiplicities, of representations derived from a given representation $V$, such as $V \otimes V$, $V^*$, $\bigwedge^k(V)$, $\mathrm{Sym}^k(V)$, and $\bigwedge^k(\bigwedge^l V)$.

</div>

### 1.3 Examples: Abelian Groups; $\mathfrak{S}_3$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Irreducible Representations of Abelian Groups)</span></p>

If $V$ is a representation of the finite group $G$, abelian or not, each $g \in G$ gives a map $\rho(g)\colon V \to V$; but this map is *not generally* a $G$-module homomorphism. Indeed, $\rho(g)$ will be $G$-linear for every $\rho$ if and only if $g$ is in the center $Z(G)$ of $G$.

In particular, if $G$ is abelian, and $V$ is an irreducible representation, then by Schur's lemma every element $g \in G$ acts on $V$ by a scalar multiple of the identity. Every subspace of $V$ is thus invariant; so $V$ must be **one-dimensional**. The irreducible representations of an abelian group $G$ are thus simply elements of the dual group, that is, homomorphisms

$$\rho\colon G \to \mathbb{C}^*.$$

</div>

#### The Symmetric Group $\mathfrak{S}_3$

We consider the simplest nonabelian group, $G = \mathfrak{S}_3$. As with any nontrivial symmetric group, we have two one-dimensional representations:

- The **trivial representation** $U$, with $gv = v$ for all $g$.
- The **alternating representation** $U'$, defined by $gv = \mathrm{sgn}(g)v$.

Since $G$ is a permutation group, we have a natural **permutation representation** on $\mathbb{C}^3$ by permuting coordinates. Explicitly, if $\lbrace e_1, e_2, e_3 \rbrace$ is the standard basis, then $g \cdot e_i = e_{g(i)}$, or equivalently,

$$g \cdot (z_1, z_2, z_3) = (z_{g^{-1}(1)}, z_{g^{-1}(2)}, z_{g^{-1}(3)}).$$

This permutation representation is not irreducible: the line spanned by the sum $(1, 1, 1)$ of the basis vectors is invariant, with complementary subspace

$$V = \lbrace (z_1, z_2, z_3) \in \mathbb{C}^3 \colon z_1 + z_2 + z_3 = 0 \rbrace.$$

This two-dimensional representation $V$ is easily seen to be irreducible; we call it the **standard representation** of $\mathfrak{S}_3$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analyzing Representations of $\mathfrak{S}_3$ via Abelian Subgroups)</span></p>

To describe an arbitrary representation $W$ of $\mathfrak{S}_3$, we look at the action of the abelian subgroup $\mathfrak{A}_3 = \mathbb{Z}/3 \subset \mathfrak{S}_3$ on $W$. Take $\tau$ to be any generator of $\mathfrak{A}_3$ (i.e., any three-cycle). Then $W$ is spanned by eigenvectors $v_i$ for $\tau$, whose eigenvalues are powers of a cube root of unity $\omega = e^{2\pi i/3}$. Thus,

$$W = \bigoplus V_i, \quad \text{where } V_i = \mathbb{C}v_i \text{ and } \tau v_i = \omega^{a_i} v_i.$$

Let $\sigma$ be any transposition, so that $\sigma$ and $\tau$ generate $\mathfrak{S}_3$ with the relation $\sigma\tau\sigma = \tau^2$. If $v$ is an eigenvector for $\tau$ with eigenvalue $\omega^i$, then $\sigma(v)$ is again an eigenvector for $\tau$, with eigenvalue $\omega^{2i}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Irreducible Representations of $\mathfrak{S}_3$)</span></p>

The **only three irreducible representations of $\mathfrak{S}_3$** are the trivial $U$, the alternating $U'$, and the standard representation $V$. Moreover, for an arbitrary representation $W$ of $\mathfrak{S}_3$, we can write

$$W = U^{\oplus a} \oplus U'^{\oplus b} \oplus V^{\oplus c},$$

where the multiplicities $a$, $b$, $c$ can be determined from the eigenvalues of $\tau$ and $\sigma$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tensor Decomposition: $V \otimes V$ for $\mathfrak{S}_3$)</span></p>

Let $V$ be the standard two-dimensional representation with basis $\alpha, \beta$ such that $\tau\alpha = \omega\alpha$, $\tau\beta = \omega^2\beta$, $\sigma\alpha = \beta$, $\sigma\beta = \alpha$. For $V \otimes V$, the basis vectors $\alpha \otimes \alpha$, $\alpha \otimes \beta$, $\beta \otimes \alpha$, $\beta \otimes \beta$ are eigenvectors for $\tau$ with eigenvalues $\omega^2$, $1$, $1$, and $\omega$, respectively, and $\sigma$ interchanges $\alpha \otimes \alpha$ with $\beta \otimes \beta$, and $\alpha \otimes \beta$ with $\beta \otimes \alpha$. Thus $\alpha \otimes \beta + \beta \otimes \alpha$ spans a trivial representation $U$, and $\alpha \otimes \beta - \beta \otimes \alpha$ spans $U'$, so

$$V \otimes V \cong U \oplus U' \oplus V.$$

</div>

---

## Lecture 2: Characters

This lecture contains the heart of the representation theory of finite groups: the definition of the **character** of a representation, and the main theorem that the characters of the irreducible representations form an orthonormal basis for the space of class functions on $G$.

### 2.1 Characters

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Character)</span></p>

If $V$ is a representation of $G$, its **character** $\chi_V$ is the complex-valued function on the group defined by

$$\chi_V(g) = \mathrm{Tr}(g\vert_V),$$

the trace of $g$ on $V$. In particular, $\chi_V(hgh^{-1}) = \chi_V(g)$, so $\chi_V$ is constant on the conjugacy classes of $G$; such a function is called a **class function**. Note that $\chi_V(1) = \dim V$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.1 — Characters of Constructions)</span></p>

Let $V$ and $W$ be representations of $G$. Then

$$\chi_{V \oplus W} = \chi_V + \chi_W, \qquad \chi_{V \otimes W} = \chi_V \cdot \chi_W,$$

$$\chi_{V^*} = \overline{\chi}_V \quad \text{and} \quad \chi_{\wedge^2 V}(g) = \tfrac{1}{2}[\chi_V(g)^2 - \chi_V(g^2)].$$

**Proof.** We compute the values of these characters on a fixed element $g \in G$. For the action of $g$, $V$ has eigenvalues $\lbrace \lambda_i \rbrace$ and $W$ has eigenvalues $\lbrace \mu_j \rbrace$. Then $\lbrace \lambda_i \rbrace \cup \lbrace \mu_j \rbrace$ and $\lbrace \lambda_i \cdot \mu_j \rbrace$ are eigenvalues for $V \oplus W$ and $V \otimes W$, from which the first two formulas follow. Similarly, $\lbrace \lambda_i^{-1} = \bar{\lambda}_i \rbrace$ are the eigenvalues for $g$ on $V^*$, since all eigenvalues are $n$th roots of unity, with $n$ the order of $g$. Finally, $\lbrace \lambda_i \lambda_j \mid i < j \rbrace$ are the eigenvalues for $g$ on $\bigwedge^2 V$, and

$$\sum_{i < j} \lambda_i \lambda_j = \frac{(\sum \lambda_i)^2 - \sum \lambda_i^2}{2},$$

and since $g^2$ has eigenvalues $\lbrace \lambda_i^2 \rbrace$, the last formula follows. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(2.6 — Character Table of $\mathfrak{S}_3$)</span></p>

The conjugacy classes of $\mathfrak{S}_3$ are $[1]$, $[(12)]$, and $[(123)]$, with 1, 3, and 2 elements respectively. The trivial representation takes values $(1, 1, 1)$; the alternating representation has values $(1, -1, 1)$. The character of the standard representation is found from the permutation representation: since $\mathbb{C}^3 = U \oplus V$, we have $\chi_V = \chi_{\mathbb{C}^3} - \chi_U = (3, 1, 0) - (1, 1, 1) = (2, 0, -1)$. The character table is:

| $\mathfrak{S}_3$ | $1$ | $(12)$ | $(123)$ |
|---|---|---|---|
| | 1 | 3 | 2 |
| trivial $U$ | 1 | 1 | 1 |
| alternating $U'$ | 1 | $-1$ | 1 |
| standard $V$ | 2 | 0 | $-1$ |

A representation $W$ is **determined up to isomorphism by its character** $\chi_W$: if $W \cong U^{\oplus a} \oplus U'^{\oplus b} \oplus V^{\oplus c}$, then $\chi_W = a\chi_U + b\chi_{U'} + c\chi_V$.

</div>

### 2.2 The First Projection Formula and Its Consequences

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Invariant Subspace $V^G$)</span></p>

For any representation $V$ of a group $G$, set

$$V^G = \lbrace v \in V \colon gv = v \quad \forall g \in G \rbrace.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.8 — Projection onto $V^G$)</span></p>

The map

$$\varphi = \frac{1}{\lvert G \rvert} \sum_{g \in G} g \in \mathrm{End}(V)$$

is a projection of $V$ onto $V^G$.

**Proof.** If $v = \varphi(w) = (1/\lvert G \rvert)\sum gw$, then for any $h \in G$, $hv = (1/\lvert G \rvert)\sum hgw = (1/\lvert G \rvert)\sum gw = v$, so the image of $\varphi$ is contained in $V^G$. Conversely, if $v \in V^G$, then $\varphi(v) = (1/\lvert G \rvert)\sum v = v$, so $V^G \subset \mathrm{Im}(\varphi)$; and $\varphi \circ \varphi = \varphi$. $\square$

</div>

We have

$$m = \dim V^G = \mathrm{Trace}(\varphi) = \frac{1}{\lvert G \rvert} \sum_{g \in G} \mathrm{Trace}(g) = \frac{1}{\lvert G \rvert} \sum_{g \in G} \chi_V(g). \tag{2.9}$$

In particular, for an irreducible representation $V$ other than the trivial one, the sum over all $g \in G$ of $\chi_V(g)$ is zero.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Inner Product on Class Functions)</span></p>

Let $\mathbb{C}_\mathrm{class}(G) = \lbrace \text{class functions on } G \rbrace$ and define a Hermitian inner product on $\mathbb{C}_\mathrm{class}(G)$ by

$$(\alpha, \beta) = \frac{1}{\lvert G \rvert} \sum_{g \in G} \overline{\alpha(g)}\beta(g). \tag{2.11}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.12 — Orthonormality of Characters)</span></p>

In terms of the inner product $(2.11)$, the characters of the irreducible representations of $G$ are **orthonormal**:

$$\frac{1}{\lvert G \rvert} \sum_{g \in G} \overline{\chi_V(g)}\chi_W(g) = \begin{cases} 1 & \text{if } V \cong W, \\ 0 & \text{if } V \not\cong W. \end{cases} \tag{2.10}$$

**Proof sketch.** The key is to use $\mathrm{Hom}(V, W)^G = \lbrace G\text{-module homomorphisms from } V \text{ to } W \rbrace$. If $V$ is irreducible, then by Schur's lemma $\dim \mathrm{Hom}_G(V, W)$ is the multiplicity of $V$ in $W$; similarly, if $W$ is irreducible, $\dim \mathrm{Hom}_G(V, W)$ is the multiplicity of $W$ in $V$; in the case where both $V$ and $W$ are irreducible, we have $\dim \mathrm{Hom}_G(V, W) = 1$ if $V \cong W$ and $0$ otherwise. The character of $\mathrm{Hom}(V, W) = V^* \otimes W$ is $\overline{\chi_V} \cdot \chi_W$. Applying formula (2.9) to this case gives the result. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(2.13)</span></p>

The number of irreducible representations of $G$ is less than or equal to the number of conjugacy classes.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(2.14 — Determination by Character)</span></p>

Any representation is **determined by its character**.

Indeed, if $V \cong V_1^{\oplus a_1} \oplus \cdots \oplus V_k^{\oplus a_k}$, with the $V_i$ distinct irreducible representations, then $\chi_V = \sum a_i \chi_{V_i}$, and the $\chi_{V_i}$ are linearly independent.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(2.15 — Irreducibility Criterion)</span></p>

A representation $V$ is irreducible if and only if $(\chi_V, \chi_V) = 1$.

In fact, if $V \cong V_1^{\oplus a_1} \oplus \cdots \oplus V_k^{\oplus a_k}$, then $(\chi_V, \chi_V) = \sum a_i^2$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(2.16 — Multiplicity Formula)</span></p>

The multiplicity $a_i$ of $V_i$ in $V$ is the inner product of $\chi_V$ with $\chi_{V_i}$, i.e., $a_i = (\chi_V, \chi_{V_i})$.

</div>

#### The Regular Representation and Consequences

Applying these corollaries to the regular representation $R$ of $G$, whose character is

$$\chi_R(g) = \begin{cases} 0 & \text{if } g \ne e, \\ \lvert G \rvert & \text{if } g = e, \end{cases}$$

we obtain $R = \bigoplus V_i^{\oplus a_i}$, with $V_i$ distinct irreducibles, and

$$a_i = (\chi_{V_i}, \chi_R) = \frac{1}{\lvert G \rvert}\chi_{V_i}(e) \cdot \lvert G \rvert = \dim V_i. \tag{2.17}$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(2.18 — Irreducibles in the Regular Representation)</span></p>

Any irreducible representation $V$ of $G$ appears in the regular representation $\dim V$ times. In particular, there are only finitely many irreducible representations. As a numerical consequence,

$$\lvert G \rvert = \dim(R) = \sum_i (\dim V_i)^2. \tag{2.19}$$

Also, for $g \ne e$,

$$0 = \sum (\dim V_i) \cdot \chi_{V_i}(g). \tag{2.20}$$

These two formulas amount to the Fourier inversion formula for finite groups.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Column Orthogonality)</span></p>

The orthogonality of the rows of the character table is equivalent to an orthogonality for the columns. Written out, this says:

(i) For $g \in G$:

$$\sum_\chi \overline{\chi(g)}\chi(g) = \frac{\lvert G \rvert}{c(g)},$$

where the sum is over all irreducible characters, and $c(g)$ is the number of elements in the conjugacy class of $g$.

(ii) If $g$ and $h$ are elements of $G$ that are **not** conjugate, then

$$\sum_\chi \overline{\chi(g)}\chi(h) = 0.$$

</div>

### 2.3 Examples: $\mathfrak{S}_4$ and $\mathfrak{A}_4$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Character Table of $\mathfrak{S}_4$)</span></p>

The conjugacy classes of $\mathfrak{S}_4$ correspond to partitions of 4:

| Class | $1$ | $(12)$ | $(123)$ | $(1234)$ | $(12)(34)$ |
|---|---|---|---|---|---|
| Size | 1 | 6 | 8 | 6 | 3 |

We start with the same representations as for $\mathfrak{S}_3$: the trivial $U$, the alternating $U'$, and the standard representation $V$ (the quotient of the permutation representation on $\mathbb{C}^4$ by the trivial subrepresentation). The character of the permutation representation on $\mathbb{C}^4$ is $\chi_{\mathbb{C}^4} = (4, 2, 1, 0, 0)$, and correspondingly $\chi_V = (3, 1, 0, -1, -1)$. Note that $\lvert\chi_V\rvert = 1$, so $V$ is irreducible.

Since $1 + 1 + 9 = 11$ but $\lvert \mathfrak{S}_4 \rvert = 24$, there must be additional irreducible representations with dimensions squaring to $24 - 11 = 13$. By Corollary 2.13 there are at most two more (of dimensions 2 and 3). Tensoring the standard representation $V$ with the alternating one $U'$ gives a representation $V'$ with character $\chi_{V'} = \chi_V \cdot \chi_{U'} = (3, -1, 0, 1, -1)$, which is irreducible. As for the remaining representation of degree two, call it $W$; we can determine its character from the orthogonality relations. The complete character table for $\mathfrak{S}_4$ is:

| $\mathfrak{S}_4$ | $1$ | $(12)$ | $(123)$ | $(1234)$ | $(12)(34)$ |
|---|---|---|---|---|---|
| | 1 | 6 | 8 | 6 | 3 |
| trivial $U$ | 1 | 1 | 1 | 1 | 1 |
| alternating $U'$ | 1 | $-1$ | 1 | $-1$ | 1 |
| standard $V$ | 3 | 1 | 0 | $-1$ | $-1$ |
| $V' = V \otimes U'$ | 3 | $-1$ | 0 | 1 | $-1$ |
| $W$ | 2 | 0 | $-1$ | 0 | 2 |

The key to identifying $W$: the 2 in the last column for $\chi_W$ says that the action of $(12)(34)$ on the two-dimensional vector space $W$ is an involution of trace 2, and so must be the identity. Thus $W$ is really a representation of the quotient group $\mathfrak{S}_4/\lbrace 1, (12)(34), (13)(24), (14)(23) \rbrace \cong \mathfrak{S}_3$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(2.24 — $\mathfrak{S}_4$ as Symmetries of the Cube)</span></p>

$\mathfrak{S}_4$ is the group of rigid motions of a cube, acting on the four long diagonals. It follows that $\mathfrak{S}_4$ acts as well on the set of faces, of edges, of vertices, etc.; and to each of these is associated a permutation representation of $\mathfrak{S}_4$.

For the **faces** of the cube, the permutation character $\chi$ takes values: $\chi(1) = 6$, $\chi(12) = 0$ (rotation by $180°$ about edge-midpoints fixes no faces), $\chi(123) = 0$ (rotation by $120°$ about a long diagonal fixes no faces), $\chi(1234) = 2$ (rotation by $90°$ about a face-center fixes 2 faces), $\chi((12)(34)) = 2$ (rotation by $180°$ about a face-center fixes 2 faces). Now $(\chi, \chi) = 3$, so $\chi$ is the sum of three distinct irreducible representations. From the table, $(\chi, \chi_U) = (\chi, \chi_{V'}) = (\chi, \chi_W) = 1$, and the inner products with the others are zero, so this representation is $U \oplus V' \oplus W$. In fact, the sums of opposite faces span a three-dimensional subrepresentation which contains $U$ (spanned by the sum of all faces), so this representation is $U \oplus W$. The differences of opposite faces therefore span $V'$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(2.26 — Character Table of $\mathfrak{A}_4$)</span></p>

The alternating group $\mathfrak{A}_4$ has four conjugacy classes. Three representations $U$, $U'$, and $U''$ come from the representations of

$$\mathfrak{A}_4/\lbrace 1, (12)(34), (13)(24), (14)(23) \rbrace \cong \mathbb{Z}/3,$$

so there is one more irreducible representation $V$ of dimension 3. The character table, with $\omega = e^{2\pi i/3}$:

| $\mathfrak{A}_4$ | $1$ | $(123)$ | $(132)$ | $(12)(34)$ |
|---|---|---|---|---|
| | 1 | 4 | 4 | 3 |
| $U$ | 1 | 1 | 1 | 1 |
| $U'$ | 1 | $\omega$ | $\omega^2$ | 1 |
| $U''$ | 1 | $\omega^2$ | $\omega$ | 1 |
| $V$ | 3 | 0 | 0 | $-1$ |

</div>

### 2.4 More Projection Formulas; More Consequences

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.28 — Class Functions and $G$-linear Endomorphisms)</span></p>

Let $\alpha\colon G \to \mathbb{C}$ be any function on the group $G$, and for any representation $V$ of $G$ set

$$\varphi_{\alpha, V} = \sum_{g} \alpha(g) \cdot g \colon V \to V.$$

Then $\varphi_{\alpha, V}$ is a homomorphism of $G$-modules for all $V$ if and only if $\alpha$ is a **class function**.

**Proof.** We write out the condition that $\varphi_{\alpha, V}$ be $G$-linear:

$$\varphi_{\alpha, V}(hv) = \sum \alpha(g) \cdot g(hv) = \sum \alpha(hgh^{-1}) \cdot hg(v) = h(\sum \alpha(hgh^{-1}) \cdot g(v)),$$

which equals $h(\varphi_{\alpha, V}(v)) = h(\sum \alpha(g) \cdot g(v))$ if and only if $\alpha$ is a class function. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.30 — Number of Irreducibles Equals Number of Conjugacy Classes)</span></p>

The number of irreducible representations of $G$ is equal to the number of conjugacy classes of $G$. Equivalently, their characters $\lbrace \chi_V \rbrace$ form an **orthonormal basis** for $\mathbb{C}_\mathrm{class}(G)$.

**Proof.** Suppose $\alpha\colon G \to \mathbb{C}$ is a class function and $(\alpha, \chi_V) = 0$ for all irreducible representations $V$; we must show that $\alpha = 0$. Consider the endomorphism $\varphi_{\alpha, V} = \sum \alpha(g) \cdot g\colon V \to V$ as defined above. By Proposition 2.28, $\varphi_{\alpha, V}$ is a $G$-module homomorphism. Hence, if $V$ is irreducible, by Schur's lemma $\varphi_{\alpha, V} = \lambda \cdot \mathrm{Id}$, and then

$$\lambda = \frac{1}{n}\mathrm{trace}(\varphi_{\alpha, V}) = \frac{1}{n}\sum \alpha(g)\chi_V(g) = \frac{\lvert G \rvert}{n}(\alpha, \chi_{V^*}) = 0.$$

Thus $\varphi_{\alpha, V} = 0$ or $\sum \alpha(g) \cdot g = 0$ on any representation $V$ of $G$; in particular, this is true for the regular representation $V = R$. But in $R$ the elements $\lbrace g(e) \rbrace$, thought of as elements of $\mathrm{End}(R)$, are linearly independent. Thus $\alpha(g) = 0$ for all $g$, as required. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Representation Ring)</span></p>

The **representation ring** $R(G)$ of a group $G$ is the free abelian group generated by all (isomorphism classes of) representations of $G$, modded out by the subgroup generated by elements of the form $V + W - (V \oplus W)$. Equivalently, given complete reducibility, we can take all integral linear combinations $\sum a_i \cdot V_i$ of the irreducible representations $V_i$ of $G$; elements of $R(G)$ are correspondingly called **virtual representations**. The ring structure is given by tensor product.

The character defines a map $\chi\colon R(G) \to \mathbb{C}_\mathrm{class}(G)$ which is in fact a ring homomorphism. The statement that a representation is determined by its character says that $\chi$ is injective; Proposition 2.30 says that $\chi$ induces an isomorphism $\chi_\mathbb{C}\colon R(G) \otimes \mathbb{C} \to \mathbb{C}_\mathrm{class}(G)$.

</div>

#### The General Projection Formula

The argument for Proposition 2.30 also suggests a more general projection formula. If $W$ is a fixed irreducible representation, then for any representation $V$, look at the weighted sum

$$\psi = \frac{1}{\lvert G \rvert} \sum_{g \in G} \overline{\chi_W(g)} \cdot g \in \mathrm{End}(V).$$

By Proposition 2.28, $\psi$ is a $G$-module homomorphism. If $V$ is irreducible, by Schur's lemma $\psi = \lambda \cdot \mathrm{Id}$, and

$$\lambda = \frac{1}{\dim V} \cdot \frac{1}{\lvert G \rvert} \sum \overline{\chi_W(g)} \cdot \chi_V(g) = \begin{cases} \frac{1}{\dim V} & \text{if } V = W, \\ 0 & \text{if } V \ne W. \end{cases}$$

For arbitrary $V$,

$$\psi_V = \dim W \cdot \frac{1}{\lvert G \rvert} \sum_{g \in G} \overline{\chi_W(g)} \cdot g \colon V \to V \tag{2.31}$$

is the projection of $V$ onto the factor consisting of the sum of all copies of $W$ appearing in $V$. In other words, if $V = \bigoplus V_i^{\oplus a_i}$, then

$$\pi_i = \dim V_i \cdot \frac{1}{\lvert G \rvert} \sum_{g \in G} \overline{\chi_{V_i}(g)} \cdot g \tag{2.32}$$

is the projection of $V$ onto $V_i^{\oplus a_i}$.

---

## Lecture 3: Examples; Induced Representations; Group Algebras; Real Representations

This lecture is a grabbag. We start with examples illustrating the use of the techniques of the preceding lecture ($\mathfrak{S}_5$, $\mathfrak{A}_5$), then discuss exterior powers of the standard representation of $\mathfrak{S}_d$, introduce induced representations and the group algebra, and finally classify real representations.

### 3.1 Examples: $\mathfrak{S}_5$ and $\mathfrak{A}_5$

#### Representations of $\mathfrak{S}_5$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Character Table of $\mathfrak{S}_5$)</span></p>

The conjugacy classes of $\mathfrak{S}_5$ and their sizes are:

| $\mathfrak{S}_5$ | $1$ | $(12)$ | $(123)$ | $(1234)$ | $(12345)$ | $(12)(34)$ | $(12)(345)$ |
|---|---|---|---|---|---|---|---|
| Size | 1 | 10 | 20 | 30 | 24 | 15 | 20 |

We start with the trivial representation $U$, the alternating representation $U'$, the standard representation $V$ (quotient of the permutation representation on $\mathbb{C}^5$ by the trivial sub), and $V' = V \otimes U'$. The first four rows are:

| | $1$ | $(12)$ | $(123)$ | $(1234)$ | $(12345)$ | $(12)(34)$ | $(12)(345)$ |
|---|---|---|---|---|---|---|---|
| $U$ | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| $U'$ | 1 | $-1$ | 1 | $-1$ | 1 | 1 | $-1$ |
| $V$ | 4 | 2 | 1 | 0 | $-1$ | 0 | $-1$ |
| $V'$ | 4 | $-2$ | 1 | 0 | $-1$ | 0 | 1 |

We need three more irreducible representations. By the formula $\chi_{\wedge^2 V}(g) = \frac{1}{2}(\chi_V(g)^2 - \chi_V(g^2))$, the character of $\bigwedge^2 V$ is $(6, 0, 0, 0, 1, -2, 0)$. One checks $(\chi, \chi) = 1$, so $\bigwedge^2 V$ is a fifth irreducible representation. (Also $\bigwedge^2 V \otimes U' = \bigwedge^2 V$, so we get nothing new that way.)

Since $1^2 + 1^2 + 4^2 + 4^2 + 6^2 = 70$ and $5! = 120$, we need $n_1^2 + n_2^2 = 50$. The only possibility is $n_1 = n_2 = 5$. Let $W$ denote one of these five-dimensional representations, and set $W' = W \otimes U'$. Using the orthogonality relations or (2.20), the last two rows are determined up to interchanging $W$ and $W'$:

| | $1$ | $(12)$ | $(123)$ | $(1234)$ | $(12345)$ | $(12)(34)$ | $(12)(345)$ |
|---|---|---|---|---|---|---|---|
| $\bigwedge^2 V$ | 6 | 0 | 0 | 0 | 1 | $-2$ | 0 |
| $W$ | 5 | 1 | $-1$ | $-1$ | 0 | 1 | 1 |
| $W'$ | 5 | $-1$ | $-1$ | 1 | 0 | 1 | $-1$ |

From the decomposition $V \oplus U = \mathbb{C}^5$, we also have $\bigwedge^4 V = \bigwedge^4(\mathbb{C}^5/U) \cong \bigwedge^5 \mathbb{C}^5 / \bigwedge^4 \mathbb{C}^5 = U'$, and $V^* = V$. The perfect pairing $V \times \bigwedge^3 V \to \bigwedge^4 V = U'$ shows that $\bigwedge^3 V$ is isomorphic to $V'$.

</div>

#### Representations of the Alternating Group $\mathfrak{A}_5$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Character Table of $\mathfrak{A}_5$)</span></p>

When we replace $\mathfrak{S}_5$ by $\mathfrak{A}_5$, all odd conjugacy classes disappear. The class of five-cycles breaks into two conjugacy classes: $(12345)$ and $(21345)$, each having 12 elements.

The restrictions of $U$, $V$, and $W$ from $\mathfrak{S}_5$ stay irreducible (since their characters satisfy $(\chi, \chi) = 1$). Restricting $U$ and $U'$ both give the trivial representation, $V$ and $V'$ both restrict to $V$, and $W$ and $W'$ both restrict to $W$. The character of $\bigwedge^2 V$ has values $(6, 0, -2, 1, 1)$ on the conjugacy classes, so $(\chi, \chi) = 2$ and $\bigwedge^2 V$ is the sum of two irreducible representations $Y$ and $Z$, each of dimension 3.

Since the sums of the squares of all dimensions is $60$: $1 + 16 + 25 + 9 + 9 = 60 = \lvert\mathfrak{A}_5\rvert$, these are all irreducible representations. The character table, with $\omega = e^{2\pi i / 5}$:

| $\mathfrak{A}_5$ | $1$ | $(123)$ | $(12)(34)$ | $(12345)$ | $(21345)$ |
|---|---|---|---|---|---|
| Size | 1 | 20 | 15 | 12 | 12 |
| $U$ | 1 | 1 | 1 | 1 | 1 |
| $V$ | 4 | 1 | 0 | $-1$ | $-1$ |
| $W$ | 5 | $-1$ | 1 | 0 | 0 |
| $Y$ | 3 | 0 | $-1$ | $\frac{1+\sqrt{5}}{2}$ | $\frac{1-\sqrt{5}}{2}$ |
| $Z$ | 3 | 0 | $-1$ | $\frac{1-\sqrt{5}}{2}$ | $\frac{1+\sqrt{5}}{2}$ |

The representations $Y$ and $Z$ may be familiar: $\mathfrak{A}_5$ can be realized as the group of motions of an icosahedron (or equivalently, of a dodecahedron) and $Y$ is the corresponding representation. Note that $Y$ and $Z$ differ only on the conjugacy classes of $(12345)$ and $(21345)$, and correspond to the same image in $\mathrm{GL}_3(\mathbb{R})$ under the two representations $\mathfrak{A}_5 \to \mathrm{GL}_3(\mathbb{R})$, but they differ by an outer automorphism of $\mathfrak{A}_5$.

Note also that $\bigwedge^2 V$ does not decompose over $\mathbb{Q}$; this reflects the fact that the vertices of a dodecahedron cannot all have rational coordinates.

</div>

### 3.2 Exterior Powers of the Standard Representation of $\mathfrak{S}_d$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.12 — Irreducibility of Exterior Powers)</span></p>

Each exterior power $\bigwedge^k V$ of the standard representation $V$ of $\mathfrak{S}_d$ is irreducible, $0 \le k \le d - 1$.

**Proof.** From the decomposition $\mathbb{C}^d = V \oplus U$, we have

$$\bigwedge^k \mathbb{C}^d = (\bigwedge^k V \otimes \bigwedge^0 U) \oplus (\bigwedge^{k-1} V \otimes \bigwedge^1 U) = \bigwedge^k V \oplus \bigwedge^{k-1} V,$$

so it suffices to show that $(\chi, \chi) = 2$, where $\chi$ is the character of the representation $\bigwedge^k \mathbb{C}^d$. Let $A = \lbrace 1, 2, \ldots, d \rbrace$. For a subset $B$ of $A$ with $k$ elements, and $g \in G = \mathfrak{S}_d$, let

$$\lbrace g \rbrace_B = \begin{cases} 0 & \text{if } g(B) \ne B, \\ 1 & \text{if } g(B) = B \text{ and } g\vert_B \text{ is an even permutation}, \\ -1 & \text{if } g(B) = B \text{ and } g\vert_B \text{ is odd}. \end{cases}$$

Then $\chi(g) = \sum_B \lbrace g \rbrace_B$. Computing $(\chi, \chi)$:

$$(\chi, \chi) = \frac{1}{d!} \sum_{g \in G} \Bigl(\sum_B \lbrace g \rbrace_B\Bigr)^2 = \frac{1}{d!} \sum_B \sum_C \sum_g (\mathrm{sgn}\, g\vert_B)(\mathrm{sgn}\, g\vert_C),$$

where the last sum is over $g$ with $g(B) = B$ and $g(C) = C$. The sum over $g$ is nonzero only unless $k - l = 0$ or $1$, where $l = \lvert B \cap C \rvert$. The case $k = l$ gives 1, and the terms with $k - l = 1$ also add up to 1, so $(\chi, \chi) = 2$, as required. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Symmetric Powers Are Not Generally Irreducible)</span></p>

By way of contrast, the symmetric powers $\mathrm{Sym}^k V$ of the standard representation of $\mathfrak{S}_d$ are almost never irreducible. For example, $\mathrm{Sym}^2 V$ always contains one copy of the trivial representation (since every irreducible real representation, such as $V$, admits a unique inner product, up to scalars, invariant under the group action).

</div>

### 3.3 Induced Representations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Induced Representation)</span></p>

Let $H \subset G$ be a subgroup, and let $W$ be a representation of $H$. If $V$ is a representation of $G$, and $W \subset V$ is a subspace which is $H$-invariant, we say that $V$ is **induced** by $W$ if every element in $V$ can be written uniquely as a sum of elements in translates of $W$, i.e.,

$$V = \bigoplus_{\sigma \in G/H} \sigma \cdot W.$$

In this case we write $V = \mathrm{Ind}_H^G\, W = \mathrm{Ind}\, W$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(3.13 — Permutation Representation as Induced)</span></p>

The permutation representation associated to the left action of $G$ on $G/H$ is induced from the trivial one-dimensional representation $W$ of $H$. Here $V$ has basis $\lbrace e_\sigma \colon \sigma \in G/H \rbrace$, and $W = \mathbb{C} \cdot e_H$, with $H$ acting trivially on the coset $H$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(3.14 — Regular Representation as Induced)</span></p>

The regular representation of $G$ is induced from the regular representation of $H$. Here $V$ has basis $\lbrace e_g \colon g \in G \rbrace$, whereas $W$ has basis $\lbrace e_h \colon h \in H \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Construction of Induced Representations)</span></p>

Given a representation $W$ of $H$, we can construct $V = \mathrm{Ind}(W)$ explicitly. Choose a representative $g_\sigma \in G$ for each coset $\sigma \in G/H$, with $e$ representing the trivial coset $H$. Each element of $V$ has a unique expression $v = \sum g_\sigma w_\sigma$ for elements $w_\sigma$ in $W$. Take a copy $W^\sigma$ of $W$ for each left coset $\sigma \in G/H$; for $w \in W$, let $g_\sigma w$ denote the element of $W^\sigma$ corresponding to $w$ in $W$. Let $V = \bigoplus_{\sigma \in G/H} W^\sigma$. Given $g \in G$, define

$$g \cdot (g_\sigma w_\sigma) = g_\tau(hw_\sigma) \quad \text{if } g \cdot g_\sigma = g_\tau \cdot h.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Character of Induced Representation)</span></p>

To compute the character of $V = \mathrm{Ind}\, W$, note that $g \in G$ maps $\sigma W$ to $g\sigma W$, so the trace is calculated from those cosets $\sigma$ with $g\sigma = \sigma$, i.e., $s^{-1}gs \in H$ for $s \in \sigma$. Therefore,

$$\chi_{\mathrm{Ind}\, W}(g) = \sum_{\substack{g\sigma = \sigma}} \chi_W(s^{-1}gs) \qquad (s \in \sigma \text{ arbitrary}). \tag{3.18}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.17 — Frobenius Reciprocity, Universal Property Form)</span></p>

Let $W$ be a representation of $H$, $U$ a representation of $G$, and suppose $V = \mathrm{Ind}\, W$. Then any $H$-module homomorphism $\varphi\colon W \to U$ extends uniquely to a $G$-module homomorphism $\tilde{\varphi}\colon V \to U$. That is,

$$\mathrm{Hom}_H(W,\, \mathrm{Res}\, U) = \mathrm{Hom}_G(\mathrm{Ind}\, W,\, U).$$

In particular, this universal property determines $\mathrm{Ind}\, W$ up to canonical isomorphism.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.20 — Frobenius Reciprocity)</span></p>

If $W$ is a representation of $H$, and $U$ a representation of $G$, then

$$(\chi_{\mathrm{Ind}\, W},\, \chi_U)_G = (\chi_W,\, \chi_{\mathrm{Res}\, U})_H.$$

In particular, if both $W$ and $U$ are irreducible, Frobenius reciprocity says: *the number of times $U$ appears in $\mathrm{Ind}\, W$ is the same as the number of times $W$ appears in $\mathrm{Res}\, U$.*

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(3.21 — Induction from $\mathfrak{S}_2 \subset \mathfrak{S}_3$)</span></p>

Let $H = \mathfrak{S}_2 \subset G = \mathfrak{S}_3$, $W = V_2 = U_2'$ (the alternating representation of $\mathfrak{S}_2$). The irreducible representations of $\mathfrak{S}_3$ restrict to: $U_3 \to U_2$, $U_3' \to U_2'$, $V_3 \to U_2 \oplus U_2'$. By Frobenius reciprocity, $\mathrm{Ind}\, V_2 = U_3' \oplus V_3$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(3.22 — Induction from $\mathfrak{S}_3 \subset \mathfrak{S}_4$)</span></p>

Let $H = \mathfrak{S}_3 \subset G = \mathfrak{S}_4$, $W = V_3$ (the standard representation of $\mathfrak{S}_3$). We know the irreducible representations of $\mathfrak{S}_4$ and their restrictions: $\mathrm{Res}\, U_4 = U_3$, $\mathrm{Res}\, U_4' = U_3'$, $\mathrm{Res}\, V_4' = U_3' \oplus V_3$, $\mathrm{Res}\, V_4 = U_3 \oplus V_3$, $\mathrm{Res}\, W_4 = V_3$. The vector $(1, 1, 1, -3) \in V_4 = \lbrace (x_1, x_2, x_3, x_4) \colon \sum x_i = 0 \rbrace$ is fixed by $H$, so $\mathrm{Res}\, V_4 = U_3 \oplus V_3$, and $\mathrm{Res}\, W_4 = V_3$ (as one may see directly). Hence, by Frobenius, $\mathrm{Ind}\, V_3 = V_4 \oplus V_4' \oplus W_4$.

</div>

#### Artin and Brauer Theorems

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.27 — Artin's Theorem)</span></p>

The characters of induced representations from **cyclic** subgroups of $G$ generate a lattice of finite index in $\Lambda$ (the image of $R(G)$ in $\mathbb{C}_\mathrm{class}(G)$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.28 — Brauer's Theorem)</span></p>

The characters of induced representations from **elementary** subgroups of $G$ generate the lattice $\Lambda$ itself.

(A subgroup $H$ of $G$ is *$p$-elementary* if $H = A \times B$, with $A$ cyclic of order prime to $p$ and $B$ a $p$-group.)

</div>

### 3.4 The Group Algebra

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Group Algebra)</span></p>

The **group algebra** $\mathbb{C}G$ of a finite group $G$ is the vector space with basis $\lbrace e_g \rbrace$ corresponding to elements of the group $G$ (the underlying vector space of the regular representation), with the algebra structure defined by

$$e_g \cdot e_h = e_{gh}.$$

A representation of the algebra $\mathbb{C}G$ on a vector space $V$ is simply an algebra homomorphism $\mathbb{C}G \to \mathrm{End}(V)$. Thus a representation $V$ of $\mathbb{C}G$ is the same thing as a left $\mathbb{C}G$-module. Note that a representation $\rho\colon G \to \mathrm{Aut}(V)$ will extend by linearity to a map $\tilde{\rho}\colon \mathbb{C}G \to \mathrm{End}(V)$, so that representations of $\mathbb{C}G$ correspond exactly to representations of $G$; the left $\mathbb{C}G$-module given by $\mathbb{C}G$ itself corresponds to the regular representation.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.29 — Artin–Wedderburn Decomposition)</span></p>

As algebras,

$$\mathbb{C}G \cong \bigoplus_i \mathrm{End}(W_i),$$

where $\lbrace W_i \rbrace$ are the irreducible representations of $G$.

**Proof.** For any representation $W$ of $G$, the map $G \to \mathrm{Aut}(W)$ extends by linearity to a map $\mathbb{C}G \to \mathrm{End}(W)$; applying this to each of the irreducible representations $W_i$ gives a canonical map

$$\varphi\colon \mathbb{C}G \to \bigoplus \mathrm{End}(W_i).$$

This is injective since the representation on the regular representation is faithful. Since both sides have dimension $\sum (\dim W_i)^2 = \lvert G \rvert$, the map is an isomorphism. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequences of Artin–Wedderburn)</span></p>

Several consequences follow:

- The isomorphism $\varphi$ can be interpreted as the **Fourier transform** (cf. Exercise 3.32).
- Proposition 2.28 has a natural interpretation: the center of $\mathbb{C}G$ consists of those $\sum \alpha(g)e_g$ for which $\alpha$ is a class function.
- The decomposition implies that the matrix entries of the irreducible representations give a basis for the space of **all** functions on $G$ (cf. Exercise 2.35).
- Any irreducible representation is isomorphic to a (minimal) left ideal in $\mathbb{C}G$. These left ideals are generated by idempotents. In fact, the projection formulas of Lecture 2 say that

$$\dim W \cdot \frac{1}{\lvert G \rvert} \sum_{g \in G} \overline{\chi_W(g)} \cdot e_g \in \mathbb{C}G$$

are the idempotents corresponding to the direct sum factors in the decomposition.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Induced Representations via the Group Algebra)</span></p>

The group algebra also gives another description of induced representations: if $W$ is a representation of a subgroup $H$ of $G$, then the induced representation may be constructed simply by

$$\mathrm{Ind}\, W = \mathbb{C}G \otimes_{\mathbb{C}H} W,$$

where $G$ acts on the first factor: $g \cdot (e_{g'} \otimes w) = e_{gg'} \otimes w$. The Frobenius reciprocity theorem is then a special case of a general formula for a change of rings $\mathbb{C}H \to \mathbb{C}G$:

$$\mathrm{Hom}_{\mathbb{C}H}(W,\, U) = \mathrm{Hom}_{\mathbb{C}G}(\mathbb{C}G \otimes_{\mathbb{C}H} W,\, U).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convolution and the Fourier Transform)</span></p>

If $\mathbb{C}G$ is identified with the space of functions on $G$, the function $\varphi$ corresponding to $\sum \varphi(g)e_g$, then the product in $\mathbb{C}G$ corresponds to the **convolution** of functions:

$$(\varphi * \psi)(g) = \sum_{h \in G} \varphi(h)\psi(h^{-1}g).$$

If $\rho\colon G \to \mathrm{GL}(V_\rho)$ is a representation and $\varphi$ is a function on $G$, the **Fourier transform** $\hat{\varphi}(\rho) \in \mathrm{End}(V_\rho)$ is defined by

$$\hat{\varphi}(\rho) = \sum_{g \in G} \varphi(g) \cdot \rho(g).$$

The **Fourier inversion formula** is then

$$\varphi(g) = \frac{1}{\lvert G \rvert} \sum_\rho \dim(V_\rho) \cdot \mathrm{Trace}(\rho(g^{-1}) \cdot \hat{\varphi}(\rho)),$$

the sum over the irreducible representations $\rho$ of $G$. This formula is equivalent to formulas (2.19) and (2.20). The **Plancherel formula** is:

$$\sum_{g \in G} \varphi(g^{-1})\psi(g) = \frac{1}{\lvert G \rvert} \sum_\rho \dim(V_\rho) \cdot \mathrm{Trace}(\hat{\varphi}(\rho) \hat{\psi}(\rho)).$$

</div>

### 3.5 Real Representations and Representations over Subfields of $\mathbb{C}$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Real Representation)</span></p>

If a group $G$ acts on a real vector space $V_0$, then we say the corresponding complex representation $V = V_0 \otimes_\mathbb{R} \mathbb{C}$ is **real**. To the extent that we are interested in the action of a finite group $G$ on real rather than complex vector spaces, the problem is to classify which of the complex representations of $G$ we have studied are in fact real.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Real Character Does Not Imply Real Representation)</span></p>

A first guess might be that a representation is real if and only if its character is real-valued. This turns out not to be the case: the character of a real representation is certainly real-valued, but the converse need not be true.

For example, suppose $G \subset \mathrm{SU}(2)$ is a finite, nonabelian subgroup. Then $G$ acts on $\mathbb{C}^2 = V$ with a real-valued character (since the trace of any matrix in $\mathrm{SU}(2)$ is real). If $V$ were a real representation, however, then $G$ would be a subgroup of $\mathrm{SO}(2) = S^1$, which is abelian. So $V$ is not real, despite having a real character.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(3.35 — Criterion for Real Representations)</span></p>

An irreducible representation $V$ of $G$ is real if and only if there is a nondegenerate **symmetric** bilinear form $B$ on $V$ preserved by $G$.

**Proof.** If $V$ is a real representation coming from $V_0$, then one can find a positive definite symmetric bilinear form on $V_0$ which is preserved by $G$ (by averaging). This gives a symmetric bilinear form on $V$.

Conversely, if we have such a $B$, and an arbitrary nondegenerate Hermitian form $H$ (also $G$-invariant), then $V \xrightarrow{B} V^* \xrightarrow{H} V$ gives a conjugate linear isomorphism $\varphi$ from $V$ to $V$ with $B(x, y) = H(\varphi(x), y)$, and $\varphi$ commutes with the action of $G$. Then $\varphi^2 = \varphi \circ \varphi$ is a complex linear $G$-module homomorphism, so $\varphi^2 = \lambda \cdot \mathrm{Id}$. From $H(\varphi^2(x), y) = H(x, \varphi^2(y))$, it follows that $\lambda$ is a positive real number. Scaling $H$ so that $\lambda = 1$, we get $\varphi^2 = \mathrm{Id}$, so $V$ is the sum of the real eigenspaces $V_+$ and $V_-$ for $\varphi$ (eigenvalues $1$ and $-1$). Since $\varphi$ commutes with $G$ and $\varphi(ix) = -i\varphi(x)$, we have $iV_+ = V_-$, so $V = V_+ \otimes \mathbb{C}$. $\square$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quaternionic Representation)</span></p>

A **(complex) representation** $V$ is **quaternionic** if it has a $G$-invariant homomorphism $J\colon V \to V$ that is conjugate linear, and satisfies $J^2 = -\mathrm{Id}$. Thus, a skew-symmetric nondegenerate $G$-invariant bilinear form $B$ determines a quaternionic structure on $V$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.37 — Classification of Irreducible Representations by Type)</span></p>

An irreducible representation $V$ is one and only one of the following:

1. **Complex:** $\chi_V$ is not real-valued; $V$ does not have a $G$-invariant nondegenerate bilinear form.
2. **Real:** $V = V_0 \otimes \mathbb{C}$, a real representation; $V$ has a $G$-invariant symmetric nondegenerate bilinear form.
3. **Quaternionic:** $\chi_V$ is real, but $V$ is not real; $V$ has a $G$-invariant skew-symmetric nondegenerate bilinear form.

The type is detected by the **Frobenius–Schur indicator**: for $V$ irreducible,

$$\frac{1}{\lvert G \rvert} \sum_{g \in G} \chi_V(g^2) = \begin{cases} 0 & \text{if } V \text{ is complex}, \\ 1 & \text{if } V \text{ is real}, \\ -1 & \text{if } V \text{ is quaternionic}. \end{cases}$$

This verifies that the three cases are mutually exclusive. It also implies that if the order of $G$ is odd, all nontrivial representations must be complex.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Representations over Subfields of $\mathbb{C}$)</span></p>

More generally, let $K \subset \mathbb{C}$ be any subfield. A *$K$-representation* of $G$ is a vector space $V_0$ over $K$ on which $G$ acts; the complex representation $V = V_0 \otimes \mathbb{C}$ is said to be **defined over $K$**. One can introduce the **representation ring** $R_K(G)$ of $G$ over $K$, defined just like the ordinary representation ring but using $K$-representations. A complex representation of $G$ can be defined over $K$ if and only if its character belongs to $R_K(G)$.

</div>

---

## Lecture 4: Representations of $\mathfrak{S}_d$: Young Diagrams and Frobenius's Character Formula

This lecture gives a complete description of the irreducible representations of the symmetric group $\mathfrak{S}_d$, a construction of the representations (via Young symmetrizers), and a formula (Frobenius's formula) for their characters. These results turn out to be of substantial interest in Lie theory as well: analogs of the Young symmetrizers will give a construction of the irreducible representations of $\mathrm{SL}_n\mathbb{C}$.

### 4.1 Statements of the Results

The number of irreducible representations of $\mathfrak{S}_d$ equals the number of conjugacy classes, which is the number $p(d)$ of **partitions** of $d$: $d = \lambda_1 + \cdots + \lambda_k$, $\lambda_1 \ge \cdots \ge \lambda_k \ge 1$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Partition Numbers)</span></p>

The generating function for the partition numbers is

$$\sum_{d=0}^{\infty} p(d)t^d = \prod_{n=1}^{\infty} \frac{1}{1 - t^n} = (1 + t + t^2 + \cdots)(1 + t^2 + t^4 + \cdots)(1 + t^3 + \cdots)\cdots,$$

converging for $\lvert t \rvert < 1$. The partition number $p(d)$ is asymptotically equal to $(1/\alpha d)e^{\beta\sqrt{d}}$, with $\alpha = 4\sqrt{3}$ and $\beta = \pi\sqrt{2/3}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Young Diagram, Conjugate Partition)</span></p>

To a partition $\lambda = (\lambda_1, \ldots, \lambda_k)$ is associated a **Young diagram** (or Ferrers diagram) with $\lambda_i$ boxes in the $i$th row, the rows of boxes lined up on the left.

The **conjugate partition** $\lambda' = (\lambda_1', \ldots, \lambda_l')$ to $\lambda$ is defined by interchanging rows and columns in the Young diagram, i.e., reflecting the diagram in the $45°$ line. Equivalently, $\lambda_i'$ is the number of terms in the partition $\lambda$ that are greater than or equal to $i$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tableau, Row and Column Groups, Young Symmetrizer)</span></p>

A **tableau** on a given Young diagram is a numbering of the boxes by the integers $1, \ldots, d$. Given a tableau, define two subgroups of $\mathfrak{S}_d$:

$$P = P_\lambda = \lbrace g \in \mathfrak{S}_d \colon g \text{ preserves each row} \rbrace,$$

$$Q = Q_\lambda = \lbrace g \in \mathfrak{S}_d \colon g \text{ preserves each column} \rbrace.$$

In the group algebra $\mathbb{C}\mathfrak{S}_d$, we introduce two elements corresponding to these subgroups:

$$a_\lambda = \sum_{g \in P} e_g \quad \text{and} \quad b_\lambda = \sum_{g \in Q} \mathrm{sgn}(g) \cdot e_g. \tag{4.1}$$

The element $a_\lambda$ acts on $V^{\otimes d}$ (for any vector space $V$) by projecting onto $\mathrm{Sym}^{\lambda_1}V \otimes \mathrm{Sym}^{\lambda_2}V \otimes \cdots$ (grouping the factors by rows), while $b_\lambda$ projects onto $\bigwedge^{\mu_1}V \otimes \bigwedge^{\mu_2}V \otimes \cdots$ (where $\mu$ is the conjugate partition).

The **Young symmetrizer** is defined as

$$c_\lambda = a_\lambda \cdot b_\lambda \in \mathbb{C}\mathfrak{S}_d. \tag{4.2}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.3 — Young Symmetrizers Give Irreducible Representations)</span></p>

Some scalar multiple of $c_\lambda$ is idempotent, i.e., $c_\lambda^2 = n_\lambda c_\lambda$, and the image of $c_\lambda$ (by right multiplication on $\mathbb{C}\mathfrak{S}_d$) is an irreducible representation $V_\lambda$ of $\mathfrak{S}_d$. Every irreducible representation of $\mathfrak{S}_d$ can be obtained in this way for a unique partition.

As a corollary, each irreducible representation of $\mathfrak{S}_d$ can be defined over the rational numbers since $c_\lambda$ is in the rational group algebra $\mathbb{Q}\mathfrak{S}_d$. Note also that the theorem gives a direct correspondence between conjugacy classes in $\mathfrak{S}_d$ and irreducible representations of $\mathfrak{S}_d$ — something which has never been achieved for general groups.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(4.5 — Young Diagrams for Small Symmetric Groups)</span></p>

For $\lambda = (d)$, we have $c_{(d)} = a_{(d)} = \sum_{g \in \mathfrak{S}_d} e_g$, and $V_{(d)}$ is the **trivial** representation $U$. For $\lambda = (1, \ldots, 1)$, we have $c_{(1,\ldots,1)} = b_{(1,\ldots,1)} = \sum_{g} \mathrm{sgn}(g) e_g$, and $V_{(1,\ldots,1)}$ is the **alternating** representation $U'$.

The standard representation $V$ corresponds to the partition $d = (d-1) + 1$. The exterior powers $\bigwedge^s V$ of the standard representation correspond to "hook" partitions $\lambda = (d - s, 1, \ldots, 1)$.

The correspondences for $d \le 5$ (matching the character tables computed in earlier lectures) are:

| $\mathfrak{S}_3$ | $(3) \to U$, $(1,1,1) \to U'$, $(2,1) \to V$ |
|---|---|
| $\mathfrak{S}_4$ | $(4) \to U$, $(1^4) \to U'$, $(3,1) \to V$, $(2,1,1) \to V'$, $(2,2) \to W$ |
| $\mathfrak{S}_5$ | $(5) \to U$, $(1^5) \to U'$, $(4,1) \to V$, $(2,1,1,1) \to V'$, $(3,2) \to \bigwedge^2 V$, $(3,1,1) \to W$, $(2,2,1) \to W'$ |

In general, the representation associated to the conjugate partition $\lambda'$ satisfies $V_{\lambda'} = V_\lambda \otimes U'$.

</div>

#### Frobenius's Character Formula

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conjugacy Class Indexing and Power Sums)</span></p>

Let $C_\mathbf{i}$ denote the conjugacy class in $\mathfrak{S}_d$ determined by the sequence

$$\mathbf{i} = (i_1, i_2, \ldots, i_d) \quad \text{with } \sum \alpha i_\alpha = d,$$

consisting of those permutations made up of $i_1$ 1-cycles, $i_2$ 2-cycles, $\ldots$, $i_d$ $d$-cycles. The number of elements in $C_\mathbf{i}$ is

$$\lvert C_\mathbf{i} \rvert = \frac{d!}{1^{i_1}i_1! \cdot 2^{i_2}i_2! \cdots d^{i_d}i_d!}. \tag{4.30}$$

Introduce independent variables $x_1, \ldots, x_k$ (with $k$ at least as large as the number of rows in the Young diagram of $\lambda$). Define the **power sums** $P_j(x) = x_1^j + x_2^j + \cdots + x_k^j$ and the **discriminant** $\Delta(x) = \prod_{i < j}(x_i - x_j)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.10 — Frobenius's Character Formula)</span></p>

Given a partition $\lambda\colon \lambda_1 \ge \cdots \ge \lambda_k \ge 0$ of $d$, set $l_i = \lambda_i + k - i$ (a strictly decreasing sequence of non-negative integers). The character of $V_\lambda$ evaluated on $g \in C_\mathbf{i}$ is given by the remarkable formula

$$\chi_\lambda(C_\mathbf{i}) = \left[\Delta(x) \cdot \prod_j P_j(x)^{i_j}\right]_{(l_1, \ldots, l_k)},$$

where $[f(x)]_{(l_1, \ldots, l_k)}$ denotes the coefficient of $x_1^{l_1} \cdots x_k^{l_k}$ in $f$.

In terms of **Schur polynomials** $S_\lambda$, Frobenius's formula can be expressed as $\prod_j P_j(x)^{i_j} = \sum_\lambda \chi_\lambda(C_\mathbf{i}) S_\lambda$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Dimension via Frobenius's Formula)</span></p>

To compute the dimension of $V_\lambda$, evaluate the character on the identity element, which corresponds to $\mathbf{i} = (d, 0, \ldots, 0)$. Then

$$\dim V_\lambda = \chi_\lambda(C_{(d)}) = \left[\Delta(x) \cdot (x_1 + \cdots + x_k)^d\right]_{(l_1, \ldots, l_k)}.$$

Now $\Delta(x)$ is the Vandermonde determinant $\sum_{\sigma \in \mathfrak{S}_k} (\mathrm{sgn}\,\sigma) x_{\sigma(1)}^{k-1} \cdots x_{\sigma(k)}^0$, and the other factor is $(x_1 + \cdots + x_k)^d = \sum \frac{d!}{r_1! \cdots r_k!} x_1^{r_1} \cdots x_k^{r_k}$. Pairing off corresponding terms and applying column reduction to the Vandermonde determinant yields:

$$\dim V_\lambda = \frac{d!}{l_1! \cdots l_k!} \prod_{i < j}(l_i - l_j), \tag{4.11}$$

with $l_i = \lambda_i + k - i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.12 — Hook Length Formula)</span></p>

The **hook length** of a box in a Young diagram is the number of squares directly below or directly to the right of the box, including the box once. Then

$$\dim V_\lambda = \frac{d!}{\prod(\text{Hook lengths})}.$$

For example, for the partition $4 + 3 + 1$ of $8$, the hook lengths are $6, 4, 3, 1$ (first row), $4, 2, 1$ (second row), $1$ (third row), and the dimension is $8!/(6 \cdot 4 \cdot 3 \cdot 1 \cdot 4 \cdot 2 \cdot 1 \cdot 1) = 70$.

</div>

### 4.2 Irreducible Representations of $\mathfrak{S}_d$

We now prove that the representations $V_\lambda$ constructed in §4.1 are exactly the irreducible representations of $\mathfrak{S}_d$. Let $A = \mathbb{C}\mathfrak{S}_d$ be the group ring. For a partition $\lambda$ of $d$, let $P$ and $Q$ be the row- and column-preserving subgroups, let $a = a_\lambda$, $b = b_\lambda$, $c = c_\lambda = ab$, and let $V_\lambda = Ac_\lambda$ be the corresponding representation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(4.21 — Properties of $a_\lambda$, $b_\lambda$, $c_\lambda$)</span></p>

1. For $p \in P$, $p \cdot a = a \cdot p = a$.
2. For $q \in Q$, $(\mathrm{sgn}(q)q) \cdot b = b \cdot (\mathrm{sgn}(q)q) = b$.
3. For all $p \in P$, $q \in Q$, $p \cdot c \cdot (\mathrm{sgn}(q)q) = c$, and, up to multiplication by a scalar, $c$ is the only such element in $A$.

</div>

We order partitions **lexicographically**: $\lambda > \mu$ if the first nonvanishing $\lambda_i - \mu_i$ is positive.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(4.23 — Orthogonality of Young Symmetrizers)</span></p>

1. If $\lambda > \mu$, then for all $x \in A$, $a_\lambda \cdot x \cdot b_\mu = 0$. In particular, if $\lambda > \mu$, $c_\lambda \cdot c_\mu = 0$.
2. For all $x \in A$, $c_\lambda \cdot x \cdot c_\lambda$ is a scalar multiple of $c_\lambda$. In particular, $c_\lambda \cdot c_\lambda = n_\lambda c_\lambda$ for some $n_\lambda \in \mathbb{C}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(4.25 — $V_\lambda$ Is Irreducible)</span></p>

1. Each $V_\lambda$ is an irreducible representation of $\mathfrak{S}_d$.
2. If $\lambda \ne \mu$, then $V_\lambda$ and $V_\mu$ are not isomorphic.

**Proof of (1).** Note that $c_\lambda V_\lambda \subset \mathbb{C}c_\lambda$ by Lemma 4.23. If $W \subset V_\lambda$ is a subrepresentation, then $c_\lambda W$ is either $\mathbb{C}c_\lambda$ or $0$. If the first, then $V_\lambda = A \cdot c_\lambda \subset A \cdot W = W$. Otherwise $c_\lambda W = 0$; but a projection from $A$ onto $W$ is given by right multiplication by an element $\varphi \in A$ with $\varphi = \varphi^2 \in W \cdot W$, so $c_\lambda \varphi = 0$, which shows $c_\lambda V_\lambda = 0$, a contradiction.

**Proof of (2).** If $\lambda > \mu$, then $c_\lambda V_\mu = c_\lambda \cdot A \cdot c_\mu = 0$ by Lemma 4.23, but $c_\lambda V_\lambda \ne 0$. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(4.26 — Scalar $n_\lambda$)</span></p>

For any $\lambda$, $c_\lambda \cdot c_\lambda = n_\lambda c_\lambda$, with $n_\lambda = d!/\dim V_\lambda$.

**Proof.** Let $F$ be right multiplication by $c_\lambda$ on $A$. Since $F$ is multiplication by $n_\lambda$ on $V_\lambda$, and zero on $\ker(c_\lambda)$, the trace of $F$ is $n_\lambda$ times the dimension of $V_\lambda$. But the coefficient of $e_g$ in $e_g \cdot c_\lambda$ is 1 (for the identity coefficient), so $\mathrm{trace}(F) = \lvert \mathfrak{S}_d \rvert = d!$. $\square$

</div>

Since there are as many $V_\lambda$ as conjugacy classes of $\mathfrak{S}_d$, and they are pairwise non-isomorphic irreducible representations, this completes the proof of Theorem 4.3.

### 4.3 Proof of Frobenius's Formula

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Young Subgroup, Induced Representation $U_\lambda$)</span></p>

For any partition $\lambda$ of $d$, there is a **Young subgroup**

$$\mathfrak{S}_\lambda = \mathfrak{S}_{\lambda_1} \times \cdots \times \mathfrak{S}_{\lambda_k} \hookrightarrow \mathfrak{S}_d. \tag{4.27}$$

Let $U_\lambda$ be the representation of $\mathfrak{S}_d$ induced from the trivial representation of $\mathfrak{S}_\lambda$. Equivalently, $U_\lambda = A \cdot a_\lambda$, with $a_\lambda$ as in (4.1). Let

$$\psi_\lambda = \chi_{U_\lambda} = \text{character of } U_\lambda. \tag{4.28}$$

Note that $V_\lambda$ appears in $U_\lambda$ (since there is a surjection $U_\lambda = Aa_\lambda \twoheadrightarrow V_\lambda = Aa_\lambda b_\lambda$, given by $x \mapsto x \cdot b_\lambda$).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kostka Numbers)</span></p>

For any partitions $\lambda$ and $\mu$ of $d$, the **Kostka number** $K_{\mu\lambda}$ is the number of ways to fill the boxes of the Young diagram of $\mu$ with $\lambda_1$ 1's, $\lambda_2$ 2's, $\ldots$, $\lambda_k$ $k$'s, in such a way that the entries in each row are nondecreasing and those in each column are strictly increasing. Such fillings are called **semistandard tableaux** of $\mu$ of type $\lambda$.

Key properties:

- $K_{\lambda\lambda} = 1$, and $K_{\mu\lambda} = 0$ for $\mu < \lambda$.
- $K_{\mu\lambda}$ equals the coefficient of $x_1^{\mu_1} \cdots x_k^{\mu_k}$ in the Schur polynomial $S_\lambda$.

</div>

The character $\psi_\lambda$ of $U_\lambda$ is easy to compute directly since $U_\lambda$ is an induced representation. Using the formula for characters of induced representations (Exercise 3.19), one obtains

$$\psi_\lambda(C_\mathbf{i}) = [P^{(\mathbf{i})}]_\lambda = \text{coefficient of } X^\lambda = x_1^{\lambda_1} \cdots x_k^{\lambda_k} \text{ in } P^{(\mathbf{i})}, \tag{4.33}$$

where $P^{(\mathbf{i})} = (x_1 + \cdots + x_k)^{i_1} \cdot (x_1^2 + \cdots + x_k^2)^{i_2} \cdots (x_1^d + \cdots + x_k^d)^{i_d}$ is the power sum symmetric polynomial.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Strategy of the Proof)</span></p>

Define auxiliary functions $\omega_\lambda(\mathbf{i}) = [\Delta \cdot P^{(\mathbf{i})}]_l$, where $l = (\lambda_1 + k - 1, \lambda_2 + k - 2, \ldots, \lambda_k)$. Frobenius's formula asserts that $\chi_\lambda(C_\mathbf{i}) = \omega_\lambda(\mathbf{i})$.

The proof proceeds by showing:

1. The functions $\omega_\lambda$ satisfy the same orthogonality relations as irreducible characters (equation (4.36)):

$$\frac{1}{d!} \sum_\mathbf{i} \lvert C_\mathbf{i} \rvert \overline{\omega_\lambda(\mathbf{i})} \omega_\mu(\mathbf{i}) = \delta_{\lambda\mu}.$$

2. From the expansion $\psi_\lambda = \sum_\mu K_{\mu\lambda} \omega_\mu(\mathbf{i})$ and the triangularity of the Kostka matrix ($K_{\lambda\lambda} = 1$, $K_{\mu\lambda} = 0$ for $\mu < \lambda$), one deduces that each $\omega_\lambda$ is $\pm\chi$ for some irreducible character $\chi$.

3. By induction (in lexicographic order), one shows that in fact $\omega_\lambda = \chi_\lambda$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(4.37 — Frobenius's Formula Is the Character)</span></p>

Let $\chi_\lambda = \chi_{V_\lambda}$ be the character of $V_\lambda$. Then for any conjugacy class $C_\mathbf{i}$ of $\mathfrak{S}_d$,

$$\chi_\lambda(C_\mathbf{i}) = \omega_\lambda(\mathbf{i}).$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(4.39 — Young's Rule)</span></p>

The Kostka number $K_{\mu\lambda}$ is the multiplicity of the irreducible representation $V_\mu$ in the induced representation $U_\lambda$:

$$U_\lambda \cong V_\lambda \oplus \bigoplus_{\mu > \lambda} K_{\mu\lambda} V_\mu, \qquad \psi_\lambda = \chi_\lambda + \sum_{\mu > \lambda} K_{\mu\lambda}\chi_\mu.$$

Note that when $\lambda = (1, \ldots, 1)$, $U_\lambda$ is just the regular representation, so $K_{\mu(1,\ldots,1)} = \dim V_\mu$. This shows that **the dimension of $V_\lambda$ is the number of standard tableaux on $\lambda$**, i.e., the number of ways to fill the Young diagram of $\lambda$ with the numbers from 1 to $d$ such that all rows and columns are increasing.

</div>

#### Decomposition Rules

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Outer Product and Littlewood–Richardson Rule)</span></p>

Given any non-negative integers $d_1, \ldots, d_k$, and representations $V_i$ of $\mathfrak{S}_{d_i}$, we can form the **outer product** (or external tensor product) $V_1 \boxtimes \cdots \boxtimes V_k$ of $\mathfrak{S}_{d_1} \times \cdots \times \mathfrak{S}_{d_k}$, and then induce it to $\mathfrak{S}_d$ (where $d = \sum d_i$). This product is commutative and associative. For two factors, one has

$$V_\lambda \circ V_\mu = \sum_\nu C_{\lambda\mu\nu}\, V_\nu, \tag{4.41}$$

where the $C_{\lambda\mu\nu}$ are the coefficients given by the **Littlewood–Richardson rule** (see Appendix A).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Pieri's Formula / Branching Rule)</span></p>

When $m = 1$ and $\mu = (1)$, the outer product formula simplifies to **Pieri's formula**: the induced representation $\mathrm{Ind}_{\mathfrak{S}_{d-1}}^{\mathfrak{S}_d} V_\lambda$ decomposes as

$$\mathrm{Ind}_{\mathfrak{S}_{d-1}}^{\mathfrak{S}_d} V_\lambda = \bigoplus_\nu V_\nu, \tag{4.42}$$

the sum over all $\nu$ whose Young diagram can be obtained from that of $\lambda$ by adding one box. Dually, by Frobenius reciprocity,

$$\mathrm{Res}_{\mathfrak{S}_{d-1}}^{\mathfrak{S}_d} V_\nu = \bigoplus_\lambda V_\lambda,$$

the sum over all $\lambda$ obtained from $\nu$ by removing one box. This is known as the **branching theorem**, and is useful for inductive proofs and constructions, particularly because the decomposition is **multiplicity free**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Murnaghan–Nakayama Rule)</span></p>

The **Murnaghan–Nakayama rule** gives an efficient inductive method for computing character values. If $\lambda$ is a partition of $d$ and $g \in \mathfrak{S}_d$ is written as a product of an $m$-cycle and a disjoint permutation $h \in \mathfrak{S}_{d-m}$, then

$$\chi_\lambda(g) = \sum (-1)^{r(\mu)}\chi_\mu(h),$$

where the sum is over all partitions $\mu$ of $d - m$ that are obtained from $\lambda$ by removing a **skew hook** of length $m$, and $r(\mu)$ is the number of vertical steps in the hook (i.e., one less than the number of rows in the hook). A **skew hook** for $\lambda$ is a connected region of boundary boxes of $\lambda$'s Young diagram such that removing them leaves a smaller Young diagram.

If $\lambda$ has no hooks of length $m$, then $\chi_\lambda(g) = 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Representation Ring as a Hopf Algebra)</span></p>

Let $R_d = R(\mathfrak{S}_d)$ denote the representation ring, and set $R = \bigoplus_{d=0}^{\infty} R_d$. The outer product of (4.41) makes $R$ into a commutative, graded $\mathbb{Z}$-algebra. Restriction determines maps

$$R_{n+m} = R(\mathfrak{S}_{n+m}) \to R(\mathfrak{S}_n \times \mathfrak{S}_m) = R_n \otimes R_m,$$

which defines a **co-product** $\delta\colon R \to R \otimes R$, making $R$ into a graded Hopf algebra. As an algebra,

$$R \cong \mathbb{Z}[H_1, \ldots, H_d, \ldots],$$

where $H_d$ is an indeterminate of degree $d$ corresponding to the trivial representation of $\mathfrak{S}_d$. Setting $\Lambda = \mathbb{Z}[H_1, H_2, \ldots] = \bigoplus \Lambda_d$, the ring of symmetric polynomials in Appendix A, we can identify $\Lambda_d$ with $R_d$ via the correspondences:

$$H_\lambda \leftrightarrow U_\lambda, \qquad S_\lambda \leftrightarrow V_\lambda, \qquad E_\lambda \leftrightarrow U_\lambda'.$$

The scalar product on class functions in (2.11) corresponds to the scalar product on $\Lambda$ defined in (A.16). The involution $\vartheta$ of Exercise A.32 corresponds to tensoring a representation with the alternating representation $U'$.

</div>

---

## Lecture 5: Representations of $\mathfrak{A}_d$ and $\mathrm{GL}_2(\mathbb{F}_q)$

This lecture analyzes two more types of groups: the alternating groups $\mathfrak{A}_d$ and the linear groups $\mathrm{GL}_2(\mathbb{F}_q)$ and $\mathrm{SL}_2(\mathbb{F}_q)$ over finite fields. In the former case, we use the relationship between a group and a subgroup of index two; in the latter case, we start essentially from scratch.

### 5.1 Representations of $\mathfrak{A}_d$

The basic method for analyzing representations of $\mathfrak{A}_d$ is by restricting the representations we know from $\mathfrak{S}_d$. In general, when $H$ is a subgroup of index two in a group $G$, there is a close relationship between their representations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conjugate Representation)</span></p>

Let $G$ be a group, $H$ a subgroup of index two. If $W$ is any representation of $H$, there is a **conjugate** representation defined by conjugating by any element $t \in G$ not in $H$: $h \mapsto \psi(tht^{-1})$, where $\psi$ is the character of $W$. Since $t$ is unique up to multiplication by an element of $H$, the conjugate representation is unique up to isomorphism.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.1 — Restriction from Index-Two Subgroup)</span></p>

Let $V$ be an irreducible representation of $G$, and let $W = \mathrm{Res}_H^G\, V$. Then exactly one of the following holds:

1. $V$ is not isomorphic to $V' = V \otimes U'$; $W$ is irreducible and isomorphic to its conjugate; $\mathrm{Ind}_H^G\, W \cong V \oplus V'$.
2. $V \cong V'$; $W = W' \oplus W''$, where $W'$ and $W''$ are irreducible and conjugate but not isomorphic; $\mathrm{Ind}_H^G\, W' \cong \mathrm{Ind}_H^G\, W'' \cong V$.

Every irreducible representation of $H$ arises uniquely in this way, noting that in case (1) $V'$ and $V$ determine the same representation.

**Proof.** Let $\chi$ be the character of $V$. We have

$$\lvert G \rvert = 2\lvert H \rvert = \sum_{h \in H} \lvert\chi(h)\rvert^2 + \sum_{t \notin H} \lvert\chi(t)\rvert^2.$$

Since the first sum is an integral multiple of $\lvert H \rvert$, this multiple must be 1 or 2, which are the two cases of the proposition. The second case happens when $\chi(t) = 0$ for all $t \notin H$, which is the case when $V' \cong V$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Split and Nonsplit Conjugacy Classes)</span></p>

There are two types of conjugacy classes $c$ in $H$: those that are also conjugacy classes in $G$, and those such that $c \cup c'$ is a conjugacy class in $G$ (where $c' = tct^{-1}$, $t \notin H$); the latter are called **split**. When $W$ is irreducible, its character assumes the same values on split conjugacy classes as the character of its conjugate $W''$; in the other case the characters of $W'$ and $W''$ agree on nonsplit classes but disagree on some split classes.

</div>

#### Application to $\mathfrak{A}_d$

Recall from Lecture 4 that if $\lambda'$ is the conjugate partition to $\lambda$, then $V_{\lambda'} = V_\lambda \otimes U'$. The two cases of Proposition 5.1 correspond to:

- **Case 1** ($\lambda' \ne \lambda$): Let $W_\lambda$ be the restriction of $V_\lambda$ to $\mathfrak{A}_d$. Then $\mathrm{Res}\, V_\lambda = \mathrm{Res}\, V_{\lambda'} = W_\lambda$ is irreducible, and $\mathrm{Ind}\, W_\lambda = V_\lambda \oplus V_{\lambda'}$.
- **Case 2** ($\lambda' = \lambda$, i.e., $\lambda$ is **self-conjugate**): Let $W_\lambda'$ and $W_\lambda''$ be the two irreducible representations whose sum is the restriction of $V_\lambda$ to $\mathfrak{A}_d$. Then $\mathrm{Ind}\, W_\lambda' = \mathrm{Ind}\, W_\lambda'' = V_\lambda$, and $\mathrm{Res}\, V_\lambda = W_\lambda' \oplus W_\lambda''$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Self-Conjugate Partitions and Split Classes)</span></p>

The number of self-conjugate representations of $\mathfrak{S}_d$ equals the number of symmetric Young diagrams, which equals the number of split pairs of conjugacy classes in $\mathfrak{A}_d$, which equals the number of conjugacy classes in $\mathfrak{S}_d$ that break into two in $\mathfrak{A}_d$.

A conjugacy class of an element written as a product of disjoint cycles is split if and only if all the cycles have odd length and no two cycles have the same length. So the number of self-conjugate representations is the number of partitions of $d$ as a sum of **distinct odd numbers**. There is a natural bijection between these two sets: any such partition corresponds to a symmetric Young diagram, assembled from hooks of the given odd lengths.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.3 — Characters of $\mathfrak{A}_d$ on Split Classes)</span></p>

For a self-conjugate partition $\lambda$, let $\chi_\lambda'$ and $\chi_\lambda''$ denote the characters of $W_\lambda'$ and $W_\lambda''$, and let $c$ and $c'$ be a pair of split conjugacy classes, consisting of cycles of odd lengths $q_1 > q_2 > \cdots > q_r$. The following proposition of Frobenius completes the description of the character table of $\mathfrak{A}_d$.

1. If $c$ and $c'$ do not correspond to the partition $\lambda$, then

$$\chi_\lambda'(c) = \chi_\lambda''(c) = \chi_\lambda'(c') = \chi_\lambda''(c') = \tfrac{1}{2}\chi_\lambda(c \cup c').$$

2. If $c$ and $c'$ correspond to $\lambda$, then

$$\chi_\lambda'(c') = x, \quad \chi_\lambda''(c') = y, \quad \chi_\lambda'(c) = y, \quad \chi_\lambda''(c) = x,$$

with $x$ and $y$ the two numbers

$$\tfrac{1}{2}\bigl((-1)^m \pm \sqrt{(-1)^m q_1 \cdots q_r}\bigr),$$

and $m = \frac{1}{2}(d - r) = \frac{1}{2}\sum(q_i - 1) \equiv \frac{1}{4}(\prod q_i - 1) \pmod{2}$.

Note that the integer $m$ appearing in (2) is the number of squares above the diagonal in the Young diagram of $\lambda$.

</div>

### 5.2 Representations of $\mathrm{GL}_2(\mathbb{F}_q)$ and $\mathrm{SL}_2(\mathbb{F}_q)$

#### The Group $\mathrm{GL}_2(\mathbb{F}_q)$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key Subgroups of $\mathrm{GL}_2(\mathbb{F}_q)$)</span></p>

Let $G = \mathrm{GL}_2(\mathbb{F}_q)$, the group of invertible $2 \times 2$ matrices over the finite field $\mathbb{F}_q$ ($q$ a prime power). Key subgroups include:

- **Borel subgroup:** $B = \left\lbrace \begin{pmatrix} a & b \\ 0 & d \end{pmatrix} \right\rbrace$, the upper triangular matrices.
- **Unipotent subgroup:** $N = \left\lbrace \begin{pmatrix} 1 & b \\ 0 & 1 \end{pmatrix} \right\rbrace$.
- **Diagonal torus:** $D = \left\lbrace \begin{pmatrix} a & 0 \\ 0 & d \end{pmatrix} \right\rbrace \cong \mathbb{F}^* \times \mathbb{F}^*$.
- **Cyclic subgroup:** $K = (\mathbb{F}')^*$ where $\mathbb{F}' = \mathbb{F}_{q^2}$ is the degree-two extension of $\mathbb{F}$, embedded via a choice of $\mathbb{F}$-basis of $\mathbb{F}'$; $K$ is a cyclic subgroup of order $q^2 - 1$.

Since $G$ acts transitively on $\mathbb{P}^1(\mathbb{F}_q)$ with $B$ the isotropy group of the point $(1:0)$, we have $\lvert G \rvert = \lvert B \rvert \cdot \lvert\mathbb{P}^1(\mathbb{F}_q)\rvert = (q-1)^2 q(q+1)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Conjugacy Classes of $\mathrm{GL}_2(\mathbb{F}_q)$)</span></p>

The conjugacy classes of $G$ are easily found:

| Representative | No. Elements | No. Classes |
|---|---|---|
| $a_x = \begin{pmatrix} x & 0 \\ 0 & x \end{pmatrix}$ | $1$ | $q - 1$ |
| $b_x = \begin{pmatrix} x & 1 \\ 0 & x \end{pmatrix}$ | $q^2 - 1$ | $q - 1$ |
| $c_{x,y} = \begin{pmatrix} x & 0 \\ 0 & y \end{pmatrix}$, $x \ne y$ | $q^2 + q$ | $\frac{(q-1)(q-2)}{2}$ |
| $d_{x,y} = \begin{pmatrix} x & \varepsilon y \\ y & x \end{pmatrix}$, $y \ne 0$ | $q^2 - q$ | $\frac{q(q-1)}{2}$ |

Here $c_{x,y}$ and $c_{y,x}$ are conjugate, and $d_{x,y}$ and $d_{x,-y}$ are conjugate. The total is $q^2 - 1$ conjugacy classes.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Irreducible Representations of $\mathrm{GL}_2(\mathbb{F}_q)$)</span></p>

The $q^2 - 1$ irreducible representations of $G$ are found by several methods:

**One-dimensional representations $U_\alpha$:** For each character $\alpha\colon \mathbb{F}^* \to \mathbb{C}^*$, define $U_\alpha(g) = \alpha(\det(g))$. This gives $q - 1$ representations.

**$(q+1)$-dimensional representations $W_{\alpha,\beta}$:** For each pair $\alpha, \beta$ of characters of $\mathbb{F}^*$, induce the character $\begin{pmatrix} a & b \\ 0 & d \end{pmatrix} \mapsto \alpha(a)\beta(d)$ from $B$ to $G$. This gives a representation of dimension $[G:B] = q + 1$. We have $W_{\alpha,\beta} \cong W_{\beta,\alpha}$, and $W_{\alpha,\alpha} \cong U_\alpha \oplus V_\alpha$ where $V_\alpha = V \otimes U_\alpha$ with $V$ the standard $q$-dimensional representation. This yields $\frac{1}{2}(q-1)(q-2)$ more irreducible representations of dimension $q + 1$.

**$q$-dimensional representations $V_\alpha$:** $V_\alpha = V \otimes U_\alpha$, for $q - 1$ characters $\alpha$, giving $q - 1$ irreducible representations of dimension $q$.

**$(q-1)$-dimensional representations $X_\varphi$:** Induce a character $\varphi\colon K = (\mathbb{F}')^* \to \mathbb{C}^*$ from the cyclic subgroup $K$. For $\varphi^q \ne \varphi$, $\mathrm{Ind}(\varphi)$ is not irreducible. Using tensor product analysis with $V \otimes W_{\alpha,1}$, one extracts irreducible representations $X_\varphi$ of dimension $q - 1$ from the virtual character $\chi_\varphi = \chi_{V \otimes W_{\alpha,1}} - \chi_{W_{\alpha,1}} - \chi_{\mathrm{Ind}(\varphi)}$. This gives $\frac{1}{2}q(q-1)$ representations.

The complete character table is:

| $\mathrm{GL}_2(\mathbb{F}_q)$ | $a_x$ | $b_x$ | $c_{x,y}$ | $d_{x,y} = \zeta$ |
|---|---|---|---|---|
| $U_\alpha$ | $\alpha(x^2)$ | $\alpha(x^2)$ | $\alpha(xy)$ | $\alpha(\zeta^q)$ |
| $V_\alpha$ | $q\alpha(x^2)$ | $0$ | $\alpha(xy)$ | $-\alpha(\zeta^q)$ |
| $W_{\alpha,\beta}$ | $(q+1)\alpha(x)\beta(x)$ | $\alpha(x)\beta(x)$ | $\alpha(x)\beta(y) + \alpha(y)\beta(x)$ | $0$ |
| $X_\varphi$ | $(q-1)\varphi(x)$ | $-\varphi(x)$ | $0$ | $-(\varphi(\zeta) + \varphi(\zeta^q))$ |

where in the last column we identify $d_{x,y}$ with $\zeta = x + y\sqrt{\varepsilon} \in K = (\mathbb{F}')^*$.

</div>

#### The Group $\mathrm{SL}_2(\mathbb{F}_q)$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Conjugacy Classes of $\mathrm{SL}_2(\mathbb{F}_q)$, $q$ odd)</span></p>

The conjugacy classes of $\mathrm{SL}_2(\mathbb{F}_q)$ for $q$ odd are:

| Type | Representative | No. Elements | No. Classes |
|---|---|---|---|
| (1) | $e = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ | $1$ | $1$ |
| (2) | $-e = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}$ | $1$ | $1$ |
| (3)–(6) | $\begin{pmatrix} \pm 1 & 1 \\ 0 & \pm 1 \end{pmatrix}$, $\begin{pmatrix} \pm 1 & \varepsilon \\ 0 & \pm 1 \end{pmatrix}$ | $\frac{q^2-1}{2}$ each | $1$ each |
| (7) | $\begin{pmatrix} x & 0 \\ 0 & x^{-1} \end{pmatrix}$, $x \ne \pm 1$ | $q(q+1)$ | $\frac{q-3}{2}$ |
| (8) | $\begin{pmatrix} x & y \\ \varepsilon y & x \end{pmatrix}$, $x \ne \pm 1$ | $q(q-1)$ | $\frac{q-1}{2}$ |

The total number of conjugacy classes is $q + 4$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Strategy for $\mathrm{SL}_2(\mathbb{F}_q)$)</span></p>

To find the $q + 4$ irreducible representations, we restrict from $\mathrm{GL}_2(\mathbb{F}_q)$:

1. The $U_\alpha$ all restrict to the trivial representation $U$.
2. The restriction of $V$ (the standard representation) is irreducible.
3. The restriction $W_\alpha$ of $W_{\alpha,1}$ is irreducible if $\alpha^2 \ne 1$, and $W_\alpha \cong W_\beta$ when $\beta = \alpha$ or $\beta = \alpha^{-1}$. This gives $\frac{1}{2}(q - 3)$ irreducible representations of dimension $q + 1$. When $\alpha^2 = 1$, $\alpha \ne 1$ (i.e., $\alpha = \tau$, the unique character of order 2), the restriction of $W_{\tau,1}$ splits into the sum of two distinct irreducible representations $W'$ and $W''$ of dimension $\frac{1}{2}(q + 1)$.
4. The restriction of $X_\varphi$ depends on $\varphi\vert_C$ where $C = \lbrace \zeta \in (\mathbb{F}')^* \colon \zeta^{q+1} = 1 \rbrace$. For $\varphi^2 \ne 1$ on $C$, the representation is irreducible of dimension $q - 1$, giving $\frac{1}{2}(q - 1)$ representations. When $\psi^2 = 1$, $\psi \ne 1$ on $C$, the restriction of $X_\psi$ splits into two conjugate irreducible representations $X'$ and $X''$ of dimension $\frac{1}{2}(q - 1)$.

Altogether this gives $q + 4$ distinct irreducible representations, completing the list.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Determining Characters on Split Classes)</span></p>

To finish the character table of $\mathrm{SL}_2(\mathbb{F}_q)$, we use a subgroup $H$ of index two in $\mathrm{GL}_2(\mathbb{F}_q)$ that contains $\mathrm{SL}_2(\mathbb{F}_q)$ (namely, the matrices whose determinant is a square). There are $q - 1$ split conjugacy classes. The character values $s, t$ of $W'$ and $W''$ on these classes satisfy $s + t = \chi_{W_{\tau,1}}\bigl(\begin{smallmatrix} 1 & 1 \\ 0 & 1 \end{smallmatrix}\bigr) = 1$, plus additional constraints from the relations $\chi(g^{-1}) = \overline{\chi(g)}$ and $\chi(-g) = \chi(g) \cdot \chi(-e)/\chi(e)$. The key identity $s' = \tau(-1)s$ and $t' = \tau(-1)t$ (where primes denote values on classes (4) and (6) vs. (3) and (5)), combined with $(\chi, \chi) = 1$, gives $s, t = \frac{1}{2} \pm \frac{1}{2}\sqrt{\omega q}$, where $\omega = \tau(-1)$ is $1$ or $-1$ according as $q \equiv 1$ or $3 \pmod{4}$. Similarly for $X'$ and $X''$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connections to Other Groups)</span></p>

The quotient $\mathrm{PGL}_2(\mathbb{F}_q) = \mathrm{GL}_2(\mathbb{F}_q)/\mathbb{F}^*$ is the automorphism group of the finite projective line $\mathbb{P}^1(\mathbb{F}_q)$. The quotients $\mathrm{PSL}_2(\mathbb{F}_q) = \mathrm{SL}_2(\mathbb{F}_q)/\lbrace \pm 1 \rbrace$ are simple groups if $q$ is odd and greater than 3. For small $q$:

$$\mathrm{SL}_2(\mathbb{F}_2) \cong \mathfrak{S}_3, \quad \mathrm{PSL}_2(\mathbb{F}_3) \cong \mathfrak{A}_4, \quad \mathrm{SL}_2(\mathbb{F}_4) \cong \mathfrak{A}_5.$$

Although the characters of these groups were found by the early pioneers in representation theory, actually producing the representations in a natural way is more difficult. There has been a great deal of work extending this story to $\mathrm{GL}_n(\mathbb{F}_q)$ and $\mathrm{SL}_n(\mathbb{F}_q)$ for $n > 2$, and for corresponding groups called finite Chevalley groups, related to other Lie groups.

</div>

---

## Lecture 6: Weyl's Construction

In this lecture we introduce and study an important collection of functors generalizing the symmetric powers and exterior powers. These are defined simply in terms of the Young symmetrizers $c_\lambda$ introduced in Lecture 4: given a representation $V$ of an arbitrary group $G$, we consider the $d$th tensor power of $V$, on which both $G$ and $\mathfrak{S}_d$ act. We then take the image of the action of $c_\lambda$ on $V^{\otimes d}$; this is again a representation of $G$, denoted $\mathbb{S}_\lambda V$. The main application will be to Lie groups: these functors will generate all representations of $\mathrm{SL}_n\mathbb{C}$ from the standard representation $\mathbb{C}^n$ of $\mathrm{SL}_n\mathbb{C}$.

### 6.1 Schur Functors and Their Characters

For any finite-dimensional complex vector space $V$, the group $\mathrm{GL}(V)$ acts on $V^{\otimes d} = V \otimes V \otimes \cdots \otimes V$ ($d$ factors). The symmetric group $\mathfrak{S}_d$ also acts on $V^{\otimes d}$, on the right, by permuting the factors:

$$(v_1 \otimes \cdots \otimes v_d) \cdot \sigma = v_{\sigma(1)} \otimes \cdots \otimes v_{\sigma(d)}.$$

This action commutes with the left action of $\mathrm{GL}(V)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Schur Functor / Weyl Module)</span></p>

For any partition $\lambda$ of $d$, we have the Young symmetrizer $c_\lambda \in \mathbb{C}\mathfrak{S}_d$. We denote the image of $c_\lambda$ on $V^{\otimes d}$ by $\mathbb{S}_\lambda V$:

$$\mathbb{S}_\lambda V = \mathrm{Im}(c_\lambda \vert_{V^{\otimes d}}).$$

This is again a representation of $\mathrm{GL}(V)$. We call the functor $V \leadsto \mathbb{S}_\lambda V$ the **Schur functor** or **Weyl module**, or simply **Weyl's construction**, corresponding to $\lambda$. The functoriality means that a linear map $\varphi\colon V \to W$ of vector spaces determines a linear map $\mathbb{S}_\lambda(\varphi)\colon \mathbb{S}_\lambda V \to \mathbb{S}_\lambda W$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Basic Schur Functors)</span></p>

- For $\lambda = (d)$: $\mathbb{S}_{(d)}V = \mathrm{Sym}^d V$.
- For $\lambda = (1, \ldots, 1)$: $\mathbb{S}_{(1,\ldots,1)}V = \bigwedge^d V$.
- For $\lambda = (2, 1)$ (partition of 3): the Young symmetrizer is $c_{(2,1)} = 1 + e_{(12)} - e_{(13)} - e_{(132)}$, and $\mathbb{S}_{(2,1)}V$ is the subspace of $V^{\otimes 3}$ spanned by all vectors $v_1 \otimes v_2 \otimes v_3 + v_2 \otimes v_1 \otimes v_3 - v_3 \otimes v_2 \otimes v_1 - v_3 \otimes v_1 \otimes v_2$. If $\bigwedge^2 V \otimes V$ is embedded in $V^{\otimes 3}$ by $(v_1 \wedge v_2) \otimes v_3 \mapsto v_1 \otimes v_2 \otimes v_3 - v_2 \otimes v_1 \otimes v_3$, then the image of $c_{(2,1)}$ is the subspace of $\bigwedge^2 V \otimes V$ spanned by all $(v_1 \wedge v_3) \otimes v_2 + (v_2 \wedge v_3) \otimes v_1$, which is $\ker(\bigwedge^2 V \otimes V \to \bigwedge^3 V)$.

Note that $\mathbb{S}_\lambda V$ can be zero if $V$ has small dimension; this happens precisely when the number of rows in the Young diagram of $\lambda$ is greater than $\dim V$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Characters of Schur Functors)</span></p>

Any endomorphism $g$ of $V$ gives rise to an endomorphism of $\mathbb{S}_\lambda V$. To compute the trace, let $x_1, \ldots, x_k$ be the eigenvalues of $g$ on $V$, $k = \dim V$. The two simplest cases are:

$$\chi_{\mathbb{S}_{(d)}V}(g) = H_d(x_1, \ldots, x_k), \tag{6.1}$$

$$\chi_{\mathbb{S}_{(1,\ldots,1)}V}(g) = E_d(x_1, \ldots, x_k), \tag{6.2}$$

where $H_d$ is the complete symmetric polynomial and $E_d$ is the elementary symmetric polynomial. The polynomials $H_d$ and $E_d$ are special cases of the **Schur polynomials** $S_\lambda = S_\lambda(x_1, \ldots, x_k)$, which form a basis for the symmetric polynomials of degree $d$ in $k$ variables as $\lambda$ varies over the partitions of $d$ in at most $k$ parts.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(6.3 — Properties of Schur Functors)</span></p>

Let $k = \dim V$. Then:

1. $\mathbb{S}_\lambda V$ is zero if $\lambda_{k+1} \ne 0$. If $\lambda = (\lambda_1 \ge \cdots \ge \lambda_k \ge 0)$, then

$$\dim \mathbb{S}_\lambda V = S_\lambda(1, \ldots, 1) = \prod_{1 \le i < j \le k} \frac{\lambda_i - \lambda_j + j - i}{j - i}.$$

2. Let $m_\lambda$ be the dimension of the irreducible representation $V_\lambda$ of $\mathfrak{S}_d$ corresponding to $\lambda$. Then

$$V^{\otimes d} \cong \bigoplus_\lambda \mathbb{S}_\lambda V^{\oplus m_\lambda}$$

as representations of $\mathrm{GL}(V)$.

3. For any $g \in \mathrm{GL}(V)$, the trace of $g$ on $\mathbb{S}_\lambda V$ is the value of the Schur polynomial on the eigenvalues $x_1, \ldots, x_k$ of $g$ on $V$:

$$\chi_{\mathbb{S}_\lambda V}(g) = S_\lambda(x_1, \ldots, x_k).$$

4. Each $\mathbb{S}_\lambda V$ is an **irreducible** representation of $\mathrm{GL}(V)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(6.6 — Decomposition of $V^{\otimes d}$ via Group Ring)</span></p>

If $c \in \mathbb{C}\mathfrak{S}_d$ and $(\mathbb{C}\mathfrak{S}_d) \cdot c = \bigoplus_\lambda V_\lambda^{\oplus r_\lambda}$ as representations of $\mathfrak{S}_d$, then there is a corresponding decomposition of $\mathrm{GL}(V)$-spaces:

$$V^{\otimes d} \cdot c = \bigoplus_\lambda \mathbb{S}_\lambda V^{\oplus r_\lambda}.$$

If $x_1, \ldots, x_k$ are the eigenvalues of an endomorphism of $V$, the trace of the induced endomorphism of $V^{\otimes d} \cdot c$ is $\sum r_\lambda S_\lambda(x_1, \ldots, x_k)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-Isomorphic Schur Functors Have Different Characters)</span></p>

If $\lambda$ and $\mu$ are different partitions, each with at most $k = \dim V$ parts, the irreducible $\mathrm{GL}(V)$-spaces $\mathbb{S}_\lambda V$ and $\mathbb{S}_\mu V$ are not isomorphic. Indeed, their characters are the Schur polynomials $S_\lambda$ and $S_\mu$, which are different. More generally, for those representations of $\mathrm{GL}(V)$ which can be decomposed into a direct sum of copies of the representations $\mathbb{S}_\lambda V$, *the representations are completely determined by their characters*.

Note, however, that we cannot hope to get *all* finite-dimensional irreducible representations of $\mathrm{GL}(V)$ this way, since the duals of these representations are not included. We will see in Lecture 15 that this is essentially the only omission.

</div>

#### Tensor Products and Pieri/Littlewood–Richardson Formulas for Schur Functors

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Littlewood–Richardson Rule for Schur Functors)</span></p>

For partitions $\lambda$ of $d$ and $\mu$ of $m$, the tensor product of Schur functors decomposes as

$$\mathbb{S}_\lambda V \otimes \mathbb{S}_\mu V \cong \bigoplus_\nu N_{\lambda\mu\nu}\, \mathbb{S}_\nu V, \tag{6.7}$$

where the sum is over partitions $\nu$ of $d + m$, and $N_{\lambda\mu\nu}$ are the Littlewood–Richardson numbers.

</div>

Two important special cases that involve only the simpler Pieri formula (A.7):

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Pieri Formulas for Schur Functors)</span></p>

For $\mu = (m)$ (a single row):

$$\mathbb{S}_\lambda V \otimes \mathrm{Sym}^m V \cong \bigoplus_\nu \mathbb{S}_\nu V, \tag{6.8}$$

the sum over all $\nu$ whose Young diagram is obtained by adding $m$ boxes to $\lambda$, with no two in the same column. Similarly for $\mu = (1, \ldots, 1)$ (a single column):

$$\mathbb{S}_\lambda V \otimes \bigwedge^m V \cong \bigoplus_\pi \mathbb{S}_\pi V, \tag{6.9}$$

the sum over all $\pi$ whose Young diagram is obtained by adding $m$ boxes to $\lambda$, with no two in the same row.

In particular, from $\mathrm{Sym}^d V \otimes V = \mathrm{Sym}^{d+1}V \oplus \mathbb{S}_{(d,1)}V$, it follows that

$$\mathbb{S}_{(d,1)}V = \ker(\mathrm{Sym}^d V \otimes V \to \mathrm{Sym}^{d+1}V),$$

and similarly for the conjugate partition,

$$\mathbb{S}_{(2,1,\ldots,1)}V = \ker(\bigwedge^d V \otimes V \to \bigwedge^{d+1} V).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Decomposition of Symmetric and Exterior Powers of Direct Sums and Tensor Products)</span></p>

The Littlewood–Richardson rule also gives the decomposition of a Schur functor of a direct sum over $\mathrm{GL}(V) \times \mathrm{GL}(W)$:

$$\mathbb{S}_\nu(V \oplus W) = \bigoplus N_{\lambda\mu\nu}(\mathbb{S}_\lambda V \otimes \mathbb{S}_\mu W),$$

and for a tensor product:

$$\mathbb{S}_\nu(V \otimes W) = \bigoplus C_{\lambda\mu\nu}(\mathbb{S}_\lambda V \otimes \mathbb{S}_\mu W),$$

where the $C_{\lambda\mu\nu}$ are defined in Exercise 4.51. In particular,

$$\mathrm{Sym}^d(V \otimes W) = \bigoplus_\lambda \mathbb{S}_\lambda V \otimes \mathbb{S}_\lambda W, \qquad \bigwedge^d(V \otimes W) = \bigoplus_\lambda \mathbb{S}_\lambda V \otimes \mathbb{S}_{\lambda'} W,$$

the sums over partitions $\lambda$ of $d$ with at most $\dim V$ rows and $\dim W$ columns.

</div>

#### Skew Schur Functors

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Skew Young Diagram and Skew Schur Functor)</span></p>

If $\lambda$ and $\mu$ are partitions with $\mu_i \le \lambda_i$ for all $i$, the **skew Young diagram** $\lambda/\mu$ is the complement of the Young diagram for $\mu$ in that of $\lambda$. Each $\lambda/\mu$ determines elements $a_{\lambda/\mu}$, $b_{\lambda/\mu}$, and Young symmetrizers $c_{\lambda/\mu} = a_{\lambda/\mu} b_{\lambda/\mu}$ in $A = \mathbb{C}\mathfrak{S}_d$ (where $d = \sum \lambda_i - \mu_i$), exactly as in §4.1, and hence a representation $V_{\lambda/\mu}$ of $\mathfrak{S}_d$ and a **skew Schur functor** $\mathbb{S}_{\lambda/\mu} V$.

The decompositions into irreducibles are:

$$V_{\lambda/\mu} = \sum N_{\mu\nu\lambda}\, V_\nu, \tag{v}$$

$$\mathbb{S}_{\lambda/\mu} V \cong \sum N_{\mu\nu\lambda}\, \mathbb{S}_\nu V, \tag{ix}$$

where $N_{\mu\nu\lambda}$ is the Littlewood–Richardson number.

The character of $\mathbb{S}_{\lambda/\mu}V$ is the **skew Schur function** $S_{\lambda/\mu}(x_1, \ldots, x_k)$, which can be defined equivalently as:

- $S_{\lambda/\mu} = \lvert H_{\lambda_i - \mu_j - i + j} \rvert$ (determinant),
- $S_{\lambda/\mu} = \lvert E_{\lambda_i' - \mu_j' - i + j} \rvert$,
- $S_{\lambda/\mu} = \sum m_a\, x_1^{a_1} \cdots x_k^{a_k}$, where $m_a$ counts semistandard tableaux on $\lambda/\mu$,
- $S_{\lambda/\mu} = \sum N_{\mu\nu\lambda}\, S_\nu$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Schur Functors as Kernels of Contractions)</span></p>

Several alternative descriptions of $\mathbb{S}_\lambda V$ are available:

- $\mathbb{S}_\lambda V$ is the image of a composite map (Exercise 6.14):

$$\bigotimes_i (\bigwedge^{\mu_i} V) \to V^{\otimes d} \to \bigotimes_j (\mathrm{Sym}^{\lambda_j} V),$$

where the first map groups factors by columns and the second by rows, or equivalently the image of $\bigotimes_i (\mathrm{Sym}^{\lambda_i'} V) \to V^{\otimes d} \to \bigotimes_j (\bigwedge^{\mu_j} V)$.

- $\mathbb{S}_\lambda V$ can be realized as a subspace of tensors in $V^{\otimes d}$ that are invariant by automorphisms preserving the rows of a Young tableau, *or* as a subspace that is anti-invariant under those preserving the columns, but not both (cf. Exercise 4.48).

- For $\lambda = (p, 1, \ldots, 1)$ (a hook with one row of length $p$ and $d - p$ additional rows of length 1):

$$\mathbb{S}_{(p,1,\ldots,1)} V = \ker(\mathrm{Sym}^p V \otimes \bigwedge^{d-p} V \to \mathrm{Sym}^{p+1} V \otimes \bigwedge^{d-p-1} V).$$

More generally, $\mathbb{S}_\lambda V$ is the intersection of the kernels of $k - 1$ contraction maps (Exercise 6.20).

</div>

### 6.2 The Proofs

The proofs of Theorem 6.3 use tools from the theory of semisimple algebras applied to the group algebra $A = \mathbb{C}\mathfrak{S}_d$ and the commutant algebra $B = \mathrm{Hom}_G(U, U)$ where $U = V^{\otimes d}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Commutant Algebra)</span></p>

If $U$ is a right module over $A = \mathbb{C}G$, let

$$B = \mathrm{Hom}_G(U, U) = \lbrace \varphi\colon U \to U \colon \varphi(v \cdot g) = \varphi(v) \cdot g,\; \forall v \in U,\, g \in G \rbrace.$$

Note that $B$ acts on the left, commuting with the right action of $A$; $B$ is called the **commutator algebra** (or commutant). If $U = \bigoplus_i U_i^{\oplus n_i}$ is an irreducible decomposition with $U_i$ nonisomorphic irreducible right $A$-modules, then by Schur's Lemma,

$$B = \bigoplus_i \mathrm{Hom}_G(U_i^{\oplus n_i}, U_i^{\oplus n_i}) = \bigoplus_i M_{n_i}(\mathbb{C}).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.22 — Tensor Product Decomposition)</span></p>

Let $U$ be a finite-dimensional right $A$-module.

1. For any $c \in A$, the canonical map $U \otimes_A Ac \to Uc$ is an isomorphism of left $B$-modules.
2. If $W = Ac$ is an irreducible left $A$-module, then $U \otimes_A W = Uc$ is an irreducible left $B$-module.
3. If $W_i = Ac_i$ are the distinct irreducible left $A$-modules, with $m_i$ the dimension of $W_i$, then

$$U \cong \bigoplus_i (U \otimes_A W_i)^{\oplus m_i} \cong \bigoplus_i (Uc_i)^{\oplus m_i}$$

is the decomposition of $U$ into irreducible left $B$-modules.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(6.23 — Commutant of $V^{\otimes d}$)</span></p>

The algebra $B$ (the commutant of $V^{\otimes d}$ as a right $\mathbb{C}\mathfrak{S}_d$-module) is spanned as a linear subspace of $\mathrm{End}(V^{\otimes d})$ by $\mathrm{End}(V)$. A subspace of $V^{\otimes d}$ is a sub-$B$-module if and only if it is $\mathrm{GL}(V)$-invariant.

**Proof sketch.** For any finite-dimensional vector space $W$, $\mathrm{Sym}^d W$ is the subspace of $W^{\otimes d}$ spanned by all $w^d = d!\, w \otimes \cdots \otimes w$. Applying this to $W = \mathrm{End}(V) = V^* \otimes V$ proves the first statement, since $\mathrm{End}(V^{\otimes d}) = (V^*)^{\otimes d} \otimes V^{\otimes d}$, with compatible $\mathfrak{S}_d$-actions. The second statement follows from the fact that $\mathrm{GL}(V)$ is dense in $\mathrm{End}(V)$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 6.3)</span></p>

The proof of Theorem 6.3 combines Lemmas 6.22 and 6.23. Note that $\mathbb{S}_\lambda V$ is $Uc_\lambda = V^{\otimes d} \cdot c_\lambda$. Since $V_\lambda = A \cdot c_\lambda$ and $U_\lambda = A \cdot a_\lambda$, from Lemma 6.22 we have an isomorphism of $\mathrm{GL}(V)$-modules:

$$\mathbb{S}_\lambda V \cong V^{\otimes d} \otimes_A V_\lambda. \tag{6.24}$$

Similarly for $U_\lambda$, using $U_\lambda = A \cdot a_\lambda$ and the fact that the image of right multiplication by $a_\lambda$ on $V^{\otimes d}$ is the tensor product of symmetric powers:

$$\mathrm{Sym}^{\lambda_1}V \otimes \mathrm{Sym}^{\lambda_2}V \otimes \cdots \otimes \mathrm{Sym}^{\lambda_k}V \cong V^{\otimes d} \otimes_A U_\lambda. \tag{6.25}$$

By Young's rule (Corollary 4.39), $U_\lambda \cong \bigoplus_\mu K_{\mu\lambda} V_\mu$ as $A$-modules, so we deduce an isomorphism of $\mathrm{GL}(V)$-modules:

$$\mathrm{Sym}^{\lambda_1}V \otimes \cdots \otimes \mathrm{Sym}^{\lambda_k}V \cong \bigoplus_\mu K_{\mu\lambda}\, \mathbb{S}_\mu V. \tag{6.26}$$

The trace of $g$ on the left-hand side of (6.26) is the product $H_{\lambda_1}(x_1, \ldots, x_k) \cdots H_{\lambda_k}(x_1, \ldots, x_k)$ of complete symmetric polynomials. Writing $H_\lambda = \sum_\mu K_{\mu\lambda} S_\mu$ and using the fact that the Kostka matrix is upper triangular with 1's on the diagonal, one deduces that $\mathrm{Trace}(\mathbb{S}_\lambda(g)) = S_\lambda(x_1, \ldots, x_k)$, proving part (3). Part (1) then follows by evaluating the Schur polynomial at $x_1 = \cdots = x_k = 1$, and parts (2) and (4) follow from Lemma 6.22. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Schur–Weyl Duality)</span></p>

The canonical decomposition from Exercise 6.30 gives:

$$V^{\otimes d} = \bigoplus_\lambda \mathbb{S}_\lambda V \otimes_\mathbb{C} V_\lambda,$$

the sum over partitions $\lambda$ of $d$ into at most $k = \dim V$ parts; this decomposition is compatible with the actions of $\mathrm{GL}(V)$ (on the left factor) and $\mathfrak{S}_d$ (on the right factor). In particular, the number of times $V_\lambda$ occurs in the representation $V^{\otimes d}$ of $\mathfrak{S}_d$ is the dimension of $\mathbb{S}_\lambda V$. This is the essence of **Schur–Weyl duality**: the representations of $\mathrm{GL}(V)$ and $\mathfrak{S}_d$ on $V^{\otimes d}$ are mutual commutants.

</div>

---

# Part II: Lie Groups and Lie Algebras

From a naive point of view, Lie groups seem to stand at the opposite end of the spectrum of groups from finite ones. As abstract groups they seem enormously complicated; on the other hand, they come with a topology and a manifold structure, making it possible to use geometric concepts.

The key simplification is the **Lie algebra**: it extracts much of the structure of a Lie group (primarily its algebraic structure) while seemingly getting rid of the topological complexity. Passing from a Lie group to its Lie algebra "linearizes" the group. Since a Lie algebra is just a vector space with bilinear operation, we may further simplify by tensoring with $\mathbb{C}$ and studying **complex Lie algebras**. Elementary Lie algebra theory then leads us to focus on **semisimple Lie algebras**, which are direct sums of simple ones and whose representation theory can be approached in a completely uniform manner.

The progression of objects is:

$$\text{Lie group} \leadsto \text{Lie algebra} \leadsto \text{complex Lie algebra} \leadsto \text{semisimple complex Lie algebra.}$$

---

## Lecture 7: Lie Groups

### 7.1 Lie Groups: Definitions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lie Group)</span></p>

A **Lie group** is a set $G$ endowed simultaneously with the structure of a group and a $\mathscr{C}^\infty$ manifold, such that the multiplication $\times\colon G \times G \to G$ and the inverse $\iota\colon G \to G$ are differentiable maps (equivalently, such that the single map $(x, y) \mapsto x \cdot y^{-1}$ is $\mathscr{C}^\infty$).

A **map** (or morphism) between two Lie groups $G$ and $H$ is a map $\rho\colon G \to H$ that is both differentiable and a group homomorphism.

A **complex Lie group** is defined analogously, replacing "differentiable manifold" by "complex manifold." An **algebraic group** replaces "differentiable manifold" by "algebraic variety" and "differentiable" by "regular morphism."

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lie Subgroup)</span></p>

A **Lie subgroup** (or **closed Lie subgroup**) of a Lie group $G$ is a subset that is simultaneously a subgroup and a **closed** submanifold. An **immersed subgroup** is the image of a Lie group $H$ under an injective morphism to $G$ (with everywhere injective differential).

The distinction matters because the topology of an immersed submanifold may not agree with the subspace topology. For example, a line of irrational slope in the torus $\mathbb{R}^2/\mathbb{Z}^2 = S^1 \times S^1$ is an immersed subgroup that is dense but not closed.

</div>

### 7.2 Examples of Lie Groups

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(General Linear Group and Representations)</span></p>

The basic example is the **general linear group** $\mathrm{GL}_n\mathbb{R}$ of invertible $n \times n$ real matrices, an open subset of the vector space of all $n \times n$ matrices. The multiplication map is differentiable (polynomial in entries), and the inverse is differentiable (by Cramer's formula). We also write $\mathrm{GL}(V)$ or $\mathrm{Aut}(V)$ for the group of automorphisms of a real vector space $V$.

A **representation** of a Lie group $G$ is a morphism from $G$ to $\mathrm{GL}(V)$ for some finite-dimensional real or complex vector space $V$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classical Real Lie Groups)</span></p>

Most Lie groups arise as subgroups of $\mathrm{GL}_n\mathbb{R}$ defined by preserving some structure:

- **Special linear group** $\mathrm{SL}_n\mathbb{R}$: matrices with $\det A = 1$.
- **Borel subgroup** $B_n$: upper-triangular invertible matrices; preserves the standard complete flag $0 \subset V_1 \subset \cdots \subset V_n = \mathbb{R}^n$.
- **Unipotent subgroup** $N_n$: upper-triangular matrices with 1's on the diagonal.
- **(Special) orthogonal group** $\mathrm{SO}_n\mathbb{R}$: matrices preserving the standard symmetric bilinear form $Q(v,w) = {}^tv \cdot w$ and having determinant 1, i.e., ${}^tA \cdot A = I$, $\det A = 1$. More generally, $\mathrm{SO}_{k,l}\mathbb{R}$ preserves a symmetric form of signature $(k, l)$.
- **Symplectic group** $\mathrm{Sp}_{2n}\mathbb{R}$: matrices preserving a nondegenerate skew-symmetric bilinear form. Note $n$ must be even (written as $2n$).
- **Unitary group** $\mathrm{U}(n)$: complex linear automorphisms of $\mathbb{C}^n$ preserving the standard positive definite Hermitian form $H(v,w) = {}^t\bar{v} \cdot w$; equivalently ${}^t\bar{A} \cdot A = I$. The **special unitary group** is $\mathrm{SU}(n) = \mathrm{U}(n) \cap \mathrm{SL}_n\mathbb{C}$.

The condition for preserving a bilinear form $Q$ represented by a matrix $M$ is ${}^tA \cdot M \cdot A = M$. For a Hermitian form: ${}^t\bar{A} \cdot M \cdot A = M$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Quaternionic Groups)</span></p>

The group $\mathrm{GL}_n\mathbb{H}$ of quaternionic linear automorphisms of $\mathbb{H}^n$ is a real Lie subgroup of $\mathrm{GL}_{2n}\mathbb{C}$. We view $\mathbb{H} = \mathbb{C} \oplus j\mathbb{C} \cong \mathbb{C}^2$, so $\mathbb{H}^n \cong \mathbb{C}^{2n}$, and a $\mathbb{C}$-linear map $\varphi\colon \mathbb{C}^{2n} \to \mathbb{C}^{2n}$ is $\mathbb{H}$-linear exactly when it commutes with multiplication by $j$, which acts as the matrix $J = \begin{pmatrix} 0 & -I_n \\ I_n & 0 \end{pmatrix}$. Thus $\mathrm{GL}_n\mathbb{H} = \lbrace A \in \mathrm{GL}_{2n}\mathbb{C} \colon AJ = J\bar{A} \rbrace$.

The **compact symplectic group** $\mathrm{Sp}(n) = \mathrm{U}_\mathbb{H}(n)$ preserves the standard quaternionic Hermitian form $K(v,w) = \sum \bar{v}_i w_i$ on $\mathbb{H}^n$. One has the relation

$$\mathrm{Sp}(n) = \mathrm{U}(2n) \cap \mathrm{Sp}_{2n}\mathbb{C}.$$

This shows that the two notions of "symplectic" are compatible.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Complex Lie Groups)</span></p>

The complex Lie groups are fewer in number than the real ones. The elementary examples all arise as subgroups of $\mathrm{GL}_n\mathbb{C}$:

- $\mathrm{SL}_n\mathbb{C}$: determinant 1.
- $\mathrm{SO}_n\mathbb{C} \subset \mathrm{SL}_n\mathbb{C}$: preserves a nondegenerate symmetric bilinear form $Q$. Since all such forms are isomorphic over $\mathbb{C}$, there is only one orthogonal subgroup up to conjugation (no signature).
- $\mathrm{Sp}_{2n}\mathbb{C}$: preserves a nondegenerate skew-symmetric bilinear form.

Note that $\mathrm{SU}(n) \subset \mathrm{SL}_n\mathbb{C}$ is **not** a complex Lie subgroup (the defining equations involve conjugation and are not holomorphic). Any compact complex Lie group is abelian. A **representation** of a complex Lie group $G$ is a map of complex Lie groups from $G$ to $\mathrm{GL}_n\mathbb{C}$.

</div>

### 7.3 Two Constructions

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.9 — Covering Spaces and Lie Group Structure)</span></p>

Let $G$ be a Lie group, $H$ a connected manifold, and $\varphi\colon H \to G$ a covering space map. Let $e'$ be an element lying over the identity $e$ of $G$. Then there is a **unique** Lie group structure on $H$ such that $e'$ is the identity and $\varphi$ is a map of Lie groups; the kernel of $\varphi$ is in the center of $H$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.10 — Quotient by Discrete Central Subgroup)</span></p>

Let $H$ be a Lie group, and $\Gamma \subset Z(H)$ a discrete subgroup of its center. Then the quotient group $G = H/\Gamma$ has a unique Lie group structure such that the quotient map $H \to G$ is a Lie group map.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Isogeny, Simply Connected Form, Adjoint Form)</span></p>

A Lie group map between two Lie groups $G$ and $H$ is an **isogeny** if it is a covering space map of the underlying manifolds. Two Lie groups are **isogenous** if there is an isogeny between them (in either direction).

Every isogeny equivalence class has an initial member: the **universal covering space** $\tilde{G}$ of any one member, called the **simply connected form**. If the center $Z(\tilde{G})$ is discrete, then $\tilde{G}/Z(\tilde{G})$ is a final object called the **adjoint form**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Centers and Quotients of Classical Groups)</span></p>

The center of $\mathrm{SL}_n$ (over $\mathbb{R}$ or $\mathbb{C}$) is the subgroup of scalar multiples of the identity by $n$th roots of unity. The quotient is denoted $\mathrm{PSL}_n\mathbb{R}$ or $\mathrm{PSL}_n\mathbb{C}$. In the complex case, $\mathrm{PSL}_n\mathbb{C}$ is isomorphic to the quotient of $\mathrm{GL}_n\mathbb{C}$ by its center $\mathbb{C}^*$, so one often writes $\mathrm{PGL}_n\mathbb{C}$ instead.

The center of $\mathrm{SO}_n$ is $\lbrace \pm I \rbrace$ when $n$ is even, and trivial when $n$ is odd; the quotient is $\mathrm{PSO}_n$. The center of $\mathrm{Sp}_{2n}$ is $\lbrace \pm I \rbrace$, giving $\mathrm{PSp}_{2n}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Spin Groups and Exceptional Coverings)</span></p>

The orthogonal groups $\mathrm{SO}_n\mathbb{R}$ and $\mathrm{SO}_n\mathbb{C}$ have fundamental group $\mathbb{Z}/2$, so there exist connected, two-sheeted covers called the **spin groups** $\mathrm{Spin}_n\mathbb{R}$ and $\mathrm{Spin}_n\mathbb{C}$ (to be discussed in Lecture 20).

In low dimensions, several exceptional isogenies arise:

$$\mathrm{SU}(2) = \mathrm{Sp}(1) \xrightarrow{2:1} \mathrm{SO}(3),$$

$$\mathrm{SL}_2\mathbb{C} \times \mathrm{SL}_2\mathbb{C} \xrightarrow{2:1} \mathrm{SO}_4\mathbb{C},$$

$$\mathrm{SL}_2\mathbb{C} \xrightarrow{2:1} \mathrm{SO}_3\mathbb{C}.$$

The first is realized by identifying $\mathbb{R}^3$ with the imaginary quaternions and letting $q \in \mathrm{SU}(2)$ act by $v \mapsto qv\bar{q}$. The second uses the action of $\mathrm{SL}_2\mathbb{C}$ on $M_2\mathbb{C} = \mathbb{C}^4$ (the space of $2 \times 2$ matrices with the symmetric form $Q = \frac{1}{4}\mathrm{Trace}(AB^a)$ where $B^a$ is the adjugate) by $A \mapsto gAh^{-1}$.

</div>

---

## Lecture 8: Lie Algebras and Lie Groups

This crucial lecture introduces the definition of the Lie algebra associated to a Lie group and establishes their relationship. The key results are the **First Principle** (a map of connected Lie groups is determined by its differential at the identity) and the **Second Principle** (a linear map between Lie algebras is the differential of a Lie group map if and only if it preserves the bracket). The **exponential map** provides the bridge.

### 8.1 Lie Algebras: Motivation and Definition

Given that we want to study representations of a Lie group, the strategy is to use the continuous structure of the group. A map $\rho\colon G \to H$ between connected Lie groups will be determined by what it does on any open set containing the identity, i.e., by its germ at $e \in G$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(First Principle)</span></p>

*First Principle:* Let $G$ and $H$ be Lie groups, with $G$ connected. A map $\rho\colon G \to H$ is uniquely determined by its differential $d\rho_e\colon T_eG \to T_eH$ at the identity.

</div>

To understand what additional structure a homomorphism $\rho$ imposes on the differential $d\rho_e$, we proceed through several characterizations:

1. A homomorphism $\rho$ respects the action of a group $G$ on itself by **left or right multiplication**: $\rho \circ m_g = m_{\rho(g)} \circ \rho$.
2. Equivalently, $\rho$ respects the **conjugation** maps $\Psi_g(h) = ghg^{-1}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Adjoint Representation)</span></p>

For any $g \in G$, the conjugation map $\Psi_g\colon G \to G$ fixes $e$, so its differential at $e$ gives a linear map on the tangent space:

$$\mathrm{Ad}(g) = (d\Psi_g)_e\colon T_eG \to T_eG. \tag{8.2}$$

This defines a representation

$$\mathrm{Ad}\colon G \to \mathrm{Aut}(T_eG) \tag{8.3}$$

called the **adjoint representation** of the group $G$ on its own tangent space.

</div>

The condition that a homomorphism $\rho$ respects conjugation translates to:

$$d\rho_e(\mathrm{Ad}(g)(v)) = \mathrm{Ad}(\rho(g))(d\rho_e(v)) \tag{8.4}$$

for any tangent vector $v \in T_eG$. Taking the differential of Ad itself at the identity yields a map

$$\mathrm{ad}\colon T_eG \to \mathrm{End}(T_eG), \tag{8.5}$$

and the condition on $d\rho_e$ becomes

$$d\rho_e(\mathrm{ad}(X)(Y)) = \mathrm{ad}(d\rho_e(X))(d\rho_e(Y)). \tag{8.7}$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lie Bracket)</span></p>

The **Lie bracket** on $T_eG$ is defined by

$$[X, Y] \stackrel{\mathrm{def}}{=} \mathrm{ad}(X)(Y). \tag{8.6}$$

This is a bilinear map $T_eG \times T_eG \to T_eG$.

For the general linear group $G = \mathrm{GL}_n\mathbb{R}$, where $T_eG = \mathrm{End}(\mathbb{R}^n) = M_n\mathbb{R}$ and $\mathrm{Ad}(g)(M) = gMg^{-1}$, the bracket is simply the matrix commutator:

$$[X, Y] = X \cdot Y - Y \cdot X.$$

In general, for any Lie group given as a subgroup of $\mathrm{GL}_n\mathbb{R}$, the bracket on its tangent space (viewed as a subspace of $M_n\mathbb{R}$) coincides with the commutator.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lie Algebra)</span></p>

A **Lie algebra** $\mathfrak{g}$ is a vector space together with a skew-symmetric bilinear map

$$[\,,\,]\colon \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g}$$

satisfying the **Jacobi identity**: for any three tangent vectors $X$, $Y$, $Z$,

$$[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0.$$

The skew-symmetry and Jacobi identity both follow from the description of the bracket as a commutator. A **map** of Lie algebras is a linear map preserving the bracket. A **Lie subalgebra** is a vector subspace closed under the bracket.

We write $\mathfrak{gl}(V)$ for the Lie algebra $\mathrm{End}(V)$ with the commutator bracket.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Second Principle — Preview)</span></p>

*Second Principle:* Let $G$ and $H$ be Lie groups, with $G$ connected and simply connected. A linear map $T_eG \to T_eH$ is the differential of a homomorphism $\rho\colon G \to H$ **if and only if** it preserves the bracket operation, in the sense of (8.8):

$$d\rho_e([X, Y]) = [d\rho_e(X),\, d\rho_e(Y)].$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Representation of a Lie Algebra)</span></p>

A **representation** of a Lie algebra $\mathfrak{g}$ on a vector space $V$ is simply a map of Lie algebras

$$\rho\colon \mathfrak{g} \to \mathfrak{gl}(V) = \mathrm{End}(V),$$

i.e., a linear map such that $\rho([X, Y]) = \rho(X)\rho(Y) - \rho(Y)\rho(X)$, or equivalently, an action of $\mathfrak{g}$ on $V$ such that $[X, Y](v) = X(Y(v)) - Y(X(v))$.

The Second Principle implies that **representations of a connected and simply connected Lie group are in one-to-one correspondence with representations of its Lie algebra**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lie Algebra Actions on Tensors)</span></p>

The relation between group and Lie algebra representations on tensors differs in an important way. If $V$ and $W$ are group representations, $g$ acts on $V \otimes W$ by $g(v \otimes w) = g(v) \otimes g(w)$. But for Lie algebra representations, differentiating gives the **Leibniz rule**:

$$X(v \otimes w) = X(v) \otimes w + v \otimes X(w). \tag{8.12}$$

Similarly, on $\mathrm{Sym}^2 V$: the group acts by $g(v^2) = g(v)^2$, but the Lie algebra acts by $X(v^2) = 2v \cdot X(v)$.

For the **dual representation**: if $\rho\colon G \to \mathrm{GL}(V)$ gives a representation with $\rho'(g) = {}^t\rho(g^{-1})$ on $V^*$, then the Lie algebra representation is

$$\rho'(X) = -{}^t\rho(X)\colon V^* \to V^*. \tag{8.14}$$

A Lie algebra $\mathfrak{g}$ acting on $V$ **preserves** a structure (e.g., a bilinear form $Q$) on $V$ if the induced action on $\mathrm{Sym}^2 V^*$ kills $Q$:

$$Q(v, X(w)) + Q(X(v), w) = 0 \tag{8.15}$$

for all $X \in \mathfrak{g}$ and $v, w \in V$.

</div>

### 8.2 Examples of Lie Algebras

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Lie Algebras of Classical Groups)</span></p>

The Lie algebras of the groups introduced in Lecture 7 are all subspaces of $\mathfrak{gl}_n\mathbb{R} = M_n\mathbb{R}$ (or $\mathfrak{gl}_n\mathbb{C}$):

- $\mathfrak{sl}_n\mathbb{R} = \lbrace X \in M_n\mathbb{R} \colon \mathrm{Trace}(X) = 0 \rbrace$ — traceless $n \times n$ matrices.
- $\mathfrak{so}_n\mathbb{R} = \lbrace X \in M_n\mathbb{R} \colon Q(X(v), w) + Q(v, X(w)) = 0 \rbrace$. If $Q(v,w) = {}^tv \cdot M \cdot w$ with $M$ symmetric, the condition on $X$ is

$${}^tX \cdot M + M \cdot X = 0. \tag{8.21}$$

For the standard form $M = I$, this is ${}^tX = -X$: the **skew-symmetric** matrices. Intrinsically, $\mathfrak{so}_n\mathbb{R} = \bigwedge^2 V \subset V \otimes V = \mathrm{End}(V)$.

- $\mathfrak{sp}_{2n}\mathbb{R} = \lbrace X \in M_{2n}\mathbb{R} \colon Q(X(v), w) + Q(v, X(w)) = 0 \rbrace$, where $Q$ is skew-symmetric. The condition becomes ${}^tX \cdot M + M \cdot X = 0$ with $M$ skew-symmetric, which amounts to $Q(X(v), w) = Q(X(w), v)$, so intrinsically $\mathfrak{sp}_{2n}\mathbb{R} = \mathrm{Sym}^2 V \subset V \otimes V = \mathrm{End}(V)$.

- $\mathfrak{u}(n) = \lbrace X \in M_n\mathbb{C} \colon {}^t\bar{X} = -X \rbrace$ — skew-Hermitian matrices.

The complex versions $\mathfrak{sl}_n\mathbb{C}$, $\mathfrak{so}_n\mathbb{C}$, $\mathfrak{sp}_{2n}\mathbb{C}$ are defined identically but over $\mathbb{C}$.

- $\mathfrak{b}_n\mathbb{R}$: upper triangular matrices; $\mathfrak{n}_n\mathbb{R}$: strictly upper triangular matrices.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Derivations and Alternative Descriptions)</span></p>

If $A$ is any (real or complex) algebra, a **derivation** is a linear map $D\colon A \to A$ satisfying the Leibniz rule $D(ab) = aD(b) + D(a)b$. The space $\mathrm{Der}(A)$ forms a Lie algebra under $[D, E] = D \circ E - E \circ D$.

For any Lie algebra $\mathfrak{g}$, the map $\mathfrak{g} \to \mathrm{Der}(\mathfrak{g})$ given by $X \mapsto D_X$, where $D_X(Y) = [X, Y]$, is a map of Lie algebras. The Lie algebra of the automorphism group $\mathrm{Aut}(\mathfrak{g})$ is $\mathrm{Der}(\mathfrak{g})$.

There is also a description via **left-invariant vector fields**: for $X \in \mathfrak{g}$, define a vector field $v_X$ on $G$ by $v_X(g) = (m_g)_*(X)$. The $\mathscr{C}^\infty$ vector fields on $G$ form a Lie algebra under the Lie bracket of vector fields, and the left-invariant ones form a finite-dimensional Lie subalgebra isomorphic to $\mathfrak{g}$.

</div>

### 8.3 The Exponential Map

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(One-Parameter Subgroup and Exponential Map)</span></p>

For $X \in \mathfrak{g}$, the left-invariant vector field $v_X$ can be integrated: there exists a unique homomorphism

$$\varphi_X\colon \mathbb{R} \to G$$

such that $\varphi_X'(t) = v_X(\varphi_X(t))$ for all $t$, with $\varphi_X(0) = e$ and tangent vector $\varphi_X'(0) = X$. This is called the **one-parameter subgroup** of $G$ with tangent vector $X$ at the identity.

The **exponential map** $\exp\colon \mathfrak{g} \to G$ is defined by

$$\exp(X) = \varphi_X(1). \tag{8.32}$$

Note that $\varphi_{\lambda X}(t) = \varphi_X(\lambda t)$, so the restriction of $\exp$ to each line through the origin in $\mathfrak{g}$ gives a one-parameter subgroup.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(8.33 — Characterization of the Exponential Map)</span></p>

The exponential map is the unique map from $\mathfrak{g}$ to $G$ taking $0$ to $e$ whose differential at the origin

$$(\exp_*)_0\colon T_0\mathfrak{g} = \mathfrak{g} \to T_eG = \mathfrak{g}$$

is the identity, and whose restrictions to the lines through the origin in $\mathfrak{g}$ are one-parameter subgroups.

The exponential map is **natural**: for any map $\psi\colon G \to H$ of Lie groups, the diagram

$$\mathfrak{g} \xrightarrow{\psi_*} \mathfrak{h} \qquad \exp \downarrow \quad \downarrow \exp \qquad G \xrightarrow{\psi} H$$

commutes. This naturality, together with the First Principle, implies that if $G$ is connected, a map $\psi$ is determined by its differential $(d\psi)_e$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Explicit Formula for Matrix Groups)</span></p>

For any subgroup of $\mathrm{GL}_n\mathbb{R}$ (or $\mathrm{GL}_n\mathbb{C}$), the exponential map is given by the standard power series: for $X \in \mathrm{End}(V)$,

$$\exp(X) = I + X + \frac{X^2}{2} + \frac{X^3}{6} + \cdots = \sum_{k=0}^{\infty} \frac{X^k}{k!}. \tag{8.34}$$

This always converges and is invertible (with inverse $\exp(-X)$). Its differential at the origin is the identity, and it restricts to one-parameter subgroups on lines, so it agrees with the abstract definition. For any subgroup $G \subset \mathrm{GL}_n\mathbb{R}$, the exponential map of $G$ is the restriction of (8.34).

</div>

#### The Campbell–Hausdorff Formula

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Campbell–Hausdorff Formula)</span></p>

For $X$ and $Y$ in a sufficiently small neighborhood of the origin in $\mathfrak{g}$, define

$$X * Y = \log(\exp(X) \cdot \exp(Y)),$$

where $\log$ is the local inverse of $\exp$. Then $X * Y$ can be expressed purely in terms of $X$, $Y$, and the bracket operation. To degree three:

$$X * Y = X + Y + \tfrac{1}{2}[X, Y] + \tfrac{1}{12}[X, [X, Y]] + \tfrac{1}{12}[Y, [Y, X]] + \cdots$$

The key point is that $\log(\exp(X) \cdot \exp(Y))$ can be expressed purely in terms of $X$, $Y$, and iterated brackets — no reference to the ambient algebra $\mathrm{End}(V)$ is needed. This makes precise the claim that **the group structure of $G$ is encoded in the Lie algebra $\mathfrak{g}$**.

</div>

#### Lie Subalgebras and Lie Subgroups

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(8.41 — Lie Subalgebras Generate Lie Subgroups)</span></p>

Let $G$ be a Lie group, $\mathfrak{g}$ its Lie algebra, and $\mathfrak{h} \subset \mathfrak{g}$ a Lie subalgebra. Then the subgroup of $G$ generated by $\exp(\mathfrak{h})$ is an immersed Lie subgroup $H$ with tangent space $T_eH = \mathfrak{h}$.

Moreover, every finite-dimensional Lie algebra is the Lie algebra of some Lie group (this follows from Ado's theorem, which embeds any Lie algebra into $\mathfrak{gl}_n$, combined with the above).

</div>

#### Proof of the Second Principle

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Second Principle)</span></p>

Let $G$ and $H$ be Lie groups with $G$ simply connected, and let $\mathfrak{g}$ and $\mathfrak{h}$ be their Lie algebras. A linear map $\alpha\colon \mathfrak{g} \to \mathfrak{h}$ is the differential of a map $A\colon G \to H$ of Lie groups **if and only if** $\alpha$ is a map of Lie algebras.

**Proof sketch.** Let $\mathfrak{j} = \mathrm{graph}(\alpha) \subset \mathfrak{g} \oplus \mathfrak{h}$. The hypothesis that $\alpha$ preserves brackets is equivalent to $\mathfrak{j}$ being a Lie subalgebra of $\mathfrak{g} \oplus \mathfrak{h}$. By Proposition 8.41, there exists an immersed Lie subgroup $J \subset G \times H$ with $T_eJ = \mathfrak{j}$. The projection $\pi\colon J \to G$ on the first factor has differential $d\pi_e\colon \mathfrak{j} \to \mathfrak{g}$ which is an isomorphism (since $\alpha$ is a map, $\mathfrak{j}$ projects isomorphically onto $\mathfrak{g}$). Since $G$ is simply connected, $\pi$ is an isomorphism. The projection $\eta\colon G \cong J \to H$ on the second factor is then a Lie group map whose differential at the identity is $\alpha$. $\square$

</div>

---

## Lecture 9: Initial Classification of Lie Algebras

In this lecture we define various subclasses of Lie algebras: nilpotent, solvable, semisimple, etc., and prove basic facts about their representations. The discussion is entirely elementary (largely because the hard theorems are stated without proof for now); there are no prerequisites beyond linear algebra. The purpose is to motivate the narrowing of our focus to semisimple algebras.

### 9.1 Rough Classification of Lie Algebras

We begin with a preliminary classification of Lie algebras, reflecting the degree to which a given Lie algebra $\mathfrak{g}$ fails to be abelian. The goal ultimately is to narrow our focus onto *semisimple* Lie algebras.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Center, Abelian Lie Algebra)</span></p>

The **center** $Z(\mathfrak{g})$ of a Lie algebra $\mathfrak{g}$ is the subspace of elements $X \in \mathfrak{g}$ such that $[X, Y] = 0$ for all $Y \in \mathfrak{g}$. We say $\mathfrak{g}$ is **abelian** if all brackets are zero, i.e., $Z(\mathfrak{g}) = \mathfrak{g}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ideal of a Lie Algebra)</span></p>

A Lie subalgebra $\mathfrak{h} \subset \mathfrak{g}$ is an **ideal** if it satisfies

$$[X, Y] \in \mathfrak{h} \quad \text{for all } X \in \mathfrak{h},\, Y \in \mathfrak{g}.$$

Just as connected subgroups of a Lie group correspond to subalgebras of its Lie algebra, the notion of ideal in a Lie algebra corresponds to the notion of normal subgroup: a connected subgroup $H \subset G$ is normal if and only if $\mathfrak{h}$ is an ideal of $\mathfrak{g}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Quotient Lie Algebras)</span></p>

The bracket operation on $\mathfrak{g}$ induces a bracket on the quotient space $\mathfrak{g}/\mathfrak{h}$ if and only if $\mathfrak{h}$ is an ideal in $\mathfrak{g}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Simple Lie Algebra)</span></p>

A Lie algebra $\mathfrak{g}$ is **simple** if $\dim \mathfrak{g} > 1$ and it contains no nontrivial ideals. Equivalently, the adjoint group $G$ of the Lie algebra $\mathfrak{g}$ has no nontrivial normal Lie subgroups.

</div>

#### Descending Series

To classify Lie algebras, we introduce two descending chains of subalgebras.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lower Central Series)</span></p>

The **lower central series** of subalgebras $\mathscr{D}_k\mathfrak{g}$ is defined inductively by

$$\mathscr{D}_1\mathfrak{g} = [\mathfrak{g}, \mathfrak{g}]$$

and

$$\mathscr{D}_k\mathfrak{g} = [\mathfrak{g}, \mathscr{D}_{k-1}\mathfrak{g}].$$

The subalgebras $\mathscr{D}_k\mathfrak{g}$ are in fact ideals in $\mathfrak{g}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Derived Series)</span></p>

The **derived series** $\mathscr{D}^k\mathfrak{g}$ is defined by

$$\mathscr{D}^1\mathfrak{g} = [\mathfrak{g}, \mathfrak{g}]$$

and

$$\mathscr{D}^k\mathfrak{g} = [\mathscr{D}^{k-1}\mathfrak{g},\, \mathscr{D}^{k-1}\mathfrak{g}].$$

We have $\mathscr{D}^k\mathfrak{g} \subset \mathscr{D}_k\mathfrak{g}$ for all $k$, with equality when $k = 1$; we often write simply $\mathscr{D}\mathfrak{g} = \mathscr{D}_1\mathfrak{g} = \mathscr{D}^1\mathfrak{g}$ and call this the **commutator subalgebra**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nilpotent, Solvable, Perfect, Semisimple)</span></p>

1. $\mathfrak{g}$ is **nilpotent** if $\mathscr{D}_k\mathfrak{g} = 0$ for some $k$.
2. $\mathfrak{g}$ is **solvable** if $\mathscr{D}^k\mathfrak{g} = 0$ for some $k$.
3. $\mathfrak{g}$ is **perfect** if $\mathscr{D}\mathfrak{g} = \mathfrak{g}$ (this is not a concept we will use much).
4. $\mathfrak{g}$ is **semisimple** if $\mathfrak{g}$ has no nonzero solvable ideals.

</div>

#### Standard Examples

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Nilpotent: Strictly Upper-Triangular Matrices)</span></p>

The standard example of a nilpotent Lie algebra is $\mathfrak{n}_n\mathbb{R}$, the algebra of strictly upper-triangular $n \times n$ matrices. The $k$th subalgebra $\mathscr{D}_k\mathfrak{g}$ in the lower central series is the subspace $\mathfrak{n}_{k+1,n}\mathbb{R}$ of matrices $(a_{i,j})$ with $a_{i,j} = 0$ whenever $j \le i + k$, i.e., matrices that are zero within a distance $k$ of the diagonal. Any subalgebra of $\mathfrak{n}_n\mathbb{R}$ is likewise nilpotent, and any nilpotent Lie algebra is isomorphic to such a subalgebra.

If a Lie algebra $\mathfrak{g}$ is represented on a vector space $V$ such that each element acts as a nilpotent endomorphism, there exists a basis for $V$ such that $\mathfrak{g}$ maps to the subalgebra $\mathfrak{n}_n\mathbb{R} \subset \mathfrak{gl}_n\mathbb{R}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Solvable: Upper-Triangular Matrices)</span></p>

A standard example of a solvable Lie algebra is $\mathfrak{b}_n\mathbb{R}$, the space of upper-triangular $n \times n$ matrices. The commutator $\mathscr{D}\mathfrak{b}_n\mathbb{R}$ is the algebra $\mathfrak{n}_n\mathbb{R}$, and the derived series is $\mathscr{D}^k\mathfrak{b}_n\mathbb{R} = \mathfrak{n}_{2^{k-1},n}\mathbb{R}$. Any subalgebra of $\mathfrak{b}_n\mathbb{R}$ is likewise solvable. We will show later that, conversely, *any* representation of a solvable Lie algebra on a vector space $V$ can be put in upper-triangular form with respect to a suitable basis.

</div>

#### Properties of Solvable Lie Algebras

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Characterization of Solvability)</span></p>

$\mathfrak{g}$ is solvable if and only if $\mathfrak{g}$ has a sequence of Lie subalgebras $\mathfrak{g} = \mathfrak{g}_0 \supset \mathfrak{g}_1 \supset \cdots \supset \mathfrak{g}_k = 0$ such that $\mathfrak{g}_{i+1}$ is an ideal in $\mathfrak{g}_i$ and $\mathfrak{g}_i / \mathfrak{g}_{i+1}$ is abelian. It follows that if $\mathfrak{h}$ is an ideal in a Lie algebra $\mathfrak{g}$, then $\mathfrak{g}$ is solvable **if and only if** $\mathfrak{h}$ and $\mathfrak{g}/\mathfrak{h}$ are solvable Lie algebras.

The properties of being nilpotent or solvable are inherited by subalgebras or homomorphic images. For semisimplicity this is true in the case of homomorphic images, though not for subalgebras.

</div>

#### The Radical and the Levi Decomposition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Radical)</span></p>

The sum of two solvable ideals in a Lie algebra $\mathfrak{g}$ is again solvable. It follows that the sum of all solvable ideals in $\mathfrak{g}$ is a maximal solvable ideal, called the **radical** of $\mathfrak{g}$ and denoted $\mathrm{Rad}(\mathfrak{g})$. The quotient $\mathfrak{g}/\mathrm{Rad}(\mathfrak{g})$ is semisimple.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Exact Sequence and Levi Decomposition)</span></p>

Any Lie algebra $\mathfrak{g}$ fits into an exact sequence

$$0 \to \mathrm{Rad}(\mathfrak{g}) \to \mathfrak{g} \to \mathfrak{g}/\mathrm{Rad}(\mathfrak{g}) \to 0$$

where the first algebra is solvable and the last is semisimple. To study the representation theory of an arbitrary Lie algebra, we thus need to understand individually the representation theories of solvable and semisimple Lie algebras. Of these, the former is relatively easy — any irreducible representation of a solvable Lie algebra is one-dimensional — while the latter is extraordinarily rich and occupies most of the remainder of this book.

The existence of subalgebras of $\mathfrak{g}$ that map isomorphically onto $\mathfrak{g}/\mathrm{Rad}(\mathfrak{g})$ is called a **Levi decomposition** and is part of the general theory.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Semisimplicity and Abelian Ideals)</span></p>

A Lie algebra is semisimple **if and only if** it has no nonzero abelian ideals. Indeed, the last nonzero term in the derived sequence of ideals $\mathscr{D}^k\mathrm{Rad}(\mathfrak{g})$ would be an abelian ideal. A semisimple Lie algebra can have no center, so the adjoint representation of a semisimple Lie algebra is faithful.

</div>

### 9.2 Engel's Theorem and Lie's Theorem

We now prove the statement made above about representations of solvable Lie algebras always being upper triangular. We start with the foundational result about nilpotent actions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.9 — Engel's Theorem)</span></p>

Let $\mathfrak{g} \subset \mathfrak{gl}(V)$ be any Lie subalgebra such that every $X \in \mathfrak{g}$ is a nilpotent endomorphism of $V$. Then there exists a nonzero vector $v \in V$ such that $X(v) = 0$ for all $X \in \mathfrak{g}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequence of Engel's Theorem)</span></p>

Engel's theorem implies that there exists a basis for $V$ in terms of which the matrix representative of each $X \in \mathfrak{g}$ is strictly upper triangular: since $\mathfrak{g}$ kills $v$, it will act on the quotient $\bar{V} = V / \langle v \rangle$, and by induction we can find a basis $\bar{v}_2, \dots, \bar{v}_n$ for $\bar{V}$ in terms of which this action is strictly upper triangular. Lifting $\bar{v}_i$ to $v_i \in V$ and setting $v_1 = v$ gives a basis for $V$ as desired.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Engel's Theorem)</span></p>

**Step 1.** If $X \in \mathfrak{gl}(V)$ is nilpotent, then $\mathrm{ad}(X)\colon \mathfrak{gl}(V) \to \mathfrak{gl}(V)$ is also nilpotent. This is straightforward: nilpotency of $X$ means there is a flag $0 \subset V_1 \subset \cdots \subset V_k = V$ with $X(V_i) \subset V_{i-1}$, and one checks that $\mathrm{ad}(X)^m(Y)$ carries $V_i$ into $V_{i+k-m}$.

**Step 2.** By induction on $\dim \mathfrak{g}$, we show that $\mathfrak{g}$ contains an ideal $\mathfrak{h}$ of codimension one. Let $\mathfrak{h} \subset \mathfrak{g}$ be any maximal proper subalgebra. Since $\mathfrak{h}$ is a subalgebra, the adjoint action $\mathrm{ad}(\mathfrak{h})$ preserves $\mathfrak{h}$ and so acts on $\mathfrak{g}/\mathfrak{h}$. By induction (since every $X \in \mathfrak{h}$ has $\mathrm{ad}(X)$ nilpotent), there exists a nonzero element $\bar{Y} \in \mathfrak{g}/\mathfrak{h}$ killed by $\mathrm{ad}(X)$ for all $X \in \mathfrak{h}$; equivalently, there exists $Y \in \mathfrak{g} \setminus \mathfrak{h}$ such that $[X, Y] \in \mathfrak{h}$ for all $X \in \mathfrak{h}$. The subalgebra spanned by $\mathfrak{h}$ and $Y$ then contains $\mathfrak{h}$ as a codimension-one ideal. By maximality of $\mathfrak{h}$, this subalgebra must be all of $\mathfrak{g}$, so $\mathfrak{h}$ is an ideal of codimension one.

**Step 3.** Apply the induction hypothesis to the ideal $\mathfrak{h}$ to conclude there exists a nonzero $v \in V$ with $X(v) = 0$ for all $X \in \mathfrak{h}$. Let $W = \lbrace v \in V : X(v) = 0 \text{ for all } X \in \mathfrak{h} \rbrace$. For any element $Y$ not in $\mathfrak{h}$ (which spans $\mathfrak{g}$ together with $\mathfrak{h}$), $Y$ carries $W$ into itself: for any $w \in W$ and $X \in \mathfrak{h}$, we have $X(Y(w)) = Y(X(w)) + [X, Y](w) = 0$ since $X(w) = 0$ and $[X, Y] \in \mathfrak{h}$. Since $Y$ acts nilpotently on $V$, it acts nilpotently on $W$, so there exists $v \in W$ with $Y(v) = 0$. This $v$ is killed by all of $\mathfrak{g}$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Nilpotency via the Adjoint Representation)</span></p>

A Lie algebra $\mathfrak{g}$ is nilpotent if and only if $\mathrm{ad}(X)$ is a nilpotent endomorphism of $\mathfrak{g}$ for every $X \in \mathfrak{g}$.

</div>

Engel's theorem allows us to prove that every representation of a solvable Lie group can be put in upper-triangular form. This is implied by:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.11 — Lie's Theorem)</span></p>

Let $\mathfrak{g} \subset \mathfrak{gl}(V)$ be a complex solvable Lie algebra. Then there exists a nonzero vector $v \in V$ that is an eigenvector of $X$ for all $X \in \mathfrak{g}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequence of Lie's Theorem)</span></p>

Lie's theorem implies the existence of a basis for $V$ in terms of which the matrix representative of each $X \in \mathfrak{g}$ is upper triangular.

</div>

The key ingredient in the proof of Lie's theorem is the following lemma:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(9.13)</span></p>

Let $\mathfrak{h}$ be an ideal in a Lie algebra $\mathfrak{g}$. Let $V$ be a representation of $\mathfrak{g}$, and $\lambda\colon \mathfrak{h} \to \mathbb{C}$ a linear function. Set

$$W = \lbrace v \in V : X(v) = \lambda(X) \cdot v \;\; \forall\, X \in \mathfrak{h} \rbrace.$$

Then $Y(W) \subset W$ for all $Y \in \mathfrak{g}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Lemma 9.13)</span></p>

Let $w$ be any nonzero element of $W$; to test whether $Y(w) \in W$ we let $X$ be any element of $\mathfrak{h}$ and write

$$X(Y(w)) = Y(X(w)) + [X, Y](w) = \lambda(X) \cdot Y(w) + \lambda([X, Y]) \cdot w.$$

This differs from the Engel case: $Y(w)$ will lie in $W$ if and only if $\lambda([X, Y]) = 0$ for all $X \in \mathfrak{h}$.

To verify this, introduce the span $U$ of $w, Y(w), Y^2(w), \ldots$ One shows by induction that $\mathfrak{h}$ carries $Y^k(w)$ into $U$, using the relation

$$X(Y^k(w)) = Y(X(Y^{k-1}(w))) + [X, Y](Y^{k-1}(w)).$$

Since $X \in \mathfrak{h}$ by induction, $\mathfrak{h}$ carries $U$ into itself. For $U$ the action of any $X \in \mathfrak{h}$ is upper triangular with diagonal entries all equal to $\lambda(X)$, so $\mathrm{Tr}(X|_U) = \dim(U) \cdot \lambda(X)$. On the other hand, for any $X \in \mathfrak{h}$ the commutator $[X, Y]$ acts on $U$ and its trace is zero (being a commutator of endomorphisms of $U$). It follows that $\lambda([X, Y]) = 0$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Lie's Theorem)</span></p>

The first step is to assert that $\mathfrak{g}$ contains an ideal $\mathfrak{h}$ of codimension one. Since $\mathfrak{g}$ is solvable, $\mathscr{D}\mathfrak{g} \neq \mathfrak{g}$, so the quotient $\mathfrak{a} = \mathfrak{g}/\mathscr{D}\mathfrak{g}$ is a nonzero abelian Lie algebra; the inverse image of any codimension-one subspace of $\mathfrak{a}$ gives a codimension-one ideal in $\mathfrak{g}$.

By induction, there is a $v_0 \in V$ that is an eigenvector for all $X \in \mathfrak{h}$. Denote the eigenvalue of $X$ on $v_0$ by $\lambda(X)$. Set $W = \lbrace v \in V : X(v) = \lambda(X) \cdot v \;\; \forall\, X \in \mathfrak{h} \rbrace$. By Lemma 9.13, any $Y \in \mathfrak{g}$ carries $W$ into itself. Since $Y$ acts on $W$ over $\mathbb{C}$, it has an eigenvector in $W$, and this eigenvector is simultaneously an eigenvector for all of $\mathfrak{g}$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Irreducible Representations of Solvable Algebras)</span></p>

Any irreducible representation of a solvable Lie algebra $\mathfrak{g}$ is one-dimensional, and $\mathscr{D}\mathfrak{g}$ acts trivially.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(9.17)</span></p>

Let $\mathfrak{g}$ be a complex Lie algebra, $\mathfrak{g}_{ss} = \mathfrak{g}/\mathrm{Rad}(\mathfrak{g})$. Every irreducible representation of $\mathfrak{g}$ is of the form $V = V_0 \otimes L$, where $V_0$ is an irreducible representation of $\mathfrak{g}$ that is trivial on $\mathrm{Rad}(\mathfrak{g})$ (i.e., comes from a representation of $\mathfrak{g}_{ss}$), and $L$ is a one-dimensional representation.

</div>

### 9.3 Semisimple Lie Algebras

Many of the aspects of the representation theory of finite groups that were essential to our approach are no longer valid in the context of general Lie algebras and Lie groups. Most obvious is complete reducibility, which fails for Lie groups; another is the fact that the action of some element of a Lie algebra may be diagonalizable under one representation and not under another. If we restrict ourselves to semisimple Lie algebras, however, everything is once more as well behaved as possible.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.19 — Complete Reducibility)</span></p>

Let $V$ be a representation of the semisimple Lie algebra $\mathfrak{g}$ and $W \subset V$ a subspace invariant under the action of $\mathfrak{g}$. Then there exists a subspace $W' \subset V$ complementary to $W$ and invariant under $\mathfrak{g}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Jordan Decomposition)</span></p>

Recall the **Jordan decomposition**: any endomorphism $X$ of a complex vector space $V$ can be uniquely written in the form $X = X_s + X_n$ where $X_s$ is diagonalizable, $X_n$ is nilpotent, and the two commute. Moreover, $X_s$ and $X_n$ may be expressed as polynomials in $X$.

For an arbitrary Lie algebra $\mathfrak{g}$ and a representation $\rho\colon \mathfrak{g} \to \mathfrak{gl}_n\mathbb{C}$, the image $\rho(X)$ need not be diagonalizable. In general, nothing need be true about how $\rho(X)$ behaves with respect to the Jordan decomposition.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.20 — Preservation of Jordan Decomposition)</span></p>

Let $\mathfrak{g}$ be a semisimple Lie algebra. For any element $X \in \mathfrak{g}$, there exist $X_s$ and $X_n \in \mathfrak{g}$ such that for any representation $\rho\colon \mathfrak{g} \to \mathfrak{gl}(V)$ we have

$$\rho(X)_s = \rho(X_s) \quad \text{and} \quad \rho(X)_n = \rho(X_n).$$

In other words, if we think of $\rho$ as injective and $\mathfrak{g}$ accordingly as a Lie subalgebra of $\mathfrak{gl}(V)$, *the diagonalizable and nilpotent parts of any element $X$ of $\mathfrak{g}$ are again in $\mathfrak{g}$ and are independent of the particular representation* $\rho$.

</div>

#### A Digression on "The Unitary Trick"

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Unitary Trick)</span></p>

The statements above (complete reducibility and preservation of Jordan decomposition) can be proved readily for the representations of a compact Lie group: if the compact group $G$ acts on a vector space, there is a Hermitian metric on $V$ invariant under the action of $G$ (obtained by averaging an arbitrary metric). If $G$ fixes a subspace $W \subset V$, it fixes its orthogonal complement $W^\perp$ as well.

The key fact is that if $\mathfrak{g}$ is any complex semisimple Lie algebra, there exists a (unique) real Lie algebra $\mathfrak{g}_0$ with complexification $\mathfrak{g}_0 \otimes \mathbb{C} = \mathfrak{g}$, such that the simply connected form of the Lie algebra $\mathfrak{g}_0$ is a compact Lie group $G$. By restricting a given representation of $\mathfrak{g}$ to $\mathfrak{g}_0$, exponentiating to $G$, and using complete reducibility for $G$, we can deduce complete reducibility of the original representation. This is Weyl's famous **unitary trick**.

For example, to prove complete reducibility for $\mathfrak{sl}_n\mathbb{R}$:
1. Let $\rho'$ be the corresponding (complex) representation of $\mathfrak{sl}_n\mathbb{R}$.
2. By linearity extend $\rho'$ to a representation $\rho''$ of $\mathfrak{sl}_n\mathbb{C}$.
3. Restrict to $\mathfrak{su}_n \subset \mathfrak{sl}_n\mathbb{C}$.
4. Exponentiate to obtain a representation $\rho'''$ of the unitary group $\mathrm{SU}_n$.

Since $\mathrm{SU}_n$ is compact, a complementary invariant subspace $W'$ exists for $\mathrm{SU}_n$, and then by reversing the chain, $W'$ is invariant under $\mathfrak{sl}_n\mathbb{R}$.

</div>

### 9.4 Simple Lie Algebras

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.26 — Classification of Simple Lie Algebras)</span></p>

With five exceptions, every simple complex Lie algebra is isomorphic to either $\mathfrak{sl}_n\mathbb{C}$, $\mathfrak{so}_n\mathbb{C}$, or $\mathfrak{sp}_{2n}\mathbb{C}$ for some $n$.

The five exceptions are denoted $\mathfrak{g}_2$, $\mathfrak{f}_4$, $\mathfrak{e}_6$, $\mathfrak{e}_7$, and $\mathfrak{e}_8$. The algebras $\mathfrak{sl}_n\mathbb{C}$ (for $n > 1$), $\mathfrak{so}_n\mathbb{C}$ (for $n > 2$), and $\mathfrak{sp}_{2n}\mathbb{C}$ are commonly called the **classical Lie algebras** (and the corresponding groups the **classical Lie groups**); the other five are called the **exceptional Lie algebras**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Semisimple = Direct Sum of Simple)</span></p>

Every semisimple Lie algebra is a direct sum of simple Lie algebras. A Lie algebra is called **reductive** if its radical equals its center. Equivalently, $\mathfrak{g}$ is reductive if and only if $\mathscr{D}\mathfrak{g}$ is semisimple, if and only if $\mathfrak{g}$ is a product of a semisimple and an abelian Lie algebra.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Roadmap for the Remainder of the Book)</span></p>

The classification theorem for simple Lie algebras creates a dilemma in how we approach the subject. The plan adopted is:

1. Analyze in Lectures 11–13 a couple of examples, namely $\mathfrak{sl}_2\mathbb{C}$ and $\mathfrak{sl}_3\mathbb{C}$, on what may appear to be an ad hoc basis.
2. On the basis of these examples, propose in Lecture 14 a general paradigm for the study of representations of a simple (or semisimple) Lie algebra.
3. Proceed in Lectures 15–20 to carry out this analysis for the classical algebras $\mathfrak{sl}_n\mathbb{C}$, $\mathfrak{so}_n\mathbb{C}$, and $\mathfrak{sp}_{2n}\mathbb{C}$.
4. Give in Part IV and the appendices proofs for general simple Lie algebras, as well as the **Weyl character formula**.

</div>

---

## Lecture 10: Lie Algebras in Dimensions One, Two, and Three

To get a sense of what a Lie algebra is and what groups might be associated to it, we classify here all Lie algebras of dimension three or less. We work primarily with complex Lie algebras and Lie groups, but mention the real case as well. The analyses are completely elementary, with one exception: the classification of complex Lie groups associated to abelian Lie algebras involves the theory of complex tori.

### 10.1 Dimensions One and Two

#### Dimension One

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimension One)</span></p>

Any one-dimensional Lie algebra $\mathfrak{g}$ is clearly abelian, that is, $\mathbb{C}$ with all brackets zero. The simply connected Lie group with this Lie algebra is just the group $\mathbb{C}$ under addition; other connected Lie groups with Lie algebra $\mathfrak{g}$ are quotients of $\mathbb{C}$ by discrete subgroups $\Lambda \subset \mathbb{C}$. If $\Lambda$ has rank one, the quotient is $\mathbb{C}^*$ under multiplication.

Over the real numbers, the only real Lie algebra of dimension one is $\mathbb{R}$ with trivial bracket; the simply connected Lie group is $\mathbb{R}$ under addition, and the only other connected real Lie group is $\mathbb{R}/\mathbb{Z} \cong S^1$.

</div>

#### Dimension Two

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimension Two, Abelian Case)</span></p>

**Case 1: $\mathfrak{g}$ abelian.** The simply connected two-dimensional abelian complex Lie group is $\mathbb{C}^2$ under addition. The remaining connected Lie groups are quotients of $\mathbb{C}^2$ by discrete subgroups $\Lambda \subset \mathbb{C}^2$, which can have rank 1, 2, 3, or 4.

- **Rank 1:** $G \cong \mathbb{C}^* \times \mathbb{C}$.
- **Rank 2:** Either $\Lambda$ lies in a one-dimensional complex subspace of $\mathbb{C}^2$, giving $G \cong \mathbb{C}^* \times \mathbb{C}^*$, or $\Lambda$ lies in a complex line in $\mathbb{C}^2$, giving $G = E \times \mathbb{C}$ where $E = \mathbb{C}/(\mathbb{Z} \oplus \mathbb{Z}\tau)$ is a complex torus (elliptic curve).
- **Rank 3 and 4:** More subtle; involves the theory of complex tori and abelian varieties.

Over the reals, the two-dimensional abelian simply connected real Lie group is $\mathbb{R} \times \mathbb{R}$, and any other connected two-dimensional abelian real Lie group is the quotient by a sublattice $\Lambda \subset \mathbb{R} \times \mathbb{R}$ of rank 1 or 2, i.e., either $\mathbb{R} \times S^1$ or $S^1 \times S^1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimension Two, Non-abelian Case)</span></p>

**Case 2: $\mathfrak{g}$ not abelian.** Viewing the Lie bracket as a linear map $[\,,\,]\colon \bigwedge^2 \mathfrak{g} \to \mathfrak{g}$, we see that if it is not zero, it must have one-dimensional image. We can thus choose a basis $\lbrace X, Y \rbrace$ for $\mathfrak{g}$ with $X$ spanning the image of $[\,,\,]$; after rescaling $Y$ we will have $[X, Y] = X$. There is thus a **unique nonabelian two-dimensional Lie algebra** over either $\mathbb{R}$ or $\mathbb{C}$.

The adjoint representation of $\mathfrak{g}$ is faithful:

$$\mathrm{ad}(X) = \begin{pmatrix} 0 & 1 \\\ 0 & 0 \end{pmatrix}, \qquad \mathrm{ad}(Y) = \begin{pmatrix} -1 & 0 \\\ 0 & 0 \end{pmatrix}.$$

These generate the algebra $\begin{pmatrix} * & * \\\ 0 & 0 \end{pmatrix} \subset \mathfrak{gl}_2\mathbb{C}$; exponentiating, we arrive at the adjoint form

$$G_0 = \left\lbrace \begin{pmatrix} a & b \\\ 0 & 1 \end{pmatrix} : a \neq 0 \right\rbrace = \mathrm{GL}_2\mathbb{C}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Universal Cover of the Non-abelian Two-Dimensional Group)</span></p>

The universal cover $G$ of $G_0$ can be realized as pairs $(t, s) \in \mathbb{C} \times \mathbb{C}$ with group law

$$(t, s) \cdot (t', s') = (t + t',\, s + e^t s').$$

The center is $Z(G) = \lbrace (2\pi i n, 0) \rbrace \cong \mathbb{Z}$, so the connected groups with this Lie algebra form a tower: $G$, and quotients $G_n = G/n\mathbb{Z}$ with group law $(a, b) \cdot (a', b') = (aa', b + a^n b')$ where $(a, b) \in \mathbb{C}^* \times \mathbb{C}$.

</div>

### 10.2 Dimension Three, Rank 1

Here we classify three-dimensional Lie algebras by the rank of the bracket map $[\,,\,]\colon \bigwedge^2 \mathfrak{g} \to \mathfrak{g}$, i.e., the dimension of the commutator subalgebra $\mathscr{D}\mathfrak{g}$. Rank 0 is the abelian case. We begin with rank 1.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rank 1 Classification)</span></p>

The kernel of the bracket map $[\,,\,]\colon \bigwedge^2 \mathfrak{g} \to \mathfrak{g}$ is two-dimensional. For some $X \in \mathfrak{g}$ it consists of all vectors of the form $X \wedge Y$ with $Y$ ranging over all of $\mathfrak{g}$. Completing $X$ to a basis $\lbrace X, Y, Z \rbrace$, we can write

$$[X, Y] = [X, Z] = 0, \qquad [Y, Z] = \alpha X + \beta Y + \gamma Z.$$

If either $\beta$ or $\gamma$ is nonzero, we can rechoose the basis so that $[Y, Z] = Y$, giving $\mathfrak{g} = \mathbb{C}X \oplus \mathbb{C}Y \oplus \mathbb{C}Z$ as the product of the one-dimensional abelian Lie algebra $\mathbb{C}X$ with the nonabelian two-dimensional Lie algebra $\mathbb{C}Y \oplus \mathbb{C}Z$.

Assuming $\beta = \gamma = 0$, replacing $X$ by $\alpha X$ gives the Lie algebra with

$$[X, Y] = [X, Z] = 0, \qquad [Y, Z] = X.$$

This is the **Heisenberg algebra**: the unique three-dimensional Lie algebra with commutator subalgebra of dimension one that is not a product.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Heisenberg Algebra and Group)</span></p>

The Heisenberg algebra is just the Lie algebra $\mathfrak{n}_3$ of strictly upper-triangular $3 \times 3$ matrices. The simply connected Lie group is the group $G$ of $3 \times 3$ unipotent upper-triangular matrices:

$$G = \left\lbrace \begin{pmatrix} 1 & a & b \\\ 0 & 1 & c \\\ 0 & 0 & 1 \end{pmatrix} : a, b, c \in \mathbb{C} \right\rbrace.$$

The center is $Z(G) = \left\lbrace \begin{pmatrix} 1 & 0 & b \\\ 0 & 1 & 0 \\\ 0 & 0 & 1 \end{pmatrix} : b \in \mathbb{C} \right\rbrace \cong \mathbb{C}$, so the discrete subgroups of $Z(G)$ are lattices $\Lambda$ of rank 1 or 2; any connected group with this Lie algebra is either $G$, $G/\mathbb{Z}$, or $G/(\mathbb{Z} \times \mathbb{Z})$, i.e., an extension of $\mathbb{C} \times \mathbb{C}$ by either $\mathbb{C}$, $\mathbb{C}^*$, or a torus $E$.

Over the reals, $\mathfrak{n}_3$ is again the unique real Lie algebra of dimension three with commutator subalgebra of dimension one; its simply connected form is the group $G$ of unipotent $3 \times 3$ matrices. The quotient $H = G/\mathbb{Z}$ is an interesting example of a group that cannot be realized as a matrix group (it admits no faithful finite-dimensional representations).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Infinite-Dimensional Representation of the Heisenberg Algebra)</span></p>

The group $H = G/\mathbb{Z}$ does, however, have an important infinite-dimensional representation. It arises from the representation of $\mathfrak{g}$ on the space $V = \mathscr{C}^\infty$ of functions on the real line $\mathbb{R}$ with coordinate $x$, in which $Y$ and $X$ act by

$$Y\colon f \mapsto \pi i x \cdot f, \qquad Z\colon f \mapsto \frac{df}{dx},$$

and $X = [Y, Z]$ is $-\pi i$ times the identity. Exponentiating, $e^{tY}$ acts by multiplication by $\cos(tx) + i\sin(tx)$; $e^{tZ}$ sends $f$ to $F_t$ where $F_t(x) = f(t + x)$; and $e^{tX}$ sends $f$ to the scalar multiple $e^{-\pi it} \cdot f$.

</div>

### 10.3 Dimension Three, Rank 2

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Setting Up Rank 2)</span></p>

Write the commutator subalgebra $\mathscr{D}\mathfrak{g} \subset \mathfrak{g}$ as the span of two elements $Y$ and $Z$. The commutator of $Y$ and $Z$ can be written as $[Y, Z] = \alpha Y + \beta Z$. The endomorphism $\mathrm{ad}(Y)$ carries $\mathfrak{g}$ into $\mathscr{D}\mathfrak{g}$, kills $Y$, and sends $Z$ to $\alpha Y + \beta Z$; on the other hand, $\mathrm{ad}(Y)$ is a commutator in $\mathrm{End}(\mathfrak{g})$, so it must have trace 0. Thus $\beta = 0$, and similarly $\alpha = 0$, so the subalgebra $\mathscr{D}\mathfrak{g}$ must be abelian. For any element $X \in \mathfrak{g}$ not in $\mathscr{D}\mathfrak{g}$, the map $\mathrm{ad}(X)\colon \mathscr{D}\mathfrak{g} \to \mathscr{D}\mathfrak{g}$ must be an isomorphism.

We distinguish two possibilities: either $\mathrm{ad}(X)$ is diagonalizable or it is not.

</div>

#### Possibility A: $\mathrm{ad}(X)$ Diagonalizable

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Family $\mathfrak{g}_\alpha$)</span></p>

If $\mathrm{ad}(X)$ is diagonalizable, we use eigenvectors $Y$, $Z$ for $\mathrm{ad}(X)$ as a basis for $\mathscr{D}\mathfrak{g}$. By rescaling $X$ we can assume one eigenvalue is 1. The structure equations become

$$[X, Y] = Y, \qquad [X, Z] = \alpha Z, \qquad [Y, Z] = 0 \tag{10.5}$$

for some $\alpha \in \mathbb{C}^*$. Two Lie algebras $\mathfrak{g}_\alpha$ and $\mathfrak{g}_{\alpha'}$ corresponding to two different scalars are isomorphic if and only if $\alpha = \alpha'$ or $\alpha = 1/\alpha'$. This is our first example of a **continuously varying family** of nonisomorphic complex Lie algebras.

The adjoint representation is faithful: $\mathrm{ad}(Y)$ carries $X$ to $-Y$ and kills $Y$ and $Z$; $\mathrm{ad}(Z)$ carries $X$ to $-\alpha Z$ and kills $Y$ and $Z$; $\mathrm{ad}(X)$ carries $Y$ to itself, $Z$ to $\alpha Z$, and kills $X$. A general element $aX - bY - cZ$ is represented (with respect to the basis $\lbrace Y, Z, X \rbrace$) by the matrix

$$\begin{pmatrix} a & 0 & b \\\ 0 & \alpha a & \alpha c \\\ 0 & 0 & 0 \end{pmatrix}.$$

Exponentiating, the corresponding group is

$$G = \left\lbrace \begin{pmatrix} e^t & 0 & u \\\ 0 & e^{\alpha t} & v \\\ 0 & 0 & 1 \end{pmatrix} : t, u, v \in \mathbb{C} \right\rbrace \subset \mathrm{GL}_3\mathbb{C}.$$

If $\alpha \notin \mathbb{Q}$, the exponential map is one-to-one and hence a homeomorphism, so $G$ is simply connected. If $\alpha \in \mathbb{Q}$, the group $A \cong \mathbb{C}^*$ and correspondingly $\pi_1(G) = \mathbb{Z}$.

</div>

#### Possibility B: $\mathrm{ad}(X)$ Not Diagonalizable

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Jordan Normal Form Case)</span></p>

Choose a basis $\lbrace Y, Z \rbrace$ of $\mathscr{D}\mathfrak{g}$ with respect to which $\mathrm{ad}(X)$ is in Jordan normal form; replacing $X$ by a multiple, assume both eigenvalues are 1. The structure equations become

$$[X, Y] = Y, \qquad [X, Z] = Y + Z, \qquad [Y, Z] = 0. \tag{10.8}$$

The adjoint action of the general element $aX - bY - cZ$ is represented by the matrix

$$\begin{pmatrix} a & a & b + c \\\ 0 & a & c \\\ 0 & 0 & 0 \end{pmatrix},$$

and exponentiating gives the group

$$G = \left\lbrace \begin{pmatrix} e^t & te^t & u \\\ 0 & e^t & v \\\ 0 & 0 & 1 \end{pmatrix} : t, u, v \in \mathbb{C} \right\rbrace.$$

This group has no center and hence is the unique connected complex Lie group with its Lie algebra.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Real Case, Rank 2)</span></p>

Over $\mathbb{R}$ there is a third possibility not present over $\mathbb{C}$: $\mathrm{ad}(X)$ may have distinct complex conjugate eigenvalues $\lambda$ and $\bar{\lambda}$. In the real case, the Lie algebras given by (10.5) and (10.8) are all homeomorphic to $\mathbb{R}^3$ and have no center, so they are the only connected real Lie groups with these Lie algebras.

</div>

### 10.4 Dimension Three, Rank 3

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Arriving at $\mathfrak{sl}_2\mathbb{C}$)</span></p>

In the rank 3 case, $\mathscr{D}\mathfrak{g} = \mathfrak{g}$. We claim that we can find an element $H \in \mathfrak{g}$ such that $\mathrm{ad}(H)\colon \mathfrak{g} \to \mathfrak{g}$ has an eigenvector with nonzero eigenvalue. For any nonzero $X \in \mathfrak{g}$, the rank of $\mathrm{ad}(X)$ must be 2 (since $\ker(\mathrm{ad}(X)) = \mathbb{C}X$). Either $\mathrm{ad}(X)$ has an eigenvector with nonzero eigenvalue, or it is nilpotent. If nilpotent, there exists $Y \in \mathfrak{g}$ not in $\ker(\mathrm{ad}(X))$ but in $\ker(\mathrm{ad}(X)^2)$, i.e., $[H, X] = \alpha X$ for some nonzero $\alpha$, so $\mathrm{ad}(Y)$ will have an eigenvector $X$ with nonzero eigenvalue.

Choose $H$ and $X \in \mathfrak{g}$ so that $X$ is an eigenvector with nonzero eigenvalue for $\mathrm{ad}(H)$, and write $[H, X] = \alpha X$. Since $H \in \mathscr{D}\mathfrak{g}$, $\mathrm{ad}(H)$ is a commutator in $\mathrm{End}(\mathfrak{g})$ and so has trace 0; thus $\mathrm{ad}(H)$ must have a third eigenvector $Y$ with eigenvalue $-\alpha$. By the Jacobi identity,

$$[H, [X, Y]] = -[X, \alpha Y] - [Y, \alpha X] = 0,$$

so $[X, Y]$ is a multiple of $H$; since it must be a nonzero multiple (otherwise $\mathscr{D}\mathfrak{g} \neq \mathfrak{g}$), we can rescale. Multiplying $H$ by a scalar so that $\alpha = 1$ or any other nonzero scalar, there is only **one** possible complex Lie algebra of this type. One could look for endomorphisms $H$, $X$, and $Y$ whose commutators satisfy these relations, or simply realize that the three-dimensional Lie algebra $\mathfrak{sl}_2\mathbb{C}$ has not yet been seen, so it must be this last possibility.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Lie Algebra $\mathfrak{sl}_2\mathbb{C}$)</span></p>

A natural basis for $\mathfrak{sl}_2\mathbb{C}$ is

$$H = \begin{pmatrix} 1 & 0 \\\ 0 & -1 \end{pmatrix}, \qquad X = \begin{pmatrix} 0 & 1 \\\ 0 & 0 \end{pmatrix}, \qquad Y = \begin{pmatrix} 0 & 0 \\\ 1 & 0 \end{pmatrix},$$

whose Lie algebra is given by

$$[H, X] = 2X, \qquad [H, Y] = -2Y, \qquad [X, Y] = H. \tag{10.11}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lie Groups with Lie Algebra $\mathfrak{sl}_2\mathbb{C}$)</span></p>

$\mathrm{SL}_2\mathbb{C}$ is simply connected (the map sending a matrix to its first row expresses $\mathrm{SL}_2\mathbb{C}$ as a bundle with fiber $\mathbb{C}$ over $\mathbb{C}^2 - \lbrace (0,0) \rbrace$). The center of $\mathrm{SL}_2\mathbb{C}$ is just the subgroup $\lbrace \pm I \rbrace$ of scalar matrices, so the only other connected group with Lie algebra $\mathfrak{sl}_2\mathbb{C}$ is the quotient $\mathrm{PSL}_2\mathbb{C} = \mathrm{SL}_2\mathbb{C}/\lbrace \pm I \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Real Forms of $\mathfrak{sl}_2\mathbb{C}$)</span></p>

Over the reals, there is one additional possibility: $\mathrm{ad}(H)$ may have two distinct complex conjugate eigenvalues $\lambda$ and $\bar{\lambda}$ (purely imaginary). After rescaling, one finds $X$, $Y \in \mathfrak{g}$ with

$$[H, X] = Y \qquad \text{and} \qquad [H, Y] = -X.$$

Using the Jacobi identity, this forces $[X, Y] = H$ (up to scalar). This is the Lie algebra $\mathfrak{su}_2$ of the real Lie group $\mathrm{SU}(2)$, with the structure equations

$$[H, X] = Y, \qquad [H, Y] = -X, \qquad [X, Y] = H. \tag{10.12}$$

It is the unique compact real form of $\mathfrak{sl}_2\mathbb{C}$, since $\mathfrak{su}_2 \otimes \mathbb{C} \cong \mathfrak{sl}_2\mathbb{C}$.

</div>

#### Lie Groups with Lie Algebra $\mathfrak{sl}_2\mathbb{R}$ and $\mathfrak{su}_2$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Covering Spaces of $\mathrm{SL}_2\mathbb{R}$ and $\mathrm{PSL}_2\mathbb{R}$)</span></p>

The center of $\mathrm{SL}_2\mathbb{R}$ is again $\lbrace \pm I \rbrace$, and $\mathrm{PSL}_2\mathbb{R}$ is the only group dominated by $\mathrm{SL}_2\mathbb{R}$. However, unlike the complex case, $\mathrm{SL}_2\mathbb{R}$ is **not** simply connected: sending a $2 \times 2$ matrix to its first row, $\mathrm{SL}_2\mathbb{R}$ is a bundle with fiber $\mathbb{R}$ over $\mathbb{R}^2 - \lbrace (0,0) \rbrace$, so $\pi_1(\mathrm{SL}_2\mathbb{R}) = \mathbb{Z}$. More precisely, $\mathrm{PSL}_2\mathbb{R}$ maps to the real projective line $\mathbb{P}^1\mathbb{R}$ (the circle), with fiber homeomorphic to $\mathbb{R}^2$, so $\pi_1(\mathrm{PSL}_2\mathbb{R}) = \mathbb{Z}$. We thus have a tower of covering spaces of $\mathrm{PSL}_2\mathbb{R}$, consisting of the simply-connected group $\tilde{S}$ with center $\mathbb{Z}$ and its quotients $\tilde{S}_n = \tilde{S}/n\mathbb{Z}$ (not all of which are covers of $\mathrm{SL}_2\mathbb{R}$).

The groups $\tilde{S}$ and $\tilde{S}_n$ have no faithful finite-dimensional representations; only their universal cover can be represented as a matrix group. All finite-dimensional representations of $\tilde{S}$ factor through $\mathrm{SL}_2\mathbb{R}$ or $\mathrm{PSL}_2\mathbb{R}$ (this will be a consequence of the analysis of representations of $\mathfrak{sl}_2\mathbb{C}$ in the next lecture).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Groups with Lie Algebra $\mathfrak{su}_2$)</span></p>

$\mathrm{SU}(2)$ is simply connected (it is homeomorphic to $S^3$ via the map sending a matrix to its first row vector, which is a unit quaternion). Its center is again $\lbrace \pm I \rbrace$, so the only other group with Lie algebra $\mathfrak{su}_2$ is $\mathrm{PSU}(2)$. We may also realize $\mathrm{SU}(2)$ as the group of unit quaternions.

</div>

#### Low-Dimensional Coincidences

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Isomorphisms Among Low-Dimensional Groups)</span></p>

The Lie algebra $\mathfrak{so}_3\mathbb{C}$ is isomorphic to $\mathfrak{sl}_2\mathbb{C}$, which induces an isomorphism between the corresponding adjoint forms $\mathrm{PSL}_2\mathbb{C}$ and $\mathrm{SO}_3\mathbb{C}$ (and between the simply-connected forms $\mathrm{SL}_2\mathbb{C}$ and $\mathrm{Spin}_3\mathbb{C}$). The key diagram of isomorphisms among real and complex groups is:

$$\mathrm{SU}(1,1) = \mathrm{SL}_2\mathbb{R} \hookrightarrow \mathrm{SL}_2\mathbb{C} \hookleftarrow \mathrm{SU}(2) = \lbrace\text{unit quaternions}\rbrace$$

$$\mathrm{PSL}_2\mathbb{R} \hookrightarrow \mathrm{PSL}_2\mathbb{C} \hookleftarrow \mathrm{PSU}(2) = \lbrace\text{unit quaternions}\rbrace/\lbrace\pm 1\rbrace$$

$$\mathrm{SO}^+(2,1) \hookrightarrow \mathrm{SO}_3\mathbb{C} \hookleftarrow \mathrm{SO}_3\mathbb{R}$$

Note also the coincidences $\mathrm{Sp}_2(\mathbb{C}) = \mathrm{SL}_2(\mathbb{C})$ and $\mathrm{Sp}_2(\mathbb{R}) = \mathrm{SL}_2(\mathbb{R})$, which follow from the fact that $\mathrm{Sp}$ refers to preserving a skew-symmetric bilinear form, and for $2 \times 2$ matrices the determinant is such a form.

The real Lie algebra $\mathfrak{su}_{1,1}$ is isomorphic to either $\mathfrak{su}_2$ or $\mathfrak{sl}_2\mathbb{R}$; in fact it is the latter, inducing an isomorphism of groups $\mathrm{SU}(1,1) \cong \mathrm{SL}_2\mathbb{R}$.

</div>

---

## Lecture 11: Representations of $\mathfrak{sl}_2\mathbb{C}$

This is the first of four lectures (11–14) that comprise the heart of the book. The naive analysis of §11.1, together with the analogous parts of Lectures 12 and 13, forms the paradigm for the study of finite-dimensional representations of all semisimple Lie algebras and groups. §11.2 shows how the analysis can be used to explicitly describe tensor products of irreducible representations (plethysm). §11.3 gives geometric interpretations via classical projective geometry.

### 11.1 The Irreducible Representations

We start our discussion of representations of semisimple Lie algebras with the simplest case, that of $\mathfrak{sl}_2\mathbb{C}$. The basic idea — already seen in the representation theory of the symmetric group on three letters — is to restrict the representation to an abelian subgroup, decompose into eigenspaces, and use the remaining elements to move between them.

We use the standard basis from Lecture 10:

$$H = \begin{pmatrix} 1 & 0 \\\ 0 & -1 \end{pmatrix}, \qquad X = \begin{pmatrix} 0 & 1 \\\ 0 & 0 \end{pmatrix}, \qquad Y = \begin{pmatrix} 0 & 0 \\\ 1 & 0 \end{pmatrix},$$

satisfying

$$[H, X] = 2X, \qquad [H, Y] = -2Y, \qquad [X, Y] = H. \tag{11.1}$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Diagonalizability of $H$)</span></p>

By the preservation of Jordan decomposition (Theorem 9.20), since $\mathfrak{sl}_2\mathbb{C}$ is semisimple and $H$ is diagonalizable in the standard representation, the action of $H$ on any finite-dimensional representation $V$ is diagonalizable. We thus have a decomposition

$$V = \bigoplus V_\alpha, \tag{11.3}$$

where for any vector $v \in V_\alpha$ we have $H(v) = \alpha \cdot v$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Fundamental Calculation)</span></p>

If $v$ is an eigenvector for $H$ with eigenvalue $\alpha$, then $X(v)$ is also an eigenvector for $H$, with eigenvalue $\alpha + 2$, and $Y(v)$ is an eigenvector with eigenvalue $\alpha - 2$. In other words,

$$X\colon V_\alpha \to V_{\alpha+2}, \qquad Y\colon V_\alpha \to V_{\alpha-2}.$$

**Proof.** $H(X(v)) = X(H(v)) + [H,X](v) = X(\alpha v) + 2X(v) = (\alpha+2) \cdot X(v)$. Similarly for $Y$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(String Structure of Eigenvalues)</span></p>

As an immediate consequence of the fundamental calculation and the irreducibility of $V$, all the complex numbers $\alpha$ that appear in the decomposition (11.3) must be congruent to one another mod 2: for any $\alpha_0$ that actually occurs, the subspace $\bigoplus_{n \in \mathbb{Z}} V_{\alpha_0 + 2n}$ would be invariant under $\mathfrak{sl}_2\mathbb{C}$ and hence equal to all of $V$.

By the same token, the $V_\alpha$ that appear must form an unbroken string of numbers of the form $\beta, \beta + 2, \ldots, \beta + 2k$. We denote by $n$ the last element in this sequence. The picture of the action of $\mathfrak{sl}_2\mathbb{C}$ on $V$ is:

$$\cdots \xleftarrow{Y} V_{n-4} \underset{Y}{\overset{X}{\rightleftarrows}} V_{n-2} \underset{Y}{\overset{X}{\rightleftarrows}} V_n$$

where $H$ acts by the scalar $\alpha$ on each $V_\alpha$, $X$ raises the eigenvalue by 2, and $Y$ lowers it.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span><span class="math-callout__name">(11.4)</span></p>

Choose any nonzero vector $v \in V_n$ (the highest eigenspace). Since $V_{n+2} = (0)$, we must have $X(v) = 0$. The vectors $\lbrace v, Y(v), Y^2(v), \ldots \rbrace$ span $V$.

**Proof.** It is enough to show that the subspace $W$ spanned by these vectors is invariant under $\mathfrak{sl}_2\mathbb{C}$. Clearly $Y$ preserves $W$ (it carries $Y^m(v)$ into $Y^{m+1}(v)$). Likewise $H$ preserves $W$ since $H(Y^m(v)) = (n - 2m) \cdot Y^m(v)$. It suffices to check that $X$ carries $Y^m(v)$ into a linear combination of the $Y^l(v)$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Action of $X$ on the Basis)</span></p>

Starting from $X(v) = 0$ and using the commutation relations, we compute:

$$X(Y(v)) = [X,Y](v) + Y(X(v)) = H(v) + 0 = n \cdot v.$$

$$X(Y^2(v)) = H(Y(v)) + Y(n \cdot v) = (n-2) \cdot Y(v) + n \cdot Y(v).$$

In general,

$$X(Y^m(v)) = m(n - m + 1) \cdot Y^{m-1}(v). \tag{11.5}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequences)</span></p>

Several important consequences follow:

1. **All eigenspaces $V_\alpha$ of $H$ are one-dimensional** (since the basis $v, Y(v), \ldots, Y^n(v)$ has exactly one vector in each eigenspace).

2. Since the actions of $H$, $X$, and $Y$ on the basis vectors are completely determined by the single number $n$, **the representation $V$ is completely determined by the collection of $\alpha$ occurring in $V = \bigoplus V_\alpha$** — i.e., by the single number $n$.

3. By finite-dimensionality, we must have $Y^k(v) = 0$ for sufficiently large $k$. If $m$ is the smallest power such that $Y^m(v) = 0$ but $Y^{m-1}(v) \neq 0$, then from (11.5):
   $0 = X(Y^m(v)) = m(n - m + 1) \cdot Y^{m-1}(v)$,
   so $n - m + 1 = 0$, i.e., **$n$ is a non-negative integer**. The eigenvalues of $H$ on $V$ form a string of integers symmetric about the origin: $n, n-2, \ldots, -n+2, -n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Classification of Irreducible Representations of $\mathfrak{sl}_2\mathbb{C}$)</span></p>

For each non-negative integer $n$, there is a **unique** irreducible representation $V^{(n)}$ of $\mathfrak{sl}_2\mathbb{C}$. It is $(n+1)$-dimensional, with $H$ having eigenvalues $n, n-2, \ldots, -n+2, -n$.

Moreover, in any representation $V$ of $\mathfrak{sl}_2\mathbb{C}$ such that all eigenvalues of $H$ have the same parity and all occur with multiplicity one, $V$ is necessarily irreducible. More generally, the number of irreducible factors in an arbitrary representation $V$ of $\mathfrak{sl}_2\mathbb{C}$ is exactly the sum of the multiplicities of $0$ and $1$ as eigenvalues of $H$.

</div>

#### Realization as Symmetric Powers

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Irreducible Representations as Symmetric Powers)</span></p>

The trivial one-dimensional representation $\mathbb{C}$ is $V^{(0)}$. The standard representation on $V = \mathbb{C}^2$ is $V^{(1)}$, with $H(x) = x$ and $H(y) = -y$ for the standard basis.

The $n$th symmetric power $\mathrm{Sym}^n V$ has basis $\lbrace x^n, x^{n-1}y, \ldots, y^n \rbrace$, and

$$H(x^{n-k} y^k) = (n - k) \cdot H(x) \cdot x^{n-k-1} y^k + k \cdot H(y) \cdot x^{n-k} y^{k-1} = (n - 2k) \cdot x^{n-k} y^k,$$

so the eigenvalues of $H$ on $\mathrm{Sym}^n V$ are exactly $n, n-2, \ldots, -n$. Since all eigenvalues occur with multiplicity 1, $\mathrm{Sym}^n V$ is irreducible, and hence

$$V^{(n)} = \mathrm{Sym}^n V. \tag{11.8}$$

*Any irreducible representation of $\mathfrak{sl}_2\mathbb{C}$ is a symmetric power of the standard representation $V \cong \mathbb{C}^2$.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Representations of the Groups)</span></p>

When we exponentiate the image of $\mathfrak{sl}_2\mathbb{C}$ under the embedding $\mathfrak{sl}_2\mathbb{C} \to \mathfrak{sl}_{n+1}\mathbb{C}$ corresponding to the representation $\mathrm{Sym}^n V$, we arrive at the group $\mathrm{SL}_2\mathbb{C}$ when $n$ is odd and $\mathrm{PGL}_2\mathbb{C}$ when $n$ is even. Thus, *the representations of the group $\mathrm{PGL}_2\mathbb{C}$ are exactly the even powers* $\mathrm{Sym}^{2n} V$.

</div>

### 11.2 A Little Plethysm

Knowing the eigenspace decomposition of given representations tells us the eigenspace decomposition of all their tensor, symmetric, and alternating products and powers. We can use this to describe the decomposition of these products into irreducible representations of $\mathfrak{sl}_2\mathbb{C}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tensor Product Decomposition)</span></p>

If $V = \bigoplus V_\alpha$ and $W = \bigoplus W_\beta$, then $V \otimes W = \bigoplus (V_\alpha \otimes W_\beta)$ and $V_\alpha \otimes W_\beta$ is an eigenspace for $H$ with eigenvalue $\alpha + \beta$. In general, for $a \ge b$ we have

$$\mathrm{Sym}^a V \otimes \mathrm{Sym}^b V = \mathrm{Sym}^{a+b} V \oplus \mathrm{Sym}^{a+b-2} V \oplus \cdots \oplus \mathrm{Sym}^{a-b} V.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tensor Product $\mathrm{Sym}^2 V \otimes \mathrm{Sym}^3 V$)</span></p>

Let $V \cong \mathbb{C}^2$ be the standard representation. The eigenvalues of $\mathrm{Sym}^2 V$ are $2, 0, -2$, and those of $\mathrm{Sym}^3 V$ are $3, 1, -1, -3$. The 12 eigenvalues of the tensor product $\mathrm{Sym}^2 V \otimes \mathrm{Sym}^3 V$ are thus $5$, $3$ and $-3$ (taken twice), $1$ and $-1$ (taken three times), and $-5$.

The eigenvector with eigenvalue 5 generates a subrepresentation isomorphic to $\mathrm{Sym}^5 V$, accounting for one occurrence each of $5, 3, 1, -1, -3, -5$. The complement has eigenvalues $3, 1, -1, -3$, which gives $\mathrm{Sym}^3 V$. The further complement has eigenvalue $1, -1$, giving $\mathrm{Sym}^1 V = V$. Therefore:

$$\mathrm{Sym}^2 V \otimes \mathrm{Sym}^3 V \cong \mathrm{Sym}^5 V \oplus \mathrm{Sym}^3 V \oplus V.$$

</div>

#### Symmetric Powers of Symmetric Powers

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Symmetric Square of $\mathrm{Sym}^2 V$)</span></p>

Let $W = \mathrm{Sym}^2 V$ with eigenvalues $-2, 0, 2$. The symmetric square of $W$ will have eigenvalues the pairwise sums of these numbers (each occurring once): $-4, -2, 0$ (occurring twice), $2, 4$. Therefore:

$$\mathrm{Sym}^2(\mathrm{Sym}^2 V) = \mathrm{Sym}^4 V \oplus \mathrm{Sym}^0 V. \tag{11.12}$$

We can see this directly: there is a natural evaluation map $\mathrm{Sym}^2(\mathrm{Sym}^2 V) \to \mathrm{Sym}^4 V$ obtained by multiplication of polynomials. Its kernel is one-dimensional, spanned by $(x^2)(y^2) - (xy)^2$, which gives the trivial subrepresentation $\mathrm{Sym}^0 V$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(General Formula for Symmetric Powers of $\mathrm{Sym}^2 V$)</span></p>

For all $n$,

$$\mathrm{Sym}^n(\mathrm{Sym}^2 V) = \bigoplus_{s=0}^{\lfloor n/2 \rfloor} \mathrm{Sym}^{2n - 4s} V.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Further Plethysm Decompositions)</span></p>

By eigenvalue analysis:

$$\mathrm{Sym}^3(\mathrm{Sym}^2 V) = \mathrm{Sym}^6 V \oplus \mathrm{Sym}^2 V.$$

$$\mathrm{Sym}^2(\mathrm{Sym}^3 V) \cong \mathrm{Sym}^6 V \oplus \mathrm{Sym}^2 V. \tag{11.18}$$

$$\mathrm{Sym}^4(\mathrm{Sym}^2 V) \cong \mathrm{Sym}^8 V \oplus \mathrm{Sym}^4 V \oplus \mathrm{Sym}^0 V.$$

$$\mathrm{Sym}^3(\mathrm{Sym}^3 V) = \mathrm{Sym}^9 V \oplus \mathrm{Sym}^5 V \oplus \mathrm{Sym}^3 V.$$

</div>

### 11.3 A Little Geometric Plethysm

We now give geometric interpretations of the decompositions above. Instead of looking at the action of $\mathfrak{sl}_2\mathbb{C}$ or $\mathrm{SL}_2\mathbb{C}$ on a representation $W$, we look at the action of $\mathrm{PGL}_2\mathbb{C}$ on the associated projective space $\mathbb{P}W$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Projective Spaces and Representations)</span></p>

For a vector space $W$ of dimension $n+1$, $\mathrm{Sym}^k W^*$ is the space of homogeneous polynomials of degree $k$ on the projective space $\mathbb{P}^n = \mathbb{P}W$ of lines in $W$; dually, $\mathrm{Sym}^k W$ is the space of homogeneous polynomials of degree $k$ on $\mathbb{P}^n = \mathbb{P}(W^*)$, i.e., the space of hypersurfaces of degree $k$ in that projective space.

The group of automorphisms of projective space $\mathbb{P}^n$ — either as algebraic variety or as complex manifold — is $\mathrm{PGL}_{n+1}\mathbb{C}$.

</div>

#### The Veronese Embedding and Rational Normal Curves

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Veronese Embedding)</span></p>

For any vector space $V$ and any positive integer $n$, the **Veronese embedding** is the natural map

$$\mathbb{P}V^* \hookrightarrow \mathbb{P}(\mathrm{Sym}^n V^*)$$

that maps the line spanned by $v \in V^*$ to the line spanned by $v^n \in \mathrm{Sym}^n V^*$.

When $V$ is two-dimensional, $\mathbb{P}V^* = \mathbb{P}^1$ and we have a map $\iota_n\colon \mathbb{P}^1 \hookrightarrow \mathbb{P}^n = \mathbb{P}(\mathrm{Sym}^n V^*)$. Choosing bases $\lbrace \alpha, \beta \rbrace$ for $V^*$ and $\lbrace \ldots, \binom{n}{k} \alpha^k \beta^{n-k}, \ldots \rbrace$ for $\mathrm{Sym}^n V^*$, this map may be given in coordinates as

$$[x, y] \mapsto [x^n,\, x^{n-1}y,\, x^{n-2}y^2,\, \ldots,\, xy^{n-1},\, y^n].$$

The image is called the **rational normal curve** $C_n$ of degree $n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equations of the Rational Normal Curve)</span></p>

For $n = 2$, $C$ is the **plane conic** defined by $Z_0 Z_2 - Z_1^2 = 0$.

For $n = 3$, $C$ is the **twisted cubic curve** in $\mathbb{P}^3$, defined by the three quadratic polynomials $Z_0 Z_2 - Z_1^2$, $Z_0 Z_3 - Z_1 Z_2$, $Z_1 Z_3 - Z_2^2$.

More generally, the rational normal curve is the common zero locus of the $2 \times 2$ minors of the matrix

$$M = \begin{pmatrix} Z_0 & Z_1 & \cdots & Z_{n-1} \\\ Z_1 & Z_2 & \cdots & Z_n \end{pmatrix},$$

that is, the locus where the rank of $M$ is 1.

The group $G$ of automorphisms of $\mathbb{P}^n$ that preserve $C_n$ is precisely $\mathrm{PGL}_2\mathbb{C}$. Conversely, if $W$ is any $(n+1)$-dimensional representation of $\mathrm{SL}_2\mathbb{C}$ and $\mathbb{P}W \cong \mathbb{P}^n$ contains a rational normal curve of degree $n$, then $W \cong \mathrm{Sym}^n V$.

</div>

#### Geometric Interpretation of Plethysm

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Geometric Interpretation of $\mathrm{Sym}^2(\mathrm{Sym}^2 V)$)</span></p>

$\mathrm{SL}_2\mathbb{C}$ acts on $\mathbb{P}^2 = \mathbb{P}(\mathrm{Sym}^2 V^*)$ as the group of motions of $\mathbb{P}^2$ carrying the conic $C_2$ into itself. Its action on the space $\mathrm{Sym}^2(\mathrm{Sym}^2 V)$ of quadratic polynomials on $\mathbb{P}^2$ must preserve the one-dimensional subspace $\mathbb{C} \cdot F$ spanned by the polynomial $F$ defining the conic $C_2$.

By pullback via the Veronese $\iota_2\colon \mathbb{P}^1 \to \mathbb{P}^2$, the space of quadratic polynomials on $\mathbb{P}^2$ maps to the space of quartic polynomials on $\mathbb{P}^1$, with kernel $\mathbb{C} \cdot F$. This gives the exact sequence

$$0 \to \mathrm{Sym}^0 V \to \mathrm{Sym}^2(\mathrm{Sym}^2 V) \to \mathrm{Sym}^4 V \to 0,$$

which implies the decomposition $\mathrm{Sym}^2(\mathrm{Sym}^2 V) = \mathrm{Sym}^4 V \oplus \mathrm{Sym}^0 V$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(11.16 — Double Lines as a Subrepresentation)</span></p>

The subrepresentation $\mathrm{Sym}^4 V \subset \mathrm{Sym}^2(\mathrm{Sym}^2 V)$ is the space of conics spanned by the family of **double lines** tangent to the conic $C = C_2$. The tangent line to $C$ at the point $[1, \alpha, \alpha^2]$ is $L_\alpha = \lbrace Z : \alpha^2 Z_0 - 2\alpha Z_1 + Z_2 = 0 \rbrace$, and the double line $2L_\alpha$ is the conic with equation $\alpha^4 Z_0^2 - 4\alpha^3 Z_0 Z_1 + \cdots$. These conics generate a four-dimensional projective subspace of $\mathbb{P}(\mathrm{Sym}^2(\mathrm{Sym}^2 V))$, which is invariant under $\mathrm{SL}_2\mathbb{C}$ and visibly complementary to $\mathbb{C} \cdot F$. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Geometric Interpretation of $\mathrm{Sym}^2(\mathrm{Sym}^3 V)$)</span></p>

For the twisted cubic $C = C_3 \subset \mathbb{P}^3$, the decomposition

$$\mathrm{Sym}^2(\mathrm{Sym}^3 V) \cong \mathrm{Sym}^6 V \oplus \mathrm{Sym}^2 V \tag{11.18}$$

can be interpreted as follows: the space of quadratic polynomials on $\mathbb{P}^3$ contains, as a subrepresentation, the space of quadrics containing the curve $C$ itself (this is $\mathrm{Sym}^2 V$, three-dimensional). The quotient, via pullback $\iota_3^*$ to the space of sextic polynomials on $\mathbb{P}^1$, is $\mathrm{Sym}^6 V$.

By the action of $\mathrm{SL}_2\mathbb{C}$ on the space of quadric surfaces containing the twisted cubic $C$, we get its action on $\mathbb{P}(\mathrm{Sym}^2 V^*) \cong \mathbb{P}^2$, making explicit the isomorphism $\mathrm{SO}_3\mathbb{C} \cong \mathrm{PSL}_2\mathbb{C}$.

</div>

---

## Lecture 12: Representations of $\mathfrak{sl}_3\mathbb{C}$, Part I

This lecture develops results for $\mathfrak{sl}_3\mathbb{C}$ analogous to those of §11.1 (though not in exactly the same order). This involves generalizing some of the basic terms of §11 — e.g., the notions of eigenvalue and eigenvector have to be redefined — but the basic ideas are in some sense already in §11.1. We arrive at a classification of the representations of $\mathfrak{sl}_3\mathbb{C}$ that is every bit as detailed and explicit as the classification for $\mathfrak{sl}_2\mathbb{C}$, and — crucially — *no further concepts are needed to classify the finite-dimensional representations of all remaining semisimple Lie algebras*.

### The Cartan Subalgebra and Weights

In the case of $\mathfrak{sl}_2\mathbb{C}$ we started with a single element $H$ and decomposed any representation $V$ into eigenspaces for $H$. For $\mathfrak{sl}_3\mathbb{C}$, the analogue of $H$ is not a single element but a *subspace* $\mathfrak{h} \subset \mathfrak{sl}_3\mathbb{C}$: the two-dimensional space of all diagonal matrices (of trace zero).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cartan Subalgebra of $\mathfrak{sl}_3\mathbb{C}$)</span></p>

The **Cartan subalgebra** of $\mathfrak{sl}_3\mathbb{C}$ is

$$\mathfrak{h} = \left\lbrace \begin{pmatrix} a_1 & 0 & 0 \\\ 0 & a_2 & 0 \\\ 0 & 0 & a_3 \end{pmatrix} : a_1 + a_2 + a_3 = 0 \right\rbrace,$$

and so $\mathfrak{h}^* = \mathbb{C}\lbrace L_1, L_2, L_3 \rbrace / (L_1 + L_2 + L_3 = 0)$, where $L_i$ extracts the $i$th diagonal entry: $L_i(\mathrm{diag}(a_1, a_2, a_3)) = a_i$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Eigenvector and Eigenvalue for $\mathfrak{h}$)</span></p>

Since commuting diagonalizable matrices are simultaneously diagonalizable, any finite-dimensional representation $V$ of $\mathfrak{sl}_3\mathbb{C}$ admits a decomposition

$$V = \bigoplus V_\alpha, \tag{12.2}$$

where $\alpha$ ranges over a finite subset of $\mathfrak{h}^*$ and for any vector $v \in V_\alpha$ we have $H(v) = \alpha(H) \cdot v$ for every $H \in \mathfrak{h}$. A nonzero $v \in V_\alpha$ is called an **eigenvector** (or **weight vector**) for $\mathfrak{h}$; the linear functional $\alpha \in \mathfrak{h}^*$ is called the corresponding **eigenvalue** (or **weight**). The subspace $V_\alpha$ is the **weight space** of $\alpha$.

</div>

### Roots and the Root Space Decomposition

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Adjoint Action and Root Spaces)</span></p>

Applying the decomposition (12.2) to the adjoint representation of $\mathfrak{sl}_3\mathbb{C}$, we obtain

$$\mathfrak{sl}_3\mathbb{C} = \mathfrak{h} \oplus \bigoplus_\alpha \mathfrak{g}_\alpha, \tag{12.3}$$

where $\alpha$ ranges over a finite subset of $\mathfrak{h}^*$ and $\mathfrak{h}$ acts on each space $\mathfrak{g}_\alpha$ by scalar multiplication: for any $H \in \mathfrak{h}$ and $Y \in \mathfrak{g}_\alpha$,

$$[H, Y] = \mathrm{ad}(H)(Y) = \alpha(H) \cdot Y.$$

The six linear functionals $\alpha$ appearing are the six functionals $L_i - L_j$ ($i \neq j$); the space $\mathfrak{g}_{L_i - L_j}$ is generated by the elementary matrix $E_{i,j}$ (the $3 \times 3$ matrix with a single 1 in the $(i,j)$th position). These are the **roots** of $\mathfrak{sl}_3\mathbb{C}$, and the $\mathfrak{g}_\alpha$ are the **root spaces**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Root Diagram)</span></p>

The roots can be pictured as six points in the two-dimensional real vector space $\mathfrak{h}^*_\mathbb{R}$, forming a regular hexagon:

$$L_2 - L_1, \quad L_2 - L_3, \quad L_1 - L_3, \quad L_1 - L_2, \quad L_3 - L_2, \quad L_3 - L_1.$$

These six roots, together with the origin (representing $\mathfrak{h}$), tile the plane with the equilateral-triangle lattice. The root lattice is $\Lambda_R = \mathbb{Z}\lbrace L_i - L_j \rbrace$.

</div>

### The Fundamental Calculation (for $\mathfrak{sl}_3\mathbb{C}$)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Fundamental Calculation — Third Time)</span></p>

Let $V = \bigoplus V_\alpha$ be a representation of $\mathfrak{sl}_3\mathbb{C}$. If $X \in \mathfrak{g}_\alpha$ and $v \in V_\beta$, then $X(v)$ is again an eigenvector for $\mathfrak{h}$, with eigenvalue $\alpha + \beta$:

$$H(X(v)) = X(\beta(H) \cdot v) + (\alpha(H) \cdot X)(v) = (\alpha(H) + \beta(H)) \cdot X(v).$$

In other words, $\mathfrak{g}_\alpha\colon V_\beta \to V_{\alpha + \beta}$. We can thus represent the eigenspaces $V_\alpha$ of $V$ by dots in the plane $\mathfrak{h}^*$, and each $\mathfrak{g}_\alpha$ acts by "translation," carrying each dot $\beta$ to the dot $\alpha + \beta$.

</div>

### Positive Roots, Highest Weight Vectors

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Positive and Negative Roots)</span></p>

Choose a linear functional $l\colon \Lambda_R \to \mathbb{R}$ that is irrational with respect to the root lattice (i.e., has no kernel on $\Lambda_R$); specifically, choose $l(a_1 L_1 + a_2 L_2 + a_3 L_3) = a a_1 + b a_2 + c a_3$ with $a + b + c = 0$ and $a > b > c$.

The **positive roots** are those $\alpha$ with $l(\alpha) > 0$: these are exactly $\mathfrak{g}_{L_1 - L_2}$, $\mathfrak{g}_{L_1 - L_3}$, and $\mathfrak{g}_{L_2 - L_3}$, i.e., the matrices with a single nonzero entry above the diagonal. The **negative roots** are $L_2 - L_1$, $L_3 - L_1$, $L_3 - L_2$, corresponding to entries below the diagonal.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Highest Weight Vector)</span></p>

For any representation $V$ of $\mathfrak{sl}_3\mathbb{C}$, we go to the eigenspace $V_\alpha$ for which $l(\alpha)$ is maximal and choose $v \in V_\alpha$. Such a $v$ is called a **highest weight vector**; it satisfies:

1. $v \in V_\alpha$ is an eigenvector for $\mathfrak{h}$, i.e., $v \in V_\alpha$ for some $\alpha$; and
2. $v$ is killed by $E_{1,2}$, $E_{1,3}$, and $E_{2,3}$ (all positive root spaces).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(12.9 — Existence of Highest Weight Vectors)</span></p>

There is a vector $v \in V$ with the properties:
1. $v \in V_\alpha$ for some $\alpha$; and
2. $v$ is killed by $E_{1,2}$, $E_{1,3}$, and $E_{2,3}$.

</div>

We set $H_{i,j} = [E_{i,j}, E_{j,i}] = E_{i,i} - E_{j,j}$.

### Generation by the Highest Weight Vector

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span><span class="math-callout__name">(12.10)</span></p>

Let $V$ be an irreducible representation of $\mathfrak{sl}_3\mathbb{C}$, and $v \in V$ a highest weight vector. Then $V$ is generated by the images of $v$ under successive applications of the three operators $E_{2,1}$, $E_{3,1}$, and $E_{3,2}$ (the negative root space generators).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Claim 12.10)</span></p>

The proof is formally the same as for $\mathfrak{sl}_2\mathbb{C}$: we argue that the subspace $W$ spanned by images of $v$ under $E_{2,1}$, $E_{3,1}$, and $E_{3,2}$ is preserved by all of $\mathfrak{sl}_3\mathbb{C}$, and hence must be all of $V$ by irreducibility.

Let $w_n$ denote any word of length $n$ in the letters $E_{2,1}$ and $E_{3,2}$ (note $E_{3,1} = [E_{3,2}, E_{2,1}]$ so it is already generated), and let $W_n$ be the vector space spanned by the vectors $w_n(v)$ for all such words. It is straightforward that $\mathfrak{h}$ preserves $W$ (since every $w_n(v)$ is an eigenvector for $\mathfrak{h}$) and that $E_{2,1}$ and $E_{3,2}$ carry $W_n$ into $W_{n+1}$. The key computation is that $E_{1,2}$ and $E_{2,3}$ carry $W_n$ into $W_{n-1}$: for example,

$$E_{1,2}(E_{2,1}(v)) = (E_{2,1}(E_{1,2}(v))) + [E_{1,2}, E_{2,1}](v) = \alpha([E_{1,2}, E_{2,1}]) \cdot v$$

since $E_{1,2}(v) = 0$ and $[E_{1,2}, E_{2,1}] \in \mathfrak{h}$. The general induction then shows $E_{1,2}$ and $E_{2,3}$ (and hence $E_{1,3}$, their commutator) carry $W$ into itself, so $W$ is invariant under all of $\mathfrak{sl}_3\mathbb{C}$. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(12.11 — Irreducibility from Highest Weight)</span></p>

If $V$ is any representation of $\mathfrak{sl}_3\mathbb{C}$ and $v \in V$ is a highest weight vector, then the subrepresentation $W$ of $V$ generated by the images of $v$ under $E_{2,1}$, $E_{3,1}$, and $E_{3,2}$ is **irreducible**.

**Proof.** Let $\alpha$ be the weight of $v$. Then $W_\alpha$ is one-dimensional (spanned by $v$). If $W$ were not irreducible, $W = W' \oplus W''$ for some representations $W'$ and $W''$, and the projection of $v$ would land in exactly one of them (since $W_\alpha$ is one-dimensional), so $v$ would belong to $W'$ or $W''$, and hence $W$ is that summand. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Uniqueness of the Highest Weight Vector)</span></p>

As a corollary, any irreducible representation of $\mathfrak{sl}_3\mathbb{C}$ has a **unique** highest weight vector (up to scalars). More generally, the set of highest weight vectors in $V$ forms a union of linear subspaces $\Psi_W$ corresponding to the irreducible subrepresentations $W$ of $V$, with the dimension of $\Psi_W$ equal to the number of times $W$ appears in the direct sum decomposition of $V$ into irreducibles.

</div>

### The Shape of the Weight Diagram

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Boundary of the Weight Diagram)</span></p>

Let $V$ be irreducible with highest weight $\alpha$. The "border vectors" $(E_{2,1})^k(v)$ lie in eigenspaces $\mathfrak{g}_{\alpha + k(L_2 - L_1)}$, $k = 0, 1, \ldots$, forming an unbroken string along the direction $L_2 - L_1$ until we reach the first $m$ such that $(E_{2,1})^m(v) = 0$.

</div>

#### The $\mathfrak{sl}_2\mathbb{C}$ Subalgebras

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Embedded Copies of $\mathfrak{sl}_2\mathbb{C}$)</span></p>

For any $i \neq j$, the elements $E_{i,j}$, $E_{j,i}$, and their commutator $H_{i,j} = E_{i,i} - E_{j,j}$ span a subalgebra $\mathfrak{s}_{L_i - L_j}$ of $\mathfrak{sl}_3\mathbb{C}$ isomorphic to $\mathfrak{sl}_2\mathbb{C}$ (via $E_{i,j} \mapsto X$, $E_{j,i} \mapsto Y$, $H_{i,j} \mapsto H$).

The subalgebra $\mathfrak{s}_{L_i - L_j}$ shifts eigenvalues only in the direction of $L_i - L_j$. In particular, the subspace

$$W = \bigoplus_k \mathfrak{g}_{\alpha + k(L_i - L_j)}$$

is a representation of $\mathfrak{s}_{L_i - L_j} \cong \mathfrak{sl}_2\mathbb{C}$. From our knowledge of $\mathfrak{sl}_2\mathbb{C}$ representations, we deduce that **the eigenvalues of $H_{i,j}$ on $W$ must be integers**, and the string of dots along the line through $\alpha$ in direction $L_i - L_j$ is **symmetric under reflection** in the line $\langle H_{i,j}, L \rangle = 0$ in $\mathfrak{h}^*$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hexagonal Symmetry)</span></p>

Applying this analysis to all three subalgebras $\mathfrak{s}_{L_1 - L_2}$, $\mathfrak{s}_{L_2 - L_3}$, and $\mathfrak{s}_{L_1 - L_3}$, we find that the set of eigenvalues in any irreducible representation $V$ with highest weight $\alpha$ is bounded by a **hexagon** symmetric with respect to the three lines $\langle H_{i,j}, L \rangle = 0$ in $\mathfrak{h}^*$, with one vertex at $\alpha$. Indeed, the hexagon is the convex hull of the images of $\alpha$ under the group of isometries of the plane generated by reflections in these three lines (the Weyl group $\mathfrak{S}_3$).

</div>

### The Weight Lattice

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(12.15 — Integrality of Weights)</span></p>

All the eigenvalues of any irreducible finite-dimensional representation of $\mathfrak{sl}_3\mathbb{C}$ must lie in the lattice $\Lambda_W \subset \mathfrak{h}^*$ generated by the $L_i$, and be congruent modulo the root lattice $\Lambda_R \subset \mathfrak{h}^*$ generated by the $L_i - L_j$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weight Lattice vs. Root Lattice)</span></p>

This is exactly analogous to the situation for $\mathfrak{sl}_2\mathbb{C}$: there the eigenvalues of $H$ lay in the lattice $\Lambda_W \cong \mathbb{Z}$ of linear forms on $\mathbb{C}H$ integral on $H$, and were congruent modulo the sublattice $\Lambda_R = 2\mathbb{Z}$ generated by the eigenvalues of $H$ in the adjoint representation.

For $\mathfrak{sl}_3\mathbb{C}$, the weight lattice $\Lambda_W$ is generated by the $L_i$ and the root lattice $\Lambda_R$ by the $L_i - L_j$. The quotient $\Lambda_W / \Lambda_R \cong \mathbb{Z}/3\mathbb{Z}$. Every weight of an irreducible representation lies in a single coset of $\Lambda_R$ in $\Lambda_W$, and the set of weights fills out the lattice points inside the hexagonal boundary described above.

</div>

---

## Lecture 13: Representations of $\mathfrak{sl}_3\mathbb{C}$, Part II: Mainly Lots of Examples

In this lecture we complete the analysis of the irreducible representations of $\mathfrak{sl}_3\mathbb{C}$, culminating in §13.2 with the answers to all three questions raised at the end of Lecture 12: we explicitly construct the unique irreducible representation with given highest weight, and in particular determine its multiplicities. The latter two sections (§13.3 and §13.4) correspond to §11.2 and §11.3 in the lecture on $\mathfrak{sl}_2\mathbb{C}$.

### 13.1 Examples

We begin by stating the basic existence and uniqueness theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(13.1 — Classification of Irreducible Representations of $\mathfrak{sl}_3\mathbb{C}$)</span></p>

For any pair of natural numbers $a$, $b$ there exists a **unique** irreducible, finite-dimensional representation $\Gamma_{a,b}$ of $\mathfrak{sl}_3\mathbb{C}$ with highest weight $aL_1 - bL_3$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Highest Weight Condition)</span></p>

Recall from Lecture 12 that the highest weight of any irreducible representation must lie in the $\frac{1}{3}$-plane described by the inequalities $\langle H_{1,2}, L \rangle \ge 0$ and $\langle H_{2,3}, L \rangle \ge 0$, i.e., it must be of the form $(a+b)L_1 + bL_2 = aL_1 - bL_3$ for some pair of non-negative integers $a$ and $b$.

The existence part of the theorem follows from the observation that $\mathrm{Sym}^a V \otimes \mathrm{Sym}^b V^*$ will contain an irreducible subrepresentation $\Gamma_{a,b}$ with highest weight $aL_1 - bL_3$. The uniqueness is established via a more indirect trick (see §13.2 below).

</div>

#### The Standard Representation and its Dual

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Standard Representation $V = \Gamma_{1,0}$)</span></p>

The standard representation of $\mathfrak{sl}_3\mathbb{C}$ on $V \cong \mathbb{C}^3$ has eigenvectors $e_1, e_2, e_3$ for $\mathfrak{h}$ with eigenvalues $L_1, L_2, L_3$ respectively. Its weight diagram is a triangle with vertices at the three $L_i$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Dual Representation $V^* = \Gamma_{0,1}$)</span></p>

The dual representation $V^*$ has dual basis vectors $e_i^*$ with eigenvalues $-L_i$. Note that while in the case of $\mathfrak{sl}_2\mathbb{C}$ the weights of any representation were symmetric about the origin (and correspondingly each representation was isomorphic to its dual), this is **not** true for $\mathfrak{sl}_3\mathbb{C}$: the diagrams for $V$ and $V^*$ look different (one is a reflection of the other).

We have $V^* \cong \bigwedge^2 V$ (whose weights are the pairwise sums of the distinct weights of $V$), and likewise $V \cong \bigwedge^2 V^*$.

</div>

#### Symmetric Powers

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Symmetric Powers $\mathrm{Sym}^n V$ and $\mathrm{Sym}^n V^*$)</span></p>

The symmetric powers of $V$ and $V^*$ are exactly the representations with **triangular** weight diagrams (as opposed to hexagonal). All weights occur with multiplicity 1, so $\mathrm{Sym}^n V$ and $\mathrm{Sym}^n V^*$ are all irreducible:

$$\mathrm{Sym}^n V = \Gamma_{n,0} \qquad \text{and} \qquad \mathrm{Sym}^n V^* = \Gamma_{0,n}.$$

</div>

#### Tensor Products with the Standard Representation

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Tensor Product $V \otimes V^*$)</span></p>

The weights of $V \otimes V^*$ are the sums $L_i + (-L_j) = L_i - L_j$ (each occurring once) and $0$ (occurring with multiplicity three, with weight vectors $e_i \otimes e_i^*$). There is a $\mathfrak{sl}_3\mathbb{C}$-linear contraction map

$$V \otimes V^* \to \mathbb{C}, \qquad v \otimes u^* \mapsto u^*(v),$$

whose kernel (the subspace of traceless tensors) is the adjoint representation $\mathfrak{sl}_3\mathbb{C}$ itself, which is irreducible ($= \Gamma_{1,1}$). Thus

$$V \otimes V^* \cong \Gamma_{1,1} \oplus \mathbb{C}.$$

(Physicists call this adjoint representation of $\mathfrak{sl}_3\mathbb{C}$ (or $\mathrm{SU}(3)$) the "eightfold way," and relate its decomposition to mesons and baryons.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Tensor Product $\mathrm{Sym}^2 V \otimes V^*$)</span></p>

The weights of $\mathrm{Sym}^2 V \otimes V^*$ are $2L_i - L_j$ (each occurring once) and $L_i$ (each occurring three times). The contraction map $\iota\colon \mathrm{Sym}^2 V \otimes V^* \to V$ is surjective, and its kernel is the irreducible representation $\Gamma_{2,1}$:

$$\mathrm{Sym}^2 V \otimes V^* \cong \Gamma_{2,1} \oplus V.$$

</div>

### 13.2 Description of the Irreducible Representations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Highest Weight Vectors in Tensor Products)</span></p>

If $V$ and $W$ have highest weight vectors $v$ and $w$ with weights $\alpha$ and $\beta$ respectively, then $v \otimes w \in V \otimes W$ is a highest weight vector of weight $\alpha + \beta$. More generally, $v^n \in \mathrm{Sym}^n V$ is a highest weight vector of weight $n\alpha$, etc.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 13.1)</span></p>

**Existence:** The representation $\mathrm{Sym}^a V \otimes \mathrm{Sym}^b V^*$ will contain an irreducible subrepresentation $\Gamma_{a,b}$ with highest weight $aL_1 - bL_3$.

**Uniqueness:** Given irreducible representations $V$ and $W$ with highest weight $\alpha$, let $v \in V$ and $w \in W$ be highest weight vectors. Then $(v, w)$ is again a highest weight vector in $V \oplus W$ with weight $\alpha$; let $U \subset V \oplus W$ be the irreducible subrepresentation generated by $(v, w)$. The projection maps $\pi_1\colon U \to V$ and $\pi_2\colon U \to W$ are nonzero maps between irreducible representations of $\mathfrak{sl}_3\mathbb{C}$, hence must be isomorphisms, and we deduce $V \cong W$. $\square$

</div>

#### The Contraction Map and Explicit Construction

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span><span class="math-callout__name">(13.4 — $\Gamma_{a,b}$ as a Kernel)</span></p>

There is a general contraction map

$$\iota_{a,b}\colon \mathrm{Sym}^a V \otimes \mathrm{Sym}^b V^* \to \mathrm{Sym}^{a-1} V \otimes \mathrm{Sym}^{b-1} V^*$$

analogous to the map $\iota$ above, defined as the dual of multiplication by the identity element in $V \otimes V^* \cong \mathrm{Hom}(V, V)$. The map $\iota_{a,b}$ is surjective, and the subrepresentation $\Gamma_{a,b} \subset \mathrm{Sym}^a V \otimes \mathrm{Sym}^b V^*$ is the **irreducible representation** that is exactly $\ker(\iota_{a,b})$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Decomposition of $\mathrm{Sym}^a V \otimes \mathrm{Sym}^b V^*$)</span></p>

For $b \le a$, we have the complete decomposition

$$\mathrm{Sym}^a V \otimes \mathrm{Sym}^b V^* = \bigoplus_{i=0}^{b} \Gamma_{a-i,\, b-i}. \tag{13.5}$$

</div>

#### Multiplicities in $\Gamma_{a,b}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Shape and Multiplicities of $\Gamma_{a,b}$)</span></p>

For $a \ge b$, the weight diagram of $\Gamma_{a,b}$ (or $\mathrm{Sym}^a V \otimes \mathrm{Sym}^b V^*$) looks like a sequence of $b$ shrinking concentric (not in general regular) hexagons $H_i$ with vertices at the points $(a - i)L_1 - (b-i)L_3$ for $i = 0, 1, \ldots, b-1$, followed (after the shorter three sides of the hexagons have shrunk to points) by a sequence of $\lfloor (a-b)/3 \rfloor + 1$ triangles $T_j$ with vertices at the points $(a - b - 3j)L_1$ for $j = 0, 1, \ldots$

The multiplicities of $\Gamma_{a,b}$ **increase by one on each concentric hexagon** and are **constant on the triangles**: specifically, $\Gamma_{a,b}$ has multiplicity $(i+1)$ on $H_i$ and $b + 1$ on $T_j$.

Note in particular that $\Gamma_{2,1}$ from the preceding section is a special case of this.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Claim 13.4 — Uniqueness of Highest Weight Vectors)</span></p>

The claim is equivalent to asserting that $W = \mathrm{Sym}^a V \otimes \mathrm{Sym}^b V^*$ has exactly $b + 1$ irreducible components (assuming $a \ge b$). This in turn is equivalent to saying that the eigenspace $W_\alpha$ of $W$ contains a unique highest weight vector (up to scalars) if $\alpha$ is of the form $(a-i)L_1 - (b-i)L_3$ for $i \le b$, and none otherwise.

One proves this by explicit calculation: writing elements of $W_\alpha$ as $v = \sum c_I \cdot (e_1^{a-i} \cdot e^I) \otimes ((e_3^*)^{b-i} \cdot (e^*)^J)$ and checking that $v$ is in the kernel of $E_{1,2}$ and $E_{2,3}$ if and only if all the coefficients $c_I$ are proportional to $c / (i_1! i_2! i_3!)$ for some constant $c$. $\square$

</div>

### 13.3 A Little More Plethysm

Given our knowledge of the eigenvalue diagrams of the irreducible representations of $\mathfrak{sl}_3\mathbb{C}$ (with multiplicities), there can be no possible ambiguity about the decomposition of any representation $U$ given as the tensor product of representations whose eigenvalue diagrams are known.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Decomposition Algorithm)</span></p>

Given a representation $U$:

1. Write down the eigenvalue decomposition of $U$.
2. Find the eigenvalue $\alpha = aL_1 - bL_3$ appearing in this diagram for which $l(\alpha)$ is maximal.
3. Then $U$ contains a copy of $\Gamma_{a,b}$, i.e., $U \cong \Gamma_\alpha \oplus U'$ for some $U'$. Since we know the eigenvalue diagram of $\Gamma_\alpha$, we can write down the eigenvalue diagram of $U'$ as well.
4. Repeat this process for $U'$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tensor Product $V \otimes \Gamma_{2,1}$)</span></p>

Since $\Gamma_{2,1}$ has weights $2L_i - L_j$, $L_i + L_j - L_k$, and $L_i$ (taken twice), and $V$ has weights $L_i$, the tensor product $V \otimes \Gamma_{2,1}$ will have weights $3L_i - L_j$, $2L_i + L_j - L_k$ (taken twice), $2L_i$ (taken four times), and $L_i + L_j$ (taken five times). By the decomposition algorithm:

$$V \otimes \Gamma_{2,1} = \Gamma_{3,1} \oplus \Gamma_{1,2} \oplus \Gamma_{2,0}. \tag{13.7}$$

</div>

#### Symmetric and Exterior Powers

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Symmetric Square $\mathrm{Sym}^2(\mathrm{Sym}^2 V)$)</span></p>

Let $W = \mathrm{Sym}^2 V = \Gamma_{2,0}$. The eigenvalue diagram of $\mathrm{Sym}^2 W = \mathrm{Sym}^2(\mathrm{Sym}^2 V)$ has a unique possible decomposition:

$$\mathrm{Sym}^2(\mathrm{Sym}^2 V) \cong \mathrm{Sym}^4 V \oplus \mathrm{Sym}^2 V^*.$$

The presence of the $\mathrm{Sym}^4 V$ factor is clear: there is a natural multiplication map $\varphi\colon \mathrm{Sym}^2(\mathrm{Sym}^2 V) \to \mathrm{Sym}^4 V$. The kernel can be identified with $\mathrm{Sym}^2(\bigwedge^2 V)$, via the map $\tau\colon \mathrm{Sym}^2(\bigwedge^2 V) \to \mathrm{Sym}^2(\mathrm{Sym}^2 V)$ sending $(u \wedge v) \cdot (w \wedge z)$ to $(u \cdot w)(v \cdot z) - (u \cdot z)(v \cdot w)$, which lies visibly in $\ker(\varphi)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Symmetric Cube $\mathrm{Sym}^3(\mathrm{Sym}^2 V)$)</span></p>

By eigenvalue analysis (drawing one-sixth of the plane and labeling weights with multiplicities):

$$\mathrm{Sym}^3(\mathrm{Sym}^2 V) \cong \mathrm{Sym}^6 V \oplus \Gamma_{2,2} \oplus \mathbb{C}. \tag{13.15}$$

The first map is the obvious multiplication $\mathrm{Sym}^3(\mathrm{Sym}^2 V) \to \mathrm{Sym}^6 V$. The trivial summand $\mathbb{C}$ corresponds to the fact that there exists a cubic hypersurface $X$ in $\mathbb{P}^5 = \mathbb{P}(\mathrm{Sym}^2 V^*)$ preserved under all automorphisms of $\mathbb{P}^5$ carrying the Veronese surface $S$ into itself: this is the determinant of the $3 \times 3$ symmetric matrix $(Z_{i,j})$, which is the chordal variety of $S$ (the union of its chords, and at the same time the union of all tangent planes to $S$).

</div>

### 13.4 A Little More Geometric Plethysm

Just as in the case of $\mathfrak{sl}_2\mathbb{C}$, some of these identifications can be seen in geometric terms.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Veronese Surface)</span></p>

Recall from §11.3 the definition of the Veronese embedding. If $\mathbb{P}^2 = \mathbb{P}V^*$ is the projective space of one-dimensional subspaces of $V^*$, there is a natural embedding of $\mathbb{P}^2$ in $\mathbb{P}^5 = \mathbb{P}(\mathrm{Sym}^2 V^*)$, sending $[v^*] \in \mathbb{P}^2$ to $[v^{*2}] \in \mathbb{P}^5$. The image $S \subset \mathbb{P}^5$ is the **Veronese surface**.

The group of automorphisms of $\mathbb{P}^5$ carrying $S$ to itself is exactly $\mathrm{PGL}_3\mathbb{C}$, the group of automorphisms of $S = \mathbb{P}^2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Quadrics Containing the Veronese Surface)</span></p>

The vector space of quadratic polynomials vanishing on the Veronese surface $S$ is the kernel of the natural evaluation map $\varphi\colon \mathrm{Sym}^2(\mathrm{Sym}^2 V) \to \mathrm{Sym}^4 V$, which we identified above with $\mathrm{Sym}^2 V^*$. The Veronese surface can be represented as the locus of rank 1 among symmetric $3 \times 3$ matrices $(Z_{i,j})$:

$$S = \left\lbrace [Z] : \mathrm{rank}\begin{pmatrix} Z_{1,1} & Z_{1,2} & Z_{1,3} \\\ Z_{1,2} & Z_{2,2} & Z_{2,3} \\\ Z_{1,3} & Z_{2,3} & Z_{3,3} \end{pmatrix} = 1 \right\rbrace.$$

The space of quadratic polynomials vanishing on $S$ is generated by the $2 \times 2$ minors of this matrix. There is also a unique **cubic** hypersurface $X$ in $\mathbb{P}^5$ preserved by $\mathrm{PGL}_3\mathbb{C}$: the determinant $\det(Z_{i,j})$, which is the chordal variety (union of chords and tangent planes) of $S$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Quadrics of Rank 3 and the Invariant Surface $T$)</span></p>

There must also be a surface $T = \mathbb{P}(V^*) \subset \mathbb{P}(\ker(\varphi))$ invariant under the action of $\mathrm{PGL}_3\mathbb{C}$ on the space of quadrics containing the Veronese. This turns out to be the locus of **quadrics of rank 3** containing $S$, that is, the quadrics whose singular locus is a 2-plane. The 2-plane will be the tangent plane to $S$ at a point, giving the identification $T = S$.

</div>

---

# Part III: The Classical Lie Algebras and Their Representations

The analysis carried out for $\mathfrak{sl}_2\mathbb{C}$ and $\mathfrak{sl}_3\mathbb{C}$ carries over to other semisimple complex Lie algebras. In Lecture 14 we codify this structure, using the pattern of the examples to give a model for the analysis of arbitrary semisimple Lie algebras. The facts themselves will all be seen explicitly on a case-by-case basis for the classical Lie algebras $\mathfrak{sl}_n\mathbb{C}$, $\mathfrak{sp}_{2n}\mathbb{C}$, and $\mathfrak{so}_n\mathbb{C}$ in Lectures 15–20.

---

## Lecture 14: The General Setup — Analyzing the Structure and Representations of an Arbitrary Semisimple Lie Algebra

This is the last of the four central lectures. In §14.1, we extract from the examples of §11–13 the basic algorithm for analyzing a general semisimple Lie algebra and its representations. In §14.2, we introduce the Killing form.

### 14.1 Analyzing Simple Lie Algebras in General

The process we describe here is directly analogous, step by step, to that carried out in Lecture 12 for $\mathfrak{sl}_3\mathbb{C}$.

#### Step 1: The Cartan Subalgebra

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cartan Subalgebra)</span></p>

Find an abelian subalgebra $\mathfrak{h} \subset \mathfrak{g}$ acting diagonally on one faithful (and hence, by Theorem 9.20, on any) representation of $\mathfrak{g}$. Moreover, $\mathfrak{h}$ should be maximal among abelian, diagonalizable subalgebras; such a subalgebra is called a **Cartan subalgebra**.

</div>

#### Step 2: The Root Space Decomposition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cartan Decomposition, Roots, Root Spaces)</span></p>

Let $\mathfrak{h}$ act on $\mathfrak{g}$ by the adjoint representation and decompose $\mathfrak{g}$ accordingly. We arrive at the **Cartan decomposition**

$$\mathfrak{g} = \mathfrak{h} \oplus \left( \bigoplus_\alpha \mathfrak{g}_\alpha \right), \tag{14.1}$$

where $\alpha$ ranges over a finite set of nonzero linear functionals on $\mathfrak{h}$: for any $H \in \mathfrak{h}$ and $X \in \mathfrak{g}_\alpha$,

$$\mathrm{ad}(H)(X) = \alpha(H) \cdot X.$$

The $\alpha \in \mathfrak{h}^*$ appearing are called the **roots** of the Lie algebra, and the corresponding subspaces $\mathfrak{g}_\alpha$ are the **root spaces**. The set of all roots is denoted $R \subset \mathfrak{h}^*$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(14.3 — Maximality of the Cartan Subalgebra)</span></p>

If $0$ were to appear as an eigenvalue of $\mathfrak{h}$ on $\mathfrak{g}/\mathfrak{h}$, then the $0$-eigenspace would commute with $\mathfrak{h}$ and (given that the $\mathfrak{g}_\alpha$ are one-dimensional and act diagonally) could be used to enlarge $\mathfrak{h}$ while retaining its properties of being abelian and diagonalizable. This would contradict maximality. Similarly, the assertion that the roots span $\mathfrak{h}^*$ follows from the fact that an element of $\mathfrak{h}$ in the annihilator of all roots would be in the center of $\mathfrak{g}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Basic Properties of Roots)</span></p>

We state (and will verify case by case) the following facts:

1. Each root space $\mathfrak{g}_\alpha$ is **one-dimensional**.
2. $R$ generates a lattice $\Lambda_R \subset \mathfrak{h}^*$ of rank equal to $\dim \mathfrak{h}$.
3. $R$ is **symmetric** about the origin: if $\alpha \in R$, then $-\alpha \in R$.

The roots all lie in (and span) a real subspace of $\mathfrak{h}^*$; all our pictures will be drawn in this real subspace.

</div>

#### Step 3: The Decomposition of a Representation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weights, Weight Vectors, Weight Spaces, Weight Diagram)</span></p>

For any irreducible finite-dimensional representation $V$ of $\mathfrak{g}$, we have a direct sum decomposition

$$V = \bigoplus V_\alpha, \tag{14.4}$$

where $\mathfrak{h}$ acts diagonally on each $V_\alpha$ by $H(v) = \alpha(H) \cdot v$. The $\alpha \in \mathfrak{h}^*$ that appear are called the **weights** of the representation; the $V_\alpha$ are **weight spaces**; vectors in $V_\alpha$ are **weight vectors**; and $\dim V_\alpha$ is the **multiplicity** of $\alpha$. The picture of dots (one per weight, with annotation for multiplicities) is called the **weight diagram** of $V$.

The action of the rest of $\mathfrak{g}$ is: for any root $\beta$, the root space $\mathfrak{g}_\beta$ maps $V_\alpha$ to $V_{\alpha + \beta}$ (translation in the weight diagram).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Congruence of Weights)</span></p>

All the weights of an irreducible representation are congruent to one another modulo the root lattice $\Lambda_R$: otherwise $V' = \bigoplus_{\beta \in \Lambda_R} V_{\alpha + \beta}$ would be a proper subrepresentation.

</div>

#### Step 4: The Distinguished Subalgebras $\mathfrak{s}_\alpha \cong \mathfrak{sl}_2\mathbb{C}$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Distinguished Subalgebras)</span></p>

For any root $\alpha \in R$, since $\mathfrak{g}_\alpha$ is one-dimensional and $-\alpha$ is also a root, we can pick a basis $X_\alpha \in \mathfrak{g}_\alpha$, $Y_\alpha \in \mathfrak{g}_{-\alpha}$, and set $H_\alpha = [X_\alpha, Y_\alpha] \in \mathfrak{h}$. Then the direct sum

$$\mathfrak{s}_\alpha = \mathfrak{g}_\alpha \oplus \mathfrak{g}_{-\alpha} \oplus [\mathfrak{g}_\alpha, \mathfrak{g}_{-\alpha}] \tag{14.5}$$

is a subalgebra of $\mathfrak{g}$ isomorphic to $\mathfrak{sl}_2\mathbb{C}$, carrying $E_{i,j} \mapsto X_\alpha$, $E_{j,i} \mapsto Y_\alpha$, $H_{i,j} \mapsto H_\alpha$, with $\alpha(H_\alpha) = 2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Facts 14.6)</span></p>

Two further facts about the distinguished subalgebras (to be verified case by case and proved in general in Appendix D):

1. $[\mathfrak{g}_\alpha, \mathfrak{g}_{-\alpha}] \neq 0$; and
2. $[[\mathfrak{g}_\alpha, \mathfrak{g}_{-\alpha}], \mathfrak{g}_\alpha] \neq 0$.

Given these, $\mathfrak{s}_\alpha$ is indeed isomorphic to $\mathfrak{sl}_2\mathbb{C}$, with $H_\alpha$ uniquely characterized by $H_\alpha \in [\mathfrak{g}_\alpha, \mathfrak{g}_{-\alpha}]$ and $\alpha(H_\alpha) = 2$.

</div>

#### Step 5: Integrality — The Weight Lattice

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weight Lattice)</span></p>

By the analysis of $\mathfrak{sl}_2\mathbb{C}$ representations, all eigenvalues of $H_\alpha$ in any representation of $\mathfrak{s}_\alpha$ — and hence in any representation of $\mathfrak{g}$ — must be integers. We correspondingly let $\Lambda_W$ be the set of linear functionals $\beta \in \mathfrak{h}^*$ that are integer-valued on all the $H_\alpha$; this is a lattice called the **weight lattice** of $\mathfrak{g}$, with the property that

*all weights of all representations of $\mathfrak{g}$ will lie in $\Lambda_W$.*

Note that $R \subset \Lambda_W$ and hence $\Lambda_R \subset \Lambda_W$; in fact the root lattice will in general be a sublattice of finite index in the weight lattice.

</div>

#### Step 6: The Weyl Group and Symmetry

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weyl Group)</span></p>

For any root $\alpha$, introduce the involution $W_\alpha$ on $\mathfrak{h}^*$ with $+1$-eigenspace the hyperplane

$$\Omega_\alpha = \lbrace \beta \in \mathfrak{h}^* : \langle H_\alpha, \beta \rangle = 0 \rbrace \tag{14.7}$$

and $-1$-eigenspace the line spanned by $\alpha$ itself. Explicitly,

$$W_\alpha(\beta) = \beta - \frac{2\beta(H_\alpha)}{\alpha(H_\alpha)} \alpha = \beta - \beta(H_\alpha) \alpha. \tag{14.8}$$

The group $\mathfrak{W}$ generated by these involutions is called the **Weyl group** of the Lie algebra $\mathfrak{g}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Symmetry of Weights Under the Weyl Group)</span></p>

For any representation $V = \bigoplus V_\beta$ and any root $\alpha$, the direct sum

$$V_{[\beta]} = \bigoplus_{n \in \mathbb{Z}} V_{\beta + n\alpha} \tag{14.9}$$

is a representation of $\mathfrak{s}_\alpha \cong \mathfrak{sl}_2\mathbb{C}$, and the string of weights $\beta, \beta + \alpha, \beta + 2\alpha, \ldots, \beta + m\alpha$ (with $m = -\beta(H_\alpha)$) is an unbroken string symmetric about zero for $H_\alpha$. In particular, $W_\alpha(\beta + k\alpha) = \beta + (m-k)\alpha$.

It follows that **the set of weights (with multiplicities) of any representation of $\mathfrak{g}$ is invariant under the Weyl group**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fact 14.11 — Weyl Group Elements as Automorphisms)</span></p>

Every element of the Weyl group is induced by an automorphism of the Lie algebra $\mathfrak{g}$ carrying $\mathfrak{h}$ to itself. Specifically, to get the involution $W_\alpha$, take the adjoint action of $\exp(\pi i U_\alpha) \in G$ where $U_\alpha$ is a suitable element of the direct sum of $\mathfrak{g}_\alpha$ and $\mathfrak{g}_{-\alpha}$.

</div>

#### Step 7: Ordering of Roots and Highest Weight Vectors

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Positive and Negative Roots, Ordering)</span></p>

Choose a direction in $\mathfrak{h}^*$, i.e., a real linear functional $l$ on the lattice $\Lambda_R$ irrational with respect to this lattice. This gives a decomposition

$$R = R^+ \cup R^-, \tag{14.12}$$

where $R^+ = \lbrace \alpha : l(\alpha) > 0 \rbrace$ are the **positive roots** and $R^- = \lbrace \alpha : l(\alpha) < 0 \rbrace$ the **negative roots**. This decomposition is called an **ordering of the roots**.

A positive (resp. negative) root $\alpha \in R$ is called **primitive** or **simple** if it cannot be expressed as a sum of two positive (resp. negative) roots. Since there are only finitely many roots, every positive root is a sum of primitive positive roots.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Highest Weight Vector, Weyl Chamber)</span></p>

A nonzero vector $v \in V$ that is both an eigenvector for $\mathfrak{h}$ and in the kernel of $\mathfrak{g}_\alpha$ for all $\alpha \in R^+$ is called a **highest weight vector**.

The locus $\mathscr{W}$ in the real span of the roots of points $\alpha$ satisfying $\alpha(H_\gamma) \ge 0$ for every positive root $\gamma$ (equivalently, for every primitive positive root) is called the **(closed) Weyl chamber** associated to the ordering. The Weyl group acts simply transitively on the set of Weyl chambers and likewise on the set of orderings of the roots.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Observation 14.16 — Generation by Primitive Negative Roots)</span></p>

Any irreducible representation $V$ is generated by the images of its highest weight vector $v$ under successive applications of root spaces $\mathfrak{g}_\beta$ where $\beta$ ranges over the **primitive negative roots** only.

The set of weights of $V$ will consist of those elements of the weight lattice $\Lambda_W$ congruent to $\alpha$ modulo $\Lambda_R$ and lying in the convex hull of the images of $\alpha$ under the Weyl group.

</div>

#### Step 8: The Classification Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(14.18 — Existence and Uniqueness)</span></p>

For any $\alpha$ in the intersection of the Weyl chamber $\mathscr{W}$ associated to the ordering of the roots with the weight lattice $\Lambda_W$, there exists a **unique** irreducible, finite-dimensional representation $\Gamma_\alpha$ of $\mathfrak{g}$ with highest weight $\alpha$; this gives a bijection between $\mathscr{W} \cap \Lambda_W$ and the set of irreducible representations of $\mathfrak{g}$.

The set of weights of $\Gamma_\alpha$ will consist of those elements of the weight lattice congruent to $\alpha$ modulo the root lattice $\Lambda_R$ and lying in the convex hull of the set of points in $\mathfrak{h}^*$ conjugate to $\alpha$ under the Weyl group.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fundamental Weights)</span></p>

There are always **fundamental weights** $\omega_1, \ldots, \omega_n$ with the property that any dominant weight can be expressed uniquely as a non-negative integral linear combination of them. They can be characterized geometrically as the first weights met along the edges of the Weyl chamber, or algebraically as those elements $\omega_i \in \mathfrak{h}^*$ such that $\omega_i(H_{\alpha_j}) = \delta_{i,j}$, where $\alpha_1, \ldots, \alpha_n$ are the simple roots. We often write $\Gamma_{a_1, \ldots, a_n}$ for the irreducible representation with highest weight $a_1\omega_1 + \cdots + a_n \omega_n$.

</div>

### 14.2 About the Killing Form

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Killing Form)</span></p>

The **Killing form** is a symmetric bilinear form on the Lie algebra $\mathfrak{g}$, defined by

$$B(X, Y) = \mathrm{Tr}(\mathrm{ad}(X) \circ \mathrm{ad}(Y)\colon \mathfrak{g} \to \mathfrak{g}). \tag{14.19}$$

It can be computed in practice either from this definition or (up to scalars) by using its invariance under the group of automorphisms of $\mathfrak{g}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Orthogonality Properties)</span></p>

The decomposition

$$\mathfrak{g} = \mathfrak{h} \oplus \left( \bigoplus_{\alpha \in R^+} (\mathfrak{g}_\alpha \oplus \mathfrak{g}_{-\alpha}) \right) \tag{14.20}$$

is orthogonal with respect to $B$: since $\mathfrak{g}_\alpha$ carries $\mathfrak{g}_\beta$ into $\mathfrak{g}_{\alpha+\beta}$ (a "translation" in the root diagram), the trace of $\mathrm{ad}(X) \circ \mathrm{ad}(Y)$ is zero unless $X \in \mathfrak{g}_\alpha$ and $Y \in \mathfrak{g}_{-\alpha}$.

The restriction of $B$ to $\mathfrak{h}$ is given by

$$B|_\mathfrak{h} = \frac{1}{2} \sum_{\alpha \in R} \alpha^2, \tag{14.21}$$

viewed as an element of $\mathrm{Sym}^2(\mathfrak{h}^*)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Positive Definiteness)</span></p>

$B$ is positive definite on the real subspace of $\mathfrak{h}$ spanned by the vectors $\lbrace H_\alpha : \alpha \in R \rbrace$ (since all roots take real values on this subspace, and they span $\mathfrak{h}^*$). $B$ is zero only when $\alpha(H) = 0$ for all $\alpha$, which forces $H = 0$ since the roots span $\mathfrak{h}^*$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Associativity of the Killing Form)</span></p>

A key identity is

$$B([X, Y], Z) = B(X, [Y, Z]) \tag{14.23}$$

for all $X, Y, Z \in \mathfrak{g}$. This follows from the trace identity $\mathrm{Trace}(\overline{X}\overline{Y}\overline{Z} - \overline{Y}\overline{X}\overline{Z}) = \mathrm{Trace}(\overline{X}(\overline{Y}\overline{Z} - \overline{Z}\overline{Y}))$ for any endomorphisms. An immediate consequence is that if $\mathfrak{a}$ is any ideal in a Lie algebra $\mathfrak{g}$, then $\mathfrak{a}^\perp$ (with respect to $B$) is also an ideal. In particular, if $\mathfrak{g}$ is simple, the kernel of $B$ is zero, and so **the Killing form is nondegenerate on a semisimple Lie algebra**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(14.24 — Roots Perpendicular to Hyperplanes)</span></p>

With respect to $B$, the line spanned by each root $\alpha$ is perpendicular to the hyperplane $\Omega_\alpha$. Equivalently, the involutions $W_\alpha$ are simply reflections in the hyperplanes $\Omega_\alpha$, so that the **Weyl group is the group generated by reflections in the hyperplanes perpendicular to the roots** of the Lie algebra.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Proposition 14.24)</span></p>

To prove $\alpha \perp \Omega_\alpha$, it suffices to show $H \perp H_\alpha$ for all $H$ in the annihilator of $\alpha$. But $H_\alpha = [X_\alpha, Y_\alpha]$ is the commutator of $X_\alpha \in \mathfrak{g}_\alpha$ and $Y_\alpha \in \mathfrak{g}_{-\alpha}$. By (14.23),

$$B(H_\alpha, H) = B([X_\alpha, Y_\alpha], H) = B(X_\alpha, [Y_\alpha, H]) = B(X_\alpha, \alpha(H) Y_\alpha) = \alpha(H) B(X_\alpha, Y_\alpha), \tag{14.25}$$

which vanishes when $\alpha(H) = 0$. $\square$

</div>

#### The Killing Form on $\mathfrak{h}^*$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Isomorphism $\mathfrak{h} \cong \mathfrak{h}^*$ via $B$)</span></p>

Since $B$ is nondegenerate on $\mathfrak{h}$, it determines an isomorphism $\mathfrak{h} \cong \mathfrak{h}^*$. The element $T_\alpha \in \mathfrak{h}$ corresponding to $\alpha \in \mathfrak{h}^*$ is defined by

$$B(T_\alpha, H) = \alpha(H) \quad \text{for all } H \in \mathfrak{h}. \tag{14.26}$$

Looking at (14.25), we see that $T_\alpha = H_\alpha / B(X_\alpha, Y_\alpha) = 2H_\alpha / B(H_\alpha, H_\alpha)$. This proves:

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(14.27)</span></p>

The isomorphism of $\mathfrak{h}^*$ and $\mathfrak{h}$ determined by the Killing form $B$ carries $\alpha$ to $T_\alpha = (2/B(H_\alpha, H_\alpha)) \cdot H_\alpha$. The Killing form on $\mathfrak{h}^*$ is defined by $B(\alpha, \beta) = B(T_\alpha, T_\beta)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(14.29 — Integrality of $B$)</span></p>

If $\alpha$ and $\beta$ are roots, then

$$\frac{2B(\beta, \alpha)}{B(\alpha, \alpha)} = \beta(H_\alpha)$$

is an **integer**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(14.30)</span></p>

The Killing form $B$ is **positive definite** on the real vector space spanned by the root lattice $\Lambda_R$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(14.31 — $\mathfrak{h}^*$ is an Irreducible Representation of $\mathfrak{W}$)</span></p>

The space $\mathfrak{h}^*$ is an **irreducible** representation of the Weyl group $\mathfrak{W}$.

**Proof.** Suppose a subspace $\mathfrak{z} \subset \mathfrak{h}^*$ were preserved by $\mathfrak{W}$. Then every root $\alpha$ would either lie in $\mathfrak{z}$ or be perpendicular to it (since $W_\alpha$ is a reflection). If $\mathfrak{z} \neq \mathfrak{h}^*$ and $\mathfrak{z} \neq 0$, the subspace $\mathfrak{g}' = \bigoplus_{\alpha \in \mathfrak{z}} \mathfrak{s}_\alpha$ (together with the corresponding part of $\mathfrak{h}$) would be a proper ideal of $\mathfrak{g}$, contradicting simplicity. Hence either all roots lie in $\mathfrak{z}$ (so $\mathfrak{z} = \mathfrak{h}^*$) or all are perpendicular (so $\mathfrak{z} = 0$). $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Uniqueness of the Killing Form)</span></p>

If $\mathfrak{g}$ is simple, the Killing form on $\mathfrak{h}$ is the unique form preserved by the Weyl group (up to scalars), by Proposition 14.31 and Schur's lemma. In practice, this is most often how we will compute it: find the Killing form first, then verify the statements about the root system.

</div>

---

## Lecture 15: $\mathfrak{sl}_4\mathbb{C}$ and $\mathfrak{sl}_n\mathbb{C}$

In this lecture we illustrate the general paradigm of Lecture 14 by applying it to the Lie algebras $\mathfrak{sl}_4\mathbb{C}$ and $\mathfrak{sl}_n\mathbb{C}$.

### 15.1 Analyzing $\mathfrak{sl}_n\mathbb{C}$

#### Cartan Subalgebra and Roots

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cartan Subalgebra and Roots of $\mathfrak{sl}_n\mathbb{C}$)</span></p>

The Cartan subalgebra is the space of traceless diagonal matrices:

$$\mathfrak{h} = \lbrace a_1 H_1 + a_2 H_2 + \cdots + a_n H_n : a_1 + a_2 + \cdots + a_n = 0 \rbrace,$$

where $H_i = E_{i,i}$. We write $\mathfrak{h}^* = \mathbb{C}\lbrace L_1, \ldots, L_n \rbrace / (L_1 + L_2 + \cdots + L_n = 0)$, with $L_i(H_j) = \delta_{i,j}$.

The adjoint action on $\mathfrak{sl}_n\mathbb{C}$ gives

$$\mathrm{ad}(a_1 H_1 + \cdots + a_n H_n)(E_{i,j}) = (a_i - a_j) \cdot E_{i,j}, \tag{15.1}$$

so **the roots of $\mathfrak{sl}_n\mathbb{C}$ are exactly the pairwise differences $\lbrace L_i - L_j \rbrace_{i \neq j}$**.

</div>

#### The Killing Form

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Killing Form of $\mathfrak{sl}_n\mathbb{C}$)</span></p>

The Killing form must be invariant under the automorphism $\varphi$ of $\mathfrak{sl}_n\mathbb{C}$ that exchanges $e_i$ and $e_j$. This forces $B(L_i, L_i) = B(L_j, L_j)$ for all $i, j$ and $B(L_i, L_k) = B(L_j, L_k)$ for all $i, j$ and $k \neq i, j$. The unique such form (up to scalars) is

$$B\!\left(\sum a_i H_i,\, \sum b_i H_i\right) = 2n \sum a_i b_i, \tag{15.3}$$

or equivalently the dual form on $\mathfrak{h}^*$ is

$$B\!\left(\sum a_i L_i,\, \sum b_i L_i\right) = \frac{1}{2n}\!\left(\sum_i a_i b_i - \frac{1}{n}\sum_{i,j} a_i b_j\right). \tag{15.4}$$

Up to scalars, this is the unique form invariant under the symmetric group $\mathfrak{S}_n$ permuting the $L_i$. The $L_i$ are thus situated at the vertices of a regular $(n-1)$-simplex $\Delta$ centered at the origin.

</div>

#### Distinguished Subalgebras and Lattices

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Distinguished Subalgebras, Weight and Root Lattices)</span></p>

The root space $\mathfrak{g}_{L_i - L_j}$ is generated by $E_{i,j}$, and the subalgebra $\mathfrak{s}_{L_i - L_j}$ is spanned by $E_{i,j}$, $E_{j,i}$, and $[E_{i,j}, E_{j,i}] = H_i - H_j$. The distinguished element is $H_{L_i - L_j} = H_i - H_j$ (with $(L_i - L_j)(H_i - H_j) = 2$, as required).

The **weight lattice** is

$$\Lambda_W = \mathbb{Z}\lbrace L_1, \ldots, L_n \rbrace / \left(\textstyle\sum L_i = 0\right),$$

i.e., the lattice generated by the vertices of the simplex $\Delta$. The **root lattice** is

$$\Lambda_R = \left\lbrace \textstyle\sum a_i L_i : a_i \in \mathbb{Z},\, \sum a_i = 0 \right\rbrace.$$

The quotient $\Lambda_W / \Lambda_R \cong \mathbb{Z}/n\mathbb{Z}$ (generated by any $L_i$).

</div>

#### Weyl Group, Ordering, and Weyl Chamber

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weyl Group and Weyl Chamber of $\mathfrak{sl}_n\mathbb{C}$)</span></p>

The reflection in the hyperplane perpendicular to the root $L_i - L_j$ exchanges $L_i$ and $L_j$ and leaves the other $L_k$ alone, so **the Weyl group $\mathfrak{W}$ is the symmetric group $\mathfrak{S}_n$**, acting by permuting the generators $L_i$ of $\mathfrak{h}^*$.

Choosing the linear functional $l(\sum a_i L_i) = \sum c_i a_i$ with $c_1 > c_2 > \cdots > c_n$, the positive roots are $R^+ = \lbrace L_i - L_j : i < j \rbrace$, the simple (primitive) roots are $L_i - L_{i+1}$ for $i = 1, \ldots, n-1$, and the **(closed) Weyl chamber** is

$$\mathscr{W} = \left\lbrace \textstyle\sum a_i L_i : a_1 \ge a_2 \ge \cdots \ge a_n \right\rbrace.$$

Geometrically, $\mathscr{W}$ is the cone over one $(n-2)$-simplex of the barycentric subdivision of $\Delta$, with edges generated by the vectors $L_1$, $L_1 + L_2$, $\ldots$, $L_1 + \cdots + L_{n-1} = -L_n$.

</div>

#### Fundamental Weights and Indexing

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fundamental Weights of $\mathfrak{sl}_n\mathbb{C}$)</span></p>

The **fundamental weights** are

$$\omega_i = L_1 + L_2 + \cdots + L_i, \qquad i = 1, \ldots, n-1.$$

Every dominant weight is a unique non-negative integral linear combination of them: for an $(n-1)$-tuple $(a_1, \ldots, a_{n-1}) \in \mathbb{N}^{n-1}$, the irreducible representation $\Gamma_{a_1, \ldots, a_{n-1}}$ has highest weight

$$a_1 L_1 + a_2(L_1 + L_2) + \cdots + a_{n-1}(L_1 + \cdots + L_{n-1}).$$

The general irreducible representation $\Gamma_{a_1, \ldots, a_{n-1}}$ appears inside the tensor product $\mathrm{Sym}^{a_1} V \otimes \mathrm{Sym}^{a_2}(\bigwedge^2 V) \otimes \cdots \otimes \mathrm{Sym}^{a_{n-1}}(\bigwedge^{n-1} V)$, establishing the existence part of Theorem 14.18 for $\mathfrak{sl}_n\mathbb{C}$.

</div>

### 15.2 Representations of $\mathfrak{sl}_4\mathbb{C}$ and $\mathfrak{sl}_n\mathbb{C}$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Basic Representations of $\mathfrak{sl}_4\mathbb{C}$)</span></p>

Let $V = \mathbb{C}^4$ be the standard representation with weights $L_1, L_2, L_3, L_4$.

- $V^* \cong \bigwedge^3 V$ has weights $-L_i$, equivalently $L_j + L_k + L_l$ ($\lbrace j,k,l \rbrace = \lbrace 1,2,3,4 \rbrace \setminus \lbrace i \rbrace$). Highest weight: $-L_4 = L_1 + L_2 + L_3$.
- $\bigwedge^2 V$ has weights $L_i + L_j$ ($i < j$), a six-dimensional representation. All weights have multiplicity 1 and the diagram is symmetric about the origin (reflecting $\bigwedge^2 V \cong (\bigwedge^2 V)^*$). It is irreducible. Highest weight: $L_1 + L_2$.

Since $V$, $\bigwedge^2 V$, and $\bigwedge^3 V = V^*$ have highest weight vectors with weights $L_1$, $L_1 + L_2$, and $L_1 + L_2 + L_3$ respectively — exactly the primitive vectors along the edges of the Weyl chamber — **the existence and uniqueness theorem (14.18) is established** for $\mathfrak{sl}_4\mathbb{C}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tensor Products of Basic Representations of $\mathfrak{sl}_4\mathbb{C}$)</span></p>

- $V \otimes \bigwedge^2 V$: weights are $2L_i + L_j$ (once) and $L_i + L_j + L_k$ (three times). There is a natural surjection $\varphi\colon V \otimes \bigwedge^2 V \to \bigwedge^3 V$, whose kernel (with weights $L_i + L_j + L_k$ having multiplicity 2 rather than 3) contains the irreducible representation $\Gamma_{1,1,0}$ with highest weight $2L_1 + L_2$:

$$V \otimes \bigwedge^2 V \cong \Gamma_{1,1,0} \oplus V^*.$$

- $V \otimes \bigwedge^3 V = V \otimes V^* \cong \mathrm{Hom}(V, V) \to \mathbb{C}$ (trace map), giving $V \otimes V^* \cong \Gamma_{1,0,1} \oplus \mathbb{C}$, where $\Gamma_{1,0,1}$ is the adjoint representation with highest weight $2L_1 + L_2 + L_3 = L_1 - L_4$.

</div>

#### Representations of $\mathfrak{sl}_n\mathbb{C}$ in General

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The General Case)</span></p>

For $\mathfrak{sl}_n\mathbb{C}$, the standard representation $V = \mathbb{C}^n$ has highest weight $L_1$, and each exterior power $\bigwedge^k V$ is irreducible with highest weight $L_1 + \cdots + L_k$. These are the $n-1$ **fundamental representations** $V^{(k)} = \bigwedge^k V$.

The irreducible representation $\Gamma_{a_1, \ldots, a_{n-1}}$ with highest weight $(a_1 + \cdots + a_{n-1})L_1 + a_2(L_1 + L_2) + \cdots$ appears inside the tensor product $\mathrm{Sym}^{a_1} V \otimes \mathrm{Sym}^{a_2}(\bigwedge^2 V) \otimes \cdots \otimes \mathrm{Sym}^{a_{n-1}}(\bigwedge^{n-1} V)$.

</div>

### 15.3 Weyl's Construction and Tensor Products

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Describing $\Gamma_\mathbf{a}$ as a Subspace of $V^{\otimes d}$)</span></p>

The irreducible representation $\Gamma_{a_1, \ldots, a_{n-1}}$ of $\mathfrak{sl}_n\mathbb{C}$ appears as a subspace of the tensor power $V^{\otimes d}$ (where $d = \sum i \cdot a_i$). This subspace can be described as the image of a Young symmetrizer, connecting to the Schur functor construction from Lecture 6.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(15.15 — Schur Functors Give Irreducible Representations)</span></p>

The representation $\mathbb{S}_\lambda(\mathbb{C}^n)$ is the irreducible representation of $\mathfrak{sl}_n\mathbb{C}$ with highest weight $\lambda_1 L_1 + \lambda_2 L_2 + \cdots + \lambda_n L_n$.

In particular, $\mathbb{S}_\lambda(\mathbb{C}^n)$ and $\mathbb{S}_\mu(\mathbb{C}^n)$ are isomorphic representations of $\mathfrak{sl}_n\mathbb{C}$ if and only if $\lambda_i - \mu_i$ is constant, independent of $i$. To relate this to our earlier notation, the irreducible $\Gamma_{a_1, \ldots, a_{n-1}}$ of $\mathfrak{sl}_n\mathbb{C}$ with highest weight $a_1 L_1 + a_2(L_1 + L_2) + \cdots + a_{n-1}(L_1 + \cdots + L_{n-1})$ is obtained by applying $\mathbb{S}_\lambda$ to the standard representation $V$, where

$$\lambda = (a_1 + \cdots + a_{n-1},\, a_2 + \cdots + a_{n-1},\, \ldots,\, a_{n-1},\, 0).$$

</div>

#### Dimension Formula

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Dimension Formula for $\Gamma_{a_1, \ldots, a_{n-1}}$)</span></p>

By Theorem 6.3, the dimension of the irreducible representation is given by the formula

$$\dim(\Gamma_{a_1, \ldots, a_{n-1}}) = \prod_{1 \le i < j \le n} \frac{(a_i + \cdots + a_{j-1}) + j - i}{j - i}. \tag{15.17}$$

The multiplicities of weight spaces are given by the Kostka numbers $K_{\lambda\mu}$: the dimension of the weight space with weight $\mu_1 L_1 + \cdots + \mu_n L_n$ is the number of ways to fill the Young diagram of $\lambda$ with $\mu_1$ 1's, $\mu_2$ 2's, $\ldots$, $\mu_n$ $n$'s, such that entries in each row are nondecreasing and entries in each column are strictly increasing.

</div>

#### Decomposition Rules

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Littlewood–Richardson Rule)</span></p>

The tensor product of any two irreducible representations of $\mathfrak{sl}_n\mathbb{C}$ decomposes as

$$\mathbb{S}_\lambda(V) \otimes \mathbb{S}_\mu(V) = \bigoplus_\nu N_{\lambda\mu\nu}\, \mathbb{S}_\nu(V), \tag{15.23}$$

where the coefficients $N_{\lambda\mu\nu}$ are given by the **Littlewood–Richardson rule** — a combinatorial formula in terms of the number of ways to fill the Young diagram between $\lambda$ and $\nu$ with $\mu_1$ 1's, $\mu_2$ 2's, $\ldots$, satisfying a certain combinatorial condition (A.8).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(15.25 — Pieri's Formula)</span></p>

**(i)** The tensor product of $\Gamma_{a_1, \ldots, a_{n-1}}$ with $\mathrm{Sym}^k V = \Gamma_{k, 0, \ldots, 0}$ decomposes as

$$\Gamma_{a_1, \ldots, a_{n-1}} \otimes \Gamma_{k, 0, \ldots, 0} = \bigoplus \Gamma_{b_1, \ldots, b_{n-1}},$$

the sum over all $(b_1, \ldots, b_{n-1})$ for which there are non-negative integers $c_1, \ldots, c_n$ whose sum is $k$, with $c_{i+1} \le a_i$ for $1 \le i \le n-1$, and with $b_i = a_i + c_i - c_{i+1}$.

**(ii)** The tensor product with $\bigwedge^k V = \Gamma_{0, \ldots, 0, 1, 0, \ldots, 0}$ (1 in the $k$th place) decomposes as

$$\Gamma_{a_1, \ldots, a_{n-1}} \otimes \Gamma_{0, \ldots, 0, 1, 0, \ldots, 0} = \bigoplus \Gamma_{b_1, \ldots, b_{n-1}},$$

the sum over all $(b_1, \ldots, b_{n-1})$ for which there is a subset $S$ of $\lbrace 1, \ldots, n \rbrace$ of cardinality $k$, such that $b_i = a_i + 1$ if $i \notin S$ and $i+1 \in S$, $b_i = a_i - 1$ if $i \in S$ and $i+1 \notin S$, and $b_i = a_i$ otherwise, provided all $b_i > 0$.

</div>

### 15.4 Some More Geometry

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Grassmannian and Plücker Embedding)</span></p>

Let $V$ be an $n$-dimensional vector space and $G(k, n) = \mathrm{Grass}_k V$ the Grassmannian of $k$-planes in $V$. The **Plücker embedding** is

$$\rho\colon \mathrm{Grass}_k V \hookrightarrow \mathbb{P}(\bigwedge^k V),$$

sending the plane spanned by $v_1, \ldots, v_k$ to the line $\bigwedge^k W = \mathbb{C} \cdot (v_1 \wedge \cdots \wedge v_k)$. This embedding is compatible with the action of $\mathrm{PGL}_n\mathbb{C}$: the group $\mathrm{Aut}(\mathbb{P}(\bigwedge^k V))$ preserving $G(k,n)$ is exactly $\mathrm{PGL}_n\mathbb{C}$ (unless $n = 2k$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Ideal of the Grassmannian and Plücker Relations)</span></p>

The space of all homogeneous polynomials of degree $m$ on $\mathbb{P}(\bigwedge^k V^*)$ is $\mathrm{Sym}^m(\bigwedge^k V)$. Let $I(G)_m$ denote the subspace of those vanishing on $G$. Then $I(G)_m$ is a representation of $\mathfrak{sl}_n\mathbb{C}$ and we have an exact sequence

$$0 \to I(G)_m \to \mathrm{Sym}^m(\bigwedge^k V) \to W_m \to 0,$$

where $W_m$ is the irreducible representation $\Gamma_{0, \ldots, 0, m, 0, \ldots, 0}$ with highest weight $m(L_1 + \cdots + L_k)$.

The ideal $I(G)$ is generated by the **Plücker relations**, which are homogeneous of degree 2. The quadratic part $I(G)_2$ satisfies

$$\bigwedge^k V \otimes \bigwedge^k V = \mathrm{Sym}^2(\bigwedge^k V) \oplus \bigwedge^2(\bigwedge^k V),$$

and the Plücker relations span certain summands of this decomposition.

</div>

### 15.5 Representations of $\mathrm{GL}_n\mathbb{C}$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Determinant Representations)</span></p>

Let $D_k$ denote the one-dimensional representation of $\mathrm{GL}_n\mathbb{C}$ given by the $k$th power of the determinant: $D_k = (\bigwedge^n V)^{\otimes k}$. When $k$ is non-negative, $D_{-k}$ is the dual $(D_k)^*$ of $D_k$. Note that $D_k$ is trivial for $\mathrm{SL}_n\mathbb{C}$ but nontrivial for $\mathrm{GL}_n\mathbb{C}$ when $k \neq 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From $\mathrm{SL}_n\mathbb{C}$ to $\mathrm{GL}_n\mathbb{C}$)</span></p>

The Lie algebra $\mathfrak{gl}_n\mathbb{C}$ is the product $\mathfrak{sl}_n\mathbb{C} \times \mathbb{C}$, with the factor $\mathbb{C}$ being the scalar matrices. By Proposition 9.17, every irreducible representation of $\mathfrak{gl}_n\mathbb{C}$ is a tensor product of an irreducible representation of $\mathfrak{sl}_n\mathbb{C}$ and a one-dimensional representation.

For any index $\mathbf{a} = (a_1, \ldots, a_n)$ of length $n$, define $\Phi_\mathbf{a}$ to be the subrepresentation of the tensor product $\mathrm{Sym}^{a_1} V \otimes \cdots \otimes \mathrm{Sym}^{a_n}(\bigwedge^n V)$ spanned by the highest weight vector. By tensoring with $D_k$, we can extend the definition of $\Phi_\mathbf{a}$ to indices $\mathbf{a}$ with $a_n < 0$: $\Phi_{a_1, \ldots, a_n + k} = \Phi_{a_1, \ldots, a_n} \otimes D_k$.

Alternatively, applying the Schur functor $\mathbb{S}_\lambda$ to the standard representation $V$ of $\mathrm{GL}_n\mathbb{C}$ for any partition $\lambda = (\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_n)$ gives a representation $\Psi_\lambda$. We extend to arbitrary (not necessarily non-negative) sequences by $\Psi_{\lambda_1, \ldots, \lambda_n} = \Psi_{\lambda_1 + k, \ldots, \lambda_n + k} \otimes D_{-k}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(15.47 — Classification of Irreducible Representations of $\mathrm{GL}_n\mathbb{C}$)</span></p>

Every irreducible complex representation of $\mathrm{GL}_n\mathbb{C}$ is isomorphic to $\Psi_\lambda$ for a unique index $\lambda = \lambda_1, \ldots, \lambda_n$ with $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_n$ (equivalently, to $\Phi_\mathbf{a}$ for a unique $\mathbf{a} = a_1, \ldots, a_n$ with $a_1, \ldots, a_{n-1} \ge 0$).

The representations $\Psi_\lambda$ of $\mathrm{GL}_n\mathbb{C}$ that factor through $\mathrm{SL}_n\mathbb{C} \times \mathbb{C} \to \mathrm{GL}_n\mathbb{C}$ and are trivial on $\ker(\rho)$ — i.e., that are genuine representations of $\mathrm{GL}_n\mathbb{C}$ factoring through it — are exactly those with $w = \sum \lambda_i + kn$ for some integer $k$ (where $w$ is determined by the action of the center of $\mathrm{GL}_n\mathbb{C}$).

</div>

#### Weyl's Construction Revisited

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Algebra $\mathbb{S}'(V)$ and Semistandard Tableaux)</span></p>

All representations $\mathbb{S}_\lambda V$ of $\mathrm{GL}(V)$ can be constructed at once. Set $A^*(V) = \mathrm{Sym}^{\bullet}(\bigwedge^k V) \otimes \mathrm{Sym}^{\bullet}(\bigwedge^{k-1} V) \otimes \cdots \otimes \mathrm{Sym}^{\bullet}(V)$ (the tensor product of symmetric products of all positive exterior powers). Define $\mathbb{S}'(V) = A'(V)/I'$, where $I'$ is the graded two-sided ideal generated by all "Plücker relations"

$$(v_1 \wedge \cdots \wedge v_p) \cdot (w_1 \wedge \cdots \wedge w_q) - \sum_{i=1}^{p} (v_1 \wedge \cdots \wedge v_{i-1} \wedge w_1 \wedge v_{i+1} \wedge \cdots \wedge v_p) \cdot (v_i \wedge w_2 \wedge \cdots \wedge w_q) \tag{15.53}$$

for all $p \ge q \ge 1$ and all vectors. The algebra $\mathbb{S}'(V)$ is the direct sum of the images $\mathbb{S}^*(V)$ of the summands $A^*(V)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(15.55 — Basis from Semistandard Tableaux)</span></p>

**(1)** The projection from $A^*(V)$ to $\mathbb{S}^*(V)$ maps the subspace $\mathbb{S}_\lambda(V)$ isomorphically onto $\mathbb{S}^*(V)$.

**(2)** The elements $e_T$ for $T$ a semistandard tableau on $\lambda$ form a **basis** for $\mathbb{S}^*(V)$.

Here $e_T$ is the image in $\mathbb{S}^*(V)$ of the element $\prod_{j=1}^{l} e_{T(1,j)} \wedge e_{T(2,j)} \wedge \cdots \wedge e_{T(\mu_j, j)}$ (wedge together basis elements corresponding to entries in each column, then multiply results across columns in the symmetric algebra).

This gives an alternative description of $\mathbb{S}_\lambda(V)$ as the quotient of the space $A^*(V)$ by the Plücker relations.

</div>

---

## Lecture 16: Symplectic Lie Algebras

In this lecture we do for the symplectic Lie algebras exactly what we did for the special linear ones in §15.1 and most of §15.2: describe the structure in general (Cartan subalgebra, roots, Killing form, etc.), and then work out in some detail the representations of the specific algebra $\mathfrak{sp}_4\mathbb{C}$.

### 16.1 The Structure of $\mathrm{Sp}_{2n}\mathbb{C}$ and $\mathfrak{sp}_{2n}\mathbb{C}$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Symplectic Group and Lie Algebra)</span></p>

Let $V$ be a $2n$-dimensional complex vector space and $Q\colon V \times V \to \mathbb{C}$ a nondegenerate skew-symmetric bilinear form. The **symplectic Lie group** $\mathrm{Sp}_{2n}\mathbb{C}$ is the group of automorphisms $A$ of $V$ preserving $Q$:

$$Q(Av, Aw) = Q(v, w) \quad \text{for all } v, w \in V.$$

The **symplectic Lie algebra** $\mathfrak{sp}_{2n}\mathbb{C}$ consists of endomorphisms $A\colon V \to V$ satisfying

$$Q(Av, w) + Q(v, Aw) = 0 \quad \text{for all } v, w \in V.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Matrix Description)</span></p>

Choose a basis $e_1, \ldots, e_{2n}$ for $V$ with $Q(e_i, e_{i+n}) = 1$, $Q(e_{i+n}, e_i) = -1$, and all other pairings zero. Then $Q(x, y) = {}^t\!x \cdot M \cdot y$ where $M = \begin{pmatrix} 0 & I_n \\\ -I_n & 0 \end{pmatrix}$. A $2n \times 2n$ matrix $X = \begin{pmatrix} A & B \\\ C & D \end{pmatrix}$ lies in $\mathfrak{sp}_{2n}\mathbb{C}$ if and only if

$${}^t\!X \cdot M + M \cdot X = 0, \tag{16.1}$$

which is equivalent to: the off-diagonal blocks $B$ and $C$ are **symmetric**, and the diagonal blocks $A$ and $D$ are **negative transposes** of each other ($D = -{}^t\!A$).

</div>

#### Cartan Subalgebra and Roots

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cartan Subalgebra and Roots of $\mathfrak{sp}_{2n}\mathbb{C}$)</span></p>

The Cartan subalgebra $\mathfrak{h}$ is the subalgebra of matrices diagonal in this representation, spanned by $H_i = E_{i,i} - E_{n+i,n+i}$ for $i = 1, \ldots, n$. We write $L_j$ for the dual basis of $\mathfrak{h}^*$: $\langle L_j, H_i \rangle = \delta_{i,j}$.

The eigenvectors for $\mathfrak{h}$ in $\mathfrak{sp}_{2n}\mathbb{C}$ are:

- $X_{i,j} = E_{i,j} - E_{n+j,n+i}$ with eigenvalue $L_i - L_j$ (for $i \neq j$),
- $Y_{i,j} = E_{i,n+j} + E_{j,n+i}$ with eigenvalue $L_i + L_j$ (for $i \neq j$; and $Y_{i,i} = E_{i,n+i}$),
- $Z_{i,j} = E_{n+i,j} + E_{n+j,i}$ with eigenvalue $-L_i - L_j$ (for $i \neq j$; and $Z_{i,i} = E_{n+i,i}$),
- $U_i = E_{i,n+i}$ with eigenvalue $2L_i$,
- $V_i = E_{n+i,i}$ with eigenvalue $-2L_i$.

**The roots of $\mathfrak{sp}_{2n}\mathbb{C}$ are the vectors $\pm L_i \pm L_j \in \mathfrak{h}^*$.**

</div>

#### Killing Form

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Killing Form of $\mathfrak{sp}_{2n}\mathbb{C}$)</span></p>

Using the automorphisms of $\mathfrak{sp}_{2n}\mathbb{C}$ preserving $\mathfrak{h}$ (permutations of the $e_i$ and sign changes $e_i \mapsto e_{n+i}$, $e_{n+i} \mapsto -e_i$), the Killing form on $\mathfrak{h}$ must be a multiple of the standard quadratic form $B(H_i, H_j) = \delta_{i,j}$. Computing directly from the definition via $B(H, H') = \sum_{\alpha \in R} \alpha(H)\alpha(H')$:

$$B(H, H') = (4n + 4)\!\left(\sum a_i b_i\right). \tag{16.3}$$

The dual form on $\mathfrak{h}^*$ is correspondingly $B(L_i, L_j) = \delta_{i,j}$, so that the angles in the root diagram are exactly as drawn.

</div>

#### Distinguished Subalgebras and Lattices

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Distinguished Subalgebras of $\mathfrak{sp}_{2n}\mathbb{C}$)</span></p>

The distinguished elements are:
- $H_{L_i - L_j} = H_i - H_j$ (from $[X_{i,j}, X_{j,i}] = H_i - H_j$),
- $H_{L_i + L_j} = H_i + H_j$ (from $[Y_{i,j}, Z_{i,j}] = H_i + H_j$),
- $H_{2L_i} = H_i$ (from $[U_i, V_i] = H_i$), and $H_{-2L_i} = -H_i$.

The **weight lattice** $\Lambda_W$ consists of integral linear combinations of the $L_i$ (i.e., $\Lambda_W = \mathbb{Z}\lbrace L_1, \ldots, L_n \rbrace$). The **root lattice** $\Lambda_R$ consists of even sums $\sum a_i L_i$ with all $a_i$ of the same parity: $\Lambda_R = \lbrace \sum a_i L_i : a_i \in \mathbb{Z},\, \sum a_i \equiv 0 \pmod{2} \rbrace$. Thus $[\Lambda_W : \Lambda_R] = 2$ for all $n$.

</div>

#### Weyl Group, Ordering, and Weyl Chamber

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weyl Group and Weyl Chamber of $\mathfrak{sp}_{2n}\mathbb{C}$)</span></p>

Reflection in $\Omega_{2L_i}$ (given by $\langle H_i, L \rangle = 0$) reverses the sign of $L_i$ while fixing the other $L_j$. Reflection in $\Omega_{L_i - L_j}$ exchanges $L_i$ and $L_j$. The Weyl group $\mathfrak{W}$ is therefore the **semidirect product** $\mathfrak{S}_n \ltimes (\mathbb{Z}/2\mathbb{Z})^n$ (a wreath product), acting as the full automorphism group of the lines spanned by the $L_i$. Its order is $2^n n!$.

The positive roots (for $c_1 > c_2 > \cdots > c_n > 0$) are

$$R^+ = \lbrace L_i + L_j \rbrace_{i \le j} \cup \lbrace L_i - L_j \rbrace_{i < j}, \tag{16.4}$$

with primitive positive roots $L_i - L_{i+1}$ ($i = 1, \ldots, n-1$) and $2L_n$. The Weyl chamber is

$$\mathscr{W} = \lbrace a_1 L_1 + \cdots + a_n L_n : a_1 \ge a_2 \ge \cdots \ge a_n \ge 0 \rbrace. \tag{16.5}$$

The fundamental weights are $\omega_i = L_1 + \cdots + L_i$ for $i = 1, \ldots, n$.

</div>

### 16.2 Representations of $\mathfrak{sp}_4\mathbb{C}$

We now specialize to $n = 2$. We write $\Gamma_{a,b}$ for $\Gamma_{aL_1 + b(L_1 + L_2)}$, i.e., the irreducible representation with highest weight $(a+b)L_1 + bL_2$.

#### The Standard Representation and Its Exterior Square

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Standard Representation $V = \Gamma_{1,0}$)</span></p>

The standard representation of $\mathfrak{sp}_4\mathbb{C}$ on $V = \mathbb{C}^4$ has weights $L_1, L_2, -L_1, -L_2$. Its weight diagram is a square (or diamond). Note that $V$ is self-dual ($V \cong V^*$), since the symplectic group preserves a bilinear form $Q$ giving an identification $V \cong V^*$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exterior Square $\bigwedge^2 V$)</span></p>

The exterior square $\bigwedge^2 V$ has weights $\pm L_i \pm L_j$ (each once) and $0$ (appearing twice, as $L_1 - L_1$ and $L_2 - L_2$). This representation is **not** irreducible: by Observation 14.16, there is only one way to get from the highest weight $L_1 + L_2$ to the zero weight space by successive applications of primitive negative root spaces, so the zero weight space of the irreducible $\Gamma_{0,1}$ has dimension 1. We have

$$\bigwedge^2 V = W \oplus \mathbb{C},$$

where $W = \Gamma_{0,1}$ is the irreducible five-dimensional representation with highest weight $L_1 + L_2$. The trivial summand $\mathbb{C}$ corresponds to the skew form $Q \in \bigwedge^2 V^* \cong \bigwedge^2 V$.

</div>

#### Symmetric Powers and the Adjoint Representation

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Symmetric Square $\mathrm{Sym}^2 V$ and the Adjoint Representation)</span></p>

The weights of $\mathrm{Sym}^2 V$ are the pairwise sums of the weights of $V$: $\pm 2L_i$, $\pm L_1 \pm L_2$, and $0$ (twice). Its weight diagram looks like the root diagram plus the origin (with multiplicity 3). Since $\mathfrak{sp}_4\mathbb{C} \subset \mathrm{Hom}(V,V) = V \otimes V^* = V \otimes V$, and the defining relation (16.1) says exactly that $\mathfrak{sp}_4\mathbb{C}$ is the subspace $\mathrm{Sym}^2 V \subset V \otimes V$, we find:

$$\mathrm{Sym}^2 V = \Gamma_{2,0} = \text{the adjoint representation of } \mathfrak{sp}_4\mathbb{C}.$$

It is the irreducible 10-dimensional representation with highest weight $2L_1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Higher Symmetric Powers $\mathrm{Sym}^a V$)</span></p>

The representations $\mathrm{Sym}^a V$ are all irreducible ($= \Gamma_{a,0}$), with weight diagrams a sequence of nested diamonds $D_i$ with vertices at $aL_1$, $(a-2)L_1$, etc. The multiplicity along the diamond $D_i$ is $i$.

</div>

#### The Representation $W = \Gamma_{0,1}$ and Its Powers

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Symmetric Powers of $W$)</span></p>

The symmetric powers $\mathrm{Sym}^b W$ have eigenvalue diagrams in the shape of a sequence of squares $S_i$ with vertices at $b(L_1 + L_2)$, $(b-1)(L_1 + L_2)$, etc. The multiplicities grow quadratically but only on every other ring: the multiplicity is $i(i+1)/2$ on the $(2i-1)$st and $(2i)$th squares.

The contraction with the skew form $\varphi \in \mathrm{Sym}^2 W^*$ gives a surjection $\mathrm{Sym}^b W \to \mathrm{Sym}^{b-2} W$ whose kernel is the irreducible representation $\Gamma_{0,b}$ with highest weight $b(L_1 + L_2)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Decomposition of $\mathrm{Sym}^2 W$)</span></p>

The symmetric square of $W$ decomposes via a natural map $\mathrm{Sym}^2(\bigwedge^2 V) \to \bigwedge^4 V = \mathbb{C}$ (the wedge product), yielding

$$\mathrm{Sym}^2 W = \Gamma_{0,2} \oplus \mathbb{C}. \tag{16.6}$$

The kernel $\Gamma_{0,2}$ is the irreducible 14-dimensional representation with highest weight $2(L_1 + L_2)$.

</div>

#### Tensor Products

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tensor Product $V \otimes W$)</span></p>

The tensor product $V \otimes W$ has highest weight $2L_1 + L_2$ and must contain $\Gamma_{1,1}$. A natural surjective map $\varphi\colon V \otimes W \to V$ (coming from $\wedge\colon V \otimes \bigwedge^2 V \to \bigwedge^3 V = V^* = V$, restricted to $V \otimes W \subset V \otimes \bigwedge^2 V$) shows that it also contains a copy of $V$:

$$V \otimes W = \Gamma_{1,1} \oplus V.$$

</div>

#### The Representation $\Gamma_{2,1}$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Analyzing $\mathrm{Sym}^2 V \otimes W$)</span></p>

The tensor product $\mathrm{Sym}^2 V \otimes W$ contains $\Gamma_{2,1}$ (with highest weight $2L_1 + (L_1 + L_2) = 3L_1 + L_2$). A detailed weight-by-weight analysis shows:

$$\mathrm{Sym}^2 V \otimes W = \Gamma_{2,1} \oplus \mathrm{Sym}^2 V \oplus W.$$

The multiplicities of $\Gamma_{2,1}$ are: 1 at $2L_1$, 3 at $L_1 + L_2$, and 3 at 0 (in the notation of one-eighth of the plane). In particular, the multiplicities of the irreducible representations of $\mathfrak{sp}_4\mathbb{C}$ are **not constant on the rings** of their weight diagrams (unlike $\mathfrak{sl}_3\mathbb{C}$).

</div>

---

## Lecture 17: $\mathfrak{sp}_6\mathbb{C}$ and $\mathfrak{sp}_{2n}\mathbb{C}$

In this lecture we complete the classification of representations of the symplectic Lie algebras: we describe in detail the example of $\mathfrak{sp}_6\mathbb{C}$, then sketch the general theory for $\mathfrak{sp}_{2n}\mathbb{C}$, in particular proving the existence part of Theorem 14.18. In §17.3 we describe an analog of Weyl's construction for symplectic groups.

### 17.1 Representations of $\mathfrak{sp}_6\mathbb{C}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Root System and Weyl Chamber of $\mathfrak{sp}_6\mathbb{C}$)</span></p>

The Cartan algebra $\mathfrak{h}$ of $\mathfrak{sp}_6\mathbb{C}$ is three-dimensional, with $L_1, L_2, L_3$ forming an orthonormal basis for the Killing form. The 18 roots are the vectors $\pm L_i \pm L_j$ ($i \neq j$) and $\pm 2L_i$. These can be pictured as the midpoints of the edges of a reference cube (with faces centered at $\pm L_i$) together with the midpoints of the faces of a cube twice as large (at $\pm 2L_i$); equivalently, they are the vertices of an octahedron together with the midpoints of its edges.

The 12 roots of the form $\pm L_i \pm L_j$ ($i \neq j$) are congruent to the 12 roots of $\mathfrak{sl}_4\mathbb{C}$. The Weyl group of $\mathfrak{sp}_6\mathbb{C}$ is generated by the Weyl group of $\mathfrak{sl}_4\mathbb{C}$ (permutations of the $L_i$) plus the three additional reflections perpendicular to the $L_i$ (sign changes), giving $\mathfrak{W} = \mathfrak{S}_3 \ltimes (\mathbb{Z}/2\mathbb{Z})^3$ of order $48$. The Weyl chamber of $\mathfrak{sp}_6\mathbb{C}$ is exactly half of the Weyl chamber of $\mathfrak{sl}_4\mathbb{C}$: the cone over one part of the barycentric subdivision of a face of the reference octahedron, with edges generated by $L_1$, $L_1 + L_2$, and $L_1 + L_2 + L_3$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Indexing Representations of $\mathfrak{sp}_6\mathbb{C}$)</span></p>

The weight lattice consists of integral linear combinations of the $L_i$. The intersection of the closed Weyl chamber with the weight lattice consists of $a_1 L_1 + a_2 L_2 + a_3 L_3$ with $a_1 \ge a_2 \ge a_3 \ge 0$. For every triple $(a, b, c)$ of non-negative integers, there is a unique irreducible representation $\Gamma_{a,b,c}$ with highest weight $aL_1 + b(L_1 + L_2) + c(L_1 + L_2 + L_3) = (a+b+c)L_1 + (b+c)L_2 + cL_3$.

</div>

#### Fundamental Representations

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Standard Representation $V = \Gamma_{1,0,0}$)</span></p>

The standard representation of $\mathfrak{sp}_6\mathbb{C}$ on $V = \mathbb{C}^6$ has weights $\pm L_i$. Its weight diagram is the vertices of the reference octahedron (one-half size).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Second Exterior Power $\bigwedge^2 V$ and $W = \Gamma_{0,1,0}$)</span></p>

$\bigwedge^2 V$ has the 12 weights $\pm L_i \pm L_j$ ($i \neq j$, each once) and the weight $0$ (taken three times). The action of $\mathfrak{sp}_6\mathbb{C}$ preserves a skew form, so $\bigwedge^2 V$ contains a trivial summand $\mathbb{C}$. Its complement $W = \Gamma_{0,1,0}$ is the irreducible 14-dimensional representation with highest weight $L_1 + L_2$.

To verify that $0$ appears with multiplicity 2 (not 3) in $W$: there are only three ways to get from the highest weight $L_1 + L_2$ to the zero weight space by successive applications of primitive negative root spaces $\mathfrak{g}_{L_2 - L_1}$, $\mathfrak{g}_{L_3 - L_2}$, and $\mathfrak{g}_{-2L_3}$, confirming that the multiplicity of $0$ in $\Gamma_{0,1,0}$ is $2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Third Exterior Power $\bigwedge^3 V$ and $U = \Gamma_{0,0,1}$)</span></p>

$\bigwedge^3 V$ has 20 weights: the eight sums $\pm L_1 \pm L_2 \pm L_3$ (each once, corresponding to vertices of the reference cube) and the six weights $\pm L_i$ (each twice). Since the highest weight is $L_1 + L_2 + L_3$, the representation $\bigwedge^3 V$ must contain the irreducible $\Gamma_{0,0,1}$.

A natural contraction map $\bigwedge^3 V \to V$ (contracting with the skew form $Q$) shows that $\bigwedge^3 V$ also contains a copy of $V$:

$$\bigwedge^3 V = U \oplus V,$$

where $U = \Gamma_{0,0,1}$ is the irreducible 14-dimensional representation with highest weight $L_1 + L_2 + L_3$ and the weights $\pm L_i$ occur with multiplicity 1 in $U$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Existence of All Representations)</span></p>

Having found $V$, $W$, and $U$ with highest weights $L_1$, $L_1 + L_2$, and $L_1 + L_2 + L_3$ (the primitive vectors along the edges of the Weyl chamber), the irreducible representation $\Gamma_{a,b,c}$ with highest weight $(a+b+c)L_1 + (b+c)L_2 + cL_3$ will occur inside $\mathrm{Sym}^a V \otimes \mathrm{Sym}^b W \otimes \mathrm{Sym}^c U$, establishing the existence part of Theorem 14.18 for $\mathfrak{sp}_6\mathbb{C}$.

</div>

#### Further Tensor Products

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Representation $\Gamma_{1,1,0}$)</span></p>

The representation $V \otimes W$ contains $\Gamma_{1,1,0}$ with highest weight $2L_1 + L_2$. A natural map $V \otimes W \to U$ (via $V \otimes \bigwedge^2 V \to \bigwedge^3 V \to U$) shows that $\Gamma_{1,1,0}$ must lie in the kernel of this map. The exact multiplicities of $\Gamma_{1,1,0}$ require either explicit calculation or the Weyl character formula.

</div>

#### Geometric Interpretation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Isotropic Grassmannians)</span></p>

The group $\mathrm{PSp}_{2n}\mathbb{C}$ may be characterized as the subgroup of $\mathrm{PGL}_{2n}\mathbb{C}$ carrying isotropic subspaces of $V$ into isotropic subspaces. For each $1 < k \le n$, the subset $G_L \subset G(k, V)$ of $k$-dimensional isotropic subspaces is a subvariety of the Grassmannian, and $\mathrm{PSp}_{2n}\mathbb{C}$ acts on the projective space $\mathbb{P}(V^{(k)})$ preserving $G_L$ (when $V^{(k)} = \ker(\varphi_k)$, the fundamental representation).

</div>

### 17.2 Representations of $\mathfrak{sp}_{2n}\mathbb{C}$ in General

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General Structure)</span></p>

The weight lattice consists simply of integral linear combinations of the $L_i$, and the Weyl chamber is the cone over a simplex with edges defined by $a_1 = a_2 = \cdots = a_i > a_{i+1} = \cdots = a_n = 0$. The primitive lattice element on the $i$th ray is the fundamental weight $\omega_i = L_1 + \cdots + L_i$.

For an arbitrary $n$-tuple $(a_1, \ldots, a_n) \in \mathbb{N}^n$, the irreducible representation $\Gamma_{a_1, \ldots, a_n}$ has highest weight

$$a_1\omega_1 + a_2\omega_2 + \cdots + a_n\omega_n = (a_1 + \cdots + a_n)L_1 + (a_2 + \cdots + a_n)L_2 + \cdots + a_n L_n.$$

</div>

#### The Fundamental Representations $V^{(k)}$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(17.5 — Fundamental Representations via Contraction)</span></p>

For $1 \le k \le n$, there is a natural contraction map

$$\varphi_k\colon \bigwedge^k V \to \bigwedge^{k-2} V$$

defined by

$$\varphi_k(v_1 \wedge \cdots \wedge v_k) = \sum_{i < j} Q(v_i, v_j)(-1)^{i+j-1} v_1 \wedge \cdots \wedge \hat{v}_i \wedge \cdots \wedge \hat{v}_j \wedge \cdots \wedge v_k.$$

The kernel of $\varphi_k$ is exactly the irreducible representation $V^{(k)} = \Gamma_{0, \ldots, 0, 1, 0, \ldots, 0}$ (with 1 in the $k$th position) with highest weight $L_1 + \cdots + L_k$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Theorem 17.5)</span></p>

The restriction of $\bigwedge^k V$ to the subalgebra $\mathfrak{s} \cong \mathfrak{sl}_n\mathbb{C} \subset \mathfrak{sp}_{2n}\mathbb{C}$ (embedded as block-diagonal matrices $\begin{pmatrix} A & 0 \\\ 0 & -{}^t\!A \end{pmatrix}$) gives the decomposition $V = W \oplus W^*$ where $W = \mathbb{C}\lbrace e_1, \ldots, e_n \rbrace$, and hence

$$\bigwedge^k V = \bigoplus_{\substack{a+b=k \\ a+b \le k}} (\bigwedge^a W \otimes \bigwedge^b W^*).$$

The kernel $\ker(\varphi_k) = \bigoplus_{a+b=k} W^{(a,b)}$, and each summand $W^{(a,b)}$ is an irreducible representation of $\mathfrak{sl}_n\mathbb{C}$. One shows that elements $Z_{a,n-b} \in \mathfrak{sp}_{2n}\mathbb{C}$ carry $W^{(a,b)}$ into $W^{(a+1,b-1)}$ and $V_a \in \mathfrak{sp}_{2n}\mathbb{C}$ carry $W^{(a,b)}$ into $W^{(a-1,b+1)}$, so any $\mathfrak{sp}_{2n}\mathbb{C}$-invariant subspace of $\ker(\varphi_k)$ containing one summand must contain them all. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Existence of All Irreducible Representations)</span></p>

Any other representation $\Gamma_{a_1, \ldots, a_n}$ of $\mathfrak{sp}_{2n}\mathbb{C}$ will occur in a tensor product of these; specifically,

$$\Gamma_{a_1, \ldots, a_n} \subset \mathrm{Sym}^{a_1} V \otimes \mathrm{Sym}^{a_2} V^{(2)} \otimes \cdots \otimes \mathrm{Sym}^{a_n} V^{(n)},$$

establishing the existence part of Theorem 14.18 for $\mathfrak{sp}_{2n}\mathbb{C}$.

</div>

### 17.3 Weyl's Construction for Symplectic Groups

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Overview)</span></p>

All representations of $\mathfrak{sp}_{2n}\mathbb{C}$ can be realized concretely by intersecting certain irreducible representations of $\mathfrak{sl}_{2n}\mathbb{C}$ with the intersections of the kernels of all symplectic contractions. This is the symplectic analog of the Weyl construction given in §15.3 for $\mathfrak{sl}_n\mathbb{C}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Symplectic Schur Functors)</span></p>

Let $V^{\langle d \rangle} \subset V^{\otimes d}$ denote the intersection of the kernels of all contractions $\Phi_I\colon V^{\otimes d} \to V^{\otimes(d-2)}$ determined by the symplectic form $Q$ (one for each pair $I = \lbrace p < q \rbrace$ from $\lbrace 1, \ldots, d \rbrace$). Define

$$\mathbb{S}_{\langle \lambda \rangle} V = V^{\langle d \rangle} \cap \mathbb{S}_\lambda V, \tag{17.10}$$

where $\mathbb{S}_\lambda V$ is the usual Schur functor applied to $V = \mathbb{C}^{2n}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(17.11 — Symplectic Weyl Construction)</span></p>

The space $\mathbb{S}_{\langle \lambda \rangle}(V)$ is nonzero if and only if the Young diagram of $\lambda$ has at most $n$ rows, i.e., $\lambda_{n+1} = 0$. In this case, $\mathbb{S}_{\langle \lambda \rangle}(V)$ is the **irreducible representation** of $\mathrm{Sp}_{2n}\mathbb{C}$ with highest weight $\lambda_1 L_1 + \cdots + \lambda_n L_n$.

In other words, for an $n$-tuple $(a_1, \ldots, a_n)$ of non-negative integers,

$$\Gamma_{a_1, \ldots, a_n} = \mathbb{S}_{\langle \lambda \rangle} V,$$

where $\lambda$ is the partition $(a_1 + a_2 + \cdots + a_n, a_2 + \cdots + a_n, \ldots, a_n)$.

</div>

#### Complement in $V^{\otimes d}$ and Orthogonal Splittings

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Complement of $V^{\langle d \rangle}$ in $V^{\otimes d}$)</span></p>

For any pair $I = \lbrace p < q \rbrace$, define $\Psi_I\colon V^{\otimes(d-2)} \to V^{\otimes d}$ by inserting the element $\psi = \sum (e_i \otimes e_{n+i} - e_{n+i} \otimes e_i)$ (corresponding to the skew form $Q$) in the $p$th and $q$th factors. Then $\Phi_I \circ \Psi_I$ is multiplication by $2n = \dim V$, and we have the orthogonal decomposition

$$V^{\otimes d} = V^{\langle d \rangle} \oplus \sum_I \Psi_I(V^{\otimes(d-2)}).$$

This leads to a perpendicular decomposition of the tensor power into pieces $V^{(d)}_{d-2r}$ for $r = 0, 1, \ldots, \lfloor d/2 \rfloor$, each invariant under both $\mathfrak{S}_d$ and $\mathrm{Sp}_{2n}\mathbb{C}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(17.15 — Orthogonal Decomposition of $V^{\otimes d}$)</span></p>

The tensor power $V^{\otimes d}$ decomposes into a direct sum

$$V^{\otimes d} = V^{\langle d \rangle} \oplus V^{(d)}_{d-2} \oplus V^{(d)}_{d-4} \oplus \cdots \oplus V^{(d)}_{d-2p},$$

with $p = \lfloor d/2 \rfloor$, and for all $r \ge 1$, $F^d_r = V^{\langle d \rangle} \oplus V^{(d)}_{d-2} \oplus \cdots \oplus V^{(d)}_{d-2r+2}$. Both sums are orthogonal with respect to the standard Hermitian metric.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Symplectic Schur Functors as $\mathrm{Sp}_{2n}\mathbb{C}$-Representations)</span></p>

All the subspaces in the decomposition are invariant under both $\mathfrak{S}_d$ and $\mathrm{Sp}_{2n}\mathbb{C}$, so we have

$$\mathbb{S}_{\langle \lambda \rangle} V = V^{\langle d \rangle} \cdot c_\lambda = \mathrm{Im}(c_\lambda\colon V^{\langle d \rangle} \to V^{\langle d \rangle}), \tag{17.17}$$

where $c_\lambda$ is the Young symmetrizer.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Invariant Theory Fact and Irreducibility)</span></p>

The algebra $B$ of all endomorphisms of $V^{\langle d \rangle}$ commuting with all permutations in $\mathfrak{S}_d$ is precisely the algebra of all endomorphisms of $V^{\otimes d}$ that commute with $\mathfrak{S}_d$ *and* with all the operators $\vartheta_I = \Psi_I \circ \Phi_I$. By an invariant-theory result (Fact 17.19), $B$ is exactly the algebra of $\mathbb{C}$-linear combinations of operators $A \otimes \cdots \otimes A$ for $A \in \mathrm{Sp}_{2n}\mathbb{C}$. Since $\mathbb{C}[\mathfrak{S}_d]$ acts on $V^{\langle d \rangle}$ and makes $V^{\langle d \rangle} \cdot c_\lambda$ an irreducible $B$-module, it follows that $\mathbb{S}_{\langle \lambda \rangle}(V)$ is an **irreducible representation** of $\mathrm{Sp}_{2n}\mathbb{C}$ (Corollary 17.21).

</div>

#### The Ring $\mathbb{S}^{\langle \cdot \rangle}(V)$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Symplectic Analog of $\mathbb{S}'(V)$)</span></p>

As in Lecture 15 for $\mathrm{GL}_n\mathbb{C}$, one can assemble all the irreducible representations of $\mathrm{Sp}_{2n}\mathbb{C}$ into a single commutative algebra. Start with the ring

$$A'(V, n) = \mathrm{Sym}^{\bullet}(V \oplus \bigwedge^2 V \oplus \bigwedge^3 V \oplus \cdots \oplus \bigwedge^n V),$$

and define $\mathbb{S}'(V, n) = A'(V, n) / I'$, the quotient by the ideal $I'$ generated by the same Plücker relations (15.53) as before. This gives $\mathbb{S}'(V, n) = \bigoplus_\lambda \mathbb{S}_\lambda(V)$ as a sum over partitions $\lambda$ with at most $n$ rows. Then define the ideal $J^{\langle \cdot \rangle}$ in $\mathbb{S}'(V, n)$ generated by all elements $x \wedge \psi$ where $x \in \bigwedge^i V$, $i \le n - 2$, and $\psi$ is the element of $\bigwedge^2 V$ corresponding to the skew form $Q$. The quotient ring is

$$\mathbb{S}^{\langle \cdot \rangle} = \mathbb{S}'(V, n) / J^{\langle \cdot \rangle} = \bigoplus_\lambda \mathbb{S}_{\langle \lambda \rangle}(V),$$

the direct sum of all irreducible representations of $\mathrm{Sp}_{2n}\mathbb{C}$ (with $\lambda$ running over partitions with at most $n$ rows).

</div>

---

## Lecture 18: Orthogonal Lie Algebras

In this and the following two lectures we carry out for the orthogonal Lie algebras $\mathfrak{so}_m\mathbb{C}$ the same analysis as for the special linear and symplectic cases. There is one new phenomenon: three of the low-dimensional orthogonal Lie algebras turn out to be isomorphic to special linear or symplectic ones. The representations not obtainable from tensor powers of the standard representation — the **spin representations** — will be constructed in Lecture 20 using Clifford algebras.

### 18.1 $\mathrm{SO}_m\mathbb{C}$ and $\mathfrak{so}_m\mathbb{C}$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orthogonal Group and Lie Algebra)</span></p>

Let $V$ be an $m$-dimensional complex vector space and $Q\colon V \times V \to \mathbb{C}$ a nondegenerate symmetric bilinear form. The **orthogonal group** $\mathrm{SO}_m\mathbb{C}$ is the group of automorphisms $A$ of $V$ with $\det(A) = 1$ preserving $Q$. The **orthogonal Lie algebra** $\mathfrak{so}_m\mathbb{C}$ consists of endomorphisms $A\colon V \to V$ satisfying

$$Q(Av, w) + Q(v, Aw) = 0 \quad \text{for all } v, w \in V. \tag{18.1}$$

</div>

#### Even Case: $\mathfrak{so}_{2n}\mathbb{C}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Matrix Description of $\mathfrak{so}_{2n}\mathbb{C}$)</span></p>

Choose a basis $e_1, \ldots, e_{2n}$ with $Q(e_i, e_{i+n}) = Q(e_{i+n}, e_i) = 1$ and all other pairings zero. Then $Q(x, y) = {}^t\!x \cdot M \cdot y$ with $M = \begin{pmatrix} 0 & I_n \\\ I_n & 0 \end{pmatrix}$. A matrix $X = \begin{pmatrix} A & B \\\ C & D \end{pmatrix}$ lies in $\mathfrak{so}_{2n}\mathbb{C}$ if and only if $B$ and $C$ are **skew-symmetric** and $D = -{}^t\!A$.

The Cartan subalgebra $\mathfrak{h}$ is spanned by $H_i = E_{i,i} - E_{n+i,n+i}$ for $i = 1, \ldots, n$, with dual basis $L_j$ satisfying $\langle L_j, H_i \rangle = \delta_{i,j}$.

</div>

#### Odd Case: $\mathfrak{so}_{2n+1}\mathbb{C}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Matrix Description of $\mathfrak{so}_{2n+1}\mathbb{C}$)</span></p>

Choose a basis $e_1, \ldots, e_{2n+1}$ with $Q(e_i, e_{i+n}) = Q(e_{i+n}, e_i) = 1$ for $1 \le i \le n$, $Q(e_{2n+1}, e_{2n+1}) = 1$, and all other pairings zero. Writing $X$ in block form $\begin{pmatrix} A & B & E \\\ C & D & F \\\ G & H & J \end{pmatrix}$ (blocks of width $n, n, 1$), $X \in \mathfrak{so}_{2n+1}\mathbb{C}$ iff $B$ and $C$ are skew-symmetric, $D = -{}^t\!A$, $E = -{}^t\!H$, $F = -{}^t\!G$, and $J = 0$.

The Cartan subalgebra is again spanned by $H_i = E_{i,i} - E_{n+i,n+i}$.

</div>

#### Roots

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Roots of the Orthogonal Lie Algebras)</span></p>

For $\mathfrak{so}_{2n}\mathbb{C}$: the eigenvectors $X_{i,j} = E_{i,j} - E_{n+j,n+i}$ have eigenvalue $L_i - L_j$, and $Y_{i,j} = E_{i,n+j} - E_{j,n+i}$, $Z_{i,j} = E_{n+i,j} - E_{n+j,i}$ have eigenvalues $L_i + L_j$ and $-L_i - L_j$ respectively. **The roots of $\mathfrak{so}_{2n}\mathbb{C}$ are the vectors $\pm L_i \pm L_j$ ($i \neq j$).**

For $\mathfrak{so}_{2n+1}\mathbb{C}$: in addition to the same eigenvectors, there are $U_i = E_{i,2n+1} - E_{2n+1,n+i}$ and $V_i = E_{n+i,2n+1} - E_{2n+1,i}$ with eigenvalues $+L_i$ and $-L_i$. **The roots of $\mathfrak{so}_{2n+1}\mathbb{C}$ are $\pm L_i \pm L_j$ ($i \neq j$) together with $\pm L_i$.**

The root diagram of $\mathfrak{so}_{2n}\mathbb{C}$ looks like that of $\mathfrak{sp}_{2n}\mathbb{C}$ with the roots $\pm 2L_i$ removed; the root diagram of $\mathfrak{so}_{2n+1}\mathbb{C}$ looks like that of $\mathfrak{sp}_{2n}\mathbb{C}$ with the roots $\pm 2L_i$ replaced by $\pm L_i$.

</div>

#### Weyl Group

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weyl Groups of the Orthogonal Algebras)</span></p>

For $\mathfrak{so}_{2n+1}\mathbb{C}$: the Weyl group is the same as that of $\mathfrak{sp}_{2n}\mathbb{C}$, namely $\mathfrak{S}_n \ltimes (\mathbb{Z}/2\mathbb{Z})^n$ of order $2^n n!$:

$$1 \to (\mathbb{Z}/2\mathbb{Z})^n \to \mathfrak{W}_{\mathfrak{so}_{2n+1}} \to \mathfrak{S}_n \to 1.$$

For $\mathfrak{so}_{2n}\mathbb{C}$: the Weyl group is the **subgroup** of $\mathfrak{S}_n \ltimes (\mathbb{Z}/2\mathbb{Z})^n$ consisting of transformations whose determinant on the induced permutation of the coordinate axes agrees with the sign of the induced permutation — equivalently, the subgroup acting as $-1$ on an **even** number of axes. It fits into

$$1 \to (\mathbb{Z}/2\mathbb{Z})^{n-1} \to \mathfrak{W}_{\mathfrak{so}_{2n}} \to \mathfrak{S}_n \to 1,$$

and has order $2^{n-1} n!$.

</div>

#### Killing Form, Weyl Chamber, and Lattices

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Killing Form and Weyl Chamber)</span></p>

The Killing form is, up to scalars, the standard quadratic form $B(H_i, H_j) = \delta_{i,j}$, with normalization:

$$B\!\left(\sum a_i H_i, \sum b_i H_i\right) = \begin{cases} (4n-2)\sum a_i b_i & \text{if } m = 2n+1, \\\ (4n-4)\sum a_i b_i & \text{if } m = 2n. \end{cases}$$

For $m = 2n+1$, the Weyl chamber is identical to that of $\mathfrak{sp}_{2n}\mathbb{C}$:

$$\mathscr{W} = \lbrace \sum a_i L_i : a_1 \ge a_2 \ge \cdots \ge a_n \ge 0 \rbrace.$$

For $m = 2n$, the primitive positive roots are $L_1 - L_2, L_2 - L_3, \ldots, L_{n-1} - L_n$, and $L_{n-1} + L_n$ (note: **not** $L_n$ alone), so the Weyl chamber is

$$\mathscr{W} = \lbrace \sum a_i L_i : a_1 \ge a_2 \ge \cdots \ge a_{n-1} \ge |a_n| \rbrace.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weight Lattice)</span></p>

For **both** even and odd orthogonal Lie algebras, the weight lattice $\Lambda_W$ is the lattice generated by the $L_i$ together with the element $(L_1 + \cdots + L_n)/2$. The quotient $\Lambda_W / \Lambda_R$ is:

$$\Lambda_W / \Lambda_R \cong \begin{cases} \mathbb{Z}/2\mathbb{Z} & \text{if } m = 2n+1, \\\ \mathbb{Z}/4\mathbb{Z} & \text{if } m = 2n \text{ and } n \text{ odd}, \\\ \mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/2\mathbb{Z} & \text{if } m = 2n \text{ and } n \text{ even}. \end{cases}$$

</div>

#### Distinguished Subalgebras

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Distinguished Elements $H_\alpha$)</span></p>

The distinguished elements are:
- $H_{L_i - L_j} = H_i - H_j$ (same as for $\mathfrak{sp}_{2n}\mathbb{C}$),
- $H_{L_i + L_j} = H_i + H_j$ (note: for $\mathfrak{so}_{2n+1}\mathbb{C}$, also $H_{-L_i - L_j} = -H_i - H_j$),
- $H_{L_i} = 2H_i$ (for $\mathfrak{so}_{2n+1}\mathbb{C}$ only; so $H_{-L_i} = -2H_i$).

The configuration of distinguished elements for $\mathfrak{so}_{2n+1}\mathbb{C}$ differs from that of $\mathfrak{sp}_{2n}\mathbb{C}$ by the substitution $\pm 2H_i$ for $\pm H_i$; for $\mathfrak{so}_{2n}\mathbb{C}$ it differs by removing the $\pm H_i$ entirely.

</div>

### 18.2 Representations of $\mathfrak{so}_3\mathbb{C}$, $\mathfrak{so}_4\mathbb{C}$, and $\mathfrak{so}_5\mathbb{C}$

#### $\mathfrak{so}_3\mathbb{C} \cong \mathfrak{sl}_2\mathbb{C}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Isomorphism $\mathfrak{so}_3\mathbb{C} \cong \mathfrak{sl}_2\mathbb{C}$)</span></p>

$\mathfrak{so}_2\mathbb{C} \cong \mathbb{C}$ is not semisimple. The root diagram of $\mathfrak{so}_3\mathbb{C}$ (with $n = 1$) is $\lbrace -L_1, 0, L_1 \rbrace$ — identical to that of $\mathfrak{sl}_2\mathbb{C}$. Geometrically, $\mathrm{PSO}_3\mathbb{C}$ is the group of motions of the projective plane $\mathbb{P}^2$ carrying a conic curve $C \subset \mathbb{P}^2$ into itself, which is also $\mathrm{PGL}_2\mathbb{C}$ since $C \cong \mathbb{P}^1$.

An important distinction: the "standard" representation of $\mathfrak{so}_3\mathbb{C}$ on $\mathbb{C}^3$ corresponds to $\mathrm{Sym}^2 V$ of $\mathfrak{sl}_2\mathbb{C}$ (highest weight $2 \cdot \tfrac{1}{2}L_1 = L_1$), not the standard of $\mathfrak{sl}_2\mathbb{C}$. The irreducible representation with highest weight $\tfrac{1}{2}L_1$ is not contained in tensor powers of the standard representation — it is the first example of a **spin representation**.

</div>

#### $\mathfrak{so}_4\mathbb{C} \cong \mathfrak{sl}_2\mathbb{C} \times \mathfrak{sl}_2\mathbb{C}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Isomorphism $\mathfrak{so}_4\mathbb{C} \cong \mathfrak{sl}_2\mathbb{C} \times \mathfrak{sl}_2\mathbb{C}$)</span></p>

The root diagram of $\mathfrak{so}_4\mathbb{C}$ (with $n = 2$) has four roots $\pm L_1 \pm L_2$, which lie on the union of two complementary lines. By Exercise 14.33, $\mathfrak{so}_4\mathbb{C}$ is decomposable: it is the direct sum of the two subalgebras $\mathfrak{s}_{L_1 + L_2}$ and $\mathfrak{s}_{L_1 - L_2}$, each isomorphic to $\mathfrak{sl}_2\mathbb{C}$:

$$\mathfrak{so}_4\mathbb{C} \cong \mathfrak{sl}_2\mathbb{C} \times \mathfrak{sl}_2\mathbb{C}. \tag{18.6}$$

Geometrically, $\mathrm{PSO}_4\mathbb{C}$ is the connected component of the identity in the group of motions of $\mathbb{P}^3$ carrying a quadric hypersurface $\bar{Q}$ into itself. A quadric in $\mathbb{P}^3$ has **two rulings** by families of lines (the quadric is isomorphic to $\mathbb{P}^1 \times \mathbb{P}^1$), and $\mathrm{PSO}_4\mathbb{C}$ acts on this product, giving an inclusion $\mathrm{PSO}_4\mathbb{C} \hookrightarrow \mathrm{PGL}_2\mathbb{C} \times \mathrm{PGL}_2\mathbb{C}$ which is an isomorphism.

The standard representation of $\mathfrak{so}_4\mathbb{C}$ on $V = \mathbb{C}^4$ is the tensor product $U \otimes W$ of the standard representations of the two $\mathfrak{sl}_2\mathbb{C}$ factors. Its second exterior power decomposes as

$$\bigwedge^2 V = W_1 \oplus W_2,$$

where $W_1 = \Gamma_{L_1 + L_2}$ and $W_2 = \Gamma_{L_1 - L_2}$ — confirming that $\mathfrak{so}_4\mathbb{C}$ is the product of two copies of $\mathfrak{sl}_2\mathbb{C}$ with adjoint representations $W_1$ and $W_2$.

As in the case of $\mathfrak{so}_3\mathbb{C}$, the weights of the standard representation generate only a sublattice $\mathbb{Z}\lbrace L_1, L_2 \rbrace$ of index 2 in $\Lambda_W$. The representation $\Gamma_{(L_1 + L_2)/2}$ (pullback of the standard of $\mathfrak{sl}_2\mathbb{C}$ from the first factor) cannot be constructed from tensor powers of $V$.

</div>

#### $\mathfrak{so}_5\mathbb{C} \cong \mathfrak{sp}_4\mathbb{C}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Isomorphism $\mathfrak{so}_5\mathbb{C} \cong \mathfrak{sp}_4\mathbb{C}$)</span></p>

The root diagram of $\mathfrak{so}_5\mathbb{C}$ (with $n = 2$) has 8 roots $\pm L_1 \pm L_2$ and $\pm L_i$ — this is exactly the root diagram of $\mathfrak{sp}_4\mathbb{C}$ rotated through $\pi/4$. Indeed, $\mathfrak{so}_5\mathbb{C} \cong \mathfrak{sp}_4\mathbb{C}$.

The geometric explanation: the locus of isotropic lines for a quadric in $\mathbb{P}^4$ is isomorphic to $\mathbb{P}^3$, so $\mathrm{PSO}_5\mathbb{C}$ acts on $\mathbb{P}^3$ preserving a certain skew-symmetric bilinear form $\tilde{Q}$, giving an inclusion $\mathrm{PSO}_5\mathbb{C} \hookrightarrow \mathrm{PSp}_4\mathbb{C}$ which is an isomorphism.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Representations of $\mathfrak{so}_5\mathbb{C}$)</span></p>

The standard representation $V = \mathbb{C}^5$ of $\mathfrak{so}_5\mathbb{C}$ has weights $\pm L_1, \pm L_2, 0$, which corresponds to $W = \bigwedge^2(\mathbb{C}^4)/\mathbb{C} \cdot Q$ — the five-dimensional fundamental representation of $\mathfrak{sp}_4\mathbb{C}$.

The second exterior power $\bigwedge^2 V$ of the standard representation of $\mathfrak{so}_5\mathbb{C}$ has weights $\pm L_1 \pm L_2$, $\pm L_i$, and $0$ (twice). It is the adjoint representation, irreducible with highest weight $L_1 + L_2$. Under the isomorphism $\mathfrak{so}_5\mathbb{C} \cong \mathfrak{sp}_4\mathbb{C}$, this corresponds to $\mathrm{Sym}^2 V$ of $\mathfrak{sp}_4\mathbb{C}$.

Contraction with the quadratic form $Q$ induces maps $\varphi\colon \mathrm{Sym}^a V \to \mathrm{Sym}^{a-2} V$, and the kernel of $\varphi$ is the irreducible representation with highest weight $a \cdot L_1$.

The weights of the standard representation of $\mathfrak{so}_5\mathbb{C}$ generate only the sublattice $\mathbb{Z}\lbrace L_1, L_2 \rbrace$ of index 2 in $\Lambda_W$. The representation with highest weight $(L_1 + L_2)/2$ — a "symmetric square root" of the adjoint — cannot be obtained from tensor powers of $V$; via the isomorphism $\mathfrak{so}_5\mathbb{C} \cong \mathfrak{sp}_4\mathbb{C}$, it is just the standard representation of $\mathfrak{sp}_4\mathbb{C}$ on $\mathbb{C}^4$. These missing representations are the **spin representations**, to be constructed in Lecture 20 using Clifford algebras.

</div>

---

## Lecture 19: $\mathfrak{so}_6\mathbb{C}$, $\mathfrak{so}_7\mathbb{C}$, and $\mathfrak{so}_m\mathbb{C}$

This lecture completes the analysis of representations of the orthogonal Lie algebras (modulo the spin representations, which are constructed in Lecture 20). We work through the examples $\mathfrak{so}_6\mathbb{C}$ and $\mathfrak{so}_7\mathbb{C}$, then describe the general pattern for even and odd orthogonal algebras, and finally give Weyl's construction for orthogonal groups.

### 19.1 Representations of $\mathfrak{so}_6\mathbb{C}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Isomorphism $\mathfrak{so}_6\mathbb{C} \cong \mathfrak{sl}_4\mathbb{C}$)</span></p>

The root diagram of $\mathfrak{so}_6\mathbb{C}$ (with $n = 3$) has 12 roots $\pm L_i \pm L_j$ ($i \neq j$) — exactly the root diagram of $\mathfrak{sl}_4\mathbb{C}$. The isomorphism can be seen geometrically: $\mathrm{PGL}_4\mathbb{C}$ acts on $\mathbb{P}(\bigwedge^2 V) = \mathbb{P}^5$ preserving the Grassmannian $G(2,4) \subset \mathbb{P}^5$. The subgroup $\mathrm{PSp}_4\mathbb{C} \subset \mathrm{PGL}_4\mathbb{C}$ preserves a hyperplane $\mathbb{P}W \subset \mathbb{P}^5$, and $\mathrm{PSp}_4\mathbb{C}$ was identified with $\mathrm{PSO}_5\mathbb{C}$; removing this constraint, the full group $\mathrm{PGL}_4\mathbb{C}$ preserves the quadric Grassmannian in $\mathbb{P}^5$ and is thereby identified with $\mathrm{PSO}_6\mathbb{C}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weyl Chamber and Fundamental Weights of $\mathfrak{so}_6\mathbb{C}$)</span></p>

The Weyl chamber has edges generated by $L_1$, $L_1 + L_2$, $(L_1 + L_2 + L_3)/2$, and $(L_1 + L_2 - L_3)/2$. Note that the fundamental weight $L_1 + L_2$ lies in the **interior** of a face of the Weyl chamber, not on an edge. The edges corresponding to the two "half-sum" fundamental weights $\alpha = (L_1 + L_2 + L_3)/2$ and $\beta = (L_1 + L_2 - L_3)/2$ correspond to the two **spin representations**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Standard and Exterior Power Representations of $\mathfrak{so}_6\mathbb{C}$)</span></p>

- The **standard representation** $V = \mathbb{C}^6$ has weights $\pm L_i$, with highest weight $L_1$ corresponding to $\bigwedge^2 W$ of $\mathfrak{sl}_4\mathbb{C}$ (where $W = \mathbb{C}^4$).
- $\bigwedge^2 V$ is the **adjoint representation** with weights $\pm L_i \pm L_j$ and highest weight $L_1 + L_2$.
- $\bigwedge^3 V$ has weights $\pm L_1 \pm L_2 \pm L_3$ (each once) and $\pm L_i$ (each with multiplicity 2). It **splits** into two irreducible summands:

$$\bigwedge^3 V = \Gamma_{L_1 + L_2 + L_3} \oplus \Gamma_{L_1 + L_2 - L_3},$$

each a four-dimensional tetrahedron (the two sets of alternate vertices of the reference cube). These correspond to the standard representation $W$ and its dual $W^*$ of $\mathfrak{sl}_4\mathbb{C}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Interpretation of the Splitting of $\bigwedge^3 V$)</span></p>

The direct sum decomposition of $\bigwedge^3 V$ corresponds to a geometric feature of a quadric hypersurface $\bar{Q} \subset \mathbb{P}^5$: the variety of 2-planes lying on $\bar{Q}$ is disconnected, consisting of two components that are each isomorphic to $\mathbb{P}^3$ under the Plücker embedding of $G(3, 6)$ into $\mathbb{P}(\bigwedge^3 \mathbb{C}^6) = \mathbb{P}^{19}$. These two components span complementary 9-dimensional projective subspaces $\mathbb{P}W_1$ and $\mathbb{P}W_2$, giving the decomposition $\bigwedge^3 V = W_1 \oplus W_2$.

Via the isomorphism $\mathfrak{so}_6\mathbb{C} \cong \mathfrak{sl}_4\mathbb{C}$, $\bigwedge^3(\bigwedge^2 W) \cong \mathrm{Sym}^2 W \oplus \mathrm{Sym}^2 W^*$.

</div>

### 19.2 Representations of the Even Orthogonal Algebras

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weyl Chamber and Fundamental Weights of $\mathfrak{so}_{2n}\mathbb{C}$)</span></p>

The weight lattice of $\mathfrak{so}_{2n}\mathbb{C}$ is generated by $L_1, \ldots, L_n$ together with $(L_1 + \cdots + L_n)/2$. The Weyl chamber is

$$\mathscr{W} = \lbrace \sum a_i L_i : a_1 \ge a_2 \ge \cdots \ge a_{n-1} \ge |a_n| \rbrace,$$

a simplicial cone with $n$ faces. The edges are generated by the vectors $L_1$, $L_1 + L_2$, $\ldots$, $L_1 + \cdots + L_{n-2}$, and the two "half-sum" vectors

$$\alpha = (L_1 + \cdots + L_{n-1} + L_n)/2 \quad \text{and} \quad \beta = (L_1 + \cdots + L_{n-1} - L_n)/2.$$

Note that $L_1 + \cdots + L_{n-1}$ is **not** on an edge of the Weyl chamber (it lies in the interior of a face). The fundamental weights are $\omega_i = L_1 + \cdots + L_i$ for $i = 1, \ldots, n-2$, plus $\omega_{n-1} = \beta$ and $\omega_n = \alpha$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(19.2 — Exterior Powers of the Standard Representation)</span></p>

**(i)** The exterior powers $\bigwedge^k V$ of the standard representation $V$ of $\mathfrak{so}_{2n}\mathbb{C}$ are irreducible for $k = 1, 2, \ldots, n-1$.

**(ii)** The exterior power $\bigwedge^n V$ has exactly **two** irreducible factors:

$$\bigwedge^n V = \Gamma_{2\alpha} \oplus \Gamma_{2\beta},$$

where $2\alpha = L_1 + \cdots + L_n$ and $2\beta = L_1 + \cdots + L_{n-1} - L_n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 19.2)</span></p>

The proof follows the same lines as for the symplectic algebras (Theorem 17.5). Restrict $V = W \oplus W^*$ to the subalgebra $\mathfrak{s} \cong \mathfrak{sl}_n\mathbb{C} \subset \mathfrak{so}_{2n}\mathbb{C}$. Then $\bigwedge^k V = \bigoplus_{a+b=k} (\bigwedge^a W \otimes \bigwedge^b W^*)$, and each summand $W^{(a,b)} = \ker(\Psi_{a,b})$ is an irreducible $\mathfrak{sl}_n\mathbb{C}$-module. One shows that elements of $\mathfrak{so}_{2n}\mathbb{C}$ connect adjacent summands, so any $\mathfrak{so}_{2n}\mathbb{C}$-invariant subspace containing one $W^{(a,b)}$ must contain all of them — unless $a + b = k = n$, in which case the vectors $w^{(n,0)}$ and $w^{(n-1,1)}$ are killed by every positive root space, producing two separate irreducible components. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reducibility of $\bigwedge^n V$ via Bilinear Forms)</span></p>

The splitting $\bigwedge^n V = \Gamma_{2\alpha} \oplus \Gamma_{2\beta}$ can also be seen without weight diagrams. The group $\mathrm{SO}_{2n}\mathbb{C}$ preserves a bilinear form $Q$ on $V$ and a wedge product $\varphi\colon \bigwedge^n V \times \bigwedge^n V \to \bigwedge^{2n} V = \mathbb{C}$; it also preserves the isomorphism $\tau\colon \bigwedge^n V \to \bigwedge^n V$ (via $Q$ and $\varphi$), and $\tau^2 = \mathrm{id}$. The $+1$ and $-1$ eigenspaces of $\tau$ give the two irreducible subrepresentations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Spin Representations and Existence)</span></p>

The exterior powers $V, \bigwedge^2 V, \ldots, \bigwedge^{n-2} V$ provide irreducible representations along the first $n - 2$ edges of the Weyl chamber; $\bigwedge^n V$ provides $\Gamma_{2\alpha}$ and $\Gamma_{2\beta}$ along the remaining two edges. But the highest weights $\alpha$ and $\beta$ themselves are **not** integer linear combinations of the $L_i$ — they involve half-integer coefficients. The irreducible representations $\Gamma_\alpha$ and $\Gamma_\beta$ with these highest weights are the **spin representations** of $\mathfrak{so}_{2n}\mathbb{C}$; they will be constructed in Lecture 20.

Assuming their existence, the representation $\Gamma_\gamma$ with any highest weight $\gamma$ in the closed Weyl chamber can be found in the tensor product

$$\mathrm{Sym}^{a_1} V \otimes \cdots \otimes \mathrm{Sym}^{a_{n-2}}(\bigwedge^{n-2} V) \otimes \mathrm{Sym}^{a_{n-1}}(\Gamma_\beta) \otimes \mathrm{Sym}^{a_n}(\Gamma_\alpha).$$

The spin representations $\Gamma_\alpha$ and $\Gamma_\beta$ are dual to one another when $n$ is odd, and self-dual when $n$ is even.

</div>

### 19.3 Representations of $\mathfrak{so}_7\mathbb{C}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Root System and Weyl Chamber of $\mathfrak{so}_7\mathbb{C}$)</span></p>

The root diagram of $\mathfrak{so}_7\mathbb{C}$ (with $n = 3$) has 18 roots $\pm L_i \pm L_j$ ($i \neq j$) and $\pm L_i$. Unlike $\mathfrak{so}_5\mathbb{C} \cong \mathfrak{sp}_4\mathbb{C}$, the algebra $\mathfrak{so}_7\mathbb{C}$ is **not** isomorphic to any previously studied algebra (the long and short roots cannot be interchanged here). The Weyl chamber looks like that of $\mathfrak{sp}_6\mathbb{C}$, but the weight lattice contains the additional vector $\alpha = (L_1 + L_2 + L_3)/2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Representations of $\mathfrak{so}_7\mathbb{C}$)</span></p>

- The **standard representation** $V = \mathbb{C}^7$ has weights $\pm L_i$ and $0$.
- $\bigwedge^2 V$ is the **adjoint representation** with weights $\pm L_i \pm L_j$, $\pm L_i$, and $0$ (multiplicity 3), and highest weight $L_1 + L_2$.
- $\bigwedge^3 V$ has weights $\pm L_1 \pm L_2 \pm L_3$, $\pm L_i \pm L_j$ (multiplicity 2), $\pm L_i$ (multiplicity 2), and $0$ (multiplicity 3). It is **irreducible** (one can verify directly that it has no highest weight vector of weight $L_1$, by checking that no linear combination of the two eigenvectors of weight $L_1$ is killed by both $X_{2,3}$ and $U_3$). $\bigwedge^3 V$ does not contain the trivial representation.

These three representations have highest weight vectors $L_1$, $L_1 + L_2$, and $L_1 + L_2 + L_3$ along the three edges of the Weyl chamber, establishing the existence of all representations in the sublattice $\mathbb{Z}\lbrace L_1, L_2, L_3 \rbrace$. The remaining representations — those with highest weight involving odd multiples of $\alpha = (L_1 + L_2 + L_3)/2$ — require the spin representation $\Gamma_\alpha$, constructed in Lecture 20.

</div>

### 19.4 Representations of the Odd Orthogonal Algebras

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General Structure for $\mathfrak{so}_{2n+1}\mathbb{C}$)</span></p>

The Weyl chamber is the same as for $\mathfrak{sp}_{2n}\mathbb{C}$:

$$\mathscr{W} = \lbrace \sum a_i L_i : a_1 \ge a_2 \ge \cdots \ge a_n \ge 0 \rbrace,$$

with edges generated by $L_1, L_1 + L_2, \ldots, L_1 + \cdots + L_{n-1}$, and $\alpha = (L_1 + \cdots + L_n)/2$. The fundamental weights are $\omega_i = L_1 + \cdots + L_i$ for $i = 1, \ldots, n-1$ and $\omega_n = \alpha$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(19.14 — Exterior Powers for $\mathfrak{so}_{2n+1}\mathbb{C}$)</span></p>

For $k = 1, \ldots, n$, the exterior power $\bigwedge^k V$ of the standard representation $V$ of $\mathfrak{so}_{2n+1}\mathbb{C}$ is the **irreducible** representation with highest weight $L_1 + \cdots + L_k$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Existence of Representations)</span></p>

The exterior powers $V, \bigwedge^2 V, \ldots, \bigwedge^n V$ provide irreducible representations with highest weights along the first $n$ edges of the Weyl chamber, covering the $n - 1$ "integer" fundamental weights. Any dominant weight $\gamma$ that is an even multiple of $\alpha$ can be found in tensor products of these.

The remaining representations — those with highest weight involving odd multiples of $\alpha$ — require the **spin representation** $\Gamma_\alpha$ with highest weight $(L_1 + \cdots + L_n)/2$. Once this is exhibited, all representations of $\mathfrak{so}_{2n+1}\mathbb{C}$ can be found in

$$\mathrm{Sym}^{a_1} V \otimes \cdots \otimes \mathrm{Sym}^{a_{n-1}}(\bigwedge^{n-1} V) \otimes \mathrm{Sym}^{a_n}(\Gamma_\alpha).$$

</div>

### 19.5 Weyl's Construction for Orthogonal Groups

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orthogonal Schur Functors)</span></p>

For $V = \mathbb{C}^m$ with the symmetric form $Q$, and $d = \sum \lambda_i$, define $V^{[d]}$ to be the intersection of the kernels of all contractions $\Phi_I\colon V^{\otimes d} \to V^{\otimes(d-2)}$ determined by $Q$. For any partition $\lambda = (\lambda_1 \ge \cdots \ge \lambda_m \ge 0)$, set

$$\mathbb{S}_{[\lambda]} V = V^{[d]} \cap \mathbb{S}_\lambda V. \tag{19.18}$$

This is a representation of the orthogonal group $\mathrm{O}_m\mathbb{C}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(19.22 — Orthogonal Weyl Construction)</span></p>

**(i)** If $m = 2n + 1$ and $\lambda = (\lambda_1 \ge \cdots \ge \lambda_n \ge 0)$, then $\mathbb{S}_{[\lambda]} V$ is the irreducible representation of $\mathfrak{so}_m\mathbb{C}$ with highest weight $\lambda_1 L_1 + \cdots + \lambda_n L_n$.

**(ii)** If $m = 2n$ and $\lambda = (\lambda_1 \ge \cdots \ge \lambda_{n-1} \ge 0)$ (i.e., $\lambda_n = 0$), then $\mathbb{S}_{[\lambda]} V$ is the irreducible representation of $\mathfrak{so}_m\mathbb{C}$ with highest weight $\lambda_1 L_1 + \cdots + \lambda_n L_n$.

**(iii)** If $m = 2n$ and $\lambda = (\lambda_1 \ge \cdots \ge \lambda_{n-1} \ge \lambda_n > 0)$, then $\mathbb{S}_{[\lambda]} V$ is the **sum of two** irreducible representations of $\mathfrak{so}_m\mathbb{C}$ with highest weights $\lambda_1 L_1 + \cdots + \lambda_n L_n$ and $\lambda_1 L_1 + \cdots + \lambda_{n-1} L_{n-1} - \lambda_n L_n$.

$\mathbb{S}_{[\lambda]} V$ is nonzero if and only if the sum of the lengths of the first two columns of $\lambda$ is at most $m$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Ring $\mathbb{S}^{[\cdot]}(V)$)</span></p>

As in the symplectic case, all irreducible representations of $\mathrm{SO}_m\mathbb{C}$ (obtainable from the standard representation) can be assembled into a single commutative algebra $\mathbb{S}^{[\cdot]}(V) = \mathbb{S}'(V, n) / J^{[\cdot]}$, where $J^{[\cdot]}$ is the ideal generated by appropriate contraction relations involving the symmetric form $Q$. For $m$ odd, this ring contains each irreducible representation of $\mathrm{SO}_{2n+1}\mathbb{C}$ exactly once. For $m = 2n$ even, one must additionally add relations of the form $x - \tau(x)$ for $x \in \bigwedge^n V$ (where $\tau$ is the involution from Theorem 19.2) to separate the two summands in part (iii).

</div>

---

## Lecture 20: Spin Representations of $\mathfrak{so}_m\mathbb{C}$

In this lecture we complete the picture of representations of the orthogonal Lie algebras by constructing the spin representations $S^{\pm}$ of $\mathfrak{so}_m\mathbb{C}$ using Clifford algebras. This also yields a description of the spin groups $\mathrm{Spin}_m\mathbb{C}$. §20.3 briefly describes the phenomenon of **triality** for $\mathrm{Spin}_8\mathbb{C}$.

### 20.1 Clifford Algebras and Spin Representations of $\mathfrak{so}_m\mathbb{C}$

#### Motivation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Spin Representations Are Missing)</span></p>

For $\mathrm{SL}_n\mathbb{C}$ and $\mathrm{Sp}_{2n}\mathbb{C}$, all irreducible representations appear in tensor powers of the standard representation. For $\mathrm{SO}_m\mathbb{C}$, only half do — those whose highest weight lies in the sublattice $\mathbb{Z}\lbrace L_1, \ldots, L_n \rbrace$. The missing representations have highest weights involving half-integer coefficients $(L_1 + \cdots + L_n)/2$.

The topological explanation: $\mathrm{SL}_n\mathbb{C}$ and $\mathrm{Sp}_{2n}\mathbb{C}$ are simply connected, while $\mathrm{SO}_m\mathbb{C}$ has fundamental group $\mathbb{Z}/2\mathbb{Z}$ (for $m > 2$). The missing representations are those of the simply connected double cover $\mathrm{Spin}_m\mathbb{C}$ that do not factor through $\mathrm{SO}_m\mathbb{C}$.

</div>

#### The Clifford Algebra

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Clifford Algebra)</span></p>

Given a symmetric bilinear form $Q$ on a vector space $V$, the **Clifford algebra** $C = C(Q) = \mathrm{Cliff}(V, Q)$ is the associative algebra with unit, generated by $V$ and subject to the relation

$$v \cdot w + w \cdot v = 2Q(v, w) \tag{20.1}$$

for all $v, w \in V$. In particular, $v \cdot v = Q(v, v)$ for all $v \in V$. Equivalently, $C(Q) = T^*(V) / I(Q)$, where $T^*(V) = \bigoplus_{n \ge 0} V^{\otimes n}$ is the tensor algebra and $I(Q)$ is the two-sided ideal generated by all $v \otimes v - Q(v, v) \cdot 1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Universal Property)</span></p>

$C(Q)$ is universal: if $E$ is any associative algebra with unit and $j\colon V \to E$ is a linear map with $j(v)^2 = Q(v,v) \cdot 1$ (equivalently, $j(v) \cdot j(w) + j(w) \cdot j(v) = 2Q(v, w) \cdot 1$), then there is a unique algebra homomorphism $C(Q) \to E$ extending $j$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(20.3 — Basis of the Clifford Algebra)</span></p>

If $e_1, \ldots, e_m$ form a basis for $V$, then the products $e_I = e_{i_1} \cdot e_{i_2} \cdots e_{i_k}$ for $I = \lbrace i_1 < i_2 < \cdots < i_k \rbrace$, together with $e_\emptyset = 1$, form a basis for $C(Q)$.

In particular, $\dim C(Q) = 2^m$ where $m = \dim V$, and the canonical map $V \to C$ is an embedding. When $Q \equiv 0$, the Clifford algebra is just the exterior algebra $\bigwedge^* V$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($\mathbb{Z}/2\mathbb{Z}$-Grading)</span></p>

Since $I(Q)$ is generated by even-degree elements, $C$ inherits a $\mathbb{Z}/2\mathbb{Z}$-grading:

$$C = C^{\mathrm{even}} \oplus C^{\mathrm{odd}},$$

with $C^{\mathrm{even}}$ a subalgebra of dimension $2^{m-1}$.

</div>

#### Embedding $\mathfrak{so}(Q)$ in the Clifford Algebra

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Isomorphism $\bigwedge^2 V \cong \mathfrak{so}(Q)$)</span></p>

The orthogonal Lie algebra $\mathfrak{so}(Q) = \lbrace X \in \mathrm{End}(V) : Q(Xv, w) + Q(v, Xw) = 0 \rbrace$ is isomorphic to $\bigwedge^2 V$ via

$$a \wedge b \mapsto \varphi_{a \wedge b}, \qquad \varphi_{a \wedge b}(v) = 2(Q(b, v)a - Q(a, v)b). \tag{20.4}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(20.7 — Embedding of $\mathfrak{so}(Q)$ in $C(Q)$)</span></p>

The map $\psi\colon \bigwedge^2 V \to C(Q)^{\mathrm{even}}$ defined by

$$\psi(a \wedge b) = \tfrac{1}{2}(a \cdot b - b \cdot a) = a \cdot b - Q(a, b) \tag{20.6}$$

is a **Lie algebra embedding** of $\mathfrak{so}(Q)$ into the even part of the Clifford algebra (with the commutator bracket $[x, y] = x \cdot y - y \cdot x$).

</div>

#### Identifying the Clifford Algebra with Matrix Algebras

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(20.9 — Even Case: $m = 2n$)</span></p>

Write $V = W \oplus W'$, where $W$ and $W'$ are $n$-dimensional isotropic subspaces for $Q$. The decomposition determines an isomorphism of algebras

$$C(Q) \cong \mathrm{End}(\bigwedge^* W),$$

where $\bigwedge^* W = \bigwedge^0 W \oplus \bigwedge^1 W \oplus \cdots \oplus \bigwedge^n W$.

**Proof sketch.** Define maps $l\colon W \to E = \mathrm{End}(\bigwedge^* W)$ and $l'\colon W' \to E$ by: $l(w)(\xi) = w \wedge \xi$ (left multiplication) and $l'(w') = D_\vartheta$ (contraction), where $\vartheta(w) = 2Q(w, w')$. One verifies that $l(w)^2 = 0$, $l'(w')^2 = 0$, and $l(w) \circ l'(w') + l'(w') \circ l(w) = 2Q(w, w') \cdot I$, so the universal property gives a homomorphism $C(Q) \to E$. Both sides have dimension $2^{2n}$, so it is an isomorphism. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(20.16 — Odd Case: $m = 2n + 1$)</span></p>

Write $V = W \oplus W' \oplus U$, where $W$ and $W'$ are $n$-dimensional isotropic subspaces and $U = \mathbb{C} \cdot u_0$ with $Q(u_0, u_0) = 1$. Then

$$C(Q) \cong \mathrm{End}(\bigwedge^* W) \oplus \mathrm{End}(\bigwedge^* W').$$

The even subalgebra maps isomorphically onto either factor: $C(Q)^{\mathrm{even}} \cong \mathrm{End}(\bigwedge^* W)$.

</div>

#### The Spin Representations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Even and Odd Decomposition of $\bigwedge^* W$)</span></p>

The exterior algebra decomposes as $\bigwedge^* W = \bigwedge^{\mathrm{even}} W \oplus \bigwedge^{\mathrm{odd}} W$, and $C(Q)^{\mathrm{even}}$ respects this splitting. In the even case ($m = 2n$):

$$C(Q)^{\mathrm{even}} \cong \mathrm{End}(\bigwedge^{\mathrm{even}} W) \oplus \mathrm{End}(\bigwedge^{\mathrm{odd}} W), \tag{20.14}$$

so $\mathfrak{so}(Q) \subset C(Q)^{\mathrm{even}}$ has two representations: $S^+ = \bigwedge^{\mathrm{even}} W$ and $S^- = \bigwedge^{\mathrm{odd}} W$.

In the odd case ($m = 2n + 1$): $C(Q)^{\mathrm{even}} \cong \mathrm{End}(\bigwedge^* W)$, giving a single representation $S = \bigwedge^* W$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(20.15 — Half-Spin Representations for $\mathfrak{so}_{2n}\mathbb{C}$)</span></p>

The representations $S^{\pm}$ are the irreducible representations of $\mathfrak{so}_{2n}\mathbb{C}$ with highest weights

$$S^+ = \Gamma_\alpha, \quad S^- = \Gamma_\beta,$$

where $\alpha = \tfrac{1}{2}(L_1 + \cdots + L_n)$ and $\beta = \tfrac{1}{2}(L_1 + \cdots + L_{n-1} - L_n)$. More precisely:

- If $n$ is **even**: $S^+ = \Gamma_\alpha$ and $S^- = \Gamma_\beta$.
- If $n$ is **odd**: $S^+ = \Gamma_\beta$ and $S^- = \Gamma_\alpha$.

Each $e_I \in \bigwedge^* W$ is a weight vector with weight $\tfrac{1}{2}(\sum_{i \in I} L_i - \sum_{j \notin I} L_j)$. All such weights with $|I|$ of given parity are congruent under the Weyl group, so each of $S^+$ and $S^-$ is irreducible. The representations $S^+$ and $S^-$ are called the **half-spin representations**, and $S = S^+ \oplus S^-$ is the **spin representation**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(20.20 — Spin Representation for $\mathfrak{so}_{2n+1}\mathbb{C}$)</span></p>

The representation $S = \bigwedge^* W$ is the irreducible representation of $\mathfrak{so}_{2n+1}\mathbb{C}$ with highest weight

$$\alpha = \tfrac{1}{2}(L_1 + \cdots + L_n).$$

This completes the proof of the existence theorem for representations of $\mathfrak{so}_m\mathbb{C}$, and hence for all classical complex semisimple Lie algebras.

</div>

### 20.2 The Spin Groups $\mathrm{Spin}_m\mathbb{C}$ and $\mathrm{Spin}_m\mathbb{R}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Involutions on $C(Q)$)</span></p>

The Clifford algebra $C$ has two important anti-automorphisms:

- The **conjugation** $x \mapsto x^*$, defined by $(v_1 \cdots v_r)^* = (-1)^r v_r \cdots v_1$ (reverse order and negate).
- The **main antiautomorphism** (or **reversal**) $\tau$, defined by $\tau(v_1 \cdots v_r) = v_r \cdots v_1$ (reverse order only).
- The **main involution** $\alpha$, which is $+1$ on $C^{\mathrm{even}}$ and $-1$ on $C^{\mathrm{odd}}$.

These satisfy $x^* = \tau(\alpha(x)) = \alpha(\tau(x))$ and $(x \cdot y)^* = y^* \cdot x^*$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Spin Group)</span></p>

The **spin group** is defined as

$$\mathrm{Spin}(Q) = \lbrace x \in C(Q)^{\mathrm{even}} : x \cdot x^* = 1 \text{ and } x \cdot V \cdot x^* \subset V \rbrace. \tag{20.27}$$

For $x \in \mathrm{Spin}(Q)$, the map $\rho(x)(v) = x \cdot v \cdot x^*$ defines an endomorphism of $V$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(20.28 — Double Covering)</span></p>

For $x \in \mathrm{Spin}(Q)$, $\rho(x)$ lies in $\mathrm{SO}(Q)$. The mapping

$$\rho\colon \mathrm{Spin}(Q) \to \mathrm{SO}(Q)$$

is a **connected two-sheeted covering** homomorphism with $\ker(\rho) = \lbrace 1, -1 \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch)</span></p>

**Surjectivity:** The orthogonal group $\mathrm{O}(Q)$ is generated by reflections $R_w$ in hyperplanes perpendicular to unit vectors $w$ (with $Q(w, w) \neq 0$). For such $w$, $\rho(w)(v) = \alpha(w) \cdot v \cdot w^* = -w \cdot v \cdot w = R_w(v)$. Since $w \cdot w^* = -Q(w,w) = -1$, we have $w \in \mathrm{Pin}(Q)$, and $w_1 \cdots w_r \in \mathrm{Spin}(Q)$ when $r$ is even. Every element of $\mathrm{SO}(Q)$ is a product of an even number of reflections, giving surjectivity.

**Kernel:** If $\rho(x) = \mathrm{id}$, writing $x = x_0 + x_1$ with $x_0 \in C^{\mathrm{even}}$ and $x_1 \in C^{\mathrm{odd}}$, one shows $x_0$ is a scalar, $x_1 = 0$, and $x_0^2 = 1$, so $x = \pm 1$.

**Explicit description:**

$$\mathrm{Spin}(Q) = \lbrace \pm w_1 \cdots w_r : w_i \in V,\, Q(w_i, w_i) = -1 \rbrace. \tag{20.31}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Low-Dimensional Isomorphisms via Spin Groups)</span></p>

The spin and half-spin representations recover all the low-dimensional isomorphisms:

| Spin group | Isomorphism |
| --- | --- |
| $\mathrm{Spin}_2\mathbb{C}$ | $\cong \mathrm{GL}_1\mathbb{C} = \mathbb{C}^*$ |
| $\mathrm{Spin}_3\mathbb{C}$ | $\cong \mathrm{SL}_2\mathbb{C}$ |
| $\mathrm{Spin}_4\mathbb{C}$ | $\cong \mathrm{SL}_2\mathbb{C} \times \mathrm{SL}_2\mathbb{C}$ |
| $\mathrm{Spin}_5\mathbb{C}$ | $\cong \mathrm{Sp}_4\mathbb{C}$ |
| $\mathrm{Spin}_6\mathbb{C}$ | $\cong \mathrm{SL}_4\mathbb{C}$ |

The spin representation $S$ maps $\mathrm{Spin}(Q) \to \mathrm{GL}(S)$; the half-spin representations for $m = 2n$ map $\mathrm{Spin}_{2n}\mathbb{C} \to \mathrm{SL}(S^+)$ and $\mathrm{SL}(S^-)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Real Spin Groups)</span></p>

For the real case, one uses the real Clifford algebra $\mathrm{Cliff}(\mathbb{R}^m, Q)$ with $Q = -Q_m$ (minus the standard positive-definite form), giving products $v_i \cdot v_j = -v_j \cdot v_i$ for $i \neq j$ and $v_i \cdot v_i = -1$. The definitions of $\mathrm{Pin}_m(\mathbb{R})$ and $\mathrm{Spin}_m(\mathbb{R})$ carry over identically, yielding double coverings $\mathrm{Spin}_m(\mathbb{R}) \to \mathrm{SO}_m(\mathbb{R})$.

More generally, for a quadratic form on $\mathbb{R}^m$ with $p$ positive and $q$ negative eigenvalues, one gets $\mathrm{Spin}^+(p,q) \to \mathrm{SO}^+(p,q)$.

</div>

### 20.3 $\mathrm{Spin}_8\mathbb{C}$ and Triality

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Triality for $\mathfrak{so}_8\mathbb{C}$)</span></p>

When $m$ is even, there is always an outer automorphism of $\mathrm{Spin}_m(\mathbb{C})$ that interchanges the two half-spin representations $S^+$ and $S^-$ while preserving the standard representation $V$.

The case $m = 8$ is special: here $V$, $S^+$, and $S^-$ are all **eight-dimensional**. The Dynkin diagram of $\mathfrak{so}_8\mathbb{C}$ has four simple roots $\alpha_1 = L_1 - L_2$, $\alpha_2 = L_2 - L_3$, $\alpha_3 = L_3 - L_4$, $\alpha_4 = L_3 + L_4$, arranged so that $\alpha_1$, $\alpha_3$, $\alpha_4$ are mutually perpendicular and each makes a $120°$ angle with $\alpha_2$:

The group of outer automorphisms of $\mathfrak{so}_8\mathbb{C}$ modulo inner automorphisms is the symmetric group $\mathfrak{S}_3$ on three elements, which permutes the three representations $V$, $S^+$, $S^-$ arbitrarily. This phenomenon is called **triality**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Triality)</span></p>

Triality also has a purely geometric description. An even-dimensional quadric $Q \subset \mathbb{P}^7$ admits two families of maximal (3-dimensional) isotropic subspaces, denoted $Q^+$ and $Q^-$ (each a six-dimensional quadric itself). There are natural correspondences:

$$\text{Point in } Q \longleftrightarrow \text{3-plane in } Q^+ \longleftrightarrow \text{3-plane in } Q^-,$$

forming a hexagon of maps between the three varieties $Q$, $Q^+$, $Q^-$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Algebraic Triality and the Cubic Form)</span></p>

Algebraically, the three representations $V$, $S^+$, $S^-$ of $\mathrm{Spin}_8\mathbb{C}$ fit together to form a direct sum $A = V \oplus S^+ \oplus S^-$, on which there is a symmetric **trilinear form** $\Phi$ defined by

$$(v, s, t) \mapsto \langle v \cdot s, t \rangle_{S^-},$$

where $\langle\,,\,\rangle$ denotes the natural pairing and $v \cdot s$ denotes the Clifford action. One can construct an automorphism $J$ of $A$ of **order three** that sends $V \to S^+$, $S^+ \to S^-$, $S^- \to V$, preserving $\Phi$ and compatible with the group action. The induced automorphism $j'\colon \mathfrak{so}_8\mathbb{C} \to \mathfrak{so}_8\mathbb{C}$ satisfies the **local triality equation**:

$$\Phi(Xv, s, t) + \Phi(v, Ys, t) + \Phi(v, s, Zt) = 0$$

for $X \in \mathfrak{so}_8\mathbb{C}$, $Y = j'(X)$, $Z = j'(Y)$.

</div>

---

# Part IV: Lie Theory

The purpose of this final part is threefold: (1) translate the representation theory of Lie algebras back to Lie groups; (2) classify all semisimple Lie algebras via Dynkin diagrams, including the five exceptional algebras; (3) give formulas for the multiplicities of weights in irreducible representations (Weyl character formula and friends).

---

## Lecture 21: The Classification of Complex Simple Lie Algebras

In this lecture we introduce the Dynkin diagram of a semisimple Lie algebra — a simple combinatorial object that determines the Lie algebra up to isomorphism — and use it to classify all simple Lie algebras over $\mathbb{C}$.

### 21.1 Dynkin Diagrams Associated to Semisimple Lie Algebras

#### Abstract Root Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Abstract Root System)</span></p>

A finite set $R$ of elements in a Euclidean space $\mathbb{E}$ (with inner product $(\,,\,)$) is an **(abstract) root system** if it satisfies:

1. $R$ is a finite set spanning $\mathbb{E}$.
2. $\alpha \in R \Rightarrow -\alpha \in R$, but $k \cdot \alpha \notin R$ for any real number $k$ other than $\pm 1$.
3. For $\alpha \in R$, the reflection $W_\alpha$ in the hyperplane $\alpha^\perp$ maps $R$ to itself.
4. For $\alpha, \beta \in R$, the number $n_{\beta\alpha} = 2\frac{(\beta, \alpha)}{(\alpha, \alpha)}$ is an integer.

The dimension $n = \dim_\mathbb{R} \mathbb{E}$ is called the **rank** of the root system (or of the Lie algebra).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Constraints on Root Pairs)</span></p>

If $\vartheta$ is the angle between two roots $\alpha$ and $\beta$, then $n_{\beta\alpha} = 2\cos(\vartheta) \cdot \|\beta\| / \|\alpha\|$ and $n_{\alpha\beta} n_{\beta\alpha} = 4\cos^2(\vartheta)$, which must be an integer between 0 and 4. The case $4\cos^2(\vartheta) = 4$ gives $\beta = \pm\alpha$. Excluding this, the only possibilities for the angle between two roots and their length ratio are given by the following table:

| $\cos(\vartheta)$ | $\vartheta$ | $n_{\beta\alpha}$ | $n_{\alpha\beta}$ | $\|\beta\|/\|\alpha\|$ |
| --- | --- | --- | --- | --- |
| $\sqrt{3}/2$ | $\pi/6$ | $3$ | $1$ | $\sqrt{3}$ |
| $\sqrt{2}/2$ | $\pi/4$ | $2$ | $1$ | $\sqrt{2}$ |
| $1/2$ | $\pi/3$ | $1$ | $1$ | $1$ |
| $0$ | $\pi/2$ | $0$ | $0$ | $*$ |
| $-1/2$ | $2\pi/3$ | $-1$ | $-1$ | $1$ |
| $-\sqrt{2}/2$ | $3\pi/4$ | $-2$ | $-1$ | $\sqrt{2}$ |
| $-\sqrt{3}/2$ | $5\pi/6$ | $-3$ | $-1$ | $\sqrt{3}$ |

</div>

#### Root Systems by Rank

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Root Systems of Small Rank)</span></p>

**Rank 1.** The only possibility is $(\mathrm{A}_1)$: the root system of $\mathfrak{sl}_2\mathbb{C}$.

**Rank 2.** There are exactly four irreducible root systems:
- $(\mathrm{A}_2)$: the root system of $\mathfrak{sl}_3\mathbb{C}$ (angle $2\pi/3$, equal lengths),
- $(\mathrm{B}_2)$: the root system of $\mathfrak{so}_5\mathbb{C} \cong \mathfrak{sp}_4\mathbb{C}$ (angle $3\pi/4$, ratio $\sqrt{2}$),
- $(\mathrm{G}_2)$: a new root system with 12 roots (angle $5\pi/6$, ratio $\sqrt{3}$),
- $(\mathrm{A}_1) \times (\mathrm{A}_1)$: the root system of $\mathfrak{so}_4\mathbb{C} \cong \mathfrak{sl}_2\mathbb{C} \times \mathfrak{sl}_2\mathbb{C}$ (reducible).

**Rank 3.** The irreducible root systems are $(\mathrm{A}_3)$ ($\mathfrak{sl}_4\mathbb{C} \cong \mathfrak{so}_6\mathbb{C}$), $(\mathrm{B}_3)$ ($\mathfrak{so}_7\mathbb{C}$), and $(\mathrm{C}_3)$ ($\mathfrak{sp}_6\mathbb{C}$). A root system is **irreducible** if it is not the orthogonal direct sum of two root systems; a semisimple Lie algebra is simple if and only if its root system is irreducible.

</div>

#### Simple Roots and the Dynkin Diagram

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Simple Roots)</span></p>

Choose a linear functional $l\colon \mathbb{E} \to \mathbb{R}$ irrational with respect to $R$ to decompose $R = R^+ \cup R^-$. A positive root is **simple** if it cannot be expressed as a sum of two positive roots. The simple roots for the classical Lie algebras are:

| Type | Lie algebra | Simple roots |
| --- | --- | --- |
| $(\mathrm{A}_n)$ | $\mathfrak{sl}_{n+1}\mathbb{C}$ | $L_1 - L_2,\, L_2 - L_3,\, \ldots,\, L_n - L_{n+1}$ |
| $(\mathrm{B}_n)$ | $\mathfrak{so}_{2n+1}\mathbb{C}$ | $L_1 - L_2,\, L_2 - L_3,\, \ldots,\, L_{n-1} - L_n,\, L_n$ |
| $(\mathrm{C}_n)$ | $\mathfrak{sp}_{2n}\mathbb{C}$ | $L_1 - L_2,\, L_2 - L_3,\, \ldots,\, L_{n-1} - L_n,\, 2L_n$ |
| $(\mathrm{D}_n)$ | $\mathfrak{so}_{2n}\mathbb{C}$ | $L_1 - L_2,\, L_2 - L_3,\, \ldots,\, L_{n-1} - L_n,\, L_{n-1} + L_n$ |

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Properties of Simple Roots)</span></p>

Key consequences of the root system axioms:

**(5)** If $\alpha, \beta$ are roots with $\beta \neq \pm \alpha$, then the $\alpha$-string through $\beta$ has at most four elements and $p - q = n_{\beta\alpha}$.

**(6)** If $\alpha, \beta$ are roots with $(\beta, \alpha) > 0$ then $\alpha - \beta$ is a root; if $(\beta, \alpha) < 0$ then $\alpha + \beta$ is a root. If $(\beta, \alpha) = 0$ then $\alpha - \beta$ and $\alpha + \beta$ are simultaneously roots or nonroots.

**(7)** If $\alpha$ and $\beta$ are distinct simple roots, then $\alpha - \beta$ and $\beta - \alpha$ are not roots.

**(8)** The angle between two distinct simple roots cannot be acute.

**(9)** The simple roots are linearly independent.

**(10)** There are precisely $n$ simple roots. Every positive root can be written uniquely as a non-negative integral linear combination of simple roots. No root is a linear combination of simple roots with coefficients of mixed sign.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynkin Diagram)</span></p>

The **Dynkin diagram** of a root system is a graph with one node $\circ$ for each simple root $\alpha_i$, and two nodes are joined by a number of lines depending on the angle $\vartheta$ between them:

- No line if $\vartheta = \pi/2$,
- One line if $\vartheta = 2\pi/3$,
- Two lines (with arrow from longer to shorter root) if $\vartheta = 3\pi/4$,
- Three lines (with arrow from longer to shorter root) if $\vartheta = 5\pi/6$.

A root system is irreducible if and only if its Dynkin diagram is connected.

</div>

### 21.2 Classifying Dynkin Diagrams

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(21.11 — Classification of Dynkin Diagrams)</span></p>

The Dynkin diagrams of irreducible root systems are precisely:

**Classical series:**
- $(\mathrm{A}_n)$, $n \ge 1$: a chain of $n$ nodes connected by single lines.
- $(\mathrm{B}_n)$, $n \ge 2$: a chain of $n$ nodes, the last edge a double line with arrow.
- $(\mathrm{C}_n)$, $n \ge 3$: a chain of $n$ nodes, the last edge a double line (arrow reversed from $\mathrm{B}_n$).
- $(\mathrm{D}_n)$, $n \ge 4$: a chain of $n - 2$ nodes with a fork at the end (two branches of length 1).

**Exceptional:**
- $(\mathrm{E}_6)$: a chain of 5 nodes with one branch of length 1 from the third node.
- $(\mathrm{E}_7)$: a chain of 6 nodes with one branch of length 1 from the third node.
- $(\mathrm{E}_8)$: a chain of 7 nodes with one branch of length 1 from the third node.
- $(\mathrm{F}_4)$: 4 nodes in a chain, the middle edge a double line.
- $(\mathrm{G}_2)$: 2 nodes connected by a triple line.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Low-Dimensional Coincidences via Dynkin Diagrams)</span></p>

When $n = 1$, all four classical diagrams $(\mathrm{A}_1) = (\mathrm{B}_1) = (\mathrm{C}_1)$ coincide (single node), reflecting $\mathfrak{sp}_2\mathbb{C} \cong \mathfrak{so}_3\mathbb{C} \cong \mathfrak{sl}_2\mathbb{C}$. For $n = 2$: $(\mathrm{D}_2)$ is two disjoint nodes ($\mathfrak{so}_4\mathbb{C} \cong \mathfrak{sl}_2\mathbb{C} \times \mathfrak{sl}_2\mathbb{C}$), and $(\mathrm{C}_2) = (\mathrm{B}_2)$ reflects $\mathfrak{sp}_4\mathbb{C} \cong \mathfrak{so}_5\mathbb{C}$. For $n = 3$: $(\mathrm{D}_3) = (\mathrm{A}_3)$ reflects $\mathfrak{so}_6\mathbb{C} \cong \mathfrak{sl}_4\mathbb{C}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of Theorem 21.11)</span></p>

The proof uses the notion of **admissible diagrams** (Coxeter graphs): $n$ unit vectors $e_1, \ldots, e_n$ in $\mathbb{E}$ with pairwise inner products $0$, $-1/2$, $-\sqrt{2}/2$, or $-\sqrt{3}/2$, with the number of connecting lines being $4(e_i, e_j)^2 \in \lbrace 0, 1, 2, 3 \rbrace$. The key steps:

**(i)** Any subdiagram of an admissible diagram is admissible.

**(ii)** There are at most $n - 1$ lines; the diagram has **no cycles**.

**(iii)** No node has more than **three** lines emanating from it.

**(iv)** Any string of singly-connected nodes can be collapsed to a single node, preserving admissibility.

These constraints, combined with explicit Cauchy–Schwarz inequality arguments, rule out all configurations except the ones listed. The constraint for legs emanating from a triple node is $1/p + 1/q + 1/r > 1$. $\square$

</div>

#### The Cartan Matrix

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cartan Matrix)</span></p>

The **Cartan matrix** of a root system (or Lie algebra) is the $n \times n$ matrix $(n_{i,j})$ where $n_{i,j} = n_{\alpha_i \alpha_j} = 2(\alpha_i, \alpha_j)/(\alpha_j, \alpha_j)$ and $n_{i,i} = 2$. For example, the Cartan matrix of $(\mathrm{A}_n)$ is tridiagonal with $2$'s on the diagonal and $-1$'s on the super- and sub-diagonals.

</div>

### 21.3 Recovering a Lie Algebra from Its Dynkin Diagram

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reconstructing the Root System)</span></p>

Given the Dynkin diagram with simple roots $\alpha_1, \ldots, \alpha_n$, we know $(\alpha_i, \alpha_j)$ for all $i \neq j$. The positive roots are recovered level by level: level-one roots are the simple roots; a root $\beta$ of level $m$ yields a root $\beta + \alpha_j$ of level $m + 1$ exactly when $n_{\beta\alpha_j} > 0$, i.e., when reflecting the known positive root $\beta$ in the hyperplane perpendicular to $\alpha_j$ yields a new positive root. The explicit root systems for the exceptional diagrams $(\mathrm{G}_2)$, $(\mathrm{F}_4)$, $(\mathrm{E}_6)$, $(\mathrm{E}_7)$, $(\mathrm{E}_8)$ can be computed this way; they have 6, 24, 36, 63, and 120 positive roots, respectively.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Recovering the Lie Algebra)</span></p>

Choose $H_i = H_{\alpha_i}$ for the Cartan subalgebra, $X_i \in \mathfrak{g}_{\alpha_i}$ and $Y_i \in \mathfrak{g}_{-\alpha_i}$ with $[X_i, Y_i] = H_i$. For each positive root $\beta = \alpha_{i_1} + \cdots + \alpha_{i_r}$ (with each partial sum a root), set

$$X_\beta = [X_{i_r}, [X_{i_{r-1}}, \ldots, [X_{i_2}, X_{i_1}] \cdots ]], \qquad Y_\beta = [Y_{i_r}, [Y_{i_{r-1}}, \ldots, [Y_{i_2}, Y_{i_1}] \cdots ]].$$

The collection $\lbrace H_i, X_\beta, Y_\beta : 1 \le i \le n,\, \beta \in R^+ \rbrace$ forms a basis for $\mathfrak{g}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(21.22 — Structure Constants from the Dynkin Diagram)</span></p>

The bracket of any two basis elements in (21.20) is a **rational** multiple of another basis element, with the multiple determined entirely from the Dynkin diagram.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span><span class="math-callout__name">(21.25 — Uniqueness)</span></p>

There is a **unique** isomorphism from $\mathfrak{g}$ to $\mathfrak{g}'$ extending the isomorphism of $\mathfrak{h}$ with $\mathfrak{h}'$ and mapping $X_i$ to $X_i'$ for all $i$.

**Proof sketch.** The existence of the map follows from the subalgebra $\tilde{\mathfrak{g}} = \mathfrak{g} \oplus \mathfrak{g}'$ generated by $\tilde{H}_i = H_i \oplus H_i'$, $\tilde{X}_i = X_i \oplus X_i'$, $\tilde{Y}_i = Y_i \oplus Y_i'$. Since $\mathfrak{g}$ is simple, $\tilde{\mathfrak{g}}$ cannot contain a nontrivial ideal mapping to zero under projection to either factor (this would produce a one-dimensional weight space $W_\beta$ that is two-dimensional — a contradiction). Hence both projections are isomorphisms. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conjugacy and Independence of Choices)</span></p>

To complete the classification, one must show that different Dynkin diagrams give non-isomorphic Lie algebras, and that the Dynkin diagram is independent of all choices. The key facts (proved in Appendix D) are:

1. Any two Cartan subalgebras of a semisimple Lie algebra are conjugate (i.e., related by an inner automorphism).
2. Any two decompositions of $R$ into positive and negative roots differ by an element of the Weyl group.

The existence of a simple Lie algebra for each Dynkin diagram follows from Serre's construction: form the free Lie algebra on generators $H_i, X_i, Y_i$ modulo the Serre relations determined by the Cartan matrix. For the classical types $(\mathrm{A}_n)$–$(\mathrm{D}_n)$, existence was established in Lectures 15–20. For the exceptional types, it will be explored in Lecture 22.

</div>

---

## Lecture 22: $\mathfrak{g}_2$ and Other Exceptional Lie Algebras

This lecture is mainly about $\mathfrak{g}_2$, with enough discussion of the other exceptional Lie algebras to give the reader a sense of their complexity. $\mathfrak{g}_2$, being only 14-dimensional, admits an explicit construction from its Dynkin diagram; we verify the Jacobi identity and analyze its representations. In §22.4 we sketch abstract constructions for all five exceptional algebras.

### 22.1 Construction of $\mathfrak{g}_2$ from Its Dynkin Diagram

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Root System of $\mathfrak{g}_2$)</span></p>

The root system associated to the Dynkin diagram $(\mathrm{G}_2)$ has 12 roots in $\mathbb{R}^2$, with simple roots $\alpha_1$ and $\alpha_2$ at an angle of $5\pi/6$. The six positive roots are:

$$\alpha_1,\quad \alpha_2,\quad \alpha_3 = \alpha_1 + \alpha_2,\quad \alpha_4 = 2\alpha_1 + \alpha_2,\quad \alpha_5 = 3\alpha_1 + \alpha_2,\quad \alpha_6 = 3\alpha_1 + 2\alpha_2.$$

The Weyl group is the dihedral group of order 12 (generated by rotation through $\pi/3$ and reflection).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Basis and Multiplication Table)</span></p>

Following the general recipe of §21.3, set $X_1, X_2$ to be generators of the root spaces $\mathfrak{g}_{\alpha_1}, \mathfrak{g}_{\alpha_2}$, choose $Y_1, Y_2$ with $[X_i, Y_i] = H_i$, and define

$$X_3 = [X_1, X_2],\quad X_4 = [X_1, X_3],\quad X_5 = [X_1, X_4],\quad X_6 = [X_2, X_5],$$

with $Y_3, \ldots, Y_6$ defined similarly. The 14 elements $H_1, H_2, X_1, \ldots, X_6, Y_1, \ldots, Y_6$ form a basis for $\mathfrak{g}_2$.

By a suitable rescaling (dividing $X_4, Y_4$ by 2 and $X_5, X_6, Y_5, Y_6$ by 6), one obtains a **symmetric** multiplication table in which $H_i = [X_i, Y_i]$ and

$$[H_i, X_i] = 2X_i, \quad [H_i, Y_i] = -2Y_i \tag{22.3}$$

for $i = 1, \ldots, 6$, with distinguished elements

$$H_3 = H_1 + 3H_2,\; H_4 = 2H_1 + 3H_2,\; H_5 = H_1 + H_2,\; H_6 = H_1 + 2H_2. \tag{22.2}$$

</div>

### 22.2 Verifying That $\mathfrak{g}_2$ Is a Lie Algebra

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Subalgebra $\mathfrak{g}_0 \cong \mathfrak{sl}_3\mathbb{C}$)</span></p>

The root diagram of $(\mathrm{G}_2)$ consists of two concentric hexagons (long and short roots). The six **long** roots $\alpha_5, \alpha_2, \alpha_6$ and their negatives form the root system of $\mathfrak{sl}_3\mathbb{C}$. The subalgebra $\mathfrak{g}_0 = \mathbb{C}\lbrace H_5, H_2, X_5, Y_5, X_2, Y_2, X_6, Y_6 \rbrace$ is isomorphic to $\mathfrak{sl}_3\mathbb{C}$ (its multiplication table matches exactly that of $\mathfrak{sl}_3\mathbb{C}$ in the standard basis).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Decomposition $\mathfrak{g}_2 = \mathfrak{sl}_3\mathbb{C} \oplus W \oplus W^*$)</span></p>

The remaining root spaces correspond to the short roots, which form the weight diagrams of the standard representation $W$ and its dual $W^*$ of $\mathfrak{sl}_3\mathbb{C}$. We have

$$\mathfrak{g}_2 = \mathfrak{g}_0 \oplus W \oplus W^*, \qquad W = \mathbb{C}\lbrace X_4, Y_1, Y_3 \rbrace, \quad W^* = \mathbb{C}\lbrace Y_4, X_1, X_3 \rbrace.$$

The brackets $\mathfrak{g}_0 \times W \to W$ and $\mathfrak{g}_0 \times W^* \to W^*$ are the standard actions of $\mathfrak{sl}_3\mathbb{C}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Brackets Between $W$ and $W^*$)</span></p>

The bracket $[W, W] \subset W^*$ is identified with the map $W \times W \to W^* = \bigwedge^2 W$, $v \times w \mapsto -2 \cdot v \wedge w$. Similarly $[W^*, W^*] \subset W$ via $\varphi \times \psi \mapsto 2 \cdot \varphi \wedge \psi$. The bracket $[W, W^*] \subset \mathfrak{g}_0 = \mathfrak{sl}_3\mathbb{C}$ is given by

$$[v, \varphi] = 18 \cdot v * \varphi, \tag{22.7}$$

where $v * \varphi \in \mathfrak{sl}_3\mathbb{C}$ is the element satisfying $B(v * \varphi, Z) = \varphi(Z \cdot v)$ for all $Z \in \mathfrak{sl}_3\mathbb{C}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Verification of the Jacobi Identity)</span></p>

The Jacobi identity can be verified by checking it on triples from the three summands $\mathfrak{g}_0$, $W$, $W^*$:

- **Three elements from $\mathfrak{g}_0$:** automatic (since $\mathfrak{g}_0 \cong \mathfrak{sl}_3\mathbb{C}$ is a Lie algebra).
- **One from $\mathfrak{g}_0$, two from $W$:** equivalent to the fact that $W$ and $W^*$ are genuine representations of $\mathfrak{sl}_3\mathbb{C}$.
- **Two from $W$, one from $W^*$:** reduces to the identity $ab(v \wedge w)(\varphi) = c \cdot ((\varphi(v))w - (\varphi(w))v)$.
- **One from $W$, two from $W^*$:** follows by symmetry.

The key identity for the last nontrivial case ($v \in W$, $\varphi, \psi \in W^*$) is

$$ab(\varphi \wedge \psi) \wedge v = c \cdot (B(w * \psi, [v, \varphi]) - B(w * \varphi, [v, \psi])),$$

which is verified using the Killing form. $\square$

</div>

### 22.3 Representations of $\mathfrak{g}_2$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weight Lattice and Fundamental Weights)</span></p>

The weight lattice $\Lambda_W$ of $\mathfrak{g}_2$ equals the root lattice $\Lambda_R$ (generated by $\alpha_1$ and $\alpha_2$). The Weyl chamber is the cone between the roots $\alpha_6$ and $\alpha_4$. The fundamental weights are

$$\omega_1 = 2\alpha_1 + \alpha_2, \qquad \omega_2 = 3\alpha_1 + 2\alpha_2.$$

We write $\Gamma_{a,b}$ for the irreducible representation with highest weight $a\omega_1 + b\omega_2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Standard Representation $V = \Gamma_{1,0}$)</span></p>

The standard representation $V$ has highest weight $\omega_1$ and dimension 7. The multiplicity of the weight $0$ is 1 (since there is a unique path from $\omega_1$ to $0$ by subtraction of simple positive roots). $V$ is the **smallest** representation of $\mathfrak{g}_2$, and every irreducible representation appears in some tensor power $V^{\otimes m}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Adjoint Representation $\Gamma_{0,1}$)</span></p>

The adjoint representation has highest weight $\omega_2$ and dimension 14. The multiplicity of $0$ is 2. Since $\Gamma_{0,1}$ appears in $\bigwedge^2 V$ (the adjoint representation of $\mathfrak{g}_2$ is a subrepresentation of $\bigwedge^2 V$):

$$\bigwedge^2 V \cong \Gamma_{0,1} \oplus V.$$

This decomposition also shows that every $\Gamma_{a,b}$ appears in $\mathrm{Sym}^a V \otimes \mathrm{Sym}^b \Gamma_{0,1}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Symmetric and Exterior Powers of $V$)</span></p>

Further decompositions:

$$\mathrm{Sym}^2 V = \Gamma_{2,0} \oplus \mathbb{C}, \qquad \bigwedge^3 V \cong \Gamma_{2,0} \oplus V \oplus \mathbb{C}.$$

In particular, the action of $\mathfrak{g}_2$ on the standard representation $V = \mathbb{C}^7$ preserves a **quadratic form** (since $\mathrm{Sym}^2 V$ contains the trivial representation), so $\mathfrak{g}_2 \subset \mathfrak{sl}(V) = \mathfrak{sl}_7\mathbb{C}$ is in fact contained in $\mathfrak{so}_7\mathbb{C}$.

The action also preserves a **skew-symmetric trilinear (cubic) form** $\omega$ on $V$ (since $\bigwedge^3 V$ contains $\mathbb{C}$). This trilinear form is the key invariant: the space $\bigwedge^3 V$ of alternating 3-forms has dimension 35, while $\mathfrak{gl}(V)$ has dimension 49, and the difference $49 - 35 = 14 = \dim \mathfrak{g}_2$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(22.12 — $\mathfrak{g}_2$ as Endomorphisms Preserving a Cubic Form)</span></p>

The algebra $\mathfrak{g}_2$ is exactly the algebra of endomorphisms of a seven-dimensional vector space $V$ preserving a general skew-symmetric cubic form $\omega$ on $V$.

The map $\varphi\colon \mathfrak{gl}(V) \to \bigwedge^3 V$ sending $A$ to $A(\omega)$ is surjective, with kernel $\mathfrak{g}_2$.

</div>

### 22.4 Algebraic Constructions of the Exceptional Lie Algebras

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Freudenthal's Construction)</span></p>

The construction of $\mathfrak{g}_2$ as $\mathfrak{g}_0 \oplus W \oplus W^*$ generalizes. Given a semisimple Lie algebra $\mathfrak{g}_0$, a representation $W$ with dual $W^*$, and trilinear maps $T\colon \bigwedge^3 W \to \mathbb{C}$ and $T'\colon \bigwedge^3 W^* \to \mathbb{C}$ inducing maps $\wedge\colon \bigwedge^2 W \to W^*$ and $\wedge\colon \bigwedge^2 W^* \to W$, define brackets on $\mathfrak{g} = \mathfrak{g}_0 \oplus W \oplus W^*$ by rules (i)–(vi):

$$[X, Y] = \text{bracket in } \mathfrak{g}_0, \quad [X, v] = X \cdot v, \quad [X, \varphi] = X \cdot \varphi,$$

$$[v, w] = a \cdot (v \wedge w), \quad [\varphi, \psi] = b \cdot (\varphi \wedge \psi), \quad [v, \varphi] = c \cdot (v * \varphi),$$

where $v * \varphi$ is defined by $B(v * \varphi, Z) = \varphi(Z \cdot v)$ for all $Z \in \mathfrak{g}_0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(22.19 — Freudenthal)</span></p>

Given a semisimple Lie algebra $\mathfrak{g}_0$, a representation $W$, and trilinear forms $T$ and $T'$ inducing maps $\bigwedge^2 W \to W^*$ and $\bigwedge^2 W^* \to W$, such that the Jacobi identities (22.17) and (22.18) are satisfied and $abc \neq 0$, the above products make $\mathfrak{g} = \mathfrak{g}_0 \oplus W \oplus W^*$ into a Lie algebra. If all weight spaces of $W$ are one-dimensional and the roots of $\mathfrak{g}_0$ are all distinct, then $\mathfrak{g}$ is semisimple with the same Cartan subalgebra as $\mathfrak{g}_0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Constructing $\mathfrak{e}_8$, $\mathfrak{e}_7$, $\mathfrak{e}_6$)</span></p>

To construct $\mathfrak{e}_8$: take $\mathfrak{g}_0 = \mathfrak{sl}_9\mathbb{C}$ with $V = \mathbb{C}^9$, $W = \bigwedge^3 V$ (so $W^* = \bigwedge^3 V^*$); the trilinear map is the wedge product $\bigwedge^3 V \otimes \bigwedge^3 V \otimes \bigwedge^3 V \to \bigwedge^9 V = \mathbb{C}$. Since $\dim \mathfrak{sl}_9\mathbb{C} = 80$ and $\dim W = \dim W^* = 84$, the sum has dimension $80 + 84 + 84 = 248$, as predicted by the root system of $(\mathrm{E}_8)$ (which has 120 positive roots, hence $\dim \mathfrak{e}_8 = 8 + 2 \times 120 = 248$).

Once $\mathfrak{e}_8$ is constructed, **$\mathfrak{e}_7$ and $\mathfrak{e}_6$ are found as subalgebras**, by removing one or two nodes from the long arm of the Dynkin diagram of $(\mathrm{E}_8)$, yielding $(\mathrm{E}_7)$ (dimension 133) and $(\mathrm{E}_6)$ (dimension 78).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Construction of $\mathfrak{f}_4$)</span></p>

$\mathfrak{f}_4$ can be constructed as the invariant subalgebra of $\mathfrak{e}_6$ under an involution (the evident symmetry of the $(\mathrm{E}_6)$ Dynkin diagram). More concretely, $\mathfrak{f}_4$ is the derivation algebra of a 27-dimensional **Jordan algebra** $\mathbb{J}$ consisting of $3 \times 3$ Hermitian matrices over the octonions $\mathbb{O}$, with product $x \circ y = \tfrac{1}{2}(xy + yx)$. The group $(\mathrm{F}_4)$ is the group of automorphisms of $\mathbb{J}$ preserving the scalar product and scalar triple product.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Octonions and $\mathfrak{g}_2$)</span></p>

The exceptional group $G_2$ can be realized as the group of automorphisms of the complexification of the **Cayley algebra** (octonions) $\mathbb{O}$. The octonions are an eight-dimensional nonassociative but **alternative** algebra (satisfying $(x \circ x) \circ y = x \circ (x \circ y)$ and $(y \circ x) \circ x = y \circ (x \circ x)$) constructed from pairs of quaternions $(a, b)$ with multiplication

$$(a, b) \circ (c, d) = (ac - \bar{d}b,\, da + b\bar{c}).$$

The Lie algebra of derivations of $\mathbb{O}$ is isomorphic to $\mathfrak{su}_3 \cong \mathfrak{g}_2$ (compact form). The octonions can also be reconstructed from $\mathrm{Spin}_8\mathbb{C}$ using the triality automorphism of §20.3: define a product on $V$ by $v \circ w = (v \cdot t_1) \cdot (w \cdot s_1)$, where $v \cdot t_1 \in S^+$ and $w \cdot s_1 \in S^-$ are determined by the Clifford action.

</div>

---

## Lecture 23: Complex Lie Groups; Characters

This lecture makes the transition back from Lie algebras to Lie groups. In §23.1 we classify the groups having a given semisimple Lie algebra and say which representations lift to which groups. In §23.2 we introduce the representation ring and character homomorphism. In §23.3 we sketch the interrelationships among Dynkin diagrams, homogeneous spaces, and irreducible representations. §23.4 gives a brief introduction to the Bruhat decomposition.

### 23.1 Representations of Complex Simple Lie Groups

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(23.1 — Topology of the Classical Groups)</span></p>

For all $n \ge 1$, the Lie groups $\mathrm{SL}_n\mathbb{C}$ and $\mathrm{Sp}_{2n}\mathbb{C}$ are connected and **simply connected**. For $n \ge 1$, $\mathrm{SO}_n\mathbb{C}$ is connected, with $\pi_1(\mathrm{SO}_2\mathbb{C}) = \mathbb{Z}$ and $\pi_1(\mathrm{SO}_n\mathbb{C}) = \mathbb{Z}/2$ for $n \ge 3$.

**Proof sketch.** By induction using the long exact homotopy sequence of a fibration: $\mathrm{SL}_n\mathbb{C}$ acts transitively on $\mathbb{C}^n \setminus \lbrace 0 \rbrace$ with isotropy group $\mathrm{SL}_{n-1}\mathbb{C}$; since $\mathbb{C}^n \setminus \lbrace 0 \rbrace$ has the homotopy type of $S^{2n-1}$ (simply connected for $n \ge 2$), the claim follows by induction. Similarly for $\mathrm{Sp}_{2n}\mathbb{C}$ (acting on pairs $(v,w)$ with $Q(v,w) = 1$) and $\mathrm{SO}_n\mathbb{C}$ (acting on vectors with $Q(v,v) = 1$). $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classical Groups and Their Representations)</span></p>

The simply-connected complex Lie groups corresponding to the classical Lie algebras are:

$$\tilde{G} = \mathrm{SL}_n\mathbb{C}, \quad \mathrm{Sp}_{2n}\mathbb{C}, \quad \text{and} \quad \mathrm{Spin}_m\mathbb{C}.$$

All other connected groups with these Lie algebras are quotients $\tilde{G}/C$ by subgroups $C \subset Z(\tilde{G})$. Representations of $\tilde{G}/C$ are exactly those representations of $\mathfrak{g}$ that are trivial on $C$.

Each classical group $G$ has a compact subgroup ($\mathrm{SU}(n)$, $\mathrm{Sp}(n)$, $\mathrm{SO}(n)$) that is a deformation retract and induces isomorphisms of fundamental groups.

</div>

#### The Cartan Subgroup and the Exponential Map

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cartan Subgroup)</span></p>

The **Cartan subgroup** $H$ is the connected subgroup of $G$ whose Lie algebra is the Cartan subalgebra $\mathfrak{h}$. For $G = \mathrm{SL}_n\mathbb{C}$, $H$ consists of diagonal matrices; for $\mathrm{Sp}_{2n}\mathbb{C}$ and $\mathrm{SO}_{2n}\mathbb{C}$, $H = \lbrace \mathrm{diag}(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1}) \rbrace$. In each case the exponential map $\exp\colon \mathfrak{h} \to H$ is the usual matrix exponential of diagonal matrices.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(23.9 — Exponential in $\mathrm{Spin}_m\mathbb{C}$)</span></p>

For any complex numbers $a_1, \ldots, a_n$,

$$\exp(a_1 H_1 + \cdots + a_n H_n) = w(e^{a_1/2}, \ldots, e^{a_n/2})$$

in $\mathrm{Spin}_m\mathbb{C}$, where $w(z_1, \ldots, z_n) = w_1(z_1) \cdots w_n(z_n)$ and $w_j(z) = z^{-1} + \frac{z - z^{-1}}{2} e_j \cdot e_{n+j}$ are elements of the Clifford algebra.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(23.12 — Criterion for Lifting Representations)</span></p>

The irreducible representation $\Gamma_\lambda$ of $\mathfrak{g}$ is a representation of the group $G = \tilde{G}/C$ if and only if

$$\lambda(X) \in 2\pi i \mathbb{Z} \quad \text{whenever } \exp(X) \in C.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(23.13 — Which Representations Lift to Which Groups)</span></p>

For each subgroup $C$ of the center of $\tilde{G}$, the representation $\Gamma_\lambda$ is a representation of $\tilde{G}/C$ precisely under the following conditions:

**(i)** $\tilde{G} = \mathrm{SL}_{n+1}\mathbb{C}$, $C = \lbrace e^{2\pi i l/m} \cdot I \rbrace$ of order $m \mid (n+1)$: $\sum \lambda_j \equiv 0 \pmod{m}$.

**(ii)** $\tilde{G} = \mathrm{Sp}_{2n}\mathbb{C}$, $C = \lbrace \pm 1 \rbrace$: $\sum \lambda_j$ is even.

**(iii)** $\tilde{G} = \mathrm{Spin}_{2n}\mathbb{C}$ or $\mathrm{Spin}_{2n+1}\mathbb{C}$, $C = \lbrace \pm 1 \rbrace$: all $\lambda_i$ are integers (equivalently, representations of $\mathrm{SO}_m\mathbb{C}$).

**(iv)** $\tilde{G} = \mathrm{Spin}_{2n}\mathbb{C}$, $C = \lbrace 1, \omega \rbrace$ or $\lbrace 1, -\omega \rbrace$: all $\lambda_i$ are integers **and** $\sum \lambda_j$ is even (representations of $\mathrm{PSO}_{2n}\mathbb{C}$).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(23.15)</span></p>

The group $\Lambda_W / \Lambda_R$ is finite, of order equal to the **determinant of the Cartan matrix**. The center $Z(\tilde{G})$ is naturally isomorphic to the dual $\Gamma_W / \Gamma_R$, where $\Gamma_R = \lbrace X \in \mathfrak{h} : \alpha(X) \in \mathbb{Z} \text{ for all } \alpha \in R \rbrace$ and $\Gamma_W$ is defined dually. In particular, $\Lambda_W / \Lambda_R$ is trivial for $(\mathrm{G}_2)$, $(\mathrm{F}_4)$, $(\mathrm{E}_8)$; cyclic of order 2 for $(\mathrm{E}_7)$; and cyclic of order 3 for $(\mathrm{E}_6)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(23.16 — Lattice of Groups)</span></p>

There is a **one-to-one correspondence** between connected Lie groups $G$ with Lie algebra $\mathfrak{g}$ and lattices $\Lambda$ with $\Lambda_R \subset \Lambda \subset \Lambda_W$. The correspondence sends $G$ to the lattice $\Lambda$ dual to the kernel of $\exp\colon \mathfrak{g} \to G$; the largest lattice $\Lambda_W$ corresponds to the simply connected group, and the smallest $\Lambda_R$ to the adjoint group. The representation $\Gamma_\lambda$ lifts to $G$ if and only if $\lambda \in \Lambda$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weyl Group as $N(H)/H$)</span></p>

The Weyl group $\mathfrak{W}$ can be realized inside the Lie group $G$: if $H$ is the Cartan subgroup and $N(H) = \lbrace g \in G : gHg^{-1} = H \rbrace$ its normalizer, then

$$N(H)/H \cong \mathfrak{W}. \tag{23.20}$$

</div>

### 23.2 Representation Rings and Characters

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Representation Ring and Character Homomorphism)</span></p>

The **representation ring** $R(\mathfrak{g})$ is the free abelian group on isomorphism classes $[V]$ of finite-dimensional representations, with $[V] = [V'] + [V'']$ when $V \cong V' \oplus V''$ and $[V] \cdot [W] = [V \otimes W]$. It is a polynomial ring on the classes $[\Gamma_1], \ldots, [\Gamma_n]$ of the fundamental representations.

Let $\Lambda = \Lambda_W$ be the weight lattice and $\mathbb{Z}[\Lambda]$ the integral group ring with basis $e(\lambda)$ for $\lambda \in \Lambda$ and product $e(\alpha) \cdot e(\beta) = e(\alpha + \beta)$. The **character homomorphism** is the ring homomorphism

$$\mathrm{Char}\colon R(\mathfrak{g}) \to \mathbb{Z}[\Lambda], \qquad \mathrm{Char}[V] = \sum_\lambda \dim(V_\lambda)\, e(\lambda). \tag{23.23}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Properties of the Character Map)</span></p>

1. **Char is injective**: a representation is determined by the multiplicities of its weight spaces.
2. **Char is a ring homomorphism**: $\mathrm{Char}[V \otimes W] = \mathrm{Char}[V] \cdot \mathrm{Char}[W]$, since $(V \otimes W)_\lambda = \bigoplus_{\mu + \nu = \lambda} V_\mu \otimes W_\nu$.
3. **The image of Char lies in $\mathbb{Z}[\Lambda]^{\mathfrak{W}}$**: the set of weights (with multiplicities) of any representation is invariant under the Weyl group.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(23.24 — Structure of the Representation Ring)</span></p>

**(a)** The representation ring $R(\mathfrak{g})$ is a polynomial ring on the variables $\Gamma_1, \ldots, \Gamma_n$ (classes of the fundamental representations).

**(b)** The character homomorphism $R(\mathfrak{g}) \to \mathbb{Z}[\Lambda]^{\mathfrak{W}}$ is an **isomorphism**.

In other words, $\mathbb{Z}[\Lambda]^{\mathfrak{W}}$ is a polynomial ring on $\mathrm{Char}(\Gamma_1), \ldots, \mathrm{Char}(\Gamma_n)$.

</div>

### 23.3 Homogeneous Spaces

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dynkin Diagrams and Homogeneous Spaces)</span></p>

There is a beautiful correspondence between the combinatorics of the Dynkin diagram, the geometry of homogeneous spaces, and the representation theory:

- **Nodes of the Dynkin diagram** $\leftrightarrow$ **edges of the Weyl chamber** $\leftrightarrow$ **fundamental representations** $\leftrightarrow$ **maximal parabolic subgroups** $\leftrightarrow$ **Grassmannians** (or their analogs).
- **Subsets of nodes** $\leftrightarrow$ **faces of the Weyl chamber** $\leftrightarrow$ **parabolic subgroups** $\leftrightarrow$ **partial flag manifolds**.
- **All nodes** $\leftrightarrow$ **interior of the Weyl chamber** $\leftrightarrow$ **Borel subgroup** $\leftrightarrow$ **full flag manifold** $G/B$.

For $\mathrm{SL}_n\mathbb{C}$: ordinary Grassmannians $G(k, n)$. For $\mathrm{Sp}_{2n}\mathbb{C}$: Lagrangian Grassmannians. For $\mathrm{SO}_m\mathbb{C}$: orthogonal Grassmannians of isotropic subspaces. For $\mathrm{Spin}_{2n+1}\mathbb{C}$: the spin representation gives the spinor variety $G/P \hookrightarrow \mathbb{P}(S)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span><span class="math-callout__name">(23.52 — Unique Closed Orbit)</span></p>

Let $V = \Gamma_\lambda$ be an irreducible representation of a simple group $G$, and let $p \in \mathbb{P}V$ be the point corresponding to the highest weight eigenspace. The orbit $G \cdot p$ is the **unique closed orbit** of the action of $G$ on $\mathbb{P}V$.

The stabilizer of $p$ is the **parabolic subgroup** $P_\lambda$ corresponding to the subset of simple roots perpendicular to $\lambda$, and $G \cdot p \cong G/P_\lambda$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Borel–Weil Theorem)</span></p>

Starting from the group $G$ alone, one can construct the flag manifold $G/B$ and realize every irreducible representation as the space of holomorphic sections of a line bundle on $G/B$. The weight $\lambda \in \mathfrak{h}^*$ exponentiates to a character $\mathbb{C}_\lambda$ of $H$, which extends to the Borel subgroup $B$ and determines a line bundle

$$L_\lambda = G \times_B \mathbb{C}_\lambda$$

on $G/B$. By Bott's theorem, $H^i(G/B, L_\lambda) = 0$ for $i \neq i(\lambda)$, where $i(\lambda)$ depends on which Weyl chamber $\lambda$ belongs to.

For a **dominant** weight $\lambda$ (i.e., $\lambda$ in the closed positive Weyl chamber), $i(\lambda) = 0$ and:

$$H^0(G/B, L_{-\lambda}) = \Gamma_\lambda,$$

the irreducible representation with highest weight $\lambda$.

</div>

### 23.4 Bruhat Decompositions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(23.59 — Bruhat Decomposition)</span></p>

The group $G$ is a **disjoint union** of $|\mathfrak{W}|$ double cosets $B \cdot n_W \cdot B$, as $W$ varies over the Weyl group:

$$G = \bigsqcup_{W \in \mathfrak{W}} B \cdot n_W \cdot B,$$

where $n_W$ is any representative of $W$ in the normalizer $N(H)$, and $B$ is the Borel subgroup.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bruhat Cells)</span></p>

The quotient $G/B$ is correspondingly a disjoint union of **Bruhat cells** $X_W = B \cdot n_W \cdot B / B$, each isomorphic to an affine space $\mathbb{C}^{l(W)}$, where $l(W)$ is the **length** of $W$ — the minimum number of reflections in simple roots needed to express $W$.

For $G = \mathrm{SL}_m\mathbb{C}$: $N(H)$ consists of monomial matrices, $\mathfrak{W} = \mathfrak{S}_m$ acts by permutations, and the Bruhat decomposition amounts to the fact that any invertible matrix can be reduced by left multiplication by upper-triangular matrices to a monomial matrix. The Bruhat cell $U(\sigma) \cong \mathbb{C}^{l(\sigma)}$ where $l(\sigma) = \#\lbrace (i,j) : i > j,\, \sigma^{-1}(i) < \sigma^{-1}(j) \rbrace$ is the number of inversions of $\sigma$.

The cell $X_{W'}$ corresponding to the longest element $W'$ of the Weyl group (sending every positive root to its negative) is a dense open subset of $G/B$, giving the **big cell**.

</div>

---

## Lecture 24: Weyl Character Formula

This lecture states the Weyl character formula and works out its consequences for each of the classical Lie algebras and for $\mathfrak{g}_2$. In particular, we derive determinantal formulas expressing the character of an irreducible representation as a polynomial in the characters of fundamental representations.

### 24.1 The Weyl Character Formula

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Alternating Sum $A_\mu$)</span></p>

For any weight $\mu$, define the element $A_\mu \in \mathbb{Z}[\Lambda]$ by

$$A_\mu = \sum_{W \in \mathfrak{W}} (-1)^W e(W(\mu)), \tag{24.1}$$

where $(-1)^W = \det(W)$ is the sign of the Weyl group element. The element $A_\mu$ is **alternating** under the Weyl group: $W(A_\mu) = (-1)^W A_\mu$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Half-Sum of Positive Roots $\rho$)</span></p>

Let $\rho$ denote half the sum of the positive roots:

$$\rho = \frac{1}{2} \sum_{\alpha \in R^+} \alpha.$$

Equivalently, $\rho$ is the sum of the fundamental weights $\omega_1 + \cdots + \omega_n$, and satisfies $\rho(H_{\alpha_i}) = 1$ for each simple root $\alpha_i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(24.2 — Weyl Character Formula)</span></p>

Let $\rho$ be half the sum of the positive roots, and $A_\rho \neq 0$. The character of the irreducible representation $\Gamma_\lambda$ with highest weight $\lambda$ is

$$\mathrm{Char}(\Gamma_\lambda) = \frac{A_{\lambda + \rho}}{A_\rho}. \tag{WCF}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(24.3 — Denominator Formula)</span></p>

The denominator $A_\rho$ of Weyl's formula is

$$A_\rho = \prod_{\alpha \in R^+} (e(\alpha/2) - e(-\alpha/2)) = e(\rho) \prod_{\alpha \in R^+} (1 - e(-\alpha)) = e(-\rho) \prod_{\alpha \in R^+} (e(\alpha) - 1).$$

In particular, $A_\rho \neq 0$ and has highest weight $\rho$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(24.6 — Weyl Dimension Formula)</span></p>

The dimension of the irreducible representation $\Gamma_\lambda$ is

$$\dim \Gamma_\lambda = \prod_{\alpha \in R^+} \frac{\langle \lambda + \rho, \alpha \rangle}{\langle \rho, \alpha \rangle} = \prod_{\alpha \in R^+} \frac{(\lambda + \rho, \alpha)}{(\rho, \alpha)},$$

where $\langle \alpha, \beta \rangle = 2(\alpha, \beta)/(\beta, \beta)$ and $(\,,\,)$ is the Killing form.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof of Dimension Formula)</span></p>

To extract the dimension from (WCF), factor the character homomorphism through the ring of power series $\mathbb{C}[[t]]$ by sending $e(\alpha) \mapsto e^{(\rho, \alpha)t}$, which maps $A_\mu$ to $\prod_{\alpha \in R^+}(\mu, \alpha) \cdot t^{|R^+|} + \text{higher order}$. Taking the ratio $A_{\lambda+\rho}/A_\rho$ and evaluating at $t = 0$ gives the product formula. $\square$

</div>

### 24.2 Applications to Classical Lie Algebras and Groups

#### The Case of $\mathrm{GL}_n\mathbb{C}$ ($\mathfrak{sl}_n\mathbb{C}$)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Schur Polynomial as Character)</span></p>

For $\mathrm{GL}_n\mathbb{C}$, the character of $\Gamma_\lambda$ with $\lambda = (\lambda_1, \ldots, \lambda_n)$ is the **Schur polynomial**

$$S_\lambda(x_1, \ldots, x_n) = \frac{|x_j^{\lambda_i + n - i}|}{|x_j^{n-i}|},$$

where $x_i = e(L_i)$, and $\rho = \sum (n-i)L_i = (n-1, n-2, \ldots, 0)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(24.10 — Giambelli / First Determinantal Formula for $\mathfrak{sl}_n\mathbb{C}$)</span></p>

Let $H_d = \mathrm{Char}(\mathrm{Sym}^d(\mathbb{C}^n))$ be the $d$th complete symmetric polynomial. For $\lambda = (\lambda_1 \ge \cdots \ge \lambda_k > 0)$:

$$\mathrm{Char}(\Gamma_\lambda) = |H_{\lambda_i + j - i}| = \det \begin{pmatrix} H_{\lambda_1} & H_{\lambda_1 + 1} & \cdots & H_{\lambda_1 + k - 1} \\\ H_{\lambda_2 - 1} & H_{\lambda_2} & \cdots \\\ \vdots & & \ddots \\\ H_{\lambda_k - k + 1} & \cdots & & H_{\lambda_k} \end{pmatrix}. \tag{24.10}$$

Equivalently, letting $E_d = \mathrm{Char}(\bigwedge^d(\mathbb{C}^n))$ and $\mu$ the conjugate partition:

$$\mathrm{Char}(\Gamma_\lambda) = |E_{\mu_i + j - i}|. \tag{24.11}$$

</div>

#### The Symplectic Case ($\mathfrak{sp}_{2n}\mathbb{C}$)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weyl Group and $A_\mu$ for $\mathfrak{sp}_{2n}\mathbb{C}$)</span></p>

The Weyl group elements can be written as products $\varepsilon\sigma$ where $\sigma \in \mathfrak{S}_n$ permutes the $L_i$ and $\varepsilon = (\varepsilon_1, \ldots, \varepsilon_n)$ with $\varepsilon_i = \pm 1$ changes signs. Here $\rho = (n, n-1, \ldots, 1)$ and

$$A_\mu = |x_j^{\mu_i} - x_j^{-\mu_i}|, \tag{24.14}$$

$$A_\rho = \Delta(x_1 + x_1^{-1}, \ldots, x_n + x_n^{-1}) \cdot (x_1 - x_1^{-1}) \cdots (x_n - x_n^{-1}), \tag{24.16}$$

where $\Delta$ is the discriminant.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Character and Dimension for $\mathfrak{sp}_{2n}\mathbb{C}$)</span></p>

For $\lambda = \sum \lambda_i L_i$ with $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$:

$$\mathrm{Char}(\Gamma_\lambda) = \frac{|x_j^{\lambda_i + n - i + 1} - x_j^{-(\lambda_i + n - i + 1)}|}{|x_j^{n - i + 1} - x_j^{-(n-i+1)}|}, \tag{24.18}$$

$$\dim(\Gamma_\lambda) = \prod_{i < j} \frac{l_i - l_j}{j - i} \cdot \prod_{i \le j} \frac{l_i + l_j}{2n + 2 - i - j} = \prod_{i < j} \frac{l_i^2 - l_j^2}{m_i^2 - m_j^2} \cdot \prod_i \frac{l_i}{m_i}, \tag{24.19}$$

where $l_i = \lambda_i + n - i + 1$ and $m_i = n - i + 1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(24.22 — Determinantal Formula for $\mathfrak{sp}_{2n}\mathbb{C}$)</span></p>

Let $J_d(x_1, \ldots, x_n) = H_d(x_1, \ldots, x_n, x_1^{-1}, \ldots, x_n^{-1})$ be the character of $\mathrm{Sym}^d(\mathbb{C}^{2n})$. If $\lambda = (\lambda_1 \ge \cdots \ge \lambda_r > 0)$, the character of $\Gamma_\lambda$ is the determinant of the $r \times r$ matrix whose $i$th row is

$$(J_{\lambda_i - i + 1} \quad J_{\lambda_i - i + 2} + J_{\lambda_i - i} \quad J_{\lambda_i - i + 3} + J_{\lambda_i - i - 1} \quad \cdots \quad J_{\lambda_i - i + r} + J_{\lambda_i - i - r + 2}).$$

</div>

#### The Odd Orthogonal Case ($\mathfrak{so}_{2n+1}\mathbb{C}$)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Character and Dimension for $\mathfrak{so}_{2n+1}\mathbb{C}$)</span></p>

Here $\rho = (n - \tfrac{1}{2}, n - \tfrac{3}{2}, \ldots, \tfrac{1}{2})$ and for $\lambda = \sum \lambda_i L_i$ with $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$:

$$\mathrm{Char}(\Gamma_\lambda) = \frac{|x_j^{\lambda_i + n - i + 1/2} - x_j^{-(\lambda_i + n - i + 1/2)}|}{|x_j^{n - i + 1/2} - x_j^{-(n-i+1/2)}|}, \tag{24.28}$$

$$\dim(\Gamma_\lambda) = \prod_{i < j} \frac{l_i - l_j}{j - i} \cdot \prod_{i \le j} \frac{l_i + l_j}{2n + 1 - i - j} = \prod_{i < j} \frac{l_i^2 - l_j^2}{m_i^2 - m_j^2} \cdot \prod_i \frac{l_i}{m_i}, \tag{24.29}$$

where $l_i = \lambda_i + n - i + \tfrac{1}{2}$ and $m_i = n - i + \tfrac{1}{2}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(24.33 — Determinantal Formula for $\mathrm{SO}_{2n+1}\mathbb{C}$)</span></p>

Let $K_d = H_d(x_1, \ldots, x_n, x_1^{-1}, \ldots, x_n^{-1}, 1)$ be the character of $\ker(\mathrm{Sym}^d(\mathbb{C}^{2n+1}) \to \mathrm{Sym}^{d-2}(\mathbb{C}^{2n+1}))$. If $\lambda = (\lambda_1 \ge \cdots \ge \lambda_r > 0)$ with all $\lambda_i$ integral, the character of $\Gamma_\lambda$ is the determinant of the $r \times r$ matrix whose $i$th row is

$$(K_{\lambda_i - i + 1} \quad K_{\lambda_i - i + 2} + K_{\lambda_i - i} \quad K_{\lambda_i - i + 3} + K_{\lambda_i - i - 1} \quad \cdots \quad K_{\lambda_i - i + r} + K_{\lambda_i - i - r + 2}).$$

</div>

#### The Even Orthogonal Case ($\mathfrak{so}_{2n}\mathbb{C}$)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Character and Dimension for $\mathfrak{so}_{2n}\mathbb{C}$)</span></p>

Here $\rho = (n-1, n-2, \ldots, 0)$ and the Weyl group uses only sign changes $\varepsilon$ with $\prod \varepsilon_i = +1$. For $\lambda = \sum \lambda_i L_i$ with $\lambda_1 \ge \cdots \ge |\lambda_n| \ge 0$:

$$A_\mu = \tfrac{1}{2}(|x_j^{\mu_i} + x_j^{-\mu_i}| + |x_j^{\mu_i} - x_j^{-\mu_i}|), \tag{24.37}$$

$$\mathrm{Char}(\Gamma_\lambda) = \frac{|x_j^{l_i} + x_j^{-l_i}| + |x_j^{l_i} - x_j^{-l_i}|}{|x_j^{n-i} + x_j^{-(n-i)}| + |x_j^{n-i} - x_j^{-(n-i)}|}, \tag{24.40}$$

where $l_i = \lambda_i + n - i$. When $\lambda_n = 0$ the second determinant in both numerator and denominator vanishes, and $\Gamma_\lambda$ is a representation of $\mathrm{O}_{2n}\mathbb{C}$. When $\lambda_n \neq 0$, $\Gamma_\lambda$ is the sum of two representations with highest weights $(\lambda_1, \ldots, \pm \lambda_n)$.

$$\dim(\Gamma_\lambda) = \prod_{i < j} \frac{(l_i - l_j)(l_i + l_j)}{(j - i)(2n - i - j)} = \prod_{i < j} \frac{l_i^2 - l_j^2}{m_i^2 - m_j^2}. \tag{24.41}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(24.44 — Determinantal Formula for $\mathrm{O}_{2n}\mathbb{C}$)</span></p>

Let $L_d = H_d(x_1, \ldots, x_n, x_1^{-1}, \ldots, x_n^{-1})$ be the character of $\ker(\mathrm{Sym}^d(\mathbb{C}^{2n}) \to \mathrm{Sym}^{d-2}(\mathbb{C}^{2n}))$. Given integers $\lambda_1 \ge \cdots \ge \lambda_r > 0$, the character of the irreducible representation of $\mathrm{O}_{2n}\mathbb{C}$ with highest weight $\lambda$ is the determinant of the $r \times r$ matrix whose $i$th row is

$$(L_{\lambda_i - i + 1} \quad L_{\lambda_i - i + 2} + L_{\lambda_i - i} \quad \cdots \quad L_{\lambda_i - i + r} + L_{\lambda_i - i - r + 2}).$$

</div>

#### The Exceptional Case $\mathfrak{g}_2$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(24.48 — Character of $\mathfrak{g}_2$)</span></p>

For $\mathfrak{g}_2$, with $\rho = 2\omega_1 + 3L_2 = (a+b+2)L_1 + (2b+3)L_2$ and highest weight $\lambda = a\omega_1 + b\omega_2$:

$$\mathrm{Char}(\Gamma_{a,b}) = \frac{S_{(a+2b+1, a+b+1)} - S_{(a+2b+1, b)}}{S_{(1,1)} - S_{(1)}},$$

where $S_{p,q,r}$ denotes the Schur polynomial for $\mathrm{GL}_3\mathbb{C}$.

The **dimension formula** for $\Gamma_{a,b}$ is

$$\dim(\Gamma_{a,b}) = \frac{(a+1)(a+b+2)(2a+3b+5)(a+2b+3)(a+3b+4)(b+1)}{120}.$$

This gives dimensions 7 and 14 for $a=1, b=0$ and $a=0, b=1$ (standard and adjoint), and $\dim(\Gamma_{2,0}) = 27$, confirming $\bigwedge^3 V = \Gamma_{2,0} \oplus V \oplus \mathbb{C}$ and $\mathrm{Sym}^2 V = \Gamma_{2,0} \oplus \mathbb{C}$.

</div>

---

## Lecture 25: More Character Formulas

In this lecture we give two more formulas for the multiplicities of an irreducible representation of a semisimple Lie algebra or group. First, Freudenthal's formula ($\S$25.1) gives a straightforward way of calculating the multiplicity of a given weight once we know the multiplicity of all higher ones. This in turn allows us to prove in $\S$25.2 the Weyl character formula, as well as another multiplicity formula due to Kostant. Finally, in $\S$25.3 we give Steinberg's formula for the decomposition of the tensor product of two arbitrary irreducible representations of a semisimple Lie algebra, and also give formulas for some pairs $\mathfrak{h} \subset \mathfrak{g}$ for the decomposition of the restriction to $\mathfrak{h}$ of irreducible representations of $\mathfrak{g}$.

### 25.1 Freudenthal's Multiplicity Formula

Freudenthal's formula gives a general way of computing the multiplicities of a representation, i.e., the dimensions of its weight spaces, by working down successively from the highest weight. The result is similar to (but more complicated than) what we did for $\mathfrak{sl}_3\mathbb{C}$ in Lecture 13, where we found the multiplicities along successive concentric hexagons in the weight diagram.

Let $\Gamma_\lambda$ be the irreducible representation with highest weight $\lambda$, which will be fixed throughout this discussion. Let $n_\mu = n_\mu(\Gamma_\lambda)$ be the dimension of the weight space of weight $\mu$ in $\Gamma_\lambda$, i.e., $\mathrm{Char}(\Gamma_\lambda) = \sum n_\mu e(\mu)$. Freudenthal gives a formula for $n_\mu$ in terms of multiplicities of weights that are higher than $\mu$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(25.1 — Freudenthal's Multiplicity Formula)</span></p>

With the above notation,

$$c(\mu) \cdot n_\mu(\Gamma_\lambda) = 2 \sum_{\alpha \in R^+} \sum_{k \ge 1} (\mu + k\alpha,\, \alpha)\, n_{\mu + k\alpha},$$

where $c(\mu) = \lVert \lambda + \rho \rVert^2 - \lVert \mu + \rho \rVert^2$.

Here $\lVert \beta \rVert^2 = (\beta, \beta)$, $(\,,\,)$ is the Killing form, and $\rho$ is half the sum of the positive roots.

</div>

Note that $c(\mu)$ is positive if $\mu \neq \lambda$ and $n_\mu > 0$, so the formula determines every multiplicity recursively from $n_\lambda = 1$.

#### The Casimir Operator

The proof of Freudenthal's formula uses a **Casimir operator**, denoted $C$. This is an endomorphism of any representation $V$ of the semisimple Lie algebra $\mathfrak{g}$, and is constructed as follows. Take any basis $U_1, \ldots, U_l$ for $\mathfrak{g}$, and let $U_1', \ldots, U_l'$ be the dual basis with respect to the Killing form on $\mathfrak{g}$. Set

$$C = U_1 U_1' + \cdots + U_l U_l',$$

i.e., for any $v \in V$, $C(v) = \sum U_i \cdot (U_i' \cdot v)$.

The key fact is that $C$ commutes with every operation in $\mathfrak{g}$:

$$C(X \cdot v) = X \cdot C(v) \quad \text{for all } X \in \mathfrak{g},\; v \in V.$$

The idea is to use a special basis for the construction of $C$, so that each term $U_i U_i'$ will act as multiplication by a constant on any weight space, and this constant can be calculated in terms of multiplicities. Then Schur's lemma can be applied to know that, in case $V$ is irreducible, $C$ itself is multiplication by a scalar. Taking traces will lead to a relation among multiplicities, and a little algebraic manipulation will give Freudenthal's formula.

The basis for $\mathfrak{g}$ to use is a natural one: Choose the basis $H_1, \ldots, H_n$ for the Cartan subalgebra $\mathfrak{h}$, where $H_i = H_{\alpha_i}$ corresponds to the simple root $\alpha_i$, and let $H_i'$ be the dual basis for the restriction of the Killing form to $\mathfrak{h}$. For each root $\alpha$, choose a nonzero $X_\alpha \in \mathfrak{g}_\alpha$. The dual basis will then have $X_\alpha'$ in $\mathfrak{g}_{-\alpha}$. In fact, if we let $Y_\alpha \in \mathfrak{g}_{-\alpha}$ be the usual element so that $X_\alpha$, $Y_\alpha$, and $H_\alpha = [X_\alpha, Y_\alpha]$ are the canonical basis for the subalgebra $\mathfrak{s}_\alpha \cong \mathfrak{sl}_2\mathbb{C}$ that they span, then

$$X_\alpha' = ((\alpha, \alpha)/2)\, Y_\alpha. \tag{25.5}$$

Now we have the Casimir operator

$$C = \sum H_i H_i' + \sum_{\alpha \in R} X_\alpha X_\alpha',$$

and we analyze the action of $C$ on the weight space $V_\mu$ corresponding to weight $\mu$ for any representation $V$. Let $n_\mu = \dim(V_\mu)$. First we have

$$\sum H_i H_i' \text{ acts on } V_\mu \text{ by multiplication by } (\mu, \mu) = \lVert \mu \rVert^2. \tag{25.7}$$

Indeed, $H_i H_i'$ acts by multiplication by $\mu(H_i)\mu(H_i')$. If we write $\mu = \sum r_i \omega_i$, where the $\omega_i$ are the fundamental weights, then $\mu(H_i) = r_i$, and if $\mu = \sum r_i' \omega_i'$ with $\omega_i'$ the dual basis to $\omega_i$, then similarly $\mu(H_i') = r_i'$. Hence $\sum \mu(H_i)\mu(H_i') = \sum r_i r_i' = (\mu, \mu)$, as asserted.

Now consider the action of $X_\alpha X_\alpha' = ((\alpha, \alpha)/2) X_\alpha Y_\alpha$ on $V_\mu$. Restricting to the subalgebra $\mathfrak{s}_\alpha \cong \mathfrak{sl}_2$ and to the subrepresentation $\bigoplus_i V_{\mu + i\alpha}$ corresponding to the $\alpha$-string through $\mu$, we are in a situation which we know very well. Suppose this string is

$$V_\beta \oplus V_{\beta - \alpha} \oplus \cdots \oplus V_{\beta - m\alpha},$$

so $m = \beta(H_\alpha)$ [cf. (14.10)], and let $k$ be the integer such that $\mu = \beta - k\alpha$. On the first term $V_\beta$, $X_\alpha Y_\alpha$ acts by multiplication by $(\beta, \alpha)$. In general, on the part of $V_{\beta - k\alpha}$ which is the image of $V_\beta$ by $(Y_\alpha)^k$, $X_\alpha Y_\alpha$ acts by multiplication by $(k+1)(m-k)$. After rewriting in terms of $\mu$ and $\alpha$:

$$(k+1)(\beta, \alpha) - k(\alpha, \alpha)/2 = (k+1)((\mu, \alpha) + k(\alpha, \alpha)/2) - k(\alpha, \alpha)/2.$$

Continuing to peel off subrepresentations (over $\mathfrak{s}_\alpha$) of $V$ spanned by $V_\beta$, and applying the same reasoning to what is left, the space $V_\mu$ is decomposed into pieces on which $X_\alpha X_\alpha'$ acts by multiplication by a scalar. The trace of $X_\alpha X_\alpha'\vert_{V_\mu}$ is therefore the sum

$$\mathrm{Trace}(X_\alpha X_\alpha'\vert_{V_\mu}) = \sum_{i=0}^{k} (\mu + i\alpha, \alpha)\, n_{\mu + i\alpha}. \tag{25.8}$$

One pleasant fact about this sum is that it may be extended to all $i \ge 0$, since $n_{\mu + i\alpha} = 0$ for $i > k$.

In case $k \ge m/2$, the computation is similar, peeling off representations from the other end, starting with $V_{\beta - m\alpha}$. The only difference is that the action of $X_\alpha Y_\alpha$ on $V_{\beta - m\alpha}$ is zero. The result is

$$\mathrm{Trace}(X_\alpha X_\alpha'\vert_{V_\mu}) = -\sum_{i=1}^{\infty} (\mu - i\alpha, \alpha)\, n_{\mu - i\alpha}. \tag{25.9}$$

In fact, (25.8) is valid for all $\mu$ and $\alpha$, as we see from the identity

$$\sum_{i=-\infty}^{\infty} (\mu + i\alpha, \alpha)\, n_{\mu + i\alpha} = 0. \tag{25.11}$$

Now we add the assumption that $V$ is irreducible, so $C$ is multiplication by some scalar $c$. Taking the trace of $C$ on $V_\mu$ and adding, we get

$$c\, n_\mu = (\mu, \mu)\, n_\mu + \sum_{\alpha \in R} \sum_{i \ge 0} (\mu + i\alpha, \alpha)\, n_{\mu + i\alpha}. \tag{25.13}$$

Note that when $i = 0$ the two terms for $\alpha$ and $-\alpha$ cancel each other, so the summation can begin at $i = 1$ instead. Rewriting this in terms of the positive roots, and using (25.11) the sums become

$$\sum_{\alpha \in R^+} \sum_{i=1}^{\infty} (\mu + i\alpha, \alpha)\, n_{\mu + i\alpha} + \sum_{\alpha \in R^+} \sum_{i=1}^{\infty} (\mu - i\alpha, \alpha)\, n_{\mu - i\alpha}$$

$$= n_\mu \sum_{\alpha \in R^+} (\mu, \alpha) + 2 \sum_{\alpha \in R^+} \sum_{i=1}^{\infty} (\mu + i\alpha, \alpha)\, n_{\mu + i\alpha}.$$

Summarizing, and observing that $\sum_{\alpha \in R^+} (\mu, \alpha) = (\mu, 2\rho)$, we have

$$c\, n_\mu = ((\mu, \mu) + (\mu, 2\rho))\, n_\mu + 2 \sum_{\alpha \in R^+} \sum_{i=1}^{\infty} (\mu + i\alpha, \alpha)\, n_{\mu + i\alpha}. \tag{25.14}$$

Note that $(\mu, \mu) + (\mu, 2\rho) = (\mu + \rho, \mu + \rho) - (\rho, \rho) = \lVert \mu + \rho \rVert^2 - \lVert \rho \rVert^2$. To evaluate the constant we evaluate on the highest weight space $V_\lambda$, where $n_\lambda = 1$ and $n_{\lambda + i\alpha} = 0$ for $i > 0$. Hence,

$$c = (\lambda, \lambda) + (\lambda, 2\rho) = \lVert \lambda + \rho \rVert^2 - \lVert \rho \rVert^2. \tag{25.14'}$$

Combining the preceding two equations yields Freudenthal's formula. $\square$

### 25.2 Proof of (WCF); the Kostant Multiplicity Formula

It is not unreasonable to anticipate that Weyl's character formula can be deduced from Freudenthal's inductive formula, but some algebraic manipulation is certainly required. Let

$$\chi_\lambda = \mathrm{Char}(\Gamma_\lambda) = \sum n_\mu\, e(\mu)$$

be the character of the irreducible representation with highest weight $\lambda$. Freudenthal's formula, in form (25.13), reads

$$c \cdot \chi_\lambda = \sum_\mu (\mu, \mu)\, n_\mu\, e(\mu) + \sum_\mu \sum_{\alpha \in R} \sum_{i=0}^{\infty} (\mu + i\alpha, \alpha)\, n_{\mu + i\alpha}\, e(\mu),$$

where $c = \lVert \lambda + \rho \rVert^2 - \lVert \rho \rVert^2$. To get this to look anything like Weyl's formula, we must get rid of the inside sums over $i$. If $\alpha$ is fixed, they will disappear if we multiply by $e(\alpha) - 1$, as successive terms cancel:

$$(e(\alpha) - 1) \cdot \sum_\mu \sum_{i=0}^{\infty} (\mu + i\alpha, \alpha)\, n_{\mu + i\alpha}\, e(\mu) = \sum_\mu (\mu, \alpha)\, n_\mu\, e(\mu + \alpha).$$

Let $P = \prod_{\alpha \in R}(e(\alpha) - 1) = (e(\alpha) - 1) \cdot P_\alpha$, where $P_\alpha = \prod_{\beta \neq \alpha}(e(\beta) - 1)$. The preceding two formulas give

$$c \cdot P \cdot \chi_\lambda = P \cdot \sum_\mu (\mu, \mu)\, n_\mu\, e(\mu) + \sum_\mu (\mu, \alpha)\, P_\alpha\, n_\mu\, e(\mu + \alpha). \tag{25.17}$$

Note also that

$$P = (-1)^r A_\rho \cdot A_\rho,$$

where $r$ is the number of positive roots, so at least the formula now involves the ingredients that go into (WCF).

#### The Laplacian Approach

We want to prove (WCF): $A_\rho \cdot \chi_\lambda = A_{\lambda + \rho}$. We have seen in $\S$24.1 that both sides of this equation are alternating, and that both have highest weight term $e(\lambda + \rho)$, with coefficient 1. On the right-hand side the only terms that appear are those of the form $\pm e(W(\lambda + \rho))$, for $W$ in the Weyl group. To prove (WCF), it suffices to prove that the only terms appearing with nonzero coefficients in $A_\rho \cdot \chi_\lambda$ are these same $e(W(\lambda + \rho))$, for then the alternating property and the knowledge of the coefficient of $e(\lambda + \rho)$ determine all the coefficients.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span><span class="math-callout__name">(Key Claim for WCF)</span></p>

The only terms $e(\nu)$ occurring in $A_\rho \cdot \chi_\lambda$ with nonzero coefficient are those with $\lVert \nu \rVert = \lVert \lambda + \rho \rVert$.

</div>

To see that this is equivalent, note that by definition of $A_\rho$ and $\chi_\lambda$, the terms in $A_\rho \cdot \chi_\lambda$ are all of the form $\pm e(\nu)$, where $\nu = \mu + W(\rho)$, for $\mu$ a weight of $\Gamma_\lambda$ and $W$ in the Weyl group. But if $\lVert \mu + W(\rho) \rVert = \lVert \lambda + \rho \rVert$, since the metric is invariant by the Weyl group, this gives $\lVert W^{-1}(\mu) + \rho \rVert = \lVert \lambda + \rho \rVert$. But we saw in Exercise 25.2 that this cannot happen unless $\mu = W(\lambda)$, as required.

This suggests looking at the **Laplacian** operator that maps $e(\mu)$ to $\lVert \mu \rVert^2 e(\mu)$, that is, the map

$$\Delta\colon \mathbb{C}[\Lambda] \to \mathbb{C}[\Lambda]$$

defined by

$$\Delta\!\left(\sum m_\mu\, e(\mu)\right) = \sum (\mu, \mu)\, m_\mu\, e(\mu).$$

The claim is equivalent to the assertion that $F = A_\rho \cdot \chi_\lambda$ satisfies the "differential equation"

$$\Delta(F) = \lVert \lambda + \rho \rVert^2 F.$$

From the definition $\Delta(\chi_\lambda) = \sum (\mu, \mu)\, n_\mu\, e(\mu)$. And $\Delta(A_\rho) = \lVert \rho \rVert^2 A_\rho$. In general, $\Delta(A_\alpha) = \sum (-1)^W \lVert W(\alpha) \rVert^2 e(W(\alpha)) = \lVert \alpha \rVert^2 A_\alpha$ for all $W \in \mathfrak{W}$.

So we would be in good shape if we had a formula for $\Delta$ of a product of two functions. One expects such a formula to take the form

$$\Delta(fg) = \Delta(f)g + 2(\nabla f, \nabla g) + f\Delta(g), \tag{25.18}$$

where $\nabla$ is a "gradient," and $(\,,\,)$ is an "inner product." Taking $f = e(\mu)$, $g = e(\nu)$, we see that we need to have $(\nabla e(\mu), \nabla e(\nu)) = (\mu, \nu)\, e(\mu + \nu)$. There is indeed such a gradient and inner product. Define a homomorphism

$$\nabla\colon \mathbb{C}[\Lambda] \to \mathfrak{h}^* \otimes \mathbb{C}[\Lambda] = \mathrm{Hom}(\mathfrak{h}, \mathbb{C}[\Lambda])$$

by the formula $\nabla(e(\mu)) = \mu \cdot e(\mu)$, and define the bilinear form $(\,,\,)$ on $\mathfrak{h}^* \otimes \mathbb{C}[\Lambda]$ by the formula $(\alpha e(\mu), \beta e(\nu)) = (\alpha, \beta)\, e(\mu + \nu)$, where $(\alpha, \beta)$ is the Killing form on $\mathfrak{h}^*$.

For example, $\nabla(\chi_\lambda) = \sum_\mu n_\mu\, \mu \cdot e(\mu)$, and, by the Leibnitz rule,

$$\nabla(P) = \sum_{\alpha \in R^+} P_\alpha \cdot e(\alpha).$$

But now look at formula (25.17). It says exactly that

$$c \cdot P \chi_\lambda = P \Delta(\chi_\lambda) + 2(\nabla P, \nabla \chi_\lambda).$$

Since, also by the exercise, $\nabla(P) = 2(-1)^r A_\rho \nabla(A_\rho)$, we may cancel $(-1)^r A_\rho$ from each term in the equation, getting

$$c \cdot A_\rho \chi_\lambda = A_\rho \Delta(\chi_\lambda) + 2(\nabla A_\rho, \nabla \chi_\lambda).$$

By the identity (25.18), the right-hand side of this equation is

$$\Delta(A_\rho \chi_\lambda) - \Delta(A_\rho) \chi_\lambda = \Delta(A_\rho \chi_\lambda) - \lVert \rho \rVert^2 A_\rho \chi_\lambda.$$

Since $c = \lVert \lambda + \rho \rVert^2 - \lVert \rho \rVert^2$, this gives $\lVert \lambda + \rho \rVert^2 A_\rho \chi_\lambda = \Delta(A_\rho \chi_\lambda)$, which finishes the proof. $\square$

#### Kostant's Multiplicity Formula

We conclude this section with a proof of another general multiplicity formula, discovered by Kostant. It gives an elegant closed formula for the multiplicities, but at the expense of summing over the entire Weyl group (although as we will indicate below, there are many interesting cases where all but a few terms of the sum vanish). It also involves a kind of partition counting function. For each weight $\mu$, let $P(\mu)$ be the number of ways to write $\mu$ as a sum of positive roots; set $P(0) = 1$. Equivalently,

$$\prod_{\alpha \in R^+} \frac{1}{1 - e(\alpha)} = \sum_\mu P(\mu)\, e(\mu). \tag{25.20}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(25.21 — Kostant's Multiplicity Formula)</span></p>

The multiplicity $n_\mu(\Gamma_\lambda)$ of weight $\mu$ in the irreducible representation $\Gamma_\lambda$ is given by

$$n_\mu(\Gamma_\lambda) = \sum_{W \in \mathfrak{W}} (-1)^W P(W(\lambda + \rho) - (\mu + \rho)),$$

where $\rho$ is half the sum of the positive roots.

</div>

**Proof.** Write $(A_\rho)^{-1} = e(-\rho) \prod (1 - e(-\alpha)) = \sum_\nu P(\nu)\, e(-\nu - \rho)$. By (WCF),

$$\chi_\lambda = A_{\lambda+\rho}(A_\rho)^{-1} = \sum_{W,\nu} (-1)^W P(\nu)\, e(W(\lambda+\rho) - \nu - \rho)$$

$$= \sum_{W,\mu} (-1)^W P(W(\lambda + \rho) - (\mu + \rho))\, e(\mu),$$

as seen by writing $\mu = W(\lambda + \rho) - \nu - \rho$. $\square$

In fact, the proof shows that Kostant's formula is equivalent to Weyl's formula, cf. [Cart].

One way to interpret Kostant's formula, at least for weights $\mu$ close to the highest weight $\lambda$ of $\Gamma_\lambda$, is as a sort of converse to Proposition 14.13(ii). Recall that this says that $\Gamma_\lambda$ will be generated by the images of its highest weight vector $v$ under successive applications of the generators of the negative root spaces; in practice, we used this fact to bound from above the multiplicities of various weights $\mu$ close to $\lambda$ by counting the number of ways of getting from $\lambda$ to $\mu$ by adding negative roots. The problem in making this precise was always that we did not know how many relations there were among these images, if any. Kostant's formula gives an answer: for example, if the difference $\lambda - \mu$ is small relative to $\lambda$, we see that the only nonzero term in the sum is the principle term, corresponding to $W = 1$; in this case the answer is that there are no relations other than the trivial ones $X(Y(v)) - Y(X(v)) = [X, Y](v)$.

#### The Partition Function $P$ for $\mathfrak{sl}_3\mathbb{C}$

To see how the partition function $P$ works in practice, note that for $\mathfrak{sl}_3\mathbb{C}$ the function $P(\mu)$ will be a constant 1 on the rays $\lbrace aL_2 - aL_1\rbrace_{a \ge 0}$ and $\lbrace aL_3 - aL_2\rbrace_{a \ge 0}$ through the origin in the direction of the two simple positive roots $L_2 - L_1$ and $L_3 - L_2$. It will have value 2 on the translates $\lbrace aL_2 - (a+3)L_1\rbrace_{a \ge -1}$ and $\lbrace aL_3 - (a-3)L_2\rbrace_{a \ge 2}$ of these two rays by the third positive root $L_3 - L_1$: for example, the first of these can be written as

$$aL_2 - (a+3)L_1 = (a+1) \cdot (L_2 - L_1) + L_3 - L_1$$

$$= (a+2) \cdot (L_2 - L_1) + L_3 - L_2;$$

and correspondingly its value will increase by 1 on each successive translate of these rays by $L_3 - L_1$.

Now, the prescription given in the Kostant formula for the multiplicities is to take six copies of this function flipped about the origin, translated so that the vertex of the outer shell lies at the points $w(\lambda + \rho) - \rho$ and take their alternating sum. Superimposing the six pictures gives the hexagonal pattern of the multiplicities.

### 25.3 Tensor Products and Restrictions to Subgroups

In the case of the general or special linear groups, we saw general formulas for describing how the tensor product $\Gamma_\lambda \otimes \Gamma_\mu$ of two irreducible representations decomposes:

$$\Gamma_\lambda \otimes \Gamma_\mu = \bigoplus N_{\lambda\mu\nu}\, \Gamma_\nu.$$

In these cases the multiplicities $N_{\lambda\mu\nu}$ can be described by a combinatorial formula: the Littlewood--Richardson rule. In general, such a decomposition is equivalent to writing

$$\chi_\lambda \chi_\mu = \sum N_{\lambda\mu\nu}\, \chi_\nu \tag{25.25}$$

in $\mathbb{Z}[\Lambda]$, where $\chi_\lambda = \mathrm{Char}(\Gamma_\lambda)$ denotes the character. By Weyl's character formula, these multiplicities $N_{\lambda\mu\nu}$ are determined by the identity

$$A_{\lambda+\rho} \cdot A_{\mu+\rho} = \sum N_{\lambda\mu\nu}\, A_\rho \cdot A_{\nu+\rho}. \tag{25.26}$$

This formula gives an effective procedure for calculating the coefficients $N_{\lambda\mu\nu}$, if one that is tedious in practice: we can peel off highest weights, i.e., successively subtract from $A_{\lambda+\rho} \cdot A_{\mu+\rho}$ multiples of $A_\rho \cdot A_{\nu+\rho}$ for the highest $\nu$ that appears.

There are some explicit formulas for the other classical groups. R. C. King [Ki2] has showed that for both the symplectic or orthogonal groups, the multiplicities $N_{\lambda\mu\nu}$ are given by the formula

$$N_{\lambda\mu\nu} = \sum_{\zeta, \sigma, \tau} M_{\zeta\sigma\lambda} \cdot M_{\zeta\tau\mu} \cdot M_{\sigma\tau\nu}, \tag{25.27}$$

where the $M$'s denote the Littlewood--Richardson multiplicities, i.e., the corresponding numbers for the general linear group, and the sum is over all partitions $\zeta$, $\sigma$, $\tau$.

#### Steinberg's Formula

Steinberg has also given a general formula for the multiplicities $N_{\lambda\mu\nu}$. Since it involves a double summation over the Weyl group, using it in a concrete situation may be a challenge.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(25.29 — Steinberg's Formula)</span></p>

The multiplicity of $\Gamma_\nu$ in $\Gamma_\lambda \otimes \Gamma_\mu$ is

$$N_{\lambda\mu\nu} = \sum_{W, W'} (-1)^{WW'} P(W(\lambda + \rho) + W'(\mu + \rho) - \nu - 2\rho),$$

where the sum is over pairs $W, W' \in \mathfrak{W}$, and $P$ is the counting function appearing in Kostant's multiplicity formula.

</div>

#### Branching Rules and Restrictions to Subgroups

The following is the generalization of something we have seen several times:

If $\lambda$ and $\mu$ are dominant weights, and $\alpha$ is a simple root with $\lambda(H_\alpha)$ and $\mu(H_\alpha)$ not zero, show that $\lambda + \mu - \alpha$ is a dominant weight and $\Gamma_\lambda \otimes \Gamma_\mu$ contains the irreducible representation $\Gamma_{\lambda+\mu-\alpha}$ with multiplicity one. So

$$\Gamma_\lambda \otimes \Gamma_\mu = \Gamma_{\lambda+\mu} \oplus \Gamma_{\lambda+\mu-\alpha} \oplus \text{others}.$$

In case $\mu = \lambda$, with $\lambda(H_\alpha) \neq 0$, $\mathrm{Sym}^2(\Gamma_\lambda)$ contains $\Gamma_{\lambda+\mu}$, while $\bigwedge^2(\Gamma_\lambda)$ contains $\Gamma_{\lambda+\mu-\alpha}$.

Similarly, one can ask for formulas for decomposing restrictions for other inclusions, such as the natural embeddings: $\mathrm{Sp}_{2n}\mathbb{C} \subset \mathrm{SL}_{2n}\mathbb{C}$, $\mathrm{SO}_m\mathbb{C} \subset \mathrm{SL}_m\mathbb{C}$, $\mathrm{GL}_m\mathbb{C} \times \mathrm{GL}_n\mathbb{C} \subset \mathrm{GL}_{m+n}\mathbb{C}$, and many more. Such formulas are determined in principle by computing what happens to generators of the representation rings; one need only decompose exterior or symmetric products of standard representations. Such formulas are often called **branching formulas** or **modification rules**.

We state what happens when the irreducible representations of $\mathrm{GL}_m\mathbb{C}$ are restricted to the orthogonal or symplectic subgroups, referring to [Lit3] for the proofs:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Restriction to $\mathrm{O}_m\mathbb{C}$)</span></p>

For $\mathrm{O}_m\mathbb{C} \subset \mathrm{GL}_m\mathbb{C}$, with $m = 2n$ or $2n+1$, given $\lambda = (\lambda_1 \ge \cdots \ge \lambda_n \ge 0)$, the restriction is

$$\mathrm{Res}^{\mathrm{GL}_m\mathbb{C}}_{\mathrm{O}_m\mathbb{C}}(\Gamma_\lambda) = \bigoplus N_{\lambda\bar{\lambda}}\, \Gamma_{\bar{\lambda}}, \tag{25.37}$$

the sum over all $\bar{\lambda} = (\bar{\lambda}_1 \ge \cdots \ge \bar{\lambda}_n \ge 0)$, where

$$N_{\lambda\bar{\lambda}} = \sum_\delta N_{\delta\bar{\lambda}\lambda},$$

with $N_{\delta\bar{\lambda}\lambda}$ the Littlewood--Richardson coefficient, and the sum over all $\delta = (\delta_1 \ge \delta_2 \ge \cdots)$ with all $\delta_i$ even.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Restriction to $\mathrm{Sp}_{2n}\mathbb{C}$)</span></p>

Similarly for $\mathrm{Sp}_{2n}\mathbb{C} \subset \mathrm{GL}_{2n}\mathbb{C}$,

$$\mathrm{Res}^{\mathrm{GL}_{2n}\mathbb{C}}_{\mathrm{Sp}_{2n}\mathbb{C}}(\Gamma_\lambda) = \bigoplus N_{\lambda\bar{\lambda}}\, \Gamma_{\bar{\lambda}}, \tag{25.39}$$

the sum over all $\bar{\lambda} = (\bar{\lambda}_1 \ge \cdots \ge \bar{\lambda}_n \ge 0)$, where

$$N_{\lambda\bar{\lambda}} = \sum_\eta N_{\eta\bar{\lambda}\lambda},$$

$N_{\eta\bar{\lambda}\lambda}$ is the Littlewood--Richardson coefficient, and the sum is over all $\eta = (\eta_1 = \eta_2 \ge \eta_3 = \eta_4 \ge \cdots)$ with each part occurring an even number of times.

</div>

It is perhaps worth pointing out that the decomposition of tensor products is a special case of the decomposition of restrictions: the exterior tensor product $\Gamma_\lambda \boxtimes \Gamma_\mu$ of two irreducible representations of $G$ is an irreducible representation of $G \times G$, and the restriction of this to the diagonal embedding of $G$ in $G \times G$ is the usual tensor product $\Gamma_\lambda \otimes \Gamma_\mu$.

#### Branching Rules for Classical Inclusions

For $\mathfrak{so}_{2n}\mathbb{C} \subset \mathfrak{so}_{2n+1}\mathbb{C}$, and $\Gamma_\lambda$ the irreducible representation of $\mathfrak{so}_{2n+1}\mathbb{C}$ given by $\lambda = (\lambda_1 \ge \cdots \ge \lambda_n \ge 0)$, the restriction is

$$\mathrm{Res}^{\mathfrak{so}_{2n+1}\mathbb{C}}_{\mathfrak{so}_{2n}\mathbb{C}}(\Gamma_\lambda) = \bigoplus \Gamma_{\bar{\lambda}}, \tag{25.34}$$

the sum over all $\bar{\lambda} = (\bar{\lambda}_1, \ldots, \bar{\lambda}_n)$ with

$$\lambda_1 \ge \bar{\lambda}_1 \ge \lambda_2 \ge \bar{\lambda}_2 \ge \cdots \ge \lambda_{n-1} \ge \bar{\lambda}_n \ge \lambda_n \ge |\bar{\lambda}_n|,$$

with the $\bar{\lambda}_i$ and $\lambda_i$ simultaneously all integers or all half integers.

For $\mathfrak{so}_{2n-1}\mathbb{C} \subset \mathfrak{so}_{2n}\mathbb{C}$, and $\Gamma_\lambda$ the irreducible representation of $\mathfrak{so}_{2n}\mathbb{C}$ given by $\lambda = (\lambda_1 \ge \cdots \ge |\lambda_n|)$,

$$\mathrm{Res}^{\mathfrak{so}_{2n}\mathbb{C}}_{\mathfrak{so}_{2n-1}\mathbb{C}}(\Gamma_\lambda) = \bigoplus \Gamma_{\bar{\lambda}}, \tag{25.35}$$

the sum over all $\bar{\lambda} = (\bar{\lambda}_1, \ldots, \bar{\lambda}_{n-1})$ with

$$\lambda_1 \ge \bar{\lambda}_1 \ge \lambda_2 \ge \bar{\lambda}_2 \ge \cdots \ge \lambda_{n-1} \ge \bar{\lambda}_{n-1} \ge |\lambda_n|,$$

with the $\bar{\lambda}_i$ and $\lambda_i$ simultaneously all integers or all half integers.

For $\mathfrak{sp}_{2n-2}\mathbb{C} \subset \mathfrak{sp}_{2n}\mathbb{C}$, and $\Gamma_\lambda$ the irreducible representation of $\mathfrak{sp}_{2n}\mathbb{C}$ given by $\lambda = (\lambda_1 \ge \cdots \ge \lambda_n \ge 0)$, the restriction is

$$\mathrm{Res}^{\mathfrak{sp}_{2n}\mathbb{C}}_{\mathfrak{sp}_{2n-2}\mathbb{C}}(\Gamma_\lambda) = \bigoplus N_{\lambda\bar{\lambda}}\, \Gamma_{\bar{\lambda}}, \tag{25.36}$$

the sum over all $\bar{\lambda} = (\bar{\lambda}_1, \ldots, \bar{\lambda}_{n-1})$ with $\bar{\lambda}_1 \ge \cdots \ge \bar{\lambda}_{n-1} \ge 0$, and the multiplicity $N_{\lambda\bar{\lambda}}$ is the number of sequences $p_1, \ldots, p_n$ of integers satisfying

$$\lambda_1 \ge p_1 \ge \lambda_2 \ge p_2 \ge \cdots \ge \lambda_n \ge p_n \ge 0$$

and

$$p_1 \ge \bar{\lambda}_1 \ge p_2 \ge \bar{\lambda}_2 \ge \cdots \ge p_{n-1} \ge \bar{\lambda}_{n-1} \ge p_n.$$

As in the case of $\mathrm{GL}_n\mathbb{C}$, these formulas are equivalent to identities among symmetric polynomials. As we saw in the case of the general linear group, these branching rules can be used inductively to compute the dimensions of the weight spaces. For example, for $\mathfrak{so}_m\mathbb{C}$ consider the chain

$$\mathfrak{so}_m\mathbb{C} \supset \mathfrak{so}_{m-1}\mathbb{C} \supset \mathfrak{so}_{m-2}\mathbb{C} \supset \cdots \supset \mathfrak{so}_3\mathbb{C}.$$

Decomposing a representation successively from one layer to the next will finally write it as a sum of one-dimensional weight spaces, and the dimension can be read off from the number of "partitions" in chains that start with the given $\lambda$. The representations can be constructed from these chains, as described by Gelfand and Zetlin, cf. [$\check{\text{Z}}$el, $\S$10].

#### Restriction to a General Subalgebra

There are also some general formulas, valid whenever $\bar{\mathfrak{g}}$ is a semisimple subalgebra of a semisimple Lie algebra $\mathfrak{g}$. Assume that the Cartan subalgebra $\bar{\mathfrak{h}}$ is a subalgebra of $\mathfrak{h}$, and we assume the half-spaces determining positive roots are compatible. We write $\bar{\mu}$ for weights of $\bar{\mathfrak{g}}$, and we write $\mu \downarrow \bar{\mu}$ to mean that a weight $\mu$ of $\mathfrak{g}$ restricts to $\bar{\mu}$. Similarly write $\bar{W}$ for a typical element of the Weyl group of $\bar{\mathfrak{g}}$, and $\bar{\rho}$ for half the sum of its positive weights. If $\lambda$ (resp. $\bar{\lambda}$) is a dominant weight for $\mathfrak{g}$ (resp. $\bar{\mathfrak{g}}$), let $N_{\lambda\bar{\lambda}}$ denote the multiplicity with which $\Gamma_{\bar{\lambda}}$ appears in the restriction of $\Gamma_\lambda$ to $\bar{\mathfrak{g}}$, i.e.,

$$\mathrm{Res}(\Gamma_\lambda) = \bigoplus_{\bar{\lambda}} N_{\lambda\bar{\lambda}}\, \Gamma_{\bar{\lambda}}.$$

Then for any dominant weight $\lambda$ of $\mathfrak{g}$ and any weight $\bar{\mu}$ of $\bar{\mathfrak{g}}$,

$$\sum_{\mu \downarrow \bar{\mu}} n_\mu(\Gamma_\lambda) = \sum_{\bar{\lambda}} N_{\lambda\bar{\lambda}}\, n_{\bar{\mu}}(\Gamma_{\bar{\lambda}}).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Klimyk's Formula)</span></p>

$$N_{\lambda\bar{\lambda}} = \sum_{\bar{W}} (-1)^{\bar{W}} \sum_{\mu \downarrow \bar{\lambda} + \bar{\rho} - \bar{W}(\bar{\rho})} n_\mu(\Gamma_\lambda).$$

</div>

#### Cartan Multiplication and the Ideal $J^*$

Finally, we note that it is possible, for any semisimple Lie algebra $\mathfrak{g}$, to make the direct sum of all its irreducible representations into a commutative algebra, generalizing constructions we saw in Lectures 15, $\S$17, and $\S$19. Let $\Gamma_{\omega_1}, \ldots, \Gamma_{\omega_n}$ be the irreducible representations corresponding to the fundamental weights $\omega_1, \ldots, \omega_n$. Let

$$A^* = \mathrm{Sym}^*(\Gamma_{\omega_1} \oplus \cdots \oplus \Gamma_{\omega_n}).$$

This is a commutative graded algebra, the direct sum of pieces

$$A^{\mathbf{a}} = \bigoplus_{a_1, \ldots, a_n} \mathrm{Sym}^{a_1}(\Gamma_{\omega_1}) \otimes \cdots \otimes \mathrm{Sym}^{a_n}(\Gamma_{\omega_n}),$$

where $\mathbf{a} = (a_1, \ldots, a_n)$ is an $n$-tuple of non-negative integers. Then $A^{\mathbf{a}}$ is the direct sum of the irreducible representation $\Gamma_\lambda$ whose highest weight is $\lambda = \sum a_i \omega_i$, and a sum $J^{\mathbf{a}}$ of representations whose highest weight is strictly smaller. As before, weight considerations show that $J^* = \bigoplus_{\mathbf{a}} J^{\mathbf{a}}$ is an ideal in $A^*$, so the quotient

$$A^*/J^* = \bigoplus_\lambda \Gamma_\lambda$$

is the direct sum of all the irreducible representations. The product

$$\Gamma_\lambda \otimes \Gamma_\mu \to \Gamma_{\lambda+\mu}$$

in this ring is often called **Cartan multiplication**; note that the fact that $\Gamma_{\lambda+\mu}$ occurs once in the tensor product determines such a projection, but only up to multiplication by a scalar.

Using ideas of $\S$25.1, it is possible to give generators for the ideal $J^*$. If $C$ is the Casimir operator, we know that $C$ acts on all representations and is multiplication by the constant $c_\lambda = (\lambda, \lambda) + (2\lambda, \rho)$ on the irreducible representation with highest weight $\lambda$. Therefore, if $\lambda = \sum a_i \omega_i$, the endomorphism $C - c_\lambda I$ of $A^*$ vanishes on the factor $\Gamma_\lambda$, and on each of the representations $\Gamma_\mu$ of lower weight $\mu$ it is multiplication by $c_\mu - c_\lambda \neq 0$ [cf. (25.2)]. It follows that

$$J^{\mathbf{a}} = \mathrm{Image}(C - c_\lambda I\colon A^{\mathbf{a}} \to A^{\mathbf{a}}).$$

From this exercise follows a theorem of Kostant: $J^*$ is generated by the elements

$$\sum (U_i(v) \cdot U_i'(w) + U_i'(v) \cdot U_i(w)) - 2(\alpha, \beta)\, v \cdot w$$

for $v \in \Gamma_\alpha$, $w \in \Gamma_\beta$, with $\alpha$ and $\beta$ fundamental roots. For the classical Lie algebras, this formula can be used to find concrete realizations of the ring.

---

## Lecture 26: Real Lie Algebras and Lie Groups

In this lecture we indicate how to complete the last step in the process outlined at the beginning of Part II: to take our knowledge of the classification and representation theory of complex algebras and groups and deduce the corresponding statements in the real case. We do this in the first section, giving a list of the simple classical real Lie algebras and saying a few words about the corresponding groups and their (complex) representations. The existence of a compact group whose Lie algebra has as complexification a given semisimple complex Lie algebra makes it possible to give another (indeed, the original) way to prove the Weyl character formula; we sketch this in $\S$26.2. Finally, we can ask in regard to real Lie groups $G$ a question analogous to one asked for the representations of finite groups in $\S$3.5: which of the complex representations $V$ of $G$ actually come from real ones. We answer this in the most commonly encountered cases in $\S$26.3.

### 26.1 Classification of Real Simple Lie Algebras and Groups

Having described the semisimple complex Lie algebras, we now address the analogous problem for real Lie algebras. Since the complexification $\mathfrak{g}_0 \otimes_\mathbb{R} \mathbb{C}$ of a semisimple real Lie algebra $\mathfrak{g}_0$ is a semisimple complex Lie algebra and we have classified those, we are reduced to the problem of describing the *real forms* of the complex semisimple Lie algebras: that is, for a given complex Lie algebra $\mathfrak{g}$, finding all real Lie algebras $\mathfrak{g}_0$ with

$$\mathfrak{g}_0 \otimes_\mathbb{R} \mathbb{C} \cong \mathfrak{g}.$$

We saw many of the real forms of the classical complex Lie groups and algebras back in Lectures 7 and 8. We will indicate one way to approach the question systematically, but will only include sketches of proofs.

#### Real Forms of $\mathfrak{sl}_2\mathbb{C}$

To get the idea of what to expect, suppose $\mathfrak{g}_0$ is any real Lie subalgebra of $\mathfrak{sl}_2\mathbb{C}$, with $\mathfrak{g}_0 \otimes_\mathbb{R} \mathbb{C} = \mathfrak{sl}_2\mathbb{C}$. The natural thing to do is to try to carry out our analysis of semisimple Lie algebras for the real Lie algebra $\mathfrak{g}_0$: that is, find an element $H \in \mathfrak{g}_0$ such that $\mathrm{ad}(H)$ acts semisimply on $\mathfrak{g}_0$, decompose $\mathfrak{g}_0$ into eigenspaces, and so on. Since the subset of $\mathfrak{sl}_2\mathbb{C}$ of non-semisimple matrices is a proper algebraic subvariety, it cannot contain the real subspace $\mathfrak{g}_0 \subset \mathfrak{sl}_2\mathbb{C}$, so we can certainly find a semisimple $H \in \mathfrak{g}_0$.

The next thing is to consider the eigenspaces of $\mathrm{ad}(H)$ acting on $\mathfrak{g}$. Of course, $\mathrm{ad}(H)$ has one eigenvalue 0, corresponding to the eigenspace $\mathfrak{h}_0 = \mathbb{R} \cdot H$ spanned by $H$. The remaining two eigenvalues must then sum to zero, which leaves just two possibilities:

**(i)** $\mathrm{ad}(H)$ has eigenvalues $\lambda$ and $-\lambda$, for $\lambda$ a nonzero real number; multiplying $H$ by a real scalar, we can take $\lambda = 2$. In this case we obtain a decomposition of the vector space $\mathfrak{g}_0$ into one-dimensional eigenspaces

$$\mathfrak{g}_0 = \mathfrak{h}_0 \oplus \mathfrak{g}_2 \oplus \mathfrak{g}_{-2}.$$

We can then choose $X \in \mathfrak{g}_2$ and $Y \in \mathfrak{g}_{-2}$; the standard argument then shows that the bracket $[X, Y]$ is a nonzero multiple of $H$, which we may take to be 1 by rechoosing $X$ and $Y$. We thus have the real form $\mathfrak{sl}_2\mathbb{R}$, with the basis

$$H = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, \quad X = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}.$$

**(ii)** $\mathrm{ad}(H)$ has eigenvalues $i\lambda$ and $-i\lambda$ for $\lambda$ some nonzero real number; again, adjusting $H$ by a real scalar, we may take $\lambda = 1$. In this case, of course, there are no real eigenvectors for the action of $\mathrm{ad}(H)$ on $\mathfrak{g}_0$; but we can decompose $\mathfrak{g}_0$ into the direct sum of $\mathfrak{h}_0$ and the two-dimensional subspace $\mathfrak{g}_{\lbrace i, -i\rbrace}$ corresponding to the pair of eigenvalues $i$ and $-i$. We may then choose a basis $B$ and $C$ for $\mathfrak{g}_{\lbrace i, -i\rbrace}$ with

$$[H, B] = C \quad \text{and} \quad [H, C] = -B.$$

The commutator $[B, C]$ will then be a nonzero multiple of $H$, which we may take to be either $H$ or $-H$. In the latter case, we see that $\mathfrak{g}_0$ is isomorphic to $\mathfrak{sl}_2\mathbb{R}$ again. If the commutator $[B, C] = H$, we do get a new example: $\mathfrak{g}_0$ is in this case isomorphic to the algebra

$$\mathfrak{su}_2 = \lbrace A \colon {}^t\!\bar{A} = -A \text{ and } \mathrm{trace}(A) = 0\rbrace \subset \mathfrak{sl}_2\mathbb{C},$$

which has as basis

$$H = \begin{pmatrix} i/2 & 0 \\ 0 & -i/2 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \text{and} \quad C = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.$$

This completes our analysis of the real forms of $\mathfrak{sl}_2\mathbb{C}$.

#### Cartan Subalgebras of Real Forms

In the general case, we can try to apply a similar analysis, and indeed at least one aspect generalizes: given a real form $\mathfrak{g}_0 \subset \mathfrak{g}$ of the complex semisimple Lie algebra $\mathfrak{g}$, we can find a real subalgebra $\mathfrak{h}_0 \subset \mathfrak{g}_0$ such that $\mathfrak{h} = \mathfrak{h}_0 \otimes \mathbb{C}$ is a Cartan subalgebra of $\mathfrak{g}$; this is called a **Cartan subalgebra** of $\mathfrak{g}_0$. There is a further complication in the case of Lie algebras of rank 2 or more: the values on $\mathfrak{h}_0$ of a root $\alpha \in R$ of $\mathfrak{g}$ need not be either all real or all purely imaginary. We thus need to consider the root spaces $\mathfrak{g}_\alpha$, $\mathfrak{g}_{\bar{\alpha}}$, and $\mathfrak{g}_{-\bar{\alpha}}$, and the subalgebra they generate, at the same time. Moreover, whether the values of the roots $\alpha \in R$ of $\mathfrak{g}$ on the real subspace $\mathfrak{h}_0$ are real, purely imaginary, or neither will in general depend on the choice of $\mathfrak{h}_0$.

#### The Classification Table

It turns out to be enough to work out the complexifications $\mathfrak{g}_0 \otimes_\mathbb{R} \mathbb{C} = \mathfrak{g}_0 \oplus i \cdot \mathfrak{g}_0$ of the real Lie algebras $\mathfrak{g}_0$ we know. The list is:

| Real Lie algebra | Complexification |
| --- | --- |
| $\mathfrak{sl}_n\mathbb{R}$ | $\mathfrak{sl}_n\mathbb{C}$ |
| $\mathfrak{sl}_n\mathbb{C}$ | $\mathfrak{sl}_n\mathbb{C} \times \mathfrak{sl}_n\mathbb{C}$ |
| $\mathfrak{sl}_n\mathbb{H} = \mathfrak{gl}_n\mathbb{H}/\mathbb{R}$ | $\mathfrak{sl}_{2n}\mathbb{C}$ |
| $\mathfrak{so}_{p,q}\mathbb{R}$ | $\mathfrak{so}_{p+q}\mathbb{C}$ |
| $\mathfrak{so}_n\mathbb{C}$ | $\mathfrak{so}_n\mathbb{C} \times \mathfrak{so}_n\mathbb{C}$ |
| $\mathfrak{sp}_{2n}\mathbb{R}$ | $\mathfrak{sp}_{2n}\mathbb{C}$ |
| $\mathfrak{sp}_{2n}\mathbb{C}$ | $\mathfrak{sp}_{2n}\mathbb{C} \times \mathfrak{sp}_{2n}\mathbb{C}$ |
| $\mathfrak{su}_{p,q}$ | $\mathfrak{sl}_{p+q}\mathbb{C}$ |
| $\mathfrak{u}_{p,q}\mathbb{H}$ | $\mathfrak{sp}_{2(p+q)}\mathbb{C}$ |
| $\mathfrak{u}_n^*\mathbb{H}$ | $\mathfrak{so}_{2n}\mathbb{C}$ |

The last two in the left-hand column are the Lie algebras of the groups $U_{p,q}\mathbb{H}$ and $U_n^*\mathbb{H}$ of automorphisms of a quaternionic vector space preserving a Hermitian form with signature $(p, q)$, and a skew-symmetric Hermitian form, respectively.

The theorem, which also goes back to Cartan, is that *this includes the complete list of simple real Lie algebras associated to the classical complex types* ($A_n$)--($D_n$). In fact, there are an additional 17 simple real Lie algebras associated with the five exceptional Lie algebras. The proof of this theorem is rather long, and we refer to the literature (cf. [H-S], [Hel], [Ar]) for it.

#### Split Forms and Compact Forms

Rather than try to classify in general the real forms $\mathfrak{g}_0$ of a semisimple Lie algebra $\mathfrak{g}$, we would like to focus here on two particular forms that are possessed by every semisimple Lie algebra and that are by far the most commonly dealt with in practice: the **split form** and the **compact form**.

These represent the two extremes of behavior of the decomposition $\mathfrak{g} = \mathfrak{h} \oplus (\bigoplus \mathfrak{g}_\alpha)$ with respect to the real subalgebra $\mathfrak{g}_0 \subset \mathfrak{g}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Split Form)</span></p>

The **split form** of $\mathfrak{g}$ is a form $\mathfrak{g}_0$ such that there exists a Cartan subalgebra $\mathfrak{h}_0 \subset \mathfrak{g}_0$ (that is, a subalgebra whose complexification $\mathfrak{h} = \mathfrak{h}_0 \otimes \mathbb{C} \subset \mathfrak{g}_0 \otimes \mathbb{C} = \mathfrak{g}$ is a Cartan subalgebra of $\mathfrak{g}$) whose action on $\mathfrak{g}_0$ has all real eigenvalues --- i.e., such that all the roots $\alpha \in R \subset \mathfrak{h}^*$ of $\mathfrak{g}$ (with respect to the Cartan subalgebra $\mathfrak{h} = \mathfrak{h}_0 \otimes \mathbb{C} \subset \mathfrak{g}$) assume all real values on the subspace $\mathfrak{h}_0$. In this case we have a direct sum decomposition

$$\mathfrak{g}_0 = \mathfrak{h}_0 \oplus \left(\bigoplus \mathfrak{j}_\alpha\right)$$

of $\mathfrak{g}_0$ into $\mathfrak{h}_0$ and one-dimensional eigenspaces $\mathfrak{j}_\alpha$ for the action of $\mathfrak{h}_0$ (each $\mathfrak{j}_\alpha$ will just be the intersection of the root space $\mathfrak{g}_\alpha \subset \mathfrak{g}$ with $\mathfrak{g}_0$); each pair $\mathfrak{j}_\alpha$ and $\mathfrak{j}_{-\alpha}$ will generate a subalgebra isomorphic to $\mathfrak{sl}_2\mathbb{R}$. This uniquely characterizes the real form $\mathfrak{g}_0$ of $\mathfrak{g}$; it is sometimes called the **natural real form**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Compact Form)</span></p>

By contrast, in the **compact form** all the roots $\alpha \in R \subset \mathfrak{h}^*$ of $\mathfrak{g}$ (with respect to the Cartan subalgebra $\mathfrak{h} = \mathfrak{h}_0 \otimes \mathbb{C} \subset \mathfrak{g}$) assume all purely imaginary values on the subspace $\mathfrak{h}_0$. We accordingly have a direct sum decomposition

$$\mathfrak{g}_0 = \mathfrak{h}_0 \oplus \left(\bigoplus \mathfrak{l}_\alpha\right)$$

of $\mathfrak{g}_0$ into $\mathfrak{h}_0$ and two-dimensional spaces on which $\mathfrak{h}_0$ acts by rotation (each $\mathfrak{l}_\alpha$ will just be the intersection of the root space $\mathfrak{g}_\alpha \oplus \mathfrak{g}_{-\alpha}$ with $\mathfrak{g}_0$); each $\mathfrak{l}_\alpha$ will generate a subalgebra isomorphic to $\mathfrak{su}_2$.

</div>

The existence of the split form of a semisimple complex Lie algebra was already established in Lecture 21: one way to construct a real --- even rational --- form $\mathfrak{g}_0$ of a semisimple Lie algebra $\mathfrak{g}$ is by starting with any generator $X_{\alpha_i}$ for the root space for each positive simple root $\alpha_i$, completing it to standard basis $X_{\alpha_i}$, $Y_{\alpha_i}$, and $H_i = [X_{\alpha_i}, Y_{\alpha_i}]$ for the corresponding $\mathfrak{s}_{\alpha_i} = \mathfrak{sl}_2\mathbb{C}$, and taking $\mathfrak{g}_0$ to be the real subalgebra generated by these elements. The algebra $\mathfrak{g}_0$ is determined up to isomorphism; it is the only real form of $\mathfrak{g}$ that has a Cartan subalgebra $\mathfrak{h}_0$ acting on $\mathfrak{g}_0$ with all real eigenvalues.

The split forms for the classical complex simple Lie algebras are:

| Complex simple Lie algebra | Split form |
| --- | --- |
| $\mathfrak{sl}_{n+1}\mathbb{C}$ | $\mathfrak{sl}_{n+1}\mathbb{R}$ |
| $\mathfrak{so}_{2n+1}\mathbb{C}$ | $\mathfrak{so}_{n+1,n}$ |
| $\mathfrak{sp}_{2n}\mathbb{C}$ | $\mathfrak{sp}_{2n}\mathbb{R}$ |
| $\mathfrak{so}_{2n}\mathbb{C}$ | $\mathfrak{so}_{n,n}$ |

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(26.4 — Characterization of the Compact Form)</span></p>

Suppose $\mathfrak{g}$ is any complex semisimple Lie algebra and $\mathfrak{g}_0 \subset \mathfrak{g}$ a real form of $\mathfrak{g}$. Let $\mathfrak{h}_0$ be a Cartan subalgebra of $\mathfrak{g}_0$, $\mathfrak{h} = \mathfrak{h}_0 \otimes \mathbb{C}$ the corresponding Cartan subalgebra of $\mathfrak{g}$. The following are equivalent:

1. Each root $\alpha \in R \subset \mathfrak{h}^*$ of $\mathfrak{g}$ assumes purely imaginary values on $\mathfrak{h}_0$, and for each root $\alpha$ the subalgebra of $\mathfrak{g}_0$ generated by the intersection $\mathfrak{l}_\alpha$ of $(\mathfrak{g}_\alpha \oplus \mathfrak{g}_{-\alpha})$ with $\mathfrak{g}_0$ is isomorphic to $\mathfrak{su}_2$;
2. The restriction to $\mathfrak{g}_0$ of the Killing form of $\mathfrak{g}$ is negative definite;
3. The real Lie group $G_0$ with Lie algebra $\mathfrak{g}_0$ is compact.

</div>

**Proof sketch.** (i) $\Rightarrow$ (ii): The value of the Killing form on $H \in \mathfrak{h}_0$ is visibly

$$B(H, H) = \sum (\alpha(H))^2 < 0.$$

Next, the subspaces $\mathfrak{l}_\alpha$ are orthogonal to one another with respect to $B$, so it remains only to verify $B(Z, Z) < 0$ for a general member $Z \in \mathfrak{l}_\alpha$. To do this, let $X$ and $Y$ be generators of $\mathfrak{g}_\alpha$ and $\mathfrak{g}_{-\alpha} \subset \mathfrak{g}$ respectively, chosen so as to form, together with their commutator $H = [X, Y]$ a standard basis for $\mathfrak{sl}_2\mathbb{C}$. By the analysis of real forms of $\mathfrak{sl}_2\mathbb{C}$ above, we may take as generators of the algebra generated by $\mathfrak{l}_\alpha$ the elements $iH$, $U = X - Y$ and $V = iX + iY$. If we set $Z = aU + bV = (a + ib) \cdot X + (-a + ib) \cdot Y$, then

$$\mathrm{trace}(\mathrm{ad}(Z) \circ \mathrm{ad}(Z)) = -2 \cdot (a^2 + b^2) \cdot \mathrm{trace}(\mathrm{ad}(X) \circ \mathrm{ad}(Y)). \tag{26.5}$$

By direct examination, in the representation $\mathrm{Sym}^n V$ of $\mathfrak{sl}_2\mathbb{C}$, $\mathrm{ad}(X) \circ \mathrm{ad}(Y)$ acts by multiplication by $(n - \lambda)(n + \lambda - 2)/4 \ge 0$ on the $\lambda$-eigenspace for $H$, from which we deduce that the right-hand side of (26.5) is negative.

(ii) $\Rightarrow$ (iii): The adjoint form $G_0$ is the connected component of the identity of the adjoint group of $\mathfrak{g}$, and it acts faithfully on the real vector space $\mathfrak{g}_0$, preserving the bilinear form $B$. If $B$ is negative definite it follows that $G_0$ is a closed subgroup of the orthogonal group $\mathrm{SO}_m\mathbb{R}$, which is compact.

(iii) $\Rightarrow$ (i): If $G_0$ is compact, by averaging we can construct a positive definite inner product on $\mathfrak{g}_0$ invariant under the action of $G_0$. For any $X$ in $\mathfrak{g}_0$, $\mathrm{ad}(X)$ is represented by a skew-symmetric matrix $A = (a_{i,j})$ with respect to an orthonormal basis of $\mathfrak{g}_0$ (cf. (14.23)), so $B(X, X) = \mathrm{Tr}(A \circ A) = -\sum a_{i,j}^2 \le 0$. In particular, the eigenvalues of $\mathrm{ad}(X)$ must be purely imaginary. Therefore $\alpha(\mathfrak{h}_0) \subset i\mathbb{R}$ and $\bar{\alpha} = -\alpha$ for any root $\alpha$, from which (i) follows. $\square$

We now claim that *every semisimple complex Lie algebra has a unique compact form*. To see this we need an algebraic notion which is, in fact, crucial to the classification theorem mentioned above: that of **conjugate linear involution**. If $\mathfrak{g} = \mathfrak{g}_0 \otimes_\mathbb{R} \mathbb{C}$ is the complexification of a real Lie algebra $\mathfrak{g}_0$, there is a map $\sigma\colon \mathfrak{g} \to \mathfrak{g}$ which takes $x \otimes z$ to $x \otimes \bar{z}$ for $x \in \mathfrak{g}_0$ and $z \in \mathbb{C}$; it is conjugate linear, preserves Lie brackets, and $\sigma^2$ is the identity. The real algebra $\mathfrak{g}_0$ is the fixed subalgebra of $\sigma$, and conversely, given such a conjugate linear involution $\sigma$ of a complex Lie algebra $\mathfrak{g}$, its fixed subalgebra $\mathfrak{g}^\sigma$ is a real form of $\mathfrak{g}$.

To prove the claim, we start with the split, or natural form, as constructed in Lecture 21 and referred to above. With a basis for $\mathfrak{g}$ chosen as in this construction, it is not hard to show that there is a unique Lie algebra automorphism $\varphi$ of $\mathfrak{g}$ that takes each element of $\mathfrak{h}$ to its negative and takes each $X_\alpha$ to $Y_\alpha$ (this follows from Claim 21.25). This automorphism $\varphi$ is a complex linear involution which preserves the real subalgebra $\mathfrak{g}_0$. This automorphism commutes with the associated conjugate linear $\sigma$. The composite $\sigma\varphi = \varphi\sigma$ is a conjugate linear involution, from which it follows that its fixed part $\mathfrak{g}_c = \mathfrak{g}^{\sigma\varphi}$ is another real form of $\mathfrak{g}$. This has Cartan subalgebra $\mathfrak{h}_c = \mathfrak{h}^{\sigma\varphi} = i \cdot \mathfrak{h}_0$. We have seen that the restriction of the Killing form to $\mathfrak{h}_0$ is positive definite. It follows that its restriction to $\mathfrak{h}_c$ is negative definite, and hence that $\mathfrak{g}_c$ is a compact form of $\mathfrak{g}$.

We may see directly from this construction that

$$\mathfrak{g}_c = \mathfrak{h}_c \oplus \bigoplus_{\alpha \in R^+} \mathfrak{l}_\alpha,$$

where $\mathfrak{l}_\alpha = (\mathfrak{g}_\alpha \oplus \mathfrak{g}_{-\alpha})^{\sigma\varphi}$ is a real plane with $\mathfrak{l}_\alpha \otimes_\mathbb{R} \mathbb{C} = \mathfrak{g}_\alpha \oplus \mathfrak{g}_{-\alpha}$ and $[\mathfrak{h}_c, \mathfrak{l}_\alpha] \subset \mathfrak{l}_\alpha$.

#### Cartan Decomposition

Starting with a real form $\mathfrak{g}_0$ of $\mathfrak{g}$ with associated conjugation $\sigma$, one can always find a compact form $\mathfrak{g}_c$ of $\mathfrak{g}$ such that $\sigma(\mathfrak{g}_c) = \mathfrak{g}_c$, and such that

$$\mathfrak{g}_0 = \mathfrak{t} \oplus \mathfrak{p},$$

where $\mathfrak{t} = \mathfrak{g}_0 \cap \mathfrak{g}_c$, and $\mathfrak{p} = \mathfrak{g}_0 \cap (i \cdot \mathfrak{g}_c)$. Such a decomposition is called a **Cartan decomposition** of $\mathfrak{g}_0$. It is unique up to inner automorphism.

#### Low-Dimensional Isomorphisms

Naturally, the various special isomorphisms between complex semisimple Lie algebras ($\mathfrak{sl}_2\mathbb{C} \cong \mathfrak{so}_3\mathbb{C} \cong \mathfrak{sp}_2\mathbb{C}$, etc.) give rise to special isomorphisms among their real forms. For example, we have already seen that

$$\mathfrak{sl}_2\mathbb{R} \cong \mathfrak{su}_{1,1} \cong \mathfrak{so}_{2,1} \cong \mathfrak{sp}_2\mathbb{R},$$

while

$$\mathfrak{su}_2 \cong \mathfrak{so}_3\mathbb{R} \cong \mathfrak{sl}_1\mathbb{H} \cong \mathfrak{u}_1\mathbb{H}.$$

Similarly, each of the remaining three special isomorphisms of complex semisimple Lie algebras gives rise to isomorphisms between their real forms, as follows:

**(i)** $\mathfrak{so}_4\mathbb{C} \cong \mathfrak{sl}_2\mathbb{C} \times \mathfrak{sl}_2\mathbb{C}$
- compact forms: $\mathfrak{so}_4\mathbb{R} \cong \mathfrak{su}_2 \times \mathfrak{su}_2$
- split forms: $\mathfrak{so}_{2,2} \cong \mathfrak{sl}_2\mathbb{R} \times \mathfrak{sl}_2\mathbb{R}$
- others: $\mathfrak{so}_{3,1} \cong \mathfrak{sl}_2\mathbb{C}$, $\mathfrak{u}_2^*\mathbb{H} \cong \mathfrak{su}_2 \times \mathfrak{sl}_2\mathbb{R}$.

**(ii)** $\mathfrak{sp}_4\mathbb{C} \cong \mathfrak{so}_5\mathbb{C}$
- compact forms: $\mathfrak{u}_2\mathbb{H} \cong \mathfrak{so}_5\mathbb{R}$
- split forms: $\mathfrak{sp}_4\mathbb{R} \cong \mathfrak{so}_{3,2}$
- other: $\mathfrak{u}_{1,1}\mathbb{H} \cong \mathfrak{so}_{4,1}$.

**(iii)** $\mathfrak{sl}_4\mathbb{C} \cong \mathfrak{so}_6\mathbb{C}$
- compact forms: $\mathfrak{su}_4 \cong \mathfrak{so}_6\mathbb{R}$
- split forms: $\mathfrak{sl}_4\mathbb{R} \cong \mathfrak{so}_{3,3}$
- others: $\mathfrak{su}_{2,2} \cong \mathfrak{so}_{4,2}$; $\mathfrak{su}_{3,1} \cong \mathfrak{u}_3^*\mathbb{H}$; $\mathfrak{sl}_2\mathbb{H} \cong \mathfrak{so}_{5,1}$.

In addition, the extra automorphism of $\mathfrak{so}_8\mathbb{C}$ coming from triality gives rise to an isomorphism $\mathfrak{u}_4^*\mathbb{H} \cong \mathfrak{so}_{6,2}$.

#### Real Groups

We turn now to the problem of describing the real Lie groups with these Lie algebras. Let $G$ be the adjoint form of the semisimple complex Lie algebra $\mathfrak{g}$. If $\mathfrak{g}_0$ is a real form of $\mathfrak{g}$, the associated conjugate linear involution $\sigma$ of $\mathfrak{g}$ that fixes $\mathfrak{g}_0$ lifts to an involution $\bar{\sigma}$ of $G$. (This follows from the functorial nature of the adjoint form, noting that $G$ is regarded now as a real Lie group.) The fixed points $G^{\bar{\sigma}}$ of this involution then form a closed subgroup of $G$; its connected component of the identity $G_0$ is a real Lie group whose Lie algebra is $\mathfrak{g}_0$. $G$ is called the **complexification** of $G_0$.

We have seen in $\S$23.1 that if $\Gamma = \Gamma_\mathfrak{w}$ is the lattice of those elements in $\mathfrak{h}$ on which all roots take integral values, then $2\pi i\Gamma$ is the kernel of the exponential mapping $\exp\colon \mathfrak{h} \to G$ to the adjoint form. If $\mathfrak{h}_0$ is a Cartan subalgebra of $\mathfrak{g}_0$, $T = \exp(\mathfrak{h}_0)$ will be compact precisely when the intersection of $\mathfrak{h}_0$ with the kernel is a lattice of maximal rank. In this case, $T$ will be a product of $n$ copies of the circle $S^1$, $n = \dim(\mathfrak{h})$, and, since the Killing form on $\mathfrak{h}_0$ is negative definite, the corresponding real group $G_0$ will also be compact. Such a $G_0$ will be a maximal compact subgroup of $G$.

When $G_0 \subset G$ is a maximal compact subgroup, they have the same irreducible complex representations. Indeed, for any complex group $G'$, each complex homomorphism from $G$ to $G'$ is the extension of a unique real homomorphism from $G_0$ to $G'$. This follows from the corresponding fact for Lie algebras and the fact that $G_0$ and $G$ have the same fundamental group. This is another general fact, which implies the finiteness of the fundamental group of $G_0$; we omit the proof, noting only that it can be seen directly in the classical cases.

It is another general fact that any compact (connected) Lie group is a quotient

$$(G_1 \times G_2 \times \cdots \times G_r \times T)/Z,$$

where the $G_i$ are simple compact Lie groups, $T \cong (S^1)^k$ is a torus, and $Z$ is a discrete subgroup of the center. In particular, its Lie algebra is the direct sum of a semisimple Lie algebra and an abelian Lie algebra. This provides another reason why the classification of irreducible representations in the real compact case and the semisimple complex case are essentially the same.

#### Representations of Real Lie Algebras

Finally, we should say a word here about the irreducible representations (always here in complex vector spaces!) of simple real Lie algebras. In some cases these are easily described in terms of the complex case: for example, the irreducible representations of $\mathfrak{su}_m$ or $\mathfrak{sl}_m\mathbb{R}$ are the same as those for $\mathfrak{sl}_m\mathbb{C}$, i.e., they are the restrictions of the irreducible representations $\Gamma_\lambda = \mathbb{S}_\lambda\mathbb{C}^m$ corresponding to partitions or Young diagrams $\lambda$. This is the situation in general whenever the complexification $\mathfrak{g} = \mathfrak{g}_0 \otimes \mathbb{C}$ of the real Lie algebra $\mathfrak{g}_0$ is still simple: the representations of $\mathfrak{g}_0$ on complex vector spaces are exactly the representations of $\mathfrak{g}$.

The situation is slightly different when we have a simple real Lie algebra whose complexification is not simple: for example, the irreducible representations of $\mathfrak{sl}_m\mathbb{C}$, regarded as a real Lie algebra, are of the form $\Gamma_\lambda \otimes \bar{\Gamma}_\mu$, where $\bar{\Gamma}_\mu$ is the conjugate representation of $\Gamma_\mu$. The situation in general is expressed in the following:

If $\mathfrak{g}_0$ is a simple real Lie algebra whose complexification $\mathfrak{g}$ is simple, its irreducible representations are the restrictions of (uniquely determined) irreducible representations of $\mathfrak{g}$. If $\mathfrak{g}_0$ is the underlying real algebra of a simple complex Lie algebra, the irreducible representations of $\mathfrak{g}_0$ are of the form $V \otimes \bar{W}$, where $V$ and $W$ are (uniquely determined) irreducible representations of the complex Lie algebra.

### 26.2 Second Proof of Weyl's Character Formula

The title of this section is perhaps inaccurate: what we will give here is actually a sketch of the first proof of the Weyl character formula. Weyl, in his original proof, used what he called the "unitarian trick," which is to say he introduces the compact form of a given semisimple Lie algebra and uses integration on the corresponding compact group $G$. (This trick was already described in $\S$9.3, in the context of proving complete reducibility of representations of a semisimple algebra.)

Indeed, the main reason for including this section (which is, after all, logically unnecessary) is to acquaint the reader with the "classical" treatment of Lie groups via their compact forms. This treatment follows very much the same lines as the representation theory of finite groups. To begin with, we replace the average $(1/\lvert G\rvert)\sum_{g \in G} f(g)$ by the integral $\int_G f(g)\, d\mu$, the volume element $d\mu$ chosen to be translation invariant and such that $\int_G d\mu = 1$. If $\rho\colon G \to \mathrm{Aut}(V)$ is a finite-dimensional representation, with character

$$\chi_V(g) = \mathrm{Trace}(\rho(g)),$$

then $\int_G \rho(g)\, d\mu \in \mathrm{Hom}(V, V)$ is idempotent, and it is the projection onto the invariant subspace $V^G$. So $\int_G \chi_V(g)\, d\mu = \dim(V^G)$. Applied to $\mathrm{Hom}(V, W)$ as before, since $\chi_{\mathrm{Hom}(V, W)} = \bar{\chi}_V \chi_W$, it follows that

$$\int_G \bar{\chi}_V \chi_W\, d\mu = \dim(\mathrm{Hom}_G(V, W)).$$

So if $V$ and $W$ are irreducible,

$$\int_G \bar{\chi}_V \chi_W\, d\mu = \begin{cases} 1 & \text{if } V \cong W \\ 0 & \text{otherwise}. \end{cases}$$

Up to now, everything is completely analogous to the case of finite groups, and is proved in exactly the same way. The last general fact, analogous to the basic Proposition 2.30, is harder in the compact case:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Peter--Weyl)</span></p>

The characters of irreducible representations span a dense subspace of the space of continuous class functions.

</div>

It is, moreover, the case that the coordinate functions of the irreducible matrix representations span a dense subspace of all continuous (or $L^2$) functions on $G$. Given the fundamental role that (2.30) played in the analysis of representations of finite groups, it is not surprising that the Peter--Weyl theorem is the cornerstone of most treatments of compact groups, even though it has played no role so far in this book.

#### Weyl's Integration Formula and Second Proof of WCF

In this compact setting, $G$ will denote a fixed compact group, whose Lie algebra $\mathfrak{g}$ is a real form of the semisimple complex Lie algebra $\mathfrak{g}_C = \mathfrak{g} \otimes_\mathbb{R} \mathbb{C}$. We have seen that

$$\mathfrak{g} = \mathfrak{h} \oplus \bigoplus_{\alpha \in R^+} \mathfrak{l}_\alpha,$$

compatible with the usual decomposition $\mathfrak{g}_C = \mathfrak{h}_C \oplus \bigoplus (\mathfrak{g}_\alpha \oplus \mathfrak{g}_{-\alpha})$ when complexified. The real Cartan algebra $\mathfrak{h}$ acts by rotations on the planes $\mathfrak{l}_\alpha$.

Now let $T = \exp(\mathfrak{h}) \subset G$. As before we have chosen $\mathfrak{h}$ so that it contains the lattice $2\pi i \Gamma$ which is the kernel of the exponential map from $\mathfrak{h}_C$ to the simply-connected form of $G_C$, so $T \cong (S^1)^n$ is a compact torus.

In this compact case we can realize the Weyl group on the group level again:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Claim</span><span class="math-callout__name">(26.15)</span></p>

$N(T)/T \cong \mathfrak{W}$.

</div>

**Proof.** For each pair of roots $\alpha, -\alpha$, we have a subalgebra $\mathfrak{s}_\alpha \cong \mathfrak{sl}_2\mathbb{C} \subset \mathfrak{g}_C$, with a corresponding $\mathfrak{su}_2 \subset \mathfrak{g}$. Exponentiating gives a subgroup $\mathrm{SU}(2) \subset G$. The element $\begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$ acts by Ad, taking $H$ to $-H$, $X$ to $Y$, and $Y$ to $X$. It is in $N(T)$, and, with $B$ as in the preceding section, $\exp\!\left(\frac{1}{2}\pi i B\right) \in \mathfrak{g}$ acts by reflection in the hyperplane $\alpha^\perp \subset \mathfrak{h}$. $\square$

Note that $\mathfrak{W}$ acting on $\mathfrak{h}$ takes the lattice $2\pi i\Gamma$ to itself, so $\mathfrak{W}$ acts on $T = \mathfrak{h}/2\pi i\Gamma$ by conjugation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(26.16 — Conjugacy of Maximal Tori)</span></p>

Every element of $G$ is conjugate to an element of $T$. A general element is conjugate to $\lvert\mathfrak{W}\rvert$ such elements of $T$.

</div>

**Sketch of a proof:** Note that $G$ acts by left multiplication on the left coset space $X = G/T$. For any $z \in G$, consider the map $f_z\colon X \to X$ which takes $yT$ to $zyT$. The claim is that $f_z$ must have a fixed point, i.e., there is a $y$ such that $y^{-1}zy \in T$. Since all $f_z$ are homotopic, and $X$ is compact, the Lefschetz number of $f_z$ is the topological Euler characteristic of $X$. The first statement follows from the claim that this Euler characteristic is not zero. This is a good exercise for the classical groups; see [Bor2] for a general proof. For another proof see Remark 26.20 below.

For the second assertion, check first that any element that commutes with every element of $T$ is in $T$. Take an "irrational" element $x$ in $T$ so that its multiples are dense in $T$. Then for any $y \in G$, $yxy^{-1} \in T \Leftrightarrow yTy^{-1} = T$, and $yxy^{-1} = x \Leftrightarrow y \in T$. This gives precisely $\lvert\mathfrak{W}\rvert$ conjugates of $x$ that are in $T$. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(26.17)</span></p>

The class functions on $G$ are the $\mathfrak{W}$-invariant functions on $T$.

</div>

Suppose $G$ is a real form of the complex semisimple group $G_C$, i.e., $G$ is a real analytic closed subgroup of $G_C$, and the Lie algebra of $G_C$ is $\mathfrak{g}_C$. The characters on $G_C$ can be written $\sum n_\mu e^{2\pi i \mu}$, the sum over $\mu$ in the weight lattice $\Lambda$; they are invariant under the Weyl group $\mathfrak{W}$. From what we have seen, they can be identified with $\mathfrak{W}$-invariant functions on the torus $T$. Let us work this out for the classical groups:

**Case ($A_n$):** $G = \mathrm{SU}(n+1)$. The Lie algebra $\mathfrak{su}_{n+1}$ consists of skew-Hermitian matrices, $\mathfrak{h} = \mathfrak{su}_{n+1} \cap \mathfrak{sl}_{n+1}\mathbb{R} = \lbrace\text{imaginary diagonal matrices of trace 0}\rbrace$, and $T = \lbrace\mathrm{diag}(e^{2\pi i \vartheta_1}, \ldots, e^{2\pi i \vartheta_{n+1}})\colon \sum \vartheta_j = 0\rbrace$. In this case, the Weyl group $\mathfrak{W}$ is the symmetric group $\mathfrak{S}_{n+1}$, represented by permutation matrices. So characters on $\mathrm{SU}(n+1)$ are symmetric polynomials in $z_1, \ldots, z_{n+1}$ modulo the relation $z_1 \cdots z_{n+1} = 1$. Therefore, characters on $\mathrm{SU}(n+1)$ are symmetric polynomials in $z_1, \ldots, z_{n+1}$.

**Case ($B_n$):** $G = \mathrm{SO}(2n+1)$. $\mathfrak{h}$ consists of matrices with $n$ $2 \times 2$ blocks of the form $\begin{pmatrix} \cos(2\pi\vartheta_i) & -\sin(2\pi\vartheta_i) \\ \sin(2\pi\vartheta_i) & \cos(2\pi\vartheta_i) \end{pmatrix}$ along the diagonal, and 1 in the lower right corner. Again we see that $T = (S^1)^n$. This time $N(T)$ will have block permutations to interchange the $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ in the squares along the diagonal, with the other blocks $2 \times 2$ identity matrices, with a $\pm 1$ in the corner to make the determinant positive; these take $\vartheta_i$ to $-\vartheta_i$ for each $i$ where a block is $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. This again realizes the Weyl group as a semidirect product of $\mathfrak{S}_n$ and $(\mathbb{Z}/2)^n$. With $z_i$ identified with $e^{2\pi i \vartheta_i}$ again, we see that the characters are the symmetric polynomials in the variables $z_i + z_i^{-1}$, i.e., in $\cos(2\pi\vartheta_1), \ldots, \cos(2\pi\vartheta_n)$.

**Case ($D_n$):** $G = \mathrm{SO}(2n)$. $\mathfrak{h}$ is as in the preceding case, but with no lower corner. Since we have no corner to put a $-1$ in, there can be only an even number of blocks $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, reflecting the fact that $\mathfrak{W}$ is a semidirect product of $(\mathbb{Z}/2)^{n-1}$ and $\mathfrak{S}_n$. This time the invariants are symmetric polynomials in the $z_i + z_i^{-1}$, and one additional $\prod_i(z_i - z_i^{-1})$.

**Case ($C_n$):** $G = \mathrm{Sp}(2n)$. $\mathfrak{h}$ consists of imaginary diagonal matrices, $T$ consists of diagonal matrices with entries $e^{2\pi i \vartheta_i}$. The Weyl group is generated by permutation matrices and diagonal matrices with entries which are 1's and quaternionic $j$'s. The invariants are symmetric polynomials in the $z_i + z_i^{-1}$.

The key to Weyl's analysis is to calculate the integral of a class function $f$ on $G$ as a suitable integral over the torus $T$. For this, consider the map

$$\pi\colon G/T \times T \to G, \qquad \pi(xT, y) = xyx^{-1}.$$

By what we said earlier, $\pi$ is a generically finite-sheeted covering, with $\lvert\mathfrak{W}\rvert$ sheets. It follows that

$$\int_G f\, d\mu = \frac{1}{\lvert\mathfrak{W}\rvert} \int_{G/T \times T} \pi^*(f)\, \pi^* d\mu.$$

Now $\pi^*(f)(xT, y) = f(y)$ since $f$ is a class function. To calculate $\pi^* d\mu$, consider the induced map on tangent spaces

$$\pi_*\colon d\pi\colon \mathfrak{g}/\mathfrak{h} \times \mathfrak{h} \to \mathfrak{g}.$$

At the point $(x_0 T, y_0) \in G/T \times T$, we want to calculate

$$\frac{d}{dt}(x_0 e^{tx} y_0 e^{ty} e^{-tx} x_0^{-1})\vert_{t=0}(x_0 y_0 x_0^{-1})^{-1},$$

which is

$$x_0(xy_0 + y_0 y - y_0 x)x_0^{-1}(x_0 y_0 x_0^{-1})^{-1} = x_0(x + y_0 y y_0^{-1} - y_0 x y_0^{-1})x_0^{-1}.$$

Now $y_0 y y_0^{-1} = y$ since $y_0 \in T$ and $y \in \mathfrak{h}$. To calculate the determinant of $\pi_*$, we can ignore the volume-preserving transformation $x_0(\,\cdot\,)x_0^{-1}$. If we identify $\mathfrak{g}/\mathfrak{h} \times \mathfrak{h}$ with $\mathfrak{g}$, the matrix becomes

$$\begin{pmatrix} I - \mathrm{Ad}(y_0) & 0 \\ 0 & I \end{pmatrix}.$$

So the determinant of $\pi_*$ is $\det(I - \mathrm{Ad}(y_0))$. Now $(\mathfrak{g}/\mathfrak{h})_C = \bigoplus \mathfrak{g}_\alpha$, and $\mathrm{Ad}(y_0)$ acts as $e^{2\pi i \alpha(y)}$ on $\mathfrak{g}_\alpha$. Hence

$$\det(\pi_*) = \prod_{\alpha \in R} (1 - e^{2\pi i \alpha}), \tag{26.18}$$

as a function on $T$ alone, independent of the factor $G/T$. This gives **Weyl's integration formula**:

$$\int_G f\, d\mu_G = \frac{1}{\lvert\mathfrak{W}\rvert} \int_T f(y) \prod_{\alpha \in R} (1 - e^{2\pi i \alpha(y)})\, d\mu_T. \tag{26.19}$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(26.20)</span></p>

The same argument gives another proof of the theorem that $G$ is covered by conjugates of $T$. By what we saw above, for a generic point $y_0 \in T$ there are exactly $\lvert\mathfrak{W}\rvert$ points in $\pi^{-1}(y_0)$, and at each of these the Jacobian determinant is the same (nonzero) number. It follows that the topological degree of the map $\pi$ is $\lvert\mathfrak{W}\rvert$, so the map must be surjective.

</div>

Now $(1 - e^{2\pi i \alpha})(1 - e^{-2\pi i \alpha}) = (e^{\pi i \alpha} - e^{-\pi i \alpha})(\bar{e^{\pi i \alpha} - e^{-\pi i \alpha}})$, so if we set

$$\Delta = \prod_{\alpha \in R^+} (e^{\pi i \alpha} - e^{-\pi i \alpha}),$$

then $\det(\pi_*) = \Delta\bar{\Delta}$. As we saw in Lemma 24.3, $\Delta = A_\rho$, where $\rho$ is half the sum of the positive roots and, for any weight $\mu$,

$$A_\mu = \sum_{W \in \mathfrak{W}} (-1)^W e^{2\pi i W(\mu)}.$$

Now we can complete the second proof of Weyl's character formula: the character of the representation with highest weight $\lambda$ is $A_{\lambda+\rho}/A_\rho$. Since we saw in $\S$24.1 that $A_{\lambda+\rho}/A_\rho$ has highest weight $\lambda$ and (see Corollary 24.6) its value at the identity is positive, it suffices to show that the integral of $\int_G \chi\bar{\chi} = 1$, where $\chi = A_{\lambda+\rho}/A_\rho$. By Weyl's integration formula,

$$\int_G \chi\bar{\chi} = \frac{1}{\lvert\mathfrak{W}\rvert} \int_T \chi\bar{\chi} \Delta\bar{\Delta} = \frac{1}{\lvert\mathfrak{W}\rvert} \int_T A_{\lambda+\rho} \bar{A}_{\lambda+\rho}$$

$$= \frac{1}{\lvert\mathfrak{W}\rvert} \int_T \sum_{W \in \mathfrak{W}} (-1)^W e^{2\pi i W(\lambda+\rho)} \cdot \sum_{W' \in \mathfrak{W}} (-1)^{W'} e^{-2\pi i W'(\lambda+\rho)} = 1,$$

which concludes the proof.

### 26.3 Real, Complex, and Quaternionic Representations

The final topic we want to take up is the classification of irreducible complex representations of semisimple Lie groups or algebras into those of real, quaternionic, or complex type. To define our terms, given a real semisimple Lie group $G_0$ or its Lie algebra $\mathfrak{g}_0$ and a representation of $G_0$ or $\mathfrak{g}_0$ on a complex vector space $V$ we say that the representation $V$ is:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Real, Quaternionic, and Complex Representations)</span></p>

- **Real**, or of *real type*, if it comes from a representation of $G_0$ or $\mathfrak{g}_0$ on a real vector space $V_0$ by extension of scalars ($V = V_0 \otimes_\mathbb{R} \mathbb{C}$); this is equivalent to saying that it has a conjugate linear endomorphism whose square is the identity.
- **Quaternionic**, or of *quaternionic type*, if it comes from a quaternionic representation by restriction of scalars, or equivalently if it has a conjugate linear endomorphism whose square is minus the identity.
- **Complex**, or of *complex type*, if it is neither of these.

(Compare with Theorem 3.37 for finite groups.)

</div>

Having completely classified the irreducible representations of the classical complex Lie algebras, and having described all the real forms of these Lie algebras, we have a clear-cut problem: to determine the type of the restriction of each representation to each real form. We will focus on the cases of the split forms (where the answer is easy) and the compact forms (where the answer is more interesting and where we have more tools to play with). We assume the complexification $\mathfrak{g}$ of $\mathfrak{g}_0$ is simple, so irreducible representations of $\mathfrak{g}_0$ are restrictions of unique irreducible representations of $\mathfrak{g}$; in particular, we have the classification of irreducible representations by dominant weights.

#### Tensor Products and Types

To begin with, the tensor products of two real, or two quaternionic, or of a pair of complex conjugate representations is always real; and exterior powers of real and quaternionic representations are equally easy to analyze, as for finite groups (see Exercise 3.43). Such tensor and exterior powers may not be irreducible, but the following criterion can often be used to describe an irreducible component of highest weight that occurs inside them:

If $W$ is a representation of a semisimple group $G$ that is real or quaternionic, and suppose $W$ has a highest weight $\lambda$ that occurs with multiplicity 1, then the irreducible representations $\Gamma_\lambda$ with highest weight $\lambda$ has the same type as $W$.

We may apply this in particular to the tensor product $\Gamma_\lambda \otimes \Gamma_\mu$ of the irreducible representations of $\mathfrak{g}$ with highest weights $\lambda$ and $\mu$; since the irreducible representation $\Gamma_{\lambda+\mu}$ with highest weight $\lambda + \mu$ appears once in this tensor product, we deduce: (i) If $\Gamma_\lambda$ and $\Gamma_\mu$ are both real or both quaternionic, then $\Gamma_{\lambda+\mu}$ is real. (ii) If $\Gamma_\lambda$ is real and $\Gamma_\mu$ is quaternionic, then $\Gamma_{\lambda+\mu}$ is quaternionic. (iii) If $\Gamma_\lambda$ and $\Gamma_\mu$ are complex and conjugate, then $\Gamma_{\lambda+\mu}$ is real.

#### Split Forms

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(26.23 — Representations of Split Forms are Real)</span></p>

Every irreducible representation of the split forms $\mathfrak{sl}_{n+1}\mathbb{R}$, $\mathfrak{so}_{n+1,n}\mathbb{R}$, $\mathfrak{sp}_{2n}\mathbb{R}$, and $\mathfrak{so}_{n,n}\mathbb{R}$ of the classical Lie algebras is real.

</div>

**Proof.** In each of these cases, the standard representation $V$ is real, from which it follows that the exterior powers $\bigwedge^k V$ are real, from which it follows that the symmetric powers $\mathrm{Sym}^{a_k}(\bigwedge^k V)$ are real. Now, in the cases of $\mathfrak{sl}_{n+1}\mathbb{R}$ and $\mathfrak{sp}_{2n}\mathbb{R}$, we have seen that the highest weights $\omega_k$ of the representations $\bigwedge^k V$ for $k = 1, \ldots, n$ form a set of fundamental weights: that is, every irreducible representation $\Gamma$ has highest weight $\sum a_k \cdot \omega_k$ for some non-negative integers $a_1, \ldots, a_n$. It follows that $\Gamma$ appears once in the tensor product

$$\mathrm{Sym}^{a_1} V \otimes \mathrm{Sym}^{a_2}(\bigwedge^2 V) \otimes \cdots \otimes \mathrm{Sym}^{a_n}(\bigwedge^n V)$$

and so is real. (Alternatively, Weyl's construction produces real representations when applied to real vector spaces.)

The only difference in the orthogonal case is that some of the exterior powers $\bigwedge^k V$ of the standard representation must be replaced in this description by the spin representation(s). That the spin representations are real follows from the construction in Lecture 20, cf. Exercise 20.23; the result in this case then follows as before. $\square$

#### The Compact Case

We turn now to the compact forms of the classical Lie algebras. In this case, the theory behaves very much like that of finite groups, discussed in Lecture 5. Specifically, any action of a compact group $G_0$ on a complex vector space $V$ preserves a nondegenerate Hermitian inner product (obtained, for example, by choosing one arbitrarily and averaging its translates under the action of $G_0$). It follows that the dual of $V$ is isomorphic to its conjugate, so that $V$ will be either real or quaternionic exactly when it is isomorphic to its dual $V^*$. (In terms of characters, this says that the character $\mathrm{Char}(V)$ is invariant under the automorphism of $\mathbb{Z}[\Lambda]$ which takes $e(\mu)$ to $e(-\mu)$; for groups, this says the character is real.) More precisely, an irreducible representation of a compact group/Lie algebra will be real (resp. quaternionic) if and only if it has an invariant nondegenerate symmetric (resp. skew-symmetric) bilinear form. In other words, the classification of an irreducible $V$ is determined by whether

$$V \otimes V = \mathrm{Sym}^2 V \oplus \bigwedge^2 V$$

contains the trivial representation, and, if so, in which factor. So determining which type a representation belongs to is a very special case of the general plethysm problem of decomposing such representations.

#### Compact Unitary Groups: $\mathfrak{su}_n$

Let $\Gamma_\lambda$ be the irreducible representation of $\mathfrak{sl}_n\mathbb{C}$ with highest weight $\lambda = \sum a_i \cdot \omega_i$, where $\omega_i = L_1 + \cdots + L_i$, $i = 1, \ldots, n-1$ are the fundamental weights of $\mathfrak{sl}_n\mathbb{C}$. The dual of $\Gamma$ will have highest weight $\sum a_{n-i} \cdot \omega_i$, so that $\Gamma$ will be real or quaternionic if and only if $a_i = a_{n-i}$ for all $i$. We now distinguish three cases:

**(i)** If $n$ is odd, then the sublattice of weights $\lambda = \sum a_i \cdot \omega_i$ with $a_i = a_{n-i}$ for all $i$ is freely generated by the sums $\omega_i + \omega_{n-i}$ for $i = 1, \ldots, (n-1)/2$. Now, $\omega_i$ is the highest weight of the exterior power $\bigwedge^i V$, so that the irreducible representation with highest weight $\omega_i + \omega_{n-i}$ will appear once in the tensor product

$$\bigwedge^i V \otimes \bigwedge^{n-i} V = (\bigwedge^i V) \otimes (\bigwedge^i V)^*,$$

which by Exercise 26.21 above is real. It follows that for any weight $\lambda = \sum a_i \cdot \omega_i$ with $a_i = a_{n-i}$ for all $i$, the irreducible representation $\Gamma_\lambda$ is real.

**(iia)** If $n = 2k$ is even, then the sublattice of weights $\lambda = \sum a_i \cdot \omega_i$ with $a_i = a_{n-i}$ for all $i$ is freely generated by the sums $\omega_i + \omega_{n-i}$ for $i = 1, \ldots, k-1$, together with the weight $\omega_k$. As before, the irreducible representations with highest weight $\omega_i + \omega_{n-i}$ are all real. Moreover, in case $n$ is divisible by 4 the representation $\bigwedge^k V$ admits a symmetric bilinear form

$$\bigwedge^k V \otimes \bigwedge^k V \to \bigwedge^{2k} V = \mathbb{C}$$

given by wedge product. It follows then as before that for any weight $\lambda = \sum a_i \cdot \omega_i$ with $a_i = a_{n-i}$ for all $i$, the irreducible representation $\Gamma_\lambda$ is real.

**(iib)** In case $n \equiv 2 \pmod{4}$, the analysis is similar to the last case except that wedge product gives a skew-symmetric bilinear pairing on $\bigwedge^k V$. The representation $\bigwedge^k V$ is thus quaternionic, and it follows that for any weight $\lambda = \sum a_i \cdot \omega_i$ with $a_i = a_{n-i}$ for all $i$, the irreducible representation $\Gamma_\lambda$ is real if $a_k$ is even, quaternionic if $a_k$ is odd.

In sum, then, we have

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(26.24 — Types for $\mathfrak{su}_n$)</span></p>

For any weight $\lambda = \sum a_i \cdot \omega_i$ of $\mathfrak{su}_n$, the irreducible representation $\Gamma_\lambda$ with highest weight $\lambda$ is: **complex** if $a_i \neq a_{n-i}$ for any $i$; **real** if $a_i = a_{n-i}$ for all $i$ and $n$ is odd, or $n = 4k$, or $n = 4k + 2$ and $a_{2k+1}$ is even; and **quaternionic** if $a_i = a_{n-i}$ for all $i$ and $n = 4k + 2$ and $a_{2k+1}$ is odd.

</div>

#### Compact Symplectic Groups: $\mathfrak{u}_n\mathbb{H}$ ($\mathfrak{sp}_{2n}\mathbb{C}$)

Next, we consider the compact form $\mathfrak{u}_n\mathbb{H}$ of $\mathfrak{sp}_{2n}\mathbb{C}$. To begin with, we note that since the restriction to $\mathfrak{u}_n\mathbb{H}$ of the standard representation of $\mathfrak{sp}_{2n}\mathbb{C}$ on $V \cong \mathbb{C}^{2n}$ is quaternionic, the exterior power $\bigwedge^k V$ is real for $k$ even and quaternionic for $k$ odd. Since the highest weights $\omega_k$ of $\bigwedge^k V$ for $k = 1, \ldots, n$ form a set of fundamental weights, this completely determines the type of the irreducible representations of $\mathfrak{u}_n\mathbb{H}$: we have

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(26.25 — Types for $\mathfrak{u}_n\mathbb{H}$)</span></p>

For any weight $\lambda = \sum a_i \cdot \omega_i$ of $\mathfrak{u}_n\mathbb{H}$, the irreducible representation $\Gamma_\lambda$ with highest weight $\lambda$ is real if $a_i$ is even for all odd $i$, and quaternionic if $a_i$ is odd for any odd $i$.

</div>

#### Odd Orthogonal Groups: $\mathfrak{so}_{2n+1}\mathbb{R}$

Next, we consider the odd orthogonal algebras. Part of this is easy: since the restriction to $\mathfrak{so}_{2n+1}\mathbb{R}$ of the standard representation $V$ of $\mathfrak{so}_{2n+1}\mathbb{C}$ is real, so are all its exterior powers; and it follows that any representation of $\mathfrak{so}_{2n+1}\mathbb{R}$ whose highest weight lies in the sublattice of index two generated by the highest weights of these exterior powers is real. It remains, then, to describe the type of the spin representation $\Gamma_\alpha$ of $\mathfrak{so}_{2n+1}\mathbb{C}$ (that is, the irreducible representation whose highest weight is one-half the highest weight of $\bigwedge^n V$). The verification is left as Exercise 26.28 below.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(26.26 — Types for $\mathfrak{so}_{2n+1}\mathbb{R}$)</span></p>

Let $\omega_i$ be the highest weight of the representation $\bigwedge^i V$ of $\mathfrak{so}_{2n+1}\mathbb{C}$. For any weight $\lambda = a_1\omega_1 + \cdots + a_{n-1}\omega_{n-1} + a_n \omega_n/2$ of $\mathfrak{so}_{2n+1}\mathbb{R}$, the irreducible representation $\Gamma_\lambda$ with highest weight $\lambda$ is real if $a_n$ is even, or if $n$ is congruent to 0 or 3 mod 4; if $a_n$ is odd and $n \equiv 1$ or $2 \pmod{4}$, then $\Gamma_\lambda$ is quaternionic.

</div>

(Note that, in each of the last two cases, the fact that every representation is either real or quaternionic follows from the observation that the Weyl group action on the Cartan subalgebra $\mathfrak{h} \subset \mathfrak{g}$ includes multiplication by $-1$.)

#### Even Orthogonal Groups: $\mathfrak{so}_{2n}\mathbb{R}$

Finally, we have the even orthogonal Lie algebras. As before, the exterior powers of the standard representation $V$ are all real, but we now have two spin representations to deal with, with highest vectors (in the notation of Lecture 19) $\alpha = (L_1 + \cdots + L_n)/2$ and $\beta = (L_1 + \cdots + L_{n-1} - L_n)/2$. The first question is whether these two are self-conjugate or conjugate to each other. In case $n$ is even, the Weyl group action on the Cartan subalgebra $\mathfrak{h} \subset \mathfrak{g}$ contains multiplication by $-1$ (the Weyl group contains the automorphism of $\mathfrak{h}^*$ reversing the sign of any even number of the basis elements $L_i$), so that $\Gamma_\alpha$ and $\Gamma_\beta$ will be isomorphic to their duals; if $n$ is odd, on the other hand, $\Gamma_\alpha$ will have $-\beta$ as a weight, so that $\Gamma_\alpha$ and $\Gamma_\beta$ will be complex representations dual to each other. We consider these cases in turn.

**(i)** Suppose first $n$ is odd, and say $\lambda$ is any weight, written as

$$\lambda = a_1 \omega_1 + \cdots + a_{n-2}\omega_{n-2} + a_{n-1}\beta + a_n \alpha.$$

If $a_{n-1} \neq a_n$, the representation $\Gamma_\lambda$ with highest weight $\lambda$ will not be isomorphic to its dual, and so will be complex. On the other hand, $\Gamma_{\alpha+\beta}$ appears once in $\Gamma_\alpha \otimes \Gamma_\beta = \mathrm{End}(\Gamma_\alpha)$, and so is real; thus, if $a_{n-1} = a_n$, the representation $\Gamma_\lambda$ will be real.

**(ii)** If, by contrast, $n$ is even then all representations of $\mathfrak{so}_{2n}\mathbb{R}$ will be either real or quaternionic. The half-spin representations $\Gamma_\alpha$ and $\Gamma_\beta$ are real if $n \equiv 0 \pmod{4}$, quaternionic if $n \equiv 2 \pmod{4}$, a fact that we leave as Exercise 26.28.

It follows that, with $\lambda$ as above, $\Gamma_\lambda$ will be real if either $n$ is divisible by 4, or if $a_{n-1} + a_n$ is even; if $n \equiv 2 \pmod{4}$ and $a_{n-1} + a_n$ is odd, $\Gamma_\lambda$ will be quaternionic.

In sum, then, we have

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(26.27 — Types for $\mathfrak{so}_{2n}\mathbb{R}$)</span></p>

The representation $\Gamma_\lambda$ of $\mathfrak{so}_{2n}\mathbb{R}$ with highest weight $\lambda = a_1\omega_1 + \cdots + a_{n-2}\omega_{n-2} + a_{n-1}\beta + a_n\alpha$ will be **complex** if $n$ is odd and $a_{n-1} \neq a_n$; it will be **quaternionic** if $n \equiv 2 \pmod{4}$ and $a_{n-1} + a_n$ is odd; and it will be **real** otherwise.

</div>

---

# Appendices

These appendices contain proofs of some of the general Lie algebra facts that were postponed during the course, as well as some results from algebra and invariant theory which were used particularly in the "Weyl construction--Schur functor" descriptions of representations.

---

## Appendix A: On Symmetric Functions

### A.1 Basic Symmetric Polynomials and Relations among Them

The vector space of homogeneous symmetric polynomials of degree $d$ in $k$ variables $x_1, \ldots, x_k$ has several important bases, usually indexed by the partitions $\lambda = (\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_k \ge 0)$ of $d$ into at most $k$ parts, or by Young diagrams with at most $k$ rows (see $\S$4.1). We list four of these bases, which are all valid for polynomials with integer coefficients, or coefficients in any commutative ring.

#### The Four Standard Bases

**1. Complete symmetric polynomials.** First we have the monomials in the complete symmetric polynomials:

$$H_\lambda = H_{\lambda_1} \cdot H_{\lambda_2} \cdots H_{\lambda_k}, \tag{A.1}$$

where $H_j$ is the $j$th **complete symmetric polynomial**, i.e., the sum of all distinct monomials of degree $j$; equivalently,

$$\prod_{i=1}^{k} \frac{1}{1 - x_i t} = \sum_{j=0}^{\infty} H_j t^j.$$

**2. Monomial symmetric polynomials.** Next are the **monomial symmetric polynomials**:

$$M_\lambda = \sum X^\alpha, \tag{A.2}$$

the sum over all distinct permutations $\alpha = (\alpha_1, \ldots, \alpha_k)$ of $(\lambda_1, \ldots, \lambda_k)$; here $X^\alpha = x_1^{\alpha_1} \cdots x_k^{\alpha_k}$.

**3. Elementary symmetric polynomials.** The third are the monomials in the elementary symmetric functions. Unlike the first two, these are parametrized by partitions $\mu$ of $d$ in integers no larger than $k$, i.e., $k \ge \mu_1 \ge \cdots \ge \mu_l \ge 0$. These are exactly the partitions that are conjugate to a partition $\lambda$ with at most $k$ parts. For such $\mu$ set

$$E_\mu = E_{\mu_1} \cdot E_{\mu_2} \cdots E_{\mu_l}, \tag{A.3}$$

where $E_j$ is the $j$th **elementary symmetric polynomial**, i.e.,

$$E_j = \sum_{i_1 < \cdots < i_j} x_{i_1} \cdots x_{i_j}, \qquad \prod_{i=1}^{k} (1 + x_i t) = \sum_{j=0}^{\infty} E_j t^j.$$

**4. Schur polynomials.** The fourth are the **Schur polynomials**, which may be the most important, although they are less often met in modern algebra courses:

$$S_\lambda = \frac{\lvert x_j^{\lambda_i + k - i}\rvert}{\lvert x_j^{k-i}\rvert} = \frac{\lvert x_j^{\lambda_i + k - i}\rvert}{\Delta}, \tag{A.4}$$

where $\Delta = \prod_{i < j}(x_i - x_j)$ is the discriminant, and $\lvert a_{i,j}\rvert$ denotes the determinant of a $k \times k$ matrix.

#### Determinantal Formulas (Giambelli's Formulas)

The first task of this appendix is to describe some relations among these symmetric polynomials. The following are known as **determinantal formulas** or the **Jacobi--Trudi identity**. The first two are sometimes called **Giambelli's formulas**, and the third is **Pieri's formula**. The proofs will be given in the next section.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.5 — Jacobi--Trudi Identity)</span></p>

$$S_\lambda = \lvert H_{\lambda_i + j - i}\rvert = \begin{vmatrix} H_{\lambda_1} & H_{\lambda_1 + 1} & \cdots & H_{\lambda_1 + k - 1} \\ H_{\lambda_2 - 1} & H_{\lambda_2} & \cdots \\ \vdots & & \ddots \\ H_{\lambda_k - k + 1} & \cdots & & H_{\lambda_k} \end{vmatrix}.$$

Note that if $\lambda_{p+1} = \cdots = \lambda_k = 0$, the determinant on the right is the same as the determinant of the upper left $p \times p$ corner.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.6 — Dual Jacobi--Trudi Identity)</span></p>

$$S_\lambda = \lvert E_{\mu_i + j - i}\rvert = \begin{vmatrix} E_{\mu_1} & E_{\mu_1 + 1} & \cdots & E_{\mu_1 + l - 1} \\ E_{\mu_2 - 1} & E_{\mu_2} & \cdots \\ \vdots & & \ddots \\ E_{\mu_l - l + 1} & \cdots & & E_{\mu_l} \end{vmatrix},$$

where $\mu = (\mu_1, \ldots, \mu_l)$ is the conjugate partition to $\lambda$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.7 — Pieri's Formula)</span></p>

The product of a Schur polynomial $S_\lambda$ by a basic Schur polynomial $S_{(m)} = H_m$ is:

$$S_\lambda S_{(m)} = \sum S_\nu,$$

the sum over all $\nu$ whose Young diagram can be obtained from that of $\lambda$ by adding a total of $m$ boxes to the rows, but with no two new boxes in the same column, i.e., those $\nu = (\nu_1, \ldots, \nu_k)$ with

$$\nu_1 \ge \lambda_1 \ge \nu_2 \ge \lambda_2 \ge \cdots \ge \nu_k \ge \lambda_k \ge 0,$$

and $\sum \nu_j = \sum \lambda_j + m$.

</div>

#### The Littlewood--Richardson Rule

One can use the Pieri and determinantal formulas to multiply any two Schur polynomials, but there is a more direct formula, which generalizes Pieri's formula. This **Littlewood--Richardson rule** gives a combinatorial formula for the coefficients $N_{\lambda\mu\nu}$ in the expansion of a product as a linear combination of Schur polynomials:

$$S_\lambda \cdot S_\mu = \sum N_{\lambda\mu\nu} S_\nu. \tag{A.8}$$

Here $\lambda$ is a partition of $d$, $\mu$ a partition of $m$, and the sum is over all partitions $\nu$ of $d + m$ (each with at most $k$ parts). The Littlewood--Richardson rule says that $N_{\lambda\mu\nu}$ *is the number of ways the Young diagram for $\lambda$ can be expanded to the Young diagram for $\nu$ by a strict $\mu$-expansion*. If $\mu = (\mu_1, \ldots, \mu_k)$, a $\mu$-expansion of a Young diagram is obtained by first adding $\mu_1$ boxes, according to Pieri's formula, and putting the integer 1 in each of these $\mu_1$ boxes; then adding similarly $\mu_2$ boxes with a 2, continuing until finally $\mu_k$ boxes are added with the integer $k$. The expansion is called **strict** if, when the integers in the boxes are listed from right to left, starting with the top row and working down, and one looks at the first $t$ entries in this list (for any $t$ between 1 and $\mu_1 + \cdots + \mu_k$), each integer $p$ between 1 and $k - 1$ occurs at least as many times as the next integer $p + 1$.

#### Kostka Numbers and Semistandard Tableaux

Formula (A.7), applied inductively, yields

$$H_\lambda = H_{\lambda_1} \cdot H_{\lambda_2} \cdots H_{\lambda_k} = \sum K_{\mu\lambda} S_\mu, \tag{A.9}$$

where $K_{\mu\lambda}$ *is the number of ways one can fill the boxes of the Young diagram of $\mu$ with $\lambda_1$ 1's, $\lambda_2$ 2's, up to $\lambda_k$ $k$'s, in such a way that the entries in each row are nondecreasing, and those in each column are strictly increasing*. Such a tableau is called a **semistandard tableau on $\mu$ of type $\lambda$**. These integers $K_{\mu\lambda}$ are all non-negative, with

$$K_{\lambda\lambda} = 1 \quad \text{and} \quad K_{\mu\lambda} = 0 \quad \text{if } \lambda > \mu, \tag{A.10}$$

i.e., if the first nonvanishing $\lambda_i - \mu_i$ is positive; in addition, $K_{\mu\lambda} = 0$ if $\lambda$ has more nonzero terms than $\mu$. The integers $K_{\mu\lambda}$ are called **Kostka numbers**.

When $\lambda = (1, 1, \ldots, 1)$, $K_{\mu(1, \ldots, 1)}$ *is the number of standard tableaux on the diagram of $\mu$*, where a **standard tableau** is a numbering of the $d$ boxes of a Young diagram by the integers 1 through $d$, increasing in both rows and columns.

#### Cauchy's Identity and the Inner Product

We need one more formula involving Schur polynomials, which comes from an identity of Cauchy. Let $y_1, \ldots, y_k$ be another set of indeterminates, and write $P(x)$ and $P(y)$ for the same polynomial $P$ expressed in terms of variables $x_1, \ldots, x_k$ and $y_1, \ldots, y_k$, respectively.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(A.12 — Cauchy's Identity)</span></p>

$$\det\!\left\lvert\frac{1}{1 - x_i y_j}\right\rvert = \frac{\Delta(x)\,\Delta(y)}{\prod_{i,j}(1 - x_i y_j)}.$$

</div>

**Proof.** By induction on $k$. Subtract the first row from each of the other rows, noting that

$$\frac{1}{1 - x_i y_j} - \frac{1}{1 - x_1 y_j} = \frac{x_i - x_1}{1 - x_1 y_j} \cdot \frac{y_j}{1 - x_i y_j}$$

and factor out common factors. Then subtract the first column from each of the other columns, using the equation

$$\frac{y_j}{1 - x_i y_j} - \frac{y_1}{1 - x_i y_1} = \frac{y_j - y_1}{1 - x_i y_1} \cdot \frac{1}{1 - x_i y_j}$$

to factor out common factors. One is left with a matrix whose first row is $(1\; 0 \ldots 0)$, and whose lower right square has the original entries. The formula follows by induction (cf. [We1, p. 202]). $\square$

Another form of Cauchy's identity is

$$\frac{1}{\prod_{i,j}(1 - x_i y_j)} = \sum_\lambda S_\lambda(x) S_\lambda(y), \tag{A.13}$$

the sum over all partitions $\lambda$ with at most $k$ terms.

Expansion of the left-hand side of (A.13) gives

$$\frac{1}{\prod_{i,j}(1 - x_i y_j)} = \prod_j \left(\sum_{m=0}^{\infty} H_m(x) y_j^m\right) = \sum_\lambda H_\lambda(x) M_\lambda(y). \tag{A.15}$$

#### The Inner Product on Symmetric Polynomials

Since the $H_\lambda$ as well as the $M_\mu$ form a basis for the symmetric polynomials, one can define a bilinear form $\langle\,,\,\rangle$ on the space of homogeneous symmetric polynomials of degree $d$ in $k$ variables, by requiring that

$$\langle H_\lambda,\, M_\mu\rangle = \delta_{\lambda,\mu}, \tag{A.16}$$

where $\delta_{\lambda,\mu}$ is 1 if $\lambda = \mu$ and 0 otherwise.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.17 — Orthonormality of Schur Polynomials)</span></p>

The **Schur polynomials form an orthonormal basis** for this pairing:

$$\langle S_\lambda,\, S_\mu\rangle = \delta_{\lambda,\mu}.$$

</div>

In particular, this implies that the pairing $\langle\,,\,\rangle$ is **symmetric**. This is easily deduced from the preceding equations. Write $S_\lambda = \sum a_{\lambda\gamma} H_\gamma = \sum b_{\gamma\lambda} M_\gamma$, for some integer matrices $a_{\lambda\gamma}$ and $b_{\gamma\lambda}$. Then

$$\langle S_\lambda, S_\mu\rangle = \sum_\gamma a_{\lambda\gamma} b_{\gamma\mu}. \tag{A.18}$$

In order that $\sum_\lambda S_\lambda(x) S_\lambda(y) = \sum_{\lambda,\gamma,\rho} a_{\lambda\gamma} H_\gamma(x) b_{\rho\lambda} M_\rho(y)$ be equal to $\sum_\gamma H_\gamma(x) M_\gamma(y)$, which it must by (A.13) and (A.15), we must have $\sum_\lambda b_{\rho\lambda} a_{\lambda\gamma} = \delta_{\rho,\gamma}$. This is equivalent to $\sum_\gamma a_{\lambda\gamma} b_{\gamma\mu} = \delta_{\lambda,\mu}$, which by (A.18) implies (A.17).

Because of this duality, formula (A.9) is equivalent to the equation

$$S_\mu = \sum_\lambda K_{\mu\lambda} M_\lambda. \tag{A.19}$$

#### The Numbers $\psi_\lambda(P)$ and $\omega_\lambda(P)$

The identities (A.9) and (A.19) for the basic symmetric polynomials allow us to relate the coefficients of $X^\lambda$ in any symmetric polynomial $P$ with the coefficients expanding $P$ as a linear combination of Schur polynomials. If $P$ is any homogeneous symmetric polynomial of degree $d$ in $k$ variables, and $\lambda$ is any partition of $d$ into at most $k$ parts, define numbers $\psi_\lambda(P)$ and $\omega_\lambda(P)$ by

$$\psi_\lambda(P) = [P]_\lambda, \tag{A.20}$$

where $[P]_\lambda$ denotes the coefficient of $X^\lambda = x_1^{\lambda_1} \cdots x_k^{\lambda_k}$ in $P$, and

$$\omega_\lambda(P) = [\Delta \cdot P]_l, \quad l = (\lambda_1 + k - 1, \lambda_2 + k - 2, \ldots, \lambda_k); \tag{A.21}$$

here $\Delta = \prod_{i < j}(x_i - x_j)$. We want to compare these two collections of numbers, as $\lambda$ varies over the partitions. The first numbers $\psi_\lambda(P)$ are the coefficients in the expression

$$P = \sum \psi_\lambda(P) M_\lambda \tag{A.22}$$

for $P$ as a linear combination of the monomial symmetric polynomials $M_\lambda$. The integers $\omega_\lambda(P)$ have a similar interpretation in terms of Schur polynomials:

$$P = \sum \omega_\lambda(P) S_\lambda. \tag{A.23}$$

Note from the definition that the coefficient of $X^l$ in $\Delta \cdot S_\lambda$ is 1, and that no other monomial with strictly decreasing exponents appears in $\Delta \cdot S_\lambda$; from this, formula (A.23) is evident. In this terminology we may rewrite (A.19) and (A.9) as

$$K_{\mu\lambda} = \psi_\lambda(S_\mu) = \text{coefficient of } X^\lambda \text{ in } S_\mu \tag{A.24}$$

and

$$K_{\mu\lambda} = \omega_\mu(H_\lambda) = [\Delta \cdot H_\lambda]_{(\lambda_1 + k - 1, \ldots, \lambda_k)}. \tag{A.25}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(A.26)</span></p>

For any symmetric polynomial $P$ of degree $d$ in $k$ variables,

$$\psi_\lambda(P) = \sum_\mu K_{\mu\lambda} \cdot \omega_\mu(P).$$

</div>

**Proof.** We have $\sum_\lambda \psi_\lambda(P) M_\lambda = P = \sum_\mu \omega_\mu(P) S_\mu = \sum_{\lambda,\mu} \omega_\mu(P) K_{\mu\lambda} M_\lambda$, and the result follows, since the $M_\lambda$ are independent. $\square$

#### Newton Polynomials

We want to apply the preceding discussion when $P$ is a product of sums of powers of the variables. Let $P_j = x_1^j + \cdots + x_k^j$, and for $\mathbf{i} = (i_1, \ldots, i_d)$, a $d$-tuple of non-negative integers with $\sum \alpha i_\alpha = d$, set

$$P^{(\mathbf{i})} = P_1^{i_1} \cdot P_2^{i_2} \cdots P_d^{i_d}.$$

These **Newton** or **power sum polynomials** form a basis for the symmetric functions with rational coefficients, but not with integer coefficients. Let

$$\omega_\lambda(\mathbf{i}) = \omega_\lambda(P^{(\mathbf{i})}). \tag{A.27}$$

Equivalently,

$$P^{(\mathbf{i})} = \sum \omega_\lambda(\mathbf{i}) S_\lambda.$$

For the proof of Frobenius's formula in Lecture 4 we need a formal lemma about these coefficients $\omega_\lambda(\mathbf{i})$:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(A.28)</span></p>

For partitions $\lambda$ and $\mu$ of $d$,

$$\sum_{\mathbf{i}} \frac{1}{1^{i_1} i_1! \cdots d^{i_d} i_d!}\, \omega_\lambda(\mathbf{i})\, \omega_\mu(\mathbf{i}) = \begin{cases} 1 & \text{if } \lambda = \mu \\ 0 & \text{otherwise}. \end{cases}$$

</div>

**Proof.** We will use Cauchy's formula (A.13). Note that

$$\log\!\left(\prod_{i,j}(1 - x_i y_j)^{-1}\right) = \sum_{j=1}^{\infty} \frac{1}{j}\, P_j(x)\, P_j(y),$$

so

$$\frac{1}{\prod_{i,j}(1 - x_i y_j)} = \prod_j \exp\!\left(\frac{1}{j}\, P_j(x)\, P_j(y)\right)$$

$$= \sum_{\mathbf{i}} \frac{1}{1^{i_1} i_1! \cdots d^{i_d} i_d!} \sum_\lambda \omega_\lambda(\mathbf{i})\, S_\lambda(x) \sum_\mu \omega_\mu(\mathbf{i})\, S_\mu(y).$$

Comparing with (A.13), the conclusion follows. $\square$

#### Specializations: $S_\lambda(1, \ldots, 1)$

We should remark that we have chosen to write our formulas for a fixed number $k$ of variables, since that often simplifies computations when $k$ is small. It is more usual to require the number of variables to be large, at least as large as the numbers being partitioned --- or in the limiting ring with an infinite number of variables, cf. Exercise A.32. The formulas for smaller $k$ are then recovered by setting the variables $x_i = 0$ for $i > k$.

The next two exercises give formulas for the value of the Schur polynomials when the variables $x_i$ are all set equal to 1; these numbers are the dimensions of the corresponding representations.

When $x_i = x^{i-1}$, the numerators in (A.4) are van der Monde determinants, leading to

$$S_\lambda(1, x, x^2, \ldots, x^{k-1}) = x^k \prod_{i < j} \frac{x^{\lambda_i - \lambda_j + j - i} - 1}{x^{j-i} - 1}.$$

Taking the limit as $x \to 1$, one finds

$$S_\lambda(1, \ldots, 1) = \prod_{i < j} \frac{\lambda_i - \lambda_j + j - i}{j - i}.$$

By (A.5) and (A.6) we have also the following two formulas:

$$S_\lambda(1, \ldots, 1) = \lvert h_{\lambda_i + j - i}\rvert, \quad \text{where } \sum h_j t^j = \frac{1}{(1 - t)^k},$$

$$S_\lambda(1, \ldots, 1) = \left\lvert\binom{k}{\mu_i + j - i}\right\rvert, \quad \text{where } (\mu_1, \ldots, \mu_l) = \lambda'.$$

#### The Involution $\vartheta$

In the limiting ring $\Lambda = \lim \Lambda(k)$ (where $\Lambda(k)$ denotes the ring of symmetric polynomials in $k$ variables), we have $\Lambda = \mathbb{Z}[H_1, \ldots, H_k, \ldots] = \mathbb{Z}[E_1, \ldots, E_k, \ldots]$. A ring homomorphism $\vartheta\colon \Lambda \to \Lambda$ can be defined by requiring

$$\vartheta(E_i) = H_i \quad \text{for all } i.$$

This is an involution: $\vartheta^2 = \vartheta$. Equivalently, $\vartheta(H_i) = E_i$. If $\lambda'$ is the conjugate partition to $\lambda$, then $\vartheta(S_\lambda) = S_{\lambda'}$.

### A.2 Proofs of the Determinantal Identities

To prove the Jacobi--Trudi identity (A.5), note the identities

$$x_j^p - E_1 x_j^{p-1} + E_2 x_j^{p-2} - \cdots + (-1)^k E_k x_j^{p-k} = 0, \tag{A.33}$$

for any $1 \le j \le k$, $p \ge k$. And for any $0 \le m < k$ and $p \ge k$,

$$H_{p-m} - E_1 H_{p-m-1} + E_2 H_{p-m-2} + \cdots + (-1)^k E_k H_{p-m-k} = 0. \tag{A.34}$$

Both of these follow immediately from the defining power series for the $E_j$ and $H_j$. Since these two recursion relations are the same, there are universal polynomials $A(p, q)$ in the variables $E_1, \ldots, E_k$ such that

$$x_j^p = A(p, 1) x_j^{k-1} + A(p, 2) x_j^{k-2} + \cdots + A(p, k), \tag{A.35}$$

$$H_{p-m} = A(p, 1) H_{k-m-1} + A(p, 2) H_{k-m-2} + \cdots + A(p, k) H_{-m}.$$

For any integers $\lambda_1, \ldots, \lambda_k$ this leads to matrix identities

$$(x_j^{\lambda_i + k - i})_{ij} = (A(\lambda_i + k - i, r))_{ir} \cdot (x_j^{k-r})_{rj}, \tag{A.36}$$

$$(H_{\lambda_i + j - i})_{ij} = (A(\lambda_i + k - i, r))_{ir} \cdot (H_{j-r})_{rj},$$

where $(\,)_{pq}$ denotes the $k \times k$ matrix whose $p, q$ entry is specified between the parentheses.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(A.37)</span></p>

The matrices $(H_{q-p})$ and $((-1)^{q-p} E_{q-p})$ are lower-triangular matrices with 1's along the diagonal, and are inverses of each other.

</div>

The identities (A.36) therefore combine to give

$$(x_j^{\lambda_i + k - i})_{ip} \cdot ((-1)^{q-p} E_{q-p})_{pq} \cdot (x_j^{k-q})_{qj} \tag{A.38}$$

Taking determinants gives (A.5), since the determinant of the matrix in the middle is 1.

To complete the proofs of the assertions in $\S$A.1, we show that the two determinants appearing in the Giambelli formulas (A.5) and (A.6) are equal, i.e., if $\lambda = (\lambda_1, \ldots, \lambda_k)$ and $\mu = (\mu_1, \ldots, \mu_l)$ are conjugate partitions, then

$$\lvert H_{\lambda_i + j - i}\rvert = \lvert E_{\mu_i + j - i}\rvert. \tag{A.40}$$

Here the $H_i$ and $E_i$ can be any elements (in a commutative ring) satisfying the identity $(\sum H_i t^i) \cdot (\sum (-1)^i E_i t^i) = 1$, with $H_0 = E_0 = 1$ and $H_i = E_i = 0$ for $i < 0$. To prove it, we need a combinatorial characterization of the conjugacy of partitions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(A.42)</span></p>

Let $A$ and $B$ be $r \times r$ matrices whose product is a scalar matrix $c \cdot I_r$. Let $(S, S')$ and $(T, T')$ be permutations of the sequence $(1, \ldots, r)$, where $S$ and $T$ consists of $k$ integers, $S'$ and $T'$ of $r - k$. Then

$$c^{r-k} \cdot A_{S,T} = \varepsilon \cdot \det(A) \cdot B_{T', S'},$$

where $\varepsilon$ is the product of the signs of the two permutations.

</div>

Applying this lemma to $A = (H_{q-p})$ and $B = ((-1)^{q-p} E_{q-p})$ with the same permutations $(S, S')$ and $(T, T')$ used in the proof of (A.40), one obtains (A.40). $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.44 — Folded Determinantal Identities)</span></p>

Let $\lambda = (\lambda_1, \ldots, \lambda_k)$ and $\mu = (\mu_1, \ldots, \mu_l)$ be conjugate partitions. Set

$$E_i' = E_i \text{ for } i \le 1, \quad \text{and} \quad E_i' = E_i - E_{i-2} \text{ for } i \ge 2.$$

Then the determinant of the $k \times k$ matrix whose $i$th row is

$$(H_{\lambda_i - i + 1} \quad H_{\lambda_i - i + 2} + H_{\lambda_i - i} \quad H_{\lambda_i - i + 3} + H_{\lambda_i - i - 1} \quad \cdots \quad H_{\lambda_i - i + k} + H_{\lambda_i - i - k+2})$$

is equal to the determinant of the $l \times l$ matrix whose $i$th row is

$$(E_{\mu_i - i + 1}' \quad E_{\mu_i - i + 2}' + E_{\mu_i - i}' \quad \cdots \quad E_{\mu_i - i + l}' + E_{\mu_i - i - l+2}').$$

Each of these determinants is equal to the determinant $\lvert E_{\mu_i + j} - E_{\mu_i - i - j}\rvert$ and to the determinant $\lvert H_{\lambda_i - i + j} - H_{\lambda_i - i - j}\rvert$.

</div>

Define $S_{\lbrace\lambda\rbrace}$ to be the determinant of this corollary:

$$S_{\lbrace\lambda\rbrace} = \lvert H_{\lambda_i - i + 1}' \quad H_{\lambda_i - i + 2}' + H_{\lambda_i - i}' \quad \cdots \quad H_{\lambda_i - i + k}' + H_{\lambda_i - i - k+2}'\rvert. \tag{A.47}$$

### A.3 Other Determinantal Identities

In this final section we prove some variations of these formulas which are useful for calculating characters of symplectic and orthogonal groups. We want to compare minors, not of $H = (H_{i-j})$ and $E = ((-1)^{i-j} E_{i-j})$, but of matrices $H^+$ and $E^-$ constructed from them by the following procedures.

For an $r \times r$ matrix $H = (H_{i,j})$, and a fixed integer $k$ between 1 and $r$, $H^+$ denotes the $r \times r$ matrix obtained from $H$ by folding $H$ along the $k$th column, and adding each column to the right of the $k$th column to the column the same distance to the left. That is,

$$H_{i,j}^+ = \begin{cases} H_{i,j} + H_{i,2k-j} & \text{if } j < k \\ H_{i,j} & \text{if } j \ge k \end{cases}$$

(with the convention that $H_{p,q} = 0$ if $p$ or $q$ is not between 1 and $r$). The matrix $E^-$ is obtained by folding $E$ along its $k$th row, and subtracting rows above this row from those below:

$$E_{i,j}^- = \begin{cases} E_{i,j} - E_{2k-i,j} & \text{if } i > k \\ E_{i,j} & \text{if } i \le k. \end{cases}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(A.43)</span></p>

If $H$ and $E$ are lower-triangular matrices with 1's along the diagonal, that are inverses to each other, then the same is true for $H^+$ and $E^-$.

</div>

#### Symplectic Characters

For applications to symplectic and orthogonal characters we need to specialize the variables $x_1, \ldots, x_k$. First (for the symplectic group $\mathrm{Sp}_{2n}$) take $k = 2n$, let $z_1, \ldots, z_n$ be independent variables, and specialize

$$x_1 \mapsto z_1, \ldots, x_n \mapsto z_n, \quad x_{n+1} \mapsto z_1^{-1}, \ldots, x_{2n} \mapsto z_n^{-1}.$$

Set

$$J_j = H_j(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1}) \tag{A.49}$$

in the field $\mathbb{Q}(z_1, \ldots, z_n)$ of rational functions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.50 — Symplectic Character Determinant)</span></p>

Given integers $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$, we have

$$\frac{\lvert z_j^{\lambda_i + n - i + 1} - z_j^{-(\lambda_i + n - i + 1)}\rvert}{\lvert z_j^{n - i + 1} - z_j^{-(n - i + 1)}\rvert} = \lvert J_\lambda\rvert,$$

where $J_\lambda$ is the $n \times n$ matrix whose $i$th row is

$$(J_{\lambda_i - i + 1} \quad J_{\lambda_i - i + 2} + J_{\lambda_i - i} \quad \cdots \quad J_{\lambda_i - i + n} + J_{\lambda_i - i - n + 2}).$$

</div>

Corollary A.46 gives three other alternative expressions for this determinant, e.g.,

$$\lvert J_\lambda\rvert = \lvert e_{\mu_i - i + j} - e_{\mu_i - i - j}\rvert, \tag{A.51}$$

where $e_j = E_j(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1})$, and $\mu$ is the conjugate partition to $\lambda$.

#### Odd Orthogonal Characters

Next (for the odd orthogonal groups $\mathrm{O}_{2n+1}$) let $k = 2n + 1$, and specialize the variables $x_1, \ldots, x_{2n}$ as above, and $x_{2n+1} \mapsto 1$. We introduce variables $z_j^{1/2}$ and $z_j^{-1/2}$, square roots of the variables just considered, and we work in the field $\mathbb{Q}(z_1^{1/2}, \ldots, z_n^{1/2})$. Set

$$K_j = H_j'(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1}, 1) = H_j(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1}, 1) - H_{j-2}(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1}, 1), \tag{A.59}$$

where $H_j$ is the $j$th complete symmetric polynomial in $2n + 1$ variables.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.60 — Odd Orthogonal Character Determinant)</span></p>

Given integers $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$, we have

$$\frac{\lvert z_j^{\lambda_i + n - i + 1/2} - z_j^{-(\lambda_i + n - i + 1/2)}\rvert}{\lvert z_j^{n - i + 1/2} - z_j^{-(n - i + 1/2)}\rvert} = \lvert K_\lambda\rvert,$$

where $K_\lambda$ is the $n \times n$ matrix whose $i$th row is

$$(K_{\lambda_i - i + 1} \quad K_{\lambda_i - i + 2} + K_{\lambda_i - i} \quad \cdots \quad K_{\lambda_i - i + n} + K_{\lambda_i - i - n + 2}).$$

</div>

Corollary A.46 gives three other alternative expressions for this determinant, e.g.,

$$\lvert K_\lambda\rvert = \lvert h_{\lambda_i - i + j} - h_{\lambda_i - i - j}\rvert, \tag{A.61}$$

where $h_j = H_j(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1}, 1)$.

#### Even Orthogonal Characters

Finally (for the even orthogonal groups $\mathrm{O}_{2n}$), let $k = 2n$, and specialize the variables $x_1, \ldots, x_{2n}$ as above. Set

$$L_j = H_j'(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1}) = H_j(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1}) - H_{j-2}(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1}), \tag{A.63}$$

with $H_j$ the complete symmetric polynomial in $2n$ variables.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(A.64 — Even Orthogonal Character Determinant)</span></p>

Given integers $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$, we have

$$\frac{\lvert z_j^{\lambda_i + n - i} + z_j^{-(\lambda_i + n - i)}\rvert}{\lvert z_j^{n - i} + z_j^{-(n - i)}\rvert} = \begin{cases} \frac{1}{2}\lvert L_\lambda\rvert & \text{if } \lambda_n > 0 \\ \lvert L_\lambda\rvert & \text{if } \lambda_n = 0, \end{cases}$$

where $L_\lambda$ is the $n \times n$ matrix whose $i$th row is

$$(L_{\lambda_i - i + 1} \quad L_{\lambda_i - i + 2} + L_{\lambda_i - i} \quad \cdots \quad L_{\lambda_i - i + n} + L_{\lambda_i - i - n + 2}).$$

</div>

As before, there are other expressions for these determinants, e.g.,

$$\lvert L_\lambda\rvert = \lvert h_{\lambda_i - i + j} - h_{\lambda_i - i - j}\rvert, \tag{A.65}$$

where $h_j = H_j(z_1, \ldots, z_n, z_1^{-1}, \ldots, z_n^{-1})$.

---

## Appendix B: On Multilinear Algebra

In this appendix we state the basic facts about tensor products and exterior and symmetric powers that are used in the text. It is hoped that a reader with some linear algebra background can fill in details of the proofs.

### B.1 Tensor Products

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tensor Product)</span></p>

The **tensor product** of two vector spaces $V$ and $W$ over a field $K$ is a vector space $V \otimes W$ equipped with a bilinear map

$$V \times W \to V \otimes W, \qquad v \times w \mapsto v \otimes w,$$

which is **universal**: for any bilinear map $\beta\colon V \times W \to U$ to a vector space $U$, there is a unique linear map from $V \otimes W$ to $U$ that takes $v \otimes w$ to $\beta(v, w)$. This universal property determines the tensor product up to canonical isomorphism.

If $\lbrace e_i\rbrace$ and $\lbrace f_j\rbrace$ are bases for $V$ and $W$, the elements $\lbrace e_i \otimes f_j\rbrace$ form a basis for $V \otimes W$. The construction is functorial: linear maps $V \to V'$ and $W \to W'$ determine a linear map from $V \otimes W$ to $V' \otimes W'$.

Similarly one has the tensor product $V_1 \otimes \cdots \otimes V_n$ of $n$ vector spaces, with its universal multilinear map

$$V_1 \times \cdots \times V_n \to V_1 \otimes \cdots \otimes V_n.$$

</div>

---

## Appendix C: On Semisimplicity

### C.1 The Killing Form and Cartan's Criterion

We recall first the Jordan decomposition of a linear transformation $X$ of a finite-dimensional complex vector space $V$ as a sum of its semisimple and nilpotent parts: $X = X_s + X_n$, where $X_s$ is the semisimple part of $X$, and $X_n$ the nilpotent part. It is uniquely characterized by the fact that $X_s$ is semisimple (diagonalizable), $X_n$ is nilpotent, and $X_s$ and $X_n$ commute with each other. In fact, $X_s$ and $X_n$ can be written as polynomials in $X$, so any endomorphism that commutes with $X$ automatically commutes with $X_s$ and $X_n$. One case of the invariance of Jordan decomposition is an easy calculation:

For any $X \in \mathfrak{gl}(V)$, the endomorphism $\mathrm{ad}(X)$ of $\mathfrak{gl}(V)$ satisfies $\mathrm{ad}(X)_s = \mathrm{ad}(X_s)$ and $\mathrm{ad}(X)_n = \mathrm{ad}(X_n)$.

There is a Killing form $B_V$ defined on $\mathfrak{gl}(V)$ by the formula

$$B_V(X, Y) = \mathrm{Tr}(X \circ Y), \tag{C.2}$$

where $\mathrm{Tr}$ is the trace and $\circ$ denotes composition of transformations. As in (14.23), the identity

$$B_V(X, [Y, Z]) = B_V([X, Y], Z) \tag{C.3}$$

holds for all $X$, $Y$, $Z$ in $\mathfrak{gl}(V)$.

The Killing form $B$ on a Lie algebra $\mathfrak{g}$ is that of Exercise C.1 for the adjoint representation: $B(X, Y) = B_\mathfrak{g}(\mathrm{ad}(X), \mathrm{ad}(Y))$. This was introduced in Lecture 14, where a few of its properties were proved. Here we use the Killing form to characterize solvability and semisimplicity of the Lie algebra.

If $\mathfrak{g}$ is solvable, by Lie's theorem its adjoint representation can be put in upper-triangular form. It follows that $\mathscr{D}\mathfrak{g} = [\mathfrak{g}, \mathfrak{g}]$ acts by strictly upper-triangular matrices. So if $X$ is in $\mathscr{D}\mathfrak{g}$ and $Y$ in $\mathfrak{g}$, then $\mathrm{ad}(X) \circ \mathrm{ad}(Y)$ is strictly upper triangular; in particular its trace $B(X, Y)$ is zero.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(C.4 — Solvability Criterion)</span></p>

The Lie algebra $\mathfrak{g}$ is solvable if and only if $B(\mathfrak{g}, \mathscr{D}\mathfrak{g}) = 0$.

</div>

We will prove something that looks a little weaker, but will turn out to be a little stronger:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(C.5 — Cartan's Criterion)</span></p>

If $\mathfrak{g}$ is a subalgebra of $\mathfrak{gl}(V)$ and $B_V(X, Y) = 0$ for all $X$ and $Y$ in $\mathfrak{g}$, then $\mathfrak{g}$ is solvable.

</div>

**Proof.** It suffices to show that every element of $\mathscr{D}\mathfrak{g}$ is nilpotent, for then by Engel's theorem $\mathscr{D}\mathfrak{g}$ must be a nilpotent ideal, and therefore $\mathfrak{g}$ is solvable.

So take $X \in \mathscr{D}\mathfrak{g}$, and let $\lambda_1, \ldots, \lambda_r$ be its eigenvalues (counted with multiplicity) for $X$ as an endomorphism of $V$. We must show the $\lambda_i$ are all zero. These eigenvalues satisfy some obvious relations; for example, $\sum \bar{\lambda}_i \lambda_i = \mathrm{Tr}(X \circ X) = B_V(X, X) = 0$. What we need to show is

$$\bar{\lambda}_1 \lambda_1 + \cdots + \bar{\lambda}_r \lambda_r = 0. \tag{C.6}$$

To prove this, take a basis for $V$ so that $X$ is in Jordan canonical form, with $\lambda_1, \ldots, \lambda_r$ down the diagonal; the semisimple part $D = X_s$ of $X$ is this diagonal transformation. Let $\bar{D}$ be the endomorphism of $V$ given by the diagonal matrix with $\bar{\lambda}_1, \ldots, \bar{\lambda}_r$ down the diagonal. Since $\mathrm{Tr}(\bar{D} \circ X) = \sum \bar{\lambda}_i \lambda_i$, it suffices to prove

$$\mathrm{Tr}(\bar{D} \circ X) = 0. \tag{C.7}$$

Since $X$ is a sum of commutators $[Y, Z]$, with $Y$ and $Z$ in $\mathfrak{g}$, $\mathrm{Tr}(\bar{D} \circ X)$ is a sum of terms of the form $\mathrm{Tr}(\bar{D} \circ [Y, Z]) = \mathrm{Tr}([\bar{D}, Y] \circ Z)$. So we will be done if we know that $[\bar{D}, Y]$ belongs to $\mathfrak{g}$, for our hypothesis is that $\mathrm{Tr}(\mathfrak{g} \circ \mathfrak{g}) \equiv 0$. That is, we are reduced to showing

$$\mathrm{ad}(\bar{D})(\mathfrak{g}) \subset \mathfrak{g}. \tag{C.8}$$

For this it suffices to prove that $\mathrm{ad}(\bar{D})$ can be written as a polynomial in $\mathrm{ad}(X)$, for we know that $\mathrm{ad}(X)^k(Y)$ is in $\mathfrak{g}$ if $X$ and $Y$ are in $\mathfrak{g}$. Since $\mathrm{ad}(D) = \mathrm{ad}(X_s) = \mathrm{ad}(X)_s$ is a polynomial in $\mathrm{ad}(X)$, it suffices to show that $\mathrm{ad}(\bar{D})$ can be written as a polynomial in $\mathrm{ad}(D)$. This is a simple computation: using the usual basis $\lbrace E_{ij}\rbrace$ for $\mathfrak{gl}(V)$, $\mathrm{ad}(D)$ and $\mathrm{ad}(\bar{D})$ are complex conjugate diagonal matrices, and any such are polynomials in each other. $\square$

We can prove now that if $\mathfrak{g}$ is a Lie algebra for which $B(\mathscr{D}\mathfrak{g}, \mathscr{D}\mathfrak{g}) \equiv 0$, then $\mathfrak{g}$ is solvable, which certainly implies Proposition C.4. By what we just proved, the image of $\mathscr{D}\mathfrak{g}$ by the adjoint representation in $\mathfrak{gl}(\mathfrak{g})$ is solvable. Since the kernel of the adjoint map is abelian, this makes $\mathscr{D}\mathfrak{g}$ solvable (cf. Exercise 9.8), and by definition this makes $\mathfrak{g}$ solvable. $\square$

It is easy to deduce from Cartan's criterion a criterion for semisimplicity --- part of which we saw in Lecture 14, but there assuming some facts we had not proved yet:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(C.10 — Semisimplicity Criterion)</span></p>

A Lie algebra $\mathfrak{g}$ is semisimple if and only if its Killing form $B$ is nondegenerate.

</div>

**Proof.** By (C.3) the null-space $\mathfrak{s} = \lbrace X \in \mathfrak{g}\colon B(X, Y) = 0 \text{ for all } Y \in \mathfrak{g}\rbrace$ is an ideal. Suppose $\mathfrak{g}$ is semisimple. By Cartan's criterion, the image $\mathrm{ad}(\mathfrak{s}) \subset \mathfrak{gl}(\mathfrak{g})$ is solvable; as in the preceding proof, $\mathfrak{s}$ is then solvable, so $\mathfrak{s} = 0$ by the definition of semisimple. Conversely, if $B$ is nondegenerate, we must show that any abelian ideal $\mathfrak{a}$ in $\mathfrak{g}$ must be zero. If $X \in \mathfrak{a}$ and $Y \in \mathfrak{g}$, then $A = \mathrm{ad}(X) \circ \mathrm{ad}(Y)$ maps $\mathfrak{g}$ into $\mathfrak{a}$ and $\mathfrak{a}$ to 0, so $\mathrm{Tr}(A) = 0$. This implies that $\mathfrak{a} \subset \mathfrak{s} = 0$, as required. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(C.11)</span></p>

A semisimple Lie algebra is a direct product of simple Lie algebras.

</div>

**Proof.** For any ideal $\mathfrak{h}$ of $\mathfrak{g}$, the annihilator $\mathfrak{h}^\perp = \lbrace X \in \mathfrak{g}\colon B(X, Y) = 0 \text{ for all } Y \in \mathfrak{h}\rbrace$ is an ideal. By Cartan's criterion, $\mathfrak{h} \cap \mathfrak{h}^\perp$ is solvable, hence zero, so $\mathfrak{g} = \mathfrak{h} \oplus \mathfrak{h}^\perp$. The decomposition follows by a simple induction. $\square$

It follows that $\mathfrak{g} = \mathscr{D}\mathfrak{g}$, and that all ideals and images of $\mathfrak{g}$ are semisimple. In fact, if $\mathfrak{g}$ is a direct product of simple Lie algebras, the only ideals in $\mathfrak{g}$ are sums of some of the factors. In particular, the decomposition into simple factors is unique (not just up to isomorphism).

### C.2 Complete Reducibility and the Jordan Decomposition

We repeat that this section is optional, since the results can be deduced from the existence of a compact group such that the complexification of its Lie algebra is a given semisimple Lie algebra. We include here the standard algebraic approach. A finite-dimensional representation of a Lie algebra $\mathfrak{g}$ will be called a $\mathfrak{g}$-module, and a $\mathfrak{g}$-invariant subspace a submodule.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(C.15 — Complete Reducibility)</span></p>

Let $V$ be a representation of the semisimple Lie algebra $\mathfrak{g}$ and $W \subset V$ a submodule. Then there exists a submodule $W' \subset V$ complementary to $W$.

</div>

**Proof.** Since the image of $\mathfrak{g}$ by the representation is semisimple, we may assume $\mathfrak{g} \subset \mathfrak{gl}(V)$. We will require a slight generalization of the Casimir operator $C_V$ which was used in $\S$25.1 in the proof of Freudenthal's formula. We take a basis $U_1, \ldots, U_l$ for $\mathfrak{g}$, and a dual basis $U_1', \ldots, U_l'$, but this time with respect to the Killing form $B_V$ defined in Exercise C.1: $B_V(X, Y) = \mathrm{Tr}(X \circ Y)$. (Note by Cartan's criterion that $B_V$ is nondegenerate.) Then $C_V$ is defined by the formula $C_V(v) = \sum U_i \cdot (U_i' \cdot v)$.

As before, a simple calculation shows that $C_V$ is an endomorphism of $V$ that commutes with the action of $\mathfrak{g}$. Its trace is

$$\mathrm{Tr}(C_V) = \sum \mathrm{Tr}(U_i \circ U_i') = \sum B_V(U_i, U_i') = \dim(\mathfrak{g}). \tag{C.16}$$

We note also that since $C_V$ maps any submodule $W$ to itself, its kernel $\mathrm{Ker}(C_V)$ and image are submodules.

Note first that all one-dimensional representations of a semisimple $\mathfrak{g}$ are trivial, since $\mathscr{D}\mathfrak{g}$ must act trivially on a one-dimensional space, and $\mathfrak{g} = \mathscr{D}\mathfrak{g}$.

We proceed to the proof itself. As should be familiar from Lecture 9, the basic case to prove is when $W \subset V$ is an irreducible invariant subspace of codimension one. Then $C_V$ maps $W$ into itself, and $C_V$ acts trivially on $V/W$. But now by Schur's lemma, since $W$ is irreducible, $C_V$ is multiplication by a scalar on $W$. This scalar is not zero, or (C.16) would be contradicted. Hence $V = W \oplus \mathrm{Ker}(C_V)$, which finishes this special case.

It follows easily by induction on the dimension that the same is true whenever $W \subset V$ has codimension one. For if $W$ is not irreducible, let $Z$ be a nonzero submodule, and find a complement to $W/Z \subset V/Z$ (by induction), say $Y/Z$. Since $Y/Z$ is one dimensional, find (by induction) $U$ so that $Y = Z \oplus U$. Then $V = W \oplus U$.

By the same argument, it suffices to prove the statement of the theorem when $W$ is irreducible. Consider the restriction map

$$\rho\colon \mathrm{Hom}(V, W) \to \mathrm{Hom}(W, W),$$

a homomorphism of $\mathfrak{g}$-modules. The second contains the one-dimensional submodule $\mathrm{Hom}_\mathfrak{g}(W, W)$. By the preceding case, there is a one-dimensional submodule of $\rho^{-1}(\mathrm{Hom}_\mathfrak{g}(W, W)) \subset \mathrm{Hom}(V, W)$ which maps onto $\mathrm{Hom}_\mathfrak{g}(W, W)$ by $\rho$. Since one-dimensional modules are trivial, this means there is a $\mathfrak{g}$-invariant $\psi$ in $\mathrm{Hom}(V, W)$ such that $\rho(\psi) = 1$. But this means that $\psi$ is a $\mathfrak{g}$-invariant projection of $V$ onto $W$, so $V = W \oplus \mathrm{Ker}(\psi)$, as required. $\square$

#### Invariance of the Jordan Decomposition

We will apply this to prove the invariance of Jordan decomposition (Theorem 9.20). The essential point is:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(C.17 — Jordan Decomposition in Semisimple Lie Algebras)</span></p>

Let $\mathfrak{g}$ be a semisimple Lie subalgebra of $\mathfrak{gl}(V)$. Then for any element $X \in \mathfrak{g}$, the semisimple part $X_s$ and the nilpotent part $X_n$ are also in $\mathfrak{g}$.

</div>

**Proof.** The idea is to write $\mathfrak{g}$ as an intersection of Lie subalgebras of $\mathfrak{gl}(V)$ for which the conclusion of the theorem is easy to prove. For example, we know $\mathfrak{g} \subset \mathfrak{sl}(V)$ since $\mathfrak{g} = \mathscr{D}\mathfrak{g}$, and clearly $X_s$ and $X_n$ are traceless if $X$ is. Similarly, if $V$ is not irreducible, for any submodule $W$ of $V$, let

$$\mathfrak{s}_W = \lbrace Y \in \mathfrak{gl}(V)\colon Y(W) \subset W \text{ and } \mathrm{Tr}(Y\vert_W) = 0\rbrace.$$

Then $\mathfrak{g}$ is also a subalgebra of $\mathfrak{s}_W$, and $X_s$ and $X_n$ are also in $\mathfrak{s}_W$.

Since $[X, \mathfrak{g}] \subset \mathfrak{g}$, it follows that $[p(X), \mathfrak{g}] \subset \mathfrak{g}$ for any polynomial $p(T)$. Hence $[X_s, \mathfrak{g}] \subset \mathfrak{g}$ and $[X_n, \mathfrak{g}] \subset \mathfrak{g}$. In other words, $X_s$ and $X_n$ belong to the Lie subalgebra $\mathfrak{n}$ of $\mathfrak{gl}(V)$ consisting of those endomorphisms $A$ such that $[A, \mathfrak{g}] \subset \mathfrak{g}$. Now we claim that $\mathfrak{g}$ is the intersection of $\mathfrak{n}$ and all the algebras $\mathfrak{s}_W$ for all submodules $W$ of $V$. This claim, as we saw, will finish the proof. Let $\mathfrak{g}'$ be the intersection of all these Lie algebras. Then $\mathfrak{g}' = \mathfrak{g} \oplus U$. Since $[\mathfrak{g}, \mathfrak{g}'] \subset \mathfrak{g}$, we must have $[\mathfrak{g}, U] = 0$. To show that $U$ is 0, it suffices to show that for any $Y \in U$ its restriction to any irreducible submodule $W$ of $V$ is zero (noting that $Y$ preserves $W$ since $Y \in \mathfrak{s}_W$, and that $V$ is a sum of irreducible submodules). But since $Y$ commutes with $\mathfrak{g}$, Schur's lemma implies that the restriction of $Y$ to $W$ is multiplication by a scalar, and the assumption that $Y \in \mathfrak{s}_W$ means that $\mathrm{Tr}(Y\vert_W) = 0$, so $Y\vert_W = 0$, as required. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(C.18)</span></p>

If $\rho\colon \mathfrak{g} \to \mathfrak{gl}(V)$ is any representation of a semisimple Lie algebra $\mathfrak{g}$, then $\rho(X_s)$ is the semisimple part of $\rho(X)$ and $\rho(X_n)$ is the nilpotent part of $\rho(X)$.

</div>

It follows that an element $X$ in a semisimple Lie algebra that is semisimple in one faithful representation is semisimple in all representations.

### C.3 On Derivations

In this final section we collect a few facts relating the Killing form, solvability, and nilpotency with derivations of Lie algebras, mainly for use in Appendix E. We first prove a couple of lemmas related to the Lie--Engel theory of Lecture 9. For these $\mathfrak{g}$ is any Lie algebra, $\mathfrak{r} = \mathrm{Rad}(\mathfrak{g})$ denotes its radical, and $\mathscr{D}\mathfrak{g} = [\mathfrak{g}, \mathfrak{g}]$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(C.19)</span></p>

For any representation $\rho\colon \mathfrak{g} \to \mathfrak{gl}(V)$, every element of $\rho(\mathscr{D}\mathfrak{g} \cap \mathfrak{r})$ is a nilpotent endomorphism.

</div>

**Proof.** It suffices to treat the case where the representation $V$ is irreducible. We may replace $\mathfrak{g}$ by its image, so we may assume $\rho$ is injective. In this case we show that $\mathscr{D}\mathfrak{g} \cap \mathfrak{r} = 0$. We may assume $\mathfrak{r} \neq 0$. Consider the largest integer $k$ such that $\mathfrak{a} = \mathscr{D}^k \mathfrak{r}$ is not zero. This $\mathfrak{a}$ is an abelian ideal of $\mathfrak{g}$. It suffices to show that $\mathscr{D}\mathfrak{g} \cap \mathfrak{a} = 0$, for if $k > 0$, then $\mathfrak{a} \subset \mathscr{D}\mathfrak{g}$.

We need three facts:
1. If $\mathfrak{g} \subset \mathfrak{gl}(V)$ is an irreducible representation and $\mathfrak{b}$ is any ideal of $\mathfrak{g}$ that consists of nilpotent transformations of $V$, then $\mathfrak{b} = 0$. (Indeed, by Engel's theorem, $W = \lbrace v \in V\colon X(v) = 0 \text{ for all } X \in \mathfrak{b}\rbrace$ is nonzero, and by Lemma 9.13, $W$ is preserved by $\mathfrak{g}$. Since $V$ is irreducible, $W = V$, which says that $\mathfrak{b} = 0$.)
2. A transformation $X$ is nilpotent exactly when $\mathrm{Tr}(X^n) = 0$ for all positive integers $n$.
3. $\mathrm{Tr}([X, Y] \cdot Z) = 0$ whenever $[Y, Z] = 0$. (This follows from $\mathrm{Tr}([X, Y] \cdot Z) = \mathrm{Tr}(X \cdot [Y, Z])$.)

Next we can see that $[\mathfrak{g}, \mathfrak{a}] = 0$. If $X \in \mathfrak{g}$ and $Y \in \mathfrak{a}$, then $[X, Y] \in \mathfrak{a}$; since $\mathfrak{a}$ is abelian, $Y$ commutes with $[X, Y]$ and hence with powers of $[X, Y]$. Applying (iii) with $Z = [X, Y]^{n-1}$ gives $\mathrm{Tr}([X, Y]^n) = 0$ for $n > 0$, and (ii) and (i) imply that $[\mathfrak{g}, \mathfrak{a}] = 0$.

Finally we show that $\mathscr{D}\mathfrak{g} \cap \mathfrak{a} = 0$. If $X \in \mathfrak{g}$, $Y \in \mathfrak{g}$ and $[X, Y] \in \mathfrak{a}$, then $Y$ commutes with $[X, Y]$ and the same argument shows that $\mathrm{Tr}([X, Y]^n) = 0$, and (ii) and (i) again show that $\mathscr{D}\mathfrak{g} \cap \mathfrak{a} = 0$. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(C.20)</span></p>

For any Lie algebra $\mathfrak{g}$, $[\mathfrak{g}, \mathfrak{r}]$ is nilpotent.

</div>

**Proof.** Look at the images $\bar{\mathfrak{g}}$ and $\bar{\mathfrak{r}}$ of $\mathfrak{g}$ and $\mathfrak{r}$ by the adjoint representation $\mathrm{ad}\colon \mathfrak{g} \to \mathfrak{gl}(\mathfrak{g})$. By Lemma C.19 and Engel's theorem, $[\bar{\mathfrak{g}}, \bar{\mathfrak{r}}]$ is a nilpotent ideal of $\bar{\mathfrak{g}}$. Since the kernel of the adjoint representation is the center of $\mathfrak{g}$, it follows that the quotient of $[\mathfrak{g}, \mathfrak{r}]$ by a central ideal is nilpotent, which implies that $[\mathfrak{g}, \mathfrak{r}]$ itself is nilpotent. $\square$

#### Characteristic Ideals and Derivations

An ideal $\mathfrak{a}$ of a Lie algebra $\mathfrak{g}$ is called **characteristic** if any derivation of $\mathfrak{g}$ maps $\mathfrak{a}$ into itself. Note that an ideal is just a subspace that is preserved by all inner derivations $D_X = \mathrm{ad}(X)$. It follows from the definitions that if $\mathfrak{a}$ is any ideal in $\mathfrak{g}$, then any characteristic ideal in $\mathfrak{a}$ is automatically an ideal in $\mathfrak{g}$.

The following simple construction is useful for turning questions about general derivations into questions about inner derivations. Given any Lie algebra $\mathfrak{g}$ and a derivation $D$ of $\mathfrak{g}$, let $\mathfrak{g}' = \mathfrak{g} \oplus \mathbb{C}$, and define a bracket on $\mathfrak{g}'$ by

$$[(X, \lambda), (Y, \mu)] = ([X, Y] + \lambda D(Y) - \mu D(X), 0).$$

It is easy to verify that $\mathfrak{g}'$ is a Lie algebra containing $\mathfrak{g} = \mathfrak{g} \oplus 0$ as an ideal, and that, setting $\xi = (0, 1)$, the restriction of $D_\xi = \mathrm{ad}(\xi)$ to $\mathfrak{g}$ is the given derivation $D$.

As a simple application of this construction, if $B$ is the Killing form on $\mathfrak{g}$, we have the identity

$$B(D(X), Y) + B(X, D(Y)) = 0 \tag{C.21}$$

for any derivation $D$ of $\mathfrak{g}$, and any $X$ and $Y$ in $\mathfrak{g}$. Indeed, if $B'$ is the Killing form on $\mathfrak{g}'$, (C.3) gives $B'([\xi, X], Y) + B'(X, [\xi, Y]) = 0$; since $\mathfrak{g}$ is an ideal in $\mathfrak{g}'$, $B$ is the restriction of $B'$ to $\mathfrak{g}$, and (C.21) follows.

From (C.21) it follows that if $\mathfrak{a}$ is a characteristic ideal of $\mathfrak{g}$, then its orthogonal complement with respect to the Killing form is also a characteristic ideal of $\mathfrak{g}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(C.23)</span></p>

If $\mathfrak{a}$ is an ideal in a Lie algebra $\mathfrak{g}$, then $\mathrm{Rad}(\mathfrak{a}) = \mathrm{Rad}(\mathfrak{g}) \cap \mathfrak{a}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(C.24)</span></p>

If $D$ is a derivation of a Lie algebra $\mathfrak{g}$, then $D(\mathrm{Rad}(\mathfrak{g}))$ is contained in a nilpotent ideal of $\mathfrak{g}$.

</div>

**Proof.** Construct $\mathfrak{g}' = \mathfrak{g} \oplus \mathbb{C}$ as before, with $\xi = (0, 1)$. Since $\mathrm{Rad}(\mathfrak{g}) \subset \mathrm{Rad}(\mathfrak{g}')$, we have $D(\mathrm{Rad}(\mathfrak{g})) = [\xi, \mathrm{Rad}(\mathfrak{g})] \subset [\mathfrak{g}', \mathrm{Rad}(\mathfrak{g}')] \cap \mathfrak{g}$. By Lemma C.20, $[\mathfrak{g}', \mathrm{Rad}(\mathfrak{g}')]$ is a nilpotent ideal in $\mathfrak{g}'$, so its intersection with $\mathfrak{g}$ is also nilpotent. $\square$

Just as with the notion of solvability, any Lie algebra $\mathfrak{g}$ contains a largest nilpotent ideal, usually called the **nil radical** of $\mathfrak{g}$, and denoted $\mathrm{Nil}(\mathfrak{g})$ or $\mathfrak{n}$. Proposition C.24 says that any derivation maps $\mathfrak{r}$ into $\mathfrak{n}$, which includes the result of Lemma C.20 that $[\mathfrak{g}, \mathfrak{r}] \subset \mathfrak{n}$. The existence of this ideal follows from:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(C.25)</span></p>

If $\mathfrak{a}$ and $\mathfrak{b}$ are nilpotent ideals in a Lie algebra $\mathfrak{g}$, then $\mathfrak{a} + \mathfrak{b}$ is also a nilpotent ideal.

</div>

Since $\mathrm{Nil}(\mathfrak{g}) \subset \mathrm{Rad}(\mathfrak{g})$, it follows from Proposition C.24 that $\mathrm{Nil}(\mathfrak{g})$ is a characteristic ideal of $\mathfrak{g}$. The same reasoning as in Corollary C.23 gives:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(C.26)</span></p>

If $\mathfrak{a}$ is an ideal in a Lie algebra $\mathfrak{g}$, then $\mathrm{Nil}(\mathfrak{a}) = \mathrm{Nil}(\mathfrak{g}) \cap \mathfrak{a}$.

</div>

#### The Universal Enveloping Algebra

If $\mathfrak{g}$ is a Lie algebra, its **universal enveloping algebra** $U = U(\mathfrak{g})$ is the quotient of the tensor algebra of $\mathfrak{g}$ modulo the two-sided ideal generated by all $X \otimes Y - Y \otimes X - [X, Y]$ for all $X$, $Y$ in $\mathfrak{g}$. It is an associative algebra, with a map $\iota\colon \mathfrak{g} \to U$ such that

$$\iota([X, Y]) = [\iota(X), \iota(Y)] = \iota(X)\iota(Y) - \iota(Y)\iota(X),$$

and satisfying the universal property: for any linear map $\varphi$ from $\mathfrak{g}$ to an associative algebra $A$ such that $\varphi([X, Y]) = [\varphi(X), \varphi(Y)]$ for all $X$, $Y$, there is a unique homomorphism of algebras $\tilde{\varphi}\colon U \to A$ such that $\varphi = \tilde{\varphi} \circ \iota$. For example, a representation $\rho\colon \mathfrak{g} \to \mathfrak{gl}(V)$ determines an algebra homomorphism $\tilde{\rho}\colon U(\mathfrak{g}) \to \mathrm{End}(V)$. Conversely, any representation arises in this way.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(C.27)</span></p>

For any derivation $D$ of a Lie algebra $\mathfrak{g}$, there is a unique derivation $\tilde{D}$ of the associative algebra $U(\mathfrak{g})$ such that $\tilde{D} \circ \iota = \iota \circ D$.

</div>

**Proof.** Define an endomorphism of the tensor algebra of $\mathfrak{g}$ which is zero on the zeroth tensor power, and on the $n$th tensor power is

$$X_1 \otimes \cdots \otimes X_n \mapsto DX_1 \otimes X_2 \otimes \cdots \otimes X_n + X_1 \otimes DX_2 \otimes \cdots \otimes X_n + \cdots + X_1 \otimes X_2 \otimes \cdots \otimes DX_n.$$

This is well defined, since it is multilinear in each factor, and is easily checked to be a derivation of the tensor algebra; denote it by $D'$. To see that $D'$ passes to the quotient $U(\mathfrak{g})$ one checks routinely that it vanishes on generators for the ideal of relations. $\square$

It is a fact that the canonical map $\iota$ embeds $\mathfrak{g}$ in $U(\mathfrak{g})$. The **Poincar&eacute;--Birkhoff--Witt theorem** asserts that, in fact, if $U(\mathfrak{g})$ is filtered with the $n$th piece generated by all products of elements of $\iota(\mathfrak{g})$ of at most $n$ products, then the associated graded ring is the symmetric algebra on $\mathfrak{g}$. Equivalently, if $X_1, \ldots, X_r$ is a basis for $\mathfrak{g}$, then the monomials $X_1^{i_1} \cdots X_r^{i_r}$ form a basis for $U(\mathfrak{g})$. We do not need this theorem, but we will use the fact that these monomials generate $U(\mathfrak{g})$; this follows by a simple induction, using the equations $X_i \cdot X_j - X_j \cdot X_i = [X_i, X_j]$ to rearrange the order in products.

---

## Appendix D: Cartan Subalgebras

Our task here is to prove the basic general facts that were stated in Lecture 14 about the decomposition of a semisimple Lie algebra $\mathfrak{g}$ into a Cartan algebra $\mathfrak{h}$ and a sum of root spaces $\mathfrak{g}_\alpha$, including the existence of such $\mathfrak{h}$ and its uniqueness up to conjugation.

### D.1 The Existence of Cartan Subalgebras

Note that if we have a decomposition as in Lecture 14, and $H$ is any element of $\mathfrak{h}$ such that $\alpha(H) \neq 0$ for all roots $\alpha$, then $\mathfrak{h}$ is determined by $H$: $\mathfrak{h} = \mathfrak{c}(H)$, where

$$\mathfrak{c}(H) = \lbrace X \in \mathfrak{g}\colon [H, X] = 0\rbrace. \tag{D.1}$$

The elements of $\mathfrak{h}$ with this property are called **regular**. They form a Zariski open subset of $\mathfrak{h}$: the complement of the union of the hyperplanes defined by the equations $\alpha = 0$. In particular, regular elements are dense in $\mathfrak{h}$. If $H \in \mathfrak{h}$ is not regular, then $\mathfrak{c}(H)$ is larger than $\mathfrak{h}$, since it contains other root spaces. Note that all elements of $\mathfrak{h}$ are also semisimple, i.e., they are equal to their semisimple parts.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(D.2 — Rank and Regular Elements)</span></p>

The **rank** $n$ of a semisimple Lie algebra $\mathfrak{g}$ is the minimum of the dimension of $\mathfrak{c}(H)$ as $H$ varies over all semisimple elements of $\mathfrak{g}$. A semisimple element $H$ is called **regular** if $\mathfrak{c}(H)$ has dimension $n$. A **Cartan subalgebra** of $\mathfrak{g}$ is an abelian subalgebra all of whose elements are semisimple, and that is not contained in any larger such subalgebra.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(D.3)</span></p>

If $H$ is regular, then $\mathfrak{c}(H)$ is a Cartan subalgebra.

</div>

For any semisimple element $H$, $\mathfrak{g}$ decomposes into eigenspaces for the adjoint action of $H$:

$$\mathfrak{g} = \bigoplus_\lambda \mathfrak{g}_\lambda(H) = \mathfrak{c}(H) \oplus \bigoplus_{\lambda \neq 0} \mathfrak{g}_\lambda(H), \tag{D.4}$$

where $\mathfrak{g}_\lambda(H) = \lbrace X \in \mathfrak{g}\colon [H, X] = \lambda X\rbrace$, and $\mathfrak{c}(H) = \mathfrak{g}_0(H)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(D.6)</span></p>

If $H$ is regular, then $\mathfrak{g}_0(H)$ is abelian.

</div>

**Proof.** Consider how the Killing form $B$ respects the decomposition (D.4) --- again knowing what to expect from Lecture 14. If $Y$ is in $\mathfrak{g}_\lambda(H)$ with $\lambda \neq 0$, then $\mathrm{ad}(Y)$ maps each eigenspace to a different eigenspace (by Exercise D.5), as does $\mathrm{ad}(X)$ for $X \in \mathfrak{g}_0(H)$. The trace of such an endomorphism is zero, i.e., $B(X, Y) = 0$ for such $X$ and $Y$.

Because $\mathfrak{g}$ is semisimple, $B$ is nondegenerate. Since we have shown that $\mathfrak{g}_0(H)$ is perpendicular to the other weight spaces, it follows that the restriction of $B$ to $\mathfrak{g}_0(H)$ is nondegenerate.

Consider the Jordan decomposition $X = X_s + X_n$ of an element $X$ in $\mathfrak{g}_0(H)$. Since $\mathrm{ad}(X_n) = \mathrm{ad}(X)_n$ is nilpotent and semisimple on $\mathfrak{g}_0(H)$, so it vanishes there. But this already shows that $\mathrm{ad}(X) = \mathrm{ad}(X_s) + \mathrm{ad}(X_n)$ is a nilpotent endomorphism of $\mathfrak{g}_0(H)$. Hence, by Engel's theorem, $\mathfrak{g}_0(H)$ has a basis in which the endomorphisms $\mathrm{ad}(X)$ are upper-triangular for all $X \in \mathfrak{g}_0(H)$. In particular, for $X, Y, Z \in \mathfrak{g}_0(H)$, $[X, Y] = 0$, and $\mathfrak{g}_0(H)$ is abelian. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(D.7)</span></p>

If $H$ is regular, then any element of $\mathfrak{g}_0(H)$ is semisimple.

</div>

**Proof.** We saw that if $X$ is in $\mathfrak{g}_0(H)$ then $X_n$ is also. Using the same basis as in the preceding proof, we see that $\mathrm{ad}(X_n)$ has a strictly upper-triangular matrix. Hence, $B(X_n, Y) = \mathrm{Tr}(\mathrm{ad}(X_n) \circ \mathrm{ad}(Y)) = 0$ for all $Y$ in $\mathfrak{g}_0(H)$. By the nondegeneracy again, $X_n = 0$, as required. $\square$

It follows from Lemma D.6 that if $H$ is regular, and $X$ is in $\mathfrak{g}_0(H)$, then $\mathfrak{g}_0(X)$ contains $\mathfrak{g}_0(H)$, and they are equal exactly when $X$ is also regular. To finish the proof of the proposition we must prove the following lemma, which also shows that the temporary definition of regular agrees with the first one:

### D.2 On the Structure of Semisimple Lie Algebras

Let $\mathfrak{h}$ be a Cartan subalgebra of a semisimple Lie algebra $\mathfrak{g}$. Under the adjoint representation it consists of commuting semisimple endomorphisms. It is then a standard linear algebra fact that this action is simultaneously diagonalizable:

$$\mathfrak{g} = \bigoplus \mathfrak{g}_\alpha, \tag{D.10}$$

where the eigenspaces are parametrized by some set of linear forms $\alpha \in \mathfrak{h}^*$, including $\alpha = 0$, and where

$$\mathfrak{g}_\alpha = \lbrace X \in \mathfrak{g}\colon [H, X] = \alpha(H) \cdot X \text{ for all } H \in \mathfrak{h}\rbrace.$$

In particular, $\mathfrak{g}_0$ is the centralizer of $\mathfrak{h}$ in $\mathfrak{g}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(D.11)</span></p>

$\mathfrak{h} = \mathfrak{g}_0$.

</div>

The nonzero $\alpha$ are called **roots**. As before, we have $[\mathfrak{g}_\alpha, \mathfrak{g}_\beta] \subset \mathfrak{g}_{\alpha+\beta}$. It follows that if $\alpha + \beta \neq 0$, then $B(\mathfrak{g}_\alpha, \mathfrak{g}_\beta) = 0$, and if $X \in \mathfrak{g}_\alpha$ and $Y \in \mathfrak{g}_\beta$, then $\mathrm{ad}(X) \circ \mathrm{ad}(Y)$ is nilpotent, so its trace is zero, i.e.,

$$\text{If } \alpha + \beta \neq 0, \text{ then } B(\mathfrak{g}_\alpha, \mathfrak{g}_\beta) = 0. \tag{D.12}$$

Now for any root $\alpha$, if $-\alpha$ were not a root, this implies $\mathfrak{g}_\alpha$ is perpendicular to all $\mathfrak{g}_\beta$ (including $\beta = 0$), which would contradict the nondegeneracy of $B$. So we get one of the assertions in Lecture 14:

$$\text{If } \alpha \text{ is a root, then } -\alpha \text{ is also a root.} \tag{D.13}$$

Moreover, the pairing $B\colon \mathfrak{g}_\alpha \times \mathfrak{g}_{-\alpha} \to \mathbb{C}$ is nondegenerate. Another fact also follows easily:

$$\text{The roots } \alpha \text{ span } \mathfrak{h}^*. \tag{D.14}$$

For if not there would be a nonzero $X \in \mathfrak{h}$ with $\alpha(X) = 0$ for all roots $\alpha$, which means that $[X, Y] = 0$ for all $Y$ in all $\mathfrak{g}_\alpha$. But then $X$ is in the center of $\mathfrak{g}$, which is zero by semisimplicity of $\mathfrak{g}$.

Now let $\alpha$ be a root, let $X \in \mathfrak{g}_\alpha$, $Y \in \mathfrak{g}_{-\alpha}$, and take any $H \in \mathfrak{h}$. Then

$$B(H, [X, Y]) = B([H, X], Y) = \alpha(H) B(X, Y). \tag{D.15}$$

This cannot be zero for all $H$, $X$, and $Y$ without contradicting what we have just proved. In particular,

$$\text{For any root } \alpha, \; [\mathfrak{g}_\alpha, \mathfrak{g}_{-\alpha}] \neq 0. \tag{D.16}$$

Let $T_\alpha \in \mathfrak{h}$ be the element dual to $\alpha$ via the pairing $B$ on $\mathfrak{h}$, i.e., characterized by the identity $B(T_\alpha, H) = \alpha(H)$ for all $H$ in $\mathfrak{h}$. We claim next that

$$[X, Y] = B(X, Y) T_\alpha \quad \text{for all } X \in \mathfrak{g}_\alpha, \; Y \in \mathfrak{g}_{-\alpha}. \tag{D.17}$$

To see it, pair both sides with an arbitrary element $H$ of $\mathfrak{h}$. Using (D.15), $B(H, B(X,Y)T_\alpha) = \alpha(H)B(X,Y) = B(H, [X,Y])$, as required.

Furthermore:

$$\alpha(T_\alpha) \neq 0. \tag{D.18}$$

Suppose this were false. Choose $X \in \mathfrak{g}_\alpha$, $Y \in \mathfrak{g}_{-\alpha}$ such that $B(X, Y) = c \neq 0$. Then $[X, Y] = cT_\alpha$. If $\alpha(T_\alpha) = 0$, $\mathfrak{s}$ is solvable. Since $[X, Y] \in \mathscr{D}\mathfrak{s}$, it follows that $\mathrm{ad}([X, Y])$ is a nilpotent endomorphism of $\mathfrak{g}$. But $T_\alpha$ is in $\mathfrak{h}$ and all elements of $\mathfrak{h}$ are semisimple, so $T_\alpha = 0$, a contradiction. This gives another claim from Lecture 14.

$$\text{For any root } \alpha, \; [[\mathfrak{g}_\alpha, \mathfrak{g}_{-\alpha}], \mathfrak{g}_\alpha] \neq 0. \tag{D.19}$$

The last remaining fact about root spaces left unproved from Lecture 14 is

$$\text{For any root } \alpha, \; \mathfrak{g}_\alpha \text{ is one-dimensional.} \tag{D.20}$$

By what we have seen, we can find $X \in \mathfrak{g}_\alpha$, $Y \in \mathfrak{g}_{-\alpha}$, so that $[X, Y] \neq 0$, and $\alpha(H) \neq 0$. Adjusting by scalars, they generate a subalgebra $\mathfrak{s}$ isomorphic to $\mathfrak{sl}_2\mathbb{C}$, with standard basis $H$, $X$, $Y$, so in particular $\alpha(H) = 2$. Consider the adjoint action of $\mathfrak{s}$ on the sum $V = \mathfrak{h} \oplus \bigoplus_{k \neq 0} \mathfrak{g}_{k\alpha}$, the sum over all nonzero complex multiples $k\alpha$ of $\alpha$. From what we know about the weights of representations of $\mathfrak{s}$, the only $k$ that can occur are integral multiples of $\frac{1}{2}$.

Now $\mathfrak{s}$ acts trivially on $\mathrm{Ker}(\alpha) \subset \mathfrak{h}$, and it acts irreducibly on $\mathfrak{s} \subset V$. Together these cover the zero weight space $\mathfrak{h}$, since $H$ is not in $\mathrm{Ker}(\alpha)$. So the only even weights occurring can be 0 and $\pm 2$. In particular,

$$2\alpha \text{ cannot be a root.} \tag{D.21}$$

But this implies that $\frac{1}{2}\alpha$ cannot be a root, which says that 1 is not a weight occurring in $V$, i.e., there can be no other representations occurring in $V$, i.e., $V = \mathrm{Ker}(\alpha) \oplus \mathfrak{s}$, which proves (D.20). $\square$

### D.3 The Conjugacy of Cartan Subalgebras

We show that any two Cartan subalgebras are conjugate by an inner automorphism of the adjoint subgroup of $\mathrm{Aut}(\mathfrak{g})$. Fix one Cartan subalgebra $\mathfrak{h}$, and consider the decomposition (D.10). For any element $X$ in a root space $\mathfrak{g}_\alpha$, $\mathrm{ad}(X) \in \mathfrak{gl}(\mathfrak{g})$ is nilpotent, as we have seen, so its exponential $\exp(\mathrm{ad}(X)) \in \mathrm{GL}(\mathfrak{g})$ is just a finite polynomial in $\mathrm{ad}(X)$. Set

$$e(X) = \exp(\mathrm{ad}(X)).$$

Let $E(\mathfrak{h})$ be the subgroup of $\mathrm{Aut}(\mathfrak{g})$ generated by all such $e(X)$. We want to prove now that this group is independent of the choice of $\mathfrak{h}$, and that all Cartan subalgebras are conjugate by elements in this group.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(D.22 — Conjugacy of Cartan Subalgebras)</span></p>

Let $\mathfrak{h}$ and $\mathfrak{h}'$ be two Cartan subalgebras of $\mathfrak{g}$. Then (i) $E(\mathfrak{h}) = E(\mathfrak{h}')$, and (ii) there is an element $g \in E = E(\mathfrak{h})$ so that $g(\mathfrak{h}) = \mathfrak{h}'$.

</div>

**Proof.** Fix a Cartan subalgebra $\mathfrak{h}$. Let $\alpha_1, \ldots, \alpha_r$ be its roots. Consider the mapping

$$F\colon \mathfrak{g}_{\alpha_1} \times \cdots \times \mathfrak{g}_{\alpha_r} \times \mathfrak{h} \to \mathfrak{g}$$

defined by $F(X_1, \ldots, X_r, H) = e(X_1) \circ \cdots \circ e(X_r)(H)$. Note that $F$ is a polynomial mapping from one complex vector space to another of the same dimension. We want to show that not only is the image of $F$ dense, but that, if $\mathfrak{h}_{\mathrm{reg}}$ denotes the set of regular elements in $\mathfrak{h}$, then

$$F(\mathfrak{g}_{\alpha_1} \times \cdots \times \mathfrak{g}_{\alpha_r} \times \mathfrak{h}_{\mathrm{reg}}) \text{ contains a Zariski open set,} \tag{D.23}$$

i.e., it contains the complement of a hypersurface defined by a polynomial equation.

Suppose that this claim is proved. It follows that for any other Cartan subalgebra $\mathfrak{h}'$, the corresponding image also contains a Zariski open set. But two nonempty Zariski open sets always meet. This means $E(\mathfrak{h}) \cdot \mathfrak{h}_{\mathrm{reg}}$ meets $E(\mathfrak{h}') \cdot \mathfrak{h}'_{\mathrm{reg}}$. That is, there are $g \in E(\mathfrak{h})$, $H \in \mathfrak{h}_{\mathrm{reg}}$, $g' \in E(\mathfrak{h}')$, $H' \in \mathfrak{h}'_{\mathrm{reg}}$ such that $g(H) = g'(H')$. But then since $H$ and $H'$ are regular,

$$g(\mathfrak{h}) = g(\mathfrak{g}_0(H)) = \mathfrak{g}_0(g(H)) = \mathfrak{g}_0(g'(H')) = g'(\mathfrak{h}').$$

This proves the conjugacy of $\mathfrak{h}$ and $\mathfrak{h}'$. And since $E(\mathfrak{h}) = g E(\mathfrak{h}) g^{-1} = E(g(\mathfrak{h})) = E(g'(\mathfrak{h}')) = g' E(\mathfrak{h}') (g')^{-1} = E(\mathfrak{h}')$, both statements of the theorem are proved. $\square$

To prove (D.23), we use a special case of a very general fact from basic algebraic geometry: if $F\colon \mathbb{C}^N \to \mathbb{C}^N$ is a polynomial mapping whose derivative $dF_*\vert_P$ is invertible at some point $P$, then for any nonempty Zariski open set $U \subset \mathbb{C}^N$, $F(U)$ contains a nonempty Zariski open set. So it suffices to show that $dF_*\vert_P$ is surjective at a point $P = (0, \ldots, 0, H)$, where $H \in \mathfrak{h}_{\mathrm{reg}}$. This is a simple calculation: $dF_*\vert_P(0, \ldots, 0, Z) = Z$ for $Z \in \mathfrak{h}$, and $dF_*\vert_P(0, \ldots, 0, Y, 0, \ldots, 0, 0) = \mathrm{ad}(Y)(H) = -\mathrm{ad}(H)(Y)$ for $Y \in \mathfrak{g}_\alpha$. Since $\mathrm{ad}(H)$ is invertible on each root space (as $H$ is regular), $dF_*\vert_P$ is surjective. $\square$

### D.4 On the Weyl Group

In this section we complete the proofs of some of the general facts about the Weyl group that were stated in Lectures 14 and 21. The notation will be as in those sections: $\mathbb{E}$ is the real space generated by the roots $R$; $\mathfrak{W}$ is the Weyl group, generated by the involutions $W_\alpha$ of $\mathbb{E}$ determined by

$$W_\alpha(\beta) = \beta - \beta(H_\alpha)\alpha = \beta - 2\frac{(\beta, \alpha)}{(\alpha, \alpha)}\alpha,$$

where $(\,,\,)$ denotes the Killing form (or any inner product invariant for the Weyl group). We consider a decomposition

$$R = R^+ \cup R^-$$

into positive and negative roots, given by some $l\colon \mathbb{E} \to \mathbb{R}$ as in Lecture 14, and we let $S \subset R^+$ be the set of simple roots for this decomposition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(D.25)</span></p>

If $\alpha$ is a simple root, then $W_\alpha$ permutes all the other positive roots, i.e., $W_\alpha$ maps $R^+ \setminus \lbrace\alpha\rbrace$ to itself.

</div>

**Proof.** This follows from the expression of positive roots as sums $\beta = \sum m_i \alpha_i$, with the $m_i$ non-negative integers. If $\alpha = \alpha_i$, $W_\alpha(\beta)$ differs from $\beta$ only by an integral multiple of $\alpha_i$. If $\beta \neq \alpha_i$, $W_\alpha(\beta)$ still has some positive coefficients, so it must be a positive root. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(D.26)</span></p>

Any root $\beta$ can be written in the form $\beta = W(\alpha)$ for some $\alpha \in S$ and $W \in \mathfrak{W}_0$, where $\mathfrak{W}_0$ is the subgroup of $\mathfrak{W}$ generated by the $W_\alpha$ as $\alpha$ varies over the simple roots. In particular, $R = \mathfrak{W}_0(S)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(D.27)</span></p>

The Weyl group is generated by the reflections in the simple roots, i.e., $\mathfrak{W} = \mathfrak{W}_0$.

</div>

**Proof.** Given a root $\beta$, we must show that $W_\beta$ is in $\mathfrak{W}_0$. By the preceding lemma, write $\beta = U(\alpha)$, $\alpha \in S$. Then $W_\beta = W_{U(\alpha)} = U \cdot W_\alpha \cdot U^{-1}$, since both sides act the same on $\beta$ and $\beta^\perp$. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(D.29 — Transitivity on Decompositions)</span></p>

The Weyl group acts simply transitively on the set of decompositions of $R$ into positive and negative roots.

</div>

**Proof.** For the transitivity, suppose $R = Q^+ \cup Q^-$ is another decomposition. We induct on the number of roots that are in $R^+$ but not in $Q^+$. If this number is zero, then $R^+ = Q^+$. Otherwise there must be some simple root $\alpha$ that is not in $Q^+$. It suffices to prove that $W_\alpha(Q^+)$ has more roots in common with $R^+$ than $Q^+$ does, for then by induction we can write $W_\alpha(Q^+) = W(R^+)$ for some $W \in \mathfrak{W}$, so $Q^+ = W_\alpha W(R^+)$, as required. In fact, we have, by Lemma D.25, $W_\alpha(Q^+ \cap R^+) \supset W_\alpha(Q^+ \cap R^+ \setminus \lbrace\alpha\rbrace) = Q^+ \cap R^+ \setminus \lbrace\alpha\rbrace \cup \lbrace-\alpha\rbrace$.

For simple transitivity, we must show that if an element $W$ in the Weyl group takes $R^+$ to itself, then it must be the identity. If not, write $W$ as a product of reflections in simple roots, $W = W_1 \cdots W_r$, with $W_i$ the reflection in the simple root $\beta_i$. Let $\alpha = \beta_r$. It suffices to show that

$$W_1 \cdots W_r = W_1 \cdots W_{s-1} W_{s+1} \cdots W_{r-1}$$

for some $s$, $1 \le s \le r - 2$. Let $U_s = W_{s+1} \cdots W_{r-1}$. This equation is equivalent to the equation $W_s U_s W_r = U_s$, or $U_s W_r U_s^{-1} = W_s$, or $U_s(\alpha) = \beta_s$ (since by (D.28), $W_\beta = U W_\alpha U^{-1}$).

To finish the proof we must find an $s$ so that $U_s(\alpha) = \beta_s$. Note that $U_{r-1}(\alpha) = W_{r-1}(\alpha)$ is a positive root (by Lemma D.25, since $\beta_{r-1} \neq \alpha$). On the other hand, $U_0(\alpha) = W_1 \cdots W_{r-1}(\alpha) = -W(\alpha)$ is a negative root. So there must be some $s$ with $1 \le s \le r - 2$ such that $U_s(\alpha)$ is positive and $U_{s-1}(\alpha)$ is negative. This means that $W_s$ takes the positive root $U_s(\alpha)$ to the negative root $U_{s-1}(\alpha)$. But by Lemma D.25 again, this can happen only if $W_s$ is the reflection in the root $U_s(\alpha)$, i.e., $\beta_s = U_s(\alpha)$. $\square$

The simple roots $S$ for a decomposition $R = R^+ \cup R^-$ are called a **basis** for the roots. Since $S$ and $R^+$ determine each other, the proposition is equivalent to the assertion that *the Weyl group acts simply transitively on the set of bases*.

#### Weyl Chambers

If $\Omega_\alpha$ denotes the hyperplane in $\mathbb{E}$ perpendicular to the root $\alpha$, the (closed) **Weyl chambers** are the closures of the connected components of the complement $\mathbb{E} \setminus \bigcup \Omega_\alpha$ of these hyperplanes. For a decomposition $R = R^+ \cup R^-$ with simple roots $S$, the set

$$\mathscr{W} = \lbrace \beta \in \mathbb{E}\colon (\beta, \alpha) \ge 0, \;\forall \alpha \in R^+\rbrace = \lbrace \beta \in \mathbb{E}\colon (\beta, \alpha) \ge 0, \;\forall \alpha \in S\rbrace$$

is one of these Weyl chambers.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(D.31)</span></p>

For any $\beta$ in $\mathbb{E}$ there is some $W \in \mathfrak{W}$ such that $(W(\beta), \alpha) \ge 0$ for all $\alpha \in S$.

</div>

Thus, the orbit of one Weyl chamber by the Weyl group covers $\mathbb{E}$, so all Weyl chambers are conjugate to each other by the action of the Weyl group. So all arise by partitioning $R$ into positive and negative roots. This partitioning is uniquely determined by the Weyl chamber. In fact, the walls of the Weyl chamber are the hyperplanes $\Omega_\alpha$ as $\alpha$ varies over the $n$ corresponding simple roots, $n = \dim(\mathbb{E})$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(D.32)</span></p>

The Weyl group acts simply transitively on Weyl chambers.

</div>

#### Coroots, Weight Lattice, and Fundamental Weights

Our next goal is to show that the lattice $\mathbb{Z}\lbrace H_\alpha\colon \alpha \in R\rbrace \subset \mathfrak{h}$ has a basis of $H_\alpha$ where $\alpha$ varies over the simple roots. This is analogous to the statement that the root lattice $\Lambda_R$ in $\mathfrak{h}^*$ is generated by simple roots. The first statement can be deduced from the second, using the Killing form to map $\mathfrak{h}$ to $\mathfrak{h}^*$, $H \mapsto (H, -)$. We saw in Lecture 14 that this map takes $H_\alpha$ to $\alpha' = (2/(\alpha, \alpha))\alpha$. Given a root system $R$ in a Euclidean space $\mathbb{E}$, to each root $\alpha$ one can define its **coroot** $\alpha'$ in $\mathbb{E}$ by the formula

$$\alpha' = \frac{2}{(\alpha, \alpha)}\alpha.$$

Let $R' = \lbrace\alpha'\colon \alpha \in R\rbrace$ be the set of coroots. For any $0 \neq \alpha \in \mathfrak{h}$, set $\alpha' = (2/(\alpha, \alpha))\alpha$, and for any $\alpha, \beta \in \mathfrak{h}^*$, set $n_{\beta\alpha} = 2(\beta, \alpha)/(\alpha, \alpha)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(D.34)</span></p>

(i) The set $R'$ of coroots forms a root system in $\mathbb{E}$.
(ii) The set $S' = \lbrace\alpha'\colon \alpha \in S\rbrace$ is a set of simple roots for $R'$.
(iii) For $\alpha, \beta \in S$, $n_{\beta'\alpha'} = n_{\alpha\beta}$.

</div>

The root system $R'$ is called the **dual** of $R$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(D.36 — Weight Lattice and Fundamental Weights)</span></p>

(i) The elements $H_\alpha$ for $\alpha \in S$ generate the lattice $\mathbb{Z}\lbrace H_\alpha\colon \alpha \in R\rbrace$.
(ii) If $\omega_\alpha \in \mathfrak{h}$ are defined by the property that $\omega_\alpha(H_\beta) = \delta_{\alpha,\beta}$, then the elements $\omega_\alpha$ generate the **weight lattice** $\Lambda_\mathfrak{W}$.
(iii) The nonnegative integral linear combinations of the fundamental weights $\omega_\alpha$ are precisely the weights in $\mathscr{W} \cap \Lambda_\mathfrak{W}$, where $\mathscr{W}$ is the closed Weyl chamber corresponding to $R^+$.

</div>

#### The Weyl Group as Automorphisms

If we identify $\mathfrak{h}$ with $\mathfrak{h}^*$ by means of the Killing form, we can regard $\mathfrak{W}$ as a group of automorphisms of $\mathfrak{h}$. By means of this, the reflection $W_\alpha$ corresponding to a root $\alpha$ becomes the automorphism of $\mathfrak{h}$ which takes an element $H$ to $H - \alpha(H) \cdot H_\alpha$. We have a last debt (Fact 14.11) to pay about the Weyl group:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(D.37)</span></p>

Every element of the Weyl group is induced by an automorphism of $\mathfrak{g}$ which maps $\mathfrak{h}$ to itself.

</div>

**Proof.** It suffices to produce the generating involutions $W_\alpha$ in this way. The claim is that if $X_\alpha$ and $Y_\alpha$ are generators of $\mathfrak{g}_\alpha$ and $\mathfrak{g}_{-\alpha}$ as usual, then $\vartheta_\alpha = e(X_\alpha) e(-Y_\alpha) e(X_\alpha)$ is such an automorphism, where, as in the preceding section, we write $e(X)$ for $\exp(\mathrm{ad}(X))$. We must show that $\vartheta_\alpha(H) = H - \alpha(H) \cdot H_\alpha$ for all $H$ in $\mathfrak{h}$. It suffices to do this for $H$ with $\alpha(H) = 0$, and for $H = H_\alpha$, it suffices to calculate on the subalgebra $\mathfrak{s}_\alpha = \mathbb{C}\lbrace H_\alpha, X_\alpha, Y_\alpha\rbrace \cong \mathfrak{sl}_2\mathbb{C}$, and this is a simple calculation.

For $\mathfrak{sl}_2\mathbb{C}$ with its standard basis, $\vartheta = e(X)e(Y)e(X)$ maps $H$ to $-H$, $X$ to $-Y$, and $Y$ to $-X$.

We need a refinement of the preceding calculation. For a root $\alpha$ and a nonzero complex number $t$, define two automorphisms of $\mathfrak{g}$:

$$\vartheta_\alpha(t) = e(t \cdot X_\alpha) \circ e(-(t)^{-1} \cdot Y_\alpha) \circ e(t \cdot X_\alpha)$$

and

$$\Phi_\alpha(t) = \vartheta_\alpha(t) \circ \vartheta_\alpha(-1).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(D.39)</span></p>

The automorphism $\Phi_\alpha(t)$ is the identity on $\mathfrak{h}$, and for any root $\beta$, it is multiplication by $t^{\beta(H_\alpha)}$ on $\mathfrak{g}_\beta$.

</div>

**Proof.** Look first in $\mathfrak{sl}_2$, with $X = X_\alpha$, $Y = Y_\alpha$. It is simplest to calculate in the covering $\mathrm{SL}_2\mathbb{C}$ of the adjoint group. Here $\vartheta_\alpha(t)$ lifts to

$$\exp(tX) \cdot \exp(-t^{-1}Y) \cdot \exp(tX) = \begin{pmatrix} 1 & t \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ -t^{-1} & 1 \end{pmatrix} \begin{pmatrix} 1 & t \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & t \\ -t^{-1} & 0 \end{pmatrix},$$

so $\Phi_\alpha(t)$ lifts to

$$\begin{pmatrix} 0 & t \\ -t^{-1} & 0 \end{pmatrix} \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} t & 0 \\ 0 & t^{-1} \end{pmatrix}.$$

To see how $\Phi_\alpha(t)$ acts on $\mathfrak{g}_\beta$, for $\beta \neq \pm\alpha$, it suffices to consider the action of the $\mathrm{SL}_2\mathbb{C}$ corresponding to $\mathfrak{s}_\alpha = \mathbb{C}\lbrace H_\alpha, X_\alpha, Y_\alpha\rbrace$ on the $\alpha$-string through $\beta$, i.e., the sum $\bigoplus_k \mathfrak{g}_{\beta + k\alpha}$. The weights for this $\mathfrak{sl}_2$-representation are $\beta(H_\alpha), \beta(H_\alpha) \pm 2, \ldots$. The diagonal matrix $\mathrm{diag}(t, t^{-1})$ acts on the weight space of weight $m$ by $t^m$. In particular, on $\mathfrak{g}_\beta$ (weight $\beta(H_\alpha)$) it acts by $t^{\beta(H_\alpha)}$. $\square$

---

## Appendix E: Ado's and Levi's Theorems

### E.1 Levi's Theorem

The object of this section is to prove Levi's theorem:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(E.1 — Levi's Theorem)</span></p>

Let $\mathfrak{g}$ be a Lie algebra with radical $\mathfrak{r}$. Then there is a subalgebra $\mathfrak{l}$ of $\mathfrak{g}$ such that $\mathfrak{g} = \mathfrak{r} \oplus \mathfrak{l}$.

</div>

**Proof.** There are several simple reductions. First, we may assume there is no nonzero ideal of $\mathfrak{g}$ that is properly contained in $\mathfrak{r}$. For if $\mathfrak{a}$ were such an ideal, by induction on the dimension of $\mathfrak{g}$, $\mathfrak{g}/\mathfrak{a}$ would have a subalgebra complementary to $\mathfrak{r}/\mathfrak{a}$, and this subalgebra has the form $\mathfrak{l}/\mathfrak{a}$, with $\mathfrak{l}$ as required. In particular, we may assume $\mathfrak{r}$ is abelian, since otherwise $\mathscr{D}\mathfrak{r}$ is a proper ideal in $\mathfrak{r}$ which is an ideal in $\mathfrak{g}$ by Corollary C.23. We may also assume that $[\mathfrak{g}, \mathfrak{r}] = \mathfrak{r}$, for if $[\mathfrak{g}, \mathfrak{r}] = 0$ then the adjoint representation factors through $\mathfrak{g}/\mathfrak{r}$, and since $\mathfrak{g}/\mathfrak{r}$ is semisimple, the submodule $\mathfrak{r} \subset \mathfrak{g}$ has a complement, which is the required $\mathfrak{l}$.

Now $V = \mathfrak{gl}(\mathfrak{g})$ is a $\mathfrak{g}$-module via the adjoint representation: for $X \in \mathfrak{g}$ and $\varphi \in V$,

$$(X \cdot \varphi)(Y) = [X, \varphi(Y)] - \varphi([X, Y]). \tag{E.2}$$

The trick is to consider the following subspaces of $V$:

$$C = \lbrace\varphi \in V\colon \varphi(\mathfrak{g}) \subset \mathfrak{r} \text{ and } \varphi\vert_\mathfrak{r} \text{ is multiplication by a scalar}\rbrace$$

$$\supset\; B = \lbrace\varphi \in V\colon \varphi(\mathfrak{g}) \subset \mathfrak{r} \text{ and } \varphi(\mathfrak{r}) = 0\rbrace$$

$$\supset\; A = \lbrace\mathrm{ad}(X)\colon X \in \mathfrak{r}\rbrace.$$

These are easily checked to be $\mathfrak{g}$-submodules of $V$, included in each other as indicated. And $C/B$ is a trivial $\mathfrak{g}$-module of rank 1, i.e. $C/B = \mathbb{C}$, by taking $\varphi$ in $C$ to the scalar $\lambda$ such that $\varphi\vert_\mathfrak{r} = \lambda \cdot I$. (Note that $C/B \neq 0$ since one can find an endomorphism of $\mathfrak{g}$ which is the identity on $\mathfrak{r}$ and zero on a vector space complement to $\mathfrak{r}$.) We claim also that

$$\mathfrak{g} \cdot C \subset B \quad \text{and} \quad \mathfrak{r} \cdot C \subset A. \tag{E.3}$$

To prove these let $\varphi \in C$, and assume the restriction of $\varphi$ to $\mathfrak{r}$ is multiplication by the scalar $c$. If $X \in \mathfrak{g}$ and $Y \in \mathfrak{r}$, then by (E.2),

$$(X \cdot \varphi)(Y) = [X, cY] - c[X, Y] = 0,$$

so $X \cdot \varphi \in B$; this proves the first inclusion. If $X \in \mathfrak{r}$, and $Y \in \mathfrak{g}$, then $[X, \varphi(Y)] \in [\mathfrak{r}, \mathfrak{r}] = 0$, so

$$(X \cdot \varphi)(Y) = -\varphi([X, Y]) = [-cX, Y],$$

and $X \cdot \varphi = \mathrm{ad}(-cX)$ is in $A$, which proves the second inclusion.

This means that the map $C/A \to C/B = \mathbb{C}$ is a surjection of $\mathfrak{g}/\mathfrak{r}$-modules, which must split since $\mathfrak{g}/\mathfrak{r}$ is semisimple. In other words, there is an element $\varphi$ in $C$ such that $\varphi\vert_\mathfrak{r} = \mathrm{id}$, and $\mathfrak{g} \cdot \varphi$ is contained in $A$. Now let

$$\mathfrak{l} = \lbrace X \in \mathfrak{g}\colon X \cdot \varphi = 0\rbrace.$$

It is easy to check that $\mathfrak{l}$ is a subalgebra of $\mathfrak{g}$: (i) $\mathfrak{l} \cap \mathfrak{r} = 0$; and (ii) $\mathfrak{g} = \mathfrak{l} + \mathfrak{r}$. For the first, if $X$ is a nonzero element of the intersection, then, as we saw above, $X \cdot \varphi = \mathrm{ad}(-X)$, so $\mathrm{ad}(X) = 0$. Hence $\mathbb{C} \cdot X$ is a nonzero ideal in $\mathfrak{r}$, contradicting our assumptions. For (ii), let $X \in \mathfrak{g}$. Then $X \cdot \varphi$ is in $A$, so $X \cdot \varphi = \mathrm{ad}(Y)$ for some $Y$ in $\mathfrak{r}$. We saw that $\mathrm{ad}(Y) = -Y \cdot \varphi$, so $(X + Y) \cdot \varphi = 0$, i.e., $X + Y$ belongs to $\mathfrak{l}$. Hence $X = (X + Y) - Y$ is in the sum of $\mathfrak{l}$ and $\mathfrak{r}$. $\square$

This proves the existence of Levi subalgebras $\mathfrak{l}$ of any Lie algebra. We have no need to prove the companion fact that any two Levi subalgebras are conjugate, cf. [Bour, I, $\S$6.8].

### E.2 Ado's Theorem

The goal is Ado's theorem that every Lie algebra is linear, i.e., is a subalgebra of $\mathfrak{gl}(V)$ for some vector space $V$, which is the same as saying it has a finite-dimensional faithful representation. As in the previous section, there are some easy steps, and then a clever argument is needed to create an appropriate representation.

We start, of course, with the adjoint representation, which is about the only representation we have for an abstract Lie algebra $\mathfrak{g}$. Since the kernel of the adjoint representation is the center $\mathfrak{c}$ of $\mathfrak{g}$, it suffices to find a representation of $\mathfrak{g}$ which is faithful on $\mathfrak{c}$. For then the sum of this representation and the adjoint representation is a faithful representation of $\mathfrak{g}$.

The abelian Lie algebra $\mathfrak{c}$ has a faithful representation by nilpotent matrices. For example, when $\mathfrak{c} = \mathbb{C}$ is one dimensional, one can take the representation $\lambda \mapsto \begin{pmatrix} 0 & \lambda \\ 0 & 0 \end{pmatrix}$; in general a direct sum of such representations will suffice.

We can choose a sequence of subalgebras

$$\mathfrak{c} = \mathfrak{g}_0 \subset \mathfrak{g}_1 \subset \cdots \subset \mathfrak{g}_p = \mathfrak{n} \subset \mathfrak{g}_{p+1} \subset \cdots \subset \mathfrak{g}_q = \mathfrak{r} \subset \mathfrak{g}_{q+1} = \mathfrak{g},$$

each an ideal in the next, with $\mathfrak{n} = \mathrm{Nil}(\mathfrak{g})$ the largest nilpotent ideal of $\mathfrak{g}$, and $\mathfrak{r} = \mathrm{Rad}(\mathfrak{g})$ the largest solvable ideal; as in $\S$9.1 we may assume $\dim(\mathfrak{g}_i/\mathfrak{g}_{i-1}) = 1$ for $i \le q$. The plan is to start with a faithful representation of $\mathfrak{g}_0$, and construct successively representations of each $\mathfrak{g}_i$ which are faithful on $\mathfrak{c}$. Similarly to go from $\mathfrak{r}$ to $\mathfrak{g}$, use Levi's theorem to write $\mathfrak{g} = \mathfrak{r} \oplus \mathfrak{h}$ for a semisimple subalgebra $\mathfrak{h}$.

Call a representation $\rho$ of a Lie algebra $\mathfrak{g}$ a **nilrepresentation** if $\rho(X)$ is a nilpotent endomorphism for every $X$ in $\mathrm{Nil}(\mathfrak{g})$. A stronger version of Ado's theorem is:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(E.4 — Ado's Theorem)</span></p>

Every Lie algebra has a faithful finite-dimensional nilrepresentation.

</div>

The crucial step is:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(E.5)</span></p>

Let $\mathfrak{g}$ be a Lie algebra which is a direct sum of a solvable ideal $\mathfrak{a}$ and a subalgebra $\mathfrak{h}$. Let $\sigma$ be a nilrepresentation of $\mathfrak{a}$. Then there is a representation $\rho$ of $\mathfrak{g}$ such that

$$\mathfrak{h} \cap \mathrm{Ker}(\rho) \subset \mathrm{Ker}(\sigma).$$

If $\mathrm{Nil}(\mathfrak{g}) = \mathrm{Nil}(\mathfrak{a})$ or $\mathrm{Nil}(\mathfrak{g}) = \mathfrak{g}$, then $\rho$ may be taken to be a nilrepresentation.

</div>

Ado's theorem follows readily from this proposition. Starting with a faithful representation $\rho_0$ of $\mathfrak{c} = \mathfrak{g}_0$ by nilpotent matrices, one uses the proposition to construct successively nilrepresentations $\rho_i$ of $\mathfrak{g}_i$. The displayed condition assures that they are all faithful on $\mathfrak{c}$. Note that if $i \le p$, $\mathrm{Nil}(\mathfrak{g}_i) = \mathfrak{g}_i$, while if $i > p$ we have $\mathrm{Nil}(\mathfrak{g}_i) = \mathrm{Nil}(\mathfrak{g}_{i-1}) = \mathfrak{n}$ by Corollary C.26, so the hypotheses assure that all representations can be taken to be nilrepresentations. $\square$

**Proof of Proposition E.5.** Suppose $\mathfrak{g} = \mathfrak{a} \oplus \mathfrak{h}$ is a Lie algebra which is a direct sum of an ideal $\mathfrak{a}$ and a subalgebra $\mathfrak{h}$. Let $U = U(\mathfrak{a})$ be the universal enveloping algebra of $\mathfrak{a}$. Any $Y$ in $\mathfrak{a}$ determines a linear endomorphism $L_Y$ of $U$, which is simply left multiplication by the image of $Y$ in $U$. Any $X$ in $\mathfrak{g}$ determines an inner derivation $Y \mapsto [X, Y]$ of $\mathfrak{a}$; let $D_X$ be the corresponding derivation of $U$, cf. Lemma C.27. For each $X$ in $\mathfrak{g}$ we define a linear mapping $T_X\colon U \to U$ by writing $X = Y + Z$ with $Y \in \mathfrak{a}$ and $Z \in \mathfrak{h}$, and setting

$$T_X = L_Y + D_Z.$$

A straightforward calculation shows that

$$T_{[X_1, X_2]} = T_{X_1} \circ T_{X_2} - T_{X_2} \circ T_{X_1}. \tag{E.6}$$

If $\mathfrak{gl}(U)$ denotes the infinite-dimensional Lie algebra of endomorphisms of $U$, with the usual bracket $[A, B] = A \circ B - B \circ A$, this means that the mapping $\mathfrak{a} \to \mathfrak{gl}(U)$, $X \mapsto T_X$, is a homomorphism of Lie algebras.

Suppose $\sigma\colon \mathfrak{a} \to \mathfrak{gl}(V)$ is a finite-dimensional representation of $\mathfrak{a}$. Let $\tilde{\sigma}\colon U \to \mathrm{End}(V)$ be the corresponding homomorphism of algebras, as in $\S$C.3, and let $I$ be the kernel of $\tilde{\sigma}$. The basic step is:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(E.7)</span></p>

Assume that $\mathfrak{a}$ is solvable. Suppose $I$ is an ideal of $U = U(\mathfrak{a})$ satisfying the following two properties: (i) $U/I$ is finite dimensional; (ii) the image of every element of $\mathrm{Nil}(\mathfrak{a})$ in $U/I$ is nilpotent. Then there is an ideal $J \subset I$ of $U$, and also (iii) for every derivation $D$ of $\mathfrak{a}$, the corresponding derivation of $U$ maps $J$ into itself.

</div>

Granting this lemma, we prove Proposition E.5 as follows. From the representation $\sigma$ we constructed an ideal $I$ whose existence is asserted in the lemma. By (E.6), the mapping $X \mapsto \bar{T}_X$ is a homomorphism of Lie algebras from $\mathfrak{g}$ to $\mathfrak{gl}(U/J)$. Note that if $X$ is in $\mathfrak{a}$, then $\bar{T}_X$ is just left multiplication $X$ on $U/J$. By (E.6), the mapping $X \mapsto \bar{T}_X$ is the representation $\rho$ required in the proposition.

We first verify that $\mathrm{Ker}(\rho) \cap \mathfrak{a} \subset \mathrm{Ker}(\sigma)$. Note that if $X$ is in $\mathfrak{a}$, then $\bar{T}_X$ is just left multiplication on $U/J$, so the image of $X$ in $U$ must be in $J$; since $J \subset I \subset \mathrm{End}(V)$, this means $\sigma(X) = 0$, as required.

It remains to show that, under either of the additional hypotheses, $\rho$ is a nilrepresentation. Note first that each $X$ in $\mathfrak{a}$ acts on $U/J$ by left multiplication, and if $X$ is in $\mathrm{Nil}(\mathfrak{a})$, by (ii) its image in $U/I$ is nilpotent. Thus $\rho(X)$ is nilpotent for every $X$ in $\mathrm{Nil}(\mathfrak{a})$. In particular, this shows that $\rho$ is a nilrepresentation when $\mathrm{Nil}(\mathfrak{g}) = \mathrm{Nil}(\mathfrak{a})$.

In the other case, $\mathfrak{g}$ is nilpotent, so $\mathfrak{a}$ is also nilpotent. By the Leibnitz rule for derivations, it follows that the corresponding derivation $D_Z$ of $U$ is nilpotent on any element, although the power required to annihilate an element may be unbounded. However, since $U/J$ is finite dimensional, $\rho(Z)$ is nilpotent for every $Z$ in $\mathfrak{h}$. By the preceding paragraph, and choose $l$ so that $\rho(Z)^l = 0$. It follows that $\rho(X)^{kl}$ vanishes, since, when the latter is expanded, each summand either has $\rho(Y)$ occurring at least $k$ times, or else $\rho(Z)^l$ occurs somewhere in the product. $\square$

**Proof of Lemma E.7.** Let $Q$ be the two-sided ideal in the algebra $U/I$ generated by the image of $\mathrm{Nil}(\mathfrak{a})$. Since $U/I$ is generated by the image of $\mathfrak{a}$, the same argument as in the proof before last shows that $Q^k = Q \cdots Q = 0$ for some $k$. Write $Q = K/I$ for an ideal $K$ of $U$, and set $J = K^k$. Clearly $J \subset I$, and we claim that $J$ satisfies the conditions (i)--(iii) of the lemma.

To see that $J$ has finite codimension, let $x_1, \ldots, x_n$ be a basis for the image of $\mathfrak{a}$ in $U$, and choose monic polynomials $p_i$ such that $p_i(x_i)$ is in $K$; this is possible since $U/K$ is finite dimensional. Therefore, $p_i(x_i)^k$ is in $J$, so the images of the $x_i$ satisfy monic equations in $U/J$. Since $U$ is generated by the monomials $x_1^{i_1} \cdots x_n^{i_n}$, it follows readily that $U/J$ is spanned by a finite number of these elements.

Property (ii) is clear from the construction, for if $x \in U$ is the image of an element of $\mathrm{Nil}(\mathfrak{a})$, by (ii) its image in $U/I$ is nilpotent, and if $X$ is in $\mathrm{Nil}(\mathfrak{a})$, some power $x^p$ is in $I^k \subset K^k = J$.

For (iii), if $D$ is a derivation of $\mathfrak{a}$, since $\mathfrak{a}$ is solvable, it follows from Proposition C.24 that $D$ maps $\mathfrak{a}$ into $\mathrm{Nil}(\mathfrak{a})$. The corresponding derivation of $U$ therefore maps $U$ into $K$, from which it follows that it maps $J = K^k$ to itself. $\square$

As before, the results of this section also apply to real Lie algebras: if $\mathfrak{g}$ is real, a faithful representation (complex) of $\mathfrak{g} \otimes \mathbb{C}$ is automatically a faithful real representation, and embeds $\mathfrak{g}$ in some $\mathfrak{gl}_n\mathbb{R}$.

---

## Appendix F: Invariant Theory for the Classical Groups

The object is to derive just enough classical invariant theory for the classical groups to verify the claims made in the text. We follow a classical, constructive approach, using an identity of Capelli.

### F.1 The Polynomial Invariants

Let $V = \mathbb{C}^n$, regarded as the standard representation of $\mathrm{GL}_n\mathbb{C}$, so of any of the subgroups $G = \mathrm{SL}_n\mathbb{C}$, $\mathrm{O}_n\mathbb{C}$, $\mathrm{SO}_n\mathbb{C}$, or $\mathrm{Sp}_n\mathbb{C}$ (for $n$ even); $e_1, \ldots, e_n$ denotes a standard basis for $V$, compatible with one of the standard realizations of $G$. The goal is to find those polynomials $F(x^{(1)}, \ldots, x^{(m)})$ of $m$ variables on $V$ which are invariant by $G$. For example, if $Q\colon V \otimes V \to \mathbb{C}$ is the bilinear form determining the orthogonal or symplectic group, the polynomials $Q(x^{(i)}, x^{(j)})$ are invariants. In addition, if $G$ is a subgroup of $\mathrm{SL}(V)$, the **bracket** $[x^{(1)}\; x^{(2)} \cdots x^{(n)}]$, given by the determinant,

$$[x^{(1)}\; x^{(2)} \cdots x^{(n)}] = \det(x_j^{(i)}), \tag{F.1}$$

is an invariant of $G$. The **first fundamental theorem** of invariant theory for these groups asserts that any invariant is a polynomial function of these basic invariants. This is the goal of this appendix.

#### Polarization Operators

We denote by $S^d$ the homogeneous polynomial functions of degree $d$ on $V$, i.e., $S^d = \mathrm{Sym}^d(V^*)$. For an $m$-tuple $\mathbf{d} = (d_1, \ldots, d_m)$ of non-negative integers, let $S^\mathbf{d} = S^{d_1} \otimes \cdots \otimes S^{d_m}$ be the polynomials on $V^{\oplus m}$ which are homogeneous of degree $d_i$ in the $i$th variable. We write $F(x^{(1)}, \ldots, x^{(m)})$ for such a polynomial.

For integers $i$ and $j$ between 1 and $m$ there is a canonical "polarization" map $D_{ij}$ which takes a polynomial $F$ of $m$ variables to the polynomial

$$D_{ij}(F) = \sum_{k=1}^{n} x_k^{(i)} \frac{\partial F}{\partial x_k^{(j)}}. \tag{F.2}$$

This operator lowers the $j$th degree by 1, while it increases the $i$th degree by 1, i.e., it maps $S^\mathbf{d}$ to $S^{\mathbf{d}'}$, where $\mathbf{d}'$ is the same sequence of multi-indices as $\mathbf{d}$, but with $d_j' = d_j - 1$ and $d_i' = d_i + 1$. Note also that these $D_{ij}$ are derivations:

$$D_{ij}(F_1 \cdot F_2) = D_{ij}(F_1) \cdot F_2 + F_1 \cdot D_{ij}(F_2). \tag{F.3}$$

These maps may be described intrinsically in terms of the multilinear algebra of Appendix B. Since only two factors are involved, it suffices to look at the map $D_{12}$ when there are only two factors. In this case

$$D_{12}\colon S^d \otimes S^e \to S^{d+1} \otimes S^{e-1}$$

is the composite $S^d \otimes S^e \to S^d \otimes (S^1 \otimes S^{e-1}) \to S^{d+1} \otimes S^{e-1}$, where the second is the product of symmetric powers, and the first by the dual map $S^e \to S^1 \otimes S^{e-1}$ (which takes $F(x)$ to $\sum_k x_k \otimes \partial F/\partial x_k$). This shows, if there were any doubt, that the $D_{ij}$ are maps of $\mathrm{GL}(V)$-modules, i.e., they are independent of choice of coordinates.

A first idea is that, if $F$ is an invariant by a group $G \subset \mathrm{GL}(V)$, then $D_{ij}(F)$ will also be an invariant, and these invariants will be known by induction if $i < j$. If one also knew the second term in the above expression for $D_{ji} \circ D_{ij}(F)$, one could determine $e \cdot F$, which suffices to determine $F$, provided $e$ is not zero.

#### The Capelli Identity

In general, it is not evident how to proceed, but in case $\dim V = 2$, and $\mathbf{d} = (d, e)$, this can be carried through as follows. Some of the terms in the second term also occur in the expression

$$[xy] \cdot \Omega(F) = (x_1 y_2 - x_2 y_1) \left(\frac{\partial^2 F}{\partial x_1 \partial y_2} - \frac{\partial^2 F}{\partial x_2 \partial y_1}\right).$$

Comparing the preceding three formulas gives the identity

$$(d + 1)e \cdot F = D_{21} \circ D_{12}(F) + [xy] \cdot \Omega(F). \tag{F.4}$$

This plan of attack, in fact, extends to find all polynomial invariants of all the classical subgroups of $\mathrm{GL}(V)$. What is needed is an appropriate generalization of the identity (F.4). About a century ago Capelli found such an identity. The clue is to write (F.4) in the more suggestive form

$$\begin{vmatrix} D_{11} + 1 & D_{12} \\ D_{21} & D_{22} \end{vmatrix}(F) = [xy] \cdot \Omega(F),$$

where the determinant on the left is evaluated by expanding as usual, but being careful to read the composition of operators from left to right, since they do not commute.

This is the formula which generalizes. If $F$ is a function of $m$ variables from $V$, and $\dim V = m$, define, following Cayley,

$$\Omega(F) = \sum_{\sigma \in \mathfrak{S}_m} \mathrm{sgn}(\sigma) \frac{\partial^m F}{\partial x_{\sigma(1)}^{(1)} \cdots \partial x_{\sigma(m)}^{(m)}}; \tag{F.5}$$

in symbols, $\Omega$ is given by the determinant of the $m \times m$ matrix of partial derivative operators $(\partial/\partial x_j^{(i)})$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(F.6 — Capelli's Identity)</span></p>

The **Capelli identity** is the formula:

$$\begin{vmatrix} D_{11} + m - 1 & D_{12} & \cdots & D_{1m} \\ D_{21} & D_{22} + m - 2 & \cdots & D_{2m} \\ \vdots & & \ddots & \vdots \\ D_{m1} & D_{m2} & \cdots & D_{mm} \end{vmatrix} = [x^{(1)}\; x^{(2)} \cdots x^{(m)}] \cdot \Omega.$$

This is an identity of operators acting on functions $F = F(x^{(1)}, \ldots, x^{(m)})$ of $m$ variables, with $m = n = \dim V$, and as always the determinant is expanded with compositions of operators reading from left to right.

</div>

Note the important corollary: if the number of variables $m$ is greater than the dimension $n$, then

$$\begin{vmatrix} D_{11} + m - 1 & D_{12} & \cdots & D_{1m} \\ D_{21} & D_{22} + m - 2 & \cdots & D_{2m} \\ \vdots & & \ddots & \vdots \\ D_{m1} & D_{m2} & \cdots & D_{mm} \end{vmatrix}(F) = 0. \tag{F.7}$$

This follows by regarding $F$ as a function on $\mathbb{C}^m$ which is independent of the last $m - n$ coordinates. Since $\Omega(F) = 0$ for such a function, (F.7) follows from (F.6).

Let $K$ denote the operator on the left-hand side of these Capelli identities. The expansion of $K$ has a main diagonal term, the product of the diagonal entries $D_{ii} + m - i$, which are scalars on multihomogeneous functions. Note that in any other product of the expansion, the last nondiagonal term which occurs is one of the $D_{ij}$ with $i < j$. Since the diagonal terms commute with the others, we can group the products that precede a given $D_{ij}$ into one operator, so we can write, for $F \in S^\mathbf{d}$,

$$K(F) = \rho \cdot F - \sum_{i < j} P_{ij} D_{ij}(F), \tag{F.8}$$

where $\rho = (d_1 + m - 1) \cdot (d_2 + m - 2) \cdots (d_m)$, and each $P_{ij}$ is a linear combination of compositions of various $D_{ab}$. Capelli's identities say that

$$\rho \cdot F = \sum_{i < j} P_{ij} D_{ij}(F) \quad \text{if } m > n; \tag{F.8'}$$

$$\rho \cdot F = \sum_{i < j} P_{ij} D_{ij}(F) + [x^{(1)} \cdots x^{(m)}] \cdot \Omega(F) \quad \text{if } m = n. \tag{F.9}$$

Just as in the above special case, if $F$ is an invariant of a group $G$, each $D_{ij}(F)$ is also an invariant in a $S^{\mathbf{d}'}$ where we will know all such invariants by induction. If $G$ is a subgroup of $\mathrm{SL}(V)$, and $m = n$, then $\Omega(F)$ is also an invariant, as follows from the definition or Capelli's identity.

#### Invariants for $\mathrm{SL}_n\mathbb{C}$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(F.10 — Invariants of $\mathrm{SL}_n\mathbb{C}$)</span></p>

Polynomial invariants $F(x^{(1)}, \ldots, x^{(m)})$ of $\mathrm{SL}_n\mathbb{C}$ can be written as polynomials in the brackets

$$[x^{(i_1)}\; x^{(i_2)} \cdots x^{(i_n)}], \quad 1 \le i_1 < i_2 < \cdots < i_n \le m.$$

</div>

**Proof.** We must show that $F$ can be written as a polynomial in the basic bracket polynomials. In particular, if $m < n$, we must verify that there are no invariants except the constants in $S^0 = \mathbb{C}$. This is a simple consequence of the fact that for a dense open set of $m$-tuples of vectors — namely, those which are linearly independent — there is an automorphism of $\mathrm{SL}_n\mathbb{C}$ taking them to a fixed $m$-tuple of independent vectors, say $e_1, \ldots, e_m$. So an invariant function must take the same value on all such $m$-tuples. By the density, it must be constant.

For $m \ge n$, we proceed by induction as above. All $D_{ij}F$ are known to be invariants (for $i < j$), and these invariants will be known by induction. Also, $\Omega(F)$ is an $\mathrm{SL}_n\mathbb{C}$-invariant, and by induction it is a polynomial in the brackets. To complete the proof, by Capelli's identities (F.8) and (F.9), it suffices to see that the operators $D_{ab}$ all take brackets to scalar multiples of brackets. This is an obvious calculation: $D_{ab}$ takes a bracket $[x^{(i_1)} \cdots x^{(i_n)}]$ to zero if $b$ does not appear as one of the superscripts, or to the bracket with the variable $x^{(b)}$ replaced by $x^{(a)}$ if $x^{(a)}$ also occurs and $a = b$ it is zero, and is a bracket otherwise. To avoid repeats, one needs only consider brackets where the superscripts are increasing. This completes the proof. $\square$

#### Invariants for $\mathrm{Sp}_n\mathbb{C}$

Let $r = n/2$, and let $Q$ be the skew form defining the symplectic group $\mathrm{Sp}_n\mathbb{C}$, e.g. $Q(x, y) = \sum_{i=1}^{r} x_i y_{r+i} - x_{r+i} y_i$ in standard coordinates. Note first that the brackets are not needed:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(F.13 — Invariants of $\mathrm{Sp}_n\mathbb{C}$)</span></p>

Polynomial invariants $F(x^{(1)}, \ldots, x^{(m)})$ of $\mathrm{Sp}_n\mathbb{C}$ can be written as polynomials in functions

$$Q(x^{(i)}, x^{(j)}), \quad 1 \le i < j \le m.$$

</div>

The proof that any $\mathrm{Sp}_n\mathbb{C}$-invariant polynomial in $m$ variables can be written as a polynomial in the basic polynomials $Q(x^{(i)}, x^{(j)})$ uses the antilexicographic induction with the Capelli identities as before, and the verification that the operators $D_{ab}$ preserve polynomials in the basic invariants.

For $m < n$, the situation requires additional argument, exploiting the fact that the restriction of an invariant polynomial $F$ on $V = \mathbb{C}^n$ to $V' = \mathbb{C}^{n-2}$ (perpendicular to the plane spanned by $e_r$ and $e_n$) is an invariant of the subgroup $\mathrm{Sp}_{n-2}\mathbb{C}$. By induction this restriction is a polynomial in the basic invariants. Subtracting, it suffices to show that if an invariant $F$ restricts to zero on $V'$, then $F$ is identically zero. This is done by showing that the restriction to the larger subspace $W = V' \oplus \mathbb{C}e_r$ must be zero, and then that the restriction to any hyperplane of the form $g \cdot W$, $g \in \mathrm{Sp}_n\mathbb{C}$, is zero. Since $(n-1)$-tuples in such hyperplanes form an open dense subset, $F = 0$. $\square$

#### Invariants for $\mathrm{SO}_n\mathbb{C}$

This time brackets may be needed, as well as the functions given by the symmetric form $Q$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(F.15 — Invariants of $\mathrm{SO}_n\mathbb{C}$)</span></p>

Polynomial invariants $F(x^{(1)}, \ldots, x^{(m)})$ of $\mathrm{SO}_n\mathbb{C}$ can be written as polynomials in functions

$$Q(x^{(i)}, x^{(j)}) \quad \text{and} \quad [x^{(i_1)}\; x^{(i_2)} \cdots x^{(i_n)}],$$

with $1 \le i \le j \le m$, $1 \le i_1 < i_2 < \cdots < i_n \le m$.

</div>

The proof proceeds similarly. The restriction $F'$ of an $\mathrm{SO}_n\mathbb{C}$-invariant polynomial $F$ on $V = \mathbb{C}^n$ to $V' = \mathbb{C}^{n-1}$ (the orthogonal complement to $e_n$) is $\mathrm{SO}_{n-1}\mathbb{C}$-invariant, and by induction we know it is a polynomial in the restrictions of the basic polynomials $Q(x^{(i)}, x^{(j)})$ and in the bracket $[x^{(1)} \cdots x^{(n-1)}]$. An apparent snag is met here, however, since this bracket is not the restriction of an invariant on $V$. By Exercise F.14, we can write

$$F' = A + B \cdot [x^{(1)} \cdots x^{(n-1)}],$$

where $A$ and $B$ are polynomials in the $Q$'s alone. In particular, $A$ and $B$ are even, i.e., they are invariants of the full orthogonal group $\mathrm{O}_{n-1}\mathbb{C}$. But $F'$ is also even, since any element of $\mathrm{O}_{n-1}\mathbb{C}$ is the restriction of some element in $\mathrm{SO}_n\mathbb{C}$ (mapping $e_n$ to $\pm e_n$). Since the bracket is taken to minus itself by automorphisms of determinant $-1$, we must have $F' = A$. This means that we can subtract a polynomial in the invariants $Q(x^{(i)}, x^{(j)})$ from $F$, so we can assume $F' = 0$. Therefore, the restriction of $F$ to any hyperplane of the form $g \cdot V'$, $g \in \mathrm{SO}_n\mathbb{C}$, is zero. But it is easy to verify that $(n-1)$-tuples in such hyperplanes form an open dense subset of all $(n-1)$-tuples in $\mathbb{C}^n$ (the condition is that there be an orthogonal vector $e$ with $Q(e \cdot e) \neq 0$). This proves $F = 0$, completing the proof. $\square$

### F.2 Applications to Symplectic and Orthogonal Groups

We consider the symplectic group $\mathrm{Sp}_n\mathbb{C}$ and the orthogonal group $\mathrm{O}_n\mathbb{C}$ together, letting $Q$ denote the corresponding skew or symmetric form. The results in the first section, applied to the case $\mathbf{d} = (1, \ldots, 1)$, say that the invariants in $(V^*)^{\otimes m}$ are all polynomials in the polynomials $Q(x^{(i)}, x^{(j)})$, and by degree considerations $m$ must be even, and they are all linear combinations of products

$$Q(x^{(\sigma(1))}, x^{(\sigma(2))}) \cdot Q(x^{(\sigma(3))}, x^{(\sigma(4))}) \cdots Q(x^{(\sigma(m-1))}, x^{(\sigma(m))}) \tag{F.17}$$

for permutations $\sigma$ of $\lbrace 1, \ldots, m\rbrace$ such that $\sigma(2i - 1) < \sigma(2i)$ for $1 \le i \le m/2$. Regarding $Q \in V^* \otimes V^*$, these are obtained from the invariant $Q \otimes \cdots \otimes Q$ ($m/2$ times) by permuting the factors. In other words, one pairs off the $m$ components, and inserts $Q$ in the place indicated by each pair.

The form $Q$ gives an isomorphism of $V$ with $V^*$, which takes $v$ to $Q(v, -)$. Using this we can find all invariants of tensor products $(V^*)^{\otimes k} \otimes (V)^{\otimes l}$, via the isomorphism

$$(V^*)^{\otimes(k+l)} = (V^*)^{\otimes k} \otimes (V^*)^{\otimes l} \cong (V^*)^{\otimes k} \otimes (V)^{\otimes l}.$$

They are linear combinations of the images of the above invariants under this identification. To see what happens to $Q$ under the isomorphisms $V^* \otimes V^* \cong V^* \otimes V = \mathrm{Hom}(V, V) = \mathrm{End}(V)$:

$Q$ maps to the identity endomorphism. Let $\psi$ be the image of $Q$ under the canonical isomorphism $V^* \otimes V^* \cong V \otimes V$. Then:

$$\psi = \sum_{i=1}^{r} e_i \otimes e_{r+i} - e_{r+i} \otimes e_i \quad \text{for } G = \mathrm{Sp}_n\mathbb{C},\; n = 2r;$$

$$\psi = \sum_{i=1}^{n} e_i \otimes e_i \quad \text{for } G = \mathrm{O}_n\mathbb{C}.$$

For the applications in Lectures 17 and 19, we need only the case $l = k$, but we want to reinterpret these invariants by way of the canonical isomorphism

$$(V^*)^{\otimes 2d} \cong (V^*)^{\otimes d} \otimes (V)^{\otimes d} \cong \mathrm{Hom}(V^{\otimes d}, V^{\otimes d}) = \mathrm{End}(V^{\otimes d}). \tag{F.19}$$

In $\S\S$17.3 and 19.5 we defined endomorphisms $\vartheta_I \in \mathrm{End}(V^{\otimes d})$ for each pair $I$ of integers from $\lbrace 1, \ldots, d\rbrace$; for $I$ the first pair,

$$\vartheta_I(v_1 \otimes v_2 \otimes v_3 \otimes \cdots \otimes v_d) = Q(v_1, v_2) \cdot \psi \otimes v_3 \otimes \cdots \otimes v_d;$$

the case for general $I$ is a permutation of this. We claim that an invariant in $(V^*)^{\otimes 2d}$ of the form (F.17) is taken by the isomorphism (F.19) to a composition of operators $\vartheta_I$ and permutations $\sigma$ in $\mathfrak{S}_d$. The invariant in (F.17) is described by pairing the integers from 1 to $2d$. These pairs are either from the first $d$, the last $d$, or one of each.

Now let $A$ be the subalgebra of the ring $\mathrm{End}(V^{\otimes d})$ generated by all $g \otimes \cdots \otimes g$ for $g$ in the group $G = \mathrm{Sp}_n\mathbb{C}$ (or $\mathrm{O}_n\mathbb{C}$). By the simplicity of the group, we know that $A$ is a semisimple algebra of endomorphisms. We have just computed that $B$, the commutator of $A$, is the ring generated by all permutations in $\mathfrak{S}_d$ and the operators $\vartheta_I$. By the general theory of semisimple algebras, cf. $\S$6.2, $A$ must be the commutator algebra of $B$. In English, *any endomorphism of $V^{\otimes d}$ which commutes with permutations and with the operators $\vartheta_I$ must be a finite linear combination of operators of the form $g \otimes \cdots \otimes g$ for $g$ in $G$*. This is precisely the fact from invariant theory that was used in the text.

### F.3 Proof of Capelli's Identity

The proof is not essentially different from the case $m = 2$, once one has a good notational scheme to keep track of the algebraic manipulations which come about because the basic operators $D_{ij}$ do not commute with each other. A convenient way to do this is as follows. For indices $i_1, j_1, \ldots, i_p, j_p$ between 1 and $m$, define an operator $\Lambda_{i_1 j_1} \Lambda_{i_2 j_2} \cdots \Lambda_{i_p j_p}$ which takes a function $F$ of $m$ variables $x^{(1)}, \ldots, x^{(m)}$ to the function

$$\Lambda_{i_1 j_1} \cdots \Lambda_{i_p j_p}(F) = \sum_{k_1, \ldots, k_p = 1}^{n} x_{k_1}^{(i_1)} \cdots x_{k_p}^{(i_p)} \cdot \frac{\partial^p F}{\partial x_{k_1}^{(j_1)} \cdots \partial x_{k_p}^{(j_p)}}.$$

For $p = 1$, $\Lambda_{ij}$ is just the operator $D_{ij}$, but for $p > 1$, this is *not* the composition of the operators $\Lambda_{i_s j_s}$. Note that the order of the terms in the expression $\Lambda_{i_1 j_1} \cdots \Lambda_{i_p j_p}$ is unimportant.

We can form determinants of $p \times p$ matrices with entries these $\Lambda_{ij}$, which act on functions by expanding the determinant as usual, with each of the $p!$ products operating as above. For example, for the $m \times m$ matrix $(\Lambda_{ij})$,

$$\lvert \Lambda_{ij}\rvert(F) = \sum_{\sigma \in \mathfrak{S}_m} \mathrm{sgn}(\sigma)\, \Lambda_{1\sigma(1)} \Lambda_{2\sigma(2)} \cdots \Lambda_{m\sigma(m)}(F).$$

The matrix $(\Lambda_{ij})$ is a product of matrices $(x_k^{(i)}) \cdot (\partial/\partial x_k^{(j)})$, and taking determinants gives the following:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(F.21)</span></p>

For $m = n$, $\lvert\Lambda_{ij}\rvert(F) = [x^{(1)} \cdots x^{(m)}] \cdot \Omega(F)$.

</div>

To prove Capelli's identity (F.6), then, we must prove the following identity of operators on functions $F(x^{(1)}, \ldots, x^{(m)})$:

$$\begin{vmatrix} D_{11} + m - 1 & D_{12} & \cdots & D_{1m} \\ D_{21} & D_{22} + m - 2 & \cdots & D_{2m} \\ \vdots & & \ddots & \vdots \\ D_{m1} & D_{m2} & \cdots & D_{mm} \end{vmatrix} = \begin{vmatrix} \Lambda_{11} & \Lambda_{12} & \cdots & \Lambda_{1m} \\ \Lambda_{21} & \Lambda_{22} & \cdots & \Lambda_{2m} \\ \vdots & & \ddots & \vdots \\ \Lambda_{m1} & \Lambda_{m2} & \cdots & \Lambda_{mm} \end{vmatrix}. \tag{F.22}$$

This is a formal identity, based on the simple identities:

$$D_{qp} \circ D_{ab} = \Lambda_{qp}\Lambda_{ab} \quad \text{if } p \neq a;$$

$$D_{qp} \circ D_{ab} = \Lambda_{qp}\Lambda_{ab} + D_{qb} \quad \text{if } p = a.$$

Similarly, if $p \neq a_k$ for all $k$, then

$$D_{qp} \circ \Lambda_{a_1 b_1} \cdots \Lambda_{a_r b_r} = \Lambda_{qp}\Lambda_{a_1 b_1} \cdots \Lambda_{a_r b_r};$$

while if there is just one $k$ with $p = a_k$, then

$$D_{qp} \circ \Lambda_{a_1 b_1} \cdots \Lambda_{a_r b_r} = \Lambda_{qp}\Lambda_{a_1 b_1} \cdots \Lambda_{a_r b_r} + \Lambda_{a_1 b_1} \cdots \Lambda_{q b_k} \cdots \Lambda_{a_r b_r}, \tag{F.24}$$

where in the last term the $\Lambda_{q b_k}$ replaces $\Lambda_{a_k b_k}$.

We prove (F.22) by showing inductively that all $r \times r$ minors of the two matrices of (F.22) which are taken from the last $r$ columns are equal (as operators on functions $F$ as always). This is obvious when $r = 1$. We suppose it has been proved for $r = m - p$, and show it for $r + 1$. By induction, we may replace the last $r$ columns of the matrix on the left by the last $r$ columns of the matrix on the right. The difference of a minor on the left and the corresponding minor on the right will then be a maximal minor of the matrix

$$\begin{pmatrix} D_{1p} - \Lambda_{1p} & \Lambda_{1,p+1} & \cdots & \Lambda_{1m} \\ D_{2p} - \Lambda_{2p} & \Lambda_{2,p+1} & \cdots & \Lambda_{2m} \\ \vdots & & & \vdots \\ D_{pp} - \Lambda_{pp} + r & \Lambda_{p,p+1} & \cdots & \Lambda_{pm} \\ \vdots & & & \vdots \\ D_{mp} - \Lambda_{mp} & \Lambda_{m,p+1} & \cdots & \Lambda_{mm} \end{pmatrix},$$

so we must show that all maximal minors of this matrix are zero. Suppose that the minor is chosen using the $q_i$th rows, for $1 \le q_0 < q_1 < \cdots < q_r \le m$. Expanding along the first column, this determinant is

$$E_0 M_0 - E_1 M_1 + E_2 M_2 - \cdots + (-1)^r E_r M_r, \tag{F.25}$$

where $E_k = D_{q_k p} - \Lambda_{q_k p}$ if $q_k \neq p$, and $E_k = D_{pp} - \Lambda_{pp} + r$ if $q_k = p$, and $M_k$ is the corresponding cofactor ($r \times r$ determinant).

To show that (F.25) is zero, there are two cases. In the first case, the $p$th row is not included in the minor, i.e., $q_i \neq p$ for all $i$. In this case each term $E_i M_i$ is zero, since $E_i = D_{q_i p} - \Lambda_{q_i p}$, and all the products in the expansion of $M_i$ are of the form $\Lambda_{a_1 b_1} \cdots \Lambda_{a_r b_r}$ with all $a_i \neq p$, and the assertion follows from (F.23).

In the second case, the $p$th row is included, i.e., $q_k = p$ for some $k$. As in the first case, $(D_{pp} - \Lambda_{pp}) M_k = 0$, and since $E_k = D_{pp} - \Lambda_{pp} + r$, we have

$$E_k M_k = r \cdot M_k.$$

We claim that each of the other terms $E_i M_i$, for $i \neq k$, is equal to $(-1)^{k-i+1} M_k$, from which it follows that the alternating sum in (F.25) is zero. When $M_i$ is written out as a determinant and it is multiplied by $E_i = D_{q_i p} - \Lambda_{q_i p}$, an application of (F.24) shows that one gets the same determinant as (F.26), but expanded with the $q_i$th row moved between the $q_{k-1}$th and the $q_{k+1}$th rows. This transposition of rows accounts for the sign $(-1)^{k-i+1}$, yielding $E_i M_i = (-1)^{k-i+1} M_k$, as required. $\square$
