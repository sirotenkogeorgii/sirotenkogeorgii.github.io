---
title: Problems from the Introduction to Geometric Deep Learning course
layout: default
noindex: true
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

# Introduction to Geometric Deep Learning — Course Problems

**Table of Contents**
- TOC
{:toc}

---

## Exercise Sheet 1 — Duality, projections, charts, and derivations

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intro</span><span class="math-callout__name">(Conventions used throughout)</span></p>

We work with finite-dimensional real vector spaces. Throughout this sheet, $\mathcal X$ and $\mathcal Y$ are equipped with scalar products $\ell$ and $m$, with **duality mappings** $L \in \mathcal L(\mathcal X, \widecheck{\mathcal X})$ and $M \in \mathcal L(\mathcal Y, \widecheck{\mathcal Y})$ defined by

$$
\langle L x, x'\rangle = \ell(x, x'), \qquad \langle M y, y'\rangle = m(y, y'),
\tag{A.2}
$$

and the **summation convention**: any index that appears once as a superscript and once as a subscript is summed. A duality pairing $\langle p, x\rangle$ stands for $p(x)$ — a covector evaluated on a vector.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 1</span><span class="math-callout__name">(Dual basis in $\mathbb R^2$)</span></p>

Let $\mathcal X = \mathbb R^2$ equipped with the scalar product $\ell(\cdot, \cdot)$ and the corresponding duality pairing

$$
\langle L\cdot, \cdot\rangle = \langle \cdot, \cdot\rangle_{\widecheck{\mathcal X} \times \mathcal X}
$$

defined by equation (A.2a), to be evaluated like the canonical Euclidean inner product, i.e.

$$
\ell(x, x') \equiv \langle L x, x'\rangle \equiv (Lx)(x') \equiv (Lx)_i\,(x')^i \qquad \text{(summation convention)}.
\tag{0.1}
$$

Let $e_1, e_2$ be any basis.

**(a)** Determine the dual basis $\{\check e^1, \check e^2\} \subset \widecheck{\mathcal X}$ such that $\langle \check e^i, e_j\rangle = \delta^i_j$. *Hint.* Lemma A.2.

**(b)** Apply (a) and compute the basis that is dual to the basis $e_1 = \binom{1}{0}$, $e_2 = \binom{1}{1}$.

**(c)** Now identify $\widecheck{\mathcal X} = \mathcal X = \mathbb R^2$ equipped with the *canonical* Euclidean inner product. Compute the dual basis $\check e^1, \check e^2 \subset \widecheck{\mathcal X}$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 1 (a) — dual basis from the basis-change matrix</summary>

**Setup via Lemma A.2.** Pack the basis $(e_1, e_2)$ into the linear map

$$
B : \mathbb R^2 \to \mathcal X, \qquad B v = v^i e_i,
\tag{$\star$}
$$

so that $B$ sends the canonical basis $\hat e_1 = \binom{1}{0},\ \hat e_2 = \binom{0}{1}$ of $\mathbb R^2$ to $e_1, e_2$. As a matrix, the columns of $B$ are exactly $e_1, e_2$.

**Defining property of the dual basis.** We seek $\check e^i \in \widecheck{\mathcal X}$ with

$$
\langle \check e^i, e_j\rangle = \delta^i_j, \qquad i, j \in \{1, 2\}.
\tag{0.2}
$$

Read covectors as row-vectors and use the matrix realisation. If we write $\check e^i$ as the $i$-th *row* of a matrix $E^\vee \in \mathbb R^{2 \times 2}$, then (0.2) says

$$
(E^\vee\, B)^i_{\ j} = \delta^i_j \qquad \Longleftrightarrow \qquad E^\vee\, B = I_2,
$$

so $E^\vee = B^{-1}$ and the dual basis is given by the **rows of $B^{-1}$**. Equivalently, viewed as column vectors via the canonical identification $\widecheck{\mathbb R^2} \cong \mathbb R^2$,

$$
\boxed{\ \check e^i \;=\; i\text{-th column of }B^{-T}.\ }
$$

This is exactly Lemma A.2: the relation $L = (B \widecheck{B})^{-1}$ translates the basis-change matrix $B$ into the duality mapping, and the dual basis is read off as the rows of $B^{-1}$. $\blacksquare$

</details>

<details class="accordion" markdown="1">
<summary>Solution 1 (b) — explicit computation for $e_1=(1,0),\ e_2=(1,1)$</summary>

**Form $B$ and invert.** With $e_1 = \binom{1}{0}$, $e_2 = \binom{1}{1}$,

$$
B = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \qquad
B^{-1} = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}.
$$

**Read off the dual basis.** The rows of $B^{-1}$ give the dual covectors; identifying $\widecheck{\mathbb R^2} \cong \mathbb R^2$,

$$
\check e^1 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}, \qquad
\check e^2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}.
$$

**Sanity check via biorthogonality.**

$$
\begin{aligned}
\langle \check e^1, e_1\rangle &= 1 \cdot 1 + (-1)\cdot 0 = 1, &
\langle \check e^1, e_2\rangle &= 1 \cdot 1 + (-1)\cdot 1 = 0, \\
\langle \check e^2, e_1\rangle &= 0 \cdot 1 + 1 \cdot 0 = 0, &
\langle \check e^2, e_2\rangle &= 0 \cdot 1 + 1 \cdot 1 = 1.
\end{aligned}
$$

All four pairings equal the corresponding $\delta^i_j$, so $\{\check e^1, \check e^2\}$ is the dual basis. $\blacksquare$

**Geometric reading.** The covector $\check e^1$ has level sets $\{x : x^1 - x^2 = c\}$ — these are the diagonal lines parallel to $e_2$. So $\check e^1$ "measures the $e_1$-component" of $x$ in the *oblique* coordinate system $(e_1, e_2)$, *not* in the canonical one. This is why $\check e^1 \ne e_1$ even though we use the canonical inner product to evaluate the pairing — the dual basis depends on the *primal* basis, not on the inner product alone.

</details>

<details class="accordion" markdown="1">
<summary>Solution 1 (c) — canonical Euclidean inner product and self-dual standard basis</summary>

**Identify $\widecheck{\mathcal X} = \mathcal X = \mathbb R^2$ via the canonical inner product.** Under the canonical inner product $\ell(x, x') = (x)_i (x')^i$ (the dot product), the duality mapping $L : \mathcal X \to \widecheck{\mathcal X}$ is the identity in coordinates: $L = I$.

**Case 1 — basis $(e_1, e_2) = (\hat e_1, \hat e_2)$.** Then $B = I$, $B^{-1} = I$, and the rows of $I$ give

$$
\check e^1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix} = e_1, \qquad
\check e^2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix} = e_2.
$$

The standard basis is **self-dual** under the canonical inner product: an orthonormal basis (under $\ell$) coincides with its own dual basis.

**Case 2 — same basis as in (b), but read as column vectors after identification.** If we keep the basis $e_1 = \binom{1}{0}$, $e_2 = \binom{1}{1}$ from part (b) and now view its dual basis as elements of $\mathcal X = \mathbb R^2$ via the Riesz isomorphism $L^{-1}$, the coordinate answer is unchanged because $L = I$:

$$
\check e^1 = \begin{pmatrix} 1 \\ -1 \end{pmatrix} \in \mathbb R^2, \qquad
\check e^2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \in \mathbb R^2.
$$

What *did* change is the interpretation: $\check e^i$ is now an honest *vector* in $\mathbb R^2$ (the unique vector representing the linear functional $x \mapsto \langle \check e^i, x\rangle$ via the canonical dot product). Biorthogonality then reads

$$
\check e^i \cdot e_j = \delta^i_j,
$$

which is exactly the same numerics as in (b).

**Punchline.** The dual basis depends on the primal basis through $B^{-1}$, *not* on the inner product (once we use the Euclidean identification). The inner product enters only when we want to *identify* covectors with vectors — which it does here trivially, because the canonical Riesz iso is the identity. $\blacksquare$

</details>

<details class="accordion" markdown="1">
<summary>Visualisation — dual basis vs. canonical basis</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/sheet1_dual_basis.png' | relative_url }}" alt="Left: vectors e1=(1,0), e2=(1,1) and their dual basis covectors ě1=(1,-1), ě2=(0,1) plotted in R² with level sets of the covectors as faded diagonal and horizontal lines. Right: the canonical basis is its own dual under the canonical Euclidean inner product." loading="lazy">
  <figcaption>Left (b): the dual basis $\check e^1 = (1, -1)$, $\check e^2 = (0, 1)$ for the oblique basis $e_1, e_2$. Faded red lines are level sets of $\check e^1$ (diagonals along direction $e_2$); faded orange lines are level sets of $\check e^2$ (horizontals along $e_1$). Each covector $\check e^i$ kills $e_j$ for $j \ne i$. Right (c): the canonical basis under the canonical inner product is self-dual.</figcaption>
</figure>

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 2</span><span class="math-callout__name">(Orthogonal projections via $A^-$ and $A^+$)</span></p>

Consider the basic factorisation of a linear mapping $A \in \mathcal L(\mathcal X, \mathcal Y)$:

$$
\begin{array}{ccc}
\mathcal X & \xrightarrow{\ A\ } & \mathcal Y \\
{\scriptstyle\pi}\downarrow & & \uparrow{\scriptstyle\iota} \\
\mathcal X / \ker(A) & \xrightarrow{\ \underline A\ } & \mathrm{rge}(A)
\end{array}
\tag{0.2}
$$

representing $A = \iota \circ \underline A \circ \pi$ as a composition of a surjection $\pi$, a *bijective* mapping $\underline A$, and an injection $\iota$. Take the basic scenario $\mathcal X = \mathbb R^n$, $\mathcal Y = \mathbb R^m$, $A \in \mathbb R^{m \times n} \cong \mathcal L(\mathbb R^n, \mathbb R^m)$. Recall

$$
\begin{aligned}
n = \dim(\mathcal X) &= \dim(\ker(A)) + \dim(\mathrm{rge}(A)) &&\text{(0.3a)} \\
&= \dim(\ker(A)) + \mathrm{rank}(A), &&\text{(0.3b)} \\
\mathrm{rank}(A) &\le \min\{m, n\}. &&\text{(0.3c)}
\end{aligned}
$$

Assume throughout that $A$ has *full rank*, i.e. equality in (0.3c).

**(i)** Suppose $A$ is *injective*. Derive the orthogonal projection onto $\mathrm{rge}(A)$ with respect to the inner product $m(\cdot, \cdot)$ on $\mathcal Y$. Interpret the orthogonal left-inverse $A^-$.

**(ii)** Suppose $A$ is *surjective*. Derive the orthogonal projection onto $\ker(A)$ with respect to the inner product $\ell(\cdot, \cdot)$ on $\mathcal X$. Interpret the orthogonal right-inverse $A^+$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 2 (i) — projection onto $\mathrm{rge}(A)$ and the meaning of $A^-$</summary>

**Setup.** $A : \mathcal X \to \mathcal Y$ injective and full rank, so $m \ge n$, $\mathrm{rge}(A) \subset \mathcal Y$ is an $n$-dimensional subspace, and $\mathrm{rge}(A) \ne \mathcal Y$ in general. Given $y \in \mathcal Y$, we want the closest point $\hat y \in \mathrm{rge}(A)$ in the $m$-norm $\mu(\cdot) = \sqrt{m(\cdot,\cdot)}$.

**Variational formulation.** Every $\hat y \in \mathrm{rge}(A)$ has a unique preimage $x \in \mathcal X$ (since $A$ is injective). Minimise

$$
J(x) \;:=\; \tfrac{1}{2}\,m(Ax - y, Ax - y) \qquad \text{over } x \in \mathcal X.
\tag{0.4}
$$

The first-order condition is found by computing the directional derivative in any direction $h \in \mathcal X$:

$$
\frac{d}{dt}\bigg|_{t=0} J(x + th) \;=\; m(Ax - y, Ah) \;=\; \langle M(Ax - y), Ah\rangle.
$$

Using the transpose $\widecheck A : \widecheck{\mathcal Y} \to \widecheck{\mathcal X}$ characterised by $\langle \widecheck A p, x\rangle = \langle p, Ax\rangle$ for all $x, p$,

$$
\frac{d}{dt}\bigg|_{t=0} J(x + th) \;=\; \langle \widecheck A M(Ax - y), h\rangle.
$$

**Critical point.** Setting this to zero for every $h$ gives the **normal equations**

$$
\widecheck A M A\, x \;=\; \widecheck A M y.
\tag{0.5}
$$

Now $A$ injective and $M$ positive definite imply $\widecheck A M A : \mathcal X \to \widecheck{\mathcal X}$ is invertible (it is the Gramian of $A$ in the metric $m$, and is symmetric positive definite). So

$$
x^\ast \;=\; (\widecheck A M A)^{-1}\, \widecheck A M\, y.
$$

**Reading off $A^-$ and the projection.** Comparing with (A.11),

$$
\boxed{\ A^- \;=\; (\widecheck A M A)^{-1}\, \widecheck A M \;\in\; \mathcal L(\mathcal Y, \mathcal X),\qquad x^\ast \;=\; A^- y.\ }
$$

Substituting back, $\hat y = A x^\ast = A A^- y$, so

$$
\boxed{\ A A^- \;=\; \Pi_{\mathrm{rge}(A)} \quad\text{(orthogonal projection onto $\mathrm{rge}(A)$ in the $m$-metric)}.\ }
$$

**Sanity properties.**

  • *Idempotence.* $\widecheck A M A x^\ast = \widecheck A M y$ and re-applying gives $A A^- (A A^- y) = A x^\ast = A A^- y$, so $(AA^-)^2 = AA^-$.

  • *Self-adjointness in the $m$-metric.* For any $y, y'$, $m(AA^- y, y') = \langle M y', A x^\ast\rangle = \langle \widecheck A M y', x^\ast\rangle = \langle (\widecheck A M A) x^{\ast\prime}, x^\ast\rangle = \langle (\widecheck A M A) x^\ast, x^{\ast\prime}\rangle = m(y, AA^- y')$, where $x^{\ast\prime} = A^- y'$. So $AA^-$ is $m$-symmetric, hence an orthogonal projector.

  • *Identity on $\mathrm{rge}(A)$.* $A^- A x = (\widecheck A M A)^{-1}(\widecheck A M A) x = x$ — the left-inverse property (A.12a).

**Interpretation of $A^-$.** $A^-$ is the **least-squares solver** for the over-determined system $Ax = y$. For $y \in \mathrm{rge}(A)$, $A^- y$ is the unique exact pre-image. For general $y \in \mathcal Y$, $A^- y$ is the unique minimiser of $\mu(Ax - y)$. The orthogonal projection $AA^-$ "snaps" $y$ onto $\mathrm{rge}(A)$ before reading off the pre-image.

In matrix terms, with $M = I$ (canonical inner product), this is the textbook formula $A^- = (A^T A)^{-1} A^T$ and $\Pi_{\mathrm{rge}(A)} = A(A^T A)^{-1}A^T$. The general $M$ replaces $A^T$ by $\widecheck A M$, weighting the residual by the metric $m$. $\blacksquare$

</details>

<details class="accordion" markdown="1">
<summary>Solution 2 (ii) — projection onto $\ker(A)$ and the meaning of $A^+$</summary>

**Setup.** $A : \mathcal X \to \mathcal Y$ surjective and full rank, so $m \le n$, $\ker(A) \subset \mathcal X$ has dimension $n - m$, and $\mathcal X = \ker(A) \oplus \ker(A)^{\perp_\ell}$ where the orthogonal complement is taken in the $\ell$-inner product.

**Strategy.** Decompose $x = x_\perp + x_K$ with $x_\perp \in \ker(A)^{\perp_\ell}$ and $x_K \in \ker(A)$, and find $x_\perp$ first. Then $\Pi_{\ker(A)} x = x_K = x - x_\perp$.

**Identifying $\ker(A)^{\perp_\ell}$.** A vector $z \in \mathcal X$ lies in $\ker(A)^{\perp_\ell}$ iff $\ell(z, n) = 0$ for all $n \in \ker(A)$. Using $\ell(z, n) = \langle Lz, n\rangle$ and the rank-nullity duality

$$
\ker(A)^{\circ} \;=\; \mathrm{rge}(\widecheck A) \;\subset\; \widecheck{\mathcal X},
$$

(every covector annihilating $\ker(A)$ is of the form $\widecheck A p$ for some $p \in \widecheck{\mathcal Y}$), we get

$$
\ker(A)^{\perp_\ell} \;=\; L^{-1}\bigl(\mathrm{rge}(\widecheck A)\bigr) \;=\; \{L^{-1} \widecheck A p : p \in \widecheck{\mathcal Y}\}.
$$

So $x_\perp = L^{-1} \widecheck A\, p$ for some $p \in \widecheck{\mathcal Y}$.

**Determining $p$.** Since $x_K \in \ker(A)$, $A x = A x_\perp$, so

$$
A L^{-1} \widecheck A\, p \;=\; A x.
$$

The operator $A L^{-1} \widecheck A : \widecheck{\mathcal Y} \to \mathcal Y$ is invertible — it is symmetric positive definite as a "Gramian-of-the-transpose" computation, using $A$ surjective and $L$ positive definite. Hence

$$
p \;=\; (A L^{-1} \widecheck A)^{-1}\, A x, \qquad
x_\perp \;=\; L^{-1} \widecheck A\, (A L^{-1} \widecheck A)^{-1}\, A x.
$$

**Reading off $A^+$ and the projection.** Comparing with (A.13),

$$
\boxed{\ A^+ \;=\; L^{-1} \widecheck A\, (A L^{-1} \widecheck A)^{-1} \;\in\; \mathcal L(\mathcal Y, \mathcal X),\qquad x_\perp \;=\; A^+ A\, x.\ }
$$

Therefore

$$
\boxed{\ \Pi_{\ker(A)} \;=\; I - A^+ A \quad\text{(orthogonal projection onto $\ker(A)$ in the $\ell$-metric)}.\ }
$$

**Sanity properties.**

  • *Right-inverse.* $A A^+ y = A L^{-1}\widecheck A (A L^{-1}\widecheck A)^{-1} y = y$ for all $y \in \mathcal Y$ — formula (A.14a).

  • *Idempotence on $\ker(A)^{\perp_\ell}$.* $A^+ A$ projects onto $\ker(A)^{\perp_\ell}$, so $(A^+ A)^2 = A^+ A$. Equivalently $(I - A^+ A)^2 = I - A^+ A$.

  • *$\ell$-self-adjointness of $A^+ A$.* Using $\widecheck{(L^{-1})} = L^{-1}$ and $\widecheck{(A^+ A)} = \widecheck A \widecheck{A^+}$, one checks $\ell(A^+ A x, x') = \ell(x, A^+ A x')$ — so the projection is orthogonal in the $\ell$-metric.

**Interpretation of $A^+$.** $A^+$ is the **minimum-norm solver** for the under-determined system $Ax = y$. The set $A^{-1}(\{y\}) = x_p + \ker(A)$ is an affine subspace of dimension $n - m$, and among all solutions, $A^+ y$ is the unique one orthogonal to $\ker(A)$ in $\ell$ — equivalently, the one with smallest $\lambda$-norm. The orthogonal projection $A^+ A$ takes any $x$ to its "essential part" $x_\perp$ that determines $Ax$, while $I - A^+ A$ extracts the kernel component which contributes nothing to $Ax$.

In matrix terms with $L = I$, $A^+ = A^T (A A^T)^{-1}$ and $\Pi_{\ker(A)} = I - A^T(AA^T)^{-1}A$. $\blacksquare$

**Duality between (i) and (ii).** Comparing the two cases, $A^-$ kills the part of $y$ orthogonal to $\mathrm{rge}(A)$ and $A^+$ kills the part of $x$ inside $\ker(A)$ — they are dual constructions, and the sheet's pseudo-inverse $C^\dagger = A^+ B^-$ glues them together for the rank-deficient case.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation — orthogonal projections onto range and kernel</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/sheet1_orthogonal_projections.png' | relative_url }}" alt="Left: a 1D range line in R² with a target point y, its orthogonal projection AA⁻y onto the range, and the residual y − AA⁻y. Right: a kernel line and its orthogonal complement in R², a point x decomposed as x = (I−A⁺A)x + A⁺Ax." loading="lazy">
  <figcaption>Left (i): for $A$ injective, $AA^- y$ is the closest point of $\mathrm{rge}(A)$ to $y$ in the $m$-metric; the residual $y - AA^- y$ is $m$-orthogonal to $\mathrm{rge}(A)$. Right (ii): for $A$ surjective, $\mathcal X$ splits $\ell$-orthogonally as $\ker(A) \oplus \ker(A)^\perp$; the projection $A^+ A x$ lands in $\ker(A)^\perp = \mathrm{rge}(A^+)$ and $(I - A^+A)x$ in $\ker(A)$.</figcaption>
</figure>

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3</span><span class="math-callout__name">(Two charts of the 2-sphere agree on tangent spaces)</span></p>

Consider the two charts of the 2-sphere $S^2 \subset \mathbb R^3$

$$
\gamma : \mathbb D \to S^2, \qquad
\binom{u}{v} \mapsto
\begin{pmatrix} u \\ v \\ \sqrt{1 - u^2 - v^2} \end{pmatrix},
\tag{0.4a}
$$

$$
\tau : \mathbb D \to S^2, \qquad
\binom{u}{v} \mapsto
\begin{pmatrix} u \\ \sqrt{1 - u^2 - v^2} \\ v \end{pmatrix},
\tag{0.4b}
$$

where $\mathbb D = \{\binom{u}{v} : u^2 + v^2 < 1\}$. Show that

$$
T_{\tau(1/\sqrt 3,\,1/\sqrt 3)} S^2 \;=\; T_{\gamma(1/\sqrt 3,\,1/\sqrt 3)} S^2.
\tag{0.5}
$$

</div>

<details class="accordion" markdown="1">
<summary>Solution 3 — same point, same tangent plane</summary>

**Step 1 — both charts hit the same point of $S^2$.** Evaluate $\gamma$ and $\tau$ at $(u, v) = (1/\sqrt 3, 1/\sqrt 3)$. The hidden coordinate is $\sqrt{1 - 1/3 - 1/3} = \sqrt{1/3} = 1/\sqrt 3$. Hence

$$
p \;:=\; \gamma\!\left(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\right) \;=\; \begin{pmatrix} 1/\sqrt 3 \\ 1/\sqrt 3 \\ 1/\sqrt 3\end{pmatrix} \;=\; \tau\!\left(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\right).
$$

So both tangent spaces are taken at the **same point** $p \in S^2$ — only the *parametrisations* of the chart differ.

**Step 2 — tangent space from $\gamma$.** By definition, $T_p S^2$ from the chart $\gamma$ is the image of the differential $D\gamma$ at $(1/\sqrt 3, 1/\sqrt 3)$, i.e. the span of the partial derivatives:

$$
\partial_u \gamma(u, v) = \begin{pmatrix} 1 \\ 0 \\ -u/\sqrt{1 - u^2 - v^2} \end{pmatrix}, \qquad
\partial_v \gamma(u, v) = \begin{pmatrix} 0 \\ 1 \\ -v/\sqrt{1 - u^2 - v^2} \end{pmatrix}.
$$

At $(1/\sqrt 3, 1/\sqrt 3)$, $u/\sqrt{1 - u^2 - v^2} = (1/\sqrt 3)/(1/\sqrt 3) = 1$, so

$$
\partial_u \gamma\bigl(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\bigr) = \begin{pmatrix} 1 \\ 0 \\ -1\end{pmatrix},\qquad
\partial_v \gamma\bigl(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\bigr) = \begin{pmatrix} 0 \\ 1 \\ -1\end{pmatrix}.
$$

Hence

$$
T_{\gamma(1/\sqrt 3,\,1/\sqrt 3)} S^2 \;=\; \mathrm{span}\!\left\{(1, 0, -1)^T,\, (0, 1, -1)^T\right\}.
$$

**Step 3 — tangent space from $\tau$.** Similarly,

$$
\partial_u \tau(u, v) = \begin{pmatrix} 1 \\ -u/\sqrt{1 - u^2 - v^2} \\ 0\end{pmatrix},\qquad
\partial_v \tau(u, v) = \begin{pmatrix} 0 \\ -v/\sqrt{1 - u^2 - v^2} \\ 1\end{pmatrix}.
$$

At $(1/\sqrt 3, 1/\sqrt 3)$,

$$
\partial_u \tau\bigl(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\bigr) = \begin{pmatrix} 1 \\ -1 \\ 0\end{pmatrix},\qquad
\partial_v \tau\bigl(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\bigr) = \begin{pmatrix} 0 \\ -1 \\ 1\end{pmatrix}.
$$

So

$$
T_{\tau(1/\sqrt 3,\,1/\sqrt 3)} S^2 \;=\; \mathrm{span}\!\left\{(1, -1, 0)^T,\, (0, -1, 1)^T\right\}.
$$

**Step 4 — both spans equal the plane $\{v : v \cdot p = 0\}$.** All four vectors are orthogonal to $p = (1/\sqrt 3, 1/\sqrt 3, 1/\sqrt 3)^T$:

$$
\begin{aligned}
(1, 0, -1) \cdot p &= 1/\sqrt 3 - 1/\sqrt 3 = 0, &
(0, 1, -1) \cdot p &= 1/\sqrt 3 - 1/\sqrt 3 = 0, \\
(1, -1, 0) \cdot p &= 1/\sqrt 3 - 1/\sqrt 3 = 0, &
(0, -1, 1) \cdot p &= -1/\sqrt 3 + 1/\sqrt 3 = 0.
\end{aligned}
$$

So both spans lie inside the 2-dimensional hyperplane $H_p := \{v \in \mathbb R^3 : v \cdot p = 0\}$. Each pair is linearly independent (the $\gamma$-pair has a $2\times 2$ minor with determinant $1$; same for the $\tau$-pair). A 2D subspace of a 2D space equals the whole space, so both spans equal $H_p$.

**Step 5 — explicit change-of-basis between the two pairs.** A direct check:

$$
\begin{aligned}
(1, -1, 0) &= (1, 0, -1) - (0, 1, -1), \\
(0, -1, 1) &= -\,(0, 1, -1).
\end{aligned}
$$

So

$$
\begin{pmatrix} \partial_u \tau \\ \partial_v \tau\end{pmatrix} \;=\; \begin{pmatrix} 1 & -1 \\ 0 & -1\end{pmatrix} \begin{pmatrix} \partial_u \gamma \\ \partial_v \gamma\end{pmatrix},
$$

with the change-of-basis matrix being exactly the Jacobian of the transition map $\tau^{-1} \circ \gamma$ at $(1/\sqrt 3, 1/\sqrt 3)$. This is the standard manifold-theoretic statement: tangent spaces are chart-independent because the Jacobian of any transition map is invertible. $\blacksquare$

**Geometric reading.** $S^2$ is a 2-sphere in $\mathbb R^3$, and at $p = (1/\sqrt 3, 1/\sqrt 3, 1/\sqrt 3)$ the tangent plane is the affine plane through $p$ normal to $p$ itself (since the sphere is the unit level set of $x \mapsto \|x\|^2$, and the gradient of $\|x\|^2$ at $p$ is $2p$). The two charts $\gamma$ and $\tau$ produce *different* coordinate bases of this same plane, but the plane itself is intrinsic.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation — two charts of $S^2$ at the common point</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/sheet1_sphere_charts.png' | relative_url }}" alt="The unit sphere with two coloured chart patches (top hemisphere via γ, front hemisphere via τ) overlapping at p=(1/√3, 1/√3, 1/√3). Four tangent vectors emanate from p: two from γ, two from τ, all lying in the same yellow tangent plane normal to p." loading="lazy">
  <figcaption>The two charts $\gamma$ (blue patch, projects onto $z = \sqrt{1 - u^2 - v^2}$) and $\tau$ (green patch, projects onto $y = \sqrt{1 - u^2 - v^2}$) both contain the point $p = (1/\sqrt 3, 1/\sqrt 3, 1/\sqrt 3)$. The four partial-derivative vectors all lie in the yellow tangent plane $T_p S^2 = \{v \cdot p = 0\}$, and span it.</figcaption>
</figure>

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4</span><span class="math-callout__name">(Derivations form a vector space)</span></p>

Let $\mathcal M$ be a smooth manifold. Any tangent vector $v \in T_x \mathcal M$ to $\mathcal M$ at $x \in \mathcal M$ can be considered as a *derivation* of the space $\mathcal F(\mathcal M)$ of smooth functions $f : \mathcal M \to \mathbb R$, i.e. an $\mathbb R$-linear mapping satisfying the **Leibniz rule**

$$
v(fg) \;=\; f\,v(g) + g\,v(f), \qquad f, g \in \mathcal F(\mathcal M).
\tag{0.6}
$$

Show that all such vectors $v$ form a vector space.

</div>

<details class="accordion" markdown="1">
<summary>Solution 4 — the set of derivations is a real vector subspace of $\mathrm{Hom}_{\mathbb R}(\mathcal F(\mathcal M), \mathbb R)$</summary>

**Setup.** Let

$$
\mathcal D_x \;:=\; \bigl\{\, v : \mathcal F(\mathcal M) \to \mathbb R \;\big|\; v\ \text{is}\ \mathbb R\text{-linear and satisfies (0.6)}\,\bigr\}.
$$

We want to show that $\mathcal D_x$ is a vector space over $\mathbb R$. The slick path: $\mathcal D_x \subset \mathrm{Hom}_{\mathbb R}(\mathcal F(\mathcal M), \mathbb R)$, which is itself a vector space (functions to $\mathbb R$ form a vector space pointwise). It then suffices to show $\mathcal D_x$ is a *subspace* — i.e. it contains $0$ and is closed under addition and scalar multiplication.

**Step 1 — the zero map is a derivation.** Define $0 : \mathcal F(\mathcal M) \to \mathbb R$ by $0(f) = 0$. It is $\mathbb R$-linear trivially. Leibniz:

$$
0(fg) = 0 = f \cdot 0 + g \cdot 0 = f\,0(g) + g\,0(f). \quad \checkmark
$$

So $0 \in \mathcal D_x$.

**Step 2 — closure under addition.** Let $v, w \in \mathcal D_x$ and define $(v + w)(f) := v(f) + w(f)$.

  • *Linearity.* For $\alpha, \beta \in \mathbb R$ and $f, g \in \mathcal F(\mathcal M)$,

  $$
  (v + w)(\alpha f + \beta g) = v(\alpha f + \beta g) + w(\alpha f + \beta g) = \alpha v(f) + \beta v(g) + \alpha w(f) + \beta w(g)
  = \alpha (v + w)(f) + \beta(v + w)(g).
  $$

  • *Leibniz.* For $f, g \in \mathcal F(\mathcal M)$,

  $$
  \begin{aligned}
  (v + w)(fg) &= v(fg) + w(fg) \\
              &= \bigl[f\,v(g) + g\,v(f)\bigr] + \bigl[f\,w(g) + g\,w(f)\bigr] \\
              &= f\,\bigl(v(g) + w(g)\bigr) + g\,\bigl(v(f) + w(f)\bigr) \\
              &= f\,(v + w)(g) + g\,(v + w)(f). \quad \checkmark
  \end{aligned}
  $$

So $v + w \in \mathcal D_x$.

**Step 3 — closure under scalar multiplication.** Let $v \in \mathcal D_x$, $\alpha \in \mathbb R$, and define $(\alpha v)(f) := \alpha\, v(f)$.

  • *Linearity.* For $\beta, \gamma \in \mathbb R$ and $f, g \in \mathcal F(\mathcal M)$,

  $$
  (\alpha v)(\beta f + \gamma g) = \alpha\,v(\beta f + \gamma g) = \alpha(\beta v(f) + \gamma v(g)) = \beta(\alpha v)(f) + \gamma(\alpha v)(g).
  $$

  • *Leibniz.* For $f, g \in \mathcal F(\mathcal M)$,

  $$
  (\alpha v)(fg) = \alpha\,v(fg) = \alpha\bigl[f\,v(g) + g\,v(f)\bigr] = f\,\alpha v(g) + g\,\alpha v(f) = f\,(\alpha v)(g) + g\,(\alpha v)(f). \quad \checkmark
  $$

So $\alpha v \in \mathcal D_x$.

**Step 4 — vector-space axioms inherited.** Associativity and commutativity of $+$, distributivity, the unit law $1 \cdot v = v$, and existence of inverses $(-1) \cdot v = -v$ all hold in $\mathrm{Hom}_{\mathbb R}(\mathcal F(\mathcal M), \mathbb R)$ pointwise; since $\mathcal D_x$ is a subset closed under these operations and contains $0$, all axioms transfer. $\blacksquare$

**Why this matters.** This abstract definition of $T_x \mathcal M$ via derivations is exactly the one that lifts to manifolds without an ambient $\mathbb R^n$ — there is no notion of "velocity vector in $\mathbb R^3$" on an abstract manifold, so the definition through Leibniz is what survives. The vector-space structure shown here is what makes $T_x \mathcal M$ a *linear* tangent space, which in turn is what makes the differential $df_x : T_x \mathcal M \to T_{f(x)} \mathcal N$ a linear map and gives the chain rule its tensorial form. A coordinate basis of $T_x \mathcal M$ is then provided by the partial derivatives $\partial_{u^i}|_x$ in any chart — exactly the four vectors that appeared in Exercise 3.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation — derivations close under linear combinations</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/sheet1_derivations.png' | relative_url }}" alt="Three rows of arrows from F(M) to R: the top labelled v, the middle labelled w, the bottom labelled αv + βw, with the Leibniz identity for the combined derivation displayed below." loading="lazy">
  <figcaption>Two derivations $v, w$ at $x$ combine $\mathbb R$-linearly into $\alpha v + \beta w$. The Leibniz rule is preserved by linear combinations because it is linear in the derivation: $(\alpha v + \beta w)(fg) = f(\alpha v + \beta w)(g) + g(\alpha v + \beta w)(f)$. This is what makes $T_x \mathcal M$ a vector space.</figcaption>
</figure>

</details>
