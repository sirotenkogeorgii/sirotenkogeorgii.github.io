---
title: Problems from the Introduction to Geometric Deep Learning course
layout: default
noindex: true
tags:
  - machine-learning
  - geometric-deep-learning
  - linear-algebra
  - differential-geometry
  - manifolds
  - exercises
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

### Dual basis in $\mathbb R^2$

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

**(a)** Determine the dual basis $\lbrace \check e^1, \check e^2\rbrace \subset \widecheck{\mathcal X}$ such that $\langle \check e^i, e_j\rangle = \delta^i_j$. *Hint.* Lemma A.2.

**(b)** Apply (a) and compute the basis that is dual to the basis $e_1 = \binom{1}{0}$, $e_2 = \binom{1}{1}$.

**(c)** Now identify $\widecheck{\mathcal X} = \mathcal X = \mathbb R^2$ equipped with the *canonical* Euclidean inner product. Compute the dual basis $\check e^1, \check e^2 \subset \widecheck{\mathcal X}$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 1 (a) — dual basis from the basis-change matrix</summary>

Let $e_i,e_j\in\widecheck{\mathcal{X}}$ be such that $\langle \widecheck e^i, e_j \rangle = \delta_j^i$. By definition,

$$\langle \widecheck e_i, e_j\rangle = \langle \widecheck e_i, L^{-1}\widecheck e_j\rangle = l(L^{-1} \widecheck e_i, L^{-1}\widecheck e_j),$$

Recall that $L = (B\widecheck{B})^{-1}$, which implies $L^{-1} = B\widecheck{B}$. Thus,

$$l(L^{-1} \widecheck e_i, L^{-1}\widecheck e_j) = l(B\underbrace{\widecheck{B} \widecheck e_i}_{=:u_i}, B\underbrace{\widecheck{B}\widecheck e_j}_{=:u_j}) = l(Bu_i, Bu_j).$$

Let $u_i = \widecheck{B} \widecheck e_i = \delta_i$ be an ansatz, then

$$l(Bu_i, Bu_j) = l(B\delta_i, B\delta_j) = l(e_i, e_j) = \langle Le_i, e_j\rangle = \langle \widecheck e_i, e_j\rangle = \delta_j^i.$$

Thus,

$$\widecheck e_i = \widecheck{B}^{-1}\delta_i.$$

</details>

<figure class="math-figure">
  <svg viewBox="0 0 660 280" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:660px" aria-label="Map schema between R d, X, and X dual in Lemma A.2">
    <defs>
      <marker id="arrow-gdl-a2-map-schema" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
        <path d="M0,0 L9,4.5 L0,9 Z" fill="#5b6270" />
      </marker>
    </defs>

    <g font-family="serif" fill="#1f2430">
      <!-- Space labels -->
      <g>
        <rect x="72" y="52" width="160" height="68" rx="4" fill="#fff7e0" stroke="#a86f00" stroke-width="1.5" />
        <text x="152" y="84" text-anchor="middle" font-size="24" font-style="italic">R<tspan baseline-shift="super" font-size="14">d</tspan></text>
        <text x="152" y="104" text-anchor="middle" font-size="11" fill="#5b6270">coordinate space</text>
      </g>

      <g>
        <rect x="428" y="52" width="160" height="68" rx="4" fill="rgba(44,73,148,0.10)" stroke="#2c4994" stroke-width="1.5" />
        <text x="508" y="86" text-anchor="middle" font-size="25" font-style="italic">X</text>
        <text x="508" y="104" text-anchor="middle" font-size="11" fill="#5b6270">vector space</text>
      </g>

      <g>
        <rect x="428" y="184" width="160" height="68" rx="4" fill="rgba(214,83,54,0.10)" stroke="#d65336" stroke-width="1.5" />
        <text x="508" y="216" text-anchor="middle" font-size="25" font-style="italic">X</text>
        <path d="M499 192 L505 187 L515 194" fill="none" stroke="#1f2430" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" />
        <text x="508" y="236" text-anchor="middle" font-size="11" fill="#5b6270">covectors on X</text>
      </g>

      <!-- Maps -->
      <g stroke="#5b6270" stroke-width="1.5" fill="none">
        <line x1="232" y1="86" x2="428" y2="86" marker-end="url(#arrow-gdl-a2-map-schema)" />
        <line x1="508" y1="120" x2="508" y2="184" marker-end="url(#arrow-gdl-a2-map-schema)" />
        <line x1="428" y1="205" x2="224" y2="113" marker-end="url(#arrow-gdl-a2-map-schema)" />
      </g>

      <g font-size="18" font-style="italic" fill="#1f2430">
        <text x="330" y="72" text-anchor="middle">B</text>
        <text x="526" y="156" text-anchor="start">L</text>
        <text x="326" y="179" text-anchor="middle">B</text>
        <path d="M319 158 L324 154 L333 160" fill="none" stroke="#1f2430" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round" />
      </g>

      <text x="330" y="260" text-anchor="middle" font-size="12" fill="#5b6270">
        The return arrow uses the transpose map after identifying (R^d) dual with R^d.
      </text>
    </g>
  </svg>
  <figcaption>The maps in Lemma A.2. The basis map $B : \mathbb R^d \to \mathcal X$ sends coordinate vectors to vectors in $\mathcal X$; the duality map $L : \mathcal X \to \widecheck{\mathcal X}$ lowers a vector to a covector; and the transposed map $\widecheck B : \widecheck{\mathcal X} \to \mathbb R^d$ returns coordinates after the standard identification $\widecheck{\mathbb R^d} \cong \mathbb R^d$.</figcaption>
</figure>

<details class="accordion" markdown="1">
<summary>Solution 1 (a)</summary>

<!-- Let $e_i,e_j\in\widecheck{\mathcal{X}}$ be such that $\langle \widecheck e^i, e_j \rangle = \delta_j^i$. By definition,

$$\langle \widecheck e_i, e_j\rangle = \langle \widecheck e_i, L^{-1}\widecheck e_j\rangle = l(L^{-1} \widecheck e_i, L^{-1}\widecheck e_j),$$

Recall that $L = (B\widecheck{B})^{-1}$, which implies $L^{-1} = B\widecheck{B}$. Thus,

$$l(L^{-1} \widecheck e_i, L^{-1}\widecheck e_j) = l(B\underbrace{\widecheck{B} \widecheck e_i}_{=:u_i}, B\underbrace{\widecheck{B}\widecheck e_j}_{=:u_j}) = l(Bu_i, Bu_j).$$

Let $u_i = \widecheck{B} \widecheck e_i = \delta_i$ be an ansatz, then

$$l(Bu_i, Bu_j) = l(B\delta_i, B\delta_j) = l(e_i, e_j) = \langle Le_i, e_j\rangle = \langle \widecheck e_i, e_j\rangle = \delta_j^i.$$

Thus,

$$\widecheck e_i = \widecheck{B}^{-1}\delta_i.$$ -->

The clean proof (metric‑free, as part (a) wants). 

Use only $e_j=B\delta_j$ and the transpose relation $\langle\widecheck B,p,v\rangle=\langle p,Bv\rangle$: 

$$\langle \widecheck B^{-1}\delta_i,;e_j\rangle =\langle \widecheck B^{-1}\delta_i,;B\delta_j\rangle =\langle \widecheck B,\widecheck B^{-1}\delta_i,;\delta_j\rangle =\langle \delta_i,\delta_j\rangle =\delta^i_j.$$

So $\check e^i=\widecheck B^{-1}\delta_i$ = the $i$‑th row of $B^{-1}$, full stop — no $L$, no orthonormality. 

</details>

<details class="accordion" markdown="1">
<summary>Solution 1 (b)</summary>

**(b) Explicit computation.** With $e_1=\binom10,\ e_2=\binom11$,

$$
B=\begin{pmatrix}1&1\\0&1\end{pmatrix},\qquad
B^{-1}=\begin{pmatrix}1&-1\\0&1\end{pmatrix},
$$

so the rows give

$$
\boxed{\;\check e^1=(1,-1),\qquad \check e^2=(0,1).\;}
$$

Check: $\langle\check e^1,e_1\rangle=(1,-1)\!\cdot\!(1,0)=1$, $\langle\check e^1,e_2\rangle=(1,-1)\!\cdot\!(1,1)=0$, $\langle\check e^2,e_1\rangle=0$, $\langle\check e^2,e_2\rangle=1$. Note how $\check e^1$ already "knows" it must annihilate $e_2$ — that is the whole content of biorthogonality.

</details>

<details class="accordion" markdown="1">
<summary>Solution 1 (c) </summary>



</details>

<details class="accordion" markdown="1">
<summary>Visualisation — dual basis vs. canonical basis</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/sheet1_dual_basis.png' | relative_url }}" alt="Left: vectors e1=(1,0), e2=(1,1) and their dual basis covectors ě1=(1,-1), ě2=(0,1) plotted in R² with level sets of the covectors as faded diagonal and horizontal lines. Right: the canonical basis is its own dual under the canonical Euclidean inner product." loading="lazy">
  <figcaption>Left (b): the dual basis $\check e^1 = (1, -1)$, $\check e^2 = (0, 1)$ for the oblique basis $e_1, e_2$. Faded red lines are level sets of $\check e^1$ (diagonals along direction $e_2$); faded orange lines are level sets of $\check e^2$ (horizontals along $e_1$). Each covector $\check e^i$ kills $e_j$ for $j \ne i$. Right (c): the canonical basis under the canonical inner product is self-dual.</figcaption>
</figure>

</details>

### Orthogonal projections via $A^-$ and $A^+$

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
\mathrm{rank}(A) &\le \min\lbrace m, n\rbrace. &&\text{(0.3c)}
\end{aligned}
$$

Assume throughout that $A$ has *full rank*, i.e. equality in (0.3c).

**(i)** Suppose $A$ is *injective*. Derive the orthogonal projection onto $\mathrm{rge}(A)$ with respect to the inner product $m(\cdot, \cdot)$ on $\mathcal Y$. Interpret the orthogonal left-inverse $A^-$.

**(ii)** Suppose $A$ is *surjective*. Derive the orthogonal projection onto $\ker(A)$ with respect to the inner product $\ell(\cdot, \cdot)$ on $\mathcal X$. Interpret the orthogonal right-inverse $A^+$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 2 (i) — projection onto $\mathrm{rge}(A)$ and the meaning of $A^-$</summary>

**The picture first.** $A : \mathcal X \to \mathcal Y$ is injective with full rank, so $m \ge n$ and $\mathrm{rge}(A) \subset \mathcal Y$ is an $n$-dimensional subspace that is generally a *strict* subspace of $\mathcal Y$. Given an arbitrary $y \in \mathcal Y$, the equation $Ax = y$ has no solution unless $y$ already happens to live in $\mathrm{rge}(A)$. The natural "best replacement" is the closest point $\hat y \in \mathrm{rge}(A)$ to $y$, measured by the inner product $m$. The orthogonal projection onto $\mathrm{rge}(A)$ is the map $y \mapsto \hat y$. We will *derive* it from the least-squares problem and then *read off* $A^-$ from the formula.

**Why least squares finds the projection.** A classical fact: the orthogonal projection of $y$ onto a subspace $V$ (with respect to an inner product) is the unique minimiser of $v \mapsto \|v - y\|$ over $v \in V$. Since $A$ is injective, every $v \in \mathrm{rge}(A)$ has a unique pre-image $x \in \mathcal X$ with $v = Ax$, so parametrising by $x$ is harmless. This motivates:

**Step 1 — Variational formulation.** Minimise the squared $m$-distance from $Ax$ to $y$:

$$
J(x) \;:=\; \tfrac{1}{2}\,m(Ax - y, Ax - y) \qquad \text{over } x \in \mathcal X.
\tag{0.4}
$$

The factor $\tfrac12$ is cosmetic — it makes the derivative cleaner.

**Step 2 — First-order condition via directional derivative.** At a minimiser $x^\ast$, $J$ must be stationary in every direction $h \in \mathcal X$. Compute

$$
\frac{d}{dt}\bigg|_{t=0} J(x + th)
\;=\; \frac{d}{dt}\bigg|_{t=0} \tfrac12\, m(Ax - y + t Ah,\; Ax - y + t Ah)
\;=\; m(Ax - y,\, Ah)
$$

(the cross-term from bilinearity at $t = 0$).

**Step 3 — Convert $m(\cdot, \cdot)$ to a duality pairing, then move $A$ to the other side.** Using $m(u, v) = \langle Mu, v\rangle$,

$$
m(Ax - y, Ah) \;=\; \langle M(Ax - y),\, Ah\rangle.
$$

To remove $A$ from $Ah$ (which is in $\mathcal Y$) and have everything tested on $h$ (which is in $\mathcal X$), use the **transpose** $\widecheck A : \widecheck{\mathcal Y} \to \widecheck{\mathcal X}$, defined exactly by the property

$$
\langle p, Ax\rangle \;=\; \langle \widecheck A p, x\rangle \qquad \text{for all } p \in \widecheck{\mathcal Y},\ x \in \mathcal X.
$$

Applied with $p = M(Ax - y)$:

$$
\frac{d}{dt}\bigg|_{t=0} J(x + th) \;=\; \langle M(Ax - y),\, Ah\rangle \;=\; \langle \widecheck A M(Ax - y),\, h\rangle.
$$

**Step 4 — Setting the derivative to zero gives the normal equations.** This must hold for *every* $h$, which is equivalent to the covector itself being zero (a covector that pairs to $0$ with every $h \in \mathcal X$ is the zero covector). Hence

$$
\widecheck A M A\, x \;=\; \widecheck A M y.
\tag{0.5}
$$

**Step 5 — Solve the normal equations.** We claim the operator $\widecheck A M A : \mathcal X \to \widecheck{\mathcal X}$ is invertible. To see this, note it is the "Gramian" of $A$ in the $m$-metric: for any $x \in \mathcal X$,

$$
\langle \widecheck A M A\,x,\, x\rangle \;=\; \langle M (Ax),\, Ax\rangle \;=\; m(Ax, Ax) \;\ge\; 0,
$$

with equality iff $Ax = 0$ iff $x = 0$ (since $A$ is injective). So $\widecheck A M A$ is symmetric positive-definite, hence invertible. Therefore

$$
x^\ast \;=\; (\widecheck A M A)^{-1}\, \widecheck A M\, y.
$$

**Step 6 — Read off $A^-$ and the projection.** Comparing with (A.11) in the appendix,

$$
\boxed{\ A^- \;=\; (\widecheck A M A)^{-1}\, \widecheck A M \;\in\; \mathcal L(\mathcal Y, \mathcal X),\qquad x^\ast \;=\; A^- y.\ }
$$

Then the closest point in the range is $\hat y = Ax^\ast = A A^- y$, so

$$
\boxed{\ A A^- \;=\; \Pi_{\mathrm{rge}(A)} \quad\text{(orthogonal projection onto $\mathrm{rge}(A)$ in the $m$-metric)}.\ }
$$

**Sanity properties.** A projection has to satisfy three conditions: idempotence, self-adjointness (in the relevant metric), and the appropriate left/right inverse identity. Let's check all three.

  • *Idempotence.* The normal equations at $y$ and at $AA^- y$ give the same $x^\ast$: indeed $\widecheck A M A x^\ast = \widecheck A M y = \widecheck A M (AA^- y)$ (the second equality because $\widecheck A M A x^\ast = \widecheck A M y$ by definition), so $A^-(AA^- y) = A^- y$ and hence $(AA^-)^2 = AA^-$.

  • *$m$-self-adjointness.* For any $y, y' \in \mathcal Y$, write $x^\ast = A^- y$ and $x^{\ast\prime} = A^- y'$. Then
  
  $$
  m(AA^- y, y') = \langle M y', AA^- y\rangle = \langle M y', A x^\ast\rangle = \langle \widecheck A M y', x^\ast\rangle.
  $$
  
  Since $\widecheck A M y' = \widecheck A M A x^{\ast\prime}$ (normal equation at $y'$), this equals $\langle \widecheck A M A x^{\ast\prime}, x^\ast\rangle$, which is symmetric in $x^\ast, x^{\ast\prime}$ (because $\widecheck A M A$ is symmetric). Tracing back, this equals $m(y, AA^- y')$. So $AA^-$ is $m$-symmetric.

  Idempotence + self-adjointness in $m$ $\Longrightarrow$ orthogonal projection in the $m$-metric.

  • *Left-inverse.* For any $x \in \mathcal X$, $A^- A x = (\widecheck A M A)^{-1}(\widecheck A M A) x = x$. This is property (A.12a): if $y = Ax$ is already in the range, the least-squares solver returns the exact pre-image.

**Interpretation of $A^-$.** $A^-$ is the **least-squares solver** for the (generically over-determined) system $Ax = y$:

  - If $y \in \mathrm{rge}(A)$, then $A^- y$ is the unique exact pre-image.
  - If $y \notin \mathrm{rge}(A)$, then $A^- y$ is the unique $x$ minimising $\mu(Ax - y)$ — equivalently, the pre-image of the projection $\hat y = AA^- y$ of $y$ onto the range.

In matrix terms with $M = I$ (canonical inner product on $\mathcal Y$), the transpose $\widecheck A$ is just the matrix transpose $A^T$, and the boxed formula reduces to the textbook least-squares solution

$$
A^- = (A^T A)^{-1} A^T, \qquad \Pi_{\mathrm{rge}(A)} = A(A^T A)^{-1}A^T.
$$

The general $M$ replaces $A^T$ by $\widecheck A M$, which is the "$m$-weighted" transpose — the residual $Ax - y$ is measured by the metric $m$, so the metric enters the formula symmetrically through $M$. $\blacksquare$

</details>

<details class="accordion" markdown="1">
<summary>Solution 2 (ii) — projection onto $\ker(A)$ and the meaning of $A^+$</summary>

Again, be definition, orthogonal projection means minimizing the norm in the following form, which we for simplicity square and multiply my $\frac{1}{2}$:

$$x^*_\perp = \arg\min_{x_\perp \in \text{Ker}(A)} \frac{1}{2}\lVert x - x_\perp \rVert^2$$

Because optimizing over the set $\text{Ker}(A)$ systematically is not easy (how to do so?), we will optimize its Lagrangian in the following form:

$$x^*_\perp = \arg\min_{x_\perp \in \mathcal{X}} \frac{1}{2}\lVert x - x_\perp \rVert^2 + \langle y, Ax_\perp \rangle = \arg\min_{x_\perp \in \mathcal{X}} \mathcal{L}(x_\perp)$$

Optimality conditions are

$$\frac{\partial \mathcal{L}}{\partial x_\perp}(x^*_\perp) = 0 \qquad \frac{\partial \mathcal{L}}{\partial y}(y^*) = 0,$$

where we have

$$\frac{\partial \mathcal{L}}{\partial x_\perp} = x - L^{-1}\widecheck{A}\widecheck{y} \overset{!}{=} 0 \qquad \frac{\partial \mathcal{L}}{\partial y} = Ax \overset{!}{=} 0,$$

From the first expression we obtain $x^\ast_\pert = x - L^{-1}\widecheck{A}\widecheck{y}^\ast$, and after plugging it into the second expression we obtain $y^\ast = (AL^{-1}\widecheck{A})^{-1}Ax$, then 

$$x^\ast_\pert = x - L^{-1}\widecheck{A}\widecheck{y}^\ast = x - L^{-1}\widecheck{A}(AL^{-1}\widecheck{A})^{-1}Ax$$

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

where $\mathbb D = \lbrace\binom{u}{v} : u^2 + v^2 < 1\rbrace$. Show that

$$
T_{\tau(1/\sqrt 3,\,1/\sqrt 3)} S^2 \;=\; T_{\gamma(1/\sqrt 3,\,1/\sqrt 3)} S^2.
\tag{0.5}
$$

</div>

<details class="accordion" markdown="1">
<summary>Solution 3 — same point, same tangent plane</summary>

**Strategy.** The claim is that two charts of $S^2$ give the *same* tangent space at a common point. We do this in four steps: (1) confirm both charts hit the same point $p$; (2) compute the tangent space as $\mathrm{span}\lbrace\partial_u, \partial_v\rbrace$ for each chart; (3) show both spans equal a single 2-plane (the one orthogonal to $p$); (4) make the chart-change explicit. The takeaway: the tangent plane is **intrinsic** — it depends on the point $p$, not on which chart we used to land there.

**What does "tangent space from a chart" mean?** If $\gamma : U \subset \mathbb R^2 \to S^2$ is a chart with $\gamma(u_0, v_0) = p$, the differential $D\gamma_{(u_0, v_0)} : \mathbb R^2 \to \mathbb R^3$ is a linear map whose image is, by definition, the tangent space $T_p S^2$. In a basis, this image is the span of the column vectors $\partial_u \gamma$ and $\partial_v \gamma$ evaluated at $(u_0, v_0)$.

**Step 1 — Both charts hit the same point.** Evaluate $\gamma$ and $\tau$ at $(u, v) = (1/\sqrt 3, 1/\sqrt 3)$. The "hidden" third coordinate is $\sqrt{1 - 1/3 - 1/3} = \sqrt{1/3} = 1/\sqrt 3$. So

$$
p \;:=\; \gamma\!\left(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\right) \;=\; \begin{pmatrix} 1/\sqrt 3 \\ 1/\sqrt 3 \\ 1/\sqrt 3\end{pmatrix} \;=\; \tau\!\left(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\right).
$$

So both tangent spaces are taken at the **same point** $p \in S^2$ — only the *parametrisations* of the patch differ.

**Step 2 — Tangent space from $\gamma$.** Differentiate componentwise; the only non-trivial derivative is of the third entry $\sqrt{1 - u^2 - v^2}$:

$$
\partial_u \gamma(u, v) = \begin{pmatrix} 1 \\ 0 \\ -u/\sqrt{1 - u^2 - v^2} \end{pmatrix}, \qquad
\partial_v \gamma(u, v) = \begin{pmatrix} 0 \\ 1 \\ -v/\sqrt{1 - u^2 - v^2} \end{pmatrix}.
$$

At $(1/\sqrt 3, 1/\sqrt 3)$, the denominator $\sqrt{1 - u^2 - v^2} = 1/\sqrt 3$, so the ratio $u/\sqrt{1 - u^2 - v^2} = (1/\sqrt 3)/(1/\sqrt 3) = 1$. Hence

$$
\partial_u \gamma\bigl(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\bigr) = \begin{pmatrix} 1 \\ 0 \\ -1\end{pmatrix},\qquad
\partial_v \gamma\bigl(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\bigr) = \begin{pmatrix} 0 \\ 1 \\ -1\end{pmatrix},
$$

and therefore

$$
T_{\gamma(1/\sqrt 3,\,1/\sqrt 3)} S^2 \;=\; \mathrm{span}\!\left\lbrace(1, 0, -1)^T,\, (0, 1, -1)^T\right\rbrace.
$$

**Step 3 — Tangent space from $\tau$.** The chart $\tau$ stuffs the square-root into the *second* slot instead of the third, so the non-trivial derivative shifts accordingly:

$$
\partial_u \tau(u, v) = \begin{pmatrix} 1 \\ -u/\sqrt{1 - u^2 - v^2} \\ 0\end{pmatrix},\qquad
\partial_v \tau(u, v) = \begin{pmatrix} 0 \\ -v/\sqrt{1 - u^2 - v^2} \\ 1\end{pmatrix}.
$$

At $(1/\sqrt 3, 1/\sqrt 3)$,

$$
\partial_u \tau\bigl(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\bigr) = \begin{pmatrix} 1 \\ -1 \\ 0\end{pmatrix},\qquad
\partial_v \tau\bigl(\tfrac{1}{\sqrt 3}, \tfrac{1}{\sqrt 3}\bigr) = \begin{pmatrix} 0 \\ -1 \\ 1\end{pmatrix},
$$

and

$$
T_{\tau(1/\sqrt 3,\,1/\sqrt 3)} S^2 \;=\; \mathrm{span}\!\left\lbrace(1, -1, 0)^T,\, (0, -1, 1)^T\right\rbrace.
$$

**Step 4 — Both spans equal the plane $\lbrace v : v \cdot p = 0\rbrace$.** Here is the clean geometric reason both spans must be equal:

$S^2$ is the level set $\lbrace x \in \mathbb R^3 : \|x\|^2 = 1\rbrace$. The gradient of $\|x\|^2$ at $p$ is $2p$, so the tangent space to $S^2$ at $p$ is **the plane through the origin orthogonal to $p$** — this is independent of any chart. Let us verify that both spans land inside this plane (call it $H_p := \lbrace v \in \mathbb R^3 : v \cdot p = 0\rbrace$):

$$
\begin{aligned}
(1, 0, -1) \cdot p &= 1/\sqrt 3 - 1/\sqrt 3 = 0, &
(0, 1, -1) \cdot p &= 1/\sqrt 3 - 1/\sqrt 3 = 0, \\
(1, -1, 0) \cdot p &= 1/\sqrt 3 - 1/\sqrt 3 = 0, &
(0, -1, 1) \cdot p &= -1/\sqrt 3 + 1/\sqrt 3 = 0.
\end{aligned}
$$

So both spans lie inside $H_p$. Each pair is linearly independent (e.g. the $\gamma$-pair has the $2 \times 2$ minor $\begin{pmatrix}1 & 0\\0 & 1\end{pmatrix}$ with determinant $1$; same for the $\tau$-pair). A 2-dimensional subspace of a 2-dimensional space is the whole space, so both spans equal $H_p$. In particular they are equal to each other — this proves (0.5).

**Step 5 — Explicit change-of-basis between the two pairs.** It's instructive to write down the linear relation explicitly. By inspection:

$$
\begin{aligned}
(1, -1, 0) &= (1, 0, -1) - (0, 1, -1), \\
(0, -1, 1) &= -\,(0, 1, -1).
\end{aligned}
$$

So

$$
\begin{pmatrix} \partial_u \tau \\ \partial_v \tau\end{pmatrix} \;=\; \begin{pmatrix} 1 & -1 \\ 0 & -1\end{pmatrix} \begin{pmatrix} \partial_u \gamma \\ \partial_v \gamma\end{pmatrix}.
$$

**What is this matrix?** This is the standard "basis vectors transform contravariantly" story, but the conventions are worth checking carefully. Let $\psi := \gamma^{-1} \circ \tau$ be the transition map from $\tau$-coordinates $(u', v')$ to $\gamma$-coordinates $(u, v)$. Concretely, solving $\gamma(u, v) = \tau(u', v')$ gives $u = u'$, $v = \sqrt{1 - u'^2 - v'^2}$, so

$$
\psi(u', v') \;=\; \bigl(u',\ \sqrt{1 - u'^2 - v'^2}\bigr).
$$

Its Jacobian at $(1/\sqrt 3, 1/\sqrt 3)$ is

$$
J_\psi \;=\; \begin{pmatrix} 1 & 0 \\ -u'/\sqrt{1 - u'^2 - v'^2} & -v'/\sqrt{1 - u'^2 - v'^2}\end{pmatrix}\bigg|_{(1/\sqrt 3, 1/\sqrt 3)} \;=\; \begin{pmatrix} 1 & 0 \\ -1 & -1\end{pmatrix}.
$$

The standard chain-rule identity (basis vectors transform by the Jacobian of the transition map between the corresponding coordinates) reads $\partial/\partial u'^i = (\partial u^j / \partial u'^i)\, \partial/\partial u^j = (J_\psi)^j_{\ i}\, \partial/\partial u^j$. Written as a matrix equation with basis vectors stacked as a *column of rows*, this becomes

$$
\begin{pmatrix} \partial_u \tau \\ \partial_v \tau\end{pmatrix}
= J_\psi^T \begin{pmatrix} \partial_u \gamma \\ \partial_v \gamma\end{pmatrix}
= \begin{pmatrix} 1 & -1 \\ 0 & -1\end{pmatrix}\begin{pmatrix} \partial_u \gamma \\ \partial_v \gamma\end{pmatrix},
$$

so the matrix is **the transpose of the Jacobian of the transition map $\psi = \gamma^{-1} \circ \tau$**. This is the standard manifold-theoretic statement: tangent spaces are chart-independent because the Jacobian of any transition map is invertible (hence so is its transpose). $\blacksquare$

**Geometric reading.** At $p = (1/\sqrt 3, 1/\sqrt 3, 1/\sqrt 3)$, the tangent plane is the affine plane through $p$ normal to $p$ itself — a feature of the *sphere*, not of any chart. The two charts $\gamma$ and $\tau$ produce *different* coordinate bases of this same plane (compare $\lbrace(1,0,-1), (0,1,-1)\rbrace$ vs. $\lbrace(1,-1,0), (0,-1,1)\rbrace$), but the plane itself is intrinsic. This is the prototypical example of why manifolds are interesting: the same geometric object admits many charts, and the meaningful structures (tangent spaces, vector fields, metrics) live independently of any choice.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation — two charts of $S^2$ at the common point</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/sheet1_sphere_charts.png' | relative_url }}" alt="The unit sphere with two coloured chart patches (top hemisphere via γ, front hemisphere via τ) overlapping at p=(1/√3, 1/√3, 1/√3). Four tangent vectors emanate from p: two from γ, two from τ, all lying in the same yellow tangent plane normal to p." loading="lazy">
  <figcaption>The two charts $\gamma$ (blue patch, projects onto $z = \sqrt{1 - u^2 - v^2}$) and $\tau$ (green patch, projects onto $y = \sqrt{1 - u^2 - v^2}$) both contain the point $p = (1/\sqrt 3, 1/\sqrt 3, 1/\sqrt 3)$. The four partial-derivative vectors all lie in the yellow tangent plane $T_p S^2 = \lbrace v \cdot p = 0\rbrace$, and span it.</figcaption>
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

**Why we care.** On an abstract manifold there is no ambient $\mathbb R^N$, so a "tangent vector at $x$" cannot be defined as a velocity in some surrounding space. The derivation definition replaces "a direction at $x$" with "a linear map on smooth functions that obeys Leibniz" — and *this* definition makes sense without any embedding. To use this as a *tangent space* in linear algebra, we need it to be a vector space. That is what this exercise establishes.

**Strategy.** The space of *all* $\mathbb R$-linear maps $v : \mathcal F(\mathcal M) \to \mathbb R$ is automatically a real vector space (linear maps to $\mathbb R$ always form a vector space under pointwise operations — this is just the dual space $\mathrm{Hom}_{\mathbb R}(\mathcal F(\mathcal M), \mathbb R)$). Our $\mathcal D_x$ is the *subset* of those linear maps satisfying the Leibniz rule (0.6). So instead of re-proving the eight vector space axioms from scratch, we use the **subspace criterion**: a subset of a vector space is itself a vector space iff it contains $0$ and is closed under addition and scalar multiplication.

**Setup.** Let

$$
\mathcal D_x \;:=\; \bigl\lbrace\, v : \mathcal F(\mathcal M) \to \mathbb R \;\big|\; v\ \text{is}\ \mathbb R\text{-linear and satisfies (0.6)}\,\bigr\rbrace
\;\subset\; \mathrm{Hom}_{\mathbb R}(\mathcal F(\mathcal M), \mathbb R).
$$

We will check three things: $0 \in \mathcal D_x$, $\mathcal D_x$ is closed under addition, $\mathcal D_x$ is closed under scalar multiplication. Crucially, both linearity and Leibniz are *linear* conditions on $v$, so they get preserved under linear combinations — this is the underlying reason the proof works.

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

**Step 4 — vector-space axioms inherited.** This is the heart of the subspace criterion: associativity and commutativity of $+$, distributivity, the unit law $1 \cdot v = v$, and existence of inverses $(-1) \cdot v = -v$ all hold in $\mathrm{Hom}\_{\mathbb R}(\mathcal F(\mathcal M), \mathbb R)$ pointwise (linear maps to $\mathbb R$ form a vector space under pointwise operations). Steps 1–3 showed that $\mathcal D_x$ contains $0$ and is closed under addition and scalar multiplication, so all these axioms automatically restrict from the ambient space — no separate verification needed. Hence $\mathcal D_x$ is a real vector space. $\blacksquare$

**Why this matters — connection to Exercise 3.** The vector-space structure proved here is what makes $T_x \mathcal M$ a *linear* tangent space, which in turn:

  - makes the differential $df_x : T_x \mathcal M \to T_{f(x)} \mathcal N$ a linear map (not just a set-theoretic function), giving the chain rule its tensorial form;
  - admits a coordinate basis: the partial-derivative derivations $\partial_{u^i}\bigr\rvert_x$ in any chart span $T_x \mathcal M$;
  - matches Exercise 3 perfectly — the four vectors $\partial_u \gamma, \partial_v \gamma, \partial_u \tau, \partial_v \tau$ are concrete instances of such basis derivations, and Step 5 of that exercise was exactly the change-of-basis formula between two such coordinate frames.

So the abstract construction here (Leibniz $\Rightarrow$ vector space) and the concrete computation there (two charts of $S^2$) are two sides of the same coin: the tangent space is a vector space, charts give bases, transition maps give change-of-basis matrices.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation — derivations close under linear combinations</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/geometric_deep_learning_hd/sheet1_derivations.png' | relative_url }}" alt="Three rows of arrows from F(M) to R: the top labelled v, the middle labelled w, the bottom labelled αv + βw, with the Leibniz identity for the combined derivation displayed below." loading="lazy">
  <figcaption>Two derivations $v, w$ at $x$ combine $\mathbb R$-linearly into $\alpha v + \beta w$. The Leibniz rule is preserved by linear combinations because it is linear in the derivation: $(\alpha v + \beta w)(fg) = f(\alpha v + \beta w)(g) + g(\alpha v + \beta w)(f)$. This is what makes $T_x \mathcal M$ a vector space.</figcaption>
</figure>

</details>

---

## Exercise Sheet 2 — Generalized inverses, SVD, interpolation, and semidefinite projections

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 1</span><span class="math-callout__name">(Dualizing orthogonal partial inverses)</span></p>

Consider the first relation of Proposition A.4(a), i.e. the equation

$$
\widecheck{A^{-}} = \widecheck{A}^{+}.
$$

**(i)** Check that the linear mappings on both sides are defined on the same space.

**(ii)** Compute the explicit expression of the operator $\widecheck{A^{-}}$, i.e. dualize the orthogonal left-inverse of $A$, based on the equation

$$
\langle \widecheck{x}, A^{-}y\rangle = \langle \widecheck{A^{-}}\widecheck{x}, y\rangle,
\qquad
\forall\, \widecheck{x} \in \widecheck{\mathcal X},\ y \in \mathcal Y.
\tag{0.1}
$$

**(iii)** Compute the explicit expression of the orthogonal right-inverse $\widecheck A^{+}$ of the dual operator $\widecheck A$ of $A$ through "symbol substitution": turn the given expression for the mapping $A \mapsto A^{+}$ (Definition A.3(b)) into the desired mapping $\widecheck A \mapsto \widecheck A^{+}$ by taking into account the spaces $\mathcal X, \mathcal Y$ of $A \in \mathcal L(\mathcal X,\mathcal Y)$ and the mapping $L^{-1} : \widecheck{\mathcal X} \to \mathcal X$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 1 (i) — both sides live in $\mathcal L(\widecheck{\mathcal X}, \widecheck{\mathcal Y})$</summary>

We track the arrows. Throughout, the transpose of $T \in \mathcal L(\mathcal U, \mathcal V)$ is the operator $\widecheck T \in \mathcal L(\widecheck{\mathcal V}, \widecheck{\mathcal U})$ fixed by

$$
\langle p, T u\rangle = \langle \widecheck T p,\, u\rangle, \qquad \forall\, p \in \widecheck{\mathcal V},\ u \in \mathcal U.
$$

**Left-hand side $\widecheck{A^{-}}$.** Here $A \in \mathcal L(\mathcal X, \mathcal Y)$ is injective, so by (A.11) its orthogonal left-inverse is $A^{-} \in \mathcal L(\mathcal Y, \mathcal X)$. Dualizing an operator swaps domain and codomain and replaces each space by its dual, so

$$
A^{-} \in \mathcal L(\mathcal Y, \mathcal X)
\quad\Longrightarrow\quad
\widecheck{A^{-}} \in \mathcal L(\widecheck{\mathcal X}, \widecheck{\mathcal Y}).
$$

**Right-hand side $\widecheck A^{+}$.** The transpose $\widecheck A \in \mathcal L(\widecheck{\mathcal Y}, \widecheck{\mathcal X})$. Since $A$ is injective, $\widecheck A$ is *surjective* (**the transpose of an injection is a surjection**), so its orthogonal right-inverse $\widecheck A^{+}$ from Definition A.3(b) is defined. A right-inverse of a map $\widecheck{\mathcal Y} \to \widecheck{\mathcal X}$ goes the other way:

$$
\widecheck A \in \mathcal L(\widecheck{\mathcal Y}, \widecheck{\mathcal X})
\quad\Longrightarrow\quad
\widecheck A^{+} \in \mathcal L(\widecheck{\mathcal X}, \widecheck{\mathcal Y}).
$$

**Conclusion.** Both $\widecheck{A^{-}}$ and $\widecheck A^{+}$ are elements of $\mathcal L(\widecheck{\mathcal X}, \widecheck{\mathcal Y})$, so the equation $\widecheck{A^{-}} = \widecheck A^{+}$ is at least *well-typed* — it compares two maps with the same domain and codomain. Parts (ii) and (iii) compute each side and show they coincide.

</details>

<details class="accordion" markdown="1">
<summary>Solution 1 (ii) — dualizing the left-inverse $A^{-}$ via the pairing (0.1)</summary>

**Two facts we will use.** The duality map $M$ is *symmetric*, $\langle M y, y'\rangle = m(y, y') = m(y', y) = \langle M y', y\rangle$, i.e. $\widecheck M = M$ (and likewise $\widecheck L = L$); and the double dual is the identity, $\widecheck{\widecheck A} = A$, under the reflexive identification $\widecheck{\widecheck{\mathcal X}} \cong \mathcal X$. TODO: the duality map M is symmetric, but why? Does any M\in\mathcal{L}(X,\widecheck X) induce a duality mapping? Is any M\in\mathcal{L}(X,\widecheck X) symmetric?

Abbreviate the symmetric "$m$-Gramian" of $A$ by

$$
N := \widecheck A M A \in \mathcal L(\mathcal X, \widecheck{\mathcal X}),
\qquad
\widecheck N = \widecheck A\, \widecheck M\, \widecheck{\widecheck A} = \widecheck A M A = N,
$$

so $\widecheck{N^{-1}} = (\widecheck N)^{-1} = N^{-1}$. With this, $A^{-} = N^{-1}\widecheck A M$ by (A.11).

**Peel the operators off the right argument of (0.1).** Starting from the right side of $(0.1)$ and applying the transpose property once per factor:

$$
\begin{aligned}
\langle \widecheck x,\, A^{-} y\rangle
&= \langle \widecheck x,\; N^{-1}\,\widecheck A M\, y\rangle
&&\text{(definition of } A^{-}) \\
&= \langle \widecheck{N^{-1}}\,\widecheck x,\; \widecheck A M\, y\rangle
 = \langle N^{-1}\widecheck x,\; \widecheck A M\, y\rangle
&&(\widecheck{N^{-1}} = N^{-1}) \\
&= \langle M y,\; A\, N^{-1}\widecheck x\rangle
&&\text{(transpose of } \widecheck A,\ \text{using } \widecheck{\widecheck A}=A) \\
&= m\bigl(y,\, A N^{-1}\widecheck x\bigr)
 = m\bigl(A N^{-1}\widecheck x,\, y\bigr)
 = \langle M A\, N^{-1}\widecheck x,\; y\rangle.
&&(m \text{ symmetric})
\end{aligned}
$$

**Read off the transpose.** Comparing the last line with the defining identity $(0.1)$, $\langle \widecheck x, A^{-} y\rangle = \langle \widecheck{A^{-}}\widecheck x, y\rangle$, which must hold for all $\widecheck x \in \widecheck{\mathcal X}$, $y \in \mathcal Y$, gives

$$
\boxed{\ \widecheck{A^{-}} \;=\; M A\, N^{-1} \;=\; M A\,(\widecheck A M A)^{-1} \;\in\; \mathcal L(\widecheck{\mathcal X}, \widecheck{\mathcal Y}).\ }
$$

The codomain check matches part (i): $(\widecheck A M A)^{-1} : \widecheck{\mathcal X} \to \mathcal X$, then $A : \mathcal X \to \mathcal Y$, then $M : \mathcal Y \to \widecheck{\mathcal Y}$.

</details>

<details class="accordion" markdown="1">
<summary>Solution 1 (iii) — $\widecheck A^{+}$ by symbol substitution into Definition A.3(b)</summary>

**The template.** Definition A.3(b) gives the orthogonal right-inverse of a *surjective* operator $A \in \mathcal L(\mathcal X, \mathcal Y)$ as

$$
A^{+} = L^{-1}\,\widecheck A\,\bigl(A\, L^{-1}\,\widecheck A\bigr)^{-1},
$$

built from three ingredients: the operator $A$ itself, its transpose $\widecheck A$, and the inverse duality map $L^{-1} : \widecheck{\mathcal X} \to \mathcal X$ of its **domain** $\mathcal X$.

**The substitution.** We now feed the operator $\widecheck A \in \mathcal L(\widecheck{\mathcal Y}, \widecheck{\mathcal X})$ (surjective, by part (i)) into the same template. We must re-read each ingredient for $\widecheck A$:

| ingredient for $A$ | ingredient for $\widecheck A$ |
|---|---|
| the operator $A$ | the operator $\widecheck A$ |
| its transpose $\widecheck A$ | its transpose $\widecheck{\widecheck A} = A$ |
| $L^{-1}$, inverse duality map of the domain $\mathcal X$ | inverse duality map of the domain $\widecheck{\mathcal Y}$ |

The only subtle entry is the last. The domain of $\widecheck A$ is $\widecheck{\mathcal Y}$, whose scalar product is the *dual* scalar product $\widecheck m(p, p') = \langle p, M^{-1} p'\rangle$ from (A.5a); its duality map is therefore $M^{-1} : \widecheck{\mathcal Y} \to \mathcal Y$. So the "$L^{-1}$" of the template — the *inverse* of the domain's duality map — becomes $M : \mathcal Y \to \widecheck{\mathcal Y}$.

**Assemble.** Substituting $A \rightsquigarrow \widecheck A$, $\ \widecheck A \rightsquigarrow A$, $\ L^{-1} \rightsquigarrow M$:

$$
\widecheck A^{+}
= \underbrace{M}_{L^{-1}}\,\underbrace{A}_{\widecheck{\widecheck A}}\,\Bigl(\underbrace{\widecheck A}_{A}\,\underbrace{M}_{L^{-1}}\,\underbrace{A}_{\widecheck{\widecheck A}}\Bigr)^{-1}
$$

i.e.

$$
\boxed{\ \widecheck A^{+} \;=\; M A\,(\widecheck A M A)^{-1} \;\in\; \mathcal L(\widecheck{\mathcal X}, \widecheck{\mathcal Y}).\ }
$$

**Conclusion.** This is *exactly* the expression obtained for $\widecheck{A^{-}}$ in part (ii). Hence

$$
\widecheck{A^{-}} \;=\; M A\,(\widecheck A M A)^{-1} \;=\; \widecheck A^{+},
$$

which is the first relation (A.15a) of Proposition A.4(a). The two operations — "dualize the left-inverse" and "right-invert the dual" — produce the same map.

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 2</span><span class="math-callout__name">(SVD factorization and pseudo-inverse)</span></p>

The **singular value decomposition (SVD)** of a matrix $A \in \mathcal L(\mathbb R^n, \mathbb R^m) \cong \mathbb R^{m \times n}$ with $\operatorname{rank}(A) = r$ reads

$$
A = U\Sigma V^{\top},
\tag{0.2}
$$

with

$$
\begin{aligned}
U &= (U_1, U_2) = (u_1, \dots, u_m) \in \mathrm O(m),
&
U^{\top}U &= UU^{\top} = I_m,
&&\text{(0.3a)} \\
V &= (V_1, V_2) = (v_1, \dots, v_n) \in \mathrm O(n),
&
V^{\top}V &= VV^{\top} = I_n,
&&\text{(0.3b)}
\end{aligned}
$$

such that

$$
U^{\top}AV =
\begin{pmatrix}
U_1^{\top}AV_1 & U_1^{\top}AV_2 \\
U_2^{\top}AV_1 & U_2^{\top}AV_2
\end{pmatrix}
= \Sigma =
\begin{pmatrix}
\Sigma_r & 0_{r \times (n-r)} \\
0_{(m-r) \times r} & 0_{(m-r) \times (n-r)}
\end{pmatrix},
\tag{0.4}
$$

and

$$
\Sigma_r = \operatorname{Diag}(\sigma_1, \dots, \sigma_r).
\qquad
\text{(singular values)}
\tag{0.5}
$$

In particular, one has with

$$
\mathcal N(A) := \ker(A)
\tag{0.6}
$$

$$
\begin{aligned}
\operatorname{span}\{v_1, \dots, v_r\} &= \mathcal N(A)^{\perp},
&
\operatorname{span}\{v_{r+1}, \dots, v_n\} &= \mathcal N(A),
&&\text{(0.7a)} \\
\operatorname{span}\{u_1, \dots, u_r\} &= \mathcal R(A),
&
\operatorname{span}\{u_{r+1}, \dots, u_m\} &= \mathcal R(A)^{\perp},
&&\text{(0.7b)} \\
V_1V_1^{\top} &= \Pi_{\mathcal N(A)^{\perp}},
&
V_2V_2^{\top} &= \Pi_{\mathcal N(A)},
&&\text{(0.7c)} \\
U_1U_1^{\top} &= \Pi_{\mathcal R(A)},
&
U_2U_2^{\top} &= \Pi_{\mathcal R(A)^{\perp}}.
&&\text{(0.7d)}
\end{aligned}
$$

**(i)** Use the SVD of $A$ and provide a factorization

$$
A = BC
\tag{0.8}
$$

analogous to Eq. (A.18) of Definition A.5 (with interchanged roles of $A$ and $C$), i.e. with *injective* factor $B$ and *surjective* factor $C$, which is equivalent to $\operatorname{rank}(B) = \operatorname{rank}(C) = \operatorname{rank}(A) = r$.

**(ii)** Compute the pseudo-inverse $A^{\dagger}$ as specified by Equation (A.18), assuming *self-duality*, i.e. the linear duality mappings associated with $B$ and $C$ are the unit matrices of appropriate dimensions.

**(iii)** Interpret the resulting factorization of $A^{\dagger}$ based on your result of exercise 2 on sheet 1.

**(iv)** Now take into account the duality mappings $L : \mathcal X \to \widecheck{\mathcal X}$ and $M : \mathcal Y \to \widecheck{\mathcal Y}$. Additionally, we define the latent space $\mathcal Z$,

$$
C \in \mathcal L(\mathcal X, \mathcal Z),
\qquad
B \in \mathcal L(\mathcal Z, \mathcal Y)
\tag{0.9}
$$

and the duality mapping

$$
R : \mathcal Z \to \widecheck{\mathcal Z}.
\tag{0.10}
$$

Compute the explicit expression of the generalized inverse $A^{\dagger}$.

*Hint.* Apply the result of exercise 1 on this sheet.

</div>

<details class="accordion" markdown="1">
<summary>Solution 2 (i) — the thin SVD as an injective $\times$ surjective factorization</summary>

**Throw away the zero blocks.** Multiplying out (0.2) with the block structure (0.4) keeps only the top-left $\Sigma_r$ block, so only the first $r$ columns of $U$ and $V$ survive:

$$
A = U\Sigma V^{\top} = U_1\,\Sigma_r\,V_1^{\top},
\qquad
U_1 = (u_1, \dots, u_r) \in \mathbb R^{m \times r},\quad
V_1 = (v_1, \dots, v_r) \in \mathbb R^{n \times r}.
$$

This **thin SVD** already exhibits $A$ as a product passing through the $r$-dimensional latent space $\mathcal Z = \mathbb R^r$. Split it as

$$
\boxed{\ A = B C, \qquad B := U_1\Sigma_r \in \mathbb R^{m \times r}, \qquad C := V_1^{\top} \in \mathbb R^{r \times n}.\ }
$$

**Why this is the required factorization.** Definition A.5 wants an *injective* factor and a *surjective* factor, equivalently $\operatorname{rank}(B) = \operatorname{rank}(C) = r$:

  - $C = V_1^{\top} : \mathbb R^n \to \mathbb R^r$ has orthonormal rows ($V_1^{\top}V_1 = I_r$ by (0.3b)), so it has full row rank $r$ — **surjective**. Its kernel is $\operatorname{span}\lbrace v_{r+1}, \dots, v_n\rbrace = \mathcal N(A)$ by (0.7a), matching $\ker(C) = \ker(A)$.
  - $B = U_1\Sigma_r : \mathbb R^r \to \mathbb R^m$ is the product of $U_1$ (orthonormal columns, full column rank) and the invertible diagonal $\Sigma_r$, so it has full column rank $r$ — **injective**. Its range is $\operatorname{span}\lbrace u_1, \dots, u_r\rbrace = \mathcal R(A)$ by (0.7b), matching $\mathrm{rge}(B) = \mathrm{rge}(A)$.

(The split is not unique — any $B = U_1\Sigma_r S^{-1}$, $C = S V_1^{\top}$ with $S$ invertible works; the choice $S = I_r$ is the cleanest.)

</details>

<details class="accordion" markdown="1">
<summary>Solution 2 (ii) — the self-dual pseudo-inverse is the Moore–Penrose inverse</summary>

By Definition A.5 with the roles of $A$ and $C$ interchanged (here our operator is $A = BC$ with injective $B$, surjective $C$), the pseudo-inverse is

$$
A^{\dagger} = C^{+} B^{-}.
$$

**Self-duality.** "The duality mappings associated with $B$ and $C$ are unit matrices" means all inner products are the canonical Euclidean ones, $L = M = R = I$. Then every transpose $\widecheck{(\cdot)}$ is the ordinary matrix transpose, and (A.11), (A.13) collapse to the textbook one-sided inverses:

$$
B^{-} = (B^{\top}B)^{-1}B^{\top},
\qquad
C^{+} = C^{\top}(C C^{\top})^{-1}.
$$

**Plug in the SVD factors.** Using $U_1^{\top}U_1 = I_r$ and $V_1^{\top}V_1 = I_r$:

$$
\begin{aligned}
B^{\top}B &= \Sigma_r U_1^{\top}U_1\Sigma_r = \Sigma_r^{2}
&&\Longrightarrow&&
B^{-} = \Sigma_r^{-2}\,\Sigma_r U_1^{\top} = \Sigma_r^{-1}U_1^{\top}, \\
C C^{\top} &= V_1^{\top}V_1 = I_r
&&\Longrightarrow&&
C^{+} = V_1\,(I_r)^{-1} = V_1.
\end{aligned}
$$

Therefore

$$
\boxed{\ A^{\dagger} = C^{+}B^{-} = V_1\,\Sigma_r^{-1}\,U_1^{\top}
= V\Sigma^{\dagger}U^{\top},\quad
\Sigma^{\dagger} = \begin{pmatrix}\Sigma_r^{-1} & 0 \\ 0 & 0\end{pmatrix} \in \mathbb R^{n \times m}.\ }
$$

This is exactly the **Moore–Penrose pseudo-inverse**: invert the nonzero singular values, transpose the shape, leave the zero blocks zero.

</details>

<details class="accordion" markdown="1">
<summary>Solution 2 (iii) — reading $A^{\dagger} = C^{+}B^{-}$ through Sheet 1, Exercise 2</summary>

On Sheet 1 we found that the two one-sided inverses are *solvers* with complementary jobs:

  - $B^{-}$ is the **least-squares / range-projection** operator: $B B^{-} = \Pi_{\mathrm{rge}(B)}$ in the codomain metric ($AA^{-} = \Pi_{\mathrm{rge}(A)}$ there);
  - $C^{+}$ is the **minimum-norm** operator: $C^{+}C = \Pi_{\ker(C)^{\perp}}$ in the domain metric ($I - A^+A = \Pi_{\ker(A)}$ there).

The factorization $A^{\dagger} = C^{+}B^{-}$ glues these two halves, and the SVD form $A^{\dagger}y = V_1\Sigma_r^{-1}U_1^{\top}y$ makes the pipeline visible when read **right to left**:

  1. $U_1^{\top}y$ — read off the components of $y$ along $u_1, \dots, u_r$, i.e. project onto $\mathcal R(A)$ and discard the part in $\mathcal R(A)^{\perp} = \operatorname{span}\lbrace u_{r+1}, \dots, u_m\rbrace$. This is the **least-squares step** carried by $B^{-}$.
  2. $\Sigma_r^{-1}$ — undo the bijective core $\underline A$ by dividing each latent coordinate by its singular value $\sigma_i$.
  3. $V_1\,(\cdot)$ — map back into $\mathcal X$, landing inside $\operatorname{span}\lbrace v_1, \dots, v_r\rbrace = \mathcal N(A)^{\perp}$. This is the **minimum-norm step** carried by $C^{+}$: among all preimages it picks the one orthogonal to $\ker(A)$.

The two projection identities confirm it (using (0.7c)–(0.7d) and $U_1^{\top}U_1 = V_1^{\top}V_1 = I_r$):

$$
A A^{\dagger} = U_1\Sigma_r\big(V_1^{\top}V_1\big)\Sigma_r^{-1}U_1^{\top} = U_1U_1^{\top} = \Pi_{\mathcal R(A)},
\qquad
A^{\dagger}A = V_1\Sigma_r^{-1}\big(U_1^{\top}U_1\big)\Sigma_r V_1^{\top} = V_1V_1^{\top} = \Pi_{\mathcal N(A)^{\perp}}.
$$

So $A^{\dagger}$ realizes the Sheet 1 recipe in one map: *project the data onto the range (least squares), then return the minimum-norm preimage* — precisely the picture of Definition A.5.

</details>

<details class="accordion" markdown="1">
<summary>Solution 2 (iv) — general duality maps: the latent metric $R$ drops out</summary>

Now restore arbitrary inner products $L$ on $\mathcal X$, $M$ on $\mathcal Y$, and $R$ on the latent space $\mathcal Z$. Again $A^{\dagger} = C^{+}B^{-}$, but now each factor must use the *correct* metric. Exercise 1 of this sheet is exactly the bookkeeping device for that: it showed that the left-inverse $X^{-}$ is built from the **codomain** metric of $X$, while the right-inverse $X^{+}$ is built from the **domain** metric of $X$. Apply that reading to the two factors:

  - $B \in \mathcal L(\mathcal Z, \mathcal Y)$ is injective, so by (A.11) its left-inverse uses the metric on its codomain $\mathcal Y$, namely $M$:

$$
B^{-} = (\widecheck B\, M\, B)^{-1}\,\widecheck B\, M \;\in\; \mathcal L(\mathcal Y, \mathcal Z).
$$

  - $C \in \mathcal L(\mathcal X, \mathcal Z)$ is surjective, so by (A.13) its right-inverse uses the metric on its domain $\mathcal X$, namely $L$:

$$
C^{+} = L^{-1}\,\widecheck C\,(C\, L^{-1}\,\widecheck C)^{-1} \;\in\; \mathcal L(\mathcal Z, \mathcal X).
$$

Composing,

$$
\boxed{\ A^{\dagger} = C^{+}B^{-}
= L^{-1}\,\widecheck C\,\bigl(C\, L^{-1}\,\widecheck C\bigr)^{-1}\,\bigl(\widecheck B\, M\, B\bigr)^{-1}\,\widecheck B\, M
\;\in\; \mathcal L(\mathcal Y, \mathcal X).\ }
$$

**The punchline.** The latent-space metric $R$ **never appears**: $B^{-}$ only sees $M$ (codomain of $B$) and $C^{+}$ only sees $L$ (domain of $C$). This is exactly as it must be — $A^{\dagger}$ is determined by $A$ together with the geometries $L, M$ of the spaces it actually connects, and is independent of how we metrize the auxiliary factorization through $\mathcal Z$. Setting $L = M = R = I$ recovers part (ii).

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3</span><span class="math-callout__name">(Interpolation property of $W_{\mathcal D}$)</span></p>

Show that the operator $W_{\mathcal D}$ defined by Theorem 1.2 satisfies

$$
W_{\mathcal D}x_i = y_i,
\tag{0.11}
$$

where $(x_i, y_i)$ is any fixed input-output pair in the data (training) set $\mathcal D$ which defines $W_{\mathcal D}$.

*Hint.* Equations (A.42), (A.43) regarding the action of $W_{\mathcal D}$ on $x_i$; inspect the definition of the coefficients $G^{kl}$, $k,l \in [n]$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 3 — the inverse Gramian collapses the correction to the $i$-th residual</summary>

Using Theorem 1.2,

$$W_{\mathcal D} = U-G^{kl}(Lx_l)\otimes(Ux_k-y_k).$$

Fix $i\in[n]$. Apply $W_{\mathcal D}$ to $x_i$:

W_{\mathcal D}x_i = Ux_i = G^{kl}\bigl((Lx_l)\otimes(Ux_k-y_k)\bigr)(x_i).

By the tensor-action rule,

$$(p\otimes y)(x)=p(x)y,$$

so

$$\bigl((Lx_l)\otimes(Ux_k-y_k)\bigr)(x_i) = \langle Lx_l,x_i\rangle (Ux_k-y_k).$$

But

$$\langle Lx_l,x_i\rangle=\ell(x_l,x_i)=G_{li}.$$

Therefore

$$W_{\mathcal D}x_i = Ux_i-G^{kl}G_{li}(Ux_k-y_k).$$

Since $(G^{kl})$ is the inverse Gramian,

$$G^{kl}G_{li}=\delta_i^k.$$

Hence

$$W_{\mathcal D}x_i = Ux_i-\delta_i^k(Ux_k-y_k) = Ux_i-(Ux_i-y_i) = y_i.$$

So

$$\boxed{W_{\mathcal D}x_i=y_i.}$$

The whole proof is just: apply the rank-one/tensor action formula, then use that $G^{kl}$ is the inverse of $G_{kl}$. The theorem defines $G_{kl}$, $G^{kl}$, and $W_{\mathcal D}$ exactly in this way. 

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4</span><span class="math-callout__name">(Projection onto the positive semidefinite cone)</span></p>

Let $\mathbb S_{\succeq 0}^{n} \subset \mathbb R^{n \times n}$ denote the convex cone of symmetric and positive semidefinite matrices.

**(i)** Let $\widetilde L \in \mathbb R^{n \times n}$. Prove that the orthogonal projection

$$
L := \Pi_{\mathbb S_{\succeq 0}^{n}}(\widetilde L)
\tag{0.12}
$$

of $\widetilde L$ onto $\mathbb S_{\succeq 0}^{n} \subset \mathbb R^{n \times n}$ with respect to the canonical inner product $\langle A, B\rangle = \operatorname{tr}(A^{\top}B)$ is given by

$$
L = V\Lambda_{+}V^{\top}
= V\operatorname{Diag}([\lambda_1]_{+}, \dots, [\lambda_n]_{+})V^{\top},
\tag{0.13}
$$

where $\widetilde L = V\Lambda V^{\top} = V\operatorname{Diag}(\lambda_1, \dots, \lambda_n)V^{\top}$ is the spectral decomposition of $\widetilde L$ and the function $[\cdot]\_{+} : \mathbb R \to \mathbb R_{\ge 0}$ maps negative numbers to $0$.

*Hint.* Apply the *low-rank approximation theorem*: for any matrices $A, B \in \mathbb R^{m \times n}$, choosing $B$ as the best $k$-rank approximation of $A$ yields

$$
\lVert A - B\rVert_F^2 =
\sum_{i=k+1}^{\operatorname{rank}(A)} \sigma_i^2,
\tag{0.14}
$$

where $\sigma_i$, $i \in [r]$ are the singular values of $A$.

**(ii)** Visualize the set

$$
\mathbb S_{\succeq 0}^{2} \subset \mathbb R^{2 \times 2}
\tag{0.15}
$$

of all positive semidefinite $2 \times 2$ matrices, which have three degrees of freedom. Two natural options for a visualization are:

**(a)** Use the vectors

$$
(L_{11}, L_{22}, L_{12})^{\top}
\tag{0.16}
$$

in a bounded subset for representing corresponding matrices $L = (L_{ij})\_{i,j \in [2]}$ and apply the definition $L \in \mathbb S_{\succeq 0}^{2}$ for determining which vectors represent matrices in $\mathbb S_{\succeq 0}^{2}$.

**(b)** Exploit the mapping

$$
\exp_m : \mathbb S^{2} \to \mathbb S_{\succeq 0}^{2},
\qquad
L = \exp_m(S),
\qquad
S =
\begin{pmatrix}
S_{11} & S_{12} \\
S_{12} & S_{22}
\end{pmatrix}
\tag{0.17}
$$

from the vector space of symmetric $2 \times 2$ matrices to $\mathbb S_{\succeq 0}^{2}$ given by

$$
\begin{aligned}
L_{11}
&= e^{\frac{1}{2}(S_{11}+S_{22})}
\left(
\frac{(S_{11}-S_{22})\sinh\left(\frac{1}{2}\sqrt{4S_{12}^{2}+(S_{11}-S_{22})^{2}}\right)}
{\sqrt{4S_{12}^{2}+(S_{11}-S_{22})^{2}}}
+
\cosh\left(\frac{1}{2}\sqrt{4S_{12}^{2}+(S_{11}-S_{22})^{2}}\right)
\right),
&&\text{(0.18a)} \\
L_{22}
&= e^{\frac{1}{2}(S_{11}+S_{22})}
\left(
\frac{(S_{22}-S_{11})\sinh\left(\frac{1}{2}\sqrt{4S_{12}^{2}+(S_{11}-S_{22})^{2}}\right)}
{\sqrt{4S_{12}^{2}+(S_{11}-S_{22})^{2}}}
+
\cosh\left(\frac{1}{2}\sqrt{4S_{12}^{2}+(S_{11}-S_{22})^{2}}\right)
\right),
&&\text{(0.18b)} \\
L_{12}
&=
\frac{2e^{\frac{1}{2}(S_{11}+S_{22})}S_{12}\sinh\left(\frac{1}{2}\sqrt{4S_{12}^{2}+(S_{11}-S_{22})^{2}}\right)}
{\sqrt{4S_{12}^{2}+(S_{11}-S_{22})^{2}}},
&&\text{(0.18c)}
\end{aligned}
$$

and visualize the image of the unit ball $\mathbb B_1(0)$

$$
\mathbb R^3 \supset \mathbb B_1(0) \ni (S_{11}, S_{22}, S_{12})^{\top}
\mapsto
(L_{11}, L_{22}, L_{12})^{\top}.
\tag{0.19}
$$

Do both visualizations show the same set?

**(iii)** Consider the set of points in the 2D unit ball $\mathbb B_1(0) \subset \mathbb R^2$ (file: `UnitBallSamples.csv`) and their images (file: `UnitBallSamples-Transformed.csv`)

$$
\mathbb B_1(0) \ni x_i \mapsto \widetilde{x}_i = Lx_i,
\qquad
i \in [N]
\tag{0.20}
$$

as depicted by Figure 1.

**Figure 1.** Samples in the unit ball (left panel) mapped by an unknown linear transform (right panel).

Estimate numerically the duality map $L \in \mathbb R^{2 \times 2}$ by solving a suitable optimization problem and taking into account the properties of $L$. A simple numerical method is projected gradient descent using the projection map from subtask (i). Use the initialization $L_{(0)} = \begin{pmatrix}0 & 0 \\ 0 & 0\end{pmatrix}$.

*Hint.* Apply the derivative rules

$$
\begin{aligned}
\partial_X \operatorname{tr}(XA) &= A^{\top},
&&\text{(0.21a)} \\
\partial_X \operatorname{tr}(X^{2}A) &= (XA + AX)^{\top}.
&&\text{(0.21b)}
\end{aligned}
$$

</div>

---

## Exercise Sheet 3

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 1</span></p>

Let $\mathcal X = \mathbb R^d$ with $d \ge n$ and the scalar product $\langle Lx, x'\rangle$, where $\langle \cdot, \cdot\rangle$ is the canonical Euclidean scalar product and the matrix $0 \prec L \in \mathcal P^d \subset \mathbb R^{d \times d}$ is symmetric and positive-definite. Define the input and output data matrices

$$
X := (x_1, \dots, x_n) \in \mathbb R^{d \times n}
\tag{0.1a}
$$

$$
Y := (y_1, \dots, y_n) \in \mathbb R^{d \times n}.
\tag{0.1b}
$$

Rewrite each of the three expressions for the operator $W_{\mathcal D}$ defined by Theorem 1.2 *solely in terms of the matrices $U, G, X, Y$ and corresponding operations*: multiplication, inversion, and transposition *written for this task only in the common form $x^\top, A^\top$ etc.* For example,

$$
(\check x \otimes y)(z)
= \check x(z)y
= \langle Lx, z\rangle y
= y(Lx)^\top z
= yx^\top Lz,
\qquad
\text{(recall: } L = L^\top\text{)}
\tag{0.2}
$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 2</span></p>

Consider the scalar second-order ODE

$$
\ddot z(t) + \frac{1}{4}\dot z(t) + z(t) = 0,
\qquad
z(0) = 1,
\qquad
\dot z(0) = 0.
\tag{0.3}
$$

**(a)** Set

$$
x(t) =
\begin{pmatrix}
z(t) \\
\dot z(t)
\end{pmatrix}
\tag{0.4}
$$

and rewrite the ODE in the form

$$
\dot x(t) = Ax(t),
\qquad
x(0) = x_0.
\tag{0.5}
$$

**(b)** Use the expression

$$
\exp_m(tA)
= e^{-\frac{t}{8}}
\begin{pmatrix}
\cos(ct) + \frac{1}{8c}\sin(ct) & \frac{1}{c}\sin(ct) \\
-\frac{1}{c}\sin(ct) & \cos(ct) - \frac{1}{8c}\sin(ct)
\end{pmatrix},
\qquad
c = \frac{3}{8}\sqrt 7
\tag{0.6}
$$

and display the solution $z(t) = x^1(t)$, $t \in [0, 20]$ based on the variation of constants formula.

**(c)** Now consider the ODE

$$
\dot x(t) = Ax(t) + W(t),
\qquad
W(t) =
\begin{pmatrix}
W^1(t) \\
W^2(t)
\end{pmatrix},
\qquad
x(0) = x_0.
\tag{0.7}
$$

Determine the minimal norm solution $t \mapsto W(t)$ as specified by Theorem 1.18 such that

$$
x(T) = x_T^*.
\tag{0.8}
$$

You are *not* expected to evaluate the integrals. Just summarize and simplify the steps that eventually determine the final expression of $W(t)$ which basically involves terms of the transition function (matrix exponential), its transpose and an integral over such terms. Assume for simplicity that all duality mappings are given by unit matrices. Does the structure of the resulting expression of $W(t)$ look familiar to you? See also the proof of Theorem 1.18 for an answer.

**(d)** Apply the 'variation of constants formula' to the ODE (0.7) and compute $x(T)$. What do you observe?

</div>
