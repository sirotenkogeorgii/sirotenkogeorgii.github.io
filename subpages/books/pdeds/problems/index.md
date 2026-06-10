---
title: Problems from the PDEs in Data Science course
layout: default
noindex: true
tags:
  - pde
  - optimization
  - convex-analysis
  - gradient-flows
  - machine-learning
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

# PDEs in Data Science — Course Problems

**Table of Contents**
- TOC
{:toc}

---

## Exercise Sheet 1 — Subdifferentials and convex functions

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intro</span><span class="math-callout__name">(Subdifferentials and convex functions)</span></p>

Throughout, $E : \mathbb{R}^N \to [0, \infty]$ is convex, i.e.

$$
E(\lambda x + (1-\lambda) y) \;\le\; \lambda E(x) + (1-\lambda) E(y) \qquad \forall x, y \in \mathbb{R}^N,\ \lambda \in [0,1].
\tag{1}
$$

The **subdifferential** of $E$ at $x$ is

$$
\partial E(x) \;:=\; \bigl\{\, p \in \mathbb{R}^N \;:\; E(y) \ge E(x) + \langle p,\, y - x\rangle \ \text{for all } y \in \mathbb{R}^N \,\bigr\}.
\tag{2}
$$

Geometrically, $p \in \partial E(x)$ iff the affine function $y \mapsto E(x) + \langle p, y - x\rangle$ minorises $E$ globally and is exact at $x$. Equivalently, the hyperplane in $\mathbb{R}^N \times \mathbb{R}$ with normal $(p, -1)$ supports the epigraph $\mathrm{epi}(E) = \lbrace (y, r) : r \ge E(y)\rbrace$ at $(x, E(x))$.

</div>

<details class="accordion" markdown="1">
<summary>Visualisation — supporting affine minorants</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet1_supporting_epigraph.png' | relative_url }}" alt="A convex function with its epigraph shaded; one supporting affine line at a smooth point and a fan of supporting lines at a kink" loading="lazy">
  <figcaption>Each $p \in \partial E(x_0)$ is the slope of a hyperplane supporting $\mathrm{epi}(E)$ at $(x_0, E(x_0))$. At a smooth point this hyperplane is unique (the tangent); at a kink there is a whole fan of them.</figcaption>
</figure>

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.1</span><span class="math-callout__name">(Subdifferential)</span></p>

Show the following statements.

**a)** The set $\partial E(x)$ is convex for all $x \in \mathbb{R}^N$. Furthermore, $\partial E(x)$ is non-empty if $E$ is continuous at $x$ and $E(x) < \infty$.

**b)** Let $E$ be lower semicontinuous and $(x_k)\_{k \in \mathbb{N}}, (p_k)\_{k \in \mathbb{N}} \subset \mathbb{R}^N$ such that $p_k \in \partial E(x_k)$. Additionally, let $x, p \in \mathbb{R}^N$ such that $x_k \to x$ and $p_k \to p$ as $k \to \infty$. Then $p \in \partial E(x)$.

**c)** Let $E \in C^1(\mathbb{R}^N)$. Then $\partial E(x) = \lbrace \nabla E(x)\rbrace$ for all $x \in \mathbb{R}^N$.

**d)** $E$ is differentiable at $x$ if and only if $\partial E(x)$ is a singleton, i.e. #$\partial E(x) = 1$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 1.1 (a) — convexity and non-emptiness of $\partial E(x)$</summary>

**Convexity.** Fix $x \in \mathbb{R}^N$ and take $p, q \in \partial E(x)$, $\lambda \in [0,1]$. By (2),

$$E(y) \ge E(x) + \langle p, y - x\rangle, \qquad E(y) \ge E(x) + \langle q, y - x\rangle, \qquad \forall y \in \mathbb{R}^N.$$

Multiplying the first inequality by $\lambda$, the second by $1 - \lambda$, and adding,

$$E(y) \;\ge\; E(x) + \langle \lambda p + (1-\lambda) q,\ y - x\rangle \qquad \forall y,$$

so $\lambda p + (1-\lambda) q \in \partial E(x)$. Hence $\partial E(x)$ is convex.

**Non-emptiness via supporting hyperplane.** Assume $E$ is continuous at $x$ and $E(x) < \infty$. The **epigraph**

$$\mathrm{epi}(E) \;=\; \{(y, r) \in \mathbb{R}^N \times \mathbb{R} : r \ge E(y)\}$$

is convex (because $E$ is convex) and the point $(x, E(x))$ lies on its boundary. Continuity of $E$ at $x$ ensures $\mathrm{epi}(E)$ has non-empty interior near $(x, E(x))$: for every $\varepsilon > 0$ there is $\delta > 0$ with $E(y) < E(x) + \varepsilon$ for $\|y - x\| < \delta$, so $(y, E(x) + \varepsilon)$ lies in the interior.

By the **supporting-hyperplane theorem** for convex sets with non-empty interior, there exists a non-zero pair $(a, b) \in \mathbb{R}^N \times \mathbb{R}$ such that

$$\langle a, y - x\rangle + b\,(r - E(x)) \;\le\; 0 \qquad \forall (y, r) \in \mathrm{epi}(E).$$

Taking $y = x$ and $r \to \infty$ forces $b \le 0$. If $b = 0$, the inequality reduces to $\langle a, y - x\rangle \le 0$ for every $y$ in a neighbourhood of $x$ (since $E$ is continuous, every nearby $y$ has $(y, E(x) + \varepsilon) \in \mathrm{epi}(E)$ for small $\varepsilon$); letting $y - x$ range over an open ball forces $a = 0$, contradicting $(a, b) \ne 0$. Hence $b < 0$.

Set $p := -a/b$. Plugging $r = E(y)$ into the hyperplane inequality gives

$$\langle a, y - x\rangle + b(E(y) - E(x)) \le 0 \;\;\Longleftrightarrow\;\; E(y) - E(x) \;\ge\; \langle p, y - x\rangle,$$

so $p \in \partial E(x)$. $\blacksquare$

</details>

<details class="accordion" markdown="1">
<summary>Solution 1.1 (b) — closedness of the subdifferential graph</summary>

For each $k$ and every $y \in \mathbb{R}^N$, $p_k \in \partial E(x_k)$ gives

$$E(y) \;\ge\; E(x_k) + \langle p_k,\, y - x_k\rangle. \tag{$\ast$}$$

Take $\liminf_{k\to\infty}$ on both sides. The LHS does not depend on $k$. For the RHS, $\langle p_k, y - x_k\rangle \to \langle p, y - x\rangle$ since both $p_k \to p$ and $x_k \to x$, and the inner product is continuous. Lower semicontinuity of $E$ gives

$$\liminf_{k\to\infty} E(x_k) \;\ge\; E(x).$$

Hence

$$E(y) \;\ge\; \liminf_{k\to\infty} \bigl[E(x_k) + \langle p_k, y - x_k\rangle\bigr] \;\ge\; E(x) + \langle p, y - x\rangle.$$

This holds for every $y$, so $p \in \partial E(x)$. $\blacksquare$

**Why l.s.c. is needed.** Without lower semicontinuity, $\liminf_k E(x_k)$ can be strictly less than $E(x)$, which would only give a *weaker* lower bound on $E(y)$ — not enough to conclude $p \in \partial E(x)$. The graph of $\partial E$ closes precisely because the RHS of $(\ast)$ converges from below to $E(x) + \langle p, y - x\rangle$ along every sequence with $x_k \to x$.

</details>

<details class="accordion" markdown="1">
<summary>Solution 1.1 (c) — $E \in C^1$ forces $\partial E(x) = \{\nabla E(x)\}$</summary>

**$\nabla E(x) \in \partial E(x)$.** Fix $x, y \in \mathbb{R}^N$ and $\lambda \in (0, 1]$. By convexity,

$$E\bigl((1-\lambda) x + \lambda y\bigr) \;\le\; (1-\lambda) E(x) + \lambda E(y),$$

i.e.

$$\frac{E(x + \lambda(y - x)) - E(x)}{\lambda} \;\le\; E(y) - E(x).$$

The LHS is a forward difference quotient of $E$ at $x$ in direction $y - x$. Since $E \in C^1$, sending $\lambda \to 0^+$ gives the directional derivative $dE(x).(y - x) = \langle \nabla E(x), y - x\rangle$. Hence

$$E(y) \;\ge\; E(x) + \langle \nabla E(x),\, y - x\rangle \qquad \forall y, $$

so $\nabla E(x) \in \partial E(x)$.

**No other subgradient.** Let $p \in \partial E(x)$. For any $v \in \mathbb{R}^N$ and $t > 0$, taking $y = x + t v$ in (2) yields

$$E(x + tv) - E(x) \;\ge\; t\,\langle p, v\rangle, \qquad \text{so}\qquad \frac{E(x + tv) - E(x)}{t} \;\ge\; \langle p, v\rangle.$$

Sending $t \to 0^+$ and using $C^1$,

$$\langle \nabla E(x), v\rangle \;\ge\; \langle p, v\rangle \qquad \forall v \in \mathbb{R}^N.$$

Replacing $v$ by $-v$ and combining the two inequalities gives $\langle \nabla E(x) - p, v\rangle = 0$ for all $v$, hence $p = \nabla E(x)$. $\blacksquare$

</details>

<details class="accordion" markdown="1">
<summary>Solution 1.1 (d) — differentiability $\Leftrightarrow$ singleton subdifferential</summary>

**($\Rightarrow$) Differentiability implies singleton.** If $E$ is differentiable at $x$, only Gateaux differentiability at $x$ was used in part (c) for both directions, so the same argument gives $\partial E(x) = \lbrace\nabla E(x)\rbrace$. (Convexity of $E$ is essential — the inclusion $\nabla E(x) \in \partial E(x)$ relies on the difference-quotient inequality coming from (1).)

**($\Leftarrow$) Singleton implies differentiability.** Assume $\partial E(x) = \{p\}$. The strategy: show that the right-derivative $E'(x; v) := \lim_{t \to 0^+} \tfrac{E(x + tv) - E(x)}{t}$ exists for every direction $v$, equals $\langle p, v\rangle$, and that this is enough for full differentiability when $E$ is finite-valued and convex on $\mathbb{R}^N$.

*Step 1 — directional derivatives exist.* For convex $E$, the difference quotient

$$\varphi_v(t) \;:=\; \frac{E(x + tv) - E(x)}{t}, \qquad t > 0,$$

is **non-decreasing** in $t$: this follows from (1) applied at $x + sv = \tfrac{s}{t}(x + tv) + (1 - \tfrac{s}{t}) x$ for $0 < s < t$. Together with the lower bound $\varphi_v(t) \ge \langle p, v\rangle$ (since $p \in \partial E(x)$), the limit

$$E'(x; v) \;=\; \inf_{t > 0} \varphi_v(t) \;\in\; [\langle p, v\rangle,\, \infty)$$

exists. The map $v \mapsto E'(x; v)$ is convex and positively homogeneous (sublinear), hence convex.

*Step 2 — sublinear support of the subdifferential.* A standard fact (Hahn–Banach / Rockafellar) is that

$$\partial E(x) \;=\; \bigl\{\, p \in \mathbb{R}^N \;:\; \langle p, v\rangle \le E'(x; v) \ \forall v \,\bigr\}, \qquad E'(x; v) \;=\; \sup_{q \in \partial E(x)} \langle q, v\rangle.$$

If $\partial E(x) = \{p\}$ then the supremum is attained at the unique element $p$, so

$$E'(x; v) \;=\; \langle p, v\rangle \qquad \forall v \in \mathbb{R}^N.$$

In particular $E'(x; -v) = \langle p, -v\rangle = -E'(x; v)$, so the *two-sided* directional derivative exists and equals $\langle p, v\rangle$ — this is **Gâteaux differentiability** with $\nabla E(x) = p$.

*Step 3 — Gâteaux $\Rightarrow$ Fréchet for finite convex $E$ on $\mathbb{R}^N$.* On finite-dimensional spaces, a convex function that is finite in a neighbourhood is locally Lipschitz (see Exercise 1.2(c) for the bonus proof). For locally Lipschitz convex functions, Gâteaux differentiability at $x$ implies Fréchet differentiability at $x$; this can be seen by combining a $1$-Lipschitz modulus with monotonicity of the difference quotients on a basis. Hence $E$ is differentiable at $x$ in the strong sense, with $\nabla E(x) = p$. $\blacksquare$

</details>

<details class="accordion" markdown="1">
<summary>Visualisation — kink vs. smooth point</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet1_subdiff_kink_vs_smooth.png' | relative_url }}" alt="Left: |x| with a fan of supporting lines through the origin. Right: x squared with its unique tangent line at a smooth point" loading="lazy">
  <figcaption>Left: $E(x)=|x|$ has a kink at $0$, so $\partial E(0)=[-1,1]$ — the whole fan of supporting lines. Right: $E(x)=x^2$ is $C^1$ everywhere, so $\partial E(x_0)=\{\nabla E(x_0)\}=\{2x_0\}$ — a unique tangent. This is the picture behind parts (c)–(d).</figcaption>
</figure>

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2</span><span class="math-callout__name">(Convex functions)</span></p>

Let $E : \mathbb{R}^N \to [0, \infty)$ be convex and finite-valued.

**a)** Let $x_i, p_i \in \mathbb{R}^N$ such that $p_i \in \partial E(x_i)$ for $i = 1, 2$. Show that

$$\langle p_1 - p_2,\, x_1 - x_2 \rangle \;\ge\; 0.$$

**b)** Show the equivalence of the following statements:

  i) $x_0 \in \mathbb{R}^N$ is a local minimum of $E$.

  ii) $x_0 \in \mathbb{R}^N$ is a global minimum of $E$.

  iii) $0 \in \partial E(x_0)$.

**c)** *(Bonus)* Show that $E$ is continuous.

</div>

<details class="accordion" markdown="1">
<summary>Solution 1.2 (a) — monotonicity of the subgradient</summary>

By definition of $\partial E$ applied at $x_1$ tested against $y = x_2$, and at $x_2$ tested against $y = x_1$:

$$E(x_2) \;\ge\; E(x_1) + \langle p_1, x_2 - x_1\rangle, \qquad E(x_1) \;\ge\; E(x_2) + \langle p_2, x_1 - x_2\rangle.$$

Adding these two inequalities,

$$0 \;\ge\; \langle p_1, x_2 - x_1\rangle + \langle p_2, x_1 - x_2\rangle \;=\; \langle p_2 - p_1, x_1 - x_2\rangle,$$

i.e. $\langle p_1 - p_2, x_1 - x_2\rangle \ge 0$. $\blacksquare$

**Reading.** This is the *monotone-operator* property of $\partial E$: as a multivalued map $x \mapsto \partial E(x)$, it is **monotone**. In 1-D it says that subgradients increase with $x$. It is the discrete-time, set-valued analogue of "$E'' \ge 0$" for smooth convex functions.

</details>

<details class="accordion" markdown="1">
<summary>Solution 1.2 (b) — local = global = $0 \in \partial E$</summary>

We prove the cycle (iii) $\Rightarrow$ (ii) $\Rightarrow$ (i) $\Rightarrow$ (iii).

**(iii) $\Rightarrow$ (ii).** If $0 \in \partial E(x_0)$, then by (2) with $p = 0$,

$$E(y) \;\ge\; E(x_0) + \langle 0, y - x_0\rangle \;=\; E(x_0) \qquad \forall y \in \mathbb{R}^N,$$

so $x_0$ is a global minimum.

**(ii) $\Rightarrow$ (i).** Trivial — global minima are local minima.

**(i) $\Rightarrow$ (iii).** Let $x_0$ be a local minimum, so there is $r > 0$ with $E(x_0) \le E(y)$ for all $y$ with $\|y - x_0\| < r$. Fix any $y \in \mathbb{R}^N$ and pick $\lambda \in (0, 1]$ small enough that $\|x_0 + \lambda(y - x_0) - x_0\| = \lambda \|y - x_0\| < r$. Convexity gives

$$E(x_0) \;\le\; E\bigl((1-\lambda) x_0 + \lambda y\bigr) \;\le\; (1-\lambda) E(x_0) + \lambda E(y),$$

where the first inequality uses local minimality at the perturbed point. Rearranging,

$$\lambda\,E(x_0) \;\le\; \lambda\,E(y) \;\;\Longrightarrow\;\; E(x_0) \le E(y).$$

This holds for every $y$, so $x_0$ is a global minimum, and trivially $E(y) \ge E(x_0) + \langle 0, y - x_0\rangle$, i.e. $0 \in \partial E(x_0)$. $\blacksquare$

**Why this is the right "first-order optimality condition" in the convex non-smooth setting.** For $C^1$ convex $E$, (iii) reduces to $\nabla E(x_0) = 0$ by Exercise 1.1(c). The subdifferential generalisation is exactly what is needed for non-smooth energies — e.g. $E(x) = \|x\|$ has its unique minimum at $0$ characterised by $0 \in \partial E(0) = [-1, 1]$, even though $E$ has no derivative there.

</details>

<details class="accordion" markdown="1">
<summary>Solution 1.2 (c) — bonus: $E$ is continuous (in fact locally Lipschitz)</summary>

We show that any *finite-valued* convex function $E : \mathbb{R}^N \to [0, \infty)$ is **locally Lipschitz**, hence continuous. The argument has three steps.

**Step 1 — local upper bound.** Fix $x_0 \in \mathbb{R}^N$ and $r > 0$, and let

$$Q := \prod_{i=1}^N [x_0^i - r,\, x_0^i + r] \;\supset\; \overline{B(x_0, r)}.$$

Let $V = \{v_1, \dots, v_{2^N}\}$ be the $2^N$ vertices of $Q$ and set

$$M := \max_{1 \le j \le 2^N} E(v_j) \;<\; \infty.$$

Every $y \in Q$ is a convex combination $y = \sum_j \lambda_j v_j$ with $\lambda_j \ge 0$, $\sum_j \lambda_j = 1$. Iterated convexity (or a $2^N$-term Jensen-style induction from (1)) gives

$$E(y) \;\le\; \sum_j \lambda_j E(v_j) \;\le\; M.$$

Hence $E$ is bounded above by $M$ on $\overline{B(x_0, r)}$.

**Step 2 — local lower bound.** Take $y \in \overline{B(x_0, r)}$ and let $y' := 2 x_0 - y$ be its reflection through $x_0$. Then $\|y' - x_0\| = \|y - x_0\| \le r$, so $y' \in \overline{B(x_0, r)}$ as well. Since $x_0 = \tfrac{1}{2}(y + y')$, convexity gives

$$E(x_0) \;\le\; \tfrac{1}{2} E(y) + \tfrac{1}{2} E(y'),$$

so

$$E(y) \;\ge\; 2 E(x_0) - E(y') \;\ge\; 2 E(x_0) - M \;=:\; m.$$

Hence $E$ is bounded on $\overline{B(x_0, r)}$: $m \le E \le M$ there.

**Step 3 — Lipschitz on the half-ball.** Take $x, y \in B(x_0, r/2)$ with $x \ne y$, and let $\delta := \|y - x\|$. Define

$$z \;:=\; y + \frac{r/2}{\delta}\,(y - x).$$

Then $\|z - x_0\| \le \|y - x_0\| + r/2 \le r$, so $z \in \overline{B(x_0, r)}$ and the bound $E(z) \le M$ from Step 1 applies. By construction $y$ lies on the segment from $x$ to $z$:

$$y \;=\; \frac{r/2}{r/2 + \delta}\, x \;+\; \frac{\delta}{r/2 + \delta}\, z.$$

(One checks $\frac{r/2}{r/2 + \delta} x + \frac{\delta}{r/2 + \delta} z = y$ by direct substitution.)

Convexity of $E$ then yields

$$E(y) \;\le\; \frac{r/2}{r/2 + \delta} E(x) + \frac{\delta}{r/2 + \delta} E(z) \;\le\; \frac{r/2}{r/2 + \delta} E(x) + \frac{\delta}{r/2 + \delta} M,$$

so

$$E(y) - E(x) \;\le\; \frac{\delta}{r/2 + \delta}\,(M - E(x)) \;\le\; \frac{\delta}{r/2}\,(M - m) \;=\; L\,|y - x|, \qquad L := \tfrac{2(M - m)}{r}.$$

By symmetry in $x, y$, $\|E(y) - E(x)\| \le L\,\|y - x\|$ on $B(x_0, r/2)$. Hence $E$ is **locally Lipschitz** on $\mathbb{R}^N$, and therefore continuous. $\blacksquare$

**Why finite-valuedness matters.** The argument breaks once $E$ is allowed to take the value $+\infty$ — Step 1 fails because $M$ may be infinite. This is why the theorem assumes $E : \mathbb{R}^N \to [0, \infty)$. For *extended-valued* convex functions, continuity is only automatic in the *interior* of the effective domain $\{E < \infty\}$.

**Connection to Exercise 1.1(a) and 1.1(d).** Local Lipschitzness is exactly what makes the subdifferential machinery work cleanly: the supporting-hyperplane construction in 1.1(a) needs an interior of $\mathrm{epi}(E)$, the Gâteaux-to-Fréchet upgrade in 1.1(d) needs a Lipschitz modulus, and the closedness of the graph of $\partial E$ in 1.1(b) gains its full force on l.s.c. convex functions. Continuity of $E$ on $\mathrm{int}\,\mathrm{dom}(E)$ is the structural fact that quietly underwrites all of these.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation — subgradient monotonicity</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet1_subgrad_monotonicity.png' | relative_url }}" alt="Left: a convex function with two supporting lines whose slopes increase with x. Right: scatter of (x_i - x_j, p_i - p_j) pairs all lying in the upper-right or lower-left quadrants" loading="lazy">
  <figcaption>Left: at $x_1 < x_2$, the slope $p_1$ of the supporting line is below $p_2$ — slopes are monotone in $x$. Right: every pair $(x_i - x_j,\ p_i - p_j)$ lands in the closed half-plane $\langle p_1 - p_2, x_1 - x_2\rangle \ge 0$. The shaded "forbidden" quadrants are exactly where (a) would fail.</figcaption>
</figure>

</details>

<details class="accordion" markdown="1">
<summary>Visualisation — local = global only for convex</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet1_local_eq_global.png' | relative_url }}" alt="Left: a convex parabola whose unique critical point is global. Right: a non-convex double-well with a local maximum at zero between two global minima" loading="lazy">
  <figcaption>Left: any local minimum of a convex $E$ is global, and the horizontal supporting line at $x_0$ encodes $0\in\partial E(x_0)$. Right: without convexity, critical points of $E$ can be local-only (or even saddles/maxima), and $0\in\nabla E$ is no longer sufficient for being a minimiser.</figcaption>
</figure>

</details>

## Exercise Sheet 2 — Subdifferentials of common energies, Łojasiewicz, convergence rates

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intro</span><span class="math-callout__name">(What this sheet is about)</span></p>

Sheet 1 set up the subdifferential abstractly and showed that it captures *first-order optimality* for non-smooth convex energies. Sheet 2 makes this concrete: we **compute** $\partial E$ for the four most-recurring non-smooth energies (the absolute value, the $\ell^1$- and $\ell^2$-norms, and a max-of-affine), and then we look at *long-term asymptotics* of the gradient flow under two different regimes — the **Łojasiewicz inequality** (gives finite length and convergence to a single limit point with no convexity) and **plain convexity** (gives a quantitative $1/t$ rate via Theorem 5).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.1</span><span class="math-callout__name">(Subdifferential of common non-smooth energies)</span></p>

**a)** $E(x) := \lvert x\rvert$ for $x \in \mathbb R$. Compute $\partial E(x)$ for all $x \in \mathbb R$.

**b)** $E(x) := \|x\|_1$ for $x \in \mathbb R^N$. Compute $\partial E(x)$ for all $x \in \mathbb R^N$.

**c)** $E(x) := \|x\|_2$ for $x \in \mathbb R^N$. Compute $\partial E(x)$ for all $x \in \mathbb R^N$.

**d)** $E(x) := \max_{i = 1, \dots, m}(a_i \cdot x + b_i)$ for $x \in \mathbb R^N$, where $(a_i)_{i=1}^m \subset \mathbb R^N$ and $(b_i)_{i=1}^m \subset \mathbb R$. Compute $\partial E(x)$ for all $x \in \mathbb R^N$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 2.1 (a) — $\partial \lvert x\rvert$ on $\mathbb R$</summary>

We claim

$$
\partial E(x) \;=\; \begin{cases} \{+1\}, & x > 0, \\ [-1, 1], & x = 0, \\ \{-1\}, & x < 0. \end{cases}
$$

**Off the kink.** For $x \ne 0$, $E$ is $C^1$ in a neighbourhood of $x$ with $E'(x) = \mathrm{sign}(x)$. Exercise 1.1(c) gives $\partial E(x) = \{E'(x)\} = \{\mathrm{sign}(x)\}$.

**At the kink $x = 0$.** By definition (2),

$$p \in \partial E(0) \;\Longleftrightarrow\; |y| \;\ge\; p\,y \quad \forall y \in \mathbb R.$$

Test $y > 0$: $y \ge p y$ forces $p \le 1$. Test $y < 0$: $-y \ge p y$, dividing by $-y > 0$ gives $1 \ge -p$, i.e. $p \ge -1$. Conversely, for any $p \in [-1, 1]$, $\lvert y\rvert \ge p\,y$ holds because $\lvert p\,y\rvert = \lvert p\rvert\,\lvert y\rvert \le \lvert y\rvert$. Hence $\partial E(0) = [-1, 1]$. $\blacksquare$

**Reading.** The kink is the *only* place where $\partial E$ is multivalued, and its width $\lvert\partial E(0)\rvert = 2$ matches the size of the slope jump $E'(0^+) - E'(0^-) = 2$ — this is the general picture for piecewise-$C^1$ convex functions.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation 2.1 (a) — fan of supports at $0$, single tangent elsewhere</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet2_subdiff_abs.png' | relative_url }}" alt="Left: graph of |x| with a fan of supporting lines through the origin and unique tangents at smooth points. Right: the multivalued map x → ∂E(x) drawn as a graph in the (x, p) plane, with a vertical segment [-1, 1] over x=0" loading="lazy">
  <figcaption>Left: every $p \in [-1, 1]$ is the slope of a line passing through $(0, 0)$ that lies *under* $\lvert x\rvert$ — the whole orange fan is $\partial E(0)$. At smooth points $\pm 1.2$ there is a unique tangent (slopes $\pm 1$). Right: the *graph* of the multivalued map $x \mapsto \partial E(x)$ — a step function with a vertical jump of length $2$ glued in at $x = 0$.</figcaption>
</figure>

</details>

<details class="accordion" markdown="1">
<summary>Solution 2.1 (b) — $\partial \|x\|_1$ on $\mathbb R^N$</summary>

The $\ell^1$-norm is **separable**: $\|x\|_1 = \sum_{i=1}^N \lvert x_i\rvert$, with each summand depending only on one coordinate. We will exploit this to reduce to (a).

**Claim.**

$$
\partial E(x) \;=\; \bigl\{\, p \in \mathbb R^N \;:\; p_i = \mathrm{sign}(x_i)\ \text{if } x_i \ne 0,\ \ p_i \in [-1, 1]\ \text{if } x_i = 0 \,\bigr\}.
$$

Equivalently, $\partial E(x) = \prod_{i=1}^N \partial \lvert\,\cdot\,\rvert(x_i)$ — the **Cartesian product** of the 1D subdifferentials from (a).

**($\supseteq$).** Take $p$ in the displayed set. For any $y \in \mathbb R^N$, applying the 1D inequality $\lvert y_i\rvert \ge \lvert x_i\rvert + p_i (y_i - x_i)$ coordinatewise (which holds because $p_i \in \partial \lvert\,\cdot\,\rvert(x_i)$ by (a)) and summing,

$$\|y\|_1 \;=\; \sum_i |y_i| \;\ge\; \sum_i \bigl[|x_i| + p_i (y_i - x_i)\bigr] \;=\; \|x\|_1 + \langle p, y - x\rangle.$$

So $p \in \partial E(x)$.

**($\subseteq$).** Take $p \in \partial E(x)$. Fix an index $j$ and a perturbation $h \in \mathbb R$, and consider $y = x + h\,e_j$ (only the $j$-th coordinate moves). Then $\lvert y_i\rvert = \lvert x_i\rvert$ for $i \ne j$, so the subgradient inequality reduces to

$$|x_j + h| \;\ge\; |x_j| + p_j h \qquad \forall h \in \mathbb R.$$

This says $p_j \in \partial \lvert\,\cdot\,\rvert(x_j)$, which by (a) is exactly the coordinatewise condition. $\blacksquare$

**Geometric reading.** $\partial \|\cdot\|_1(x)$ is the face of the dual unit cube $[-1, 1]^N$ matched to $\mathrm{sign}(x)$:

* if $x$ has all coordinates non-zero, $\partial E(x)$ is a **vertex** (a single $\pm 1$ pattern),
* if $k$ coordinates of $x$ are zero, $\partial E(x)$ is a **$k$-dimensional face** of $[-1, 1]^N$,
* at $x = 0$, $\partial E(0) = [-1, 1]^N$ — the **whole cube**.

This is exactly the *dual ball* picture: $\partial \|\cdot\|_p(0) = \{p : \|p\|_{p'} \le 1\}$, and the faces of the dual ball are visited along the way to a vertex.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation 2.1 (b) — corner / edge / facet of the dual cube</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet2_subdiff_l1.png' | relative_url }}" alt="Three panels showing subdifferential of the L1 norm at three points: (0,0) gives the whole square [-1,1]^2, (1,0) gives the right edge {1}×[-1,1], (0.8, 1.2) gives the corner (1,1)" loading="lazy">
  <figcaption>The three sparsity regimes. Left: $x = 0$ — both coordinates are slack, $\partial E(0) = [-1,1]^2$ (the whole dual cube). Middle: $x_1 \ne 0$, $x_2 = 0$ — the first coordinate is fixed to $\mathrm{sign}(x_1)$ but the second is free, giving an edge $\{1\} \times [-1, 1]$. Right: both coordinates are non-zero — both signs are determined and $\partial E$ collapses to a vertex. The picture extends literally to $N$ dimensions.</figcaption>
</figure>

</details>

<details class="accordion" markdown="1">
<summary>Solution 2.1 (c) — $\partial \|x\|_2$ on $\mathbb R^N$</summary>

We claim

$$
\partial E(x) \;=\; \begin{cases} \{x / \|x\|_2\}, & x \ne 0, \\ \overline{B(0, 1)} = \{p : \|p\|_2 \le 1\}, & x = 0. \end{cases}
$$

**Off zero.** For $x \ne 0$, $E$ is $C^1$ in a neighbourhood of $x$ with $\nabla E(x) = x / \|x\|_2$ (chain rule on $\sqrt{\langle x, x\rangle}$). By Exercise 1.1(c), $\partial E(x) = \{x / \|x\|_2\}$.

**At zero.** $p \in \partial E(0)$ iff $\|y\|_2 \ge \langle p, y\rangle$ for all $y \in \mathbb R^N$.

*($\Leftarrow$)* If $\|p\|_2 \le 1$, Cauchy–Schwarz gives $\langle p, y\rangle \le \|p\|_2 \|y\|_2 \le \|y\|_2$ for every $y$. So $p \in \partial E(0)$.

*($\Rightarrow$)* If $\|p\|_2 > 1$, choose $y = p$; then $\langle p, y\rangle = \|p\|_2^2 > \|p\|_2 = \|y\|_2$, contradicting the subgradient inequality. So $\|p\|_2 \le 1$. $\blacksquare$

**Reading.** The pattern $\partial E(0) = \{p : \|p\|_{p'} \le 1\}$ is general for the dual norm $p'$ of $p$. For $\|\cdot\|_2$, the dual is again $\|\cdot\|_2$, so $\partial E(0)$ is the *closed unit Euclidean ball*. Compare with (b): for $\|\cdot\|_1$ the dual is $\|\cdot\|_\infty$, so $\partial E(0) = [-1, 1]^N$. The shape of the subdifferential at the origin literally **is** the unit ball of the dual norm.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation 2.1 (c) — disk at the origin, unit-vector field elsewhere</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet2_subdiff_l2.png' | relative_url }}" alt="Left: closed unit disk shaded as the subdifferential at zero, with three colored points showing x/||x|| for three sample base points outside the disk. Right: contour plot of ||x||_2 with the unit disk shaded and a green arrow showing the gradient direction at a smooth point" loading="lazy">
  <figcaption>Left: $\partial \|\cdot\|_2(0)$ is the closed unit disk; for any $x \ne 0$, the subdifferential collapses to the single radial unit vector $x/\|x\|_2$, which lives on the *boundary* of that disk. Right: in $x$-space, the level sets of $\|x\|_2$ are concentric circles and the gradient field $x/\|x\|_2$ points radially outwards everywhere except at the origin, where it is replaced by the whole disk of admissible "directions of increase".</figcaption>
</figure>

</details>

<details class="accordion" markdown="1">
<summary>Solution 2.1 (d) — $\partial E$ for $E(x) = \max_i (a_i \cdot x + b_i)$</summary>

Define the **active set** at $x$,

$$I(x) \;:=\; \bigl\{\, i \in \{1, \dots, m\} \;:\; a_i \cdot x + b_i = E(x) \,\bigr\}.$$

We prove

$$
\partial E(x) \;=\; \mathrm{conv}\bigl\{\, a_i \;:\; i \in I(x) \,\bigr\}, \tag{$\ast$}
$$

the **convex hull of the gradients of the affine pieces that are active at $x$**.

**($\supseteq$).** Let $p = \sum_{i \in I(x)} \lambda_i a_i$ be a convex combination ($\lambda_i \ge 0$, $\sum \lambda_i = 1$). For any $y \in \mathbb R^N$ and any $i$, $E(y) \ge a_i \cdot y + b_i$. Multiply by $\lambda_i \ge 0$ and sum over $i \in I(x)$:

$$E(y) \;\ge\; \sum_{i \in I(x)} \lambda_i (a_i \cdot y + b_i) \;=\; \langle p, y\rangle + \sum_{i \in I(x)} \lambda_i b_i.$$

For $i \in I(x)$, $a_i \cdot x + b_i = E(x)$, so $b_i = E(x) - a_i \cdot x$ and

$$\sum_{i \in I(x)} \lambda_i b_i \;=\; E(x) - \langle p, x\rangle.$$

Substituting,

$$E(y) \;\ge\; E(x) + \langle p, y - x\rangle \qquad \forall y,$$

so $p \in \partial E(x)$.

**($\subseteq$).** Suppose for contradiction that some $p \in \partial E(x)$ does *not* lie in the closed convex set $C := \mathrm{conv}\{a_i : i \in I(x)\}$. By the **strict separating-hyperplane theorem** (applicable because $C$ is the convex hull of finitely many points, hence compact), there exists a unit vector $v \in \mathbb R^N$ with

$$\langle p, v\rangle \;>\; \max_{i \in I(x)} \langle a_i, v\rangle. \tag{†}$$

Move infinitesimally along $v$: set $y = x + t v$ for small $t > 0$. For $i \notin I(x)$, $a_i \cdot x + b_i < E(x)$ strictly; by continuity, for $t$ small enough $a_i \cdot y + b_i < E(x) + t \cdot \max_i \langle a_i, v\rangle$ remains *inactive* relative to the active pieces. Hence

$$E(y) \;=\; \max_{i \in I(x)} (a_i \cdot y + b_i) \;=\; E(x) + t \,\max_{i \in I(x)} \langle a_i, v\rangle.$$

The subgradient inequality at $y$ gives $E(y) \ge E(x) + t \langle p, v\rangle$, so dividing by $t > 0$,

$$\max_{i \in I(x)} \langle a_i, v\rangle \;\ge\; \langle p, v\rangle,$$

contradicting (†). Hence $p \in C$. $\blacksquare$

**Reading and consistency check.**

* Off the active manifold (i.e. when $\lvert I(x)\rvert = 1$), $\partial E(x) = \{a_{i^\ast}\}$ is a singleton — exactly what Exercise 1.1(c) predicts, since $E$ is locally affine and hence $C^1$ near $x$.
* On the kink set (where $\lvert I(x)\rvert \ge 2$), $\partial E(x)$ is a **non-trivial polytope** — its dimension is at most $\lvert I(x)\rvert - 1$.
* The formula recovers (a) as a special case: $\lvert x\rvert = \max(x, -x)$ with $a_1 = 1$, $a_2 = -1$, so at $x = 0$ both pieces are active and $\partial E(0) = \mathrm{conv}\{1, -1\} = [-1, 1]$. ✓
* It also recovers the classical chain rule: if $E(x) = f(x) := \max_i \ell_i(x)$ for affine $\ell_i$, the subdifferential is the convex hull of the gradients of the active $\ell_i$ — the **Danskin-type rule** in its cleanest form.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation 2.1 (d) — active gradients filling out a polytope at each kink</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet2_max_affine.png' | relative_url }}" alt="Left: three affine pieces and the upper envelope drawn as a thick blue piecewise-linear curve, with two red dots marking the kinks. Right: the multivalued slope map x → ∂E(x), constant on each linear piece and jumping by a vertical segment at each kink" loading="lazy">
  <figcaption>Three affine pieces $a_i x + b_i$ and their upper envelope $E$ (thick blue). The kinks (red dots) are exactly the points where the active set $I(x)$ jumps in size. Right panel: the subgradient $p \in \partial E(x)$ is constant equal to $a_{i^\ast}$ on each smooth segment, and *fills the interval* $[\min a_i, \max a_i]$ over each kink (red verticals) — the 1D shadow of the convex-hull rule $(\ast)$.</figcaption>
</figure>

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.2</span><span class="math-callout__name">(Long-term asymptotics via the Łojasiewicz inequality)</span></p>

Let $E : \mathbb R^N \to [0, \infty)$ be continuously differentiable and assume the **Łojasiewicz inequality**: for every $x \in \mathbb R^N$ there exists a neighbourhood $U$ of $x$, a constant $C > 0$, and an exponent $\theta \in (0, 1)$ such that

$$
\bigl|E(x) - E(y)\bigr|^\theta \;\le\; C\,\lvert\nabla E(x)\rvert \qquad \forall y \in U. \tag{Ł}
$$

Let $x : [0, \infty) \to \mathbb R^N$ be the unique solution of the gradient flow $\dot x = -\nabla E(x)$, and let $x^\ast$ be a local minimum of $E$. Show: **if $x^\ast$ is a limit point of $x$, then the curve $t \mapsto x(t)$ has finite length and $x(t) \to x^\ast$ as $t \to \infty$.**

</div>

<details class="accordion" markdown="1">
<summary>Solution 2.2 — desingularizing $E^{1-\theta}$ and the trapping argument</summary>

The proof has three steps: a **differential identity** for $E^{1-\theta}$ along the flow, a **trapping** argument that keeps $x(t)$ in the Łojasiewicz neighbourhood once it gets close enough, and a **Cauchy** conclusion.

**Setup.** Let $E_\infty := E(x^\ast)$. Since $x^\ast$ is a local minimum and $E \in C^1$, $\nabla E(x^\ast) = 0$. The energy $t \mapsto E(x(t))$ is non-increasing (by $\dot E = -\lvert\nabla E\rvert^2 \le 0$), bounded below by $0$, hence convergent to some $E_\flat \ge 0$. Because $x^\ast$ is a *limit point*, there exists a sequence $t_n \to \infty$ with $x(t_n) \to x^\ast$, so by continuity $E_\flat = E_\infty$. Define the **excess energy**

$$\mathcal E(t) \;:=\; E(x(t)) - E_\infty \;\ge\; 0,$$

which is non-increasing with $\mathcal E(t) \to 0$.

**Step 1 — differentiate $\mathcal E^{1-\theta}$.** On the set where $\mathcal E(t) > 0$,

$$\frac{d}{dt} \mathcal E^{1-\theta}(t) \;=\; (1 - \theta)\,\mathcal E^{-\theta}(t)\,\dot{\mathcal E}(t) \;=\; -(1 - \theta)\,\mathcal E^{-\theta}(t)\,|\nabla E(x(t))|^2.$$

Along the flow $\lvert\dot x\rvert = \lvert\nabla E(x)\rvert$, so $\lvert\nabla E\rvert^2 = \lvert\nabla E\rvert\,\lvert\dot x\rvert$, giving

$$\frac{d}{dt} \mathcal E^{1-\theta}(t) \;=\; -(1 - \theta)\,\mathcal E^{-\theta}(t)\,|\nabla E(x(t))|\,|\dot x(t)|. \tag{$\diamond$}$$

This is an *exact* identity, valid wherever $\mathcal E > 0$.

**Step 2 — apply Łojasiewicz inside a neighbourhood.** Apply (Ł) at the point $x^\ast$: there exists a neighbourhood $U$ of $x^\ast$, $C, \theta$ as in the hypothesis, such that for $y \in U$,

$$|E(x^\ast) - E(y)|^\theta \;\le\; C\,|\nabla E(x^\ast)| = 0.$$

This would force $E$ to be constant on $U$, which is the *degenerate* case (then $\nabla E \equiv 0$ on $U$, the gradient flow is stationary once it enters $U$, and the conclusion is trivial). The **non-degenerate case** uses the standard form: Łojasiewicz at $x^\ast$ implies

$$|E(y) - E_\infty|^\theta \;\le\; C\,|\nabla E(y)| \qquad \forall y \in U. \tag{Ł$_\ast$}$$

(This is the symmetric version: applying the hypothesis at $y$ near $x^\ast$ and using continuity of $\nabla E$ produces an inequality of this form on a possibly smaller neighbourhood with a larger constant; in the variant of (Ł) typically stated in the lecture notes, this is precisely (Ł'). We use it as the standing hypothesis.)

For $t$ such that $x(t) \in U$ and $\mathcal E(t) > 0$, (Ł$_\ast$) gives $\mathcal E(t)^\theta \le C\,\lvert\nabla E(x(t))\rvert$, equivalently

$$\mathcal E^{-\theta}(t)\,|\nabla E(x(t))| \;\ge\; \frac{1}{C}.$$

Substituting into $(\diamond)$,

$$-\frac{d}{dt}\mathcal E^{1-\theta}(t) \;\ge\; \frac{1 - \theta}{C}\,|\dot x(t)|. \tag{$\heartsuit$}$$

This is the key differential inequality: **the rate of decrease of $\mathcal E^{1-\theta}$ is bounded below by a multiple of the speed**. Integrating it converts a bound on the *energy budget* into a bound on the *length of the trajectory*.

**Step 3 — trapping.** Fix $\delta > 0$ small enough that $B(x^\ast, 2\delta) \subset U$. Choose $n$ large enough that simultaneously

$$x(t_n) \in B(x^\ast, \delta),\qquad \frac{C}{1 - \theta}\,\mathcal E(t_n)^{1 - \theta} \;<\; \delta. \tag{$\sharp$}$$

This is possible: $x(t_n) \to x^\ast$ takes care of the first; $\mathcal E(t_n) \to 0$ takes care of the second (use $\mathcal E(t)^{1-\theta} \to 0$).

**Claim.** $x(t) \in B(x^\ast, 2\delta)$ for all $t \ge t_n$.

If not, there is a *first* time $t^\sharp > t_n$ with $\lvert x(t^\sharp) - x^\ast\rvert = 2\delta$. On the interval $[t_n, t^\sharp]$, $x(t) \in B(x^\ast, 2\delta) \subset U$, so $(\heartsuit)$ holds throughout. Integrating from $t_n$ to $t^\sharp$,

$$\int_{t_n}^{t^\sharp} |\dot x(\tau)|\,d\tau \;\le\; \frac{C}{1 - \theta}\bigl[\mathcal E^{1 - \theta}(t_n) - \mathcal E^{1 - \theta}(t^\sharp)\bigr] \;\le\; \frac{C}{1 - \theta}\,\mathcal E(t_n)^{1 - \theta} \;<\; \delta.$$

But by the triangle inequality,

$$|x(t^\sharp) - x(t_n)| \;\le\; \int_{t_n}^{t^\sharp} |\dot x|\,d\tau \;<\; \delta,$$

so $\lvert x(t^\sharp) - x^\ast\rvert \le \lvert x(t^\sharp) - x(t_n)\rvert + \lvert x(t_n) - x^\ast\rvert < \delta + \delta = 2\delta$ — a contradiction.

Hence $x(t) \in U$ for all $t \ge t_n$, and the integration above extends to $t \to \infty$:

$$\int_{t_n}^\infty |\dot x(\tau)|\,d\tau \;\le\; \frac{C}{1 - \theta}\,\mathcal E(t_n)^{1 - \theta} \;<\; \infty. \tag{$\flat$}$$

The trajectory has **finite length**.

**Step 4 — convergence.** Finite length means $x(t)$ is Cauchy in $\mathbb R^N$: for $t > s > t_n$,

$$|x(t) - x(s)| \;\le\; \int_s^t |\dot x(\tau)|\,d\tau \;\le\; \int_s^\infty |\dot x(\tau)|\,d\tau \;\xrightarrow[s \to \infty]{} \;0.$$

So $x(t) \to x^\ast_\infty$ for some $x^\ast_\infty \in \mathbb R^N$. Since $x^\ast$ is a limit point along $t_n$, we have $x^\ast_\infty = x^\ast$ by uniqueness of limits in $\mathbb R^N$. $\blacksquare$

**What the desingularizing exponent does.** The trick is to differentiate not $\mathcal E$ itself but $\mathcal E^{1-\theta}$. The exponent $1 - \theta \in (0, 1)$ is *forced*: it is the unique power that, combined with (Ł), turns the energy-balance estimate into a *length* estimate (which has the right physical units of $\lvert\dot x\rvert \cdot dt$). Without this concave reparametrisation the proof would only conclude $\int_0^\infty \lvert\dot x\rvert^2\,dt < \infty$ — square-integrability — which by itself does **not** force convergence of $x(t)$. The Kurdyka–Łojasiewicz framework formalises this template; see [Appendix A](../../index.md#appendix-a) of the main notes for the full discussion.

</details>

<details class="accordion" markdown="1">
<summary>Visualisation 2.2 — selecting a single limit point on a continuum of minimizers</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet2_lojasiewicz_trap.png' | relative_url }}" alt="Left: a 2D landscape with a circular valley of minimizers; two gradient flow trajectories from different starting points each select a single point on the circle, with a dotted neighborhood drawn around one limit point. Right: a schematic showing the desingularizing function E^(1-θ) decreasing in time and the length integral increasing but staying bounded by the initial budget" loading="lazy">
  <figcaption>Left: the energy $E(x) = (\|x\|^2 - \tfrac12)^2$ has its minimum on the *whole circle* $\|x\| = 1/\sqrt 2$ — a continuum of critical points. Each gradient-flow trajectory still converges to a *single* point on the circle (green dots); Łojasiewicz selects which one based on the initial condition. The dotted purple ball around the right limit point is the trapping neighbourhood $U$ from Step 3. Right: schematic of $(\heartsuit)$. The blue curve $\mathcal E^{1-\theta}$ (non-increasing) bounds the red length integral; the red shaded "remaining budget" never exceeds the dashed initial ceiling $\mathcal E(0)^{1-\theta}$. This is the picture of "energy budget $\Rightarrow$ length budget".</figcaption>
</figure>

</details>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.3</span><span class="math-callout__name">(Convergence rates: convex vs. uniformly convex)</span></p>

Find a convex (but not uniformly convex) energy $E : \mathbb R \to [0, \infty)$ such that the convergence to equilibrium of the gradient flow is only **algebraic** and not exponential. Is the bound from Theorem 5 sharp?

</div>

<details class="accordion" markdown="1">
<summary>Solution 2.3 — $E(x) = \tfrac{1}{4} x^4$ and a sharpness discussion</summary>

**The example.** Take

$$E(x) \;=\; \tfrac{1}{4} x^4, \qquad x \in \mathbb R.$$

* **Convex.** $E''(x) = 3 x^2 \ge 0$ for all $x$.
* **Not uniformly convex.** $E''(0) = 0$, so there is no constant $\lambda > 0$ with $E''(x) \ge \lambda$ — the Hessian *degenerates* at the unique minimiser $x^\ast = 0$.

**Solve the gradient flow.** $\dot x = -E'(x) = -x^3$. Separating variables (assume $x_0 > 0$, the case $x_0 < 0$ is symmetric, $x_0 = 0$ is stationary):

$$\frac{dx}{x^3} \;=\; -dt \;\;\Longrightarrow\;\; -\frac{1}{2 x^2} \;=\; -t + C.$$

The constant $C$ is fixed by $x(0) = x_0$: $C = -1/(2 x_0^2)$. So

$$\boxed{\; x(t) \;=\; \frac{x_0}{\sqrt{1 + 2 x_0^2\,t}}.\; }$$

In particular $x(t) \to 0$ as $t \to \infty$, with the **algebraic** rate $\lvert x(t)\rvert = O(t^{-1/2})$.

**Energy decay.**

$$\mathcal E(t) \;=\; E(x(t)) \;=\; \tfrac{1}{4}\,x(t)^4 \;=\; \frac{x_0^4 / 4}{(1 + 2 x_0^2 t)^2} \;=\; O\!\bigl(t^{-2}\bigr),\quad t \to \infty,$$

with explicit asymptote $\mathcal E(t) \sim 1/(16\,t^2)$ for $x_0 \to \infty$ regime, or in general $\mathcal E(t) \sim x_0^4 / (16\,x_0^4\,t^2) = 1/(16 t^2)$ as $t \to \infty$.

This is **strictly slower than exponential** but strictly faster than the $1/t$ bound in Theorem 5.

**Compare with the uniformly convex case.** For $E(x) = \tfrac{1}{2} \lambda x^2$ ($\lambda > 0$, uniformly convex), the gradient flow is $\dot x = -\lambda x$, $x(t) = x_0 e^{-\lambda t}$, $\mathcal E(t) = \tfrac{1}{2}\lambda x_0^2 e^{-2\lambda t}$ — *exponential*. Removing uniform convexity loses one order of decay rate.

**Is Theorem 5 sharp?** Theorem 5 says: for convex $E \in C^2$ and $\mathcal H(0) := \lvert x_0 - x^\ast\rvert^2$,

$$\mathcal E(t) \;\le\; \frac{\mathcal H(0)}{t}.$$

For our example, $\mathcal H(0) = x_0^2$ and the actual $\mathcal E(t) \sim 1/(16 t^2)$ — much smaller than $x_0^2 / t$. So **the bound is not attained on $E = x^4/4$.**

Sharpness in the strong sense is more subtle — and the answer is *yes, the **exponent** is sharp* in the sense that no faster universal rate holds across all convex $E$. To see this, consider the family

$$E_p(x) \;:=\; \frac{|x|^p}{p}, \qquad p \in (2, \infty).$$

Each $E_p$ is convex (for $p \ge 1$), with $E_p''(0) = 0$ when $p > 2$, so none is uniformly convex. The gradient flow $\dot x = -\lvert x\rvert^{p-2} x$ solves (for $x_0 > 0$, $p > 2$)

$$x(t) \;=\; \bigl(x_0^{2-p} + (p - 2)\,t\bigr)^{-1/(p-2)},$$

so

$$\mathcal E_p(t) \;=\; \frac{\lvert x(t)\rvert^p}{p} \;\sim\; \frac{1}{p\,((p-2)\,t)^{p/(p-2)}} \;=\; O\!\bigl(t^{-p/(p-2)}\bigr) \quad \text{as } t \to \infty.$$

The exponent is $p/(p-2) = 1 + 2/(p-2) \to 1^+$ as $p \to \infty$. So:

* for **any fixed $p > 2$**, the actual decay $t^{-p/(p-2)}$ is *strictly faster* than the Theorem 5 bound $t^{-1}$;
* but for **any** $\varepsilon > 0$, choosing $p > 2 + 2/\varepsilon$ gives a convex energy whose decay is no faster than $t^{-(1 + \varepsilon)}$.

This means: **the exponent $-1$ in Theorem 5 cannot be improved to $-(1 + \varepsilon)$ uniformly over the class of convex energies** — although for each individual energy the rate is faster. The exponent is sharp in the *worst-case sense* but not in the *pointwise sense*. The constant $\mathcal H(0)$ matches the dimensional scaling and is also sharp up to a factor (this can be seen by tracking $x_0$-dependence in the prefactor of $\mathcal E_p(t)$).

**Why this is the "right" gap.** The proof of Theorem 5 hinges on the algebraic relation $\mathcal E \le (\mathcal H \mathcal D)^{1/2}$, which becomes an equality (up to a factor) in the *worst-case* configuration where the trajectory is barely descending. The family $E_p$ realises exactly this — as $p \to \infty$, $E_p$ becomes increasingly *flat* near the minimiser, $\lvert\nabla E_p\rvert$ near $0$ becomes negligible, and the algebraic bound becomes nearly tight. Adding *any* uniform convexity ($\lambda > 0$) restores the exponential rate of Theorem 3 and changes the picture qualitatively. $\blacksquare$

</details>

<details class="accordion" markdown="1">
<summary>Visualisation 2.3 — algebraic decay and the $\lvert x\rvert^p$ family approaching the $1/t$ bound</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet2_x4_decay.png' | relative_url }}" alt="Three panels. Left: trajectories of x(t) under dot x = -x^3 for several initial conditions, with an exponential decay reference dashed. Middle: log-log plot of E(t) for the x^4/4 case versus the Theorem 5 bound H(0)/t and an asymptotic 1/(16t^2) reference. Right: log-log decay curves for E = |x|^p / p with p = 3, 4, 6, 10, 50, all sitting strictly below the Theorem 5 bound 1/t but approaching it as p → ∞" loading="lazy">
  <figcaption>Left: the explicit trajectory $x(t) = x_0 / \sqrt{1 + 2 x_0^2 t}$ under the gradient flow of $\tfrac14 x^4$. The dashed grey curve $x_0 e^{-t}$ — what we'd get from a uniformly convex $\tfrac12 x^2$ — falls off vastly faster. Middle: actual $\mathcal E(t) \sim 1/(16 t^2)$ (solid) is well *below* the Theorem 5 bound $x_0^2 / t$ (dotted) — the bound is not sharp on this example. Right: the family $E_p = \lvert x\rvert^p / p$ shows the rate $t^{-p/(p-2)}$ approaching the bound $t^{-1}$ as $p \to \infty$ — so the *exponent* in Theorem 5 cannot be improved across the convex class.</figcaption>
</figure>

</details>

## Exercise Sheet 7 — Displacement convexity of power laws and of the Dirichlet energy; the gluing lemma

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intro</span><span class="math-callout__name">(What this sheet is about)</span></p>

This sheet sits squarely in the *Displacement convexity* section of Chapter 2. Exercise 7.1 takes the **Eulerian machinery** built there — the continuity equation (2.22) and the pressureless Euler equation (2.23), which together describe the displacement interpolation as a flow of densities — and runs it on two new functionals:

* the **power-law internal energy** $\mathcal U(\rho)=\int\rho^p$, $p>1$, where the entropy computation generalises and convexity *survives* (with an extra positive term coming from the pressure);
* the **Dirichlet energy** $E(\rho)=\frac12\int\lvert\nabla\rho\rvert^2$ — the simplest *first-order* functional, depending on $\nabla\rho$ rather than $\rho$ — where displacement convexity **fails**. We construct an explicit geodesic along which the energy is strictly concave near $t=0$.

The moral of 7.1: McCann's theory (Theorem 32) is a statement about *zeroth-order* functionals of the density. Geodesics in Wasserstein space happily create *shape distortion* — steepening a profile while stretching it elsewhere — which functionals of $\nabla\rho$ can see and zeroth-order functionals cannot.

Exercise 7.2 is about the **gluing lemma**: two transference plans $\pi\_{1,2}\in\Pi(\mu\_1,\mu\_2)$ and $\pi\_{2,3}\in\Pi(\mu\_2,\mu\_3)$ that share the middle marginal $\mu\_2$ can be glued into a single *triple coupling* $\pi\in\mathcal P(\mathbb R^d\times\mathbb R^d\times\mathbb R^d)$ — the key step in proving the triangle inequality for the Wasserstein distance. The exercise asks for the explicit formula in the two structured cases: plans with densities (the answer is a Bayes/Markov formula) and plans supported on graphs (the answer is composition of maps).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.1</span><span class="math-callout__name">(Displacement convexity in Eulerian coordinates)</span></p>

**a)** Show that

$$
\mathcal U(\rho) = \int_{\mathbb R^d} \rho^p\,dx \qquad\text{for } p>1
$$

is displacement convex. Use the Eulerian representation (2.22) and (2.23). (Do **not** use Theorem 32.)

**b)** Show that

$$
E(\rho) := \int_{\mathbb R^d} \frac12\,\lvert\nabla\rho\rvert^2\,dx
$$

is **not** displacement convex.

</div>

<details class="accordion" markdown="1">
<summary>Solution 7.1 (a) — power laws via (2.22)–(2.23)</summary>

**Set-up.** Let $\mu,\nu\in\mathcal P\_{ac}(\mathbb R^d)$ and let $\rho\_t:=[\mu,\nu]\_t=(T\_t)\_\sharp\mu$ be the displacement interpolation, $T\_t=(1-t)\,\mathrm{id}+t\nabla\varphi$ with $\nabla\varphi$ the Brenier map. By Theorem 33 and Proposition 34, the density–velocity pair $(\rho\_t,v\_t)$, where $v\_t=\frac{d}{dt}T\_t\circ T\_t^{-1}$, solves the Eulerian system

$$
\partial_t\rho_t + \nabla\cdot(\rho_t v_t) = 0, \tag{2.22}
$$

$$
\partial_t v_t + (v_t\cdot\nabla)v_t = 0. \tag{2.23}
$$

As in the entropy computation of the notes, we compute *formally*: we assume $\rho\_t$, $v\_t$ are smooth enough and decay fast enough that we may interchange $\frac{d}{dt}$ with the integral and integrate by parts without boundary terms. We drop the subscript $t$ inside integrals.

**The pressure.** For $U(s)=s^p$ define the **pressure**

$$
P(s) := s\,U'(s) - U(s) = p\,s^p - s^p = (p-1)\,s^p,
$$

chosen precisely so that $P'(s)=s\,U''(s)$ and hence

$$
\nabla\bigl[P(\rho)\bigr] = P'(\rho)\nabla\rho = \rho\,U''(\rho)\,\nabla\rho = \rho\,\nabla\bigl[U'(\rho)\bigr].
$$

This identity converts the awkward integrand $\rho\,\nabla U'(\rho)$ into a perfect gradient — it is the exact analogue of the "special algebra of the logarithm" ($\rho\,U''(\rho)=1$ for the entropy) that made the computation in the notes collapse.

**First derivative.** Using (2.22) and integrating by parts twice,

$$
\begin{aligned}
\frac{d}{dt}\,\mathcal U(\rho_t)
&= \int_{\mathbb R^d} U'(\rho)\,\partial_t\rho\,dx
\overset{(2.22)}{=} -\int_{\mathbb R^d} U'(\rho)\,\nabla\cdot(\rho v)\,dx
= \int_{\mathbb R^d} \nabla\bigl[U'(\rho)\bigr]\cdot v\,\rho\,dx\\[2pt]
&= \int_{\mathbb R^d} \nabla\bigl[P(\rho)\bigr]\cdot v\,dx
= -\int_{\mathbb R^d} P(\rho)\,\nabla\cdot v\,dx
= -(p-1)\int_{\mathbb R^d}\rho^p\,\nabla\cdot v\,dx.
\tag{7.1}
\end{aligned}
$$

**Second derivative.** Differentiating (7.1) in $t$ (product rule, then (2.22) on $\partial\_t\rho$ and (2.23) on $\partial\_t v$):

$$
\begin{aligned}
\frac{d^2}{dt^2}\,\mathcal U(\rho_t)
&= -(p-1)\Bigl[\int p\,\rho^{p-1}\,(\partial_t\rho)\,(\nabla\cdot v)\,dx + \int \rho^p\,\nabla\cdot(\partial_t v)\,dx\Bigr]\\
&\overset{(2.22),(2.23)}{=} (p-1)\Bigl[\underbrace{\,p\int \rho^{p-1}\,\nabla\cdot(\rho v)\,(\nabla\cdot v)\,dx\,}_{=:A} + \underbrace{\int \rho^p\,\nabla\cdot\bigl((v\cdot\nabla)v\bigr)\,dx}_{=:B}\Bigr].
\end{aligned}
$$

We work out $A$ and $B$ in indices (summation convention). For $A$, the observation $p\,\rho^{p-1}\partial\_j(\rho v\_j) = v\_j\,\partial\_j(\rho^p) + p\,\rho^p\,\partial\_j v\_j$ gives

$$
A = \int v_j\,\partial_j(\rho^p)\,(\partial_i v_i)\,dx + p\int \rho^p\,(\nabla\cdot v)^2\,dx.
$$

For $B$, expanding $\partial\_i(v\_j\partial\_j v\_i)=\partial\_i v\_j\,\partial\_j v\_i + v\_j\,\partial\_j(\partial\_i v\_i)$ and integrating the second piece by parts,

$$
B = \int \rho^p\,\operatorname{tr}\bigl((\nabla v)^2\bigr)\,dx
- \int v_j\,\partial_j(\rho^p)\,(\partial_i v_i)\,dx - \int \rho^p\,(\nabla\cdot v)^2\,dx.
$$

The mixed terms $\pm\int v\cdot\nabla(\rho^p)\,(\nabla\cdot v)$ cancel exactly, and we are left with

$$
\frac{d^2}{dt^2}\,\mathcal U(\rho_t)
= (p-1)^2\int_{\mathbb R^d} \rho^p\,(\nabla\cdot v)^2\,dx
\;+\; (p-1)\int_{\mathbb R^d} \rho^p\,\operatorname{tr}\bigl((\nabla v)^2\bigr)\,dx.
\tag{7.2}
$$

**Sign.** Both terms in (7.2) are non-negative:

* the first because $p>1$ and $(\nabla\cdot v)^2\ge 0$;
* the second because along the displacement interpolation $\nabla v\_t$ is a **symmetric** matrix, so $\operatorname{tr}((\nabla v)^2)=\lvert\nabla v\rvert^2\ge 0$ (squared Frobenius norm). Symmetry is the same argument as in the notes' entropy computation: $v\_t(y)=\frac1t\bigl(y-T\_t^{-1}(y)\bigr)$ and $T\_t^{-1}$ is the gradient of a convex function (it is the Brenier map from $\rho\_t$ back to $\mu$). Explicitly, $\nabla v\_t = (\nabla^2\varphi - I)\bigl[(1-t)I + t\nabla^2\varphi\bigr]^{-1}$ — two commuting symmetric matrices, hence a symmetric product.

So $t\mapsto\mathcal U(\rho\_t)$ is continuous on $[0,1]$ and has non-negative second derivative on $(0,1)$, hence is convex. Since $\mu,\nu\in\mathcal P\_{ac}(\mathbb R^d)$ were arbitrary, $\mathcal U$ is displacement convex. $\blacksquare$

**Reading 1 — the general pressure formula.** Nothing in the computation used the specific power law until the last step. For a general internal energy density $U$ with pressure $P(s)=sU'(s)-U(s)$, the same manipulations give

$$
\frac{d^2}{dt^2}\,\mathcal U(\rho_t)
= \int \bigl[\rho\,P'(\rho)-P(\rho)\bigr](\nabla\cdot v)^2\,dx + \int P(\rho)\,\operatorname{tr}\bigl((\nabla v)^2\bigr)\,dx.
$$

For the entropy, $P(s)=s$ and $\rho P'(\rho)-P(\rho)=0$: the divergence term *vanishes identically* and only $\int\rho\lvert\nabla v\rvert^2$ remains — exactly the formula in the notes. For $U(s)=s^p$, $\rho P'(\rho)-P(\rho)=(p-1)^2\rho^p$ and $P(\rho)=(p-1)\rho^p$ are both non-negative *precisely because* $p>1$. The entropy is the borderline case where the first term degenerates.

**Reading 2 — consistency checks.** (i) McCann's condition (2.19), which we were not allowed to use: $\Psi(r)=r^d U(r^{-d})=r^{d(1-p)}$ is convex and non-increasing for $p>1$ — agreement. (ii) In $d=1$ the Lagrangian computation is actually explicit and confirms (7.2): with $T\_t=\mathrm{id}+t v\_0$,

$$
\mathcal U(\rho_t) = \int \frac{\rho_0^p}{(1+tv_0')^p}\,(1+tv_0')\,dx = \int \rho_0^p\,(1+tv_0')^{1-p}\,dx,
$$

whose integrand has $t$-second-derivative $p(p-1)\,\rho\_0^p\,(v\_0')^2(1+tv\_0')^{-1-p}\ge 0$ pointwise; changing variables back, this is exactly $(7.2)$ in $d=1$, where $(\nabla\cdot v)^2=\operatorname{tr}((\nabla v)^2)=(v')^2$ and $(p-1)^2+(p-1)=p(p-1)$.

</details>

<details class="accordion" markdown="1">
<summary>Solution 7.1 (b) — the Dirichlet energy is not displacement convex</summary>

We extend $E$ to all of $\mathcal P\_{ac}(\mathbb R^d)$ by setting $E(\rho)=+\infty$ when $\rho\notin H^1$, so that displacement convexity in the sense of Definition 29 is a meaningful claim on the displacement convex set $A=\mathcal P\_{ac}(\mathbb R^d)$. To disprove it, it suffices to exhibit **one** geodesic $(\rho\_t)\_{t\in[0,1]}$ with all energies finite along which $g(t):=E(\rho\_t)$ is not convex. We do this in $d=1$ (see Reading 3 for $d\ge 2$).

**Why the obvious tests fail.** Translations $T=\mathrm{id}+c$ leave $E$ invariant: $g$ constant. Dilations $T=\lambda\,\mathrm{id}$ give, by scaling, $g(t)=\bigl(1-t+t\lambda\bigr)^{-(d+2)}E(\rho\_0)$ — a *convex* function of $t$. So the one-parameter families that detect McCann's condition (2.19) are blind here; the counterexample must genuinely distort the *shape* of the density. The mechanism will be: a density with an **exponentially growing window**, and a displacement concentrated inside the window.

**Step 1: a geodesic and an exact formula for $g$.** Work on $\mathbb R$. Let $\rho\_0$ be a smooth, compactly supported probability density and let $v\colon\mathbb R\to\mathbb R$ be Lipschitz with $v'\ge 0$. Then $T:=\mathrm{id}+v$ is strictly increasing, hence (being monotone, i.e. the derivative of the convex function $x\mapsto \frac{x^2}{2}+\int\_0^x v$) it is the **optimal** map from $\mu:=\rho\_0\,dx$ to $\nu:=T\_\sharp\mu$, and

$$
\rho_t := (T_t)_\sharp\mu, \qquad T_t = \mathrm{id} + t\,v, \quad t\in[0,1]
$$

is the displacement interpolation $[\mu,\nu]\_t$. The one-dimensional change-of-variables (Monge–Ampère) identity reads

$$
\rho_t\bigl(T_t(x)\bigr)\,\bigl(1+t\,v'(x)\bigr) = \rho_0(x).
$$

Differentiating in $x$ and dividing by $T\_t'=1+tv'$,

$$
\rho_t'\bigl(T_t(x)\bigr) = \frac{\rho_0'(x)\,(1+t v'(x)) - t\,\rho_0(x)\,v''(x)}{(1+t v'(x))^3},
$$

and substituting $y=T\_t(x)$, $dy=(1+tv')\,dx$ in $E(\rho\_t)=\frac12\int(\rho\_t')^2\,dy$:

$$
g(t) = E(\rho_t) = \frac12\int_{\mathbb R} \frac{\bigl[\rho_0'\,(1+t v') - t\,\rho_0\,v''\bigr]^2}{(1+t v')^5}\,dx.
\tag{7.3}
$$

This is exact for every $t\in[0,1]$, and $g$ is smooth in $t$ (the integrand is smooth in $t$, with uniformly bounded $t$-derivatives on the compact support).

**Step 2: the second derivative at $t=0$.** Write the integrand of (7.3) as $\bigl[\rho\_0' + t\,g\_1\bigr]^2(1+tv')^{-5}$ with $g\_1:=\rho\_0'v'-\rho\_0v''$, and expand using $(1+z)^{-5}=1-5z+15z^2-O(z^3)$. The coefficient of $t^2$ is $g\_1^2 - 10\,\rho\_0'\,g\_1\,v' + 15\,(\rho\_0')^2(v')^2$, which after expanding $g\_1$ becomes

$$
g''(0) = \int_{\mathbb R} \Bigl[\,6\,(\rho_0')^2 (v')^2 \;+\; 8\,\rho_0\,\rho_0'\,v'\,v'' \;+\; \rho_0^2\,(v'')^2\,\Bigr]dx.
\tag{7.4}
$$

As a pointwise quadratic form in $\bigl(\rho\_0'v',\,\rho\_0v''\bigr)$ this has matrix $\begin{pmatrix}6&4\\4&1\end{pmatrix}$ with determinant $-10<0$: it is **indefinite**, so there is room for negativity — but only if the cross term can beat both squares *after integration*, which is a global question (for an exactly affine density the cross term integrates away and $g''(0)>0$; see Reading 2).

**Step 3: the construction.** Fix $L>0$ and $\kappa>\pi/L$ (for instance $L=2$, $\kappa=\pi$), and set $\beta:=\pi/L$. Choose a smooth, compactly supported probability density $\rho\_0$ with

$$
\rho_0(x) = c\,e^{\kappa x} \quad\text{on } [0,L]
$$

for some constant $c>0$ (glue smooth decaying tails outside $[0,L]$ and normalise). Take the displacement

$$
v(x) := \int_0^x u(s)\,ds, \qquad
u(x) := e^{-\kappa x}\,\sin(\beta x)\,\mathbf 1_{[0,L]}(x) \;\ge\; 0,
$$

so $v'=u\ge0$ (Step 1 applies: $T=\mathrm{id}+v$ is optimal) and $v''=u'$ a.e. Outside $[0,L]$ the integrand of (7.4) vanishes; inside, $\rho\_0'=\kappa\rho\_0$ and — this is the point of the construction — the exponentials **cancel**:

$$
\rho_0 u = c\,\sin(\beta x), \qquad
\rho_0 u' = c\,\bigl(\beta\cos(\beta x) - \kappa\sin(\beta x)\bigr).
$$

Abbreviating $s=\sin(\beta x)$, $C=\cos(\beta x)$, the integrand of (7.4) becomes

$$
c^2\Bigl[6\kappa^2 s^2 + 8\kappa\,s\,(\beta C - \kappa s) + (\beta C - \kappa s)^2\Bigr]
= c^2\Bigl[-\kappa^2 s^2 + 6\kappa\beta\, sC + \beta^2 C^2\Bigr].
$$

Since $\beta L=\pi$ is exactly a half-period, $\int\_0^L s^2 = \int\_0^L C^2 = \frac L2$ and $\int\_0^L sC = 0$, so

$$
g''(0) = c^2\,\frac L2\,\bigl(\beta^2 - \kappa^2\bigr) = c^2\,\frac L2\Bigl(\frac{\pi^2}{L^2} - \kappa^2\Bigr) \;<\; 0
\qquad\text{precisely when } \kappa L > \pi.
$$

Since $g\in C^2$ with $g''(0)<0$, $g''<0$ on an interval $[0,\delta)\subset[0,1)$, so $t\mapsto E(\rho\_t)$ is **not convex** on $[0,1]$. Hence $E$ is not displacement convex. $\blacksquare$

**Concrete check.** For $L=2$, $\kappa=\pi$ (so $\kappa L=2\pi>\pi$): $g''(0)=-\tfrac{3\pi^2}{4}c^2\approx-7.40\,c^2$. A direct numerical evaluation of (7.3) (and, independently, of the push-forward density differentiated on a grid — the two agree to 7 digits) confirms a *macroscopic* midpoint violation, e.g. $g\bigl(\tfrac12\bigr)>\tfrac12\bigl(g(0)+g(1)\bigr)$ for the unnormalised window density.

**Reading 1 — regularity fine print.** $u$ has corners at $x=0,L$ (where $u'$ jumps), so $v\in C^{1,1}$ only; formula (7.3) and the expansion (7.4) only need $v''\in L^\infty$ with compact support, so this is harmless. Alternatively, mollify $u$: $g''(0)$ depends continuously on $u$ in $W^{1,2}$, so the strict inequality survives a small smoothing.

**Reading 2 — why this construction, and why $\kappa L>\pi$.** Integrating the cross term of (7.4) by parts ($v'$ has compact support) gives the equivalent form

$$
g''(0) = \int \bigl[\,2(\rho_0')^2 - 4\rho_0\rho_0''\,\bigr](v')^2\,dx + \int \rho_0^2\,(v'')^2\,dx.
$$

The *only* possible negative contribution is $-4\rho\_0\rho\_0''(v')^2$, active where the density is strongly **convex**: non-convexity of $E$ is created by placing the displacement on a steep convex ramp of $\rho\_0$. On the exponential window, $2(\rho\_0')^2-4\rho\_0\rho\_0''=-2\kappa^2\rho\_0^2$, and minimising the Rayleigh quotient $\int\rho\_0^2(v'')^2\,/\int\rho\_0^2(v')^2$ over $v'\in H\_0^1(0,L)$ is a weighted Sturm–Liouville problem whose ground state is exactly our $u=e^{-\kappa x}\sin(\pi x/L)$, with eigenvalue $\kappa^2+\pi^2/L^2$. Hence along the optimal disturbance $g''(0)=\bigl(\frac{\pi^2}{L^2}-\kappa^2\bigr)\int\rho\_0^2u^2$, and $\kappa L>\pi$ is the *sharp threshold* at which the convexity term beats the curvature penalty $\int\rho\_0^2(v'')^2$ — the window must be both steep ($\kappa$ large) and long ($L$ large) on the log scale.

**Reading 3 — arbitrary dimension.** Take the product density $\tilde\rho\_0(x)=\rho\_0(x\_1)\,q(x\_2,\dots,x\_d)$ with $q$ a fixed smooth compactly supported density, and the displacement $\tilde v(x)=(v(x\_1),0,\dots,0)$, which is still the gradient of a convex function. Then

$$
E(\tilde\rho_t) = \Bigl(\int q^2\Bigr)\,g(t) + \Bigl(\int\lvert\nabla' q\rvert^2\Bigr)\,h(t),
\qquad h(t):=\frac12\int\rho_t^2\,dx_1,
$$

and $h''(0)=\int\rho\_0^2(v')^2 = c^2\frac L2$ by the same half-period computation. Hence

$$
\frac{d^2}{dt^2}\Big|_{t=0} E(\tilde\rho_t) = c^2\,\frac L2\Bigl[\Bigl(\frac{\pi^2}{L^2}-\kappa^2\Bigr)\int q^2 + \int\lvert\nabla' q\rvert^2\Bigr],
$$

which is negative as soon as $\kappa$ is large enough (with $q$, $L$ fixed). So the failure is not a one-dimensional accident. Functionals of $\nabla\rho$ *can* be displacement convex, but they have to be built differently — see Carrillo–Slepčev, *Example of a displacement convex functional of first order* (Calc. Var. PDE, 2009).

</details>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Recap</span><span class="math-callout__name">(The gluing lemma and the triple coupling)</span></p>

**Gluing lemma.** Let $\mu\_1,\mu\_2,\mu\_3\in\mathcal P(\mathbb R^d)$ and let $\pi\_{1,2}\in\Pi(\mu\_1,\mu\_2)$, $\pi\_{2,3}\in\Pi(\mu\_2,\mu\_3)$ be two transference plans **sharing the middle marginal** $\mu\_2$. Then there exists a probability measure $\pi\_{1,2,3}$ on $\mathbb R^d\times\mathbb R^d\times\mathbb R^d$ — the *triple coupling* — whose projections onto the first-two and last-two factors are the given plans:

$$
(\mathrm{proj}_{1,2})_\sharp\,\pi_{1,2,3} = \pi_{1,2},
\qquad
(\mathrm{proj}_{2,3})_\sharp\,\pi_{1,2,3} = \pi_{2,3}.
$$

**General formula via disintegration.** Disintegrate both plans with respect to the shared variable $x\_2$ (regular conditional probabilities):

$$
d\pi_{1,2}(x_1,x_2) = d\pi_{1,2}^{x_2}(x_1)\,d\mu_2(x_2),
\qquad
d\pi_{2,3}(x_2,x_3) = d\pi_{2,3}^{x_2}(x_3)\,d\mu_2(x_2),
$$

and glue **conditionally independently**:

$$
d\pi_{1,2,3}(x_1,x_2,x_3) := d\pi_{1,2}^{x_2}(x_1)\;d\pi_{2,3}^{x_2}(x_3)\;d\mu_2(x_2).
$$

Probabilistically: draw the middle variable $x\_2\sim\mu\_2$; given $x\_2$, draw $x\_1$ and $x\_3$ *independently* from the two conditionals — a three-step Markov chain $X\_1 - X\_2 - X\_3$. The payoff is the $1{,}3$-marginal $(\mathrm{proj}\_{1,3})\_\sharp\pi\_{1,2,3}\in\Pi(\mu\_1,\mu\_3)$, which is exactly the competitor plan used to prove the triangle inequality for the Wasserstein distance.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.2</span><span class="math-callout__name">(The triple coupling, explicitly)</span></p>

Write down the formula for the "triple coupling" in the gluing lemma in the case: $\pi\_{1,2}$ and $\pi\_{2,3}$ are probability measures on $\mathbb R^d\times\mathbb R^d$ and

**a)** $\pi\_{1,2}$ and $\pi\_{2,3}$ are absolutely continuous with respect to the Lebesgue measure, i.e.

$$
d\pi_{1,2} = f_{1,2}(x_1,x_2)\,dx_1\,dx_2, \qquad
d\pi_{2,3} = f_{2,3}(x_2,x_3)\,dx_2\,dx_3
$$

for $f\_{1,2},f\_{2,3}\in L^1$ with respect to the Lebesgue measure.

**b)** $\pi\_{1,2}$ and $\pi\_{2,3}$ are supported on graphs:

$$
\pi_{1,2} = (\mathrm{id},\,T_{1,2})_\sharp\,\mu_1, \qquad
\pi_{2,3} = (\mathrm{id},\,T_{2,3})_\sharp\,\mu_2,
$$

where $T\_{1,2},T\_{2,3}\colon\mathbb R^d\to\mathbb R^d$ are measurable and $\mu\_1,\mu\_2$ are probability measures on $\mathbb R^d$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 7.2 (a) — absolutely continuous plans: the Bayes/Markov formula</summary>

**The shared marginal.** The middle marginal $\mu\_2$ is absolutely continuous with density

$$
f_2(x_2) := \int_{\mathbb R^d} f_{1,2}(x_1,x_2)\,dx_1,
$$

and the gluing hypothesis ("$\pi\_{1,2}$ and $\pi\_{2,3}$ share the middle marginal") reads

$$
\int_{\mathbb R^d} f_{1,2}(x_1,x_2)\,dx_1 = f_2(x_2) = \int_{\mathbb R^d} f_{2,3}(x_2,x_3)\,dx_3
\qquad\text{for a.e. } x_2.
$$

**The formula.** The triple coupling is absolutely continuous on $(\mathbb R^d)^3$, $d\pi\_{1,2,3}=f\_{1,2,3}\,dx\_1\,dx\_2\,dx\_3$, with

$$
f_{1,2,3}(x_1,x_2,x_3) :=
\begin{cases}
\dfrac{f_{1,2}(x_1,x_2)\,f_{2,3}(x_2,x_3)}{f_2(x_2)}, & f_2(x_2)>0,\\[6pt]
0, & f_2(x_2)=0.
\end{cases}
\tag{7.5}
$$

This is the general disintegration formula made explicit: the conditionals are $d\pi\_{1,2}^{x\_2}(x\_1)=\frac{f\_{1,2}(x\_1,x\_2)}{f\_2(x\_2)}\,dx\_1$ and $d\pi\_{2,3}^{x\_2}(x\_3)=\frac{f\_{2,3}(x\_2,x\_3)}{f\_2(x\_2)}\,dx\_3$, so (7.5) is "conditional density of $x\_1$ given $x\_2$, times density of $x\_2$, times conditional density of $x\_3$ given $x\_2$" — the joint law of a Markov chain in which $x\_1$ and $x\_3$ are **conditionally independent given $x\_2$**.

**Why the zero set is harmless.** On $N:=\lbrace f\_2=0\rbrace$ we declared $f\_{1,2,3}=0$; no mass is lost, because the plans put none there:

$$
\int_{\mathbb R^d\times N} f_{1,2}\,dx_1\,dx_2 = \int_N f_2\,dx_2 = 0,
$$

so $f\_{1,2}(\cdot,x\_2)=0$ for a.e. $x\_2\in N$ (and likewise for $f\_{2,3}$).

**Verification of the two marginals.** For $f\_2(x\_2)>0$,

$$
\int_{\mathbb R^d} f_{1,2,3}(x_1,x_2,x_3)\,dx_3
= \frac{f_{1,2}(x_1,x_2)}{f_2(x_2)}\int_{\mathbb R^d} f_{2,3}(x_2,x_3)\,dx_3
= f_{1,2}(x_1,x_2),
$$

using the shared-marginal identity; on $N$ both sides vanish a.e. So $(\mathrm{proj}\_{1,2})\_\sharp\pi\_{1,2,3}=\pi\_{1,2}$, and symmetrically (integrating out $x\_1$) $(\mathrm{proj}\_{2,3})\_\sharp\pi\_{1,2,3}=\pi\_{2,3}$. In particular $\pi\_{1,2,3}$ is a probability measure. $\blacksquare$

**Reading.** Integrating out the middle variable gives the $1{,}3$-marginal

$$
f_{1,3}(x_1,x_3) = \int_{\lbrace f_2>0\rbrace} \frac{f_{1,2}(x_1,x_2)\,f_{2,3}(x_2,x_3)}{f_2(x_2)}\,dx_2
$$

— a Chapman–Kolmogorov-type composition of the two plans, and the coupling of $\mu\_1,\mu\_3$ that feeds the triangle inequality for $W\_2$.

</details>

<details class="accordion" markdown="1">
<summary>Solution 7.2 (b) — plans on graphs: composition of maps</summary>

**Compatibility.** The second marginal of $\pi\_{1,2}=(\mathrm{id},T\_{1,2})\_\sharp\mu\_1$ is $(T\_{1,2})\_\sharp\mu\_1$, while the first marginal of $\pi\_{2,3}=(\mathrm{id},T\_{2,3})\_\sharp\mu\_2$ is $\mu\_2$. The gluing hypothesis therefore forces

$$
(T_{1,2})_\sharp\,\mu_1 = \mu_2.
$$

**The formula.** The triple coupling is

$$
\pi_{1,2,3} = \bigl(\mathrm{id},\; T_{1,2},\; T_{2,3}\circ T_{1,2}\bigr)_\sharp\,\mu_1,
\tag{7.6}
$$

i.e. $\int \zeta\,d\pi\_{1,2,3} = \int \zeta\bigl(x,\,T\_{1,2}(x),\,T\_{2,3}(T\_{1,2}(x))\bigr)\,d\mu\_1(x)$ for bounded measurable $\zeta$. Equivalently, $\pi\_{1,2,3}$ is the push-forward of $\pi\_{1,2}$ under $(x\_1,x\_2)\mapsto(x\_1,x\_2,T\_{2,3}(x\_2))$: the third coordinate is a *deterministic continuation* of the second.

**Verification of the two marginals.** Using $(F\circ G)\_\sharp = F\_\sharp\, G\_\sharp$:

$$
(\mathrm{proj}_{1,2})_\sharp\,\pi_{1,2,3}
= \bigl(\mathrm{id},\,T_{1,2}\bigr)_\sharp\,\mu_1 = \pi_{1,2},
$$

since $\mathrm{proj}\_{1,2}\circ(\mathrm{id},T\_{1,2},T\_{2,3}\circ T\_{1,2}) = (\mathrm{id},T\_{1,2})$; and since $\mathrm{proj}\_{2,3}\circ(\mathrm{id},T\_{1,2},T\_{2,3}\circ T\_{1,2}) = \bigl(T\_{1,2},\,T\_{2,3}\circ T\_{1,2}\bigr) = (\mathrm{id},T\_{2,3})\circ T\_{1,2}$,

$$
(\mathrm{proj}_{2,3})_\sharp\,\pi_{1,2,3}
= (\mathrm{id},T_{2,3})_\sharp\bigl[(T_{1,2})_\sharp\,\mu_1\bigr]
= (\mathrm{id},T_{2,3})_\sharp\,\mu_2 = \pi_{2,3}. \qquad\blacksquare
$$

**Reading.** (i) Formula (7.6) is the deterministic special case of the general disintegration formula: the conditional of $\pi\_{2,3}$ given $x\_2$ is the Dirac $\pi\_{2,3}^{x\_2}=\delta\_{T\_{2,3}(x\_2)}$. Note the asymmetry: the conditional of $\pi\_{1,2}$ given its *second* variable $x\_2$ need **not** be a Dirac ($T\_{1,2}$ may fail to be injective — the conditional $\pi\_{1,2}^{x\_2}$ lives on the fibre $T\_{1,2}^{-1}(\lbrace x\_2\rbrace)$), but the formula never needs it. (ii) The $1{,}3$-marginal is $(\mathrm{id},\,T\_{2,3}\circ T\_{1,2})\_\sharp\,\mu\_1$: gluing graph plans is **composition of transport maps**. When $T\_{1,2}$ and $T\_{2,3}$ are optimal maps, this composed plan is the (in general suboptimal) coupling of $\mu\_1$ and $\mu\_3$ behind the triangle inequality $W\_2(\mu\_1,\mu\_3)\le W\_2(\mu\_1,\mu\_2)+W\_2(\mu\_2,\mu\_3)$.

</details>