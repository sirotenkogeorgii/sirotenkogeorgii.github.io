---
title: Problems from the PDEs in Data Science course
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

# PDEs in Data Science — Course Problems

**Table of Contents**
- TOC
{:toc}

---

## Exercise Sheet 1 — Subdifferentials and convex functions

*Keywords: convex function, subdifferential, supporting hyperplane, subgradient monotonicity, local vs. global minima.*

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

<details class="accordion" markdown="1">
<summary>Visualisation — supporting affine minorants</summary>

<figure>
  <img src="{{ '/assets/images/notes/books/pdeds/sheet1_supporting_epigraph.png' | relative_url }}" alt="A convex function with its epigraph shaded; one supporting affine line at a smooth point and a fan of supporting lines at a kink" loading="lazy">
  <figcaption>Each $p \in \partial E(x_0)$ is the slope of a hyperplane supporting $\mathrm{epi}(E)$ at $(x_0, E(x_0))$. At a smooth point this hyperplane is unique (the tangent); at a kink there is a whole fan of them.</figcaption>
</figure>

</details>

---

### Exercise 1.1 — Subdifferential

Show the following statements.

**a)** The set $\partial E(x)$ is convex for all $x \in \mathbb{R}^N$. Furthermore, $\partial E(x)$ is non-empty if $E$ is continuous at $x$ and $E(x) < \infty$.

**b)** Let $E$ be lower semicontinuous and $(x_k)\_{k \in \mathbb{N}}, (p_k)\_{k \in \mathbb{N}} \subset \mathbb{R}^N$ such that $p_k \in \partial E(x_k)$. Additionally, let $x, p \in \mathbb{R}^N$ such that $x_k \to x$ and $p_k \to p$ as $k \to \infty$. Then $p \in \partial E(x)$.

**c)** Let $E \in C^1(\mathbb{R}^N)$. Then $\partial E(x) = \lbrace \nabla E(x)\rbrace$ for all $x \in \mathbb{R}^N$.

**d)** $E$ is differentiable at $x$ if and only if $\partial E(x)$ is a singleton, i.e. $\#\partial E(x) = 1$.

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

---

### Exercise 1.2 — Convex functions

Let $E : \mathbb{R}^N \to [0, \infty)$ be convex and finite-valued.

**a)** Let $x_i, p_i \in \mathbb{R}^N$ such that $p_i \in \partial E(x_i)$ for $i = 1, 2$. Show that

$$\langle p_1 - p_2,\, x_1 - x_2 \rangle \;\ge\; 0.$$

**b)** Show the equivalence of the following statements:

  i) $x_0 \in \mathbb{R}^N$ is a local minimum of $E$.

  ii) $x_0 \in \mathbb{R}^N$ is a global minimum of $E$.

  iii) $0 \in \partial E(x_0)$.

**c)** *(Bonus)* Show that $E$ is continuous.

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
