---
title: Problems from the SDEs & Diffusion Models course
layout: default
noindex: true
tags:
  - stochastic-differential-equations
  - stochastic-processes
  - markov-processes
  - levy-processes
  - feller-semigroups
  - functional-analysis
  - exercises
---

# SDEs & Diffusion Models — Course Problems

**Table of Contents**
- TOC
{:toc}

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">QUESTIONS</span><span class="math-callout__name">(SOLVE!!!)</span></p>



</div>

Let $d \in \mathbb{N}$ be arbitrarily fixed. Throughout this exercise sheet, we write

$$|x|_{\mathbb{R}^d} := \left( \sum_{i=1}^d x_i^2 \right)^{1/2}$$

for the Euclidean standard norm/2-norm of an arbitrary vector $x := (x_1,\ldots,x_d)^\top \in \mathbb{R}^d$

Recall the space $(\mathcal{B}\_b(\mathbb{R}^d),\lVert\cdot\rVert_\infty)$ of all bounded measurable functions $\mathbb{R}^d \to \mathbb{R}$ as well as $(C_\infty(\mathbb{R}^d),\lVert\cdot\rVert_\infty)$, the space of all continuous functions $\mathbb{R}^d \to \mathbb{R}$, vanishing at infinity; see script, p. 18.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name"></span></p>

Throughout, $(P_t)\_{t\ge 0}$ is a Feller semigroup on $\mathcal B_b(\mathbb R^d)$ in the sense of Definition 2.8, i.e. it satisfies properties (b)–(f) of Proposition 2.7: contraction $\lVert P_t u\rVert_\infty\le\lVert u\rVert_\infty$ on $\mathcal B_b$, positivity preservation, the sub‑Markov property, the Feller property $P_t(C_\infty)\subseteq C_\infty$, and strong continuity $\lVert P_tu-u\rVert_\infty\to 0$ on $C_\infty$. We write $C_\infty=C_\infty(\mathbb R^d)$, $\mathcal B_b=\mathcal B_b(\mathbb R^d)$, $\mathbf 1$ for the constant function, and $\mathrm{Id}$ for the identity. All operator limits are in the uniform norm $\lVert\cdot\rVert_\infty$ unless stated otherwise.
 
A fact we use repeatedly without comment: **for $u\in C_\infty$ the orbit $t\mapsto P_tu$ is $\lVert\cdot\rVert_\infty$‑continuous.** Indeed, by the semigroup law and contraction,

$$
\lVert P_{t+h}u-P_tu\rVert_\infty=\lVert P_t(P_hu-u)\rVert_\infty\le\lVert P_hu-u\rVert_\infty,\qquad
\lVert P_tu-P_{t-h}u\rVert_\infty\le\lVert P_hu-u\rVert_\infty,
$$

and the right‑hand sides vanish as $h\downarrow 0$ by strong continuity at $0$.

</div>

## Exercise 4.1 — $\alpha$-Potential Operators

This exercise is concerned with **$\alpha$-potential operators**, associated to a given Feller semigroup $(P_t)\_{t \geq 0}$ on $\mathcal{B}\_b(\mathbb{R}^d)$; see script, p. 20.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">($\alpha$-Potential Operator)</span></p>

For any $\alpha > 0$, the **$\alpha$-potential operator** of $(P_t)\_{t \geq 0}$ is the **Laplace transform**

$$
U_\alpha : \mathcal{B}_b(\mathbb{R}^d) \to \mathcal{B}_b(\mathbb{R}^d),
\qquad
U_\alpha u := \int_0^\infty e^{-\alpha t} \cdot P_t u \, \mathrm{d}t.
$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">($\alpha$-potential operator existence and linearity)</span></p>

Verify that this definition makes sense in the first place, namely show:

**(a)** For any $\alpha > 0$, the $\alpha$-potential operator $U_\alpha$ above exists, and it defines a linear operator

$$
\mathcal{B}_b(\mathbb{R}^d) \to \mathcal{B}_b(\mathbb{R}^d).
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>What does it mean to show the existence of the operator $U_\alpha$?</summary>

In the context of Exercise 4.1, showing the "existence" of the $\alpha$-potential operator $U_\alpha$ means mathematically proving that the improper integral used to define it actually converges to a finite, well-defined value for any valid input function.

Here is a breakdown of what you exactly need to show for part (a):

#### 1. The Integral Converges (Existence)

You are given the definition:

$$U_\alpha u := \int_0^\infty e^{-\alpha t} \cdot P_t u \, dt$$

To show this exists, you must prove that taking the absolute value of the integral yields a finite number for any function $u \in \mathcal{B}\_b(\mathbb{R}^d)$ and any $x \in \mathbb{R}^d$.

**How to approach it:**

* You know that $u$ is a bounded measurable function, meaning there is some finite bound $\|u\|_\infty$.

* You also know $(P_t)\_{t \ge 0}$ is a Feller semigroup on $\mathcal{B}\_b(\mathbb{R}^d)$. A key property of Markov/Feller semigroups is that they are contractions (or at least bounded), meaning $\|P_t u\|_\infty \le \|u\|_\infty$.

* By taking the absolute value of the integrand, you can bound it:

$$|e^{-\alpha t} P_t u(x)| \le e^{-\alpha t} \|P_t u\|_\infty \le e^{-\alpha t} \|u\|_\infty$$

* Since $\alpha > 0$, the exponential term $e^{-\alpha t}$ decays to zero as $t \to \infty$. Evaluating the integral of this bound from $0$ to $\infty$ will result in $\frac{\|u\|_\infty}{\alpha}$, which is a finite number. Because the integral is absolutely convergent, the operator exists.

#### 2. It Maps to $\mathcal{B}\_b(\mathbb{R}^d)$

The problem also states you need to show it defines a linear operator $\mathcal{B}\_b(\mathbb{R}^d) \rightarrow \mathcal{B}\_b(\mathbb{R}^d)$.

* **Boundedness:** The convergence proof above inherently shows that $\|U_\alpha u\|_\infty \le \frac{\|u\|_\infty}{\alpha}$. Because this bound is finite, $U_\alpha u$ is a bounded function.
* **Measurability:** You must briefly acknowledge that because $P_t u$ is measurable and the integral is a limit of measurable Riemann sums, $U_\alpha u$ remains a measurable function.

#### 3. It is a Linear Operator

Finally, you must show that $U_\alpha$ is linear. This means demonstrating that for any constants $c, k \in \mathbb{R}$ and functions $u, v \in \mathcal{B}\_b(\mathbb{R}^d)$:

$$U_\alpha(cu + kv) = c U_\alpha u + k U_\alpha v$$

This follows directly from the linearity of the integral and the linearity of the Feller semigroup $(P_t)\_{t \ge 0}$.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>But cannot a proper operator take infinite values?</summary>

Yes, an operator can take infinite values, but it depends heavily on the mathematical definition and the specific space you are working in.

For this specific $\alpha$-potential operator, showing existence means proving that for any bounded measurable function $u$, the integral converges to a well-defined, finite function that also belongs to $\mathcal{B}\_b(\mathbb{R}^d)$.

#### Why This Operator Cannot Take Infinite Values

In your previous question, you noted that operators can take infinite values. However, this specific operator stays finite due to three constraints:

* **Bounded Inputs:** The domain $\mathcal{B}\_b$ contains only functions where $\vert{}u(x)\vert{} \leq M$ for some finite constant $M$.
* **Semigroup Property:** The transition operator $P_t$ is a contraction, meaning $\Vert{}P_t u\Vert{}\_\infty \leq \Vert{}u\Vert{}\_\infty$. It cannot amplify the function's size.
* **Exponential Decay:** Because $\alpha > 0$, the term $e^{-\alpha t}$ acts as a powerful dampening factor over time.

#### How to Prove Existence (The Steps)

To complete exercise 4.1, you must formally show that the output function $U_\alpha u$ is bounded and measurable:

* **Step 1: Bound the integrand.** Replace $P_t u$ with its maximum possible value, $\Vert{}u\Vert{}\_\infty$.
* **Step 2: Evaluate the integral.** Solve $\int_0^\infty e^{-\alpha t} \Vert{}u\Vert{}\_\infty \, \mathrm{d}t$.
* **Step 3: Show finiteness.** The calculation yields $\Vert{}U_\alpha u\Vert{}\_\infty \leq \frac{1}{\alpha} \Vert{}u\Vert{}\_\infty$.
* **Step 4: Prove measurability.** Use Fubini's theorem to show the integral preserves measurability.

Since $\alpha > 0$ and $\Vert{}u\Vert{}\_\infty$ is finite, the final output is guaranteed to be a finite, bounded function.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Why are you considering the absolute value of the integrand? To show that the upper bound of the given integral is bounded implying that the given integral is bounded also, or there is some deeper reason to show that the operator maps the function to $L^1$ function?</summary>

Taking the absolute value is precisely to establish an upper bound to prove the integral converges to a finite number.

Here is the breakdown of exactly why we do this and why your second hypothesis (about $L^1$ functions) is a very smart thought, but slightly misapplied in this specific context.

#### 1. The Primary Reason: Absolute Convergence

In mathematics—particularly in the measure theory and Lebesgue integration framework typically used at this level of stochastic analysis—an integral $\int f \, d\mu$ is considered well-defined and finite if and only if the integral of its absolute value is finite: $\int \lvert f\rvert \, d\mu < \infty$.

By looking at the absolute value of the integrand, we can use the **comparison test**. If we can find a larger, simpler function that we *know* integrates to a finite number, then our original integral must also be finite.

Because we know the Feller semigroup $(P_t)\_{t \ge 0}$ is a contraction (or at least bounded), we know that $\lvert P_t u(x)\rvert \le \|u\|_\infty$. This allows us to bound the complex integrand with a simple exponential decay function:

$$\int_0^\infty |e^{-\alpha t} P_t u(x)| \, dt \le \int_0^\infty e^{-\alpha t} \|u\|_\infty \, dt = \frac{\|u\|_\infty}{\alpha}$$

Because $\frac{\|u\|_\infty}{\alpha}$ is a finite number, we have proven the integral exists and is bounded.

#### 2. Clarifying the $L^1$ Misconception

Your second thought—wondering if this relates to $L^1$ functions—is a very natural connection to make when you see an integral of an absolute value. However, it is important to distinguish between the **time variable ($t$)** and the **spatial variable ($x$)**.

* **Integrability in Time:** By showing the integral of the absolute value is finite, we *are* technically showing that for any fixed spatial point $x$, the function $t \mapsto e^{-\alpha t} P_t u(x)$ belongs to $L^1([0, \infty))$. We need this to ensure the operator gives us a finite number at every point $x$.
* **The Output Space (Space):** We are **not** trying to show that the resulting function $U_\alpha u$ belongs to $L^1(\mathbb{R}^d)$ with respect to the spatial variable $x$. In fact, Exercise 4.1 explicitly states that the $\alpha$-potential operator maps bounded measurable functions to bounded measurable functions: $U_\alpha: \mathcal{B}\_b(\mathbb{R}^d) \rightarrow \mathcal{B}\_b(\mathbb{R}^d)$. A function can be bounded without being integrable over the entire real line (for example, the constant function $f(x) = 1$ is in $\mathcal{B}\_b(\mathbb{R}^d)$ but not in $L^1(\mathbb{R}^d)$).

**Summary:** You take the absolute value to prove the improper integral over time $t$ converges to a finite number via the comparison test. It guarantees the operator $U_\alpha u(x)$ is mathematically well-defined at every point $x$, rather than mapping the function into a spatial $L^1$ space.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Short Solution</summary>

Let $\alpha > 0$, and let $u\in\mathbb{B}\_b(\mathbb{R}^d)$ be arbitrary. Since the Feller semigroup $(P_t)\_{t\geq 0}$ is a contraction on $\mathbb{B}\_b(\mathbb{R}^d)$, we have

$$\lvert e^{-\alpha t} \cdot P_t u \rvert \leq e^{-\alpha t} \cdot \| P_t u \|_\infty \leq e^{-\alpha t} \cdot \|u\|_\infty, \quad \forall t\in[0, \infty)$$

where the first inequality comes from the definition of infinity norm as a supremum over all values, the second inequality comes from the contraction property of the Feller semigroup $\|Pu\|_\infty \leq \|u\|_\infty$. Then

$$\int_0^\infty \lvert e^{-\alpha t} \cdot P_t u \rvert \, \mathrm{d}t \leq \| u\|_\infty \int_0^\infty \lvert e^{-\alpha t} \, \mathrm{d}t = \alpha^{-1}\| u\|_\infty \le \infty,$$

where $\| u\|_\infty \le \infty$, because $u\in\mathbb{B}\_b(\mathbb{R}^d)$. This guarantees the existence of the integral in the definition. The linearity of $U_\alpha$ is an immediate consequence of the linearity of $(P_t)\_{t\geq 0}$ and the integral.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Prove properties of $\alpha$-potential operator)</span></p>

The $\alpha$-potential operators share many properties of the original semigroup $(P_t)\_{t \geq 0}$. Prove the following counterpart of Proposition 2.7:

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Properties of $\alpha$-potential operator)</span></p>

Let $(U_\alpha)\_{\alpha > 0}$ be the family of $\alpha$-potential operators of $(P_t)\_{t \geq 0}$. Then, for any $\alpha > 0$, we have:

**(b) Conservativity.** If $(P_t)\_{t \geq 0}$ is conservative, then $\alpha \cdot U_\alpha$ is conservative as well,

$$
\alpha \cdot U_\alpha 1 = 1.
$$

**(c) Contraction on $\mathcal{B}\_b(\mathbb{R}^d)$.**

$$
\lVert\alpha \cdot U_\alpha u\rVert_\infty \leq \lVert u\rVert_\infty
\qquad \text{for all } u \in \mathcal{B}_b(\mathbb{R}^d).
$$

**(d) Positivity Preserving on $\mathcal{B}\_b(\mathbb{R}^d)$.**

$$
u \in \mathcal{B}_b(\mathbb{R}^d) \text{ with } u \geq 0
\quad \Longrightarrow \quad
\alpha \cdot U_\alpha u \geq 0.
$$

**(e) Feller Property on $C_\infty(\mathbb{R}^d)$.**

$$
u \in C_\infty(\mathbb{R}^d)
\quad \Longrightarrow \quad
\alpha \cdot U_\alpha u \in C_\infty(\mathbb{R}^d).
$$

**(f) Strong Continuity on $C_\infty(\mathbb{R}^d)$.**

$$
\lim_{\alpha \to \infty} \lVert\alpha \cdot U_\alpha u - u\rVert_\infty = 0
\qquad \text{for all } u \in C_\infty(\mathbb{R}^d).
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution (b)</summary>

Suppose that $(P_t)\_{t \geq 0}$ is conservative, that is, $P_t 1 = 1$. Then, it follows

$$\alpha \cdot U_\alpha 1=\alpha \int_0^\infty e^{-\alpha t} \cdot P_t 1 \, dt=\alpha \int_0^\infty e^{-\alpha t} \, dt=1,$$

meaning that $\alpha \cdot U_\alpha$ is indeed conservative.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution (c)</summary>

Since $(P_t)\_{t \geq 0}$ is a Feller semigroup, it is a contraction on $\mathcal{B}\_b(\mathbb{R}^d)$, that is, $\|P_t u\|_\infty \leq \|u\|_\infty$ for all $u \in \mathcal{B}\_b(\mathbb{R}^d)$. Then, for any $u \in \mathcal{B}\_b(\mathbb{R}^d)$, it follows

$$\|\alpha \cdot U_\alpha u\|_\infty \leq \alpha \int_0^\infty e^{-\alpha t} \cdot \|P_t u\|_\infty \, dt \leq \alpha \cdot \|u\|_\infty \int_0^\infty e^{-\alpha t} \, dt = \|u\|_\infty,$$

meaning that $\alpha \cdot U_\alpha$ is indeed a contraction on $\mathcal{B}\_b(\mathbb{R}^d)$.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution (d)</summary>

Let $u \in \mathcal{B}\_b(\mathbb{R}^d)$ with $u \geq 0$ be arbitrary. Since $(P_t)\_{t \geq 0}$ is a Feller semigroup, it is positivity preserving on $\mathcal{B}_b(\mathbb{R}^d)$ such that $P_t u \geq 0$. Then, it follows

$$\alpha \cdot U_\alpha u = \alpha \int_0^\infty e^{-\alpha t} \cdot P_t u \, dt \geq 0,$$

meaning that $\alpha \cdot U_\alpha$ is indeed positivity preserving on $\mathcal{B}\_b(\mathbb{R}^d)$.
\end{enumerate}

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution (e)</summary>


</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution (f)</summary>


</details>
</div>


## Exercise 4.2 — Heat Kernel Bound II

We continue on the discussion from Exercise 3.2 by deriving another $L^2$-heat kernel bound.
In that direction, for any $t > 0$, we consider the heat kernel $q_t : \mathbb{R}^d \to \mathbb{R}$, given by

$$
q_t(x) := p_t(x,0)
= \frac{1}{(2\pi t)^{d/2}} \cdot
\exp\left( -\frac{|x|_{\mathbb{R}^d}^2}{2t} \right)
\qquad \text{for all } x \in \mathbb{R}^d,
$$

where we recall the Gaussian transition density $p_t$ from Definition 2.4. In order to simplify the notation, in what follows, let us introduce the norm

$$
\lVert\cdot\rVert_{(L^1 \cap L^2)(\mathbb{R}^d)} :
L^1(\mathbb{R}^d) \cap L^2(\mathbb{R}^d) \to [0,\infty),
\qquad
\lVert f\rVert_{(L^1 \cap L^2)(\mathbb{R}^d)}
:= \lVert f\rVert_{L^1(\mathbb{R}^d)} + \lVert f\rVert_{L^2(\mathbb{R}^d)}.
$$

Prove the following lemma:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Heat Kernel Bound)</span></p>

Let $u \in L^2(\mathbb{R}^d)$ be arbitrary. Then, for any $t > 0$, it holds

$$
\lVert|\cdot|_{\mathbb{R}^d}^2 \cdot (q_t * u)\rVert_{L^2(\mathbb{R}^d)}^2
\leq
C \cdot (1 \vee t) \cdot \left(1 \wedge t^{-d/4}\right)
\cdot
\lVert\left(1 + |\cdot|_{\mathbb{R}^d} + |\cdot|_{\mathbb{R}^d}^2\right) \cdot u\rVert_{(L^1 \cap L^2)(\mathbb{R}^d)},
$$

where the numerical constant $C := C(d) > 0$ only depends on the dimension $d$.

</div>

**Hints.**
For every $i \in \lbrace 1,\ldots,d\rbrace$, let $\Pi_i : \mathbb{R}^d \to \mathbb{R}$ be the canonical projection, given by
$\Pi_i(x) := x_i$ for all $x := (x_1,\ldots,x_d)^\top \in \mathbb{R}^d$. You may then proceed as follows:

- For any $x := (x_1,\ldots,x_d)^\top \in \mathbb{R}^d$ and any $i \in \lbrace 1,\ldots,d\rbrace$, first show the identity

  $$
      x_i \cdot (q_t * u)(x)
      = -t \cdot (\partial_i q_t * u)(x)
      + \bigl(q_t * (\Pi_i \cdot u)\bigr)(x),
  $$

  where $\partial_i := \frac{\partial}{\partial x_i}$ denotes the partial derivative with respect to $x_i$.

- For any $i \in \lbrace 1,\ldots,d\rbrace$, write $v_i := q_t * (\Pi_i \cdot u)$, and use the first bullet point to infer

  $$
      \lVert|\cdot|_{\mathbb{R}^d}^2 \cdot (q_t * u)\rVert_{L^2(\mathbb{R}^d)}
      \leq
      t^2 \cdot \lVert\Delta(q_t * u)\rVert_{L^2(\mathbb{R}^d)}
      + t \sum_{i=1}^d \lVert\partial_i v_i\rVert_{L^2(\mathbb{R}^d)}
      + \lVert q_t * \left(|\cdot|_{\mathbb{R}^d}^2 \cdot u\right)\rVert_{L^2(\mathbb{R}^d)}.
  $$

- Regarding the sum in the previous bound, for any $i \in \lbrace 1,\ldots,d\rbrace$, verify that

  $$
      \lVert\partial_i v_i\rVert_{L^2(\mathbb{R}^d)}^2
      \leq
      \lVert\Delta\bigl(q_t * (\Pi_i \cdot u)\bigr)\rVert_{L^2(\mathbb{R}^d)}
      \cdot
      \lVert q_t * (\Pi_i \cdot u)\rVert_{L^2(\mathbb{R}^d)}.
  $$

- Apply the heat kernel bounds from Exercise 3.2 (b),(c) twice and conclude.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name">(Convolution Convention)</span></p>

For functions $f,g \in L^1(\mathbb{R}^d)$, the (additive) convolution of $f$ and $g$ is defined as

$$
    (f * g)(x) := \int_{\mathbb{R}^d} f(y) \cdot g(x-y)\,\mathrm{d}y
    = \int_{\mathbb{R}^d} f(x-y) \cdot g(y)\,\mathrm{d}y
    \qquad \text{for all } x \in \mathbb{R}^d,
$$

whenever the integral is finite.

</div>

## Exercise 4.3 — Lévy Process, Markov Process, Feller Process

This exercise deals with real-valued Lévy processes on some probability space $(\Omega,\mathcal{A},\mathbb{P})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Real-Valued Lévy Process)</span></p>

A real-valued stochastic process $L := (L_t)\_{t \geq 0}$ on $(\Omega,\mathcal{A},\mathbb{P})$ is called a Lévy process iff it fulfils the following properties:

- **(L1) Starting in $0$.** $L_0 = 0$, $\mathbb{P}$-a.s.

- **(L2) Independent Increments.** For any $n \in \mathbb{N}$ and $0 =: t_0 < t_1 < \cdots < t_n$, the increments

  $$L_{t_1} - L_{t_0}, \quad L_{t_2} - L_{t_1}, \quad \ldots, \quad L_{t_n} - L_{t_{n-1}}$$

  are mutually stochastically independent.

- **(L3) Stationary Increments.** For any $t \geq s \geq 0$, it holds

  $$L_t - L_s \overset{d}{=} L_{t-s}.$$

- **(L4) Continuity in Probability.** For any $\delta > 0$ and any $t \geq 0$, it holds

  $$\lim_{h \to 0} \mathbb{P}\left( |L_{t+h} - L_t| > \delta \right) = 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Lévy process, Markov process, Feller process)</span></p>

In the sequel, let $L := (L_t)\_{t \geq 0}$ be a fixed Lévy process. We denote its canonical/natural filtration by
$\mathbb{F} := (\mathcal{F}\_t)\_{t \geq 0}$, that is,

$$\mathcal{F}_t := \sigma\bigl(L_s : s \in [0,t]\bigr) \qquad \text{for all } t \geq 0.$$

The goal is to establish the Feller property for $L$ (see Definition 2.11). To do so, work along the following tasks:

**(a)** Prove that $L$ defines a Markov process.

**Hint.** You have to verify the Markov property for $L$, with the canonical/natural filtration
$\mathbb{F} = (\mathcal{F}\_t)\_{t \geq 0}$. To achieve this, you may reproduce and adapt the arguments from the proof of Proposition 0.15 to $L$. In particular, use without proof that the paths of $L$ can — after possibly replacing $L$ by a suitable modification — be assumed to lie in the Skorokhod space $D([0,\infty))$ of real-valued càdlàg functions $[0,\infty) \to \mathbb{R}$.

Let now $(P_t^L)\_{t \geq 0}$ be the transition semigroup, associated to $L$.

**(b)** Verify that $(P_t^L)\_{t \geq 0}$ is indeed an operator semigroup on $\mathcal{B}\_b(\mathbb{R})$.

**Hint.** Use without proof that, for any $u \in \mathcal{B}\_b(\mathbb{R})$, it holds $P_t^L u \in \mathcal{B}\_b(\mathbb{R})$ for all $t \geq 0$.

**(c)** Conclude by showing that $(P_t^L)\_{t \geq 0}$ admits all properties to be a Feller semigroup.

</div>


## Exercise 4.4 — Resolvent

We relate the $\alpha$-potential operators from Exercise 4.1 to the resolvent of the respective Feller generator (see Definition 2.13). In general, the resolvent of an (un-)bounded linear operator constitutes a well-known and widely used construction in Functional Analysis. We verify that the potential operators and the resolvent of the Feller generator in fact coincide, and further derive two utterly useful identities. More concretely, prove the following proposition:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Resolvent Identities)</span></p>

Let $(P_t)\_{t \geq 0}$ be a Feller semigroup on $\mathcal{B}\_b(\mathbb{R}^d)$ with Feller generator $(A,\mathcal{D}(A))$.
Denote the family of $\alpha$-potential operators of $(P_t)\_{t \geq 0}$ by $(U_\alpha)_{\alpha > 0}$.

**(a)** For any $\alpha > 0$, the regularisation $\alpha \cdot \mathrm{Id}\_{\mathcal{D}(A)} - A$ is invertible, with bounded linear inverse

$$
\left(\alpha \cdot \mathrm{Id}_{\mathcal{D}(A)} - A\right)^{-1} = U_\alpha.
$$

**(b) First/Hilbert's Resolvent Identity.** For any $\alpha,\beta > 0$, we have

$$
U_\alpha - U_\beta
= (\beta - \alpha) \cdot U_\alpha U_\beta
\qquad \text{on } \mathcal{B}_b(\mathbb{R}^d).
$$

Let now $(Q_t)\_{t \geq 0}$ be another Feller semigroup on $\mathcal{B}\_b(\mathbb{R}^d)$ with Feller generator $(C,\mathcal{D}(C))$, and denote the corresponding family of $\alpha$-potential operators by $(V_\alpha)\_{\alpha > 0}$. Suppose in addition that $\mathcal{D}(A) = \mathcal{D}(C)$.

**(c) Second Resolvent Identity.** For any $\alpha > 0$, we have

$$
U_\alpha - V_\alpha
= V_\alpha (A - C) U_\alpha
\qquad \text{on } \mathcal{B}_b(\mathbb{R}^d).
$$

**Hint for Part (a).** Let $\alpha > 0$ be arbitrary. Show that the $\alpha$-potential operator $U_\alpha$ is the left as well as the right inverse of the regularisation $\alpha \cdot \mathrm{Id}\_{\mathcal{D}(A)} - A$. To approach this, you may first verify that $U_\alpha u \in \mathcal{D}(A)$, with

$$
A U_\alpha u = \alpha \cdot U_\alpha u - u,
\qquad \text{for any } u \in C_\infty(\mathbb{R}^d).
$$

**(d)** Use the previous proposition to deduce that in the very same setup, for any $\alpha > 0$, it holds

$$
\lim_{\beta \to \alpha} \lVert U_\alpha u - U_\beta u\rVert_\infty = 0
\qquad \text{for all } u \in \mathcal{B}_b(\mathbb{R}^d).
$$

</div>

