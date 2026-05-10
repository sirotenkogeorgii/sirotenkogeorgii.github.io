---
layout: default
title: Variational Methods — Lectures 1–8
date: 2025-01-01
excerpt: Lecture notes on variational methods based on John Ball's MIGSAA 2019 course, covering Sobolev spaces, weak convergence, embedding theorems, and the direct method of the calculus of variations.
tags:
  - variational-methods
  - pde
  - functional-analysis
---

# Variational Methods — Lectures 1–8

These notes follow the lecture notes by John Ball (January 2019) for the MIGSAA course on variational methods.

## 1. A User's Guide to Sobolev Spaces

In order to give an unambiguous definition of what is meant by a solution of a system of partial differential equations, appropriate function spaces must be defined. The most important of these spaces for variational methods are the **Sobolev spaces**, based on the classical $L^p$ spaces of functions whose $p$-th powers are integrable.

### 1.1 Review of $L^p$ Spaces

If $x \in \mathbb{R}^n$ we write $x = (x_1, \ldots, x_n)$, where the $x_i$ are the coordinates of $x$ with respect to a fixed orthonormal basis $e_i$ of $\mathbb{R}^n$. Let $\mathcal{L}^n$ denote $n$-dimensional Lebesgue measure; if $E \subset \mathbb{R}^n$ is $\mathcal{L}^n$-measurable we denote its measure by $\mathcal{L}^n(E)$, writing $d\mathcal{L}^n = dx$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^p$ Space)</span></p>

If $E \subset \mathbb{R}^n$ is $\mathcal{L}^n$-measurable and $1 \le p \le \infty$, then $L^p(E)$ is the space of (equivalence classes of) $\mathcal{L}^n$-measurable functions $u : E \to \mathbb{R}$ with $\|u\|_p < \infty$, where

$$\|u\|_p = \left( \int_E |u(x)|^p \, dx \right)^{1/p}, \quad \text{if } 1 \le p < \infty,$$

$$\|u\|_\infty = \operatorname*{ess\,sup}_{x \in E} |u(x)|.$$

Here two functions $u, v$ are equivalent if $u(x) = v(x)$ $\mathcal{L}^n$ almost everywhere (that is, for all $x \in E \setminus N$ where $\mathcal{L}^n(N) = 0$). The essential supremum is defined as

$$\operatorname*{ess\,sup}_{x \in E} |u(x)| \overset{\text{def}}{=} \inf \lbrace \alpha \ge 0 : |u(x)| \le \alpha \text{ for a.e. } x \in E \rbrace.$$

</div>

Most of the time we will consider $L^p(\Omega)$, where $\Omega \subset \mathbb{R}^n$ is open. Endowed with the norm $\| \cdot \|_p$, $L^p(E)$ is a **Banach space** (i.e. a complete normed linear space; *complete* means that each Cauchy sequence converges).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Key Inequalities in $L^p$)</span></p>

**Minkowski's inequality** (the triangle inequality):

$$\|u + v\|_p \le \|u\|_p + \|v\|_p.$$

**Hölder's inequality:**

$$\|uv\|_1 \le \|u\|_p \|v\|_{p'} \quad \text{for all } u \in L^p(\Omega),\, v \in L^{p'}(\Omega),$$

where $\frac{1}{p} + \frac{1}{p'} = 1$. In particular, since

$$\| \, |u|^q \| \le \| \, |u|^q \|_{p/q} \|1\|_{(p/q)'}$$

we have that $L^p(E) \subset L^q(E)$ whenever $1 \le q \le p$ and $\mathcal{L}^n(E) < \infty$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Dual Spaces and Reflexivity)</span></p>

If $1 \le p < \infty$ then the dual space $L^p(E)^\ast$ (the Banach space of all continuous linear mappings from $L^p(E)$ to $\mathbb{R}$) can be identified with $L^{p'}(E)$. More precisely, if $T \in L^p(E)^\ast$ there exists a unique $\varphi = \varphi_T$ in $L^{p'}(E)$ such that

$$\langle T, u \rangle = \int_E u \varphi \, dx \quad \text{for all } u \in L^p(E),$$

and the mapping $T \mapsto \varphi_T$ is an isometric isomorphism of $L^p(E)^\ast$ onto $L^{p'}(E)$ (i.e. it is 1-1, onto and $\|T\|_{L^p(E)^\ast} = \|\varphi_T\|_{L^{p'}(E)}$). From this it follows easily that if $1 < p < \infty$ then $L^p(E)$ is **reflexive**.

If $1 \le p < \infty$ then $L^p(\Omega)$ is **separable** (that is, contains a countable dense subset). But if $\mathcal{L}^n(E) > 0$ then $L^\infty(E)$ is not separable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convergence in $L^p$)</span></p>

Assume $1 \le p < \infty$ and let $u^{(j)} \to u$ in $L^p(\Omega)$. Then there exists a subsequence $u^{(j_k)}$ of $u^{(j)}$ which converges to $u$ a.e. in $\Omega$ (i.e. $u^{(j_k)}(x) \to u^{(j)}(x)$ for all $x \in E \setminus N$, where $\mathcal{L}^n(N) = 0$). More generally, this holds if $u^{(j)} \to u$ in *measure*, i.e. given any $\varepsilon > 0$

$$\lim_{j \to \infty} \mathcal{L}^n(\lbrace x \in \Omega : |u^{(j)}(x) - u(x)| > \varepsilon \rbrace) = 0.$$

</div>

### 1.2 Approximation by Smooth Functions

Let $C^\infty(\Omega)$ be the space of infinitely differentiable functions $\varphi : \Omega \to \mathbb{R}$ and denote by $C_0^\infty(\Omega)$ the subset of $C^\infty(\Omega)$ consisting of those $\varphi : \Omega \to \mathbb{R}$ with compact support in $\Omega$ (i.e. such that $\varphi(x) = 0$ for $x \in \Omega \setminus K$, where $K \subset \Omega$ is compact; the smallest such $K$ is called the **support** $\operatorname{supp} \varphi$ of $\varphi$).

Note that a nonzero $\varphi \in C_0^\infty(\Omega)$ cannot be analytic, since all the Taylor coefficients are zero for $x \notin \operatorname{supp} \varphi$; an example of a nonzero $\varphi \in C_0^\infty(\mathbb{R}^n)$ is given by

$$\varphi(x) = \begin{cases} \exp\left(\frac{1}{|x|^2 - 1}\right) & \text{if } |x| < 1, \\ 0 & \text{if } |x| \ge 1. \end{cases}$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mollifier)</span></p>

Let $\rho \in C_0^\infty(\mathbb{R}^n)$ satisfy $\rho \ge 0$, $\rho(x) = 0$ if $|x| \ge 1$, and $\int_{\mathbb{R}^n} \rho \, dx = 1$. For $\varepsilon > 0$ define

$$\rho_\varepsilon(x) = \varepsilon^{-n} \rho\!\left(\frac{x}{\varepsilon}\right).$$

The function $\rho_\varepsilon$ is called a **mollifier**. It satisfies $\rho_\varepsilon \ge 0$, $\rho_\varepsilon(x) = 0$ if $|x| \ge \varepsilon$, and $\int_{\mathbb{R}^n} \rho_\varepsilon(x) \, dx = 1$, so that $\rho_\varepsilon$ approximates the delta function as $\varepsilon \to 0$.

The **convolution** with $u$ is defined as

$$(\rho_\varepsilon \ast u)(x) := \int_{\mathbb{R}^n} \rho_\varepsilon(x - y) u(y) \, dy.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Mollification in $L^p$)</span></p>

Let $1 \le p < \infty$ and $u \in L^p(\Omega)$. Define $u$ to be zero outside $\Omega$. Then:

1. $\rho_\varepsilon \ast u \in C^\infty(\mathbb{R}^n)$,
2. $\|\rho_\varepsilon \ast u\|_p \le \|u\|_p$,
3. $\lim_{\varepsilon \to 0} \|\rho_\varepsilon \ast u - u\|_p = 0$.

In particular $C^\infty(\Omega)$ is dense in $L^p(\Omega)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2</span><span class="math-callout__name">(Differentiation of Convolutions)</span></p>

Let $1 \le p < \infty$, $h \in C_0^\infty(\mathbb{R}^n)$ and $u \in L^p(\mathbb{R}^n)$. Then $h \ast u$ is continuously differentiable on $\mathbb{R}^n$ and for $i = 1, \ldots, n$

$$\frac{\partial(h \ast u)}{\partial x_i}(x) = \int_{\mathbb{R}^n} \frac{\partial h}{\partial x_i}(x - y) u(y) \, dy.$$

</div>

<details class="accordion" markdown="1">
<summary>Proof of Lemma 2</summary>

By definition $(h \ast u)(x_j) = \int_{\mathbb{R}^n} h(x_j - y) u(y) \, dy$. The integrand vanishes for all $j$ for $y$ outside some bounded set, and is bounded in absolute value by $\mathrm{const.} |u(y)|$. Hence by the dominated convergence theorem $(h \ast u)(x_j) \to (h \ast u)(x)$ and so $h \ast u$ is continuous. For $x \in \Omega$ and $|t| \le 1$:

$$\frac{(h \ast v)(x + te_i) - (h \ast v)(x)}{t} = \int_{\mathbb{R}^n} \left( \frac{h(x + te_i - y) - h(x - y)}{t} \right) v(y) \, dy.$$

Since $h \in C_0^\infty(\mathbb{R}^n)$ the integrand is bounded by $\mathrm{const.} |v(y)|$ and is zero for $y$ outside some bounded set. Hence by the dominated convergence theorem $\partial(h \ast v)/\partial x_i$ exists and is given by the formula above. By the first part of the argument applied to the kernel $\partial h / \partial x_i$ we see that each $\partial(h \ast v)/\partial x_i$ is continuous and so by a standard result $h \ast v$ is continuously differentiable.

</details>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 1</summary>

**(i)** Follows by applying Lemma 2 inductively to $u$ and its partial derivatives.

**(ii)** We write $\rho_\varepsilon(x - y) u(y) = \rho_\varepsilon(x - y)^{1/p'} \rho_\varepsilon(x - y)^{1/p} u(y)$. By Hölder's inequality and $\int_{\mathbb{R}^n} \rho_\varepsilon(z) \, dz = 1$:

$$\left| \int_{\mathbb{R}^n} \rho_\varepsilon(x - y) u(y) \, dy \right| \le \left( \int_{\mathbb{R}^n} \rho_\varepsilon(x - y) |u(y)|^p \, dy \right)^{1/p}.$$

Hence, using Fubini's theorem:

$$\int_\Omega |\rho_\varepsilon \ast u|^p \, dx \le \int_{\mathbb{R}^n} |u(y)|^p \left( \int_\Omega \rho_\varepsilon(x - y) \, dx \right) dy \le \int_\Omega |u(y)|^p \, dy.$$

**(iii)** Given $\tau > 0$ there exists a continuous function $w$ of compact support in $\Omega$ with $\|u - w\|_p < \tau$. Since

$$\int_\Omega |(\rho_\varepsilon \ast w)(x) - w(x)|^p \, dx \le \kappa(\varepsilon)^p \mathcal{L}^n(N_\varepsilon),$$

where $\kappa(\varepsilon) := \sup_{|x - y| < \varepsilon} |w(x) - w(y)|$ and $N_\varepsilon = \lbrace x \in \mathbb{R}^n : \mathrm{dist}(x, \operatorname{supp} w) \le \varepsilon \rbrace$, it follows that $\lim_{\varepsilon \to 0} \|\rho_\varepsilon \ast w - w\|_p = 0$. Since $\|\rho_\varepsilon \ast u - u\|_p \le \|\rho_\varepsilon \ast w - w\|_p + \|\rho_\varepsilon \ast (u - w) - (u - w)\|_p$, it follows from (ii) that $\lim_{\varepsilon \to 0} \|\rho_\varepsilon \ast u - u\|_p \le 2\tau$. Since $\tau$ is arbitrary this completes the proof.

</details>

### 1.3 Weak and Weak$^\ast$ Convergence

Let $X$ be a Banach space with dual space $X^\ast$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak and Weak$^\ast$ Convergence)</span></p>

A sequence $u^{(j)}$ **converges weakly** to $u$ in $X$ (written $u^{(j)} \rightharpoonup u$ in $X$) if

$$\langle T, u^{(j)} \rangle \to \langle T, u \rangle \quad \text{for all } T \in X^\ast.$$

A sequence $T^{(j)}$ **converges weak$^\ast$** to $T$ in $X^\ast$ (written $T^{(j)} \overset{\ast}{\rightharpoonup} T$) if

$$\langle T^{(j)}, u \rangle \to \langle T, u \rangle \quad \text{for all } u \in X.$$

</div>

Applying these definitions to $X = L^p(E)$, and using the characterization of $L^p(E)^\ast$, we find that if $1 \le p < \infty$ then $u^{(j)} \rightharpoonup u$ in $X = L^p(E)$ if and only if

$$\int_E u^{(j)} \varphi \, dx \to \int_E u \varphi \, dx \quad \text{for all } \varphi \in L^{p'}(E),$$

and $u^{(j)} \overset{\ast}{\rightharpoonup} u$ in $L^\infty(E)$ if and only if

$$\int_E u^{(j)} \varphi \, dx \to \int_E u \varphi \, dx \quad \text{for all } \varphi \in L^1(E).$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.1</span><span class="math-callout__name">(Rademacher Functions)</span></p>

Let $\Omega = (0, 1)$, $0 < \lambda < 1$, $a, b \in \mathbb{R}$ and define $\theta : \mathbb{R} \to \mathbb{R}$ by

$$\theta(x) = \begin{cases} a, & 0 < x \le \lambda \\ b, & \lambda < x \le 1 \end{cases}$$

extended to the whole of $\mathbb{R}$ as a function of period 1. Define $\theta^{(j)}(x) = \theta(jx)$, $j = 1, 2, \ldots$ For large $j$, $\theta^{(j)}$ oscillates fast between the values $a$ and $b$, taking these values with relative frequency $\lambda$ to $1 - \lambda$. Let $c = \lambda a + (1 - \lambda)b$. Then $\theta^{(j)} \overset{\ast}{\rightharpoonup} c$ in $L^\infty(0,1)$ as $j \to \infty$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Example 1.1</summary>

We first calculate $\lim_{j \to \infty} \int_r^s \theta^{(j)} \, dx$ for $0 \le r < s \le 1$. We have

$$\int_r^s \theta^{(j)}(x) \, dx = \int_r^s \theta(jx) \, dx = \frac{1}{j} \int_{jr}^{js} \theta(\tau) \, d\tau.$$

The interval $(jr, js)$ contains $N_j$ integers, where $|N_j - (js - jr)| \le 1$. Since $\theta$ is 1-periodic and $\int_0^1 \theta(\tau) \, d\tau = c$ it follows that

$$\int_{jr}^{js} \theta(\tau) \, d\tau = (js - jr)c + \epsilon_j,$$

where $|\epsilon_j| \le \mathrm{constant}$. Combining, $\lim_{j \to \infty} \int_r^s \theta^{(j)} \, dx = \int_r^s c \, dx$. This holds for any step function $\varphi$. But step functions are dense in $L^1(0,1)$; given any $\varphi \in L^1(0,1)$ there exists a sequence $\varphi^{(k)}$ of step functions converging strongly to $\varphi$ in $L^1(0,1)$. Hence

$$\left| \int_0^1 \theta^{(j)} \varphi \, dx - \int_0^1 c\varphi \, dx \right| \le \left| \int_0^1 (\theta^{(j)} - c)\varphi^{(k)} \, dx \right| + K \|\varphi^{(k)} - \varphi\|_1,$$

where $K$ is a constant. Letting $j \to \infty$ and then $k \to \infty$ we deduce that $\lim_{j \to \infty} \int_0^1 \theta^{(j)} \varphi \, dx = \int_0^1 c\varphi \, dx$ for all $\varphi \in L^1(0,1)$, and thus $\theta^{(j)} \overset{\ast}{\rightharpoonup} c$ in $L^\infty(0,1)$.

</details>

A key reason why weak convergence is important for variational methods is that suitably bounded sequences have weakly (or weak$^\ast$) convergent subsequences.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4</span><span class="math-callout__name">(Weak$^\ast$ Compactness)</span></p>

Let $X$ be a separable Banach space, and let $T^{(j)}$ be a bounded sequence in $X^\ast$, i.e. $\sup_j \|T^{(j)}\|_{X^\ast} = M < \infty$. Then there exists a subsequence $T^{(j_k)}$ of $T^{(j)}$ converging weak$^\ast$ to some $T$ in $X^\ast$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 4</summary>

Let $\lbrace \psi_i \rbrace_{i=1}^\infty$ be a countable dense subset of $X$. Since $|\langle T^{(j)}, \psi_1 \rangle| \le M \|\psi_1\|$, the sequence $\langle T^{(j)}, \psi_1 \rangle$ of real numbers is bounded. Hence there exists a subsequence $T^{(n_1(j))}$ of $T^{(j)}$ such that $\lim_{j \to \infty} \langle T^{(n_1(j))}, \psi_1 \rangle$ exists. Similarly, the sequence $\langle T^{(n_1(j))}, \psi_2 \rangle$ is bounded, and so there exists a subsequence $T^{(n_2(j))}$ of $T^{(n_1(j))}$ such that $\lim_{j \to \infty} \langle T^{(n_2(j))}, \psi_2 \rangle$ exists. Proceeding in this way we obtain for each $i$ a subsequence $T^{(n_i(j))}$ of $T^{(n_{i-1}(j))}$ such that $\lim_{j \to \infty} \langle T^{(n_i(j))}, \psi_i \rangle$ exists. Consider the "diagonal sequence" $T^{(n_j(j))}$. Since $\lbrace T^{(n_j(j))} \rbrace_{j=i}^\infty$ is a subsequence of $\lbrace T^{(n_i(j))} \rbrace_{j=1}^\infty$ it follows that $\lim_{j \to \infty} \langle T^{(n_i(j))}, \psi_i \rangle$ exists for each $i$.

Now let $\psi \in X$ be arbitrary. Given $\varepsilon > 0$ there exists $I$ with $\|\psi - \psi_I\| \le \varepsilon / (2M)$. Then

$$|\langle T^{(n_j(j))}, \psi \rangle - \langle T^{(n_k(k))}, \psi \rangle| \le |\langle T^{(n_j(j))}, \psi_I \rangle - \langle T^{(n_k(k))}, \psi_I \rangle| + \varepsilon$$

and hence $\langle T^{(n_k(k))}, \psi \rangle$ is a Cauchy sequence, so that

$$T(\psi) \overset{\text{def}}{=} \lim_{k \to \infty} \langle T^{(n_k(k))}, \psi \rangle$$

exists. Clearly $T$ is linear in $\psi$, and since $|T(\psi)| \le M \|\psi\|$ it follows that $T \in X^\ast$. Thus $T^{(j_k)} \overset{\ast}{\rightharpoonup} T$ in $X^\ast$ with $j_k = n_k(k)$.

</details>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5</span><span class="math-callout__name">(Weak Compactness in Reflexive Spaces)</span></p>

A bounded sequence in a reflexive Banach space $X$ has a weakly convergent subsequence.

</div>

Thus a bounded sequence in $L^p(E)$, $1 < p < \infty$, has a weakly convergent subsequence, and a bounded sequence in $L^\infty(E)$ has a weak$^\ast$ convergent subsequence. A bounded sequence in $L^1(E)$ need not have a weakly convergent subsequence (consider, for example, $E = (0,1)$, $u^{(j)} = j\chi_{(0, 1/j)}$), and an extra condition is needed to ensure this.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">(de la Vallée Poussin)</span></p>

A sequence $u^{(j)}$ in $L^1(E)$ has a weakly convergent subsequence if

$$\sup_j \int_E \Phi(|u^{(j)}|) \, dx < \infty$$

for some continuous $\Phi : [0, \infty) \to [0, \infty)$ with $\lim_{t \to \infty} \Phi(t)/t = \infty$.

</div>

### 1.4 The Multi-Index Notation for Derivatives

It is convenient to have a compact notation for expressing mixed partial derivatives of functions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multi-Index)</span></p>

A **multi-index** $\alpha$ is an $n$-tuple $\alpha = (\alpha_1, \ldots, \alpha_n)$ of nonnegative integers $\alpha_i$, and we write $|\alpha| = \alpha_1 + \cdots + \alpha_n$.

Let $\Omega \subset \mathbb{R}^n$ be open and $u : \Omega \to \mathbb{R}$ be smooth. Then we define

$$D^\alpha u = \left(\frac{\partial}{\partial x_1}\right)^{\alpha_1} \cdots \left(\frac{\partial}{\partial x_n}\right)^{\alpha_n} u = \frac{\partial^{|\alpha|} u}{\partial x_1^{\alpha_1} \cdots \partial x_n^{\alpha_n}}.$$

For example, if $n = 3$ and $\beta = (2, 1, 0)$, then $D^\beta u = \frac{\partial^3 u}{\partial x_1^2 \partial x_2}$.

Note that if $\alpha, \beta$ are multi-indices then so is $\alpha + \beta = (\alpha_1 + \beta_1, \ldots, \alpha_n + \beta_n)$, and $D^{\alpha + \beta} u = D^\alpha D^\beta u = D^\beta D^\alpha u$.

</div>

### 1.5 Weak Derivatives

Let $\Omega \subset \mathbb{R}^n$ be open with boundary $\partial\Omega$, and let $v \in C^1(\Omega)$, $\varphi \in C_0^\infty(\Omega)$. Then for any $j = 1, \ldots, n$, integrating over $\Omega$ and using the divergence theorem we have

$$\int_\Omega v \frac{\partial \varphi}{\partial x_j} \, dx = -\int_\Omega \frac{\partial v}{\partial x_j} \varphi \, dx.$$

This can be thought of as the formula for **integration by parts** in $n$ dimensions. More generally, if $\alpha = (\alpha_1, \ldots, \alpha_n)$ is a multi-index and $u \in C^{|\alpha|}(\Omega)$, applying the above $\alpha_j$ times for each $j$ we deduce

$$\int_\Omega u D^\alpha \varphi \, dx = (-1)^{|\alpha|} \int_\Omega D^\alpha u \cdot \varphi \, dx.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak Derivative)</span></p>

Define $L^1_{\mathrm{loc}}(\Omega) = \lbrace u : \Omega \to \mathbb{R} : u|_E \in L^1(E) \text{ for all bounded open } E \text{ with } \bar{E} \subset \Omega \rbrace$.

Let $u \in L^1_{\mathrm{loc}}(\Omega)$ and $\alpha$ be a multi-index. A function $v \in L^1_{\mathrm{loc}}(\Omega)$ is said to be an **$\alpha^{\text{th}}$ weak derivative** of $u$ if

$$\int_\Omega u D^\alpha \varphi \, dx = (-1)^{|\alpha|} \int_\Omega v \varphi \, dx \quad \text{for all } \varphi \in C_0^\infty(\Omega),$$

and we write $v = D^\alpha u$.

</div>

If $v_1$ and $v_2$ are two $\alpha^{\text{th}}$ weak derivatives, their difference $w = v_1 - v_2$ satisfies $\int_\Omega w \varphi \, dx = 0$ for all $\varphi \in C_0^\infty(\Omega)$, and so by the following lemma $v_1 = v_2$. Hence weak derivatives are unique.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7</span><span class="math-callout__name">(Fundamental Lemma of the Calculus of Variations)</span></p>

Let $w \in L^1_{\mathrm{loc}}(\Omega)$ satisfy

$$\int_\Omega w \varphi \, dx = 0 \quad \text{for all } \varphi \in C_0^\infty(\Omega).$$

Then $w = 0$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Lemma 7</summary>

Let $\rho_\varepsilon$ be a mollifier. Let $E$ be bounded and open with $\bar{E} \subset \Omega$. If $\varepsilon < \mathrm{dist}(E, \partial\Omega)$ then for each $x \in E$ the function $\varphi_{\varepsilon,x}$ defined by $\varphi_{\varepsilon,x}(y) = \rho_\varepsilon(x - y)$ belongs to $C_0^\infty(\Omega)$. Hence

$$(\rho_\varepsilon \ast w)(x) = \int_\Omega \rho_\varepsilon(x - y) w(y) \, dy = 0$$

for all $x \in E$. But $\rho_\varepsilon \ast w \to w$ in $L^1(E)$ as $\varepsilon \to 0$, and so $w = 0$ a.e. in $E$. Since $E$ is arbitrary the result follows.

</details>

### 1.6 The Sobolev Space $W^{m,p}(\Omega)$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sobolev Space)</span></p>

Let $m$ be a non-negative integer and let $1 \le p \le \infty$. The **Sobolev space** $W^{m,p}(\Omega)$ is the linear space of functions $u \in L^p(\Omega)$ such that for each $\alpha$, $0 \le |\alpha| \le m$, the weak derivative $D^\alpha u$ exists and belongs to $L^p(\Omega)$. We norm $W^{m,p}(\Omega)$ by

$$\|u\|_{m,p} = \begin{cases} \left( \sum_{0 \le |\alpha| \le m} \|D^\alpha u\|_p^p \right)^{1/p} & \text{if } 1 \le p < \infty \\ \max_{0 \le |\alpha| \le m} \|D^\alpha u\|_\infty & \text{if } p = \infty. \end{cases}$$

If $p = 2$ an alternative notation is often used: $H^m(\Omega) = W^{m,2}(\Omega)$.

Note that $W^{0,p}(\Omega) = L^p(\Omega)$, while

$$W^{1,p}(\Omega) = \left\lbrace u \in L^p(\Omega) : \frac{\partial u}{\partial x_j} \in L^p(\Omega) \quad \text{for } i = 1, \ldots, n \right\rbrace$$

with norm

$$\|u\|_{1,p} = \left( \int_\Omega |u|^p \, dx + \sum_{i=1}^n \int_\Omega \left| \frac{\partial u}{\partial x_i} \right|^p dx \right)^{1/p}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8</span><span class="math-callout__name">($W^{m,p}(\Omega)$ is a Banach Space)</span></p>

$W^{m,p}(\Omega)$ is a Banach space.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 8</summary>

$W^{m,p}(\Omega)$ is clearly a normed linear space, and we have to show that it is complete. Let $u^{(j)}$ be a Cauchy sequence in $W^{m,p}(\Omega)$. Then $u^{(j)}$ is a Cauchy sequence in $L^p(\Omega)$, and since $L^p(\Omega)$ is complete $u^{(j)} \to u$ in $L^p(\Omega)$ as $j \to \infty$ for some $u$. Similarly, if $0 < |\alpha| \le m$ then $D^\alpha u^{(j)}$ is a Cauchy sequence in $L^p(\Omega)$ and so $D^\alpha u^{(j)} \to u_\alpha$ in $L^p(\Omega)$. But by the definition of weak derivatives

$$\int_\Omega u^{(j)} D^\alpha \varphi \, dx = (-1)^{|\alpha|} \int_\Omega D^\alpha u^{(j)} \cdot \varphi \, dx$$

for all $\varphi \in C_0^\infty(\Omega)$. Passing to the limit $j \to \infty$ using Hölder's inequality we obtain

$$\int_\Omega u D^\alpha \varphi \, dx = (-1)^{|\alpha|} \int_\Omega u_\alpha \varphi \, dx,$$

for all $\varphi \in C_0^\infty(\Omega)$ so that $u_\alpha = D^\alpha u$. Hence $u^{(j)} \to u$ in $W^{m,p}(\Omega)$, so that $W^{m,p}(\Omega)$ is complete.

</details>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9</span><span class="math-callout__name">(Separability and Reflexivity of $W^{m,p}$)</span></p>

$W^{m,p}(\Omega)$ is separable if $1 \le p < \infty$ and is reflexive if $1 < p < \infty$.

</div>

### 1.7 Examples

#### 1.7.1 Smooth Functions

Let $u \in C^m(\Omega)$ with $\|u\|_{m,p} < \infty$. Then by the integration by parts formula the weak derivatives $D^\alpha u$ for $0 \le |\alpha| \le m$ equal the usual ones, and hence $u \in W^{m,p}(\Omega)$. In particular, if $\Omega$ is bounded and $u \in C^\infty(\mathbb{R}^n)$ then $u|_\Omega \in W^{m,p}(\Omega)$ for all $m, p$.

#### 1.7.2 Piecewise Affine Functions

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Piecewise Affine Function)</span></p>

Let $n = 1$, $\Omega = (0, 1)$, and let $u$ be defined by

$$u(x) = \begin{cases} x & \text{if } 0 < x < \tfrac{1}{2} \\ 1 - x & \text{if } \tfrac{1}{2} < x < 1. \end{cases}$$

Then $u \in W^{1,\infty}(0,1)$ (and hence $u \in W^{1,p}(0,1)$ for $1 \le p \le \infty$). The weak derivative is

$$\frac{du}{dx}(x) = \begin{cases} 1 & \text{if } 0 < x < \tfrac{1}{2} \\ -1 & \text{if } \tfrac{1}{2} < x < 1. \end{cases}$$

More generally, if $u$ is a piecewise affine function on $(0,1)$ (i.e. $u$ is continuous on $(0,1)$ and affine on each interval $(a_i, a_{i+1})$, where $0 = a_1 < a_2 < \ldots < a_n = 1$) then $u \in W^{1,\infty}(0,1)$.

</div>

#### 1.7.3 The Heaviside Function

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Heaviside Function)</span></p>

The Heaviside function $H$ is defined by

$$H(x) = \begin{cases} 1 & x \ge 0 \\ 0 & x < 0. \end{cases}$$

Clearly $H \in L^\infty(-1,1)$. We ask whether $H \in W^{1,p}(-1,1)$. Since the derivative $dH/dx(x) = 0$ for $x \in (-1,0) \cup (0,1)$, it is tempting to conclude that $dH/dx \in L^\infty(-1,1)$, so that $H \in W^{1,\infty}(-1,1)$. But this is **false**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10</span></p>

$H \notin W^{1,p}(-1,1)$ for any $p$, $1 \le p \le \infty$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Proposition 10</summary>

Suppose for contradiction that $H \in W^{1,1}(-1,1)$. Let $dH/dx \in L^1(-1,1)$ denote the weak derivative of $H$. Then, since $H$ is smooth in $(-1,0) \cup (0,1)$, $dH/dx = 0$ a.e. in $(-1,0) \cup (0,1)$ and so $dH/dx = 0$ a.e. in $(-1,1)$. But by the definition of weak derivative

$$\int_{-1}^1 H \frac{d\varphi}{dx} \, dx = -\int_{-1}^1 \frac{dH}{dx} \varphi \, dx,$$

so that $\int_{-1}^1 H \frac{d\varphi}{dx} \, dx = \int_0^1 \frac{d\varphi}{dx} \, dx = -\varphi(0) = 0$ for all $\varphi \in C_0^\infty(-1,1)$, a contradiction.

</details>

#### 1.7.4 The Function $\ln|x|$ on $\mathbb{R}^n$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\ln|x|$ on the Unit Ball)</span></p>

Let $n > 1$, $B = \lbrace x \in \mathbb{R}^n : |x| < 1 \rbrace$. For $x \ne 0$ define $u(x) = \ln r$, $r = |x|$. We show that $u \in W^{1,p}(B)$ if and only if $1 \le p < n$.

**Formal calculation.** For $r > 0$, $u$ is smooth and $\frac{\partial u}{\partial x_i} = \frac{1}{r} \frac{\partial r}{\partial x_i} = \frac{x_i}{r^2}$. Hence $|\nabla u|^2 = \frac{1}{r^2}$ and so

$$\int_B (|u|^p + |\nabla u|^p) \, dx = \omega_{n-1} \int_0^1 r^{n-1}(|\log r|^p + r^{-p}) \, dr,$$

where $\omega_{n-1} = \mathcal{H}^{n-1}(S^{n-1})$, and this is finite if and only if $1 \le p < n$.

The weak derivatives are $\frac{\partial u}{\partial x_i} = \frac{x_i}{r^2}$, which can be verified by showing that $\int_B u \frac{\partial \varphi}{\partial x_i} \, dx = -\int_B \frac{x_i}{r^2} \varphi \, dx$ for all $\varphi \in C_0^\infty(B)$, using integration by parts on $B \setminus B_\varepsilon$ (where $B_\varepsilon = B(0,\varepsilon)$) and passing to the limit $\varepsilon \to 0$.

</div>

### 1.8 Approximation by Smooth Functions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mollification in Sobolev Spaces)</span></p>

Let $u \in W^{m,p}(\Omega)$. Let $E \subset \Omega$ be open with $\varepsilon_0 := \mathrm{dist}(E, \partial\Omega) > 0$. Let $\rho_\varepsilon$ be a mollifier. Then if $0 < \varepsilon \le \varepsilon_0$ the mollified function $(\rho_\varepsilon \ast u)(x) = \int_\Omega \rho_\varepsilon(x - y) u(y) \, dy$ is well-defined for all $x \in E$. If $|\alpha| \le m$ then for $x \in E$

$$D^\alpha(\rho_\varepsilon \ast u)(x) = (\rho_\varepsilon \ast D^\alpha u)(x),$$

i.e. **the derivatives of the mollified function are the mollified derivatives**. Applying Theorem 1 we deduce that if $1 \le p < \infty$ then $\rho_\varepsilon \ast u \to u$ in $W^{m,p}(E)$ as $\varepsilon \to 0$.

</div>

Because of the restriction that $\mathrm{dist}(E, \partial\Omega) > 0$, this does not directly provide an approximation of $u$ in $W^{m,p}(\Omega)$ by functions in $C^\infty(\Omega)$. However, by a more careful argument using a partition of unity one can prove:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11</span><span class="math-callout__name">(Meyers & Serrin)</span></p>

Let $1 \le p < \infty$. Then $C^\infty(\Omega)$ is dense in $W^{m,p}(\Omega)$.

</div>

Can any $u \in W^{m,p}(\Omega)$ be approximated by functions in $C^\infty(\bar{\Omega})$? In general the answer is no.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2</span><span class="math-callout__name">(Failure of Approximation across Boundary)</span></p>

Let $\Omega = (-1,0) \cup (0,1)$, $u(x) = H(x)$. Then $u \in C^\infty(\Omega)$, so that $u \in W^{m,p}(\Omega)$ for any $m, p$. Suppose that there were a sequence $u^{(j)} \in C^1(\mathbb{R})$ with $u^{(j)} \to u$ in $W^{1,p}(\Omega)$. Then we may assume that $u^{(j)} \to u$ a.e. in $\Omega$. Choosing $x_- \in (-1,0)$, $x_+ \in (0,1)$ with $u^{(j)}(x_-) \to 0$, $u^{(j)}(x_+) \to 1$ we have that $u^{(j)}(x_+) - u^{(j)}(x_-) \to 1$. But

$$\lim_{j \to \infty} (u^{(j)}(x_+) - u^{(j)}(x_-)) = \lim_{j \to \infty} \int_{x_-}^{x_+} \frac{du^{(j)}}{dx} \, dx = 0,$$

a contradiction.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lipschitz Boundary)</span></p>

An open set $\Omega \subset \mathbb{R}^n$ has a **$C^m$ (respectively Lipschitz) boundary** if given any $\bar{x} \in \partial\Omega$ there exist $r > 0$ and a $C^m$ (respectively Lipschitz) function $a : \mathbb{R}^{n-1} \to \mathbb{R}$ such that, in a suitable Cartesian coordinate system,

$$\Omega \cap B(\bar{x}, r) = \lbrace x \in \mathbb{R}^n : x_n > a(x_1, \ldots, x_{n-1}) \rbrace \cap B(\bar{x}, r),$$

so that the boundary is locally the graph of a $C^m$ (resp. Lipschitz) function.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12</span><span class="math-callout__name">(Density with $C^0$ Boundary)</span></p>

Let $\Omega$ have $C^0$ boundary, and let $1 \le p < \infty$. Then the set of restrictions to $\Omega$ of functions in $C_0^\infty(\mathbb{R}^n)$ is dense in $W^{m,p}(\Omega)$. In particular, $C^\infty(\bar{\Omega})$ is dense in $W^{m,p}(\Omega)$.

</div>

### 1.9 Boundary Values

Let $\Omega \subset \mathbb{R}^n$ have Lipschitz boundary. How can we define the boundary values of a function $u \in W^{1,p}(\Omega)$? This is not a trivial matter even if $\partial\Omega$ is smooth, since (a) $u$ is in principle defined only in $\Omega$, (b) even if $u$ could be extended to a function $\tilde{u} \in W^{1,p}(\mathbb{R}^n)$ the values of $\tilde{u}$ on $\partial\Omega$ appear to have no meaning since $\mathcal{L}^n(\partial\Omega) = 0$ and $\tilde{u}$ may be altered at will on sets of $\mathcal{L}^n$ measure zero.

If $\Omega$ has Lipschitz boundary we can define $L^p(\partial\Omega)$ as the space of (equivalence classes of) $\mathcal{H}^{n-1}$ measurable functions $u : \partial\Omega \to \mathbb{R}$ such that $\|u\|_{L^p(\partial\Omega)} < \infty$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(Trace Inequality)</span></p>

Let $\Omega \subset \mathbb{R}^n$ be bounded and open with Lipschitz boundary, and let $1 \le p < \infty$. Then there exists a constant $c > 0$ such that

$$\int_{\partial\Omega} |u|^p \, d\mathcal{H}^{n-1} \le c \|u\|_{1,p}^p$$

for all $u \in C^1(\bar{\Omega})$.

</div>

If $u \in W^{1,p}(\Omega)$ there exists a sequence $u^{(j)} \in C^1(\bar{\Omega})$ with $u^{(j)} \to u$ in $W^{1,p}(\Omega)$. Hence $u^{(j)}$ is a Cauchy sequence in $W^{1,p}(\Omega)$, and by the theorem is also a Cauchy sequence in $L^p(\partial\Omega)$. Hence $u^{(j)}|_{\partial\Omega} \to \operatorname{tr} u$ in $L^p(\partial\Omega)$ for some function $\operatorname{tr} u$, the **trace** of $u$ on $\partial\Omega$. The mapping $\operatorname{tr} : W^{1,p}(\Omega) \to L^p(\partial\Omega)$ is a bounded linear operator.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($W_0^{m,p}(\Omega)$)</span></p>

For $1 \le p < \infty$ denote by $W_0^{m,p}(\Omega)$ the closure of $C_0^\infty(\Omega)$ in $W^{m,p}(\Omega)$. If $p = \infty$ we define $W_0^{m,\infty}(\Omega)$ to be the set of $v \in W^{m,\infty}(\Omega)$ that are the a.e. limit of a sequence $\varphi^{(j)} \in C_0^\infty(\Omega)$ that is bounded in $W^{m,\infty}(\Omega)$. $W_0^{m,p}(\Omega)$ is a closed linear subspace of $W^{m,p}(\Omega)$, and hence is a Banach space with the same norm. We write $H_0^m(\Omega) = W_0^{m,2}(\Omega)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14</span><span class="math-callout__name">(Characterization of $W_0^{m,p}$)</span></p>

Let $\Omega \subset \mathbb{R}^n$ be open with Lipschitz boundary. Then if $1 \le p \le \infty$

$$W_0^{m,p}(\Omega) = \lbrace u \in W^{m,p}(\Omega) : \operatorname{tr} D^\alpha u = 0 \text{ if } |\alpha| < m \rbrace.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 15</span></p>

If $1 < p < \infty$ then $W^{m,p}(\mathbb{R}^n) = W_0^{m,p}(\mathbb{R}^n)$.

</div>

### 1.10 Lipschitz Mappings and $W^{1,\infty}$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 16</span></p>

A mapping $u \in W^{1,\infty}_{\mathrm{loc}}(\Omega; \mathbb{R}^m)$ if and only if $u$ has a representative that is locally Lipschitz.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 17</span></p>

Let $\Omega \subset \mathbb{R}^n$ be bounded and open with Lipschitz boundary. Then $u \in W^{1,\infty}(\Omega; \mathbb{R}^m)$ if and only if $u$ has a representative that is Lipschitz on $\Omega$.

</div>

### 1.11 Embedding Theorems

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3</span><span class="math-callout__name">(1D Sobolev Embedding)</span></p>

Let $n = 1$, $-\infty < a < b < \infty$. Then $W^{1,1}(a,b)$ is continuously embedded in $C([a,b])$, i.e. each equivalence class $v$ of functions in $W^{1,1}(a,b)$ has a representative $\tau v \in C([a,b])$ and there is a constant $K > 0$ such that $\|\tau v\|_{C([a,b])} \le K \|v\|_{1,1}$.

Note that the argument also shows that the continuous representative of $v$ satisfies the fundamental theorem of calculus: $v(y) = v(x) + \int_x^y v'(t) \, dt$ for all $x, y \in [a,b]$, so that $v$ is absolutely continuous.

For $p > 1$, the embedding $W^{1,p}(a,b) \to C([a,b])$ is **compact** (bounded sequences in $W^{1,1}(a,b)$ are relatively compact in $C([a,b])$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.4</span><span class="math-callout__name">(Sobolev Embedding in $\mathbb{R}^3$)</span></p>

Let $n = 3$, $m = 1$. Then $H^1(\Omega) = W^{1,2}(\Omega) \subset L^6(\Omega)$ and the embedding $W^{1,2}(\Omega) \subset L^{6-\varepsilon}(\Omega)$ is compact. $W^{1,3}(\Omega) \subset L^q(\Omega)$ for $1 \le q < \infty$ but $W^{1,3}(\Omega) \not\subset L^\infty(\Omega)$. $W^{1,p}(\Omega) \subset C^0(\bar{\Omega})$ compact if $p > 3$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 18</span><span class="math-callout__name">(Sobolev Embedding)</span></p>

Let $\Omega \subset \mathbb{R}^n$ be bounded, open with Lipschitz boundary, and let $1 \le p \le \infty$.

* If $mp < n$ then $W^{m,p}(\Omega) \subset L^q(\Omega)$, $\frac{1}{q} \ge \frac{1}{p} - \frac{m}{n}$,
* if $mp = n$ then $W^{m,p}(\Omega) \subset L^q(\Omega)$, $1 \le q < \infty$,
  * (if $p = 1$ and $m = n$ then in addition $W^{n,1}(\Omega) \subset L^\infty(\Omega)$),
* if $mp > n$ then $W^{m,p}(\Omega) \subset C^0(\bar{\Omega})$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19</span><span class="math-callout__name">(Rellich–Kondrachoff)</span></p>

The embedding $W^{m,p}(\Omega) \subset L^q(\Omega)$ is **compact** if $mp < n$, $\frac{1}{q} > \frac{1}{p} - \frac{m}{n}$ or if $mp = n$, $1 \le q < \infty$. The embedding $W^{m,p}(\Omega) \subset C^0(\bar{\Omega})$ is compact if $mp > n$.

</div>

As an application of the embedding theorems:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 20</span><span class="math-callout__name">(Generalized Poincaré Inequality)</span></p>

Let $\Omega \subset \mathbb{R}^n$ be a bounded domain (i.e. open and connected) with Lipschitz boundary, and let $1 < p < \infty$. Then there exists a constant $C = C(\Omega, p)$ such that

$$\int_\Omega |u|^p \, dx \le C \left( \left| \int_\Omega u \, dx \right|^p + \int_\Omega |\nabla u|^p \, dx \right)$$

for all $u \in W^{1,p}(\Omega)$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 20</summary>

Suppose not. Then there exist $u^{(j)} \in W^{1,p}(\Omega)$ with

$$1 = \int_\Omega |u^{(j)}|^p \, dx > j \left( \left| \int_\Omega u^{(j)} \, dx \right|^p + \int_\Omega |\nabla u^{(j)}|^p \, dx \right).$$

Hence $u^{(j)}$ is bounded in $W^{1,p}(\Omega)$ and we can suppose that $u^{(j)} \rightharpoonup u$ in $W^{1,p}(\Omega)$. By the compactness of the embedding $W^{1,p}(\Omega) \subset L^p(\Omega)$ we have $\int_\Omega |u|^p \, dx = 1$. We now use the inequality

$$|\mathbf{a}|^p \ge |\mathbf{b}|^p + p|\mathbf{b}|^{p-2}\mathbf{b} \cdot (\mathbf{a} - \mathbf{b}) \quad \text{for } \mathbf{a}, \mathbf{b} \in \mathbb{R}^n.$$

Thus $\int_\Omega |\nabla u^{(j)}|^p \, dx \ge \int_\Omega |\nabla u|^p \, dx + p \int_\Omega |\nabla u|^{p-2} \nabla u \cdot (\nabla u^{(j)} - \nabla u) \, dx$. Hence

$$0 = \lim_{j \to \infty} \left( \left| \int_\Omega u^{(j)} \, dx \right|^p + \int_\Omega |\nabla u^{(j)}|^p \, dx \right) \ge \left| \int_\Omega u \, dx \right|^p + \int_\Omega |\nabla u|^p \, dx$$

(since $\nabla u^{(j)} \rightharpoonup \nabla u$ in $(L^p)^n$ and $|\nabla u|^{p-2}\nabla u \in (L^{p'})^n$). Hence $\nabla u = 0$, so $u$ is constant and thus $u = 0$. Contradiction.

</details>

## 2. The One-Dimensional Calculus of Variations

Consider for $-\infty < a < b < \infty$ the integral functional

$$I(u) = \int_a^b f(x, u(x), u_x(x)) \, dx$$

for $f$ continuous and bounded below. Here $u \in W^{1,1}(a,b) = AC[a,b]$, and satisfies the boundary conditions:

$$\text{either } u(a) = \alpha,\, u(b) = \beta, \qquad \text{or } u(a) = \alpha.$$

(Note that for such $u$ we may have $I(u) = +\infty$.)

### 2.1 Existence of Minimizers

We begin with some counterexamples showing that minimizers need not exist.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.1</span><span class="math-callout__name">(Bolza)</span></p>

$$I(u) = \int_0^1 [(u_x^2 - 1)^2 + u^2] \, dx, \quad u(0) = u(1) = 0.$$

$I$ does not attain an absolute minimum in $W_0^{1,1}(0,1)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21</span><span class="math-callout__name">(Non-Attainment for Bolza Problem)</span></p>

$I$ does not attain an absolute minimum in $W_0^{1,1}(0,1)$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 21</summary>

Let $u^{(j)}$ be a zigzag function so that $u_x^{(j)}(x) = \pm 1$ a.e. and $|u^{(j)}(x)| \le \frac{1}{2j}$. Then

$$I(u^{(j)}) = \int_0^1 u^{(j)2} \, dx \le \frac{1}{4j^2} \to 0 \text{ as } j \to \infty.$$

Hence $\inf_{W_0^{1,1}} I = 0$. But $I(u) = 0$ implies $u = 0$, hence $u_x = 0$ and $I(u) = 1$. Contradiction.

</details>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On the Bolza Problem)</span></p>

1. The same argument works for the boundary conditions $u(0) = 0$, $u(1)$ free.
2. We can think of there being a minimizer which is a "generalized curve" in the sense of L.C. Young, with track $u = 0$ and derivative given by the probability measure $\nu = \frac{1}{2}(\delta_{-1} + \delta_1)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.2</span></p>

$$I(u) = \int_0^1 x^2 u_x^2 \, dx, \quad u(0) = 0, \, u(1) = 1.$$

The minimum is not attained. Taking a minimizing sequence $u^{(j)}$ with slope $j$ on $(0, 1/j)$ and $u^{(j)} = 1$ on $(1/j, 1)$, we get $I(u^{(j)}) = \int_0^{1/j} x^2 j^2 \, dx = \frac{1}{3j} \to 0$. But there is no $u \in W^{1,1}(0,1)$ with $u(0) = 0$, $u(1) = 1$ and $I(u) = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.3</span></p>

$$I(u) = \int_0^1 (|u_x| + (u - 1)^2) \, dx, \quad u(0) = 0, \, u(1) = 1.$$

Then $I(u) \ge \left| \int_0^1 u_x \, dx \right| + \int_0^1 (u-1)^2 \, dx = 1 + \int_0^1 (u-1)^2 \, dx$. Taking $u^{(j)}$ as in Example 2.2, $I(u^{(j)}) = \int_0^{1/j} [j + (jx - 1)^2] \, dx \to 1$ as $j \to \infty$. Thus $\inf I = 1$ and is not attained.

</div>

In Example 2.1, $f(x,u,\cdot)$ is not convex. In Examples 2.2 and 2.3, $f(x,u,p)$ does not have superlinear growth in $p$. To prove the existence of minimizers we need an appropriate lower semicontinuity theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 22</span><span class="math-callout__name">(Lower Semicontinuity)</span></p>

Let $\Omega \subset \mathbb{R}^n$ be bounded open, and let $f : \Omega \times \mathbb{R}^s \times \mathbb{R}^\sigma \to [0, \infty]$ satisfy:

1. $f(\cdot, z, v) : \Omega \to [0, \infty]$ is measurable for every $z \in \mathbb{R}^s$, $v \in \mathbb{R}^\sigma$,
2. $f(x, \cdot, \cdot) : \mathbb{R}^s \times \mathbb{R}^\sigma \to [0, \infty]$ is continuous for a.e. $x \in \Omega$,
3. $f(x, z, \cdot) : \mathbb{R}^\sigma \to [0, \infty]$ is convex for a.e. $x \in \Omega$ and all $z \in \mathbb{R}^s$.

Let $z^{(j)}, z : \Omega \to \mathbb{R}^s$ be measurable mappings such that $z^{(j)} \to z$ a.e., and let $v^{(j)} \rightharpoonup v$ in $L^1(\Omega; \mathbb{R}^\sigma)$. Then

$$\int_\Omega f(x, z(x), v(x)) \, dx \le \liminf_{j \to \infty} \int_\Omega f(x, z^{(j)}(x), v^{(j)}(x)) \, dx.$$

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 22</summary>

We may assume that $\liminf_{j \to \infty} \int_\Omega f(x, z^{(j)}(x), v^{(j)}(x)) \, dx = a < \infty$. We first claim that $h^{(j)}(x) = f(x, z^{(j)}(x), v^{(j)}(x)) - f(x, z(x), v^{(j)}(x))$ converges to zero in measure as $j \to \infty$. This is shown by contradiction using the continuity of $f(x, \cdot, \cdot)$ and the boundedness of the integrals.

Extracting a subsequence from $h^{(j)}$, we may suppose that $h^{(j)}(x) \to 0$ a.e. in $\Omega$. By Mazur's theorem there exist convex combinations $\xi^{(k)} = \sum_{j=k}^\infty \lambda_j^k v^{(j)}$, where only finitely many $\lambda_j^k$ are nonzero for each $k$, such that $\xi^{(k)} \to v(x)$ a.e. as $k \to \infty$. Since $f(x, z(x), \cdot)$ is convex,

$$f(x, z(x), \xi^{(k)}(x)) + \sum_{j=k}^\infty \lambda_j^k h^{(j)}(x) \le \sum_{j=k}^\infty \lambda_j^k f(x, z^{(j)}(x), v^{(j)}(x))$$

for a.e. $x$ and large enough $k$. Integrating over $\Omega$, taking the $\liminf$ as $k \to \infty$, and applying Fatou's Lemma, we obtain the result.

</details>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 23</span><span class="math-callout__name">(Tonelli)</span></p>

Let $f = f(x, u, p)$ be convex in $p$ for each $x, u$ and suppose that

$$f(x, u, p) \ge \Phi(p) \quad \text{for all } x, u$$

for some continuous $\Phi$ with $\frac{\Phi(p)}{|p|} \to \infty$ as $|p| \to \infty$. Let

$$\mathcal{A} = \lbrace v \in W^{1,1}(a,b) : v(a) = \alpha,\, v(b) = \beta \rbrace$$

or $\mathcal{A} = \lbrace v \in W^{1,1}(a,b) : v(a) = \alpha \rbrace$.

Then $I$ attains an absolute minimum on $\mathcal{A}$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 23</summary>

Let $l = \inf_{\mathcal{A}} I$. Then $\infty > l > -\infty$. Let $u^{(j)} \in \mathcal{A}$ be a minimizing sequence, so that $I(u^{(j)}) \to l$. Then

$$\sup_j \int_a^b \Phi(u_x^{(j)}) \, dx < \infty$$

and so by Theorem 6 (de la Vallée Poussin) there exists a subsequence, still denoted $u^{(j)}$, such that $v^{(j)} := u_x^{(j)} \rightharpoonup v$ in $L^1(a,b)$ for some $v$. Therefore

$$u^{(j)}(x) = \alpha + \int_a^x v^{(j)}(s) \, ds \to u(x) := \alpha + \int_a^x v(s) \, ds \quad \text{for all } x \in [a,b].$$

In particular for the two-point boundary conditions we have $u(b) = \beta$. By the lower semicontinuity Theorem 22,

$$l = \liminf_{j \to \infty} I(u^{(j)}) \ge \int_a^b f(x, u(x), v(x)) \, dx = I(u) \ge l,$$

and hence $u$ is a minimizer.

</details>

### 2.2 Local Minimizers

Consider again the integral functional $I(u) = \int_a^b f(x, u(x), u_x(x)) \, dx$ with $f$ continuous and bounded below, with set of admissible functions $\mathcal{A} = \lbrace u \in W^{1,1}(a,b) : u(a) = \alpha, u(b) = \beta \rbrace$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Local Minimizers)</span></p>

$u \in \mathcal{A}$ is a **weak local minimizer** of $I$ if $I(u) < \infty$ and there exists $\varepsilon > 0$ such that $I(v) \ge I(u)$ for all $v \in \mathcal{A}$ with

$$\operatorname*{ess\,sup}_{x \in [a,b]} [|v(x) - u(x)| + |v_x(x) - u_x(x)|] < \varepsilon.$$

$u \in \mathcal{A}$ is a **strong local minimizer** of $I$ if $I(u) < \infty$ and there exists $\varepsilon > 0$ such that $I(v) \ge I(u)$ for all $v \in \mathcal{A}$ with

$$\max_{x \in [a,b]} |v(x) - u(x)| < \varepsilon.$$

</div>

Thus $u$ is a weak (resp. strong) local minimizer if it is a local minimizer with respect to the $W^{1,\infty}$ (resp. $L^\infty$) norm. A strong local minimizer is a weak local minimizer, but in general a weak local minimizer need not be a strong local minimizer.

### 2.3 Necessary Conditions for Local Minimizers

We now assume for simplicity that $f = f(x, u, p)$ is $C^3$ in its arguments $x, u, p$. Let $u \in \mathcal{A} \cap W^{1,\infty}(a,b)$ be a weak local minimizer. If $\varphi \in C_0^\infty(a,b)$ then $I(u + \tau\varphi)$ has a local minimum at $\tau = 0$, so that $\frac{d}{d\tau} I(u + \tau\varphi)|_{\tau=0} = 0$, provided this derivative exists.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Euler–Lagrange Equations)</span></p>

A weak local minimizer $u \in \mathcal{A} \cap W^{1,\infty}(a,b)$ satisfies the **weak Euler–Lagrange equation (WEL)**:

$$\int_a^b [f_u \varphi + f_p \varphi_x] \, dx = 0 \quad \text{for all } \varphi \in C_0^\infty(a,b),$$

i.e. $u$ satisfies the **Euler–Lagrange equation** $\frac{d}{dx} f_p = f_u$ in the sense of distributions. Equivalently, $u$ satisfies the **integrated Euler–Lagrange equation (IEL)**:

$$f_p = \int_a^x f_u \, ds + c, \quad x \in [a,b],$$

where $c$ is a constant.

The **second variation** satisfies $\delta^2 I(u) \ge 0$, that is

$$\int_a^b [f_{uu} \varphi^2 + 2f_{up} \varphi \varphi_x + f_{pp} \varphi_x^2] \, dx \ge 0 \quad \text{for all } \varphi \in C_0^\infty(a,b).$$

</div>

Now let $u \in \mathcal{A} \cap W^{1,\infty}(a,b)$ be a **strong** local minimizer. For $\varphi \in C_0^\infty(a,b)$ and $|\tau|$ small enough there is a unique smooth increasing solution $z_\tau(x)$ to $z + \tau\varphi(z) = x$ for $x \in [a,b]$. Define the **inner variation** $u_\tau(x) = u(z_\tau(x))$, which rearranges the values of $u$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Du Bois-Reymond Equation)</span></p>

A strong local minimizer $u \in \mathcal{A} \cap W^{1,\infty}(a,b)$ satisfies the **weak Du Bois-Reymond equation (WDBR)**:

$$\int_a^b [f_x \varphi + (f - u_x f_p) \varphi_x] \, dx = 0 \quad \text{for all } \varphi \in C_0^\infty(a,b),$$

i.e. $\frac{d}{dx}(f - u_x f_p) = f_x$ in the sense of distributions. Equivalently, the **integrated Du Bois-Reymond equation (IDBR)**:

$$f - u_x f_p = \int_a^x f_x \, ds + c, \quad x \in [a,b].$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(WDBR vs WEL)</span></p>

Note that (WDBR) does not follow from (WEL). In the special case $f = f(p)$ the "broken extremal"

$$u(x) = \begin{cases} qx & x \in [-1, 0] \\ rx & x \in [0, 1] \end{cases}$$

satisfies (WEL) on $[-1,1]$ if and only if $f_p(q) = f_p(r)$, i.e. the tangents to $f$ at $q, r$ have the same slope. If also (WDBR) holds then $f(q) - qf_p(q) = f(r) - rf_p(r)$, i.e. the tangents at $q, r$ are a common tangent.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Weierstrass Necessary Condition)</span></p>

For a strong local minimizer $u \in \mathcal{A} \cap W^{1,\infty}(a,b)$, the **Weierstrass necessary condition** holds: for a.e. $x \in (a,b)$,

$$f(x, u(x), u_x(x) + p) \ge f(x, u(x), u_x(x)) + p f_p(x, u(x), u_x(x)) \quad \text{for all } p.$$

That is, the tangent at $u_x(x)$ to the graph of $f(x, u(x), \cdot)$ does not lie above the graph.

</div>

This is derived by considering the localized variation $u_\varepsilon(x_0, x) = u(x) + \varepsilon \psi\!\left(\frac{x - x_0}{\varepsilon}\right)$ for $\psi \in W_0^{1,\infty}(-1,1)$, $\varphi \ge 0$, integrating and passing to the limit $\varepsilon \to 0$ to obtain (in 1D) quasiconvexity: $\int_{-1}^1 f(x, u(x), u_x(x) + \psi_y(y)) \, dy \ge \int_{-1}^1 f(x, u(x), u_x(x)) \, dy$.

### 2.4 Sufficient Conditions for Local Minimizers

By slightly strengthening the necessary conditions we can obtain sufficient conditions for a sufficiently regular $u \in \mathcal{A}$ to be a weak or strong local minimizer.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Strict Positivity of the Second Variation)</span></p>

For $u \in \mathcal{A} \cap W^{1,\infty}(a,b)$ write $\delta^2 I(u) > 0$ if

$$\int_a^b (f_{uu} \varphi^2 + 2f_{up} \varphi \varphi_x + f_{pp} \varphi_x^2) \, dx \ge \mu \int_a^b (\varphi^2 + \varphi_x^2) \, dx$$

for all $\varphi \in C_0^\infty(a,b)$ and some constant $\mu > 0$. Note that this holds for all $\varphi \in W_0^{1,2}(a,b)$ by density.

Note also that $\delta^2 I(u) > 0$ implies the **strong Legendre condition**: for a.e. $x \in (a,b)$

$$f_{pp}(x, u(x), u_x(x)) \ge \mu.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 24</span><span class="math-callout__name">(Sufficient Condition for Weak Local Minimizer)</span></p>

If $u \in \mathcal{A} \cap W^{1,\infty}(a,b)$ satisfies (WEL) and $\delta^2 I(u) > 0$ then $u$ is a strict weak local minimizer (i.e. there exists $\varepsilon > 0$ such that $I(v) > I(u)$ for all $v \in \mathcal{A}$ with $0 < \|v - u\|_{1,\infty} < \varepsilon$).

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 24</summary>

Let $\varphi \in W_0^{1,\infty}(a,b)$. Setting $\theta(t) = f(x, u + t\varphi, u_x + t\varphi_x)$ and using $\theta(1) - \theta(0) = \theta'(0) + \int_0^1 (1-t)\theta''(t) \, dt$, we obtain

$$I(u + \varphi) - I(u) = \int_a^b \underbrace{(f_u \varphi + f_p \varphi_x)}_{= 0 \text{ by WEL}} dx + \frac{1}{2}\delta^2 I(u)(\varphi, \varphi) + R(u, \varphi)$$

where $R(u, \varphi) = \int_a^b \int_0^1 (1-t)[(f_{uu}(x, u+t\varphi, u_x + t\varphi_x) - f_{uu}(x, u, u_x))\varphi^2 + \cdots] \, dt \, dx$. For $\varepsilon > 0$ sufficiently small and $\|\varphi\|_{1,\infty} < \varepsilon$, we have $R(u, \varphi) \ge -\frac{\mu}{4} \int_a^b (\varphi^2 + \varphi_x^2) \, dx$, and hence

$$I(u + \varphi) - I(u) \ge \frac{\mu}{4} \int_a^b (\varphi^2 + \varphi_x^2) \, dx,$$

as required.

</details>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Strengthened Weierstrass Condition)</span></p>

We say that $u \in \mathcal{A} \cap C^1([a,b])$ satisfies the **strengthened Weierstrass condition** if there exists $\delta > 0$ such that for all $x \in [a,b]$ and $p \in \mathbb{R}$

$$f(x, v, p) \ge f(x, v, q) + (p - q) f_p(x, v, q)$$

whenever $|v - u(x)| < \delta$, $|q - u_x(x)| < \delta$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 25</span><span class="math-callout__name">(Weierstrass — Sufficient Condition for Strong Local Minimizer)</span></p>

Let $u \in \mathcal{A} \cap C^1([a,b])$ satisfy (WEL), $\delta^2 I(u) > 0$ and the strengthened Weierstrass condition. Then $u$ is a strong local minimizer. If strict inequality holds in the Weierstrass condition for $p \ne q$ then $u$ is a strict strong local minimizer.

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch of Theorem 25 (Hilbert's field method)</summary>

Using $\delta^2 I(u) > 0$, the analysis of the Jacobi equation (the Euler–Lagrange equation of $\delta^2 I(u)(\varphi, \varphi)$) and conjugate points leads to the conclusion that $u$ is embedded in a **field of extremals**, that is there is a one-parameter family $U(x, \gamma)$, $\gamma \in [-\tau, \tau]$, $\tau > 0$, of solutions to the Euler–Lagrange equation for $f$ on $[a,b]$ such that:

1. $u(x) = U(x, 0)$ for all $x \in [a,b]$,
2. the field simply covers a neighbourhood of the graph of $u$, i.e. there exists $\varepsilon > 0$ such that for each $x \in [a,b]$, $v \in \mathbb{R}$, with $|v - u(x)| < \varepsilon$, there is a unique $\gamma = \gamma(x,v) \in [-\tau, \tau]$ with $U(x, \gamma) = v$.

We write $p(x, v) = U_x(x, \gamma(x,v))$ and call $p(\cdot, \cdot)$ the **slope function** of the field. For $v \in \mathcal{A}$ with $\|v - u\|_\infty$ sufficiently small:

$$I(v) - I(u) = \int_a^b [f(x, v, v_x) - f(x, v, p(x,v)) - f_p(x, v, p(x,v))(v_x - p(x,v))] \, dx,$$

where $p(x,v)$ is the slope function of the field. Thus $I(v) \ge I(u)$ by the strengthened Weierstrass condition.

</details>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Null Lagrangians)</span></p>

The key computation in the proof of Theorem 25 can be interpreted as showing that $L(x, v, v_x) = f(x, v, p(x,v)) + f_p(x, v, p(x,v))(v_x - p(x,v))$ is a **null Lagrangian**, i.e. the corresponding Euler–Lagrange equation reduces to $0 = 0$.

</div>

### 2.5 Regularity and the Lavrentiev Phenomenon

We assumed above that $u \in C^1([a,b])$. But when is this true? A first regularity result is:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 26</span><span class="math-callout__name">(Regularity under Strict Ellipticity)</span></p>

Suppose that $f \in C^2$ and that $f_{pp}(x, v, p) > 0$ for all $x, v, p$. If $u \in \mathcal{A} \cap W^{1,\infty}(a,b)$ solves (WEL) then $u \in C^2([a,b])$ and

$$u_{xx} = F(x, u, u_x) \quad \text{for all } x \in [a,b],$$

where $F = \frac{f_u - f_{xp} - f_{up}p}{f_{pp}}$.

</div>

<details class="accordion" markdown="1">
<summary>Proof of Theorem 26</summary>

**Step 1.** We prove that $u \in C^1([a,b])$. Choose the continuous representative of $u$. We have $|u_x(x)| \le M < \infty$ and $f_p(x, u(x), u_x(x)) = c + \int_a^x f_u \, dy$ (IEL) for all $x \in E$, where $\mathrm{meas}\, E = b - a$. We claim that $p(x) := \lim_{z \to x,\, z \in E} u_x(z)$ exists. Suppose not. Then $u_x(x_j) \to p_1$, $u_x(y_j) \to p_2$ for sequences $x_j, y_j \in E$, $p_1 \ne p_2$, $x_j, y_j \to x$. But from (IEL) we deduce $f_p(x, u(x), p_1) = f_p(x, u(x), p_2)$. Since $f_{pp} > 0$ this is a contradiction.

**Step 2.** We prove that $u \in C^2([a,b])$. For each $x \in [a,b]$:

$$\lim_{h \to 0} \frac{f_p(x+h, u(x+h), u_x(x+h)) - f_p(x, u(x), u_x(x))}{h} = f_u(x, u(x), u_x(x)).$$

Expanding the left-hand side and using $f_{pp} > 0$, we get that $u_x$ is differentiable with $u_{xx} = F(x, u, u_x)$.

</details>

But does the global minimizer $u$ given by Theorem 23 belong to $W^{1,\infty}(a,b)$ or satisfy (WEL)?

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.4</span><span class="math-callout__name">(Lavrentiev Phenomenon)</span></p>

Let

$$I(u) = \int_{-1}^1 [(u^5 - x^3)^2 u_x^{20} + \varepsilon u_x^2] \, dx,$$

where $\varepsilon > 0$ is sufficiently small, and $\mathcal{A} = \lbrace v \in W^{1,1}(-1,1) : v(-1) = -1, v(1) = 1 \rbrace$. Note that $f(x, u, p) = (u^5 - x^3)^2 p^{20} + \varepsilon p^2$ is a polynomial with $f_{pp} \ge 2\varepsilon > 0$, and that $f$ has superlinear growth in $p$, so that $f$ satisfies the hypotheses of Theorem 23. Hence there exists an absolute minimizer $u^\ast$.

We claim that if $u \in \mathcal{A} \cap W^{1,\infty}(-1,1)$ then $I(u) \ge \frac{2^{14}}{3^{20}}$. But choosing $u = |x|^{3/5} \operatorname{sign} x$ we have $\inf_{\mathcal{A}} I \le 2\varepsilon \cdot \frac{9}{5}$. Hence if $\varepsilon < \varepsilon_0 := \frac{5}{18} \cdot \frac{2^{14}}{3^{20}}$ we have

$$\inf_{\mathcal{A} \cap W^{1,\infty}} I > \inf_{\mathcal{A}} I \quad !!!$$

This is the **Lavrentiev phenomenon**: the infimum can be different in different function spaces.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Properties of the Lavrentiev Minimizer)</span></p>

The global minimizer $u^\ast$ satisfies $u^\ast(0) = 0$ and $f_p(x, u^\ast, u_x^\ast)$ is unbounded in the neighbourhood of $x = 0$. In particular (IEL) **does not hold**. Indeed if $u^\ast(0) \ne 0$ we get $I(u^\ast) \ge \frac{2^{14}}{3^{20}} > I(u^\ast)$, a contradiction. Hence $u_x^\ast$ is unbounded near 0 and so is $|f_p| = |20(u^5 - x^3)^2 u_x^{\ast 19} + 2\varepsilon u_x^\ast| \ge 2\varepsilon |u_x^\ast|$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 27</span><span class="math-callout__name">(Tonelli's Partial Regularity Theorem)</span></p>

Let $f$ be $C^3$ with $f_{pp} > 0$. If $u \in \mathcal{A}$ is a strong local minimizer of $I$ in $\mathcal{A}$, then there is a closed set $E \subset [a,b]$ with $\mathrm{meas}\, E = 0$ such that $u$ is a $C^3$ solution of the Euler–Lagrange equation on $[a,b] \setminus E$. Furthermore the derivative

$$u'(x) := \lim_{h \to 0} \frac{u(x+h) - u(x)}{h}$$

exists for all $x \in [a,b]$ as an element of $\bar{\mathbb{R}}$ (one-sided limits if $x = a$ or $x = b$), and $u' : [a,b] \to \bar{\mathbb{R}}$ is continuous with $E = \lbrace x \in [a,b] : |u'(x)| = \infty \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequences of the Lavrentiev Phenomenon)</span></p>

1. The example shows that an elliptic regularization (adding $\varepsilon u_x^2$ to a degenerate elliptic problem) may not smooth minimizers.
2. If $\varphi \in C_0^\infty(-1,1)$, $\varphi(0) \ne 0$, then $I(u^\ast + t\varphi) = \infty$ for all $t \ne 0$, since $I(u^\ast + t\varphi) \ge \delta \int_{-r}^r u_x^{\ast 20} \, dx = \infty$.
3. The Lavrentiev phenomenon shows that typical finite element schemes for minimizing $I$ among piecewise affine functions may not converge to a minimizer.

</div>
