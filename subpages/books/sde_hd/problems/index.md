---
title: Problems from the SDEs & Diffusion Models course
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

# SDEs & Diffusion Models — Course Problems

**Table of Contents**
- TOC
{:toc}

---

## Exercise Sheet 1 — Brownian Motion

*Keywords: Brownian motion, Gaussian process, scaling invariance, independent-increments properties.*

Throughout, let $(\Omega, \mathcal{A}, \mathbb{P})$ be an underlying probability space and $I \in \{[0, T], [0, \infty)\}$ an index set (with $T > 0$). Properties (W1)–(W4) refer to Definition 0.3 in the main notes.

---

### Exercise 1.1 — Characterisation of Brownian motion

Prove the characterisation:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Gaussian-process characterisation of BM)</span></p>

A real-valued stochastic process $W := (W_t)_{t \in I}$ on $(\Omega, \mathcal{A}, \mathbb{P})$ is a one-dimensional standard Brownian motion if, and only if,

- **(W123)** $W$ is a *centred Gaussian process* with covariance function $\gamma(s, t) := \operatorname{Cov}(W_s, W_t) = s \wedge t$ for all $s, t \in I$;
- **(W4)** the paths $t \mapsto W_t$ are $\mathbb{P}$-a.s. continuous on $I$.

</div>

<details class="accordion" markdown="1">
<summary>Solution 1.1</summary>

(W4) is common to both sides, so only (W1)–(W3) $\Longleftrightarrow$ (W123) requires proof.

**"$\Longrightarrow$": assume (W1)–(W4).**

Fix any $n \in \mathbb{N}$ and times $0 \leq t_1 < t_2 < \dots < t_n$ in $I$. Write $t_0 := 0$. By (W1)/(W3) the increments

$$\Delta_k := W_{t_k} - W_{t_{k-1}} \sim \mathcal{N}(0, t_k - t_{k-1}), \qquad k = 1, \dots, n,$$

and by (W2) they are mutually independent. Hence the joint law of $(\Delta_1, \dots, \Delta_n)$ is a multivariate Gaussian on $\mathbb{R}^n$, and

$$\begin{pmatrix} W_{t_1} \\ W_{t_2} \\ \vdots \\ W_{t_n} \end{pmatrix} = \underbrace{\begin{pmatrix} 1 & 0 & \cdots & 0 \\ 1 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \cdots & 1 \end{pmatrix}}_{=: L} \begin{pmatrix} \Delta_1 \\ \vdots \\ \Delta_n \end{pmatrix}$$

is an affine image of a Gaussian vector, hence Gaussian. This makes the finite-dimensional distributions of $W$ multivariate normal — by Kolmogorov's extension theorem, $W$ is a Gaussian process.

*Mean*: $\mathbb{E}[W_t] = \mathbb{E}[W_t - W_0] = 0$ by (W1)/(W3), so $W$ is centred.

*Covariance*: for $s \leq t$,

$$\operatorname{Cov}(W_s, W_t) = \operatorname{Cov}\big(W_s,\, W_s + (W_t - W_s)\big) = \operatorname{Var}(W_s) + \underbrace{\operatorname{Cov}(W_s, W_t - W_s)}_{= 0\text{ by (W2)}} = s = s \wedge t,$$

using $\operatorname{Var}(W_s) = s$ from (W3) with $W_0 = 0$. By symmetry, $\gamma(s, t) = s \wedge t$ for all $s, t$.

**"$\Longleftarrow$": assume (W123).**

*(W1).* Setting $s = t = 0$, $\operatorname{Var}(W_0) = \gamma(0, 0) = 0$, so $W_0 = \mathbb{E}[W_0] = 0$ $\mathbb{P}$-a.s.

*(W3).* For $s < t$, $W_t - W_s$ is Gaussian (a linear combination of the Gaussian vector $(W_s, W_t)$), with mean $0$ and variance

$$\operatorname{Var}(W_t - W_s) = \operatorname{Var}(W_t) + \operatorname{Var}(W_s) - 2\operatorname{Cov}(W_s, W_t) = t + s - 2s = t - s.$$

So $W_t - W_s \sim \mathcal{N}(0, t - s)$.

*(W2).* Fix $0 = t_0 < t_1 < \dots < t_n$ in $I$ and set $\Delta_k := W_{t_k} - W_{t_{k-1}}$. The vector $(\Delta_1, \dots, \Delta_n)$ is an affine image of the Gaussian vector $(W_{t_1}, \dots, W_{t_n})$, hence Gaussian. For $i < j$,

$$\operatorname{Cov}(\Delta_i, \Delta_j) = \gamma(t_i, t_j) - \gamma(t_i, t_{j-1}) - \gamma(t_{i-1}, t_j) + \gamma(t_{i-1}, t_{j-1}) = t_i - t_i - t_{i-1} + t_{i-1} = 0,$$

using $\gamma(u, v) = u \wedge v = u$ whenever $u \leq v$. So the increments are pairwise uncorrelated, and for Gaussian vectors *uncorrelated $\Longleftrightarrow$ mutually independent*. Hence (W2). $\blacksquare$

</details>

---

### Exercise 1.2 — Constructing Brownian motions

**(a) Scaling invariance.** Let $W$ be a one-dimensional standard Brownian motion and let $\alpha \in (0, \infty)$. Show that

$$W^{(\alpha)}_t := \alpha \, W_{t/\alpha^2}, \qquad t \in I,$$

is again a standard Brownian motion.

**(b)** Let $W^{(1)}, W^{(2)}$ be independent standard BMs. Find all $a, b \in \mathbb{R}$ such that

$$B_t := a\, W^{(1)}_t + b\, W^{(2)}_t, \qquad t \in I,$$

is a standard Brownian motion.

<details class="accordion" markdown="1">
<summary>Solution 1.2</summary>

The Exercise 1.1 characterisation reduces both parts to checking (i) centred Gaussianity, (ii) covariance $s \wedge t$, and (iii) $\mathbb{P}$-a.s. path continuity.

**(a) Scaling invariance.**

*(i)* For any $t_1, \dots, t_n$, the vector $(W^{(\alpha)}_{t_1}, \dots, W^{(\alpha)}_{t_n}) = \alpha\,(W_{t_1/\alpha^2}, \dots, W_{t_n/\alpha^2})$ is a scalar multiple of a Gaussian vector, hence Gaussian. Centred because $\mathbb{E}[W^{(\alpha)}_t] = \alpha\, \mathbb{E}[W_{t/\alpha^2}] = 0$.

*(ii)* Covariance:

$$\operatorname{Cov}(W^{(\alpha)}_s, W^{(\alpha)}_t) = \alpha^2 \operatorname{Cov}(W_{s/\alpha^2}, W_{t/\alpha^2}) = \alpha^2 \cdot \big(s/\alpha^2 \wedge t/\alpha^2\big) = s \wedge t. \checkmark$$

*(iii)* $t \mapsto W^{(\alpha)}_t = \alpha W_{t/\alpha^2}$ is the composition of the a.s.-continuous map $t \mapsto W_t$ with the continuous map $t \mapsto t/\alpha^2$, multiplied by $\alpha$. So paths are $\mathbb{P}$-a.s. continuous. $\blacksquare$

**(b) Linear combination of independent BMs.**

*(i) & (iii)* $(B_t)$ is a linear combination of two independent Gaussian processes with a.s.-continuous paths, so $B$ is itself a centred Gaussian process with a.s.-continuous paths, regardless of $(a, b)$.

*(ii)* Using independence of $W^{(1)}$ and $W^{(2)}$,

$$\operatorname{Cov}(B_s, B_t) = a^2 \operatorname{Cov}(W^{(1)}_s, W^{(1)}_t) + b^2 \operatorname{Cov}(W^{(2)}_s, W^{(2)}_t) = (a^2 + b^2)(s \wedge t).$$

This equals $s \wedge t$ (take $s = t > 0$) **iff** $a^2 + b^2 = 1$. Hence

$$\boxed{\; B \text{ is a standard BM} \iff (a, b) \in S^1 = \{(a, b) \in \mathbb{R}^2 : a^2 + b^2 = 1\}. \;}$$

Geometrically: a unit-vector rotation in the plane spanned by $W^{(1)}$ and $W^{(2)}$ preserves the BM law — a foretaste of rotational invariance of multidimensional Brownian motion.

</details>

---

### Exercise 1.3 — Independent-increments properties

Let $W := (W_t)_{t \in I}$ be a real-valued process on $(\Omega, \mathcal{A}, \mathbb{P})$ with $W_0 = 0$. Let $\mathbb{F} := (\mathcal{F}_t)_{t \in I}$, $\mathcal{F}_t := \sigma(W_s : s \in [0, t])$, be its canonical filtration. Consider

- **(W2)** *Independent increments*: for any $n \in \mathbb{N}$ and $0 =: t_0 < t_1 < \dots < t_n$ in $I$, the increments $W_{t_1} - W_{t_0}, \dots, W_{t_n} - W_{t_{n-1}}$ are mutually independent.
- **(W2′)** $\mathbb{F}$-*independent increments*: for all $s < t$ in $I$, $W_t - W_s$ is independent of $\mathcal{F}_s$.

Prove (W2) $\Longleftrightarrow$ (W2′).

**(a)** Show (W2′) $\Rightarrow$ (W2). For the reverse direction, fix $s < t$ in $I$ and define

$$\mathcal{D} := \{F \in \mathcal{F}_s : W_t - W_s \text{ is independent of } F\},$$

$$\mathcal{C} := \bigcup_{n \in \mathbb{N}} \bigcup_{0 =: t_0 < t_1 < \dots < t_n \leq s} (W_{t_1} - W_{t_0}, \dots, W_{t_n} - W_{t_{n-1}})^{-1}(\mathcal{B}(\mathbb{R}^n)).$$

**(b)** Show $\mathcal{D}$ is a Dynkin ($\lambda$-)system.
**(c)** Show $\mathcal{C}$ is a $\pi$-system generating $\mathcal{F}_s$.
**(d)** Verify $\mathcal{C} \subseteq \mathcal{D}$ and apply Dynkin's $\pi$-$\lambda$ theorem to conclude $\mathcal{D} = \mathcal{F}_s$.

<details class="accordion" markdown="1">
<summary>Solution 1.3</summary>

**(a) (W2′) $\Rightarrow$ (W2).**

Fix $0 = t_0 < t_1 < \dots < t_n$ in $I$ and Borel sets $A_1, \dots, A_n \in \mathcal{B}(\mathbb{R})$. For each $k \leq n - 1$, the increment $W_{t_k} - W_{t_{k-1}}$ is $\mathcal{F}_{t_{n-1}}$-measurable (since $t_k, t_{k-1} \leq t_{n-1}$), so the event

$$E := \{W_{t_k} - W_{t_{k-1}} \in A_k \text{ for } k = 1, \dots, n-1\} \in \mathcal{F}_{t_{n-1}}.$$

By (W2′) applied to $s = t_{n-1} < t = t_n$,

$$\mathbb{P}\big(E \cap \{W_{t_n} - W_{t_{n-1}} \in A_n\}\big) = \mathbb{P}(E)\,\mathbb{P}(W_{t_n} - W_{t_{n-1}} \in A_n).$$

Iterating this identity with $n \mapsto n-1, n-2, \dots, 2$ yields

$$\mathbb{P}\!\left(\bigcap_{k=1}^{n}\{W_{t_k} - W_{t_{k-1}} \in A_k\}\right) = \prod_{k=1}^{n} \mathbb{P}(W_{t_k} - W_{t_{k-1}} \in A_k),$$

which is exactly the mutual independence (W2). $\blacksquare$

**(b) $\mathcal{D}$ is a Dynkin system.**

Let $\mu_{t,s}(A) := \mathbb{P}(W_t - W_s \in A)$ for $A \in \mathcal{B}(\mathbb{R})$. $F \in \mathcal{D}$ means $\mathbb{P}(\{W_t - W_s \in A\} \cap F) = \mu_{t,s}(A) \mathbb{P}(F)$ for *every* $A \in \mathcal{B}(\mathbb{R})$.

- *Contains $\Omega$*: trivial, $\mathbb{P}(\{W_t - W_s \in A\} \cap \Omega) = \mu_{t,s}(A) = \mu_{t,s}(A)\cdot 1$.
- *Closed under complements in $\mathcal{F}_s$*: if $F \in \mathcal{D}$, then for any $A$,

  $$\mathbb{P}(\{W_t - W_s \in A\} \cap F^c) = \mu_{t,s}(A) - \mu_{t,s}(A)\mathbb{P}(F) = \mu_{t,s}(A)\mathbb{P}(F^c).$$

- *Closed under countable disjoint unions*: if $F_1, F_2, \dots \in \mathcal{D}$ are pairwise disjoint,

  $$\mathbb{P}\!\left(\{W_t - W_s \in A\} \cap \bigsqcup_i F_i\right) = \sum_i \mathbb{P}(\{W_t - W_s \in A\} \cap F_i) = \sum_i \mu_{t,s}(A)\mathbb{P}(F_i) = \mu_{t,s}(A)\mathbb{P}\!\left(\bigsqcup_i F_i\right).$$

So $\mathcal{D}$ is a Dynkin system. $\blacksquare$

**(c) $\mathcal{C}$ is a $\pi$-system and $\sigma(\mathcal{C}) = \mathcal{F}_s$.**

*Simplified representation.* Since $W_0 = 0$, the linear map

$$T_n : (x_1, \dots, x_n) \mapsto (x_1, x_2 - x_1, \dots, x_n - x_{n-1})$$

is a bijection of $\mathbb{R}^n$ with $T_n, T_n^{-1}$ Borel-measurable (both are linear). Therefore

$$(W_{t_1} - W_{t_0}, \dots, W_{t_n} - W_{t_{n-1}})^{-1}(\mathcal{B}(\mathbb{R}^n)) = (W_{t_1}, \dots, W_{t_n})^{-1}(\mathcal{B}(\mathbb{R}^n))$$

for any $0 < t_1 < \dots < t_n \leq s$. (The condition $t_0 = 0$ can be dropped since the first component is then $W_{t_1} - W_0 = W_{t_1}$.) Hence

$$\mathcal{C} = \bigcup_{n \in \mathbb{N}} \bigcup_{0 < t_1 < \dots < t_n \leq s} (W_{t_1}, \dots, W_{t_n})^{-1}(\mathcal{B}(\mathbb{R}^n)).$$

*Generates $\mathcal{F}_s$.* Trivially $\mathcal{C} \subseteq \mathcal{F}_s$ (each event is $\sigma(W_r : r \in (0, s])$-measurable, and $W_0 = 0$ adds nothing). Conversely, for each $r \in [0, s]$, $\{W_r \in B\} = (W_r)^{-1}(B) \in \mathcal{C}$ (taking $n = 1$, $t_1 = r$; the $r = 0$ case is the trivial event $\{0 \in B\}$). So $\mathcal{F}_s = \sigma(W_r : r \in [0, s]) \subseteq \sigma(\mathcal{C})$. Hence $\sigma(\mathcal{C}) = \mathcal{F}_s$.

*$\pi$-system.* Take $F, F' \in \mathcal{C}$, say $F = (W_{t_1}, \dots, W_{t_n})^{-1}(B)$ and $F' = (W_{t_1'}, \dots, W_{t_m'})^{-1}(B')$. Let $\{r_1 < r_2 < \dots < r_N\} := \{t_i\} \cup \{t_j'\}$ (the ordered union of times, with duplicates collapsed). Both $F$ and $F'$ can be rewritten as preimages under the bigger vector $(W_{r_1}, \dots, W_{r_N})$ of suitable cylindrical Borel sets $\widetilde B, \widetilde B' \in \mathcal{B}(\mathbb{R}^N)$, and

$$F \cap F' = (W_{r_1}, \dots, W_{r_N})^{-1}(\widetilde B \cap \widetilde B') \in \mathcal{C}. \quad\blacksquare$$

**(d) $\mathcal{C} \subseteq \mathcal{D}$ and conclusion.**

Take $F \in \mathcal{C}$, say $F = (W_{t_1} - W_{t_0}, \dots, W_{t_n} - W_{t_{n-1}})^{-1}(B)$ with $0 = t_0 < t_1 < \dots < t_n \leq s$. Insert $s$ and $t$ into the time grid (if $t_n < s$), producing the extended grid $0 = t_0 < t_1 < \dots < t_n \leq s < t$. By the already-assumed (W2), the increments

$$\Delta_1, \dots, \Delta_n, \Delta_{n+1} := W_s - W_{t_n}, \ \Delta_{n+2} := W_t - W_s$$

are mutually independent. Hence the $\sigma$-algebra generated by $(\Delta_1, \dots, \Delta_{n+1})$ — which contains $F$ — is independent of $\sigma(\Delta_{n+2}) = \sigma(W_t - W_s)$. (If $t_n = s$, skip the $W_s - W_{t_n}$ step: $W_t - W_s$ is then directly independent of $(\Delta_1, \dots, \Delta_n)$.) So $F \in \mathcal{D}$.

Hence $\mathcal{C} \subseteq \mathcal{D}$. Since $\mathcal{C}$ is a $\pi$-system and $\mathcal{D}$ is a Dynkin system, **Dynkin's $\pi$-$\lambda$ theorem** gives $\sigma(\mathcal{C}) \subseteq \mathcal{D}$. Combining with $\sigma(\mathcal{C}) = \mathcal{F}_s$ and $\mathcal{D} \subseteq \mathcal{F}_s$, we conclude $\mathcal{D} = \mathcal{F}_s$ — that is, $W_t - W_s$ is independent of every $F \in \mathcal{F}_s$, which is (W2′). $\blacksquare$

</details>

---

### Exercise 1.4 — Path properties of Brownian motion

Let $(W_t)_{t \geq 0}$ be a continuous standard one-dimensional Brownian motion.

**(a)** Show **Proposition 0.11 (iii)**: on every interval $[a, b] \subset [0, \infty)$ with $a < b$, the paths $t \mapsto W_t$ are $\mathbb{P}$-a.s. not monotonically increasing.

**(b)** Let $\ell, u : [0, 1] \to \mathbb{R}$ be $\ell(t) := t$, $u(t) := 2t$. Show **Proposition 0.11 (iv)**: the paths $t \mapsto W_t$ on $[0, 1]$ are $\mathbb{P}$-a.s. *not* sandwiched by $\ell$ and $u$.

<details class="accordion" markdown="1">
<summary>Solution 1.4</summary>

**(a) No monotone interval.** Define

$$A := \{\omega : t \mapsto W_t(\omega) \text{ is not monotonically increasing on any } [a, b] \subset [0, \infty),\ a < b\}.$$

*Step 1 — measurability via rationals.* Let $A'$ be the analogous set but with $a, b$ ranging over $[0, \infty) \cap \mathbb{Q}$. By continuity of paths, $W$ is monotonically increasing on a real interval $[a, b]$ $\Longleftrightarrow$ it is monotonically increasing on the rational interval $[a', b']$ for *some* (equivalently, every) $a < a' < b' < b$ rational. Hence $A = A'$, and

$$A' = \bigcap_{\substack{a, b \in \mathbb{Q}_{\geq 0} \\ a < b}} \{\omega : t \mapsto W_t(\omega) \text{ is not monotone on } [a, b]\} \in \mathcal{A}$$

as a countable intersection of measurable sets.

*Step 2 — grid event.* Fix rationals $a < b$ in $[0, \infty)$ and $n \in \mathbb{N}$. With grid points $t_k := a + k(b-a)/n$, $k = 0, \dots, n$, define

$$M_{a,b}(n) := \{W_{t_0} \leq W_{t_1} \leq \dots \leq W_{t_n}\}.$$

The $n$ increments $\Delta_k := W_{t_k} - W_{t_{k-1}} \sim \mathcal{N}\big(0, (b-a)/n\big)$ are mutually independent (W2), and each satisfies $\mathbb{P}(\Delta_k \geq 0) = \tfrac{1}{2}$. Therefore

$$\mathbb{P}(M_{a,b}(n)) = \mathbb{P}(\Delta_1 \geq 0, \dots, \Delta_n \geq 0) = \Big(\tfrac{1}{2}\Big)^n \xrightarrow{n \to \infty} 0.$$

*Step 3 — link.* If the path is monotonically increasing on $[a, b]$, it is in particular non-decreasing on the grid, so $\{t \mapsto W_t \text{ monotone on } [a, b]\} \subseteq M_{a, b}(n)$ for every $n$. Consequently

$$\mathbb{P}(t \mapsto W_t \text{ monotone on } [a, b]) \leq \mathbb{P}(M_{a, b}(n)) = 2^{-n} \quad\forall n \in \mathbb{N},$$

so the probability is $0$. Taking a countable union over rational $a < b$,

$$\mathbb{P}(A^c) = \mathbb{P}(A'^c) \leq \sum_{\substack{a, b \in \mathbb{Q}_{\geq 0} \\ a < b}} \mathbb{P}(W \text{ monotone on } [a, b]) = 0,$$

so $\mathbb{P}(A) = 1$. $\blacksquare$

**(b) Path not sandwiched by $\ell$ and $u$.** Define

$$B := \{\omega : \ell(t) \leq W_t(\omega) \leq u(t) \ \forall\, t \in [0, 1]\} = \{\omega : t \leq W_t(\omega) \leq 2t \ \forall t \in [0, 1]\}.$$

The claim is $\mathbb{P}(B) = 0$. *(The hint in the sheet prints "show $\mathbb{P}(B) = 1$" — a typo; the conclusion "Deduce that $\mathbb{P}(B) = 0$" is the intended one.)*

*Measurability.* By continuity of paths, it suffices to check the sandwich on a countable dense subset:

$$B = \bigcap_{q \in [0, 1] \cap \mathbb{Q}} \{\ell(q) \leq W_q \leq u(q)\} \in \mathcal{A}.$$

*Reduction to a shrinking-window estimate.* For $n \in \mathbb{N}$ and $k \in \{0, \dots, n\}$, set

$$S_k(n) := \Big\{\tfrac{k}{n^2} \leq W_{k/n^2} \leq \tfrac{2k}{n^2}\Big\}.$$

Since $k/n^2 \in [0, 1/n] \subset [0, 1]$ for $k \leq n$, the sandwich on all of $[0, 1]$ implies the sandwich at each of these times:

$$B \subseteq \bigcap_{k=0}^{n} S_k(n) \subseteq S_1(n) = \Big\{\tfrac{1}{n^2} \leq W_{1/n^2} \leq \tfrac{2}{n^2}\Big\}.$$

*Key estimate.* Standardise: $n W_{1/n^2} \sim \mathcal{N}(0, 1)$. With $Z \sim \mathcal{N}(0, 1)$,

$$\mathbb{P}(S_1(n)) = \mathbb{P}\Big(\tfrac{1}{n} \leq Z \leq \tfrac{2}{n}\Big) \leq \phi(0) \cdot \tfrac{1}{n} = \frac{1}{n\sqrt{2\pi}} \xrightarrow{n \to \infty} 0,$$

where $\phi$ is the standard-normal density, maximised at $0$.

Hence $\mathbb{P}(B) \leq \mathbb{P}(S_1(n)) \to 0$, so $\mathbb{P}(B) = 0$. $\blacksquare$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What is going on geometrically?)</span></p>

Near $t = 0$, the "typical" fluctuation of $W_t$ is $\sqrt{t}$, but the sandwich $[\ell(t), u(t)] = [t, 2t]$ has *linear* width in $t$. As $t \to 0^+$, $t / \sqrt{t} = \sqrt{t} \to 0$, so the sandwich window, measured in standard deviations, *collapses* to the single point $0$. Since $W_0 = 0$ lies on the lower boundary $\ell$, not strictly inside, the path immediately has positive probability to escape either below $\ell$ or above $u$ — and part (b) shows this escape is certain.

The same law of the iterated logarithm intuition — $\limsup_{t \to 0^+} W_t / \sqrt{2t \log \log(1/t)} = 1$ — turns the "certainty" into a quantitative statement, and also explains why the statement stays true for *any* pair $0 < c_1 < c_2$ replacing the slopes $1, 2$.

</div>

</details>
