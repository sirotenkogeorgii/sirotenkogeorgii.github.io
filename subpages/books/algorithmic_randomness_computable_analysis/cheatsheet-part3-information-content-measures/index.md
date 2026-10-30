---
title: "Cheatsheet — Part 3: Information Content Measures"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - kolmogorov-complexity
  - information-content-measure
  - kraft-inequality
  - kraft-chaitin
  - computability
  - 1-randomness
  - cheatsheet
---

# Cheatsheet — Part 3: Information Content Measures

*Exam recap: definitions, formulas and key results only, covering the chapter Information Content Measures (results 3.1–3.8 and the two exercises) up to (but excluding) Incompressibility and Typicality. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}#information-content-measures). Prerequisites in [Part 1]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part1-plain-complexity/' | relative_url }}) and [Part 2]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part2-prefix-free-complexity/' | relative_url }}).*

## Why information content measures

Part 2 ended with two facts about $K$ that never mention a machine: $\sum\_w 2^{-K(w)} \le 1$ (Thm. 2.26) and $K$ is upper-semicomputable (Prop. 2.23). This chapter turns them into **axioms**. Any function with these two properties is an *information content measure* (i.c.m.), and $K$ is the **smallest** one up to an additive constant (Thm. 3.6). That characterizes $K$ **without Turing machines** — the exact $K$-analogue of the axiomatic characterization of $C$ in Part 1 (uniformly c.e. families with $\lvert V\_n\rvert < 2^n$), with **counting** replaced by **Kraft mass**.

Two payoffs:

* a third, machine-free way to bound $K$ from above — *exhibit an i.c.m.* — after machines (Part 2) and KC sets (Cor. 2.28.1);
* **Chaitin's counting theorem** (3.7): the maximal $K$ at length $n$ sits at $n + K(\mathrm{bin}(n))$, and almost every word has $K(w) > l(w)$ (3.7.1). This removes the counting obstruction (Thm. 1.3) that made "all prefixes incompressible" vacuous for $C$, and opens the door to Def. 3.8.

## Definition and examples

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.1 — Information content measure)</span></p>

An **information content measure** is a partial function $F :\subseteq \lbrace 0,1\rbrace^{\ast} \to \mathbb{N}$ with

$$\sum_{\sigma \in \mathrm{dom}(F)} 2^{-F(\sigma)} \le 1 \tag{19}$$

$$\lbrace (\sigma, k) : F(\sigma) \le k\rbrace \ \text{ is c.e.} \tag{20}$$

(19) is the **Kraft bound**, (20) says the **epigraph** is c.e.

</div>

* **(20) = upper-semicomputability** (Def. 2.7): $F = \lim\_t F(\cdot, t)$ for a computable nonincreasing staircase (value $\infty$ off the domain). You can *certify* $F(\sigma) \le k$, never $F(\sigma) > k$.
* **(19) = Kraft** with $F$ in place of $l$ or $K$: each domain word spends mass $2^{-F(\sigma)}$ out of a unit budget.
* **No machines anywhere.** $\mathrm{dom}(F)$ need *not* be prefix-free: $\mathrm{dom}(F) = \lbrace 0, 10, 001\rbrace$ with values $2, 4, 5$ is an i.c.m. although $0 \sqsubset 001$.

| i.c.m. | domain | (20) holds because | (19) holds because |
| :-- | :-- | :-- | :-- |
| $f(0^n1) = n + 1$ | $\lbrace 0^n 1 : n \ge 0\rbrace$ | partially computable | $\sum\_n 2^{-(n+1)} = 1$ — budget spent **exactly** |
| $K$ | all words | Prop. 2.23 | Thm. 2.26 |
| $l$ restricted to $\mathrm{dom}(M)$, $M$ prefix-free | $\mathrm{dom}(M)$, c.e. | enumerate the halting inputs | Kraft, Prop. 2.25 |
| **atomic** $F\_{i\_w}$: $F(w) = 0$, undefined elsewhere | $\lbrace w\rbrace$ | finite | one word takes the whole budget (used in Prop. 3.5) |
| $F(\mathrm{bin}(n)) = f(n) + d$, $f$ computable, $2^{-d}\sum\_n 2^{-f(n)} \le 1$ | $\mathrm{bin}(\mathbb{N})$ | computable | normalized by $d$ (exercise below) |
| $F(\mathrm{bin}(n)) = \lceil -\log W\_n \rceil$, $W\_n := \sum\_{l(\sigma) = n} 2^{-K(\sigma)}$ | $\mathrm{bin}(\mathbb{N})$ | $W\_n$ is attained at a finite stage (Thm. 3.7) | $\sum\_n W\_n = \sum\_\sigma 2^{-K(\sigma)} \le 1$ |
| ✗ $F(\sigma) = l(\sigma) - n$ on the layers $S\_n$ of a Schnorr test | $\bigcup\_n S\_n$ | c.e. | **fails**: every layer spends mass $1$ (exercise below) |

## Enumerating all information content measures

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.2 — Enumeration of all information content measures)</span></p>

There is a computable enumeration $F\_0, F\_1, \dots$ of **all** information content measures.

</div>

* **Idea (Kraft filter).** Whether a c.e. epigraph satisfies (19) is undecidable, so list *all* c.e. epigraphs and copy each one pair by pair **while the running Kraft sum** (minimal weight per distinct word) **stays $\le 1$**, stopping for good at the first violation. Genuine i.c.m.s pass unchanged; everything else is cut at a finite stage and still satisfies (19).
* **Pattern** (shared with 2.20): *can't decide the property → effectively enforce it, leaving the objects that already have it untouched.*

## The minimal information content measure

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.3 — Minimal information content measure)</span></p>

Fix the enumeration of Thm. 3.2 and set

$$\widetilde F(w) := \min_{i\,:\,w \in \mathrm{dom}(F_i)} \bigl(F_i(w) + i + 1\bigr).$$

</div>

* **Why the penalty $i + 1$.** Without it $\min\_i F\_i(w) = 0$ for every $w$ (atomic measures), Kraft sum $\infty$. With it the $i$-th measure gets the share $2^{-(i+1)}$ of the budget, $\sum\_i 2^{-(i+1)} = 1$. Note $i + 1 = l(1^i0)$: the penalty is the cost of **naming the $i$-th measure in unary** — the i.c.m. counterpart of the machine code $\rho\_M$.
* ⚠️ **$\max$ instead of $\min$ is not an i.c.m.:** infinitely many $F\_i$ are defined at any $w$, so the maximum is not in $\mathbb{N}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.4</span><span class="math-callout__name">($\widetilde F$ is an information content measure)</span></p>

$\widetilde F$ satisfies (19) and (20).

</div>

* **(19):** a minimum of exponents is a maximum of masses, and $\max \le \sum$; swap the sums, each inner Kraft sum is $\le 1$, the outer one is $\sum\_i 2^{-(i+1)} = 1$.
* **(20):** a minimum needs only **one witness** — dovetail all $F\_i(w)$ and enumerate $(w, k)$ as soon as some $i$ certifies $F\_i(w) + i + 1 \le k$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.5 — $\widetilde F$ is total)</span></p>

$\widetilde F(w)$ is defined for every $w$.

</div>

Proof: the **atomic** i.c.m. for $w$ is in the list, so the minimum runs over a non-empty set.

## $K$ is the smallest information content measure

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.6 — $\widetilde F$ coincides with $K$)</span></p>

$$K(w) = \widetilde F(w) \pm O(1) \qquad \text{for all } w.$$

</div>

* **$\widetilde F \le K + O(1)$:** $K$ is an i.c.m., so $K = F\_k$ for some $k$ and $\widetilde F \le K + k + 1$.
* **$K \le \widetilde F + O(1)$:** the **epigraph of $\widetilde F$ is a KC set** — requests $(k + 1, w)$ for all $k \ge \widetilde F(w)$ are c.e. by (20), and the geometric tails give total weight $\sum\_w 2^{-\widetilde F(w)} \le 1$ by (19); Cor. 2.28.1 finishes. Requesting the whole column instead of the unknown value $\widetilde F(w)$ costs exactly one bit.

The form you actually use:

$$\boxed{\,\text{for every i.c.m. } F \text{ there is a constant } c_F \text{ with } \quad K(w) \le F(w) + c_F \quad \text{for all } w \in \mathrm{dom}(F).\,} \tag{M}$$

(Directly: the epigraph of *any* i.c.m. is a KC set.)

* **What "smallest" means:** $K$ is the least element of the class modulo $+O(1)$ — exactly as $C$ was the least function whose sublevel sets are uniformly c.e. of size $< 2^n$, with counting traded for Kraft mass.
* **Three ways to bound $K$ from above**, each with less bookkeeping than the one before:

| Level | What you exhibit | Conclusion | Where |
| :-- | :-- | :-- | :-- |
| machine | a prefix-free $M$ with $M(\tau) = w$ | $K(w) \le l(\tau) + c\_M$ | invariance, Def. 2.19 / Thm. 2.20 |
| KC set | c.e. requests $(d\_i, \tau\_i)$ with $\sum\_i 2^{-d\_i} \le 1$ | $K(\tau\_i) \le d\_i + O(1)$ | Cor. 2.28.1 |
| i.c.m. | upper-semicomputable $F$ with $\sum 2^{-F} \le 1$ | $K(w) \le F(w) + c\_F$ | (M), Thm. 3.6 |

## Chaitin's counting theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.7 — Chaitin's counting theorem)</span></p>

There is a constant $c$ such that for all $n, r \in \mathbb{N}$

$$\#\lbrace \sigma \in \lbrace 0,1\rbrace^n : K(\sigma) \le n + K(\mathrm{bin}(n)) - r\rbrace \;\le\; 2^{\,n - r + c}. \tag{22}$$

In particular

$$\max\lbrace K(\sigma) : l(\sigma) = n\rbrace = n + K(\mathrm{bin}(n)) \pm O(1). \tag{23}$$

</div>

**Idea.** Let $W\_n := \sum\_{l(\sigma) = n} 2^{-K(\sigma)}$ be the **$K$-mass of length $n$**, so $\sum\_n W\_n \le 1$. Then $\mathrm{bin}(n) \mapsto \lceil -\log W\_n\rceil$ is an i.c.m. (Kraft by $\sum W\_n \le 1$; c.e. because $W\_n$ is a finite sum of staircases and is attained at a finite stage), and minimality (M) gives

$$W_n \;\le\; 2^{\,c}\, 2^{-K(\mathrm{bin}(n))}:$$

**all $2^n$ words of length $n$ together weigh at most a constant times the single word $\mathrm{bin}(n)$.** Markov turns the mass bound into the count bound (22): each word counted in (22) weighs $\ge 2^{-n - K(\mathrm{bin}(n)) + r}$. For (23), Thm. 2.29 is the upper half and $r := c + 1$ in (22) leaves some word of length $n$ above the threshold.

* **The reference level shifts.** For $C$ the ceiling is $l(w)$ (Cor. 2.3.1), for $K$ it is $l(w) + K(l(w))$; in both cases **each bit below the ceiling halves the count**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(3.7.1 — Almost every word is incompressible)</span></p>

$$\frac{\#\lbrace w \in \lbrace 0,1\rbrace^n : K(w) > n\rbrace}{2^n} \xrightarrow[\;n \to \infty\;]{} 1.$$

</div>

Proof: $r := K(\mathrm{bin}(n))$ in (22) makes the threshold exactly $n$ and the fraction $\le 2^{\,c - K(\mathrm{bin}(n))} \to 0$, since $K(\mathrm{bin}(n)) \to \infty$ (finitely many words below any fixed complexity).

* **Contrast with $C$:** Cor. 2.3.1 guarantees only *one* word per length with $C \ge n$; for $K$ the room above the length — $K(\mathrm{bin}(n))$ bits of it — is where most words live.

## Two exercises on the Kraft side

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Characterization of computable upper bounds for $K$)</span></p>

For a total computable $f : \mathbb{N} \to \mathbb{N}$:

$$\sum_{n} 2^{-f(n)} < \infty \quad\Longleftrightarrow\quad \exists c\ \forall n: \ K(\mathrm{bin}(n)) \le f(n) + c.$$

</div>

* **($\Rightarrow$):** normalize $f + d$ to Kraft sum $\le 1$; it is an i.c.m., apply (M). **($\Leftarrow$):** $2^{-f(n)} \le 2^c 2^{-K(\mathrm{bin}(n))}$, sum and use 2.26.
* **Dictionary.** Computable upper bounds for $K$ on $\mathbb{N}$ *are* the computable Kraft-summable functions; divergence even gives $K(\mathrm{bin}(n)) > f(n) + c$ for **infinitely many** $n$, for every $c$.

| $f(n)$ | $\sum\_n 2^{-f(n)}$ | verdict |
| :-- | :-- | :-- |
| $\log n$ | $\sum 1/n = \infty$ | ✗ — cf. Prop. 2.27 |
| $\log n + \log\log n$ | $\sum 1/(n\log n) = \infty$ | ✗ |
| $\log n + 2\log\log n$ | $\sum 1/(n\log^2 n) < \infty$ | ✓ — cf. the iterated form of 2.29 |

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Test-to-i.c.m. translation is not straightforward)</span></p>

A **Schnorr test** is a Martin-Löf test with $\mu(S\_i) = 2^{-i}$ **exactly**. Let $(S\_n)$ be uniformly c.e. prefix-free sets of words whose cylinders form a Schnorr test. Then

$$F(\sigma) := l(\sigma) - n \qquad (\sigma \in S_n)$$

is **not** an information content measure.

</div>

* **Each layer spends the whole budget:** prefix-freeness makes $\sum\_{\sigma \in S\_n} 2^{-l(\sigma)} = 2^{-n}$, so layer $n$ costs $2^n \cdot 2^{-n} = 1$ and the total is $\infty$. (⚠️ The notes print the exercise without the definition of $F$; $l(\sigma) - n$ is the intended reading.)
* **The fix — thin the layers:** claim only $i$ bits from the layer $S\_{i^2}$. Layer $i$ then costs $2^{i - i^2}$, which is summable. *A layer of measure $2^{-m}$ can buy $m$ bits, but buying all of them costs mass $1$ per layer.*
* **What it is for.** $X \in \bigcap\_i S\_{i^2}$ gives $K(X \upharpoonright n) \le n - i + O(1)$ for some prefix, for every $i$: this is the "typical ⇒ incompressible" half of Thm. 4.1 in the next chapter.

## 1-randomness

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.8 — 1-randomness)</span></p>

$X \in \lbrace 0,1\rbrace^{\omega}$ is **1-random** iff there is a constant $c$ with

$$K(X \upharpoonright n) \ge n - c \qquad \text{for all } n,$$

and **1-nonrandom** otherwise.

</div>

* **Negation:** $X$ is 1-nonrandom $\iff$ $K(X \upharpoonright n) - n$ is **unbounded below**. A bounded deficit is harmless; a deficit that grows without bound kills the sequence.
* **Why this is not vacuous for $K$.** For plain $C$, Thm. 1.3 makes the analogous definition empty. For $K$ there is room above the length (3.7, 3.7.1), so a sequence *can* stay above $n - c$ forever. Existence (and the equivalence with Martin-Löf randomness) comes from Thm. 4.1; directly: Markov on $W\_m$ shows fewer than $2^{n-c}$ words of length $n$ have a compressible prefix, and König's lemma gives a path with all prefixes incompressible.
* **Only one constant.** The slack $c$ depends on $X$ but not on $n$: the prefix-free version of the "constant-width band" that failed for $C$.

## One-glance summary

| Result | Statement | Tool |
| :-- | :-- | :-- |
| 3.1 | i.c.m. $=$ partial $F$ with $\sum 2^{-F} \le 1$ and c.e. epigraph | Kraft + upper-semicomputability, no machines |
| 3.2 | computable enumeration of all i.c.m.s | Kraft filter on all c.e. epigraphs; good streams pass unchanged |
| 3.3 / 3.4 / 3.5 | $\widetilde F = \min\_i (F\_i + i + 1)$ is a **total** i.c.m. | $\max \le \sum$, $\sum\_i 2^{-(i+1)} = 1$; atomic measures |
| 3.6 / (M) | $K = \widetilde F \pm O(1)$; hence $K \le F + c\_F$ for every i.c.m. $F$ | $K$ is in the list; epigraph of $\widetilde F$ is a KC set |
| 3.7 | $\sharp\lbrace \sigma \in 2^n : K(\sigma) \le n + K(\mathrm{bin}(n)) - r\rbrace \le 2^{n-r+c}$; $\max\_{l = n} K = n + K(\mathrm{bin}(n)) \pm O(1)$ | $W\_n \le 2^c 2^{-K(\mathrm{bin}(n))}$ via (M); Markov; 2.29 |
| 3.7.1 | fraction of length-$n$ words with $K > n$ tends to $1$ | $r := K(\mathrm{bin}(n))$, and $K(\mathrm{bin}(n)) \to \infty$ |
| Ex. (upper bounds) | computable $f$: $\sum 2^{-f(n)} < \infty \iff K(\mathrm{bin}(n)) \le f(n) + c$ | ($\Rightarrow$) (M) on $f + d$; ($\Leftarrow$) 2.26 |
| Ex. (Schnorr) | $l - n$ on $S\_n$ is **not** an i.c.m.; $l - i + O(1)$ on $S\_{i^2}$ is | each layer costs $1$; $\sum 2^{i - i^2} < \infty$ |
| 3.8 | $X$ 1-random $\iff$ $\exists c\, \forall n: K(X \upharpoonright n) \ge n - c$ | 3.7.1 removes the obstruction of Thm. 1.3; existence via König, or Thm. 4.1 |

**Punchline.** Strip $K$ down to the two properties that survived Part 2 — a c.e. epigraph and a Kraft sum $\le 1$ — and call anything with those properties an information content measure. The class can be effectively listed (3.2), its penalized lower envelope $\widetilde F$ is again a member (3.4), and $\widetilde F$ *is* $K$ up to a constant (3.6): $K$ is the **least** information content measure, a machine-free characterization that mirrors Part 1's characterization of $C$ with counting replaced by mass. The usable form (M), "$K \le F + c\_F$ for every i.c.m. $F$", is the new upper-bound tool — exhibiting a function replaces building a machine — and its first application is Chaitin's counting theorem: the mass of a length is at most the mass of its length, so at most $2^{n-r+c}$ words of length $n$ sit $r$ bits below the ceiling $n + K(\mathrm{bin}(n))$ (3.7), and almost every word has $K > l$ (3.7.1). That surplus is precisely what plain $C$ lacked, and it is what lets Def. 3.8 ask for *all* prefixes to be incompressible without running into Thm. 1.3.
