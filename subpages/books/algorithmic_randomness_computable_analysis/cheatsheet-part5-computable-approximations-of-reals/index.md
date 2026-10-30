---
title: "Cheatsheet — Part 5: Computable Approximations of Real Numbers"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - computable-analysis
  - computability
  - chaitin-constant
  - halting-problem
  - kraft-chaitin
  - kolmogorov-complexity
  - k-triviality
  - cheatsheet
---

# Cheatsheet — Part 5: Computable Approximations of Real Numbers

*Exam recap: definitions, formulas and key results only, covering the chapter Computable Approximations of Real Numbers (results 5.1–5.11 and the three exercises: $\emptyset^{\prime} \le\_T \Omega$, Solovay domination, and computable ⇒ K-trivial ⇒ c.a.) up to (but excluding) Basics of Computability Theory and Prefix-Free Machines. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}#computable-approximations-of-real-numbers). Prerequisites in [Part 1]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part1-plain-complexity/' | relative_url }}), [Part 2]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part2-prefix-free-complexity/' | relative_url }}), [Part 3]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part3-information-content-measures/' | relative_url }}) and [Part 4]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part4-incompressibility-typicality/' | relative_url }}).*

## Where we are

Parts 1–4 defined randomness for infinite binary sequences and proved that incompressibility and typicality agree (Thm. 4.1). This chapter moves to the **real line**: a real $\alpha \in [0,1]$ is identified with its binary expansion, $\alpha = 0.A$, and the question becomes which reals an algorithm can *approximate*, and how well. The destination is Chaitin's $\Omega$ (Def. 5.10), the real whose expansion is the promised "natural" random sequence — the limit of the **slowest** nondecreasing computable Cauchy sequence. Its randomness is proved later (Thm. 6.2); here it is placed in the landscape of approximable reals.

**The dictionary sequence ↔ real.**

| sequence side | real side |
| :-- | :-- |
| $A \in \lbrace 0,1\rbrace^{\mathbb{N}}$ | $\alpha = 0.A \in [0,1]$ |
| prefix $A \upharpoonright n$ | dyadic $a\_n = 0.(A \upharpoonright n)$, with $\alpha \in [a\_n,\ a\_n + 2^{-n}]$ |
| $A$ computable | $\alpha$ computable (Def. 5.1) |
| $A \in \Delta^0\_2$, i.e. $A \le\_T \emptyset^{\prime}$ | $\alpha$ computably approximable (Thm. 5.7) |
| a set of rationals | the **left cut** $L(\alpha) = \lbrace q \in \mathbb{Q} : q < \alpha\rbrace$ (Def. 5.2), with $\alpha = \sup L(\alpha)$ |

**The organizing principle.** The left cut turns a real into a *set of rationals*, so the complexity classes for sets — computable, c.e., $\Delta^0\_2$ — become classes of reals, and each matches exactly one kind of approximation. The chapter is three "the following are equivalent" results, one per row:

| class of reals | approximation $(a\_n)$, computable with | left cut $L(\alpha)$ | third face | result |
| :-- | :-- | :-- | :-- | :-- |
| **computable** | $\lvert \alpha - a\_n\rvert < 2^{-n}$ (effective) | computable | $\alpha = 0.A$ with $A$ computable | Prop. 5.6 |
| **left-c.e.** | $a\_n$ nondecreasing | c.e. | $\alpha = \lambda(\operatorname{dom}\widetilde M)$, a halting probability | Prop. 5.9 |
| **computably approximable** (c.a.) | nothing beyond Cauchy | $\Delta^0\_2$ | $A \in \Delta^0\_2$: $A(n) = \lim\_k g(k,n)$ | Thm. 5.7 |

$$\text{computable} \;\subsetneq\; \text{left-c.e.} \;\subsetneq\; \text{computably approximable} \;=\; \Delta^0_2\text{-reals}.$$

The inclusions are immediate from the definitions. $\Omega$ separates the first two classes (Prop. 5.11) and $1 - \Omega$ the last two (see below).

## Vocabulary (5.1–5.5)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definitions</span><span class="math-callout__name">(5.1–5.5 — Reals, cuts, approximations)</span></p>

* **(5.1)** $\alpha$ is **computable** if $\alpha = 0.A$ for a computable binary sequence $A$.
* **(5.2)** The **left cut** of $\alpha$ is $L(\alpha) := \lbrace q \in \mathbb{Q} : q < \alpha\rbrace$.
* **(5.3)** A sequence of rationals $(q\_n)$ is **computable** if some Turing machine returns $q\_i$ on input $i$.
* **(5.4)** $(a\_n)$ is a **Cauchy sequence** if for every $\epsilon > 0$ there is $N$ with $\lvert a\_n - a\_m\rvert < \epsilon$ for all $m, n \ge N$.
* **(5.5)** $\alpha$ is **computably approximable** (c.a.) if some computable Cauchy sequence converges to $\alpha$, and **left computably enumerable** (left-c.e.) if some *nondecreasing* computable Cauchy sequence does. Such sequences are called (left-c.e.) **approximations** of $\alpha$.
* An approximation with $\lvert \alpha - a\_n\rvert < 2^{-n}$ for every $n$ is an **effective approximation** (25).
* A set is **computable** if a Turing machine computes its characteristic function and halts on every input; rationals are coded as reduced pairs $(a, b)$.

</div>

* **Dyadic rationals.** $0.w1000\ldots = 0.w0111\ldots$: two expansions, both computable, so Def. 5.1 is unambiguous. Every "dyadic boundary" case distinction below comes from this double expansion.
* **Cauchy is not effective.** The three classes differ in what stage $n$ tells you about the error $\alpha - a\_n$: a **bound** ($< 2^{-n}$, computable reals), a **sign** ($a\_n \le \alpha$, left-c.e. reals), **nothing** (c.a. reals).
* **"Left-c.e." is about the cut, not the bits.** $A$ c.e. $\Rightarrow 0.A$ left-c.e.; the converse is not claimed.

## Computable reals: Proposition 5.6

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.6 — Characterizations of computable reals)</span></p>

For a real $\alpha$, the following are equivalent:

- **(i)** $\alpha$ is computable;
- **(ii)** $L(\alpha)$ is computable;
- **(iii)** $\alpha$ has an effective approximation: a computable sequence $(a\_n)$ of rationals with

$$\lvert \alpha - a_n\rvert < 2^{-n} \qquad \text{for every } n. \tag{25}$$

</div>

Everything runs on the **dyadic bracket** $\alpha \in [\alpha \upharpoonright n,\ \alpha \upharpoonright n + 2^{-n}]$ (27).

* **(i) ⇒ (iii):** the prefixes themselves. **(i) ⇒ (ii):** compute prefixes until $q$ falls left of the bracket (yes) or right of it (no); halts for every $q \ne \alpha$. **(ii) ⇒ (i):** $\alpha \upharpoonright n$ is the largest level-$n$ dyadic in $L(\alpha)$ — a max over $2^n$ decidable candidates. **(iii) ⇒ (i):** wait until the error interval $(a\_n \pm 2^{-n})$ sits inside a single level-$k$ dyadic interval and read off its index; a non-dyadic $\alpha$ lies in the interior of exactly one, so the wait ends.
* **Non-uniformity.** Rational / dyadic $\alpha$ are handled by hard-coding finitely much information; harmless, since "computable" asks only for *some* machine per $\alpha$.
* ⚠️ **Approximation ≠ prefix.** $\lvert \alpha - a\_n\rvert < 2^{-n}$ does not make $a\_n$ share $n$ bits with $\alpha$ (dyadic boundary) — hence the waiting in (iii) ⇒ (i), and hence the K-trivial exercise argues from Def. 5.1, not from (25).

**Relativization (Cor. 5.6.1).** (i) ⇔ (ii) uses the machine for $A$, resp. the decision procedure for $L(\alpha)$, only as a black box, so it holds relative to any oracle: $A \le\_T \emptyset^{\prime} \iff L(\alpha) \le\_T \emptyset^{\prime}$. This is the bridge that Thm. 5.7 crosses.

## Computably approximable reals: Theorem 5.7

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.7 — c.a. reals are exactly $\Delta^0_2$)</span></p>

A real $\alpha = 0.A$ is computably approximable if and only if $A \in \Delta^0\_2$.

</div>

**Reminder (limit lemma).** $A \in \Delta^0\_2 = \Sigma^0\_2 \cap \Pi^0\_2 \iff A \le\_T \emptyset^{\prime} \iff A(n) = \lim\_{k} g(k, n)$ for some computable $g$.

* **(⇐):** take as $a\_n$ the stage-$n$ guess of the first $n$ bits; each finite block of guesses is eventually correct, so $a\_n \to \alpha$ — with no monotonicity and no modulus, exactly the c.a. freedom.
* **(⇒):** $q < \alpha \iff \exists n\, \forall m \ge n\ (q < a\_m) \iff \forall n\, \exists m \ge n\ (q < a\_m)$ puts $L(\alpha)$ in $\Sigma^0\_2 \cap \Pi^0\_2$ (up to the single point $q = \alpha$, which is harmless), so $L(\alpha) \le\_T \emptyset^{\prime}$ and Cor. 5.6.1 transfers this to $A$.

**Reading.** "Limit of a computable sequence" is the same notion on both sides — the limit lemma for sets, Thm. 5.7 for reals. Every c.a. real is $\le\_T \emptyset^{\prime}$.

## Left-c.e. reals: Definition 5.8 and Proposition 5.9

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.8 — Halting probability)</span></p>

For a prefix-free machine $\widetilde M$,

$$\alpha_{\widetilde M} := \sum_{\sigma \in \operatorname{dom}(\widetilde M)} 2^{-l(\sigma)} = \lambda\big(\operatorname{dom}(\widetilde M)\big) \le 1$$

is its **halting probability** — the measure of the domain, at most $1$ by Kraft (Prop. 2.25).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.9 — Characterizations of left-c.e. reals)</span></p>

For a real $\alpha$ (in $[0,1]$ for (iii)), the following are equivalent:

- **(i)** $\alpha$ is left-c.e.;
- **(ii)** $L(\alpha)$ is c.e.;
- **(iii)** $\alpha = \alpha\_{\widetilde M}$ is the halting probability of some prefix-free machine $\widetilde M$.

</div>

* **(i) ⇒ (ii):** with $a\_n \nearrow \alpha$, "$a\_n > q$" is a **certificate** for $q < \alpha$ — monotonicity is what makes it one.
* **(ii) ⇒ (i):** running maximum of an enumeration of $L(\alpha)$.
* **(iii) ⇒ (i):** halted mass only grows — stage sums $\sum\_{\sigma \in \operatorname{dom}\widetilde M[n],\ l(\sigma) < n} 2^{-l(\sigma)}$.
* **(i) ⇒ (iii), Kraft–Chaitin:** climb along *dyadic* lower bounds $b\_s \nearrow \alpha$; each increment $b\_{s+1} - b\_s$ is a finite sum of powers of two, so request codewords of those lengths (output irrelevant). The Kraft sum telescopes to $\alpha \le 1$, and KC delivers a machine whose halting probability is exactly $\alpha$. *Dyadic* because KC takes integer lengths.

**Reading 5.9.**

* A left-c.e. real is "**mass enumerated so far**"; the left-c.e. reals in $[0,1]$ are exactly the Lebesgue measures of c.e. open subsets of Cantor space.
* **Strictness at both ends.** $\Omega$ is left-c.e. and not computable (Prop. 5.11). $1 - \Omega$ is c.a. but **not** left-c.e.: otherwise both cuts of the irrational $\Omega$ would be c.e. and $\Omega$ computable.

## Chaitin's $\Omega$: Definition 5.10 and Proposition 5.11

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.10 — Chaitin's $\Omega$)</span></p>

For a universal prefix-free machine $\widetilde U$,

$$\Omega := \sum_{\sigma \in \operatorname{dom}(\widetilde U)} 2^{-l(\sigma)}, \qquad w_s := \sum_{\sigma \in \operatorname{dom}(\widetilde U[s]),\ l(\sigma) \le s} 2^{-l(\sigma)} .$$

$(w\_s)$ is a nondecreasing computable approximation of $\Omega$, so $\Omega$ is **left-c.e.**

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.11 — $\Omega$ and the halting problem)</span></p>

$$\emptyset' \equiv_T \Omega .$$

In particular $\Omega$ is a left-c.e. real that is **not computable**.

</div>

* **$\Omega \le\_T \emptyset^{\prime}$:** nothing specific to $\Omega$ — $L(\Omega)$ is c.e., hence decidable from $\emptyset^{\prime}$, and Cor. 5.6.1. The same holds for every c.a. real.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The halting problem is Turing reducible to Chaitin's $\Omega$)</span></p>

Show $\emptyset^{\prime} \le\_T \Omega$. *Hint:* for given $e$, build a prefix-free machine $\widetilde M$ with $\widetilde M(0^e1)\downarrow \iff \Phi\_e(e)\downarrow$.

</div>

* **Idea.** Connect two views of $\Omega$: the oracle gives **any precision of $\Omega$**, and running $\widetilde U$ tracks the **halting-mass partial sums in real time**; together they bound the mass still to come. Package "$\Phi\_e(e)\downarrow$?" as one $\widetilde U$-code $\rho\,0^e1$ of known length $N$. Once the remaining mass is below $2^{-N}$, that code has either already halted or never will — its contribution $2^{-N}$ no longer fits.
* **Reading.** The first $N$ bits of $\Omega$ are *not* the sum over halting programs of length $\le N$ (longer programs carry). What is true is the **gap**: $\Omega \upharpoonright (N+1)$ decides halting for *all* $\widetilde U$-programs of length $\le N$ — $\Omega$ is the halting problem in maximally compressed form.

### Why "slowest": Solovay domination

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Solovay domination property of $\Omega$)</span></p>

Let $\alpha = \alpha\_{\widetilde M}$ be the halting probability of a prefix-free machine, $\Omega$ that of a universal $\widetilde U$, and $(a\_n)$ any left-c.e. approximation of $\alpha$. Then there are a left-c.e. approximation $(w\_n)$ of $\Omega$ and a constant $c$ with

$$a_{n+1} - a_n \;\le\; c\,(w_{n+1} - w_n) \qquad \text{for all } n.$$

</div>

* **Idea.** $\widetilde M$ sits inside $\widetilde U$ behind a code of length $d$, so $\Omega = 2^{-d}\alpha + \beta$ with $\beta$ (the mass of all other codes) left-c.e.; take $w\_n := 2^{-d}a\_n + b\_n$ for any $b\_n \nearrow \beta$ and compare increments, $c = 2^d$.
* **Reading.** Per step, $\Omega$ moves at least a fixed fraction of what *any* left-c.e. real moves (Prop. 5.9): large late increments are slow convergence, so $\Omega$'s approximation is the **slowest**. Forward: this is Solovay reducibility $\alpha \le\_S \Omega$ (Part 7).

## Exercise: computable ⇒ K-trivial ⇒ computably approximable

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definitions</span><span class="math-callout__name">(Trees, isolated paths, K-triviality)</span></p>

* $T \subseteq \lbrace 0,1\rbrace^{\ast}$ is a **binary tree** if it is closed under prefixes: $w \in T$ and $v \sqsubseteq w$ imply $v \in T$.
* $A$ is a **path** in $T$ if $A \upharpoonright n \in T$ for every $n$; $[T]$ denotes the set of paths.
* A path $A$ is **isolated** if for some $n$ no other path $B \ne A$ has $B \upharpoonright n = A \upharpoonright n$, i.e. $[A \upharpoonright n] \cap [T] = \lbrace A\rbrace$.
* A real $\alpha$ (or sequence) is **K-trivial** if there is $c$ with $K(\alpha \upharpoonright n) \le K(\mathrm{bin}(n)) + c$ for all $n$.

</div>

**Where K-trivial sits.** $K(\mathrm{bin}(n)) - O(1) \le K(A \upharpoonright n) \le n + K(\mathrm{bin}(n)) + O(1)$ for every $A$ (length is computable from the prefix; Thm. 3.7 for the max). 1-random sequences sit at the top, K-trivial ones at the bottom: each prefix carries no more information than its length.

* **Computable ⇒ K-trivial.** Prefix and length are computable from each other, so $K(\alpha \upharpoonright n) = K(\mathrm{bin}(n)) \pm O(1)$. (Argue from Def. 5.1, not from (25).)
* **K-trivial ⇒ c.a.** The tree $T\_c$ of words all of whose prefixes are K-trivial with constant $c$ is **computable in $\emptyset^{\prime}$** ("$K(\tau) \le k$" is $\Sigma^0\_1$) and has **uniformly bounded levels**: Chaitin's counting theorem with $r := n - c$ gives at most $2^{c+d}$ nodes per level. A K-trivial $A$ is a path of $T\_c$, and bounded width forces it to be **isolated** (infinitely many splittings would put more than $2^{c+d}$ nodes on one level). An isolated path of a computable tree is computable — hard-code an isolating prefix and wait until all nodes above it agree on $n$ bits (König: otherwise a second path) — and relativized to $\emptyset^{\prime}$ this gives $A \in \Delta^0\_2$, so $\alpha$ is c.a. by Thm. 5.7.
* **Reading.** Where finite evidence runs out ("$K > k$", "has an infinite extension") the proof reaches for an oracle, compactness, or finite advice. The exercise lands K-trivial reals in $\Delta^0\_2$; it does not claim they are computable.

## The three classes side by side

| | computable | left-c.e. | computably approximable |
| :-- | :-- | :-- | :-- |
| approximation | $\lvert \alpha - a\_n\rvert < 2^{-n}$ | $a\_n \nearrow \alpha$ | $a\_n \to \alpha$ |
| what a stage certifies | the error bound | the sign: $a\_n \le \alpha$ | nothing |
| left cut $L(\alpha)$ | computable | c.e. | $\Delta^0\_2$ |
| expansion $A$ | computable | — | $\Delta^0\_2$, i.e. $\le\_T \emptyset^{\prime}$ |
| extra face | $\alpha = 0.A$ | halting probability $\lambda(\operatorname{dom}\widetilde M)$ | $A(n) = \lim\_k g(k,n)$ |
| separating example | rationals, dyadics | $\Omega$ | $1 - \Omega$ |
| closed under | $+$, $-$, $\max$ | $+$, $\max$ (not $-$) | $+$, $-$ |
| result | Prop. 5.6 | Prop. 5.9 | Thm. 5.7 |

## One-glance summary

| Result | Statement | Tool |
| :-- | :-- | :-- |
| 5.1–5.5 | computable real $= 0.A$ with $A$ computable; $L(\alpha) = \lbrace q < \alpha\rbrace$; c.a. $=$ limit of a computable Cauchy sequence; left-c.e. $=$ of a nondecreasing one | definitions |
| 5.6 | computable $\iff$ $L(\alpha)$ computable $\iff$ effective approximation $\lvert \alpha - a\_n\rvert < 2^{-n}$ | dyadic bracket (27); bracket search; non-uniform for rational/dyadic $\alpha$ |
| 5.6.1 | $A \le\_T \emptyset^{\prime} \iff L(\alpha) \le\_T \emptyset^{\prime}$ | relativize (i) ⇔ (ii) |
| 5.7 | c.a. $\iff A \in \Delta^0\_2$ | limit lemma; $L(\alpha) \in \Sigma^0\_2 \cap \Pi^0\_2$; 5.6.1 |
| 5.8 | $\alpha\_{\widetilde M} = \sum\_{\sigma \in \operatorname{dom}\widetilde M} 2^{-l(\sigma)} = \lambda(\operatorname{dom}\widetilde M) \le 1$ | Kraft 2.25 |
| 5.9 | left-c.e. $\iff$ $L(\alpha)$ c.e. $\iff$ halting probability | monotone certificates; running max; stage sums; dyadic climb + Kraft–Chaitin 2.28 |
| 5.10 | $\Omega = \alpha\_{\widetilde U}$, approximation $w\_s$ = mass halted by stage $s$ | left-c.e. by construction |
| 5.11 | $\emptyset^{\prime} \equiv\_T \Omega$; $\Omega$ left-c.e. and noncomputable | $\le$: c.e. cut + 5.6.1; $\ge$: gap argument |
| Ex. ($\emptyset^{\prime} \le\_T \Omega$) | $\Omega \upharpoonright (N+1)$ decides halting of all $\widetilde U$-codes of length $\le N$ | code $\rho 0^e1$ of known length; wait until the remaining mass is $< 2^{-N}$ |
| Ex. (Solovay domination) | $a\_{n+1} - a\_n \le 2^{d}(w\_{n+1} - w\_n)$ for a suitable $(w\_n) \nearrow \Omega$ | $\Omega = 2^{-d}\alpha + \beta$; $w\_n := 2^{-d}a\_n + b\_n$ |
| Ex. (K-trivial) | computable $\Rightarrow$ K-trivial $\Rightarrow$ c.a. | prefix ↔ length; $T\_c$ is a thin $\emptyset^{\prime}$-computable tree (3.7 with $r = n - c$); isolated path ⇒ computable, relativized |

**Punchline.** A real is as computable as its left cut, and the left cut is as computable as the best approximation one can run: an error *bound* at every stage makes $L(\alpha)$ decidable and $\alpha$ computable; a known *sign* of the error — approximation from below — makes $L(\alpha)$ c.e. and $\alpha$ the halting probability of a prefix-free machine, Kraft–Chaitin turning every dyadic climb into halting programs; no information at all still leaves $L(\alpha)$ in $\Delta^0\_2$, the limit lemma on the real line. Chaitin's $\Omega$ lives in the middle layer and marks its edge: left-c.e. by construction, Turing-equivalent to the halting problem because the unseen halting mass below $2^{-N}$ cannot hide a program of length $N$, and the slowest of all left-c.e. approximations because every other halting probability is a scaled part of it. At the opposite end, the K-trivial reals — prefixes no more complex than their lengths — are trapped as isolated paths of a thin $\emptyset^{\prime}$-computable tree and so are $\Delta^0\_2$ as well. The chapter's hierarchy, **computable $\subsetneq$ left-c.e. $\subsetneq$ c.a.**, is the stage on which the randomness of $\Omega$ is played out next.
