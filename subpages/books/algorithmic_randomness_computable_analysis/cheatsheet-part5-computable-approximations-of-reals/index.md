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

The inclusions are immediate from the definitions ($a\_n := \alpha \upharpoonright n$ is a nondecreasing approximation; a nondecreasing approximation is an approximation). $\Omega$ separates the first two classes (Prop. 5.11) and $1 - \Omega$ the last two (see below).

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

* **Dyadic rationals.** $0.w1000\ldots = 0.w0111\ldots$: a dyadic rational has two expansions, both computable, so Def. 5.1 is unambiguous; dyadics are identified with their finite words. Every "dyadic boundary" case distinction below comes from this double expansion.
* **Cauchy is not effective.** Def. 5.4 only asserts that $N(\epsilon)$ *exists*. The three classes differ in what stage $n$ tells you about the error $\alpha - a\_n$: a **bound** ($< 2^{-n}$, computable reals), a **sign** ($a\_n \le \alpha$, left-c.e. reals), **nothing** (c.a. reals).
* **"Left-c.e." is about the cut, not the bits.** If the *set* $A$ is c.e., then $0.A$ is left-c.e. (enumerate the ones; the partial sums climb). The converse is not claimed: left-c.e.-ness is c.e.-ness of $L(\alpha)$ (Prop. 5.9).

## Computable reals: Proposition 5.6

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.6 — Characterizations of computable reals)</span></p>

For a real $\alpha$, the following are equivalent:

- **(i)** $\alpha$ is computable;
- **(ii)** $L(\alpha)$ is computable;
- **(iii)** $\alpha$ has an effective approximation: a computable sequence $(a\_n)$ of rationals with

$$\lvert \alpha - a_n\rvert < 2^{-n} \qquad \text{for every } n. \tag{25}$$

</div>

All four implications run on the **dyadic bracket**: for $\alpha = 0.A$ and $a\_n := \alpha \upharpoonright n := 0.A(0)\cdots A(n-1)$,

$$\alpha \in \big[\, a_n,\ a_n + 2^{-n}\,\big] \qquad \text{for all } n. \tag{27}$$

* **(i) ⇒ (iii): prefixes are an effective approximation.** $a\_n = \alpha \upharpoonright n$ is computable from $n$, and (27) gives $\lvert \alpha - a\_n\rvert < 2^{-n}$ (strict, since the expansion has no all-ones tail).
* **(i) ⇒ (ii): decide $q < \alpha$ by a bracket search.** Compute $a\_0, a\_1, \ldots$ until some $N$ satisfies

  $$q < a_N \qquad\text{or}\qquad q \ge a_N + 2^{-N}. \tag{26}$$

  Left alternative: $q < a\_N \le \alpha$, answer **yes**. Right alternative: $q \ge a\_N + 2^{-N} \ge \alpha$ by (27), answer **no**. The search halts for every $q \ne \alpha$: if $q < \alpha$, then $a\_N \to \alpha$ eventually exceeds $q$; if $q > \alpha$, then $a\_N + 2^{-N} \to \alpha$ eventually drops to $q$ or below.

  ⚠️ For $q = \alpha$ neither alternative ever holds ($a\_N \le \alpha < a\_N + 2^{-N}$ for all $N$), so the notes' "for every $q$ the search halts" needs the case $\alpha \in \mathbb{Q}$ separately: then $L(\alpha)$ is decidable outright by rational comparison with the hard-coded $\alpha$. Non-uniform, exactly like the dyadic case of (iii) ⇒ (i).
* **(ii) ⇒ (i): read the prefix off the cut.**

  $$\alpha \upharpoonright n = \max\lbrace 0.\sigma : l(\sigma) = n \text{ and } 0.\sigma \in L(\alpha)\rbrace,$$

  a maximum over $2^n$ decidable candidates. (If $\alpha$ is itself a level-$n$ dyadic, the formula returns the prefix of the other expansion $w0111\ldots$ — still a computable expansion of $\alpha$.)
* **(iii) ⇒ (i): wait until the bracket settles.** If $\alpha$ is dyadic, hard-code its finite expansion. Otherwise, to get $\alpha \upharpoonright k$, search for an $n$ with

  $$\big(a_n - 2^{-n},\ a_n + 2^{-n}\big) \subseteq \Big[\, \tfrac{j}{2^k},\ \tfrac{j+1}{2^k}\Big) \qquad \text{for some } j,$$

  and output the $k$-bit binary representation of $j$. Such an $n$ exists because a non-dyadic $\alpha$ lies in the *interior* of exactly one level-$k$ interval and the error intervals shrink onto it.

  ⚠️ **Approximation ≠ prefix.** Even with $\lvert \alpha - a\_n\rvert < 2^{-n}$, $a\_n$ need not share $n$ bits with $\alpha$: near a dyadic boundary a tiny error flips the $n$-th digit. That is why one waits for a nested interval — and why the K-trivial exercise below must argue from Def. 5.1, not from (25).

**Why two directions are non-uniform.** Deciding from an approximation whether $\alpha$ is *exactly* rational or dyadic is not an effective task; the proofs sidestep it by hard-coding finitely much information. This is harmless, because "$\alpha$ is computable" asks only for *some* machine per $\alpha$.

**Relativization (Cor. 5.6.1).** (i) ⇔ (ii) uses the machine for $A$, resp. the decision procedure for $L(\alpha)$, only as a black box, so it holds relative to any oracle. With the halting problem: $A \le\_T \emptyset^{\prime} \iff L(\alpha) \le\_T \emptyset^{\prime}$. This is the bridge that Thm. 5.7 crosses.

## Computably approximable reals: Theorem 5.7

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.7 — c.a. reals are exactly $\Delta^0_2$)</span></p>

A real $\alpha = 0.A$ is computably approximable if and only if $A \in \Delta^0\_2$.

</div>

**Reminder (Post's theorem / the limit lemma).** For $A \subseteq \mathbb{N}$: $A \in \Delta^0\_2 = \Sigma^0\_2 \cap \Pi^0\_2 \iff A \le\_T \emptyset^{\prime} \iff A(n) = \lim\_{k} g(k, n)$ for some computable $g : \mathbb{N}^2 \to \lbrace 0,1\rbrace$.

* **(⇐) from a limit to an approximation.** Let $A(n) = \lim\_k g(k,n)$ and take the stage-$n$ guesses of the first $n$ bits,

  $$a_n := 0.A_n, \qquad A_n := g(n,0)\,g(n,1)\cdots g(n,n-1).$$

  This is computable, and for each $m$ the first $m$ guesses are correct from some stage on; from then on $a\_n$ and $\alpha$ share $m$ bits, so $\lvert a\_n - \alpha\rvert \le 2^{-m}$. Hence $a\_n \to \alpha$. Note what is *missing*: monotonicity and a modulus — precisely the c.a. freedom.
* **(⇒) from an approximation to $\Delta^0\_2$.** Let $\alpha = \lim\_n a\_n$. Then $L(\alpha)$ lies in both halves of level two,

  $$q \in L(\alpha) \iff \exists n\, \forall m \ge n\ (q < a_m) \quad (\Sigma^0_2), \qquad\qquad q \in L(\alpha) \iff \forall n\, \exists m \ge n\ (q < a_m) \quad (\Pi^0_2),$$

  so $L(\alpha) \in \Delta^0\_2$, i.e. $L(\alpha) \le\_T \emptyset^{\prime}$, and Cor. 5.6.1 gives $A \le\_T \emptyset^{\prime}$, i.e. $A \in \Delta^0\_2$. $\square$

  ⚠️ Both equivalences are exact for $q \ne \alpha$ but can fail at $q = \alpha$: if $a\_m \searrow \alpha$ from above, $q = \alpha$ satisfies both right-hand sides. Each formula therefore describes $L(\alpha)$ up to at most one rational, and a finite change leaves $\Sigma^0\_2$ and $\Pi^0\_2$ unchanged, so the conclusion stands. A uniform repair of the $\Sigma^0\_2$ side: $q < \alpha \iff \exists k\, \exists n\, \forall m \ge n\ (a\_m > q + 2^{-k})$.

**Reading.** "Limit of a computable sequence" is the same notion on both sides — the limit lemma for sets, Thm. 5.7 for reals. The c.a. reals are therefore also called the **$\Delta^0\_2$ reals**, and every one of them is $\le\_T \emptyset^{\prime}$.

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

### The four implications

* **(i) ⇒ (ii): monotone stages certify the cut.** With $a\_n \nearrow \alpha$, $L(\alpha) = \operatorname{dom}(f)$ for the partial computable $f(q) := \min\lbrace n : a\_n > q\rbrace$. Monotonicity is what makes "$a\_n > q$" a **certificate**: $q < a\_n \le \alpha$. For a non-monotone approximation, $a\_n > q$ certifies nothing.
* **(ii) ⇒ (i): climb along an enumeration.** Given an enumeration $q\_0, q\_1, \ldots$ of $L(\alpha)$, take the record subsequence $i\_0 := 0$, $i\_n := \min\lbrace m > i\_{n-1} : q\_m > q\_{i\_{n-1}}\rbrace$ (or simply the running maximum $a\_n := \max(q\_0, \ldots, q\_n)$): computable, nondecreasing, and converging to $\sup L(\alpha) = \alpha$.
* **(iii) ⇒ (i): halted mass only grows.** Run $\widetilde M$ for $n$ steps and add up what has halted,

  $$a_n := \sum_{\sigma \in \operatorname{dom}(\widetilde M[n]),\ l(\sigma) < n} 2^{-l(\sigma)} .$$

  A finite computation (the length cut-off), nondecreasing (halted stays halted), and $a\_n \to \alpha\_{\widetilde M}$.
* **(i) ⇒ (iii): realize a climb as halting mass (Kraft–Chaitin).** Let $\alpha \in [0,1]$ be left-c.e.
  1. **Make the climb dyadic.** $L(\alpha)$ is c.e. by (ii), so enumerate the dyadic rationals below $\alpha$ and let $b\_s$ be the maximum of $0$ and the first $s$ of them: $0 = b\_0 \le b\_1 \le \cdots \nearrow \alpha$.
  2. **Split each increment into powers of two.** $\delta\_s := b\_{s+1} - b\_s$ is dyadic, hence a *finite* sum $\delta\_s = \sum\_i 2^{-d\_{s,i}}$. Issue the requests $(d\_{s,i}, \lambda)$ — all with empty output; only the domain matters.
  3. **The Kraft sum telescopes.** $\sum\_{s,i} 2^{-d\_{s,i}} = \sum\_s (b\_{s+1} - b\_s) = \alpha \le 1$.
  4. **Kraft–Chaitin (Thm. 2.28)** yields a prefix-free $\widetilde M$ with codewords of exactly these lengths, so $\alpha\_{\widetilde M} = \sum\_{s,i} 2^{-d\_{s,i}} = \alpha$. $\square$

  *Why dyadic:* KC takes integer lengths, and a rational increment such as $\tfrac13$ is an infinite sum of powers of two. Passing from $(a\_s)$ to the dyadic approximants $(b\_s)$ makes every increment a finite batch of requests.

### Reading Proposition 5.9

* A left-c.e. real is "**mass enumerated so far**": every stage-wise nondecreasing enumeration of dyadic mass is the halting probability of some machine, and conversely. Via $\alpha\_{\widetilde M} = \lambda(\operatorname{dom}\widetilde M)$, the left-c.e. reals in $[0,1]$ are exactly the Lebesgue measures of c.e. open subsets of Cantor space.
* **Strictness at both ends** (the second is not in the notes). $\Omega$ is left-c.e. and not computable (Prop. 5.11). And $1 - \Omega$ is c.a. (negate an approximation) but **not** left-c.e.: otherwise $L(\Omega)$ and $R(\Omega) := \lbrace q : q > \Omega\rbrace$ would both be c.e.; as $\Omega$ is irrational (a rational is computable), every $q$ lies in exactly one of them, so running both enumerations decides $L(\Omega)$ and $\Omega$ would be computable. Hence computable $\subsetneq$ left-c.e. $\subsetneq$ c.a.

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

### $\Omega \le\_T \emptyset^{\prime}$: nothing specific to $\Omega$

$L(\Omega)$ is c.e. (Prop. 5.9), hence decidable from $\emptyset^{\prime}$, hence $\Omega \le\_T \emptyset^{\prime}$ by Cor. 5.6.1. (The notes phrase it through $(w\_s)$: $\emptyset^{\prime}$ answers the $\Sigma^0\_1$ question "do the first $n$ bits of $w\_s$ still change after stage $s$?") The same holds for every c.a. real (Thm. 5.7).

### $\emptyset^{\prime} \le\_T \Omega$: the gap argument

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The halting problem is Turing reducible to Chaitin's $\Omega$)</span></p>

Show $\emptyset^{\prime} \le\_T \Omega$. *Hint:* for given $e$, build a prefix-free machine $\widetilde M$ with $\widetilde M(0^e1)\downarrow \iff \Phi\_e(e)\downarrow$.

</div>

1. **Package the question as one $\widetilde U$-code.** Let $\widetilde M(0^e 1) \downarrow \iff \Phi\_e(e) \downarrow$; its domain lies in the prefix-free set $\lbrace 0^e 1 : e \in \mathbb{N}\rbrace$. By universality there is a fixed $\rho$ with $\widetilde U(\rho\, 0^e 1) = \widetilde M(0^e 1)$. Put $p\_e := \rho\, 0^e 1$ and $N := l(p\_e) = l(\rho) + e + 1$, both computable from $e$.
2. **Read an upper bound off the oracle.** From $\Omega \upharpoonright (N+1)$ form $u := 0.(\Omega \upharpoonright (N+1)) + 2^{-(N+1)}$. Then

   $$\Omega < u < \Omega + 2^{-N}.$$

3. **Wait until the unseen mass is small.** Dovetail $\widetilde U$ and let $\omega\_s := \sum\_{\sigma \in \operatorname{dom}(\widetilde U[s])} 2^{-l(\sigma)} \nearrow \Omega$. Wait for a stage $s$ with $u - \omega\_s < 2^{-N}$; it comes, since $u - \omega\_s \to u - \Omega < 2^{-N}$.
4. **Decide.** If $p\_e \in \operatorname{dom}(\widetilde U[s])$, then $\Phi\_e(e)\downarrow$. Otherwise $p\_e$ never halts: it would add $2^{-N}$ to an unseen mass $\Omega - \omega\_s < u - \omega\_s < 2^{-N}$. $\square$

* ⚠️ The naive version — "the first $N$ bits of $\Omega$ are the sum over the halting programs of length $\le N$" — is false: longer programs carry into the first $N$ bits. What is true, and enough, is the **gap**: once the unseen mass is below $2^{-N}$, no program of length $\le N$ is still pending. So $\Omega \upharpoonright (N+1)$ decides halting for *all* $\widetilde U$-programs of length $\le N$ — $\Omega$ is the halting problem in maximally compressed form.
* **Why the detour through $\widetilde M$.** $\emptyset^{\prime}$ asks about arbitrary computations $\Phi\_e(e)$, which are not codes of the prefix-free $\widetilde U$; $\widetilde M$ translates question $e$ into a single $\widetilde U$-code $p\_e$ of *known* length. The notes' "simpler solution" is the same argument, phrased as comparing the potential increment $2^{-N}$ of $p\_e$ with the current upper bound on $\Omega - \omega\_s$.
* **"In particular".** $\Omega$ is left-c.e. by Def. 5.10; were it computable, $\emptyset^{\prime} \le\_T \Omega$ would make the halting problem decidable.

### Why "slowest": Solovay domination

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Solovay domination property of $\Omega$)</span></p>

Let $\alpha = \alpha\_{\widetilde M}$ be the halting probability of a prefix-free machine, $\Omega$ that of a universal $\widetilde U$, and $(a\_n)$ any left-c.e. approximation of $\alpha$. Then there are a left-c.e. approximation $(w\_n)$ of $\Omega$ and a constant $c$ with

$$a_{n+1} - a_n \;\le\; c\,(w_{n+1} - w_n) \qquad \text{for all } n.$$

</div>

* **Locate $\alpha$ inside $\Omega$.** By universality, $\widetilde U(i\_{\widetilde M}\, \sigma) = \widetilde M(\sigma)$ for a fixed code $i\_{\widetilde M}$; let $d := l(i\_{\widetilde M})$. The codes with prefix $i\_{\widetilde M}$ contribute $\sum\_{\sigma \in \operatorname{dom}\widetilde M} 2^{-d - l(\sigma)} = 2^{-d}\alpha$, the remaining codes a left-c.e. real $\beta$ (stage sums over a c.e. set of codes). So $\Omega = 2^{-d}\alpha + \beta$.
* **Build the approximation.** Pick a left-c.e. approximation $b\_n \nearrow \beta$ and set $w\_n := 2^{-d} a\_n + b\_n$: computable, nondecreasing, $\to \Omega$.
* **Compare increments.** $w\_{n+1} - w\_n = 2^{-d}(a\_{n+1} - a\_n) + (b\_{n+1} - b\_n) \ge 2^{-d}(a\_{n+1} - a\_n)$, so $c := 2^{d}$ works. $\square$

**Reading.**

* Per step, $\Omega$ moves at least a fixed fraction of what any other halting probability moves — and by Prop. 5.9 that means of what *any* left-c.e. real moves. Large late increments are slow convergence: $\Omega$'s approximation is the **slowest** one. The constant $2^{d}$ is the cost of coding $\widetilde M$ inside $\widetilde U$.
* The approximation $(w\_n)$ is tailored to the given $(a\_n)$; the statement is existential in $(w\_n)$, not about a fixed approximation of $\Omega$.
* Forward reference: this relation is **Solovay reducibility** $\alpha \le\_S \Omega$, and the exercise says that every left-c.e. real is Solovay-reducible to $\Omega$ (chapter Solovay Reducibility).

## Exercise: computable ⇒ K-trivial ⇒ computably approximable

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definitions</span><span class="math-callout__name">(Trees, isolated paths, K-triviality)</span></p>

* $T \subseteq \lbrace 0,1\rbrace^{\ast}$ is a **binary tree** if it is closed under prefixes: $w \in T$ and $v \sqsubseteq w$ imply $v \in T$.
* $A$ is a **path** in $T$ if $A \upharpoonright n \in T$ for every $n$; $[T]$ denotes the set of paths.
* A path $A$ is **isolated** if for some $n$ no other path $B \ne A$ has $B \upharpoonright n = A \upharpoonright n$, i.e. $[A \upharpoonright n] \cap [T] = \lbrace A\rbrace$.
* A real $\alpha$ (or sequence) is **K-trivial** if there is $c$ with $K(\alpha \upharpoonright n) \le K(\mathrm{bin}(n)) + c$ for all $n$.

</div>

**Where K-trivial sits.** For every $A$ and $n$, $K(\mathrm{bin}(n)) - O(1) \le K(A \upharpoonright n) \le n + K(\mathrm{bin}(n)) + O(1)$ — the length is computable from the prefix, and Thm. 3.7 gives the maximum. 1-random sequences sit at the top ($K(A \upharpoonright n) \ge n - c$, Def. 3.8); K-trivial ones sit at the bottom: each prefix carries no more information than its length.

1. **Computable ⇒ K-trivial.** Let $\alpha = 0.A$ with $A$ computable. From $\mathrm{bin}(n)$ the machine for $A$ prints $A \upharpoonright n = \alpha \upharpoonright n$, so $K(\alpha \upharpoonright n) \le K(\mathrm{bin}(n)) + O(1)$; conversely $\mathrm{bin}(n)$ is computable from $\alpha \upharpoonright n$, so $K(\mathrm{bin}(n)) \le K(\alpha \upharpoonright n) + O(1)$. Hence $K(\alpha \upharpoonright n) = K(\mathrm{bin}(n)) \pm O(1)$. The constant is the code of the (fixed, $\alpha$-dependent) machine.

   ⚠️ Argue from Def. 5.1, not from an effective approximation (25): $a\_n$ need not share $n$ bits with $\alpha$ (dyadic boundary).
2. **$T\_c := \lbrace \sigma : \forall \tau \sqsubseteq \sigma\ \ K(\tau) \le K(\mathrm{bin}(l(\tau))) + c\rbrace$ is a tree computable in $\emptyset^{\prime}$.** Prefix-closed by the "$\forall \tau$". And $K \le\_T \emptyset^{\prime}$: "$K(\tau) \le k$" is $\Sigma^0\_1$ (some program of length $\le k$ halts with output $\tau$), so $\emptyset^{\prime}$ answers it for $k = 0, 1, 2, \ldots$ until the first yes. Membership in $T\_c$ is finitely many such queries.
3. **The levels of $T\_c$ are uniformly bounded.** Chaitin's counting theorem (3.7) in uniform form: there is $d$ with

   $$\sharp\lbrace \sigma \in \lbrace 0,1\rbrace^n : K(\sigma) \le n + K(\mathrm{bin}(n)) - r\rbrace \le 2^{\,n - r + d} \qquad \text{for all } n, r.$$

   For $\sigma \in T\_c$ of length $n$, $K(\sigma) \le K(\mathrm{bin}(n)) + c = n + K(\mathrm{bin}(n)) - (n - c)$. Choose $r := n - c$: the level has at most $2^{c + d} =: N$ nodes. Letting $r$ depend on $n$ is exactly what cancels the $n$-dependence.
4. **A K-trivial sequence is an isolated path of $T\_c$.** $A \upharpoonright n \in T\_c$ for all $n$, so $A$ is a path. If $A$ were not isolated, then beyond every level another path leaves $A$, so $A$ has infinitely many splitting points; take $N$ of them, $s\_1 < \cdots < s\_N$, follow the side branch at each to a path $B\_j$, and pick a level $L > s\_N$. The $N + 1$ words $A \upharpoonright L, B\_1 \upharpoonright L, \ldots, B\_N \upharpoonright L$ are distinct nodes of $T\_c$ (branches of a prefix tree never merge) — contradicting step 3.
5. **An isolated path of a computable tree is computable.** Let $\rho \sqsubset A$ be an isolating prefix and hard-code it. On input $n \ge l(\rho)$, search for a level $m \ge n$ at which all $\sigma \in T$ with $l(\sigma) = m$ and $\rho \sqsubseteq \sigma$ agree on their first $n$ bits; output that common prefix — it is $A \upharpoonright n$, since $A \upharpoonright m$ is among the nodes. The search halts: otherwise some $n$-bit word $\tau \ne A \upharpoonright n$ (finitely many candidates) has nodes above it at infinitely many levels, hence at all levels, and König's lemma gives a second path through $\rho$.
6. **K-trivial ⇒ c.a.** Let $\alpha = 0.A$ be K-trivial with constant $c$. By step 4, $A$ is an isolated path of $T\_c$; by step 2, $T\_c \le\_T \emptyset^{\prime}$; step 5 relativized to $\emptyset^{\prime}$ gives $A \le\_T \emptyset^{\prime}$, i.e. $A \in \Delta^0\_2$. By Thm. 5.7, $\alpha$ is computably approximable. $\square$

**Reading.**

* **Finite evidence versus infinite properties.** "$K(\sigma) \le k$" has a finite witness (a short halting program) and is c.e.; "$K(\sigma) > k$" has none. "$\sigma \in T$" is finite; "$\sigma$ has an infinite extension in $T$" is not. Where the finite evidence runs out, the proof reaches for an oracle ($\emptyset^{\prime}$ in step 2), compactness (König in step 5), or finite advice ($\rho$ in step 5).
* **Non-uniformity.** Step 5 proves "for each isolated path there is a machine", not "from $T$, list the isolated paths". The isolating prefix is used, never found.
* **What the exercise does not say.** It lands K-trivial reals in $\Delta^0\_2$, one level above the computable reals; it does not claim that they are computable.

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
| 5.6 | computable $\iff$ $L(\alpha)$ computable $\iff$ effective approximation $\lvert \alpha - a\_n\rvert < 2^{-n}$ | dyadic bracket (27); bracket search (26); non-uniform for rational/dyadic $\alpha$ |
| 5.6.1 | $A \le\_T \emptyset^{\prime} \iff L(\alpha) \le\_T \emptyset^{\prime}$ | relativize (i) ⇔ (ii) |
| 5.7 | c.a. $\iff A \in \Delta^0\_2$ | limit lemma; $L(\alpha) \in \Sigma^0\_2 \cap \Pi^0\_2$ (up to the point $q = \alpha$); 5.6.1 |
| 5.8 | $\alpha\_{\widetilde M} = \sum\_{\sigma \in \operatorname{dom}\widetilde M} 2^{-l(\sigma)} = \lambda(\operatorname{dom}\widetilde M) \le 1$ | Kraft 2.25 |
| 5.9 | left-c.e. $\iff$ $L(\alpha)$ c.e. $\iff$ halting probability | $f(q) = \min\lbrace n : a\_n > q\rbrace$; running max; stage sums; dyadic climb + Kraft–Chaitin 2.28 |
| 5.10 | $\Omega = \alpha\_{\widetilde U}$, approximation $w\_s$ = mass halted by stage $s$ | left-c.e. by construction |
| 5.11 | $\emptyset^{\prime} \equiv\_T \Omega$; $\Omega$ left-c.e. and noncomputable | $\le$: c.e. cut + 5.6.1; $\ge$: gap argument |
| Ex. ($\emptyset^{\prime} \le\_T \Omega$) | $\Omega \upharpoonright (N+1)$ decides halting of all $\widetilde U$-codes of length $\le N$ | $\widetilde M(0^e1)\downarrow \iff \Phi\_e(e)\downarrow$; wait until $u - \omega\_s < 2^{-N}$ |
| Ex. (Solovay domination) | $a\_{n+1} - a\_n \le 2^{d}(w\_{n+1} - w\_n)$ for a suitable $(w\_n) \nearrow \Omega$ | $\Omega = 2^{-d}\alpha + \beta$; $w\_n := 2^{-d}a\_n + b\_n$ |
| Ex. (K-trivial), 1 | computable $\Rightarrow K(\alpha \upharpoonright n) = K(\mathrm{bin}(n)) \pm O(1)$ | the machine for $A$; length from prefix |
| Ex. (K-trivial), 2–3 | $T\_c$ is an $\emptyset^{\prime}$-computable tree with levels $\le 2^{c+d}$ | $K \le\_T \emptyset^{\prime}$; counting theorem 3.7 with $r = n - c$ |
| Ex. (K-trivial), 4–5 | K-trivial $A$ is an isolated path of $T\_c$; isolated paths of computable trees are computable | pigeonhole on a level; König + hard-coded $\rho$ |
| Ex. (K-trivial), 6 | K-trivial $\Rightarrow$ c.a. | relativize 5 to $\emptyset^{\prime}$; 5.7 |

**Punchline.** A real is as computable as its left cut, and the left cut is as computable as the best approximation one can run: an error *bound* at every stage makes $L(\alpha)$ decidable and $\alpha$ computable; a known *sign* of the error — approximation from below — makes $L(\alpha)$ c.e. and $\alpha$ the halting probability of a prefix-free machine, Kraft–Chaitin turning every dyadic climb into halting programs; no information at all still leaves $L(\alpha)$ in $\Delta^0\_2$, the limit lemma on the real line. Chaitin's $\Omega$ lives in the middle layer and marks its edge: left-c.e. by construction, Turing-equivalent to the halting problem because the unseen halting mass below $2^{-N}$ cannot hide a program of length $N$, and the slowest of all left-c.e. approximations because every other halting probability is a scaled part of it. At the opposite end, the K-trivial reals — prefixes no more complex than their lengths — are trapped as isolated paths of a thin $\emptyset^{\prime}$-computable tree and so are $\Delta^0\_2$ as well. The chapter's hierarchy, **computable $\subsetneq$ left-c.e. $\subsetneq$ c.a.**, is the stage on which the randomness of $\Omega$ is played out next.
