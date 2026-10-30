---
title: "Cheatsheet — Part 6: Randomness of Reals"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - computable-analysis
  - computability
  - chaitin-constant
  - halting-problem
  - martin-löf-test
  - 1-randomness
  - kolmogorov-complexity
  - recursion-theorem
  - cheatsheet
---

# Cheatsheet — Part 6: Randomness of Reals

*Exam recap: definitions, formulas and key results only, covering the chapters Basics of Computability Theory and Prefix-Free Machines (the toolkit 5.12–5.15) and Randomness of Reals (6.1–6.2: random reals, and the randomness of Chaitin's $\Omega$) up to (but excluding) Solovay Reducibility. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}#randomness-of-reals). Prerequisites in [Part 1]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part1-plain-complexity/' | relative_url }}), [Part 2]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part2-prefix-free-complexity/' | relative_url }}), [Part 3]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part3-information-content-measures/' | relative_url }}), [Part 4]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part4-incompressibility-typicality/' | relative_url }}) and [Part 5]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part5-computable-approximations-of-reals/' | relative_url }}).*

## Where we are

Parts 1–4 defined randomness for infinite sequences and proved that its two faces agree: $X$ is Martin-Löf random (no effective null test catches it, Def. 1.5) iff $X$ is 1-random ($K(X \upharpoonright n) \ge n - c$, Def. 3.8) — Thm. 4.1. Part 5 moved to reals via $\alpha = 0.A$, built the hierarchy computable $\subsetneq$ left-c.e. $\subsetneq$ c.a., and placed Chaitin's $\Omega$ (Def. 5.10) in the middle layer: left-c.e., Turing-equivalent to $\emptyset^{\prime}$ (Prop. 5.11), and the *slowest* left-c.e. approximation (Solovay domination). What was still missing is the promised **natural example of a random object**. This part delivers it:

$$\Omega \text{ is Martin-Löf random} \qquad (\text{Thm. 6.2}).$$

The proof needs one classical tool — a machine that can refer to **its own index** — so a short toolkit chapter (5.12–5.15) comes first.

| step | result | what it contributes |
| :-- | :-- | :-- |
| toolkit | 5.12 $s$-$m$-$n$ → 5.13 fixed point → 5.14 Kleene II → 5.15 prefix-free recursion | an index $e$ with $\widetilde M\_e = h(\cdot, e)$ for any uniform prefix-free family $h$ |
| definition | 6.1 random real | randomness of $\alpha$ := randomness of its expansion; tests on $[0,1]$ |
| remark | computable reals are not random | the one-interval test $[[\alpha \upharpoonright i]]$ |
| theorem | 6.2 $\Omega$ is random | a short code for $\Omega \upharpoonright n$ would force a late halting event of mass $\ge 2^{-n}$ |

**The one inequality behind Part 5 and Part 6.** If $w\_s \upharpoonright n = \Omega \upharpoonright n$, then $0 \le \Omega - w\_s < 2^{-n}$: no $\widetilde U$-code of length $\le n$ can still be pending at stage $s$. Prop. 5.11 used this *positively* (read the halting problem off $\Omega$); Thm. 6.2 uses it *negatively* (a short description of $\Omega \upharpoonright n$ would create such a pending code).

## The toolkit: machines that know their own index (5.12–5.15)

**Notation.** $M\_0, M\_1, \ldots$ is the standard enumeration of all Turing machines on words; $\widetilde M\_e := M\_{T(e)}$ is its prefix-free trimming from the proof of Thm. 2.20, so that $\widetilde U(i\_e\, \tau) = \widetilde M\_e(\tau)$ for the self-delimiting code $i\_e$ of the index $e$. We write $d\_e := l(i\_e)$ for the length of that code — the **coding constant** of $\widetilde M\_e$; $e \mapsto d\_e$ is computable.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorems</span><span class="math-callout__name">(5.12–5.15 — $s$-$m$-$n$, fixed point, recursion)</span></p>

* **(5.12, $s$-$m$-$n$)** For every partial computable $g(x, y)$ there is a **total** computable $s$ with

  $$M_{s(x)}(y) = g(x, y) \qquad \text{for all } x, y.$$

* **(5.13, fixed point)** For every total computable $f$ there is an index $e$ with

  $$M_e = M_{f(e)}.$$

* **(5.14, Kleene's second recursion theorem)** For every partial computable $g(x, y)$ there is an index $e$ with

  $$M_e(y) = g(e, y) \qquad \text{for all } y.$$

* **(5.15, recursion theorem for prefix-free machines)** Let $h :\subseteq \lbrace 0,1\rbrace^{\ast} \times \mathbb{N} \to \lbrace 0,1\rbrace^{\ast}$ be partial computable such that every section $h(\cdot, e)$ has prefix-free domain. Then there is an index $e$ with

  $$\widetilde M_e = h(\cdot, e).$$

</div>

* **5.12:** hard-code the parameter $x$ into the program; always succeeds, so $s$ is total.
* **5.13:** self-application — let $d$ be total with $M\_{d(e)} = M\_{M\_e(e)}$ ($s$-$m$-$n$), pick $i$ with $M\_i = f \circ d$, then $n := d(i)$ satisfies $M\_n = M\_{M\_i(i)} = M\_{f(n)}$.
* **5.14:** a fixed point of the $s$-$m$-$n$ function $s$ for $g$.
* **5.15:** apply 5.14 to $g(e, y) := h(y, e)$; the resulting $M\_e$ is prefix-free, and trimming leaves prefix-free machines untouched, so $\widetilde M\_e = M\_e$.
* **What it buys.** A program may say "do $g(\text{my own index}, y)$"; in Thm. 6.2 the index is used only to know the **coding constant** $d\_e$. The prefix-free version is needed because $K$ is defined via the trimmed machines; the construction in 6.2 makes every section prefix-free for free ($\operatorname{dom} h(\cdot, c) \subseteq \operatorname{dom} \widetilde U$).

## Random reals: Definition 6.1

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(6.1 — Martin-Löf random real)</span></p>

$\alpha \in [0,1]$ is **Martin-Löf random** if its binary expansion $A$ (the terminating one for dyadic rationals) is a Martin-Löf random sequence (Def. 1.5). Equivalently — Levin–Schnorr, i.e. Thm. 4.1 —

$$\exists c\ \forall n \qquad K(\alpha \upharpoonright n) \ge n - c,$$

where $\alpha \upharpoonright n := A \upharpoonright n$ are the first $n$ bits of the expansion.

</div>

* **The dyadic convention is harmless.** Both expansions of a dyadic rational are computable, hence nonrandom either way.
* **Tests directly on $[0,1]$.** Uniformly c.e. unions of dyadic intervals $U\_i$ with $\lambda(U\_i) \le 2^{-i}$, via $[[\sigma]] \leftrightarrow I\_\sigma = [0.\sigma, 0.\sigma + 2^{-l(\sigma)}]$; the map $A \mapsto 0.A$ is measure-preserving and injective off the dyadics, so a non-dyadic $\alpha$ is caught on $[0,1]$ iff its expansion is caught on Cantor space.
* **Almost every real is random**, and $\alpha \upharpoonright n$ names the level-$n$ dyadic interval of length $2^{-n}$ containing $\alpha$ — a non-dyadic $\alpha$ lies in its *interior*, which is the fact Thm. 6.2 uses.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Computable reals are null-test visible)</span></p>

If $\alpha$ is computable, then $U\_i := [[\alpha \upharpoonright i]]$ — one interval per layer — is a Martin-Löf test: the intervals are uniformly computable and $\lambda(U\_i) = 2^{-i}$. Since $\alpha \in U\_i$ for every $i$, $\alpha$ is not Martin-Löf random.

</div>

* **In the other currency:** $K(\alpha \upharpoonright n) \le K(\mathrm{bin}(n)) + O(1)$ for computable $\alpha$ (K-trivial exercise, Part 5). A computable real can be *pointed at* with $O(\log n)$ bits.
* **What is ruled out, and what is not.** Random reals are irrational and noncomputable, but may well be left-c.e. and compute $\emptyset^{\prime}$ — $\Omega$ is the example. "Random" measures description length of prefixes, not usefulness as an oracle.

## Theorem 6.2: Chaitin's $\Omega$ is random

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(6.2 — Chaitin's $\Omega$ is Martin-Löf random)</span></p>

$\Omega = \sum\_{\sigma \in \operatorname{dom}(\widetilde U)} 2^{-l(\sigma)}$ is Martin-Löf random: there is a constant $d$ with $K(\Omega \upharpoonright n) \ge n - d$ for all $n$.

</div>

Let $w\_s \nearrow \Omega$ be the stage approximation of Def. 5.10.

* **Idea (saboteur with self-knowledge).** For each $c$ build a machine $f^c$ that watches the approximation: whenever some $\widetilde U$-code $\tau$ of the *current* prefix $w\_s \upharpoonright n$ is so short that $l(\tau) + d\_c \le n$, it outputs a **fresh** word, one that $\widetilde U$ has not output by stage $s$. The recursion theorem (5.15) gives an $e$ with $\widetilde M\_e = f^e$: the saboteur reacts exactly to codes that beat **its own** coding constant $d\_e$.
* **Contradiction.** Suppose $K(\Omega \upharpoonright n) \le n - d\_e$ via $\tau$. Since $\Omega$ is non-dyadic the prefix of $w\_s$ stabilizes, so at some stage $s$ the saboteur fires on $\tau$ and outputs a fresh $\sigma$. The code $i\_e\tau$ of $\sigma$ has length $\le n$ and **halts after stage $s$** (freshness), so mass $\ge 2^{-n}$ is still pending — but a stabilized prefix means $\Omega - w\_s < 2^{-n}$. $\square$
* **Why these two ingredients.** *Freshness of the output* is the only handle on the halting time of the code $i\_e\tau$. *Self-knowledge of $d\_e$* is needed so that $l(i\_e\tau) \le n$ — a machine cannot in general know how expensive it is to simulate itself; the fixed point gives one that does. Noncomputability of $\Omega$ (Prop. 5.11, a weaker fact) is used to make the prefix stabilize.
* **Alternative without the recursion theorem (not in the notes).** React to *every* code $\tau$ of its current prefix by issuing the KC request $(l(\tau), \text{fresh } \sigma)$; total weight $\le \Omega \le 1$, so $K(\sigma) \le l(\tau) + d$ by Cor. 2.28.1, and the same pending-mass argument gives $l(\tau) > n - d$. The coding constant then appears only at the end.

## $\Omega$'s two faces side by side

| | information: $\emptyset^{\prime} \equiv\_T \Omega$ (Prop. 5.11) | incompressibility: $K(\Omega \upharpoonright n) \ge n - d$ (Thm. 6.2) |
| :-- | :-- | :-- |
| what $\Omega \upharpoonright n$ does | decides halting of every $\widetilde U$-code of length $\le n$ | cannot be produced by any $\widetilde U$-code of length $\le n - d$ |
| the stage | wait until $w\_s$ agrees with $\Omega$ to $n$ bits | the stage at which $w\_s$ agrees with $\Omega$ to $n$ bits |
| the inequality | pending mass $< 2^{-n}$ ⇒ no pending code of length $\le n$ | a pending code of length $\le n$ ⇒ pending mass $\ge 2^{-n}$ |
| used | positively, to *decide* | negatively, to *refute* a short description |
| extra tool | a machine with $\widetilde M(0^e1)\downarrow \iff \Phi\_e(e)\downarrow$ | the recursion theorem (5.15) |
| moral | $\Omega \upharpoonright n$ contains the halting problem for short codes | and that information cannot be compressed: it *is* $n$ bits |

## Consequences and outlook

* **The natural example exists.** Part 1 asked for a random sequence that is more than "almost every sequence"; the expansion of $\Omega$ is one, defined by a two-line formula from $\widetilde U$.
* **Random left-c.e. reals.** $\Omega$ is left-c.e. (Def. 5.10) and random, so the middle layer of Part 5's hierarchy contains random reals while the bottom layer contains none (Remark). Forward: a left-c.e. real is random iff it is Solovay-complete, i.e. Solovay-equivalent to $\Omega$ (Kučera–Slaman, Thm. 7.6).
* **Randomness spreads upward along speed of approximation.** The Solovay-domination exercise of Part 5 says that $\Omega$'s approximation is the slowest. The next chapter turns this into a reducibility $\le\_S$ (Def. 7.1), shows that randomness transfers upward along it (Prop. 7.3) and puts $\Omega$ on top of all left-c.e. reals (Prop. 7.4). In particular *every* universal machine's $\Omega$ is random, and all of them are Solovay-equivalent.
* **Random ≠ useless.** $\Omega \equiv\_T \emptyset^{\prime}$: a Martin-Löf random real can compute the halting problem. Martin-Löf tests are *effective* null sets, so randomness is always relative to what counts as effective — relative to $\emptyset^{\prime}$, $\Omega$ is computable and hence nonrandom (chapter Relative Randomness and Relative Complexity).

## One-glance summary

| Result | Statement | Tool |
| :-- | :-- | :-- |
| 5.12 | $M\_{s(x)}(y) = g(x,y)$ with $s$ total computable | hard-code $x$ |
| 5.13 | $f$ total computable ⇒ $\exists e: M\_e = M\_{f(e)}$ | $M\_{d(e)} = M\_{M\_e(e)}$, $M\_i = f \circ d$, $e = d(i)$ |
| 5.14 | $g$ partial computable ⇒ $\exists e: M\_e(y) = g(e, y)$ | fixed point of the $s$-$m$-$n$ function |
| 5.15 | sections $h(\cdot, e)$ prefix-free ⇒ $\exists e: \widetilde M\_e = h(\cdot, e)$ | 5.14; trimming (2.20) fixes prefix-free machines |
| convention | dyadic rationals: use the terminating expansion; both expansions are computable | — |
| 6.1 | $\alpha$ ML-random $:\iff$ its expansion is ML-random $\iff \exists c\,\forall n: K(\alpha \upharpoonright n) \ge n - c$ | Def. 1.5, Def. 3.8, Thm. 4.1 |
| tests on $[0,1]$ | uniformly c.e. unions of dyadic intervals with $\lambda(U\_i) \le 2^{-i}$ | $[[\sigma]] \leftrightarrow I\_\sigma$, same measure |
| Remark | computable ⇒ not random | one-interval test $[[\alpha \upharpoonright i]]$; or $K(\alpha \upharpoonright n) \le K(\mathrm{bin}(n)) + O(1)$ |
| 6.2 | $K(\Omega \upharpoonright n) \ge n - d\_e$ for all $n$: $\Omega$ is 1-random, hence ML-random | saboteur $f^e$ that knows $d\_e$ (5.15); fresh output ⇒ pending code of length $\le n$ ⇒ pending mass $\ge 2^{-n}$, against the stabilized prefix |
| alternative | KC requests $(l(\tau), \sigma\_\tau)$ of total weight $\le \Omega \le 1$ | 2.28.1; no recursion theorem |
| outlook | random left-c.e. reals $=$ Solovay-complete reals; all $\Omega$'s are Solovay-equivalent | 7.3, 7.4, 7.6 |

**Punchline.** A real is random when its binary expansion is, and Chaitin's $\Omega$ is the first real for which this can be *proved*. The proof is the gap inequality of Part 5 turned against compression: once $w\_s$ agrees with $\Omega$ to $n$ bits, less than $2^{-n}$ of halting mass is still pending, so no $\widetilde U$-code of length $\le n$ may halt later. A code $\tau$ for $\Omega \upharpoonright n$ with $l(\tau) + d\_e \le n$ would let the saboteur machine $\widetilde M\_e$ — which, by the recursion theorem, knows its own simulation cost $d\_e$ — output a fresh word, and the code $i\_e\tau$ of that word, of length $\le n$, would be exactly such a late halting event. Hence $K(\Omega \upharpoonright n) \ge n - d\_e$: **$\Omega$ is 1-random, hence Martin-Löf random**. The same $n$ bits that decide the halting problem for all short programs are therefore incompressible — $\Omega$ is maximal information in minimal form, and its two faces, Turing-completeness and randomness, are one inequality read in two directions.
