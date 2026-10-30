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

**Notation.** $M\_0, M\_1, \ldots$ is the standard enumeration of all Turing machines on words; $\widetilde M\_e := M\_{T(e)}$ is its prefix-free trimming from the proof of Thm. 2.20, so that $\widetilde U(i\_e\, \tau) = \widetilde M\_e(\tau)$ for the self-delimiting code $i\_e$ of the index $e$. We write $d\_e := l(i\_e)$ for the length of that code — the **coding constant** of $\widetilde M\_e$ (for instance $d\_e = e + 1$ if $i\_e = 0^e1$); $e \mapsto d\_e$ is computable.

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

### The four proofs

* **5.12 — hard-code the parameter.** Given $x$, build the machine "on input $y$, write $(x, y)$ and run the machine for $g$". Its index $s(x)$ is computable from $x$, and the construction always succeeds, so $s$ is total even where $g(x, \cdot)$ is nowhere defined.
* **5.13 — self-application plus $s$-$m$-$n$.** Let $d$ be total computable with

  $$M_{d(e)} = M_{M_e(e)} \ \text{ if } M_e(e)\downarrow, \qquad M_{d(e)} = \text{the nowhere-defined machine otherwise}$$

  ($s$-$m$-$n$ applied to the partial computable $(e, k) \mapsto M\_{M\_e(e)}(k)$). Pick $i$ with $M\_i = f \circ d$ — total, so $M\_i(i)\downarrow = f(d(i))$ — and put $n := d(i)$. Then

  $$M_n = M_{d(i)} = M_{M_i(i)} = M_{f(d(i))} = M_{f(n)}.$$

* **5.14 — a fixed point of the $s$-$m$-$n$ function.** Take total $s$ with $M\_{s(x)}(y) = g(x, y)$ and a fixed point $e$ of $s$ (5.13): $M\_e(y) = M\_{s(e)}(y) = g(e, y)$.
* **5.15 — trimming is invisible on prefix-free machines.** Apply 5.14 to $g(e, y) := h(y, e)$: some $M\_e = h(\cdot, e)$. This machine is prefix-free, and the construction of Thm. 2.20 only removes inputs that conflict with an earlier comparable one, so it leaves $M\_e$ unchanged: $\widetilde M\_e = M\_{T(e)} = M\_e = h(\cdot, e)$.

### Reading the toolkit

* **What the recursion theorem buys.** Any program may be written as "on input $y$, do $g(\text{my own index}, y)$": the index is unknown while $g$ is being written, yet some $e$ makes the self-reference exact. In Thm. 6.2 the index is used for one thing only — to know the **coding constant** $d\_e$ at which $\widetilde U$ simulates the machine.
* **Why a prefix-free version.** $K$ is defined through $\widetilde U$, which simulates the *trimmed* machines $\widetilde M\_e$. A fixed point for the plain enumeration is useless unless it is also one for the tilde enumeration; 5.15 says it is, provided every section of $h$ is prefix-free. The construction in 6.2 guarantees this for free: $\operatorname{dom} h(\cdot, c) \subseteq \operatorname{dom} \widetilde U$.
* **Effective, but opaque.** The fixed point $n = d(i)$ is *computed* from an index of $f \circ d$. What is not available is any information about what $M\_n$ does beyond $M\_n = M\_{f(n)}$ — in particular, whether it halts anywhere.

## Random reals: Definition 6.1

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(6.1 — Martin-Löf random real)</span></p>

$\alpha \in [0,1]$ is **Martin-Löf random** if its binary expansion $A$ (the terminating one for dyadic rationals) is a Martin-Löf random sequence (Def. 1.5). Equivalently — Levin–Schnorr, i.e. Thm. 4.1 —

$$\exists c\ \forall n \qquad K(\alpha \upharpoonright n) \ge n - c,$$

where $\alpha \upharpoonright n := A \upharpoonright n$ are the first $n$ bits of the expansion.

</div>

* **The dyadic convention is harmless.** Only dyadic rationals have two expansions, $0.w1000\ldots = 0.w0111\ldots$; both are computable, hence (Remark below) nonrandom either way. Every other real has exactly one expansion, so Def. 6.1 is well posed.
* **Tests directly on $[0,1]$.** A **Martin-Löf test for reals** is a uniformly c.e. sequence of open sets $U\_i \subseteq [0,1]$, each a c.e. union of dyadic intervals (closed or open — the endpoints are dyadic, hence computable and irrelevant), with $\lambda(U\_i) \le 2^{-i}$; $\alpha$ is random iff $\alpha \notin \bigcap\_i U\_i$ for every such test. This is Def. 1.5 transported along

  $$[\![\sigma]\!] \;\longleftrightarrow\; I_\sigma := \bigl[\,0.\sigma,\ 0.\sigma + 2^{-l(\sigma)}\,\bigr], \qquad \lambda(I_\sigma) = 2^{-l(\sigma)} = \lambda([\![\sigma]\!]),$$

  the map $A \mapsto 0.A$ being measure-preserving and injective off the dyadic rationals. Pushing a layer forward does not change its measure; pulling one back adds at most the second names of dyadic endpoints — computable sequences. So a non-dyadic $\alpha$ is caught by a test on $[0,1]$ iff its expansion is caught by one on Cantor space.
* **Almost every real is random.** The 1-nonrandom sequences form a null set (Part 4) and $A \mapsto 0.A$ preserves measure, so $\lambda(\lbrace \alpha \in [0,1] : \alpha \text{ is ML-random}\rbrace) = 1$.
* **Geometry.** $\alpha \upharpoonright n$ names the level-$n$ dyadic interval $I\_{\alpha \upharpoonright n} \ni \alpha$ of length $2^{-n}$ — the real-line face of the dyadic bracket (27) of Part 5. A non-dyadic $\alpha$ lies in its *interior*; this is the fact used at step $(\ast)$ of Thm. 6.2.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Computable reals are null-test visible)</span></p>

If $\alpha$ is computable, then $U\_i := [[\alpha \upharpoonright i]]$ — one interval per layer — is a Martin-Löf test: the intervals are uniformly computable and $\lambda(U\_i) = 2^{-i}$. Since $\alpha \in U\_i$ for every $i$, $\alpha$ is not Martin-Löf random.

</div>

* **In the other currency.** For computable $\alpha$, $K(\alpha \upharpoonright n) \le K(\mathrm{bin}(n)) + O(1) \le 2\log n + O(1)$ (step 1 of the K-trivial exercise, Part 5) — nowhere near $n - c$. Both proofs say the same thing: a computable real can be *pointed at* with $O(\log n)$ bits, or trapped in intervals of length $2^{-n}$ by a procedure that costs nothing.
* **What is ruled out, and what is not.** Random reals are irrational, non-dyadic and noncomputable. They need *not* fail to be left-c.e. or $\Delta^0\_2$: Thm. 6.2 exhibits a random real that is left-c.e. and computes $\emptyset^{\prime}$. "Random" measures the description length of prefixes, not the usefulness of the real as an oracle.

## Theorem 6.2: Chaitin's $\Omega$ is random

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(6.2 — Chaitin's $\Omega$ is Martin-Löf random)</span></p>

$\Omega = \sum\_{\sigma \in \operatorname{dom}(\widetilde U)} 2^{-l(\sigma)}$ is Martin-Löf random: there is a constant $d$ with $K(\Omega \upharpoonright n) \ge n - d$ for all $n$.

</div>

Fix the approximation $w\_s = \sum\_{\sigma \in \operatorname{dom}(\widetilde U[s]),\ l(\sigma) \le s} 2^{-l(\sigma)} \nearrow \Omega$ of Def. 5.10, where $\widetilde U[s]$ is $\widetilde U$ run for $s$ steps on the inputs of length $\le s$ (the length bound of the sum); so $\operatorname{dom}(\widetilde U[s])$ and $\operatorname{range}(\widetilde U[s])$ are finite and computable from $s$.

### Step 1 — a uniform family of saboteur machines

For every $c \in \mathbb{N}$ define a partial computable $f^c$ by enumerating its graph. **Stage $s$:** for every $n \le s$ and every $\tau \in \operatorname{dom}(\widetilde U[s])$ not yet in $\operatorname{dom}(f^c)$ with

$$\widetilde U(\tau)[s]\downarrow = w_s \upharpoonright n \qquad \text{and} \qquad l(\tau) + d_c \le n, \tag{S}$$

choose a word $\sigma \notin \operatorname{range}(\widetilde U[s])$ — say the length-lexicographically least — and set $f^c(\tau) := \sigma$.

* **Reading (S).** $\tau$ is a $\widetilde U$-code of the *current* $n$-bit prefix of the approximation, and it is so short that even after the simulation overhead $d\_c$ it still fits into $n$ bits.
* **Prefix-free.** $\operatorname{dom}(f^c) \subseteq \operatorname{dom}(\widetilde U)$, which is prefix-free.
* **Uniform.** $h(\tau, c) := f^c(\tau)$ is partial computable: dovetail over $c$ and $s$; each stage checks finitely many pairs $(n, \tau)$.
* **Fresh output is the whole point.** Every $\widetilde U$-code of $\sigma$ halts *after* stage $s$, because $\sigma$ is not among the outputs seen by stage $s$.

### Step 2 — the fixed point

By 5.15 there is an index $e$ with $\widetilde M\_e = f^e$: a machine that acts only on codes beating **its own** coding constant $d\_e$. For every $\tau \in \operatorname{dom}(f^e)$ put $\nu := i\_e\, \tau$; then

$$\widetilde U(\nu) = \widetilde M_e(\tau) = f^e(\tau), \qquad l(\nu) = l(\tau) + d_e .$$

### Step 3 — the contradiction

**Claim:** $K(\Omega \upharpoonright n) > n - d\_e$ for every $n$. Suppose not: some $\tau$ has $\widetilde U(\tau) = \Omega \upharpoonright n$ and $l(\tau) + d\_e \le n$.

1. **The prefix stabilizes.** $\Omega$ is noncomputable (Prop. 5.11), in particular not dyadic, so $0.(\Omega \upharpoonright n) < \Omega < 0.(\Omega \upharpoonright n) + 2^{-n}$ strictly. As $w\_s \nearrow \Omega$, from some stage on $w\_s$ lies in this open interval, i.e. $w\_s \upharpoonright n = \Omega \upharpoonright n$; once $\widetilde U(\tau)$ has halted as well, (S) holds for $(n, \tau)$ at all large stages $s$. So $f^e(\tau) =: \sigma$ is defined, at some stage $s$ with

   $$\widetilde U(\tau)[s]\downarrow = w_s \upharpoonright n = \Omega \upharpoonright n. \tag{$\ast$}$$

2. **A late halting event.** $\sigma \notin \operatorname{range}(\widetilde U[s])$, so $\nu = i\_e\tau$ satisfies $\widetilde U(\nu) = \sigma$ but $\nu \notin \operatorname{dom}(\widetilde U[s])$: its mass is not yet in $w\_s$. Hence

   $$\Omega - w_s \;\ge \sum_{\rho \,\in\, \operatorname{dom}(\widetilde U) \setminus \operatorname{dom}(\widetilde U[s])} 2^{-l(\rho)} \;\ge\; 2^{-l(\nu)} \;\ge\; 2^{-n}.$$

3. **But the prefix had stabilized.** By $(\ast)$, $0.(\Omega \upharpoonright n) \le w\_s \le \Omega < 0.(\Omega \upharpoonright n) + 2^{-n}$, so $\Omega - w\_s < 2^{-n}$. Contradiction. $\square$

Hence $K(\Omega \upharpoonright n) \ge n - d\_e + 1$ for all $n$: $\Omega$ is 1-random (Def. 3.8) and therefore Martin-Löf random (Thm. 4.1).

### Why the details are the way they are

* **Why a fresh $\sigma$.** The contradiction needs a code of length $\le n$ that halts *after* stage $s$. Outputting an old word would not guarantee this: the code $\nu$ might already lie in $\operatorname{dom}(\widetilde U[s])$ with its mass already counted in $w\_s$. Freshness of the *output* is the only handle on the halting time of the *code*.
* **Why the recursion theorem.** The machine must ignore codes with $l(\tau) + d > n$: otherwise $\nu$ could be longer than $n$ and the mass $2^{-l(\nu)}$ too small to contradict anything. So the threshold in (S) must be the machine's own coding constant, which only 5.15 makes available. A machine cannot in general know how expensive it is to simulate itself; the fixed point gives one that does.
* **Why noncomputability of $\Omega$ is an input.** It guarantees step 1. The prefixes of an approximation from below need never agree with those of a *dyadic* limit ($0.0111\ldots1 \to 0.1$ — the dyadic-boundary issue of Part 5). A weaker, earlier fact (Prop. 5.11) is thus used to prove a stronger one.
* **Where prefix-freeness and Kraft enter.** $\Omega \le 1$ and $\Omega = \lambda(\operatorname{dom} \widetilde U)$ (Def. 5.8) are Kraft; $\operatorname{dom}(f^c) \subseteq \operatorname{dom}(\widetilde U)$ makes 5.15 applicable without any trimming. The inequality in step 2 itself only uses that masses are nonnegative.
* **What the construction never needs to know.** Whether $w\_s \upharpoonright n$ really is $\Omega \upharpoonright n$. $f^e$ reacts to every short code of every *current* prefix; most reactions are harmless (they define $f^e$ on more of $\operatorname{dom} \widetilde U$). Only the reaction to a true prefix is contradictory — which is exactly why no true prefix has a short code.
* ⚠️ **Source slips.** The notes state the stage condition as "$K(\tau) < l(\tau) - n$", with $K(\tau)$ where a length comparison belongs and with no reference to $c$ at all; the intended condition is (S). They also write $l(\nu) < l(\tau) + e < n$, conflating the fixed-point index $e$ with the length of its code; with $l(\nu) = l(\tau) + d\_e \le n$ the argument goes through unchanged, since $w\_s \upharpoonright n = \Omega \upharpoonright n$ already forces $\Omega - w\_s < 2^{-n}$.

### An alternative without the recursion theorem (not in the notes)

Act on *every* code instead of only on the short ones, and pay with Kraft mass. At stage $s$, for each unhandled $\tau \in \operatorname{dom}(\widetilde U[s])$ whose output is the current prefix of its own length, $\widetilde U(\tau)[s] = w\_s \upharpoonright l(\widetilde U(\tau))$, issue the request $(l(\tau), \sigma\_\tau)$ with $\sigma\_\tau \notin \operatorname{range}(\widetilde U[s])$ fresh. The request set is c.e. of total weight $\le \sum\_{\tau \in \operatorname{dom} \widetilde U} 2^{-l(\tau)} = \Omega \le 1$, so Kraft–Chaitin (Cor. 2.28.1) gives a constant $d$ with $K(\sigma\_\tau) \le l(\tau) + d$. If $\tau$ is a code of the true prefix $\Omega \upharpoonright n$, then at its acting stage $s$ a code of $\sigma\_\tau$ of length $\le l(\tau) + d$ is still pending, so $2^{-l(\tau) - d} \le \Omega - w\_s < 2^{-n}$, i.e. $l(\tau) > n - d$. Hence $K(\Omega \upharpoonright n) > n - d$ for all $n$. The coding constant now appears only at the *end*, so no self-reference is needed; the price is the Kraft bookkeeping, which the notes' version gets for free from $\operatorname{dom}(f^c) \subseteq \operatorname{dom}(\widetilde U)$.

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
| Remark | computable ⇒ not random | one-interval test $[[\alpha \upharpoonright i]]$; or $K(\alpha \upharpoonright n) \le 2\log n + O(1)$ |
| 6.2, step 1 | $f^c(\tau) :=$ fresh $\sigma$ whenever $\widetilde U(\tau)[s] = w\_s \upharpoonright n$ and $l(\tau) + d\_c \le n$ | $\operatorname{dom} f^c \subseteq \operatorname{dom} \widetilde U$ ⇒ prefix-free; uniform in $c$ |
| 6.2, step 2 | $\widetilde M\_e = f^e$, so $\widetilde U(i\_e\tau) = f^e(\tau)$ with $l(i\_e\tau) = l(\tau) + d\_e$ | 5.15 |
| 6.2, step 3 | short code of $\Omega \upharpoonright n$ ⇒ pending code of length $\le n$ ⇒ $\Omega - w\_s \ge 2^{-n}$; but $w\_s \upharpoonright n = \Omega \upharpoonright n$ ⇒ $\Omega - w\_s < 2^{-n}$ | $\Omega$ non-dyadic (5.11) for the stage; freshness for the pending code |
| 6.2 | $K(\Omega \upharpoonright n) \ge n - d\_e$ for all $n$: $\Omega$ is 1-random, hence ML-random | 4.1 |
| alternative | KC requests $(l(\tau), \sigma\_\tau)$ of total weight $\le \Omega \le 1$ | 2.28.1; no recursion theorem |
| outlook | random left-c.e. reals $=$ Solovay-complete reals; all $\Omega$'s are Solovay-equivalent | 7.3, 7.4, 7.6 |

**Punchline.** A real is random when its binary expansion is, and Chaitin's $\Omega$ is the first real for which this can be *proved*. The proof is the gap inequality of Part 5 turned against compression: once $w\_s$ agrees with $\Omega$ to $n$ bits, less than $2^{-n}$ of halting mass is still pending, so no $\widetilde U$-code of length $\le n$ may halt later. A code $\tau$ for $\Omega \upharpoonright n$ with $l(\tau) + d\_e \le n$ would let the saboteur machine $\widetilde M\_e$ — which, by the recursion theorem, knows its own simulation cost $d\_e$ — output a fresh word, and the code $i\_e\tau$ of that word, of length $\le n$, would be exactly such a late halting event. Hence $K(\Omega \upharpoonright n) \ge n - d\_e$: **$\Omega$ is 1-random, hence Martin-Löf random**. The same $n$ bits that decide the halting problem for all short programs are therefore incompressible — $\Omega$ is maximal information in minimal form, and its two faces, Turing-completeness and randomness, are one inequality read in two directions.
