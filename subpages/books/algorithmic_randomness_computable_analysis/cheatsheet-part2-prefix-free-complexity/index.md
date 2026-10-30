---
title: "Cheatsheet — Part 2: Prefix-Free Machines & Prefix-Free Complexity"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - kolmogorov-complexity
  - prefix-free-turing-machine
  - computability
  - kraft-inequality
  - kraft-chaitin
  - cheatsheet
---

# Cheatsheet — Part 2: Prefix-Free Machines & Prefix-Free Complexity

*Exam recap: definitions, formulas and key results only, covering §2.3 up to (but excluding) Information Content Measures. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}#23-prefix-free-turing-machines). Prerequisites in [Part 1]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part1-plain-complexity/' | relative_url }}).*

## Why prefix-free

Plain $C$ has one structural defect: **a plain program is not self-delimiting**, so a decoder reading $\tau\_x\tau\_y$ cannot find the boundary. That forced the length header in Prop. 2.14 and, by Prop. 2.16, the logarithmic overhead is **unavoidable** for $C$.

**Fix:** restrict machines to domains where *no valid program is a proper prefix of another valid program*. Then the first program ends **exactly when the first computation accepts** — the boundary is free.

## Prefix order and comparability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.17 — Prefix and comparability)</span></p>

$$x \sqsubseteq w \;:\Longleftrightarrow\; \exists y:\, w = xy, \qquad x \sqsubset w \;:\Longleftrightarrow\; x \sqsubseteq w \text{ and } x \ne w,$$

$$w \sim v \;:\Longleftrightarrow\; w \sqsubseteq v \;\text{ or }\; v \sqsubseteq w \qquad (\textbf{prefix-comparable}).$$

</div>

* A set $D$ is **prefix-free** iff it is an **antichain** in the full binary tree: no element lies on the path below another.
* **Prop. 2.18:** $\sqsubseteq$ is reflexive and transitive, and $\sim$ satisfies the **cancellation property** $vw \sim v^{\prime}w^{\prime} \Rightarrow v \sim v^{\prime}$ (comparable words are prefixes of one common word, hence so are their prefixes). *This is the workhorse* behind the prefix-freeness of the universal machine.
* ⚠️ The notes also call $\sim$ **transitive**; it is not ($0 \sim \lambda \sim 1$ but $0 \not\sim 1$). Only cancellation is used downstream.
* **Exercise:** products $V\_1 V\_2 \cdots V\_n$ of prefix-free sets are prefix-free — induction, cancellation pushes comparability down to the first blocks.

## Prefix-free machines and $K$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.19 — Prefix-free machines and prefix-free complexity)</span></p>

$M$ is **prefix-free** iff $\mathrm{dom}(M)$ is prefix-free ($x, w \in \mathrm{dom}(M)$ with $x \sqsubset w$ never occurs).

$\widetilde U$ is a **universal prefix-free machine** iff it is prefix-free and every prefix-free $M$ has a code $i\_M$ with $\;\widetilde U(i\_M, w) = M(w)$ for all $w$.

$$K(w) := \min\lbrace l(\sigma) : \widetilde U(\sigma) = w \rbrace.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.20 — Existence of a universal prefix-free machine)</span></p>

A universal prefix-free machine $\widetilde U$ exists.

</div>

* **Idea.** "Is $M\_i$ prefix-free?" is undecidable, so instead **trim every machine**: dovetail $M\_i$ and block any halting input that is comparable with an earlier halting one. The trimmed $M\_{T(i)}$ is always prefix-free and equals $M\_i$ when $M\_i$ already was; $\widetilde U(i, w) := M\_{T(i)}(w)$ is prefix-free by cancellation.
* **Same invariance as before:** $M$ prefix-free $\Rightarrow K(w) \le C\_M(w) + c\_M$. Every upper bound below is "build a prefix-free $M$, count input bits, add $c\_M$".

## The payoff: subadditivity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.21 — Concatenation for prefix-free complexity)</span></p>

$$\boxed{\,K(vw) \le K(v) + K(w) + c\,}$$

</div>

* **Idea.** Feed $\sigma\_v\sigma\_w$ to a machine that tries all splits in parallel. Prefix-freeness of $\widetilde U$ makes the split **unique** (only one prefix of the input is a valid program) and makes the machine itself prefix-free. **No $\log$ term** — contrast Prop. 2.14/2.16.

## $K$ versus $C$ and versus length

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.29 — Chaitin)</span></p>

$$K(\sigma) \le l(\sigma) + K(l(\sigma)) + O(1),$$

where $K(l(\sigma))$ means $K(\mathrm{bin}(l(\sigma)))$. *(Same construction as part 1 of the $C$-vs-$K$ exercise; = Sheet 2, Exercise 4.1.)*

</div>

Two forms:

$$\text{(1)}\; K(w) \le l(w) + K(l(w)) + O(1), \qquad \text{(2)}\; K(w) \le C(w) + K(C(w)) + O(1).$$

* **Idea.** Program $=$ ⟨prefix-free code of a **number**⟩ $\cdot$ ⟨payload⟩: the number is $l(w)$ (resp. $C(w)$), the payload is $w$ verbatim (resp. an optimal plain code run through $U$). The header is found uniquely and tells the machine exactly how many more bits to read.
* **Reading:** $K$ exceeds $C$ or $l$ only by the cost of **announcing where the payload ends**, itself the $K$ of one number. Iterating on the header: $K(w) \le l(w) + \log l(w) + 2\log\log l(w) + O(1)$.

## What carries over from $C$ unchanged

| Result for $K$ | Statement | Note vs the $C$ version |
| :-- | :-- | :-- |
| **2.22** | $K(f(w)) < K(w) + c\_f$ for computable $f$ | same proof, $M := f \circ \widetilde U$; $\mathrm{dom}(M) \subseteq \mathrm{dom}(\widetilde U)$ so $M$ is automatically prefix-free |
| **2.23** | $K$ is upper-semicomputable | same staircase $F(w,t) = \min\lbrace l(\sigma) : \widetilde U(\sigma)[t]\!\downarrow\, = w\rbrace$, via Prop. 2.9 |
| **2.24** | no computable **unbounded** $f$ with $K(w) > f(w)$ on $\mathrm{dom}(f)$ | ⇒ $K$ is **not computable** either |

**Thm. 2.24 — the one real difference.** The diagonal machine's inputs must be prefix-free, so index $n$ is fed in **unary** as $1^n0$ ($n + 1$ bits instead of $\log n$); hence search for $f(\sigma) > 2n$ and compare with $K(M(1^n0)) \le n + 1 + c\_M$.

## Kraft: the measure side

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.25 — Kraft inequality for prefix-free domains)</span></p>

For every prefix-free machine $M$,

$$\sum_{\tau \in \mathrm{dom}(M)} 2^{-l(\tau)} \le 1.$$

</div>

* **Idea:** each $\tau$ claims the cylinder $[[\tau]]$ of measure $2^{-l(\tau)}$; prefix-free $\iff$ the cylinders are **pairwise disjoint**, and they live inside a space of measure $1$.
* **Equality** iff the set is complete; for a halting machine the slack $1 - \sum 2^{-l(\sigma)}$ is the probability that a random input never halts. This is why Chaitin's $\Omega = \sum\_{M(\sigma)\downarrow} 2^{-l(\sigma)}$ is a number in $[0,1]$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.26 — Summability of prefix-free complexity)</span></p>

$$\boxed{\,\sum_{\sigma \in \lbrace 0,1\rbrace^{\ast}} 2^{-K(\sigma)} \le 1\,}$$

</div>

Proof: one optimal code per word; these form a prefix-free subset of $\mathrm{dom}(\widetilde U)$, apply 2.25.

* **2.25 vs 2.26:** 2.25 sums over **inputs**, 2.26 over **outputs**.
* **This is the lower-bound tool for $K$** — it replaces the counting bound (Prop. 2.3). Any claim "$K$ must be large somewhere" comes from making a series diverge.

### The logarithmic overhead is real

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.27 — $K$ exceeds length by a logarithmic overhead)</span></p>

For every constant $c$ there is a word $w$ with

$$K(w) > l(w) + \log l(w) + c.$$

</div>

Proof: if $K(w) \le l(w) + \log l(w) + d$ for **all** $w$, then $\sum\_w 2^{-K(w)} \ge 2^{-d}\sum\_n 2^n \cdot 2^{-n} \cdot \frac1n = \infty$, contradicting 2.26. **The harmonic series is the whole argument** — so 2.29 is essentially tight.

## Kraft–Chaitin: the constructive converse

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.28 — Kraft-Chaitin Theorem)</span></p>

Let $(d\_0,\tau\_0),(d\_1,\tau\_1),\dots$ be a **computably enumerable** sequence with $d\_i \in \mathbb{N}$, $\tau\_i \in \lbrace 0,1\rbrace^{\ast}$ and

$$\sum_{i \in \mathbb{N}} 2^{-d_i} \le 1. \tag{13}$$

Then there is a computable **prefix-free** machine $M$ and codewords $\sigma\_i$ with

$$l(\sigma_i) = d_i, \qquad M(\sigma_i) = \tau_i \qquad \text{for all } i.$$

</div>

Such a sequence is called a **KC set**. Kraft (2.25) says "prefix-free $\Rightarrow$ the measures fit"; Kraft–Chaitin is the converse: **if the measures fit, and the requests are c.e., you can effectively realize them.**

* **Idea (online memory allocator).** Keep the free measure $R\_s = 1 - \sum\_{i<s} 2^{-d\_i}$ as a *terminating* dyadic expansion, with one unused **placeholder** word of length $n$ for every $1$-bit at position $n$. A request of length $d$ takes the placeholder at $d$ if that bit is $1$; otherwise it **splits** the nearest larger placeholder — binary subtraction with a borrow — and the siblings produced by the split become the new placeholders. Terminating expansions guarantee a $1$-bit to borrow from (a non-terminating tail $0.0\cdots0111\ldots$ could *equal* $2^{-d}$ and leave nothing of length $\le d$), and every new word lies below the one placeholder removed, so prefix-freeness is kept. Blocks are never moved or freed, which is why a c.e. request list suffices.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(2.28.1 — KC sets bound $K$)</span></p>

For a KC set $(d\_i, \tau\_i)$: $\qquad K(\tau\_i) \le d\_i + O(1)$ for all $i$.

</div>

**This is the standard way to prove upper bounds on $K$:** don't build a machine — just **enumerate requests** "give $\tau\_i$ a codeword of length $d\_i$" and check the Kraft sum.

* **Exercise (functional upper bound).** If $f$ is computable with $\;\sum\_w 2^{-f(w)} < \infty$, then $K(w) \le f(w) + c$ for all $w$: normalize by a constant $\tilde c$ so the sum is $\le 1$, then $(f(w) + \tilde c, w)$ is a KC set. ⚠️ The notes print the hypothesis as $\sum\_w 2^{-f(w)} \le \infty$, which is vacuous — it must be $< \infty$.

## $C$ versus $K$ side by side

| | plain $C$ | prefix-free $K$ |
| :-- | :-- | :-- |
| Domain of machines | arbitrary | **antichain** (prefix-free) |
| Concatenation | $C(xy) \le C(x) + 2\log C(x) + C(y) + c$; the $\log$ is **unavoidable** (2.16) | $K(vw) \le K(v) + K(w) + c$ — **subadditive** (2.21) |
| Versus length | $C(w) \le l(w) + c$ (2.1) | $K(w) \le l(w) + K(l(w)) + O(1)$ (2.29); exceeds $l + \log l + c$ for some $w$ (2.27) |
| Size/mass bound | $\lvert\lbrace w : C(w) < n\rbrace\rvert < 2^n$ (2.3) | $\sum\_w 2^{-K(w)} \le 1$ (2.26) |
| Computable images | $C(f(w)) < C(w) + c\_f$ (2.2) | $K(f(w)) < K(w) + c\_f$ (2.22) |
| Semicomputability | upper only; not computable (2.10, 2.11) | upper only; not computable (2.23, 2.24) |
| Converse construction | — | **Kraft–Chaitin** (2.28): c.e. requests + Kraft sum $\Rightarrow$ machine |

## One-glance summary

| Result | Statement | Tool |
| :-- | :-- | :-- |
| 2.18 | cancellation: $vw \sim v^{\prime}w^{\prime} \Rightarrow v \sim v^{\prime}$ | common-prefix argument |
| 2.20 | universal prefix-free $\widetilde U$ exists | effectively trim each $M\_i$ to $M\_{T(i)}$ |
| 2.21 | $K(vw) \le K(v) + K(w) + c$ | unique halting prefix = free boundary |
| 2.22 / 2.23 / 2.24 | computable images, upper-semicomputable, no computable lower bound | copy the $C$ proofs; 2.24 needs $\lbrace 1^n0 \rbrace$ |
| 2.25 | $\sum\_{\tau \in \mathrm{dom}(M)} 2^{-l(\tau)} \le 1$ | disjoint cylinders in Cantor space |
| 2.26 | $\sum\_w 2^{-K(w)} \le 1$ | optimal codes $\subseteq \mathrm{dom}(\widetilde U)$, then 2.25 |
| 2.27 | $K(w) > l(w) + \log l(w) + c$ for some $w$ | harmonic series diverges, vs 2.26 |
| 2.28 / 2.28.1 | KC set $\Rightarrow$ prefix-free machine; $K(\tau\_i) \le d\_i + O(1)$ | place dyadic blocks, split placeholders |
| 2.29 | $K(\sigma) \le l(\sigma) + K(l(\sigma)) + O(1)$ | self-delimiting length header |

**Punchline.** Making the domain an antichain buys **self-delimiting programs**, and everything follows: concatenation becomes subadditive with no parsing term (2.21), and the disjoint-cylinder picture turns $K$ into a **measure**, giving $\sum 2^{-K(w)} \le 1$ (2.26) as the replacement for plain complexity's counting bound. That single inequality is now the source of *both* directions — it forces $K$ above $l + \log l$ somewhere (2.27), and, read backwards through **Kraft–Chaitin** (2.28), it lets you bound $K$ from above by merely enumerating requests whose weights sum to $1$, with no machine construction at all.
