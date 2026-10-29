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

* **(20) = upper-semicomputability** (Def. 2.7, with $\le k$ instead of $< y$ — equivalent). Equivalently (Prop. 2.9, with the staircase allowed the value $\infty$ off the domain): $F(\sigma) = \lim\_t F(\sigma, t)$ for a computable nonincreasing staircase. You can *certify* $F(\sigma) \le k$, never $F(\sigma) > k$.
* **(19) = Kraft** (Prop. 2.25 / Thm. 2.26) with $F$ in place of $l$ or $K$: each domain word spends mass $2^{-F(\sigma)}$ out of a unit budget.
* **No machines anywhere.** $\mathrm{dom}(F)$ need *not* be prefix-free: $\mathrm{dom}(F) = \lbrace 0, 10, 001\rbrace$ with values $2, 4, 5$ is an i.c.m. (mass $11/32$), although $0 \sqsubset 001$. Words outside the domain cost nothing.

| i.c.m. | domain | (20) holds because | (19) holds because |
| :-- | :-- | :-- | :-- |
| $f(0^n1) = n + 1$ | $\lbrace 0^n 1 : n \ge 0\rbrace$ | partially computable | $\sum\_n 2^{-(n+1)} = 1$ — budget spent **exactly**, the domain is a complete prefix-free set |
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

* **Obstruction** (same as in Thm. 2.20): "does this upper-semicomputable $F$ satisfy (19)?" is undecidable, so you cannot filter the list of epigraphs.
* **Fix — a Kraft filter.** Start from a computable enumeration $E\_0, E\_1, \dots$ of *all* c.e. epigraphs, i.e. of all upper-semicomputable partial $\widetilde F\_i$ (these satisfy (20) for free). Machine $M(i)$ copies the pairs $(\sigma^i\_n, k^i\_n)$ of $E\_i$ in order. After $n$ pairs it keeps, for each **distinct** word seen so far, its **minimal** weight $k$, and copies the $n$-th pair iff all previous pairs were copied **and**

$$\sum_{\text{distinct } \sigma \text{ seen so far}} 2^{-\min k} \;\le\; 1. \tag{21}$$

  Once a pair is blocked the stream stops for good.
* **Why it is exact.** The partial sums in (21) increase to the Kraft sum of $\widetilde F\_i$. If $\widetilde F\_i$ is an i.c.m. they never exceed $1$, nothing is blocked, and $F\_{M(i)} = \widetilde F\_i$. If it is not, some *finite* stage exceeds $1$, the copy stops just before, and $F\_{M(i)}$ satisfies (19) anyway. So $(F\_{M(i)})\_i$ lists exactly the i.c.m.s.
* **Pattern to remember** (2.20 and 3.2 share it): *can't decide the property → effectively enforce it, in a way that leaves the objects which already have it untouched.*

## The minimal information content measure

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.3 — Minimal information content measure)</span></p>

Fix the enumeration of Thm. 3.2 and set

$$\widetilde F(w) := \min_{i\,:\,w \in \mathrm{dom}(F_i)} \bigl(F_i(w) + i + 1\bigr).$$

</div>

* **Why the penalty $i + 1$.** Without it, $\min\_i F\_i(w) = 0$ for *every* $w$ (the atomic measures are in the list), so the Kraft sum would be $\infty$. With it, the $i$-th measure receives only the share $2^{-(i+1)}$ of the budget, and $\sum\_i 2^{-(i+1)} = 1$. Note $i + 1 = l(1^i 0)$: the penalty is the cost of **naming the $i$-th measure with a self-delimiting unary index** — the i.c.m. counterpart of the machine code $\rho\_M$ in the invariance inequality.
* **Lower envelope.** $\widetilde F$ is the pointwise minimum of the shifted graphs $F\_i + i + 1$: for each word, the cheapest measure *after paying for its name*.
* ⚠️ **$\max$ instead of $\min$ is not an i.c.m.:** infinitely many $F\_i$ are defined at any $w$ (for a start, $F(w) = j$ with domain $\lbrace w\rbrace$ is an i.c.m. for every $j \in \mathbb{N}$), so $F\_i(w) + i + 1 \to \infty$ and the maximum is not in $\mathbb{N}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.4</span><span class="math-callout__name">($\widetilde F$ is an information content measure)</span></p>

$\widetilde F$ satisfies (19) and (20).

</div>

* **(19) — $\max \le \sum$, then swap.** A minimum of exponents is a maximum of masses, and a maximum is at most the sum:

$$2^{-\widetilde F(w)} = \max_i 2^{-(F_i(w) + i + 1)} \le \sum_i 2^{-(F_i(w) + i + 1)},$$

$$\sum_{w} 2^{-\widetilde F(w)} \;\le\; \sum_{i} 2^{-(i+1)} \sum_{w \in \mathrm{dom}(F_i)} 2^{-F_i(w)} \;\le\; \sum_i 2^{-(i+1)} \cdot 1 \;=\; 1.$$

  (Undefined $F\_i(w)$ contribute $0$; the inner sum is (19) for $F\_i$.)
* **(20) — one witness suffices.** Dovetail all $F\_i(w)$ and enumerate $(w, k)$ as soon as *some* $i$ certifies $F\_i(w) + i + 1 \le k$. A minimum needs only one witness, so a true inequality is found in finite time.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.5 — $\widetilde F$ is total)</span></p>

$\widetilde F(w)$ is defined for every $w$.

</div>

Proof: the **atomic** i.c.m. $F\_{i\_w}$ ($F(w) = 0$, undefined elsewhere) is in the list, so the minimum runs over a non-empty set. *Totality comes from the list containing a measure for every single word, not from any computability.*

## $K$ is the smallest information content measure

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.6 — $\widetilde F$ coincides with $K$)</span></p>

$$K(w) = \widetilde F(w) \pm O(1) \qquad \text{for all } w.$$

</div>

* **$\widetilde F \le K + O(1)$ — $K$ is in the list.** $K$ is an i.c.m., so $K = F\_k$ for some $k$, and by definition $\widetilde F(w) \le F\_k(w) + k + 1 = K(w) + k + 1$.
* **$K \le \widetilde F + O(1)$ — the epigraph is a KC set.** The set $\lbrace (k + 1, w) : \widetilde F(w) \le k\rbrace$ of length-requests is c.e. by (20), and its Kraft sum is

$$\sum_{w} \sum_{k \ge \widetilde F(w)} 2^{-(k+1)} = \sum_{w} 2^{-\widetilde F(w)} \le 1$$

  by (19). Cor. 2.28.1 gives $K(w) \le \widetilde F(w) + 1 + O(1)$ (totality of $\widetilde F$ makes sure every $w$ is requested).
* **Why the $+1$ is free.** The epigraph requests *every* $k \ge \widetilde F(w)$, not just $\widetilde F(w)$ itself; the geometric tail $\sum\_{k \ge m} 2^{-(k+1)} = 2^{-m}$ absorbs all of them at the price of exactly one extra bit.

The form you actually use:

$$\boxed{\,\text{for every i.c.m. } F \text{ there is a constant } c_F \text{ with } \quad K(w) \le F(w) + c_F \quad \text{for all } w \in \mathrm{dom}(F).\,} \tag{M}$$

(Either $K \le \widetilde F + O(1) \le F + i\_F + 1 + O(1)$, or directly: the epigraph of $F$ is itself a KC set.)

* **What "smallest" means** (the question left open in the notes): $K$ lies below *every* member of the class up to an additive constant — the least element of the class modulo $+O(1)$. Exactly as $C$ was the least function whose sublevel sets are uniformly c.e. and of size $< 2^n$ (Part 1), with the counting condition traded for the Kraft condition.
* **Three ways to bound $K$ from above**, each with less bookkeeping than the one before:

| Level | What you exhibit | Conclusion | Where |
| :-- | :-- | :-- | :-- |
| machine | a prefix-free $M$ with $M(\tau) = w$ | $K(w) \le l(\tau) + c\_M$ | invariance, Def. 2.19 / Thm. 2.20 |
| KC set | c.e. requests $(d\_i, \tau\_i)$ with $\sum\_i 2^{-d\_i} \le 1$ | $K(\tau\_i) \le d\_i + O(1)$ | Cor. 2.28.1 |
| i.c.m. | upper-semicomputable $F$ with $\sum 2^{-F} \le 1$ | $K(w) \le F(w) + c\_F$ | (M), Thm. 3.6 |

  An i.c.m. is a KC set up to one extra bit: its *graph* is not c.e. (only the epigraph is), so you request the whole column $(k + 1, w)$, $k \ge F(w)$, exactly as in the proof of 3.6. From here on, upper bounds on $K$ are proved by **checking two conditions on a function**, no construction at all.

## Chaitin's counting theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.7 — Chaitin's counting theorem)</span></p>

There is a constant $c$ such that for all $n, r \in \mathbb{N}$

$$\#\lbrace \sigma \in \lbrace 0,1\rbrace^n : K(\sigma) \le n + K(\mathrm{bin}(n)) - r\rbrace \;\le\; 2^{\,n - r + c}. \tag{22}$$

In particular

$$\max\lbrace K(\sigma) : l(\sigma) = n\rbrace = n + K(\mathrm{bin}(n)) \pm O(1). \tag{23}$$

</div>

**Notation.** $W\_n := \sum\_{l(\sigma) = n} 2^{-K(\sigma)}$ is the **$K$-mass of length $n$**; by Thm. 2.26, $\sum\_n W\_n \le 1$.

**Proof in four steps.**

1. **$F(\mathrm{bin}(n)) := \lceil -\log W\_n\rceil$ is an i.c.m.** (19): $2^{-F(\mathrm{bin}(n))} \le W\_n$, so $\sum\_n 2^{-F(\mathrm{bin}(n))} \le \sum\_n W\_n \le 1$. (20): $F(\mathrm{bin}(n)) \le k \iff W\_n \ge 2^{-k}$. Approximate $W\_n$ from below by $W\_n^t := \sum\_{l(\sigma) = n} 2^{-K\_t(\sigma)}$ with the staircase $K\_t$ of Prop. 2.23; each $K\_t(\sigma)$ is a nonincreasing *integer* sequence, hence eventually constant, and there are only $2^n$ words — so $W\_n^t = W\_n$ from some finite $t$ on. Enumerate $(\mathrm{bin}(n), k)$ once $W\_n^t \ge 2^{-k}$.
   ⚠️ The notes write $F = -\log W\_n$ without the ceiling, which is not $\mathbb{N}$-valued; rounding up lowers the masses, so (19) survives, and it costs at most one bit in step 2, absorbed into the constant. The notes' c.e. argument also tests $F > k$ where the epigraph needs $F \le k$.
2. **Minimality bounds the mass of a length by the mass of its length.** (M) gives $K(\mathrm{bin}(n)) \le F(\mathrm{bin}(n)) + c < -\log W\_n + 1 + c$, hence, after renaming the constant,

$$W_n \;\le\; 2^{\,c}\, 2^{-K(\mathrm{bin}(n))} \qquad \text{for all } n:$$

   all $2^n$ words of length $n$ together weigh at most a constant times the single word $\mathrm{bin}(n)$. ⚠️ The notes print this step as $2^{K(\mathrm{bin}(n))} \ge 2^{-F(\mathrm{bin}(n)) - c}$ — sign typo; the intended inequality is $2^{-K(\mathrm{bin}(n))} \ge 2^{-F(\mathrm{bin}(n)) - c}$ (and they announce a constant $\varepsilon > 0$ that is never used).
3. **Mass bound → count bound (Markov).** Put $S\_{n,r} := \lbrace \sigma \in \lbrace 0,1\rbrace^n : K(\sigma) \le n + K(\mathrm{bin}(n)) - r\rbrace$. Each member weighs at least $2^{-n - K(\mathrm{bin}(n)) + r}$, so

$$\#S_{n,r} \cdot 2^{-n - K(\mathrm{bin}(n)) + r} \;\le\; \sum_{\sigma \in S_{n,r}} 2^{-K(\sigma)} \;\le\; W_n \;\le\; 2^{c}\, 2^{-K(\mathrm{bin}(n))} \quad\Longrightarrow\quad \#S_{n,r} \le 2^{\,n - r + c}.$$

4. **(23).** Upper bound: Chaitin's Thm. 2.29, $K(\sigma) \le n + K(\mathrm{bin}(n)) + c\_2$. Lower bound: take $r := c + 1$ in (22), so $\sharp S\_{n,c+1} \le 2^{n-1} < 2^n$ and some word of length $n$ lies outside, i.e. has $K(\sigma) > n + K(\mathrm{bin}(n)) - c - 1$.
   ⚠️ The sheet prints the lower bound as $n - K(\mathrm{bin}(n)) - c\_1$; it must read $n + K(\mathrm{bin}(n)) - c\_1$, matching (23).

* **The reference level shifts.** For $C$ the ceiling is $l(w)$: at most $2^{n-k+1} - 1$ words of length $n$ have $C \le n - k$ (Cor. 2.3.1). For $K$ the ceiling is $l(w) + K(l(w))$: at most $2^{n - r + c}$ have $K \le n + K(\mathrm{bin}(n)) - r$. Same shape — **each bit below the ceiling halves the count** — but the ceiling moved up by $K(\mathrm{bin}(n))$.
* Prop. 2.27 only said *some* word has $K > l + \log l + c$; (23) says the overhead $K(\mathrm{bin}(n))$ is attained at *every* length, up to a constant.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(3.7.1 — Almost every word is incompressible)</span></p>

$$\frac{\#\lbrace w \in \lbrace 0,1\rbrace^n : K(w) > n\rbrace}{2^n} \xrightarrow[\;n \to \infty\;]{} 1.$$

</div>

Proof: $r := K(\mathrm{bin}(n))$ in (22) makes the threshold exactly $n$:

$$\frac{\#\lbrace w : K(w) \le n\rbrace}{2^n} \le \frac{2^{\,n - K(\mathrm{bin}(n)) + c}}{2^n} = 2^{\,c - K(\mathrm{bin}(n))} \longrightarrow 0,$$

because $K(\mathrm{bin}(n)) \to \infty$: only finitely many words lie below any fixed complexity (at most $2^{m+1} - 1$ programs of length $\le m$), and the $\mathrm{bin}(n)$ are distinct.

* **Contrast with $C$:** Cor. 2.3.1 guarantees only *one* word per length with $C \ge n$, and the average is $\approx n$ (Prop. 2.4). For $K$ the room above the length — $K(\mathrm{bin}(n))$ bits of it — is where most words live.

## Two exercises on the Kraft side

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Characterization of computable upper bounds for $K$)</span></p>

For a total computable $f : \mathbb{N} \to \mathbb{N}$:

$$\sum_{n} 2^{-f(n)} < \infty \quad\Longleftrightarrow\quad \exists c\ \forall n: \ K(\mathrm{bin}(n)) \le f(n) + c.$$

</div>

* **($\Rightarrow$) via (M).** Pick $d$ with $2^{-d}\sum\_n 2^{-f(n)} \le 1$ and set $F(\mathrm{bin}(n)) := f(n) + d$: computable, Kraft sum $\le 1$, so an i.c.m.; (M) gives $K(\mathrm{bin}(n)) \le f(n) + d + c\_F$.
* **($\Leftarrow$) via 2.26.** $2^{-f(n)} \le 2^{c}\, 2^{-K(\mathrm{bin}(n))}$, so $\sum\_n 2^{-f(n)} \le 2^c \sum\_\sigma 2^{-K(\sigma)} \le 2^c$.
* **Dictionary.** Computable upper bounds for $K$ on $\mathbb{N}$ (up to $+c$) *are* the computable Kraft-summable functions — summability is the whole obstruction. Since finitely many violations can be absorbed into $c$, divergence even gives: for every $c$, $K(\mathrm{bin}(n)) > f(n) + c$ for **infinitely many** $n$.
* **Examples** (round $f$ up to integers; small $n$ do not matter):

| $f(n)$ | $\sum\_n 2^{-f(n)}$ | verdict |
| :-- | :-- | :-- |
| $\log n$ | $\sum 1/n = \infty$ | ✗ — cf. Prop. 2.27 |
| $\log n + \log\log n$ | $\sum 1/(n\log n) = \infty$ | ✗ — $K(\mathrm{bin}(n))$ exceeds it by any constant, infinitely often |
| $\log n + 2\log\log n$ | $\sum 1/(n\log^2 n) < \infty$ | ✓ — $K(\mathrm{bin}(n)) \le \log n + 2\log\log n + O(1)$, cf. the iterated form of 2.29 |

* Compare the Part-2 exercise "functional upper bound": that was ($\Rightarrow$) for $f$ on words, proved with a KC set. Here the same direction takes one line via (M), and 2.26 supplies the converse.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Test-to-i.c.m. translation is not straightforward)</span></p>

A **Schnorr test** is a Martin-Löf test with $\mu(S\_i) = 2^{-i}$ **exactly**. Let $(S\_n)$ be uniformly c.e. prefix-free sets of words whose cylinders form a Schnorr test. Then

$$F(\sigma) := l(\sigma) - n \qquad (\sigma \in S_n)$$

is **not** an information content measure.

</div>

* **Each layer spends the whole budget.** Prefix-freeness turns the measure into a Kraft sum, $\sum\_{\sigma \in S\_n} 2^{-l(\sigma)} = \mu(S\_n) = 2^{-n}$, hence

$$\sum_{\sigma \in S_n} 2^{-(l(\sigma) - n)} = 2^{n} \sum_{\sigma \in S_n} 2^{-l(\sigma)} = 2^n \cdot 2^{-n} = 1, \qquad \sum_n 1 = \infty.$$

* ⚠️ The notes print the exercise with the definition of $F$ missing (only the expression $\sum\_{\sigma \in S\_n} 2^{-l(\sigma) + n}$ is shown); $F(\sigma) = l(\sigma) - n$ is the intended reading (a word lying in several layers gets its smallest value; the sum still diverges).
* **The fix — thin the layers.** Use only the layers $n = i^2$ and claim only $i$ bits: $F(\sigma) := l(\sigma) - i$ for $\sigma \in S\_{i^2}$ (a word in several layers takes the smallest value; $\max \le \sum$ as in 3.4). Layer $i$ now costs $2^{i} \cdot 2^{-i^2}$, and

$$\sum_i 2^{\,i - i^2} < \infty,$$

  so a constant shift makes it an i.c.m. **Moral:** a layer of measure $2^{-m}$ can buy up to $m$ bits of compression, but buying all of them costs mass $1$ per layer; buy $i$ bits from a layer of measure $2^{-i^2}$ and the costs are summable.
* **What it is for.** If $X \in \bigcap\_i S\_{i^2}$, some prefix of $X$ lies in every $S\_{i^2}$, and (M) gives $K(X \upharpoonright n) \le n - i + O(1)$ for it — so $X$ is 1-nonrandom in the sense below. Read contrapositively, this is the **incompressible ⇒ typical** half (1-random ⇒ Martin-Löf random) of Thm. 4.1 in the next chapter; the other half is the exercise "Typicality implies incompressibility" there.

## 1-randomness

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.8 — 1-randomness)</span></p>

$X \in \lbrace 0,1\rbrace^{\omega}$ is **1-random** iff there is a constant $c$ with

$$K(X \upharpoonright n) \ge n - c \qquad \text{for all } n,$$

and **1-nonrandom** otherwise.

</div>

* **Negation:** $X$ is 1-nonrandom $\iff$ for every $c$ some $n$ has $K(X \upharpoonright n) < n - c$ $\iff$ $K(X \upharpoonright n) - n$ is **unbounded below**. A bounded deficit is harmless; a deficit that grows without bound kills the sequence.
* **Why this is not vacuous for $K$.** For plain $C$, Thm. 1.3 shows *every* sequence has $C(A \upharpoonright n) < n - c$ infinitely often, so the analogous definition has no models. For $K$ the maximal complexity at length $n$ is $n + K(\mathrm{bin}(n))$ (3.7) and the fraction of words with $K > n$ tends to $1$ (3.7.1): there is room above the length, so a sequence *can* stay above $n - c$ forever. The notes get existence — and the equivalence with Martin-Löf randomness — from Thm. 4.1 in the next chapter.
* **Existence in one line** (beyond the notes; Markov on $W\_m$ as in step 3 of 3.7). Fewer than $2^{m-c} W\_m$ words $\tau$ of length $m$ have $K(\tau) < m - c$, and each has $2^{n-m}$ extensions of length $n$, so the words of length $n$ with *some* compressible prefix number at most

$$\sum_{m \le n} 2^{n-m} \cdot 2^{m-c}\, W_m \;\le\; 2^{\,n-c}\sum_m W_m \;\le\; 2^{\,n-c}.$$

  For $c \ge 1$ at least $2^n(1 - 2^{-c}) > 0$ words of every length have **all** prefixes incompressible; they form an infinite, finitely branching tree, and König's lemma yields a path $X$ with $K(X \upharpoonright n) \ge n - c$ for all $n$.
* **Only one constant.** The slack $c$ depends on $X$ but not on $n$; "$\ge n - c$" is the prefix-free version of the "constant-width band" that failed for $C$.

## One-glance summary

| Result | Statement | Tool |
| :-- | :-- | :-- |
| 3.1 | i.c.m. $=$ partial $F$ with $\sum 2^{-F} \le 1$ and c.e. epigraph | Kraft + upper-semicomputability, no machines |
| 3.2 | computable enumeration of all i.c.m.s | Kraft filter (21) on all c.e. epigraphs; good streams pass unchanged |
| 3.3 / 3.4 / 3.5 | $\widetilde F = \min\_i (F\_i + i + 1)$ is a **total** i.c.m. | $\max \le \sum$, $\sum\_i 2^{-(i+1)} = 1$; atomic measures |
| 3.6 / (M) | $K = \widetilde F \pm O(1)$; hence $K \le F + c\_F$ for every i.c.m. $F$ | $K$ is in the list; epigraph of $\widetilde F$ is a KC set |
| 3.7 | $\sharp\lbrace \sigma \in 2^n : K(\sigma) \le n + K(\mathrm{bin}(n)) - r\rbrace \le 2^{n-r+c}$; $\max\_{l = n} K = n + K(\mathrm{bin}(n)) \pm O(1)$ | $W\_n \le 2^c 2^{-K(\mathrm{bin}(n))}$ via (M); Markov; 2.29 |
| 3.7.1 | fraction of length-$n$ words with $K > n$ tends to $1$ | $r := K(\mathrm{bin}(n))$, and $K(\mathrm{bin}(n)) \to \infty$ |
| Ex. (upper bounds) | computable $f$: $\sum 2^{-f(n)} < \infty \iff K(\mathrm{bin}(n)) \le f(n) + c$ | ($\Rightarrow$) (M) on $f + d$; ($\Leftarrow$) 2.26 |
| Ex. (Schnorr) | $l - n$ on $S\_n$ is **not** an i.c.m.; $l - i + O(1)$ on $S\_{i^2}$ is | each layer costs $1$; $\sum 2^{i - i^2} < \infty$ |
| 3.8 | $X$ 1-random $\iff$ $\exists c\, \forall n: K(X \upharpoonright n) \ge n - c$ | 3.7.1 removes the obstruction of Thm. 1.3; existence via König, or Thm. 4.1 |

**Punchline.** Strip $K$ down to the two properties that survived Part 2 — a c.e. epigraph and a Kraft sum $\le 1$ — and call anything with those properties an information content measure. The class can be effectively listed (3.2), its penalized lower envelope $\widetilde F$ is again a member (3.4), and $\widetilde F$ *is* $K$ up to a constant (3.6): $K$ is the **least** information content measure, a machine-free characterization that mirrors Part 1's characterization of $C$ with counting replaced by mass. The usable form (M), "$K \le F + c\_F$ for every i.c.m. $F$", is the new upper-bound tool — exhibiting a function replaces building a machine — and its first application is Chaitin's counting theorem: the mass of a length is at most the mass of its length, so at most $2^{n-r+c}$ words of length $n$ sit $r$ bits below the ceiling $n + K(\mathrm{bin}(n))$ (3.7), and almost every word has $K > l$ (3.7.1). That surplus is precisely what plain $C$ lacked, and it is what lets Def. 3.8 ask for *all* prefixes to be incompressible without running into Thm. 1.3.
