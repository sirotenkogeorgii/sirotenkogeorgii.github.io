---
title: "Cheatsheet — Part 9: Universal Randomness Tests"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - computable-analysis
  - computability
  - martin-löf-test
  - schnorr-randomness
  - solovay-test
  - universal-test
  - borel-cantelli
  - cheatsheet
---

# Cheatsheet — Part 9: Universal Randomness Tests

*Exam recap: definitions, formulas and key results only, covering the chapter Universal Randomness Tests (the definition of a universal test, Theorems 9.1–9.6, the five-part exercise behind Theorem 9.2 and the exercise on Schnorr randomness under total Solovay reductions) up to (but excluding) Relative Randomness of Computably Approximable Reals. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}#universal-randomness-tests). Prerequisites in [Part 1]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part1-plain-complexity/' | relative_url }}), [Part 3]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part3-information-content-measures/' | relative_url }}), [Part 4]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part4-incompressibility-typicality/' | relative_url }}), [Part 6]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part6-randomness-of-reals/' | relative_url }}) and [Part 7]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part7-solovay-reducibility/' | relative_url }}).*

## Where we are

Parts 1–4 defined Martin-Löf randomness by tests (Def. 1.5: uniformly c.e. open layers with $\lambda(L\_i) \le 2^{-i}$; failing means lying in every layer) and met the **Schnorr** variant with $\lambda(L\_i) = 2^{-i}$ exactly (Part 3 exercise); Parts 6–7 ran tests on reals (6.2, 7.3, 7.5). So far every "random" has quantified over **all** tests. This part asks when a single test suffices — a **universal** test, failed by exactly the nonrandom sequences — and introduces a second, flat test format, the **Solovay test**, which is often the most convenient one in practice.

| test format | measure bookkeeping | characterizes | universal test? |
| :-- | :-- | :-- | :-- |
| Martin-Löf test | $\lambda(L\_i) \le 2^{-i}$, c.e. from below | Martin-Löf randomness (Def. 1.5) | **yes** (9.1) |
| Schnorr test | $\lambda(L\_i) = 2^{-i}$, computable | Schnorr randomness | **no** (9.2) |
| Solovay test | $\sum\_i \lambda(I\_i) < \infty$ | Martin-Löf randomness (9.4) | **yes** (9.5) |
| total Solovay test | $\sum\_i \lambda(I\_i)$ computable | Schnorr randomness (9.6) | **no** (9.2 + 9.6) |

**The one idea.** *Universality is decided by whether the measure requirement can be enforced effectively.* "$\lambda \le 2^{-i}$" can only be **violated** at a finite stage, so any c.e. candidate can be trimmed into a genuine test without touching the genuine ones; that yields a computable list of *all* tests, and a shifted diagonal union of the list is one test that catches everything. "$\lambda = 2^{-i}$" can be neither certified nor repaired at a finite stage — and, worse, exact computable measures let a computable sequence steer around any single Schnorr test, so no Schnorr test even catches the computable sequences. Solovay tests replace layers by one flat list with finite total length (Borel–Cantelli); coverage multiplicity recovers the layers, and "computable total" is the flat-format name for "exact layer measures".

## Universal tests

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Universal randomness test)</span></p>

For a randomness notion $X$, an $X$-randomness test $T$ is **universal** if *all* $X$-nonrandom sequences fail $T$.

</div>

* **Failure set $=$ nonrandom set.** The converse inclusion is automatic (whatever fails an $X$-test is $X$-nonrandom), so a universal test's failure set is *exactly* the $X$-nonrandom sequences — one effectively presented null set instead of a countable family of them. For Martin-Löf randomness this makes the nonrandom sequences a single $\Pi^0\_2$ set $\bigcap\_i \bar L\_i$.
* **What it buys.** To build an $X$-random real, survive one test; to certify one, check one test; the infinite quantifier "for all tests" collapses to a single condition one can construct against.

## Theorem 9.1: a universal Martin-Löf test

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.1 — Existence of a universal Martin-Löf test)</span></p>

There exists a universal Martin-Löf test.

</div>

* **Idea (enumerate, trim, shifted diagonal).** List all uniformly c.e. sequences of open sets $U^p = (U^p\_1, U^p\_2, \dots)$ — the $p$-th c.e. set of pairs $(\sigma, j)$ read as "$\sigma$ enters layer $j$". **Trim:** enumerate each layer cylinder by cylinder, but stop the layer for good the moment the next cylinder would push its measure above $2^{-j}$ — a decidable comparison of dyadic rationals. Every trimmed $L^p$ is a Martin-Löf test, and a genuine test is never trimmed, so $(L^p)\_p$ enumerates **all** Martin-Löf tests. **Diagonalize with a shift:**

  $$\bar L_i := \bigcup_{p \ge 0} L^p_{i+p+1}, \qquad \lambda(\bar L_i) \le \sum_{p \ge 0} 2^{-(i+p+1)} = 2^{-i}.$$

  A sequence failing $L^p$ lies in $L^p\_{i+p+1} \subseteq \bar L\_i$ for every $i$, so it fails $\bar L$.
* **Why the shift.** The $p$-th test contributes only its $(i+p+1)$-st, exponentially thin, layer to $\bar L\_i$; that is what keeps the countable union a test. Without the shift the measures $2^{-i}$ would be summed over all $p$.
* **Why trimming is legitimate.** Exceeding "$\lambda \le 2^{-j}$" is a c.e. event — the measure of a c.e. open set is left-c.e. — and the stop only ever fires for junk. This is the one place where "$\le$" rather than "$=$" is used (Remark below 9.2).

## Theorem 9.2: no universal Schnorr test

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.2 — No universal Schnorr test)</span></p>

There exists no universal Schnorr test.

</div>

* **Idea (computable sequences steer around every Schnorr test).** Every computable sequence is Schnorr nonrandom — its own cylinders $([[A \upharpoonright i]])\_{i \ge 1}$ form a Schnorr test, measures exactly $2^{-i}$. A universal $L$ would therefore have every computable sequence in $L\_1$. But $L\_1$ has **exact computable measure** $\tfrac12$, and from exact measures one can *compute* a sequence that avoids $L\_1$: at each step take the half-cylinder in which $L\_1$ has the smaller relative measure. So no Schnorr test is universal.
* **The five exercise parts** (Sheet 8, Ex. 2):
  * **(1)** $A \in U$ c.e. open $\Rightarrow$ some $[[A \upharpoonright n]] \subseteq U$. Used contrapositively in (3): if no prefix-cylinder of $A$ lies inside $U$, then $A \notin U$.
  * **(2)** If $\lambda(U) = c$ is computable, so is $\lambda(U \cap [[\sigma]])$, uniformly: it is left-c.e., and $c$ minus the other cylinders' left-c.e. measures makes it right-c.e. (Prop. 2.8). Hence from $\varepsilon$ one computes a bit $b$ with $\lambda(U \cap [[b]]) \le (1+\varepsilon)\,c/2$.
  * **(3)** Greedy descent with $\varepsilon\_n := 3^{-2^n}$: the relative measure $m\_n := 2^n \lambda(U \cap [[A \upharpoonright n]])$ grows by at most the factor $1 + \varepsilon\_n$ per bit, so $m\_n \le \tfrac12 \prod\_k (1 + 3^{-2^k}) = \tfrac12 \cdot \tfrac32 = \tfrac34 < 1$ (telescoping product). A cylinder inside $U$ would have $m\_n = 1$; so by (1) $A \notin U$, and $A$ is computable since every bit comes from a terminating computation.
  * **(4)** Apply (3) to $U := L\_1$ of a Schnorr test ($\lambda = \tfrac12$ exactly): a computable sequence that **passes** $L$. ⚠️ The sheet says "fails $L$"; the layers need not be nested, so that is false, and (5) needs "passes".
  * **(5)** Computable $\Rightarrow$ Schnorr nonrandom (own cylinders), so a universal $L$ would put the computable $A$ of (4) into $L\_1$ — contradiction.
* **Why trimming cannot save it (Remark).** "$\lambda(L\_i) = 2^{-i}$" is a $\Pi$-type requirement: no finite stage certifies it, and truncating a defective candidate only lowers the measure, breaking the test the other way. So the Schnorr tests are not effectively enumerable, and by 9.2 no other construction compensates.

## Definition 9.3: Solovay tests

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(9.3 — Solovay test, total Solovay test)</span></p>

* A **Solovay test** $S = (I\_0, I\_1, \dots)$ is a computable sequence of intervals with rational endpoints such that

  $$\sum_{i \in \mathbb{N}} \lambda(I_i) < \infty.$$

  A sequence $A$ **fails** $S$ if the real $0.A$ lies in infinitely many $I\_i$.

* $S$ is **total** iff $\sum\_i \lambda(I\_i)$ is a computable real.

</div>

* **Unstratified.** No layers and no index in the failure condition; the measure condition is one global convergent series. Borel–Cantelli makes it a null test by construction: the points covered infinitely often form a null set.
* **Practical gain.** One keeps a single running total finite instead of distributing intervals into layers of controlled measure — e.g. the steps of a finite-variation approximation (d.c.e. exercise, Part 7).
* **Total.** "Total" is the flat-format analogue of "Schnorr": a computable total is what lets one compute how far down the list the remaining mass drops below $2^{-k}$ (9.6).

## Theorem 9.4: Solovay tests characterize Martin-Löf randomness

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.4 — Solovay characterization of Martin-Löf randomness)</span></p>

A sequence $A$ is Martin-Löf nonrandom $\iff$ it fails some Solovay test.

</div>

* **($\Rightarrow$) Flatten the layers.** Present each layer by pairwise disjoint cylinders (the splitting device of Thm. 4.1), turn cylinders into dyadic intervals $[0.\sigma,\ 0.\sigma + 2^{-l(\sigma)}]$ and dovetail all layers into one list. Total length $= \sum\_i \lambda(\bar L\_i) \le \sum\_i 2^{-i} = 1$; a sequence lying in every layer is hit once per layer, hence infinitely often.
* **($\Leftarrow$) Coverage multiplicity recovers the layers.** Discard a finite initial part so that the tail $j \ge N$ has total length $< 1$. Let $F(x) := \sharp\lbrace j \ge N : x \in (l\_j, r\_j)\rbrace$ and set $\bar L\_i := \lbrace F \ge 2^i \rbrace$, the points covered at least $2^i$ times. **Markov:** $2^i \lambda(\bar L\_i) \le \int F \, d\lambda = \sum\_{j \ge N}(r\_j - l\_j) < 1$. Each $\bar L\_i$ is c.e. open (the union of its finite-stage versions, which are finite unions of rational intervals), and $F(0.A) = \infty$ puts $A$ into every layer.
* **Why open intervals and irrationality.** A rational $0.A$ is computable, hence nonrandom outright; an irrational one is never an endpoint, so "hit infinitely often" means hit by infinitely many *open* intervals — which is what makes the layers open and c.e.
* **Reading.** The two directions trade stratification for multiplicity: the layer index $i$ becomes "the $i$-th hit", and "hit at least $2^i$ times" becomes a layer index. This is Borel–Cantelli made effective; the Markov step is the same budget-to-measure conversion as in Thm. 4.1 and Prop. 7.5.

## Theorem 9.5: a universal Solovay test

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.5 — Existence of a universal Solovay test)</span></p>

There exists a universal Solovay test.

</div>

* **Idea.** Flatten the universal Martin-Löf test $\bar L$ of 9.1 by the ($\Rightarrow$) translation of 9.4. Every Martin-Löf nonrandom sequence lies in every $\bar L\_i$, so it is hit infinitely often; by 9.4 these are exactly the sequences failing *some* Solovay test. Total length $\le 1$.
* **Necessarily not total.** Its total $\sum\_i \lambda(\bar L\_i)$ is only left-c.e., and no universal Solovay test can be total: the ($\Leftarrow$) construction of 9.6 would turn it into a Schnorr test failed by every sequence that fails it, in particular by every Schnorr nonrandom one — a universal Schnorr test against 9.2.

## Theorem 9.6: total Solovay tests characterize Schnorr randomness

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(9.6 — Solovay characterization of Schnorr randomness)</span></p>

A sequence is Schnorr nonrandom $\iff$ it fails some **total** Solovay test.

</div>

⚠️ The source prints "Schnorr **random** iff it fails some total Solovay test"; failing a test is what makes a sequence *non*random, exactly as in 9.4.

* **Rational case.** For rational $0.A$ both sides hold outright: $A$ is computable, hence Schnorr nonrandom, and it fails the total Solovay test $([0.A, 0.A], [0.A, 0.A], \dots)$ of total length $0$. So assume $0.A$ irrational, never an endpoint.
* **($\Rightarrow$) Flatten a Schnorr test.** The translation of 9.4; now the total length is $\sum\_k \lambda(L\_k) = \sum\_k 2^{-k} = 1$ **exactly**, a computable real. This is where exact layer measures pay — with "$\le$" the total would only be left-c.e.
* **($\Leftarrow$) Computable tails, then inflate.** From a computable total one can *compute* $N(k)$ with $\sum\_{i \ge N(k)} \lambda(I\_i) \le 2^{-k}$ — the only use of totality. The union $V\_k$ of the open tail intervals is c.e. open, has $\lambda(V\_k) \le 2^{-k}$, contains $0.A$ (hit infinitely often $\Rightarrow$ hit in the tail), and has **computable** measure (a finite part exactly, the rest below $\varepsilon$). Inflate to exact measure: $L\_k := V\_k \cup [0, \theta\_k)$ with $\theta\_k := \sup\lbrace q \in \mathbb{Q} : \lambda(V\_k \cup [0,q)) < 2^{-k} \rbrace$. The threshold is left-c.e., which is all that c.e.-openness of $[0, \theta\_k)$ needs, and continuity of $t \mapsto \lambda(V\_k \cup [0,t))$ gives $\lambda(L\_k) = 2^{-k}$ exactly.
* **Why the inflation works.** $\lambda(V\_k \cap [q, 1])$ is computable for rational $q$ by the left-c.e. $+$ right-c.e. trick of exercise (2), so "$\lambda(V\_k \cup [0,q)) < 2^{-k}$" is a $\Sigma^0\_1$ condition and $\theta\_k$ is left-c.e.; $\theta\_k$ itself need not be computable.
* **Reading.** Total $\leftrightarrow$ Schnorr is the dictionary Solovay $\leftrightarrow$ Martin-Löf with "computable" carried along: a computable total gives computable tails, computable layer measures, and finally exact layer measures by padding from the left. In every row of the table above the same pattern decides universality: a requirement that can only be violated effectively can be enumerated and diagonalized; one that demands an exactly computable value cannot.

## Exercise: Schnorr randomness travels upward along total reductions

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Totality and translation of Schnorr tests)</span></p>

If $\alpha$ is a Schnorr random real and $\alpha \le\_S \beta$ via a **total** computable $f$, then $\beta$ is Schnorr random as well.

</div>

* **Idea (test translation, Schnorr version; the notes leave the solution blank).** By contradiction: a Schnorr test $(L\_i)$ catches $\beta$, and $f$ has constant $c$. Push every basic interval $[l, r]$ of every layer through $f$ to $[\,f(l),\ f(l) + c\,(r - l)\,]$ and dovetail: a computable list of rational intervals with total length exactly $c \sum\_i \lambda(L\_i) = c$, a computable real — a **total Solovay test**. $\beta$ is irrational (else it is computable, hence so is $\alpha$ by C1, which is then Schnorr nonrandom), so every layer has an interval with $l < \beta < r$, and (S) puts $\alpha$ into its image. Thus $\alpha$ is hit infinitely often and is Schnorr nonrandom by 9.6 — contradiction.
* **Where totality enters.** With a partial $f$, intervals whose left endpoint is not below $\beta$ might never be translated, the enumeration would silently drop them, and the total length would be merely left-c.e.: a Solovay test, but not a total one. For Martin-Löf randomness (Prop. 7.3) this did not matter.
* **Reading.** Same forward push of intervals as in Prop. 7.3, but Martin-Löf randomness needed only "$\le$", dropped layers and a partial $f$; Schnorr randomness needs an exactly accounted total, and 9.6 converts it back into exact layers.

## One-glance summary

| Result | Statement | Tool |
| :-- | :-- | :-- |
| universal test | failure set $=$ exactly the $X$-nonrandom sequences | converse inclusion is automatic |
| 9.1 | a universal Martin-Löf test exists | trim c.e. candidates at $2^{-j}$ (c.e. violation); $\bar L\_i = \bigcup\_p L^p\_{i+p+1}$, $\lambda \le 2^{-i}$ |
| 9.2 | no universal Schnorr test | computable sequences are Schnorr nonrandom, yet greedy bisection on exact measures computes one passing any given Schnorr test ($m\_n \le \tfrac34$) |
| 9.3 | Solovay test: $\sum\_i \lambda(I\_i) < \infty$, fail $=$ hit infinitely often; total $=$ computable sum | Borel–Cantelli null set |
| 9.4 | Martin-Löf nonrandom $\iff$ fails a Solovay test | flatten disjoint cylinders to dyadic intervals / coverage count $F$, $\bar L\_i = \lbrace F \ge 2^i\rbrace$, Markov |
| 9.5 | a universal Solovay test exists | flatten the universal Martin-Löf test; necessarily not total |
| 9.6 | Schnorr nonrandom $\iff$ fails a **total** Solovay test | exact total $1$ / computable tails $N(k)$, $V\_k$ with computable measure, inflate by $[0, \theta\_k)$ with left-c.e. $\theta\_k$ |
| Ex. | $\alpha$ Schnorr random, $\alpha \le\_S \beta$ via total $f$ $\Rightarrow$ $\beta$ Schnorr random | push intervals through $f$: total Solovay test of length $c$; 9.6 |

**Punchline.** A universal test exists exactly when the measure bookkeeping of a test format is a condition one can only *violate* effectively. For Martin-Löf tests it is: any c.e. candidate is trimmed into a test the moment it would overspend, so all tests can be listed and a shifted diagonal of the list is a single test that catches every nonrandom sequence. For Schnorr tests it is not: exact measure can be neither certified nor repaired, and exact measure is precisely what lets a computable sequence bisect its way out of any given test — so no Schnorr test catches even the computable sequences. Solovay tests say the same thing in a flat language: one list of intervals with finite total length, failure means being hit infinitely often, and grading the line by coverage multiplicity turns "infinitely often" back into layers. Finite total ↔ Martin-Löf, computable total ↔ Schnorr, and universality follows the same split: **"at most" can be diagonalized, "exactly" cannot.**
