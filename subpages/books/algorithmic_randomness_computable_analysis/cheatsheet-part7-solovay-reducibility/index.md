---
title: "Cheatsheet — Part 7: Solovay Reducibility"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - computable-analysis
  - computability
  - solovay-reducibility
  - chaitin-constant
  - left-ce-reals
  - martin-löf-test
  - kolmogorov-complexity
  - kucera-slaman
  - cheatsheet
---

# Cheatsheet — Part 7: Solovay Reducibility

*Exam recap: definitions, formulas and key results only, covering the chapter Solovay Reducibility (7.1–7.6 together with the structure results from the exercises — the preorder properties, the computable-reals proposition, the d.c.e. reals — and the Kučera–Slaman theorem) up to (but excluding) Relative Randomness and Relative Complexity. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}#solovay-reducibility). Prerequisites in [Part 1]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part1-plain-complexity/' | relative_url }}), [Part 2]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part2-prefix-free-complexity/' | relative_url }}), [Part 3]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part3-information-content-measures/' | relative_url }}), [Part 4]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part4-incompressibility-typicality/' | relative_url }}), [Part 5]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part5-computable-approximations-of-reals/' | relative_url }}) and [Part 6]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part6-randomness-of-reals/' | relative_url }}).*

## Where we are

Part 5 built the hierarchy computable $\subsetneq$ left-c.e. $\subsetneq$ c.a. and showed, in the Solovay-domination exercise, that the approximation of Chaitin's $\Omega$ is the *slowest* of all left-c.e. approximations; Part 6 proved that $\Omega$ is Martin-Löf random (Thm. 6.2). This part turns "converges at least as fast as" into a reducibility $\le\_S$ and shows that, on the left-c.e. reals, randomness is a *position in an order*:

$$\gamma \text{ left-c.e. is Martin-Löf random} \iff \alpha \le_S \gamma \text{ for every left-c.e. } \alpha \qquad (\text{Thm. 7.6}).$$

| step | result | what it contributes |
| :-- | :-- | :-- |
| definition | 7.1 $\alpha \le\_S \beta$ | a partial computable $g$ turns lower bounds $q < \beta$ into lower bounds $g(q) < \alpha$, with gap $\le c \cdot$ gap |
| structure | Properties P1–P3; computable-reals proposition C1–C3 | preorder, $\le\_S \Rightarrow \le\_K$; left-c.e. and computable reals closed downward; computable reals $=$ least degree; a right-c.e. real with nothing below it |
| 7.2 | four speed-of-convergence faces | (iv): one synchronized pair with $\alpha - a\_n < c(\beta - b\_n)$ |
| 7.3 | randomness travels upward | test translation: a test catching $\beta$ becomes a test catching $\alpha$, at the price of $N$ layers |
| 7.4 | every left-c.e. real is $\le\_S \Omega$ | halting probabilities embed into $\Omega$ program by program |
| 7.5 | every left-c.e. real is $\le\_S$ every random left-c.e. real | Kučera–Slaman test: a $\beta$-approximation that stalls under an $\alpha$-increment is caught |
| 7.6 | random left-c.e. reals $=$ greatest Solovay degree | 7.5 for "random ⇒ top", 7.3 for "top ⇒ random" |

**The one idea.** A Solovay reduction is a computable *exchange rate between precisions from below*: whoever knows $\beta$ to within $\varepsilon$ from below learns $\alpha$ to within $c\varepsilon$ from below. Every result of the chapter is this sentence in other coordinates — left cuts (7.1), approximations (7.2), null tests (7.3, 7.5), halting masses (7.4), prefixes (P3).

## Definition 7.1

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.1 — Solovay reducibility)</span></p>

A real $\alpha$ is **Solovay reducible** to a real $\beta$, written $\alpha \le\_S \beta$, if there are a constant $c > 0$ and a partial computable $g :\subseteq \mathbb{Q} \to \mathbb{Q}$ with

$$0 < \alpha - g(q)\downarrow < c\,(\beta - q) \qquad \text{for every rational } q < \beta. \tag{S}$$

$\alpha$ and $\beta$ are **Solovay equivalent** if $\alpha \le\_S \beta \le\_S \alpha$; the equivalence classes are the **Solovay degrees**, partially ordered by $\le\_S$.

</div>

* **Lower bounds in, lower bounds out.** $g$ maps $L(\beta)$ into $L(\alpha)$, and the gap left below $\alpha$ is at most $c$ times the gap left below $\beta$; so $g(L(\beta))$ is cofinal in $L(\alpha)$. **Nothing is asked above $\beta$** — this one-sidedness is why left-c.e. reals are the natural domain (P2, C2) and right-c.e. reals misbehave (C3).
* **The constant is part of the data** and is what every construction pays: $2^{e+1}$ in 7.4 (coding a machine into $\widetilde U$), $2^k$ in 7.5 (the level at which $\beta$ escapes a test), $N$ dropped layers in 7.3. ⚠️ The source prints Def. 7.1 without $c$ and then uses a constant from Prop. 7.2 onward; the version with $c$ is Solovay's.
* **Irrationality is used, not assumed.** 7.3 and 7.5 need $l < \beta$ *strictly* for a test interval $[l, r] \ni \beta$ and get it from "random ⇒ irrational".

## Structure: the exercise propositions

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(P1–P3 — Solovay reducibility on $\mathbb{R}$)</span></p>

* **(P1)** $\le\_S$ is reflexive and transitive — a preorder on $\mathbb{R}$, so it induces a degree structure.
* **(P2)** The left-c.e. reals are closed downward: $\beta$ left-c.e. and $\alpha \le\_S \beta$ imply $\alpha$ left-c.e. So the Solovay degrees of the left-c.e. reals sit inside those of $\mathbb{R}$.
* **(P3)** $\alpha \le\_S \beta \implies K(\alpha \upharpoonright n) \le K(\beta \upharpoonright n) + O(1)$ (the notes write the $O(1)$ on the left with a minus sign — same statement).

</div>

* **P1:** compose the reductions; constants multiply. (Reflexive: $g = \mathrm{id}$ with $c = 2$.)
* **P2:** enumerate the c.e. cut $L(\beta)$, apply $g$, take running maxima — a left-c.e. approximation of $\alpha$, cofinal by (S).
* **P3 — precision transfer is description transfer.** A prefix $\beta \upharpoonright (n + d)$ *is* a lower bound of $\beta$ with gap $< 2^{1-(n+d)}$; for $d$ with $c\,2^{1-d} < 1$, (S) puts $\alpha$ in an interval of length $< 2^{-n}$ with known left end, and one extra bit fixes $\alpha \upharpoonright n$. So $n + O(1)$ bits of $\beta$ describe $n$ bits of $\alpha$.
* **Consequence (not in the notes).** The same computation shows $\le\_S \Rightarrow \le\_T$; the converse fails badly — $\Omega$ has many Turing-equivalent left-c.e. companions that are not random, hence not Solovay-equivalent to it (Thm. 7.6).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(C1–C3 — Solovay reducibility and computable reals)</span></p>

* **(C1)** The computable reals are closed downward in $\mathbb{R}$: $\beta$ computable and $\alpha \le\_S \beta$ imply $\alpha$ computable.
* **(C2)** The computable reals form the **least** Solovay degree on the left-c.e. reals: $\alpha$ computable and $\beta$ left-c.e. imply $\alpha \le\_S \beta$ (even with $c = 1$).
* **(C3)** There is a right-c.e. real $\beta$ with $\alpha \not\le\_S \beta$ for *every* left-c.e. $\alpha$, the computable ones included.

</div>

* **C1:** shift an effective approximation of $\beta$ to the left ($b\_n - 2^{-n}$), push it through $g$, and reindex by $\lceil 1 + \log\_2 c \rceil$ to restore the exact bound $2^{-n}$.
* **C2:** given $q < \beta$, search the left-c.e. approximation of $\beta$ for a stage $b\_s > q$ and answer with an approximant of $\alpha$ that is accurate to better than $b\_s - q$, shifted below $\alpha$. The only use of "left-c.e." is that search.
* **C3 — sketch (the notes leave it open).** By C2 and P1 it suffices to make $\tfrac12 \not\le\_S \beta$. Finite injury on a nonincreasing $r\_s \searrow \beta$: requirement $(g\_e, 2^k)$ holds a witness $q < r\_s$ and, if $g\_e(q)\downarrow < \tfrac12$, lowers $r$ to $q + 2^{-k}(\tfrac12 - g\_e(q))$, so that the ratio in (S) fails at $q$ forever; otherwise the witness violates (S) by itself. Right-c.e. reals have no computable supply of lower bounds, which is what Def. 7.1 is about.

## Proposition 7.2: four faces of "at least as fast"

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.2 — Speed-of-convergence characterizations)</span></p>

For left-c.e. reals $\alpha$ and $\beta$ the following are equivalent.

* **(i)** $\alpha \le\_S \beta$.
* **(ii)** There is a constant $c$ such that for *every* pair of left-c.e. approximations $(a\_n) \nearrow \alpha$, $(b\_n) \nearrow \beta$ there is a computable $g : \mathbb{N} \to \mathbb{N}$ with $\alpha - a\_{g(n)} < c\,(\beta - b\_n)$ for all $n$.
* **(iii)** For every left-c.e. approximation $(b\_s) \nearrow \beta$ there are a constant $d$ and a left-c.e. approximation $(a\_s) \nearrow \alpha$ with $a\_s - a\_{s-1} \le d\,(b\_s - b\_{s-1})$ for all $s$.
* **(iv)** There are left-c.e. approximations $(a\_n) \nearrow \alpha$, $(b\_n) \nearrow \beta$ and a constant $c$ with $\alpha - a\_n < c\,(\beta - b\_n)$ for all $n$.

</div>

* **(i) ⇒ (ii):** wait until $a\_m$ overtakes $f(b\_n)$ — a *computable* condition, unlike the target inequality.
* **(i) ⇒ (iii):** follow $f(b\_s)$ but **cap each step** at $d(b\_s - b\_{s-1})$; the invariant $\alpha - a\_s < d(\beta - b\_s)$ survives both cases and forces convergence. ⚠️ "$a\_0 < a\_1 < \cdots$" in (iii) must be read as nondecreasing.
* **(ii) ⇒ (iv):** running maximum repairs monotonicity. **(iii) ⇒ (iv):** tails telescope — compare *remaining* increments. **(iv) ⇒ (i):** given $q < \beta$, read off the pair with $b\_n \ge q$ and answer $a\_n$.
* **Reading.** **(iv) is the workhorse** — 7.4 and 7.5 each produce one synchronized pair. (iii) is the increment form (the Solovay-domination exercise of Part 5 is (iii) for $\alpha \le\_S \Omega$); (ii) says slowness is a property of the *real*, not of the approximation. Left-c.e. on both sides is essential: $f$ must only be fed inputs below $\beta$, and lower bounds of $\beta$ must be enumerable — which is what C3 exploits.

## Proposition 7.3: randomness travels upward

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.3 — Solovay reducibility transfers randomness upward)</span></p>

If $\alpha$ is Martin-Löf random and $\alpha \le\_S \beta$, then $\beta$ is Martin-Löf random. Equivalently: everything Solovay-below a nonrandom real is nonrandom.

</div>

* **Idea (test translation).** Take $c = 2^N$ and a test $(L\_i)$ catching $\beta$. Push every basic interval $[l, r]$ of $L\_i$ forward through $g$ to $[\,g(l),\ g(l) + 2^N(r - l)\,]$: measures scale by $2^N$, so dropping the first $N$ layers gives a test $(M\_i)$. Since $\beta$ is irrational, $l < \beta$ strictly, hence $g(l)\downarrow$ and (S) puts $\alpha$ into the image interval — $\alpha$ is caught, contradiction.
* **Reading.** $l$ is the only point of $[l, r]$ known to be a lower bound of $\beta$, and $r - l$ the only computable bound on the gap $\beta - l$. Intervals around $\beta$ are pushed *forward* to intervals around $\alpha$ and there is no computable way back: **randomness rises along $\le\_S$, nonrandomness sinks.**

## Proposition 7.4: $\Omega$ is on top

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.4 — $\Omega$ is Solovay-above every left-c.e. real)</span></p>

Every left-c.e. real $\alpha$ satisfies $\alpha \le\_S \Omega$.

</div>

* **Idea (weight transfer).** $\alpha$ is the halting probability of a prefix-free $\widetilde M$ (Prop. 5.9), which $\widetilde U$ simulates behind a code of length $e$. Read both approximations off the same enumeration of $\operatorname{dom}\widetilde U$: the $\alpha$-mass still missing at stage $s$ is $2^e$ times the mass of the not-yet-halted codes $i\sigma$, which is part of the $\Omega$-mass still missing — $\alpha - a\_s \le 2^e(\Omega - w\_s)$, i.e. (iv) with constant $2^{e+1}$.
* **Reading.** Part 5's Solovay-domination exercise is the increment form (iii) of the same fact. **All $\Omega$'s are alike:** two universal machines give Solovay-equivalent halting probabilities, and by 7.3 all of them are random.

## Proposition 7.5 and the Kučera–Slaman theorem

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.5 — Random left-c.e. reals are Solovay-complete)</span></p>

If $\alpha$ and $\beta$ are left-c.e. and $\beta$ is Martin-Löf random, then $\alpha \le\_S \beta$.

</div>

Throughout: $\alpha, \beta \in (0,1)$ with left-c.e. approximations $(a\_n)$, $(b\_n)$, and $\beta$ random, hence irrational. The notes split the proof into three exercises.

* **Step 1 — steep spots are rare.** A nondecreasing piecewise linear $g$ with total rise $\le \alpha < 1$ has left derivative $> 2^k$ only on a set of measure $< 2^{-k}$ (Markov against the rise). The test below spends the increments of $\alpha$'s approximation as interval lengths in exactly this way.
* **Step 2 — the test.** For level $i$, lay down the interval $(b\_{n\_j},\ b\_{n\_j} + 2^{-i}(a\_{j+1} - a\_j))$ just above the current $\beta$-approximant, and advance to the next $\alpha$-increment only once $\beta$'s approximation has **climbed out** of it. Total length $\le 2^{-i}\alpha$, so $(U\_i)$ is a Martin-Löf test.
* **Step 3 — the dichotomy.** If $\beta$'s approximation ever **stalls** under a laid-down interval, $\beta$ lies in it (irrationality excludes the endpoint): $\beta \in U\_i$. If it never stalls, every $\alpha$-increment is $< 2^i$ times the matching $\beta$-increment, tails telescope, and the synchronized pair is (iv) with constant $2^i$. A random $\beta$ escapes some $U\_i$, so for that $i$ it never stalls. $\square$
* **Reading.** Either $\beta$ is pinned down by a scaled $\alpha$-increment, or it is at least as fast as $\alpha$ increment by increment. The constant $2^i$ is the level at which $\beta$ escapes the test — it exists but is not uniform in $\beta$. Used: left-c.e.-ness of $\alpha$ (budget), of $\beta$ (computable search), $\alpha < 1$, irrationality of $\beta$.
* **The sheet's test works too (not in the notes).** The sheet's $L\_i = \bigcup\_{n<m}(b\_m,\ b\_n + 2^{-i}(a\_m - a\_n))$ — "some chord of the interpolation of $(b\_n, a\_n)$ ending above $x$ has slope $> 2^i$" — also has $\lambda(L\_i) \le 2^{-i}\alpha$, by the rising-sun lemma applied to $g(x) - 2^i x$; with it, escaping $L\_k$ gives (iv) for the *original* approximations with constant $2^{k+1}$.

## Theorem 7.6: Kučera–Slaman

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.6 — Kučera–Slaman theorem, 2001)</span></p>

On the set of left-c.e. reals, the Martin-Löf random reals form the **greatest Solovay degree**: a left-c.e. $\gamma$ is random iff $\alpha \le\_S \gamma$ for every left-c.e. $\alpha$.

</div>

**Proof.** $\Omega$ is a random left-c.e. real (5.10, 6.2). By 7.5 every random left-c.e. real is above all left-c.e. reals, so the greatest degree exists and contains them all; conversely, $\gamma$ in the greatest degree has $\Omega \le\_S \gamma$, and 7.3 makes $\gamma$ random. $\square$

* **Randomness as an order-theoretic property.** On the left-c.e. reals, "random" means "as hard to approximate from below as a left-c.e. real can be"; all random left-c.e. reals are Solovay equivalent to $\Omega$. For left-c.e. $\gamma$: random $\iff$ $\Omega \le\_S \gamma$ $\iff$ $\gamma$ is above every left-c.e. real.
* **The picture.** Bottom: the computable reals (C2). Top: the random reals (7.6). In between: noncomputable nonrandom left-c.e. reals, e.g. $0.A$ for a noncomputable c.e. set $A$ ($K(A \upharpoonright n) = O(\log n)$: $n$ and the number of elements below $n$ determine the prefix). Turing reducibility sees none of this — $0.A$ with $A \equiv\_T \emptyset^{\prime}$ is Turing-equivalent to $\Omega$ and sits strictly below it in the Solovay order.
* **Two faces, third reading.** Part 5: $\Omega$ computes the halting problem. Part 6: $\Omega$ is incompressible. Part 7: $\Omega$ is the slowest left-c.e. real, and slowest *is* random — one object seen through $\le\_T$, $K$ and $\le\_S$.
* **Forward.** 7.6 says nothing about random reals that are not left-c.e. (e.g. $1 - \Omega$); §10 extends $\le\_S$ to all c.a. reals by the two-sided reducibility $\le^{2a}\_S$, which agrees with $\le\_S$ on the left-c.e. reals (Thm. 10.2).

## D.c.e. reals: three exercises

A real $\gamma$ is **d.c.e.** ("difference of c.e.") if $\gamma = \alpha - \beta$ for left-c.e. $\alpha, \beta$; it is **right-c.e.** if $-\gamma$ is left-c.e. The notes state three exercises and leave their solutions blank.

* **(1) d.c.e. $\iff$ bounded-variation approximation.** $\gamma$ is d.c.e. iff some computable $(q\_n) \to \gamma$ has $\sum\_n \lvert q\_{n+1} - q\_n \rvert < \infty$: (⇒) $a\_n - b\_n$ has variation $\le \alpha + \beta$; (⇐) split the increments by sign into two nondecreasing bounded sequences.
* **(2) random d.c.e. reals are one-sided.** A d.c.e. $\gamma$ that is neither left- nor right-c.e. is not random: the intervals between consecutive $q\_n$ form a Solovay test (total length = variation; Def. 9.3 / Thm. 9.4), so a random $\gamma$ is straddled only finitely often, and from then on the running max or min is a one-sided approximation.
* **(3) d.c.e. reals are closed downward** relative to the two-sided reducibility $\le^{2a}\_S$ of §10 within the c.a. reals — the d.c.e. analogue of P2; proved in the §10 material.

## One-glance summary

| Result | Statement | Tool |
| :-- | :-- | :-- |
| 7.1 | $\alpha \le\_S \beta$ $:\iff$ $\exists\, c, g$ partial computable: $0 < \alpha - g(q) < c(\beta - q)$ for all rational $q < \beta$ | lower bounds of $\beta$ ↦ lower bounds of $\alpha$, gap ratio $\le c$ |
| P1 | preorder; Solovay degrees | compose; constants multiply |
| P2 | left-c.e. reals closed downward | enumerate $L(\beta)$, apply $g$, running max |
| P3 | $\le\_S \Rightarrow K(\alpha \upharpoonright n) \le K(\beta \upharpoonright n) + O(1)$ | $n + d$ bits of $\beta$ are a lower bound with gap $< 2^{-n}/c$; one extra bit |
| C1 | computable reals closed downward | shift $b\_n - 2^{-n}$, push through $g$, reindex |
| C2 | computable $\le\_S$ every left-c.e. ($c = 1$): least degree | find $b\_s > q$, answer an $\alpha$-approximant accurate to $b\_s - q$ |
| C3 | a right-c.e. $\beta$ with no left-c.e. real below it | reduce to $\tfrac12 \not\le\_S \beta$ (C2 + P1); finite injury |
| 7.2 | (i) cut map $\iff$ (ii) any approximations, computable delay $\iff$ (iii) increments $\le d \cdot$ increments $\iff$ (iv) one synchronized pair | wait to overtake $f$; capped follower; running max; telescoping tails; read off at $q$ |
| 7.3 | $\alpha$ random, $\alpha \le\_S \beta$ ⇒ $\beta$ random | $[l, r] \mapsto [g(l), g(l) + 2^N(r-l)]$; drop $N$ layers; $l < \beta$ by irrationality |
| 7.4 | left-c.e. $\alpha \le\_S \Omega$, constant $2^{e+1}$ | missing $\alpha$-mass $= 2^e \cdot$ part of missing $\Omega$-mass; (iv) |
| 7.5 | left-c.e. $\alpha \le\_S$ random left-c.e. $\beta$ | lay down $2^{-i}\Delta a\_j$ above $b$, advance when $b$ climbs out; stall ⇒ $\beta \in U\_i$, never stall ⇒ (iv) with $2^i$ |
| sheet's $L\_i$ | $\bigcup\_{n<m}(b\_m, b\_n + 2^{-i}(a\_m - a\_n))$ is also a test, $\lambda \le 2^{-i}\alpha$ | rising sun lemma for $g - 2^i x$ |
| 7.6 | random left-c.e. reals $=$ greatest Solovay degree $=$ degree of $\Omega$ | 6.2 + 7.5 (⇐), 7.3 (⇒) |
| d.c.e. | $\alpha - \beta$ $\iff$ finite-variation approximation; random d.c.e. ⇒ left- or right-c.e. | sign-split increments; Solovay test of the steps |

**Punchline.** Solovay reducibility is a computable exchange rate between lower bounds: $\alpha \le\_S \beta$ when every rational $q < \beta$ can be converted into a rational $g(q) < \alpha$ whose remaining gap is at most $c$ times the remaining gap below $\beta$. On the left-c.e. reals this is the same as "some approximation of $\alpha$ converges at least as fast as one of $\beta$", and it is monotone for randomness: a test that traps $\beta$ is translated, interval by interval through $g$, into a test that traps $\alpha$, so randomness can only travel upward. Chaitin's $\Omega$ is on top because every halting probability is a scaled part of it; every random left-c.e. real is on top because a $\beta$-approximation that ever stalls under a scaled increment of another left-c.e. approximation is caught by a Martin-Löf test. Together (Kučera–Slaman): **a left-c.e. real is Martin-Löf random iff it is Solovay-complete, iff it is Solovay-equivalent to $\Omega$** — on this class, randomness is not a property of bits or of tests but a position in an order, and there is exactly one top.
