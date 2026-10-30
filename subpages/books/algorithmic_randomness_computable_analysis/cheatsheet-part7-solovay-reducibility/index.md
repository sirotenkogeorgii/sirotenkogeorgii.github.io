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

* **Lower bounds in, lower bounds out.** On the left cut $L(\beta)$ the function $g$ is total and maps into $L(\alpha)$; (S) says more: the gap left below $\alpha$ is at most $c$ times the gap left below $\beta$. Hence $g(q) \to \alpha$ as $q \nearrow \beta$ — the image $g(L(\beta))$ is cofinal in $L(\alpha)$.
* **Nothing is asked above $\beta$.** For $q \ge \beta$, $g(q)$ may diverge or output anything. A reduction is a promise about the left cut only; this one-sidedness is why the left-c.e. reals are the natural domain (P2, C2) and why right-c.e. reals misbehave (C3).
* **The constant is part of the data.** It is the exchange rate, and it is what every construction pays: $2^{e+1}$ in 7.4 (the cost of coding a machine into $\widetilde U$), $2^{k}$ in 7.5 (the level at which $\beta$ escapes a test), the dropped $N$ layers in 7.3, the $O(1)$ in P3. ⚠️ The source prints Def. 7.1 without $c$ ("$0 < \alpha - g(q) < \beta - q$") and then uses a constant from Prop. 7.2 onward; the version with $c$ is Solovay's and the one the notes use.
* **Irrationality is used, not assumed.** The proofs of 7.3 and 7.5 need $l < \beta$ *strictly* for an interval $[l, r] \ni \beta$ enumerated by a test, and get it from "random ⇒ irrational" (Part 6) — or, in 7.3, from C1.

## Structure: the exercise propositions

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(P1–P3 — Solovay reducibility on $\mathbb{R}$)</span></p>

* **(P1)** $\le\_S$ is reflexive and transitive — a preorder on $\mathbb{R}$, so it induces a degree structure.
* **(P2)** The left-c.e. reals are closed downward: $\beta$ left-c.e. and $\alpha \le\_S \beta$ imply $\alpha$ left-c.e. So the Solovay degrees of the left-c.e. reals sit inside those of $\mathbb{R}$.
* **(P3)** $\alpha \le\_S \beta \implies K(\alpha \upharpoonright n) \le K(\beta \upharpoonright n) + O(1)$ (the notes write the $O(1)$ on the left with a minus sign — same statement).

</div>

* **P1 — compose the exchange rates.** Reflexive: $g = \mathrm{id}$, $c = 2$ gives $0 < \alpha - q < 2(\alpha - q)$. Transitive: $\alpha \le\_S \beta$ via $(f, c)$ and $\beta \le\_S \gamma$ via $(h, d)$; for $q < \gamma$, $h(q) < \beta$, so $f(h(q))\downarrow$ and

  $$\alpha - f(h(q)) < c\,(\beta - h(q)) < cd\,(\gamma - q):$$

  $f \circ h$ and $cd$ witness $\alpha \le\_S \gamma$. Constants multiply.
* **P2 — push the c.e. left cut through $g$.** $L(\beta)$ is c.e. (Prop. 5.9). Enumerate it as $q\_0, q\_1, \ldots$, compute every $g(q\_k)$ (it halts, since $q\_k < \beta$) and take running maxima $a\_k := \max\_{j \le k} g(q\_j)$: computable, nondecreasing, below $\alpha$, and cofinal because $\alpha - g(q\_k) < c(\beta - q\_k)$ while the enumeration comes arbitrarily close to $\beta$. So $(a\_k)$ is a left-c.e. approximation of $\alpha$.
* **P3 — precision transfer is description transfer.** Fix $d$ with $c\,2^{1-d} < 1$. From $\beta \upharpoonright (n+d)$ compute the rational $q := 0.(\beta \upharpoonright (n+d)) - 2^{-(n+d)}$: then $q < \beta$ and $\beta - q < 2^{1-(n+d)}$ (dyadic bracket of Part 5), so by (S)

  $$0 < \alpha - g(q) < c\,2^{1-(n+d)} < 2^{-n}.$$

  Thus $\alpha$ lies in an interval of length $< 2^{-n}$ with known rational left end, which meets at most two level-$n$ dyadic intervals: one extra bit fixes $\alpha \upharpoonright n$. Hence $K(\alpha \upharpoonright n) \le K(\beta \upharpoonright (n+d)) + O(1) \le K(\beta \upharpoonright n) + d + O(1)$, the last step by a prefix-free machine that runs $\widetilde U$ and then reads exactly $d$ literal bits. Reading more than $n$ bits of $\beta$ is legal because $d$ depends on $c$ only: *$n + O(1)$ bits of $\beta$ determine $n$ bits of $\alpha$*. A prefix $\beta \upharpoonright n$ *is* a rational approximation of precision $2^{-n}$ — that is how a statement about approximations (S) becomes one about descriptions.
* **Consequence (not in the notes).** The same computation with $\beta$ itself as oracle shows $\alpha \le\_S \beta \Rightarrow \alpha \le\_T \beta$: Solovay reducibility refines Turing reducibility, and by a lot — $\Omega \equiv\_T \emptyset^{\prime}$ has many Turing-equivalent left-c.e. companions that are not random, hence (Thm. 7.6) not Solovay-equivalent to it.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(C1–C3 — Solovay reducibility and computable reals)</span></p>

* **(C1)** The computable reals are closed downward in $\mathbb{R}$: $\beta$ computable and $\alpha \le\_S \beta$ imply $\alpha$ computable.
* **(C2)** The computable reals form the **least** Solovay degree on the left-c.e. reals: $\alpha$ computable and $\beta$ left-c.e. imply $\alpha \le\_S \beta$ (even with $c = 1$).
* **(C3)** There is a right-c.e. real $\beta$ with $\alpha \not\le\_S \beta$ for *every* left-c.e. $\alpha$, the computable ones included.

</div>

* **C1 — shift an effective approximation to the left.** Take $(b\_n)$ computable with $\lvert \beta - b\_n \rvert < 2^{-n}$ (Prop. 5.6) and set $b\_n^{\prime} := b\_n - 2^{-n}$; then $b\_n^{\prime} < \beta$ and $\beta - b\_n^{\prime} < 2^{1-n}$, so $0 < \alpha - g(b\_n^{\prime}) < c\,2^{1-n} \le 2^{k-n}$ for $k := \lceil 1 + \log\_2 c \rceil$. Reindex: $a\_n := g(b^{\prime}\_{n+k})$ is computable ($g$ halts on every input below $\beta$) and $\lvert \alpha - a\_n \rvert < 2^{-n}$. Reindexing, not merely "$n \ge k$", is what restores the exact error bound $2^{-n}$.
* **C2 — a left-c.e. $\beta$ lets you find a lower bound above $q$.** Let $\lvert \alpha - a\_n\rvert < 2^{-n}$ (computable $\alpha$) and $b\_s \nearrow \beta$. Given $q < \beta$, search for $s$ with $b\_s > q$, pick $n$ with $2^{1-n} < b\_s - q$, and set $g(q) := a\_n - 2^{-n}$. Then

  $$0 < \alpha - g(q) = (\alpha - a\_n) + 2^{-n} < 2^{1-n} < b\_s - q \le \beta - q .$$

  The only use of "left-c.e." is the search for $s$: it needs $L(\beta)$ to be c.e.
* **C3 — sketch (the notes leave it open).** *Reduction.* If some left-c.e. $\alpha \le\_S \beta$, then $\tfrac12 \le\_S \alpha \le\_S \beta$ by C2 and P1; so it suffices to build a right-c.e. $\beta$ with $\tfrac12 \not\le\_S \beta$. *Finite injury.* Let $\beta := \lim\_s r\_s$ for a computable nonincreasing $(r\_s)$ with $r\_0 := 1$. Requirement $R\_{e,k}$: the pair $(g\_e, 2^k)$ does not witness $\tfrac12 \le\_S \beta$. Each $R\_{e,k}$ holds a rational witness $q$ with $q < r\_s$, chosen above the witnesses of all higher-priority requirements, and waits for $g\_e(q)$. If $g\_e(q)\uparrow$ or $g\_e(q) \ge \tfrac12$, the witness alone violates (S) as soon as $q < \beta$. If $g\_e(q)\downarrow < \tfrac12$, act once:

  $$r_{s+1} := \min\bigl\lbrace\, r_s,\ q + 2^{-k}\bigl(\tfrac12 - g_e(q)\bigr) \bigr\rbrace ,$$

  which keeps $\beta > q$ possible but forces $\tfrac12 - g\_e(q) \ge 2^k(\beta - q)$; lower-priority witnesses $\ge r\_{s+1}$ are cancelled and re-chosen inside $(q, r\_{s+1})$. Lower-priority actions never push $r$ below their own witnesses, which lie above $q$, and higher-priority requirements act only finitely often; so every requirement ends with a permanent witness $q < \beta$ that violates (S). Hence no left-c.e. real lies below $\beta$. Right-c.e. reals are therefore *not* closed downward (contrast P2): Def. 7.1 only talks about lower bounds, and a right-c.e. real has no computable supply of them.

## Proposition 7.2: four faces of "at least as fast"

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.2 — Speed-of-convergence characterizations)</span></p>

For left-c.e. reals $\alpha$ and $\beta$ the following are equivalent.

* **(i)** $\alpha \le\_S \beta$.
* **(ii)** There is a constant $c$ such that for *every* pair of left-c.e. approximations $(a\_n) \nearrow \alpha$, $(b\_n) \nearrow \beta$ there is a computable $g : \mathbb{N} \to \mathbb{N}$ with $\alpha - a\_{g(n)} < c\,(\beta - b\_n)$ for all $n$.
* **(iii)** For every left-c.e. approximation $(b\_s) \nearrow \beta$ there are a constant $d$ and a left-c.e. approximation $(a\_s) \nearrow \alpha$ with $a\_s - a\_{s-1} \le d\,(b\_s - b\_{s-1})$ for all $s$.
* **(iv)** There are left-c.e. approximations $(a\_n) \nearrow \alpha$, $(b\_n) \nearrow \beta$ and a constant $c$ with $\alpha - a\_n < c\,(\beta - b\_n)$ for all $n$.

</div>

### The five implications

* **(i) ⇒ (ii) — wait until the approximation overtakes $f$.** With (S) witnessed by $f, c$ put $g(n) := \min\lbrace m : a\_m > f(b\_n)\rbrace$. The search halts because $f(b\_n) < \alpha$ and $a\_m \nearrow \alpha$; then $\alpha - a\_{g(n)} < \alpha - f(b\_n) < c(\beta - b\_n)$. The waiting condition must be the *computable* $a\_m > f(b\_n)$, not the target inequality $\alpha - a\_m < c(\beta - b\_n)$, which mentions two unknown reals.
* **(i) ⇒ (iii) — follow $f$, but cap each step.** Given $(b\_s)$ let $t\_s := f(b\_s)$ and $d := c$ (the notes take $d = 2c$; any $d \ge c$ works). Set $a\_0 := t\_0$ and

  $$a_s := \min\bigl\lbrace\, \max(a_{s-1}, t_s),\ a_{s-1} + d\,(b_s - b_{s-1}) \bigr\rbrace .$$

  The sequence is computable, nondecreasing, below $\alpha$, and its increments are capped by construction. Invariant $\alpha - a\_s < d(\beta - b\_s)$: if the max wins, $\alpha - a\_s \le \alpha - t\_s < c(\beta - b\_s)$; if the cap wins, $\alpha - a\_s = (\alpha - a\_{s-1}) - d(b\_s - b\_{s-1}) < d(\beta - b\_{s-1}) - d(b\_s - b\_{s-1})$. The invariant forces $a\_s \to \alpha$. ⚠️ The "$a\_0 < a\_1 < \cdots$" in (iii) cannot be literally strict: when $b\_s = b\_{s-1}$ the increment bound forces $a\_s = a\_{s-1}$; read "nondecreasing".
* **(ii) ⇒ (iv) — repair monotonicity with a running maximum.** $(a\_{g(n)})\_n$ need not be monotone, so take $a^{\prime}\_n := \max\_{m \le n} a\_{g(m)}$: computable, nondecreasing, below $\alpha$, and $\alpha - a^{\prime}\_n \le \alpha - a\_{g(n)} < c(\beta - b\_n)$ (which also gives $a^{\prime}\_n \to \alpha$).
* **(iii) ⇒ (iv) — tails telescope.** $\alpha - a\_n = \sum\_{s > n}(a\_s - a\_{s-1}) \le d \sum\_{s > n}(b\_s - b\_{s-1}) = d(\beta - b\_n)$; strict with $2d$. Compare *remaining* increments, not full sums — then no normalization $a\_0 = b\_0 = 0$ is needed.
* **(iv) ⇒ (i) — read the pair off at $q$.** Given $q < \beta$, enumerate the pairs $(a\_n, b\_n)$ until $b\_n \ge q$ and set $g(q) := a\_n$: $0 < \alpha - a\_n < c(\beta - b\_n) \le c(\beta - q)$. The $a\_n$ are rationals by the definition of a left-c.e. approximation, so $g$ is a legal $\mathbb{Q} \to \mathbb{Q}$ map.

### Reading 7.2

* **(iv) is the workhorse.** Both 7.4 and 7.5 produce one synchronized pair of approximations and invoke (iv). (iii) is the increment form — the Solovay-domination exercise of Part 5 is exactly (iii) for $\alpha \le\_S \Omega$. (ii) says the constant does not depend on which approximations are used: slowness is a property of the *real*, not of the approximation.
* **Why left-c.e. on both sides.** (i) ⇒ (ii)/(iii) feed $f$ only inputs $b\_s < \beta$, so $f$ halts; (iv) ⇒ (i) needs a computable enumeration of lower bounds of $\beta$ to find $b\_n \ge q$. Without approximations from below none of the four clauses even makes sense — which is what C3 exploits.

## Proposition 7.3: randomness travels upward

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.3 — Solovay reducibility transfers randomness upward)</span></p>

If $\alpha$ is Martin-Löf random and $\alpha \le\_S \beta$, then $\beta$ is Martin-Löf random. Equivalently: everything Solovay-below a nonrandom real is nonrandom.

</div>

**Setup.** Let (S) hold via $g$ and $c$; enlarge $c$ to a power of two $c = 2^N$ (enlarging $c$ preserves (S)). Suppose $\beta$ is not random and fix a Martin-Löf test $L = (L\_1, L\_2, \ldots)$ with $\beta \in \bigcap\_i L\_i$, each $L\_i$ a c.e. union of dyadic intervals $[l, r] = I\_\sigma$.

**Translation.** For every $[l, r]$ enumerated into $L\_i$ with $i \ge N+1$, try to compute $g(l)$; if $g(l)\downarrow$, enumerate the interval

$$[\,g(l),\ g(l) + 2^N (r - l)\,]$$

into $M\_i$ (re-encoded as a finite union of dyadic intervals).

**$M$ is a Martin-Löf test.** Each $M\_i$ is uniformly c.e. open and

$$\lambda(M_i) \le \sum_{[l,r] \in L_i} 2^N \lambda([l, r]) = 2^N \lambda(L_i) \le 2^{N-i},$$

so $(M\_{N+1}, M\_{N+2}, \ldots)$, reindexed from $1$, has layers of measure $\le 2^{-j}$.

**$M$ catches $\alpha$.** For $i \ge N+1$ pick $[l, r] \in L\_i$ with $\beta \in [l, r]$. As $\beta$ is irrational, $l < \beta$, so $g(l)\downarrow$ and by (S)

$$g(l) < \alpha < g(l) + 2^N(\beta - l) \le g(l) + 2^N(r - l),$$

i.e. $\alpha \in M\_i$. Hence $\alpha$ fails $M$, contradicting its randomness. $\square$

### Why the details are the way they are

* **Why $\beta$ may be taken irrational.** A rational $\beta$ is computable, so $\alpha \le\_S \beta$ would make $\alpha$ computable (C1), hence nonrandom — against the hypothesis. (Alternatively feed $g$ the point $l^{\prime} := l - (r - l) < \beta$ and double every layer's measure; then no irrationality is needed.)
* **Why $g(l)$ and nothing else.** $l$ is the only point of $[l, r]$ guaranteed to be a *lower* bound of $\beta$, and lower bounds are the only inputs on which (S) promises anything.
* **Why length $2^N(r - l)$.** (S) locates $\alpha$ in $(g(l), g(l) + 2^N(\beta - l))$; the gap $\beta - l$ is unknown, and $r - l$ is its only computable bound.
* **Why a power of two, and why start at $N+1$.** Scaling lengths by $2^N$ turns the bound $2^{-i}$ into $2^{-(i-N)}$ — an index shift. The first $N$ layers are too fat after scaling and are simply dropped; a Martin-Löf test may start at any index.
* **Which way the flow goes.** Intervals around $\beta$ are pushed *forward* to intervals around $\alpha$; there is no computable way back, so a test for $\alpha$ cannot be pulled back to $\beta$. Randomness rises along $\le\_S$, nonrandomness sinks.
* **Rational endpoints.** The *basic* intervals of a layer have dyadic endpoints — that is what makes $g(l)$ a legal input. The boundary of a whole layer need not be computable: $(0, \alpha)$ for a noncomputable left-c.e. $\alpha$ is c.e. open.
* **Name of the mechanism.** This is a *randomness-test translation*; it is the reason Downey–Hirschfeldt treat $\le\_S$ (and its variants, §10) as a standard tool for measuring *relative randomness*.

## Proposition 7.4: $\Omega$ is on top

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.4 — $\Omega$ is Solovay-above every left-c.e. real)</span></p>

Every left-c.e. real $\alpha$ satisfies $\alpha \le\_S \Omega$.

</div>

**Proof (weight transfer).** By Prop. 5.9 (iii), $\alpha = \sum\_{\sigma \in \operatorname{dom} \widetilde M} 2^{-l(\sigma)}$ for a prefix-free machine $\widetilde M$. By Thm. 2.20 there is a self-delimiting code $i$ with $\widetilde U(i\sigma) = \widetilde M(\sigma)$ for all $\sigma$; put $e := l(i)$ (the coding constant of $\widetilde M$, called $d\_e$ in Part 6). With $\widetilde U[s]$ the part of $\operatorname{dom} \widetilde U$ enumerated by stage $s$ (Def. 5.10), define the synchronized pair

$$a_s := \sum_{\sigma \,:\, i\sigma \in \widetilde U[s]} 2^{-l(\sigma)} \nearrow \alpha, \qquad w_s := \sum_{\tau \in \widetilde U[s]} 2^{-l(\tau)} \nearrow \Omega .$$

The $\alpha$-mass still missing at stage $s$ is $2^e$ times the mass of the not-yet-enumerated codes $i\sigma$, which are among the not-yet-enumerated $\widetilde U$-codes:

$$\alpha - a_s = 2^{e} \sum_{\sigma \,:\, i\sigma \in \operatorname{dom}\widetilde U \setminus \widetilde U[s]} 2^{-l(i\sigma)} \le 2^{e} \sum_{\tau \in \operatorname{dom}\widetilde U \setminus \widetilde U[s]} 2^{-l(\tau)} = 2^{e}(\Omega - w_s) < 2^{e+1}(\Omega - w_s),$$

the last step because $\Omega - w\_s > 0$ ($\operatorname{dom} \widetilde U$ is infinite). This is (iv) of Prop. 7.2 with constant $2^{e+1}$. $\square$

* **Part 5 revisited.** The Solovay-domination exercise proved the increment form (iii) of the same fact; here it is the tail form (iv). Either way the constant is the simulation cost $2^{e}$: every program of $\widetilde M$ reappears in $\Omega$ with its weight divided by $2^{e}$.
* **All $\Omega$'s are alike.** For two universal machines, each $\Omega$ is left-c.e., so 7.4 gives $\Omega\_{\widetilde U\_1} \le\_S \Omega\_{\widetilde U\_2}$ and back: the halting probabilities of all universal prefix-free machines are Solovay equivalent, and by 7.3 all are random (a second proof of Thm. 6.2 for the *other* universal machines, given one).
* ⚠️ **Source slips.** The script cites "Proposition 2.20" for the halting-probability representation (it is Prop. 5.9 (iii)) and ends with the non-strict $\alpha - a\_s \le 2^e(\Omega - w\_s)$ while claiming constant $2^e$ for the strict (S); the notes fix this with $2^{e+1}$.

## Proposition 7.5 and the Kučera–Slaman theorem

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.5 — Random left-c.e. reals are Solovay-complete)</span></p>

If $\alpha$ and $\beta$ are left-c.e. and $\beta$ is Martin-Löf random, then $\alpha \le\_S \beta$.

</div>

Throughout: $\alpha, \beta \in (0,1)$, left-c.e. approximations $a\_0 = 0 < a\_1 \le \cdots \nearrow \alpha$ and $b\_0 = 0 \le b\_1 \le \cdots \nearrow \beta$, and $\beta$ random, hence irrational, so $b\_n < \beta$ for all $n$. The notes split the proof into three exercises.

### Step 1 (Ex. 1) — steep spots are rare

For a nondecreasing piecewise linear $g : [0, \beta] \to [0, \alpha]$ with left derivative $g^{(l)}(x) := \lim\_{y \nearrow x} \frac{g(x) - g(y)}{x - y}$,

$$\lambda\lbrace x \in [0, \beta) : g^{(l)}(x) > 2^k \rbrace < 2^{-k} \qquad \text{for every } k .$$

*Proof.* On each linear piece $(x\_{i-1}, x\_i]$ with slope $s\_i$ the left derivative equals $s\_i$, so the set is $\bigcup\_{i : s\_i > 2^k} (x\_{i-1}, x\_i]$ up to finitely many points, and

$$2^k \lambda(E) \le \sum_{i : s_i > 2^k} s_i (x_i - x_{i-1}) = \sum_{i : s_i > 2^k} \bigl(g(x_i) - g(x_{i-1})\bigr) \le g(\beta) - g(0) \le \alpha < 1 .$$

Markov against the total rise: a function that climbs at most $\alpha$ can climb faster than $2^k$ only on a set of measure $< 2^{-k}\alpha$. The Kučera–Slaman test spends the increments of $\alpha$'s approximation as interval lengths in exactly this way.

### Step 2 (Ex. 2) — the test

Fix $i$ and build $U\_i$ in steps $j = 0, 1, \ldots$ with indices $n\_0 := 0 < n\_1 < \cdots$. Having $n\_j$, enumerate

$$I_{i,j} := \bigl(\, b_{n_j},\ b_{n_j} + 2^{-i}(a_{j+1} - a_j) \,\bigr)$$

into $U\_i$, then search for the least $n\_{j+1} > n\_j$ with $b\_{n\_{j+1}} > b\_{n\_j} + 2^{-i}(a\_{j+1} - a\_j)$ — "wait until $\beta$'s approximation has climbed out of the interval". If the search never ends, $U\_i$ is complete. The $U\_i$ are uniformly c.e. open and

$$\lambda(U_i) \le \sum_j 2^{-i}(a_{j+1} - a_j) \le 2^{-i} \alpha < 2^{-i},$$

so $(U\_i)\_{i \ge 1}$ is a Martin-Löf test. *Reading:* the $j$-th increment of $\alpha$'s approximation, scaled by $2^{-i}$, is laid down as an interval just above the current $\beta$-approximation; the next increment is laid down only once $\beta$'s approximation has passed the previous interval.

### Step 3 (Ex. 3) — the dichotomy

Fix $i$. Exactly one of the following happens.

1. **The search stops at some $j$.** Then $b\_n \le b\_{n\_j} + 2^{-i}(a\_{j+1} - a\_j)$ for all $n > n\_j$, so $b\_{n\_j} < \beta \le b\_{n\_j} + 2^{-i}(a\_{j+1} - a\_j)$; as $\beta$ is irrational it is not the right endpoint, so $\beta \in I\_{i,j} \subseteq U\_i$.
2. **The search never stops.** Then $a\_{j+1} - a\_j < 2^{i}\,(b\_{n\_{j+1}} - b\_{n\_j})$ for every $j$, and the tails telescope:

   $$\alpha - a_j = \sum_{m \ge j} (a_{m+1} - a_m) < 2^{i} \sum_{m \ge j} (b_{n_{m+1}} - b_{n_m}) = 2^{i}\,(\beta - b_{n_j}).$$

   The computable pair $(a\_j)\_j$, $(b\_{n\_j})\_j$ is (iv) of Prop. 7.2 with constant $2^{i}$: $\alpha \le\_S \beta$.

Since $\beta$ is random, case 1 fails for some $i$, so case 2 holds for that $i$. $\square$

* **Reading.** Either $\beta$'s approximation *stalls* under some scaled $\alpha$-increment — then the test pins $\beta$ down to an interval of that length — or it never stalls, and then it is at least as fast as $\alpha$'s, increment by increment. A random $\beta$ cannot be pinned down, so it never stalls. The constant $2^i$ is the level at which $\beta$ escapes the test: it exists but is not uniform in $\beta$.
* **What was used.** Left-c.e.-ness of $\alpha$ (the increments are the test's budget), left-c.e.-ness of $\beta$ (the search is computable), $\alpha < 1$ (the budget is below $1$), irrationality of $\beta$ (the open interval).
* ⚠️ **Source slips.** The notes' Step 3 argues with the *sheet's* test $L\_k$ (the ratio formulation below) although their Step 2 replaced it by $U\_k$; for $U\_k$ the correct inference is the dichotomy above. Conversely, the notes' verdict that the sheet's $L\_i$ is "incorrect as written" is too harsh — see next.

### The sheet's test works too (not in the notes)

The sheet defines

$$x \in L_i \iff \exists\, n < m \ \Bigl( b_m < x \ \text{ and } \ \frac{a_m - a_n}{x - b_n} > 2^{i} \Bigr), \qquad\text{i.e.}\qquad L_i = \bigcup_{n < m} \bigl(\, b_m,\ b_n + 2^{-i}(a_m - a_n) \,\bigr),$$

uniformly c.e. open. **Claim:** $\lambda(L\_i) \le 2^{-i}\alpha$. Let $g$ be the piecewise linear interpolation of the points $(b\_n, a\_n)$, extended by the constant $\alpha$ to the right of $\beta$, and $h(x) := g(x) - 2^{i}x$. If $x \in L\_i$ with witnesses $n < m$, then $g(x) \ge a\_m$ (as $x > b\_m$), so $h(x) > h(b\_n)$ with $b\_n < x$: $x$ lies in $E := \lbrace x : h(x) > \min\_{[0,x]} h \rbrace$. $E$ is open, and on each of its components $(c, d)$ the endpoints are running minima of $h$ while no interior point is, which forces $h(c) = h(d)$, i.e. $g(d) - g(c) = 2^{i}(d - c)$ — the *rising sun lemma*. Summing over the disjoint components, $2^{i}\lambda(E) \le \sum\_k (g(d\_k) - g(c\_k)) \le \alpha$. This is the chord version of Step 1. With $(L\_k)$ a Martin-Löf test, the notes' Step 3 is correct as written: $\beta \notin L\_k$ means $a\_m - a\_n \le 2^k(\beta - b\_n)$ for all $n < m$, and $m \to \infty$ gives $\alpha - a\_n \le 2^k(\beta - b\_n)$ — (iv) with constant $2^{k+1}$, for the *original* approximations. Both routes prove 7.5; the sequential $U\_i$ is the textbook one.

## Theorem 7.6: Kučera–Slaman

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.6 — Kučera–Slaman theorem, 2001)</span></p>

On the set of left-c.e. reals, the Martin-Löf random reals form the **greatest Solovay degree**: a left-c.e. $\gamma$ is random iff $\alpha \le\_S \gamma$ for every left-c.e. $\alpha$.

</div>

**Proof.** A random left-c.e. real exists: $\Omega$ is left-c.e. (Def. 5.10, Prop. 5.9) and random (Thm. 6.2). By 7.5 every random left-c.e. real is $\le\_S$-above all left-c.e. reals, so the greatest degree exists and contains all of them. Conversely, if $\gamma$ lies in the greatest degree then $\Omega \le\_S \gamma$, and 7.3 makes $\gamma$ random. $\square$

* **Randomness as an order-theoretic property.** On the left-c.e. reals, "random" means "as hard to approximate from below as a left-c.e. real can be". In particular all random left-c.e. reals are Solovay equivalent to $\Omega$: there is exactly one $\Omega$-like degree, and it is the top. Three equivalent conditions for left-c.e. $\gamma$: random $\iff$ $\Omega \le\_S \gamma$ $\iff$ $\gamma$ is $\le\_S$-above every left-c.e. real.
* **The picture.** Bottom: the computable reals (C2 — the least degree). Top: the random reals (7.6). In between: noncomputable nonrandom left-c.e. reals, e.g. $0.A$ for a noncomputable c.e. set $A$ — left-c.e. and noncomputable, but $K(A \upharpoonright n) = O(\log n)$ because $n$ and the number of elements of $A$ below $n$ determine $A \upharpoonright n$. Randomness rises along $\le\_S$ (7.3), nonrandomness sinks; Turing reducibility sees none of this — $0.A$ with $A \equiv\_T \emptyset^{\prime}$ is Turing-equivalent to $\Omega$ and sits strictly below it in the Solovay order.
* **Two faces, third reading.** Part 5: $\Omega$ computes the halting problem. Part 6: $\Omega$ is incompressible. Part 7: $\Omega$ is the slowest left-c.e. real, and slowest *is* random — the three faces are one object seen through $\le\_T$, $K$ and $\le\_S$.
* **Forward.** 7.6 says nothing about random reals that are not left-c.e. (e.g. $1 - \Omega$, which is right-c.e. and random); §10 extends $\le\_S$ to all c.a. reals by the two-sided reducibility $\le^{2a}\_S$, which agrees with $\le\_S$ on the left-c.e. reals (Thm. 10.2).

## D.c.e. reals: three exercises

A real $\gamma$ is **d.c.e.** ("difference of c.e.") if $\gamma = \alpha - \beta$ for left-c.e. $\alpha, \beta$; it is **right-c.e.** if $-\gamma$ is left-c.e. The notes state three exercises and leave their solutions blank; sketches follow.

* **(1) d.c.e. $\iff$ bounded-variation approximation.** $\gamma$ is d.c.e. iff some computable $(q\_n) \to \gamma$ has $\sum\_n \lvert q\_{n+1} - q\_n \rvert < \infty$. (⇒) $q\_n := a\_n - b\_n$ has variation $\le (\alpha - a\_0) + (\beta - b\_0)$. (⇐) Split the increments by sign: $a\_n := q\_0 + \sum\_{m < n}(q\_{m+1} - q\_m)^{+}$ and $b\_n := \sum\_{m < n}(q\_{m+1} - q\_m)^{-}$ are computable, nondecreasing and bounded by the total variation, and $a\_n - b\_n = q\_n$; so $\gamma = \lim a\_n - \lim b\_n$ is d.c.e.
* **(2) random d.c.e. reals are one-sided.** A d.c.e. $\gamma$ that is neither left- nor right-c.e. is not Martin-Löf random. *Sketch (forward reference to Def. 9.3 and Thm. 9.4):* the intervals $I\_n$ with endpoints $q\_n, q\_{n+1}$ form a Solovay test, since their total length is the finite variation. If $\gamma$ were random it would lie in only finitely many $I\_n$, so from some $n\_0$ on consecutive terms never straddle $\gamma$: all $q\_n$ with $n \ge n\_0$ lie on one side of $\gamma$, and their running maximum (resp. minimum) is a left-c.e. (resp. right-c.e.) approximation of $\gamma$. Contrapositive: two-sided wobbling forever is a certificate of nonrandomness.
* **(3) d.c.e. reals are closed downward** relative to the two-sided reducibility $\le^{2a}\_S$ of §10 (Def. 10.1) within the c.a. reals — the d.c.e. analogue of P2; stated here, proved in the §10 material.

## One-glance summary

| Result | Statement | Tool |
| :-- | :-- | :-- |
| 7.1 | $\alpha \le\_S \beta$ $:\iff$ $\exists\, c, g$ partial computable: $0 < \alpha - g(q) < c(\beta - q)$ for all rational $q < \beta$ | lower bounds of $\beta$ ↦ lower bounds of $\alpha$, gap ratio $\le c$ |
| P1 | preorder; Solovay degrees | $g = \mathrm{id}$, $c = 2$; $f \circ h$, $cd$ |
| P2 | left-c.e. reals closed downward | enumerate $L(\beta)$, apply $g$, running max |
| P3 | $\le\_S \Rightarrow K(\alpha \upharpoonright n) \le K(\beta \upharpoonright n) + O(1)$ | $q := 0.(\beta\upharpoonright(n+d)) - 2^{-(n+d)}$; $c\,2^{1-d} < 1$; one extra bit |
| C1 | computable reals closed downward | shift $b\_n - 2^{-n}$, reindex by $\lceil 1 + \log\_2 c\rceil$ |
| C2 | computable $\le\_S$ every left-c.e. ($c = 1$): least degree | find $b\_s > q$, set $g(q) := a\_n - 2^{-n}$ with $2^{1-n} < b\_s - q$ |
| C3 | a right-c.e. $\beta$ with no left-c.e. real below it | reduce to $\tfrac12 \not\le\_S \beta$ (C2 + P1); finite injury |
| 7.2 | (i) cut map $\iff$ (ii) any approximations, computable delay $\iff$ (iii) increments $\le d \cdot$ increments $\iff$ (iv) one synchronized pair | $\min\lbrace m : a\_m > f(b\_n)\rbrace$; capped follower; running max; telescoping tails; read off at $q$ |
| 7.3 | $\alpha$ random, $\alpha \le\_S \beta$ ⇒ $\beta$ random | $[l, r] \mapsto [g(l), g(l) + 2^N(r-l)]$; drop $N$ layers; $l < \beta$ by irrationality |
| 7.4 | left-c.e. $\alpha \le\_S \Omega$, constant $2^{e+1}$ | $\alpha - a\_s = 2^e \cdot$ (missing $i\sigma$-mass) $\le 2^e(\Omega - w\_s)$; (iv) |
| 7.5, Step 1 | $\lambda\lbrace g^{(l)} > 2^k\rbrace < 2^{-k}$ for monotone piecewise linear $g$ | Markov against the total rise $\le \alpha$ |
| 7.5, Step 2 | $U\_i$: lay down $(b\_{n\_j}, b\_{n\_j} + 2^{-i}\Delta a\_j)$, advance $n\_j$ once $b$ climbs out | $\lambda(U\_i) \le 2^{-i}\alpha$ |
| 7.5, Step 3 | search stops ⇒ $\beta \in U\_i$; never stops ⇒ $\Delta a\_j < 2^i \Delta b\_{n\_j}$ ⇒ (iv) | random $\beta$ escapes some $U\_i$ |
| sheet's $L\_i$ | $\bigcup\_{n<m}(b\_m, b\_n + 2^{-i}(a\_m - a\_n))$ is also a test, $\lambda \le 2^{-i}\alpha$ | rising sun lemma for $g - 2^i x$ |
| 7.6 | random left-c.e. reals $=$ greatest Solovay degree $=$ degree of $\Omega$ | 6.2 + 7.5 (⇐), 7.3 (⇒) |
| d.c.e. | $\alpha - \beta$ $\iff$ finite-variation approximation; random d.c.e. ⇒ left- or right-c.e. | sign-split increments; Solovay test of the steps |

**Punchline.** Solovay reducibility is a computable exchange rate between lower bounds: $\alpha \le\_S \beta$ when every rational $q < \beta$ can be converted into a rational $g(q) < \alpha$ whose remaining gap is at most $c$ times the remaining gap below $\beta$. On the left-c.e. reals this is the same as "some approximation of $\alpha$ converges at least as fast as one of $\beta$", and it is monotone for randomness: a test that traps $\beta$ is translated, interval by interval through $g$, into a test that traps $\alpha$, so randomness can only travel upward. Chaitin's $\Omega$ is on top because every halting probability is a scaled part of it; every random left-c.e. real is on top because a $\beta$-approximation that ever stalls under a scaled increment of another left-c.e. approximation is caught by a Martin-Löf test. Together (Kučera–Slaman): **a left-c.e. real is Martin-Löf random iff it is Solovay-complete, iff it is Solovay-equivalent to $\Omega$** — on this class, randomness is not a property of bits or of tests but a position in an order, and there is exactly one top.
