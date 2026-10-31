---
title: "Cheatsheet — Part 8: Relative Randomness and Relative Complexity"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - computable-analysis
  - computability
  - solovay-reducibility
  - kolmogorov-reducibility
  - kolmogorov-complexity
  - left-ce-reals
  - chaitin-constant
  - k-triviality
  - cheatsheet
---

# Cheatsheet — Part 8: Relative Randomness and Relative Complexity

*Exam recap: definitions, formulas and key results only, covering the chapter Relative Randomness and Relative Complexity (Definitions 8.1–8.2 and Theorem 8.3) up to (but excluding) Universal Randomness Tests. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}#relative-randomness-and-relative-complexity). Prerequisites in [Part 1]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part1-plain-complexity/' | relative_url }}), [Part 2]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part2-prefix-free-complexity/' | relative_url }}), [Part 3]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part3-information-content-measures/' | relative_url }}), [Part 4]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part4-incompressibility-typicality/' | relative_url }}), [Part 5]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part5-computable-approximations-of-reals/' | relative_url }}), [Part 6]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part6-randomness-of-reals/' | relative_url }}) and [Part 7]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part7-solovay-reducibility/' | relative_url }}).*

## Where we are

Part 7 compared reals by **approximation speed from below**: $\alpha \le\_S \beta$ when lower bounds of $\beta$ convert computably into lower bounds of $\alpha$ with gap ratio $\le c$, and on the left-c.e. reals the random reals are exactly the top Solovay degree (Kučera–Slaman, Thm. 7.6). Kolmogorov complexity suggests a second comparison, **prefix by prefix**: $\alpha$ is no more random than $\beta$ if its initial segments are no harder to describe. This part turns the second comparison into a reducibility ($\le\_K$, with a strong form $\ll\_K$), adds a strong form $\ll\_S$ of Solovay reducibility, and shows how tightly the two scales are aligned:

$$\alpha \ll_S \beta \;\Longrightarrow\; \alpha \le_S \beta \;\Longrightarrow\; \alpha \le_K \beta \quad (\text{all reals}), \qquad\qquad \alpha \ll_K \beta \;\Longrightarrow\; \alpha \ll_S \beta \quad (\text{left-c.e. reals}).$$

| step | result | what it contributes |
| :-- | :-- | :-- |
| 8.1 | $\le\_K$, $\ll\_K$ | description-length comparison of prefixes: the difference $K(\beta \upharpoonright n) - K(\alpha \upharpoonright n)$ is bounded below, resp. tends to $\infty$ |
| 8.2 | $\ll\_S$ | approximation-speed comparison with **vanishing** exchange rate: gap ratio $\to 0$ |
| 8.3 | $\ll\_S \Rightarrow \le\_S \Rightarrow \le\_K$; on left-c.e. reals $\ll\_K \Rightarrow \ll\_S$ | speed from below controls complexity everywhere; on left-c.e. reals a large complexity gap forces an infinitely faster approximation |
| forward (§10) | Prop. 10.9 | $\le\_S \Rightarrow \le\_K$ survives on c.a. reals (as $\le^{2a}\_S \Rightarrow \le\_K$); $\ll\_K \Rightarrow \ll\_S$ does **not** |

**The one idea.** *Precision from below and description length are the same currency on a log scale.* A rational lower bound with gap $< 2^{-n}$ and a prefix of length $n$ carry the same information about a real, up to one bit; a constant factor $c = 2^N$ in gaps is $N$ bits in prefixes. So a Solovay reduction, which converts gaps at a bounded rate, converts prefixes at a bounded cost ($\le\_S \Rightarrow \le\_K$), and a vanishing gap ratio ($\ll\_S$) is the multiplicative twin of an unbounded complexity difference ($\ll\_K$). In the other direction, on left-c.e. reals there is only one candidate reduction — read $\alpha$'s approximant off at the stage where $\beta$'s approximation passes $q$ — and its quality is a race between the two approximations, which Kolmogorov complexity can referee.

## Definition 8.1: comparing descriptions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(8.1 — Kolmogorov reducibility, strong Kolmogorov reducibility)</span></p>

A real $\alpha$ is **Kolmogorov reducible** to a real $\beta$, written $\alpha \le\_K \beta$, if

$$K(\alpha \upharpoonright n) \le K(\beta \upharpoonright n) + O(1).$$

A real $\alpha$ is **strongly Kolmogorov reducible** to $\beta$, written $\alpha \ll\_K \beta$, if

$$K(\beta \upharpoonright n) - K(\alpha \upharpoonright n) \xrightarrow[n \to \infty]{} \infty.$$

</div>

* **Additive scale.** $\le\_K$ asks that $K(\beta \upharpoonright n) - K(\alpha \upharpoonright n)$ be bounded *below*; $\ll\_K$ asks that it tend to $+\infty$. Both compare *descriptions of prefixes* only — nothing is computed from $\beta$, no oracle is involved.
* **Structure.** $\le\_K$ is reflexive and transitive (constants add), so it induces **K-degrees**. $\ll\_K$ is irreflexive and implies $\le\_K$ (a difference tending to $\infty$ is eventually $\ge 0$); $\alpha \ll\_K \beta$ excludes $\beta \le\_K \alpha$.
* **Bottom and top.** $K(\mathrm{bin}(n)) - O(1) \le K(\alpha \upharpoonright n)$ for every $\alpha$ (the length is computable from the prefix), so "$\alpha \le\_K$ some computable real" says exactly that $\alpha$ is **K-trivial** (Part 5): the K-trivial reals form the least K-degree. Random reals have $K(\beta \upharpoonright n) \ge n - O(1)$ (Thm. 4.1, Def. 6.1), so computable $\alpha \ll\_K$ random $\beta$ — $O(\log n)$ bits against $n$.

## Definition 8.2: an infinitely faster approximation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(8.2 — Strong Solovay reducibility)</span></p>

A real $\alpha$ is **strongly Solovay reducible** to a real $\beta$, written $\alpha \ll\_S \beta$, if $\alpha \le\_S \beta$ via a partial computable $g$ that moreover fulfills

$$\lim_{q \nearrow \beta} \frac{\alpha - g(q)}{\beta - q} = 0.$$

</div>

* **Vanishing exchange rate.** $\le\_S$ bounds the ratio of remaining gaps by a constant; $\ll\_S$ makes it tend to $0$ as $q \nearrow \beta$ — *$\alpha$ can be approximated from below infinitely faster than $\beta$*. The limit says nothing about $q$ far from $\beta$, so the definition keeps "$g$ is a Solovay reduction" ($0 < \alpha - g(q)\downarrow$ for every $q < \beta$) as the precondition and strengthens only the bound.
* **Computable reals are $\ll\_S$ every left-c.e. real** (not in the notes). Given $q < \beta$, wait for a stage $b\_s > q$ and answer an $\alpha$-approximant below $\alpha$ accurate to $(b\_s - q)^2$; the ratio is $\le \beta - q \to 0$. This is C2 of Part 7 with a squared error. In particular $\ll\_S$ is reflexive on computable reals while $\ll\_K$ is never reflexive, so $\ll\_S \not\Rightarrow \ll\_K$.
* **$\Omega \not\ll\_S \Omega$** (not in the notes; via the nonspeedability Lemma 10.10). A reduction $g$ with vanishing ratio would turn the standard approximation $(w\_n)$ into the left-c.e. approximation $w^{\prime}\_n := \max\_{m \le n} g(w\_m)$ with $(\Omega - w^{\prime}\_n)/(\Omega - w\_n) \to 0$, whereas the lemma forces $\to 1$. So $\ll\_S \Rightarrow \le\_S$ is strict.

## Theorem 8.3: the two scales aligned

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.3 — Solovay reducibility sits between the complexity reducibilities)</span></p>

For all reals $\alpha$ and $\beta$:

$$\alpha \ll_S \beta \;\Longrightarrow\; \alpha \le_S \beta \;\Longrightarrow\; \alpha \le_K \beta.$$

On the set of left-c.e. reals, moreover:

$$\alpha \ll_K \beta \;\Longrightarrow\; \alpha \ll_S \beta.$$

</div>

* **$\ll\_S \Rightarrow \le\_S$.** Built into Def. 8.2.
* **$\le\_S \Rightarrow \le\_K$ — idea (precision is description).** Take $c = 2^N$. The prefix $\beta \upharpoonright n$ is itself a rational lower bound of $\beta$ with gap $\le 2^{-n}$, so (S) places $\alpha$ in an interval of length $2^{-(n-N)}$ whose left end $g(\beta \upharpoonright n)$ is computable from $\beta \upharpoonright n$; that interval meets at most two dyadic grid cells of its own length, so **one advice bit** selects $\alpha \upharpoonright (n-N)$. A prefix-free machine reading the bit and then an optimal code of $\beta \upharpoonright n$ gives $K(\alpha \upharpoonright (n-N)) \le K(\beta \upharpoonright n) + O(1)$, and restoring the $N$ missing bits costs $O(1)$. (Dyadic $\beta$ is computable, hence so is $\alpha$ by C1 — trivial case.) This is P3 of Part 7.
* **$\ll\_K \Rightarrow \ll\_S$ on left-c.e. reals — idea ($\beta$'s approximation lags where $\alpha$'s has settled).** Reduce as in C2: $g(q) := a\_m$ for the first stage $m$ with $b\_m > q$. Let $s\_n$ be the stage at which $(a\_s)$ settles the first $n$ bits of $\alpha$. The word $b\_{s\_n} \upharpoonright n$ is computable from $\alpha \upharpoonright n$, hence **cheap**: $K(b\_{s\_n} \upharpoonright n) \le K(\alpha \upharpoonright n) + O(1) \ll K(\beta \upharpoonright n)$. A cheap word cannot lie within boundedly many steps $2^{-n}$ of $\beta$ — otherwise $\beta \upharpoonright n$ would be cheap too — so

  $$2^n\,(\beta - b_{s_n}) \xrightarrow[n \to \infty]{} \infty :$$

  when $\alpha$'s approximation is within $2^{-n}$, $\beta$'s is still unboundedly many $2^{-n}$-steps away. For a $q$ answered between stages $s\_n$ and $s\_{n+1}$ this gives $(\alpha - g(q))/(\beta - q) \le 2^{-n}/(\beta - b\_{s\_{n+1}}) \to 0$.
* **Where left-c.e. enters.** Both approximations *are* the reduction, and "first stage with $b\_m > q$" needs the lower bounds of $\beta$ to be enumerable. The complexity hypothesis is used exactly once — to certify that $\beta$'s approximation lags. §10 shows this is sharp: for c.a. reals $\ll\_K$ no longer forces any Solovay-type reduction (Prop. 10.9 (2)).
* **Reading.** The two nontrivial implications are one dictionary read in opposite directions. Downwards (speed ⇒ complexity): a Solovay reduction *is* a short description of $\alpha$'s prefixes from $\beta$'s. Upwards (complexity ⇒ speed): the only reduction available is the race of the approximations, and complexity decides the race — a word that an approximation outputs at a stage computable from a cheap word is cheap, so $\beta$'s expensive prefixes can only appear late.

## The map of implications

| implication | holds? | why / where |
| :-- | :-- | :-- |
| $\ll\_S \Rightarrow \le\_S$ | all reals | Def. 8.2 |
| $\le\_S \Rightarrow \le\_K$ | all reals | 8.3 — precision is description (P3 of Part 7) |
| $\ll\_K \Rightarrow \le\_K$ | all reals | the difference is eventually $\ge 0$ |
| $\ll\_K \Rightarrow \ll\_S$ | **left-c.e. reals only** | 8.3 — the lagging approximation; fails for c.a. reals: a c.a. $\alpha \ll\_K \Omega$ with $\alpha \not\le^{2a}\_S \Omega$ (Prop. 10.9 (2)) |
| $\le^{2a}\_S \Rightarrow \le\_K$ | c.a. reals | Prop. 10.9 (1): the advice-bit machine with two-sided error |
| $\ll\_S \Rightarrow \ll\_K$ | **fails** | computable $\alpha \ll\_S \alpha$, but $\ll\_K$ is irreflexive |
| $\le\_S \Rightarrow \ll\_S$ | **fails** | $\Omega \le\_S \Omega$ but $\Omega \not\ll\_S \Omega$ (Lemma 10.10) |
| $\le\_K \Rightarrow \le\_S$ | **fails** (not in the notes) | a noncomputable K-trivial left-c.e. real (Solovay) is $\le\_K$ every computable real but, by C1, $\le\_S$ none |

## Consequences on the left-c.e. reals (not stated in the notes)

* **Random $\iff$ K-top.** Every left-c.e. $\alpha$ is $\le\_S \Omega$ (7.4), hence $\le\_K \Omega$ (8.3), and likewise $\le\_K$ every random left-c.e. $\gamma$ (7.6 + 8.3). Conversely $\Omega \le\_K \gamma$ gives $K(\gamma \upharpoonright n) \ge n - O(1)$, i.e. $\gamma$ random (Def. 6.1). So for left-c.e. $\gamma$: random $\iff$ $\le\_S$-top (7.6) $\iff$ $\le\_K$-top — **the two orders agree at the top.**
* **The bottoms differ.** The least $\le\_S$-degree is the computable reals (C2); the least $\le\_K$-degree is the K-trivial reals, which contain the computable ones (Part 5) and, by Solovay, more. $\le\_K$ is the coarser order at the bottom, and this is exactly where $\le\_K \not\Rightarrow \le\_S$.
* **$\Omega$ in a fourth coordinate system.** Part 5: $\Omega$ computes $\emptyset^{\prime}$. Part 6: $\Omega$ is incompressible. Part 7: $\Omega$ is the slowest left-c.e. real. Part 8: the prefixes of every left-c.e. real are describable from those of $\Omega$ at constant extra cost, and $\Omega$'s own approximation cannot be sped up even by itself.

## One-glance summary

| Result | Statement | Tool |
| :-- | :-- | :-- |
| 8.1 | $\alpha \le\_K \beta :\iff K(\alpha \upharpoonright n) \le K(\beta \upharpoonright n) + O(1)$; $\alpha \ll\_K \beta :\iff K(\beta \upharpoonright n) - K(\alpha \upharpoonright n) \to \infty$ | additive scale; preorder, resp. irreflexive; least K-degree $=$ K-trivial reals |
| 8.2 | $\alpha \ll\_S \beta :\iff \alpha \le\_S \beta$ via $g$ with $(\alpha - g(q))/(\beta - q) \to 0$ as $q \nearrow \beta$ | multiplicative scale; vanishing exchange rate |
| 8.3 (a) | $\ll\_S \Rightarrow \le\_S$ | definition |
| 8.3 (b) | $\le\_S \Rightarrow \le\_K$, all reals | $\beta \upharpoonright n$ is a lower bound with gap $\le 2^{-n}$; $g(\beta \upharpoonright n)$ plus one advice bit $= \alpha \upharpoonright (n-N)$ |
| 8.3 (c) | $\ll\_K \Rightarrow \ll\_S$, left-c.e. reals | $g(q) := a\_m$ at the first $b\_m > q$; $b\_{s\_n} \upharpoonright n$ is cheap $\Rightarrow 2^n(\beta - b\_{s\_n}) \to \infty$ |
| non-implications | $\ll\_S \not\Rightarrow \ll\_K$; $\le\_S \not\Rightarrow \ll\_S$; $\le\_K \not\Rightarrow \le\_S$; $\ll\_K \not\Rightarrow \ll\_S$ on c.a. reals | computable reals; $\Omega$ against Lemma 10.10; noncomputable K-trivial + C1; Prop. 10.9 (2) |
| consequence | left-c.e. $\gamma$: random $\iff$ $\le\_S$-top $\iff$ $\le\_K$-top | 7.4, 7.6, 8.3, Def. 6.1 |

**Punchline.** Two ways to say "$\alpha$ is no more random than $\beta$": every lower bound of $\beta$ converts into a lower bound of $\alpha$ with a bounded loss of precision ($\le\_S$), or every prefix of $\alpha$ is describable at no more than a bounded extra cost over the same prefix of $\beta$ ($\le\_K$). Since $n$ bits of precision from below *are* $n$ bits of description up to one advice bit, the first implies the second on all reals, and the same dictionary turns a vanishing precision ratio ($\ll\_S$) into an unbounded description gap. On left-c.e. reals the dictionary also reads backwards: the race of the two approximations is the only reduction there is, and complexity referees it — the stage at which $\alpha$'s approximation settles $n$ bits is cheap to name, so whatever $\beta$'s approximation shows at that stage is a cheap word, and cheap words stay far from an expensive $\beta$. Hence $\ll\_K \Rightarrow \ll\_S \Rightarrow \le\_S \Rightarrow \le\_K$ on the left-c.e. reals, where speed from below and description length are two readings of one quantity; one level up, on the c.a. reals, the backward reading breaks (Prop. 10.9).
