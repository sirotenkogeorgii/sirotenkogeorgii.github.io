---
title: "Cheatsheet — Part 10: Relative Randomness of Computably Approximable Reals"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - computable-analysis
  - computability
  - solovay-reducibility
  - kolmogorov-complexity
  - left-ce-reals
  - dce-reals
  - chaitin-constant
  - cheatsheet
---

# Cheatsheet — Part 10: Relative Randomness of Computably Approximable Reals

*Exam recap: definitions, formulas and key results only, covering Relative Randomness of Computably Approximable Reals (10.1–10.10), up to (but excluding) Martingales. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}#relative-randomness-of-computably-approximable-reals). Prerequisites: [Part 5 — computable approximations]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part5-computable-approximations-of-reals/' | relative_url }}), [Part 7 — Solovay reducibility and d.c.e. reals]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part7-solovay-reducibility/' | relative_url }}), [Part 8 — relative complexity]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part8-relative-randomness-and-relative-complexity/' | relative_url }}) and [Part 9 — randomness tests]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part9-universal-randomness-tests/' | relative_url }}).*

## Where we are

**Conventions.** A **computable approximation** is a computable sequence of rationals converging to its real; a computable rate of convergence is **not** required. Its limit is **computably approximable (c.a.)**. Left-c.e. approximations are nondecreasing; a **d.c.e.** real is a difference of two left-c.e. reals, equivalently a limit of a computable approximation of finite total variation. For binary prefixes use reals in $[0,1)$, positions numbered from $0$, and the terminating expansion at dyadic rationals; $K$ is prefix-free complexity, and “random” means Martin-Löf random. Constants are independent of the stage or prefix length. Here “relative randomness” compares two reals; it does not mean randomness relative to an oracle.

**The one idea.** *A vanishing additive error lets two approximations be synchronized effectively.* Outside the left-c.e. reals, approximations can overshoot and oscillate, so replace signed gaps by absolute errors. Two approximations of the same middle real eventually agree to any **fixed** precision, and the allowance $2^{-n}$ pays for their discrepancy when reductions are composed. This preserves the Solovay order on left-c.e. reals and its implication to prefix complexity, but not the converse from a large complexity gap.

| part of the chapter | result | what to remember |
| :-- | :-- | :-- |
| definition and structure | 10.1–10.5 | two-sided errors plus additive slack; preorder; computable bottom; slack is essential |
| the cone below $\Omega$ | 10.6–10.8 | exactly the d.c.e. reals; their greatest degree consists of the random ones, which are one-sided |
| comparison with complexity | 10.9–10.10 | $\le^{2a}\_S \Rightarrow \le\_K$, but even $\ll\_K \not\Rightarrow \le^{2a}\_S$; use nonspeedability of $\Omega$ |

## Definition 10.1: two-sided approximation with slack

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(10.1 — $S2a$-reducibility)</span></p>

For c.a. reals $\alpha,\beta$, write $\alpha \le^{2a}\_S \beta$ if there are computable rational approximations $a\_n \to \alpha$, $b\_n \to \beta$ and a constant $c>0$ such that

$$\lvert \alpha-a_n\rvert \le c\bigl(\lvert\beta-b_n\rvert+2^{-n}\bigr) \qquad\text{for every }n. \tag{34}$$

</div>

* **Quantifiers matter.** The definition asks for **some pair** of approximations; it does not assert the inequality for arbitrary approximations or give a computable bound on either error.
* **Two regimes.** When $\lvert\beta-b\_n\rvert \gg 2^{-n}$, the multiplicative comparison dominates. When $\lvert\beta-b\_n\rvert \ll 2^{-n}$, the permitted error for $\alpha$ is still of order $2^{-n}$; even an exact hit $b\_n=\beta$ need not force $a\_n=\alpha$.
* **What is not free.** The additive term is a tolerance in the comparison, not a promise that every c.a. real has an approximation accurate to $2^{-n}$. Such a promise would make the real computable.

## Theorem 10.2: nothing changes on the left-c.e. reals

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(10.2 — Conservative extension)</span></p>

For **left-c.e.** $\alpha,\beta$:

$$\alpha\le^{2a}_S\beta \quad\Longleftrightarrow\quad \alpha\le_S\beta.$$

</div>

**Idea: thin to lower bounds, then spend genuine progress instead of slack.** The forward implication from $\le\_S$ to $\le^{2a}\_S$ uses the usual increasing witnesses and simply allows an extra positive term. Conversely, use independent left-c.e. approximations to certify lower bounds and thin the two-sided witnesses, first for $\alpha$ and then for $\beta$ along the same subsequence; an eventually-from-above case makes the corresponding left-c.e. real computable and is handled separately. Once both witnesses increase from below, choose a later stage $m$ with $b\_m-b\_k>2^{-m}$, so

$$\alpha-a_m\le c\bigl(\beta-b_m+2^{-m}\bigr)<c(\beta-b_k),$$

which is the ordinary Solovay tail inequality.

**Where the hypothesis enters.** The left-c.e. approximations certify that selected values lie below the limits; convergence alone cannot do that. The nested thinning preserves the pairing of the witnesses.

## Proposition 10.3: why the relation is transitive

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(10.3 — Preorder)</span></p>

$\le^{2a}\_S$ is reflexive and transitive on the c.a. reals. Mutual reducibility defines $S2a$-degrees.

</div>

**Idea: synchronize to the output precision.** Reflexivity uses the same approximation on both sides and $c=1$. For $\alpha\le^{2a}\_S\beta\le^{2a}\_S\gamma$, let $b\_m,b^{\prime}\_m\to\beta$ be the two middle witnesses and computably choose increasing stages $f(n)\ge n$ with

$$\lvert b_{f(n)}-b'_{f(n)}\rvert<2^{-n}.$$

The triangle inequality then composes the witnesses, since both original slack terms and the discrepancy are bounded by $2^{-n}$; for constants $c\_1,c\_2\ge1$ one may take $C=c\_1c\_2+2c\_1$:

$$\lvert\alpha-a_{f(n)}\rvert\le C\bigl(\lvert\gamma-d_{f(n)}\rvert+2^{-n}\bigr).$$

**Precision trap.** Keep $2^{-n}$ fixed while searching over $m$. Convergence of $b\_m-b^{\prime}\_m$ to zero does **not** ensure $\lvert b\_m-b^{\prime}\_m\rvert<2^{-m}$ at any stage: take $b\_m=0$ and $b^{\prime}\_m=1/(m+1)$. The moving threshold in the main notes' displayed search must therefore be replaced by the fixed output threshold above.

## Proposition 10.4: the bottom degree

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(10.4 — Computable reals form the least degree)</span></p>

Every computable real is $S2a$-below every c.a. real. Conversely, if $\alpha\le^{2a}\_S\beta$ and $\beta$ is computable, then $\alpha$ is computable.

</div>

**Idea: effective accuracy fits inside the slack.** For computable $\alpha$, take $\lvert\alpha-a\_n\rvert\le2^{-n}$, which works against any $\beta$. In the converse, the witnessing upper bound $c(\lvert\beta-b\_n\rvert+2^{-n})$ is a uniformly computable real sequence tending to zero, so search for a stage where it is smaller than the requested accuracy. This is a search for one certified accurate output, not an assumption that the original approximation has a computable modulus.

## Proposition 10.5: removing the slack breaks composition

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(10.5 — The additive-term-free variant is not transitive)</span></p>

Define $\alpha\le^2\_S\beta$ by requiring some computable approximations and a constant $c>0$ with

$$\lvert\alpha-a_n\rvert\le c\lvert\beta-b_n\rvert\qquad\text{for every }n.$$

There exist c.a. reals $\alpha,\beta,\gamma$ such that

$$\alpha\le^2_S\beta,\qquad\beta\le^2_S\gamma,\qquad\alpha\not\le^2_S\gamma.$$

</div>

**Idea: a rational middle real exposes the failure.** Choose a computable irrational $\alpha$ and diagonalize to obtain a d.c.e. $\gamma$ with $\alpha\not\le^2\_S\gamma$: each candidate pair of approximations and constant is defeated by forcing $\gamma$ sufficiently close to one candidate rational while $\alpha$ has positive distance from its candidate. A finite-injury construction with summably small movements meets all requirements. Put $\beta=0$: approximating $0$ by $2^{-n}$ accommodates a fast approximation of $\alpha$, whereas approximating $0$ exactly makes $0\le^2\_S\gamma$ automatic. This is the construction behind [Zheng–Rettinger, Theorem 2.5](https://www.researchgate.net/publication/221427240_On_the_Extensions_of_Solovay-Reducibility).

**Statement-level correction.** The source's three inequalities must omit $2^{-n}$, and the failed comparison is $\alpha$ against $\gamma$, not against $\beta$; otherwise the proposition contradicts 10.3. The failure can already occur among d.c.e. reals.

## Proposition 10.6: finite variation survives downward

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(10.6 — Downward closure of the d.c.e. reals)</span></p>

If $\alpha\le^{2a}\_S\beta$ and $\beta$ is d.c.e., then $\alpha$ is d.c.e.

</div>

**Idea: add the remaining variation to turn two-sided information into lower bounds.** Synchronize the reduction's $\beta$-witness with a finite-variation approximation, obtaining witnesses $A\_n,B\_n$ with $\sum\_n\lvert B\_{n+1}-B\_n\rvert=V<\infty$ and $\lvert\alpha-A\_n\rvert\le C(\lvert\beta-B\_n\rvert+2^{-n})$. If $V\_n$ is the accumulated variation through $B\_n$, then $\lvert\beta-B\_n\rvert\le V-V\_n$, hence

$$A_n+C V_n-C2^{-n}\le\alpha+CV,\qquad A_n+C V_n-C2^{-n}\longrightarrow\alpha+CV.$$

Taking running maxima makes $\alpha+CV$ left-c.e.; $CV$ is left-c.e. too (choose rational $C>0$), so $\alpha=(\alpha+CV)-CV$ is d.c.e.

**Recall the dictionary.** Finite variation $\Longleftrightarrow$ difference of two increasing bounded computable sequences: split the increments into their positive and negative parts. The proof uses the existence of the finite variation, not its computability.

## Theorem 10.7: exactly what lies below $\Omega$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(10.7 — Completeness for the d.c.e. reals)</span></p>

For every c.a. real $\alpha$:

$$\alpha\text{ is d.c.e.}\quad\Longleftrightarrow\quad\alpha\le^{2a}_S\Omega.$$

</div>

**Idea: dominate the two positive pieces separately.** Write $\alpha=u-v$ with $u,v$ left-c.e.; their Solovay reductions to $\Omega$ (7.4) give approximations along one increasing approximation $w\_n\to\Omega$ with $u-u\_n\le c\_u(\Omega-w\_n)$ and $v-v\_n\le c\_v(\Omega-w\_n)$. Subtract and use the triangle inequality:

$$\lvert\alpha-(u_n-v_n)\rvert\le(c_u+c_v)(\Omega-w_n).$$

The reverse implication is 10.6, since $\Omega$ is itself d.c.e.

**Scope.** $\Omega$ is greatest for the **d.c.e.** reals, not for all c.a. reals. The counterexample in 10.9 lies outside its lower cone and hence is not d.c.e.

## Theorem 10.8: identify the greatest degree

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(10.8 — Rettinger–Zheng, 2005)</span></p>

On the d.c.e. reals, the greatest **$S2a$-degree** consists exactly of the Martin-Löf random left-c.e. or right-c.e. reals. Equivalently, for d.c.e. $\beta$:

$$\beta\equiv^{2a}_S\Omega\quad\Longleftrightarrow\quad\beta\text{ is Martin-Löf random}.$$

</div>

**Idea: random finite-variation limits are eventually approached from one side.** If a finite-variation approximation crosses its limit infinitely often, the intervals between consecutive values give a Solovay test covering the limit infinitely often; thus a random d.c.e. real must be left- or right-c.e. The left-c.e. random reals are Solovay complete by Kučera–Slaman and 10.2, and reflection $x\mapsto1-x$ preserves both randomness and absolute approximation errors, putting the right-c.e. random reals in the same $S2a$-degree. Conversely, $\Omega\le^{2a}\_S\beta$ implies $\Omega\le\_K\beta$ by 10.9(1), so $K(\beta\upharpoonright n)\ge n-O(1)$.

**Notation trap.** “Greatest Solovay degree” in this chapter means the extended order $\le^{2a}\_S$. Ordinary $\le\_S$ retains a direction and cannot identify random left-c.e. and right-c.e. reals in this way.

## Proposition 10.9: approximation still controls complexity

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(10.9 — One implication survives, the other fails)</span></p>

For all c.a. $\alpha,\beta$:

$$\alpha\le^{2a}_S\beta\quad\Longrightarrow\quad\alpha\le_K\beta.$$

However, there is a c.a. $\alpha$ with

$$\alpha\ll_K\Omega\qquad\text{but}\qquad\alpha\not\le^{2a}_S\Omega.$$

Recall: $\alpha\le\_K\beta$ means $K(\alpha\upharpoonright n)\le K(\beta\upharpoonright n)+O(1)$, whereas $\alpha\ll\_K\beta$ means the difference $K(\beta\upharpoonright n)-K(\alpha\upharpoonright n)$ tends to $+\infty$.

</div>

**(1) Idea: a prefix locates a good stage, then bounded advice locates the output prefix.** From $\beta\upharpoonright n$, get the left endpoint $q$ of its dyadic cell and search for $s\ge n$ with $\lvert b\_s-q\rvert<2\cdot2^{-n}$. Then $\lvert\beta-b\_s\rvert<3\cdot2^{-n}$ and $2^{-s}\le2^{-n}$, so $\alpha$ lies within $4c\,2^{-n}$ of $a\_s$, an interval meeting only a fixed number of length-$2^{-n}$ cells. A fixed number of advice bits selects $\alpha\upharpoonright n$, giving the required $O(1)$ description overhead.

**Consequence.** Randomness travels upward along $\le^{2a}\_S$: a reduction from a random $\alpha$ forces the same lower bound $n-O(1)$ for the prefix complexity of $\beta$. The converse in 10.9 is much stronger than merely $\le\_K\not\Rightarrow\le^{2a}\_S$: even an unbounded complexity advantage does not guarantee a reduction.

## Lemma 10.10 and the counterexample in 10.9(2)

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(10.10 — Nonspeedability of $\Omega$)</span></p>

For any two computable nondecreasing rational approximations $w\_n,w^{\prime}\_n\to\Omega$ from below,

$$\frac{\Omega-w_n}{\Omega-w'_n}\longrightarrow1.$$

</div>

**What to use.** This includes computable accelerations: for any increasing total computable $f$, the ratio $(\Omega-w\_{f(n)})/(\Omega-w\_n)$ also tends to $1$. Thus an alleged reduction that yields a computable speedup with ratio $\le1/2$ infinitely often is impossible; the lemma does not give a computable rate for its own convergence.

**The construction: record changes sparsely, but at a coarser scale.** Start with a strictly increasing dyadic approximation $w\_s\to\Omega$ and $a\_0=0$; if the first changed bit from $w\_{s-1}$ to $w\_s$ has position $i\_s$, put $k=\lfloor i\_s/4\rfloor$ and **toggle** bit $2k$:

$$a_s(2k)=1-a_{s-1}(2k),\qquad a_s(p)=a_{s-1}(p)\quad(p\ne2k).$$

Each bit changes only finitely often, so $a\_s\to\alpha$; every odd bit remains zero.

**Cheap prefixes.** Encode the even bits of $\alpha\upharpoonright n$ literally, together with a self-delimiting description of $n$. Then

$$K(\alpha\upharpoonright n)\le\lceil n/2\rceil+K(\mathrm{bin}(n))+O(1),\qquad K(\Omega\upharpoonright n)\ge n-O(1),$$

so their complexity gap tends to infinity. The zero odd bits give a **limsup bound** $\limsup_n K(\alpha\upharpoonright n)/n\le1/2$; they do not by themselves prove the equality asserted in the main notes, and no equality is needed.

**No reduction: the two scales conflict.** A hypothetical $S2a$-reduction would synchronize with the constructed approximation and constrain its error to the remaining $\Omega$-error plus negligible slack. At the relevant blocks, an unresolved toggle of $\alpha$ has size of order $2^{-2k}$, while the remaining movement of $\Omega$ is of order $2^{-4k}$, so sampled approximations must already agree with the final recorded bit. The block/parity argument then forces, infinitely often, either an already resolved block or an even batch of hidden changes, in either case producing a computable acceleration with error ratio at most $1/2$, contrary to 10.10.

**Why a toggle matters.** Setting the bit permanently to $1$ would make $\alpha$ left-c.e.; then $\alpha\ll\_K\Omega$ would force $\alpha\le\_S\Omega$ by 8.3 and hence $\alpha\le^{2a}\_S\Omega$ by 10.2. The parity information and the two-sided changes are essential.

## One-glance summary

| Result | Statement | Key move |
| :-- | :-- | :-- |
| 10.1 | $\lvert\alpha-a\_n\rvert\le c(\lvert\beta-b\_n\rvert+2^{-n})$ for some computable approximations | absolute errors; additive tolerance |
| 10.2 | $\le^{2a}\_S\iff\le\_S$ on left-c.e. reals | nested thinning; absorb slack into positive progress |
| 10.3 | reflexive and transitive | make middle witnesses agree to fixed precision $2^{-n}$ |
| 10.4 | least degree $=$ computable reals | fast approximation fits in slack; computable upper error bound gives a search |
| 10.5 | $\le^2\_S$ is not transitive | diagonalize against an irrational computable target; insert a rational middle real |
| 10.6 | d.c.e. reals are downward closed | add accumulated variation to manufacture lower bounds |
| 10.7 | $\alpha\le^{2a}\_S\Omega\iff\alpha$ d.c.e. | dominate both left-c.e. pieces; use 10.6 |
| 10.8 | greatest d.c.e. degree $=$ random left- or right-c.e. reals | crossings form a Solovay test; reflection; Kučera–Slaman |
| 10.9(1) | $\le^{2a}\_S\Rightarrow\le\_K$ | input prefix locates a stage; bounded output-cell advice |
| 10.9(2) | some c.a. $\alpha\ll\_K\Omega$ but $\alpha\not\le^{2a}\_S\Omega$ | sparse toggle record: cheap prefixes, expensive stabilization |
| 10.10 | ratios of any two left-c.e. approximation errors for $\Omega$ tend to $1$ | forbids the speedup extracted from the hypothetical reduction |

**The structural picture.** On left-c.e. reals, $\ll\_K\Rightarrow\ll\_S\Rightarrow\le\_S\Rightarrow\le\_K$. On c.a. reals, $\le^{2a}\_S\Rightarrow\le\_K$ survives, but $\ll\_K\Rightarrow\le^{2a}\_S$ fails: short descriptions need not come with an approximation that stabilizes quickly enough.

**The boundary is finite variation.** The cone below $\Omega$ expands from left-c.e. reals under $\le\_S$ to exactly d.c.e. reals under $\le^{2a}\_S$. Within that cone, randomness still identifies the greatest degree; outside it, even very low prefix complexity can coexist with approximation behavior that $\Omega$ cannot dominate.
