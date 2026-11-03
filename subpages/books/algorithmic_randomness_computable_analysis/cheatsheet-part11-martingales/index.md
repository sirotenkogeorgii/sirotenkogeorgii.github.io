---
title: "Cheatsheet — Part 11: Martingales"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - computable-analysis
  - computability
  - martingales
  - supermartingales
  - martin-löf-test
  - kolmogorov-complexity
  - cantor-space
  - cheatsheet
---

# Cheatsheet — Part 11: Martingales

*Exam recap: definitions, formulas and key results only, covering Martingales (11.1–11.6), including the two prefix-free/maximal-inequality exercises. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}#martingales). Prerequisites: [Part 4 — incompressibility and typicality]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part4-incompressibility-typicality/' | relative_url }}), [Part 5 — computable and left-c.e. reals]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part5-computable-approximations-of-reals/' | relative_url }}) and [Part 9 — randomness tests]({{ '/subpages/books/algorithmic_randomness_computable_analysis/cheatsheet-part9-universal-randomness-tests/' | relative_url }}).*

## Where we are

**Conventions.** Work on fair-coin Cantor space $\lbrace0,1\rbrace^{\mathbb N}$, with cylinder $[[\sigma]]$, length $l(\sigma)$ and measure $\lambda([[\sigma]])=2^{-l(\sigma)}$. As in the notes, $\lambda$ also denotes the empty word, so $d(\lambda)$ is initial capital; $\tau\sqsubseteq\sigma$ means “$\tau$ is a prefix of $\sigma$.” All capital values are finite and nonnegative. “C.e.” for a real-valued function means uniformly **lower semicomputable**, and “random” means Martin-Löf random; $K$ denotes prefix-free complexity. We do not assume initial capital is computable or normalize it by dividing by an unknown real.

**The one idea.** *Small measure corresponds to a large possible gain in capital.* Start with capital $2^{-l(\sigma)}$ and bet everything on each successive bit of $\sigma$: every correct prediction doubles the capital, so it reaches $1$ exactly when the observed sequence begins with $\sigma$; a wrong prediction leaves $0$. Conversely, a strategy starting with $c$ can reach capital $k$ on a set of measure at most $c/k$. These two observations convert effective tests into successful betting strategies and successful strategies into effective tests, completing the equivalence between untypicality, compressibility and predictability.

| step | result | what it contributes |
| :-- | :-- | :-- |
| 11.1 | (super)martingale and success | formalize fair betting and unbounded wealth |
| 11.2 + exercises | prefix-free capital bound; maximal inequality | turn large capital into small measure |
| 11.3–11.4 | effective real-valued functions; summable portfolios | build one c.e. strategy from enumerated bets |
| 11.5 | savings trick | turn unbounded peaks into capital tending to infinity |
| 11.6 | Ville–Schnorr characterization | Martin-Löf random iff no c.e. (super)martingale succeeds |

## Definition 11.1: a fair game, with no borrowing

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(11.1 — Supermartingale, martingale, success)</span></p>

A **supermartingale** is a function $d:\lbrace0,1\rbrace^{\ast}\to[0,\infty)$ with

$$d(\sigma)\ge\frac{d(\sigma0)+d(\sigma1)}2. \tag{39}$$

It is a **martingale** if equality holds at every word:

$$d(\sigma)=\frac{d(\sigma0)+d(\sigma1)}2. \tag{40}$$

It **succeeds** on $X$ if

$$\limsup_{n\to\infty}d(X\upharpoonright n)=\infty.$$

</div>

**Reading.** A martingale preserves expected capital at each fair-coin bet; a supermartingale may discard some capital. Nonnegativity is essential: the gambler cannot finance unlimited bets by borrowing without bound. Success means arbitrarily high peaks, not necessarily eventual growth or a positive long-run average return.

**Two immediate bounds.** Iterating the inequality gives

$$\sum_{l(\sigma)=n}2^{-n}d(\sigma)\le d(\lambda),\qquad d(\sigma)\le2^{l(\sigma)}d(\lambda).$$

The first is an equality for martingales. If $d(\lambda)=0$, nonnegativity forces $d\equiv0$, so a successful strategy must start with positive capital.

## Theorem 11.2: how expensive is reaching a threshold?

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(11.2 — Kolmogorov's maximal inequality)</span></p>

For a supermartingale $d$ and $k>0$, let

$$R_k=\lbrace\sigma:d(\sigma)\ge k\rbrace,\qquad \bar R_k=\bigcup_{\sigma\in R_k}[[\sigma]].$$

Then

$$\lambda(\bar R_k)=\lambda\bigl(\lbrace X:\exists n\ d(X\upharpoonright n)\ge k\rbrace\bigr)\le\frac{d(\lambda)}k.$$

</div>

**Idea: charge each path only at its first crossing.** Take the prefix-minimal words $S\subseteq R\_k$; their cylinders are disjoint and cover $\bar R\_k$. The prefix-free capital bound below gives

$$k\lambda(\bar R_k)=k\sum_{\sigma\in S}2^{-l(\sigma)}\le\sum_{\sigma\in S}2^{-l(\sigma)}d(\sigma)\le d(\lambda).$$

The point is the word **ever**: a bound at one fixed level alone would not control the union over all levels.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Prefix-free capital bound)</span></p>

For every finite or infinite prefix-free family $S$ of extensions of $\sigma$,

$$\sum_{\tau\in S}2^{-l(\tau)}d(\tau)\le2^{-l(\sigma)}d(\sigma).$$

</div>

**Idea: prune the tree.** For finite $S$, repeatedly replace weighted child capitals by the upper bound at their parent, stopping at words of $S$ and discarding unused branches. For infinite $S$, apply the finite bound to finite subsets and take the increasing limit of the nonnegative sums.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Maximal inequality relative to initial capital)</span></p>

If $d(\lambda)>0$ and $k>0$, then

$$\lambda\bigl(\lbrace X:\exists n\ d(X\upharpoonright n)\ge k\,d(\lambda)\rbrace\bigr)\le\frac1k.$$

</div>

**Idea: substitute the threshold.** Apply 11.2 at $k\,d(\lambda)$. The positivity assumption matters: if $d\equiv0$, the weak threshold $d\ge k\,d(\lambda)=0$ is met everywhere.

**Sharpness.** Starting from $1$, bet everything on a fixed word of length $m$ and stop after it: capital reaches $2^m$ exactly on that cylinder, of measure $2^{-m}$. Equality in the first-crossing proof requires both exact threshold capital at the crossing words and no loss in the prefix-free capital bound; overshooting or discarding capital makes the corresponding step strict.

**Effectivity trap.** $\bar R\_k$ is always open, but need not be **c.e. open**. For lower semicomputable $d$, the relation $d(\sigma)>k$ is c.e.; equality at $k$ need never be certified, so $d(\sigma)\ge k$ need not be c.e. Use strict thresholds in effective tests, and keep $\ge$ for the noneffective maximal inequality. The source's original $d(\sigma)\le k$ has the wrong direction altogether.

## Definition 11.3: what can the gambler compute?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(11.3 — Effective real-valued functions)</span></p>

A function $f:\lbrace0,1\rbrace^{\ast}\to\mathbb R$ is **computable**, in the standard approximation sense, if a total computable rational-valued function $q(\sigma,n)$ satisfies

$$\lvert q(\sigma,n)-f(\sigma)\rvert\le2^{-n}.$$

It is **c.e. (lower semicomputable)** if its strict lower cuts are uniformly c.e.:

$$\lbrace(\sigma,r):r\in\mathbb Q,\ r<f(\sigma)\rbrace\quad\text{is c.e.}$$

Equivalently, there is a uniformly computable rational approximation $q\_s(\sigma)\nearrow f(\sigma)$.

</div>

**Two clocks.** The length $l(\sigma)$ counts revealed bits; the approximation stage $s$ counts computation spent discovering the capital at that *fixed* word. Increasing $s$ cannot decrease a lower approximation, but revealing the next bit may certainly decrease actual capital.

**Uniformity correction.** The notes define computability by uniformly decidable strict left cuts. This is stronger than uniform approximation to prescribed precision: a uniformly computable family can have undecidable positivity (take $x\_e=2^{-t}$ if machine $e$ first halts at time $t$, and $x\_e=0$ otherwise). The c.e. clause is correct as stated; the approximation definition above is the standard one used in the effectiveness claims here.

## Proposition 11.4: a finite budget for infinitely many strategies

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(11.4 — Summing a c.e. family)</span></p>

Let $I$ be c.e. and $d\_i$ a uniformly c.e. family of nonnegative (super)martingales. If

$$\sum_{i\in I}d_i(\lambda)<\infty,$$

then $d(\sigma)=\sum\_{i\in I}d\_i(\sigma)$ is a finite-valued c.e. (super)martingale.

</div>

**Idea: initial capital controls the whole tree.** Since $d\_i(\sigma)\le2^{l(\sigma)}d\_i(\lambda)$, summability at the root implies summability at every word. Sum the fairness inequalities term by term, and dovetail nonnegative lower approximations to the enumerated summands to obtain a lower approximation to the total.

**What the hypothesis buys.** The sum of starting capitals need only be finite, not computable. Therefore the conclusion is c.e.; even computable individual strategies can sum to a noncomputable one when there is no effective bound on the remaining tail.

## Theorem 11.5: bank the peaks

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(11.5 — Savings trick)</span></p>

For every nonnegative (super)martingale $d$ there is a supermartingale $\tilde d$ with

$$\lbrace X:\limsup_n d(X\upharpoonright n)=\infty\rbrace=\lbrace X:\lim_n\tilde d(X\upharpoonright n)=\infty\rbrace.$$

If $d$ is computable or c.e., respectively, $\tilde d$ can be chosen in the same effective class.

</div>

**Idea: remember the peaks, so later losses cannot erase all earlier gains.** Merely replacing $d$ by $\log(1+d)$ would compress the peaks but still crash back towards $0$ whenever $d$ does. Instead, remember the largest capital reached so far, $M$, and split the transformed capital into **protected savings**, depending only on $M$, and a **small active stake**, depending on the current $d$. Only the active part can be lost when $d$ falls.

**A smooth formula for the trick.** The following version avoids deciding equality at saving thresholds. Put

$$M(\sigma)=\max_{\tau\sqsubseteq\sigma}d(\tau),\qquad s(M)=\log(1+M)-\frac{M}{1+M},\qquad \tilde d(\sigma)=s(M(\sigma))+\frac{d(\sigma)}{1+M(\sigma)},$$

where $\log$ is the natural logarithm.

**Between records: the savings stay fixed.** While the original capital remains below its previous maximum $M$, the transformed strategy holds $s(M)$ untouched and follows $d$ at the reduced scale $1/(1+M)$. Since $0\le d\le M$, its active capital is always less than one unit:

$$0\le\frac{d}{1+M}<1,\qquad s(M)\le\tilde d<s(M)+1.$$

Even if $d$ loses almost everything, $\tilde d$ can only fall back towards the savings already accumulated. For example, after a record $M=8$, savings are $s(8)\approx1.308$; a crash from $d=8$ to $d=0.5$ reduces transformed capital from $\log9\approx2.197$ to about $1.364$, while those savings remain intact.

**At a new record: save more and reduce the scale of future bets.** If $d$ reaches a new maximum $x>M$, the savings increase from $s(M)$ to $s(x)$ and the active multiplier decreases from $1/(1+M)$ to $1/(1+x)$. The savings really do grow with the record:

$$s'(M)=\frac{M}{(1+M)^2}\ge0,\qquad s(M)=\log(1+M)-1+\frac1{1+M}\longrightarrow\infty.$$

Thus unbounded records produce an unbounded, nondecreasing floor under the new capital. The logarithm deliberately makes that floor grow slowly; the theorem asks for divergence, not a fast rate of gain.

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/algorithmic_randomness_computable_analysis/arca_smooth_savings.png' | relative_url }}" alt="Two aligned plots of a martingale capital path and its savings transform. Original capital reaches 8, 16, 32, and 64, repeatedly falling to 0.5. Protected savings never decrease, and transformed capital always lies between the savings and savings plus one." loading="lazy" width="1696" height="1248">
  <figcaption>The same finite path before and after the transformation; the vertical scales differ. Red is the original capital $d_n$, orange its running maximum $M_n$, green the protected savings $s(M_n)$, and blue the transformed capital. The shaded strip is $[s(M_n),s(M_n)+1]$: blue always stays inside it. At each crash, green stays fixed. Extending this pattern with unbounded peaks would make the green floor, and hence the blue capital, tend to infinity.</figcaption>
</figure>

**Where the formula comes from: use the tangent at the last record.** For $\phi(x)=\log(1+x)$, the tangent line at $M$ is

$$\phi(M)+\phi'(M)(d-M)=\underbrace{\log(1+M)-\frac{M}{1+M}}_{\text{protected savings}}+\underbrace{\frac{d}{1+M}}_{\text{active capital}}.$$

So the savings are exactly the tangent's intercept, and the active multiplier is its slope. Below the record we follow this line; at a new record $x$ we switch to the tangent at $x$, whose value there is $\phi(x)$. Concavity puts $\phi(x)$ below the old tangent, so switching never requires extra money: any surplus can be discarded, as a supermartingale allows.

**Why the strategy remains a supermartingale.** For either child with original capital $x=d(\sigma i)$, the new total is at most $s(M(\sigma))+x/(1+M(\sigma))$: equality holds if no new record is set, and concavity gives the bound otherwise. Averaging over the next bit therefore gives

$$\frac{\tilde d(\sigma0)+\tilde d(\sigma1)}2\le s(M(\sigma))+\frac{d(\sigma0)+d(\sigma1)}{2(1+M(\sigma))}\le\tilde d(\sigma).$$

**Why success and effectiveness are preserved.** Unbounded $d$ means $M\to\infty$, so the savings force $\tilde d\to\infty$; bounded $d$ keeps $M$, the savings and the total bounded. The full expression $F(d,M)=s(M)+d/(1+M)$ is computable and nondecreasing in both variables on $0\le d\le M$:

$$\frac{\partial F}{\partial d}=\frac1{1+M}>0,\qquad \frac{\partial F}{\partial M}=\frac{M-d}{(1+M)^2}\ge0.$$

Consequently, computable approximations remain computable, and increasing lower approximations to $d$ and its finite running maximum yield increasing lower approximations to the total (with the usual rational approximations to the logarithm). It is the **whole expression** that preserves lower semicomputability, not its quotient term considered separately.

**What actually increases.** Savings are monotone; total capital need not be. The conclusion is convergence to infinity, not monotone divergence. Also, arbitrary quotients of left-c.e. reals need not be left-c.e., so the source's betting-ratio formulas alone do not establish the c.e. preservation claim; the monotone formula above does.

## Theorem 11.6: unpredictability is Martin-Löf randomness

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(11.6 — Ville–Schnorr characterization)</span></p>

For an infinite binary sequence $A$, the following are equivalent:

* $A$ is Martin-Löf random.
* No c.e. martingale succeeds on $A$.
* No c.e. supermartingale succeeds on $A$.

</div>

**Strategy → test: capital thresholds have small measure.** If a c.e. supermartingale $d$ succeeds, fix an integer $k>d(\lambda)$ and form

$$U_n=\bigcup_{d(\sigma)>2^n k}[[\sigma]],\qquad n\ge1.$$

Strict comparison makes the $U\_n$ uniformly c.e. open, while 11.2 gives $\lambda(U\_n)\le d(\lambda)/(2^n k)\le2^{-n}$. Unbounded capital puts $A$ in every $U\_n$, so $A$ fails this Martin-Löf test. The integer $k$ is a fixed finite choice; one need not compute the exact initial capital from its lower approximation.

**The elementary bet: predict a fixed prefix.** For a word $\sigma$, define

$$g_\sigma(\tau)=2^{l(\tau)}\lambda([[\sigma]]\cap[[\tau]])=
\begin{cases}
2^{l(\tau)-l(\sigma)},&\tau\text{ is a proper prefix of }\sigma,\\
1,&\sigma\sqsubseteq\tau,\\
0,&\sigma\text{ and }\tau\text{ are incompatible}.
\end{cases}$$

This computable rational-valued martingale starts with $2^{-l(\sigma)}$, doubles on each correct bit, loses everything on a wrong bit, and freezes at $1$ after reaching $\sigma$. It is the conditional probability of the target cylinder given the observed prefix.

**Test → strategy: each failed layer contributes another permanent unit of capital.** Write a failed Martin-Löf test as $U\_i=\bigcup\_{\sigma\in L\_i}[[\sigma]]$, with uniformly c.e. prefix-free presentations $L\_i$, and set

$$d=\sum_{i\ge1}\sum_{\sigma\in L_i}g_\sigma,\qquad d(\lambda)=\sum_{i\ge1}\lambda(U_i)\le\sum_{i\ge1}2^{-i}=1.$$

Proposition 11.4 gives a c.e. martingale. For each $i$, some prefix of $A$ belongs to $L\_i$, and that component contributes $1$ at every later prefix; after the first $r$ layers have all been hit, $d(A\upharpoonright n)\ge r$ forever. Thus this construction already gives $d(A\upharpoonright n)\to\infty$, and the remaining implication follows because every martingale is a supermartingale.

**Bookkeeping that matters.** Count pairs $(i,\sigma)$, not just words: the same cylinder appearing in different layers needs a separate copy of its betting strategy for each layer. A prefix-free presentation prevents double-counting measure within a layer, and the layer-hit lengths need not be increasing—use their running maximum.

## The three intuitions, with the quantifiers visible

| nonrandom means… | exact condition | bridge |
| :-- | :-- | :-- |
| compressible | $\sup\_n\bigl(n-K(A\upharpoonright n)\bigr)=\infty$, equivalently $\forall c\ \exists n\ K(A\upharpoonright n)<n-c$ | 4.1 |
| untypical | $A\in\bigcap\_i U\_i$ for some Martin-Löf test; equivalently, $A$ fails some Solovay test | 1.5, 9.4 |
| predictable | some c.e. (super)martingale has unbounded capital along $A$ | 11.6 |
| sustainably predictable | some c.e. supermartingale has capital tending to infinity along $A$ | 11.5 |

**Quantifier trap.** “$K(A\upharpoonright n)\le n-O(1)$ infinitely often” is not the right definition of nonrandomness: a fixed bounded deficiency does not violate randomness. The deficiencies must be **unbounded**. Likewise, beating every fixed capital threshold is stronger than merely making a profit once.

## One-glance summary

| Result | Statement | Key move |
| :-- | :-- | :-- |
| 11.1 | $d(\sigma)\ge\tfrac12(d(\sigma0)+d(\sigma1))$; equality for martingales; success $=$ unbounded capital | fair expectation, no borrowing |
| prefix-free exercise | $\sum\_{\tau\in S}2^{-l(\tau)}d(\tau)\le2^{-l(\sigma)}d(\sigma)$ | finite tree pruning, then monotone convergence |
| 11.2 | measure of ever reaching $k$ is at most $d(\lambda)/k$ | disjoint first-crossing cylinders |
| normalized exercise | reaching $k$ times positive initial capital has probability at most $1/k$ | substitute the threshold; fixed-word betting gives equality at $k=2^m$ |
| 11.3 | computable $=$ uniform prescribed accuracy; c.e. $=$ uniform lower approximation | strict threshold crossings are enumerable |
| 11.4 | summable initial capitals give a finite-valued c.e. sum | $d\_i(\sigma)\le2^{l(\sigma)}d\_i(\lambda)$; dovetail |
| 11.5 | $\limsup d=\infty\iff\lim\tilde d=\infty$, preserving effectiveness | unbounded savings plus a bounded active stake |
| 11.6, strategy → test | $U\_n=\bigcup\_{d(\sigma)>2^nk}[[\sigma]]$ | maximal inequality and strict comparison |
| 11.6, test → strategy | $d=\sum\_{i,\sigma\in L\_i}g\_\sigma$, initial capital $\le1$ | each layer pays one permanent unit on a failing path |

**Connection to probability.** The quantity $m(\sigma)=2^{-l(\sigma)}d(\sigma)$ satisfies $m(\sigma)\ge m(\sigma0)+m(\sigma1)$: a supermartingale is a tree semimeasure divided by fair-coin cylinder mass. For martingales equality holds, so the masses extend to a finite measure. This is why first-crossing capital bounds look like measure bounds, and why conditional cylinder probabilities are the elementary fair bets.

**Connection to weaker randomness.** Restricting the adversary to **computable** martingales defines computable randomness, a strictly weaker requirement than Martin-Löf randomness; the lower-semicomputable class in 11.6 is essential. See [Figueira–Nies, *Feasible analysis, randomness, and base invariance*, §1–2](https://www.cs.auckland.ac.nz/~nies/papersnew/FigueiraNies_Polyrandomness_baseinvariant.pdf).
