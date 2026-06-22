---
layout: default
title: "What is an integral, really? From Newton to measures, and back"
date: 2026-06-22
categories: [analysis, measure-theory, probability]
mathjax: true
---

> *An integral is a way of turning a function into a single number — a total, or an
> average — in a manner that is linear, monotone, and respects a notion of "size" on
> the domain. The history of integration is a sequence of progressively more robust
> answers to one question: what does it mean to add up infinitely many infinitesimal
> contributions? The modern answer, which subsumes almost all the others, is that we
> integrate **against a measure**.*

## Conventions and the central question

Throughout, $X$ is a set carrying whatever structure the context demands; most often
$X = \mathbb{R}$ or $\mathbb{R}^n$, or an abstract **measurable space** $(X,\mathcal{A})$
with a $\sigma$-algebra $\mathcal{A}$ of "admissible" subsets. Functions are
$f\colon X \to \mathbb{R}$ (sometimes $\mathbb{C}$, or $[0,\infty]$ when positivity is
in play). I write $\mathbf{1}\_E$ for the indicator of a set $E$, "a.e." for *almost
everywhere*, and reserve $\lambda$ (or $dx$) for **Lebesgue measure** on $\mathbb{R}^n$.
The object of interest is the integral

$$
\int_X f \, d\mu,
$$

read as "the integral of $f$ with respect to the measure $\mu$". The thesis of this
post is that this single notation is not a special case among many integrals — it is,
up to one sharp and instructive exception, *the* integral, and Newton's, Riemann's, and
Lebesgue's constructions are three different routes to it (or, in Newton's case, to a
close cousin of it).

A warning up front, since the slogan "every integral is an integral against a measure"
is both the moral of the story and, taken literally, false. The precise statement is:

> **Every *absolutely convergent*, positively-weighted, linear notion of integration is
> integration against a measure.** Conditionally convergent integrals (improper Riemann,
> the Newton/gauge integral of a wild derivative) genuinely escape the measure-theoretic
> framework, because a measure can only ever produce *absolutely* convergent sums.

Keep that boundary in mind; we will meet it head-on at the end.

## 1. What do we actually want from an integral?

The honest way to discover integration is to work backwards from the properties we
insist on, rather than to guess a formula and check it. Suppose we want a rule
$f \mapsto I(f)$ assigning a number to (some class of) functions on $X$. The
non-negotiable demands are:

- **Linearity.** $I(af + bg) = aI(f) + bI(g)$. Doubling the integrand doubles the total.
- **Monotonicity / positivity.** If $f \ge 0$ then $I(f) \ge 0$; equivalently $f \le g
  \Rightarrow I(f)\le I(g)$. The total respects order.
- **Normalization on sets.** There should be a notion of the "size" of a subset
  $E\subseteq X$, namely $\mu(E) := I(\mathbf{1}\_E)$, and $I$ should be consistent with
  it. For an interval $[a,b]\subset\mathbb{R}$ we want $\mu([a,b]) = b-a$.
- **A limit property.** We want $I$ to interact reasonably with limits, so that
  $I(\lim_n f_n) = \lim_n I(f_n)$ holds under mild hypotheses. This is the demand that
  separates the great theories from the merely workable ones.

The key driving idea of the whole subject is already visible: **positivity plus
linearity forces the existence of a "size function" $\mu(E) = I(\mathbf{1}\_E)$, and the
integral is whatever extends that size function from indicators to general functions.**
Everything below is a story about how to perform that extension, and how much
"size" we are allowed to measure.

## 2. Newton: the integral as undoing a derivative

The earliest mature integral is not about area at all; it is about reversing
differentiation. Given $f$ on $[a,b]$, **Newton's integral** asks for a function $F$
with $F' = f$ *everywhere* on $[a,b]$, and declares

$$
\int_a^b f := F(b) - F(a).
$$

(Newton thought in terms of *fluents* and *fluxions* — quantities and their rates — but
this is the content.) Its great virtue is computational: if you can guess the
antiderivative, you are done, with no limiting process in sight. Its conceptual weakness
is that it conflates two genuinely different ideas under one name, a conflation the
later theories had to untangle. The "fundamental theorem of calculus" is, from a modern
vantage, the deep and *non-trivial* assertion that the area-accumulation operation and
the antiderivative operation coincide — and Newton's definition simply postulates them
to be the same thing.

**What's really going on.** The Newton integral is defined by the conclusion of the FTC
rather than by summing contributions, and this makes it incomparable with the others in
subtle ways. It can integrate functions the area-based theories cannot, and vice versa.
Two examples, which we will return to, make the point sharp:

- (**Lebesgue can fail where Newton succeeds.**) Let $F(x) = x^2\sin(1/x^2)$ for $x\ne0$
  and $F(0)=0$. Then $F$ is differentiable *everywhere* on $[0,1]$, so $f := F'$ has a
  Newton integral, equal to $F(1)-F(0)=\sin 1$. But near $0$ the derivative contains a
  term $\sim \tfrac{2}{x}\cos(1/x^2)$ whose absolute value is not integrable, so
  $f \notin L^1[0,1]$: **the Lebesgue integral of $f$ does not exist.**

This already tells us Newton's integral is *not* an integral against a measure: it
assigns a finite value to a function whose absolute integral is infinite, something a
measure can never do.

## 3. Riemann: chop up the domain

Riemann's idea is the one drawn on every blackboard: to find the area under $y=f(x)$,
slice the *domain* $[a,b]$ into small pieces, approximate $f$ by a constant on each
piece, and add up the rectangles. Concretely, take a partition
$a = x_0 < x_1 < \dots < x_n = b$, sample points $\xi_i \in [x_{i-1},x_i]$, and form the
**Riemann sum**

$$
S = \sum_{i=1}^n f(\xi_i)\,(x_i - x_{i-1}).
$$

We say $f$ is Riemann integrable with integral $\int_a^b f$ if these sums converge to a
common limit as the mesh $\max_i (x_i - x_{i-1}) \to 0$, independently of the sampling.
The cleaner bookkeeping uses the **Darboux sums**, replacing $f(\xi_i)$ by the inf and
sup of $f$ on each subinterval; integrability is then the statement that the *lower
integral* (sup of lower sums) and *upper integral* (inf of upper sums) agree.

This theory is concrete, geometric, and recovers the FTC for continuous integrands. Its
limitations, however, are exactly the ones our wish-list warned about:

- **It chokes on rough functions.** The Dirichlet function $\mathbf{1}\_{\mathbb{Q}}$ on
  $[0,1]$ has lower integral $0$ and upper integral $1$: not Riemann integrable, even
  though it is "obviously" $0$ in any sensible accounting (it is nonzero only on a
  countable, hence negligible, set).
- **It is fragile under limits.** The pointwise limit of Riemann-integrable functions
  need not be Riemann integrable (enumerate the rationals and take finite-then-infinite
  indicators), so one cannot freely exchange $\lim$ and $\int$. Riemann's theory is not
  *complete* in this sense.
- **It does not transplant.** The definition leans on the order and interval structure
  of $\mathbb{R}$; it has no native meaning on an abstract space.

Lebesgue's diagnosis of all three is a single sentence: *Riemann partitions the domain,
and the domain is the wrong thing to partition.*

## 4. Lebesgue: partition the range, and measure the preimages

Here is Lebesgue's own analogy. To total a pile of coins, Riemann counts them in the
order he picks them up; Lebesgue first **sorts by denomination**, counts how many coins
of each value there are, and multiplies. The reorganization is trivial for coins and
revolutionary for functions.

Slice the *range* into thin layers $y \in [y_{k}, y_{k+1})$. For each level, ask the
decisive question:

$$
\textit{how large is the set } \lbrace x : f(x) \in [y_k, y_{k+1})\rbrace\,?
$$

If we can assign a size $\mu\big(\lbrace x : f(x)\approx y_k\rbrace\big)$ to such a set, the integral
is $\sum_k y_k \cdot \mu(\lbrace f \approx y_k\rbrace)$ in the limit. **This is the moment measure
theory is born:** the preimages $\lbrace f \in B\rbrace$ are arbitrary-looking sets, far from
intervals, so Lebesgue's integral *requires* a prior notion of the size of fairly
general sets. The integral does not come first and the measure second; logically, the
measure comes first.

The construction proceeds in three honest stages.

- **Simple functions first.** A *simple* function is a finite combination
  $\varphi = \sum_{k} c_k \mathbf{1}\_{E_k}$ with $E_k$ measurable. Its integral is forced
  by linearity and normalization:
  
  $$\int \varphi\,d\mu = \sum_k c_k\,\mu(E_k).$$

- **Nonnegative functions by approximation from below.** For measurable $f \ge 0$,
  
  $$
  \int f\,d\mu := \sup\Big\{ \int \varphi\,d\mu : 0 \le \varphi \le f,\ \varphi \text{ simple}\Big\}.
  $$
  
  The supremum is taken from *below*, which is what makes the monotone convergence
  theorem fall out for free.
- **General functions by splitting.** Write $f = f^{+} - f^{-}$ (positive and negative
  parts) and set $\int f = \int f^{+} - \int f^{-}$, *provided* at least one of the two
  pieces is finite. The function is **integrable** ($f \in L^1$) exactly when
  $\int \lvert f\rvert\,d\mu < \infty$ — note the built-in demand for absolute convergence, which is
  precisely the line Newton's integral crosses.

The Dirichlet function is now trivially integrable with integral $0$, because
$\mu(\mathbb{Q}) = 0$. And the fragility under limits is *repaired* — this is the entire
payoff, packaged as three theorems.

## 5. The payoff: measure theory and its three convergence theorems

A **$\sigma$-algebra** $\mathcal{A}$ is a collection of subsets closed under complements
and countable unions; a **measure** $\mu\colon \mathcal{A}\to[0,\infty]$ satisfies
$\mu(\varnothing)=0$ and **countable additivity**,
$\mu\big(\bigsqcup_n E_n\big)=\sum_n \mu(E_n)$ for disjoint $E_n$. Countable additivity
is the technical heart: it is exactly the hypothesis that lets infinite limiting
operations commute with measurement. From it flow the three pillars that Riemann lacked.

- **Monotone Convergence (MCT).** If $0 \le f_1 \le f_2 \le \cdots$ pointwise, then
  $\int \lim_n f_n \, d\mu = \lim_n \int f_n\,d\mu$. Suprema pass through the integral.
- **Fatou's Lemma.** For $f_n \ge 0$, $\int \liminf_n f_n \le \liminf_n \int f_n$. A
  one-sided inequality that needs no hypotheses at all, and from which much else follows.
- **Dominated Convergence (DCT).** If $f_n \to f$ pointwise and $\lvert f_n\rvert \le g$ for a
  single integrable $g$, then $\int f_n \to \int f$. *This* is the theorem one reaches
  for daily; the domination hypothesis is the price of admission, and the standard
  cautionary example — $f_n = n\,\mathbf{1}\_{(0,1/n)}$, with $\int f_n = 1$ but $f_n \to 0$
  — shows it cannot be dropped (mass escapes to a spike).

The structural reward is **completeness**: the spaces $L^p(\mu)$ of functions with
$\int \lvert f\rvert^p\,d\mu < \infty$ are Banach spaces (Hilbert when $p=2$). Limits stay inside
the theory, which is what makes Lebesgue's framework the natural home for Fourier
analysis, PDE, and probability.

## 6. The unification: is every integral an integral against a measure?

We can now answer the central question precisely. There are three threads to pull
together.

**Riemann is Lebesgue against $dx$.** Every (properly) Riemann-integrable $f$ on
$[a,b]$ is Lebesgue integrable with respect to Lebesgue measure, with the *same value*:
$\int_a^b f \, dx_{\text{Riemann}} = \int_{[a,b]} f \, d\lambda$. Lebesgue's own theorem
even characterizes the Riemann-integrable functions: a bounded $f$ is Riemann integrable
iff its set of discontinuities is $\lambda$-null. So Riemann's integral is not a rival;
it is the restriction of the Lebesgue integral to a special class of integrands.

**Summation is integration against counting measure.** Put the **counting measure**
$\#$ on $\mathbb{N}$, so $\#(E) = \lvert E\rvert$. Then for $a = (a_n)$,

$$
\int_{\mathbb{N}} a \, d\# = \sum_{n} a_n.
$$

Series are integrals. The convergence theorems specialize to classical facts about
swapping sums and limits (Tonelli becomes "you may reorder a double series of
nonnegative terms"). Even more pointedly, the **Dirac measure** $\delta_x$, defined by
$\delta_x(E) = \mathbf{1}\_E(x)$, gives $\int f\,d\delta_x = f(x)$: *evaluation at a point*
is integration against a measure. Discrete and continuous summation are the same
operation wearing different measures — this is perhaps the cleanest justification of the
unifying slogan.

**Stieltjes integration is integration against the induced measure.** Given an
increasing right-continuous $g$, the Riemann–Stieltjes integral $\int f\,dg$ is exactly
$\int f \, d\mu_g$ for the **Lebesgue–Stieltjes measure** $\mu_g\big((a,b]\big) = g(b)-g(a)$.
This is the bridge to probability: a cumulative distribution function $F$ *is* such a
$g$, and "$dF$" is its law.

**The general principle (Riesz–Markov–Kakutani).** All of the above are instances of a
single theorem. On a reasonable space $X$ (locally compact Hausdorff), let
$I\colon C_c(X)\to\mathbb{R}$ be any **positive linear functional** on continuous
compactly-supported functions. Then there is a unique Radon measure $\mu$ with

$$
I(f) = \int_X f \, d\mu \qquad \text{for all } f \in C_c(X).
$$

In words: *a positive, linear notion of "total" is never anything other than integration
against a measure.* This is the rigorous content of the slogan, and it is why the search
for "another kind of integral" essentially always terminates either in a measure or in
the abandonment of positivity.

**Where the slogan breaks — and why that is the right place for it to break.** Recall
$f = F'$ with $F(x)=x^2\sin(1/x^2)$, and recall the improper Riemann integral
$\int_0^\infty \tfrac{\sin x}{x}\,dx = \tfrac{\pi}{2}$, which converges *conditionally*
while $\int_0^\infty \tfrac{\lvert\sin x\rvert}{x}\,dx = \infty$. Both assign a finite, meaningful
value to a function that is **not absolutely integrable**. No measure can do this: in the
measure framework, $f$ is integrable iff $|f|$ is, by construction. The integrals that
escape are exactly the conditionally convergent ones, and the integral that captures all
of them at once — the **Henstock–Kurzweil (gauge) integral**, a featherlight modification
of Riemann's definition in which the mesh is allowed to vary from point to point — does
so precisely by *not* being given by a measure. It contains Newton (it integrates every
derivative and satisfies the FTC with no fine print), it contains improper Riemann, and
it contains Lebesgue, all at once; the price is that it is not an absolutely convergent,
positive functional, and so falls outside Riesz's theorem. That is the sharp boundary of
the slogan, and it is a clean one:

> Integration *with positivity / absolute convergence* $\iff$ integration against a measure.
> Integration *without* it (conditional convergence) is strictly larger and genuinely
> measure-free.

## 7. Switching between measures

Once integrals live against measures, the natural question is how to convert between
them. Two operations both get called "changing the measure," and conflating them is the
most common source of confusion, so I will keep them rigidly apart. One reweights mass
*on the same space*; the other transports mass *to a different space*.

### 7a. Reweighting: densities and Radon–Nikodym

Suppose $\mu$ and $\nu$ live on the same measurable space, and we want to compute a
$\nu$-integral but only know how to integrate against $\mu$. The clean case is when $\nu$
is **absolutely continuous** with respect to $\mu$, written $\nu \ll \mu$, meaning

$$
\mu(E) = 0 \ \Longrightarrow\ \nu(E) = 0 :
$$

$\nu$ assigns no mass to anything $\mu$ regards as negligible. The **Radon–Nikodym
theorem** ($\sigma$-finiteness assumed) then produces a nonnegative density
$\rho = \dfrac{d\nu}{d\mu}$ such that, for every integrable $f$,

$$
\boxed{\ \int_X f \, d\nu \;=\; \int_X f \,\frac{d\nu}{d\mu}\, d\mu\ }
$$

This is the precise sense of *"integrating a quantity defined in the world of $\nu$
through the lens of $\mu$"*: you pay a pointwise toll $\rho(x)$ at each point and then
integrate normally. The notation behaves like a genuine fraction, justifying Leibniz: if
$\nu \ll \lambda \ll \mu$ then the **chain rule**

$$
\frac{d\nu}{d\mu} = \frac{d\nu}{d\lambda}\cdot\frac{d\lambda}{d\mu} \quad (\mu\text{-a.e.})
$$

holds, and $\frac{d\mu}{d\nu} = \big(\frac{d\nu}{d\mu}\big)^{-1}$ when both are
absolutely continuous each way.

**When can we do this?** Exactly when $\nu \ll \mu$. The failure mode is instructive and
completely classified by the **Lebesgue decomposition**: *any* $\sigma$-finite $\nu$
splits uniquely as

$$
\nu = \nu_{\mathrm{ac}} + \nu_{\mathrm{sing}}, \qquad
\nu_{\mathrm{ac}} \ll \mu, \quad \nu_{\mathrm{sing}} \perp \mu,
$$

where the singular part $\nu_{\mathrm{sing}}$ is concentrated on a $\mu$-null set (think
of a Dirac mass relative to Lebesgue measure, or the Cantor measure). The density exists
for $\nu_{\mathrm{ac}}$ and provably cannot exist for the singular part — there is no
function $\rho$ with $\delta_0 = \rho\,d\lambda$, since that would demand an "infinite
spike on a null set." This is the sharp answer to "when may we change measures": you may
reweight precisely up to the absolutely continuous part, and the singular part is the
honest obstruction.

### 7b. Transport: pushforward and change of variables

Now let $T\colon X \to Y$ be a measurable map between (possibly different) spaces. It
carries $\mu$ to the **pushforward** $T_\ast \mu$ on $Y$, defined by

$$
(T_*\mu)(B) := \mu\big(T^{-1}(B)\big).
$$

The corresponding integral identity is the **abstract change of variables**, and unlike
Radon–Nikodym it requires *no* absolute-continuity hypothesis — only measurability:

$$
\boxed{\ \int_Y g \, d(T_\ast \mu) \;=\; \int_X (g\circ T)\, d\mu\ }
$$

The classical Jacobian formula is the special case where $X,Y$ are open sets in
$\mathbb{R}^n$, $T$ is a diffeomorphism, and we measure both sides against Lebesgue. Then
transport and reweighting combine: pushing Lebesgue measure through $T$ produces a
measure with Radon–Nikodym density $\lvert\det DT\rvert$ relative to Lebesgue, giving the familiar

$$
\int_{T(\Omega)} g(y)\,dy = \int_{\Omega} g\big(T(x)\big)\,\lvert \det DT(x)\rvert\,dx.
$$

So the $\lvert\det DT\rvert$ that students memorize is not a separate rule; it is the
Radon–Nikodym derivative $\frac{d(T_\ast \lambda)}{d\lambda}$ in disguise. Transport tells
you *which* measure you have; reweighting expresses it back in coordinates you trust.

### 7c. Integrating in stages: product measures and Fubini

A third manipulation deserves mention. On a product space $X\times Y$ with the **product
measure** $\mu\otimes\nu$, the theorems of **Tonelli** (for $f\ge 0$, no integrability
needed) and **Fubini** (for $f\in L^1$, integrability needed) license

$$
\int_{X\times Y} f \, d(\mu\otimes\nu)
= \int_X\!\Big(\int_Y f(x,y)\,d\nu(y)\Big)d\mu(x)
= \int_Y\!\Big(\int_X f(x,y)\,d\mu(x)\Big)d\nu(y).
$$

This is "changing the measure" in the sense of factoring a high-dimensional integral
into iterated lower-dimensional ones, and the asymmetry between Tonelli and Fubini is the
same absolute-vs-conditional theme yet again: positivity lets you swap freely;
otherwise you must first certify $\int\lvert f\rvert < \infty$, on pain of the classic counterexample
where the two iterated integrals disagree.

## 8. Probability: integration is expectation

A **probability measure** is simply a measure $P$ with total mass $P(X) = 1$. Nothing in
the theory changes; only the vocabulary does, and the dictionary is worth stating because
it makes the previous section's machinery into the working tools of statistics and
finance.

- A random variable $X\colon \Omega \to \mathbb{R}$ is a measurable function, and its
  **expectation is an integral**: $\mathbb{E}[X] = \int_\Omega X \, dP$.
- Its **law** (distribution) is the **pushforward** $P_X := X_\ast P$ on $\mathbb{R}$. The
  *law of the unconscious statistician* is then nothing but the abstract change of
  variables of §7b:
  
  $$
  \mathbb{E}\big[g(X)\big] = \int_\Omega g(X(\omega))\,dP(\omega)
  = \int_{\mathbb{R}} g(x)\, dP_X(x).
  $$
  
  This is why one may compute $\mathbb{E}[g(X)]$ knowing only the distribution of $X$,
  never mentioning the underlying $\Omega$ again.
- The **density** of $X$, when it exists, is the Radon–Nikodym derivative
  $\frac{dP_X}{d\lambda}$ of §7a; the **CDF** is its Stieltjes integrator of §6. Discrete
  random variables are the same statement with counting measure in place of $\lambda$.

The reweighting operation of §7a is the engine room of modern probability:

- **Change of measure.** If $P \ll Q$ with likelihood ratio
  $L = \frac{dP}{dQ}$, then $\mathbb{E}\_P[X] = \mathbb{E}\_Q[XL]$. This single identity
  *is* importance sampling (sample under the convenient $Q$, reweight to recover $P$),
  the risk-neutral pricing change of measure (and, in continuous time, Girsanov's
  theorem), and the mechanism by which a Bayesian posterior is a reweighting of the prior
  by the likelihood.
- **Relative entropy.** The Kullback–Leibler divergence
  $D(P\,\|\,Q) = \int \log\frac{dP}{dQ}\,dP$ is finite only when $P \ll Q$ — the
  obstruction of §7a returns as the statement that you cannot compare distributions whose
  supports disagree on a positive set.

Probability, in short, is measure theory with the total mass normalized to one, and
every "change of measure" a practitioner performs is one of the two operations above.

## 9. What's really going on

Three closing observations, in the spirit of stepping back from the construction.

**An integral is a positive linear functional, and a measure is its infinitesimal
shadow.** Strip away the constructions and an integral is just a linear, monotone map
from functions to numbers; Riesz's theorem says the data of such a map is exactly a
measure. The various historical integrals differ not in *what* they compute but in *how
wide a class of functions* they can compute it for, and how gracefully they handle
limits. Lebesgue wins on both counts among the absolutely convergent theories, which is
why it became the default.

**The boundary of the slogan is the boundary of positivity.** Everything genuinely
escaping the measure framework — improper Riemann integrals, the Newton/gauge integral of
a non-Lebesgue derivative, conditionally convergent series — does so by trading positivity
for cancellation. Once you permit $\int f$ to exist while $\int \lvert f\rvert = \infty$, you have
left the world of measures and entered the world of *limits of measures* or of *linear
functionals without sign*. The next stop along that road is the theory of distributions
(generalized functions), where one integrates against objects like $\delta'$ that are not
measures at all — bought, again, at the cost of positivity.

**The right notation was Leibniz's all along.** The fact that $\frac{d\nu}{d\mu}$ obeys
the chain rule, that $dg$ in a Stieltjes integral really is the differential of a measure,
that $\lvert \det DT\rvert\,dx$ is a transported $dy$ — all of this vindicates treating "$d\mu$" as a
first-class object: an infinitesimal weighting that you may reweight, transport, and
factor. Integration against a measure is the precise theory of that intuition, and once
you have it, Newton, Riemann, Lebesgue, summation, and expectation are five dialects of a
single language.

---

*Suggested further reading:* Tao's own *An Introduction to Measure Theory* for the
construction done carefully; Folland's *Real Analysis* for Radon–Nikodym, Riesz
representation, and product measures in one place; and Bartle's *A Modern Theory of
Integration* for the Henstock–Kurzweil integral and the FTC done without fine print.