---
title: Gibbs Sampling
layout: default
noindex: true
---

# Gibbs Sampling

In statistics, **Gibbs sampling** or a **Gibbs sampler** is a Markov chain Monte Carlo (MCMC) algorithm for sampling from a specified multivariate probability distribution when direct sampling from the joint distribution is difficult, but sampling from the conditional distribution is more practical. This sequence can be used to approximate the joint distribution (e.g., to generate a histogram of the distribution); to approximate the marginal distribution of one of the variables, or some subset of the variables (for example, the unknown parameters or latent variables); or to compute an integral (such as the expected value of one of the variables).

The **Gibbs sampler** can be viewed as a particular instance of the Metropolis–Hastings algorithm for generating $n$-dimensional random vectors. Due to its importance it is presented separately. The distinguishing feature of the Gibbs sampler is that the underlying Markov chain is constructed from a sequence of
conditional distributions, in either a deterministic or random fashion. Suppose that we wish to sample a random vector $X = (X_1,\dots,X_n)$ according to a target pdf $f(x)$. Let $f(x_i \mid x_1,\dots,x_i−1, x_i+1,\dots,x_n)$ represent the conditional pdf of the $i$-th component, $X_i$, given the other components $x_1,\dots,x_i−1,x_i+1,\dots,x_n$. Here we use a Bayesian notation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gibbs Sampler)</span></p>

Given an initial state $X_0$, repeat the following steps for $t = 0, 1, \ldots$:

1. Given the current state $X_t$, generate a new vector
   $Y = (Y_1, \ldots, Y_n)$ as follows:
   * (a) Sample $Y_1$ from the conditional density $f(x_1 \mid X_{t,2}, \ldots, X_{t,n})$.
   * (b) For $i = 2, \ldots, n-1$, sample $Y_i$ from $f(x_i \mid Y_1, \ldots, Y_{i-1}, X_{t,i+1}, \ldots, X_{t,n})$.
   * (c) Sample $Y_n$ from $f(x_n \mid Y_1, \ldots, Y_{n-1})$.
2. Set $X_{t+1} = Y$.

</div>


### What you collect in Gibbs sampling

Gibbs produces a **Markov chain of full vectors**

$$X^{(t)} = \big(x^{(t)}_1,\dots,x^{(t)}_m\big), \quad t=1,2,\dots,n$$

(typically after some **burn-in** iterations you discard).

### How you approximate a marginal

To approximate the marginal of coordinate $j$, you look at the **sequence of the $j$-th components** across iterations:

$$x^{(1)}_j,; x^{(2)}_j,; \dots,; x^{(n)}_j$$

and treat them as (correlated) draws from the marginal $p(x_j)$ once the chain has mixed.

So your statement “take all sampled values at position 1” is right:

* marginal of $X_1$ is approximated by the empirical distribution of $\lbrace x^{(t)}_1\rbrace_{t=1}^n$ (after burn-in).

### The key clarifications

1. **The marginal isn’t “defined by the values”** — it’s **estimated** by the *empirical distribution* built from those values (histogram/KDE for continuous; frequency table for discrete; or just compute expectations like mean/variance).
2. **Samples are not independent.** Consecutive $x^{(t)}_j$ are usually autocorrelated, so “$n$ iterations” is not the same as “$n$ i.i.d. samples”. That affects accuracy (effective sample size), not the basic procedure.

### Tiny example

If you keep 5 post–burn-in Gibbs states:

$$X^{(1)},\dots,X^{(5)}$$

then to estimate $p(x_1)$ you use:

$$x^{(1)}_1,;x^{(2)}_1,;x^{(3)}_1,;x^{(4)}_1,;x^{(5)}_1.$$

If you tell me whether your variables are discrete or continuous, I can show exactly what the marginal estimate looks like (frequency table vs histogram/KDE, plus how to estimate mean/CI).


Because in Gibbs you **don’t have the full target distribution** (p(x)) in a usable form — you only have the **conditional pieces** (p(x_j \mid x_{-j})) (your “(f)”) that are easy to sample from.

Two ways to see it:

### 1) The conditionals are not what you ultimately want

Usually you want marginals/expectations under the **joint** (p(x_1,\dots,x_m)), like

* (p(x_1)), (p(x_1,x_2))
* (\mathbb E[g(X)]), (\mathbb P(X_1>0)), etc.

But the conditional you can sample from is
[
p(x_j \mid x_{-j})
]
which depends on the *current values of the other variables*. That conditional changes at every step, so sampling from it does **not** directly give you samples from the marginal (p(x_j)) or the joint.

### 2) You can sample conditionals, but you can’t *integrate them*

To get a marginal from a conditional you’d need an integral/sum over the unknown distribution of the other variables:
[
p(x_j) = \int p(x_j \mid x_{-j}), p(x_{-j}) , dx_{-j}.
]
You know (p(x_j \mid x_{-j})), but you **don’t know (p(x_{-j}))** (that’s part of the same hard joint). Computing that integral is typically intractable.

Gibbs is a workaround: it generates (x_{-j}^{(t)}) whose distribution (after mixing) is approximately (p(x_{-j})), and then the empirical distribution of (x_j^{(t)}) approximates (p(x_j)).

### A helpful analogy

Knowing the conditional (p(x_1 \mid x_2)) doesn’t tell you (p(x_1)) unless you also know how (x_2) itself is distributed.

### One more subtle point

Even if the joint (p(x)) is known “up to a constant” (common in Bayesian posteriors),
[
p(x) \propto \tilde p(x),
]
that still doesn’t let you compute marginals/expectations analytically because the normalizing constant and high-dimensional integrals are hard. The conditionals might be simple (conjugacy), so Gibbs uses them.

If you want, tell me what “(f)” is in your setup (e.g., Gaussian conditional, categorical, etc.), and what target quantity you want (a marginal? a mean?), and I’ll connect it to the exact formula above.

Mostly yes — but with a nuance:

* In Gibbs sampling you typically **can evaluate the joint density only up to a normalizing constant**:
  [
  p(x) \propto \tilde p(x)
  ]
  so you *don’t* have a tractable closed form for marginals like (p(x_1)=\int p(x_1,x_2,\dots),dx_{2:m}).

* What you **do** have are **full conditionals**
  [
  p(x_j \mid x_{-j})
  ]
  in a form that is **easy to sample from** (often standard distributions).

So it’s not necessarily “we don’t have the full pdf at all”; it’s “even if we have it (or (\tilde p)), we **can’t integrate it** to get marginals/expectations, but the conditionals are easy to sample.”

One more important clarification:
Having conditionals doesn’t automatically mean there exists a unique consistent joint (there are compatibility conditions). In the usual Gibbs/MCMC setting, the conditionals are derived from a well-defined joint (e.g., a Bayesian posterior), so it’s consistent.

**Bottom line:** we use Gibbs because **marginals are hard high-dimensional integrals**, while **sampling from the conditionals is easy**, and the resulting chain (after burn-in/mixing) gives samples whose coordinates behave like draws from the desired marginals.
