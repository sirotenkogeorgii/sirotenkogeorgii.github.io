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