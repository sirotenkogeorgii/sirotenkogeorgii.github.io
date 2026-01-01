---
layout: default
title: Variational Bayesian Methods
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Variational Bayesian Methods

At its core, **variational inference (VI)** is a Bayesian approach. Like in standard machine learning, the model learns from data—but Bayesian methods additionally encode prior beliefs about the parameters via a **prior distribution**, and they return **uncertainty** about the parameters in the form of a **posterior distribution**, rather than a single point estimate.

Suppose we have a training set $X=(x_1,\dots,x_n)^\top$ with $n$ examples and model parameters $\theta$. Bayes’ rule gives the posterior: 

$$p(\theta \mid X)=\frac{p(X\mid \theta),p(\theta)}{p(X)}.$$

This posterior $p(\theta\mid X)$ represents a *range of plausible parameter values* (a full distribution), in contrast to many conventional ML approaches that typically aim for a **single best parameter value**.

### Bayesian inference vs. point estimation

In many “classical” ML setups, we choose parameters by minimizing a loss. If we use maximum likelihood, this corresponds to **maximizing** the log-likelihood: 

$$\theta^\star=\arg\max_\theta \log p(X\mid \theta)$$

(or equivalently minimizing the negative log-likelihood, $-\log p(X\mid \theta)$).

Bayesian inference, instead, aims to characterize 

$$p(\theta\mid X),$$ 

the posterior distribution of parameters given the data. Computing this posterior exactly is often intractable because the **evidence** (also called the marginal likelihood) $p(X)=\int p(X\mid\theta)p(\theta),d\theta$ is difficult to evaluate.

In practice, two common strategies are used:
1. **Sampling-based methods**, e.g. Markov chain Monte Carlo (MCMC)
2. **Optimization-based methods**, including variational inference

## The evidence lower bound (ELBO)

- [Evidence lower bound (ELBO)](/subpages/variational-bayesian-methods/elbo/)

The key idea in variational inference is to approximate the true posterior $p(\theta\mid X)$ with a simpler distribution $q(\theta)$ from a chosen family (e.g., Gaussians). We introduce variational parameters $\phi$ that control this approximation and write $q(\theta\mid \phi)$. We then choose $\phi$ so that $q(\theta\mid \phi)$ is as close as possible to $p(\theta\mid X)$.

This is commonly done by maximizing the **evidence lower bound (ELBO)**:

$$
\mathcal{L}(\phi)
= \mathbb{E}_{q(\theta\mid \phi)}
\Big[ \log p(X,\theta) - \log q(\theta\mid \phi) \Big],
$$

where the expectation is taken with respect to $q(\theta\mid \phi)$.

*(Note: $\phi$ typically depends on the dataset $X$; for notational simplicity this dependence is often left implicit.)*
