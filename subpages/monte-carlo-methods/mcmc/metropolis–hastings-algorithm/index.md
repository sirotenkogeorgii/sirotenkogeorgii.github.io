---
title: Metropolis-Hastings Algorithm
layout: default
noindex: true
---

# Metropolis-Hastings Algorithm

In statistics and statistical physics, the **Metropolis–Hastings algorithm** is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of random samples from a probability distribution from which direct sampling is difficult. 

New samples are added to the sequence in two steps: 
1. a new sample is proposed based on the previous sample, 
2. proposed sample is either added to the sequence or rejected depending on the value of the probability distribution at that point. 
 
The resulting sequence can be used to approximate the distribution (e.g. to generate a histogram) or to compute an integral (e.g. an expected value).

Metropolis–Hastings and other MCMC algorithms are generally used for sampling from multi-dimensional distributions, especially when the number of dimensions is high. For single-dimensional distributions, there are usually other methods (e.g. adaptive rejection sampling) that can directly return independent samples from the distribution, and these are free from the problem of autocorrelated samples that is inherent in MCMC methods.

### Metropolis–Hastings algorithm

The goal is to draw samples from a possibly high-dimensional density of the form

$$f(x) = \frac{p(x)}{Z}, \qquad x \in \mathcal{X},$$

where $p(x)$ is a known nonnegative function and $Z$ is a normalizing constant that may or may not be known.
To move around the state space (transition density), we introduce a **proposal (instrumental) distribution** $q(y \mid x)$, which specifies how to propose a candidate next state $y$ given the current state $x$. Like acceptance–rejection methods, Metropolis–Hastings proceeds by repeatedly proposing and then accepting or rejecting the proposal.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Metropolis–Hastings)</span></p>

To sample from a density $f$ known only up to normalization, choose an initial point $X_0$ with $f(X_0)>0$. For $t = 0,1,2,\dots, T-1$, do:

1. **Propose:** given the current state $X_t$, draw a candidate
   
   $$Y \sim q(\cdot \mid X_t).$$

2. **Accept/reject:** draw $U \sim \mathrm{Unif}(0,1)$ and set
   
   $$
   X_{t+1} =
   \begin{cases}
   Y, & \text{if } U \le \alpha(X_t, Y),\\
   X_t, & \text{otherwise},
   \end{cases}
   $$

   where the **acceptance probability** is
   
   $$\alpha(x,y) = \min\lbrace\frac{f(y)q(x\mid y)}{f(x)q(y\mid x)}, 1\rbrace.$$
   
   Since $f(x)\propto p(x)$, the ratio can equivalently be computed using $p$ in place of $f$.

This produces a Markov chain $X_0, X_1, \dots, X_T$ whose distribution approaches $f$ as $T$ grows (under standard conditions).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How to choose proposal distribution)</span></p>

First, how is $q$, called the jumping distribution chosen? It's up to you, the model-er. A reasonable assumption, as always, would be a Gaussian, but this may change according to the problem at hand. The choice of the jumping distribution will change how you walk, of course, but it an arbitrary choice.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Some intuition behind $\alpha$)</span></p>

* The core of Metropolis-Hastings is the choice of $\alpha$. You can think of $\alpha$ as the way you control the sampling procedure. The main idea behind MCMC is that in order to estimation an unknown distribution, you 'walk around' the distribution such that the amount of time spent in each location is proportional to the height of the distribution. What $\alpha$ does is ask, 'compared to our previous location, how much higher/lower are we?' If we are higher, then the chance that we pick to move to the next point is higher, and if we are lower, then it's more likely that we stay where we are (this refers to Step 3 from the algorithm you reference). The precise functional form of $\alpha$ can be derived, fundamentally, it comes from the condition that we want our final distribution to be stationary.

* In $f(y)q(x\mid y)$ the term $q(x\mid y)$ could be viewed as a measure of how much it is likely to jump to $x$ from $y$ and $f(y)$ is a measure of trust to the measure $q(x\mid y)$. Similarly for $f(x)q(y\mid x)$.

</div>

## Transition kernel and stationarity

One Metropolis–Hastings step can be viewed as sampling from the transition density $\kappa(\cdot \mid x)$. It has two parts: moving to the proposed point, or staying put when the proposal is rejected:

$$\kappa(y\mid x) = \alpha(x,y)q(y\mid x) + \bigl(1-\alpha^*(x)\bigr)\delta_x(y),$$

where

$$\alpha^*(x) = \int \alpha(x,y)q(y\mid x),dy$$

is the overall probability of accepting a proposal from $x$, and $\delta_x$ denotes a point mass (Dirac delta) at $x$.

A key property is **detailed balance**:

$$f(x)\kappa(y\mid x) = f(y)\kappa(x\mid y),$$

which implies that $f$ is a stationary distribution of the chain. If, additionally, the chain can reach any region of the space (e.g., $q(y\mid x)>0$ for all $x,y\in\mathcal{X}$) and there is a positive chance of rejection (so the chain can “hold” at a state), then $f$ is not just stationary but also the limiting distribution.

## Estimating expectations (ergodic average)

To approximate $\mathbb{E}[H(X)]$ for $X\sim f$, one can use the empirical average along the chain:

$$\frac{1}{T+1}\sum_{t=0}^{T} H(X_t)$$

## Metropolis as a special case

The original Metropolis method corresponds to the case of a **symmetric proposal**, meaning

$$q(y\mid x) = q(x\mid y).$$

Hastings extended the method to allow **asymmetric** proposals, leading to the general Metropolis–Hastings acceptance rule above.
