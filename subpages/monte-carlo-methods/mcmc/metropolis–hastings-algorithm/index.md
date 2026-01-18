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
 
The resulting sequence can be used to approximate the distribution (e.g. to generate a histogram) or to compute an integral (e.g. an expected value). These sample values are produced iteratively in such a way, that the distribution of the next sample depends only on the current sample value, which makes the sequence of samples a **Markov chain**.

Metropolis–Hastings and other MCMC algorithms are generally used for sampling from multi-dimensional distributions, especially when the number of dimensions is high. For single-dimensional distributions, there are usually other methods (e.g. adaptive rejection sampling) that can directly return independent samples from the distribution, and these are free from the problem of autocorrelated samples that is inherent in MCMC methods.

### Metropolis–Hastings algorithm

The goal is to draw samples from a possibly high-dimensional density of the form

$$f(x) = \frac{p(x)}{Z}, \qquad x \in \mathcal{X},$$

where $p(x)$ is a known nonnegative function and $Z$ is a normalizing constant that may or may not be known.
To move around the state space (transition density), we introduce a **proposal (instrumental) distribution** $q(y \mid x)$, which specifies how to propose a candidate next state $y$ given the current state $x$. Like acceptance–rejection methods, Metropolis–Hastings proceeds by repeatedly proposing and then accepting or rejecting the proposal.

MCMC methods let us **bypass computing the normalization constant** and still approximate the posterior. The Metropolis–Hastings algorithm can draw samples from any probability distribution with probability density $P(x)$, provided that we know a function $f(x)$ proportional to the density $P$ and the values of $f(x)$ can be calculated. **The requirement that $f(x)$ must only be proportional to the density, rather than exactly equal to it, makes the Metropolis–Hastings algorithm particularly useful**, because it removes the need to calculate the density's normalization factor, which is often extremely difficult in practice.

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

<figure>
  <img src="{{ '/assets/images/notes/monte-carlo-methods/matropoolis-hastings-algorithm-bayesian-framework.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>A specific case of the Metropolis-Hastings algorithm in the Bayesian framework where the proposal density is a uniform prior distribution, sampling a normal one-dimensional posterior probability distribution</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How to choose proposal distribution)</span></p>

First, how is $q$, called the jumping distribution chosen? It's up to you, the model-er. A reasonable assumption, as always, would be a Gaussian, but this may change according to the problem at hand. The choice of the jumping distribution will change how you walk, of course, but it an arbitrary choice.

A common choice for $g(x\mid y)$ is a Gaussian distribution centered at $y$, so that points closer to $y$ are more likely to be visited next, making the sequence of samples into a Gaussian random walk. In the original paper by Metropolis et al. (1953), $g(x\mid y)$ was suggested to be a uniform distribution limited to some maximum distance from $y$. More complicated proposal functions are also possible, such as those of **Hamiltonian Monte Carlo**, **Langevin Monte Carlo**, or **preconditioned Crank–Nicolson**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Metropolis–Hastings and other MCMC methods are autocorrelated!)</span></p>

Compared with methods such as **adaptive rejection sampling**, which produces **independent draws** directly from a distribution, **Metropolis–Hastings and other MCMC methods** come with several drawbacks:
* **Samples are autocorrelated.** Although the chain approaches the correct target distribution $P(x)$ in the long run, successive draws tend to resemble one another, so nearby samples are not truly representative of independent variation: a set of nearby samples will be correlated with each other and not correctly reflect the distribution. This reduces the **effective sample size**, often making it much smaller than the total number of draws and potentially increasing estimation error.
  * To obtain nearly independent samples, a technique called **thinning** is used, which involves keeping only every $n$-th sample and discarding the rest.
* **Early samples may be biased.** Even if the chain eventually converges to the desired distribution, the initial part of the run can reflect a very different distribution—especially if the starting point lies in a low-probability region. As a result, a **burn-in** period is typically necessary, where an initial number of samples are thrown away.

However, most simple **rejection sampling** techniques run into the **curse of dimensionality**: as the number of dimensions increases, the rejection rate typically rises **exponentially**, making these methods quickly impractical.

Metropolis–Hastings and other **MCMC** approaches are generally **less severely affected** by this issue. Because of that, when the target distribution is high-dimensional, MCMC methods are often among the few workable options. For this reason, they are widely used to generate samples for **hierarchical Bayesian models** and many other modern high-dimensional statistical models across a range of fields.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Some intuition behind $\alpha$)</span></p>

* The core of Metropolis-Hastings is the choice of $\alpha$. You can think of $\alpha$ as the way you control the sampling procedure. The main idea behind MCMC is that in order to estimation an unknown distribution, you 'walk around' the distribution such that the amount of time spent in each location is proportional to the height of the distribution. What $\alpha$ does is ask, 'compared to our previous location, how much higher/lower are we?' If we are higher, then the chance that we pick to move to the next point is higher, and if we are lower, then it's more likely that we stay where we are (this refers to Step 3 from the algorithm you reference). The precise functional form of $\alpha$ can be derived, fundamentally, it comes from the condition that we want our final distribution to be stationary.

* In $f(y)q(x\mid y)$ the term $q(x\mid y)$ could be viewed as a measure of how much it is likely to jump to $x$ from $y$ and $f(y)$ is a measure of trust to the measure $q(x\mid y)$. Similarly for $f(x)q(y\mid x)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Standard Metropolis–Hastings in high-dim could be challenging)</span></p>

For **multivariate** distributions, the standard Metropolis–Hastings method proposes a new point in the **full multi-dimensional space** at each step. In high dimensions, choosing an effective proposal (or “jumping”) distribution becomes challenging: different coordinates can have very different scales and behavior, and the step size needs to be tuned to be “just right” across all dimensions. If it isn’t, the chain can mix very slowly.

A common alternative that often performs better in these settings is **Gibbs sampling** ((link))[/subpages/monte-carlo-methods/mcmc/gibbs-sampling/]. Instead of proposing a new value for the entire vector at once, Gibbs updates **one component at a time**, sampling each variable while holding the others fixed. This turns a difficult high-dimensional sampling task into a sequence of lower-dimensional (often one-dimensional) sampling problems.

This approach is particularly useful when the joint distribution is built from many **random variables** where each variable depends on only a small subset of the others — as is typical in many **hierarchical models**. The sampler cycles through variables, updating each using the most recent values of the rest.

Depending on the form of the conditional distributions, the individual coordinate updates can be generated in different ways — for example using **adaptive rejection sampling**, **adaptive rejection Metropolis** methods, a simple **one-dimensional Metropolis–Hastings** step, or **slice sampling**.

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition behind Metropolis algorithm)</span></p>

In Metropolis algorithm (symmetric proposal distribution), if we attempt to move to a point that is more probable than the existing point (i.e. a point in a higher-density region of $P(x)$ corresponding to an $\alpha >1$), we will always accept the move. However, if we attempt to move to a less probable point, we will sometimes reject the move, and the larger the relative drop in probability, the more likely we are to reject the new point. 

Thus, we will tend to stay in (and return large numbers of samples from) high-density regions of $P(x)$, while only occasionally visiting low-density regions. Intuitively, this is why this algorithm works and returns samples that follow the desired distribution with density $P(x)$.

</div>