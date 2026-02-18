---
title: Metropolis-Adjusted Langevin Algorithm
layout: default
noindex: true
---

# Metropolis-Adjusted Langevin Algorithm

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Goal</span><span class="math-callout__name">(Metropolis-adjusted Langevin algorithm (MALA))</span></p>

We want to sample $x \in \mathbb{R}^d$ from a target density $\pi(x)$ (e.g., a Bayesian posterior). Ideally we want i.i.d. samples, but in practice we build a **Markov chain** whose stationary distribution is $\pi$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Metropolis-adjusted Langevin algorithm (MALA))</span></p>

The **Metropolis-adjusted Langevin algorithm (MALA)** is an **MCMC** method for sampling from a target density $\pi(x)$ when you can compute $\log \pi(x)$ and its gradient $\nabla \log \pi(x)$.

It combines:

1. a **Langevin (gradient-informed) proposal** (an Euler step of an SDE whose stationary distribution is $\pi$), and
2. a **Metropolis–Hastings accept/reject step** to make $\pi$ *exactly* invariant (correcting discretization bias).

It tends to propose points in higher-probability regions (often improving mixing vs random-walk Metropolis), while still allowing exploration via noise.

Informally, the Langevin dynamics drive the random walk towards regions of high probability in the manner of a gradient flow, while the Metropolis–Hastings accept/reject mechanism improves the mixing and convergence properties of this random walk.

</div>

## Langevin diffusion

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Langevin diffusion)</span></p>

Consider the **overdamped Langevin Itô SDE**

$$dX_t = \nabla \log \pi(X_t)dt + \sqrt{2}dW_t$$

where $W_t$ is standard $d$-dimensional Brownian motion.

**Equivalent normalization you may also see:**

$$dX_t = \tfrac{1}{2}\nabla \log \pi(X_t)dt + dW_t$$

which is the same dynamics after a rescaling (it’s just a convention).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Langevin diffusion)</span></p>

* Under suitable conditions, the distribution of $X_t$ converges as $t\to\infty$ to a stationary distribution $\rho_\infty$.
* For this diffusion, the stationary distribution is exactly
  
  $$\rho_\infty = \pi$$

</div>
  
## Euler–Maruyama discretization of Langevin diffusion (discrete-time approximation)

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Euler–Maruyama discretization of Langevin diffusion)</span></p>

To simulate the diffusion, choose a step size $\tau>0$ and define a discrete chain $(X_k)$ by

$$
X_{k+1} = X_k + \tau \nabla \log \pi(X_k) + \sqrt{2\tau}\xi_k,
\qquad \xi_k \sim \mathcal{N}(0,I_d)\ \text{i.i.d.}
$$

So, conditional on $X_k=x$,

$$X_{k+1}\mid X_k=x \sim \mathcal{N}\big(x+\tau\nabla\log\pi(x),\ 2\tau I_d\big)$$

**Important:** this discretized chain does *not* generally have $\pi$ as its exact stationary distribution (it’s biased unless $\tau\to 0$).

</div>

## MALA: Metropolis-adjusted Langevin algorithm (fix the bias)

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Metropolis-adjusted Langevin algorithm)</span></p>

MALA takes the Euler–Maruyama step as a **proposal** and then applies a Metropolis–Hastings accept/reject step to make $\pi$ invariant **exactly**.

#### 1. Proposal

Given current state $x$, propose

$$x' = x + \tau \nabla \log \pi(x) + \sqrt{2\tau}\xi,\quad \xi\sim\mathcal{N}(0,I_d)$$

This corresponds to the proposal density

$$q(x' \mid x) = \mathcal{N}\big(x+\tau\nabla\log\pi(x),\ 2\tau I_d\big)$$

Note: typically $q(x'\mid x)\neq q(x\mid x')$ (asymmetric proposal).

#### 2. Acceptance probability

Accept with probability

$$\alpha(x,x')=\min\left(1,\ \frac{\pi(x')q(x\mid x')}{\pi(x)q(x'\mid x)}\right)$$

* If accepted: $X_{k+1}=x'$
* If rejected: $X_{k+1}=x$

This MH correction enforces **detailed balance**, hence $\pi$ becomes the stationary distribution of the Markov chain.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why MALA is useful vs random-walk Metropolis)</span></p>

* Random-walk MH proposes $x' = x + \text{noise}$ (no “guidance”).
* MALA proposes in a direction influenced by $\nabla\log\pi(x)$, so it tends to move toward higher-density regions, often improving mixing.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Anisotropy + preconditioning (scaled directions))</span></p>

If $\pi$ has very different scales in different directions (strongly anisotropic), you often need a very small $\tau$ unless you **precondition**.

Choose a symmetric positive definite matrix $A\in\mathbb{R}^{d\times d}$ and propose

$$x' = x + \tau A\nabla\log\pi(x) + \sqrt{2\tau A}\xi,\quad \xi\sim\mathcal{N}(0,I_d)$$

so that

$$x'\mid x \sim \mathcal{N}\big(x+\tau A\nabla\log\pi(x),\ 2\tau A\big)$$

Then apply the same MH acceptance formula with this new $q(\cdot\mid\cdot)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical notes)</span></p>

* **Needs gradients**: you must compute $\nabla \log\pi(x)$ each step (more expensive than random-walk MH).
* **Stepsize $\tau$ tuning** matters. In certain high-dimensional asymptotic regimes, a commonly cited “optimal” average acceptance rate is about **0.574**.
* There are **preconditioned** versions using a matrix $A$ (helpful when $\pi$ has very different scales in different directions).

</div>
