---
title: Problems from the NMBIP course
layout: default
noindex: true
tags:
  - inverse-problems
  - bayesian-inference
  - numerical-methods
  - probability
  - exercises
---

<style>
  .accordion summary {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    background-color: var(--accent-soft, #f5f6ff);
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--accent-strong, #2c3e94);
    border-radius: 0.25rem;
  }
</style>

# NMBIP Course Problems

**Table of Contents**
- TOC
{:toc}

---

* [Exercise Sheet 4](/subpages/books/numerical_methods_for_bip/problems/sheet04/)
* [Exercise Sheet 5](/subpages/books/numerical_methods_for_bip/problems/sheet05/)

## Exercise Sheet 0 — Warm-up Quiz

### Exercise 0.1: Bertrand's Paradox

Consider two concentric circles of radius $1$ and $2$. Choose a chord of this circle at random. What is the probability that the chord intersects the inner circle?

Three solutions are proposed:

1. **Midpoint argument.** The chord intersects the inner circle if and only if the midpoint of the chord belongs to the inner circle. Thus

   $$\Pr(\text{chord intersects inner circle}) = \frac{\text{area of inner circle}}{\text{area of outer circle}} = \frac{\pi \cdot 1^2}{\pi \cdot 2^2} = \frac{1}{4}.$$

2. **Endpoint–angle argument.** By rotational symmetry we may assume the chord's starting point is at $(-2, 0)$. The chord hits the inner circle iff the angle $\varphi$ with the horizontal belongs to $[-\pi/6, \pi/6]$. Thus

   $$\Pr(\text{chord intersects inner circle}) = \frac{2(\pi/6)}{2(\pi/2)} = \frac{1}{3}.$$

3. **Perpendicular–distance argument.** By symmetry assume the chord is vertical, parameterized by its signed distance $d \in [-2, 2]$ from the centre. The chord intersects the inner circle iff $|d| < 1$, hence

   $$\Pr(\text{chord intersects inner circle}) = \frac{2}{4} = \frac{1}{2}.$$

**Why do we obtain three different results?**

<details class="accordion" markdown="1">
<summary>Solution 0.1</summary>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The phrase "at random" is underdetermined)</span></p>

The three answers are all *correct* — they simply answer three *different* questions. The statement "choose a chord at random" does not single out a probability distribution on the (infinite) space of chords of a disk: it silently picks one of many possible parameterisations and transports Lebesgue measure through it. Which parameterisation is used is exactly what distinguishes the three methods:

| Method | Uniform random variable | Induced distribution on chords |
|:--|:--|:--|
| (1) midpoint | point in the disk of radius 2 | uniform on chord midpoints |
| (2) endpoint–angle | angle $\varphi \in [-\pi/2, \pi/2]$ with one endpoint fixed | uniform on one endpoint direction |
| (3) perpendicular distance | distance $d \in [-2, 2]$ of the chord from the centre | uniform on the signed distance |

These are genuinely *different* measures on the set of chords. For instance, method (1) concentrates on chords with near-diametric length (a midpoint close to the centre is a long chord), while method (3) gives equal mass to every perpendicular offset, favouring shorter chords near the boundary. Hence they disagree on the probability that the chord crosses a given set.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Invariance cuts the paradox)</span></p>

Jaynes (1973) proposed that any "natural" random chord law should be invariant under translations, rotations, and scalings of the ambient line/disk (the problem itself has no preferred centre or scale once the circle is given). Under these three invariances, the law on chords is *unique* up to normalisation — and it coincides with method (3), giving $\tfrac{1}{2}$. Methods (1) and (2) violate translation and scale invariance respectively.

The lesson for Bayesian inverse problems: *a prior is a modelling choice*, and seemingly innocent phrases such as "uniform" or "at random" implicitly fix one. Changing the parameterisation changes the prior, so posterior answers can shift dramatically even when the likelihood is identical.

</div>

</details>

---

### Exercise 0.2: Monte Carlo — Volume of the Unit Ball

Suppose we have access only to samples from a Gaussian. We want to estimate the volume of the $d$-dimensional Euclidean unit ball

$$B_d := \{x \in \mathbb{R}^d : \|x\| < 1\}, \qquad \|x\| = \sqrt{\textstyle\sum_{j=1}^d x_j^2}.$$

**(0.2a)** Sketch a simple algorithm that achieves this.

**(0.2b)** What difficulties might occur as $d$ grows? Plot the error convergence for $d \in \{2^j : j = 0, \dots, 5\}$ and explain your observations.

<details class="accordion" markdown="1">
<summary>Solution 0.2</summary>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Reference value</span><span class="math-callout__name">(Analytic volume)</span></p>

$$\operatorname{vol}(B_d) = \frac{\pi^{d/2}}{\Gamma\!\left(\tfrac{d}{2}+1\right)}.$$

This is the target we compare our Monte Carlo estimator against.

</div>

**(0.2a) Algorithm via importance sampling.**

We only have Gaussian samples, not uniform samples on a bounding box, so we cannot directly run hit-or-miss Monte Carlo with respect to Lebesgue measure. Instead rewrite the volume as a Gaussian expectation. Let $\varphi_d(x) = (2\pi)^{-d/2} \exp(-\tfrac{1}{2}\|x\|^2)$ be the standard Gaussian density. Then

$$\operatorname{vol}(B_d) = \int_{\mathbb{R}^d} \mathbf{1}_{B_d}(x)\, dx = \int_{\mathbb{R}^d} \frac{\mathbf{1}_{B_d}(x)}{\varphi_d(x)}\, \varphi_d(x)\, dx = (2\pi)^{d/2}\, \mathbb{E}_{X \sim \mathcal{N}(0,I_d)}\!\left[\mathbf{1}\{\|X\| < 1\}\, e^{\|X\|^2/2}\right].$$

The unbiased Monte Carlo estimator from $N$ i.i.d. Gaussian samples $X^{(1)}, \dots, X^{(N)}$ is therefore

$$\widehat{V}_d^{(N)} \;=\; \frac{(2\pi)^{d/2}}{N}\sum_{i=1}^{N} \mathbf{1}\{\|X^{(i)}\| < 1\}\, \exp\!\Big(\tfrac{1}{2}\|X^{(i)}\|^2\Big).$$

```text
Input:  dimension d, sample size N
Output: estimate of vol(B_d)

1. V_sum <- 0
2. for i = 1..N:
     sample X ~ N(0, I_d)         # only Gaussian samples used
     r2    <- ||X||^2
     if r2 < 1:
         V_sum <- V_sum + exp(r2 / 2)
3. return (2*pi)^(d/2) * V_sum / N
```

**(0.2b) Curse of dimensionality.**

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why variance explodes in $d$)</span></p>

Under $X \sim \mathcal{N}(0, I_d)$, $\|X\|^2 \sim \chi^2_d$ concentrates near $d$. Concretely $\mathbb{E}\|X\|^2 = d$ and $\operatorname{Var}\|X\|^2 = 2d$. Two things go wrong simultaneously as $d$ grows:

1. **Almost no samples fall in $B_d$.** The acceptance probability

   $$p_d := \Pr(\|X\| < 1) = \Pr(\chi^2_d < 1)$$

   decays super-exponentially in $d$: a Gaussian in high dimensions lives in a thin shell of radius $\sqrt{d}$, so the unit ball is exponentially far out in the *left* tail. For $d=32$, $p_d \approx 10^{-22}$.

2. **Conditional on a hit, the weight is fine, but the normalising constant is huge.** The weight $\exp(\tfrac{1}{2}\|X\|^2)$ is $O(1)$ when $\|X\| < 1$, which is bounded, but the prefactor $(2\pi)^{d/2}$ grows exponentially, so even tiny relative fluctuations in the number of hits translate to huge absolute fluctuations in the estimate.

Combining the two effects, the relative Monte Carlo error

$$\frac{\operatorname{std}(\widehat{V}_d^{(N)})}{\operatorname{vol}(B_d)} = \frac{1}{\sqrt{N}}\sqrt{\frac{(2\pi)^{d/2}\, \mathbb{E}_\varphi\!\big[\mathbf{1}_{B_d}\, e^{\|X\|^2}\big]}{\operatorname{vol}(B_d)^2} - 1}$$

grows **exponentially in $d$** for fixed $N$ (by Stirling, roughly like $(2\pi / e)^{d/4}/\sqrt{N}$).

</div>

**Expected plot.** Running the estimator with a fixed budget $N$ (say $N=10^5$) and plotting the relative error on a log scale against $d \in \{1, 2, 4, 8, 16, 32\}$:

- $d = 1, 2, 4$: relative error $\sim N^{-1/2}$, roughly the textbook Monte Carlo rate.
- $d = 8$: relative error visibly larger, variance starts to dominate.
- $d = 16$: very few (or zero) samples fall in $B_d$; the estimator is either $0$ or dominated by one or two hits.
- $d = 32$: with any reasonable budget the estimator returns $\widehat{V}_d = 0$ with overwhelming probability; relative error is $\approx 1$.

On a semilog plot ($d$ vs. $\log$ relative error), the curve is roughly linear (exponential growth), confirming the variance bound above.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How one would actually do this)</span></p>

In practice, *importance sampling with the prescribed Gaussian proposal* is the wrong tool for $d \gtrsim 10$. Better routes: (i) change the proposal to a truncated Gaussian or a uniform on $B_d$; (ii) exploit spherical symmetry — $\|X\|$ alone determines membership, and $\operatorname{vol}(B_d)$ can be written in closed form from the $\chi^2_d$ CDF; (iii) use MCMC to sample the indicator-weighted density. The moral is that *variance* is the right currency for Monte Carlo, and blindly plugging in the first proposal that is easy to sample from can cost many orders of magnitude — a theme that returns forcefully in Bayesian inverse problems in high dimensions.

</div>

</details>

---

### Exercise 0.3: Random ODE

Let $\alpha \sim \mathcal{N}(0, 1)$. For each realisation of $\alpha$ consider

$$\frac{dX}{dt} = \alpha X, \qquad X(0) = 1. \tag{0.3.1}$$

**(0.3a)** Sketch realisations of the trajectory of $X$ for $t \in [0, 1]$.

**(0.3b)** Determine the probability distribution of $X(1)$ and write down its density.

<details class="accordion" markdown="1">
<summary>Solution 0.3</summary>

**(0.3a) Trajectories.**

For a fixed realisation of $\alpha$, separating variables gives

$$X(t) = e^{\alpha t}.$$

Qualitative sketches on $t \in [0, 1]$, starting at $X(0) = 1$:

- $\alpha > 0$: exponential *growth*, curve concave-up, ending at $X(1) = e^{\alpha} > 1$.
- $\alpha = 0$: constant trajectory $X \equiv 1$.
- $\alpha < 0$: exponential *decay*, curve concave-down, ending at $X(1) = e^{\alpha} < 1$.

Since $\Pr(\alpha < 0) = \Pr(\alpha > 0) = \tfrac{1}{2}$, the ensemble of trajectories is symmetric in a *multiplicative* sense: trajectories with $\alpha$ and $-\alpha$ are reciprocals at every fixed time, not reflections.

**(0.3b) Distribution of $X(1)$.**

$X(1) = e^{\alpha}$ with $\alpha \sim \mathcal{N}(0,1)$, so $X(1)$ is **log-normal** with parameters $(\mu, \sigma^2) = (0, 1)$: $\log X(1) \sim \mathcal{N}(0, 1)$.

*Change of variables.* For $y > 0$, let $y = e^{a}$, so $a = \log y$ and $\frac{da}{dy} = \frac{1}{y}$. Then

$$p_{X(1)}(y) = p_\alpha(\log y)\left|\frac{d\log y}{dy}\right| = \frac{1}{y\sqrt{2\pi}} \exp\!\Big(-\tfrac{(\log y)^2}{2}\Big), \qquad y > 0.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sanity checks)</span></p>

- **Median** of $X(1)$: $e^{0} = 1$ (since $\alpha = 0$ is the median of $\mathcal{N}(0,1)$).
- **Mean**: $\mathbb{E}[X(1)] = \mathbb{E}[e^\alpha] = e^{1/2} \approx 1.649$ — the MGF of a standard normal. Mean > median, consistent with the right skew of a log-normal.
- **Variance**: $\operatorname{Var}(X(1)) = (e - 1)\, e \approx 4.671$.
- Already at $t=1$, the variance is several times the median — a small reminder that *expectations of non-linear functionals of Gaussian parameters can be much larger than the "typical" trajectory*, a phenomenon central to uncertainty quantification.

</div>

</details>

---

### Exercise 0.4: Parameter Estimation for Exponential Data

Let $X \sim \operatorname{Exp}(\lambda)$ with density $p(x) = \lambda e^{-\lambda x}$, $x > 0$. Given $n$ i.i.d. samples $X_1, \dots, X_n$, how would you estimate $\lambda$? What can you say about the error?

<details class="accordion" markdown="1">
<summary>Solution 0.4</summary>

**Maximum likelihood.** The log-likelihood is

$$\ell(\lambda) = \sum_{i=1}^{n} \log p(X_i; \lambda) = n \log \lambda - \lambda \sum_{i=1}^{n} X_i.$$

Setting $\ell'(\lambda) = \tfrac{n}{\lambda} - \sum_i X_i = 0$ gives

$$\widehat{\lambda}_{\mathrm{MLE}} = \frac{n}{\sum_{i=1}^{n} X_i} = \frac{1}{\overline{X}_n}.$$

The second derivative $\ell''(\lambda) = -n/\lambda^2 < 0$ confirms a maximum.

**Exact sampling distribution.** Since $X_i \overset{\text{iid}}{\sim} \operatorname{Exp}(\lambda)$, $S_n := \sum_i X_i \sim \operatorname{Gamma}(n, \lambda)$, so $\widehat\lambda = n/S_n$ has an *inverse-gamma* law. Using $\mathbb{E}[S_n^{-1}] = \lambda/(n-1)$ (for $n \geq 2$) and $\mathbb{E}[S_n^{-2}] = \lambda^2/((n-1)(n-2))$ (for $n \geq 3$):

$$\mathbb{E}[\widehat\lambda] = \frac{n}{n-1}\lambda, \qquad \operatorname{Var}(\widehat\lambda) = \frac{n^2 \lambda^2}{(n-1)^2(n-2)}.$$

So the MLE has an $O(1/n)$ positive bias: $\operatorname{Bias}(\widehat\lambda) = \lambda/(n-1)$. A bias-corrected estimator is

$$\widetilde{\lambda} = \frac{n-1}{n}\widehat\lambda = \frac{n-1}{\sum_i X_i},$$

which is unbiased for $n \geq 2$.

**Asymptotic error via CLT + delta method.** The CLT gives $\sqrt{n}\,(\overline{X}_n - 1/\lambda) \Rightarrow \mathcal{N}(0, \operatorname{Var}(X_1))$ with $\operatorname{Var}(X_1) = 1/\lambda^2$. Applying the delta method to $g(x) = 1/x$, using $g'(1/\lambda) = -\lambda^2$:

$$\sqrt{n}\,(\widehat\lambda - \lambda) \;\Rightarrow\; \mathcal{N}\!\big(0,\; \lambda^2\big).$$

Hence the asymptotic standard error is $\operatorname{SE}(\widehat\lambda) \approx \lambda/\sqrt{n}$, i.e. **relative error $1/\sqrt{n}$**, independent of $\lambda$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cramér–Rao and efficiency)</span></p>

The Fisher information for a single observation is

$$I(\lambda) = -\mathbb{E}\!\left[\frac{\partial^2 \log p(X; \lambda)}{\partial \lambda^2}\right] = -\mathbb{E}\!\left[-\frac{1}{\lambda^2}\right] = \frac{1}{\lambda^2}.$$

So the Cramér–Rao lower bound for any unbiased estimator is $\lambda^2/n$, which the MLE *attains asymptotically*: it is asymptotically efficient. In finite samples the MLE is biased but its MSE is still $\lambda^2/n + O(1/n^2)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bayesian variant)</span></p>

A conjugate prior is $\lambda \sim \operatorname{Gamma}(\alpha_0, \beta_0)$. The posterior is $\lambda \mid X_{1:n} \sim \operatorname{Gamma}(\alpha_0 + n,\, \beta_0 + \sum X_i)$, with posterior mean $(\alpha_0 + n)/(\beta_0 + \sum X_i)$ — recovering the MLE in the Jeffreys-like limit $\alpha_0, \beta_0 \to 0$, up to the finite-$n$ bias correction above.

</div>

</details>

---

### Exercise 0.5: Distributed Parameter Estimation

Let $a(x) > 0$ be known. Consider the elliptic PDE

$$\nabla \cdot (a(x)\nabla u(x)) = -f(x)$$

on a bounded $C^1$ domain $D \subset \mathbb{R}^2$, with homogeneous Neumann boundary conditions $a \nabla u \cdot \mathbf{n} = 0$ on $\partial D$.

Given noisy observations of $u$ at a few points in $D$, describe how you would estimate $f$, what difficulties arise, how to characterise *uncertainty* in $f$, and what intuitively drives that uncertainty.

<details class="accordion" markdown="1">
<summary>Solution 0.5 (sketch — open-ended)</summary>

**Setup.** Write the forward map

$$\mathcal{G}: f \longmapsto \big(u(x_1), \dots, u(x_m)\big), \qquad y = \mathcal{G}(f) + \eta, \quad \eta \sim \mathcal{N}(0, \Sigma_\eta),$$

where $u$ solves the PDE for the given $f$. Three structural facts drive everything:

1. $\mathcal{G}$ is **linear** in $f$: the PDE is linear, and point evaluation is linear, provided $u$ is continuous (fine in 2D for Lipschitz enough $a, f$).
2. With pure Neumann BCs, $u$ is determined only up to an additive constant, and solvability requires the **compatibility condition** $\int_D f\, dx = 0$. Point values of $u$ are therefore meaningful only up to a common shift — either fix a gauge (e.g. $\int_D u = 0$) or work with differences of observations.
3. The map from $f$ to $u$ is a **smoothing operator** — applying $(-\nabla\cdot a\nabla)^{-1}$ damps a Fourier mode of wavenumber $k$ by roughly $k^{-2}$. Pointwise sampling of $u$ further restricts to a finite-dimensional observation space.

Composition: a severely ill-posed inverse problem. Loss of two derivatives on the smoothing side, plus $m$ point observations trying to constrain an infinite-dimensional function.

**Estimators.**

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Strategy 1</span><span class="math-callout__name">(Tikhonov-regularised least squares)</span></p>

Choose a functional prior norm (e.g., $H^s$ or $H^1$ Sobolev norm) and solve

$$\widehat{f}_\alpha = \arg\min_f\ \tfrac{1}{2}\|y - \mathcal{G}(f)\|_{\Sigma_\eta^{-1}}^2 \;+\; \tfrac{\alpha}{2}\|f\|_R^2,$$

with regularisation parameter $\alpha > 0$ chosen by e.g. the Morozov discrepancy principle, the L-curve, or generalised cross-validation. $\|\cdot\|_R$ penalises roughness and restores well-posedness at the price of bias towards smooth $f$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Strategy 2</span><span class="math-callout__name">(Bayesian inverse problem)</span></p>

Place a Gaussian prior $f \sim \mu_0 = \mathcal{N}(m_0, C_0)$ (e.g. Matérn or squared-exponential GP, possibly log-Gaussian if $f$ must be positive). With Gaussian noise and linear $\mathcal{G}$, the posterior is Gaussian:

$$\mu^y = \mathcal{N}(m_{\text{post}}, C_{\text{post}}),\qquad C_{\text{post}} = (C_0^{-1} + \mathcal{G}^*\Sigma_\eta^{-1}\mathcal{G})^{-1},$$

$$m_{\text{post}} = m_0 + C_0 \mathcal{G}^*\big(\mathcal{G} C_0 \mathcal{G}^* + \Sigma_\eta\big)^{-1}(y - \mathcal{G} m_0).$$

The posterior *is* the UQ: $m_{\text{post}}$ is the point estimate (also the MAP and conditional mean), and $C_{\text{post}}$ yields credible intervals, variance maps, and samples of plausible $f$.

</div>

**Difficulties.**

1. **Undersampling.** $m$ scalar observations cannot pin down a function. The problem is trivially underdetermined without a prior / regulariser.
2. **Smoothing.** High-frequency components of $f$ are attenuated by the PDE solve, so they are barely visible in $u$. Tiny noise in $y$ translates to *unbounded* error in those modes — the classic ill-posedness of the inverse Laplacian.
3. **Gauge ambiguity.** Neumann BCs leave $u$ determined only up to constants; $f$ is constrained to zero mean. Any estimator must respect this.
4. **Discretisation.** A faithful numerical solution requires finite elements / finite differences on $D$; discretisation bias interacts with regularisation and noise.
5. **Prior sensitivity.** In the under-informed regime (small $m$, strong smoothing), the posterior inherits many features of $\mu_0$ — as in Bertrand's paradox above, *the prior is doing most of the work*.

**Characterising uncertainty.**

- Posterior covariance $C_{\text{post}}$: the pointwise variance map $x \mapsto C_{\text{post}}(x, x)$ visualises where $f$ is well-constrained versus prior-dominated.
- Posterior samples of $f$: give a *functional* sense of plausibility beyond marginals.
- Functionals: push the posterior through any quantity of interest $Q(f)$ to quote UQ on derived quantities.

**Intuitive drivers of uncertainty in $f$.**

| Driver | Effect |
|:--|:--|
| observation noise $\Sigma_\eta$ | scales posterior variance roughly linearly |
| number $m$ and placement of sensors | clustered / sparse sensors leave "dead zones" in $D$ |
| smoothing in $\mathcal{G}$ | high-frequency modes of $f$ stay near-prior, regardless of how much data you collect |
| prior correlation length in $C_0$ | dictates the spatial scale at which posterior variance reduces around sensors |
| discretisation refinement | too-coarse grids bias posterior towards piecewise features |

Rule of thumb: *near a sensor, in directions the forward map preserves, you learn $f$; far from sensors, or in the fine-scale directions the PDE has damped, you do not — you just recover the prior.*

</details>

---

## Exercise Sheet 1

### Exercise 1.1: Eigenvalue Computation

For an $n \times n$ matrix $A$ ($n \geq 3$) consider the perturbed Toeplitz matrix

$$A^\delta := \begin{pmatrix}
1 & 1 & 0 & \cdots & 0 \\
0 & 1 & 1 & \ddots & \vdots \\
\vdots & \ddots & \ddots & \ddots & 0 \\
0 & \cdots & 0 & 1 & 1 \\
\delta & 0 & \cdots & 0 & 1
\end{pmatrix}.$$

**(1.1a)** Show that $A^0 - I$ is nilpotent. What are the eigenvalues of $A^0$?

**(1.1b)** Plot the eigenvalues of $A^\delta$ for $n = 50$ and $\delta \in \{0,\ 10^{-12}\}$ in the complex plane.

**(1.1c)** What are the eigenvalues of $A^\delta$? Explain your observations in (1.1b). *Hint*: compute the eigenvalues of $A^\delta - I$.

<details class="accordion" markdown="1">
<summary>Solution 1.1</summary>

**(1.1a) Nilpotency and eigenvalues of $A^0$.**

Let $N := A^0 - I$. The non-zero entries of $N$ are exactly the $1$'s on the superdiagonal: $N_{i,i+1} = 1$ for $i = 1, \dots, n-1$. So $N e_j = e_{j-1}$ for $j \geq 2$ and $N e_1 = 0$ (this is the *backward* shift on the standard basis). Iterating,

$$N^k e_j = \begin{cases} e_{j-k} & \text{if } j > k, \\ 0 & \text{if } j \leq k.\end{cases}$$

Hence $N^n e_j = 0$ for every $j$, so $N^n = 0$ — i.e. $A^0 - I$ is nilpotent of index $n$.

The eigenvalues of any nilpotent matrix are all zero (its characteristic polynomial is $\lambda^n$). Therefore

$$\sigma(A^0) = \sigma(I + N) = \{1\}, \quad \text{algebraic multiplicity } n.$$

Note $A^0$ is *not* diagonalisable: $A^0 - I = N$ has rank $n-1$, so $\dim \ker(A^0 - I) = 1$. There is a single Jordan block of size $n$ at eigenvalue $1$.

**(1.1b) Numerical experiment.**

```python
import numpy as np, matplotlib.pyplot as plt

n = 50
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
for ax, delta in zip(axes, [0.0, 1e-12]):
    A = np.eye(n) + np.eye(n, k=1)
    A[-1, 0] = delta
    eigs = np.linalg.eigvals(A)
    ax.plot(eigs.real, eigs.imag, 'o')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title(f"n={n}, δ={delta:g}")
```

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/sheet1_toeplitz_eigs.png' | relative_url }}" alt="Left: for delta=0 all 50 eigenvalues collapse to lambda=1. Right: for delta=1e-12 the 50 eigenvalues spread out onto a circle of radius delta^(1/n) ≈ 0.575 centred at 1 in the complex plane, matching the predicted formula 1 + delta^(1/n) e^{2 pi i k / n}." loading="lazy">
  <figcaption>For $\delta=0$ all $50$ eigenvalues coincide at $\lambda = 1$. For $\delta = 10^{-12}$ they fan out onto a circle of radius $\delta^{1/n} \approx 0.575$ around $1$ — a perturbation $10^{-12}$ in the matrix produces an $\mathcal{O}(0.6)$ perturbation in the spectrum.</figcaption>
</figure>

**(1.1c) Eigenvalues of $A^\delta$.**

Following the hint, compute $\det(\lambda I - (A^\delta - I))$. Set $S := A^\delta - I$. The matrix $\lambda I - S$ has $\lambda$ on the diagonal, $-1$ on the superdiagonal, and $-\delta$ in the $(n,1)$ entry. Expand the determinant along the last row, whose only non-zero entries are $(-\delta)$ in column $1$ and $\lambda$ in column $n$:

$$\det(\lambda I - S) = (-\delta)\,(-1)^{n+1} M_{n,1} + \lambda\, M_{n,n}.$$

- **$M_{n,n}$** is the determinant of the upper-triangular $(n-1)\times(n-1)$ matrix with $\lambda$ on the diagonal and $-1$ on the superdiagonal: $M_{n,n} = \lambda^{n-1}$.
- **$M_{n,1}$**: removing row $n$ and column $1$ leaves a lower-triangular block with $-1$ on the diagonal and $\lambda$ on the subdiagonal, so $M_{n,1} = (-1)^{n-1}$.

Combining,

$$\det(\lambda I - S) = (-\delta)\,(-1)^{n+1}(-1)^{n-1} + \lambda^n = -\delta + \lambda^n.$$

Therefore

$$\boxed{\ \lambda \in \sigma(A^\delta - I) \iff \lambda^n = \delta \iff \lambda = \delta^{1/n}\, e^{2\pi i k / n},\ k = 0, \dots, n-1.\ }$$

Adding $1$ back, the eigenvalues of $A^\delta$ itself are

$$\lambda_k(A^\delta) = 1 + \delta^{1/n}\, e^{2\pi i k / n}, \qquad k = 0, \dots, n-1,$$

i.e. they sit on the circle of radius $\delta^{1/n}$ around $1$ in $\mathbb{C}$. For $n = 50$ and $\delta = 10^{-12}$ that radius is $10^{-12/50} = 10^{-0.24} \approx 0.575$ — *exactly* the circle we saw in the picture.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why a tiny $\delta$ blows up the spectrum)</span></p>

The spectral *radius* of the perturbation $A^\delta - A^0$ is just $\lvert\delta\rvert$. Yet the eigenvalues of $A^\delta$ move by $\delta^{1/n}$ — an *$n$-th root* sensitivity. Two complementary ways to see why:

- **Companion-matrix shape.** $S = A^\delta - I$ is exactly the *companion matrix* of the polynomial $\lambda^n - \delta$. Its roots are $n$-th roots of $\delta$. Companion matrices are notoriously ill-conditioned; pseudo-spectra of $A^0$ are large discs around $1$.
- **Jordan-block sensitivity.** $A^0 = I + N$ is one Jordan block at $1$. Bauer–Fike-type bounds for Jordan blocks of size $n$ give exactly $\lvert\lambda - 1\rvert \lesssim \|E\|^{1/n}$ for a perturbation $E$. The exponent $1/n$ is sharp and is the curse here.

The lesson for inverse problems: *backward stability of an algorithm is not enough.* Even if measurement noise produces a backward-stable perturbation $\|A^\delta - A^0\| = \delta$, the eigenvalues — and hence anything we compute *through them* (PCA, model reduction, spectral filtering) — can be off by $\delta^{1/n}$. For $n = 50$ and machine precision $\delta = 10^{-12}$ the loss is essentially total.

</div>

</details>

---

### Exercise 1.2: Moore–Penrose Pseudoinverse — basic identities

Let $X, Y$ be separable Hilbert spaces and $A \in \mathcal{L}(X, Y)$. Show that

**(1.2a)** $A A^\dagger A = A$,

**(1.2b)** $A^\dagger A A^\dagger = A^\dagger$,

**(1.2c)** $A^\dagger A = I_X - P_{\mathcal{N}}$, where $I_X : X \to X$ is the identity and $P_{\mathcal{N}}$ is the orthogonal projection onto $\mathcal{N}(A)$.

<details class="accordion" markdown="1">
<summary>Solution 1.2</summary>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Defining property of $A^\dagger$)</span></p>

Recall the orthogonal decomposition

$$X = \mathcal{N}(A) \oplus \mathcal{N}(A)^\perp, \qquad Y = \overline{\mathcal{R}(A)} \oplus \mathcal{R}(A)^\perp.$$

For $y \in \mathcal{D}(A^\dagger) := \mathcal{R}(A) \oplus \mathcal{R}(A)^\perp$, the *minimum-norm least-squares solution* $A^\dagger y$ is uniquely characterised by

$$A^\dagger y \in \mathcal{N}(A)^\perp, \qquad A(A^\dagger y) = P_{\overline{\mathcal{R}(A)}}\, y.$$

Equivalently $A^\dagger$ is the unique operator satisfying $A A^\dagger = P_{\overline{\mathcal{R}(A)}}\big\rvert_{\mathcal{D}(A^\dagger)}$ and $\mathcal{R}(A^\dagger) \subseteq \mathcal{N}(A)^\perp$. We use these two facts repeatedly below.

</div>

**(1.2a) $AA^\dagger A = A$.**

Take any $x \in X$ and let $y := A x \in \mathcal{R}(A) \subseteq \mathcal{D}(A^\dagger)$. Then

$$A A^\dagger A x \;=\; A A^\dagger y \;=\; P_{\overline{\mathcal{R}(A)}}\, y \;=\; P_{\overline{\mathcal{R}(A)}}\, (Ax) \;=\; A x,$$

since $A x \in \mathcal{R}(A)$ already lies in the closed range. So $A A^\dagger A x = A x$ for every $x$. $\square$

**(1.2b) $A^\dagger A A^\dagger = A^\dagger$.**

Pick $y \in \mathcal{D}(A^\dagger)$ and set $x := A^\dagger y \in \mathcal{N}(A)^\perp$. By (1.2a), $A A^\dagger A x = A x$, so $A x' = A x$ where $x' := A^\dagger A x$. Then $x - x' \in \mathcal{N}(A)$. But by definition $x \in \mathcal{N}(A)^\perp$, and also $x' = A^\dagger(A x) \in \mathcal{N}(A)^\perp$. Hence $x - x' \in \mathcal{N}(A)\cap \mathcal{N}(A)^\perp = \{0\}$, giving $x' = x$. That is exactly $A^\dagger A A^\dagger y = A^\dagger y$. $\square$

**(1.2c) $A^\dagger A = I_X - P_{\mathcal{N}(A)}$.**

For any $x \in X$ write $x = x_\mathcal{N} + x_\perp$ with $x_\mathcal{N} \in \mathcal{N}(A)$, $x_\perp \in \mathcal{N}(A)^\perp$. Then $A x = A x_\perp$, so $A^\dagger A x = A^\dagger A x_\perp$.

We claim $A^\dagger A x_\perp = x_\perp$. Indeed, both sides lie in $\mathcal{N}(A)^\perp$ ($x_\perp$ by definition; $A^\dagger A x_\perp$ because $\mathcal{R}(A^\dagger) \subseteq \mathcal{N}(A)^\perp$). Their difference $z := x_\perp - A^\dagger A x_\perp$ satisfies $A z = A x_\perp - A A^\dagger A x_\perp = A x_\perp - A x_\perp = 0$ by (1.2a). So $z \in \mathcal{N}(A) \cap \mathcal{N}(A)^\perp = \{0\}$.

Therefore $A^\dagger A x = x_\perp = x - x_\mathcal{N} = (I_X - P_{\mathcal{N}(A)}) x$. $\square$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Symmetric companion identity)</span></p>

A symmetric calculation (or applying (1.2c) to $A^*$ together with $(A^*)^\dagger = (A^\dagger)^*$) gives the dual identity

$$A A^\dagger = P_{\overline{\mathcal{R}(A)}},$$

i.e. $A A^\dagger$ is the orthogonal projector onto the closure of the range. Together (1.2a)–(1.2c) plus this identity are the four *Penrose conditions*; they characterise $A^\dagger$ uniquely in the bounded case.

</div>

</details>

---

### Exercise 1.3: Moore–Penrose Pseudoinverse — explicit computation

Compute $A^\dagger \in \mathbb{R}^{2\times 2}$ for

$$A := \begin{pmatrix} 1 & 2 \\ 1 & 2 \end{pmatrix}.$$

<details class="accordion" markdown="1">
<summary>Solution 1.3</summary>

**Rank-1 SVD.** Notice $A = u v^\top$ with $u = \begin{pmatrix}1\\1\end{pmatrix}$, $v = \begin{pmatrix}1\\2\end{pmatrix}$. Normalise:

$$u = \sqrt{2}\, \hat u,\qquad \hat u = \tfrac{1}{\sqrt 2}\begin{pmatrix}1\\1\end{pmatrix}; \qquad v = \sqrt 5\, \hat v,\qquad \hat v = \tfrac{1}{\sqrt 5}\begin{pmatrix}1\\2\end{pmatrix}.$$

So $A = \sigma_1 \hat u \hat v^\top$ with the single non-zero singular value $\sigma_1 = \sqrt 2 \cdot \sqrt 5 = \sqrt{10}$. The four fundamental subspaces:

$$\mathcal{R}(A) = \operatorname{span}(\hat u),\quad \mathcal{N}(A^\top) = \operatorname{span}(\hat u^\perp),\quad \mathcal{R}(A^\top) = \operatorname{span}(\hat v),\quad \mathcal{N}(A) = \operatorname{span}(\hat v^\perp).$$

**Pseudoinverse via SVD.** $A^\dagger = \sigma_1^{-1} \hat v \hat u^\top$:

$$A^\dagger = \frac{1}{\sqrt{10}} \cdot \tfrac{1}{\sqrt 5}\begin{pmatrix}1\\2\end{pmatrix} \cdot \tfrac{1}{\sqrt 2}\begin{pmatrix}1 & 1\end{pmatrix} = \frac{1}{10}\begin{pmatrix}1 & 1\\ 2 & 2\end{pmatrix}.$$

So

$$\boxed{\ A^\dagger = \frac{1}{10}\begin{pmatrix}1 & 1\\ 2 & 2\end{pmatrix}.\ }$$

**Verification of the four Penrose identities.**

$$A A^\dagger = \frac{1}{10}\begin{pmatrix}1\,{+}\,2 & 1\,{+}\,2 \\ 1\,{+}\,2 & 1\,{+}\,2\end{pmatrix} \cdot ? \text{ — let's compute carefully:}$$

$$A A^\dagger = \begin{pmatrix}1 & 2 \\ 1 & 2\end{pmatrix} \cdot \frac{1}{10}\begin{pmatrix}1 & 1\\ 2 & 2\end{pmatrix} = \frac{1}{10}\begin{pmatrix} 1+4 & 1+4 \\ 1+4 & 1+4 \end{pmatrix} = \frac{1}{2}\begin{pmatrix}1 & 1\\ 1 & 1\end{pmatrix} = \hat u \hat u^\top = P_{\mathcal{R}(A)}. \checkmark$$

$$A^\dagger A = \frac{1}{10}\begin{pmatrix}1 & 1\\ 2 & 2\end{pmatrix} \begin{pmatrix}1 & 2\\ 1 & 2\end{pmatrix} = \frac{1}{10}\begin{pmatrix} 2 & 4 \\ 4 & 8\end{pmatrix} = \frac{1}{5}\begin{pmatrix}1 & 2\\ 2 & 4\end{pmatrix} = \hat v \hat v^\top = P_{\mathcal{R}(A^\top)}. \checkmark$$

In particular both $AA^\dagger$ and $A^\dagger A$ are symmetric orthogonal projections, $AA^\dagger A = A$, and $A^\dagger A A^\dagger = A^\dagger$ — the four Penrose conditions.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What does $A^\dagger y$ *mean*?)</span></p>

Given a measurement $y = (y_1, y_2)^\top$, the equation $A x = y$ is

$$x_1 + 2 x_2 = y_1, \qquad x_1 + 2 x_2 = y_2.$$

If $y_1 \neq y_2$ there is *no* solution; if $y_1 = y_2$ there is a *whole line* $\{x : x_1 + 2 x_2 = y_1\}$ of them. The pseudoinverse handles both:

$$A^\dagger y = \frac{y_1 + y_2}{10}\begin{pmatrix}1\\2\end{pmatrix}.$$

It first projects $y$ onto $\mathcal{R}(A) = \operatorname{span}(1,1)^\top$ — i.e. averages $y_1, y_2$ — then returns the unique element of the solution line lying in $\mathcal{R}(A^\top) = \operatorname{span}(1,2)^\top$, the minimum-$\ell^2$-norm solution. Geometrically, this is exactly the picture of "least-squares + minimum norm" — Strang's *four subspaces* in two dimensions.

</div>

</details>

---

### Exercise 1.4: Hard Thresholding of Compact Operators

Let $X$ be a separable Hilbert space, $T \in \mathcal{K}(X, X)$ self-adjoint compact with spectral decomposition $T x = \sum_{n\in\mathbb{N}} \sigma_n \langle x, u_n\rangle_X u_n$, $(u_n)$ an ONB of $X$. With $f_\alpha(t) = t \mathbf{1}\{\,\lvert t\rvert > \alpha\}$ define

$$T_\alpha x := \sum_{n\in\mathbb{N}} f_\alpha(\sigma_n)\langle x, u_n\rangle_X u_n.$$

**(1.4a)** Prove $T_\alpha^\dagger$ is bounded for every $\alpha > 0$.

**(1.4b)** Prove $\|T - T T_\alpha^\dagger T\| \leq \alpha$ and $\|I_X - T_\alpha^\dagger T\| = 1$.

**(1.4c)** For fixed $w \in X$ and $\delta > 0$, $y^\delta := T x + \delta w$, set $x_\alpha^\delta := T_\alpha^\dagger y^\delta$. Assuming $\mathcal{N}(T) = \{0\}$, show

$$\sup_{\|w\|=1} \lim_{\delta \to 0} \|x - x_{\sqrt\delta}^\delta\| = 0, \qquad \lim_{\alpha \to 0} \sup_{\|w\|=1} \|x - x_\alpha^\delta\| = \infty.$$

<details class="accordion" markdown="1">
<summary>Solution 1.4</summary>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Set-up</span><span class="math-callout__name">(Indices we keep / drop)</span></p>

Define

$$I_\alpha := \{n \in \mathbb{N} : \lvert\sigma_n\rvert > \alpha\}, \qquad I_\alpha^c := \mathbb{N}\setminus I_\alpha.$$

Then $T_\alpha x = \sum_{n\in I_\alpha} \sigma_n \langle x, u_n\rangle u_n$, with $T_\alpha u_n = \sigma_n u_n$ for $n \in I_\alpha$ and $T_\alpha u_n = 0$ for $n \in I_\alpha^c$. Compactness of $T$ gives $\sigma_n \to 0$, hence $I_\alpha$ is *finite* for every $\alpha > 0$. Also let $P_\alpha$ be the orthogonal projection onto $V_\alpha := \overline{\operatorname{span}}\{u_n : n \in I_\alpha\}$ and $P_\alpha^\perp = I_X - P_\alpha$ the projection onto its orthogonal complement.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/sheet1_hard_threshold.png' | relative_url }}" alt="Three panels showing the decreasing sequence sigma_n with three threshold levels alpha. Modes above alpha are kept (orange bars), modes below are zeroed out. Lower alpha keeps more modes; the kept modes always have |sigma_n| > alpha, so 1/|sigma_n| < 1/alpha, hence T_alpha-dagger has norm bounded by 1/alpha." loading="lazy">
  <figcaption>Hard thresholding $f_\alpha$ keeps only spectral modes with $\lvert\sigma_n\rvert > \alpha$. Compactness ($\sigma_n \to 0$) makes $\lvert I_\alpha\rvert$ finite, so $T_\alpha$ has finite-dimensional range; on that range it is bounded *below* by $\alpha$, which is exactly what makes $T_\alpha^\dagger$ bounded above by $1/\alpha$.</figcaption>
</figure>

**(1.4a) $T_\alpha^\dagger$ is bounded.**

$T_\alpha$ has finite-dimensional range ($\dim \mathcal{R}(T_\alpha) = \lvert I_\alpha\rvert < \infty$), hence closed range, hence $T_\alpha^\dagger$ is everywhere defined on $X$. Explicitly, on the orthonormal basis,

$$T_\alpha^\dagger u_n = \begin{cases} \sigma_n^{-1} u_n & n \in I_\alpha, \\ 0 & n \in I_\alpha^c.\end{cases}$$

For any $y = \sum_n d_n u_n$, $T_\alpha^\dagger y = \sum_{n\in I_\alpha} \sigma_n^{-1} d_n u_n$, so by Parseval

$$\|T_\alpha^\dagger y\|^2 = \sum_{n\in I_\alpha} \sigma_n^{-2}\, \lvert d_n\rvert^2 \leq \alpha^{-2} \sum_n \lvert d_n\rvert^2 = \alpha^{-2}\,\|y\|^2.$$

Hence $\|T_\alpha^\dagger\| \leq 1/\alpha < \infty$. (The bound is sharp once $\min_{n \in I_\alpha} \lvert\sigma_n\rvert$ approaches $\alpha$.)

**(1.4b) The two norm identities.**

*Both reduce to a single observation*: $T_\alpha^\dagger T = P_\alpha$. To see this, expand $x = \sum c_n u_n$:

$$T x = \sum_n \sigma_n c_n u_n, \qquad T_\alpha^\dagger T x = \sum_{n\in I_\alpha} \sigma_n^{-1} \sigma_n c_n u_n = \sum_{n\in I_\alpha} c_n u_n = P_\alpha x. \checkmark$$

*First inequality.* $T T_\alpha^\dagger T = T P_\alpha = \sum_{n \in I_\alpha} \sigma_n c_n u_n$, so

$$T x - T T_\alpha^\dagger T x = \sum_{n \in I_\alpha^c} \sigma_n c_n u_n,\qquad \|T - T T_\alpha^\dagger T\|^2 = \sup_{\|x\|=1}\!\!\sum_{n\in I_\alpha^c}\!\! \sigma_n^2 \lvert c_n\rvert^2 \leq \sup_{n \in I_\alpha^c} \sigma_n^2 \leq \alpha^2.$$

Taking square roots gives $\|T - T T_\alpha^\dagger T\| \leq \alpha$. (Equality holds whenever some $\lvert\sigma_n\rvert$ saturates the threshold from below.)

*Second equality.* $I_X - T_\alpha^\dagger T = I_X - P_\alpha = P_\alpha^\perp$, the orthogonal projection onto $V_\alpha^\perp = \overline{\operatorname{span}}\{u_n : n \in I_\alpha^c\}$.

For *any* $\alpha > 0$, $V_\alpha^\perp \neq \{0\}$: by compactness $\sigma_n \to 0$, so either there are infinitely many $n$ with $\lvert\sigma_n\rvert \le \alpha$ (true whenever $T$ has infinite rank), or $T$ has finite rank in which case $X$ is infinite-dimensional and $\sigma_n = 0$ for all but finitely many $n$ — those zero $\sigma_n$ are also in $I_\alpha^c$. Either way $V_\alpha^\perp$ contains some unit vector $u$, and $\|(I - P_\alpha) u\| = \|u\| = 1$. As an orthogonal projection onto a non-trivial subspace, $\|P_\alpha^\perp\| = 1$. Hence

$$\|I_X - T_\alpha^\dagger T\| = 1. \quad\square$$

**(1.4c) Reconstruction error.**

Decompose

$$x_\alpha^\delta = T_\alpha^\dagger y^\delta = T_\alpha^\dagger(T x + \delta w) = P_\alpha x + \delta T_\alpha^\dagger w,$$

so the *total reconstruction error* splits into a deterministic *approximation error* $E_{\text{approx}}(\alpha) := P_\alpha^\perp x$ and a stochastic *propagated noise* $E_{\text{noise}}(\alpha,\delta,w) := \delta T_\alpha^\dagger w$:

$$x - x_\alpha^\delta \;=\; \underbrace{P_\alpha^\perp x}_{\text{approximation}} \;-\; \underbrace{\delta\, T_\alpha^\dagger w}_{\text{propagated noise}}, \qquad \|x - x_\alpha^\delta\| \leq \|P_\alpha^\perp x\| + \delta \|T_\alpha^\dagger\|\,\|w\|.$$

This is the *prototype* of the bias–variance tradeoff in regularisation: small $\alpha$ ⇒ small bias ($P_\alpha^\perp x \to 0$) but large variance ($\|T_\alpha^\dagger\|$ blows up), and vice versa.

*First claim (parameter choice $\alpha = \sqrt\delta$).* Use the bound and $\|T_{\sqrt\delta}^\dagger\| \leq \delta^{-1/2}$:

$$\|x - x_{\sqrt\delta}^\delta\| \leq \|P_{\sqrt\delta}^\perp x\| + \delta \cdot \delta^{-1/2} \cdot 1 = \|P_{\sqrt\delta}^\perp x\| + \sqrt\delta.$$

The second term goes to $0$ with $\delta$ uniformly in $w$ (under $\|w\|=1$). For the first: $\mathcal{N}(T) = \{0\}$ together with the spectral decomposition forces $\sigma_n \neq 0$ for all $n$, so as $\alpha \to 0$ each fixed index $n$ eventually leaves $I_\alpha^c$. Writing $x = \sum c_n u_n$ with $\sum \lvert c_n\rvert^2 < \infty$,

$$\|P_\alpha^\perp x\|^2 = \sum_{n \in I_\alpha^c} \lvert c_n\rvert^2 \xrightarrow{\alpha \to 0} 0$$

by dominated convergence (pointwise convergence of $\mathbf{1}_{I_\alpha^c}(n)$ to $0$, dominated by the summable $\lvert c_n\rvert^2$). Setting $\alpha = \sqrt\delta \to 0$ gives $\|P_{\sqrt\delta}^\perp x\| \to 0$. Both terms vanish, so

$$\sup_{\|w\| = 1}\, \lim_{\delta \to 0}\, \|x - x_{\sqrt\delta}^\delta\| = 0.$$

*Second claim ($\delta$ fixed, $\alpha \to 0$).* Here we have the freedom to *choose the worst $w$*. For any $M > 0$, by $\sigma_n \to 0$ pick an index $n^\star$ with $\lvert\sigma_{n^\star}\rvert$ small enough that $\delta / \lvert\sigma_{n^\star}\rvert > M + \|x\|$. For every $\alpha < \lvert\sigma_{n^\star}\rvert$ we have $n^\star \in I_\alpha$. Choose the worst-case noise $w := u_{n^\star}$ (unit vector). Then

$$T_\alpha^\dagger u_{n^\star} = \sigma_{n^\star}^{-1} u_{n^\star}, \quad \|\delta T_\alpha^\dagger u_{n^\star}\| = \delta / \lvert\sigma_{n^\star}\rvert.$$

Reverse triangle:

$$\|x - x_\alpha^\delta\| \;\geq\; \|\delta T_\alpha^\dagger w\| - \|P_\alpha^\perp x\| \;\geq\; \frac{\delta}{\lvert\sigma_{n^\star}\rvert} - \|x\| > M.$$

Taking $\sup_{\|w\|=1}$ for each such $\alpha$ gives a quantity exceeding $M$. Since $M$ was arbitrary,

$$\lim_{\alpha \to 0}\, \sup_{\|w\| = 1} \|x - x_\alpha^\delta\| = +\infty. \quad\square$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What we just proved is the canonical regularisation theorem)</span></p>

This pair of statements is *the* prototype of regularisation theory:

- **Convergence in the noise-free limit (1st claim)**: with $\alpha$ chosen as a "parameter choice rule" $\alpha(\delta) \to 0$ slow enough that $\delta / \alpha(\delta) \to 0$, the regularised solution $x_{\alpha(\delta)}^\delta$ converges to the truth $x$ in operator norm. Here $\alpha = \sqrt\delta$ is one such rule (the *a priori* choice for hard thresholding).
- **Failure of an unguarded inverse (2nd claim)**: keeping $\delta$ positive but letting $\alpha \to 0$ — i.e., trying to use the un-regularised pseudoinverse $T^\dagger$ on noisy data — admits a worst-case noise direction $u_{n^\star}$ whose contribution $\delta / \lvert \sigma_{n^\star}\rvert$ to the error is unbounded.

The two together say: *regularisation is necessary, parameter choice is necessary, and a $\delta$-dependent rule is non-negotiable.*

The same template — "approximation error + amplified noise, trade off via a parameter" — reappears in Tikhonov ($\alpha I$ added to $T^*T$), spectral cut-off, Landweber iteration, conjugate gradient on the normal equations, etc. The hard threshold is the most transparent illustration.

</div>

</details>

---

### Exercise 1.5: Derivative and Integration

Let

$$X = \Big\{ f \in L^2([0,1];\mathbb{C}) : \int_0^1 f = 0 \Big\}, \qquad Y = \Big\{ f \in H^1([0,1];\mathbb{C}) : \int_0^1 f = 0\Big\},$$

with sesquilinear inner products $\langle f,g\rangle_X = \int_0^1 f\overline{g}$, $\langle f,g\rangle_Y = \int_0^1 f' \overline{g'}$, and let $T: X \to Y$ be the integration operator $T f = u$ where $u' = f$, $\int_0^1 u = 0$.

**(1.5a)** Show $T \in \mathcal{L}(X, Y)$ and $T^\dagger \in \mathcal{L}(Y, X)$. Is the inverse problem well-posed?

**(1.5b)** Let $\iota : Y \to X$ be the embedding $\iota(f) = f$. Show $\iota \in \mathcal{L}(Y, X)$.

**(1.5c)** What is $\iota^\dagger$?

**(1.5d)** Set $\widetilde T := \iota \circ T \in \mathcal{L}(X, X)$. Is the inverse problem for $\widetilde T$ well-posed?

<details class="accordion" markdown="1">
<summary>Solution 1.5</summary>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Set-up</span><span class="math-callout__name">(Fourier basis)</span></p>

The exponentials $e_n(x) := \exp(2\pi i n x)$, $n \in \mathbb{Z}$, form an ONB of $L^2([0,1];\mathbb{C})$. Mean-zero functions are exactly those with $c_0 = 0$. So:

$$\{e_n : n \in \mathbb{Z}\setminus\{0\}\} \text{ is an ONB of } X, \qquad \Big\{ \tfrac{e_n}{2\pi \lvert n\rvert} : n \in \mathbb{Z}\setminus\{0\}\Big\} \text{ is an ONB of } Y$$

(the $Y$-norm of $e_n$ is $\|e_n\|_Y = \|e_n'\|_{L^2} = 2\pi \lvert n\rvert$).

</div>

**(1.5a) Boundedness of $T$ and $T^\dagger$.**

Given $f = \sum_{n \neq 0} c_n e_n \in X$, an antiderivative is $\sum_{n \neq 0} c_n / (2\pi i n)\, e_n$, which is already mean-zero (no $n=0$ Fourier coefficient), so it equals $T f$. Hence

$$T e_n = \frac{1}{2\pi i n}\, e_n \quad (n \neq 0), \qquad (Tf)' = f.$$

For the operator norm:

$$\|Tf\|_Y^2 = \int_0^1 \lvert(Tf)'\rvert^2 = \int_0^1 \lvert f\rvert^2 = \|f\|_X^2,$$

so $T$ is an *isometric* embedding. In particular $T \in \mathcal{L}(X,Y)$ with $\|T\|=1$.

$T$ is injective (as an isometry), so we ask about its range. Computing $\|T e_n\|_Y = \|e_n\|_X / (2\pi \lvert n\rvert) \cdot 2\pi \lvert n\rvert = 1$ confirms isometry on basis vectors. The range is $\mathcal{R}(T) = \{u \in Y : u(1) = u(0)\}$ — equivalently, the closed subspace $\overline{\operatorname{span}}\{e_n / (2\pi \lvert n\rvert) : n \neq 0\}$ in $Y$ — closed because it is the kernel of the continuous functional $u \mapsto u(1) - u(0)$ on $H^1$.

Since $T$ is a closed-range injective bounded operator, $T^\dagger : \mathcal{R}(T) \oplus \mathcal{R}(T)^\perp \to X$ is bounded. Concretely: $T^\dagger u = u'$ on $\mathcal{R}(T)$ (differentiation, well-defined because $u \in H^1$), and $T^\dagger \equiv 0$ on $\mathcal{R}(T)^\perp$ (the codimension-$1$ subspace of $u \in Y$ with $u(1) \neq u(0)$). Norm: $\|T^\dagger\| = 1$.

*Is the inverse problem well-posed?* Hadamard:

| | for $T$ |
|:--|:--|
| existence | only for $u \in \mathcal{R}(T)$ — codimension $1$ in $Y$ |
| uniqueness | $T$ injective $\checkmark$ |
| stability | $\|T^\dagger\| = 1 < \infty$, even on $\mathcal{R}(T)^\perp$ via projection $\checkmark$ |

So **yes** — choosing the spaces $X, Y$ correctly (and in particular *using the $H^1$-seminorm on $Y$*) makes integration a well-posed inverse problem. The catch is in (1.5d) below: the moment we forget the $H^1$ norm and treat $u$ as an $L^2$-object (= apply $\iota$), well-posedness collapses.

**(1.5b) Embedding $\iota : Y \hookrightarrow X$.**

By Parseval in $X$, $\|f\|_X^2 = \sum_{n\neq 0} \lvert c_n\rvert^2$ and $\|f\|_Y^2 = \sum_{n\neq 0} 4\pi^2 n^2 \lvert c_n\rvert^2$. For $n \neq 0$, $n^2 \geq 1$, so

$$\|f\|_X^2 = \sum_{n\neq 0} \lvert c_n\rvert^2 \leq \sum_{n\neq 0} n^2 \lvert c_n\rvert^2 = \frac{1}{4\pi^2}\|f\|_Y^2.$$

Hence $\|\iota f\|_X \leq \tfrac{1}{2\pi}\, \|f\|_Y$ — Poincaré's inequality with sharp constant $C = \tfrac{1}{4\pi^2}$. So $\iota \in \mathcal{L}(Y, X)$ with $\|\iota\| = \tfrac{1}{2\pi}$.

**(1.5c) The pseudoinverse $\iota^\dagger$.**

$\iota$ is injective and has dense range: $\mathcal{R}(\iota) = Y$ as a subset of $X$, and $\overline{Y}^X = X$ (the trigonometric polynomials with $c_0=0$ are dense in $X$ and are in $Y$). However, $\mathcal{R}(\iota)$ is *not closed* in $X$ — and that is the crucial difference with $T$.

On the basis: $\iota$ sends the unit-$Y$ vector $e_n / (2\pi \lvert n\rvert)$ to the $X$-vector $e_n / (2\pi\lvert n\rvert)$, which has $X$-norm $1/(2\pi \lvert n\rvert)$. So the singular values of $\iota$ are

$$\sigma_n(\iota) = \frac{1}{2\pi \lvert n\rvert}, \qquad n \neq 0,$$

an unbounded sequence accumulating at $0$. The pseudoinverse acts by inverting the non-zero singular values:

$$\iota^\dagger e_n = (2\pi \lvert n\rvert)\, \frac{e_n}{2\pi \lvert n\rvert} \cdot 2\pi \lvert n\rvert\ \text{(in $Y$-units)} \;\Longleftrightarrow\; \iota^\dagger \,\text{is the identity on $\mathcal{R}(\iota) = Y \subset X$.}$$

More plainly: $\iota^\dagger f = f$ for $f \in Y$, but viewed as an element of $Y$ (i.e. equipped with the $H^1$-seminorm). The catch is that $\iota^\dagger$ is only densely defined on $X$ and *unbounded*: for the family $f_n = e_n \in Y \subset X$ with $\|f_n\|_X = 1$, $\|\iota^\dagger f_n\|_Y = 2\pi \lvert n\rvert \to \infty$. In Fourier coefficients,

$$\iota^\dagger\Big(\sum_{n\neq 0} d_n e_n\Big) = \sum_{n\neq 0} d_n e_n \quad \text{interpreted in $Y$, with norm } \Big(\sum_{n\neq 0} 4\pi^2 n^2 \lvert d_n\rvert^2\Big)^{1/2}.$$

Operationally: $\iota^\dagger$ is *differentiation*. (Recall in $Y$, $\|f\|_Y = \|f'\|_{L^2}$, so demanding $\|f\|_Y < \infty$ is exactly demanding $f' \in L^2$.) The pseudoinverse of "forget that we differentiated" is "differentiate again."

**(1.5d) Well-posedness of $\widetilde T = \iota \circ T : X \to X$.**

$\widetilde T f = $ the antiderivative of $f$, viewed as an $L^2$-object instead of an $H^1$-object. On the Fourier basis,

$$\widetilde T e_n = \iota T e_n = \iota \frac{e_n}{2\pi i n} = \frac{e_n}{2\pi i n}.$$

So the singular values of $\widetilde T$ are $\sigma_n(\widetilde T) = 1/(2\pi \lvert n\rvert)$ for $n \neq 0$. Same as $\iota$ alone (because $T$ is an isometry — composing with an isometry doesn't change singular values).

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/sheet1_T_iota_singvals.png' | relative_url }}" alt="Singular values of three operators on the Fourier basis e_n. T (X to Y, top blue line) has all sigma_n = 1: it is an isometry, bounded inverse, well-posed. iota (Y to X, red dashed) has sigma_n = 1/(2 pi n) decaying like 1/n: unbounded inverse, ill-posed. T-tilde = iota composed with T (teal triangles) coincides with iota since T contributes a factor of 1 only." loading="lazy">
  <figcaption>Singular values on the Fourier basis. <strong>$T$:</strong> flat at $1$ — the integration operator is an isometry from $X$ into $Y$, hence bounded inverse. <strong>$\iota$ and $\widetilde T = \iota \circ T$:</strong> singular values decay like $1/n$, accumulating at $0$ — Hadamard's stability fails, so the inverse problem is ill-posed. The arrow of "ill-posedness" enters with $\iota$, *not* with $T$.</figcaption>
</figure>

Since the singular values of $\widetilde T$ accumulate at $0$, $\widetilde T$ is a *compact* operator (the spectral filter argument from 1.4 applies). Its range $\mathcal{R}(\widetilde T) = Y \subset X$ is dense but *not closed*; $\widetilde T^\dagger$ is unbounded. Concretely $\widetilde T^\dagger u = u'$ — *differentiation in $L^2$* — is the textbook ill-posed operator. Hadamard:

| | for $\widetilde T$ |
|:--|:--|
| existence | only for $u \in Y$ — dense but not closed in $X$ |
| uniqueness | $\widetilde T$ injective ($T$ injective, $\iota$ injective) |
| **stability** | **fails**: $\widetilde T^\dagger$ unbounded since $\sigma_n(\widetilde T) \to 0$ |

So **the inverse problem for $\widetilde T$ is *ill-posed*** — and this is the prototype Hadamard had in mind: differentiation of $L^2$ data is unstable, since high-frequency $L^2$-noise is amplified by the factor $2\pi \lvert n\rvert$ when one tries to reconstruct the underlying $f$ from $\widetilde T f$. *Numerical differentiation* and *deconvolution* are the practical avatars.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The whole point of this exercise)</span></p>

The same operator (integration) is **well-posed** (as $T : X \to Y$) and **ill-posed** (as $\widetilde T : X \to X$). The difference is *which Hilbert structure one places on the output*:

- $Y = H^1$: output norm tracks the *derivative*, which is exactly what the inverse asks for ⇒ stability is automatic.
- $X = L^2$: output norm tracks pointwise $L^2$-content; recovering the derivative from $L^2$-data costs a factor of $n$ per Fourier mode ⇒ instability.

This is the analyst's dual of the *bias–variance / regularisation* picture from 1.4: choosing a *stronger* output norm (more derivatives) is the same kind of thing as putting a *prior* on the input — both shrink the effective ill-posedness by ruling out high-frequency modes. In Bayesian inverse problems this manifests as choosing a Sobolev-class Gaussian prior for $f$, and is the reason Matérn / squared-exponential GP priors are the workhorses of modern UQ.

</div>

</details>

---

### Exercise 1.6: Retrograde Analysis

White to play. The position is

| 8 | ♚ | · | ♔ | · | · | · | · | · |
| 7 | · | · | · | · | · | · | · | · |
| 6 | · | · | · | · | · | · | · | · |
| 5 | · | · | · | · | · | · | · | · |
| 4 | · | · | · | · | · | · | · | · |
| 3 | · | · | · | · | · | · | · | · |
| 2 | · | · | · | · | · | · | · | ♙ |
| 1 | · | · | · | · | · | · | ♗ | · |
|   | a | b | c | d | e | f | g | h |

(Black king a8, white king c8, white bishop g1, white pawn h2.) What were the last two moves?

<details class="accordion" markdown="1">
<summary>Solution 1.6</summary>

**Answer.** White's last move was $\textbf{N b6\!-\!a8\!+}$ (a *discovered* check from the bishop on $g1$). Black's reply was $\textbf{K\!\times\!a8}$, capturing the knight.

**Reasoning by retrograde analysis.** Notation: $S_0$ = current position, $S_1$ = before black's last move, $S_2$ = before white's last move. White is to move in $S_0$; black moved $S_1 \to S_0$; white moved $S_2 \to S_1$.

*Step 1 — black's last move was a king move.* Black has only the king in $S_0$, hence also in $S_1$ (a move can only reduce *opposing* piece count, not your own). So black's last move was a king move ending at $a8$, and its previous square is one of $\{a7, b7, b8\}$.

*Step 2 — eliminate $b7, b8$.* Kings are never adjacent; in $S_1$, white $K$ is on $c8$ (white moved $S_2 \to S_1$, but did it move the king? Even if so, it ended on $c8$). Both $b7$ and $b8$ are adjacent to $c8$, so black king cannot have been on either. Hence

$$\text{Black king was on } a7 \text{ in } S_1, \text{ moved } a7 \to a8.$$

*Step 3 — bishop attack on $a7$ in $S_1$.* In $S_0$ the diagonal $a7$–$g1$ ($f2$, $e3$, $d4$, $c5$, $b6$) is empty. In $S_1$ this diagonal is the same *up to whatever black's move can change*, but black's move is the king move $a7 \to a8$ — which doesn't touch any diagonal square. So in $S_1$ the diagonal is also empty, hence the bishop on $g1$ attacks $a7$. *Black was in check in $S_1$* — consistent with black moving the king out of check.

*Step 4 — what white did in $S_2 \to S_1$ to put black in check.* In $S_2$, black king on $a7$ must *not* be in check (a position where it's white's turn and black just moved cannot leave black in check). So in $S_2$ either the bishop is not on $g1$, or the diagonal is blocked.

- If bishop is on $g1$ in $S_2$ already, the diagonal must be blocked there. The blocker is on one of $\{f2, e3, d4, c5, b6\}$. White's move $S_2 \to S_1$ must remove this blocker — but it can only do so if (i) white's piece moves *off* the blocking square, and (ii) the blocker survives in $S_1$ on its new location. Since the diagonal is empty in $S_1$, the moved piece must end *off* the diagonal. The only piece that *moves off the diagonal yet ends on $a8$* (so it can be captured by black king in step 5 below, accounting for it being absent in $S_0$) is the **knight**, by $\mathbf{N b6 \!-\! a8}$ — the L-move $b6 \to a8$ both leaves the diagonal at $b6$ open *and* uncovers the bishop's attack on $a7$ (discovered check).
- If the bishop wasn't on $g1$ in $S_2$, white's move would be a bishop move to $g1$. But then in $S_2$ the bishop is on the diagonal $a7$–$g1$ at some square, all of which directly attack $a7$ — putting black in check in $S_2$, contradiction. So this case is excluded.

Hence white's move was $\mathbf{N b6 \!-\! a8\!+}$.

*Step 5 — black's reply.* Black is in double check? Let's check: in $S_1$, the knight on $a8$ attacks $\{b6, c7\}$ — *not* $a7$. So black is in *single* check (only the bishop attacks $a7$). Black has three options:
- **Capture the bishop**: bishop is far away, no piece can reach $g1$.
- **Block the diagonal**: black has no other piece.
- **King move**: $a7 \to a8$, $a7 \to a6$, $a7 \to b6$. Of these, $a8$ is a *capture* (knight there), $a6$ doesn't escape — wait, let me recheck attack squares. Bishop $g1$ attacks $a7$ but *not* $a6$ (different colour) or $a8$ (a8 is light, $g1$ is dark, different diagonals); $b6$ is on the bishop's diagonal so still attacked. So legal escapes are $a6$ and $a8$.

The position $S_0$ has black king on $a8$, so black played $\mathbf{K \!\times\! a8}$. (The other escape, $a6$, is also legal in $S_1$, but the puzzle's $S_0$ has the king on $a8$, fixing the move uniquely.)

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/sheet1_retrograde.png' | relative_url }}" alt="Chessboard showing the current position with black king a8, white king c8, white bishop g1 and white pawn h2. A faded white knight is drawn at b6 to indicate where it stood before white's last move. A teal solid arrow shows the white knight's move b6 to a8; a red dashed arrow shows the black king's reply a7 to a8. A dotted teal line shows the bishop's diagonal from g1 to a7, illustrating the discovered check that the knight's departure from b6 unmasked." loading="lazy">
  <figcaption>The unique consistent prelude to the diagram. <strong>White (last):</strong> $\mathrm{N\,b6\!-\!a8\!+}$ — the knight leaves $b6$, simultaneously unmasking the bishop's diagonal (dotted) and putting the king in discovered check. <strong>Black (reply):</strong> $\mathrm{K\!\times\!a8}$ — the only way out of check that lands on $a8$.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What this puzzle is doing on a numerical-methods sheet)</span></p>

It is here as a *retrograde* sanity check: from the *output* (current board) infer a *consistent input* (history). That is exactly what an inverse problem asks. In this small finite setting we get away with deduction up to one residual ambiguity (move 5: $a6$ vs $a8$, resolved by an extra observation), but the moves of the *last two plies* are uniquely determined — the analogue of "the inverse problem $T x = y$ has a *unique* minimum-norm solution". When constraints (legality + king-non-adjacency + check-after-move-must-resolve + spectator pieces unchanged) pin down history uniquely, the inverse problem is well-posed; when several histories are consistent, regularisation (here: extra information) is needed.

</div>

</details>
