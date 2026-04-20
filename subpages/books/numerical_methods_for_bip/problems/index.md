---
title: Problems from the NMBIP course
layout: default
noindex: true
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
