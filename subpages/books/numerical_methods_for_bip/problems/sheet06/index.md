---
title: Problems from the Numerical Methods for Bayesian Inverse Problems course. Sheet 06
layout: default
noindex: true
tags:
  - inverse-problems
  - bayesian-inference
  - importance-sampling
  - ratio-estimator
  - self-normalised-estimator
  - effective-sample-size
  - monte-carlo
---

**Table of Contents**
- TOC
{:toc}

## Exercise 6.1 — Importance Sampling

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 6.1</span><span class="math-callout__name">(Importance sampling)</span></p>

Let $\mu$ be a probability measure on $\mathbb{R}$ with *unnormalized* density

$$\widetilde{f}(x) = \exp\left(-\frac{x^2}{2}\right) \big( \sin^2(6x) + 3\cos^2(x)\sin^2(4x) + 1 \big). \tag{6.1.1}$$

We wish to approximate

$$I = \int_{-\infty}^{\infty} x^2 \, \mathrm{d}\mu(x)$$

using only $\widetilde{f}$. Let $g$ be the standard normal density, i.e. $g(x) = \frac{1}{\sqrt{2\pi}} \exp\big( -\frac{x^2}{2} \big)$. We define the unnormalized importance weight

$$w_u(x) := \frac{\widetilde{f}(x)}{g(x)}. \tag{6.1.2}$$

**(6.1a)** Explain why this estimator is self-normalized. Derive the estimator

$$\widehat{I}_N^{\mathrm{RE}} = \frac{\sum_{j=1}^N X_j^2 \, w_u(X_j)}{\sum_{j=1}^N w_u(X_j)},$$

where $X\_1, \ldots, X\_N$ are iid samples from $g$.

**(6.1b)** Implement the estimator for a range of sample sizes $N$. Repeat the experiment several times and plot the empirical mean and variance of $\widehat{I}\_N^{\mathrm{RE}}$ as functions of $N$. Estimate the reference value of $I$ by an independent reference run with $N\_{\mathrm{ref}} = 10^6$ samples and report the value of $N\_{\mathrm{ref}}$.

**(6.1c)** For the same runs compute the normalized weights

$$\overline{w}_{u,j} = \frac{w_u(X_j)}{\sum_{i=1}^N w_u(X_i)}$$

and the effective sample size

$$N_{\mathrm{eff}} = \frac{1}{\sum_{j=1}^N \overline{w}_{u,j}^2}.$$

Plot $N\_{\mathrm{eff}}/N$ against $N$ and comment on the quality of the proposal density $g$ for this example.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution (6.1b)</summary>

**Setup and one key simplification.** Before writing any code it pays to look at the weight (6.1.2) analytically. Since $\widetilde{f}$ carries the *same* Gaussian envelope as the proposal $g$, the exponentials cancel exactly:

$$w_u(x) = \frac{\widetilde{f}(x)}{g(x)} = \sqrt{2\pi} \, \underbrace{\big( \sin^2(6x) + 3\cos^2(x)\sin^2(4x) + 1 \big)}_{=: h(x)}, \qquad 1 \le h(x) \le 5.$$

Two consequences:

* the weights are **uniformly bounded above and below**, $\sqrt{2\pi} \le w\_u(x) \le 5\sqrt{2\pi}$ — the worst weight exceeds the best by at most a factor $5$, whatever $x$ is sampled;
* the constant $\sqrt{2\pi}$ (and any other multiplicative constant) **cancels** in the ratio estimator, exactly as self-normalisation promises — we could equally well work with $h$ alone. This is the whole point of the estimator: it never needs the normalisation constant $Z = \int \widetilde{f}$.

**Implementation.** For each sample size $N \in \lbrace 2^5, \ldots, 2^{17} \rbrace$ we run $R = 200$ independent repetitions of the estimator (vectorised over the repetitions), and one independent reference run with $N\_{\mathrm{ref}} = 10^6$ samples.

```python
import numpy as np

rng = np.random.default_rng(0)

def h(x):                     # bounded modulation factor, 1 <= h <= 5
    return np.sin(6*x)**2 + 3*np.cos(x)**2*np.sin(4*x)**2 + 1.0

def w_u(x):                   # (6.1.2); the Gaussians cancel analytically
    return np.sqrt(2*np.pi) * h(x)

def ratio_estimate(x):        # self-normalised IS estimator of E_mu[X^2]
    w = w_u(x)
    return np.sum(x**2 * w) / np.sum(w)

# independent reference run
N_ref = 10**6
I_ref = ratio_estimate(rng.standard_normal(N_ref))   # 0.828767

# repeated runs over a range of N
Ns, R = 2**np.arange(5, 18), 200
means, variances = [], []
for N in Ns:
    x = rng.standard_normal((R, N))
    w = w_u(x)
    est = (x**2 * w).sum(axis=1) / w.sum(axis=1)     # R ratio estimates
    means.append(est.mean())
    variances.append(est.var(ddof=1))
```

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/sheet6_is_mean_var.png' | relative_url }}" alt="Left: the empirical mean of the self-normalised estimator over 200 repetitions, plotted with two-standard-error bars against N from 32 to 131072 on a logarithmic axis; the error bars shrink steadily and the means settle on the dashed reference line at 0.8288. Right: the empirical variance against N on a log-log scale, following a dashed reference line of slope minus one over four decades." loading="lazy">
  <figcaption>Empirical mean with $\pm 2$ s.e. bars (left) and empirical variance (right) of $\widehat{I}_N^{\mathrm{RE}}$ over $R = 200$ repetitions. The dashed line on the left is the reference value $\widehat{I}_{\mathrm{ref}} \approx 0.8288$ from an independent run with $N_{\mathrm{ref}} = 10^6$ samples; the dashed line on the right has slope $N^{-1}$.</figcaption>
</figure>

**Results and comments.**

* **Reference value.** The independent reference run with $N\_{\mathrm{ref}} = 10^6$ samples gives

  $$\widehat{I}_{\mathrm{ref}} = 0.8288.$$

  As a sanity check (not part of the task), deterministic trapezoidal quadrature of $\int x^2 \widetilde{f} \big/ \int \widetilde{f}$ on a fine grid gives $I \approx 0.8273$; the two agree within one standard error of the reference run ($\mathrm{s.e.} \approx \sqrt{2.4/10^6} \approx 1.6 \cdot 10^{-3}$), as they should.
* **Mean.** The empirical mean is statistically consistent with the reference value for every $N$ — the deviations at small $N$ lie inside the $\pm 2$ s.e. bars. The ratio estimator *is* biased at finite $N$ (numerator and denominator are correlated random quantities), but the bias is $O(1/N)$, one order below the $O(N^{-1/2})$ statistical error, and is invisible at this number of repetitions — consistent with Lemma 5.5.4 of the lecture.
* **Variance.** The empirical variance follows the line $\mathrm{Var} \approx \sigma^2/N$ with $\sigma^2 \approx 2.4$ over four decades — each doubling of $N$ halves the variance. This is the CLT for self-normalised importance sampling: $\sqrt{N}\big( \widehat{I}\_N^{\mathrm{RE}} - I \big) \to \mathcal{N}(0, \sigma\_q^2)$ with the asymptotic variance $\sigma\_q^2$ from the lecture (Lemma 5.5.4, eq. (5.5.6)).

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution (6.1c)</summary>

**Implementation.** The normalised weights and the effective sample size are computed from the same runs as in (6.1b) — note that any multiplicative constant in $w\_u$ (in particular the $\sqrt{2\pi}$, and the unknown $Z$) cancels in $\overline{w}\_{u,j}$, so $N\_{\mathrm{eff}}$ is computable *without any normalisation knowledge*, just like the estimator itself:

```python
# continuing inside the loop over N from part (b):
    wbar = w / w.sum(axis=1, keepdims=True)   # normalised weights, rows sum to 1
    N_eff = 1.0 / (wbar**2).sum(axis=1)       # effective sample size per run
    ess_fraction = N_eff / N                  # in (0, 1]
```

For a deterministic prediction of where $N\_{\mathrm{eff}}/N$ should settle, expand the definition: by the law of large numbers,

$$\frac{N_{\mathrm{eff}}}{N} = \frac{\big( \frac{1}{N}\sum_j w_u(X_j) \big)^2}{\frac{1}{N}\sum_j w_u(X_j)^2} \; \xrightarrow{N \to \infty} \; \frac{\big( \mathbb{E}_g[w_u] \big)^2}{\mathbb{E}_g[w_u^2]} \approx 0.857,$$

where the limit was evaluated by quadrature.

<figure>
  <img src="{{ '/assets/images/notes/books/numerical_methods_for_bip/sheet6_is_ess.png' | relative_url }}" alt="The effective sample size fraction plotted against N from 32 to 131072 on a logarithmic axis. The curve is essentially a horizontal line at 0.857 for all N, with small error bars at small N that shrink to invisible, lying exactly on the dashed theoretical limit line." loading="lazy">
  <figcaption>Effective sample size fraction $N_{\mathrm{eff}}/N$ (mean $\pm$ std over the $R = 200$ runs) against $N$. The dashed line is the deterministic limit $(\mathbb{E}_g[w_u])^2 / \mathbb{E}_g[w_u^2] = 0.857$, computed by quadrature. The fraction is flat in $N$ — no weight degeneracy whatsoever.</figcaption>
</figure>

**Comments on the quality of $g$.** The proposal is close to ideal for this target, and the plot shows it: $N\_{\mathrm{eff}}/N \approx 0.86$ *independently of $N$*, i.e. out of every $N$ weighted samples we retain the statistical power of about $0.86\,N$ ideal samples, and this does not deteriorate as $N$ grows. The structural reason is the one identified in part (b):

* **Exact tail match.** $g$ carries the same Gaussian envelope $e^{-x^2/2}$ as $\widetilde{f}$, so the weight $w\_u = \sqrt{2\pi}\, h$ is bounded above *and below* ($1 \le h \le 5$). No single sample can ever dominate the weight sum — the worst possible weight imbalance is a factor $5$. This is precisely the textbook criterion (cf. Remark 5.5.3 of the lecture): the proposal has tails at least as heavy as the target, so light-tail weight blow-up is impossible.
* **What is lost, and why.** The missing $14\%$ is the price of the weight *fluctuation*: $h$ oscillates between $1$ and $5$ across the support, so the weights are not constant (only a constant weight, i.e. $g \propto \widetilde{f}$, would give $N\_{\mathrm{eff}} = N$). Since $h$ oscillates fast compared to the Gaussian, the loss is a fixed, benign constant.
* **The contrast to keep in mind.** This is the opposite of the degeneracy scenario of Section 5.5.3 of the lecture notes, where the target (a concentrating posterior) and the proposal (the prior) drift apart and $N\_{\mathrm{eff}}/N$ collapses like $n^{-s/2}$. Here target and proposal share their global shape and differ only by a bounded, oscillatory factor — importance sampling then works essentially at full efficiency, and increasing $N$ buys variance reduction at the ideal $N^{-1}$ rate seen in part (b).

</details>
</div>
