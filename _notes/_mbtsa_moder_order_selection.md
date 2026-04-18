## Model Order Selection

A critical step in VAR modeling is determining the appropriate order, $p$. This is a model selection problem where we aim to balance model fit with model complexity. Common criteria include AIC and BIC, but a formal statistical test is the Likelihood Ratio Test.

### Likelihood Ratio Test (Wilks' Theorem)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Likelihood Ratio Test (Wilks' Theorem))</span></p>

The **Likelihood Ratio (LR) test** provides a framework for comparing nested models.

- Let $\mathcal{M}_0$ be a **"restricted" model (null hypothesis, $H_0$)** with parameter space $\Theta_0$.
- Let $\mathcal{M}_1$ be a **"full" model (alternative hypothesis, $H_1$)** with parameter space $\Theta_1$, where $\Theta_0 \subset \Theta_1$.
- Let $\ell\_{\text{max}}(\mathcal{M}_0)$ and $\ell\_{\text{max}}(\mathcal{M}_1)$ be the maximized log-likelihoods for each model.

The **LR test statistic** is defined as:

$$
D = -2 \log \left(\frac{\sup_{\theta \in \Theta_0} \mathcal{L}(\theta)}{\sup_{\theta \in \Theta_1} \mathcal{L}(\theta)}\right) = -2 (\ell_{\text{max}}(\mathcal{M}_0) - \ell_{\text{max}}(\mathcal{M}_1))
$$

Under suitable regularity conditions and assuming $H_0$ is true, **the statistic $D$ follows a chi-squared distribution**:

$$D \sim \chi^2(d_1 - d_0)$$

where $d_1$ and $d_0$ are the **number of free parameters** in the full and restricted models, respectively.

**Decision Rule:** We compare the empirically observed statistic $D_{\text{empirical}}$ to the $\chi^2$ distribution. If the probability of observing a value as large as $D_{\text{empirical}}$ is small (e.g., $p < \alpha$, where $\alpha=0.05$ by convention), we reject the null hypothesis $H_0$ in favor of the more complex model $\mathcal{M}_1$.

</div>

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/chi-squared-distribution-diff-k.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <figcaption>Chi-squared distribution for different $k$.</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/chi-squared-distribution.png' | relative_url }}" alt="Poisson PMF" loading="lazy">
    <figcaption>Chi-squared distribution with $p$ value.</figcaption>
  </figure>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(VAR($p$) vs. VAR($p+1$))</span></p>

We can use the LR test to decide if a VAR($p+1$) model provides a significantly better fit than a VAR($p$) model.

- **Restricted Model ($H_0$):** 
  - VAR($p$) process, 
  - $X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t$
  - This is equivalent to a VAR($p+1$) model with the constraint $A_{p+1} = 0$
- **Full Model ($H_1$):** 
  - VAR($p+1$) process, 
  - $X_t = c + \sum_{i=1}^{p+1} A_i X_{t-i} + \varepsilon_t$

The **LR test** compares the "explained variation" in the data under both models. The maximized log-likelihood is related to the determinant of the estimated residual covariance matrix, $\hat{\Sigma}_\varepsilon$.

$$
\ell_{\text{restricted}}
= \sum_{t=p+2}^{T}
\left(
-\frac{N}{2}\log(2\pi)
-\frac{1}{2}\log\lvert\hat{\Sigma}_0\rvert
-\frac{1}{2}(x_t-\hat{x}_t)^{\top}\hat{\Sigma}_0^{-1}(x-\hat{x}_t)
\right)
$$

$$
\ell_{\text{full}}
= \sum_{t=p+2}^{T}
\left(
-\frac{N}{2}\log(2\pi)
-\frac{1}{2}\log\lvert\hat{\Sigma}\rvert
-\frac{1}{2}(x_t-\hat{x}_t)^{\top}\hat{\Sigma}^{-1}(x-\hat{x}_t)
\right),
\qquad
(x_t-\hat{x}_t) \eqqcolon \varepsilon_t
$$

$$
Q(\varepsilon)
= \sum_{t=p+2}^{T} \varepsilon_t^{\top}\hat{\Sigma}^{-1}\varepsilon_t
= \text{tr}\left(\hat{\Sigma}^{-1}\sum_{t=p+2}^{T}\varepsilon_t\varepsilon_t^{\top}\right)
\quad \text{(trace identity)}
$$

$$\ell_{\text{restricted}} = -\sum_{t=p+2}^T\frac{1}{2} \log(\lvert\hat{\Sigma}_0\rvert) + \text{const}$$

$$\ell_{\text{full}} = -\sum_{t=p+2}^T\frac{1}{2} \log(\lvert\hat{\Sigma}\rvert) + \text{const}$$

The test statistic becomes:

$$D = -2(\ell_{\text{restricted}} - \ell_{\text{full}})$$

$$= -2(-\sum_{t=p+2}^T\frac{1}{2} \log(\lvert\hat{\Sigma}_0\rvert)+\sum_{t=p+2}^T\frac{1}{2} \log(\lvert\hat{\Sigma}\rvert))$$

$$= (T-(p+1))(\log(\lvert\hat{\Sigma}_0\rvert) - \log(\lvert\hat{\Sigma}\rvert))$$

This statistic is compared to a $\chi^2(N^2)$ distribution, as the VAR($p+1$) model has $N^2$ additional free parameters in the matrix $A_{p+1}$.

</div>

### Granger Causality

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Granger Causality)</span></p>

* **Granger causality** is a statistical concept of causality based on prediction.
* Provides a formal method to test for **directed influence between time series** within the VAR framework.

If the past of a time series $X$ contains information that improves the prediction of a time series $Y$, beyond the information already contained in the past of $Y$ and all other relevant variables, then we say that $X$ Granger-causes $Y$ (denoted $X \to Y$).

* Let $X_t, Y_t$ be two time series processes and $Z_t$ represent all other knowledge in the world. 
* Let $\mathcal{E}[Y_{t+1} \mid \text{past}]$ denote the optimal prediction of $Y_{t+1}$ given information from the past. 
* Then $X \to Y$ if:

$$
\mathcal{E}[Y_{t+1} \mid Y_{t-\text{past}}, X_{t-\text{past}}, Z_{t-\text{past}}] \neq \mathcal{E}[Y_{t+1} \mid Y_{t-\text{past}}, Z_{t-\text{past}}]
$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Granger Causality for VAR Models)</span></p>

* To make this concept testable, Granger proposed embedding it within a VAR model. 
* Consider a **bivariate system** $(X_t, Y_t)$. 
* To test if $X \to Y$, we set up **two nested models**:

1. **Full Model:** A VAR($p$) model where past values of $X$ are used to predict $Y$. In the equation for $Y_t$, the coefficients on lagged $X_t$ are unrestricted.
   * $Y_t = a_0 + \sum_{i=1}^{p} A_i Y_{t-i} + \sum_{j=1}^{q} B_j X_{t-j} + \sum_{l=1}^{m} C_l Z_{t-l} + \varepsilon_t$
2. **Restricted Model:** A VAR($p$) model where the influence of past $X$ on $Y$ is removed. This is achieved by setting all coefficients that link lagged $X_t$ to $Y_t$ to zero.
   * $Y_t = a_0 + \sum_{i=1}^{p} A_i Y_{t-i} + \sum_{l=1}^{m} C_l Z_{t-l} + \varepsilon_t$

We then perform a likelihood ratio test (or an F-test) comparing these two models.

- Let $\hat{\Sigma}\_{\text{full}}$ and $\hat{\Sigma}\_{\text{restr}}$ be the estimated residual covariance matrices.
- The LR test statistic is: 
  - $D = T_{\text{eff}} (\log(\lvert\hat{\Sigma}\_{\text{restr}}\rvert) - \log(\lvert\hat{\Sigma}\_{\text{full}}\rvert))$
  - $T_{\text{eff}} = T - \max(p, q, m) - 1$
- Under $H_0$ (no Granger causality), $D \sim \chi^2(r)$, where $r = q \cdot (\text{dim of } X) \cdot (\text{dim of } Y)$ is the number of zero-restrictions imposed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation)</span></p>

If adding the past of $X$ significantly reduces the residual covariance (i.e., the prediction error) for $Y$, then the test statistic will be large, leading to a rejection of the null hypothesis. We conclude that $X$ Granger-causes $Y$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Caveats in Interpretation)</span></p>

It is crucial to interpret "Granger causality" with care, as it is a statement about predictive power, not necessarily true causal influence.

- **Hidden Common Causes:** If an unobserved variable $Z$ drives both $X$ and $Y$, a spurious Granger-causal relationship $X \to Y$ might be detected.
- **Linearity Assumption:** The standard test is based on linear VAR models and only detects linear forms of dependence. The general definition of Granger causality is not restricted to linear relationships.
- **Gaussian Assumptions:** The statistical properties of the LR test rely on the assumption of Gaussian-distributed residuals. Strong deviations from this assumption may invalidate the test statistics.

</div>
