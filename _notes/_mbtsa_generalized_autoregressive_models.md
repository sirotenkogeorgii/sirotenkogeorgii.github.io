## Generalized Autoregressive Models

This chapter extends the autoregressive framework to model non-Gaussian time series, such as binary sequences or count data, using the principles of Generalized Linear Models (GLMs).

### AR Models for Binary Processes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(AR Models for Binary Processes)</span></p>

Consider a time series of binary outcomes, $X_t \in \lbrace 0, 1\rbrace$. We can model this using an autoregressive structure similar to logistic regression.

- **Data:** A binary time series, e.g., $\lbrace X_t\rbrace_{t=1}^T$.
- **Model Architecture:** The probability of a "success" $(X_t=1)$ is conditioned on the past $p$ values.
  
  $$
  X_t \mid X_{t-1}, \dots, X_{t-p} \sim \text{Bernoulli}(\pi_t)
  $$

  where $\pi_t = P(X_t = 1 \mid \text{past})$.
- **Linear Predictor:** The probability $\pi_t$ is related to a linear combination of past observations through a link function (typically the logit function). The linear predictor $\eta_t$ is defined as:
  
  $$
  \eta_t = c + \sum_{i=1}^{p} \alpha_i X_{t-i}
  $$
  
- **Link Function:** The relationship between $\eta_t$ and $\pi_t$ is given by the inverse link function (sigmoid or logistic function):
  
  $$
  \pi_t = \frac{e^{\eta_t}}{1 + e^{\eta_t}} = \frac{1}{1 + e^{-\eta_t}}
  $$

  This is analogous to logistic regression, but the predictors are now the lagged values of the time series itself.

- **Training:** Model parameters are estimated by maximizing the likelihood, which does not have a closed-form solution. Numerical optimization methods like Gradient Descent or Newton-Raphson are required.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Main Limitations of Binary AR Models)</span></p>

- **Unstable Estimates:** For strongly dependent series, certain histories (e.g., a long string of 1s) may lead to perfect prediction, causing the MLE estimates for coefficients to become very large or non-existent.
- **Limited Dynamical Structure:** These models may struggle to capture complex temporal patterns like "bursts" of activity versus periods of "silence".
- **Scalability:** Extending this framework to the multivariate case is challenging.
- **Numerical Inference:** Inference about the model parameters is purely numerical.

A common remedy for instability is to reduce the model order $p$, thereby reducing complexity.

</div>

### Change Point Models for Binary Time Series

An alternative approach for binary series is to model the success probability $\pi_t$ as an explicit function of time, allowing for a single change point.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Change Point Model)</span></p>

- **Problem:** We suspect the underlying success probability $\pi_t$ is not constant over time and want to infer the location of a change.
- **Goal:** Infer the change point location $\tau$ from the observed data $X_1, \dots, X_T$.
- **Model:** We model the time-varying probability $\pi_t$ using a sigmoid function of time:
  
  $$
  \pi_t = f(t, \theta) = m + \frac{d}{1 + e^{-a(t-c)}}
  $$

- **The parameters $\theta = \lbrace m, d, a, c\rbrace$ have clear interpretations:**
  - $m$: Baseline probability before the change $(0 \le m \le 1)$.
  - $d$: Amplitude of the change $(0 \le d \le 1-m)$. The success probability after the change is $m+d$.
  - $a$: Inverse slope of the change $(a \ge 0)$. Smaller values of $a$ correspond to a steeper, more abrupt change.
  - $c$: Change point location in time $(0 \le c \le T-1)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Parameter Estimation for Change Point Model)</span></p>

**Likelihood:** The likelihood of the observed data given the parameters is the product of Bernoulli probabilities:
  
  $$P(X \mid \theta) = \prod_{t=1}^{T} \pi_t^{X_t} (1-\pi_t)^{1-X_t}$$

  The log-likelihood is 
  
  $$\ell(\theta) = \sum_{t=1}^{T} [X_t \log(\pi_t) + (1-X_t)\log(1-\pi_t)]$$
  
  These parameters are typically found via MLE.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Challenge of such a change point model)</span></p>

This specific sigmoid structure only allows for a single, monotonic change point in the success probability.

</div>

### AR Models for Count Processes (Poisson GLM)

For time series of counts, $C_t \in \lbrace 0, 1, 2, \dots\rbrace$, we can use a Poisson distribution where the rate is modeled with an autoregressive structure.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(AR Models for Count Processes (Poisson GLM))</span></p>

- **Data:** A univariate or multivariate count process. For example, the number of customers entering a store per hour, or the number of spikes from $N$ neurons in discrete time bins. 
  
  $$C_t = (C_{1t}, \dots, C_{Nt})^\top$$

- **Model:** For each process $i=1, \dots, N$, we model the count $C_{it}$ conditioned on the past as a Poisson random variable.
  
  $$
  C_{it} \mid \text{past} \sim \text{Poisson}(\lambda_{it})
  $$

  The Poisson PMF implies that 
  
  $$\mathbb{E}[C_{it} \mid \text{past}] = \text{Var}(C_{it} \mid \text{past}) = \lambda_{it}$$ 
  
  The rate $\lambda_{it}$ determines both the mean and variance.
- **Rate Model:** We model the vector of rates $\lambda_t = (\lambda_{1t}, \dots, \lambda_{Nt})^\top$ using an AR structure with an exponential link function to ensure positivity of the rates.
  
  $$
  \lambda_t = \exp\left(c + \sum_{j=1}^{p} A_j C_{t-j}\right)
  $$

  The $\exp$ is the inverse link function in this Poisson GLM. The coefficient matrices $A_j$ capture the influence of past counts on current rates, representing "effective couplings" between processes.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Parameter Estimation for Poisson GLM)</span></p>

  **Maximum Likelihood Estimation:** The log-likelihood, assuming conditional independence of the individual processes given the past, is:
  
  $$\ell(\lbrace c, A_j\rbrace) = \sum_{t=p+1}^{T} \sum_{i=1}^{N} \log P(C_{it} \mid \text{past})$$

  Substituting the Poisson PMF, 
  
  $$\log P(C_{it}) = C_{it} \log(\lambda_{it}) - \lambda_{it} - \log(C_{it}!)$$ 
  
  and our model for $\lambda_{it}$:
  
  $$\ell(\lbrace c, A_j\rbrace) = \sum_{t, i} \Bigl[ C_{it} \Bigl(c_i + \sum_{j,k} (A_j)_{ik} C_{k,t-j}\Bigr) - \exp\Bigl(c_i + \sum_{j,k} (A_j)_{ik} C_{k,t-j}\Bigr) \Bigr] + \text{const}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(No closed-form solution for the parameters of Poisson GLM)</span></p>

* Although there is **no closed-form solution for the parameters of Poisson GLM**, the **Poisson log-likelihood function is concave**. 
* This is a **significant advantage**, as it guarantees that standard numerical optimization methods (like Gradient Descent) will converge to the **unique global maximum**.

</div>
