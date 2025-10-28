---
layout: default
title: Model-Based Time Series Analysis
date: 2024-11-01
excerpt: A survey of state-space models, Kalman filtering, and spectral tools for forecasting structured time-evolving data.
tags:
  - time-series
  - statistics
  - modeling
# math: true
---

# Model-Based Time Series Analysis

## 1. Review on Statistical Inference

To reason about uncertainty in a mathematically sound way, we begin with the concept of a probability space. This structure consists of three essential components that formalize an experiment and its outcomes.

### 1.1 The Probability Space

An **experiment** is a process that yields an observation. The mathematical abstraction for this is the **probability space**, a tuple $(\Omega, \mathcal{A}, P)$. 

* **Sample Space $\Omega$:** The set of all possible indecomposable outcomes of an experiment. These are called simple events.
  * *Example (Die Toss):* $\Omega = \lbrace 1, 2, 3, 4, 5, 6 \rbrace$
* **$\sigma$-algebra $(\mathcal{A})$:** A collection of subsets of $\Omega$ (called events) about which we can ask probabilistic questions. A valid $\sigma$-algebra must satisfy three properties:
  1. It contains the sample space: $\Omega \in \mathcal{A}$.
  2. It is closed under complementation: If $A \in \mathcal{A}$, then its complement $A^c$ is also in $\mathcal{A}$.
  3. It is closed under countable unions: If $A_1, A_2, \dots \in \mathcal{A}$, then their union $\bigcup_{i=1}^{\infty} A_i$ is also in $\mathcal{A}$.
  * For a finite sample space $\Omega$, the $\sigma$-algebra is often the power set $\mathcal{P}(\Omega)$, which is the set of all possible subsets of $\Omega$.
* **Probability Measure $P$:** A function $P: \mathcal{A} \to [0, 1]$ that assigns a likelihood to each event in the $\sigma$-algebra. It must satisfy:
  1. $P(\emptyset) = 0$.
  2. $P(\Omega) = 1$.
  3. For any sequence of pairwise disjoint events $A_1, A_2, \dots$, the probability of their union is the sum of their probabilities:
     $P\!\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i).$

The pair $(\Omega, \mathcal{A})$ is called a **measurable space**. Elements of $\mathcal{A}$ are **measurable sets**.
* *Example (Die Toss):*
  * $\Omega = \lbrace 1, 2, 3, 4, 5, 6 \rbrace$.
  * $\mathcal{A} = \mathcal{P}(\Omega)$ or $\mathcal{A} = \lbrace \emptyset, \lbrace 1,3,5 \rbrace, \lbrace 2,4,5 \rbrace, \Omega \rbrace$.

### 1.2. Random Variables and Probability Distributions

Random variables allow us to map outcomes from the sample space to the real numbers, enabling the use of powerful mathematical tools.

<!-- ### 2.1 Random Variables -->

**Definition (Random Variable):** A Random Variable (RV) is a measurable function $X: \Omega \to \mathbb{R}$ that assigns a real number to each outcome in the sample space.

* *Example (Coin Toss):* For a coin toss where $\Omega = \lbrace \text{Heads}, \text{Tails} \rbrace$, we can define an RV $X$ such that $X(\text{Heads}) = 1$ and $X(\text{Tails}) = 0$.

The mapping itself is deterministic; the randomness is induced by the underlying probability measure on $\Omega$. Random variables induce probability distributions.

<!-- ### 2.2 Cumulative Distribution Function (CDF) -->

The **Cumulative Distribution Function (CDF)** uniquely characterizes the distribution of any random variable.

**Definition (CDF):** A function $F: \mathbb{R} \to [0, 1]$ is a CDF if it satisfies:

1. $0 \le F(x) \le 1$ for all $x \in \mathbb{R}$.
2. $F$ is non-decreasing.
3. $F$ is right-continuous.
4. $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$.

The CDF connects directly to the probability measure: $F(x) = P(X \le x)$.

### 1.3 Discrete Distributions and PMFs

For a discrete RV, the probability is concentrated on a countable set of points.

**Definition (Probability Mass Function — PMF):** Let $a_1, a_2, \dots$ be the values a discrete RV $X$ can take. A function $f: \mathbb{R} \to [0, 1]$ is a PMF if $f(x) = P(X=x)$. Specifically, if $p_k = P(X=a_k)$ with $\sum_k p_k = 1$, then
$$
f(x) =
\begin{cases}
  p_k & \text{if } x = a_k,\\
  0   & \text{otherwise}.
\end{cases}
$$
The CDF for a discrete RV is a step function: $F(t) = \sum_{k: a_k \le t} p_k$.

Common Discrete Distributions:

| Distribution | Parameters | PMF $P(X=x)$ | Description |
| --- | --- | --- | --- |
| Bernoulli | $p \in [0, 1]$ | $p^x (1-p)^{1-x}$ for $x \in \{0, 1\}$ | Models a single trial with two outcomes (e.g., success/failure). |
| Binomial | $n \in \mathbb{N},\ p \in [0, 1]$ | $\binom{n}{x} p^x (1-p)^{n-x}$ for $x \in \{0, \dots, n\}$ | Models the number of successes in $n$ independent Bernoulli trials. |
| Poisson | $\lambda > 0$ | $\dfrac{\lambda^x e^{-\lambda}}{x!}$ for $x \in \{0, 1, 2, \dots\}$ | Models the number of events occurring in a fixed interval of time or space. |

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/pmf_bernoulli.png' | relative_url }}" alt="Bernoulli PMF" loading="lazy">
    <figcaption>Bernoulli PMF</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/pmf_binomial.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <figcaption>Binomial PMF</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/pmf_poisson.png' | relative_url }}" alt="Poisson PMF" loading="lazy">
    <figcaption>Poisson PMF</figcaption>
  </figure>
</div>

### 1.4 Continuous Distributions and PDFs

For a continuous RV, the probability of any single point is zero. Probability is defined over intervals.

**Definition (Probability Density Function — PDF):** A function $f: \mathbb{R} \to [0, \infty)$ is a PDF if:

1. $f(x) \ge 0$ for all $x \in \mathbb{R}$ (non-negativity).
2. $\int_{-\infty}^{\infty} f(x)\,dx = 1$.

The probability over an interval is given by the integral of the PDF: $P(a \le X \le b) = \int_a^b f(x)\,dx$. The corresponding CDF is $F(t) = \int_{-\infty}^{t} f(x)\,dx$.

Common Continuous Distributions:

| Distribution | Parameters | PDF $f(x)$ |
| --- | --- | --- |
| Normal | $\mu \in \mathbb{R},\ \sigma^2 > 0$ | $\dfrac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)$ |
| Multivariate Normal | $\mu \in \mathbb{R}^d,\ \Sigma \in \mathbb{R}^{d \times d}$ | $\dfrac{1}{(2\pi)^{d/2} \sqrt{\det \Sigma}} \exp\!\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)$ |
| Exponential | $\lambda > 0$ | $\lambda e^{-\lambda x}$ for $x \ge 0$ |
| Uniform | $a, b \in \mathbb{R},\ a < b$ | $\dfrac{1}{b-a}$ for $x \in [a, b]$ |

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/pdf_normal.png' | relative_url }}" alt="Normal PDF" loading="lazy">
    <figcaption>Normal PDF</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/pdf_exp.png' | relative_url }}" alt="Exponential PDF" loading="lazy">
    <figcaption>Exponential PDF</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/pdf_uniform.png' | relative_url }}" alt="Uniform PDF" loading="lazy">
    <figcaption>Uniform PDF</figcaption>
  </figure>
</div>

### 1.5 Properties of Random Variables

**Expected Value** (Mean):

* Continuous: $\mathbb{E}[X] = \int_{-\infty}^{\infty} x f(x)\,dx$
* Discrete: $\mathbb{E}[X] = \sum_{i} x_i P(X=x_i)$

**Moments** and **Variance**:

* The $k$-th moment of an RV $X$ is $\mathbb{E}[X^k]$. The expected value is the first moment.
* The variance measures the spread of the distribution:
  $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2.$


### 1.6 Rules and Laws of Probability

This section covers essential rules for manipulating probabilities and two cornerstone theorems of probability theory.

#### 1.6.1 Rules of Probability

Let $X$ and $Y$ be random variables with realizations $x$ and $y$.

* **Conditional Probability:** The probability of $x$ occurring given that $y$ has occurred.
  $P(x \mid y) = \frac{P(x, y)}{P(y)}.$
* **Chain Rule:** Decomposes a joint probability into a product of conditional probabilities.
  $P(x_1, x_2, \dots, x_n)
  = P(x_1) P(x_2 \mid x_1) P(x_3 \mid x_1, x_2) \dots P(x_n \mid x_1, \dots, x_{n-1}).$
* **Marginalization:** The probability of one variable can be found by summing (or integrating) over all possible values of other variables. 
  $P(x) = \sum_y P(x, y) = \sum_y P(x \mid y) P(y).$
* **Bayes' Rule:** Relates a conditional probability to its inverse and underpins Bayesian inference.
  $P(y \mid x) = \frac{P(x \mid y) P(y)}{P(x)}.$
* **Independence:** Two RVs $X$ and $Y$ are independent if learning the value of one provides no information about the other. This holds iff
  $P(x, y) = P(x) P(y)
  \quad \text{or equivalently} \quad
  P(x \mid y) = P(x), \quad
  P(y \mid x) = P(y).$

#### 1.6.2 Asymptotic Theorems

These theorems describe the behavior of the sum of a large number of random variables.

<!-- **Theorem (Strong Law of Large Numbers — SLLN):** Let $(X_i)_{i=1}^\infty$ be a sequence of independent and identically distributed (i.i.d.) random variables with finite mean $E[X_i] = \mu$. Then the sample mean converges almost surely to the true mean:
$\frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{\text{a.s.}} \mu
\quad \text{as } n \to \infty.$ -->

**Theorem (Strong Law of Large Numbers — SLLN):**  
Let $(X_i)_{i=1}^\infty$ be a sequence of independent and identically distributed (i.i.d.) random variables with finite mean $\mathbb{E}[X_i] = \mu$. Then the sample mean converges almost surely to the true mean:

$$
\frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{\text{a.s.}} \mu
\quad \text{as } n \to \infty.
$$


**Theorem (Central Limit Theorem — CLT):** 
Let $(X_i)_{i=1}^\infty$ be a sequence of i.i.d. random variables with mean $\mu$ and finite variance $\sigma^2$. Then the distribution of the standardized sample mean converges to a standard Normal distribution:

$$
\frac{\sqrt{n}\left(\frac{1}{n}\sum_{i=1}^n X_i - \mu\right)}{\sigma}
\xrightarrow{d} \mathcal{N}(0, 1)
\quad \text{as } n \to \infty.
$$

---

## 2. Review on Parameter Estimation

### 2.1 Statistical Inference

Statistical inference aims to deduce properties of an underlying population or data-generating process from a finite sample of data. To do so, we often assume that our data is drawn from a family of distributions parametrized by a finite set of parameters.

**Definition (Parametric Model):** Let $\Theta \subseteq \mathbb{R}^n$ is a parameter space.. A family of probability distributions $\mathcal{P}_{\Theta} = \lbrace p_{\theta} \mid \theta \in \Theta \rbrace$ on a measurable space is called a parametric model.

Our goal is to estimate the true, unknown parameter $\theta$ that generated the data. We distinguish between:

* **Parameter $\theta$:** The true, fixed (in the frequentist view) but unknown value.
* **Estimator $\hat{\theta}$:** A function of our data that is used to estimate $\theta$. It is a random variable.
* **Estimate**: A specific numerical value of the estimator realized from a specific data sample.

If I recompute samples and each time evaluate my estimate, I obtain a distribution over these estimates. This is called the **sampling distribution** of my estimator. The standard deviation of sampling distribution is the **standard error**.

### 2.2 Properties of Estimators

* **Bias:** The difference between the expected value of the estimator and the true parameter:
  $\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta.$
  An estimator is unbiased if its bias is zero.
* **Efficiency:** An estimator is more efficient than another if it has a smaller variance (i.e., a smaller standard error).

**Definition (Frequentist vs. Bayesian Viewpoints):**
* **Frequentist View:** Parameters $\theta$ are fixed, unknown constants. Randomness arises solely from the data sampling process.
* **Bayesian View:** Parameters are random variables themselves, possessing their own probability distributions $p(\theta)$. Data is used to update our beliefs about the parameters.

### 2.3. Paradigms of Parameter Estimation

There are three primary approaches to estimating parameters from data.

#### 2.3.1 Method of Least Squares (LS)

The LS method estimates parameters by minimizing the sum of the squared differences between observed values and the values predicted by the model. It requires no assumptions about the underlying distribution of the data.

Let $(x_i, y_i)$ be $N$ data pairs. We approximate the true relationship $y=f(x)$
with a parameterized function $f_\theta(x)$. The residual for the $i$-th data point is
$\epsilon_i = y_i - f_\theta(x_i)$. The LS estimator is

\[
\hat{\theta}_{\mathrm{LS}} = \arg\min_{\theta} \sum_{i=1}^N \epsilon_i^2
= \arg\min_{\theta} \sum_{i=1}^N \bigl(y_i - f_\theta(x_i)\bigr)^2 .
\]


* *Example: Estimating the Population Mean*
  * **Model:** Assume data points $X_1, \dots, X_N$ are generated by $X_i = \mu + \epsilon_i$, where $\mu$ is the true mean. We want to estimate $\mu$.
  * **Cost Function:** $S(\mu) = \sum_{i=1}^N (X_i - \mu)^2$.
  * **LS Estimation:** $\hat{\mu} = \arg\min_{\mu} \underbrace{\sum_{i=1}^N (X_i - \mu)^2}_{\text{Err}(\mu)}$.
  * **Derivation:** Differentiate with respect to $\mu$ and set to zero.
    $\frac{\partial S}{\partial \mu} = \sum_{i=1}^N -2(X_i - \mu) = -2 \left( \sum_i X_i - \mu \right) = 0$
    $\sum_i X_i - N\mu = 0 \implies \hat{\mu} = \frac{1}{N} \sum_{i=1}^N X_i.$
  * The LS estimate for the population mean is the sample mean.

#### 2.3.2 Maximum Likelihood Estimation (MLE)

MLE selects the parameter values that make the observed data most probable under the assumed parametric model.

Let $X = \lbrace x_1, \dots, x_N \rbrace$ be observed data by assumption drawn from a model with density $p(x \mid \theta)$ ($X \sim P_(\Theta)^{(N)}$).

* **Likelihood Function:** The joint density of the observed data, viewed as a function of the parameter $\theta$:
  $\mathcal{L}(\theta \mid X) = p(x_1, \dots, x_N \mid \theta).$
* If the data are i.i.d., the likelihood factorizes:
  $\mathcal{L}(\theta \mid X) = \prod_{i=1}^N p(x_i \mid \theta).$
* **MLE Definition:** The MLE is the value of $\theta$ that maximizes the likelihood function:
  $\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} \mathcal{L}(\theta \mid X)$.
  In practice, it is often easier to maximize the log-likelihood $\ell(\theta \mid X) = \log \mathcal{L}(\theta \mid X)$, since the logarithm is monotonic and the resulting sums are easier to differentiate.
* *Example: Estimating the Mean of a Normal Distribution*
  * **Model:** $X_i \sim \mathcal{N}(\mu, \sigma^2)$ i.i.d., with $\sigma^2$ known.
  * **Log-Likelihood:**
    $\ell(\mu \mid X) = \log \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(X_i-\mu)^2}{2\sigma^2}\right) = \sum_{i=1}^N \left( -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(X_i-\mu)^2}{2\sigma^2} \right).$
  * **Maximization:** Differentiate with respect to $\mu$ and set to zero.
    $\frac{\partial \ell}{\partial \mu} = \sum_{i=1}^N \frac{X_i-\mu}{\sigma^2} = \frac{1}{\sigma^2} \sum_{i=1}^N (X_i-\mu) = 0$
    $\sum_i X_i - N\mu = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N X_i.$
  * For this model, the MLE and LS estimates for the mean coincide.

#### 2.3.3 Bayesian Inference (BI)

Bayesian inference uses Bayes' rule to update knowledge about parameters after observing data. It produces a posterior distribution for the parameters, not just a point estimate.

The core of Bayesian inference is Bayes' rule:
$p(\theta \mid X) = \frac{p(X \mid \theta) p(\theta)}{p(X)}.$
Here:

* $p(\theta \mid X)$ is the **posterior** distribution: our belief about $\theta$ after seeing data $X$.
* $p(X \mid \theta)$ is the **likelihood**, as in MLE.
* $p(\theta)$ is the **prior** distribution: our belief about $\theta$ before seeing any data.
* $p(X) = \int p(X \mid \theta) p(\theta)\,d\theta$ is the **evidence** or marginal likelihood of the data.

This is often summarized as: Posterior $\propto$ Likelihood $\times$ Prior.

* *Example: Inferring the mean $\mu$ of a Normal distribution ($\sigma^2$ known) with a Normal prior on $\mu$.*
  * Likelihood: $p(X \mid \mu) \propto \exp\left(-\frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2\right)$.
  * Prior: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$, so $p(\mu) \propto \exp\left(-\frac{(\mu - \mu_0)^2}{2\sigma_0^2}\right)$.
  * The posterior $p(\mu \mid X)$ is also a Normal distribution: 
    * $p(\mu \mid X) = p(X\mid \mu)p(\mu) / p(x) = \mathcal{N}(\mu, \sigma^2)\mathcal{N}(\mu_0, \sigma_0^2) / p(x)$.
  * A common challenge is that the evidence $p(X)$ is often an intractable integral, requiring numerical methods such as MCMC or variational inference.

### 2.4. Parameter Estimation for Intractable Problems

When closed-form solutions for estimators are not available, we turn to numerical optimization algorithms to find the parameters that minimize a cost function (e.g., sum of squared errors, negative log-likelihood).

#### 2.4.1 Gradient Descent (GD)

Gradient Descent is an iterative first-order optimization algorithm for finding a local minimum of a differentiable function. The core idea is to take repeated steps in the opposite direction of the gradient of the function at the current point, as this is the direction of steepest descent.

**Algorithm:**

1. Initialize parameter guess $\theta_0$.
2. Repeat for $n$ iterations:
   * $\theta_{n+1} = \theta_n - \gamma \nabla J(\theta_n)$, 
   * where $J(\theta)$ is the cost function and $\gamma > 0$ is the learning rate.
1. Stop when convergence is reached (e.g., $\|J(\theta_{i}) - J(\theta_{i+1})\| < \epsilon$) or no more iterations.

* *Example: GD for LS Estimation of the Mean*
  * Model: $X_i = \mu + \epsilon_i$
  * Cost: $J(\mu) = \frac{1}{2} \sum_i (X_i - \mu)^2$
  * Gradient: $\nabla J(\mu) = \frac{\partial J}{\partial \mu} = -\sum_i (X_i - \mu)$
  * Update Rule: $\mu_{n+1} = \mu_n - \gamma (-\sum_i (X_i - \mu_n)) = \mu_n + \gamma \sum_i (X_i - \mu_n)$

**Challenges and Solutions:**
* Local minima.
* Slow convergence in flat regions of the cost landscape.
* Overshooting the minimum if the learning rate $\gamma$ is too large.
* Practical remedies:
  * **Random Restarts:** Run the algorithm from multiple random initial conditions to increase the chance of finding a global minimum.
  * **Stochastic Gradient Descent (SGD):** Compute the gradient on a small random subset of data (a minibatch) at each step, leading to faster but noisier updates.
  * **Adaptive Learning Rates:** Algorithms like Momentum, Adagrad, and Adam dynamically adjust the learning rate to navigate varying curvatures in the cost landscape.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/gd_localMinima.png' | relative_url }}" alt="Gradient descent stuck in local minima" loading="lazy">
    <figcaption>Gradient descent trapped in local minima and saddle regions.</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/gd_slowConvergence.png' | relative_url }}" alt="Gradient descent slow convergence" loading="lazy">
    <figcaption>Slow convergence in flat valleys or overshooting due to poorly scaled gradients.</figcaption>
  </figure>
</div>

#### 2.4.2 Newton-Raphson Method

The Newton-Raphson method is a second-order optimization algorithm that uses the curvature of the loss landscape to take more informed steps. It adapts the learning rate by incorporating the Hessian matrix (the matrix of second partial derivatives).

**Update Rule:**
$\theta_{n+1} = \theta_n - H_{f(\theta_n)}^{-1} \nabla f(\theta_n),$
where $H_{f(\theta_n)}$ is the Hessian matrix of the function $f$ evaluated at $\theta_n$. The term $H^{-1}$ acts as an adaptive, matrix-valued learning rate. While Newton-Raphson can converge much faster than GD, computing and inverting the Hessian is computationally expensive for the high-dimensional parameter spaces common in deep learning.
