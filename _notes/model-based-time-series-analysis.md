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

**Definition (Parametric Model):** Let $\Theta \subseteq \mathbb{R}^n$ is a parameter space. A family of probability distributions $P(\Theta) = \lbrace p_{\theta} \mid \theta \in \Theta \rbrace $ on a measurable space is called a parametric model.

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

$$
\hat{\theta}_{\mathrm{LS}} = \arg\min_{\theta} \sum_{i=1}^N \epsilon_i^2
= \arg\min_{\theta} \sum_{i=1}^N \bigl(y_i - f_\theta(x_i)\bigr)^2 .
$$


* *Example: Estimating the Population Mean*
  * **Model:** Assume data points $X_1, \dots, X_N$ are generated by $X_i = \mu + \epsilon_i$, where $\mu$ is the true mean. We want to estimate $\mu$.
  * **Cost Function:** $S(\mu) = \sum_{i=1}^N (X_i - \mu)^2$.
  * **LS Estimation:** $\hat{\mu} = \arg\min_{\mu} \underbrace{\sum_{i=1}^N (X_i - \mu)^2}_{\text{Err}(\mu)}$.
  * **Derivation:** Differentiate with respect to $\mu$ and set to zero.
    * $\frac{\partial S}{\partial \mu} = \sum_{i=1}^N -2(X_i - \mu) = -2 \left( \sum_i X_i - \mu \right) = 0$
    * $\sum_i X_i - N\mu = 0 \implies \hat{\mu} = \frac{1}{N} \sum_{i=1}^N X_i.$
  * The LS estimate for the population mean is the sample mean.

#### 2.3.2 Maximum Likelihood Estimation (MLE)

MLE selects the parameter values that make the observed data most probable under the assumed parametric model.

Let $X = \lbrace x_1, \dots, x_N \rbrace$ be observed data by assumption drawn from a model with density $p(x \mid \theta)$ ($X \sim P(\Theta)^{(N)}$).

* **Likelihood Function:** The joint density of the observed data, viewed as a function of the parameter $\theta$:
  $\mathcal{L}(\theta \mid X) = p(x_1, \dots, x_N \mid \theta).$
* If the data are i.i.d., the likelihood factorizes:
  $\mathcal{L}(\theta \mid X) = \prod_{i=1}^N p(x_i \mid \theta).$
* **MLE Definition:** The MLE is the value of $\theta$ that maximizes the likelihood function:
  * $$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} \mathcal{L}(\theta \mid X)$$

In practice, it is often easier to maximize the log-likelihood $\ell(\theta \mid X) = \log \mathcal{L}(\theta \mid X)$, since the logarithm is monotonic and the resulting sums are easier to differentiate.

* *Example: Estimating the Mean of a Normal Distribution*
  * **Model:** $x_i = \mu + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ or equivalently $X_i \sim \mathcal{N}(\mu, \sigma^2)$ i.i.d., with $\sigma^2$ known.
  * **Log-Likelihood:**
    $\ell(\mu \mid X) = \log \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(X_i-\mu)^2}{2\sigma^2}\right) = \sum_{i=1}^N \left( -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(X_i-\mu)^2}{2\sigma^2} \right).$
  * **Maximization:** Differentiate with respect to $\mu$ and set to zero.
    $$\frac{\partial \ell}{\partial \mu} = \sum_{i=1}^N \frac{X_i-\mu}{\sigma^2} = \frac{1}{\sigma^2} \sum_{i=1}^N (X_i-\mu) = 0$$
    $$\sum_i X_i - N\mu = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N X_i$$
  
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

The Newton-Raphson method is an iterative algorithm for finding the roots of a function $f(x)$, i.e., finding $x$ such that $f(x)=0$. In the context of model optimization, the goal is often to find the roots of the *derivative* of the loss function, which correspond to potential minima or maxima. This method is a second-order optimization algorithm that uses the curvature of the loss landscape to take more informed steps. It adapts the learning rate by incorporating the Hessian matrix (the matrix of second partial derivatives).

**Update Rule:**
$\theta_{n+1} = \theta_n - H_{f(\theta_n)}^{-1} \nabla f(\theta_n),$
where $H_{f(\theta_n)}$ is the Hessian matrix of the function $f$ evaluated at $\theta_n$. The term $H^{-1}$ acts as an adaptive, matrix-valued learning rate. While Newton-Raphson can converge much faster than GD, computing and inverting the Hessian is computationally expensive for the high-dimensional parameter spaces common in deep learning.

The update rule is derived from the first-order Taylor expansion of the function $f$ around an initial guess $\theta_0$:
1. $f(\theta_1) \approx f(\theta_0) + (\theta_1 - \theta_0)f'(\theta_0)$
2.  To find the root, we set $f(\theta_1) = 0$:
    * $0 = f(\theta_0) + (\theta_1 - \theta_0)f'(\theta_0)$
3. Rearranging the terms to solve for the next guess, $\theta_1$:
    * $(\theta_1 - \theta_0)f'(\theta_0) = -f(\theta_0)$
    * $\theta_1 f'(\theta_0) - \theta_0 f'(\theta_0) = -f(\theta_0)$
    * $\theta_1 f'(\theta_0) = \theta_0 f'(\theta_0) - f(\theta_0)$
    * $\theta_1 = \theta_0 - \frac{f(\theta_0)}{f'(\theta_0)}$

The general iterative update equation is:

$$
\theta_n = \theta_{n-1} - [f'(\theta_{n-1})]^{-1} f(\theta_{n-1})
$$

This can be viewed as an analogue to Gradient Descent (GD), but with an adaptive learning rate determined by the inverse of the derivative. For a multivariate function $f$, the first derivative $f'$ becomes the gradient, and the second derivative (used to find the root of the first derivative) is the matrix of second partial derivatives known as the **Hessian**.

**Remarks:**
* While Gradient Descent searches linearly, the Newton-Raphson method, using the Hessian, searches quadratically. This requires the computation of second derivatives and matrix inversions.
* The method converges more quickly than Gradient Descent.
* Newton-Raphson will find both minima and maxima, as it seeks any point where the gradient is zero.

#### 2.4.3 Parameter Estimation for Intractable Bayesian Problems

In Bayesian inference, we are interested in the posterior distribution of parameters $\theta$ given data $X$, which is given by Bayes' theorem:
$p(\theta \mid X) \propto p(X \mid \theta) p(\theta)$. A significant problem arises when the normalizing constant, or model evidence, $p(X) = \int p(X \mid \theta)p(\theta) d\theta$, is intractable to compute. This intractability gives rise to a class of powerful simulation-based methods known as **Markov Chain Monte Carlo (MCMC)**.

**Core Concepts of MCMC**

* **Monte Carlo Integration:** This principle states that we can estimate expectations by sampling. For a function $h(\theta)$, its expectation with respect to a probability distribution $p(\theta \mid X)$ is:

    $$
    \mathbb{E}[h(\theta) \mid X] = \int p(\theta \mid X) h(\theta) d\theta
    $$

    By drawing $N$ samples $\theta^{(i)}$ from the posterior $p(\theta \mid X)$, we can approximate this expectation using the law of large numbers:
    
    $$
    \mathbb{E}[h(\theta) \mid X] \approx \frac{1}{N} \sum_{i=1}^{N} h(\theta^{(i)})
    $$

* **Markov Chain:** This is a method to generate samples sequentially, where the next state depends only on the current state (a memoryless transition). The transition probability is defined as $p(\theta_t \mid \theta_{t-1}, ..., \theta_0) = p(\theta_t \mid \theta_{t-1})$.

**Posterior Sampling with Metropolis-Hastings**

We can evaluate $p(⋅)$ for individual points but don’t know its normalization constant. Since we only care about the shape of the posterior, MCMC methods let us bypass computing the normalization constant and still approximate the posterior.

The general idea is to generate a sequence of parameter samples, $\theta^{(0)}, \theta^{(1)}, ..., \theta^{(N)}$, that form a Markov chain whose stationary distribution is the target posterior distribution $p(\theta \mid X)$. We can work with the unnormalized posterior density, as the algorithm only depends on the ratio of densities, where the normalizing constant cancels out.
$\text{Posterior Density} \propto p(X \mid \theta)p(\theta)$

**Metropolis-Hastings Algorithm with a Symmetric Proposal:**

1.  **Initialization:** Choose an initial parameter value $\theta^{(0)}$ and specify the number of samples $N$.
2.  **Iteration:** Loop for $i = 1, ..., N$:
    * **Propose:** Generate a new candidate sample $\theta_{prop}$ from a symmetric proposal distribution $q(\cdot \mid \theta^{(i-1)})$. A common choice is a normal distribution centered at the current sample: $\theta_{prop} \sim \mathcal{N}(\theta^{(i-1)}, \sigma^2 I)$.
    * **Compute Acceptance Ratio:** Calculate the ratio of the posterior densities at the proposed and current points. This is typically done in log-space for numerical stability.
        * $r_{prop} := p(X \mid \theta_{prop})p(\theta_{prop})$
        * $r_{curr} := p(X \mid \theta^{(i-1)})p(\theta^{(i-1)})$
        * The acceptance ratio is $r = \frac{r_{prop}}{r_{curr}}$.
    * **Accept or Reject:** Draw a random number $u$ from a uniform distribution, $u \sim \text{Unif}(0, 1)$.
        * If $u < \min(1, r)$, accept the proposal: $\theta^{(i)} = \theta_{prop}$.
        * Else, reject the proposal and stay at the current state: $\theta^{(i)} = \theta^{(i-1)}$.

Under mild conditions, the resulting sequence of samples $\{\theta^{(i)}\}_{i=1}^N$ will be drawn from the target posterior distribution $p(\theta \mid X)$. MCMC “walks around” the space in a way that favors high-probability regions but still occasionally explores others. Over time, the visited points represent the true distribution.

<div class="accordion">
  <details>
    <summary>Does this method depend on the proposal distribution?</summary>
    <p>
      However, the beauty of Metropolis–Hastings is that it <strong>does not need the proposal to be a perfect guess of the target</strong>. The proposal just defines how the chain moves around, and the acceptance step ensures that — in the long run — the chain still samples from the true target distribution $p(x)$.
    </p>
    <p>That said:</p>
    <ul>
      <li>If $q(x' \mid x_t)$ is <strong>too narrow</strong>, the chain moves slowly (samples are highly correlated).</li>
      <li>If it’s <strong>too wide</strong>, the chain proposes jumps into low-probability regions often — so many proposals get rejected.</li>
      <li>If it’s <strong>reasonably tuned</strong>, you get efficient mixing and faster convergence.</li>
    </ul>
    <p>So, the proposal influences <strong>efficiency</strong>, not <strong>correctness</strong> (assuming detailed balance holds).</p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>So, we sample new data from the proposal, but judge it based on the true target distribution?</summary>
    <p>
      <strong>Exactly.</strong> The proposal generates <strong>candidates</strong>, and the <strong>target distribution</strong> determines whether they are “good enough” to accept.This acceptance rule corrects any bias introduced by the proposal and ensures that the long-run distribution of samples is the true target $p(x)$. That balance between exploration and exploitation is what makes MCMC powerful.
    </p>
  </details>
</div>


In practice, MCMC is used for 
* **Bayesian inference**: Sampling from posterior distributions when they can’t be computed analytically.
* **Machine learning**: Training latent-variable models like topic models or VAEs.

**Challenges in MCMC**

* **Choice of Proposal Density:**
    * A proposal density that is too wide (high variance) will often generate proposals in regions of low probability, leading to a high rejection rate.
    * A proposal density that is too narrow will lead to a very high acceptance rate, but the chain will explore the parameter space very slowly, potentially undersampling important regions.
* **Initial Conditions and Burn-in:** The initial samples of the chain are biased by the starting condition $\theta^{(0)}$. To mitigate this bias, an initial "burn-in" period is defined, where the first $M$ samples are discarded from the final estimate.
* **Convergence:** To assess whether the chain has converged to its stationary distribution, it is common practice to run multiple chains from different, overdispersed initial conditions and check if they converge to the same distribution.
* **Parameter Constraints:** If parameters must conform to certain constraints (e.g., positivity), proposals that violate these constraints must be rejected, or transformations must be applied.
* **Autocorrelation:** Consecutive samples in the chain are often highly correlated. To obtain nearly independent samples, a technique called **thinning** is used, which involves keeping only every $n$-th sample and discarding the rest.

<div class="accordion">
  <details>
    <summary>Why does autocorrelation matter?</summary>
    <p>
      When you run an MCMC sampler, each sample depends on the <strong>previous one</strong> — that’s the <em>Markov chain</em> property. This dependence means that <strong>successive samples are correlated</strong>, especially if the chain moves slowly through the space. This is called <strong>autocorrelation</strong>.
    </p>
    <p>
      Formally, for a sequence of samples $x^{(1)}, x^{(2)}, \dots$, the <strong>lag-\(k\)</strong> autocorrelation is:
    </p>
    $$
    \rho_k = \frac{\operatorname{Cov}(x^{(t)}, x^{(t+k)})}{\operatorname{Var}(x)}
    $$
    <ul>
      <li>If $\rho_k$ is high (close to 1), it means samples $k$ steps apart are <strong>very similar</strong>.</li>
      <li>If $\rho_k$ drops to 0 quickly, your chain is <strong>mixing well</strong> — it’s exploring the target distribution efficiently.</li>
    </ul>
    <p><strong>Why it matters</strong></p>
    <p>High autocorrelation means:</p>
    <ul>
      <li>You have <strong>less independent information</strong> per sample.</li>
      <li>You need <strong>more samples</strong> to get a good estimate of expectations.</li>
    </ul>
    <p>
      For example, if you have 10 000 samples but strong autocorrelation, your <strong>effective sample size (ESS)</strong> might only be a few hundred.
    </p>
    <p><strong>Thinning</strong></p>
    <p><strong>Thinning</strong> is a simple (though not always ideal) technique to reduce autocorrelation.</p>
    <p>
      You keep only every $k$-th sample and discard the rest:
    </p>
    $$
    x^{(1)},\; x^{(k+1)},\; x^{(2k+1)},\; \dots
    $$
    <p>
      The idea: if the chain is correlated over short lags, spacing samples apart might make them <em>approximately independent</em>.
    </p>
  </details>
</div>

Different MCMC algorithms exist for different problem structures. For instance, **Gibbs Sampling** is highly effective when dealing with models that have dependent parameters and where conditional distributions are easy to sample from.

**Parameter Inference for Latent Variable Models**

When the data generating process depends on both parameters $\theta$ and unobserved **latent variables** $z$, the model is specified as $p_\theta(X, z)$. The log-likelihood of the observed data $X$ requires marginalizing out these latent variables:

$$
\log p_\theta(X) = \log \int p_\theta(X, z) dz
$$

This integration is often intractable, necessitating methods like MCMC or Variational Inference.

<div class="accordion">
  <details>
    <summary>How is MCMC applied in Latent Variable Models?</summary>
    <p>
      #TODO
    </p>
  </details>
</div>

---

## 3. Fundamental Concepts of Time Series Analysis

### 3.1 Stochastic Processes and Time Series

Intuitively, a time series is a realization or **sample path** of a random process, such as $\lbrace X_1, X_2, ..., X_T\rbrace$. A time series is univariate if $X_t \in \mathbb{R}$ and multivariate if $X_t \in \mathbb{R}^k$ for $k > 1$. The fundamental assumption in time series analysis is that our observations are realizations of an underlying stochastic process.

$\textbf{Definition (Stochastic Process):}$ Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space, $(E, \mathcal{E})$ be a measurable space (the state space), and $I \subseteq \mathbb{R}$ be an index set. A family of random variables $X = \lbrace X_t\rbrace_{t \in I}$ with values in $E$ is called a **stochastic process**.

$\textbf{Definition (Time Series):}$ A **time series** is a stochastic process $X = \lbrace X_t\rbrace_{t \in I}$, where each random variable $X_t$ shares the same state space but may have a different probability distribution.

$\textbf{Definition (Sample Path (Realization)):}$ A **sample path** is a single outcome or sequence of observations $\lbrace x_t\rbrace_{t \in I}$ from a stochastic process. For example, $\lbrace x_1, x_2, ..., x_T\rbrace$.

$\textbf{Definition (Discrete vs. Continuous Time):}$
* If the index set $I$ is countable (e.g., $I = \mathbb{Z}$ or $I = \mathbb{N}$), the process is a **discrete-time process**.
* If the index set $I$ is an interval (e.g., $I = [0, T]$), the process is a **continuous-time process**.

**Definition (Ensemble):** The **ensemble** is the population of all possible realizations that a stochastic process can generate. For a process $X_t = A \sin(\omega t + \phi) + \epsilon_t$, the ensemble would be the set of all possible sine waves generated by different values of the random variables $A$, $\phi$, and $\epsilon_t$.

#### 3.1.1 Example: White Noise Process

**Definition (White Noise Process):** Let $\sigma^2 > 0$. A time series $X = \lbrace X_t\rbrace_{t \in I}$ is called a **white noise process** with variance $\sigma^2$, denoted $X_t \sim WN(0, \sigma^2)$, if it satisfies:
1.  $\mathbb{E}[X_t] = 0$ for all $t \in I$.
2.  $\text{Cov}(X_s, X_t) = \begin{cases} \sigma^2 & \text{if } s=t \\ 0 & \text{if } s \neq t \end{cases}$

<div class="accordion">
  <details>
    <summary>Short explanation of the second point.</summary>
    <p>
    <strong>“same distribution” ≠ “same random variable”</strong> and doesn’t imply any <strong>dependence</strong>.
    </p>

    $$
    \mathrm{Cov}(X_s,X_t)=\mathbb{E}\big[(X_s-\mu)(X_t-\mu)\big].
    $$

  </details>
</div>

This property is often imposed on the error terms $\epsilon_t$ of statistical models. If, additionally, $X_t \sim \mathcal{N}(0, \sigma^2)$, the process is called a **Gaussian white noise process**.

### 3.2 Autocovariance, Autocorrelation, and Cross-Correlation

**Definition (Autocovariance Function (ACVF)):** Let $X=\lbrace X_t\rbrace_{t \in I}$ be a stochastic process with $\mathbb{E}[X_t^2] < \infty$. The autocovariance function is a map $\gamma_{XX} : I \times I \to \mathbb{R}$ defined as:

$$
\gamma_{XX}(s, t) = \text{Cov}(X_s, X_t) = \mathbb{E}[(X_s - \mathbb{E}[X_s])(X_t - \mathbb{E}[X_t])]
$$

Using the property $\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$, and letting $\mu_t = \mathbb{E}[X_t]$, this can be written as:

$$
\gamma_{XX}(s, t) = \mathbb{E}[X_s X_t] - \mu_s \mu_t
$$

**Basic Properties of ACVF:**
* **Symmetry:** $\gamma_{XX}(s, t) = \gamma_{XX}(t, s)$
* **Variance:** $\gamma_{XX}(t, t) = \text{Var}(X_t)$
* **Cauchy-Schwarz Inequality:** $\lvert \gamma_{XX}(s, t)\rvert \leq \sqrt{\text{Var}(X_s)\text{Var}(X_t)}$
* The autocovariance function for a white noise process $X_t \sim WN(0, \sigma^2)$ is:
    * $\gamma_{XX}(s, t) = \sigma^2 \delta_{s,t}$
    * where $\delta_{s,t}$ is the Kronecker delta.

<div class="accordion">
  <details>
    <summary>Kronecker delta function</summary>
    <p>
      $$
      \delta_{ij} =
      \begin{cases}
      1 & \text{if } i = j \
      0 & \text{if } i \neq j
      \end{cases}
      $$
    </p>
  </details>
</div>

**Definition (Autocorrelation Function (ACF)):** The **autocorrelation function** is the normalized version of the autocovariance function, mapping $\rho_{XX}: I \times I \to [-1, 1]$:
$$
\rho_{XX}(s, t) = \frac{\gamma_{XX}(s, t)}{\sqrt{\gamma_{XX}(s, s)\gamma_{XX}(t, t)}} = \frac{\text{Cov}(X_s, X_t)}{\sqrt{\text{Var}(X_s)\text{Var}(X_t)}}
$$

**Definition (Cross-Covariance and Cross-Correlation Functions):** For two stochastic processes $X=\lbrace X_t\rbrace_{t \in I}$ and $Y=\lbrace Y_t\rbrace_{t \in I}$, the **cross-covariance function** is:
$$
\gamma_{XY}(s, t) = \text{Cov}(X_s, Y_t) = \mathbb{E}[(X_s - \mu_{X,s})(Y_t - \mu_{Y,t})]
$$
The **cross-correlation function** is its normalized version:
$$
\rho_{XY}(s, t) = \frac{\gamma_{XY}(s, t)}{\sqrt{\text{Var}(X_s)\text{Var}(Y_t)}}
$$

### 3.3 Stationarity and Ergodicity

#### 3.3.1 Strong (Strict) Stationarity

**Definition (Strong Stationarity):** Let $h \in \mathbb{R}$ and $m \in \mathbb{N}$. A stochastic process $X = \lbrace X_t\rbrace_{t \in I}$ is strongly stationary if for any choice of time points $t_1, \dots, t_m \in I$, the joint probability distribution of $(X_{t_1}, \dots, X_{t_m})$ is the same as the joint probability distribution of $(X_{t_1+h}, \dots, X_{t_m+h})$, provided all time points remain in $I$.

$$
(X_{t_1}, \dots, X_{t_m}) \stackrel{d}{=} (X_{t_1+h}, \dots, X_{t_m+h})
$$

where $\stackrel{d}{=}$ denotes equality in distribution.

* Strong stationarity is a statement about the entire joint distribution ("laws") of the process, which must be invariant to shifts in time.
* This is a foundational assumption for many time series models.
* In practice, verifying the equality of all moments and distributions is impossible from a single finite realization.

#### 3.3.2 Weak Stationarity

**Definition (Weak Stationarity):** A stochastic process $X = \lbrace X_t\rbrace_{t \in I}$ is **weakly stationary** (or covariance stationary) if it satisfies the following three conditions:
1.  The mean is constant for all $t$: $\mathbb{E}[X_t] = \mu$.
2.  The variance is finite for all $t$: $\mathbb{E}[X_t^2] < \infty$.
3.  The autocovariance between any two points depends only on their time lag $h = t-s$:
    * $\gamma_{XX}(s, t) = \gamma_{XX}(s, s+h) = \gamma_X(h)$

* Strong stationarity implies weak stationarity (provided the first two moments exist). The reverse is not generally true.
* For a Gaussian process, weak stationarity implies strong stationarity, because the entire distribution is defined by its first two moments (mean and covariance).
* For a weakly stationary process, the autocovariance function simplifies to $\gamma_X(h) = \mathbb{E}[(X_{t+h} - \mu)(X_t - \mu)]$.

#### 3.3.3 Ergodicity

**Definition (Ergodicity):** A stationary process is **ergodic** if its time average converges to its ensemble average (expected value) almost surely as the time horizon grows to infinity. For the mean:

$$
\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T X_t = \mathbb{E}[X_t] = \mu
$$

* Ergodicity allows us to infer properties of the entire process (the ensemble) from a single, sufficiently long sample path.
* Ergodicity requires stationarity. It also typically requires conditions of stability (small perturbations do not cause large changes) and mixing (the influence of initial conditions fades over time).

### 3.4 Computing Properties from a Time Series

Under the assumptions of weak stationarity and ergodicity, we can estimate the moments of the stochastic process from a single time series realization $\lbrace x_t\rbrace_{t=1}^T$.

* **Sample Mean:** $\hat{\mu} = \bar{x} = \frac{1}{T} \sum_{t=1}^T x_t$
* **Sample Variance:** $\hat{\gamma}(0) = \hat{\sigma}^2 = \frac{1}{T} \sum_{t=1}^T (x_t - \bar{x})^2$
* **Sample Autocovariance at lag $h$:** $\hat{\gamma}(h) = \frac{1}{T} \sum_{t=1}^{T-h} (x_t - \bar{x})(x_{t+h} - \bar{x})$

### 3.5 Dealing with Non-stationarity

If a time series is non-stationary, it must often be transformed before standard models can be applied. Common techniques include:

* **High-pass filtering:** Transforming the series from the time domain to the frequency domain to remove low-frequency (trending) components.
* **Differencing:** Creating a new series by taking the difference between consecutive observations, e.g., $y_t = x_t - x_{t-1}$.
* **Detrending:** Fitting a deterministic trend (e.g., a linear function of time) and subtracting it from the data. For a linear trend $y_t = \beta_0 + \beta_1 t$, the detrended series is:
    $x_t^* = x_t - (\hat{\beta}_0 + \hat{\beta}_1 t)$

---

## 4. Linear Regression

### 4.1 Model Components

A complete statistical model generally consists of four key components:

1.  **Model Architecture:** The mathematical form of the relationship between variables.
2.  **Loss Function:** A function that quantifies the error between model predictions and actual data.
3.  **Training Algorithm:** An optimization procedure to find the parameters that minimize the loss function.
4.  **Data:** The observations used to train and evaluate the model.

### 4.2 Model Architecture

Given a dataset $D = \lbrace(x_t, y_t)\rbrace_{t=1}^T$ where $y_t$ are responses and $x_t$ are predictors, the univariate linear regression model assumes a linear relationship:
$y_t = \beta_0 + \beta_1 x_t + \epsilon_t$
The error term $\epsilon_t$ is typically assumed to be a white noise process, often Gaussian:
$\epsilon_t \sim \mathcal{N}(0, \sigma^2)$
This implies a conditional distribution for the response variable:
$y_t \mid x_t \sim \mathcal{N}(\beta_0 + \beta_1 x_t, \sigma^2)$

For a model with $p$ predictors, this can be vectorized. Let $y$ be a $T \times 1$ vector of responses, $X$ be a $T \times (p+1)$ design matrix (with a column of ones for the intercept), and $\beta$ be a $(p+1) \times 1$ vector of parameters. The model is:
$$
y = X\beta + \epsilon
$$

### 4.3 Loss Function

The most common loss function for linear regression is the **Sum of Squared Errors (SSE)**, also known as the **Least Squares Error (LSE)**. The objective is to find the parameter vector $\beta$ that minimizes this quantity:
$$
\text{LSE}(\beta) = \sum_{t=1}^T (y_t - \hat{y}_t)^2 = \sum_{t=1}^T (y_t - x_t^T \beta)^2
$$
In vector notation, this is:
$$
\text{LSE}(\beta) = (y - X\beta)^T(y - X\beta)
$$

### 4.4 Training Algorithm

The optimal parameters $\hat{\beta}$ are found by minimizing the LSE loss function. This is achieved by taking the derivative of the loss function with respect to $\beta$ and setting it to zero.

Expanding the LSE expression:
$\text{LSE}(\beta) = y^Ty - y^TX\beta - \beta^TX^Ty + \beta^TX^TX\beta$
Since $y^TX\beta$ is a scalar, it is equal to its transpose $(\beta^TX^Ty)$. Therefore:

$$
\text{LSE}(\beta) = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta
$$

Now, we take the derivative with respect to the vector $\beta$:

$$
\frac{\partial \text{LSE}(\beta)}{\partial \beta} = \frac{\partial}{\partial \beta} (y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta)
$$

Using the matrix calculus rules:
* $\frac{\partial(a^Tx)}{\partial x} = a$
* $\frac{\partial(x^TAx)}{\partial x} = (A + A^T)x$

The derivative is:
$\frac{\partial \text{LSE}(\beta)}{\partial \beta} = -2X^Ty + (X^TX + (X^TX)^T)\beta$
Since $X^TX$ is a symmetric matrix, $(X^TX)^T = X^TX$. The derivative simplifies to:

$$
\frac{\partial \text{LSE}(\beta)}{\partial \beta} = -2X^Ty + 2X^TX\beta
$$

Setting the derivative to zero to find the minimum:
$-2X^Ty + 2X^TX\hat{\beta} = 0$
$2X^TX\hat{\beta} = 2X^Ty$
$X^TX\hat{\beta} = X^Ty$

The solution for $\hat{\beta}$, known as the ordinary least squares (OLS) estimator, is:
$$
\hat{\beta} = (X^TX)^{-1}X^Ty
$$

## 5. Regression Models for Time Series

This chapter introduces foundational regression techniques, which form the building blocks for more complex time series models. We will cover linear regression, extensions for non-linear relationships, models for non-Gaussian data, and approaches for handling multivariate and hierarchical data structures.

### 5.1 Linear Regression

A statistical model is formally composed of four key components:

1.  **Model Architecture:** The mathematical form of the relationship between variables.
2.  **Loss Function:** A function that quantifies the error between model predictions and observed data.
3.  **Training Algorithm:** An optimization procedure to find model parameters that minimize the loss function.
4.  **Data:** The observed measurements used to train and evaluate the model.

In the context of time series, we consider a dataset $D$ consisting of $T$ observations. At each time step $t=1, \dots, T$, we observe a response variable $x_t$ and a vector of $p$ predictor variables $u_t$. The dataset is thus represented as: $D = \lbrace(u_t, x_t)\rbrace_{t=1}^T$

#### 5.1.1 Model Architecture

The simple linear regression model assumes a linear relationship between the predictors and the response, corrupted by additive Gaussian noise.

**Definition (Linear Regression Model):** The response variable $x_t$ is modeled as a linear combination of the predictor variables $u_t$ plus an error term $\epsilon_t$.

$$
x_t = u_t^T \beta + \epsilon_t
$$

where:

* $x_t \in \mathbb{R}$ is the response variable at time $t$.
* $u_t \in \mathbb{R}^{p+1}$ is the vector of predictor variables at time $t$ (including a constant term for the intercept).
* $\beta \in \mathbb{R}^{p+1}$ is the vector of model parameters or coefficients.
* $\epsilon_t$ is the error term, assumed to be independent and identically distributed (i.i.d.) Gaussian noise: $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$.

This implies that the conditional distribution of the response variable is also Gaussian:

$$
x\_t \mid u\_t, \beta, \sigma^2 \sim \mathcal{N}(u\_t^T \beta, \sigma^2)
$$

For the entire dataset, we can express the model in a vectorized form:

$$
X = U\beta + E
$$

where:

* $X = (x_1, \dots, x_T)^T$ is the $T \times 1$ vector of responses.
* $U$ is the $T \times (p+1)$ design matrix, where each row is $u_t^T$.
* $\beta = (\beta_0, \dots, \beta_p)^T$ is the $(p+1) \times 1$ parameter vector.
* $E = (\epsilon_1, \dots, \epsilon_T)^T$ is the $T \times 1$ vector of errors.

#### 5.1.2 Parameter Estimation: Least Squares

The most common method for estimating the parameters $\beta$ in a linear regression model is the **method of ordinary least squares (LSE)**. This involves minimizing the sum of the squared differences between the observed responses and the responses predicted by the model.

**Definition (Least Squares Loss Function):** The LSE loss function for the parameter vector $\beta$ is the sum of squared residuals:

$$
L\_{\text{LSE}}(\beta) = \sum\_{t=1}^T (x\_t - u\_t^T \beta)^2
$$

In vector form, this is expressed as:

$$
L_{\text{LSE}}(\beta) = (X - U\beta)^T(X - U\beta)
$$

The training algorithm consists of finding the value of $\beta$ that minimizes this loss function. This can be achieved by taking the derivative of $L_{\text{LSE}}(\beta)$ with respect to $\beta$ and setting it to zero.

To find the estimator $\hat{\beta}_{\text{LS}}$ that minimizes the loss, we compute the gradient of the loss function with respect to $\beta$.

1.  **Expand the loss function:**

$$
L\_{\text{LSE}}(\beta) = (X - U\beta)^T(X - U\beta) = X^T X - X^T U \beta - \beta^T U^T X + \beta^T U^T U \beta
$$


2. **Compute the derivative with respect to $\beta$:**
   * Note that $X^T U \beta$ is a scalar, so it equals its transpose $\beta^T U^T X$. Using this, the loss is $L_{\text{LSE}}(\beta) = X^T X - 2\beta^T U^T X + \beta^T U^T U \beta$.
   * Using the matrix calculus rules $\frac{\partial(a^T x)}{\partial x} = a$ and $\frac{\partial(x^T A x)}{\partial x} = (A + A^T)x$:
     $$
     \frac{\partial L\_{\text{LSE}}(\beta)}{\partial \beta} = \frac{\partial}{\partial \beta} (X^T X - 2\beta^T U^T X + \beta^T U^T U \beta)
     $$

     $$
     \frac{\partial L\_{\text{LSE}}(\beta)}{\partial \beta} = 0 - 2U^T X + (U^T U + (U^T U)^T)\beta
     $$

   * Since $U^T U$ is symmetric, $(U^T U)^T = U^T U$.
     $$
     \frac{\partial L\_{\text{LSE}}(\beta)}{\partial \beta} = -2U^T X + 2U^T U \beta
     $$

3. **Set the derivative to zero and solve for $\beta$:**
    $$
    -2U^T X + 2U^T U \hat{\beta} = 0
    $$
    
    $$
    2U^T U \hat{\beta} = 2U^T X
    $$
    
    $$
    U^T U \hat{\beta} = U^T X
    $$

4. Assuming that the matrix $U^T U$ is invertible, we can solve for $\hat{\beta}$:
    $$
    \hat{\beta}\_{\text{LS}} = (U^T U)^{-1} U^T X
    $$

This is the celebrated **normal equation** solution for ordinary least squares.

#### 5.1.3 Parameter Estimation: Maximum Likelihood

An alternative framework for parameter estimation is **Maximum Likelihood Estimation (MLE)**. This approach finds the parameter values that maximize the likelihood of observing the given data.

**Definition (Likelihood Function):** Given the model assumption $x_t \sim \mathcal{N}(u_t^T \beta, \sigma^2)$, the likelihood of observing the entire dataset $D$ is the product of the probability densities for each observation:

$$
\mathcal{L}(\beta, \sigma^2) = p(X \mid U, \beta, \sigma^2) = \prod_{t=1}^T p(x_t \mid u_t, \beta, \sigma^2)
$$

$$
\mathcal{L}(\beta, \sigma^2) = \prod_{t=1}^T \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_t - u_t^T \beta)^2}{2\sigma^2}\right)
$$

It is often more convenient to work with the **log-likelihood function**:

$$
l(\beta, \sigma^2) = \log \mathcal{L}(\beta, \sigma^2) = \sum\_{t=1}^T \left[ -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(x\_t - u\_t^T beta)^2 \right]
$$

To find the MLE for $\beta$, we maximize $l(\beta, \sigma^2)$ with respect to $\beta$. Notice that the terms involving $\sigma^2$ and $2\pi$ are constant with respect to $\beta$. Therefore, maximizing the log-likelihood is equivalent to minimizing the sum of squared errors:

$$
\arg\max_\beta l(\beta, \sigma^2) \equiv \arg\min_\beta \sum_{t=1}^T (x_t - u_t^T \beta)^2
$$

For the linear regression model with the assumption of i.i.d. Gaussian errors, the Least Squares Estimator (LSE) and the Maximum Likelihood Estimator (MLE) for the regression coefficients $\beta$ are identical.

$$
\hat{\beta}*{\text{LS}} = \hat{\beta}*{\text{MLE}}
$$

-----

### 5.2 Model Diagnostics

After fitting a model, it is crucial to assess its validity by examining the **residuals**, which are the differences between the observed and fitted values.

The model decomposes the observed data $x_t$ into a fitted signal $\hat{x}_t$ and a residual component $res_t$:

$$x\_t = \hat{x}\_t + res\_t
$$

where $\hat{x}_t = u_t^T \hat{\beta}$ is the predicted value.

The core assumptions of the linear regression model (linearity, normality of errors, constant variance, independence of errors) should be checked by analyzing the residuals.

**Common Diagnostic Visualizations:**

  * **Error Histogram:** A histogram of the residuals $res_t$ can provide a first check on the assumption that the errors are normally distributed.
  * **Q-Q (Quantile-Quantile) Plot:** This is a more rigorous way to check for normality. It plots the sorted quantiles of the residuals against the theoretical quantiles of a standard normal distribution. If the residuals are normally distributed, the points on the Q-Q plot will lie close to a straight diagonal line.
  * **ACF Plot of Residuals:** The Autocorrelation Function (ACF) plot of the residuals is used to check the assumption of independence. For time series data, it is critical to ensure there is no remaining temporal structure (autocorrelation) in the residuals. Significant spikes in the ACF plot suggest that the model has failed to capture the temporal dynamics in the data.

**Sources of "Bad Residuals" (Model Misspecification):**

  * **Wrong assumption on modality:** The errors may not be normally distributed.
  * **Wrong assumption on relationship:** The true relationship between predictors and response may be non-linear.
  * **Missing external predictors:** Important explanatory variables may have been omitted from the model.
  * **Missing temporal dynamics:** The model fails to account for dependencies between observations over time.

-----

### 5.3 Handling Non-Linearity: Basis Expansion

One way to address a non-linear relationship between predictors and the response is through **basis expansion**. This technique extends the linear model by including non-linear transformations of the original predictors as additional regressors.

Suppose we have a set of predictors $u_t = (u_{1t}, \dots, u_{pt})^T$.

**Definition (Basis Expansion Model):** The model architecture is extended by introducing a set of $K$ basis functions, $\phi_k(\cdot)$:

$$
x_t = \beta_0 + \sum_{k=1}^K \beta_k \phi_k(u_t) + \epsilon_t
$$

where $\phi_k(\cdot)$ are chosen functions that transform the original predictor vector $u_t$. This model is still linear in the parameters $\beta_k$, so the standard LSE and MLE solutions still apply, but with a new design matrix whose columns are the transformed predictors $\phi_k(u_t)$.

**Common Choices for Basis Functions:**

* **Polynomials:** $\phi_k(u_j) = u_{jt}^k$ (e.g., $u_t, u_t^2, u_t^3, \dots$).
* **Interaction Terms:** $\phi(u_i, u_j) = u_{it} \cdot u_{jt}$.
* **Radial Basis Functions:** Functions that depend on the distance from a center point.
* **Fourier Basis:** Sines and cosines to model periodic patterns, e.g., $\sin(\omega t)$ and $\cos(\omega t)$.

-----

### 5.4 Multivariate Linear Regression

In many scenarios, we want to predict multiple response variables simultaneously. This leads to **multivariate linear regression**.

**Data:** At each time $t=1, \dots, T$:

* Response vector: $x_t = (x_{1t}, \dots, x_{qt})^T \in \mathbb{R}^q$.
* Predictor vector: $u_t = (u_{1t}, \dots, u_{pt})^T \in \mathbb{R}^p$.

**Model:** The model is a direct extension of the univariate case, written in matrix form:

$$
X = UB + E
$$

where:

  * $X$ is the $T \times q$ matrix of response variables.
  * $U$ is the $T \times (p+1)$ design matrix.
  * $B$ is the $(p+1) \times q$ matrix of parameters, where each column corresponds to a response variable.
  * $E$ is the $T \times q$ matrix of errors.

**Estimation (LSE/MLE):** The solution for the parameter matrix $B$ is analogous to the univariate case:

$$
\hat{B} = (U^T U)^{-1} U^T X
$$

This is equivalent to performing $q$ separate univariate linear regressions, one for each response variable.

-----

### 5.5 Generalized Linear Models (GLMs)

Linear regression assumes a normally distributed response variable. **Generalized Linear Models (GLMs)** provide a framework to handle response variables with other distributions (e.g., binary, count data).

GLMs are composed of three components:

1.  **Random Component:** The response variable $y_t$ follows a probability distribution from the exponential family (e.g., Bernoulli, Poisson, Gamma).
2.  **Systematic Component:** A **linear predictor**, $\eta_t$, is constructed as a linear combination of the predictors:

$$
\eta_t = u_t^T \beta
$$

3.  **Link Function:** A function $g(\cdot)$ that links the expected value of the response, $\mu_t = E[y_t]$, to the linear predictor:

$$
g(\mu_t) = \eta_t
$$

The inverse of the link function, $g^{-1}(\cdot)$, maps the linear predictor back to the mean of the response: $\mu_t = g^{-1}(\eta_t)$.

The standard linear regression model is a special case of a GLM.

  * **Random Component:** $y_t \sim \mathcal{N}(\mu_t, \sigma^2)$.
  * **Systematic Component:** $\eta_t = u_t^T \beta$.
  * **Link Function:** The identity link, $g(\mu_t) = \mu_t$. Therefore, $\eta_t = \mu_t$, which gives us the familiar $E[y_t] = u_t^T \beta$.

#### 5.5.1 Example: Logistic Regression

Logistic regression is a GLM used for modeling binary response variables.

**Model Architecture:**

  * **Data:** Observed binary response variable $x_t \in \lbrace 0, 1 \rbrace$ for $t=1, \dots, T$, with predictor vector $u_t$.
  * **Random Component:** The response is assumed to follow a Bernoulli distribution:
    
    $$
    x_t \sim \text{Bernoulli}(\pi_t)
    $$
    
    where $\pi_t = P(x_t=1 \mid u_t)$ is the "success" probability.
  * **Systematic Component:** The linear predictor is $\eta_t = u_t^T \beta$.
  * **Link Function:** The **logit** link function is used, which is the natural logarithm of the odds:
    
    $$
    g(\pi_t) = \log\left(\frac{\pi_t}{1-\pi_t}\right) = \eta_t
    $$
    
    The inverse link function is the **sigmoid** (or logistic) function, which maps the linear predictor to a probability between 0 and 1:
    
    $$
    \pi_t = g^{-1}(\eta_t) = \frac{e^{\eta_t}}{1+e^{\eta_t}} = \frac{1}{1+e^{-\eta_t}} = \sigma(\eta_t)
    $$

#### 5.5.2 Maximum Likelihood Estimation for Logistic Regression

Parameters in a GLM are typically estimated using MLE.

**Loss Function (Negative Log-Likelihood):** The probability mass function for a single Bernoulli observation is $p(x_t \mid \pi_t) = \pi_t^{x_t}(1-\pi_t)^{1-x_t}$. The likelihood for the entire dataset is:

$$
\mathcal{L}(\beta) = \prod_{t=1}^T p(x_t \mid u_t, \beta) = \prod_{t=1}^T \pi_t^{x_t} (1-\pi_t)^{1-x_t}
$$

The log-likelihood is:

$$
l(\beta) = \sum_{t=1}^T \left[ x_t \log(\pi_t) + (1-x_t)\log(1-\pi_t) \right]
$$

Substituting $\pi_t = \sigma(u_t^T \beta)$, we can express the log-likelihood in a form common to the exponential family:

$$
l(\beta) = \sum_{t=1}^T \left[ x_t (u_t^T \beta) - \log(1+e^{u_t^T \beta}) \right]
$$

**Training Algorithm:** We maximize the log-likelihood by taking its derivative with respect to $\beta$ and setting it to zero.

$$
\nabla_\beta l(\beta) = \nabla_\beta \sum_{t=1}^T \left[ x_t \log(\sigma(u_t^T\beta)) + (1-x_t)\log(1-\sigma(u_t^T\beta)) \right]
$$

The derivative of the log-likelihood for a single observation with respect to $\beta$ is:

$$
\nabla_\beta l_t(\beta) = \left(x_t - \sigma(u_t^T\beta)\right)u_t = (x_t - \pi_t)u_t
$$

Summing over all observations gives the full gradient:

$$
\nabla_\beta l(\beta) = \sum_{t=1}^T (x_t - \pi_t) u_t
$$

Unlike in linear regression, setting this equation to zero does not yield a closed-form solution for $\beta$. Therefore, iterative optimization algorithms like **gradient ascent** are used.

The gradient ascent update rule is:

$$
\beta_{\text{new}} = \beta_{\text{old}} + \alpha \nabla_\beta l(\beta_{\text{old}})
$$where $\alpha$ is the learning rate. For a single data point (stochastic gradient ascent), the rule is:

$$
\beta_{\text{new}} = \beta_{\text{old}} + \alpha (x_t - \pi_t) u_t
$$

-----

### 5.6 Modeling Complex Data Structures

#### 5.6.1 Multimodal Regression

This approach models datasets where each observation consists of multiple types of data, or **modalities**.

**Setup:**

* We have multiple observed modalities, for example, a binary variable $x_{1t} \in \lbrace 0,1 \rbrace$ and a continuous variable $x_{2t} \in \mathbb{R}$.
* These are collected into a response vector $x_t = (x_{1t}, x_{2t})^T$.
* The full dataset is $D = \lbrace(x_t, u_t)\rbrace_{t=1}^T$.

**Model:** We model each modality separately, conditional on the predictors $u_t$. This typically involves different parameter vectors ($\beta_1, \beta_2$) and distributions for each modality.

* Linear predictor for modality 1: $\eta_{1t} = \beta_1^T u_t$
* Linear predictor for modality 2: $\eta_{2t} = \beta_2^T u_t$
* Model for modality 1: $x_{1t} \mid u_t \sim \text{Bernoulli}(\pi_t)$, where $\pi_t = \sigma(\eta_{1t})$.
* Model for modality 2: $x_{2t} \mid u_t \sim \mathcal{N}(\mu_t, \sigma^2)$, where $\mu_t = \eta_{2t}$.

A key assumption is **conditional independence**: the different modalities are independent of each other, given the predictors.

$$
p(x_{1t}, x_{2t} \mid u_t) = p(x_{1t} \mid u_t) p(x_{2t} \mid u_t)
$$

**Loss Function (MLE):** Due to the conditional independence assumption, the total log-likelihood is the sum of the log-likelihoods for each modality:

$$
l(\beta_1, \beta_2, \sigma^2) = \sum_{t=1}^T \log p(x_{1t} \mid u_t) + \sum_{t=1}^T \log p(x_{2t} \mid u_t)
$$

$$
l(\beta_1, \beta_2, \sigma^2) = \sum_{t=1}^T [x_{1t}\eta_{1t} - \log(1+e^{\eta_{1t}})] + \sum_{t=1}^T \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(x_{2t} - \eta_{2t})^2 \right]
$$

This loss function can be maximized jointly to find the parameters for both models.

If the assumption of conditional independence between modalities is not reasonable, a common approach is to introduce a shared **latent variable** $\epsilon_t$ that captures common dynamics or unobserved factors influencing all modalities. Conditional on both the predictors $u_t$ and the latent variable $\epsilon_t$, the modalities are then assumed to be independent.

#### 5.6.2 Hierarchical Modeling

Hierarchical (or multilevel) models are designed for datasets with a nested or grouped structure, such as multiple time series from different subjects.

**Data:** We have $N$ independent datasets, one for each subject $j=1, \dots, N$:

$$
D_j = \lbrace (u_{1j}, x_{1j}), \dots, (u_{T_j j}, x_{T_j j}) \rbrace
$$

**Modeling Strategies:**

  * **Separate Models (No Sharing):** Fit a completely separate model with parameters $\theta_j$ for each subject $j$ using only their data $D_j$.
      * **Pros:** Captures individual differences perfectly.
      * **Cons:** If $T_j$ is small, estimates of $\theta_j$ can be noisy and prone to overfitting.
  * **Fully Pooled (Complete Sharing):** Concatenate all data $D = [D_1, \dots, D_N]$ and fit a single parameter vector $\theta$ for all subjects.
      * **Pros:** Provides more stable estimates by borrowing statistical strength across subjects.
      * **Cons:** Fails to capture inter-individual differences.

**The Hierarchical Approach:** Hierarchical modeling provides a compromise between these two extremes. The core idea is to model the individual-specific parameters themselves as being drawn from a common group-level distribution.

  * **Subject-Specific Parameters:** Each subject $j$ has their own parameter vector, $\theta_j$.
  * **Parent Distribution:** These parameters are not arbitrary but are assumed to be drawn from a common parent distribution, which is governed by hyperparameters $\xi$:
    
    $$
    \theta_j \sim p(\theta \mid \xi)
    $$

  * **Hyperprior:** To complete the Bayesian formulation, a prior distribution (a **hyperprior**) is placed on the hyperparameters $\xi$:
    
    $$
    \xi \sim p(\xi)
    $$
    
    This creates a hierarchical chain of dependencies:

$$
\xi \to \theta_j \to D_j \quad \text{for } j=1, \dots, N
$$

The joint distribution over all data and parameters is:

$$
p(\xi, \lbrace\theta_j\rbrace_{j=1}^N, \lbrace D_j\rbrace_{j=1}^N) = p(\xi) \prod_{j=1}^N \left[ p(\theta_j \mid \xi) p(D_j \mid \theta_j) \right]
$$

The goal of hierarchical Bayesian analysis is to compute the posterior distribution of the subject-level parameters and group-level hyperparameters, given the observed data:

$$
p(\xi, \lbrace\theta_j\rbrace_{j=1}^N \mid \lbrace D_j\rbrace_{j=1}^N)
$$

This approach allows for "partial pooling," where information is shared across subjects through the parent distribution, leading to more robust estimates while still allowing for individual variation.