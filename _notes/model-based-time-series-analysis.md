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
1. Stop when convergence is reached (e.g., $\lvert J(\theta_{i}) - J(\theta_{i+1})\rvert < \epsilon$) or no more iterations.

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

$$
x_t = \hat{x}_t + res_t
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
p(\xi, \lbrace\theta_j\rbrace_{j=1}^N \mid \lbrace D_j\rbrace_{j=1}^N) = \dfrac{p(\lbrace D_j\rbrace_{j=1}^N \mid \lbrace\theta_j\rbrace_{j=1}^N)p(\lbrace\theta_j\rbrace_{j=1}^N \mid \xi)p(\xi)}{p(\lbrace D_j\rbrace_{j=1}^N)}
$$

This approach allows for "partial pooling," where information is shared across subjects through the parent distribution, leading to more robust estimates while still allowing for individual variation.


<!-- ---

## 6. Hierarchical Modeling for Multiple Time Series

Hierarchical (or multilevel) models are especially useful when we observe **multiple related time series**, for example different subjects in an experiment or multiple sensors measuring similar phenomena. They provide a principled compromise between:

* **Separate models (no sharing)** — good for individuality, bad for stability when data per subject is small.
* **Fully pooled model (complete sharing)** — good for stability, bad for individuality.

### 6.1 Problem Setup: Multiple Subjects / Time Series

Suppose we have $N$ subjects (or units). For each subject $i = 1, \dots, N$ we observe a time series dataset $D_i$:

* Time points: $t = 1, \dots, T_i$ (often we assume $T_i = T$ for all $i$).
* Observations: $x_{it}$.
* Optionally, inputs or covariates: $w_{it}$.

We collect

$$
D_i = \{x_{i1}, \dots, x_{iT_i}\}, 
\qquad
W_i = \{w_{i1}, \dots, w_{iT_i}\}.
$$

The goal is to model the time series of **each subject** while still **sharing information** across subjects.

#### 6.1.1 Two Extreme Modeling Strategies

**(1) Separate Models (No Sharing)**

* For each subject $i$, fit an independent model with its own parameters $\theta_i$ using only $D_i$.
* **Pros:**  
  * Maximal flexibility — each subject has its own parameters.  
  * Captures individual differences.
* **Cons:**  
  * If $T_i$ is small and the model is complex, estimates of $\theta_i$ are noisy and prone to overfitting.  
  * No information sharing across subjects.

**(2) Fully Pooled Model (Complete Sharing)**

* Concatenate all datasets:
  $$
  D_{\text{pooled}} = \{D_1, \dots, D_N\}
  $$
* Fit a single parameter vector $\theta$ to all data.
* **Pros:**  
  * Very stable parameter estimates (large effective sample size).
* **Cons:**  
  * Ignores inter-individual differences — assumes all subjects are governed by the same dynamics.

Both extremes are often unsatisfactory in practice.

#### 6.1.2 The Hierarchical Approach

Hierarchical modeling introduces **subject-specific parameters** while constraining them through a **shared parent distribution**.

**Core idea:**

* Each subject $i$ has parameters $\theta_i$.
* These parameters are themselves random draws from a parent distribution with **hyperparameters** $\beta$.

Formally,

* **Subject–level parameters:**
  $$
  \theta_i \sim p(\theta \mid \beta), \quad i = 1, \dots, N
  $$
* **Hyperprior on group-level parameters:**
  $$
  \beta \sim p(\beta)
  $$

This creates the hierarchical chain

$$
\beta \;\rightarrow\; \theta_i \;\rightarrow\; D_i, 
\qquad i = 1, \dots, N.
$$

**Joint distribution.** For all parameters and data:

$$
p\bigl(\beta, \{\theta_i\}_{i=1}^N, \{D_i\}_{i=1}^N \bigr)
= p(\beta)\,\prod_{i=1}^N p(\theta_i \mid \beta)\, p(D_i \mid \theta_i).
$$

**Posterior of interest:**

$$
p(\theta_1, \dots, \theta_N, \beta \mid D_1, \dots, D_N).
$$

This structure allows the model to **borrow statistical strength**:

* Data from subject $j$ updates $\theta_j$.
* This in turn updates $\beta$ (group-level hyperparameters).
* The updated $\beta$ regularizes and improves the estimates of $\theta_i$ for all other subjects $i \neq j$.

---

### 6.2 Example: Hierarchical Delay Discounting

A common application of hierarchical modeling in cognitive science is the **delay discounting** task.

#### 6.2.1 Data and Task Structure

* We have $N$ individuals.
* Subject $i$ performs $T_i$ trials, $t = 1, \dots, T_i$.
* On each trial $t$, subject $i$ chooses between:
  * An **immediate reward** of amount $A_{I,it}$.
  * A **delayed reward** of amount $A_{D,it}$ available after delay $D_{it}$.
* Observed choice (binary):
  $$
  y_{it} = 
  \begin{cases}
  1 & \text{if the delayed reward is chosen} \\
  0 & \text{if the immediate reward is chosen}
  \end{cases}
  $$
* Full dataset for subject $i$:
  $$
  Y_i = \{y_{it}\}_{t=1}^{T_i}.
  $$

#### 6.2.2 Model Architecture: Hyperbolic Delay Discounting

We model **subjective value** of the options as follows.

* **Immediate option:**
  $$
  V_{I,it} = A_{I,it}
  $$
* **Delayed option (hyperbolic discounting):**
  $$
  V_{D,it} = \frac{A_{D,it}}{1 + k_i D_{it}},
  $$
  where $k_i > 0$ is the **discount rate** of subject $i$.  
  Larger $k_i$ $\Rightarrow$ stronger devaluation of future rewards.

Define the value difference
$$
\Delta V_{it} = V_{D,it} - V_{I,it}.
$$

A positive $\Delta V_{it}$ indicates a preference for the delayed option.

We map $\Delta V_{it}$ to a **choice probability** using a logistic (softmax) function:

$$
p_{it} := P(y_{it} = 1 \mid k_i, \beta_i) 
= \sigma(\beta_i \Delta V_{it})
= \frac{1}{1 + \exp(-\beta_i \Delta V_{it})},
$$

where $\beta_i > 0$ is a **temperature / sensitivity** parameter for subject $i$:

* Large $\beta_i$ $\Rightarrow$ more deterministic choices.
* Small $\beta_i$ $\Rightarrow$ more random choices.

Assuming conditional independence across trials,

$$
\mathcal{L}(k_i, \beta_i \mid Y_i)
= \prod_{t=1}^{T_i} p_{it}^{y_{it}} (1 - p_{it})^{1 - y_{it}}.
$$

#### 6.2.3 Hierarchical Priors

We now place **hierarchical priors** on the subject-specific parameters $k_i$ and $\beta_i$.

We require $k_i > 0$ and $\beta_i > 0$, so we model their **logarithms** as Gaussian:

* Individual parameters:
  $$
  \log k_i \sim \mathcal{N}(\mu_k, \sigma_k^2),
  \qquad
  \log \beta_i \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2),
  $$
  for $i = 1, \dots, N$.

* Group-level hyperparameters:
  $$
  \mu_k, \mu_\beta \in \mathbb{R}, \qquad
  \sigma_k^2, \sigma_\beta^2 > 0.
  $$

* Hyperpriors on means:
  $$
  \mu_k \sim \mathcal{N}(\mu_{k0}, \sigma_{k0}^2),
  \qquad
  \mu_\beta \sim \mathcal{N}(\mu_{\beta 0}, \sigma_{\beta 0}^2).
  $$

* Hyperpriors on variances (e.g., Inverse-Gamma):
  $$
  \sigma_k^2 \sim \text{Inverse-Gamma}(a_k, b_k),
  \qquad
  \sigma_\beta^2 \sim \text{Inverse-Gamma}(a_\beta, b_\beta).
  $$

#### 6.2.4 Full Bayesian Formulation

Let $k = \rbrace k_i\lbrace_{i=1}^N$ and $\beta = \lbrace\beta_i\rbrace_{i=1}^N$, and let $Y = \lbraceY_i\rbrace_{i=1}^N$.

We seek the posterior

$$
p(k, \beta, \mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2 \mid Y).
$$

By Bayes’ theorem:

$$
p(\cdot \mid Y) \propto 
P(Y \mid k, \beta)\,
P(k, \beta \mid \mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2)\,
P(\mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2).
$$

Breaking this down:

* **Likelihood (across subjects):**
  $$
  P(Y \mid k, \beta) = \prod_{i=1}^N P(Y_i \mid k_i, \beta_i).
  $$

* **Prior on individual parameters:**
  $$
  P(k, \beta \mid \dots) 
  = \prod_{i=1}^N P(\log k_i \mid \mu_k, \sigma_k^2)
    \prod_{i=1}^N P(\log \beta_i \mid \mu_\beta, \sigma_\beta^2).
  $$

* **Hyperprior (assuming independence):**
  $$
  P(\mu_k, \mu_\beta, \sigma_k^2, \sigma_\beta^2)
  = P(\mu_k)\,P(\mu_\beta)\,P(\sigma_k^2)\,P(\sigma_\beta^2).
  $$

Posterior inference is typically performed via **MCMC** or other approximate Bayesian methods.

---

### 6.3 Alternative Parameterization: Parent Matrix

In high-dimensional settings, it is often useful to **reduce dimensionality** of subject-specific parameters.

#### 6.3.1 Naive Parameterization

Assume each subject has a $p$-dimensional parameter vector

$$
\theta_i \in \mathbb{R}^p, \quad i = 1, \dots, N.
$$

Total number of subject-level parameters: $p \times N$.

#### 6.3.2 Low-Rank Parent Matrix Parameterization

Introduce:

* A shared **parent matrix**:
  $$
  W \in \mathbb{R}^{p \times k}, \quad k < p,
  $$
* A low-dimensional **individual vector** for each subject:
  $$
  h_i \in \mathbb{R}^k.
  $$

Subject parameters are then constructed as

$$
\theta_i = W h_i.
$$

This reduces the total number of parameters to:

$$
p \times k \;+\; k \times N
$$

which can be **much smaller** than $p \times N$ if $k$ is chosen appropriately.

The hierarchical chain becomes

$$
W, \{h_i\}_{i=1}^N \;\rightarrow\; \{\theta_i\}_{i=1}^N \;\rightarrow\; \{D_i\}_{i=1}^N.
$$

This acts as a **shared basis** (columns of $W$) with **subject-specific weights** $h_i$.

---

## 7. Autoregressive Moving Average (ARMA) Models

ARMA models are a core class of models for **stationary time series**. They are based on the idea that the current value of the series can be expressed as a combination of:

* Its **own past values** (autoregressive part).
* Past **random shocks / errors** (moving average part).

### 7.1 Motivation and Components

If, after fitting a regression model, the **residuals** exhibit autocorrelation, then important temporal structure has been missed. ARMA models aim to capture this structure.

#### 7.1.1 Autoregressive (AR) Component

**Definition (AR($p$) Process):**  
An autoregressive process of order $p$, AR($p$), is given by

$$
X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t,
$$

where $\epsilon_t$ is a white noise process, typically $\epsilon_t \sim WN(0, \sigma^2)$.

The series “regresses” on its own past values.

#### 7.1.2 Moving Average (MA) Component

**Definition (MA($q$) Process):**  
A moving average process of order $q$, MA($q$), is

$$
X_t = b_0 + \epsilon_t + \sum_{j=1}^q b_j \epsilon_{t-j}.
$$

Here $X_t$ depends on **past error terms** $\epsilon_{t-j}$, not directly on past $X_{t-j}$.

#### 7.1.3 ARMA($p,q$) Model

Combining both parts yields the ARMA($p,q$) model:

$$
X_t = c + \sum_{i=1}^p a_i X_{t-i}
      + \sum_{j=1}^q b_j \epsilon_{t-j}
      + \epsilon_t.
$$

The model parameters are

$$
\theta = \{c, a_1, \dots, a_p, b_1, \dots, b_q, \sigma^2\}.
$$

* A pure AR($p$) model corresponds to $q = 0$.
* A pure MA($q$) model corresponds to $p = 0$.

---

### 7.2 Duality and Stationarity

There is a fundamental **duality** between AR and MA processes:  
under suitable stability conditions, a finite-order AR can be represented as an infinite-order MA, and vice versa.

#### 7.2.1 Example: AR(1) as Infinite MA

Consider an AR(1) process:

$$
X_t = a_0 + a_1 X_{t-1} + \epsilon_t.
$$

We can iteratively substitute $X_{t-1}$:

$$X_t = a_0 + a_1 X_{t-1} + \epsilon_t$$
$$= a_0 + a_1 (a_0 + a_1 X_{t-2} + \epsilon_{t-1}) + \epsilon_t$$
$$= a_0(1 + a_1) + a_1^2 X_{t-2} + a_1 \epsilon_{t-1} + \epsilon_t$$
$$= a_0(1 + a_1 + a_1^2) + a_1^3 X_{t-3} + a_1^2 \epsilon_{t-2} + a_1 \epsilon_{t-1} + \epsilon_t$$
$$= a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \epsilon_{t-k}$$

This infinite expansion is valid only if the geometric series converges.

#### 7.2.2 Stationarity in the Mean for AR(1)

Take expectations:

$$
\mathbb{E}[X_t]
= a_0 \sum_{k=0}^\infty a_1^k 
  + \sum_{k=0}^\infty a_1^k \mathbb{E}[\epsilon_{t-k}].
$$

Since $\mathbb{E}[\epsilon_{t-k}] = 0$, the second term vanishes:

$$
\mathbb{E}[X_t] = a_0 \sum_{k=0}^\infty a_1^k.
$$

The geometric series converges iff $\rvert a_1\lvert < 1$, giving

$$
\mathbb{E}[X_t] = \frac{a_0}{1 - a_1}, \quad \text{if } \rvert a_1\lvert < 1.
$$

Thus, a necessary condition for **stationarity** of an AR(1) process is

$$
\rvert a_1\lvert < 1.
$$

#### 7.2.3 State-Space Representation and Stability

Any scalar AR($p$) process can be written as a **$p$-variate VAR(1)** process.

Consider

$$
X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t.
$$

Define the state vector

$$
\mathbf{X}_t =
\begin{pmatrix}
X_t \\
X_{t-1} \\
\vdots \\
X_{t-p+1}
\end{pmatrix}.
$$

Then

$$
\mathbf{X}_t = \mathbf{a} + A \mathbf{X}_{t-1} + \boldsymbol{\epsilon}_t,
$$

where

* 
  $
  \mathbf{a} =
  \begin{pmatrix}
  a_0 \\ 0 \\ \vdots \\ 0
  \end{pmatrix}
  $
* 
  $
  A =
  \begin{pmatrix}
  a_1 & a_2 & \dots & a_p \\
  1   & 0   & \dots & 0 \\
  \vdots & \ddots & \ddots & \vdots \\
  0   & \dots & 1 & 0
  \end{pmatrix}
  $
* 
  $
  \boldsymbol{\epsilon}_t =
  \begin{pmatrix}
  \epsilon_t \\ 0 \\ \vdots \\ 0
  \end{pmatrix}.
  $

The process is **stationary** if the spectral radius of $A$ is less than 1:

$$
\max_i \lvert\lambda_i(A)\rvert < 1.
$$

---

### 7.3 Model Identification via Autocorrelation

To choose orders $p$ and $q$ in ARMA($p,q$), we use:

* **Autocorrelation Function (ACF)**.
* **Partial Autocorrelation Function (PACF)**.

#### 7.3.1 Autocorrelation in AR(1)

Consider a zero-mean AR(1):

$$
X_t = a_1 X_{t-1} + \epsilon_t.
$$

Let $\gamma(k) = \text{Cov}(X_t, X_{t-k})$.

* Lag 1:
  $$
  \gamma(1) = a_1 \gamma(0).
  $$
* Lag 2:
  $$
  \gamma(2) = a_1 \gamma(1) = a_1^2 \gamma(0).
  $$
* In general:
  $$
  \gamma(k) = a_1^k \gamma(0).
  $$

Thus, the autocorrelation function

$$
\rho(k) = \frac{\gamma(k)}{\gamma(0)} = a_1^k
$$

**decays exponentially** to zero. For a general AR($p$), the ACF is a mixture of decaying exponentials / damped sinusoids.

#### 7.3.2 Autocorrelation in MA($q$)

Let $X_t$ be a zero-mean MA($q$):

$$
X_t = \epsilon_t + \sum_{j=1}^q b_j \epsilon_{t-j},
$$

with $\epsilon_t$ white noise.

For lag $k > q$, one can show $\gamma(k) = 0$, hence

$$
\text{ACF}(k) = 0 \quad \text{for all } k > q.
$$

So:

* ACF of MA($q$) **cuts off** after lag $q$.

#### 7.3.3 Partial Autocorrelation Function (PACF)

The PACF at lag $k$ is the correlation between $X_t$ and $X_{t-k}$ **after** removing the linear effect of the intervening lags.

Key property for AR($p$):

$$
\text{PACF}(k) = 0 \quad \text{for all } k > p.
$$`

So:

* AR($p$) $\Rightarrow$ **PACF cuts off** after lag $p$, ACF decays.
* MA($q$) $\Rightarrow$ **ACF cuts off** after lag $q$, PACF decays.

#### 7.3.4 Summary Heuristic

| Process | ACF | PACF |
| ------ | --- | ---- |
| AR($p$) | Decays (exponential / sinusoidal) | Cuts off after lag $p$ |
| MA($q$) | Cuts off after lag $q$ | Decays (exponential / sinusoidal) |

This heuristic is widely used in initial ARMA order selection.

---

### 7.4 Fitting and Using ARMA Models

#### 7.4.1 Parameter Estimation

For a pure AR($p$) model, estimation is equivalent to a **linear regression** problem.

Define:

* Target vector:
  $$
  y =
  \begin{pmatrix}
  X_T \\
  X_{T-1} \\
  \vdots \\
  X_{p+1}
  \end{pmatrix}
  $$
* Design matrix:
  $$
  X =
  \begin{pmatrix}
  1 & X_{T-1} & \dots & X_{T-p} \\
  1 & X_{T-2} & \dots & X_{T-p-1} \\
  \vdots & \vdots & \ddots & \vdots \\
  1 & X_p & \dots & X_1
  \end{pmatrix}.
  $$

Standard OLS yields estimates of the intercept and AR coefficients.

For ARMA($p,q$) models, parameters appear **nonlinearly** through the MA part; estimation typically uses:

* Maximum likelihood (often via numerical optimization),
* Or specialized algorithms (e.g., innovations algorithm).

#### 7.4.2 Goals of ARMA Modeling

Once an ARMA model is fitted, it can be used for:

* **Goodness-of-fit diagnostics** (check residuals for remaining structure).
* **Stationarity analysis** (via characteristic polynomial roots / eigenvalues).
* **Understanding memory and dependence** (via orders $p$ and $q$).
* **Hypothesis testing** on specific coefficients (e.g., $H_0: a_i = 0$).
* **Forecasting** future values $X_{T+1}, X_{T+2}, \dots$.
* **Control / interventions** in more advanced settings.

---

## 8. Vector Autoregressive (VAR) Models

VAR models generalize AR models to **multivariate** time series, where several variables are measured jointly over time.

### 8.1 Model Architecture

Let

$$
\mathbf{X}_t = 
\begin{pmatrix}
X_{1t} \\
X_{2t} \\
\vdots \\
X_{Nt}
\end{pmatrix}
\in \mathbb{R}^N
$$

be the vector of $N$ time series at time $t$.

**Definition (VAR($p$) Model):**

$$
\mathbf{X}_t = \mathbf{c} + \sum_{i=1}^p A_i \mathbf{X}_{t-i} + \boldsymbol{\epsilon}_t,
$$

where:

* $\mathbf{c} \in \mathbb{R}^N$ is an intercept vector,
* $A_i \in \mathbb{R}^{N \times N}$ are coefficient matrices,
* $\epsilon_t \sim WN(0, \Sigma_\epsilon)$ is a white noise vector with covariance matrix $\Sigma_\epsilon$ (not necessarily diagonal).

The structure of $A_i$ is informative:

$$
A_i =
\begin{pmatrix}
a_{11}^{(i)} & a_{12}^{(i)} & \dots & a_{1N}^{(i)} \\
a_{21}^{(i)} & a_{22}^{(i)} & \dots & a_{2N}^{(i)} \\
\vdots       & \vdots       & \ddots & \vdots \\
a_{N1}^{(i)} & a_{N2}^{(i)} & \dots & a_{NN}^{(i)}
\end{pmatrix}.
$$

* **Diagonal entries** $a_{jj}^{(i)}$: effect of past of variable $j$ on itself.
* **Off-diagonal entries** $a_{jk}^{(i)}$: effect of past of variable $k$ on variable $j$.

This is the basis for concepts like **Granger causality**.

---

### 8.2 VAR as a State-Space Model and Stationarity

Any VAR($p$) in $N$ variables can be written as a VAR(1) in $Np$ variables.

#### 8.2.1 Companion Form

Stack the lags into a state vector:

$$
\mathbf{Z}_t =
\begin{pmatrix}
\mathbf{X}_t \\
\mathbf{X}_{t-1} \\
\vdots \\
\mathbf{X}_{t-p+1}
\end{pmatrix} 
\in \mathbb{R}^{Np}.
$$

Then the VAR($p$) can be written as

$$
\mathbf{Z}_t = \mathbf{c}^\ast + A^\ast \mathbf{Z}_{t-1} + \boldsymbol{\eta}_t,
$$

for suitable companion matrix $A^\ast$ and noise vector $\boldsymbol{\eta}_t$.

#### 8.2.2 Stationarity Condition

For the VAR(1) representation

$$
\mathbf{Z}_t = \mathbf{c}^\ast + A^\ast \mathbf{Z}_{t-1} + \boldsymbol{\eta}_t,
$$

the process is **stationary** if and only if all eigenvalues of $A^\ast$ lie **inside the unit circle**:

$$
\max_i \lvert \lambda_i(A^\ast) \rvert < 1.
$$

Equivalently, all roots of the **characteristic polynomial**

$$
\det(I_N - A(z)) = 0
$$

must lie **outside** the unit circle, where $A(z)$ encodes the lag structure.

This generalizes the AR(1) condition $\lvert a_1\rvert < 1$ and the AR($p$) “roots outside unit circle” criterion to the multivariate case. -->

Here are the new notes reformatted to match the style and structure of your previous lecture notes. I have added the front matter, the CSS for the accordions, and integrated specific image tags and LaTeX formatting to ensure consistency.


## 6. Hierarchical Modeling for Multiple Time Series

When analyzing data from multiple subjects or independent sources, we often face a modeling choice between two extremes. This section introduces hierarchical modeling as a powerful intermediate approach that balances individual specificity with group-level stability.

<div class="accordion">
  <details>
    <summary>Framework of Bayesian hierarchical model</summary>
    <p>
      Let $y_j$ be an observation and $\theta_j$ a parameter governing the data generating process for $y_j$. Assume further that the parameters $\theta_1, \theta_2, \dots, \theta_j$ are generated exchangeably from a common population, with distribution governed by a hyperparameter $\phi$.
    </p>
    <p>The Bayesian hierarchical model contains the following stages:</p>
    <p><strong>Stage I:</strong> $y_j \mid \theta_j, \phi \sim P(y_j \mid \theta_j, \phi)$</p>
    <p><strong>Stage II:</strong> $\theta_j \mid \phi \sim P(\theta_j \mid \phi)$</p>
    <p><strong>Stage III:</strong> $\phi \sim P(\phi)$</p>
    <p>
      The likelihood, as seen in stage I is $P(y_j \mid \theta_j, \phi)$, with $P(\theta_j, \phi)$ as its prior distribution. Note that the likelihood depends on $\phi$ only through $\theta_j$.
    </p>
    <p>The prior distribution from stage I can be broken down into:</p>
    $$P(\theta_j, \phi) = P(\theta_j \mid \phi)P(\phi) \quad \textit{[from the definition of conditional probability]}$$
    <p>With $\phi$ as its hyperparameter with hyperprior distribution, $P(\phi)$.</p>
    <p>Thus, the posterior distribution is proportional to:</p>
    $$P(\phi, \theta_j \mid y) \propto P(y_j \mid \theta_j, \phi)P(\theta_j, \phi) \quad \textit{[using Bayes' Theorem]}$$
    $$P(\phi, \theta_j \mid y) \propto P(y_j \mid \theta_j)P(\theta_j \mid \phi)P(\phi)$$
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Example calculation of Bayesian hierarchical modeling in two variants</summary>
    <h4>Example calculation</h4>
    <p>
      As an example, a teacher wants to estimate how well a student did on the SAT. The teacher uses the current grade point average (GPA) of the student for an estimate. Their current GPA, denoted by $Y$, has a likelihood given by some probability function with parameter $\theta$, i.e. $Y \mid \theta \sim P(Y \mid \theta)$. This parameter $\theta$ is the SAT score of the student. The SAT score is viewed as a sample coming from a common population distribution indexed by another parameter $\phi$, which is the high school grade of the student (freshman, sophomore, junior or senior).$^{[14]}$ That is, $\theta \mid \phi \sim P(\theta \mid \phi)$. Moreover, the hyperparameter $\phi$ follows its own distribution given by $P(\phi)$, a hyperprior.
    </p>
    <p>These relationships can be used to calculate the likelihood of a specific SAT score relative to a particular GPA:</p>
    $$P(\theta, \phi \mid Y) \propto P(Y \mid \theta, \phi)P(\theta, \phi)$$
    $$P(\theta, \phi \mid Y) \propto P(Y \mid \theta)P(\theta \mid \phi)P(\phi)$$
    <p>
      All information in the problem will be used to solve for the posterior distribution. Instead of solving only using the prior distribution and the likelihood function, using hyperpriors allows a more nuanced distinction of relationships between given variables.$^{[15]}$
    </p>
    <h4>2-stage hierarchical model</h4>
    <p>In general, the joint posterior distribution of interest in 2-stage hierarchical models is:</p>
    $$P(\theta, \phi \mid Y) = \frac{P(Y \mid \theta, \phi)P(\theta, \phi)}{P(Y)} = \frac{P(Y \mid \theta)P(\theta \mid \phi)P(\phi)}{P(Y)}$$
    $$P(\theta, \phi \mid Y) \propto P(Y \mid \theta)P(\theta \mid \phi)P(\phi)^{[15]}$$
    <h4>3-stage hierarchical model</h4>
    <p>For 3-stage hierarchical models, the posterior distribution is given by:</p>
    $$P(\theta, \phi, X \mid Y) = \frac{P(Y \mid \theta)P(\theta \mid \phi)P(\phi \mid X)P(X)}{P(Y)}$$
    $$P(\theta, \phi, X \mid Y) \propto P(Y \mid \theta)P(\theta \mid \phi)P(\phi \mid X)P(X)^{[15]}$$
  </details>
</div>

### 1.1 Hierarchical Modeling

#### 1.1.1 Problem Formulation

Consider a scenario with $N$ subjects, where for each subject $i=1, \dots, N$, we have a time series dataset $D_i$.

  * The data for subject $i$ consists of $T_i$ time steps: $D_i = \lbrace x_{i1}, x_{i2}, \dots, x_{iT_i} \rbrace$.
  * For simplicity, we can assume all time series have the same length, $T_i = T$.
  * Each observation may also be associated with inputs: $W_i = \lbrace w_{i1}, w_{i2}, \dots, w_{iT_i} \rbrace$.

#### 1.1.2 Two Extreme Modeling Strategies

1.  **Separate Models:**
    Fit a completely independent model with parameters $\theta_i$ to each dataset $D_i$.

      * **Advantage:** This approach is excellent for capturing individual differences, as each model is tailored specifically to its own data.
      * **Disadvantage:** If the number of time steps $T$ is small and the model is complex, the estimates for $\theta_i$ can be noisy and highly prone to overfitting. There is no sharing of information across subjects.

2.  **Fully Pooled Model:**
    Concatenate all data into a single large dataset: $D_{\text{pooled}} = \lbrace D_1, D_2, \dots, D_N \rbrace$. Fit a single parameter vector $\theta$ to all the data.

      * **Advantage:** This method yields more stable parameter estimates because it leverages the entire data pool.
      * **Disadvantage:** It completely ignores inter-individual differences, assuming all subjects are governed by the exact same process.

We want to find a middle ground that finds a compromise between these extremes – **partial pooling**. This brings us to Bayesian hierarchical modeling, also known as multilevel modeling.

#### 1.1.3 The Hierarchical Approach

Hierarchical modeling provides a principled compromise between these two extremes.

**Core Idea:** We introduce subject-specific parameters $\theta_1, \theta_2, \dots, \theta_N$, but we assume they are not arbitrary. Instead, they are drawn from a common parent distribution, which is itself described by hyperparameters.

**Mathematical Formulation:**

Bayesian hierarchical modeling makes use of two important concepts in deriving the posterior distribution, namely:

* **Hyperparameters:** parameters of the prior distribution
* **Hyperpriors:** distributions of Hyperparameters

Each subject's parameter $\theta_i$ is drawn from a parent distribution parameterized by $\beta$:
$$\theta_i \sim p(\theta \mid \beta)$$

We then place a prior distribution, known as a **hyperprior**, on the hyperparameters $\beta$:
$$\beta \sim p(\beta)$$

This creates a hierarchical chain of dependencies:
$$\beta \rightarrow \theta_i \rightarrow D_i \quad \text{for } i=1, \dots, N$$

The goal of hierarchical Bayesian inference is to compute the joint posterior distribution of all subject-level parameters and group-level hyperparameters given the observed data from all subjects.

The joint distribution over all data ($D_{1:N}$), parameters ($\theta_{1:N}$), and hyperparameters ($\beta$) is given by:

$$
p(\beta, \lbrace\theta_i\rbrace_{i=1}^N, \lbrace D_i\rbrace_{i=1}^N) = p(\beta) \prod_{i=1}^N \left[ p(\theta_i \mid \beta) p(D_i \mid \theta_i) \right]
$$

Our primary target is the posterior distribution $p(\theta_1, \dots, \theta_N, \beta \mid D_1, \dots, D_N)$. This structure allows the model to "borrow statistical strength" across subjects. The data from subject $j$ informs the posterior of $\theta_j$, which in turn informs the posterior of the group hyperparameter $\beta$. This updated knowledge about $\beta$ then helps to regularize and improve the estimates for all other subjects' parameters $\theta_i$ where $i \neq j$.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/hier_model1.png' | relative_url }}" alt="hier_model1" loading="lazy">
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/hier_model2.jpg' | relative_url }}" alt="hier_model2" loading="lazy">
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/hier_model3.png' | relative_url }}" alt="hier_model3" loading="lazy">
  </figure>
</div>

### 1.2 Example: Hierarchical Delay Discounting

Let's consider a concrete example of hierarchical modeling applied to a delay discounting task in cognitive science.

#### 1.2.1 Data and Task Structure

We have data from $N$ individuals. Subject $i$ performs $T_i$ trials, indexed by $t=1, \dots, T_i$. In each trial, the subject chooses between two options:

1.  An immediate reward of value $A_{I,it}$.
2.  A delayed reward of value $A_{D,it}$ available after a delay $D_{it}$.

The observed data for each trial is a binary choice $y_{it}$:

$$
y_{it} = \begin{cases} 1 & \text{if delayed choice is selected} \\ 0 & \text{if immediate choice is selected} \end{cases}
$$

The full dataset for subject $i$ is $Y_i = \lbrace y_{it}\rbrace_{t=1}^{T_i}$.

#### 1.2.2 Model Architecture: Hyperbolic Delay Discounting

The subjective value of the options is modeled as follows:

  * **Value of Immediate Option:** $V_{I,it} = A_{I,it}$
  * **Value of Delayed Option:** The value of the delayed reward is discounted hyperbolically based on the delay duration.
    $$V_{D,it} = \frac{A_{D,it}}{1 + k_i D_{it}}$$
    Here, $k_i > 0$ is the subject-specific **discount rate**. A higher $k_i$ means the subject devalues future rewards more steeply.

The choice is then modeled based on the utility difference, $\Delta V_{it} = V_{D,it} - V_{I,it}$. A positive $\Delta V_{it}$ indicates a preference for the delayed option.

The probability of choosing the delayed option is modeled using a logistic function (i.e., logistic regression):

$$
p_{it} := P(y_{it} = 1 \mid k_i, \beta_i) = \sigma(\beta_i \Delta V_{it}) = \frac{1}{1 + \exp(-\beta_i \Delta V_{it})}
$$

where $\beta_i$ is a subject-specific "softmax" temperature parameter controlling the steepness of the decision boundary. The likelihood for the observed choices of a single individual $i$ follows a Bernoulli distribution:

$$
\mathcal{L}(k_i, \beta_i \mid Y_i) = \prod_{t=1}^{T_i} p_{it}^{y_{it}} (1 - p_{it})^{1 - y_{it}}
$$

#### 1.2.3 Hierarchical Priors

Instead of treating each $k_i$ and $\beta_i$ as independent, we impose a group-level structure. Since $k_i$ and $\beta_i$ must be positive, we model their logarithms as being drawn from a Normal distribution.

  * **Individual Parameters:** $k = \lbrace k_i\rbrace_{i=1}^N$ and $\beta = \lbrace \beta_i\rbrace_{i=1}^N$.

    $$
    \log(k_i) \sim \mathcal{N}(\mu_k, \sigma_k^2) \quad \log(\beta_i) \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)
    $$

  * **Group-level Hyperparameters:** The means $(\mu_k, \mu_\beta)$ and variances $(\sigma_k^2, \sigma_\beta^2)$ of the parent distributions.
  * **Hyperpriors:** We place priors on the hyperparameters themselves.
      * *Priors on Means:* $\mu_k \sim \mathcal{N}(\mu_{k0}, \sigma_{k0}^2)$ and $\mu_\beta \sim \mathcal{N}(\mu_{\beta 0}, \sigma_{\beta 0}^2)$.
      * *Priors on Variances:* $\sigma_k^2 \sim \text{Inverse-Gamma}(a_k, b_k)$ and $\sigma_\beta^2 \sim \text{Inverse-Gamma}(a_\beta, b_\beta)$.

#### 1.2.4 Full Bayesian Formulation

The goal is to obtain the full posterior distribution over all parameters and hyperparameters given the data $Y$:
$$P(k, \beta, \mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2 \mid Y)$$

By Bayes' theorem, this is proportional to the product of the likelihood and the priors:
$$\propto P(Y \mid k, \beta) P(k, \beta \mid \mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2) P(\mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2)$$

  * **Joint Likelihood:** Assuming independence across subjects given their parameters:
    $$P(Y \mid k, \beta) = \prod_{i=1}^N P(Y_i \mid k_i, \beta_i)$$
  * **Joint Prior on Individual Parameters:**
    $$P(k, \beta \mid \dots) = \left[ \prod_{i=1}^N P(\log(k_i) \mid \mu_k, \sigma_k^2) \right] \left[ \prod_{i=1}^N P(\log(\beta_i) \mid \mu_\beta, \sigma_\beta^2) \right]$$
  * **Joint Hyperprior:** Assuming independence of the hyperparameters:
    $$P(\mu_k, \mu_\beta, \sigma_k^2, \sigma_\beta^2) = P(\mu_k) P(\mu_\beta) P(\sigma_k^2) P(\sigma_\beta^2)$$

By combining these expressions, we define the full model. Inference is typically performed using sampling techniques like Markov Chain Monte Carlo (MCMC).

### 1.3 Alternative: Hierarchical Modeling with a Parent Matrix

An alternative parametrization for hierarchical models, particularly useful in high-dimensional settings, involves a shared parent matrix.

  * **Naive Parametrization:** Each subject $i$ has a parameter vector $\theta_i \in \mathbb{R}^p$. The total number of parameters to estimate for the subject level is $p \times N$.
  * **Parent Matrix Parametrization:** We introduce a shared parent matrix $W \in \mathbb{R}^{p \times k}$ (with $k < p$) and individual subject vectors $h_i \in \mathbb{R}^k$. The subject-specific parameter vector $\theta_i$ is then constructed as a linear combination of the columns of $W$:
    $$\theta_i = W h_i$$
    The total number of parameters is now $(p \times k) + (k \times N)$, which can be significantly smaller than $p \times N$ if $k$ is chosen well. This acts as a form of dimensionality reduction.
  * **Hierarchical Chain:** The dependency structure becomes: $W, h_i \rightarrow \theta_i \rightarrow D_i$.

#### 1.2.5 Additional Sources

* [Introduction to hierarchical modeling](https://towardsdatascience.com/introduction-to-hierarchical-modeling-a5c7b2ebb1ca/)
* [Bayesian hierarchical modeling](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling)
* [Hierarchical Modeling](https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html)

-----

## 7\. Autoregressive Moving Average (ARMA) Models

ARMA models are a fundamental class of models for analyzing stationary time series. They are built on the principle that the current value of a series can be explained by a combination of its own past values and past random shocks.

### 2.1 Motivation and Components

If the residuals of a regression model on time series data are found to be autocorrelated or cross-correlated, it implies that the model is missing important temporal structure. ARMA models are designed to capture this very structure.

  * **Autoregressive (AR) Part:** This component regresses the time series on its own past values. It captures the "memory" or persistence in the series.
  * **Moving Average (MA) Part:** This component models the current value as a function of past random perturbations or "shocks". It can be thought of as a sequence of weighted random shocks.

#### 2.1.1 The Autoregressive AR(p) Model

An **Autoregressive model of order p**, denoted **AR(p)**, is defined as:

$$
X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t
$$

where $\epsilon_t$ is a white noise process, typically $\epsilon_t \sim \mathcal{WN}(0, \sigma^2)$.

#### 2.1.2 The Moving Average MA(q) Model

A **Moving Average model of order q**, denoted **MA(q)**, is defined as:

$$
X_t = b_0 + \epsilon_t + \sum_{j=1}^q b_j \epsilon_{t-j}
$$

Note that $X_t$ depends on past **error terms**, not past values of $X$ itself.

#### 2.1.3 The ARMA(p,q) Model

Combining these two components gives the **ARMA(p,q)** model:

$$
X_t = c + \sum_{i=1}^p a_i X_{t-i} + \sum_{j=1}^q b_j \epsilon_{t-j} + \epsilon_t
$$

This can also be extended to include external inputs $u_t$. The full set of model parameters to be estimated is $\theta = \lbrace c, a_1, \dots, a_p, b_1, \dots, b_q, \sigma^2 \rbrace$.

### 2.2 Duality and Stationarity

#### 2.2.1 Duality of AR and MA Processes

There is a fundamental duality between AR and MA processes. Under certain stability conditions, any finite-order AR process can be represented as an infinite-order MA process, and vice-versa.

Let's examine this with a simple AR(1) process: $X_t = a_0 + a_1 X_{t-1} + \epsilon_t$. We can recursively expand this expression:

$$
\begin{aligned}
X_t &= a_0 + a_1(a_0 + a_1 X_{t-2} + \epsilon_{t-1}) + \epsilon_t \\
&= a_0 + a_1 a_0 + a_1^2 X_{t-2} + a_1 \epsilon_{t-1} + \epsilon_t \\
&= a_0(1 + a_1) + a_1^2 (a_0 + a_1 X_{t-3} + \epsilon_{t-2}) + a_1 \epsilon_{t-1} + \epsilon_t \\
&= a_0(1 + a_1 + a_1^2) + a_1^3 X_{t-3} + a_1^2 \epsilon_{t-2} + a_1 \epsilon_{t-1} + \epsilon_t \\
&\dots \\
&= a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \epsilon_{t-k}
\end{aligned}
$$

This infinite expansion is only valid if the series converges.

#### 2.2.2 Stationarity in the Mean for AR(1)

For the process to be stationary in the mean, its expected value must be constant and finite. Taking the expectation of the expanded form:

$$
\mathbb{E}[X_t] = \mathbb{E}\left[ a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \epsilon_{t-k} \right]
$$

$$
\mathbb{E}[X_t] = a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \mathbb{E}[\epsilon_{t-k}]
$$

Since $\mathbb{E}[\epsilon_{t-k}]=0$, the second term vanishes. The first term is a geometric series which converges if and only if $\lvert a_1 \rvert < 1$.

$$
\mathbb{E}[X_t] = \frac{a_0}{1-a_1} \quad \text{if } \lvert a_1 \rvert < 1
$$

Therefore, the condition for stationarity of an AR(1) process is $\lvert a_1 \rvert < 1$.

<div class="accordion">
<details>
<summary>Why must $\lvert a_1 \rvert < 1$?</summary>
<p>
This condition arises from the convergence of the geometric series. If $\lvert a_1 \rvert \ge 1$, the influence of past shocks ($\epsilon_{t-k}$) does not diminish over time. Instead, it persists or explodes, meaning the mean and variance of the process would not be constant, violating the definition of stationarity.
</p>
</details>
</div>

#### 2.2.3 State-Space Representation and Stability

A powerful technique for analyzing AR models is to write them in a state-space (or vector) form. Any scalar AR(p) process can be represented as a p-variate VAR(1) process.

Consider an AR(p) process $X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t$. We can define a $p$-dimensional state vector $\mathbf{X}_t$:

$$
\mathbf{X}_t = \begin{pmatrix} X_t \\ X_{t-1} \\ \vdots \\ X_{t-p+1} \end{pmatrix}
$$

The process can then be written in the form $\mathbf{X}\_t = \mathbf{a} + A \mathbf{X}\_{t-1} + \mathbf{\epsilon}\_t$, where:

$$
\mathbf{a} = \begin{pmatrix} a_0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \quad
A = \begin{pmatrix}
a_1 & a_2 & \dots & a_p \\
1 & 0 & \dots & 0 \\
\vdots & \ddots & & \vdots \\
0 & \dots & 1 & 0
\end{pmatrix}, \quad
\mathbf{\epsilon}_t = \begin{pmatrix} \epsilon_t \\ 0 \\ \vdots \\ 0 \end{pmatrix}
$$

The stability and stationarity of the entire process can then be assessed by examining the eigenvalues of the companion matrix $A$. For the process to be stationary, the spectral radius of $A$ must be less than 1.

$$
\max_i \lvert \lambda_i(A) \rvert < 1
$$

where $\lambda_i(A)$ are the eigenvalues of $A$.

<div class="accordion">
  <details>
    <summary>Why eigenvalues (not singular values) determine stationarity</summary>
    <p>
      This is a profound question that touches on the subtle difference between <strong>transient behavior</strong> (short-term) and <strong>asymptotic behavior</strong> (long-term).
    </p>
    <p>
      You are absolutely correct that Singular Value Decomposition (SVD) gives a clearer picture of how a matrix distorts space in the immediate, orthogonal sense. However, for stationarity, we rely on eigenvalues.
    </p>
    <hr>
    <h4>1. The Time Horizon: "Eventually" vs. "Immediately"</h4>
    <ul>
      <li><strong>Stationarity</strong> asks: "If I run this process for infinite time ($t \to \infty$), does it blow up?"</li>
      <li><strong>Singular values</strong> ask: "What is the maximum possible stretch this matrix causes in a <em>single</em> step?"</li>
    </ul>
    <p>A system can be stationary even if it stretches vectors significantly in the short run, provided that it eventually shrinks them back down.</p>
    <h4>2. The Math of Iteration: $A^k$</h4>
    <p>Stationary conditions usually involve iterating a transition matrix $A$. We look at the state vector evolving over time: $x_t = A^t x_0$.</p>
    <p>Let's look at what happens to the powers of the matrix using both decompositions.</p>
    <h5>The Eigendecomposition (Spectral Analysis)</h5>
    <p>
      If $A$ is diagonalizable, write $A = P \Lambda P^{-1}$. Then:
    </p>
    $$
    A^t = (P \Lambda P^{-1})(P \Lambda P^{-1}) \dots = P \Lambda^t P^{-1}
    $$
    <ul>
      <li>If $\lvert \lambda_i \rvert < 1$ for all $i$, then $\lambda_i^t \to 0$ as $t \to \infty$.</li>
      <li>Therefore, $A^t \to 0$ and the system is stable/stationary.</li>
    </ul>
    <h5>The SVD (Singular Values)</h5>
    <p>If we use SVD, $A = U \Sigma V^T$, then</p>
    $$
    A^t = (U \Sigma V^T)(U \Sigma V^T) \dots
    $$
    <ul>
      <li>$V^T U$ is <em>not</em> the identity (unless $A$ is normal/symmetric), so the rotations do not cancel.</li>
      <li>Therefore, $A^t \neq U \Sigma^t V^T$ and singular values cannot be simply raised to $t$ to predict the future.</li>
    </ul>
    <blockquote><strong>Key takeaway:</strong> Eigenvalues dictate the "fate" of the system because they survive repeated multiplication. Singular values describe the matrix right now, but that description gets scrambled during iteration.</blockquote>
    <hr>
    <h4>3. The "Strictness" Trap: Sufficient vs. Necessary</h4>
    <p>Forcing all singular values below 1 is a sufficient but not necessary condition.</p>
    <ul>
      <li><strong>$\sigma_{\text{max}} &lt; 1$</strong>: the system contracts in Euclidean length at every step (monotonic decay).</li>
      <li><strong>$\lvert \lambda_{\text{max}} \rvert &lt; 1$</strong>: the system contracts eventually; it may expand transiently before it dies out.</li>
    </ul>
    <p>If we required $\sigma_{\text{max}} &lt; 1$, we would reject many valid stationary models that merely experience transient growth.</p>
    <hr>
    <h4>4. A Concrete Counter-Example (Non-Normal Matrix)</h4>
    <p>Consider a shear matrix that is stable overall but stretches space heavily in the short term:</p>
    $$
    A = \begin{bmatrix} 0.5 & 100 \\ 0 & 0.5 \end{bmatrix}
    $$
    <p><strong>Eigenvalues ($\lambda$):</strong> Since it is upper triangular, $\lambda_1 = 0.5, \lambda_2 = 0.5$. Because $0.5 &lt; 1$, the system is stationary; $A^t \to 0$ as $t \to \infty$.</p>
    <p><strong>Singular values ($\sigma$):</strong> $\sigma_1 \approx 100$ and $\sigma_2 \approx 0.0025$, so $\sigma_{\max} \gg 1$.</p>
    <p>If we enforced $\sigma_{\max} &lt; 1$, we would incorrectly label this stable system as unstable. The matrix can grow vectors 100x in step 1, but by step 50 the $0.5^{50}$ factor dominates and the vector collapses.</p>
    <h4>Conclusion</h4>
    <p>Stationarity is an asymptotic property ($t \to \infty$). Eigenvalues track that long-run fate; singular values measure single-step gain. Singular values are great for understanding numerical stability and transient spikes, but eigenvalues are the gatekeepers of whether a process explodes or stabilizes over time.</p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Transient growth (stable eigenvalues, large singular values)</summary>
    <p>
      Here is a Python demonstration of <strong>transient growth</strong>: a system that is asymptotically stable (eigenvalues &lt; 1) but effectively unstable in the short term (singular values &gt;&gt; 1). Energy humps upward before eventually decaying.
    </p>
    <h4>The setup</h4>
    <p>Use a non-normal shear matrix:</p>
    $$
    A = \begin{bmatrix} 0.9 & 5 \\ 0 & 0.9 \end{bmatrix}
    $$
    <ul>
      <li><strong>Eigenvalues:</strong> $\lambda = 0.9$ (eventually decays to 0).</li>
      <li><strong>Singular values:</strong> $\sigma_{\max} \approx 5.3$ (can grow $5\times$ in one step).</li>
    </ul>
    <p>
      <img src="{{ '/assets/images/notes/model-based-time-series-analysis/transient_growth.png' | relative_url }}" alt="Transient growth demo: shear matrix with stable eigenvalues and large singular values" loading="lazy">
    </p>
    <h4>What the graph shows</h4>
    <h5>Phase 1: the SVD phase (steps 0–~20)</h5>
    <p>The line shoots upward; magnitude grows from about 1.0 to nearly 15.0 even though long-term decay is 0.9.</p>
    <ul>
      <li><strong>Why?</strong> The shear term (5) dominates the diagonal decay (0.9), stretching the vector into a high-gain direction.</li>
    </ul>
    <h5>Phase 2: the eigen phase (steps 20+)</h5>
    <p>The curve peaks and crashes toward zero.</p>
    <ul>
      <li><strong>Why?</strong> Once the transient shear is exhausted, repeated multiplication by 0.9 takes over; the eigenvalue constraint wins and the system stabilizes.</li>
    </ul>
    <h4>The geometric reason: non-orthogonal eigenvectors</h4>
    <p>
      In a normal (symmetric) matrix, eigenvectors are orthogonal. In this shear matrix, they are nearly parallel. To represent the initial state, you subtract two large, nearly collinear eigenvector components; as time evolves one decays slightly faster, the cancellation breaks, and the large magnitude is revealed before it decays. Singular values flag this transient spike, while eigenvalues certify eventual stability.
    </p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>How non-normal shear squeezes eigenvectors</summary>
    <p>This happens because <strong>Non-Normal</strong> matrices (matrices that do not commute with their transpose, $A^T A \neq A A^T$) contain <em>shear</em>.</p>
    <p>In a Symmetric (Normal) matrix, the transformation is pure stretching along orthogonal axes. In a Non-Normal matrix, the transformation includes a sliding or shearing motion that corrupts orthogonality. Here is the geometric and mathematical reason why this squeezes the eigenvectors together.</p>
    <h4>1. The Geometric Intuition: Stretching vs. Shearing</h4>
    <p>Imagine painting a grid on a rubber sheet and applying a matrix transformation.</p>
    <ul>
      <li><strong>Symmetric Matrix (Stretch):</strong> You pull the sheet north/south and squash it east/west. The grid lines remain at 90 degrees to each other. These lines are your eigenvectors; they are orthogonal.</li>
      <li><strong>Non-Normal Matrix (Shear):</strong> You place your hand on the top of the sheet and slide it to the right while holding the bottom fixed. The vertical lines tilt over while the horizontal lines stay horizontal.
        <ul>
          <li>One eigenvector is still horizontal.</li>
          <li>The other eigenvector (which used to be vertical) tilts to chase the shear.</li>
          <li><strong>Result:</strong> The two eigenvectors are no longer at 90 degrees; they are squeezed toward each other.</li>
        </ul>
      </li>
    </ul>
    <h4>2. The $2 \times 2$ Proof</h4>
    <p>Consider a simple upper triangular matrix and its eigenvectors:</p>
    $$
    A = \begin{bmatrix} 1 & k \\ 0 & 2 \end{bmatrix}
    $$
    <p>Here, $k$ represents the shear (non-normality).</p>
    <ul>
      <li><strong>Eigenvalues:</strong> The diagonal entries, $\lambda_1 = 1$ and $\lambda_2 = 2$.</li>
      <li><strong>Eigenvector 1 ($v_1$):</strong> Associated with $\lambda_1 = 1$, $v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (horizontal).</li>
      <li><strong>Eigenvector 2 ($v_2$):</strong> Associated with $\lambda_2 = 2$. Solving $(A - 2I)v = 0$ gives $v_2 = \begin{bmatrix} k \\ 1 \end{bmatrix}$.</li>
    </ul>
    <p><strong>Observe the angle:</strong></p>
    <ul>
      <li>$v_1$ points East $(1, 0)$.</li>
      <li>$v_2$ points North-East $(k, 1)$.</li>
    </ul>
    <p>As the shear $k$ grows (or as the difference between eigenvalues shrinks), $v_2$ tilts toward the horizontal axis.</p>
    <ul>
      <li>If $k = 100$, $v_2 = (100, 1)$, almost parallel to $v_1 = (1, 0)$.</li>
      <li>The angle between them is nearly zero—they are squeezed.</li>
    </ul>
    <h4>3. Why is this ill-conditioned? (The cancellation problem)</h4>
    <p>This squeezing creates a numerical nightmare called <strong>ill-conditioning</strong>. To represent a simple state vector, like vertical <em>Up</em> $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$, using these eigenvectors, you must use a linear combination: $x = c_1 v_1 + c_2 v_2$.</p>
    <p>If $v_1$ and $v_2$ are nearly parallel (e.g., $v_1 = [1, 0]$ and $v_2 = [1, 0.01]$), you need massive coefficients to describe a small vertical vector:</p>
    $$
    \begin{bmatrix} 0 \\ 1 \end{bmatrix} = -100 \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 100 \cdot \begin{bmatrix} 1 \\ 0.01 \end{bmatrix}
    $$
    <ul>
      <li><strong>Physically:</strong> Hidden energy; the system looks quiet but internally fights with massive opposing modes.</li>
      <li><strong>Numerically:</strong> Unstable; tiny rounding errors ruin the cancellation, causing transient growth.</li>
    </ul>
    <p>The condition number of the eigenvector matrix $\kappa(V)$ measures this. If eigenvectors are orthogonal, $\kappa(V) = 1$. As they squeeze together, $\kappa(V) \to \infty$.</p>
    <h4>Summary</h4>
    <p>Eigenvectors are directions that do not change orientation. When you apply shear (non-normality), you tilt the space. The eigenvectors tilt with it, losing their orthogonality and clamping together like a closing pair of scissors.</p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Why non-normal weights trigger exploding gradients (deep nets)</summary>
    <p>This is the exact mathematical reason why exploding gradients are so dangerous in deep learning, especially in recurrent neural networks (RNNs). In deep learning, <em>depth</em> plays the role of <em>time</em>: layer 1, layer 2, layer 3 are steps $t=1, t=2, t=3$.</p>
    <p>During backpropagation, the gradient is multiplied by the weight matrix $W$ at every layer. If $W$ acts like a shear matrix, gradients follow the same transient-growth curve: they explode in middle layers even if the network is theoretically stable.</p>
    <h4>1. The finite-time trap</h4>
    <p>Control theory cares about $t \to \infty$; deep nets care about $t \approx 50$ or $100$ (network depth).</p>
    <ul>
      <li>At step 100: values are near 0 because stable eigenvalues eventually dominate.</li>
      <li>At step 10: values are huge because unstable singular values dominate.</li>
      <li>If a network is only 10 layers deep, it lives inside that spike; it never reaches the safe asymptotic zone, so gradients blow up and updates can become NaN.</li>
    </ul>
    <h4>2. The mechanics of backpropagation</h4>
    <p>Backpropagation forms a long product of Jacobian matrices. Even if $W$ is initialized so eigenvalues are small (e.g., $\lambda = 0.9$), a non-normal $W$ can have large singular values.</p>
    <ul>
      <li>The gradient aligns with the top singular vector.</li>
      <li>The gradient grows by $\sigma_{\max}$ at every layer.</li>
      <li>Example: if $\sigma_{\max} = 5$, by layer 5 the gradient is $5^5 = 3125$ times larger, overwhelming earlier layers.</li>
    </ul>
    <h4>3. Why weight matrices are "sheared"</h4>
    <p>Randomly initialized high-dimensional matrices are rarely normal; they are typically highly non-normal, which means their eigenvectors are squeezed and ill-conditioned—perfect conditions for transient growth.</p>
    <p>This motivates <strong>orthogonal initialization</strong> ($W^T W = I$):</p>
    <ul>
      <li>For orthogonal matrices, singular values equal eigenvalues and both equal 1.</li>
      <li>No hidden shear, no transient growth; signals travel deep without exploding or vanishing.</li>
    </ul>
    <h4>Summary</h4>
    <table>
      <thead>
        <tr>
          <th>Concept</th>
          <th>In Control Theory</th>
          <th>In Deep Learning</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Goal</td>
          <td>Stability as $t \to \infty$</td>
          <td>Stability at finite depth $L$</td>
        </tr>
        <tr>
          <td>Danger</td>
          <td>Unstable eigenvalues ($|\lambda| &gt; 1$)</td>
          <td>Non-normal weights with large $\sigma_{\max}$ causing transient spikes</td>
        </tr>
        <tr>
          <td>The Trap</td>
          <td>System looks unstable initially but settles</td>
          <td>Gradient explodes before reaching the input</td>
        </tr>
        <tr>
          <td>The Fix</td>
          <td>Wait longer</td>
          <td>Gradient clipping or orthogonal init</td>
        </tr>
      </tbody>
    </table>
    <p>Want a quick visual? See “Vanishing AND Exploding Gradient Problem Explained” (video) for animations of how gradients shrink or blow up as they flow backward.</p>
    <p>Would you like an extra note here explaining gradient clipping—the brute-force way to chop off that transient spike?</p>
  </details>
</div>

### 2.3 Model Identification Using Autocorrelation

A key step in ARMA modeling is identifying the orders $p$ and $q$. The Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) are the primary tools for this task.

#### 2.3.1 Autocorrelation in AR(1) Processes

Consider a zero-mean ($a_0=0$) AR(1) process: $X_t = a_1 X_{t-1} + \epsilon_t$. The autocovariance at lag $k$, $\gamma(k)$, can be calculated.

  * **Lag 1:** $\mathbb{E}[X_t X_{t-1}] = \mathbb{E}[(a_1 X_{t-1} + \epsilon_t)X_{t-1}] = a_1 \mathbb{E}[X_{t-1}^2] + \mathbb{E}[\epsilon_t X_{t-1}]$. Since $\epsilon_t$ is uncorrelated with past values of $X$, $\mathbb{E}[\epsilon_t X_{t-1}]=0$. Thus, $\gamma(1) = a_1 \gamma(0)$.
  * **Lag 2:** $\mathbb{E}[X_t X_{t-2}] = \mathbb{E}[(a_1 X_{t-1} + \epsilon_t)X_{t-2}] = a_1 \mathbb{E}[X_{t-1} X_{t-2}] = a_1 \gamma(1) = a_1^2 \gamma(0)$.
  * **General Lag $k$:** $\gamma(k) = a_1^k \gamma(0)$.

The autocorrelation function, $\rho(k) = \gamma(k)/\gamma(0)$, is therefore $\rho(k) = a_1^k$. The ACF of an AR(1) process **decays exponentially** to zero.

#### 2.3.2 Autocorrelation in MA(q) Processes

Consider a zero-mean MA(q) process: $X_t = \epsilon_t + \sum_{j=1}^q b_j \epsilon_{t-j}$. Let's calculate the autocovariance at lag $k > q$.

Because the error terms are white noise, $\mathbb{E}[\epsilon_i \epsilon_j] = \sigma^2$ if $i=j$ and 0 otherwise. For the expectation $\mathbb{E}[X_t X_{t-k}]$ to be non-zero, there must be at least one pair of matching indices in the sums. If we consider a lag $k>q$, it is impossible to satisfy the matching index condition. Therefore, for any $k>q$, all cross-product terms have an expectation of zero.

$$
\text{ACF}(k) = 0 \quad \text{for all } k > q
$$

This provides a clear signature: the ACF of an MA(q) process **sharply cuts off** to zero after lag $q$.

#### 2.3.3 The Partial Autocorrelation Function (PACF)

The PACF at lag $k$ measures the correlation between $X_t$ and $X_{t-k}$ after removing the linear dependence on the intervening variables ($X_{t-1}, X_{t-2}, \dots, X_{t-k+1}$). A key property of the PACF for an AR(p) process is:

$$
\text{PACF}(k) = 0 \quad \text{for all } k > p
$$

This is because in an AR(p) model, the direct relationship between $X_t$ and $X_{t-k}$ (for $k>p$) is fully mediated by the first $p$ lags.

#### 2.3.4 Summary for Model Identification

| Process | Autocorrelation Function (ACF) | Partial Autocorrelation Function (PACF) |
| :--- | :--- | :--- |
| **AR(p)** | Decays exponentially or sinusoidally | Cuts off to zero after lag $p$ |
| **MA(q)** | Cuts off to zero after lag $q$ | Decays exponentially or sinusoidally |

### 2.4 Modeling with ARMA

#### 2.4.1 Parameter Estimation

For a pure AR(p) model, parameter estimation is equivalent to a linear regression problem. We can construct a predictor matrix $X$ and a target vector $y$:

$$
y = \begin{pmatrix} X_T \\ X_{T-1} \\ \vdots \\ X_{p+1} \end{pmatrix} \quad
X = \begin{pmatrix}
1 & X_{T-1} & \dots & X_{T-p} \\
1 & X_{T-2} & \dots & X_{T-p-1} \\
\vdots & \vdots & \ddots & \vdots \\
1 & X_p & \dots & X_1
\end{pmatrix}
$$

The parameters can then be estimated using ordinary least squares. For ARMA models with an MA component, estimation is more complex and typically requires numerical optimization methods like maximum likelihood estimation.

#### 2.4.2 Goals of ARMA Modeling

Once an ARMA model is fitted, it can be used for:

  * **Goodness-of-Fit:** Assess how well the model describes the temporal structure of the process.
  * **Stationarity Analysis:** Determine if the process properties are stable over time.
  * **Memory and Dependence:** The orders $p$ and $q$ define a "memory horizon."
  * **Hypothesis Testing:** Test the significance of specific coefficients (e.g., $H_0: a_i = 0$).
  * **Forecasting:** Predict future values of the time series.
  * **Control:** Understand how to steer the system towards a desired state.

-----

## 3\. Vector Autoregressive (VAR) Models

Vector Autoregressive (VAR) models are a direct extension of the univariate AR models to multivariate time series.

### 3.1 Model Architecture

A VAR model is used for analyzing a set of $N$ time series variables recorded simultaneously.

  * **Data:** At each time point $t$, we have a vector $\mathbf{X}\_t = (X\_{1t}, X\_{2t}, \dots, X\_{Nt})^\top \in \mathbb{R}^N$.
  * **New Phenomenon:** The primary interest in VAR modeling is to capture not only the autocorrelation within each series but also the **cross-correlation** (or cross-covariance) between different series.

A **VAR(p) model** is defined as:

$$
\mathbf{X}_t = \mathbf{c} + \sum_{i=1}^p A_i \mathbf{X}_{t-i} + \mathbf{\epsilon}_t
$$

  * $\mathbf{c}$ is an $N \times 1$ intercept vector.
  * $A_i$ are $N \times N$ coefficient matrices for each lag $i$.
  * $\mathbf{\epsilon}\_t$ is an $N \times 1$ white noise vector process with mean zero and a covariance matrix $\Sigma_\epsilon$. Importantly, the covariance matrix $\Sigma_\epsilon$ is generally not diagonal, allowing for contemporaneous correlation between the shocks to different variables.

The structure of a coefficient matrix $A_i$ is informative:

$$
A_i = \begin{pmatrix}
a_{11}^{(i)} & a_{12}^{(i)} & \dots & a_{1N}^{(i)} \\
a_{21}^{(i)} & a_{22}^{(i)} & \dots & a_{2N}^{(i)} \\
\vdots & \vdots & \ddots & \vdots \\
a_{N1}^{(i)} & a_{N2}^{(i)} & \dots & a_{NN}^{(i)}
\end{pmatrix}
$$

  * **Diagonal entries** ($a_{jj}^{(i)}$) relate the past of variable $j$ to its own current value.
  * **Off-diagonal entries** ($a_{jk}^{(i)}$) quantify how the past of variable $k$ influences the current value of variable $j$. This is the basis for concepts like Granger causality.

### 3.2 State-Space Representation and Stationarity

Similar to the univariate case, VAR(p) models can be compactly represented as VAR(1) models. Any VAR(p) process in $N$ variables can be written as an $Np$-variate VAR(1) process by stacking the lagged vectors into a larger state vector.

For a VAR(1) process defined by $\mathbf{X}\_t = \mathbf{c} + A \mathbf{X}\_{t-1} + \mathbf{\epsilon}\_t$, a necessary and sufficient condition for stationarity is that all eigenvalues of the matrix $A$ have a modulus less than 1.

$$
\max_i \lvert\lambda_i(A)\rvert < 1
$$

where $\lambda_i$ are the eigenvalues of $A$. This is equivalent to saying all roots of the characteristic polynomial $\det(I_N - Az) = 0$ lie outside the unit circle.


[Fourier Transform](/subpages/model-based-time-series-analysis/fourier_transform/)

## Chapter 1: Vector Autoregressive (VAR) Models

### 1.1 Introduction to Multivariate Time Series

In contrast to univariate analysis, multivariate time series analysis considers datasets where multiple variables are recorded simultaneously over time.

- **Multivariate Time Series Data:** A vector $X_t$ representing observations at time $t$. $X_t = (X_{1t}, \dots, X_{Nt})^T \in \mathbb{R}^N$. Here, $N$ is the number of simultaneously recorded variables.
- **New Phenomena of Interest:** The primary advantage of the multivariate approach is the ability to model interactions between time series. A key phenomenon is the cross-correlation between the different component series.

### 1.2 The VAR($p$) Model Architecture

The Vector Autoregressive (VAR) model is a natural extension of the univariate autoregressive (AR) model to multivariate time series. A VAR($p$) model expresses each variable as a linear function of its own past values, the past values of all other variables in the system, and a random error term.

The general form of a VAR($p$) model is:

$$
X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t
$$

- **Intercept:** $c \in \mathbb{R}^N$ is the intercept vector.
- **Coefficient Matrices:** $A_i \in \mathbb{R}^{N \times N}$ are the coefficient matrices for each lag $i=1, \dots, p$.
- **Error Term:** $\varepsilon_t$ is a vector of white noise error terms, typically assumed to be multivariate normal: $\varepsilon_t \sim \mathcal{N}(0, \Sigma_\varepsilon)$.

The covariance matrix of the error term, $\Sigma_\varepsilon$, is given by:

$$
\Sigma_\varepsilon = E[\varepsilon_t \varepsilon_t^T] =
\begin{pmatrix}
  E[\varepsilon_{1t}^2] & E[\varepsilon_{1t}\varepsilon_{2t}] & \dots \\
  \vdots & \ddots & \vdots \\
  E[\varepsilon_{Nt}\varepsilon_{1t}] & \dots & E[\varepsilon_{Nt}^2]
\end{pmatrix}
$$

Crucially, the off-diagonal elements of $\Sigma_\varepsilon$ are allowed to be non-zero, meaning the contemporaneous error terms for different variables can be correlated.

#### Structure of Coefficient Matrices

Each coefficient matrix $A_i$ captures the influence of variables at lag $i$ on the current state of the system.

$$
A_i = \begin{pmatrix}
a_{11}^{(i)} & \dots & a_{1N}^{(i)} \\
\vdots & \ddots & \vdots \\
a_{N1}^{(i)} & \dots & a_{NN}^{(i)}
\end{pmatrix}
$$

- **Diagonal Entries ($a_{jj}^{(i)}$):** These entries relate a variable to its own past. They capture the internal time constants and autoregressive properties of each individual series.
- **Off-Diagonal Entries ($a_{jk}^{(i)}$ for $j \neq k$):** These entries quantify how the past of variable $k$ influences the present of variable $j$. They are the key to understanding the cross-series dynamics and interactions.

### 1.3 Equivalence and Companion Form

A significant theoretical result is that any VAR($p$) process can be rewritten as a VAR(1) process. This is extremely useful for analysis, particularly for assessing model stability.

1. Any scalar AR($p$) process can be written as a p-variate VAR(1) process.
2. Any VAR($p$) process in $K$ variables can be written as a $Kp$-variate VAR(1) process.

This transformation allows us to study the stability and properties of a high-order model by analyzing a single, larger coefficient matrix corresponding to the VAR(1) representation.

### 1.4 Stationarity of VAR Processes

The stability of a VAR process is determined by the properties of its coefficient matrices. For a VAR(1) process, the condition for stationarity is based on the eigenvalues of the coefficient matrix.

For the VAR(1) process $X_t = c + AX_{t-1} + \varepsilon_t$, a necessary and sufficient condition for stationarity is that all eigenvalues of the matrix $A$ have a modulus less than 1.

$$
\max_i(\rvert\lambda_i(A)\lvert) < 1
$$

where $\lambda_i$ are the eigenvalues of $A$.

#### Proof Sketch for Stationarity

1. **Iterative Substitution:** Consider the process without the intercept and noise terms for simplicity: $X_t = AX_{t-1}$. By iterating backwards, we can express $X_t$ in terms of an initial state $X_0$:
   
   $$
   X_t = A X_{t-1} = A (A X_{t-2}) = A^2 X_{t-2} = \dots = A^t X_0
   $$

2. **Eigendecomposition:** We can decompose the matrix $A$ into its eigenvalues and eigenvectors: $A = V \Lambda V^{-1}$ where $\Lambda$ is a diagonal matrix containing the eigenvalues $\lambda_i$, and $V$ is the matrix of corresponding eigenvectors.
3. **Power of $A$:** Using the eigendecomposition, the $t$-th power of $A$ is:
   
   $$
   A^t = (V \Lambda V^{-1})^t = (V \Lambda V^{-1})(V \Lambda V^{-1}) \dots = V \Lambda^t V^{-1}
   $$

4. **System Evolution:** Substituting this back into the expression for $X_t$:
   
   $$
   X_t = V \Lambda^t V^{-1} X_0
   $$

5. **Condition for Stability:** The system is stable (i.e., stationary) if $X_t \to 0$ as $t \to \infty$. This requires that $A^t \to 0$. This, in turn, depends on the behavior of $\Lambda^t$, which is a diagonal matrix with entries $\lambda_i^t$.
   - If $\max_i(\lvert\lambda_i\rvert) < 1$, then all $\lambda_i^t \to 0$ as $t \to \infty$. Consequently, $\Lambda^t \to 0$, $A^t \to 0$, and the process is stable and stationary.
   - If $\max_i(\lvert\lambda_i\rvert) > 1$, at least one eigenvalue has a modulus greater than 1. Its corresponding term $\lambda_i^t$ will grow exponentially, causing $X_t$ to explode along the direction of the corresponding eigenvector. The process is non-stationary (divergent).
   - If $\max_i(\lvert\lambda_i\rvert) = 1$, the system is marginally stable. This can lead to behaviors like a random walk.

### 1.5 Parameter Estimation

The parameters of a VAR($p$) model $(c, A_1, \dots, A_p, \Sigma_\varepsilon)$ can be estimated using Maximum Likelihood Estimation (MLE), which in the case of Gaussian errors is equivalent to multivariate least squares.

Given a VAR($p$) model:

$$
X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \Sigma_\varepsilon)
$$

The conditional probability density of $X_t$ given its past is:

$$
p(X_t \mid X_{t-1}, \dots, X_{t-p}, \theta) = \mathcal{N}\left(c + \sum_{i=1}^{p} A_i X_{t-i}, \Sigma_\varepsilon\right)
$$

where $\theta$ represents all model parameters.

The log-likelihood for a sequence of observations $X_1, \dots, X_T$ is then:

$$
\ell(\theta) = \sum_{t=p+1}^{T} \log p(X_t \mid X_{t-1}, \dots, X_{t-p}, \theta)
$$

Maximizing this log-likelihood function provides the parameter estimates. This can be framed as a multivariate linear regression problem.

### 1.6 Model Order Selection

A critical step in VAR modeling is determining the appropriate order, $p$. This is a model selection problem where we aim to balance model fit with model complexity. Common criteria include AIC and BIC, but a formal statistical test is the Likelihood Ratio Test.

#### Likelihood Ratio Test (Wilks' Theorem)

The Likelihood Ratio (LR) test provides a framework for comparing nested models.

- Let $\mathcal{M}_0$ be a "restricted" model (null hypothesis, $H_0$) with parameter space $\Theta_0$.
- Let $\mathcal{M}_1$ be a "full" model (alternative hypothesis, $H_1$) with parameter space $\Theta_1$, where $\Theta_0 \subset \Theta_1$.
- Let $\ell_{max}(\mathcal{M}_0)$ and $\ell_{max}(\mathcal{M}_1)$ be the maximized log-likelihoods for each model.

The LR test statistic is defined as:

$$
D = -2 \log \left(\frac{\sup_{\theta \in \Theta_0} \mathcal{L}(\theta)}{\sup_{\theta \in \Theta_1} \mathcal{L}(\theta)}\right) = -2 (\ell_{max}(\mathcal{M}_0) - \ell_{max}(\mathcal{M}_1))
$$

Under suitable regularity conditions and assuming $H_0$ is true, the statistic $D$ follows a chi-squared distribution:

$$
D \sim \chi^2(d_1 - d_0)
$$

where $d_1$ and $d_0$ are the number of free parameters in the full and restricted models, respectively.

**Decision Rule:** We compare the empirically observed statistic $D_{empirical}$ to the $\chi^2$ distribution. If the probability of observing a value as large as $D_{empirical}$ is small (e.g., $p < \alpha$, where $\alpha=0.05$ by convention), we reject the null hypothesis $H_0$ in favor of the more complex model $\mathcal{M}_1$.

#### Application to VAR($p$) vs. VAR($p+1$)

We can use the LR test to decide if a VAR($p+1$) model provides a significantly better fit than a VAR($p$) model.

- **Restricted Model ($H_0$):** VAR($p$) process, $X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t$. This is equivalent to a VAR(p+1) model with the constraint $A_{p+1} = 0$.
- **Full Model ($H_1$):** VAR($p+1$) process, $X_t = c + \sum_{i=1}^{p+1} A_i X_{t-i} + \varepsilon_t$.

The LR test compares the "explained variation" in the data under both models. The maximized log-likelihood is related to the determinant of the estimated residual covariance matrix, $\hat{\Sigma}_\varepsilon$.

$$
\ell_{max} \propto -\frac{T_{eff}}{2} \log(\lvert\hat{\Sigma}_\varepsilon\rvert)
$$

The test statistic becomes:

$$
D = T_{eff} \left(\log(\lvert\hat{\Sigma}_{restr}\rvert) - \log(\rvert\hat{\Sigma}_{full}\lvert)\right)
$$

This statistic is compared to a $\chi^2(N^2)$ distribution, as the VAR(p+1) model has $N^2$ additional free parameters in the matrix $A_{p+1}$.

### 1.7 Granger Causality

Granger causality is a statistical concept of causality based on prediction. It provides a formal method to test for directed influence between time series within the VAR framework.

- **Definition:** If the past of a time series $X$ contains information that improves the prediction of a time series $Y$, beyond the information already contained in the past of $Y$ and all other relevant variables, then we say that $X$ Granger-causes $Y$ (denoted $X \to Y$).

Let $X_t, Y_t$ be two time series processes and $Z_t$ represent all other knowledge in the world. Let $\mathcal{E}[Y_{t+1} \mid \text{past}]$ denote the optimal prediction of $Y_{t+1}$ given information from the past. Then $X \to Y$ if:

$$
\mathcal{E}[Y_{t+1} \mid Y_{t-\text{past}}, X_{t-\text{past}}, Z_{t-\text{past}}] \neq \mathcal{E}[Y_{t+1} \mid Y_{t-\text{past}}, Z_{t-\text{past}}]
$$

#### Testing for Granger Causality with VAR Models

To make this concept testable, Granger proposed embedding it within a VAR model. Consider a bivariate system $(X_t, Y_t)$. To test if $X \to Y$, we set up two nested models:

1. **Full Model:** A VAR($p$) model where past values of $X$ are used to predict $Y$. In the equation for $Y_t$, the coefficients on lagged $X_t$ are unrestricted.
2. **Restricted Model:** A VAR($p$) model where the influence of past $X$ on $Y$ is removed. This is achieved by setting all coefficients that link lagged $X_t$ to $Y_t$ to zero.

We then perform a likelihood ratio test (or an F-test) comparing these two models.

- Let $\hat{\Sigma}_{full}$ and $\hat{\Sigma}_{restr}$ be the estimated residual covariance matrices.
- The LR test statistic is: $D = T_{eff} (\log(\lvert\hat{\Sigma}_{restr}\rvert) - \log(\lvert\hat{\Sigma}_{full}\rvert))$.
- Under $H_0$ (no Granger causality), $D \sim \chi^2(q)$, where $q$ is the number of zero-restrictions imposed (in this case, $p \times (\text{dim of } X) \times (\text{dim of } Y)$).

**Interpretation:** If adding the past of $X$ significantly reduces the residual covariance (i.e., the prediction error) for $Y$, then the test statistic will be large, leading to a rejection of the null hypothesis. We conclude that $X$ Granger-causes $Y$.

#### Caveats in Interpretation

It is crucial to interpret "Granger causality" with care, as it is a statement about predictive power, not necessarily true causal influence.

- **Hidden Common Causes:** If an unobserved variable $Z$ drives both $X$ and $Y$, a spurious Granger-causal relationship $X \to Y$ might be detected.
- **Linearity Assumption:** The standard test is based on linear VAR models and only detects linear forms of dependence. The general definition of Granger causality is not restricted to linear relationships.
- **Gaussian Assumptions:** The statistical properties of the LR test rely on the assumption of Gaussian-distributed residuals. Strong deviations from this assumption may invalidate the test statistics.

## Chapter 2: Generalized Autoregressive Models

This chapter extends the autoregressive framework to model non-Gaussian time series, such as binary sequences or count data, using the principles of Generalized Linear Models (GLMs).

### 2.1 AR Models for Binary Processes

Consider a time series of binary outcomes, $X_t \in \lbrace 0, 1\rbrace$. We can model this using an autoregressive structure similar to logistic regression.

- **Data:** A binary time series, e.g., $X_t = \lbrace 0, 1, 1, 0, \dots\rbrace$.
- **Model Architecture:** The probability of a "success" $(X_t=1)$ is conditioned on the past $p$ values.
- 
  $$
  X_t \mid X_{t-1}, \dots, X_{t-p} \sim \text{Bernoulli}(\pi_t)
  $$

  where $\pi_t = P(X_t = 1 \mid \text{past})$.
- **Linear Predictor:** The probability $\pi_t$ is related to a linear combination of past observations through a link function (typically the logit function). The linear predictor $\eta_t$ is defined as:
- 
  $$
  \eta_t = c + \sum_{i=1}^{p} \alpha_i X_{t-i}
  $$
  
- **Link Function:** The relationship between $\eta_t$ and $\pi_t$ is given by the inverse link function (sigmoid or logistic function):
- 
  $$
  \pi_t = \frac{e^{\eta_t}}{1 + e^{\eta_t}} = \frac{1}{1 + e^{-\eta_t}}
  $$

  This is analogous to logistic regression, but the predictors are now the lagged values of the time series itself.
- **Training:** Model parameters are estimated by maximizing the likelihood, which does not have a closed-form solution. Numerical optimization methods like Gradient Descent or Newton-Raphson are required.

#### Main Limitations of Binary AR Models

- **Unstable Estimates:** For strongly dependent series, certain histories (e.g., a long string of 1s) may lead to perfect prediction, causing the MLE estimates for coefficients to become very large or non-existent.
- **Limited Dynamical Structure:** These models may struggle to capture complex temporal patterns like "bursts" of activity versus periods of "silence".
- **Scalability:** Extending this framework to the multivariate case is challenging.
- **Numerical Inference:** Inference about the model parameters is purely numerical.

A common remedy for instability is to reduce the model order $p$, thereby reducing complexity.

### 2.2 Change Point Models for Binary Time Series

An alternative approach for binary series is to model the success probability $\pi_t$ as an explicit function of time, allowing for a single change point.

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
- **Likelihood:** The likelihood of the observed data given the parameters is the product of Bernoulli probabilities:
  
  $$
  P(X \mid \theta) = \prod_{t=1}^{T} \pi_t^{X_t} (1-\pi_t)^{1-X_t}
  $$

  The log-likelihood is $\ell(\theta) = \sum_{t=1}^{T} [X_t \log(\pi_t) + (1-X_t)\log(1-\pi_t)]$. These parameters are typically found via MLE.
- **Challenge:** This specific sigmoid structure only allows for a single, monotonic change point in the success probability.

### 2.3 AR Models for Count Processes (Poisson GLM)

For time series of counts, $C_t \in \lbrace 0, 1, 2, \dots\rbrace$, we can use a Poisson distribution where the rate is modeled with an autoregressive structure.

- **Data:** A univariate or multivariate count process. For example, the number of customers entering a store per hour, or the number of spikes from $N$ neurons in discrete time bins. $C_t = (C_{1t}, \dots, C_{Nt})^T$.
- **Model:** For each process $i=1, \dots, N$, we model the count $C_{it}$ conditioned on the past as a Poisson random variable.
  
  $$
  C_{it} \mid \text{past} \sim \text{Poisson}(\lambda_{it})
  $$

  The Poisson PMF implies that $E[C_{it} \mid \text{past}] = \text{Var}(C_{it} \mid \text{past}) = \lambda_{it}$. The rate $\lambda_{it}$ determines both the mean and variance.
- **Rate Model:** We model the vector of rates $\lambda_t = (\lambda_{1t}, \dots, \lambda_{Nt})^T$ using an AR structure with an exponential link function to ensure positivity of the rates.
  
  $$
  \lambda_t = \exp\left(c + \sum_{j=1}^{p} A_j C_{t-j}\right)
  $$

  The $\exp$ is the inverse link function in this Poisson GLM. The coefficient matrices $A_j$ capture the influence of past counts on current rates, representing "effective couplings" between processes.
- **Maximum Likelihood Estimation:** The log-likelihood, assuming conditional independence of the individual processes given the past, is:
  
  $$
  \ell(\{c, A_j\}) = \sum_{t=p+1}^{T} \sum_{i=1}^{N} \log P(C_{it} \mid \text{past})
  $$

  Substituting the Poisson PMF, $\log P(C_{it}) = C_{it} \log(\lambda_{it}) - \lambda_{it} - \log(C_{it}!)$, and our model for $\lambda_{it}$:
  
  $$
  \ell(\lbrace c, A_j\rbrace) = \sum_{t, i} \left[ C_{it} \left(c_i + \sum_{j,k} (A_j)_{ik} C_{k,t-j}\right) - \exp\left(c_i + \sum_{j,k} (A_j)_{ik} C_{k,t-j}\right) \right] + \text{const}
  $$

  Although there is no closed-form solution for the parameters, the Poisson log-likelihood function is concave. This is a significant advantage, as it guarantees that standard numerical optimization methods (like Gradient Descent) will converge to the unique global maximum.

## Chapter 3: Nonlinear Dynamical Systems

This chapter introduces the fundamental concepts of nonlinear dynamical systems, moving beyond the linear framework of VAR models to explore more complex behaviors like fixed points, cycles, and chaos.

### 3.1 Motivation: From Linear to Nonlinear Systems

A VAR(1) model, $X_t = c + AX_{t-1}$, is a Linear Dynamical System (LDS). We can generalize this by replacing the linear function with a nonlinear function $F(\cdot)$, such as a recurrent neural network (RNN).

$$
X_t = F(X_{t-1})
$$

This defines a nonlinear dynamical system. Understanding the behavior of simple nonlinear systems provides the foundation for analyzing more complex models.

### 3.2 Analysis of 1D Systems

Let's start with a simple first-order nonlinear difference equation: $x_{t+1} = f(x_t)$.

#### Fixed Points

A central concept is the fixed point (FP), a state where the system remains unchanged over time.

- **Definition:** A point $x^*$ is a fixed point of the map $f$ if it satisfies the equation:
  
  $$
  x^* = f(x^*)
  $$

- **Example (Linear AR(1)):** For $f(x) = \alpha x + c$, the fixed point equation is $x^* = \alpha x^* + c$.
  - Solving for $x^*: x^*(1-\alpha) = c$, which gives $x^* = \dfrac{c}{1-\alpha}$ (provided $\alpha \neq 1$).

#### Stability of Fixed Points

The behavior of the system near a fixed point determines its stability.

- A stable fixed point, or attractor, is a point such that trajectories starting in its vicinity converge to it.
  - Formally, $x^*$ is an attractor if there exists a set $C$, called the basin of attraction, such that for any initial condition $x_0 \in C$, $\lim_{t \to \infty} x_t = x^*$.
- An unstable fixed point, or repeller, is a point from which nearby trajectories diverge.

For the linear system $x_{t+1} = \alpha x_t + c$, the fixed point $x^*$ is:

- Stable if $\lvert\alpha\rvert < 1$. Trajectories converge to $x^*$.
- Unstable if $\lvert\alpha\rvert > 1$. Trajectories diverge from $x^*$.
- Neutrally stable if $\lvert\alpha\rvert = 1$.

#### Cycles

A system may not settle on a single point but may instead visit a set of points periodically.

- **$p$-cycle:** A set of $p$ distinct points $\lbrace x_1^*, \dots, x_p^*\rbrace$ that the system visits in sequence: $f(x_1^*) = x_2^*, \dots, f(x_p^*) = x_1^*$.
- A point on a $p$-cycle is a fixed point of the $p$-th iterated map, $f^p(\cdot)$. That is, $x^* = f^p(x^*)$.
- **Example (2-cycle):** For the map $x_{t+1} = -x_t + c$, if we start at $x_0$, then $x_1 = -x_0+c$, and $x_2 = -x_1+c = -(-x_0+c)+c = x_0$. Every point is part of a 2-cycle. This is a neutrally stable cycle.

#### Summary of 1D Linear System Behaviors

1. Convergence to a solitary, stable fixed point (attractor) for $\lvert\alpha\rvert < 1$.
2. Divergence from an isolated, unstable fixed point for $\lvert\alpha\rvert > 1$.
3. An infinite set of neutrally stable fixed points (e.g., a line) if $\alpha=1$, $c=0$.
4. No fixed point or cycle (linear drift) if $\alpha=1$, $c \neq 0$.
5. An infinite set of neutrally stable cycles if $\alpha=-1$.

#### Multivariate Extension (Linear Case)

For a multivariate linear system $X_t = AX_{t-1} + c$ where $X_t \in \mathbb{R}^N$:

- **Fixed Point:** $X^* = AX^* + c \implies (I-A)X^* = c \implies X^* = (I-A)^{-1}c$, provided $(I-A)$ is invertible.
- **Stability:** The stability of $X^*$ is determined by the eigenvalues of the Jacobian matrix, which is simply $A$.
  1. $X^*$ is a stable fixed point if $\max_i(\lvert\lambda_i(A)\rvert) < 1$.
  2. $X^*$ is unstable if $\max_i(\lvert\lambda_i(A)\rvert) > 1$.
  3. $X^*$ is neutrally stable if $\max_i(\lvert\lambda_i(A)\rvert) = 1$.

### 3.3 The Logistic Map: A Case Study in Nonlinearity

The logistic map is a simple, archetypal example of a nonlinear system that exhibits complex behavior, including chaos.

- **Equation:**
  
  $$
  x_{t+1} = f(x_t) = \alpha x_t (1 - x_t)
  $$

- **Constraints:** We consider initial conditions $x_0 \in [0, 1]$ and the parameter $\alpha \in [0, 4]$. These constraints ensure that if $x_t$ is in $[0, 1]$, then $x_{t+1}$ will also be in $[0, 1]$.

#### Fixed Points of the Logistic Map

We solve $x^* = f(x^*) = \alpha x^* (1 - x^*)$.

$$
x^* - \alpha x^* + \alpha (x^*)^2 = 0 \implies x^*(1 - \alpha + \alpha x^*) = 0
$$

This gives two fixed points:

1. FP1: $x_1^* = 0$.
2. FP2: $1 - \alpha + \alpha x^* = 0 \implies x_2^* = \dfrac{\alpha - 1}{\alpha}$. This fixed point is only physically relevant (i.e., in our state space $[0,1]$) when $\alpha \ge 1$.

#### Formal Stability Analysis via Linearization

To analyze the stability of a fixed point $x^*$ for a general nonlinear map $f(x)$, we consider a small perturbation $\varepsilon_t$ around the fixed point: $x_t = x^* + \varepsilon_t$.

$$
x_{t+1} = x^* + \varepsilon_{t+1} = f(x^* + \varepsilon_t)
$$

Using a first-order Taylor expansion of $f(x)$ around $x^*$:

$$
f(x^* + \varepsilon_t) \approx f(x^*) + f'(x^*) \cdot \varepsilon_t
$$

Since $f(x^*) = x^*$, this simplifies to:

$$
x^* + \varepsilon_{t+1} \approx x^* + f'(x^*) \cdot \varepsilon_t \implies \varepsilon_{t+1} \approx f'(x^*) \cdot \varepsilon_t
$$

This is a linear difference equation for the perturbation $\varepsilon_t$. The perturbation will decay to zero (i.e., the FP is stable) if $\lvert f'(x^*)\rvert < 1$.

- **Applying to the Logistic Map:** The derivative is $f'(x) = \alpha(1-2x)$.
  1. At FP1 ($x_1^*=0$): $f'(0) = \alpha(1-0) = \alpha$.
     - The fixed point at 0 is stable if $\lvert f'(0)\rvert < 1$, which means $0 \le \alpha < 1$.
  2. At FP2 ($x_2^* = (\alpha-1)/\alpha$): $f'(x_2^*) = \alpha\left(1 - 2\frac{\alpha-1}{\alpha}\right) = \alpha\left(\frac{\alpha - 2\alpha + 2}{\alpha}\right) = 2-\alpha$.
     - This fixed point is stable if $\lvert f'(x_2^*)\rvert = \lvert 2-\alpha\rvert < 1$. This inequality holds for $1 < \alpha < 3$.
- **Stability Summary:**
  - For $0 \le \alpha < 1$: One stable FP at $x^*=0$.
  - For $1 < \alpha < 3$: The FP at $x^*=0$ becomes unstable, and a new stable FP appears at $x^*=(\alpha-1)/\alpha$.

#### Multivariate Linearization

For a multivariate system $X_t = F(X_{t-1})$ with fixed point $X^* = F(X^*)$, the stability is determined by linearizing around $X^*$. The evolution of a small perturbation $\mathcal{E}_t$ is governed by the Jacobian matrix $J$ of $F$ evaluated at $X^*$.

$$
\mathcal{E}_{t+1} \approx J(X^*) \mathcal{E}_t
$$

The fixed point $X^*$ is stable if all eigenvalues of the Jacobian matrix have a modulus less than 1: $\max_i(|\lambda_i(J(X^*))|) < 1$.

### 3.4 Bifurcation and Chaos

As the parameter $\alpha$ increases beyond 3 in the logistic map, the system's behavior undergoes a series of qualitative changes known as bifurcations.

- **Period-Doubling:** At $\alpha=3$, the fixed point $x_2^*$ becomes unstable, and a stable 2-cycle emerges. As $\alpha$ increases further, this 2-cycle becomes unstable and bifurcates into a stable 4-cycle, then an 8-cycle, and so on. This cascade is known as the "period-doubling route to chaos."
- **Chaos:** For larger values of $\alpha$ (e.g., $\alpha \gtrsim 3.57$), the system's behavior becomes chaotic.
  - The trajectory is aperiodic and appears irregular or random, but it is still fully deterministic.
  - A key feature is sensitivity to initial conditions: two trajectories starting arbitrarily close to each other will diverge exponentially fast.

#### Chaotic Attractors and the Lyapunov Exponent

- **Chaotic Attractor:** A set $A$ is a chaotic attractor if trajectories starting within its basin of attraction converge to $A$, and within $A$, the dynamics are chaotic and sensitive to initial conditions.
- **Lyapunov Exponent ($\lambda$):** This value quantifies the rate of separation of infinitesimally close trajectories. For a 1D map $f(x)$, it is defined as:
  
  $$
  \lambda(x_0) = \lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \ln\lvert f'(x_t)\rvert
  $$

  - $\lambda > 0$: Exponential divergence, a signature of chaos.
  - $\lambda < 0$: Exponential convergence, corresponding to a stable fixed point or cycle.

### 3.5 Implications for Prediction and Modeling

The existence of chaotic dynamics has profound implications for time series modeling:

1. Prediction Horizon: Sensitivity to initial conditions makes long-term prediction fundamentally impossible, as any tiny error in measuring the current state will be exponentially amplified.
2. Chaos vs. Noise: It can be extremely difficult to distinguish between a deterministic chaotic process and a stochastic (noisy) process based on observed data alone.
3. Loss Functions: Traditional loss functions like Mean Squared Error (MSE) may be problematic for evaluating models of chaotic systems, as even a perfect model will produce trajectories that diverge from the data due to initial condition uncertainty.
4. Parameter Estimation: The loss landscapes for models of chaotic systems can be highly non-convex and irregular, making optimization and parameter estimation very challenging.
