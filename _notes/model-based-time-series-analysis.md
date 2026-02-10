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

## Review on Statistical Inference

To reason about uncertainty in a mathematically sound way, we begin with the concept of a probability space. This structure consists of three essential components that formalize an experiment and its outcomes.

### The Probability Space

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probability Space, Experiment)</span></p>

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

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Die Toss)</span></p>

  * $\Omega = \lbrace 1, 2, 3, 4, 5, 6 \rbrace$.
  * $\mathcal{A} = \mathcal{P}(\Omega)$ or $\mathcal{A} = \lbrace \emptyset, \lbrace 1,3,5 \rbrace, \lbrace 2,4,5 \rbrace, \Omega \rbrace$.

</div>

### Random Variables and Probability Distributions

Random variables allow us to map outcomes from the sample space to the real numbers, enabling the use of powerful mathematical tools.

<!-- ### 2.1 Random Variables -->

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Random Variable)</span></p>

A **Random Variable** (RV) is a measurable function $X: \Omega \to \mathbb{R}$ that assigns a real number to each outcome in the sample space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Coin Toss)</span></p>

For a coin toss where $\Omega = \lbrace \text{Heads}, \text{Tails} \rbrace$, we can define an RV $X$ such that $X(\text{Heads}) = 1$ and $X(\text{Tails}) = 0$.

</div>

The mapping itself is deterministic; the randomness is induced by the underlying probability measure on $\Omega$. Random variables induce probability distributions.

The **Cumulative Distribution Function (CDF)** uniquely characterizes the distribution of any random variable.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cumulative Distribution Function)</span></p>

A function $F: \mathbb{R} \to [0, 1]$ is a CDF if it satisfies:

1. $0 \le F(x) \le 1$ for all $x \in \mathbb{R}$.
2. $F$ is non-decreasing.
3. $F$ is right-continuous.
4. $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$.

The CDF connects directly to the probability measure: $F(x) = P(X \le x)$.

</div>

### Discrete Distributions and PMFs

For a discrete RV, the probability is concentrated on a countable set of points.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probability Mass Function)</span></p>

Let $a_1, a_2, \dots$ be the values a discrete RV $X$ can take. A function $f: \mathbb{R} \to [0, 1]$ is a PMF if $f(x) = P(X=x)$. Specifically, if $p_k = P(X=a_k)$ with $\sum_k p_k = 1$, then

$$
f(x) =
\begin{cases}
  p_k & \text{if } x = a_k,\\
  0   & \text{otherwise}.
\end{cases}
$$

The CDF for a discrete RV is a step function: $F(t) = \sum_{k: a_k \le t} p_k$.

</div>

Common Discrete Distributions:

| Distribution | Parameters | PMF $P(X=x)$ | Description |
| --- | --- | --- | --- |
| Bernoulli | $p \in [0, 1]$ | $p^x (1-p)^{1-x}$ for $x \in \lbrace 0, 1\rbrace$ | Models a single trial with two outcomes (e.g., success/failure). |
| Binomial | $n \in \mathbb{N},\ p \in [0, 1]$ | $\binom{n}{x} p^x (1-p)^{n-x}$ for $x \in \lbrace0, \dots, n\rbrace$ | Models the number of successes in $n$ independent Bernoulli trials. |
| Poisson | $\lambda > 0$ | $\dfrac{\lambda^x e^{-\lambda}}{x!}$ for $x \in \lbrace0, 1, 2, \dots\rbrace$ | Models the number of events occurring in a fixed interval of time or space. |

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

### Continuous Distributions and PDFs

For a continuous RV, the probability of any single point is zero. Probability is defined over intervals.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probability Density Function)</span></p>

A function $f: \mathbb{R} \to [0, \infty)$ is a PDF if:

1. $f(x) \ge 0$ for all $x \in \mathbb{R}$ (non-negativity).
2. $\int_{-\infty}^{\infty} f(x)\,dx = 1$.

The probability over an interval is given by the integral of the PDF: $P(a \le X \le b) = \int_a^b f(x)\,dx$. The corresponding CDF is $F(t) = \int_{-\infty}^{t} f(x)\,dx$.
</div>

| Distribution | Parameters | PDF $f(x)$ |
| --- | --- | --- |
| **Normal** | $\mu \in \mathbb{R},\ \sigma^2 > 0$ | $\dfrac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)$ |
| **Multivariate Normal** | $\mu \in \mathbb{R}^d,\ \Sigma \in \mathbb{R}^{d \times d}$ | $\dfrac{1}{(2\pi)^{d/2} \sqrt{\det \Sigma}} \exp\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)$ |
| **Exponential** | $\lambda > 0$ | $\lambda e^{-\lambda x}$ for $x \ge 0$ |
| **Uniform** | $a, b \in \mathbb{R},\ a < b$ | $\dfrac{1}{b-a}$ for $x \in [a, b]$ |

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

| Distribution                                   | Parameters                             | Expected value $\mathbb{E}[X]$ | Variance $\mathrm{Var}(X)$ |
| ---------------------------------------------- | ------------------------------------------------ | -------------------------------- | ---------------------------- |
| **Bernoulli**                                  | $p\in[0,1]$, $X\in\lbrace 0,1\rbrace$                         | $p$                              | $p(1-p)$                     |
| **Binomial**                                   | $n\in\mathbb{N}$, $p\in[0,1]$, $X\in\lbrace 0,\dots,n\rbrace$ | $np$                             | $np(1-p)$                    |
| **Exponential**                                | rate $\lambda>0$, $X\ge 0$                      | $1/\lambda$                      | $1/\lambda^2$                |
| **Poisson**                                    | $\lambda>0$, $X\in\lbrace 0,1,2,\dots\rbrace$                | $\lambda$                        | $\lambda$                    |
| **Geometric** *(# trials until first success)* | $p\in(0,1]$, $X\in\lbrace 1,2,\dots\rbrace$                   | $1/p$                            | $(1-p)/p^2$                  |

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Random Variables)</span></p>

**Expected Value** (Mean):

* Continuous: $\mathbb{E}[X] = \int_{-\infty}^{\infty} x f(x)\,dx$
* Discrete: $\mathbb{E}[X] = \sum_{i} x_i P(X=x_i)$

**Moments** and **Variance**:

* The $k$-th moment of an RV $X$ is $\mathbb{E}[X^k]$. The expected value is the first moment.
* The variance measures the spread of the distribution:
  $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2.$

</div>

### Rules and Laws of Probability

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Rules of Probability)</span></p>

Let $X$ and $Y$ be random variables with realizations $x$ and $y$.

* **Conditional Probability:** The probability of $x$ occurring given that $y$ has occurred.
  
  $$P(x \mid y) = \frac{P(x, y)}{P(y)}.$$

* **Chain Rule:** Decomposes a joint probability into a product of conditional probabilities.
  
  $$P(x_1, x_2, \dots, x_n) = P(x_1) P(x_2 \mid x_1) P(x_3 \mid x_1, x_2) \dots P(x_n \mid x_1, \dots, x_{n-1}).$$

* **Marginalization:** The probability of one variable can be found by summing (or integrating) over all possible values of other variables. 
  
  $$P(x) = \sum_y P(x, y) = \sum_y P(x \mid y) P(y).$$

* **Bayes' Rule:** Relates a conditional probability to its inverse and underpins Bayesian inference.
  
  $$P(y \mid x) = \frac{P(x \mid y) P(y)}{P(x)}.$$

* **Independence:** Two RVs $X$ and $Y$ are independent if learning the value of one provides no information about the other. This holds iff
  
  $$P(x, y) = P(x) P(y) \quad \text{or equivalently} \quad P(x \mid y) = P(x), \quad P(y \mid x) = P(y).$$

</div>

#### Asymptotic Theorems

These theorems describe the behavior of the sum of a large number of random variables.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Strong Law of Large Numbers)</span></p>

Let $(X_i)_{i=1}^\infty$ be a sequence of independent and identically distributed (i.i.d.) random variables with finite mean $\mathbb{E}[X_i] = \mu$. Then the sample mean converges almost surely to the true mean:

$$
\frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{\text{a.s.}} \mu
\quad \text{as } n \to \infty.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Central Limit Theorem)</span></p>

Let $(X_i)_{i=1}^\infty$ be a sequence of i.i.d. random variables with mean $\mu$ and finite variance $\sigma^2$. Then the distribution of the standardized sample mean converges to a standard Normal distribution:

$$
\frac{\sqrt{n}\left(\frac{1}{n}\sum_{i=1}^n X_i - \mu\right)}{\sigma}
\xrightarrow{d} \mathcal{N}(0, 1)
\quad \text{as } n \to \infty.
$$

</div>

## Review on Parameter Estimation

### Statistical Inference

Statistical inference aims to deduce properties of an underlying population or data-generating process from a finite sample of data. To do so, we often assume that our data is drawn from a family of distributions parametrized by a finite set of parameters.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parametric Model)</span></p>

Let $\Theta \subseteq \mathbb{R}^n$ is a parameter space. A family of probability distributions $P(\Theta) = \lbrace p_{\theta} \mid \theta \in \Theta \rbrace $ on a measurable space is called a **parametric model**.

</div>

Our goal is to estimate the true, unknown parameter $\theta$ that generated the data. We distinguish between the following

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Estimator, Estimate, Parameter)</span></p>

* **Parameter $\theta$:** The true, fixed (in the frequentist view) but unknown value.
* **Estimator $\hat{\theta}$:** A function of our data that is used to estimate $\theta$. It is a random variable.
* **Estimate**: A specific numerical value of the estimator realized from a specific data sample.

If I recompute samples and each time evaluate my estimate, I obtain a distribution over these estimates. This is called the **sampling distribution** of my estimator. The standard deviation of sampling distribution is the **standard error**.

</div>


<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Estimators)</span></p>

* **Bias:** The difference between the expected value of the estimator and the true parameter:
  
  $$\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta.$$

  An estimator is unbiased if its bias is zero.
* **Efficiency:** An estimator is more efficient than another if it has a smaller variance (i.e., a smaller standard error).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Frequentist vs. Bayesian Viewpoints)</span></p>

* **Frequentist View:** Parameters $\theta$ are fixed, unknown constants. Randomness arises solely from the data sampling process.
* **Bayesian View:** Parameters are random variables themselves, possessing their own probability distributions $p(\theta)$. Data is used to update our beliefs about the parameters.

</div>

### Paradigms of Parameter Estimation

There are three primary approaches to estimating parameters from data.

#### Method of Least Squares (LS)

The LS method estimates parameters by minimizing the sum of the squared differences between observed values and the values predicted by the model. It requires no assumptions about the underlying distribution of the data.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Method of Least Squares)</span></p>

Let $(x_i, y_i)$ be $N$ data pairs. We approximate the true relationship $y=f(x)$ with a parameterized function $f_\theta(x)$. The residual for the $i$-th data point is $\epsilon_i = y_i - f_\theta(x_i)$. The LS estimator is

$$
\hat{\theta}_{\text{LS}} = \arg\min_{\theta} \sum_{i=1}^N \epsilon_i^2
= \arg\min_{\theta} \sum_{i=1}^N \bigl(y_i - f_\theta(x_i)\bigr)^2 .
$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Estimating the Population Mean)</span></p>

* **Model:** Assume data points $X_1, \dots, X_N$ are generated by $X_i = \mu + \epsilon_i$, where $\mu$ is the true mean. We want to estimate $\mu$.
* **Cost Function:** 
  * $S(\mu) = \sum_{i=1}^N (X_i - \mu)^2$.
* **LS Estimation:** 
  * $\hat{\mu} = \arg\min_{\mu} \underbrace{\sum_{i=1}^N (X_i - \mu)^2}_{\text{Err}(\mu)}$.
* **Derivation:** Differentiate with respect to $\mu$ and set to zero.
  * $\frac{\partial S}{\partial \mu} = \sum_{i=1}^N -2(X_i - \mu) = -2 \left( \sum_i X_i - \mu \right) = 0$
  * $\sum_i X_i - N\mu = 0 \implies \hat{\mu} = \frac{1}{N} \sum_{i=1}^N X_i.$
* The LS estimate for the population mean is the sample mean.

</div>

#### Maximum Likelihood Estimation (MLE)

MLE selects the parameter values that make the observed data most probable under the assumed parametric model.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Likelihood Estimation)</span></p>

Let $X = \lbrace x_1, \dots, x_N \rbrace$ be observed data by assumption drawn from a model with density $p(x \mid \theta)$ ($X \sim P(\Theta)^{(N)}$).

* **Likelihood Function:** The joint density of the observed data, viewed as a function of the parameter $\theta$:
  * $\mathcal{L}(\theta \mid X) = p(x_1, \dots, x_N \mid \theta)$
* If the data are i.i.d., the likelihood factorizes:
  * $\mathcal{L}(\theta \mid X) = \prod_{i=1}^N p(x_i \mid \theta)$
* **MLE Definition:** The MLE is the value of $\theta$ that maximizes the likelihood function:
  
  $$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} \mathcal{L}(\theta \mid X)$$

</div>

In practice, it is often easier to maximize the log-likelihood $\ell(\theta \mid X) = \log \mathcal{L}(\theta \mid X)$, since the logarithm is monotonic and the resulting sums are easier to differentiate.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Estimating the Mean of a Normal Distribution)</span></p>

* **Model:** $x_i = \mu + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ or equivalently $X_i \sim \mathcal{N}(\mu, \sigma^2)$ i.i.d., with $\sigma^2$ known.
* **Log-Likelihood:**
  
  $$\ell(\mu \mid X) = \log \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(X_i-\mu)^2}{2\sigma^2}\right) = \sum_{i=1}^N \left( -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(X_i-\mu)^2}{2\sigma^2} \right).$$

* **Maximization:** Differentiate with respect to $\mu$ and set to zero.
  
  $$\frac{\partial \ell}{\partial \mu} = \sum_{i=1}^N \frac{X_i-\mu}{\sigma^2} = \frac{1}{\sigma^2} \sum_{i=1}^N (X_i-\mu) = 0$$
  
  $$\sum_i X_i - N\mu = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N X_i$$

* For this model, the MLE and LS estimates for the mean coincide.

</div>

#### Bayesian Inference (BI)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bayesian Inference)</span></p>

Bayesian inference uses Bayes' rule to update knowledge about parameters after observing data. It produces a posterior distribution for the parameters, not just a point estimate.

The core of Bayesian inference is Bayes' rule:

$$p(\theta \mid X) = \frac{p(X \mid \theta) p(\theta)}{p(X)}.$$

Here:

* $p(\theta \mid X)$ is the **posterior** distribution: our belief about $\theta$ after seeing data $X$.
* $p(X \mid \theta)$ is the **likelihood**, as in MLE.
* $p(\theta)$ is the **prior** distribution: our belief about $\theta$ before seeing any data.
* $p(X) = \int p(X \mid \theta) p(\theta)\,d\theta$ is the **evidence** or marginal likelihood of the data.

This is often summarized as: Posterior $\propto$ Likelihood $\times$ Prior.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Inferring the mean $\mu$ of a Normal distribution ($\sigma^2$ known) with a Normal prior on $\mu$.)</span></p>

* Likelihood: 
  * $p(X \mid \mu) \propto \exp\left(-\frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2\right)$.
* Prior: 
  * $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$, so $p(\mu) \propto \exp\left(-\frac{(\mu - \mu_0)^2}{2\sigma_0^2}\right)$.
* The posterior $p(\mu \mid X)$ is also a Normal distribution: 
  * $p(\mu \mid X) = p(X\mid \mu)p(\mu) / p(x) = \mathcal{N}(\mu, \sigma^2)\mathcal{N}(\mu_0, \sigma_0^2) / p(x)$.
* A common challenge is that the evidence $p(X)$ is often an intractable integral, requiring numerical methods such as MCMC or variational inference.

</div>


### Parameter Estimation for Intractable Problems

When closed-form solutions for estimators are not available, we turn to numerical optimization algorithms to find the parameters that minimize a cost function (e.g., sum of squared errors, negative log-likelihood).

#### Gradient Descent (GD)

Gradient Descent is an iterative first-order optimization algorithm for finding a local minimum of a differentiable function. The core idea is to take repeated steps in the opposite direction of the gradient of the function at the current point, as this is the direction of steepest descent.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gradient Descent)</span></p>

1. Initialize parameter guess $\theta_0$.
2. Repeat for $n$ iterations:
   * $\theta_{n+1} = \theta_n - \gamma \nabla J(\theta_n)$, 
   * where $J(\theta)$ is the cost function and $\gamma > 0$ is the learning rate.
3. Stop when convergence is reached (e.g., $\lvert J(\theta_{i}) - J(\theta_{i+1})\rvert < \epsilon$) or no more iterations.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(GD for LS Estimation of the Mean)</span></p>

* Model: $X_i = \mu + \epsilon_i$
* Cost: $J(\mu) = \frac{1}{2} \sum_i (X_i - \mu)^2$
* Gradient: $\nabla J(\mu) = \frac{\partial J}{\partial \mu} = -\sum_i (X_i - \mu)$
* Update Rule: $\mu_{n+1} = \mu_n - \gamma (-\sum_i (X_i - \mu_n)) = \mu_n + \gamma \sum_i (X_i - \mu_n)$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Challenges and Solutions of GD)</span></p>

* Local minima.
* Slow convergence in flat regions of the cost landscape.
* Overshooting the minimum if the learning rate $\gamma$ is too large.
* Practical remedies:
  * **Random Restarts:** Run the algorithm from multiple random initial conditions to increase the chance of finding a global minimum.
  * **Stochastic Gradient Descent (SGD):** Compute the gradient on a small random subset of data (a minibatch) at each step, leading to faster but noisier updates.
  * **Adaptive Learning Rates:** Algorithms like Momentum, Adagrad, and Adam dynamically adjust the learning rate to navigate varying curvatures in the cost landscape.

</div>

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

#### Newton-Raphson Method

The Newton-Raphson method is an iterative algorithm for finding the roots of a function $f(x)$, i.e., finding $x$ such that $f(x)=0$. In the context of model optimization, the goal is often to find the roots of the *derivative* of the loss function, which correspond to potential minima or maxima. This method is a second-order optimization algorithm that uses the curvature of the loss landscape to take more informed steps. It adapts the learning rate by incorporating the Hessian matrix (the matrix of second partial derivatives).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Original Newton-Raphson Method)</span></p>

**Newton–Raphson method** (often just “Newton’s method”) is an iterative numerical algorithm for finding a root of a differentiable function $f(x)=0$.

Starting from an initial guess $x_0$, it repeatedly updates

$$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}.$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/NewtonIteration_Ani.gif' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>Newton–Raphson iterations converge to a root by following tangent-line intersections.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Original Newton-Raphson Method)</span></p>

Geometrically, each step takes the tangent line to $f$ at $x_k$ and uses where that tangent crosses the $x$-axis as the next estimate. When $x_0$ is close to a simple root and $f'(x^{\ast})\neq 0$), the method typically converges very fast (quadratically). It can fail or converge slowly if $f'(x_k)$ is near zero, the initial guess is poor, or the function is not well-behaved near the root.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Original Newton-Raphson Method)</span></p>

In machine learning, the Newton–Raphson idea becomes a **second-order optimization method** for minimizing a loss $L(\theta)$ by using both the **gradient** and the **curvature** (Hessian).

* **Goal:** $\min_\theta L(\theta)$
* **Update (Newton step):**
  
  $$\theta_{k+1} = \theta_k - H(\theta_k)^{-1}\nabla L(\theta_k),$$
  
  where $\nabla L(\theta)$ is the gradient and $H(\theta)=\nabla^2 L(\theta)$ is the Hessian.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Newton-Raphson Method in ML)</span></p>

**Intuition:** around $\theta_k$, approximate the loss with a quadratic (2nd-order Taylor expansion) and jump directly to the minimizer of that quadratic. This can converge in very few iterations near the optimum (often **quadratically**), especially for convex problems.

This can be viewed as an analogue to Gradient Descent (GD), but with an adaptive learning rate determined by the inverse of the derivative. For a multivariate function $f$, the first derivative $f'$ becomes the gradient, and the second derivative (used to find the root of the first derivative) is the matrix of second partial derivatives known as the **Hessian**.

**Practical notes in ML:**

* Computing and inverting the Hessian is expensive in high dimensions ($O(d^2)$ memory, $O(d^3)$ naive solve).
* Hessians can be indefinite (not always positive definite), so the raw Newton step may not be a descent direction unless modified (e.g., damping).
* Common practical variants:

  * **(Damped) Newton / Levenberg–Marquardt:** use $(H+\lambda I)^{-1}\nabla L$
  * **Quasi-Newton (BFGS/L-BFGS):** approximate $H^{-1}$ without forming $H$
  * **Second-order for least squares / logistic regression:** leads to classic fast solvers; in GLMs it relates closely to IRLS.

*“Newton’s method optimizes a loss by taking steps based on the gradient scaled by the inverse Hessian, using curvature to reach the minimum much faster than gradient descent when feasible.”*

</div>

**Update Rule:**

$$\theta_{n+1} = \theta_n - H_{f(\theta_n)}^{-1} \nabla f(\theta_n),$$

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


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

* While Gradient Descent searches linearly, the Newton-Raphson method, using the Hessian, searches quadratically. This requires the computation of second derivatives and matrix inversions.
* The method converges more quickly than Gradient Descent.
* Newton-Raphson will find both minima and maxima, as it seeks any point where the gradient is zero.

</div>

#### Parameter Estimation for Intractable Bayesian Problems

In Bayesian inference, we are interested in the posterior distribution of parameters $\theta$ given data $X$, which is given by Bayes' theorem:
$p(\theta \mid X) \propto p(X \mid \theta) p(\theta)$. A significant problem arises when the normalizing constant, or model evidence, $p(X) = \int p(X \mid \theta)p(\theta) d\theta$, is intractable to compute. This intractability gives rise to a class of powerful simulation-based methods known as **Markov Chain Monte Carlo (MCMC)**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Core Concepts of MCMC)</span></p>

* **Monte Carlo Integration:** This principle states that we can estimate expectations by sampling. For a function $h(\theta)$, its expectation with respect to a probability distribution $p(\theta \mid X)$ is:

    $$
    \mathbb{E}[h(\theta) \mid X] = \int p(\theta \mid X) h(\theta) d\theta
    $$

    By drawing $N$ samples $\theta^{(i)}$ from the posterior $p(\theta \mid X)$, we can approximate this expectation using the law of large numbers:
    
    $$
    \mathbb{E}[h(\theta) \mid X] \approx \frac{1}{N} \sum_{i=1}^{N} h(\theta^{(i)})
    $$

* **Markov Chain:** This is a method to generate samples sequentially, where the next state depends only on the current state (a memoryless transition). The transition probability is defined as $p(\theta_t \mid \theta_{t-1}, \dots, \theta_0) = p(\theta_t \mid \theta_{t-1})$.

</div>

**Posterior Sampling with Metropolis-Hastings**

[More on Metropolis-Hastings](/subpages/monte-carlo-methods/mcmc/metropolis–hastings-algorithm/)

We can evaluate $p(⋅)$ for individual points but don’t know its normalization constant. Since we only care about the shape of the posterior, MCMC methods let us bypass computing the normalization constant and still approximate the posterior.

The general idea is to generate a sequence of parameter samples, $\theta^{(0)}, \theta^{(1)}, \dots, \theta^{(N)}$, that form a Markov chain whose stationary distribution is the target posterior distribution $p(\theta \mid X)$. We can work with the unnormalized posterior density, as the algorithm only depends on the ratio of densities, where the normalizing constant cancels out.
$\text{Posterior Density} \propto p(X \mid \theta)p(\theta)$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Metropolis-Hastings with a Symmetric Proposal for posterior $p(\theta \mid X)$ estimation)</span></p>

1.  **Initialization:** Choose an initial parameter value $\theta^{(0)}$ and specify the number of samples $N$.
2.  **Iteration:** Loop for $i = 1, \dots, N$:
    * **Propose:** Generate a new candidate sample $\theta_{\text{prop}}$ from a symmetric proposal distribution $q(\cdot \mid \theta^{(i-1)})$. A common choice is a normal distribution centered at the current sample: $\theta_{\text{prop}} \sim \mathcal{N}(\theta^{(i-1)}, \sigma^2 I)$.
    * **Compute Acceptance Ratio:** Calculate the ratio of the posterior densities at the proposed and current points. This is typically done in log-space for numerical stability.
        * $r_{\text{prop}} := p(X \mid \theta_{\text{prop}})p(\theta_{\text{prop}})$
        * $r_{\text{curr}} := p(X \mid \theta^{(i-1)})p(\theta^{(i-1)})$
        * The acceptance ratio is $r = \frac{r_{\text{prop}}}{r_{\text{curr}}}$.
    * **Accept or Reject:** Draw a random number $u$ from a uniform distribution, $u \sim \text{Unif}(0, 1)$.
        * If $u < \min(1, r)$, accept the proposal: $\theta^{(i)} = \theta_{\text{prop}}$.
        * Else, reject the proposal and stay at the current state: $\theta^{(i)} = \theta^{(i-1)}$.

</div>

Under mild conditions, the resulting sequence of samples $\lbrace\theta^{(i)}\rbrace_{i=1}^N$ will be drawn from the target posterior distribution $p(\theta \mid X)$. MCMC “walks around” the space in a way that favors high-probability regions but still occasionally explores others. Over time, the visited points represent the true distribution.

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Challenges in MCMC)</span></p>

* **Choice of Proposal Density:**
    * A proposal density that is too wide (high variance) will often generate proposals in regions of low probability, leading to a high rejection rate.
    * A proposal density that is too narrow will lead to a very high acceptance rate, but the chain will explore the parameter space very slowly, potentially undersampling important regions.
* **Initial Conditions and Burn-in:** The initial samples of the chain are biased by the starting condition $\theta^{(0)}$. To mitigate this bias, an initial "burn-in" period is defined, where the first $M$ samples are discarded from the final estimate.
* **Convergence:** To assess whether the chain has converged to its stationary distribution, it is common practice to run multiple chains from different, overdispersed initial conditions and check if they converge to the same distribution.
* **Parameter Constraints:** If parameters must conform to certain constraints (e.g., positivity), proposals that violate these constraints must be rejected, or transformations must be applied.
* **Autocorrelation:** Consecutive samples in the chain are often highly correlated. To obtain nearly independent samples, a technique called **thinning** is used, which involves keeping only every $n$-th sample and discarding the rest.

</div>


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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gibbs Sampling)</span></p>

Different MCMC algorithms exist for different problem structures. For instance, **Gibbs Sampling** is highly effective when dealing with models that have dependent parameters and where conditional distributions are easy to sample from.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Parameter Inference for Latent Variable Models)</span></p>

When the data generating process depends on both parameters $\theta$ and unobserved **latent variables** $z$, the model is specified as $p_\theta(X, z)$. The log-likelihood of the observed data $X$ requires marginalizing out these latent variables:

$$\log p_\theta(X) = \log \int p_\theta(X, z) dz$$

**This integration is often intractable, necessitating methods like MCMC or Variational Inference.**

</div>

## Fundamental Concepts of Time Series Analysis

### Stochastic Processes and Time Series

Intuitively, a time series is a realization or **sample path** of a random process, such as $\lbrace X_1, X_2, \dots, X_T\rbrace$. A time series is univariate if $X_t \in \mathbb{R}$ and multivariate if $X_t \in \mathbb{R}^k$ for $k > 1$. The fundamental assumption in time series analysis is that our observations are realizations of an underlying stochastic process.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Process)</span></p>

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space, $(E, \mathcal{E})$ be a measurable space (the state space), and $I \subseteq \mathbb{R}$ be an index set. A family of random variables $X = \lbrace X_t\rbrace_{t \in I}$ with values in $E$ is called a **stochastic process**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Time Series)</span></p>

A **time series** is a stochastic process $X = \lbrace X_t\rbrace_{t \in I}$, where each random variable $X_t$ shares the same state space but may have a different probability distribution.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sample Path or Realization)</span></p>

A **sample path** is a single outcome or sequence of observations $\lbrace x_t\rbrace_{t \in I}$ from a stochastic process. For example, $\lbrace x_1, x_2, \dots, x_T\rbrace$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Discrete vs. Continuous Time)</span></p>

* If the index set $I$ is countable (e.g., $I = \mathbb{Z}$ or $I = \mathbb{N}$), the process is a **discrete-time process**.
* If the index set $I$ is an interval (e.g., $I = [0, T]$), the process is a **continuous-time process**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ensemble)</span></p>

The **ensemble** is the population of all possible realizations that a stochastic process can generate. For a process $X_t = A \sin(\omega t + \phi) + \epsilon_t$, the ensemble would be the set of all possible sine waves generated by different values of the random variables $A$, $\phi$, and $\epsilon_t$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(White Noise Process)</span></p>

Let $\sigma^2 > 0$. A time series $X = \lbrace X_t\rbrace_{t \in I}$ is called a **white noise process** with variance $\sigma^2$, denoted $X_t \sim \mathcal{WN}(0, \sigma^2)$, if it satisfies:
1.  $\mathbb{E}[X_t] = 0$ for all $t \in I$.
2.  $$\text{Cov}(X_s, X_t) = \begin{cases} \sigma^2 & \text{if } s=t \\ 0 & \text{if } s \neq t \end{cases}$$

</div>

<div class="accordion">
  <details>
    <summary>Short explanation of the second point.</summary>
    <p>
    <strong>“same distribution” ≠ “same random variable”</strong> and doesn’t imply any <strong>dependence</strong>.
    </p>

    $$\mathrm{Cov}(X_s,X_t)=\mathbb{E}\big[(X_s-\mu)(X_t-\mu)\big].$$

  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gaussian white noise process)</span></p>

This property is often imposed on the error terms $\epsilon_t$ of statistical models. If, additionally, $X_t \sim \mathcal{N}(0, \sigma^2)$, the process is called a **Gaussian white noise process**.

</div>

### Autocovariance, Autocorrelation, and Cross-Correlation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Autocovariance Function)</span></p>

Let $X=\lbrace X_t\rbrace_{t \in I}$ be a stochastic process with $\mathbb{E}[X_t^2] < \infty$. The **autocovariance function** is a map $\gamma_{XX} : I \times I \to \mathbb{R}$ defined as:

$$\gamma_{XX}(s, t) = \text{Cov}(X_s, X_t) = \mathbb{E}[(X_s - \mathbb{E}[X_s])(X_t - \mathbb{E}[X_t])]$$

Using the property $\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$, and letting $\mu_t = \mathbb{E}[X_t]$, this can be written as:

$$\gamma_{XX}(s, t) = \mathbb{E}[X_s X_t] - \mu_s \mu_t$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(ACVF)</span></p>

* **Symmetry:** $\gamma_{XX}(s, t) = \gamma_{XX}(t, s)$
* **Variance:** $\gamma_{XX}(t, t) = \text{Var}(X_t)$
* **Cauchy-Schwarz Inequality:** $\lvert \gamma_{XX}(s, t)\rvert \leq \sqrt{\text{Var}(X_s)\text{Var}(X_t)}$
* The autocovariance function for a white noise process $X_t \sim \mathcal{WN}(0, \sigma^2)$: $\gamma_{XX}(s, t) = \sigma^2 \delta_{s,t}$
    * $\delta_{s,t}$ is the Kronecker delta.

</div>

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Autocorrelation Function)</span></p>

The **autocorrelation function** is the normalized version of the autocovariance function, mapping $\rho_{XX}: I \times I \to [-1, 1]$:

$$\rho_{XX}(s, t) = \frac{\gamma_{XX}(s, t)}{\sqrt{\gamma_{XX}(s, s)\gamma_{XX}(t, t)}} = \frac{\text{Cov}(X_s, X_t)}{\sqrt{\text{Var}(X_s)\text{Var}(X_t)}}$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cross-Covariance and Cross-Correlation Functions)</span></p>

For two stochastic processes $X=\lbrace X_t\rbrace_{t \in I}$ and $Y=\lbrace Y_t\rbrace_{t \in I}$, the **cross-covariance function** is:

$$\gamma_{XY}(s, t) = \text{Cov}(X_s, Y_t) = \mathbb{E}[(X_s - \mu_{X,s})(Y_t - \mu_{Y,t})]$$

The **cross-correlation function** is its normalized version:

$$\rho_{XY}(s, t) = \frac{\gamma_{XY}(s, t)}{\sqrt{\text{Var}(X_s)\text{Var}(Y_t)}}$$

</div>

### Stationarity and Ergodicity

#### Strong (Strict) Stationarity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Strong Stationarity)</span></p>

Let $h \in \mathbb{R}$ and $m \in \mathbb{N}$. A stochastic process $X = \lbrace X_t\rbrace_{t \in I}$ is **strongly stationary** if for any choice of time points $t_1, \dots, t_m \in I$, the joint probability distribution of $(X_{t_1}, \dots, X_{t_m})$ is the same as the joint probability distribution of $(X_{t_1+h}, \dots, X_{t_m+h})$, provided all time points remain in $I$.

$$(X_{t_1}, \dots, X_{t_m}) \stackrel{d}{=} (X_{t_1+h}, \dots, X_{t_m+h})$$

where $\stackrel{d}{=}$ denotes equality in distribution.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Strong Stationarity)</span></p>

* Strong stationarity is a statement about the entire joint distribution ("laws") of the process, which must be **invariant to shifts in time**.
* This is a **foundational assumption for many time series models**.
* In practice, **strong stationarity is difficult to prove** because you would theoretically need to test every possible moment (mean, variance, skewness, kurtosis, etc.) and every possible joint distribution across time.

</div>

<div class="math-callout math-callout--auestion" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Where Strong Stationarity holds)</span></p>

Despite being "unrealistic," the assumption is a necessary "working fiction" in several fields:
* **Physics/Thermodynamics:** In a sealed container of gas at equilibrium, strong stationarity is a very safe bet. This is where ergodicity originated.
* **Monte Carlo Simulations:** Since we program the rules of the simulation, we can ensure the process is strictly stationary to guarantee our results are valid.
* **Information Theory:** When compressing data or transmitting signals, we often model the source as stationary and ergodic to establish the theoretical limits of data transfer (like the Shannon-Hartley theorem).

</div>

#### Weak Stationarity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak Stationarity)</span></p>

A stochastic process $X = \lbrace X_t\rbrace_{t \in I}$ is **weakly stationary** (or covariance stationary) if it satisfies the following three conditions:
1.  The mean is constant for all $t$: $\mathbb{E}[X_t] = \mu$.
2.  The variance is finite for all $t$: $\mathbb{E}[X_t^2] < \infty$.
3.  The autocovariance between any two points depends only on their time lag $h = t-s$:
  
  $$\gamma_{XX}(s, t) = \gamma_{XX}(s, s+h) = \gamma_X(h)$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Weak Stationarity)</span></p>

* **Strong stationarity implies weak stationarity** (provided the first two moments exist). The reverse is not generally true.
* **For a Gaussian process, weak stationarity implies strong stationarity**, because the entire distribution is defined by its first two moments (mean and covariance).
* For a weakly stationary process, the autocovariance function simplifies to
  
  $$\gamma_X(h) = \mathbb{E}[(X_{t+h} - \mu)(X_t - \mu)]$$

</div>

#### Ergodicity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ergodicity)</span></p>

A stationary process is **ergodic** if its time average converges to its ensemble average (expected value) almost surely as the time horizon grows to infinity. For the mean:

$$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T X_t = \mathbb{E}[X_t] = \mu$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Ergodicity)</span></p>

* **Ergodicity allows us to infer properties of the entire process (the ensemble) from a single, sufficiently long sample path.**
* **Ergodicity requires (strong) stationarity.** It also typically requires conditions of stability (small perturbations do not cause large changes) and mixing (the influence of initial conditions fades over time).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Strong Stationarity for Ergodicity?)</span></p>

**Why Strong Stationarity?**
The core idea of ergodicity is that a **time average** (following one path over a long period) is equal to the **ensemble average** (taking a snapshot of many paths at one moment).
* **The Scope:** Ergodicity implies that a single realization of the process will eventually visit every part of the state space. To guarantee that this "sample path" represents the entire probability distribution, the entire distribution must be invariant over time.
* **The Limitation of Weak Stationarity:** Weak (wide-sense) stationarity only guarantees that the **mean** and **autocovariance** are constant. It says nothing about higher-order moments (like skewness or kurtosis) or the shape of the distribution itself. If those higher-order properties change over time, the time average cannot reliably converge to the ensemble average.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Theory vs. Practice Trade-off)</span></p>

Mathematically, we need strong stationarity because ergodicity is a statement about the **entirety** of a system's behavior. If the "shape" of the probability distribution changes at any point, the past becomes a poor predictor of the future, and the time average loses its meaning.

However, in **Applied Science and Engineering**, we often "downgrade" our requirements:
* **Ergodicity in the Mean:** We only care if the time-averaged mean converges to the ensemble mean. For this, we only need **weak stationarity** (and a condition that the autocovariance decays to zero).
* **Local Stationarity:** We assume the system is stationary over a "short enough" window of time to perform calculations (e.g., analyzing a 20-millisecond slice of a speech signal).

</div>

### Computing Properties from a Time Series

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name"></span></p>

Under the assumptions of **weak stationarity and ergodicity**, we can estimate the moments of the stochastic process from a single time series realization $\lbrace x_t\rbrace_{t=1}^T$.

* **Sample Mean:** $\hat{\mu} = \bar{x} = \frac{1}{T} \sum_{t=1}^T x_t$
* **Sample Variance:** $\hat{\gamma}(0) = \hat{\sigma}^2 = \frac{1}{T} \sum_{t=1}^T (x_t - \bar{x})^2$
* **Sample Autocovariance at lag $h$:** $\hat{\gamma}(h) = \frac{1}{T} \sum_{t=1}^{T-h} (x_t - \bar{x})(x_{t+h} - \bar{x})$

</div>

### Dealing with Non-stationarity

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Transformation of non-stationarity data)</span></p>

If a time series is non-stationary, it must often be transformed before standard models can be applied. Common techniques include:

* **High-pass filtering:** Transforming the series from the time domain to the frequency domain to remove low-frequency (trending) components.
* **Differencing:** Creating a new series by taking the difference between consecutive observations, e.g., $y_t = x_t - x_{t-1}$.
* **Detrending:** Fitting a deterministic trend (e.g., a linear function of time) and subtracting it from the data. For a linear trend $y_t = \beta_0 + \beta_1 t$, the detrended series is:
    
  $$x_t^{\ast} = x_t - (\hat{\beta}_0 + \hat{\beta}_1 t)$$

</div>

## Regression Models for Time Series

### Linear Regression

#### Model Components

A complete statistical model generally consists of four key components:

1.  **Model Architecture:** The mathematical form of the relationship between variables.
2.  **Loss Function:** A function that quantifies the error between model predictions and actual data.
3.  **Training Algorithm:** An optimization procedure to find the parameters that minimize the loss function.
4.  **Data:** The observations used to train and evaluate the model.

#### Model Architecture

The simple linear regression model assumes a linear relationship between the predictors and the response, corrupted by additive Gaussian noise.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Univariate Linear Regression Model)</span></p>

The response variable $y_t$ is modeled as a linear combination of the predictor variables $x_t$ plus an error term $\epsilon_t$.

$$y_t = x_t^\top \beta + \epsilon_t$$

where:

* $y_t \in \mathbb{R}$ is the response variable at time $t$.
* $x_t \in \mathbb{R}^{p+1}$ is the vector of predictor variables at time $t$ (including a constant term for the intercept).
* $\beta \in \mathbb{R}^{p+1}$ is the vector of model parameters or coefficients.
* $\epsilon_t$ is the error term, assumed to be independent and identically distributed (i.i.d.) Gaussian noise: $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$.

This implies that the conditional distribution of the response variable is also Gaussian:

$$y_t \mid x_t, \beta, \sigma^2 \sim \mathcal{N}(x_t^T \beta, \sigma^2)$$

For the entire dataset, we can express the model in a vectorized form:

$$Y = X\beta + E$$

where:

* $Y = (y_1, \dots, y_T)^\top$ is the $T \times 1$ vector of responses.
* $X$ is the $T \times (p+1)$ design matrix, where each row is $x_t^\top$.
* $\beta = (\beta_0, \dots, \beta_p)^T$ is the $(p+1) \times 1$ parameter vector.
* $E = (\epsilon_1, \dots, \epsilon_T)^T$ is the $T \times 1$ vector of errors.
</div>

#### Loss Function

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Least Squares Error or Sum of Squared Errors)</span></p>

$$\text{LSE}(\beta) = \sum_{t=1}^T (y_t - \hat{y}_t)^2 = \sum_{t=1}^T (y_t - x_t^\top \beta)^2$$

In vector notation, this is:

$$\text{LSE}(\beta) = (y - X\beta)^\top(y - X\beta)$$

</div>

The objective is to find the parameter vector $\beta$ that minimizes it.

#### Training Algorithm

The most common method for estimating the parameters $\beta$ in a linear regression model is the **method of ordinary least squares (LSE)**. This involves minimizing the sum of the squared differences between the observed responses and the responses predicted by the model. This is achieved by taking the derivative of the loss function with respect to $\beta$ and setting it to zero.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(OLS Estimator)</span></p>

The solution for $\hat{\beta}$, known as the **ordinary least squares (OLS) estimator**, is:

$$\hat{\beta} = (X^\top X)^{-1}X^\top y$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Expanding the LSE expression:</p>
    $$\text{LSE}(\beta) = y^\top y - y^\top X\beta - \beta^\top X^\top y + \beta^\top X^\top X\beta$$
    <p>Since $y^\top X\beta$ is a scalar, it is equal to its transpose $(\beta^\top X^\top y)$. Therefore:</p>
    $$\text{LSE}(\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta$$
    <p>Now, we take the derivative with respect to the vector $\beta$:</p>
    $$\frac{\partial \text{LSE}(\beta)}{\partial \beta} = \frac{\partial}{\partial \beta} (y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta)$$
    <p>The derivative is:</p>
    $$\frac{\partial \text{LSE}(\beta)}{\partial \beta} = -2X^\top y + (X^\top X + (X^\top X)^\top)\beta$$
    <p>Since $X^\top X$ is a symmetric matrix, $(X^\top X)^\top = X^\top X$. The derivative simplifies to:</p>
    $$\frac{\partial \text{LSE}(\beta)}{\partial \beta} = -2X^\top y + 2X^\top X\beta$$
    <p>Setting the derivative to zero to find the minimum:</p>
    $$-2X^\top y + 2X^\top X\hat{\beta} = 0$$
    $$2X^\top X\hat{\beta} = 2X^\top y$$
    $$X^\top X\hat{\beta} = X^\top y$$
    $$\hat{\beta} = (X^\top X)^{-1}X^\top y \qquad\square$$
  </details>
</div>

#### Parameter Estimation: Maximum Likelihood

An alternative framework for parameter estimation is **Maximum Likelihood Estimation (MLE)**. This approach finds the parameter values that maximize the likelihood of observing the given data.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Coincidence of MLE and LSE solutions)</span></p>

For the linear regression model with the assumption of i.i.d. Gaussian errors, the Least Squares Estimator (LSE) and the Maximum Likelihood Estimator (MLE) for the regression coefficients $\beta$ are identical:

$$\hat{\beta}_{\text{LS}} = \hat{\beta}_{\text{MLE}}$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Given the model assumption $y_t \sim \mathcal{N}(x_t^\top \beta, \sigma^2)$, the likelihood of observing the entire dataset $D$ is the product of the probability densities for each observation:</p>
    $$
    \mathcal{L}(\beta, \sigma^2) = p(Y \mid X, \beta, \sigma^2) = \prod_{t=1}^T p(y_t \mid x_t, \beta, \sigma^2)
    $$
    $$
    \mathcal{L}(\beta, \sigma^2) = \prod_{t=1}^T \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_t - x_t^\top \beta)^2}{2\sigma^2}\right)
    $$
    <p>It is often more convenient to work with the <strong>log-likelihood function</strong>:</p>
    $$
    l(\beta, \sigma^2) = \log \mathcal{L}(\beta, \sigma^2) = \sum_{t=1}^T \left[ -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(y_t - x_t^\top \beta)^2 \right]
    $$
    <p>To find the MLE for $\beta$, we maximize $l(\beta, \sigma^2)$ with respect to $\beta$. Notice that the terms involving $\sigma^2$ and $2\pi$ are constant with respect to $\beta$. Therefore, maximizing the log-likelihood is equivalent to minimizing the sum of squared errors:</p>
    $$
    \arg\max_\beta l(\beta, \sigma^2) \equiv \arg\min_\beta \sum_{t=1}^T (y_t - x_t^\top \beta)^2
    $$
    $$\hat{\beta}_{\text{LS}} = \hat{\beta}_{\text{MLE}}$$
  </details>
</div>

#### Model Diagnostics

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Examining the residuals, Assumptions of LRM on errors)</span></p>

After fitting a model, it is crucial to assess its validity by examining the **residuals**, which are the differences between the observed and fitted values.

The model decomposes the observed data $x_t$ into a fitted signal $\hat{y}_t$ and a residual component $res_t$:

$$y_t = \hat{y}_t + res_t$$

where $\hat{y}_t = x_t^\top \hat{\beta}$ is the predicted value.

The core assumptions of the linear regression model:
* **linearity, normality of errors**
* **constant variance**
* **independence of errors** 

should be checked by analyzing the residuals.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Common Diagnostic Visualizations)</span></p>

  * **Error Histogram:** A histogram of the residuals $res_t$ can provide a first check on the assumption that the errors are normally distributed.
  * **Q-Q (Quantile-Quantile) Plot:** This is a more rigorous way to check for normality. It plots the sorted quantiles of the residuals against the theoretical quantiles of a standard normal distribution. If the residuals are normally distributed, the points on the Q-Q plot will lie close to a straight diagonal line.
  * **ACF Plot of Residuals:** The Autocorrelation Function (ACF) plot of the residuals is used to check the assumption of independence. For time series data, it is critical to ensure there is no remaining temporal structure (autocorrelation) in the residuals. Significant spikes in the ACF plot suggest that the model has failed to capture the temporal dynamics in the data.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Sources of "Bad Residuals" (Model Misspecification))</span></p>

  * **Wrong assumption on modality:** The errors may not be normally distributed.
  * **Wrong assumption on relationship:** The true relationship between predictors and response may be non-linear.
  * **Missing external predictors:** Important explanatory variables may have been omitted from the model.
  * **Missing temporal dynamics:** The model fails to account for dependencies between observations over time.

</div>

### Handling Non-Linearity: Basis Expansion

One way to address a non-linear relationship between predictors and the response is through **basis expansion**. This technique extends the linear model by including non-linear transformations of the original predictors as additional regressors.

Suppose we have a set of predictors $x_t = (x_{1t}, \dots, x_{pt})^\top$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Basis Expansion Model)</span></p>

The model architecture is extended by introducing a set of $K$ **basis functions**, $\phi_k(\cdot)$:

$$y_t = \beta_0 + \sum_{k=1}^K \beta_k \phi_k(x_t) + \epsilon_t$$

where $\phi_k(\cdot)$ are chosen functions that transform the original predictor vector $x_t$. This model is still linear in the parameters $\beta_k$, so the standard LSE and MLE solutions still apply, but with a new design matrix whose columns are the transformed predictors $\phi_k(x_t)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Common Choices for Basis Functions)</span></p>

* **Polynomials:** $\phi_k(u_j) = u_{jt}^k$ (e.g., $u_t, u_t^2, u_t^3, \dots$).
* **Interaction Terms:** $\phi(u_i, u_j) = u_{it} \cdot u_{jt}$.
* **Radial Basis Functions:** Functions that depend on the distance from a center point.
* **Fourier Basis:** Sines and cosines to model periodic patterns, e.g., $\sin(\omega t)$ and $\cos(\omega t)$.

</div>

### Multivariate Linear Regression

In many scenarios, we want to predict multiple response variables simultaneously. This leads to **multivariate linear regression**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multivariate Linear Regression Model)</span></p>

**Data:** At each time $t=1, \dots, T$:

* **Response vector:** $y_t = (y_{1t}, \dots, y_{qt})^\top \in \mathbb{R}^q$.
* **Predictor vector:** $x_t = (x_{1t}, \dots, x_{pt})^\top \in \mathbb{R}^p$.

**Model:** The model is a direct extension of the univariate case, written in matrix form:

$$Y = XB + E$$

where:

  * $Y$ is the $T \times q$ matrix of response variables.
  * $X$ is the $T \times (p+1)$ design matrix.
  * $B$ is the $(p+1) \times q$ matrix of parameters, where each column corresponds to a response variable.
  * $E$ is the $T \times q$ matrix of errors.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Estimation of LSE/MLE for Multivariate Linear Regression Model)</span></p>

The solution for the parameter matrix $B$ is analogous to the univariate case:

$$
\hat{B} = (X^\top X)^{-1} X^\top Y
$$

This is equivalent to performing $q$ separate univariate linear regressions, one for each response variable.

</div>

### Generalized Linear Models (GLMs)

* Linear regression assumes a normally distributed response variable. **Generalized Linear Models (GLMs)** provide a framework to handle response variables with other distributions (e.g., binary, count data).
* Parameters in a GLM are typically estimated using MLE.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Linear Models (GLMs))</span></p>

GLMs are composed of three components:

1.  **Random Component:** The response variable $y_t$ follows a probability distribution from the exponential family (e.g., Bernoulli, Poisson, Gamma).
2.  **Systematic Component:** A **linear predictor**, $\eta_t$, is constructed as a linear combination of the predictors:

$$\eta_t = x_t^\top \beta$$

3.  **Link Function:** A function $g(\cdot)$ that links the expected value of the response, $\mu_t = \mathbb{E}[y_t]$, to the linear predictor:

$$g(\mu_t) = \eta_t$$

The inverse of the link function, $g^{-1}(\cdot)$, maps the linear predictor back to the mean of the response: $\mu_t = g^{-1}(\eta_t)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(LRM is a spacial case GLM)</span></p>

The standard linear regression model is a special case of a GLM.

  * **Random Component:** $y_t \sim \mathcal{N}(\mu_t, \sigma^2)$.
  * **Systematic Component:** $\eta_t = x_t^\top \beta$.
  * **Link Function:** The identity link, $g(\mu_t) = \mu_t$. Therefore, $\eta_t = \mu_t$, which gives us the familiar $\mathbb{E}[y_t] = x_t^\top \beta$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Logistic Regression)</span></p>

Logistic regression is a GLM used for modeling **binary response variables**.

**Model Architecture:**

  * **Data:** Observed binary response variable $y_t \in \lbrace 0, 1 \rbrace$ for $t=1, \dots, T$, with predictor vector $x_t$.
  * **Random Component:** The response is assumed to follow a Bernoulli distribution:
    
    $$y_t \sim \text{Bernoulli}(\pi_t)$$
    
    where $\pi_t = P(y_t=1 \mid x_t)$ is the "success" probability.
  * **Systematic Component:** The linear predictor is $\eta_t = x_t^\top \beta$.
  * **Link Function:** The **logit** link function is used, which is the natural logarithm of the odds:
    
    $$g(\pi_t) = \log\left(\frac{\pi_t}{1-\pi_t}\right) = \eta_t$$
    
    The inverse link function is the **sigmoid** (or logistic) function, which maps the linear predictor to a probability between 0 and 1:
    
    $$\pi_t = g^{-1}(\eta_t) = \frac{\exp(\eta_t)}{1+\exp(\eta_t)} = \frac{1}{1+\exp(-\eta_t)} = \sigma(\eta_t)$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Maximum Likelihood Estimation for Logistic Regression)</span></p>

The following statements are true for logistic regression:
* Does not have close-form solution (unlike linear regression)
* $\nabla_\beta \text{log-likelihood}(\beta) = \sum_{t=1}^T (y_t - \pi_t) x_t$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Loss Function (Negative Log-Likelihood):</strong> The probability mass function for a single Bernoulli observation is</p>
    $$p(y_t \mid \pi_t) = \pi_t^{y_t}(1-\pi_t)^{1-y_t}$$
    <p>The likelihood for the entire dataset is:</p>
    $$\mathcal{L}(\beta) = \prod_{t=1}^T p(y_t \mid x_t, \beta) = \prod_{t=1}^T \pi_t^{y_t} (1-\pi_t)^{1-y_t}$$
    <p>The log-likelihood is:</p>
    $$l(\beta) = \sum_{t=1}^T \left[ x_t \log(\pi_t) + (1-x_t)\log(1-\pi_t) \right]$$
    <p>Substituting $\pi_t = \sigma(x_t^\top \beta)$, we can express the log-likelihood in a form common to the exponential family:</p>
    $$l(\beta) = \sum_{t=1}^T \left[ y_t (x_t^\top \beta) - \log(1+\exp(x_t^\top \beta)) \right]$$
    <p><strong>Training Algorithm:</strong> We maximize the log-likelihood by taking its derivative with respect to $\beta$ and setting it to zero.</p>
    $$\nabla_\beta l(\beta) = \nabla_\beta \sum_{t=1}^T \left[ y_t \log(\sigma(x_t^\top\beta)) + (1-y_t)\log(1-\sigma(x_t^\top\beta)) \right]$$
    <p>The derivative of the log-likelihood for a single observation with respect to $\beta$ is:</p>
    $$\nabla_\beta l_t(\beta) = \left(y_t - \sigma(x_t^\top\beta)\right)x_t = (y_t - \pi_t)x_t$$
    <p>Summing over all observations gives the full gradient:</p>
    $$\nabla_\beta l(\beta) = \sum_{t=1}^T (y_t - \pi_t) x_t$$
    <p>Unlike in linear regression, setting this equation to zero does not yield a closed-form solution for $\beta$. Therefore, iterative optimization algorithms like <strong>gradient ascent</strong> are used.</p>
    <p>The gradient ascent update rule is:</p>
    $$\beta_{\text{new}} = \beta_{\text{old}} + \alpha \nabla_\beta l(\beta_{\text{old}})$$
    <p>where $\alpha$ is the learning rate. For a single data point (stochastic gradient ascent), the rule is:</p>
    $$\beta_{\text{new}} = \beta_{\text{old}} + \alpha (y_t - \pi_t) x_t$$
  </details>
</div>

### Modeling Complex Data Structures

#### Multimodal Regression

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multivariate Linear Regression Model)</span></p>

This approach models datasets where each observation consists of multiple types of data, or **modalities**.

**Setup:**

* We have multiple observed modalities, for example, a binary variable $y_{1t} \in \lbrace 0,1 \rbrace$ and a continuous variable $y_{2t} \in \mathbb{R}$.
* These are collected into a response vector $y_t = (y_{1t}, y_{2t})^T$.
* The full dataset is $D = \lbrace(y_t, x_t)\rbrace_{t=1}^T$.

**Model:** We model each modality separately, conditional on the predictors $x_t$. This typically involves different parameter vectors ($\beta_1, \beta_2$) and distributions for each modality.

* Linear predictor for modality 1: $\eta_{1t} = \beta_1^\top x_t$
* Linear predictor for modality 2: $\eta_{2t} = \beta_2^\top x_t$
* Model for modality 1: $y_{1t} \mid x_t \sim \text{Bernoulli}(\pi_t)$, where $\pi_t = \sigma(\eta_{1t})$.
* Model for modality 2: $y_{2t} \mid x_t \sim \mathcal{N}(\mu_t, \sigma^2)$, where $\mu_t = \eta_{2t}$.

**Key assumption:**

* **conditional independence**: the different modalities are independent of each other, given the predictors.

$$p(y_{1t}, y_{2t} \mid x_t) = p(y_{1t} \mid x_t) p(y_{2t} \mid x_t)$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Loss Function (MLE) for Multimodal Regression Above)</span></p>

Due to the conditional independence assumption, the total log-likelihood is the sum of the log-likelihoods for each modality:

$$l(\beta_1, \beta_2, \sigma^2) = \sum_{t=1}^T \log p(y_{1t} \mid x_t) + \sum_{t=1}^T \log p(y_{2t} \mid x_t)$$

$$
l(\beta_1, \beta_2, \sigma^2) = \sum_{t=1}^T [y_{1t}\eta_{1t} - \log(1+\exp(\eta_{1t}))] + \sum_{t=1}^T \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(y_{2t} - \eta_{2t})^2 \right]
$$

This loss function can be maximized jointly to find the parameters for both models.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What this calculation actually proves)</span></p>

* If the assumption of conditional independence between modalities is not reasonable, a common approach is to introduce a shared **latent variable** $\epsilon_t$ that **captures common dynamics or unobserved factors influencing all modalities**. 
* Conditional on both the predictors $x_t$ and the latent variable $\epsilon_t$, the modalities are then assumed to be independent.

</div>

## Hierarchical Modeling for Multiple Time Series

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
    $$P(\theta_j, \phi) = P(\theta_j \mid \phi)P(\phi) \quad \text{from the definition of conditional probability}$$
    <p>With $\phi$ as its hyperparameter with hyperprior distribution, $P(\phi)$.</p>
    <p>Thus, the posterior distribution is proportional to:</p>
    $$P(\phi, \theta_j \mid y) \propto P(y_j \mid \theta_j, \phi)P(\theta_j, \phi) \quad \text{using Bayes' Theorem}$$
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

### Hierarchical Modeling

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Multiple Datasets)</span></p>

Consider a scenario with $N$ subjects, where for each subject $i=1, \dots, N$, we have a time series dataset $D_i$.

  * The data for subject $i$ consists of $T_i$ time steps: $D_i = \lbrace x_{i1}, x_{i2}, \dots, x_{iT_i} \rbrace$.
  * For simplicity, we can assume all time series have the same length, $T_i = T$.
  * Each observation may also be associated with inputs: $W_i = \lbrace w_{i1}, w_{i2}, \dots, w_{iT_i} \rbrace$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposed Solutions</span><span class="math-callout__name">(Two Extreme Modeling Strategies)</span></p>

1.  **Separate Models:** Fit a completely independent model with parameters $\theta_i$ to each dataset $D_i$.

      * **Advantage:** This approach is excellent for capturing individual differences, as each model is tailored specifically to its own data.
      * **Disadvantage:** If the number of time steps $T$ is small and the model is complex, the estimates for $\theta_i$ can be noisy and highly prone to overfitting. There is no sharing of information across subjects.

1.  **Fully Pooled Model:** Union all data into a single large dataset: $D_{\text{pooled}} = \lbrace D_1, D_2, \dots, D_N \rbrace$. Fit a single parameter vector $\theta$ to all the data.

      * **Advantage:** This method yields more stable parameter estimates because it leverages the entire data pool.
      * **Disadvantage:** It completely ignores inter-individual differences, assuming all subjects are governed by the exact same process.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Partial pooling)</span></p>

We want to find a middle ground that finds a compromise between these extremes – **partial pooling**. This brings us to Bayesian hierarchical modeling, also known as multilevel modeling.

</div>

#### The Hierarchical Approach

Hierarchical modeling provides a principled compromise between these two extremes.

**Core Idea:** We introduce subject-specific parameters $\theta_1, \theta_2, \dots, \theta_N$, but we assume they are not arbitrary. Instead, they are drawn from a common parent distribution, which is itself described by hyperparameters.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bayesian hierarchical modeling)</span></p>

**Bayesian hierarchical modeling** makes use of two important concepts in deriving the posterior distribution, namely:

* **Hyperparameters:** parameters of the prior distribution
* **Hyperpriors:** distributions of Hyperparameters

Each subject's parameter $\theta_i$ is drawn from a **parent distribution** parameterized by $\beta$:

$$\theta_i \sim p(\theta \mid \beta)$$

We then place a prior distribution, known as a **hyperprior**, on the hyperparameters $\beta$:

$$\beta \sim p(\beta)$$

**Hierarchical chain of dependencies**:

$$\beta \rightarrow \theta_i \rightarrow D_i \quad \text{for } i=1, \dots, N$$

The **joint distribution** over all data ($D_{1:N}$), parameters ($\theta_{1:N}$), and hyperparameters ($\beta$) is given by:

$$p(\beta, \lbrace\theta_i\rbrace_{i=1}^N, \lbrace D_i\rbrace_{i=1}^N) = p(\beta) \prod_{i=1}^N \left[ p(\theta_i \mid \beta) p(D_i \mid \theta_i) \right]$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hierarchical Bayesian inference)</span></p>

The **goal of hierarchical Bayesian inference** is to compute the joint posterior distribution of all subject-level parameters and group-level hyperparameters given the observed data from all subjects.

$$
p(\beta, \lbrace\theta_i\rbrace_{i=1}^N \mid \lbrace D_i\rbrace_{i=1}^N) = \dfrac{p(\lbrace D_i\rbrace_{i=1}^N \mid \lbrace\theta_i\rbrace_{i=1}^N)p(\lbrace\theta_i\rbrace_{i=1}^N \mid \beta)p(\beta)}{p(\lbrace D_i\rbrace_{i=1}^N)}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information Sharing in Bayesian hierarchical modeling)</span></p>

* Our primary target is the posterior distribution $p(\theta_1, \dots, \theta_N, \beta \mid D_1, \dots, D_N)$. 
* This structure allows the model to "borrow statistical strength" across subjects. 
* The data from subject $j$ informs the posterior of $\theta_j$, 
* which in turn informs the posterior of the group hyperparameter $\beta$. 
* This updated knowledge about $\beta$ then helps to regularize and improve the estimates for all other subjects' parameters $\theta_i$ where $i \neq j$.

</div>

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

<div class="accordion">
  <details>
    <summary>Example: Hierarchical Delay Discounting</summary>
    <h3>Example: Hierarchical Delay Discounting</h3>
    <p>Let's consider a concrete example of hierarchical modeling applied to a delay discounting task in cognitive science.</p>
    <h4>Data and Task Structure</h4>
    <p>We have data from $N$ individuals. Subject $i$ performs $T_i$ trials, indexed by $t=1, \dots, T_i$. In each trial, the subject chooses between two options:</p>
    <ol>
      <li>An immediate reward of value $A_{I,it}$.</li>
      <li>A delayed reward of value $A_{D,it}$ available after a delay $D_{it}$.</li>
    </ol>
    <p>The observed data for each trial is a binary choice $y_{it}$:</p>
    $$y_{it} = \begin{cases} 1 & \text{if delayed choice is selected} \\ 0 & \text{if immediate choice is selected} \end{cases}$$
    <p>The full dataset for subject $i$ is $Y_i = \lbrace y_{it}\rbrace_{t=1}^{T_i}$.</p>
    <h4>Model Architecture: Hyperbolic Delay Discounting</h4>
    <p>The subjective value of the options is modeled as follows:</p>
    <ul>
      <li><strong>Value of Immediate Option:</strong> $V_{I,it} = A_{I,it}$</li>
      <li><strong>Value of Delayed Option:</strong> The value of the delayed reward is discounted hyperbolically based on the delay duration.</li>
    </ul>
    $$V_{D,it} = \frac{A_{D,it}}{1 + k_i D_{it}}$$
    <p>Here, $k_i > 0$ is the subject-specific <strong>discount rate</strong>. A higher $k_i$ means the subject devalues future rewards more steeply.</p>
    <p>The choice is then modeled based on the utility difference, $\Delta V_{it} = V_{D,it} - V_{I,it}$. A positive $\Delta V_{it}$ indicates a preference for the delayed option.</p>
    <p>The probability of choosing the delayed option is modeled using a logistic function (i.e., logistic regression):</p>
    $$p_{it} := P(y_{it} = 1 \mid k_i, \beta_i) = \sigma(\beta_i \Delta V_{it}) = \frac{1}{1 + \exp(-\beta_i \Delta V_{it})}$$
    <p>where $\beta_i$ is a subject-specific "softmax" temperature parameter controlling the steepness of the decision boundary. The likelihood for the observed choices of a single individual $i$ follows a Bernoulli distribution:</p>
    $$\mathcal{L}(k_i, \beta_i \mid Y_i) = \prod_{t=1}^{T_i} p_{it}^{y_{it}} (1 - p_{it})^{1 - y_{it}}$$
    <h4>Hierarchical Priors</h4>
    <p>Instead of treating each $k_i$ and $\beta_i$ as independent, we impose a group-level structure. Since $k_i$ and $\beta_i$ must be positive, we model their logarithms as being drawn from a Normal distribution.</p>
    <p><strong>Individual Parameters:</strong> $k = \lbrace k_i\rbrace_{i=1}^N$ and $\beta = \lbrace \beta_i\rbrace_{i=1}^N$.</p>
    $$\log(k_i) \sim \mathcal{N}(\mu_k, \sigma_k^2) \quad \log(\beta_i) \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)$$
    <p><strong>Group-level Hyperparameters:</strong> The means $(\mu_k, \mu_\beta)$ and variances $(\sigma_k^2, \sigma_\beta^2)$ of the parent distributions.</p>
    <p><strong>Hyperpriors:</strong> We place priors on the hyperparameters themselves.</p>
    <p><em>Priors on Means:</em> $\mu_k \sim \mathcal{N}(\mu_{k0}, \sigma_{k0}^2)$ and $\mu_\beta \sim \mathcal{N}(\mu_{\beta 0}, \sigma_{\beta 0}^2)$.</p>
    <p><em>Priors on Variances:</em> $\sigma_k^2 \sim \text{Inverse-Gamma}(a_k, b_k)$ and $\sigma_\beta^2 \sim \text{Inverse-Gamma}(a_\beta, b_\beta)$.</p>
    <h4>Full Bayesian Formulation</h4>
    <p>The goal is to obtain the full posterior distribution over all parameters and hyperparameters given the data $Y$:</p>
    $$P(k, \beta, \mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2 \mid Y)$$
    <p>By Bayes' theorem, this is proportional to the product of the likelihood and the priors:</p>
    $$\propto P(Y \mid k, \beta) P(k, \beta \mid \mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2) P(\mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2)$$
    <ul>
      <li><strong>Joint Likelihood:</strong> Assuming independence across subjects given their parameters:</li>
    </ul>
    $$P(Y \mid k, \beta) = \prod_{i=1}^N P(Y_i \mid k_i, \beta_i)$$
    <ul>
      <li><strong>Joint Prior on Individual Parameters:</strong></li>
    </ul>
    $$P(k, \beta \mid \dots) = \left[ \prod_{i=1}^N P(\log(k_i) \mid \mu_k, \sigma_k^2) \right] \left[ \prod_{i=1}^N P(\log(\beta_i) \mid \mu_\beta, \sigma_\beta^2) \right]$$
    <ul>
      <li><strong>Joint Hyperprior:</strong> Assuming independence of the hyperparameters:</li>
    </ul>
    $$P(\mu_k, \mu_\beta, \sigma_k^2, \sigma_\beta^2) = P(\mu_k) P(\mu_\beta) P(\sigma_k^2) P(\sigma_\beta^2)$$
    <p>By combining these expressions, we define the full model. Inference is typically performed using sampling techniques like Markov Chain Monte Carlo (MCMC).</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Alternative: Hierarchical Modeling with a Parent Matrix)</span></p>

An alternative parametrization for hierarchical models, particularly useful in high-dimensional settings, involves a shared parent matrix.

  * **Naive Parametrization:** Each subject $i$ has a parameter vector $\theta_i \in \mathbb{R}^p$. The total number of parameters to estimate for the subject level is $p \times N$.
  * **Parent Matrix Parametrization:** We introduce a shared parent matrix $W \in \mathbb{R}^{p \times k}$ (with $k < p$) and individual subject vectors $h_i \in \mathbb{R}^k$. The subject-specific parameter vector $\theta_i$ is then constructed as a linear combination of the columns of $W$:
    
    $$\theta_i = W h_i$$

    The total number of parameters is now $(p \times k) + (k \times N)$, which can be significantly smaller than $p \times N$ if $k$ is chosen well. This acts as a form of dimensionality reduction.
  * **Hierarchical Chain:** The dependency structure becomes: $W, h_i \rightarrow \theta_i \rightarrow D_i$.

</div>

#### Additional Sources

* [Introduction to hierarchical modeling](https://towardsdatascience.com/introduction-to-hierarchical-modeling-a5c7b2ebb1ca/)
* [Bayesian hierarchical modeling](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling)
* [Hierarchical Modeling](https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html)

## Autoregressive Moving Average (ARMA) Models

ARMA models are a fundamental class of models for analyzing stationary time series. They are built on the principle that the current value of a series can be explained by a combination of its own past values and past random shocks.

### Motivation and Components

If the residuals of a regression model on time series data are found to be autocorrelated or cross-correlated, it implies that the model is missing important temporal structure. ARMA models are designed to capture this very structure.

  * **Autoregressive (AR) Part:** This component regresses the time series on its own past values. It captures the "memory" or persistence in the series.
  * **Moving Average (MA) Part:** This component models the current value as a function of past random perturbations or "shocks". It can be thought of as a sequence of weighted random shocks.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Autoregressive model (AR))</span></p>

An **Autoregressive model of order $p$**, denoted **AR($p$)**, is defined as:

$$X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t$$

where $\epsilon_t$ is a white noise process, typically $\epsilon_t \sim \mathcal{WN}(0, \sigma^2)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Moving Average Model (MA))</span></p>

A **Moving Average model of order $q$**, denoted **MA($q$)**, is defined as:

$$X_t = b_0 + \sum_{j=1}^q b_j \epsilon_{t-j} + \epsilon_t$$

Note that $X_t$ depends on past **error terms**, not past values of $X$ itself.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(ARMA Model)</span></p>

Combining these two components gives the **ARMA($p$,$q$)** model:

$$X_t = c + \sum_{i=1}^p a_i X_{t-i} + \sum_{j=1}^q b_j \epsilon_{t-j} + \epsilon_t$$

This can also be extended to include external inputs $u_t$. The full set of model parameters to be estimated is 

$$\theta = \lbrace c, a_1, \dots, a_p, b_1, \dots, b_q, \sigma^2 \rbrace$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What's the point of considering errors in predicting a new state? If they are errors, what's their predictive power?)</span></p>

The trick is to stop thinking of these as "mistakes" and start thinking of them as **shocks** or **innovations**.

**The "Impulse" Idea**

In an MA model, $\epsilon_t$ represents a new piece of information that enters the system at time $t$ that was **not** explained by the past.

Imagine you are modeling the water level in a harbor.

* **The Shock ($\epsilon$):** A giant ship enters the harbor, creating a displacement wave.
* **The Persistence:** That wave doesn't disappear instantly. It ripples and bounces for a while.
* **The Prediction:** Even if you can't predict *when* the next ship will arrive, if you know a ship arrived one minute ago ($\epsilon_{t-1}$), you can predict that the water will still be choppy *now* ($X_t$).

**The "error" has predictive power because its effects linger.**

**Information vs. Noise**

In econometrics, we often call these terms **innovations**.

* **White Noise:** Purely random and unpredictable *at the moment it happens*.
* **The Model's Job:** To capture how much of that "random shock" stays in the system for the next step.

If the coefficient $\theta$ is 0.8, it means 80% of yesterday's unexpected shock is still influencing today's value. We aren't predicting the *error itself*; we are predicting the **observed value** based on the fact that a specific shock recently occurred.

**Error vs. Residual**

There is a subtle but vital distinction here:

1. **The Theoretical Error ($\epsilon$):** This is the "shock." We assume it happened.
2. **The Residual ($\hat{\epsilon}$):** This is what we calculate after the fact.

When we "predict" using an MA model, we use the **residuals** from previous steps. If our model predicted the value would be 100, but it turned out to be 110, we know there was a  "shock." Since the MA model says shocks linger, we add a fraction of that  to our prediction for the next step.

**Why not just use past values of X (Autoregression)?**

You might ask: *"Why not just use the previous water level ($X_{t-1}$) instead of the previous shock ($\epsilon_{t-1}$)?*

* **AR models (using $X$):** Assume the *entire* past value influences the future. This creates a "long memory" where effects decay slowly.
* **MA models (using $\epsilon$):** Assume only the *random shocks* influence the future. This creates a "short memory." After  steps, the shock is completely gone from the system.

**Summary: The "Pothole" Analogy**

Think of driving a car with bad shocks:

* **AR Model:** The car’s height right now depends on its height a second ago (the car stays bouncy).
* **MA Model:** The car’s height right now depends on the fact that you hit a **pothole** (the error/shock) three seconds ago.

The "point" of the MA model is to capture those temporary, lingering effects of specific events without assuming the entire history of the variable matters.

</div>

### Duality and Stationarity

There is a fundamental duality between AR and MA processes. Under certain stability conditions, any finite-order AR process can be represented as an infinite-order MA process, and vice-versa.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Finite AR $\iff$ Infinite MA)</span></p>

Any finite-order AR process can be represented as an infinite-order MA process, and vice-versa:

$$X_t = a_0 + a_1 X_{t-1} + \epsilon_t = a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \epsilon_{t-k}$$

**This infinite expansion is only valid if the series converges.**

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let's examine this with a simple AR(1) process:</p>
    $$X_t = a_0 + a_1 X_{t-1} + \epsilon_t$$
    <p>We can recursively expand this expression:</p>
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
  </details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Stationarity in the Mean for AR(1))</span></p>

* **For the process to be stationary in the mean, its expected value must be constant and finite.**
* The condition for stationarity of an AR(1) process is $\lvert a_1 \rvert < 1$.

$$\mathbb{E}[X_t] = \frac{a_0}{1-a_1} \quad \text{if } \lvert a_1 \rvert < 1$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Taking the expectation of the expanded form:</p>
    $$\mathbb{E}[X_t] = \mathbb{E}\left[ a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \epsilon_{t-k} \right]$$
    $$\mathbb{E}[X_t] = a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \mathbb{E}[\epsilon_{t-k}]$$
    <p>Since $\mathbb{E}[\epsilon_{t-k}]=0$, the second term vanishes. The first term is a geometric series which converges if and only if $\lvert a_1 \rvert < 1$.</p>
    $$\mathbb{E}[X_t] = \frac{a_0}{1-a_1} \quad \text{if } \lvert a_1 \rvert < 1$$
    <p>Therefore, the condition for stationarity of an AR(1) process is $\lvert a_1 \rvert < 1$.</p>
  </details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/ar_1_different_a1.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>AR(1) time series with different $\rvert a_1\rvert$.</figcaption>
</figure>

#### State-Space Representation and Stability

A powerful technique for analyzing AR models is to write them in a **state-space (or vector) form**. 

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(scalar AR($p$) $\implies$ $p$-variate VAR(1))</span></p>

Any scalar AR($p$) process can be represented as a $p$-variate VAR(1) process:

$$X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t \quad\implies\quad \mathbf{X}_t = \mathbf{a} + A \mathbf{X}_{t-1} + \mathbf{\epsilon}_t$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Consider an AR($p$) process</p>
    $$X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t$$
    <p>We can define a $p$-dimensional state vector $\mathbf{X}_t$:</p>
    $$\mathbf{X}_t = \begin{pmatrix} X_t \\ X_{t-1} \\ \vdots \\ X_{t-p+1} \end{pmatrix}$$
    <p>The process can then be written in the form</p>
    $$\mathbf{X}_t = \mathbf{a} + A \mathbf{X}_{t-1} + \mathbf{\epsilon}_t$$
    <p>where:</p>
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
  </details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Stationarity $\implies$ spectral radius of $A$ is less than 1)</span></p>

* The stability and stationarity of the entire process can then be assessed by examining the eigenvalues of the companion matrix $A$. 
* For the process to be stationary, the spectral radius of $A$ must be less than 1.

$$\max_i \lvert \lambda_i(A) \rvert < 1$$

where $\lambda_i(A)$ are the eigenvalues of $A$.

</div>

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

### Model Identification Using Autocorrelation

* A key step in ARMA modeling is identifying the orders $p$ and $q$. 
* **The Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) are the primary tools for this task.**

#### Autocorrelation in AR(1) Processes

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Autocovariance and autocorrelation for zero-mean AR(1))</span></p>

For a zero-mean ($a_0=0$) AR(1) process 

$$X_t = a_1 X_{t-1} + \epsilon_t$$

The **autocovariance** at lag $k$ ($\gamma(k)$) and **autocorrelation** at lag $k$ ($\rho(k)$):

$$\gamma(k) = a_1^k \gamma(0) \qquad \rho(k) = a_1^k$$

The ACF of an AR(1) process **decays exponentially** to zero.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Consider a zero-mean ($a_0=0$) AR(1) process:</p>
    $$X_t = a_1 X_{t-1} + \epsilon_t$$
    <p>The autocovariance at lag $k$, $\gamma(k)$, can be calculated.</p>
    <ul>
      <li><strong>Lag 1:</strong></li>
    </ul>
    $$\mathbb{E}[X_t X_{t-1}] = \mathbb{E}[(a_1 X_{t-1} + \epsilon_t)X_{t-1}] = a_1 \mathbb{E}[X_{t-1}^2] + \mathbb{E}[\epsilon_t X_{t-1}]$$
    <p>$\mathbb{E}[\epsilon_t X_{t-1}]=0$ since $\epsilon_t$ is uncorrelated with past values of $X$. Thus, $\gamma(1) = a_1 \gamma(0)$.</p>
    <ul>
      <li><strong>Lag 2:</strong></li>
    </ul>
    $$\mathbb{E}[X_t X_{t-2}] = \mathbb{E}[(a_1 X_{t-1} + \epsilon_t)X_{t-2}] = a_1 \mathbb{E}[X_{t-1} X_{t-2}] = a_1 \gamma(1) = a_1^2 \gamma(0)$$
    <ul>
      <li><strong>General Lag $k$:</strong></li>
    </ul>
    $$\gamma(k) = a_1^k \gamma(0)$$
    <p>The autocorrelation function, $\rho(k) = \gamma(k)/\gamma(0)$, is therefore $\rho(k) = a_1^k$. The ACF of an AR(1) process <strong>decays exponentially</strong> to zero.</p>
  </details>
</div>

#### Autocorrelation in MA($q$) Processes

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Autocorrelation in MA($q$) Processes)</span></p>

For zero-mean MA($q$) process 

$$X_t = \epsilon_t + \sum_{j=1}^q b_j \epsilon_{t-j}$$

$$\text{ACF}(k) = 0 \quad \text{for all } k > q$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Consider a zero-mean MA($q$) process: $X_t = \epsilon_t + \sum_{j=1}^q b_j \epsilon_{t-j}$. Let's calculate the autocovariance at lag $k > q$.</p>
    <p>Because the error terms are white noise, $\mathbb{E}[\epsilon_i \epsilon_j] = \sigma^2$ if $i=j$ and 0 otherwise. For the expectation $\mathbb{E}[X_t X_{t-k}]$ to be non-zero, there must be at least one pair of matching indices in the sums. If we consider a lag $k>q$, it is impossible to satisfy the matching index condition. Therefore, for any $k>q$, all cross-product terms have an expectation of zero.</p>
    $$\text{ACF}(k) = 0 \quad \text{for all } k > q$$
  </details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/ACF_MA.png' | relative_url }}" alt="Filtering Smoothing Schema" loading="lazy">
  <figcaption>Autocorrelation in MA($q$) Process.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(ACF of an MA($q$) process)</span></p>

This provides a clear signature: the ACF of an MA($q$) process **sharply cuts off** to zero after lag $q$.

</div>

#### The Partial Autocorrelation Function (PACF)

The PACF at lag $k$ measures the correlation between $X_t$ and $X_{t-k}$ after removing the linear dependence on the intervening variables ($X_{t-1}, X_{t-2}, \dots, X_{t-k+1}$). A key property of the PACF for an AR($p$) process is:

$$\text{PACF}(k) = 0 \quad \text{for all } k > p$$

This is because in an AR($p$) model, the direct relationship between $X_t$ and $X_{t-k}$ (for $k>p$) is fully mediated by the first $p$ lags.

### Modeling with ARMA

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Parameter Estimation for AR and ARMA)</span></p>

**For a pure AR($p$) model, parameter estimation is equivalent to a linear regression problem.**

$$
y = \begin{pmatrix} X_T \\ X_{T-1} \\ \vdots \\ X_{p+1} \end{pmatrix} \quad
X = \begin{pmatrix}
1 & X_{T-1} & \dots & X_{T-p} \\
1 & X_{T-2} & \dots & X_{T-p-1} \\
\vdots & \vdots & \ddots & \vdots \\
1 & X_p & \dots & X_1
\end{pmatrix}
$$

* The parameters can then be estimated using ordinary least squares. 
* **For ARMA models with an MA component, estimation is more complex and typically requires numerical optimization methods like maximum likelihood estimation.**

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Goals of ARMA Modeling)</span></p>

Once an ARMA model is fitted, it can be used for:

  * **Goodness-of-Fit:** Assess how well the model describes the temporal structure of the process.
  * **Stationarity Analysis:** Determine if the process properties are stable over time.
  * **Memory and Dependence:** The orders $p$ and $q$ define a "memory horizon."
  * **Hypothesis Testing:** Test the significance of specific coefficients (e.g., $H_0: a_i = 0$).
  * **Forecasting:** Predict future values of the time series.
  * **Control:** Understand how to steer the system towards a desired state.

</div>

## Fourier Transform

TODO: [Fourier Transform](/subpages/model-based-time-series-analysis/fourier_transform/)

## Vector Autoregressive (VAR) Models

### Introduction to Multivariate Time Series

In contrast to univariate analysis, multivariate time series analysis considers datasets where multiple variables are recorded simultaneously over time.

- **Multivariate Time Series Data:** A vector $X_t$ representing observations at time $t$. $X_t = (X_{1t}, \dots, X_{Nt})^\top \in \mathbb{R}^N$. Here, $N$ is the number of simultaneously recorded variables.
- **New Phenomena of Interest:** The primary advantage of the multivariate approach is the ability to model interactions between time series. A key phenomenon is the cross-correlation between the different component series.

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/var3.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>VAR(3)</figcaption>
</figure>

### The VAR($p$) Model Architecture

The Vector Autoregressive (VAR) model is a natural extension of the univariate autoregressive (AR) model to multivariate time series. A VAR($p$) model expresses each variable as a linear function of its own past values, the past values of all other variables in the system, and a random error term.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(VAR($p$) Model)</span></p>

The general form of a **VAR($p$) model** is:

$$X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t$$

- **Intercept:** $c \in \mathbb{R}^N$ is the intercept vector.
- **Coefficient Matrices:** $A_i \in \mathbb{R}^{N \times N}$ are the coefficient matrices for each lag $i=1, \dots, p$.
- **Error Term:** $\varepsilon_t$ is a vector of white noise error terms, typically assumed to be multivariate normal: $\varepsilon_t \sim \mathcal{N}(0, \Sigma_\varepsilon)$.

The **covariance matrix of the error** term, $\Sigma_\varepsilon$, is given by:

$$
\Sigma_\varepsilon = \mathbb{E}[\varepsilon_t \varepsilon_t^\top] =
\begin{pmatrix}
  \mathbb{E}[\varepsilon_{1t}^2] & \mathbb{E}[\varepsilon_{1t}\varepsilon_{2t}] & \dots \\
  \vdots & \ddots & \vdots \\
  \mathbb{E}[\varepsilon_{Nt}\varepsilon_{1t}] & \dots & \mathbb{E}[\varepsilon_{Nt}^2]
\end{pmatrix}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Empirical Covariance)</span></p>

For the VAR($p$) process, the **empirical covariance** based on the path of the length $T$ is 

$$\hat{\Sigma_\varepsilon} = \frac{1}{T - (p + 1)} \sum_{t=p+2}^T \epsilon_t \epsilon_t^\top$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Off-diagonal elements of $\Sigma_\varepsilon$)</span></p>

Crucially, the off-diagonal elements of $\Sigma_\varepsilon$ are allowed to be non-zero, meaning the contemporaneous error terms for different variables can be correlated.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Structure of Coefficient Matrices)</span></p>

Each coefficient matrix $A_i$ captures the influence of variables at lag $i$ on the current state of the system.

$$
A_i = \begin{pmatrix}
a_{11}^{(i)} & \dots & a_{1N}^{(i)} \\
\vdots & \ddots & \vdots \\
a_{N1}^{(i)} & \dots & a_{NN}^{(i)}
\end{pmatrix}
$$

- **Diagonal Entries ($a_{jj}^{(i)}$):** 
  * These entries relate a variable to its own past. 
  * They capture the internal time constants and autoregressive properties of each individual series.
- **Off-Diagonal Entries ($a_{jk}^{(i)}$ for $j \neq k$):** 
  * These entries quantify how the past of variable $k$ influences the present of variable $j$. 
  * They are the key to understanding the cross-series dynamics and interactions. 
  * This is the basis for concepts like Granger causality.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Equivalence and Companion Form)</span></p>

1. Any **scalar AR($p$) process** can be written as a **$p$-variate VAR(1) process**.
2. Any **VAR($p$) process in $K$ variables** can be written as a **$Kp$-variate VAR(1) process**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Significance of the equivalence)</span></p>

* A significant theoretical result is that any VAR($p$) process can be rewritten as a VAR(1) process. **This is extremely useful for analysis, particularly for assessing model stability.**
* This transformation allows us to study the stability and properties of a high-order model by analyzing a **single, larger coefficient matrix corresponding to the VAR(1) representation.**

</div>

### Stationarity of VAR Processes

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Stationarity $\implies$ spectral radius of $A$ is less than 1)</span></p>

* The stability of a VAR process is determined by the properties of its coefficient matrices. 
* For a VAR(1) process, the condition for stationarity is based on the eigenvalues of the coefficient matrix.
* For the VAR(1) process $X_t = c + AX_{t-1} + \varepsilon_t$, the necessary and sufficient condition for stationarity

$$\max_i \lvert \lambda_i(A) \rvert < 1$$

where $\lambda_i(A)$ are the eigenvalues of $A$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <h4>Proof Sketch for Stationarity</h4>
    <ol>
      <li><strong>Iterative Substitution:</strong> Consider the process without the intercept and noise terms for simplicity: $X_t = AX_{t-1}$. By iterating backwards, we can express $X_t$ in terms of an initial state $X_0$:</li>
    </ol>
    $$
    X_t = A X_{t-1} = A (A X_{t-2}) = A^2 X_{t-2} = \dots = A^t X_0
    $$
    <ol>
      <li><strong>Eigendecomposition:</strong> We can decompose the matrix $A$ into its eigenvalues and eigenvectors: $A = V \Lambda V^{-1}$ where $\Lambda$ is a diagonal matrix containing the eigenvalues $\lambda_i$, and $V$ is the matrix of corresponding eigenvectors.</li>
      <li><strong>Power of $A$:</strong> Using the eigendecomposition, the $t$-th power of $A$ is:</li>
    </ol>
    $$
    A^t = (V \Lambda V^{-1})^t = (V \Lambda V^{-1})(V \Lambda V^{-1}) \dots = V \Lambda^t V^{-1}
    $$
    <ol>
      <li><strong>System Evolution:</strong> Substituting this back into the expression for $X_t$:</li>
    </ol>
    $$
    X_t = V \Lambda^t V^{-1} X_0
    $$
    <ol>
      <li><strong>Condition for Stability:</strong> The system is stable (i.e., stationary) if $X_t \to 0$ as $t \to \infty$. This requires that $A^t \to 0$. This, in turn, depends on the behavior of $\Lambda^t$, which is a diagonal matrix with entries $\lambda_i^t$.</li>
    </ol>
    <ul>
      <li>If $\max_i(\lvert\lambda_i\rvert) < 1$, then all $\lambda_i^t \to 0$ as $t \to \infty$. Consequently, $\Lambda^t \to 0$, $A^t \to 0$, and the process is stable and stationary.</li>
      <li>If $\max_i(\lvert\lambda_i\rvert) > 1$, at least one eigenvalue has a modulus greater than 1. Its corresponding term $\lambda_i^t$ will grow exponentially, causing $X_t$ to explode along the direction of the corresponding eigenvector. The process is non-stationary (divergent).</li>
      <li>If $\max_i(\lvert\lambda_i\rvert) = 1$, the system is marginally stable. This can lead to behaviors like a random walk.</li>
    </ul>
  </details>
</div>

### Parameter Estimation

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Parameter Estimation for VAR)</span></p>

The parameters of a VAR($p$) model $(c, A_1, \dots, A_p, \Sigma_\varepsilon)$ can be estimated using Maximum Likelihood Estimation (MLE).

Given a VAR($p$) model:

$$X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t$$

The log-likelihood for a sequence of observations $X_1, \dots, X_T$ is then:

$$\ell(\theta) = \sum_{t=p+1}^{T} \log p(X_t \mid X_{t-1}, \dots, X_{t-p}, \theta)$$

Maximizing this log-likelihood function provides the parameter estimates.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian errors)</span></p>

In the case of Gaussian errors, MLI is equivalent to multivariate least squares.

Given a VAR($p$) model:

$$X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \Sigma_\varepsilon)$$

$$p(X_t \mid X_{t-1}, \dots, X_{t-p}, \theta) = \mathcal{N}\left(c + \sum_{i=1}^{p} A_i X_{t-i}, \Sigma_\varepsilon\right)$$

where $\theta$ represents all model parameters.

The log-likelihood for a sequence of observations $X_1, \dots, X_T$ is then:

$$\ell(\theta) = \sum_{t=p+1}^{T} \log p(X_t \mid X_{t-1}, \dots, X_{t-p}, \theta)$$

For Gaussian errors, this can be framed as a multivariate linear regression problem.

</div>

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

## Nonlinear Dynamical Systems

This chapter introduces the fundamental concepts of nonlinear dynamical systems, moving beyond the linear framework of VAR models to explore more complex behaviors like fixed points, cycles, and chaos.

### Motivation: From Linear to Nonlinear Systems

A VAR(1) model, 

$$X_t = c + AX_{t-1}$$

is a **Linear Dynamical System (LDS)**. We can generalize this by replacing the linear function with a nonlinear function $F(\cdot)$, such as a recurrent neural network (RNN).

$$X_t = F(X_{t-1})$$

This defines a nonlinear dynamical system. Understanding the behavior of simple nonlinear systems provides the foundation for analyzing more complex models.

### Analysis of 1D Systems

Let's start with a simple first-order nonlinear difference equation: $x_{t+1} = f(x_t)$.

#### Fixed Points

A central concept is the fixed point (FP), a state where **the system remains unchanged over time**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed point of map)</span></p>

A point $x^{\ast}$ is a **fixed point of the map** $f$ if it satisfies the equation:
  
$$x^{\ast} = f(x^{\ast})$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Fixed point of Linear AR(1))</span></p>

For $f(x) = \alpha x + c$, the fixed point equation is 

$$x^{\ast} = \alpha x^{\ast} + c$$

Solving for $x^{\ast}: x^{\ast}(1-\alpha) = c$, which gives 

$$x^{\ast} = \dfrac{c}{1-\alpha}$$ 

provided $\alpha \neq 1$.

</div>

#### Stability of Fixed Points

The behavior of the system near a fixed point determines its stability.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stable FP = Attractor, Unstable FP = Repeller)</span></p>

* A **stable fixed point**, or attractor, is a point such that trajectories starting in its vicinity converge to it:
  * $x^{\ast}$ is an attractor if there exists a set $C$, called the basin of attraction, such that for any initial condition $x_0 \in C$, $\lim_{t \to \infty} x_t = x^{\ast}$
* An **unstable fixed point**, or repeller, is a point from which nearby trajectories diverge.
  * $x^{\ast}$ is a repeller if there exists a set $C$, such that for any initial condition $x_0 \in C$, $\lim_{t \to -\infty} x_t = x^{\ast}$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Stability of FP of Linear System)</span></p>

For the linear system $x_{t+1} = \alpha x_t + c$, the fixed point $x^{\ast}$ is:

- **Stable** if $\lvert\alpha\rvert < 1$. Trajectories converge to $x^{\ast}$.
- **Unstable** if $\lvert\alpha\rvert > 1$. Trajectories diverge from $x^{\ast}$.
- **Neutrally** stable if $\lvert\alpha\rvert = 1$.

</div>

#### Cycles

A system may not settle on a single point but may instead visit a set of points periodically.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($p$-cycle)</span></p>

A set of $p$ distinct points $\lbrace x_1^{\ast}, \dots, x_p^{\ast}\rbrace$ that the system visits in sequence: 

$$f(x_1^{\ast}) = x_2^{\ast}, \dots, f(x_p^{\ast}) = x_1^{\ast}$$

A point on a **$p$-cycle is a fixed point of the $p$-th iterated map** $f^p(\cdot)$. That is, 

$$x^{\ast} = f^p(x^{\ast})$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(2-cycle and neutrally stable cycle)</span></p>

For the map 

$$x_{t+1} = -x_t + c$$ 

if we start at $x_0$, then 

$$x_1 = -x_0+c$$

$$x_2 = -x_1+c = -(-x_0+c)+c = x_0$$

Every point is part of a 2-cycle. This is a **neutrally stable cycle**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(1D Linear System)</span></p>

1. **Convergence** to a solitary, stable fixed point (attractor) for $\lvert\alpha\rvert < 1$.
2. **Divergence** from an isolated, unstable fixed point for $\lvert\alpha\rvert > 1$.
3. An infinite set of **neutrally stable fixed points** (e.g., a line) if $\alpha=1$, $c=0$.
4. **No fixed point or cycle** (linear drift) if $\alpha=1$, $c \neq 0$.
5. An infinite set of **neutrally stable cycles** if $\alpha=-1$.

</div>

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case1.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case2.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case3.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case4.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case5.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Multivariate Extension (Linear Case))</span></p>

For a **multivariate linear system** 

$$X_t = AX_{t-1} + c$$

where $X_t \in \mathbb{R}^N$:

- **Fixed Point:** 
  
  $$X^{\ast} = AX^{\ast} + c$$
  
  $$(I-A)X^{\ast} = c$$
  
  $$X^{\ast} = (I-A)^{-1}c$$ 
  
  provided $(I-A)$ is invertible.
- **Stability:** The stability of $X^{\ast}$ is determined by the eigenvalues of the Jacobian matrix, which is simply $A$.
  1. $X^{\ast}$ is a **stable fixed point** if $\max_i(\lvert\lambda_i(A)\rvert) < 1$.
  2. $X^{\ast}$ is **unstable** if $\max_i(\lvert\lambda_i(A)\rvert) > 1$.
  3. $X^{\ast}$ is **neutrally stable** if $\max_i(\lvert\lambda_i(A)\rvert) = 1$.

</div>


### The Logistic Map: A Case Study in Nonlinearity


<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Logistic Map)</span></p>

The **logistic map** is a simple, archetypal example of a nonlinear system that exhibits complex behavior, including chaos.

- **Equation:**
  
  $$x_{t+1} = f(x_t) = \alpha x_t (1 - x_t)$$

- **Constraints:** We consider initial conditions $x_0 \in [0, 1]$ and the parameter $\alpha \in [0, 4]$. These constraints ensure that if $x_t$ is in $[0, 1]$, then $x_{t+1}$ will also be in $[0, 1]$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Fixed Points of the Logistic Map)</span></p>

The logistic map has two fixed points:

1. $x_1^{\ast} = 0$
2. $x_2^{\ast} = \dfrac{\alpha - 1}{\alpha}$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    $$x^{\ast} = f(x^{\ast}) = \alpha x^{\ast} (1 - x^{\ast})$$
    $$x^{\ast} - \alpha x^{\ast} + \alpha (x^{\ast})^2 = 0 \implies x^{\ast}(1 - \alpha + \alpha x^{\ast}) = 0$$
    <p>This gives two fixed points:</p>
    <ol>
      <li>FP1: $x_1^{\ast} = 0$.</li>
      <li>FP2: $1 - \alpha + \alpha x^{\ast} = 0 \implies x_2^{\ast} = \dfrac{\alpha - 1}{\alpha}$. This fixed point is only physically relevant (i.e., in our state space $[0,1]$) when $\alpha \ge 1$.</li>
    </ol>
  </details>
</div>

#### Formal Stability Analysis via Linearization

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Stability Analysis</span><span class="math-callout__name">(Univariate Linearization)</span></p>

For a univariate system $x_t = f(x_{t-1})$ with fixed point $x^{\ast} = f(x^{\ast})$, the stability is determined by linearizing around $x^{\ast}$. The **evolution of a small perturbation** $\varepsilon_t$ is **governed by the derivative** $f'$ of $f$ evaluated at $x^{\ast}$.

$$\varepsilon_{t+1} \approx f'(x^{\ast}) \varepsilon_t$$

The fixed point $x^{\ast}$ is **stable if absolute value of $f'(x^{\ast})$ is less than 1**: 

$$\lvert f'(x^{\ast})\rvert < 1$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>To analyze the stability of a fixed point $x^{\ast}$ for a general nonlinear map $f(x)$, we consider a small perturbation $\varepsilon_t$ around the fixed point:</p>
    $$x_t = x^{\ast} + \varepsilon_t$$
    $$x_{t+1} = x^{\ast} + \varepsilon_{t+1} = f(x^{\ast} + \varepsilon_t)$$
    <p>Using a first-order Taylor expansion of $f(x)$ around $x^{\ast}$:</p>
    $$f(x^{\ast} + \varepsilon_t) \approx f(x^{\ast}) + f'(x^{\ast}) \cdot \varepsilon_t$$
    <p>Since $f(x^{\ast}) = x^{\ast}$, this simplifies to:</p>
    $$x^{\ast} + \varepsilon_{t+1} \approx x^{\ast} + f'(x^{\ast}) \cdot \varepsilon_t \implies \varepsilon_{t+1} \approx f'(x^{\ast}) \cdot \varepsilon_t$$
    <p>This is a linear difference equation for the perturbation $\varepsilon_t$. The perturbation will decay to zero (i.e., the FP is stable) if $\lvert f'(x^{\ast})\rvert < 1$.</p>
  </details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Stability Analysis of Logistic Map)</span></p>

- **Applying to the Logistic Map:** The derivative is $f'(x) = \alpha(1-2x)$.
  1. At FP1 $x_1^{\ast}=0$: 
   
    $$f'(0) = \alpha(1-0) = \alpha$$

    - The fixed point at $0$ is stable if $\lvert f'(0)\rvert < 1$, which means $0 \le \alpha < 1$.
  2. At FP2 $x_2^{\ast} = (\alpha-1)/\alpha$: 
 
    $$f'(x_2^{\ast}) = \alpha\left(1 - 2\frac{\alpha-1}{\alpha}\right) = \alpha\left(\frac{\alpha - 2\alpha + 2}{\alpha}\right) = 2-\alpha$$
    
    - This fixed point is stable if $\lvert f'(x_2^{\ast})\rvert = \lvert 2-\alpha\rvert < 1$. This inequality holds for $1 < \alpha < 3$.

- **Stability Summary:**
  - For $0 \le \alpha < 1$: One stable FP at $x^{\ast}=0$.
  - For $1 < \alpha < 3$: The FP at $x^{\ast}=0$ becomes unstable, and a new stable FP appears at $x^{\ast}=(\alpha-1)/\alpha$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Stability Analysis</span><span class="math-callout__name">(Multivariate Linearization)</span></p>

For a multivariate system $X_t = F(X_{t-1})$ with fixed point $X^{\ast} = F(X^{\ast})$, the stability is determined by linearizing around $X^{\ast}$. The **evolution of a small perturbation** $\mathcal{E}_t$ is **governed by the Jacobian matrix** $J$ of $F$ evaluated at $X^{\ast}$.

$$\mathcal{E}_{t+1} \approx J(X^{\ast}) \mathcal{E}_t$$

The fixed point $X^{\ast}$ is **stable if all eigenvalues of the Jacobian matrix have a modulus less than 1**: 

$$\max_i(\lvert\lambda_i(J(X^{\ast}))\rvert) < 1$$

</div>

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case1.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case2.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case3.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case4.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case5.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
</div>

### Bifurcation and Chaos

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Bifurcation, Period-Doubling and Chaos)</span></p>

As the parameter $\alpha$ increases beyond $3$ in the logistic map, the system's behavior undergoes a series of qualitative changes known as **bifurcations**.

- **Period-Doubling:** At $\alpha=3$, the fixed point $x_2^{\ast}$ becomes unstable, and a stable 2-cycle emerges. As $\alpha$ increases further, this 2-cycle becomes unstable and bifurcates into a stable 4-cycle, then an 8-cycle, and so on. This cascade is known as the "period-doubling route to chaos."
- **Chaos:** For larger values of $\alpha$ (e.g., $\alpha \gtrsim 3.57$), the system's behavior becomes chaotic.
  - The trajectory is aperiodic and appears irregular or random, but it is still **fully deterministic**.
  - A key feature is sensitivity to initial conditions: two trajectories starting arbitrarily close to each other will diverge exponentially fast.

</div>

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/bifurcation_diagram.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/bifurcation_graph.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
</div>

#### Chaotic Attractors and the Lyapunov Exponent

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Chaotic Attractor)</span></p>

A set $A$ is a **chaotic attractor** if trajectories starting within its basin of attraction converge to $A$, and within $A$, the dynamics are chaotic and sensitive to initial conditions.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lyapunov Exponent)</span></p>

**Lyapunov Exponent** $\lambda$ quantifies the rate of separation of infinitesimally close trajectories. For a 1D map $f(x)$, it is defined as:
  
$$\lambda(x_0) = \lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \ln\lvert f'(x_t)\rvert$$

- $\lambda > 0$: **Exponential divergence**, a signature of chaos.
- $\lambda < 0$: **Exponential convergence**, corresponding to a stable fixed point or cycle.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implications for Prediction and Modeling)</span></p>

The existence of chaotic dynamics has profound implications for time series modeling:

1. **Prediction Horizon:** Sensitivity to initial conditions makes long-term prediction fundamentally impossible, as any tiny error in measuring the current state will be exponentially amplified.
2. **Chaos vs. Noise:** It can be extremely difficult to distinguish between a deterministic chaotic process and a stochastic (noisy) process based on observed data alone.
3. **Loss Functions:** Traditional loss functions like Mean Squared Error (MSE) may be problematic for evaluating models of chaotic systems, as even a perfect model will produce trajectories that diverge from the data due to initial condition uncertainty.
4. **Parameter Estimation:** The loss landscapes for models of chaotic systems can be highly non-convex and irregular, making optimization and parameter estimation very challenging.

</div>

## Latent Variable Models

So far, models have been of the form: 

$$x_t = f_\theta(x_{t-1}, \dots, x_{t-d})$$

where $x_t \in \mathbb{R}^N$. This chapter introduces a latent variable, $z_t \in \mathbb{R}^M$, to model the underlying state of the system.

### Architecture

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Latent Variable Model)</span></p>

**Latent variable models**, also known as **State-Space Models (SSMs)**, are composed of two primary components:

1. **Latent Model (or Process Model):** This describes the evolution of the unobserved state over time.
  * Transition probability: $z_t \sim p_{\text{lat}}(z_t \mid z_{t-1}, \theta)$
  * Initial condition: $p(z_1)$
2. **Observation Model:** This describes how the observed data is generated from the current unobserved state.
  * Emission probability: $x_t \sim p_{\text{obs}}(x_t \mid z_t, \theta)$

The complete set of model parameters is $\theta = [\theta_{\text{lat}}, \theta_{\text{obs}}]$.

</div>

To make inference tractable, two key assumptions are made:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(SSM core assumptions)</span></p>

State-space models are built upon two critical assumptions that simplify the probabilistic structure of the time series.

* **Markov Property:** The current state $z_t$ is conditionally independent of all past states given the immediately preceding state $z_{t-1}$.  
  
  $$p(z_t \mid z_1, \dots, z_{t-1}) = p(z_t \mid z_{t-1})$$

* **Conditional Independence of Observations:** The current observation $x_t$ is conditionally independent of all past states and observations given the current state $z_t$.  
  
  $$p(x_t \mid x_1, \dots, x_{t-1}, z_1, \dots, z_t) = p_{\text{obs}}(x_t \mid z_t)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Joint distribution under assumptions above)</span></p>

Under these assumptions, the **joint distribution** over a sequence of observations $X_{1:T}$ and latent states $Z_{1:T}$ factorizes as follows:  

$$p(X_{1:T}, Z_{1:T} \mid \theta) = p(z_1) \left( \prod_{t=2}^{T} p_{\text{lat}}(z_t \mid z_{t-1}, \theta) \right) \left( \prod_{t=1}^{T} p_{\text{obs}}(x_t \mid z_t, \theta) \right)$$ 

</div>

This structure is represented by the following graphical model:

```markdown
 z_1 ---> z_2 ---> ... ---> z_t
  |       |                 |
  |       |                 |
  v       v                 v
 x_1     x_2               x_t
```

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Primary goals when working with LVMs)</span></p>

The primary goals when working with latent variable models are:

1. **Learning:** Learn the parameters $\theta$ from the observed data $X_{1:T}$.
2. **Inference:** Infer the latent trajectory, such as a point estimate $\mathbb{E}[Z_{1:T} \mid X_{1:T}]$.
3. **Full Posterior Inference:** Infer the full posterior distribution over the latent paths, $p(Z_{1:T} \mid X_{1:T})$.

</div>

### Inference Problem

For Maximum Likelihood Estimation (MLE), the objective is to maximize the log-likelihood of the observed data:  

$$\log p_\theta(X_{1:T})$$  

However, this requires marginalizing out the latent variables, which involves a high-dimensional integral:  

$$p_\theta(X_{1:T}) = \int p_\theta(X_{1:T}, Z_{1:T}) dZ_1 dZ_2 \dots dZ_T$$  

Applying the logarithm directly to this integral is problematic as the log cannot be pushed inside the integral:  

$$\log p_\theta(X) = \log \left( \int p_\theta(X, Z) dZ \right)$$  

This makes direct optimization difficult.

### Evidence Lower Bound (ELBO)

Let $X = X_{1:T}$ and $Z = Z_{1:T}$. We can introduce an arbitrary **"proposal density"** $q(Z)$ to derive a **lower bound on the log-likelihood**.

#### Two Equivalent Forms of the ELBO

The ELBO can be expressed in two common, equivalent forms.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(ELBO = Expected Joint + Entropy )</span></p>

   $$\text{ELBO}(q, p_\theta) = \mathbb{E}_{q(Z\mid X)}[\log p_\theta(X, Z)] - \mathbb{E}_{q(Z\mid X)}[\log q(Z\mid X)]$$ 
   
   This can be rewritten using the definition of entropy, 
   
   $$\text{ELBO}(q, p_\theta) = \mathbb{E}_{q(Z\mid X)}[\log p_\theta(X, Z)] + H(q(Z\mid X)),$$ 

   where $H(q(Z\mid X)) = -\mathbb{E}_{q(Z\mid X)}[\log q(Z\mid X)]$.
   
   The entropy term favors solutions where the proposal distribution $q(Z)$ is not overly confident (i.e., has high entropy).

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Starting with the log-likelihood:</p>
    $$\log p_\theta(X) = \log \int p_\theta(X, Z) dZ$$
    $$\log p_\theta(X) = \log \int q(Z) \frac{p_\theta(X, Z)}{q(Z)} dZ = \log \mathbb{E}_{q(Z)}\left[ \frac{p_\theta(X, Z)}{q(Z)} \right]$$
    <p><strong>Jensen's Inequality:</strong> For a concave function $f$: $f(\mathbb{E}[Y]) \ge \mathbb{E}[f(Y)]$.</p>
    <p>Since the logarithm is a concave function, we can apply Jensen's inequality:</p>
    $$\log \mathbb{E}_{q(Z)}\left[ \frac{p_\theta(X, Z)}{q(Z)} \right] \ge \mathbb{E}_{q(Z)}\left[ \log \frac{p_\theta(X, Z)}{q(Z)} \right]$$
    <p>This gives us the Evidence Lower Bound (ELBO):</p>
    $$\log p_\theta(X) \ge \mathbb{E}_{q(Z)}\left[ \log p_\theta(X, Z) \right] - \mathbb{E}_{q(Z)}\left[ \log q(Z) \right]$$
  </details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(ELBO = Log-Likelihood - KL)</span></p>

$$\begin{aligned} \text{ELBO}(q, p_\theta) &= \int q(Z) \log \frac{p_\theta(X, Z)}{q(Z)} dZ \\ &= \int q(Z) \log \frac{p_\theta(Z\mid X) p_\theta(X)}{q(Z)} dZ \\ &= \int q(Z) \log p_\theta(X) dZ + \int q(Z) \log \frac{p_\theta(Z\mid X)}{q(Z)} dZ \\ &= \log p_\theta(X) \int q(Z) dZ - \int q(Z) \log \frac{q(Z)}{p_\theta(Z\mid X)} dZ \\ &= \log p_\theta(X) - \text{KL}(q(Z) \parallel  p_\theta(Z\mid X)) \end{aligned}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of the second form of ELBO)</span></p>

From the second form of the ELBO, we see that if $\text{KL}(q(Z) \parallel p_\theta(Z\mid X)) = 0$ (i.e., the proposal density $q(Z)$ is equal to the true posterior $p_\theta(Z\mid X)$), the bound becomes exact: $\text{ELBO} = \log p_\theta(X)$. Therefore, maximizing the ELBO becomes equivalent to maximizing the log-likelihood.

</div>

### Two Inference Strategies Suggested by the ELBO

#### Expectation-Maximization (EM)

If the true posterior $p_\theta(Z\mid X)$ is tractable, we can choose our proposal density to be exactly this posterior: $q(Z) = p_\theta(Z\mid X)$. This choice makes the KL divergence term in the ELBO zero, turning the lower bound into an equality. This is the core idea behind the Expectation-Maximization (EM) algorithm.

The EM algorithm consists of two alternating steps:

1. **E-Step:** Compute the exact posterior over the latent variables given the current parameter estimate $\theta^{\ast}$.
2. **M-Step:** Maximize the expected complete-data log-likelihood with respect to the parameters $\theta$, using the posterior computed in the E-Step.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variational Inference)</span></p>

If the true posterior $p_\theta(Z\mid X)$ is intractable, we restrict $q(Z)$ to a tractable family of distributions (e.g., fully-factorized Gaussians). We then jointly optimize the ELBO with respect to both the parameters of the proposal distribution $q$ and the model parameters $\theta$:

$$\max_{q, \theta} \text{ELBO}(q, \theta)$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Expectation–Maximization (EM))</span></p>

The EM algorithm is an iterative procedure for finding maximum-likelihood estimates in latent-variable models by alternating between updating a distribution over latent variables and updating parameters.

1. **Initialize** parameters $\theta^{(0)}$.
2. **Repeat** until convergence (let $\theta^{old}$ denote the current parameters):

* **E-step (Expectation):** Fix $\theta^{old}$ and choose a distribution over latents **conditioned on the observed data** $X$ by maximizing the ELBO. The optimum is the true posterior under the current parameters (for exact EM), which makes the bound tight.
  
  $$q^{\ast}(Z\mid X)=\arg\max_{q} \mathrm{ELBO}\big(q(Z\mid X),\theta^{old}\big)$$
  
  $$\iff\quad q^{\ast}(Z\mid X)=\arg\min_{q} \text{KL}\left(q(Z\mid X) \parallel p_{\theta^{old}}(Z\mid X)\right)$$
  
  Hence (if exact EM, meaning we can compute $p_{\theta^{old}}(Z\mid X)$):
  
  $$q^{\ast}(Z\mid X)=p_{\theta^{old}}(Z\mid X)$$

* **M-step (Maximization):** Fix $q^{\ast}(Z\mid X)$ and update parameters by maximizing the ELBO with respect to $\theta$:
  
  $$\theta^{new}=\arg\max_{\theta} \mathrm{ELBO}\big(q^{\ast}(Z\mid X),\theta\big)$$
  
  Since the entropy term of $q^{\ast}$ does not depend on $\theta$, this is equivalent to maximizing the expected complete-data log-likelihood:
  
  $$\theta^{new}=\arg\max_{\theta} \mathbb{E}_{q^{\ast}(Z\mid X)}\big[\log p_{\theta}(X,Z)\big].$$
  
</div>

### Linear State-Space Models

A Linear Dynamical System (LDS) is a specific type of state-space model where the transition and observation models are linear functions with Gaussian noise.

The graphical model is:

```markdown
 z_1 ---> z_2 ---> ... ---> z_t
  |       |                 |
  |       |                 |
  v       v                 v
 x_1     x_2               x_t
```

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Linear State-Space Models)</span></p>

1. **Observation Model (Linear Gaussian):** The observation $x_t \in \mathbb{R}^N$ is a linear combination of the latent state $z_t \in \mathbb{R}^M$ plus Gaussian noise. 
   
   $$x_t = B z_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Gamma)$$ 
   
   So, the conditional distribution is: 
   
   $$p(x_t \mid z_t) = \mathcal{N}(B z_t, \Gamma)$$
   
2. **Latent Model / Process Model (Linear Gaussian):** The latent state $z_t \in \mathbb{R}^M$ evolves as a linear function of the previous state $z_{t-1}$ plus Gaussian noise.
   
   $$z_t = A z_{t-1} + \omega_t, \quad \omega_t \sim \mathcal{N}(0, \Sigma)$$
   
   So, the conditional distribution is: 
   
   $$p(z_t \mid z_{t-1}) = \mathcal{N}(A z_{t-1}, \Sigma)$$

3. **Initial Distribution:** The initial state is drawn from a Gaussian distribution. 
   
   $$z_1 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

The model parameters are:

* **Observation parameters:** $\theta_{\text{obs}} = \lbrace B, \Gamma\rbrace$
* **Latent parameters:** $\theta_{\text{lat}} = \lbrace A, \Sigma, \mu_0, \Sigma_0\rbrace$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Parameters of init distribution are inferred from data)</span></p>

* $\mu_0$ and $\Sigma_0$ are usually **inferred from the data we have**.
* For convenience, we will assume $\Sigma_0 = \Sigma$. Additionally, external inputs could be added to these models, but they are omitted here for simplicity.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Latent variable and observation)</span></p>

* The **true state** $z_t$ requires both position and velocity.
* We only observe a noisy GPS measurement of the position $x_t$. This is an incomplete **observation** of the true state.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(General LDS Vocabulary)</span></p>

* **Smoothing (Offline Inference):** The goal is to infer past states based on all available observations up to time $T$: 
  
  $$p(z_t \mid x_1, \dots, x_T), \quad t < T$$

* **Filtering (Online Inference):** The goal is to infer the current state using observations up to the current time step: 
  
  $$p(z_t \mid x_1, \dots, x_t)$$

* **Prediction (Forecasting):** The goal is to infer future states based on past observations: 
  
  $$p(z_t \mid x_1, \dots, x_T), \quad t > T$$

* **Most Probable Path:** The goal is to infer the single "best" latent trajectory (Maximum A Posteriori estimate): 
  
  $$Z_{1:T}^* = \arg\max_{Z_{1:T}} p(Z_{1:T} \mid X_{1:T})$$

* **Learning (Parameter Estimation):** The goal is to estimate the model parameters:
    
    $$\theta_{\text{lat}} = \lbrace A, \Sigma, \mu_0, \Sigma_0\rbrace, \quad \theta_{\text{obs}} = \lbrace B, \Gamma\rbrace$$

</div>

#### EM for LDS Models

The EM algorithm alternates between the E-step and M-step to learn the parameters of an LDS.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">EM for LDS Models</span><span class="math-callout__name">(M-Step)</span></p>

Let $X=\lbrace X_{1:T}\rbrace$ and $Z=\lbrace Z_{1:T}\rbrace$.

**Goal:** $\theta^* = \max_\theta \mathbb{E}_q[\log p(X, Z)]$.

$$A = \left(\sum_{t=2}^T \mathbb{E}_q[z_t z_{t-1}^\top]\right) \left(\sum_{t=2}^T \mathbb{E}_q[z_{t-1} z_{t-1}^\top]\right)^{-1}$$

The M-step for the other parameters ($B, \Sigma, \Gamma, \mu_0, \Sigma_0$) proceeds in a similar fashion.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

In the M-step for an LDS, all parameter updates can be written in terms of the first and second moments of the latent variables under the posterior distribution $q(Z) = p(Z\mid X)$. These required moments are:

* $\mathbb{E}[z_t]$ (First moment, across one time step)
* $\mathbb{E}[z_t z_t^\top]$ (Second moment, at one time step)
* $\mathbb{E}[z_t z_{t-1}^\top]$ (Second moment, across two time steps)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name"></span></p>

1. Recall that for a Gaussian variable $y \in \mathbb{R}^k \sim \mathcal{N}(\mu, \Sigma)$, the log-probability is: 

  $$\log p(y) = -\frac{k}{2}\log(2\pi) - \frac{1}{2}\log\lvert \Sigma\rvert - \frac{1}{2}(y-\mu)^\top\Sigma^{-1}(y-\mu)$$

1. $x^\top A y = \text{tr}(A yx^\top)$
2. $\mathbb{E}[\text{tr}(A)] = \text{tr}(\mathbb{E}[A])$
3. $\frac{\partial \text{tr}(CXB)}{\partial X} = C^\top B^\top$
4. $\frac{\partial \text{tr}(X^\top C)}{\partial X} = C$
5. $\frac{\partial \text{tr}(X^\top BXC)}{\partial X} = BXC + B^\top XC^\top$

</div>

<!-- <div class="accordion">
  <details>
    <summary>proof of M-Step for LDS models</summary>
    <h4>Setting the things up</h4>
    <p>The complete-data log-likelihood factorizes due to the Markov and conditional independence properties:</p>
    $$\log p(X, Z) = \log p(z_1) + \sum_{t=2}^T \log p(z_t\mid z_{t-1}) + \sum_{t=1}^T \log p(x_t\mid z_t)$$
    <p>The expectation can be split into three parts:</p>
    $$\mathbb{E}_q[\log p(X, Z)] = \underbrace{\mathbb{E}_q[\log p(z_1)]}_{\text{initial condition}} + \underbrace{\mathbb{E}_q\left[\sum_{t=2}^T \log p(z_t\mid z_{t-1})\right]}_{\text{latent model}} + \underbrace{\mathbb{E}_q\left[\sum_{t=1}^T \log p(x_t\mid z_t)\right]}_{\text{observation model}}$$
    <h4>The objective expansion given the distributions of variables</h4>
    <p>These correspond to the initial distribution, process model, and observation model, respectively.</p>
    <p>Applying this to the LDS model components ($p(z_1)$, $p(z_t\mid z_{t-1})$, $p(x_t\mid z_t)$), the expected complete-data log-likelihood becomes a function of terms like:</p>
    $$\mathbb{E}_q[\log p_\theta(X,Z)] =$$
    $$\mathbb{E}_q[c_1 - \frac{1}{2}\log \lvert \Sigma_0\rvert - \frac{1}{2}(z_1 - \mu_0)^\top \Sigma_0^{-1} (z_1 - \mu_0)]$$
    $$+\mathbb{E}_q[c_2 \underbrace{- \frac{T-1}{2}\log \lvert \Sigma \rvert -\frac{1}{2} \sum_{t=2}^T (z_t - Az_{t-1})^\top \Sigma^{-1} (z_t - Az_{t-1})}_{:= Q(A)}]$$
    $$+\mathbb{E}_q[c_3 - \frac{T}{2}\log \lvert \Gamma \rvert -\frac{1}{2} \sum_{t=1}^T (x_t - Bz_{t})^\top \Gamma^{-1} (x_t - Bz_{t})]$$
    <h4>Derivation of the Update for $A$</h4>
    <p>To find the update for the transition matrix $A$, we differentiate the part of the expected log-likelihood related to the process model, which we denote $Q(A)$, with respect to $A$ and set it to zero.</p>
    <p>The relevant term is:</p>
    $$Q(A) = \mathbb{E}_q[- \frac{T-1}{2}\log \lvert \Sigma \rvert -\frac{1}{2}\sum_{t=2}^T(z_t - Az_{t-1})^\top \Sigma^{-1} (z_t - Az_{t-1})]$$
    $$Q(A) = - \frac{T-1}{2}\log \lvert \Sigma \rvert -\frac{1}{2}\sum_{t=2}^T\mathbb{E}_q[(z_t - Az_{t-1})^\top \Sigma^{-1} (z_t - Az_{t-1})]$$
    <p>Next we use the identities $x^\top A y = \text{tr}(Ayx^\top)$ and $\mathbb{E}[\text{tr}(A)] = \text{tr}(\mathbb{E}[A])$.</p>
    $$Q(A) = - \frac{T-1}{2}\log \lvert \Sigma \rvert -\frac{1}{2}\sum_{t=2}^T\mathbb{E}_q[z_t^\top\Sigma^{-1}z_t - z_{t-1}^\top A^\top\Sigma^{-1}z_t -z_t^\top\Sigma^{-1}Az_{t-1} + z_{t-1}^\top A^\top\Sigma^{-1}Az_{t-1}]$$
    $$Q(A) = - \frac{T-1}{2}\log \lvert \Sigma \rvert -\frac{1}{2}\sum_{t=2}^T \mathbb{E}_{q}[\text{tr}(\Sigma^{-1}z_t^\top z_t)] - \mathbb{E}_{q}[\text{tr}(A^\top\Sigma^{-1}z_t z_{t-1}^\top)] -\mathbb{E}_{q}[\text{tr}(\Sigma^{-1}Az_{t-1}z_t^\top)] + \mathbb{E}_{q}[\text{tr}(A^\top\Sigma^{-1}Az_{t-1}z_{t-1}^\top)]$$
    $$Q(A) = - \frac{T-1}{2}\log \lvert \Sigma \rvert -\frac{1}{2}\sum_{t=2}^T \text{tr}(\Sigma^{-1}\mathbb{E}_{q}[z_t^\top z_t]) - \text{tr}(A^\top\Sigma^{-1}\mathbb{E}_{q}[z_t z_{t-1}^\top]) - \text{tr}(\Sigma^{-1}A\mathbb{E}_{q}[z_{t-1}z_t^\top]) + \text{tr}(A^\top\Sigma^{-1}A\mathbb{E}_{q}[z_{t-1}z_{t-1}^\top])$$
    <p>Dropping terms that are constant with respect to $A$:</p>
    $$\widetilde{Q}(A) = -\frac{1}{2} \sum_{t=2}^T \left( -\text{tr}(A^\top\Sigma^{-1}\mathbb{E}_{q}[z_t z_{t-1}^\top]) \right)$$
    $$- \text{tr}(\Sigma^{-1}A\mathbb{E}_{q}[z_{t-1}z_t^\top])$$
    $$+ \text{tr}(A^\top\Sigma^{-1}A\mathbb{E}_{q}[z_{t-1}z_{t-1}^\top])$$
    <p>Dropping terms that are constant with respect to $A$ and differentiating with respect to $A$ and setting to zero:</p>
    $$\nabla_A \widetilde{Q}(A) = -\frac{1}{2} \sum_{t=2}^T \left( \Sigma^{-1} \mathbb{E}_q[z_t z_{t-1}^\top] - \Sigma^{-1}\mathbb{E}_q[z_t z_{t-1}^\top] + \Sigma^{-1}A\mathbb{E}_q[z_{t-1} z_{t-1}^\top] - \Sigma^{-1} A \mathbb{E}_q[z_{t-1} z_{t-1}^\top] \right)$$
    $$= \Sigma^{-1} \sum_{t=2}^T \mathbb{E}_q[z_t z_{t-1}^\top] - \Sigma^{-1} A \sum_{t=2}^T \mathbb{E}_q[z_{t-1} z_{t-1}^\top] = 0$$
    $$\implies \cancel{\Sigma^{-1}} \sum_{t=2}^T \mathbb{E}_q[z_t z_{t-1}^\top] = \cancel{\Sigma^{-1}} A \sum_{t=2}^T \mathbb{E}_q[z_{t-1} z_{t-1}^\top]$$
    <p>Solving for $A$ yields the update rule:</p>
    $$A = \left(\sum_{t=2}^T \mathbb{E}_q[z_t z_{t-1}^\top]\right) \left(\sum_{t=2}^T \mathbb{E}_q[z_{t-1} z_{t-1}^\top]\right)^{-1}$$
    <p>The M-step for the other parameters ($B, \Sigma, \Gamma, \mu_0, \Sigma_0$) proceeds in a similar fashion.</p>
  </details>
</div> -->

<div class="accordion">
  <details>
    <summary>proof (elegant M-step for LDS via expected sufficient statistics)</summary>

We consider the linear–Gaussian LDS (time-homogeneous):

$$
z_1\sim\mathcal N(\mu_0,\Sigma_0),\qquad
z_t = A z_{t-1}+w_t,\; w_t\sim\mathcal N(0,\Sigma)\ (t\ge 2),
\qquad
x_t = B z_t + \varepsilon_t,\; \varepsilon_t\sim\mathcal N(0,\Gamma).
$$

In EM, the E-step provides $q(Z)=p(Z\mid X,\theta^{old})$ and we maximize

$$Q(\theta) := \mathbb E_q[\log p_\theta(X,Z)]$$

Using the Markov structure,

$$\log p_\theta(X,Z)=\log p(z_1)+\sum_{t=2}^T \log p(z_t\mid z_{t-1})+\sum_{t=1}^T \log p(x_t\mid z_t),$$

so $Q(\theta)$ splits into three independent maximizations: initial, dynamics, observation.

---

### Expected sufficient statistics (from the smoother)
Let

$$
\bar z_t := \mathbb E_q[z_t],\qquad
P_t := \mathbb E_q[z_t z_t^\top],\qquad
P_{t,t-1}:=\mathbb E_q[z_t z_{t-1}^\top].
$$

Also define the sums

$$
S_{00}:=\sum_{t=2}^T \mathbb E_q[z_{t-1}z_{t-1}^\top]=\sum_{t=2}^T P_{t-1},\qquad
S_{10}:=\sum_{t=2}^T \mathbb E_q[z_t z_{t-1}^\top]=\sum_{t=2}^T P_{t,t-1},\qquad
S_{11}:=\sum_{t=2}^T \mathbb E_q[z_t z_t^\top]=\sum_{t=2}^T P_t,
$$

and for the observation part (since $x_t$ is observed/fixed),

$$
S_{xz}:=\sum_{t=1}^T x_t\,\bar z_t^\top,\qquad
S_{zz}:=\sum_{t=1}^T \mathbb E_q[z_t z_t^\top]=\sum_{t=1}^T P_t,\qquad
S_{xx}:=\sum_{t=1}^T x_t x_t^\top.
$$

---

## 1) Dynamics parameters $(A,\Sigma)$

### Update for $A$
The dynamics contribution to $Q$ is

$$
Q_{\text{dyn}}(A,\Sigma)
= -\frac{T-1}{2}\log|\Sigma|
-\frac12 \sum_{t=2}^T \mathbb E_q\!\left[(z_t-Az_{t-1})^\top\Sigma^{-1}(z_t-Az_{t-1})\right]
+\text{const}.
$$

For fixed $\Sigma$, maximizing w.r.t. $A$ is equivalent to minimizing the expected quadratic form.
Using $\mathbb E[u^\top M u]=\mathrm{tr}\!\left(M\,\mathbb E[uu^\top]\right)$,

$$
\sum_{t=2}^T \mathbb E_q[(z_t-Az_{t-1})(z_t-Az_{t-1})^\top]
= S_{11} - A S_{10}^\top - S_{10} A^\top + A S_{00}A^\top.
$$

Thus (dropping constants)

$$
Q_{\text{dyn}}(A,\Sigma)\equiv
-\frac12\,\mathrm{tr}\!\Big(\Sigma^{-1}\big(S_{11}-A S_{10}^\top - S_{10}A^\top + A S_{00}A^\top\big)\Big).
$$

Differentiate w.r.t. $A$ and set to zero (standard matrix calculus):

$$
\frac{\partial}{\partial A}\,\mathrm{tr}\!\big(\Sigma^{-1}A S_{00}A^\top\big)=2\Sigma^{-1}A S_{00},
\qquad
\frac{\partial}{\partial A}\,\mathrm{tr}\!\big(\Sigma^{-1}S_{10}A^\top\big)=\Sigma^{-1}S_{10}.
$$

So the stationarity condition is

$$
2\Sigma^{-1}A S_{00} - 2\Sigma^{-1}S_{10}=0
\quad\Longrightarrow\quad
A S_{00}=S_{10}.
$$

Therefore

$$
\boxed{
A^{new}=S_{10}\,S_{00}^{-1}
=\left(\sum_{t=2}^T \mathbb E_q[z_t z_{t-1}^\top]\right)
\left(\sum_{t=2}^T \mathbb E_q[z_{t-1} z_{t-1}^\top]\right)^{-1}.
}
$$

### Update for $\Sigma$
For fixed $A$, $Q_{\text{dyn}}$ has the form

$$
Q_{\text{dyn}}(\Sigma)
= -\frac{T-1}{2}\log|\Sigma|-\frac12\,\mathrm{tr}\!\big(\Sigma^{-1}\,S_{\text{res}}\big)+\text{const},
$$

where

$$
S_{\text{res}}
:=\sum_{t=2}^T \mathbb E_q\!\left[(z_t-Az_{t-1})(z_t-Az_{t-1})^\top\right].
$$

The maximizer is the empirical covariance of the (expected) residuals:

$$
\boxed{
\Sigma^{new}=\frac{1}{T-1}\,S_{\text{res}}
=\frac{1}{T-1}\sum_{t=2}^T \mathbb E_q\!\left[(z_t-A^{new}z_{t-1})(z_t-A^{new}z_{t-1})^\top\right].
}
$$

If you want it expanded in sufficient statistics:

$$
S_{\text{res}} = S_{11} - A^{new} S_{10}^\top - S_{10}(A^{new})^\top + A^{new} S_{00}(A^{new})^\top.
$$

---

## 2) Observation parameters $(B,\Gamma)$

### Update for $B$
Similarly,

$$
Q_{\text{obs}}(B,\Gamma)
= -\frac{T}{2}\log|\Gamma|
-\frac12\sum_{t=1}^T \mathbb E_q\!\left[(x_t-Bz_t)^\top\Gamma^{-1}(x_t-Bz_t)\right]+\text{const}.
$$

For fixed $\Gamma$, maximizing w.r.t. $B$ gives the normal equations of multivariate linear regression:

$$
\boxed{
B^{new}=S_{xz}\,S_{zz}^{-1}
=\left(\sum_{t=1}^T x_t\,\bar z_t^\top\right)\left(\sum_{t=1}^T \mathbb E_q[z_t z_t^\top]\right)^{-1}.
}
$$

### Update for $\Gamma$
For fixed $B$,

$$
\boxed{
\Gamma^{new}
=\frac{1}{T}\sum_{t=1}^T \mathbb E_q\!\left[(x_t-B^{new}z_t)(x_t-B^{new}z_t)^\top\right].
}
$$

Expanded (useful for implementation):

$$
\Gamma^{new}
=\frac{1}{T}\Big(S_{xx}-B^{new}S_{xz}^\top-S_{xz}(B^{new})^\top+B^{new}S_{zz}(B^{new})^\top\Big).
$$

---

## 3) Initial parameters $(\mu_0,\Sigma_0)$
From

$$Q_{\text{init}}(\mu_0,\Sigma_0)=\mathbb E_q[\log \mathcal N(z_1\mid \mu_0,\Sigma_0)],$$

the maximizers match the posterior mean/covariance at $t=1$:

$$
\boxed{\mu_0^{new}=\bar z_1,\qquad \Sigma_0^{new}=P_1-\bar z_1\bar z_1^\top.}
$$

---

### Interpretation (why this proof is “clean”)
Each M-step is just a **Gaussian linear regression** with “data” replaced by posterior expectations:
- $A,\Sigma$ regress $z_t$ on $z_{t-1}$,
- $B,\Gamma$ regress $x_t$ on $z_t$,
- $\mu_0,\Sigma_0$ match the posterior at the initial time.

  </details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">EM for LDS Models</span><span class="math-callout__name">(E-Step)</span></p>

**Goal:** Minimize the KL divergence between the proposal $q(Z)$ and the true posterior $p_\theta(Z\mid X)$, which means setting $q(Z) = p_\theta(Z\mid X)$.

For an LDS, the posterior distribution $p_\theta(Z\mid X)$ is also Gaussian and can be computed analytically and efficiently. The algorithm for this is the Kalman filter-smoother (developed by Rudolph Kalman in 1960). This algorithm was famously used for navigation in the Apollo program and is now widely used in GPS tracking, self-driving cars, and more.

The Kalman filter-smoother operates in two passes:

1. **Forward Pass (Filtering):** This pass iterates from $t=1$ to $T$, computing the filtered distributions 
   
   $$p(z_t \mid x_1, \dots, x_t)$$

2. **Backward Pass (Smoothing):** This pass iterates from $t=T-1$ down to $1$, using the results of the forward pass to compute the smoothed distributions 
  
  $$p(z_t \mid x_1, \dots, x_T)$$

From these smoothed distributions, we can compute all the moments ($\mathbb{E}[z_t]$, $\mathbb{E}[z_t z_t^\top]$, $\mathbb{E}[z_t z_{t-1}^\top]$) required for the M-step.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Backward Pass (Smoothing): Practical point)</span></p>

To compute $\mathbb{E}[z_t z_{t-1}^\top]$ you typically need **pairwise smoothed marginals** $p(z_{t-1},z_t\mid x_{1:T})$, not just the single-time smoothed $p(z_t\mid x_{1:T})$. The Kalman **smoother** (often specifically the RTS smoother) provides exactly what you need.

</div>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/filtering_smoothing.png' | relative_url }}" alt="Filtering Smoothing Schema" loading="lazy">
  <figcaption>Filtering: forward loop. Smoothing: backward loop.</figcaption>
</figure>

## The Kalman Filter and Smoother

### Foundational Concepts for Bayesian Filtering

The core objective of filtering in time series analysis is to recursively estimate the state of a dynamic system from a series of noisy measurements. We aim to compute the probability distribution of the current state, $z_t$, given all observations up to the current time, $x_1, \dots, x_t$. This is formally expressed as the filtering distribution, $p(z_t \mid x_1, \dots, x_t)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name"></span></p>

The derivation of the filtering equations relies on fundamental principles of probability theory.

1. **Bayes' Rule:** 
   
  $$p(A\mid B) = \frac{p(B\mid A)p(A)}{p(B)}$$

2. **Chain Rule of Probability:**
   
  $$p(A_1, A_2, \dots, A_n) = p(A_n \mid A_1, \dots, A_{n-1}) p(A_1, \dots, A_{n-1})$$

</div>

### The Recursive Nature of Bayesian Filtering

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Bayesian Filtering)</span></p>

$$p(z_t \mid x_1, \dots, x_t) = \frac{p(x_t\mid z_t)\int p(z_t \mid z_{t-1})p(z_{t-1} \mid x_1,\dots,x_{t-1})dz_{t-1}}{p(x_t \mid x_1,\dots,x_{t-1})}$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>The filtering distribution $p(z_t \mid x_1, \dots, x_t)$ can be computed recursively. This means the estimate at time $t$ is an update of the estimate from time $t-1$, incorporating the new observation $x_t$.</p>
    <p>The derivation begins by applying Bayes' rule:</p>
    $$p(z_t \mid x_1, \dots, x_t) = \frac{p(z_t,x_1,\dots,x_t)}{p(x_1,\dots,x_t)}$$
    $$= \frac{p(x_t\mid z_t,\cancel{x_1,\dots,x_{t-1}})p(z_t\mid x_1,\dots,x_{t-1})\cancel{p(x_1,\dots,x_{t-1})}}{p(x_t \mid x_1,\dots,x_{t-1})\cancel{p(x_1,\dots,x_{t-1})}}$$
    $$= \frac{p(x_t\mid z_t)\overbrace{p(z_t\mid x_1,\dots,x_{t-1})}^{\text{prediction}}}{p(x_t \mid x_1,\dots,x_{t-1})}$$
    $$= \frac{p(x_t\mid z_t)\int p(z_t, z_{t-1}\mid x_1,\dots,x_{t-1})dz_{t-1}}{p(x_t \mid x_1,\dots,x_{t-1})}$$
    $$\implies \boxed{p(z_t \mid x_1, \dots, x_t) = \frac{\overbrace{p(x_t\mid z_t)}^{\text{obs. model}}\int \overbrace{p(z_t \mid z_{t-1})}^{\text{lat. model}}\overbrace{p(z_{t-1} \mid x_1,\dots,x_{t-1})}^{\text{filter at } t-1}dz_{t-1}}{p(x_t \mid x_1,\dots,x_{t-1})}}$$
    $$\implies \text{recursive in time}$$
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Recursive structure of Bayesian filtering)</span></p>

This formulation reveals the recursive structure: the predictive distribution at time $t$ is found by integrating the process model $p(z_t \mid z_{t-1})$ against the filtering distribution from the previous step, $p(z_{t-1} \mid x_1, \dots, x_{t-1})$.

</div>

### The Kalman Filter: A Linear Gaussian Solution

The Kalman filter provides an analytical solution to the recursive filtering problem for a specific class of models: the Linear Gaussian State-Space Model.

In this framework, **all relevant distributions are assumed to be Gaussian**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Linear Gaussian State-Space Model)</span></p>

* **Initial Distribution:** The initial state is normally distributed.  
  
  $$z_0 \sim \mathcal{N}(m_0, \Sigma_0)$$

* **Process Model:** The state evolves linearly with additive Gaussian noise.  
  
  $$p(z_t \mid z_{t-1}) = \mathcal{N}(z_t \mid A z_{t-1}, \Sigma_z)$$

* **Observation Model:** The observations are a linear function of the state with additive Gaussian noise.
  
  $$p(x_t \mid z_t) = \mathcal{N}(x_t \mid B z_t, \Gamma)$$

* **Posterior Distribution:** A key property of this model is that if the distribution at $t-1$ is Gaussian, the filtering distribution at time $t$ will also be Gaussian.  
  
  $$p(z_t \mid x_1, \dots, x_t) = \mathcal{N}(z_t \mid m_t, V_t)$$

</div>

#### Derivation of the Predictive Distribution

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Bayesian Filter for Linear Gaussian SSM is Gaussian)</span></p>

<!-- $$p(z_t \mid x_1, \dots, x_{t-1}) = \mathcal{N}(z_t \mid m_t, V_t)$$ -->
$$p(z_t \mid x_1, \dots, x_t) = \mathcal{N}(z_t \mid m_t, V_t)$$

</div>

<!-- <div class="accordion">
  <details>
    <summary>proof</summary>
    $$p(z_t \mid x_1, \dots, x_{t-1}) = \int p(z_t \mid z_{t-1}) p(z_{t-1} \mid x_1, \dots, x_{t-1}) dz_{t-1}$$
    $$\int \mathcal{N}(z_t \mid A z_{t-1}, \Sigma_z) \mathcal{N}(z_{t-1} \mid m_{t-1}, V_{t-1}) dz_{t-1}$$
    <p>The product of two Gaussian functions results in another Gaussian function (up to a scaling constant):</p>
    $$\text{Constant} = (2\pi)^{-\frac{M}{2}}\lvert \Sigma \rvert^{-\frac{1}{2}} \cdot (2\pi)^{-\frac{M}{2}}\lvert V_{t-1} \rvert^{-\frac{1}{2}}$$
    $$\text{Exponent} = -\frac{1}{2} \left( (z_t - Az_{t-1})^\top\Sigma^{-1}(z_t - Az_{t-1}) + (z_{t-1} - \mu_{t-1})^\top V_{t_1}^{-1}(z_{t-1} - \mu_{t-1}) \right)$$
    <p>Collecting terms in $z_{t-1}$:</p>
    $$= -\frac{1}{1} \Bigl[ z_{t-1}^\top \underbrace{(A^\top \Sigma^{-1} A + V^{-1}_{t-1})}_{:= H^{-1}} z_{t-1} - z_{t-1}^\top {(A^\top \Sigma^{-1} z_t + V^{-1}_{t-1}\mu_{t-1})}_{:= H^{-1}\mu}$$
    $$- \underbrace{(z_t^\top A^\top \Sigma^{-1} A + \mu_{t-1}^\top V_{t-1}^{-1})}_{\mu^\top H^{-1}} z_{t-1} + z_t^\top \Sigma^{-1} z_t + \mu_{t-1}^\top V_{t-1}^{-1} \mu_{t-1} \Bigr]$$
    $$= -\frac{1}{2} \Bigl[ z_{t-1}^\top H^{-1} z_{t-1} - z_{t-1}^\top H^{-1} \mu - \mu^\top H^{-1} z_{t-1}^\top$$
    $$+ \mu^\top H^{-1}\mu - \mu^\top H^{-1}\mu$$
    $$+ z_t^\top \Sigma^{-1} z_t + \mu_{t-1}^\top V_{t-1}^{-1} \mu_{t-1} \Bigr]$$
    $$\int (2\pi)^{-\frac{M}{2}} \underbrace{\lvert \Sigma \rvert^{-\frac{1}{2}} \cdot (2\pi)^{-\frac{M}{2}}\lvert V_{t-1} \rvert^{-\frac{1}{2}}}_{:= C}$$
    $$\cdot \underbrace{\exp(-\frac{1}{2}(- \mu^\top H^{-1}\mu + z_t^\top \Sigma^{-1} z_t + \mu_{t-1}^\top V_{t-1}^{-1} \mu_{t-1}))}_{:= \widetilde{C}} \exp(-\frac{1}{2}((z_{t-1}-\mu)^\top H^{-1}(z_{t-1}-\mu))) dz_{t-1}$$
    $$= C \lvert H \rvert^{-\frac{1}{2}} \widetilde{C} \underbrace{\int (2\pi)^{-\frac{M}{2}} \lvert H \rvert^{-\frac{1}{2}} \exp(-\frac{1}{2}((z_{t-1}-\mu)^\top H^{-1}(z_{t-1}-\mu))) dz_{t-1}}_{=1}$$
    $$= C \lvert H \rvert^{-\frac{1}{2}} \widetilde{C}$$
    <p>Focus on the exponent of $\widetilde{C}$:</p>
    $$\text{exponent}(\tilde c) := -\frac{1}{2}\Bigl(-\mu^\top H^{-1} \mu + z_t^\top \Sigma^{-1} z_t + \mu_{t-1}^\top V_{t-1}^{-1}\mu_{t-1}\Bigr)$$
    $$= -\frac{1}{2}\Bigl[-\bigl(H(A^\top \Sigma^{-1}z_t + V_{t-1}^{-1}\mu_{t-1})\bigr)^\top H^{-1}H(A^\top \Sigma^{-1}z_t + V_{t-1}^{-1}\mu_{t-1})$$
    $$+ z_t^\top \Sigma^{-1} z_t + \mu_{t-1}^\top V_{t-1}^{-1}\mu_{t-1}\Bigr]$$
    $$= -\frac{1}{2}\Bigl(z_t^\top \Sigma^{-1} z_t + \mu_{t-1}^\top V_{t-1}^{-1}\mu_{t-1} - \bigl(z_t^\top \Sigma^{-1}A + \mu_{t-1}^\top V_{t-1}^{-1}\bigr)H$$
    $$\bigl(A^\top \Sigma^{-1}z_t + V_{t-1}^{-1}\mu_{t-1}\bigr) \Bigr)$$
    $$= -\frac{1}{2}\Bigl[\underbrace{z_t^\top\Bigl(\Sigma^{-1} - \Sigma^{-1} A H A^\top \Sigma^{-1}\Bigr)z_t}_{(\Sigma + A V_{t-1} A^\top)^{-1}}$$
    $$- z_t^\top \underbrace{\Sigma^{-1} A H V_{t-1}^{-1}}_{(\Sigma + A V_{t-1} A^\top)^{-1}A}\mu_{t-1}$$
    $$- \mu_{t-1}^\top \underbrace{V_{t-1}^{-1} H A^\top \Sigma^{-1}}_{A^\top(\Sigma + A V_{t-1} A^\top)^{-1}} z_t$$
    $$+ \mu_{t-1}^\top\underbrace{\Bigl(V_{t-1}^{-1} - V_{t-1}^{-1} H V_{t-1}^{-1}\Bigr)}_{A^\top(\Sigma + A V_{t-1} A^\top)^{-1}A}\mu_{t-1}\Bigr]$$
    $$= -\frac{1}{2}\Bigl[z_t^\top L_{t-1}^{-1} z_t - z_t^\top L_{t-1}^{-1}A\mu_{t-1}$$
    $$- \mu_{t-1}^\top A^\top L_{t-1}^{-1} z_t + \mu_{t-1}^\top A^\top L_{t-1}^{-1}A\mu_{t-1} \Bigr]$$
    $$= -\frac{1}{2}(z_t - A\mu_{t-1})^\top L_{t-1}^{-1}(z_t - A\mu_{t-1})$$
    $$\quad\Longrightarrow\quad \mathcal{N}(A\mu_{t-1}, L_{t-1})$$
    $$\Longrightarrow\quad I = \int p(z_t\mid z_{t-1})p(z_{t-1}\mid x_1,\dots,x_{t-1})dz_{t-1} = \mathcal{N}(A\mu_{t-1},\,L_{t-1})$$
    <p><strong>Finally:</strong></p>
    $$p(z_t\mid x_1,\dots,x_t) = \frac{p(x_t\mid z_t)\overbrace{\int p(z_t\mid z_{t-1})p(z_{t-1}\mid x_1,\dots,x_{t-1})dz_{t-1}}^{:=I}}{p(x_t\mid x_1,\dots,x_{t-1})}$$
    $$= \frac{\mathcal{N}(Bz_t,\Gamma)\mathcal{N}(A\mu_{t-1},L_{t-1})}{p(x_t\mid x_1,\dots,x_{t-1})}$$
    $$\Longrightarrow\quad p(z_t\mid x_1,\ldots,x_t) = \mathcal{N}(\mu_t,V_t).$$
    $$\Rightarrow\ \text{combining the remaining Gaussians is similar.}$$
    <p><strong>Used identities</strong></p>
    <p>Let</p>
    $$H := \bigl(A^\top \Sigma^{-1}A + V_{t-1}^{-1}\bigr)^{-1}$$
    <p>(i)</p>
    $$(\Sigma + A V_{t-1} A^\top)^{-1} = \Sigma^{-1} - \Sigma^{-1} A H A^\top \Sigma^{-1}. $$
    <p>(ii)</p>
    $$(\Sigma + A V_{t-1} A^\top)^{-1}A = \Sigma^{-1} A H V_{t-1}^{-1}$$
    <p>(iii)</p>
    $$A^\top(\Sigma + A V_{t-1} A^\top)^{-1} = V_{t-1}^{-1} H A^\top \Sigma^{-1}$$
    <p>(iv)</p>
    $$A^\top(\Sigma + A V_{t-1} A^\top)^{-1}A = V_{t-1}^{-1} - V_{t-1}^{-1} H V_{t-1}^{-1}$$
    <p>Also (as annotated):</p>
    $$(\Sigma + A V_{t-1} A^\top)^{-1} := L_t^{-1}.$$
  </details>
</div> -->

<div class="accordion">
  <details>
    <summary>proof (clean: induction + “Gaussian in information form”)</summary>

We prove by induction that the filtering distribution is Gaussian:

$$p(z_t\mid x_{1:t})=\mathcal N(z_t\mid m_t,V_t)$$


### Step 0 (base case)

By assumption the initial state is Gaussian, e.g.

$$p(z_0)=\mathcal N(z_0\mid m_0,V_0),$$

so the claim holds at $t=0$.

### Induction hypothesis

Assume for some $t-1\ge 0$ that

$$p(z_{t-1}\mid x_{1:t-1})=\mathcal N(z_{t-1}\mid m_{t-1},V_{t-1}).$$


### 1) Prediction is Gaussian (pushforward of a Gaussian)

The dynamics are linear-Gaussian:

$$z_t = A z_{t-1} + w_t,\qquad w_t\sim\mathcal N(0,\Sigma_z),\quad w_t\perp z_{t-1}.$$

Conditioned on $x_{1:t-1}$, $z_{t-1}$ is Gaussian, and $Az_{t-1}$ is therefore Gaussian. Adding independent Gaussian noise keeps it Gaussian. Hence

$$
p(z_t\mid x_{1:t-1})=\mathcal N(z_t\mid \hat m_t,\hat V_t),
\qquad
\hat m_t = A m_{t-1},\quad
\hat V_t = A V_{t-1}A^\top + \Sigma_z.
$$

### 2) Update keeps it Gaussian (product of Gaussians in canonical form)

The observation model is

$$
p(x_t\mid z_t)=\mathcal N(x_t\mid B z_t,\Gamma).
$$

For fixed $x_t$, this is a Gaussian *function of $z_t$*:

$$
p(x_t\mid z_t)\propto
\exp\!\left(-\tfrac12 (Bz_t-x_t)^\top\Gamma^{-1}(Bz_t-x_t)\right),
$$

which is quadratic in $z_t$.

Now apply Bayes’ rule:

$$
p(z_t\mid x_{1:t}) \propto p(x_t\mid z_t)\,p(z_t\mid x_{1:t-1}).
$$

Both factors are Gaussians in $z_t$, so their product is also Gaussian. The cleanest way to see this is to use **information form**.

Write the predicted prior as $\mathcal N(\hat m_t,\hat V_t)$ with

$$
\Lambda_t := \hat V_t^{-1},\qquad \eta_t := \Lambda_t \hat m_t.
$$

Up to normalization,

$$
p(z_t\mid x_{1:t-1}) \propto \exp\!\left(-\tfrac12 z_t^\top \Lambda_t z_t + \eta_t^\top z_t\right).
$$

Similarly, the likelihood contributes the quadratic/linear terms

$$
p(x_t\mid z_t)\propto \exp\!\left(-\tfrac12 z_t^\top (B^\top\Gamma^{-1}B) z_t + (B^\top\Gamma^{-1}x_t)^\top z_t\right).
$$

Multiplying the exponentials just **adds** these natural parameters, giving

$$
V_t^{-1} = \hat V_t^{-1} + B^\top\Gamma^{-1}B,
\qquad
V_t^{-1} m_t = \hat V_t^{-1}\hat m_t + B^\top\Gamma^{-1}x_t.
$$

Equivalently,

$$
\boxed{
V_t = \left(\hat V_t^{-1} + B^\top\Gamma^{-1}B\right)^{-1},
\qquad
m_t = V_t\left(\hat V_t^{-1}\hat m_t + B^\top\Gamma^{-1}x_t\right).
}
$$

Therefore $p(z_t\mid x_{1:t})$ is Gaussian, completing the induction.

> (These information-form updates are algebraically equivalent to the usual Kalman-gain form
> $K_t=\hat V_t B^\top(B\hat V_t B^\top+\Gamma)^{-1}$,
> $m_t=\hat m_t+K_t(x_t-B\hat m_t)$,
> $V_t=(I-K_tB)\hat V_t$.)

  </details>
</div>

<div class="accordion">
  <details>
    <summary>proof (clean: affine transformation of a Gaussian)</summary>

We assume the **filtered posterior at time $t-1$** is Gaussian:

$$z_{t-1}\mid x_{1:t-1}\sim \mathcal{N}(m_{t-1},V_{t-1})$$

The **linear Gaussian dynamics** are

$$z_t = A z_{t-1} + w_t,\qquad w_t\sim\mathcal{N}(0,\Sigma_z),\quad w_t \perp z_{t-1}.$$


Define $y := A z_{t-1}$. Since an affine map of a Gaussian is Gaussian,

$$y\mid x_{1:t-1}\sim \mathcal{N}(A m_{t-1}, A V_{t-1}A^\top).$$


Now $z_t = y + w_t$ is a sum of two **independent Gaussians**, hence also Gaussian. Its mean and covariance follow from linearity of expectation and independence:

$$
\mathbb{E}[z_t\mid x_{1:t-1}]
= \mathbb{E}[y\mid x_{1:t-1}] + \mathbb{E}[w_t]
= A m_{t-1},
$$

and

$$
\mathrm{Cov}(z_t\mid x_{1:t-1})
= \mathrm{Cov}(y\mid x_{1:t-1}) + \mathrm{Cov}(w_t)
= A V_{t-1}A^\top + \Sigma_z,
$$

because the cross-covariance terms vanish $y$ and $w_t$ are independent.

Therefore the **predictive (one-step-ahead) distribution** is Gaussian:

$$
p(z_t\mid x_{1:t-1})=\mathcal{N}\big(z_t \mid \hat m_t,\hat V_t\big),
\qquad
\hat m_t = A m_{t-1},\quad
\hat V_t = A V_{t-1}A^\top + \Sigma_z.
$$

  </details>
</div>

#### The Kalman Filter Recursion Equations

The full derivation of the update step (multiplying the Gaussian predictive distribution by the Gaussian likelihood $p(x_t\mid z_t)$) is algebraically intensive. The final, well-established recursion equations are presented here.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Kalman Filter Recursion)</span></p>

Let $m_{t-1}$ and $V_{t-1}$ be the mean and covariance of the state at time $t-1$.

1. **Prediction Step:**
  * **Predicted state mean:**
  
  $$\hat{m}_t = A m_{t-1}$$
  
  * **Predicted state covariance:** 
  
  $$\hat{V}_t = A V_{t-1} A^\top + \Sigma_z$$
  
2. **Update Step:**
  * **Kalman Gain $K_t$:** The gain determines how much the new observation $x_t$ influences the updated state estimate.
    
    $$K_t = \hat{V}_t B^\top (B \hat{V}_t B^\top + \Gamma)^{-1}$$

  * **Updated state mean $m_t$:** The new mean is the predicted mean plus a correction term based on the prediction error $(x_t - B \hat{m}_t)$.  
  
    $$m_t = \hat{m}_t + K_t (x_t - B \hat{m}_t)$$

  * **Updated state covariance $V_t$:** The new covariance is reduced from the predicted covariance. The equation from the source is presented as:  
  
    $$V_t = [I - K_t B] \hat{V}_t$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition behind Kalman Gain)</span></p>

Kalman gain $K_t$ is the knob that decides **how much you trust the new measurement vs. your model prediction**.

At time $t$ you have:

* a **prediction** (from dynamics): mean $\mu_{t\mid t-1}$ with uncertainty $P_{t\mid t-1}$
* a **measurement** $x_t$ with noise $R_t$ ($=\Gamma$)

The update is:

$$\mu_t=\mu_{t\mid t-1}+K_t\big(x_t - B_t\mu_{t\mid t-1}\big)$$

The term in parentheses is the **innovation / residual**: “what the sensor says minus what I expected to see.”

#### It’s a trust-weight

* If the **sensor is very noisy** ($R_t$ large), $K_t$ becomes small → *ignore the measurement more*.
* If your **prediction is uncertain** ($\hat{V}_t$ large), $K_t$ becomes large → *listen to the measurement more*.

So it automatically balances “model vs sensor” based on uncertainty.

#### Multi-D: It accounts for geometry and correlations

In general,

$$K_t = \hat{V}_t B_t^\top \big(B_t \hat{V}_t B_t^\top + R_t\big)^{-1}$$

* $B_t \hat{V}_t B_t^\top$ = predicted uncertainty **projected into measurement space**
* $+R_t$ = total expected uncertainty of the innovation
* Multiplying by $\hat{V}_t B_t^\top$ maps measurement corrections back into the state.

So $K_t$ also decides *which components of the state* a measurement should correct (e.g., GPS position should update position strongly and maybe velocity a bit if they are correlated).

#### Another good mental model

Kalman gain is chosen to make the updated estimate **as certain as possible** (minimize posterior covariance), while staying unbiased under the model assumptions. It’s the “best linear” way to fuse prediction and measurement when everything is Gaussian.

If you tell me a concrete example (e.g., GPS position updates a ($[pos, vel]$) state), I can show how $K$ gets a nonzero block that also nudges velocity because of the position–velocity coupling in $P$.

</div>

### The Kalman Smoother

While the Kalman filter provides the optimal estimate of the state $z_t$ given observations up to time $t$, denoted $p(z_t \mid x_1, \dots, x_t)$, we often desire a more accurate estimate that incorporates the entire dataset, including future observations. This is the goal of smoothing.

The smoothing problem is to find the distribution $p(z_t \mid x_1, \dots, x_T)$, where $T > t$.

The Kalman smoother is an efficient algorithm that accomplishes this with a backward pass through the data after a full forward pass of the Kalman filter has been completed.

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/kalman_smoother.png' | relative_url }}" alt="Kalman Smoother Schema" loading="lazy">
  <figcaption>Kalman Smoother.</figcaption>
</figure>

#### Derivation of the Predictive Distribution

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Equation</span><span class="math-callout__name">(Bayesian Smoother)</span></p>

$$\gamma_t = \alpha_t \cdot \beta_t = \alpha_t \cdot \frac{\int \alpha_{t+1}^{-1}\gamma_{t+1} p(x_{t+1} \mid z_{t+1})p(z_{t+1} \mid z_t)dz_{t+1}}{p(x_{t+1}\mid x_{1:t})}$$

where 

$$\gamma_t := p(z_t \mid x_{1:T})$$

$$\alpha_t := \frac{p(z_t, x_{1:t})}{p(x_{1:t})}$$

$$\beta_t := \frac{p(x_{t+1:T}\mid z_t)}{p(x_{t+1:T}\mid x_{1:t})}$$

Recursion comes from the dependence of $\gamma_t$ on $\gamma_{t+1}$ (backward in time).

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    $$\underbrace{p(z_t \mid x_{1:T})}_{\gamma_t} = \underbrace{\frac{p(z_t, x_{1:t})}{p(x_{1:t})}}_{=p(z_t\mid x_{1:t})\sim\mathcal{N}(\mu_t,V_t), := \alpha_t} \times \underbrace{\frac{p(x_{t+1:T}\mid z_t)}{p(x_{t+1:T}\mid x_{1:t})}}_{:= \beta_t}$$
    $$\gamma_t = \alpha_t \cdot \beta_t$$
    $$= \alpha_t \cdot \frac{\int p(z_{t+1}, x_{t+1:T}\mid z_t)dz_{t+1}}{p(x_{t+1:T}\mid x_{1:t})}$$
    $$= \alpha_t \cdot \frac{\int p(x_{t+2:T}\mid \cancel{x_{t+1}, z_t}, z_{t+1})\overbrace{p(x_{t+1} \mid z_{t+1})}^{\text{obs. model}}\overbrace{p(z_{t+1} \mid z_t)}^{\text{lat. model}}dz_{t+1}}{p(x_{t+2:T}\mid x_{1:t+1})p(x_{t+1}\mid x_{1:t})}$$
    <p>where</p>
    $$\beta_{t+1}=\alpha_{t+1}^{-1}\gamma_{t+1}={p(x_{t+2:T}\mid \cancel{x_{t+1}, z_t}, z_{t+1})}{p(x_{t+2:T}\mid x_{1:t+1})}$$
    $$\implies \gamma_t = \alpha_t \cdot \frac{\int \alpha_{t+1}^{-1}\gamma_{t+1} p(x_{t+1} \mid z_{t+1})p(z_{t+1} \mid z_t)dz_{t+1}}{p(x_{t+1}\mid x_{1:t})}$$
  </details>
</div>

#### Kalman Smoother Recursion Equations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Kalman Smoother Recursion)</span></p>

The algorithm requires the stored results (means and covariances) from the forward Kalman filter pass: $\lbrace m_t, V_t\rbrace_{t=1}^T$ and $\lbrace \hat{m}\_t, \hat{V}\_t\rbrace_{t=1}^T$.

1. **Initialization:** The recursion starts at the final time step $T$. The smoothed estimate at this point is simply the filtered estimate.
  * **Smoothed mean at time $T$:**
    * $m_T^s := m_T$
  
  * **Smoothed covariance at time $T$:** 
    * $V_T^s := V_T$

2. **Backward Recursion:** 
  * The algorithm proceeds backward in time, from $t = T-1$ down to $0$.

3. **For each step $t$:**
  * **Define Smoother Gain $J_t$:**  
  
    $$J_t = V_t A^\top (\hat{V}_{t+1})^{-1}$$  
  
    (Where $\hat{V}_{t+1} = A V_t A^\top + \Sigma_z$ is the one-step predictive covariance from time $t$ to $t+1$).
  
  * **Update Smoothed Mean $m_t^s$:**
    
    $$m_t^s = m_t + J_t (m_{t+1}^s - \hat{m}_{t+1})$$

  * **Update Smoothed Covariance $V_t^s$:**  
    
    $$V_t^s = V_t + J_t (V_{t+1}^s - \hat{V}_{t+1}) J_t^\top$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Kalman Smoother)</span></p>

After the backward pass is complete, the set of distributions $\lbrace \mathcal{N}(m_t^s, V_t^s) \rbrace_{t=1}^T$ represents the full smoothed posterior distributions $p(z_t \mid x_1, \dots, x_T)$ for all time steps:

$$p(z_t \mid x_{1:T}) \mathcal{N}(m_t^s, V_t^s)$$

$$\mathbb{E}[z_t] = m_t^s$$

$$\mathbb{E}[z_t z_t^\top] = \text{Cov}(z_t) + \mathbb{E}[z_t]\mathbb{E}[z_t^\top]$$

$$\mathbb{E}[z_t z_t^\top] = V_t^s J_{t-1}^\top + m_t^s (m_t^s)^\top$$

$$\implies \text{Cov}(z_t, z_{t-1}\mid X) = V_t^s J_{t-1}^\top$$

</div>

## The Poisson State Space Model

The Poisson State Space Model is designed to handle sequences of count data by linking them to an underlying, unobserved (latent) continuous state that evolves over time.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Poisson State Space Model)</span></p>

1. **Observation Model (Poisson):** The observation $c_t$ is a vector of counts at time $t$. Each count $c_t$ is drawn from a Poisson distribution whose rate, $\lambda_t$, is determined by the corresponding latent state $z_t$. $c_t \mid z_t \sim \text{Poisson}(\lambda_t)$. The probability mass function is given by: 
   
  $$p(c_t \mid z_t) = \frac{\lambda_t^{c_t} e^{-\lambda_t}}{c_t!} \quad\iff\quad c_t \mid z_t \sim \text{Poisson}(\lambda_t)$$
   
  * **Link Function:** A logarithmic link function connects the rate $\lambda_t$ to the latent state $z_t$. This ensures that the rate $\lambda_t$ is always non-negative. The function is applied elementwise if $c_t$ is a vector. $\log(\lambda_t) = b_0 + B_1 z_t$. This implies that the rate is an exponential function of a linear transformation of the state: $\lambda_t = \exp(b_0 + B_1 z_t)$, where $b_0$ is an offset vector and $B_1$ is a matrix of weights.
   
2. **Latent Model / Process Model (Linear Gaussian):** The latent state $z_t \in \mathbb{R}^M$ evolves as a linear function of the previous state $z_{t-1}$ plus Gaussian noise.
   
   $$z_t = A z_{t-1} + \omega_t, \quad \omega_t \sim \mathcal{N}(0, \Sigma)$$
   
   So, the conditional distribution is: 
   
   $$p(z_t \mid z_{t-1}) = \mathcal{N}(A z_{t-1}, \Sigma)$$

3. **Initial Distribution:** The initial state is drawn from a Gaussian distribution. 
   
   $$z_1 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

The model parameters are:

* **Observation parameters:** $\theta_{\text{obs}} = \lbrace b_0, B_1\rbrace$
* **Latent parameters:** $\theta_{\text{lat}} = \lbrace A, \Sigma, \mu_0, \Sigma_0\rbrace$

</div>

### Parameter Estimation using EM

The parameters of the model, $\theta = \lbrace b_0, B_1, A, \Sigma, \mu_0, \Sigma_0\rbrace$, are estimated using the Expectation-Maximization (EM) algorithm. This iterative approach is well-suited for models with latent variables. The EM algorithm alternates between two steps: an Expectation (E) step and a Maximization (M) step.

The core objective is to maximize the log-likelihood of the observed data, which involves marginalizing over the latent variables: 

$$\log p(C \mid \theta) = \log \int p(C, Z \mid \theta) dZ$$

The EM algorithm instead maximizes a lower bound on this quantity, defined as the expected complete-data log-likelihood.

The M-step objective is to maximize the following function with respect to $\theta$: 

$$\mathcal{Q}(\theta, \theta^{\text{old}}) = \mathbb{E}_{q(Z)}[\log p(Z, C \mid \theta)],$$ 

where $q(Z) = p(Z \mid C, \theta^{\text{old}})$

is the posterior distribution over the latent states, computed in the E-step using the parameters from the previous iteration.

This objective can be decomposed based on the model structure: 

$$\log p(Z, C \mid \theta) = \log p(C \mid Z, \theta) + \log p(Z \mid \theta)$$ 

Therefore, the expectation becomes: 

$$\mathbb{E}_{q(Z)}[\log p(Z, C)] = \underbrace{\mathbb{E}_{q(Z)}[\log p(C \mid Z)]}_{\prod_{t=1}^T p(c_t\mid z_t)} + \underbrace{\mathbb{E}_{q(Z)}[\log p(Z)]}_{p(z_1)\prod_{t=2}^T p(z_t\mid z_{t-1})}$$

### The M-Step: Maximization

In the M-step, we find the parameters $\theta$ that maximize the $\mathcal{Q}$ function, holding the posterior distribution $q(Z)$ fixed. We analyze the two terms of the objective separately.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">EM for Poisson SSM</span><span class="math-callout__name">(M-Step)</span></p>

Let $X=\lbrace X_{1:T}\rbrace$ and $Z=\lbrace Z_{1:T}\rbrace$.

**Goal:** $\theta^* = \max_\theta \mathbb{E}_q[\log p(X, Z)]$.

**Update for Latent Parameters $\lbrace A, \Sigma, \mu_0, \Sigma_0\rbrace$:**

Equivalent to the M-step of a standard linear Gaussian State Space Model:

$$A = \left(\sum_{t=2}^T \mathbb{E}_q[z_t z_{t-1}^\top]\right) \left(\sum_{t=2}^T \mathbb{E}_q[z_{t-1} z_{t-1}^\top]\right)^{-1}$$

The M-step for the other parameters ($B, \Sigma, \Gamma, \mu_0, \Sigma_0$) proceeds in a similar fashion. 

**Update for Observation Parameters $\lbrace b_0, B_1\rbrace$:**

The objective for $b_0$ and $B_1$ is: 

$$\mathcal{L}(b_0, B_1) = \sum_{t=1}^{T} \left( c_t^\top(b_0 + B_1 \mu_t) - \mathbf{1}^\top \mathbb{E}_{q(z_t)}[\lambda_t] \right)$$ 

where 

$$\mathbb{E}[\lambda_t] = \exp\left(b_0 + B_1\mu_t + \frac{1}{2}\mathrm{diag}(B_1 V_t B_1^\top)\right)$$

the outer $\exp(\cdot)$ is elementwise.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Latent Parameters are updated like in standard linear Gaussian SSM)</span></p>

* Maximizing the expectation of this term is **equivalent to the M-step of a standard linear Gaussian State Space Model**. 
* The updates for $\lbrace A, \Sigma, \mu_0, \Sigma_0 \rbrace$ depend on the expected sufficient statistics (moments) of the latent states, such as $\mathbb{E}\_q[z_t]$, $\mathbb{E}\_q[z_t z_t^\top]$, and $\mathbb{E}\_q[z_t z_{t-1}^\top]$, **which are computed during the E-step**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(No closed form solution for $b_0$ and $B_1$)</span></p>

* This expression **does not have a closed-form solution for $b_0$ and $B_1$**. 
* However, this **objective function is concave**, which guarantees that a unique maximum exists. 
* This maximum can be found efficiently using numerical optimization methods like the **Newton-Raphson algorithm**.

</div>

<div class="accordion">
  <details>
    <summary>proof for Latent Parameters</summary>
    <h4>Latent Dynamics Parameters $\lbrace A, \Sigma, \mu_0, \Sigma_0\rbrace$</h4>
    <p>The term $\mathbb{E}_{q(Z)}[\log p(Z)]$ depends only on the latent dynamics parameters.</p>
    $$\log p(Z) = \underbrace{\log p(z_1 \mid \mu_0, \Sigma_0)}_{(1)} + \sum_{t=2}^{T} \underbrace{\log p(z_t \mid z_{t-1}, A, \Sigma)}_{(2)}$$
    $$(1) = \log p(z_t \mid z_{t-1}) = \frac{M}{2}\log(2\pi) -\frac{1}{2}\log\mid \Sigma\mid - \frac{1}{2}(z_t - Az_{t-1})^\top \Sigma^{-1} (z_t - Az_{t-1})$$
    $$(2) = \log p(z_1) = \frac{M}{2}\log(2\pi) -\frac{1}{2}\log\mid \Sigma_0\mid - \frac{1}{2}(z_1 - \mu_0)^\top \Sigma_{0}^{-1} (z_1 - \mu_0)$$
  </details>
</div>


<div class="accordion">
  <details>
    <summary>proof for Observation Parameters</summary>
    <h4>Observation Parameters $\lbrace b_0, B_1\rbrace$</h4>
    <p>The term $\mathbb{E}_{q(Z)}[\log p(C \mid Z)]$ depends only on the observation parameters.</p>
    $$\mathbb{E}_{q(Z)}[\log p(C \mid Z)] = \sum_{t=1}^{T} \mathbb{E}_{q(z_t)}[\log p(c_t \mid z_t)]$$
    <p>The log-likelihood of the Poisson observation is:</p>
    $$\log p(c_t \mid z_t, b_0, B_1) = c_t^\top \log \lambda_t - \mathbf{1}^\top\lambda_t - \log(c_t!)$$
    <p>Dropping terms constant with respect to the parameters, we need to maximize:</p>
    $$\sum_{t=1}^{T} \mathbb{E}_{q(z_t)}[c_t^\top \log \lambda_t - \mathbf{1}^T\lambda_t] = \sum_{t=1}^{T} (c_t^\top \mathbb{E}_{q(z_t)}[\log \lambda_t] - \mathbf{1}^\top\mathbb{E}_{q(z_t)}[\lambda_t])$$
    <p>Substituting $\log \lambda_t = b_0 + B_1 z_t$, the first part is:</p>
    $$\mathbb{E}_{q(z_t)}[\log \lambda_t] = b_0 + B_1 \mathbb{E}_{q(z_t)}[z_t]$$
    <p>The second part involves the expectation of an exponential function of a Gaussian random variable. Assuming the approximate posterior from the E-step is Gaussian, $q(z_t) = \mathcal{N}(z_t \mid \mu_t, V_t)$, then</p>
    $$\mathbb{E}_{q(z_t)}[\lambda_t] = \mathbb{E}_{q(z_t)}[\exp(b_0 + B_1 z_t)]$$
    <p>Recall from probability theory that if a random vector $X \sim \mathcal{N}(\mu, \Sigma)$, then for a constant vector $k$, the expectation of $\exp(k^\top X)$ is given by:</p>
    $$\mathbb{E}[\exp(k^\top X)] = \exp(k^\top \mu + \frac{1}{2} k^\top \Sigma k)$$
    <p>Let</p>
    <ul>
      <li>$z \sim \mathcal{N}(\mu, V)$ with $z\in\mathbb{R}^M$</li>
      <li>$b_0\in\mathbb{R}^D$, $B_1\in\mathbb{R}^{D\times M}$</li>
      <li>$\lambda = \exp(b_0 + B_1 z)$ where the exp is <strong>elementwise</strong>.</li>
    </ul>
    <p>Look at the $i$-th component:</p>
    $$\lambda_i = \exp\big(b_{0,i} + (B_1 z)_i\big) = \exp\big(b_{0,i} + b_i^\top z\big)$$
    <p>where $b_i^\top$ is the $i$-th row of $B_1$ (so $b_i\in\mathbb{R}^M$).</p>
    <p>Now this matches the scalar identity with:</p>
    <ul>
      <li>scalar constant $a = b_{0,i}$</li>
      <li>constant vector $k = b_i$.</li>
    </ul>
    <p>So</p>
    $$\mathbb{E}[\lambda_i] = \mathbb{E}\big[\exp(a + k^\top z)\big] = \exp(a)\mathbb{E}\big[\exp(k^\top z)\big]$$
    <p>For Gaussian $z\sim\mathcal N(\mu_t,V_t)$,</p>
    $$\mathbb{E}[\exp(k^\top z)] = \exp\left(k^\top \mu_t + \frac{1}{2} k^\top V_t k\right)$$
    <p>Therefore</p>
    $$\boxed{\mathbb{E}[\lambda_{i}^{(t)}]= \exp\left(b_{0,i} + b_i^\top \mu_t + \frac{1}{2} b_i^\top V_t b_i\right)}$$
    <p>If you stack all components, you get a vector:</p>
    $$
    \mathbb {E}[\lambda_t] =
    \begin{bmatrix}
    \exp(b_{0,1} + b_1^\top \mu_t + \tfrac12 b_1^\top V_t b_1)\\
    \vdots\\
    \exp(b_{0,D} + b_D^\top \mu_t + \tfrac12 b_D^\top V_t b_D)
    \end{bmatrix}
    $$
    <p>More compact:</p>
    $$
    \boxed{\mathbb{E}[\lambda_t] = \exp\left(b_0 + B_1\mu_t + \frac{1}{2}\mathrm{diag}(B_1 V_t B_1^\top)\right)}
    $$
    <p>where:</p>
    <ul>
      <li>the outer $\exp(\cdot)$ is elementwise.</li>
    </ul>
    <p>The final objective for $b_0$ and $B_1$ is:</p>
    $$\mathcal{L}(b_0, B_1) = \sum_{t=1}^{T} \left( c_t^\top(b_0 + B_1 \mu_t) - \mathbf{1}^\top \mathbb{E}_{q(z_t)}[\lambda_t] \right)$$
  </details>
</div>

### The E-Step: Laplace Approximation

The E-step requires computing the posterior distribution over the latent states, $p(Z \mid C, \theta^{\text{old}})$. For the Poisson SSM, this posterior is analytically intractable due to the non-conjugate relationship between the Gaussian latent dynamics and the Poisson observation model. 

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Laplace Approximation in Statistics)</span></p>

**Laplace approximation** in statistics replaces complicated posteriors by Gaussian that matches its
* **Mode**
* **Curvature**

It is a second-order saddle-point approximation.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem and Solution</span><span class="math-callout__name">(Non-Gaussian Distribution $\implies$ Laplace Approximation)</span></p>

$$\overbrace{p(z_t \mid c_{1:t})}^{\approx \mathcal{N}(\mu_t, V_t)} = \frac{\overbrace{p(c_t \mid z_t)}^{\text{obs. model}}\int \overbrace{p(z_t\mid z_{t-1})}^{\mathcal{N}(Az_{t-1}, \Sigma)}\overbrace{p(z_{t-1}\mid c_{1:t-1})}^{\approx \mathcal{N}(\mu_{t-1}, V_{t-1})}dz_{t-1}}{p(c_t\mid c_{1:t-1})}$$ 

* $p(z_t \mid c_{1:t-1})$ is the Gaussian predictive distribution from the Kalman filter's predict step, but $p(c_t \mid z_t)$ is a **Poisson likelihood**. 
  * Their product **does not yield a standard distribution.**
* To overcome this, we **approximate the true posterior with a Gaussian distribution using the Laplace approximation**. 
  * This method **finds a Gaussian that matches the mode and the curvature** at the mode of the target distribution.
* **The process follows the standard predict-update cycle of a Kalman filter**, but with a modified update step.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(E-step for Poisson SSM)</span></p>

The E-step for the Poisson SSM consists of a forward filtering pass using the Laplace approximation at each update step.

* **Predict:** 
  * Compute the Gaussian predictive distribution:
  * $\mu_{t\mid t-1} = A \mu_{t-1}$
  * $V_{t\mid t-1} = A V_{t-1} A^\top + \Sigma$
* **Update:** 
  * Approximate the posterior $p(z_t \mid c_{1:t})$ with a Gaussian $\mathcal{N}(\mu_t, V_t)$ where:
  * $\mu_t = \arg\max_{z_t} Q(z_t)$
  * $V_t = (-\nabla_{z_t}^2 Q(z_t)\mid_{z_t=\mu_t})^{-1}$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Procedure is repeated)</span></p>

This **procedure is repeated for each time step** $t=1, \dots, T$ to obtain the approximate filtering distributions required for the M-step. To get the full posterior $q(Z)$, a subsequent backward pass (analogous to the Rauch-Tung-Striebel smoother) is typically performed.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($V_t$ for Gaussian (our case))</span></p>

Laplace Estimate for Gaussian:

$$V_t = \left[ -\nabla_{z_t}^2 Q(z_t) \right]^{-1}_{z_t=\mu_t}$$

$$ = \left[ -\nabla_{z_t}^2 \log \mathcal{N}(\mu, \Sigma) \right]^{-1}_{z_t=\mu_t} = -\Sigma^{-1}$$

**Recall:**

$$\frac{\partial f(z)}{\partial z \partial z^\top} = \frac{\partial}{\partial z \partial z^\top} -\frac{1}{2}(z-\mu)^\top\Sigma^{-1}(z-\mu) = -\Sigma^{-1}$$

</div>

<div class="accordion">
  <details>
    <summary>proof of Predict Step</summary>
    <h4>Predict Step</h4>
    <p>Assuming the filtered posterior at time $t-1$ is approximated by a Gaussian</p>
    $$p(z_{t-1} \mid c_{1:t-1}) \approx \mathcal{N}(\mu_{t-1}, V_{t-1})$$
    <p>the one-step-ahead predictive distribution is also Gaussian:</p>
    $$p(z_t \mid c_{1:t-1}) = \int p(z_t \mid z_{t-1}) p(z_{t-1} \mid c_{1:t-1}) dz_{t-1} \approx \mathcal{N}(z_t \mid \mu_{t\mid t-1}, V_{t\mid t-1}),$$
    <p>where:</p>
    <ul>
      <li>Predicted Mean: $\mu_{t\mid t-1} = A \mu_{t-1}$</li>
      <li>Predicted Covariance: $V_{t\mid t-1} = A V_{t-1} A^\top + \Sigma$</li>
    </ul>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>proof of Update Step</summary>
    <h4>Update Step</h4>
    <p>The goal is to approximate the true filtering posterior</p>
    $$p(z_t \mid c_{1:t}) = \frac{p(c_t \mid z_t) p(z_t \mid c_{1:t-1})}{p(c_t\mid c_{1:{t-1}})} \approx \frac{p(c_t \mid z_t) \mathcal{N}(z_t \mid \mu_{t\mid t-1}, V_{t\mid t-1})}{p(c_t\mid c_{1:{t-1}})}$$
    $$p(z_t \mid c_{1:t}) \propto p(c_t \mid z_t) p(z_t \mid c_{1:t-1})$$
    <p>with a new Gaussian $\mathcal{N}(\mu_t, V_t)$.</p>
    <ol>
      <li>A Gaussian distribution is fully specified by its mean and covariance matrix.</li>
      <li>The maximum of the logarithm of a Gaussian's PDF occurs at its mean. The curvature at this maximum is determined by the inverse of its covariance.</li>
    </ol>
    <p>We can therefore find the parameters of our Gaussian approximation by maximizing the logarithm of the target posterior. Let $Q(z_t)$ be the unnormalized log-posterior:</p>
    $$Q(z_t) = \log p(c_t \mid z_t) + \log p(z_t \mid c_{1:t-1})$$
    <p>Dropping constants, this is:</p>
    $$Q(z_t) = \left( c_t^\top (b_0 + B_1 z_t) - \mathbf{1}^\top\exp(b_0 + B_1 z_t) \right) - \frac{1}{2}(z_t - \mu_{t\mid t-1})^\top (V_{t\mid t-1})^{-1} (z_t - \mu_{t\mid t-1})$$
    <ul>
      <li><strong>Approximated Mean (Mode):</strong> The mean $\mu_t$ of the Laplace approximation is set to the mode of the true posterior, which is found by maximizing $Q(z_t)$:</li>
    </ul>
    $$\mu_t := \arg\max_{z_t} Q(z_t)$$
    <p>This maximization must be performed numerically.</p>
    <ul>
      <li><strong>Approximated Covariance (Curvature):</strong> The covariance $V_t$ is the negative inverse of the Hessian (second derivative matrix) of $Q(z_t)$ evaluated at the mode $\mu_t$:</li>
    </ul>
    $$V_t := \left[ -\nabla_{z_t}^2 Q(z_t) \right]^{-1}_{z_t=\mu_t}$$
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Laplace approximation has nothing to do with Laplace distribution)</span></p>

It’s called the **Laplace approximation** because the Gaussian you’re building is exactly the classical approximation that comes from **Laplace’s method** for approximating integrals.

**The core idea (Laplace’s method)**

Suppose you have an integral of the form

$$I=\int \exp(f(z))dz$$

and $f(z)$ has a sharp maximum $z^\star$ (the mode). Laplace’s method says: **expand $f$ to second order around the mode**,

$$f(z)\approx f(z^\star) + \frac{1}{2} (z-z^\star)^\top H (z-z^\star) \quad H=\nabla^2 f(z^\star)$$

(where $H$ is negative definite at a maximum), so

$$\exp(f(z))\approx \exp(f(z^\star))\exp\left(-\frac{1}{2} (z-z^\star)^\top(-H)(z-z^\star)\right)$$

That second factor is the kernel of a Gaussian. Therefore

$$\exp(f(z)) \approx \text{const}\cdot \mathcal{N}\left(z\mid z^\star, (-H)^{-1}\right)$$

and the integral becomes approximately

$$\int \exp(f(z))dz \approx \exp(f(z^\star)) (2\pi)^{d/2}\lvert {-H} \rvert^{-1/2}$$

That’s the “Laplace approximation”: approximate a peaked density (or the integrand of a marginal likelihood) by a Gaussian formed from the **mode + curvature**.

</div>

## Generative Recurrent Neural Networks

This chapter explores generative models for time series data that incorporate the non-linear dynamics characteristic of Recurrent Neural Networks (RNNs) within a state-space model framework. We will focus on the inference and learning challenges posed by these models and examine solutions based on the Expectation-Maximization (EM) algorithm, including the Extended Kalman Filter and Particle Filter for the E-Step.

### The EM Algorithm for State-Space RNNs

The Expectation-Maximization (EM) algorithm provides a framework for performing maximum likelihood estimation in models with latent variables. For generative RNNs, which are a form of non-linear state-space model, EM proceeds by iterating between two steps:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(EM for State-Space RNNs)</span></p>

* **E-Step (Expectation):** With the model parameters $\theta$ held fixed, compute the posterior distribution over the latent variables, $Z$, given the observed data, $X$. This step involves finding the distribution $q$ that maximizes the expected log-likelihood.
  
  $$q^{(t+1)} = \underset{q}{\text{argmax}} \mathbb{E}_q[\log p(X,Z\mid \theta^{(t)})]$$  
  
  This step is fundamentally about inference.
* **M-Step (Maximization):** With the posterior distribution $q^{(t+1)}$ fixed, update the model parameters $\theta$ to maximize the expected log-likelihood.  
  
  $$\theta^{(t+1)} = \underset{\theta}{\text{argmax}} \mathbb{E}_{q^{(t+1)}}[\log p(X,Z\mid \theta)]$$  
  
  This step is fundamentally about learning.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Comparison to Linear State-Space Models)</span></p>

In the case of a linear state-space model (SSM), the E-step is solved exactly and efficiently using established algorithms:

* The **Kalman filter** is used to compute the filtering distribution, $p(z_t \mid  x_1, \dots, x_t)$.
* The **Kalman smoother** is used to compute the smoothing distribution, $p(z_t \mid  x_1, \dots, x_T)$.

For non-linear models like generative RNNs, these exact solutions are no longer tractable, necessitating the approximation methods discussed in this chapter.

</div>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/generative_rnn.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>Generative RNN.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generative RNN)</span></p>

A **generative RNN** can be formulated as a non-linear state-space model with the following components:

* **Observation Model:** The observed data $x_t$ at time $t$ is generated from the corresponding latent state $z_t$, typically through a linear-Gaussian relationship.  
  
  $$x_t \mid  z_t \sim \mathcal{N}(B z_t, \Gamma)$$

* **Latent Model (Transition Model):** The latent state $z_t$ evolves over time based on the previous state $z_{t-1}$ through a non-linear transition function $F_\theta$, which represents the recurrent neural network dynamics.  
  
  $$z_t \mid  z_{t-1} \sim \mathcal{N}(F_\theta(z_{t-1}), \Sigma)$$  
  
  The function $F_\theta$ is the core of the RNN, often taking the form of a single neural network layer:  
  
  $$F_\theta(z_{t-1}) = \phi(W z_{t-1} + h)$$  
  
  where $\phi$ is a non-linear activation function (e.g., tanh, ReLU), and the parameters to be learned are $\theta = \lbrace W, h, B, \Gamma, \Sigma\rbrace$.
* **Initial State:** The initial latent state is typically assumed to follow a standard normal distribution.  
  
  $$z_0 \sim \mathcal{N}(0, I)$$

</div>

### The E-Step: Inference via the Extended Kalman Filter (EKF)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem and Solution</span><span class="math-callout__name">(Non-linear function breaks KF assumptions $\implies$ Linearize using Extended Kalman Filter)</span></p>

The primary challenge in the E-step is computing the one-step-ahead predictive distribution for the latent state:  

$$p(z_t \mid  x_1, \dots, x_t) = \frac{p(x_t\mid z_t)p(z_t \mid  x_1, \dots, x_{t-1})}{p(x_t\mid x_1, \dots, x_{t-1})}$$  

$$p(z_t \mid  x_1, \dots, x_{t-1}) = \int \overbrace{p(z_t \mid  z_{t-1}, \theta)}^{\mathcal{N}(F_\theta(z_{t-1}), \Sigma)} p(z_{t-1} \mid  x_1, \dots, x_{t-1}) dz_{t-1}$$  

Due to the **non-linear function** $F_\theta$ inside $p(z_t \mid  z_{t-1}, \theta)$, this **integral is intractable**. The **Extended Kalman Filter (EKF)** provides an **approximate solution**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Extended Kalman Filter)</span></p>

The core idea of the **Extended Kalman Filter** is to perform a **local linearization of the non-linear dynamics** at each time step.

1. It **assumes** that the posterior distribution from the previous step, $p(z_{t-1} \mid  x_1, \dots, x_{t-1})$, is **concentrated around its mean** $m_{t-1}$.
2. If this assumption holds, then it is likely that the true latent state $z_{t-1}$ is near $m_{t-1}$.
3. Therefore, we can linearize the non-linear function $F_\theta$ around this current mean estimate $m_{t-1}$, creating a **locally linear Gaussian model**.

This linearization is achieved using a first-order Taylor expansion of $F_\theta(z_{t-1})$ around $m_{t-1}$:  

$$F_\theta(z_{t-1})\biggr\rvert_{m_{t-1}} \approx F_\theta(m_{t-1}) + J_{t-1}(z_{t-1} - m_{t-1})$$  

where $J_{t-1}$ is the Jacobian matrix of $F_\theta$ evaluated at $m_{t-1}$:  

$$
J_{t-1} = 
\begin{pmatrix}
\frac{\partial F^{\theta}_1}{\partial z^{(t-1)}_1} & \dots & \frac{\partial F^{\theta}_1}{\partial z^{(t-1)}_M} \\
\vdots & \dots & \vdots \\
\frac{\partial F^{\theta}_M}{\partial z^{(t-1)}_1} & \dots & \frac{\partial F^{\theta}_M}{\partial z^{(t-1)}_M} \\
\end{pmatrix}
$$

By substituting this approximation back into the latent model, the **transition distribution** becomes:  

$$p(z_t \mid  z_{t-1}, \theta) \approx \mathcal{N}(F_\theta(m_{t-1}) + J_{t-1}(z_{t-1} - m_{t-1}), \Sigma)$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Extended Kalman Filter)</span></p>

* The resulting **expression for the mean is now linear** in $z_{t-1}$, which makes the predictive integral tractable, similar to the standard Kalman filter.
* The mean of the posterior $m_t$, is available at each successive time step, allowing for a new linearization point for the next step.
* This constitutes a **local update** at each time step, adapting the linearization to the most recent state estimate.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General case: more non-linearities)</span></p>

In more general case we use:

$$x_t\mid z_t \sim \mathcal{N}(G(z_t), \Gamma)$$

$$z_t\mid z_{t_t} \sim \mathcal{N}(F(z_{t-1}), \Sigma)$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Extended Kalman Filter)</span></p>

* Let $m_{t-1}$ and $V_{t-1}$ be the mean and covariance of $p(z_{t-1}\mid x_1, \dots, x_{t-1})$.
* Let $J_{t-1} := \frac{\partial F}{\partial z}\mid_{m_{t-1}}$, $\nabla_t := \frac{\partial G}{\partial z}\mid_{F(m_{t-1})}$

1. **Prediction Step:** The one-step-ahead predictive distribution $p(z_t\mid x_1, \dots, x_{t-1})$ is approximated as a Gaussian $\mathcal{N}(m_{t\mid t-1}, V_{t\mid t-1})$ with:
  * **Predicted state mean:**
  
  $$m_{t\mid t-1} = F_\theta(m_{t-1})$$
  
  * **Predicted state covariance:** 
  
  $$V_{t\mid t-1} = J_{t-1} V_{t-1} J_{t-1}^\top + \Sigma$$ 
  
2. **Update Step:** The filtering distribution $p(z_t\mid x_1, \dots, x_t)$ is approximated as a Gaussian $\mathcal{N}(m_t, V_t)$ with:
  * **Kalman Gain $K_t$:** The gain determines how much the new observation $x_t$ influences the updated state estimate.
    
    $$K_t = V_{t\mid t-1} \nabla_t^\top (\nabla_t V_{t\mid t-1} \nabla_t^\top + \Gamma)^{-1}$$

  * **Updated state mean $m_t$:** The new mean is the predicted mean plus a correction term based on the prediction error $(x_t - B m_{t\mid t-1})$.  
  
    $$m_t = m_{t\mid t-1} + K_t (x_t - G(m_{t\mid t-1}))$$

  * **Updated state covariance $V_t$:** The new covariance is reduced from the predicted covariance. The equation from the source is presented as:  
  
    $$V_t = V_{t\mid t-1} - K_t \nabla_t V_{t\mid t-1}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Comparison with Standard Kalman Filter)</span></p>

It is instructive to compare the EKF prediction equations with those of the standard Kalman Filter, where the transition is linear ($z_t = A z_{t-1} + w_t$):

* **KF Predicted Mean:** $m_{t\mid t-1} = A m_{t-1}$
* **KF Predicted Covariance:** $V_{t\mid t-1} = A V_{t-1} A^\top + \Sigma$

The EKF replaces the static transition matrix $A$ with the non-linear function $F_\theta$ for propagating the mean and with its local Jacobian $J_{t-1}$ for propagating the covariance.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Smoother is applicable in the same form as in KF)</span></p>

The **Kalman smoother algorithm can be applied in the same form as in the linear Kalman Filter case** to obtain the smoothed distributions $p(z_t\mid x_1, \dots, x_T)$, which are required for the M-step.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Challenges of the EKF)</span></p>

* The primary drawback of the EKF is that the **local linearizations can introduce errors**. 
* These **errors can accumulate over time**, potentially causing the filter to diverge from the true state distribution, especially for **highly non-linear systems**.

</div>

### The M-Step: Parameter Learning

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Computing expectations involving non-linearity)</span></p>

* In the M-step, we maximize the **expected log-joint likelihood** with respect to the parameters $\theta = \lbrace W, h, B, \Gamma, \Sigma\rbrace$. The objective function is:  

$$\mathbb{E}_q[\log p(X,Z\mid \theta)] = \mathbb{E}_q \left[ -\frac{1}{2}\sum_t \log\lvert \Gamma\rvert  - \frac{1}{2}\sum_t(x_t - Bz_t)^\top \Gamma^{-1} (x_t - Bz_t) \right]$$

$$+ \mathbb{E}_q \left[ -\frac{1}{2}\sum_t \log\lvert \Sigma\rvert  - \frac{1}{2}\sum_t(z_t - F_\theta(z_{t-1}))^\top \Sigma^{-1} (z_t - F_\theta(z_{t-1})) \right] + \text{obs. model term}$$  

* **To perform this maximization, we require posterior expectations of various terms.**
  * The **EKF/EKS provides the necessary smoothed expectations for linear terms**, such as $\mathbb{E}[z_t]$ and $\mathbb{E}[z_t z_t^\top]$.
* However, the non-linear term $F_\theta(z_{t-1})$ introduces a significant problem. We now need to compute **expectations involving this non-linearity**, such as:  

$$\mathbb{E}[F_\theta(z_t)]$$  

and other related terms. This requires solving integrals of the form 

$$\int p(z_t)F_\theta(z_t) dz_t$$ 

which are often **intractable for arbitrary non-linearities** $\phi$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposed Solution</span><span class="math-callout__name">(Simplifying the M-Step Integral)</span></p>

* The **difficulty of the M-step integrals is highly dependent on the choice of non-linearity** in the model. 
* One potential idea is to **select a non-linearity that simplifies these integrals**. 
  * For example, structuring the transition as:  

$$F_\theta(z_{t-1}) = W h(z_{t-1}) + h$$  

where $h(z) = \max(0, z)$ (the ReLU function), might lead to a **more tractable expectation** calculation for **certain distributions**.

</div>

### Alternative E-Step: The Particle Filter

An alternative to the deterministic approximation of the EKF is a sampling-based approach known as the **Particle Filter (PF)**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Particle Filter (PF))</span></p>

**Core Idea:** Weighted Particle Representation

We computed:

$$p(z_t\mid x_1, \dots x_T) = \frac{p(x_t\mid z_t)\overbrace{\int p(z_t\mid z_{t-1})p(z_{t-1}\mid x_1,\dots,x_{t-1})dz_{t-1}}^{p(z_t\mid x_1,\dots,x_{t-1})\text{ 1-step forward density}}}{p(x_t\mid x_1,\dots,x_{t-1})}$$

The particle filter approximates the filtering distribution $p(z_t\mid x_1, \dots, x_t)$ with a set of $K$ weighted samples, or particles.

The one-step forward density, $p(z_t \mid  x_1, \dots, x_{t-1})$, is represented by a set of $K$ unweighted particles:  

$$\lbrace z_t^{(1)}, z_t^{(2)}, \dots, z_t^{(K)} \rbrace \sim p(z_t \mid  x_1, \dots, x_{t-1})$$  

The posterior distribution is then represented by combining these particles with the observation model, $p(x_t\mid z_t)$, to define a set of importance weights:  

$$\lbrace w_t^{(1)}, w_t^{(2)}, \dots, w_t^{(K)} \rbrace$$  

where each weight is defined as:

$$w_t^{(k)} = \frac{p(x_t \mid  z_t^{(k)})}{\sum_{k=1}^K p(x_t\mid z^{(1)}_t)} \qquad (\implies w_t^{(k)} \propto p(x_t \mid  z_t^{(k)}))$$  

The weights are normalized such that $\sum_{k=1}^K w_t^{(k)} = 1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Particle Filter (PF))</span></p>

* **Upsides:** Particle filters provide consistent estimates. In the limit of an infinite number of particles, the sampled distribution converges to the true posterior distribution.
* **Downsides:** They incur high computational costs and can suffer from practical issues.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(The Particle Filter (Sequential Importance Resampling))</span></p>

The algorithm proceeds sequentially through time:

1. **Initialization ($t=1$):**
  * Draw $K$ initial particles $\lbrace z_1^{(k)}\rbrace_{k=1}^K$ from the prior distribution, $p(z_1\mid z_0)$. This set serves as the initial estimate of the one-step forward density.
  * Pass these samples through the observation model and normalize to obtain the initial weights: $w_1^{(k)} \propto p(x_1\mid z_1^{(k)})$.
2. **Iteration (for $t > 1$):** The process for each subsequent time step involves three stages: Resample, Propagate, and Weight.
  * (**a) Resample:** Generate a new set of particles $\lbrace z'^{(k)}\_{t-1}\rbrace_{k=1}^K$ by sampling with replacement from the previous particle set $\lbrace z\_{t-1}^{(k)}\rbrace_{k=1}^K$, where the probability of drawing particle $k$ is given by its weight $w_{t-1}^{(k)}$.
  * **(b) Propagate / Predict:** For each new particle $z'^{(k)}\_{t-1}$, pass it through the process model to obtain a new particle for the current time step:  
    
    $$\lbrace z_t^{(k)}\rbrace_{k=1}^K \sim p(z_t \mid  z'^{(k)}_{t-1})$$  
    
    This new set of unweighted particles represents the estimate of the one-step forward density, $p(z_t\mid x_1, \dots, x_{t-1})$.
  
  * **(c) Update Weights:** Pass the propagated particles through the observation model to obtain the new, unnormalized weights:  
  
    $$\tilde{w}_t^{(k)} = p(x_t \mid  z_t^{(k)})$$  
    
    Normalize the weights to obtain the final set $\lbrace w_t^{(k)}\rbrace_{k=1}^K$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Using Particles for Expectation)</span></p>

The weighted particle set provides a straightforward way to **approximate expectations needed for the M-step**, **replacing intractable integrals with finite sums**:  

$$\mathbb{E}[\varphi(z_t)] \approx \sum_{k=1}^K w_t^{(k)} \varphi(z_t^{(k)})$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Challenges of the Particle Filter)</span></p>

Despite its theoretical appeal, the particle filter faces significant practical challenges:

* **Computational Expense:** The need to propagate and weight many particles at each time step makes the method computationally intensive. The number of particles required for an accurate approximation often scales exponentially with the dimension of the latent state space.
* **Filter Collapse (Particle Degeneracy):** After several resampling steps, it is common for only a few particles to have non-negligible weights. The algorithm may repeatedly select these high-weight particles, causing the diversity of the particle set to collapse. This results in an impoverished representation of the posterior distribution, which may unnaturally shrink and misrepresent the true uncertainty.

</div>

## Generative Recurrent Models with Variational Inference

### Recap: Latent Variable Models

A latent variable model is defined by a joint probability distribution over observed data $X$ and unobserved (latent) variables $z$, denoted as $p_\theta(X, z)$, where $\theta$ represents the model parameters.

* **Generative Model Architecture:** The model specifies a generative process where latent variables $z$ are first sampled from a prior distribution, and then the observed data $X$ is generated from a conditional likelihood distribution.
  * **Prior Distribution:** The latent variables are assumed to follow a prior distribution, typically a standard normal distribution for simplicity:
  
  $$p(z) = \mathcal{N}(z \mid  0, I)$$

  * **Likelihood (Observation Model):** The observed data is generated conditioned on the latent variables, specified by the likelihood $p_\theta(X\mid z)$.
* **Training Objectives:** Common approaches for training such models include:
  * Expectation-Maximization (EM)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Challenges in the E-Step)</span></p>

The E-Step in the EM algorithm requires computing the posterior distribution of the latent variables, $p_\theta(z\mid X)$. This step is often the primary bottleneck.

* **Analytical Computation:** The posterior is frequently intractable for complex models (e.g., when neural networks are involved).
* **Approximation Methods:** When an exact solution is not feasible, approximation methods are necessary. These include:
  * **Sampling:** Using methods like Markov Chain Monte Carlo (MCMC) to draw samples from the posterior.
  * **Variational Inference (VI):** Approximating the true posterior with a simpler, tractable distribution from a chosen family of distributions.

</div>

### Variational Inference (VI)

Variational Inference reformulates the problem of posterior inference as an optimization problem. We introduce a family of distributions over the latent variables, $q_\phi(z\mid X)$, parameterized by $\phi$, and aim to find the member of this family that is "closest" to the true posterior $p_\theta(z\mid X)$.

#### The Evidence Lower Bound (ELBO)

The log-likelihood of the data, $\log p_\theta(X)$, which is often intractable to compute directly, can be lower-bounded. This bound is known as the Evidence Lower Bound (ELBO).

The log-likelihood of the observed data is bounded by:  

$$\log p_\theta(X) \ge \mathbb{E}_{q\phi(z\mid X)} \left[ \log \frac{p_\theta(X, z)}{q_\phi(z\mid X)} \right] := \text{ELBO}(\phi, \theta)$$

Maximizing the ELBO with respect to both the model parameters $\theta$ and the variational parameters $\phi$ serves as a tractable proxy for maximizing the true log-likelihood.

#### VI as an Optimization Problem

The ELBO can be rewritten to reveal its connection to the Kullback-Leibler (KL) divergence between the approximate posterior $q_\phi(z\mid X)$ and the true posterior $p_\theta(z\mid X)$.

$$\text{ELBO}(\phi, \theta) = \log p_\theta(X) - \text{KL}(q_\phi(z\mid X) \parallel p_\theta(z\mid X))$$

Since the KL divergence is always non-negative, this confirms that the ELBO is a lower bound on the log-evidence. From this formulation, it is clear that maximizing the ELBO is equivalent to minimizing the KL divergence between the approximate and true posteriors. VI searches for an optimal density $q^{\ast}$ out of a chosen family of distributions $\mathcal{Q}$.

The optimal approximate posterior $q^{\ast}$ is found by solving:  

$$q^{\ast} = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(z\mid X) \parallel p_\theta(z\mid X))$$

#### An Alternative Formulation of the ELBO

For practical implementation and a more intuitive understanding, the ELBO is often rewritten in a different form.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(ELBO = Reconstruction - Regularization)</span></p>

$$\text{ELBO}(\phi, \theta) = \underbrace{\mathbb{E}_{q\phi(z\mid X)}[\log p_\theta(X\mid z)]}_{\text{Reconstruction}} - \underbrace{\text{KL}(q_\phi(z\mid X) \parallel p_\theta(z))}_{\text{Regularization}}$$

This form consists of two terms:

1. **Reconstruction Term:**
  * Encourages the model to learn latent variables $z$ from which the original data $X$ can be accurately reconstructed.
2. **Regularization Term:** 
  * The negative KL divergence acts as a regularizer, pushing the approximate posterior $q_\phi(z\mid X)$ to be close to the prior distribution $p_\theta(z)$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Starting from the definition, we can expand the joint probability $p_\theta(X, z) = p_\theta(X\mid z) p_\theta(z)$:</p>
    $$\text{ELBO} = \mathbb{E}_{q\phi(z\mid X)} \left[ \log \frac{p_\theta(X\mid z)p_\theta(z)}{q_\phi(z\mid X)} \right]$$
    <p>Using the properties of logarithms and the linearity of expectation:</p>
    $$= \mathbb{E}_{q\phi(z\mid X)}[\log p_\theta(X\mid z) + \log p_\theta(z) - \log q_\phi(z\mid X)]$$
    $$= \mathbb{E}_{q\phi(z\mid X)}[\log p_\theta(X\mid z)] + \mathbb{E}_{q\phi(z\mid X)}[\log p_\theta(z)] - \mathbb{E}_{q\phi(z\mid X)}[\log q_\phi(z\mid X)]$$
    <p>Rearranging the terms, we can identify the KL divergence between the approximate posterior and the prior:</p>
    $$= \mathbb{E}_{q\phi(z\mid X)}[\log p_\theta(X\mid z)] - \left( \mathbb{E}_{q\phi(z\mid X)}[\log q_\phi(z\mid X)] - \mathbb{E}_{q\phi(z\mid X)}[\log p_\theta(z)] \right)$$
    <p>This gives the final, commonly used form:</p>
    $$\text{ELBO}(\phi, \theta) = \underbrace{\mathbb{E}_{q\phi(z\mid X)}[\log p_\theta(X\mid z)]}_{\text{Reconstruction}} - \underbrace{\text{KL}(q_\phi(z\mid X) \parallel p_\theta(z))}_{\text{Regularization}}$$
  </details>
</div>

With a parameterized family of densities $q_\phi(z\mid X)$, VI becomes a joint optimization problem.

We seek to find the optimal parameters for both the generative and inference models:  

$$\phi^{\ast}, \theta^{\ast} = \arg\max_{\phi, \theta} \text{ELBO}(\phi, \theta)$$

For a detailed review of Variational Inference, see Blei et al. (2017), "Variational Inference: A Review for Statisticians."

### Sequential Variational Autoencoders (SVAE)

Introduced by Kingma & Welling (2013) and Rezende et al. (2014), the Variational Autoencoder framework can be extended to handle sequential data, leading to models often referred to as Sequential Variational Autoencoders (SVAE) or Variational Recurrent Neural Networks (VRNN).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SVAE Architecture)</span></p>

An SVAE consists of two primary components: a generative model that defines a distribution over sequences and an inference model that approximates the posterior over the sequence of latent variables.

* **Generative Model $p_\theta(X, Z)$:**
  * **Prior Model $p_\theta(Z)$:** Defines the dynamics of the latent variables over time, typically in an autoregressive manner, e.g., $p_\theta(z_t \mid  z_{t-1})$. This captures the temporal structure in the latent space.
  * **Observation Model $p_{\theta, \text{obs}}(X\mid Z)$:** The "decoder" that generates the observed data $x_t$ at each time step, conditioned on the corresponding latent state $z_t$.
* **Inference Model (Encoder) $q_\phi(Z\mid X)$:**
  * The "encoder" approximates the true posterior $p_\theta(Z\mid X)$. It maps an observed sequence $X$ to a distribution over the latent sequence $Z$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/SVAE.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>SVAE Architecture.</figcaption>
</figure>

#### Architectural Possibilities

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Architectural Possibilities)</span></p>

The core idea is to use simple, tractable distributional forms (e.g., Gaussian) for the model components, but to parameterize their means and covariances in complex, non-linear ways using neural networks (like RNNs, MLPs, or CNNs). This provides a rich blend of probabilistic modeling and deep learning.

* **Latent Model (Prior):**  
  
  $$p(z_t \mid  z_{t-1}) = \mathcal{N}(\mu_t, \Sigma_t)$$  
  
  where the mean $\mu_t = f_\mu(z_{t-1})$ and covariance $\Sigma_t = f_\Sigma(z_{t-1})$ are functions of the previous latent state, often implemented as an RNN.
* **Observation Model (Decoder):**  
  
  $$p(x_t \mid  z_t) = \mathcal{N}(\mu_x, \Sigma_x)$$  
  
  where the mean $\mu_x = g_\mu(z_t)$ and covariance $\Sigma_x = g_\Sigma(z_t)$ are functions of the current latent state. For $\Sigma_x$ to be a valid covariance matrix, it must be positive semi-definite. Common choices include:
  * **Diagonal covariance:** $\Sigma_x = \text{diag}(f(z_t))$
  * **Low-rank factorization:** $\Sigma_x = V V^\top$, where $V$ is the output of a neural network.
* **Inference Model (Encoder):**  
 
$$q_\phi(z_t \mid  x_{\le t}, \dots) = \mathcal{N}(\mu_\phi, \Sigma_\phi) \implies g_\phi(X\mid Z) = \prod_{t=1}^T g_\phi(z_t\mid x_t)$$  

where the variational parameters $\mu_\phi$ and $\Sigma_\phi$ are complex functions of the input data, typically implemented by a recurrent neural network that processes the sequence $X$.

</div>

#### The Sequential ELBO

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sequential ELBO)</span></p>

Starting from the practical form of the ELBO and applying factorization assumptions for sequential data, the objective can be decomposed into a sum over time steps.

$$\text{ELBO}(\phi, \theta, X) = \sum_t \left( \mathbb{E}_{q_\phi(z_t\mid X)}[\log p_\theta(x_t\mid z_t)] - \mathbb{E}_{q_\phi(Z_{<t}\mid X)}[\text{KL}(q_\phi(z_t\mid X) \ \parallel \ p_\theta(z_t\mid z_{t-1}))] \right)$$

This objective can be conceptualized as a sum of per-timestep reconstruction losses and KL-divergence penalties:  

$$\text{ELBO} = \sum_t \left( \mathcal{L}_{\text{rec}}(t) + \mathcal{L}_{\text{KL}}(t) \right)$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Starting from the last definition of ELBO:</p>
    $$\text{ELBO}(\phi, \theta) = \mathbb{E}_{q\phi(Z\mid X)}[\log p_\theta(X\mid Z)] -\text{KL}(q_\phi(Z\mid X) \parallel p_\theta(Z))$$
    $$=\text{ELBO}(\phi, \theta) = \mathbb{E}_{q\phi(Z\mid X)}[\log \prod_t^T p_\theta(x_z\mid z_t)] - \mathbb{E}_{q\phi(Z\mid X)}[\log q_\phi(Z\mid X) - \log p_\theta(Z)]$$
    $$=\text{ELBO}(\phi, \theta) = \sum_t^T \mathbb{E}_{q\phi(z_t\mid x_t)}[\log p_\theta(x_z\mid z_t)] - \sum_t^T \mathbb{E}_{q\phi(z_t\mid x_t)}[\log q_\phi(z_t\mid x_t) - \log p_\theta(z_t\mid z_{t-1})]$$
    $$=\text{ELBO}(\phi, \theta) = \sum_t \left( \mathbb{E}_{q_\phi(z_t\mid X)}[\log p_\theta(x_t\mid z_t)] - \mathbb{E}_{q_\phi(Z_{<t}\mid X)}[\text{KL}(q_\phi(z_t\mid X) \ \parallel \ p_\theta(z_t\mid z_{t-1}))] \right)$$
  </details>
</div>

### Training: Stochastic Gradient Variational Bayes (SGVB)

The ELBO contains expectations over $q_\phi(z\mid X)$ that are generally intractable to compute analytically. SGVB provides a method for creating a stochastic, differentiable estimator of the ELBO, enabling training with gradient-based optimizers.

#### The Challenge and Two Tricks

**Problem:** The expectations in the ELBO are difficult to compute.

**Solution:** SGVB employs two key techniques:

1. **Monte Carlo (MC) Estimates:** Approximate the expectation with samples from $q_\phi$.
2. **The Reparameterization Trick:** Restructure the sampling process to allow gradients to flow through to the parameters $\phi$.

#### Monte Carlo Estimation

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Monte Carlo ELBO Estimation)</span></p>

The ELBO can be expressed as an expectation:  

$$\text{ELBO}(\phi, \theta) = \mathbb{E}_{z \sim q\phi(z\mid X)} [\log p_\theta(X, z) - \log q_\phi(z\mid X)]$$  

We can approximate this expectation using $L$ samples drawn from the variational posterior $q_\phi(z\mid X)$:  

$$\text{ELBO}(\phi, \theta) \approx \frac{1}{L} \sum_{l=1}^L \left[ \log p_\theta(X, z^{(l)}) - \log q_\phi(z^{(l)}\mid X) \right] \quad \text{where } z^{(l)} \sim q_\phi(z\mid X)$$  

In practice, a single sample ($L=1$) is often used for each gradient step.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Naive Monte Carlo ELBO Estimation Problem)</span></p>

A critical problem with this naive MC estimation is that the sampling operation $z^{(l)} \sim q_\phi(z\mid X)$ makes the objective non-differentiable with respect to the variational parameters $\phi$. This prevents the use of standard backpropagation.

</div>

#### The Reparameterization Trick

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(The Reparameterization Trick)</span></p>

The reparameterization trick, introduced by Rezende et al. (2014), resolves this differentiability issue.

**Idea:** Replace samples from the variational density $q_\phi$ with a deterministic function $g$ of its parameters $\phi$ and an auxiliary, unparameterized random variable $\epsilon$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Reparameterization Trick: Univariate Gaussian)</span></p>

If $z \sim \mathcal{N}(\mu, \sigma^2)$, we can reparameterize the sampling process as:  

$$z = \mu + \sigma \cdot \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, 1)$$  

Here, $z = g(\mu, \sigma, \epsilon)$ is a deterministic function of the parameters $\mu$ and $\sigma$, and the stochasticity is isolated in $\epsilon$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Reparameterization Trick: Multivariate Gaussian)</span></p>

If $z \sim \mathcal{N}(\mu, \Sigma)$, we can reparameterize as:  

$$z = \mu + R \cdot \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I) \text{ and } \Sigma = R R^T$$  

$R$ can be obtained via Cholesky decomposition of the covariance matrix $\Sigma$.

This trick works for many distributions. By reparameterizing, we can move the expectation with respect to the parameters outside the gradient operator, making the objective differentiable with respect to $\phi$:  

$$
\nabla_\phi \mathbb{E}_{z \sim q\phi(z\mid X)}[f(z)] = \nabla_\phi \mathbb{E}_{\epsilon \sim p(\epsilon)}[f(g(\phi, \epsilon))] = \mathbb{E}_{\epsilon \sim p(\epsilon)}[\nabla_\phi f(g(\phi, \epsilon))]
$$

</div>

#### Applying SGVB to SVAEs

For our SVAE model, the ELBO consists of the reconstruction term and the KL term:  

$$\mathcal{L}(\theta, \phi, X) = \mathcal{L}_{\text{rec}}(\theta, \phi, X) + \mathcal{L}_{\text{KL}}(\phi, X)$$

Wherever we have analytic solutions to parts of the ELBO, we should use them instead of sampling. This reduces the variance of the gradient estimates and leads to more stable training.

* **The Reconstruction Term $\mathcal{L}\_{\text{rec}}$:** This term, $\mathbb{E}\_{q_\phi(Z\mid X)}[\sum_t \log p_\theta(x_t\mid z_t)]$, does not have a closed-form solution and requires sampling $Z$ using the reparameterization trick.
* **The KL-Divergence Term $\mathcal{L}\_{\text{KL}}$:** For the common case where both the inference model $q\_\phi(z_t\mid X) = \mathcal{N}(\mu_q, \Sigma_q)$ and the prior model $p\_\theta(z_t\mid z_{t-1}) = \mathcal{N}(\mu_p, \Sigma_p)$ are Gaussian, the KL divergence has a closed-form analytical solution.

By using the analytical form for the KL term, we only need to use MC estimation for the reconstruction term.

#### Parameter Update Rules

With a differentiable estimator for the ELBO, we can perform gradient ascent to jointly optimize both the generative parameters $\theta$ and the variational parameters $\phi$:  

$$\theta \leftarrow \theta + \eta \nabla_\theta \mathcal{L}(\theta, \phi, X)$$

$$\phi \leftarrow \phi + \eta \nabla_\phi \mathcal{L}(\theta, \phi, X)$$  

where $\eta$ is the learning rate.

### Final Remarks: Advanced Variational Densities

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advanced Variational Densities)</span></p>

The choice of a simple unimodal Gaussian for the variational density $q_\phi$ may be too restrictive to accurately model a complex, multi-modal true posterior. More expressive families of distributions can be used to achieve a better approximation.

1. **Gaussian Mixture Models (GMMs):** The variational posterior can be modeled as a mixture of $M$ Gaussian components, allowing it to capture multi-modal distributions.
  
  $$q_\phi(z\mid X) = \sum_{m=1}^M \pi_m(X) \mathcal{N}(z \mid  \mu_m(X), \Sigma_m(X))$$

2. **Normalizing Flows (Rezende & Mohamed, 2015):** Normalizing flows provide a general method for constructing complex distributions from a simple base distribution through a sequence of invertible transformations.
  * Idea: Start with a simple random variable $z_0$ with a known density $p_0(z_0)$ (e.g., $z_0 \sim \mathcal{N}(0, I)$).
  * Apply a sequence of smooth, invertible transformations $f_1, f_2, \dots, f_K$:
  
  $$z_K = f_K \circ f_{K-1} \circ \dots \circ f_1(z_0)$$

  * The density of the resulting variable $z_K$ can be computed exactly using the change of variables formula. This allows for the construction of highly flexible and expressive variational posteriors capable of modeling arbitrarily complex distributions.

</div>
