## Statistical Inference

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