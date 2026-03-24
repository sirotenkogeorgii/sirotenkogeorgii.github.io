---
title: Problems from the MBTSA course
layout: default
noindex: true
---

# Selected MBTSA Course Problems

**Table of Contents**
- TOC
{:toc}

## Exercise 1 (Strong law of large numbers)

### Task 1

* Simulate $n = 10000$ random samples $\lbrace x_t\rbrace_{t=1}^n$ from $N(0,1)$
* Construct cumulative sums
  
  $$y_i = \sum_{t=1}^i x_t \quad \text{for each } i \in {1,\dots,n}$$
  
* Compute the mean
  
  $$\bar{y}_k = \frac{1}{k}\sum_{i=1}^k y_i \quad \text{for each } k \in \lbrace 1,\dots,n\rbrace$$
  
* Plot:
  * $x$-axis: number of samples
  * $y$-axis: mean of the cumulative sums
* Explain whether the mean converges to the expected value, and why

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution: Expected value coincidences, but the variance blows up</summary>

Let

$$x_1,\dots,x_n \overset{\text{i.i.d.}}{\sim} \mathcal N(0,1), \qquad n=10000,$$

and define

$$y_i=\sum_{t=1}^i x_t, \qquad i=1,\dots,n.$$

So $(y_i)$ is a **Gaussian random walk**.

#### 1. Expected value

Since each $x_t\sim\mathcal N(0,1)$ has mean $0$,

$$
\mathbb E[y_i]
=
\mathbb E\left[\sum_{t=1}^i x_t\right]
=
\sum_{t=1}^i \mathbb E[x_t]
=
0.
$$

Therefore,

$$\mathbb E[\bar y_k] = \mathbb E\left[\frac1k\sum_{i=1}^k y_i\right] = \frac1k\sum_{i=1}^k \mathbb E[y_i] = 0$$

So the expected value is always $0$.

#### 2. Does $\bar y_k$ converge to $0$?

Rewrite $\bar y_k$ by exchanging the sums:

$$\bar y_k = \frac1k\sum_{i=1}^k y_i = \frac1k\sum_{i=1}^k \sum_{t=1}^i x_t = \frac1k\sum_{t=1}^k (k-t+1)x_t$$

So $\bar y_k$ is a linear combination of independent Gaussian variables, hence it is Gaussian itself.

Its variance is

$$\operatorname{Var}(\bar y_k) = \frac{1}{k^2}\sum_{t=1}^k (k-t+1)^2 \operatorname{Var}(x_t) = \frac{1}{k^2}\sum_{j=1}^k j^2 = \frac{k(k+1)(2k+1)}{6k^2}$$

Thus

$$\operatorname{Var}(\bar y_k) = \frac{(k+1)(2k+1)}{6k} \sim \frac{k}{3}\qquad (k\to\infty)$$

This variance **grows linearly** instead of going to $0$.

So

$$\bar y_k \sim \mathcal N\left(0,\frac{(k+1)(2k+1)}{6k}\right),$$

and the spread becomes larger and larger as $k$ increases.

For any fixed $\varepsilon>0$,

$$\mathbb P(\lvert \bar y_k\rvert >\varepsilon)\to 1.$$

That means $\bar y_k$ does **not** converge to $0$ in probability, hence it also does not converge almost surely to $0$.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Why the blowing variance prevents the mean of Gaussian Random Walk from convergence?</summary>

**In this problem** it follows from the Gaussian form, not from variance growth alone.

Using the standard normal CDF $\Phi$,

$$
\mathbb P(\lvert \bar y_k\rvert >\varepsilon) = # \mathbb P\left(\lvert Z\rvert>\frac{\varepsilon}{\sigma_k}\right) = 2\left(1-\Phi\left(\frac{\varepsilon}{\sigma_k}\right)\right)
$$

As $k\to\infty$,

$$\frac{\varepsilon}{\sigma_k}\to 0,$$

hence

$$2\left(1-\Phi\left(\frac{\varepsilon}{\sigma_k}\right)\right) \to 2(1-\Phi(0)) = 2\left(1-\frac12\right)=1$$

#### Why this works conceptually

A centered normal $N(0,\sigma_k^2)$ is just a rescaled version of $N(0,1)$.
As $\sigma_k$ becomes huge, the interval $[-\varepsilon,\varepsilon]$ becomes tiny relative to its scale. So the probability of landing inside that fixed interval goes to $0$, which means the probability of being outside goes to $1$.

#### Important subtlety

Be careful:

$$\operatorname{Var}(X_k)\to\infty$$

**does not by itself imply**

$$\mathbb P(\lvert X_k\rvert >\varepsilon)\to 1$$

for arbitrary random variables.

It works here because $\bar y_k$ is specifically **Gaussian with mean $0$**.

If you want, I can also show why this implies that $\bar y_k$ cannot converge to $0$ in probability.


</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Note Aside: For the mean of the cumsum to converge to the mean, the variance of the convergence must decrease as well?</summary>

Almost, but not quite.

For this kind of question, there are two important facts:

1. **A very useful sufficient condition** is
   
   $$
   \mathbb E[X_k]\to \mu
   \quad\text{and}\quad
   \operatorname{Var}(X_k)\to 0
   $$

   Then $X_k\to \mu$ in probability, by Chebyshev:
   
   $$
   \mathbb P(\lvert X_k-\mu\rvert>\varepsilon)
   \le
   \frac{\operatorname{Var}(X_k)}{\varepsilon^2}.
   $$

So if the variance goes to $0$, convergence to the mean is guaranteed.

2. **But variance going to $0$ is not necessary in general.**
   A sequence can still converge in probability to a constant even if its variance does not go to $0$.

Example:

$$
X_n =
\begin{cases}
\sqrt n, & \text{with probability } \frac1n,\\
0, & \text{with probability } 1-\frac1n.
\end{cases}
$$

Then for every fixed $\varepsilon>0$,

$$\mathbb P(\lvert X_n\rvert >\varepsilon)=\frac1n\to 0,$$

so $X_n\to 0$ in probability.

But

$$\mathbb E[X_n^2]=n\cdot \frac1n = 1$$

so the variance does **not** go to $0$.

#### In this exercise

For

$$\bar y_k=\frac1k\sum_{i=1}^k y_i,$$

we found that $\bar y_k$ is Gaussian with mean $0$ and

$$\operatorname{Var}(\bar y_k)\sim \frac{k}{3}\to\infty.$$

Here that is a decisive sign that $\bar y_k$ does **not** converge to $0$, because a Gaussian random variable with larger and larger variance spreads out more and more instead of concentrating near its mean.

So the right takeaway is:

* **General rule:** variance $\to 0$ is a strong sufficient condition for convergence to the mean, but not a necessary one.
* **In this problem:** since the variance actually grows, the averages do not concentrate near $0$, so $\bar y_k$ does not converge to the expected value.

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/running_mean_iid_normal.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/running_mean_random_walk.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

## Exercise 2: Maximum likelihood estimation

* For each distribution:
  * State the likelihood function
  * State the log-likelihood function
  * Maximize the log-likelihood by finding roots of the derivative
  * Check whether the estimator is unbiased

### Task 1

* Distribution:
  * Poisson distribution $\mathrm{Poi}(\lambda)$, $\lambda > 0$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Maximum Likelihood Estimation</summary>

The Poisson PMF is:
$$P(X = x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x \in \{0, 1, 2, \ldots\}$$

---

**Step 1 — Likelihood function:**

$$L(\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} = \frac{\lambda^{\sum x_i} \, e^{-n\lambda}}{\prod x_i!}$$

**Step 2 — Log-likelihood:**

$$\ell(\lambda) = \ln L(\lambda) = \left(\sum_{i=1}^n x_i\right) \ln \lambda - n\lambda - \sum_{i=1}^n \ln(x_i!)$$

**Step 3 — Maximize (find root of derivative):**

$$\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n \stackrel{!}{=} 0$$

$$\Rightarrow \boxed{\hat{\lambda}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}}$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Check whether the estimator is unbiased</summary>

**Step 4 — Bias check:**

$$\mathbb{E}[\hat{\lambda}] = \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i] = \frac{1}{n} \cdot n\lambda = \lambda \checkmark$$

The MLE is **unbiased**.

</details>
</div>

* Distribution:
  * Exponential distribution $\mathrm{Exp}(\lambda)$, $\lambda > 0$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Maximum Likelihood Estimation</summary>

The Exponential PDF is:
$$f(x; \lambda) = \lambda \, e^{-\lambda x}, \quad x \geq 0$$

---

**Step 1 — Likelihood function:**

$$L(\lambda) = \prod_{i=1}^n \lambda \, e^{-\lambda x_i} = \lambda^n \, e^{-\lambda \sum x_i}$$

**Step 2 — Log-likelihood:**

$$\ell(\lambda) = n \ln \lambda - \lambda \sum_{i=1}^n x_i$$

**Step 3 — Maximize:**

$$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i \stackrel{!}{=} 0$$

$$\Rightarrow \boxed{\hat{\lambda}_{\text{MLE}} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}}$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Check whether the estimator is unbiased</summary>

Let

$$S=\sum_{i=1}^n X_i$$

If $X_i\sim \mathrm{Exp}(\lambda)$ with rate $\lambda$, then

$$S\sim \mathrm{Gamma}(n,\lambda)$$

(shape $n$, rate $\lambda$). A standard fact is

$$\mathbb E\left[\frac1S\right]=\frac{\lambda}{n-1}, \qquad n>1$$

Hence

$$
\mathbb E[\hat\lambda]
=\mathbb E\left[\frac{n}{S}\right]
= n,\mathbb E\left[\frac1S\right]
= n\cdot \frac{\lambda}{n-1}
= \frac{n}{n-1}\lambda
$$

Thus

$$\mathbb E[\hat\lambda]\ne \lambda$$

So the MLE is biased upward.

$$\boxed{\hat\lambda=\frac1{\bar X} \text{ is biased, with bias } \frac{\lambda}{n-1}\ (n>1).}$$

For $n=1$, even $\mathbb E[\hat\lambda]$ does not exist.

The MLE is **biased** (overestimates $\lambda$), but **asymptotically unbiased** as $n \to \infty$.

An unbiased estimator would be $\hat{\lambda}_{\text{unbiased}} = \frac{n-1}{\sum x_i}$.

</details>
</div>

### Task 3

* Distribution:
  * Multivariate normal distribution $N(\mu, \Sigma)$
* Parameters:
  * Mean vector $\mu \in \mathbb{R}^d$
  * Covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$
* Data:
  * Each $x_i \in \mathbb{R}^d$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Maximum Likelihood Estimation For Mean</summary>

The PDF is ($x_i \in \mathbb{R}^d$):

$$f(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)$$

---

**Step 1 — Likelihood function:**

$$L(\mu, \Sigma) = \prod_{i=1}^n (2\pi)^{-d/2} |\Sigma|^{-1/2} \exp\left(-\frac{1}{2}(x_i-\mu)^\top \Sigma^{-1}(x_i-\mu)\right)$$

**Step 2 — Log-likelihood:**

$$\ell(\mu, \Sigma) = -\frac{nd}{2}\ln(2\pi) - \frac{n}{2}\ln|\Sigma| - \frac{1}{2}\sum_{i=1}^n (x_i-\mu)^\top \Sigma^{-1}(x_i-\mu)$$

**Step 3a — MLE for $\mu$:**

Taking the derivative w.r.t. $\mu$ and using $\frac{\partial}{\partial \mu}(x_i-\mu)^\top \Sigma^{-1}(x_i-\mu) = -2\Sigma^{-1}(x_i-\mu)$:

$$\frac{\partial \ell}{\partial \mu} = \sum_{i=1}^n \Sigma^{-1}(x_i - \mu) = \Sigma^{-1}\sum_{i=1}^n(x_i - \mu) \stackrel{!}{=} 0$$

Since $\Sigma^{-1}$ is invertible:

$$\sum_{i=1}^n (x_i - \mu) = 0 \quad \Rightarrow \quad \boxed{\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}}$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Maximum Likelihood Estimation For Covariance</summary>

**Step 3b — MLE for $\Sigma$:**

Using matrix calculus identities $\frac{\partial}{\partial \Sigma}\ln|\Sigma| = \Sigma^{-1}$ and $\frac{\partial}{\partial \Sigma} a^\top \Sigma^{-1} a = -\Sigma^{-1}aa^\top\Sigma^{-1}$:

$$\frac{\partial \ell}{\partial \Sigma} = -\frac{n}{2}\Sigma^{-1} + \frac{1}{2}\Sigma^{-1}\left(\sum_{i=1}^n (x_i-\hat{\mu})(x_i-\hat{\mu})^\top\right)\Sigma^{-1} \stackrel{!}{=} 0$$

Multiplying by $\Sigma$ from both sides:

$$\boxed{\hat{\Sigma}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^\top}$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Check whether the MLE estimator is unbiased for Mean</summary>

$\hat{\mu}_{\text{MLE}} = \bar{x}$ is **unbiased**: $\mathbb{E}[\hat{\mu}] = \mu$ $\checkmark$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Check whether the MLE estimator is unbiased for Covariance</summary>

Let

$$\hat\Sigma=\frac1n\sum_{i=1}^n (X_i-\bar X)(X_i-\bar X)^\top$$

Use the identity

$$\sum_{i=1}^n (X_i-\bar X)(X_i-\bar X)^\top = \sum_{i=1}^n (X_i-\mu)(X_i-\mu)^\top - n(\bar X-\mu)(\bar X-\mu)^\top$$

Take expectations:

$$\mathbb E\left[\sum_{i=1}^n (X_i-\mu)(X_i-\mu)^\top\right] = n\Sigma$$

and

$$\mathbb E\left[n(\bar X-\mu)(\bar X-\mu)^\top\right] = n,\mathrm{Cov}(\bar X) = n\cdot \frac{\Sigma}{n} = \Sigma$$

So

$$\mathbb E\left[\sum_{i=1}^n (X_i-\bar X)(X_i-\bar X)^\top\right] = n\Sigma-\Sigma = (n-1)\Sigma$$

Therefore

$$\mathbb E[\hat\Sigma] = \frac1n (n-1)\Sigma = \frac{n-1}{n}\Sigma$$

Hence $\hat\Sigma$ is biased downward.

$$
\boxed{
\mathbb E[\hat\Sigma]=\frac{n-1}{n}\Sigma,
\qquad
\hat\Sigma \text{ is biased.}
}
$$

The unbiased estimator uses the **Bessel correction**: 

$$\hat{\Sigma}_{\text{unbiased}} = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^\top$$

</details>
</div>

## Exercise 3: Auto-correlation function

Assume **stationarity** and **ergodicity**.

### Task 1

* Implement the **auto-correlation function** in Python.
* Input:

  * an $N \times (T+1)$ matrix
    * $N$: number of sample paths
    * $T+1$: number of time points
  * $h_{\max}$: maximum lag
* Output:

  * auto-correlation values for each (h \in {0,\dots,h_{\max}})
* Use the estimator of the **auto-covariance function**
  
  $$\widehat{\mathrm{acov}}(t,t+h)=\frac{1}{T+1-h}\sum_{t=0}^{T-h}(x_t-\bar x)(x_{t+h}-\bar x)$$
  
  with
  
  $$\bar x = \frac{1}{T+1}\sum_{t=0}^T x_t$$
  
* Then use the relationship between **auto-correlation** and **auto-covariance**

### Task 2

* Let $(\varepsilon_t)_{t\in{0,\dots,100}}$ be a **Gaussian white noise** process with variance $\sigma^2=1$
* Sample:
  * $d=50$ paths
  * use $h_{\max}=5$
* Plot the auto-correlation function for:

#### Process 1

* Gaussian white noise:
  
  $$X_t=\varepsilon_t,\quad t\in\lbrace 0,\dots,100\rbrace$$
  

#### Process 2

* Moving-average process of order 1, **MA(1)**:
  [
  X_t = 0.6\varepsilon_{t-1}+\varepsilon_t,\quad t\in{1,\dots,100}, \qquad X_0=\varepsilon_0
  ]

---

## Exercise 2: Rosenbrock function optimization

### Given

* Gradient descent with momentum:
  [
  u_{t+1}=\alpha u_t-\varepsilon \nabla f(x_t), \qquad x_{t+1}=x_t+u_{t+1}
  ]
* Parameters:

  * (\alpha \in [0,1]): momentum parameter
  * (\varepsilon): learning rate
  * (x_0): initial value

### Objective

* Minimize the Rosenbrock-type function
  [
  f(x,y)=(1-x)^2 + 100(1+y-x^2)^2
  ]
* True minimum:
  [
  f(1,0)=0
  ]

### Given derivatives

* Gradient:
  [
  \nabla f(x,y)=
  \begin{pmatrix}
  2x-2-400x(1+y-x^2) \
  200(1+y-x^2)
  \end{pmatrix}
  ]
* Hessian:
  [
  Hf(x,y)=
  \begin{pmatrix}
  2-400(1+y-3x^2) & -400x \
  -400x & 200
  \end{pmatrix}
  ]

### Task 1

* Implement:

  * gradient descent with momentum for

    * (\alpha=0)
    * (\alpha=0.8)
    * (\alpha=0.9)
    * (\alpha=0.95)
  * Newton–Raphson
* Do **not** use PyTorch or other machine learning packages
* Use common settings for comparison:

  * learning rate (\varepsilon=0.001)
  * initial value (x_0=(-1,1))
  * number of iterations: (1000)

### Task 2

* Plot the results:

  * use `imshow` from Matplotlib to display the Rosenbrock function as a 2D image
  * draw the trajectories
    [
    x_0,x_1,\dots,x_{1000}
    ]
    on top of it
* Optional:

  * use `matplotlib.animation` to animate the optimization process

---

## Exercise 3: Barrier option pricing

### Given

* Stock price process:
  [
  S_t = S_0 e^{\sigma B_t - \frac{\sigma^2}{2}t}, \quad t\in[0,T], \quad S_0\ge 0
  ]
* ( (B_t)_{t\in[0,T]} ): Brownian motion
* (\sigma>0): volatility

### Barrier option payoff

* For barrier (B>0), strike (K<B), and maturity (T>0):
  [
  P(B,K,T)=\max{0,S_T-K}\mathbf{1}*{{M_T<B}}
  ]
  where
  [
  M_T=\max*{t\in[0,T]} S_t
  ]
* Interpretation:

  * payoff is (S_T-K) if positive
  * but becomes zero if the stock price ever reaches the barrier (B)

### Task 1

* Write a Python function that simulates a **discretized Brownian motion**
  [
  B_0, B_{T/N}, B_{2T/N}, \dots, B_T
  ]
  on the grid
  [
  0,\frac{T}{N},\frac{2T}{N},\dots,T
  ]
* Hint:

  * use the distribution of Brownian increments
  * (B_0=0)

### Task 2

* Use the function from Task 1 to simulate

  * (d=10{,}000) price processes
  * initial price (S_0=5)
  * volatility (\sigma=1)
  * time interval ([0,1])
  * (N=1000) discretization points
* Construct each price path by inserting the simulated Brownian motion into the stock-price formula

### Task 3

* Consider a barrier option with:

  * strike (K=2)
  * barrier (B=10)
  * maturity (T=1)
  * offered price (0.60)
* For each simulated Brownian motion path:

  * compute the realized payoff
* Compute:

  * the sample mean payoff
* Decide:

  * whether the offered price seems fair

If you want, I can also turn this into a **compact tree diagram** or a **checklist version**.

Here is the hierarchy for **Exercise Sheet 3** 

## Exercise 1: MLE and LSE equivalence

### Goal

* Derive the MLE for the coefficient matrix (B) in a multivariate Gaussian linear model
* Show that it coincides with the multivariate least-squares estimator
* Check whether the estimator is unbiased

### Given

* Responses:

  * (Y_1,\dots,Y_n \in \mathbb{R}^d)
* Regressors:

  * (x_i \in \mathbb{R}^{p-1})
  * augmented regressor:
    [
    \tilde x_i = (1, x_i^\top)^\top \in \mathbb{R}^p
    ]
* Design matrix:

  * (X \in \mathbb{R}^{n\times p})
  * leading column of ones
  * (\mathrm{rank}(X)=p), (n>p)
* Model:
  [
  Y = XB + \varepsilon,\qquad \varepsilon \overset{i.i.d.}{\sim} N_d(0,\sigma^2 I_{n\times d})
  ]
* Unknown:

  * (B \in \mathbb{R}^{p\times d})
* Known:

  * (\sigma^2>0)

### Tasks

* State the likelihood function
* State the log-likelihood function
* Maximize the log-likelihood by finding the roots of the score with respect to (B)
* Check whether the estimator is unbiased

---

## Exercise 2: Linear regression

### Task 1

* Simulate and plot the periodic noisy time series
  [
  X_t = f_\beta(t)+\varepsilon_t = \beta_0 + \beta_1\sin(t)+\beta_2\cos(t)+\varepsilon_t,
  \qquad t\in{0,\dots,100}
  ]
* Parameters:
  * (\beta_0=2)
  * (\beta_1=10)
  * (\beta_2=5)
* Noise:

  * ((\varepsilon_t)_{t\in{0,\dots,100}}) is Gaussian white noise

### Task 2

* Assume the true function (f_\beta) is unknown
* Perform linear regression with predictor variables:

  * (u_{1,t}=\sin(t))
  * (u_{2,t}=\cos(t))
* Use the model
  [
  x_t = \beta_0 + \beta_1\sin(t)+\beta_2\cos(t)+\varepsilon_t,
  \qquad t\in{0,\dots,100}
  ]
* Compute the least-squares estimator (\hat\beta_{\mathrm{LSE}})

### Task 3

* Test the predictions from the fitted model
* Plot:

  * predictions against the true function
  * QQ-plot
  * histogram of residuals
  * residuals

### Task 4

* Repeat Tasks 2 and 3 for the reduced model
  [
  x_t = \beta_0 + \beta_1\cos(t)+\varepsilon_t,
  \qquad t\in{0,\dots,100}
  ]
* Explain what changed

---

## Exercise 3: Linear collinearity

### Given

* Standard fixed-design linear regression model:
  [
  Y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \varepsilon_i
  ]
* Errors:
  [
  \varepsilon_i \sim N(0,1)
  ]
  independently
* Sample size:

  * (n=100)
* True parameters:

  * (\beta_0=1)
  * (\beta_1=2)
  * (\beta_2=3)

### Task 1: Approximately independent predictors

#### Data generation

* For (i=1,\dots,n), generate independently:
  [
  X_{i1}\sim N(0,1),\qquad X_{i2}\sim N(0,1)
  ]
* Generate:
  [
  \varepsilon_i\sim N(0,1)
  ]
* Define:
  [
  Y_i = 1 + 2x_{i1} + 3x_{i2} + \varepsilon_i
  ]

#### Tasks

* Fit the linear model
  [
  Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon
  ]
* Report estimated coefficients:

  * ((\hat\beta_0,\hat\beta_1,\hat\beta_2))
* Compare them qualitatively to the true values ((1,2,3))
* Compute the sample correlation:
  [
  \widehat{\mathrm{corr}}(X_1,X_2)
  ]
* Comment on whether the predictors appear approximately uncorrelated

### Task 2: Perfectly collinear predictors

#### Data generation

* For (i=1,\dots,n), generate:
  [
  X_{i1}\sim N(0,1)
  ]
* Define:
  [
  X_{i2}=2X_{i1}
  ]
* Generate:
  [
  \varepsilon_i\sim N(0,1)
  ]
* Define:
  [
  Y_i = 1 + 2x_{i1} + 3x_{i2} + \varepsilon_i
  ]

#### Tasks

* Fit the same linear model
  [
  Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon
  ]
* Answer:

  1. What warning or error message does the software output?
  2. What happens to the reported coefficient estimates?

     * Is one predictor dropped automatically?
     * Is one coefficient undefined?
* Compute the sample correlation:
  [
  \widehat{\mathrm{corr}}(X_1,X_2)
  ]

I can also merge sheets 1–3 into one clean master hierarchy.

Here is the hierarchy for **Exercise Sheet 4** 

## Exercise 1: Logistic Regression

### Task 1

* Simulate and plot a noisy time series ((X_t)_{t\in{0,\dots,100}})
* Data-generating model:
  [
  X_t \sim \mathrm{Ber}!\left(\sigma(\beta_0+\beta_1\sin(t)+\beta_2\cos(t))\right),
  \quad t\in{0,\dots,100}
  ]
* Parameters:

  * (\beta_0=2)
  * (\beta_1=10)
  * (\beta_2=5)
* Logistic function:
  [
  \sigma(x)=\frac{1}{1+e^{-x}}
  ]
* Bernoulli distribution:

  * ( \mathrm{Ber}(\pi) )

### Task 2

* Assume the true parameters are unknown
* Perform logistic regression to estimate:

  * (\hat\beta_0)
  * (\hat\beta_1)
  * (\hat\beta_2)
* Implement the **Newton–Raphson** method
* Use it to compute the MLE:
  [
  \hat\beta^{\mathrm{MLE}}
  ]

### Task 3

* Plot over Newton–Raphson iterations:

  * the negative log-likelihood loss
  * the parameter estimates
* Compare the estimated parameters with the true values

---

## Exercise 2: Autoregressive Processes

### Task 1

* Write a Python function to simulate an autoregressive process of order (p\in\mathbb{N})
* Inputs:

  * parameter vector (a), containing:

    * intercept
    * autoregressive coefficients
  * number of steps (N\in\mathbb{N}), with (N>p)
* Simulate for (N=100) using:

#### Case 1

* (a_0=0)
* (a_1=1.5)

#### Case 2

* (a_0=0)
* (a_1=0.5)

#### Case 3

* (a_0=0)
* (a_1=1)

#### Then

* Discuss your observations

### Task 2

* Write a Python function to estimate the **order** of an autoregressive process
* Use the **partial auto-correlation function (PACF)**
* Input:

  * time series (X=(x_t)_{t\in{0,\dots,N}})
  * maximum lag/order to test

#### For each lag (l>1)

* Perform a linear regression with:

  * predictor variables:
    [
    x_{t-l+1},\dots,x_{t-1}
    ]
  * outcome variables:

    * (x_t)
    * (x_{t-l})
  * for all (t\in{l,\dots,N})
* Compute the two residual series
* Use their correlation as the estimate of the PACF value at lag (l)

#### Also include

* the partial auto-correlation for:

  * (l=0)
  * (l=1)

### Task 3

* Write a Python function to estimate the parameters of an autoregressive process
* Inputs:

  * time series (X=(x_t)_{t\in{1,\dots,N}})
  * estimated order (\hat p\in\mathbb{N})
* Method:

  * perform linear regression with:

    * predictor variables:
      [
      x_{t-p},\dots,x_{t-1}
      ]
    * outcome variable:
      [
      x_t
      ]
    * for all (t\in{p,\dots,N})

### Task 4

* Simulate a stationary autoregressive process of your choice using the function from Task 1
* Test:

  * the order-estimation function from Task 2
  * the parameter-estimation function from Task 3

---

## Exercise 3: Vector Autoregressive Processes

### Task 1

* Let
  [
  \varepsilon=\left((\varepsilon_t^1,\varepsilon_t^2)^T\right)_{t\in{0,\dots,1000}}
  ]
  be a two-dimensional Gaussian white noise process
* Transform the following processes into a **VAR(1)** process

#### Case 1: AR(2) process

* Process:
  [
  X_t = 0.5X_{t-1}-0.3X_{t-2}+\varepsilon_t^1
  ]
* For:
  [
  t\in{2,\dots,100}
  ]
* Initial values:

  * (X_0=x_0\in\mathbb{R})
  * (X_1=x_1\in\mathbb{R})

#### Case 2: VAR(2) process

* Process:
  [
  X_t=
  \begin{pmatrix}
  0.4 & 0.2 \
  0.1 & 0.5
  \end{pmatrix}
  X_{t-1}
  +
  \begin{pmatrix}
  -0.1 & 0.3 \
  0.2 & 0.2
  \end{pmatrix}
  X_{t-2}
  +
  \begin{pmatrix}
  \varepsilon_t^1 \
  \varepsilon_t^2
  \end{pmatrix}
  ]
* For:
  [
  t\in{2,\dots,100}
  ]
* Initial values:

  * (X_0=x_0\in\mathbb{R}^2)
  * (X_1=x_1\in\mathbb{R}^2)

### Task 2

* Consider the process
  [
  X_t = aX_{t-1}+0.5X_{t-2}+\varepsilon_t
  ]
* For:
  [
  t\in{2,\dots,100}
  ]
* Initial values:

  * (X_0=x_0\in\mathbb{R})
  * (X_1=x_1\in\mathbb{R})
* Goal:

  * prove analytically for which (a\in\mathbb{R}) the process is stationary

I can also merge **Sheets 1–4** into one single master hierarchy in the same format.

Here is the hierarchy for **Exercise Sheet 5** 

## Exercise 1: Hierarchical Models

### General task

For each of the three scenarios:

* propose a **hierarchical time series model**
* write the model explicitly by stating:

  * distributions
  * parameters
* explain why the model is appropriate
* no likelihood derivation is required

### Scenario 1: Bears and heart rates

#### Given

* (B) bears
* heart rates:
  [
  y_t^{(b)} \in \mathbb{R}, \quad t\in\mathbb{Z}, \quad b=1,\dots,B
  ]
* heart rates are generally similar across bears
* they fluctuate depending on random activity
* some bears are hibernating
* hibernation indicator:
  [
  H_b \in {0,1}
  ]

#### Modeling goal

* build a hierarchical model that captures:

  * shared similarity of heart rates across bears
  * within-bear fluctuations over time
  * large shift caused by hibernation status

### Scenario 2: Plant species and snails

#### Given

* (N) plant species
* plant counts:
  [
  p_t^{(n)} \in \mathbb{N}, \quad n=1,\dots,N,\quad t\in\mathbb{Z}
  ]
* snail counts:
  [
  s_t \in \mathbb{N}, \quad t\in\mathbb{Z}
  ]
* current plant count depends on past plant count
* more plants can produce more seeds
* snails reduce reproduction
* snail impact differs by plant species

#### Modeling goal

* build a hierarchical model that captures:

  * count-valued time series
  * temporal dependence on past plant numbers
  * shared snail effect structure
  * species-specific sensitivity to snails

### Scenario 3: Temperatures across locations

#### Given

* $L$ locations in Germany
* daily mean temperatures:
  
  $$h_t^{(l)} \in \mathbb{R}, \quad l=1,\dots,L,\quad t\in\mathbb{Z}$$
  
* one year corresponds to 365 time steps
* there is:
  * a seasonal trend
  * a linear upward trend due to climate change
* both trends are the same across all locations
* locations differ in mean temperature because of elevation

#### Modeling goal

* build a hierarchical model that captures:
  * a global seasonal component
  * a global linear trend
  * location-specific baseline offsets


## Exercise 2: Fourier Analysis

### General task

* Several discrete time series are shown on the **left**
* Several power spectra are shown on the **right**
* Determine which time series corresponds to which power spectrum

### What is given

* 10 time series plots, labeled:
  * $1,2,\dots,10$
* 10 power spectrum plots, labeled:
  * $A,B,\dots,J$

### Goal

* match each left-side time series to its correct right-side power spectrum

### Implicit analysis steps

* identify whether a signal contains:
  * one dominant frequency
  * multiple frequencies
  * low-frequency or high-frequency oscillations
  * noise or mixed components
* use these properties to match each time-domain signal to its spectrum

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/fourier_analysis_exercise.png' | relative_url }}" alt="Filtering Smoothing Schema" loading="lazy">
  <figcaption>Left: time series plot. Right: power spectrum plot.</figcaption>
</figure>

## Exercise 3: Denoising

### Task 1: Detect signal vs. pure noise

#### Given

* file:
  * `raw_time_series.py`
* contains:
  * a NumPy array with $10$ noisy time series
* each time series has:
  * $1000$ time steps
* sampling frequency:
  * $200$ Hz
* some time series contain a signal with several frequency components
* some time series are pure noise

#### Required steps

* load the array using `numpy.load`
* compute the Discrete Fourier Transform using `numpy.fft`
* inspect the frequency components
* decide:
  * which time series are pure noise
  * which contain a signal

### Task 2: Denoise the time series

#### Required steps

* remove frequency components that do not clearly belong to a signal
* set those frequency components to $0$
* transform the modified frequency representation back using the inverse DFT
* plot:
  * original noisy time series
  * denoised time series
* compare the denoised result against the original

If you want, I can also merge **Sheets 1–5** into one master hierarchy.

Here is the hierarchy for **Exercise Sheet 6** 

## Exercise 1: Granger Causality

### Task 1

* Simulate a two-dimensional **VAR(2)** process
* Noise process:
  
  $$\varepsilon = \big((\varepsilon_t^1,\varepsilon_t^2)^T\big)_{t\in{0,\dots,1000}}$$
  
  is Gaussian white noise with covariance matrix
  
  $$
  \Sigma=
  \begin{pmatrix}
  0.01 & 0\\
  0 & 0.01
  \end{pmatrix}
  $$

* Process:
  
  $$
  X_t=
  \begin{pmatrix}
  0.2 & -2\\
  0 & 0.1
  \end{pmatrix}
  X_{t-1}
  +
  \begin{pmatrix}
  -0.1 & -1\\
  0 & 0.1
  \end{pmatrix}
  X_{t-2}
  +
  \begin{pmatrix}
  \varepsilon_t^1\\
  \varepsilon_t^2
  \end{pmatrix},
  \qquad t\in{2,\dots,1000}
  $$

* Initial values:
  
  $$
  X_0=
  \begin{pmatrix}
  0\\
  0
  \end{pmatrix},
  \qquad
  X_1=
  \begin{pmatrix}
  0\\
  0
  \end{pmatrix}
  $$

### Task 2

* Using the model parameters from Task 1, determine:
  * whether $X^1$ Granger-causes $X^2$
  * whether $X^2$ Granger-causes $X^1$
* Explain the answer using **Granger causality definition**

### Task 3

* Use the **log-likelihood-ratio test statistic** to confirm the conclusion from Task 2
* Hint from sheet:
  * either impose constraints so that tested parameters remain zero
  * or reparameterize the VAR coefficient matrices cleverly

## Exercise 2: Change Point Detection

### Given

* A sequence of coin toss results stored in:
  * `coin_heads.npy`
* Encoding:
  * head $=1$
  * tail $=0$

### Model

* Fit a **sigmoid change-point model**
  
  $$\pi_t = p_0 + \frac{p_1-p_0}{1+\exp!\left(\frac{c-t}{a}\right)}$$
  
* Interpretation:
  * $p_0$: original head probability
  * $p_1$: changed head probability

### Constraints

* $0 \le p_0, p_1 \le 1$
* $a>0$
* Therefore:
  * use a **constrained optimization** method for maximizing the log-likelihood

### Questions to answer

1. Did the wizard really enchant the coin?
2. If yes:
   * when did the change happen?
   * how fast did it happen?
3. Was the original impression correct that the coin was initially fair?

## Exercise 3: Autoregressive Poisson Process

### Given model

* One-dimensional autoregressive Poisson process:
  [
  x_t \sim \mathrm{Poisson}(\lambda_t),
  \qquad
  \log \lambda_t = a_0 + a_1 x_{t-1}
  ]

### Task 1

* Derive the **log-likelihood function**

### Task 2

* Compute the gradients of the log-likelihood with respect to:

  * (a_0)
  * (a_1)
* Explain why the log-likelihood cannot be maximized analytically

### Task 3

* Discuss whether **gradient descent** converges to a **global maximum** of the log-likelihood
* Explain why or why not


## Exercise 1: Dynamical systems quiz

### General task

* Decide which statements are correct
* For each statement:

  * say whether it is true or false
  * explain why

### Statement 1

* Claim:

  * bifurcation plots show qualitative changes in a dynamical system’s behavior over the course of time

### Statement 2

* Claim:

  * if a dynamical system (f(x_t)) has a cycle of order (p), then there exists another dynamical system (g(x_t)) with (p) distinct fixed points, namely exactly the points on that cycle

### Statement 3

* Claim:

  * the maximum Lyapunov exponent is
    [
    \lambda_{\max}=\lim_{t\to\infty}\frac{1}{t}\sum_{\tau=0}^{t-1}\log|f'(x_\tau)|
    ]
  * and if (\lambda_{\max}>0), then the nonlinear dynamical system is guaranteed to be chaotic

### Statement 4

* Claim:

  * a nonlinear dynamical model may fit time series data well even if the mean squared error is large

## Exercise 2: Logistic map

### Given

* Map:
  [
  x_{n+1}=f(x_n)=r x_n(1-x_n)
  ]

### Task 1

* Prove that for

  * (0 \le x_n \le 1)
  * (0 \le r \le 4)
* one has
  [
  0 \le f(x_n) \le 1
  ]

### Task 2

* Use the given cobweb-plot construction:

  * plot ((x_1,x_2))
  * connect to ((x_2,x_2))
  * connect to ((x_2,x_3))
  * continue similarly
* Also include:

  * the diagonal (y=x)
  * the map (f)
* Starting from some (x_1 \in (0,1)), make cobweb plots with 30 steps for:

  * (r=0.5)
  * (r=1.5)
  * (r=2.5)
  * (r=3.5)
  * (r=3.9)
* Then answer:

  * does the logistic map have a cycle for one of these (r)-values?

### Task 3

* For each
  [
  r \in {0.001,0.002,0.003,\dots,3.998,3.999}
  ]
* produce:

  * 1000 trajectories
  * random initial conditions
  * 100 steps each
* Plot:

  * only the endpoints of the trajectories
  * against their corresponding (r)-value
  * in a 2D scatter plot
* Hints from sheet:

  * only keep the last points, not the full trajectories
  * make the code flexible enough for other maps too

### Task 4

* Redo the bifurcation-style endpoint plot for
  [
  r \in {3.44500, 3.44501, 3.44502, \dots, 3.56999, 3.57000}
  ]
* Use the plot to find:

  * another value of (r) such that the logistic map has a cycle of order (p \ge 3)

---

## Exercise 3: Bifurcations in discrete dynamical systems

### General task

For each system:

1. Identify the equilibrium points / fixed points
2. Analyze local stability near those points
3. Find the condition on the bifurcation parameter (\mu) where:

   * existence changes, or
   * stability changes
4. Describe what happens
5. Draw a bifurcation diagram with (\mu) on the x-axis

### Task 1: Saddle-node bifurcation

* System:
  [
  x_{t+1}=-x_t^2+x_t+\mu
  ]
* Variables:

  * (x_t \in \mathbb{R})
  * (\mu \in \mathbb{R})

### Task 2: Transcritical bifurcation

* System:
  [
  x_{t+1}=(\mu-x_t)x_t
  ]
* Variables:

  * (x_t \in \mathbb{R})
  * (\mu \in \mathbb{R})

### Task 3: Neimark–Sacker bifurcation

* System:
  [
  x_{t+1}=(1+\mu)
  \begin{pmatrix}
  \cos(\theta) & -\sin(\theta)\
  \sin(\theta) & \cos(\theta)
  \end{pmatrix}
  x_t - |x_t|^2 x_t
  ]
* Variables:

  * (x_t \in \mathbb{R}^2)
  * (\mu \in \mathbb{R})
* Parameter condition:

  * (\theta \in (0,\pi)) is fixed
  * (\theta) is not a rational multiple of (\pi)

## Exercise 1: Hopfield Network

### Given

* (p) binary neurons
  [
  s_i \in {-1,1}
  ]
* (M \le p) stored patterns
  [
  \xi_i^\mu,\qquad \mu=1,\dots,M,; i=1,\dots,p
  ]
* Energy of a configuration:
  [
  E(s) = -\frac12 s^\top W s
  ]
* Weight matrix from the Hebbian learning rule:
  [
  W = \frac1p \sum_{\mu=1}^M \left(\xi^\mu (\xi^\mu)^\top - I_p\right)
  ]

### Task 1

* Compute the energy difference
  [
  \Delta E_k
  ]
  caused by flipping a single neuron
  [
  s_k \to -s_k
  ]
* Show that the asynchronous update rule
  [
  s_k \leftarrow \operatorname{sign}!\left(\sum_{j=1}^p W_{kj}s_j\right)
  ]
  minimizes the network energy

### Task 2

* Argue that the network always converges to a fixed point
* Condition:

  * updates that leave the energy unchanged are not allowed

### Task 3 (optional)

* Determine the condition under which a pattern
  [
  \xi^\mu
  ]
  is a fixed point
* Assume the overlap between patterns is sufficiently small:
  [
  |m^{\mu\nu}| :=
  \left|
  \frac1p \sum_{j=1}^p \xi_j^\mu \xi_j^\nu
  \right|
  \le
  \frac{1-\frac{M}{p}}{M-1}
  \qquad \forall \mu \ne \nu
  ]
* Prove that under this condition, all stored patterns are fixed points of the network

---

## Exercise 2: Linearization

### Given

* A smooth map
  [
  f:\mathbb{R}^n \to \mathbb{R}^n
  ]
* Discrete-time dynamical system:
  [
  x_{t+1} = f(x_t)
  ]
* Fixed point:
  [
  x^* \quad \text{with} \quad f(x^*) = x^*
  ]
* Jacobian:
  [
  J = \frac{df}{dx}
  ]
* Assumption:

  * (J) is diagonalizable
  * eigenvalues are
    [
    \lambda_1,\dots,\lambda_n
    ]

### Task 1

* Prove:

#### Case 1: Stable fixed point

* If all eigenvalues satisfy
  [
  |\lambda_i| < 1
  ]
  then, in a neighborhood of (x^*), all trajectories converge to (x^*) exponentially fast

#### Case 2: Unstable fixed point

* If all eigenvalues satisfy
  [
  |\lambda_i| > 1
  ]
  then, in a neighborhood of (x^*), all trajectories diverge away from (x^*) exponentially fast

#### Hint / method

* First linearize the nonlinear system near the fixed point
* Then use diagonalizability of the Jacobian

### Task 2

* Analyze the mixed case:

  * there exist (i,j) such that
    [
    |\lambda_i| < 1,\qquad |\lambda_j| > 1
    ]
* Explain what happens in that situation

---

## Exercise 3: BPTT and Vanishing Gradients

### Given

* Multivariate RNN:
  [
  z_t = \phi(W_z z_{t-1} + W_x x_t + b)
  ]
  [
  y_t = W_y z_t
  ]
* Variables:

  * (z_t \in \mathbb{R}^n): hidden state
  * (x_t \in \mathbb{R}^m): input
  * (y_t \in \mathbb{R}^p): output
* Activation:
  [
  \phi:\mathbb{R}\to\mathbb{R}
  ]
  differentiable
* Loss:
  [
  L = \sum_{t=1}^T |y_t - \hat y_t|_2^2
  ]
* Goal:

  * reconstruct the target sequence (\hat y_t \in \mathbb{R}^p)

### Task 1

* Compute the gradients:
  [
  \nabla_{W_z} L,\qquad \nabla_{W_x} L,\qquad \nabla_{W_y} L
  ]

### Task 2

* Use the recursive gradient expression involving products of the form
  [
  \prod_{j=k+1}^{t}
  \phi'!\bigl(W_z z_{j-1}+W_x x_j+b\bigr), W_z
  ]
* Find conditions on:

  * the singular values of (W_z)
  * the derivative (\phi')
* such that gradients do **not**:

  * vanish
  * explode
* when propagated backward through time

## Exercise 1: Kalman Filter

### Given

* Linear Gaussian state-space model (SSM)
* Latent state:
  [
  z_t \in \mathbb{R}^M
  ]
* Observation:
  [
  x_t \in \mathbb{R}^N
  ]
* Observation equation:
  [
  x_t = B z_t + \eta_t,\qquad \eta_t \sim N(0,\Gamma)
  ]
* State equation:
  [
  z_t = A z_{t-1} + C u_t + \epsilon_t,\qquad \epsilon_t \sim N(0,\Sigma)
  ]
* Assumption in this sheet:

  * autonomous dynamics, so
    [
    u_t = 0 \quad \forall t
    ]
* Initial state:
  [
  z_0 \sim N(\mu_0,\Sigma_0)
  ]

### Goal

* Compute the filtering posterior
  [
  p(z_t\mid x_{1:t})
  ]
* Kalman filter gives the optimal causal estimator

### Task 1

* Write down and explain the assumptions of SSMs introduced in the lecture
* Briefly explain the role of each assumption in the context of the Kalman filter

### Task 2

Assume that at time (t-1) the posterior is already known:
[
p(z_{t-1}\mid x_{1:t-1}) = N(\mu_{t-1\mid t-1},V_{t-1\mid t-1})
]

#### Part (a): Prediction step

* Derive the mean
  [
  \mu_{t\mid t-1}
  ]
  of the predictive distribution
* Derive the covariance
  [
  V_{t\mid t-1}
  ]
  of the predictive distribution
* Target distribution:
  [
  p(z_t\mid x_{1:t-1})
  ]

#### Part (b): Update step

* Compute the posterior
  [
  p(z_t\mid x_{1:t}) \propto p(x_t\mid z_t),p(z_t\mid x_{1:t-1})
  ]
* Use the linear correction rule
  [
  \hat z_t = \mu_{t\mid t-1} + K_t(x_t - B\mu_{t\mid t-1})
  ]
* Define the estimation error
  [
  e_t = z_t - \hat z_t
  ]

##### Part (i)

* Show that
  [
  e_t = (I-K_tB)(z_t-\mu_{t\mid t-1}) - K_t\eta_t
  ]

##### Part (ii)

* Using independence of prediction error and observation noise, show that
  [
  V_{t\mid t} = \mathbb{E}[e_t e_t^\top]
  ]
  is given by
  [
  V_{t\mid t} =
  (I-K_tB)V_{t\mid t-1}(I-K_tB)^\top + K_t\Gamma K_t^\top
  ]
* Then interpret the posterior covariance in one sentence

##### Part (iii)

* Minimize the trace of
  [
  V_{t\mid t}
  ]
  with respect to (K_t)
* Derive the closed-form expression for the Kalman gain
  [
  K_t
  ]
* Then interpret the Kalman gain in one sentence

---

## Exercise 2: System Identification with the Kalman Filter

### Given

* In many applications, the SSM parameters
  [
  (A,B,\Sigma,\Gamma)
  ]
  are unknown
* They must be estimated from observations
  [
  x_{0:T}
  ]
* This is called **system identification**
* Use the **Expectation-Maximization (EM)** algorithm
* Latent states:
  [
  z_{0:T}
  ]
  are treated as hidden variables
* ELBO:
  [
  \mathcal L(\theta,q,x_{0:T})
  ============================

  \mathbb E_{q(z_{0:T})}!\left[\log p(z_{0:T},x_{0:T}\mid \theta)\right] + H(q)
  ]

### Task 1

* Explain precisely why the latent variables
  [
  z_{0:T}
  ]
  make the marginal log-likelihood
  [
  \ell(\theta)
  ]
  intractable

### Task 2

* Assume only the observation matrix (B) is unknown
* All other parameters are fixed
* Derive the M-step for (B)
* Write down the expected log-joint likelihood objective, ignoring constants w.r.t. (B)
* Show all steps leading to the update rule
  [
  B_{\text{new}}
  ==============

  \left(\sum_{t=1}^T x_t,\mathbb E_q[z_t]^\top\right)
  \left(\sum_{t=1}^T \mathbb E_q[z_t z_t^\top]\right)^{-1}
  ]

### Task 3

* Compare the structure of (B_{\text{new}}) to the OLS estimator from linear regression
* Explain in one sentence what the expectations
  [
  \mathbb E_q[\cdot]
  ]
  represent in this context

---

## Exercise 3: ISS Tracking with the Kalman Filter

### Given

* Goal:

  * track the International Space Station (ISS) from noisy position measurements
  * learn latent system dynamics simultaneously
* Ground-truth trajectory:

  * generated from a real TLE orbit description and a physics-based propagator
* Observations:

  * synthetic noisy measurements created by adding Gaussian noise

### State-space model

* Latent state:
  [
  z_t =
  \begin{pmatrix}
  p_t\
  v_t
  \end{pmatrix}
  \in \mathbb{R}^6
  ]
* Position:
  [
  p_t \in \mathbb{R}^3
  ]
* Velocity:
  [
  v_t \in \mathbb{R}^3
  ]
* Dynamics:
  [
  z_t = A z_{t-1} + \epsilon_t,\qquad \epsilon_t \sim N(0,\Sigma)
  ]
* Observation model:
  [
  x_t = B z_t + \eta_t,\qquad \eta_t \sim N(0,\Gamma)
  ]
* Observation matrix:
  [
  B = (I_3;;0) \in \mathbb{R}^{3\times 6}
  ]
* Meaning:

  * only the position component is observed directly

### Task 1

* Execute the data-generation cell in the notebook
* Complete the `LinearGaussianSSM` class
* Implement the Kalman filter prediction and update steps from Exercise 1

### Task 2

* Complete the EM algorithm for (A) inside `train_em_A`
* Implement the M-step update for (A) derived in the lecture

### Task 3

* Run the EM training loop
* Run the visualization of the reconstructed trajectories by filter and smoother
* Discuss:

  * whether reconstruction was successful
  * why filter and smoother estimates are similar in this setup
  * two scenarios where a smoother would be strongly preferred over a filter

### Task 4

* Visualize the eigenvalues of the learned matrix (A)
* Consider the deterministic system
  [
  z_t = A z_{t-1}
  ]
* Discuss:

  * what the eigenvalues imply about stability
  * what they imply about long-term behavior
  * whether this is consistent with a bounded trajectory over the observed time window

I can also merge **Sheets 1–9** into one single master hierarchy.
