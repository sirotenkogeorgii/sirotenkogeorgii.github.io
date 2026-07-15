---
title: Non-Stationary Processes and Recency-Weighted Estimation
layout: default
noindex: true
math: true
tags:
  - machine-learning
  - reinforcement-learning
  - multi-armed-bandits
  - time-series
  - stochastic-processes
  - stochastic-approximation
  - optimization
---

These notes grew out of the constant step-size section of the [RL notes](../../index.md). Starting from the observation that sample averages fail on non-stationary bandits, they work outward: what *stationarity* precisely means for a stochastic process, what the unit root phenomenon is, in what sense AR/VAR models are (and are not) examples of sample-average failure, how stochastic gradient descent itself can be read as value tracking in a non-stationary environment, and how adaptive step sizes — from the Kalman gain through Adam to attention — generalize the same recursive template

$$
Q_{t+1} \;=\; Q_t + \alpha_t\,(R_t - Q_t).
$$

## The problem: recency bias in value estimation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Sample averages fail in nonstationary problems)</span></p>

The sample average weights every observed reward *equally*. This is ideal when the true value $q\_\ast(a)$ is a fixed constant — the LLN then tells us $Q\_t(a) \to q\_\ast(a)$. But if $q\_\ast(a)$ *drifts over time*, equal weighting is a disaster: ancient rewards from a stale regime drag the estimate away from the current truth just as hard as fresh rewards pull it toward it.

**In a non-stationary bandit, convergence to a fixed number is the *wrong* goal**

$$\implies$$

What we want is **tracking** — an estimate that follows the moving target.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(Constant Step-Size Update)</span></p>

For a non-stationary bandit we replace the shrinking step size $1/n$ by a **constant** $\alpha \in (0, 1]$:

$$
Q_{n+1} \;=\; Q_n + \alpha\,(R_n - Q_n)
\;=\; (1 - \alpha)\, Q_n + \alpha\, R_n.
$$

A constant step size makes recent rewards matter more than old ones.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Exponential Recency Weighting)</span></p>

Unrolling the constant step-size recursion from an initial estimate $Q\_1$:

$$
Q_{n+1} \;=\; (1 - \alpha)^n Q_1 + \sum_{i=1}^n \alpha \,(1 - \alpha)^{n - i}\, R_i.
$$

The coefficient of $R\_i$ is $\alpha(1 - \alpha)^{n-i}$, which decays geometrically in $n - i$ — the age of the reward.

* Recent rewards receive the largest weights.
* Older rewards are downweighted exponentially fast.
* A constant step size therefore implements an **exponentially recency-weighted average**.

This is the mechanism that lets the estimate forget obsolete data and track the current value.

</div>

The rest of these notes make precise what "non-stationary" means, when the sample average genuinely fails, and how far the constant-$\alpha$ template can be generalized.

## What "stationary" actually means

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Key idea</span><span class="math-callout__name">(Stationarity is a property of the whole process)</span></p>

Stationarity is a property of the **entire stochastic process** $(X\_t)\_{t \in T}$, describing invariance under shifts of time. It is not merely a property of each $X\_t$ considered separately.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Strict stationarity)</span></p>

A process $(X\_t)$ is **strictly stationary** if, for every $k$, every choice of times $t\_1,\dots,t\_k$, and every shift $h$,

$$
(X_{t_1},\dots,X_{t_k})
\;\stackrel{d}{=}\;
(X_{t_1+h},\dots,X_{t_k+h}).
$$

Thus all finite-dimensional joint distributions are invariant under time translation.

</div>

In particular, taking $k = 1$,

$$
X_t \stackrel{d}{=} X_{t+h},
$$

so every $X\_t$ has the same marginal distribution. But equal marginals alone are **not sufficient**: the dependence structure must also be invariant. For example, stationarity requires that the joint law of $(X\_t, X\_{t+1})$ be the same as that of $(X\_{t+10}, X\_{t+11})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak / covariance stationarity)</span></p>

When second moments exist, one often uses the weaker notion of **weak** or **covariance stationarity**:

$$
\mathbb{E}[X_t] = \mu
$$

is independent of $t$, and

$$
\operatorname{Cov}(X_t, X_s) = \gamma(t-s)
$$

depends only on the time lag $t - s$, not on the absolute times $t, s$. Hence:

* constant mean;
* constant variance $\gamma(0)$;
* time-invariant autocovariance structure.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relations between the two notions)</span></p>

Strict stationarity does not automatically imply weak stationarity unless the second moments exist. For Gaussian processes, however, mean and covariance determine all finite-dimensional distributions, so weak stationarity implies strict stationarity.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Marginals vs. the joint process)</span></p>

Does stationarity concern each $X\_t$, or the whole process? Both, but principally the whole process. Stationarity implies identical marginal laws, $X\_t \stackrel{d}{=} X\_s$, but it also constrains every joint distribution. Two processes can have the same marginal distribution at every time while not being stationary, because their temporal dependence changes. So the correct hierarchy is

$$
\boxed{\text{stationary joint process}
\implies
\text{identical marginals},}
$$

but not conversely.

</div>

### Random walks: non-stationary with stationary increments

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Random walk)</span></p>

Consider

$$
X_t = X_{t-1} + \varepsilon_t,
\qquad
X_0 = 0,
$$

where $(\varepsilon\_t)$ are i.i.d. with mean $0$ and variance $\sigma^2$. Then

$$
X_t = \sum_{i=1}^t \varepsilon_i,
\qquad
\operatorname{Var}(X_t) = t\sigma^2,
$$

so the variance grows with $t$. The marginal distribution changes over time, hence the random walk is neither weakly nor strictly stationary.

A **Gaussian random walk**, with $\varepsilon\_t \sim \mathcal{N}(0, \sigma^2)$, satisfies

$$
X_t \sim \mathcal{N}(0, t\sigma^2).
$$

It is still non-stationary: Gaussianity does not help because the variance keeps changing.

However, its **increments are stationary**:

$$
X_{t+h} - X_t
= \sum_{i=t+1}^{t+h} \varepsilon_i
\;\stackrel{d}{=}\;
\sum_{i=1}^{h} \varepsilon_i.
$$

Thus a random walk is a **non-stationary process with stationary increments**.

</div>

### Asymptotic stationarity

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Convergence to equilibrium)</span></p>

A process can be non-stationary initially but approach a stationary regime. For example,

$$
X_t = \phi X_{t-1} + \varepsilon_t,
\qquad \|\phi\| < 1,
$$

started from a fixed $X\_0 = x$, is not generally stationary at finite times because the distribution still remembers $x$:

$$
X_t = \phi^t x + \sum_{i=1}^t \phi^{t-i} \varepsilon_i.
$$

But since $\phi^t x \to 0$, its distribution converges to the unique stationary distribution. This is often called **asymptotic stationarity**, convergence to stationarity, or convergence to equilibrium.

If instead $X\_0$ is already sampled from the invariant distribution, then the process is stationary from time $0$.

</div>

### Partial forms of stationarity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Taxonomy</span><span class="math-callout__name">(Weaker and modified notions)</span></p>

Several weaker or modified notions are useful:

* **Mean stationarity:** only $\mathbb{E}[X\_t]$ is constant.
* **Variance stationarity:** the variance is constant, but other properties may change.
* **Weak stationarity:** mean and autocovariance are time-invariant.
* **Stationary increments:** $X\_{t+h} - X\_t$ depends in distribution only on $h$.
* **Trend stationarity:** $X\_t = m(t) + Y\_t$, where $Y\_t$ is stationary; removing the deterministic trend gives stationarity.
* **Difference stationarity:** differences such as $X\_t - X\_{t-1}$ are stationary, as for a random walk.
* **Local stationarity:** over short time windows the process is approximately stationary, although its parameters drift globally.
* **Cyclostationarity:** statistical properties repeat periodically rather than remaining constant.

So stationarity is not all-or-nothing in practice; one must specify exactly **which distributions or moments are invariant and under what transformations**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What "stationary bandit" means in the RL notes)</span></p>

For the RL notes, "stationary bandit" usually means something simpler: for each action $a$, the conditional reward law

$$
R_t \mid A_t = a
$$

does not depend on $t$. A drifting action value violates that condition even if the drift itself is generated by some stationary latent process.

</div>

## The unit root phenomenon

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Key idea</span><span class="math-callout__name">(Unit root)</span></p>

A unit root means that an autoregressive process retains shocks indefinitely instead of pulling back toward a stable level. A unit-root series behaves like a random walk: shocks leave a permanent imprint on the long-term path rather than fading away.

Core characteristics:

* **No mean reversion:** the series trends without being pulled back to a level.
* **Permanent shocks:** unexpected events permanently alter the long-term path.
* **Long memory:** past shocks influence the future indefinitely.
* **Growing variance:** total variance increases with time, making long-horizon prediction increasingly difficult.

</div>

### The AR(1) picture

Consider the AR(1) model

$$
X_t = c + \phi X_{t-1} + \varepsilon_t,
$$

with white-noise innovations $\varepsilon\_t$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Stable case, $\|\phi\| < 1$)</span></p>

Iterating gives

$$
X_t
= \phi^t X_0
+ c \sum_{j=0}^{t-1} \phi^j
+ \sum_{i=1}^t \phi^{t-i} \varepsilon_i.
$$

Because $\phi^t \to 0$, both the initial condition and old shocks are forgotten geometrically. The process converges toward a stationary distribution with mean

$$
\mu = \frac{c}{1 - \phi}.
$$

A shock $\varepsilon\_t$ influences $X\_{t+k}$ by the factor $\phi^k$, which tends to zero.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Unit-root case, $\phi = 1$)</span></p>

Then

$$
X_t = c + X_{t-1} + \varepsilon_t,
\qquad\text{so}\qquad
X_t = X_0 + ct + \sum_{i=1}^t \varepsilon_i.
$$

This is a random walk, possibly with drift $c$. A shock does not decay:

$$
\frac{\partial X_{t+k}}{\partial \varepsilon_t} = 1
\qquad\text{for every } k \ge 0.
$$

Its effect is permanent. If $\operatorname{Var}(\varepsilon\_t) = \sigma^2$, then

$$
\operatorname{Var}(X_t) = t\sigma^2,
$$

so the distribution spreads indefinitely. Hence $X\_t$ is non-stationary.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Where the name comes from)</span></p>

Write the model in lag-polynomial form,

$$
(1 - \phi L) X_t = c + \varepsilon_t,
\qquad L X_t = X_{t-1}.
$$

For $\phi = 1$, the polynomial $1 - z$ has the root $z = 1$, which lies **on the unit circle** — hence "unit root."

For a general AR($p$) process,

$$
X_t = \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \varepsilon_t,
$$

we examine the characteristic polynomial

$$
1 - \phi_1 z - \cdots - \phi_p z^p.
$$

A unit root occurs when this polynomial has a root on the unit circle, particularly $z = 1$. A root at $1$ means the polynomial contains a factor $1 - z$, corresponding to a difference operator $1 - L$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Differencing removes the unit root)</span></p>

For the random walk,

$$
X_t - X_{t-1} = c + \varepsilon_t.
$$

Thus although $X\_t$ is non-stationary, its first difference $\Delta X\_t$ is stationary. Such a process is called **integrated of order one**, written $I(1)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary of the "phenomenon", and near-unit roots)</span></p>

The unit root *phenomenon* is the combination of:

* no mean reversion;
* permanent effects of shocks;
* increasing or otherwise nonconstant variance;
* strong persistence;
* non-stationarity that can often be removed by differencing.

A **near-unit-root** process, such as $\phi = 0.99$, is technically stationary but forgets shocks extremely slowly. Over a finite sample it can look almost indistinguishable from a random walk.

</div>

### Unit root processes vs. random walks

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Not every unit root process is a random walk)</span></p>

Every random walk is a unit root process, but not conversely — the random walk is the most basic, minimalist member of the family.

* **Random walk:** the current value depends only on the immediate past value plus white noise. There is no memory of older shocks and no short-run dynamics.
* **Unit root process:** a much broader category — any series that becomes stationary after differencing. It can carry complex short-term dynamics, cycles, and delayed reactions to past shocks (e.g. general ARIMA models).

| Process type | Equation | Behavior |
|---|---|---|
| Pure random walk | $Y\_t = Y\_{t-1} + \epsilon\_t$ | Changes depend only on the immediate past value and the current shock. |
| Random walk with drift | $Y\_t = \alpha + Y\_{t-1} + \epsilon\_t$ | A unit root process that climbs or falls along a deterministic trend $\alpha$. |
| General unit root (e.g. ARIMA) | $Y\_t = Y\_{t-1} + \theta \epsilon\_{t-1} + \epsilon\_t$ | Past shocks still ripple through the short-term dynamics. |

The distinction matters in modeling: a general unit root process can represent short-term cycles and momentum, whereas a pure random walk rules them out — beyond the last observation the future is entirely unpredictable.

</div>

### Unit roots in VAR(p) systems: the eigenvalue picture

A Vector Autoregression of order $p$ models $k$ variables jointly, each as a linear function of its own lags and the lags of all other variables:

$$
Y_t = \Phi_1 Y_{t-1} + \Phi_2 Y_{t-2} + \dots + \Phi_p Y_{t-p} + \epsilon_t.
$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Unit root $\iff$ eigenvalue $1$ of the companion matrix)</span></p>

Stack the current and past vectors, $\mathbf{Z}\_t = (Y\_t, Y\_{t-1}, \dots, Y\_{t-p+1})$, to rewrite the system as a first-order matrix equation:

$$
\mathbf{Z}_t = \mathbf{F} \mathbf{Z}_{t-1} + \mathbf{U}_t,
$$

where $\mathbf{F}$ is the $(kp \times kp)$ **companion matrix**

$$
\mathbf{F} = \begin{bmatrix} \Phi_1 & \Phi_2 & \dots & \Phi_{p-1} & \Phi_p \\ \mathbf{I}_k & \mathbf{0} & \dots & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{I}_k & \dots & \mathbf{0} & \mathbf{0} \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ \mathbf{0} & \mathbf{0} & \dots & \mathbf{I}_k & \mathbf{0} \end{bmatrix}.
$$

Stability and stationarity of the entire system depend on the eigenvalues of $\mathbf{F}$ — equivalently, on the roots of

$$
\det(\mathbf{I} - \Phi_1 z - \dots - \Phi_p z^p) = 0.
$$

Three scenarios:

* **All $\lvert\lambda\rvert < 1$ (inside the unit circle):** the system is stationary; shocks to any variable die out.
* **At least one $\lambda = 1$ (on the unit circle):** the system has a unit root; shocks have a permanent effect on the long-run path.
* **Any $\lvert\lambda\rvert > 1$ (outside the unit circle):** the system is explosive; values grow exponentially.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multiplicity and cointegration)</span></p>

In a multivariate VAR you can have multiple eigenvalues equal to $1$. The number of unit roots counts the independent **stochastic trends** driving the system:

* If the $k$ variables share **fewer than $k$** unit roots, they are **cointegrated**: they move together in the long run.
* If there are **exactly $k$** unit roots, every variable behaves like an independent random-walk-like process with no shared long-term equilibrium.

</div>

### Why the sample average fails under a unit root

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Breakdown of the LLN and of standard inference)</span></p>

In a stationary, ergodic time series the process is mean-reverting and its properties are stable over time, which guarantees that the sample average $\bar{y}$ converges to a single, fixed population mean as the sample grows. Under a unit root this mechanism breaks:

* **Stochastic trends:** shocks have a permanent, cumulative effect.
* **Time-varying moments:** the variance grows linearly in $t$; the series never settles around a value.
* **Non-ergodicity:** because the process drifts endlessly, one long historical sample cannot represent the range of potential outcomes. The Law of Large Numbers no longer applies, and $\bar{y}$ fails to yield a meaningful, consistent estimate of a population mean — indeed there is no constant mean to estimate.

Consequences for estimation:

* **AR($p$) with a unit root:** OLS estimators of the coefficients have non-standard limiting behavior, and the usual $t$-tests become invalid due to skewed, non-standard distributions.
* **VAR($p$) with unit roots:** applying standard estimation directly produces **spurious regression** — regressing two independent non-stationary series on each other routinely yields highly "significant" relationships where none exist, unless the variables are cointegrated.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Detecting and removing unit roots)</span></p>

**Detection:**

* Visual inspection: plotting the data and looking for strong trends that never revert.
* **Augmented Dickey–Fuller (ADF) test:** null hypothesis = a unit root is present.
* **Phillips–Perron (PP) test:** a modification robust to serial correlation and structural breaks.
* **KPSS test:** null hypothesis = stationarity (the reverse of ADF); useful as a cross-check.

**Remedies:**

* **Differencing:** work with $Y\_t - Y\_{t-1}$, converting $I(1)$ series to $I(0)$.
* **Detrending:** remove a deterministic time trend when the series is trend-stationary.
* **Cointegration modeling:** if several non-stationary series share a long-run relationship, use a **Vector Error Correction Model (VECM)**, which captures both short-term fluctuations and the long-term equilibrium without discarding level information.

</div>

## Are AR(p) and VAR(p) models examples where the sample average fails?

This question deserves care, because the naive answer ("yes, AR/VAR are classic non-stationarity examples") is wrong as stated.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Answer</span><span class="math-callout__name">(Only with a unit root or time variation — not in general)</span></p>

**Not AR($p$) and VAR($p$) in general.** They can be stationary or non-stationary. The key distinctions:

* **Dependence over time does not imply non-stationarity.**
* A stable AR($p$) or VAR($p$) process can be stationary and ergodic, so its sample mean still converges despite correlated observations.
* They become good non-stationary examples exactly when they contain a **unit root**, a deterministic trend, time-varying coefficients, or another mechanism causing the mean or distribution to drift.

For example, the AR(1) process

$$
X_t = X_{t-1} + \varepsilon_t
$$

is a random walk and is non-stationary, while

$$
X_t = 0.5\, X_{t-1} + \varepsilon_t
$$

has a stationary solution whose sample mean converges under standard assumptions. A time-varying model such as

$$
X_t = \phi_t X_{t-1} + \varepsilon_t
$$

can also represent changing dynamics.

**So: a unit-root or time-varying AR/VAR model is a valid example; "AR/VAR" alone is not.**

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dependence on the past is not non-stationarity)</span></p>

For

$$
X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \varepsilon_t,
$$

the past affects $X\_t$, so observations are temporally correlated. Nevertheless the process can have a time-invariant distribution: in the stable case the mean and variance do not change with $t$, and the effect of the distant past decays geometrically. By contrast, in the random walk shocks accumulate permanently and the variance grows with $t$. The clean separation is:

$$
\boxed{\text{autoregression} = \text{temporal dependence}}
$$

whereas

$$
\boxed{\text{non-stationarity} = \text{the probabilistic law or target changes over time}.}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Two different means: conditional vs. unconditional)</span></p>

Doesn't the changing distribution of $X\_t$ carry information that the mean has moved? Yes — but one must distinguish **two different means**.

For an AR($p$) process, the **conditional distribution** given the past changes with time:

$$
X_t \mid X_{t-1}, \dots, X_{t-p}
$$

has conditional mean

$$
\mathbb{E}[X_t \mid X_{t-1}, \dots, X_{t-p}]
= a_0 + \sum_{i=1}^p a_i X_{t-i}.
$$

Since the observed past values change, this conditional mean changes — the past indeed carries information about the **current target**.

However, for a *stationary* AR process, the **unconditional mean**

$$
\mathbb{E}[X_t] = \mu
$$

remains constant over time, and the sample average estimates this long-run $\mu$, not the changing conditional mean:

$$
\boxed{\text{stationary AR: constant marginal mean, changing conditional prediction.}}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The RL-specific subtlety: "non-stationary" used loosely)</span></p>

In the bandit setting, the most direct analogue is to let an arm's true value evolve as an AR process. With a unit root,

$$
q_{t+1}(a) = q_t(a) + \varepsilon_t,
$$

the true action value continuously wanders and never returns toward a fixed long-run mean; old rewards become increasingly irrelevant, and the full-history sample average grows increasingly stale. This is exactly the drifting target for which the notes recommend constant step sizes.

But there is a subtlety: if the arm's value evolves as

$$
q_{t+1}(a) = \phi\, q_t(a) + \varepsilon_t,
\qquad \|\phi\| < 1,
$$

then $(q\_t(a))$ is statistically **stationary**, yet the **current value** $q\_t(a)$ still moves. A full-history sample average estimates its long-run mean, not its current value, so a constant step-size estimator may track it better. Thus RL often uses "non-stationary bandit" somewhat loosely to mean "the action value changes over time" — a **time-varying target in the RL sense**, which need not be non-stationary in the strict time-series sense.

</div>

## Gradient descent as tracking a moving target

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Constant-$\alpha$ value update = SGD on the instantaneous squared loss)</span></p>

Suppose we estimate a scalar value $q\_t$ from an observed reward $R\_t$ using the instantaneous squared loss

$$
\ell_t(Q) = \tfrac12 (R_t - Q)^2,
\qquad
\nabla_Q \ell_t(Q) = Q - R_t.
$$

One gradient-descent step gives

$$
Q_{t+1}
= Q_t - \alpha (Q_t - R_t)
= Q_t + \alpha (R_t - Q_t),
$$

which is exactly the constant step-size value update from the notes.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Chasing a moving minimizer)</span></p>

In a stationary environment, $q\_t = q$ is fixed, and decreasing step sizes can make $Q\_t$ converge to $q$. In a non-stationary environment, the expected loss changes with time:

$$
L_t(Q) = \tfrac12 \mathbb{E}[(R_t - Q)^2],
\qquad
\arg\min_Q L_t(Q) = \mathbb{E}[R_t] = q_t.
$$

Gradient descent is then chasing the moving minimizer $q\_t$, and a constant step size prevents the estimator from becoming too rigid:

$$
\boxed{\text{constant-step SGD} \;\approx\; \text{noisy tracking of a moving optimum}.}
$$

Equivalently, SGD with a constant learning rate behaves like an **exponentially weighted moving average (EWMA)**: it heavily weights recent gradients, exponentially forgets older data, and estimates the *current local state* rather than the historical average. In a stationary environment one decays $\alpha \to 0$ to converge to a point; in a non-stationary one, $\alpha$ is kept bounded away from zero to keep tracking.

This interpretation generalizes beyond scalar values: online gradient descent on time-varying objectives $L\_t(\theta)$ attempts to track the sequence of minimizers $\theta\_t^\ast$. Not every use of gradient descent is value estimation, however — for example, gradient bandits optimize policy preferences rather than directly estimating action values.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The tracking trade-off: bias vs. variance)</span></p>

The learning rate $\alpha$ governs a fundamental trade-off for tracking:

* **High $\alpha$ — fast tracking, high variance:** the estimator reacts quickly to environmental shifts, but overreacts to noise, so the estimate chatters around the true value.
* **Low $\alpha$ — slow tracking, high bias:** the estimate is smooth and filters out noise, but it lags behind — a *tracking bias* — and fails to keep up if the environment shifts rapidly.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to the Kalman filter)</span></p>

In engineering and control theory, the canonical tool for estimating a hidden, drifting state is the **Kalman filter**. Constant-gain gradient descent is structurally a simplified Kalman filter: both update the current estimate by adding a correction equal to a gain factor (the learning rate / the Kalman gain) times a prediction error (the negative gradient / the innovation). The Kalman filter chooses this gain *optimally* from the model's noise statistics — see the adaptive step-size section below.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reinforcement learning context: TD learning)</span></p>

This viewpoint is highly visible in temporal-difference learning. The TD(0) update

$$
V(S) \leftarrow V(S) + \alpha \left[ R + \gamma V(S') - V(S) \right]
$$

has exactly the tracking form: a step, scaled by $\alpha$, in the direction of the prediction error. (Precisely speaking it is a **semi-gradient** step on the squared TD error — the target $R + \gamma V(S')$ is treated as fixed when differentiating.) Because the effective target keeps changing — through bootstrapping, through policy improvement, and possibly through the environment itself — $\alpha$ is typically not decayed to zero, so the value estimator continuously adapts.

</div>

## Can every SGD be seen as mean estimation of a non-stationary process?

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Answer</span><span class="math-callout__name">(Yes algebraically — but the process is endogenous)</span></p>

Algebraically, any SGD update can be written as an adaptive mean-estimation update, but the resulting "observations" are usually endogenous rather than samples from an external process.

Let

$$
\theta_{t+1} = \theta_t - \alpha_t g_t,
\qquad
g_t = g(\theta_t, \xi_t),
\qquad
\mathbb{E}[g_t \mid \theta_t] = \nabla L(\theta_t).
$$

Define the **pseudo-observation**

$$
Y_t := \theta_t - g_t.
$$

Then

$$
\theta_{t+1}
= (1 - \alpha_t)\theta_t + \alpha_t Y_t
= \theta_t + \alpha_t (Y_t - \theta_t),
$$

which has exactly the form of the incremental mean/value-estimation update from the notes.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The estimated mean is state-dependent)</span></p>

Conditionally on $\theta\_t$,

$$
\mathbb{E}[Y_t \mid \theta_t]
= \theta_t - \nabla L(\theta_t).
$$

Thus SGD is tracking a moving conditional mean

$$
m(\theta_t) := \theta_t - \nabla L(\theta_t).
$$

Because $\theta\_t$ changes, the distribution of $Y\_t$ and its conditional mean change as well. The desired solution is a fixed point:

$$
m(\theta^\ast) = \theta^\ast
\iff
\nabla L(\theta^\ast) = 0.
$$

So SGD can be interpreted as repeatedly estimating a mean whose distribution is altered by the current estimate itself.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(But this is not ordinary mean estimation)</span></p>

In standard mean estimation, observations $Y\_t$ come from an external distribution with a fixed or drifting mean. In SGD,

$$
Y_t = \theta_t - g(\theta_t, \xi_t)
$$

depends on $\theta\_t$: the algorithm influences the process it is observing. This is an **endogenous non-stationary process**. That is why the more natural general description of SGD is **stochastic approximation**, or stochastic root finding:

$$
\text{find } \theta^\ast
\quad\text{such that}\quad
\mathbb{E}[g(\theta^\ast, \xi)] = 0.
$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(For quadratic losses the interpretation is literal)</span></p>

Suppose

$$
L(\theta) = \tfrac12\, \mathbb{E}\,\lVert \theta - X \rVert^2.
$$

Then $g\_t = \theta\_t - X\_t$, and SGD gives

$$
\theta_{t+1}
= \theta_t - \alpha_t(\theta_t - X_t)
= \theta_t + \alpha_t (X_t - \theta_t).
$$

This is exactly online estimation of $\mathbb{E}[X]$. For a general loss the same algebra works, but $Y\_t$ is an artificial pseudo-observation rather than raw data whose mean has an independent statistical meaning.

So the accurate statement is:

$$
\boxed{
\text{Every SGD can be represented as mean estimation of an endogenous, generally non-stationary pseudo-process.}
}
$$

The deeper structure is not mean estimation itself, but noisy iteration toward a fixed point.

</div>

## Fixed mean: bias–variance decomposition and kernel mean embeddings

Suppose now the mean *is* fixed. Can we split the objective into "mean plus a positive function of the current state" and view SGD as estimating a distribution? Almost — the precise statement is the **bias–variance decomposition**, and in an RKHS it becomes exactly **kernel mean estimation**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Bias–variance decomposition of the squared loss)</span></p>

Assume $X$ takes values in a Hilbert space, with fixed mean $\mu = \mathbb{E}[X]$, and let $q$ be our current estimate. Consider

$$
L(q) = \tfrac12\, \mathbb{E}\,\lVert X - q \rVert^2.
$$

Writing $X - q = (X - \mu) + (\mu - q)$, the cross term vanishes because $\mathbb{E}[X - \mu] = 0$, giving

$$
\mathbb{E}\,\lVert X - q \rVert^2
= \underbrace{\mathbb{E}\,\lVert X - \mu \rVert^2}_{\text{variance; constant in } q}
+ \underbrace{\lVert q - \mu \rVert^2}_{\text{nonnegative estimation error}}.
$$

So the correct phrasing is

$$
\boxed{\text{objective} = \text{constant variance} + \text{squared distance to the mean},}
$$

not "mean plus a positive function": the mean is the *location of the minimizer*, and the constant term is the variance. Consequently

$$
\nabla L(q) = q - \mu,
$$

and replacing the unknown $\mu$ by a single sample $X\_t$ gives the stochastic gradient $q\_t - X\_t$, so SGD becomes

$$
q_{t+1}
= q_t - \alpha_t (q_t - X_t)
= q_t + \alpha_t (X_t - q_t)
$$

— precisely the incremental mean-estimation update from the RL notes.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(SGD on squared RKHS loss = online kernel mean estimation)</span></p>

Map the observations into an RKHS $\mathcal{H}$ through the feature map

$$
\phi(x) = k(x, \cdot).
$$

The distribution $P$ has the **kernel mean embedding**

$$
\mu_P = \mathbb{E}_{X \sim P}[\phi(X)] \in \mathcal{H}.
$$

Consider the objective

$$
J(m) = \tfrac12\, \mathbb{E}\,\lVert \phi(X) - m \rVert_{\mathcal H}^2.
$$

The same decomposition gives

$$
J(m)
= \tfrac12\, \mathbb{E}\,\lVert \phi(X) - \mu_P \rVert_{\mathcal H}^2
+ \tfrac12\, \lVert m - \mu_P \rVert_{\mathcal H}^2,
$$

so the minimizer is $m^\ast = \mu\_P$. A stochastic-gradient step is

$$
m_{t+1}
= m_t + \alpha_t \bigl( \phi(X_t) - m_t \bigr),
$$

and with $\alpha\_t = 1/t$,

$$
m_t = \frac1t \sum_{i=1}^t k(X_i, \cdot),
$$

the empirical kernel mean embedding. Hence

$$
\boxed{
\text{SGD on squared RKHS loss}
=
\text{online estimation of a distribution's kernel mean embedding}.
}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Qualifications)</span></p>

* $\mu\_P$ is a *representation* of the distribution, not generally a probability density. For a **characteristic kernel**, however, the map $P \mapsto \mu\_P$ is injective, so the embedding uniquely determines the distribution.
* This is not a universal interpretation of every SGD problem. Any objective can formally be written as

  $$
  L(q) = L(q^\ast) + \bigl( L(q) - L(q^\ast) \bigr),
  $$

  where the second term is nonnegative, but only special losses — most cleanly the squared Hilbert-space loss — turn that excess term into a variance-like distance and make SGD literal mean estimation.

</div>

## Adaptive step sizes

What if $\alpha$ is not fixed, but adaptive in time — some function of the previous values? Then the update becomes

$$
Q_{t+1} = Q_t + \alpha_t (R_t - Q_t),
\qquad
\alpha_t = f(R_{t-1}, R_{t-2}, \dots),
$$

and should be viewed as an **adaptive weighted estimator**. Depending on how the function of the past is constructed, one arrives at three major frameworks: statistical filtering (Kalman), adaptive machine-learning optimizers (AdaGrad/RMSProp/Adam), and meta-learning (hypergradients).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(General unrolled weights)</span></p>

The step size determines how quickly the estimator forgets the past. Unrolling the recursion gives

$$
Q_{n+1}
=
\left( \prod_{j=1}^{n} (1 - \alpha_j) \right) Q_1
+
\sum_{i=1}^{n}
\alpha_i
\left( \prod_{j=i+1}^{n} (1 - \alpha_j) \right) R_i.
$$

Each reward $R\_i$ receives a weight determined by all *subsequent* step sizes. If $\alpha\_t$ depends on previous observations, these weights are themselves random and data-dependent.

Important special cases:

* $\alpha\_t = 1/t$ — the ordinary sample average;
* $\alpha\_t = \alpha$ constant — exponential recency weighting;
* an adaptive rule $\alpha\_t = f(\text{recent prediction errors}, \text{estimated volatility}, \text{uncertainty})$ — increase the learning rate when the environment appears to change, decrease it when the estimate appears stable. For instance, use a larger $\alpha\_t$ when recent errors $\|R\_t - Q\_t\|$ are persistently large, interpreting this as evidence that the target has moved.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stationary vs. moving target: the fundamental distinction)</span></p>

* For a **stationary target**, one often chooses $\alpha\_t \to 0$. The standard stochastic-approximation (Robbins–Monro) conditions are

  $$
  \sum_t \alpha_t = \infty,
  \qquad
  \sum_t \alpha_t^2 < \infty,
  $$

  which allow continued learning while suppressing asymptotic noise.

* For a **moving target**, letting $\alpha\_t \to 0$ completely is dangerous: eventually the estimator becomes unable to react. One usually maintains a positive lower bound, or increases $\alpha\_t$ after detecting change.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A subtlety: the step size should be predictable)</span></p>

It is mathematically cleaner for $\alpha\_t$ to depend only on information available **before** observing $R\_t$. If it depends directly on the current noise realization, it may introduce bias rather than merely adapting the memory length.

</div>

### The statistically optimal choice: the Kalman gain

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Principle</span><span class="math-callout__name">(Kalman filter as the optimally adaptive step size)</span></p>

In a linear Gaussian state-space model — the true parameter drifts, $\theta\_t = \theta\_{t-1} + w\_t$, and observations are noisy, $y\_t = \theta\_t + v\_t$ — the optimal step size is the **Kalman gain** $K\_t$:

$$
Q_{t+1} = Q_t + K_t (R_t - Q_t).
$$

* **How it is computed:** from all previous steps the filter tracks two quantities — the uncertainty of the current estimate, and the noise level of the environment. $K\_t$ is large when the estimator is uncertain and small when it is confident.
* **Behavior:** if the environment is stable, $K\_t$ shrinks to filter out noise. If the environment shifts — signaled by a run of unexpectedly large prediction errors — $K\_t$ rises again to rapidly track the new regime.

So the Kalman filter is an *optimally adaptive* instance of the same RL update template, and constant-gain gradient descent is its simplified, fixed-gain version.

</div>

### The machine-learning approach: AdaGrad, RMSProp, momentum, Adam

These optimizers make the step size a function of historical *gradients*. Instead of a single scalar they compute a **vector** of step sizes, tailored to each individual parameter. Start from SGD:

$$
\theta_{t+1} = \theta_t - \eta\, g_t,
\qquad
g_t \approx \nabla L(\theta_t).
$$

Even when $L$ is fixed, the distribution of $g\_t$ changes because (i) $\theta\_t$ moves through different regions of the loss landscape; (ii) different mini-batches produce different noise; (iii) curvature and gradient scale differ across coordinates.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(AdaGrad — a function of *all* past gradients)</span></p>

AdaGrad divides $\eta$ by the square root of the sum of squares of **all** historical gradients, coordinatewise.

**Failure mode in non-stationarity:** because it accumulates the entire history, the effective step size strictly decays to zero. If the environment changes late in training, the optimizer has become "blind" and cannot adapt — exactly the $\alpha\_t \to 0$ danger from the moving-target remark above.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Momentum — tracks the recent gradient direction)</span></p>

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t,
\qquad
\theta_{t+1} = \theta_t - \eta\, m_t.
$$

Unrolling,

$$
m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{\,t-i} g_i,
$$

so momentum is an exponentially recency-weighted estimate of the gradient's local mean direction — the same structure as the constant-$\alpha$ value estimate, applied to the gradient signal. It suppresses rapidly fluctuating noise and preserves directions that stay consistent over several iterations.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(RMSProp — tracks the recent gradient scale)</span></p>

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
\qquad
\theta_{t+1} = \theta_t - \eta\, \frac{g_t}{\sqrt{v_t} + \varepsilon}
\quad\text{(coordinatewise)}.
$$

Here $v\_t$ estimates the recent second moment of each gradient coordinate — an exponentially decaying average that fixes AdaGrad's flaw by limiting the memory to a recent effective window. Coordinates with persistently large gradients receive smaller effective steps; coordinates with small gradients receive larger relative steps. If a coordinate's gradients suddenly become large and volatile, its local step size shrinks, preventing exploding updates.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Adam — tracks both)</span></p>

Adam combines

$$
m_t \approx \text{recent gradient mean},
\qquad
v_t \approx \text{recent gradient second moment},
$$

and updates

$$
\theta_{t+1}
= \theta_t
- \eta\,
\frac{\widehat m_t}{\sqrt{\widehat v_t} + \varepsilon},
$$

(with bias-corrected $\widehat m\_t, \widehat v\_t$). Its effective learning rate is therefore coordinate- and time-dependent:

$$
\alpha_{t,j}^{\mathrm{eff}}
=
\frac{\eta}{\sqrt{\widehat v_{t,j}} + \varepsilon}.
$$

This is genuinely an adaptive reweighting mechanism.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two different senses of non-stationarity)</span></p>

It is tempting to say Adam-like methods are "designed for non-stationarity," but the claim needs a distinction:

* **Optimization-induced non-stationarity:** the gradient distribution changes because $\theta\_t$ changes (and mini-batch noise varies). Adam and RMSProp are primarily aimed at *this*.
* **Environment-induced non-stationarity:** the objective itself changes, $L\_t(\theta) \neq L\_{t+1}(\theta)$. Adam or momentum may help track the moving optimum, but they were not designed as optimal non-stationary estimators — their memory can even *hurt* after abrupt regime changes, because stale gradient moments persist.

From an SDE/dynamical-systems viewpoint: SGD corresponds approximately to noisy first-order dynamics; momentum introduces a velocity state, giving second-order dynamics; RMSProp and Adam introduce additional state variables that estimate local gradient statistics and modify the geometry of motion. The accurate summary is

$$
\boxed{\text{Adam-like methods are adaptive filters of a non-stationary gradient signal.}}
$$

They are analogous to the adaptive step-size estimators above, but the "target" being tracked is the local gradient field and its scale, rather than directly an action value.

</div>

### The meta-learning approach: hypergradient descent

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Hypergradient descent)</span></p>

One can treat the learning rate itself as a parameter optimized by gradient descent. Using the last two gradients:

$$
\alpha_t = \alpha_{t-1} + \beta\, \langle g_t, g_{t-1} \rangle.
$$

If successive gradients point in the same direction (positive inner product), the optimizer is moving too slowly and $\alpha\_t$ grows; if they oscillate, $\alpha\_t$ shrinks.

**Behavior in non-stationarity:** during a sustained drift, the gradients consistently point toward the moving target; hypergradient descent senses this, automatically increases $\alpha\_t$ to speed up tracking, and shrinks $\alpha\_t$ once the target stops moving.

</div>

### Summary of adaptive step-size behaviors

| Approach | Function memory | Best used for | Behavior in non-stationarity |
|---|---|---|---|
| Fixed $\alpha$ | None | Simple tracking | Constant lag, constant chatter |
| Sample average $\alpha\_t = 1/t$ | All rewards, equally | Stationary targets | Goes blind: cannot react to change |
| Kalman gain $K\_t$ | All past errors and variances | Linear-Gaussian systems with noise | Statistically optimal tracking; gain spikes on regime shifts |
| RMSProp / Adam | Recent gradient moments (exponential window) | Neural networks / deep RL | Per-parameter step scaling; stale moments can lag after abrupt breaks |
| Hypergradient | Last two gradients | General optimization | Grows the step during long drifts, shrinks it at rest |

## Attention as adaptive weighting

Can the attention mechanism be seen as the ultimate adaptive step size? Largely yes — attention is a strictly more general adaptive *weighting* mechanism — but it is worth being precise about what it generalizes.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Attention assigns per-observation, data-dependent weights)</span></p>

In the recursive estimator

$$
Q_{t+1} = (1 - \alpha_t) Q_t + \alpha_t R_t,
$$

the scalar $\alpha\_t$ decides how much weight to give to the *entire compressed past* $Q\_t$ versus the new observation $R\_t$. Attention instead assigns **separate, data-dependent weights to many previous observations**:

$$
\widehat q_t = \sum_{i<t} w_{t,i} R_i,
\qquad
w_{t,i}
=
\frac{\exp\!\left( q_t^\top k_i / \sqrt{d_k} \right)}
{\sum_{j<t} \exp\!\left( q_t^\top k_j / \sqrt{d_k} \right)},
$$

where the weight is a function of the interaction between a **query** $q\_t$ (what we are looking for now) and a **key** $k\_i$ (what the historical observation contains). Attention can decide that some old observation is highly relevant while a more recent one is not: it behaves like a **learned, context-dependent averaging kernel**.

This yields a hierarchy of estimators:

$$
\text{sample average}
\;\subset\;
\text{exponential moving average}
\;\subset\;
\text{adaptive scalar step size}
\;\subset\;
\text{attention-style adaptive weighting}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this helps with non-stationary sequences)</span></p>

Text, audio, and financial series are highly non-stationary: the relevant context changes rapidly. An observation from 50 steps ago may suddenly become the most relevant piece of information, while the most recent observation may be noise.

* **Fixed-$\alpha$ failure:** any exponentially decaying average *must* forget the observation from 50 steps ago at a prescribed rate.
* **Attention:** because the weight $w\_{t,i}$ is computed on the fly from content, the model can assign nearly all weight to a critical past event and nearly none to recent noise — dynamically re-opening access to any point in the past.

For tracking a non-stationary target, attention could learn to emphasize recent observations, observations from the same regime, observations with similar context, or observations indicating a change point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Caveats: attention is not drift detection; gates are closer to $\alpha\_t$)</span></p>

Two qualifications keep the analogy honest:

1. **Attention does not automatically solve non-stationarity.** Its weights are based on *learned similarity*, not on statistical uncertainty or drift detection. Nothing forces the learned kernel to behave like a Kalman gain.
2. **A learned recurrent gate is actually closer to the adaptive step-size equation.** A GRU-style update

   $$
   Q_{t+1} = (1 - z_t) Q_t + z_t \widetilde Q_t
   $$

   has the update gate $z\_t$ playing almost exactly the role of $\alpha\_t$: one scalar (per unit) arbitrating between the compressed past and the new information. Hence

   $$
   \boxed{\text{attention generalizes adaptive memory weights, while a recurrent gate more directly generalizes } \alpha_t.}
   $$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fast weights, and what adapts along which axis)</span></p>

In the meta-learning literature, attention is often described as a system of **fast weights**: the ordinary network weights ("slow weights") are updated by gradient descent across training, while the attention matrix acts as a set of instantly reconfigured weights, recomputed for every input, that route information based on context. (Linear attention makes this concrete: it reduces to a recurrent, gradient-descent-like update of an associative fast-weight matrix.)

The three adaptive mechanisms in these notes then differ mainly in *which axis* they adapt along:

| Mechanism | What determines the "step size"? | Axis of adaptation |
|---|---|---|
| Adam / RMSProp | Recent history of training gradients | Training time (updates change across steps/epochs) |
| Kalman filter | Prediction errors and modeled uncertainties | Chronological time (gain changes as time flows) |
| Attention | Contextual similarity of current state to past states | Sequence/context space (weights change with data content) |

</div>

## Excursion: unit roots in macroeconomics

The unit root is not only a technical nuisance — its empirical discovery reshaped macroeconomic theory.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Background</span><span class="math-callout__name">(Nelson–Plosser, 1982)</span></p>

Historically, economists assumed the economy followed a smooth trend path determined by demographics and technology; recessions were temporary deviations that would fade as the economy returned to trend. In 1982, Charles Nelson and Charles Plosser published a seminal study arguing that most macroeconomic aggregates (real GDP, employment, wages) contain a **unit root** — implying that economic shocks are *permanent*, not temporary.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Where unit roots appear in economic models)</span></p>

1. **Real Business Cycle (RBC) models** (Kydland–Prescott). Business cycles are driven by technology (total-factor-productivity) shocks, modeled as a **random walk with drift** — a unit root process. A technological breakthrough or a supply shock permanently shifts the economy's path: it does not bounce back to the old trend but starts tracking a new one.

2. **Labor economics and hysteresis.** Traditional models assume unemployment reverts to a "natural rate" after a recession. Empirically, unemployment in many economies (notably late-20th-century Europe) fails unit-root tests for stationarity. This motivated **hysteresis** models in which a severe recession permanently scars workers — skills erode, workers exit the labor force — and thereby permanently raises baseline unemployment.

3. **The Permanent Income Hypothesis (consumption).** Friedman's hypothesis says consumers adjust spending only in response to *permanent* income changes. Campbell and Mankiw used unit-root econometrics to argue that since GDP has a unit root, a current drop in income signals persistently lower future income — so forward-looking consumers cut lifetime spending immediately when a recession hits, recognizing the shock as permanent.

4. **DSGE models.** Modern central-bank (New Keynesian DSGE) models explicitly include integrated $I(1)$ variables to capture non-stationary realities — trend productivity, population growth, inflation objectives — and are then transformed (detrended/stationarized) so that policy experiments can be simulated around a stationary state.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Impact on the worldview)</span></p>

| | Pre-unit-root era (trend-stationary) | Post-unit-root era (difference-stationary) |
|---|---|---|
| View of GDP | Fluctuates temporarily around a fixed deterministic trend line | Driven by a stochastic trend; shocks permanently alter the trajectory |
| Recession impact | A bad dream; the economy bounces back to normal | A permanent loss of output that is never fully recovered |
| Policy focus | Smooth out short-term demand fluctuations | Long-term supply factors, productivity, structural growth |

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Which macroeconomic variables do *not* have a unit root?)</span></p>

While aggregate levels (GDP, price indices, money supply) almost always contain a unit root, variables representing **rates, ratios, returns, or spreads** are typically stationary: they are bounded by economic behavior, institutional ceilings, or arithmetic, so shocks eventually fade and the series reverts to a long-run mean.

1. **Growth rates and percentage changes.** The *level* of CPI has a unit root, but the inflation rate is generally stationary over long periods (though possibly very persistent, depending on the monetary regime). Likewise real GDP climbs indefinitely, but its growth rate reverts to a steady-state average.

2. **Financial spreads and real returns.** A spread is a difference of two variables sharing a stochastic trend, so the non-stationarity cancels. The yield-curve spread (e.g. 10-year minus 2-year yields) is highly stationary; the ex-post real interest rate (nominal rate minus inflation) tends to be stationary even when its two components individually mimic unit-root processes.

3. **Bounded ratios.** Capacity utilization, the household savings rate, and the trade-balance-to-GDP ratio cannot drift to infinity; long-run market forces and physical constraints pull them back toward equilibrium ranges.

4. **Cyclical and survey indicators.** Diffusion indices like the PMI or consumer-confidence indices fluctuate around a fixed neutral baseline by construction; the output gap is stationary by definition, since the long-run trend has been stripped out.

**The notable exception: the unemployment rate.** Mathematically it is a bounded ratio and "should" be stationary, yet in practice — especially in Europe — standard tests often fail to reject a unit root. The resolution is hysteresis again: structural labor-market shocks can shift the baseline unemployment level for decades, making the bounded series behave, over any observed sample, like an integrated one.

</div>
