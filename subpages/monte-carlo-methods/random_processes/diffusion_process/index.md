---
title: Diffusion Process
layout: default
noindex: true
---

# Diffusion Process

## Stochastic Differential Equations

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic differential equation (SDE))</span></p>

A **stochastic differential equation (SDE)** describes a stochastic process $\lbrace X_t, t\ge 0\rbrace$ via

$$dX_t = a(X_t,t)dt + b(X_t,t)dW_t \qquad (4.3)$$

* $\lbrace W_t\rbrace$ is a **Wiener process** (Brownian motion).
* $a(x,t)$: **drift** coefficient (systematic/average motion): determinisitc function
* $b(x,t)$: noise amplitude : determinisitc function
  * $b^2(x,t)$ is called the **diffusion coefficient** (some authors call $b$ itself the diffusion coefficient).
* The resulting $\lbrace X_t\rbrace$ is a **(Itô) diffusion process**: a Markov process with continuous sample paths.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(SDE = deterministic part + stochastic part)</span></p>

**Stochastic diﬀerential equations** are based on the same principle as ordinary diﬀerential equations, relating an unknown function to its derivatives, but with the additional feature that part of the unknown function is driven by randomness.

</div>

## Differential form vs integral form (the real meaning)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Integral form of SDE and Itô integral)</span></p>

The precise meaning of (4.3) is the **integral equation**

$$X_t = X_0 + \int_0^t a(X_s,s),ds + \int_0^t b(X_s,s),dW_s \qquad (4.4)$$

* The second integral is an **Itô integral**.
* Intuition: over a tiny time step,

  * deterministic change $\approx a(X_t,t)dt$
  * random change $\approx b(X_t,t)dW_t$ where $dW_t$ behaves like $\sqrt{dt}\times \mathcal{N}(0,1)$.

</div>

## Multidimensional SDEs

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multidimensional SDE)</span></p>

For $X_t \in \mathbb{R}^m$:

$$dX_t = a(X_t,t)dt + B(X_t,t)dW_t \qquad (4.5)$$

* $W_t$ is an **$n$-dimensional** Wiener process.
* $a(x,t)\in\mathbb{R}^m$: drift vector.
* $B(x,t)\in\mathbb{R}^{m\times n}$: diffusion matrix (noise mixing).
* Define the **diffusion matrix**
  
  $$C(x,t) = B(x,t)B(x,t)^\top\quad (\in \mathbb{R}^{m\times m})$$
  
* If $a$ and $B$ do **not** depend explicitly on $t$, the diffusion is **time-homogeneous**.

</div>

## Euler’s Method (Euler–Maruyama) for SDE simulation

### Idea: discretize time

Assume SDE

$$dX_t = a(X_t,t)dt + b(X_t,t)dW_t,\quad t\ge 0 \qquad (4.6)$$

Pick step size $h$, define $t_k = kh$, and approximate $X_{t_k}$ by $Y_k$.

Key approximation:

$$
W_{t_{k+1}} - W_{t_k} \sim \mathcal{N}(0,h)
\quad\Rightarrow\quad
\Delta W_k \approx \sqrt{h}Z_k, Z_k\sim\mathcal{N}(0,1)
$$

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Euler–Maruyama update (1D))</span></p>

$$Y_{k+1} = Y_k + a(Y_k,kh)h + b(Y_k,kh)\sqrt{h}Z_k \qquad (4.7)$$

* $\lbrace Z_k\rbrace$ i.i.d. $\mathcal{N}(0,1)$
* Interpretation: “ODE Euler step” + “random kick”.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Euler’s Method (1D))</span></p>

1. Sample $Y_0$ from the distribution of $X_0$. Set $k=0$.
2. Draw $Z_k \sim \mathcal{N}(0,1)$.
3. Compute $Y_{k+1}$ using 1D Euler–Maruyama update (4.7) as an approximation to $X_{(k+1)h}$.
4. Increase $k$ and repeat.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpolation between grid points)</span></p>

Euler gives values only at $t=kh$. For $t \in [kh,(k+1)h]$, a simple approximation is **linear interpolation**:

$$
X_t \approx
\left(k+1-\frac{t}{h}\right)Y_k
+
\left(\frac{t}{h}-k\right)Y_{k+1},
\quad t\in[kh,(k+1)h]
$$

(Or even cruder: use $Y_k$ throughout the interval.)

<figure>
  <img src="{{ 'assets/images/notes/random/InterpolationbetweenEulergridpointsforanSDE.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Interpolation between Euler grid points for an SDE (Ornstein–Uhlenbeck)</figcaption>
</figure>

</div>

## Multidimensional Euler–Maruyama

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Euler–Maruyama update (ND))</span></p>

Euler generalizes to:

$$Y_{k+1} = Y_k + a(Y_k,kh)h + B(Y_k,kh)\sqrt{h}Z_k$$

where now $Z_k \sim \mathcal{N}(0,I)$ is an $n$-dimensional standard normal vector.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Euler’s Method (ND))</span></p>

1. Sample $Y_0$ from distribution of $X_0$. Set $k=0$.
2. Draw $Z_k \sim \mathcal{N}(0,I)$.
3. Compute $Y_{k+1}$ using the multidimensional Euler–Maruyama update.
4. Increase $k$ and repeat.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Simplified Duffing–Van der Pol Oscillator (2D SDE))</span></p>

A two-dimensional system $(X_t,Y_t)$:

$$dX_t = Y_tdt$$

$$dY_t = \big(X_t(\alpha - X_t^2) - Y_t\big)dt + \sigma X_tdW_t$$

Notes:

* Noise enters the **$Y$** equation, scaled by $\sigma X_t$ (state-dependent noise).
* Mentioned simulation setup (in the text): $\alpha=1$, $\sigma=\tfrac12$, over $t\in[0,1000]$, step $h=10^{-3}$, start at $(-2,0)$.
* Qualitative behavior: the process “oscillates between two modes” (bistable/alternating behavior in trajectories).

<figure>
  <img src="{{ 'assets/images/notes/random/VanderPolOscillator.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Simplified Duﬃng–Van der Pol Oscillator</figcaption>
</figure>

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Quick mental model)</span></p>

* **Drift $a$**: where the system *wants* to go on average.
* **Diffusion $b$ / $B$**: how randomness pushes it around (strength + direction mixing).
* **Euler–Maruyama**: replace $dW_t$ by $\sqrt{h},Z$ each step.

</div>