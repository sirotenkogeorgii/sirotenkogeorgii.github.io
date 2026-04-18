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
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Latent variable and observation in GPS)</span></p>

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

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

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

**Dynamics parameters $(A,\Sigma)$**

### Update for $A$
The dynamics contribution to $Q$ is

$$
Q_{\text{dyn}}(A,\Sigma)
= -\frac{T-1}{2}\log|\Sigma|
-\frac12 \sum_{t=2}^T \mathbb E_q\!\left[(z_t-Az_{t-1})^\top\Sigma^{-1}(z_t-Az_{t-1})\right]
+\text{const}.
$$

For fixed $\Sigma$, maximizing w.r.t. $A$ is equivalent to minimizing the expected quadratic form. 

Using $\mathbb E[u^\top M u]=\mathrm{tr}\\left(M\,\mathbb E[uu^\top]\right)$,

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

**Observation parameters $(B,\Gamma)$**

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

**Initial parameters $(\mu_0,\Sigma_0)$**
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
