---
title: The Principles of Diffusion Models
layout: default
noindex: true
---

# The Principles of Diffusion Models

## Deep Generative Modeling

**Goal of DGM:** 
* DNN to parameterize a model distribution $p_\phi(x)$, 
* $\phi$ represents the network’s trainable parameters.
* find 

$$p_{\phi^*}(x) = p_{\text{data}}(x)$$

**Capability of DGM**
1. Sampling from $p_\phi(x)$
2. Compute the probability (or likelihood) of any given data sample $x'$: $p_\phi(x')$.

**Training of DGM**
* learn parameters $ϕ$ of a model family $\lbrace p_\phi\rbrace$ 
* by minimizing a discrepancy $\mathcal{D}(p_{\text{data}},p_\phi)$:

$$\phi^*\in\arg\min_\phi \mathcal{D}(p_{\text{data}},p_\phi)$$

### Divergences

In statistics, divergence is a non-negative measure of the difference, dissimilarity, or distance between two probability distributions ($P$ and $Q$).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Divergence (statistics))</span></p>

Given a **differentiable manifold** $M$ of dimension  $n$, a **divergence** on $M$ is a $C^2$ $\mathcal{D}:M\times M\to [0,\infty )$ satisfying:
1. $\mathcal{D}(p,q)\geq 0$ for all $p,q\in M$ (non-negativity),
2. $\mathcal{D}(p,q)=0$ if and only if $p=q$ (positivity),
3. At every point $p\in M$, $D(p,p+dp)$ is a positive-definite **quadratic form** for infinitesimal displacements $dp$ from $p$.

In applications to statistics, the manifold $M$ is typically the space of parameters of a **parametric family of probability distributions**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Divergence could be view as a metric for probability measures)</span></p>

Informally, people sometimes describe divergences as measuring the "distance" between probability distributions. This risks confusion with formal distance metrics, which must satisfy some extra requirements. In addition to the requirements above, a distance metric must also be symmetric: 

$$\mathcal{D}(a,b)=\mathcal{D}(b,a)$$

And, it must satisfy the triangle inequality: 

$$\mathcal{D}(a,c)\leq \mathcal{D}(a,b) + \mathcal{D}(b,c)$$

As a side note, divergences are defined specifically on probability distributions, whereas distance metrics can be defined on other types of objects too.

All distance metrics between probability distributions are also divergences, but the converse is not true--a divergence may or may not be a distance metric. For example, the KL divergence is a divergence, but not a distance metric because it's not symmetric and doesn't obey the triangle inequality. In contrast, the Hellinger distance is both a divergence and a distance metric. To avoid confusion with formal distance metrics, I prefer to say that divergences measure the dissimilarity between distributions.

</div>


## Study notes: DGM setup, training divergences, and modeling challenges

### 1) Mathematical setup (Section 1.1.1)

**Data assumption**

* You observe a finite dataset of **i.i.d.** samples
  
  $$x^{(i)} \sim p_{\text{data}}(x), \quad i=1,\dots,N$$
  
  where $p_{\text{data}}$ is an **unknown**, complex distribution.

**Goal of a Deep Generative Model (DGM)**

* Learn a **tractable** model distribution $p_\phi(x)$ (parameterized by a neural network with parameters $\phi$) such that
  
  $$p_{\phi^*}(x) \approx p_{\text{data}}(x)$$
  
* Intuition: since $p_{\text{data}}$ is unknown (and you only have samples), you fit $p_\phi$ so it can act as a proxy for the data distribution.

**What “having a generative model” gives you**

* **Sampling:** generate arbitrarily many new samples (e.g., via Monte Carlo methods) from $p_\phi$.
* **Likelihood / density evaluation:** compute $p_\phi(x')$ (or $\log p_\phi(x')$) for a given point $x'$ *if the model family supports tractable evaluation*.

---

### 2) Training objective via a discrepancy / divergence

**General training principle**

* Choose a discrepancy measure $D(p_{\text{data}}, p_\phi)$ between distributions and solve
  
  $$\phi^* \in \arg\min_\phi D(p_{\text{data}}, p_\phi) \qquad\text{(1.1.1)}$$
  
* Since $p_{\text{data}}$ is not directly accessible, $D$ must be something you can **estimate from samples**.

**Figure intuition**

* You only see samples $x_i$ from $p_{\text{data}}$, and you tune $p_\phi$ to reduce the “gap” $D(p_{\text{data}}, p_\phi)$.

---

### 3) Forward KL divergence and Maximum Likelihood Estimation (MLE)

**Forward KL definition**

$$
D_{\mathrm{KL}}(p_{\text{data}}|p_\phi)
:=\int p_{\text{data}}(x)\log\frac{p_{\text{data}}(x)}{p_\phi(x)},dx
= \mathbb{E}_{x\sim p_{\text{data}}}\left[\log p_{\text{data}}(x)-\log p_\phi(x)\right]
$$

* **Asymmetric:**
  
  $$D_{\mathrm{KL}}(p_{\text{data}}\lvert p_\phi)\neq D_{\mathrm{KL}}(p_\phi\rvert p_{\text{data}})$$

#### Mode covering effect (important intuition)

* Minimizing **forward KL** encourages **mode covering**:

  * If there is a set (A) with positive data probability $p_{\text{data}}(A)>0$ but the model assigns zero density there ($p_\phi(x)=0$ for $x\in A$), then the integrand contains $\log(p_{\text{data}}(x)/0)=+\infty$ on $A$, hence the KL becomes infinite.
  * **Consequence:** forward KL strongly pressures the model to put probability mass wherever the data has support.

#### KL decomposition → MLE equivalence

Rewrite forward KL:

$$
\begin{aligned}
D_{\mathrm{KL}}(p_{\text{data}}|p_\phi)
&= \mathbb{E}_{x\sim p_{\text{data}}}\left[\log\frac{p_{\text{data}}(x)}{p_\phi(x)}\right] \
&= -\mathbb{E}_{x\sim p_{\text{data}}}\left[\log p_\phi(x)\right] + \underbrace{\left(-\mathbb{E}_{x\sim p_{\text{data}}}[\log p_{\text{data}}(x)]\right)}_{\mathcal{H}(p_{\text{data}})}
\end{aligned}
$$

- $\mathcal{H}(p_{\text{data}})$ is the **entropy** of the data distribution and does **not** depend on $\phi$.
- Therefore:

**Lemma (Minimizing KL $\iff$ MLE)**

$$
\min_\phi D_{\mathrm{KL}}(p_{\text{data}}|p_\phi)
\quad \Longleftrightarrow \quad
\max_\phi \mathbb{E}_{x\sim p_{\text{data}}}[\log p_\phi(x)] \qquad\text{(1.1.2)}
$$

#### Empirical MLE objective (what you actually optimize)

Replace the expectation with the sample average (Monte Carlo estimate):

$$\hat{\mathcal{L}}_{\mathrm{MLE}}(\phi) := -\frac{1}{N}\sum_{i=1}^N \log p_\phi(x^{(i)})$$

optimized with stochastic gradients / minibatches.
Key point: **you never need to evaluate** $p_{\text{data}}(x)$.

---

### 4) Fisher divergence (score discrepancy) and score matching

**Definition (Fisher divergence)**
For distributions $p$ and $q$:

$$D_F(p\mid q) := \mathbb{E}_{x\sim p}\left[\left\|\nabla_x \log p(x) - \nabla_x \log q(x)\right\|_2^2\right] \qquad \text{(1.1.3)}$$

**Core concept: the score**

* The **score function** of a density $p$ is
  
  $$s_p(x) := \nabla_x \log p(x)$$
  
* Fisher divergence measures how close the **vector fields** $s_p(x)$ and $s_q(x)$ are.

**Key property**

* It’s **invariant to normalization constants**, because gradients of log-densities ignore additive constants:
  * If $q(x)\propto \tilde q(x)$, then $\nabla_x \log q(x)=\nabla_x \log \tilde q(x)$.
* This makes it a natural basis for **score matching** and connects directly to **score-based / diffusion modeling**, where you train a model to match the data score field.

---

### 5) Beyond KL: other divergences

Different divergences encode different notions of “closeness” and can change learning behavior.

#### 5.1) $f$-divergences (Csiszár family)

A broad family:

$$D_f(p\mid q) = \int q(x), f\left(\frac{p(x)}{q(x)}\right)dx, \qquad f(1)=0$$

where $f:\mathbb{R}_+\to\mathbb{R}$ is **convex**. $\text{(1.1.4)}$

**Examples**
* **Forward KL:** $f(u)=u\log u \Rightarrow D_f = D_{\mathrm{KL}}(p\mid q)$
* **Jensen–Shannon (JS):** $f(u)=\tfrac12\Big[u\log u-(u+1)\log\frac{u+1}{2}\Big] \Rightarrow D_f=D_{\mathrm{JS}}(p\mid q)$
* **Total variation (TV):** $f(u)=\tfrac12\lvert u-1\rvert \Rightarrow D_f=D_{\mathrm{TV}}(p,q)$

#### 5.2) Explicit forms for JS and TV

* **JS divergence**
  
  $$D_{\mathrm{JS}}(p\mid q) =\tfrac12 D_{\mathrm{KL}}\left(p \parallel \tfrac12(p+q)\right) +\tfrac12 D_{\mathrm{KL}}\left(q \parallel \tfrac12(p+q)\right)$$

  Intuition: **smooth + symmetric**, balances both distributions, avoids some unbounded KL behavior; later useful for interpreting GANs.

* **Total variation distance**
  
  $$D_{\mathrm{TV}}(p,q) =\tfrac12\int_{\mathbb{R}^D} \lvert p-q\rvert dx = \sup_{A\subset \mathbb{R}^D} \lvertp(A)-q(A)\rvert$$

  Intuition: captures the **largest possible** difference in probability the two distributions can assign to any event $A$.

#### 5.3) Optimal transport viewpoint: Wasserstein distances

* Unlike $f$-divergences (which compare **density ratios**), **Wasserstein** distances depend on the **geometry of the sample space** and can remain meaningful even if the supports of $p$ and $q$ **do not overlap**.

---

### 6) Challenges in modeling distributions (Section 1.1.2)

To model a complex data distribution, we can parameterize the probability density function $p_{\text{data}}$ using a neural network with parameters $\phi$, creating a model we denote as $p_\phi$. To model a density $p_\phi(x)$ with a neural network, $p_\phi$ must satisfy:

1. **Non-negativity:** $p_\phi(x)\ge 0$ for all $x$.
2. **Normalization:** $\int p_\phi(x)dx = 1$.

#### Practical construction via an unnormalized “energy” output

Let the network output a scalar

$$E_\phi(x)\in\mathbb{R}$$

To interpret this output as a valid density, it must be transformed to satisfy conditions (1) and (2).

Interpret it as defining an **unnormalized** density.

**Step 1: enforce non-negativity**
Use a positive mapping, commonly the exponential:

$$\tilde p_\phi(x) = \exp(E_\phi(x))$$

Interpret it as an **unnormalized** density.

**Step 2: enforce normalization**

$$p_\phi(x) = \frac{\tilde p_\phi(x)}{\int \tilde p_\phi(x')dx'} = \frac{\exp(E_\phi(x))}{\int \exp(E_\phi(x'))dx'}$$

The denominator is the **normalizing constant / partition function**:

$$Z(\phi) := \int \exp(E_\phi(x'))dx'$$

#### Central difficulty

* In **high dimensions**, computing $Z(\phi)$ (and often its gradients) is typically **intractable**.
* This intractability is a major motivation for many DGM families: they’re designed to **avoid**, **approximate**, or **circumvent** the cost of evaluating the partition function.

---

# Variational Perspective: From VAEs to DDPMs

## Big picture

* **Core theme:** VAEs, hierarchical VAEs, and diffusion models can all be viewed as optimizing a **tractable variational lower bound** (a likelihood surrogate) on an otherwise **intractable log-likelihood**.
* **VAE template (learned encoder + learned decoder):**

  * Encoder maps observations → latent distribution.
  * Decoder maps latents → observation distribution, “closing the loop.”
* **DDPM template (fixed encoder + learned decoder):**

  * The “encoder” is a **fixed forward noising process** mapping data → noise.
  * Training learns a **reverse denoising decoder** that inverts this path step-by-step.

---

## 2.1 Variational Autoencoder (VAE)

### Why not a plain autoencoder?

* A standard autoencoder has:

  * deterministic **encoder**: compresses $x$ into a low-dim code
  * deterministic **decoder**: reconstructs $x$
* It can reconstruct well, but the **latent space is unstructured**:

  * sampling random latent codes usually yields meaningless outputs
  * not a reliable **generative** model

### VAE idea (Kingma & Welling, 2013)

* Make the latent space **probabilistic + regularized**, so that:

  * sampling $z$ from a simple prior produces meaningful outputs
  * the model becomes a true generative model

---

## 2.1.1 Probabilistic encoder and decoder

### Variables

* **Observed variable:** $x$ (e.g., an image)
* **Latent variable:** $z$ (captures hidden factors: shape, color, style, $\dots$)

### Prior over latents

Typically a simple prior, e.g.

$$z \sim p(z) = \mathcal N(0, I)$$

### Decoder / generator

Define a conditional likelihood (“decode latents into data”):

$$p_\phi(x \mid z)$$

In practice this is often kept **simple**, e.g. a **factorized Gaussian**, to encourage learning useful latent features rather than memorizing data.

### Sampling procedure

1. Sample $z \sim p(z)$
2. Sample $x \sim p_\phi(x \mid z)$

---

## Latent-variable marginal likelihood (why it’s hard)

A VAE defines the data likelihood via marginalization:

$$p_\phi(x) = \int p_\phi(x \mid z), p(z) dz$$

* Ideally, we would learn $\phi$ by maximizing $\log p_\phi(x)$ (MLE).
* But for expressive nonlinear decoders, the integral over $z$ is **intractable**, so **direct MLE is computationally infeasible**.

---

## Construction of the encoder (inference network)

### True posterior (intractable)

Given $x$, the “correct” latent posterior is:

$$p_\phi(z \mid x) = \frac{p_\phi(x \mid z), p(z)}{p_\phi(x)}$$

* The denominator $p_\phi(x)$ is exactly the intractable marginal likelihood, so **exact inference is prohibitive**.

### Variational approximation

Introduce a learnable approximate posterior (encoder):

$$q_\theta(z \mid x) \approx p_\phi(z \mid x)$$

* This gives a feasible, trainable pathway from $x \to z$.

---

## 2.1.2 Training via the Evidence Lower Bound (ELBO)

### The ELBO bound (Theorem 2.1.1)

For any data point $x$:

$$\log p_\phi(x)  \ge  \mathcal L_{\text{ELBO}}(\theta,\phi; x)$$

where

$$
\mathcal L_{\text{ELBO}}(\theta,\phi; x)
=

\underbrace{\mathbb E_{z\sim q_\theta(z\mid x)}\big[\log p_\phi(x\mid z)\big]}_{\text{Reconstruction term}}
 - 
\underbrace{D_{\mathrm{KL}} \big(q_\theta(z\mid x)\parallel p(z)\big)}_{\text{Latent regularization}}.
$$


### Proof sketch (Jensen’s inequality)

Start from:

$$\log p_\phi(x) = \log \int p_\phi(x,z)dz$$

Multiply and divide by $q_\theta(z\mid x)$:

$$
\log p_\phi(x)
=

\log \int q_\theta(z\mid x)\,\frac{p_\phi(x,z)}{q_\theta(z\mid x)}\,dz
= \log \mathbb E_{z\sim q_\theta(z\mid x)} \left[\frac{p_\phi(x,z)}{q_\theta(z\mid x)}\right].
$$

Apply Jensen:

$$
\log \mathbb E[\cdot]  \ge  \mathbb E[\log(\cdot)]
\quad\Rightarrow\quad
\log p_\phi(x)
\ge
\mathbb E_{q_\theta}\left[\log \frac{p_\phi(x,z)}{q_\theta(z\mid x)}\right]
$$

which rearranges into the ELBO form above.

---

## Interpreting the two ELBO terms

### 1) Reconstruction term

$$\mathbb E_{z\sim q_\theta(z\mid x)}[\log p_\phi(x\mid z)]$$

* Encourages accurate recovery of $x$ from its latent code $z$.
* Under Gaussian encoder/decoder assumptions, this reduces to the familiar **reconstruction loss** of autoencoders.

### 2) Latent KL regularization

$$D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel p(z))$$

* Encourages the encoder distribution to stay close to a simple prior $p(z)$ (e.g. $\mathcal N(0,I)$).
* Shapes the latent space to be smooth/continuous so samples from the prior decode meaningfully.

**Key trade-off:** good reconstructions vs. a well-structured latent space that supports sampling.

---

## Information-theoretic view: ELBO as a divergence bound

### MLE view

Maximum likelihood training corresponds to minimizing:

$$D_{\mathrm{KL}}(p_{\text{data}}(x)\parallel p_\phi(x))$$

which measures how well $p_\phi$ approximates the data distribution (but is generally intractable to optimize directly).

### Joint-distribution trick (variational framework)

Introduce two joint distributions:

* **Generative joint**
  
  $$p_\phi(x,z) = p(z),p_\phi(x\mid z)$$
  
* **Inference joint**
  
  $$q_\theta(x,z) = p_{\text{data}}(x),q_\theta(z\mid x)$$

Comparing them yields:

$$
D_{\mathrm{KL}}(p_{\text{data}}(x)\parallel p_\phi(x))
 \le 
D_{\mathrm{KL}}(q_\theta(x,z)\parallel p_\phi(x,z)).
\qquad\text{(2.1.2)}
$$
**Intuition:** comparing only marginals over $x$ can hide mismatches that become visible when considering the full joint over $(x,z)$.

### Chain rule / decomposition of the joint KL

Expanding the joint KL:

$$
D_{\mathrm{KL}}(q_\theta(x,z),|,p_\phi(x,z))
=

D_{\mathrm{KL}}(p_{\text{data}}(x),|,p_\phi(x))
+
\mathbb E_{p_{\text{data}}(x)}
\Big[
D_{\mathrm{KL}}(q_\theta(z\mid x),|,p_\phi(z\mid x))
\Big].
$$


* First term: **true modeling error** (how well $p_\phi(x)$ matches data)
* Second term: **inference error** (gap between approximate and true posterior)

Because the inference error is nonnegative, you get inequality (2.1.2).

### ELBO gap equals posterior KL

For each $x$,

$$
\log p_\phi(x) - \mathcal L_{\text{ELBO}}(\theta,\phi;x)
=

D_{\mathrm{KL}}(q_\theta(z\mid x),|,p_\phi(z\mid x)).
$$

So **maximizing ELBO** is exactly **reducing the inference gap**, i.e. pushing the variational posterior toward the true posterior.

---

## Connection forward: hierarchical VAEs → DDPMs (conceptual bridge)

* **Hierarchical VAEs:** stack multiple latent layers to capture structure at multiple scales.
* **DDPMs as “many-layer” variational models:**

  * the forward noising process plays the role of a (fixed) encoder that gradually maps data to noise
  * the reverse denoising model is the learned decoder that inverts this mapping step-by-step
* The shared variational viewpoint: all optimize a **variational bound** on likelihood rather than the exact likelihood directly.

---

## Quick formula sheet (from these pages)

* Prior: $p(z)=\mathcal N(0,I)$
* Decoder: $p_\phi(x\mid z)$
* Marginal likelihood: $\displaystyle p_\phi(x)=\int p_\phi(x\mid z)p(z)dz$
* True posterior: $\displaystyle p_\phi(z\mid x)=\frac{p_\phi(x\mid z)p(z)}{p_\phi(x)}$
* Variational posterior: $q_\theta(z\mid x)\approx p_\phi(z\mid x)$
* ELBO:

$$
\mathcal L_{\text{ELBO}}(x) = \mathbb E_{q_\theta(z\mid x)}[\log p_\phi(x\mid z)]
- D_{\mathrm{KL}}(q_\theta(z\mid x) \parallel p(z))
$$


* Joint KL decomposition:
  
  $$D_{\mathrm{KL}}(q(x,z)\parallel p(x,z)) = D_{\mathrm{KL}}(p_{\text{data}}(x)\parallel p_\phi(x)) - \mathbb E_{p_{\text{data}}(x)}D_{\mathrm{KL}}(q(z\mid x)\parallel p_\phi(z\mid x))$$

* ELBO gap:
  
  $$\log p_\phi(x) - \mathcal L_{\text{ELBO}}(x) = D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel p_\phi(z\mid x))$$


## 2.1.3 Gaussian VAE (standard “Gaussian–Gaussian” VAE)

### Setup and notation

* Data: $x \in \mathbb{R}^D$
* Latent: $z \in \mathbb{R}^d$
* Prior: $p_{\text{prior}}(z)$ (often $\mathcal N(0,I)$)

### Encoder (approximate posterior)

The encoder is a diagonal-covariance Gaussian:

$$q_\theta(z\mid x) := \mathcal N \Big(z;\ \mu_\theta(x),\ \mathrm{diag}(\sigma_\theta^2(x))\Big)$$

where

* $\mu_\theta:\mathbb R^D\to\mathbb R^d$
* $\sigma_\theta:\mathbb R^D\to\mathbb R_+^d$
  are deterministic neural-network outputs.

### Decoder (likelihood / generator)

The decoder is a Gaussian with **fixed** variance:

$$p_\phi(x\mid z) := \mathcal N\big(x;\ \mu_\phi(z),\ \sigma^2 I\big)$$

where $\mu_\phi:\mathbb R^d\to\mathbb R^D$ is a neural network and $\sigma>0$ is a (small) constant.

### ELBO specialization (\Rightarrow) MSE reconstruction

Under this likelihood,

$$
\mathbb E_{q_\theta(z\mid x)}\big[\log p_\phi(x\mid z)\big]
= -\frac{1}{2\sigma^2}\ \mathbb E_{q_\theta(z\mid x)}\Big[|x-\mu_\phi(z)|^2\Big] + C,
$$

where $C$ is constant w.r.t. $\theta,\phi$.

So maximizing the ELBO is equivalent (up to constants/sign) to minimizing:

$$
\min_{\theta,\phi}\ \mathbb E_{q_\theta(z\mid x)}\Big[\frac{1}{2\sigma^2}|x-\mu_\phi(z)|^2\Big]
 + D_{\mathrm{KL}} \big(q_\theta(z\mid x),|,p_{\text{prior}}(z)\big).
$$

**Interpretation:** training becomes “regularized reconstruction”:
* a **reconstruction loss** (scaled MSE),
* plus a **KL regularizer** pushing $q_\theta(z\mid x)$ toward the prior.

**Why KL is “easy” here:** for Gaussian $q_\theta$ (and typical Gaussian prior), the KL has a closed form (commonly used in implementations).

---

## 2.1.4 Drawbacks of a standard VAE: blurry outputs

### Why Gaussian VAEs often look blurry (core mechanism)

Consider:

* a **fixed** Gaussian encoder $q_{\text{enc}}(z\mid x)$,
* and a Gaussian decoder with fixed variance
  
  $$p_{\text{dec}}(x\mid z)=\mathcal N(x;\mu(z),\sigma^2I)$$

With an arbitrary encoder, optimizing the ELBO (up to an additive constant) reduces to minimizing an expected squared error:

$$\arg\min_{\mu}\ \mathbb E_{p_{\text{data}}(x),q_{\text{enc}}(z\mid x)}\Big[|x-\mu(z)|^2\Big]$$

This is a least-squares regression problem in $\mu(z)$. The optimal solution is the **conditional mean**:

$$\mu^*(z)=\mathbb E_{q_{\text{enc}}(x\mid z)}[x]$$

### What is $q_{\text{enc}}(x\mid z)$?

It’s the “encoder-induced posterior on inputs given latents”, obtained via Bayes’ rule:

$$q_{\text{enc}}(x\mid z)=\frac{q_{\text{enc}}(z\mid x),p_{\text{data}}(x)}{p_{\text{prior}}(z)}$$

An equivalent (often useful) form:

$$
\mu^*(z)
=\frac{\mathbb E_{p_{\text{data}}(x)}\big[q_{\text{enc}}(z\mid x),x\big]}
{\mathbb E_{p_{\text{data}}(x)}\big[q_{\text{enc}}(z\mid x)\big]}.
$$

### Where blur comes from (mode averaging)

If two distinct inputs $x\neq x'$ are mapped to **overlapping regions** in latent space (i.e., supports of $q_{\text{enc}}(\cdot\mid x)$ and $q_{\text{enc}}(\cdot\mid x')$ intersect), then for such a $z$,

$$\mu^*(z)=\mathbb E[x\mid z]$$

**averages across multiple (possibly unrelated) inputs**. Averaging “conflicting modes” produces **non-distinct, blurry** reconstructions/samples.

**Key takeaway:** with a Gaussian decoder + MSE-like training signal, the optimal prediction is a mean, and means of multimodal/ambiguous conditionals look blurry.

---

## 2.1.5 (Optional) From standard VAE to Hierarchical VAEs (HVAEs)

### Motivation

Hierarchical VAEs introduce **multiple latent layers** to capture structure at different abstraction levels (coarse $\to$ fine). (Referenced: Vahdat & Kautz, 2020.)

### Generative model (top-down hierarchy)

Introduce $z_{1:L}=(z_1,\dots,z_L)$. A common top-down factorization:

$$p_\phi(x,z_{1:L})  =  p_\phi(x\mid z_1)\ \prod_{i=2}^{L} p_\phi(z_{i-1}\mid z_i)\ p(z_L)$$

The marginal data density:

$$p_{\text{HVAE}}(x)  :=  \int p_\phi(x,z_{1:L}),dz_{1:L}$$

**Sampling/generation is progressive:**

1. sample top latent $z_L\sim p(z_L)$
2. decode downward $z_{L-1}\sim p_\phi(z_{L-1}\mid z_L)$, $\dots$, $z_1\sim p_\phi(z_1\mid z_2)$
3. generate $x\sim p_\phi(x\mid z_1)$

### Inference model (bottom-up, mirrors hierarchy)

A common structured encoder uses a bottom-up Markov factorization:

$$q_\theta(z_{1:L}\mid x)  =  q_\theta(z_1\mid x)\ \prod_{i=2}^{L} q_\theta(z_i\mid z_{i-1})$$

---

## HVAE ELBO (derivation + form)

### Jensen’s inequality derivation (standard ELBO trick)

$$
\log p_{\text{HVAE}}(x)
= \log \int p_\phi(x,z_{1:L}),dz_{1:L}
= \log \mathbb E_{q_\theta(z_{1:L}\mid x)}\Big[\frac{p_\phi(x,z_{1:L})}{q_\theta(z_{1:L}\mid x)}\Big]
$$

$$
\ge \mathbb E_{q_\theta(z_{1:L}\mid x)}\Big[\log \frac{p_\phi(x,z_{1:L})}{q_\theta(z_{1:L}\mid x)}\Big]
;=:;\mathcal L_{\text{ELBO}}
$$

Substituting the factorizations:

$$
\mathcal L_{\text{ELBO}}
=\mathbb E_{q_\theta(z_{1:L}\mid x)}
\Bigg[
\log \frac{
p(z_L)\ \prod_{i=2}^L p_\phi(z_{i-1}\mid z_i)\ p_\phi(x\mid z_1)}
{q_\theta(z_1\mid x)\ \prod_{i=2}^L q_\theta(z_i\mid z_{i-1})}
\Bigg]
$$

### Interpretable decomposition (reconstruction + “adjacent” KLs)

A key decomposition shown:

$$
\mathcal L_{\text{ELBO}}(x)
=

\mathbb E_q[\log p_\phi(x\mid z_1)]
-\mathbb E_q \Big[D_{\mathrm{KL}}(q_\theta(z_1\mid x),|,p_\phi(z_1\mid z_2))\Big]

$$
\quad
-\sum_{i=2}^{L-1}\mathbb E_q \Big[D_{\mathrm{KL}}(q_\theta(z_i\mid z_{i-1}),|,p_\phi(z_i\mid z_{i+1}))\Big]
-\mathbb E_q \Big[D_{\mathrm{KL}}(q_\theta(z_L\mid z_{L-1}),|,p(z_L))\Big],
$$

where $\mathbb E_q$ denotes expectation under the encoder-induced joint over $(x,z_{1:L})$ (as written in the text).

**Meaning:** each inference conditional is regularized toward its corresponding **top-down** conditional prior:

+ $q(z_1\mid x)$ vs $p(z_1\mid z_2)$,
+ $q(z_i\mid z_{i-1})$ vs $p(z_i\mid z_{i+1})$,
+ top level $q(z_L\mid z_{L-1})$ vs $p(z_L)$.

### Observation 2.1.1 (core intuition)

Stacking layers lets the model generate **progressively** (coarse $\to$ fine), which helps capture complex high-dimensional structure.

---

## Why “just make a flat VAE deeper” is not enough

### Limitation 1: the variational family is still too simple

In a standard flat VAE,

$$q_\theta(z\mid x)=\mathcal N\big(z;\mu_\theta(x),\mathrm{diag}(\sigma_\theta^2(x))\big)$$

is **one unimodal Gaussian** per $x$. Making networks deeper can improve $\mu_\theta,\sigma_\theta$, but does **not** change the fact that the posterior family is unimodal (even full-covariance remains a single ellipsoid).

If the true posterior $p_\phi(z\mid x)$ is **multi-peaked**, this mismatch loosens the ELBO and weakens inference. Fix needs a **richer posterior class**, not just deeper nets.

### Limitation 2: posterior collapse with an expressive decoder

Recall the expected objective:

$$
\mathbb E_{p_{\text{data}}(x)}[\mathcal L_{\text{ELBO}}(x)]
=

\mathbb E_{p_{\text{data}}(x),q_\theta(z\mid x)}[\log p_\phi(x\mid z)]
-\mathbb E_{p_{\text{data}}(x)}[D_{\mathrm{KL}}(q_\theta(z\mid x),|,p(z))].
$$


This can be rewritten as:

$$\mathbb E_{p_{\text{data}}(x),q_\theta(z\mid x)}[\log p_\phi(x\mid z)] -\mathcal I_q(x;z) -D_{\mathrm{KL}}(q_\theta(z)\parallel p(z))$$

where

$$\mathcal I_q(x;z)=\mathbb E_{q(x,z)}\Big[\log \frac{q_\theta(z\mid x)}{q_\theta(z)}\Big] =\mathbb E_{p_{\text{data}}(x)}\Big[D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel q_\theta(z))\Big]$$

and the aggregated posterior is

$$q_\theta(z)=\int p_{\text{data}}(x),q_\theta(z\mid x)dx$$

**Collapse story:** if the decoder can model the data well **without using $z$** (i.e., effectively $p_\phi(x\mid z)\approx r(x)\approx p_{\text{data}}(x)$), then an ELBO maximizer can choose

$$q_\theta(z\mid x)=p(z)$$

making $\mathcal I_q(x;z)=0$ and $q_\theta(z)=p(z)$. Then $z$ carries no information about $x$, and changing $z$ doesn’t affect outputs (controllability fails). Making the networks deeper does not automatically remove this “ignore $z$” solution.

---

## What hierarchy changes (and what new issues appear)

### What improves conceptually

The HVAE ELBO uses **multiple adjacent KL terms**, so the “information penalty” is:

* **distributed across layers**, and
* **localized** (each layer matches to its neighbor’s conditional prior),
  which comes from the hierarchical latent graph—not simply from depth in the encoder/decoder networks.

### Training challenges (as noted)

Even though HVAEs are more expressive, training can be unstable because:

* lower layers + decoder may already reconstruct $x$, leaving higher latents with little signal,
* gradients to deep latents can be indirect/weak,
* overly expressive conditionals can dominate reconstruction and suppress higher-level latents,
  so capacity balancing becomes important.

### Forward pointer

The text notes that diffusion models can be seen as inheriting the *progressive hierarchy idea* while sidestepping key HVAE weaknesses by fixing the encoding process and learning the generative reversal.

### Notation note

To avoid ambiguity, the text mentions deviating from the “$q$=encoder, $p$=generator” convention and instead using $p$ with clear subscripts/superscripts to indicate roles.

---


## Study notes — Variational perspective on DDPMs (Sections 2.2–2.2.3)

### Big picture: DDPM as a “VAE-like” variational model

DDPMs (Denoising Diffusion Probabilistic Models) can be viewed as a variational generative model with two coupled stochastic processes:

* **Forward process (fixed encoder)**: progressively **corrupt** data with Gaussian noise through a *fixed* Markov chain.
* **Reverse process (learnable decoder)**: learn a Markov chain that **denoises** step-by-step, starting from pure noise.

This “gradual generation” is easier to learn than generating a full sample in one shot.

---

## 1) The two chains and their roles

### 1.1 Forward pass: fixed corruption (encoder)

A Markov chain:

$$x_0 \to x_1 \to \cdots \to x_L$$

where each step injects Gaussian noise via a fixed kernel $p(x_i\mid x_{i-1})$. As $i$ grows, the distribution becomes close to an isotropic Gaussian (“pure noise”).

### 1.2 Reverse denoising: learnable generation (decoder)

A reverse chain:

$$x_L \to x_{L-1} \to \cdots \to x_0$$

where we learn a parametric transition:

$$p_\phi(x_{i-1}\mid x_i)$$

so that starting from $x_L \sim p_{\text{prior}}$, we iteratively denoise to obtain a realistic $x_0$.

---

## 2) Forward process (fixed encoder) — formalization

### 2.1 Fixed Gaussian transitions

Each forward step uses a fixed Gaussian transition kernel:

$$p(x_i\mid x_{i-1}) := \mathcal N\left(x_i;; \sqrt{1-\beta_i^2},x_{i-1},; \beta_i^2 I\right)$$

where $\lbrace\beta_i\rbrace_{i=1}^L$ is a predetermined increasing noise schedule, $\beta_i\in(0,1)$.

Define

$$\alpha_i := \sqrt{1-\beta_i^2}$$

Then the transition can be written as the intuitive iterative update:

$$x_i = \alpha_i x_{i-1} + \beta_i \varepsilon_i,\qquad \varepsilon_i\sim\mathcal N(0,I)\text{ iid.}$$

**Interpretation**

* $\alpha_i$ shrinks the previous state.
* $\beta_i\varepsilon_i$ adds controlled Gaussian noise.

---

### 2.2 Perturbation kernel (closed form $x_i\mid x_0$)

By composing Gaussian transitions, you get a closed-form distribution of $x_i$ given the original data $x_0$:

$$p_i(x_i\mid x_0)=\mathcal N \left(x_i;; \bar\alpha_i x_0,; (1-\bar\alpha_i^2)I\right)$$

where

$$\bar\alpha_i := \prod_{k=1}^i \alpha_k$$

#### Direct sampling form (Eq. 2.2.1)

You can sample $x_i$ in one shot:

$$x_i = \bar\alpha_i x_0 + \sqrt{1-\bar\alpha_i^2},\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I)$$

This is the key computational convenience in DDPM training: you don’t need to simulate all intermediate steps to get $x_i$.

---

### 2.3 Prior distribution from the long-run limit

If the noise schedule increases and $L$ is large, the forward marginal converges:

$$p_L(x_L\mid x_0)\to \mathcal N(0,I)\quad \text{as }L\to\infty$$

motivating the **prior**

$$p_{\text{prior}} := \mathcal N(0,I)$$

independent of $x_0$.

---

### 2.4 Continuous-time-like shorthand (identity in distribution)

Often we write (for a fixed index $t$):

$$p_t(x_t\mid x_0)=\mathcal N(x_t;\alpha_t x_0,\sigma_t^2 I)$$

equivalently (identity in distribution)

$$x_t \overset{d}{=} \alpha_t x_0 + \sigma_t \varepsilon$$

meaning $x_t$ and $\alpha_t x_0+\sigma_t\varepsilon$ have the same *law* (same density), hence same expectations for test functions.

---

## 3) Reverse denoising process (learnable decoder)

### 3.1 The core question (Question 2.2.1)

Can we compute—or approximate—the true reverse transition

$$p(x_{i-1}\mid x_i)$$

even though $x_i\sim p_i(x_i)$ is complicated?

### 3.2 Why the “obvious” Bayes formula is intractable

Bayes gives:

$$p(x_{i-1}\mid x_i)=p(x_i\mid x_{i-1})\frac{p_{i-1}(x_{i-1})}{p_i(x_i)}$$

But the marginals involve the unknown data distribution:

$$p_i(x_i)=\int p_i(x_i\mid x_0),p_{\text{data}}(x_0),dx_0$$

(and similarly for $p_{i-1}$), so exact densities are unavailable.

---

## 4) The conditioning trick: make the target tractable

### 4.1 Condition on the clean sample

Instead of targeting $p(x_{i-1}\mid x_i)$ directly, consider:

$$p(x_{i-1}\mid x_i, x)$$

where $x$ is the *clean* data sample (effectively $x=x_0$).

Using:

* the **Markov property** of the forward process $p(x_i\mid x_{i-1},x)=p(x_i\mid x_{i-1})$,
* and the fact all relevant distributions are **Gaussian**,

the conditional reverse kernel becomes Gaussian and has a closed form.

---

### 4.2 Lemma 2.2.2 — reverse conditional transition kernel (Eq. 2.2.4)

$$p(x_{i-1}\mid x_i,x)=\mathcal N \left(x_{i-1};\mu(x_i,x,i),,\sigma^2(i)I\right)$$

with

$$
\mu(x_i,x,i)=
\frac{\bar\alpha_{i-1}\beta_i^2}{1-\bar\alpha_i^2},x
 + 
\frac{(1-\bar\alpha_{i-1}^2)\alpha_i}{1-\bar\alpha_i^2},x_i,
$$

and

$$\sigma^2(i)=\frac{1-\bar\alpha_{i-1}^2}{1-\bar\alpha_i^2},\beta_i^2$$

**Intuition**

* The posterior mean is a *precision-weighted blend* of the clean signal $x$ and the noisy observation $x_i$.
* As noise increases, $x_i$ becomes less informative, and the weights shift accordingly.

---

## 5) Training objective via KL minimization

### 5.1 “Ideal” objective (marginal KL; Eq. 2.2.2)

Introduce a learnable model $p_\phi(x_{i-1}\mid x_i)$ and aim to minimize:

$$\mathbb E_{p_i(x_i)} \left[ D_{\mathrm{KL}} \big(p(x_{i-1}\mid x_i)\parallel p_\phi(x_{i-1}\mid x_i)\big)\right]$$

But this involves the intractable $p(x_{i-1}\mid x_i)$.

---

### 5.2 Theorem 2.2.1 — equivalence between marginal and conditional KL (Eq. 2.2.3)

The key equality:

$$
\mathbb E_{p_i(x_i)} \left[ D_{\mathrm{KL}} \big(p(x_{i-1}\mid x_i)\parallel p_\phi(x_{i-1}\mid x_i)\big)\right]
=

\mathbb E_{p_{\text{data}}(x)}\mathbb E_{p(x_i\mid x)}
 \left[
D_{\mathrm{KL}} \big(p(x_{i-1}\mid x_i,x)\parallel p_\phi(x_{i-1}\mid x_i)\big)
\right] + C,
$$

where $C$ does not depend on $\phi$.

So **minimizing the intractable marginal KL** is equivalent (up to an additive constant) to **minimizing a tractable conditional KL** with $x\sim p_{\text{data}}$ and $x_i\sim p(x_i\mid x)$.

Also, the minimizer satisfies the mixture identity:

$$p^*(x_{i-1}\mid x_i) = \mathbb E_{p(x\mid x_i)}[p(x_{i-1}\mid x_i,x)] = p(x_{i-1}\mid x_i),\qquad x_i\sim p_i$$

**Interpretation**
* The true reverse kernel is a mixture (over possible clean $x$ consistent with $x_i$) of the tractable conditional posteriors.
* Training on the conditional KL is “the right thing” to recover the marginal reverse.

---

## 6) Modeling $p_\phi(x_{i-1}\mid x_i)$ and simplifying the loss

### 6.1 Gaussian parameterization (Eq. 2.2.5)

DDPM assumes each reverse transition is Gaussian:

$$p_\phi(x_{i-1}\mid x_i):=\mathcal N\left(x_{i-1};\mu_\phi(x_i,i),\sigma^2(i)I\right)$$

* $\mu_\phi(\cdot,i):\mathbb R^D\to\mathbb R^D$ is a learnable mean function (neural net).
* $\sigma^2(i)$ is **fixed**, taken from the closed-form posterior variance in Eq. (2.2.4).

---

### 6.2 Diffusion loss as sum of KLs (Eq. 2.2.6)

Define (for one clean sample $x_0$):

$$
\mathcal L_{\text{diffusion}}(x_0;\phi):=
\sum_{i=1}^L
\mathbb E_{p(x_i\mid x_0)}
\left[
D_{\mathrm{KL}}\big(p(x_{i-1}\mid x_i,x_0)\parallel p_\phi(x_{i-1}\mid x_i)\big)
\right].
$$

---

### 6.3 Closed-form simplification to weighted MSE (Eq. 2.2.7)

Since both distributions in the KL are Gaussians with the **same covariance** $\sigma^2(i)I$, the KL reduces to a squared error between means (plus constant):

$$
\mathcal L_{\text{diffusion}}(x_0;\phi)=
\sum_{i=1}^L
\frac{1}{2\sigma^2(i)}
\left\|\mu_\phi(x_i,i)-\mu(x_i,x_0,i)\right\|_2^2
 + C
$$

Here $\mu(x_i,x_0,i)$ is the *analytic target* from Lemma 2.2.2.

---

### 6.4 Final DDPM training objective (Eq. 2.2.8)

Average over the data distribution and drop the constant:

$$
\mathcal L_{\text{DDPM}}(\phi):=
\sum_{i=1}^L
\frac{1}{2\sigma^2(i)}
\mathbb E_{x_0\sim p_{\text{data}}}
\mathbb E_{p(x_i\mid x_0)}
\left[\left\|\mu_\phi(x_i,i)-\mu(x_i,x_0,i)\right\|_2^2\right].
$$

**What you learn**

* You’re effectively training a network to match the **posterior mean** of $x_{i-1}$ given $(x_i,x_0)$, across all noise levels.

---

## 7) Practical “mental model” summary

### Forward (known, easy)

* Pick schedule $\lbrace\beta_i\rbrace$, compute $\alpha_i=\sqrt{1-\beta_i^2}$, $\bar\alpha_i=\prod_{k\le i}\alpha_k$.
* Sample noisy state directly:
  
  $$x_i=\bar\alpha_i x_0+\sqrt{1-\bar\alpha_i^2},\varepsilon$$

### Reverse (learned, step-by-step)

* Start from $x_L\sim\mathcal N(0,I)$.
* For $i=L,\dots,1$, sample:
  
  $$x_{i-1}\sim \mathcal N\big(\mu_\phi(x_i,i),\sigma^2(i)I\big)$$

### Training signal comes from tractable conditioning

* Instead of computing $p(x_{i-1}\mid x_i)$ (hard), compute $p(x_{i-1}\mid x_i,x_0)$ (Gaussian, closed form).
* The theorem guarantees this yields an equivalent optimization problem.

---

## 8) Key equations to memorize (minimal set)

1. **Forward step**
   
   $$p(x_i\mid x_{i-1})=\mathcal N(x_i;\alpha_i x_{i-1},\beta_i^2 I),\quad \alpha_i=\sqrt{1-\beta_i^2}$$
   

2. **Closed form perturbation**
   
   $$p(x_i\mid x_0)=\mathcal N(x_i;\bar\alpha_i x_0,(1-\bar\alpha_i^2)I),\quad \bar\alpha_i=\prod_{k\le i}\alpha_k$$

3. **Direct sampling**
   
   $$x_i=\bar\alpha_i x_0+\sqrt{1-\bar\alpha_i^2}\varepsilon$$

4. **Reverse conditional posterior (Lemma 2.2.2)**
   
   $$p(x_{i-1}\mid x_i,x_0)=\mathcal N(x_{i-1};\mu(x_i,x_0,i),\sigma^2(i)I)$$

   with the explicit $\mu(\cdot)$, $\sigma^2(\cdot)$ above.

5. **Final training objective**
   
   $$\mathcal L_{\text{DDPM}}(\phi)=\sum_{i=1}^L\frac{1}{2\sigma^2(i)} \mathbb E\left[|\mu_\phi(x_i,i)-\mu(x_i,x_0,i)|_2^2\right]$$

---

## Notation (quick recap)

* Clean data: $x_0 \sim p_{\text{data}}$.
* Noisy latent at step $i$: $x_i$.
* Noise: $\epsilon \sim \mathcal N(0, I)$.
* Noise schedule scalars:
  * $\alpha_i \in (0,1)$ (per-step “signal keep” factor),
  * $\bar \alpha_i := \prod_{j=1}^i \alpha_j$ (cumulative keep),
  * so $\bar \alpha_i^2$ appears frequently.
* Forward noising (DDPM forward process):
  
  $$x_i  =  \bar \alpha_i x_0  +  \sqrt{1-\bar \alpha_i^2},\epsilon \qquad \text{(2.2.9)}$$

---

# 2.2.4 Practical Choices of Predictions and Loss

## A. $\epsilon$-prediction (noise prediction)

### 1) Why reparameterize?

Although DDPM can be written as predicting the **reverse mean** $\mu(\cdot)$ directly (a “mean prediction” view), implementations typically train a network to predict the **added noise** $\epsilon$. This is an *equivalent reparameterization* but is simpler and numerically well-scaled.

### 2) Reverse mean written in terms of $\epsilon$

Using the forward identity $x_i = \bar\alpha_i x_0 + \sqrt{1-\bar\alpha_i^2}\epsilon$, the reverse mean $\mu(x_i,x_0,i)$ can be rewritten as:

$$
\mu(x_i, x_0, i)
=

\frac{1}{\alpha_i}\Bigg(
x_i - \frac{1-\alpha_i^2}{\sqrt{1-\bar\alpha_i^2}},\epsilon
\Bigg).
$$


### 3) Parameterizing the mean via a noise network

Define a neural net $\epsilon_\phi(x_i,i)$ and plug it into the same functional form:

$$
\mu_\phi(x_i,i)
=

\frac{1}{\alpha_i}\Bigg(
x_i - \frac{1-\alpha_i^2}{\sqrt{1-\bar\alpha_i^2}},\epsilon_\phi(x_i,i)
\Bigg).
$$


### 4) Loss becomes an $\ell_2$ noise regression (up to a weight)

Because $\mu_\phi$ depends linearly on $\epsilon_\phi$,

$$
\lvert\mu_\phi(x_i,i) - \mu(x_i,x_0,i)\rvert_2^2
 \propto 
\lvert\epsilon_\phi(x_i,i) - \epsilon\rvert_2^2,
$$

with a proportionality factor that depends on $i$ (a timestep-dependent weight).

**Interpretation:** the model is a “noise detective” that estimates what noise was added; subtracting it moves $x_i$ toward a cleaner sample; repeating this over steps reconstructs data from pure noise.

---

## B. Simplified training loss (the standard DDPM loss)

In practice one often *drops the timestep-dependent weighting*, giving the widely-used objective:

$$
\mathcal L_{\text{simple}}(\phi)
:=
\mathbb E_{i};
\mathbb E_{x_0\sim p_{\text{data}}};
\mathbb E_{\epsilon\sim \mathcal N(0,I)}
\Big[\lvert\epsilon_\phi(x_i,i)-\epsilon\rvert_2^2\Big],
\qquad\text{(2.2.10)}
$$

where $x_i = \bar\alpha_i x_0 + \sqrt{1-\bar\alpha_i^2}\epsilon$.

**Key practical reason:** the target noise $\epsilon$ has **unit variance at every step**, so the loss scale stays consistent across timesteps and you avoid exploding/vanishing targets and explicit weighting.

### Optimal solution under $\ell_2$

Because it’s a least-squares regression problem:

$$\epsilon^*(x_i,i) = \mathbb E[\epsilon \mid x_i],\qquad x_i \sim p_i$$

So at optimum, the network predicts the **conditional expectation** of the true noise given the noisy input.

---

## C. Another equivalent parameterization: $x$-prediction (clean prediction)

Instead of predicting noise, you can predict the clean sample directly with a network $x_\phi(x_i,i)\approx x_0$.

### 1) Reverse mean expressed with a clean predictor

Replacing the ground-truth $x_0$ in the reverse mean expression with $x_\phi(x_i,i)$ yields a model of the form:

$$
\mu_\phi(x_i,i)
=

\frac{\bar\alpha_{i-1}\beta_i^2}{1-\bar\alpha_i^2},x_\phi(x_i,i)
 + 
\frac{(1-\bar\alpha_{i-1}^2)\alpha_i}{1-\bar\alpha_i^2},x_i.
$$

(Exact coefficients depend on the schedule/notation, but the important point is: **$\mu_\phi$** is an affine combination of the predicted clean sample and the current noisy sample.)

### 2) Training objective becomes a weighted clean regression

Analogously,

$$
\lvert\mu_\phi(x_i,i)-\mu(x_i,x_0,i)\rvert_2^2
 \propto 
\lvert x_\phi(x_i,i)-x_0|\rvert_2^2,
$$

so the mean-matching loss reduces to:

$$
\mathbb E_i\mathbb E_{x_0,\epsilon}
\Big[\omega_i|x_\phi(x_i,i)-x_0|_2^2\Big]
$$

for some weight (\omega_i).

### Optimal solution

Again least squares implies:

$$
x^*(x_i,i) = \mathbb E[x_0\mid x_i],\qquad x_i\sim p_i.
\qquad\text{(2.2.11)}
$$

### 3) Connection between $\epsilon$-pred and $x$-pred

They are linked by the forward noising relation:

$$
x_i
=

\bar\alpha_i,x_\phi(x_i,i)
+
\sqrt{1-\bar\alpha_i^2},\epsilon_\phi(x_i,i).
\tag{2.2.12}
$$

So, given a noise estimate you can compute a clean estimate:

$$
\hat x_0(x_i,i)
=

\frac{x_i-\sqrt{1-\bar\alpha_i^2},\epsilon_\phi(x_i,i)}{\bar\alpha_i}.
$$

And conversely you can get $\hat \epsilon$ from $\hat x_0$.

---

# 2.2.5 DDPM’s ELBO (variational/MLE grounding)

## A. DDPM generative model as a reverse-time latent variable model

Define the reverse Markov chain:

$$
p_\phi(x_0, x_{1:L})
:=
p_\phi(x_0\mid x_1),p_\phi(x_1\mid x_2)\cdots p_\phi(x_{L-1}\mid x_L),p_{\text{prior}}(x_L),
$$

and the marginal model:

$$p_\phi(x_0) := \int p_\phi(x_0,x_{1:L}),dx_{1:L}$$


## B. Theorem (ELBO decomposition)

The objective corresponds to an ELBO (lower bound on log-likelihood):

$$
-\log p_\phi(x_0)
 \le 
-\mathcal L_{\text{ELBO}}(x_0;\phi)
:=
\mathcal L_{\text{prior}}(x_0)
+
\mathcal L_{\text{recon}}(x_0;\phi)
+
\mathcal L_{\text{diffusion}}(x_0;\phi)
\qquad\text{(2.2.13)}
$$

Where:

* **Prior-matching term**
  
  $$
  \mathcal L_{\text{prior}}(x_0)
  :=
  D_{\mathrm{KL}}\big(p(x_L\mid x_0)\parallel p_{\text{prior}}(x_L)\big).
  $$

* **Reconstruction / decoder term**
  
  $$
  \mathcal L_{\text{recon}}(x_0;\phi)
  :=
  \mathbb E_{p(x_1\mid x_0)}\big[-\log p_\phi(x_0\mid x_1)\big].
  $$

* **Diffusion (sum of per-step KLs)**
  
  $$
  \mathcal L_{\text{diffusion}}(x_0;\phi)
  :=
  \sum_{i=1}^L
  \mathbb E_{p(x_i\mid x_0)}
  \Big[
  D_{\mathrm{KL}}\big(
  p(x_{i-1}\mid x_i,x_0)\parallel p_\phi(x_{i-1}\mid x_i)
  \big)
  \Big].
  $$

**Proof idea (high level):** Jensen’s inequality, like VAE/HVAE ELBO derivations.

## C. Practical remarks from the text

* $\mathcal L_{\text{prior}}$ can be made small by choosing the noise schedule so that $p(x_L\mid x_0)\approx p_{\text{prior}}$ (typically $\mathcal N(0,I)$).
* $\mathcal L_{\text{recon}}$ is handled via Monte Carlo estimates in practice.
* $\mathcal L_{\text{diffusion}}$ enforces that each learned reverse conditional matches the corresponding true reverse conditional.

## D. Data processing inequality view

With latents $z=x_{1:L}$:

$$
D_{\mathrm{KL}}(p_{\text{data}}(x_0)\parallel p_\phi(x_0))
 \le 
D_{\mathrm{KL}}(p(x_0,x_{1:L})\parallel p_\phi(x_0,x_{1:L})),
$$

where $p(x_0,x_{1:L})$ is the forward-process joint.

## E. HVAE-style interpretation (important conceptual framing)

* “Encoder” is the **fixed forward noising chain** (not learned).
* Latents $x_{1:T}$ share the same dimensionality as data.
* No per-level learned encoder or per-level KL terms like in standard HVAEs.
* Training decomposes into **well-conditioned denoising subproblems** from large noise to small noise (coarse-to-fine), which stabilizes optimization and tends to yield high sample quality.

---

# 2.2.6 Sampling (generation)

Assume the $\epsilon$-prediction model has been trained and frozen: $\epsilon_{\phi^*}$.

## A. Standard DDPM sampling recursion

Start from Gaussian noise:

$$x_L \sim p_{\text{prior}} = \mathcal N(0,I)$$

For $i=L,L-1,\dots,1$, sample:

$$
x_{i-1}
\leftarrow
\underbrace{
\frac{1}{\alpha_i}
\Big(
x_i - \frac{1-\alpha_i^2}{\sqrt{1-\bar\alpha_i^2}},
\epsilon_{\phi^*}(x_i,i)
\Big)}_{\mu_{\phi^*}(x_i,i)}
 + 
\sigma(i),\epsilon_i,
\qquad
\epsilon_i\sim\mathcal N(0,I).
\qquad\text{(2.2.14)}
$$

This repeats until $x_0$ is produced.

## B. Another interpretation: “predict clean then step”

From the (\epsilon)-estimate, define the implied clean prediction:

$$
x_{\phi^*}(x_i,i)
=

\frac{x_i-\sqrt{1-\bar\alpha_i^2},\epsilon_{\phi^*}(x_i,i)}{\bar\alpha_i}.
$$

Plugging into the update shows:

$$
x_{i-1}
\leftarrow
(\text{interpolation between }x_i\text{ and }x_{\phi^*})
 + 
\sigma(i)\epsilon_i
$$

So each step:

1. **Estimate clean signal** from the current noisy latent,
2. **Move to a slightly less noisy latent** (plus controlled Gaussian noise).

## C. Why early steps are “coarse” and later steps add “detail”

Even if $x_{\phi^*}$ is optimal (it predicts $\mathbb E[x_0\mid x_i]$), it only returns the **average** clean sample consistent with $x_i$. At high noise, many clean images map to similar $x_i$, so the conditional expectation can look **blurry**.

Sampling proceeds **high noise $\to$ low noise**, progressively refining:

* early steps set global structure,
* later steps sharpen and add fine details.

---

# Why DDPM sampling is slow (and the core bottleneck)

DDPM sampling is inherently slow because it is a **sequential** reverse-time process.

Main factors described:

1. In theory, an expressive $p_\phi(x_{i-1}\mid x_i)$ can match the true reverse $p(x_{i-1}\mid x_i)$.
2. In practice, $p_\phi(x_{i-1}\mid x_i)$ is typically modeled as a **Gaussian**, limiting expressiveness.
3. For **small** forward noise $\beta_i$, the true reverse is approximately Gaussian $\implies$ good fit.
4. For **large** $\beta_i$, the true reverse can be **multimodal / non-Gaussian**, which a single Gaussian can’t capture well.
5. To maintain accuracy, DDPM uses **many small $\beta_i$ steps**, yielding $O(L)$ sequential network evaluations $\epsilon_{\phi^*}(x_i,i)$, which:

   * prevents parallelization,
   * slows generation.

This motivates later continuous-time / differential-equation viewpoints and faster samplers.

---

## High-yield “exam style” takeaways

* $\epsilon$-prediction, $x$-prediction, and mean-prediction are **equivalent parameterizations** of the same underlying reverse model; they differ mainly by *what the network outputs* and the induced loss scaling.
* With $\ell_2$ loss:

  * $\epsilon^*(x_i,i)=\mathbb E[\epsilon\mid x_i]$,
  * $x^*(x_i,i)=\mathbb E[x_0\mid x_i]$.
* DDPM training is grounded as **ELBO maximization** with a sum of KLs across timesteps.
* Sampling is **iterative denoising** from $x_L\sim\mathcal N(0,I)$ down to $x_0$.
* DDPM is slow because generation is **$L$-step sequential** and uses many small noise steps to keep Gaussian reverse approximations accurate.


# Score-Based Perspective: From EBMs to NCSN


## Big picture: why EBMs show up in diffusion / score-based modeling

* **Energy-Based Model (EBM)** viewpoint: represent a probability distribution through an **energy landscape**:
  * **Low energy** where data are likely
  * **High energy** elsewhere
* **Sampling** often uses **Langevin dynamics**: you iteratively move a point in the direction that increases probability (decreases energy), with some noise to explore.
* The key object that guides this motion is the **score**:
  * The score is a **vector field** pointing toward **higher probability density** regions.
* **Central observation:** knowing the **score field** is enough for generation:
  * You can move samples toward likely regions **without computing the normalization constant**.
* **Score-based diffusion models** build on this:
  * Instead of learning the score of the clean data distribution directly, they learn scores for a **sequence of Gaussian-noise–perturbed distributions** (easier to approximate).
  * Generation becomes **progressive denoising** guided by these learned vector fields.

## Energy-Based Models: Modeling Probability Distributions Using Energy Functions

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Density via an energy function)</span></p>

Let $x \in \mathbb{R}^D$ be a data point. An EBM defines an energy function $E_\phi(x)$ (parameters $\phi$).

**Normalized density:**

$$p_\phi(x) := \frac{\exp(-E_\phi(x))}{Z_\phi}, \qquad Z_\phi := \int_{\mathbb{R}^D} \exp(-E_\phi(x))dx$$

* $Z_\phi$ is the **partition function** that enforces normalization:
  
  $$\int_{\mathbb{R}^D} p_\phi(x)dx = 1$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key interpretation)</span></p>

* Lower $E_\phi(x)$  
* $\Rightarrow$ larger $\exp(-E_\phi(x))$ 
* $\Rightarrow$ **higher probability**.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/ebm_training.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of EBM training. The model lowers density (raises energy) at "bad" data points (red arrows), and raises density (lowers energy) at "good" data points (green arrows).</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(“Only relative energies matter”)</span></p>

* If you add a constant $c$ to all energies, $E_\phi(x)\mapsto E_\phi(x)+c$:
  * numerator $\exp(-E_\phi(x)-c)$ and denominator $Z_\phi$ both get multiplied by $\exp(-c)$
  * $p_\phi(x)$ stays the same
    * $\implies$ EBMs are invariant to global energy shifts.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Global trade-off due to normalization)</span></p>

Because probabilities must sum to 1:
* decreasing energy in one region (increasing its probability mass) necessarily **decreases probability elsewhere**.
* EBMs therefore impose a **global coupling**: “making one valley deeper makes others shallower.”

</div>

#### Maximum likelihood for EBMs — and why it’s hard

In principle, EBMs can be trained by maximum likelihood, which naturally balances fitting the data with global regularization:

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum likelihood for EBMs)</span></p>

$$
\mathcal{L}_{\text{MLE}}(\phi)
= \mathbb{E}_{p_{\text{data}}(x)}
\left[
\log \frac{\exp(-E_\phi(x))}{Z_\phi}
\right]
$$

$$\mathcal{L}_{\text{MLE}}(\phi) = -\mathbb{E}_{p_{\text{data}}}[E_\phi(x)] - \log \int \exp(-E_\phi(x))dx$$

* **Term 1:** lowers energy on real data
* **Term 2:** enforces normalization via $Z_\phi$ (a kind of global regularization)

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The bottleneck of Maximum likelihood for EBMs)</span></p>

* In high dimensions, $\log Z_\phi$ and especially its gradient are **intractable**
* Because computing gradients involves expectations under the **model distribution**.
* This motivates alternatives:
  * approximate the hard term (e.g. contrastive divergence)
  * or bypass it entirely via **score matching**

</div>

### The score function

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Score function)</span></p>

For a density $p(x)$ on $\mathbb{R}^D$, the **score** is:

$$s(x) := \nabla_x \log p(x), \qquad s:\mathbb{R}^D \to \mathbb{R}^D$$

* $s(x)$ forms a vector field pointing in the direction where $\log p(x)$ increases fastest,
* i.e. where the probability density increases.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/score_vector_fields.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of score vector fields. Scor evector fields $∇_x \log p(x)$ indicate directions of increasing density.</figcaption>
</figure>

#### Why model scores instead of densities?

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(“Only relative energies matter”)</span></p>

Many models are only known up to an unnormalized density $\tilde p(x)$:

$$p(x) = \frac{\tilde p(x)}{Z}, \qquad Z = \int \tilde p(x)dx$$

$$\nabla_x \log p(x) = \nabla_x \log \tilde p(x) - \nabla_x \log Z = \nabla_x \log \tilde p(x)$$

because $Z$ is constant in $x$.
$\implies$ **The score ignores the partition function.**

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Benefit 2: The score is a complete representation (up to a constant))</span></p>

Since $s(x) = \nabla_x \log p(x)$, you can recover $\log p(x)$ (up to an additive constant) by integrating the score:

$$\log p(x) = \log p(x_0) + \int_0^1 s\big(x_0 + t(x-x_0)\big)^\top (x-x_0)dt$$

* $x_0$ is a reference point.
* $\log p(x_0)$ is fixed by normalization.

$\implies$ Modeling the score can be as expressive as modeling the density $p(x)$ itself, while often more tractable for generative modeling.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(EBM's score)</span></p>

For an EBM:

$$p_\phi(x) = \frac{\exp(-E_\phi(x))}{Z_\phi}$$

$$\log p_\phi(x) = -E_\phi(x) - \log Z_\phi$$

$$\nabla_x \log p_\phi(x) = -\nabla_x E_\phi(x)$$

because $\nabla_x \log Z_\phi = 0$.
$\implies$ **The model score equals $-\nabla_x E_\phi(x)$** and does **not** depend on $Z_\phi$.

</div>

### Training EBMs via score matching

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Core Score Matching Objective)</span></p>

Score matching trains by aligning:
* model score $\nabla_x \log p_\phi(x)$
* with the (unknown) data score $\nabla_x \log p_{\text{data}}(x)$

$$\mathcal{L}_{\text{SM}}(\phi) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)} \left[ \|\nabla_x \log p_\phi(x) - \nabla_x \log p_{\text{data}}(x)\|^2 \right]$$

</div>

#### How can this work if the data score is unknown?

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Getting rid of data score)</span></p>

Using integration by parts, this becomes an equivalent expression depending only on the energy and its derivatives:

$$\mathcal{L}_{\text{SM}}(\phi) = \mathbb{E}_{p_{\text{data}}(x)} \left[ \mathrm{Tr}\left(\nabla_x^2 E_\phi(x)\right) + \frac{1}{2}\|\nabla_x E_\phi(x)\|^2 \right] + C$$

- $\nabla_x^2 E_\phi(x)$ is the **Hessian**
- $C$ is a constant independent of $\phi$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pros / Cons)</span></p>

**Pros**
* Eliminates the partition function $Z_\phi$
* Avoids sampling from the model during training

**Main drawback**
* Requires **second-order derivatives** (Hessians / traces), which can be expensive in high dimensions.

</div>

### Langevin sampling with score functions

Sampling from EBMs can be performed using **Langevin dynamics**.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Useful mental model)</span></p>

* **deterministic part:** move “uphill in probability” (follow the score / descend energy)
* **stochastic part:** add noise to keep exploring and not get stuck.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/langevin_sampling.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of Langevin sampling. Langevin sampling using the score function $∇_x \log p_ϕ(x)$ to guide trajectories toward high-density regions via the update in Equation (3.1.5) (indicating by arrows).</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Discrete-time Langevin dynamics)</span></p>

**Update rule (energy form):**

$$x_{n+1}=x_n-\eta\nabla_x E_\phi(x_n)+\sqrt{2\eta}\varepsilon_n,\qquad \varepsilon_n\sim\mathcal N(0,I)$$

* $x_0$ is initialized from some easy distribution (often Gaussian).
* $\eta>0$ is the step size.
* Noise term prevents getting stuck in local minima by adding randomness.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Discrete-time Langevin dynamics for EMBs)</span></p>

**Same update in score form** (using $\nabla_x\log p_\phi(x)=-\nabla_x E_\phi(x)$):

$$x_{n+1}=x_n+\eta\nabla_x\log p_\phi(x_n)+\sqrt{2\eta}\varepsilon_n$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation)</span></p>

* **Deterministic part:** takes a small step **toward higher probability density** (gradient ascent on $\log p_\phi$).
* **Stochastic part:** adds Gaussian exploration to cross energy barriers.

$$\boxed{\text{This “score + noise” form is the bridge to diffusion/score-based models.}}$$

</div>


<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Continuous-time Langevin dynamics (SDE limit))</span></p>

As $\eta\to 0$, the discrete updates converge to the **Langevin SDE**:

$$dx(t)=\nabla_x\log p_\phi(x(t))dt+\sqrt{2}dw(t)$$

* $w(t)$ is **standard Brownian motion** (a Wiener process).
* The distribution of $x(t)$ converges (under standard regularity assumptions, e.g. confining smooth $\log p_\phi$ / $E_\phi$) to $p_\phi$ as $t\to\infty$.
  So sampling = simulate this process long enough.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Continuous-time Langevin dynamics for EMBs)</span></p>

Equivalently in energy form:

$$dx(t)=-\nabla_x E_\phi(x(t))dt+\sqrt{2}dw(t)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Langevin sampling works (physics intuition))</span></p>

Think of $E_\phi(x)$ as a **potential energy landscape**.

**According to Newtonian dynamics, the motion of a particle under the force field derived from this energy is described by the ordinary differential equation (ODE). Pure deterministic dynamics (gradient flow / “Newtonian” lens):**

$$dx(t)=-\nabla_x E_\phi(x(t))dt$$

* Always moves “downhill” in energy → ends up in a **local minimum**.
* Bad for sampling multimodal distributions (gets trapped).

**Add noise → Langevin:**

$$dx(t)=-\nabla_x E_\phi(x(t))dt+\sqrt{2}dw(t)$$

* Noise helps escape local minima by crossing energy barriers.
* The stationary distribution becomes the **Boltzmann distribution**:
  
  $$p_\phi(x)\propto e^{-E_\phi(x)}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why it can be hard in practice (inherent challenges))</span></p>

Even though Langevin is conceptually clean, it can be inefficient in high dimensions:
* **Sensitive hyperparameters:** efficiency depends strongly on
  * step size $\eta$,
  * noise scale (here tied to $\eta$),
  * number of iterations.
* **Poor mixing time:** if the target has many separated modes,
  * local stochastic steps struggle to move between distant high-probability regions,
  * mixing gets much worse as dimension grows,
  * can miss diversity (mode dropping in practice).

This motivates **more structured / guided sampling** — which is exactly where diffusion/score-based methods come in (they guide samples through a sequence of easier, noise-smoothed distributions).

</div>

## Mini self-check questions

1. Why does adding a constant to $E_\phi(x)$ not change $p_\phi(x)$?
2. Write $\log p_\phi(x)$ for an EBM and show why the score does not depend on $Z_\phi$.

## From Energy-Based to Score-Based Generative Models

### Big picture

* **Key message:** to *generate* samples (e.g., via Langevin dynamics), you don’t need the full normalized density $p(x)$. You only need the **score**
  
  $$s(x)=\nabla_x \log p(x)$$

  which points toward **higher-probability (higher log-density)** regions.
* **Why move away from energies?**

  * EBMs define $p_\theta(x)\propto e^{-E_\theta(x)}$. The **partition function** is hard, but the **score** is easy:
    
    $$\nabla_x \log p_\theta(x)= -\nabla_x E_\theta(x)\quad(\text{no partition function term in } \nabla_x)$$
    
  * However, **training through an energy** with score matching tends to require **second derivatives** (Hessians).
* **Core shift:** since sampling uses only the score, we can **learn the score directly** with a neural network $s_\phi(x)$. This is the foundation of **score-based generative models**.

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/score_matching.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of Score Matching. The neural network score $s_ϕ(x)$ is trained to match the ground truth score $s(x)$ using a MSE loss. Both are represented as vector fields.</figcaption>
</figure>

### Notation

Let $x\in\mathbb{R}^D$, and $s_\phi(x)\in\mathbb{R}^D$.

* **Score:** $s(x)=\nabla_x\log p_{\text{data}}(x)$
* **Jacobian of a vector field:** $\nabla_x s_\phi(x)\in\mathbb{R}^{D\times D}$ with entries $\frac{\partial (s_\phi)_i}{\partial x_j}$
* **Trace of Jacobian = divergence:**
  
  $$\mathrm{Tr}(\nabla_x s_\phi(x))=\sum_{i=1}^D \frac{\partial (s_\phi)_i}{\partial x_i} = \nabla\cdot s_\phi(x)$$
  
* If $s_\phi=\nabla_x u$ for scalar $u$, then $\nabla_x s_\phi = \nabla_x^2 u$ (the Hessian), and
  
  $$\nabla\cdot s_\phi = \mathrm{Tr}(\nabla_x^2 u)=\Delta u \quad(\text{Laplacian})$$
  

### Learning data score

**Goal:** Approximate the **unknown** true score $s(x)=\nabla_x \log p_{\text{data}}(x)$ from samples $x\sim p_{\text{data}}$ using a neural net $s_\phi(x)$.

**Direct (infeasible):** $\mathcal{L}\_{\mathrm{SM}}(\phi) =\frac{1}{2}\mathbb{E}\_{x\sim p\_{\text{data}}}\Big[\|s_\phi(x)-s(x)\|_2^2\Big]$

**Hyvärinen & Dayan (2005) show:** $\mathcal{L}_{\mathrm{SM}}(\phi)=\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi)+C$

where $C$ does **not** depend on $\phi$, and 

$$\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi) =\mathbb{E}_{x\sim p_{\text{data}}}\left[\mathrm{Tr}\big(\nabla_x s_\phi(x)\big)+\frac{1}{2}\|s_\phi(x)\|_2^2\right]$$

So you can minimize $\tilde{\mathcal{L}}\_{\mathrm{SM}}$ **using only samples** $x\sim p\_{\text{data}}$, without ever knowing the true score.

The **optimal solution** (**minimizer**) is the true score: $s^*(\cdot)=\nabla_x \log p(\cdot)$

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Why this helps computationally)</span></p>

* If you parameterize an **energy** $E_\theta$ and set $s_\theta=-\nabla_x E_\theta$, then
  $\mathrm{Tr}(\nabla_x s_\theta)= -\mathrm{Tr}(\nabla_x^2 E_\theta)$: **second derivatives** of the energy.
* If you parameterize $s_\phi$ **directly**, $\mathrm{Tr}(\nabla_x s_\phi)$ uses **first derivatives** of the score network output w.r.t. input $x$ (still not cheap, but avoids “derivative-of-a-derivative” through an energy).

</div>

### Interpretation of the two terms in $\tilde{\mathcal{L}}_{\mathrm{SM}}$

$$
\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi)
=\mathbb{E}_{p_{\text{data}}}\left[\underbrace{\mathrm{Tr}(\nabla_x s_\phi(x))}_{\text{divergence term}}+\underbrace{\frac{1}{2}\|s_\phi(x)\|_2^2}_{\text{magnitude term}}\right]
$$

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">("Stationarity from the magnitude term")</span></p>

* The expectation is under $p_{\text{data}}$, so **high-density regions dominate**.
* Minimizing $\frac12\| s_\phi(x)\|^2$ pushes
  
  $$s_\phi(x)\to 0 \quad \text{in high-probability regions}$$
  
* Points where $s_\phi(x)=0$ are **stationary points** of the learned flow (no deterministic drift there).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">("Concavity / sinks from the divergence term")</span></p>

* The term $\mathrm{Tr}(\nabla_x s_\phi(x))=\nabla\cdot s_\phi(x)$ encourages **negative divergence** in high-density regions.
* **Negative divergence** means nearby vectors **converge** (flow contracts) rather than spread out → stationary points become **attractive sinks**.

**Making it precise when $s_\phi\approx\nabla_x u$**

Assume $s_\phi=\nabla_x u$. Then:

* $\nabla_x s_\phi = \nabla_x^2 u$ (Hessian)
* $\nabla\cdot s_\phi = \mathrm{Tr}(\nabla_x^2 u)$

At a stationary point $x_\star$ where $\nabla_x u(x_\star)=0$, Taylor expansion:

$$u(x)=u(x_\star)+\frac{1}{2}(x-x_\star)^\top \nabla_x^2u(x_\star)(x-x_\star)+o(\|x-x_\star\|^2)$$

* If $\nabla_x^2u(x_\star)$ is **negative definite**, then $u$ is locally concave → log-density has a **strict local maximum** there.
* Negative definite Hessian $\implies$ all eigenvalues negative $\implies$ trace negative $\implies$ $\mathrm{Tr}(\nabla_x^2u(x_\star))<0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Important nuance)</span></p>

* $\mathrm{Tr}(\nabla_x^2u)<0$ only means the **sum** of eigenvalues is negative.
* Some eigenvalues can still be positive $\implies$ could be a **saddle** rather than a true maximum.

</div>

### Sampling with Langevin dynamics

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Discrete-time Langevin dynamics for trained score)</span></p>

Once you have a trained score model $s_{\phi^*}(x)$, you can sample by iterating:

$$x_{n+1}=x_n+\eta s_{\phi^*}(x_n)+\sqrt{2\eta}\varepsilon_n,\quad \varepsilon_n\sim\mathcal{N}(0,I).$$

* $\eta>0$ is the step size.
* Deterministic part $\eta s(x)$: moves “uphill” in log-density.
* Noise $\sqrt{2\eta}\varepsilon$: keeps exploration and yields the correct stationary distribution (in the idealized limit).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Prologue: Score-Based Generative Models)</span></p>

* The **score function** started as a way to train EBMs efficiently.
* It has become the **central object** in modern **score-based diffusion models**:
  * Theoretical formulation + practical implementation are built around learning scores
  * Generation becomes “simulate (reverse) stochastic processes using learned scores”

</div>

## Denoising Score Matching (DSM) + Sliced Score Matching (Hutchinson)

### Vanilla score matching is hard even with score training

Minimizaing the “direct” score matching loss is infeasible. A classic workaround (Hyvärinen-style) is an equivalent objective that removes the explicit data-score target but introduces a **trace-of-Jacobian** term:

$$\tilde{\mathcal{L}}_{\text{SM}}(\phi)=\mathbb{E}_{x\sim p_{\text{data}}}\Big[\mathrm{Tr}(\nabla_x s_\phi(x))+\frac12\|s_\phi(x)\|_2^2\Big]$$

**Problem:** Computing $\mathrm{Tr}(\nabla_x s_\phi(x))$ (trace of the Jacobian of a $D$-dimensional vector field) has **worst-case complexity $\mathcal{O}(D^2)$** $\implies$ not scalable in high dimensions.

### Sliced score matching via Hutchinson’s estimator

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Hutchinson identity (trace estimator))</span></p>

Let $u \in \mathbb{R}^D$ be **isotropic** with:

$$\mathbb{E}[u]=0,\qquad \mathbb{E}[uu^\top]=I$$

(e.g. **Rademacher** entries $\pm 1$ or standard Gaussian).

Then for any square matrix $A$,

$$\mathrm{Tr}(A)=\mathbb{E}_u[u^\top A u]$$

Also, for any vector $v$,

$$\mathbb{E}_u[(u^\top v)^2] = v^\top\mathbb{E}[uu^\top]v = \|v\|_2^2$$

</div>

#### Applying it to score matching

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sliced Score Matching Objective)</span></p>

Using $A = \nabla_x s_\phi(x)$, the objective becomes (exactly, in expectation):

$$\tilde{\mathcal{L}}_{\text{SM}}(\phi) = \mathbb{E}_{x,u}\Big[u^\top\nabla_x s_\phi(x)u+\frac12 (u^\top s_\phi(x))^2\Big]$$

**Interpretation:** You “test” the model’s behavior only along **random directions** (“random slices”), rather than fully constraining all partial derivatives.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Practical computation (why this helps))</span></p>

* You don’t form the full Jacobian.
* You compute **directional derivatives** and **Jacobian–vector products**:
  * $(\nabla_x s_\phi(x))u$ via JVP
  * then dot with $u$ to get $u^\top(\nabla_x s_\phi(x))u$

If you average over $K$ random probes $u$, you get:
* **unbiased** estimator
* variance $\mathcal{O}(1/K)$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Limitations)</span></p>

Even if it avoids explicit Jacobians:
* It still relies on the **raw data distribution**.
* For image-like data that may lie on a **low-dimensional manifold**, the score $\nabla_x \log p_{\text{data}}(x)$ can be **undefined or unstable**.
* It mainly constrains the vector field **at observed points**, giving weaker control in neighborhoods.
* Has probe-induced variance and repeated JVP/VJP compute costs.

This motivates DSM as a more robust alternative.

</div>

### Training: Denoising Score Matching (DSM): Vincent (2011)

#### Conditioning trick: corrupt the data with known noise

To overcome the intractability of $\nabla_x \log p\_{\text{data}}(x)$, Vincent (2011) proposed injecting noise into the data $x \sim p_{\text{data}}$ via a known conditional distribution $p_\sigma(\tilde x \mid x)$ with scale $σ$.

Introduce a **known corruption kernel**: $\tilde x \sim p_\sigma(\tilde x \mid x)$ where $\sigma>0$ controls noise scale. This defines a **perturbed (smoothed) marginal** distribution:

$$p_\sigma(\tilde x) = \int p_\sigma(\tilde x \mid x)p_{\text{data}}(x)dx$$

The neural network $s_\phi(\tilde x;\sigma)$ is trained to approximate the score of the marginal perturbed distribution: train a model $s_\phi(\tilde x;\sigma)$ to approximate the **score of the marginal**:

$$s_\phi(\tilde x;\sigma)\approx \nabla_{\tilde x}\log p_\sigma(\tilde x)$$

A natural objective is

$$
\mathcal{L}_{\text{SM}}(\phi;\sigma)
= \frac12\mathbb{E}_{\tilde x\sim p_\sigma}\Big[
\|s_\phi(\tilde x;\sigma) - \nabla_{\tilde x}\log p_\sigma(\tilde x)\|_2^2
\Big],
$$

but $\nabla_{\tilde x}\log p_\sigma(\tilde x)$ is still generally intractable.

#### DSM objective (tractable target)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(DSM objective)</span></p>

Vincent’s key result: **condition on the clean sample $x$** and replace the intractable marginal score target with the **conditional score** of the corruption kernel (which we choose and thus know):

$$
\mathcal{L}_{\text{DSM}}(\phi;\sigma)
:= \frac12\mathbb{E}_{x\sim p_{\text{data}},\tilde x\sim p_\sigma(\cdot\mid x)}
\Big[
\|s_\phi(\tilde x;\sigma)-\nabla_{\tilde x}\log p_\sigma(\tilde x\mid x)\|_2^2
\Big]
$$

This is the **denoising** viewpoint: the target $\nabla_{\tilde x}\log p_\sigma(\tilde x\mid x)$ tends to point from noisy $\tilde x$ back toward clean $x$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(DSM is equivalent to marginal score matching)</span></p>

For any fixed $\sigma>0$:

$$\mathcal{L}_{\text{SM}}(\phi;\sigma) = \mathcal{L}_{\text{DSM}}(\phi;\sigma) + C,$$

where $C$ does **not** depend on $\phi$.

So minimizing DSM is effectively minimizing the marginal score matching objective.

Also, the minimizer $s^*(\cdot;\sigma)$ of both losses satisfies (for almost every $\tilde x$):

$$s^*(\tilde x;\sigma)=\nabla_{\tilde x}\log p_\sigma(\tilde x),$$

i.e. the learned model recovers the correct marginal score at that noise level.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Special case: additive Gaussian noise (the key diffusion-model case))</span></p>

Assume the corruption is Gaussian:

$$\tilde x = x + \sigma \varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,I)$$

$$p_\sigma(\tilde x\mid x)=\mathcal{N}(\tilde x;x,\sigma^2 I)$$

**Conditional score has closed form**

$$\nabla_{\tilde x}\log p_\sigma(\tilde x\mid x) = \frac{x-\tilde x}{\sigma^2}$$

Plugging into DSM gives:

$$
\mathcal{L}_{\text{DSM}}(\phi;\sigma)
= \frac12\mathbb{E}_{x,\tilde x}\Big[
\big\|s_\phi(\tilde x;\sigma)-\frac{x-\tilde x}{\sigma^2}\big\|_2^2
\Big]
= \frac12\mathbb{E}_{x,\varepsilon}\Big[
\big\|s_\phi(x+\sigma\varepsilon;\sigma)+\frac{\varepsilon}{\sigma}\big\|_2^2
\Big]
$$

where $\varepsilon \sim \mathcal N(0, I)$.

This objective is the **core of score-based diffusion models**.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/dsm_via_the_conditioning_technique.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of DSM via the conditioning technique. By perturbing the data distribution $p_{\text{data}}$ with small additive Gaussian noise $\mathcal{N}(0,σ^2I)$, the resulting conditional distribution $p_σ(\tilde{x}\mid x) = \mathcal{N}(\tilde{x}; x,σ^2I)$ admits a closed-form score function.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key intuition for $\sigma\to 0$)</span></p>

As $\sigma\approx 0$,
* $p_\sigma(\tilde x)\approx p_{\text{data}}(x)$
* so $s^*(\tilde x;\sigma)=\nabla_{\tilde x}\log p_\sigma(\tilde x)\approx \nabla_x\log p_{\text{data}}(x)$


**Meaning: learning scores of slightly-noised data recovers (approximately) the true data score.**

But it breaks assumption on differentiability of log densities, so implicit and explicit score matching are not equivalent anymore.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Training Gaussian DSM)</span></p>

For a fixed $\sigma$:

1. Sample $x\sim p_{\text{data}}$
2. Sample $\varepsilon\sim\mathcal{N}(0,I)$
3. Form $\tilde x = x+\sigma\varepsilon$
4. Target is $-\varepsilon/\sigma$ (equivalently $(x-\tilde x)/\sigma^2$)
5. Minimize:
   
   $$\frac12\left\|s_\phi(\tilde x;\sigma)+\frac{\varepsilon}{\sigma}\right\|_2^2$$
   

(Extensions usually train over many $\sigma$ values, but that part isn’t shown on these pages.)

</div>
  
<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(DSM is more robust than raw score matching)</span></p>

* Adding noise makes the distribution **smooth/full-dimensional**, which avoids the “score undefined on a manifold” issue.
* Training constrains the score field in **neighborhoods** around data, not only exactly on the data.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conditioning Technique)</span></p>

The conditioning technique also appears in the variational view of diffusion models in DDPM, where conditioning on a data point $x$ turns an intractable loss into a tractable one for Monte Carlo estimation. A similar idea arises in the flow-based perspective.

* You turn an intractable objective involving an unknown marginal score into a tractable regression by conditioning on a clean data point $x$.
* Similar conditioning ideas appear in diffusion-model variational views and also relate to other modern generative-training paradigms.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Advantages of injecting noise)</span></p>

**Why use $p_\sigma$ instead of $p_{\text{data}}$ directly?**

Compared to "vanilla" score matching on the original data distribution, adding Gaussian noise to define $p_\sigma$ gives:
1. **Well-defined gradients (scores exist everywhere)**
   * Real data often lies near a **low-dimensional manifold** in $\mathbb R^D$, so $\nabla\log p_{\text{data}}(x)$ can be ill-behaved/off-manifold.
   * Convolving with Gaussian noise spreads mass over all of $\mathbb R^D$, making $p_\sigma$ have **full support** in $\mathbb R^D$.
   * Therefore the score $\nabla_{\tilde x}\log p_\sigma(\tilde x)$ is (typically) **well-defined everywhere**.
2. **Improved coverage between modes**
   * Noise **smooths** the distribution, filling in low-density “gaps” between separated modes.
   * This improves training signal and helps Langevin dynamics move through low-density regions more effectively (less getting stuck).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sampling with a trained score model)</span></p>

As before, we use discrete time Langevin dynamics:

Given a score model $s_\phi(\cdot;\sigma)$ at a fixed noise level $\sigma$, iterate

$$\tilde x_{n+1} = \tilde x_n + \eta s_\phi(\tilde x_n;\sigma) + \sqrt{2\eta}\varepsilon_n, \qquad \varepsilon_n\sim\mathcal N(0,I)$$

* $\eta>0$ here is the **step size** (careful: later pages reuse $\eta$ for “natural parameter” in exponential families).
* This is Langevin sampling where the “force” term $\nabla \log p_\sigma(\tilde x)$ is replaced by the learned $s_\phi$.

</div>


---

## 3.3.4 Why DSM is denoising: Tweedie’s formula

### Setup (Gaussian corruption with scaling)

Assume:
* $x\sim p_{\text{data}}$
* $\tilde x\mid x \sim \mathcal N(\alpha x,\sigma^2 I)$, with $\alpha\neq 0$

Define the noisy marginal:

$$p_\sigma(\tilde x) = \int \mathcal N(\tilde x;\alpha x,\sigma^2 I)p_{\text{data}}(x)dx$$

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Tweedie’s formula)</span></p>

$$\alpha\mathbb E[x\mid \tilde x] = \tilde x + \sigma^2 \nabla_{\tilde x}\log p_\sigma(\tilde x)$$

Equivalently, the **posterior mean / denoiser** is

$$\mathbb E[x\mid \tilde x] = \frac{1}{\alpha}\Big(\tilde x + \sigma^2 \nabla_{\tilde x}\log p_\sigma(\tilde x)\Big)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition (why this is “denoising”))</span></p>

* The score $\nabla_{\tilde x}\log p_\sigma(\tilde x)$ points toward regions where noisy samples are more likely.
* Moving $\tilde x$ by a step of size $\sigma^2$ in the score direction produces the **conditional mean of the clean signal** (up to the $\alpha$ scaling).

</div>

### Connection to DSM-trained score networks

If DSM gives $s_\phi(\tilde x)\approx \nabla_{\tilde x}\log p_\sigma(\tilde x)$, then an estimated denoiser is:

$$\widehat{x}(\tilde x) = \frac{1}{\alpha}\Big(\tilde x + \sigma^2, s_\phi(\tilde x)\Big)$$

So: **learning the score is (almost directly) learning a denoiser** via Tweedie.

---

## (Optional) Higher-order Tweedie via an exponential-family view

### Exponential family observation model

Assume the conditional law of $\tilde x$ given a latent natural parameter $\eta\in\mathbb R^D$ is

$$q_\sigma(\tilde x\mid \eta) = \exp(\eta^\top \tilde x - \psi(\eta)) q_0(\tilde x)$$

* $q_0(\tilde x)$ is the **base measure** (independent of $\eta$).
* For additive Gaussian noise with variance $\sigma^2 I$,
  
  $$q_0(\tilde x) = (2\pi\sigma^2)^{-D/2}\exp\left(-\frac{\|\tilde x\|^2}{2\sigma^2}\right)$$

Let $p(\eta)$ be a prior over $\eta$. The noisy marginal is

$$p_\sigma(\tilde x) = \int q_\sigma(\tilde x\mid \eta)p(\eta)d\eta$$

Define the “log-normalizer in $\tilde x$”:

$$\lambda(\tilde x) := \log p_\sigma(\tilde x) - \log q_0(\tilde x)$$

Then the posterior has the form

$$p(\eta\mid \tilde x)\propto \exp(\eta^\top \tilde x - \psi(\eta) - \lambda(\tilde x))p(\eta)$$

### Derivatives of $\lambda$ give posterior cumulants

A core exponential-family identity:
* $\nabla_{\tilde x}\lambda(\tilde x) = \mathbb E[\eta\mid \tilde x]$
* $\nabla_{\tilde x}^2\lambda(\tilde x) = \mathrm{Cov}[\eta\mid \tilde x]$
* More generally:
  
  $$\nabla_{\tilde x}^{(k)}\lambda(\tilde x) = \kappa_k(\eta\mid \tilde x),\quad k\ge 3$$
  
  where $\kappa_k$ are conditional cumulants.

### Specialize to Gaussian location noise (recover classic Tweedie + covariance)

For Gaussian location models, one can take $\eta = x/\sigma^2$. Then:

* Posterior mean:
  
  $$\mathbb E[x\mid \tilde x] = \tilde x + \sigma^2 \nabla_{\tilde x}\log p_\sigma(\tilde x)$$
  
* Posterior covariance:
  
  $$\mathrm{Cov}[x\mid \tilde x] = \sigma^2 I + \sigma^4 \nabla_{\tilde x}^2\log p_\sigma(\tilde x)$$
  
* Higher cumulants scale with higher derivatives of $\log p_\sigma(\tilde x)$.

**Takeaway:** not only denoising (mean), but also **uncertainty estimates** (covariance) and higher statistics are encoded in higher-order “scores” (higher derivatives).

---

## Quick “what to remember” checklist

* **Sampling:** $\tilde x_{n+1}=\tilde x_n+\eta s_\phi(\tilde x_n;\sigma)+\sqrt{2\eta}\varepsilon_n$.
* **Why noise helps:** (i) score well-defined everywhere, (ii) smoother landscape improves mode coverage.
* **Tweedie:** $\mathbb E[x\mid \tilde x]=\frac{1}{\alpha}(\tilde x+\sigma^2\nabla_{\tilde x}\log p_\sigma(\tilde x))$.
* **DSM ⇒ denoiser:** replace $\nabla \log p_\sigma$ by $s_\phi$.
* **Higher-order:** derivatives of $\log p_\sigma$ relate to posterior covariance and cumulants.

## Study notes: SURE, Tweedie, and (Generalized) Score Matching

These pages explain two closely related ideas:

1. **SURE** gives an *unbiased, observable* estimate of denoising MSE using only noisy data.
2. The **SURE-optimal denoiser** is the **Bayes posterior mean**, which equals a **score-based correction** (Tweedie). This directly links denoisers $\iff$ scores $\iff$ score matching objectives.
3. **Generalized score matching (GSM)** unifies classical score matching, denoising score matching, and higher-order variants through a general linear operator $\mathcal L$.

---

# 3.3.5 Why DSM is Denoising: SURE

## Setup: additive Gaussian noise

We observe

$$\tilde{\mathbf x}=\mathbf x+\sigma \boldsymbol\epsilon,\qquad \boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I)$$

where $\mathbf x\in\mathbb R^d$ is the unknown clean signal and $\tilde{\mathbf x}$ is noisy.

A **denoiser** is a (weakly differentiable) map

$$\mathbf D:\mathbb R^d\to\mathbb R^d,\qquad \mathbf D(\tilde{\mathbf x})\approx \mathbf x$$

## True denoising quality: conditional MSE risk

For a fixed (unknown) clean $\mathbf x$,

$$R(\mathbf D;\mathbf x):=\mathbb E_{\tilde{\mathbf x}\mid \mathbf x}\Big[\|\mathbf D(\tilde{\mathbf x})-\mathbf x\|_2^2\ \big|\ \mathbf x\Big]$$

Problem: this depends on $\mathbf x$, so you can’t compute it from $\tilde{\mathbf x}$ alone.

---

## SURE: an observable surrogate for the MSE

**Stein’s Unbiased Risk Estimator (SURE)** provides:

$$
\mathrm{SURE}(\mathbf D;\tilde{\mathbf x}) = \|\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x}\|_2^2 + 2\sigma^2\nabla_{\tilde{\mathbf x}}\cdot \mathbf D(\tilde{\mathbf x})- D\sigma^2.
$$

* $\nabla_{\tilde{\mathbf x}}\cdot \mathbf D(\tilde{\mathbf x})$ is the **divergence** of $\mathbf D$:
  
  $$\nabla_{\tilde{\mathbf x}}\cdot \mathbf D(\tilde{\mathbf x})=\sum_{i=1}^d \frac{\partial D_i(\tilde{\mathbf x})}{\partial \tilde{x}_i}$$
  
* Importantly: **SURE depends only on (\tilde{\mathbf x})** (and $\sigma$), not on $\mathbf x$.

### Why the terms make sense (intuition)

* $\|\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x}\|^2$: how much the denoiser changes the input.
  * Alone, it *underestimates* true error because $\tilde{\mathbf x}$ is already corrupted.
* $2\sigma^2 \nabla\cdot \mathbf D(\tilde{\mathbf x})$: **correction term** accounting for noise variance via sensitivity of $\mathbf D$.
* $-d\sigma^2$: constant offset that fixes the bias.

---

## Unbiasedness property (the key guarantee)

For any fixed but unknown $\mathbf x$,

$$\mathbb E_{\tilde{\mathbf x}\mid \mathbf x}\big[\mathrm{SURE}(\mathbf D;\mathbf x+\sigma\epsilon)\ \big|\ \mathbf x\big] = R(\mathbf D;\mathbf x)$$

So **minimizing SURE (in expectation or empirically)** is equivalent to minimizing the true denoising MSE risk, while using only noisy data.

### Derivation sketch (how Stein’s identity enters)

Start from:

$$\|\mathbf D(\tilde{\mathbf x})-\mathbf x\|^2 = \|\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x} + (\tilde{\mathbf x}-\mathbf x)\|^2$$

Expand and use $\tilde{\mathbf x}-\mathbf x=\sigma\epsilon$. The cross-term contains $\mathbb E[\epsilon^\top g(\mathbf x+\sigma\epsilon)]$ with $g(\tilde{\mathbf x})=\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x}$. Stein’s lemma converts this to a divergence term:

$$\mathbb E[\epsilon^\top g(\mathbf x+\sigma\epsilon)] = \sigma\mathbb E[\nabla_{\tilde{\mathbf x}}\cdot g(\tilde{\mathbf x})]$$

Since $\nabla\cdot(\tilde{\mathbf x})=d$, you get exactly the SURE formula.

---

# Link to Tweedie’s formula and Bayes optimality

## Noisy marginal

Let the noisy marginal be the convolution:

$$p_\sigma(\tilde{\mathbf x}) := (p_{\text{data}} * \mathcal N(0,\sigma^2\mathbf I))(\tilde{\mathbf x})$$

## SURE minimization ⇒ Bayes optimal denoiser

SURE is unbiased *w.r.t. noise* conditional on $\mathbf x$:

$$
\mathbb E_{\tilde{\mathbf x}\mid \mathbf x}[\mathrm{SURE}(\mathbf D;\tilde{\mathbf x})] = \mathbb E_{\tilde{\mathbf x}\mid \mathbf x}\big[\|\mathbf D(\tilde{\mathbf x})-\mathbf x\|^2\big].
$$

Averaging also over $\mathbf x\sim p_{\text{data}}$ gives the **Bayes risk**:

$$
\mathbb E_{\mathbf x,\tilde{\mathbf x}}\big[\|\mathbf D(\tilde{\mathbf x})-\mathbf x\|^2\big] = \mathbb E_{\tilde{\mathbf x}}\Big[\mathbb E_{\mathbf x\mid \tilde{\mathbf x}} \|\mathbf D(\tilde{\mathbf x})-\mathbf x\|^2\Big].
$$

This decomposes pointwise in $\tilde{\mathbf x}$, so the optimal denoiser is:

$$\mathbf D^*(\tilde{\mathbf x})=\mathbb E[\mathbf x\mid \tilde{\mathbf x}]$$

## Tweedie’s identity: posterior mean = score correction

A central identity:

$$\mathbf D^*(\tilde{\mathbf x})=\mathbb E[\mathbf x\mid \tilde{\mathbf x}] = \tilde{\mathbf x}+\sigma^2\nabla_{\tilde{\mathbf x}}\log p_\sigma(\tilde{\mathbf x})$$

So the Bayes-optimal denoiser equals **input + $\sigma^2$ times the noisy score**.

---

# Relationship between SURE and score matching

## Parameterize denoiser via a score field

Motivated by Tweedie:

$$\mathbf D(\tilde{\mathbf x}) = \tilde{\mathbf x}+\sigma^2 \mathbf s_\phi(\tilde{\mathbf x};\sigma)$$

where $\mathbf s_\phi(\cdot;\sigma)\approx \nabla_{\tilde{\mathbf x}}\log p_\sigma(\cdot)$.

## Plugging into SURE yields Hyvärinen’s objective (up to constants)

Substitute into SURE and simplify:

$$
\frac{1}{2\sigma^4}\mathrm{SURE}(\mathbf D;\tilde{\mathbf x}) = \mathrm{Tr}\big(\nabla_{\tilde{\mathbf x}}\mathbf s_\phi(\tilde{\mathbf x};\sigma)\big)
+
\frac12\|\mathbf s_\phi(\tilde{\mathbf x};\sigma)\|_2^2
+
\text{const}(\sigma)
$$

Taking expectation over $\tilde{\mathbf x}\sim p_\sigma$, minimizing SURE is equivalent (up to an additive constant) to minimizing **Hyvärinen’s alternative score matching objective** at noise level $\sigma$.
**Conclusion:** SURE and score matching share the same minimizer, corresponding to the denoiser $\tilde{\mathbf x}+\sigma^2\nabla \log p_\sigma(\tilde{\mathbf x})$.

---

# 3.3.6 Generalized Score Matching (GSM)

## Motivation: unify many “score-like” training targets

Classical score matching, denoising score matching, and higher-order variants all target a quantity of the form

$$\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}$$

for some **linear operator** $\mathcal L$ acting on the density $p$.

* Classical case $\mathcal L=\nabla_{\mathbf x}$:
  
  $$\frac{\mathcal L p}{p}=\frac{\nabla p}{p}=\nabla \log p$$

Key idea: the $\frac{\mathcal L p}{p}$ structure enables **integration by parts** to remove unknown normalizing constants, producing a tractable objective depending only on samples and the learned field.

---

## Generalized Fisher divergence

Let $p$ be data and $q$ be a model density. Define

$$
\mathcal D_{\mathcal L}(p\parallel q) := \int p(\mathbf x)\left\|\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}-\frac{\mathcal L q(\mathbf x)}{q(\mathbf x)}\right\|_2^2 d\mathbf x
$$

If $\mathcal L$ is **complete** (informally: $\frac{\mathcal L p_1}{p_1}=\frac{\mathcal L p_2}{p_2}$ a.e. implies $p_1=p_2$ a.e.), then $\mathcal D_{\mathcal L}(p\parallel q)=0$ identifies $q=p$.
For $\mathcal L=\nabla$, this recovers the classical Fisher divergence.

---

## Score parameterization (avoid explicit normalized $q$)

Instead of modeling $q$, directly learn a vector field $\mathbf s_\phi(\mathbf x)$ to approximate $\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}$:

$$
\mathcal D_{\mathcal L}(p\parallel \mathbf s_\phi) := \mathbb E_{\mathbf x\sim p}\left[\left\|\mathbf s_\phi(\mathbf x)-\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}\right\|_2^2\right].
$$

The target is unknown, but integration by parts makes the loss computable.

### Adjoint operator and integration by parts trick

Define the adjoint $\mathcal L^\dagger$ by:

$$\int (\mathcal L f)^\top g = \int f (\mathcal L^\dagger g) \quad \text{for all test functions } f,g$$

(assuming boundary terms vanish).

Expanding the square and applying the adjoint identity yields the tractable objective:

$$
\mathcal L_{\text{GSM}}(\phi) = \mathbb E_{\mathbf x\sim p}\Big[\frac12\|\mathbf s_\phi(\mathbf x)\|_2^2-(\mathcal L^\dagger \mathbf s_\phi)(\mathbf x)\Big]
+\text{const},
$$

where “const” does not depend on $\phi$.

### Check: recovering Hyvärinen’s score matching

For $\mathcal L=\nabla$, we have $\mathcal L^\dagger=-\nabla\cdot$ (negative divergence), so:

$$\mathbb E_p\Big[\tfrac12\|\mathbf s_\phi\|^2-(\mathcal L^\dagger \mathbf s_\phi)\Big] = \mathbb E_p\Big[\tfrac12\|\mathbf s_\phi\|^2+\nabla\cdot\mathbf s_\phi\Big]$$

which is Hyvärinen’s classical objective.

---

# Examples of operators $\mathcal L$

## 1) Classical score matching

Take $\mathcal L=\nabla_{\mathbf x}$. Then

$$\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}=\nabla_{\mathbf x}\log p(\mathbf x)$$

## 2) Denoising score matching (Gaussian corruption)

For additive Gaussian noise at level $\sigma$, define an operator on scalar $f$:

$$(\mathcal L f)(\tilde{\mathbf x})=\tilde{\mathbf x}f(\tilde{\mathbf x})+\sigma^2\nabla_{\tilde{\mathbf x}} f(\tilde{\mathbf x})$$

Then

$$\frac{\mathcal L p_\sigma(\tilde{\mathbf x})}{p_\sigma(\tilde{\mathbf x})} = \tilde{\mathbf x}+\sigma^2\nabla_{\tilde{\mathbf x}}\log p_\sigma(\tilde{\mathbf x})\mathbb E[\mathbf x_0\mid \tilde{\mathbf x}]$$

which is exactly **Tweedie’s identity**. Minimizing $\mathcal L_{\text{GSM}}$ with this operator trains $\mathbf s_\phi$ to approximate the **denoiser**, recovering denoising score matching behavior.

## 3) Higher-order targets

By stacking derivatives inside $\mathcal L$, you can target:

* $\nabla^2 \log p$ (Hessian of log-density),
* higher derivatives,
  which relate to **posterior covariance** and higher-order cumulants.

---

# Key takeaways / mental model

* **SURE** lets you estimate denoising MSE without clean targets; its correction term is a **divergence**.
* Minimizing expected **SURE** yields the **posterior mean denoiser**:
  
  $$\mathbf D^*(\tilde{\mathbf x})=\mathbb E[\mathbf x\mid\tilde{\mathbf x}]$$
  
* **Tweedie** rewrites that denoiser using the **score of the noisy marginal**:
  
  $$\mathbb E[\mathbf x\mid\tilde{\mathbf x}] = \tilde{\mathbf x}+\sigma^2\nabla \log p_\sigma(\tilde{\mathbf x})$$
  
* Parameterizing $\mathbf D(\tilde{\mathbf x})=\tilde{\mathbf x}+\sigma^2\mathbf s_\phi(\tilde{\mathbf x};\sigma)$ turns SURE minimization into (alternative) **score matching** (up to constants).
* **Generalized score matching**: pick an operator $\mathcal L$; learn $\mathbf s_\phi \approx \mathcal Lp/p$; integration by parts gives a tractable loss. This **unifies** classical SM, DSM, and higher-order variants.


## Multi-Noise Denoising Score Matching (NCSN) + Annealed Langevin Dynamics (Sections 3.4–3.6)

### Big picture

* **Goal (score-based generative modeling):** learn the **score**
  
  $$\nabla_x \log p(x)$$
  
  (gradient of log-density), which lets you **generate samples** by running dynamics that follow this gradient plus noise (e.g., Langevin).
* **Problem:** learning / sampling with a **single** noise level is unreliable and slow.
* **Fix (NCSN, Song & Ermon 2019):** train **one network conditioned on noise level** to estimate scores for **many noise scales**, then sample by **annealing** from high noise → low noise.

### Multi-Noise Levels of Denoising Score Matching (NCSN)

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/score_matching_inaccuracy.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of SM inaccuracy (revisiting Illustration of Score Matching). the red region indicates low-density areas with potentially inaccurate score estimates due to limited sample coverage, while high-density regions tend to yield more accurate estimates.</figcaption>
</figure>

### Motivation: why one noise level is not enough

Adding Gaussian noise “smooths” the data distribution, but:

* **Low noise (small variance):**
  * Distribution is sharp/multi-modal; **Langevin struggles to move between modes**.
  * In low-density regions, the score can be inaccurate and gradients can vanish → **poor exploration**.
* **High noise (large variance):**
  * Sampling/mixing is easier, but the model captures only **coarse structure** → samples look **blurry**, lose fine detail.
* **High-dimensional issues:** Langevin can be **slow**, sensitive to **initialization**, can get stuck near **plateaus/saddles**.

**Core idea:** use **multiple noise levels**:
* High noise: explore globally / cross modes.
* Low noise: refine details.

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/ncsn.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of NCSN. The forward process perturbs the data with multiple levels of additive Gaussian noise $p_σ(x_σ\mid x)$. Generation proceeds via Langevin sampling at each noise level, using the result from the current level to initialize sampling at the next lower variance.</figcaption>
</figure>

### Training

### Noise levels

Choose a sequence of noise scales:

$$0 < \sigma_1 < \sigma_2 < \cdots < \sigma_L$$

* $\sigma_1$: small enough to preserve fine details
* $\sigma_L$: large enough to heavily smooth the distribution (easier learning)

### Forward perturbation (data → noisy)

Sample clean $x \sim p_{\text{data}}$. Create noisy version:

$$x_\sigma = x + \sigma \epsilon, \quad \epsilon \sim \mathcal N(0, I)$$

**Perturbation kernel:**

$$p_\sigma(x_\sigma \mid x) := \mathcal N(x_\sigma; x, \sigma^2 I)$$

**Marginal (smoothed) distribution at noise $\sigma$:**

$$p_\sigma(x_\sigma) = \int p_\sigma(x_\sigma\mid x)p_{\text{data}}(x)dx$$

**Interpretation:** $p_\sigma$ is a Gaussian-smoothed version of $p_{\text{data}}$. Larger $\sigma$ $\implies$ smoother.

### Noise-conditional score network

Train a single network $s_\phi(x,\sigma)$ to approximate:

$$s_\phi(x,\sigma) \approx \nabla_x \log p_\sigma(x)$$

---

## Training objective of NCSN (DSM across all noise levels)

### Weighted multi-noise DSM loss

$$\mathcal L_{\text{NCSN}}(\phi) := \sum_{i=1}^{L}\lambda(\sigma_i)\mathcal L_{\text{DSM}}(\phi;\sigma_i)$$

where

$$
\mathcal L_{\text{DSM}}(\phi;\sigma) = \frac12\mathbb E_{x\sim p_{\text{data}},\tilde x \sim p_\sigma(\tilde x\mid x)}
\left[
\left\|
s_\phi(\tilde x,\sigma)-\frac{x-\tilde x}{\sigma^2}
\right\|_2^2
\right]
$$

* $\lambda(\sigma_i)>0$: weight per scale (balances contributions of different noise levels).

### Key fact (optimal solution)

Minimizing DSM at each $\sigma$ yields:

$$s^*(\cdot,\sigma) = \nabla_x \log p_\sigma(\cdot), \quad \forall \sigma \in {\sigma_i}_{i=1}^L$$

So you learn the **true score of the smoothed distribution** at every noise scale.

---

## Relationship to DDPM loss (Tweedie connection)

Let $x_\sigma = x + \sigma \epsilon$, $\epsilon\sim \mathcal N(0,I)$. By **Tweedie’s formula**:

$$\nabla_{x_\sigma}\log p_\sigma(x_\sigma) = -\frac{1}{\sigma}\mathbb E[\epsilon \mid x_\sigma]$$

So:

* NCSN’s target (score) is proportional to the **posterior mean noise**.
* If a DDPM-style model predicts $\epsilon^*(x_\sigma,\sigma)=\mathbb E[\epsilon\mid x_\sigma]$, then:
  
$$
s^*(x_\sigma,\sigma)= -\frac{1}{\sigma}\epsilon^*(x_\sigma,\sigma),
\quad
\epsilon^*(x_\sigma,\sigma)= -\sigma s^*(x_\sigma,\sigma)
$$

**Discrete DDPM notation shown:**

$$x_i = \bar\alpha_i x_0 + \sqrt{1-\bar\alpha_i}\epsilon$$

then similarly:

$$s^*(x_i,i)= -\frac{1}{\sigma_i}\mathbb E[\epsilon\mid x_i]$$

**Takeaway:** *Noise-prediction (DDPM) and score-prediction (NCSN) are the same information, just scaled/parameterized differently.*

---

## 3.4.3 Sampling — Annealed Langevin Dynamics (ALD)

### Why annealing helps

* At large $\sigma$, $p_\sigma$ is smooth ⇒ sampling is easier (better mixing).
* Gradually reduce $\sigma$ and **refine** samples using the next score model.
* Each stage uses the previous stage’s output as a strong initialization.

### Langevin update at noise level $\sigma_\ell$

Given current $\tilde x_n$:

$$\tilde x_{n+1} = \tilde x_n + \eta_\ell s_\phi(\tilde x_n,\sigma_\ell) + \sqrt{2\eta_\ell}\epsilon_n, \quad \epsilon_n\sim\mathcal N(0,I)$$

### Algorithm (as given)

* Initialize $x^{\sigma_L}\sim\mathcal N(0,I)$ (often equivalent to choosing a large-noise prior).
* For $\ell = L, L-1,\dots,2$:
  * run $N_\ell$ Langevin steps using $s_\phi(\cdot,\sigma_\ell)$
  * set $x^{\sigma_{\ell-1}}\leftarrow$ final sample (init for next level)
* Output $x^{\sigma_1}$.

**Step size scaling (typical):**

$$\eta_\ell = \delta\cdot \frac{\sigma_\ell^2}{\sigma_1^2},\quad \delta>0$$

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

Bigger noise $\implies$ you can take bigger steps.

</div>

---

## Why NCSN sampling is slow (important bottleneck)

NCSN sampling uses **annealed MCMC** across scales $\lbrace\sigma_i\rbrace_{i=1}^L$. If you do $K$ updates per scale, you need $\sim L\times K$ network evaluations.

Two reasons $L\times K$ must be large:

1. **Local accuracy & stability:** learned score is only reliable locally → requires small step sizes and many steps to avoid bias/instability.
2. **Slow mixing in high dimensions:** local MCMC moves explore multi-modal high-D distributions inefficiently → many iterations to reach typical regions.

Overall cost:

$$\mathcal O(LK)$$

sequential network passes ⇒ computationally slow.

---

## 3.5 Summary: Comparative view of NCSN and DDPM

### Forward / corruption process (conceptual comparison)

* **NCSN:** additive Gaussian noise at multiple scales. The table shows transitions like:

  $$x_{i+1} = x_i + \sqrt{\sigma_{i+1}^2 - \sigma_i^2}\epsilon$$
  
  (incrementally increasing variance).
* **DDPM:** Markov chain with variance schedule $\beta_i$:
  
  $$x_{i+1} = \sqrt{1-\beta_i}x_i + \sqrt{\beta_i}\epsilon$$

### Loss / training target

* **NCSN:** score loss equivalent to
  
  $$\mathbb E\big[\|s_\phi(x_i,\sigma_i) + \epsilon/\sigma_i\|^2\big]$$
  
  (score matches scaled negative noise).
* **DDPM:** noise prediction loss
  
  $$\mathbb E\big[\|\epsilon_\phi(x_i,i)-\epsilon\|^2\big]$$

### Sampling

* **NCSN:** Langevin per noise “layer”; output initializes next lower noise.
* **DDPM:** traverse learned reverse chain $p_\phi(x_{i-1}\mid x_i)$.

### Shared bottleneck

Both rely on **dense discretization** ⇒ often **hundreds/thousands** of steps ⇒ slow generation.

**Question 3.5.1:** *How can we accelerate sampling in diffusion models?*
(Flag for later chapters on faster solvers / fewer steps.)

---

## 3.6 Closing remarks (what this chapter sets up)

* Score-based view comes from EBMs: score avoids dealing directly with the **intractable partition function**.
* Progression:

  1. score matching →
  2. **denoising score matching (DSM)** via noise perturbation →
  3. **Tweedie’s formula** connects score estimation to denoising →
  4. extend from single noise to **NCSN** (multi-noise) + **annealed Langevin**.
* Key convergence: **NCSN and DDPM** look different but share structure and **same bottleneck** (slow sequential sampling).
* Next step: move to **continuous time**, unify methods as discretizations of a **Score SDE**, and connect variational + score-based views via differential equations (motivates advanced numerical methods to speed up sampling).

---

## Quick “exam-ready” checklist

* Can you write:

  * $x_\sigma = x + \sigma\epsilon$ and $p_\sigma(x_\sigma\mid x)=\mathcal N(x, \sigma^2I)$?
  * $p_\sigma(x)=\int p_\sigma(x\mid x_0)p_{\text{data}}(x_0)dx_0$?
  * DSM loss target $(x-\tilde x)/\sigma^2$?
  * Multi-noise objective $\sum_i \lambda(\sigma_i)\mathcal L_{\text{DSM}}(\sigma_i)$?
  * Langevin update $\tilde x_{n+1}=\tilde x_n+\eta s_\phi(\tilde x_n,\sigma)+\sqrt{2\eta}\epsilon$?
* Can you explain (in words) why:

  * low noise ⇒ hard mode traversal; high noise ⇒ blurry?
  * annealing helps?
  * sampling cost is $\mathcal O(LK)$?
* Can you derive the DDPM/NCSN link:

$$
\nabla\log p_\sigma(x_\sigma)=-(1/\sigma)\mathbb E[\epsilon\mid x_\sigma]  \Rightarrow  \epsilon^*=-\sigma s^* ; ?
$$


<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/discrete_time_noise_adding_step.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of the discrete-time noise-adding step. It adds noise from tto $t+ ∆t$ with mean drift $f(xt,t)$ and diffusion coefficient $g(t)$.</figcaption>
</figure>
<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/forward_process_in_a_diffusion_model.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>(1D) Visualization of the forward process in a diffusion model. The process starts from initial points sampled (denoted as "$\times$") from a complex bimodal data distribution ($p_0 = p_{\text{data}}$) and evolves toward a simple, unimodal Gaussian prior ($p_T \approx p_{\text{prior}}$). The background heatmap illustrates the evolving marginal probability density, $p_t$, which smooths over time. Sample trajectories are shown evolving from $t = 0$ to $t= T$, comparing the stochastic forward SDE process (blue paths) with its deterministic counterpart, the PF-ODE (white paths). Note that the PF-ODE is a deterministic transport map for densities, not generally the mean of sample paths started from a single point.</figcaption>
</figure>
<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/reverse_time_stochastic_process_for_data_generation.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Visualization of the reverse-time stochastic process for data generation. It begins from samples drawn from a simple prior distribution ($p_{\text{prior}})$ at $t=T$ (denoted as "$\times$"), which are evolved backward in time using a reverse-SDE. The resulting trajectories terminate at $t=0$ and collectively form the target bimodal data distribution ($p_0 = p_{\text{data}}$). The background heatmap illustrates how the probability density is gradually transformed from a simple Gaussian into the complex target distribution.</figcaption>
</figure>
<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/temporal_evolution_of_the_marginal_density.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>(2D) Temporal evolution of the marginal density $p_t$. The forward SDE has $f ≡ 0$ and $g(t) = \sqrt{2t}$ on $[0,T]$. It starts with $p_0 = p_{\text{data}}$ a two-mode Gaussian mixture and ends at $p_T ≈ p_{\text{prior}} := \mathcal{N}(0,T^2I)$. The temporal-spatial evolution of $p_t$ follows the Fokker–Planck equation.</figcaption>
</figure>


# Score SDE Framework

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Why "Score SDE")</span></p>

You’ve seen diffusion models in **discrete time** (e.g., **DDPM**) and in the **score-based / noise-conditional** view (e.g., **NCSN**). The **Score SDE framework** is the **continuous-time limit** that **unifies** them.

Key idea:
* The forward “add-noise” process can be written as a **(stochastic) differential equation**.
* **Generation (sampling)** becomes “solve a differential equation backward in time”.
* This gives a clean mathematical foundation and lets you use tools from **numerical analysis** (Euler/Euler–Maruyama, better solvers, stability/efficiency ideas).

</div>

## Discrete-time noise injection: two classic forms

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name">(NCSN (noise-conditional score networks))</span></p>

NCSN uses a **sequence of noise levels** $\lbrace\sigma_i\rbrace_{i=1}^L$.
A clean sample $x \sim p_{\text{data}}$ is perturbed by

$$
x_{\sigma_i} = x + \sigma_i ,\varepsilon_i,
\qquad \varepsilon_i \sim \mathcal N(0, I).
$$

**Interpretation:** you can think of a “time” index where the **noise level increases** as time increases.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Recall</span><span class="math-callout__name">(DDPM (incremental noising with a variance schedule))</span></p>

DDPM injects noise **step-by-step** with a schedule $\lbrace\beta_i\rbrace_{i=1}^L$:

$$
x_i = \sqrt{1-\beta_i^2}x_{i-1} + \beta_i \varepsilon_i,
\qquad \varepsilon_i \sim \mathcal N(0, I).
$$

**Interpretation:** each step slightly shrinks the signal and adds fresh Gaussian noise.

</div>

### A unified “small step” view on a time grid

Consider a discrete time grid with step $\Delta t$. The update from $x_t$ to $x_{t+\Delta t}$ can be written in a common pattern.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(NCSN viewed as a small step)</span></p>

A step increases the noise level from $\sigma_t$ to $\sigma_{t+\Delta t}$:

$$
x_{t+\Delta t} = x_t + \sqrt{\sigma_{t+\Delta t}^2-\sigma_t^2}\varepsilon_t
\approx x_t + \sqrt{\frac{d\sigma^2(t)}{dt}\Delta t}\varepsilon_t
$$

So in the unified form (below), NCSN corresponds to:

* **drift** $f(x,t)=0$
* **diffusion** $g(t)=\sqrt{\frac{d,\sigma^2(t)}{dt}}$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(DDPM viewed as a small step)</span></p>

For small $\Delta t$, you can write a step (conceptually) as

$$
x_{t+\Delta t} \approx \sqrt{1-\beta(t)\Delta t},x_t + \sqrt{\beta(t)\Delta t},\varepsilon_t
\approx x_t - \tfrac12 \beta(t)x_t,\Delta t + \sqrt{\beta(t)\Delta t},\varepsilon_t.
$$

So DDPM corresponds to:
* **drift** $f(x,t)= -\tfrac12 \beta(t)x$
* **diffusion** $g(t)=\sqrt{\beta(t)}$

(The approximation uses $\sqrt{1-u}\approx 1-\tfrac12 u$ for small $u$.)

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Common structural pattern (discrete-time))</span></p>

Both processes match:

$$
x_{t+\Delta t} \approx x_t + f(x_t,t),\Delta t + g(t)\sqrt{\Delta t},\varepsilon_t,
\qquad \varepsilon_t \sim \mathcal N(0,I)
\text{(4.1.1)}
$$

Notation:

* $x_t \in \mathbb R^D$
* $f:\mathbb R^D\times \mathbb R \to \mathbb R^D$ (drift)
* $g:\mathbb R \to \mathbb R$ (diffusion “strength”)


</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(NCSN and DDPM are Gaussian transitions)</span></p>

This discrete update corresponds to a Gaussian conditional:

$$
p(x_{t+\Delta t}\mid x_t) := \mathcal N\Big(x_{t+\Delta t};; x_t + f(x_t,t)\Delta t,; g(t)^2\Delta t,I\Big), \qquad \text{(4.1.2)}
$$

Interpretation:
* Mean moves by the deterministic drift $f(x_t,t)\Delta t$
* Covariance is $g(t)^2\Delta t,I$

**Note:** we treat $x_t$ as a fixed sample and $x_{t+\delta t}$ as a random variable.

</div>


### Continuous-time limit: from discrete updates to an SDE

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Discrete updates converge to SDE)</span></p>

As $\Delta t \to 0$ (think: **infinitely many infinitesimal noise layers**), the process converges to the **forward-time SDE**:

$$d x(t) = f(x(t),t)dt + g(t)dw(t),$$

where $w(t)$ is a **standard Wiener process** (Brownian motion).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Infinitesimal increment)</span></p>

The **infinitesimal increment** is defined as:

$$dw(t) := w(t+dt)-w(t), \qquad dw(t)\sim \mathcal N(0, dt,I)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Wiener process)</span></p>

* $w(0)=0$
* **independent increments**
* for $s<t$: $w(t)-w(s)\sim \mathcal N(0,(t-s)I)$
* continuous paths, but *nowhere differentiable* (so $dw/dt$ doesn’t exist)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How to "match" discrete and continuous symbols)</span></p>

The excerpt gives the conceptual correspondences:

* $x(t+\Delta t)-x(t) \approx dx(t)$
* $\Delta t \approx dt$
* $\sqrt{\Delta t}\varepsilon_t \approx dw(t)$

This is the key intuition for why the noise term scales like $\sqrt{dt}$: many tiny Gaussian kicks accumulate into Brownian motion.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this matters for diffusion models (as stated in the excerpt))</span></p>

Once you specify **drift** $f(x,t)$ and **diffusion** $g(t)$, the forward SDE (data $\to$ noise) automatically induces a **reverse-time SDE** that transports **terminal noise back to the data distribution**.

Crucial point highlighted:
* The reverse dynamics involve (essentially) one unknown ingredient: the **score function** at each time/noise level.
* This motivates **score matching** as the training objective.
* Once the score is learned, **sampling** becomes **numerically integrating** the reverse-time SDE using that learned score.

</div>

### Quick reference: mapping 

| Framework            | Discrete small-step form                                                                           | Drift $f(x,t)$        | Diffusion $g(t)$                  |
| -------------------- | -------------------------------------------------------------------------------------------------- | --------------------- | --------------------------------- |
| **NCSN**             | $x_{t+\Delta t}\approx x_t + \sqrt{\tfrac{d\sigma^2}{dt}\Delta t},\varepsilon_t$                   | $0$                   | $\sqrt{\tfrac{d\sigma^2(t)}{dt}}$ |
| **DDPM**             | $x_{t+\Delta t}\approx x_t - \tfrac12 \beta(t)x_t\Delta t + \sqrt{\beta(t)\Delta t},\varepsilon_t$ | $-\tfrac12 \beta(t)x$ | $\sqrt{\beta(t)}$                 |
| **Unified**          | $x_{t+\Delta t}\approx x_t + f(x_t,t)\Delta t + g(t)\sqrt{\Delta t}\varepsilon_t$                  | general               | general                           |
| **Continuous limit** | $dx(t)=f(x(t),t)dt+g(t)dw(t)$                                                                      | same $f$              | same $g$                          |


## Forward-Time SDEs: From Data to Noise

### Big picture

Discrete-time diffusion methods (e.g., **NCSN** and **DDPM**) can be viewed in a **continuous-time** framework by defining a stochastic process $\mathbf{x}(t)$ on $t\in[0,T]$ that gradually corrupts data into noise via a **forward SDE**.

---

### Forward SDE (data $\to$ noise)

$$
d\mathbf{x}(t)=\mathbf{f}(\mathbf{x}(t),t),dt + g(t),d\mathbf{w}(t),
\qquad \mathbf{x}(0)\sim p_{\text{data}}.
\qquad\text{(4.1.3)}
$$

**Objects**

* $\mathbf{f}(\cdot,t):\mathbb{R}^D\to\mathbb{R}^D$: **drift** (deterministic trend).
* $g(t)\in\mathbb{R}$: **scalar diffusion coefficient** (noise strength schedule).
* $\mathbf{w}(t)$: standard **Wiener process** (Brownian motion).

**Interpretation**

* Once $\mathbf{f}$ and $g$ are chosen, the forward process is fully specified.
* It describes how clean data is progressively corrupted by injecting **Gaussian noise** over time.

---

### Figure intuition (forward process)

* At $t=0$: distribution is complex, e.g. bimodal $p_0=p_{\text{data}}$.
* As $t$ increases: the **marginal density** $p_t$ “smooths out”.
* At $t=T$: $p_T\approx p_{\text{prior}}$ (typically a simple Gaussian).

**PF-SDE vs PF-ODE paths (from the figure caption)**

* **PF-SDE**: sample trajectories are stochastic (wiggly).
* **PF-ODE**: deterministic counterpart gives a **transport map for densities**; it is *not* generally the mean of SDE sample paths from a single initial point.

---

## Perturbation kernels and marginals

### Perturbation kernel

$$p_t(\mathbf{x}_t\mid \mathbf{x}_0)$$

describes how one clean sample $\mathbf{x}_0\sim p_{\text{data}}$ becomes a noisy $\mathbf{x}_t$ at time $t$.

### Marginal density (mixture over data)

$$
p_t(\mathbf{x}_t)=\int p_t(\mathbf{x}_t\mid \mathbf{x}_0),p_{\text{data}}(\mathbf{x}_0),d\mathbf{x}_0,
\qquad (p_0=p_{\text{data}}).
\qquad\text{(4.1.5)}
$$

So $p_t$ is a (generally complicated) mixture induced by the kernel + the data distribution.

---

## Affine drift special case (closed-form Gaussian kernels)

A common analytically convenient assumption is that drift is **linear in $\mathbf{x}$**:

$$
\mathbf{f}(\mathbf{x},t)=f(t),\mathbf{x},
\qquad\text{(4.1.4)}
$$

where $f(t)$ is scalar (typically **non-positive**, so the signal decays).

### Consequence: Gaussian conditional at every time

Under this structure, the process stays Gaussian conditionally:

$$p_t(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N}\big(\mathbf{x}_t;,\mathbf{m}(t),,P(t)\mathbf{I}_D\big)$$

with

$$
\mathbf{m}(t)=\exp\Big(\int_0^t f(u),du\Big)\mathbf{x}_0,
\qquad
P(t)=\int_0^t \exp\Big(2\int_s^t f(u),du\Big),g^2(s)ds,
$$

and initial conditions $\mathbf{m}(0)=\mathbf{x}_0,;P(0)=0$.

**Why this matters**

* You can sample $\mathbf{x}_t\mid \mathbf{x}_0$ **directly** without numerically simulating the SDE (“simulation-free”).
* Both **NCSN** and **DDPM** fall into this affine-drift setting (in the continuous-time view).

---

## Convergence to a simple prior

By choosing $f(t)$ and $g(t)$ appropriately, the forward diffusion eventually “forgets” the initial condition.

### Mean decays (forgetting $\mathbf{x}_0$)

If $f(u)\le 0$,

$$
\mathbf{m}(T)=\exp\Big(\int_0^T f(u),du\Big)\mathbf{x}_0 \to 0
\quad \text{as }T\to\infty,
$$

so dependence on $\mathbf{x}_0$ vanishes.

### Marginal approaches a prior

As the conditional becomes independent of $\mathbf{x}_0$, the marginal simplifies:

$$
p_T(\mathbf{x}_T)\approx p_{\text{prior}}(\mathbf{x}_T),
\qquad
p_T(\mathbf{x}_T\mid \mathbf{x}_0)\approx p_{\text{prior}}(\mathbf{x}_T).
$$

Thus, the forward SDE maps a complex data distribution into a tractable prior, giving a clean starting point for *reversal/generation*.

---

# 4.1.3 Reverse-Time Stochastic Process for Generation

### Goal

Generate data by “reversing” the forward corruption:

* Start at $t=T$ from $\mathbf{x}_T\sim p_{\text{prior}}\approx p_T$,
* Evolve **backward** to $t=0$ to obtain a sample from $p_{\text{data}}$.

### Why reversing is subtle for SDEs

* For ODEs: time reversal is basically tracing trajectories backward.
* For SDEs: individual stochastic paths aren’t reversible in a naive sense; the key fact is that **the distributional evolution** *is* reversible in a precise way.

This is formalized by a time-reversal result (attributed here to **Anderson (1982)**): the time-reversed process is again an SDE with a modified drift involving the **score**.

---

## Reverse-time SDE (noise (\to) data)

Let $\bar{\mathbf{x}}(t)$ denote the reverse-time process (the “bar” distinguishes it from forward $\mathbf{x}(t)$). Then:

$$
d\bar{\mathbf{x}}(t)=
\Big[\mathbf{f}(\bar{\mathbf{x}}(t),t)-g^2(t)\nabla_{\mathbf{x}}\log p_t(\bar{\mathbf{x}}(t))\Big],dt

+ g(t),d\bar{\mathbf{w}}(t),
  \qquad
  \bar{\mathbf{x}}(T)\sim p_{\text{prior}}\approx p_T.
  \tag{4.1.6}
$$


### Reverse-time Brownian motion

$$\bar{\mathbf{w}}(t) := \mathbf{w}(T-t)-\mathbf{w}(T)$$

is a Wiener process when viewed in reverse time.

### Key new ingredient: the score term

$$\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$

is the **score** of the marginal at time $t$. The extra drift correction

$$-g^2(t)\nabla_{\mathbf{x}}\log p_t(\cdot)$$

is what makes the reverse dynamics reproduce the correct marginals.

**Important:** the reverse process does **not** inject arbitrary randomness: the diffusion term $g(t),d\bar{\mathbf{w}}(t)$ is *paired* with the score-driven drift so that the distribution flows correctly back to data.

---

## Conceptual intuition: why does the reverse process work?

At first it seems paradoxical: you add noise in reverse time too, so why don’t you just get “more random”?

The intuition is:
* The **score drift** points toward **higher-density regions** of $p_t$, pulling samples toward structured regions (toward the “data manifold” at small $t$).
* The Brownian term provides **controlled exploration**, but its effect is balanced by the score correction.
* Together they produce a process whose marginals match the reversed marginals of the forward SDE.

---

## Connection to Langevin dynamics (special case $f(t)=0$)

If $\mathbf{f}(t)=0$, (4.1.6) becomes

$$d\bar{\mathbf{x}}(t)= -g^2(t)\nabla_{\mathbf{x}}\log p_t(\bar{\mathbf{x}}(t))dt + g(t),d\bar{\mathbf{w}}(t)$$

### Reparameterize time forward

Let $s=T-t$ (so $dt=-ds$) and rename Brownian motion so that $d\bar{\mathbf{w}}(t)=-d\mathbf{w}_s$.
Define $\bar{\mathbf{x}}_s := \bar{\mathbf{x}}(T-s)$ and $\pi_s := p_{T-s}$. Then:

$$d\bar{\mathbf{x}}_s = g^2(T-s)\nabla_{\mathbf{x}}\log \pi_s(\bar{\mathbf{x}}_s)ds + g(T-s),d\mathbf{w}_s$$

Now define a “temperature” schedule

$$\tau(s) := \tfrac12 g^2(T-s)$$

Then

$$
d\bar{\mathbf{x}}*s
= 2\tau(s)\nabla_{\mathbf{x}}\log \pi_s(\bar{\mathbf{x}}_s)\,ds

+ \sqrt{2\tau(s)},d\mathbf{w}_s,
$$

  which is exactly **Langevin form**, but with **time-varying temperature** $\tau(s)$ and a time-evolving target density $\pi_s$.

### Annealing intuition

* Early in reverse time (near $t\approx T$, i.e. $s\approx 0$): $g(T-s)$ is typically larger → more noise → broad exploration.
* As you approach $t\to 0$ (i.e. $s\to T$): $g(T-s)$ decreases → noise weakens, score term dominates → trajectories concentrate near high-density (data-like) regions.

---

## Reverse-time SDE capabilities and learning

### Central role of the score

Define the score function:

$$\mathbf{s}(\mathbf{x},t) := \nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$

Once forward coefficients $\mathbf{f}$ and $g$ are fixed, **the score is the only unknown** needed to run the reverse SDE.

### Practical approach

The “oracle” score is not available, so we learn a neural net $\mathbf{s}_\phi(\mathbf{x},t)$ via **score matching** (later section referenced as 4.2.1). Plugging it into (4.1.6) yields a fully specified generative dynamics.

### Sampling statement

Generation = solve the reverse-time SDE from $t=T$ to $t=0$:

* initialize $\mathbf{x}_T\sim p_{\text{prior}}$,
* integrate reverse dynamics using learned score,
* output $\mathbf{x}_0$ which should follow $p_{\text{data}}$ approximately, assuming $p_{\text{prior}}\approx p_T$.

---

## Minimal “memory hooks” (quick recall)

* **Forward:** $d\mathbf{x}=\mathbf{f}(\mathbf{x},t)dt+g(t)d\mathbf{w}$, $\mathbf{x}(0)\sim p_{\text{data}}$.
* **Kernel:** $p_t(\mathbf{x}_t\mid \mathbf{x}_0)$ (often Gaussian if $\mathbf{f}(\mathbf{x},t)=f(t)\mathbf{x}$).
* **Marginal:** $p_t(\mathbf{x})=\int p_t(\mathbf{x}\mid\mathbf{x}_0)p_{\text{data}}(\mathbf{x}_0)d\mathbf{x}_0$.
* **Reverse:** $d\bar{\mathbf{x}}=[\mathbf{f}-g^2\nabla\log p_t]dt+g,d\bar{\mathbf{w}}$.
* **Key unknown:** the **score** $\nabla\log p_t$.
* **Langevin view:** reverse SDE looks like annealed Langevin with $\tau(s)=\tfrac12 g^2(T-s)$ when $f=0$.


## 4.1.4 Deterministic Process for Generation: Probability Flow ODE (PF-ODE)

### Motivation (Question 4.1.1)

Forward diffusion is usually defined as an SDE that adds noise:

* It is natural to ask: **do we *have* to sample (generate) with the reverse-time SDE**, or can we generate deterministically?

Key idea: **No, SDE sampling is not necessary.** There exists a **deterministic ODE** whose solutions have the **same marginal distributions** as the forward SDE at every time $t$.

---

### Probability Flow ODE (PF-ODE)

Given the forward SDE (from earlier in the text) of the form

$$d\mathbf{x}(t)=\mathbf{f}(\mathbf{x}(t),t),dt + g(t),d\mathbf{w}(t)$$

Song et al. introduce the **Probability Flow ODE**:

$$
\frac{d\tilde{\mathbf{x}}(t)}{dt}
=

\mathbf{f}(\tilde{\mathbf{x}}(t),t)
-\frac{1}{2}g(t)^2 \nabla_{\mathbf{x}}\log p_t(\tilde{\mathbf{x}}(t)).
\tag{PF-ODE}
$$


**Important:** the PF-ODE drift is **not** obtained by “just removing noise.”
The **(\tfrac12)** factor is essential and comes from the **Fokker–Planck** matching principle (next section).

---

### Sampling / generation with PF-ODE

To generate data:

1. Sample an initial point from the terminal distribution (the “prior”):
   
   $$\tilde{\mathbf{x}}(T) \sim p_{\text{prior}} \approx p_T$$
   
2. Integrate the ODE **backwards in time** from $t=T$ down to $t=0$:

$$
\tilde{\mathbf{x}}(0)
=

   \tilde{\mathbf{x}}(T)
   +\int_T^0
   \Big[
   \mathbf{f}(\tilde{\mathbf{x}}(\tau),\tau)
   -\tfrac12 g(\tau)^2 \nabla_{\mathbf{x}}\log p_\tau(\tilde{\mathbf{x}}(\tau))
   \Big],d\tau.
$$

3. In practice the integral is not closed-form ⇒ use **numerical ODE solvers** (Euler, RK methods, adaptive solvers, etc.).
4. As usual in diffusion models, replace the true score $\nabla \log p_t$ with a learned approximation.

---

### Advantages vs reverse-time SDE sampling

* **Bidirectional integration:** you can run the same ODE forward $0\to T$ or backward $T\to 0$, just changing the endpoint initial condition.
* **ODE solver ecosystem:** many mature, accurate, off-the-shelf numerical solvers exist for ODEs.

---

## 4.1.5 Matching Marginal Distributions: Forward/Reverse SDEs and PF-ODE

### High-level goal (Question 4.1.2)

Different stochastic/deterministic processes can yield the **same time-indexed marginals** $\lbrace p_t\rbrace_{t\in[0,T]}$.
What matters is constructing a process whose marginals match the target evolution—especially so that at $t=0$ we recover $p_{\text{data}}$.

---

### Figure intuition (Fig. 4.4)

The forward process gradually transforms an initial complicated distribution $p_0=p_{\text{data}}$ (e.g., a multi-modal mixture) into a simple terminal distribution $p_T \approx p_{\text{prior}}$ (often Gaussian-like).
This evolution of the marginal density $p_t$ is governed by the **Fokker–Planck equation**.

---

## Theorem 4.1.1: Fokker–Planck ensures marginals align

### Forward SDE and its Fokker–Planck PDE

If $\lbrace\mathbf{x}(t)\rbrace_{t\in[0,T]}$ follows the forward SDE

$$
d\mathbf{x}(t)=\mathbf{f}(\mathbf{x}(t),t),dt + g(t),d\mathbf{w}(t),
\qquad \mathbf{x}(0)\sim p_0=p_{\text{data}},
$$

then its marginals $p_t(\mathbf{x})$ satisfy:

$$
\partial_t p_t(\mathbf{x})
=

-\nabla_{\mathbf{x}}\cdot\big(\mathbf{f}(\mathbf{x},t),p_t(\mathbf{x})\big)
+\frac12 g(t)^2,\Delta_{\mathbf{x}}p_t(\mathbf{x}).
\tag{Fokker–Planck}
$$


This can be rewritten as a **continuity equation**

$$
\partial_t p_t(\mathbf{x})
=

-\nabla_{\mathbf{x}}\cdot\big(\mathbf{v}(\mathbf{x},t),p_t(\mathbf{x})\big),
$$

where the **velocity field** is

$$
\mathbf{v}(\mathbf{x},t)
=

\mathbf{f}(\mathbf{x},t)
-\frac12 g(t)^2 \nabla_{\mathbf{x}}\log p_t(\mathbf{x}).
$$


**This $\mathbf{v}$ is exactly the PF-ODE drift**, explaining the $\tfrac12$ factor.

---

### Consequence: PF-ODE and reverse-time SDE share the same marginals

#### (i) PF-ODE

$$\frac{d\tilde{\mathbf{x}}(t)}{dt}=\mathbf{v}(\tilde{\mathbf{x}}(t),t)$$

* If started from $\tilde{\mathbf{x}}(0)\sim p_0$ and run forward, then $\tilde{\mathbf{x}}(t)\sim p_t$.
* Equivalently, if started from $\tilde{\mathbf{x}}(T)\sim p_T$ and run backward, it also matches the same marginals.

#### (ii) Reverse-time SDE (stochastic sampler)


$$
d\bar{\mathbf{x}}(t)
=

\Big[\mathbf{f}(\bar{\mathbf{x}}(t),t)-g(t)^2\nabla_{\mathbf{x}}\log p_t(\bar{\mathbf{x}}(t))\Big],dt
+g(t),d\bar{\mathbf{w}}(t),
$$

initialized at $\bar{\mathbf{x}}(0)\sim p_T$, where $\bar{\mathbf{w}}(t)$ is a Wiener process in reverse time.

**Key point:** PF-ODE and reverse-time SDE differ at the *trajectory level* (deterministic vs stochastic), but are designed to be consistent with the **same family of marginals** governed by Fokker–Planck.

---

## Flow map view and “many conditionals, one marginal”

### PF-ODE flow map

Define the flow map $\Psi_{s\to t}:\mathbb{R}^D\to\mathbb{R}^D$ by “evolving the ODE from time $s$ to $t$”:

$$
\Psi_{s\to t}(\mathbf{x}_s)
=

\mathbf{x}_s + \int_s^t \mathbf{v}(\mathbf{x}_\tau,\tau),d\tau.
\qquad\text{(4.1.9)}
$$

Under mild smoothness assumptions, $\Psi_{s\to t}$ is a **smooth bijection**.

---

### Pushforward density under the ODE

If $\mathbf{x}_0\sim p_{\text{data}}$ and $\mathbf{x}_t=\Psi_{0\to t}(\mathbf{x}_0)$, then the induced density at time $t$ is the pushforward:

$$
p_t^{\text{fwd}}(\mathbf{x}_t)
:=
\int \delta\left(\mathbf{x}_t-\Psi_{0\to t}(\mathbf{x}_0)\right),p_{\text{data}}(\mathbf{x}_0),d\mathbf{x}_0.
$$

The theorem ensures $p_t^{\text{fwd}}=p_t$, matching the forward SDE marginals.

---

### Non-uniqueness of conditionals $Q_t(\mathbf{x}_t\mid \mathbf{x}_0$)

A marginal constraint

$$p_t(\mathbf{x}_t)=\int Q_t(\mathbf{x}_t\mid \mathbf{x}_0),p_{\text{data}}(\mathbf{x}_0),d\mathbf{x}_0$$

does **not** uniquely determine the conditional kernel $Q_t$. Examples that all yield the same $p_t$:

* **Stochastic (simulation-free):**
  
  $$Q_t(\mathbf{x}_t\mid \mathbf{x}_0)=p_t(\mathbf{x}_t\mid \mathbf{x}_0)$$
  
  (the forward SDE transition kernel)

* **Deterministic (ODE-based):**
  
  $$Q_t(\mathbf{x}_t\mid \mathbf{x}_0)=\delta(\mathbf{x}_t-\Psi_{0\to t}(\mathbf{x}_0))$$

* **Mixture family:**

$$
Q_t(\mathbf{x}_t\mid \mathbf{x}_0)
=

  \lambda,p_t(\mathbf{x}_t\mid \mathbf{x}_0)

  + (1-\lambda),\delta(\mathbf{x}_t-\Psi_{0\to t}(\mathbf{x}_0)),
    \quad \lambda\in[0,1].
$$


**Interpretation:** many different dynamics (stochastic/deterministic/hybrid) can satisfy the same marginal evolution—what “selects” the right marginals is the **Fokker–Planck equation**.

---

## Observation 4.1.1: What really matters

* Multiple processes can produce the **same sequence of marginals** $\lbrace p_t\rbrace$.
* The crucial requirement is: **the process must satisfy the Fokker–Planck evolution** for the prescribed $p_t$.
* This gives significant flexibility in designing generative processes from $p_{\text{prior}}\to p_{\text{data}}$ (or the reverse).

---

## Compact cheat sheet (core equations to memorize)

Forward SDE:

$$d\mathbf{x}=\mathbf{f}(\mathbf{x},t),dt + g(t),d\mathbf{w}$$

Fokker–Planck:

$$\partial_t p = -\nabla\cdot(\mathbf{f}p)+\tfrac12 g^2 \Delta p$$

Probability flow velocity:

$$\mathbf{v}=\mathbf{f}-\tfrac12 g^2 \nabla \log p$$

PF-ODE:

$$\dot{\tilde{\mathbf{x}}}=\mathbf{v}(\tilde{\mathbf{x}},t)$$

Reverse-time SDE:

$$d\bar{\mathbf{x}}=\big[\mathbf{f}-g^2\nabla\log p\big]dt + g,d\bar{\mathbf{w}}$$


## Study notes — Score SDE: training, sampling, inversion, likelihood (Sec. 4.2)

### 0) Setup and notation (what objects appear in these pages)

* We have a **forward (noising) diffusion** over continuous time $t\in[0,T]$ that induces a family of marginal densities $\lbrace p_t(\mathbf{x})}_{t\in[0,T]\rbrace$.
* The central quantity is the **score**
  
  $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$
  
  which is generally **intractable**.
* We approximate it with a **time-conditional neural network**
  
  $$\mathbf{s}_\phi(\mathbf{x},t)\approx \nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$
  
* The forward SDE is typically written with:

  * drift $\mathbf{f}(\mathbf{x},t)$
  * diffusion scale $g(t)$
    so the reverse-time and PF-ODE formulas later use these same $f,g$.

---

## 1) Training the score model

### 1.1 “Oracle” score matching objective (intractable target)

The conceptual objective is: fit $\mathbf{s}_\phi$ to the true score at every time:

$$
\mathcal{L}_{\text{SM}}(\phi;\omega(\cdot))
:= \frac{1}{2},\mathbb{E}_{t\sim p_{\text{time}}},
\mathbb{E}_{\mathbf{x}_t\sim p_t}\Big[
\omega(t),\big\|\mathbf{s}_\phi(\mathbf{x}_t,t)-\nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t)\big\|_2^2
\Big].
$$

**Pieces:**

* $p_{\text{time}}$: distribution over $t$ (often uniform on $[0,T]$).
* $\omega(t)$: a **time weighting** (used to emphasize or de-emphasize certain noise levels).

**Problem:** $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ is an **oracle** (unknown).

---

### 1.2 Denoising Score Matching (DSM) objective (tractable target)

To avoid the oracle score, use the **conditional** distribution of the forward process:

* sample a clean data point $\mathbf{x}_0\sim p_{\text{data}}$
* sample $\mathbf{x}_t\sim p_t(\mathbf{x}_t\mid \mathbf{x}_0)$

Then optimize:

$$
\mathcal{L}_{\text{DSM}}(\phi;\omega(\cdot))
:= \frac{1}{2},\mathbb{E}_{t},\mathbb{E}_{\mathbf{x}_0},
\mathbb{E}_{\mathbf{x}_t\sim p_t(\mathbf{x}_t\mid \mathbf{x}_0)}
\Big[
\omega(t),\big\|\mathbf{s}_\phi(\mathbf{x}_t,t)-\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)\big\|_2^2
\Big].
$$

Key point: for many SDE choices (and especially the diffusion-model cases),
$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)$ is **analytically available**.

**Interpretation:** DSM is “regress the network output onto a known conditional score target.”

---

### 1.3 What does DSM learn? (Proposition 4.2.1)

**Proposition (Minimizer of DSM).** The optimal function $\mathbf{s}^*$ satisfies

$$
\mathbf{s}^*(\mathbf{x}_t,t)
= \mathbb{E}_{\mathbf{x}_0\sim p(\mathbf{x}_0\mid \mathbf{x}_t)}
\big[\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)\big]
= \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t),
$$

for (almost) every $\mathbf{x}_t\sim p_t$ and $t\in[0,T]$.

**Why this is true (high-level):**

* For fixed $t$, DSM is a **least-squares regression problem** in the random variable $\mathbf{x}_t$.
* The minimizer of $\mathbb{E}\lvert h(\mathbf{x}_t)-Y\lrvert^2$ is $h^*(\mathbf{x}_t)=\mathbb{E}[Y\mid \mathbf{x}_t]$.
* Here $Y=\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)$, so
  
  $$\mathbf{s}^*(\mathbf{x}_t,t)=\mathbb{E}[Y\mid \mathbf{x}_t]$$

* Then, using Bayes’ rule, that conditional expectation equals the **marginal** score $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)$.

**Takeaway:** DSM lets you train $\mathbf{s}_\phi$ using a tractable conditional target, yet the optimum corresponds to the true marginal score.

---

### 1.4 Practical training recipe (what you do in code)

For each SGD step:

1. Sample $t\sim p_{\text{time}}$.
2. Sample $\mathbf{x}_0\sim p_{\text{data}}$.
3. Sample $\mathbf{x}_t\sim p_t(\mathbf{x}_t\mid \mathbf{x}_0)$ using the forward noising rule.
4. Compute the analytic target $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)$.
5. Minimize the weighted squared error with $\omega(t)$.

---

## 2) Sampling and inference after training (Sec. 4.2.2)

Once trained, denote the learned score as

$$\mathbf{s}_{\phi^\star}(\mathbf{x},t)\approx \nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$

Now replace the oracle score in the **reverse-time SDE** and in the **probability flow ODE (PF-ODE)**.

A helpful visual intuition (Fig. 4.5): starting from $\mathbf{x}_T\sim p_{\text{prior}}$, both:

* solving the reverse-time SDE (stochastic path),
* solving the PF-ODE (deterministic path),
  end near the data manifold at $t=0$ (if the score is accurate).

---

## 3) Generation via the empirical reverse-time SDE

### 3.1 Empirical reverse-time SDE (Eq. 4.2.3)


$$
d\mathbf{x}^{\text{SDE}}_{\phi^\star}(t)
=

\Big[
\mathbf{f}\big(\mathbf{x}^{\text{SDE}}_{\phi^\star}(t),t\big)
-

g^2(t),\mathbf{s}_{\phi^\star}\big(\mathbf{x}^{\text{SDE}}_{\phi^\star}(t),t\big)
\Big]dt
+
g(t),d\bar{\mathbf{w}}(t).
$$


* $\bar{\mathbf{w}}(t)$ is the Brownian motion in reverse-time formulation.
* The learned score modifies the drift by $-g^2(t),s_{\phi^\star}$.

### 3.2 Euler–Maruyama discretization (Eq. 4.2.4)

To sample:

1. Draw $\mathbf{x}_T\sim p_{\text{prior}}$.
2. Step backward from $t=T$ to $t=0$:

$$
\mathbf{x}_{t-\Delta t}
   \leftarrow
   \mathbf{x}_{t}

*

\Big[
\mathbf{f}(\mathbf{x}_t,t)-g^2(t)\mathbf{s}_{\phi^\star}(\mathbf{x}_t,t)
\Big]\Delta t
+
g(t)\sqrt{\Delta t},\boldsymbol{\epsilon},
\quad
\boldsymbol{\epsilon}\sim \mathcal{N}(0,\mathbf{I}).
$$

The final $\mathbf{x}^{\text{SDE}}_{\phi^\star}(0)$ is your generated sample, and ideally

$$p^{\text{SDE}}_{\phi^\star}(\cdot;0)\approx p_{\text{data}}(\cdot)$$

**Connection noted:** DDPM sampling is a **special case** of this Euler–Maruyama discretization for specific choices of $\mathbf{f}$ and $g$.

---

## 4) Generation via the empirical PF-ODE (probability flow ODE)

### 4.1 Empirical PF-ODE (Eq. 4.2.5)


$$
\frac{d}{dt}\mathbf{x}^{\text{ODE}}_{\phi^\star}(t)
=

\mathbf{f}\big(\mathbf{x}^{\text{ODE}}_{\phi^\star}(t),t\big)
-\frac{1}{2}g^2(t),\mathbf{s}_{\phi^\star}\big(\mathbf{x}^{\text{ODE}}_{\phi^\star}(t),t\big).
$$


* Deterministic dynamics (no stochastic term).
* Defines a **continuous flow** that connects $p_{\text{prior}}$ and $p_{\text{data}}$.

Sampling procedure:

1. Draw $\mathbf{x}_T\sim p_{\text{prior}}$.
2. Numerically solve the ODE backward from $T\to 0$.

Equivalent integral form shown:

$$
\mathbf{x}^{\text{ODE}}_{\phi^\star}(0)
=

\mathbf{x}_T
+
\int_T^0
\Big[
\mathbf{f}(\mathbf{x}^{\text{ODE}}_{\phi^\star}(\tau),\tau)
-\frac{1}{2}g^2(\tau)\mathbf{s}_{\phi^\star}(\mathbf{x}^{\text{ODE}}_{\phi^\star}(\tau),\tau)
\Big]d\tau.
$$


### 4.2 Euler method update (Eq. 4.2.6)

With step size $\Delta t>0$:

$$
\mathbf{x}_{t-\Delta t}
\leftarrow
\mathbf{x}_{t}
-

\Big[
\mathbf{f}(\mathbf{x}_t,t)
-\frac{1}{2}g^2(t)\mathbf{s}_{\phi^\star}(\mathbf{x}_t,t)
\Big]\Delta t.
$$

The resulting distribution $p^{\text{ODE}}_{\phi^\star}(\cdot;0)$ should approximate $p_{\text{data}}$.

---

## 5) Core insight: generation = solving an ODE/SDE (Insight 4.2.1)

> Sampling from diffusion models is fundamentally equivalent to solving a corresponding **reverse-time SDE** or **probability flow ODE**.

**Implication:** Sampling can be slow because numerical solvers are iterative and may require many function evaluations (note: typical diffusion setups can use $\sim 1000$ evaluations).

---

## 6) Inversion with PF-ODE (encoder viewpoint)

Unlike SDE sampling, the PF-ODE can be solved both:

* **forward**: $0 \to T$,
* **backward**: $T \to 0$,

because it’s a deterministic ODE (under standard well-posedness assumptions).

**Forward solve interpretation:**
Solving PF-ODE forward maps $\mathbf{x}_0$ to a noisy latent $\mathbf{x}(T)$. This acts like an **encoder**, and enables applications like controllable generation / translation / editing.

---

## 7) Exact log-likelihood via PF-ODE (continuous normalizing flow view)

### 7.1 Define the velocity field

Treat the PF-ODE dynamics as a (Neural ODE–style) flow with velocity

$$
\mathbf{v}_{\phi^\star}(\mathbf{x},t)
:=
\mathbf{f}(\mathbf{x},t)-\frac{1}{2}g^2(t)\mathbf{s}_{\phi^\star}(\mathbf{x},t)
$$

### 7.2 Log-density evolution along the flow

Along the PF-ODE trajectory $\lbrace \mathbf{x}^{\text{ODE}}_{\phi^\star}(t)\rbrace$,

$$
\frac{d}{dt}\log p^{\text{ODE}}_{\phi^\star} \Big(\mathbf{x}^{\text{ODE}}_{\phi^\star}(t),t\Big)
=

-\nabla\cdot \mathbf{v}_{\phi^\star} \Big(\mathbf{x}^{\text{ODE}}_{\phi^\star}(t),t\Big),
$$

where $\nabla\cdot \mathbf{v}$ is the divergence w.r.t. $\mathbf{x}$.

### 7.3 Augmented ODE to compute likelihood (Eq. 4.2.7)

To compute likelihood for $\mathbf{x}_0\sim p_{\text{data}}$, integrate forward from $t=0$ to $t=T$:

$$
\frac{d}{dt}
\begin{bmatrix}
\mathbf{x}(t)\
\delta(t)
\end{bmatrix}
=

\begin{bmatrix}
\mathbf{v}_{\phi^\star}(\mathbf{x}(t),t)\
\nabla\cdot \mathbf{v}_{\phi^\star}(\mathbf{x}(t),t)
\end{bmatrix},
\qquad
\begin{bmatrix}
\mathbf{x}(0)\
\delta(0)
\end{bmatrix}
=

\begin{bmatrix}
\mathbf{x}_0\
0
\end{bmatrix}.
$$

Here $\delta(t)$ **accumulates the log-density change** in the direction needed to recover $\log p(\mathbf{x}_0)$ from the terminal density.

After solving to $T$, you have $\mathbf{x}(T)$ and $\delta(T)$, and:

$$
\log p^{\text{ODE}}_{\phi^\star}(\mathbf{x}_0;0)
=

\log p_{\text{prior}}(\mathbf{x}(T))+\delta(T),
$$

where $p_{\text{prior}}(\mathbf{x}(T))$ is available in closed form (e.g. standard Gaussian).

**Mental model:** PF-ODE gives a reversible flow + change-of-variables, so diffusion models can support **exact likelihood evaluation** (under the ODE formulation).

---


## 4.3 Instantiations of SDEs (Score-SDE framework)

We consider the **forward SDE** (diffusion / noising process)

$$\mathrm{d}\mathbf{x}(t)= f(\mathbf{x},t),\mathrm{d}t + g(t),\mathrm{d}\mathbf{w}(t)$$

where $\mathbf{w}(t)$ is a $D$-dimensional Wiener process (independent coordinates). Song et al. categorize forward SDEs by how the **variance evolves** over time. Here we focus on two widely used cases:

* **VE SDE** = *Variance Exploding*
* **VP SDE** = *Variance Preserving*

A key object is the **perturbation kernel** (transition density)

$$p_t(\mathbf{x}_t\mid \mathbf{x}_0)$$

which tells you what distribution you get after noising clean data $\mathbf{x}_0$ up to time $t$. This kernel is what you sample from during training (e.g., for denoising/score matching), and it also determines a natural **prior** $p_{\text{prior}} = p_T(\mathbf{x}_T)$ used for generation.

---

## Table 4.1 — Summary (VE vs VP)

### VE SDE

* **Drift:** $f(\mathbf{x},t)=0$
* **Diffusion:** $g(t)=\sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}}$
* **SDE:**
  
  $$\mathrm{d}\mathbf{x}(t)= g(t),\mathrm{d}\mathbf{w}(t)$$
  
* **Perturbation kernel:**
  
  $$p_t(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N} \Big(\mathbf{x}_t;\mathbf{x}_0,;(\sigma^2(t)-\sigma^2(0))\mathbf{I}\Big)$$
  
* **Prior (typical):**
  
  $$p_{\text{prior}}=\mathcal{N}(\mathbf{0},\sigma^2(T)\mathbf{I})$$

### VP SDE

* **Drift:** $f(\mathbf{x},t)= -\tfrac12 \beta(t)\mathbf{x}$
* **Diffusion:** $g(t)=\sqrt{\beta(t)}$
* **SDE:**
  
  $$\mathrm{d}\mathbf{x}(t)= -\tfrac12 \beta(t)\mathbf{x}(t),\mathrm{d}t + \sqrt{\beta(t)},\mathrm{d}\mathbf{w}(t)$$
  
* **Perturbation kernel:**
  
  $$
  p_t(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N} \Big(\mathbf{x}_t;;\mathbf{x}_0 e^{-\frac12\int_0^t \beta(\tau)\mathrm{d}\tau},;\mathbf{I}-\mathbf{I}e^{-\int_0^t \beta(\tau)\mathrm{d}\tau}\Big)
  $$

  (equivalently covariance $=(1-e^{-B(t)})\mathbf{I}$ with $B(t)=\int_0^t\beta(s),ds$)
* **Prior:**
  
  $$p_{\text{prior}}=\mathcal{N}(\mathbf{0},\mathbf{I})$$

---

## 4.3.1 VE SDE (Variance Exploding)

### Definition

* Drift term is **zero**:
  
  $$f(\mathbf{x},t)=0$$
  
* Diffusion is controlled by a variance schedule $\sigma(t)$:
  
  $$g(t)=\sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}}$$
  
  So the forward SDE is
  
  $$\mathrm{d}\mathbf{x}(t)=\sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}},\mathrm{d}\mathbf{w}(t)$$

### Perturbation kernel (what noising does)

Because there is no drift, the process does not “shrink” $\mathbf{x}$; it only adds Gaussian noise:

$$
p_t(\mathbf{x}_t\mid \mathbf{x}_0)=
\mathcal{N} \Big(\mathbf{x}_t;\mathbf{x}_0,;(\sigma^2(t)-\sigma^2(0))\mathbf{I}\Big).
$$

### Prior choice

Assume $\sigma(t)$ is increasing on $[0,T]$ and $\sigma^2(T)\gg\sigma^2(0)$. Then a natural prior is:

$$p_{\text{prior}}:=\mathcal{N}(\mathbf{0},\sigma^2(T)\mathbf{I})$$

### Typical instance: NCSN (discretized VE)

A standard VE design uses a **geometric** schedule (for $t\in(0,1]$):

$$\sigma(t):=\sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{t}$$

so the variance levels form a geometric sequence. NCSN can be viewed as a discretization of this VE SDE.

**Intuition:** VE keeps the mean at $\mathbf{x}_0$ but steadily increases the noise scale—eventually the signal is drowned by large variance.

---

## 4.3.2 VP SDE (Variance Preserving)

### Definition

Let $\beta:[0,T]\to\mathbb{R}_{\ge 0}$ be a nonnegative “noise rate” schedule.

* Drift pulls $\mathbf{x}(t)$ toward zero:
  
  $$f(\mathbf{x},t)= -\tfrac12\beta(t)\mathbf{x}$$
  
* Diffusion injects noise:
  
  $$g(t)=\sqrt{\beta(t)}$$

Forward SDE:

$$\mathrm{d}\mathbf{x}(t)= -\tfrac12 \beta(t)\mathbf{x}(t),\mathrm{d}t + \sqrt{\beta(t)},\mathrm{d}\mathbf{w}(t)$$

### Perturbation kernel

Define

$$B(t):=\int_0^t\beta(s),\mathrm{d}s$$

Then

* mean:
  
  $$\mathbb{E}[\mathbf{x}_t\mid \mathbf{x}_0]=e^{-\frac12B(t)}\mathbf{x}_0$$
  
* covariance (isotropic):
  
  $$\mathrm{Cov}[\mathbf{x}_t\mid \mathbf{x}_0]=(1-e^{-B(t)})\mathbf{I}$$
  
  So
  
  $$p_t(\mathbf{x}_t\mid \mathbf{x}_0)= \mathcal{N}\Big(\mathbf{x}_t;;e^{-\frac12B(t)}\mathbf{x}_0,;(1-e^{-B(t)})\mathbf{I}\Big)$$

### Prior choice

At large time (typical design makes $B(T)$ large), the mean vanishes and covariance approaches $\mathbf{I}$, hence:

$$p_{\text{prior}}:=\mathcal{N}(\mathbf{0},\mathbf{I})$$

### Note on computing scores

Since $p_t(\mathbf{x}_t\mid \mathbf{x}_0)$ is Gaussian with known mean/covariance, its **score**

$$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)$$

has a closed form (for isotropic covariance it’s proportional to $-(\mathbf{x}_t-\text{mean})$).

### Typical instance: DDPM (discretized VP)

A classic VP schedule (for $t\in[0,1]$) is **linear**:

$$\beta(t):=\beta_{\min}+t(\beta_{\max}-\beta_{\min})$$

DDPM can be interpreted as a discretization of the VP SDE.

**Intuition:** VP simultaneously (i) shrinks the signal and (ii) adds noise so that the total variance stays controlled and ends near standard normal.

---

## 4.3.3 (Optional) How the perturbation kernel $p_t(\mathbf{x}_t\mid \mathbf{x}_0)$ is derived

### Linear-drift case ⇒ conditional Gaussian

If the drift is **linear in $\mathbf{x}$**:

$$f(\mathbf{x},t)=f(t)\mathbf{x} \quad (f(t)\in\mathbb{R})$$

then the SDE is

$$\mathrm{d}\mathbf{x}(t)= f(t)\mathbf{x}(t),\mathrm{d}t + g(t),\mathrm{d}\mathbf{w}(t)$$

Even if the data distribution $p_{\text{data}}$ is non-Gaussian, the *conditional* transition is Gaussian:

$$p_t(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N}\big(\mathbf{x}_t;\mathbf{m}(t),P(t)\mathbf{I}_D\big)$$

where

$$
\mathbf{m}(t)=\mathbb{E}[\mathbf{x}_t\mid \mathbf{x}_0],\qquad
P(t)\mathbf{I}_D=\mathrm{Cov}[\mathbf{x}_t\mid \mathbf{x}_0].
$$

### Moment ODEs

The mean and (scalar) variance satisfy:

$$
\frac{\mathrm{d}\mathbf{m}(t)}{\mathrm{d}t}=f(t)\mathbf{m}(t),\qquad
\frac{\mathrm{d}P(t)}{\mathrm{d}t}=2f(t)P(t)+g^2(t),
$$

with initial conditions $\mathbf{m}(0)=\mathbf{x}_0$, $P(0)=0$.

### Closed-form solution via integrating factor

Define the exponential integrating factor

$$\mathcal{E}(s\to t):=\exp \left(\int_s^t f(u),\mathrm{d}u\right)$$

Then:

$$
\mathbf{m}(t)=\mathcal{E}(0\to t)\mathbf{x}_0,\qquad
P(t)=\int_0^t \mathcal{E}^2(s\to t),g^2(s),\mathrm{d}s.
$$

---

## Worked examples (transition kernels)

### VE SDE

Here $f=0$, $g(t)=\sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}}$.

* Mean ODE: $\frac{\mathrm{d}\mathbf{m}}{\mathrm{d}t}=0\Rightarrow \mathbf{m}(t)=\mathbf{x}_0$
* Variance ODE: $\frac{\mathrm{d}P}{\mathrm{d}t}=\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}\Rightarrow P(t)=\sigma^2(t)-\sigma^2(0)$

So

$$
p_t(\mathbf{x}_t\mid \mathbf{x}_0)=
\mathcal{N} \Big(\mathbf{x}_t;\mathbf{x}_0,;(\sigma^2(t)-\sigma^2(0))\mathbf{I}_D\Big).
$$

### VP SDE

Here $f(t)=-\tfrac12\beta(t)$, $g(t)=\sqrt{\beta(t)}$, and $B(t)=\int_0^t\beta(s)ds$.

* Mean:
  
  $$\frac{\mathrm{d}\mathbf{m}}{\mathrm{d}t}=-\tfrac12\beta(t)\mathbf{m}(t) \Rightarrow  \mathbf{m}(t)=e^{-\frac12B(t)}\mathbf{x}_0$$
  
* Variance:
  
  $$\frac{\mathrm{d}P}{\mathrm{d}t}=2f(t)P+g^2(t)= -\beta(t)P(t)+\beta(t)$$
  
  Multiply by $e^{B(t)}$ (integrating factor):
  
  $$\frac{\mathrm{d}}{\mathrm{d}t}\big(P(t)e^{B(t)}\big)=\beta(t)e^{B(t)} \Rightarrow P(t)=1-e^{-B(t)}$$

Final:

$$p_t(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N} \Big(\mathbf{x}_t;;e^{-\frac12B(t)}\mathbf{x}_0,;(1-e^{-B(t)})\mathbf{I}_D\Big)$$

---

## Mental model: VE vs VP (what to remember)

* **VE:** mean stays $\mathbf{x}_0$; variance grows like $\sigma^2(t)$ (eventually huge).
* **VP:** mean decays to $0$; variance rises but is capped to approach $1$ (standard normal), giving a clean $\mathcal{N}(0,I)$ prior.


# Study notes — Section 4.4: Rethinking forward kernels in score-based and variational diffusion models

## 1) Why “rethink” the forward kernel?

Diffusion/Score-SDE models are often introduced via **incremental** forward transitions:

* **DDPM (discrete):** $p(x_t\mid x_{t-\Delta t})$
* **Score SDE (continuous):** an SDE that implies infinitesimal transitions

But in practice (especially in the common losses), what matters most is the **accumulated / marginal perturbation kernel from data**:

$$p_t(x_t\mid x_0)$$

Both DDPM and Score-SDE ultimately rely on this kernel:

* DDPM: by recursive composition of step kernels
* Score-SDE: by solving ODEs (for moments) induced by the SDE

**Key message:** defining $p_t(x_t\mid x_0)$ directly is often **cleaner**, **more interpretable**, and aligns naturally with loss/prior design (e.g., what happens as $t\to T$).

---

## 2) A general affine forward perturbation process $p_t(x_t\mid x_0)$

### Definition (Eq. 4.4.1)

Assume a Gaussian perturbation kernel:

$$p_t(x_t\mid x_0) := \mathcal N \big(x_t;\ \alpha_t x_0,\ \sigma_t^2 I\big)$$

where $x_0\sim p_{\text{data}}$, and $\alpha_t,\sigma_t\ge 0$ for $t\in[0,T]$, typically satisfying:

* $\alpha_t>0$ and $\sigma_t>0$ for $t\in(0,T]$ (allowing $\sigma_0=0$)
* usually $\alpha_0=1,\ \sigma_0=0$

### Sampling form

$$x_t = \alpha_t x_0 + \sigma_t \varepsilon,\qquad \varepsilon\sim \mathcal N(0,I)$$

### This single form subsumes common “forward types”

* **VE (NCSN) kernel:** $\alpha_t\equiv 1,\ \sigma_T\gg 1$
* **VP (DDPM) kernel:** $\alpha_t := \sqrt{1-\sigma_t^2}$ so that $\alpha_t^2+\sigma_t^2=1$
* **FM kernel:** $\alpha_t=1-t,\ \sigma_t=t$ (linear interpolation between $x_0$ and noise)

---

## 3) Connection to Score SDE: marginal kernel $\Longleftrightarrow$ linear SDE

### Score-SDE forward process (linear-in-$x$ form)

If $p_t(x_t\mid x_0)$ has the affine Gaussian form above, it corresponds to an SDE

$$dx(t)= f(t),x(t),dt + g(t),dw(t)$$

where $w(t)$ is Brownian motion (so $dw(t)$ is “Gaussian noise” with variance $\propto dt$).

### Lemma 4.4.1 (Forward perturbation kernel ⇔ linear SDE)

Define

$$\lambda_t := \log\frac{\alpha_t}{\sigma_t}\quad (t\in(0,T])$$

Given $x_t=\alpha_t x_0+\sigma_t\varepsilon$, the corresponding SDE coefficients are:

$$f(t)=\frac{d}{dt}\log \alpha_t$$

$$
g^2(t)=\frac{d}{dt}\sigma_t^2 - 2\frac{d}{dt}\log\alpha_t\ \sigma_t^2
= -2\sigma_t^2\frac{d}{dt}\lambda_t
$$

Conversely, any linear SDE whose conditionals are $\mathcal N(\alpha_t x_0,\sigma_t^2 I)$ must satisfy these relations.

#### Proof idea (what’s happening)

For a linear SDE, the conditional mean $m(t)$ and covariance $P(t)$ satisfy ODEs:

* $m'(t)= f(t),m(t)$
* $P'(t)=2f(t)P(t)+g^2(t)I$

Matching $m(t)=\alpha_t x_0$ and $P(t)=\sigma_t^2 I$ yields the formulas above.

### Observation 4.4.1

> Defining $p_t(x_t\mid x_0)$ is **equivalent** to specifying the linear SDE coefficients $f(t)$ and $g(t)$.

So you can design the forward process **either** by:

* choosing $\alpha_t,\sigma_t$ directly (marginal view), **or**
* choosing $f,g$ (SDE view)

---

## 4) Terminal prior and why “exact Gaussian prior at finite time” can be pathological

To exactly match a Gaussian prior at terminal time $T$, you’d like the process to **forget $x_0$**:

* require $\alpha_T = 0$
* and set $\sigma_T^2$ to the desired prior variance

But in the SDE formulation,

$$\alpha_t=\exp \left(\int_0^t f(u),du\right)$$

To force ($\alpha_T=0$ at finite $T$, you need

$$\int_0^T f(u),du = -\infty$$

meaning the drift $f(t)$ must contract “infinitely fast” near $T$. At the same time, maintaining the prescribed variance forces the diffusion to blow up; the text notes this is reflected by

$$g^2(t)=\sigma_t^{2,\prime}-2\frac{\alpha_t'}{\alpha_t}\sigma_t^2 \to \infty\quad \text{as }t\to T$$

**Practical takeaway:** if $f$ and $g$ stay bounded on $[0,T]$, then $\alpha_T>0$ and some dependence on $x_0$ remains; the Gaussian prior is then reached only **asymptotically** (e.g., in the limit $t\to T$ without exact attainment, or on an infinite horizon with reparameterization).

---

## 5) Connection to variational diffusion (DDPM/VDM): Bayes rule and reverse kernels

### Core DDPM identity (Eq. 4.4.3)

A reverse conditional (posterior) can be written using Bayes’ rule:

$$
p(x_{t-\Delta t}\mid x_t, x)
=

p(x_t\mid x_{t-\Delta t})
\cdot
\frac{p_{t-\Delta t}(x_{t-\Delta t}\mid x)}{p_t(x_t\mid x)}.
$$

Typically $x=x_0\sim p_{\text{data}}$. This posterior is central:

* gives tractable training targets (ELBO terms)
* yields efficient sampling updates

The section’s theme: even if DDPM starts from incremental kernels, $p_t(x_t\mid x_0)$ is often the clearer “primary object.”

---

## 6) Closed-form reverse conditional transitions for the general affine kernel

Let $0\le t < s \le T$.

### Useful “between-time” parameters

Define

$$
\alpha_{s\mid t} := \frac{\alpha_s}{\alpha_t},\qquad
\sigma_{s\mid t}^2 := \sigma_s^2 - \alpha_{s\mid t}^2\sigma_t^2.
$$

### Forward transition between noisy times (Eq. 4.4.5)

$$p(x_s\mid x_t)=\mathcal N \big(x_s;\ \alpha_{s\mid t}x_t,\ \sigma_{s\mid t}^2 I\big)$$

### Lemma 4.4.2 (Reverse conditional transition kernels)

The reverse conditional kernel has Gaussian form:

$$p(x_t\mid x_s, x)=\mathcal N \big(x_t;\ \mu(x_s,x;,s,t),\ \sigma^2(s,t)I\big)$$

with

$$
\mu(x_s,x;,s,t)
=

\frac{\alpha_{s\mid t}\sigma_t^2}{\sigma_s^2}x_s
+
\frac{\alpha_t\sigma_{s\mid t}^2}{\sigma_s^2}x,
$$


$$\sigma^2(s,t)=\sigma_{s\mid t}^2\frac{\sigma_t^2}{\sigma_s^2}$$

**Interpretation:** the posterior mean is a **weighted blend** of:

* the later noisy sample $x_s$
* the clean conditioning variable $x$ (usually $x_0$)

Weights depend entirely on the noise schedule $(\alpha,\sigma)$.

---

## 7) Reverse model parameterization (x-prediction and ε-prediction)

In variational diffusion / ELBO training, you model a parametric reverse process:

$$p_\phi(x_t\mid x_s) := \mathcal N\big(x_t;\ \mu_\phi(x_s,s,t),\ \sigma^2(s,t)I\big)$$

and plug in a learned predictor $x_\phi(x_s,s)$ for the clean signal $x$ in the mean:

$$
\mu_\phi(x_s,s,t)
=

\frac{\alpha_{s\mid t}\sigma_t^2}{\sigma_s^2}x_s
+
\frac{\alpha_t\sigma_{s\mid t}^2}{\sigma_s^2}x_\phi(x_s,s).
$$

This is the **x-prediction** parameterization.

There is also an equivalent **ε-prediction** view via:

$$x_s = \alpha_s x_\phi(x_s,s) + \sigma_s \varepsilon_\phi(x_s,s)$$

mirroring the standard DDPM identity relating $x_0$-prediction and noise-prediction.

---

## 8) Diffusion loss becomes a weighted regression loss (Eq. 4.4.7)

For the KL term in the diffusion objective, because both distributions are Gaussian with the same covariance $\sigma^2(s,t)I$, the KL reduces to a squared error between means:

$$
D_{\mathrm{KL}} \big(p(x_t\mid x_s,x_0)\ |\ p_\phi(x_t\mid x_s)\big)
=

\frac{1}{2\sigma^2(s,t)}
\left\|\mu(x_s,x_0;s,t)-\mu_\phi(x_s,s,t)\right\|_2^2.
$$


This simplifies neatly to:

$$= \frac12\big(\mathrm{SNR}(t)-\mathrm{SNR}(s)\big)\ \lvert x_0-x_\phi(x_s,s)\rvert_2^2,$$

where

$$\mathrm{SNR}(u):=\frac{\alpha_u^2}{\sigma_u^2}$$

**Takeaway:** the ELBO training signal is essentially **x0 regression** with a time-dependent weight given by an SNR difference.

---

## 9) Continuous-time limit: VDM objective (Kingma et al., 2021)

Kingma et al. study the limit $t\to s$ of the weighted regression term, yielding:

$$
\mathcal L^{\infty}_{\mathrm{VDM}}(x_0)
=

-\frac12,\mathbb E_{s,\ \varepsilon\sim\mathcal N(0,I)}
\Big[\mathrm{SNR}'(s)\ \lvert x_0-x_\phi(x_s,s)\rvert_2^2\Big].
$$

Typically $\mathrm{SNR}(s)$ decreases with $s$, so $\mathrm{SNR}'(s)<0$, making the overall weight $-\mathrm{SNR}'(s)$ positive.

This perspective also suggests a **learnable noise schedule** via learning $\mathrm{SNR}(s)$ (though extensions are beyond the shown excerpt).

---

## 10) Sampling update (generalized DDPM step) — Eq. 4.4.8

To sample backward from time $s$ to $t$ (with $t<s$), use the parametric reverse kernel:

$$
x_t
=

\frac{\alpha_{s\mid t}\sigma_t^2}{\sigma_s^2}x_s
+
\frac{\alpha_t\sigma_{s\mid t}^2}{\sigma_s^2}x_\phi(x_s,s)
+
\sigma_{s\mid t}\frac{\sigma_t}{\sigma_s},\varepsilon_s,
\qquad \varepsilon_s\sim\mathcal N(0,I).
$$

This is exactly the familiar DDPM-style update, but expressed for the **general** $(\alpha_t,\sigma_t)$ schedule.

---

# “Cheat sheet” summary (what to remember)

### Forward (marginal) design

* Pick $\alpha_t,\sigma_t$ $\implies$ defines $p_t(x_t\mid x_0)=\mathcal N(\alpha_t x_0,\sigma_t^2 I)$
* Sample: $x_t=\alpha_t x_0+\sigma_t\varepsilon$

### Convert to SDE (linear)

* $f(t)=\frac{d}{dt}\log\alpha_t$
* $g^2(t)=\sigma_t^{2,\prime}-2(\log\alpha_t)'\sigma_t^2=-2\sigma_t^2\lambda_t'$, $\lambda_t=\log(\alpha_t/\sigma_t)$

### Between-time forward kernel

* $\alpha_{s\mid t}=\alpha_s/\alpha_t$
* $\sigma_{s\mid t}^2=\sigma_s^2-\alpha_{s\mid t}^2\sigma_t^2$
* $p(x_s\mid x_t)=\mathcal N(\alpha_{s\mid t}x_t,\sigma_{s\mid t}^2I)$

### Reverse posterior + model

* True posterior: $p(x_t\mid x_s,x)$ Gaussian with mean/var in Lemma 4.4.2
* Model replaces $x$ by $x_\phi(x_s,s)$
* KL term $\implies$ weighted regression: $\frac12(\mathrm{SNR}(t)-\mathrm{SNR}(s))\lvert x_0-x_\phi\rvert^2$

---


# Study notes — 4.5 Fokker–Planck equation & reverse-time SDEs (via marginalization + Bayes)

## Setup and notation

* State: $\mathbf{x}_t \in \mathbb{R}^D$

* Marginal density: $p_t(\mathbf{x})$

* Forward (infinitesimal / Euler–Maruyama) transition kernel (Eq. 4.1.2 style):
  
  $$p(\mathbf{x}_{t+\Delta t}\mid \mathbf{x}_t) = \mathcal{N} \Big(\mathbf{x}_{t+\Delta t};\ \mathbf{x}_t + \mathbf{f}(\mathbf{x}_t,t)\Delta t,\ g^2(t)\Delta t,\mathbf{I}\Big)$$

  This corresponds to the forward SDE (informally):
  
  $$d\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t,t),dt + g(t),d\mathbf{w}_t$$
  
  with isotropic diffusion $g(t)\mathbf{I}$.

* Operators:

  * $\nabla_{\mathbf{x}}$: gradient
  * $\nabla_{\mathbf{x}}\cdot$: divergence
  * $\Delta_{\mathbf{x}} = \sum_i \partial_{x_i}^2$: Laplacian

---

## 4.5.1 Fokker–Planck from marginalizing transition kernels

### Step 1: Chapman–Kolmogorov / marginalization

Using the Markov property,

$$
p_{t+\Delta t}(\mathbf{x})
= \int p(\mathbf{x}\mid \mathbf{y}),p_t(\mathbf{y}),d\mathbf{y}
= \int \mathcal{N} \Big(\mathbf{x};\ \mathbf{y}+\mathbf{f}(\mathbf{y},t)\Delta t,\ g^2(t)\Delta t,\mathbf{I}\Big),p_t(\mathbf{y}),d\mathbf{y}
$$

### Step 2: Change of variables to center the Gaussian

Define

$$
\mathbf{u} := \mathbf{y} + \mathbf{f}(\mathbf{y},t)\Delta t
$$

For small $\Delta t$, this map is invertible and admits expansions:

$$
\mathbf{y} = \mathbf{u} - \mathbf{f}(\mathbf{u},t)\Delta t + \mathcal{O}(\Delta t^2),
\qquad
\left\|\det\frac{\partial \mathbf{y}}{\partial \mathbf{u}}\right\|
= 1 - (\nabla_{\mathbf{u}}\cdot \mathbf{f})(\mathbf{u},t)\Delta t + \mathcal{O}(\Delta t^2).
$$

Substituting and expanding $p_t(\mathbf{y})$ around $\mathbf{u}$ gives (to first order):

$$
p_{t+\Delta t}(\mathbf{x})
= \int \mathcal{N} \big(\mathbf{x};\mathbf{u},g^2(t)\Delta t,\mathbf{I}\big)
\Big[
p_t(\mathbf{u})

+ \Delta t,\mathbf{f}(\mathbf{u},t)\cdot\nabla_{\mathbf{u}}p_t(\mathbf{u})
+ \Delta t,(\nabla_{\mathbf{u}}\cdot\mathbf{f})(\mathbf{u},t),p_t(\mathbf{u})
  \Big],d\mathbf{u}

- \mathcal{O}(\Delta t^2).
$$


> The bracketed combination is exactly the “drift acting on density” term:
> 
> $$\mathbf{f}\cdot\nabla p + (\nabla\cdot \mathbf{f})p = \nabla\cdot(\mathbf{f}p)$$
> 

### Step 3: Taylor–Gaussian smoothing formula

For smooth $\phi:\mathbb{R}^D\to\mathbb{R}$ and $\sigma^2>0$, with $\mathbf{z}\sim \mathcal{N}(0,\mathbf{I})$,

$$
\int \mathcal{N}(\mathbf{x};\mathbf{u},\sigma^2\mathbf{I}),\phi(\mathbf{u}),d\mathbf{u}
= \mathbb{E}[\phi(\mathbf{x}+\sigma\mathbf{z})]
= \phi(\mathbf{x}) + \frac{\sigma^2}{2}\Delta_{\mathbf{x}}\phi(\mathbf{x}) + \mathcal{O}(\sigma^4).
$$

This comes from Taylor expanding $\phi(\mathbf{x}+\sigma\mathbf{z})$ and using
$\mathbb{E}[\mathbf{z}]=0$, $\mathbb{E}[\mathbf{z}\mathbf{z}^\top]=\mathbf{I}$.

Here $\sigma^2 = g^2(t)\Delta t$.

### Step 4: Keep terms up to $\mathcal{O}(\Delta t)$

* Convolving $p_t(\cdot)$ produces the Laplacian correction $\frac{g^2(t)\Delta t}{2}\Delta p_t(\mathbf{x})$.
* The other terms already have a prefactor $\Delta t$, so their Gaussian-smoothing corrections would be $\mathcal{O}(\Delta t^2)$ and can be dropped.

Thus:

$$
p_{t+\Delta t}(\mathbf{x}) - p_t(\mathbf{x})
= -\Delta t,\mathbf{f}(\mathbf{x},t)\cdot\nabla_{\mathbf{x}}p_t(\mathbf{x})
-\Delta t,(\nabla_{\mathbf{x}}\cdot\mathbf{f})(\mathbf{x},t),p_t(\mathbf{x})

+ \frac{g^2(t)}{2}\Delta t,\Delta_{\mathbf{x}}p_t(\mathbf{x})
+ \mathcal{O}(\Delta t^2).
$$

  Combine the drift terms:

$$
p_{t+\Delta t} - p_t
  = -\Delta t,\nabla_{\mathbf{x}}\cdot(\mathbf{f}p_t)
+ \frac{g^2(t)}{2}\Delta t,\Delta_{\mathbf{x}}p_t
+ \mathcal{O}(\Delta t^2).
$$


### Step 5: Take $\Delta t\to 0$ ⇒ Fokker–Planck

Divide by $\Delta t$ and let $\Delta t\to 0$:

$$
\boxed{
\partial_t p_t(\mathbf{x})
=

-\nabla_{\mathbf{x}}\cdot\big(\mathbf{f}(\mathbf{x},t),p_t(\mathbf{x})\big)
+\frac{g^2(t)}{2},\Delta_{\mathbf{x}}p_t(\mathbf{x})
}
$$

(For isotropic diffusion $g(t)\mathbf{I}$.)

**Interpretation (useful intuition):** this is a conservation/continuity equation for probability, where drift transports mass and diffusion spreads it.

---

## 4.5.2 Why the reverse-time SDE has a score term (Bayes-rule derivation)

Goal: find the **reverse-time transition** $p(\mathbf{x}_t \mid \mathbf{x}_{t+\Delta t})$ from the forward kernel, then take $\Delta t\to 0$.

### Step 1: Bayes rule for the reverse kernel


$$
p(\mathbf{x}_t\mid \mathbf{x}_{t+\Delta t})
=

p(\mathbf{x}_{t+\Delta t}\mid \mathbf{x}_t),
\frac{p_t(\mathbf{x}_t)}{p_{t+\Delta t}(\mathbf{x}_{t+\Delta t})}
=

p(\mathbf{x}_{t+\Delta t}\mid \mathbf{x}_t),
\exp \Big(\log p_t(\mathbf{x}_t)-\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t})\Big).
$$


### Step 2: First-order Taylor expansion of the log-density term

Expand $\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t})$ around $(\mathbf{x}_t,t)$:

$$
\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t})
=

\log p_t(\mathbf{x}_t)
+
\nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t)\cdot(\mathbf{x}_{t+\Delta t}-\mathbf{x}_t)
+
\partial_t\log p_t(\mathbf{x}_t),\Delta t
+
\mathcal{O}(\lvert\mathbf{h}\rvert^2),
$$

with $\mathbf{h}:=(\mathbf{x}_{t+\Delta t}-\mathbf{x}_t,\Delta t)$.

So:

$$
\log p_t(\mathbf{x}_t)-\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t})
=

-\nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t)\cdot(\mathbf{x}_{t+\Delta t}-\mathbf{x}_t)
-\partial_t\log p_t(\mathbf{x}_t)\Delta t
+\mathcal{O}(\lvert \mathbf{h}\rvert^2).
$$


A key scaling fact for diffusions: $\mathbb{E}\lvert\mathbf{x}_{t+\Delta t}-\mathbf{x}_t\rvert_2^2=\mathcal{O}(\Delta t)$, so the remainder is $\mathcal{O}(\Delta t^2)$ in expectation (hence negligible at first order).

### Step 3: Substitute and complete the square

Use the forward Gaussian

$$
p(\mathbf{x}_{t+\Delta t}\mid \mathbf{x}_t)\propto
\exp \left(-\frac{\lvert\mathbf{x}_{t+\Delta t}-\mathbf{x}_t-\mathbf{f}(\mathbf{x}_t,t)\Delta t\rvert_2^2}{2g^2(t)\Delta t}\right).
$$

Let

$$
\boldsymbol{\delta}:=\mathbf{x}_{t+\Delta t}-\mathbf{x}_t,\qquad
\boldsymbol{\mu}:=\mathbf{f}(\mathbf{x}_t,t)\Delta t
$$

The Bayes correction contributes a linear term in $\boldsymbol{\delta}$ involving $\nabla\log p_t(\mathbf{x}_t)$. Completing the square yields (up to $\mathcal{O}(\Delta t)$ multiplicative errors):

$$
p(\mathbf{x}_t\mid \mathbf{x}_{t+\Delta t})
\approx
\mathcal{N}\Big(
\mathbf{x}_t;\
\mathbf{x}_{t+\Delta t}-[\mathbf{f}(\mathbf{x}_t,t)-g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t)]\Delta t,\
g^2(t)\Delta t,\mathbf{I}
\Big),(1+\mathcal{O}(\Delta t))
$$

* The extra $\lvert g^2(t)\Delta t,\nabla\log p_t\rvert^2$ term produced by completing the square is $\mathcal{O}(\Delta t^2)$ → absorbed into the error.
* The $\partial_t\log p_t,\Delta t$ factor affects normalization at $\mathcal{O}(\Delta t)$, not the leading-order mean/covariance structure.

### Step 4: Replace $(\mathbf{x}_t,t)$ by $(\mathbf{x}_{t+\Delta t},t+\Delta t)$ (smoothness)

Under smoothness,

$$
\mathbf{f}(\mathbf{x}_t,t)\approx \mathbf{f}(\mathbf{x}_{t+\Delta t},t+\Delta t),\quad
g(t)\approx g(t+\Delta t),\quad
\nabla\log p_t(\mathbf{x}_t)\approx \nabla\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t})
=:\mathbf{s}(\mathbf{x}_{t+\Delta t},t+\Delta t),
$$

where $\mathbf{s}(\mathbf{x},t)$ is the **score**.

So the reverse kernel says:

* **Mean step backward** is “forward drift” minus a **score correction** $g^2,\mathbf{s}$
* **Covariance** is still $g^2\Delta t,\mathbf{I}$

### Step 5: Continuous-time limit ⇒ reverse-time SDE

Heuristically, as $\Delta t\to 0$, the reverse-time process satisfies:

$$
\boxed{
d\mathbf{x}_t
=

\big[\mathbf{f}(\mathbf{x}_t,t)-g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t)\big]dt
+
g(t),d\bar{\mathbf{w}}_t
}
$$

where $\bar{\mathbf{w}}_t$ is a Brownian motion in reverse time (and the process is run with time decreasing from $T$ to $0$).

**Intuition:** the score term points toward higher-density regions of $p_t$, so when you run time backward it acts like a *denoising drift* that counteracts the forward diffusion.

**Practical link (diffusion/score models):** if you learn $\mathbf{s}_\theta(\mathbf{x},t)\approx \nabla_{\mathbf{x}}\log p_t(\mathbf{x})$, you can sample by simulating the reverse-time SDE from noise (large $t$) back to data (small $t$).

---

## 4.6 Closing remarks (big picture takeaways)

* **Unification:** DDPM-style discrete diffusion and NCSN/score-based models can be viewed as *discretizations of SDEs* with different choices of drift/volatility.
* **Reverse-time SDE is the generative engine:** it “reverses” the forward noising process. Crucially, its drift depends on one unknown object:
  
  $$\nabla_{\mathbf{x}}\log p_t(\mathbf{x}) \quad \text{(the score)}$$
  
  This explains why score learning is central.
* **Probability Flow ODE (PF-ODE):** a deterministic counterpart whose trajectories share the same marginals $\lbrace p_t\rbrace$ as the SDE; this equivalence rests on the Fokker–Planck equation.
* **Core implication:** generation $\approx$ solving a differential equation; training $\approx$ learning the vector field (score / velocity); sampling $\approx$ numerical integration.
* This PF-ODE viewpoint bridges toward **flow-based generative modeling** (Normalizing Flows, Neural ODEs) and motivates the transition to **Flow Matching**.

---
