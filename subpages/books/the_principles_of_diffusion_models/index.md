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
1. Sampling from $p_ϕ(x)$
2. Compute the probability (or likelihood) of any given data sample $x'$: $p_ϕ(x')$.

**Training of DGM**
* learn parameters $ϕ$ of a model family $\lbrace pϕ\rbrace$ 
* by minimizing a discrepancy $\mathcal{D}(p_{\text{data}},p_ϕ)$:

$$\phi^*\in\arg\min_\phi \mathcal{D}(p_{\text{data}},p_ϕ)$$

### Divergences

In statistics, divergence is a non-negative measure of the difference, dissimilarity, or distance between two probability distributions (P and Q).

#definition(Divergence(statistics)) {
Given a **differentiable manifold** $M$ of dimension  $n$, a **divergence** on $M$ is a $C^2$ $\mathcal{D}:M\times M\to [0,\infty )$ satisfying:
1. $\mathcal{D}(p,q)\geq 0$ for all $p,q\in M$ (non-negativity),
2. $\mathcal{D}(p,q)=0$ if and only if $p=q$ (positivity),
3. At every point p\in M, $D(p,p+dp)$ is a positive-definite **quadratic form** for infinitesimal displacements $dp$ from $p$.

In applications to statistics, the manifold M is typically the space of parameters of a **parametric family of probability distributions**.
}

#remark(Divergence could be view as a metric for probability measures) {
Informally, people sometimes describe divergences as measuring the "distance" between probability distributions. This risks confusion with formal distance metrics, which must satisfy some extra requirements. In addition to the requirements above, a distance metric must also be symmetric: 𝐷(𝑎,𝑏)=𝐷(𝑏,𝑎). And, it must satisfy the triangle inequality: 𝐷(𝑎,𝑐)≤𝐷(𝑎,𝑏)+𝐷(𝑏,𝑐). As a side note, divergences are defined specifically on probability distributions, whereas distance metrics can be defined on other types of objects too.

All distance metrics between probability distributions are also divergences, but the converse is not true--a divergence may or may not be a distance metric. For example, the KL divergence is a divergence, but not a distance metric because it's not symmetric and doesn't obey the triangle inequality. In contrast, the Hellinger distance is both a divergence and a distance metric. To avoid confusion with formal distance metrics, I prefer to say that divergences measure the dissimilarity between distributions.
}

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

* Choose a discrepancy measure (D(p_{\text{data}}, p_\phi)) between distributions and solve
  
  $$\phi^* \in \arg\min_\phi D(p_{\text{data}}, p_\phi). \qquad\text{(1.1.1)}$$
  
* Since $p_{\text{data}}$ is not directly accessible, $D$ must be something you can **estimate from samples**.

**Figure intuition**

* You only see samples $x_i$ from $p_{\text{data}}$, and you tune $p_\phi$ to reduce the “gap” $D(p_{\text{data}}, p_\phi)$.

---

### 3) Forward KL divergence and Maximum Likelihood Estimation (MLE)

**Forward KL definition**

$$
D_{\mathrm{KL}}(p_{\text{data}}|p_\phi)
:=\int p_{\text{data}}(x)\log\frac{p_{\text{data}}(x)}{p_\phi(x)},dx
= \mathbb{E}_{x\sim p_{\text{data}}}\left[\log p_{\text{data}}(x)-\log p_\phi(x)\right].
$$

* **Asymmetric:**
  
  $$D_{\mathrm{KL}}(p_{\text{data}}|p_\phi)\neq D_{\mathrm{KL}}(p_\phi|p_{\text{data}})$$

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
&= -\mathbb{E}_{x\sim p_{\text{data}}}\left[\log p_\phi(x)\right]!!!\underbrace{\left(-\mathbb{E}_{x\sim p_{\text{data}}}[\log p_{\text{data}}(x)]\right)}_{\mathcal{H}(p_{\text{data}})}.
\end{aligned}
$$

- $\mathcal{H}(p_{\text{data}})$ is the **entropy** of the data distribution and does **not** depend on $\phi$.
- Therefore:

**Lemma (Minimizing KL $\iff$ MLE)**

$$
\min_\phi D_{\mathrm{KL}}(p_{\text{data}}|p_\phi)
\quad \Longleftrightarrow \quad
\max_\phi \mathbb{E}_{x\sim p*{\text{data}}}[\log p_\phi(x)]. \qquad\text{(1.1.2)}
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

$$D_F(p\mid q) := \mathbb{E}_{x\sim p}\left[\left|\nabla_x \log p(x) - \nabla_x \log q(x)\right|_2^2\right]. \qquad \text{(1.1.3)}$$

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

$$D_f(p\mid q) = \int q(x), f!\left(\frac{p(x)}{q(x)}\right)dx, \qquad f(1)=0$$

where $f:\mathbb{R}_+\to\mathbb{R}$ is **convex**. $\text{(1.1.4)}$

**Examples**
* **Forward KL:** $f(u)=u\log u \Rightarrow D_f = D_{\mathrm{KL}}(p\mid q)$
* **Jensen–Shannon (JS):**
  
  $$f(u)=\tfrac12\Big[u\log u-(u+1)\log\frac{u+1}{2}\Big] \Rightarrow D_f=D_{\mathrm{JS}}(p\mid q)$$
  
* **Total variation (TV):** $f(u)=\tfrac12|u-1| \Rightarrow D_f=D_{\mathrm{TV}}(p,q)$

#### 5.2) Explicit forms for JS and TV

* **JS divergence**
  
  $$
  D_{\mathrm{JS}}(p\mid q)
  =\tfrac12 D_{\mathrm{KL}}\left(p ,\Big|, \tfrac12(p+q)\right)
  +\tfrac12 D_{\mathrm{KL}}\left(q ,\Big|, \tfrac12(p+q)\right).
  $$

  Intuition: **smooth + symmetric**, balances both distributions, avoids some unbounded KL behavior; later useful for interpreting GANs.

* **Total variation distance**
  
  $$D_{\mathrm{TV}}(p,q) =\tfrac12\int_{\mathbb{R}^D} |p-q|dx = \sup_{A\subset \mathbb{R}^D} |p(A)-q(A)|$$

  Intuition: captures the **largest possible** difference in probability the two distributions can assign to any event $A$.

#### 5.3) Optimal transport viewpoint: Wasserstein distances

* Unlike $f$-divergences (which compare **density ratios**), **Wasserstein** distances depend on the **geometry of the sample space** and can remain meaningful even if the supports of $p$ and $q$ **do not overlap**.

---

### 6) Challenges in modeling distributions (Section 1.1.2)

To model a density $p_\phi(x)$ with a neural network, $p_\phi$ must satisfy:

1. **Non-negativity:** $p_\phi(x)\ge 0$ for all $x$.
2. **Normalization:** $\int p_\phi(x)dx = 1$.

#### Practical construction via an unnormalized “energy” output

Let the network output a scalar

$$E_\phi(x)\in\mathbb{R}.$$

Interpret it as defining an **unnormalized** density.

**Step 1: enforce non-negativity**
Use a positive mapping, commonly the exponential:

$$\tilde p_\phi(x) = \exp(E_\phi(x))$$

**Step 2: enforce normalization**

$$p_\phi(x) = \frac{\tilde p_\phi(x)}{\int \tilde p_\phi(x'),dx'} = \frac{\exp(E_\phi(x))}{\int \exp(E_\phi(x')),dx'}$$

The denominator is the **normalizing constant / partition function**:

$$Z(\phi) := \int \exp(E_\phi(x')),dx'.$$

#### Central difficulty

* In **high dimensions**, computing $Z(\phi)$ (and often its gradients) is typically **intractable**.
* This intractability is a major motivation for many DGM families: they’re designed to **avoid**, **approximate**, or **circumvent** the cost of evaluating the partition function.

---

# Variational Perspective: From VAEs to DDPMs


# Score-Based Perspective: From EBMs to NCSN

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/dsm_via_the_conditioning_technique.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>**Illustration of DSM via the conditioning technique.** By perturbing the data distribution pdata with small additive Gaussian noise $\mathcal{N}(0,σ^2I)$, the resulting conditional distribution $p_σ(\wildetilde{x}\mid x) = \mathcal{N}(\wildetilde{x}; x,σ^2I)$ admits a closed-form score function.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/ncsn.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>**Illustration of NCSN.** The forward process perturbs the data with multiple levels of additive Gaussian noise $p_σ(x_σ\mid x)$. Generation proceeds via Langevin sampling at each noise level, using the result from the current level to initialize sampling at the next lower variance.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/score_matching_inaccuracy.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>**Illustration of SM inaccuracy (revisiting Illustration of Score Matching).** the red region indicates low-density areas with potentially inaccurate score estimates due to limited sample coverage, while high-density regions tend to yield more accurate estimates.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/score_matching.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>**Illustration of Score Matching.** The neural network score $s_ϕ(x)$ is trained to match the ground truth score $s(x)$ using a MSE loss. Both are represented as vector fields.</figcaption>
</figure>


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

### Defining a density via an energy function

Let $x \in \mathbb{R}^D$ be a data point. An EBM defines an energy function $E_\phi(x)$ (parameters $\phi$).

**Normalized density:**

$$p_\phi(x) := \frac{\exp(-E_\phi(x))}{Z_\phi}, \qquad Z_\phi := \int_{\mathbb{R}^D} \exp(-E_\phi(x))dx$$

* $Z_\phi$ is the **partition function** that enforces normalization:
  
  $$\int_{\mathbb{R}^D} p_\phi(x)dx = 1$$

**Key interpretation:**
* Lower $E_\phi(x)$  $\Rightarrow$ larger $\exp(-E_\phi(x))$ $\Rightarrow$ **higher probability**.

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/ebm_training.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>**Illustration of EBM training.** The model lowers density (raises energy) at "bad" data points (red arrows), and raises density (lowers energy) at "good" data points (green arrows).</figcaption>
</figure>

#### “Only relative energies matter”

* If you add a constant $c$ to all energies, $E_\phi(x)\mapsto E_\phi(x)+c$:
  * numerator $\exp(-E_\phi(x)-c)$ and denominator $Z_\phi$ both get multiplied by $\exp(-c)$
  * $p_\phi(x)$ stays the same
    * $\implies$ EBMs are invariant to global energy shifts.

#### Global trade-off due to normalization

Because probabilities must sum to 1:
* decreasing energy in one region (increasing its probability mass) necessarily **decreases probability elsewhere**.
* EBMs therefore impose a **global coupling**: “making one valley deeper makes others shallower.”
* 

#### Maximum likelihood for EBMs — and why it’s hard

In principle, EBMs can be trained by maximum likelihood, which naturally balances fitting the data with global regularization:

$$
\mathcal{L}_{\text{MLE}}(\phi)
= \mathbb{E}_{p_{\text{data}}(x)}
\left[
\log \frac{\exp(-E_\phi(x))}{Z_\phi}
\right]
$$

Expanding:

$$\mathcal{L}_{\text{MLE}}(\phi) = -\mathbb{E}_{p_{\text{data}}}[E_\phi(x)] - \log \int \exp(-E_\phi(x))dx$$

Interpretation:
* **Term 1:** lowers energy on real data
* **Term 2:** enforces normalization via $Z_\phi$ (a kind of global regularization)

**The bottleneck:**
* In high dimensions, $\log Z_\phi$ and especially its gradient are **intractable**
* Because computing gradients involves expectations under the **model distribution**.
* This motivates alternatives:
  * approximate the hard term (e.g. contrastive divergence)
  * or bypass it entirely via **score matching**

### The score function

#### Definition

For a density $p(x)$ on $\mathbb{R}^D$, the **score** is:

$$s(x) := \nabla_x \log p(x), \qquad s:\mathbb{R}^D \to \mathbb{R}^D$$

**Intuition:**
* $s(x)$ forms a vector field pointing in the direction where $\log p(x)$ increases fastest,
* i.e. where the probability density increases.

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/score_vector_fields.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>**Illustration of score vector fields.** Scor evector fields $∇_x \log p(x)$ indicate directions of increasing density.</figcaption>
</figure>

#### Why model scores instead of densities?

**Benefit 1: No normalization constant needed**

Many models are only known up to an unnormalized density $\tilde p(x)$:

$$p(x) = \frac{\tilde p(x)}{Z}, \qquad Z = \int \tilde p(x)dx$$

Then:

$$\nabla_x \log p(x) = \nabla_x \log \tilde p(x) - \nabla_x \log Z \nabla_x \log \tilde p(x)$$

because $Z$ is constant in $x$.
$\implies$ **The score ignores the partition function.**

**Benefit 2: The score is a complete representation (up to a constant)**

Since $s(x) = \nabla_x \log p(x)$, you can recover $\log p(x)$ (up to an additive constant) by integrating the score:

$$\log p(x) = \log p(x_0) + \int_0^1 s\big(x_0 + t(x-x_0)\big)^\top (x-x_0)dt$$

* $x_0$ is a reference point.
* $\log p(x_0)$ is fixed by normalization.

$\implies$ Modeling the score can be as expressive as modeling the density $p(x)$ itself, while often more tractable for generative modeling.

### EBMs + scores: the crucial simplification

For an EBM:

$$p_\phi(x) = \frac{\exp(-E_\phi(x))}{Z_\phi}$$

Take logs:

$$\log p_\phi(x) = -E_\phi(x) - \log Z_\phi$$

Differentiate w.r.t. $x$:

$$\nabla_x \log p_\phi(x) = -\nabla_x E_\phi(x)$$

because $\nabla_x \log Z_\phi = 0$.
$\implies$ **The model score equals $-\nabla_x E_\phi(x)$** and does **not** depend on $Z_\phi$.

### Training EBMs via score matching

#### Core score matching objective

Score matching trains by aligning:
* model score $\nabla_x \log p_\phi(x)$
* with the (unknown) data score $\nabla_x \log p_{\text{data}}(x)$

Objective:

$$\mathcal{L}_{\text{SM}}(\phi) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)} \left[ \lvert\nabla_x \log p_\phi(x) - \nabla_x \log p_{\text{data}}(x)\rvert^2 \right]$$

#### How can this work if the data score is unknown?

Using integration by parts, this becomes an equivalent expression depending only on the energy and its derivatives:

$$\mathcal{L}_{\text{SM}}(\phi) = \mathbb{E}_{p_{\text{data}}(x)} \left[ \mathrm{Tr}\left(\nabla_x^2 E_\phi(x)\right) + \frac{1}{2}\lvert\nabla_x E_\phi(x)\rvert^2 \right] + C$$

- $\nabla_x^2 E_\phi(x)$ is the **Hessian**
- $C$ is a constant independent of $\phi$

#### Pros / cons

**Pros**
* Eliminates the partition function $Z_\phi$
* Avoids sampling from the model during training

**Main drawback**
* Requires **second-order derivatives** (Hessians / traces), which can be expensive in high dimensions.

### Langevin sampling with score functions (what this section is setting up)

* Sampling from EBMs can be performed using **Langevin dynamics**.
* The text indicates they will:
  1. present a **discrete-time Langevin update**
  2. then its **continuous-time limit** as an **SDE**
  3. and explain the intuition for why this explores complex energy landscapes efficiently

A useful mental model:
* deterministic part: move “uphill in probability” (follow the score / descend energy)
* stochastic part: add noise to keep exploring and not get stuck.

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/langevin_sampling.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>**Illustration of Langevin sampling.** Langevin sampling using the score function $∇_x \log p_ϕ(x)$ to guide trajectories toward high-density regions via the update in Equation (3.1.5) (indicating by arrows).</figcaption>
</figure>

#### Discrete-time Langevin dynamics (the practical sampler)

**Update rule (energy form):**

$$x_{n+1}=x_n-\eta\nabla_x E_\phi(x_n)+\sqrt{2\eta}\varepsilon_n,\qquad \varepsilon_n\sim\mathcal N(0,I)$$

* $x_0$ is initialized from some easy distribution (often Gaussian).
* $\eta>0$ is the step size.
* Noise term prevents getting stuck in local minima by adding randomness.

**Same update in score form** (using $\nabla_x\log p_\phi(x)=-\nabla_x E_\phi(x)$):

$$x_{n+1}=x_n+\eta\nabla_x\log p_\phi(x_n)+\sqrt{2\eta}\varepsilon_n$$

Interpretation:
* **Deterministic part:** takes a small step **toward higher probability density** (gradient ascent on $\log p_\phi$).
* **Stochastic part:** adds Gaussian exploration to cross energy barriers.

$$\boxed{\text{This “score + noise” form is the bridge to diffusion/score-based models.}}$$

#### Continuous-time Langevin dynamics (SDE limit)

As $\eta\to 0$, the discrete updates converge to the **Langevin SDE**:

$$dx(t)=\nabla_x\log p_\phi(x(t))dt+\sqrt{2}dw(t)$$

Equivalently in energy form:

$$dx(t)=-\nabla_x E_\phi(x(t))dt+\sqrt{2}dw(t)$$

* $w(t)$ is **standard Brownian motion** (a Wiener process).
* The distribution of $x(t)$ converges (under standard regularity assumptions, e.g. confining smooth $E_\phi$) to $p_\phi$ as $t\to\infty$.
  So sampling = simulate this process long enough.

#### Discrete update = Euler–Maruyama discretization

Brownian increments satisfy:

$$w(t+\eta)-w(t)\sim\mathcal N(0,\eta I)$$

So $\sqrt{2},dw(t)$ over a step $\eta$ becomes:

$$\sqrt{2},(w(t+\eta)-w(t))=\sqrt{2\eta}\varepsilon_n,\quad \varepsilon_n\sim\mathcal N(0,I)$$

That’s exactly where the **$\sqrt{\eta}$** scaling (and the $\sqrt{2\eta}$) in the discrete update comes from.

#### Why Langevin sampling works (physics intuition)

Think of $E_\phi(x)$ as a **potential energy landscape**.

**Pure deterministic dynamics (gradient flow / “Newtonian” lens):**

$$dx(t)=-\nabla_x E_\phi(x(t)),dt$$

* Always moves “downhill” in energy → ends up in a **local minimum**.
* Bad for sampling multimodal distributions (gets trapped).

**Add noise → Langevin:**

$$dx(t)=-\nabla_x E_\phi(x(t))dt+\sqrt{2}dw(t)$$

* Noise helps escape local minima by crossing energy barriers.
* The stationary distribution becomes the **Boltzmann distribution**:
  
  $$p_\phi(x)\propto e^{-E_\phi(x)}$$

#### Stationarity and the special role of $\sqrt{2}$

A key property mentioned in the text:

* With the **$\sqrt{2}$** factor, $p_\phi$ is unchanged over time (stationary):

  * If $x(0)\sim p_\phi$, then $x(t)\sim p_\phi$ for all $t\ge 0$.

This is also reflected via the **Fokker–Planck equation** for the density $\rho(x,t)$:

$$\partial_t\rho = -\nabla\cdot(\rho,\nabla\log p_\phi) + \frac{\sigma^2}{2}\Delta \rho$$

Setting $\rho=p_\phi$ gives a condition that holds only for the right noise scale, i.e. $\sigma=\sqrt{2}$ (as stated).

#### Why it can be hard in practice (inherent challenges)

Even though Langevin is conceptually clean, it can be inefficient in high dimensions:

* **Sensitive hyperparameters:** efficiency depends strongly on

  * step size $\eta$,
  * noise scale (here tied to $\eta$),
  * number of iterations.
* **Poor mixing time:** if the target has many separated modes,

  * local stochastic steps struggle to move between distant high-probability regions,
  * mixing gets much worse as dimension grows,
  * can miss diversity (mode dropping in practice).

This motivates **more structured / guided sampling**—which is exactly where diffusion/score-based methods come in (they guide samples through a sequence of easier, noise-smoothed distributions).

---

## Quick cheat sheet

* **EBM score:** $\nabla\log p_\phi(x)=-\nabla E_\phi(x)$
* **Discrete Langevin (ULA):**
  
  $$x_{n+1}=x_n+\eta,\nabla\log p_\phi(x_n)+\sqrt{2\eta},\varepsilon_n$$
  
* **Continuous Langevin SDE:**
  
  $$dx=\nabla\log p_\phi(x),dt+\sqrt{2},dw$$
  
* **Discrete is Euler–Maruyama** for the SDE.
* **Noise helps escape minima**; $\sqrt{2}$ makes $p_\phi$ stationary.

# Quick cheat sheet (key equations)

* **EBM density**
  
  $$p_\phi(x)=\frac{e^{-E_\phi(x)}}{Z_\phi},\quad Z_\phi=\int e^{-E_\phi(x)}dx$$
  
* **Score definition**
  
  $$s(x)=\nabla_x \log p(x)$$
  
* **Score ignores normalization**
  
  $$p=\tilde p/Z \Rightarrow \nabla_x\log p=\nabla_x\log \tilde p$$
  
* **EBM score**
  
  $$\nabla_x\log p_\phi(x)=-\nabla_x E_\phi(x)$$
  
* **Score matching**
  
  $$\mathcal{L}*{SM}=\tfrac12 \mathbb{E}_{p_{data}}| \nabla_x\log p_\phi - \nabla_x\log p_{data}|^2$$
  
  Equivalent:
  
  $$\mathcal{L}*{SM}=\mathbb{E}_{p_{data}}\left[\mathrm{Tr}(\nabla_x^2E_\phi)+\tfrac12|\nabla_xE_\phi|^2\right]+C$$
  

---

## Mini self-check questions

1. Why does adding a constant to $E_\phi(x)$ not change $p_\phi(x)$?
2. Write $\log p_\phi(x)$ for an EBM and show why the score does not depend on $Z_\phi$.

## Study Notes — Section 3.2: From Energy-Based to Score-Based Generative Models

### 0) Big picture

* **Key message:** to *generate* samples (e.g., via Langevin dynamics), you don’t need the full normalized density $p(x)$. You only need the **score**
  
  $$s(x)=\nabla_x \log p(x)$$

  which points toward **higher-probability (higher log-density)** regions.
* **Why move away from energies?**

  * EBMs define $p_\theta(x)\propto e^{-E_\theta(x)}$. The **partition function** is hard, but the **score** is easy:
    
    $$\nabla_x \log p_\theta(x)= -\nabla_x E_\theta(x)\quad(\text{no partition function term in } \nabla_x)$$
    
  * However, **training through an energy** with score matching tends to require **second derivatives** (Hessians).
* **Core shift:** since sampling uses only the score, we can **learn the score directly** with a neural network $s_\phi(x)$. This is the foundation of **score-based generative models**.

---

## 1) Notation / operators cheat sheet

Let $x\in\mathbb{R}^D$, and $s_\phi(x)\in\mathbb{R}^D$.

* **Score:** $s(x)=\nabla_x\log p_{\text{data}}(x)$
* **Jacobian of a vector field:** $\nabla_x s_\phi(x)\in\mathbb{R}^{D\times D}$ with entries $\frac{\partial (s_\phi)_i}{\partial x_j}$
* **Trace of Jacobian = divergence:**
  
  $$\mathrm{Tr}(\nabla_x s_\phi(x))=\sum_{i=1}^D \frac{\partial (s_\phi)_i}{\partial x_i} = \nabla\cdot s_\phi(x)$$
  
* If $s_\phi=\nabla_x u$ for scalar $u$, then $\nabla_x s_\phi = \nabla_x^2 u$ (the Hessian), and
  
  $$\nabla\cdot s_\phi = \mathrm{Tr}(\nabla_x^2 u)=\Delta u \quad(\text{Laplacian})$$
  

## 2) Score Matching objective (3.2.1)

### Goal

Approximate the **unknown** true score $s(x)=\nabla_x \log p_{\text{data}}(x)$ from samples $x\sim p_{\text{data}}$ using a neural net $s_\phi(x)$.

### Direct (infeasible) regression view

$$
\mathcal{L}_{\mathrm{SM}}(\phi)
=\frac{1}{2},\mathbb{E}*{x\sim p_{\text{data}}}\Big[;|s_\phi(x)-s(x)|_2^2;\Big].
$$

* Looks like ordinary MSE regression, **but** the target $s(x)$ is unknown.

**Figure intuition (vector field picture):** the score field is drawn as arrows flowing toward high-density regions (modes). Training aims to match that vector field.

## 3) Hyvärinen’s tractable score matching (Proposition 3.2.1, eq. 3.2.2)

### Key result (integration by parts trick)

Hyvärinen & Dayan (2005) show:

$$\mathcal{L}_{\mathrm{SM}}(\phi)=\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi)+C,$$

where $C$ does **not** depend on $\phi$, and

$$\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi) =\mathbb{E}_{x\sim p_{\text{data}}}\left[\mathrm{Tr}\big(\nabla_x s_\phi(x)\big)+\frac{1}{2}|s_\phi(x)|*2^2\right]$$

So you can minimize $\tilde{\mathcal{L}}_{\mathrm{SM}}$ **using only samples** $x\sim p_{\text{data}}$, without ever knowing the true score.

### What is the minimizer?

The optimal solution is the true score:

$$s^*(\cdot)=\nabla_x \log p(\cdot)$$

### Why this helps computationally

* If you parameterize an **energy** $E_\theta$ and set $s_\theta=-\nabla_x E_\theta$, then
  $\mathrm{Tr}(\nabla_x s_\theta)= -\mathrm{Tr}(\nabla_x^2 E_\theta)$: **second derivatives** of the energy.
* If you parameterize $s_\phi$ **directly**, $\mathrm{Tr}(\nabla_x s_\phi)$ uses **first derivatives** of the score network output w.r.t. input $x$ (still not cheap, but avoids “derivative-of-a-derivative” through an energy).

## 4) Intuition for the two terms in (\tilde{\mathcal{L}}_{\mathrm{SM}})

$$
\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi)
=\mathbb{E}_{p_{\text{data}}}\left[\underbrace{\mathrm{Tr}(\nabla_x s_\phi(x))}_{\text{divergence term}}+\underbrace{\frac{1}{2}|s_\phi(x)|*2^2}_{\text{magnitude term}}\right]
$$

### (A) “Stationarity from the magnitude term”

* The expectation is under $p_{\text{data}}$, so **high-density regions dominate**.
* Minimizing $\frac12|s_\phi(x)|^2$ pushes
  
  $$s_\phi(x)\to 0 \quad \text{in high-probability regions}$$
  
* Points where $s_\phi(x)=0$ are **stationary points** of the learned flow (no deterministic drift there).

### (B) “Concavity / sinks from the divergence term”

* The term $\mathrm{Tr}(\nabla_x s_\phi(x))=\nabla\cdot s_\phi(x)$ encourages **negative divergence** in high-density regions.
* **Negative divergence** means nearby vectors **converge** (flow contracts) rather than spread out → stationary points become **attractive sinks**.

#### Making it precise when (s_\phi\approx\nabla_x u)

Assume $s_\phi=\nabla_x u$. Then:

* $\nabla_x s_\phi = \nabla_x^2 u$ (Hessian)
* $\nabla\cdot s_\phi = \mathrm{Tr}(\nabla_x^2 u)$

At a stationary point $x_\star$ where $\nabla_x u(x_\star)=0$, Taylor expansion:

$$u(x)=u(x_\star)+\frac{1}{2}(x-x_\star)^\top \nabla_x^2u(x_\star)(x-x_\star)+o(|x-x_\star|^2)$$

* If $\nabla_x^2u(x_\star)$ is **negative definite**, then $u$ is locally concave → log-density has a **strict local maximum** there.
* Negative definite Hessian $\implies$ all eigenvalues negative $\implies$ trace negative $\implies$ $\mathrm{Tr}(\nabla_x^2u(x_\star))<0$.

**Important nuance (from the footnote):**

* $\mathrm{Tr}(\nabla_x^2u)<0$ only means the **sum** of eigenvalues is negative.
* Some eigenvalues can still be positive → could be a **saddle** rather than a true maximum.

---

## 5) Sampling with Langevin dynamics (3.2.2)

Once you have a trained score model $s_{\phi^*}(x)$, you can sample by iterating:

$$x_{n+1}=x_n+\eta, s_{\phi^*}(x_n)+\sqrt{2\eta},\varepsilon_n,\quad \varepsilon_n\sim\mathcal{N}(0,I).$$

* $\eta>0$ is the step size.
* Deterministic part (\eta s(x)): moves “uphill” in log-density.
* Noise $\sqrt{2\eta}\varepsilon$: keeps exploration and yields the correct stationary distribution (in the idealized limit).

### Continuous-time view (Langevin SDE)

This recursion is Euler–Maruyama discretization of:

$$dx(t)= s_{\phi^*}(x(t))dt+\sqrt{2}dw(t)$$

where $w(t)$ is a standard Wiener process (Brownian motion).

* As $\eta\to 0$, discrete and continuous formulations coincide.
* In practice, you can run the discrete sampler or simulate the SDE.

## 6) Prologue / why this matters for diffusion models (3.2.3)

* The **score function** started as a way to train EBMs efficiently.
* It has become the **central object** in modern **score-based diffusion models**:

  * Theoretical formulation + practical implementation are built around learning scores
  * Generation becomes “simulate (reverse) stochastic processes using learned scores”

## 7) Quick “exam-ready” summary

* **Score:** $s(x)=\nabla_x\log p(x)$, points toward higher density.
* **Score matching (naive):** minimize $\mathbb{E}|s_\phi(x)-s(x)|^2$ (but $s$ unknown).
* **Hyvärinen trick:** minimize instead
  
  $$\mathbb{E}\left[\nabla\cdot s_\phi(x)+\tfrac12|s_\phi(x)|^2\right]$$
  
  (same optimum up to constant).
* **Intuition:** magnitude term makes $s_\phi\approx 0$ in high-density regions (stationary points); divergence term makes them **sinks** (contracting flow).
* **Sampling:** Langevin update $x_{n+1}=x_n+\eta s(x_n)+\sqrt{2\eta}\varepsilon$ = Euler–Maruyama for $dx=s(x)dt+\sqrt2dw$.
* **Big shift:** learn **scores directly** $\implies$ foundation of score-based generative models and diffusion.

If you want, I can also turn these into a 1-page “formula sheet” (definitions + key equations only) for quick revision.

## Study notes: Denoising Score Matching (DSM) + Sliced Score Matching (Hutchinson)

These notes summarize what’s on the attached pages (Section **3.3 Denoising Score Matching**, incl. motivation, sliced score matching, and the DSM objective + Gaussian special case).

---

# 0) Notation & goal

* Data: $x \in \mathbb{R}^D,\quad x \sim p_{\text{data}}(x)$
* **Score (of a density (p))**:
  
  $$\nabla_x \log p(x)$$
  
* A neural net $s_\phi(\cdot)$ (or $s_\phi(\cdot;\sigma)$) is trained to approximate a score field.

**Core aim:** learn a vector field that points toward higher-density regions of the data distribution (or a smoothed/noisy version of it), so we can later generate samples by moving “uphill” in log-density (via Langevin dynamics / diffusion sampling, etc.).

---

# 1) Why vanilla score matching is hard

The “direct” score matching loss is

$$
\mathcal{L}_{\text{SM}}(\phi)
= \frac12,\mathbb{E}_{x\sim p_{\text{data}}}\Big[|s_\phi(x)-\nabla_x \log p_{\text{data}}(x)|*2^2\Big],
$$

but $\nabla_x \log p_{\text{data}}(x)$ is **unknown/intractable** because $p_{\text{data}}$ is unknown.

A classic workaround (Hyvärinen-style) is an equivalent objective that removes the explicit data-score target but introduces a **trace-of-Jacobian** term:

$$
\tilde{\mathcal{L}}_{\text{SM}}(\phi)
=\mathbb{E}_{x\sim p_{\text{data}}}\Big[\mathrm{Tr}(\nabla_x s_\phi(x))+\frac12|s_\phi(x)|_2^2\Big].
$$

### Problem

* Computing $\mathrm{Tr}(\nabla_x s_\phi(x))$ (trace of the Jacobian of a $D$-dimensional vector field) has **worst-case complexity $\mathcal{O}(D^2)$** $\implies$ not scalable in high dimensions.

---

# 2) Sliced score matching via Hutchinson’s estimator

## 2.1 Hutchinson identity (trace estimator)

Let $u \in \mathbb{R}^D$ be **isotropic** with:

$$\mathbb{E}[u]=0,\qquad \mathbb{E}[uu^\top]=I$$

(e.g. **Rademacher** entries $\pm 1$ or standard Gaussian).

Then for any square matrix $A$,

$$\mathrm{Tr}(A)=\mathbb{E}_u[u^\top A u]$$

Also, for any vector $v$,

$$
\mathbb{E}_u[(u^\top v)^2]
= v^\top\mathbb{E}[uu^\top]v
= |v|_2^2.
$$

## 2.2 Applying it to score matching

Using $A = \nabla_x s_\phi(x)$, the objective becomes (exactly, in expectation):

$$
\tilde{\mathcal{L}}_{\text{SM}}(\phi)
= \mathbb{E}_{x,u}\Big[
u^\top(\nabla_x s_\phi(x))u
+\frac12 (u^\top s_\phi(x))^2
\Big].
$$

### Practical computation (why this helps)

* You don’t form the full Jacobian.
* You compute **directional derivatives** and **Jacobian–vector products**:

  * $(\nabla_x s_\phi(x))u$ via JVP
  * then dot with $u$ to get $u^\top(\nabla_x s_\phi(x))u$

If you average over $K$ random probes $u$, you get:

* **unbiased** estimator
* variance $\mathcal{O}(1/K)$

### Key intuition

You “test” the model’s behavior only along **random directions** (“random slices”), rather than fully constraining all partial derivatives.

## 2.3 Limitations (from the text)

Even if it avoids explicit Jacobians:

* It still relies on the **raw data distribution**.
* For image-like data that may lie on a **low-dimensional manifold**, the score
  $\nabla_x \log p_{\text{data}}(x)$ can be **undefined or unstable**.
* It mainly constrains the vector field **at observed points**, giving weaker control in neighborhoods.
* Has probe-induced variance and repeated JVP/VJP compute costs.

This motivates DSM as a more robust alternative.

---

# 3) Denoising Score Matching (DSM): Vincent (2011)

## 3.1 Conditioning trick: corrupt the data with known noise

Introduce a **known corruption kernel**:

$$\tilde x \sim p_\sigma(\tilde x \mid x)$$

where $\sigma>0$ controls noise scale.

This defines a **perturbed (smoothed) marginal** distribution:

$$p_\sigma(\tilde x) = \int p_\sigma(\tilde x \mid x)p_{\text{data}}(x)dx$$

Train a model $s_\phi(\tilde x;\sigma)$ to approximate the **score of the marginal**:

$$s_\phi(\tilde x;\sigma)\approx \nabla_{\tilde x}\log p_\sigma(\tilde x)$$

## 3.2 “Marginal” score matching at noise level $\sigma$

A natural objective is

$$
\mathcal{L}_{\text{SM}}(\phi;\sigma)
= \frac12,\mathbb{E}_{\tilde x\sim p_\sigma}\Big[
|s_\phi(\tilde x;\sigma) - \nabla_{\tilde x}\log p_\sigma(\tilde x)|*2^2
\Big],
$$

but $\nabla_{\tilde x}\log p_\sigma(\tilde x)$ is still generally intractable.

## 3.3 DSM objective (tractable target)

Vincent’s key result: **condition on the clean sample $x$** and replace the intractable marginal score target with the **conditional score** of the corruption kernel (which we choose and thus know):

$$
\boxed{
\mathcal{L}_{\text{DSM}}(\phi;\sigma)
:= \frac12,\mathbb{E}_{x\sim p_{\text{data}},;\tilde x\sim p_\sigma(\cdot\mid x)}
\Big[
|s_\phi(\tilde x;\sigma)-\nabla_{\tilde x}\log p_\sigma(\tilde x\mid x)|_2^2
\Big]
}
$$

This is the **denoising** viewpoint: the target $\nabla_{\tilde x}\log p_\sigma(\tilde x\mid x)$ tends to point from noisy $\tilde x$ back toward clean $x$.

# 4) Theorem: DSM is equivalent to marginal score matching (up to a constant)

For any fixed $\sigma>0$:

$$\mathcal{L}_{\text{SM}}(\phi;\sigma) = \mathcal{L}_{\text{DSM}}(\phi;\sigma) + C,$$

where $C$ does **not** depend on $\phi$.

So minimizing DSM is effectively minimizing the marginal score matching objective.

Also, the minimizer satisfies (almost everywhere in $\tilde x$):

$$s^*(\tilde x;\sigma)=\nabla_{\tilde x}\log p_\sigma(\tilde x),$$

i.e. the learned model recovers the correct marginal score at that noise level.

**Proof idea (as stated):** expand both MSEs; all $\phi$-dependent terms match and the remainder collapses to a $\phi$-independent constant.

# 5) Special case: additive Gaussian noise (the key diffusion-model case)

Assume the corruption is Gaussian:

$$
\tilde x = x + \sigma \varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,I),
$$

so

$$p_\sigma(\tilde x\mid x)=\mathcal{N}(\tilde x;,x,\sigma^2 I)$$


### Conditional score has closed form

$$\nabla_{\tilde x}\log p_\sigma(\tilde x\mid x) = \frac{x-\tilde x}{\sigma^2}$$

Plugging into DSM gives:

$$
\boxed{
\mathcal{L}_{\text{DSM}}(\phi;\sigma)
= \frac12,\mathbb{E}_{x,\tilde x}\Big[
\big|s_\phi(\tilde x;\sigma)-\frac{x-\tilde x}{\sigma^2}\big|*2^2
\Big]
= \frac12,\mathbb{E}_{x,\varepsilon}\Big[
\big|s_\phi(x+\sigma\varepsilon;\sigma)+\frac{\varepsilon}{\sigma}\big|_2^2
\Big].
}
$$

This objective is described on the page as forming the **core of score-based diffusion models**.

### Key intuition for $\sigma\to 0$

As $\sigma\approx 0$,
* $p_\sigma(\tilde x)\approx p_{\text{data}}(x)$
* so
  
  $$s^*(\tilde x;\sigma)=\nabla_{\tilde x}\log p_\sigma(\tilde x)\approx \nabla_x\log p_{\text{data}}(x)$$
  

Meaning: learning scores of slightly-noised data recovers (approximately) the true data score.

---

# 6) Conceptual takeaway (how to remember DSM)

### “Conditioning technique” (Insight 3.3.1 on the page)

* You turn an intractable objective involving an unknown marginal score into a tractable regression by conditioning on a clean data point $x$.
* Similar conditioning ideas appear in diffusion-model variational views and also relate to other modern generative-training paradigms.

### Why DSM is more robust than raw score matching

* Adding noise makes the distribution **smooth/full-dimensional**, which avoids the “score undefined on a manifold” issue.
* Training constrains the score field in **neighborhoods** around data, not only exactly on the data.

---

# 7) Minimal “training recipe” (Gaussian DSM, from the formulas)

For a fixed $\sigma$:

1. Sample $x\sim p_{\text{data}}$
2. Sample $\varepsilon\sim\mathcal{N}(0,I)$
3. Form $\tilde x = x+\sigma\varepsilon$
4. Target is $-\varepsilon/\sigma$ (equivalently $(x-\tilde x)/\sigma^2$)
5. Minimize:
   
   $$\frac12\left|s_\phi(\tilde x;\sigma)+\frac{\varepsilon}{\sigma}\right|_2^2$$
   

(Extensions usually train over many $\sigma$ values, but that part isn’t shown on these pages.)



















---


## Study notes (from the attached pages)

### Notation used on these pages

* Clean data: $x \sim p_{\text{data}}$
* Noisy/corrupted observation: $\tilde x$
* Noisy marginal (a “smoothed” data distribution):
  
  $$p_\sigma(\tilde x) ;=; \int p(\tilde x\mid x),p_{\text{data}}(x)dx$$
  
* **Score** of a density $p$: $\nabla_{\tilde x}\log p(\tilde x)$
* Learned score network at noise level $\sigma$: $s_\phi(\tilde x;\sigma)\approx \nabla_{\tilde x}\log p_\sigma(\tilde x)$

---

## 3.3.3 Sampling with a trained score model (Langevin dynamics)

### Goal

Generate samples from (approximately) $p_{\text{data}}$ using a learned approximation to the score.

### Langevin dynamics update (discrete-time)

Given a score model $s_\phi(\cdot;\sigma)$ at a fixed noise level $\sigma$, iterate

$$\tilde x_{n+1} = \tilde x_n + \eta, s_\phi(\tilde x_n;\sigma);+;\sqrt{2\eta},\varepsilon_n, \qquad \varepsilon_n\sim\mathcal N(0,I)$$

* $\eta>0$ here is the **step size** (careful: later pages reuse $\eta$ for “natural parameter” in exponential families).
* This is Langevin sampling where the “force” term $\nabla \log p_\sigma(\tilde x)$ is replaced by the learned $s_\phi$.

### Interpretation

* The deterministic part $\eta,s_\phi(\tilde x_n;\sigma)$ pushes samples **toward higher probability regions** of $p_\sigma$.
* The noise term $\sqrt{2\eta}\varepsilon_n$ ensures exploration and makes the Markov chain target $p_\sigma$ (in the idealized limit).

### What distribution do you sample from?

* With the *true* score and suitable conditions, Langevin dynamics has stationary distribution $p_\sigma$.
* If $\sigma$ is **small**, then $p_\sigma$ is close to $p_{\text{data}}$, so after enough iterations $\tilde x_n$ can approximate samples from $p_{\text{data}}$.

### Minimal pseudocode

1. Initialize $\tilde x_0$ (often random noise).
2. For $n=0,\dots,N-1$:
   $\tilde x \leftarrow \tilde x + \eta, s_\phi(\tilde x;\sigma) + \sqrt{2\eta},\varepsilon$, with $\varepsilon\sim\mathcal N(0,I)$.
3. Output $\tilde x_N$.

---

## Advantages of injecting noise (why use $p_\sigma$ instead of $p_{\text{data}}$ directly?)

Compared to “vanilla” score matching on the original data distribution, adding Gaussian noise to define $p_\sigma$ gives:

### 1) Well-defined gradients (scores exist everywhere)

* Real data often lies near a **low-dimensional manifold** in $\mathbb R^D$, so $\nabla\log p_{\text{data}}(x)$ can be ill-behaved/off-manifold.
* Convolving with Gaussian noise spreads mass over all of $\mathbb R^D$, making $p_\sigma$ have **full support**.
* Therefore the score $\nabla_{\tilde x}\log p_\sigma(\tilde x)$ is (typically) **well-defined everywhere**.

### 2) Improved coverage between modes

* Noise **smooths** the distribution, filling in low-density “gaps” between separated modes.
* This improves training signal and helps Langevin dynamics move through low-density regions more effectively (less getting stuck).

---

## 3.3.4 Why DSM is denoising: Tweedie’s formula

### Setup (Gaussian corruption with scaling)

Assume:
* $x\sim p_{\text{data}}$
* $\tilde x\mid x \sim \mathcal N(\alpha x,\sigma^2 I)$, with $\alpha\neq 0$

Define the noisy marginal:

$$p_\sigma(\tilde x) = \int \mathcal N(\tilde x;\alpha x,\sigma^2 I),p_{\text{data}}(x)dx$$

### Lemma (Tweedie’s formula)

$$\alpha,\mathbb E[x\mid \tilde x] = \tilde x + \sigma^2 \nabla_{\tilde x}\log p_\sigma(\tilde x)$$

Equivalently, the **posterior mean / denoiser** is

$$\mathbb E[x\mid \tilde x] = \frac{1}{\alpha}\Big(\tilde x + \sigma^2 \nabla_{\tilde x}\log p_\sigma(\tilde x)\Big)$$

### Key intuition (why this is “denoising”)

* The score $\nabla_{\tilde x}\log p_\sigma(\tilde x)$ points toward regions where noisy samples are more likely.
* Moving $\tilde x$ by a step of size $\sigma^2$ in the score direction produces the **conditional mean of the clean signal** (up to the $\alpha$ scaling).

### Connection to DSM-trained score networks

If DSM gives $s_\phi(\tilde x)\approx \nabla_{\tilde x}\log p_\sigma(\tilde x)$, then an estimated denoiser is:

$$\widehat{x}(\tilde x) = \frac{1}{\alpha}\Big(\tilde x + \sigma^2, s_\phi(\tilde x)\Big)$$

So: **learning the score is (almost directly) learning a denoiser** via Tweedie.

---

## (Optional) Higher-order Tweedie via an exponential-family view

### Exponential family observation model

Assume the conditional law of $\tilde x$ given a latent natural parameter $\eta\in\mathbb R^D$ is

$$q_\sigma(\tilde x\mid \eta) ;=; \exp(\eta^\top \tilde x - \psi(\eta)),q_0(\tilde x)$$

* $q_0(\tilde x)$ is the **base measure** (independent of $\eta$).
* For additive Gaussian noise with variance $\sigma^2 I$,
  
  $$q_0(\tilde x) = (2\pi\sigma^2)^{-D/2}\exp!\left(-\frac{|\tilde x|^2}{2\sigma^2}\right)$$

Let $p(\eta)$ be a prior over $\eta$. The noisy marginal is

$$p_\sigma(\tilde x) = \int q_\sigma(\tilde x\mid \eta),p(\eta),d\eta$$

Define the “log-normalizer in $\tilde x$”:

$$\lambda(\tilde x) := \log p_\sigma(\tilde x) - \log q_0(\tilde x)$$

Then the posterior has the form

$$p(\eta\mid \tilde x)\propto \exp(\eta^\top \tilde x - \psi(\eta) - \lambda(\tilde x))p(\eta)$$

### Derivatives of (\lambda) give posterior cumulants

A core exponential-family identity:

* $\nabla_{\tilde x}\lambda(\tilde x) ;=; \mathbb E[\eta\mid \tilde x]$
  
* $\nabla_{\tilde x}^2\lambda(\tilde x) ;=; \mathrm{Cov}[\eta\mid \tilde x]$
* More generally:
  
  $$\nabla_{\tilde x}^{(k)}\lambda(\tilde x) = \kappa_k(\eta\mid \tilde x),\quad k\ge 3$$
  
  where $\kappa_k$ are conditional cumulants.

### Specialize to Gaussian location noise (recover classic Tweedie + covariance)

For Gaussian location models, one can take $\eta = x/\sigma^2$. Then:

* Posterior mean:
  
  $$\mathbb E[x\mid \tilde x] ;=; \tilde x + \sigma^2 \nabla_{\tilde x}\log p_\sigma(\tilde x)$$
  
* Posterior covariance:
  
  $$\mathrm{Cov}[x\mid \tilde x] ;=; \sigma^2 I ;+; \sigma^4 \nabla_{\tilde x}^2\log p_\sigma(\tilde x)$$
  
* Higher cumulants scale with higher derivatives of $\log p_\sigma(\tilde x)$.

**Takeaway:** not only denoising (mean), but also **uncertainty estimates** (covariance) and higher statistics are encoded in higher-order “scores” (higher derivatives).

---

## Quick “what to remember” checklist

* **Sampling:** $\tilde x_{n+1}=\tilde x_n+\eta s_\phi(\tilde x_n;\sigma)+\sqrt{2\eta}\varepsilon_n$.
* **Why noise helps:** (i) score well-defined everywhere, (ii) smoother landscape improves mode coverage.
* **Tweedie:** $\mathbb E[x\mid \tilde x]=\frac{1}{\alpha}(\tilde x+\sigma^2\nabla_{\tilde x}\log p_\sigma(\tilde x))$.
* **DSM ⇒ denoiser:** replace $\nabla \log p_\sigma$ by $s_\phi$.
* **Higher-order:** derivatives of $\log p_\sigma$ relate to posterior covariance and cumulants.

If you want, I can turn these into a 1–2 page “exam-style” cheat sheet (definitions + boxed formulas + common pitfalls).


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

$$R(\mathbf D;\mathbf x):=\mathbb E_{\tilde{\mathbf x}\mid \mathbf x}\Big[|\mathbf D(\tilde{\mathbf x})-\mathbf x|_2^2\ \big|\ \mathbf x\Big]$$

Problem: this depends on $\mathbf x$, so you can’t compute it from $\tilde{\mathbf x}$ alone.

---

## SURE: an observable surrogate for the MSE

**Stein’s Unbiased Risk Estimator (SURE)** provides:

$$
\mathrm{SURE}(\mathbf D;\tilde{\mathbf x}) = |\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x}|*2^2 + 2\sigma^2,\nabla_{\tilde{\mathbf x}}\cdot \mathbf D(\tilde{\mathbf x})!!!d\sigma^2.
$$

* $\nabla_{\tilde{\mathbf x}}\cdot \mathbf D(\tilde{\mathbf x})$ is the **divergence** of $\mathbf D$:
  
  $$\nabla_{\tilde{\mathbf x}}\cdot \mathbf D(\tilde{\mathbf x})=\sum_{i=1}^d \frac{\partial D_i(\tilde{\mathbf x})}{\partial \tilde{x}_i}$$
  
* Importantly: **SURE depends only on (\tilde{\mathbf x})** (and $\sigma$), not on $\mathbf x$.

### Why the terms make sense (intuition)

* $|\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x}|^2$: how much the denoiser changes the input.

  * Alone, it *underestimates* true error because $\tilde{\mathbf x}$ is already corrupted.
* $2\sigma^2 \nabla\cdot \mathbf D(\tilde{\mathbf x})$: **correction term** accounting for noise variance via sensitivity of $\mathbf D$.
* $-d\sigma^2$: constant offset that fixes the bias.

---

## Unbiasedness property (the key guarantee)

For any fixed but unknown $\mathbf x$,

$$
\mathbb E_{\tilde{\mathbf x}\mid \mathbf x}\big[\mathrm{SURE}(\mathbf D;\mathbf x+\sigma\epsilon)\ \big|\ \mathbf x\big]!!!R(\mathbf D;\mathbf x).
$$

So **minimizing SURE (in expectation or empirically)** is equivalent to minimizing the true denoising MSE risk, while using only noisy data.

### Derivation sketch (how Stein’s identity enters)

Start from:

$$|\mathbf D(\tilde{\mathbf x})-\mathbf x|^2 = |\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x} + (\tilde{\mathbf x}-\mathbf x)|^2$$

Expand and use $\tilde{\mathbf x}-\mathbf x=\sigma\epsilon$. The cross-term contains $\mathbb E[\epsilon^\top g(\mathbf x+\sigma\epsilon)]$ with $g(\tilde{\mathbf x})=\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x}$. Stein’s lemma converts this to a divergence term:

$$\mathbb E[\epsilon^\top g(\mathbf x+\sigma\epsilon)] = \sigma,\mathbb E[\nabla_{\tilde{\mathbf x}}\cdot g(\tilde{\mathbf x})]$$

Since $\nabla\cdot(\tilde{\mathbf x})=d$, you get exactly the SURE formula.

---

# Link to Tweedie’s formula and Bayes optimality

## Noisy marginal

Let the noisy marginal be the convolution:

$$p_\sigma(\tilde{\mathbf x}) := (p_{\text{data}} * \mathcal N(0,\sigma^2\mathbf I))(\tilde{\mathbf x})$$

## SURE minimization ⇒ Bayes optimal denoiser

SURE is unbiased *w.r.t. noise* conditional on $\mathbf x$:

$$
\mathbb E_{\tilde{\mathbf x}\mid \mathbf x}[\mathrm{SURE}(\mathbf D;\tilde{\mathbf x})] = \mathbb E_{\tilde{\mathbf x}\mid \mathbf x}\big[|\mathbf D(\tilde{\mathbf x})-\mathbf x|^2\big].
$$

Averaging also over $\mathbf x\sim p_{\text{data}}$ gives the **Bayes risk**:

$$
\mathbb E_{\mathbf x,\tilde{\mathbf x}}\big[|\mathbf D(\tilde{\mathbf x})-\mathbf x|^2\big] = \mathbb E_{\tilde{\mathbf x}}\Big[\mathbb E_{\mathbf x\mid \tilde{\mathbf x}} |\mathbf D(\tilde{\mathbf x})-\mathbf x|^2\Big].
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

$$\mathbf D(\tilde{\mathbf x}) = \tilde{\mathbf x}+\sigma^2,\mathbf s_\phi(\tilde{\mathbf x};\sigma)$$

where $\mathbf s_\phi(\cdot;\sigma)\approx \nabla_{\tilde{\mathbf x}}\log p_\sigma(\cdot)$.

## Plugging into SURE yields Hyvärinen’s objective (up to constants)

Substitute into SURE and simplify:

$$
\frac{1}{2\sigma^4}\mathrm{SURE}(\mathbf D;\tilde{\mathbf x}) = \mathrm{Tr}\big(\nabla_{\tilde{\mathbf x}}\mathbf s_\phi(\tilde{\mathbf x};\sigma)\big)
+
\frac12|\mathbf s_\phi(\tilde{\mathbf x};\sigma)|*2^2
+
\text{const}(\sigma).
$$

Taking expectation over $\tilde{\mathbf x}\sim p*\sigma$, minimizing SURE is equivalent (up to an additive constant) to minimizing **Hyvärinen’s alternative score matching objective** at noise level $\sigma$.
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
\mathcal D_{\mathcal L}(p|q) := \int p(\mathbf x)\left|\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}-\frac{\mathcal L q(\mathbf x)}{q(\mathbf x)}\right|*2^2,d\mathbf x.
$$

If $\mathcal L$ is **complete** (informally: $\frac{\mathcal L p_1}{p_1}=\frac{\mathcal L p_2}{p_2}$ a.e. implies $p_1=p_2$ a.e.), then $\mathcal D_{\mathcal L}(p|q)=0$ identifies $q=p$.
For $\mathcal L=\nabla$, this recovers the classical Fisher divergence.

---

## Score parameterization (avoid explicit normalized $q$)

Instead of modeling $q$, directly learn a vector field $\mathbf s_\phi(\mathbf x)$ to approximate $\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}$:

$$
\mathcal D_{\mathcal L}(p|\mathbf s_\phi) := \mathbb E_{\mathbf x\sim p}\left[\left|\mathbf s_\phi(\mathbf x)-\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}\right|_2^2\right].
$$

The target is unknown, but integration by parts makes the loss computable.

### Adjoint operator and integration by parts trick

Define the adjoint $\mathcal L^\dagger$ by:

$$\int (\mathcal L f)^\top g = \int f,(\mathcal L^\dagger g) \quad \text{for all test functions } f,g$$

(assuming boundary terms vanish).

Expanding the square and applying the adjoint identity yields the tractable objective:

$$
\mathcal L_{\text{GSM}}(\phi) = \mathbb E_{\mathbf x\sim p}\Big[\frac12|\mathbf s_\phi(\mathbf x)|*2^2-(\mathcal L^\dagger \mathbf s*\phi)(\mathbf x)\Big]
+\text{const},
$$

where “const” does not depend on $\phi$.

### Check: recovering Hyvärinen’s score matching

For $\mathcal L=\nabla$, we have $\mathcal L^\dagger=-\nabla\cdot$ (negative divergence), so:

$$\mathbb E_p\Big[\tfrac12|\mathbf s_\phi|^2-(\mathcal L^\dagger \mathbf s_\phi)\Big] = \mathbb E_p\Big[\tfrac12|\mathbf s_\phi|^2+\nabla\cdot\mathbf s_\phi\Big]$$

which is Hyvärinen’s classical objective.

---

# Examples of operators $\mathcal L$

## 1) Classical score matching

Take $\mathcal L=\nabla_{\mathbf x}$. Then

$$\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}=\nabla_{\mathbf x}\log p(\mathbf x)$$

## 2) Denoising score matching (Gaussian corruption)

For additive Gaussian noise at level $\sigma$, define an operator on scalar $f$:

$$(\mathcal L f)(\tilde{\mathbf x})=\tilde{\mathbf x},f(\tilde{\mathbf x})+\sigma^2\nabla_{\tilde{\mathbf x}} f(\tilde{\mathbf x})$$

Then

$$\frac{\mathcal L p_\sigma(\tilde{\mathbf x})}{p_\sigma(\tilde{\mathbf x})}!!!\tilde{\mathbf x}+\sigma^2\nabla_{\tilde{\mathbf x}}\log p_\sigma(\tilde{\mathbf x})\mathbb E[\mathbf x_0\mid \tilde{\mathbf x}]$$

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

If you want, I can also turn these notes into a one-page “cheat sheet” of only the core equations and what each one is used for (training vs theory vs interpretation).



## Study Notes — Multi-Noise Denoising Score Matching (NCSN) + Annealed Langevin Dynamics (Sections 3.4–3.6)

### 0) Big picture

* **Goal (score-based generative modeling):** learn the **score**
  
  $$\nabla_x \log p(x)$$
  
  (gradient of log-density), which lets you **generate samples** by running dynamics that follow this gradient plus noise (e.g., Langevin).
* **Problem:** learning / sampling with a **single** noise level is unreliable and slow.
* **Fix (NCSN, Song & Ermon 2019):** train **one network conditioned on noise level** to estimate scores for **many noise scales**, then sample by **annealing** from high noise → low noise.

---

## 3.4 Multi-Noise Levels of Denoising Score Matching (NCSN)

### 3.4.1 Motivation: why one noise level is not enough

Adding Gaussian noise “smooths” the data distribution, but:

* **Low noise (small variance):**

  * Distribution is sharp/multi-modal; **Langevin struggles to move between modes**.
  * In low-density regions, the score can be inaccurate and gradients can vanish → **poor exploration**.
* **High noise (large variance):**

  * Sampling/mixing is easier, but the model captures only **coarse structure** → samples look **blurry**, lose fine detail.
* **High-dimensional issues:** Langevin can be **slow**, sensitive to **initialization**, can get stuck near **plateaus/saddles**.

**Figure intuition (SM inaccuracy):**

* In **low-density** areas, score estimates can be unreliable due to limited sample coverage; **high-density** areas are estimated better.

**Core idea:** use **multiple noise levels**:

* High noise: explore globally / cross modes.
* Low noise: refine details.

---

## 3.4.2 Training

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

$$\mathcal L_{\text{NCSN}}(\phi) := \sum_{i=1}^{L}\lambda(\sigma_i),\mathcal L_{\text{DSM}}(\phi;\sigma_i)$$

where

$$
\mathcal L_{\text{DSM}}(\phi;\sigma) = \frac12,\mathbb E_{x\sim p_{\text{data}},,\tilde x \sim p_\sigma(\tilde x\mid x)}
\left[
\left|
s_\phi(\tilde x,\sigma)-\frac{x-\tilde x}{\sigma^2}
\right|_2^2
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

$$\nabla_{x_\sigma}\log p_\sigma(x_\sigma) = -\frac{1}{\sigma},\mathbb E[\epsilon \mid x_\sigma]$$

So:

* NCSN’s target (score) is proportional to the **posterior mean noise**.
* If a DDPM-style model predicts $\epsilon^*(x_\sigma,\sigma)=\mathbb E[\epsilon\mid x_\sigma]$, then:
  
$$
s^*(x_\sigma,\sigma)= -\frac{1}{\sigma}\epsilon^*(x_\sigma,\sigma),
\quad
\epsilon^*(x_\sigma,\sigma)= -\sigma,s^*(x_\sigma,\sigma)
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

$$\tilde x_{n+1} = \tilde x_n + \eta_\ell, s_\phi(\tilde x_n,\sigma_\ell) + \sqrt{2\eta_\ell}\epsilon_n, \quad \epsilon_n\sim\mathcal N(0,I)$$

### Algorithm (as given)

* Initialize $x^{\sigma_L}\sim\mathcal N(0,I)$ (often equivalent to choosing a large-noise prior).
* For $\ell = L, L-1,\dots,2$:
  * run $N_\ell$ Langevin steps using $s_\phi(\cdot,\sigma_\ell)$
  * set $x^{\sigma_{\ell-1}}\leftarrow$ final sample (init for next level)
* Output $x^{\sigma_1}$.

**Step size scaling (typical):**

$$\eta_\ell = \delta\cdot \frac{\sigma_\ell^2}{\sigma_1^2},\quad \delta>0$$

Intuition: bigger noise $\implies$ you can take bigger steps.

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
  
  $$x_{i+1} = \sqrt{1-\beta_i},x_i + \sqrt{\beta_i}\epsilon$$
  

### Loss / training target

* **NCSN:** score loss equivalent to
  
  $$\mathbb E\big[|s_\phi(x_i,\sigma_i) + \epsilon/\sigma_i|^2\big]$$
  
  (score matches scaled negative noise).
* **DDPM:** noise prediction loss
  
  $$\mathbb E\big[|\epsilon_\phi(x_i,i)-\epsilon|^2\big]$$

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

  * (x_\sigma = x + \sigma\epsilon) and (p_\sigma(x_\sigma\mid x)=\mathcal N(x, \sigma^2I))?
  * (p_\sigma(x)=\int p_\sigma(x\mid x_0)p_{\text{data}}(x_0)dx_0)?
  * DSM loss target ((x-\tilde x)/\sigma^2)?
  * Multi-noise objective (\sum_i \lambda(\sigma_i)\mathcal L_{\text{DSM}}(\sigma_i))?
  * Langevin update (\tilde x_{n+1}=\tilde x_n+\eta s_\phi(\tilde x_n,\sigma)+\sqrt{2\eta}\epsilon)?
* Can you explain (in words) why:

  * low noise ⇒ hard mode traversal; high noise ⇒ blurry?
  * annealing helps?
  * sampling cost is (\mathcal O(LK))?
* Can you derive the DDPM/NCSN link:
  [
  \nabla\log p_\sigma(x_\sigma)=-(1/\sigma)\mathbb E[\epsilon\mid x_\sigma] ;\Rightarrow; \epsilon^*=-\sigma s^* ; ?
  ]

If you want, I can also turn these notes into a clean 2–3 page “one-pager” with only the essential equations + intuition (more compact for revision).








































