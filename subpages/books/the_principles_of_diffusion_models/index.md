---
title: The Principles of Diffusion Models
layout: default
noindex: true
---

# The Principles of Diffusion Models

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/unified_diffusion_models.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption><strong>Unifying and Principled Perspectives on Diffusion Models.</strong> This diagram visually connects classical generative modeling approaches—Variational Autoencoders, Energy-Based Models, and Normalizing Flows—with their corresponding diffusion model formulations. Each vertical path illustrates a conceptual lineage, culminating in the continuous-time framework. The three views (Variational, Score-Based, and Flow-Based) offer distinct yet mathematically equivalent interpretations.</figcaption>
</figure>

## Deep Generative Modeling

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Deep Generative Modeling)</span></p>

**Goal of DGM:** 
* DNN to parameterize a model distribution $p_\phi(x)$, 
* $\phi$ represents the network’s trainable parameters.
* find 

$$p_{\phi^*}(x) \approx p_{\text{data}}(x)$$

**Capability of DGM**
1. Sampling from $p_\phi(x)$ using sampling methods (MC or other)
2. Compute the probability (or likelihood) of any given data sample $x'$: $p_\phi(x')$.
3. While the sampling from $p_\phi(x)$ may be possible, the density $p_\phi(x)$ may or may not be directly computable, depending on the model class

**Training of DGM**
* learn parameters $ϕ$ of a model family $\lbrace p_\phi\rbrace$ 
* by minimizing a discrepancy $\mathcal{D}(p_{\text{data}},p_\phi)$:

$$\phi^*\in\arg\min_\phi \mathcal{D}(p_{\text{data}},p_\phi)$$

**Note:** $p_\phi(x)$ is commonly referred as a **generative model**.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/discrepancy_in_dgm.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption><strong>Illustration of the target in DGM.</strong> Training a DGM is essentially minimizing the discrepancy between the model distribution $p_\phi$ and the unknown data distribution $p_{\text{data}}$. Since pdata is not directly accessible, this discrepancy must be estimated efficiently using a finite set of independent and identically distributed (i.i.d.) samples, $\boldsymbol{x_i}$, drawn from it.</figcaption>
</figure>

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


### Mathematical setup

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Mathematical setup)</span></p>

**Data assumption**

* You observe a finite dataset of **i.i.d.** samples
  
  $$x^{(i)} \sim p_{\text{data}}(x), \quad i=1,\dots,N$$
  
  where $p_{\text{data}}$ is an **unknown**, complex distribution.

**Goal of a Deep Generative Model (DGM)**

* Learn a **tractable** model distribution $p_\phi(x)$ (parameterized by a neural network with parameters $\phi$) such that
  
  $$p_{\phi^*}(x) \approx p_{\text{data}}(x)$$
  
* Intuition: since $p_{\text{data}}$ is unknown (and you only have samples), you fit $p_\phi$ so it can act as a proxy for the data distribution.

**What "having a generative model" gives you**

* **Sampling:** generate arbitrarily many new samples (e.g., via Monte Carlo methods) from $p_\phi$.
* **Likelihood / density evaluation:** compute $p_\phi(x')$ (or $\log p_\phi(x')$) for a given point $x'$ *if the model family supports tractable evaluation*.

</div>

### Training objective via a discrepancy / divergence

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Training objective via a discrepancy / divergence)</span></p>

**General training principle**

* Choose a discrepancy measure $D(p_{\text{data}}, p_\phi)$ between distributions and solve
  
  $$\phi^* \in \arg\min_\phi D(p_{\text{data}}, p_\phi) \qquad\text{(1.1.1)}$$
  
* Since $p_{\text{data}}$ is not directly accessible, $D$ must be something you can **estimate from samples**.

</div>


### Forward KL divergence and Maximum Likelihood Estimation (MLE)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forward KL divergence and Maximum Likelihood Estimation (MLE))</span></p>

**Forward KL definition**

$$D_{\mathrm{KL}}(p_{\text{data}}\parallel p_\phi) :=\int p_{\text{data}}(x)\log\frac{p_{\text{data}}(x)}{p_\phi(x)}dx$$

$$= \mathbb{E}_{x\sim p_{\text{data}}}\left[\log p_{\text{data}}(x)-\log p_\phi(x)\right]$$

* **Asymmetric:**
  
  $$D_{\mathrm{KL}}(p_{\text{data}}\parallel p_\phi)\neq D_{\mathrm{KL}}(p_\phi\parallel p_{\text{data}})$$

</div>

#### Mode covering effect (important intuition)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(Mode covering effect (important intuition))</span></p>

* Minimizing **forward KL** encourages **mode covering**:

  * If there is a set $A$ with positive data probability $p_{\text{data}}(A)>0$ but the model assigns zero density there ($p_\phi(x)=0$ for $x\in A$), then the integrand contains $\log(p_{\text{data}}(x)/0)=+\infty$ on $A$, hence the KL becomes infinite.
  * **Consequence:** forward KL strongly pressures the model to put probability mass wherever the data has support.

</div>

#### KL decomposition $\implies$ MLE equivalence

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(KL decomposition $\implies$ MLE equivalence)</span></p>

Rewrite forward KL:

$$
\begin{aligned}
D_{\mathrm{KL}}(p_{\text{data}}\parallel p_\phi)
&= \mathbb{E}_{x\sim p_{\text{data}}}\left[\log\frac{p_{\text{data}}(x)}{p_\phi(x)}\right] \\
&= -\mathbb{E}_{x\sim p_{\text{data}}}\left[\log p_\phi(x)\right] + \underbrace{\left(-\mathbb{E}_{x\sim p_{\text{data}}}[\log p_{\text{data}}(x)]\right)}_{\mathcal{H}(p_{\text{data}})}
\end{aligned}
$$

- $\mathcal{H}(p_{\text{data}})$ is the **entropy** of the data distribution and does **not** depend on $\phi$.
- Therefore:

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Minimizing KL $\iff$ MLE)</span></p>

$$
\min_\phi D_{\mathrm{KL}}(p_{\text{data}}\parallel p_\phi)
\quad \Longleftrightarrow \quad
\max_\phi \mathbb{E}_{x\sim p_{\text{data}}}[\log p_\phi(x)] \qquad\text{(1.1.2)}
$$

</div>

#### Empirical MLE objective (what you actually optimize)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Empirical MLE objective (what you actually optimize))</span></p>

Replace the expectation with the sample average (Monte Carlo estimate):

$$\hat{\mathcal{L}}_{\mathrm{MLE}}(\phi) := -\frac{1}{N}\sum_{i=1}^N \log p_\phi(x^{(i)})$$

optimized with stochastic gradients / minibatches.
Key point: **you never need to evaluate** $p_{\text{data}}(x)$.

</div>

### Fisher divergence (score discrepancy) and score matching

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fisher Divergence)</span></p>

For distributions $p$ and $q$:

$$D_F(p\parallel q) := \mathbb{E}_{x\sim p}\left[\left\|\nabla_x \log p(x) - \nabla_x \log q(x)\right\|_2^2\right] \qquad \text{(1.1.3)}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fisher divergence (score discrepancy) and score matching)</span></p>

**Core concept: the score**

* The **score function** of a density $p$ is
  
  $$s_p(x) := \nabla_x \log p(x)$$
  
* Fisher divergence measures how close the **vector fields** $s_p(x)$ and $s_q(x)$ are.

**Key property**

* It’s **invariant to normalization constants**, because gradients of log-densities ignore additive constants:
  * If $q(x)\propto \tilde q(x)$, then $\nabla_x \log q(x)=\nabla_x \log \tilde q(x)$.
* This makes it a natural basis for **score matching** and connects directly to **score-based / diffusion modeling**, where you train a model to match the data score field.

</div>

### Beyond KL: other divergences

Different divergences encode different notions of "closeness" and can change learning behavior.

#### $f$-divergences (Csiszár family)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($f$-divergence)</span></p>

A broad family:

$$D_f(p\parallel q) = \int q(x) f\left(\frac{p(x)}{q(x)}\right)dx, \qquad f(1)=0$$

where $f:\mathbb{R}_+\to\mathbb{R}$ is **convex**. $\text{(1.1.4)}$

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($f$-divergence instances)</span></p>

* **Forward KL:** $f(u)=u\log u \Rightarrow D_f = D_{\mathrm{KL}}(p\parallel q)$
* **Jensen–Shannon (JS):** $f(u)=\tfrac12\Big[u\log u-(u+1)\log\frac{u+1}{2}\Big] \Rightarrow D_f=D_{\mathrm{JS}}(p\parallel q)$
* **Total variation (TV):** $f(u)=\tfrac12\lvert u-1\rvert \Rightarrow D_f=D_{\mathrm{TV}}(p,q)$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Explicit forms for JS and TV)</span></p>

* **JS divergence**
  
  $$D_{\mathrm{JS}}(p\parallel q) =\tfrac12 D_{\mathrm{KL}}\left(p \parallel \tfrac12(p+q)\right) +\tfrac12 D_{\mathrm{KL}}\left(q \parallel \tfrac12(p+q)\right)$$

  Intuition: **smooth + symmetric**, balances both distributions, avoids some unbounded KL behavior; later useful for interpreting GANs. It helps interpret the Generative Adversarial Network (GAN) framework.

* **Total variation distance**
  
  $$D_{\mathrm{TV}}(p,q) =\tfrac12\int_{\mathbb{R}^D} \lvert p-q\rvert dx = \sup_{A\subset \mathbb{R}^D} \lvert p(A)-q(A)\rvert$$

  Intuition: captures the **largest possible** difference in probability the two distributions can assign to any event $A$.

</div>

#### Optimal transport viewpoint: Wasserstein distances

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Wasserstein distances)</span></p>

**Wasserstein** distances measure the minimal cost of moving probability mass from one distribution to another

Unlike $f$-divergences (which compare **density ratios**), **Wasserstein** distances depend on the **geometry of the sample space** and can remain meaningful even if the supports of $p$ and $q$ **do not overlap**.

</div>

### Challenges in modeling distributions (Section 1.1.2)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Challenges in modeling distributions)</span></p>

To model a complex data distribution, we can parameterize the probability density function $p_{\text{data}}$ using a neural network with parameters $\phi$, creating a model we denote as $p_\phi$. To model a density $p_\phi(x)$ with a neural network, $p_\phi$ must satisfy:

1. **Non-negativity:** $p_\phi(x)\ge 0$ for all $x$.
2. **Normalization:** $\int p_\phi(x)dx = 1$.

</div>

#### Practical construction via an unnormalized "energy" output

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical construction via an unnormalized "energy" output)</span></p>

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

</div>

#### Central difficulty

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Central difficulty)</span></p>

* In **high dimensions**, computing $Z(\phi)$ (and often its gradients) is typically **intractable**.
* This intractability is a major motivation for many DGM families: they’re designed to **avoid**, **approximate**, or **circumvent** the cost of evaluating the partition function.

</div>

### Prominent Deep Generative Models

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(DGMs)</span></p>

A major goal in generative modeling is to learn **expressive probabilistic models** that capture the complex structure of high-dimensional data.

Different model families make different trade-offs between:

* **tractability**
* **expressiveness**
* **training efficiency**

This section introduces several major families of deep generative models:

* **Energy-Based Models (EBMs)**
* **Autoregressive Models (ARs)**
* **Variational Autoencoders (VAEs)**
* **Normalizing Flows (NFs)**
* **Generative Adversarial Networks (GANs)**

</div>

#### Energy-Based Models (EBMs)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Energy-Based Models (EBMs))</span></p>

**EBMs** define a probability distribution using an **energy function** $E_\phi(x)$.

* Lower energy means the data point is **more probable**
* Probability is defined as:

$$p_\phi(x) = \frac{1}{Z(\phi)} \exp(-E_\phi(x))$$

where the partition function is

$$Z(\phi) = \int \exp(-E_\phi(x))dx$$

**Key points:**
* EBMs model probability indirectly through energy
* Training usually aims to **maximize log-likelihood**
* The main difficulty is the **partition function** $Z(\phi)$, which is often intractable

**Limitation:**
* Computing or approximating the partition function is computationally hard

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Importance of EBMs)</span></p>

* EBMs are closely related to **score-based / diffusion ideas**
* Diffusion models avoid explicitly computing the partition function by working with the **gradient of the log density**

</div>

#### Autoregressive Models (ARs)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Autoregressive Models (ARs))</span></p>

**AR models** factorize the joint data distribution into a product of conditional probabilities using the **chain rule of probability**:

$$p_{\text{data}}(x) = \prod_{i=1}^{D} p_\phi(x_i \mid x_{<i})$$

where:
* $x = (x_1, \dots, x_D)$
* $x_{<i} = (x_1, \dots, x_{i-1})$

**Key points:**
* Each conditional distribution is parameterized by a neural network, often a **Transformer**
* Since each term is normalized, **global normalization is easy**
* AR models support **exact likelihood**

**Training:**
* Trained by **maximizing exact likelihood**
* Equivalent to minimizing **negative log-likelihood**

**Strengths:**
* Strong density estimation
* Exact tractable likelihoods
* Foundational class of likelihood-based models

**Limitations:**
* Sampling is **slow** because generation is sequential
* Fixed ordering can restrict flexibility

</div>

#### Variational Autoencoders (VAEs)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Variational Autoencoders (VAEs))</span></p>

**VAEs** extend autoencoders by introducing a latent variable $z$ that captures hidden structure in the data.

They learn:

* an **encoder** $q_\theta(z \mid x)$, which approximates the posterior over latent variables
* a **decoder** $p_\phi(x \mid z)$, which reconstructs data from latent variables

**Training objective:**

Instead of directly maximizing the true log-likelihood, VAEs maximize the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}_{\text{ELBO}}(\theta, \phi; x)
= \mathbb{E}_{q_\theta(z \mid x)}[\log p_\phi(x \mid z)] = D_{\text{KL}}(q_\theta(z \mid x)\parallel p_{\text{prior}}(z))
$$

**Interpretation of ELBO:**
* First term: encourages **accurate reconstruction**
* Second term: regularizes latent variables so they stay close to a simple prior, usually Gaussian

**Strengths:**
* Principled combination of neural networks and latent-variable modeling
* One of the most widely used likelihood-based generative approaches
* Important foundation for diffusion models

**Limitations:**
* Can produce **blurry / less sharp samples**
* Can suffer from training pathologies such as **posterior collapse** (encoder ignores latent variables)

</div>

#### Normalizing Flows (NFs)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normalizing Flows (NFs))</span></p>

**Normalizing flows** learn an **invertible mapping** between:

* a simple latent distribution $z$
* a complex data distribution $x$

This mapping is a bijection $f_\phi$.

**Variants mentioned:**
* **Normalizing Flows (NFs)**
* **Neural Ordinary Differential Equations (NODEs)**

**Likelihood computation:**

They use the **change-of-variables formula**:

$$
\log p_\phi(x) = \log p(z)
+
\log \left|\det \frac{\partial f_\phi^{-1}(x)}{\partial x}\right|
$$

**Key points:**
* Because the transformation is invertible, densities are modeled **exactly**
* Likelihood computation is **tractable**
* Training is done with **maximum likelihood estimation (MLE)**

**Strengths:**
* Exact normalized densities
* Exact and tractable likelihoods
* Elegant theoretical formulation

**Limitations:**
* NFs often require restrictive architectures to ensure bijectivity
* NODEs can be computationally expensive because they require solving ODEs
* Both can struggle to scale to high-dimensional data

</div>

#### Generative Adversarial Networks (GANs)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generative Adversarial Networks (GANs))</span></p>

GANs consist of two networks:

* a **generator** $G_\phi$
* a **discriminator** $D_\zeta$

The generator maps noise $z \sim p_{\text{prior}}$ to samples $G_\phi(z)$, while the discriminator tries to distinguish real from generated samples.

**Objective:**

GANs use an adversarial min-max game:

$$
\min_{G_\phi} \max_{D_\zeta}
\mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D_\zeta(x)]
+
\mathbb{E}_{z \sim p_{\text{prior}}(z)}[\log(1 - D_\zeta(G_\phi(z)))]
$$

**Key points:**
* GANs do **not** define an explicit density function
* They avoid direct likelihood estimation
* They focus on generating realistic samples

**Divergence perspective:**

For a fixed generator, the optimal discriminator is:

$$\frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_{G_\phi}(x)}$$

Under this discriminator, GAN training reduces to minimizing:

$$2 D_{\text{JS}}(p_{\text{data}} \parallel p_{G_\phi}) - \log 4$$

where $D_{\text{JS}}$ is the **Jensen-Shannon divergence**.

Extension:

* **$f$-GANs** generalize this idea to a broader family of **$f$-divergences**

**Strengths:**
* Can generate very high-quality samples
* Very influential in generative modeling

**Limitations:**
* Training is often **unstable**
* Requires careful architecture and optimization tricks
* Has later become more of an auxiliary component in modern systems, especially alongside diffusion models

</div>

#### Quick comparison

| Model | Core mechanism                  | Likelihood                     | Main strength                               | Main limitation                             |
| ----- | ------------------------------- | ------------------------------ | ------------------------------------------- | ------------------------------------------- |
| EBM   | Energy function                 | Defined via partition function | Flexible probabilistic modeling             | Partition function is intractable           |
| AR    | Product of conditionals         | Exact                          | Exact likelihood, strong density estimation | Slow sequential sampling                    |
| VAE   | Latent-variable encoder-decoder | Approximate via ELBO           | Principled latent representation learning   | Blurry samples, posterior collapse          |
| NF    | Invertible transformation       | Exact                          | Exact tractable density                     | Architectural and computational constraints |
| GAN   | Adversarial sample generation   | Implicit / no explicit density | High-quality samples                        | Unstable training                           |

### Taxonomy of Modelings

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Explicit vs Implicit Generative Models)</span></p>

Deep generative models can be distinguished by **how they parameterize the data distribution**:

* **Explicit models** specify the distribution $p_\phi(x)$ **directly**
* **Implicit models** define the distribution **indirectly**, usually through a sampling process

This distinction is about **how the model represents the distribution**, not about the training objective.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Explicit Generative Models)</span></p>

**Explicit models** directly parameterize a probability distribution $p_\phi(x)$ using a:

* tractable density/mass function, or
* approximately tractable one

They define $p_\phi(x)$ either:

* exactly, or
* via a tractable bound

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Explicit Generative Models)</span></p>

* **ARs** (Autoregressive models)
* **NFs** (Normalizing Flows)
* **VAEs** (Variational Autoencoders)
* **DMs** (Diffusion Models)

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Implicit Generative Models)</span></p>

**Implicit models** specify a distribution only through a **sampling procedure**, typically:

$$x = G_\phi(z), \quad z \sim p_{\text{prior}}$$

So:

* samples can be generated,
* but $p_\phi(x)$ is **not available in closed form**
* and may sometimes **not be defined at all**

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Implicit Generative Models)</span></p>

* **GANs**
* **Markov-process samplers without tractable density**
  * Some MCMC-based generative procedures, when used only as sample generators

</div>

| Category  | Explicit (Exact Likelihood) | Explicit (Approx. Likelihood) | Implicit |
|-----------|------------------------------|--------------------------------|----------|
| Likelihood | Tractable | Bound/Approx. | Not Directly Modeled / Intractable |
| Objective | MLE | ELBO | Adversarial |
| Examples | NFs, ARs | VAEs, DMs | GANs |

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection Diffusion Models)</span></p>

* **Explicit models** try to model the probability distribution directly
* **Implicit models** only define how to sample from the distribution
* **Diffusion models** are explicit (via tractable bounds) and are connected to VAEs, EBMs, and NFs

Diffusion models relate to several classical families of deep generative models:

* **VAEs** through **variational training objectives**
* **EBMs** through **score-matching** methods that learn gradients of the log-density
* **NFs** through **continuous-time transformations**

</div>

### Graphs of Prominent Deep Generative Models

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/prominent_gen_models.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption><strong>Computation graphs of prominent deep generative models.</strong> Top to bottom: <strong>EBM</strong> maps an input $\mathbb{x}$ to a scalar energy; <strong>AR</strong> generates a sequence $\lbrace x\ell\rbrace$ left to right with causal dependencies; <strong>VAE</strong> encodes $\mathbb{x}$ to a latent $\mathbb{z}$ and decodes to a reconstruction $\mathbb{x}$; <strong>NF</strong> applies an invertible map $f_\phi$ between $\mathbb{x}$ and $\mathbb{z}$ and uses $f{−1}_\phi$ to produce $\mathbb{x'}$; <strong>GAN</strong> transforms noise $\mathbb{z}$ to a sample $\mathbb{x'}$ that is judged against real $\mathbb{x}$ by a discriminator $D_ζ$; <strong>DM</strong> iteratively refines a noisy sample through a multi-step denoising chain $\lbrace x_\ell\rbrace$. Boxes denote variables, trapezoids are learnable networks, ovals are scalars; arrows indicate computation flow.</figcaption>
</figure>

## Variational Perspective: From VAEs to DDPMs

### Big picture

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Big picture)</span></p>

* **Core theme:** VAEs, hierarchical VAEs, and diffusion models can all be viewed as optimizing a **tractable variational lower bound** (a likelihood surrogate) on an otherwise **intractable log-likelihood**.
* **VAE template (learned encoder + learned decoder):**

  * Encoder maps observations → latent distribution.
  * Decoder maps latents → observation distribution, "closing the loop."
* **DDPM template (fixed encoder + learned decoder):**

  * The "encoder" is a **fixed forward noising process** mapping data → noise.
  * Training learns a **reverse denoising decoder** that inverts this path step-by-step.

</div>

### Variational Autoencoder (VAE)

### Why not a plain autoencoder?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why not a plain autoencoder?)</span></p>

* A standard autoencoder has:

  * deterministic **encoder**: compresses $x$ into a low-dim code
  * deterministic **decoder**: reconstructs $x$
* It can reconstruct well, but the **latent space is unstructured**:

  * sampling random latent codes usually yields meaningless outputs
  * not a reliable **generative** model

</div>

### VAE idea (Kingma & Welling, 2013)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(VAE idea (Kingma & Welling, 2013))</span></p>

* Make the latent space **probabilistic + regularized**, so that:

  * sampling $z$ from a simple prior produces meaningful outputs
  * the model becomes a true generative model

</div>

---

## 2.1.1 Probabilistic encoder and decoder

### Variables

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Variables)</span></p>

* **Observed variable:** $x$ (e.g., an image)
* **Latent variable:** $z$ (captures hidden factors: shape, color, style, $\dots$)

</div>

### Prior over latents

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Prior over latents)</span></p>

Typically a simple prior, e.g.

$$z \sim p(z) = \mathcal N(0, I)$$

</div>

### Decoder / generator

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Decoder / generator)</span></p>

Define a conditional likelihood ("decode latents into data"):

$$p_\phi(x \mid z)$$

In practice this is often kept **simple**, e.g. a **factorized Gaussian**, to encourage learning useful latent features rather than memorizing data.

</div>

### Sampling procedure

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sampling procedure)</span></p>

1. Sample $z \sim p(z)$
2. Sample $x \sim p_\phi(x \mid z)$

</div>

---

## Latent-variable marginal likelihood (why it’s hard)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Latent-variable marginal likelihood (why it’s hard))</span></p>

A VAE defines the data likelihood via marginalization:

$$p_\phi(x) = \int p_\phi(x \mid z), p(z) dz$$

* Ideally, we would learn $\phi$ by maximizing $\log p_\phi(x)$ (MLE).
* But for expressive nonlinear decoders, the integral over $z$ is **intractable**, so **direct MLE is computationally infeasible**.

</div>

---

## Construction of the encoder (inference network)

### True posterior (intractable)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(True posterior (intractable))</span></p>

Given $x$, the "correct" latent posterior is:

$$p_\phi(z \mid x) = \frac{p_\phi(x \mid z), p(z)}{p_\phi(x)}$$

* The denominator $p_\phi(x)$ is exactly the intractable marginal likelihood, so **exact inference is prohibitive**.

</div>

### Variational approximation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variational approximation)</span></p>

Introduce a learnable approximate posterior (encoder):

$$q_\theta(z \mid x) \approx p_\phi(z \mid x)$$

* This gives a feasible, trainable pathway from $x \to z$.

</div>

---

## 2.1.2 Training via the Evidence Lower Bound (ELBO)

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(ELBO bound (2.1.1))</span></p>

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

</div>


### Proof sketch (Jensen’s inequality)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof sketch (Jensen’s inequality))</span></p>

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

</div>

---

## Interpreting the two ELBO terms

### 1) Reconstruction term

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reconstruction term)</span></p>

$$\mathbb E_{z\sim q_\theta(z\mid x)}[\log p_\phi(x\mid z)]$$

* Encourages accurate recovery of $x$ from its latent code $z$.
* Under Gaussian encoder/decoder assumptions, this reduces to the familiar **reconstruction loss** of autoencoders.

</div>

### 2) Latent KL regularization

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Latent KL regularization)</span></p>

$$D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel p(z))$$

* Encourages the encoder distribution to stay close to a simple prior $p(z)$ (e.g. $\mathcal N(0,I)$).
* Shapes the latent space to be smooth/continuous so samples from the prior decode meaningfully.

**Key trade-off:** good reconstructions vs. a well-structured latent space that supports sampling.

</div>

---

## Information-theoretic view: ELBO as a divergence bound

### MLE view

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(MLE view)</span></p>

Maximum likelihood training corresponds to minimizing:

$$D_{\mathrm{KL}}(p_{\text{data}}(x)\parallel p_\phi(x))$$

which measures how well $p_\phi$ approximates the data distribution (but is generally intractable to optimize directly).

</div>

### Joint-distribution trick (variational framework)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Joint-distribution trick (variational framework))</span></p>

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

</div>

### Chain rule / decomposition of the joint KL

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Chain rule / decomposition of the joint KL)</span></p>

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

</div>

### ELBO gap equals posterior KL

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(ELBO gap equals posterior KL)</span></p>

For each $x$,

$$
\log p_\phi(x) - \mathcal L_{\text{ELBO}}(\theta,\phi;x)
=

D_{\mathrm{KL}}(q_\theta(z\mid x),|,p_\phi(z\mid x)).
$$

So **maximizing ELBO** is exactly **reducing the inference gap**, i.e. pushing the variational posterior toward the true posterior.

</div>

---

## Connection forward: hierarchical VAEs → DDPMs (conceptual bridge)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection forward: hierarchical VAEs → DDPMs (conceptual bridge))</span></p>

* **Hierarchical VAEs:** stack multiple latent layers to capture structure at multiple scales.
* **DDPMs as "many-layer" variational models:**

  * the forward noising process plays the role of a (fixed) encoder that gradually maps data to noise
  * the reverse denoising model is the learned decoder that inverts this mapping step-by-step
* The shared variational viewpoint: all optimize a **variational bound** on likelihood rather than the exact likelihood directly.

</div>

---

## Quick formula sheet (from these pages)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Quick formula sheet (from these pages))</span></p>

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

</div>

## 2.1.3 Gaussian VAE (standard "Gaussian–Gaussian" VAE)

### Setup and notation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Setup and notation)</span></p>

* Data: $x \in \mathbb{R}^D$
* Latent: $z \in \mathbb{R}^d$
* Prior: $p_{\text{prior}}(z)$ (often $\mathcal N(0,I)$)

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gaussian VAE encoder and decoder)</span></p>

**Encoder (approximate posterior):** The encoder is a diagonal-covariance Gaussian:

$$q_\theta(z\mid x) := \mathcal N \Big(z;\ \mu_\theta(x),\ \mathrm{diag}(\sigma_\theta^2(x))\Big)$$

where $\mu_\theta:\mathbb R^D\to\mathbb R^d$ and $\sigma_\theta:\mathbb R^D\to\mathbb R_+^d$ are deterministic neural-network outputs.

**Decoder (likelihood / generator):** The decoder is a Gaussian with **fixed** variance:

$$p_\phi(x\mid z) := \mathcal N\big(x;\ \mu_\phi(z),\ \sigma^2 I\big)$$

where $\mu_\phi:\mathbb R^d\to\mathbb R^D$ is a neural network and $\sigma>0$ is a (small) constant.

</div>

### ELBO specialization (\Rightarrow) MSE reconstruction

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(ELBO specialization (\Rightarrow) MSE reconstruction)</span></p>

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

**Interpretation:** training becomes "regularized reconstruction":
* a **reconstruction loss** (scaled MSE),
* plus a **KL regularizer** pushing $q_\theta(z\mid x)$ toward the prior.

**Why KL is "easy" here:** for Gaussian $q_\theta$ (and typical Gaussian prior), the KL has a closed form (commonly used in implementations).

</div>

---

## 2.1.4 Drawbacks of a standard VAE: blurry outputs

### Why Gaussian VAEs often look blurry (core mechanism)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Gaussian VAEs often look blurry (core mechanism))</span></p>

Consider:

* a **fixed** Gaussian encoder $q_{\text{enc}}(z\mid x)$,
* and a Gaussian decoder with fixed variance
  
  $$p_{\text{dec}}(x\mid z)=\mathcal N(x;\mu(z),\sigma^2I)$$

With an arbitrary encoder, optimizing the ELBO (up to an additive constant) reduces to minimizing an expected squared error:

$$\arg\min_{\mu}\ \mathbb E_{p_{\text{data}}(x),q_{\text{enc}}(z\mid x)}\Big[|x-\mu(z)|^2\Big]$$

This is a least-squares regression problem in $\mu(z)$. The optimal solution is the **conditional mean**:

$$\mu^*(z)=\mathbb E_{q_{\text{enc}}(x\mid z)}[x]$$

</div>

### What is $q_{\text{enc}}(x\mid z)$?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What is $q_{\text{enc}}(x\mid z)$?)</span></p>

It’s the "encoder-induced posterior on inputs given latents", obtained via Bayes’ rule:

$$q_{\text{enc}}(x\mid z)=\frac{q_{\text{enc}}(z\mid x),p_{\text{data}}(x)}{p_{\text{prior}}(z)}$$

An equivalent (often useful) form:

$$
\mu^*(z)
=\frac{\mathbb E_{p_{\text{data}}(x)}\big[q_{\text{enc}}(z\mid x),x\big]}
{\mathbb E_{p_{\text{data}}(x)}\big[q_{\text{enc}}(z\mid x)\big]}.
$$

</div>

### Where blur comes from (mode averaging)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Where blur comes from (mode averaging))</span></p>

If two distinct inputs $x\neq x'$ are mapped to **overlapping regions** in latent space (i.e., supports of $q_{\text{enc}}(\cdot\mid x)$ and $q_{\text{enc}}(\cdot\mid x')$ intersect), then for such a $z$,

$$\mu^*(z)=\mathbb E[x\mid z]$$

**averages across multiple (possibly unrelated) inputs**. Averaging "conflicting modes" produces **non-distinct, blurry** reconstructions/samples.

**Key takeaway:** with a Gaussian decoder + MSE-like training signal, the optimal prediction is a mean, and means of multimodal/ambiguous conditionals look blurry.

</div>

---

## 2.1.5 (Optional) From standard VAE to Hierarchical VAEs (HVAEs)

### Motivation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(HVAE motivation)</span></p>

Hierarchical VAEs introduce **multiple latent layers** to capture structure at different abstraction levels (coarse $\to$ fine). (Referenced: Vahdat & Kautz, 2020.)

</div>

### Generative model (top-down hierarchy)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Generative model (top-down hierarchy))</span></p>

Introduce $z_{1:L}=(z_1,\dots,z_L)$. A common top-down factorization:

$$p_\phi(x,z_{1:L})  =  p_\phi(x\mid z_1)\ \prod_{i=2}^{L} p_\phi(z_{i-1}\mid z_i)\ p(z_L)$$

The marginal data density:

$$p_{\text{HVAE}}(x)  :=  \int p_\phi(x,z_{1:L}),dz_{1:L}$$

**Sampling/generation is progressive:**

1. sample top latent $z_L\sim p(z_L)$
2. decode downward $z_{L-1}\sim p_\phi(z_{L-1}\mid z_L)$, $\dots$, $z_1\sim p_\phi(z_1\mid z_2)$
3. generate $x\sim p_\phi(x\mid z_1)$

</div>

### Inference model (bottom-up, mirrors hierarchy)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Inference model (bottom-up, mirrors hierarchy))</span></p>

A common structured encoder uses a bottom-up Markov factorization:

$$q_\theta(z_{1:L}\mid x)  =  q_\theta(z_1\mid x)\ \prod_{i=2}^{L} q_\theta(z_i\mid z_{i-1})$$

</div>

---

## HVAE ELBO (derivation + form)

### Jensen’s inequality derivation (standard ELBO trick)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Jensen’s inequality derivation (standard ELBO trick))</span></p>

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

</div>

### Interpretable decomposition (reconstruction + "adjacent" KLs)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretable decomposition (reconstruction + "adjacent" KLs))</span></p>

A key decomposition shown:

$$
\mathcal L_{\text{ELBO}}(x)
=

\mathbb E_q[\log p_\phi(x\mid z_1)]
-\mathbb E_q \Big[D_{\mathrm{KL}}(q_\theta(z_1\mid x),|,p_\phi(z_1\mid z_2))\Big]

$$

$$
-\sum_{i=2}^{L-1}\mathbb E_q \Big[D_{\mathrm{KL}}(q_\theta(z_i\mid z_{i-1}),|,p_\phi(z_i\mid z_{i+1}))\Big]
-\mathbb E_q \Big[D_{\mathrm{KL}}(q_\theta(z_L\mid z_{L-1}),|,p(z_L))\Big],
$$

where $\mathbb E_q$ denotes expectation under the encoder-induced joint over $(x,z_{1:L})$ (as written in the text).

**Meaning:** each inference conditional is regularized toward its corresponding **top-down** conditional prior:

+ $q(z_1\mid x)$ vs $p(z_1\mid z_2)$,
+ $q(z_i\mid z_{i-1})$ vs $p(z_i\mid z_{i+1})$,
+ top level $q(z_L\mid z_{L-1})$ vs $p(z_L)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Observation 2.1.1)</span></p>

Stacking layers lets the model generate **progressively** (coarse $\to$ fine), which helps capture complex high-dimensional structure.

</div>

---

## Why "just make a flat VAE deeper" is not enough

### Limitation 1: the variational family is still too simple

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Limitation 1: the variational family is still too simple)</span></p>

In a standard flat VAE,

$$q_\theta(z\mid x)=\mathcal N\big(z;\mu_\theta(x),\mathrm{diag}(\sigma_\theta^2(x))\big)$$

is **one unimodal Gaussian** per $x$. Making networks deeper can improve $\mu_\theta,\sigma_\theta$, but does **not** change the fact that the posterior family is unimodal (even full-covariance remains a single ellipsoid).

If the true posterior $p_\phi(z\mid x)$ is **multi-peaked**, this mismatch loosens the ELBO and weakens inference. Fix needs a **richer posterior class**, not just deeper nets.

</div>

### Limitation 2: posterior collapse with an expressive decoder

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Limitation 2: posterior collapse with an expressive decoder)</span></p>

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

making $\mathcal I_q(x;z)=0$ and $q_\theta(z)=p(z)$. Then $z$ carries no information about $x$, and changing $z$ doesn’t affect outputs (controllability fails). Making the networks deeper does not automatically remove this "ignore $z$" solution.

</div>

---

## What hierarchy changes (and what new issues appear)

### What improves conceptually

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What improves conceptually)</span></p>

The HVAE ELBO uses **multiple adjacent KL terms**, so the "information penalty" is:

* **distributed across layers**, and
* **localized** (each layer matches to its neighbor’s conditional prior),
  which comes from the hierarchical latent graph—not simply from depth in the encoder/decoder networks.

</div>

### Training challenges (as noted)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Training challenges (as noted))</span></p>

Even though HVAEs are more expressive, training can be unstable because:

* lower layers + decoder may already reconstruct $x$, leaving higher latents with little signal,
* gradients to deep latents can be indirect/weak,
* overly expressive conditionals can dominate reconstruction and suppress higher-level latents,
  so capacity balancing becomes important.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forward pointer)</span></p>

Diffusion models can be seen as inheriting the *progressive hierarchy idea* while sidestepping key HVAE weaknesses by fixing the encoding process and learning the generative reversal.

To avoid ambiguity, the text deviates from the "$q$=encoder, $p$=generator" convention and instead uses $p$ with clear subscripts/superscripts to indicate roles.

</div>

---


## Variational Perspective on DDPMs

### Big picture: DDPM as a "VAE-like" variational model

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Big picture: DDPM as a "VAE-like" variational model)</span></p>

DDPMs (Denoising Diffusion Probabilistic Models) can be viewed as a variational generative model with two coupled stochastic processes:

* **Forward process (fixed encoder)**: progressively **corrupt** data with Gaussian noise through a *fixed* Markov chain.
* **Reverse process (learnable decoder)**: learn a Markov chain that **denoises** step-by-step, starting from pure noise.

This "gradual generation" is easier to learn than generating a full sample in one shot.

</div>

---

## The two chains and their roles

### 1.1 Forward pass: fixed corruption (encoder)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forward pass: fixed corruption (encoder))</span></p>

A Markov chain:

$$x_0 \to x_1 \to \cdots \to x_L$$

where each step injects Gaussian noise via a fixed kernel $p(x_i\mid x_{i-1})$. As $i$ grows, the distribution becomes close to an isotropic Gaussian ("pure noise").

</div>

### 1.2 Reverse denoising: learnable generation (decoder)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reverse denoising: learnable generation (decoder))</span></p>

A reverse chain:

$$x_L \to x_{L-1} \to \cdots \to x_0$$

where we learn a parametric transition:

$$p_\phi(x_{i-1}\mid x_i)$$

so that starting from $x_L \sim p_{\text{prior}}$, we iteratively denoise to obtain a realistic $x_0$.

</div>

---

## Forward process (fixed encoder) — formalization

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(DDPM forward transition)</span></p>

Each forward step uses a fixed Gaussian transition kernel:

$$p(x_i\mid x_{i-1}) := \mathcal N\left(x_i;; \sqrt{1-\beta_i^2},x_{i-1},; \beta_i^2 I\right)$$

where $\lbrace\beta_i\rbrace_{i=1}^L$ is a predetermined increasing noise schedule, $\beta_i\in(0,1)$.

Define

$$\alpha_i := \sqrt{1-\beta_i^2}$$

Then the transition can be written as the intuitive iterative update:

$$x_i = \alpha_i x_{i-1} + \beta_i \varepsilon_i,\qquad \varepsilon_i\sim\mathcal N(0,I)\text{ iid.}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forward process (fixed encoder) — formalization)</span></p>

**Interpretation**

* $\alpha_i$ shrinks the previous state.
* $\beta_i\varepsilon_i$ adds controlled Gaussian noise.

</div>

---

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(Perturbation kernel — closed form $x_i\mid x_0$)</span></p>

By composing Gaussian transitions, you get a closed-form distribution of $x_i$ given the original data $x_0$:

$$p_i(x_i\mid x_0)=\mathcal N \left(x_i;; \bar\alpha_i x_0,; (1-\bar\alpha_i^2)I\right)$$

where

$$\bar\alpha_i := \prod_{k=1}^i \alpha_k$$

**Direct sampling form (Eq. 2.2.1):**

$$x_i = \bar\alpha_i x_0 + \sqrt{1-\bar\alpha_i^2},\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I)$$

This is the key computational convenience in DDPM training: you don’t need to simulate all intermediate steps to get $x_i$.

</div>

---

### 2.3 Prior distribution from the long-run limit

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Prior distribution from the long-run limit)</span></p>

If the noise schedule increases and $L$ is large, the forward marginal converges:

$$p_L(x_L\mid x_0)\to \mathcal N(0,I)\quad \text{as }L\to\infty$$

motivating the **prior**

$$p_{\text{prior}} := \mathcal N(0,I)$$

independent of $x_0$.

</div>

---

### 2.4 Continuous-time-like shorthand (identity in distribution)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Continuous-time-like shorthand (identity in distribution))</span></p>

Often we write (for a fixed index $t$):

$$p_t(x_t\mid x_0)=\mathcal N(x_t;\alpha_t x_0,\sigma_t^2 I)$$

equivalently (identity in distribution)

$$x_t \overset{d}{=} \alpha_t x_0 + \sigma_t \varepsilon$$

meaning $x_t$ and $\alpha_t x_0+\sigma_t\varepsilon$ have the same *law* (same density), hence same expectations for test functions.

</div>

---

## Reverse denoising process (learnable decoder)

### 3.1 The core question (Question 2.2.1)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The core question (Question 2.2.1))</span></p>

Can we compute—or approximate—the true reverse transition

$$p(x_{i-1}\mid x_i)$$

even though $x_i\sim p_i(x_i)$ is complicated?

</div>

### 3.2 Why the "obvious" Bayes formula is intractable

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the "obvious" Bayes formula is intractable)</span></p>

Bayes gives:

$$p(x_{i-1}\mid x_i)=p(x_i\mid x_{i-1})\frac{p_{i-1}(x_{i-1})}{p_i(x_i)}$$

But the marginals involve the unknown data distribution:

$$p_i(x_i)=\int p_i(x_i\mid x_0),p_{\text{data}}(x_0),dx_0$$

(and similarly for $p_{i-1}$), so exact densities are unavailable.

</div>

---

## The conditioning trick: make the target tractable

### 4.1 Condition on the clean sample

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Condition on the clean sample)</span></p>

Instead of targeting $p(x_{i-1}\mid x_i)$ directly, consider:

$$p(x_{i-1}\mid x_i, x)$$

where $x$ is the *clean* data sample (effectively $x=x_0$).

Using:

* the **Markov property** of the forward process $p(x_i\mid x_{i-1},x)=p(x_i\mid x_{i-1})$,
* and the fact all relevant distributions are **Gaussian**,

the conditional reverse kernel becomes Gaussian and has a closed form.

</div>

---

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(2.2.2 — Reverse conditional transition kernel)</span></p>

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

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition for Lemma 2.2.2)</span></p>

* The posterior mean is a *precision-weighted blend* of the clean signal $x$ and the noisy observation $x_i$.
* As noise increases, $x_i$ becomes less informative, and the weights shift accordingly.

</div>

---

## Training objective via KL minimization

### 5.1 "Ideal" objective (marginal KL; Eq. 2.2.2)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">("Ideal" objective (marginal KL; Eq. 2.2.2))</span></p>

Introduce a learnable model $p_\phi(x_{i-1}\mid x_i)$ and aim to minimize:

$$\mathbb E_{p_i(x_i)} \left[ D_{\mathrm{KL}} \big(p(x_{i-1}\mid x_i)\parallel p_\phi(x_{i-1}\mid x_i)\big)\right]$$

But this involves the intractable $p(x_{i-1}\mid x_i)$.

</div>

---

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.2.1 — Equivalence between marginal and conditional KL)</span></p>

$$
\mathbb E_{p_i(x_i)} \left[ D_{\mathrm{KL}} \big(p(x_{i-1}\mid x_i)\parallel p_\phi(x_{i-1}\mid x_i)\big)\right]
=

\mathbb E_{p_{\text{data}}(x)}\mathbb E_{p(x_i\mid x)}
 \left[
D_{\mathrm{KL}} \big(p(x_{i-1}\mid x_i,x)\parallel p_\phi(x_{i-1}\mid x_i)\big)
\right] + C,
$$

where $C$ does not depend on $\phi$.

The minimizer satisfies the mixture identity:

$$p^*(x_{i-1}\mid x_i) = \mathbb E_{p(x\mid x_i)}[p(x_{i-1}\mid x_i,x)] = p(x_{i-1}\mid x_i),\qquad x_i\sim p_i$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalence interpretation)</span></p>

**Minimizing the intractable marginal KL** is equivalent (up to an additive constant) to **minimizing a tractable conditional KL** with $x\sim p_{\text{data}}$ and $x_i\sim p(x_i\mid x)$.

* The true reverse kernel is a mixture (over possible clean $x$ consistent with $x_i$) of the tractable conditional posteriors.
* Training on the conditional KL is "the right thing" to recover the marginal reverse.

</div>

---

## Modeling $p_\phi(x_{i-1}\mid x_i)$ and simplifying the loss

### 6.1 Gaussian parameterization (Eq. 2.2.5)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gaussian parameterization (Eq. 2.2.5))</span></p>

DDPM assumes each reverse transition is Gaussian:

$$p_\phi(x_{i-1}\mid x_i):=\mathcal N\left(x_{i-1};\mu_\phi(x_i,i),\sigma^2(i)I\right)$$

* $\mu_\phi(\cdot,i):\mathbb R^D\to\mathbb R^D$ is a learnable mean function (neural net).
* $\sigma^2(i)$ is **fixed**, taken from the closed-form posterior variance in Eq. (2.2.4).

</div>

---

### 6.2 Diffusion loss as sum of KLs (Eq. 2.2.6)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Diffusion loss as sum of KLs (Eq. 2.2.6))</span></p>

Define (for one clean sample $x_0$):

$$
\mathcal L_{\text{diffusion}}(x_0;\phi):=
\sum_{i=1}^L
\mathbb E_{p(x_i\mid x_0)}
\left[
D_{\mathrm{KL}}\big(p(x_{i-1}\mid x_i,x_0)\parallel p_\phi(x_{i-1}\mid x_i)\big)
\right].
$$

</div>

---

### 6.3 Closed-form simplification to weighted MSE (Eq. 2.2.7)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Closed-form simplification to weighted MSE (Eq. 2.2.7))</span></p>

Since both distributions in the KL are Gaussians with the **same covariance** $\sigma^2(i)I$, the KL reduces to a squared error between means (plus constant):

$$
\mathcal L_{\text{diffusion}}(x_0;\phi)=
\sum_{i=1}^L
\frac{1}{2\sigma^2(i)}
\left\|\mu_\phi(x_i,i)-\mu(x_i,x_0,i)\right\|_2^2
 + C
$$

Here $\mu(x_i,x_0,i)$ is the *analytic target* from Lemma 2.2.2.

</div>

---

### 6.4 Final DDPM training objective (Eq. 2.2.8)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Final DDPM training objective (Eq. 2.2.8))</span></p>

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

</div>

---

## Practical "mental model" summary

### Forward (known, easy)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Forward (known, easy))</span></p>

* Pick schedule $\lbrace\beta_i\rbrace$, compute $\alpha_i=\sqrt{1-\beta_i^2}$, $\bar\alpha_i=\prod_{k\le i}\alpha_k$.
* Sample noisy state directly:
  
  $$x_i=\bar\alpha_i x_0+\sqrt{1-\bar\alpha_i^2},\varepsilon$$

</div>

### Reverse (learned, step-by-step)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Reverse (learned, step-by-step))</span></p>

* Start from $x_L\sim\mathcal N(0,I)$.
* For $i=L,\dots,1$, sample:
  
  $$x_{i-1}\sim \mathcal N\big(\mu_\phi(x_i,i),\sigma^2(i)I\big)$$

</div>

### Training signal comes from tractable conditioning

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Training signal comes from tractable conditioning)</span></p>

* Instead of computing $p(x_{i-1}\mid x_i)$ (hard), compute $p(x_{i-1}\mid x_i,x_0)$ (Gaussian, closed form).
* The theorem guarantees this yields an equivalent optimization problem.

</div>

---

## Key equations to memorize (minimal set)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Key equations to memorize (minimal set))</span></p>

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

</div>

---

## Notation (quick recap)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Notation (quick recap))</span></p>

* Clean data: $x_0 \sim p_{\text{data}}$.
* Noisy latent at step $i$: $x_i$.
* Noise: $\epsilon \sim \mathcal N(0, I)$.
* Noise schedule scalars:
  * $\alpha_i \in (0,1)$ (per-step "signal keep" factor),
  * $\bar \alpha_i := \prod_{j=1}^i \alpha_j$ (cumulative keep),
  * so $\bar \alpha_i^2$ appears frequently.
* Forward noising (DDPM forward process):
  
  $$x_i  =  \bar \alpha_i x_0  +  \sqrt{1-\bar \alpha_i^2},\epsilon \qquad \text{(2.2.9)}$$

</div>

---

## 2.2.4 Practical Choices of Predictions and Loss

## A. $\epsilon$-prediction (noise prediction)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why reparameterize?)</span></p>

Although DDPM can be written as predicting the **reverse mean** $\mu(\cdot)$ directly (a "mean prediction" view), implementations typically train a network to predict the **added noise** $\epsilon$. This is an *equivalent reparameterization* but is simpler and numerically well-scaled.

</div>

### Reverse mean written in terms of $\epsilon$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reverse mean written in terms of $\epsilon$)</span></p>

Using the forward identity $x_i = \bar\alpha_i x_0 + \sqrt{1-\bar\alpha_i^2}\epsilon$, the reverse mean $\mu(x_i,x_0,i)$ can be rewritten as:

$$
\mu(x_i, x_0, i)
=

\frac{1}{\alpha_i}\Bigg(
x_i - \frac{1-\alpha_i^2}{\sqrt{1-\bar\alpha_i^2}},\epsilon
\Bigg).
$$

</div>

### 3) Parameterizing the mean via a noise network

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parameterizing the mean via a noise network)</span></p>

Define a neural net $\epsilon_\phi(x_i,i)$ and plug it into the same functional form:

$$
\mu_\phi(x_i,i)
=

\frac{1}{\alpha_i}\Bigg(
x_i - \frac{1-\alpha_i^2}{\sqrt{1-\bar\alpha_i^2}},\epsilon_\phi(x_i,i)
\Bigg).
$$

</div>

### 4) Loss becomes an $\ell_2$ noise regression (up to a weight)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Loss becomes an $\ell_2$ noise regression (up to a weight))</span></p>

Because $\mu_\phi$ depends linearly on $\epsilon_\phi$,

$$
\lvert\mu_\phi(x_i,i) - \mu(x_i,x_0,i)\rvert_2^2
 \propto 
\lvert\epsilon_\phi(x_i,i) - \epsilon\rvert_2^2,
$$

with a proportionality factor that depends on $i$ (a timestep-dependent weight).

**Interpretation:** the model is a "noise detective" that estimates what noise was added; subtracting it moves $x_i$ toward a cleaner sample; repeating this over steps reconstructs data from pure noise.

</div>

---

## B. Simplified training loss (the standard DDPM loss)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Simplified DDPM loss)</span></p>

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

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key practical reason)</span></p>

The target noise $\epsilon$ has **unit variance at every step**, so the loss scale stays consistent across timesteps and you avoid exploding/vanishing targets and explicit weighting.

</div>

### Optimal solution under $\ell_2$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimal solution under $\ell_2$)</span></p>

Because it’s a least-squares regression problem:

$$\epsilon^*(x_i,i) = \mathbb E[\epsilon \mid x_i],\qquad x_i \sim p_i$$

So at optimum, the network predicts the **conditional expectation** of the true noise given the noisy input.

</div>

---

## C. Another equivalent parameterization: $x$-prediction (clean prediction)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($x$-prediction overview)</span></p>

Instead of predicting noise, you can predict the clean sample directly with a network $x_\phi(x_i,i)\approx x_0$.

</div>

### 1) Reverse mean expressed with a clean predictor

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reverse mean expressed with a clean predictor)</span></p>

Replacing the ground-truth $x_0$ in the reverse mean expression with $x_\phi(x_i,i)$ yields a model of the form:

$$
\mu_\phi(x_i,i)
=

\frac{\bar\alpha_{i-1}\beta_i^2}{1-\bar\alpha_i^2},x_\phi(x_i,i)
 + 
\frac{(1-\bar\alpha_{i-1}^2)\alpha_i}{1-\bar\alpha_i^2},x_i.
$$

(Exact coefficients depend on the schedule/notation, but the important point is: **$\mu_\phi$** is an affine combination of the predicted clean sample and the current noisy sample.)

</div>

### 2) Training objective becomes a weighted clean regression

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Training objective becomes a weighted clean regression)</span></p>

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

</div>

### Optimal solution

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimal solution)</span></p>

Again least squares implies:

$$
x^*(x_i,i) = \mathbb E[x_0\mid x_i],\qquad x_i\sim p_i.
\qquad\text{(2.2.11)}
$$

</div>

### 3) Connection between $\epsilon$-pred and $x$-pred

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection between $\epsilon$-pred and $x$-pred)</span></p>

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

</div>

---

## 2.2.5 DDPM’s ELBO (variational/MLE grounding)

## A. DDPM generative model as a reverse-time latent variable model

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(DDPM as a reverse-time latent variable model)</span></p>

Define the reverse Markov chain:

$$
p_\phi(x_0, x_{1:L})
:=
p_\phi(x_0\mid x_1),p_\phi(x_1\mid x_2)\cdots p_\phi(x_{L-1}\mid x_L),p_{\text{prior}}(x_L),
$$

and the marginal model:

$$p_\phi(x_0) := \int p_\phi(x_0,x_{1:L}),dx_{1:L}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(ELBO decomposition)</span></p>

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

</div>

## C. Practical remarks from the text

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical remarks)</span></p>

* $\mathcal L_{\text{prior}}$ can be made small by choosing the noise schedule so that $p(x_L\mid x_0)\approx p_{\text{prior}}$ (typically $\mathcal N(0,I)$).
* $\mathcal L_{\text{recon}}$ is handled via Monte Carlo estimates in practice.
* $\mathcal L_{\text{diffusion}}$ enforces that each learned reverse conditional matches the corresponding true reverse conditional.

</div>

## D. Data processing inequality view

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Data processing inequality view)</span></p>

With latents $z=x_{1:L}$:

$$
D_{\mathrm{KL}}(p_{\text{data}}(x_0)\parallel p_\phi(x_0))
 \le 
D_{\mathrm{KL}}(p(x_0,x_{1:L})\parallel p_\phi(x_0,x_{1:L})),
$$

where $p(x_0,x_{1:L})$ is the forward-process joint.

</div>

## E. HVAE-style interpretation (important conceptual framing)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(HVAE-style interpretation)</span></p>

* "Encoder" is the **fixed forward noising chain** (not learned).
* Latents $x_{1:T}$ share the same dimensionality as data.
* No per-level learned encoder or per-level KL terms like in standard HVAEs.
* Training decomposes into **well-conditioned denoising subproblems** from large noise to small noise (coarse-to-fine), which stabilizes optimization and tends to yield high sample quality.

</div>

---

## 2.2.6 Sampling (generation)



<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(DDPM sampling)</span></p>

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

</div>

## B. Another interpretation: "predict clean then step"

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Predict clean then step)</span></p>

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

</div>

## C. Why early steps are "coarse" and later steps add "detail"

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why early steps are coarse and later steps add detail)</span></p>

Even if $x_{\phi^*}$ is optimal (it predicts $\mathbb E[x_0\mid x_i]$), it only returns the **average** clean sample consistent with $x_i$. At high noise, many clean images map to similar $x_i$, so the conditional expectation can look **blurry**.

Sampling proceeds **high noise $\to$ low noise**, progressively refining:

* early steps set global structure,
* later steps sharpen and add fine details.

</div>

---

## Why DDPM sampling is slow (and the core bottleneck)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why DDPM sampling is slow (and the core bottleneck))</span></p>

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

</div>

---

## High-yield "exam style" takeaways

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(High-yield "exam style" takeaways)</span></p>

* $\epsilon$-prediction, $x$-prediction, and mean-prediction are **equivalent parameterizations** of the same underlying reverse model; they differ mainly by *what the network outputs* and the induced loss scaling.
* With $\ell_2$ loss:

  * $\epsilon^*(x_i,i)=\mathbb E[\epsilon\mid x_i]$,
  * $x^*(x_i,i)=\mathbb E[x_0\mid x_i]$.
* DDPM training is grounded as **ELBO maximization** with a sum of KLs across timesteps.
* Sampling is **iterative denoising** from $x_L\sim\mathcal N(0,I)$ down to $x_0$.
* DDPM is slow because generation is **$L$-step sequential** and uses many small noise steps to keep Gaussian reverse approximations accurate.

</div>

## Score-Based Perspective: From EBMs to NCSN


## Big picture: why EBMs show up in diffusion / score-based modeling

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Big picture: why EBMs show up in diffusion / score-based modeling)</span></p>

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

</div>

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
<p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">("Only relative energies matter")</span></p>

* If you add a constant $c$ to all energies, $E_\phi(x)\mapsto E_\phi(x)+c$:
  * numerator $\exp(-E_\phi(x)-c)$ and denominator $Z_\phi$ both get multiplied by $\exp(-c)$
  * $p_\phi(x)$ stays the same
    * $\implies$ EBMs are invariant to global energy shifts.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Global trade-off due to normalization)</span></p>

Because probabilities must sum to 1:
* decreasing energy in one region (increasing its probability mass) necessarily **decreases probability elsewhere**.
* EBMs therefore impose a **global coupling**: "making one valley deeper makes others shallower."

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
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">("Only relative energies matter")</span></p>

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

* **deterministic part:** move "uphill in probability" (follow the score / descend energy)
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

$$\boxed{\text{This "score + noise" form is the bridge to diffusion/score-based models.}}$$

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

**According to Newtonian dynamics, the motion of a particle under the force field derived from this energy is described by the ordinary differential equation (ODE). Pure deterministic dynamics (gradient flow / "Newtonian" lens):**

$$dx(t)=-\nabla_x E_\phi(x(t))dt$$

* Always moves "downhill" in energy → ends up in a **local minimum**.
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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Mini self-check questions)</span></p>

1. Why does adding a constant to $E_\phi(x)$ not change $p_\phi(x)$?
2. Write $\log p_\phi(x)$ for an EBM and show why the score does not depend on $Z_\phi$.

</div>

## From Energy-Based to Score-Based Generative Models

### Big picture

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Big picture)</span></p>

* **Key message:** to *generate* samples (e.g., via Langevin dynamics), you don’t need the full normalized density $p(x)$. You only need the **score**
  
  $$s(x)=\nabla_x \log p(x)$$

  which points toward **higher-probability (higher log-density)** regions.
* **Why move away from energies?**

  * EBMs define $p_\theta(x)\propto e^{-E_\theta(x)}$. The **partition function** is hard, but the **score** is easy:
    
    $$\nabla_x \log p_\theta(x)= -\nabla_x E_\theta(x)\quad(\text{no partition function term in } \nabla_x)$$
    
  * However, **training through an energy** with score matching tends to require **second derivatives** (Hessians).
* **Core shift:** since sampling uses only the score, we can **learn the score directly** with a neural network $s_\phi(x)$. This is the foundation of **score-based generative models**.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/score_matching.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of Score Matching. The neural network score $s_ϕ(x)$ is trained to match the ground truth score $s(x)$ using a MSE loss. Both are represented as vector fields.</figcaption>
</figure>

### Notation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Notation)</span></p>

Let $x\in\mathbb{R}^D$, and $s_\phi(x)\in\mathbb{R}^D$.

* **Score:** $s(x)=\nabla_x\log p_{\text{data}}(x)$
* **Jacobian of a vector field:** $\nabla_x s_\phi(x)\in\mathbb{R}^{D\times D}$ with entries $\frac{\partial (s_\phi)_i}{\partial x_j}$
* **Trace of Jacobian = divergence:**
  
  $$\mathrm{Tr}(\nabla_x s_\phi(x))=\sum_{i=1}^D \frac{\partial (s_\phi)_i}{\partial x_i} = \nabla\cdot s_\phi(x)$$
  
* If $s_\phi=\nabla_x u$ for scalar $u$, then $\nabla_x s_\phi = \nabla_x^2 u$ (the Hessian), and
  
  $$\nabla\cdot s_\phi = \mathrm{Tr}(\nabla_x^2 u)=\Delta u \quad(\text{Laplacian})$$

</div>

### Learning data score

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Learning the data score)</span></p>

**Goal:** Approximate the **unknown** true score $s(x)=\nabla_x \log p_{\text{data}}(x)$ from samples $x\sim p_{\text{data}}$ using a neural net $s_\phi(x)$.

**Direct (infeasible):** $\mathcal{L}\_{\mathrm{SM}}(\phi) =\frac{1}{2}\mathbb{E}\_{x\sim p\_{\text{data}}}\Big[\|s_\phi(x)-s(x)\|_2^2\Big]$

**Hyvärinen & Dayan (2005) show:** $\mathcal{L}_{\mathrm{SM}}(\phi)=\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi)+C$

where $C$ does **not** depend on $\phi$, and 

$$\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi) =\mathbb{E}_{x\sim p_{\text{data}}}\left[\mathrm{Tr}\big(\nabla_x s_\phi(x)\big)+\frac{1}{2}\|s_\phi(x)\|_2^2\right]$$

So you can minimize $\tilde{\mathcal{L}}\_{\mathrm{SM}}$ **using only samples** $x\sim p\_{\text{data}}$, without ever knowing the true score.

The **optimal solution** (**minimizer**) is the true score: $s^*(\cdot)=\nabla_x \log p(\cdot)$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Why this helps computationally)</span></p>

* If you parameterize an **energy** $E_\theta$ and set $s_\theta=-\nabla_x E_\theta$, then
  $\mathrm{Tr}(\nabla_x s_\theta)= -\mathrm{Tr}(\nabla_x^2 E_\theta)$: **second derivatives** of the energy.
* If you parameterize $s_\phi$ **directly**, $\mathrm{Tr}(\nabla_x s_\phi)$ uses **first derivatives** of the score network output w.r.t. input $x$ (still not cheap, but avoids "derivative-of-a-derivative" through an energy).

</div>

### Interpretation of the two terms in $\tilde{\mathcal{L}}_{\mathrm{SM}}$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of the two terms in $\tilde{\mathcal{L}}_{\mathrm{SM}}$)</span></p>

$$
\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi)
=\mathbb{E}_{p_{\text{data}}}\left[\underbrace{\mathrm{Tr}(\nabla_x s_\phi(x))}_{\text{divergence term}}+\underbrace{\frac{1}{2}\|s_\phi(x)\|_2^2}_{\text{magnitude term}}\right]
$$

</div>

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
* Deterministic part $\eta s(x)$: moves "uphill" in log-density.
* Noise $\sqrt{2\eta}\varepsilon$: keeps exploration and yields the correct stationary distribution (in the idealized limit).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Prologue: Score-Based Generative Models)</span></p>

* The **score function** started as a way to train EBMs efficiently.
* It has become the **central object** in modern **score-based diffusion models**:
  * Theoretical formulation + practical implementation are built around learning scores
  * Generation becomes "simulate (reverse) stochastic processes using learned scores"

</div>

## Denoising Score Matching (DSM) + Sliced Score Matching (Hutchinson)

### Vanilla score matching is hard even with score training

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Vanilla score matching is hard even with score training)</span></p>

Minimizaing the "direct" score matching loss is infeasible. A classic workaround (Hyvärinen-style) is an equivalent objective that removes the explicit data-score target but introduces a **trace-of-Jacobian** term:

$$\tilde{\mathcal{L}}_{\text{SM}}(\phi)=\mathbb{E}_{x\sim p_{\text{data}}}\Big[\mathrm{Tr}(\nabla_x s_\phi(x))+\frac12\|s_\phi(x)\|_2^2\Big]$$

**Problem:** Computing $\mathrm{Tr}(\nabla_x s_\phi(x))$ (trace of the Jacobian of a $D$-dimensional vector field) has **worst-case complexity $\mathcal{O}(D^2)$** $\implies$ not scalable in high dimensions.

</div>

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

**Interpretation:** You "test" the model’s behavior only along **random directions** ("random slices"), rather than fully constraining all partial derivatives.

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conditioning trick: corrupt the data with known noise)</span></p>

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

</div>

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

* Adding noise makes the distribution **smooth/full-dimensional**, which avoids the "score undefined on a manifold" issue.
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
   * Noise **smooths** the distribution, filling in low-density "gaps" between separated modes.
   * This improves training signal and helps Langevin dynamics move through low-density regions more effectively (less getting stuck).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sampling with a trained score model)</span></p>

As before, we use discrete time Langevin dynamics:

Given a score model $s_\phi(\cdot;\sigma)$ at a fixed noise level $\sigma$, iterate

$$\tilde x_{n+1} = \tilde x_n + \eta s_\phi(\tilde x_n;\sigma) + \sqrt{2\eta}\varepsilon_n, \qquad \varepsilon_n\sim\mathcal N(0,I)$$

* $\eta>0$ here is the **step size** (careful: later pages reuse $\eta$ for "natural parameter" in exponential families).
* This is Langevin sampling where the "force" term $\nabla \log p_\sigma(\tilde x)$ is replaced by the learned $s_\phi$.

</div>


---

## 3.3.4 Why DSM is denoising: Tweedie’s formula

### Setup (Gaussian corruption with scaling)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Setup (Gaussian corruption with scaling))</span></p>

Assume:
* $x\sim p_{\text{data}}$
* $\tilde x\mid x \sim \mathcal N(\alpha x,\sigma^2 I)$, with $\alpha\neq 0$

Define the noisy marginal:

$$p_\sigma(\tilde x) = \int \mathcal N(\tilde x;\alpha x,\sigma^2 I)p_{\text{data}}(x)dx$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Tweedie’s formula)</span></p>

$$\alpha\mathbb E[x\mid \tilde x] = \tilde x + \sigma^2 \nabla_{\tilde x}\log p_\sigma(\tilde x)$$

Equivalently, the **posterior mean / denoiser** is

$$\mathbb E[x\mid \tilde x] = \frac{1}{\alpha}\Big(\tilde x + \sigma^2 \nabla_{\tilde x}\log p_\sigma(\tilde x)\Big)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition (why this is "denoising"))</span></p>

* The score $\nabla_{\tilde x}\log p_\sigma(\tilde x)$ points toward regions where noisy samples are more likely.
* Moving $\tilde x$ by a step of size $\sigma^2$ in the score direction produces the **conditional mean of the clean signal** (up to the $\alpha$ scaling).

</div>

### Connection to DSM-trained score networks

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to DSM-trained score networks)</span></p>

If DSM gives $s_\phi(\tilde x)\approx \nabla_{\tilde x}\log p_\sigma(\tilde x)$, then an estimated denoiser is:

$$\widehat{x}(\tilde x) = \frac{1}{\alpha}\Big(\tilde x + \sigma^2, s_\phi(\tilde x)\Big)$$

So: **learning the score is (almost directly) learning a denoiser** via Tweedie.

</div>

---

## (Optional) Higher-order Tweedie via an exponential-family view

### Exponential family observation model

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Exponential family observation model)</span></p>

Assume the conditional law of $\tilde x$ given a latent natural parameter $\eta\in\mathbb R^D$ is

$$q_\sigma(\tilde x\mid \eta) = \exp(\eta^\top \tilde x - \psi(\eta)) q_0(\tilde x)$$

* $q_0(\tilde x)$ is the **base measure** (independent of $\eta$).
* For additive Gaussian noise with variance $\sigma^2 I$,
  
  $$q_0(\tilde x) = (2\pi\sigma^2)^{-D/2}\exp\left(-\frac{\|\tilde x\|^2}{2\sigma^2}\right)$$

Let $p(\eta)$ be a prior over $\eta$. The noisy marginal is

$$p_\sigma(\tilde x) = \int q_\sigma(\tilde x\mid \eta)p(\eta)d\eta$$

Define the "log-normalizer in $\tilde x$":

$$\lambda(\tilde x) := \log p_\sigma(\tilde x) - \log q_0(\tilde x)$$

Then the posterior has the form

$$p(\eta\mid \tilde x)\propto \exp(\eta^\top \tilde x - \psi(\eta) - \lambda(\tilde x))p(\eta)$$

</div>

### Derivatives of $\lambda$ give posterior cumulants

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Derivatives of $\lambda$ give posterior cumulants)</span></p>

A core exponential-family identity:
* $\nabla_{\tilde x}\lambda(\tilde x) = \mathbb E[\eta\mid \tilde x]$
* $\nabla_{\tilde x}^2\lambda(\tilde x) = \mathrm{Cov}[\eta\mid \tilde x]$
* More generally:
  
  $$\nabla_{\tilde x}^{(k)}\lambda(\tilde x) = \kappa_k(\eta\mid \tilde x),\quad k\ge 3$$
  
  where $\kappa_k$ are conditional cumulants.

</div>

### Specialize to Gaussian location noise (recover classic Tweedie + covariance)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Specialize to Gaussian location noise (recover classic Tweedie + covariance))</span></p>

For Gaussian location models, one can take $\eta = x/\sigma^2$. Then:

* Posterior mean:
  
  $$\mathbb E[x\mid \tilde x] = \tilde x + \sigma^2 \nabla_{\tilde x}\log p_\sigma(\tilde x)$$
  
* Posterior covariance:
  
  $$\mathrm{Cov}[x\mid \tilde x] = \sigma^2 I + \sigma^4 \nabla_{\tilde x}^2\log p_\sigma(\tilde x)$$
  
* Higher cumulants scale with higher derivatives of $\log p_\sigma(\tilde x)$.

**Takeaway:** not only denoising (mean), but also **uncertainty estimates** (covariance) and higher statistics are encoded in higher-order "scores" (higher derivatives).

</div>

---

## Quick "what to remember" checklist

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Quick "what to remember" checklist)</span></p>

* **Sampling:** $\tilde x_{n+1}=\tilde x_n+\eta s_\phi(\tilde x_n;\sigma)+\sqrt{2\eta}\varepsilon_n$.
* **Why noise helps:** (i) score well-defined everywhere, (ii) smoother landscape improves mode coverage.
* **Tweedie:** $\mathbb E[x\mid \tilde x]=\frac{1}{\alpha}(\tilde x+\sigma^2\nabla_{\tilde x}\log p_\sigma(\tilde x))$.
* **DSM ⇒ denoiser:** replace $\nabla \log p_\sigma$ by $s_\phi$.
* **Higher-order:** derivatives of $\log p_\sigma$ relate to posterior covariance and cumulants.

</div>

## SURE, Tweedie, and Generalized Score Matching

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Study notes: SURE, Tweedie, and (Generalized) Score Matching)</span></p>

These pages explain two closely related ideas:

1. **SURE** gives an *unbiased, observable* estimate of denoising MSE using only noisy data.
2. The **SURE-optimal denoiser** is the **Bayes posterior mean**, which equals a **score-based correction** (Tweedie). This directly links denoisers $\iff$ scores $\iff$ score matching objectives.
3. **Generalized score matching (GSM)** unifies classical score matching, denoising score matching, and higher-order variants through a general linear operator $\mathcal L$.

</div>

---

## 3.3.5 Why DSM is Denoising: SURE

## Setup: additive Gaussian noise

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Setup: additive Gaussian noise)</span></p>

We observe

$$\tilde{\mathbf x}=\mathbf x+\sigma \boldsymbol\epsilon,\qquad \boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I)$$

where $\mathbf x\in\mathbb R^d$ is the unknown clean signal and $\tilde{\mathbf x}$ is noisy.

A **denoiser** is a (weakly differentiable) map

$$\mathbf D:\mathbb R^d\to\mathbb R^d,\qquad \mathbf D(\tilde{\mathbf x})\approx \mathbf x$$

</div>

## True denoising quality: conditional MSE risk

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(True denoising quality: conditional MSE risk)</span></p>

For a fixed (unknown) clean $\mathbf x$,

$$R(\mathbf D;\mathbf x):=\mathbb E_{\tilde{\mathbf x}\mid \mathbf x}\Big[\|\mathbf D(\tilde{\mathbf x})-\mathbf x\|_2^2\ \big|\ \mathbf x\Big]$$

Problem: this depends on $\mathbf x$, so you can’t compute it from $\tilde{\mathbf x}$ alone.

</div>

---

## SURE: an observable surrogate for the MSE

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stein’s Unbiased Risk Estimator (SURE))</span></p>

$$
\mathrm{SURE}(\mathbf D;\tilde{\mathbf x}) = \|\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x}\|_2^2 + 2\sigma^2\nabla_{\tilde{\mathbf x}}\cdot \mathbf D(\tilde{\mathbf x})- D\sigma^2.
$$

* $\nabla_{\tilde{\mathbf x}}\cdot \mathbf D(\tilde{\mathbf x})$ is the **divergence** of $\mathbf D$:

  $$\nabla_{\tilde{\mathbf x}}\cdot \mathbf D(\tilde{\mathbf x})=\sum_{i=1}^d \frac{\partial D_i(\tilde{\mathbf x})}{\partial \tilde{x}_i}$$

* Importantly: **SURE depends only on $\tilde{\mathbf x}$** (and $\sigma$), not on $\mathbf x$.

</div>

### Why the terms make sense (intuition)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the terms make sense (intuition))</span></p>

* $\|\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x}\|^2$: how much the denoiser changes the input.
  * Alone, it *underestimates* true error because $\tilde{\mathbf x}$ is already corrupted.
* $2\sigma^2 \nabla\cdot \mathbf D(\tilde{\mathbf x})$: **correction term** accounting for noise variance via sensitivity of $\mathbf D$.
* $-d\sigma^2$: constant offset that fixes the bias.

</div>

---

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(SURE unbiasedness)</span></p>

For any fixed but unknown $\mathbf x$,

$$\mathbb E_{\tilde{\mathbf x}\mid \mathbf x}\big[\mathrm{SURE}(\mathbf D;\mathbf x+\sigma\epsilon)\ \big|\ \mathbf x\big] = R(\mathbf D;\mathbf x)$$

So **minimizing SURE (in expectation or empirically)** is equivalent to minimizing the true denoising MSE risk, while using only noisy data.

</div>

### Derivation sketch (how Stein’s identity enters)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Derivation sketch (how Stein’s identity enters))</span></p>

Start from:

$$\|\mathbf D(\tilde{\mathbf x})-\mathbf x\|^2 = \|\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x} + (\tilde{\mathbf x}-\mathbf x)\|^2$$

Expand and use $\tilde{\mathbf x}-\mathbf x=\sigma\epsilon$. The cross-term contains $\mathbb E[\epsilon^\top g(\mathbf x+\sigma\epsilon)]$ with $g(\tilde{\mathbf x})=\mathbf D(\tilde{\mathbf x})-\tilde{\mathbf x}$. Stein’s lemma converts this to a divergence term:

$$\mathbb E[\epsilon^\top g(\mathbf x+\sigma\epsilon)] = \sigma\mathbb E[\nabla_{\tilde{\mathbf x}}\cdot g(\tilde{\mathbf x})]$$

Since $\nabla\cdot(\tilde{\mathbf x})=d$, you get exactly the SURE formula.

</div>

---

## Link to Tweedie’s formula and Bayes optimality

## Noisy marginal

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Noisy marginal)</span></p>

Let the noisy marginal be the convolution:

$$p_\sigma(\tilde{\mathbf x}) := (p_{\text{data}} * \mathcal N(0,\sigma^2\mathbf I))(\tilde{\mathbf x})$$

</div>

## SURE minimization ⇒ Bayes optimal denoiser

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(SURE minimization ⇒ Bayes optimal denoiser)</span></p>

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

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Tweedie’s identity)</span></p>

$$\mathbf D^*(\tilde{\mathbf x})=\mathbb E[\mathbf x\mid \tilde{\mathbf x}] = \tilde{\mathbf x}+\sigma^2\nabla_{\tilde{\mathbf x}}\log p_\sigma(\tilde{\mathbf x})$$

So the Bayes-optimal denoiser equals **input + $\sigma^2$ times the noisy score**.

</div>

---

## Relationship between SURE and score matching

## Parameterize denoiser via a score field

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parameterize denoiser via a score field)</span></p>

Motivated by Tweedie:

$$\mathbf D(\tilde{\mathbf x}) = \tilde{\mathbf x}+\sigma^2 \mathbf s_\phi(\tilde{\mathbf x};\sigma)$$

where $\mathbf s_\phi(\cdot;\sigma)\approx \nabla_{\tilde{\mathbf x}}\log p_\sigma(\cdot)$.

</div>

## Plugging into SURE yields Hyvärinen’s objective (up to constants)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Plugging into SURE yields Hyvärinen’s objective (up to constants))</span></p>

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

</div>

---

## 3.3.6 Generalized Score Matching (GSM)

## Motivation: unify many "score-like" training targets

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation: unify many "score-like" training targets)</span></p>

Classical score matching, denoising score matching, and higher-order variants all target a quantity of the form

$$\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}$$

for some **linear operator** $\mathcal L$ acting on the density $p$.

* Classical case $\mathcal L=\nabla_{\mathbf x}$:
  
  $$\frac{\mathcal L p}{p}=\frac{\nabla p}{p}=\nabla \log p$$

Key idea: the $\frac{\mathcal L p}{p}$ structure enables **integration by parts** to remove unknown normalizing constants, producing a tractable objective depending only on samples and the learned field.

</div>

---

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Fisher Divergence)</span></p>

Let $p$ be data and $q$ be a model density. Define

$$
\mathcal D_{\mathcal L}(p\parallel q) := \int p(\mathbf x)\left\|\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}-\frac{\mathcal L q(\mathbf x)}{q(\mathbf x)}\right\|_2^2 d\mathbf x
$$

If $\mathcal L$ is **complete** (informally: $\frac{\mathcal L p_1}{p_1}=\frac{\mathcal L p_2}{p_2}$ a.e. implies $p_1=p_2$ a.e.), then $\mathcal D_{\mathcal L}(p\parallel q)=0$ identifies $q=p$.
For $\mathcal L=\nabla$, this recovers the classical Fisher divergence.

</div>

---

## Score parameterization (avoid explicit normalized $q$)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Score parameterization (avoid explicit normalized $q$))</span></p>

Instead of modeling $q$, directly learn a vector field $\mathbf s_\phi(\mathbf x)$ to approximate $\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}$:

$$
\mathcal D_{\mathcal L}(p\parallel \mathbf s_\phi) := \mathbb E_{\mathbf x\sim p}\left[\left\|\mathbf s_\phi(\mathbf x)-\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}\right\|_2^2\right].
$$

The target is unknown, but integration by parts makes the loss computable.

</div>

### Adjoint operator and integration by parts trick

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Adjoint operator and integration by parts trick)</span></p>

Define the adjoint $\mathcal L^\dagger$ by:

$$\int (\mathcal L f)^\top g = \int f (\mathcal L^\dagger g) \quad \text{for all test functions } f,g$$

(assuming boundary terms vanish).

Expanding the square and applying the adjoint identity yields the tractable objective:

$$
\mathcal L_{\text{GSM}}(\phi) = \mathbb E_{\mathbf x\sim p}\Big[\frac12\|\mathbf s_\phi(\mathbf x)\|_2^2-(\mathcal L^\dagger \mathbf s_\phi)(\mathbf x)\Big]
+\text{const},
$$

where "const" does not depend on $\phi$.

</div>

### Check: recovering Hyvärinen’s score matching

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Check: recovering Hyvärinen’s score matching)</span></p>

For $\mathcal L=\nabla$, we have $\mathcal L^\dagger=-\nabla\cdot$ (negative divergence), so:

$$\mathbb E_p\Big[\tfrac12\|\mathbf s_\phi\|^2-(\mathcal L^\dagger \mathbf s_\phi)\Big] = \mathbb E_p\Big[\tfrac12\|\mathbf s_\phi\|^2+\nabla\cdot\mathbf s_\phi\Big]$$

which is Hyvärinen’s classical objective.

</div>

---

## Examples of operators $\mathcal L$

## Classical score matching

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classical score matching)</span></p>

Take $\mathcal L=\nabla_{\mathbf x}$. Then

$$\frac{\mathcal L p(\mathbf x)}{p(\mathbf x)}=\nabla_{\mathbf x}\log p(\mathbf x)$$

</div>

## Denoising score matching (Gaussian corruption)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Denoising score matching (Gaussian corruption))</span></p>

For additive Gaussian noise at level $\sigma$, define an operator on scalar $f$:

$$(\mathcal L f)(\tilde{\mathbf x})=\tilde{\mathbf x}f(\tilde{\mathbf x})+\sigma^2\nabla_{\tilde{\mathbf x}} f(\tilde{\mathbf x})$$

Then

$$\frac{\mathcal L p_\sigma(\tilde{\mathbf x})}{p_\sigma(\tilde{\mathbf x})} = \tilde{\mathbf x}+\sigma^2\nabla_{\tilde{\mathbf x}}\log p_\sigma(\tilde{\mathbf x})\mathbb E[\mathbf x_0\mid \tilde{\mathbf x}]$$

which is exactly **Tweedie’s identity**. Minimizing $\mathcal L_{\text{GSM}}$ with this operator trains $\mathbf s_\phi$ to approximate the **denoiser**, recovering denoising score matching behavior.

</div>

## Higher-order targets

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Higher-order targets)</span></p>

By stacking derivatives inside $\mathcal L$, you can target:

* $\nabla^2 \log p$ (Hessian of log-density),
* higher derivatives,
  which relate to **posterior covariance** and higher-order cumulants.

</div>

---

## Key takeaways / mental model

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Key takeaways / mental model)</span></p>

* **SURE** lets you estimate denoising MSE without clean targets; its correction term is a **divergence**.
* Minimizing expected **SURE** yields the **posterior mean denoiser**:
  
  $$\mathbf D^*(\tilde{\mathbf x})=\mathbb E[\mathbf x\mid\tilde{\mathbf x}]$$
  
* **Tweedie** rewrites that denoiser using the **score of the noisy marginal**:
  
  $$\mathbb E[\mathbf x\mid\tilde{\mathbf x}] = \tilde{\mathbf x}+\sigma^2\nabla \log p_\sigma(\tilde{\mathbf x})$$
  
* Parameterizing $\mathbf D(\tilde{\mathbf x})=\tilde{\mathbf x}+\sigma^2\mathbf s_\phi(\tilde{\mathbf x};\sigma)$ turns SURE minimization into (alternative) **score matching** (up to constants).
* **Generalized score matching**: pick an operator $\mathcal L$; learn $\mathbf s_\phi \approx \mathcal Lp/p$; integration by parts gives a tractable loss. This **unifies** classical SM, DSM, and higher-order variants.

</div>

## Multi-Noise Denoising Score Matching (NCSN) + Annealed Langevin Dynamics (Sections 3.4–3.6)

### Big picture

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Big picture)</span></p>

* **Goal (score-based generative modeling):** learn the **score**
  
  $$\nabla_x \log p(x)$$
  
  (gradient of log-density), which lets you **generate samples** by running dynamics that follow this gradient plus noise (e.g., Langevin).
* **Problem:** learning / sampling with a **single** noise level is unreliable and slow.
* **Fix (NCSN, Song & Ermon 2019):** train **one network conditioned on noise level** to estimate scores for **many noise scales**, then sample by **annealing** from high noise → low noise.

</div>

### Multi-Noise Levels of Denoising Score Matching (NCSN)

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/score_matching_inaccuracy.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of SM inaccuracy (revisiting Illustration of Score Matching). the red region indicates low-density areas with potentially inaccurate score estimates due to limited sample coverage, while high-density regions tend to yield more accurate estimates.</figcaption>
</figure>

### Motivation: why one noise level is not enough

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation: why one noise level is not enough)</span></p>

Adding Gaussian noise "smooths" the data distribution, but:

* **Low noise (small variance):**
  * Distribution is sharp/multi-modal; **Langevin struggles to move between modes**.
  * In low-density regions, the score can be inaccurate and gradients can vanish → **poor exploration**.
* **High noise (large variance):**
  * Sampling/mixing is easier, but the model captures only **coarse structure** → samples look **blurry**, lose fine detail.
* **High-dimensional issues:** Langevin can be **slow**, sensitive to **initialization**, can get stuck near **plateaus/saddles**.

**Core idea:** use **multiple noise levels**:
* High noise: explore globally / cross modes.
* Low noise: refine details.

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/ncsn.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Illustration of NCSN. The forward process perturbs the data with multiple levels of additive Gaussian noise $p_σ(x_σ\mid x)$. Generation proceeds via Langevin sampling at each noise level, using the result from the current level to initialize sampling at the next lower variance.</figcaption>
</figure>

### Training

### Noise levels

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Noise levels)</span></p>

Choose a sequence of noise scales:

$$0 < \sigma_1 < \sigma_2 < \cdots < \sigma_L$$

* $\sigma_1$: small enough to preserve fine details
* $\sigma_L$: large enough to heavily smooth the distribution (easier learning)

</div>

### Forward perturbation (data → noisy)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forward perturbation (data → noisy))</span></p>

Sample clean $x \sim p_{\text{data}}$. Create noisy version:

$$x_\sigma = x + \sigma \epsilon, \quad \epsilon \sim \mathcal N(0, I)$$

**Perturbation kernel:**

$$p_\sigma(x_\sigma \mid x) := \mathcal N(x_\sigma; x, \sigma^2 I)$$

**Marginal (smoothed) distribution at noise $\sigma$:**

$$p_\sigma(x_\sigma) = \int p_\sigma(x_\sigma\mid x)p_{\text{data}}(x)dx$$

**Interpretation:** $p_\sigma$ is a Gaussian-smoothed version of $p_{\text{data}}$. Larger $\sigma$ $\implies$ smoother.

</div>

### Noise-conditional score network

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Noise-conditional score network)</span></p>

Train a single network $s_\phi(x,\sigma)$ to approximate:

$$s_\phi(x,\sigma) \approx \nabla_x \log p_\sigma(x)$$

</div>

---

## Training objective of NCSN (DSM across all noise levels)

### Weighted multi-noise DSM loss

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weighted multi-noise DSM loss)</span></p>

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

</div>

### Key fact (optimal solution)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(Optimal solution))</span></p>

Minimizing DSM at each $\sigma$ yields:

$$s^*(\cdot,\sigma) = \nabla_x \log p_\sigma(\cdot), \quad \forall \sigma \in {\sigma_i}_{i=1}^L$$

So you learn the **true score of the smoothed distribution** at every noise scale.

</div>

---

## Relationship to DDPM loss (Tweedie connection)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relationship to DDPM loss (Tweedie connection))</span></p>

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

</div>

---

## 3.4.3 Sampling — Annealed Langevin Dynamics (ALD)

### Why annealing helps

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why annealing helps)</span></p>

* At large $\sigma$, $p_\sigma$ is smooth ⇒ sampling is easier (better mixing).
* Gradually reduce $\sigma$ and **refine** samples using the next score model.
* Each stage uses the previous stage’s output as a strong initialization.

</div>

### Langevin update at noise level $\sigma_\ell$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Langevin update at noise level $\sigma_\ell$)</span></p>

Given current $\tilde x_n$:

$$\tilde x_{n+1} = \tilde x_n + \eta_\ell s_\phi(\tilde x_n,\sigma_\ell) + \sqrt{2\eta_\ell}\epsilon_n, \quad \epsilon_n\sim\mathcal N(0,I)$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Annealed Langevin Dynamics)</span></p>

* Initialize $x^{\sigma_L}\sim\mathcal N(0,I)$ (often equivalent to choosing a large-noise prior).
* For $\ell = L, L-1,\dots,2$:
  * run $N_\ell$ Langevin steps using $s_\phi(\cdot,\sigma_\ell)$
  * set $x^{\sigma_{\ell-1}}\leftarrow$ final sample (init for next level)
* Output $x^{\sigma_1}$.

**Step size scaling (typical):**

$$\eta_\ell = \delta\cdot \frac{\sigma_\ell^2}{\sigma_1^2},\quad \delta>0$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

Bigger noise $\implies$ you can take bigger steps.

</div>

---

## Why NCSN sampling is slow (important bottleneck)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why NCSN sampling is slow (important bottleneck))</span></p>

NCSN sampling uses **annealed MCMC** across scales $\lbrace\sigma_i\rbrace_{i=1}^L$. If you do $K$ updates per scale, you need $\sim L\times K$ network evaluations.

Two reasons $L\times K$ must be large:

1. **Local accuracy & stability:** learned score is only reliable locally → requires small step sizes and many steps to avoid bias/instability.
2. **Slow mixing in high dimensions:** local MCMC moves explore multi-modal high-D distributions inefficiently → many iterations to reach typical regions.

Overall cost:

$$\mathcal O(LK)$$

sequential network passes ⇒ computationally slow.

</div>

---

## 3.5 Summary: Comparative view of NCSN and DDPM

### Forward / corruption process (conceptual comparison)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Forward / corruption process (conceptual comparison))</span></p>

* **NCSN:** additive Gaussian noise at multiple scales. The table shows transitions like:

  $$x_{i+1} = x_i + \sqrt{\sigma_{i+1}^2 - \sigma_i^2}\epsilon$$
  
  (incrementally increasing variance).
* **DDPM:** Markov chain with variance schedule $\beta_i$:
  
  $$x_{i+1} = \sqrt{1-\beta_i}x_i + \sqrt{\beta_i}\epsilon$$

</div>

### Loss / training target

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Loss / training target)</span></p>

* **NCSN:** score loss equivalent to
  
  $$\mathbb E\big[\|s_\phi(x_i,\sigma_i) + \epsilon/\sigma_i\|^2\big]$$
  
  (score matches scaled negative noise).
* **DDPM:** noise prediction loss
  
  $$\mathbb E\big[\|\epsilon_\phi(x_i,i)-\epsilon\|^2\big]$$

</div>

### Sampling

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sampling)</span></p>

* **NCSN:** Langevin per noise "layer"; output initializes next lower noise.
* **DDPM:** traverse learned reverse chain $p_\phi(x_{i-1}\mid x_i)$.

</div>

### Shared bottleneck

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Shared bottleneck)</span></p>

Both rely on **dense discretization** ⇒ often **hundreds/thousands** of steps ⇒ slow generation.

**Question 3.5.1:** *How can we accelerate sampling in diffusion models?*
(Flag for later chapters on faster solvers / fewer steps.)

</div>

---

## 3.6 Closing remarks (what this chapter sets up)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Closing remarks (what this chapter sets up))</span></p>

* Score-based view comes from EBMs: score avoids dealing directly with the **intractable partition function**.
* Progression:

  1. score matching →
  2. **denoising score matching (DSM)** via noise perturbation →
  3. **Tweedie’s formula** connects score estimation to denoising →
  4. extend from single noise to **NCSN** (multi-noise) + **annealed Langevin**.
* Key convergence: **NCSN and DDPM** look different but share structure and **same bottleneck** (slow sequential sampling).
* Next step: move to **continuous time**, unify methods as discretizations of a **Score SDE**, and connect variational + score-based views via differential equations (motivates advanced numerical methods to speed up sampling).

</div>

---

## Quick "exam-ready" checklist

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Quick "exam-ready" checklist)</span></p>

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

</div>

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


## Score SDE Framework

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Why "Score SDE")</span></p>

You’ve seen diffusion models in **discrete time** (e.g., **DDPM**) and in the **score-based / noise-conditional** view (e.g., **NCSN**). The **Score SDE framework** is the **continuous-time limit** that **unifies** them.

Key idea:
* The forward "add-noise" process can be written as a **(stochastic) differential equation**.
* **Generation (sampling)** becomes "solve a differential equation backward in time".
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

**Interpretation:** you can think of a "time" index where the **noise level increases** as time increases.

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

### A unified "small step" view on a time grid

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
* $g:\mathbb R \to \mathbb R$ (diffusion "strength")


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

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Forward SDE (data $\to$ noise))</span></p>

$$
d\mathbf{x}(t)=\mathbf{f}(\mathbf{x}(t),t),dt + g(t),d\mathbf{w}(t),
\qquad \mathbf{x}(0)\sim p_{\text{data}}.
\qquad\text{(4.1.3)}
$$

* $\mathbf{f}(\cdot,t):\mathbb{R}^D\to\mathbb{R}^D$: **drift** (deterministic trend).
* $g(t)\in\mathbb{R}$: **scalar diffusion coefficient** (noise strength schedule).
* $\mathbf{w}(t)$: standard **Wiener process** (Brownian motion).

Once $\mathbf{f}$ and $g$ are chosen, the forward process is fully specified.
It describes how clean data is progressively corrupted by injecting **Gaussian noise** over time.

</div>

---

### Figure intuition (forward process)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Figure intuition (forward process))</span></p>

* At $t=0$: distribution is complex, e.g. bimodal $p_0=p_{\text{data}}$.
* As $t$ increases: the **marginal density** $p_t$ "smooths out".
* At $t=T$: $p_T\approx p_{\text{prior}}$ (typically a simple Gaussian).

**PF-SDE vs PF-ODE paths (from the figure caption)**

* **PF-SDE**: sample trajectories are stochastic (wiggly).
* **PF-ODE**: deterministic counterpart gives a **transport map for densities**; it is *not* generally the mean of SDE sample paths from a single initial point.

</div>

---

## Perturbation kernels and marginals

### Perturbation kernel

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Perturbation kernel)</span></p>

$$p_t(\mathbf{x}_t\mid \mathbf{x}_0)$$

describes how one clean sample $\mathbf{x}_0\sim p_{\text{data}}$ becomes a noisy $\mathbf{x}_t$ at time $t$.

</div>

### Marginal density (mixture over data)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Marginal density (mixture over data))</span></p>

$$
p_t(\mathbf{x}_t)=\int p_t(\mathbf{x}_t\mid \mathbf{x}_0),p_{\text{data}}(\mathbf{x}_0),d\mathbf{x}_0,
\qquad (p_0=p_{\text{data}}).
\qquad\text{(4.1.5)}
$$

So $p_t$ is a (generally complicated) mixture induced by the kernel + the data distribution.

</div>

---

## Affine drift special case (closed-form Gaussian kernels)

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Affine drift special case (closed-form Gaussian kernels))</span></p>

A common analytically convenient assumption is that drift is **linear in $\mathbf{x}$**:

$$
\mathbf{f}(\mathbf{x},t)=f(t),\mathbf{x},
\qquad\text{(4.1.4)}
$$

where $f(t)$ is scalar (typically **non-positive**, so the signal decays).

</div>

### Consequence: Gaussian conditional at every time

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequence: Gaussian conditional at every time)</span></p>

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

* You can sample $\mathbf{x}_t\mid \mathbf{x}_0$ **directly** without numerically simulating the SDE ("simulation-free").
* Both **NCSN** and **DDPM** fall into this affine-drift setting (in the continuous-time view).

</div>

---

## Convergence to a simple prior

By choosing $f(t)$ and $g(t)$ appropriately, the forward diffusion eventually "forgets" the initial condition.

### Mean decays (forgetting $\mathbf{x}_0$)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mean decays (forgetting $\mathbf{x}_0$))</span></p>

If $f(u)\le 0$,

$$
\mathbf{m}(T)=\exp\Big(\int_0^T f(u),du\Big)\mathbf{x}_0 \to 0
\quad \text{as }T\to\infty,
$$

so dependence on $\mathbf{x}_0$ vanishes.

</div>

### Marginal approaches a prior

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Marginal approaches a prior)</span></p>

As the conditional becomes independent of $\mathbf{x}_0$, the marginal simplifies:

$$
p_T(\mathbf{x}_T)\approx p_{\text{prior}}(\mathbf{x}_T),
\qquad
p_T(\mathbf{x}_T\mid \mathbf{x}_0)\approx p_{\text{prior}}(\mathbf{x}_T).
$$

Thus, the forward SDE maps a complex data distribution into a tractable prior, giving a clean starting point for *reversal/generation*.

</div>

---

## 4.1.3 Reverse-Time Stochastic Process for Generation

### Goal

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Goal)</span></p>

Generate data by "reversing" the forward corruption:

* Start at $t=T$ from $\mathbf{x}_T\sim p_{\text{prior}}\approx p_T$,
* Evolve **backward** to $t=0$ to obtain a sample from $p_{\text{data}}$.

</div>

### Why reversing is subtle for SDEs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why reversing is subtle for SDEs)</span></p>

* For ODEs: time reversal is basically tracing trajectories backward.
* For SDEs: individual stochastic paths aren’t reversible in a naive sense; the key fact is that **the distributional evolution** *is* reversible in a precise way.

This is formalized by a time-reversal result (attributed here to **Anderson (1982)**): the time-reversed process is again an SDE with a modified drift involving the **score**.

</div>

---

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reverse-time SDE (noise $\to$ data))</span></p>

Let $\bar{\mathbf{x}}(t)$ denote the reverse-time process. Then:

$$
d\bar{\mathbf{x}}(t)=
\Big[\mathbf{f}(\bar{\mathbf{x}}(t),t)-g^2(t)\nabla_{\mathbf{x}}\log p_t(\bar{\mathbf{x}}(t))\Big],dt

+ g(t),d\bar{\mathbf{w}}(t),
  \qquad
  \bar{\mathbf{x}}(T)\sim p_{\text{prior}}\approx p_T.
  \tag{4.1.6}
$$

</div>

### Reverse-time Brownian motion

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reverse-time Brownian motion)</span></p>

$$\bar{\mathbf{w}}(t) := \mathbf{w}(T-t)-\mathbf{w}(T)$$

is a Wiener process when viewed in reverse time.

</div>

### Key new ingredient: the score term

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key new ingredient: the score term)</span></p>

$$\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$

is the **score** of the marginal at time $t$. The extra drift correction

$$-g^2(t)\nabla_{\mathbf{x}}\log p_t(\cdot)$$

is what makes the reverse dynamics reproduce the correct marginals.

**Important:** the reverse process does **not** inject arbitrary randomness: the diffusion term $g(t),d\bar{\mathbf{w}}(t)$ is *paired* with the score-driven drift so that the distribution flows correctly back to data.

</div>

---

## Conceptual intuition: why does the reverse process work?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conceptual intuition: why does the reverse process work?)</span></p>

At first it seems paradoxical: you add noise in reverse time too, so why don’t you just get "more random"?

The intuition is:
* The **score drift** points toward **higher-density regions** of $p_t$, pulling samples toward structured regions (toward the "data manifold" at small $t$).
* The Brownian term provides **controlled exploration**, but its effect is balanced by the score correction.
* Together they produce a process whose marginals match the reversed marginals of the forward SDE.

</div>

---

## Connection to Langevin dynamics (special case $f(t)=0$)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to Langevin dynamics (special case $f(t)=0$))</span></p>

If $\mathbf{f}(t)=0$, (4.1.6) becomes

$$d\bar{\mathbf{x}}(t)= -g^2(t)\nabla_{\mathbf{x}}\log p_t(\bar{\mathbf{x}}(t))dt + g(t),d\bar{\mathbf{w}}(t)$$

</div>

### Reparameterize time forward

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reparameterize time forward)</span></p>

Let $s=T-t$ (so $dt=-ds$) and rename Brownian motion so that $d\bar{\mathbf{w}}(t)=-d\mathbf{w}_s$.
Define $\bar{\mathbf{x}}_s := \bar{\mathbf{x}}(T-s)$ and $\pi_s := p_{T-s}$. Then:

$$d\bar{\mathbf{x}}_s = g^2(T-s)\nabla_{\mathbf{x}}\log \pi_s(\bar{\mathbf{x}}_s)ds + g(T-s),d\mathbf{w}_s$$

Now define a "temperature" schedule

$$\tau(s) := \tfrac12 g^2(T-s)$$

Then

$$
d\bar{\mathbf{x}}_s
= 2\tau(s)\nabla_{\mathbf{x}}\log \pi_s(\bar{\mathbf{x}}_s)\,ds

+ \sqrt{2\tau(s)},d\mathbf{w}_s,
$$

  which is exactly **Langevin form**, but with **time-varying temperature** $\tau(s)$ and a time-evolving target density $\pi_s$.

</div>

### Annealing intuition

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Annealing intuition)</span></p>

* Early in reverse time (near $t\approx T$, i.e. $s\approx 0$): $g(T-s)$ is typically larger → more noise → broad exploration.
* As you approach $t\to 0$ (i.e. $s\to T$): $g(T-s)$ decreases → noise weakens, score term dominates → trajectories concentrate near high-density (data-like) regions.

</div>

---

## Reverse-time SDE capabilities and learning

### Central role of the score

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Central role of the score)</span></p>

Define the score function:

$$\mathbf{s}(\mathbf{x},t) := \nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$

Once forward coefficients $\mathbf{f}$ and $g$ are fixed, **the score is the only unknown** needed to run the reverse SDE.

</div>

### Practical approach

The "oracle" score is not available, so we learn a neural net $\mathbf{s}_\phi(\mathbf{x},t)$ via **score matching** (later section referenced as 4.2.1). Plugging it into (4.1.6) yields a fully specified generative dynamics.

### Sampling statement

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sampling statement)</span></p>

Generation = solve the reverse-time SDE from $t=T$ to $t=0$:

* initialize $\mathbf{x}_T\sim p_{\text{prior}}$,
* integrate reverse dynamics using learned score,
* output $\mathbf{x}_0$ which should follow $p_{\text{data}}$ approximately, assuming $p_{\text{prior}}\approx p_T$.

</div>

---

## Minimal "memory hooks" (quick recall)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Minimal "memory hooks" (quick recall))</span></p>

* **Forward:** $d\mathbf{x}=\mathbf{f}(\mathbf{x},t)dt+g(t)d\mathbf{w}$, $\mathbf{x}(0)\sim p_{\text{data}}$.
* **Kernel:** $p_t(\mathbf{x}_t\mid \mathbf{x}_0)$ (often Gaussian if $\mathbf{f}(\mathbf{x},t)=f(t)\mathbf{x}$).
* **Marginal:** $p_t(\mathbf{x})=\int p_t(\mathbf{x}\mid\mathbf{x}_0)p_{\text{data}}(\mathbf{x}_0)d\mathbf{x}_0$.
* **Reverse:** $d\bar{\mathbf{x}}=[\mathbf{f}-g^2\nabla\log p_t]dt+g,d\bar{\mathbf{w}}$.
* **Key unknown:** the **score** $\nabla\log p_t$.
* **Langevin view:** reverse SDE looks like annealed Langevin with $\tau(s)=\tfrac12 g^2(T-s)$ when $f=0$.

</div>

## 4.1.4 Deterministic Process for Generation: Probability Flow ODE (PF-ODE)

### Motivation (Question 4.1.1)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation (Question 4.1.1))</span></p>

Forward diffusion is usually defined as an SDE that adds noise:

* It is natural to ask: **do we *have* to sample (generate) with the reverse-time SDE**, or can we generate deterministically?

Key idea: **No, SDE sampling is not necessary.** There exists a **deterministic ODE** whose solutions have the **same marginal distributions** as the forward SDE at every time $t$.

</div>

---

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probability Flow ODE (PF-ODE))</span></p>

Given the forward SDE $d\mathbf{x}(t)=\mathbf{f}(\mathbf{x}(t),t),dt + g(t),d\mathbf{w}(t)$, Song et al. introduce the **Probability Flow ODE**:

$$
\frac{d\tilde{\mathbf{x}}(t)}{dt}
=

\mathbf{f}(\tilde{\mathbf{x}}(t),t)
-\frac{1}{2}g(t)^2 \nabla_{\mathbf{x}}\log p_t(\tilde{\mathbf{x}}(t)).
\tag{PF-ODE}
$$

**Important:** the PF-ODE drift is **not** obtained by "just removing noise."
The $\tfrac12$ factor is essential and comes from the **Fokker–Planck** matching principle.

</div>

---

### Sampling / generation with PF-ODE

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sampling / generation with PF-ODE)</span></p>

To generate data:

1. Sample an initial point from the terminal distribution (the "prior"):
   
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

</div>

---

### Advantages vs reverse-time SDE sampling

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advantages vs reverse-time SDE sampling)</span></p>

* **Bidirectional integration:** you can run the same ODE forward $0\to T$ or backward $T\to 0$, just changing the endpoint initial condition.
* **ODE solver ecosystem:** many mature, accurate, off-the-shelf numerical solvers exist for ODEs.

</div>

---

## 4.1.5 Matching Marginal Distributions: Forward/Reverse SDEs and PF-ODE

### High-level goal (Question 4.1.2)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(High-level goal (Question 4.1.2))</span></p>

Different stochastic/deterministic processes can yield the **same time-indexed marginals** $\lbrace p_t\rbrace_{t\in[0,T]}$.
What matters is constructing a process whose marginals match the target evolution—especially so that at $t=0$ we recover $p_{\text{data}}$.

</div>

---

### Figure intuition (Fig. 4.4)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Figure intuition (Fig. 4.4))</span></p>

The forward process gradually transforms an initial complicated distribution $p_0=p_{\text{data}}$ (e.g., a multi-modal mixture) into a simple terminal distribution $p_T \approx p_{\text{prior}}$ (often Gaussian-like).
This evolution of the marginal density $p_t$ is governed by the **Fokker–Planck equation**.

</div>

---

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.1.1 — Fokker–Planck ensures marginals align)</span></p>

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

</div>

---

### Consequence: PF-ODE and reverse-time SDE share the same marginals

#### (i) PF-ODE

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">((i) PF-ODE)</span></p>

$$\frac{d\tilde{\mathbf{x}}(t)}{dt}=\mathbf{v}(\tilde{\mathbf{x}}(t),t)$$

* If started from $\tilde{\mathbf{x}}(0)\sim p_0$ and run forward, then $\tilde{\mathbf{x}}(t)\sim p_t$.
* Equivalently, if started from $\tilde{\mathbf{x}}(T)\sim p_T$ and run backward, it also matches the same marginals.

</div>

#### (ii) Reverse-time SDE (stochastic sampler)


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">((ii) Reverse-time SDE (stochastic sampler))</span></p>

$$
d\bar{\mathbf{x}}(t)
=

\Big[\mathbf{f}(\bar{\mathbf{x}}(t),t)-g(t)^2\nabla_{\mathbf{x}}\log p_t(\bar{\mathbf{x}}(t))\Big],dt
+g(t),d\bar{\mathbf{w}}(t),
$$

initialized at $\bar{\mathbf{x}}(0)\sim p_T$, where $\bar{\mathbf{w}}(t)$ is a Wiener process in reverse time.

**Key point:** PF-ODE and reverse-time SDE differ at the *trajectory level* (deterministic vs stochastic), but are designed to be consistent with the **same family of marginals** governed by Fokker–Planck.

</div>

---

## Flow map view and "many conditionals, one marginal"

### PF-ODE flow map

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(PF-ODE flow map)</span></p>

Define the flow map $\Psi_{s\to t}:\mathbb{R}^D\to\mathbb{R}^D$ by "evolving the ODE from time $s$ to $t$":

$$
\Psi_{s\to t}(\mathbf{x}_s)
=

\mathbf{x}_s + \int_s^t \mathbf{v}(\mathbf{x}_\tau,\tau),d\tau.
\qquad\text{(4.1.9)}
$$

Under mild smoothness assumptions, $\Psi_{s\to t}$ is a **smooth bijection**.

</div>

---

### Pushforward density under the ODE

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pushforward density under the ODE)</span></p>

If $\mathbf{x}_0\sim p_{\text{data}}$ and $\mathbf{x}_t=\Psi_{0\to t}(\mathbf{x}_0)$, then the induced density at time $t$ is the pushforward:

$$
p_t^{\text{fwd}}(\mathbf{x}_t)
:=
\int \delta\left(\mathbf{x}_t-\Psi_{0\to t}(\mathbf{x}_0)\right),p_{\text{data}}(\mathbf{x}_0),d\mathbf{x}_0.
$$

The theorem ensures $p_t^{\text{fwd}}=p_t$, matching the forward SDE marginals.

</div>

---

### Non-uniqueness of conditionals $Q_t(\mathbf{x}_t\mid \mathbf{x}_0$)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-uniqueness of conditionals $Q_t(\mathbf{x}_t\mid \mathbf{x}_0$))</span></p>

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


**Interpretation:** many different dynamics (stochastic/deterministic/hybrid) can satisfy the same marginal evolution—what "selects" the right marginals is the **Fokker–Planck equation**.

</div>

---

## Observation 4.1.1: What really matters

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Observation 4.1.1: What really matters)</span></p>

* Multiple processes can produce the **same sequence of marginals** $\lbrace p_t\rbrace$.
* The crucial requirement is: **the process must satisfy the Fokker–Planck evolution** for the prescribed $p_t$.
* This gives significant flexibility in designing generative processes from $p_{\text{prior}}\to p_{\text{data}}$ (or the reverse).

</div>

---

## Compact cheat sheet (core equations to memorize)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Compact cheat sheet (core equations to memorize))</span></p>

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

</div>

## Score SDE: Training, Sampling, Inversion, and Likelihood

### 0) Setup and notation (what objects appear in these pages)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Setup and notation (what objects appear in these pages))</span></p>

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

</div>

---

## Training the score model

### 1.1 "Oracle" score matching objective (intractable target)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">("Oracle" score matching objective (intractable target))</span></p>

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

</div>

---

### 1.2 Denoising Score Matching (DSM) objective (tractable target)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Denoising Score Matching (DSM) objective (tractable target))</span></p>

To avoid the oracle score, use the **conditional** distribution of the forward process:

* sample a clean data point $\mathbf{x}\_0\sim p_{\text{data}}$
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

**Interpretation:** DSM is "regress the network output onto a known conditional score target."

</div>

---

### 1.3 What does DSM learn? (Proposition 4.2.1)

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(4.2.1 — Minimizer of DSM)</span></p>

The optimal function $\mathbf{s}^*$ satisfies

$$
\mathbf{s}^*(\mathbf{x}_t,t)
= \mathbb{E}_{\mathbf{x}_0\sim p(\mathbf{x}_0\mid \mathbf{x}_t)}
\big[\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)\big]
= \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t),
$$

for (almost) every $\mathbf{x}_t\sim p_t$ and $t\in[0,T]$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(What does DSM learn? (Proposition 4.2.1))</span></p>

**Why this is true (high-level):**

* For fixed $t$, DSM is a **least-squares regression problem** in the random variable $\mathbf{x}_t$.
* The minimizer of $\mathbb{E}\lvert h(\mathbf{x}_t)-Y\rvert^2$ is $h^*(\mathbf{x}_t)=\mathbb{E}[Y\mid \mathbf{x}_t]$.
* Here $Y=\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)$, so
  
  $$\mathbf{s}^*(\mathbf{x}_t,t)=\mathbb{E}[Y\mid \mathbf{x}_t]$$

* Then, using Bayes’ rule, that conditional expectation equals the **marginal** score $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)$.

**Takeaway:** DSM lets you train $\mathbf{s}_\phi$ using a tractable conditional target, yet the optimum corresponds to the true marginal score.

</div>

---

### 1.4 Practical training recipe (what you do in code)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical training recipe (what you do in code))</span></p>

For each SGD step:

1. Sample $t\sim p_{\text{time}}$.
2. Sample $\mathbf{x}\_0\sim p_{\text{data}}$.
3. Sample $\mathbf{x}_t\sim p_t(\mathbf{x}_t\mid \mathbf{x}_0)$ using the forward noising rule.
4. Compute the analytic target $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)$.
5. Minimize the weighted squared error with $\omega(t)$.

</div>

---

## Sampling and inference after training (Sec. 4.2.2)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Sampling and inference after training))</span></p>

Once trained, denote the learned score as

$$\mathbf{s}_{\phi^\star}(\mathbf{x},t)\approx \nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$

Now replace the oracle score in the **reverse-time SDE** and in the **probability flow ODE (PF-ODE)**.

A helpful visual intuition (Fig. 4.5): starting from $\mathbf{x}\_T\sim p_{\text{prior}}$, both:

* solving the reverse-time SDE (stochastic path),
* solving the PF-ODE (deterministic path),
  end near the data manifold at $t=0$ (if the score is accurate).

</div>

---

## Generation via the empirical reverse-time SDE

### 3.1 Empirical reverse-time SDE (Eq. 4.2.3)


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Empirical reverse-time SDE (Eq. 4.2.3))</span></p>

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

</div>

### 3.2 Euler–Maruyama discretization (Eq. 4.2.4)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Euler–Maruyama discretization (Eq. 4.2.4))</span></p>

To sample:

1. Draw $\mathbf{x}\_T\sim p_{\text{prior}}$.
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

</div>

---

## Generation via the empirical PF-ODE (probability flow ODE)

### 4.1 Empirical PF-ODE (Eq. 4.2.5)


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Empirical PF-ODE (Eq. 4.2.5))</span></p>

$$
\frac{d}{dt}\mathbf{x}^{\text{ODE}}_{\phi^\star}(t)
=

\mathbf{f}\big(\mathbf{x}^{\text{ODE}}_{\phi^\star}(t),t\big)
-\frac{1}{2}g^2(t),\mathbf{s}_{\phi^\star}\big(\mathbf{x}^{\text{ODE}}_{\phi^\star}(t),t\big).
$$


* Deterministic dynamics (no stochastic term).
* Defines a **continuous flow** that connects $p_{\text{prior}}$ and $p_{\text{data}}$.

Sampling procedure:

1. Draw $\mathbf{x}\_T\sim p_{\text{prior}}$.
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

</div>

### 4.2 Euler method update (Eq. 4.2.6)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Euler method update (Eq. 4.2.6))</span></p>

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

The resulting distribution $p^{\text{ODE}}\_{\phi^\star}(\cdot;0)$ should approximate $p_{\text{data}}$.

</div>

---

## Core insight: generation = solving an ODE/SDE (Insight 4.2.1)

> Sampling from diffusion models is fundamentally equivalent to solving a corresponding **reverse-time SDE** or **probability flow ODE**.

**Implication:** Sampling can be slow because numerical solvers are iterative and may require many function evaluations (note: typical diffusion setups can use $\sim 1000$ evaluations).

---

## Inversion with PF-ODE (encoder viewpoint)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Inversion with PF-ODE (encoder viewpoint))</span></p>

Unlike SDE sampling, the PF-ODE can be solved both:

* **forward**: $0 \to T$,
* **backward**: $T \to 0$,

because it’s a deterministic ODE (under standard well-posedness assumptions).

**Forward solve interpretation:**
Solving PF-ODE forward maps $\mathbf{x}_0$ to a noisy latent $\mathbf{x}(T)$. This acts like an **encoder**, and enables applications like controllable generation / translation / editing.

</div>

---

## Exact log-likelihood via PF-ODE (continuous normalizing flow view)

### 7.1 Define the velocity field

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Define the velocity field)</span></p>

Treat the PF-ODE dynamics as a (Neural ODE–style) flow with velocity

$$
\mathbf{v}_{\phi^\star}(\mathbf{x},t)
:=
\mathbf{f}(\mathbf{x},t)-\frac{1}{2}g^2(t)\mathbf{s}_{\phi^\star}(\mathbf{x},t)
$$

</div>

### 7.2 Log-density evolution along the flow

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Log-density evolution along the flow)</span></p>

Along the PF-ODE trajectory $\lbrace \mathbf{x}^{\text{ODE}}_{\phi^\star}(t)\rbrace$,

$$
\frac{d}{dt}\log p^{\text{ODE}}_{\phi^\star} \Big(\mathbf{x}^{\text{ODE}}_{\phi^\star}(t),t\Big)
=

-\nabla\cdot \mathbf{v}_{\phi^\star} \Big(\mathbf{x}^{\text{ODE}}_{\phi^\star}(t),t\Big),
$$

where $\nabla\cdot \mathbf{v}$ is the divergence w.r.t. $\mathbf{x}$.

</div>

### 7.3 Augmented ODE to compute likelihood (Eq. 4.2.7)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Augmented ODE to compute likelihood (Eq. 4.2.7))</span></p>

To compute likelihood for $\mathbf{x}\_0\sim p_{\text{data}}$, integrate forward from $t=0$ to $t=T$:

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

</div>

---


## 4.3 Instantiations of SDEs (Score-SDE framework)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Instantiations of SDEs (Score-SDE framework))</span></p>

We consider the **forward SDE** (diffusion / noising process)

$$\mathrm{d}\mathbf{x}(t)= f(\mathbf{x},t),\mathrm{d}t + g(t),\mathrm{d}\mathbf{w}(t)$$

where $\mathbf{w}(t)$ is a $D$-dimensional Wiener process (independent coordinates). Song et al. categorize forward SDEs by how the **variance evolves** over time. Here we focus on two widely used cases:

* **VE SDE** = *Variance Exploding*
* **VP SDE** = *Variance Preserving*

A key object is the **perturbation kernel** (transition density)

$$p_t(\mathbf{x}_t\mid \mathbf{x}_0)$$

which tells you what distribution you get after noising clean data $\mathbf{x}\_0$ up to time $t$. This kernel is what you sample from during training (e.g., for denoising/score matching), and it also determines a natural **prior** $p_{\text{prior}} = p_T(\mathbf{x}\_T)$ used for generation.

</div>

---

## Table 4.1 — Summary (VE vs VP)

### VE SDE

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(VE SDE summary)</span></p>

* **Drift:** $f(\mathbf{x},t)=0$
* **Diffusion:** $g(t)=\sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}}$
* **SDE:**
  
  $$\mathrm{d}\mathbf{x}(t)= g(t),\mathrm{d}\mathbf{w}(t)$$
  
* **Perturbation kernel:**
  
  $$p_t(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N} \Big(\mathbf{x}_t;\mathbf{x}_0,;(\sigma^2(t)-\sigma^2(0))\mathbf{I}\Big)$$
  
* **Prior (typical):**
  
  $$p_{\text{prior}}=\mathcal{N}(\mathbf{0},\sigma^2(T)\mathbf{I})$$

</div>

### VP SDE

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(VP SDE summary)</span></p>

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

</div>

---

## 4.3.1 VE SDE (Variance Exploding)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(VE SDE)</span></p>

* Drift term is **zero**:

  $$f(\mathbf{x},t)=0$$

* Diffusion is controlled by a variance schedule $\sigma(t)$:

  $$g(t)=\sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}}$$

  So the forward SDE is

  $$\mathrm{d}\mathbf{x}(t)=\sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}},\mathrm{d}\mathbf{w}(t)$$

</div>

### Perturbation kernel (what noising does)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Perturbation kernel (what noising does))</span></p>

Because there is no drift, the process does not "shrink" $\mathbf{x}$; it only adds Gaussian noise:

$$
p_t(\mathbf{x}_t\mid \mathbf{x}_0)=
\mathcal{N} \Big(\mathbf{x}_t;\mathbf{x}_0,;(\sigma^2(t)-\sigma^2(0))\mathbf{I}\Big).
$$

</div>

### Prior choice

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Prior choice)</span></p>

Assume $\sigma(t)$ is increasing on $[0,T]$ and $\sigma^2(T)\gg\sigma^2(0)$. Then a natural prior is:

$$p_{\text{prior}}:=\mathcal{N}(\mathbf{0},\sigma^2(T)\mathbf{I})$$

</div>

### Typical instance: NCSN (discretized VE)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Typical instance: NCSN (discretized VE))</span></p>

A standard VE design uses a **geometric** schedule (for $t\in(0,1]$):

$$\sigma(t):=\sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{t}$$

so the variance levels form a geometric sequence. NCSN can be viewed as a discretization of this VE SDE.

**Intuition:** VE keeps the mean at $\mathbf{x}_0$ but steadily increases the noise scale—eventually the signal is drowned by large variance.

</div>

---

## 4.3.2 VP SDE (Variance Preserving)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(VP SDE)</span></p>

Let $\beta:[0,T]\to\mathbb{R}_{\ge 0}$ be a nonnegative "noise rate" schedule.

* Drift pulls $\mathbf{x}(t)$ toward zero:

  $$f(\mathbf{x},t)= -\tfrac12\beta(t)\mathbf{x}$$

* Diffusion injects noise:

  $$g(t)=\sqrt{\beta(t)}$$

Forward SDE:

$$\mathrm{d}\mathbf{x}(t)= -\tfrac12 \beta(t)\mathbf{x}(t),\mathrm{d}t + \sqrt{\beta(t)},\mathrm{d}\mathbf{w}(t)$$

</div>

### Perturbation kernel

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Perturbation kernel)</span></p>

Define

$$B(t):=\int_0^t\beta(s),\mathrm{d}s$$

Then

* mean:
  
  $$\mathbb{E}[\mathbf{x}_t\mid \mathbf{x}_0]=e^{-\frac12B(t)}\mathbf{x}_0$$
  
* covariance (isotropic):
  
  $$\mathrm{Cov}[\mathbf{x}_t\mid \mathbf{x}_0]=(1-e^{-B(t)})\mathbf{I}$$
  
  So
  
  $$p_t(\mathbf{x}_t\mid \mathbf{x}_0)= \mathcal{N}\Big(\mathbf{x}_t;;e^{-\frac12B(t)}\mathbf{x}_0,;(1-e^{-B(t)})\mathbf{I}\Big)$$

</div>

### Prior choice

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Prior choice)</span></p>

At large time (typical design makes $B(T)$ large), the mean vanishes and covariance approaches $\mathbf{I}$, hence:

$$p_{\text{prior}}:=\mathcal{N}(\mathbf{0},\mathbf{I})$$

</div>

### Note on computing scores

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Note on computing scores)</span></p>

Since $p_t(\mathbf{x}_t\mid \mathbf{x}_0)$ is Gaussian with known mean/covariance, its **score**

$$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid \mathbf{x}_0)$$

has a closed form (for isotropic covariance it’s proportional to $-(\mathbf{x}_t-\text{mean})$).

</div>

### Typical instance: DDPM (discretized VP)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Typical instance: DDPM (discretized VP))</span></p>

A classic VP schedule (for $t\in[0,1]$) is **linear**:

$$\beta(t):=\beta_{\min}+t(\beta_{\max}-\beta_{\min})$$

DDPM can be interpreted as a discretization of the VP SDE.

**Intuition:** VP simultaneously (i) shrinks the signal and (ii) adds noise so that the total variance stays controlled and ends near standard normal.

</div>

---

## 4.3.3 (Optional) How the perturbation kernel $p_t(\mathbf{x}_t\mid \mathbf{x}_0)$ is derived

### Linear-drift case ⇒ conditional Gaussian

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Linear-drift case ⇒ conditional Gaussian)</span></p>

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

</div>

### Moment ODEs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Moment ODEs)</span></p>

The mean and (scalar) variance satisfy:

$$
\frac{\mathrm{d}\mathbf{m}(t)}{\mathrm{d}t}=f(t)\mathbf{m}(t),\qquad
\frac{\mathrm{d}P(t)}{\mathrm{d}t}=2f(t)P(t)+g^2(t),
$$

with initial conditions $\mathbf{m}(0)=\mathbf{x}_0$, $P(0)=0$.

</div>

### Closed-form solution via integrating factor

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Closed-form solution via integrating factor)</span></p>

Define the exponential integrating factor

$$\mathcal{E}(s\to t):=\exp \left(\int_s^t f(u),\mathrm{d}u\right)$$

Then:

$$
\mathbf{m}(t)=\mathcal{E}(0\to t)\mathbf{x}_0,\qquad
P(t)=\int_0^t \mathcal{E}^2(s\to t),g^2(s),\mathrm{d}s.
$$

</div>

---

## Worked examples (transition kernels)

### VE SDE

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(VE SDE worked example)</span></p>

Here $f=0$, $g(t)=\sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}}$.

* Mean ODE: $\frac{\mathrm{d}\mathbf{m}}{\mathrm{d}t}=0\Rightarrow \mathbf{m}(t)=\mathbf{x}_0$
* Variance ODE: $\frac{\mathrm{d}P}{\mathrm{d}t}=\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}\Rightarrow P(t)=\sigma^2(t)-\sigma^2(0)$

So

$$
p_t(\mathbf{x}_t\mid \mathbf{x}_0)=
\mathcal{N} \Big(\mathbf{x}_t;\mathbf{x}_0,;(\sigma^2(t)-\sigma^2(0))\mathbf{I}_D\Big).
$$

</div>

### VP SDE

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(VP SDE worked example)</span></p>

Here $f(t)=-\tfrac12\beta(t)$, $g(t)=\sqrt{\beta(t)}$, and $B(t)=\int_0^t\beta(s)ds$.

* Mean:
  
  $$\frac{\mathrm{d}\mathbf{m}}{\mathrm{d}t}=-\tfrac12\beta(t)\mathbf{m}(t) \Rightarrow  \mathbf{m}(t)=e^{-\frac12B(t)}\mathbf{x}_0$$
  
* Variance:
  
  $$\frac{\mathrm{d}P}{\mathrm{d}t}=2f(t)P+g^2(t)= -\beta(t)P(t)+\beta(t)$$
  
  Multiply by $e^{B(t)}$ (integrating factor):
  
  $$\frac{\mathrm{d}}{\mathrm{d}t}\big(P(t)e^{B(t)}\big)=\beta(t)e^{B(t)} \Rightarrow P(t)=1-e^{-B(t)}$$

Final:

$$p_t(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N} \Big(\mathbf{x}_t;;e^{-\frac12B(t)}\mathbf{x}_0,;(1-e^{-B(t)})\mathbf{I}_D\Big)$$

</div>

---

## Mental model: VE vs VP (what to remember)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Mental model: VE vs VP (what to remember))</span></p>

* **VE:** mean stays $\mathbf{x}_0$; variance grows like $\sigma^2(t)$ (eventually huge).
* **VP:** mean decays to $0$; variance rises but is capped to approach $1$ (standard normal), giving a clean $\mathcal{N}(0,I)$ prior.

</div>

### Rethinking Forward Kernels in Score-Based and Variational Diffusion Models

## Why "rethink" the forward kernel?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why "rethink" the forward kernel?)</span></p>

Diffusion/Score-SDE models are often introduced via **incremental** forward transitions:

* **DDPM (discrete):** $p(x_t\mid x_{t-\Delta t})$
* **Score SDE (continuous):** an SDE that implies infinitesimal transitions

But in practice (especially in the common losses), what matters most is the **accumulated / marginal perturbation kernel from data**:

$$p_t(x_t\mid x_0)$$

Both DDPM and Score-SDE ultimately rely on this kernel:

* DDPM: by recursive composition of step kernels
* Score-SDE: by solving ODEs (for moments) induced by the SDE

**Key message:** defining $p_t(x_t\mid x_0)$ directly is often **cleaner**, **more interpretable**, and aligns naturally with loss/prior design (e.g., what happens as $t\to T$).

</div>

---

## A general affine forward perturbation process $p_t(x_t\mid x_0)$

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(General affine forward perturbation (Eq. 4.4.1))</span></p>

Assume a Gaussian perturbation kernel:

$$p_t(x_t\mid x_0) := \mathcal N \big(x_t;\ \alpha_t x_0,\ \sigma_t^2 I\big)$$

where $x_0\sim p_{\text{data}}$, and $\alpha_t,\sigma_t\ge 0$ for $t\in[0,T]$, typically satisfying:

* $\alpha_t>0$ and $\sigma_t>0$ for $t\in(0,T]$ (allowing $\sigma_0=0$)
* usually $\alpha_0=1,\ \sigma_0=0$

**Sampling form:**

$$x_t = \alpha_t x_0 + \sigma_t \varepsilon,\qquad \varepsilon\sim \mathcal N(0,I)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Common forward kernel types)</span></p>

This single form subsumes common "forward types":

* **VE (NCSN) kernel:** $\alpha_t\equiv 1,\ \sigma_T\gg 1$
* **VP (DDPM) kernel:** $\alpha_t := \sqrt{1-\sigma_t^2}$ so that $\alpha_t^2+\sigma_t^2=1$
* **FM kernel:** $\alpha_t=1-t,\ \sigma_t=t$ (linear interpolation between $x_0$ and noise)

</div>

---

## Connection to Score SDE: marginal kernel $\Longleftrightarrow$ linear SDE

### Score-SDE forward process (linear-in-$x$ form)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Score-SDE forward process (linear-in-$x$ form))</span></p>

If $p_t(x_t\mid x_0)$ has the affine Gaussian form above, it corresponds to an SDE

$$dx(t)= f(t),x(t),dt + g(t),dw(t)$$

where $w(t)$ is Brownian motion (so $dw(t)$ is "Gaussian noise" with variance $\propto dt$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(4.4.1 — Forward perturbation kernel $\Leftrightarrow$ linear SDE)</span></p>

Define

$$\lambda_t := \log\frac{\alpha_t}{\sigma_t}\quad (t\in(0,T])$$

Given $x_t=\alpha_t x_0+\sigma_t\varepsilon$, the corresponding SDE coefficients are:

$$f(t)=\frac{d}{dt}\log \alpha_t$$

$$
g^2(t)=\frac{d}{dt}\sigma_t^2 - 2\frac{d}{dt}\log\alpha_t\ \sigma_t^2
= -2\sigma_t^2\frac{d}{dt}\lambda_t
$$

Conversely, any linear SDE whose conditionals are $\mathcal N(\alpha_t x_0,\sigma_t^2 I)$ must satisfy these relations.

</div>

#### Proof idea (what’s happening)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof idea (what’s happening))</span></p>

For a linear SDE, the conditional mean $m(t)$ and covariance $P(t)$ satisfy ODEs:

* $m'(t)= f(t),m(t)$
* $P'(t)=2f(t)P(t)+g^2(t)I$

Matching $m(t)=\alpha_t x_0$ and $P(t)=\sigma_t^2 I$ yields the formulas above.

</div>

### Observation 4.4.1

> Defining $p_t(x_t\mid x_0)$ is **equivalent** to specifying the linear SDE coefficients $f(t)$ and $g(t)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Observation 4.4.1)</span></p>

So you can design the forward process **either** by:

* choosing $\alpha_t,\sigma_t$ directly (marginal view), **or**
* choosing $f,g$ (SDE view)

</div>

---

## Terminal prior and why "exact Gaussian prior at finite time" can be pathological

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Terminal prior and why "exact Gaussian prior at finite time" can be pathological)</span></p>

To exactly match a Gaussian prior at terminal time $T$, you’d like the process to **forget $x_0$**:

* require $\alpha_T = 0$
* and set $\sigma_T^2$ to the desired prior variance

But in the SDE formulation,

$$\alpha_t=\exp \left(\int_0^t f(u),du\right)$$

To force ($\alpha_T=0$ at finite $T$, you need

$$\int_0^T f(u),du = -\infty$$

meaning the drift $f(t)$ must contract "infinitely fast" near $T$. At the same time, maintaining the prescribed variance forces the diffusion to blow up; the text notes this is reflected by

$$g^2(t)=\sigma_t^{2,\prime}-2\frac{\alpha_t'}{\alpha_t}\sigma_t^2 \to \infty\quad \text{as }t\to T$$

**Practical takeaway:** if $f$ and $g$ stay bounded on $[0,T]$, then $\alpha_T>0$ and some dependence on $x_0$ remains; the Gaussian prior is then reached only **asymptotically** (e.g., in the limit $t\to T$ without exact attainment, or on an infinite horizon with reparameterization).

</div>

---

## Connection to variational diffusion (DDPM/VDM): Bayes rule and reverse kernels

### Core DDPM identity (Eq. 4.4.3)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Core DDPM identity (Eq. 4.4.3))</span></p>

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

The section’s theme: even if DDPM starts from incremental kernels, $p_t(x_t\mid x_0)$ is often the clearer "primary object."

</div>

---

## Closed-form reverse conditional transitions for the general affine kernel

Let $0\le t < s \le T$.

### Useful "between-time" parameters

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Useful "between-time" parameters)</span></p>

Define

$$
\alpha_{s\mid t} := \frac{\alpha_s}{\alpha_t},\qquad
\sigma_{s\mid t}^2 := \sigma_s^2 - \alpha_{s\mid t}^2\sigma_t^2.
$$

</div>

### Forward transition between noisy times (Eq. 4.4.5)

$$p(x_s\mid x_t)=\mathcal N \big(x_s;\ \alpha_{s\mid t}x_t,\ \sigma_{s\mid t}^2 I\big)$$

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(4.4.2 — Reverse conditional transition kernels)</span></p>

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

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forward transition between noisy times (Eq. 4.4.5))</span></p>

**Interpretation:** the posterior mean is a **weighted blend** of:

* the later noisy sample $x_s$
* the clean conditioning variable $x$ (usually $x_0$)

Weights depend entirely on the noise schedule $(\alpha,\sigma)$.

</div>

---

## Reverse model parameterization (x-prediction and ε-prediction)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reverse model parameterization (x-prediction and ε-prediction))</span></p>

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

</div>

---

## Diffusion loss becomes a weighted regression loss (Eq. 4.4.7)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Diffusion loss becomes a weighted regression loss (Eq. 4.4.7))</span></p>

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

</div>

---

## Continuous-time limit: VDM objective (Kingma et al., 2021)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(VDM objective)</span></p>

Kingma et al. study the limit $t\to s$ of the weighted regression term, yielding:

$$
\mathcal L^{\infty}_{\mathrm{VDM}}(x_0)
=

-\frac12,\mathbb E_{s,\ \varepsilon\sim\mathcal N(0,I)}
\Big[\mathrm{SNR}'(s)\ \lvert x_0-x_\phi(x_s,s)\rvert_2^2\Big].
$$

Typically $\mathrm{SNR}(s)$ decreases with $s$, so $\mathrm{SNR}'(s)<0$, making the overall weight $-\mathrm{SNR}'(s)$ positive.

</div>

This perspective also suggests a **learnable noise schedule** via learning $\mathrm{SNR}(s)$ (though extensions are beyond the shown excerpt).

---

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Generalized DDPM sampling step (Eq. 4.4.8))</span></p>

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

</div>

---

## "Cheat sheet" summary (what to remember)

### Forward (marginal) design

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Forward (marginal) design)</span></p>

* Pick $\alpha_t,\sigma_t$ $\implies$ defines $p_t(x_t\mid x_0)=\mathcal N(\alpha_t x_0,\sigma_t^2 I)$
* Sample: $x_t=\alpha_t x_0+\sigma_t\varepsilon$

</div>

### Convert to SDE (linear)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Convert to SDE (linear))</span></p>

* $f(t)=\frac{d}{dt}\log\alpha_t$
* $g^2(t)=\sigma_t^{2,\prime}-2(\log\alpha_t)'\sigma_t^2=-2\sigma_t^2\lambda_t'$, $\lambda_t=\log(\alpha_t/\sigma_t)$

</div>

### Between-time forward kernel

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Between-time forward kernel)</span></p>

* $\alpha_{s\mid t}=\alpha_s/\alpha_t$
* $\sigma_{s\mid t}^2=\sigma_s^2-\alpha_{s\mid t}^2\sigma_t^2$
* $p(x_s\mid x_t)=\mathcal N(\alpha_{s\mid t}x_t,\sigma_{s\mid t}^2I)$

</div>

### Reverse posterior + model

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Reverse posterior + model)</span></p>

* True posterior: $p(x_t\mid x_s,x)$ Gaussian with mean/var in Lemma 4.4.2
* Model replaces $x$ by $x_\phi(x_s,s)$
* KL term $\implies$ weighted regression: $\frac12(\mathrm{SNR}(t)-\mathrm{SNR}(s))\lvert x_0-x_\phi\rvert^2$

</div>

---


### Fokker–Planck Equation and Reverse-Time SDEs

## Setup and notation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Setup and notation)</span></p>

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

</div>

---

## 4.5.1 Fokker–Planck from marginalizing transition kernels

### Step 1: Chapman–Kolmogorov / marginalization

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 1: Chapman–Kolmogorov / marginalization)</span></p>

Using the Markov property,

$$
p_{t+\Delta t}(\mathbf{x})
= \int p(\mathbf{x}\mid \mathbf{y}),p_t(\mathbf{y}),d\mathbf{y}
= \int \mathcal{N} \Big(\mathbf{x};\ \mathbf{y}+\mathbf{f}(\mathbf{y},t)\Delta t,\ g^2(t)\Delta t,\mathbf{I}\Big),p_t(\mathbf{y}),d\mathbf{y}
$$

</div>

### Step 2: Change of variables to center the Gaussian

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Step 2: Change of variables to center the Gaussian)</span></p>

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


> The bracketed combination is exactly the "drift acting on density" term:
> 
> $$\mathbf{f}\cdot\nabla p + (\nabla\cdot \mathbf{f})p = \nabla\cdot(\mathbf{f}p)$$
> 

</div>

### Step 3: Taylor–Gaussian smoothing formula

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 3: Taylor–Gaussian smoothing formula)</span></p>

For smooth $\phi:\mathbb{R}^D\to\mathbb{R}$ and $\sigma^2>0$, with $\mathbf{z}\sim \mathcal{N}(0,\mathbf{I})$,

$$
\int \mathcal{N}(\mathbf{x};\mathbf{u},\sigma^2\mathbf{I}),\phi(\mathbf{u}),d\mathbf{u}
= \mathbb{E}[\phi(\mathbf{x}+\sigma\mathbf{z})]
= \phi(\mathbf{x}) + \frac{\sigma^2}{2}\Delta_{\mathbf{x}}\phi(\mathbf{x}) + \mathcal{O}(\sigma^4).
$$

This comes from Taylor expanding $\phi(\mathbf{x}+\sigma\mathbf{z})$ and using
$\mathbb{E}[\mathbf{z}]=0$, $\mathbb{E}[\mathbf{z}\mathbf{z}^\top]=\mathbf{I}$.

Here $\sigma^2 = g^2(t)\Delta t$.

</div>

### Step 4: Keep terms up to $\mathcal{O}(\Delta t)$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 4: Keep terms up to $\mathcal{O}(\Delta t)$)</span></p>

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

</div>

### Step 5: Take $\Delta t\to 0$ ⇒ Fokker–Planck

Divide by $\Delta t$ and let $\Delta t\to 0$:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fokker–Planck equation)</span></p>

$$
\partial_t p_t(\mathbf{x})
=

-\nabla_{\mathbf{x}}\cdot\big(\mathbf{f}(\mathbf{x},t),p_t(\mathbf{x})\big)
+\frac{g^2(t)}{2},\Delta_{\mathbf{x}}p_t(\mathbf{x})
$$

(For isotropic diffusion $g(t)\mathbf{I}$.)

</div>

**Interpretation (useful intuition):** this is a conservation/continuity equation for probability, where drift transports mass and diffusion spreads it.

---

## 4.5.2 Why the reverse-time SDE has a score term (Bayes-rule derivation)

Goal: find the **reverse-time transition** $p(\mathbf{x}_t \mid \mathbf{x}_{t+\Delta t})$ from the forward kernel, then take $\Delta t\to 0$.

### Step 1: Bayes rule for the reverse kernel


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 1: Bayes rule for the reverse kernel)</span></p>

$$
p(\mathbf{x}_t\mid \mathbf{x}_{t+\Delta t})
=

p(\mathbf{x}_{t+\Delta t}\mid \mathbf{x}_t),
\frac{p_t(\mathbf{x}_t)}{p_{t+\Delta t}(\mathbf{x}_{t+\Delta t})}
=

p(\mathbf{x}_{t+\Delta t}\mid \mathbf{x}_t),
\exp \Big(\log p_t(\mathbf{x}_t)-\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t})\Big).
$$

</div>

### Step 2: First-order Taylor expansion of the log-density term

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 2: First-order Taylor expansion of the log-density term)</span></p>

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


A key scaling fact for diffusions: 

$$\mathbb{E}\lvert\mathbf{x}_{t+\Delta t}-\mathbf{x}_t\rvert_2^2=\mathcal{O}(\Delta t)$$

so the remainder is $\mathcal{O}(\Delta t^2)$ in expectation (hence negligible at first order).

</div>

### Step 3: Substitute and complete the square

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 3: Substitute and complete the square)</span></p>

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

</div>

### Step 4: Replace $(\mathbf{x}_t,t)$ by $(\mathbf{x}_{t+\Delta t},t+\Delta t)$ (smoothness)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 4: Replace $(\mathbf{x}_t,t)$ by $(\mathbf{x}_{t+\Delta t},t+\Delta t)$ ...)</span></p>

Under smoothness,

$$\mathbf{f}(\mathbf{x}_t,t)\approx \mathbf{f}(\mathbf{x}_{t+\Delta t},t+\Delta t)$$

$$g(t)\approx g(t+\Delta t)$$

$$\nabla\log p_t(\mathbf{x}_t)\approx \nabla\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) =:\mathbf{s}(\mathbf{x}_{t+\Delta t},t+\Delta t)$$

where $\mathbf{s}(\mathbf{x},t)$ is the **score**.

So the reverse kernel says:

* **Mean step backward** is "forward drift" minus a **score correction** $g^2,\mathbf{s}$
* **Covariance** is still $g^2\Delta t,\mathbf{I}$

</div>

### Step 5: Continuous-time limit ⇒ reverse-time SDE

Heuristically, as $\Delta t\to 0$, the reverse-time process satisfies:

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Reverse-time SDE)</span></p>

$$
d\mathbf{x}_t
=

\big[\mathbf{f}(\mathbf{x}_t,t)-g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t)\big]dt
+
g(t)d\bar{\mathbf{w}}_t
$$

where $\bar{\mathbf{w}}_t$ is a Brownian motion in reverse time (and the process is run with time decreasing from $T$ to $0$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 5: Continuous-time limit ⇒ reverse-time SDE)</span></p>

**Intuition:** the score term points toward higher-density regions of $p_t$, so when you run time backward it acts like a *denoising drift* that counteracts the forward diffusion.

**Practical link (diffusion/score models):** if you learn 

$$\mathbf{s}_\theta(\mathbf{x},t)\approx \nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$

you can sample by simulating the reverse-time SDE from noise (large $t$) back to data (small $t$).

</div>

---

## 4.6 Closing remarks (big picture takeaways)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Closing remarks (big picture takeaways))</span></p>

* **Unification:** DDPM-style discrete diffusion and NCSN/score-based models can be viewed as *discretizations of SDEs* with different choices of drift/volatility.
* **Reverse-time SDE is the generative engine:** it "reverses" the forward noising process. Crucially, its drift depends on one unknown object:
  
  $$\nabla_{\mathbf{x}}\log p_t(\mathbf{x}) \quad \text{(the score)}$$
  
  This explains why score learning is central.
* **Probability Flow ODE (PF-ODE):** a deterministic counterpart whose trajectories share the same marginals $\lbrace p_t\rbrace$ as the SDE; this equivalence rests on the Fokker–Planck equation.
* **Core implication:** generation $\approx$ solving a differential equation; training $\approx$ learning the vector field (score / velocity); sampling $\approx$ numerical integration.
* This PF-ODE viewpoint bridges toward **flow-based generative modeling** (Normalizing Flows, Neural ODEs) and motivates the transition to **Flow Matching**.

</div>

---

## Chapter 5: Flow-Based Perspective: From NFs to Flow Matching

The *change-of-variables formula*, a cornerstone of probability theory, takes on new life in modern generative modeling. While Score SDEs offer a differential equation framework to bridge data and prior distributions via the Fokker--Planck equation, this continuous evolution is, at its core, a dynamic form of the same fundamental principle.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Change-of-Variables Formula of Densities)</span></p>

Given an invertible transformation $\mathbf{f}$, the density of $\mathbf{x} = \mathbf{f}(\mathbf{z})$ where $\mathbf{z} \sim p_{\text{prior}}$ is:

$$p(\mathbf{x}) = p_{\text{prior}}(\mathbf{z}) \left\lvert \det \frac{\partial \mathbf{f}^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right\rvert, \quad \text{where } \mathbf{z} = \mathbf{f}^{-1}(\mathbf{x}). \qquad (5.0.1)$$

This formula unlocks exact, bidirectional transport of densities and samples when $\mathbf{f}$ is tractable, forming the very foundation of Normalizing Flows.

</div>

---

## 5.1 Flow-Based Models: Normalizing Flows and Neural ODEs

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Overview)</span></p>

**Normalizing Flows (NFs)** enable flexible and tractable probability density estimation by applying a series of invertible transformations to a simple base distribution. **Neural Ordinary Differential Equations (NODEs)** extend this framework to continuous time, where the transformation is governed by an ODE.

</div>

### 5.1.1 Normalizing Flows

NFs model a complex data distribution $p_{\text{data}}(\mathbf{x})$ by transforming a simple prior $p_{\text{prior}}(\mathbf{z})$ (e.g., standard Gaussian $\mathcal{N}(\mathbf{0}, \mathbf{I})$) via an invertible mapping

$$\mathbf{f}_\phi : \mathbb{R}^D \to \mathbb{R}^D,$$

with $\mathbf{x} = \mathbf{f}_\phi(\mathbf{z})$ and $\mathbf{z} \sim p_{\text{prior}}$. Here, $\mathbf{x}$ and $\mathbf{z}$ share the same dimension.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(NF Model Likelihood)</span></p>

Using the change-of-variables formula (Equation 5.0.1), the model likelihood is:

$$\log p_\phi(\mathbf{x}) = \log p_{\text{prior}}(\mathbf{z}) + \log \left\lvert \det \frac{\partial \mathbf{f}_\phi^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right\rvert. \qquad (5.1.1)$$

**Training Objective.** Parameters $\phi$ are learned by maximizing the likelihood over data:

$$\mathcal{L}_{\text{NF}}(\phi) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log p_\phi(\mathbf{x}) \right]. \qquad (5.1.2)$$

Computing the Jacobian determinant in Equation (5.1.1) can be costly, scaling as $\mathcal{O}(D^3)$ in general.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Constructing Invertible Transformations)</span></p>

A single complex invertible network can be expensive due to its Jacobian determinant. Conversely, simple transforms (e.g., linear) are efficient but lack expressivity.

To balance this, NFs employ a sequence of $K$ trainable invertible mappings $\lbrace \mathbf{f}_k \rbrace_{k=0}^{L-1}$, each with efficiently computable Jacobians:

$$\mathbf{f}_\phi = \mathbf{f}_{L-1} \circ \mathbf{f}_{L-2} \circ \cdots \circ \mathbf{f}_0.$$

Samples transform via $\mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k)$ for $k = 0, \ldots, L-1$, with $\mathbf{z} = \mathbf{x}_0 \sim p_{\text{prior}}$ and $\mathbf{x} = \mathbf{x}_L$ corresponding to data. The resulting (log-)density is:

$$\log p_\phi(\mathbf{x}) = \log p_{\text{prior}}(\mathbf{x}_0) + \sum_{k=0}^{L-1} \log \left\lvert \det \frac{\partial \mathbf{f}_k}{\partial \mathbf{x}_k} \right\rvert^{-1}. \qquad (5.1.4)$$

</div>

#### Examples of Invertible Flows

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Planar Flows)</span></p>

Planar Flows apply a simple transformation:

$$\mathbf{f}(\mathbf{z}) = \mathbf{z} + \mathbf{u} h(\mathbf{w}^\top \mathbf{z} + b),$$

where $\mathbf{u}, \mathbf{w} \in \mathbb{R}^D$, $b \in \mathbb{R}$, and $h(\cdot)$ is an activation. The Jacobian determinant is:

$$\left\lvert 1 + \mathbf{u}^\top h'(\mathbf{w}^\top \mathbf{z} + b) \mathbf{w} \right\rvert.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Residual Flows)</span></p>

Define the transform $\mathbf{f}$ as:

$$\mathbf{f}(\mathbf{z}) = \mathbf{z} + \mathbf{v}(\mathbf{z}), \qquad (5.1.5)$$

with $\mathbf{v}$ contractive (Lipschitz constant $< 1$). This ensures invertibility via the Banach fixed-point theorem. The log-determinant of the Jacobian reduces to a trace expansion:

$$\log \left\lvert \det \frac{\partial \mathbf{f}(\mathbf{z})}{\partial \mathbf{z}} \right\rvert = \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k} \operatorname{Tr}\left( \left( \frac{\partial \mathbf{v}(\mathbf{z})}{\partial \mathbf{z}} \right)^k \right), \qquad (5.1.6)$$

making evaluation efficient via trace estimators (Hutchinson, 1989).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sampling and Inference)</span></p>

Sampling from NFs is straightforward: draw $\mathbf{x}_0 \sim p_{\text{prior}}$ and compute $\mathbf{x} = \mathbf{f}_\phi(\mathbf{x}_0)$. Exact likelihoods are obtained from Equation (5.1.4).

</div>

---

### 5.1.2 Neural ODEs

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(From Discrete-Time NFs to Continuous-Time NFs (Neural ODEs))</span></p>

NFs are typically formulated as a sequence of $L$ discrete, invertible transformations. Viewed through Equation (5.1.3) and the "Residual Flow" formulation in Equation (5.1.5), each layer can be written as:

$$\mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k) := \mathbf{x}_k + \mathbf{v}_{\phi_k}(\mathbf{x}_k, k),$$

where $\mathbf{v}_{\phi_k}(\cdot, k)$ is a layer-dependent velocity field parameterized by neural networks. This is the **Euler discretization** of a continuous-time ODE. In the limit of infinite layers and vanishing step size ($\Delta t \to 0$), the discrete NFs converge to a continuous model: **Neural ODEs (NODEs)**, also known as **Continuous Normalizing Flows (CNFs)**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Neural ODE)</span></p>

A Neural ODE defines a continuous transformation through:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{v}_\phi(\mathbf{x}(t), t), \quad t \in [0, T] \qquad (5.1.7)$$

where:
* $\mathbf{x}(t) \in \mathbb{R}^D$ is the state at time $t$; we sometimes write $\mathbf{x}_t$ for brevity.
* $\mathbf{v}_\phi(\mathbf{x}(t), t)$ is a neural network parameterized by $\phi$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Goal of NODE)</span></p>

Starting from the initial condition $\mathbf{x}(0) \sim p_{\text{prior}}$, the ODE evolves the state continuously over time, inducing a family of marginal distributions $p_\phi(\mathbf{x}_t, t)$ (similar to PF-ODEs). The goal is to learn the neural vector field $\mathbf{v}_\phi$, which intuitively represents a velocity that transports points along continuous trajectories in data space. By learning this velocity, the terminal distribution at $t=0$ matches the target distribution $p_{\text{data}}(\cdot)$. This continuous transformation unifies discrete normalizing flows and neural ODEs within a single framework.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(5.1.1: Instantaneous Change of Variables)</span></p>

Let $\mathbf{z}(t)$ be a continuous random process with time-dependent density $p(\mathbf{z}(t), t)$, and suppose it evolves according to the ODE

$$\frac{\mathrm{d}\mathbf{z}(t)}{\mathrm{d}t} = \mathbf{F}(\mathbf{z}(t), t).$$

Assuming $\mathbf{F}$ is uniformly Lipschitz in $\mathbf{z}$ and continuous in $t$, the time derivative of the log-density satisfies:

$$\frac{\partial \log p(\mathbf{z}(t), t)}{\partial t} = -\nabla_\mathbf{z} \cdot \mathbf{F}(\mathbf{z}(t), t). \qquad (5.1.9)$$

This is the *Instantaneous Change-of-Variables Formula* (a special case of the Fokker--Planck equation, specifically its deterministic form known as the **Continuity Equation**). It can also be interpreted as the continuous time limit of Equation (5.1.4).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to Discrete-Time Formula)</span></p>

The NODE likelihood in Equation (5.1.8),

$$\log p_\phi(\mathbf{x}(T), T) = \log p_{\text{prior}}(\mathbf{x}(0), 0) - \int_0^T \nabla_\mathbf{x} \cdot \mathbf{v}_\phi(\mathbf{x}(t), t) \, \mathrm{d}t,$$

can be seen as the continuous-time analogue of the discrete normalizing flow formulation in Equation (5.1.4):

$$\log p_\phi(\mathbf{x}_L) = \log p_{\text{prior}}(\mathbf{x}_0) - \sum_{k=0}^{L-1} \log \left\lvert \det \frac{\partial \mathbf{f}_k}{\partial \mathbf{x}_k} \right\rvert.$$

The integral mirrors the summation, and the trace operator replaces the log-determinant (as discussed in Equation (5.1.6)). The identity $\operatorname{Tr}\left(\frac{\partial \mathbf{F}}{\partial \mathbf{z}(t)}\right) = \nabla_\mathbf{z} \cdot \mathbf{F}$ holds for any vector field $\mathbf{F}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Training NODEs)</span></p>

Based on Equation (5.1.8), NODEs learn a parameterized velocity field $\mathbf{v}_\phi$ such that the terminal distribution $p_\phi(\cdot, T) \approx p_{\text{data}}$, where trajectories evolve from latent variables $\mathbf{x}(0) \sim p_{\text{prior}}$ via the ODE flow. Training follows the MLE framework:

$$\mathcal{L}_{\text{NODE}}(\phi) := \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log p_\phi(\mathbf{x}, T) \right].$$

**Exact Log-Likelihood Computation.** To compute $\log p_\phi(\mathbf{x}, T)$ for a data point $\mathbf{x}$, we integrate the change-of-variables formula:

$$\log p_\phi(\mathbf{x}, T) = \log p_{\text{prior}}(\mathbf{z}(0)) - \int_0^T \nabla_\mathbf{z} \cdot \mathbf{v}_\phi(\mathbf{z}(t), t) \, \mathrm{d}t. \qquad (5.1.10)$$

Here, $\mathbf{z}(t)$ solves the ODE reversely from $t = T$ to $t = 0$ with $\mathbf{z}(T) = \mathbf{x}$. The prior term $\log p_{\text{prior}}(\mathbf{z}(0))$ is tractable for standard distributions. This enables exact likelihood evaluation by numerically solving the ODE.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gradient-Based Optimization and Inference with NODEs)</span></p>

**Gradient-Based Optimization.** Maximizing $\mathcal{L}_{\text{NODE}}$ requires backpropagation through the ODE solver. The adjoint sensitivity method computes gradients via an auxiliary ODE with $\mathcal{O}(1)$ memory complexity, but NODE training remains expensive due to numerical integration at each step.

**Inference with NODEs.** Sampling with a trained model $\mathbf{v}_{\phi^\star}$ proceeds by drawing $\mathbf{x}(0) \sim p_{\text{prior}}$ and integrating forward (by numerical solvers):

$$\mathbf{x}(T) = \mathbf{x}(0) + \int_0^T \mathbf{v}_{\phi^\star}(\mathbf{x}(t), t) \, \mathrm{d}t.$$

The terminal state $\mathbf{x}(T)$ approximates a sample from $p_{\text{data}}$. The divergence can be efficiently estimated using stochastic trace estimators, such as Hutchinson's estimator.

</div>

---

## 5.2 Flow Matching Framework

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Flow Matching Overview)</span></p>

Score SDEs and NODEs offer an alternative perspective on generative modeling: learning a continuous-time flow, either stochastic or deterministic, that transports a simple Gaussian prior sample $\boldsymbol{\epsilon} \sim p_{\text{prior}}$ to a data-like sample from $p_{\text{data}}$.

The **Flow Matching (FM)** framework builds on this idea, but generalizes it to learn a flow between two *arbitrary* fixed endpoint distributions: a source distribution $p_{\text{src}}$ and a target distribution $p_{\text{tgt}}$, both assumed to be easy to sample from. In this broader setup, the generation task becomes a special case where $p_{\text{src}}$ is a Gaussian prior and $p_{\text{tgt}}$ is the data distribution. When $p_{\text{src}}$ is Gaussian, we refer to this setting as **Gaussian Flow Matching**.

FM's core principle: learning a time-dependent vector field $\mathbf{v}_t(\mathbf{x}_t)$ whose associated ODE flow matches a predefined probability path $\lbrace p_t \rbrace_{t \in [0,1]}$ subject to the boundary conditions

$$p_0 = p_{\text{src}}, \quad p_1 = p_{\text{tgt}}.$$

</div>

### 5.2.1 Lesson from Score-Based Methods

We revisit the Score SDE framework using a slightly different but equivalent formulation to extract key insights that motivate the FM approach.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 1: Defining a Conditional Path and Its Marginal Densities)</span></p>

A diffusion model specifies a continuous-time family of densities $\lbrace p_t \rbrace_{t \in [0,1]}$ that transports a simple prior $p_{\text{prior}}$ (e.g., Gaussian) at $t = 1$, used as the source, to a target data distribution $p_{\text{data}}$ at $t = 0$:

$$p_1(\mathbf{x}_1) = p_{\text{prior}}(\mathbf{x}_1), \quad p_0(\mathbf{x}_0) = p_{\text{data}}(\mathbf{x}_0).$$

This path is implicitly defined via the forward conditional distribution

$$p_t(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I}), \quad \mathbf{x}_0 \sim p_{\text{data}} \qquad (5.2.1)$$

which induces the marginal density

$$p_t(\mathbf{x}_t) := \int p_t(\mathbf{x}_t \mid \mathbf{x}_0) p_{\text{data}}(\mathbf{x}_0) \, \mathrm{d}\mathbf{x}_0.$$

The increasing variance $\sigma_t^2$ of the conditional Gaussian drives the evolution of $p_t$ toward the Gaussian prior.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 2: Velocity Field)</span></p>

The time evolution of the marginal density $p_t$ is governed by a velocity field $\mathbf{v}_t : \mathbb{R}^D \to \mathbb{R}^D$, derived from the Fokker--Planck equation:

$$\mathbf{v}_t(\mathbf{x}) := f(t)\mathbf{x} - \tfrac{1}{2}g^2(t) \nabla_\mathbf{x} \log p_t(\mathbf{x}), \qquad (5.2.2)$$

which defines a deterministic particle flow through the PF-ODE:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \underbrace{f(t)\mathbf{x}(t) - \tfrac{1}{2}g^2(t) \nabla_\mathbf{x} \log p_t(\mathbf{x}(t))}_{\mathbf{v}_t(\mathbf{x}(t))}.$$

This ODE transports an initial random variable $\mathbf{x}(0) \sim p_{\text{data}}$ forward in time or $\mathbf{x}(1) \sim p_{\text{prior}}$ backward in time, such that the evolving marginal density of $\mathbf{x}(t)$ matches $p_t$ at every $t \in [0, 1]$.

The scalar functions $f(t)$ and $g(t)$ are determined by the coefficients of the associated forward SDE, or equivalently the Gaussian kernel parameters $\alpha_t$ and $\sigma_t$ defined in the conditional path.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 3: Learning via the Conditional Strategy)</span></p>

The goal is to approximate the oracle velocity field $\mathbf{v}_t(\mathbf{x}_t)$ using a neural network $\mathbf{s}_\phi(\mathbf{x}_t, t)$ trained via the expected squared error:

$$\mathcal{L}_{\text{SM}}(\phi) = \mathbb{E}_{t \sim \mathcal{U}[0,1], \mathbf{x}_t \sim p_t} \left[ \lVert \mathbf{s}_\phi(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \rVert^2 \right].$$

Since the marginal score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$ is inaccessible, we exploit the tractable conditional distribution to define the conditional velocity:

$$\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{x}_0) := f(t)\mathbf{x}_t - \tfrac{1}{2}g^2(t) \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0).$$

By the law of total expectation, the marginal score is recovered as

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot \mid \mathbf{x}_t)} \left[ \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) \right]. \qquad (5.2.3)$$

This justifies the surrogate training objective:

$$\mathcal{L}_{\text{SM}}(\phi) = \underbrace{\mathbb{E}_{t, \mathbf{x}_0 \sim p_{\text{data}}, \mathbf{x}_t \sim p_t(\cdot \mid \mathbf{x}_0)} \left[ \lVert \mathbf{s}_\phi(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) \rVert^2 \right]}_{\mathcal{L}_{\text{DSM}}(\phi)} + C,$$

where $C$ is a constant independent of $\phi$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Underlying Rule: The Fokker--Planck Equation)</span></p>

The marginal density $p_t$ evolves according to the Fokker--Planck equation:

$$\frac{\partial p_t(\mathbf{x})}{\partial t} + \nabla \cdot \left( \underbrace{\left( f(t)\mathbf{x} - \tfrac{1}{2}g^2(t) \nabla_\mathbf{x} \log p_t(\mathbf{x}) \right)}_{\mathbf{v}_t(\mathbf{x})} p_t(\mathbf{x}) \right) = 0.$$

This PDE ensures that the density given by the PF-ODE matches the marginal distribution of the forward SDE. Running the PF-ODE backward from $t = 1$ to $t = 0$, starting with $\mathbf{x}_1 \sim p_{\text{prior}}$, we obtain time-dependent densities through the pushforward formula:

$$p_t^{\text{rev}}(\mathbf{x}) = \int \delta\left(\mathbf{x} - \boldsymbol{\Psi}_{1 \to t}(\mathbf{x}_1)\right) p_{\text{prior}}(\mathbf{x}_1) \, \mathrm{d}\mathbf{x}_1. \qquad (5.2.4)$$

The Fokker--Planck equation ensures that the induced density path coincides with the same evolving density:

$$p_t^{\text{rev}} = p_t. \qquad (5.2.5)$$

In particular, $p_0^{\text{rev}} = p_0 = p_{\text{data}}$, thereby recovering the data distribution at time $t = 0$.

</div>

---

### 5.2.2 Flow Matching Framework

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(FM Framework Setup)</span></p>

The analysis in Section 5.2.1 reveals that diffusion models succeed by learning a velocity field (specifically, the score) that transports between distributions while satisfying boundary conditions. FM builds on this insight and extends it to learning continuous flows that transport samples between two *arbitrary* distributions, $p_{\text{src}}$ and $p_{\text{tgt}}$.

**Step 1: Defining a Conditional Path and Its Marginal Densities.** Consider arbitrary source and target probability distributions $p_{\text{src}}$ and $p_{\text{tgt}}$ on $\mathbb{R}^D$. We set:

$$p_0(\mathbf{x}) = p_{\text{src}}(\mathbf{x}), \quad p_1(\mathbf{x}) = p_{\text{tgt}}(\mathbf{x}). \qquad (5.2.6)$$

FM implicitly defines a continuous family of intermediate densities $\lbrace p_t \rbrace_{t \in [0,1]}$ interpolating between these endpoints. Each marginal $p_t$ is expressed via a latent variable $\mathbf{z}$ drawn from a known distribution $\pi(\mathbf{z})$ and a conditional distribution $p_t(\mathbf{x}_t \mid \mathbf{z})$:

$$p_t(\mathbf{x}_t) = \int p_t(\mathbf{x}_t \mid \mathbf{z}) \pi(\mathbf{z}) \, \mathrm{d}\mathbf{z}, \qquad (5.2.7)$$

with $(\pi(\mathbf{z}), \lbrace p_t(\cdot \mid \mathbf{z}) \rbrace)$ chosen to satisfy the boundary conditions in Equation (5.2.6).

Common choices for $\mathbf{z}$ include:
* **Two-sided conditioning:** $\mathbf{z} = (\mathbf{x}_0, \mathbf{x}_1) \sim p_{\text{src}}(\mathbf{x}_0) p_{\text{tgt}}(\mathbf{x}_1)$, where $\pi$ couples source and target distributions. This allows FM to define transport between arbitrary distributions.
* **One-sided conditioning:** $\mathbf{z} = \mathbf{x}_0$ or $\mathbf{z} = \mathbf{x}_1$. It especially recovers diffusion-like setups when the source distribution is chosen to be Gaussian.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 2: Velocity Field and Continuity Equation)</span></p>

The goal is to find a velocity field $\mathbf{v}_t(\mathbf{x})$ such that the induced ODE,

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{v}_t(\mathbf{x}(t)), \quad t \in [0, 1],$$

produces marginal distributions of $\mathbf{x}(t)$ that match with $p_t$ at each time $t$, whether integrating forward from $\mathbf{x}(0) \sim p_{\text{src}}$ or backward from $\mathbf{x}(1) \sim p_{\text{tgt}}$.

This requirement is captured by the **continuity equation**:

$$\frac{\partial p_t(\mathbf{x})}{\partial t} + \nabla \cdot (\mathbf{v}_t(\mathbf{x}) p_t(\mathbf{x})) = 0. \qquad (5.2.8)$$

Any velocity field $\mathbf{v}_t$ that satisfies Equation (5.2.8) ensures that the ODE flow transports samples in a way that exactly follows the prescribed $p_t$ (see Section 5.2.4 for details). Since Equation (5.2.8) is a scalar equation while $\mathbf{v}_t$ is a vector field in $\mathbb{R}^D$, the equation admits infinitely many solutions. For example, if $\mathbf{v}_t$ solves the equation, then so does

$$\mathbf{v}_t + \frac{1}{p_t} \tilde{\mathbf{v}}_t,$$

for any divergence-free vector field $\tilde{\mathbf{v}}_t$ (i.e., $\nabla \cdot \tilde{\mathbf{v}}_t = 0$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Step 3: Learning via the Conditional Strategy)</span></p>

The goal of FM training is to approximate the oracle velocity field $\mathbf{v}_t$ using a neural network $\mathbf{v}_\phi$, by minimizing the expected squared error:

$$\mathcal{L}_{\text{FM}}(\phi) = \mathbb{E}_{t, \mathbf{x}_t \sim p_t} \left[ \lVert \mathbf{v}_\phi(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t) \rVert^2 \right].$$

We refer to this neural network parameterization as **v-prediction** (velocity prediction), which aims to learn the ODE drift term directly.

As in Section 5.2.1, the oracle velocity $\mathbf{v}_t(\mathbf{x})$ is generally intractable. To address this, introduce a latent variable $\mathbf{z} \sim \pi(\mathbf{z})$ and define a conditional velocity field $\mathbf{v}_t(\mathbf{x} \mid \mathbf{z})$ by construction. This allows us to rewrite the loss via the law of total expectation:

$$\mathcal{L}_{\text{FM}}(\phi) = \underbrace{\mathbb{E}_{t, \mathbf{z} \sim \pi(\mathbf{z}), \mathbf{x}_t \sim p_t(\cdot \mid \mathbf{z})} \left[ \lVert \mathbf{v}_\phi(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t \mid \mathbf{z}) \rVert^2 \right]}_{\mathcal{L}_{\text{CFM}}(\phi)} + C, \qquad (5.2.9)$$

where $C$ is a constant independent of $\phi$. The main term $\mathcal{L}_{\text{CFM}}$ is referred to as **conditional flow matching**.

For $\mathcal{L}_{\text{CFM}}(\phi)$ to enable tractable, simulation-free training, two requirements must be met:

1. Sampling from the conditional probability path $p_t(\mathbf{x}_t \mid \mathbf{z})$ should be straightforward (simulation-free).
2. The conditional velocity $\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{z})$, used as the regression target, must admit a simple closed-form expression.

One such field can be recovered by marginalizing the conditional velocity fields:

$$\mathbf{v}_t(\mathbf{x}_t) := \mathbb{E}_{\mathbf{z} \sim p(\cdot \mid \mathbf{x}_t)} \left[ \mathbf{v}_t(\mathbf{x}_t \mid \mathbf{z}) \right], \qquad (5.2.10)$$

and the minimizer $\mathbf{v}^\star$ of the conditional flow matching objective in Equation (5.2.9) recovers this marginal velocity:

$$\mathbf{v}^\star(\mathbf{x}_t, t) = \mathbf{v}_t(\mathbf{x}_t). \qquad (5.2.11)$$

Thus, learning to match the conditional velocity field $\mathbf{v}_t(\cdot \mid \mathbf{z})$ suffices to recover a valid unconditional velocity field.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.2.1: Equivalence of $\mathcal{L}_{\text{FM}}$ and $\mathcal{L}_{\text{CFM}}$)</span></p>

The following holds:

$$\mathcal{L}_{\text{FM}}(\phi) = \mathcal{L}_{\text{CFM}}(\phi) + C,$$

where $C$ is a constant independent of the parameter $\phi$. Furthermore, the minimizer $\mathbf{v}^\star$ of both losses satisfies

$$\mathbf{v}^\star(\mathbf{x}_t, t) = \mathbf{v}_t(\mathbf{x}_t), \quad \text{for almost every } \mathbf{x}_t \sim p_t,$$

where $\mathbf{v}_t(\mathbf{x}_t)$ is defined in Equation (5.2.10).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bayes' rule decomposition of the marginal score)</span></p>

Taking $\pi = p_{\text{data}}$, we can apply Bayes' rule:

$$p(\mathbf{x}_0 \mid \mathbf{x}_t) = \frac{p_t(\mathbf{x}_t \mid \mathbf{x}_0) p_{\text{data}}(\mathbf{x}_0)}{p_t(\mathbf{x}_t)},$$

and a similar decomposition of Equation (5.2.10) appears in score-based models:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot \mid \mathbf{x}_t)} \left[ \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) \right] = \mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}} \left[ \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) \cdot \frac{p_t(\mathbf{x}_t \mid \mathbf{x}_0)}{p_t(\mathbf{x}_t)} \right],$$

which mirrors the marginalization strategy in Equation (5.2.10).

</div>

---

### 5.2.3 Comparison of Diffusion Models, General Flow Matching, and NODEs

| Aspect | Diffusion Model | General FM |
| --- | --- | --- |
| Source dist. $p_{\text{src}}$ | Gaussian prior | Any |
| Target dist. $p_{\text{tgt}}$ | Data distribution | Any |
| Latent dist. $\pi(\mathbf{z})$ | $p_{\text{data}}$ | See Section 5.3.2 |
| Cond. dist. $p_t(\mathbf{x}_t \mid \mathbf{z})$ | $\mathcal{N}(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$ | See Section 5.3.2 |
| Marginal dist. $p_t(\mathbf{x}_t)$ | $\int p_t(\mathbf{x}_t \mid \mathbf{x}_0) p_{\text{data}}(\mathbf{x}_0) \, \mathrm{d}\mathbf{x}_0$ | $\int p_t(\mathbf{x}_t \mid \mathbf{z}) \pi(\mathbf{z}) \, \mathrm{d}\mathbf{z}$ |
| Cond. velocity $\mathbf{v}_t(\mathbf{x} \mid \mathbf{z})$ | $f(t)\mathbf{x} - \frac{1}{2}g^2(t) \nabla \log p_t(\mathbf{x} \mid \mathbf{x}_0)$ | See Section 5.3.2 |
| Marginal velocity $\mathbf{v}_t(\mathbf{x})$ | $f(t)\mathbf{x} - \frac{1}{2}g^2(t) \nabla \log p_t(\mathbf{x})$ | See Equation (5.2.10) |
| Learning objective | $\mathcal{L}_{\text{SM}} = \mathcal{L}_{\text{DSM}} + C$ | $\mathcal{L}_{\text{FM}} = \mathcal{L}_{\text{CFM}} + C$ |
| Underlying Rule | Fokker--Planck / Continuity Equation | Continuity Equation |

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to NODEs)</span></p>

FM can be viewed as a simulation-free alternative to NODEs, introduced in Section 5.1.2. While CNFs require solving ODEs during maximum likelihood training, which is computationally intensive, FM bypasses this by directly regressing a prescribed velocity field through a simple regression loss. The key insight is that when the marginal density path connecting the source and target distributions is fixed, exact simulation during training becomes unnecessary.

</div>

---

### 5.2.4 (Optional) Underlying Rules

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Continuity Equation: Mass Conservation Criterion)</span></p>

Consider the ODE describing the flow of particles under a time-dependent velocity field $\mathbf{v}_t$:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{v}_t(\mathbf{x}(t)).$$

This ODE defines a flow map $\boldsymbol{\Psi}_{s \to t}(\mathbf{x}_0)$ for any $s, t \in [0, 1]$, which in particular transports an initial point $\mathbf{x}_0 \sim p_{\text{src}}$ at time $0$ to its state at time $t$. The induced distribution at time $t$ is given by the pushforward:

$$p_t^{\text{fwd}}(\mathbf{x}) = \int \delta(\mathbf{x} - \boldsymbol{\Psi}_{0 \to t}(\mathbf{x}_0)) \, p_{\text{src}}(\mathbf{x}_0) \, \mathrm{d}\mathbf{x}_0 =: \boldsymbol{\Psi}_{0 \to t} \# p_{\text{src}}, \qquad (5.2.12)$$

so that $\boldsymbol{\Psi}_{0 \to t}(\mathbf{x}_0) \sim p_t^{\text{fwd}}$ whenever $\mathbf{x}_0 \sim p_{\text{src}}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(5.2.1)</span></p>

Under what conditions does the flow-induced density $p_t^{\text{fwd}}$ exactly match the target density $p_t$ for all $t \in [0, 1]$?

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.2.2: Mass Conservation Criterion)</span></p>

The flow-induced density $p_t^{\text{fwd}}$ equals the prescribed path $p_t$ for all $t \in [0, 1]$; i.e.,

$$p_t^{\text{fwd}} = p_t, \quad \text{for all } t \in [0, 1],$$

if and only if the pair $(p_t, \mathbf{v}_t)$ satisfies the continuity equation:

$$\partial_t p_t(\mathbf{x}) + \nabla_\mathbf{x} \cdot (p_t(\mathbf{x}) \mathbf{v}_t(\mathbf{x})) = 0,$$

for all $t \in [0, 1]$ and $\mathbf{x}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Conditional to Marginal Paths)</span></p>

As seen in Section 5.2.2, we begin by defining a conditional probability path $p_t(\cdot \mid \mathbf{z})$ and a corresponding conditional velocity field $\mathbf{v}_t(\cdot \mid \mathbf{z})$. We then construct the marginal velocity field via:

$$\mathbf{v}_t(\mathbf{x}) = \int \mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) \frac{p_t(\mathbf{x} \mid \mathbf{z}) \pi(\mathbf{z})}{p_t(\mathbf{x})} \, \mathrm{d}\mathbf{z},$$

as in Equation (5.2.10). The key question is whether this resulting marginal velocity $\mathbf{v}_t$ induces an ODE flow whose density path aligns with the prescribed $p_t$. This verification can be done entirely at the conditional level: if each conditional velocity field $\mathbf{v}_t(\cdot \mid \mathbf{z})$ induces the conditional density path $p_t(\cdot \mid \mathbf{z})$, then the resulting marginal velocity $\mathbf{v}_t$ also induces the correct marginal path.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.2.3: Marginal VF Generates Given Marginal Density)</span></p>

If the conditional velocity fields $\mathbf{v}_t(\cdot \mid \mathbf{z})$ induce conditional density paths that match $p_t(\cdot \mid \mathbf{z})$ (starting from $p_0(\cdot \mid \mathbf{z})$), then the marginal velocity field $\mathbf{v}_t(\cdot)$ defined in Equation (5.2.10) induces a marginal density path that aligns with $p_t(\cdot)$, starting from $p_0(\cdot)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 5.2.3</summary>

This result follows by verifying that the pair $(p_t, \mathbf{v}_t)$ satisfies the Continuity Equation. Since the conditional velocity fields $\mathbf{v}_t(\mathbf{x} \mid \mathbf{z})$ induce density paths matching the conditional densities $p_t(\cdot \mid \mathbf{z})$ for $\mathbf{z} \sim \pi$, the continuity equation holds for each conditional pair:

$$\frac{\mathrm{d}}{\mathrm{d}t} p_t(\mathbf{x} \mid \mathbf{z}) = -\nabla_\mathbf{x} \cdot (\mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) p_t(\mathbf{x} \mid \mathbf{z})). \qquad (5.2.13)$$

We aim to find a velocity field $\mathbf{v}_t(\cdot)$ whose induced densities align with the marginal density $p_t$, i.e., satisfy

$$\frac{\mathrm{d}}{\mathrm{d}t} p_t(\mathbf{x}) = -\nabla_\mathbf{x} \cdot (\mathbf{v}_t(\mathbf{x}) p_t(\mathbf{x})). \qquad (5.2.14)$$

Starting from the marginal density $p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{z}) \pi(\mathbf{z}) \, \mathrm{d}\mathbf{z}$, differentiating and applying (5.2.13):

$$\frac{\mathrm{d}}{\mathrm{d}t} p_t(\mathbf{x}) = \int \frac{\mathrm{d}}{\mathrm{d}t} p_t(\mathbf{x} \mid \mathbf{z}) \pi(\mathbf{z}) \, \mathrm{d}\mathbf{z} = -\int \nabla_\mathbf{x} \cdot (\mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) p_t(\mathbf{x} \mid \mathbf{z})) \pi(\mathbf{z}) \, \mathrm{d}\mathbf{z}$$

$$= -\nabla_\mathbf{x} \cdot \left( \int \mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) p_t(\mathbf{x} \mid \mathbf{z}) \pi(\mathbf{z}) \, \mathrm{d}\mathbf{z} \right) = -\nabla_\mathbf{x} \cdot \left( p_t(\mathbf{x}) \int \mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) \frac{p_t(\mathbf{x} \mid \mathbf{z}) \pi(\mathbf{z})}{p_t(\mathbf{x})} \, \mathrm{d}\mathbf{z} \right).$$

Comparing with (5.2.14), the marginal velocity field is:

$$\mathbf{v}_t(\mathbf{x}) = \int \mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) \frac{p_t(\mathbf{x} \mid \mathbf{z}) \pi(\mathbf{z})}{p_t(\mathbf{x})} \, \mathrm{d}\mathbf{z} = \mathbb{E}_{\mathbf{z} \sim p(\cdot \mid \mathbf{x})} \left[ \mathbf{v}_t(\mathbf{x} \mid \mathbf{z}) \right],$$

which matches Equation (5.2.10), confirming that the marginal velocity satisfies the continuity equation and thus generates the correct marginal density path.

</details>
</div>

---

## Chapter 6: A Unified and Systematic Lens on Diffusion Models

This chapter presents a systematic viewpoint that connects the variational, score-based, and flow-based perspectives within a coherent picture. While motivated by different intuitions, these approaches converge on the same core mechanism underlying modern diffusion methods: define a forward corruption process that traces a path of marginals, then learn a time-varying vector field that transports a simple prior to the data distribution along this path.

---

## 6.1 Conditional Tricks: The Secret Sauce of Diffusion Models

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Three Views, One Secret Sauce)</span></p>

Until now, we have explored diffusion models from three seemingly distinct origins: variational, score-based, and flow-based perspectives. Each was originally motivated by different goals and led to its own training objectives (with a fixed $t$):

* **Variational View:** Learn a parametrized density $p_\phi(\mathbf{x}_{t-\Delta t} \mid \mathbf{x}_t)$ to approximate the oracle reverse transition $p(\mathbf{x}_{t-\Delta t} \mid \mathbf{x}_t)$ by minimizing:

$$\mathcal{J}_{\text{KL}}(\phi) := \mathbb{E}_{p_t(\mathbf{x}_t)} \left[ D_{\text{KL}}(p(\mathbf{x}_{t-\Delta t} \mid \mathbf{x}_t) \| p_\phi(\mathbf{x}_{t-\Delta t} \mid \mathbf{x}_t)) \right].$$

* **Score-Based View:** Learn a score model $\mathbf{s}_\phi(\mathbf{x}_t, t)$ to approximate the marginal score $\nabla_\mathbf{x} \log p_t(\mathbf{x}_t)$ via:

$$\mathcal{J}_{\text{SM}}(\phi) := \mathbb{E}_{p_t(\mathbf{x}_t)} \left[ \lVert \mathbf{s}_\phi(\mathbf{x}_t, t) - \nabla_\mathbf{x} \log p_t(\mathbf{x}_t) \rVert_2^2 \right].$$

* **Flow-Based View:** Learn a velocity model $\mathbf{v}_\phi(\mathbf{x}_t, t)$ to match the oracle velocity $\mathbf{v}_t(\mathbf{x}_t)$ (e.g., defined by Equation (5.2.10)) by minimizing:

$$\mathcal{J}_{\text{FM}}(\phi) := \mathbb{E}_{p_t(\mathbf{x}_t)} \left[ \lVert \mathbf{v}_\phi(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t) \rVert_2^2 \right].$$

At first glance, these objectives seem hopelessly intractable, since they all require access to oracle quantities that are fundamentally unknowable in general. But here comes the exciting twist: each method independently arrives at the same elegant solution to this problem: **conditioning on the data $\mathbf{x}_0$**. This technique transforms each intractable training target into a tractable one.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Conditioning Trick: Tractable Conditional Objectives)</span></p>

This elegant "conditioning technique" rewrites the objectives as expectations over the known Gaussian conditionals $p_t(\mathbf{x}_t \mid \mathbf{x}_0)$, yielding gradient-equivalent closed-form regression targets and tractable training objectives:

* **Variational View** (Equation (2.2.3)):

$$\mathcal{J}_{\text{KL}}(\phi) = \mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{p_t(\mathbf{x}_t \mid \mathbf{x}_0)} \left[ \underbrace{D_{\text{KL}}(p(\mathbf{x}_{t-\Delta t} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_\phi(\mathbf{x}_{t-\Delta t} \mid \mathbf{x}_t))}_{\mathcal{J}_{\text{CKL}}(\phi)} \right] + C$$

* **Score-Based View** (Equation (3.3.3)):

$$\mathcal{J}_{\text{SM}}(\phi) = \underbrace{\mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{p_t(\mathbf{x}_t \mid \mathbf{x}_0)} \left[ \lVert \mathbf{s}_\phi(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) \rVert_2^2 \right]}_{\mathcal{J}_{\text{DSM}}(\phi)} + C$$

* **Flow-Based View** (Equation (5.2.9)):

$$\mathcal{J}_{\text{FM}}(\phi) = \underbrace{\mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{p_t(\mathbf{x}_t \mid \mathbf{x}_0)} \left[ \lVert \mathbf{v}_\phi(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t \mid \mathbf{x}_0) \rVert^2 \right]}_{\mathcal{J}_{\text{CFM}}(\phi)} + C$$

The conditional versions ($\mathcal{J}_{\text{CKL}}$, $\mathcal{J}_{\text{DSM}}$, $\mathcal{J}_{\text{CFM}}$) differ from the originals ($\mathcal{J}_{\text{KL}}$, $\mathcal{J}_{\text{SM}}$, $\mathcal{J}_{\text{FM}}$) only by a constant vertical shift, which leaves the gradients unchanged and thus preserves the optimization landscape. As a result, the minimizers remain uniquely identified with the true oracle targets, since each reduces to a least-squares regression problem whose solution recovers the corresponding conditional expectation:

$$p^\star(\mathbf{x}_{t-\Delta t} \mid \mathbf{x}_t) = \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot \mid \mathbf{x}_t)} [p(\mathbf{x}_{t-\Delta t} \mid \mathbf{x}_t, \mathbf{x}_0)] = p(\mathbf{x}_{t-\Delta t} \mid \mathbf{x}_t),$$

$$\mathbf{s}^\star(\mathbf{x}_t, t) = \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot \mid \mathbf{x}_t)} [\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0)] = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t), \qquad (6.1.1)$$

$$\mathbf{v}^\star(\mathbf{x}_t, t) = \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot \mid \mathbf{x}_t)} [\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{x}_0)] = \mathbf{v}_t(\mathbf{x}_t).$$

This is no coincidence: by making training tractable, these conditional forms reveal a profound unification. Variational diffusion, score-based SDEs, and flow matching are simply different facets of the same principle. Three perspectives, one insight, elegantly connected.

</div>

---

## 6.2 A Roadmap for Elucidating Training Losses in Diffusion Models

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Setup)</span></p>

Throughout this section, we consider the forward perturbation kernel

$$p_t(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I}),$$

where $\mathbf{x}_0 \sim p_{\text{data}}$. Let $\omega : [0, T] \to \mathbb{R}_{>0}$ denote a positive time-weighting function.

</div>

### 6.2.1 Four Common Parameterizations in Diffusion Models

The four standard parameterizations (noise $\boldsymbol{\epsilon}_\phi$, clean $\mathbf{x}_\phi$, score $\mathbf{s}_\phi$, and velocity $\mathbf{v}_\phi$), together with their respective minimizers $\boldsymbol{\epsilon}^\star$, $\mathbf{x}^\star$, $\mathbf{s}^\star$, and $\mathbf{v}^\star$, are summarized below.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variational View: Noise and Clean Predictions)</span></p>

Based on the KL divergence in DDPMs, this approach reduces to predicting either the expected noise that produces $\mathbf{x}_t$ or the expected clean signal that $\mathbf{x}_t$ was perturbed from.

1. **$\boldsymbol{\epsilon}$-Prediction (Noise Prediction):**

$$\boldsymbol{\epsilon}_\phi(\mathbf{x}_t, t) \approx \mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t] = \boldsymbol{\epsilon}^\star(\mathbf{x}_t, t) \qquad (6.2.1)$$

with training objective

$$\mathcal{L}_{\text{noise}}(\phi) := \mathbb{E}_t \left[ \omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \lVert \boldsymbol{\epsilon}_\phi(\mathbf{x}_t, t) - \boldsymbol{\epsilon} \rVert_2^2 \right].$$

Here, $\boldsymbol{\epsilon}^\star$ means the average noise that was injected to obtain the given $\mathbf{x}_t$.

2. **$\mathbf{x}$-Prediction (Clean Prediction):**

$$\mathbf{x}_\phi(\mathbf{x}_t, t) \approx \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t] = \mathbf{x}^\star(\mathbf{x}_t, t) \qquad (6.2.2)$$

with training objective

$$\mathcal{L}_{\text{clean}}(\phi) := \mathbb{E}_t \left[ \omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \lVert \mathbf{x}_\phi(\mathbf{x}_t, t) - \mathbf{x}_0 \rVert_2^2 \right].$$

Here, $\mathbf{x}^\star$ means the average of all plausible clean guesses, given the noisy observation $\mathbf{x}_t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Score-Based View: Score Prediction)</span></p>

Predicts the score function at noise level $t$, which points in the average direction to denoise $\mathbf{x}_t$ back toward all possible clean samples that could have generated it.

3. **Score Prediction:**

$$\mathbf{s}_\phi(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \mathbb{E}[\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) \mid \mathbf{x}_t] = \mathbf{s}^\star(\mathbf{x}_t, t) \qquad (6.2.3)$$

with training objective

$$\mathcal{L}_{\text{score}}(\phi) := \mathbb{E}_t \left[ \omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \lVert \mathbf{s}_\phi(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) \rVert_2^2 \right],$$

where the conditional score satisfies $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{1}{\sigma_t} \boldsymbol{\epsilon}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Flow-Based View: Velocity Prediction)</span></p>

Predicts the instantaneous average velocity of the data as it evolves through $\mathbf{x}_t$.

4. **$\mathbf{v}$-Prediction (Velocity Prediction):**

$$\mathbf{v}_\phi(\mathbf{x}_t, t) \approx \mathbb{E}\left[\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} \,\middle|\, \mathbf{x}_t \right] = \mathbf{v}^\star(\mathbf{x}_t, t) \qquad (6.2.4)$$

with training objective

$$\mathcal{L}_{\text{velocity}}(\phi) := \mathbb{E}_t \left[ \omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \lVert \mathbf{v}_\phi(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t \mid \mathbf{x}_0, \boldsymbol{\epsilon}) \rVert_2^2 \right],$$

where the conditional velocity is $\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{x}_0, \boldsymbol{\epsilon}) = \alpha_t' \mathbf{x}_0 + \sigma_t' \boldsymbol{\epsilon}$.

Here, $\mathbf{v}^\star$ indicates the average velocity vector passing through the observation point $\mathbf{x}_t$.

</div>

Building on the insight from Equation (6.1.1), all four prediction types ultimately aim to approximate a conditional expectation in the form of the average noise, clean data, score, or velocity given an observed $\mathbf{x}_t$.

### 6.2.2 Disentangling the Training Objective of Diffusion Models

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(General Template for Diffusion Training)</span></p>

As shown in Section 6.2.1, the objective functions for the four prediction types commonly share the following template form for diffusion model training:

$$\mathcal{L}(\phi) := \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \underbrace{\mathbb{E}_{p_{\text{time}}(t)}}_{\text{time distribution}} \left[ \underbrace{\omega(t)}_{\text{time weighting}} \underbrace{\lVert \text{NN}_\phi(\mathbf{x}_t, t) - (A_t \mathbf{x}_0 + B_t \boldsymbol{\epsilon}) \rVert_2^2}_{\text{MSE part}} \right]. \qquad (6.2.5)$$

To enhance training efficiency and optimize the diffusion model learning pipeline, several key design choices are crucial:

**(A)** Noise schedule in the forward process of $\mathbf{x}_t$ via $\alpha_t$ and $\sigma_t$;

**(B)** Prediction types of $\text{NN}_\phi$ and their associated regression targets $(A_t \mathbf{x}_0 + B_t \boldsymbol{\epsilon})$;

**(C)** Time-weighting function $\omega(\cdot) : [0, T] \to \mathbb{R}_{\geq 0}$;

**(D)** Time distribution $p_{\text{time}}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Component (A): Noise Schedule $\alpha_t$ and $\sigma_t$)</span></p>

Users have the flexibility to choose schedules tailored to their applications. As we will demonstrate in Equations (6.3.3) and (6.3.5), all affine flows of the form $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$ are mathematically equivalent. Specifically, any such interpolation can be converted to the canonical linear schedule ($\alpha_t = 1 - t$, $\sigma_t = t$) or to a trigonometric schedule ($\alpha_t = \cos t$, $\sigma_t = \sin t$) by appropriate time reparametrization and spatial rescaling.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Component (B): Parameterization and Regression Target $A_t \mathbf{x}_0 + B_t \boldsymbol{\epsilon}$)</span></p>

Users can flexibly choose the model's prediction target: the clean signal, noise, score, or velocity prediction. All these prediction types share a common regression target of the form

$$\text{Regression Target} = A_t \mathbf{x}_0 + B_t \boldsymbol{\epsilon},$$

where the coefficients $A_t$ and $B_t$ depend on both the chosen prediction type and the schedule $(\alpha_t, \sigma_t)$:

| Prediction Type | $A_t$ | $B_t$ |
| --- | --- | --- |
| Clean ($\mathbf{x}$) | $1$ | $0$ |
| Noise ($\boldsymbol{\epsilon}$) | $0$ | $1$ |
| Conditional Score | $0$ | $-\frac{1}{\sigma_t}$ |
| Conditional Velocity | $\alpha_t'$ | $\sigma_t'$ |

Although these four parameterizations appear distinct, they can be transformed into one another through simple algebraic manipulations (Equation (6.3.1)), and the squared-$\ell_2$ loss term in Equation (6.2.5) remains gradient-equivalent across all prediction types, differing only by a time-weighting factor that depends solely on the noise schedule $(\alpha_t, \sigma_t)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Component (C): Time Distribution $p_{\text{time}}(t)$)</span></p>

Since the training loss is an expectation over $t$, sampling times from $p_{\text{time}}(t)$ is mathematically equivalent to weighting the per-$t$ MSE by $p_{\text{time}}(t)$; this factor can be absorbed into the existing time weighting $\omega(t)$. However, empirical evidence indicates that different choices of $p_{\text{time}}(t)$ can affect performance. A common choice is the uniform distribution over $[0, T]$. Alternative options include the log-normal distribution and adaptive importance sampling methods.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Component (D): Time-Weighting Function $\omega(t)$)</span></p>

A common choice is $\omega \equiv 1$. Certain choices of $\omega(t)$ transform Equation (6.2.5) into a tighter upper bound on the negative log-likelihood, effectively reformulating the objective as maximum likelihood training. Notable weighting schemes include:

* $\omega(t) = g^2(t)$, where $g$ is the diffusion coefficient from the forward SDE.
* Signal-to-noise ratio (SNR) weighting.
* Monotonic weighting functions of time.

Overall, regardless of the choice of noise scheduler, prediction type, or time sampling distribution, these factors theoretically converge to influencing the time-weighting in the objective functions. This time-weighting can impact the practical training landscape and, consequently, the model's performance.

</div>

---

## 6.3 Equivalence in Diffusion Models

### 6.3.1 Four Prediction Types Are Equivalent

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalence of Prediction Types)</span></p>

We have seen that the four prediction types are not independent choices but different views of the same underlying quantity. For example, noise and clean predictions are directly related, as are score and noise predictions. This recurring pattern points to a deeper principle: all four parameterizations are algebraically equivalent and can be converted into one another through simple transformations.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(6.3.1: Equivalence of Parameterizations)</span></p>

Let the optimal predictions minimizing their respective objectives be

$$\boldsymbol{\epsilon}^\star(\mathbf{x}_t, t), \quad \mathbf{x}^\star(\mathbf{x}_t, t), \quad \mathbf{s}^\star(\mathbf{x}_t, t), \quad \mathbf{v}^\star(\mathbf{x}_t, t),$$

corresponding to noise, clean, score, and velocity parameterizations. These satisfy the following equivalences:

$$\boldsymbol{\epsilon}^\star(\mathbf{x}_t, t) = -\sigma_t \mathbf{s}^\star(\mathbf{x}_t, t),$$

$$\mathbf{x}^\star(\mathbf{x}_t, t) = \frac{1}{\alpha_t} \mathbf{x}_t + \frac{\sigma_t^2}{\alpha_t} \mathbf{s}^\star(\mathbf{x}_t, t), \qquad (6.3.1)$$

$$\mathbf{v}^\star(\mathbf{x}_t, t) = \alpha_t' \mathbf{x}^\star + \sigma_t' \boldsymbol{\epsilon}^\star = f(t)\mathbf{x}_t - \tfrac{1}{2}g^2(t) \mathbf{s}^\star(\mathbf{x}_t, t).$$

Here, $f(t)$ and $g(t)$ are related to $\alpha_t$ and $\sigma_t$ via Lemma 4.4.1. Moreover, these minimizers satisfy the identities given in Equations (6.2.1) to (6.2.4).

Equation (6.3.1) induces a one-to-one conversion (at each $t$, given the forward noising coefficients) between the four parameterizations $\boldsymbol{\epsilon}_\phi(\mathbf{x}_t, t)$, $\mathbf{x}_\phi(\mathbf{x}_t, t)$, $\mathbf{s}_\phi(\mathbf{x}_t, t)$, $\mathbf{v}_\phi(\mathbf{x}_t, t)$. In practice, we train a single network in one parameterization (e.g., $\boldsymbol{\epsilon}_\phi$). The other quantities are then *defined post hoc* by the conversions in Equation (6.3.1).

</div>

### 6.3.2 PF-ODE in Different Parameterizations

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(6.3.2: PF-ODE in Different Parameterizations)</span></p>

Let $\alpha_t$ and $\sigma_t$ be the forward perturbation schedules, and denote time derivatives by $\alpha_t' := \frac{\mathrm{d}\alpha_t}{\mathrm{d}t}$ and $\sigma_t' := \frac{\mathrm{d}\sigma_t}{\mathrm{d}t}$. Then the empirical PF-ODE admits the equivalent forms:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \frac{\alpha_t'}{\alpha_t}\mathbf{x}(t) - \sigma_t \left(\frac{\alpha_t'}{\alpha_t} - \frac{\sigma_t'}{\sigma_t}\right) \boldsymbol{\epsilon}^\star(\mathbf{x}(t), t)$$

$$= \frac{\sigma_t'}{\sigma_t}\mathbf{x}(t) + \alpha_t \left(\frac{\alpha_t'}{\alpha_t} - \frac{\sigma_t'}{\sigma_t}\right) \mathbf{x}^\star(\mathbf{x}(t), t) \qquad (6.3.2)$$

$$= \frac{\alpha_t'}{\alpha_t}\mathbf{x}(t) + \sigma_t^2 \left(\frac{\alpha_t'}{\alpha_t} - \frac{\sigma_t'}{\sigma_t}\right) \mathbf{s}^\star(\mathbf{x}(t), t)$$

$$= \alpha_t' \mathbf{x}^\star(\mathbf{x}(t), t) + \sigma_t' \boldsymbol{\epsilon}^\star(\mathbf{x}(t), t)$$

$$= \mathbf{v}^\star(\mathbf{x}(t), t).$$

To see the Score SDE notation, if we set $f(t) = \frac{\alpha_t'}{\alpha_t}$ and $g^2(t) = \frac{\mathrm{d}}{\mathrm{d}t}(\sigma_t^2) - 2\frac{\alpha_t'}{\alpha_t}\sigma_t^2 = 2\sigma_t\sigma_t' - 2\frac{\alpha_t'}{\alpha_t}\sigma_t^2$, then the PF-ODE can be written in the familiar Score SDE form:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = f(t)\mathbf{x}(t) - \tfrac{1}{2}g^2(t)\mathbf{s}^\star(\mathbf{x}(t), t).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Semilinear Form vs. v-Prediction)</span></p>

Under the $\mathbf{x}$-, $\boldsymbol{\epsilon}$-, and $\mathbf{s}$-parameterizations, the PF-ODE drift takes a **semilinear form**:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \underbrace{L(t)\mathbf{x}(t)}_{\text{linear part}} + \underbrace{\mathbf{N}_\phi(\mathbf{x}(t), t)}_{\text{nonlinear part}}, \quad \mathbf{N}_\phi \in \lbrace \mathbf{x}_\phi, \boldsymbol{\epsilon}_\phi, \mathbf{s}_\phi \rbrace.$$

When the linear drift $L(t)\mathbf{x}(t)$ drives changes in $\mathbf{x}(t)$ at very different rates in some directions compared with the nonlinear part, the system is *stiff*. In such cases, explicit solvers must take very small time steps to remain numerically stable. Higher-order stable solvers often apply an *integrating factor* that treats the linear term $L(t)\mathbf{x}$ analytically and discretizes only the slower nonlinear remainder.

**PF-ODE under v-Prediction.** With v-prediction, the model directly learns the velocity field and integrates $\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{v}_\phi(\mathbf{x}(t), t) \approx \mathbf{v}^\star(\mathbf{x}(t), t)$. The explicit linear term is absorbed into a single learned field, so the dynamics no longer split into separate parts. The step size is thus governed by how smoothly $\mathbf{v}_\phi(\mathbf{x}, t)$ varies with $\mathbf{x}$ and $t$, rather than by the magnitude of a prescribed scalar coefficient $L(t)$. This reduces time-scale disparity and simplifies numerical integration.

</div>

### 6.3.3 All Affine Flows Are Equivalent

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(State-Level Equivalence: Canonical FM Path)</span></p>

A convenient canonical interpolation used in FM and RF is:

$$\mathbf{x}_t^{\text{FM}} = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon} = \mathbf{x}_0 + t(\boldsymbol{\epsilon} - \mathbf{x}_0),$$

whose velocity is the constant vector $\boldsymbol{\epsilon} - \mathbf{x}_0$. The key point is that any affine interpolation

$$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$$

can be written as a time-reparameterized and rescaled version of the canonical path. Define

$$c(t) := \alpha_t + \sigma_t, \qquad \tau(t) := \frac{\sigma_t}{\alpha_t + \sigma_t} \quad (c(t) \neq 0).$$

Then $\mathbf{x}_t = c(t)\mathbf{x}_{\tau(t)}^{\text{FM}}$. Hence every affine path is the image of the canonical FM path under the change of variables $t \mapsto \tau(t)$ and the spatial rescaling $\mathbf{x} \mapsto c(t)\mathbf{x}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(6.3.3: Equivalence of Affine Flows)</span></p>

Let $\mathbf{x}_t^{\text{FM}} = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$ and $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$ with $c(t) := \alpha_t + \sigma_t \neq 0$ and $\tau(t) := \sigma_t / (\alpha_t + \sigma_t)$. Then

$$\mathbf{x}_t = c(t) \mathbf{x}_{\tau(t)}^{\text{FM}}, \qquad (6.3.3)$$

$$\mathbf{v}(\mathbf{x}_t, t) = c'(t) \mathbf{x}_{\tau(t)}^{\text{FM}} + c(t) \tau'(t) \mathbf{v}^{\text{FM}}\left(\mathbf{x}_{\tau(t)}^{\text{FM}}, \tau(t)\right).$$

In particular, all affine interpolations are equivalent up to time reparameterization and spatial rescaling.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalence with Trigonometric Flow)</span></p>

Another widely used affine flow is the trigonometric interpolation:

$$\mathbf{x}_u^{\text{Trig}} := \cos(u)\mathbf{x}_0 + \sin(u)\boldsymbol{\epsilon}. \qquad (6.3.4)$$

Let $R_t := \sqrt{\alpha_t^2 + \sigma_t^2}$ and assume $R_t > 0$. Choose an angle $\tau_t$ so that $\cos \tau_t = \alpha_t / R_t$ and $\sin \tau_t = \sigma_t / R_t$. Then every affine interpolation $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$ is a rescaled and re-timed trigonometric path:

$$\mathbf{x}_t = R_t \mathbf{x}_{\tau_t}^{\text{Trig}}. \qquad (6.3.5)$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Conclusion</span><span class="math-callout__name">(6.3.1)</span></p>

Regardless of the schedule $(\alpha_t, \sigma_t)$, including VE, VP (such as trigonometric), FM, or RF, affine interpolations are mutually convertible by a suitable change of time variable and a scalar rescaling.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Training Objectives of Four Parameterizations)</span></p>

Let $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$ with $\sigma_t > 0$ and differentiable $(\alpha_t, \sigma_t)$ such that $\alpha_t' \sigma_t - \alpha_t \sigma_t' \neq 0$. Consider the oracle targets

$$\boldsymbol{\epsilon}^\star(\mathbf{x}_t, t) = \mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t], \quad \mathbf{x}_0^\star(\mathbf{x}_t, t) = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t], \quad \mathbf{v}^\star(\mathbf{x}_t, t) = \mathbb{E}[\alpha_t' \mathbf{x}_0 + \sigma_t' \boldsymbol{\epsilon} \mid \mathbf{x}_t].$$

From Proposition 6.3.1, they satisfy:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{1}{\sigma_t}\boldsymbol{\epsilon}^\star(\mathbf{x}_t, t) = \frac{\alpha_t}{\sigma_t^2}\left(\mathbf{x}_0^\star(\mathbf{x}_t, t) - \frac{\mathbf{x}_t}{\alpha_t}\right), \quad \mathbf{v}^\star = \alpha_t' \mathbf{x}_0^\star + \sigma_t' \boldsymbol{\epsilon}^\star.$$

Under the head conversions

$$\mathbf{s}_\phi \equiv -\frac{1}{\sigma_t}\boldsymbol{\epsilon}_\phi \equiv \frac{\alpha_t}{\sigma_t^2}\left(\mathbf{x}_\phi - \frac{\mathbf{x}_t}{\alpha_t}\right),$$

the per-sample squared losses match up to time-dependent weights:

$$\lVert \mathbf{s}_\phi - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \rVert_2^2 = \frac{1}{\sigma_t^2} \lVert \boldsymbol{\epsilon}_\phi - \boldsymbol{\epsilon}^\star \rVert_2^2 = \frac{\alpha_t^2}{\sigma_t^4} \lVert \mathbf{x}_\phi - \mathbf{x}_0^\star \rVert_2^2 = \left(\frac{\alpha_t}{\sigma_t(\alpha_t'\sigma_t - \alpha_t\sigma_t')}\right)^2 \lVert \mathbf{v}_\phi - \mathbf{v}^\star \rVert_2^2. \qquad (6.3.6)$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Conclusion</span><span class="math-callout__name">(6.3.2)</span></p>

Score, noise, clean, and velocity training objectives are theoretically equivalent up to time-dependent weights (and, for velocity, an affine head conversion involving $\mathbf{x}_t$) determined by $(\alpha_t, \sigma_t)$.

</div>

### 6.3.4 (Optional) Conceptual Analysis of Parameterizations and the Canonical Flow

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Perspective 1: Why $(\alpha_t, \sigma_t) = (1-t, t)$ is a Natural Schedule)</span></p>

We distinguish between two types of velocity fields. The **conditional velocity**, which serves as a tractable training target, is defined as

$$\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{z}) = \mathbf{x}_t' = \alpha_t' \mathbf{x}_0 + \sigma_t' \boldsymbol{\epsilon}, \quad \text{where } \mathbf{z} = (\mathbf{x}_0, \boldsymbol{\epsilon}),$$

while the **oracle (marginalized) velocity**, used to move samples during inference of PF-ODE solving, is given by

$$\mathbf{v}^\star(\mathbf{x}, t) = \mathbb{E}[\mathbf{v}_t(\cdot \mid \mathbf{z}) \mid \mathbf{x}_t = \mathbf{x}].$$

Writing $\sigma_t := \rho(t)$ and $\alpha_t := 1 - \rho(t)$ for a time-varying $\rho(t)$, the conditional velocity becomes $\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{z}) = \rho'(t)(\boldsymbol{\epsilon} - \mathbf{x}_0)$.

**Unit-Scale Regression Targets.** For the canonical schedule $\rho(t) = t$, the conditional velocity $\mathbf{v}_t(\cdot \mid \mathbf{z})$ satisfies

$$\mathbb{E}\left[\lVert \mathbf{v}_t(\cdot \mid \mathbf{z}) \rVert_2^2\right] = \mathbb{E}_{\boldsymbol{\epsilon}} \lVert \boldsymbol{\epsilon} \rVert_2^2 + \mathbb{E}_{\mathbf{x}_0} \lVert \mathbf{x}_0 \rVert_2^2 = D + \operatorname{Tr}\operatorname{Cov}[\mathbf{x}_0] + \lVert \mathbb{E}\mathbf{x}_0 \rVert_2^2. \qquad (6.3.7)$$

Thus the expected target magnitude is **constant in $t$**. After standardizing the data to zero mean and identity covariance, the two components $\alpha_t' \mathbf{x}_0$ and $\sigma_t' \boldsymbol{\epsilon}$ contribute comparably for all $t$, avoiding gradient explosion/vanishing near the endpoints.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interplay of the Canonical Schedule and v-Prediction)</span></p>

Under the affine path $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$, the oracle velocity decomposes as

$$\mathbf{v}^\star(\mathbf{x}, t) = \alpha_t' \mathbf{x}^\star(\mathbf{x}, t) + \sigma_t' \boldsymbol{\epsilon}^\star(\mathbf{x}, t),$$

with $\mathbf{x}^\star = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]$ and $\boldsymbol{\epsilon}^\star = \mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t = \mathbf{x}]$. Differentiating at fixed $\mathbf{x}$ gives

$$\partial_t \mathbf{v}_t^\star = \underbrace{\alpha_t'' \mathbf{x}^\star + \sigma_t'' \boldsymbol{\epsilon}^\star}_{\text{schedule curvature}} + \alpha_t' \partial_t \mathbf{x}^\star + \sigma_t' \partial_t \boldsymbol{\epsilon}^\star.$$

With the linear schedule $\alpha_t = 1 - t$, $\sigma_t = t$, the curvature terms vanish ($\alpha_t'' = \sigma_t'' = 0$), so the time-variation of $\mathbf{v}_t^\star$ primarily reflects the posterior evolution ($\partial_t \mathbf{x}^\star$, $\partial_t \boldsymbol{\epsilon}^\star$) rather than the schedule. The coefficients $\alpha_t'$ and $\sigma_t'$ are constants ($-1$ and $+1$), avoiding extra $t$-dependent rescaling in the drift. By contrast, score-, $\mathbf{x}_0$-, or $\boldsymbol{\epsilon}$-parameterizations often introduce ratios such as $\sigma_t'/\sigma_t$ or $\alpha_t'/\alpha_t$ that can vary sharply near the endpoints.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Minimizing the Conditional Energy)</span></p>

The **conditional kinetic energy** quantifies the total expected motion of the conditional velocity along the forward path, i.e., the amount of instantaneous movement (or kinetic effort) required to traverse from $\mathbf{x}_0$ to $\boldsymbol{\epsilon}$:

$$\mathcal{K}[\rho] := \int_0^1 \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}[\lVert \mathbf{v}_t(\cdot \mid \mathbf{z}) \rVert_2^2] \, \mathrm{d}t = \left(D + \operatorname{Tr}\operatorname{Cov}[\mathbf{x}_0] + \lVert \mathbb{E}\mathbf{x}_0 \rVert_2^2\right) \int_0^1 (\rho'(t))^2 \, \mathrm{d}t.$$

Minimizing $\mathcal{K}[\rho]$ therefore corresponds to finding the smoothest, least-energy path in expectation. With the boundary conditions $\rho(0) = 0$ and $\rho(1) = 1$, the Euler--Lagrange equation $\rho''(t) = 0$ gives the minimizer $\rho(t) = t$, corresponding to a straight conditional path. This means that, among all smooth interpolations connecting $\mathbf{x}_0$ and $\boldsymbol{\epsilon}$, the canonical flow $\rho(t) = t$ is the most energy-efficient way to move between them.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark on the Oracle Velocity)</span></p>

If instead we evaluate the energy defined by marginal velocities

$$\int_0^1 \mathbb{E}_{\mathbf{x}_t \sim p_t} [\lVert \mathbf{v}^\star(\mathbf{x}_t, t) \rVert^2] \, \mathrm{d}t,$$

then with $\mathbf{z} = (\mathbf{x}_0, \boldsymbol{\epsilon})$ and $\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{z}) = \rho'(t)(\boldsymbol{\epsilon} - \mathbf{x}_0)$,

$$\mathbf{v}^\star(\mathbf{x}, t) = \mathbb{E}[\mathbf{v}_t(\cdot \mid \mathbf{z}) \mid \mathbf{x}_t = \mathbf{x}] = \rho'(t)\mathbb{E}[\boldsymbol{\epsilon} - \mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}];$$

and hence, the energy of the marginal velocity becomes

$$\int_0^1 \mathbb{E}_{\mathbf{x}_t \sim p_t}[\lVert \mathbf{v}^\star(\mathbf{x}_t, t) \rVert^2] \, \mathrm{d}t = \int_0^1 (\rho'(t))^2 \kappa(t) \, \mathrm{d}t,$$

where $\kappa(t) := \mathbb{E}_{\mathbf{x}_t \sim p_t}[\lVert \mathbb{E}[\boldsymbol{\epsilon} - \mathbf{x}_0 \mid \mathbf{x}_t] \rVert_2^2]$. Consequently, the *marginal*-optimal schedule $\rho(t)$ need not be linear. It is linear iff $\kappa(t)$ is constant; in general, the Euler--Lagrange condition $(\kappa(t)\rho'(t))' = 0$ gives $\rho'(t) \propto 1/\kappa(t)$. Intuitively, $\kappa(t)$ quantifies how predictable the label $(\boldsymbol{\epsilon} - \mathbf{x}_0)$ is from $\mathbf{x}_t \sim p_t$: the oracle flow slows down where $\kappa(t)$ is large (the oracle velocity has high expected magnitude) and speeds up where $\kappa(t)$ is small. Hence, even though the conditional flow uses the linear schedule $(1-t, t)$, the corresponding marginalized (oracle) dynamics are generally nonlinear.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Perspective 2: Why Velocity Prediction Can Be Considered Natural for Sampling)</span></p>

**Semilinear Form of the PF-ODE under $\mathbf{x}$-, $\boldsymbol{\epsilon}$-, and $\mathbf{s}$-Predictions.** Under these parameterizations, the drift takes a semilinear form

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \underbrace{L(t)\mathbf{x}(t)}_{\text{linear part}} + \underbrace{\mathbf{N}_\phi(\mathbf{x}(t), t)}_{\text{nonlinear part}}, \quad \mathbf{N}_\phi \in \lbrace \mathbf{x}_\phi, \boldsymbol{\epsilon}_\phi, \mathbf{s}_\phi \rbrace.$$

When $L(t)$ is large, the system becomes *stiff*, requiring very small time steps for explicit solvers or the use of an *exponential integrator* that treats the linear term analytically.

**PF-ODE under v-Prediction.** With v-prediction, the model directly learns the full velocity field:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{v}_\phi(\mathbf{x}(t), t) \approx \mathbf{v}^\star(\mathbf{x}(t), t).$$

The explicit linear term is absorbed into a single learned field, reducing time-scale disparity and simplifying numerical integration. The plain Euler update naturally coincides with the DDIM formulation. In contrast, for $\boldsymbol{\epsilon}$-, $\mathbf{x}$-, or $\mathbf{s}$-parameterizations, a plain Euler step only *approximates* the linear term, requiring exponential integrators to handle it exactly.

**Conclusion.** While v-prediction combined with the canonical linear schedule offers certain theoretical advantages (constant target magnitude, absence of schedule curvature, simpler ODE structure), it does not necessarily make it universally superior in practice. Model performance depends on a range of interacting factors, and the optimal configuration is ultimately an empirical question.

</div>

---

## 6.4 Beneath It All: The Fokker--Planck Equation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Unifying the Three Perspectives)</span></p>

The three main perspectives of diffusion models -- variational, score-based, and flow-based -- are not separate constructions but arise from a single unifying principle: the continuity (Fokker--Planck) equation that governs density evolution under a chosen forward process.

* The variational perspective, based on discrete kernels and Bayes' rule, is unified with the score-based SDE perspective of continuous dynamics (Section 4.5), showing that variational models act as consistent discretizations of the underlying forward and reverse SDEs.
* The flow-based and score-based views are connected through Section 6.4.1, which shows that an ODE flow determines a density path whose marginals can always be realized by a family of stochastic processes. This places deterministic flows and stochastic SDEs within the same family.

Together, these results unify the three perspectives under one framework (see Figure 6.2).

</div>

### 6.4.1 Connection of Flow-Based Approach and Score SDE

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From ODE to SDE: Bridging Deterministic and Stochastic Dynamics)</span></p>

A remarkable aspect of diffusion models lies in how different dynamic systems, deterministic or stochastic, can trace out the same evolution of probability distributions. In this section, we reveal a natural and elegant connection between ODE-based flows of Section 5.2 and Score SDEs. Specifically, the velocity field defining a generative ODE can be transformed into a stochastic counterpart that follows the same Fokker--Planck dynamics, providing a principled bridge between deterministic interpolation and stochastic sampling.

We consider the continuous-time setup where the perturbation kernel is $p_t(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$ with $\mathbf{x}_0 \sim p_{\text{data}}$. To match this density path, consider the ODE

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{v}_t(\mathbf{x}(t)), \quad t \in [0, T], \qquad (6.4.1)$$

where $\mathbf{v}_t(\mathbf{x}) = \mathbb{E}[\alpha_t' \mathbf{x}_0 + \sigma_t' \boldsymbol{\epsilon} \mid \mathbf{x}]$ is the oracle velocity as shown in Equation (5.2.10). Integrating equation (6.4.1) backward from $\mathbf{x}(T) \sim p_{\text{prior}}$ yields samples from $p_0$.

Although this ODE suffices for generating high-quality samples, incorporating stochasticity may improve sample diversity.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(6.4.1)</span></p>

Is there an SDE whose dynamics, starting from $p_{\text{prior}}$, yield the same marginal densities as the ODE in Equation (6.4.1)?

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(6.4.1: Reverse-Time SDEs Generate the Same Marginals as Interpolations)</span></p>

Let $\gamma(t) \geq 0$ be an arbitrary time-dependent coefficient. Consider the reverse-time SDE

$$\mathrm{d}\bar{\mathbf{x}}(t) = \left[\mathbf{v}^\star(\bar{\mathbf{x}}(t), t) - \tfrac{1}{2}\gamma^2(t)\mathbf{s}^\star(\bar{\mathbf{x}}(t), t)\right]\mathrm{d}\bar{t} + \gamma(t) \, \mathrm{d}\bar{\mathbf{w}}(t), \qquad (6.4.2)$$

evolving backward from $\bar{\mathbf{x}}(T) \sim p_T$ down to $t = 0$. Then this process $\lbrace \bar{\mathbf{x}}(t) \rbrace_{t \in [0,T]}$ matches the prescribed marginals $\lbrace p_t \rbrace_{t \in [0,T]}$ induced by the ODE's density path. Here, $\mathbf{s}(\mathbf{x}, t) := \nabla_\mathbf{x} \log p_t(\mathbf{x})$ is the score function, and it is related to the velocity field $\mathbf{v}(\mathbf{x}, t)$ by

$$\mathbf{v}^\star(\mathbf{x}, t) = f(t)\mathbf{x} - \tfrac{1}{2}g^2(t)\mathbf{s}^\star(\mathbf{x}, t), \quad \mathbf{s}^\star(\mathbf{x}, t) = \frac{1}{\sigma_t} \cdot \frac{\alpha_t \mathbf{v}^\star(\mathbf{x}, t) - \alpha_t'\mathbf{x}}{\alpha_t'\sigma_t - \alpha_t\sigma_t'}. \qquad (6.4.3)$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 6.4.1</summary>

The reverse-time Fokker--Planck equation corresponding to Equation (6.4.2) is

$$\partial_{\bar{t}} p = -\nabla \cdot \left([\mathbf{v}^\star - \tfrac{1}{2}\gamma^2 \mathbf{s}^\star] p\right) + \tfrac{1}{2}\gamma^2 \Delta p.$$

Using the identity $\nabla \cdot (\mathbf{s}^\star p) = \nabla \cdot ((\nabla \log p) p) = \Delta p$ (since $\mathbf{s}^\star = \nabla \log p$), the second-order terms cancel, yielding

$$\partial_{\bar{t}} p = -\nabla \cdot (\mathbf{v}^\star p),$$

i.e., the first-order (drift-only) Fokker--Planck equation associated with the PF-ODE. Hence the reverse-time SDE and the ODE induce the same marginal density path $\lbrace p_t \rbrace$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Role of $\gamma(t)$: Interpolating Between ODE and SDE)</span></p>

The hyperparameter $\gamma(t)$ can be chosen arbitrarily, independently of $\alpha_t$ and $\sigma_t$, even after training, as it does not affect the velocity $\mathbf{v}(\mathbf{x}, t)$ or the score $\mathbf{s}(\mathbf{x}, t)$. Below are some examples:

* Setting $\gamma(t) = 0$ recovers the ODE in Equation (6.4.1).
* When $\gamma(t) = g(t)$, Equation (6.4.2) becomes the reverse-time SDE from the Score SDE framework, since the oracle velocity $\mathbf{v}(\mathbf{x}, t)$ satisfies $\mathbf{v}^\star = f(t)\mathbf{x} - \frac{1}{2}g^2(t)\mathbf{s}^\star$.
* Other choices for $\gamma(t)$ have been explored; e.g., selecting $\gamma(t)$ to minimize the KL gap between $p_{\text{data}}$ and the $t = 0$ density obtained by solving Equation (6.4.2) from $t = T$.

Following Score SDE, the trained velocity field $\mathbf{v}_{\phi^\star}(\mathbf{x}, t)$ can be converted into a parameterized score function $\mathbf{s}_{\phi^\star}(\mathbf{x}, t)$ via Equation (6.4.3). Plugging this into Equation (6.4.2) defines an *empirical reverse-time SDE*, which can be sampled by numerically integrating from $t = T$ with $\bar{\mathbf{x}}(T) \sim p_{\text{prior}}$.

This proposition highlights a remarkable flexibility of diffusion models: once a marginal density path $\lbrace p_t \rbrace_{t \in [0,T]}$ is fixed, an entire family of dynamics can reproduce it, including both the PF-ODE and the reverse-time SDEs

$$\mathrm{d}\bar{\mathbf{x}}(t) = [\mathbf{v}^\star(\bar{\mathbf{x}}, t) - \tfrac{1}{2}\gamma^2(t)\mathbf{s}^\star(\bar{\mathbf{x}}, t)] \, \mathrm{d}\bar{t} + \gamma(t) \, \mathrm{d}\bar{\mathbf{w}}(t), \quad \gamma(t) \geq 0.$$

All such dynamics satisfy the same reverse-time Fokker--Planck equation and hence yield the same marginal evolution. The function $\gamma(t)$ continuously modulates the level of stochasticity without affecting the one-time distributions, revealing a deep connection between the deterministic flow-based ODE and its stochastic SDE counterpart.

</div>

---

## 6.5 Closing Remarks

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Chapter 6: Key Takeaways)</span></p>

This chapter has served as the keystone of our theoretical exploration, synthesizing the variational, score-based, and flow-based perspectives into a single, cohesive framework.

**Two core insights:**

1. **The conditioning trick** is the secret sauce common to all frameworks: it transforms an intractable marginal training objective into a tractable conditional one, enabling stable and efficient learning.

2. **The Fokker--Planck equation** is the universal law governing the evolution of probability densities. All three perspectives, in their own way, construct a generative process that respects this fundamental dynamic.

**Equivalence of parameterizations:** The various model parameterizations -- noise, clean data, score, or velocity prediction -- are all interchangeable. The choice of prediction target is a matter of implementation and stability rather than a fundamental modeling difference.

**The ultimate takeaway:** Modern diffusion methods, despite their diverse origins, all instantiate the same core principle: they learn a time-dependent vector field to transport a simple prior to the data distribution. With this unified foundation, we are now equipped to move from foundational theory to practical application and acceleration of diffusion models.

</div>

---

## Chapter 7: (Optional) Diffusion Models and Optimal Transport

Mapping one distribution to another (with generation as a special case) is a central challenge. Flow matching addresses this by learning a time-dependent velocity field that transports mass from source to target. This naturally connects to transport theory: classical optimal transport seeks the minimal cost path between distributions, while its entropy regularized form, the Schrodinger bridge, selects the most likely controlled diffusion relative to a reference such as Brownian motion.

### 7.1 Prologue of Distribution-to-Distribution Translation

Diffusion models fix the terminal distribution to a standard Gaussian, $p_{\text{prior}}$. However, many applications require *distribution-to-distribution* translation: transforming a source distribution $p_{\text{src}}$ into a different target $p_{\text{tgt}}$.

*One-endpoint* methods such as SDEdit begin with a source image at $t = 0$, diffuse it to an intermediate step $t$, and then use a pre-trained diffusion model for the target domain to reverse the process.

*Two-endpoint* methods, like Dual Diffusion Bridge, connect the two domains through a shared latent distribution, typically a Gaussian at $t = 1$. A forward probability-flow ODE transports samples from $p_{\text{src}}$ into this latent space, while a reverse ODE trained on the target domain maps them back to $p_{\text{tgt}}$.

The Flow Matching framework offers a training-based alternative: it directly learns an ODE flow that continuously moves mass from $p_{\text{src}}$ to $p_{\text{tgt}}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(7.1.1)</span></p>

*Given two probability distributions, what is the most efficient way to transform one into the other while minimizing the total cost?*

Here, the cost $c(\mathbf{x}, \mathbf{y})$ is a non-negative function that assigns a penalty for moving a unit of mass from a point $\mathbf{x}$ to a point $\mathbf{y}$. A common choice is the squared distance, $c(\mathbf{x}, \mathbf{y}) = \lVert \mathbf{x} - \mathbf{y} \rVert^2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(7.1.2)</span></p>

*Is a diffusion model a form of optimal transport connecting $p_{\text{data}}$ and $p_{\text{prior}}$, and in what sense?*

</div>

To address this question, we first clarify what "optimality" means in Section 7.2. We review classical *optimal transport (OT)* in the static Monge--Kantorovich form and its dynamic Benamou--Brenier formulation, as well as the entropy regularized variant (*entropic OT*), which is equivalent to the *Schrodinger Bridge Problem*.

The discussion is then split into two parts:

1. **Section 7.4:** The fixed forward noising SDE used in standard diffusion models is not, by itself, a Schrodinger bridge between arbitrary $p_{\text{src}}$ and $p_{\text{tgt}}$. It is not entropic OT optimal unless one explicitly solves the SB problem. It is an optimal solution to the *half-bridge* problem as it is anchored with one starting point.

2. **Section 7.5:** The PF-ODE defines a deterministic map that transports $p_{\text{prior}}$ to $p_{\text{data}}$ by construction. However, this flow is generally not an OT map for a prescribed transport cost (e.g., quadratic $\mathcal{W}_2$). The exact characterization between diffusion model's PF-ODE map and OT remains challenging and unsolved.

---

### 7.2 Taxonomy of the Problem Setups

#### 7.2.1 Optimal Transport (OT)

**Monge--Kantorovich (Static) Formulation of OT Problem.** We fix a cost function $c : \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}$ that specifies the expense of sending probability mass from $\mathbf{x}$ to $\mathbf{y}$. The goal is to transform the source distribution $p_{\text{src}}$ into the target distribution $p_{\text{tgt}}$ as cheaply as possible.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coupling)</span></p>

A **coupling** is a joint distribution $\gamma$ on $\mathbb{R}^D \times \mathbb{R}^D$ whose marginals are $p_{\text{src}}$ and $p_{\text{tgt}}$. Sampling $(\mathbf{x}, \mathbf{y}) \sim \gamma$ means we match $\mathbf{x}$ from the source with $\mathbf{y}$ from the target. If $\gamma$ admits a density $\gamma(\mathbf{x}, \mathbf{y})$ w.r.t. Lebesgue measure, the marginal constraints read

$$\int_{\mathbb{R}^D} \gamma(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\mathbf{y} = p_{\text{src}}(\mathbf{x}), \qquad \int_{\mathbb{R}^D} \gamma(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\mathbf{x} = p_{\text{tgt}}(\mathbf{y}).$$

Two standard examples:
1. **Discrete Supports.** If $p_{\text{src}}$ and $p_{\text{tgt}}$ are supported on finitely many points, a coupling is represented by a nonnegative matrix $(\gamma_{ij})$ whose row sums equal $p_{\text{src}}(i)$ and column sums equal $p_{\text{tgt}}(j)$.
2. **Deterministic Map.** If there exists a measurable map $\mathbf{T}$ with $\mathbf{T}_\# p_{\text{src}} = p_{\text{tgt}}$, then $\gamma = (\mathbf{I}, \mathbf{T})_\# p_{\text{src}}$ is a deterministic coupling that moves each point $\mathbf{x}$ directly to $\mathbf{T}(\mathbf{x})$.

</div>

Once a coupling $\gamma$ is fixed, the transport cost is simply the average unit cost under this plan:

$$\int c(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\gamma(\mathbf{x}, \mathbf{y}) = \mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim \gamma}[c(\mathbf{x}, \mathbf{y})].$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimal Transport (Monge--Kantorovich))</span></p>

The optimal transport problem is to choose, among all admissible couplings, the one that minimizes the expected cost:

$$\text{OT}(p_{\text{src}}, p_{\text{tgt}}) := \inf_{\gamma \in \Gamma(p_{\text{src}}, p_{\text{tgt}})} \int c(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\gamma(\mathbf{x}, \mathbf{y}), \tag{7.2.1}$$

where the feasible set enforces the marginal (mass-conservation) constraints:

$$\Gamma(p_{\text{src}}, p_{\text{tgt}}) = \left\lbrace \gamma \in \mathcal{P}(\mathbb{R}^D \times \mathbb{R}^D) : \int \gamma(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\mathbf{y} = p_{\text{src}}(\mathbf{x}), \; \int \gamma(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\mathbf{x} = p_{\text{tgt}}(\mathbf{y}) \right\rbrace.$$

</div>

**A Special Case: Wasserstein-2 Distance.** The Wasserstein-2 distance is a special case of the Monge--Kantorovich problem with the quadratic cost $c(\mathbf{x}, \mathbf{y}) = \lVert \mathbf{x} - \mathbf{y} \rVert^2$:

$$\mathcal{W}_2^2(p_{\text{src}}, p_{\text{tgt}}) := \inf_{\gamma \in \Gamma(p_{\text{src}}, p_{\text{tgt}})} \mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim \gamma}\left[\lVert \mathbf{x} - \mathbf{y} \rVert^2\right].$$

Under suitable assumptions on $p_{\text{src}}$ and $p_{\text{tgt}}$, Brenier's theorem guarantees that the optimal coupling $\gamma$ for the quadratic cost is concentrated on the graph of a deterministic map $\mathbf{T} : \mathbb{R}^D \to \mathbb{R}^D$. Consequently, the Wasserstein-2 distance can be equivalently expressed as:

$$\mathcal{W}_2^2(p_{\text{src}}, p_{\text{tgt}}) = \inf_{\substack{\mathbf{T}: \mathbb{R}^D \to \mathbb{R}^D, \\ \mathbf{T}_\# p_{\text{src}} = p_{\text{tgt}}}} \mathbb{E}_{\mathbf{x} \sim p_{\text{src}}}\left[\lVert \mathbf{T}(\mathbf{x}) - \mathbf{x} \rVert^2\right]. \tag{7.2.2}$$

The optimal transport map $\mathbf{T}^\ast(\mathbf{x})$, known as the *Monge map*, yields the most efficient way to transform $p_{\text{src}}$ into $p_{\text{tgt}}$.

---

**Benamou--Brenier (Dynamic) Formulation of OT.** Instead of mapping distributions directly in a static manner, transport can also be modeled as a continuous-time flow:

$$p_0 := p_{\text{src}} \to p_t \to p_1 := p_{\text{tgt}}, \quad t \in [0, 1].$$

This dynamic formulation, introduced by Benamou and Brenier, seeks a smooth velocity field $\mathbf{v}_t(\mathbf{x})$ that describes how mass in $p_t(\mathbf{x})$ evolves over time.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Benamou--Brenier Formulation)</span></p>

For the quadratic cost $c(\mathbf{x}, \mathbf{y}) = \lVert \mathbf{x} - \mathbf{y} \rVert_2^2$ (i.e., the $\mathcal{W}_2$ distance), the optimal value of the static OT problem in Equation (7.2.1) is equal to the optimal value of the kinetic energy minimization problem:

$$\mathcal{W}_2^2(p_{\text{src}}, p_{\text{tgt}}) = \min_{\substack{(p_t, \mathbf{v}_t) \text{ s.t. } \partial_t p_t + \nabla \cdot (p_t \mathbf{v}_t) = 0, \\ p_0 = p_{\text{src}}, \; p_1 = p_{\text{tgt}}}} \int_0^1 \int_{\mathbb{R}^D} \lVert \mathbf{v}_t(\mathbf{x}) \rVert^2 p_t(\mathbf{x}) \, \mathrm{d}\mathbf{x} \, \mathrm{d}t \tag{7.2.3}$$

where $p_t$ is a probability distribution on $\mathbb{R}^D$ for each $t \in [0, 1]$.

</div>

The optimal transport flow follows *McCann's displacement interpolation*:

$$\mathbf{T}_t^\ast(\mathbf{x}) = (1 - t)\mathbf{x} + t\mathbf{T}^\ast(\mathbf{x}),$$

where $\mathbf{T}^\ast(\mathbf{x})$ is the OT map that transports $p_{\text{src}}$ to $p_{\text{tgt}}$. This linear interpolation moves mass along straight lines with constant velocity: $p_t = \mathbf{T}_t^\ast{}_\# p_{\text{src}}$ for each $t \in [0, 1]$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Monge--Ampere Equation)</span></p>

The optimal transport map $\mathbf{T}^\ast$ satisfies the **Monge--Ampere equation**:

$$p_{\text{tgt}}(\nabla \psi(\mathbf{x})) \det\left(\nabla^2 \psi(\mathbf{x})\right) = p_{\text{src}}(\mathbf{x}), \tag{7.2.4}$$

where $\mathbf{T}^\ast(\mathbf{x}) = \nabla \psi(\mathbf{x})$ for some convex function $\psi$ by Brenier's theorem. However, this nonlinear PDE is typically intractable for explicit solutions.

Note that normalizing flows parametrize an invertible transport map with a tractable Jacobian determinant, but do not in general impose the gradient-of-potential structure $\mathbf{T}^\ast = \nabla \psi$; consequently, a trained flow can differ substantially from the Brenier/OT map.

</div>

---

#### 7.2.2 Entropy-Regularized Optimal Transport (EOT)

Classical OT in the discrete setting (taking counting measures in Equation (7.2.1)) reduces to

$$\min_{\gamma = (\gamma_{ij})} \sum_{i,j} C_{ij} \gamma_{ij},$$

over all feasible couplings $\gamma = (\gamma_{ij})$, where $C_{ij} = c(\mathbf{x}^{(i)}, \mathbf{y}^{(j)})$. Two main issues arise:

1. **Non-Uniqueness and Instability:** The minimizer $\gamma^\ast$ need not be unique. Small changes in the inputs $(a, b, C)$ can cause abrupt jumps in the solution.
2. **High Computational Cost:** The problem is a linear program with $n^2$ variables and $2n$ constraints. Practical solvers typically scale as $\mathcal{O}(n^3)$, which is infeasible for large $n$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Entropy-Regularized OT)</span></p>

To overcome these bottlenecks, EOT introduces a regularization term to the classical OT problem, controlled by a parameter $\varepsilon > 0$:

$$\text{EOT}_\varepsilon(p_{\text{src}}, p_{\text{tgt}}) := \min_{\gamma \in \Gamma(p_{\text{src}}, p_{\text{tgt}})} \int c(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\gamma(\mathbf{x}, \mathbf{y}) + \varepsilon \mathcal{D}_{\text{KL}}(\gamma \lVert M). \tag{7.2.5}$$

The reference measure $M$ is typically chosen as the product of the marginals, $p_{\text{src}} \otimes p_{\text{tgt}}$. The KL divergence term is directly related to the Shannon entropy of the transport plan $\gamma$:

$$\mathcal{D}_{\text{KL}}(\gamma \lVert p_{\text{src}} \otimes p_{\text{tgt}}) = -\mathcal{H}(\gamma) + \text{Constant},$$

where $\mathcal{H}(\gamma) := -\int \gamma(\mathbf{x}, \mathbf{y}) \log \gamma(\mathbf{x}, \mathbf{y}) \, \mathrm{d}\mathbf{x} \, \mathrm{d}\mathbf{y}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Why Entropy Regularizer Helps)</span></p>

1. **Mass Spreading.** Since $t \mapsto t \log t$ is convex and grows rapidly for large $t$, minimizing $\int \gamma \log \gamma$ penalizes *peaky* couplings. It favors plans where $\gamma(\mathbf{x}, \mathbf{y})$ is more evenly distributed, promoting higher "uncertainty" (diffuseness).

2. **Strict Convexity and Uniqueness.** Because $\mathcal{H}$ is strictly concave, the objective in Equation (7.2.5) is strictly convex in $\gamma$, yielding a *unique* minimizer $\gamma_\varepsilon^\ast$ that depends continuously on $(p_{\text{src}}, p_{\text{tgt}}, c)$.

3. **Sinkhorn Form and Positivity.** Under mild conditions, the optimizer has the *Schrodinger/Sinkhorn form*

$$\gamma_\varepsilon^\ast(\mathbf{x}, \mathbf{y}) = u(\mathbf{x}) \exp\!\left(-\frac{c(\mathbf{x}, \mathbf{y})}{\varepsilon}\right) v(\mathbf{y}) p_{\text{src}}(\mathbf{x}) p_{\text{tgt}}(\mathbf{y}),$$

for positive scaling functions $u, v$ (unique up to a global factor). The Sinkhorn/IPFP algorithm solves it efficiently: each iteration costs $\mathcal{O}(n^2)$ time and $\mathcal{O}(n^2)$ memory.

4. **Limits in $\varepsilon$.** As $\varepsilon \to 0$, the optimal plan $\gamma_\varepsilon^\ast$ becomes increasingly concentrated, approaching a (possibly singular) classical OT coupling. As $\varepsilon$ increases, $\gamma_\varepsilon^\ast$ gradually spreads out and approaches the independent coupling $p_{\text{src}} \otimes p_{\text{tgt}}$.

</div>

---

#### 7.2.3 Schrodinger Bridge (SB)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Schrodinger Bridge Problem)</span></p>

Suppose particles move according to some simple reference dynamics, such as Brownian motion. We observe the particles at two times: at $t = 0$ their distribution is $p_{\text{src}}$, and at $t = T$ it is $p_{\text{tgt}}$. Among all possible stochastic processes that connect these two distributions, the SB problem seeks the one that deviates the least from the reference.

Let $\mathbf{x}_{0:T} := \lbrace \mathbf{x}_t \rbrace_{t \in [0,T]}$ denote a complete trajectory. We write $P$ for the *law of trajectories*, that is, the probability distribution over entire sample paths. The time-$t$ marginal of $P$ is denoted by $p_t$ (or $P_t$), which describes the distribution of the state $\mathbf{x}_t$ at a single time.

Consider a reference diffusion $\lbrace \mathbf{x}_t \rbrace_{t \in [0,T]}$ governed by the SDE

$$\mathrm{d}\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t) \, \mathrm{d}t + g(t) \, \mathrm{d}\mathbf{w}_t, \tag{7.2.6}$$

where $\mathbf{f} : \mathbb{R}^D \times [0, T] \to \mathbb{R}^D$, $g : [0, T] \to \mathbb{R}$, and $\lbrace \mathbf{w}_t \rbrace_{t \in [0,T]}$ is a standard Brownian motion. Let $R$ denote the path law of this reference; this $R$ will serve as the *reference* trajectory distribution.

The **Schrodinger Bridge problem** seeks a trajectory law $P$ that is closest to $R$ in KL divergence while matching the prescribed endpoint marginals:

$$\text{SB}(p_{\text{src}}, p_{\text{tgt}}) := \min_P \mathcal{D}_{\text{KL}}(P \lVert R) \quad \text{s.t.} \quad P_0 = p_{\text{src}}, \; P_T = p_{\text{tgt}}. \tag{7.2.7}$$

The optimizer $P^\ast$ depends on the chosen reference process $R$.

</div>

**Stochastic control view of SB.** Rather than optimizing over arbitrary path distributions $P$, a more tractable approach is to take the reference dynamics as an anchor and allow it to drift. This is done by introducing a time-dependent drift $\mathbf{v}_t(\mathbf{x}_t)$, which perturbs the reference process. The resulting dynamics take the form of a *controlled diffusion*:

$$\mathrm{d}\mathbf{x}_t = [\mathbf{f}(\mathbf{x}_t, t) + \mathbf{v}_t(\mathbf{x}_t)] \, \mathrm{d}t + g(t) \, \mathrm{d}\mathbf{w}_t,$$

where $\mathbf{v}_t : \mathbb{R}^D \to \mathbb{R}^D$ is the drift to be optimized. By Girsanov's theorem, the KL divergence between the controlled law $P$ and the reference $R$ admits the dynamic (kinetic) form

$$\mathcal{D}_{\text{KL}}(P \lVert R) = \mathbb{E}_P\left[\frac{1}{2} \int_0^T \frac{\lVert \mathbf{v}_t(\mathbf{x}_t) \rVert^2}{g^2(t)} \, \mathrm{d}t\right] = \frac{1}{2} \int_0^T \int_{\mathbb{R}^D} \frac{\lVert \mathbf{v}_t(\mathbf{x}) \rVert^2}{g^2(t)} p_t(\mathbf{x}) \, \mathrm{d}\mathbf{x} \, \mathrm{d}t.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SB: Stochastic Control Formulation)</span></p>

The SB problem can be reformulated as minimizing the expected control energy over all admissible drifts $\mathbf{v}_t$ that steer the process from $p_{\text{src}}$ at $t = 0$ to $p_{\text{tgt}}$ at $t = T$:

$$\text{SB}_\varepsilon(p_{\text{src}}, p_{\text{tgt}}) = \min_{\substack{\mathbf{v}_t \text{ s.t. } \mathrm{d}\mathbf{x}_t = [\mathbf{f}(\mathbf{x}_t,t) + \mathbf{v}_t(\mathbf{x}_t)] \mathrm{d}t + g(t) \mathrm{d}\mathbf{w}_t, \\ \mathbf{x}_0 \sim p_{\text{src}}, \; \mathbf{x}_T \sim p_{\text{tgt}}}} \frac{1}{2} \int_0^T \int_{\mathbb{R}^D} \frac{\lVert \mathbf{v}_t(\mathbf{x}) \rVert^2}{g^2(t)} p_t(\mathbf{x}) \, \mathrm{d}\mathbf{x} \, \mathrm{d}t. \tag{7.2.8}$$

The control drift $\mathbf{v}_t$ is chosen precisely to "bridge" the reference dynamics between these marginals while staying as close as possible (in KL divergence) to the reference process $R$.

</div>

**A Special Brownian Reference.** Equation (7.2.8) resembles the Benamou--Brenier formulation of OT in Equation (7.2.3), especially when the reference process $R^\varepsilon$ (with $\varepsilon > 0$) is chosen to be a Brownian motion:

$$\mathrm{d}\mathbf{x}_t = \sqrt{\varepsilon} \, \mathrm{d}\mathbf{w}_t,$$

so that $\mathbf{f} \equiv \mathbf{0}$ and $g(t) \equiv \sqrt{\varepsilon}$. In this setting, the SB problem seeks a path distribution $P$ that stays closest (in KL divergence) to the Brownian reference $R^\varepsilon$, while matching the endpoint marginals:

$$\text{SB}_\varepsilon(p_{\text{src}}, p_{\text{tgt}}) := \min_P \mathcal{D}_{\text{KL}}(P \lVert R^\varepsilon) \quad \text{s.t.} \quad P_0 = p_{\text{src}}, \; P_T = p_{\text{tgt}}. \tag{7.2.9}$$

The equivalent stochastic control formulation then becomes

$$\text{SB}_\varepsilon(p_{\text{src}}, p_{\text{tgt}}) = \min_{\substack{\mathbf{v}_t \text{ s.t. } \mathrm{d}\mathbf{x}_t = \sqrt{\varepsilon} \, \mathrm{d}\mathbf{w}_t, \\ \mathbf{x}_0 \sim p_{\text{src}}, \; \mathbf{x}_T \sim p_{\text{tgt}}}} \frac{1}{2\varepsilon} \int_0^T \int_{\mathbb{R}^D} \lVert \mathbf{v}_t(\mathbf{x}) \rVert^2 p_t(\mathbf{x}) \, \mathrm{d}\mathbf{x} \, \mathrm{d}t. \tag{7.2.10}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why We Need to Specify a Reference Distribution)</span></p>

Unlike in classical OT, the SB problem requires a reference distribution due to its stochastic nature. In OT, the cost function (e.g., $c(\mathbf{x}, \mathbf{y}) \propto \lVert \mathbf{x} - \mathbf{y} \rVert^2$) implicitly defines a unique, deterministic geodesic path, making a reference unnecessary. In contrast, the SB setting admits infinitely many stochastic processes connecting the marginals, with no intrinsic notion of a "natural" path. The reference measure $R$ encodes the system's underlying physics or geometric structure (e.g., Brownian motion) and defines the KL-based optimization objective $\mathcal{D}_{\text{KL}}(P \lVert R)$, without which the notion of optimality is undefined.

</div>

**Coupled PDE Characterization.** A convenient way to describe the SB solution is through two space-time potentials $\Psi(x, t)$ and $\widehat{\Psi}(x, t)$. Let $p_t^{\text{SB}}$ denote the marginal at time $t \in [0, T]$ of the optimal trajectory law $P^\ast$ in Equation (7.2.7). Then one has the symmetric factorization:

$$p_t^{\text{SB}}(x) = \Psi(x, t) \widehat{\Psi}(x, t), \tag{7.2.11}$$

where $\Psi$ and $\widehat{\Psi}$ solve the (linear) *Schrodinger system*:

$$\frac{\partial \Psi}{\partial t}(\mathbf{x}, t) = -\nabla_\mathbf{x} \Psi(\mathbf{x}, t) \cdot \mathbf{f}(\mathbf{x}, t) - \frac{g^2(t)}{2} \Delta_\mathbf{x} \Psi(\mathbf{x}, t),$$

$$\frac{\partial \widehat{\Psi}}{\partial t}(\mathbf{x}, t) = -\nabla_\mathbf{x} \cdot (\widehat{\Psi}(\mathbf{x}, t) \, \mathbf{f}(\mathbf{x}, t)) + \frac{g^2(t)}{2} \Delta_\mathbf{x} \widehat{\Psi}(\mathbf{x}, t) \tag{7.2.12}$$

subject to $\Psi(\mathbf{x}, 0) \widehat{\Psi}(\mathbf{x}, 0) = p_{\text{src}}(\mathbf{x})$ and $\Psi(\mathbf{x}, T) \widehat{\Psi}(\mathbf{x}, T) = p_{\text{tgt}}(\mathbf{x})$.

**Forward-Time Schrodinger Bridge SDE.** Once $\Psi$ is known, the optimal dynamics is the reference diffusion tilted by the space-time factor $\Psi$:

$$\mathrm{d}\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) + g^2(t) \nabla_\mathbf{x} \log \Psi(\mathbf{x}_t, t)\right] \mathrm{d}t + g(t) \, \mathrm{d}\mathbf{w}_t, \quad \mathbf{x}_0 \sim p_{\text{src}}. \tag{7.2.13}$$

The minimizer $\mathbf{v}_t^\ast$ to Equation (7.2.8) is: $\mathbf{v}_t^\ast(\mathbf{x}) = g^2(t) \nabla_\mathbf{x} \log \Psi(\mathbf{x}, t)$. That is, drift correction $g^2 \nabla_\mathbf{x} \log \Psi$ is precisely the minimal KL perturbation of the reference needed to match the endpoint marginals.

**Reverse-Time Schrodinger Bridge SDE.** Using the standard time-reversal identity for diffusions, the reverse-time SDE reads

$$\mathrm{d}\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g^2(t) \nabla_\mathbf{x} \log \widehat{\Psi}(\mathbf{x}_t, t)\right] \mathrm{d}t + g(t) \, \mathrm{d}\bar{\mathbf{w}}_t, \quad \mathbf{x}_T \sim p_{\text{tgt}}. \tag{7.2.14}$$

Both the forward and reverse descriptions yield the same optimal path law $P^\ast$ which are linked by

$$\nabla \log p_t^{\text{SB}} = \nabla \log \Psi + \nabla \log \widehat{\Psi}, \qquad \mathbf{b}^- = \mathbf{b}^+ - g^2 \nabla \log p_t^{\text{SB}},$$

so their marginals coincide with $p_t^{\text{SB}}$ at every time. The additional drift terms $g^2 \nabla \log \Psi$ (forward) and $-g^2 \nabla \log \widehat{\Psi}$ (reverse-time) act as control forces that steer the reference diffusion to match the endpoint marginals while staying closest to the reference in relative entropy.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Obstacles to the Coupled PDE Approach)</span></p>

To construct the generative process based on Equation (7.2.14), one must solve the coupled PDEs in Equation (7.2.12) to obtain the backward Schrodinger potential $\widehat{\Psi}$. However, these PDEs are notoriously difficult to solve, even in low-dimensional settings. Several alternative strategies have been proposed: leveraging Score SDE techniques to iteratively solve each half-bridge problem; optimizing surrogate likelihood bounds; or designing simulation-free training based on an analytical solution of the posterior $\mathbf{x}_t \vert \mathbf{x}_0, \mathbf{x}_T$ for sample pairs $(\mathbf{x}_0, \mathbf{x}_T) \sim p_{\text{src}} \otimes p_{\text{tgt}}$.

</div>

---

#### 7.2.4 Global Pushforwards and Local Dynamics: An OT Analogy for DGMs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(OT Analogy for DGMs)</span></p>

From the optimal-transport viewpoint (Equation (7.2.1)), one can leverage deep generative models to learn a transport (pushforward) map from a simple prior to the data, i.e., $\mathbf{G}_\phi{}_\# p_{\text{prior}} \approx p_{\text{data}}$. Although $\mathbf{G}_\phi$ generally does not coincide with the optimal transport map, the Benamou--Brenier formulation (Equation (7.2.3)) provides a complementary, dynamic perspective: rather than directly learning a single global map, it describes transport as a continuous flow generated by a time-dependent local vector field, tracing a smooth path between $p_{\text{prior}}$ and $p_{\text{data}}$.

This dynamic formulation parallels the relationship between the static Schrodinger Bridge problem (Equation (7.2.7)) and its stochastic-control counterpart (Equation (7.2.8)), where the optimal coupling is realized as a controlled diffusion process. A similar analogy emerges in generative modeling: standard DGMs such as GANs or VAEs learn a global pushforward map, whereas diffusion models learn a time-dependent local vector field that drives the generative dynamics.

</div>

---

### 7.3 Relationship of Variant Optimal Transport Formulations

At a high level, the different formulations of optimal transport and its entropic regularizations are connected as follows:

- **(i)** $\text{SB}_\varepsilon$ (stochastic control) $\Leftrightarrow$ $\text{SB}_\varepsilon$ (static formulation), where $p_t$ are precisely the time-$t$ slices of the optimal path measure $P$ (see Section 7.3.1);
- **(ii)** Static formulation of $\text{SB}_\varepsilon$ connects directly to the entropic OT problem, $\text{EOT}_\varepsilon$ (see Section 7.3.1);
- **(iii)** $\text{EOT}_\varepsilon$, in turn, can be related back to the static formulation of entropic OT, $\text{OT}_\varepsilon$ (see Section 7.3.2);
- **(iv)** Stochastic control perspective of $\text{SB}_\varepsilon$ can also be linked to the dynamic formulation of classical OT (see Section 7.3.3).

#### 7.3.1 SB and EOT are (Dual) Equivalent

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(SB and EOT Equivalence)</span></p>

SB and EOT are essentially equivalent. Unlike classical OT, which produces a single deterministic map, SB yields a *stochastic* flow of particles transported probabilistically, with marginals evolving under diffusion-like dynamics.

**From the static viewpoint,** SB coincides with EOT: the goal is to find a coupling between the two endpoint distributions that balances transport cost with entropy.

**From the dynamic viewpoint,** SB describes a controlled diffusion process that remains as close as possible to a simple reference (such as Brownian motion) while still matching the desired endpoints.

</div>

**Static Schrodinger Bridge.** Let

$$\tilde{R}^\varepsilon(\mathbf{x}, \mathbf{y}) := \frac{1}{Z_\varepsilon} e^{-c(\mathbf{x}, \mathbf{y})/\varepsilon} p_{\text{src}}(\mathbf{x}) p_{\text{tgt}}(\mathbf{y}),$$

with a normalizing constant $Z_\varepsilon := \iint e^{-c(\mathbf{x}, \mathbf{y})/\varepsilon} p_{\text{src}}(\mathbf{x}) p_{\text{tgt}}(\mathbf{y}) \, \mathrm{d}\mathbf{x} \, \mathrm{d}\mathbf{y}$. Then the entropic OT objective

$$\min_{\gamma \in \Gamma(p_{\text{src}}, p_{\text{tgt}})} \left\lbrace \int c \, \mathrm{d}\gamma + \varepsilon \mathcal{D}_{\text{KL}}(\gamma \lVert p_{\text{src}} \otimes p_{\text{tgt}}) \right\rbrace = \varepsilon \min_{\gamma \in \Gamma(p_{\text{src}}, p_{\text{tgt}})} \mathcal{D}_{\text{KL}}(\gamma \lVert \tilde{R}^\varepsilon) - \varepsilon \log Z_\varepsilon, \tag{7.3.1}$$

so it is equivalent (up to an additive constant) to the static Schrodinger Bridge (Equation (7.2.9)):

$$\min_{\gamma \in \Gamma} \mathcal{D}_{\text{KL}}(\gamma \lVert \tilde{R}^\varepsilon).$$

**Dynamic Equivalence (Brownian Reference).** A classical result (Mikami and Thieullen, 2006) says that entropic OT with quadratic cost

$$c(\mathbf{x}, \mathbf{y}) = \frac{\lVert \mathbf{y} - \mathbf{x} \rVert^2}{2T}$$

is affinely equivalent to the SB problem where the reference path law $R^\varepsilon$ is Brownian motion on $[0, T]$. In particular, let $P^\ast$ be the optimal path distribution for SB and let $\gamma^\ast$ be the optimal transport plan for EOT. Then if $\mathbf{x}_{[0:T]} \sim P^\ast$, the pair of endpoints $(\mathbf{x}_0, \mathbf{x}_T)$ has distribution $\gamma^\ast$:

$$P^\ast \text{ solves SB} \iff \gamma^\ast \text{ solves EOT and } (\mathbf{x}_0, \mathbf{x}_T) \sim \gamma^\ast.$$

**SB with General Reference Determines the EOT Cost.** The SB problem is not restricted to Brownian motion; it can be defined with any (well-posed) reference process. This choice uniquely determines the cost function in the corresponding EOT problem. The key connection is that the SB *reference dynamics* induce the EOT *cost function*. Let the reference process be governed by an SDE over $[0, T]$, yielding a transition density $p_T(\mathbf{y} \vert \mathbf{x})$. Then the EOT cost function is given (up to a scaling constant) by

$$c(\mathbf{x}, \mathbf{y}) \propto -\log p_T(\mathbf{y} \vert \mathbf{x}).$$

In short, choosing the reference dynamics in SB is mathematically equivalent to specifying the transport cost in EOT.

---

#### 7.3.2 $\text{EOT}_\varepsilon$ is Reduced to OT as $\varepsilon \to 0$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.3.1: (Informal) $\text{EOT}_\varepsilon$ Converges to OT)</span></p>

As $\varepsilon \to 0$, the optimal values converge:

$$\lim_{\varepsilon \to 0} \text{EOT}_\varepsilon(p_{\text{src}}, p_{\text{tgt}}) = \text{OT}(p_{\text{src}}, p_{\text{tgt}}).$$

Moreover, the optimal plans $\gamma_\varepsilon^\ast$ *converge weakly* to $\gamma^\ast$. That is,

$$\mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim \gamma_\varepsilon^\ast}[g(\mathbf{x}, \mathbf{y})] \to \mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim \gamma^\ast}[g(\mathbf{x}, \mathbf{y})],$$

for all bounded continuous (test) functions $g : \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch for Theorem 7.3.1</summary>

Let us denote the corresponding optimal values by $V_\varepsilon := \text{EOT}_\varepsilon(p_{\text{src}}, p_{\text{tgt}})$ and $V_0 := \text{OT}(p_{\text{src}}, p_{\text{tgt}})$.

**Upper Bound.** By optimality of $\gamma_\varepsilon^\ast$, its value $V_\varepsilon$ is bounded by the cost of using the plan $\gamma^\ast$:

$$V_\varepsilon \le \int c \, \mathrm{d}\gamma^\ast + \varepsilon \mathcal{D}_{\text{KL}}(\gamma^\ast \lVert p_{\text{src}} \otimes p_{\text{tgt}}).$$

Assuming the KL term is a finite constant $K$, we get $V_\varepsilon \le V_0 + \varepsilon K$. Taking the limit superior yields $\limsup_{\varepsilon \to 0} V_\varepsilon \le V_0$.

**Lower Bound.** Since the KL-divergence is non-negative, $V_\varepsilon \ge \int c \, \mathrm{d}\gamma_\varepsilon^\ast$. By definition of $V_0$ as the minimal transport cost, any plan's cost is at least $V_0$, so $\int c \, \mathrm{d}\gamma_\varepsilon^\ast \ge V_0$. This implies $V_\varepsilon \ge V_0$ for all $\varepsilon > 0$, and thus $\liminf_{\varepsilon \to 0} V_\varepsilon \ge V_0$.

Combining the upper and lower bounds shows $\lim_{\varepsilon \to 0} V_\varepsilon = V_0$. The convergence of the optimal plan itself, $\gamma_\varepsilon^\ast \to \gamma^\ast$ in the weak sense, is a more advanced result from $\Gamma$-convergence theory.

</details>
</div>

This convergence result is both fundamental and practically important. One of the reasons is that the entropy-regularized OT problem $\text{EOT}_\varepsilon$ admits efficient numerical solutions via algorithms such as Sinkhorn. Thus, the result provides theoretical justification for using $\text{EOT}_\varepsilon$ with small $\varepsilon$ as a computationally tractable proxy for the classical OT problem in Equation (7.2.1).

---

#### 7.3.3 $\text{SB}_\varepsilon$ is Reduced to OT as $\varepsilon \to 0$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.3.2: (Informal) $\text{SB}_\varepsilon$ Converges to OT)</span></p>

As $\varepsilon \to 0$, we have:

$$\lim_{\varepsilon \to 0} \text{SB}_\varepsilon(p_{\text{src}}, p_{\text{tgt}}) = \text{OT}(p_{\text{src}}, p_{\text{tgt}}),$$

where OT is of the Benamou--Brenier formulation as in Equation (7.2.3). Moreover, $p_t^\varepsilon$ converges weakly to $p_t^0$, and $\mathbf{v}_t^\varepsilon$ converges weakly to $\mathbf{v}_t^0$ in the appropriate function spaces.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch for Theorem 7.3.2</summary>

In the stochastic control formulation of the SB problem (Equation (7.2.10)), the controlled SDE is:

$$\mathrm{d}\mathbf{x}_t = \mathbf{v}_t^\varepsilon(\mathbf{x}_t) \, \mathrm{d}t + \sqrt{2\varepsilon} \, \mathrm{d}\mathbf{w}_t.$$

As $\varepsilon \to 0$, the noise term vanishes, and the SDE formally approaches a deterministic ODE:

$$\mathrm{d}\mathbf{x}_t = \mathbf{v}_t^0(\mathbf{x}_t) \, \mathrm{d}t.$$

This suggests that the optimal value of the SB problem converges to that of the optimal transport problem:

$$\lim_{\varepsilon \to 0} \text{SB}_\varepsilon(p_{\text{src}}, p_{\text{tgt}}) = \text{OT}(p_{\text{src}}, p_{\text{tgt}}).$$

In parallel, the marginal density $p_t^\varepsilon$ satisfies the Fokker--Planck equation:

$$\partial_t p_t^\varepsilon + \nabla \cdot (p_t^\varepsilon \mathbf{v}_t^\varepsilon) = \varepsilon \Delta p_t^\varepsilon.$$

Again, as $\varepsilon \to 0$, the diffusion term vanishes, and the equation formally reduces to the continuity equation:

$$\partial_t p_t^0 + \nabla \cdot \left(p_t^0 \mathbf{v}_t^0\right) = 0.$$

</details>
</div>

---

### 7.4 Is Diffusion Model's SDE Optimal Solution to SB Problem?

#### 7.4.1 Diffusion Models as a Special Case of Schrodinger Bridges

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Diffusion Models as Zero-Control Limit of SB)</span></p>

The SB framework extends (score-based) diffusion models by enabling nonlinear interpolation between arbitrary source and target distributions. It achieves this by adding control drift terms derived from scalar potentials $\Psi(\mathbf{x}, t)$ and $\widehat{\Psi}(\mathbf{x}, t)$, which guide a reference diffusion process to match prescribed endpoint marginals (see Equation (7.2.12)) and follow the decomposition:

$$\nabla \log \Psi(x, t) + \nabla \log \hat{\Psi}(x, t) = \nabla \log p_t^{\text{SB}}(\mathbf{x}).$$

**Connection to Diffusion Models.** Diffusion models arise as a special case of the SB framework. Suppose the potential is constant, $\Psi(\mathbf{x}, t) \equiv 1$. Under this assumption, the second PDE in Equation (7.2.12) reduces to the standard Fokker--Planck equation, whose solution is the marginal density of the reference process:

$$\widehat{\Psi}(\mathbf{x}, t) = p_t^{\text{SB}}(\mathbf{x}). \tag{7.4.1}$$

The corresponding SB forward SDE thus becomes the uncontrolled reference process:

$$\mathrm{d}\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t) \, \mathrm{d}t + g(t) \, \mathrm{d}\mathbf{w}_t,$$

and the SB backward SDE simplifies to:

$$\mathrm{d}\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g^2(t) \nabla \log p_t^{\text{SB}}(\mathbf{x}_t)\right] \mathrm{d}t + g(t) \, \mathrm{d}\bar{\mathbf{w}}_t,$$

which matches Anderson's reverse-time SDE used in diffusion models. This correspondence shows that diffusion models can be interpreted as the zero-control limit of SB, where no additional drift is introduced by the potentials.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Boundary Conditions and Generality)</span></p>

The above reduction is purely formal unless the boundary constraints are compatible. For arbitrary source/target $(p_{\text{src}}, p_{\text{tgt}})$, the PDE boundary conditions in Equation (7.2.12) are not generally satisfied by the choice $\Psi \equiv 1$. Full SB resolves this by learning nontrivial potentials that induce a nonlinear control drift, bending the reference dynamics to match any prescribed endpoints. By contrast, diffusion models fix one endpoint to a simple prior (typically Gaussian) and learn only the reverse-time score to reach the data. With this perspective, SB is the more flexible umbrella: with nontrivial potentials it bridges arbitrary endpoints; with $\Psi \equiv 1$ it collapses to the diffusion-model case. We additionally remark that in the standard linear diffusion model, $p_T \approx p_{\text{prior}}$ holds only as $T \to \infty$, so the match to the prior is merely approximate.

</div>

---

#### 7.4.2 Diffusion Models as Schrodinger Half-Bridges

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Schrodinger Half-Bridges)</span></p>

The SB problem asks for a stochastic process whose law is closest (in KL divergence) to a simple reference process, while *matching two endpoint distributions* $p_{\text{src}}$ and $p_{\text{tgt}}$. Solving the full bridge requires enforcing both boundary conditions, which is often computationally difficult. A useful relaxation is the *half-bridge* problem: instead of matching both endpoints, we match only one of them.

Formally, let $R$ be the reference path distribution.

* The **forward half-bridge** seeks a path distribution $P$ minimizing $\min_{P: P_0 = p_{\text{src}}} \mathcal{D}_{\text{KL}}(P \lVert R)$, subject to the single constraint $P_0 = p_{\text{src}}$.
* The **backward half-bridge** constrains only the terminal distribution: $\min_{P: P_T = p_{\text{tgt}}} \mathcal{D}_{\text{KL}}(P \lVert R)$.

By combining these two relaxations iteratively, one can approximate the full SB.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Diffusion Models Miss Exact Endpoint Matching)</span></p>

A key difference between diffusion models and the SB framework lies in the treatment of the terminal distribution $p_T$. In standard diffusion models, the forward SDE is typically linear and designed so that $p_T$ *approximates* the prior only as $T \to \infty$:

$$p_T \approx p_{\text{prior}}.$$

At finite time, however, $p_T$ is a Gaussian whose parameters depend on $p_{\text{data}}$. As a result, it generally does not match the desired prior without careful tuning.

In contrast, the SB framework enforces exact marginal matching at a finite time $T$ by introducing an additional control drift of the form $g^2(t) \nabla_\mathbf{x} \log \Psi(\mathbf{x}, t)$. This ensures that the terminal distribution precisely satisfies $p_T = p_{\text{prior}}$, regardless of the initial data distribution $p_0 = p_{\text{data}}$. In summary:
* **Diffusion Models:** $p_T \approx p_{\text{prior}}$, asymptotically as $T \to \infty$.
* **Schrodinger Bridge:** $p_T = p_{\text{prior}}$ exactly at finite $T$, enabled by solving for the control potentials $\Psi$ and $\widehat{\Psi}$.

Standard diffusion models therefore do not enforce $P_T = p_{\text{prior}}$, and thus only solve a Schrodinger *half-bridge* from $p_{\text{data}}$ to $p_{\text{prior}}$.

</div>

**Diffusion Schrodinger Bridge.** To address this, the Diffusion Schrodinger Bridge (DSB) alternates between matching both endpoint marginals by following the idea of the Iterative Proportional Fitting (IPF) algorithm, an alternating projection method:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Diffusion Schrodinger Bridge (DSB))</span></p>

* **Step 0: Reference Process.** Initialize with $P^{(0)} := R_{\text{fwd}}$, the reference forward SDE:

$$\mathrm{d}\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t) \, \mathrm{d}t + g(t) \, \mathrm{d}\mathbf{w}_t, \quad \mathbf{x}_0 \sim p_{\text{data}}.$$

This ensures $P_0^{(0)} = p_{\text{data}}$, but typically $P_T^{(0)} \neq p_{\text{prior}}$.

* **Step 1: Backward Pass.** Compute the process $P^{(1)}$ that matches $p_{\text{prior}}$ at time $T$ while staying close to $P^{(0)}$:

$$P^{(1)} = \arg\min_{P: P_T = p_{\text{prior}}} \mathcal{D}_{\text{KL}}(P \lVert P^{(0)}).$$

This is achieved via approximating the oracle score function with a neural network $\mathbf{s}_{\phi^\times}$, which results in the reverse-time SDE:

$$\mathrm{d}\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g^2(t) \mathbf{s}_{\phi^\times}(\mathbf{x}_t, t)\right] \mathrm{d}t + g(t) \, \mathrm{d}\bar{\mathbf{w}}_t,$$

simulated backward from $\mathbf{x}_T \sim p_{\text{prior}}$.

* **Iteration.** The process $P^{(1)}$ satisfies $P_T^{(1)} = p_{\text{prior}}$, but its initial marginal $P_0^{(1)}$ typically deviates from $p_{\text{data}}$. IPF addresses this by learning a forward SDE to adjust $P_0^{(1)}$ back to $p_{\text{data}}$, followed by another backward pass to enforce $p_{\text{prior}}$. This alternation continues, refining the process until convergence to the optimal bridge $P^\ast$, which satisfies both $P_0^\ast = p_{\text{data}}$ and $P_T^\ast = p_{\text{prior}}$. De Bortoli et al. (2021) prove convergence under mild conditions.

</div>

---

### 7.5 Is Diffusion Model's ODE an Optimal Map to OT Problem?

#### 7.5.1 PF-ODE Flow Is Generally Not Optimal Transport

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Setup: PF-ODE Flow and OT)</span></p>

This section presents the result of Lavenant and Santambrogio (2022), which demonstrates that the solution map of the PF-ODE does not generally yield the optimal transport map under a quadratic cost.

**Setup.** We consider a VP SDE, specifically the Ornstein--Uhlenbeck process, which evolves a smooth initial density $p_0$ toward the standard Gaussian $\mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\mathrm{d}\mathbf{x}(t) = -\mathbf{x}(t) \, \mathrm{d}t + \sqrt{2} \, \mathrm{d}\mathbf{w}(t), \quad \mathbf{x}(0) \sim p_0.$$

The associated PF-ODE is:

$$\frac{\mathrm{d}\mathbf{S}_t(\mathbf{x})}{\mathrm{d}t} = -\mathbf{S}_t(\mathbf{x}) - \nabla \log p_t(\mathbf{S}_t(\mathbf{x})), \quad \mathbf{S}_0(\mathbf{x}) = \mathbf{x}.$$

Here, $\mathbf{S}_t$ denotes the flow map pushing forward $p_0$ to the marginal $p_t$. As $t \to \infty$, the map transports the initial distribution to the prior: $\mathbf{S}_\infty{}_\# p_0 = \mathcal{N}(\mathbf{0}, \mathbf{I})$.

**Objective of Lavenant and Santambrogio's Argument.** They construct a specific initial distribution $p_0$ and examine the entire PF-ODE trajectory. Their key observation is that optimality may fail at some point along the flow. They consider the intermediate marginal $p_{t_0} = \mathbf{S}_{t_0}{}_\# p_0$ and define the residual transport map from $p_{t_0}$ to the Gaussian as $\mathbf{T}_{t_0 \to \infty} := \mathbf{S}_\infty \circ \mathbf{S}_{t_0}^{-1}$. They show there exists a time $t_0 \ge 0$ such that $\mathbf{T}_{t_0 \to \infty}$ is *not* the quadratic-cost optimal transport map from $p_{t_0}$ to $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.1: Informal Brenier's Theorem)</span></p>

Let $\nu_1, \nu_2$ be two probability distributions on $\mathbb{R}^D$ with smooth densities. A smooth map $\mathbf{T} : \mathbb{R}^D \to \mathbb{R}^D$ is the optimal transport from $\nu_1$ to $\nu_2$ (under quadratic cost) if and only if $\mathbf{T} = \nabla u$ for some convex function $u$. In this case, $\mathrm{D}\mathbf{T}$ is symmetric and positive semi-definite, and $u$ satisfies the Monge--Ampere equation:

$$\det \mathrm{D}^2 u(\mathbf{x}) = \frac{\nu_1(\mathbf{x})}{\nu_2(\nabla u(\mathbf{x}))}.$$

A map is the optimal transport between two distributions if and only if its inverse is the optimal transport in the reverse direction.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch: PF-ODE Is Not an OT Map in General</summary>

Lavenant and Santambrogio employ a proof by contradiction: they assume that for every $t \ge 0$, the map $\mathbf{T}_t = \mathbf{S}_t \circ \mathbf{S}_\infty^{-1}$ is the quadratic-cost optimal transport map from $\mathcal{N}(\mathbf{0}, \mathbf{I})$ to $p_t$.

**Step 1: Brenier's Theorem.** By Brenier's Theorem, the Jacobian of any optimal transport map from Gaussian must be symmetric and positive semi-definite. Thus,

$$\mathrm{D}\mathbf{T}_t(\mathbf{x}) = \mathrm{D}\mathbf{S}_t(\mathbf{S}_\infty^{-1}(\mathbf{x})) \mathrm{D}(\mathbf{S}_\infty^{-1})(\mathbf{x})$$

must be symmetric for all $t$ and $\mathbf{x}$.

**Step 2: Time-Differentiating the Symmetry Condition.** Differentiating in time and using the flow ODE, one obtains that $(-\mathbf{I} - \mathrm{D}^2 \log p_t(\mathbf{S}_t)) \cdot \mathrm{D}\mathbf{S}_t \cdot \mathrm{D}(\mathbf{S}_\infty^{-1})$ is symmetric for all $t \ge 0$.

**Step 3: The Commutation Condition.** Since $\mathbf{T}_0 = \mathbf{S}_\infty^{-1}$ is assumed to be optimal, its Jacobian $D\mathbf{T}_0 = D(\mathbf{S}_\infty^{-1})$ is symmetric. Moreover, the Hessian $\mathrm{D}^2 \log p_0$ is symmetric. Two symmetric matrices multiply to a symmetric matrix if and only if they commute. Hence, for all $\mathbf{y} \in \mathbb{R}^D$:

$$\mathrm{D}^2 \log p_0(\mathbf{y}) \quad \text{must commute with} \quad \mathrm{D}\mathbf{S}_\infty(\mathbf{y}).$$

Since $\mathbf{S}_\infty$ is optimal between $p_0$ and $\mathcal{N}(\mathbf{0}, \mathbf{I})$, Brenier's theorem guarantees that $\mathbf{S}_\infty = \nabla u$ for some convex function $u$. From the Monge--Ampere equation, $\log p_0(\mathbf{y}) = \log \det(\mathrm{D}^2 u(\mathbf{y})) - \frac{1}{2}\lVert \nabla u(\mathbf{y}) \rVert^2 + \text{Constant}$. The condition becomes (with $\mathrm{D}\mathbf{S}_\infty = \mathrm{D}^2 u$):

$$\mathrm{D}^2\left(\log \det \mathrm{D}^2 u - \tfrac{1}{2}\lVert \nabla u \rVert^2\right) \quad \text{must commute with} \quad \mathrm{D}^2 u. \tag{7.5.1}$$

**Step 4: Constructing the Counterexample.** Consider $u(\mathbf{x}) = \frac{1}{2}\lVert \mathbf{x} \rVert^2 + \varepsilon \phi(\mathbf{x})$ for a small $\varepsilon$. Then $\mathrm{D}^2 u(\mathbf{0}) = \mathbf{I} + \varepsilon \mathrm{D}^2 \phi(\mathbf{0})$, and the commutation condition at $\mathbf{x} = \mathbf{0}$ requires $\mathrm{D}^2 \phi(\mathbf{0})$ to commute with $\mathrm{D}^2(\Delta \phi)(\mathbf{0})$. In $\mathbb{R}^2$, the choice $\phi(x_1, x_2) = x_1 x_2 + x_1^4$ provides a counterexample where these Hessians do not commute.

This contradiction shows that $\mathbf{T}_t$ cannot be optimal for all $t \ge 0$. Therefore, there exists some $t_0 \ge 0$ such that the map $\mathbf{T}_{t_0 \to \infty}$ is not optimal.

</details>
</div>

---

#### 7.5.2 Can Canonical Linear Flow and Reflow Lead to an OT Map?

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(7.5.1)</span></p>

*Does the linear interpolation flow $(1 - t)\mathbf{x}_0 + t\mathbf{x}_1$ with $\mathbf{x}_0 \sim p_{\text{src}}$ and $\mathbf{x}_1 \sim p_{\text{tgt}}$, when applied to the independent coupling $\pi(\mathbf{x}_0, \mathbf{x}_1) = p_{\text{src}}(\mathbf{x}_0) p_{\text{tgt}}(\mathbf{x}_1)$, recover the OT map?*

The answer to the question is no.

</div>

Nevertheless, combining a linear path with a given coupling offers a practical upper bound on the true OT cost. Among all possible paths, linear interpolation provides the tightest such upper bound.

**Canonical Linear Flow and Optimal Transport.** Focusing on optimal transport with quadratic cost, we consider the equivalent form of the Benamou--Brenier formulation (Equation (7.2.3)):

$$\mathcal{K}(p_{\text{src}}, p_{\text{tgt}}) := \min_{\substack{(p_t, \mathbf{v}_t) \text{ s.t. } \partial_t p_t + \nabla \cdot (p_t \mathbf{v}_t) = 0, \\ p_0 = p_{\text{src}}, \; p_1 = p_{\text{tgt}}}} \int_0^1 \int_{\mathbb{R}^D} \lVert \mathbf{v}_t(\mathbf{x}) \rVert^2 p_t(\mathbf{x}) \, \mathrm{d}\mathbf{x} \, \mathrm{d}t.$$

Solving this directly is typically intractable. However, Liu (2022) and Lipman et al. (2024) reveal that its kinetic energy admits a practical upper bound by restricting the search to a simpler family of *conditional flows*, where each path is defined by its fixed endpoints $(\mathbf{x}_0, \mathbf{x}_1)$ drawn from a coupling $\pi_{0,1}$ of the source and target distributions.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.5.1: An Upper Bound on OT Kinetic Energy via Conditional Flows)</span></p>

Let $\pi_{0,1}$ be any coupling between $p_{\text{src}}$ and $p_{\text{tgt}}$.

**(1)** The kinetic energy is bounded above by the expected path energy of any conditional flow $\Psi_t(\mathbf{x}_0, \mathbf{x}_1)$ that connects the endpoints:

$$\mathcal{K}(p_{\text{src}}, p_{\text{tgt}}) \le \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1) \sim \pi_{0,1}} \left[\int_0^1 \lVert \Psi_t'(\mathbf{x}_0, \mathbf{x}_1) \rVert^2 \, \mathrm{d}t\right].$$

**(2)** The unique conditional flow $\Psi_t^\ast$ that minimizes the upper bound on the right-hand side is the linear interpolation path:

$$\Psi_t^\ast(\mathbf{x}_0, \mathbf{x}_1) = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1.$$

Substituting this optimal path yields the tightest version of the bound:

$$\mathcal{K}(p_{\text{src}}, p_{\text{tgt}}) \le \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1) \sim \pi_{0,1}} \lVert \mathbf{x}_1 - \mathbf{x}_0 \rVert^2.$$

</div>

In other words, the linear interpolation $\Psi_t^\ast$ (i.e., the forward kernel used by Flow Matching and Rectified Flow) minimizes an upper bound on the true kinetic energy for any chosen coupling $\pi_{0,1}$.

We emphasize that optimality within this class of conditional flows does not guarantee global optimality on the marginal distributions.

**Reflow and Optimal Transport.** The most naive transport plan between two distributions is to connect their samples with straight lines using a simple independent coupling. However, this approach is demonstrably not optimal, as the failure lies not in the straight-line paths themselves, but in the inefficient initial pairing of points.

The Reflow procedure may offer a constructive response. It is an iterative algorithm designed specifically to correct this pairing, and crucially, each step is guaranteed to be cost-non-increasing.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(7.5.2)</span></p>

*What happens if we apply the* `Rectify` *operator iteratively? Can the resulting sequence of transport plans converge to the optimal one, or does the fixed point of the Reflow process yield the OT map?*

The short answer is no in general.

</div>

The Reflow procedure iteratively refines the coupling between $p_{\text{src}}$ and $p_{\text{tgt}}$ via the update:

$$\pi^{(k+1)} = \texttt{Rectify}(\pi^{(k)}),$$

initialized with the product coupling $\pi^{(0)} := p_{\text{src}}(\mathbf{x}_0) p_{\text{tgt}}(\mathbf{x}_1)$. More precisely, `Rectify` outputs the updated coupling $\pi^{(k+1)}$ via the following: At each iteration $k = 0, 1, 2, \dots$, a velocity field $\mathbf{v}_t^{(k)}$ is learned via:

$$\mathbf{v}_t^{(k)} \in \arg\min_{\mathbf{u}_t} \mathcal{L}(\mathbf{u}_t \vert \pi^{(k)}),$$

where $\mathcal{L}(\mathbf{u}_t \vert \pi^{(k)})$ is the loss (e.g., RF or FM loss). The updated coupling is then given by:

$$\pi^{(k+1)}(\mathbf{x}_0, \mathbf{x}_1) := p_{\text{src}}(\mathbf{x}_0) \, \delta\!\left(\mathbf{x}_1 - \Psi_1^{(k)}(\mathbf{x}_0)\right),$$

where $\Psi_1^{(k)}$ denotes the solution map at time $t = 1$ obtained by integrating $\mathbf{v}_t^{(k)}$ from initial condition $\mathbf{x}_0$.

Motivated by the Benamou--Brenier framework, Liu (2022) proposed an additional constraint: the velocity field $\mathbf{v}_t$ should be the gradient of a potential function. Accordingly, the objective is modified to restrict $\mathbf{v}_t$ to the space of gradient vector fields, also known as *potential flows*:

$$\mathbf{w}_t^{(k)} \in \arg\min_{\mathbf{u}_t : \, \mathbf{u}_t = \nabla \varphi \text{ for some } \varphi : \mathbb{R}^D \to \mathbb{R}} \mathcal{L}(\mathbf{u}_t \vert \pi^{(k)}). \tag{7.5.2}$$

We denote this associated operator as `Rectify`$_\perp$, emphasizing the projection onto irrotational vector fields. Liu, Gong, et al. (2022) conjecture the following equivalence characterizing optimality:

* *(i)* $\pi$ is an optimal transport coupling.
* *(ii)* $\pi = \texttt{Rectify}_\perp(\pi)$.
* *(iii)* There exists a gradient velocity field $\mathbf{v}_t = \nabla \varphi_t$ such that the rectify loss vanishes: $\mathcal{L}(\mathbf{v}_t \vert \pi) = 0$.

However, Hertrich et al. (2025) exhibit two types of counterexamples:

1. When the intermediate distributions $p_t$ have disconnected support, one can find fixed points of `Rectify`$_\perp$ with zero Reflow loss and gradient velocity fields that nonetheless fail to produce the optimal coupling.
2. Even when both endpoint distributions are Gaussian, there exist couplings whose loss is arbitrarily small but whose deviation from the optimal coupling is arbitrarily large.

Therefore, while rectified flows may yield strong generative models, their reliability as optimal transport solvers remains limited.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Transport Cost vs. Downstream Performance)</span></p>

Transport cost does not always correlate with downstream performance; as such, computing the exact optimal transport map may not necessarily lead to better practical outcomes. Nonetheless, variants of optimal transport remain fundamental to many problems in science and engineering. Diffusion models offer a powerful framework for exploring these challenges.

</div>

---

## Chapter 8: Guidance and Controllable Generation

Diffusion models are powerful generative frameworks. In the unconditional setting, the goal is to learn $p_{\text{data}}(\mathbf{x})$ and generate samples without external input. Many applications, however, require *conditional generation*, where outputs satisfy user-specified criteria. This can be achieved by steering an unconditional model or directly learning the conditional distribution $p_0(\mathbf{x} \vert \mathbf{c})$, with condition $\mathbf{c}$ (e.g., label, text description, or sketch) guiding the process.

### 8.1 Prologue

The generation process of diffusion models proceeds in a coarse-to-fine manner, providing a flexible framework for controllable generation. At each step, a small amount of noise is removed and the sample becomes clearer, gradually revealing more structure and detail. This property enables control over the generation process: by adding a guidance term to the learned, time-dependent velocity field, we can steer the generative trajectory to reflect user intent.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Bayesian Decomposition of the Conditional Score)</span></p>

A principled foundation for guidance-based sampling in diffusion models is the Bayesian decomposition of the conditional score. For each noise level $t$,

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}) = \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)}_{\text{unconditional direction}} + \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} \vert \mathbf{x}_t)}_{\text{guidance direction}}. \tag{8.1.1}$$

This identity shows that conditional sampling can be implemented by adding a guidance term $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} \vert \mathbf{x}_t)$ on top of the unconditional score.

</div>

Once such an approximation is available, sampling simply replaces the unconditional score with its conditional counterpart. Using Equation (8.1.1), the PF-ODE becomes

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = f(t)\mathbf{x}(t) - \frac{1}{2}g^2(t) \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}(t) \vert \mathbf{c})}_{\text{conditional score}} = f(t)\mathbf{x}(t) - \frac{1}{2}g^2(t)\Big[\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}(t)) + \nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} \vert \mathbf{x}(t))\Big]. \tag{8.1.2}$$

We highlight that steering these time-dependent vector fields fundamentally relies on their linearity, so the discussion below, formulated in score prediction, naturally extends to $\mathbf{x}$-, $\boldsymbol{\epsilon}$-, and $\mathbf{v}$-prediction through their linear relationships.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Instantiations of the Guidance Direction)</span></p>

1. **Classifier Guidance (CG).** In Section 8.2, CG trains a time-conditional classifier $p_\psi(\mathbf{c} \vert \mathbf{x}_t, t)$ on noised data $\mathbf{x}_t$. At sampling time, its input gradient provides the guidance term:

$$\nabla_{\mathbf{x}_t} \log p_{\psi^\times}(\mathbf{c} \vert \mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} \vert \mathbf{x}_t),$$

which is then added to the unconditional score.

2. **Classifier-Free Guidance (CFG).** In Section 8.3, CFG directly trains a single conditional model

$$\mathbf{s}_\phi(\mathbf{x}_t, t, \mathbf{c}) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}),$$

where the unconditional model is learned jointly by randomly replacing the condition with a special null token for a fraction of the training steps.

3. **Training-Free (Surrogate) Guidance.** The conditional $p_t(\mathbf{c} \vert \mathbf{x}_t)$ is generally intractable because it requires marginalizing over the clean latent $\mathbf{x}_0$:

$$p_t(\mathbf{c} \vert \mathbf{x}_t) = \int p(\mathbf{c} \vert \mathbf{x}_0) p(\mathbf{x}_0 \vert \mathbf{x}_t) \, \mathrm{d}\mathbf{x}_0.$$

In Section 8.4.1, training-free (loss-based) guidance avoids evaluating $p_t(\mathbf{c} \vert \mathbf{x}_t)$ directly. Instead, it introduces an off-the-shelf loss $\ell(\mathbf{x}_t, \mathbf{c}; t)$ and defines a surrogate conditional distribution $\tilde{p}_t(\mathbf{c} \vert \mathbf{x}_t)$ as

$$\tilde{p}_t(\mathbf{c} \vert \mathbf{x}_t) \propto \exp\left(-\tau \, \ell(\mathbf{x}_t, \mathbf{c}; t)\right), \quad \tau > 0,$$

which acts as a pseudo-likelihood. Its conditional score is computed solely by the gradient of the chosen loss with $\tau$:

$$\nabla_{\mathbf{x}_t} \log \tilde{p}_t(\mathbf{c} \vert \mathbf{x}_t) = -\tau \nabla_{\mathbf{x}_t} \ell(\mathbf{x}_t, \mathbf{c}; t).$$

This term is added to the unconditional score with a guidance weight $w_t$. In this view, classifier guidance is simply surrogate guidance with a learned classifier $\tilde{p}_t(\mathbf{c} \vert \mathbf{x}_t) := p_{\psi^\times}(\mathbf{c} \vert \mathbf{x}_t, t)$ via $\ell(\mathbf{x}_t, \mathbf{c}; t) = -\log p_{\psi^\times}(\mathbf{c} \vert \mathbf{x}_t, t)$, $\tau = 1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Guided PF-ODE Does Not Sample from the Tilted Family)</span></p>

Guided PF-ODE does *not* sample from the tilted family (in general). Even with exact scores and exact ODE integration, replacing the score by the *tilted* score does not make the time-$t$ marginals equal to $\lbrace \tilde{p}_t^{\text{tilt}}(\cdot \vert \mathbf{c}) \rbrace_{t \in [0,1]}$, nor the terminal law equal to $\tilde{p}_0^{\text{tilt}}(\cdot \vert \mathbf{c})$.

Define $\mathbf{v}_t^{\text{orig}} = \mathbf{f} - \frac{1}{2}g^2(t)\nabla \log p_t$, $\mathbf{h}_t(\mathbf{x}) = e^{-w_t \tau \ell(\mathbf{x}, \mathbf{c}; t)}$, and $\tilde{p}_t^{\text{tilt}} = \frac{p_t \mathbf{h}_t}{Z_t}$. The guided PF-ODE uses $\mathbf{v}_t^{\text{tilt}} = \mathbf{v}_t^{\text{orig}} - \frac{1}{2}g^2(t)\nabla \log \mathbf{h}_t$. If $\tilde{p}_t^{\text{tilt}}$ were the true marginals, they would satisfy $\partial_t \tilde{p}_t^{\text{tilt}} + \nabla \cdot (\tilde{p}_t^{\text{tilt}} \mathbf{v}_t^{\text{tilt}}) = 0$. But a direct calculation gives the residual

$$\partial_t \tilde{p}_t^{\text{tilt}} + \nabla \cdot (\tilde{p}_t^{\text{tilt}} \mathbf{v}_t^{\text{tilt}}) = \tilde{p}_t^{\text{tilt}}\Big[\partial_t \log \mathbf{h}_t + \mathbf{v}_t^{\text{orig}} \cdot \nabla \log \mathbf{h}_t - \tfrac{1}{2}g^2(t)(\Delta \log \mathbf{h}_t + \lVert \nabla \log \mathbf{h}_t \rVert^2) - \tfrac{Z_t'}{Z_t}\Big].$$

This vanishes for all $\mathbf{x}$ if and only if $\omega_t \equiv 0$ (unconditional generation) or in very special cases of $w_t$ or $\ell$. Therefore, in general, $\lbrace \tilde{p}_t^{\text{tilt}} \rbrace$ are *not* the PF-ODE marginals, and terminal samples are *not* distributed as $\tilde{p}_0^{\text{tilt}}(\mathbf{x}_0 \vert \mathbf{c})$.

</div>

**From Control to Better Alignment with Direct Preference Optimization.** Strong control can be on-condition but off-preference: a sample may satisfy the conditioning signal (e.g., the prompt) yet deviate from what humans actually prefer. We formalize this by *tilting* the conditional target by a preference rating:

$$\tilde{p}_0^{\text{tilt}}(\mathbf{x}_0 \vert \mathbf{c}) \propto p_0(\mathbf{x}_0 \vert \mathbf{c}) \exp\!\left(\beta r(\mathbf{x}_0, \mathbf{c})\right),$$

where $r(\mathbf{x}_0, \mathbf{c})$ is a scalar alignment rating (reward) for a clean sample $\mathbf{x}_0$ and condition $\mathbf{c}$ (larger $r$ indicates better alignment). In practice, $r$ may be (i) the logit or log-probability of an external reward/classifier, (ii) a similarity measure (e.g., CLIP/perceptual), or (iii) a learned preference model. Existing methods typically collect human labels of the relative quality of model generations and fine-tune the conditional diffusion model to align with these preferences, often through reinforcement learning from human feedback (RLHF). However, RLHF is complex and often unstable. This motivates *Diffusion-DPO*, an adaptation of Direct Preference Optimization that learns the preference tilt directly from pairwise choices, so the conditional diffusion model is fine-tuned to align to preferences without a separate reward model (see Section 8.5).

---

### 8.2 Classifier Guidance

#### 8.2.1 Foundation of Classifier Guidance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Classifier Guidance)</span></p>

Let $\mathbf{c}$ denote a conditioning variable drawn from a distribution $p(\mathbf{c})$, such as a class label, caption, or other auxiliary information. Our goal is to draw samples from $p_0(\mathbf{x} \vert \mathbf{c})$.

The conditional score can be decomposed via Bayes' rule as:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}) = \nabla_{\mathbf{x}_t} \log \left(\frac{p_t(\mathbf{x}_t) p_t(\mathbf{c} \vert \mathbf{x}_t)}{p(\mathbf{c})}\right) = \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)}_{\text{unconditional score}} + \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} \vert \mathbf{x}_t)}_{\text{classifier gradient}}. \tag{8.2.1}$$

This decomposition motivates *Classifier Guidance* (CG), proposed by Dhariwal and Nichol (2021), which leverages a pre-trained time-dependent classifier $p_t(\mathbf{c} \vert \mathbf{x}_t)$ to steer the generation process. Specifically, we define a one-parameter family of *guided densities* (tilted conditionals) with guidance scale $\omega \ge 0$:

$$p_t(\mathbf{x}_t \vert \mathbf{c}, \omega) \propto p_t(\mathbf{x}_t) p_t(\mathbf{c} \vert \mathbf{x}_t)^\omega, \tag{8.2.2}$$

which yields the score function:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}, \omega) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \omega \nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} \vert \mathbf{x}_t). \tag{8.2.3}$$

Geometrically, this tilts the unconditional flow in the direction that increases the class likelihood. The scalar $\omega \ge 0$ modulates the influence of the classifier:

* $\omega = 1$: recovers the true conditional score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c})$.
* $\omega > 1$: amplifies the classifier signal, typically increasing conditional fidelity (often at the expense of diversity).
* $0 \le \omega < 1$: down-weights the classifier signal, typically increasing sample diversity while weakening conditioning.

</div>

**Practical Approximation in CG.** In practice, CG is a training-free method (w.r.t. the diffusion model) for steering a pre-trained unconditional diffusion model, $\mathbf{s}_{\phi^\times}(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$. CG is applied only at sampling time, without modifying the diffusion model itself. To enable this, a time-dependent classifier $p_\psi(\mathbf{c} \vert \mathbf{x}_t, t)$ is trained separately to predict the condition $\mathbf{c}$ from noisy inputs $\mathbf{x}_t$ at different noise levels $t$. The classifier is trained in a standard way by minimizing the cross-entropy loss:

$$\mathbb{E}_{t \sim \mathcal{U}[0,T], (\mathbf{x}, \mathbf{c}) \sim p_{\text{data}}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}\left[-\log p_\psi(\mathbf{c} \vert \mathbf{x}_t, t)\right], \tag{8.2.4}$$

where $(\mathbf{x}, \mathbf{c}) \sim p_{\text{data}}$ denotes paired labeled data, and $\mathbf{x}_t = \alpha_t \mathbf{x} + \sigma_t \boldsymbol{\epsilon}$ is the noisy input at time $t$. The classifier must be explicitly conditioned on $t$ (e.g., via time embeddings), since it is expected to operate reliably across all noise levels.

#### 8.2.2 Inference with CG

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(CG Inference)</span></p>

At inference time, the classifier gradient $\nabla_{\mathbf{x}_t} \log p_{\psi^\times}(\mathbf{c} \vert \mathbf{x}_t, t)$ is added to the unconditional score function and scaled by a guidance weight $\omega$, yielding an approximation to the guided score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}, \omega)$ from Equation (8.2.3):

$$\mathbf{s}^{\text{CG}}(\mathbf{x}_t, t, \mathbf{c}; \omega) := \underbrace{\mathbf{s}_{\phi^\times}(\mathbf{x}_t, t)}_{\text{uncond. direction}} + \omega \underbrace{\nabla_{\mathbf{x}_t} \log p_{\psi^\times}(\mathbf{c} \vert \mathbf{x}_t, t)}_{\text{guidance direction}} \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}, \omega).$$

Accordingly, one simply replaces the unconditional score function $\mathbf{s}_{\phi^\times}(\mathbf{x}_t, t)$ in the reverse-time SDE or PF-ODE with the guided score $\mathbf{s}^{\text{CG}}(\mathbf{x}_t, t, \mathbf{c}; \omega)$ for a specified $\omega$ as in Equation (8.1.2), thereby steering the generative trajectory toward samples that align with the condition $\mathbf{c}$.

</div>

#### 8.2.3 Advantages and Limitations

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(CG: Advantages and Limitations)</span></p>

CG provides a simple and flexible mechanism for conditional generation, allowing for explicit control over the strength of conditioning via $\omega$. It can be used with any pre-trained unconditional diffusion model, requiring only an additional classifier for conditioning.

However, the approach has notable limitations:

* **Training Cost:** The classifier must be trained to operate across all noise levels, which is computationally expensive.
* **Robustness:** Classifiers must generalize well to severely corrupted inputs $\mathbf{x}_t$, especially for large $t$, which can be challenging.
* **Separate Training:** Since the classifier is trained independently of the diffusion model, it may not align perfectly with the learned data distribution.

</div>

---

### 8.3 Classifier-Free Guidance

#### 8.3.1 Foundation of Classifier-Free Guidance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Classifier-Free Guidance (CFG))</span></p>

*Classifier-free guidance* (CFG) (Ho and Salimans, 2021) is a simplified approach to classifier-based guidance that eliminates the need for a separate classifier. The key idea is to modify the gradient of the score function in a way that allows for effective conditioning without explicit classifiers. Specifically, the gradient of the log-probability of the conditional distribution is adjusted as follows:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} \vert \mathbf{x}_t) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t). \tag{8.3.1}$$

Substituting this expression into Equation (8.2.3) yields the following formulation for the conditioned score:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}, \omega) = \omega \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c})}_{\text{conditional score}} + (1 - \omega) \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)}_{\text{unconditional score}}. \tag{8.3.2}$$

The hyperparameter $\omega$ again plays a critical role in controlling the influence of the conditioning information (we take $\omega \ge 0$):

* At $\omega = 0$, the model behaves as an unconditional diffusion model, completely ignoring the conditioning.
* At $\omega = 1$, the model uses the conditional score without additional guidance.
* For $\omega > 1$, the model places more emphasis on the conditional score and less on the unconditional score, strengthening alignment with $\mathbf{c}$ but typically reducing diversity.

</div>

#### 8.3.2 Training and Sampling of CFG

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Joint Training of Unconditional and Conditional Models via CFG)</span></p>

Unlike CG, CFG requires retraining a diffusion model that explicitly accounts for the conditioning variable $\mathbf{c}$. Training two separate models for the conditional and unconditional score functions, however, is often computationally prohibitive. To address this, CFG adopts a single model $\mathbf{s}_\phi(\mathbf{x}_t, t; \mathbf{c})$ that learns both score functions within a single model by treating $\mathbf{c}$ as an additional input. The training procedure is defined as follows:

* For unconditional training, a null token $\emptyset$ is passed in place of the conditioning input, yielding $\mathbf{s}_\phi(\mathbf{x}_t, t, \emptyset)$.
* For conditional training, the true conditioning variable $\mathbf{c}$ is provided as input, resulting in $\mathbf{s}_\phi(\mathbf{x}_t, t, \mathbf{c})$.

These two training regimes are unified by randomly replacing $\mathbf{c}$ with the null input $\emptyset$ with probability $p_{\text{uncond}}$ (a user-defined hyperparameter typically set to 0.1). This joint training strategy enables the model to simultaneously learn both conditional and unconditional score functions. We remark that during training, the CFG weight $\omega$ is not utilized.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(CFG for Conditional Diffusion Models)</span></p>

**Training (Algorithm 4):**

**Input:** $p_{\text{uncond}}$: probability of unconditional dropout.
1. **Repeat**
2. $\quad (\mathbf{x}, \mathbf{c}) \sim p_{\text{data}}(\mathbf{x}, \mathbf{c})$
3. $\quad \mathbf{c} \leftarrow \emptyset$ with probability $p_{\text{uncond}}$
4. $\quad t \sim \mathcal{U}[0, T]$
5. $\quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
6. $\quad \mathbf{x}_t = \alpha_t \mathbf{x} + \sigma_t \boldsymbol{\epsilon}$
7. $\quad$ Take gradient step on: $\nabla_\phi \lVert \mathbf{s}_\phi(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{s} \rVert^2$
8. **until** converged

</div>

**Conditioned Sampling with CFG.** Once the model $\mathbf{s}_{\phi^\times}(\mathbf{x}_t, t, \mathbf{c})$ is trained, the CFG can be applied during sampling. The gradient of the log-probability is given by:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}, \omega) = \omega \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{c}) + (1 - \omega) \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$$

$$\approx \omega \underbrace{\mathbf{s}_{\phi^\times}(\mathbf{x}_t, t, \mathbf{c})}_{\text{conditional score}} + (1 - \omega) \underbrace{\mathbf{s}_{\phi^\times}(\mathbf{x}_t, t, \emptyset)}_{\text{unconditional score}} =: \mathbf{s}_{\phi^\times}^{\text{CFG}}(\mathbf{x}_t, t, \mathbf{c}; \omega). \tag{8.3.3}$$

During sampling, a fixed (or optionally time-dependent) classifier-free guidance weight $\omega$ is applied. The unconditional score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$ in the reverse-time SDE or PF-ODE is then replaced by the guided score $\mathbf{s}_{\phi^\times}^{\text{CFG}}(\mathbf{x}_t, t, \mathbf{c}; \omega)$ as in Equation (8.1.2), which combines conditional and unconditional scores in a weighted manner.

This formulation enables controllable generation by adjusting $\omega$, allowing samples to be guided toward the conditioning signal $\mathbf{c}$ while retaining diversity. CFG thus offers an effective and computationally efficient way to achieve precise conditional generation, as it requires training only a single diffusion model.

---

### 8.4 (Optional) Training-Free Guidance

In this section, we present the high-level philosophy underlying a wide range of training-free guidance methods. Despite variations in implementation and application, these methods are unified by the central principle expressed in Equation (8.1.1).

#### 8.4.1 Conceptual Framework for Training-Free Guidance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Training-Free Guidance Setup)</span></p>

**Setup and Notations.** Let $\mathbf{c}$ denote a conditioning variable. We assume access to a pre-trained diffusion model $\mathbf{s}_{\phi^\times}(\mathbf{x}_t, t)$ expressed in score prediction. In addition, suppose we are given a non-negative function

$$\ell(\cdot, \mathbf{c}) : \mathbb{R}^D \to \mathbb{R}_{\ge 0}$$

that quantifies how well a sample $\mathbf{x} \in \mathbb{R}^D$ aligns with the condition $\mathbf{c}$, where smaller values of $\ell(\mathbf{x}, \mathbf{c})$ indicate stronger alignment. Concrete examples include: (i) $\mathbf{c}$ is a reference image, and $\ell(\cdot, \mathbf{c})$ is a similarity score measuring perceptual closeness; (ii) $\ell(\cdot, \mathbf{c})$ is a feature-based similarity score computed via a pre-trained model such as CLIP.

Consider the standard linear-Gaussian forward noising kernel $p_t(\cdot \vert \mathbf{x}_0) := \mathcal{N}(\cdot; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$. We recall the DDIM update and take it as an example:

$$\mathbf{x}_{t \to t-1} = \alpha_{t-1} \underbrace{\hat{\mathbf{x}}_0(\mathbf{x}_t)}_{\text{in data space}} - \sigma_{t-1} \sigma_t \underbrace{\hat{\mathbf{s}}(\mathbf{x}_t)}_{\text{in noise space}}, \tag{8.4.1}$$

where $\hat{\mathbf{x}}_0(\mathbf{x}_t) := \mathbf{x}_{\phi^\times}(\mathbf{x}_t, t)$ is the (clean) $\mathbf{x}$-prediction, and $\hat{\mathbf{s}}(\mathbf{x}_t) := \mathbf{s}_{\phi^\times}(\mathbf{x}_t, t)$ as the score-prediction from $\mathbf{x}_t$ at time level $t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Guidance in Data Space vs. Noise Space)</span></p>

Most training-free guidance methods introduce corrections either in the *data space* or the *noise space* to steer the DDIM update in Equation (8.4.1) toward satisfying the condition $\mathbf{c}$:

$$\mathbf{x}_{t \to t-1} = \alpha_{t-1}\underbrace{\left(\hat{\mathbf{x}}_0(\mathbf{x}_t) + \eta_t^{\text{data}} \mathcal{G}_0\right)}_{\text{A. data space}} - \sigma_{t-1}\sigma_t \underbrace{\left(\hat{\mathbf{s}}(\mathbf{x}_t) + \eta_t^{\text{latent}} \mathcal{G}_t\right)}_{\text{B. noise space}}, \tag{8.4.2}$$

where $\eta_t^{\text{data}}, \eta_t^{\text{latent}} \ge 0$ are time-dependent guidance strengths, and $\mathcal{G}_0$, $\mathcal{G}_t$ are correction terms defined below.

**A. Guidance in Data Space.** By descending along the negative gradient direction $\mathcal{G}_0 := -\nabla_{\mathbf{x}_0} \ell(\mathbf{x}_0, \mathbf{c})$, the modified clean estimate $\hat{\mathbf{x}}_0(\mathbf{x}_t) + \eta_t^{\text{data}} \mathcal{G}_0$ can be gradually steered toward samples that better satisfy the condition $\mathbf{c}$. This gradient-descent scheme can be applied iteratively to progressively improve alignment. Representative examples include MPGD and UGD.

**B. Guidance in Noise Space.** The conditional score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} \vert \mathbf{x}_t)$ is generally intractable. A practical approximation is to introduce a surrogate likelihood $\tilde{p}_t(\mathbf{c} \vert \mathbf{x}_t)$:

$$\tilde{p}_t(\mathbf{c} \vert \mathbf{x}_t) \propto \exp\left(-\eta \, \ell(\hat{\mathbf{x}}_0(\mathbf{x}_t), \mathbf{c})\right)$$

with a re-scaling constant $\eta > 0$ so that

$$\nabla_{\mathbf{x}_t} \log \tilde{p}_t(\mathbf{c} \vert \mathbf{x}_t) = -\eta \nabla_{\mathbf{x}_t} \ell(\hat{\mathbf{x}}_0(\mathbf{x}_t), \mathbf{c}) =: \mathcal{G}_t,$$

where $\hat{\mathbf{x}}_0(\mathbf{x}_t)$ is obtained via the diffusion model's prediction. This serves as the correction in the noise space. However, evaluating $\mathcal{G}_t$ requires backpropagation through the $\mathbf{x}$-prediction, i.e., $\nabla_{\mathbf{x}_t} \hat{\mathbf{x}}_0(\mathbf{x}_t)^\top \cdot \nabla_{\mathbf{x}_0} \log \ell_c(\mathbf{x}_0) \vert_{\mathbf{x}_0 = \hat{\mathbf{x}}_0(\mathbf{x}_t)}$, which may result in substantial computational cost.

</div>

#### 8.4.2 Examples of Training-Free Approaches to Inverse Problems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Inverse Problem Setup)</span></p>

Let $\mathcal{A}$ be a corruption operator (which may be linear or nonlinear, known or unknown), such as a blurring kernel or inpainting, and let $\mathbf{y}$ be an observation generated by the following corruption model:

$$\mathbf{y} = \mathcal{A}(\mathbf{x}_0) + \sigma_\mathbf{y} \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{8.4.3}$$

The objective of inverse problems is to sample from the posterior distribution $p_0(\mathbf{x}_0 \vert \mathbf{y})$, where there may exist infinitely many reconstructions $\mathbf{x}_0$ corresponding to the given observation $\mathbf{y}$. The goal is to recover an $\mathbf{x}_0$ that removes the corruptions in $\mathbf{y}$ while preserving its faithful and semantic features.

</div>

**Pre-Trained Diffusion Models as Inverse Problems Solvers.** The conditional score can be decomposed via Bayes' rule:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \vert \mathbf{y}) = \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)}_{\text{data score}} + \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert \mathbf{x}_t)}_{\text{measurement alignment}}. \tag{8.4.4}$$

This decomposition separates the data score and a measurement alignment term with $\mathbf{y}$ specific to the inverse problem. It enables solving Equation (8.4.3) in an unsupervised manner by modeling the clean data distribution $p_{\text{data}}$ and applying it during inversion:

* **Data score** $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$: Approximated using a pre-trained diffusion model $\mathbf{s}_{\phi^\times}(\mathbf{x}_t, t)$ trained on clean data.
* **Measurement alignment** $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert \mathbf{x}_t)$: Intractable in closed form, as it involves marginalizing over latent variables.

Most training-free approaches focus on approximating $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert \mathbf{x}_t)$. A common meta-form is:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert \mathbf{x}_t) \approx -\frac{\mathcal{P}_t \, \mathcal{M}_t}{\gamma_t},$$

where $\mathcal{M}_t$ is an error vector quantifying the mismatch between the observation $\mathbf{y}$ and the estimated signal, $\mathcal{P}_t$ is a mapping that projects $\mathcal{M}_t$ back to the ambient space of $\mathbf{x}_t$, and $\gamma_t$ is a scalar controlling the guidance strength.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Instantiations of Diffusion-Based Inverse Problem Solvers)</span></p>

Representative methods that leverage a pre-trained diffusion model to provide unsupervised approaches (requiring no paired data) for inverse problems:

**Score SDE (Song et al., 2020c).** Considers a known linear corruption model $\mathbf{A}$ with $\sigma_\mathbf{y} = 0$. Since $\mathbf{A}$ is linear, one can form a noise-level--matched observation $\mathbf{y}_t := \alpha_t \mathbf{y} + \sigma_t \boldsymbol{\epsilon}$, and use the residual $\mathbf{y}_t - \mathbf{A}\mathbf{x}_t$ (note: $\mathbf{y}_t \neq \mathbf{A}\mathbf{x}_t$ in general) to drive a likelihood-style correction. A common approximation is:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert \mathbf{x}_t) \approx -\mathbf{A}^\top (\mathbf{y}_t - \mathbf{A}\mathbf{x}_t).$$

**Iterative Latent Variable Refinement (ILVR) (Choi et al., 2021).** Using the same setup as ScoreSDE's case, ILVR estimates:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert \mathbf{x}_t) \approx -\mathbf{A}^\dagger(\mathbf{y}_t - \mathbf{A}\mathbf{x}_t) = -(\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top (\mathbf{y}_t - \mathbf{A}\mathbf{x}_t),$$

where $\mathbf{A}^\dagger$ is the Moore--Penrose pseudoinverse.

**Diffusion Posterior Sampling (DPS) (Chung et al., 2022).** A widely used method for inverse problems with known nonlinear forward operator $\mathcal{A}$ and additive Gaussian noise level $\sigma_\mathbf{y} \ge 0$. DPS approximates

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert \mathbf{x}_t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert X_0 = \hat{\mathbf{x}}_0(\mathbf{x}_t)), \tag{8.4.5}$$

where $\hat{\mathbf{x}}_0(\mathbf{x}_t) := \mathbb{E}[\mathbf{x}_0 \vert \mathbf{x}_t]$ denotes the conditional mean of the clean sample given the noisy observation $\mathbf{x}_t$ at time $t$, typically estimated using Tweedie's formula from a pre-trained diffusion model.

This one-point approximation assumes that $p(\mathbf{x}_0 \vert \mathbf{x}_t)$ is sharply concentrated around its mean. Since $p_t(\mathbf{y} \vert X_0 = \hat{\mathbf{x}}_0(\mathbf{x}_t)) = \mathcal{N}(\mathbf{y}; \mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t)), \sigma_\mathbf{y}^2 \mathbf{I})$, we compute

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert \mathbf{x}_t) \approx \frac{1}{\sigma_\mathbf{y}^2}\left[\mathcal{J}_\mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t)) \cdot \nabla_{\mathbf{x}_t} \hat{\mathbf{x}}_0(\mathbf{x}_t)\right]^\top \left(\mathbf{y} - \mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t))\right),$$

where $\mathcal{J}_\mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t)) := \nabla_{\mathbf{x}_0} \mathcal{A}(\mathbf{x}) \vert_{\mathbf{x} = \hat{\mathbf{x}}_0(\mathbf{x}_t)}$ denotes the Jacobian of the forward operator. This formula propagates the gradient through the score approximation pipeline, reflecting how the measurement likelihood changes with respect to perturbations in the noisy sample $\mathbf{x}_t$.

For linear inverse problems, this further simplifies to:

$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} \vert \mathbf{x}_t) \approx \frac{1}{\sigma_\mathbf{y}^2}\left[\mathbf{A} \cdot \nabla_{\mathbf{x}_t} \hat{\mathbf{x}}_0(\mathbf{x}_t)\right]^\top \left(\mathbf{y} - \mathbf{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t))\right).$$

</div>

---

### 8.5 From Reinforcement Learning to Direct Preference Optimization for Model Alignment

#### 8.5.1 The Motivation: Circumventing the Pitfalls of RLHF

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(RLHF Pipeline and Its Limitations)</span></p>

The goal of alignment is to steer a base, pre-trained model (e.g., an SFT model) toward outputs that humans prefer. RLHF proceeds in three stages:

1. **Supervised fine-tuning (SFT)** trains a base model on prompt--response pairs.
2. **Reward modeling (RM)** fits a model on preference data consisting of prompts $\mathbf{c}$ and paired responses (a preferred "winner" $\mathbf{x}_w$ and a dispreferred "loser" $\mathbf{x}_l$), learning a scalar $r(\mathbf{c}, \mathbf{x})$ with $r(\mathbf{c}, \mathbf{x}_w) > r(\mathbf{c}, \mathbf{x}_l)$.
3. **RL fine-tuning** optimizes the SFT model (policy $\pi$) with an algorithm such as PPO, maximizing expected reward from $r$ while regularizing by a KL penalty that keeps $\pi$ close to the reference/SFT distribution.

Despite its impact, this pipeline suffers from drawbacks: the RL stage is unstable and computationally expensive because it is on-policy; it also requires training and hosting multiple large models (SFT, reward, and sometimes a value model); and it optimizes only a proxy for human preferences, so flaws in the reward model can be exploited.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(8.5.1)</span></p>

*Can we eliminate explicit reward modeling and the unstable RL step, directly optimizing the model on preference data?*

</div>

#### 8.5.2 RLHF: Bradley--Terry View

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reward Model Training via Bradley--Terry)</span></p>

RLHF begins with a learned judge: a reward model $r_\psi$ that assigns a scalar preference score to candidate responses for the same prompt $\mathbf{c}$. The dataset $\mathcal{D}$ consists of pairs $(\tilde{\mathbf{x}}, \mathbf{x})$ annotated with a label $y$ indicating whether $\tilde{\mathbf{x}}$ is preferred over $\mathbf{x}$. The training objective is a simple logistic loss

$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(\mathbf{c}, \tilde{\mathbf{x}}, \mathbf{x}, y) \sim \mathcal{D}}\Big[y \log \sigma(r_\psi(\mathbf{c}, \tilde{\mathbf{x}}) - r_\psi(\mathbf{c}, \mathbf{x})) + (1 - y)\log(1 - \sigma(r_\psi(\mathbf{c}, \tilde{\mathbf{x}}) - r_\psi(\mathbf{c}, \mathbf{x})))\Big], \tag{8.5.1}$$

where $\sigma(u) = 1/(1 + e^{-u})$. Under the standard convention where $\mathcal{D}$ stores pairs in ordered format (winner, loser), with $y = 1$, the loss reduces to:

$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(\mathbf{c}, \mathbf{x}_w, \mathbf{x}_l) \sim \mathcal{D}}\left[\log \sigma(r_\psi(\mathbf{c}, \mathbf{x}_w) - r_\psi(\mathbf{c}, \mathbf{x}_l))\right]. \tag{8.5.2}$$

This is interpreted through the **Bradley--Terry (BT) model**, $p_{r_\psi}(\tilde{\mathbf{x}} \succ \mathbf{x} \vert \mathbf{c}) := \sigma(r_\psi(\mathbf{c}, \tilde{\mathbf{x}}) - r_\psi(\mathbf{c}, \mathbf{x}))$, which converts two scalar scores into a win probability. Minimizing the logistic loss is equivalent to minimizing the KL divergence between the empirical Bernoulli distribution of human labels and the model's predicted Bernoulli distribution.

</div>

**KL Regularized Policy Optimization (with Fixed Reward).** With the fitted reward $r := r_{\psi^\times}$, RLHF then adjusts a learnable policy $\pi_\theta(\mathbf{x} \vert \mathbf{c})$, usually fine-tuned on top of $p_{\phi^\times}(\mathbf{x} \vert \mathbf{c})$, toward higher-reward responses. At the same time, the policy is regularized to stay close to a reference model, taken as the pre-trained diffusion model $\pi_{\text{ref}}(\mathbf{x} \vert \mathbf{c}) := p_{\phi^\times}(\mathbf{x} \vert \mathbf{c})$, using a $\mathcal{D}_{\text{KL}}$ penalty:

$$\max_\theta \mathbb{E}_{\mathbf{c} \sim p(\mathbf{c})}\Big[\mathbb{E}_{\mathbf{x} \sim \pi_\theta(\cdot \vert \mathbf{c})}[r_\psi(\mathbf{c}, \mathbf{x})] - \beta \mathcal{D}_{\text{KL}}(\pi_\theta(\cdot \vert \mathbf{c}) \lVert \pi_{\text{ref}}(\cdot \vert \mathbf{c}))\Big]. \tag{8.5.4}$$

In summary, RLHF proceeds in two stages: first fit the reward $r^\ast$ by minimizing the loss in Equation (8.5.2); then optimize the policy $\pi^\ast$ by solving Equation (8.5.4).

#### 8.5.3 DPO Framework

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Direct Preference Optimization (DPO))</span></p>

**The Bridge from RLHF.** The KL-regularized policy objective in Equation (8.5.4) has a simple closed-form solution for each prompt $\mathbf{c}$, given the fitted reward $r := r_{\psi^\times}$, expressed in the following energy-based form:

$$\pi^\ast(\mathbf{x} \vert \mathbf{c}) = \frac{1}{Z(\mathbf{c})} \pi_{\text{ref}}(\mathbf{x} \vert \mathbf{c}) \exp(r(\mathbf{c}, \mathbf{x}) / \beta), \tag{8.5.5}$$

where $\pi_{\text{ref}}(\mathbf{x} \vert \mathbf{c}) := p_{\phi^\times}(\mathbf{x} \vert \mathbf{c})$, and $Z(\mathbf{c})$ is the partition function ensuring $\int \pi^\ast(\mathbf{x} \vert \mathbf{c}) \, \mathrm{d}\mathbf{x} = 1$.

For smaller $\beta$, $\exp(r/\beta)$ becomes sharper, so $\pi^\ast$ concentrates on high reward regions: reward dominates, the policy moves farther from $\pi_{\text{ref}}$, diversity decreases, and training may become unstable or prone to reward hacking. For larger $\beta$, $\exp(r/\beta)$ flattens, keeping $\pi^\ast$ closer to $\pi_{\text{ref}}$: the KL term dominates, updates are conservative, diversity follows the reference, but reward gains are limited.

**Defining an Implicit Reward.** Since our aim is to fine-tune the policy directly (without training a separate reward model), Equation (8.5.5) lets us *define* an *implicit reward* from any policy. For any policy $\pi$ (with support contained in $\pi_{\text{ref}}$), define

$$r_\pi(\mathbf{c}, \mathbf{x}) = \beta \log \frac{\pi(\mathbf{x} \vert \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x} \vert \mathbf{c})} + \beta \log Z(\mathbf{c}). \tag{8.5.6}$$

Then Equation (8.5.5) holds with $\pi$ in place of $\pi^\ast$, i.e., $\pi$ would be the optimizer of Equation (8.5.4) for the reward function $r_\pi$. In this sense, $r_\pi$ is an implicit *(policy-induced) reward*: it is identified up to the prompt-dependent constant $\beta \log Z(\mathbf{c})$, which vanishes in any pairwise comparison such as in the BT model:

$$r_\pi(\mathbf{c}, \mathbf{x}_w) - r_\pi(\mathbf{c}, \mathbf{x}_l) = \beta\left(\log \frac{\pi(\mathbf{x}_w \vert \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_w \vert \mathbf{c})} - \log \frac{\pi(\mathbf{x}_l \vert \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_l \vert \mathbf{c})}\right).$$

**DPO's Training Loss.** Plug the implicit reward Equation (8.5.6) into the BT model of Equation (8.5.2) for a labeled pair $(\mathbf{x}_w, \mathbf{x}_l)$ under the same prompt $\mathbf{c}$. The constants $\log Z(\mathbf{c})$ cancel between winner and loser, yielding a single logistic-loss objective on log-probability differences:

$$\mathcal{L}_{\text{DPO}}(\boldsymbol{\theta}; \pi_{\text{ref}}) = -\mathbb{E}_{(\mathbf{c}, \mathbf{x}_w, \mathbf{x}_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta\Big(\log \frac{\pi_\theta(\mathbf{x}_w \vert \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_w \vert \mathbf{c})} - \log \frac{\pi_\theta(\mathbf{x}_l \vert \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_l \vert \mathbf{c})}\Big)\right)\right].$$

In words: DPO pushes up the (temperature-scaled) advantage of the winner over the loser, measured as the difference of log-likelihood improvements over the reference. This achieves the goal of RLHF in a single, stable maximum-likelihood--style stage, without training an explicit reward model.

</div>

#### 8.5.4 Diffusion-DPO

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Naive DPO Fails for Diffusion Models)</span></p>

Evaluating the sample likelihood $\pi_\theta(\mathbf{x} \vert \mathbf{c})$ in diffusion models requires the instantaneous change-of-variables formula (divergence of the drift) of ODE solving, which is computationally intensive. Moreover, differentiating through the entire sampling trajectory can suffer from vanishing or exploding gradients. To avoid these issues, Diffusion-DPO works at the *path* level. We take the discrete-time diffusion model (e.g., DDPM) as an illustrative example; the continuous-time diffusion model is analogous.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Diffusion-DPO)</span></p>

**Defining Pathwise Implicit Rewards.** Let a trajectory be $\mathbf{x}_{0:T} := (\mathbf{x}_T, \dots, \mathbf{x}_0)$ under the reverse-time Markov chain with conditionals $\pi(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{c})$. Here, $\mathbf{x}_T$ denotes a sample from the prior (highest noise), and $\mathbf{x}_0$ is the clean output in data space. We assign each trajectory a reward $R(\mathbf{c}, \mathbf{x}_{0:T})$. The optimizer for each prompt $\mathbf{c}$ has the simple energy-based form

$$\pi^\ast(\mathbf{x}_{0:T} \vert \mathbf{c}) = \frac{1}{Z(\mathbf{c})} \pi_{\text{ref}}(\mathbf{x}_{0:T} \vert \mathbf{c}) \exp\!\left(R(\mathbf{c}, \mathbf{x}_{0:T}) / \beta\right), \tag{8.5.7}$$

with $Z(\mathbf{c})$ a normalizer. Inverting Equation (8.5.7) motivates the definition of an *implicit path reward* for any policy $\pi$:

$$R_\pi(\mathbf{c}, \mathbf{x}_{0:T}) := \beta \log \frac{\pi(\mathbf{x}_{0:T} \vert \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_{0:T} \vert \mathbf{c})} + \beta \log Z(\mathbf{c}).$$

Applying the Bradley--Terry model to paths for a labeled pair $(\mathbf{x}_0^w, \mathbf{x}_0^l)$ under the same prompt $\mathbf{c}$, and using the standard logistic log-loss yields the Diffusion-DPO loss:

$$\mathcal{L}_{\text{Diff-DPO}}(\boldsymbol{\theta}; \pi_{\text{ref}}) := -\mathbb{E}_{(\mathbf{c}, \mathbf{x}_0^w, \mathbf{x}_0^l) \sim \mathcal{D}}\left[\log \sigma\!\left(\Delta R(\mathbf{c}; \boldsymbol{\theta})\right)\right], \tag{8.5.8}$$

where $\Delta R(\mathbf{c}; \boldsymbol{\theta})$ involves expectations over latent denoising trajectories conditioned on each endpoint.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Challenges of Equation (8.5.8))</span></p>

Equation (8.5.8) is impractical for three reasons:

1. **Endpoint Conditioning Induces an Intractable Path Posterior.** The term $\mathbb{E}_{\pi_\theta(\mathbf{x}_{1:T} \vert \mathbf{x}_0, \mathbf{c})}[\cdot]$ averages over reverse paths constrained to hit $\mathbf{x}_0$, whereas the sampler runs $\mathbf{x}_T \to \cdots \to \mathbf{x}_0$ without this constraint. Conditioning on the endpoint creates a diffusion-bridge posterior with generally no closed form and costly sampling.

2. **Nested, $\boldsymbol{\theta}$-Coupled Expectations.** The loss $-\log \sigma(\Delta R(\mathbf{c}; \boldsymbol{\theta}))$ has both the path joint distribution and the integrand $R_{\pi_\theta}$ depending on $\boldsymbol{\theta}$. Thus $\nabla_\theta$ must differentiate through the sampling distribution, leading to REINFORCE/pathwise couplings and high-variance gradients.

3. **Long Chains, Large Sums, and Expensive Backpropagation.** Computing $R_{\pi_\theta}(\mathbf{c}, \mathbf{x}_{0:T})$ requires $\mathcal{O}(T)$ per-step log-densities with $T \sim 10^2$--$10^3$, for both policy and reference, and for both winner/loser paths. Backpropagating through these stochastic chains is memory and compute heavy.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tractable Diffusion-DPO Surrogate (Stepwise Form))</span></p>

By exploiting the Markov property of the reverse process and applying Jensen's inequality, we can optimize a tractable upper bound on the Diffusion-DPO loss. For the reverse chain,

$$\pi_\theta(\mathbf{x}_{0:T} \vert \mathbf{c}) = \pi_\theta(\mathbf{x}_T \vert \mathbf{c}) \prod_{t=1}^T \pi_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{c}),$$

so if $\pi_\theta(\mathbf{x}_T \vert \mathbf{c}) = \pi_{\text{ref}}(\mathbf{x}_T \vert \mathbf{c})$ (same prior), then

$$\log \frac{\pi_\theta(\mathbf{x}_{0:T} \vert \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_{0:T} \vert \mathbf{c})} = \sum_{t=1}^T \log \frac{\pi_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{c})}.$$

Applying a single step Jensen upper bound (sample $t \sim \mathcal{U}\lbrace 1, \dots, T \rbrace$ and rescale by $T$), the final objective is an expected per-step surrogate:

$$\mathcal{L}_{\text{Diff-DPO}}(\boldsymbol{\theta}; \pi_{\text{ref}}) \le -\mathbb{E}_{\substack{(\mathbf{c}, \mathbf{x}_0^w, \mathbf{x}_0^l) \sim \mathcal{D} \\ t \sim \mathcal{U}\lbrace 1, \dots, T \rbrace}}\left[\log \sigma(\beta T \Delta_t)\right],$$

where each per-step contribution is

$$\Delta_t = \log \frac{\pi_\theta(\mathbf{x}_{t-1}^w \vert \mathbf{x}_t^w, \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_{t-1}^w \vert \mathbf{x}_t^w, \mathbf{c})} - \log \frac{\pi_\theta(\mathbf{x}_{t-1}^l \vert \mathbf{x}_t^l, \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_{t-1}^l \vert \mathbf{x}_t^l, \mathbf{c})}.$$

For Gaussian reverse conditionals used in diffusion models (take $\boldsymbol{\epsilon}$-prediction as an example),

$$\log \frac{\pi_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{c})}{\pi_{\text{ref}}(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{c})} = \text{const} - \lambda_t\left(\lVert \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_t \rVert^2 - \lVert \hat{\boldsymbol{\epsilon}}_{\text{ref}}(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_t \rVert^2\right),$$

where $\lambda_t > 0$ absorbs noise schedule factors. Thus each per-time contribution is proportional to an MSE difference (policy vs. reference) at slice $t$. Define for any $\mathbf{x}_t$:

$$\Delta\text{MSE}(\mathbf{x}_t) := \lVert \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon} \rVert^2 - \lVert \hat{\boldsymbol{\epsilon}}_{\text{ref}}(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon} \rVert^2.$$

This motivates the following practical surrogate for $\mathcal{L}_{\text{Diff-DPO}}(\boldsymbol{\theta}; \pi_{\text{ref}})$:

$$\tilde{\mathcal{L}}_{\text{Diff-DPO}}(\boldsymbol{\theta}; \pi_{\text{ref}}) := \mathbb{E}_{\substack{(\mathbf{c}, \mathbf{x}_0^w, \mathbf{x}_0^l) \sim \mathcal{D} \\ t \sim \mathcal{U}\lbrace 1, \dots, T \rbrace, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}}\Big[w(t)\left(\Delta\text{MSE}(\mathbf{x}_t^w) - \Delta\text{MSE}(\mathbf{x}_t^l)\right)\Big],$$

where $\mathbf{x}_t^w = \alpha_t \mathbf{x}_0^w + \sigma_t \boldsymbol{\epsilon}$ and $\mathbf{x}_t^l = \alpha_t \mathbf{x}_0^l + \sigma_t \boldsymbol{\epsilon}$ share the same noise $\boldsymbol{\epsilon}$ for variance reduction, and $w(t) > 0$ collects the time weighting (e.g., $w(t) \propto \lambda_t$).

Intuitively, minimizing $\tilde{\mathcal{L}}_{\text{Diff-DPO}}$ increases the model's prediction accuracy on the winner relative to the reference and decreases it on the loser. Because improvements are always measured relative to $\pi_{\text{ref}}$ at the same time step, the policy is nudged toward winner-like denoising trajectories and away from loser-like ones, while remaining anchored to the reference.

</div>

---

### 8.6 Closing Remarks

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Chapter 8: Key Takeaways)</span></p>

This chapter has shifted our focus from foundational principles to the practical challenge of controllable generation. We established a unified framework for guidance based on the Bayesian decomposition of the conditional score, which elegantly separates the generative process into an unconditional direction and a steering term.

**Key techniques covered:**

1. **Classifier Guidance (CG):** Uses an external classifier trained on noised data, applied at inference time to steer a pre-trained unconditional model. Simple and flexible, but requires a separate noisy classifier.

2. **Classifier-Free Guidance (CFG):** Learns conditional and unconditional scores within a single model via conditional dropout. More efficient and widely adopted, as it requires training only a single diffusion model.

3. **Training-Free Guidance:** Steers a pre-trained model at inference time by defining a surrogate likelihood from an arbitrary loss function, enabling applications from artistic control to solving inverse problems without any retraining.

4. **Direct Preference Optimization (DPO) and Diffusion-DPO:** Beyond simple conditioning, DPO bypasses the need for an explicit reward model and reinforcement learning by deriving a loss directly from preference data. Diffusion-DPO adapts this to the pathwise structure of diffusion models, yielding a tractable single-step MSE-difference surrogate.

**The next frontier:** Having addressed *what* to generate (this chapter) and *how to steer* generation, the next chapter tackles the equally important question of *how fast* we can generate it -- exploring sophisticated numerical solvers designed to drastically reduce the number of required sampling steps.

</div>

---

## Chapter 9: Sophisticated Solvers for Fast Sampling

The generation process of a diffusion model, which maps noise to data samples, is mathematically equivalent to solving either an SDE or its associated ODE. This procedure is inherently slow, since it relies on numerical solvers that approximate solution trajectories with many small integration steps. Accelerating inference has therefore become a central research objective. Broadly, existing approaches fall into two categories:

- **Training-Free Approaches:** Develop advanced numerical solvers to improve the efficiency of diffusion sampling without additional training (focus of this chapter).
- **Training-Based Approaches:** Distill a pre-trained diffusion model into a fast generator, or directly learn the ODE flow map so that only a few sampling steps are required (Chapters 10 and 11).

### 9.1 Prologue

#### 9.1.1 Advanced Solvers for Diffusion Models

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(PF-ODE Sampling Setup)</span></p>

Given a pre-trained diffusion model $\mathbf{s}_{\phi^\times}(\mathbf{x}, t) \approx \nabla_\mathbf{x} \log p_t(\mathbf{x})$, sampling can be viewed as solving the PF-ODE with initial condition $\mathbf{x}(T) \sim p_{\text{prior}}$, integrated backward from $t = T$ down to $t = 0$:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{f}(\mathbf{x}(t), t) - \frac{1}{2}g^2(t)\underbrace{\nabla_\mathbf{x}\log p_t(\mathbf{x}(t))}_{\approx \mathbf{s}_{\phi^\times}(\mathbf{x}(t), t)}.$$

This ODE is directly associated with the forward stochastic process $\mathrm{d}\mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t), t)\,\mathrm{d}t + g(t)\,\mathrm{d}\mathbf{w}(t)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Oracle vs. Empirical Flow Map)</span></p>

The exact solution of the PF-ODE can be written in integral form:

$$\widetilde{\boldsymbol{\Psi}}_{T \to 0}(\mathbf{x}(T)) = \mathbf{x}(T) + \int_T^0 \left[f(\tau)\mathbf{x}(\tau) - \tfrac{1}{2}g^2(\tau)\mathbf{s}_{\phi^\times}(\mathbf{x}(\tau), \tau)\right] \mathrm{d}\tau$$

Here $\boldsymbol{\Psi}_{s \to t}(\mathbf{x})$ denotes the flow map of the *oracle* PF-ODE, while $\widetilde{\boldsymbol{\Psi}}_{s \to t}(\mathbf{x})$ denotes the flow map of the *empirical* PF-ODE (replacing the true score with the learned approximation). Since the integral cannot be evaluated in closed form, sampling must rely on *numerical solvers*.

</div>

**Discretized Approximation of Continuous Trajectories.** Let $\mathbf{x}_T$ denote the initial state at time $T$, and consider a decreasing partition

$$T = t_0 > t_1 > \cdots > t_M = 0.$$

Starting from $\tilde{\mathbf{x}}_{t_0} = \mathbf{x}_T \sim p_{\text{prior}}$, the solver produces a sequence $\lbrace \tilde{\mathbf{x}}_{t_i}\rbrace_{i=0}^{M}$ that ideally approximates the empirical PF-ODE flow. The final iterate $\tilde{\mathbf{x}}_{t_M}$ serves as an estimate of the clean sample $\mathbf{x}_0$ at $t = 0$.

#### 9.1.2 A Common Framework for Designing Solvers in Literature

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Three Design Principles)</span></p>

Zhang and Chen (2022) highlighted three practical principles for designing numerical solvers for the PF-ODE:

**I. Semilinear Structure.** In most scheduler formulations the drift is instantiated in a linear form $\mathbf{f}(\mathbf{x}, t) := f(t)\,\mathbf{x}$, which induces a *semilinear* PF-ODE:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \underbrace{f(t)\mathbf{x}(t)}_{\text{linear part}} - \underbrace{\tfrac{1}{2}g^2(t)\mathbf{s}_{\phi^\times}(\mathbf{x}(t), t)}_{\text{nonlinear part}}.$$

**II. Parameterizations beyond the Score.** As $t \to 0$, the score $\nabla_\mathbf{x}\log p_t(\cdot)$ can blow up at rate $1/\sigma_t^2$, while the noise predictor $\boldsymbol{\epsilon}^*$ stays bounded (since $\mathbb{E}\lVert\boldsymbol{\epsilon}^*\rVert_2^2 \le D$). Hence a widely used alternative is to predict the noise $\boldsymbol{\epsilon}_{\phi^\times}$:

$$\mathbf{s}_{\phi^\times}(\mathbf{x}, t) = -\frac{1}{\sigma_t}\boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}, t).$$

Substituting into the PF-ODE gives:

$$\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \underbrace{f(t)\mathbf{x}(t)}_{\text{linear part}} + \underbrace{\tfrac{1}{2}\tfrac{g^2(t)}{\sigma_t}\boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}(t), t)}_{\text{nonlinear part}}.$$

**III. Exponential Integrators for Semilinear PF-ODEs.** The *exponential integrator* $\mathcal{E}(s \to t) := \exp\!\bigl(\int_s^t f(u)\,\mathrm{d}u\bigr)$ provides an exact alternative representation of the solution:

$$\widetilde{\boldsymbol{\Psi}}_{s \to t}(\mathbf{x}_s) = \underbrace{\mathcal{E}(s \to t)\mathbf{x}_s}_{\text{linear part}} + \frac{1}{2}\int_s^t \frac{g^2(\tau)}{\sigma_\tau}\mathcal{E}(\tau \to t)\boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_\tau, \tau)\,\mathrm{d}\tau.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Exponential Euler vs. Plain Euler)</span></p>

The **exponential-Euler** update (freezing the nonlinear part as constant over $[s - \Delta s, s]$):

$$\mathbf{x}_{s-\Delta s}^{\text{Exp-Euler}} = e^{-f(s)\Delta s}\mathbf{x}_s + \frac{e^{-f(s)\Delta s} - 1}{f(s)}\,\mathbf{N}(\mathbf{x}_s, s)$$

exactly computes the linear factor $e^{-f(s)\Delta s}$ (no approximation). In contrast, the **plain Euler** step approximates this as $(1 - f(s)\Delta s)$, incurring a relative error of order $a/2$ where $a = -f(s)\Delta s$. This purely linear distortion from the discretization is especially important when taking large steps.

</div>

#### 9.1.3 Approaches of PF-ODE Numerical Solvers

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Two Categories of Solvers)</span></p>

**Time Stepping Methods:** Discretize the time interval $[0, T]$ and approximate the PF-ODE using various numerical integration schemes. Key examples:
- **DDIM** (Section 9.2): single exponential-Euler step, first-order.
- **DEIS** (Section 9.3): multistep method using Lagrange polynomial interpolation of past model evaluations.
- **DPM-Solver family** (Sections 9.4--9.5): Taylor expansion in log-SNR time $\lambda$, yielding higher-order single-step and multistep solvers.

**(Optional) Time Parallel Methods:**
- **ParaDiGMs** (Section 9.8): reformulates the ODE solution as a fixed-point problem, allowing integral terms to be evaluated in parallel.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Number of Function Evaluations)</span></p>

The true computational cost is dominated not by the number of discretization steps, but by how many times we must call the model network. We refer to this count as the **number of function evaluations (NFE)**. If a sampler performs $m$ evaluations per step over $N$ steps, the cost scales as $\text{NFE} = m\,N$.

- First-order Euler or exponential-Euler schemes have $m = 1$.
- Single-step $k$th-order methods typically require $m \ge k$.
- Multistep methods reuse past evaluations so that the average $m$ is close to 1.
- Classifier-free guidance effectively doubles the number of calls at each step.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalent Parameterizations)</span></p>

The interchangeable use of the equivalent parameterizations $(f(t), g(t))$ and $(\alpha_t, \sigma_t)$ of the perturbation kernel with $\mathbf{x}_t | \mathbf{x}_0 \sim \mathcal{N}(\cdot; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$, are related via:

$$f(t) = \frac{\alpha_t'}{\alpha_t}, \quad g^2(t) = \frac{\mathrm{d}}{\mathrm{d}t}(\sigma_t^2) - 2\frac{\alpha_t'}{\alpha_t}\sigma_t^2 = 2\sigma_t\sigma_t' - 2\frac{\alpha_t'}{\alpha_t}\sigma_t^2.$$

</div>

---

### 9.2 DDIM

*Denoising Diffusion Implicit Models* (DDIM) is one of the pioneering and most widely used ODE-based solvers. Although its name suggests a variational origin, its practical update rule can be interpreted as a straightforward application of the Euler method to approximate the exponential-integrator formula.

#### 9.2.1 Interpreting DDIM as an ODE Solver

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DDIM as Exponential Euler)</span></p>

Let $s > t$ denote two discrete time steps ($s$ = start, $t$ = target). To approximate the integral in the exponential-integrator formula, assume that

$$\boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_\tau, \tau) \approx \boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_s, s), \quad \text{for all } \tau \in [t, s].$$

This Euler approximation leads to an analytically tractable integral, resulting in the DDIM update formula.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(9.2.1: DDIM = Euler Method (Exponential Euler))</span></p>

The update rule derived by applying the Euler method to the exponential integrator form yields the following DDIM update:

$$\tilde{\mathbf{x}}_t = \frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s - \alpha_t\left(\frac{\sigma_s}{\alpha_s} - \frac{\sigma_t}{\alpha_t}\right)\boldsymbol{\epsilon}_{\phi^\times}(\tilde{\mathbf{x}}_s, s).$$

</div>

#### 9.2.2 Intuition Behind DDIM with Different Parameterizations

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(9.2.1: DDIM in Different Parametrizations)</span></p>

Let $s > t$. Starting from $\tilde{\mathbf{x}}_s \sim p_s$ and ending at time $t$, the DDIM update in different parametrizations are as:

$$\tilde{\mathbf{x}}_t = \frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s + \alpha_t\left(\frac{\sigma_t}{\alpha_t} - \frac{\sigma_s}{\alpha_s}\right)\boldsymbol{\epsilon}^*(\tilde{\mathbf{x}}_s, s) = \frac{\sigma_t}{\sigma_s}\tilde{\mathbf{x}}_s + \alpha_s\left(\frac{\alpha_t}{\alpha_s} - \frac{\sigma_t}{\sigma_s}\right)\mathbf{x}^*(\tilde{\mathbf{x}}_s, s) = \alpha_t\underbrace{\mathbf{x}^*(\tilde{\mathbf{x}}_s, s)}_{\text{estimated clean}} + \sigma_t\underbrace{\boldsymbol{\epsilon}^*(\tilde{\mathbf{x}}_s, s)}_{\text{estimated noise}}.$$

The last identity gives a clear view: starting from $\tilde{\mathbf{x}}_s \sim p_s$, the estimated clean part $\mathbf{x}^*(\tilde{\mathbf{x}}_s, s)$ and estimated noise part $\boldsymbol{\epsilon}^*(\tilde{\mathbf{x}}_s, s)$ act as interpolation endpoints that reconstruct $\tilde{\mathbf{x}}_t \sim p_t$ with coefficients $(\alpha_t, \sigma_t)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(9.2.1: (Exponential) Euler and DDIM Updates)</span></p>

Given the same schedulers $(\alpha_t, \sigma_t)$:

$$\mathbf{v}\text{-prediction:}\quad \text{Euler} = \text{DDIM},$$

$$\boldsymbol{\epsilon}\text{-, } \mathbf{x}\text{-, or } \mathbf{s}\text{-prediction:}\quad \text{exp-Euler} = \text{DDIM} \neq \text{plain Euler},$$

where in the $\boldsymbol{\epsilon}$-, $\mathbf{x}$-, or $\mathbf{s}$-prediction cases, the plain Euler step is *not* equivalent to DDIM, since the linear term is only approximated and may lead to reduced stability.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Illustrative Example: DDIM Under Different Parameterizations)</span></p>

Assume the forward kernel $\alpha_t = 1$ and $\sigma_t = t$ (Karras et al., 2022). The DDIM (exp-Euler) update:

- **$\boldsymbol{\epsilon}$-prediction:** $\tilde{\mathbf{x}}_t = \tilde{\mathbf{x}}_s - (s - t)\,\boldsymbol{\epsilon}^*(\tilde{\mathbf{x}}_s, s)$, which pushes $\tilde{\mathbf{x}}_s$ toward a cleaner estimate by subtracting the oracle noise estimate.
- **$\mathbf{x}$-prediction:** $\tilde{\mathbf{x}}_t = \frac{t}{s}\,\tilde{\mathbf{x}}_s + \left(1 - \frac{t}{s}\right)\mathbf{x}^*(\tilde{\mathbf{x}}_s, s)$, a convex combination of the current sample and the estimated clean data (denoising residual contracts by factor $t/s \in (0, 1)$, so no overshoot).
- **score-prediction:** $\tilde{\mathbf{x}}_t = \tilde{\mathbf{x}}_s + (s - t)\,s\,\nabla_\mathbf{x} \log p_s(\tilde{\mathbf{x}}_s)$, moving uphill along the score field.
- **$\mathbf{v}$-prediction:** $\tilde{\mathbf{x}}_t = \tilde{\mathbf{x}}_s + (t - s)\,\mathbf{v}^*(\tilde{\mathbf{x}}_s, s)$, a straight-line step following the local ODE drift. The secant slope satisfies $\frac{\tilde{\mathbf{x}}_t - \tilde{\mathbf{x}}_s}{t - s} = \mathbf{v}^*(\tilde{\mathbf{x}}_s, s)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Challenge of DDIM)</span></p>

The first-order Euler discretization has global error $\mathcal{O}(h)$, so accuracy degrades as the maximum step size $h := \max_i |t_i - t_{i-1}|$ grows. To improve accuracy, the literature develops higher-order schemes that raise the global order to $\mathcal{O}(h^k)$ ($k \ge 2$) through richer local approximations. The true measure of efficiency is the NFE, and "faster" means reaching the desired quality with a smaller NFE.

</div>

#### 9.2.3 (Optional) A Variational Perspective on DDIM

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Revisiting DDPM's Variational View)</span></p>

In DDPM, training fixes a family of marginal perturbation kernels $p_t(\mathbf{x}_t | \mathbf{x}_0)$ and optimizes a surrogate objective that depends only on these marginals. The reverse conditional at sampling time is the Bayesian posterior under the one-step forward kernel:

$$p(\mathbf{x}_{t-\Delta t} | \mathbf{x}_t, \mathbf{x}_0) = \frac{p(\mathbf{x}_t | \mathbf{x}_{t-\Delta t})\,p_{t-\Delta t}(\mathbf{x}_{t-\Delta t} | \mathbf{x}_0)}{p_t(\mathbf{x}_t | \mathbf{x}_0)}.$$

If one tries to skip steps by enlarging $\Delta t$ while reusing the same one-step kernel, this no longer matches the true multi-step posterior and typically degrades the marginals.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Original DDIM Motivation)</span></p>

DDIM observes that the training objective constrains only the marginals $p_t(\mathbf{x}_t | \mathbf{x}_0)$, not the intermediate reverse transitions. Hence one may *specify* a family of reverse conditionals $\pi(\mathbf{x}_t | \mathbf{x}_s, \mathbf{x}_0)$ for any $t < s$ that are **one-step marginally consistent**:

$$\int \pi(\mathbf{x}_t | \mathbf{x}_s, \mathbf{x}_0)\,p_s(\mathbf{x}_s | \mathbf{x}_0)\,\mathrm{d}\mathbf{x}_s = p_t(\mathbf{x}_t | \mathbf{x}_0).$$

This construction removes any dependence on the forward one-step kernel $p(\mathbf{x}_t | \mathbf{x}_{t-\Delta t})$ and legitimizes coarse (skipped) time steps.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Derivation of Discrete-Time DDIM)</span></p>

Consider the general forward perturbation $p_t(\mathbf{x}_t | \mathbf{x}_0) := \mathcal{N}(\mathbf{x}_t;\, \alpha_t \mathbf{x}_0,\, \sigma_t^2 \mathbf{I})$, where $\mathbf{x}_0 \sim p_{\text{data}}$. For any $t < s$ we posit the Gaussian family:

$$\pi(\mathbf{x}_t | \mathbf{x}_s, \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; a_{t,s}\,\mathbf{x}_0 + b_{t,s}\,\mathbf{x}_s,\; c_{t,s}^2\,\mathbf{I}\right),$$

with coefficients $(a_{t,s}, b_{t,s}, c_{t,s})$ to be determined by the marginal-consistency constraint. Equating means and variances:

$$\alpha_t = a_{t,s} + b_{t,s}\,\alpha_s, \qquad \sigma_t^2 = b_{t,s}^2\,\sigma_s^2 + c_{t,s}^2.$$

This system is underdetermined, so we treat $c_{t,s}$ as a free parameter with $0 \le c_{t,s} \le \sigma_t$, and solve:

$$b_{t,s} = \frac{\sqrt{\sigma_t^2 - c_{t,s}^2}}{\sigma_s}, \qquad a_{t,s} = \alpha_t - \alpha_s\,b_{t,s}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(9.2.2: DDIM Coefficients)</span></p>

Let $\pi(\mathbf{x}_t | \mathbf{x}_s, \mathbf{x}_0)$ be given by the Gaussian family above. If the marginal-consistency condition holds, then the coefficients are exactly those given above, with $0 \le c_{t,s} \le \sigma_t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DDIM Sampler Family)</span></p>

The DDIM sampler follows from the chosen reverse kernel by replacing $\mathbf{x}_0$ with a predictor from a pre-trained model. Using the $\boldsymbol{\epsilon}$-prediction network, we set $\mathbf{x}_{\phi^\times}(\mathbf{x}_s, s) := \frac{\mathbf{x}_s - \sigma_s\,\boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_s, s)}{\alpha_s}$. The update becomes:

$$\mathbf{x}_t = \frac{\alpha_t}{\alpha_s}\,\mathbf{x}_s + \left(\sqrt{\sigma_t^2 - c_{t,s}^2} - \frac{\alpha_t}{\alpha_s}\sigma_s\right)\boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_s, s) + c_{t,s}\,\boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$

where $c_{t,s} \in [0, \sigma_t]$ controls stochasticity. By varying $c_{t,s}$, one obtains a family of samplers sharing the same pre-trained diffusion model and requiring no retraining:

- **DDPM Step (Posterior Variance):** $c_{t,s} = \frac{\sigma_s}{\sigma_t}\,\sigma_{t|s}$ makes $\pi(\mathbf{x}_t | \mathbf{x}_s, \mathbf{x}_0)$ equal to the Bayesian posterior $p(\mathbf{x}_t | \mathbf{x}_s, \mathbf{x}_0)$ induced by the one-step forward kernel.
- **Deterministic DDIM ($\eta = 0$):** $c_{t,s} = 0$ gives $\mathbf{x}_t = \alpha_{t|s}\,\mathbf{x}_s + (\sigma_t - \alpha_{t|s}\,\sigma_s)\,\boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_s, s)$, which matches the ODE-view DDIM jump.
- **Interpolation:** Define $c_{t,s} = \eta\,\frac{\sigma_s}{\sigma_t}\,\sigma_{t|s}$, $\eta \in [0, 1]$, so that $\eta$ smoothly interpolates between the stochastic DDPM update ($\eta = 1$) and the deterministic DDIM update ($\eta = 0$).

</div>

#### 9.2.4 DDIM as Conditional Flow Matching

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DDIM as Conditional Flow Matching)</span></p>

Deterministic DDIM can be understood as searching for a conditional flow map that pushes $p_s(\cdot | \mathbf{x}_0)$ forward to $p_t(\cdot | \mathbf{x}_0)$. Under the linear-Gaussian path $\mathbf{x}_\tau = \alpha_\tau\,\mathbf{x}_0 + \sigma_\tau\,\boldsymbol{\epsilon}$, this leads to the **conditional map**

$$\boldsymbol{\Psi}_{s \to t}(\mathbf{x}_s | \mathbf{x}_0) = \frac{\sigma_t}{\sigma_s}\,\mathbf{x}_s + \left(\alpha_t - \alpha_s\,\frac{\sigma_t}{\sigma_s}\right)\mathbf{x}_0,$$

whose instantaneous **conditional velocity** is

$$\mathbf{v}_t^*(\mathbf{x} | \mathbf{x}_0) = \frac{\sigma_t'}{\sigma_t}\,\mathbf{x} + \left(\alpha_t' - \alpha_t\,\frac{\sigma_t'}{\sigma_t}\right)\mathbf{x}_0.$$

The CFM regression target $\mathcal{L}_{\text{CFM}}(\phi) = \mathbb{E}\left[\lVert \mathbf{v}_\phi(\mathbf{x}_t, t) - \mathbf{v}_t^*(\mathbf{x}_t | \mathbf{x}_0)\rVert^2\right]$ equals the conditional velocity of the DDIM conditional map.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(9.2.2: Conditional Level)</span></p>

Along the conditional Gaussian path, the DDIM conditional map and the CFM target generate the same conditional flow $\boldsymbol{\Psi}_{s \to t}(\cdot | \mathbf{x}_0)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Marginal Level)</span></p>

Averaging the conditional velocity over the posterior of $\mathbf{x}_0$ given $\mathbf{x}_t = \mathbf{x}$ yields the marginal PF-ODE drift: $\mathbf{v}^*(\mathbf{x}, t) = \mathbb{E}\left[\mathbf{v}_t^*(\mathbf{x} | \mathbf{x}_0) | \mathbf{x}_t = \mathbf{x}\right]$. In short, DDIM is (i) a deterministic conditional transport whose tangent equals the CFM target, and (ii) after marginalizing that tangent, an Euler step of the PF-ODE whose step coincides with the DDIM update.

</div>

---

### 9.3 DEIS

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(DEIS: Key Idea)</span></p>

In the exponential-integrator formula, the only unknown is the model output $\boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_\tau, \tau)$; the schedule terms and the weight $\mathcal{E}(\tau \to t)$ are fixed. DDIM (Euler's method) approximates this integral by holding the model output constant. A natural question then arises: *can we make better use of the model evaluations already computed?*

DEIS reuses previous outputs (anchors) to fit a simple curve in time (a *Lagrange polynomial*) and replaces the hard integral of an unknown function with the exact integral of an approximating curve defined by past model calls.

</div>

#### 9.3.1 Polynomial Extrapolation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lagrange Polynomial Interpolation)</span></p>

Given anchor points $(\tau_0, \mathbf{Y}_0), \ldots, (\tau_n, \mathbf{Y}_n)$ with $\tau_0 < \tau_1 < \cdots < \tau_n$, the **Lagrange polynomial** is the unique degree-$n$ polynomial passing through all anchors:

$$\mathbf{Y}(\tau) = \sum_{j=0}^{n} \ell_j(\tau)\,\mathbf{Y}_j, \quad \ell_j(\tau_k) = \delta_{jk}, \quad \sum_{j=0}^{n} \ell_j(\tau) = 1,$$

where $\ell_j(\tau) = \prod_{\substack{k=0 \\ k \neq j}}^{n} \frac{\tau - \tau_k}{\tau_j - \tau_k}$ are the Lagrange basis functions. Each $\ell_j(\tau)$ acts like a "spotlight", taking value 1 at its own anchor ($\ell_j(\tau_j) = 1$) and 0 at all others.

**Small cases:**
- $n = 0$ (Constant): $\mathbf{Y}(\tau) \equiv \mathbf{Y}_n$.
- $n = 1$ (Line): $\mathbf{Y}(\tau) = \frac{\tau - \tau_n}{\tau_{n-1} - \tau_n}\,\mathbf{Y}_{n-1} + \frac{\tau - \tau_{n-1}}{\tau_n - \tau_{n-1}}\,\mathbf{Y}_n$.
- $n = 2$ (Quadratic): passes a parabola through three anchors.

</div>

#### 9.3.2 DEIS: Lagrange Polynomial Approximation of the PF-ODE Integral

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(AB-DEIS-n Update)</span></p>

Let $n \ge 0$ be the chosen polynomial degree. At step $i$, we approximate $\tau \mapsto \boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_\tau, \tau)$ over $[t_{i-1}, t_i]$ by a degree-$n$ polynomial interpolant built from past model outputs, and substitute this approximation into the exponential-integrator update.

**Case I: $i = n+1, \ldots, M$ (Sufficient History).** DEIS reuses the last $n+1$ model evaluations as anchors, constructs the degree-$n$ Lagrange polynomial $P_n(\tau)$, and substitutes into the exponential-integrator formula:

$$\tilde{\mathbf{x}}_{t_i} = \mathcal{E}(t_{i-1} \to t_i)\,\tilde{\mathbf{x}}_{t_{i-1}} + \sum_{j=0}^{n} C_{i,j}\,\boldsymbol{\epsilon}_{\phi^\times}(\tilde{\mathbf{x}}_{t_{i-1-j}}, t_{i-1-j}),$$

with coefficients $C_{i,j} := \frac{1}{2}\int_{t_{i-1}}^{t_i} \frac{g^2(\tau)}{\sigma_\tau}\,\mathcal{E}(\tau \to t_i)\,\ell_j^{(i)}(\tau)\,\mathrm{d}\tau$ that depend only on the schedule $(\alpha_\tau, \sigma_\tau)$ and the grid $\lbrace t_i\rbrace$, so they can be precomputed exactly in closed form.

**Case II: $i = 1, \ldots, n$ (Insufficient History / Warm Start).** Set the degree to $i-1$ and use all $i$ available anchors. The degree ramps up from 0 (constant) to the target degree $n$ as more history accumulates.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Truncation Error of AB-DEIS-n)</span></p>

When $i \ge n+1$ (sufficient history), the step attains local truncation error $\mathcal{O}(h^{n+1})$ under standard smoothness assumptions. During warm start ($i \le n$), the per-step order is $\mathcal{O}(h^{\min\lbrace n,\,i-1\rbrace+1})$, ramping up until full order is reached. Very large $n$ often degrades performance due to interpolation ill-conditioning, noise amplification, and tighter stability constraints; small degrees ($n \in \lbrace 1, 2, 3\rbrace$) usually provide the best accuracy-stability trade-off.

</div>

#### 9.3.3 DDIM = AB-DEIS-0

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(9.3.1: DDIM = AB-DEIS-0)</span></p>

When $n = 0$ (constant polynomial), the AB-DEIS-0 update is exactly the exponential-Euler step (constant-in-time $\boldsymbol{\epsilon}_{\phi^\times}$ over $[t_{i-1}, t_i]$), which coincides with the deterministic DDIM update.

</div>

---

### 9.4 DPM-Solver

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(DPM-Solver: Overview)</span></p>

The DPM-Solver family (DPM-Solver, DPM-Solver++, DPM-Solver-v3) represents a major advance in PF-ODE solvers. The goal is simple: achieve similar sample quality with far fewer steps. In practice, these methods reduce the steps required by DDIM from more than 50 to about 10--15.

Like DEIS, DPM-Solver starts from the semilinear form of the PF-ODE and works in the $\boldsymbol{\epsilon}$-prediction parameterization, using the exponential integrator (variation of constants) representation. The key idea is to reparameterize time by the half-log signal-to-noise ratio, so that the nonlinear term becomes an exponentially weighted integral that admits low-cost Taylor expansions.

</div>

#### 9.4.1 DPM-Solver's Insight: Time Reparameterization via Log-SNR

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Half-Log Signal-to-Noise Ratio)</span></p>

$$\lambda_t := \frac{1}{2}\log\frac{\alpha_t^2}{\sigma_t^2} = \log\frac{\alpha_t}{\sigma_t}.$$

For common noise schedules, $\lambda_t$ is strictly decreasing in $t$, so it has an inverse function $t_\lambda(\cdot)$ satisfying $t = t_\lambda(\lambda(t))$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(9.4.1: Exponentially Weighted Exact Solution)</span></p>

Given an initial value $\mathbf{x}_s$ at time $s > 0$, the exact solution $\widetilde{\boldsymbol{\Psi}}_{s \to t}(\mathbf{x}_s)$ at time $t \in [0, s]$ of the PF-ODE can be re-expressed as:

$$\widetilde{\boldsymbol{\Psi}}_{s \to t}(\mathbf{x}_s) = \frac{\alpha_t}{\alpha_s}\mathbf{x}_s - \alpha_t \int_{\lambda_s}^{\lambda_t} e^{-\lambda}\,\hat{\boldsymbol{\epsilon}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda)\,\mathrm{d}\lambda,$$

where $\hat{\mathbf{x}}_\lambda := \mathbf{x}_{t_\lambda(\lambda)}$ and $\hat{\boldsymbol{\epsilon}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda) := \boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_{t_\lambda(\lambda)}, t_\lambda(\lambda))$ denote the reparameterized quantities.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Reparameterize Time?)</span></p>

In $\lambda$-time, the model appears inside an exponentially weighted integral $\int e^{-\lambda}\hat{\boldsymbol{\epsilon}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda)\,\mathrm{d}\lambda$, where the $e^{-\lambda}$ factor produces closed-form coefficients and smooths the integrand, which is exactly what high-order local approximations require.

The PF-ODE in $\lambda$-time becomes: $\frac{\mathrm{d}\hat{\mathbf{x}}_\lambda}{\mathrm{d}\lambda} = \frac{\alpha_\lambda'}{\alpha_\lambda}\,\hat{\mathbf{x}}_\lambda - \sigma_\lambda\,\hat{\boldsymbol{\epsilon}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda).$

For strictly monotone $\lambda(t)$, a first-order change of variables gives $\Delta t \approx \Delta\lambda / |\lambda'(t)|$. So $\Delta t$ is smaller where $|\lambda'(t)|$ is large (i.e., where $\lambda$ changes rapidly with $t$), and larger where $|\lambda'(t)|$ is small. This often makes the integrand smoother to approximate on a uniform $\lambda$ grid.

</div>

#### 9.4.2 Estimating the Integral with Taylor Expansion

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(DPM-Solver-n Update)</span></p>

Starting with the previous point $\tilde{\mathbf{x}}_s$ at time $s$, the solution $\tilde{\mathbf{x}}_t$ at time $t$ is given by:

$$\tilde{\mathbf{x}}_t = \frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s - \alpha_t \int_{\lambda_s}^{\lambda_t} e^{-\lambda}\,\hat{\boldsymbol{\epsilon}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda)\,\mathrm{d}\lambda.$$

We approximate $\hat{\boldsymbol{\epsilon}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda)$ by its $(n-1)$th-order Taylor expansion about $\lambda_s$:

$$\hat{\boldsymbol{\epsilon}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda) = \sum_{k=0}^{n-1} \frac{(\lambda - \lambda_s)^k}{k!}\,\hat{\boldsymbol{\epsilon}}_{\phi^\times}^{(k)}(\hat{\mathbf{x}}_{\lambda_s}, \lambda_s) + \mathcal{O}((\lambda - \lambda_s)^n).$$

Substituting and integrating yields a closed-form approximation with coefficients $C_k := \int_{\lambda_s}^{\lambda_t} e^{-\lambda}\frac{(\lambda - \lambda_s)^k}{k!}\,\mathrm{d}\lambda$ that can be precomputed analytically:

$$\tilde{\mathbf{x}}_t = \frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s - \alpha_t \sum_{k=0}^{n-1} \hat{\boldsymbol{\epsilon}}_{\phi^\times}^{(k)}(\hat{\mathbf{x}}_{\lambda_s}, \lambda_s)\,C_k + \mathcal{O}(h^{n+1}),$$

where $h := \lambda_t - \lambda_s$ and $\varphi_1(h) = \frac{e^h - 1}{h},\; \varphi_2(h) = \frac{e^h - h - 1}{h^2},\; \varphi_3(h) = \frac{e^h - \frac{h^2}{2} - h - 1}{h^3}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(DPM-Solver-1)</span></p>

Consider $n = 1$ (first order). Starting from the previous estimated point $\tilde{\mathbf{x}}_s$:

$$\tilde{\mathbf{x}}_t = \frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s - \sigma_t(e^h - 1)\,\boldsymbol{\epsilon}_{\phi^\times}(\tilde{\mathbf{x}}_s, s) + \mathcal{O}(h^2).$$

This is exactly the DDIM update (proved in Proposition 9.4.2).

</div>

#### 9.4.3 Implementation of DPM-Solver-n

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Approximating Higher-Order Derivatives)</span></p>

DPM-Solver-$n$ with $n \ge 2$ requires the $k$th-derivative $\hat{\boldsymbol{\epsilon}}_{\phi^\times}^{(k)}(\hat{\mathbf{x}}_\lambda, \lambda)$ for $k \le n - 1$. Directly computing higher-order derivatives is expensive. Instead, Lu et al. (2022b) introduce an intermediate timestep $s^{\text{mid}}$ between $s$ and $t$ to approximate higher-order derivatives via finite differences.

For $n = 2$ with $\gamma = \frac{1}{2}$ (midpoint), the two-stage update (Algorithm 5) is:

1. **Predict midpoint:** $\mathbf{x}_i^{\text{mid}} \leftarrow \frac{\alpha_{s^{\text{mid}}}}{\alpha_{t_{i-1}}}\tilde{\mathbf{x}}_{t_{i-1}} - \sigma_{s^{\text{mid}}}(e^{h_i/2} - 1)\,\boldsymbol{\epsilon}_{\phi^\times}(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1})$
2. **Correct:** $\tilde{\mathbf{x}}_{t_i} \leftarrow \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\mathbf{x}}_{t_{i-1}} - \sigma_{t_i}(e^{h_i} - 1)\,\boldsymbol{\epsilon}_{\phi^\times}(\mathbf{x}_i^{\text{mid}}, s_i^{\text{mid}})$

Each step requires exactly two model evaluations: one at $(\tilde{\mathbf{x}}_s, s)$ and one at the predicted midpoint $(\mathbf{x}^{\text{mid}}, s^{\text{mid}})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Selection of Sampling Timesteps)</span></p>

Lu et al. (2022b) propose selecting timesteps based on **uniform spacing in log-SNR time** $\lambda_t$:

$$\lambda_{t_i} = \lambda_T + \frac{i}{M}(\lambda_0 - \lambda_T), \quad i = 0, \ldots, M.$$

Uniform spacing in $\lambda$ yields approximately uniform local error across the trajectory, resulting in finer (denser) steps in $t$ where the signal dominates (high SNR), and coarser (sparser) steps in the noise-dominated regime.

</div>

#### 9.4.4 DDIM = DPM-Solver-1

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(9.4.2: DDIM is DPM-Solver-1)</span></p>

The update rule of DDIM, given in Equation (9.2.2), is identical to that of DPM-Solver-1, given in Equation (9.4.8).

*Proof.* By the definition of $\lambda$, we have $\frac{\sigma_s}{\alpha_s} = e^{-\lambda_s}$ and $\frac{\sigma_t}{\alpha_t} = e^{-\lambda_t}$. Substituting these expressions, along with $h = \lambda_t - \lambda_s$, into Equation (9.2.2) recovers the update rule in Equation (9.4.8).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why DDIM Outperforms Traditional Euler)</span></p>

DDIM may outperform traditional Euler methods in $t$-parametrization because it effectively exploits the semilinearity of the diffusion ODE under a more suitable $\lambda$-reparametrization. When the Score SDE paper appeared, Runge-Kutta (RK45) was commonly used to solve the vanilla PF-ODE, but the semilinearity of its drift remained unexploited. DPM-Solver-$k$ ($k \ge 2$) explicitly leverages this semilinearity via a time reparameterization, explaining why DPM-Solver attains higher-order accuracy with far fewer NFEs, reducing a typical DDIM schedule of several hundred steps to about 10--15 steps while preserving high sample quality.

</div>

#### 9.4.5 Discussion on DPM-Solver-2 and Classic Heun Updates

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Euler and Heun Analogues Across Parameterizations)</span></p>

Different parameterizations of the PF-ODE lead to different interpretations of classical updates:

**First-order (Euler vs. DDIM):**

$$\mathbf{v}\text{-prediction:}\quad \text{Euler} = \text{DDIM},$$

$$\boldsymbol{\epsilon}\text{-, }\mathbf{x}\text{-, or }\mathbf{s}\text{-prediction:}\quad \text{exp-Euler} = \text{DDIM} \neq \text{plain Euler}.$$

**Second-order (Heun vs. DPM-Solver-2):**

$$\mathbf{v}\text{-prediction:}\quad \text{Heun} = \text{DPM-Solver-2},$$

$$\boldsymbol{\epsilon}\text{-, }\mathbf{x}\text{-, or }\mathbf{s}\text{-prediction:}\quad \text{exp-Heun} = \text{DPM-Solver-2} \neq \text{plain Heun}.$$

For $\boldsymbol{\epsilon}$-, $\mathbf{x}$-, or $\mathbf{s}$-prediction, the PF-ODE in log-SNR time $\lambda$ naturally takes a semilinear form. The exponential-Heun update integrates the linear term exactly while approximating the nonlinear term by averaging its effect across the step (a predictor-corrector scheme). The plain Heun method only approximates the linear term, so the two schemes differ when $L(\lambda) \neq 0$. For $\mathbf{v}$-prediction, $L(\lambda) \equiv 0$, so they coincide.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(9.4.1: Heun and DPM-Solver-2 Updates)</span></p>

Given the PF-ODEs in log-SNR time $\lambda$,

$$\mathbf{v}\text{-prediction:}\quad \text{Heun} = \text{DPM-Solver-2},$$

$$\boldsymbol{\epsilon}\text{-, }\mathbf{x}\text{-, or }\mathbf{s}\text{-prediction:}\quad \text{exp-Heun} = \text{DPM-Solver-2} \neq \text{plain Heun},$$

where in the $\boldsymbol{\epsilon}$-, $\mathbf{x}$-, or $\mathbf{s}$-prediction cases, the plain Heun step is not equivalent to DPM-Solver-2, since the linear term is only approximated instead of being integrated exactly.

</div>

---

### 9.5 DPM-Solver++

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(DPM-Solver++: Motivation)</span></p>

High-order solvers enable faster sampling without guidance. However, diffusion models are prized for their controllable and flexible generation, typically achieved via guidance (e.g., CFG). DPM-Solver++ identifies a key limitation of prior high-order solvers: they suffer from stability issues and may become slower than DDIM under large guidance scales (stronger condition). The authors attribute this instability to the amplification of both the output and its derivatives by large guidance scales. Since high-order solvers depend on higher-order derivatives, they are especially sensitive to this effect.

</div>

#### 9.5.1--9.5.2 DPM-Solver++'s Methodology

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DPM-Solver++ Key Changes)</span></p>

To address stability issues, DPM-Solver++ proposes:

1. Adopting **$\mathbf{x}$-prediction** parameterization instead of $\boldsymbol{\epsilon}$-prediction;
2. Applying **thresholding methods** (e.g., dynamic thresholding) to keep the predicted data within training data bounds (mitigating the train-test mismatch at large guidance scales).

Based on the $\mathbf{x}$-prediction, DPM-Solver++ provides two solver variants:
- **Higher-Order Single-Step Solver (2S):** Uses higher-order Taylor expansions with an intermediate evaluation (analogous to DPM-Solver but in $\mathbf{x}$-prediction).
- **Multistep (Two-Step) Solver (2M):** Reuses two previous points to estimate the next step. Each update requires only a single new diffusion model evaluation.

</div>

#### 9.5.3 DPM-Solver++ Single-Step by Taylor Expansion

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DPM-Solver++ Single-Step)</span></p>

DPM-Solver++ derives higher-order solvers in the $\mathbf{x}$-parameterization. The exact PF-ODE solution in $\mathbf{x}$-prediction form is:

$$\widetilde{\boldsymbol{\Psi}}_{s \to t}(\mathbf{x}_s) = \frac{\sigma_t}{\sigma_s}\,\mathbf{x}_s + \sigma_t \int_{\lambda_s}^{\lambda_t} e^\lambda\,\hat{\mathbf{x}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda)\,\mathrm{d}\lambda.$$

The $(n-1)$th-order Taylor expansion of $\hat{\mathbf{x}}_{\phi^\times}$ at $\lambda_{i-1}$ gives the DPM-Solver++ single-step update. When $n = 1$, it reduces to the DDIM update. When $n = 2$ and $\hat{\mathbf{x}}_{\phi^\times}^{(1)}$ is approximated via a finite difference at the midpoint, it gives **DPM-Solver++(2S)**, an update analogous to DPM-Solver-2 (Algorithm 5) but using the $\mathbf{x}$-prediction.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(DPM-Solver++(2S), midpoint special case)</span></p>

**Input:** initial value $\mathbf{x}_T$, time steps $\lbrace t_i\rbrace_{i=0}^M$, data-prediction model $\hat{\mathbf{x}}_{\phi^\times}$

1. $\tilde{\mathbf{x}}_{t_0} \leftarrow \mathbf{x}_T$; $\lambda_{t_i} \leftarrow \log(\alpha_{t_i}/\sigma_{t_i})$
2. $\hat{\mathbf{x}}_0 \leftarrow \hat{\mathbf{x}}_{\phi^\times}(\tilde{\mathbf{x}}_{t_0}, t_0)$ (cache at start)
3. **for** $i \leftarrow 1$ to $M$ **do**
4. $\quad h_i \leftarrow \lambda_{t_i} - \lambda_{t_{i-1}}$; $s_i^{\text{mid}} \leftarrow t_\lambda\!\left(\frac{\lambda_{t_{i-1}} + \lambda_{t_i}}{2}\right)$
5. $\quad \mathbf{u}_i \leftarrow \frac{\sigma_{s^{\text{mid}}}}{\sigma_{t_{i-1}}}\,\tilde{\mathbf{x}}_{t_{i-1}} + \alpha_{s^{\text{mid}}}(1 - e^{-h_i/2})\,\hat{\mathbf{x}}_{i-1}$ (forecast to midpoint)
6. $\quad \mathbf{D}_i^{\text{mid}} \leftarrow \hat{\mathbf{x}}_{\phi^\times}(\mathbf{u}_i, s_i^{\text{mid}})$ (one new model call at the midpoint)
7. $\quad \tilde{\mathbf{x}}_{t_i} \leftarrow \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}\,\tilde{\mathbf{x}}_{t_{i-1}} - \alpha_{t_i}(e^{-h_i} - 1)\,\mathbf{D}_i^{\text{mid}}$
8. $\quad \hat{\mathbf{x}}_i \leftarrow \hat{\mathbf{x}}_{\phi^\times}(\tilde{\mathbf{x}}_{t_i}, t_i)$ (cache for next step)
9. **end for**
10. **return** $\tilde{\mathbf{x}}_{t_M}$

</div>

#### 9.5.4 DPM-Solver++ Multistep by Recycling History

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DPM-Solver++ Multistep (2M))</span></p>

High-order single-step solvers rely (explicitly or implicitly) on higher derivatives of the model output; under strong CFG these derivatives can be strongly amplified and destabilize the update. DPM-Solver++ mitigates this with a **multistep** (Adams-type) strategy in log-SNR time $\lambda$: it reuses a short history of past data-prediction evaluations along the trajectory to approximate the needed derivatives via finite differences. This requires only one new model call per step.

**Case I. Warm start ($i = 1$; no history):** Use the first-order DPM-style update (which matches the deterministic DDIM step in $\mathbf{x}$-pred.).

**Case II. Two history anchors ($i \ge 2$):** The simplified update rule with the same local truncation error (provided step ratios are bounded) is:

$$\tilde{\mathbf{x}}_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}\,\tilde{\mathbf{x}}_{t_{i-1}} + \alpha_{t_i}(1 - e^{-h_i})\,\mathbf{D}_i^{\text{sim}}(\tilde{\mathbf{x}}_{t_{i-1}}, \tilde{\mathbf{x}}_{t_{i-2}}),$$

where $\mathbf{D}_i^{\text{sim}}(\tilde{\mathbf{x}}_{t_{i-1}}, \tilde{\mathbf{x}}_{t_{i-2}}) := \left(1 + \tfrac{1}{2}r_i\right)\hat{\mathbf{x}}_{\phi^\times}(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1}) - \tfrac{1}{2}r_i\,\hat{\mathbf{x}}_{\phi^\times}(\tilde{\mathbf{x}}_{t_{i-2}}, t_{i-2})$, with step ratio $r_i = h_i / h_{i-1}$. This has local error $\mathcal{O}(h_i^3)$.

If the log-SNR steps are uniform ($h_i \equiv h$, so $r_i = 1$), then $\mathbf{D}_i^{\text{sim}} = \frac{3}{2}\hat{\mathbf{x}}_{i-1} - \frac{1}{2}\hat{\mathbf{x}}_{i-2}$, which are exactly the Adams-Bashforth 2 weights for uniform steps.

</div>

---

### 9.6 PF-ODE Solver Families and Their Numerical Analogues

#### 9.6.1 PF-ODE Solver Families and Classical Counterparts

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(PF-ODE Solvers and Their Classical Analogues)</span></p>

Once the linear drift is treated by an integrating factor, each sampler aligns naturally with an established time-stepping scheme. "exp." denotes integrating-factor (semilinear) treatment of the linear term. AB = Adams-Bashforth, RK = Runge-Kutta.

| PF-ODE Solver | Type | Classical Numerical Analogue |
| --- | --- | --- |
| DDIM | single step | $\mathbf{v}$-prediction: plain Euler; $\boldsymbol{\epsilon}/\mathbf{x}/\mathbf{s}$-prediction: exp. Euler |
| DEIS | multistep | exp. AB ($n$th-order) |
| DPM-Solver-$n$ | single step | exp. RK ($n$th-order) in log-SNR |
| DPM-Solver-2 | single step | $\mathbf{v}$-prediction: plain Heun in log-SNR (2nd-order); $\boldsymbol{\epsilon}/\mathbf{x}/\mathbf{s}$-prediction: exp. Heun in log-SNR (2nd-order) |
| DPM-Solver++ 2S | single step | exp. RK (2nd-order) |
| DPM-Solver++ 2M | multistep | exp. AB (2nd-order) |

**Key equivalences** (for a fixed schedule $(\alpha_t, \sigma_t)$):

$$\mathbf{v}\text{-prediction:}\quad \text{DDIM} = \text{DPM-Solver-1} = \text{DEIS-1} = \text{Euler},$$

$$\boldsymbol{\epsilon}\text{-, }\mathbf{x}\text{-, or }\mathbf{s}\text{-prediction:}\quad \text{DDIM} = \text{DPM-Solver-1} = \text{DEIS-1} = \text{exp Euler}.$$

</div>

#### 9.6.2 Discussion on DEIS and DPM-Solver++

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DEIS vs. DPM-Solver++)</span></p>

Both DEIS and DPM++ are exponential integrator samplers that integrate the linear part exactly and approximate the residual integral by a low-degree polynomial. In unconditional generation, both can achieve high fidelity with as few as 10--20 ODE steps. For conditional generation with CFG, however, DPM++ is often preferable due to its stability under large guidance scales.

| Aspect | DEIS | DPM++ |
| --- | --- | --- |
| Core Viewpoint | Exponential-integrator: integrates the linear term exactly; approximates the nonlinear residual by a polynomial over past nodes. | Same integrator idea; formulated in log-SNR time $\lambda$ with data prediction. |
| Step type | Multistep only | Single-step (2S) and Multistep (2M) |
| Polynomial Basis | Lagrange interpolation across past anchors (high-order multistep). | Backward divided differences (Newton/Adams-type) in $\lambda$-time for 2M; algebraically spans the same polynomial space as Lagrange, but not presented as a Lagrange fit. |
| History Use | Uses $r+1$ past evaluations to build a high-order update. | 2S: one intermediate eval (single-step). 2M: reuses two anchors; after warm start, one model call per step. |

In other words, for the same anchor points and function values, the Lagrange and Newton forms are two different coordinate systems for the same polynomial interpolant.

</div>

---

### 9.7 (Optional) DPM-Solver-v3

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(DPM-Solver-v3: Overview)</span></p>

Both DPM-Solver and DPM-Solver++ design their solvers based on specific parameterizations of the diffusion model ($\boldsymbol{\epsilon}$-/$\mathbf{x}$-prediction), which lacks a principled approach for selecting the parametrization and may not represent the optimal choice. DPM-Solver-v3 (Zheng et al., 2023) addresses this issue and enhances sample quality with fewer timesteps or at large guidance scales.

The core idea is to introduce three additional undetermined/free variables into the PF-ODE (in log-SNR time $\lambda$), enabling the original ODE solution to be reformulated equivalently with a new model parameterization. An efficient search method is then proposed to identify an optimal set of these variables, computed on the pre-trained model, with the objective of minimizing discretization errors.

</div>

#### 9.7.1 Insight 1: Adjusting the Linear Term

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rosenbrock-Type Exponential Integrators)</span></p>

The PF-ODE in $\lambda$-time can be decomposed as $\frac{\mathrm{d}\mathbf{x}_\lambda}{\mathrm{d}\lambda} = \mathbf{L}\mathbf{x} + \mathbf{N}(\mathbf{x}, \lambda)$, where $\mathbf{L}$ is a linear operator and $\mathbf{N}$ the nonlinear remainder. DPM-Solver-v3 introduces a free/undetermined variable $\boldsymbol{\ell}_\lambda$ (a $D$-dimensional quantity depending solely on $\lambda$) to adjust the linear-nonlinear split:

$$\frac{\mathrm{d}\mathbf{x}_\lambda}{\mathrm{d}\lambda} = \underbrace{\left(\frac{\alpha_\lambda'}{\alpha_\lambda} - \boldsymbol{\ell}_\lambda\right)\mathbf{x}_\lambda}_{\text{linear part}} - \underbrace{\left(\sigma_\lambda\hat{\boldsymbol{\epsilon}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda) - \boldsymbol{\ell}_\lambda\mathbf{x}_\lambda\right)}_{\text{nonlinear part}}.$$

Zheng et al. (2023) propose selecting $\boldsymbol{\ell}_\lambda$ by solving a least-squares problem $\boldsymbol{\ell}_\lambda^* = \arg\min_{\boldsymbol{\ell}_\lambda} \mathbb{E}_{\mathbf{x}_\lambda \sim p_\lambda^{\phi^\times}(\mathbf{x}_\lambda)} \lVert\nabla_\mathbf{x}\mathbf{N}_{\phi^\times}(\mathbf{x}_\lambda, \lambda)\rVert_F^2$, which can be solved analytically. This selection leverages preconditioning information from pre-trained models, conceptually making $\mathbf{N}_{\phi^\times}$ less sensitive to errors in $\mathbf{x}$.

</div>

#### 9.7.2 Insight 2: Introducing Free Variables in Model Parameterization

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Minimizing Discretization Error via Free Variables)</span></p>

DPM-Solver-v3 additionally introduces two free variables $\mathbf{a}_\lambda$ and $\mathbf{b}_\lambda$ that define a new parameterization:

$$\mathbf{N}_{\phi^\times}^{\text{new}}(\mathbf{x}_\lambda, \lambda) := e^{-\int_{\lambda_s}^\lambda \mathbf{a}_u\,\mathrm{d}u}\,\mathbf{N}_{\phi^\times}(\mathbf{x}_\lambda, \lambda) - \int_{\lambda_s}^\lambda e^{-\int_{\lambda_s}^r \mathbf{a}_u\,\mathrm{d}u}\,\mathbf{b}_r\,\mathrm{d}r.$$

The optimal $(\mathbf{a}_\lambda^*, \mathbf{b}_\lambda^*)$ are found by solving a least squares optimization:

$$(\mathbf{a}_\lambda^*, \mathbf{b}_\lambda^*) = \arg\min_{\mathbf{a}_\lambda, \mathbf{b}_\lambda} \mathbb{E}_{\mathbf{x}_\lambda \sim p_\lambda^{\phi^\times}(\mathbf{x}_\lambda)} \left[\lVert\mathbf{N}_{\phi^\times}^{(1)}(\mathbf{x}_\lambda, \lambda) - (\mathbf{a}_\lambda\mathbf{N}_{\phi^\times}(\mathbf{x}_\lambda, \lambda) + \mathbf{b}_\lambda)\rVert_2^2\right].$$

This admits an analytical solution, depending on the pre-trained diffusion model, which can be precomputed. The new parameterization conceptually has the form $\mathbf{T}_{\phi^\times}(\mathbf{x}_\lambda, \lambda) := \boldsymbol{\alpha}(\lambda)\hat{\boldsymbol{\epsilon}}_{\phi^\times}(\hat{\mathbf{x}}_\lambda, \lambda) + \boldsymbol{\beta}(\lambda)\mathbf{x}_\lambda + \boldsymbol{\gamma}(\lambda)$.

</div>

#### 9.7.3--9.7.4 Combining Both Insights and Higher-Order DPM-Solver-v3

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implementation of DPM-Solver-v3)</span></p>

In practice, $\boldsymbol{\ell}_\lambda^*$, $\mathbf{a}_\lambda^*$, and $\mathbf{b}_\lambda^*$ have analytical solutions involving the Jacobian-vector product of the pre-trained diffusion model $\boldsymbol{\epsilon}_{\phi^\times}$. Their computation requires evaluating expectations over $p_\lambda^{\phi^\times}(\mathbf{x}_\lambda)$.

In practice, these quantities are estimated via a Monte Carlo (MCMC) approach. Specifically, a batch of datapoints $\mathbf{x}_\lambda \sim p_\lambda^{\phi^\times}$ (roughly 1K--4K samples) is drawn by applying an alternative solver (e.g., the 200-step DPM-Solver++), after which the relevant terms related to $\boldsymbol{\epsilon}_{\phi^\times}$ are computed analytically. Importantly, all these statistics can all be precomputed, ensuring that when DPM-Solver-v3 is applied, the computational overhead is avoided.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(9.7.1: Reducing First-Order Discretization Error Helps Higher-Order Solvers)</span></p>

Starting from the same initial condition $\mathbf{x}_{\lambda_s}$, let the approximated solution $\tilde{\mathbf{x}}_{\lambda_t}$ use the DPM-Solver-v3 update (with Taylor expansion of $\mathbf{N}_{\phi^\times}^{\text{new}}$), and the exact solution $\widetilde{\boldsymbol{\Psi}}_{\lambda_s \to \lambda_t}(\mathbf{x}_{\lambda_s})$ be given by the exact exponential-integrator formula. Then the discretization error depends on $\mathbf{N}_{\phi^\times}^{\text{new},(1)}$, and by controlling $\lVert\mathbf{N}_{\phi^\times}^{\text{new},(1)}\rVert_2$, we reduce $\lVert\tilde{\mathbf{x}}_{\lambda_t} - \widetilde{\boldsymbol{\Psi}}_{\lambda_s \to \lambda_t}(\mathbf{x}_{\lambda_s})\rVert_2$, assuming sufficient smoothness.

</div>

#### 9.7.5--9.7.6 Interpretations and Connections

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection of DPM-Solver-v3 to Other Methods)</span></p>

DPM-Solver-v3's $\mathbf{N}_{\phi^\times}^{\text{new}}$ is a general parametrization. By setting $\boldsymbol{\ell}_\lambda$, $\mathbf{a}_\lambda$, and $\mathbf{b}_\lambda$ to specific values, we can recover previous ODE formulations:

- $\boldsymbol{\epsilon}$-prediction: $(\boldsymbol{\ell}_\lambda, \mathbf{a}_\lambda, \mathbf{b}_\lambda) = (\mathbf{0}_D, -\mathbf{1}_D, \mathbf{0}_D)$
- $\mathbf{x}$-prediction: $(\boldsymbol{\ell}_\lambda, \mathbf{a}_\lambda, \mathbf{b}_\lambda) = (\mathbf{1}_D, \mathbf{0}_D, \mathbf{0}_D)$

**First-Order Discretization as an Improved DDIM.** Since $\mathbf{N}_{\phi^\times}^{\text{new}}$ represents neither noise nor data parameterization but an improved parameterization aimed at minimizing the first-order discretization error, the first-order DPM-Solver-v3 update differs from the standard DDIM update.

</div>

---

### 9.8 (Optional) ParaDiGMs

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(ParaDiGMs: Key Idea)</span></p>

ParaDiGMs (Shih et al., 2023) offers a complementary strategy to accelerate sampling by **parallelizing computations across different time intervals**, rather than processing them strictly in sequence. It reformulates the ODE solution as a fixed-point problem, allowing integral terms to be evaluated in parallel. Importantly, this approach is solver-agnostic: the fixed-point formulation wraps any time-stepping rule by replacing the integral with a weighted sum of model evaluations at selected times.

</div>

#### 9.8.1 From Time-Stepping to Time-Parallel Solver

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Trajectories to Picard Iteration as a Fixed-Point Update)</span></p>

The exact evolution from $T$ to any intermediate time $t$ is:

$$\widetilde{\boldsymbol{\Psi}}_{T \to t}(\mathbf{x}(T)) = \mathbf{x}(T) + \int_T^t \mathbf{v}_{\phi^\times}(\mathbf{x}(\tau), \tau)\,\mathrm{d}\tau, \quad \mathbf{x}(T) \sim p_{\text{prior}}.$$

This integral can be understood as a map $\mathcal{L}$ that takes an entire trajectory and produces a new one. A true solution trajectory $\mathbf{x}^*(\cdot)$ is a **fixed point** of $\mathcal{L}$. By **Picard iteration** (successive substitution), starting from any initial path $\mathbf{x}^{(0)}(\cdot)$ (e.g., a constant path $\mathbf{x}^{(0)}(t) \equiv \mathbf{x}^{(0)}(T)$):

$$\mathbf{x}^{(k+1)}(t) = \mathbf{x}^{(k)}(T) + \int_T^t \mathbf{v}_{\phi^\times}(\mathbf{x}^{(k)}(\tau), \tau)\,\mathrm{d}\tau, \quad k = 0, 1, 2, \ldots$$

This is parallel-friendly: each drift evaluation $\mathbf{v}_{\phi^\times}(\mathbf{x}_i^{(k)}, t_i)$ depends only on the previous iterate at the *same* time node, so all evaluations for $i = 0, \ldots, j-1$ can be computed independently across the grid.

</div>

#### 9.8.2 Methodology of ParaDiGMs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(ParaDiGMs with Sliding Windows)</span></p>

Place a uniform, decreasing grid on $[0, T]$ with $t_j := T - j\,\Delta t$, $j = 0, 1, \ldots, M$. The discrete Picard update is:

$$\mathbf{x}_j^{(k+1)} = \mathbf{x}_0^{(k)} - \Delta t\sum_{i=0}^{j-1}\mathbf{v}_{\phi^\times}(\mathbf{x}_i^{(k)}, t_i), \quad j = 1, \ldots, M.$$

To limit memory and exploit parallel hardware, ParaDiGMs applies the same idea locally on short **sliding blocks** of indices with window length $p$. The algorithm has three steps per window:

1. **Parallel Drift Evaluation:** Compute drifts at all $p$ window nodes in parallel using the previous iterate (Picard freezing).
2. **Left-Anchored Cumulative Updates:** Form cumulative sums (prefix-sum/scan) over the windowed drifts and update values.
3. **Progress Control and Window Advance:** Measure local convergence by the pointwise Picard change $\text{error}_j := \lVert\mathbf{x}_{\ell+j}^{(k+1)} - \mathbf{x}_{\ell+j}^{(k)}\rVert^2$ and advance the window up to the first unconverged node.

Increasing $p$ expands parallelism (more nodes advanced per window) without changing the overall step count $N$. When $p = 1$, the method collapses to a first-order time-stepping update (equivalent to DDIM with the same discretization choice).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compatibility with Higher-Order Solvers)</span></p>

The sliding-window Picard structure controls *how* increments are computed (in parallel and accumulated by a scan), not *which* local formula defines those increments. Consequently, one may replace the left-endpoint rule by any consistent higher-order quadrature without changing the parallel layout. For example, multistep or exponential-integrator updates used by DPM solvers can be inserted by replacing each windowed increment with the corresponding higher-order linear combination of past model evaluations. The parallel scheme is independent of the solver (discretization) choice; accuracy comes from the base solver, the windowed prefix-sum just makes it fast.

</div>

---

### 9.9 Closing Remarks

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Chapter 9: Key Takeaways)</span></p>

This chapter has confronted one of the most significant practical limitations of diffusion models: their slow, iterative sampling process. We explored a powerful class of training-free solutions that accelerate generation by leveraging the rich field of numerical methods for differential equations. The core strategy has been to more efficiently solve the PF-ODE, which defines the deterministic generative trajectory from noise to data.

1. We began with the foundational **DDIM**, which can be understood as a first-order exponential Euler method.
2. We then moved to higher-order multi-step methods like **DEIS**, which improve accuracy by using a history of past evaluations (Lagrange polynomial interpolation).
3. Finally, we examined the highly efficient **DPM-Solver** family, which achieves remarkable performance by introducing a crucial log-SNR time reparameterization, yielding Taylor-expansion-based updates with closed-form coefficients.

Through these sophisticated solvers, the number of function evaluations (NFEs) required for high-quality generation has been dramatically reduced from hundreds to as few as 10--20, making diffusion models significantly more practical.

However, these training-free methods are still fundamentally iterative. This raises a natural and ambitious question: *can we achieve high-quality generation in just one or a very few discrete steps?* The next chapters will explore this question through training-based acceleration (distillation and consistency training).

</div>
