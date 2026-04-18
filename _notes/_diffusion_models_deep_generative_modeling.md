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

$$
\begin{aligned}
D_{\mathrm{KL}}(p_{\text{data}}\parallel p_\phi) := \int p_{\text{data}}(x)\log\frac{p_{\text{data}}(x)}{p_\phi(x)}dx \\
&= \mathbb{E}_{x\sim p_{\text{data}}}\left[\log p_{\text{data}}(x)-\log p_\phi(x)\right]
\end{aligned}
$$

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
