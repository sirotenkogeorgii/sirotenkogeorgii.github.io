### Variational Autoencoder (VAE)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why not a plain autoencoder?)</span></p>

* A standard autoencoder has:

  * deterministic **encoder**: compresses $x$ into a low-dim code
  * deterministic **decoder**: reconstructs $x$
* It can reconstruct well, but the **latent space is unstructured**:

  * sampling random latent codes usually yields meaningless outputs
  * not a reliable **generative** model

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(VAE idea (Kingma & Welling, 2013))</span></p>

* Make the latent space **probabilistic + regularized**, so that:

  * sampling $z$ from a simple prior produces meaningful outputs
  * the model becomes a true generative model

</div>

### Probabilistic encoder and decoder

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/vae.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption><strong>Illustration of a VAE.</strong> It consists of a stochastic encoder $q_θ(z\mid x)$ that maps data $x$ to a latent variable $z$, and a decoder $p_ϕ(x\mid z)$ that reconstructs data from the latent.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Hidden Variables vs. Observable Variables)</span></p>

* **Observed variable:** $x$ (e.g., an image)
* **Latent variable:** $z$ (captures hidden factors: shape, color, style, $\dots$)

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Prior over latents)</span></p>

Typically a simple prior, e.g.

$$z \sim p(z) = p_{\text{prior}}(z) = \mathcal N(0, I)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Decoder / generator)</span></p>

Define a conditional likelihood ("decode latents into data"):

$$p_\phi(x \mid z)$$

In practice this is often kept **simple**, e.g. a **factorized Gaussian**, to encourage learning useful latent features rather than memorizing data.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sampling procedure)</span></p>

1. Sample $z \sim p(z)$
2. Sample $x \sim p_\phi(x \mid z)$

</div>

### Latent-variable marginal likelihood (why it’s hard)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Why do we need $p(z\mid x)$)</span></p>

Because once we observe a datapoint $x$, the relevant question is:

> **Which latent $z$'s could plausibly have produced this particular $x$?**

That is exactly what the posterior does:

$$p_\phi(z\mid x)=\frac{p_\phi(x\mid z)p(z)}{p_\phi(x)}$$

So $p(z\mid x)$ is not part of the model because we “like it aesthetically” — it is the mathematically correct conditional distribution over latent causes of $x$.

**Intuition**

Suppose $x$ is an image of a handwritten “3”.

* The prior $p(z)$ is broad: it includes latent codes for all digits, writing styles, thicknesses, rotations, etc.
* But after seeing **this specific image**, only a small subset of latent codes are plausible.
* That narrowed distribution is $p(z\mid x)$.

So posterior inference is:
**given data, infer the hidden explanation**.

**Why this matters for learning**

The marginal likelihood is

$$p_\phi(x)=\int p_\phi(x\mid z)p(z)dz$$

This integral averages over **all** latent codes. But for a fixed $x$, most $z$'s under the prior contribute almost nothing. The posterior tells you where the mass that actually explains $x$ is concentrated.

So $p(z\mid x)$ is useful because it identifies the “important” latent regions for a given observation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Latent-variable marginal likelihood (why it’s hard))</span></p>

**The VAE thereby defines a latent-variable generative model through the marginal likelihood:**

$$p_\phi(x) = \int p_\phi(x \mid z) p(z) dz$$

* Ideally, we would learn $\phi$ by maximizing $\log p_\phi(x)$ (MLE).
* But for expressive nonlinear decoders, the integral over $z$ is **intractable**, so **direct MLE is computationally infeasible**.

</div>

### Construction of the encoder (inference network)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(True posterior (intractable))</span></p>

To connect our intractable generator to real data, consider the reverse question: given an observation $x$, what latent codes $z$ could have produced it?

Given $x$, the "correct" latent posterior is:

$$p_\phi(z \mid x) = \frac{p_\phi(x \mid z) p(z)}{p_\phi(x)}$$

* The denominator $p_\phi(x)$ is exactly the intractable marginal likelihood, so **exact inference is prohibitive**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variational approximation)</span></p>

The **"variational" step in VAEs** addresses this by replacing the intractable posterior with a tractable approximation. Introduce a learnable approximate posterior (encoder):

$$q_\theta(z \mid x) \approx p_\phi(z \mid x)$$

* This gives a feasible, trainable pathway $x \to z$.

</div>

### The deeper reason VAEs need $q_\theta(z\mid x)$

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The deeper reason VAEs need $q_\theta(z\mid x)$)</span></p>

There are really **two problems** in VAEs:

**1. **Learning problem****

We want to maximize $\log p_\phi(x)$, but that integral over $z$ is hard.

**2. **Inference problem****

Given $x$, we want to infer which latent $z$ explains it.

The true answer to problem 2 is $p_\phi(z\mid x)$, but it is intractable because it depends on $p_\phi(x)$.

So VAEs introduce

$$q_\theta(z\mid x)\approx p_\phi(z\mid x)$$

to solve both problems at once:

* it gives an **encoder** from $x$ to latent code
* it provides a tractable proposal distribution concentrated on relevant $z$'s
* it yields the ELBO, a trainable lower bound on $\log p_\phi(x)$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lots of general models do not bother about $p(z\mid x)$)</span></p>

Only **latent-variable models** need this kind of posterior inference.

**Models that do *not* need $p(z\mid x)$**

These models do not introduce an unobserved latent $z$ that must be inferred for each datapoint:

* **Autoregressive models**
  They model $p(x)$ directly.
* **Normalizing flows**
  They define an invertible transformation, so likelihood is tractable without variational inference.
* **Standard discriminative models**
  They model $p(y\mid x)$, not hidden latent causes of $x$.
* **GANs**
  They use latent $z$ for generation, but typically do **not** define or optimize an explicit posterior $p(z\mid x)$.

**Models that *do* need something like $p(z\mid x)$**

Whenever you have hidden variables and want likelihood-based learning, posterior inference appears naturally:

* mixture models
* HMMs
* factor analysis
* VAEs
* general latent-variable probabilistic models

So the issue is not “general vs non-general models”.
The issue is:

> **Does the model contain hidden variables that must be inferred from observed data?**

If yes, then a posterior like $p(z\mid x)$ is central.

</div>

### Training via the Evidence Lower Bound (ELBO)

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(ELBO bound)</span></p>

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

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

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

</details>
</div>

### Interpreting the two ELBO terms

#### 1) Reconstruction term

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reconstruction term)</span></p>

$$\mathbb E_{z\sim q_\theta(z\mid x)}[\log p_\phi(x\mid z)]$$

* Encourages accurate recovery of $x$ from its latent code $z$.
* Under Gaussian encoder/decoder assumptions, this reduces to the familiar **reconstruction loss** of autoencoders.

</div>

#### 2) Latent KL regularization

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Latent KL regularization)</span></p>

$$D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel p(z))$$

* Encourages the encoder distribution to stay close to a simple prior $p(z)$ (e.g. $\mathcal N(0,I)$).
* Shapes the latent space to be smooth/continuous so samples from the prior decode meaningfully.

**Key trade-off:** good reconstructions vs. a well-structured latent space that supports sampling.

</div>

### Information-theoretic view: ELBO as a divergence bound

#### MLE view

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(MLE view)</span></p>

Maximum likelihood training corresponds to minimizing:

$$D_{\mathrm{KL}}(p_{\text{data}}(x)\parallel p_\phi(x))$$

which measures how well $p_\phi$ approximates the data distribution (but is generally intractable to optimize directly).

</div>

#### Joint-distribution trick (variational framework)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Joint-distribution trick (variational framework))</span></p>

Introduce two joint distributions:

* **Generative joint**
  
  $$p_\phi(x,z) = p(z)p_\phi(x\mid z)$$
  
* **Inference joint**
  
  $$q_\theta(x,z) = p_{\text{data}}(x)q_\theta(z\mid x)$$

Comparing them yields:

$$
D_{\mathrm{KL}}(p_{\text{data}}(x)\parallel p_\phi(x))
\le 
D_{\mathrm{KL}}(q_\theta(x,z)\parallel p_\phi(x,z)).
\qquad\text{(2.1.2)}
$$

**Intuition:** comparing only marginals over $x$ can hide mismatches that become visible when considering the full joint over $(x,z)$.

</div>

#### Chain rule / decomposition of the joint KL

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Chain rule / decomposition of the joint KL)</span></p>

Expanding the joint KL:

$$
\begin{aligned}
\underbrace{D_{\mathrm{KL}}(q_\theta(x,z)\parallel p_\phi(x,z))}_{\text{Total Error Bound}}
&= \mathbb{E}_{q_\theta(x,z)} \Big[ \log \dfrac{p_{\text{data}}(x)q_\theta(z\mid x)}{p_\phi(x)p_\phi(z\mid x)} \Big] \\
&= \mathbb{E}_{p_{\text{data}}(x)} \Big[ \log \dfrac{p_{\text{data}}(x)}{p_\phi(x)} + D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel p_\phi(z\mid x)) \Big] \\
&= \underbrace{D_{\mathrm{KL}}(p_{\text{data}}(x)\parallel p_\phi(x))}_{\text{True Modeling Error}} + \underbrace{\mathbb E_{p_{\text{data}}(x)} \Big[ D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel p_\phi(z\mid x))\Big]}_{\text{Inference Error}}.
\end{aligned}
$$

* First term: **true modeling error** (how well $p_\phi(x)$ matches data)
* Second term: **inference error** (gap between approximate and true posterior)

Because the inference error is nonnegative, you get inequality (2.1.2).

</div>

#### ELBO gap equals posterior KL

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(ELBO gap equals posterior KL)</span></p>

For each $x$,

$$
\log p_\phi(x) - \mathcal L_{\text{ELBO}}(\theta,\phi;x)
=

D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel p_\phi(z\mid x)).
$$

So **maximizing ELBO** is exactly **reducing the inference gap**, i.e. pushing the variational posterior toward the true posterior.

</div>

### Connection forward: hierarchical VAEs → DDPMs (conceptual bridge)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection forward: hierarchical VAEs → DDPMs (conceptual bridge))</span></p>

* **Hierarchical VAEs:** stack multiple latent layers to capture structure at multiple scales.
* **DDPMs as "many-layer" variational models:**

  * the forward noising process plays the role of a (fixed) encoder that gradually maps data to noise
  * the reverse denoising model is the learned decoder that inverts this mapping step-by-step
* The shared variational viewpoint: all optimize a **variational bound** on likelihood rather than the exact likelihood directly.

</div>

### Gaussian VAE (standard "Gaussian–Gaussian" VAE)

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

### ELBO specialization $\Rightarrow$ MSE reconstruction

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(ELBO specialization $\Rightarrow$ MSE reconstruction)</span></p>

Under this likelihood,

$$
\mathbb E_{q_\theta(z\mid x)}\big[\log p_\phi(x\mid z)\big]
= -\frac{1}{2\sigma^2}\ \mathbb E_{q_\theta(z\mid x)}\Big[\|x-\mu_\phi(z)\|^2\Big] + C,
$$

where $C$ is constant w.r.t. $\theta,\phi$.

So maximizing the ELBO is equivalent (up to constants/sign) to minimizing:

$$
\min_{\theta,\phi}\ \mathbb E_{q_\theta(z\mid x)}\Big[\frac{1}{2\sigma^2}\|x-\mu_\phi(z)\|^2\Big]
 + D_{\mathrm{KL}} \big(q_\theta(z\mid x)\parallel p_{\text{prior}}(z)\big).
$$

**Interpretation:** training becomes "regularized reconstruction":
* a **reconstruction loss** (scaled MSE),
* plus a **KL regularizer** pushing $q_\theta(z\mid x)$ toward the prior.

**Why KL is "easy" here:** for Gaussian $q_\theta$ (and typical Gaussian prior), the KL has a closed form (commonly used in implementations).

</div>

### Drawbacks of a standard VAE: blurry outputs

#### Why Gaussian VAEs often look blurry (core mechanism)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Gaussian VAEs often look blurry (core mechanism))</span></p>

Consider:

* a **fixed** Gaussian encoder $q_{\text{enc}}(z\mid x)$,
* and a Gaussian decoder with fixed variance
  
  $$p_{\text{dec}}(x\mid z)=\mathcal N(x;\mu(z),\sigma^2I)$$

With an arbitrary encoder, optimizing the ELBO (up to an additive constant) reduces to minimizing an expected squared error:

$$\arg\min_{\mu}\ \mathbb E_{p_{\text{data}}(x),q_{\text{enc}}(z\mid x)}\Big[\|x-\mu(z)\|^2\Big]$$

This is a least-squares regression problem in $\mu(z)$. The optimal solution is the **conditional mean**:

$$\mu^*(z)=\mathbb E_{q_{\text{enc}}(x\mid z)}[x]$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What is $q_{\text{enc}}(x\mid z)$?)</span></p>

It’s the "encoder-induced posterior on inputs given latents", obtained via Bayes’ rule:

$$q_{\text{enc}}(x\mid z)=\frac{q_{\text{enc}}(z\mid x)p_{\text{data}}(x)}{p_{\text{prior}}(z)}$$

An equivalent (often useful) form:

$$
\mu^*(z)
=\frac{\mathbb E_{p_{\text{data}}(x)}\big[q_{\text{enc}}(z\mid x),x\big]}
{\mathbb E_{p_{\text{data}}(x)}\big[q_{\text{enc}}(z\mid x)\big]}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Where blur comes from (mode averaging))</span></p>

If two distinct inputs $x\neq x'$ are mapped to **overlapping regions** in latent space (i.e., supports of $q_{\text{enc}}(\cdot\mid x)$ and $q_{\text{enc}}(\cdot\mid x')$ intersect), then for such a $z$,

$$\mu^*(z)=\mathbb E[x\mid z]$$

**averages across multiple (possibly unrelated) inputs**. Averaging "conflicting modes" produces **non-distinct, blurry** reconstructions/samples. Suppose that two distinct inputs $x \neq x'$ are mapped to overlapping regions in latent space, i.e., the supports of $q_{\text{enc}}(\cdot\mid x)$ and $q_{\text{enc}}(\cdot\mid x')$ intersect.

**Key takeaway:** with a Gaussian decoder + MSE-like training signal, the optimal prediction is a mean, and means of multimodal/ambiguous conditionals look blurry.

</div>

### From standard VAE to Hierarchical VAEs (HVAEs)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(HVAE motivation)</span></p>

Hierarchical VAEs introduce **multiple latent layers** to capture structure at different abstraction levels (coarse $\to$ fine). (Referenced: Vahdat & Kautz, 2020.)

</div>

<figure>
  <img src="{{ '/assets/images/notes/books/diffusion_models/hvae.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption><strong>Computation graph of the HVAE.</strong> It has a hierarchical structure with stacked, trainable encoders and decoders across multiple latent layers.</figcaption>
</figure>

#### Generative model (top-down hierarchy)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Generative model (top-down hierarchy))</span></p>

Introduce $z_{1:L}=(z_1,\dots,z_L)$. A common top-down factorization:

$$p_\phi(x,z_{1:L})  =  p_\phi(x\mid z_1)\ \prod_{i=2}^{L} p_\phi(z_{i-1}\mid z_i)\ p(z_L)$$

The marginal data density:

$$p_{\text{HVAE}}(x)  :=  \int p_\phi(x,z_{1:L}) dz_{1:L}$$

**Sampling/generation is progressive:**

1. sample top latent $z_L\sim p(z_L)$
2. decode downward $z_{L-1}\sim p_\phi(z_{L-1}\mid z_L)$, $\dots$, $z_1\sim p_\phi(z_1\mid z_2)$
3. generate $x\sim p_\phi(x\mid z_1)$

</div>

#### Inference model (bottom-up, mirrors hierarchy)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Inference model (bottom-up, mirrors hierarchy))</span></p>

A common structured encoder uses a bottom-up Markov factorization:

$$q_\theta(z_{1:L}\mid x)  =  q_\theta(z_1\mid x)\ \prod_{i=2}^{L} q_\theta(z_i\mid z_{i-1})$$

</div>

### HVAE ELBO

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(ELBO for HVAE)</span></p>

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

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$$
\log p_{\text{HVAE}}(x)
= \log \int p_\phi(x,z_{1:L}),dz_{1:L}
= \log \mathbb E_{q_\theta(z_{1:L}\mid x)}\Big[\frac{p_\phi(x,z_{1:L})}{q_\theta(z_{1:L}\mid x)}\Big]
$$

$$
\ge \mathbb E_{q_\theta(z_{1:L}\mid x)}\Big[\log \frac{p_\phi(x,z_{1:L})}{q_\theta(z_{1:L}\mid x)}\Big]
=: \mathcal L_{\text{ELBO}}
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

</details>
</div>

#### Interpretable decomposition (reconstruction + "adjacent" KLs)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretable decomposition (reconstruction + "adjacent" KLs))</span></p>

A key decomposition shown:

$$
\mathcal L_{\text{ELBO}}(x)
=
\mathbb E_q[\log p_\phi(x\mid z_1)]
-\mathbb E_q \Big[D_{\mathrm{KL}}(q_\theta(z_1\mid x)\parallel p_\phi(z_1\mid z_2))\Big]

$$

$$
-\sum_{i=2}^{L-1}\mathbb E_q \Big[D_{\mathrm{KL}}(q_\theta(z_i\mid z_{i-1})\parallel p_\phi(z_i\mid z_{i+1}))\Big]
-\mathbb E_q \Big[D_{\mathrm{KL}}(q_\theta(z_L\mid z_{L-1})\parallel p(z_L))\Big],
$$

where $\mathbb E_q$ denotes expectation under the encoder-induced joint over $(x,z_{1:L})$ (as written in the text).

**Meaning:** each inference conditional is regularized toward its corresponding **top-down** conditional prior:

+ $q(z_1\mid x)$ vs $p(z_1\mid z_2)$,
+ $q(z_i\mid z_{i-1})$ vs $p(z_i\mid z_{i+1})$,
+ top level $q(z_L\mid z_{L-1})$ vs $p(z_L)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(The core insight is simple yet powerful)</span></p>

Stacking layers lets the model generate **progressively** (coarse $\to$ fine), which helps capture complex high-dimensional structure.

</div>

### Why "just make a flat VAE deeper" is not enough

#### Limitation 1: the variational family is still too simple

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Limitation 1: the variational family is still too simple)</span></p>

In a standard flat VAE,

$$q_\theta(z\mid x)=\mathcal N\big(z;\mu_\theta(x),\mathrm{diag}(\sigma_\theta^2(x))\big)$$

is **one unimodal Gaussian** per $x$. Making networks deeper can improve $\mu_\theta,\sigma_\theta$, but does **not** change the fact that the posterior family is unimodal (even full-covariance remains a single ellipsoid).

If the true posterior $p_\phi(z\mid x)$ is **multi-peaked**, this mismatch loosens the ELBO and weakens inference. Fix needs a **richer posterior class**, not just deeper nets.

</div>

#### Limitation 2: posterior collapse with an expressive decoder

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Limitation 2: posterior collapse with an expressive decoder)</span></p>

Recall the expected objective:

$$
\mathbb E_{p_{\text{data}}(x)}[\mathcal L_{\text{ELBO}}(x)]
=
\mathbb E_{p_{\text{data}}(x),q_\theta(z\mid x)}[\log p_\phi(x\mid z)]
-\mathbb E_{p_{\text{data}}(x)}[D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel p(z))].
$$


This can be rewritten as:

$$\mathbb E_{p_{\text{data}}(x),q_\theta(z\mid x)}[\log p_\phi(x\mid z)] -\mathcal I_q(x;z) -D_{\mathrm{KL}}(q_\theta(z)\parallel p(z))$$

where

$$\mathcal I_q(x;z)=\mathbb E_{q(x,z)}\Big[\log \frac{q_\theta(z\mid x)}{q_\theta(z)}\Big] =\mathbb E_{p_{\text{data}}(x)}\Big[D_{\mathrm{KL}}(q_\theta(z\mid x)\parallel q_\theta(z))\Big]$$

and the aggregated posterior is

$$q_\theta(z)=\int p_{\text{data}}(x)q_\theta(z\mid x)dx$$

**Collapse story:** if the decoder can model the data well **without using $z$** (i.e., effectively $p_\phi(x\mid z)\approx r(x)\approx p_{\text{data}}(x)$), then an ELBO maximizer can choose

$$q_\theta(z\mid x)=p(z)$$

making $\mathcal I_q(x;z)=0$ and $q_\theta(z)=p(z)$. Then $z$ carries no information about $x$, and changing $z$ doesn’t affect outputs (controllability fails). Making the networks deeper does not automatically remove this "ignore $z$" solution.

</div>

### What hierarchy changes (and what new issues appear)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What improves conceptually)</span></p>

The HVAE ELBO uses **multiple adjacent KL terms**, so the "information penalty" is:

* **distributed across layers**, and
* **localized** (each layer matches to its neighbor’s conditional prior),
  which comes from the hierarchical latent graph—not simply from depth in the encoder/decoder networks.

</div>

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

### Posterior Collapse in VAEs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Posterior Collapse in classical VAEs)</span></p>

In a **classical VAE**, posterior collapse means the encoder stops putting meaningful information about $x$ into $z$. Formally, for many $x$, the approximate posterior becomes almost the prior,

$$q_\phi(z\mid x)\approx p(z)$$

so the KL term is near zero and the decoder effectively ignores $z$. This is the standard “latent variable ignored” failure mode described in the VAE literature.

</div>

#### Explanation 1: Too Expressive Decoder

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Too Expressive Decoder)</span></p>

The easiest way to see **why an expressive decoder causes this** is to look at the ELBO:

$$\mathcal L(x) = \mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] D_{\mathrm{KL}}(q_\phi(z\mid x)|p(z))$$

Using $z$ helps only if it improves the reconstruction term enough to justify paying the KL cost. But if the decoder is very strong, it can often model $x$ well **without** relying much on $z$. Then the optimizer gets an easy win: keep reconstruction good using the decoder alone, and drive the KL term toward zero by making $q_\phi(z\mid x)$ close to $p(z)$. That is collapse. Bowman et al. describe this in sequence VAEs as the model initially learning to ignore $z$ and explain the data with the "more easily optimized decoder," after which little gradient passes between encoder and decoder and training settles into a zero-KL equilibrium.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Mechanism</span><span class="math-callout__name">(Core Mechanism of prior collapse with strong decoder)</span></p>

So the core mechanism is:

1. **Strong decoder** can already model $x$ well.
2. **Latents cost bits** through the KL term.
3. The ELBO prefers “don’t use $z$” unless $z$ gives a clear enough reconstruction benefit.
4. Encoder collapses toward the prior, decoder learns to ignore $z$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remakr</span><span class="math-callout__name">(Information-theoretic view)</span></p>

An information-theoretic way to say the same thing is: the KL term is the **rate** spent to transmit information about $x$ through $z$. If the decoder can reconstruct well at very low rate, the optimum can sit near rate $0$. Alemi et al. show that there can even be a family of models with the **same ELBO** but very different mutual information $I(X;Z)$, and they explicitly note that powerful stochastic decoders may ignore the latent code.

</div>

#### Explanation 2: Local Dependencies Oversignal the Codes

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Local Dependencies)</span></p>

For **autoregressive decoders** this is especially severe. In text VAEs, the decoder predicts each token from previous tokens, so it already has a very strong signal from local context. Bowman et al. found that training without special tricks “reliably results” in models with performance similar to a baseline RNN language model and **zero KL divergence**; weakening the decoder with word dropout forces it to use $z$ more.

For autoregressive decoders

$$p_\theta(x\mid z)=\prod_i p_\theta(x_i\mid x_{<i},z)$$

then a sufficiently powerful autoregressive network can set

$$p_\theta(x_i\mid x_{<i},z)\approx p_{\text{data}}(x_i\mid x_{<i})$$


A very clean statement of the same idea appears in the Variational Lossy Autoencoder paper. It notes that earlier attempts to combine VAEs with autoregressive models ran into the problem that the autoregressive part explains all the structure while the latents are unused, and for an autoregressive RNN decoder, the model can in principle represent any distribution over $x$ even without dependence on $z$. It then states the key point directly: if $p(x\mid z)$ can model the data distribution without using information from $z$, then it will not use $z$, the true posterior becomes the prior, and it is easy to set $q(z\mid x)=p(z)$.

A good intuition is:
* $z$ is supposed to carry global summary information
* a strong decoder may already extract enough structure from local context or its own capacity
* then passing information through $z$ is redundant and costly

</div>

#### Explanation 3: Local Minima

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Local Minima)</span></p>

One subtle but important correction: the “expressive decoder bypasses $z$” story is very important, but it is **not the whole story**. Lucas et al. show that posterior collapse is not only an artifact of the ELBO or amortized inference; collapsed solutions can also be related to local optima of the underlying marginal likelihood itself. So in practice there are often **two effects together**: a decoder that makes $z$ unnecessary, and training dynamics or local optima that make collapse hard to escape once it starts.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary on explanations of posterior collapse)</span></p>

A good intuition is this:

* The latent variable $z$ is supposed to carry **global information**.
* A powerful decoder can often reconstruct $x$ from **local dependencies** or its own expressive dynamics.
* Then the model says: "Why pay KL to encode information in $z$ if I can already do the job without it?"

</div>

#### Discussion: How could VAE ignore the code, if I want to use it as a control signal

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Control ignoring in VAR)</span></p>

You want $z$ to act like a **control signal**:

* $z=z_1$ $\to$ generate a $5$
* $z=z_2$ $\to$ generate a $9$

But the **vanilla VAE objective does not require that**.

It only requires the model to fit the **marginal data distribution**

$$p_\theta(x)=\int p_\theta(x\mid z)p(z)dz$$

That means the optimizer is happy as long as the model generates the right **overall population of images**, even if $z$ is useless.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Control intuition vs. what ELBO actually asks for)</span></p>

What you want is something like:

> “$z$ should determine the semantic content of the sample.”

But ELBO only says:

> “Make the generated samples follow the training distribution, and don’t use more information in $z$ than necessary.”

So if the decoder can already generate both 5s and 9s by itself, then the model does **not** need $z$ to decide "$5$ or $9$".

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Generating digits 5s and 9s)</span></p>

Suppose your dataset is 50% 5s and 50% 9s.

A collapsed model can learn something like:

$$p_\theta(x\mid z)= \text{distribution over images that are 50% 5s and 50% 9s}$$

for every $z$.

Then:

* changing $z$ does nothing
* but samples from the decoder still look like valid digits
* overall the generated dataset still matches the true data distribution

So from the ELBO viewpoint, this may already be good enough.

What is lost is **control** and **representation**, not necessarily sample quality.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this feels wrong intuitively: the code only serves to help)</span></p>

It feels natural to think:

> “If I want to generate a specific kind of sample, I need $z$.”

That is true for **controllable generation**.

But for **unconditional generation**, the model only needs to produce samples from the right distribution overall. It does not need a human-interpretable knob saying "now generate a $5$".

The latent variable is one possible mechanism for organizing generation — but not the only one.

A strong decoder may organize generation internally and leave $z$ unused.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(If you want $z$ to actually control semantics)</span></p>

Then you usually need extra pressure beyond the plain ELBO, such as:

* weakening the decoder
* KL annealing / free bits
* $\beta$-VAE style changes
* mutual-information-promoting objectives
* supervision or semi-supervision on attributes/classes

These methods try to force or encourage $z$ to carry meaningful information like "$5$ vs $9$".

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary on the discrepance between contol signal and ELBO)</span></p>

* **yes**, your intended use of $z$ is to guide what gets generated
* **but** vanilla VAE training does not guarantee that
* with an expressive decoder, the model can generate 5s and 9s in the right proportions **without storing that choice in $z$**

</div>

#### How is it possible that a decoder produces the true data distribution without codes?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Powerfull decoder / generator does not need codes)</span></p>

If the decoder is powerfull enough to store the needed information to produce the true data distribution directly in the weights, then there is not need to rely on the codes as an external source of information about the true data distribution. The code basically serves as a **signal to generate something, inside of a control signal**.

In the prior collapse case we for any code $z$ we get the same output distribution, it means that any vector $z$ or any noise as an input will be mapped to the same parameters defining the output distribution: **any latent vector is treated the same**.

For example, if the decoder is Gaussian, then 

$$p_\theta(x\mid z) = \mathcal{N}(\mu_\theta(z), \Sigma_\theta(z))$$

then collapse means roughly 

$$\mu_\theta(z) \approx \mu_0 \quad \Sigma_\theta(z) \approx \Sigma_0$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sometimes only part of $z$ collapses)</span></p>

You may have a $32$-dimensional latent vector, but only $3$ coordinates are actually used and the other $29$ are ignored. So **collapse does not always mean "all of $z$ is useless.**"

It can mean:
* some latent dimensions carry information,
* many others are effectively mapped away.

</div>

### Classis VAE Decoder vs. Classic Autoencoder Decoder

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Decoder in classic VAE)</span></p>

**the parameters of** $p_\theta(x\mid z)$ are deterministic functions of $z$, but the **sample $x$** from that distribution is generally **not** deterministic.

**The clean separation**

For a fixed $z$, the decoder does this in two steps:

**Step 1: deterministic**

Compute the parameters of the conditional distribution from $z$.

For example, in a Gaussian decoder:

$$\mu_\theta(z),\ \sigma_\theta(z)$$

are deterministic functions of $z$.

**Step 2: stochastic**

Then define or sample

$$x \sim p_\theta(x\mid z)$$

So for a fixed $z$, the distribution $p_\theta(x\mid z)$ is fixed, but $x$ itself is still random.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Collorary</span><span class="math-callout__name">(Comparison to the decoder of the classic autoencoder)</span></p>

In implementation, the decoder of the classic autoencoder often outputs a tensor that looks like a reconstructed image, but in the probabilistic VAE interpretation that tensor is usually the parameter of a distribution, not necessarily the final sampled image itself.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian decoder)</span></p>

Suppose

$$p_\theta(x\mid z)=\mathcal N(\mu_\theta(z), \sigma^2 I)$$

Then for fixed $z$:

* (\mu_\theta(z)) is deterministic
* (\sigma^2) is fixed
* but (x) sampled from that Gaussian is random

So the decoder is **deterministic as a map**

$$z \mapsto (\mu,\sigma)$$

but **stochastic as a map**

$$z \mapsto x$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bernoulli decoder)</span></p>

For binary images, often

$$p_\theta(x\mid z)=\prod_i \mathrm{Bernoulli}(x_i; \pi_i(z))$$

For fixed (z):

* the probabilities (\pi_i(z)) are deterministic
* but each pixel (x_i) is still random

Again: deterministic distribution parameters, stochastic sample.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The decoder ignores $z$)</span></p>

The deterministic output distribution parameters become almost the same for different $z$'s.

So for many $z_1,z_2$,

$$p_\theta(x\mid z_1)\approx p_\theta(x\mid z_2)$$

For a Gaussian decoder, that would mean roughly

$$
\mu_\theta(z_1)\approx \mu_\theta(z_2),
\qquad
\sigma_\theta(z_1)\approx \sigma_\theta(z_2).
$$

</div>
