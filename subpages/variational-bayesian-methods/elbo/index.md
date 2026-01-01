---
title: Evidence Lower Bound (ELBO)
layout: default
noindex: true
---

# Evidence Lower Bound (ELBO)

In **variational Bayesian methods**, the **evidence lower bound** (often abbreviated **ELBO**, also sometimes called the **variational lower bound** or **negative variational free energy**) is a useful lower bound on the log-likelihood of some observed data.

The ELBO is useful because it provides a guarantee on the worst-case for the log-likelihood of some distribution (e.g. $p(X)$) which models a set of data. The actual log-likelihood may be higher (indicating an even better fit to the distribution) because the ELBO includes a **Kullback-Leibler divergence** (KL divergence) term which decreases the ELBO due to an internal part of the model being inaccurate despite good fit of the model overall. Thus improving the ELBO score indicates either improving the likelihood of the model $p(X)$ or the fit of a component internal to the model, or both, and the ELBO score makes a good loss function, e.g., for training a deep neural network to improve both the model overall and the internal component. (The internal component is $q_{\phi}(\cdot \mid x)$, defined in detail later in this article.)

## 1. The core problem ELBO is trying to solve

In **latent-variable probabilistic models**, we often define a joint distribution over **observed data** $x$ and **latent variables** $z$:

$$p_\theta(x,z) = p_\theta(x\mid z)p(z),$$

and we want the **marginal likelihood** (a.k.a. evidence):

$$p_\theta(x) = \int p_\theta(x,z)dz.$$

**INTRACTABILITY.** In general, that integral has no closed form and is the bottleneck behind “exact” Bayesian inference in latent-variable models. For many interesting models (e.g., with neural-network likelihoods), this integral is intractable to compute exactly.

**INTERPRETATION.** When practitioners say “$p_\theta(x)$ or $\log p_\theta(x)$  is intractable,” they usually mean “the integral is high-dimensional or ugly enough that exact evaluation (and, crucially, exact *differentiation*) is not practical at scale.” That’s a computational statement, not a theorem about impossibility.

### Connection to Bayesian posterior inference

Bayesian posterior inference is similarly hard because

$$p_\theta(z\mid x) = \frac{p_\theta(x,z)}{p_\theta(x)}$$

depends on the same intractable evidence term $p_\theta(x)$.

### Interpretation (not a new fact)

So we’re blocked in two places at once:

* We can’t evaluate $p_\theta(x)$ (so maximum likelihood learning is hard).
* We can’t evaluate $p_\theta(z\mid x)$ (so posterior inference is hard).

The ELBO is the standard move that “unblocks” both by turning inference into optimization.

## 2. Introducing a variational approximation

Variational inference introduces a tractable approximation $q_\phi(z\mid x)$ to the true posterior $p_\theta(z\mid x)$. This $q_\phi$ can be any distribution family you can sample from / evaluate (often parameterized by a neural net). (Some sources write $q(z)$ without conditioning; that’s mostly a notational choice we’ll reconcile later.)

Given any choice of $q_\phi(z\mid x)$, we can rewrite the log evidence:

$$
\log p_\theta(x) = \log \int p_\theta(x,z)dz
= \log \int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}dz
= \log \mathbb{E}_{z\sim q_\phi(z\mid x)}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right].
$$

A Wikipedia-style definition packages the ELBO as

$$\mathcal L(\phi,\theta;x) =  \mathbb E_{z\sim q_\phi(\cdot\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right] = \mathbb E_{q}\big[\log p_\theta(x,z)\big] + H[q_\phi(\cdot\mid x)].$$

This same expression can be rewritten as

$$\mathcal L(\phi,\theta;x) = \log p_\theta(x) - \mathrm{KL}\big(q_\phi(z\mid x)\mid p_\theta(z\mid x)\big),$$

which immediately implies $\mathcal L(\phi,\theta;x)\le \log p_\theta(x)$ because KL divergence is nonnegative. ([Wikipedia][2])

A Princeton blog note by Ryan Adams presents the key “accounting identity” in the opposite direction:

$$\log p_\theta(x)= \mathrm{KL}(q(z)\mid p(z\mid x,\theta)) + \mathrm{ELBO}(q),$$

so the ELBO is exactly “log evidence minus a gap,” and the gap is a KL divergence.

**INTERPRETATION.** Mentally, you can treat ELBO maximization as doing two things at once:
1. pushing $q$ toward the true posterior (shrinking the KL gap),
2. adjusting $\theta$ to make the model explain the data well. Which of these dominates in practice depends on parameterization and optimization, not on the algebra.

or **Maximizing ELBO** (with $\theta,\phi$) is equivalent to:
  * pushing up $\log p_\theta(x)$ (better generative model), and
  * pushing down $\mathrm{KL}(q_\phi(z\mid x)\mid p_\theta(z\mid x))$ (better inference).
  
## 3. Three derivations that look different but are the same proof in different clothing

All sources ultimately cash out to the same inequality $\log p(x)\ge \mathrm{ELBO}$. What differs is *which inequality you “pay for”* to get it.

### 3.1 KL-first derivation (no Jensen required)

The Adams note starts from $\mathrm{KL}(q(z)\mid p(z\mid x,\theta))$, expands it, and rearranges terms to isolate $\log p(x\mid\theta)$ on one side and the ELBO on the other. The only “inequality step” is the nonnegativity of KL.

**FACT.** This route makes the “ELBO gap” explicit: the bound is tight iff $q(z)=p(z\mid x,\theta)$ almost everywhere.

### 3.2 Jensen derivation (log of expectation vs expectation of log)

A common teaching path (used in Yunfan’s blog post) inserts $q(z\mid x)/q(z\mid x)$ into the marginal likelihood integral, rewrites it as a log of an expectation under $q$, then applies Jensen:

$$\log p(x) = \log \mathbb E_{z\sim q}\left[\frac{p(x,z)}{q(z\mid x)}\right] \ge \mathbb E_{z\sim q}\left[\log\frac{p(x,z)}{q(z\mid x)}\right].$$

That right-hand side is the **Evidence Lower Bound (ELBO)**. Yunfan’s write-up explicitly shows the identity $\log p(x)=\mathcal L + \mathrm{KL}(q\mid p)$ and then turns it into the inequality via KL nonnegativity (another equivalent route).

**INTERPRETATION.** Jensen is pedagogically nice because it highlights the “swap log and integral” issue: $\log\int \cdot \neq \int \log(\cdot)$.

### 3.3 A convex-duality / Hölder-flavored view

A shorter, more geometric blog post (cgad.ski) frames $\log \int e^{H}$ as a convex functional whose tangent lower bounds are expectations under a Gibbs/Boltzmann distribution. After normalizing, it arrives at an inequality of the form

$$v(G)\ge \mathbb E_H[G-H],$$

and identifies this as the ELBO with slack equal to a KL divergence. ([cgad.ski][4])

This perspective explicitly connects ELBO to “tangent-line” style lower bounds from convexity (and links the slack to KL). ([cgad.ski][4])

**INTERPRETATION.** If you like Legendre transforms, you can read the ELBO as a variational (dual) representation of a log-partition-like quantity. If you don’t, you can ignore this entirely and still do VI.

## 4. The ELBO as two coupled problems: inference and learning

Write the “main form” (also listed by Wikipedia) as

$$\mathcal L(\phi,\theta;x) = \underbrace{\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]}_{\text{fit / reconstruction}} - \underbrace{\mathrm{KL}(q_\phi(z\mid x)\mid p_\theta(z))}_{\text{regularize toward prior}}$$

This makes the coupled roles explicit:

* Optimize $\phi$: make $q_\phi(z\mid x)$ approximate the posterior (reduce the ELBO gap). Also I like the point of view through **complexity / regularization** penalty that limits how much information $z$ can and should carry about $x$, given the prior $p_\theta$.
* Optimize $\theta$: make the generative model explain data well under latent samples from $q_\phi$.

Wikipedia explicitly connects this to amortized inference: learning $q_\phi(z\mid x)$ lets you infer $z$ cheaply for new $x$.

**INTERPRETATION.** The “reconstruction vs KL” decomposition encourages an engineering view: you can often diagnose training pathologies by checking which term dominates or collapses. (This is widely used practice, but it’s not a theorem that *every* pathology shows up cleanly here.)

## 5. Notation reconciliation: $q(z)$ vs $q(z\mid x)$, and what the ELBO “really” is

Across the sources:
* Wikipedia and Yunfan use **amortized** notation $q_\phi(z\mid x)$: one inference model produces a different approximate posterior per datapoint $x$.
* The Adams note uses $q(z)$ for simplicity, essentially treating $x$ as fixed and focusing on a single posterior approximation.
* Some Cross Validated discussion is triggered by tutorials that mix measures and densities, or write KL with objects that don’t type-check.

A detailed Cross Validated answer (postylem) points out a recurring notational bug: you can’t meaningfully write “KL between a measure $\mathbb P$ and a pdf $q$”—KL is defined between distributions (or between densities w.r.t. the same base measure), not “distribution vs. density” as mismatched types.

**FACT.** A safe, standard statement is: if both $q$ and $p$ are absolutely continuous w.r.t. a common base measure and have densities $q(z)$ and $p(z)$, then

$$\mathrm{KL}(q\mid p)=\int q(z)\log\frac{q(z)}{p(z)},dz.$$

(That is exactly the kind of correction being emphasized in that answer.)

**Resolution of an apparent contradiction.** One Cross Validated answer writes an ELBO-looking expression as $\log p(x_i\mid\theta,z_i)-\mathrm{KL}(q(z_i)|p(z_i\mid\theta))$. Read this as a *shorthand* for the more standard

$$\mathbb E_{q(z_i)}[\log p(x_i\mid z_i,\theta)] - \mathrm{KL}(q(z_i)\mid p(z_i\mid\theta)),$$

i.e., the “expected log-likelihood minus KL to the prior” form that Wikipedia also lists among “main forms.”
































## 5. Tightness: when is the bound exact?

### Fact

ELBO becomes equal to $\log p_\theta(x)$ if and only if

$$q_\phi(z\mid x) = p_\theta(z\mid x)$$

(almost everywhere), because the gap is exactly the KL divergence to the true posterior. ([Wikipedia][1])

### Implication (still factual, derived directly from the identity)

Even if you optimize perfectly within your chosen family for $q_\phi$, the bound can remain loose when:

* the true posterior is too complex (multi-modal, heavy-tailed, etc.),
* but your variational family is restricted (e.g., diagonal Gaussian).

## 6. “Inference becomes optimization”: what you actually optimize

There are two optimization roles:

### Fact (variational step)

For a fixed generative model $p_\theta$, choosing $q_\phi$ to maximize ELBO is equivalent to minimizing

$$D_{\mathrm{KL}}(q_\phi(z\mid x)\mid p_\theta(z\mid x)).$$

This is the variational approximation step. ([Wikipedia][1])

### Fact (learning step)

For fixed $q_\phi$, maximizing ELBO w.r.t. $\theta$ increases a lower bound on the log marginal likelihood, so it is a principled surrogate objective for learning $\theta$. ([ar5iv][2])

### Opinion (practical perspective)

In modern deep learning, people often treat ELBO primarily as a **training objective** rather than a bound whose tightness they rigorously diagnose. This is productive engineering-wise, but it can hide when improvements come from better modeling vs. simply “gaming the bound” with a limited $q$.

## 7. The bottleneck: gradients through expectations

ELBO contains expectations over $z\sim q_\phi(z\mid x)$. Optimizing w.r.t. $\theta$ is often straightforward; optimizing w.r.t. $\phi$ can be hard if you rely on naïve Monte Carlo gradient estimators.

### Fact

The VAE paper highlights that a naïve score-function/REINFORCE-style estimator can have high variance, motivating an alternative estimator. ([ar5iv][2])

## 8. The reparameterization trick (SGVB / AEVB)

### Fact

The key idea in *Auto-Encoding Variational Bayes* is to express sampling from $q_\phi(z\mid x)$ as a deterministic transformation of noise:

$$\epsilon \sim p(\epsilon), \qquad z = g_\phi(x,\epsilon),$$

so that expectations over $z$ become expectations over $\epsilon$, and gradients can flow through $g_\phi$. ([ar5iv][2])

### Fact

For the common diagonal-Gaussian posterior,

$$z = \mu_\phi(x) + \sigma_\phi(x)\odot \epsilon,\qquad \epsilon\sim\mathcal{N}(0,I),$$

which makes stochastic gradient optimization practical at scale. ([ar5iv][2])

### Interpretation

This is why VAEs were a turning point historically: ELBO existed long before, but this trick made it **work smoothly with backprop** and minibatches.

## 9. Amortized inference: why the encoder exists

Classical VI could optimize separate variational parameters per datapoint. The VAE approach “amortizes” by learning a single inference network that maps $x \mapsto q_\phi(z\mid x)$.

### Fact

The VAE paper explicitly frames this as fitting a “recognition model” (encoder) jointly with the generative model (decoder), enabling efficient approximate posterior inference via ancestral sampling, without iterative per-datapoint inference loops. ([ar5iv][2])

### Opinion

Amortization is a trade: you get speed and scalability, but you can introduce an **amortization gap** (the encoder family and training dynamics may prevent reaching the best per-datapoint variational optimum).

## 10. Resolving overlaps and apparent contradictions across your sources

Your three sources largely agree; most “contradictions” are actually differences in emphasis or terminology. Here are the main ones people stumble over:

### 10.1 “ELBO is a bound on likelihood” vs “bound on log-likelihood”

### Fact

Mathematically, ELBO is a lower bound on **$\log p_\theta(x)$** (the log evidence / log marginal likelihood), not directly on $p_\theta(x)$. That’s the standard statement in definitions and derivations. ([Wikipedia][1])

### Resolution

When someone casually says “a lower bound on the likelihood,” they usually mean “on the *log*-likelihood,” because training almost always uses logs.

### 10.2 “ELBO includes a KL term which decreases it” (Wikipedia phrasing)

### Fact

The identity

$$\text{ELBO} = \log p_\theta(x) - D_{\mathrm{KL}}(q_\phi(z\mid x)\mid p_\theta(z\mid x))$$

means the bound is smaller than $\log p_\theta(x)$ by exactly a KL divergence. ([Wikipedia][1])

### Resolution

This is not saying KL is “bad”; it’s saying **the looseness of the bound** is measured by that KL. If you can make $q$ match the true posterior, KL goes to zero and the bound tightens.

### 10.3 “ELBO as free energy” vs “ELBO as a training loss”

### Fact

ELBO is also referred to as the **negative variational free energy** in some literature, reflecting a connection to entropy and energy-based decompositions. ([Wikipedia][1])

### Resolution

Same object, different coordinate system:

* Physics-ish view: maximize negative free energy.
* ML view: maximize ELBO or minimize (-)ELBO as a loss.

### 10.4 Direction of KL: why $D_{\mathrm{KL}}(q\mid p)$ and not $D_{\mathrm{KL}}(p\mid q)$?

### Fact

The bound derived via Jensen yields $D_{\mathrm{KL}}(q_\phi(z\mid x)\mid p_\theta(z\mid x))$ (the “exclusive” KL) naturally. ([Wikipedia][1])

### Opinion (common intuition that helps)

This KL direction tends to produce **mode-seeking** behavior (it heavily penalizes placing probability mass where the true posterior has little mass, sometimes under-covering multiple modes). This is not “wrong,” but it explains why simple variational families can miss posterior multi-modality.

## 11. Practical takeaways for technical work

### Fact-backed guidance

* **If you can compute the KL term analytically**, do it: the VAE paper notes this often reduces estimator variance because only the reconstruction expectation needs Monte Carlo. ([ar5iv][2])
* **The ELBO gap is diagnostic**: the gap to $\log p_\theta(x)$ is a KL to the true posterior, so a poor variational family can make ELBO improvements misleading about true likelihood improvements. ([Wikipedia][1])

### Opinion (engineering heuristics)

* If you see posterior collapse (encoder ignores $x$), it’s often because the KL term is “too easy” to minimize and overwhelms reconstruction. The ELBO decomposition makes that failure mode unsurprising.
* When comparing models, treat ELBO as **a training objective and a lower bound**, not a guaranteed proxy for downstream representation quality.



---

## Appendix

## Appendix (i): Deriving the “expected log-likelihood minus KL-to-prior” form

Start from the **canonical definition** of the ELBO for a latent-variable model $p_\theta(x,z)$ and an auxiliary distribution $q_\phi(z\mid x)$:

$$
\mathcal L(\phi,\theta;x)
:=\mathbb E_{z\sim q_\phi(\cdot\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
=\mathbb E_q[\log p_\theta(x,z)]-\mathbb E_q[\log q_\phi(z\mid x)].
$$

This “log joint plus entropy” view (and its equivalent “evidence minus KL-to-posterior” identity) is stated explicitly in the Wikipedia definition section.

Now apply the **model factorization**:

$$
p_\theta(x,z)=p_\theta(x\mid z),p_\theta(z)
\quad\Rightarrow\quad
\log p_\theta(x,z)=\log p_\theta(x\mid z)+\log p_\theta(z).
$$

Plugging into the ELBO:

$$\mathcal L(\phi,\theta;x) = \mathbb E_q[\log p_\theta(x\mid z)] \mathbb E_q[\log p_\theta(z)-\log q_\phi(z\mid x)].$$

Recognize the second term as **(minus) a KL divergence to the prior**:

$$
\mathrm{KL}\big(q_\phi(z\mid x)\mid p_\theta(z)\big)
= \mathbb E_q\left[\log \frac{q_\phi(z\mid x)}{p_\theta(z)}\right]
= \mathbb E_q[\log q_\phi(z\mid x)]-\mathbb E_q[\log p_\theta(z)].
$$

Rearrange:

$$\mathbb E_q[\log p_\theta(z)-\log q_\phi(z\mid x)] = -\mathrm{KL}\big(q_\phi(z\mid x)\mid p_\theta(z)\big).$$

So you get the “main” engineering-friendly decomposition:

$$\boxed{\mathcal L(\phi,\theta;x) = \mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] -\mathrm{KL}!\big(q_\phi(z\mid x),|,p_\theta(z)\big).}$$

### Facts vs opinions about this form

* **FACT.** It is algebraically equivalent to the canonical ELBO definition (no approximations were made).
* **FACT.** It makes clear which parts need estimation: an expectation under $q_\phi(z\mid x)$ plus a KL term (sometimes closed-form for convenient families). Wikipedia notes that sampling $z\sim q_\phi(\cdot\mid x)$ makes the log-ratio term an unbiased estimator of the ELBO, and also notes a common special case where the KL has a closed form (e.g., Gaussian–Gaussian).
* **OPINION.** This decomposition is the best one to debug training because you can watch “fit” and “regularize” terms separately (but it can still hide issues like high-variance gradients in the expectation term).

## Appendix (ii): What “amortized inference” changes, mathematically and algorithmically

### 1. Non-amortized (“per-datapoint”) variational inference

A classic VI setup introduces **variational parameters per datapoint**. Write $q_{\lambda_i}(z_i)$ for datapoint $x_i$, with its own parameter $\lambda_i$. You then solve

$$\lambda_i^*(\theta) \in \arg\max_{\lambda_i}; \mathcal L(\lambda_i,\theta; x_i),$$

and separately optimize $\theta$ using those $\lambda_i$’s.

A very common computational pattern is **coordinate ascent / EM-like alternation**:

* **E-step (variational):** hold $\theta$ fixed, improve $q$ (i.e., the $\lambda_i$’s) to increase ELBO, which (for fixed $\theta$) shrinks the KL gap to the true posterior.
* **M-step:** hold $q$ fixed, improve $\theta$ to increase ELBO (which also pushes up the likelihood term that appears in the ELBO identity).

A Cross Validated answer explains this E-step/M-step interpretation directly: maximizing w.r.t. $q$ at fixed $\theta$ is the “E-step,” then optimizing w.r.t. $\theta$ with $q$ held constant is the “M-step.”

**FACT.** This approach can be accurate within the chosen variational family, but it requires storing and updating $\lambda_i$ for each datapoint, and doing iterative inference for each new $x$. (That’s the cost amortization aims to avoid.)

### 2. Amortized variational inference (AVI)

Amortization replaces $\lambda_i$ with the output of a **shared function** (often a neural net) that maps $x$ to variational parameters:

$$\lambda_i = f_\phi(x_i), \qquad q_{\lambda_i}(z_i)=q_\phi(z_i\mid x_i).$$

Now the optimization is over **global** parameters (\phi) (and (\theta)):

$$\max_{\phi,\theta}; \sum_{i=1}^N \mathcal L(\phi,\theta;x_i).$$

**FACT.** Wikipedia describes exactly this motivation: if you can learn an approximation $q_\phi(z\mid x)\approx p_\theta(z\mid x)$ “for most $x$,” then you can infer $z$ from $x$ cheaply, and it explicitly calls that idea *amortized inference*.

**FACT.** This is also why the ELBO is attractive for large-scale learning: you can estimate it by sampling $z\sim q_\phi(\cdot\mid x)$ and using Monte Carlo to optimize $(\phi,\theta)$ with stochastic gradients; Wikipedia notes the unbiased-estimator point for the log-ratio integrand.

### 3. What changes conceptually: two “gaps” instead of one

* **FACT.** The ELBO identity still holds for whatever $q_\phi(z\mid x)$ you choose: $\log p_\theta(x)=\mathcal L(\phi,\theta;x)+\mathrm{KL}(q_\phi(z\mid x)\mid p_\theta(z\mid x))$. (This is the core definition-level relationship.)
* **OPINION (standard in the literature, but not spelled out in these pages).** Amortization adds an additional optimization constraint: even if your family $q_\lambda$ could fit the best per-point approximation, your inference network $f_\phi$ might not output those best $\lambda_i$’s. Practitioners often describe this as an “amortization gap” on top of the usual variational approximation gap.

### 4. Practical consequences (clearly labeled)

* **FACT.** **Speed at test time:** Once $\phi$ is learned, you can do “one forward pass” inference of $q_\phi(z\mid x)$ for a new $x$, which is the whole point of amortization.
* **FACT.** **Memory/scaling:** You don’t maintain separate $\lambda_i$ for every datapoint; you only store $\phi$.
* **OPINION.** **Accuracy trade-off:** amortization can be less accurate than per-datapoint optimization if $f_\phi$ is underpowered or optimization is imperfect—but in many modern settings the speed/scale win dominates.


## Appendix (iii): How amortized VI becomes the VAE objective

### Step 0 — Start from the ELBO you already have

For a single datapoint $x$, the ELBO is

$$
\mathcal L(\phi,\theta;x)
=\mathbb E_{z\sim q_\phi(\cdot\mid x)}!\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
=\mathbb E_q[\log p_\theta(x\mid z)]-\mathrm{KL}\big(q_\phi(z\mid x)\mid p_\theta(z)\big),
$$

and also

$$\mathcal L(\phi,\theta;x)=\log p_\theta(x)-\mathrm{KL}!\big(q_\phi(z\mid x)\mid p_\theta(z\mid x)\big),$$

so it is a lower bound on $\log p_\theta(x)$.

### Step 1 — Identify encoder and decoder (the VAE specialization)

A **VAE** is just the above setup with two neural networks:

* **Encoder (inference network):** $q_\phi(z\mid x)$. In practice this outputs parameters of a simple distribution (most commonly a diagonal Gaussian):
  
  $$q_\phi(z\mid x)=\mathcal N!\big(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\big).$$
  
* **Decoder (generative network):** $p_\theta(x\mid z)$, e.g. a Bernoulli for binary pixels or a Gaussian for real-valued data:
  
  $$p_\theta(x\mid z)=\text{Bernoulli}(\pi_\theta(z)) \quad \text{or} \quad \mathcal N(x; f_\theta(z),\sigma^2 I).$$
  
* **Prior:** typically fixed $p(z)=\mathcal N(0,I)$.

**FACT (from sources):** the ELBO is used as a loss for training deep neural networks and includes the “internal component” $q_\phi(\cdot\mid x)$. ([Wikipedia][1])
**NOTE (transparent):** the sources you gave don’t spell out “VAE = encoder/decoder”; this mapping is the standard, direct specialization of the ELBO to neural parameterizations.

### Step 2 — The VAE objective in the familiar “reconstruction − regularizer” form

With $p_\theta(x,z)=p_\theta(x\mid z)p(z)$, the per-datapoint VAE objective is

$$
\boxed{\mathcal L_{\text{VAE}}(\phi,\theta;x)=\underbrace{\mathbb E_{z\sim q_\phi(\cdot\mid x)}[\log p_\theta(x\mid z)]}_{\text{reconstruction / fit}} = \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\mid p(z)\big)}_{\text{regularize toward prior}}}
$$

which is exactly one of the “main forms” of the ELBO.

**Implementation fact (standard practice, not discussed in your links):** to differentiate through the expectation efficiently, VAEs usually sample via the reparameterization $z=\mu_\phi(x)+\sigma_\phi(x)\odot \varepsilon$, $\varepsilon\sim\mathcal N(0,I)$. (This is the usual low-variance gradient estimator that makes neural VI work in practice.)

## Appendix (iv): Mini-batching: how the estimator changes (and what stays unbiased)

### Full-data objective

For a dataset ${x_i}_{i=1}^N$, you maximize the sum (or average) of per-point ELBOs:

$$
\max_{\phi,\theta}\ \sum_{i=1}^N \mathcal L(\phi,\theta;x_i)
\quad\text{or}\quad
\max_{\phi,\theta}\ \frac{1}{N}\sum_{i=1}^N \mathcal L(\phi,\theta;x_i).
$$

### Mini-batch estimator

Pick a mini-batch $B\subset{1,\dots,N}$ of size $\lvert B\rvert=M$, uniformly at random.

* If you use the **sum** objective, a standard unbiased estimator is
  
  $$\widehat{\mathcal J}(\phi,\theta;B) =\frac{N}{M}\sum_{i\in B}\mathcal L(\phi,\theta;x_i).$$
  
* If you use the **average** objective, a standard estimator is just the mini-batch average
  
  $$\widehat{\mathcal J}*{\text{avg}}(\phi,\theta;B) =\frac{1}{M}\sum_{i\in B}\mathcal L(\phi,\theta;x_i),$$
  
  which differs only by a constant scaling of gradients compared to the “sum” version.

**FACT (math):** with uniform random batching, $\mathbb E_B[\widehat{\mathcal J}]=\sum_{i=1}^N \mathcal L_i$, and therefore the mini-batch gradient is an unbiased estimator of the full-data gradient (up to the same scaling choice).

### Where Monte Carlo over $z$ enters

Each $\mathcal L(\phi,\theta;x_i)$ itself is usually estimated with $K$ samples from $q_\phi(z\mid x_i)$:

$$\widehat{\mathcal L}(\phi,\theta;x_i)=\frac{1}{K}\sum_{k=1}^K \Big[\log p_\theta(x_i\mid z_{ik})\Big] -\mathrm{KL}!\big(q_\phi(z\mid x_i)\mid p(z)\big), \quad z_{ik}\sim q_\phi(\cdot\mid x_i).$$

Wikipedia explicitly notes that sampling $z\sim q_\phi(\cdot\mid x)$ yields an unbiased Monte Carlo estimator of the ELBO expectation term.

Put together, a practical training step is “double stochastic”: randomness from the mini-batch $B$ and from latent samples $z$.
