---
title: Evidence Lower Bound (ELBO)
layout: default
noindex: true
---

# Evidence Lower Bound (ELBO)


<!-- # A Technical Chapter on the Evidence Lower Bound (ELBO): From Variational Inference to Variational Autoencoders -->

## 1. The core problem ELBO is trying to solve

### Fact

In latent-variable probabilistic models, we often define a joint distribution over observed data $x$ and latent variables $z$:

$$p_\theta(x,z) = p_\theta(x\mid z)p(z),$$

and we want the **marginal likelihood** (a.k.a. evidence):

$$p_\theta(x) = \int p_\theta(x,z)dz.$$

For many interesting models (e.g., with neural-network likelihoods), this integral is intractable to compute exactly. ([Wikipedia][1])

### Fact

Bayesian posterior inference is similarly hard because

$$p_\theta(z\mid x) = \frac{p_\theta(x,z)}{p_\theta(x)}$$

depends on the same intractable evidence term $p_\theta(x)$. ([ar5iv][2])

### Interpretation (not a new fact)

So we’re blocked in two places at once:

* We can’t evaluate $p_\theta(x)$ (so maximum likelihood learning is hard).
* We can’t evaluate $p_\theta(z\mid x)$ (so posterior inference is hard).

The ELBO is the standard move that “unblocks” both by turning inference into optimization.

## 2. Introducing a variational approximation

### Fact

Variational inference introduces a tractable approximation $q_\phi(z\mid x)$ to the true posterior $p_\theta(z\mid x)$. This $q_\phi$ can be any distribution family you can sample from / evaluate (often parameterized by a neural net). ([Wikipedia][1])

### Fact

Given any choice of $q_\phi(z\mid x)$, we can rewrite the log evidence:

$$
\log p_\theta(x) = \log \int p_\theta(x,z)dz
= \log \int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}dz
= \log \mathbb{E}_{z\sim q_\phi(z\mid x)}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right].
$$

This “multiply and divide by $q$” maneuver is explicit in the derivations you linked. ([Yunfan Jiang][3])

## 3. Deriving the ELBO (Jensen’s inequality)

### Fact

Applying Jensen’s inequality to $\log \mathbb{E}[\cdot]$ yields:

$$
\log p_\theta(x) \ge \mathbb{E}_{z\sim q_\phi(z\mid x)}\left[\log p_\theta(x,z) - \log q_\phi(z\mid x)\right].
$$

That right-hand side is the **Evidence Lower Bound (ELBO)**. ([Yunfan Jiang][3])

### Fact

A common equivalent decomposition is:

$$\text{ELBO}(x;\theta,\phi) = \log p_\theta(x) - D_{\mathrm{KL}}\big(q_\phi(z\mid x)\mid p_\theta(z\mid x)\big),$$

and since KL divergence is nonnegative, ELBO is indeed a lower bound on $\log p_\theta(x)$. ([Wikipedia][1])

### Interpretation

This equality is the real “semantic payload” of ELBO:

* **Maximizing ELBO** (with $\theta,\phi$) is equivalent to:

  * pushing up $\log p_\theta(x)$ (better generative model), and
  * pushing down $D_{\mathrm{KL}}(q_\phi(z\mid x)\mid p_\theta(z\mid x))$ (better inference).

## 4. The ELBO in the “reconstruction + regularization” form

A version widely used in machine learning expands $\log p_\theta(x,z)$ into $\log p_\theta(x\mid z)+\log p(z)$:

### Fact

$$\text{ELBO}(x;\theta,\phi) = \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] - D_{\mathrm{KL}}\big(q_\phi(z\mid x)\mid p(z)\big).$$

This is the form emphasized in the variational autoencoder derivation: an expected “data fit” term minus a KL regularizer that discourages the variational posterior from drifting too far from the prior. ([ar5iv][2])

### Interpretation

* The first term is often called **reconstruction** (especially when $p_\theta(x\mid z)$ is a decoder network).
* The second term is a **complexity / regularization** penalty that limits how much information $z$ can carry about $x$, given the prior.

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

## 12. What I used from each source (narrative attribution)

* From the **Wikipedia** entry, I relied on the clean definitions and the key identity that ELBO equals log evidence minus a KL-to-true-posterior, including the terminology connections (e.g., “variational lower bound,” “negative variational free energy”). ([Wikipedia][1])
* From **Yunfan’s blog**, I used the pedagogical derivation structure: multiply/divide by (q), take expectations, then apply Jensen; plus the emphasis on the inference–optimization duality and the interpretation of the inference network/variational posterior. ([Yunfan Jiang][3])
* From **Kingma & Welling’s VAE paper** (via the ar5iv HTML rendering), I used the method-centric details: the variational bound in the VAE form, minibatch stochastic optimization framing, and especially the reparameterization trick / SGVB estimator that makes gradients practical. ([ar5iv][2])

---

[1]: https://en.wikipedia.org/wiki/Evidence_lower_bound "Evidence lower bound - Wikipedia"
[2]: https://ar5iv.org/pdf/1312.6114 "[1312.6114] Auto-Encoding Variational Bayes"
[3]: https://yunfanj.com/blog/2021/01/11/ELBO.html "ELBO — What & Why | Yunfan’s Blog"
