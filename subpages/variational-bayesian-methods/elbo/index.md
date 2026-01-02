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

$$\mathcal L(\phi,\theta;x) =  \mathbb E_{z\sim q_\phi(\cdot\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right] = \mathbb E_{z\sim q_\phi(\cdot\mid x)}\big[\log p_\theta(x,z)\big] + H[q_\phi(\cdot\mid x)].$$

Given the approximation $q_\phi(z\mid x)$ we would like to know how good we are approximating the true posterior $p(z\mid x)$. To measure these two distribution we use KL divergence as a "metric" to measure how close is $q(z\mid x)$ to $p(z\mid x)$:

$$\mathrm{KL}\big(q_\phi(z\mid x)\| p_\theta(z\mid x)\big) = \int q_\phi(z\mid x) \log \frac{q_\phi(z\mid x)}{p(z\mid x)} dz$$

$$= \int q_\phi(z\mid x) \log \frac{q_\phi(z\mid x)}{p(z\mid x)} dz$$

$$= - \int q_\phi(z\mid x) \log \frac{p(z\mid x)}{q_\phi(z\mid x)} dz$$

$$= - \int q_\phi(z\mid x) \log \frac{p(x,z)}{q_\phi(z\mid x)p(x)} dz$$

$$= - (\int q_\phi(z\mid x) \log \frac{p(x,z)}{q_\phi(z\mid x)} dz - \int q_\phi(z\mid x) \log p(x) dz)$$

$$= - \underbrace{\int q_\phi(z\mid x) \log \frac{p(x,z)}{q_\phi(z\mid x)} dz}_{\text{ELBO}:=\mathcal L(\phi,\theta;x)} - \log p(x) dz$$

So, ELBO can be rewritten as

$$\mathcal L(\phi,\theta;x) = \log p_\theta(x) - \mathrm{KL}\big(q_\phi(z\mid x)\| p_\theta(z\mid x)\big),$$

which immediately implies $\mathcal L(\phi,\theta;x)\le \log p_\theta(x)$ because KL divergence is nonnegative. The KL divergence term can be interpreted as a measure of the additional information required to express the posterior relative to the prior.

A Princeton blog note by Ryan Adams presents the key “accounting identity” in the opposite direction:

$$\log p_\theta(x)= \mathrm{KL}(q(z)\| p(z\mid x,\theta)) + \mathrm{ELBO}(q),$$

so the ELBO is exactly “log evidence minus a gap,” and the gap is a KL divergence.

**INTERPRETATION.** Mentally, you can treat ELBO maximization as doing two things at once:
1. pushing $q$ toward the true posterior (shrinking the KL gap),
2. adjusting $\theta$ to make the model explain the data well. Which of these dominates in practice depends on parameterization and optimization, not on the algebra.

or **Maximizing ELBO** (with $\theta,\phi$) is equivalent to:
  * pushing up $\log p_\theta(x)$ (better generative model), and
  * pushing down $\mathrm{KL}(q_\phi(z\mid x)\|\| p_\theta(z\mid x))$ (better inference).
  
<figure>
  <img src="{{ '/assets/images/notes/variational-bayesian-methods/sketch_of_vi.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Sketch of variational inference. We look for a distribution $q(\theta)$ ($\theta=Z$) that is close to $p(\theta|X)$. Blob is a prior on distributions (space of $q$). Because simple and more tractlable family of distributions (the blob) does not usually contain the real underlying distribution of latent variables, we have some non-zero KL divergence value.</figcaption>
</figure>

## 3. Tightness: when is the bound exact?

ELBO becomes equal to $\log p_\theta(x)$ if and only if

$$q_\phi(z\mid x) = p_\theta(z\mid x)$$

(almost everywhere), because the gap is exactly the KL divergence to the true posterior.

Even if you optimize perfectly within your chosen family for $q_\phi$, the bound can remain loose when:

* the true posterior is too complex (multi-modal, heavy-tailed, etc.),
* but your variational family is restricted (e.g., diagonal Gaussian).

## 4. Three derivations that look different but are the same proof in different clothing

All sources ultimately cash out to the same inequality $\log p(x)\ge \mathrm{ELBO}$. What differs is *which inequality you “pay for”* to get it.

### 4.1 KL-first derivation (no Jensen required)

The Adams note starts from $\mathrm{KL}(q(z)\|\| p(z\mid x,\theta))$, expands it, and rearranges terms to isolate $\log p(x\mid\theta)$ on one side and the ELBO on the other. The only “inequality step” is the nonnegativity of KL.

This route makes the “ELBO gap” explicit: the bound is tight iff $q(z)=p(z\mid x,\theta)$ almost everywhere.

### 4.2 Jensen derivation (log of expectation vs expectation of log)

A common teaching path (used in Yunfan’s blog post) inserts $q(z\mid x)/q(z\mid x)$ into the marginal likelihood integral, rewrites it as a log of an expectation under $q$, then applies Jensen:

$$\log p(x) = \log \mathbb E_{z\sim q}\left[\frac{p(x,z)}{q(z\mid x)}\right] \ge \mathbb E_{z\sim q}\left[\log\frac{p(x,z)}{q(z\mid x)}\right].$$

That right-hand side is the **Evidence Lower Bound (ELBO)**. Yunfan’s write-up explicitly shows the identity $\log p(x)=\mathcal L + \mathrm{KL}(q\|\| p)$ and then turns it into the inequality via KL nonnegativity (another equivalent route).

**INTERPRETATION.** Jensen is pedagogically nice because it highlights the “swap log and integral” issue: $\log\int \cdot \neq \int \log(\cdot)$.

### 4.3 A convex-duality / Hölder-flavored view

A shorter, more geometric blog post (cgad.ski) frames $\log \int e^{H}$ as a convex functional whose tangent lower bounds are expectations under a Gibbs/Boltzmann distribution. After normalizing, it arrives at an inequality of the form

$$v(G)\ge \mathbb E_H[G-H],$$

and identifies this as the ELBO with slack equal to a KL divergence.

This perspective explicitly connects ELBO to “tangent-line” style lower bounds from convexity (and links the slack to KL).

**INTERPRETATION.** If you like Legendre transforms, you can read the ELBO as a variational (dual) representation of a log-partition-like quantity. If you don’t, you can ignore this entirely and still do VI.

## 5. The ELBO as two coupled problems: inference and learning

Write the “main form” (also listed by Wikipedia) as

$$\mathcal L(\phi,\theta;x) = \underbrace{\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]}_{\text{fit / reconstruction}} - \underbrace{\mathrm{KL}(q_\phi(z\mid x)\| p_\theta(z))}_{\text{regularize toward prior}}$$

This makes the coupled roles explicit:

* Optimize $\phi$: make $q_\phi(z\mid x)$ approximate the posterior (reduce the ELBO gap). Also I like the point of view through **complexity / regularization** penalty that limits how much information $z$ can and should carry about $x$, given the prior $p_\theta$.
* Optimize $\theta$: make the generative model explain data well under latent samples from $q_\phi$.

Wikipedia explicitly connects this to amortized inference: learning $q_\phi(z\mid x)$ lets you infer $z$ cheaply for new $x$.

**INTERPRETATION.** The “reconstruction vs KL” decomposition encourages an engineering view: you can often diagnose training pathologies by checking which term dominates or collapses. (This is widely used practice, but it’s not a theorem that *every* pathology shows up cleanly here.)

## 6. Notation reconciliation: $q(z)$ vs $q(z\mid x)$, and what the ELBO “really” is

Across the sources:
* Wikipedia and Yunfan use **amortized** notation $q_\phi(z\mid x)$: one inference model produces a different approximate posterior per datapoint $x$.
* The Adams note uses $q(z)$ for simplicity, essentially treating $x$ as fixed and focusing on a single posterior approximation.
* Some Cross Validated discussion is triggered by tutorials that mix measures and densities, or write KL with objects that don’t type-check.

A detailed Cross Validated answer (postylem) points out a recurring notational bug: you can’t meaningfully write “KL between a measure $\mathbb P$ and a pdf $q$”—KL is defined between distributions (or between densities w.r.t. the same base measure), not “distribution vs. density” as mismatched types.

A safe, standard statement is: if both $q$ and $p$ are absolutely continuous w.r.t. a common base measure and have densities $q(z)$ and $p(z)$, then

$$\mathrm{KL}(q\| p)=\int q(z)\log\frac{q(z)}{p(z)}dz.$$

(That is exactly the kind of correction being emphasized in that answer.)

## 7. Reverse KL geometry and “mode-seeking” behavior

<!-- Yunfan’s blog spends time on intuition for the *direction* of KL used in standard VI, i.e., $\mathrm{KL}(q\|\|p)$ rather than $\mathrm{KL}(p\|\|q)$. The blog describes a “zero-forcing” effect: minimizing reverse KL penalizes placing mass where the target posterior is near zero, encouraging $q$ to sit under prominent modes. The variational posterior $q_{\phi}(z\mid x)$ is prevented from spanning the whole space relative to the true posterior $p(z\mid x)$. Consider the case where the denominator in $\mathrm{KL}(q_{\phi}(z\mid x)\|\|p(z\mid x))=\int q_{\phi}(z\mid x)\log\frac{q(z\mid x)}{p(z\mid x)}dz$ is zero, the value of $q_{\phi}(z\mid x)$ has to be zero as well otherwise the KL divergence goes to infinity. -->

The write-up characterizes reverse KL as tending toward “zero-forcing” behavior (a common shorthand for its asymmetry).

This suggests that $q_\phi(z\mid x)$ cannot assign probability mass outside the support of the true posterior $p(z\mid x)$. If the denominator in $\mathrm{KL}(q_{\phi}(z\mid x)\|\|p(z\mid x))=\int q_{\phi}(z\mid x)\log\frac{q(z\mid x)}{p(z\mid x)}dz$ is zero, then $q_\phi(z\mid x)$ must also be zero; otherwise the KL divergence diverges. The figure illustrates this: the left panel’s green region corresponds to $\frac{q_\phi(z\mid x)}{p(z\mid x)}=0$, and the right panel’s red region corresponds to $\frac{q_\phi(z\mid x)}{p(z\mid x)}=\infty$. In short, reverse KL is *zero-forcing*, driving $q_\phi(z\mid x)$ to lie within (and be “squeezed” under) $p(z\mid x)$.

<figure>
  <img src="{{ '/assets/images/notes/variational-bayesian-methods/direction_kl_divergence.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Left: $\frac{q_\phi(z\mid x)}{p(z\mid x)}=0$. Right: $\frac{q_\phi(z\mid x)}{p(z\mid x)}=\infty$</figcaption>
</figure>

**INTERPRETATION / caution.** “Reverse KL is mode-seeking” is a useful heuristic, but the exact behavior depends on the variational family. With limited families (e.g., mean-field Gaussians), the asymmetry is very visible; with richer families, the distinction can blur.

In this context, **“zero-forcing”** means:

> **When you minimize the reverse KL** $\mathrm{KL}(q\|\| p)$, the optimization **strongly penalizes** any place where $q(z)$ puts probability mass but $p(z)=0$.
> So the safest way to reduce the objective is for $q$ to **set its density to (near) zero** in those regions.

Why? Because reverse KL is

$$\mathrm{KL}(q\|p)=\int q(z)\log\frac{q(z)}{p(z)}dz.$$

If there’s a region where $p(z)=0$ but $q(z)>0$, then $\log\frac{q}{p}=\log(\infty)=\infty$, and the KL becomes **infinite**. So the optimizer “forces” $q(z)$ to be **zero** wherever $p(z)$ is zero (or extremely small).

**Intuition:** reverse KL makes $q$ “play it safe” by staying inside the support of $p$. This often leads to **mode-seeking** behavior: $q$ may focus on one high-probability region (one mode) rather than spreading out to cover all modes.

(Contrast: forward KL $\mathrm{KL}(p\|\|q)$ is more “zero-avoiding,” because it heavily penalizes $q(z)=0$ where $p(z)>0$, pushing $q$ to cover all regions that $p$ considers plausible.)


## 8. A unified mental model: ELBO is a *lower envelope* of the evidence

All seven sources can be reconciled by one geometric sentence:

> The log evidence $\log p_\theta(x)$ is hard because it is a log-integral; the ELBO is what you get when you replace that log-integral by a variational lower bound built from a chosen $q$, and the “distance” to the truth is a KL divergence.

* Wikipedia emphasizes definition, equivalent forms, unbiased estimation of the ELBO integrand under $q$, and amortized inference.
* Yunfan’s blog emphasizes the Jensen route and intuition for reverse KL asymmetry (zero-forcing).
* Ryan Adams emphasizes that you can derive everything by algebra and a basic log inequality, without invoking Jensen explicitly.
* cgad.ski emphasizes convexity/variational derivatives: ELBO as a tangent lower bound whose slack is KL.
* Cross Validated emphasizes “type correctness” (measure vs pdf), debunking “ignore the KL,” and practical reasons $\log p(x)$ is nastier than ELBO for estimation and optimization.

## 9. Why $\log p(x)$ is hard but ELBO is “tractable-ish” (and what that really means)

A Cross Validated question asks: if $\log p(x)$ is hard, why isn’t ELBO equally hard?

A good way to answer is to separate *computing a number* from *optimizing an objective*.

### 9.1 Estimating $\log p(x)$ directly: importance sampling and the log bias

One Cross Validated answer (Ben) rewrites the evidence using importance sampling:

$$\log p(x)=\log\left(\mathbb E_{Z\sim g}\left[\frac{p(x\mid Z)p(Z)}{g(Z)}\right]\right),$$

then approximates the expectation with Monte Carlo samples.

**FACT.** Even if the inside expectation can be approximated, the *log of a Monte Carlo estimate* is generally biased (because log is nonlinear). A separate Cross Validated thread on “VI vs maximum likelihood” makes the same point plainly: sampling inside the log doesn’t give an unbiased estimate of $\log p(x)$.

### 9.2 Estimating the ELBO: expectation of a log-ratio

The ELBO is

$$\mathcal L = \mathbb E_{q}\left[\log\frac{p(x,z)}{q(z\mid x)}\right].$$

**FACT.** Wikipedia explicitly notes that if you can sample $z\sim q$, then $\log\frac{p(x,z)}{q(z\mid x)}$ is an unbiased Monte Carlo estimator of the ELBO (as an expectation).

So a key computational difference is:
* $\log p(x)$ is **log of an expectation** (hard to estimate without bias, hard to differentiate cleanly).
* ELBO is **expectation of a log** (straightforward to estimate with Monte Carlo; optimizing it still has variance issues, but it’s structurally friendlier).

**INTERPRETATION.** People often say “the normalizing constant disappears in ELBO gradients.” Sometimes that’s true in the sense that you never need to evaluate $p(x)=\int p(x,z)dz$ explicitly; you only need $p(x,z)$ and $q(z\mid x)$. But “easy” is relative: if $q$ is poorly matched, Monte Carlo variance can still be brutal.

## 10. Common misconceptions (and how the sources correct them)

### Misconception A: “ELBO is always non-positive”

A Cross Validated question asked exactly this, and the answer shows why it’s false: ELBO contains a negative KL term (non-positive) *plus* a log-density/expected log-likelihood term that can be positive because densities can exceed 1.

**FACT.** ELBO can be positive; its sign is not fixed.

### Misconception B: “We can ignore the KL term to the posterior”

The same Cross Validated answer is blunt: you cannot “ignore” $\mathrm{KL}(q\|\|p(z\mid x,\theta))$; minimizing that KL is the conceptual starting point, and the ELBO is how we optimize it without needing the (intractable) posterior normalization.

**FACT.** The ELBO–evidence relationship is exactly $\log p(x)=\mathrm{ELBO}+\mathrm{KL}(q\|\|p)$; dropping the KL-to-posterior conceptually breaks the link to posterior approximation.

### Misconception C: “Why not just do maximum likelihood on $\log p(x)$?”

Another Cross Validated thread (“Why variational inference and not maximum likelihood?”) starts by correcting a classic algebra mistake: you cannot push a log inside an integral/expectation.
It then points out that naive Monte Carlo inside the log is biased, motivating the use of a lower bound via Jensen—i.e., variational inference.

## 11. VI vs. maximum likelihood: what changes when you remove terms?

Here’s the clean conceptual split:

* **Maximum likelihood (ML)**: maximize $\log p_\theta(x)$ (still hard with latent variables).
* **Variational inference (VI)**: maximize a surrogate $\mathcal L(\phi,\theta;x)\le \log p_\theta(x)$ that is tractable to optimize, while also producing an approximate posterior $q_\phi(z\mid x)$.

A Cross Validated answer in the “understanding ELBO” thread makes a pointed statement: if you drop the KL-to-prior regularizer term in the “expected log-likelihood minus KL-to-prior” form, you stop doing Bayesian inference and move toward ML-style objectives.

**FACT (in the context of that decomposition).** In the $\mathbb E_q[\log p(x\mid z)]-\mathrm{KL}(q\|p(z))$ form, removing the KL-to-prior removes the explicit Bayesian regularization toward the prior.

**INTERPRETATION.** Whether that is “good” depends on your goal:

* If you only want point estimates of $\theta$ and don’t care about posterior uncertainty over $z$, ML-ish methods (including EM variants) may be appropriate.
* If you want fast approximate posteriors for downstream tasks (uncertainty-aware decisions, latent representations, amortized inference), VI gives you a *product* (the learned $q_\phi$) that ML alone doesn’t.

## 12. Practical takeaways (explicitly opinionated)

**OPINION 1.** When teaching ELBO, start from the identity

$$\log p(x)=\mathrm{ELBO}+\mathrm{KL}(q\|p).$$

It prevents a lot of confusion because it tells you what the bound *means* (evidence minus a gap), not just how to derive it.

**OPINION 2.** Use Jensen derivations early only to highlight *why* “log of an expectation” is hard; then pivot to the KL decomposition to clarify what VI is minimizing.

**OPINION 3.** Treat “ELBO is easy” as marketing. ELBO is **easier to Monte Carlo estimate** than $\log p(x)$, and it is **easier to differentiate** in many setups, but it can still be a high-variance objective, especially with weak $q$ families or bad initialization.

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

$$\boxed{\mathcal L(\phi,\theta;x) = \mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] -\mathrm{KL}\big(q_\phi(z\mid x),|,p_\theta(z)\big).}$$

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

Now the optimization is over **global** parameters $\phi$ (and $\theta$):

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
=\mathbb E_{z\sim q_\phi(\cdot\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
=\mathbb E_q[\log p_\theta(x\mid z)]-\mathrm{KL}\big(q_\phi(z\mid x)\mid p_\theta(z)\big),
$$

and also

$$\mathcal L(\phi,\theta;x)=\log p_\theta(x)-\mathrm{KL}\big(q_\phi(z\mid x)\mid p_\theta(z\mid x)\big),$$

so it is a lower bound on $\log p_\theta(x)$.

### Step 1 — Identify encoder and decoder (the VAE specialization)

A **VAE** is just the above setup with two neural networks:

* **Encoder (inference network):** $q_\phi(z\mid x)$. In practice this outputs parameters of a simple distribution (most commonly a diagonal Gaussian):
  
  $$q_\phi(z\mid x)=\mathcal N\big(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\big).$$
  
* **Decoder (generative network):** $p_\theta(x\mid z)$, e.g. a Bernoulli for binary pixels or a Gaussian for real-valued data:
  
  $$p_\theta(x\mid z)=\text{Bernoulli}(\pi_\theta(z)) \quad \text{or} \quad \mathcal N(x; f_\theta(z),\sigma^2 I).$$
  
* **Prior:** typically fixed $p(z)=\mathcal N(0,I)$.

**FACT (from sources):** the ELBO is used as a loss for training deep neural networks and includes the “internal component” $q_\phi(\cdot\mid x)$.

**NOTE (transparent):** the sources you gave don’t spell out “VAE = encoder/decoder”; this mapping is the standard, direct specialization of the ELBO to neural parameterizations.

### Step 2 — The VAE objective in the familiar “reconstruction − regularizer” form

With $p_\theta(x,z)=p_\theta(x\mid z)p(z)$, the per-datapoint VAE objective is

$$
\boxed{\mathcal L_{\text{VAE}}(\phi,\theta;x)=\underbrace{\mathbb E_{z\sim q_\phi(\cdot\mid x)}[\log p_\theta(x\mid z)]}_{\text{reconstruction / fit}} = \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\mid p(z)\big)}_{\text{regularize toward prior}}}
$$

which is exactly one of the “main forms” of the ELBO.

**Implementation fact (standard practice, not discussed in your links):** to differentiate through the expectation efficiently, VAEs usually sample via the reparameterization $z=\mu_\phi(x)+\sigma_\phi(x)\odot \varepsilon$, $\varepsilon\sim\mathcal N(0,I)$. (This is the usual low-variance gradient estimator that makes neural VI work in practice.)

<figure>
  <img src="{{ '/assets/images/notes/variational-bayesian-methods/vae_graphical_model.png' | relative_url }}" alt="a" loading="lazy">
  <!-- <figcaption></figcaption> -->
</figure>

## Appendix (iv): Mini-batching: how the estimator changes (and what stays unbiased)

### Full-data objective

For a dataset $\lbrace x_i\rbrace_{i=1}^N$, you maximize the sum (or average) of per-point ELBOs:

$$
\max_{\phi,\theta}\ \sum_{i=1}^N \mathcal L(\phi,\theta;x_i)
\quad\text{or}\quad
\max_{\phi,\theta}\ \frac{1}{N}\sum_{i=1}^N \mathcal L(\phi,\theta;x_i).
$$

### Mini-batch estimator

Pick a mini-batch $B\subset\lbrace 1,\dots,N\rbrace$ of size $\lvert B\rvert=M$, uniformly at random.

* If you use the **sum** objective, a standard unbiased estimator is
  
  $$\widehat{\mathcal J}(\phi,\theta;B) =\frac{N}{M}\sum_{i\in B}\mathcal L(\phi,\theta;x_i).$$
  
* If you use the **average** objective, a standard estimator is just the mini-batch average
  
  $$\widehat{\mathcal J}_{\text{avg}}(\phi,\theta;B) =\frac{1}{M}\sum_{i\in B}\mathcal L(\phi,\theta;x_i),$$
  
  which differs only by a constant scaling of gradients compared to the “sum” version.

**FACT (math):** with uniform random batching, $\mathbb E_B[\widehat{\mathcal J}]=\sum_{i=1}^N \mathcal L_i$, and therefore the mini-batch gradient is an unbiased estimator of the full-data gradient (up to the same scaling choice).

### Where Monte Carlo over $z$ enters

Each $\mathcal L(\phi,\theta;x_i)$ itself is usually estimated with $K$ samples from $q_\phi(z\mid x_i)$:

$$\widehat{\mathcal L}(\phi,\theta;x_i)=\frac{1}{K}\sum_{k=1}^K \Big[\log p_\theta(x_i\mid z_{ik})\Big] -\mathrm{KL}\big(q_\phi(z\mid x_i)\mid p(z)\big), \quad z_{ik}\sim q_\phi(\cdot\mid x_i).$$

Wikipedia explicitly notes that sampling $z\sim q_\phi(\cdot\mid x)$ yields an unbiased Monte Carlo estimator of the ELBO expectation term.

Put together, a practical training step is “double stochastic”: randomness from the mini-batch $B$ and from latent samples $z$.
