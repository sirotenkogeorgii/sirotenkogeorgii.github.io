---
title: Training Products of Experts by Minimizing Contrastive Divergence
layout: default
noindex: true
---

# Training Products of Experts by Minimizing Contrastive Divergence

[link](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)



## 0) What this paper is doing (core message)

The paper introduces **Products of Experts (PoE)** as a way to combine many simple probabilistic models (“experts”) for high-dimensional data by **multiplying** their distributions and **renormalizing**. It then proposes an efficient training method that avoids expensive/unstable maximum-likelihood learning by instead **minimizing “contrastive divergence” (CD)**—a KL-based objective that can be approximated using only **one (or few) Gibbs-sampling steps from data**. This yields the famous **CD-1 learning rule** and explains how it applies especially cleanly to **Restricted Boltzmann Machines (RBMs)** (viewed as PoEs).

## 1) Why Products of Experts instead of Mixtures?

### 1.1 Mixtures are inefficient in high dimensions

Mixtures (e.g., mixture of Gaussians) combine models by weighted averaging: choose a component, generate data from it. Mixtures can approximate complex smooth densities, but in high-dimensional spaces they’re inefficient because:
* The posterior cannot be **sharper** than the components.
* Covering a complex low-dimensional manifold in high-$D$ would require many broadly tuned components, preventing sharp inference (face-manifold example: ~35 latent factors but sharp perception). 

### 1.2 PoE: “intersection of constraints” → sharp distributions

A PoE combines experts by **multiplication**:
* Each expert can focus on one “nugget” of constraint (local stroke structure, tense agreement, etc.).
* Data that satisfies one constraint but violates others gets ruled out because other experts assign low probability.
* The product can be **much sharper** than any single expert because it effectively represents the **intersection** of constraints. 
* Metaphorically speaking, a single expert in a **mixture** has the power to pass a bill while a single expert in a **product** has the power to veto it.
* For an event to be likely under a product model, all constraints must be (approximately) satisfied, while an event is likely under a mixture model if it (approximately) matches with any single template.

## 2) The PoE model (formal definition)

Assume discrete data vectors $d$ (continuous is analogous with integrals).

### 2.1 Product + renormalization

For experts $m=1,\dots,n$ with parameters $\theta_m$ and distributions $p_m(d\mid \theta_m)$, the PoE distribution is:

$$p(d \mid \theta_1,\ldots,\theta_n) = \frac{\prod_m p_m(d\mid \theta_m)}{\sum_c \prod_m p_m(c\mid \theta_m)} \quad \text{Eq. 1}$$

* Denominator is the **partition function** / normalizer $Z=\sum_c \prod_m p_m(c\mid \theta_m)$.
* Note: the paper remarks $p_m$ technically only needs to be positive; usually it is a probability distribution. 

### 2.2 Intuition about “wasted probability”

For a single model, good fit means high probability on data and low probability elsewhere.
For a PoE, each expert can “waste” probability mass in different wrong places; the product cancels those regions if the other experts penalize them. So **experts should be different**—their disagreement makes $Z$ small, so renormalization boosts probability where they agree (the data). 

## 3) Maximum likelihood learning and why it’s painful

### 3.1 Likelihood gradient for an expert

Given the iid data, one can use the log-likelihood as an objective function to train a PoE:

$$\text{Log-Likelihood}(d_1,\dots,d_N; \theta_1,\dots,\theta_n) = \sum^N_{i=1}\sum^n_{m=1}\log p_m(d_i\mid \theta_m) - N\log Z$$

For one data vector $d$ $(=d_i)$, the log-likelihood gradient wrt expert $m$’s parameters is:

$$
\frac{\partial \log p(d\mid \theta_{1:n})}{\partial \theta_m}=\frac{\partial \log p_m(d\mid \theta_m)}{\partial \theta_m} - \underbrace{\sum_c p(c\mid \theta_{1:n}) \frac{\partial \log p_m(c\mid \theta_m)}{\partial \theta_m}}_{\mathbb{E}_{c\sim p(\cdot \mid \theta_{1:n})}[\frac{\partial \log p_m(c\mid \theta_m)}{\partial \theta_m}]} \quad\text{Eq. 2}
$$

Interpretation:
* **Positive phase:** push expert to increase probability on real data $d$.
* **Negative phase:** decrease probability on “fantasy data” $c$ sampled from the full PoE model distribution. 

### 3.2 Sampling fantasy data is the bottleneck

The simplicity of the gradient in $\text{Eq. 2}$ is deceptive, it requires the evaluation of an intractable expectation over $p$ (second term). To estimate the negative phase you need samples from $p(c\mid \theta_{1:n})$.

* Rejection sampling (“each expert samples independently until all agree”) is conceptually helpful but generally hopelessly inefficient.
* Use **MCMC / Gibbs sampling** instead:
  * Given data, hidden states of different experts are **conditionally independent**, so you can update all experts’ hidden variables in parallel.
  * If, given its hidden state, each expert makes visible dimensions conditionally independent (bipartite structure), you can also update all visible variables in parallel given all hidden states—leading to alternating parallel updates (hidden $\iff$ visible). 

### 3.3 Even if you can sample: variance kills you

Even with MCMC convergence, equilibrium samples have:

* **Very high variance** (they span the model distribution).
* Worse: sample variance depends on parameters, creating “repulsion” effects even if true gradient is zero (sand-on-vibrating-plate analogy). 

## 4) Key contribution: learning by minimizing Contrastive Divergence (CD)

### 4.1 Setup: distributions involved

* $Q_0$: data distribution over visibles (think: start of a Markov chain at time $0$).
* $Q_\infty$: equilibrium distribution over visibles implied by the PoE after long Gibbs sampling (this is the model distribution).
* $Q_1$: distribution of visibles after **one full Gibbs step** starting from data (sample a reconstruction). 

### 4.2 ML is minimizing $\mathrm{KL}(Q_0 \parallel Q_\infty)$

Maximizing the log likelihood of the data (averaged over the data distribution) is equivalent to minimizing of the KL divergence between the data distribution $Q_0$ and the equilibrium distribution over the visible variables $Q_\infty$ (produced by prolongated Gibbs sampling):

$$
\mathrm{KL}(Q_0 \| Q_\infty) = H(Q_0) - \mathbb{E}_{d\sim Q_0}[\log Q_\infty(d)] \quad\text{Eq. 3 idea}
$$

Since the entropy of the data distribution $H(Q_0)$ doesn’t depend on parameters, maximizing likelihood is maximizing $\mathbb{E}_{Q_0}[\log Q_\infty(d)]$. Note that $Q_\infty(d)=p(d\mid \theta_{1:n})$. Expectation of LHS over the data distribution $Q_0$ in $\text{Eq. 2}$, can be rewritten as

$$\mathbb{E}_{Q_0}[\frac{\partial \log p(d\mid \theta_m)}{\partial \theta_m}] = \mathbb{E}_{Q_0}[\frac{\partial \log Q_\infty}{\partial \theta_m}] = \mathbb{E}_{Q_0}[\frac{\partial \log p_m(d\mid \theta_m)}{\partial \theta_m}] - \mathbb{E}_{Q_\infty}[\frac{\partial \log p_m(c\mid \theta_m)}{\partial \theta_m}]\quad\text{Eq. 4}$$

### 4.3 CD objective: compare data to *nearby* reconstructions

There is a simple and effective alternative approach to maximize likelihood which eliminates almost all of the computation required to get samples from the equilibrium distribution and also eliminates much of the variance that masks the gradient signal. The approach is to optimize a different objective. Instead of minimizing $Q_0\parallel Q_\infty$, we minimize the difference between $Q_0\parallel Q_\infty$ and $Q_1\parallel Q_\infty$. Instead of running to equilibrium, minimize the tendency of a Markov chaim produced by Gibbs sampler to move away from the data distribution immediately.

Define **contrastive divergence (CD-1)** as:

$$\mathrm{CD}_1 = \mathrm{KL}(Q_0 \| Q_\infty) - \mathrm{KL}(Q_1 \| Q_\infty)$$

Properties emphasized:

* $Q_1$ is “one step closer to equilibrium than $Q_0$”, so $\mathrm{CD}_1 \ge 0$.
* Under mild conditions (for Markov chains in which all transitions have non-zero probability), $\mathrm{CD}_1=0$ only if the model is perfect (data distribution equals equilibrium) or in other words $Q_0=Q_1$ implies $Q_0=Q_\infty$.

### 4.4 Why CD is tractable: cancellation in the gradient

The mathematical motivation for the contrastive learning is that the intractable expectation over $Q_\infty$ on the RHS of $\text{Eq. 4}$. The derivative of the CD objective wrt expert parameters yields:

$$-\frac{\partial}{\partial \theta_m}(Q_0\parallel Q_\infty - Q_1\parallel Q_\infty) = \mathbb{E}_{Q_0}\Big[\frac{\partial \log p_m(d\mid \theta_m)}{\partial \theta_m}\Big] - \mathbb{E}_{Q_1}\Big[\frac{\partial \log p_m(\hat d\mid \theta_m)}{\partial \theta_m}\Big] + \frac{\partial Q_1}{\partial \theta_m}\frac{\partial Q_1\parallel Q_\infty}{\partial Q_1}\quad\text{Eq. 5}$$

* A **data term** (expectation under $Q_0$)
* A **reconstruction term** (expectation under $Q_1$)
* Plus an extra term involving how $Q_1$ changes with parameters (the “problem term”)

Ifeach expert is chosen to be tractable, it is possible to compute the exact values of the derivative of $\log p_m(d\mid \theta_m)$ and $\log p_m(\hat d\mid \theta_m)$. It is also straightforward to sample from $Q_0$ and $Q_1$, so the first two terms of RHS are tractable.


### 4.5 The practical learning rule (approximate gradient)

The paper argues (empirically, based on experiments) that this problematic third term is typically small and rarely opposes the other terms, so you can ignore it. Update of each expert $m$ becomes:

$$
\Delta \theta_m \propto \mathbb{E}_{d\sim Q_0}\Big[\frac{\partial \log p_m(d\mid \theta_m)}{\partial \theta_m}\Big]-\mathbb{E}_{\hat d\sim Q_1}\Big[\frac{\partial \log p_m(\hat d\mid \theta_m)}{\partial \theta_m}\Big]
\quad\text{Eq. 6}
$$

This is the “**positive phase minus negative phase**”, but the negative phase uses **one-step reconstructions** rather than equilibrium samples—dramatically reducing computation and variance. This works very well in practice even if a single reconstruction of each data vector is used in place of the full probability distribution over reconstructions. The difference in the derivatives of the data vectors and their reconstructions has some variance, because the reconstruction procedure is stochastic. But when the PoE is modelling data moderately well, the one step reconstruction will be very similar to the data, so the variance will be very small. The low variance makes it feasible to perform online learning after each data vector is present, though the simulations described in the paper use batch learning. 

## 5) How to sample a one-step reconstruction $\hat d \sim Q_1$ (general PoE)

The paper gives an explicit procedure (critical for implementation): 
1. Sample a data vector $d \sim Q_0$.
2. For **each expert separately**, compute the posterior distribution over its latent variables given $d$.
3. Sample each latent variable from its posterior (per expert).
4. Given all sampled latents, compute the conditional distribution over visibles by **multiplying** the experts’ conditional distributions (and renormalizing appropriately).
5. Sample each visible variable from that conditional → this yields reconstructed vector $\hat d$.

Then apply Eq. 6 using $d$ for the positive phase and $\hat d$ for the negative phase. 

**Variance argument:** because reconstructions are close to data once learning is reasonable, the stochasticity introduces relatively small variance—like matched pairs in clinical trials. 

## 6) Geometric intuition: fitting a low-$D$ manifold

In high-dimensional datasets, data typically lies near a low-dimensional manifold. The PoE needs a “ridge” of high log-probability along the manifold.

* Ensuring each data point has higher log-probability than its typical reconstructions enforces correct **local curvature** around data.
* Concern: could assign high probability to far-away regions, but unlikely if the surface is smooth and data constraints include local curvature. One can additionally detect/remove such spurious modes by longer Gibbs sampling, but unlike Boltzmann learning it’s not essential. 

## 7) Demonstrations / experiments in the paper

### 7.1 Simple 2D example: factorized structure

* Model: 15 “unigauss” experts; each expert is a mixture of:
  * a uniform distribution
  * one axis-aligned Gaussian
* Data: clustered points arranged on (mostly) a grid.
* Result: each tight cluster is represented by intersection of two elongated Gaussians (different axes). 

**Figure 1 (page 6):** shows datapoints and ellipses (1-std contours) for each expert’s Gaussian; unused experts remain vague but still keep high mixing proportions. 
**Figure 2 (page 6):** 300 points generated by prolonged Gibbs sampling from the fitted PoE; notable that the model generates points at a missing grid location (a plausible but absent mode). 

Learning procedure per update (for each data vector):
1. compute posterior that each expert chooses Gaussian vs uniform (positive term),
2. sample that choice per expert, sample reconstruction from product of chosen Gaussians,
3. compute negative term using reconstruction. 

### 7.2 “Population code” example: 100D images of single edges

* Data: 10×10 synthetic images (100 dims), each contains one intensity edge with varying position/orientation/polarity; intensity profile across edge is sigmoid.
* PoE: 40 unigauss experts (Gaussian+uniform), each learns per-pixel variance; experts do not specialize on small subsets of pixels.
* Given an image, about half the experts choose their Gaussians with high probability; products yield excellent reconstructions.
* Learned means resemble oriented edge detectors and other symmetric patterns that help localize edge endpoints. 

**Figure 3 (page 7):** grid of 40 learned Gaussian means that visually resemble edge-like templates in different positions/orientations/polarities. 

### 7.3 Initialization finding (important practical note)

Two strategies:

* Train experts separately (using different data subsets/weights/dimensions/classes), then combine (with fractional powers to avoid overconfidence).
* Or: initialize all experts randomly as **very vague**, and train jointly with Eq. 6.

Empirical conclusion: **separate specialization early makes poor local optima more likely**. Better to start vague + random and let cooperative training find good solutions. 

## 8) Relationship to Boltzmann Machines: RBM = PoE

### 8.1 Key equivalence

A **Restricted Boltzmann Machine (RBM)** (one visible layer, one hidden layer, no intra-layer connections) can be seen as a PoE with **one expert per hidden unit**:

* Hidden unit OFF $\implies$ factorial distribution where each visible bit is equally likely on/off.
* Hidden unit ON $\implies$ different factorial distribution; weights specify log-odds of visible bits being on.
* Combining experts corresponds to **adding log-odds** across hidden units. 

Exact inference is tractable in RBMs because hidden units are conditionally independent given visibles. 

### 8.2 Standard BM learning = PoE maximum likelihood gradient

For weight $w_{ij}$ between visible $i$ and hidden $j$, the standard gradient becomes “data correlation minus model correlation”:

$$
\frac{\partial}{\partial w_{ij}} \mathrm{KL}(Q_0\|Q_\infty)
= \langle s_i s_j\rangle_{Q_0} - \langle s_i s_j\rangle_{Q_\infty}
\quad\text{Eq. 9 idea}
$$

But estimating $\langle \cdot \rangle_{Q_\infty}$ is slow/high variance. 

### 8.3 CD-1 for RBMs (classic rule)

Approximate CD gradient:

$$
\frac{\partial}{\partial w_{ij}} \mathrm{CD}_1 \approx
\langle s_i s_j\rangle_{Q_0} - \langle s_i s_j\rangle_{Q_1}
\quad\text{Eq. 10}
$$

where $Q_1$ uses **one-step reconstructions**. 

## 9) Learning digit features with RBMs (unsupervised)

### 9.1 Setup

* RBM with **500 hidden**, **256 visible** (16×16 pixels), trained on **8000** USPS digit images (all 10 classes), normalized pixel intensities to [0,1].
* Uses probability-valued visibles (treat intensities as probabilities); reconstructions use probabilities rather than sampling binary pixels. The update becomes:

$$
\approx \langle p_i p_j\rangle_{Q_0} - \langle p_i p_j\rangle_{Q_1} \quad\text{Eq. 11}
$$

* Training: 658 epochs in Matlab on ~500MHz workstation; mini-batches of 100 with balanced digits (10 exemplars per class per batch); learning rate $\sim ¼$ of divergence threshold; momentum added after 10 epochs (add 0.9× previous update). 

### 9.2 Result

* Learned **localized** features; for each image $\sim \frac{1}{3}$ of features active.
* Features include on-center/off-surround and vice versa, stroke fragments, Gabor/wavelet-like patterns.
* **Figure 4 (page 9):** receptive fields (weights) of 100 randomly selected hidden units show these local structures. 

## 10) Using PoEs for discrimination (classification)

### 10.1 Partition function problem

PoE makes it easy to compute the numerator $\prod_m p_m(d)$, but **hard to compute $\log Z$** (normalizer). So:

* You can compare two inputs under the same model ($\log Z$ cancels),
* But absolute likelihood evaluation is difficult. 

### 10.2 Two-class discrimination trick

Train separate PoEs per class (e.g., $PoE_2$, $PoE_3$).
For a test image $t$, each computes $\log p(t\mid \theta_k) + \log Z_k$.
If you can estimate $\log Z_2 - \log Z_3$ (a single scalar), you can choose the more likely class. This scalar can be estimated discriminatively on a labeled validation set. 

**Figures:**

* **Figure 5 (page 10):** 100 hidden-unit weights learned on digit “2” only; mostly local features; interpretation: features act like local deformations of a template, not just edge detectors. 
* **Figure 6 (page 11):** reconstructions of unseen “2” images using features trained on 2’s vs features trained on 3’s—2-trained features reconstruct 2s much better. 

### 10.3 Strong separation in practice (4 vs 6; 7 vs 9)

* For digits 4 vs 6: unnormalized log-probability scores separate perfectly even on held-out test images (drawn from unused portion of training set due to USPS test-set distribution mismatch). 

  * Achieved using **two hidden layers**, and averaging scores from two architectures per class:
    * model A: 200 first-layer, 100 second-layer
    * model B: 100 first-layer, 50 second-layer
  * First layer trained ignoring second; then second trained on activation probabilities of first layer. 
  * **Figure 7 (page 11):** scatter plots of scores show clear separation on training and test. 

* For digits 7 vs 9: harder; not linearly separable; errors occur near the boundary (no very confident misclassifications).

  * **Figure 8 (page 12):** shows this pattern. 

### 10.4 Multi-class (10 digits): logistic regression on PoE scores

To combine 10 PoEs:

* Train a multinomial logistic regression that takes PoE unnormalized log-prob scores as inputs.
* Each digit-class PoE provides **two scores**:
  1. score from first hidden layer model on pixels,
  2. score from second hidden layer model on first-layer activation probabilities.
* The learned logistic weights show second-layer scores add useful discriminative information (capturing correlations among first-layer features). 

**Reported performance:**

* Error rate **1.1%** on 2750 test images; compares favorably with 5.1% nearest neighbor and comparable to best elastic-model classifiers.
* With **7% rejects**, no errors on the 2750 test images.
* Caveat: multiple architectures tried and best chosen using test performance → biased estimate; later work (Mayraz & Hinton) reports careful model selection on MNIST. 

**Figure 9 (page 13):** visualization of logistic regression weights; bottom rows (2nd-layer scores) still substantial, indicating added value. 

## 11) How good is the CD approximation? (the ignored term)

The learning rule ignores the term arising from how the reconstruction distribution $Q_1$ changes with parameters (the third term in the derivation). Section 10 empirically checks safety. 

### 11.1 Exact small-network simulations

They use small RBMs where exact expectations can be computed (cost exponential in visible/hidden units), then compare:

* true change in contrastive divergence vs
* predicted change from the approximate rule (Eq. 10).

Main empirical findings:

* For an *individual* weight, approximate and true gradient can occasionally differ in sign.
* But for networks with more than 2 units per layer, a **parallel update of all weights** based on Eq. 10 is almost certain to improve CD: the update vector has positive cosine with the true gradient. 

**Figure 10 (page 14):**

* (a) histogram of improvements in CD across $10^5$ random networks (8 visible, 4 hidden, weights $\sim N(0,20)$).
* (b) histogram of log-likelihood changes for 1000 such networks; likelihood sometimes decreases (2 cases shown), even though CD tends to improve. 

**Figure 11 (page 15):** scatter plot of modeled vs unmodeled effects; unmodeled effects are almost always helpful, rarely harmful (points mostly below diagonal). 

### 11.2 Important correction: contrastive log-likelihood is a trap

Earlier interpretation: optimizing a “contrastive log-likelihood” difference can be gamed by making all vectors equiprobable (max value 0).
CD avoids this by including entropy terms—high entropy of reconstructions rules out trivial uniform solutions. 

## 12) Other expert types and extensions

### 12.1 Beyond binary stochastic pixels

Binary pixels can’t express $>1$ bit mutual information, so they fail for real images with strong continuous correlations. Alternatives: 

* **Multinomial pixels** with $n$ discrete values (awkward for images).
* **Replicated visible units:** represent one real-valued pixel by a set of identical binary units; number active approximates intensity. Reconstruction gives binomial distribution; only one probability needs computing due to shared weights.
* Similarly, replicated hidden units can approximate real-valued firing rates (binomial counts).

### 12.2 “Unifac” experts

Each expert is mixture of:

* uniform distribution, and
* a factor analyzer with **one factor**.
  Latents: a binary “use uniform vs factor analyzer” variable + a real-valued factor value.
  Parameters: factor loadings vector, mean vector, variance vector.
  (Notes: a “unigauss” expert is a unifac with zero factors.) 

### 12.3 Products of HMMs (sequences)

Motivation: single HMM has limited capacity—mutual information of $n$ bits between past and future requires $2^n$ states. Product of multiple small HMMs can represent mutual information **linearly** in number of HMMs (exponential efficiency), but naive training via forward-backward on cross-product state space loses the win. 

Proposed CD-style training for product HMMs (Brown & Hinton, in prep):

1. For each expert HMM, forward-backward gives posterior over hidden paths.
2. Sample one hidden path per expert from posterior.
3. At each time step, sample output symbol from product of output distributions from the selected expert states.
   Then update each expert’s parameters using gradient difference between observed and reconstructed sequences. 

**Figure 12 (page 17):** example HMM expert capturing a non-local regularity like patterns “… shut … up …” by giving higher probability to strings containing “shut” followed later by “up”. 

## 13) Discussion and positioning

### 13.1 Relation to earlier “recirculation / near-miss” ideas

Earlier methods tried to learn by canceling effects of brief iteration in recurrent nets (Hinton & McClelland; O’Reilly; Seung), but without a stochastic generative model and clear objective.
CD provides a principled stochastic objective; learning is driven by differences between real data and model-generated “near misses” (Winston analogy). 

### 13.2 Logarithmic opinion pools: why disagreement helps

Combining experts via geometric mean has a KL guarantee:

$$\mathrm{KL}\left(P \parallel \frac{\prod_m Q_m^{w_m}}{Z}\right) \le \sum_m w_m \mathrm{KL}(P\|Q_m) \quad\text{Eq. 12}$$

and the benefit is tied to $-\log Z$: experts help when they disagree on unobserved data, making $Z<1$. 

Temptation: learn weights $w_m$. But varying $w_m$ makes inference harder (e.g., $w_m=100$ behaves like 100 tied copies of an expert). So paper recommends fixing $w_m=1$ and letting learning tune sharpness through parameters.

### 13.3 Comparison with directed acyclic graphical models

* **PoE advantage:** inference is easy because experts are individually tractable and independent given data (no “explaining away” across experts during inference).
* **PoE disadvantage:** sampling/generation is harder (needs Gibbs/MCMC), whereas directed models can sample ancestrally.
* With CD learning, the sampling difficulty is less of a barrier.
* Subtle advantage: even with independent priors, latent variables across experts can remain **marginally dependent**, leaving structure for deeper layers—supporting greedy layer-wise learning; directed models with independent priors tend to push posteriors toward marginal independence after fitting. 

## 14) What to remember (the “thesis statements”)

1. **PoE = multiply experts, renormalize** → sharp distributions via intersection of constraints; efficient in high dimensions. 
2. **ML training is hard** mainly due to sampling from equilibrium and high-variance negative phase. 
3. **Contrastive Divergence** replaces equilibrium negative phase with **short-run reconstructions**; objective is a difference of KLs that is nonnegative and zero only when model matches data. 
4. **Learning rule:** “data statistics – reconstruction statistics” (Eq. 6 / Eq. 10 / Eq. 11). 
5. **RBM is a PoE**, and CD-1 becomes the practical RBM training algorithm; empirically works well for digits and yields useful features + strong discriminative scores. 
6. The ignored gradient term is **usually safe**; CD improves reliably, though likelihood can occasionally decrease. 
