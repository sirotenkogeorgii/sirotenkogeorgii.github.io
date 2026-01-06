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

For one data vector $d$, the log-likelihood gradient wrt expert $m$’s parameters is:

$$
\frac{\partial \log p(d\mid \theta_{1:n})}{\partial \theta_m}=\frac{\partial \log p_m(d\mid \theta_m)}{\partial \theta_m} - \sum_c p(c\mid \theta_{1:n}) \frac{\partial \log p_m(c\mid \theta_m)}{\partial \theta_m} \quad\text{Eq. 2}
$$

Interpretation:
* **Positive phase:** push expert to increase probability on real data $d$.
* **Negative phase:** decrease probability on “fantasy data” $c$ sampled from the full PoE model distribution. 

### 3.2 Sampling fantasy data is the bottleneck

To estimate the negative phase you need samples from $p(c\mid \theta_{1:n})$.

* Rejection sampling (“each expert samples independently until all agree”) is conceptually helpful but generally hopelessly inefficient.
* Use **MCMC / Gibbs sampling** instead:
  * Given data, hidden states of different experts are **conditionally independent**, so you can update all experts’ hidden variables in parallel.
  * If, given its hidden state, each expert makes visible dimensions conditionally independent (bipartite structure), you can also update all visible variables in parallel given all hidden states—leading to alternating parallel updates (hidden ↔ visible). 

### 3.3 Even if you can sample: variance kills you

Even with MCMC convergence, equilibrium samples have:

* **Very high variance** (they span the model distribution).
* Worse: sample variance depends on parameters, creating “repulsion” effects even if true gradient is zero (sand-on-vibrating-plate analogy). 

## 4) Key contribution: learning by minimizing Contrastive Divergence (CD)

### 4.1 Setup: distributions involved

* $Q_0$: data distribution over visibles (think: start of a Markov chain at time 0).
* $Q_\infty$: equilibrium distribution over visibles implied by the PoE after long Gibbs sampling (this is the model distribution).
* $Q_1$: distribution of visibles after **one full Gibbs step** starting from data (sample a reconstruction). 

### 4.2 ML is minimizing $KL((Q_0) \|\| (Q_\infty))$

The average log-likelihood objective equals minimizing:

$$
\mathrm{KL}(Q_0 \| Q_\infty) = H(Q_0) - \mathbb{E}_{d\sim Q_0}[\log Q_\infty(d)] \quad\text{Eq. 3 idea}
$$

Since $H(Q_0)$ doesn’t depend on parameters, maximizing likelihood is maximizing $\mathbb{E}_{Q_0}[\log Q_\infty(d)]$. 

### 4.3 CD objective: compare data to *nearby* reconstructions

Instead of running to equilibrium, minimize the tendency of a Gibbs chain to move away from the data distribution immediately.

Define **contrastive divergence (CD-1)** as:

$$\mathrm{CD}_1 = \mathrm{KL}(Q_0 \| Q_\infty) - \mathrm{KL}(Q_1 \| Q_\infty)$$

Properties emphasized:

* $Q_1$ is “one step closer to equilibrium than $Q_0$”, so $\mathrm{CD}_1 \ge 0$.
* Under mild conditions (nonzero transition probabilities), $\mathrm{CD}_1=0$ only if the model is perfect (data distribution equals equilibrium). 

### 4.4 Why CD is tractable: cancellation in the gradient

The derivative of the CD objective wrt expert parameters yields:
* A **data term** (expectation under $Q_0$)
* A **reconstruction term** (expectation under $Q_1$)
* Plus an extra term involving how $Q_1$ changes with parameters (the “problem term”)

The paper argues (empirically, Section 10) that this third term is typically small and rarely opposes the other terms, so you can ignore it. 

### 4.5 The practical learning rule (approximate gradient)

Update each expert $m$ by:

$$
\Delta \theta_m \propto \mathbb{E}_{d\sim Q_0}\Big[\frac{\partial \log p_m(d\mid \theta_m)}{\partial \theta_m}\Big]-\mathbb{E}_{\hat d\sim Q_1}\Big[\frac{\partial \log p_m(\hat d\mid \theta_m)}{\partial \theta_m}\Big]
\quad\text{Eq. 6}
$$

This is the “**positive phase minus negative phase**”, but the negative phase uses **one-step reconstructions** rather than equilibrium samples—dramatically reducing computation and variance. 

---

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

* Hidden unit OFF → factorial distribution where each visible bit is equally likely on/off.
* Hidden unit ON → different factorial distribution; weights specify log-odds of visible bits being on.
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

* Learned **localized** features; for each image $\sim ⅓$ of features active.
* Features include on-center/off-surround and vice versa, stroke fragments, Gabor/wavelet-like patterns.
* **Figure 4 (page 9):** receptive fields (weights) of 100 randomly selected hidden units show these local structures. 

---

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

$$\mathrm{KL}\left(P \| \frac{\prod_m Q_m^{w_m}}{Z}\right) \le \sum_m w_m \mathrm{KL}(P\|Q_m) \quad\text{Eq. 12}$$

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

<!-- ## 1. Core Idea

* Introduces **Products of Experts (PoE)**: combine several probabilistic models (“experts”) by **multiplying** their distributions and renormalizing, instead of averaging as in mixtures.
* Proposes **Contrastive Divergence (CD)** as a practical approximate learning objective for PoE models, avoiding the expensive computation of the partition function’s gradient. 
* Shows that **restricted Boltzmann machines (RBMs)** are a special case of PoE, and that CD gives an efficient learning rule for RBMs. 

---

## 2. Why Products of Experts?

### Mixtures vs Products

* **Mixture of experts**:

  * Overall distribution is a weighted **average** of individual experts’ distributions.
  * Each expert must individually cover the full high-dimensional space → tends to be **broad and inefficient** in high dimensions. 
* **Product of experts**:

  * Overall distribution is proportional to the **product** of expert distributions:
    
    $$
    p(d \mid \theta_1,\dots,\theta_n) \propto \prod_m p_m(d\mid\theta_m)
    $$

  * Each expert can focus on enforcing a **low-dimensional constraint** or pattern; the product enforces *all* constraints simultaneously. 
  * Allows **much sharper** distributions than any single expert; bad configurations get ruled out if any expert assigns them low probability.

> If the individual distributions are uni- or multivariate gaussians, their product will also be a multivariate gaussian so, unlike mixtures of gaussians, products of gaussians cannot approximate arbitrary smooth distributions. If, however, the individual models are a bit more complicated and each contains one or more latent (i.e., hidden) variables, multiplying their distributions together (and renormalizing) can be very powerful. Individual models of this kind will be called “experts.”

### Intuition

* For images: One expert may capture coarse shape, others local stroke segments, others contrast/edges.
* For language: One expert enforces tense agreement, another subject–verb number agreement, another adjective order, etc. 


---

## 3. Maximum Likelihood Learning and Its Problems

* Prediction with $n$ combined experts:
  
  $$
  p(d\mid \theta_1, \dots, \theta_n) = \frac{\prod_m f_m(d\mid \theta_m)}{\sum_c \prod_m f_m(c\mid \theta_m)}
  $$

  Here:
  * $d$ represents a data vector.
  * $f_m$ is the (unnormalized) probability of the data under expert model m with parameters $\theta_m$.
  * The denominator is a partition function summing over all possible data vectors $c$.

* Log-likelihood gradient for expert $m$ in a PoE:

  $$
  \frac{\partial \log p(\mathbf{d} \mid \theta_1, \dots, \theta_n)}{\partial \theta_m} = \frac{\partial \log f_m(\mathbf{d} \mid \theta_m)}{\partial \theta_m} - \sum_\mathbf{c} p(\mathbf{c} \mid \theta_1, \dots, \theta_n) \frac{\partial \log f_m(\mathbf{c} \mid \theta_m)}{\partial \theta_m}
  $$
  
* Main difficulty:

  * **Sampling from the equilibrium distribution** of the PoE (via rejection sampling, Gibbs sampling, etc.) is expensive.
  * The model samples have **high variance** and this variance depends on parameters → unstable learning. This high variance can completely "swamp the estimate of the derivative," making the learning signal unreliable.

These challenges with maximum likelihood training established the primary motivation for developing an alternative, more tractable objective function.

---

## 4. Contrastive Divergence (CD)

### Objective

* Instead of minimizing just the KL divergence between data distribution $Q_0$ and equilibrium model distribution $Q_\infty$, the paper minimizes the **contrastive divergence**:
  
  $$
  \text{CD} = Q_0 \| Q_1 - Q_1 \| Q_\infty
  $$

  where:

  * $Q_0$: empirical data distribution.
  * $Q_1$: distribution of **one-step reconstructions** obtained by one full Gibbs step (hidden update + visible update) starting from data. 
* Intuition:

  * We want the Markov chain (Gibbs sampler) to **leave the data distribution unchanged**.
  * Instead of running to equilibrium, measure and reduce how much the chain moves on its **first step** away from data.
  
> Fitting a PoE to data appears difficult because it appears to be necessary to compute the derivatives, with repect to the parameters, of the partition function that is used in the renormalization. As we shall see, however, these derivatives can be finessed by optimizing a less obvious objective function than the log likelihood of the data.

### Approximate Gradient

* The gradient of CD decomposes into three terms; two are tractable expectations over data and reconstructions, and one term involving how $Q_1$ itself changes with parameters is **ignored**.
* Resulting practical update rule for expert $m$:
  
  $$
  \Delta \theta_m \propto \mathbb{E}_{d\sim Q_0}\left[\frac{\partial \log p_m(d)}{\partial \theta_m}\right] - \mathbb{E}_{\hat d\sim Q_1}\left[\frac{\partial \log p_m(\hat d)}{\partial \theta_m}\right]
  $$

  where $\hat d$ is a one-step reconstruction of $d$. 

### Properties

* **Low-variance learning**: reconstructions $\hat d$ are close to data $d$ when the model is reasonable → difference of terms has low variance.
* Allows **online or mini-batch learning**.
* Empirically, ignoring the third term still tends to **improve** CD and often also improves log-likelihood. 

---

## 5. Restricted Boltzmann Machines as PoE

* An **RBM**: visible and hidden units (binary in the basic version), with:

  * No visible–visible or hidden–hidden connections (bipartite graph).
* Each hidden unit can be seen as an **expert** that defines a distribution over visible units when it is on vs off → RBM = product of such experts. 
* The **standard RBM learning rule** (data statistics minus model statistics for $\langle s_i s_j\rangle$) is exactly the PoE maximum likelihood gradient.
* Under CD with one Gibbs step (CD-1), the weight update for visible unit $i$ and hidden unit $j$ is:
  
  $$
  \Delta w_{ij} \propto \langle s_i s_j \rangle_{\text{data}} - \langle s_i s_j \rangle_{\text{one-step reconstructions}}
  $$
  
  (or using probabilities $p_i, p_j$ for real-valued inputs). 

---

## 6. Experiments & Examples

### 6.1 Toy 2D Factorized Data

* Uses “**unigauss**” experts: each is a mixture of a uniform distribution and a single axis-aligned Gaussian. 
* PoE of 15 such experts fits a 2D data distribution with grid-like clusters.
* Each cluster is explained as the **intersection** of a pair of elongated Gaussians; unnecessary experts stay vague.
* Gibbs sampling from the learned PoE generates synthetic data matching the observed grid structure (even fills in a missing grid point). 

### 6.2 100D Edge Images

* Data: synthetic 10×10 images (100D) containing a single edge with varying position, orientation, and contrast. 
* Product of 40 unigauss experts learns:

  * **Edge-like features** at different orientations and positions.
  * Some even-symmetric features marking edge endpoints.
* Each expert is broadly tuned across pixels; precision comes from their **intersection** (population coding).

### 6.3 RBM on USPS Handwritten Digits

* Model: RBM with **500 hidden units**, **256 visible units**, trained on 8,000 16×16 real-valued digit images from all 10 classes. 
* Uses CD learning with probabilities instead of binary states for pixels (but binary for hidden).
* Learned features:

  * Localized receptive fields: center-surround patterns, stroke fragments, Gabor/wavelet-like filters.
  * About one-third of features active for a given image. 

### 6.4 Class-Specific PoE Models for Digit Recognition

* Trains **separate PoE/RBM models per digit class** (e.g., only “2” images vs only “3” images).
* For classification:

  * Compute **unnormalized log probability** ($\log p(t\mid \text{model}) + \text{constant}$) under each class model.
  * Since partition function differences between models are global constants, they can be estimated discriminatively or absorbed into a discriminative layer. 
* Demonstrations:

  * Digit “2” vs “3”: reconstructed “2”s look much better under the model trained on 2s than under the 3s model.
  * Digit “4” vs “6”: with two-hidden-layer models per class and model averaging, achieves **perfect separation** on both training and test sets considered in the paper. 
  * Digit “7” vs “9”: the hardest pair; still obtains good separation, with errors only near the decision boundary. 

### 6.5 Multi-class Setup

* Trains 10 digit-specific PoEs. For each class, uses **two scores**:

  1. Unnormalized log probability of pixels under first hidden layer PoE.
  2. Unnormalized log probability of first-layer hidden activations under a **second-layer PoE**. 
* A multinomial logistic regression maps these 20 scores to class probabilities.
* Reported performance:

  * **~1.1% error** on the USPS training/test split used in the paper.
  * With ~7% reject rate (abstaining on low-confidence cases), **zero errors** among the remaining test samples. 

---

## 7. Quality of the CD Approximation

* For small RBMs (few visible and hidden units), the paper computes:

  * Exact gradients of log-likelihood and exact CD changes.
* Empirical result: 

  * For single weights, the approximate CD gradient sometimes has the wrong sign.
  * But **for the full parameter vector**, a parallel weight update using CD almost always improves the contrastive divergence.
  * Log-likelihood usually improves as well; occasional small decreases are observed.
* Scatter plots show that the **ignored term** in the CD gradient (due to change in $Q_1$) typically **helps** rather than harms the optimization.

---

## 8. Other Types of Experts

The paper sketches variations beyond simple binary RBMs: 

* **Multinomial or replicated units**:

  * Real-valued intensities modeled by multiple identical binary units (replicas) whose count of “on” states approximates the value.
* **Unifac experts**:

  * Mixture of a **uniform distribution** and a **single-factor factor analyzer** (one latent factor per expert).
  * Each expert has a loading vector, mean vector, and variance vector; can capture structured covariance in a low-rank way.
* **Products of HMMs**:

  * Several HMMs act as experts over sequences; each contributes a constraint, and the product yields exponentially more efficient modeling of mutual information between past and future (linear in number of HMMs rather than exponential in states).
  * CD learning applied by running forward–backward independently in each HMM, sampling a path, then combining outputs multiplicatively.

---

## 9. Conceptual Connections & Discussion

* **Logarithmic opinion pools**:

  * PoE is a special case of combining distributions via a (possibly weighted) geometric mean.
  * Benefit arises when experts **disagree** on unobserved data, making the normalization constant small and sharpening the distribution. 
* **Comparison with directed graphical models**:

  * Directed models: easy ancestral sampling, but inference is hard (explaining away, approximate inference needed).
  * PoE: inference is easy (experts conditionally independent given data), but sampling is hard; CD bypasses the need for long-run sampling during training. 
* **Greedy multi-layer learning**:

  * In directed models with independent priors on latent variables, learned posteriors tend to be marginally independent → little structure left for higher layers.
  * In PoEs, latent variables of different experts remain **correlated**, so higher layers can still capture structure in learned features. 
* **Analysis-by-synthesis**:

  * PoE + CD is framed as a successful instance of the old idea of explaining data by generating it from a model and comparing to near misses, but now with a clear probabilistic objective (contrastive divergence). 

---

## 10. Takeaways

* **Products of Experts** provide a powerful alternative to mixtures, especially in high dimensions, by allowing many simple experts to jointly impose strong constraints.
* **Contrastive Divergence** is a practical and surprisingly effective approximate learning method for such undirected models, avoiding full partition-function gradients while still improving both CD and log-likelihood in practice.
* The paper lays a theoretical and practical foundation for:

  * Training **RBMs** efficiently.
  * Using PoEs for **generative modeling** and **classification**, especially in vision and sequence modeling.
* This work is one of the core stepping stones toward modern deep generative models based on RBMs and energy-based learning. -->
