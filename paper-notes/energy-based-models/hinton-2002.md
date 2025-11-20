---
title: Training Products of Experts by Minimizing Contrastive Divergence
layout: default
noindex: true
---


## 1. Core Idea

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
    [
    p(d \mid \theta_1,\dots,\theta_n) \propto \prod_m p_m(d\mid\theta_m)
    ]
  * Each expert can focus on enforcing a **low-dimensional constraint** or pattern; the product enforces *all* constraints simultaneously. 
  * Allows **much sharper** distributions than any single expert; bad configurations get ruled out if any expert assigns them low probability.

### Intuition

* For images:

  * One expert may capture coarse shape, others local stroke segments, others contrast/edges.
* For language:

  * One expert enforces tense agreement, another subject–verb number agreement, another adjective order, etc. 

---

## 3. Maximum Likelihood Learning and Its Problems

* Log-likelihood gradient for expert (m) in a PoE:
  [
  \frac{\partial \log p(d)}{\partial \theta_m}
  ============================================

  ## \underbrace{\frac{\partial \log p_m(d)}{\partial \theta_m}}_{\text{data term}}

  \underbrace{\mathbb{E}*{c\sim p(\cdot)}\frac{\partial \log p_m(c)}{\partial \theta_m}}*{\text{model (negative) term}}
  ]
  where the expectation is over “fantasy” samples from the current PoE. 
* Main difficulty:

  * **Sampling from the equilibrium distribution** of the PoE (via Gibbs sampling, etc.) is expensive.
  * The model samples have **high variance** and this variance depends on parameters → unstable learning (“sand accumulating in the still regions of a vibrating sheet” analogy). 

---

## 4. Contrastive Divergence (CD)

### Objective

* Instead of minimizing just the KL divergence between data distribution (Q_0) and equilibrium model distribution (Q_\infty), the paper minimizes the **contrastive divergence**:
  [
  \text{CD} = Q_0 !\parallel! Q_1 ;-; Q_1 !\parallel! Q_\infty
  ]
  where:

  * (Q_0): empirical data distribution.
  * (Q_1): distribution of **one-step reconstructions** obtained by one full Gibbs step (hidden update + visible update) starting from data. 
* Intuition:

  * We want the Markov chain (Gibbs sampler) to **leave the data distribution unchanged**.
  * Instead of running to equilibrium, measure and reduce how much the chain moves on its **first step** away from data.

### Approximate Gradient

* The gradient of CD decomposes into three terms; two are tractable expectations over data and reconstructions, and one term involving how (Q_1) itself changes with parameters is **ignored**.
* Resulting practical update rule for expert (m):
  [
  \Delta \theta_m \propto
  \mathbb{E}_{d\sim Q_0}\left[\frac{\partial \log p_m(d)}{\partial \theta_m}\right]
  ---------------------------------------------------------------------------------

  \mathbb{E}_{\hat d\sim Q_1}\left[\frac{\partial \log p_m(\hat d)}{\partial \theta_m}\right]
  ]
  where (\hat d) is a one-step reconstruction of (d). 

### Properties

* **Low-variance learning**: reconstructions (\hat d) are close to data (d) when the model is reasonable → difference of terms has low variance.
* Allows **online or mini-batch learning**.
* Empirically, ignoring the third term still tends to **improve** CD and often also improves log-likelihood. 

---

## 5. Restricted Boltzmann Machines as PoE

* An **RBM**: visible and hidden units (binary in the basic version), with:

  * No visible–visible or hidden–hidden connections (bipartite graph).
* Each hidden unit can be seen as an **expert** that defines a distribution over visible units when it is on vs off → RBM = product of such experts. 
* The **standard RBM learning rule** (data statistics minus model statistics for (\langle s_i s_j\rangle)) is exactly the PoE maximum likelihood gradient.
* Under CD with one Gibbs step (CD-1), the weight update for visible unit (i) and hidden unit (j) is:
  [
  \Delta w_{ij} \propto
  \langle s_i s_j \rangle_{\text{data}}
  -------------------------------------

  \langle s_i s_j \rangle_{\text{one-step reconstructions}}
  ]
  (or using probabilities (p_i, p_j) for real-valued inputs). 

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

  * Compute **unnormalized log probability** (log p(t|model) + constant) under each class model.
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
* Scatter plots show that the **ignored term** in the CD gradient (due to change in (Q_1)) typically **helps** rather than harms the optimization.

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
* This work is one of the core stepping stones toward modern deep generative models based on RBMs and energy-based learning.
