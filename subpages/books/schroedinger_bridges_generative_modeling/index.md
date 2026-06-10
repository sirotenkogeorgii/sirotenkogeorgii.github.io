---
layout: default
title: Foundations of Schroedinger Bridges for Generative Modeling
date: 2024-11-01
tags:
  - machine-learning
  - generative-models
  - schroedinger-bridges
  - optimal-transport
  - stochastic-differential-equations
  - diffusion-models
  - stochastic-control
---

<style>
  .accordion summary {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    background-color: var(--accent-soft, #f5f6ff);
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--accent-strong, #2c3e94);
    border-radius: 0.25rem;
  }
</style>

**Table of Contents**
- TOC
{:toc}

# Foundations of Schroedinger Bridges for Generative Modeling

## Preface

The field of generative modeling has experienced rapid and transformative progress in recent years. Advances range from foundational theoretical developments --- such as diffusion models and flow matching --- to algorithmic improvements in sampling speed and generation quality, as well as significant applications spanning language, video, and scientific domains.

This guide introduces **Schroedinger bridges as a unifying theoretical framework for generative modeling**. This perspective generalizes a broad class of modern approaches --- including diffusion models, flow-matching methods, and stochastic control formulations --- while providing a principled and flexible foundation for addressing specialized scientific problems.

The goal is to build both intuition and a deep mathematical understanding of the core principles of Schroedinger bridges, from origins in optimal transport to the dynamic path space formulation underlying modern generative modeling frameworks. At a high level, the approach begins from a **single unifying principle**: optimal stochastic bridges between distributions can be characterized as minimal-entropy deviations from a reference process subject to marginal constraints.

### Guide Structure

**Section 1** introduces the **static Schroedinger bridge (SB) problem**, providing the theoretical foundations for optimal transport between probability distributions.

- Section 1.1: Classical **optimal mass transport** (OMT) problem (Monge and Kantorovich).
- Section 1.2: Foundational properties of **entropy** on probability spaces, including the KL divergence.
- Section 1.3: **Entropic optimal transport** (EOT) problem --- regularizing OMT with a reference coupling.
- Section 1.4: Formalizing the **static Schroedinger bridge problem** and its **dual problem** via Schroedinger potentials.
- Section 1.5: **Sinkhorn's algorithm** for solving the static SB problem.

**Section 2** lifts the static SB problem into the space of stochastic path measures: the **dynamic Schroedinger bridge (SB) problem**.

- Section 2.1: Redefining the static SB as learning a continuous-time deterministic flow --- **dynamic optimal transport** (OT).
- Section 2.7: The **dynamic Schroedinger bridge** as entropy minimization over stochastic processes.
- Section 2.3: Path measures as SDEs with **control drifts** and **transformations** through functions.
- Section 2.4: **Fokker--Planck equation** (forward density evolution) and **Feynman--Kac equation** (backward function evolution).
- Section 2.5: **Girsanov's theorem** --- changes in measure and KL divergences on path space.
- Section 2.6: **Radon--Nikodym derivatives** between path measures for defining relative entropy.

**Section 3** reformulates the SB problem through **stochastic optimal control** (SOC) theory.

- Section 3.1: General framework of SOC, Bellman's Principle of Optimality, and the value function.
- Section 3.2: Connecting SB to SOC --- the optimal bridge as an **optimal control drift**.
- Section 3.3: **Practical objectives** for solving the SOC problem.

**Section 4** presents several complementary mechanisms for **building Schroedinger bridges**.

- Section 4.1: **Mixtures of conditional bridges** with a pre-defined endpoint coupling.
- Section 4.2: **Time-reversal formula** of SDEs --- fundamental to backward dynamics in SB.
- Section 4.3: **Forward-backward stochastic differential equations** (FBSDEs) --- coupled characterization via time-dependent Schroedinger potentials.
- Section 4.4: **Doob's $h$-transform** --- constructing conditioned stochastic processes by tilting the reference process.
- Section 4.5: **Markovian and reciprocal projections** --- entropy-minimizing projections in path space converging to the optimal SB measure.
- Section 4.6: **Stochastic interpolants** --- constructing bridges as deterministic interpolants with Gaussian noise.

**Section 5** explores important **variations of the Schroedinger bridge problem**.

- Section 5.1: **Gaussian SB problem** (closed-form solutions).
- Section 5.2: **Generalized SB problem** (mean-field interactions).
- Section 5.3: **Multi-marginal SB problem** (multiple intermediate marginal constraints).
- Section 5.4: **Unbalanced SB problem** (mass creation and destruction along trajectories).
- Section 5.5: **Branched SB problem** (diverging trajectories to multiple terminal modes).
- Section 5.6: **Fractional SB problems** (long-range temporal dependencies via fractional Brownian motion).

**Section 6** connects SB theory to **modern generative modeling** frameworks.

- Section 6.1: **Score-based generative modeling** (learning gradients of log-densities).
- Section 6.2: **Likelihood training of forward-backward SDEs**.
- Section 6.3: **Diffusion Schroedinger bridge matching** (Iterative Markovian Fitting with learned Markov drift).
- Section 6.4: **Simulation-free score and flow matching**.
- Section 6.5: **Adjoint matching** (learning optimal SB without explicit sampling from targets).

**Section 7** extends SB to **discrete state spaces**.

- Section 7.1: **Continuous-time Markov chains** (CTMCs) as discrete analogues of stochastic processes.
- Section 7.2: **Discrete Schroedinger bridge problem** (Radon--Nikodym derivatives and KL divergences for CTMCs).
- Section 7.3: **Stochastic optimal control for CTMCs**.
- Section 7.4: Connecting discrete SB with SOC --- practical algorithms and objectives.
- Section 7.5: **Markovian and reciprocal projections in discrete spaces**.
- Section 7.6: **Discrete diffusion Schroedinger bridge matching**.

**Section 8** highlights diverse **applications of generative modeling with Schroedinger bridges**.

- Section 8.1: **Data translation** (mapping between structured data distributions).
- Section 8.2: **Single-cell state dynamics** (modeling cell population dynamics and perturbation responses).
- Section 8.3: **Sampling Boltzmann distributions** (generating from unnormalized energy distributions without explicit samples).

## Notation

Throughout this guide, the **control drift** refers to the added term to the reference drift, scaled by the diffusion coefficient, generally denoted $\sigma\_t \boldsymbol{u}(\boldsymbol{x}, t)$. The **velocity field** refers to the entire non-diffusion term in the SDE that appears before $dt$, including both the reference drift and the control drift, generally denoted $\boldsymbol{v}(\boldsymbol{x}, t)$.

A key notational convention: $\boldsymbol{X}\_{0:T}^{\boldsymbol{u}} = (\boldsymbol{X}\_t)\_{t \in [0,T]}^{\boldsymbol{u}}$ emphasizes that the path measure is generated under a *specific* control drift $\boldsymbol{u}(\boldsymbol{x}, t)$. When the underlying path measure is clear from context, this superscript is omitted.

### Basic Notation

| Notation | Meaning |
| --- | --- |
| $\text{KL}(\cdot \| \cdot)$ | Kullback--Leibler divergence |
| $\inf$, $\sup$ | infimum / supremum |
| $\partial\_t$ | partial derivative with respect to $t$ |
| $d, dt, d\boldsymbol{x}$ | differential, time differential, state differential |
| $\nabla$ | gradient operator |
| $\nabla \cdot$ | divergence operator |
| $\Delta = \nabla \cdot \nabla = \nabla^2$ | Laplacian operator (shorthand for $\Delta\_{\boldsymbol{x}}$) |
| $\phi \in C^2(\mathbb{R}^d)$ | twice continuously differentiable functions w.r.t. $\boldsymbol{x} \in \mathbb{R}^d$ |
| $\phi \in C^{2,1}(\mathbb{R}^d \times [0,T])$ | twice continuously differentiable in $\boldsymbol{x}$, once in $t$ |
| $\phi \in L^2$ | square integrable functions with finite $L^2$ norm |
| $I\_d$ | $d$-dimensional identity matrix |
| $\mathbf{1}\_{\boldsymbol{x}=\boldsymbol{y}}$ | indicator function: returns 1 when $\boldsymbol{x} = \boldsymbol{y}$, 0 otherwise |
| $\frac{\delta \mathcal{L}}{\delta \boldsymbol{u}}$ | functional derivative of $\mathcal{L}$ with respect to $\boldsymbol{u}$ |

### Matrix and Vector Operations

| Notation | Meaning |
| --- | --- |
| $\langle \cdot, \cdot \rangle$ | inner product |
| $\lVert \cdot \rVert^2$ | $L^2$-norm (unless specified otherwise) |
| $\boldsymbol{A}^\top$ | vector or matrix transpose |
| $\boldsymbol{A}^{-1}$ | matrix inverse |

### State Spaces

| Notation | Meaning |
| --- | --- |
| $\mathcal{X}, \mathcal{Y}$ | arbitrary state spaces ($\mathcal{X}$ source, $\mathcal{Y}$ target; in Section 7, $\mathcal{X} := \lbrace 1, \dots, d \rbrace$ for finite state spaces) |
| $\mathcal{X} \times \mathcal{Y}$ | product space of $\mathcal{X}$ and $\mathcal{Y}$ |
| $\mathbb{R}^d$ | $d$-dimensional state space |
| $\mathcal{P}(\cdot)$ | probability space |
| $\mathcal{P}(\mathbb{R}^d)$ | space of probability distributions over $\mathbb{R}^d$ |
| $C([0,T]; \mathbb{R}^d)$ | path space over time horizon $t \in [0,T]$ and state space $\mathbb{R}^d$ |
| $\mathcal{P}(C([0,T]; \mathbb{R}^d))$ | space of probability distributions over the path space |
| $\mathcal{U}$ | space of feasible control drifts |

### Probability Densities

| Notation | Meaning |
| --- | --- |
| $p\_t$ | marginal density of path measure at time $t$ (typically of $\mathbb{P}^{\boldsymbol{u}}$) |
| $p\_0, p\_T$ | initial and terminal marginals generated from path measure $\mathbb{P}$ |
| $p\_t^\star$ | marginal density of optimal path measure $\mathbb{P}^\star$ at time $t$ |
| $p\_0^\star, p\_T^\star$ | initial and terminal marginals of the optimal path measure |
| $q\_t$ | marginal density of reference path measure $\mathbb{Q}$ at time $t$ |
| $q\_0, q\_T$ | initial and terminal marginals of reference path measure $\mathbb{Q}$ |
| $\bar{p}\_s$ | marginal density w.r.t. reverse time coordinate $s := T - t$ |
| $\pi\_0$ | marginal distribution constraint at $t = 0$ |
| $\pi\_T$ | marginal distribution constraint at $t = T$ |
| $\pi\_{0,T}$ | endpoint law or coupling distribution |
| $\pi\_0 \otimes \pi\_T$ | product measure |
| $\pi\_{0,T}^\star$ | optimal OT or SB endpoint law / coupling distribution |

### Stochastic Processes and Path Measures

| Notation | Meaning |
| --- | --- |
| $\boldsymbol{x}$ | realized state in $\mathbb{R}^d$ |
| $\boldsymbol{X}\_t$ | random variable in $\mathbb{R}^d$ |
| $\boldsymbol{X}\_{0:T}, (\boldsymbol{X}\_t)\_{t \in [0,T]}$ | forward stochastic process over $t \in [0,T]$ |
| $\bar{\boldsymbol{X}}\_{0:T}, (\bar{\boldsymbol{X}}\_s)\_{s \in [0,T]}$ | backward stochastic process over reverse time $s = T - t$ |
| $\boldsymbol{X}\_{0:T}^{\boldsymbol{u}}, (\boldsymbol{X}\_t^{\boldsymbol{u}})\_{t \in [0,T]}$ | forward stochastic process with control $\boldsymbol{u}$ |
| $\boldsymbol{B}\_t$ | Brownian motion random variable |
| $\mathbb{P}, \mathbb{Q}$ | path measures in $\mathcal{P}(C([0,T]; \mathbb{R}^d))$ |
| $\mathbb{P}\_{0,T}, \mathbb{Q}\_{0,T}, \mathbb{P}\_{0,T}^\star, \mathbb{P}\_{0,T}^{\boldsymbol{u}}, \Pi\_{0,T}, \mathbb{M}\_{0,T}$ | endpoint law of path measure |
| $\mathbb{P}^{\boldsymbol{u}}$ | controlled path measure with control drift $\boldsymbol{u}$ |
| $\mathbb{Q}$ | reference path measure defining the prior dynamics |
| $\sigma \mathbb{B}$ | pure Brownian motion path measure with SDE $d\boldsymbol{X}\_t = \sigma\_t d\boldsymbol{B}\_t$ |
| $\mathbb{P}^\star$ | path measure of the optimal solution |
| $\Pi$ | mixture of bridges under the reference process: $\Pi := \Pi\_{0,T} \mathbb{Q}\_{\cdot \mid 0,T}$ |
| $\mathcal{R}(\mathbb{Q})$ | reciprocal class containing all mixtures of bridges |
| $\mathbb{M}$ | Markov path measure |
| $\mathcal{M}$ | space of Markov measures |
| $\mathbb{M}^\star := \text{proj}\_{\mathcal{M}}(\cdot)$ | Markov projection of a reciprocal measure |
| $\Pi^\star := \text{proj}\_{\mathcal{R}(\mathbb{Q})}(\cdot)$ | reciprocal projection of a reciprocal measure |
| $\mathbb{P}^{\tilde{u}} \ll \mathbb{P}^{\boldsymbol{u}}$ | absolute continuity of $\mathbb{P}^{\tilde{u}}$ w.r.t. $\mathbb{P}^{\boldsymbol{u}}$ |
| $\mathbb{P}^{\tilde{u}} \sim \mathbb{P}^{\boldsymbol{u}}$ | mutual absolute continuity |
| $\mathbb{P}\_{\tau \mid t}(\boldsymbol{x}\_\tau \mid \boldsymbol{x})$ | transition density from state $\boldsymbol{x}$ at time $t$ to $\boldsymbol{x}\_\tau$ at time $\tau$ |

### Schroedinger Bridge and Generative Modeling Theory

| Notation | Meaning |
| --- | --- |
| $\boldsymbol{f}(\boldsymbol{x}, t) : \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ | uncontrolled drift of reference process $\mathbb{Q}$ |
| $\varphi : \mathcal{X} \to \mathbb{R}, \; \hat{\varphi} : \mathcal{Y} \to \mathbb{R}$ | static Schroedinger potentials on $\mathcal{X}$ and $\mathcal{Y}$ |
| $\varphi \oplus \hat{\varphi}$ | separable sum of functions on two coordinates |
| $\varphi\_t(\boldsymbol{x}) : \mathbb{R}^d \times [0,T] \to \mathbb{R}$ | forward Schroedinger bridge potential |
| $\hat{\varphi}\_t(\boldsymbol{x}) : \mathbb{R}^d \times [0,T] \to \mathbb{R}$ | backward Schroedinger bridge potential |
| $\boldsymbol{u}(\boldsymbol{x}, t) : \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ | control drift |
| $\boldsymbol{u}^\star(\boldsymbol{x}, t)$ | Schroedinger bridge drift or optimal control drift |
| $\bar{\boldsymbol{u}}(\boldsymbol{x}, t)$ | non-gradient-tracking control drift: $\bar{\boldsymbol{u}} = \text{stopgrad}(\boldsymbol{u})$ |
| $\boldsymbol{v}(\boldsymbol{x}, t)$ | velocity field or arbitrary control drift (context-dependent) |
| $\boldsymbol{\Sigma}\_t : [0,T] \to \mathbb{R}^{d \times d}$ | diffusion covariance matrix (simplified with scalar $\sigma\_t$) |
| $\sigma\_t : [0,T] \to \mathbb{R}\_{\ge 0}$ | scalar diffusion coefficient |
| $c(\boldsymbol{x}, \boldsymbol{y}) : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ | transport cost (Section 1) |
| $c(\boldsymbol{x}, t) : \mathbb{R}^d \times [0,T] \to \mathbb{R}$ | state cost or potential (remaining sections) |
| $V\_t(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$ | value function (optimal cost-to-go, aim: minimize) |
| $\psi\_t(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$ | Lagrange multiplier (aim: maximize) |
| $\Phi(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$ | terminal cost |
| $\mathcal{A}$ | uncontrolled generator |
| $\mathcal{A}^{\boldsymbol{u}}$ | controlled generator |
| $t$ | time coordinate $t \in [0,T]$ |
| $s$ | another time coordinate; sometimes reverse time $s = T - t$ |
| $\tau$ | another time coordinate, typically for integration |
| $\Delta t$ | finite time step |
| $\theta, \phi$ | neural network parameters |
| $M$ or $M\_t$ | transport map: maps distributions via push-forward $M\_$#$ p = p'$ |

## 1. The Static Schroedinger Bridge Problem

We begin with the origins of Schroedinger bridges in the classical **optimal mass transport** (OMT) problem, in both Monge's and Kantorovich's formulations. This perspective naturally frames the static Schroedinger bridge problem as a probabilistic regularization of the OMT problem, linking deterministic transport theory with stochastic processes and entropy minimization.

### 1.1 The Optimal Mass Transport Problem

The origins of the Schroedinger bridge problem trace back to the problem of **optimal mass transport** (OMT), which defines an optimal mapping between points in one distribution to another.

Consider two probability distributions $\pi\_0 \in \mathcal{P}(\mathcal{X})$ and $\pi\_0 \in \mathcal{P}(\mathcal{Y})$ and a transport cost function $c(\boldsymbol{x}, \boldsymbol{y}) : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ which determines the cost of transporting a unit of mass from $\boldsymbol{x} \in \mathcal{X}$ to $\boldsymbol{y} \in \mathcal{Y}$. The space of **transport maps** $M : \mathcal{X} \to \mathcal{Y}$ that generate $\pi\_T$ as the pushforward of $\pi\_0$ (i.e., $M\_\# \pi\_0 = \pi\_T$) is:

$$
\mathcal{T}(\pi_0, \pi_T) = \lbrace M : \mathcal{X} \to \mathcal{Y} \mid M_\# \pi_0 = \pi_T \rbrace
$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.1</span><span class="math-callout__name">(Monge's Optimal Mass Transport Problem)</span></p>

Given probability distributions $\pi\_0 \in \mathcal{P}(\mathcal{X})$ and $\pi\_0 \in \mathcal{P}(\mathcal{Y})$ and cost function $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$, Monge's OMT problem aims to find the **optimal transport map** $M^\star$ that minimizes:

$$
\inf_{M \in \mathcal{T}(\pi_0, \pi_T)} \left\lbrace \int_{\mathcal{X}} c(\boldsymbol{x}, M(\boldsymbol{x})) d\pi_0(\boldsymbol{x}) \;\middle|\; M : \mathcal{X} \to \mathcal{Y} \text{ and } M_\# \pi_0 = \pi_T \right\rbrace
$$

</div>

Monge's OMT problem can be **ill-posed**: it may yield no solution when $\pi\_0$ is concentrated at a single point (e.g., a Dirac delta) and $\pi\_T$ is concentrated at multiple points, since mass must be *split* to reach both targets and the space of deterministic transport maps $\mathcal{T}(\pi\_0, \pi\_T)$ is empty.

To avoid ill-posedness, Kantorovich reformulated the OMT problem as an optimization over the space of **optimal couplings** $\pi\_{0,T} \in \Pi(\pi\_0, \pi\_T)$ defined as:

$$
\Pi(\pi_0, \pi_T) = \left\lbrace \pi_{0,T} \in \mathcal{P}(\mathcal{X} \times \mathcal{Y}) \;\middle|\; (\text{proj}_{\mathcal{X}})_\# \pi_{0,T} = \pi_0, \; (\text{proj}_{\mathcal{Y}})_\# \pi_{0,T} = \pi_T \right\rbrace
$$

where $(\text{proj}\_{\mathcal{X}})\_\# \pi\_{0,T}$ is the $\mathcal{X}$-marginal of $\pi\_{0,T}$ and vice versa. $\Pi(\pi\_0, \pi\_T)$ is never empty since it always contains the product measure $\pi\_0 \otimes \pi\_T$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.2</span><span class="math-callout__name">(Kantorovich's Optimal Mass Transport Problem)</span></p>

Given probability distributions $\pi\_0 \in \mathcal{P}(\mathcal{X})$ and $\pi\_0 \in \mathcal{P}(\mathcal{Y})$ and cost function $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$, Kantorovich's OMT problem aims to find the **optimal coupling** $\pi\_{0,T}^\star$ that minimizes:

$$
\inf_{\pi_{0,T}} \left\lbrace \int_{\mathcal{X} \times \mathcal{Y}} c(\boldsymbol{x}, \boldsymbol{y}) d\pi_{0,T}(\boldsymbol{x}, \boldsymbol{y}) \;\middle|\; \pi_{0,T} \in \Pi(\pi_0, \pi_T) \right\rbrace
$$

</div>

While Kantorovich's formulation guarantees existence of an optimal coupling, the resulting problem remains a linear optimization in $\pi\_{0,T} \in \mathcal{X} \times \mathcal{Y}$ that is susceptible to non-unique and deterministic mappings: for each state $\boldsymbol{x} \sim \pi\_0$, the distribution $\pi\_{0,T}(\boldsymbol{x}, \cdot) \in \mathcal{P}(\mathcal{Y})$ is sparse and concentrated at few points. To obtain a smoother and more statistically meaningful coupling, we consider **entropy regularization** as a strategy for optimizing a *stochastic coupling* where each state $\boldsymbol{x} \sim \pi\_0$ yields a smooth distribution $\pi\_{0,T}(\boldsymbol{x}, \cdot)$ over possible mappings.

### 1.2 Entropy on Probability Spaces

Before introducing **entropic optimal transport**, we establish the concept of **entropy**, which measures the *uncertainty* of one probability measure with respect to another.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.3</span><span class="math-callout__name">(Entropy Between Probability Measures)</span></p>

Consider two probability measures $p, q \in \mathcal{P}(\mathcal{X})$ on a measurable state space $\mathcal{X} \subseteq \mathbb{R}^d$. The **entropy** of $p$ *relative to* the measure $q$ is defined as:

$$
\text{KL}(p \| q) := \begin{cases} \mathbb{E}_p\!\left[\log \frac{dp}{dq}\right] & q \ll p \\ \infty & q \not\ll p \end{cases}
$$

where $q \ll p$ means that $q$ is absolutely continuous with respect to $p$ such that for all measurable sets $A \subseteq \mathcal{X}$, we have $p(A) = 0 \implies q(A) = 0$. $\text{KL}(\cdot \| \cdot)$ is also known as the **Kullback--Leibler (KL) divergence**.

</div>

Intuitively, $\text{KL}(p \| q)$ measures the *expected log-likelihood ratio under $p$*, indicating the excess *distributional mismatch* incurred when data generated under $p$ is modeled under $q$. The KL divergence is **asymmetric**: $\text{KL}(p \| q)$ measures how well $q$ approximates $p$, which is not equivalent to how well $p$ approximates $q$. We use KL divergence in entropy-regularized OT and Schroedinger bridges to penalize deviations from the *reference* coupling or path measure.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1.4</span><span class="math-callout__name">(Chain Rule of KL Divergences)</span></p>

Given two joint probability measures $\pi\_{X,Y}, \pi'\_{X,Y} \in \mathcal{P}(\mathcal{X} \times \mathcal{Y})$ that are absolutely continuous $\pi\_{0,T} \ll \pi'\_{0,T}$, denote the $\mathcal{X}$-marginals as $\pi\_X := \int\_{\mathcal{Y}} \pi\_{0,T} d\boldsymbol{y}$ and $\pi'\_X := \int\_{\mathcal{Y}} \pi'\_{0,T} d\boldsymbol{y}$, and the conditional distribution on $\mathcal{Y}$ given $\boldsymbol{x} \in \mathcal{X}$ as $\pi\_{Y \mid X}$ and $\pi'\_{Y \mid X}$. Then, the **KL divergence decomposes** into:

$$
\text{KL}(\pi_{X,Y} \| \pi'_{X,Y}) = \text{KL}(\pi_X \| \pi_Y) + \mathbb{E}_{\boldsymbol{x} \sim \pi_X} \left[ \text{KL}\!\left( \pi_{Y \mid X}(\cdot \mid \boldsymbol{x}) \| \pi'_{Y \mid X}(\cdot \mid \boldsymbol{x}) \right) \right]
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (KL Chain Rule)</summary>

The proof follows from the definition of KL divergence:

$$
\text{KL}(\pi_{X,Y} \| \pi'_{X,Y}) = \int_{\mathcal{X} \times \mathcal{Y}} \log \frac{d\pi_{X,Y}}{d\pi'_{X,Y}} d\pi_{X,Y} = \int_{\mathcal{X} \times \mathcal{Y}} \log \frac{d\pi_X}{d\pi'_X} \cdot \frac{d\pi_{Y \mid X}}{d\pi'_{Y \mid X}} \, d\pi_{X,Y}
$$

$$
= \int_{\mathcal{X} \times \mathcal{Y}} \log \frac{d\pi_X}{d\pi'_X} d\pi_{X,Y} + \int_{\mathcal{X}} \int_{\mathcal{Y}} \log \frac{d\pi_{Y \mid X}}{d\pi'_{Y \mid X}} d\pi_{Y \mid X} \, d\pi_X
$$

$$
= \text{KL}(\pi_X \| \pi'_X) + \mathbb{E}_{\pi_X}\!\left[ \text{KL}(\pi_{Y \mid X} \| \pi'_{Y \mid X}) \right]
$$

$\square$
</details>
</div>

An immediate consequence of the KL chain rule is the **data processing inequality**, which formalizes that applying the same stochastic transformation to two probability measures cannot increase their divergence.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1.5</span><span class="math-callout__name">(Data Processing Inequality)</span></p>

Given two probability measures $p, q \in \mathcal{P}(\mathcal{X})$ and a Markov kernel $\mathcal{K} : \mathcal{X} \to \mathcal{P}(\mathcal{Y})$ that maps states $\boldsymbol{x} \in \mathcal{X}$ to probability measures on $\mathcal{Y}$, define $\bar{q}, \bar{p} \in \mathcal{P}(\mathcal{Y})$ as:

$$
\bar{q}(\boldsymbol{y}) := \int_{\mathcal{X}} q(\boldsymbol{x}) \mathcal{K}(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{x}, \quad \bar{p}(\boldsymbol{y}) := \int_{\mathcal{X}} p(\boldsymbol{x}) \mathcal{K}(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{x}
$$

Then, they satisfy $\text{KL}(\bar{p} \| \bar{q}) \le \text{KL}(p \| q)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Data Processing Inequality)</summary>

Define joint probability measures via the Markov kernel $\mathcal{K}$:

$$
P(d\boldsymbol{x}, d\boldsymbol{y}) := p(d\boldsymbol{x}) \mathcal{K}(\boldsymbol{x}, d\boldsymbol{y}), \quad Q(d\boldsymbol{x}, d\boldsymbol{y}) := q(d\boldsymbol{x}) \mathcal{K}(\boldsymbol{x}, d\boldsymbol{y})
$$

where $\bar{p}$ and $\bar{q}$ are the $\mathcal{Y}$-marginals of $P$ and $Q$. Applying the KL chain rule (Lemma 1.4):

$$
\text{KL}(P \| Q) = \text{KL}(\bar{p} \| \bar{q}) + \underbrace{\mathbb{E}_{\bar{p}}[\text{KL}(P_{\mathcal{X} \mid Y} \| Q_{\mathcal{X} \mid Y})]}_{\ge 0} \ge \text{KL}(\bar{p} \| \bar{q})
$$

Since $P$ and $Q$ share the same kernel $\mathcal{K}(\boldsymbol{x}, d\boldsymbol{y})$, the conditional law $\mathcal{K}$ cancels in the KL divergence:

$$
\text{KL}(P \| Q) = \int_{\mathcal{X} \times \mathcal{Y}} \log \frac{dp}{dq}(\boldsymbol{x}) \, p(d\boldsymbol{x}) \mathcal{K}(\boldsymbol{x}, d\boldsymbol{y}) = \text{KL}(p \| q)
$$

Combining: $\text{KL}(\bar{p} \| \bar{q}) \le \text{KL}(p \| q)$. $\square$
</details>
</div>

### 1.3 Entropic Optimal Transport Problem

Having defined the OMT problem as finding a transport plan between marginal distributions that minimizes a cost function, we now extend to the **entropic optimal transport** (EOT) problem, which moves closer to the Schroedinger bridge problem.

Like the OMT problem, the EOT problem seeks an optimal transport plan $\pi\_{0,T}^\star \in \Pi(\pi\_0, \pi\_T)$ between marginals $\pi\_0$ and $\pi\_T$. However, *optimality* is no longer determined solely by the transport cost $c(\boldsymbol{x}, \boldsymbol{y})$ but also an **entropy regularization term** $\text{KL}(\pi\_{0,T} \| q)$, where $q \in \mathcal{P}(\mathcal{X} \times \mathcal{Y})$ is a reference coupling measure on the product space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.6</span><span class="math-callout__name">(Entropic Optimal Transport (EOT) Problem)</span></p>

Consider two probability distributions $\pi\_0 \in \mathcal{P}(\mathcal{X})$ and $\pi\_T \in \mathcal{P}(\mathcal{Y})$ and a cost function $c(\boldsymbol{x}, \boldsymbol{y}) : \mathcal{X} \times \mathcal{Y} \to [0, \infty)$ that defines the cost of transporting mass from $\boldsymbol{x} \in \mathcal{X}$ to $\boldsymbol{y} \in \mathcal{Y}$. The **entropic optimal transport** (EOT) problem aims to find the optimal transport plan $\pi\_{0,T}^\star$ that minimizes:

$$
\inf_{\pi_{0,T} \in \Pi(\pi_0, \pi_T)} \left\lbrace \int_{\mathcal{X} \times \mathcal{Y}} c(\boldsymbol{x}, \boldsymbol{y}) d\pi_{0,T}(\boldsymbol{x}, \boldsymbol{y}) + \alpha \text{KL}(\pi_{0,T} \| q) \right\rbrace
$$

where $\text{KL}(\cdot \| \cdot)$ is the KL divergence (Definition 1.3) between a transport plan and a fixed probability measure $q \in \mathcal{P}(\mathcal{X} \times \mathcal{Y})$.

</div>

The KL divergence acts as a measure of distance from a pre-defined *reference* coupling, preventing the optimal transport coupling from diverging too far. The penalty is scaled by $\alpha \in \mathbb{R}$, where small values allow more divergence and large values penalize even small deviations.

**Reduction to KL minimization.** Define the cost functional with the reference measure set to $dq := d(\pi\_0 \otimes \pi\_T)$ as:

$$
\mathcal{F}(\pi_{0,T}) := \int_{\mathcal{X} \times \mathcal{Y}} c(\boldsymbol{x}, \boldsymbol{y}) d\pi_{0,T}(\boldsymbol{x}, \boldsymbol{y}) + \text{KL}(\pi_{0,T} \| \pi_0 \otimes \pi_T)
$$

We define a *tilted* reference measure:

$$
d\tilde{q} := \frac{e^{-c}}{\alpha} d(\pi_0 \otimes \pi_T), \quad \alpha := \int_{\mathcal{X} \times \mathcal{Y}} e^{-c} d(\pi_0 \otimes \pi_T)
$$

where $\alpha$ ensures $dq$ integrates to one. Expanding $\text{KL}(\pi\_{0,T} \| q)$:

$$
\text{KL}(\pi_{0,T} \| \tilde{q}) - \underbrace{\log \alpha}_{\text{constant}} = \text{KL}(\pi_{0,T} \| (\pi_0 \otimes \pi_T)) + \int_{\mathcal{X} \times \mathcal{Y}} c(\boldsymbol{x}, \boldsymbol{y}) d\pi_{0,T} =: \mathcal{F}(\pi_{0,T})
$$

Since $\log \alpha$ is a constant independent of $\pi\_{0,T}$, the Entropic OT Problem reduces to a KL minimization:

$$
\inf_{\pi_{0,T} \in \Pi(\pi_0, \pi_T)} \left\lbrace \int_{\mathcal{X} \times \mathcal{Y}} c(\boldsymbol{x}, \boldsymbol{y}) d\pi_{0,T}(\boldsymbol{x}, \boldsymbol{y}) + \text{KL}(\pi_{0,T} \| q) \right\rbrace = \inf_{\pi_{0,T} \in \Pi(\pi_0, \pi_T)} \text{KL}(\pi_{0,T} \| \tilde{q})
$$

We can interpret the **entropic OT problem** as a KL projection of a reference measure $d\tilde{q}$ onto the set of couplings with prescribed marginals $\Pi(\pi\_0, \pi\_T)$. This is exactly the variational structure underlying the static Schroedinger bridge (SB) problem.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivations for Entropy Regularization)</span></p>

*Why do we need entropy regularization?*

1. **Stochastic Coupling.** The entropy term penalizes large probabilities under the coupling $\pi\_{0,T}$. While OMT is susceptible to convergence on a singular plan concentrated on a lower-dimensional set, entropy regularization ensures the solution is a **stochastic coupling** in which mass is spread smoothly over the joint probability space $\mathcal{X} \times \mathcal{Y}$, rather than a deterministic map.

2. **Strict Convexity and Uniqueness.** Monge's OMT Problem is linear in $\pi\_{0,T}$, allowing multiple minimizers with the same total cost. However, since the entropy function $\pi\_{0,T} \mapsto \int \pi\_{0,T} \log \pi\_{0,T}$ is *convex* in $\pi\_{0,T}$, the entropic OT problem has a **unique minimizer** $\pi\_{0,T}^\star$ regardless of initialization.

3. **Generalization of Optimal Mass Transport.** As $\varepsilon \to 0$, the problem reduces to the optimal mass transport problem. As $\varepsilon \to \infty$, the cost function becomes negligible and the solution equals the reference coupling $q$. The entropic OT problem is thus a generalization of OMT with tunable entropic regularization.

</div>

### 1.4 Static Schroedinger Bridge Problem

In this section, we formally define the **static Schroedinger bridge (SB) problem**, which is closely related to the entropic OT problem discussed in Section 1.3. We introduce the notion of **Schroedinger potentials** $(\varphi, \hat{\varphi})$ that *uniquely* solve a pair of equations called the **Schroedinger system**, and simultaneously define the unique static Schroedinger bridge solution.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.7</span><span class="math-callout__name">(Static Schroedinger Bridge Problem)</span></p>

Given two marginal distribution constraints $\pi\_0 \in \mathcal{P}(\mathcal{X})$ and $\pi\_T \in \mathcal{P}(\mathcal{Y})$, define the set of all couplings $\Pi(\pi\_0, \pi\_T) \subset \mathcal{P}(\mathcal{X} \times \mathcal{Y})$ with $\pi\_0$ and $\pi\_T$ as its $\mathcal{X}$- and $\mathcal{Y}$-marginals. Given a reference measure $q \sim \pi\_0 \otimes \pi\_T$, the **static Schroedinger bridge** (SB) problem is:

$$
\pi_{0,T}^\star = \underset{\pi_{0,T} \in \Pi(\pi_0, \pi_T)}{\arg\min}\; \text{KL}(\pi_{0,T} \| q)
$$

where the minimizer $\pi\_{0,T}^\star$ is *unique* and is called the **static Schroedinger bridge** between $\pi\_0$ and $\pi\_T$.

</div>

When the reference coupling takes the form $q(\boldsymbol{x}, \boldsymbol{y}) := \frac{e^{-c(\boldsymbol{x}, \boldsymbol{y})}}{\alpha} (\pi\_0 \otimes \pi\_T)(\boldsymbol{x}, \boldsymbol{y})$, the static SB problem coincides with the entropic optimal transport (EOT) problem:

$$
\pi_{0,T}^\star = \underset{\pi_{0,T} \in \Pi(\pi_0, \pi_T)}{\arg\min} \left\lbrace \int_{\mathcal{X} \times \mathcal{Y}} c(\boldsymbol{x}, \boldsymbol{y}) d\pi_{0,T}(\boldsymbol{x}, \boldsymbol{y}) + \alpha \text{KL}(\pi_{0,T} \| q) \right\rbrace
$$

Although the static SB problem is formulated as an optimization over couplings, its solution admits a simple multiplicative structure: the optimal coupling can be written as a reweighted version of the reference measure using two functions known as the **Schroedinger potentials** $(\varphi, \hat{\varphi})$ which together solve the **Schroedinger system**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.8</span><span class="math-callout__name">(Schroedinger Potentials)</span></p>

Consider a reference measure $q \ll \pi\_0 \otimes \pi\_T$, which implies the Radon--Nikodym derivative $\frac{dq}{d(\pi\_0 \otimes \pi\_T)}$ is well-defined, and given some cost function $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$:

$$
\frac{dq}{d(\pi_0 \otimes \pi_T)} = e^{-c(\boldsymbol{x}, \boldsymbol{y})}
$$

Then given two functions $\varphi : \mathcal{X} \to \mathbb{R}$ and $\hat{\varphi} : \mathcal{Y} \to \mathbb{R}$, called the **Schroedinger potentials**, we define the **Schroedinger system** as:

$$
\begin{cases} \varphi(\boldsymbol{x}) = -\log \int_{\mathcal{Y}} e^{\hat{\varphi}(\boldsymbol{y}) - c(\boldsymbol{x}, \boldsymbol{y})} \pi_T(d\boldsymbol{y}) \\[4pt] \hat{\varphi}(\boldsymbol{y}) = -\log \int_{\mathcal{X}} e^{\varphi(\boldsymbol{x}) - c(\boldsymbol{x}, \boldsymbol{y})} \pi_0(d\boldsymbol{x}) \end{cases}
$$

where the solution $\tilde{\pi}\_{0,T}$ solves the static Schroedinger bridge problem (Definition 1.7) and satisfies:

$$
d\tilde{\pi}_{0,T}(d\boldsymbol{x}, d\boldsymbol{y}) = e^{\varphi(\boldsymbol{x}) + \hat{\varphi}(\boldsymbol{y}) - c(\boldsymbol{x}, \boldsymbol{y})} d(\pi_0 \otimes \pi_T)
$$

with marginal constraints $\tilde{\pi}\_0(d\boldsymbol{x}) = \pi\_0(d\boldsymbol{x})$ and $\hat{\pi}\_T(d\boldsymbol{y}) = \pi\_T(d\boldsymbol{y})$. Furthermore, the potentials $(\varphi, \hat{\varphi})$ are **unique** up to an additive constant.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Schroedinger Potentials)</summary>

**Step 1: Derivation of Schroedinger System from Static SB Problem.** Recall the static SB problem from Definition 1.7:

$$
\min_{\tilde{\pi}_{0,T} \in \Pi(\pi_0, \pi_T)} \text{KL}(\tilde{\pi}_{0,T} \| q) = \min_{\tilde{\pi}_{0,T} \in \Pi(\pi_0, \pi_T)} \int_{\mathcal{X} \times \mathcal{Y}} \log \frac{d\tilde{\pi}_{0,T}}{dq} \, d\tilde{\pi}_{0,T}(\boldsymbol{x}, d\boldsymbol{y}), \quad dq := e^{-c(\boldsymbol{x}, \boldsymbol{y})} d(\pi_0 \otimes \pi_T)
$$

Rewrite using Lagrangian multipliers $\varphi(\boldsymbol{x})$ and $\hat{\varphi}(\boldsymbol{y})$ to enforce marginal constraints:

$$
\mathcal{L}(\tilde{\pi}_{0,T}) := \int_{\mathcal{X} \times \mathcal{Y}} \log\!\left(\frac{d\tilde{\pi}_{0,T}}{dq}\right) d\tilde{\pi}_{0,T} + \int_{\mathcal{X}} \varphi(\boldsymbol{x})(\pi_0 - \tilde{\pi}_0)(d\boldsymbol{x}) + \int_{\mathcal{Y}} \hat{\varphi}(\boldsymbol{y})(\pi_T - \hat{\pi}_T)(d\boldsymbol{y})
$$

Taking the functional derivative with respect to $\tilde{\pi}\_{0,T}$ and setting to zero:

$$
\frac{\delta}{\delta \tilde{\pi}_{0,T}} \mathcal{L}(\tilde{\pi}_{0,T}) = \log\!\left(\frac{d\tilde{\pi}_{0,T}}{dq}\right) + 1 - \varphi(\boldsymbol{x}) - \hat{\varphi}(\boldsymbol{y}) = 0
$$

$$
\implies \frac{d\tilde{\pi}_{0,T}}{dq} = e^{-1} e^{\varphi(\boldsymbol{x}) + \hat{\varphi}(\boldsymbol{y})}
$$

$$
\implies d\tilde{\pi}_{0,T}(d\boldsymbol{x}, d\boldsymbol{y}) = e^{\varphi(\boldsymbol{x}) + \hat{\varphi}(\boldsymbol{y}) - c(\boldsymbol{x}, \boldsymbol{y})} \pi_0(d\boldsymbol{x}) \pi_T(d\boldsymbol{y})
$$

where the constant $e^{-1}$ is absorbed into the Lagrange multipliers. Computing each marginal by integration recovers the Schroedinger system equations.

**Step 2: Proving Uniqueness of Schroedinger Potentials.** From Step 1, we can write 

$$\log\!\left(\frac{d\tilde{\pi}_{0,T}}{dq}\right) = \varphi(\boldsymbol{x}) + \hat{\varphi}(\boldsymbol{y})$$

If another pair $\varphi', \hat{\varphi}'$ also satisfies this, then $\varphi(\boldsymbol{x}) - \varphi'(\boldsymbol{x}) = \hat{\varphi}'(\boldsymbol{y}) - \hat{\varphi}(\boldsymbol{y})$. Since the left side depends only on $\boldsymbol{x}$ and the right side only on $\boldsymbol{y}$, both must equal a constant $a$, giving $\varphi(\boldsymbol{x}) = \varphi'(\boldsymbol{x}) + a$ and $\hat{\varphi}(\boldsymbol{y}) = \hat{\varphi}'(\boldsymbol{y}) - a$. Hence $(\varphi, \hat{\varphi})$ are unique up to an additive constant $a \in \mathbb{R}$. $\square$
</details>
</div>

Using the Schroedinger potentials $(\varphi, \hat{\varphi})$, we can show that they define the **unique optimal coupling** $\pi\_{0,T}^\star$ that solves the Static SB Problem.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.9</span><span class="math-callout__name">(Solution to Static SB Problem)</span></p>

Given two marginals $\pi\_0 \in \mathcal{P}(\mathcal{X})$ and $\pi\_T \in \mathcal{P}(\mathcal{Y})$ and a reference coupling $q \sim \pi\_0 \otimes \pi\_T$, assume the set of finite entropy couplings $\Pi(\pi\_0, \pi\_T) \neq \varnothing$. Let $\tilde{\pi}\_{0,T}$ be a coupling that satisfies:

$$
\log\!\left(\frac{d\tilde{\pi}_{0,T}}{dq}\right) = \varphi \oplus \hat{\varphi}, \quad q\text{-a.s.}
$$

for measurable functions $\varphi : \mathcal{X} \to \mathbb{R}$ and $\hat{\varphi} : \mathcal{Y} \to \mathbb{R}$. Then $\tilde{\pi}\_{0,T} = \pi\_{0,T}^\star$ solves the Static SB Problem and yields a **constant map**:

$$
\pi_{0,T} \mapsto \mathbb{E}_{\pi_{0,T}}\!\left[\log \frac{d\pi_{0,T}^\star}{dq}\right], \quad \forall \pi_{0,T} \in \Pi(\pi_0, \pi_T) \cup \lbrace \pi_{0,T}^\star \rbrace
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Solution to Static SB Problem)</summary>

**Step 1: Prove the Constant Map.** Since $\log\!\left(\frac{d\pi\_{0,T}^\star}{dq}\right) = \varphi(\boldsymbol{x}) + \hat{\varphi}(\boldsymbol{y})$ is separable:

$$
\mathbb{E}_{\pi_{0,T}}\!\left[\log \frac{d\pi_{0,T}^\star}{dq}\right] = \int_{\mathcal{X} \times \mathcal{Y}} \varphi(\boldsymbol{x}) \pi_{0,T}(d\boldsymbol{x}, d\boldsymbol{y}) + \int_{\mathcal{X} \times \mathcal{Y}} \hat{\varphi}(\boldsymbol{y}) \pi_{0,T}(d\boldsymbol{x}, d\boldsymbol{y})
$$

$$
= \underbrace{\int_{\mathcal{X}} \varphi(\boldsymbol{x}) \pi_0(d\boldsymbol{x}) + \int_{\mathcal{Y}} \hat{\varphi}(\boldsymbol{y}) \pi_T(d\boldsymbol{y})}_{\text{independent of coupling } \pi_{0,T}}
$$

since any $\pi\_{0,T} \in \Pi(\pi\_0, \pi\_T)$ has the same marginals $\pi\_0$ and $\pi\_T$.

**Step 2: Prove Optimality.** For any $\pi\_{0,T} \in \Pi(\pi\_0, \pi\_T)$:

$$
\text{KL}(\pi_{0,T} \| q) = \mathbb{E}_{\pi_{0,T}}\!\left[\log \frac{d\pi_{0,T}}{d\pi_{0,T}^\star}\right] + \underbrace{\mathbb{E}_{\pi_{0,T}}\!\left[\log \frac{d\pi_{0,T}^\star}{dq}\right]}_{\text{constant}} = \text{KL}(\pi_{0,T} \| \pi_{0,T}^\star) + C
$$

Since KL divergence is non-negative with equality iff $\pi\_{0,T} = \pi\_{0,T}^\star$, it follows that $\pi\_{0,T}^\star$ is the unique solution. $\square$
</details>
</div>

Since the Schroedinger potentials $(\varphi, \hat{\varphi})$ depend only on the marginal constraints and act as Lagrange multipliers, this naturally leads to an alternative **dual formulation of the static SB problem**, which aims to maximize the potentials to enforce the marginal constraints.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.10</span><span class="math-callout__name">(Dual Formulation of Static SB Problem)</span></p>

The equivalent **dual formulation** of the Static SB Problem is:

$$
\underbrace{\inf_{\pi_{0,T} \in \Pi(\pi_0, \pi_T)} \text{KL}(\pi_{0,T} \| q)}_{\text{primal problem}} = \underbrace{\sup_{\varphi, \hat{\varphi}} \left\lbrace \int_{\mathcal{X}} \varphi \, d\pi_0 + \int_{\mathcal{Y}} \hat{\varphi} \, d\pi_T - \int_{\mathcal{X} \times \mathcal{Y}} e^{\varphi \oplus \hat{\varphi}} dq + 1 \right\rbrace}_{\text{dual problem}}
$$

where the **supremum** is achieved by the Schroedinger potentials $(\varphi^\star, \hat{\varphi}^\star)$ which define the associated Schroedinger bridge coupling $\pi\_{0,T}^\star$ and satisfy:

$$
\text{KL}(\pi_{0,T}^\star \| q) = \int_{\mathcal{X}} \varphi^\star d\pi_0 + \int_{\mathcal{Y}} \hat{\varphi}^\star d\pi_T, \quad \frac{d\pi_{0,T}^\star}{dq} = e^{\varphi^\star \oplus \hat{\varphi}^\star} \quad q\text{-a.s.}
$$

where $\varphi^\star \in L^1(\pi\_0)$, $\hat{\varphi}^\star \in L^1(\pi\_T)$ and $\varphi^\star \oplus \hat{\varphi}^\star$ is unique.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch (Dual Formulation)</summary>

**Step 1: Weak Duality.** For any measurable $\varphi \in L^1(\pi\_0)$, $\hat{\varphi} \in L^1(\pi\_T)$ and coupling $\pi\_{0,T} \in \Pi(\pi\_0, \pi\_T)$:

$$
\text{KL}(\pi_{0,T} \| q) \ge \int_{\mathcal{X}} \varphi \, d\pi_0 + \int_{\mathcal{Y}} \hat{\varphi} \, d\pi_T - \int_{\mathcal{X} \times \mathcal{Y}} e^{\varphi \oplus \hat{\varphi}} dq + 1
$$

This follows from **Fenchel's inequality**: for $f(\alpha) = \alpha \log \alpha - \alpha$ and any $\alpha, \beta$, we have $f(\alpha) \ge \alpha\beta - f^\star(\beta)$ where $f^\star(\beta) = e^\beta$. Applying this pointwise with $\alpha = \frac{d\pi\_{0,T}}{dq}$ and $\beta = \varphi(\boldsymbol{x}) + \hat{\varphi}(\boldsymbol{y})$ and integrating against $dq$ yields the weak duality.

**Step 2: Strong Duality.** We recall the optimal Schroedinger potentials $(\varphi^\star, \hat{\varphi}^\star)$ and the optimal coupling $\pi\_{0,T}^\star \in \Pi(\pi\_0, \pi\_T)$ satisfying $\frac{d\pi\_{0,T}^\star}{dq} = e^{\varphi^\star \oplus \hat{\varphi}^\star}$ $q$-a.s. Taking the infimum on the left and supremum on the right of the Weak Duality preserves the inequality. For the **left-hand side**, applying the KL divergence for $\pi\_{0,T}^\star$:

$$
\text{KL}(\pi_{0,T}^\star \| q) = \int_{\mathcal{X} \times \mathcal{Y}} (\varphi^\star \oplus \hat{\varphi}^\star) d\pi_{0,T}^\star = \int_{\mathcal{X}} \varphi^\star d\pi_0 + \int_{\mathcal{Y}} \hat{\varphi}^\star d\pi_T
$$

For the **right-hand side**, using $d\pi\_{0,T}^\star = e^{\varphi^\star \oplus \hat{\varphi}^\star} dq$:

$$
G(\varphi^\star, \hat{\varphi}^\star) = \int_{\mathcal{X}} \varphi^\star d\pi_0 + \int_{\mathcal{Y}} \hat{\varphi}^\star d\pi_T - \underbrace{\int_{\mathcal{X} \times \mathcal{Y}} d\pi_{0,T}^\star}_{=1} + 1 = \int_{\mathcal{X}} \varphi^\star d\pi_0 + \int_{\mathcal{Y}} \hat{\varphi}^\star d\pi_T
$$

Since both sides are equal for the optimal $\varphi^\star, \hat{\varphi}^\star, \pi\_{0,T}^\star$, strong duality holds. Since $-e^{\varphi \oplus \hat{\varphi}}$ is strictly concave in $(\varphi \oplus \hat{\varphi})$, the sum $(\varphi^\star \oplus \hat{\varphi}^\star)$ is the *unique* maximizer, which also implies that $\varphi$ and $\hat{\varphi}$ are *unique up to a constant*. $\square$
</details>
</div>

The dual formulation reformulates the static SB problem from optimizing a coupling $\pi\_{0,T}^\star$ on the high-dimensional product space $\mathcal{X} \times \mathcal{Y}$ to determining two scalar SB potential functions $(\varphi, \hat{\varphi})$ defined on the marginals $\pi\_0$ and $\pi\_T$ which *uniquely* characterize the optimal bridge.

### 1.5 Sinkhorn's Algorithm

We now introduce the classical algorithm used to solve the static Schroedinger system, known as **Sinkhorn's algorithm** in optimal transport, or the **Iterative Proportional Fitting** (IPF) procedure in statistics.

Recall that the Schroedinger system consists of two equations for the pair of potentials $(\varphi, \hat{\varphi})$:

$$
\varphi(\boldsymbol{x}) = -\log \int_{\mathcal{Y}} e^{\hat{\varphi}(\boldsymbol{y}) - c(\boldsymbol{x}, \boldsymbol{y})} \pi_T(d\boldsymbol{y}) \qquad \text{(First Potential)}
$$

$$
\hat{\varphi}(\boldsymbol{y}) = -\log \int_{\mathcal{X}} e^{\varphi(\boldsymbol{x}) - c(\boldsymbol{x}, \boldsymbol{y})} \pi_0(d\boldsymbol{x}) \qquad \text{(Second Potential)}
$$

Optimizing both $\varphi$ and $\hat{\varphi}$ simultaneously would lead to a mismatch, as they depend on each other. Therefore, we consider an **alternating optimization scheme** that optimizes one potential with the other held fixed. This is exactly the intuition behind **Sinkhorn's algorithm**.

The algorithm starts by initializing $\varphi = \varphi\_0$ at some value (e.g., $\varphi\_0 := 0$) and defines an alternating optimization sequence $\lbrace \varphi\_n, \hat{\varphi}\_n \rbrace\_{n \ge 0}$. We can also consider this as maximizing the **dual problem** defined in Theorem 1.10 with the objective:

$$
G(\varphi, \hat{\varphi}) := \int_{\mathcal{X}} \varphi \, d\pi_0 + \int_{\mathcal{Y}} \hat{\varphi} \, d\pi_T - \int_{\mathcal{X} \times \mathcal{Y}} e^{\varphi \oplus \hat{\varphi}} dq + 1
$$

The alternating optimization sequence with $\varphi\_0 := 0$:

1. Solve $\hat{\varphi}\_n$ using the Second Potential with $\varphi := \varphi\_n$. Equivalently, $\hat{\varphi}\_n := \arg\max G(\varphi\_n, \cdot)$.
2. Solve $\varphi\_{n+1}$ using the First Potential with $\hat{\varphi} := \hat{\varphi}\_n$. Equivalently, $\varphi\_{n+1} := \arg\max G(\cdot, \hat{\varphi}\_n)$.

Since the Dual Objective is **strictly concave** with respect to both $\varphi$ and $\hat{\varphi}$, each iteration strictly increases the objective $G(\varphi\_n, \hat{\varphi}\_n) < G(\varphi\_{n+1}, \hat{\varphi}\_n) < G(\varphi\_{n+1}, \hat{\varphi}\_{n+1})$, unless the optimal pair $(\varphi^\star, \hat{\varphi}^\star)$ is reached. The coupling at each iteration is:

$$
d\pi_{0,T}(\varphi, \hat{\varphi}) := e^{\varphi \oplus \hat{\varphi}} dq = e^{\varphi \oplus \hat{\varphi} - c} d(\pi_0 \otimes \pi_T)
$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1.11</span><span class="math-callout__name">(Properties of Sinkhorn's Algorithm)</span></p>

The sequence of potentials $\lbrace \varphi\_n, \hat{\varphi}\_n \rbrace\_{n \ge 0}$, where $\varphi\_n \in L^1(\pi\_0)$ and $\hat{\varphi}\_n \in L^1(\pi\_T)$ are integrable, and coupled densities $\lbrace \pi\_{0,T}^{2n}, \pi\_{0,T}^{2n-1} \rbrace\_{n \ge 0}$ defined by the Sinkhorn iterations satisfy:

**(i)** Each KL step equals a difference of potentials:

$$
\text{KL}\!\left(\pi_{0,T}^{2n} \| \pi_{0,T}^{2n-1}\right) = \int_{\mathcal{Y}} (\hat{\varphi}_n - \hat{\varphi}_{n-1}) \pi_T(d\boldsymbol{y}), \quad \text{KL}\!\left(\pi_{0,T}^{2n+1} \| \pi_{0,T}^{2n}\right) = \int_{\mathcal{X}} (\varphi_{n+1} - \varphi_n) \pi_0(d\boldsymbol{x})
$$

**(ii)** The total dual potential equals the total accumulated KL cost:

$$
\pi_T(\hat{\varphi}_n) = \sum_{k=0}^{n} \text{KL}\!\left(\pi_{0,T}^{(2k)} \| \pi_{0,T}^{(2k-1)}\right), \quad \pi_0(\varphi_n) = \sum_{k=0}^{n-1} \text{KL}\!\left(\pi_{0,T}^{(2k+1)} \| \pi_{0,T}^{(2k)}\right)
$$

where $\pi\_0(\varphi\_n)$ and $\pi\_T(\hat{\varphi}\_n)$ are non-negative and increasing.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Properties of Sinkhorn's Algorithm)</summary>

**Step 1: Proof of Property (i).** Starting from the KL divergence definition:

$$
\text{KL}\!\left(\pi_{0,T}^{2n} \| \pi_{0,T}^{2n-1}\right) = \int_{\mathcal{X} \times \mathcal{Y}} \log\!\left(\frac{d\pi_{0,T}^{2n}}{d\pi_{0,T}^{2n-1}}\right) d\pi_{0,T}^{2n}
$$

From the density ratio of updates: $\frac{d\pi\_{0,T}^{2n}}{d\pi\_{0,T}^{2n-1}} = \frac{e^{\hat{\varphi}\_n(\boldsymbol{y})}}{e^{\hat{\varphi}\_{n-1}(\boldsymbol{y})}} = e^{\hat{\varphi}\_n(\boldsymbol{y}) - \hat{\varphi}\_{n-1}(\boldsymbol{y})}$, so:

$$
\text{KL}\!\left(\pi_{0,T}^{2n} \| \pi_{0,T}^{2n-1}\right) = \int_{\mathcal{X} \times \mathcal{Y}} (\hat{\varphi}_n(\boldsymbol{y}) - \hat{\varphi}_{n-1}(\boldsymbol{y})) \, d\pi_{0,T}^{2n} = \int_{\mathcal{Y}} (\hat{\varphi}_n - \hat{\varphi}_{n-1}) \pi_T(d\boldsymbol{y})
$$

Similarly, $\frac{d\pi\_{0,T}^{2n+1}}{d\pi\_{0,T}^{2n}} = e^{\varphi\_{n+1}(\boldsymbol{x}) - \varphi\_n(\boldsymbol{x})}$, giving the analogous result for $\varphi$.

**Step 2: Proof of Property (ii).** Summing the KL divergence expressions from (i) and applying the **telescoping trick**:

$$
\sum_{k=0}^{n} \int_{\mathcal{Y}} (\hat{\varphi}_k - \hat{\varphi}_{k-1}) \pi_T = \int_{\mathcal{Y}} \hat{\varphi}_n \pi_T =: \pi_T(\hat{\varphi}_n)
$$

using initialization $\hat{\varphi}\_{-1} = 0$ and $\varphi\_0 = 0$. $\square$
</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.12</span><span class="math-callout__name">(Marginal Convergence of Sinkhorn Iterations)</span></p>

Each iteration of Sinkhorn's algorithm results in a decrease in KL divergence, such that for all $n \ge -1$, the KL with the optimal coupling $\pi\_{0,T}^\star$ is decreasing in $n$:

$$
\text{KL}\!\left(\pi_{0,T}^\star \| \pi_{0,T}^{(n)}\right) = \text{KL}\!\left(\pi_{0,T}^\star \| q\right) - \sum_{k=0}^{n} \text{KL}\!\left(\pi_{0,T}^{(k)} \| \pi_{0,T}^{(k-1)}\right)
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Marginal Convergence)</summary>

Expanding $\text{KL}(\pi\_{0,T}^\star \| \pi\_{0,T}^{2n})$ using the chain of density ratios:

$$
\text{KL}(\pi_{0,T}^\star \| \pi_{0,T}^{2n}) = \mathbb{E}_{\pi_{0,T}^\star}\!\left[\log \frac{d\pi_{0,T}^\star}{d\pi_{0,T}^{2n}}\right] = \mathbb{E}_{\pi_{0,T}^\star}\!\left[\log \frac{d\pi_{0,T}^\star}{dq} \cdot \frac{dq}{d\pi_{0,T}^{2n}}\right]
$$

$$
= \text{KL}(\pi_{0,T}^\star \| q) - \mathbb{E}_{\pi_{0,T}^\star}\!\left[\hat{\varphi}_n + \varphi_n\right] = \text{KL}(\pi_{0,T}^\star \| q) - (\pi_T(\hat{\varphi}_n) + \pi_0(\varphi_n))
$$

Substituting Lemma 1.11 (ii): $\text{KL}(\pi\_{0,T}^\star \| \pi\_{0,T}^{(n)}) = \text{KL}(\pi\_{0,T}^\star \| q) - \sum\_{k=0}^{n} \text{KL}(\pi\_{0,T}^{(k)} \| \pi\_{0,T}^{(k-1)})$.

This proves the KL divergence with the optimal coupling is decreasing. $\square$
</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1.13</span><span class="math-callout__name">(Convergence of Sinkhorn's Algorithm)</span></p>

The KL divergence between the marginals satisfies the inequality:

$$
\text{KL}(\pi_0^{(k)} \| \pi_0) + \text{KL}(\pi_T^{(k)} \| \pi_T) \le \text{KL}(\pi_{0,T}^{(k)} \| \pi_{0,T}^{(k-1)})
$$

and the sum of the right-hand side for $n$ iterations is bounded:

$$
\sum_{k=1}^{n} \text{KL}\!\left(\pi_{0,T}^{(k)} \| \pi_{0,T}^{(k-1)}\right) \le \text{KL}\!\left(\pi_{0,T}^\star \| q\right)
$$

which implies $\text{KL}(\pi\_{0,T}^{(k)} \| \pi\_{0,T}^{(k-1)}) \to 0$, $\text{KL}(\pi\_0^{(k)} \| \pi\_0) \to 0$, and $\text{KL}(\pi\_T^{(k)} \| \pi\_T) \to 0$ as $n \to \infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Convergence of Sinkhorn's Algorithm)</summary>

The marginals are correct for $\pi\_0$ on odd iterations $k \ge 1$ ($\pi\_0^{(k)} = \pi\_0$) and for $\pi\_T$ on even iterations $k \ge 2$ ($\pi\_T^{(k)} = \pi\_T$). Rewriting the sum of marginal KL divergences:

$$
\text{KL}(\pi_0^{(2k)} \| \pi_0) + \underbrace{\text{KL}(\pi_T^{(2k)} \| \pi_T)}_{=0} = \text{KL}(\pi_0^{(2k)} \| \pi_0) \le \text{KL}(\pi_{0,T}^{(2k)} \| \pi_{0,T}^{(2k-1)})
$$

where the inequality follows from the **data processing inequality** (Lemma 1.5). From Proposition 1.12, we can rearrange to get an upper bound on the sum of KL divergences for $k \ge 1$:

$$
\sum_{k=1}^{n} \text{KL}\!\left(\pi_{0,T}^{(k)} \| \pi_{0,T}^{(k-1)}\right) \le \text{KL}\!\left(\pi_{0,T}^\star \| q\right) - \underbrace{\text{KL}\!\left(\pi_{0,T}^{(0)} \| q\right)}_{\ge 0}
$$

Therefore, for $n \to \infty$, the series converges and $\text{KL}(\pi\_0^{(n)} \| \pi\_0) \to 0$ and $\text{KL}(\pi\_T^{(n)} \| \pi\_T) \to 0$. Applying Pinsker's inequality ($\lVert p - q \rVert\_{\text{TV}} \le \sqrt{2\text{KL}(p \| q)}$), we conclude $\pi\_0^{(n)} \to \pi\_0$ and $\pi\_T^{(n)} \to \pi\_T$ in total variation. $\square$
</details>
</div>

While Proposition 1.12 proves that the *marginals* converge, it **does not directly imply strong convergence** to the optimal coupling $\pi\_{0,T}^\star$ or the Schroedinger potentials $(\varphi, \hat{\varphi}) \to (\varphi^\star, \hat{\varphi}^\star)$. Strong convergence requires additional conditions on $c$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.14</span><span class="math-callout__name">(Strong Convergence of Sinkhorn's Algorithm)</span></p>

Given a cost function $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ that is bounded from below and satisfies exponential integrability:

$$
\exists r > 1, \quad \int e^{rc(\boldsymbol{x}, \boldsymbol{y})} d(\pi_0 \otimes \pi_T) < \infty
$$

Then, the Sinkhorn iterates converge to the true Schroedinger potentials $\varphi\_n \to \varphi^\star$ and $\hat{\varphi}\_n \to \hat{\varphi}^\star$, and the induced couplings converge to the optimal Schroedinger bridge, $\text{KL}(\pi\_{0,T}^\star \| \pi\_{0,T}^{(n)}) \to 0$ and $\pi\_{0,T}^{(n)} \to \pi\_{0,T}^\star$ in variation.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch (Strong Convergence)</summary>

Under the Integrability Condition, the Sinkhorn iterates remain uniformly integrable and cannot develop singular behavior. Since the iterates are uniformly integrable, functional analysis establishes that there exists a subsequence where $e^{\varphi\_n}$ converges weakly in $L^1$, which passes the limit and forces convergence of $\hat{\varphi}\_n \to \hat{\varphi}^\star$ under the coupled update condition:

$$
\int_{\mathcal{X}} e^{\varphi_n(\boldsymbol{x})} e^{-c(\boldsymbol{x}, \boldsymbol{y})} d\pi_0(\boldsymbol{x}) \to \int_{\mathcal{X}} e^{\varphi^\star(\boldsymbol{x})} e^{-c(\boldsymbol{x}, \boldsymbol{y})} d\pi_0(\boldsymbol{x})
$$

$$
\implies -\log \lim_{n \to \infty} \int_{\mathcal{X}} e^{\varphi_n(\boldsymbol{x})} e^{-c(\boldsymbol{x}, \boldsymbol{y})} d\pi_0(\boldsymbol{x}) = -\log \int_{\mathcal{X}} e^{\varphi^\star(\boldsymbol{x})} e^{-c(\boldsymbol{x}, \boldsymbol{y})} d\pi_0(\boldsymbol{x}) =: \hat{\varphi}^\star(\boldsymbol{y})
$$

Since $(\varphi \oplus \hat{\varphi})$ is unique (Theorem 1.10), all subsequences converge to the same limit, giving full sequence convergence in KL and total variation to the Schroedinger bridge. $\square$
</details>
</div>

### 1.6 Closing Remarks for Section 1

In this section, we traced the origins of the Schroedinger bridge problem to the classical **optimal mass transport** (OMT) problem. Motivated by the non-uniqueness and instability of OMT solutions, we introduced **entropic optimal transport** (EOT), where an entropy-regularization term measuring deviation from a reference coupling is added to the transport objective. By reparameterizing the reference coupling, we showed that the EOT problem is equivalent to the **static Schroedinger bridge** (SB) problem --- finding a coupling closest in relative entropy to a reference endpoint law while satisfying prescribed marginal distributions. Finally, we introduced the classical **Sinkhorn algorithm**, an efficient iterative scheme that alternates between solving the two Schroedinger potential equations and provably converges to the optimal SB solution.

While the static formulation provides the foundation for understanding how probability mass can be optimally transported between distributions, its connection to modern generative modeling emerges through the **dynamic formulation** of the Schroedinger bridge problem. The dynamic viewpoint lifts the static coupling problem to the space of *stochastic path measures*, describing how probability flows through time under controlled stochastic dynamics.

## 2. The Dynamic Schroedinger Bridge Problem

Having established the connection between optimal transport and entropy-regularized couplings, we now turn to the *dynamic* formulation of the Schroedinger bridge (SB) problem. While optimal transport focuses on static couplings between distributions, the dynamic SB lifts the problem to the level of **stochastic processes**. Instead of directly transporting mass, we seek the most likely evolution of a stochastic system that transforms an initial distribution into a target distribution over time. In this section, we formalize this dynamic viewpoint and develop the stochastic calculus tools, including path measures, Ito processes, and change-of-measure techniques, necessary to analyze and solve the dynamic SB problem.

### 2.1 Dynamic Optimal Transport Problem

Just like the static Schroedinger bridge problem traces back to Monge--Kantorovich optimal mass transport, we begin the dynamic discussion with the **Benamou--Brenier (dynamic) optimal transport problem**. The **key idea** that differentiates the *dynamic* from the *static* problem is transporting mass with a *continuous flow over a time interval* rather than static couplings between marginals.

Given an initial distribution $\pi\_0 \in \mathcal{P}(\mathbb{R}^d)$ at time $t = 0$ and a target distribution $\pi\_T \in \mathcal{P}(\mathbb{R}^d)$ at time $t = T$, the dynamic formulation aims to determine the continuous-time evolution of a marginal probability density $p\_t \in \mathcal{P}(\mathbb{R}^d)$ that evolves mass from $\pi\_0 \to \pi\_T$ over $t \in [0, T]$.

Recall that Kantorovich's OMT Problem aims to minimize the cost of transporting states $\boldsymbol{x}\_0 \sim \pi\_0$ to $\boldsymbol{x}\_T \sim \pi\_T$. With the quadratic cost $c(\boldsymbol{x}\_0, \boldsymbol{x}\_T) := \lVert \boldsymbol{x}\_T - \boldsymbol{x}\_0 \rVert^2$, the quadratic transport cost from $\boldsymbol{x}\_0 \to \boldsymbol{x}\_T$ is equivalent to the integrating energy-minimizing velocity field $\boldsymbol{v}\_t(\boldsymbol{x}) := \frac{d}{dt}\boldsymbol{x}\_t$ over $t \in [0, T]$:

$$
\lVert \boldsymbol{x}_T - \boldsymbol{x}_0 \rVert^2 = \inf_{\boldsymbol{x}_t : \boldsymbol{x}_0 \to \boldsymbol{x}_T} \int_0^T \lVert \boldsymbol{v}_t(\boldsymbol{x}) \rVert^2 dt
$$

We can reframe the static OT problem as minimizing the **total kinetic energy** of transporting particles from $\pi\_0$, where particles at location $\boldsymbol{x}$ move with velocity $\boldsymbol{v}\_t(\boldsymbol{x})$, weighted by the probability mass $p\_t(\boldsymbol{x})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.1</span><span class="math-callout__name">(Dynamic Optimal Transport (OT) Problem)</span></p>

Given two marginal constraints $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$, the **dynamic optimal transport (OT) problem** aims to find the optimal probability flow $p\_t^\star : \mathbb{R}^d \times [0, T] \to \mathcal{P}(\mathbb{R}^d)$ and velocity field $\boldsymbol{v}\_t^\star : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ for $t \in [0, T]$ that minimizes:

$$
\inf_{(p_t, \boldsymbol{v}_t)} \left\lbrace \int_0^T \int_{\mathbb{R}^d} \lVert \boldsymbol{v}_t(\boldsymbol{x}) \rVert^2 p_t(\boldsymbol{x}) d\boldsymbol{x} dt \right\rbrace \quad \text{s.t.} \quad \begin{cases} \partial_t p_t + \nabla \cdot (p_t \boldsymbol{v}_t) = 0 \\ p_0 = \pi_0, \quad p_T = \pi_T \end{cases}
$$

where the continuity equation $\partial\_t p\_t + \nabla \cdot (p\_t \boldsymbol{v}\_t) = 0$ enforces conservation of probability mass over the continuous flow.

</div>

The solution to the Dynamic OT Problem is a time-interpolation of the **optimal transport map** $M^\star$:

$$
M_t^\star(\boldsymbol{x}_0) = (1 - t)\boldsymbol{x}_0 + t M^\star(\boldsymbol{x}_0), \quad \boldsymbol{x}_0 \sim \pi_0, \quad t \in [0, T]
$$

which yields the optimal marginal density $p\_t^\star$ as the pushforward of $\pi\_0$ via the transport map at time $t$: $p\_t^\star = (M\_t^\star)\_\# \pi\_0$.

This Benamou--Brenier formulation characterizes optimal transport as kinetic energy minimization over *straight, deterministic* flows satisfying the continuity equation. However, most real-world systems do not naturally evolve in straight lines, but rather stochastic paths. This naturally leads to the **dynamic Schroedinger bridge problem**, which considers the optimal transport problem where the underlying dynamics are *stochastic* rather than deterministic.

### 2.2 Dynamic Schroedinger Bridge Problem

While dynamic OT characterizes the most efficient deterministic flow, many real-world systems evolve under intrinsic stochasticity. This motivates a **stochastic generalization of dynamic OT**, leading to the **dynamic Schroedinger bridge (SB) problem**. Instead of searching over deterministic velocity fields, the SB formulation asks: *Among all stochastic evolutions that transform an initial distribution into a target distribution over a time horizon $[0,T]$, which one is **most likely** relative to a given reference dynamics?*

To answer this, the dynamic SB problem introduces a **reference path measure** $\mathbb{Q}$ which describes the baseline stochastic dynamics of a system. The dynamic Schroedinger bridge problem then selects, among all processes **matching prescribed initial and terminal marginals**, the one that deviates minimally from the reference process in relative entropy.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.2</span><span class="math-callout__name">(Dynamic Schroedinger Bridge Problem)</span></p>

Let $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$ be probability measures on state space $\mathbb{R}^d$ and let $\mathbb{Q} \in \mathcal{P}(C([0,T]; \mathbb{R}^d))$ be a **reference path measure**, where $\mathcal{P}(C([0,T]; \mathbb{R}^d))$ is the space of probability paths over $\mathbb{R}^d$.

The **dynamic Schroedinger bridge (SB) problem** seeks a new path measure $\mathbb{P} \in \mathcal{P}(C([0,T]; \mathbb{R}^d))$ with initial and terminal marginals matching $p\_0 = \pi\_0$ and $p\_T = \pi\_T$ which minimizes the relative entropy with respect to $\mathbb{Q}$:

$$
\mathbb{P}^\star = \underset{\mathbb{P} \in \mathcal{P}(C([0,T]; \mathbb{R}^d))}{\arg\min} \left\lbrace \text{KL}(\mathbb{P} \| \mathbb{Q}) : p_0 = \pi_0, \; p_T = \pi_T \right\rbrace
$$

where $\text{KL}(\cdot \| \cdot)$ denotes the KL divergence on path space: $\text{KL}(\mathbb{P} \| \mathbb{Q}) = \mathbb{E}\_{\mathbb{P}}\!\left[\log \frac{d\mathbb{P}}{d\mathbb{Q}}\right]$.

</div>

This perspective reveals the Schroedinger bridge as an entropy-regularized analogue of the Dynamic OT Problem. In the limit of vanishing noise, the problem recovers classical optimal transport, while for positive noise, it admits a rich stochastic structure. When the reference dynamics is Brownian motion, the optimal path measure inherits a Markov structure and can be characterized through a time-dependent drift correction of the reference process.

### 2.3 Path Measures and Ito Processes

Before diving deeper into Schroedinger Bridge theory, let's establish the foundations of **path measures and Ito processes**. A **stochastic process** is a random variable $\boldsymbol{X}\_t \in \mathbb{R}^d$ that evolves over a time horizon $t \in [0, T]$, denoted $\boldsymbol{X}\_{0:T} := (\boldsymbol{X}\_t)\_{t \in [0,T]}$. The distribution of such random variables follows a time-dependent marginal probability distribution $p\_t \in \mathcal{P}(\mathbb{R}^d)$.

We consider only stochastic processes $\boldsymbol{X}\_{0:T}$ that *depend* only on the **past** and **present** states of the system, which are formally said to be **adapted to the filtration** $(\mathcal{F}\_t)\_{t \in [0,T]}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.3</span><span class="math-callout__name">($\mathcal{F}\_t$-Adapted Process)</span></p>

A stochastic process $\boldsymbol{X}\_{0:T}$ is said to be $\mathcal{F}\_t$-adapted if the random variable $\boldsymbol{X}\_t$ is $\mathcal{F}\_t$-measurable for all $t \in [0, T]$, where:

$$
\mathcal{F}_t := \sigma(\boldsymbol{X}_\tau : 0 \le \tau \le t)
$$

is the sigma-algebra generated by the history of the process up to time $t$. Equivalently, for all measurable sets $S \in \mathcal{S}(\mathbb{R}^d)$, the process satisfies $\lbrace \boldsymbol{X}\_t \in S \rbrace \in \mathcal{F}\_t$, meaning that all values $\boldsymbol{X}\_t \in S$ can be determined using only information up to time $t$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.4</span><span class="math-callout__name">(Brownian Motion)</span></p>

Brownian motion $(\boldsymbol{B}\_t)\_{t \in [0,T]}$ is a type of stochastic process that starts at $\boldsymbol{B}\_0 = \boldsymbol{0}$ and evolves via **independent Gaussian increments**:

$$
\boldsymbol{B}_{t+\Delta t} = \boldsymbol{B}_t + \sqrt{\Delta t}\,\boldsymbol{z}, \quad \boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, I_d)
$$

where $\boldsymbol{z}$ is sampled independently from a unit isotropic Gaussian with zero-mean across all time steps. Each increment is Gaussian: $\boldsymbol{B}\_{t+\Delta t} - \boldsymbol{B}\_t \sim \mathcal{N}(\boldsymbol{0}, \Delta t I\_d)$ and independent of $\mathcal{F}\_t$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.5</span><span class="math-callout__name">(Ito Process)</span></p>

An Ito Process is a stochastic process $(\boldsymbol{X}\_t)\_{t \in [0,T]}$ whose state $\boldsymbol{X}\_t$ can be written as

$$
\boldsymbol{X}_t = \boldsymbol{X}_0 + \int_0^t \boldsymbol{f}(\boldsymbol{X}_t, t) ds + \int_0^t \boldsymbol{\Sigma}_t d\boldsymbol{B}_t
$$

which can be equivalently defined as the solution of an **stochastic differential equation** (SDE):

$$
d\boldsymbol{X}_t = \boldsymbol{f}(\boldsymbol{X}_t, t) dt + \boldsymbol{\Sigma}_t d\boldsymbol{B}_t
$$

where $\boldsymbol{f}(\boldsymbol{X}\_t, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ is the **drift** and $\boldsymbol{\Sigma}\_t \in \mathbb{R}^{d \times d}$ is the **diffusion coefficient matrix**. In most applications, the diffusion coefficient simplifies to a scalar $\sigma\_t : [0, T] \to \mathbb{R}\_{\ge 0}$, giving $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$.

</div>

Next we consider stochastic dynamics where an external influence perturbs the drift via a control drift. These **controlled Ito processes** can be modeled with a controlled SDE.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.6</span><span class="math-callout__name">(Controlled Ito Process)</span></p>

A controlled Ito process is obtained by introducing a control $\boldsymbol{u}(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ that modifies the drift in the directions spanned by the diffusion coefficient, yielding the **controlled stochastic differential equation** (SDE):

$$
d\boldsymbol{X}_t^{\boldsymbol{u}} = (\boldsymbol{f}(\boldsymbol{X}_t^{\boldsymbol{u}}, t) + \boldsymbol{\Sigma}_t \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{u}}, t)) dt + \boldsymbol{\Sigma}_t d\boldsymbol{B}_t
$$

where $\boldsymbol{f}(\boldsymbol{x}, t)$ is the reference drift and $\boldsymbol{\Sigma}\_t$ is the diffusion coefficient. Equivalently:

$$
\boldsymbol{X}_t^{\boldsymbol{u}} = \boldsymbol{X}_0 + \int_0^t (\boldsymbol{f}(\boldsymbol{X}_\tau^{\boldsymbol{u}}, \tau) + \boldsymbol{\Sigma}_\tau \boldsymbol{u}(\boldsymbol{X}_\tau^{\boldsymbol{u}}, \tau)) d\tau + \int_0^t \boldsymbol{\Sigma}_\tau d\boldsymbol{B}_\tau
$$

</div>

The control is scaled by the diffusion coefficient $\boldsymbol{\Sigma}\_t \boldsymbol{u}$ rather than added independently. This parameterization guarantees that the controlled process remains **absolutely continuous** with respect to the reference path measure $\mathbb{Q}$ defined using the diffusion term $\sigma\_t d\boldsymbol{B}\_t$. This property ensures the relative entropy used in the dynamic SB objective takes a tractable form. Intuitively, scaling with the diffusion coefficient ensures that the control drift **only steers the process in directions within the space supported by the stochastic noise**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.7</span><span class="math-callout__name">(Ito's Formula in $\mathbb{R}^d$)</span></p>

Consider an Ito process $(\boldsymbol{X}\_t \in \mathbb{R}^d)\_{t \in [0,T]}$ and a scalar function $\phi(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ that is twice continuously differentiable $\phi \in C^{2,1}(\mathbb{R}^d \times [0, T])$. Then, the transformed random variable $\phi(\boldsymbol{X}\_t, t)$ is also an Ito process that evolves via the SDE:

$$
d\phi(\boldsymbol{X}_t, t) = \partial_t \phi(\boldsymbol{X}_t, t) + \nabla\phi(\boldsymbol{X}_t, t)^\top d\boldsymbol{X}_t + \frac{1}{2} d\boldsymbol{X}_t^\top (\nabla^2 \phi(\boldsymbol{X}_t, t)) d\boldsymbol{X}_t
$$

Substituting $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \boldsymbol{\Sigma}\_t d\boldsymbol{B}\_t$, the formula becomes:

$$
d\phi(\boldsymbol{X}_t, t) = \left[\partial_t \phi(\boldsymbol{X}_t, t) + \boldsymbol{f}(\boldsymbol{X}_t, t)^\top \nabla\phi(\boldsymbol{X}_t, t) + \frac{1}{2}\text{Tr}\!\left(\boldsymbol{\Sigma}_t \boldsymbol{\Sigma}_t^\top \nabla^2 \phi(\boldsymbol{X}_t, t)\right)\right] dt + \nabla\phi(\boldsymbol{X}_t, t)^\top \boldsymbol{\Sigma}_t d\boldsymbol{B}_t
$$

This means the space of Ito processes is **closed under twice-differentiable functions**. When $\phi(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$ depends only on $\boldsymbol{x}$ and the diffusion is isotropic Gaussian $\boldsymbol{\Sigma}\_t = \sigma\_t I\_d$, the formula simplifies to:

$$
d\phi(\boldsymbol{X}_t) = \left[\boldsymbol{f}(\boldsymbol{X}_t, t)^\top \nabla\phi(\boldsymbol{X}_t, t) + \frac{\sigma_t^2}{2}\Delta\phi(\boldsymbol{X}_t)\right]dt + \nabla\phi(\boldsymbol{X}_t)^\top \sigma_t d\boldsymbol{B}_t
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Ito's Formula)</summary>

Substituting $d\boldsymbol{X}\_t = \boldsymbol{f}\_t dt + \boldsymbol{\Sigma}\_t d\boldsymbol{B}\_t$:

$$
d\phi(\boldsymbol{X}_t, t) = \partial_t\phi + \nabla\phi^\top(\boldsymbol{f}_t dt + \boldsymbol{\Sigma}_t d\boldsymbol{B}_t) + \frac{1}{2}(\boldsymbol{f}_t dt + \boldsymbol{\Sigma}_t d\boldsymbol{B}_t)^\top \nabla^2\phi \,(\boldsymbol{f}_t dt + \boldsymbol{\Sigma}_t d\boldsymbol{B}_t)
$$

By Ito's calculus rules: $dt \cdot dt = 0$, $(dt)^2 = 0$, and the key rule $(\boldsymbol{\Sigma}\_t d\boldsymbol{B}\_t)^\top(\nabla^2\phi)(\boldsymbol{\Sigma}\_t d\boldsymbol{B}\_t)$ simplifies. In component form, $(\boldsymbol{\Sigma}\_t d\boldsymbol{B}\_t)\_i = \sum\_k \Sigma\_t^{ik} dB\_t^k$, so:

$$
(\boldsymbol{\Sigma}_t d\boldsymbol{B}_t)^\top (\nabla^2\phi)(\boldsymbol{\Sigma}_t d\boldsymbol{B}_t) = \sum_{i,j,k,\ell} \Sigma_t^{ik} \Sigma_t^{j\ell} (\nabla^2\phi)_{ij} \, dB_t^k dB_t^\ell = \sum_{i,j,k} \Sigma_t^{ik}\Sigma_t^{jk} (\nabla^2\phi)_{ij} \, dt
$$

since $dB\_t^k dB\_t^\ell = \delta\_{k\ell} \, dt$. This gives $\text{Tr}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top \nabla^2\phi) \, dt$. When $\boldsymbol{\Sigma}\_t = \sigma\_t I\_d$:

$$\text{Tr}(\sigma_t^2 I_d \nabla^2\phi) = \sigma_t^2 \Delta\phi$$

$\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.8</span><span class="math-callout__name">(Ito Formula for Controlled SDEs)</span></p>

Consider a controlled stochastic process $(\boldsymbol{X}\_t \in \mathbb{R}^d)\_{t \in [0,T]}$ generated by the controlled SDE $d\boldsymbol{X}\_t = (\boldsymbol{f}(\boldsymbol{X}\_t, t) + \sigma\_t \boldsymbol{u}(\boldsymbol{X}\_t, t)) dt + \sigma\_t d\boldsymbol{B}\_t$. For any scalar function $\phi(\boldsymbol{x}, t) \in C^{2,1}(\mathbb{R}^d \times [0, T])$, the transformed stochastic process $\phi(\boldsymbol{X}\_t^{\boldsymbol{u}}, t)$ follows the SDE:

$$
d\phi(\boldsymbol{X}_t^{\boldsymbol{u}}, t) = \left[\partial_t\phi(\boldsymbol{X}_t^{\boldsymbol{u}}, t) + (\boldsymbol{f} + \sigma_t\boldsymbol{u})(\boldsymbol{X}_t^{\boldsymbol{u}})^\top \nabla\phi(\boldsymbol{X}_t^{\boldsymbol{u}}, t) + \frac{\sigma_t^2}{2}\Delta\phi(\boldsymbol{X}_t^{\boldsymbol{u}}, t)\right] dt + \nabla\phi(\boldsymbol{X}_t^{\boldsymbol{u}})^\top \sigma_t d\boldsymbol{B}_t
$$

If $\phi(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$ depends only on $\boldsymbol{x}$, then $\partial\_t\phi$ disappears and:

$$
d\phi(\boldsymbol{X}_t^{\boldsymbol{u}}) = \left[(\boldsymbol{f} + \sigma_t\boldsymbol{u})(\boldsymbol{X}_t^{\boldsymbol{u}})^\top \nabla\phi(\boldsymbol{X}_t^{\boldsymbol{u}}) + \frac{\sigma_t^2}{2}\Delta\phi(\boldsymbol{X}_t^{\boldsymbol{u}})\right] dt + \nabla\phi(\boldsymbol{X}_t^{\boldsymbol{u}})^\top \sigma_t d\boldsymbol{B}_t
$$

</div>

The controlled Ito formula shows that, for any sufficiently smooth test function, the drift term of the transformed process is determined by a **linear differential operator** called the **infinitesimal generator** of the SDE $\mathcal{A}\_t$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.9</span><span class="math-callout__name">(Infinitesimal Generator of Ito Process)</span></p>

For any test function $\phi(\boldsymbol{x}, t) \in C^{2,1}(\mathbb{R}^d \times [0, T])$, the **infinitesimal generator of the uncontrolled SDE** is the operator $\mathcal{A}\_t$ defined as:

$$
(\mathcal{A}_t\phi)(\boldsymbol{x}, t) := \boldsymbol{f}(\boldsymbol{x}, t)^\top \nabla\phi(\boldsymbol{x}, t) + \frac{\sigma_t^2}{2}\Delta\phi(\boldsymbol{x}, t)
$$

and the **infinitesimal generator of the controlled SDE** is the operator $\mathcal{A}\_t^{\boldsymbol{u}}$ defined as:

$$
(\mathcal{A}_t^{\boldsymbol{u}}\phi)(\boldsymbol{x}, t) := (\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t \boldsymbol{u}(\boldsymbol{x}, t))^\top \nabla\phi(\boldsymbol{x}, t) + \frac{\sigma_t^2}{2}\Delta\phi(\boldsymbol{x}, t) = (\mathcal{A}_t\phi)(\boldsymbol{x}, t) + \boldsymbol{u}(\boldsymbol{x}, t)^\top \nabla\phi(\boldsymbol{x}, t)
$$

The infinitesimal generator describes the instantaneous rate of change of the expected value of $\phi(\boldsymbol{X}\_t, t)$ at time $t$:

$$
(\mathcal{A}_t\phi)(\boldsymbol{x}, t) = \lim_{\Delta t \to 0} \frac{\mathbb{E}[\phi(\boldsymbol{X}_{t+\Delta t}, t + \Delta t) \mid \boldsymbol{X}_t = \boldsymbol{x}] - \phi(\boldsymbol{x}, t)}{\Delta t}
$$

Using the infinitesimal generator, Ito's formula can be compactly written as:

$$
d\phi(\boldsymbol{X}_t, t) = (\partial_t + \mathcal{A}_t)\phi(\boldsymbol{X}_t, t) dt + \sigma_t \nabla\phi(\boldsymbol{X}_t, t)^\top d\boldsymbol{B}_t
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.10</span><span class="math-callout__name">(Martingales)</span></p>

A $\mathcal{F}\_t$-adapted stochastic process $\boldsymbol{Y}\_{0:T} := (Y\_t)\_{t \in [0,T]}$ is called a **martingale** if it is integrable for all $t \in [0, T]$ (i.e., $\mathbb{E}[\lvert Y\_t \rvert] < \infty$) and satisfies the **martingale property**:

$$
\mathbb{E}[Y_t \mid \mathcal{F}_s] = Y_s \implies \mathbb{E}[Y_t] = \mathbb{E}[Y_s], \quad \forall 0 \le s \le t
$$

which means that the expectation is *constant over time*. Furthermore, if $Y\_t := \phi(\boldsymbol{X}\_t, t)$ is the martingale process generated by $\phi(\boldsymbol{x}, t) \in C^{2,1}(\mathbb{R}^d \times [0, T])$, the uncontrolled generator vanishes:

$$
(\mathcal{A}_t\phi)(\boldsymbol{X}_t, t) = \boldsymbol{f}(\boldsymbol{X}_t, t)^\top \nabla\phi(\boldsymbol{X}_t, t) + \frac{\sigma_t^2}{2}\Delta\phi(\boldsymbol{X}_t, t) = 0
$$

</div>

Sampling an infinite number of paths under a specific SDE forms a **probability distribution in the path space**, known as a **path measure**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.11</span><span class="math-callout__name">(Controlled Path Measure)</span></p>

A **controlled path measure** $\mathbb{P}^{\boldsymbol{u}} \in C([0,T]; \mathbb{R}^d)$ is a probability measure on the path space $C([0,T]; \mathbb{R}^d)$ induced by a stochastic differential equation (SDE) with control $\sigma\_t \boldsymbol{u}(\boldsymbol{x}, t)$ defined in the Controlled SDE (Definition 2.6). For any measurable set of trajectories $S \subseteq C([0,T]; \mathbb{R}^d)$, $\mathbb{P}^{\boldsymbol{u}}(S)$ is the probability that the trajectories in the set are generated under the controlled SDE:

$$
\mathbb{P}^{\boldsymbol{u}}(S) = \Pr(\boldsymbol{X}_{0:T} \in S \text{ under the controlled SDE of } \mathbb{P}^{\boldsymbol{u}})
$$

</div>

### 2.4 Fokker--Planck and Feynman--Kac Equations

Defining a path measure with an SDE captures the evolution of single particles along trajectories, but tells us nothing about the evolution of the *distribution* generated by many particles. The **Fokker--Planck equation** allows us to define the behavior of the **probability distribution** of particles $p\_t$ at each time point $t \in [0, T]$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.12</span><span class="math-callout__name">(Fokker--Planck Equation)</span></p>

Let $\boldsymbol{X}\_{0:T}$ be a stochastic process governed by the SDE $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$, where $\boldsymbol{f}(\boldsymbol{x}, t)$ is the drift and $\sigma\_t \in \mathbb{R}$ is the scalar diffusion coefficient. Then, the marginal density $p\_t \in \mathcal{P}(\mathbb{R}^d)$ of the particles generated by the SDE evolves via the **Fokker--Planck equation**:

$$
\partial_t p_t(\boldsymbol{x}) = -\nabla \cdot (\boldsymbol{f}(\boldsymbol{x}, t) p_t(\boldsymbol{x})) + \frac{\sigma_t^2}{2}\Delta p_t(\boldsymbol{x})
$$

where $\boldsymbol{X}\_t \sim p\_t$, i.e., $p\_t$ is the probability law of $\boldsymbol{X}\_t$: $\text{Law}(\boldsymbol{X}\_t) = p\_t$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Derivation (Fokker--Planck Equation)</summary>

Consider a test function $\phi(\boldsymbol{x}) \in C\_c^2(\mathbb{R}^d)$ that is twice differentiable and has compact support. By Ito's formula (Simplified Ito Process), the random variable $\phi(\boldsymbol{X}\_t)$ evolves as:

$$
d\phi(\boldsymbol{X}_t) = \left[\boldsymbol{f}(\boldsymbol{X}_t, t)^\top \nabla\phi(\boldsymbol{X}_t) + \frac{\sigma_t^2}{2}\Delta\phi(\boldsymbol{X}_t)\right] dt + \sigma_t \nabla\phi(\boldsymbol{X}_t)^\top d\boldsymbol{B}_t
$$

Taking the expectation (the stochastic term vanishes since the Ito integral has zero expectation) and dividing by $dt$:

$$
\partial_t \mathbb{E}_{p_t}[\phi(\boldsymbol{X}_t)] = \mathbb{E}_{p_t}\left[\boldsymbol{f}(\boldsymbol{X}_t, t)^\top \nabla\phi(\boldsymbol{X}_t) + \frac{\sigma_t^2}{2}\Delta\phi(\boldsymbol{X}_t)\right]
$$

Writing in integral form: $\int\_{\mathbb{R}^d} \phi(\boldsymbol{x}) \partial\_t p\_t(\boldsymbol{x}) d\boldsymbol{x} = \int\_{\mathbb{R}^d} \left[\boldsymbol{f}(\boldsymbol{x}, t)^\top \nabla\phi(\boldsymbol{x}) + \frac{\sigma\_t^2}{2}\Delta\phi(\boldsymbol{x})\right] p\_t(\boldsymbol{x}) d\boldsymbol{x}$.

Applying **integration by parts** twice to the drift term moves derivatives from $\phi$ to $p\_t$:

$$
\int \boldsymbol{f} \cdot \nabla\phi \, p_t \, d\boldsymbol{x} = -\int \phi \, \nabla \cdot (\boldsymbol{f} p_t) \, d\boldsymbol{x}
$$

Similarly for the diffusion term (two integrations by parts):

$$
\frac{\sigma_t^2}{2}\int \Delta\phi \, p_t \, d\boldsymbol{x} = \frac{\sigma_t^2}{2}\int \phi \, \Delta p_t \, d\boldsymbol{x}
$$

Combining: $\int \phi(\boldsymbol{x}) \partial\_t p\_t \, d\boldsymbol{x} = \int \phi(\boldsymbol{x})\left(-\nabla \cdot (\boldsymbol{f} p\_t) + \frac{\sigma\_t^2}{2}\Delta p\_t\right) d\boldsymbol{x}$. Since this holds for *any* test function $\phi$, the integrands must be equal, recovering the Fokker--Planck equation. $\square$
</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.13</span><span class="math-callout__name">(Fokker--Planck Equation Generalizes the Continuity Equation)</span></p>

The Fokker--Planck equation can be interpreted as a **stochastic generalization of the classic continuity equation** from the Dynamic OT Problem:

$$
\partial_t p_t(\boldsymbol{x}) + \nabla \cdot \left(p_t(\boldsymbol{x}) \boldsymbol{v}(\boldsymbol{x}, t)\right) = 0
$$

which expresses that probability mass is transported along the flow induced by the velocity $\boldsymbol{v}(\boldsymbol{x}, t)$. The Fokker--Planck equation is equal to the continuity equation with an additional Laplacian term $\frac{\sigma\_t^2}{2}\Delta p\_t$ that captures the **spreading of mass caused by Brownian motion**. The Laplacian $\Delta p\_t = \nabla \cdot \nabla p\_t$ measures whether the gradient of probability density $\nabla p\_t$ is spreading out (positive divergence) or converging (negative divergence), so the positive diffusion coefficient increases the **spreading out of probability density**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.14</span><span class="math-callout__name">(Controlled Fokker--Planck Equation)</span></p>

Let $(\boldsymbol{X}\_t^{\boldsymbol{u}})\_{t \in [0,T]}$ be a stochastic process governed by the SDE $d\boldsymbol{X}\_t^{\boldsymbol{u}} = (\boldsymbol{f}(\boldsymbol{X}\_t^{\boldsymbol{u}}, t) + \sigma\_t \boldsymbol{u}(\boldsymbol{X}\_t^{\boldsymbol{u}}, t)) dt + \sigma\_t d\boldsymbol{B}\_t$, where $\boldsymbol{u}(\boldsymbol{x}, t)$ is the control drift. Let $p\_t \in \mathcal{P}(\mathbb{R}^d)$ denote the marginal density such that $\boldsymbol{X}\_t^{\boldsymbol{u}} \sim p\_t$. Then $p\_t$ evolves according to the **controlled Fokker--Planck equation**:

$$
\partial_t p_t(\boldsymbol{x}) = -\nabla \cdot ((\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t \boldsymbol{u}(\boldsymbol{x}, t)) p_t(\boldsymbol{x})) + \frac{\sigma_t^2}{2}\Delta p_t(\boldsymbol{x})
$$

Equivalently, expanding the divergence term:

$$
\partial_t p_t(\boldsymbol{x}) = -\nabla \cdot (\boldsymbol{f}(\boldsymbol{x}, t) p_t(\boldsymbol{x})) - \nabla \cdot (\sigma_t \boldsymbol{u}(\boldsymbol{x}, t) p_t(\boldsymbol{x})) + \frac{\sigma_t^2}{2}\Delta p_t(\boldsymbol{x})
$$

</div>

The Fokker--Planck equation provides a forward-time description of the stochastic dynamics. However, the Schroedinger bridge problem is inherently a two-sided problem enforced by both initial *and terminal* marginal distribution constraints. We need to consider how constraints on the terminal distribution $p\_T$ evolve **backward in time**. This is captured by the **Feynman--Kac equation**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.15</span><span class="math-callout__name">(Feynman--Kac Equation)</span></p>

Consider a scalar function $r(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ that solves the linear PDE with a terminal constraint $r(\boldsymbol{x}, T) = \Phi(\boldsymbol{x})$ and running cost $c(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$:

$$
\partial_t r(\boldsymbol{x}, t) + \langle \boldsymbol{f}(\boldsymbol{x}, t), \nabla r(\boldsymbol{x}, t) \rangle + \frac{\sigma_t^2}{2}\Delta r(\boldsymbol{x}, t) - c(\boldsymbol{x}, t) r(\boldsymbol{x}, t) = 0, \quad r(\boldsymbol{x}, T) = \Phi(\boldsymbol{x})
$$

where $\boldsymbol{f}(\boldsymbol{x}, t)$ is the reference drift and $\sigma\_t$ is the diffusion coefficient of an SDE that generates a stochastic process $(\boldsymbol{X}\_\tau)\_{\tau \in [t, T]}$ defined by $d\boldsymbol{X}\_\tau = \boldsymbol{f}(\boldsymbol{X}\_\tau, \tau) d\tau + \sigma\_\tau d\boldsymbol{B}\_\tau$ with $\boldsymbol{X}\_t = \boldsymbol{x}$.

Then, the solution $r$ can be written as the following expectation over stochastic paths:

$$
r(\boldsymbol{x}, t) = \mathbb{E}\left[\exp\!\left(-\int_t^T c(\boldsymbol{X}_s, s) ds\right) \Phi(\boldsymbol{X}_T) \;\middle|\; \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

which is called the **Feynman--Kac formula**.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Derivation (Feynman--Kac Formula)</summary>

Intuitively, $r(\boldsymbol{x}, t)$ is the expected future value of the terminal constraint $\Phi(\boldsymbol{X}\_T)$ after discounting the running cost $c(\boldsymbol{X}\_s, s)$ given the current state $\boldsymbol{X}\_t = \boldsymbol{x}$ at time $t$.

**Step 1: Define the Stochastic Processes.** Consider the stochastic process describing the accumulated running cost over $[\tau, t]$:

$$
M_\tau := \underbrace{\exp\!\left(-\int_t^\tau c(\boldsymbol{X}_s, s) ds\right)}_{=: Z_\tau} r(\boldsymbol{X}_\tau, \tau), \quad \tau \in [t, T]
$$

We aim to show that $M\_\tau$ is a **martingale** (Definition 2.10), i.e., $\mathbb{E}[M\_T] = M\_t$.

**Step 2: Apply Ito's Formula.** Applying Ito's product rule to $M\_\tau = Z\_\tau \, r(\boldsymbol{X}\_\tau, \tau)$:

$$
dM_\tau = Z_\tau \, dr(\boldsymbol{X}_\tau, \tau) + r(\boldsymbol{X}_\tau, \tau) \, dZ_\tau + \underbrace{dZ_\tau \, dr(\boldsymbol{X}_\tau, \tau)}_{= O(d\tau^2) = 0}
$$

Since $dZ\_\tau = -c(\boldsymbol{X}\_\tau, \tau) Z\_\tau \, d\tau$ (no stochastic term), and by Ito's formula on $r$:

$$
dr(\boldsymbol{X}_\tau, \tau) = \underbrace{\left[\partial_\tau r + \boldsymbol{f}^\top \nabla r + \frac{\sigma_\tau^2}{2}\Delta r\right]}_{(\star)} d\tau + \nabla r^\top \sigma_\tau d\boldsymbol{B}_\tau
$$

Combining: $dM\_\tau = Z\_\tau\left[(\star) - c \, r\right] d\tau + Z\_\tau \nabla r^\top \sigma\_\tau d\boldsymbol{B}\_\tau$.

Since $r$ solves the Feynman--Kac PDE, $(\star) - c \, r = 0$, so the drift term vanishes and $M\_\tau$ is a martingale. Therefore $\mathbb{E}[M\_T \mid \boldsymbol{X}\_t = \boldsymbol{x}] = M\_t = r(\boldsymbol{x}, t)$, recovering the Feynman--Kac formula. $\square$
</details>
</div>

A special case of the Feynman--Kac formula when there is no running cost ($c \equiv 0$) is the **Kolmogorov backward equation**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.16</span><span class="math-callout__name">(Kolmogorov Backward Equation)</span></p>

Let $\Phi : \mathbb{R}^d \to \mathbb{R}$ be a terminal constraint function and define $r(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ as the expected future reward from an intermediate state $\boldsymbol{X}\_t = \boldsymbol{x}$:

$$
r(\boldsymbol{x}, t) := \mathbb{E}_{\boldsymbol{X}_\tau \sim \mathbb{Q}}[\Phi(\boldsymbol{X}_T) \mid \boldsymbol{X}_t = \boldsymbol{x}]
$$

where $\mathbb{Q}$ is the reference path measure defined by the SDE $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) ds + \sigma\_t d\boldsymbol{B}\_t$. Then $r(\boldsymbol{x}, t)$ satisfies the **Kolmogorov backward equation**:

$$
\partial_t r(\boldsymbol{x}, t) + \langle \boldsymbol{f}(\boldsymbol{x}, t), \nabla r(\boldsymbol{x}, t) \rangle + \frac{\sigma_t^2}{2}\Delta r(\boldsymbol{x}, t) = 0, \quad r(\boldsymbol{x}, T) = \Phi(\boldsymbol{x})
$$

</div>

Together, the Fokker--Planck equation and the Feynman--Kac formula describe the two complementary ways in which stochastic differential equations evolve in time. The **Fokker--Planck equation** characterizes the forward evolution of probability densities $p\_t$, describing how the distribution of particles transported by an SDE spreads and flows through state space. In contrast, the **Feynman--Kac formula** provides a backward evolution of functions along stochastic trajectories, expressing solutions to certain PDEs as expectations over future paths. The Fokker--Planck equation propagates probability mass forward in time, while the Feynman--Kac equation propagates value functions or potentials backward. This forward--backward structure plays a central role in the theory of Schroedinger bridges.

### 2.5 Girsanov's Theorem

Now that we have established **path measures** and how they evolve both on an individual trajectory level via SDEs and on a density level via the Fokker--Planck equation, we shift focus to the **relationship between path measures**. This is the **core objective** of the Schroedinger bridge problem: to define a new path measure that minimally diverges from a reference path measure with marginal constraints.

We begin with **Girsanov's theorem**, which characterizes the probability density ratio of a single stochastic process under different path measures. The key idea is to understand how a change of drift modifies the underlying path measure.

Recall from Definition 2.4 that Brownian motion can be written as a sequence of **independent Gaussian increments** in discrete time with step size $\Delta t := t\_{k+1} - t\_k$:

$$
\sigma\mathbb{B}: \quad \boldsymbol{B}_{t_{k+1}} = \boldsymbol{B}_{t_k} + \sigma\sqrt{\Delta t}\,\boldsymbol{z}, \quad \boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, I_d)
$$

The transition probability under $\sigma\mathbb{B}$ is a Gaussian centered at $\boldsymbol{B}\_{t\_k}$ with covariance $\sigma^2 \Delta t I\_d$. The joint probability of the full discrete process is:

$$
\sigma\mathbb{B}(\boldsymbol{B}_{t_1}, \dots, \boldsymbol{B}_{t_K}) = \prod_{k=1}^K \frac{1}{(2\pi\sigma^2\Delta t)^{d/2}} \exp\!\left(-\frac{\lVert \Delta\boldsymbol{B}_{t_k} \rVert^2}{2\Delta t}\right)
$$

Now consider the controlled Ito process $\mathbb{P}: \boldsymbol{X}\_{t\_{k+1}} = \boldsymbol{X}\_{t\_k} + \sigma\boldsymbol{u}(\boldsymbol{X}\_{t\_k}, t\_k)\Delta t + \sigma\Delta\boldsymbol{B}\_{t\_k}$. Its transition probability is a Gaussian centered at $\boldsymbol{X}\_{t\_k} + \sigma\boldsymbol{u}(\boldsymbol{X}\_{t\_k}, t\_k)\Delta t$ with covariance $\sigma^2 \Delta t I\_d$. The Radon--Nikodym derivative (density ratio) between $\mathbb{P}$ and $\sigma\mathbb{B}$ becomes:

$$
\frac{d\mathbb{P}}{d\sigma\mathbb{B}}(\boldsymbol{X}_{0:T}) = \prod_{k=1}^K \exp\!\left(-\frac{\lVert \sigma\Delta\boldsymbol{B}_{t_k} - \sigma\boldsymbol{u}(\boldsymbol{X}_{t_k}, t_k)\Delta t \rVert^2}{2\sigma^2\Delta t} + \frac{\lVert \Delta\boldsymbol{B}_{t_k} \rVert^2}{2\Delta t}\right)
$$

Taking the continuous time limit as $\Delta t \to 0$:

$$
\frac{d\mathbb{P}}{d\sigma\mathbb{B}}(\boldsymbol{X}_{0:T}) = \exp\!\left(-\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t, t) \rVert^2 dt + \int_0^T \boldsymbol{u}(\boldsymbol{X}_t, t)^\top d\boldsymbol{B}_t\right)
$$

This ratio is the **Radon--Nikodym derivative (RND)** between $\mathbb{P}$ and $\sigma\mathbb{B}$, which can be used to transform the likelihood of a path under $\sigma\mathbb{B}$ to its likelihood under $\mathbb{P}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.17</span><span class="math-callout__name">(Girsanov's Theorem)</span></p>

Consider $d$-dimensional Brownian motion $(\boldsymbol{B}\_t)\_{t \in [0,T]}$ adapted to the filtration $(\mathcal{F}\_t)\_{t \in [0,T]}$. Given two path measures $\mathbb{P}, \mathbb{P}' \in \mathcal{P}(\mathbb{R}^d)$, where $\mathbb{P}' \ll \mathbb{P}$, define the density process $(\boldsymbol{Z}\_t)\_{t \in [0,T]}$ of the ratio:

$$
\boldsymbol{Z}_t := \mathbb{E}_{\mathbb{P}}\!\left[\frac{d\mathbb{P}'}{d\mathbb{P}} \;\middle|\; \mathcal{F}_t\right]
$$

which is the likelihood ratio given the information up to time $t$. Then, there exists a predictable process $(\boldsymbol{\theta}\_s)\_{s \in [0,T]}$ such that:

$$
\boldsymbol{Z}_t = \exp\!\left(-\frac{1}{2}\int_0^t \lVert \boldsymbol{\theta}_s \rVert^2 ds + \int_0^t \boldsymbol{\theta}_s^\top d\boldsymbol{B}_s\right), \quad t \in [0, T]
$$

Furthermore, the stochastic process $(\boldsymbol{B}'\_t)\_{t \in [0,T]}$ defined as:

$$
\boldsymbol{B}'_t := \boldsymbol{B}_t - \int_0^t \boldsymbol{\theta}_s ds
$$

is a standard Brownian motion under $\mathbb{P}'$, such that for any discrete time intervals $0 = t\_0 \le \cdots \le t\_k \le \cdots \le t\_K \le T$ the increments $\Delta\boldsymbol{B}'\_{t\_k} := \boldsymbol{B}'\_{t\_{k+1}} - \boldsymbol{B}'\_{t\_k} \sim \mathcal{N}(\boldsymbol{0}, (t\_{k+1} - t\_k)I\_d)$ and independent.

</div>

Girsanov's theorem provides a way to change the probability measure on path space while leaving the underlying trajectories unchanged. By defining $\boldsymbol{\theta}\_t$ such that the drift of the original path probability cancels out after applying the transformation, Girsanov's theorem states that the $\boldsymbol{\theta}\_t$-tilted process becomes a standard Brownian motion under the transformed probability measure.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.18</span><span class="math-callout__name">(Novikov's Condition)</span></p>

Let $(\boldsymbol{B}\_t)\_{t \in [0,T]}$ be a $d$-dimensional Brownian motion adapted to $(\mathcal{F}\_t)\_{t \in [0,T]}$ and let $(\boldsymbol{\theta}\_t)\_{t \in [0,T]}$ be a $\mathcal{F}\_t$-adapted process that is square integrable over every finite time interval, such that $\int\_0^T \lVert \boldsymbol{\theta}\_t \rVert^2 dt < \infty$ almost surely. Then, if the exponential process:

$$
\boldsymbol{Z}_t = \exp\!\left(-\frac{1}{2}\int_0^t \lVert \boldsymbol{\theta}_s \rVert^2 ds + \int_0^t \boldsymbol{\theta}_s^\top d\boldsymbol{B}_s\right)
$$

satisfies **Novikov's condition**:

$$
\mathbb{E}\!\left[\exp\!\left(\frac{1}{2}\int_0^t \lVert \boldsymbol{\theta}_s \rVert^2 ds\right)\right] < \infty
$$

Then, $(\boldsymbol{Z}\_t)\_{t \in [0,T]}$ is a true martingale satisfying the Martingale Property with constant expectation $\mathbb{E}[Z\_t] = 1$ for $t \in [0, T]$.

</div>

Novikov's condition ensures $\boldsymbol{Z}\_t$ does not lose mass and is properly normalized, so the exponential tilt serves as a **valid likelihood ratio between path measures** and defines a **proper change of probability measure on path space**.

### 2.6 Path Measure Radon--Nikodym Derivative and KL Divergence

To understand the SB problem, it is crucial to understand the measurement used to quantify **distance** between two path measures $\mathbb{Q}, \mathbb{P} \in \mathcal{P}(C([0,T]; \mathbb{R}^d))$: the **Kullback--Leibler (KL) divergence** (also referred to as the **relative entropy** or **I-divergence**), denoted $\text{KL}(\mathbb{P} \| \mathbb{Q})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.19</span><span class="math-callout__name">(KL Divergence Between Path Measures)</span></p>

Given two probability measures over the space of paths $\mathbb{P}, \mathbb{P}' \in \mathcal{P}(C([0,T]; \mathbb{R}^d))$, the KL divergence is given by

$$
\text{KL}(\mathbb{P}' \| \mathbb{P}) = \mathbb{E}_{\mathbb{P}'}\!\left[\log \frac{d\mathbb{P}'}{d\mathbb{P}}\right] = \int_{C([0,T]; \mathbb{R}^d)} \log \frac{d\mathbb{P}'}{d\mathbb{P}}(\boldsymbol{X}_{0:T}) \, d\mathbb{P}'(\boldsymbol{X}_{0:T})
$$

where $\boldsymbol{X}\_{0:T}$ is a stochastic process and $\frac{d\mathbb{P}'}{d\mathbb{P}}$ is the Radon--Nikodym derivative between $\mathbb{P}'$ and $\mathbb{P}$.

</div>

Since we have defined **path measures** as solutions to SDEs (Section 2.3), we can now derive the **Radon--Nikodym (RN) derivative** with respect to the **drift** and **diffusion** of two Ito processes.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.20</span><span class="math-callout__name">(Radon--Nikodym Derivative of Controlled Ito Path Measures)</span></p>

Consider two controlled Ito path measures $\mathbb{P}^{\boldsymbol{u}}$ and $\mathbb{P}^{\tilde{\boldsymbol{u}}}$ with control drifts $\boldsymbol{u}$ and $\tilde{\boldsymbol{u}}$, where $\mathbb{P}^{\tilde{\boldsymbol{u}}} \ll \mathbb{P}^{\boldsymbol{u}}$. Assuming the same reference drift $\boldsymbol{f}(\boldsymbol{x}, t)$ and diffusion coefficient $\sigma\_t$, they are defined by the SDEs:

$$
\mathbb{P}^{\boldsymbol{u}}: \quad d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t)) dt + \sigma_t d\boldsymbol{B}_t^{\boldsymbol{u}}, \quad \boldsymbol{X}_0 = \boldsymbol{x}
$$

$$
\mathbb{P}^{\tilde{\boldsymbol{u}}}: \quad d\tilde{\boldsymbol{X}}_t = (\boldsymbol{f}(\tilde{\boldsymbol{X}}_t, t) + \sigma_t \tilde{\boldsymbol{u}}(\tilde{\boldsymbol{X}}_t, t)) dt + \sigma_t d\boldsymbol{B}_t^{\tilde{\boldsymbol{u}}}, \quad \tilde{\boldsymbol{X}}_0 = \boldsymbol{x}
$$

The Radon--Nikodym derivative of the corresponding path measures is:

$$
\frac{d\mathbb{P}^{\tilde{\boldsymbol{u}}}}{d\mathbb{P}^{\boldsymbol{u}}}(\boldsymbol{X}_{0:T}^{\boldsymbol{u}}) = \exp\!\left(-\frac{1}{2}\int_0^T \lVert (\tilde{\boldsymbol{u}} - \boldsymbol{u})(\boldsymbol{X}_t^{\boldsymbol{u}}, t) \rVert^2 dt + \int_0^T (\tilde{\boldsymbol{u}} - \boldsymbol{u})(\boldsymbol{X}_t^{\boldsymbol{u}}, t)^\top d\boldsymbol{B}_t^{\boldsymbol{u}}\right)
$$

where $\boldsymbol{X}\_{0:T}^{\boldsymbol{u}}$ denotes a stochastic process generated under $\mathbb{P}^{\boldsymbol{u}}$ and $\boldsymbol{B}\_t^{\boldsymbol{u}}$ denotes the Brownian motion under $\mathbb{P}^{\boldsymbol{u}}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Radon--Nikodym Derivative of Path Measures)</summary>

The **key idea** is to define how likely a stochastic process $\boldsymbol{X}\_{0:T}^{\boldsymbol{u}} := (\boldsymbol{X}\_t^{\boldsymbol{u}})\_{t \in [0,T]}$ defined as a standard Brownian motion under $\mathbb{P}^{\boldsymbol{u}}$ is under the path measure $\mathbb{P}^{\tilde{\boldsymbol{u}}}$. Since $\mathbb{P}^{\boldsymbol{u}}$ and $\mathbb{P}^{\tilde{\boldsymbol{u}}}$ differ only in their control drifts, we define the change in drifts as $\boldsymbol{\theta}\_t := (\tilde{\boldsymbol{u}} - \boldsymbol{u})(\boldsymbol{X}\_t^{\boldsymbol{u}}, t) \in \mathbb{R}^d$.

Following Girsanov's Theorem, define the density process:

$$
\boldsymbol{Z}_t := \exp\!\left(-\frac{1}{2}\int_0^t \lVert \boldsymbol{\theta}_s \rVert^2 ds + \int_0^t \boldsymbol{\theta}_s^\top d\boldsymbol{B}_s^{\boldsymbol{u}}\right) \quad \text{s.t.} \quad \mathbb{E}_{\mathbb{P}^{\boldsymbol{u}}}[\boldsymbol{Z}_T] = 1
$$

By Girsanov's Theorem, the Brownian motion $\boldsymbol{B}'$ under the new measure $\mathbb{P}'$ is:

$$
\boldsymbol{B}'_t = \boldsymbol{B}_t^{\boldsymbol{u}} - \int_0^t \boldsymbol{\theta}_s ds \implies d\boldsymbol{B}_t^{\boldsymbol{u}} = d\boldsymbol{B}'_t + \boldsymbol{\theta}_t dt
$$

Rewriting the $\boldsymbol{u}$-SDE under Brownian motion $\boldsymbol{B}'$:

$$
d\boldsymbol{X}_t^{\boldsymbol{u}} = (\boldsymbol{f} + \sigma_t \boldsymbol{u}) dt + \sigma_t(d\boldsymbol{B}'_t + \boldsymbol{\theta}_t dt) = (\boldsymbol{f} + \sigma_t(\boldsymbol{u} + \boldsymbol{\theta}_t)) dt + \sigma_t d\boldsymbol{B}'_t
$$

Substituting $\boldsymbol{\theta}\_t = (\tilde{\boldsymbol{u}} - \boldsymbol{u})$: $d\boldsymbol{X}\_t^{\boldsymbol{u}} = (\boldsymbol{f} + \sigma\_t \tilde{\boldsymbol{u}}) dt + \sigma\_t d\boldsymbol{B}'\_t$, which matches the $\tilde{\boldsymbol{u}}$-SDE. Therefore $\mathbb{P}' = \mathbb{P}^{\tilde{\boldsymbol{u}}}$ and $\frac{d\mathbb{P}^{\tilde{\boldsymbol{u}}}}{d\mathbb{P}^{\boldsymbol{u}}} = \boldsymbol{Z}\_T$. $\square$
</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.21</span><span class="math-callout__name">(KL Divergence of Ito Path Measures)</span></p>

Consider two controlled path measures $\mathbb{P}^{\tilde{\boldsymbol{u}}}$ and $\mathbb{P}^{\boldsymbol{u}}$, where $\mathbb{P}^{\tilde{\boldsymbol{u}}} \ll \mathbb{P}^{\boldsymbol{u}}$, with control drifts $\boldsymbol{u}$ and $\tilde{\boldsymbol{u}}$ and SDEs ($\boldsymbol{u}$-SDE) and ($\tilde{\boldsymbol{u}}$-SDE). The **KL divergence** of $\mathbb{P}^{\tilde{\boldsymbol{u}}}$ with respect to $\mathbb{P}^{\boldsymbol{u}}$ is:

$$
\text{KL}(\mathbb{P}^{\tilde{\boldsymbol{u}}} \| \mathbb{P}^{\boldsymbol{u}}) = \mathbb{E}_{\boldsymbol{X}_{0:T}^{\tilde{\boldsymbol{u}}} \sim \mathbb{P}^{\tilde{\boldsymbol{u}}}}\!\left[\frac{1}{2}\int_0^T \lVert \tilde{\boldsymbol{u}}(\boldsymbol{X}_t^{\tilde{\boldsymbol{u}}}, t) - \boldsymbol{u}(\boldsymbol{X}_t^{\tilde{\boldsymbol{u}}}, t) \rVert^2 ds\right]
$$

where $\boldsymbol{X}\_{0:T}^{\tilde{\boldsymbol{u}}}$ denotes stochastic trajectories sampled under the law of $\mathbb{P}^{\tilde{\boldsymbol{u}}}$. When $\mathbb{P}^{\boldsymbol{u}} := \mathbb{Q}$ is the **reference path measure** with zero control $\boldsymbol{u} \equiv 0$ and SDE $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$, the KL divergence reduces to:

$$
\text{KL}(\mathbb{P}^{\tilde{\boldsymbol{u}}} \| \mathbb{Q}) = \mathbb{E}_{\boldsymbol{X}_{0:T}^{\tilde{\boldsymbol{u}}} \sim \mathbb{P}^{\tilde{\boldsymbol{u}}}}\!\left[\frac{1}{2}\int_0^T \lVert \tilde{\boldsymbol{u}}(\boldsymbol{X}_t^{\tilde{\boldsymbol{u}}}, t) \rVert^2 ds\right]
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (KL Divergence of Ito Path Measures)</summary>

Starting from the definition of path measure KL divergence (Definition 2.19) and using the Path RND from Theorem 2.20:

$$
\text{KL}(\mathbb{P}^{\tilde{\boldsymbol{u}}} \| \mathbb{P}^{\boldsymbol{u}}) = \mathbb{E}_{\boldsymbol{X}_{0:T}^{\tilde{\boldsymbol{u}}} \sim \mathbb{P}^{\tilde{\boldsymbol{u}}}}\!\left[-\frac{1}{2}\int_0^T \lVert (\tilde{\boldsymbol{u}} - \boldsymbol{u}) \rVert^2 ds + \int_0^T (\tilde{\boldsymbol{u}} - \boldsymbol{u})^\top d\boldsymbol{B}_s^{\boldsymbol{u}}\right]
$$

Since the expectation is under $\mathbb{P}^{\tilde{\boldsymbol{u}}}$ and the Brownian motion is under $\mathbb{P}^{\boldsymbol{u}}$, we apply Girsanov's theorem: $d\boldsymbol{B}\_t^{\boldsymbol{u}} = d\boldsymbol{B}\_t^{\tilde{\boldsymbol{u}}} + \boldsymbol{\theta}\_t dt$ where $\boldsymbol{\theta}\_t = (\tilde{\boldsymbol{u}} - \boldsymbol{u})$. Substituting:

$$
\text{KL}(\mathbb{P}^{\tilde{\boldsymbol{u}}} \| \mathbb{P}^{\boldsymbol{u}}) = \mathbb{E}\!\left[-\frac{1}{2}\int_0^T \lVert \tilde{\boldsymbol{u}} - \boldsymbol{u} \rVert^2 ds + \underbrace{\int_0^T (\tilde{\boldsymbol{u}} - \boldsymbol{u})^\top d\boldsymbol{B}_s^{\tilde{\boldsymbol{u}}}}_{\text{vanishes under expectation}} + \int_0^T \lVert \tilde{\boldsymbol{u}} - \boldsymbol{u} \rVert^2 ds\right]
$$

$$
= \mathbb{E}_{\boldsymbol{X}_{0:T}^{\tilde{\boldsymbol{u}}} \sim \mathbb{P}^{\tilde{\boldsymbol{u}}}}\!\left[\frac{1}{2}\int_0^T \lVert \tilde{\boldsymbol{u}}(\boldsymbol{X}_t^{\tilde{\boldsymbol{u}}}, t) - \boldsymbol{u}(\boldsymbol{X}_t^{\tilde{\boldsymbol{u}}}, t) \rVert^2 ds\right] \quad \square
$$

</details>
</div>

The Radon--Nikodym derivative between controlled diffusion path measures provides an explicit representation of how changes in drift modify the probability of trajectories. Using Girsanov's theorem, this change of measure produces a **quadratic control cost** whose expectation reduces to the **path-space KL divergence**. Minimizing the KL divergence between controlled path measures is equivalent to the expectation of the squared difference between the drifts over sampled stochastic paths.

### 2.7 Schroedinger Bridge with Arbitrary Reference Dynamics

Within the family of controlled path measures, the **dynamic Schroedinger bridge (SB) problem** seeks the control drift $\boldsymbol{u}^\star$ that minimally perturbs the reference path measure while steering the process between prescribed marginal distributions. Minimizing the perturbation from the reference path measure is defined as **minimizing the KL divergence**, which, following Corollary 2.21, is equivalent to minimizing the kinetic energy produced from perturbing the reference dynamics with the control drift.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.22</span><span class="math-callout__name">(Dynamic Schroedinger Bridge with Arbitrary Reference Dynamics)</span></p>

Let $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$ denote the prescribed initial and terminal distributions. The **dynamic Schroedinger bridge problem**, where the reference dynamics are defined by a drift $\boldsymbol{f}$, aims to determine the optimal control drift $\boldsymbol{u}^\star$ that minimizes the KL divergence of the induced path measures subject to the marginal constraints:

$$
\inf_{\mathbb{P}^{\boldsymbol{u}} \in \mathcal{P}(C([0,T]; \mathbb{R}^d))} \text{KL}(\mathbb{P}^{\boldsymbol{u}} \| \mathbb{Q}) = \inf_{\boldsymbol{u} \in \mathcal{U}} \mathbb{E}_{\boldsymbol{X}_{0:T}^{\boldsymbol{u}} \sim \mathbb{P}^{\boldsymbol{u}}}\!\left[\int_0^T \frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{u}}, t) \rVert^2 dt\right]
$$

$$
\text{s.t.} \quad \begin{cases} d\boldsymbol{X}_t^{\boldsymbol{u}} = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t)) dt + \sigma_t d\boldsymbol{B}_t \\ \boldsymbol{X}_0^{\boldsymbol{u}} \sim \pi_0, \quad \boldsymbol{X}_T^{\boldsymbol{u}} \sim \pi_T \end{cases}
$$

where $\mathcal{U} = \lbrace \boldsymbol{u} \in C^1(\mathbb{R}^d \times [0,T]; \mathbb{R}^d) \mid \exists C > 0, \; \boldsymbol{u}(\boldsymbol{x}, t) \le C(1 + \lVert \boldsymbol{x} \rVert) \rbrace$ is the set of all **feasible control drifts**. Solving yields the optimal pair $(\boldsymbol{u}^\star, \mathbb{P}^\star)$ defining the Schroedinger bridge between $\pi\_0$ and $\pi\_T$ relative to $\boldsymbol{f}$.

</div>

Equivalently, in terms of time-dependent marginal densities $p\_t \in \mathcal{P}(\mathbb{R}^d)$:

$$
\inf_{(\boldsymbol{u}, p_t)} \int_0^T \int_{\mathbb{R}^d} \frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{x}, t) \rVert^2 p_t(\boldsymbol{x}) d\boldsymbol{x} dt \quad \text{s.t.} \quad \begin{cases} \partial_t p_t = -\nabla \cdot (p_t(\boldsymbol{f} + \sigma_t \boldsymbol{u})) + \frac{\sigma_t^2}{2}\Delta p_t \\ p_0 = \pi_0, \quad p_T = \pi_T \end{cases}
$$

where the evolution of $p\_t$ satisfies the Controlled Fokker--Planck Equation derived in Section 2.4.

Since the constrained optimization requires minimizing with respect to both $\boldsymbol{u}(\boldsymbol{x}, t)$ and $p\_t(\boldsymbol{x})$ which are **coupled** via the Fokker--Planck PDE constraint, we introduce a **Lagrange multiplier** $\psi\_t$ to obtain optimality conditions.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.23</span><span class="math-callout__name">(Optimality Conditions for Dynamic Schroedinger Bridge)</span></p>

The pair of optimal state PDF and optimal control $(\boldsymbol{u}^\star, p\_t^\star)$ that minimize the Density Dynamic SB Problem is the solution to the pair of PDEs:

$$
\begin{cases} \partial_t \psi_t + \frac{\sigma_t^2}{2}\lVert \nabla\psi_t \rVert^2 + \langle \nabla\psi_t, \boldsymbol{f} \rangle = -\frac{\sigma_t^2}{2}\Delta\psi_t \\[4pt] \partial_t p_t^\star + \nabla \cdot (p_t^\star(\boldsymbol{f} + \sigma_t^2 \nabla\psi_t)) = \frac{\sigma_t^2}{2}\Delta p_t^\star \end{cases} \quad \text{s.t.} \quad \begin{cases} p_0^\star = \pi_0 \\ p_T^\star = \pi_T \end{cases}
$$

where $\psi\_t(\boldsymbol{x})$ is the Lagrange multiplier. The optimal control $\boldsymbol{u}^\star$ can also be written in terms of $\psi\_t$ as:

$$
\forall (\boldsymbol{x}, t) \in \mathbb{R}^d \times [0, T], \quad \boldsymbol{u}^\star(\boldsymbol{x}, t) = \sigma_t \nabla\psi_t(\boldsymbol{x})
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Optimality Conditions via Lagrangian)</summary>

The Lagrangian for the Density Dynamic SB Problem is:

$$
\mathcal{L}(p, \boldsymbol{u}, \psi) := \int_0^T \int_{\mathbb{R}^d} \left\lbrace \frac{1}{2}\lVert \boldsymbol{u} \rVert^2 p_t + \psi_t\!\left(\partial_t p_t + \nabla \cdot (p_t(\boldsymbol{f} + \sigma_t \boldsymbol{u})) - \frac{\sigma_t^2}{2}\Delta p_t\right) \right\rbrace d\boldsymbol{x} \, dt
$$

Moving derivatives from $p\_t$ to $\psi\_t$ via integration by parts:

- **Time derivative term** ($\star$): $\int \psi\_t \partial\_t p\_t \, d\boldsymbol{x} \, dt \to -\int p\_t \partial\_t \psi\_t \, d\boldsymbol{x} \, dt$ (plus boundary constants)
- **Divergence term** ($\diamondsuit$): $\int \psi\_t \nabla \cdot (p\_t \boldsymbol{v}) \, d\boldsymbol{x} \, dt \to -\int p\_t \nabla\psi\_t \cdot \boldsymbol{v} \, d\boldsymbol{x} \, dt$
- **Laplacian term** ($\blacktriangle$): $-\int \psi\_t \frac{\sigma\_t^2}{2}\Delta p\_t \, d\boldsymbol{x} \, dt \to -\int \frac{\sigma\_t^2}{2} p\_t \Delta\psi\_t \, d\boldsymbol{x} \, dt$ (two integrations by parts)

After substitution:

$$
\mathcal{L} = \int_0^T \int_{\mathbb{R}^d} \left[\frac{1}{2}\lVert \boldsymbol{u} \rVert^2 - \partial_t\psi_t - \nabla\psi_t \cdot (\boldsymbol{f} + \sigma_t \boldsymbol{u}) - \frac{\sigma_t^2}{2}\Delta\psi_t\right] p_t \, d\boldsymbol{x} \, dt
$$

**Minimizing over $\boldsymbol{u}$:** Isolating $\boldsymbol{u}$-dependent terms and completing the square:

$$
\inf_{\boldsymbol{u}} \left\lbrace \frac{1}{2}\lVert \boldsymbol{u} \rVert^2 - \sigma_t \boldsymbol{u}^\top \nabla\psi_t \right\rbrace = \frac{1}{2}\lVert \boldsymbol{u} - \sigma_t\nabla\psi_t \rVert^2 - \frac{\sigma_t^2}{2}\lVert \nabla\psi_t \rVert^2
$$

This is minimized at $\boldsymbol{u}^\star = \sigma\_t \nabla\psi\_t$. The expression within the brackets must vanish for optimality over arbitrary $p\_t$, yielding the **Hamilton--Jacobi--Bellman (HJB) equation**:

$$
\partial_t\psi_t + \frac{\sigma_t^2}{2}\lVert \nabla\psi_t \rVert^2 + \langle \nabla\psi_t, \boldsymbol{f} \rangle = -\frac{\sigma_t^2}{2}\Delta\psi_t
$$

Substituting $\boldsymbol{u}^\star = \sigma\_t \nabla\psi\_t$ into the Controlled Fokker--Planck equation gives the **FP equation** for $p\_t^\star$. Together they form the coupled HJB-FP system. $\square$
</details>
</div>

Although the coupled nonlinear PDE system in the HJB-FP System fully characterizes the **optimal control--density pair** $(\boldsymbol{u}^\star, p\_t^\star)$, solving this system remains challenging in general. A key step toward tractability is the **Hopf--Cole transform**, which converts the nonlinear system into an equivalent pair of **linear PDEs**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reference Processes for Schroedinger Bridges)</span></p>

Common reference processes $\mathbb{Q}$ used in SB formulations and generative modeling:

- **Ornstein--Uhlenbeck (OU) processes:** $d\boldsymbol{X}\_t = -\beta\boldsymbol{X}\_t dt + \sigma\_t d\boldsymbol{B}\_t$. The drift $-\beta\boldsymbol{X}\_t$ pulls the system toward the origin, creating **mean-reverting** stochastic processes that approach a stationary Gaussian distribution.

- **Variance Exploding SDEs (VESDEs):** $d\boldsymbol{X}\_t = \sqrt{d\sigma\_t^2/dt} \, d\boldsymbol{B}\_t$, $\boldsymbol{X}\_0 \sim \pi\_0$. The variance $\beta\_t := \int\_0^t \sigma\_s^2 ds$ increases over time, and the marginal density is $\mathcal{N}(\boldsymbol{0}, \beta\_t I\_d)$.

- **Variance Preserving SDEs (VPSDEs):** $d\boldsymbol{X}\_t = -\frac{1}{2}\beta\_t \boldsymbol{X}\_t dt + \sqrt{\beta\_t} \, d\boldsymbol{B}\_t$, $\boldsymbol{X}\_0 \sim \pi\_0$. The drift and diffusion terms are carefully balanced so the total variance remains approximately constant and the marginal follows $\mathcal{N}(\boldsymbol{0}, I\_d)$. These underlie the widely-adopted **denoising diffusion probabilistic model** (DDPM) framework.

</div>

### 2.8 Hopf--Cole Transform

The Hopf--Cole transform allows us to transform the **non-linear PDEs coupled over the full trajectory** into a system of **linear PDEs that are only coupled via their boundary constraints**. This is achieved with a change-of-variables from the optimal control--density pair $(\boldsymbol{u}^\star, p\_t^\star)$ to a pair of potential functions $(\varphi\_t, \hat{\varphi}\_t)$ that define a **coupled system of linear PDEs**.

Recall the HJB-FP System:

$$
\partial_t\psi_t + \frac{\sigma_t^2}{2}\lVert \nabla\psi_t \rVert^2 + \langle \nabla\psi_t, \boldsymbol{f} \rangle = -\frac{\sigma_t^2}{2}\Delta\psi_t \qquad \text{(HJB Equation)}
$$

$$
\partial_t p_t^\star + \nabla \cdot (p_t^\star(\boldsymbol{f} + \sigma_t^2 \nabla\psi_t)) = \frac{\sigma_t^2}{2}\Delta p_t^\star \qquad \text{(FP Equation)}
$$

The quadratic term $\frac{\sigma\_t^2}{2}\lVert \nabla\psi \rVert^2$ is the *only* non-linear term. The goal of the transform is: *how do we define a change-of-variables that makes this linear?*

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.24</span><span class="math-callout__name">(Hopf--Cole Transform)</span></p>

Given a reference process defined by the deterministic drift $\boldsymbol{f}(\boldsymbol{x}, t)$, diffusion coefficient $\sigma\_t$, and boundary marginal distributions $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$, we can apply the following change of variables $(\psi, p\_t^\star) \mapsto (\varphi\_t, \hat{\varphi}\_t)$ defined as:

$$
\psi_t(\boldsymbol{x}) = \log \varphi_t(\boldsymbol{x}), \quad p_t^\star(\boldsymbol{x}) = \varphi_t(\boldsymbol{x})\hat{\varphi}_t(\boldsymbol{x})
$$

which transforms the HJB Equation and FP Equation into a system of **linear PDEs** for $(\varphi\_t, \hat{\varphi}\_t)$:

$$
\begin{cases} \partial_t \varphi_t + \langle \nabla\varphi_t, \boldsymbol{f} \rangle = -\frac{\sigma_t^2}{2}\Delta\varphi_t \\[4pt] \partial_t \hat{\varphi}_t + \nabla \cdot (\hat{\varphi}_t \boldsymbol{f}) = \frac{\sigma_t^2}{2}\Delta\hat{\varphi}_t \end{cases} \quad \text{s.t.} \quad \begin{cases} p_0^\star = \varphi_0 \hat{\varphi}_0 \\ p_T^\star = \varphi_T \hat{\varphi}_T \end{cases}
$$

which define the dynamics and terminal conditions of $(\varphi\_t, \hat{\varphi}\_t)$. Furthermore, the optimal control can be written as:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = \sigma_t \nabla \log \varphi_t(\boldsymbol{x})
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Derivation (Hopf--Cole Transform)</summary>

We start with a well-known property of the Laplacian: **the Laplacian of a logarithm produces squared gradients**. Define the *ansatz* $\psi \mapsto \varphi$:

$$
\psi_t(\boldsymbol{x}) = C \log \varphi_t(\boldsymbol{x}) \iff \varphi_t(\boldsymbol{x}) = \exp\!\left(\frac{\psi_t(\boldsymbol{x})}{C}\right)
$$

**(i) The quadratic term becomes:** $\frac{\sigma\_t^2}{2}\lVert \nabla\psi\_t \rVert^2 = \frac{\sigma\_t^2 C^2}{2}\frac{\lVert \nabla\varphi\_t \rVert^2}{\varphi\_t^2}$

**(ii) The Laplacian term becomes:**

$$
-\frac{\sigma_t^2}{2}\Delta\psi_t = -\frac{\sigma_t^2}{2}C \nabla \cdot \!\left(\frac{\nabla\varphi_t}{\varphi_t}\right) = \frac{\sigma_t^2}{2}C\!\left(\frac{\lVert \nabla\varphi_t \rVert^2}{\varphi_t^2} - \frac{\Delta\varphi_t}{\varphi_t}\right)
$$

Setting the quadratic terms equal to determine $C$:

$$
\frac{\sigma_t^2 C^2}{2}\frac{\lVert \nabla\varphi \rVert^2}{\varphi^2} = \frac{\sigma_t^2 C}{2}\frac{\lVert \nabla\varphi \rVert^2}{\varphi^2} \implies C = 1
$$

With $C = 1$, we have $\psi\_t = \log\varphi\_t$. Substituting into the HJB equation, the quadratic terms cancel and dividing by $\varphi\_t$ yields:

$$
\partial_t \varphi_t + \langle \nabla\varphi_t, \boldsymbol{f} \rangle = -\frac{\sigma_t^2}{2}\Delta\varphi_t
$$

For the FP equation, substituting $p\_t^\star = \varphi\_t \hat{\varphi}\_t$ and $\boldsymbol{u}^\star = \sigma\_t \nabla\log\varphi\_t = \sigma\_t \frac{\nabla\varphi\_t}{\varphi\_t}$, then dividing by $\varphi\_t$ (using the $\varphi\_t$ equation to cancel terms) yields:

$$
\partial_t \hat{\varphi}_t + \nabla \cdot (\hat{\varphi}_t \boldsymbol{f}) = \frac{\sigma_t^2}{2}\Delta\hat{\varphi}_t \quad \square
$$

</details>
</div>

The Hopf--Cole PDEs are **linear** in $\varphi\_t$ and $\hat{\varphi}\_t$ individually, and are **only coupled through their boundary conditions** $p\_0^\star = \varphi\_0\hat{\varphi}\_0$ and $p\_T^\star = \varphi\_T\hat{\varphi}\_T$. The first equation for $\varphi\_t$ is a **backward** PDE (evolves backward from the terminal condition), while the second equation for $\hat{\varphi}\_t$ is a **forward** PDE (evolves forward from the initial condition). The functions $\varphi\_t$ and $\hat{\varphi}\_t$ are the time-evolving analogues of the static Schroedinger potentials $(\varphi, \hat{\varphi})$ from Section 1.4.

The Hopf--Cole PDEs **uniquely** define the optimal control--density pair of the Schroedinger bridge:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = \sigma_t \nabla\log\varphi_t(\boldsymbol{x}), \quad p_t^\star(\boldsymbol{x}) = \varphi_t(\boldsymbol{x})\hat{\varphi}_t(\boldsymbol{x})
$$

yielding the boundary constraints $p\_0^\star(\boldsymbol{x}) = \pi\_0(\boldsymbol{x}) = \varphi\_0(\boldsymbol{x})\hat{\varphi}\_0(\boldsymbol{x})$ and $p\_T^\star(\boldsymbol{x}) = \pi\_T(\boldsymbol{x}) = \varphi\_T(\boldsymbol{x})\hat{\varphi}\_T(\boldsymbol{x})$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.25</span><span class="math-callout__name">(Forward--Backward Schroedinger Potentials)</span></p>

Let $(\varphi\_t, \hat{\varphi}\_t)$ denote the Schroedinger potentials obtained through the Hopf--Cole transform, which define the solution to the Dynamic OT Problem as:

$$
p_t^\star(\boldsymbol{x}) = \varphi_t(\boldsymbol{x})\hat{\varphi}_t(\boldsymbol{x}), \quad \boldsymbol{u}^\star(\boldsymbol{x}, t) = \sigma_t \nabla\log\varphi_t(\boldsymbol{x})
$$

Then, $(\varphi\_t, \hat{\varphi}\_t)$ can be represented as the solution to a system of equations with the transition density under the reference path measure $\mathbb{Q}$:

$$
\begin{cases} \varphi_t(\boldsymbol{x}) = \int_{\mathbb{R}^d} \mathbb{Q}_{T \mid t}(\boldsymbol{y} \mid \boldsymbol{x}) \varphi_T(\boldsymbol{y}) d\boldsymbol{y} \\[4pt] \hat{\varphi}_t(\boldsymbol{x}) = \int_{\mathbb{R}^d} \mathbb{Q}_{t \mid 0}(\boldsymbol{x} \mid \boldsymbol{y}) \hat{\varphi}_0(\boldsymbol{y}) d\boldsymbol{y} \end{cases} \quad \text{s.t.} \quad \begin{cases} \pi_0(\boldsymbol{x}) = \varphi_0(\boldsymbol{x})\hat{\varphi}_0(\boldsymbol{x}) \\ \pi_T(\boldsymbol{x}) = \varphi_T(\boldsymbol{x})\hat{\varphi}_T(\boldsymbol{x}) \end{cases}
$$

subject to the boundary factorization constraints.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Forward--Backward Schroedinger Potentials)</summary>

The system of linear PDEs derived in Theorem 2.24 aligns with the form of the Kolmogorov Backward Equation defined in Corollary 2.16.

**Forward potential $\varphi\_t$:** The first linear PDE in the Hopf--Cole PDEs is equivalent to the backward Kolmogorov equation with $r(\boldsymbol{x}, t) := \varphi\_t(\boldsymbol{x})$ and $\Phi(\boldsymbol{x}) = \varphi\_T(\boldsymbol{x})$ associated with the reference process $\mathbb{Q}$. Therefore, $\varphi\_t$ admits the Feynman--Kac representation:

$$
\varphi_t(\boldsymbol{x}) = \mathbb{E}_{\boldsymbol{X}_{t:T} \sim \mathbb{Q}}[\varphi_T(\boldsymbol{X}_T) \mid \boldsymbol{X}_t = \boldsymbol{x}]
$$

Since the conditional density of $\boldsymbol{X}\_T$ given $\boldsymbol{X}\_t = \boldsymbol{x}$ is defined by the transition density $\mathbb{Q}\_{T \mid t}(\cdot \mid \boldsymbol{x})$, we can write:

$$
\varphi_t(\boldsymbol{x}) = \int_{\mathbb{R}^d} \mathbb{Q}_{T \mid t}(\boldsymbol{x}_T \mid \boldsymbol{x}) \varphi_T(\boldsymbol{x}_T) d\boldsymbol{x}_T
$$

**Backward potential $\hat{\varphi}\_t$:** Similarly, $\hat{\varphi}\_t$ satisfies the second linear PDE, which is equivalent to the backward Kolmogorov equation with terminal constraint $\Phi(\boldsymbol{x}) := \hat{\varphi}\_0(\boldsymbol{x})$. Therefore:

$$
\hat{\varphi}_t(\boldsymbol{x}) = \mathbb{E}_{\boldsymbol{X}_{0:t} \sim \mathbb{Q}}[\hat{\varphi}_0(\boldsymbol{X}_0) \mid \boldsymbol{X}_t = \boldsymbol{x}] = \int_{\mathbb{R}^d} \mathbb{Q}_{t \mid 0}(\boldsymbol{x} \mid \boldsymbol{x}_0) \hat{\varphi}_0(\boldsymbol{x}_0) d\boldsymbol{x}_0
$$

which are exactly the equations in the Schroedinger Potentials system. $\square$
</details>
</div>

This result shows that the forward potential $\varphi\_t$ propagates backward from the terminal constraint $\pi\_T = \varphi\_T\hat{\varphi}\_T$ via a linear integral operator over the expected distribution of $\varphi\_T(\boldsymbol{X}\_T)$, and the backward potential $\hat{\varphi}\_t$ propagates forward from the reversed terminal constraint $\pi\_0 = \varphi\_0\hat{\varphi}\_0$ via a linear integral operator in the reverse time.

The Schroedinger potentials $(\varphi\_t, \hat{\varphi}\_t)$ obtained through the Hopf--Cole transform are the **dynamic or continuous-time analogues of the Schroedinger potentials**. In the static formulation (Section 1.4), the optimal coupling under the reference kernel $K(\boldsymbol{x}, \boldsymbol{y}) = e^{-c(\boldsymbol{x}, \boldsymbol{y})}$ admits the factorized form:

$$
\pi_{0,T}^\star(\boldsymbol{x}, \boldsymbol{y}) = e^{\varphi(\boldsymbol{x}) + \hat{\varphi}(\boldsymbol{y}) - c(\boldsymbol{x}, \boldsymbol{y})} = e^{\varphi(\boldsymbol{x})} K(\boldsymbol{x}, \boldsymbol{y}) e^{\hat{\varphi}(\boldsymbol{y})}
$$

The Dynamic SB Problem generalizes this structure from couplings of endpoints to path measures of stochastic processes. The time-dependent potentials $(\varphi\_t, \hat{\varphi}\_t)$ propagate according to the forward--backward linear PDE system, and their product recovers the optimal marginal density at each time $p\_t^\star = \varphi\_t\hat{\varphi}\_t$.

As shown in Theorem 1.10, the static Schroedinger system admits a *unique solution up to an additive constant*. Since the solution to the dynamic SB problem $p\_t^\star$ is also unique by the strict convexity of the KL divergence, the product $\varphi\_t\hat{\varphi}\_t$ is unique, and the individual potentials $\varphi\_t$ and $\hat{\varphi}\_t$ are **unique up to a multiplicative constant** which leaves their product invariant.

### 2.9 Schroedinger Bridges as Entropy-Regularized Dynamic Optimal Transport

One of the most important perspectives on the Schroedinger bridge problem is its close relationship to optimal transport. In Section 1, we derived the static SB problem directly from the entropic OT problem, with a simple reparameterization of the reference coupling.

In this section, we make a similar connection to the **Benamou and Brenier (dynamic) formulation of the OT problem** defined in Section 2.1. We show that by reparameterizing the control drift in the controlled Fokker--Planck equation, the stochastic dynamics can be expressed as a deterministic continuity equation. Under this transformation, the dynamic SB objective decomposes into a kinetic transport energy that aligns with the dynamic OT objective, with additional **entropy-regularization terms** induced by diffusion.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.26</span><span class="math-callout__name">(Dynamic Optimal Transport Form of Schroedinger Bridge)</span></p>

Consider the dynamic Schroedinger bridge problem written in terms of the marginal density in the Controlled Fokker--Planck Equation:

$$
\inf_{(\boldsymbol{u}, p_t)} \int_0^T \int_{\mathbb{R}^d} \frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{x}, t) \rVert^2 p_t(\boldsymbol{x}) d\boldsymbol{x} dt \quad \text{s.t.} \quad \begin{cases} \partial_t p_t = -\nabla \cdot (p_t(\boldsymbol{f} + \sigma_t \boldsymbol{u})) + \frac{\sigma_t^2}{2}\Delta p_t \\ p_0 = \pi_0, \quad p_T = \pi_T \end{cases}
$$

By reparameterizing $\boldsymbol{v}(\boldsymbol{x}, t) := \boldsymbol{u}(\boldsymbol{x}, t) + \frac{\sigma\_t}{2}\nabla\log p\_t(\boldsymbol{x})$, the dynamic SB problem takes an **equivalent form of a dynamic optimal transport problem**:

$$
\inf_{\boldsymbol{v}} \mathbb{E}_{p_t} \int_0^T \left[\frac{1}{2}\lVert \boldsymbol{v}(\boldsymbol{x}, t) \rVert^2 + \frac{\sigma_t^2}{8}\lVert \nabla\log p_t(\boldsymbol{x}) \rVert^2 - \frac{1}{2}\langle \nabla\log p_t(\boldsymbol{x}), \boldsymbol{f}(\boldsymbol{x}, t) \rangle\right] dt
$$

$$
\text{s.t.} \quad \begin{cases} \partial_t p_t = -\nabla \cdot (p_t(\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t \boldsymbol{v}(\boldsymbol{x}, t))) \\ p_0 = \pi_0, \quad p_T = \pi_T \end{cases}
$$

where the constraint is now a **continuity equation** (deterministic transport).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Dynamic OT Form of Schroedinger Bridge)</summary>

**Step 1: Rewrite Fokker--Planck as Continuity Equation.** We absorb the Laplacian diffusion term into the divergence by defining a new velocity $\boldsymbol{v}(\boldsymbol{x}, t) := \boldsymbol{u}(\boldsymbol{x}, t) - \frac{\sigma\_t}{2}\nabla\log p\_t(\boldsymbol{x})$:

$$
\partial_t p_t = -\nabla \cdot (p_t(\boldsymbol{f} + \sigma_t \boldsymbol{u})) + \frac{\sigma_t^2}{2}\Delta p_t = -\nabla \cdot (p_t(\boldsymbol{f} + \sigma_t \boldsymbol{u}) - \frac{\sigma_t^2}{2}\nabla p_t)
$$

$$
= -\nabla \cdot \!\left(p_t\!\left(\boldsymbol{f} + \sigma_t\!\left(\boldsymbol{u} - \frac{\sigma_t}{2}\nabla\log p_t\right)\right)\right) = -\nabla \cdot (p_t(\boldsymbol{f} + \sigma_t \boldsymbol{v}))
$$

So $\boldsymbol{u}(\boldsymbol{x}, t) = \boldsymbol{v}(\boldsymbol{x}, t) + \frac{\sigma\_t}{2}\nabla\log p\_t(\boldsymbol{x})$.

**Step 2: Substitute into Objective.** Substituting $\boldsymbol{u} = \boldsymbol{v} + \frac{\sigma\_t}{2}\nabla\log p\_t$ into the Density Dynamic SB Objective:

$$
\int_0^T \int_{\mathbb{R}^d} \frac{1}{2}\lVert \boldsymbol{u} \rVert^2 p_t \, d\boldsymbol{x} \, dt = \int_0^T \int_{\mathbb{R}^d} \left[\underbrace{\frac{1}{2}\lVert \boldsymbol{v} \rVert^2}_{\text{kinetic energy}} + \underbrace{\frac{\sigma_t}{2}\langle \boldsymbol{v}, \nabla\log p_t \rangle}_{\text{cross term}} + \underbrace{\frac{\sigma_t^2}{8}\lVert \nabla\log p_t \rVert^2}_{\text{Fisher information}}\right] p_t \, d\boldsymbol{x} \, dt
$$

**Step 3: Eliminate the Cross Term.** The cross term $\frac{\sigma\_t}{2}\langle \boldsymbol{v}, \nabla\log p\_t \rangle p\_t$ measures the alignment between the velocity field $\boldsymbol{v}$ and the density gradient $\nabla p\_t$. Using the change in entropy $H(p\_t) = \int p\_t\log p\_t \, d\boldsymbol{x}$:

$$
H(p_T) - H(p_0) = \int_0^T \int_{\mathbb{R}^d} (1 + \log p_t)\partial_t p_t \, d\boldsymbol{x} \, dt
$$

Substituting the continuity equation $\partial\_t p\_t = -\nabla \cdot (p\_t(\boldsymbol{f} + \sigma\_t \boldsymbol{v}))$ and applying integration by parts for divergence:

$$
H(p_T) - H(p_0) = \int_0^T \int_{\mathbb{R}^d} \langle \nabla\log p_t, \boldsymbol{f} + \sigma_t \boldsymbol{v} \rangle p_t \, d\boldsymbol{x} \, dt
$$

Therefore:

$$
\int_0^T \int_{\mathbb{R}^d} \frac{\sigma_t}{2}\langle \boldsymbol{v}, \nabla\log p_t \rangle p_t \, d\boldsymbol{x} \, dt = \frac{1}{2}\underbrace{(H(p_T) - H(p_0))}_{\text{constant}} - \int_0^T \int_{\mathbb{R}^d} \frac{1}{2}\langle \nabla\log p_t, \boldsymbol{f} \rangle p_t \, d\boldsymbol{x} \, dt
$$

Since $p\_0 = \pi\_0$ and $p\_T = \pi\_T$ are fixed, the entropy difference is a constant and can be dropped from the objective. Substituting back yields the Entropy-Regularized Dynamic OT form. $\square$
</details>
</div>

This derivation yields an objective with three distinct terms that capture both the transport and diffusion in the original Fokker--Planck constraint:

1. **Kinetic energy term** $\frac{1}{2}\lVert \boldsymbol{v}(\boldsymbol{x}, t) \rVert^2$: measures the deterministic cost of transporting probability mass along the velocity field $\boldsymbol{v}(\boldsymbol{x}, t)$. This is exactly the kinetic energy appearing in the Dynamic OT Problem.

2. **Fisher information term** $\frac{\sigma\_t^2}{4}\lVert \nabla\log p\_t(\boldsymbol{x}) \rVert^2$: measures the sharpness of the marginal distribution. Smooth distributions yield low Fisher information, while sharp and concentrated distributions yield high Fisher information. This term appears from the diffusion in the SDE acting to smooth out the distribution.

3. **Cross term (drift interaction)** $\frac{1}{2}\langle \nabla\log p\_t(\boldsymbol{x}), \boldsymbol{f}(\boldsymbol{x}, t) \rangle$: measures the interaction between the reference drift $\boldsymbol{f}(\boldsymbol{x}, t)$ and the density evolution $\nabla\log p\_t$. This term reflects how the prior dynamics influence the evolution of the distribution by either aligning with or opposing the natural directions of increasing probability mass.

This shows that the Dynamic SB Problem can be explicitly rewritten as an entropy-regularized version of the Dynamic OT Problem, where the optimization over stochastic path measures $\mathbb{P}^{\boldsymbol{u}}$ reduces to an optimization over deterministic density flows $p\_t$ and velocity fields $\boldsymbol{v}(\boldsymbol{x}, t)$. In the limit of vanishing diffusion ($\sigma\_t \to 0$), the entropy regularization from the Fisher information and cross-term vanish, and the formulation recovers the classic Dynamic OT Problem.

### 2.10 Closing Remarks for Section 2

In this section, we took the crucial step of lifting the static formulation of the Schroedinger bridge (SB) problem to the space of continuous-time **path measures**. Starting from the dynamic optimal transport (OT) problem, we showed that the static OT problem with a quadratic cost function can be written in an equivalent *dynamic* form, which aims to find the optimal probability flow that smoothly transports mass between the prescribed marginals via a velocity field over a continuous time interval.

We extended the entropic OT problem to the space of **stochastic processes**, where path measures are determined by both a deterministic drift and random fluctuations in the form of Brownian motion, and entropy regularization is performed with a reference stochastic process.

Building on this framework, we analyzed the dynamic SB problem by expressing the KL minimization over path measures as a minimization of the **kinetic energy of a control drift**, corresponding to the minimal perturbation required to steer the reference SDE so that its marginals match the prescribed constraints. We leverage Lagrange multipliers to derive the optimality conditions --- a pair of **non-linear PDEs** (HJB-FP system) describing the forward and backward dynamics of the optimal solution $(\psi\_t, p\_t^\star)$. The Hopf--Cole transform linearizes these PDEs, revealing that the optimal SB dynamics can be factorized into the product of forward and backward potentials $p\_t^\star = \varphi\_t\hat{\varphi}\_t$ which solve a pair of linear PDEs.

While we have analyzed the dynamic SB problem through the lens of minimizing the KL divergence between stochastic path measures, it admits an alternative interpretation as an **optimal control problem**, where the goal is to determine the control drift that minimizes the expected future cost of steering a stochastic system toward a desired terminal distribution. This is precisely the idea of **stochastic optimal control** (SOC), which leverages Bellman's Principle of Optimality to define an optimal control as the minimizer to the expected *cost-to-go* from an intermediate state to all possible terminal states under an SDE.

## 3. Schroedinger Bridge Problem as Optimal Control

In this section, we reformulate the Schroedinger bridge (SB) problem through the lens of **stochastic optimal control (SOC)** theory, providing a dynamic and decision-theoretic perspective on entropy-regularized transport. Rather than directly optimizing over path measures, this view interprets the SB problem as learning an optimal control that steers a reference stochastic process between prescribed marginals while minimizing a control cost.

### 3.1 Stochastic Optimal Control

While the Schroedinger bridge problem aims to optimize the intermediate bridge between fixed endpoints, it is natural to consider an alternative variational perspective where given a particle sampled from $\boldsymbol{X}\_0 \sim \pi\_0$, we want to **optimize its path such that it reaches a state in the target distribution while minimally deviating from the reference SDE**. This is the key idea behind **stochastic optimal control (SOC)** theory, which seeks an **optimal control drift** $\boldsymbol{u}^\star(\boldsymbol{x}, t)$ that corrects the particle trajectory such that it takes the path of **minimal cost** toward the target distribution $\pi\_T$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.1</span><span class="math-callout__name">(Stochastic Optimal Control (SOC) Objective)</span></p>

Given a running cost $c(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ and a terminal cost $\Phi(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$, we consider the following **stochastic optimal control** (SOC) objective:

$$
\inf_{\boldsymbol{u}} \mathbb{E}_{\boldsymbol{X}_{0:T}^{\boldsymbol{u}} \sim \mathbb{P}^{\boldsymbol{u}}}\!\left[\int_0^T \left(\frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{u}}, t) \rVert^2 + c(\boldsymbol{X}_t^{\boldsymbol{u}}, t)\right) dt + \Phi(\boldsymbol{X}_T^{\boldsymbol{u}})\right]
$$

$$
\text{s.t.} \quad d\boldsymbol{X}_t^{\boldsymbol{u}} = (\boldsymbol{f}(\boldsymbol{X}_t^{\boldsymbol{u}}, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{u}}, t)) dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0^{\boldsymbol{u}} \sim \pi_0
$$

where $\boldsymbol{u}(\boldsymbol{x}, t)$ is the control drift, $\boldsymbol{f}(\boldsymbol{x}, t)$ is the drift of the reference process $\mathbb{Q}$, $\sigma\_t$ is the diffusion coefficient, and $\boldsymbol{B}\_t$ is $d$-dimensional Brownian motion.

</div>

Under this objective, we define the **cost functional** $J(\boldsymbol{x}, t; \boldsymbol{u})$ as the *cost-to-go* from any fixed point $(\boldsymbol{x}, t) \in \mathbb{R}^d \times [0, T]$ at time $t$ under the control $\boldsymbol{u}$ as the expected running cost and terminal cost of integrating the controlled SDE from $\boldsymbol{X}\_t^{\boldsymbol{u}} = \boldsymbol{x}$ over $s \in [t, T]$:

$$
J(\boldsymbol{x}, t; \boldsymbol{u}) := \mathbb{E}_{\boldsymbol{X}_{t:T}^{\boldsymbol{u}} \sim \mathbb{P}^{\boldsymbol{u}}}\!\left[\int_t^T \left(\frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_s^{\boldsymbol{u}}, s) \rVert^2 + c(\boldsymbol{X}_s^{\boldsymbol{u}}, s)\right) ds + \Phi(\boldsymbol{X}_T^{\boldsymbol{u}}) \;\middle|\; \boldsymbol{X}_t^{\boldsymbol{u}} = \boldsymbol{x}\right]
$$

Given the cost-to-go for an arbitrary control $\boldsymbol{u}$, we define the **value function** $V\_t(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$ as the *optimal cost-to-go* obtained with the optimal control $\boldsymbol{u}^\star$:

$$
V_t(\boldsymbol{x}) := J^\star(\boldsymbol{x}, t; \boldsymbol{u}^\star) := \inf_{\boldsymbol{u}} J(\boldsymbol{x}, t; \boldsymbol{u})
$$

which solves the **Hamilton--Jacobi--Bellman (HJB) equation**, similarly to the Lagrange multiplier $\psi\_t(\boldsymbol{x})$ from Section 2.7.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.2</span><span class="math-callout__name">(Bellman's Principle of Optimality)</span></p>

Let $V\_t(\boldsymbol{x})$ denote the value function or optimal cost-to-go of a stochastic control problem starting from state $\boldsymbol{x}$ at time $t$. Then, the optimal control satisfies **Bellman's principle of optimality**, which states that for all intermediate time steps $t \le \tau \le T$ the optimal cost-to-go is equal to the cost incurred over $[t, \tau]$ and the future cost-to-go $V\_\tau(\boldsymbol{X}\_\tau)$ starting from $\boldsymbol{X}\_\tau$ over $[\tau, T]$:

$$
V_t(\boldsymbol{x}) = \inf_{\boldsymbol{u}} \mathbb{E}_{\boldsymbol{X}_{t:T}^{\boldsymbol{u}} \sim \mathbb{P}^{\boldsymbol{u}}}\!\left[\int_t^\tau \left(\frac{1}{2}\lVert \boldsymbol{u} \rVert^2 + c\right) ds + V_\tau(\boldsymbol{X}_\tau^{\boldsymbol{u}}) \;\middle|\; \boldsymbol{X}_t^{\boldsymbol{u}} = \boldsymbol{x}\right]
$$

</div>

Intuitively, this means that in an optimally controlled process, regardless of the initial state $\boldsymbol{x}$ and changes over $[t, \tau]$, the remaining controlled process from $\boldsymbol{X}\_\tau$ follows the optimal control law for the state resulting from the *first* decision. Taking the infinitesimal limit where $\tau \to t$ and $\tau - t \to 0$ yields the **Hamilton--Jacobi--Bellman (HJB) equations**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.3</span><span class="math-callout__name">(Hamilton--Jacobi--Bellman (HJB) Equations)</span></p>

Given the infinitesimal generator $\mathcal{A}\_t$ of the **uncontrolled** SDE $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$ acting on the value function $V\_t(\boldsymbol{x})$ as:

$$
(\mathcal{A}_t V_t)(\boldsymbol{x}) := \langle \boldsymbol{f}(\boldsymbol{x}, t), \nabla V_t(\boldsymbol{x}) \rangle + \frac{\sigma_t^2}{2}\Delta V_t(\boldsymbol{x})
$$

Then, $V\_t(\boldsymbol{x})$ solves the **Hamilton--Jacobi--Bellman** equation:

$$
\partial_t V_t(\boldsymbol{x}) = -(\mathcal{A}_t V_t)(\boldsymbol{x}) + \frac{\sigma_t^2}{2}\lVert \nabla V_t(\boldsymbol{x}) \rVert^2 - c(\boldsymbol{x}, t), \quad V_T(\boldsymbol{x}) = \Phi(\boldsymbol{x})
$$

Furthermore, the **optimal control** is:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = -\sigma_t \nabla V_t(\boldsymbol{x})
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (HJB Equation)</summary>

From Bellman's Principle of Optimality, define $\tau := t + \Delta t$ as a small time step. Applying the Controlled Ito Formula with $\mathcal{A}\_t^{\boldsymbol{u}}$, the change in value function over $[t, t + \Delta t]$ expands as:

$$
\mathbb{E}[V_{t+\Delta t}(\boldsymbol{X}_{t+\Delta t}^{\boldsymbol{u}}) \mid \boldsymbol{X}_t^{\boldsymbol{u}} = \boldsymbol{x}] = V_t(\boldsymbol{x}) + (\partial_t + \mathcal{A}_t^{\boldsymbol{u}})V_t(\boldsymbol{x})\Delta t + o(\Delta t)
$$

Substituting into Bellman's equation:

$$
V_t(\boldsymbol{x}) = \inf_{\boldsymbol{u}} \left[\left(\frac{1}{2}\lVert \boldsymbol{u} \rVert^2 + c(\boldsymbol{x}, t)\right)\Delta t + V_t(\boldsymbol{x}) + (\partial_t + \mathcal{A}_t^{\boldsymbol{u}})V_t(\boldsymbol{x})\Delta t\right] + O(\Delta t^2)
$$

Subtracting $V\_t(\boldsymbol{x})$, dividing by $\Delta t$, and taking $\Delta t \to 0$:

$$
0 = \inf_{\boldsymbol{u}} \left[\frac{1}{2}\lVert \boldsymbol{u} \rVert^2 + c(\boldsymbol{x}, t) + (\partial_t + \mathcal{A}_t^{\boldsymbol{u}})V_t(\boldsymbol{x})\right]
$$

Completing the square for $\boldsymbol{u}$-dependent terms:

$$
\inf_{\boldsymbol{u}} \left\lbrace \frac{1}{2}\lVert \boldsymbol{u} \rVert^2 + \langle \nabla V_t, \sigma_t \boldsymbol{u} \rangle \right\rbrace = \inf_{\boldsymbol{u}} \left\lbrace \frac{1}{2}\lVert \boldsymbol{u} + \sigma_t\nabla V_t \rVert^2 - \frac{\sigma_t^2}{2}\lVert \nabla V_t \rVert^2 \right\rbrace \implies \boldsymbol{u}^\star = -\sigma_t\nabla V_t
$$

Plugging back and rewriting the controlled generator in terms of the uncontrolled generator recovers the HJB equation. $\square$
</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.4</span><span class="math-callout__name">(Hopf--Cole Transform for Value Function)</span></p>

Let $V\_t(\boldsymbol{x})$ be the value function that solves the SOC Objective and satisfies the Value HJB Equation. Then, we define the change of variables:

$$
V_t(\boldsymbol{x}) = -\log \varphi_t(\boldsymbol{x}) \iff \varphi_t(\boldsymbol{x}) = e^{-V_t(\boldsymbol{x})}
$$

where $\varphi\_t(\boldsymbol{x})$ satisfies the **linear PDE**:

$$
\partial_t \varphi_t(\boldsymbol{x}) + \langle \boldsymbol{f}(\boldsymbol{x}, t), \nabla\varphi_t(\boldsymbol{x}) \rangle + \frac{\sigma_t^2}{2}\Delta\varphi_t(\boldsymbol{x}) - c(\boldsymbol{x}, t)\varphi_t(\boldsymbol{x}) = 0, \quad \varphi_T(\boldsymbol{x}) = e^{-\Phi(\boldsymbol{x})}
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Hopf--Cole Transform for Value Function)</summary>

From the Value HJB Equation: $\partial\_t V\_t + (\mathcal{A}\_t V\_t) - \frac{\sigma\_t^2}{2}\lVert \nabla V\_t \rVert^2 + c = 0$, $V\_T = \Phi(\boldsymbol{x})$.

Expressing each term containing $V\_t$ in terms of $\varphi = e^{-V\_t}$:

- **Time Derivative:** $\partial\_t V\_t = -\frac{\partial\_t\varphi\_t}{\varphi\_t}$
- **Gradient Term:** $\nabla V\_t = -\frac{\nabla\varphi\_t}{\varphi\_t} \implies \frac{\sigma\_t^2}{2}\lVert \nabla V\_t \rVert^2 = \frac{\sigma\_t^2}{2}\frac{\lVert \nabla\varphi\_t \rVert^2}{\varphi\_t^2}$
- **Generator Term:** $\Delta V\_t = -\frac{\Delta\varphi\_t}{\varphi\_t} + \frac{\lVert \nabla\varphi\_t \rVert^2}{\varphi\_t^2}$, so $(\mathcal{A}\_t V\_t) = -\frac{\langle \boldsymbol{f}, \nabla\varphi\_t \rangle}{\varphi\_t} - \frac{\sigma\_t^2}{2}\frac{\Delta\varphi\_t}{\varphi\_t} + \frac{\sigma\_t^2}{2}\frac{\lVert \nabla\varphi\_t \rVert^2}{\varphi\_t^2}$

Substituting back into the HJB equation, the $\frac{\lVert \nabla\varphi\_t \rVert^2}{\varphi\_t^2}$ terms cancel, and multiplying by $-\varphi\_t$:

$$
\partial_t\varphi_t + \langle \boldsymbol{f}, \nabla\varphi_t \rangle + \frac{\sigma_t^2}{2}\Delta\varphi_t - c \, \varphi_t = 0
$$

The terminal condition $V\_T = \Phi(\boldsymbol{x})$ gives $\varphi\_T = e^{-\Phi(\boldsymbol{x})}$. $\square$
</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.5</span><span class="math-callout__name">(SB as SOC)</span></p>

Although the Schroedinger bridge problem imposes constraints on both the initial and terminal distributions, it can still be reformulated as an SOC problem. The key observation is that the initial distribution can be treated as the **starting distribution** of the controlled process, while the terminal constraint can be enforced through an appropriate terminal cost that penalizes deviations from the desired terminal marginal.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.6</span><span class="math-callout__name">(Forward--Backward Stochastic Differential Equations)</span></p>

The optimal controlled process can be defined with a pair of forward and backward SDEs (FBSDEs) which define the evolution of the uncontrolled process $\boldsymbol{X}\_{0:T}$ and the value function $V\_t(\boldsymbol{X}\_t)$:

$$
\begin{cases} d\boldsymbol{X}_t = \boldsymbol{f}(\boldsymbol{X}_t, t) dt + \sigma_t d\boldsymbol{B}_t \\[4pt] dV_t(\boldsymbol{X}_t) = \left(\frac{\sigma_t^2}{2}\lVert \nabla V_t(\boldsymbol{X}_t) \rVert^2 - c(\boldsymbol{X}_t, t)\right) dt + \nabla V_t(\boldsymbol{X}_t)^\top \sigma_t d\boldsymbol{B}_t \\[4pt] V_T(\boldsymbol{X}_T) = \Phi(\boldsymbol{X}_T) \end{cases}
$$

For the controlled process $(\boldsymbol{X}\_t^{\boldsymbol{u}})\_{t \in [0,T]}$, the FBSDEs become:

$$
\begin{cases} d\boldsymbol{X}_t^{\boldsymbol{u}} = (\boldsymbol{f}(\boldsymbol{X}_t^{\boldsymbol{u}}, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{u}}, t)) dt + \sigma_t d\boldsymbol{B}_t \\[4pt] dV_t(\boldsymbol{X}_t^{\boldsymbol{u}}) = \left(\frac{\sigma_t^2}{2}\lVert \nabla V_t(\boldsymbol{X}_t^{\boldsymbol{u}}) \rVert^2 - c(\boldsymbol{X}_t^{\boldsymbol{u}}, t) + \langle \sigma_t \boldsymbol{u}, \nabla V_t(\boldsymbol{X}_t^{\boldsymbol{u}}) \rangle\right) dt + \nabla V_t(\boldsymbol{X}_t^{\boldsymbol{u}})^\top \sigma_t d\boldsymbol{B}_t \\[4pt] V_T(\boldsymbol{X}_T^{\boldsymbol{u}}) = \Phi(\boldsymbol{X}_T^{\boldsymbol{u}}) \end{cases}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.7</span><span class="math-callout__name">(Optimal Control and Value Function)</span></p>

Given a reference path measure $\mathbb{Q}$, the relationship between the **cost functional** $J(\boldsymbol{x}, t; \boldsymbol{u})$ and the **value function** $V\_t(\boldsymbol{x})$ is:

$$
J(\boldsymbol{x}, t; \boldsymbol{u}) = V_t(\boldsymbol{x}) + \mathbb{E}_{\boldsymbol{X}_{t:T}^{\boldsymbol{u}} \sim \mathbb{P}^{\boldsymbol{u}}}\!\left[\int_t^T \frac{1}{2}\lVert \sigma_s \nabla V_s(\boldsymbol{X}_s^{\boldsymbol{u}}) + \boldsymbol{u}(\boldsymbol{X}_s^{\boldsymbol{u}}, s) \rVert^2 \;\middle|\; \boldsymbol{X}_t^{\boldsymbol{u}} = \boldsymbol{x}\right]
$$

where $\mathbb{Q}$ is defined by the **uncontrolled** SDE $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$. Furthermore, the optimal control satisfies:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = -\sigma_t \nabla V_t(\boldsymbol{x})
$$

since at optimality $V\_t(\boldsymbol{x}) := J^\star(\boldsymbol{x}, t; \boldsymbol{u}^\star)$, the expectation term must vanish: $\lVert \sigma\_s \nabla V\_s + \boldsymbol{u}^\star \rVert^2 = 0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.8</span><span class="math-callout__name">(Closed-Form Expression for Value Function)</span></p>

Given a reference path measure $\mathbb{Q}$, the **value function** $V\_t(\boldsymbol{x})$ can be derived independently of the optimal control $\boldsymbol{u}^\star$ as:

$$
V_t(\boldsymbol{x}) = -\log \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{Q}}\!\left[\exp\!\left(-\int_t^T c(\boldsymbol{X}_s, s) ds + \Phi(\boldsymbol{X}_T)\right) \;\middle|\; \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

where $\mathbb{Q}$ is defined by the **uncontrolled** SDE $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Closed-Form Value Function)</summary>

This proof combines the Value HJB Equation, the Hopf--Cole Value PDE, and the Feynman--Kac Formula. The value function solves $\partial\_t V\_t + \mathcal{A}\_t V\_t - \frac{1}{2}\lVert \sigma\_t^\top \nabla V\_t \rVert^2 + c = 0$, which we showed in Corollary 3.4 transforms to the linear PDE $\partial\_t\varphi\_t + \langle \boldsymbol{f}, \nabla\varphi\_t \rangle + \frac{\sigma\_t^2}{2}\Delta\varphi\_t - c\,\varphi\_t = 0$ with $\varphi\_T = e^{-\Phi(\boldsymbol{x})}$.

Applying the Feynman--Kac Formula (Theorem 2.15):

$$
\varphi_t(\boldsymbol{x}) = e^{-V_t(\boldsymbol{x})} = \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{Q}}\!\left[\exp\!\left(-\int_t^T c(\boldsymbol{X}_s, s) ds\right) e^{-\Phi(\boldsymbol{X}_T)} \;\middle|\; \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

Taking the logarithm and inverting: 

$$V_t(\boldsymbol{x}) = -\log \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{Q}}\!\left[\exp\!\left(-\int_t^T c(\boldsymbol{X}_s, s) ds - \Phi(\boldsymbol{X}_T)\right) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]$$

$\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.9</span><span class="math-callout__name">(Radon--Nikodym Derivative Between Optimal and Reference Path Measure)</span></p>

The **Radon--Nikodym Derivative** (RND) between the optimal controlled path measure $\mathbb{P}^\star$ and the reference path measure $\mathbb{Q}$ is:

$$
\frac{d\mathbb{P}^\star}{d\mathbb{Q}}(\boldsymbol{X}_{0:T}) = e^{-\Phi(\boldsymbol{X}_T) + V_0(\boldsymbol{X}_0) - \int_0^T c(\boldsymbol{X}_t, t) dt}
$$

which directly yields the optimal path measure:

$$
\mathbb{P}^\star(\boldsymbol{X}_{0:T}) = \frac{1}{Z}\mathbb{Q}(\boldsymbol{X}_{0:T}) e^{-\Phi(\boldsymbol{X}_T) + V_0(\boldsymbol{X}_0) - \int_0^T c(\boldsymbol{X}_t, t) dt}
$$

where $Z := \mathbb{E}\_{\mathbb{Q}}\!\left[e^{-\Phi(\boldsymbol{X}\_T) + V\_0(\boldsymbol{X}\_0) - \int\_0^T c(\boldsymbol{X}\_t, t) dt}\right]$ is the normalization constant. The joint endpoint law is:

$$
\mathbb{P}^\star(\boldsymbol{X}_0, \boldsymbol{X}_T) = \frac{1}{Z}\mathbb{Q}(\boldsymbol{X}_0, \boldsymbol{X}_T) e^{-\Phi(\boldsymbol{X}_T) + V_0(\boldsymbol{X}_0) - \int_0^T c(\boldsymbol{X}_t, t) dt}
$$

which means the endpoint law of the optimal path measure is an exponential tilting of the reference path measure by the initial and terminal value functions.

</div>

The expression for the Optimal Path Measure reveals the **key challenge** in solving the SOC problem: the optimal path measure is a *tilted* version of the reference path measure by not only the terminal cost $-V\_T(\boldsymbol{X}\_T)$ but also the initial value function $V\_0(\boldsymbol{X}\_0)$, which is defined as an expectation over stochastic paths (Proposition 3.8) and is **intractable**. This motivates the definition of **memoryless reference processes**, which removes the dependency of the optimal path on the initial value function.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Memoryless Reference Processes)</span></p>

Consider a reference process $\mathbb{Q}$ where the joint distribution of initial and terminal states is **independent**, such that $\mathbb{Q}(\boldsymbol{X}\_0, \boldsymbol{X}\_T) = q\_0(\boldsymbol{X}\_0)q\_T(\boldsymbol{X}\_T)$. Then the terminal distribution under $\mathbb{P}^\star$ satisfies (with $c \equiv 0$):

$$
p_T^\star(\boldsymbol{X}_T) = \frac{1}{Z} q_T(\boldsymbol{X}_T) e^{-\Phi(\boldsymbol{X}_T)} \underbrace{\int_{\mathbb{R}^d} q_0(\boldsymbol{X}_0) e^{V_0(\boldsymbol{X}_0)} d\boldsymbol{X}_0}_{\text{constant normalization}}
$$

so the initial value function $V\_0(\boldsymbol{X}\_0)$ integrates to a constant and can be eliminated. Examples include:

1. **Variance-preserving SDEs:** $d\boldsymbol{X}\_t = -\frac{1}{2}\beta\_t \boldsymbol{X}\_t dt + \sqrt{\beta\_t} d\boldsymbol{B}\_t$ with $\boldsymbol{X}\_0 \sim \mathcal{N}(\boldsymbol{0}, I\_d)$, giving 
 
   $$\mathbb{Q}(\boldsymbol{X}_0, \boldsymbol{X}_T) \approx q_0(\boldsymbol{X}_0) q_T(\boldsymbol{X}_T)$$

2. **Brownian motion with Dirac initial condition:** $d\boldsymbol{X}\_t = \sigma\_t d\boldsymbol{B}\_t$ with $\boldsymbol{X}\_0 = \boldsymbol{0}$, giving 
   
   $$\mathbb{Q}(\boldsymbol{X}_0, \boldsymbol{X}_T) = \delta_0(\boldsymbol{X}_0) q_T(\boldsymbol{X}_T)$$

</div>

### 3.2 Schroedinger Bridges with Stochastic Optimal Control

In this section, we adapt SOC theory for solving the Dynamic SB Problem with arbitrary prior dynamics and initial distributions. The Hopf--Cole transform from Theorem 2.24 expresses the optimal control $\boldsymbol{u}^\star$ and probability density $p\_t^\star$ using a pair of **forward-backward SB potentials** $(\varphi\_t, \hat{\varphi}\_t)$:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = \sigma_t \nabla\log\varphi_t(\boldsymbol{x}), \quad p_t^\star(\boldsymbol{x}) = \varphi_t(\boldsymbol{x})\hat{\varphi}_t(\boldsymbol{x})
$$

Since the optimal control can be written **two equivalent ways** --- using the **value function** $V\_t(\boldsymbol{x})$ (Proposition 3.7) *or* the **Schroedinger potential** $\varphi\_t(\boldsymbol{x})$ --- we derive the **relationship between the value function and the SB potential**:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = -\sigma_t \nabla V_t(\boldsymbol{x}) = \sigma_t \nabla\log\varphi_t(\boldsymbol{x}) \implies V_t(\boldsymbol{x}) = -\log\varphi_t(\boldsymbol{x})
$$

Since the terminal cost is $\Phi(\boldsymbol{x}) = V\_T(\boldsymbol{x})$, we can express it in terms of the SB potentials:

$$
\Phi(\boldsymbol{x}) = V_T(\boldsymbol{x}) = -\log\varphi_T(\boldsymbol{x}) = -\log\frac{\pi_T(\boldsymbol{x})}{\hat{\varphi}_T(\boldsymbol{x})} = \log\frac{\hat{\varphi}_T(\boldsymbol{x})}{\pi_T(\boldsymbol{x})}
$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.10</span><span class="math-callout__name">(Schroedinger Bridge with Stochastic Optimal Control (SB-SOC))</span></p>

Given the Schroedinger potentials $(\varphi\_t, \hat{\varphi}\_t)$ that satisfy the linear PDEs in the Hopf--Cole PDEs, the SOC Objective can be expressed in terms of $(\varphi\_t, \hat{\varphi}\_t)$ as:

$$
\inf_{\boldsymbol{u}} \mathbb{E}_{\boldsymbol{X}_{0:T}^{\boldsymbol{u}} \sim \mathbb{P}^{\boldsymbol{u}}}\!\left[\int_0^T \frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{u}}, t) \rVert^2 dt + \log\frac{\hat{\varphi}_T(\boldsymbol{X}_T^{\boldsymbol{u}})}{\pi_T(\boldsymbol{X}_T^{\boldsymbol{u}})}\right]
$$

$$
\text{s.t.} \quad d\boldsymbol{X}_t^{\boldsymbol{u}} = (\boldsymbol{f}(\boldsymbol{X}_t^{\boldsymbol{u}}, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{u}}, t)) dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0^{\boldsymbol{u}} \sim \pi_0
$$

where $\boldsymbol{u}(\boldsymbol{x}, t)$ is the control drift, $\boldsymbol{f}(\boldsymbol{x}, t)$ is the drift of the reference process $\mathbb{Q}$, $\sigma\_t$ is the diffusion coefficient, and $d\boldsymbol{B}\_t$ is $d$-dimensional Brownian motion.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.11</span><span class="math-callout__name">(Schroedinger Potential Eliminates the Initial Value Bias)</span></p>

The SB-SOC Objective does not require computing the initial value function $V\_0(\boldsymbol{x})$. In particular, under the optimal control $\boldsymbol{u}^\star$, the induced path measure $\mathbb{P}^\star$ automatically satisfies the terminal marginal constraint $p\_T^\star(\boldsymbol{x}\_T) = \pi\_T(\boldsymbol{x}\_T)$ independently of the initial value function $V\_0(\boldsymbol{x})$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Schroedinger Potential Eliminates Initial Value Bias)</summary>

From the Optimal SB Path Measure, the optimal endpoint law is:

$$
\mathbb{P}^\star(\boldsymbol{X}_0, \boldsymbol{X}_T) = \frac{1}{Z}\mathbb{Q}(\boldsymbol{X}_0, \boldsymbol{X}_T)\exp\!\left(-\log\frac{\hat{\varphi}_T(\boldsymbol{X}_T)}{\pi_T(\boldsymbol{X}_T)} - \log\varphi_0(\boldsymbol{X}_0)\right)
$$

Integrating over $\boldsymbol{X}\_0$:

$$
p_T^\star(\boldsymbol{X}_T) = \frac{\pi_T(\boldsymbol{X}_T)}{\hat{\varphi}_T(\boldsymbol{X}_T)} \int_{\mathbb{R}^d} \mathbb{Q}(\boldsymbol{X}_0, \boldsymbol{X}_T) \frac{\hat{\varphi}_0(\boldsymbol{X}_0)}{\pi_0(\boldsymbol{X}_0)} \pi_0(\boldsymbol{X}_0) d\boldsymbol{X}_0
$$

Using the factorization $\mathbb{Q}(\boldsymbol{X}\_0, \boldsymbol{X}\_T) = \mathbb{Q}(\boldsymbol{X}\_T \mid \boldsymbol{X}\_0)\pi\_0(\boldsymbol{X}\_0)$ and the boundary condition $\pi\_0(\boldsymbol{x}) = \varphi\_0(\boldsymbol{x})\hat{\varphi}\_0(\boldsymbol{x})$:

$$
p_T^\star(\boldsymbol{X}_T) = \frac{\pi_T(\boldsymbol{X}_T)}{\hat{\varphi}_T(\boldsymbol{X}_T)} \underbrace{\int_{\mathbb{R}^d} \mathbb{Q}(\boldsymbol{X}_T \mid \boldsymbol{X}_0) \hat{\varphi}_0(\boldsymbol{X}_0) d\boldsymbol{X}_0}_{= \hat{\varphi}_T(\boldsymbol{X}_T)} = \pi_T(\boldsymbol{X}_T)
$$

where the last equality uses the backward potential definition from Corollary 2.25. $\square$
</details>
</div>

We can now characterize the full optimal path measure induced by the optimal control $\boldsymbol{u}^\star$. The Schroedinger potentials $(\varphi\_t, \hat{\varphi}\_t)$ provide a clean factorization of the optimal bridge measure with respect to the reference process $\mathbb{Q}$, revealing how the endpoint potentials reweight trajectories of the reference dynamics to produce the Schroedinger bridge.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.12</span><span class="math-callout__name">(Optimal Path Density of SB-SOC)</span></p>

The Schroedinger bridge path measure that solves the SB-SOC Objective can be written with respect to the SB potentials as:

$$
\mathbb{P}^\star(\boldsymbol{X}_{0:T}) = \frac{1}{Z}\mathbb{Q}(\boldsymbol{X}_{0:T})\varphi_T(\boldsymbol{X}_T)\frac{\hat{\varphi}_0(\boldsymbol{X}_0)}{\pi_0(\boldsymbol{X}_0)} = \frac{1}{Z}\mathbb{Q}(\boldsymbol{X}_{0:T} \mid \boldsymbol{X}_0)\varphi_T(\boldsymbol{X}_T)\hat{\varphi}_0(\boldsymbol{X}_0)
$$

and the marginal density at time $t$ factorizes as:

$$
p_t^\star(\boldsymbol{x}) = \hat{\varphi}_t(\boldsymbol{x})\varphi_t(\boldsymbol{x})
$$

Additionally, for any $s \le t$, the joint density of $\boldsymbol{X}\_s = \boldsymbol{y}$ and $\boldsymbol{X}\_t = \boldsymbol{x}$ satisfies:

$$
p_{s,t}^\star(\boldsymbol{y}, \boldsymbol{x}) = \mathbb{Q}(\boldsymbol{X}_t = \boldsymbol{x} \mid \boldsymbol{X}_s = \boldsymbol{y})\hat{\varphi}_s(\boldsymbol{y})\varphi_t(\boldsymbol{x}), \quad s \le t
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Optimal Path Density of SB-SOC)</summary>

Using the alternative definition of the value function, we derive the optimal SB path measure. From Proposition 3.9, the optimal path measure is:

$$
\mathbb{P}^\star(\boldsymbol{X}_{0:T}) = \frac{1}{Z}\mathbb{Q}(\boldsymbol{X}_{0:T})\exp(-V_T(\boldsymbol{X}_T) + V_0(\boldsymbol{X}_0))
$$

Since the dynamic SB problem has no running cost ($c \equiv 0$), substituting the SB Terminal Cost $V\_T = -\log\varphi\_T$ and the SB Value Function $V\_0 = -\log\varphi\_0$:

$$
\mathbb{P}^\star(\boldsymbol{X}_{0:T}) = \frac{1}{Z}\mathbb{Q}(\boldsymbol{X}_{0:T})\varphi_T(\boldsymbol{X}_T)\frac{1}{\varphi_0(\boldsymbol{X}_0)} = \frac{1}{Z}\mathbb{Q}(\boldsymbol{X}_{0:T})\varphi_T(\boldsymbol{X}_T)\frac{\hat{\varphi}_0(\boldsymbol{X}_0)}{\pi_0(\boldsymbol{X}_0)}
$$

where we use $\pi\_0 = \varphi\_0\hat{\varphi}\_0$. Using the identity $\mathbb{Q}(\boldsymbol{X}\_{0:T}) = \mathbb{Q}(\boldsymbol{X}\_{0:T} \mid \boldsymbol{X}\_0)\pi\_0(\boldsymbol{X}\_0)$, we get the conditional form.

**Marginal density:** To obtain $p\_t^\star$, we integrate over all paths $\boldsymbol{X}\_{0:T}$ where $\boldsymbol{X}\_t = \boldsymbol{x}$:

$$
p_t^\star(\boldsymbol{x}) = \int \mathbb{Q}(\boldsymbol{X}_T, \boldsymbol{X}_t = \boldsymbol{x} \mid \boldsymbol{X}_0)\varphi_T(\boldsymbol{X}_T)\hat{\varphi}_0(\boldsymbol{X}_0) \, d\boldsymbol{X}_0 \, d\boldsymbol{X}_T
$$

By the Markov property, this factorizes into:

$$
p_t^\star(\boldsymbol{x}) = \underbrace{\left(\int \mathbb{Q}(\boldsymbol{X}_T \mid \boldsymbol{X}_t = \boldsymbol{x})\varphi_T(\boldsymbol{X}_T) d\boldsymbol{X}_T\right)}_{= \varphi_t(\boldsymbol{x})} \underbrace{\left(\int \mathbb{Q}(\boldsymbol{X}_t = \boldsymbol{x} \mid \boldsymbol{X}_0)\hat{\varphi}_0(\boldsymbol{X}_0) d\boldsymbol{X}_0\right)}_{= \hat{\varphi}_t(\boldsymbol{x})} = \hat{\varphi}_t(\boldsymbol{x})\varphi_t(\boldsymbol{x})
$$

**Joint density:** For $s \le t$, conditioning additionally on $\boldsymbol{X}\_s = \boldsymbol{y}$ and applying the Markov property similarly:

$$
p_{s,t}^\star(\boldsymbol{y}, \boldsymbol{x}) = \mathbb{Q}(\boldsymbol{X}_t = \boldsymbol{x} \mid \boldsymbol{X}_s = \boldsymbol{y}) \underbrace{\left(\int \mathbb{Q}(\boldsymbol{X}_T \mid \boldsymbol{X}_t = \boldsymbol{x})\varphi_T d\boldsymbol{X}_T\right)}_{= \varphi_t(\boldsymbol{x})} \underbrace{\left(\int \mathbb{Q}(\boldsymbol{X}_s = \boldsymbol{y} \mid \boldsymbol{X}_0)\hat{\varphi}_0 d\boldsymbol{X}_0\right)}_{= \hat{\varphi}_s(\boldsymbol{y})}
$$

$\square$
</details>
</div>

Together, these results show that the SOC formulation of the Schroedinger bridge can be fully characterized by the reference dynamics $\mathbb{Q}$ and the terminal Schroedinger potentials $\varphi\_0$ and $\hat{\varphi}\_T$, with the intermediate potentials defined via the Schroedinger Potentials system (Corollary 2.25) without the need to explicitly compute the value function.

### 3.3 Objectives for Solving the SOC Problem

In this section, we introduce three different objective functions that can be used to solve the SOC problem defined in the SOC Objective (Definition 3.1): the **relative-entropy loss** $\mathcal{L}\_{\text{RE}}$, the **log-variance loss** $\mathcal{L}\_{\text{LV}}$, and the **cross-entropy loss** $\mathcal{L}\_{\text{CE}}$.

Since the SOC objective is inherently a KL divergence between the controlled path measure $\mathbb{P}^{\boldsymbol{u}}$ and the optimal path measure $\mathbb{P}^\star := \mathbb{P}^{\boldsymbol{u}^\star}$, it is natural to consider either the **forward KL divergence** or the **reverse KL divergence** between $\mathbb{P}^{\boldsymbol{u}}$ and $\mathbb{P}^\star$:

$$
\text{KL}(\mathbb{P}^{\boldsymbol{u}} \| \mathbb{P}^\star) = \mathbb{E}_{\mathbb{P}^{\boldsymbol{u}}}\!\left[\log \frac{d\mathbb{P}^{\boldsymbol{u}}}{d\mathbb{P}^\star}\right], \quad \text{KL}(\mathbb{P}^\star \| \mathbb{P}^{\boldsymbol{u}}) = \mathbb{E}_{\mathbb{P}^\star}\!\left[\log \frac{d\mathbb{P}^\star}{d\mathbb{P}^{\boldsymbol{u}}}\right]
$$

which both yield a **unique minimizer** at $\mathbb{P}^{\boldsymbol{u}} = \mathbb{P}^\star$. These two divergences define the **relative-entropy** (RE) and **cross-entropy** (CE) losses.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.13</span><span class="math-callout__name">(Relative Entropy (RE) Loss)</span></p>

The **relative entropy** (RE) loss between the controlled path measure $\mathbb{P}^{\boldsymbol{u}}$ and the optimal path measure $\mathbb{P}^\star$ is defined as:

$$
\mathcal{L}_{\text{RE}}(\mathbb{P}^{\boldsymbol{u}}, \mathbb{P}^\star) := \text{KL}(\mathbb{P}^{\boldsymbol{u}} \| \mathbb{P}^\star) = \mathbb{E}_{\mathbb{P}^{\boldsymbol{u}}}\!\left[\log \frac{d\mathbb{P}^{\boldsymbol{u}}}{d\mathbb{P}^\star}\right]
$$

Let $\boldsymbol{u}(\boldsymbol{x}, t)$ denote the control that generates $\mathbb{P}^{\boldsymbol{u}}$ and let $\boldsymbol{X}\_{0:T}^{\boldsymbol{u}} = (\boldsymbol{X}\_t^{\boldsymbol{u}})\_{t \in [0,T]}$ denote a stochastic process under the Controlled SDE. Then, the RE loss takes the path-integral form:

$$
\mathcal{L}_{\text{RE}}(\boldsymbol{u}) := \mathbb{E}_{\boldsymbol{X}_{0:T}^{\boldsymbol{u}} \sim \mathbb{P}^{\boldsymbol{u}}}\!\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{u}}, t) \rVert^2 dt + \Phi(\boldsymbol{X}_T^{\boldsymbol{u}}) + \int_0^T c(\boldsymbol{X}_t^{\boldsymbol{u}}, t) dt\right]
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Derivation (RE Loss)</summary>

We leverage Girsanov's theorem and the Radon--Nikodym derivative. Since $\mathbb{P}^{\boldsymbol{u}}$ is the *controlled* version of the reference process $\mathbb{Q}$, we write:

$$
\mathbb{E}_{\mathbb{P}^{\boldsymbol{u}}}\!\left[\log \frac{d\mathbb{P}^{\boldsymbol{u}}}{d\mathbb{P}^\star}\right] = \mathbb{E}_{\mathbb{P}^{\boldsymbol{u}}}\!\left[\log \frac{d\mathbb{P}^{\boldsymbol{u}}}{d\mathbb{Q}} - \log \frac{d\mathbb{P}^\star}{d\mathbb{Q}}\right]
$$

Since $\mathbb{P}^{\boldsymbol{u}}$ and $\mathbb{Q}$ differ only by the control drift $\sigma\_t\boldsymbol{u}(\boldsymbol{x}, t)$, the first term via the Path RND (Theorem 2.20) is $-\frac{1}{2}\int\_0^T \lVert \boldsymbol{u} \rVert^2 dt + \int\_0^T \boldsymbol{u}^\top d\boldsymbol{B}\_t$. The second term via the Optimal Path RND (Proposition 3.9) is $-V\_T(\boldsymbol{X}\_T) + V\_0(\boldsymbol{X}\_0) - \int\_0^T c \, dt$. Since $V\_T = \Phi$ and $\boldsymbol{X}\_0 \sim \pi\_0$ is fixed making $V\_0(\boldsymbol{X}\_0)$ a constant, and the Ito integral vanishes under expectation:

$$
\mathcal{L}_{\text{RE}}(\boldsymbol{u}) = \mathbb{E}_{\boldsymbol{X}_{0:T}^{\boldsymbol{u}} \sim \mathbb{P}^{\boldsymbol{u}}}\!\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u} \rVert^2 dt + \Phi(\boldsymbol{X}_T^{\boldsymbol{u}}) + \int_0^T c(\boldsymbol{X}_t^{\boldsymbol{u}}, t) dt\right] \quad \square
$$

</details>
</div>

While the RE loss has a unique minimizer when $\mathbb{P}^{\boldsymbol{u}} = \mathbb{P}^\star$, optimizing $\nabla\_{\boldsymbol{u}} \mathcal{L}\_{\text{RE}}(\boldsymbol{u})$ requires differentiating through the full SDE trajectories $(\boldsymbol{X}\_t^{\boldsymbol{u}})\_{t \in [0,T]}$ due to the expectation over $\mathbb{P}^{\boldsymbol{u}}$. In practice, this requires storing the full computational graph at each simulation step, which is memory-intensive.

To obtain a practical estimator, the **REINFORCE trick** allows us to rewrite the RE objective as an expectation under a stop-gradient sampling measure $\mathbb{P}^{\bar{\boldsymbol{u}}}$, where $\bar{\boldsymbol{u}} := \text{stopgrad}(\boldsymbol{u})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.14</span><span class="math-callout__name">(REINFORCE Relative Entropy (RERF) Loss)</span></p>

The **REINFORCE relative entropy** (RERF) loss is defined as an expectation over $\mathbb{P}^{\bar{\boldsymbol{u}}}$, where $\bar{\boldsymbol{u}} := \text{stopgrad}(\boldsymbol{u})$ is the non-gradient-tracking controlled generator:

$$
\mathcal{L}_{\text{RERF}}(\mathbb{P}^{\boldsymbol{u}}, \mathbb{P}^\star) := \mathbb{E}_{\mathbb{P}^{\bar{\boldsymbol{u}}}}\!\left[\log \frac{d\mathbb{P}^{\boldsymbol{u}}}{d\mathbb{P}^{\bar{\boldsymbol{u}}}}\left(\log \frac{d\mathbb{P}^{\boldsymbol{u}}}{d\mathbb{P}^\star} + C\right)\right]
$$

where $C \in \mathbb{R}$ is any constant. Crucially, the gradient aligns with the gradient of the relative-entropy loss: $\nabla\_{\boldsymbol{u}} \text{KL}(\mathbb{P}^{\boldsymbol{u}} \| \mathbb{P}^\star) = \nabla\_{\boldsymbol{u}} \mathcal{L}\_{\text{RERF}}(\mathbb{P}^{\boldsymbol{u}}, \mathbb{P}^\star)$.

</div>

While the RERF Loss provides an unbiased estimator of the RE gradient, it should be interpreted as a computational surrogate rather than a true loss function, as decreasing its value does not necessarily correspond to a monotonic reduction of the KL divergence itself.

An alternative objective that is a true loss function and doesn't require an expectation over $\mathbb{P}^{\boldsymbol{u}}$ is the **cross-entropy (CE) loss**, which is simply the *reverse KL divergence*, where the expectation is over the fixed optimal path measure $\mathbb{P}^\star$ instead of $\mathbb{P}^{\boldsymbol{u}}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.15</span><span class="math-callout__name">(Cross Entropy (CE) Loss)</span></p>

The **cross entropy** (CE) loss between the controlled path measure $\mathbb{P}^{\boldsymbol{u}}$ and the optimal path measure $\mathbb{P}^\star$ is defined as:

$$
\mathcal{L}_{\text{CE}}(\mathbb{P}^{\boldsymbol{u}}, \mathbb{P}^\star) := \text{KL}(\mathbb{P}^\star \| \mathbb{P}^{\boldsymbol{u}}) = \mathbb{E}_{\mathbb{P}^\star}\!\left[\log \frac{d\mathbb{P}^\star}{d\mathbb{P}^{\boldsymbol{u}}}\right]
$$

Let $\boldsymbol{v}(\boldsymbol{x}, t)$ denote an arbitrary **fixed** control drift that generates $\mathbb{P}^{\boldsymbol{v}}$ and let $\boldsymbol{X}\_{0:T}^{\boldsymbol{v}} = (\boldsymbol{X}\_t^{\boldsymbol{v}})\_{t \in [0,T]}$ denote the stochastic process under $\mathbb{P}^{\boldsymbol{v}}$. Then, the CE loss takes the path-integral form:

$$
\mathcal{L}_{\text{CE}}(\boldsymbol{u}) = \frac{1}{Z}\mathbb{E}_{\boldsymbol{X}_{0:T}^{\boldsymbol{v}} \sim \mathbb{P}^{\boldsymbol{v}}}\!\left[\underbrace{\exp\!\left(-g(\boldsymbol{X}_T^{\boldsymbol{v}}) - \int_0^T c(\boldsymbol{X}_t^{\boldsymbol{v}}, t) dt - \frac{1}{2}\int_0^T \lVert \boldsymbol{v} \rVert^2 dt - \int_0^T \boldsymbol{v}^\top d\boldsymbol{B}_t^{\boldsymbol{v}}\right)}_{(\star)} \cdot \left(\frac{1}{2}\int_0^T \lVert \boldsymbol{u} \rVert^2 dt - \int_0^T (\boldsymbol{u} \cdot \boldsymbol{v}) dt - \int_0^T \boldsymbol{u}^\top d\boldsymbol{B}_t^{\boldsymbol{v}} - g(\boldsymbol{X}_T^{\boldsymbol{v}}) - \int_0^T c \, dt\right)\right] + C
$$

where $g(\boldsymbol{X}\_T) = \Phi(\boldsymbol{X}\_T)$ is the terminal cost.

</div>

Since the paths $\boldsymbol{X}\_{0:T}^{\boldsymbol{v}}$ are generated under the fixed controlled process $\mathbb{P}^{\boldsymbol{v}}$ instead of the path that is being optimized $\mathbb{P}^{\boldsymbol{u}}$, it is considered an **off-policy objective**. This admits a much more computationally tractable gradient $\nabla\_{\boldsymbol{u}} \mathcal{L}\_{\text{CE}}(\boldsymbol{u})$ as it no longer depends on the SDE trajectories and does not require differentiating through or maintaining the computational graph of the SDE solver.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.16</span><span class="math-callout__name">(Convexity of the Cross-Entropy Objective)</span></p>

The cross-entropy loss $\mathcal{L}\_{\text{CE}}(\mathbb{P}^{\boldsymbol{u}}, \mathbb{P}^\star) = \text{KL}(\mathbb{P}^\star \| \mathbb{P}^{\boldsymbol{u}})$ is **convex in the controlled path measure** $\mathbb{P}^{\boldsymbol{u}}$. This follows from the fact that $\mathbb{P}^{\boldsymbol{u}} \mapsto \log \mathbb{P}^{\boldsymbol{u}}$ is convex when the reference distribution $\mathbb{P}^\star$ is fixed. Therefore, minimizing the CE objective corresponds to a convex optimization problem in the space of path measures, and the global minimum is achieved when $\mathbb{P}^{\boldsymbol{u}} = \mathbb{P}^\star$.

</div>

Now we consider an alternative class of objectives that aims to minimize the **variance** between the controlled and optimal path measures. A stochastic process $\boldsymbol{X}\_{0:T}^{\boldsymbol{u}} \sim \mathbb{P}^{\boldsymbol{u}}$ can be *reweighted* by the Radon--Nikodym derivative to match the optimal path measure $\mathbb{P}^\star$:

$$
\mathbb{P}^\star(\boldsymbol{X}_{0:T}^{\boldsymbol{u}}) = \frac{d\mathbb{P}^\star}{d\mathbb{P}^{\boldsymbol{u}}}(\boldsymbol{X}_{0:T}^{\boldsymbol{u}}) \cdot \mathbb{P}^{\boldsymbol{u}}(\boldsymbol{X}_{0:T}^{\boldsymbol{u}})
$$

The **variance of the importance weight** is a measure of the *similarity between $\mathbb{P}^{\boldsymbol{u}}$ and $\mathbb{P}^\star$*. The variance is zero only when the importance weight is a constant that is not dependent on the path, which occurs only when $\mathbb{P}^{\boldsymbol{u}} = \mathbb{P}^\star$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.17</span><span class="math-callout__name">(Variance and Log-Variance Losses)</span></p>

The **variance** and **log-variance** (LV) losses between the controlled path measure $\mathbb{P}^{\boldsymbol{u}}$ and the optimal path measure $\mathbb{P}^\star$ are defined as:

$$
\mathcal{L}_{\text{Var}_{\mathbb{P}^{\boldsymbol{v}}}}(\mathbb{P}^{\boldsymbol{u}}, \mathbb{P}^\star) := \text{Var}_{\mathbb{P}^{\boldsymbol{v}}}\!\left(\frac{d\mathbb{P}^\star}{d\mathbb{P}^{\boldsymbol{u}}}\right), \quad \mathcal{L}_{\text{Var}}^{\log}(\mathbb{P}^{\boldsymbol{u}}, \mathbb{P}^\star) := \text{Var}_{\mathbb{P}^{\boldsymbol{v}}}\!\left(\log \frac{d\mathbb{P}^\star}{d\mathbb{P}^{\boldsymbol{u}}}\right)
$$

Let $\boldsymbol{u}(\boldsymbol{x}, t)$ denote the control being optimized and $\boldsymbol{v}(\boldsymbol{x}, t)$ denote an arbitrary fixed control that generates the stochastic paths $(\boldsymbol{X}\_t^{\boldsymbol{v}})\_{t \in [0,T]}$. Then, the variance and log-variance losses take the path-integral form:

$$
\mathcal{L}_{\text{Var}_{\mathbb{P}^{\boldsymbol{v}}}}(\boldsymbol{u}) := \frac{1}{Z^2}\text{Var}_{\mathbb{P}^{\boldsymbol{v}}}\!\left(e^{\mathcal{F}_{u,v} - \Phi(\boldsymbol{X}_T^{\boldsymbol{v}})}\right), \quad \mathcal{L}_{\text{Var}}^{\log}(\boldsymbol{u}) := \text{Var}_{\mathbb{P}^{\boldsymbol{v}}}\!\left(\mathcal{F}_{u,v} - \Phi(\boldsymbol{X}_T^{\boldsymbol{v}})\right)
$$

where $\mathcal{F}\_{u,v}$ is given by:

$$
\mathcal{F}_{u,v} = \frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{v}}, t) \rVert^2 dt - \int_0^T (\boldsymbol{u} \cdot \boldsymbol{v})(\boldsymbol{X}_t^{\boldsymbol{v}}, t) dt - \int_0^T \boldsymbol{u}(\boldsymbol{X}_t^{\boldsymbol{v}}, t)^\top d\boldsymbol{B}_t^{\boldsymbol{v}} - \int_0^T c(\boldsymbol{X}_t^{\boldsymbol{v}}, t) dt
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.18</span><span class="math-callout__name">(Variance Loss Depends on Sampling Distribution)</span></p>

The variance-based objectives depend on the definition of $\boldsymbol{v}$ used to sample $\boldsymbol{X}\_{0:T}^{\boldsymbol{v}} \sim \mathbb{P}^{\boldsymbol{v}}$. However, it holds that the variance is minimized exactly when the controlled measure $\mathbb{P}^{\boldsymbol{u}}$ matches the optimal measure $\mathbb{P}^\star$:

$$
\forall \boldsymbol{u} \in \mathcal{U}, \quad \mathcal{L}_{\text{Var}_{\mathbb{P}^{\boldsymbol{v}}}}(\boldsymbol{u}) = 0 \iff \boldsymbol{u} = \boldsymbol{u}^\star \quad \text{and} \quad \mathcal{L}_{\text{Var}}^{\log}(\boldsymbol{u}) = 0 \iff \boldsymbol{u} = \boldsymbol{u}^\star
$$

This is because when $\mathbb{P}^{\boldsymbol{u}} = \mathbb{P}^\star$, the RND becomes identically one ($d\mathbb{P}^\star / d\mathbb{P}^\star \equiv 1$), and the variance of a constant is zero.

Comparing the CE Loss and the Variance Losses: both sample paths from an arbitrary fixed path measure $\mathbb{P}^{\boldsymbol{v}}$ with control $\boldsymbol{v}$. However, the CE Loss does not change with different choices of $\boldsymbol{v}$, since the dependence on $\mathbb{P}^{\boldsymbol{v}}$ cancels out when scaling with the RND $d\mathbb{P}^\star / d\mathbb{P}^{\boldsymbol{v}}$. In contrast, the variance-based objectives change with respect to different definitions of $\boldsymbol{v}$, since the distribution under which the variance is calculated, $\mathbb{P}^{\boldsymbol{v}}$, changes with different $\boldsymbol{v}$.

</div>

We have introduced three variations of the SOC objective, which are all derived from the RND between path measures. The **key insight** is that all three objectives can be **estimated from paths generated from a tractable SDE**, enabling practical learning of the optimal bridge dynamics. Although the objectives differ in their sampling laws, training stability, and convergence guarantees, they are all minimized when the controlled dynamics recover the optimal bridge $\mathbb{P}^{\boldsymbol{u}} = \mathbb{P}^\star$.

### 3.4 Closing Remarks for Section 3

In this section, we introduced the stochastic optimal control (SOC) problem, which defines a path-space variational objective that minimizes a running cost and terminal cost function generated by a *controlled SDE*. Leveraging dynamic programming theory, we introduced the **value function** $V\_t$, which solves the Hamilton--Jacobi--Bellman equation similarly to the Lagrangian $\psi\_t$ defined in Section 2.7 and connects to the optimal control drift $\boldsymbol{u}^\star$ and optimal path measure Radon--Nikodym derivative.

A **key observation** is that the quadratic control cost in the SOC objective is equivalent to the KL divergence between the controlled and reference Ito processes from Corollary 2.21. This connection allows us to reformulate the dynamic Schroedinger bridge problem as an SOC objective in which the terminal cost is expressed as a log-ratio involving the backward Schroedinger potential $\hat{\varphi}\_T$ and the terminal marginal constraint $\pi\_T$. This reformulation avoids the need for explicit couplings between $\pi\_0$ and $\pi\_T$, and we show that the backward potential $\hat{\varphi}\_T$ effectively absorbs the dependence on the initial value $V\_0(\boldsymbol{X}\_0)$ appearing in classical SOC formulations, making the framework applicable to arbitrary initial distributions and reference dynamics.

Building on this formulation, we introduced several tractable training objectives that are uniquely minimized by the optimal control drift $\boldsymbol{u}^\star$ and its corresponding path measure $\mathbb{P}^\star$. A **key step** in deriving these objectives is using Girsanov's theorem to express the Radon--Nikodym derivative between the controlled and reference path measures directly in terms of the control drift, yielding closed-form expressions and a practical way to evaluate and optimize path-space divergences using trajectories simulated from the controlled SDE.

Now that we have explored several complementary formulations of the SB problem, from the static formulation to dynamic formulations with connections to SOC, we next turn to the practical construction of Schroedinger bridges. The next section will present several concrete algorithms and computational strategies for building stochastic bridges and Schroedinger bridges in practice.

## 4. Building Schroedinger Bridges

While the Schroedinger bridge problem provides a variational characterization of the optimal stochastic transport between two marginals, the formulation alone does not immediately reveal how such bridges can be constructed in practice. In this section, we move from the abstract formulation to explore several complementary mechanisms for building stochastic bridges between prescribed endpoint distributions.

We begin by interpreting Schroedinger bridges as a mixture of conditional bridges (Section 4.1), which provides an intuitive pathwise construction. We then study time reversal (Section 4.2) and the resulting forward--backward SDE representation (Section 4.3), revealing how drift corrections arise from the score of the evolving distribution.

Throughout this section, we use a slight abuse of notation and define the **transition density** from $\boldsymbol{X}\_t$ at time $t$ to $\boldsymbol{X}\_\tau$ at a later time $\tau \ge t$ under a path measure as $\mathbb{P}\_{\tau \mid t}(\boldsymbol{x}\_\tau \mid \boldsymbol{x}\_t)$. For instance, the transition density from $\boldsymbol{x}$ at time $t$ to $\boldsymbol{x}\_T$ at time $T$ under the reference measure $\mathbb{Q}$ is denoted $\mathbb{Q}\_{T \mid t}(\boldsymbol{x}\_T \mid \boldsymbol{x}) = \mathbb{Q}(\boldsymbol{X}\_T = \boldsymbol{x}\_T \mid \boldsymbol{X}\_t = \boldsymbol{x})$.

### 4.1 Mixture of Conditional Bridges

A useful way to understand the structure of Schroedinger bridges is through endpoint conditioning. Rather than viewing the bridge as a single global stochastic process, we can interpret it as a **mixture of conditional bridges** under the reference process $\mathbb{Q}$ connecting specific endpoint pairs $(\boldsymbol{x}\_0, \boldsymbol{x}\_T)$ drawn from a coupling $\pi\_{0,T}$. Each conditional bridge describes the distribution of paths under the reference dynamics conditioned on fixed endpoints, which recovers the optimal Schroedinger bridge when conditioned on the **optimal coupling** $\pi\_{0,T}^\star$ that solves the static Schroedinger bridge problem.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.1</span><span class="math-callout__name">(Mixture of Endpoint-Conditioned Bridges)</span></p>

Consider the dynamic SB problem with reference process $\mathbb{Q}$ and marginal constraints $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$:

$$
\mathbb{P}^\star = \underset{\mathbb{P}^{\boldsymbol{u}}}{\arg\min} \left\lbrace \text{KL}(\mathbb{P}^{\boldsymbol{u}} \| \mathbb{Q}) : \mathbb{P}_0^{\boldsymbol{u}} = \pi_0, \mathbb{P}_T^{\boldsymbol{u}} = \pi_T \right\rbrace
$$

Then, the unique minimizer $\mathbb{P}^\star$ can be written as a **mixture of endpoint-conditioned bridges** $\mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$:

$$
\mathbb{P}^\star(\boldsymbol{X}_{0:T}) = \int \mathbb{Q}(\boldsymbol{X}_{0:T} \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \pi_{0,T}^\star(d\boldsymbol{x}_0, d\boldsymbol{x}_T)
$$

where $\pi\_{0,T}^\star$ is the optimal coupling that solves the **static Schroedinger bridge problem** with the reference coupling defined as the endpoint law of the reference process $\mathbb{Q}\_{0,T}$:

$$
\pi_{0,T}^\star = \underset{\pi_{0,T} \in \Pi(\pi_0, \pi_T)}{\arg\min}\; \text{KL}(\pi_{0,T} \| \mathbb{Q}_{0,T})
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Mixture of Endpoint-Conditioned Bridges)</summary>

By the law of total probability, decompose the reference path measure $\mathbb{Q}$ as:

$$
\mathbb{Q}(\boldsymbol{X}_{0:T}) = \mathbb{Q}(\boldsymbol{X}_{0:T} \mid \boldsymbol{X}_0 = \boldsymbol{x}_0, \boldsymbol{X}_T = \boldsymbol{x}_T) \mathbb{Q}_{0,T}(\boldsymbol{x}_0, \boldsymbol{x}_T)
$$

Similarly, decompose any candidate path measure $\mathbb{P}^{\boldsymbol{u}}$ with endpoint law $\mathbb{P}\_{0,T} \equiv \pi\_{0,T}$:

$$
\mathbb{P}(\boldsymbol{X}_{0:T}) = \mathbb{P}(\boldsymbol{X}_{0:T} \mid \boldsymbol{X}_0 = \boldsymbol{x}_0, \boldsymbol{X}_T = \boldsymbol{x}_T) \pi_{0,T}(\boldsymbol{x}_0, \boldsymbol{x}_T)
$$

Applying the **KL Divergence Chain Rule** (Lemma 1.4), the dynamic SB objective splits:

$$
\text{KL}(\mathbb{P}^{\boldsymbol{u}} \| \mathbb{Q}) = \underbrace{\text{KL}(\pi_{0,T} \| \mathbb{Q}_{0,T})}_{(\star)} + \underbrace{\mathbb{E}_{(\boldsymbol{x}_0, \boldsymbol{x}_T) \sim \pi_{0,T}}\!\left[\text{KL}(\mathbb{P}(\cdot \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \| \mathbb{Q}(\cdot \mid \boldsymbol{x}_0, \boldsymbol{x}_T))\right]}_{(\diamond)}
$$

For any fixed endpoint law $\pi\_{0,T}$, the minimizer of $(\diamond)$ is achieved when $\mathbb{P}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T) = \mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$ and the KL is zero. Therefore, **solving the dynamic SB over the restricted space of path measures where $\mathbb{P}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T) = \mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$ reduces to solving the static SB problem**:

$$
\pi_{0,T}^\star = \underset{\pi_{0,T} \in \Pi(\pi_0, \pi_T)}{\arg\min}\; \text{KL}(\pi_{0,T} \| \mathbb{Q}_{0,T})
$$

Since $\mathbb{P}^\star\_{0,T} = \pi\_{0,T}^\star$, the minimizer is:

$$
\mathbb{P}^\star(\boldsymbol{X}_{0:T}) = \int \mathbb{Q}(\boldsymbol{X}_{0:T} \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \pi_{0,T}^\star(d\boldsymbol{x}_0, d\boldsymbol{x}_T)
$$

which is *unique* following the proof of uniqueness of the static SB problem in Proposition 1.9. $\square$
</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.2</span><span class="math-callout__name">(Mixture of Brownian Bridges)</span></p>

Consider the dynamic SB problem with Brownian reference process $\mathbb{Q} : d\boldsymbol{X}\_t = \sigma\_t d\boldsymbol{B}\_t$ and marginal constraints $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$. Then, the unique minimizer $\mathbb{P}^\star$ can be written as a **mixture of endpoint-conditioned Brownian bridges** $\mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$:

$$
\mathbb{P}^\star(\boldsymbol{X}_{0:T}) = \int \mathbb{Q}(\boldsymbol{X}_{0:T} \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \pi_{0,T}^\star(d\boldsymbol{x}_0, d\boldsymbol{x}_T)
$$

where $\pi\_{0,T}^\star$ solves the Entropic OT Problem with quadratic state cost $c(\boldsymbol{x}, \boldsymbol{y}) := \lVert \boldsymbol{x} - \boldsymbol{y} \rVert^2$.

</div>

This factorized definition of the Dynamic SB Problem can be used to define a tractable objective based on **conditional stochastic optimal control**, which samples pairs from the optimal marginal law $\pi\_{0,T}^\star$ and optimizes the interpolating controlled dynamics such that they minimize the KL divergence from the reference drift.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.3</span><span class="math-callout__name">(Conditional Stochastic Optimal Control)</span></p>

Consider the controlled path measure $\mathbb{P}^{\boldsymbol{u}}$, where the marginal density $p\_t$ can be factorized as $p\_t(\boldsymbol{x}) = \int p\_t(\boldsymbol{x} \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T) \pi\_{0,T}(d\boldsymbol{x}\_0, d\boldsymbol{x}\_T)$. Given the optimal endpoint distribution $\pi\_{0,T}^\star$ that solves the static SB problem with reference distribution $\mathbb{Q}\_{0,T}$, the **dynamic SB problem objective decomposes into a mixture of conditional stochastic optimal control problems** with endpoints sampled from $\pi\_{0,T}^\star$:

$$
\inf_{\boldsymbol{u}} \mathbb{E}_{\pi_{0,T}^\star(d\boldsymbol{x}_0, d\boldsymbol{x}_T)} \int_0^T \mathbb{E}_{p_t(\cdot \mid \boldsymbol{x}_0, \boldsymbol{x}_T)}\!\left[\frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_t, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \rVert^2\right] dt
$$

$$
\text{s.t.} \quad d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T)) dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0 = \boldsymbol{x}_0, \quad \boldsymbol{X}_T = \boldsymbol{x}_T
$$

</div>

This mixture conditional SOC problem decomposes the path space KL minimization from Proposition 4.1 into a static SB problem and a family of conditional bridge problems. This formulation allows us to solve the dynamic SB problem through a **tractable two-stage procedure**: first, estimate the optimal static coupling $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \pi\_{0,T}^\star$, then learn the corresponding conditional stochastic control law $\boldsymbol{u}(\boldsymbol{X}\_t, t \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$.

### 4.2 Time Reversal

A useful approach for constructing stochastic bridges is to analyze the time reversal of a stochastic process, where we leverage the **key idea** that conditioning on a terminal constraint induces a *modified reverse-time dynamics* whose drift differs from the original forward drift. Rather than directly enforcing the endpoint condition, we reinterpret the forward process as a backward process and derive the forward Markovian dynamics that reproduce the same marginal density evolution under time reversal.

Consider a **forward-time stochastic process** defined on $t \in [0, T]$ generated from the SDE:

$$
d\boldsymbol{X}_t = \boldsymbol{f}(\boldsymbol{X}_t, t) dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0 = \boldsymbol{x}_0
$$

One method of building a stochastic bridge that is **conditioned to reach a target state** $\boldsymbol{X}\_T = \boldsymbol{x}\_T$ is to reverse the time coordinate of the original forward process such that it can be conditioned on the target state and then derive a **reverse stochastic process** that matches the density evolution of the forward process with time-reversal.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.4</span><span class="math-callout__name">(Time Reversal Formula)</span></p>

Consider a forward-time SDE of the form $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$, where $\boldsymbol{f}(\boldsymbol{x}, t)$ is the drift and $\sigma\_t$ is the scalar diffusion coefficient. Let $p\_t(\boldsymbol{x})$ denote the marginal density of $\boldsymbol{X}\_t$ at time $t$. Then, the time-reversed process $\tilde{\boldsymbol{X}}\_s := \boldsymbol{X}\_{T-s}$ follows the SDE:

$$
d\tilde{\boldsymbol{X}}_s = \left[-\boldsymbol{f}(\tilde{\boldsymbol{X}}_s, T - s) + \sigma_{T-s}^2 \nabla\log p(\tilde{\boldsymbol{X}}_s, T - s)\right] ds + \sigma_{T-s} d\tilde{\boldsymbol{B}}_s
$$

where $\tilde{\boldsymbol{B}}\_s$ is the Brownian motion with respect to the reverse-time filtration $\bar{\mathcal{F}} = \sigma(\tilde{\boldsymbol{X}}\_\tau : 0 \le \tau \le s)$. Together, the pair of forward and reverse-time SDEs are given by:

$$
\begin{cases} d\boldsymbol{X}_t = \boldsymbol{f}(\boldsymbol{X}_t, t) dt + \sigma_t d\boldsymbol{B}_t \\[4pt] d\tilde{\boldsymbol{X}}_s = \left[-\boldsymbol{f}(\tilde{\boldsymbol{X}}_s, T - s) + \sigma_{T-s}^2 \nabla\log p(\tilde{\boldsymbol{X}}_s, T - s)\right] ds + \sigma_{T-s} d\tilde{\boldsymbol{B}}_s \end{cases}
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Time Reversal Formula)</summary>

**Step 1: Reparameterize the Forward Process with Time-Reversal.** Define $s = T - t$ so that $s = 0$ corresponds to $t = T$ and $s = T$ corresponds to $t = 0$. Denote $\overleftarrow{\boldsymbol{X}}\_s = \boldsymbol{X}\_{T-s}$ and the density $\tilde{p}\_s(\boldsymbol{x}) = p(\boldsymbol{x}, T - s)$. The marginal density evolves via the Fokker--Planck equation:

$$
\partial_t p(\boldsymbol{x}, t) = -\nabla \cdot (\boldsymbol{f}(\boldsymbol{x}, t) p(\boldsymbol{x}, t)) + \frac{\sigma_t^2}{2}\Delta p(\boldsymbol{x}, t)
$$

Performing the change of variables $\frac{\partial}{\partial s}\tilde{p}\_s(\boldsymbol{x}) = \frac{\partial s}{\partial t}\frac{\partial}{\partial t}p(\boldsymbol{x}, T-s) = -\frac{\partial}{\partial t}p(\boldsymbol{x}, T-s)$, the time-reversed density follows:

$$
\frac{\partial}{\partial s}\tilde{p}_s(\boldsymbol{x}) = \nabla \cdot (\boldsymbol{f}(\boldsymbol{x}, T - s)\tilde{p}_s(\boldsymbol{x})) - \frac{\sigma_{T-s}^2}{2}\Delta\tilde{p}_s(\boldsymbol{x})
$$

**Step 2: Derive the Fokker--Planck Equation of the Reverse Stochastic Process.** Define $(\tilde{\boldsymbol{X}}\_s)\_{s \in [0,T]}$ as an Ito process that evolves backward in time with unknown drift $\boldsymbol{b}(\tilde{\boldsymbol{X}}\_s, s)$ and diffusion $\tilde{\sigma}\_s \equiv \sigma\_{T-s}$:

$$
d\tilde{\boldsymbol{X}}_s = \boldsymbol{b}(\tilde{\boldsymbol{X}}_s, s) ds + \tilde{\sigma}_s d\tilde{\boldsymbol{B}}_s
$$

Its Fokker--Planck equation is: $\frac{\partial}{\partial s}\tilde{p}\_s = -\nabla \cdot (\boldsymbol{b}\tilde{p}\_s) + \frac{\tilde{\sigma}\_s^2}{2}\Delta\tilde{p}\_s$.

Setting the two density evolution equations equal:

$$
-\nabla \cdot (\boldsymbol{b}\tilde{p}_s) + \frac{\tilde{\sigma}_s^2}{2}\Delta\tilde{p}_s = \nabla \cdot (\boldsymbol{f}\tilde{p}_s) - \frac{\tilde{\sigma}_s^2}{2}\Delta\tilde{p}_s
$$

$$
\nabla \cdot \!\left([\boldsymbol{b} + \boldsymbol{f}]\tilde{p}_s\right) = \tilde{\sigma}_s^2 \Delta\tilde{p}_s = \nabla \cdot (\tilde{\sigma}_s^2 \nabla\tilde{p}_s)
$$

Integrating against an arbitrary test function $\phi(\boldsymbol{x})$ and applying integration by parts to both sides, we obtain $[\boldsymbol{b} + \boldsymbol{f}]\tilde{p}\_s = \tilde{\sigma}\_s^2 \nabla\tilde{p}\_s$. Dividing by $\tilde{p}\_s \ge 0$ and using $\frac{\nabla\tilde{p}\_s}{\tilde{p}\_s} = \nabla\log\tilde{p}\_s$ (the **score function**):

$$
\boldsymbol{b}(\boldsymbol{x}, s) = -\boldsymbol{f}(\boldsymbol{x}, T - s) + \tilde{\sigma}_s^2 \nabla\log\tilde{p}_s(\boldsymbol{x})
$$

Since $\tilde{p}\_s(\boldsymbol{x}) = p(\boldsymbol{x}, T - s)$ and $\tilde{\sigma}\_s = \sigma\_{T-s}$:

$$
\boldsymbol{b}(\boldsymbol{x}, s) = -\boldsymbol{f}(\boldsymbol{x}, T - s) + \sigma_{T-s}^2 \nabla\log p(\boldsymbol{x}, T - s)
$$

which completes the derivation of the time reversal formula. $\square$
</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Denoising Diffusion as a Special Case of the Schroedinger Bridge)</span></p>

Schroedinger bridges are a **generalization** of score-based diffusion models. Specifically, score-based models are a special case where the forward-time stochastic process is the variance exploding SDE with zero drift: $d\boldsymbol{X}\_t = \sigma\_t d\boldsymbol{B}\_t$, $\boldsymbol{X}\_0 = \boldsymbol{x}\_0 \sim p\_{\text{data}}$.

Applying the time reversal, the backward stochastic process $\tilde{\boldsymbol{X}}\_s = \boldsymbol{X}\_{T-s}$ is:

$$
d\tilde{\boldsymbol{X}}_s = \left[\sigma_{T-s}^2 \nabla\log\tilde{p}_s(\boldsymbol{x})\right] ds + \sigma_{T-s} d\tilde{\boldsymbol{B}}_s
$$

Since the variance of the forward process is $\beta\_t := \int\_0^t \sigma\_s^2 ds$, the backward density is Gaussian $\tilde{p}\_s(\boldsymbol{x}) = \mathcal{N}(\boldsymbol{x}\_0, \beta\_T - \beta\_s)$ with score $\nabla\log\tilde{p}\_s(\boldsymbol{x}) = -\frac{\boldsymbol{x} - \boldsymbol{x}\_0}{\beta\_T - \beta\_s}$, yielding the SDE of a **Brownian bridge**:

$$
d\tilde{\boldsymbol{X}}_s = \left[\sigma_{T-s}^2 \frac{\boldsymbol{x}_0 - \tilde{\boldsymbol{X}}_s}{\beta_T - \beta_s}\right] ds + \sigma_{T-s} d\tilde{\boldsymbol{B}}_s
$$

In practice, we approximate $\tilde{\boldsymbol{X}}\_0 \sim \mathcal{N}(\boldsymbol{x}\_0, \beta\_T I\_d)$ with $\tilde{\boldsymbol{X}}\_0 \sim \mathcal{N}(\boldsymbol{0}, \beta\_T I\_d)$, which works when $\beta\_T$ is large relative to the data variance.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.5</span><span class="math-callout__name">(Time Reversal for Controlled SDEs)</span></p>

Consider a forward-time controlled SDE of the form $d\boldsymbol{X}\_t = (\boldsymbol{f}(\boldsymbol{X}\_t, t) + \sigma\_t \boldsymbol{u}(\boldsymbol{X}\_t, t)) dt + \sigma\_t d\boldsymbol{B}\_t$, where $\boldsymbol{u}(\boldsymbol{x}, t)$ is the control drift. Let $p\_t(\boldsymbol{x})$ denote the marginal density of $\boldsymbol{X}\_t$ at time $t$. Then, the time-reversed process $\tilde{\boldsymbol{X}}\_s := \boldsymbol{X}\_{T-s}$ follows the SDE:

$$
d\tilde{\boldsymbol{X}}_s = \left[-\boldsymbol{f}(\tilde{\boldsymbol{X}}_s, T - s) - \sigma_{T-s}\boldsymbol{u}(\tilde{\boldsymbol{X}}_s, T - s) + \sigma_{T-s}^2 \nabla\log p(\tilde{\boldsymbol{X}}_s, T - s)\right] ds + \sigma_{T-s} d\tilde{\boldsymbol{B}}_s
$$

and the pair of forward and reverse-time SDEs is:

$$
\begin{cases} d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t)) dt + \sigma_t d\boldsymbol{B}_t \\[4pt] d\tilde{\boldsymbol{X}}_s = \left[-\boldsymbol{f}(\tilde{\boldsymbol{X}}_s, T - s) - \sigma_{T-s}\boldsymbol{u}(\tilde{\boldsymbol{X}}_s, T - s) + \sigma_{T-s}^2 \nabla\log p(\tilde{\boldsymbol{X}}_s, T - s)\right] ds + \sigma_{T-s} d\tilde{\boldsymbol{B}}_s \end{cases}
$$

</div>

The **key takeaway** is that reversing a forward-time stochastic process is not as simple as inverting the time coordinate. While the forward process propagates densities $p\_t$ according to its drift $\boldsymbol{f}(\boldsymbol{X}\_t, t)$ and diffusion coefficients, running the process backward requires deriving a **new stochastic process** whose drift correctly propagates the marginal densities in the reverse time direction such that it reconstructs the same marginal densities as the forward process $s = T - t \in [0, T]$.

The Time Reversal Formula reveals that the reverse-time drift includes the negative forward drift $-\boldsymbol{f}(\boldsymbol{X}\_s, T - s)$ and an **additional correction proportional to the score function** $\nabla\log p\_{T-s}(\boldsymbol{x})$, which compensates for the spreading of probability density from the forward diffusion. This insight is fundamental to modern generative modeling frameworks, such as score-based diffusion models and Schroedinger bridge methods.

### 4.3 Forward--Backward Stochastic Differential Equations

A **key limitation** of the standard time-reversal formula from Section 4.2 is that it assumes an *uncontrolled* forward SDE, where the drift $\boldsymbol{f}(\boldsymbol{x}, t)$ is deterministic and can be easily simulated in either direction. In the general Schroedinger bridge setting, however, the goal is to interpolate between two *structured distributions*, and the optimal forward drift that transports $\pi\_0$ to $\pi\_T$ is itself *unknown* and must be solved for as part of the optimization problem. As a result, the Time Reversal Formula is no longer sufficient on its own, and we must instead derive a coupled pair of **forward-backward SDEs** that characterizes both the forward and backward controlled dynamics.

Recall that the optimal control and optimal marginal density $(\boldsymbol{u}^\star, p\_t^\star)$ are defined by the non-linear HJB-FP System (Proposition 2.23). While the Hopf--Cole transform (Section 2.8) transforms these non-linear PDEs into linear PDEs, solving them remains challenging. We show that score-based generative modeling can be derived from the time reversal of a forward SDE. This naturally leads to the question: *how can we turn the PDEs defining the solution to the dynamic SB problem into SDEs that can be solved with likelihood training?*

To answer this, we leverage the theory of **forward-backward SDEs** (FBSDEs), which extends the Feynman--Kac theory from Section 2.4. FBSDE theory introduces a **coupled system consisting of a forward SDE that generates trajectories and a backward stochastic process that evolves along those trajectories** and encodes the solution of the PDE.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forward--Backward SDE Theory)</span></p>

Consider a function $\psi\_t(\boldsymbol{x}) : \mathbb{R}^d \times [0, T] \to \mathbb{R} \in C^{2,1}(\mathbb{R}^d \times [0, T])$ and a **parabolic PDE** of the form:

$$
\partial_t\psi_t + \frac{\sigma_t^2}{2}\Delta\psi_t + \langle \boldsymbol{f}, \nabla\psi_t \rangle + h(\boldsymbol{x}, t, \psi_t, \sigma_t\nabla\psi_t) = 0, \quad \psi_T(\boldsymbol{x}) = \Phi(\boldsymbol{x})
$$

Suppose the state $\boldsymbol{X}\_t = \boldsymbol{x}$ is a stochastic process that evolves via a forward SDE $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$ with $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$. Then, we can define the stochastic process $Y\_t := \psi\_t(\boldsymbol{X}\_t)$ and $\boldsymbol{Z}\_t := \sigma\_t\nabla\psi\_t(\boldsymbol{X}\_t)$ which evolves via the **backward SDE**:

$$
dY_t = -h(\boldsymbol{X}_t, t, Y_t, \boldsymbol{Z}_t) dt + \boldsymbol{Z}_t^\top d\boldsymbol{B}_t, \quad Y_T = \Phi(\boldsymbol{X}_T)
$$

such that the solution $\psi\_t$ to the parabolic PDE is equivalent to the solution to the forward-backward SDEs.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forward--Backward SDEs for Dynamic Schroedinger Bridge)</span></p>

In Section 2.8, we transformed the non-linear HJB-FP System with the **Hopf--Cole transform** to get the system of **linear PDEs** that defines the solution to the Schroedinger bridge problem:

$$
\begin{cases} \partial_t\varphi_t + \langle \nabla\varphi_t, \boldsymbol{f} \rangle = -\frac{\sigma_t^2}{2}\Delta\varphi_t \\[4pt] \partial_t\hat{\varphi}_t + \nabla \cdot (\hat{\varphi}_t \boldsymbol{f}) = \frac{\sigma_t^2}{2}\Delta\hat{\varphi}_t \end{cases} \quad \text{s.t.} \quad \begin{cases} p_0^\star = \varphi_0\hat{\varphi}_0 \\ p_T^\star = \varphi_T\hat{\varphi}_T \end{cases}
$$

where $(\varphi\_t, \hat{\varphi}\_t)$ are the pair of **Schroedinger potentials** that *uniquely* characterize the optimal control and optimal state PDF $(\boldsymbol{u}^\star, p\_t^\star)$ given by:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = \sigma_t\nabla\log\varphi_t(\boldsymbol{x}), \quad p_t^\star(\boldsymbol{x}) = \varphi_t(\boldsymbol{x})\hat{\varphi}_t(\boldsymbol{x})
$$

Given the stochastic process $\boldsymbol{X}\_{0:T}$, we can define the random variables associated with the forward and backward potentials as:

$$
Y_t \equiv Y_t(\boldsymbol{X}_t, t) = \log\varphi_t(\boldsymbol{X}_t), \quad Z_t \equiv Z_t(\boldsymbol{X}_t, t) = \sigma_t\nabla\log\varphi_t(\boldsymbol{X}_t)
$$

$$
\hat{Y}_t \equiv \hat{Y}_t(\boldsymbol{X}_t, t) = \log\hat{\varphi}_t(\boldsymbol{X}_t), \quad \hat{Z}_t \equiv \hat{Z}_t(\boldsymbol{X}_t, t) = \sigma_t\nabla\log\hat{\varphi}_t(\boldsymbol{X}_t)
$$

These define the forward--backward SDE system for the dynamic Schroedinger bridge, where $Y\_t$ and $\hat{Y}\_t$ encode the log-potentials along the forward stochastic trajectories.

</div>

The forward--backward SDEs that define the evolution of these random variables are given by:

$$
\begin{cases} d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{Z}_t) dt + \sigma_t d\boldsymbol{B}_t \\[4pt] dY_t = \frac{1}{2}\lVert \boldsymbol{Z}_t \rVert^2 dt + \boldsymbol{Z}_t^\top d\boldsymbol{B}_t \\[4pt] d\hat{Y}_t = \left[\nabla \cdot (\sigma_t\hat{\boldsymbol{Z}}_t - \boldsymbol{f}) + \frac{1}{2}\lVert \hat{\boldsymbol{Z}}_t \rVert^2 + \boldsymbol{Z}_t^\top \hat{\boldsymbol{Z}}_t\right] dt + \hat{\boldsymbol{Z}}_t^\top d\boldsymbol{B}_t \end{cases}
$$

with boundary conditions $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$, $Y\_T + \hat{Y}\_T = \log\pi\_T(\boldsymbol{X}\_T)$, $Y\_t + \hat{Y}\_t = \log p\_t^\star(\boldsymbol{X}\_t)$, and $\boldsymbol{u}^\star(\boldsymbol{X}\_t, t) = \boldsymbol{Z}\_t$.

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Derivation (Forward--Backward SDEs for SB)</summary>

**Step 1: Derive the stochastic processes encoding optimality conditions.** The forward SDE has control drift $\boldsymbol{u}^\star = \sigma\_t\nabla\log\varphi\_t = \boldsymbol{Z}\_t$:

$$
d\boldsymbol{X}_t = [\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t^2\nabla\log\varphi_t(\boldsymbol{X}_t)] dt + \sigma_t d\boldsymbol{B}_t = [\boldsymbol{f} + \sigma_t\boldsymbol{Z}_t] dt + \sigma_t d\boldsymbol{B}_t
$$

**Step 2: SDE for $Y\_t = \log\varphi\_t(\boldsymbol{X}\_t)$.** Applying Ito's formula and substituting the Hopf--Cole PDE $\partial\_t\varphi\_t = -\langle \nabla\varphi\_t, \boldsymbol{f} \rangle - \frac{\sigma\_t^2}{2}\Delta\varphi\_t$:

$$
d\log\varphi_t = \left[\partial_t\log\varphi_t + \langle \boldsymbol{f} + \sigma_t^2\nabla\log\varphi_t, \nabla\log\varphi_t \rangle + \frac{\sigma_t^2}{2}\Delta\log\varphi_t\right] dt + (\sigma_t\nabla\log\varphi_t)^\top d\boldsymbol{B}_t
$$

Using $\partial\_t\log\varphi\_t = -\langle \nabla\log\varphi\_t, \boldsymbol{f} \rangle - \frac{\sigma\_t^2}{2}\frac{\Delta\varphi\_t}{\varphi\_t}$ and the identity $\Delta\log\varphi = \frac{\Delta\varphi}{\varphi} - \lVert \nabla\log\varphi \rVert^2$, cancellations yield:

$$
dY_t = \frac{1}{2}\lVert \boldsymbol{Z}_t \rVert^2 dt + \boldsymbol{Z}_t^\top d\boldsymbol{B}_t
$$

**Step 3: SDE for $\hat{Y}\_t = \log\hat{\varphi}\_t(\boldsymbol{X}\_t)$.** Similarly, applying Ito's formula and using the Hopf--Cole PDE $\partial\_t\hat{\varphi}\_t = -\nabla \cdot (\hat{\varphi}\_t\boldsymbol{f}) + \frac{\sigma\_t^2}{2}\Delta\hat{\varphi}\_t$. After computing $\partial\_t\log\hat{\varphi}\_t = -\langle \nabla\log\hat{\varphi}\_t, \boldsymbol{f} \rangle - \nabla \cdot \boldsymbol{f} + \frac{\sigma\_t^2}{2}\frac{\Delta\hat{\varphi}\_t}{\hat{\varphi}\_t}$, using $\frac{\Delta\hat{\varphi}\_t}{\hat{\varphi}\_t} = \Delta\log\hat{\varphi}\_t + \lVert \nabla\log\hat{\varphi}\_t \rVert^2$, and collecting terms:

$$
d\hat{Y}_t = \left[\nabla \cdot (\sigma_t\hat{\boldsymbol{Z}}_t - \boldsymbol{f}) + \frac{1}{2}\lVert \hat{\boldsymbol{Z}}_t \rVert^2 + \boldsymbol{Z}_t^\top\hat{\boldsymbol{Z}}_t\right] dt + \hat{\boldsymbol{Z}}_t^\top d\boldsymbol{B}_t \quad \square
$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.6</span><span class="math-callout__name">(Forward--Backward SDE Theory Generalizes the Time-Reversal Formula)</span></p>

The forward--backward SDE formulation provides a natural generalization of the classical time-reversal formula for uncontrolled forward SDEs. In the special case where the forward dynamics contain no control drift (i.e., $\boldsymbol{Z}\_t \equiv \boldsymbol{0}$), the forward SDE reduces to the reference diffusion, and the backward control $\hat{\boldsymbol{Z}}\_t$ becomes proportional to the **score** of the marginal density $p\_t^\star$:

$$
Y_t = \log\varphi_t(\boldsymbol{X}_t) \equiv 0 \implies \log p_t^\star(\boldsymbol{X}_t) = \hat{Y}_t = \log\hat{\varphi}_t(\boldsymbol{X}_t)
$$

$$
\implies \hat{\boldsymbol{Z}}_t = \sigma_t\nabla\log\hat{\varphi}_t(\boldsymbol{X}_t) = \sigma_t\nabla\log p_t^\star(\boldsymbol{X}_t)
$$

which yields a reverse-time SDE that is exactly the Time Reversal Formula (Proposition 4.4):

$$
d\tilde{\boldsymbol{X}}_s = \left[-\boldsymbol{f}(\tilde{\boldsymbol{X}}_s, T - s) + \sigma_{T-s}^2\nabla\log p(\tilde{\boldsymbol{X}}_s, T - s)\right] ds + \sigma_{T-s} d\tilde{\boldsymbol{B}}_s
$$

that characterizes score-based diffusion models. Therefore, the FBSDE formulation and the dynamic SB problem can be viewed as a **generalization of score-based diffusion**.

</div>

### 4.4 Doob's $h$-Transform

The **central goal** when solving the Schroedinger bridge problem is to derive a controlled stochastic process by minimally shaping a reference stochastic process that originates from an initial distribution $\pi\_0$ by **conditioning** it on a terminal distribution $\pi\_T$. One way to achieve this conditioned process is using **Doob's $h$-Transform**, which introduces a *tilting function*, known as the $h$-function, such that multiplying the transition density of the reference stochastic process by the $h$-function and re-normalizing produces the transition density of the optimal Schroedinger bridge.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.7</span><span class="math-callout__name">(Doob's $h$-Transform)</span></p>

Under a reference stochastic process $\mathbb{Q}$, let $\mathbb{Q}(\boldsymbol{X}\_\tau = \boldsymbol{y} \mid \boldsymbol{X}\_t = \boldsymbol{x})$ denote the transition kernel from $\boldsymbol{x}$ at $t$ to $\boldsymbol{y}$ at time $\tau \ge t$. Define a function $h(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ that satisfies the following space-time Markov consistency property:

$$
h(\boldsymbol{x}, t) = \mathbb{E}_{\mathbb{Q}}[h(\boldsymbol{X}_\tau, \tau) \mid \boldsymbol{X}_t = \boldsymbol{x}] = \int_{\mathbb{R}^d} \mathbb{Q}(\boldsymbol{X}_\tau = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) h(\boldsymbol{y}, \tau) d\boldsymbol{y}
$$

Then, we define the stochastic process $\mathbb{P}^h$ by **tilting** the reference process $\mathbb{Q}$ as:

$$
\mathbb{P}^h(\boldsymbol{X}_\tau = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) = \mathbb{Q}(\boldsymbol{X}_\tau = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) \frac{h(\boldsymbol{y}, \tau)}{h(\boldsymbol{x}, t)}
$$

where $\mathbb{P}^h(\boldsymbol{X}\_\tau = \boldsymbol{y} \mid \boldsymbol{X}\_t = \boldsymbol{x})$ is the tilted transition kernel of $\mathbb{P}^h$. Then, $\mathbb{P}^h$ is Markov and has the associated SDE:

$$
d\boldsymbol{X}_t = \left[\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t^2\nabla\log h(\boldsymbol{x}, t)\right] dt + \sigma_t d\boldsymbol{B}_t
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Doob's $h$-Transform)</summary>

First, the tilted transition density integrates to one:

$$
\int_{\mathbb{R}^d} \mathbb{P}^h(\boldsymbol{X}_\tau = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) d\boldsymbol{y} = \frac{1}{h(\boldsymbol{x}, t)} \underbrace{\int_{\mathbb{R}^d} \mathbb{Q}(\boldsymbol{X}_\tau = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) h(\boldsymbol{y}, \tau) d\boldsymbol{y}}_{=: h(\boldsymbol{x}, t)} = 1
$$

proving $\mathbb{P}^h$ is a valid Markov process. Next, we derive the SDE by defining a test function $\phi(\boldsymbol{x}, t)$ and computing the generator $\mathcal{A}\_t^h$ of the tilted process:

$$
(\mathcal{A}_t^h\phi)(\boldsymbol{x}) = \lim_{\Delta t \to 0} \frac{\mathbb{E}^h[\phi(\boldsymbol{X}_{t+\Delta t}, t + \Delta t) \mid \boldsymbol{X}_t = \boldsymbol{x}] - \phi(\boldsymbol{x})}{\Delta t}
$$

Using $\mathbb{E}^h[\phi(\boldsymbol{X}\_{t+\Delta t}) \mid \boldsymbol{X}\_t = \boldsymbol{x}] = \frac{1}{h(\boldsymbol{x}, t)}\mathbb{E}[\phi(\boldsymbol{X}\_{t+\Delta t})h(\boldsymbol{X}\_{t+\Delta t}, t + \Delta t) \mid \boldsymbol{X}\_t = \boldsymbol{x}]$, we get:

$$
(\mathcal{A}_t^h\phi)(\boldsymbol{x}) = \frac{1}{h(\boldsymbol{x}, t)}\mathcal{A}_t(\phi(\boldsymbol{x})h(\boldsymbol{x}, t))
$$

Expanding with the uncontrolled generator $\mathcal{A}\_t$ and using the product rule:

$$
\mathcal{A}_t^h\phi = \frac{1}{h}\left[\phi\underbrace{(\partial_t h + \langle \boldsymbol{f}, \nabla h \rangle + \frac{\sigma_t^2}{2}\Delta h)}_{= \mathcal{A}_t h = 0 \text{ (martingale)}} + h\langle \boldsymbol{f}, \nabla\phi \rangle + \phi\langle \boldsymbol{f}, \nabla h \rangle + \frac{\sigma_t^2}{2}(h\Delta\phi + \phi\Delta h + 2\nabla\phi \cdot \nabla h)\right]
$$

Since $h$ is a martingale under $\mathbb{Q}$, $\mathcal{A}\_t h = 0$. Simplifying:

$$
\mathcal{A}_t^h\phi = \langle \boldsymbol{f}, \nabla\phi \rangle + \sigma_t^2\nabla\phi \cdot \frac{\nabla h}{h} + \frac{\sigma_t^2}{2}\Delta\phi = \langle \boldsymbol{f} + \sigma_t^2\nabla\log h, \nabla\phi \rangle + \frac{\sigma_t^2}{2}\Delta\phi
$$

which identifies the drift as $\boldsymbol{v}(\boldsymbol{x}, t) = \boldsymbol{f}(\boldsymbol{x}, t) + \sigma\_t^2\nabla\log h(\boldsymbol{x}, t)$. $\square$
</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.8</span><span class="math-callout__name">(Generator of Reweighted Path Measure)</span></p>

Let $\mathbb{Q}$ be a reference path measure with infinitesimal generator $\mathcal{A}\_t$ and $h(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ be the $h$-function defined in the $h$-Function consistency property. Then, the path measure **reweighted by** $h$ is the $h$-transform of $\mathbb{Q}$ with generator defined as:

$$
\mathcal{A}_t^h\phi(\boldsymbol{x}) := \frac{\mathcal{A}_t(\phi(\boldsymbol{x})h(\boldsymbol{x}, t)) - \phi(\boldsymbol{x})\mathcal{A}_t h(\boldsymbol{x}, t)}{h(\boldsymbol{x}, t)} = \mathcal{A}_t\phi(\boldsymbol{x}) + \langle \sigma_t^2\nabla\log h(\boldsymbol{x}, t), \nabla\phi(\boldsymbol{x}) \rangle
$$

</div>

Doob's $h$-transform provides a precise mechanism for incorporating endpoint information into the dynamics of a reference process. This allows us to define an $h$-function that exactly recovers the Markov dynamics of the optimal Schroedinger bridge.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.9</span><span class="math-callout__name">(Schroedinger Bridge as Doob's $h$-Transform)</span></p>

Given the Schroedinger potentials $(\varphi\_t, \hat{\varphi}\_t)$ that generate the solution to the Dynamic SB Problem, we can define the $h$-function as:

$$
h(\boldsymbol{x}, t) := \varphi_t(\boldsymbol{x}) = \mathbb{E}_{\mathbb{Q}}[\varphi_T(\boldsymbol{X}_T) \mid \boldsymbol{X}_t = \boldsymbol{x}] = \int_{\mathbb{R}^d} \mathbb{Q}(\boldsymbol{X}_T = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) \varphi_T(\boldsymbol{y}) d\boldsymbol{y}
$$

Then, the optimal Schroedinger bridge path measure $\mathbb{P}^\star$ is the Doob's $h$-transform of $\mathbb{Q}$, where for any $0 \le t \le \tau \le T$:

$$
\mathbb{P}^\star(\boldsymbol{X}_\tau = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) = \mathbb{Q}(\boldsymbol{X}_\tau = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) \frac{h(\boldsymbol{y}, \tau)}{h(\boldsymbol{x}, t)}
$$

Equivalently, $\mathbb{P}^\star$ is Markov and has the associated SDE:

$$
d\boldsymbol{X}_t = \left[\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t^2\nabla\log h(\boldsymbol{x}, t)\right] dt + \sigma_t d\boldsymbol{B}_t
$$

</div>

This perspective shows that the Schroedinger bridge corresponds to a precise reweighting of the reference process $\mathbb{Q}$ through a harmonic function $h(\boldsymbol{x}, t) := \varphi\_t(\boldsymbol{x})$ which modifies the forward drift by $\sigma\_t^2\nabla\log h(\boldsymbol{x}, t)$. The resulting process evolves according to the controlled drift $\boldsymbol{f}(\boldsymbol{x}, t) + \sigma\_t^2\nabla\log h(\boldsymbol{x}, t)$, which can be interpreted as the **minimal modification of the reference dynamics required to enforce the desired endpoint constraints**.

### 4.5 Markovian and Reciprocal Projections

In many settings, the Schroedinger bridge yields a path measure whose dependencies span the entire trajectory, making direct simulation difficult. This raises a natural question: *Can we approximate a general bridge by a Markov process that is easier to simulate, while remaining as close as possible in relative entropy?*

The **Markovian projection** provides a principled answer. Given an arbitrary path measure, we project it onto the space of Markov measures $\mathcal{M}$ by minimizing KL divergence over processes whose future states depend *only on the present state*. Formally, the **space of Markov measures** is:

$$
\mathcal{M} := \left\lbrace \mathbb{M} \in \mathcal{P}(C([0,T]; \mathbb{R}^d)) \mid \forall 0 \le s < t \le T, \; \mathbb{E}_{\mathbb{M}}[f(\boldsymbol{X}_t) \mid \boldsymbol{X}_s] = \mathbb{E}_{\mathbb{M}}[f(\boldsymbol{X}_t) \mid \mathcal{F}_s] \right\rbrace
$$

In the Schroedinger bridge setting, we seek to construct a stochastic bridge between empirical endpoint distributions $\pi\_0$ and $\pi\_T$ that is close in relative entropy to a reference measure $\mathbb{Q}$ defined by the SDE $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$.

Given a coupling $\pi\_{0,T} \in \mathcal{P}(\mathbb{R}^d \times \mathbb{R}^d)$, we consider the path measure generated by a **mixture of endpoint-conditioned bridges** $\Pi = \pi\_{0,T}\mathbb{Q}\_{\cdot \mid 0,T}$. For fixed $(\boldsymbol{x}\_0, \boldsymbol{x}\_T)$, the **conditional bridge dynamics** under $\mathbb{Q}$ are given by the Doob's $h$-Transform SDE:

$$
\mathbb{Q}_{\cdot \mid 0,T}(\cdot \mid \boldsymbol{x}_0, \boldsymbol{x}_T) : \quad d\boldsymbol{X}_t = \left[\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t^2\nabla\log\mathbb{Q}_{T \mid t}(\boldsymbol{x}_T \mid \boldsymbol{X}_t)\right] dt + \sigma_t d\boldsymbol{B}_t
$$

Since the drift is **conditioned on the future endpoint** $\boldsymbol{x}\_T$, it is not Markov with respect to $\boldsymbol{X}\_t$. Simulating the SDE requires evaluating the transition density $\mathbb{Q}\_{T \mid t}(\boldsymbol{x}\_T \mid \boldsymbol{X}\_t)$ at every step, which is generally computationally intractable.

To obtain a tractable process, we construct the **Markovian projection of the bridge measure** $\Pi$, which is the Markov process whose drift depends only on the current state $\boldsymbol{X}\_t$ and that minimizes the KL divergence to the original bridge measure.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.10</span><span class="math-callout__name">(Markovian Projection)</span></p>

Consider a **mixture of bridges** $\Pi = \Pi\_{0,T}\mathbb{Q}\_{\cdot \mid 0,T}$ that bridges distributions $\pi\_0$ and $\pi\_T$ via the endpoint law $\Pi\_{0,T}$, where each conditional bridge is defined by the Bridge SDE generated from the reference measure. Then, the **Markovian projection** of $\Pi$ is denoted:

$$
\mathbb{M}^\star := \text{proj}_{\mathcal{M}}(\Pi) \in \mathcal{M}
$$

with the associated SDE:

$$
\mathbb{M}^\star : \quad d\boldsymbol{X}_t = [\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}^\star(\boldsymbol{X}_t, t)] dt + \sigma_t d\boldsymbol{B}_t
$$

$$
\text{s.t.} \quad \boldsymbol{u}^\star(\boldsymbol{x}, t) = \sigma_t \mathbb{E}_{\Pi_{T \mid t}}\!\left[\nabla\log\mathbb{Q}_{T \mid t}(\boldsymbol{X}_T \mid \boldsymbol{X}_t) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

where $\sigma\_t > 0$. Then, $\mathbb{M}^\star$ satisfies the following properties:

**(i)** It is the Markov measure $\mathbb{M} \in \mathcal{M}$ that **minimizes the reverse KL divergence** with the mixture of bridges $\Pi$:

$$
\mathbb{M}^\star = \arg\min_{\mathbb{M}} \lbrace \text{KL}(\Pi \mid \mathbb{M}) : \mathbb{M} \in \mathcal{M} \rbrace
$$

$$
\text{KL}(\Pi \mid \mathbb{M}) = \frac{1}{2}\int_0^T \mathbb{E}_{\Pi_{0,t}}\!\left[\left\lVert \sigma_t \mathbb{E}_{\Pi_{T \mid 0,t}}\!\left[\nabla\log\mathbb{Q}_{T \mid t}(\boldsymbol{X}_T \mid \boldsymbol{X}_t) \mid \boldsymbol{X}_0, \boldsymbol{X}_T\right] - \boldsymbol{u}^\star(\boldsymbol{X}_t, t) \right\rVert^2\right] dt
$$

**(ii)** It preserves the time marginals of $\Pi\_t$ for all $t \in [0, T]$: $\forall t \in [0, T], \; \mathbb{M}\_t^\star = \Pi\_t$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch (Markovian Projection)</summary>

**Step 1: Derive the Optimal Control Drift.** The bridge path measure can be obtained by reweighting the reference measure $\mathbb{Q}$ by the Radon--Nikodym derivative of the endpoint coupling $\Pi\_{0,T}$ with respect to the reference endpoint law $\mathbb{Q}\_{0,T}$. Conditioning on the initial state $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$, the conditional bridge law is obtained by reweighting reference diffusion by a terminal weight $h(\boldsymbol{X}\_T, T) := \frac{d\Pi(\boldsymbol{X}\_T \mid \boldsymbol{X}\_0 = \boldsymbol{x}\_0)}{d\mathbb{Q}(\boldsymbol{X}\_T \mid \boldsymbol{X}\_0 = \boldsymbol{x}\_0)}$. This is exactly the Doob's $h$-transform with:

$$
h(\boldsymbol{x}, t) = \int_{\mathbb{R}^d} \mathbb{Q}_{T \mid t}(\boldsymbol{X}_T = \boldsymbol{x}_T \mid \boldsymbol{X}_t = \boldsymbol{x}) h(\boldsymbol{X}_T, T) d\boldsymbol{x}_T = \mathbb{E}_{\mathbb{Q}}[h(\boldsymbol{X}_T, T) \mid \boldsymbol{X}_t = \boldsymbol{x}, \boldsymbol{X}_0 = \boldsymbol{x}_0]
$$

By Corollary 4.8, the bridge measure conditioned on $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$ follows the SDE $d\boldsymbol{X}\_t = [\boldsymbol{f}(\boldsymbol{X}\_t, t) + \sigma\_t^2\nabla\_{\boldsymbol{x}}\log h(\boldsymbol{X}\_t, t)] dt + \sigma\_t d\boldsymbol{B}\_t$.

The Markovian projection averages out the dependence on the endpoints by taking the conditional expectation of the bridge drift over the endpoint distribution $\Pi\_{T \mid t}$:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = \sigma_t\mathbb{E}_{\Pi_{T \mid t}}[\nabla\log\mathbb{Q}_{T \mid t}(\boldsymbol{X}_T \mid \boldsymbol{X}_t) \mid \boldsymbol{X}_t = \boldsymbol{x}]
$$

**Step 2: KL minimization and marginal preservation** follow from the fact that the Markovian projection preserves the conditional mean of the bridge drift (averaging over $\boldsymbol{X}\_T$) at each time, which simultaneously minimizes the KL divergence and preserves all time marginals. $\square$
</details>
</div>

While the Markovian projection provides a way of simulating endpoint-conditioned bridges with only dependence on the current state, it generally fails to preserve the **bridge measure** of $\mathbb{Q}\_{\cdot \mid 0,T} \equiv \mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$, which is the distribution over bridge paths conditioned on a pair of endpoints $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \Pi\_{0,T}$. To define a measure that exactly matches the bridge of the endpoint-conditioned reference measure $\mathbb{Q}\_{\cdot \mid 0,T}$, we define the **reciprocal projection** which projects any path measure to the **reciprocal class** $\mathcal{R}(\mathbb{Q})$ of $\mathbb{Q}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.11</span><span class="math-callout__name">(Reciprocal Processes)</span></p>

A stochastic process $(\boldsymbol{X}\_t)\_{t \in [0,T]}$ on a measurable state space $\mathcal{X}$ is considered a **reciprocal process** if for any $0 \le s < t < \tau \le T$ and any bounded measurable function $\phi(\boldsymbol{x}) : \mathcal{X} \to \mathbb{R}$, the **reciprocal property** holds:

$$
\mathbb{E}[\phi(\boldsymbol{X}_t) \mid \boldsymbol{X}_{0:s}, \boldsymbol{X}_{\tau:T}] = \mathbb{E}[\phi(\boldsymbol{X}_t) \mid \boldsymbol{X}_s, \boldsymbol{X}_\tau]
$$

which means that the interior of any interval $(s, \tau)$ is **conditionally independent** of the rest of the trajectory given the two boundary states $(\boldsymbol{X}\_s, \boldsymbol{X}\_\tau)$. While reciprocal processes are generally not Markov, fixing either boundary state ($\boldsymbol{X}\_s$ or $\boldsymbol{X}\_\tau$) to a constant yields a Markov process.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.12</span><span class="math-callout__name">(Reciprocal Class)</span></p>

The **reciprocal class** of $\mathbb{Q}$ is the collection of all path measures that share the same **conditional bridge distribution** given fixed endpoints $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \Pi\_{0,T}$ but may differ in their endpoint law, which defines how the endpoints are weighted. Formally:

$$
\mathcal{R}(\mathbb{Q}) := \lbrace \Pi \in \mathcal{P}(C([0,T]; \mathbb{R}^d)) : \Pi = \Pi_{0,T}\mathbb{Q}_{\cdot \mid 0,T} \rbrace
$$

where $\Pi\_{0,T} \in \mathcal{P}(\mathbb{R}^d, \mathbb{R}^d)$ denotes an arbitrary endpoint coupling and $\mathbb{Q}\_{\cdot \mid 0,T}$ is the endpoint-conditioned bridge distribution of $\mathbb{Q}$. Equivalently, a path measure belongs in the reciprocal class $\Pi \in \mathcal{R}(\mathbb{Q})$ if it admits the **mixture-of-bridges representation**: $\Pi = \Pi\_{0,T}\mathbb{Q}\_{\cdot \mid 0,T}$.

</div>

All path measures in the reciprocal class $\mathcal{R}(\mathbb{Q})$ generate trajectories with the **same bridge dynamics** as $\mathbb{Q}$, differing only in how probability mass is assigned to the endpoint pairs $(\boldsymbol{x}\_0, \boldsymbol{x}\_T)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.13</span><span class="math-callout__name">(Reciprocal Projection)</span></p>

Let $\mathbb{Q}$ be a reference path measure and $\mathbb{P}$ be an arbitrary path measure that we wish to project onto the reciprocal class of $\mathbb{Q}$. The **reciprocal projection** of $\mathbb{P}$ onto the reciprocal class $\mathcal{R}(\mathbb{Q})$ is defined as:

$$
\Pi^\star := \text{proj}_{\mathcal{R}(\mathbb{Q})}(\mathbb{P}) \in \mathcal{R}(\mathbb{Q})
$$

Then, $\Pi^\star$ is the element of the reciprocal class which **minimizes the KL divergence** from $\mathbb{P}$:

$$
\Pi^\star = \underset{\Pi \in \mathcal{R}(\mathbb{Q})}{\arg\min}\; \text{KL}(\mathbb{P} \| \Pi)
$$

Furthermore, the reciprocal projection admits the **mixture-of-bridges representation**:

$$
\Pi^\star(\boldsymbol{X}_{0:T}) = \int_{\mathbb{R}^d \times \mathbb{R}^d} \mathbb{Q}_{\cdot \mid 0,T}(\boldsymbol{X}_{0:T} \mid \boldsymbol{x}_0, \boldsymbol{x}_T) d\mathbb{P}_{0,T}(\boldsymbol{x}_0, \boldsymbol{x}_T) \iff \Pi^\star = \mathbb{P}_{0,T}\mathbb{Q}_{\cdot \mid 0,T}
$$

where $\mathbb{P}\_{0,T}$ denotes the endpoint distribution of $\mathbb{P}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Reciprocal Projection)</summary>

Apply the KL Divergence Chain Rule (Lemma 1.4) to decompose $\text{KL}(\mathbb{P} \| \Pi)$:

$$
\text{KL}(\mathbb{P} \| \Pi) = \underbrace{\text{KL}(\mathbb{P}_{0,T} \| \Pi_{0,T})}_{\text{endpoint divergence}} + \underbrace{\mathbb{E}_{(\boldsymbol{X}_0, \boldsymbol{X}_T) \sim \mathbb{P}_{0,T}}\!\left[\text{KL}(\mathbb{P}_{\cdot \mid 0,T}(\cdot \mid \boldsymbol{X}_0, \boldsymbol{X}_T) \| \Pi_{\cdot \mid 0,T}(\cdot \mid \boldsymbol{X}_0, \boldsymbol{X}_T))\right]}_{\text{bridge divergence}}
$$

Since all elements of $\mathcal{R}(\mathbb{Q})$ have the same conditional bridge as $\mathbb{Q}$, i.e., $\Pi\_{\cdot \mid 0,T}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T) = \mathbb{Q}\_{\cdot \mid 0,T}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$, the bridge divergence term does not depend on $\Pi$. The only term to minimize is $\text{KL}(\mathbb{P}\_{0,T} \| \Pi\_{0,T})$, which is *uniquely* minimized when $\Pi\_{0,T}^\star = \mathbb{P}\_{0,T}$ by strict convexity of KL divergence. Therefore: $\Pi^\star = \mathbb{P}\_{0,T}\mathbb{Q}\_{\cdot \mid 0,T}$. $\square$
</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.14</span><span class="math-callout__name">(Solution to Dynamic Schroedinger Bridge)</span></p>

The **Markov measure in the reciprocal class** of $\mathbb{Q}$, i.e., $\mathbb{M} \in \mathcal{R}(\mathbb{Q})$, that satisfies $\mathbb{M}\_0 = \pi\_0$ and $\mathbb{M}\_T = \pi\_T$ is the unique solution to the Schroedinger bridge $\mathbb{M} = \mathbb{P}^\star$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Solution to Dynamic SB)</summary>

By definition of $\mathcal{R}(\mathbb{Q})$, $\mathbb{M}$ and $\mathbb{Q}$ share the exact same bridge such that $\mathbb{M}\_{\cdot \mid 0,T} = \mathbb{Q}\_{\cdot \mid 0,T}$ for all $t \in [0, T]$. Thus, the Radon--Nikodym derivative $\frac{d\mathbb{M}}{d\mathbb{Q}}(\boldsymbol{X}\_{0:T})$ depends only on the endpoints $(\boldsymbol{X}\_0, \boldsymbol{X}\_T)$ and can be written as $\frac{d\mathbb{M}}{d\mathbb{Q}} = \xi(\boldsymbol{X}\_0, \boldsymbol{X}\_T)$.

Since $\mathbb{P}$ is defined to be Markov and $\mathbb{Q}$ is Markov by construction, conditioning on an intermediate state $\boldsymbol{X}\_t$ yields a factorization where the past and future given $\boldsymbol{X}\_t$ are **independent**. Given that the RND depends only on the endpoints, there exist two measurable functions $a, b : \mathbb{R}^d \to \mathbb{R}$ such that:

$$
\frac{d\mathbb{M}}{d\mathbb{Q}}(\boldsymbol{X}_{0:T}) = \xi(\boldsymbol{X}_0, \boldsymbol{X}_T) = a(\boldsymbol{X}_0)b(\boldsymbol{X}_T)
$$

Recall from Section 1 that the optimal endpoint law $\pi\_{0,T}^\star$ that solves the static SB problem factorizes into Schroedinger potentials $(\varphi, \hat{\varphi})$ that are **unique** up to a constant. Therefore:

$$
\frac{d\mathbb{M}}{d\mathbb{Q}}(\boldsymbol{X}_{0:T}) = \frac{d\pi_{0,T}^\star}{dq}(\boldsymbol{X}_0, \boldsymbol{X}_T) = e^{\varphi(\boldsymbol{X}_0)}e^{\hat{\varphi}(\boldsymbol{X}_T)}
$$

and $\mathbb{M} \in \mathcal{R}(\mathbb{Q})$ is *unique* and equals the Schroedinger bridge $\mathbb{P}^\star$. $\square$
</details>
</div>

Building on the ideas of Markovian and reciprocal projections, we can define the **Iterative Markovian Fitting** (IMF) scheme which constructs the optimal SB via alternating Markovian and reciprocal projections of an SDE. IMF generates a sequence of path measures $(\mathbb{P}^n)\_{n \in \mathbb{N}}$ with **alternating Markovian projections and reciprocal projections**:

$$
\mathbb{P}^{2n+1} = \text{proj}_{\mathcal{M}}(\mathbb{P}^{2n}), \quad \mathbb{P}^{2n+2} = \text{proj}_{\mathcal{R}(\mathbb{Q})}(\mathbb{P}^{2n+1})
$$

where the first path measure $\mathbb{P}^0 \in \mathcal{R}(\mathbb{Q})$ satisfies $\mathbb{P}\_0^0 = \pi\_0$ and $\mathbb{P}\_T^0 = \pi\_T$. The IMF procedure is grounded in three key theoretical results:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.15</span><span class="math-callout__name">(Pythagorean Theorem of Markovian and Reciprocal Projections)</span></p>

Let $\Pi \in \mathcal{R}(\mathbb{Q})$ be a bridge measure in the reciprocal class of $\mathbb{Q}$ and $\text{proj}\_{\mathcal{M}}(\Pi)$ be the Markovian projection of $\Pi$. Given some arbitrary Markov measure $\mathbb{M} \in \mathcal{M}$ that has finite KL divergence with $\Pi$ and $\text{proj}\_{\mathcal{M}}(\Pi)$, the following identity holds:

$$
\text{KL}(\Pi \| \mathbb{M}) = \text{KL}(\Pi \| \text{proj}_{\mathcal{M}}(\Pi)) + \text{KL}(\text{proj}_{\mathcal{M}}(\Pi) \| \mathbb{M})
$$

Similarly, for any arbitrary path measure $\mathbb{P} \in \mathcal{P}(C([0,T]; \mathbb{R}^d))$ projected onto the reciprocal class $\text{proj}\_{\mathcal{R}(\mathbb{Q})}(\mathbb{P})$, the KL divergence with $\Pi \in \mathcal{R}(\mathbb{Q})$ can be decomposed as:

$$
\text{KL}(\mathbb{P} \| \Pi) = \text{KL}(\mathbb{P} \| \text{proj}_{\mathcal{R}(\mathbb{Q})}(\mathbb{P})) + \text{KL}(\text{proj}_{\mathcal{R}(\mathbb{Q})}(\mathbb{P}) \| \Pi)
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.16</span><span class="math-callout__name">(Monotonically Decreasing KL Divergence)</span></p>

Given a sequence of Markov and reciprocal projections $(\mathbb{P}^n)\_{n \in \mathbb{N}}$ and the Schroedinger bridge path measure $\mathbb{P}^\star$, the reverse KL divergences between $\mathbb{P}^n$ and $\mathbb{P}^\star$ decreases monotonically:

$$
\forall n \in \mathbb{N}, \quad \text{KL}(\mathbb{P}^{n+1} \| \mathbb{P}^\star) \le \text{KL}(\mathbb{P}^n \| \mathbb{P}^\star) \le \infty
$$

and in the limit $n \to \infty$, the KL divergence between subsequent projections converges to zero:

$$
\lim_{n \to \infty} \text{KL}(\mathbb{P}^n \| \mathbb{P}^{n+1}) = 0
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Monotonically Decreasing KL Divergence)</summary>

Since the Schroedinger bridge $\mathbb{P}^\star \in \mathcal{M}$ and $\mathbb{P}^\star \in \mathcal{R}(\mathbb{Q})$, both Pythagorean identities from Lemma 4.15 hold with respect to $\mathbb{P}^\star$. For all $\mathbb{P}^n$:

$$
\text{KL}(\mathbb{P}^n \| \mathbb{P}^\star) = \underbrace{\text{KL}(\mathbb{P}^n \| \mathbb{P}^{n+1})}_{\ge 0} + \text{KL}(\mathbb{P}^{n+1} \| \mathbb{P}^\star) \implies \text{KL}(\mathbb{P}^n \| \mathbb{P}^\star) \ge \text{KL}(\mathbb{P}^{n+1} \| \mathbb{P}^\star)
$$

Applying the identity for each iteration $n = 0, \ldots, N$ and using a telescoping sum:

$$
\text{KL}(\mathbb{P}^0 \| \mathbb{P}^\star) = \sum_{n=0}^N \text{KL}(\mathbb{P}^n \| \mathbb{P}^{n+1}) + \text{KL}(\mathbb{P}^{N+1} \| \mathbb{P}^\star) \implies \sum_{n=0}^N \text{KL}(\mathbb{P}^n \| \mathbb{P}^{n+1}) \le \text{KL}(\mathbb{P}^0 \| \mathbb{P}^\star) < \infty
$$

Since the series of non-negative KL divergences is finite and bounded, the additive terms must converge to zero: $\lim\_{n \to \infty} \text{KL}(\mathbb{P}^n \| \mathbb{P}^{n+1}) = 0$. $\square$
</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.17</span><span class="math-callout__name">(Iterative Markovian Fitting Converges to the Unique Schroedinger Bridge)</span></p>

The sequence of path measures $(\mathbb{P}^n)\_{n \in \mathbb{N}}$ generated from alternating Markovian and reciprocal projections of the IMF algorithm has a unique fixed point $\mathbb{P}^\star$ which equals the Schroedinger bridge. Furthermore, in the limit $n \to \infty$, the KL divergence converges to the fixed point:

$$
\lim_{n \to \infty} \text{KL}(\mathbb{P}^n \| \mathbb{P}^\star) = 0
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch (IMF Convergence)</summary>

By Proposition 4.16, each path measure in the sequence $(\mathbb{P}^n)\_{n \in \mathbb{N}}$ decreases the KL divergence to $\mathbb{P}^\star$, so the sequence remains trapped in a compact region in path space that is **bounded below by zero**. Both the sequence of Markovian projections and reciprocal projections converge to their optimal fixed points $\mathbb{M}^\star \in \mathcal{M}$ and $\Pi^\star \in \mathcal{R}(\mathbb{Q})$. From Proposition 4.16, the KL divergence between each iteration converges to zero, which implies:

$$
\lim_{n \to \infty} \text{KL}(\mathbb{P}^n \| \mathbb{P}^{n+1}) = 0 \implies \text{KL}(\mathbb{M}^\star \| \Pi) = 0 \implies \mathbb{M}^\star = \Pi^\star = \mathbb{P}^\star
$$

which means the shared limit of the Markov and reciprocal projections is both Markov and in the reciprocal class $\mathcal{R}(\mathbb{Q})$, and therefore, must be the Schroedinger bridge $\mathbb{P}^\star$. Given that both subsequences converge to $\mathbb{P}^\star$, the full sequence $(\mathbb{P}^n)\_{n \in \mathbb{N}}$ also converges to $\mathbb{P}^\star$. $\square$
</details>
</div>

The relationship between Markov and reciprocal projections reveals a **key structural property** of the Schroedinger bridge. The Markovian Projection enforces the Markov property by selecting the closest Markov process in relative entropy to some bridge measure, and the Reciprocal Projection adjusts the path measure so that the endpoint marginals match the prescribed distributions, while preserving the bridge structure inherited from the reference process.

Crucially, the Schroedinger bridge $\mathbb{P}^\star$ lies exactly at the **equilibrium** of these two constraints, as it is the unique path measure that simultaneously satisfies the endpoint conditions and remains the closest Markov measure to the reference bridge dynamics. The IMF procedure will be revisited in Section 6.3, where we apply it in the context of generative modeling.

### 4.6 Stochastic Interpolants to Schroedinger Bridges

The **stochastic interpolants** framework can be used to construct the Schroedinger bridge solution.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.18</span><span class="math-callout__name">(Stochastic Interpolant)</span></p>

Let $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$ be two probability densities on the state space. The **stochastic interpolant** between $\pi\_0$ and $\pi\_T$ is a stochastic process $\boldsymbol{X}\_{0:T}$ of the form:

$$
\boldsymbol{x}_t = I(\boldsymbol{x}_0, \boldsymbol{x}_T, t) + \gamma(t)\boldsymbol{z}, \quad t \in [0, T]
$$

where the following are satisfied:

- **(i)** The map $I \in C^2((C^2(\mathbb{R}^d \times \mathbb{R}^d))^d, [0, T])$ has boundary conditions $I(\boldsymbol{x}\_0, \boldsymbol{x}\_T, 0) = \boldsymbol{x}\_0$ and $I(\boldsymbol{x}\_0, \boldsymbol{x}\_T, T) = \boldsymbol{x}\_T$ and controlled time variation: $\exists C\_1 < \infty$ s.t. $\lvert \partial\_t I(\boldsymbol{x}\_0, \boldsymbol{x}\_T, t) \rvert \le C\_1 \lvert \boldsymbol{x}\_T - \boldsymbol{x}\_0 \rvert$.

- **(ii)** The scalar noise function $\gamma : [0, T] \to \mathbb{R}$ satisfies $\gamma(0) = \gamma(1) = 0$ and $\gamma(t) > 0$ for all $t \in (0, T)$. In addition, $\gamma^2 \in C^2([0, T])$.

- **(iii)** The pair $(\boldsymbol{x}\_0, \boldsymbol{x}\_T)$ are sampled from a probability measure $\pi\_{0,T}$ whose marginals are $\pi\_0$ and $\pi\_T$.

- **(iv)** The Gaussian random variable $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, I\_d)$ is independent of $(\boldsymbol{x}\_0, \boldsymbol{x}\_T)$.

</div>

Intuitively, the stochastic interpolant connects samples from two marginal distributions with a **deterministic path** $I(\boldsymbol{x}\_0, \boldsymbol{x}\_T, t)$ that is perturbed by a time-dependent Gaussian diffusion $\gamma(t)\boldsymbol{z}$ along the interior of the time interval. By definition, the noise $\gamma(t)\boldsymbol{z}$ vanishes at $t = 0$ and $t = T$, ensuring that the process matches exactly the terminal marginals $\pi\_0$ and $\pi\_T$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.19</span><span class="math-callout__name">(Properties of Stochastic Interpolants)</span></p>

The stochastic interpolant $\boldsymbol{x}\_t = I(\boldsymbol{x}\_0, \boldsymbol{x}\_T, t)$ satisfies the following properties:

$$
\partial_t p_t + \nabla \cdot (p_t \boldsymbol{v}) = 0
$$

where the velocity is defined as the expectation of the time derivative:

$$
\boldsymbol{v}(\boldsymbol{x}, t) = \mathbb{E}_{p_t}[\dot{\boldsymbol{x}}_t \mid \boldsymbol{x}_t = \boldsymbol{x}] = \mathbb{E}[\partial_t I(\boldsymbol{x}_0, \boldsymbol{x}_T, t) + \dot{\gamma}(t)\boldsymbol{z} \mid \boldsymbol{x}_t = \boldsymbol{x}]
$$

which is bounded on the domain of the density function $p\_t(\boldsymbol{x})$: $\forall t \in [0, T] : \int\_{\mathbb{R}^d} \lVert \boldsymbol{v}(\boldsymbol{x}, t) \rVert^2 p\_t(\boldsymbol{x}) d\boldsymbol{x} < \infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch (Properties of Stochastic Interpolants)</summary>

The proof uses the Fourier transform $\mathcal{F}(\boldsymbol{\omega}, t)[p\_t] = \int\_{\mathbb{R}^d} e^{i\boldsymbol{\omega} \cdot \boldsymbol{x}} p\_t(\boldsymbol{x}) = \mathbb{E}\left[e^{i\boldsymbol{\omega} \cdot (I(\boldsymbol{x}\_0, \boldsymbol{x}\_T, t) + \gamma(t)\boldsymbol{z})}\right]$. Taking the time derivative:

$$
\partial_t \mathcal{F}(\boldsymbol{\omega}, t)[p_t] = i\boldsymbol{\omega} \cdot \mathbb{E}\left[e^{i\boldsymbol{\omega} \cdot \boldsymbol{x}_t}(\partial_t I(\boldsymbol{x}_0, \boldsymbol{x}_T, t) + \dot{\gamma}(t)\boldsymbol{z})\right]
$$

By the law of total expectation, conditioning on $\boldsymbol{X}\_t = \boldsymbol{x}$:

$$
\partial_t \mathcal{F}(\boldsymbol{\omega}, t)[p_t] = i\boldsymbol{\omega} \cdot \int_{\mathbb{R}^d} e^{i\boldsymbol{\omega} \cdot \boldsymbol{x}} \underbrace{\mathbb{E}[(\partial_t I + \dot{\gamma}\boldsymbol{z}) \mid \boldsymbol{X}_t = \boldsymbol{x}]}_{=: \boldsymbol{v}(\boldsymbol{x}, t)} p_t(\boldsymbol{x}) d\boldsymbol{x} = i\boldsymbol{\omega} \cdot \int_{\mathbb{R}^d} e^{i\boldsymbol{\omega} \cdot \boldsymbol{x}} \boldsymbol{v} p_t \, d\boldsymbol{x}
$$

Since the Fourier transform of $-\nabla \cdot (\boldsymbol{v}p\_t)$ is exactly $i\boldsymbol{\omega}\mathcal{F}(\boldsymbol{\omega}, t)[\boldsymbol{v}p\_t]$ (shown by integration by parts), we recover $\partial\_t \mathcal{F}[p\_t] = \mathcal{F}[-\nabla \cdot (\boldsymbol{v}p\_t)]$, which implies the continuity equation in real space: $\partial\_t p\_t = -\nabla \cdot (\boldsymbol{v}p\_t)$. $\square$
</details>
</div>

Stochastic interpolants induce a family of time-evolving densities $(p\_t)\_{t \in [0,T]}$ governed by the continuity equation. This characterizes deterministic mass transport under a velocity field $\boldsymbol{v}$, and forms the core dynamical constraint underlying optimal transport. While stochastic interpolants describe valid transport dynamics, they do not yet specify which trajectory is *optimal*. The SB problem resolves this ambiguity by selecting the one that is closest to a reference stochastic process in the sense of minimizing path-space KL divergence.

We now show how the SB problem can be solved via stochastic interpolants. We simplify the setting by setting $\boldsymbol{f} \equiv \boldsymbol{0}$ and $\epsilon = \frac{\sigma\_t^2}{2}$, so the HJB-FP system becomes:

$$
\begin{cases} \partial_t \psi_t + \frac{1}{2}\lVert \nabla\psi_t \rVert^2 = -\epsilon\Delta\psi_t \\[4pt] \partial_t p_t^\star + \nabla \cdot (p_t^\star \boldsymbol{u}^\star) = \epsilon\Delta p_t^\star \end{cases} \quad \text{s.t.} \quad \begin{cases} p_0^\star = \pi_0 \\ p_T^\star = \pi_T \end{cases}
$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.20</span><span class="math-callout__name">(Invertible Map)</span></p>

Define an invertible map $M : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ where $M, M^{-1} \in C^1([0, T], (C^d(\mathbb{R}^d))^d)$ such that:

$$
p_t^\star(\boldsymbol{x}) = M(\cdot, t)_\# \mathcal{N}(\boldsymbol{0}, I_d)
$$

In other words, given a Gaussian random variable $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, I\_d)$, we have $\boldsymbol{x}\_t = M(\boldsymbol{z}, t) \sim p\_t^\star$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.21</span><span class="math-callout__name">(Stochastic Interpolant Form of SB Solution)</span></p>

Given the existence of an invertible map $M$ defined in Definition 4.20, the optimal density $p\_t^\star$ that solves the dynamic SB problem can be written as a **stochastic interpolant** of the form:

$$
\boldsymbol{x}_t = M\!\left(\alpha(t)M^{-1}(\boldsymbol{x}_0, 0) + \beta(t)M^{-1}(\boldsymbol{x}_T, T), t\right) + \gamma(t)
$$

where $\alpha^2(t) + \beta^2(t) + \gamma^2(t) = 1$. This corresponds to defining the interpolant function from Definition 4.18 as $I(\boldsymbol{x}\_0, \boldsymbol{x}\_T, t) = M(\alpha(t)M^{-1}(\boldsymbol{x}\_0, 0) + \beta(t)M^{-1}(\boldsymbol{x}\_T, T), t)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (Stochastic Interpolant Form of SB Solution)</summary>

The density $p\_t^\star$ that solves the HJB-FP system is exactly the distribution of the random variable defined in the interpolant. By definition of the map $M(\cdot, t)$ which **transports** a standard Gaussian to the target density $p\_t^\star$ at time $t$, the **inverse map** $M^{-1}(\cdot, t)$ must transport $p\_t^\star$ back to a standard Gaussian:

$$
\boldsymbol{x}_0 \sim \pi_0, \quad M^{-1}(\boldsymbol{x}_0, 0) \sim \mathcal{N}(\boldsymbol{0}, I_d); \qquad \boldsymbol{x}_T \sim \pi_T, \quad M^{-1}(\boldsymbol{x}_T, T) \sim \mathcal{N}(\boldsymbol{0}, I_d)
$$

Since $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, I\_d)$ is also sampled from a standard Gaussian and $\boldsymbol{x}\_0, \boldsymbol{x}\_T, \boldsymbol{z}$ are drawn independently, the linear combination of independent standard Gaussians:

$$
\alpha(t)M^{-1}(\boldsymbol{x}_0, 0) + \beta(t)M^{-1}(\boldsymbol{x}_T, T) + \gamma(t)\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, (\alpha^2(t) + \beta^2(t) + \gamma^2(t))I_d) = \mathcal{N}(\boldsymbol{0}, I_d)
$$

since $\alpha^2(t) + \beta^2(t) + \gamma^2(t) = 1$. Applying $M(\cdot, t)$ to this standard Gaussian yields a random variable from $p\_t^\star$. $\square$
</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.22</span><span class="math-callout__name">(Solving SB Problem with Stochastic Interpolants)</span></p>

Define a scalar function $\gamma(t) : [0, T] \to [0, 1)$ which returns zero at the terminal time points (i.e., $\gamma(0) = \gamma(T) = 0$), returns non-zero at intermediate times (i.e., $\forall t \in (0, T), \gamma(t) > 0$), and satisfies $\gamma \in C^2((0, T))$ and $\gamma^2 \in C^1([0, T])$.

Then, given independent $\boldsymbol{x}\_0, \boldsymbol{x}\_T, \boldsymbol{z}$ solving the max-min problem over $\hat{I} \in C^1([0, T], (C^1(\mathbb{R}^d \times \mathbb{R}^d))^d)$ and $\hat{\boldsymbol{u}} \in C^0([0, T], (C^1(\mathbb{R}^d))^d)$ given by:

$$
\max_{\hat{I}} \min_{\hat{\boldsymbol{u}}} \int_0^T \mathbb{E}\!\left[\frac{1}{2}\lVert \hat{\boldsymbol{u}}(\hat{\boldsymbol{x}}_t, t) \rVert^2 - \left(\partial_t \hat{I}(\boldsymbol{x}_0, \boldsymbol{x}_T, t) + (\dot{\gamma}(t) - \epsilon\gamma^{-1}(t))\boldsymbol{z}\right) \cdot \hat{\boldsymbol{u}}(\hat{\boldsymbol{x}}_t, t)\right] dt
$$

$$
\text{s.t.} \quad \boldsymbol{x}_t = \hat{I}(t, \boldsymbol{x}_0, \boldsymbol{x}_T) + \gamma(t)\boldsymbol{z}, \quad \boldsymbol{x}_0 \sim \pi_0, \; \boldsymbol{x}_T \sim \pi_T, \; \boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, I_d)
$$

where, given the existence of the invertible map $M$, all optimal $(I^\star, \boldsymbol{u}^\star)$ produces the stochastic interpolant $\boldsymbol{x}\_t = I^\star(\boldsymbol{x}\_0, \boldsymbol{x}\_T, t) + \gamma(t)\boldsymbol{z}$ with marginals $p\_t^\star$ that satisfy the continuity equation $\partial\_t p\_t = -\nabla \cdot (\hat{p}\_t \hat{\boldsymbol{v}})$ where $\hat{\boldsymbol{v}}$ is the effective velocity field that accounts for the Gaussian perturbation induced by $\gamma(t)\boldsymbol{z}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch (Solving SB with Stochastic Interpolants)</summary>

**Step 1:** Define the effective velocity $\hat{\boldsymbol{v}}(\boldsymbol{x}, t) := \mathbb{E}[\partial\_t \hat{I}(\boldsymbol{x}\_0, \boldsymbol{x}\_T, t) + (\dot{\gamma}(t) - \epsilon\gamma^{-1}(t))\boldsymbol{z} \mid \boldsymbol{X}\_t = \boldsymbol{x}]$, where $-\epsilon\gamma^{-1}(t)$ **corrects for the Gaussian perturbation**. This uses the identity $-\epsilon\gamma^{-1}(t)\mathbb{E}[\boldsymbol{z} \mid \boldsymbol{X}\_t = \boldsymbol{x}] = \epsilon\nabla\log\hat{p}\_t(\boldsymbol{x})$ for the conditional Gaussian.

**Step 2:** Rewrite the Fokker--Planck equation as a continuity equation with effective velocity: $\partial\_t p\_t = -\nabla \cdot (\hat{p}\_t(\boldsymbol{v} - \epsilon\nabla\log\hat{p}\_t)) = -\nabla \cdot (\hat{p}\_t\hat{\boldsymbol{v}})$.

**Step 3:** Write as unconstrained Lagrangian with multipliers $\psi\_t(\boldsymbol{x})$, $\eta\_0(\boldsymbol{x})$, $\eta\_T(\boldsymbol{x})$ for the continuity equation and boundary constraints. Taking optimality conditions:

- Varying w.r.t. $\hat{\boldsymbol{u}}$: $\boldsymbol{u}^\star = \hat{\boldsymbol{v}}$
- Varying w.r.t. $\hat{p}\_t$: $\partial\_t\psi\_t + \frac{1}{2}\lVert \boldsymbol{u}^\star \rVert^2 = 0$
- Integration by parts on the $\hat{\boldsymbol{u}}$ and $\psi\_t$ terms: $\nabla\psi\_t - \boldsymbol{u}^\star = 0 \implies \boldsymbol{u}^\star = \nabla\psi\_t$

Combining (ii) and (iii) yields the system $\partial\_t\psi\_t + \frac{1}{2}\lVert \nabla\psi\_t \rVert^2 = 0$ and $\partial\_t p\_t^\star + \nabla \cdot (p\_t^\star\nabla\psi\_t) = 0$, which is exactly the HJB-FP system with vanishing diffusion absorbed into the effective velocity $\hat{\boldsymbol{v}}$. $\square$
</details>
</div>

This result shows that stochastic interpolants provide a constructive representation of the Schroedinger bridge dynamics $(\boldsymbol{u}^\star, p\_t^\star)$, where the optimal control and velocity fields coincide and are given by the gradient of the Lagrange multiplier $\nabla\psi\_t$, yielding the coupled HJB-FP system that defines the optimality conditions of the SB solution. Sampling from the stochastic interpolant $\boldsymbol{x}\_t = I^\star(\boldsymbol{x}\_0, \boldsymbol{x}\_T, t) + \gamma(t)\boldsymbol{z}$, where $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, I\_d)$ generates trajectories whose marginals match the marginal density flow $p\_t^\star$ of the SB path measure $\mathbb{P}^\star$.

### 4.7 Closing Remarks for Section 4

This section explored the theoretical foundations for building a Schroedinger bridge using several approaches. While each approach originates from a different mathematical viewpoint, they ultimately converge to a unified form of a Markov control drift that minimally corrects the uncontrolled reference dynamics such that they reconstruct the prescribed marginal distributions.

The **key takeaway** is that stochastic bridges are not arbitrary conditioned processes, but rather minimal-entropy corrections of a reference diffusion that preserve the bridge structure while introducing the smallest possible dynamical adjustment. This adjustment consistently appears as a gradient of a logarithmic potential, providing a unifying perspective to the structure of the Schroedinger bridge.

The different constructions can be understood as alternative ways of identifying the optimal control drift that modifies the reference dynamics:

1. **Mixture of Conditional Bridges (Section 4.1):** The dynamic SB $\mathbb{P}^\star$ can be expressed as a mixture of endpoint-conditioned stochastic bridges given samples from the optimal endpoint law $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \pi\_{0,T}^\star$, separating the SB problem into estimating the optimal static coupling and learning the conditional bridge dynamics.

2. **Time-reversal formula (Section 4.2):** The time-reversal yields a backward correction term $\nabla\log p\_{T-s}(\boldsymbol{x})$ that corresponds to the *score function* in score-based generative modeling. This formulation models the endpoint-conditioned bridge for *uncontrolled* forward processes.

3. **Forward-backward SDEs (Section 4.3):** FBSDE theory generalizes the time-reversal formula for a forward process containing a non-deterministic control drift $\nabla\log\varphi\_t(\boldsymbol{x})$ which yields a backward control drift $\nabla\log\hat{\varphi}\_t(\boldsymbol{x})$ that evolves via a system of forward-backward SDEs.

4. **Doob's $h$-transform (Section 4.4):** The $h$-function $h(\boldsymbol{x}, t)$ *reweights path transitions* by its potential at time $\tau$. By defining $h(\boldsymbol{x}, t) := \mathbb{E}\_{\mathbb{Q}}[\varphi\_T(\boldsymbol{X}\_T) \mid \boldsymbol{X}\_t = \boldsymbol{x}]$, the tilted path measure recovers the Schroedinger bridge $\mathbb{P}^\star$.

5. **Markov and reciprocal projections (Section 4.5):** The optimal Schroedinger bridge $\mathbb{P}^\star$ is the equilibrium point between projections onto the space of Markov path measures and the reciprocal class $\mathcal{R}(\mathbb{Q})$. The optimal drift is an expectation over the target-conditioned path measure $\mathbb{E}\_{\Pi\_{T \mid t}}[\nabla\log\mathbb{Q}\_{T \mid t}(\boldsymbol{X}\_T \mid \boldsymbol{X}\_t) \mid \tilde{\boldsymbol{X}}\_t]$.

6. **Stochastic interpolants (Section 4.6):** The stochastic interpolant framework represents the bridge by expressing the intermediate state as $\boldsymbol{x}\_t = I^\star(\boldsymbol{x}\_0, \boldsymbol{x}\_T, t) + \gamma(t)\boldsymbol{z}$, where the induced velocity field satisfies the same optimality conditions that characterize the Schroedinger bridge dynamics.

While this section provides a principled framework for constructing stochastic bridges between prescribed endpoint distributions that solve the Dynamic SB Problem, recent advances in generative modeling have motivated a variety of **specialized Schroedinger bridge formulations** tailored to different modeling assumptions and problem settings. These variants extend the original framework in several directions, ranging from alternative reference processes to mean-field interactions, unbalanced mass transport, and multi-marginal and multi-modal constraints.

## 5. Variations of the Schroedinger Bridge Problem

In previous sections, we have established the foundational theories and intuition behind the classical static and dynamic Schroedinger bridge problem and have shown how to derive stochastic bridges from scratch using various techniques. Now, we are ready to describe diverse variations of the SB problem that have been introduced in conjunction with novel generative modeling techniques, each of which are specialized for different settings and tasks.

Specifically, we analyze the Gaussian SB problem (Section 5.1), the generalized SB problem (Section 5.2), the multi-marginal SB problem (Section 5.3), the unbalanced SB problem (Section 5.4), the branched SB problem (Section 5.5), and finally the fractional SB problem (Section 5.6).

### 5.1 Gaussian Schroedinger Bridge Problem

While the Dynamic SB Problem defined in Section 2 does not admit a closed form solution in general, in the special case where the marginal distributions $\pi\_0, \pi\_T$ are Gaussian distributions defined as $\pi\_0 \sim \mathcal{N}(\boldsymbol{\mu}\_0, \boldsymbol{\Sigma}\_0)$, $\pi\_T \sim \mathcal{N}(\boldsymbol{\mu}\_T, \boldsymbol{\Sigma}\_T)$, the SB solution can be solved in closed form. This special case is called the **Gaussian Schroedinger Bridge** (SB) problem.

We start by defining the **Gaussian formulation of the entropic OT problem**, which will become crucial for our later derivation of the closed-form solution in the Gaussian SB setting.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.1</span><span class="math-callout__name">(Static Entropy-Regularized Gaussian Optimal Transport)</span></p>

Let $\pi\_0 = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\pi\_T = \mathcal{N}(\boldsymbol{\mu}', \boldsymbol{\Sigma}')$ be Gaussian probability measures on $\mathbb{R}^d$. Consider the entropy-regularized optimal transport problem:

$$
\min_{\pi_{0,T} \in \Pi(\pi_0, \pi_T)} \int \lVert \boldsymbol{x}_T - \boldsymbol{x}_0 \rVert^2 d\pi(\boldsymbol{x}_0, \boldsymbol{x}_T) + 2\sigma^2 \text{KL}(\pi_{0,T} \| \pi_0 \otimes \pi_T)
$$

where $\sigma \ge 0$ and $\Pi(\pi\_0, \pi\_T)$ denotes the set of couplings with marginals $\pi\_0$ and $\pi\_T$. Then, the unique optimal coupling $\pi\_{0,T}^\star$ is Gaussian and satisfies:

$$
\pi_{0,T}^\star \sim \mathcal{N}\!\left(\begin{bmatrix} \boldsymbol{\mu}_0 \\ \boldsymbol{\mu}_T \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_0 & \boldsymbol{C}_\sigma \\ \boldsymbol{C}_\sigma^\top & \boldsymbol{\Sigma}_T \end{bmatrix}\right)
$$

where

$$
\boldsymbol{C}_\sigma := \frac{1}{2}\left(\boldsymbol{\Sigma}_0^{1/2}\boldsymbol{D}_\sigma\boldsymbol{\Sigma}_0^{-1/2} - \sigma^2 I_d\right), \quad \boldsymbol{D}_\sigma := \left(4\boldsymbol{\Sigma}_0^{1/2}\boldsymbol{\Sigma}_T\boldsymbol{\Sigma}_0^{1/2} + \sigma^4 I_d\right)^{1/2}
$$

In particular, when $\sigma = 0$, the solution reduces to the classical Gaussian optimal transport coupling with quadratic transport cost.

</div>

The objective balances two competing effects: the quadratic cost $\lVert \boldsymbol{x}\_T - \boldsymbol{x}\_0 \rVert^2$ encourages pairs $(\boldsymbol{x}\_0, \boldsymbol{x}\_T)$ to be as close as possible, whereas the entropy regularization term $\text{KL}(\pi\_{0,T} \| \pi\_0 \otimes \pi\_T)$ penalizes deviations from independence. The optimal cross-covariance $\boldsymbol{C}\_\sigma$ can be interpreted as the optimal trade-off between these two objectives.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.2</span><span class="math-callout__name">(Gaussian Schroedinger Bridge Problem)</span></p>

Let $\pi\_0 = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\pi\_T = \mathcal{N}(\boldsymbol{\mu}', \boldsymbol{\Sigma}')$ be Gaussian probability measures on $\mathbb{R}^d$ and let $\mathbb{Q}$ be a reference path measure. The **Gaussian Schroedinger bridge** seeks the path measure that matches the Gaussian marginals while minimizing the relative entropy with respect to $\mathbb{Q}$:

$$
\mathbb{P}^\star = \underset{\mathbb{P} \in \mathcal{P}(C([0,T]; \mathbb{R}^d))}{\arg\min} \left\lbrace \text{KL}(\mathbb{P} \| \mathbb{Q}) : \pi_0 = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \; \pi_T = \mathcal{N}(\boldsymbol{\mu}', \boldsymbol{\Sigma}') \right\rbrace
$$

which can also be written in the form of Entropy-Regularized Dynamic OT (setting $\boldsymbol{f} \equiv \boldsymbol{0}$):

$$
\inf_{(p_t, \boldsymbol{v})} \int_0^T \mathbb{E}_{p_t}\!\left[\frac{1}{2}\lVert \boldsymbol{v}(\boldsymbol{x}, t) \rVert^2 + \frac{\sigma_t^4}{8}\lVert \nabla\log p_t(\boldsymbol{x}) \rVert^2\right] dt \quad \text{s.t.} \quad \begin{cases} \partial_t p_t = -\nabla \cdot (p_t \boldsymbol{v}) \\ p_0 = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \; p_T = \mathcal{N}(\boldsymbol{\mu}', \boldsymbol{\Sigma}') \end{cases}
$$

</div>

To obtain a tractable characterization, we exploit the special structure of Gaussian measures, which are uniquely defined by their mean and covariance. When the marginals remain Gaussian along the interpolation, the evolution is fully characterized by the trajectories of its **mean** $\boldsymbol{\mu}\_t \in \mathbb{R}^d$ and **covariance** $\boldsymbol{\Sigma}\_t \in \mathbb{S}\_{++}^d$. The Gaussian Schroedinger bridge can be interpreted as minimizing the energy of the change in covariance matrices as they move along the manifold of symmetric positive definite matrices $\boldsymbol{\Sigma} \in \mathbb{S}\_{++}^d$, known as the **Bures--Wasserstein manifold**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.3</span><span class="math-callout__name">(Lyapunov Operator and Bures--Wasserstein Manifold)</span></p>

Given a covariance matrix $\boldsymbol{\Sigma} \in \mathbb{S}\_{++}^d$ and tangent matrix $\boldsymbol{U} \in \mathcal{T}\_{\boldsymbol{\Sigma}}\mathbb{S}\_{++}^d$, the Lyapunov operator $\mathcal{L}\_{\boldsymbol{\Sigma}}[\boldsymbol{U}] : \mathcal{T}\_{\boldsymbol{\Sigma}}\mathbb{S}\_{++}^d \to \mathbb{S}\_{++}^d$ is the operator that returns the unique symmetric matrix $\boldsymbol{A}$ that solves:

$$
\boldsymbol{\Sigma}\boldsymbol{A} + \boldsymbol{A}\boldsymbol{\Sigma} = \boldsymbol{U}
$$

which defines the Riemannian metric of the Bures--Wasserstein manifold given by:

$$
\langle \boldsymbol{U}, \boldsymbol{V} \rangle_{\boldsymbol{\Sigma}} := \frac{1}{2}\text{Tr}(\mathcal{L}_{\boldsymbol{\Sigma}}[\boldsymbol{U}]\boldsymbol{V}), \quad \boldsymbol{U}, \boldsymbol{V} \in \mathcal{T}_{\boldsymbol{\Sigma}}\mathbb{S}_{++}^d
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.4</span><span class="math-callout__name">(Gaussian Schroedinger Bridge Problem as Action Minimization)</span></p>

The solution to the Gaussian SB Problem where $\mathbb{Q}$ is defined as pure Brownian motion $\mathbb{Q} : d\boldsymbol{X}\_t = \sigma\_t d\boldsymbol{B}\_t$ is equivalent to the solution of the **action minimization** problem on the Bures--Wasserstein manifold, defined as:

$$
\inf_{(\boldsymbol{\Sigma}_t)_{t \in [0,T]}} \int_0^T \left[\frac{1}{2}\lVert \dot{\boldsymbol{\Sigma}}_t \rVert_{\dot{\boldsymbol{\Sigma}}_t}^2 + \mathcal{U}_\sigma(\boldsymbol{\Sigma}_t)\right] dt \quad \text{s.t.} \quad \begin{cases} \boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma} \\ \boldsymbol{\Sigma}_T = \boldsymbol{\Sigma}' \end{cases}
$$

where $\mathcal{U}\_\sigma(\boldsymbol{\Sigma}\_t) := \frac{\sigma\_t^4}{8}\text{Tr}(\boldsymbol{\Sigma}\_t^{-1})$ is the potential energy that captures the entropic contribution of the diffusion term and $\lVert \dot{\boldsymbol{\Sigma}}\_t \rVert\_{\dot{\boldsymbol{\Sigma}}\_t}^2 = \langle \dot{\boldsymbol{\Sigma}}\_t, \dot{\boldsymbol{\Sigma}}\_t \rangle\_{\boldsymbol{\Sigma}}$ is the Riemannian metric. Furthermore, the solution satisfies the **Euler--Lagrange equation** in Bures--Wasserstein geometry:

$$
\begin{cases} \nabla_{\boldsymbol{\Sigma}_t}\dot{\boldsymbol{\Sigma}}_t = -\text{grad}\!\left(-\frac{\sigma_t^4}{8}\text{Tr}\boldsymbol{\Sigma}_t^{-1}\right) \\ \boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}, \quad \boldsymbol{\Sigma}_T = \boldsymbol{\Sigma}' \end{cases}
$$

where $\nabla\_{\boldsymbol{\Sigma}\_t}\dot{\boldsymbol{\Sigma}}\_t$ is the acceleration defined by the Riemannian gradient in the Bures--Wasserstein geometry.

</div>

This result shows that the infinite-dimensional Gaussian SB Problem reduces to a finite-dimensional action functional defined entirely on the trajectory of covariance matrices. The first term measures the **kinetic energy of the covariance transport in the Bures--Wasserstein geometry**, and the second term acts as a potential induced by the diffusion term. The Schroedinger bridge corresponds to the curve that minimizes this action while matching the endpoint covariances $\boldsymbol{\Sigma}\_0 = \boldsymbol{\Sigma}$ and $\boldsymbol{\Sigma}\_T = \boldsymbol{\Sigma}'$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.5</span><span class="math-callout__name">(Closed-Form Solution to Gaussian SB Problem)</span></p>

The solution to the Gaussian SB problem $\mathbb{P}^\star$ with linear reference measure $\mathbb{Q}$ defined with the SDE $d\boldsymbol{X}\_t^{\mathbb{Q}} = (c\_t\boldsymbol{X}\_t + \boldsymbol{\alpha}\_t) dt + \sigma\_t d\boldsymbol{B}\_t$ is itself a Gaussian Markov stochastic process $\boldsymbol{X}\_{0:T}$ where the intermediate marginals are Gaussians $p\_t = \mathcal{N}(\boldsymbol{\mu}\_t, \boldsymbol{\Sigma}\_t)$, with mean and covariance defined by:

$$
\begin{cases} \boldsymbol{\mu}_t^\star = \bar{r}_t\boldsymbol{\mu}_0 + r_t\boldsymbol{\mu}_T + \boldsymbol{\zeta}(t) - r_t\boldsymbol{\zeta}(T) \\[4pt] \boldsymbol{\Sigma}_t^\star = \bar{r}_t^2\boldsymbol{\Sigma}_0 + r_t^2\boldsymbol{\Sigma}_T + r_t\bar{r}_t(\boldsymbol{C}_{\sigma_\star} + \boldsymbol{C}_{\sigma_\star}^\top) + \kappa(t, t)(1 - \rho_t)I \end{cases}
$$

$$
\text{s.t.} \quad \begin{cases} r_t := \frac{\kappa(t, T)}{\kappa(T, T)}, \quad \bar{r}_t := \tau_t - r_t\tau_T, \quad \sigma_\star := \sqrt{\tau_T^{-1}\kappa(T, T)} \\[4pt] \boldsymbol{\zeta}(t) := \tau_t\int_0^t \tau_s^{-1}\boldsymbol{\alpha}_s ds, \quad \rho_t := \frac{\int_0^t \tau_s^{-2}\sigma_s^2 ds}{\int_0^T \tau_s^{-2}\sigma_s^2 ds} \end{cases}
$$

The time evolution of $\boldsymbol{X}\_t$ follows a closed-form SDE:

$$
d\boldsymbol{X}_t = \boldsymbol{f}_{\mathcal{N}}(\boldsymbol{X}_t, t) dt + \sigma_t d\boldsymbol{B}_t
$$

$$
\text{s.t.} \quad \begin{cases} \boldsymbol{f}_{\mathcal{N}}(\boldsymbol{x}, t) := \boldsymbol{S}_t^\top\boldsymbol{\Sigma}_t^{-1}(\boldsymbol{x} - \boldsymbol{\mu}_t) + \dot{\boldsymbol{\mu}}_t \\[4pt] \boldsymbol{P}_t := \dot{r}_t(r_t\boldsymbol{\Sigma}_T + \bar{r}_t\boldsymbol{C}_{\sigma_\star}) \\[4pt] \boldsymbol{Q}_t := -\dot{\bar{r}}_t(\bar{r}_t\boldsymbol{\Sigma}_0 + r_t\boldsymbol{C}_{\sigma_\star}) \\[4pt] \boldsymbol{S}_t := \boldsymbol{P}_t - \boldsymbol{Q}_t^\top + (c_t\kappa(t, t)(1 - \rho_t) - \sigma_t^2\rho_t)I \end{cases}
$$

where the matrix $\boldsymbol{S}\_t^\top\boldsymbol{\Sigma}\_t^{-1}$ is symmetric.

</div>

The Gaussian SB drift $\boldsymbol{f}\_{\mathcal{N}}(\boldsymbol{x}, t) = \boldsymbol{S}\_t^\top\boldsymbol{\Sigma}\_t^{-1}(\boldsymbol{x} - \boldsymbol{\mu}\_t) + \dot{\boldsymbol{\mu}}\_t$ can be decomposed into the **drift of the mean** $\dot{\boldsymbol{\mu}}\_t$ and a **shape-correcting term** $\boldsymbol{S}\_t^\top\boldsymbol{\Sigma}\_t^{-1}(\boldsymbol{x} - \boldsymbol{\mu}\_t)$ that consists of the deviation from the mean $(\boldsymbol{x} - \boldsymbol{\mu}\_t)$, how expensive it is under the covariance $\boldsymbol{\Sigma}\_t^{-1}$, and how strongly to correct it $\boldsymbol{S}\_t^\top$.

This result establishes that when the reference dynamics are linear--Gaussian, the Gaussian SB Problem admits an **exact closed-form solution that remains within the class of Gaussian Markov processes**, which is fully characterized by finite-dimensional evolutions of the mean and covariance and by an affine drift field that admits a gradient-field structure. This reveals the **core insight** that entropy-regularized transport between Gaussian marginals preserves the Gaussian structure of the distribution and induces a **potential-driven flow** that conserves distributional structure.

### 5.2 Generalized Schroedinger Bridge Problem

Up to this point, we have considered particles as acting independently via an optimal path along the Schroedinger bridge such that the total distribution over many particles matches the endpoint marginals. However, in many settings, particles evolve via stochastic trajectories that depend not only on their individual state and control but also **interactions with the population distribution** $p\_t$. Solving for the optimal evolution of particles is referred to as solving a **Mean-Field Game**, since the particles are influenced by the average dynamics of the population.

At equilibrium, each particle evolves via optimal control given the density of the particles $p\_t$, and the population $p\_t$ is generated from these optimally controlled particles. To determine the optimal dynamics, we leverage the definition of the **value function** $\psi\_t(\boldsymbol{x}) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ which defines the minimum cost of transporting $\boldsymbol{x}$ at time $t$. The gradient $\nabla\psi\_t(\boldsymbol{x})$ defines a potential field that adjusts the drift of the reference process so that the evolving distribution satisfies the endpoint constraints.

The value function $\psi\_t$ and marginal density $p\_t$ are coupled via a **Hamiltonian function** $\mathcal{H}(\boldsymbol{x}, \nabla\psi\_t, p\_t) : \mathbb{R}^d \times \mathbb{R} \times \mathcal{P}(\mathbb{R}^d) \to \mathbb{R}$, which describes the dynamics of the interacting particles, and an **interaction function** $\mathcal{I}(\boldsymbol{x}, p\_t) : \mathbb{R}^d \times \mathcal{P}(\mathbb{R}^d) \to \mathbb{R}$ that can be defined depending on the task. Given these functions, the pair of optimal value function and optimal state PDF $(\psi\_t, p\_t^\star)$ solves the following pair of PDEs:

$$
\begin{cases} \partial_t\psi_t + \mathcal{H}(\boldsymbol{x}, \nabla\psi_t, p_t^\star) + \frac{\sigma_t^2}{2}\Delta\psi_t = \mathcal{I}(\boldsymbol{x}, p_t^\star) & \psi_T(\boldsymbol{x}) = \Phi(\boldsymbol{x}) \\[4pt] \partial_t p_t^\star + \nabla \cdot (p_t^\star\nabla_{\nabla\psi_t}\mathcal{H}(\boldsymbol{x}, \nabla\psi_t, p_t^\star)) - \frac{\sigma_t^2}{2}\Delta p_t^\star = 0 & p_0 = \pi_0, \; p_T = \pi_T \end{cases}
$$

where $\Phi(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$ is a terminal cost. Given a solution pair $(\psi\_t, p\_t^\star)$, the dynamics of each particle evolve via the SDE defined by:

$$
d\boldsymbol{X}_t = -\nabla_{\nabla\psi_t}\mathcal{H}(\boldsymbol{X}_t, \nabla\psi_t(\boldsymbol{X}_t), p_t) dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0 \sim \pi_0
$$

which satisfies the marginals defined by $p\_t$ as the number of particles goes to infinity.

The coupled PDEs in the generalized SB problem resemble the HJB-FP System (Proposition 2.23) with the addition of the Hamiltonian $\mathcal{H}$ and interaction function $\mathcal{I}$. When $\mathcal{H}(\boldsymbol{x}, \nabla\psi\_t, p\_t) = \frac{1}{2}\lVert \nabla\psi\_t \rVert^2 + \langle \nabla\psi\_t, \boldsymbol{f} \rangle$ and $\mathcal{I} \equiv 0$, the system reduces to the standard dynamic SB problem. We can leverage this connection to define the **Generalized Schroedinger Bridge Problem**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.6</span><span class="math-callout__name">(Generalized Schroedinger Bridge Problem)</span></p>

Given an interaction cost $\mathcal{I}(\boldsymbol{x}, p\_t) : \mathbb{R}^d \times \mathcal{P}(\mathbb{R}^d) \to \mathbb{R}$, reference drift $\boldsymbol{f}(\boldsymbol{x}, t)$, diffusion coefficient $\sigma\_t$, and terminal marginal constraints $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$, the **generalized SB problem** can be written as:

$$
\inf_{\boldsymbol{u}} \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^u} \left[\int_0^T \left(\frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_t, t) \rVert^2 + \mathcal{I}(\boldsymbol{X}_t, p_t, t)\right) dt\right]
$$

$$
\text{s.t.} \quad \begin{cases} d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t)) dt + \sigma_t d\boldsymbol{B}_t \\ \boldsymbol{X}_0 \sim \pi_0, \quad \boldsymbol{X}_T \sim \pi_T \end{cases}
$$

which can also be written as the density-space objective:

$$
\inf_{(\boldsymbol{u}, p_t)} \left[\int_0^T \int_{\mathbb{R}^d} \left(\frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{x}, t) \rVert^2 + \mathcal{I}(\boldsymbol{x}, p_t, t)\right) p_t(\boldsymbol{x}) d\boldsymbol{x} dt\right]
$$

$$
\text{s.t.} \quad \begin{cases} \partial_t p_t(\boldsymbol{x}) = -\nabla \cdot \left(p_t(\boldsymbol{x})(\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t \boldsymbol{u}(\boldsymbol{x}, t))\right) + \frac{\sigma_t^2}{2}\Delta p_t \\ p_0 = \pi_0, \quad p_T = \pi_T \end{cases}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.7</span><span class="math-callout__name">(Optimality Conditions of Generalized Schroedinger Bridge Problem)</span></p>

Defining the Hamiltonian as:

$$
\mathcal{H}(\boldsymbol{x}, \nabla\psi_t, p_t) := \frac{1}{2}\lVert \sigma_t \nabla\psi_t \rVert^2 - \nabla\psi_t^\top \boldsymbol{f}(\boldsymbol{x}, p_t, t)
$$

We can write the optimality conditions $(\psi\_t, p\_t^\star)$ as the pair of coupled non-linear HJB-FP system with the **interaction term** $\mathcal{I}(\boldsymbol{x}, p\_t^\star) : \mathbb{R}^d \times \mathcal{P}(\mathbb{R}^d) \to \mathbb{R}$ given by:

$$
\begin{cases} \partial_t\psi_t + \frac{\sigma_t^2}{2}\lVert \nabla\psi_t \rVert^2 + \langle \nabla\psi_t, \boldsymbol{f} \rangle + \frac{\sigma_t^2}{2}\Delta\psi_t = \mathcal{I}(\boldsymbol{x}, p_t^\star) \\[4pt] \partial_t p_t^\star + \nabla \cdot (p_t^\star(\boldsymbol{f} + \sigma_t^2 \nabla\psi_t)) - \frac{\sigma_t^2}{2}\Delta p_t^\star = 0 \end{cases} \quad \text{s.t.} \quad \begin{cases} p_0 = \pi_0 \\ p_T = \pi_T \end{cases}
$$

To transform the system of non-linear PDEs to linear PDEs, we can apply the **Hopf-Cole transform**:

$$
\psi_t(\boldsymbol{x}) = \log\varphi_t(\boldsymbol{x}), \quad p_t^\star(\boldsymbol{x}) = \varphi_t(\boldsymbol{x})\hat{\varphi}_t(\boldsymbol{x})
$$

which satisfy the pair of **linear PDEs**:

$$
\begin{cases} \partial_t\varphi_t = -\langle \nabla\varphi_t, \boldsymbol{f} \rangle - \frac{\sigma_t^2}{2}\Delta\varphi_t + \mathcal{I}\varphi_t \\[4pt] \partial_t\hat{\varphi}_t = -\nabla \cdot (\hat{\varphi}_t \boldsymbol{f}) + \frac{\sigma_t^2}{2}\Delta\hat{\varphi}_t - \mathcal{I}\hat{\varphi}_t \end{cases} \quad \text{s.t.} \quad \begin{cases} \pi_0 = \varphi_0 \hat{\varphi}_0 \\ \pi_T = \varphi_T \hat{\varphi}_T \end{cases}
$$

which differs from the standard Hopf-Cole linear PDEs only by the interaction term $\mathcal{I}(\boldsymbol{x}, p\_t^\star)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.8</span><span class="math-callout__name">(Generalized SB Forward-Backward SDEs)</span></p>

Given the Schroedinger potentials $(\varphi\_t, \hat{\varphi}\_t)$ that solve the Hopf-Cole linear PDEs and a stochastic process $\boldsymbol{X}\_{0:T}$ that satisfies the forward-time SDE, we define additional stochastic processes as:

$$
\boldsymbol{Y}_t = \boldsymbol{Y}(\boldsymbol{X}_t, t) := \log\varphi_t(\boldsymbol{X}_t), \qquad \boldsymbol{Z}_t = \boldsymbol{Z}(\boldsymbol{X}_t, t) := \sigma_t \nabla\log\varphi_t(\boldsymbol{X}_t)
$$

$$
\widehat{\boldsymbol{Y}}_t = \widehat{\boldsymbol{Y}}(\boldsymbol{X}_t, t) := \log\hat{\varphi}_t(\boldsymbol{X}_t), \qquad \widehat{\boldsymbol{Z}}_t = \widehat{\boldsymbol{Z}}(\boldsymbol{X}_t, t) := \sigma_t \nabla\log\hat{\varphi}_t(\boldsymbol{X}_t)
$$

Then, the forward time evolution $t \in [0, T]$ of $\boldsymbol{X}\_{0:T}$, $(\boldsymbol{Y}\_t)\_{t \in [0,T]}$ and $(\widehat{\boldsymbol{Y}}\_t)\_{t \in [0,T]}$ are characterized by the FBSDEs:

$$
\begin{cases} d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, p_t^\star) + \sigma_t \boldsymbol{Z}_t) dt + \sigma_t d\boldsymbol{B}_t \\[4pt] d\boldsymbol{Y}_t = \left(\frac{1}{2}\lVert \boldsymbol{Z}_t \rVert^2 + \mathcal{I}(\boldsymbol{X}_t, p_t^\star)\right) dt + \boldsymbol{Z}_t^\top d\boldsymbol{B}_t \\[4pt] d\widehat{\boldsymbol{Y}}_t = \left(\frac{1}{2}\lVert \widehat{\boldsymbol{Z}}_t \rVert^2 + \nabla \cdot (\sigma_t \widehat{\boldsymbol{Z}}_t - \boldsymbol{f}(\boldsymbol{X}_t, p_t^\star)) - \widehat{\boldsymbol{Z}}_t^\top \boldsymbol{Z}_t - \mathcal{I}(\boldsymbol{X}_t, p_t^\star)\right) dt + \widehat{\boldsymbol{Z}}_t^\top d\boldsymbol{B}_t \end{cases}
$$

The time-reversed SDE $(\bar{\boldsymbol{X}}\_s)\_{s \in [0,T]}$ and corresponding FBSDEs $(\boldsymbol{Y}\_s)\_{s \in [0,T]}$, $(\widehat{\boldsymbol{Y}}\_s)\_{s \in [0,T]}$ on the coordinate $s := T - t$ are:

$$
\begin{cases} d\bar{\boldsymbol{X}}_s = (-\boldsymbol{f}(\bar{\boldsymbol{X}}_s, p_s^\star) + \sigma_s \widehat{\boldsymbol{Z}}_s) ds + \sigma_t d\boldsymbol{B}_s \\[4pt] d\boldsymbol{Y}_s = \left(\frac{1}{2}\lVert \boldsymbol{Z}_s \rVert^2 + \nabla \cdot (\sigma_t \boldsymbol{Z}_s + \boldsymbol{f}(\bar{\boldsymbol{X}}_s, p_s^\star)) - \boldsymbol{Z}_s^\top \widehat{\boldsymbol{Z}}_s - \mathcal{I}(\bar{\boldsymbol{X}}_s, p_s^\star)\right) dt + \boldsymbol{Z}_s^\top d\boldsymbol{B}_s \\[4pt] d\widehat{\boldsymbol{Y}}_s = \left(\frac{1}{2}\lVert \widehat{\boldsymbol{Z}}_s \rVert^2 + \mathcal{I}(\boldsymbol{X}_s, p_s^\star)\right) ds + \widehat{\boldsymbol{Z}}_s^\top d\boldsymbol{B}_s \end{cases}
$$

Given the SB optimality condition $p\_t^\star = \varphi\_t \hat{\varphi}\_t$, we define the interaction term $\mathcal{I} : \mathbb{R}^d \times \mathcal{P}(\mathbb{R}^d) \to \mathbb{R}$ as:

$$
\mathcal{I}(\boldsymbol{X}_t, p_t^\star) = \mathcal{I}(\boldsymbol{X}_t, \varphi_t \hat{\varphi}_t), \quad \boldsymbol{f}(\boldsymbol{X}_t, p_t^\star) = \boldsymbol{f}(\boldsymbol{X}_t, \varphi_t \hat{\varphi}_t)
$$

</div>

Compared to the dynamic SB FBSDEs derived in Section 4.3, the only structural modification appears through the interaction term $\mathcal{I}(\boldsymbol{x}, p\_t^\star)$, which introduces a mean-field dependence into both the forward and backward SDEs. The forward--backward structure remains similar to the dynamic SB FB-SDEs, but the system now encodes collective effects through the optimal marginal flow defined as $p\_t^\star = \varphi\_t \hat{\varphi}\_t$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.9</span><span class="math-callout__name">(Generalization of Dynamic SB)</span></p>

When the interaction term vanishes for all $(\boldsymbol{x}, t) \in \mathbb{R}^d \times [0, T]$, i.e., $\mathcal{I} \equiv 0$, and the reference drift is independent of the marginal, i.e., $\boldsymbol{f}(\boldsymbol{x}, p\_t, t) = \boldsymbol{f}(\boldsymbol{x}, t)$, the **generalized SB problem** reduces to the standard Dynamic SB Problem and can be interpreted as a generalization of dynamic SB to McKean-Vlasov settings with mean-field interactions.

</div>

In summary, the generalized Schroedinger bridge extends the classical dynamic SB formulation by introducing an interaction term $\mathcal{I}(\boldsymbol{x}, p\_t)$ that allows the dynamics to depend on the evolving marginal distribution $p\_t$. Crucially, the extension of the dynamic SB problem to settings with mean-field interactions *preserves the fundamental structure of the dynamic SB problem* with the only difference being an additive interaction term. Therefore, many frameworks used to solve the dynamic SB problem can be adapted to solve the generalized SB problem.

### 5.3 Multi-Marginal Schroedinger Bridge Problem

The standard dynamic SB problem aims to determine the optimal bridge that maps particles from the initial distribution $\pi\_0$ to the terminal distribution $\pi\_T$ while minimizing the KL divergence to the reference measure. A natural extension of this problem is to consider **multiple marginal constraints** at multiple points along the time horizon, which can be applied to construct feasible trajectories between observed snapshots over coarse time intervals. This variation of the SB problem is considered the **Multi-Marginal Schroedinger Bridge Problem**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.10</span><span class="math-callout__name">(Standard Multi-Marginal Schroedinger Bridge Problem)</span></p>

Given an uncontrolled reference measure $\mathbb{Q}$, multiple marginal constraints $\lbrace \pi\_{t\_k} \in \mathcal{P}(\mathbb{R}^d) \rbrace\_{k=1}^K$ at sequential time points $0 = t\_0 < \cdots < t\_k < \cdots < t\_K = T$, the standard **multi-marginal Schroedinger bridge problem** aims to determine a control $\boldsymbol{u}(\boldsymbol{x}, t)$ that minimizes:

$$
\inf_{\boldsymbol{u}} \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^u} \left[\int_0^T \frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_t, t) \rVert^2 dt\right]
$$

$$
\text{s.t.} \quad \begin{cases} d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t)) dt + \sigma_t d\boldsymbol{B}_t \\ \boldsymbol{X}_{t_k} \sim \pi_{t_k}, \; \forall k \in \lbrace 0, \ldots, K \rbrace \end{cases}
$$

where the optimal $\boldsymbol{u}^\star$ generates the path measure $\mathbb{P}^\star$ of minimal relative entropy with respect to $\mathbb{Q}$ among all controlled measures matching the prescribed marginals. This can equivalently be written as the density-space objective:

$$
\inf_{\boldsymbol{u}} \left[\int_0^T \int_{\mathbb{R}^d} \frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{x}, t) \rVert^2 p_t(\boldsymbol{x}) d\boldsymbol{x} dt\right]
$$

$$
\text{s.t.} \quad \begin{cases} \partial_t p_t(\boldsymbol{x}) = -\nabla \cdot \left(p_t(\boldsymbol{x})(\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t \boldsymbol{u}(\boldsymbol{x}, t))\right) + \frac{\sigma_t^2}{2}\Delta p_t \\ p_{t_k} = \pi_{t_k}, \; \forall k \in \lbrace 0, \ldots, K \rbrace \end{cases}
$$

</div>

Since intermediate marginal state distributions can have associated velocity information that determines how it propagates to the subsequent distribution such that $\pi\_{t\_k}(\boldsymbol{x}, \boldsymbol{v}) \in \mathcal{P}(\mathbb{R}^{2d})$, we re-formulate the multi-marginal SB problem in **phase space**, known as the **Momentum Multi-Marginal Schroedinger Bridge** problem, where the state is given by a vector $(\boldsymbol{x}, \boldsymbol{v}) \in \mathbb{R}^{2d}$. In this setting, the reference dynamics follow a pair of second-order SDEs:

$$
\begin{cases} d\boldsymbol{x}_t = \boldsymbol{v}_t dt \\ d\boldsymbol{v}_t = \boldsymbol{a}_t dt + \sigma_t d\boldsymbol{B}_t \end{cases}
$$

where stochasticity is introduced into the second-order SDE. We define the marginal distribution generated by the SDEs over the phase space as $p\_t \in \mathcal{P}(\mathbb{R}^{2d})$, the distribution of the state as $\mu(\boldsymbol{x}, t) = \int\_{\mathbb{R}^d} p\_t(\boldsymbol{x}, \boldsymbol{v}) d\boldsymbol{v}$, and the distribution over velocity as $\xi(\boldsymbol{v}, t) = \int\_{\mathbb{R}^d} p\_t(\boldsymbol{x}, \boldsymbol{v}) d\boldsymbol{x}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.11</span><span class="math-callout__name">(Momentum Multi-Marginal Schroedinger Bridge Problem)</span></p>

Given multiple marginal distributions over the phase space $\pi\_{t\_0}, \ldots, \pi\_{t\_K} \in \mathcal{P}(\mathbb{R}^{2d})$ defined at sequential time points $t\_0, \ldots, t\_K \in [0, T]$ on the time horizon, the **multi-marginal SB problem** aims to determine the optimal acceleration $\boldsymbol{a}\_t^\star \equiv \boldsymbol{a}\_t^\star(\boldsymbol{x}, \boldsymbol{v}, t) \in \mathbb{R}^{2d}$ that solves the SOC problem:

$$
\boldsymbol{a}_t^\star = \arg\min_{\boldsymbol{a}_t} \mathbb{E}_{p_t} \left[\int_0^T \frac{1}{2}\lVert \boldsymbol{a}_t \rVert^2 dt\right] \quad \text{s.t.} \quad \begin{cases} d\boldsymbol{x}_t = \boldsymbol{v}_t dt \\ d\boldsymbol{v}_t = \boldsymbol{a}_t dt + \sigma_t d\boldsymbol{B}_t \\ (\boldsymbol{x}_{t_k}, \boldsymbol{v}_{t_k}) \sim \pi_{t_k}, \; \forall k \in \lbrace 0, \ldots, K \rbrace \end{cases}
$$

Equivalently, writing the problem in terms of the distribution $p\_t$ and the Fokker-Planck equation:

$$
\boldsymbol{a}_t^\star = \arg\min_{\boldsymbol{a}_t} \mathbb{E}_{p_t} \left[\int_0^T \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} \frac{1}{2}\lVert \boldsymbol{a}_t \rVert^2 p_t d\boldsymbol{x} d\boldsymbol{v} dt\right]
$$

$$
\text{s.t.} \quad \begin{cases} \partial_t p_t(\boldsymbol{x}, \boldsymbol{v}) + \boldsymbol{v} \cdot \nabla p_t + \nabla_{\boldsymbol{v}} \cdot (\boldsymbol{a}_t p_t) - \frac{\sigma_t^2}{2}\Delta_v p_t = 0 \\ \mu(\boldsymbol{x}, t_k) = \int_{\mathbb{R}^d} \pi_{t_k}(\boldsymbol{x}, \boldsymbol{v}) d\boldsymbol{v}, \; \forall k \in \lbrace 0, \ldots, K \rbrace \end{cases}
$$

</div>

The diffusion term in the FP equation makes the dynamics *irreversible* since diffusion increases entropy over a single time direction. To absorb the diffusion term into a deterministic drift, we can define $\hat{\boldsymbol{a}}\_t := \boldsymbol{a}\_t - \frac{\sigma\_t^2}{2}\nabla\_{\boldsymbol{v}}\log p\_t$ to get the **deterministic continuity equation**:

$$
\partial_t p_t(\boldsymbol{x}, \boldsymbol{v}) + \boldsymbol{v} \cdot \nabla p_t + \nabla_{\boldsymbol{v}} \cdot (\hat{\boldsymbol{a}}_t p_t) = 0
$$

Substituting $\boldsymbol{a}\_t = \hat{\boldsymbol{a}}\_t + \frac{\sigma\_t^2}{2}\nabla\_{\boldsymbol{v}}\log p\_t$ in the squared cost, integrating the expanded form over time and phase space, and using integration by parts on the cross term, we obtain the decomposition of the objective into three components:

$$
\underbrace{\int_0^T \int \frac{1}{2}\lVert \hat{\boldsymbol{a}} \rVert^2 p_t d\boldsymbol{x} d\boldsymbol{v} dt}_{\text{transport cost}} + \underbrace{\int_0^T \int \frac{\sigma_t^2}{8}\lVert \nabla_{\boldsymbol{v}}\log p_t \rVert^2 p_t d\boldsymbol{x} d\boldsymbol{v} dt}_{\text{velocity Fisher information}} + \underbrace{\int_0^T \int \frac{1}{2}\langle \hat{\boldsymbol{a}}, \nabla_{\boldsymbol{v}}\log p_t \rangle p_t d\boldsymbol{x} d\boldsymbol{v} dt}_{\text{cross term}}
$$

The first term is the **transport cost**. The second term is the velocity **Fisher information** (squared norm of the score function), which measures the sensitivity of the distribution under infinitesimal changes to the velocity. Since the diffusion only acts on the velocity coordinate, this term acts as an uncertainty regularization after absorbing the diffusion into the deterministic drift. The cross term evaluates to a constant under the SBM marginal constraints $p\_0 = \pi\_0$ and $p\_T = \pi\_T$:

$$
\int_0^T \int \frac{1}{2}\langle \hat{\boldsymbol{a}}, \nabla_{\boldsymbol{v}}\log p_t \rangle p_t d\boldsymbol{x} d\boldsymbol{v} dt = \frac{1}{2}\int (p_T \log p_T - p_0 \log p_0) \, d\boldsymbol{v} d\boldsymbol{x}
$$

so we can drop it and write the multi-marginal SB problem as:

$$
\boldsymbol{a}_t^\star = \arg\min_{\boldsymbol{a}_t} \int_0^T \int \left[\frac{1}{2}\lVert \boldsymbol{a}_t \rVert^2 + \frac{\sigma_t^2}{8}\lVert \nabla_{\boldsymbol{v}}\log p_t \rVert^2\right] p_t \, d\boldsymbol{x} d\boldsymbol{v} dt
$$

$$
\text{s.t.} \quad \begin{cases} \partial_t p_t(\boldsymbol{x}, \boldsymbol{v}) + \boldsymbol{v} \cdot \nabla p_t + \nabla_{\boldsymbol{v}} \cdot (\hat{\boldsymbol{a}}_t p_t) = 0 \\ \mu(\boldsymbol{x}, t_k) = \int_{\mathbb{R}^d} \pi_{t_k}(\boldsymbol{x}, \boldsymbol{v}) d\boldsymbol{v}, \quad \forall k \in \lbrace 0, \ldots, K \rbrace \end{cases}
$$

This derivation shows that the multi-marginal Schroedinger bridge problem in phase space can be reformulated as an **entropy-regularized dynamic optimal transport** problem over joint position--velocity distributions. The quadratic acceleration cost $\frac{1}{2}\lVert \boldsymbol{a}\_t \rVert^2$ governs how trajectories bend in phase space, penalizing deviations from inertial motion and enforcing smooth transitions that interpolate through the prescribed intermediate marginals $\lbrace \pi\_{t\_k} \rbrace\_{k=1}^K$. In addition, the Fisher information term $\frac{\sigma\_t^2}{8}\lVert \nabla\_{\boldsymbol{v}}\log p\_t \rVert^2$ acts specifically in the velocity coordinate, regularizing the uncertainty and dispersion of velocities induced by stochasticity.

Just like how the static and dynamic Schroedinger bridge problems are an entropically regularized analogue of static and dynamic optimal transport, the multi-marginal SB problem can be interpreted as the stochastic, entropically regularized analogue of **measure-valued splines**. This perspective will help us understand the behavior of multi-marginal SB in the zero-noise limit.

#### Connection to Measure-Valued Splines

The **variational spline problem** aims to select the smoothest curve $(\boldsymbol{x}(t))\_{t \in [0,T]}$ that interpolates between a set of sequential points $\bar{\boldsymbol{x}}\_{t\_0}, \ldots, \bar{\boldsymbol{x}}\_{t\_K} \in \mathbb{R}^d$ at times $0 = t\_0 < \cdots < t\_K = T$ by minimizing:

$$
\min_{(\boldsymbol{x}_t)_{t \in [0,T]}} \int_0^T \lVert \ddot{\boldsymbol{x}}_t \rVert^2 dt \quad \text{s.t.} \quad \boldsymbol{x}_{t_k} = \bar{\boldsymbol{x}}_{t_k}, \; \forall k \in \lbrace 0, \ldots, K \rbrace
$$

where the minimization is taken over all twice-differentiable curves satisfying the interpolation constraints. The objective penalizes squared acceleration $\lVert \ddot{\boldsymbol{x}}\_t \rVert^2$, so the resulting trajectory is the smoothest curve connecting the prescribed points in the sense of minimizing the **total bending energy**.

This principle can be generalized from deterministic trajectories to evolving probability distributions by replacing the single curve $\boldsymbol{x}(t)$ with a time-dependent probability density $p\_t(\boldsymbol{x}, \boldsymbol{v})$ defined over the phase space of particle positions and velocities $(\boldsymbol{x}, \boldsymbol{v}) \in \mathbb{R}^{2d}$. Then, the distributional analogue of the variational spline problem, known as the **measure-valued spline problem**, becomes:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Measure-Valued Spline Problem)</span></p>

$$
\inf_{\boldsymbol{a}, p_t} \left\lbrace \int_0^T \int_{\mathbb{R}^d} \int_{\mathbb{R}^d} \lVert \boldsymbol{a}(\boldsymbol{x}, \boldsymbol{v}, t) \rVert^2 p_t(\boldsymbol{x}, \boldsymbol{v}) d\boldsymbol{x} d\boldsymbol{v} dt \right\rbrace
$$

$$
\text{s.t.} \quad \begin{cases} d\boldsymbol{x}_t = \boldsymbol{v} dt, \quad d\boldsymbol{v}_t = \boldsymbol{a}_t dt \\ \partial_t p_t + \langle \boldsymbol{v}, \nabla p_t \rangle + \nabla_{\boldsymbol{v}} \cdot (\boldsymbol{a} p_t) = 0 \\ \int_{\mathbb{R}^d} p_t(\boldsymbol{x}, \boldsymbol{v}) d\boldsymbol{v} = \pi_{t_k}, \; \forall k \in \lbrace 0, \ldots, K \rbrace \end{cases}
$$

where $\partial\_t p\_t + \langle \boldsymbol{v}, \nabla p\_t \rangle + \nabla\_{\boldsymbol{v}} \cdot (\boldsymbol{a} p\_t) = 0$ is the phase-space continuity equation and $\int\_{\mathbb{R}^d} p\_t(\boldsymbol{x}, \boldsymbol{v}) d\boldsymbol{v} = \pi\_{t\_k}$ enforces the position marginals. Intuitively, this problem yields the evolution of a probability distribution that passes through prescribed marginal distributions while minimizing the average squared acceleration of the particles.

Notice that this problem is exactly the Multi-Marginal SB Problem except with zero diffusion $\sigma\_t \equiv 0$, which reduces the Fokker-Planck equation with the Laplacian term $\frac{\sigma\_t^2}{2}\Delta\_v p\_t \equiv 0$ into the classic phase-space continuity equation. To this end, the multi-marginal SB problem can be interpreted as a stochastic, entropy-regularized analogue of the measure-valued spline.

</div>

This formulation extends the dynamic Schroedinger bridge problem along two key directions: incorporating multiple intermediate marginal constraints and lifting the dynamics to phase space, where marginals can encode both position and velocity information. In the zero-noise limit, the problem recovers a deterministic measure-valued spline problem, revealing how SB generalizes smooth trajectory interpolation between distributions with stochastic uncertainty.

### 5.4 Unbalanced Schroedinger Bridge Problem

Since the Schroedinger bridge problems originate from the optimal mass transport (OMT) problem, where the probability mass across the full time horizon is conserved, and no mass is lost. In many applications, such as cell dynamics, probability mass is not necessarily conserved, and particles undergo growth and death, producing **unbalanced marginal distributions**. However, this presents the *key problem* of determining the way in which particles should transport and vanish along intermediate time points that minimizes the deviation from some reference dynamics while reconstructing the unbalanced marginals.

To account for the difference in mass between terminal marginals, we can relax the mass conservation constraint in the standard regularized OT problem by introducing a **growth rate** $g(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ into the minimization objective to get the **dynamic unbalanced optimal transport** problem.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.12</span><span class="math-callout__name">(Dynamic Unbalanced Optimal Transport Problem)</span></p>

Let $\pi\_0, \pi\_T \in \mathcal{M}\_+(\mathbb{R}^d)$ be non-negative measures that may have different total mass. The **dynamic unbalanced optimal transport problem** seeks a time-dependent density $p\_t$, transport velocity $\boldsymbol{v}(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$, and growth rate $g(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ that solve the minimization:

$$
\inf_{p_t, \boldsymbol{v}, g} \int_0^T \int_{\mathbb{R}^d} \left\lbrace \frac{1}{2}\lVert \boldsymbol{v}(\boldsymbol{x}, t) \rVert^2 + \alpha\Psi(g(\boldsymbol{x}, t)) \right\rbrace p_t(\boldsymbol{x}) d\boldsymbol{x} dt
$$

$$
\text{s.t.} \quad \begin{cases} \partial_t p_t(\boldsymbol{x}) = -\nabla \cdot (p_t(\boldsymbol{x})\boldsymbol{v}(\boldsymbol{x}, t)) + g(\boldsymbol{x}, t) p_t(\boldsymbol{x}) \\ p_0 = \pi_0, \quad p_T = \pi_T \end{cases}
$$

where $\Psi : \mathbb{R} \to \mathbb{R}\_{\ge 0}$ is a non-negative function that penalizes changes in mass and $\alpha > 0$ is a hyperparameter that controls the penalty.

</div>

Notice that the **unbalanced continuity equation** constraint $\partial\_t p\_t = -\nabla \cdot (p\_t \boldsymbol{v}) + g p\_t$ contains an additional $g p\_t$ term compared to the standard continuity equation. This term models local mass creation and destruction, where the current density $p\_t(\boldsymbol{x})$ at $\boldsymbol{x}$ grows or decays proportionally to $g(\boldsymbol{x}, t)$. Therefore, $g p\_t$ accounts for the instantaneous change in mass density due to growth or decay, allowing the total mass to vary over time.

The growth penalty $\Psi(g)$, commonly defined as the quadratic growth $\Psi(g(\boldsymbol{x}, t)) := \lvert g(\boldsymbol{x}, t) \rvert^2$, regularizes the amount of mass creation or destruction along the transport path. By penalizing large growth rates, the optimization balances mass transport and mass variation, yielding the **minimal-cost combination of transport and local mass change** required to match the terminal marginals.

Extending this formulation to SDEs of the form $d\boldsymbol{X}\_t = (\boldsymbol{f}(\boldsymbol{X}\_t, t) + \sigma\_t \boldsymbol{u}(\boldsymbol{X}\_t, t)) dt + \sigma\_t d\boldsymbol{B}\_t$ yields the **unbalanced Schroedinger bridge** problem, which seeks the **most likely stochastic process** between two unbalanced marginals.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.13</span><span class="math-callout__name">(Unbalanced Schroedinger Bridge Problem)</span></p>

Given a pair of unbalanced marginals $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$, a reference path measure $\mathbb{Q}$ with drift $\boldsymbol{f}(\boldsymbol{x}, t)$ and a function that penalizes the growth rate $\Psi : \mathbb{R} \to \mathbb{R}\_{\ge 0}$, the **unbalanced Schroedinger bridge (SB) problem** aims to determine the optimal tuple density evolution $p\_t^\star$, control drift $\boldsymbol{u}^\star$, and growth rate $g^\star$ that solve the minimization problem:

$$
\inf_{p_t, \boldsymbol{u}, g} \int_0^T \int_{\mathbb{R}^d} \left[\frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{x}, t) \rVert^2 + \alpha\Psi(g(\boldsymbol{x}, t))\right] p_t(\boldsymbol{x}) d\boldsymbol{x} dt
$$

subject to the marginal constraints $p\_0 = \pi\_0$ and $p\_T = \pi\_T$ and an **unbalanced Fokker-Planck constraint** defined as:

$$
\partial_t p_t(\boldsymbol{x}) = -\nabla \cdot (p_t(\boldsymbol{x})(\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t \boldsymbol{u}(\boldsymbol{x}, t))) + \frac{\sigma_t^2}{2}\Delta p_t(\boldsymbol{x}) + g(\boldsymbol{x}, t) p_t(\boldsymbol{x})
$$

where $g(\boldsymbol{x}, t) p\_t(\boldsymbol{x})$ relaxes the mass conservation constraint of the standard Fokker-Planck equation to allow the mass to change proportionally to the growth rate.

</div>

Since the Schroedinger bridge formulation has three optimization parameters, the density $p\_t$, the control drift $\boldsymbol{u}$, and the growth rate $g$ which are all coupled in the Unbalanced FP Equation, optimizing a stochastic flow yields a difficult optimization problem. By reparameterizing the unbalanced Fokker-Planck constraint into an unbalanced continuity equation constraint, we can derive an alternative **entropy-regularized dynamic optimal transport form** of the Unbalanced SB Problem.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.14</span><span class="math-callout__name">(Entropy-Regularized Unbalanced Dynamic Optimal Transport Problem)</span></p>

The Unbalanced SB Problem from Definition 5.13 can be written as:

$$
\inf_{(p_t, \boldsymbol{v}, g)} \int_0^T \int_{\mathbb{R}^d} \left[\frac{1}{2}\lVert \boldsymbol{v} \rVert^2 + \frac{\sigma_t^2}{8}\lVert \nabla\log p_t \rVert^2 - \frac{1}{2}\langle \nabla\log p_t, \boldsymbol{f} \rangle - \frac{1}{2}(1 + \log p_t) g + \alpha\Psi(g)\right] p_t d\boldsymbol{x} dt
$$

subject to the same marginal constraints $p\_0 = \pi\_0$ and $p\_T = \pi\_T$ and the **unbalanced continuity equation constraint** defined as:

$$
\partial_t p_t(\boldsymbol{x}) = -\nabla \cdot (p_t(\boldsymbol{x})(\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t \boldsymbol{v}(\boldsymbol{x}, t))) + g(\boldsymbol{x}, t) p_t(\boldsymbol{x})
$$

where $g(\boldsymbol{x}, t) p\_t(\boldsymbol{x})$ relaxes the mass conservation constraint of the standard Continuity Equation.

</div>

*Proof.* Starting with the Unbalanced FP Equation, we follow similar steps as in the standard Continuity Equation to derive a reparameterized form of the control drift that satisfies an unbalanced continuity equation constraint with the additional growth term $g(\boldsymbol{x}, t) p\_t(\boldsymbol{x})$. Defining $\boldsymbol{v}(\boldsymbol{x}, t) := \boldsymbol{u}(\boldsymbol{x}, t) - \frac{\sigma\_t}{2}\nabla\log p\_t(\boldsymbol{x})$ and rearranging to get the change-in-variables $\boldsymbol{u}(\boldsymbol{x}, t) = \boldsymbol{v}(\boldsymbol{x}, t) + \frac{\sigma\_t}{2}\nabla\log p\_t(\boldsymbol{x})$, we write the objective as:

$$
\inf_{(p_t, \boldsymbol{v}, g)} \int_0^T \int_{\mathbb{R}^d} \left\lbrace \frac{1}{2}\lVert \boldsymbol{v}(\boldsymbol{x}, t) \rVert^2 + \frac{\sigma_t}{2}\langle \boldsymbol{v}(\boldsymbol{x}, t), \nabla\log p_t(\boldsymbol{x}) \rangle + \frac{\sigma_t^2}{8}\lVert \nabla\log p_t(\boldsymbol{x}) \rVert^2 + \alpha\Psi(g(\boldsymbol{x}, t)) \right\rbrace p_t(\boldsymbol{x}) d\boldsymbol{x} dt
$$

The expansion of the cross term deviates from the standard derivation as it incorporates the additional growth term in the continuity equation substitution:

$$
H(p_T) - H(p_0) = \int_0^T \int_{\mathbb{R}^d} (1 + \log p_t)\partial_t p_t \, d\boldsymbol{x} dt
$$

Substituting the unbalanced continuity equation $\partial\_t p\_t = -\nabla \cdot (p\_t(\boldsymbol{f} + \sigma\_t \boldsymbol{v})) + g p\_t$ and applying integration by parts, we isolate the cross term:

$$
\int_0^T \int_{\mathbb{R}^d} \frac{\sigma_t}{2}\langle \nabla\log p_t, \boldsymbol{v} \rangle p_t d\boldsymbol{x} dt = \frac{1}{2}(H(p_T) - H(p_0)) + \int_0^T \int_{\mathbb{R}^d} \frac{1}{2}\left[-\langle \nabla\log p_t, \boldsymbol{f} \rangle - (1 + \log p_t) g\right] p_t d\boldsymbol{x} dt
$$

Substituting back yields the regularized unbalanced OT form. $\square$

Since the entropy difference is fixed given the marginal constraints $p\_0 = \pi\_0$ and $p\_T = \pi\_T$, the entropy difference term can be dropped, and we get the final form of the entropy-regularized unbalanced dynamic OT objective:

$$
\inf_{(p_t, \boldsymbol{v}, g)} \int_0^T \int_{\mathbb{R}^d} \left[\frac{1}{2}\lVert \boldsymbol{v} \rVert^2 + \frac{\sigma_t^2}{8}\lVert \nabla\log p_t \rVert^2 - \frac{1}{2}\langle \nabla\log p_t, \boldsymbol{f} \rangle - \frac{1}{2}(1 + \log p_t) g + \alpha\Psi(g)\right] p_t d\boldsymbol{x} dt
$$

subject to the marginal constraints $p\_0 = \pi\_0$, $p\_T = \pi\_T$, and the Unbalanced FP Equation, which reformulates the unbalanced SB problem as a **entropy-regularized dynamic OT problem**. $\square$

The reformulation removes the coupling between $\boldsymbol{v}$ and $\nabla\log p\_t$ in the cross interaction term $\langle \boldsymbol{v}(\boldsymbol{x}, t), \sigma\_t^2 \nabla\log p\_t(\boldsymbol{x}) \rangle$, leading to a more numerically stable and computationally tractable objective. We can also observe that since the terminal marginals are fixed, the entropy difference is constant, and the objective depends solely on the path-dependent cost. This results in a problem with three **key components**: the quadratic cost of the velocity field $\frac{1}{2}\lVert \boldsymbol{v} \rVert^2$, the Fisher information term that penalizes sharp changes in density, and the contribution of the growth rate determined by the growth penalty $\Psi(g)$ and the entropy change caused by creating or destroying mass through $(1 + \log p\_t)$.

### 5.5 Branched Schroedinger Bridge Problem

Orthogonal to the problem of matching multiple subsequent marginals along the temporal evolution of probability mass is the problem of matching a complex terminal marginal distribution $\pi\_T$ with **multiple modes** $\pi\_T = \pi\_{T,1} \oplus \cdots \oplus \pi\_{T,K}$. In practice, fitting a standard Schroedinger bridge to accurately transport density from an initial distribution $\pi\_0$ to a multi-modal distribution $\pi\_T$ suffers from several challenges.

One of these challenges is **mode collapse**, in which the probability density is concentrated in only one or a few terminal modes, thereby missing the full distribution. To overcome this, one may increase the number of particles simulated from the initial distribution to improve the likelihood of discovering each terminal mode; however, this increases computational cost and does not guarantee accurate reconstruction of all modes and *their relative density weights*. Furthermore, Schroedinger bridges not only sample from the terminal distribution but also recover the energy-minimizing temporal bridge between the distributions. For multimodal target distributions, this optimal bridge can inform us about *branching times* and *mass redistribution*; however, it remains challenging to simulate effectively with standard SB methods. This is the motivation behind the **Branched Schroedinger Bridge Problem**, which enables simulation of branching and mass redistribution over a stochastic bridge to reconstruct multi-modal target distributions.

Since branching can be interpreted as the unbalanced flow of probability mass from a primary branch to multiple diverging trajectories, it can be formalized as the *sum* of Unbalanced Schroedinger bridges, where mass is progressively depleted from the primary branch and fed into the diverging paths to multiple terminal modes.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.15</span><span class="math-callout__name">(Branched Schroedinger Bridge Problem)</span></p>

Consider an initial distribution $\pi\_0$ and a terminal distribution with $K$ distinct modes $\pi\_T := \pi\_{T,1} \oplus \cdots \oplus \pi\_{T,K}$ on some potential energy landscape defined by a state cost $c(\boldsymbol{x}, t)$. Denoting the control drift of each branch as $\lbrace \boldsymbol{u}\_k(\boldsymbol{x}, t) \rbrace\_{k=0}^K$, the growth rate of each branch as $\lbrace g\_k(\boldsymbol{x}, t) \rbrace\_{k=0}^K$, and the accumulated weight of each branch as $\lbrace w\_{t,k} \rbrace\_{k=1}^K$, the Branched SB problem seeks the optimal set of control and growth rates $\lbrace \boldsymbol{u}\_k^\star, g\_k^\star \rbrace\_{k=0}^K$ that solve the following minimization:

$$
\inf_{\lbrace \boldsymbol{u}_k, g_k \rbrace_{k=0}^K} \int_0^T \left\lbrace \mathbb{E}_{p_{t,0}} \left[\frac{1}{2}\lVert \boldsymbol{u}_0(\boldsymbol{X}_{t,0}, t) \rVert^2 + c(\boldsymbol{X}_{t,0}, t)\right] w_{t,0} + \sum_{k=1}^K \mathbb{E}_{p_{t,k}} \left[\frac{1}{2}\lVert \boldsymbol{u}_k(\boldsymbol{X}_{t,k}, t) \rVert^2 + c(\boldsymbol{X}_{t,k}, t)\right] w_{t,k} \right\rbrace dt
$$

$$
\text{s.t.} \quad \begin{cases} d\boldsymbol{X}_{t,k} = (\boldsymbol{f}(\boldsymbol{X}_{t,k}, t) + \sigma_t \boldsymbol{u}_k(\boldsymbol{X}_{t,k}, t)) dt + \sigma_t d\boldsymbol{B}_t \\ \boldsymbol{X}_0 \sim \pi_0, \quad \boldsymbol{X}_{T,k} \sim \pi_{T,k} \\ w_{0,k} = \delta_{k=0}, \quad w_{T,k} = w_{T,k}^\star \end{cases}
$$

where the weight of the primary branch is given by $w\_{t,0} = 1 + \int\_0^t g\_0(\boldsymbol{X}\_{s,0}, s) ds$ and the weights of the $K$ secondary branches is given by $w\_{t,k} = \int\_0^t g\_k(\boldsymbol{X}\_{s,k}, s) ds$.

</div>

When considering a setting where the total mass across all branches is conserved, we can add the constraint $\sum\_{k=0}^K w\_{t,k} = 1$ for all $t \in [0, T]$ which enforces that the growth rates sum to zero, i.e., $g\_0(\boldsymbol{X}\_{t,0}, t) + \sum\_{k=1}^K g\_k(\boldsymbol{X}\_{t,k}, t) = 0$. In this setting, *all* the mass that is lost from the primary branch, such that growth rate is negative $g\_0 < 0$, is redistributed to the secondary branches, such that at least one branch has positive growth $\exists k, \; g\_k > 0$. To formulate a tractable form of the Branched SB Problem, we can reframe the problem as a **conditional stochastic optimal control** (CondSOC) problem that can be tractably solved given a finite set of samples from an empirical initial distribution $\pi\_0$ and a multi-modal terminal distribution $\pi\_T$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.16</span><span class="math-callout__name">(Branched Conditional Stochastic Optimal Control)</span></p>

Define the endpoint conditioned density of each branch as $p\_{t,k}(\boldsymbol{X}\_{t,k}) := \mathbb{E}\_{\pi\_{0,T,k}}[p\_{t,k}(\boldsymbol{X}\_{t,k} \mid \boldsymbol{x}\_0, \boldsymbol{x}\_{T,k})]$, where $\pi\_{0,T,k}$ is the joint coupling between the initial distribution $\pi\_0$ and the $k$th mode of the terminal distribution $\pi\_{T,k}$. The set of optimal control drifts and growth terms $\lbrace \boldsymbol{u}\_k, g\_k \rbrace\_{k=0}^K$ that solve the Branched SB problem defined in Definition 5.15 can be obtained by minimizing the sum of Unbalanced Conditional Stochastic Optimal Control problems defined as:

$$
\inf_{\lbrace \boldsymbol{u}_k, g_k \rbrace_{k=0}^K} \mathbb{E}_{(\boldsymbol{x}_0, \boldsymbol{x}_{T,0}) \sim \pi_{0,T,0}} \int_0^T \left\lbrace \mathbb{E}_{p_{t|0,T,0}} \left[\frac{1}{2}\lVert \boldsymbol{u}_0(\boldsymbol{X}_{t,0}, t) \rVert^2 + c(\boldsymbol{X}_{t,0}, t)\right] w_{t,0} \right\rbrace dt
$$

$$
+ \sum_{k=1}^K \mathbb{E}_{(\boldsymbol{x}_0, \boldsymbol{x}_{T,k}) \sim \pi_{0,T,k}} \int_0^T \left\lbrace \mathbb{E}_{p_{t|0,T,k}} \left[\frac{1}{2}\lVert \boldsymbol{u}_k(\boldsymbol{X}_{t,k}, t) \rVert^2 + c(\boldsymbol{X}_{t,k}, t)\right] w_{t,k} \right\rbrace dt
$$

$$
\text{s.t.} \quad \begin{cases} d\boldsymbol{X}_{t,k} = (\boldsymbol{f}(\boldsymbol{X}_{t,k}, t) + \sigma_t \boldsymbol{u}_k(\boldsymbol{X}_{t,k}, t)) dt + \sigma_t d\boldsymbol{B}_t \\ \boldsymbol{X}_0 = \boldsymbol{x}_0, \quad \boldsymbol{X}_{T,k} = \boldsymbol{x}_{T,k} \\ w_{0,k} = \delta_{k=0}, \quad w_{T,k} = w_{T,k}^\star \end{cases}

$$

where the weight of the primary branch is given by $w\_{t,0} = 1 + \int\_0^t g\_0(\boldsymbol{X}\_{s,0}, s) ds$ and the weights of the $K$ secondary branches is given by $w\_{t,k} = \int\_0^t g\_k(\boldsymbol{X}\_{s,k}, s) ds$.

</div>

*Proof.* To prove that the branched CondSOC problem solves the branched SB problem in Definition 5.15, we start by defining each branch $k$ as solving its own Unbalanced SB problem:

$$
\inf_{\boldsymbol{u}_k, g_k} \int_0^T \left\lbrace \mathbb{E}_{p_{t,k}} \left[\frac{1}{2}\lVert \boldsymbol{u}_k(\boldsymbol{X}_{t,k}) \rVert^2 + c(\boldsymbol{X}_{t,k}, t)\right] \left(w_{0,k} + \int_0^t g_{s,k}(\boldsymbol{X}_{s,k}) ds\right) \right\rbrace dt
$$

$$
\text{s.t.} \quad \begin{cases} \partial_t p_{t,k} = -\nabla \cdot (p_{t,k}(\boldsymbol{f} + \sigma_t \boldsymbol{u}_k)) + \frac{\sigma_t^2}{2}\Delta p_{t,k} + g_k p_{t,k} \\ p_0 = \pi_0, \quad p_{T,k} = \pi_{T,k} \end{cases}
$$

Now, it suffices to show that the sum of unbalanced CondSOC problems satisfies the **global Fokker-Planck equation** of the density over all branches:

$$
\partial_t p_t(\boldsymbol{X}_t) = -\nabla \cdot (p_t(\boldsymbol{X}_t)(\boldsymbol{f} + \sigma_t \boldsymbol{u}_t)(\boldsymbol{X}_t, t)) + \frac{\sigma_t^2}{2}\Delta p_t(\boldsymbol{X}_t)
$$

where the probability density $p\_t$ is defined as the weighted sum of the density at each branch at time $t$ given by $p\_t(\boldsymbol{X}\_t) := \sum\_{k=0}^K w\_{t,k} p\_{t,k}(\boldsymbol{X}\_t)$. Differentiating with respect to $t$ and applying the chain rule:

$$
\partial_t p_t = \sum_{k=0}^K \left[w_{t,k} \underbrace{(\partial_t p_{t,k})}_{\text{branched FP}} + \underbrace{(\partial_t w_{t,k})}_{:= g_k(\boldsymbol{X}_t)} p_{t,k}\right]
$$

First, the divergence term is rewritten using the linearity of the divergence operator. By defining $\boldsymbol{u} := \frac{1}{p\_t}\sum\_{k=0}^K w\_{t,k} \boldsymbol{u}\_k p\_{t,k}$ as the mass-weighted average of the control drift for each branch, we get:

$$
\sum_{k=0}^K \left(-w_{t,k}\nabla \cdot ((\boldsymbol{f} + \sigma_t \boldsymbol{u}_k) p_{t,k})\right) = -\nabla \cdot ((\boldsymbol{f} + \sigma_t \boldsymbol{u}) p_t)
$$

This is theoretically grounded, as under the global context, the control drift $\boldsymbol{u}\_k(\boldsymbol{X}\_t, t)$ of a particle $\boldsymbol{X}\_t = \boldsymbol{x}$ along a single branch $k$ should be scaled by its probability of being in branch $k$, given by $p\_{t,k}(\boldsymbol{X}\_t)$, the weight of the particle itself $w\_{t,k}(\boldsymbol{X}\_t)$, normalized by the total probability of the particle over all branches $p\_t(\boldsymbol{X}\_t)$.

Next, the diffusion term is rewritten by applying the linearity of the Laplacian operator:

$$
\sum_{k=0}^K w_{t,k}\frac{\sigma_t^2}{2}\Delta p_{t,k} = \frac{\sigma_t^2}{2}\Delta\left(\sum_{k=0}^K w_{t,k} p_{t,k}\right) = \frac{\sigma_t^2}{2}\Delta p_t
$$

Finally, for the growth term, it is simply the weighted sum of the growth over each branch $\sum\_{k=0}^K g\_k p\_{t,k}$ and doesn't alter the direction or motion of the particle along the branched fields in the global context. Therefore, all three terms satisfy the global Fokker-Planck equation, and we finish the proof. $\square$

This derivation provides insight into how branched Schroedinger bridges behave under the global dynamics. We observe that even when each branch evolves via its own control and growth field, the overall system evolves as a single stochastic process whose probability density is a weighted superposition of all branches. From the perspective of the global Fokker-Planck constraint, branching does not introduce additional forces or discontinuities in the particle motion. Instead, it induces a mixture of control drifts, where the effective drift at any state is the probability-weighted average of the branch-specific controls.

This solves the challenge of mode collapse because each mode is generated with its own control drift, constrained by a terminal weight to ensure that the correct probability mass and distribution are reconstructed. Furthermore, the potential energy function $c(\boldsymbol{x}, t)$ that governs the system dynamics are minimized at the *optimal branched mass redistribution*, such that the optimal control and growth fields $\lbrace \boldsymbol{u}\_k^\star, g\_k^\star \rbrace\_{k=0}^K$ that solve the Branched SB problem in Definition 5.15 yields the branching trajectories and their relative weights such that they minimize the total energy required to reconstruct the terminal distribution.

### 5.6 Fractional Schroedinger Bridge Problem

Up to this point, we have considered only stochastic differential equations (SDEs) with standard Brownian motion (BM) processes that are Markov, where each increment of the SDE is independent of all previous steps. This formulation is a specific design choice that ensures the tractability of the bridge solution and corresponding drift. While this choice is often a sufficient approximation, it ignores the effects of long-range temporal dependencies inherent in complex real-world systems.

**Fractional Brownian motion** (fBM) is a generalization of standard BM to non-memoryless processes, where each increment is *dependent* on previous increments. This dependence is characterized by the **Hurst index** ($H$), which determines the **roughness** or **pathwise regularity** of the dynamics and the *magnitude of long-range dependencies*. A Hurst index of $H = 0.5$ recovers the standard BM.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.17</span><span class="math-callout__name">(Fractional Brownian Motion)</span></p>

Given standard $d$-dimensional Brownian motion (BM) $(\boldsymbol{B}\_t \in \mathbb{R}^d)\_{t \in [0,T]}$, **fractional BM (fBM)** is a centered Gaussian process defined by integrating over the history of the BM:

$$
\boldsymbol{B}_t^H := \frac{1}{\Gamma(H + \frac{1}{2})} \int_0^t (t - s)^{H - \frac{1}{2}} d\boldsymbol{B}_s, \quad t \ge 0
$$

where $\Gamma$ is the Gamma function and $H \in (0, 1)$ is the Hurst index.

</div>

We observe that the kernel $(t - s)^{H - \frac{1}{2}}$ injects time-dependent memory to the BM by weighting recent states ($s \approx t$) heavily and weighting distant states lightly. For $H > \frac{1}{2}$, the increments are **positively correlated**, resulting in smoother trajectories where each subsequent increment is more likely to be close to the previous increment. For $H < \frac{1}{2}$, the increments are **negatively correlated**, resulting in rougher trajectories where each subsequent increment aims to *undo itself* or revert back to the mean. When $H = \frac{1}{2}$, the kernel reduces to $(t - s)^{H - \frac{1}{2}} = 1$ which recovers the standard BM process.

Given the dependence on previous states, fBM is not Markov, resulting in an intractable drift for simulation. To overcome this, Markov approximations of fBM have been introduced, which approximate fBM by a weighted sum of **Ornstein-Uhlenbeck** (OU) processes, which are a class of Markov BM processes with a restoring force pulling it back toward the mean.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.18</span><span class="math-callout__name">(Markov Approximation of Fractional Brownian Motion)</span></p>

Given a $d$-dimensional fractional Brownian motion (fBM) $(\boldsymbol{B}\_t^H)\_{t \in [0,T]}$, we define a **Markov-approximate fBM** (MA-fBM) $(\widehat{\boldsymbol{B}}\_t^H)\_{t \in [0,T]}$ as:

$$
\widehat{\boldsymbol{B}}_t^H := \sum_{k=1}^K \omega_k \boldsymbol{Y}_t^k, \quad \omega_1, \ldots, \omega_K \in \mathbb{R}
$$

where $\boldsymbol{Y}\_t^k$ denote $K$ Ornstein-Uhlenbeck (OU) processes of the form:

$$
\boldsymbol{Y}_t^k := \int_0^t e^{-\gamma_k(t - s)} d\boldsymbol{B}_s, \quad d\boldsymbol{Y}_t^k = -\gamma_k \boldsymbol{Y}_t^k dt + d\boldsymbol{B}_t, \quad k = 1, \ldots, K
$$

where $\lbrace \gamma\_k \rbrace\_{k=1}^K$ are the mean reversion coefficients that determine how strongly the dynamics are pulled back to the mean. To approximate the fBM dynamics, $\gamma\_k := r^{k - n}$ for $r > 1$ and $n = \frac{K+1}{2}$, which produces a log-uniformly spread time grid of fast-reverting and slow-reverting OU processes.

</div>

Intuitively, each OU process contributes one exponential memory scale, where short memory modes are captured by strong mean-reversion OU processes (large $\gamma\_k$) and long memory modes are captured by weak mean-reversion OU processes (small $\gamma\_k$). To determine the **optimal finite set** of coefficients $\lbrace \omega\_k \rbrace\_{k=1}^K$ for MA-fBM, one can directly minimize the *expected squared error* between the true fBM and the OU approximation over time:

$$
(\omega_1, \ldots, \omega_K) = \arg\min_{\omega_1, \ldots, \omega_K} \left\lbrace \int_0^T \mathbb{E}_{\mathbb{P}} \left[\left(\boldsymbol{B}_t^H - \widehat{\boldsymbol{B}}_t^H\right)^2\right] dt \right\rbrace
$$

which yields a $L^2(\mathbb{P})$ optimal approximation.

Since the fBM process is determined by the scaled fBM process, which we denote as $\widehat{\boldsymbol{X}}\_t := \sqrt{\varepsilon}\,\widehat{\boldsymbol{B}}\_t^H$, and the $K$ OU processes $\boldsymbol{Y}\_t := (\boldsymbol{Y}\_t^1, \ldots, \boldsymbol{Y}\_t^K)$ that approximate $\widehat{\boldsymbol{B}}\_t$, we define the full reference process $\mathbb{Q}$ as $\boldsymbol{Z} := (\sqrt{\varepsilon}\,\widehat{\boldsymbol{B}}\_t, \boldsymbol{Y}\_t)$, which follows the SDE:

$$
\mathbb{Q}: \quad d\boldsymbol{Z}_t = \boldsymbol{F}\boldsymbol{Z}_t dt + \sigma_t d\boldsymbol{B}_t
$$

where $\boldsymbol{F} \in \mathbb{R}^{d(K+1) \times d(K+1)}$ is a block matrix encoding the linear coupling between the state $\boldsymbol{X}\_t$ and the auxiliary OU processes $\boldsymbol{Y}\_t := (\boldsymbol{Y}\_t^1, \ldots, \boldsymbol{Y}\_t^K)$ and $\sigma\_t$ is the diffusion coefficient.

However, we observe that the process $\widehat{\boldsymbol{X}}\_t$ alone is not Markov, as there is no way to predict the next state $\widehat{\boldsymbol{X}}\_{t+h}$ from $\widehat{\boldsymbol{X}}\_t$ without knowledge of the OU processes, therefore only the full reference process $\boldsymbol{Z}\_t := (\widehat{\boldsymbol{X}}\_t, \boldsymbol{Y}\_t^1, \ldots, \boldsymbol{Y}\_t^K)$ is Markov. Using $\mathbb{Q}$ as the reference Markov process, we can derive the form of the Markov fractional Brownian bridge, where we leverage Doob's $h$-transform described in Section 4.4 to condition $\boldsymbol{Z}\_t$ on the endpoint marginals.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.19</span><span class="math-callout__name">(Markov Approximation of Fractional Brownian Bridge)</span></p>

Let $\mathbb{Q}$ be the reference path measure induced by the Markov augmentation $\boldsymbol{Z}\_{t|0,T}$ of a scaled fractional Brownian motion, with reference dynamics:

$$
d\boldsymbol{Z}_{t|0,T} = \boldsymbol{F}\boldsymbol{Z}_{t|0,T} dt + \sigma_t d\boldsymbol{B}_t
$$

Fixing endpoints $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$ and $\boldsymbol{X}\_T = \boldsymbol{x}\_T$, then the corresponding bridge measure obtained by conditioning $\mathbb{Q}$ on these endpoint constraints is defined by the Doob $h$-transform of $\mathbb{Q}$ with $h(\boldsymbol{z}, t) := \mathbb{S}\_{T|t}(\boldsymbol{x}\_T \mid \boldsymbol{Z}\_{t|0,T} = \boldsymbol{z})$, and the Markov approximation of the fractional Brownian bridge satisfies the SDE:

$$
d\boldsymbol{Z}_{t|0,T} = (\boldsymbol{F}\boldsymbol{Z}_{t|0,T} + \sigma_t^2 \nabla_{\boldsymbol{z}}\log\mathbb{S}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{Z}_{t|0,T} = \boldsymbol{z})) dt + \sigma_t d\boldsymbol{B}_t
$$

</div>

*Proof.* We leverage Doob's $h$-Transform, which defines an endpoint conditioned function $h(\boldsymbol{z}, t)$. Here, we define $h : \mathbb{R}^{d(K+1)} \times [0, T] \to [0, 1]$ as the endpoint probability of a state $\boldsymbol{X}\_T = \boldsymbol{x}\_T$ under the augmented path measure $\mathbb{S}$, given by:

$$
h(\boldsymbol{z}, t) := \mathbb{S}_{T|t}(\boldsymbol{X}_T = \boldsymbol{x}_T \mid \boldsymbol{Z}_t = \boldsymbol{z}) \equiv \mathbb{S}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{z})
$$

We check that $h(\boldsymbol{z}, t)$ satisfies the Martingale Property that defines a valid tilting function. To derive an expression for $\mathbb{S}\_{t+\Delta t|t}(\tilde{\boldsymbol{z}} \mid \boldsymbol{z})$, we use Bayes rule to decompose $\mathbb{S}\_{T|t,t+\Delta t}(\boldsymbol{x}\_T \mid \boldsymbol{z}, \tilde{\boldsymbol{z}})$ and apply the Markov property $\mathbb{S}\_{T|t,t+\Delta t}(\boldsymbol{x}\_T \mid \boldsymbol{z}, \tilde{\boldsymbol{z}}) = \mathbb{S}\_{T|t+\Delta t}(\boldsymbol{x}\_T \mid \tilde{\boldsymbol{z}})$ to get:

$$
\mathbb{S}_{t+\Delta t|t}(\tilde{\boldsymbol{z}} \mid \boldsymbol{z}) = \frac{\mathbb{S}_{t+\Delta t|t,T}(\tilde{\boldsymbol{z}} \mid \boldsymbol{z}, \boldsymbol{x}_T) h(\boldsymbol{z}, t)}{h(\tilde{\boldsymbol{z}}, t + \Delta t)}
$$

Substituting into the martingale condition:

$$
h(\boldsymbol{z}, t) = \int_{\mathbb{R}^d} \mathbb{S}_{t+\Delta t|t}(\boldsymbol{Z}_{t+\Delta t} = \tilde{\boldsymbol{z}} \mid \boldsymbol{Z}_t = \boldsymbol{z}) h(\tilde{\boldsymbol{z}}, t + \Delta t) d\tilde{\boldsymbol{z}}
$$

$$
= h(\boldsymbol{z}, t) \int_{\mathbb{R}^d} \mathbb{S}_{t+\Delta t|t,T}(\tilde{\boldsymbol{z}} \mid \boldsymbol{z}, \boldsymbol{x}_T) d\tilde{\boldsymbol{z}} = h(\boldsymbol{z}, t)
$$

which confirms that $h(\boldsymbol{z}, t) := \mathbb{S}\_{T|t}(\boldsymbol{x}\_T \mid \boldsymbol{z})$ is a valid tilting function and the SDE of the fractional Brownian bridge can be written as:

$$
d\boldsymbol{Z}_{t|0,T} = (\boldsymbol{F}\boldsymbol{Z}_{t|0,T} + \sigma_t^2 \nabla_{\boldsymbol{z}}\log\mathbb{S}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{Z}_{t|0,T} = \boldsymbol{z})) dt + \sigma_t d\boldsymbol{B}_t
$$

which concludes the proof. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.20</span><span class="math-callout__name">(Gaussian Form of Fractional Brownian Bridge)</span></p>

The fractional Brownian bridge has a Gaussian transition density $\mathbb{S}\_{t+\Delta t|t}(\cdot \mid \boldsymbol{z})$. Conditioned on the terminal state $\widehat{\boldsymbol{X}}\_T = \boldsymbol{x}\_T$, the gradient of the log density takes the form:

$$
\nabla_{\boldsymbol{z}}\log\mathbb{S}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{z}) = \left[\nabla_{\boldsymbol{z}}\log\mathbb{S}_{T|t}^1(\boldsymbol{x}_T \mid \boldsymbol{z}), \ldots, \nabla_{\boldsymbol{z}}\log\mathbb{S}_{T|t}^d(\boldsymbol{x}_T \mid \boldsymbol{z})\right]
$$

where for each $i \in \lbrace 1, \ldots, d \rbrace$, we have:

$$
\nabla_{\boldsymbol{z}}\log\mathbb{S}_{T|t}^i(\boldsymbol{x}_T \mid \boldsymbol{z}) = [1, \omega_1\zeta_1(t, T), \ldots, \omega_K\zeta_K(t, T)]^\top \frac{\boldsymbol{x}_T^i - \boldsymbol{\mu}_{T|t}^i(\boldsymbol{z})}{\sigma_{T|t}^2}
$$

where the conditional mean $\boldsymbol{\mu}\_{T|t}$ and covariance $\sigma\_{T|t}^2$ are given by:

$$
\boldsymbol{\mu}_{T|t}(\boldsymbol{z}) = \boldsymbol{x} + \sum_{k=1}^K \omega_k \boldsymbol{y}_k \zeta_k(t, T), \quad \sigma_{T|t}^2(\boldsymbol{z}) = \varepsilon \sum_{k=1}^K \sum_{\ell=1}^K \frac{\omega_k \omega_\ell}{\gamma_k + \gamma_\ell}\left(1 - e^{-(1-t)(\gamma_k + \gamma_\ell)}\right)
$$

where $\boldsymbol{z} := (\boldsymbol{x}, \boldsymbol{y}\_1, \ldots, \boldsymbol{y}\_K)$ are the states of the OU processes.

</div>

*Proof.* We recall the definition of the Markov-approximated fractional BM process given by the state $\boldsymbol{X}\_t$ and the OU random variables $(\boldsymbol{Y}\_t^1, \ldots, \boldsymbol{Y}\_t^K)$ defined as:

$$
\widehat{\boldsymbol{X}}_t := \sqrt{\varepsilon}\sum_{k=1}^K \omega_k \boldsymbol{Y}_t^k, \quad d\boldsymbol{Y}_t^k = -\gamma_k \boldsymbol{Y}_t^k dt + d\boldsymbol{B}_t
$$

which gives the time evolution $d\widehat{\boldsymbol{X}}\_t = -\sqrt{\varepsilon}\sum\_{k=1}^K \omega\_k\gamma\_k \boldsymbol{Y}\_t^k dt + \sqrt{\varepsilon}\sum\_{k=1}^K \omega\_k d\boldsymbol{B}\_t$. By deriving an expression for the next state $\widehat{\boldsymbol{X}}\_{t+\Delta t}$ using the integral $\int\_0^{t+\Delta t}$, splitting it, and expanding $\boldsymbol{Y}\_t^k$ via its OU definition $\boldsymbol{Y}\_r^k = e^{-\gamma\_k(r-t)}\boldsymbol{Y}\_t^k + \int\_t^r e^{-\gamma\_k(r-s)} d\boldsymbol{B}\_s$, we obtain:

$$
\widehat{\boldsymbol{X}}_{t+\Delta t} = \widehat{\boldsymbol{X}}_t + \sum_{k=1}^K \omega_k \boldsymbol{Y}_t^k \zeta(t, t + \Delta t) + \sqrt{\varepsilon}\sum_{k=1}^K \omega_k\gamma_k \int_t^{t+\Delta t} e^{-\gamma_k(t+\Delta t - s)} d\boldsymbol{B}_s
$$

where $\zeta(t, t + \Delta t) := \sqrt{\varepsilon}(e^{-\gamma\_k \Delta t} - 1)$. Setting $\Delta t = T - t$ to get the terminal state, the conditional mean given the realization $\boldsymbol{z} = (\boldsymbol{x}, \boldsymbol{y}\_1, \ldots, \boldsymbol{y}\_K)$ is:

$$
\boldsymbol{\mu}_{T|t}(\boldsymbol{z}) := \mathbb{E}[\widehat{\boldsymbol{X}}_T \mid \boldsymbol{Z}_t = \boldsymbol{z}] = \boldsymbol{x} + \sum_{k=1}^K \omega_k \boldsymbol{y}_k \zeta_k(t, T)
$$

where the stochastic integral vanishes under the conditional expectation. The conditional variance is computed using **Itô's isometry**:

$$
\sigma_{T|t}^2 := \text{Var}(\widehat{\boldsymbol{X}}_T \mid \boldsymbol{Z}_t) = \varepsilon \sum_{k=1}^K \sum_{\ell=1}^K \omega_k\omega_\ell \text{Cov}(I_k, I_\ell)
$$

where $I\_k = \int\_t^T e^{-\gamma\_k(T-s)} d\boldsymbol{B}\_s$ and using Itô's isometry:

$$
\text{Cov}(I_k, I_\ell) = \int_t^T e^{-(\gamma_k + \gamma_\ell)(T-s)} ds = \frac{1 - e^{-(\gamma_k + \gamma_\ell)(T-t)}}{\gamma_k + \gamma_\ell}
$$

Therefore, $\widehat{\boldsymbol{X}}\_T \mid (\boldsymbol{Z}\_t = \boldsymbol{z}) \sim \mathcal{N}(\boldsymbol{\mu}\_{T|t}(\boldsymbol{z}), \sigma\_{T|t}^2)$. The gradient for the log density for the $i$th dimension of $\boldsymbol{x}\_T$ is:

$$
\nabla_{\boldsymbol{z}}\log\mathbb{S}_{T|t}^i(\boldsymbol{x}_T \mid \boldsymbol{z}) = \frac{\boldsymbol{x}_T^i - \boldsymbol{\mu}_{T|t}^i(\boldsymbol{z})}{\sigma_{T|t}^2} \nabla_{\boldsymbol{z}} \boldsymbol{\mu}_{T|t}^i(\boldsymbol{z}) = [1, \omega_1\zeta_1(t, T), \ldots, \omega_K\zeta_K(t, T)]^\top \frac{\boldsymbol{x}_T^i - \boldsymbol{\mu}_{T|t}^i(\boldsymbol{z})}{\sigma_{T|t}^2}
$$

which is exactly the form for the log gradient of the $h$-function defined in the Lemma. $\square$

Substituting this into the SDE of the fractional Brownian bridge, we get the complete form:

$$
d\boldsymbol{Z}_{t|0,T} = (\boldsymbol{F}\boldsymbol{Z}_{t|0,T} + \sigma_t^2 \boldsymbol{u}(\boldsymbol{Z}_{t|0,T}, t)) dt + \sigma_t d\boldsymbol{B}_t
$$

$$
\text{s.t.} \quad \begin{cases} \boldsymbol{u}(\boldsymbol{Z}_{t|0,T}, t) = [u_1(\boldsymbol{Z}_{t|0,T}, t), \ldots, u_d(\boldsymbol{Z}_{t|0,T}, t)] \\[4pt] u_i(\boldsymbol{Z}_{t|0,T}, t) = \nabla_{\boldsymbol{z}}\log\mathbb{S}_{T|t}^i(\boldsymbol{x}_T \mid \boldsymbol{z}) = [1, \omega_1\zeta_1(t, T), \ldots, \omega_K\zeta_K(t, T)]^\top \frac{\boldsymbol{x}_T^i - \boldsymbol{\mu}_{T|t}^i(\boldsymbol{z})}{\sigma_{T|t}^2} \end{cases}
$$

By leveraging the finite-dimensional Markov lift of the MA-fBM approximation, we have derived the explicit conditional law of $\widehat{\boldsymbol{X}}\_T$ given $\boldsymbol{Z}\_t$, which remains Gaussian with affine mean and time-dependent variance and whose $h$-function can be derived in closed form. The non-Markovian nature of the marginal process $\boldsymbol{X}\_t$ provides a principled way to incorporate long-range temporal correlations into generative stochastic dynamics through a finite-dimensional Markov approximation of MA-fBM. This framework therefore, extends classical Schroedinger bridge problems beyond memoryless dynamics and offers a mathematically tractable route to learning physically realistic long-horizon dependencies.

### 5.7 Closing Remarks for Section 5

In this section, we expanded the theory of the dynamic Schroedinger bridge problem to a broader array of constraints and problem settings. We start with the **Gaussian SB problem**, which we show admits a closed-form solution that is in the class of Gaussian Markov processes. We then introduced the **generalized SB problem**, which extends the classical formulation to systems with mean-field interactions where the dynamics of individual particles depend on the evolving population distribution.

Next, we explored several important extensions of the SB framework that arise in more complex settings. These include the **multi-marginal SB problem**, which incorporates multiple intermediate marginal constraints; the **unbalanced SB problem**, which allows for the creation or destruction of mass along the transport trajectory; and the **branched SB problem**, which captures scenarios where stochastic trajectories diverge toward multiple terminal modes. Finally, we considered an alternative class of stochastic processes driven by fractional Brownian motion, leading to the formulation of the **fractional SB problem**.

These extensions illustrate the incredible flexibility of the Schroedinger bridge framework in describing complex stochastic systems across a wide range of settings. Having introduced the theoretical foundations of both the static and dynamic formulations of the SB problem, in addition to their extensions to diverse constraints and dynamics, we are now prepared to dive into modern **generative modeling frameworks** that leverage Schroedinger bridge theory to construct scalable algorithms for high-dimensional data.

## 6. Generative Modeling with Schroedinger Bridges

In this section, we develop the connection between Schroedinger bridge theory and modern generative modeling frameworks, showing how generative modeling can be formulated as the problem of learning controlled stochastic dynamics that interpolate between an initial and a target distribution while minimizing relative entropy with respect to a reference process.

We begin with a brief primer on score-based generative modeling (Section 6.1), highlighting its formulation in terms of forward and reverse-time stochastic processes. We then extend this perspective by jointly learning forward and backward controlled drifts through likelihood maximization over coupled forward--backward SDEs (Section 6.2). As an alternative paradigm, we introduce diffusion Schroedinger bridge matching, which constructs generative models via path-space reciprocal and Markov projections with parameterized drifts (Section 6.3). Finally, we present two simulation-free approaches for learning Schroedinger bridges: score and flow matching (Section 6.4) and adjoint matching (Section 6.5).

Throughout this section, we show how Schroedinger bridges provide a unifying framework that connects likelihood-based training, path-space KL minimization, and score and flow matching frameworks into a single coherent theory.

### 6.1 A Primer on Score-Based Generative Modeling

From Section 4.2, we have shown that Schroedinger bridges are a generalization of diffusion models where the prior distribution is a simple Gaussian prior. Recall the backward SDE (Time Reversal Formula) corresponding to the forward SDE with variance-exploding drift given by:

$$
d\boldsymbol{X}_t = \boldsymbol{f}(\boldsymbol{X}_t, t) dt + \sigma_t d\boldsymbol{B}_t, \qquad \boldsymbol{X}_0 \sim \pi_0 := p_{\text{data}}
$$

$$
d\bar{\boldsymbol{X}}_s = \left[-\boldsymbol{f}(\bar{\boldsymbol{X}}_s, T - s) + \sigma_{T-s}^2 \nabla\log\tilde{p}_s(\bar{\boldsymbol{X}}_s)\right] ds + \sigma_{T-s} d\widetilde{\boldsymbol{B}}_s, \qquad \bar{\boldsymbol{X}}_0 \sim \pi_T := p_{\text{prior}}
$$

where $\tilde{p}\_s(\boldsymbol{x})$ is the density generated by the diffusion SDE at time $s$ and $\nabla\log\tilde{p}\_s(\boldsymbol{x})$ is a gradient drift that pushes the density towards areas of high likelihood given the noisy data distribution at time $s$. This process guides the diffusion process to samples from the true data distribution $\tilde{p}\_T(\boldsymbol{x}) = p\_0(\boldsymbol{x})$. The expression $\nabla\log\tilde{p}\_s(\boldsymbol{x})$ is known as the **score function**.

**Score-based generative modeling** aims to train a generative model that samples from the data distribution $\pi\_0 := p\_{\text{data}}$ by parameterizing the score function with a neural network with parameters $\theta$ known as the **score-based model** $\boldsymbol{s}\_\theta(\boldsymbol{x}, s) \approx \nabla\log\tilde{p}\_s(\boldsymbol{x})$ which estimates the score function over the state space $\boldsymbol{x} \in \mathbb{R}^d$ and time coordinate $s \in [0, T]$.

Given the score-based model $\boldsymbol{s}\_\theta(\boldsymbol{x}, s)$, we can define the estimated backward SDE as:

$$
d\bar{\boldsymbol{X}}_s = \left[-\boldsymbol{f}(\bar{\boldsymbol{X}}_s, T - s) + \sigma_{T-s}^2 \boldsymbol{s}_\theta(\bar{\boldsymbol{X}}_s, s)\right] ds + \sigma_{T-s} d\widetilde{\boldsymbol{B}}_s, \quad \bar{\boldsymbol{X}}_0 \sim p_{\text{prior}}
$$

The distribution of *clean* samples generated from simulating many samples $\bar{\boldsymbol{X}}\_0 \sim p\_{\text{prior}}$ via this SDE over time $s \in [0, T]$ approximates the data distribution $p\_{\text{data}}$.

A simple objective that is minimized exactly when the score-based model matches the true score function is the **score matching loss** $\mathcal{L}\_{\text{SM}}$ defined as the weighted squared difference integrated over time:

$$
\mathcal{L}_{\text{SM}}(\theta) := \frac{1}{2}\int_0^T \mathbb{E}_{\tilde{p}_s}\left[\lambda(t)\lVert \nabla\log\tilde{p}_s(\boldsymbol{x}) - \boldsymbol{s}_\theta(\boldsymbol{x}, s) \rVert^2\right] ds
$$

For this objective to be tractable, we require a closed-form expression for $\nabla\log\tilde{p}\_s(\boldsymbol{x})$. For generative modeling of a clean data distribution, the forward stochastic process or *noise injection process* can be defined with the conditional distribution $p\_t(\tilde{\boldsymbol{x}}\_t \mid \boldsymbol{x}\_0) := \mathcal{N}(\tilde{\boldsymbol{x}}\_t; \boldsymbol{x}\_0, \sigma\_t^2 \boldsymbol{I}\_d)$ where $\boldsymbol{x}\_0 \sim p\_{\text{data}}$, which progressively smoothes the data distribution with larger variance $\sigma\_t > \sigma\_s$ for $t > s$. This yields the tractable score function:

$$
\nabla\log q_t(\tilde{\boldsymbol{x}}_t \mid \boldsymbol{x}_0) = \frac{\boldsymbol{x}_0 - \tilde{\boldsymbol{x}}_t}{\sigma_t^2}
$$

which can be optimized with the score matching objective:

$$
\mathcal{L}_{\text{SM}}(\theta) := \frac{1}{2}\int_0^T \mathbb{E}_{p_t(\tilde{\boldsymbol{x}}_t \mid \boldsymbol{x}_0), p_{\text{data}}(\boldsymbol{x}_0)} \left[\left\lVert \frac{\boldsymbol{x}_0 - \tilde{\boldsymbol{x}}_t}{\sigma_t^2} - \boldsymbol{s}_\theta(\boldsymbol{x}, t) \right\rVert^2\right] dt
$$

While score matching provides a simulation-free way to learn the reverse-time drift through estimation of the score function, it does so under the assumption that the generative process has a simple prior distribution, typically a standard Gaussian, and the forward diffusion dynamics are chosen to be linear such that it is analytically tractable without simulation. This design constraint restricts the class of admissible dynamics to perturbations of a simple reference process which transforms between noise and data. To extend this idea to model transport between **structured distributions** with an unknown forward control drift, we introduce a likelihood-based training framework for the forward-backward SDEs that characterize the Schroedinger bridge.

### 6.2 Likelihood Training of Forward-Backward SDEs

Just like how likelihood training provides a theoretically-grounded training objective of estimating the score function in score-based generative modeling, we can derive a lower bound for the log-likelihood of the **forward-backward stochastic differential equations** (SDEs) defined in Section 4.3, which yields a tractable training objective that aims to maximize the lower bound.

Since the definition of $\boldsymbol{Y}\_t = \log\varphi\_t(\boldsymbol{X}\_t)$ and $\widehat{\boldsymbol{Y}}\_t = \log\hat{\varphi}\_t(\boldsymbol{X}\_t)$ from Section 4.3 jointly determine the log density of the SB solution $\boldsymbol{Y}\_t + \widehat{\boldsymbol{Y}}\_t = \log p\_t^\star(\boldsymbol{X}\_t)$, we can write the likelihood under the FBSDEs as the estimated value of $\boldsymbol{Y}\_0 + \widehat{\boldsymbol{Y}}\_0 = \log p\_0^\star(\boldsymbol{X}\_0)$ given a data point $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$.

$$
\log p_0^\star(\boldsymbol{x}_0) = \mathbb{E}\left[\log p_0^\star(\boldsymbol{X}_0) \mid \boldsymbol{X}_0 = \boldsymbol{x}_0\right] = \mathbb{E}\left[\boldsymbol{Y}_0 + \widehat{\boldsymbol{Y}}_0 \;\middle|\; \boldsymbol{X}_0 = \boldsymbol{x}_0\right]
$$

First, we recall the set of three coupled FBSDEs that define the solution to the non-linear SB problem:

$$
\begin{cases} d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{Z}_t) \, dt + \sigma_t d\boldsymbol{B}_t \\[4pt] d\boldsymbol{Y}_t = \frac{1}{2}\lVert \boldsymbol{Z}_t \rVert^2 dt + \boldsymbol{Z}_t^\top d\boldsymbol{B}_t \\[4pt] d\widehat{\boldsymbol{Y}}_t = \left(\nabla \cdot (\sigma_t \widehat{\boldsymbol{Z}}_t - \boldsymbol{f}) + \frac{1}{2}\lVert \widehat{\boldsymbol{Z}}_t \rVert^2 + \sigma_t^2 \boldsymbol{Z}_t^\top \widehat{\boldsymbol{Z}}_t\right) dt + \widehat{\boldsymbol{Z}}_t^\top d\boldsymbol{B}_t \end{cases}
$$

Since the $\boldsymbol{Y}\_t$ and $\widehat{\boldsymbol{Y}}\_t$ evolve via **backward SDEs** with boundary constraints $\boldsymbol{Y}\_T + \widehat{\boldsymbol{Y}}\_T = \log\pi\_T(\boldsymbol{X}\_T)$, we can write $\boldsymbol{Y}\_0$ and $\widehat{\boldsymbol{Y}}\_0$ as the terminal condition integrated backward in time:

$$
\boldsymbol{Y}_0 = \boldsymbol{Y}_T - \int_0^T \frac{1}{2}\lVert \boldsymbol{Z}_t \rVert^2 dt - \int_0^T \boldsymbol{Z}_t^\top d\boldsymbol{B}_t
$$

$$
\widehat{\boldsymbol{Y}}_0 = \widehat{\boldsymbol{Y}}_T - \int_0^T \left[\nabla \cdot (\sigma_t \widehat{\boldsymbol{Z}}_t - \boldsymbol{f}) + \frac{1}{2}\lVert \widehat{\boldsymbol{Z}}_t \rVert^2 + \sigma_t^2 \boldsymbol{Z}_t^\top \widehat{\boldsymbol{Z}}_t\right] dt - \int_0^T \widehat{\boldsymbol{Z}}_t^\top d\boldsymbol{B}_t
$$

Substituting this into the log-likelihood, combining similar terms, and using the fact that the Itô integral has zero-expectation and $\boldsymbol{Y}\_T + \widehat{\boldsymbol{Y}}\_T = \log\pi\_T(\boldsymbol{X}\_T)$, we are left with:

$$
\log p_0^\star(\boldsymbol{x}_0) = \mathbb{E}\left[\log p_T^\star(\boldsymbol{X}_T)\right] - \int_0^T \mathbb{E}\left[\frac{1}{2}\lVert \boldsymbol{Z}_t \rVert^2 + \frac{1}{2}\lVert \widehat{\boldsymbol{Z}}_t \rVert^2 + \nabla \cdot (\sigma_t \widehat{\boldsymbol{Z}}_t - \boldsymbol{f}) + \sigma_t^2 \boldsymbol{Z}_t^\top \widehat{\boldsymbol{Z}}_t\right] dt
$$

which is the log-likelihood of the data point that we aim to maximize. Each term can be interpreted as follows:

- **(i)** The terminal distribution matching reward that is maximized with the distribution at time $T$ generated by the forward bridge matches $p\_T$.
- **(ii)** The energy cost of steering samples via the forward potential drift $\boldsymbol{Z}\_t = \sigma\_t \nabla\log\varphi\_t(\boldsymbol{X}\_t)$, which is minimized when the log-likelihood is maximized.
- **(iii)** The energy cost of steering samples via the backward potential drift $\widehat{\boldsymbol{Z}}\_t = \sigma\_t \nabla\log\hat{\varphi}\_t(\boldsymbol{X}\_t)$, which is minimized when the log-likelihood is maximized.
- **(iv)** The divergence term ensures that the drift is consistent with the time-evolving density $p\_t^\star(\boldsymbol{X}\_t)$. This can be understood by expanding $\nabla \cdot (\sigma\_t \widehat{\boldsymbol{Z}}\_t)$ using the identity $\nabla \cdot (u\boldsymbol{v}) = (\nabla u) \cdot \boldsymbol{v} + u\nabla \cdot \boldsymbol{v}$:

$$
\mathbb{E}_{p_t^\star}\left[\nabla \cdot (\sigma_t \widehat{\boldsymbol{Z}}_t)\right] = \int_{\mathbb{R}^d} (\sigma_t \nabla \cdot \widehat{\boldsymbol{Z}}_t) p_t^\star d\boldsymbol{X}_t = -\int_{\mathbb{R}^d} \widehat{\boldsymbol{Z}}_t \cdot \nabla p_t^\star d\boldsymbol{X}_t
$$

- **(v)** The consistency of the forward-backward potentials. Since we want the drifts to generate exactly symmetric opposite bridges and the dot product is minimized at $-1$ when the vectors are exactly opposite, this term is minimized when the log-likelihood is maximized.

Since $\boldsymbol{Z}\_t$ and $\widehat{\boldsymbol{Z}}\_t$ determine the optimal drift in the forward and backward directions based on the SB potential, we can train a generative SB model by parameterizing them with neural networks $\boldsymbol{Z}\_t^\theta \approx \boldsymbol{Z}\_t$ and $\widehat{\boldsymbol{Z}}\_t^\phi \approx \widehat{\boldsymbol{Z}}\_t$ with parameters $\theta$ and $\phi$. Defining the SB loss with the log-likelihood expression, we get a theoretically-grounded **maximization objective** that lower bounds the true log-likelihood $\log p\_0^\star(\boldsymbol{x}\_0) \ge \mathcal{L}\_{\text{SB}}$ defined as:

$$
\mathcal{L}_{\text{SB}}(\theta, \phi) = \mathbb{E}\left[\log p_T^\star(\boldsymbol{X}_T)\right] - \int_0^T \mathbb{E}\left[\frac{1}{2}\lVert \boldsymbol{Z}_t^\theta \rVert^2 + \frac{1}{2}\lVert \widehat{\boldsymbol{Z}}_t^\phi \rVert^2 + \nabla \cdot (\sigma_t \widehat{\boldsymbol{Z}}_t^\phi - \boldsymbol{f}) + \sigma_t^2 \boldsymbol{Z}_t^{\theta\top} \widehat{\boldsymbol{Z}}_t^\phi\right] dt
$$

Jointly training both $\boldsymbol{Z}\_t^\theta$ and $\widehat{\boldsymbol{Z}}\_t^\phi$ with this objective is carried out by **(i)** simulating the forward trajectory of the SDE $\boldsymbol{X}\_{0:T}$ using $\boldsymbol{Z}\_t^\theta$ and $\widehat{\boldsymbol{Z}}\_t^\phi$, **(ii)** computing the maximum likelihood objective $\mathcal{L}\_{\text{SB}}$, and **(iii)** backpropagating through the SDE solver for every time step with respect to both $\theta$ and $\phi$, which requires maintaining the full computational graph of the SDE.

While this training scheme works for low-dimensional data, it becomes computationally infeasible for high-dimensional data like images. To overcome this, we can store SDE trajectories in a replay buffer as sequences of static states $\boldsymbol{X}\_t$ while discarding the gradient path. Crucially, this **breaks the dependency of the trajectories with the current model parameters**, but it enables us to reuse the same SDEs over multiple gradient updates. Updating one model, like $\boldsymbol{Z}\_t^\theta$, with respect to $\nabla\_\theta \mathcal{L}\_{\text{SB}}$ would change the SDE $\boldsymbol{X}\_t$, so updating $\widehat{\boldsymbol{Z}}\_t^\phi$ simultaneously with the original SDE would break the symmetry of the optimization problem.

Instead, we can leverage the symmetric property of Schroedinger bridges:

- **(i)** Given the true *forward* potential drift $\boldsymbol{Z}\_t = \sigma\_t \nabla\log\varphi\_t$, the optimal *backward* potential drift $\widehat{\boldsymbol{Z}}\_t^\phi$ can be learned to match the forward trajectories.
- **(ii)** Given the true *backward* potential drift $\widehat{\boldsymbol{Z}}\_t = \sigma\_t \nabla\log\hat{\varphi}\_t$, the optimal *forward* potential drift $\boldsymbol{Z}\_t^\theta$ can be learned to match the backward trajectories.

To train the backward potential drift given trajectories generated with the frozen forward model $\boldsymbol{Z}\_t^{\bar{\theta}}$, where $\bar{\theta} := \text{stopgrad}(\theta)$, we can use the same likelihood maximization objective but dropping all terms that are not dependent on $\phi$:

$$
\bar{\mathcal{L}}_{\text{SB}}(\phi) = -\int_0^T \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^\theta}\left[\frac{1}{2}\lVert \widehat{\boldsymbol{Z}}_t^\phi \rVert^2 + \nabla \cdot (\sigma_t \widehat{\boldsymbol{Z}}_t^\phi - \boldsymbol{f}) + \sigma_t^2 \boldsymbol{Z}_t^{\theta\top} \widehat{\boldsymbol{Z}}_t^\phi\right] dt
$$

$$
\text{s.t.} \quad \mathbb{P}^\theta : d\boldsymbol{X}_t = (\boldsymbol{f} + \sigma_t^2 \nabla\log\varphi_t(\boldsymbol{X}_t)) dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0 \sim \pi_0
$$

Given that the SB is symmetric in either direction, the objective for training the forward potential drift given trajectories generated with the frozen backward model $\widehat{\boldsymbol{Z}}\_t^{\bar{\phi}}$, where $\bar{\phi} := \text{stopgrad}(\phi)$, is the same but flipping $\boldsymbol{Z}\_t^\theta$ and $\widehat{\boldsymbol{Z}}\_t^\phi$:

$$
\bar{\mathcal{L}}_{\text{SB}}(\theta) = -\int_0^T \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^{\bar{\phi}}}\left[\frac{1}{2}\lVert \boldsymbol{Z}_t^\theta \rVert^2 + \nabla \cdot (\sigma_t \boldsymbol{Z}_t^\theta - \boldsymbol{f}) + \sigma_t^2 \widehat{\boldsymbol{Z}}_t^{\bar{\phi}\top} \boldsymbol{Z}_t^\theta\right] dt
$$

$$
\text{s.t.} \quad \mathbb{P}^{\bar{\phi}} : d\boldsymbol{X}_t = (\boldsymbol{f} - \sigma_t^2 \nabla\log\hat{\varphi}_t(\boldsymbol{X}_t)) dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_T \sim \pi_T
$$

*Derivation sketch.* This symmetric objective can be derived by defining a reversed time coordinate $s := T - t$ and defining the prior non-linear process as the reverse-time analog of the forward prior dynamics with SDE $d\boldsymbol{X}\_s = -\boldsymbol{f}(\boldsymbol{X}\_s, s) ds + \sigma\_t d\boldsymbol{B}\_s$. Then, redefining the Hopf-Cole linear PDE constraints and forward-backward SDEs with negative drift $-\boldsymbol{f}$ and following a similar derivation for the log-likelihood $\log p\_t^\star(\boldsymbol{x}\_T) = \mathbb{E}[\boldsymbol{Y}\_0 + \bar{\boldsymbol{Y}}\_0 \mid \boldsymbol{X}\_T = \boldsymbol{x}\_T]$ of a sample $\boldsymbol{x}\_T$ from the *prior* distribution or $\pi\_T$. $\square$

This section shows that the optimal forward and backward control drifts defined by the Schroedinger potentials can be learned through maximizing a lower bound on the log-likelihood of reconstructing both of the marginal constraints. These objectives lead to a symmetric alternating optimization procedure that mirrors the structure of Sinkhorn's algorithm from Section 1.5. Next, we move on to an alternative perspective, considering the optimization process as performing iterative Markovian and reciprocal projections rather than maximizing likelihoods.

### 6.3 Diffusion Schroedinger Bridge Matching

Building on the Iterative Markovian Fitting (IMF) procedure from Section 4.5, we now describe the **Diffusion Schroedinger Bridge Matching** (DSBM) algorithm. This algorithm unifies ideas from denoising diffusion and flow matching to solve the SB problem with arbitrary marginal distributions by parameterizing the Markov drift optimized to match the Schroedinger bridge drift through Markovian and reciprocal projections.

While the IMF procedure provides a theoretically grounded procedure for constructing the Schroedinger bridge through alternating KL projections in path space, its formulation remains abstract and infinite-dimensional. To make these ideas computationally tractable in high-dimensional settings, we can leverage the explicit SDE representation of the Markovian projection, defined as:

$$
d\boldsymbol{X}_t = \left(\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t^2 \mathbb{E}_{\Pi_{T|t}}\left[\nabla\log\mathbb{Q}_{T|t}(\boldsymbol{X}_T \mid \boldsymbol{X}_t) \mid \boldsymbol{X}_t\right]\right) dt + \sigma_t d\boldsymbol{B}_t
$$

and parameterize the forward-time Markov control drift $\sigma\_t \mathbb{E}\_{\Pi\_{T|t}}\left[\nabla\log\mathbb{Q}\_{T|t}(\boldsymbol{X}\_T \mid \boldsymbol{X}\_t) \mid \boldsymbol{X}\_t\right]$ with $\boldsymbol{u}\_\theta(\boldsymbol{x}, t)$ such that it converges to the optimal drift $\boldsymbol{u}^\star$ through the sequence of IMF iterations.

Although the Markovian projection preserves the bridge measure $\mathbb{M}\_t^\star = \Pi\_t$ in theory (Proposition 4.10), parameterizing only the forward-time SDE results in errors in practice, where the terminal marginal $p\_T$ generated from simulating the Forward Markovian Projection SDE may not exactly match the true marginal constraint $\pi\_T$. Therefore, to avoid error accumulation during the IMF sequence, we also parameterize the **reverse-time Markovian projection**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.1</span><span class="math-callout__name">(Forward and Reverse Time Markovian Projections)</span></p>

Given a mixture of bridges $\Pi = \Pi\_{0,T}\mathbb{Q}\_{\cdot|0,T}$ in the reciprocal class $\Pi \in \mathcal{R}(\mathbb{Q})$ of the reference measure $\mathbb{Q}$ generated by the SDE $d\boldsymbol{X}\_t = \boldsymbol{f}(\boldsymbol{X}\_t, t) dt + \sigma\_t d\boldsymbol{B}\_t$, the Markovian projection $\mathbb{M} := \text{proj}\_{\mathcal{M}}(\Pi)$ can be written as both forward and reverse time SDEs defined as:

$$
d\boldsymbol{X}_t = \left(\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t^2 \mathbb{E}_{\Pi_{T|t}}\left[\nabla\log\mathbb{Q}_{T|t}(\boldsymbol{X}_T \mid \boldsymbol{X}_t) \mid \boldsymbol{X}_t\right]\right) dt + \sigma_t d\boldsymbol{B}_t
$$

$$
d\bar{\boldsymbol{X}}_s = \left(-\boldsymbol{f}(\boldsymbol{X}_s, T - s) + \sigma_{T-s}^2 \mathbb{E}_{\Pi_{0|T-s}}\left[\nabla\log\mathbb{Q}_{T-s|0}(\bar{\boldsymbol{X}}_s \mid \bar{\boldsymbol{X}}_T) \mid \bar{\boldsymbol{X}}_s\right]\right) dt + \sigma_{T-s} d\bar{\boldsymbol{B}}_s
$$

with initial conditions $\boldsymbol{X}\_0 \sim \Pi\_0$ and $\bar{\boldsymbol{X}}\_0 \sim \Pi\_T$, respectively.

*Proof.* The proof of this proposition follows directly from the definition of the Markovian projection in (4.10) and applying the Time Reversal Formula described in Section 4.2. $\square$

</div>

Parameterizing both the forward-time Markovian projection with drift $\boldsymbol{u}\_\theta(\boldsymbol{x}, t)$ and the reverse-time Markovian projection with drift $\boldsymbol{u}\_\phi(\boldsymbol{x}, t)$, we outline the Diffusion Schroedinger Bridge Matching (DSBM) algorithm as follows.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Diffusion Schroedinger Bridge Matching)</span></p>

DSBM generates a sequence of Markov projections $(\mathbb{M}^n)\_{n \in \mathbb{N}}$ and reciprocal projections $(\Pi^n)\_{n \in \mathbb{N}}$ initialized at $\Pi^0 := \pi\_{0,T}\mathbb{Q}\_{\cdot|0,T}$ by alternating between the following steps:

- **(1a)** Solve the forward-time Markovian projection $\mathbb{M}^{2n+1} := \text{proj}\_{\mathcal{M}}(\Pi^{2n})$ by updating a parameterized drift $\boldsymbol{u}\_\theta$ to minimize $\mathcal{L}\_{\text{DSBM}}(\theta) := \text{KL}(\Pi^{2n} \| \mathbb{M}^\theta)$.
- **(1b)** Define the reciprocal projection as $\Pi^{2n+1} := \mathbb{M}^{2n+1}\mathbb{Q}\_{\cdot|0,T}$
- **(2a)** Solve the backward-time Markovian projection $\mathbb{M}^{2n+2} := \text{proj}\_{\mathcal{M}}(\Pi^{2n+1})$ by updating a parameterized drift $\boldsymbol{u}\_\phi$ to minimize $\mathcal{L}\_{\text{DSBM}}(\phi) := \text{KL}(\Pi^{2n+1} \| \mathbb{M}^\phi)$.
- **(2b)** Define the reciprocal projection as $\Pi^{2n+2} := \mathbb{M}^{2n+2}\mathbb{Q}\_{\cdot|0,T}$.

</div>

To learn the forward and reverse time Markovian projections in Steps **(1a)** and **(2a)**, we can minimize loss functions defined as the KL divergence as derived in Section 2.6:

$$
\mathcal{L}_{\text{DSBM}}(\theta) = \int_0^T \mathbb{E}_{\Pi_{t,T}}\left[\left\lVert \sigma_t \nabla\log\mathbb{Q}_{T|t}(\boldsymbol{X}_T \mid \boldsymbol{X}_t) - \boldsymbol{u}_\theta(\boldsymbol{X}_t, t) \right\rVert^2\right] dt
$$

$$
\mathcal{L}_{\text{DSBM}}(\phi) = \int_0^T \mathbb{E}_{\Pi_{t,0}}\left[\left\lVert \sigma_t \nabla\log\mathbb{Q}_{t|0}(\boldsymbol{X}_t \mid \boldsymbol{X}_0) - \boldsymbol{u}_\phi(\boldsymbol{X}_t, t) \right\rVert^2\right] dt
$$

Given sufficient expressivity of $\theta, \phi$, it is easy to see that optimizing the above losses for all $(\boldsymbol{x}, t)$ exactly yields the control drift of the Markov projection $\text{proj}\_{\mathcal{M}}(\Pi)$:

$$
\boldsymbol{u}_{\theta^\star}(\boldsymbol{x}, t) = \sigma_t \mathbb{E}_{\Pi_{T|t}}\left[\nabla\log\mathbb{Q}_{T|t}(\boldsymbol{X}_T \mid \boldsymbol{X}_t) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

$$
\boldsymbol{u}_{\phi^\star}(\boldsymbol{x}, t) = \sigma_t \mathbb{E}_{\Pi_{0|t}}\left[\nabla\log\mathbb{Q}_{t|0}(\boldsymbol{X}_t \mid \boldsymbol{X}_0) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

The corresponding reciprocal projections performed in **(1b)** and **(2b)** are obtained by first simulating trajectories $\boldsymbol{X}\_{0:T}$ either with the forward SDE using $\boldsymbol{u}\_\theta$ or in the reverse SDE using $\boldsymbol{u}\_\phi$ to obtain samples from the endpoint law $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \mathbb{M}\_{0,T}$, and then sampling from the conditional bridge $\boldsymbol{X}\_t \sim \mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$ of the reference process $\mathbb{Q}$.

Optimizing these losses exactly performs the Iterative Markovian Fitting (IMF) procedure from Section 4.5 through parameterized control drifts of SDEs. Therefore, by Theorem 4.17, we have that the unique fixed point of the diffusion SBM algorithm yields the optimal Schroedinger bridge $\mathbb{P}^\star$. Rather than separating the forward and reverse Markovian projection steps, it has been shown that they can be performed simultaneously, which mimics the true IMF procedure where each projection corresponds to the exact Markov and reciprocal projections.

#### Joint Training of Forward and Reverse Markovian Projections

Since both objectives can be computed from sampling a pair $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \Pi\_{0,T}$ from the joint distribution and sampling the intermediate state $\boldsymbol{X}\_t \sim \mathbb{Q}\_{t|0,T}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$ from the bridge measure, we can optimize both $\theta$ and $\phi$ *jointly*. After each iteration $n$, the updated parameters can be used to define the forward Markov process $\mathbb{M}\_f^{n+1}$ and the backward Markov process $\mathbb{M}\_b^{n+1}$ with the following SDEs:

$$
\mathbb{M}_f^{n+1} : d\boldsymbol{X}_t = [\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t u_\theta(\boldsymbol{X}_t, t)] dt + \sigma_t d\boldsymbol{B}_t, \qquad \boldsymbol{X}_0 \sim \pi_0
$$

$$
\mathbb{M}_b^{n+1} : d\bar{\boldsymbol{X}}_s = [-\boldsymbol{f}(\bar{\boldsymbol{X}}, T - s) + \sigma_{T-s}\boldsymbol{u}_\phi(\bar{\boldsymbol{X}}_s, T - s)] ds + \sigma_{T-s} d\bar{\boldsymbol{B}}_s \qquad \bar{\boldsymbol{X}}_0 \sim \pi_T
$$

At equilibrium, the **forward and backward SDEs should match**, which means reversing $\mathbb{M}\_T^n$ yields $\mathbb{M}\_b^n$ and reversing $\mathbb{M}\_b^n$ yields the $\mathbb{M}\_f^n$. To enforce this during training, we can compute the time-reversal of the backward SDE using the Time Reversal Formula to get:

$$
d\boldsymbol{X}_t = \left[\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t\underbrace{(-\boldsymbol{u}_\phi(\boldsymbol{X}_t, t) + \sigma_t \nabla\log\Pi_t^{2n}(\boldsymbol{X}_t))}_{\text{should match } \boldsymbol{u}_\theta(\boldsymbol{X}_t, t)}\right] dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0 \sim \pi_0
$$

To ensure the control drifts are aligned, we leverage a **consistency loss** defined as:

$$
\mathcal{L}_{\text{cons}}(\theta, \phi) = \int_0^T \mathbb{E}_{\Pi_t^{2n}}\left[\left\lVert \boldsymbol{u}_\theta(\boldsymbol{X}_t, t) + \boldsymbol{u}_\phi(\boldsymbol{X}_t, t) - \sigma_t \nabla\log\Pi_t^{2n}(\boldsymbol{X}_t) \right\rVert^2\right] dt
$$

where we rewrite the score function $\nabla\log\Pi\_t^{2n}(\boldsymbol{X}\_t)$ with the known conditional densities $\mathbb{Q}\_{T|t}$ and $\mathbb{Q}\_{0|t}$ using the identity:

$$
\nabla\log\Pi_t^{2n}(\boldsymbol{x}) = \mathbb{E}_{\Pi_{T|t}^{2n}}\left[\nabla\log\mathbb{Q}_{T|t} \mid \boldsymbol{X}_t = \boldsymbol{x}\right] + \mathbb{E}_{\Pi_{0|t}^{2n}}\left[\nabla\log\mathbb{Q}_{0|t} \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

which defines the marginal score at time $t$ as the sum of the conditional Gaussian scores at the endpoints. Therefore, the total **joint training loss** can be defined as:

$$
\mathcal{L}_{\text{DSBM}}(\theta, \phi) = \mathcal{L}_{\text{DSBM}}(\theta) + \mathcal{L}_{\text{DSBM}}(\phi) + \lambda\mathcal{L}_{\text{cons}}(\theta, \phi)
$$

where $\lambda > 0$ is a positive weight that defines the strength of the consistency loss.

This formulation provides a tractable approach for performing path-space projections using parameterized drifts that can be simulated at inference time via an SDE solver. In practice, the Markov projection requires learning both forward and backward drifts to satisfy the marginal constraints, despite the underlying equivalence of the bridge measures. However, this approach remains computationally intensive, as training still relies on simulating full stochastic trajectories due to the absence of a closed-form sampling procedure for the optimal bridge. Motivated by this limitation, we now consider an alternative perspective in which the intermediate states admit tractable expression, enabling simulation-free matching objectives.

### 6.4 Simulation-Free Score and Flow Matching

To overcome the restriction to the Gaussian prior distribution of score-based generative modeling described in Section 6.1, we highlight the simulation-free score and flow matching ($[\text{SF}]^2\text{M}$) framework which extends score matching to arbitrary prior distributions. Crucially, we will show that given the **optimal entropic OT coupling** (Section Static SB Problem), $[\text{SF}]^2\text{M}$ solves the Dynamic SB Problem through a **endpoint-conditioned objective** with the same gradient as the unconditional objective.

This framework considers a data-driven SB problem, where we have empirical samples from both the marginal distributions $\boldsymbol{x}\_0 \sim \pi\_0$ and $\boldsymbol{x}\_T \sim \pi\_T$. In this setting, we can solve the entropic optimal transport (OT) problem between empirical samples to determine the optimal coupling $\pi\_{0,T}^\star$, from which the solution to the dynamic SB problem with Brownian reference process $\mathbb{Q}$ is defined simply as a **mixture of Brownian bridges** weighted by the optimal static coupling $\pi\_{0,T}^\star$ as proven in Proposition 4.3 and Corollary 4.2.

Recall from Section 4.2, where we derived the forward and backward SDEs as:

$$
\begin{cases} d\boldsymbol{X}_t = \sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t) dt + \sigma_t d\boldsymbol{B}_t \\[4pt] d\bar{\boldsymbol{X}}_s = -\sigma_t \boldsymbol{u}(\bar{\boldsymbol{X}}_s, T - s) + \sigma_{T-s}^2 \nabla\log p(\bar{\boldsymbol{X}}_s, T - s) ds + \sigma_{T-s} d\widetilde{\boldsymbol{B}}_s \end{cases}
$$

where we define the reference drift as pure Brownian motion $\boldsymbol{f} := 0$. Crucially, the score function $\nabla\log p\_t(\boldsymbol{X}\_t)$ appears in the reverse-time drift as a *correction* term that compensates for the entropy-producing forward diffusion, ensuring that the reversed dynamics reproduce the correct marginal distributions. This decomposition of the control and score function motivates a **combined objective** that learns a parameterized control drift $\boldsymbol{u}\_\theta$, which learns the *reverse control* $\boldsymbol{u}\_\theta(\boldsymbol{x}, t) \approx \boldsymbol{u}(\boldsymbol{x}, t)$, and the score function $\nabla\log p\_t(\boldsymbol{x})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.2</span><span class="math-callout__name">(Unconditional Score and Flow Matching Objective)</span></p>

The **unconditional score and flow matching objective** aims to match a parameterized control field $\boldsymbol{u}\_\theta(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ and score function $\boldsymbol{s}\_\theta(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ to the true velocity and score function defining the solution to the Dynamic SB Problem by minimizing:

$$
\mathcal{L}_{\text{U}[\text{SF}]^2\text{M}}(\theta) := \int_0^T \mathbb{E}_{p_t}\left[\underbrace{\lVert \boldsymbol{u}_\theta(\boldsymbol{x}, t) - \boldsymbol{u}(\boldsymbol{x}, t) \rVert^2}_{\text{flow matching loss}} + \underbrace{\lambda(t)^2 \lVert \boldsymbol{s}_\theta(\boldsymbol{x}, t) - \nabla\log p_t^\star(\boldsymbol{x}) \rVert^2}_{\text{score matching loss}}\right] dt
$$

where $p\_t^\star$ is the optimal marginal density of the dynamic SB, and $\lambda(t) : [0, T] \to \mathbb{R}$ is some positive weight.

</div>

While this objective is theoretically sound, both $\boldsymbol{u}(\boldsymbol{x}, t)$ and $\nabla\log p\_t(\boldsymbol{x})$ are undefined or intractable for general target distributions. In this setting, we assume access to explicit samples from both marginal distributions $\pi\_0$ and $\pi\_T$ and define the tractable control drift and score for a Brownian bridge between a predefined coupling $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \pi\_{0,T}$ given by:

$$
\boldsymbol{u}(\boldsymbol{x}, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) = \frac{1 - 2t}{t(1 - t)}(\boldsymbol{x} - (t\boldsymbol{x}_1 + (1 - t)\boldsymbol{x}_0)) + (\boldsymbol{x}_1 - \boldsymbol{x}_0)
$$

$$
\nabla\log p_t(\boldsymbol{x} \mid \boldsymbol{x}_0, \boldsymbol{x}_T) = \frac{t\boldsymbol{x}_1 + (1 - t)\boldsymbol{x}_0 - \boldsymbol{x}}{\sigma_t^2 t(1 - t)}
$$

Using this definition of the endpoint-conditioned velocity and score function, we can define the conditional velocity and score over the empirical distribution $\pi\_0$ by taking an expectation:

$$
\boldsymbol{u}(\boldsymbol{x}, t) = \mathbb{E}_{x_{0,T}}\left[\frac{\boldsymbol{u}(\boldsymbol{x}, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) p_t(\boldsymbol{x} \mid \boldsymbol{x}_0, \boldsymbol{x}_T)}{p_t(\boldsymbol{x})}\right]
$$

$$
\nabla\log p_t(\boldsymbol{x}) = \mathbb{E}_{x_{0,T}}\left[\frac{p_t(\boldsymbol{x} \mid \boldsymbol{x}_0, \boldsymbol{x}_T)}{p_t(\boldsymbol{x})}\nabla\log p_t(\boldsymbol{x} \mid \boldsymbol{x}_0, \boldsymbol{x}_T)\right]
$$

These tractable definitions for the *conditional* control drift and score function motivate the definition of the **conditional score and flow matching objective**, which we show yields the same gradients as the unconditional objective.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.3</span><span class="math-callout__name">(Conditional Score and Flow Matching Objective)</span></p>

Consider the **conditional score and flow matching objective** which aims to match a parameterized control drift $\boldsymbol{v}\_\theta(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ and score function $\boldsymbol{s}\_\theta(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ to a distribution of velocity and score functions conditioned on the endpoint $\boldsymbol{z} \sim \pi\_T$ by minimizing:

$$
\mathcal{L}_{[\text{SF}]^2\text{M}}(\theta) := \int_0^T \mathbb{E}_{p_{t|0,T}, \pi_{0,T}}\left[\underbrace{\lVert \boldsymbol{v}_\theta(\boldsymbol{x}, t) - \boldsymbol{u}(\boldsymbol{x}, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \rVert^2}_{\text{conditional flow matching loss}} + \underbrace{\lambda(t)^2 \lVert \boldsymbol{s}_\theta(\boldsymbol{x}, t) - \nabla\log p_t(\boldsymbol{x} \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \rVert^2}_{\text{conditional score matching loss}}\right] dt
$$

where the expectation is taken over samples from the endpoint law $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \pi\_{0,T}$ and samples from the conditional distribution at time $t \in [0, T]$ given target endpoints $\boldsymbol{x} \sim p\_{t|0,T}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$. Then, we have that the gradients of the conditional objective match the gradients of the Unconditional Objective such that $\nabla\_\theta \mathcal{L}\_{[\text{SF}]^2\text{M}}(\theta) = \nabla\_\theta \mathcal{L}\_{\text{U}[\text{SF}]^2\text{M}}(\theta)$.

</div>

*Proof.* The goal of this proof is to show the equivalence between the gradients of the conditional and unconditional expectations:

$$
\nabla_\theta \mathbb{E}_{p_{t|0,T}, \pi_{0,T}}\left[\lVert \boldsymbol{w}_\theta(\boldsymbol{x}, t) - \boldsymbol{w}(\boldsymbol{x}, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \rVert^2\right] = \nabla_\theta \mathbb{E}_{p_t}\left[\lVert \boldsymbol{w}_\theta(\boldsymbol{x}, t) - \boldsymbol{w}(\boldsymbol{x}, t) \rVert^2\right]
$$

which can be applied for both the conditional flow matching loss with $\boldsymbol{w}(\boldsymbol{x}, t \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T) := \boldsymbol{u}(\boldsymbol{x}, t \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$ and the conditional score matching loss $\boldsymbol{w}(\boldsymbol{x}, t \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T) := \nabla\log p\_t(\boldsymbol{x} \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$. Expanding the squared loss in the conditional objective:

$$
\lVert \boldsymbol{w}_\theta(\boldsymbol{x}, t) - \boldsymbol{w}(\boldsymbol{x}, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \rVert^2 = \underbrace{\lVert \boldsymbol{w}_\theta(\boldsymbol{x}, t) \rVert^2}_{\text{independent of } \boldsymbol{x}_0, \boldsymbol{x}_T} - 2\langle \boldsymbol{w}_\theta(\boldsymbol{x}, t), \boldsymbol{w}(\boldsymbol{x}, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \rangle + \underbrace{\lVert \boldsymbol{w}(\boldsymbol{x}, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \rVert^2}_{\text{independent of } \theta}
$$

where we observe that the first term is independent of the conditional pair $(\boldsymbol{x}\_0, \boldsymbol{x}\_T)$, which means that it is clearly equivalent to the unconditional gradient. The last term is independent of $\theta$ which has a gradient of zero with respect to $\theta$. Therefore, we can write the difference between the unconditional and conditional expectations as:

$$
\nabla_\theta \mathbb{E}_{p_{t|0,T}, \pi_{0,T}}\left[-2\langle \boldsymbol{w}_\theta(\boldsymbol{x}, t), \boldsymbol{w}(\boldsymbol{x}, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) \rangle\right] = \nabla_\theta \mathbb{E}_{p_t}\left[-2\langle \boldsymbol{w}_\theta(\boldsymbol{x}, t), \boldsymbol{w}(\boldsymbol{x}, t) \rangle\right]
$$

Now, we aim to show that the expectations are equivalent. Starting from the unconditional expectation $\nabla\_\theta \mathbb{E}\_{p\_t}\left[\langle \boldsymbol{w}\_\theta(\boldsymbol{x}, t), \boldsymbol{w}(\boldsymbol{x}, t) \rangle\right]$, we expand $\boldsymbol{w}(\boldsymbol{x}, t)$ as the marginalization of $\boldsymbol{w}(\boldsymbol{x}, t \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$ over the endpoint law weighted by $\frac{p\_t(\boldsymbol{x} \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)}{p\_t(\boldsymbol{x})}$, factor out the scalar values, and apply Fubini's theorem to change the order of integration, resulting in the equivalence between the marginal and conditional objectives. $\square$

This result is well established in flow matching literature as a way of training parameterized flows that approximate an intractable marginal distribution using empirical samples from the target data distribution. However, this marginal distribution does not yet solve the Schroedinger bridge problem, as the coupled distribution $\pi\_{0,T}$ from which $(\boldsymbol{x}\_0, \boldsymbol{x}\_T)$ is sampled does not necessarily align with the entropic OT coupling. To establish how the conditional score and flow matching objective can be used to solve the Dynamic SB Problem, we establish the following proposition.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.4</span><span class="math-callout__name">(Score and Flow Matching Recovers the Schroedinger Bridge)</span></p>

Let $\mathbb{P}^\star$ denote the path measure that solves the dynamic SB with marginal constraints $\pi\_0, \pi\_T \in \mathcal{P}(\mathbb{R}^d)$ and pure Brownian motion reference process $\sigma\mathbb{B}$. Consider the optimal endpoint law $\pi\_{0,T}^\star$ that solves the Entropic OT Problem with quadratic transport cost $c(\boldsymbol{x}, \boldsymbol{y}) := \lVert \boldsymbol{x} - \boldsymbol{y} \rVert^2$:

$$
\pi_{0,T}^\star = \arg\min_{\pi_{0,T} \in \Pi(\pi_0, \pi_T)} \left\lbrace \int_{\mathbb{R}^d \times \mathbb{R}^d} \lVert \boldsymbol{x} - \boldsymbol{y} \rVert^2 d\pi_{0,T}(\boldsymbol{x}, \boldsymbol{y}) + 2\sigma_t^2 \text{KL}(\pi_{0,T} \| \pi_0 \otimes \pi_T) \right\rbrace
$$

If the parameterized score function $\boldsymbol{s}\_\theta^\star(\boldsymbol{x}, t)$ and control drift $\boldsymbol{v}\_\theta^\star(\boldsymbol{x}, t)$ globally minimize the Conditional Objective under the coupling $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \pi\_{0,T}^\star$, then the resulting stochastic process is given by the SDE:

$$
d\boldsymbol{X}_t = \left[\boldsymbol{v}_\theta^\star(\boldsymbol{X}_t, t) + \sigma_t^2 \boldsymbol{s}_\theta^\star(\boldsymbol{X}_t, t)\right] dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0 \sim \pi_0
$$

and generates the Schroedinger bridge path measure $\mathbb{P}^\star$.

</div>

*Proof Sketch.* The **key idea** is that the Conditional Objective is minimized pointwise for all $(\boldsymbol{x}, t) \in \mathbb{R}^d \times [0, T]$ when:

$$
\boldsymbol{v}_\theta(\boldsymbol{x}, t) = \mathbb{E}_{(\boldsymbol{x}_0, \boldsymbol{x}_T) \sim p_{0,T|t}}\left[\boldsymbol{u}(\boldsymbol{x}, t \mid \boldsymbol{x}_0, \boldsymbol{x}_T)\right], \quad \boldsymbol{s}_\theta(\boldsymbol{x}, t) = \mathbb{E}_{(\boldsymbol{x}_0, \boldsymbol{x}_T) \sim p_{0,T|t}}\left[\nabla\log p_t(\boldsymbol{x} \mid \boldsymbol{x}_0, \boldsymbol{x}_T)\right]
$$

where $p\_{0,T|t}$ is the posterior distribution over the endpoint law $\pi\_{0,T}$ given an intermediate state $\boldsymbol{x}$ which can be expressed using Bayes' rule as:

$$
p_{0,T|t}(\boldsymbol{x}_0, \boldsymbol{x}_T \mid \boldsymbol{x}) = \frac{p_{t|0,T}(\boldsymbol{x} \mid \boldsymbol{x}_0, \boldsymbol{x}_T)\pi_{0,T}(\boldsymbol{x}_0, \boldsymbol{x}_T)}{p_t(\boldsymbol{x})}
$$

This minimizer is exactly the probability flow drift and score function of the **mixture of Brownian bridges**. From Section 4.1 Corollary 4.2, we showed if $\pi\_{0,T}^\star$ is chosen to be the **entropic OT plan** with quadratic transport cost, then this bridge mixture is precisely the Schroedinger bridge, so the learned SDE recovers the Schroedinger bridge $\mathbb{P}^\star$. $\square$

By leveraging a two-stage framework that first determines the optimal static SB coupling and learning the conditional velocity and score functions, score and flow matching provide a scalable, simulation-free framework for learning Schroedinger bridges. However, a key limitation of this approach is that it requires explicit samples from both the source and target distributions to construct the conditional objectives and endpoint couplings. In many practical settings, such as those that are only known up to an unnormalized density or energy function, explicit samples may not be readily available. This motivates our discussion of alternative approaches that do not rely on paired or explicit samples from the target distribution.

### 6.5 Schroedinger Bridge with Adjoint Matching

In this section, we will explore how the **adjoint matching** framework has been applied to efficiently solve the Schroedinger bridge problem as described in *Adjoint Schroedinger Bridge Sampler*. **Adjoint matching** (AM) is a generative modeling framework that efficiently solves the stochastic optimal control (SOC) problem, which has been extended to various applications, including fine-tuning and sampling. The **adjoint variable** in the context of the Schroedinger bridge problem refers to the gradient of the value function $V\_t(\boldsymbol{x})$ which defines the optimal control drift:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = \sigma_t \nabla\psi_t(\boldsymbol{x}) = -\sigma_t \nabla V_t(\boldsymbol{x}) = -\sigma_t \nabla J^\star(\boldsymbol{x}, t; u)
$$

Standard methods for solving for the adjoint variable by directly differentiating through the SOC Objective or directly matching the target $\nabla J^\star(\boldsymbol{x}, t; u^\star)$ with importance weighted matching objective, however, adjoint matching introduces a computationally favorable and fundamentally different approach.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.5</span><span class="math-callout__name">(Adjoint State)</span></p>

The **adjoint state**, denoted $\boldsymbol{a} : C([t, T], \mathbb{R}^d) \times [0, T] \to \mathbb{R}^d$, of a stochastic optimal control (SOC) problem is defined by taking the gradient of the SOC objective to get:

$$
\boldsymbol{a}(\boldsymbol{X}_{t:T}, t) = \nabla\left(\int_t^T \left(\frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_s, s) \rVert^2 + c(\boldsymbol{X}_s, s)\right) ds + \Phi(\boldsymbol{X}_T)\right)
$$

which yields the gradient field of $J(\boldsymbol{x}, t; \boldsymbol{u})$ in expectation:

$$
\nabla J(\boldsymbol{x}, t; \boldsymbol{u}) = \mathbb{E}_{\boldsymbol{X}_{t:T} \sim \mathbb{P}^u}\left[\boldsymbol{a}(\boldsymbol{X}_{t:T}, t) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

The adjoint state can be solved backward in time, given the terminal condition $\boldsymbol{a}(\boldsymbol{X}\_{t:T}, T; \boldsymbol{u}) = \nabla\_{\boldsymbol{x}\_T}\Phi(\boldsymbol{X}\_T)$ by integrating:

$$
\frac{d}{dt}\boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \boldsymbol{u}) = -\left[\boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \boldsymbol{u})^\top\left(\nabla(\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t))\right) + \nabla\left(c(\boldsymbol{X}_t, t) + \frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_t, t) \rVert^2\right)\right]
$$

which yields the alternative integral form:

$$
\boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \boldsymbol{u}) = \int_t^T \left(\nabla(\boldsymbol{f}(\boldsymbol{X}_s, s)^\top \boldsymbol{a}(\boldsymbol{X}_{s:T}, s; \boldsymbol{u}) + \sigma_t \boldsymbol{v}(\boldsymbol{X}_s, s)) + \nabla\left(c(\boldsymbol{X}_s, s) + \frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_s, s) \rVert^2\right)\right) ds + \nabla\Phi(\boldsymbol{X}_T)
$$

</div>

Rather than directly matching the target adjoint vector field $-\sigma\_t \nabla J^\star(\boldsymbol{x}, t; \boldsymbol{u}^\star)$, it considers an objective that matches the vector field generated by the *current control* $-\sigma\_t \nabla J(\boldsymbol{x}, t; \boldsymbol{u})$, which bypasses the need for importance weighting while obtaining an optimizer gradient that is *equal*, in expectation to that of the target adjoint objective.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.6</span><span class="math-callout__name">(Basic Adjoint Matching Yields the Optimal Control)</span></p>

Consider the **basic adjoint matching objective** defined with the adjoint state $\boldsymbol{a} : C([0, T]; \mathbb{R}^d) \times [0, T] \to \mathbb{R}^d$ as:

$$
\mathcal{L}_{\text{basic-AM}}(\boldsymbol{u}) := \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^{\bar{u}}}\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t, t) + \sigma_t \nabla\boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \bar{\boldsymbol{u}}) \rVert^2 dt\right], \quad \bar{\boldsymbol{u}} = \text{stopgrad}(\boldsymbol{u})
$$

where $\bar{\boldsymbol{u}} = \text{stopgrad}(\boldsymbol{u})$ is the control drift where the gradient with respect to $\boldsymbol{u}$ that generates the path are not tracked, i.e., the path $\boldsymbol{X}\_{t:T}$ cannot be differentiated through. Then, $\mathcal{L}\_{\text{basic-AM}}(\boldsymbol{u})$ has a **unique** minimizer that is equal to the optimal control $\boldsymbol{u}^\star$.

</div>

*Proof.* To derive the minimizer of the functional objective, we can compute the first variation of $\mathcal{L}\_{\text{basic-AM}}$ by defining a slightly perturbed control drift $(\boldsymbol{u} + \epsilon\boldsymbol{v})$, where $\boldsymbol{v} : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ is an arbitrary vector field, to get:

$$
\frac{d}{d\epsilon}\mathcal{L}_{\text{basic-AM}}(\boldsymbol{u} + \epsilon\boldsymbol{v}) = \mathbb{E}_{\boldsymbol{X}_{t:T} \sim \mathbb{P}^{\bar{u}}}\left[\int_0^T \langle \boldsymbol{v}(\boldsymbol{X}_t, t), \boldsymbol{u}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \bar{\boldsymbol{u}}) \rangle dt\right]
$$

$$
= \mathbb{E}_{\boldsymbol{x} \sim p_t^{\bar{u}}}\left[\int_0^T \left\langle \boldsymbol{v}(\boldsymbol{x}, t), \underbrace{\boldsymbol{u}(\boldsymbol{x}, t) + \sigma_t \mathbb{E}_{\boldsymbol{X}_{t:T} \sim \mathbb{P}^{\bar{u}}}\left[\boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \bar{\boldsymbol{u}}) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]}_{\text{must vanish point-wise at optimality}} \right\rangle dt\right]
$$

where we use the law of total expectation given that only $\boldsymbol{a}(\boldsymbol{X}\_{t:T}, t; \bar{\boldsymbol{u}})$ depends on the path $\boldsymbol{X}\_{t:T}$. Given that $\boldsymbol{v}(\boldsymbol{X}\_t, t)$ is *arbitrary*, the only solution where the first variation evaluates to zero for *all* $\boldsymbol{v}$, which occurs if and only if for all $(\boldsymbol{x}, t)$, the following is satisfied:

$$
\boldsymbol{u}(\boldsymbol{x}, t) + \sigma_t \mathbb{E}_{\boldsymbol{X}_{t:T} \sim \mathbb{P}^{\bar{u}}}\left[\boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \bar{\boldsymbol{u}}) \mid \boldsymbol{X}_t = \boldsymbol{x}\right] = 0
$$

Therefore, we can write the functional derivative of $\mathcal{L}\_{\text{basic-AM}}(\boldsymbol{u})$ with respect to $\boldsymbol{u}$ evaluated pointwise at $(\boldsymbol{x}, t)$ as:

$$
\frac{\delta}{\delta u}\mathcal{L}_{\text{basic-AM}}(\boldsymbol{u})(\boldsymbol{x}, t) = \boldsymbol{u}(\boldsymbol{x}, t) + \sigma_t \mathbb{E}_{\boldsymbol{X}_{t:T} \sim \mathbb{P}^{\bar{u}}}\left[\boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \bar{\boldsymbol{u}}) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

where the **critical points** satisfy:

$$
\boldsymbol{u}(\boldsymbol{x}, t) = -\sigma_t \mathbb{E}_{\boldsymbol{X}_{t:T} \sim \mathbb{P}^{\bar{u}}}\left[\boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \bar{\boldsymbol{u}}) \mid \boldsymbol{X}_t = \boldsymbol{x}\right] \overset{}{=} -\sigma_t \nabla J(\boldsymbol{x}, t; \boldsymbol{u})
$$

To prove that any $\boldsymbol{u}$ that satisfies this for *all* $(\boldsymbol{x}, t) \in \mathbb{R}^d \times [0, T]$ is the optimal control $\boldsymbol{u}^\star$, we establish the following Lemma.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.7</span><span class="math-callout__name">(Fixed Point Solution to Optimal Control)</span></p>

Consider a control drift $\boldsymbol{u}(\boldsymbol{x}, t)$ that satisfies $\boldsymbol{u}(\boldsymbol{x}, t) = -\sigma\_t \nabla J(\boldsymbol{x}, t; \boldsymbol{u})$ for all $(\boldsymbol{x}, t) \in \mathbb{R}^d \times [0, T]$. Then, we have that the function $J(\cdot, \cdot; \boldsymbol{u}) : \mathbb{R}^d \times [0, T] \to \mathbb{R}$ satisfies the Hamilton-Jacobi-Bellman equation. Since the HJB equation has a **unique solution**, we can conclude that:

$$
\forall (\boldsymbol{x}, t) \in \mathbb{R}^d \times [0, T], \quad J(\boldsymbol{x}, t; \boldsymbol{u}) = V_t(\boldsymbol{x}) \implies \boldsymbol{u}(\boldsymbol{x}, t) \equiv \boldsymbol{u}^\star(\boldsymbol{x}, t) = -\sigma_t \nabla V_t(\boldsymbol{x})
$$

</div>

*Proof.* First, we decompose $J(\boldsymbol{x}, t; \boldsymbol{u})$ using Bellman's Principle of Optimality which states that the optimal cost of time $t$ is equal to the incremental cost over $[t, t + \Delta t]$ and the cost of time $t + \Delta t$ to get:

$$
J(\boldsymbol{x}, t; \boldsymbol{u}) = \mathbb{E}\left[\int_t^{t+\Delta t} \left(\frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_s, s) \rVert^2 + c(\boldsymbol{X}_s, s)\right) ds \;\middle|\; \boldsymbol{X}_t = \boldsymbol{x}\right] + \mathbb{E}\left[J(\boldsymbol{X}_{t+\Delta t}, t + \Delta t; \boldsymbol{u}) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

Subtracting $J(\boldsymbol{x}, t; \boldsymbol{u})$ from both sides, dividing by $\Delta t$, and taking the continuous time limit $\Delta t \to 0$, we get:

$$
0 = \mathcal{A}^u J(\boldsymbol{x}, t; \boldsymbol{u}) + \frac{1}{2}\lVert u(\boldsymbol{x}, t) \rVert^2 + c(\boldsymbol{x}, t)
$$

Expanding the controlled generator $\mathcal{A}^u J(\boldsymbol{x}, t; \boldsymbol{u})$ and substituting $\boldsymbol{u}(\boldsymbol{x}, t) = -\sigma\_t \nabla J(\boldsymbol{x}, t; \boldsymbol{u})$, completing the square, and rearranging terms, we recover the HJB equation:

$$
\partial_t J(\boldsymbol{x}, t; \boldsymbol{u}) = -\langle \nabla J(\boldsymbol{x}, t; \boldsymbol{u}), \boldsymbol{f}(\boldsymbol{x}, t) \rangle - \frac{\sigma_t^2}{2}\Delta J(\boldsymbol{x}, t; \boldsymbol{u}) + \frac{\sigma_t^2}{2}\lVert \nabla J(\boldsymbol{x}, t; \boldsymbol{u}) \rVert^2 - c(\boldsymbol{x}, t)
$$

and since we define $J(\boldsymbol{x}, T; \boldsymbol{u}) = \Phi(\boldsymbol{x})$, we have shown that $J(\cdot, \cdot; \boldsymbol{u})$ satisfies the HJB for all $(\boldsymbol{x}, t) \in \mathbb{R}^d \times [0, T]$ given $\boldsymbol{u}(\boldsymbol{x}, t) = -\sigma\_t \nabla J(\boldsymbol{x}, t; \boldsymbol{u})$. By uniqueness of the solution to the HJB, we can conclude that $J(\boldsymbol{x}, t; \boldsymbol{u}) = V\_t(\boldsymbol{x})$ and $\boldsymbol{u}(\boldsymbol{x}, t) = \boldsymbol{u}^\star(\boldsymbol{x}, t)$ is the optimal control. $\square$

The basic adjoint matching objective $\mathcal{L}\_{\text{basic-AM}}$ provides a theoretical foundation for the adjoint matching method but remains computationally inefficient as it requires differentiation through the cost functional, which depends on the full trajectory. Since we have shown that $\boldsymbol{u}^\star(\boldsymbol{x}, t)$ is the *unique minimizer*, it can be written as the conditional expectation of the regression target:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = \mathbb{E}_{\boldsymbol{X}_{t:T} \sim \mathbb{P}^{u^\star}}\left[-\sigma_t \boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \boldsymbol{u}^\star) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

At optimality, the forward control and adjoint state should be balanced everywhere, so we can multiply both sides of the equation by the Jacobian $\nabla\boldsymbol{u}(\boldsymbol{x}, t) \in \mathbb{R}^{d \times d}$ to get:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t)^\top \nabla\boldsymbol{u}^\star(\boldsymbol{x}, t) = \mathbb{E}_{\boldsymbol{X}_{t:T} \sim \mathbb{P}^{u^\star}}\left[-\sigma_t \boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \boldsymbol{u}^\star)^\top \nabla\boldsymbol{u}^\star(\boldsymbol{x}, t) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

Since the left hand side depends only on $\boldsymbol{x}$, we can rearrange to get:

$$
\mathbb{E}_{\boldsymbol{X}_{t:T}}\left[\boldsymbol{u}^\star(\boldsymbol{x}, t)^\top \nabla\boldsymbol{u}^\star(\boldsymbol{x}, t) + \sigma_t \boldsymbol{a}(\boldsymbol{X}_{t:T}, t; \boldsymbol{u}^\star)^\top \nabla\boldsymbol{u}(\boldsymbol{x}, t) \mid \boldsymbol{X}_t = \boldsymbol{x}\right] = 0
$$

which shows that *at optimality*, both terms that depend on the control $\boldsymbol{u}(\boldsymbol{x}, t)$ from the adjoint derivative evaluate to zero. Leveraging this, we introduce the **lean adjoint state**, which drops the $\boldsymbol{u}$-dependent terms to obtain a computationally more efficient objective.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.8</span><span class="math-callout__name">(Lean Adjoint State)</span></p>

The **lean adjoint state** $\tilde{\boldsymbol{a}} : C([t, T], \mathbb{R}^d) \times [0, T] \to \mathbb{R}^d$ is defined by the following differential equation which can be solved backward in time, given the terminal condition $\tilde{\boldsymbol{a}}(\boldsymbol{X}, T) = \nabla\Phi(\boldsymbol{X}\_T)$ as:

$$
\frac{d}{dt}\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) = -\left[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t)^\top \nabla\boldsymbol{f}(\boldsymbol{X}_t, t) + \nabla c(\boldsymbol{X}_t, t)\right], \quad \tilde{\boldsymbol{a}}(\boldsymbol{X}, T) = \nabla\Phi(\boldsymbol{X}_T)
$$

Unlike the **adjoint state** defined in Definition 6.5, the lean adjoint state $\tilde{\boldsymbol{a}}$ does not depend on the control $\boldsymbol{u}$ and does not require computing the Jacobian $\nabla\boldsymbol{u}(\boldsymbol{x}, t)$.

$$
\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) = \int_t^T \left(\nabla\boldsymbol{f}(\boldsymbol{X}_s, s)^\top \tilde{\boldsymbol{a}}(\boldsymbol{X}_{s:T}, s) + \nabla\boldsymbol{f}(\boldsymbol{X}_s, s)\right) ds + \nabla c(\boldsymbol{X}_T)
$$

</div>

Using this lean adjoint state, we can construct an objective that directly matches the optimal control without requiring explicit computation of the value function or its gradients.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.9</span><span class="math-callout__name">(Lean Adjoint Matching Yields the Optimal Control)</span></p>

Consider the **lean adjoint matching objective** defined with the lean adjoint state $\tilde{\boldsymbol{a}} : C([t, T], \mathbb{R}^d) \times [0, T] \to \mathbb{R}^d$ as:

$$
\mathcal{L}_{\text{AM}}(\boldsymbol{u}) := \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^{\bar{u}}}\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t, t) + \sigma_t \tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) \rVert^2 dt\right]
$$

$$
\bar{\boldsymbol{u}} := \text{stopgrad}(\boldsymbol{u})
$$

where $\bar{\boldsymbol{u}} := \text{stopgrad}(\boldsymbol{u})$ is the non-gradient tracking control drift. Then, $\mathcal{L}\_{\text{AM}}(\boldsymbol{u})$ has a unique minimizer which is exactly the optimal control $\boldsymbol{u}^\star$.

</div>

*Proof.* To prove this, we first establish the form of *some* critical point $\hat{\boldsymbol{u}}$ of $\mathcal{L}\_{\text{AM}}(\boldsymbol{u})$ and show that it is also a critical point of the **basic adjoint matching** loss $\mathcal{L}\_{\text{basic-AM}}$. Then, we can apply the result from Proposition 6.7 to conclude that $\hat{\boldsymbol{u}}$ is unique and is equal to the optimal control $\hat{\boldsymbol{u}} = \boldsymbol{u}^\star$.

**Step 1: Derive the Critical Point.** Since the form of the Lean AM Objective depends on a random variable $\boldsymbol{X}\_{0:T}$, to take the functional derivative, we need to evaluate it for some deterministic state $\boldsymbol{X}\_t = \boldsymbol{x}$. Since $\tilde{\boldsymbol{a}}(\boldsymbol{X}\_{t:T}, t)$ is the only term in $\mathcal{L}\_{\text{AM}}$ that contains randomness in $\boldsymbol{X}\_{t:T} \sim \mathbb{P}^{\bar{u}}$ after fixing $\boldsymbol{X}\_t = \boldsymbol{x}$, we add and subtract the conditional expectation $\mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}\_{t:T}, t) \mid \boldsymbol{X}\_t]$ to the expression inside the square in $\mathcal{L}\_{\text{AM}}$. Then, substituting this expression back into the Lean AM Objective, we have:

$$
\mathcal{L}_{\text{AM}}(\boldsymbol{u}) = \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^{\bar{u}}}\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t, t) + \sigma_t \mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) \mid \boldsymbol{X}_t] \rVert^2 dt\right] + \underbrace{\mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^{\bar{u}}}\left[\frac{1}{2}\int_0^T \lVert \sigma_t(\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}) - \mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) \mid \boldsymbol{X}_t]) \rVert^2 dt\right]}_{\text{not dependent on } \boldsymbol{u}}
$$

Then, computing the functional derivative $\frac{\delta}{\delta u}\mathcal{L}\_{\text{AM}}(\boldsymbol{u})(\boldsymbol{x}, t)$ evaluated at $\boldsymbol{X}\_t = \boldsymbol{x}$, we get:

$$
\frac{\delta}{\delta u}\mathcal{L}_{\text{AM}}(\boldsymbol{u})(\boldsymbol{x}, t) = \boldsymbol{u}(\boldsymbol{x}, t) + \sigma_t \mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) \mid \boldsymbol{X}_t = \boldsymbol{x}]
$$

Since the first variation of $\frac{\delta}{\delta u}\mathcal{L}\_{\text{AM}}(\hat{\boldsymbol{u}})$ of critical points $\hat{\boldsymbol{u}}$ is zero, we get that the critical points of $\mathcal{L}\_{\text{AM}}(\hat{\boldsymbol{u}})$ satisfy:

$$
\forall \boldsymbol{x} \in \mathbb{R}^d, \quad \frac{\delta}{\delta u}\mathcal{L}_{\text{AM}}(\hat{\boldsymbol{u}})(\boldsymbol{x}, t) = 0 \implies \hat{\boldsymbol{u}}(\boldsymbol{x}, t) = -\sigma_t \mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) \mid \boldsymbol{X}_t = \boldsymbol{x}]
$$

which we will show is also a critical point for the **basic adjoint matching** objective $\mathcal{L}\_{\text{basic-AM}}$.

**Step 2: Matching Critical Point to Basic Adjoint Matching Objective.** Using the same observation for obtaining the lean adjoint state, we note that at optimality, the forward control and adjoint state can be balanced which allows us to multiply both sides of the critical point condition with the Jacobian $\nabla\hat{\boldsymbol{u}}(\boldsymbol{x}, t)$ to get:

$$
\nabla\hat{\boldsymbol{u}}(\boldsymbol{X}_t, t)^\top \hat{\boldsymbol{u}}(\boldsymbol{X}_t, t) = -\sigma_t \nabla\hat{\boldsymbol{u}}(\boldsymbol{X}_t, t)^\top \mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) \mid \boldsymbol{X}_t]
$$

$$
\implies \mathbb{E}\left[\int_t^T \left(\sigma_s \nabla\hat{\boldsymbol{u}}(\boldsymbol{X}_s, s)^\top \mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{s:T}, s) \mid \boldsymbol{X}_s] + \nabla\left(\frac{1}{2}\lVert \hat{\boldsymbol{u}}(\boldsymbol{X}_s, s) \rVert^2\right)\right) ds \;\middle|\; \boldsymbol{X}_t\right] = 0
$$

Then, adding this zero-expectation balancing condition to the conditional expectation of the lean adjoint state $\mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}\_{t:T}, t) \mid \boldsymbol{X}\_t]$ using the definition of $\tilde{\boldsymbol{a}}$, we get:

$$
\mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) \mid \boldsymbol{X}_t] = \mathbb{E}\left[\int_t^T \left(\nabla(\boldsymbol{f}(\boldsymbol{X}_s, s) + \sigma_s \hat{\boldsymbol{u}}(\boldsymbol{X}_s, s))^\top \mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{s:T}, s) \mid \boldsymbol{X}_s] + \nabla\boldsymbol{f}(\boldsymbol{X}_s, s) + \nabla\left(c(\boldsymbol{X}_T) + \frac{1}{2}\lVert \hat{\boldsymbol{u}}(\boldsymbol{X}_s, s) \rVert^2\right)\right) ds \;\middle|\; \boldsymbol{X}_t\right]
$$

Since the **adjoint state** $\boldsymbol{a}(\boldsymbol{X}\_{t:T}, t; \boldsymbol{u})$ also solves an equivalent integral equation with arbitrary $\boldsymbol{v}$, setting $\boldsymbol{v} := \hat{\boldsymbol{u}}$ yields $\mathbb{E}[\boldsymbol{a}(\boldsymbol{X}\_{t:T}, t; \hat{\boldsymbol{u}}) \mid \boldsymbol{X}\_t] = \mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}\_{t:T}, t) \mid \boldsymbol{X}\_t]$ for all $t \in [0, T]$. Substituting this equality into the functional derivative of the basic adjoint matching loss, we get:

$$
\frac{\delta}{\delta u}\mathcal{L}_{\text{basic-AM}}(\hat{\boldsymbol{u}})(\boldsymbol{x}, t) = \hat{\boldsymbol{u}}(\boldsymbol{x}, t) + \sigma_t \mathbb{E}[\tilde{\boldsymbol{a}}(\boldsymbol{X}_{t:T}, t) \mid \boldsymbol{X}_t] = 0
$$

which implies that all critical points $\hat{\boldsymbol{u}}$ of $\mathcal{L}\_{\text{AM}}$ are critical points of $\mathcal{L}\_{\text{basic-AM}}$, and by Proposition 6.7, we conclude that $\hat{\boldsymbol{u}}$ is **unique** and **equal to the optimal control** $\hat{\boldsymbol{u}} = \boldsymbol{u}^\star$. $\square$

#### Simplification for Brownian Reference and the Adjoint Schroedinger Bridge Sampler

Given the general form of the adjoint matching objective, we can further simplify it for the case where the reference process is pure Brownian motion ($\boldsymbol{f} := 0$) and the running cost is zero ($c := 0$), which results in the backward time evolution $\frac{d}{dt}\tilde{\boldsymbol{a}}$ to *vanish*. Then, the **lean adjoint state** for all $t \in [0, T]$ reduces to the gradient of the terminal cost with respect to the current state given by $\tilde{\boldsymbol{a}}(\boldsymbol{X}\_{t:T}, t) = \nabla\_{\boldsymbol{x}\_T}\Phi(\boldsymbol{X}\_T)$. Therefore, the **simplified adjoint matching** objective becomes:

$$
\mathcal{L}_{\text{simple-AM}}(\boldsymbol{u}) := \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^{\bar{u}}}\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t, t) + \sigma_t \nabla_{\boldsymbol{x}_T}\Phi(\boldsymbol{X}_T) \rVert^2 dt\right], \quad \bar{\boldsymbol{u}} = \text{stopgrad}(\boldsymbol{u})
$$

To further simplify the objective, we introduce the **reciprocal adjoint matching** objective, where rather than taking the expectation over full trajectories $\boldsymbol{X}\_{0:T} \sim \mathbb{P}^{\bar{u}}$ by repeatedly simulating the controlled SDE, we can leverage the **key property** that given $\boldsymbol{f} := 0$, $c := 0$, and $\boldsymbol{X}\_0 = 0$, the Simplified AM Objective depends only on $(\boldsymbol{X}\_t, \boldsymbol{X}\_T)$ and the joint distribution of $(\boldsymbol{X}\_t, \boldsymbol{X}\_T)$ under the optimal Schroedinger bridge measure $\mathbb{P}^\star$ be factorized as:

$$
\mathbb{P}^\star(\boldsymbol{X}_t, \boldsymbol{X}_T) = \pi_T(\boldsymbol{X}_T)\mathbb{P}_{t|T}^\star(\boldsymbol{X}_t \mid \boldsymbol{X}_T) = \pi_T(\boldsymbol{X}_T)\mathbb{Q}_{t|T}(\boldsymbol{X}_t \mid \boldsymbol{X}_T)
$$

since $\mathbb{P}^\star$ is in the reciprocal class $\mathcal{R}(\mathbb{Q})$, which shares the conditional bridge $\mathbb{P}\_{t|T}^\star(\boldsymbol{X}\_t \mid \boldsymbol{X}\_T) = \mathbb{Q}\_{t|T}(\boldsymbol{X}\_t \mid \boldsymbol{X}\_T)$ to the reference process. This gives us the objective:

$$
\mathcal{L}_{\text{RAM}}(\boldsymbol{u}) := \mathbb{E}_{\boldsymbol{X}_t \sim \mathbb{Q}_{t|T}^{\bar{u}}, \boldsymbol{X}_T \sim p_T^{\bar{u}}}\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t, t) + \sigma_t \nabla_{\boldsymbol{x}_T}\Phi(\boldsymbol{X}_T) \rVert^2 dt\right], \quad \bar{\boldsymbol{u}} = \text{stopgrad}(\boldsymbol{u})
$$

Crucially, the samples $\boldsymbol{X}\_t \sim \mathbb{Q}\_{t|T}$ are **independent** conditioned on a terminal state $\boldsymbol{X}\_T \sim p\_T^{\bar{u}}$, enabling training on arbitrarily many intermediate samples given a single terminal state $\boldsymbol{X}\_T$. To minimize this objective, we propose an efficient **iterative two-step algorithm** called **adjoint sampling**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Adjoint Sampling)</span></p>

The **adjoint sampling** algorithm is an iterative two step procedure:

- **(i)** First, simulate the forward-time controlled SDE using the non-gradient tracking control $\bar{\boldsymbol{u}}$ to obtain a fixed set of samples $\boldsymbol{X}\_T \sim p\_T^{\bar{u}}$ and evaluate the gradient of their terminal cost $\nabla\_{\boldsymbol{x}\_T}\Phi(\boldsymbol{X}\_T)$. Then, store the set of samples in a replay buffer $\mathcal{B} = \lbrace \boldsymbol{X}\_T^i, \nabla\_{\boldsymbol{x}\_T}g(\boldsymbol{X}\_T^i) \rbrace\_{i=1}^B$.
- **(ii)** Optimize the Reciprocal AM Loss by repeatedly sampling intermediate states $\boldsymbol{X}\_t \sim \mathbb{Q}\_{t|T}$ conditioned on samples from the replay buffer $\mathcal{B}$.

This process repeats until the optimal $\boldsymbol{u}^\star$ is reached.

</div>

Intuitively, optimizing an arbitrary control $\boldsymbol{v}$ with the Reciprocal AM Loss can be considered a **reciprocal projection** (Proposition 4.13), where we constrain the endpoint as $p\_T^{\bar{u}}(\boldsymbol{X}\_T)$ *and* a **Markovian projection** (Proposition 4.10) by projecting on the Markov bridge measure $\mathbb{Q}\_{t|T}(\boldsymbol{X}\_t \mid \boldsymbol{X}\_T)$, where $\mathbb{Q}$ is the Markov reference Brownian motion.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.10</span><span class="math-callout__name">(Convergence of Adjoint Sampling)</span></p>

Consider optimizing the Reciprocal AM Loss $\mathcal{L}\_{\text{RAM}}$ given the current control $\bar{\boldsymbol{u}}$ as *projecting* an arbitrary control $\boldsymbol{v}$ onto the bridge $\mathbb{P}^{\bar{u}}(\boldsymbol{X}\_{0:T}) = p\_T^{\bar{u}}(\boldsymbol{X}\_T)\mathbb{Q}(\boldsymbol{X}\_{0:T})$ where $p\_T^{\bar{u}}$ is the target distribution:

$$
\text{proj}(\boldsymbol{u}) = \arg\min_{\boldsymbol{v}} \text{KL}\left(\mathbb{P}^v(\boldsymbol{X}_{0:T}) \| p_T^{\bar{u}}(\boldsymbol{X}_T)\mathbb{Q}(\boldsymbol{X}_{0:T})\right)
$$

Then, at each iteration, we obtain the update:

$$
\boldsymbol{u}^{n+1} = \text{proj}(\boldsymbol{u}^n) - \frac{\delta}{\delta u}\mathcal{L}_{\text{AM}}(\text{proj}(\boldsymbol{u}^n))
$$

The **unique** fixed point $\boldsymbol{u} = \text{proj}(\boldsymbol{u}) = \boldsymbol{u} - \frac{\delta}{\delta u}\mathcal{L}\_{\text{AM}}(\boldsymbol{u})$ is exactly the optimal control $\boldsymbol{u}^\star = \text{proj}(\boldsymbol{u}^\star)$.

</div>

*Proof.* We leverage the form of the critical point and the functional derivative.

**Step 1: Unifying the Reciprocal Adjoint and Lean Adjoint Matching Objectives.** First, we define a more general form of the reciprocal adjoint matching objective, where the matching target is an arbitrary vector field $\boldsymbol{v}$, and the current control is $\boldsymbol{u}$.

$$
\mathcal{L}(\boldsymbol{u}; \boldsymbol{v}) = \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^v}\left[\int_0^T \frac{1}{2}\lVert \boldsymbol{u}(\boldsymbol{X}_t, t) + \sigma_t \nabla_{\boldsymbol{x}_T}\Phi(\boldsymbol{X}_T) \rVert^2 dt\right]
$$

$$
\mathcal{L}_{\text{RAM}}(\boldsymbol{u}) := \mathcal{L}(\boldsymbol{u}; \text{proj}(\boldsymbol{u})), \quad \mathcal{L}_{\text{simple-AM}}(\boldsymbol{u}) := \mathcal{L}(\boldsymbol{u}; \boldsymbol{u})
$$

which are equivalent definitions of the Reciprocal AM Loss and the Simplified AM Objective given $\bar{\boldsymbol{u}} = \text{stopgrad}(\boldsymbol{u})$.

**Step 2: Derive Expression for Adjoint Sampling Iteration.** Given the relationship between the lean adjoint and reciprocal adjoint matching losses, we can apply the critical point condition to write an optimal iteration of minimizing $\mathcal{L}(\boldsymbol{u}; \text{proj}(\boldsymbol{u}^n))$ as satisfying:

$$
\boldsymbol{u}^{n+1}(\boldsymbol{x}, t) = -\sigma_t \mathbb{E}_{\mathbb{P}^{\text{proj}(\boldsymbol{u}^n)}}\left[\nabla_{\boldsymbol{x}_T}\Phi(\boldsymbol{X}_T) \mid \boldsymbol{X} = \boldsymbol{x}\right]
$$

Since the $\mathcal{L}\_{\text{RAM}}$ is just $\mathcal{L}\_{\text{AM}}$ with the target set to $\text{proj}(\boldsymbol{u}^n)$, we also apply the functional derivative to write:

$$
\boldsymbol{u}^{n+1}(\boldsymbol{x}, t) = \text{proj}(\boldsymbol{u}^n) - \frac{\delta}{\delta u}\mathcal{L}_{\text{AM}}(\text{proj}(\boldsymbol{u}^n))
$$

which concludes the proof of the update rule.

**Step 3: Fixed Point of Adjoint Sampling Iteration.** We will now show that $\boldsymbol{u}$ is a fixed point of adjoint sampling such that $\boldsymbol{u} = \text{proj}(\boldsymbol{u})$ if and only if $\boldsymbol{u}$ is a critical point of the Lean AM Objective, which implies that $\boldsymbol{u} = \boldsymbol{u}^\star$ by Proposition 6.9.

First, we show $\boldsymbol{u} = \text{proj}(\boldsymbol{u}) \implies \boldsymbol{u} = \boldsymbol{u}^\star$. Suppose $\boldsymbol{u} = \text{proj}(\boldsymbol{u})$. By the update rule, we have $\frac{\delta}{\delta u}\mathcal{L}\_{\text{AM}}(\text{proj}(\boldsymbol{u})) = 0$. This means that $\boldsymbol{u}$ is a critical point of $\mathcal{L}\_{\text{AM}}$ and by Proposition 6.9, we have that $\boldsymbol{u}$ is unique and equal to the optimal control $\boldsymbol{u} = \boldsymbol{u}^\star$.

Next, we show $\boldsymbol{u} = \boldsymbol{u}^\star \implies \boldsymbol{u} = \text{proj}(\boldsymbol{u})$. Suppose $\boldsymbol{u} = \boldsymbol{u}^\star$. Then, by Proposition 6.9, we know $\boldsymbol{u} = \boldsymbol{u}^\star$. Since by definition, $\boldsymbol{u}^\star$ generates the optimal target of the projection $\mathbb{P}^\star(\boldsymbol{X}\_{0:T}) = \pi\_T(\boldsymbol{X}\_T)\mathbb{Q}(\boldsymbol{X}\_{0:T})$, projecting onto $\mathbb{P}^\star$ would yield itself, so $\boldsymbol{u}^\star = \text{proj}(\boldsymbol{u}^\star)$. $\square$

Adjoint sampling provides a theoretically-grounded and computationally efficient method of obtaining the optimal control $\boldsymbol{u}^\star$ when the posterior under the reference dynamics can be easily sampled as $\boldsymbol{X}\_t \sim \mathbb{Q}\_{t|T}(\cdot \mid \boldsymbol{X}\_T)$, such as the linear Brownian motion case where $\boldsymbol{f} := 0$ and the initial distribution is a Dirac delta at zero $\pi\_0 := \delta\_0$. However, its restriction to the Dirac delta prior prevents more general settings with *informative priors*, such as Gaussians or task-specific priors.

Additionally, adjoint matching requires the reference process to be **memoryless**, such that the joint distribution of $(\boldsymbol{X}\_0, \boldsymbol{X}\_T)$ can be factorized as $\mathbb{Q}(\boldsymbol{X}\_0, \boldsymbol{X}\_T) = q\_0(\boldsymbol{X}\_0)q\_T(\boldsymbol{X}\_T)$. This is to prevent the **initial value function bias** described in Box 3.1, where the optimal joint distribution $\mathbb{P}^\star(\boldsymbol{X}\_0, \boldsymbol{X}\_T)$ derived in Proposition 3.9 is dependent on an intractable initial value function $V\_0(\boldsymbol{X}\_0)$:

$$
\mathbb{P}^\star(\boldsymbol{X}_0, \boldsymbol{X}_T) = \mathbb{Q}(\boldsymbol{X}_0, \boldsymbol{X}_T)e^{-\Phi(\boldsymbol{X}_T) + V_0(\boldsymbol{X}_0)}
$$

Although leveraging a memoryless reference drift guarantees sampling paths that generate the true target distribution $\pi\_T$, it prevents the use of more informative prior distributions for sampling complex target distributions and crucially excludes all tasks where the goal is to map between distributions rather than simply sampling from a target distribution. This motivates using the SB-SOC Objective introduced in Section 3.2, which leverages the SB potentials $(\varphi, \hat{\varphi})$ defined by:

$$
\boldsymbol{u}^\star(\boldsymbol{x}, t) = -\sigma_t \nabla\log\varphi_t(\boldsymbol{x}), \quad p_t^\star(\boldsymbol{x}) = \varphi_t(\boldsymbol{x})\hat{\varphi}_t(\boldsymbol{x})
$$

where $\varphi\_t(\boldsymbol{x}) = \int\_{\mathbb{R}^d} \mathbb{Q}\_{T|t}(\boldsymbol{y} \mid \boldsymbol{x})\varphi\_T(\boldsymbol{y})d\boldsymbol{y}$, $\pi\_0 = \varphi\_0 \hat{\varphi}\_0$, and $\hat{\varphi}\_t(\boldsymbol{x}) = \int\_{\mathbb{R}^d} \mathbb{Q}\_{t|0}(\boldsymbol{x} \mid \boldsymbol{y})\hat{\varphi}\_0(\boldsymbol{y})d\boldsymbol{y}$, $\pi\_T = \varphi\_T \hat{\varphi}\_T$.

Using these equations, we can replace the terminal cost in $\mathcal{L}\_{\text{simple-AM}}$ with the definition in the SB Terminal Cost given by $\nabla\Phi(\boldsymbol{x}) = \nabla\log\frac{\hat{\varphi}\_T(\boldsymbol{x})}{\pi\_T(\boldsymbol{x})} = \nabla\log\hat{\varphi}\_T(\boldsymbol{x}) - \nabla\log\pi\_T(\boldsymbol{x})$ to get the **Schroedinger bridge adjoint matching** (SB-AM) loss.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.11</span><span class="math-callout__name">(Schroedinger Bridge Adjoint Matching Objective)</span></p>

The **Schroedinger bridge adjoint matching** (SB-AM) loss which solves the SB-SOC Objective is defined as:

$$
\mathcal{L}_{\text{SB-AM}}(\boldsymbol{u}) := \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^{\bar{u}}}\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t, t) + \sigma_t(\nabla_{\boldsymbol{x}_T}\log\hat{\varphi}_T(\boldsymbol{X}_T) - \nabla_{\boldsymbol{x}_T}\log\pi_T(\boldsymbol{X}_T)) \rVert^2 dt\right]
$$

where $\bar{\boldsymbol{u}} = \text{stopgrad}(\boldsymbol{u})$ is the non-gradient-tracking control drift.

</div>

To tractably compute $\nabla\_{\boldsymbol{x}\_T}\log\hat{\varphi}\_T(\boldsymbol{x}\_T)$, we define the **bridge-matching or corrector matching objective** (CM), which has been applied to both data-driven and sampling problems.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.12</span><span class="math-callout__name">(Corrector Matching Objective)</span></p>

The gradient of the log Schroedinger bridge potential $\nabla\log\hat{\varphi}\_t(\boldsymbol{x})$ can be expressed as the minimizer of the **Schroedinger bridge corrector matching** (SB-CM) loss defined as:

$$
\mathcal{L}_{\text{SB-CM}}(\widehat{\boldsymbol{Z}}_T) := \mathbb{E}_{p_{0,T}^\star}\left[\lVert \widehat{\boldsymbol{Z}}_T(\boldsymbol{X}_T) - \nabla_{\boldsymbol{x}_T}\log\mathbb{Q}_{T|0}(\boldsymbol{X}_T \mid \boldsymbol{X}_0) \rVert^2\right]
$$

where the minimizer defines the backward gradient of the log potential:

$$
\nabla\log\hat{\varphi}_T(\boldsymbol{x}) = \widehat{\boldsymbol{Z}}_T^\star = \arg\min_{\widehat{\boldsymbol{Z}}_T} \mathbb{E}_{p_{0,T}^\star}\left[\lVert \widehat{\boldsymbol{Z}}_T(\boldsymbol{X}_T) - \nabla\log\mathbb{Q}_{T|0}(\boldsymbol{X}_T \mid \boldsymbol{X}_0) \rVert^2\right]
$$

</div>

*Proof.* Starting with the definition of $\hat{\varphi}\_t$ from the SB Optimality, we have:

$$
\nabla\log\hat{\varphi}_t(\boldsymbol{x}) = \frac{\nabla\hat{\varphi}_t(\boldsymbol{x})}{\hat{\varphi}_t(\boldsymbol{x})} = \frac{1}{\hat{\varphi}_t(\boldsymbol{x})}\nabla\left(\int_{\mathbb{R}^d} \mathbb{Q}_{t|0}(\boldsymbol{x} \mid \boldsymbol{y})\hat{\varphi}_0(\boldsymbol{y})d\boldsymbol{y}\right)
$$

$$
= \frac{1}{\hat{\varphi}_t(\boldsymbol{x})}\int_{\mathbb{R}^d} \nabla\mathbb{Q}_{t|0}(\boldsymbol{x} \mid \boldsymbol{y})\hat{\varphi}_0(\boldsymbol{y})d\boldsymbol{y} = \frac{\varphi_t(\boldsymbol{x})}{p_t^\star(\boldsymbol{x})}\int_{\mathbb{R}^d} \nabla\log\mathbb{Q}_{t|0}(\boldsymbol{x} \mid \boldsymbol{y})\mathbb{Q}_{t|0}(\boldsymbol{x} \mid \boldsymbol{y})\hat{\varphi}_0(\boldsymbol{y})d\boldsymbol{y}
$$

Recalling from the SB-SOC Joint Density that the joint density between any two timepoints $s \le t$ is given by $p\_{s,t}^\star(\boldsymbol{y}, \boldsymbol{x}) = \mathbb{Q}(\boldsymbol{X}\_s = \boldsymbol{y} \mid \boldsymbol{X}\_t = \boldsymbol{x})\hat{\varphi}\_s(\boldsymbol{y})\varphi\_t(\boldsymbol{x})$, we observe that we can replace the highlighted terms to get:

$$
\nabla\log\hat{\varphi}_t(\boldsymbol{x}) = \int_{\mathbb{R}^d} \nabla\log\mathbb{Q}_{t|0}(\boldsymbol{x} \mid \boldsymbol{y}) \frac{p_{0,t}^\star(\boldsymbol{y}, \boldsymbol{x})}{p_t^\star(\boldsymbol{x})} d\boldsymbol{y} = \mathbb{E}_{p_{0|t}^\star}\left[\nabla\log\mathbb{Q}_{t|0}(\boldsymbol{X}_t \mid \boldsymbol{X}_0) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

This can be rewritten as a **regression objective** where pairs $(\boldsymbol{X}\_0, \boldsymbol{X}\_t) \sim p\_{0,t}^\star$ and we minimize a parameterized function $\widehat{\boldsymbol{Z}}\_t(\boldsymbol{X}\_t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ that minimizes the square loss. For $t = T$, we obtain the expression $\nabla\log\hat{\varphi}\_T(\boldsymbol{x})$ that appears in $\mathcal{L}\_{\text{SB-AM}}$:

$$
\widehat{\boldsymbol{Z}}_T^\star := \arg\min_{\widehat{\boldsymbol{Z}}_T} \mathbb{E}_{p_{0,T}^\star}\left[\lVert \widehat{\boldsymbol{Z}}_T(\boldsymbol{X}_T) - \nabla\log\mathbb{Q}_{T|0}(\boldsymbol{X}_T \mid \boldsymbol{X}_0) \rVert^2\right]
$$

which is the definition of the Corrector Matching Objective. $\square$

Although this provides a concrete variational objective for obtaining $\nabla\log\hat{\varphi}\_T(\boldsymbol{x})$ required to optimize the SB-AM Objective, it requires sampling $(\boldsymbol{X}\_0, \boldsymbol{X}\_T) \sim p\_{0,T}^\star$. In the case of sampling from an energy-based distribution, we have no explicit access to the target distribution $p\_T^\star$, which is exactly the challenge addressed by the **adjoint Schroedinger bridge sampler** algorithm. Rather than sampling pairs from the optimal joint distribution $p\_{0,T}^\star$, we optimize the corrector $\widehat{\boldsymbol{Z}}\_T$ using samples from the joint distribution $p\_{0,T}^{\bar{u}}$ generated with the frozen control $\bar{\boldsymbol{u}}$. However, optimizing $\boldsymbol{u}$ using $\mathcal{L}\_{\text{SB-AM}}$ also requires computing $\nabla\_{\boldsymbol{x}\_T}\log\hat{\varphi}\_T(\boldsymbol{X}\_T)$, which introduces cross-dependencies between the parameterized variables.

This naturally motivates an **alternating optimization scheme**, that switches between training $\boldsymbol{u}$ with the parameters of $\widehat{\boldsymbol{Z}}\_T$ fixed and optimizing $\widehat{\boldsymbol{Z}}\_T$ with the parameters of $\boldsymbol{u}$ fixed. Concretely, we can interpret this as optimizing a pair of **forward and backward SDEs** that are characterized by the control drift $\boldsymbol{u}$ in the forward time coordinate $t \in [0, T]$ and correction term $\nabla\log\hat{\varphi}\_t = \nabla\log\hat{\varphi}\_{T-s}$ in the backward time coordinate $s = T - t \in [0, T]$, respectively:

$$
\mathbb{P}^u : \quad d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t)) dt + \sigma_t d\boldsymbol{B}_t, \qquad \boldsymbol{X}_0 \sim \pi_0
$$

$$
\mathbb{P}^{\hat{Z}} : \quad d\bar{\boldsymbol{X}}_s = \left[-\boldsymbol{f}(\bar{\boldsymbol{X}}_s, s) + \sigma_s^2 \nabla\log\hat{\varphi}_{T-s}(\bar{\boldsymbol{X}}_s)\right] ds + \sigma_s d\bar{\boldsymbol{B}}_s, \qquad \bar{\boldsymbol{X}}_0 \sim \pi_T
$$

where $\hat{\varphi}\_{T-s}$ is defined in the forward time coordinate via the SB Optimality with the terminal constraint $\nabla\_{\boldsymbol{x}\_T}\log\hat{\varphi}\_T(\boldsymbol{x}\_T) = \widehat{\boldsymbol{Z}}\_T(\boldsymbol{x}\_T)$. Then, optimizing both the SB-AM Objective and Corrector Matching Objective reduces to determining the optimal pair $(\boldsymbol{u}^\star, \widehat{\boldsymbol{Z}}\_T^\star)$, where the path measures generated by both the forward and backward SDEs align with the Schroedinger bridge path $\mathbb{P}^{u^\star} = \mathbb{P}^{\hat{Z}^\star} = \mathbb{P}^\star$. Indeed, the alternating optimization scheme achieves this goal, which we will show by establishing that optimizing the SB-AM Objective generates the **optimal forward half-bridge** (Proposition 6.13) and that optimizing the Corrector Matching Objective generates the **optimal backward half-bridge with from the terminal constraint** (Proposition 6.14).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.13</span><span class="math-callout__name">(Adjoint Matching Solves the Forward Schroedinger Bridge)</span></p>

Consider optimizing the SB-AM Objective with respect to the control drift $\boldsymbol{u}^{(k)}$ at iteration $k$, then the path measure generated by the control $\mathbb{P}^{u^{(k)}}$ solves the **forward half bridge** defined by the initial marginal constraint $p\_0 = \pi\_0$ given by:

$$
\mathbb{P}^{u^{(k)}} = \arg\min_{\mathbb{P}^u} \left\lbrace \text{KL}(\mathbb{P}^u \| \mathbb{P}^{\hat{Z}}); \; \mathbb{P}_0^u = \pi_0 \right\rbrace
$$

where $\mathbb{P}^{\hat{Z}}$ is the path measure generated **time-reversed SDE** defined by the corrector $\widehat{\boldsymbol{Z}}\_T^{(k-1)}$.

</div>

*Proof.* The goal of this proof is to show that optimizing the control drift $\boldsymbol{u}^{(k)}$ using the corrector $\widehat{\boldsymbol{Z}}\_T^{(k-1)}$ from the previous iteration solves the **forward half-bridge that is closest in KL divergence to the reverse-time dynamics generated from** $\widehat{\boldsymbol{Z}}\_T^{(k-1)}$. To do this, we first define the forward-time dynamics corresponding to the reverse-time SDE induced by $\widehat{\boldsymbol{Z}}\_T^{(k-1)}$. Then, we apply Itô calculus to write the variational KL objective as an Itô integral which reduces the forward half-bridge matching objective to the SB-AM Objective.

**Step 1: Define the Forward SDE from the Backward Dynamics.** Since the corrector $h^{(k-1)}$ defines the reverse-time dynamics via the backward SDE following the time coordinate $s = T - t$, we can define the corresponding forward-time dynamics for $t$ using the Time Reversal Formula to get:

$$
\mathbb{P}^{\hat{Z}} : d\boldsymbol{X}_t = \left[\boldsymbol{f}(\boldsymbol{X}_t, t) \underbrace{-\sigma_t^2 \nabla\log\hat{\varphi}_t(\boldsymbol{X}_t) + \sigma_t^2 \nabla\log p_t^{\hat{Z}}(\boldsymbol{X}_t)}_{(\star)}\right] dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_T \sim \pi_T
$$

The SB-AM Objective aims to match $\mathbb{P}^{\hat{Z}}$ with a path measure induced by some control drift $\boldsymbol{u}$ with the forward-time SDE:

$$
\mathbb{P}^u : \quad d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \underbrace{\sigma_t \boldsymbol{u}(\boldsymbol{X}_t, t)}_{(\circ)}) dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0 \sim \pi_0
$$

**Step 2: Apply Itô's Calculus to Simplify Integral.** Matching the control drift $(\circ)$ to the target drift $(\star)$ can be expanded using the KL divergence derived via Girsanov's theorem in Section 2.6 as:

$$
\text{KL}(\mathbb{P}^u \| \mathbb{P}^{\hat{Z}}) = \mathbb{E}_{\boldsymbol{X}_{0:T}^u \sim \mathbb{P}^u}\left[\int_0^T \frac{1}{2}\left\lVert \boldsymbol{u}(\boldsymbol{X}_t^u, t) + \sigma_t\left(\nabla\log\hat{\varphi}_t(\boldsymbol{X}_t^u) - \nabla\log p_t^{\hat{Z}}(\boldsymbol{X}_t^u)\right)\right\rVert^2 dt\right]
$$

which depends on the integral of gradient terms $\nabla\log\hat{\varphi}\_t(\boldsymbol{X}\_t)$ and $\nabla\log p\_t^{\hat{Z}}(\boldsymbol{X}\_t)$. Since we know that the Itô integral evaluates to the difference between boundary conditions, we can rewrite the objective using Itô's calculus.

To expand the gradient terms, we apply Itô's Formula to the Itô processes defined by $\log\hat{\varphi}\_t(\boldsymbol{X}\_t^u)$ and $\log p\_t^{\hat{Z}}(\boldsymbol{X}\_t^u)$ which contains the desired gradient term. For $\log\hat{\varphi}\_t(\boldsymbol{X}\_t^u)$, we have:

$$
d\log\hat{\varphi}_t(\boldsymbol{X}_t^u) = \left[\partial_t\log\hat{\varphi}_t(\boldsymbol{X}_t^u) + (\boldsymbol{f} + \sigma_t \boldsymbol{u})(\boldsymbol{X}_t^u, t)^\top \nabla\log\hat{\varphi}_t(\boldsymbol{X}_t^u) + \frac{\sigma_t^2}{2}\Delta\log\hat{\varphi}_t(\boldsymbol{X}_t^u)\right] dt + \sigma_t \nabla\log\hat{\varphi}_t(\boldsymbol{X}_t^u) d\boldsymbol{B}_t
$$

To derive the expression for $\partial\_t\log\hat{\varphi}\_t$, we recall that it follows a deterministic integral $\hat{\varphi}\_t(\boldsymbol{x}) = \int\_{\mathbb{R}^d} \mathbb{Q}\_{t|0}(\boldsymbol{x} \mid \boldsymbol{y})\hat{\varphi}\_0(\boldsymbol{y})d\boldsymbol{y}$ with terminal condition $\hat{\varphi}\_T = \hat{z}\_T$. From Section 3.2, any function applied to a Markov process defined by a terminal constraint satisfies the Feynman-Kac Formula. Given the terminal constraint $\hat{\varphi}\_T(\boldsymbol{x}) = \hat{z}\_T(\boldsymbol{x})$, the Feynman-Kac Formula for $\hat{\varphi}\_t$ is given by:

$$
\partial_t\hat{\varphi}_t(\boldsymbol{x}) = -\nabla \cdot (\boldsymbol{f}(\boldsymbol{x}, t)\hat{\varphi}_t(\boldsymbol{x})) + \frac{\sigma_t^2}{2}\Delta\hat{\varphi}_t(\boldsymbol{x}), \quad \hat{\varphi}_T(\boldsymbol{x}) = \hat{z}_T(\boldsymbol{x})
$$

To obtain $\partial\_t\log\hat{\varphi}\_t$, we apply the chain rule and the divergence property $\nabla \cdot (\boldsymbol{f}\hat{\varphi}\_t) = (\nabla \cdot \boldsymbol{f})\hat{\varphi}\_t + \boldsymbol{f}^\top \nabla\hat{\varphi}\_t$ to get:

$$
\partial_t\log\hat{\varphi}_t = -\nabla \cdot \boldsymbol{f} - \boldsymbol{f}^\top \nabla\log\hat{\varphi}_t + \frac{\sigma_t^2}{2}(\Delta\log\hat{\varphi}_t + \lVert \nabla\log\hat{\varphi}_t \rVert^2)
$$

where the final equality is obtained from applying the Laplacian trick $\Delta\hat{\varphi}\_t = \hat{\varphi}\_t(\lVert \nabla\log\hat{\varphi}\_t \rVert^2 + \Delta\log\hat{\varphi}\_t)$.

Next, we apply Itô's formula to $\partial\_t\log p\_t^{\hat{Z}}(\boldsymbol{X}\_t^u)$. In this case, since $p\_t^{\hat{Z}}$ is defined by the forward-time SDE of $\mathbb{P}^{\hat{Z}}$, we can apply the Fokker-Planck Equation to write the time-evolution of the density $\partial\_t p\_t^{\hat{Z}}$ as:

$$
\partial_t p_t^{\hat{Z}} = -\nabla \cdot ((\boldsymbol{f} - \sigma_t^2 \nabla\log\hat{\varphi}_t) p_t^{\hat{Z}}) - \frac{\sigma_t^2}{2}\Delta p_t^{\hat{Z}}
$$

Then, applying the chain rule to express $\partial\_t\log p\_t^{\hat{Z}}$:

$$
\partial_t\log p_t^{\hat{Z}} = -\nabla \cdot \boldsymbol{f} + \sigma_t^2 \Delta\log\hat{\varphi}_t - (\boldsymbol{f} - \sigma_t^2 \nabla\log\hat{\varphi}_t)^\top \nabla\log p_t^{\hat{Z}} - \frac{\sigma_t^2}{2}(\Delta\log p_t^{\hat{Z}} + \lVert \nabla\log p_t^{\hat{Z}} \rVert^2)
$$

Observing that the two Itô SDEs for $\log\hat{\varphi}$ and $\log p\_t^{\hat{Z}}$ have several matching terms, we can cancel them by subtracting $d\log p\_t^{\hat{Z}}$ from $d\log\hat{\varphi}\_t$ to get:

$$
d\log\hat{\varphi}_t - d\log p_t^{\hat{Z}} = \left[(\sigma_t \boldsymbol{u})^\top(\nabla\log\hat{\varphi}_t - \nabla\log p_t^{\hat{Z}}) + \frac{\sigma_t^2}{2}\lVert \nabla\log\hat{\varphi}_t \rVert^2 + \frac{\sigma_t^2}{2}\lVert \nabla\log p_t^{\hat{Z}} \rVert^2 + \sigma_t^2 \nabla\log\hat{\varphi}_t^\top \nabla\log p_t^{\hat{Z}}\right] dt + \sigma_t \nabla\log\frac{\hat{\varphi}_t}{p_t^{\hat{Z}}} d\boldsymbol{B}_t
$$

where the terms inside the bracket are almost a perfect square of $\boldsymbol{u} + \nabla\log\hat{\varphi}\_t - \nabla\log p\_t^{\hat{Z}}$. Completing the square with $\frac{1}{2}\lVert \boldsymbol{u} \rVert^2$, we get:

$$
\left[\frac{1}{2}\lVert \boldsymbol{u} + \nabla\log\hat{\varphi}_t - \nabla\log p_t^{\hat{Z}} \rVert^2\right] dt = \frac{1}{2}\lVert \boldsymbol{u} \rVert^2 dt + d\log\hat{\varphi}_t - d\log p_t^{\hat{Z}} - \sigma_t \nabla\log\frac{\hat{\varphi}_t}{p_t^{\hat{Z}}} d\boldsymbol{B}_t
$$

which recovers the integrand from our KL divergence objective.

**Step 3: Rewriting the KL Divergence.** Using this identity, we can rewrite the KL divergence objective as:

$$
\text{KL}(\mathbb{P}^u \| \mathbb{P}^{\hat{Z}}) = \mathbb{E}_{\boldsymbol{X}_{0:T}^u \sim \mathbb{P}^u}\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t^u, t) \rVert^2 dt + \underbrace{\int_0^T d\log\hat{\varphi}_t(\boldsymbol{X}_t^u)}_{\log\hat{\varphi}_T - \log\hat{\varphi}_0} - \underbrace{\int_0^T d\log p_t^{\hat{Z}}(\boldsymbol{X}_t^u)}_{\log p_T^{\hat{Z}} - \log p_0^{\hat{Z}}}\right]
$$

$$
= \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^u}\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t, t) \rVert^2 dt + \log\frac{\hat{\varphi}_T(\boldsymbol{X}_T^u)}{p_T^{\hat{Z}}(\boldsymbol{X}_T^u)} - \underbrace{\log\frac{\hat{\varphi}_0(\boldsymbol{X}_0^u)}{p_0^{\hat{Z}}(\boldsymbol{X}_0^u)}}_{\text{constant}}\right]
$$

where the term dependent on $\boldsymbol{X}\_0^u$ is a constant with respect to $\boldsymbol{u}$ since we fix the initial distribution at $\boldsymbol{X}\_0 \sim \pi\_0$. Therefore, substituting $\hat{\varphi}\_T = \hat{z}\_T^{(k-1)}$ into the objective, we have:

$$
\text{KL}(\mathbb{P}^u \| \mathbb{P}^{\hat{Z}}) = \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^u}\left[\frac{1}{2}\int_0^T \lVert \boldsymbol{u}(\boldsymbol{X}_t, t) \rVert^2 dt + \log\frac{\hat{z}_T^{(k-1)}(\boldsymbol{X}_T)}{\pi_T(\boldsymbol{X}_T)} + \text{const}\right]
$$

which is proportional to the SB-AM Objective up to an additive constant, and we conclude that the control $\boldsymbol{u}^{(k)}$ obtained from minimizing the SB-AM Objective solves the forward half-bridge that minimizes $\text{KL}(\mathbb{P}^u \| \mathbb{P}^{\hat{Z}})$. $\square$

However, we have already shown in Section 3.1 that solving the forward half-bridge for arbitrary prior distributions and reference drifts results in a mismatch of the target distribution, which motivated the definition of the Corrector Matching Objective to define the terminal cost as $\Phi(\boldsymbol{x}) := \log\frac{\hat{\varphi}\_T(\boldsymbol{x})}{\pi\_T(\boldsymbol{x})}$. While this definition of the terminal cost provably eliminates the initial value bias as shown in Section 3.2, we can further show that by optimizing the Corrector Matching Objective, we obtain the optimal corrector $\widehat{\boldsymbol{Z}}\_T^{(k)}$ that induces the backward half-bridge which is closest in KL to the forward dynamics defined by any arbitrary control $\boldsymbol{u}$ while correcting for the bias created at the terminal distribution $\pi\_T$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.14</span><span class="math-callout__name">(Corrector Matching Solves the Backward Schroedinger Bridge)</span></p>

Consider optimizing the Corrector Matching Objective with the control drift $\boldsymbol{u}^{(k)}$ from the $k$th iteration, then the path measure generated by the corrector $\widehat{\boldsymbol{Z}}\_T^{(k)}$ solves the backward half bridge defined by the terminal constraint $p\_T = \pi\_T$ as:

$$
\mathbb{P}^{\hat{Z}} = \arg\min_{\mathbb{P}^{\hat{Z}}} \left\lbrace \text{KL}(\mathbb{P}^{u^{(k)}} \| \mathbb{P}) : p_T^{\hat{Z}} = \pi_T \right\rbrace
$$

where $\mathbb{P}^{u^{(k)}}$ is the path measure generated from the learned control at iteration $k$.

</div>

*Proof.* This proof starts by defining the time-reversal of the forward half-bridge generated by $\boldsymbol{u}^{(k)}$ from the $k$th iteration of the algorithm and showing that optimizing the Corrector Matching Objective yields the optimal reverse-time dynamics that enforce the terminal constraint.

**Step 1: Time Reversal of Forward Controlled SDE.** Given the control drift from the current iteration of the optimization algorithm $\boldsymbol{u}^{(k)}$, we can define the corresponding backward SDE following the reversed time coordinate $s := T - t$ using the Time Reversal Formula to get:

$$
\mathbb{P}^{u^{(k)}} : \quad d\bar{\boldsymbol{X}}_s = \left(-\boldsymbol{f}(\bar{\boldsymbol{X}}_s, s) - \sigma_s \boldsymbol{u}_t^{(k)}(\bar{\boldsymbol{X}}_s, s) + \sigma_s^2 \nabla\log p_s^{u^{(k)}}(\bar{\boldsymbol{X}}_s)\right) ds + \sigma_s d\bar{\boldsymbol{B}}_s, \quad \bar{\boldsymbol{X}}_0 \sim \pi_T
$$

where the highlighted terms are defined by the control $\boldsymbol{u}^{(k)}$. Since we have shown that $\boldsymbol{u}^{(k)}$ solves the forward half-bridge in Proposition 6.13, it satisfies the SB equations given by:

$$
\boldsymbol{u}_t^{(k)}(\boldsymbol{x}, t) = \sigma_t \nabla\log\varphi_t(\boldsymbol{x}), \quad p_T^{u^{(k)}}(\boldsymbol{x}) = \varphi_T(\boldsymbol{x})\hat{\varphi}_T(\boldsymbol{x})
$$

which does not necessarily satisfy the terminal constraint $p\_T^\star = \pi\_T$. The goal of the Corrector Matching Objective is to generate the backward half-bridge that minimizes the divergence from the forward half-bridge while constraining the terminal marginal to $p\_T^{\hat{Z}} = \pi\_T$.

**Step 2: Deriving the Matching Objective.** To do this, we aim to match some arbitrary control $\boldsymbol{v}(\bar{\boldsymbol{X}}\_s, s)$ to the time-reversal of the forward half-bridge where the initial states $\bar{\boldsymbol{X}}\_0 \sim \pi\_T$ are sampled from the target marginal $\pi\_T$. To define the matching loss as a KL divergence, let $\boldsymbol{v}(\boldsymbol{x}, s)$ be an arbitrary control drift defining the reverse-time SDE initialized at $\pi\_T$:

$$
\mathbb{P}^v : d\bar{\boldsymbol{X}}_s = (-\boldsymbol{f}(\bar{\boldsymbol{X}}_s, s) + \sigma_s \boldsymbol{v}(\bar{\boldsymbol{X}}_s, s)) ds + \sigma_s d\bar{\boldsymbol{B}}_s, \quad \bar{\boldsymbol{X}}_0 \sim \pi_T
$$

Expanding the KL divergence $\text{KL}(\mathbb{P}^{u^{(k)}} \| \mathbb{P}^v)$ as shown in Section 2.6, we have:

$$
\text{KL}(\mathbb{P}^{u^{(k)}} \| \mathbb{P}^v) = \mathbb{E}_{\bar{\boldsymbol{X}}_{0:T} \sim \mathbb{P}^{u^{(k)}}}\left[\int_0^T \frac{1}{2}\lVert (-\sigma_s \nabla\log\varphi_s(\bar{\boldsymbol{X}}_s) + \sigma_s \nabla\log p_s^{u^{(k)}}(\bar{\boldsymbol{X}}_s)) - \boldsymbol{v}(\bar{\boldsymbol{X}}_s, s) \rVert^2 ds\right]
$$

Minimizing yields for all $(\boldsymbol{x}, s)$, the following expression for $\boldsymbol{v}(\boldsymbol{x}, s)$:

$$
\boldsymbol{v}^\star(\boldsymbol{x}, s) = -\sigma_s \nabla\log\varphi_{T-s}(\boldsymbol{x}) + \sigma_s \nabla\log p_s^{u^{(k)}}(\boldsymbol{x}) = \sigma_s \nabla\log\frac{\varphi_s(\boldsymbol{x})}{p_s^{u^{(k)}}(\boldsymbol{x})} = \sigma_s \nabla\log\hat{\varphi}_{T-s}(\boldsymbol{x})
$$

where we use the SB Optimality which defines $p\_t^{u^{(k)}}(\boldsymbol{x}) = \varphi\_t(\boldsymbol{x})\hat{\varphi}\_t(\boldsymbol{x})$. Substituting this expression into the backward path measure $\mathbb{P}^v$, we have the **optimal half-bridge is generated by** the SDE:

$$
d\bar{\boldsymbol{X}}_s = (-\boldsymbol{f}(\bar{\boldsymbol{X}}_s, s) + \sigma_s^2 \nabla\log\hat{\varphi}_{T-s}(\bar{\boldsymbol{X}}_s)) ds + \sigma_s d\bar{\boldsymbol{B}}_s, \quad \hat{\varphi}_T = \frac{p_T^{u^{(k)}}}{\varphi_T}
$$

which is *fully characterized* by the terminal condition $\hat{\varphi}\_T = \frac{p\_T^{u^{(k)}}}{\varphi\_T}$ from which $\hat{\varphi}\_{T-s}$ can be defined using the SB Optimality. From Proposition 6.12, we show that optimizing the Corrector Matching Objective yields $\widehat{\boldsymbol{Z}}\_T^{(k)} = \nabla\log\hat{\varphi}\_T(\boldsymbol{x})$ given the optimal SB density $p\_t^\star$, so applying the same logic, we have:

$$
\widehat{\boldsymbol{Z}}_T^{(k)} := \arg\min_{\widehat{\boldsymbol{Z}}_T} \mathbb{E}_{p_{0,T}^{u^{(k)}}}\left[\left\lVert \widehat{\boldsymbol{Z}}_T(\boldsymbol{X}_T) - \nabla\log\mathbb{Q}_{T|0}(\boldsymbol{X}_T \mid \boldsymbol{X}_0) \right\rVert^2\right] \overset{}{=} \nabla\log\hat{\varphi}_T
$$

which yields the same backward time SDE through the terminal constraint as the optimal drift $\boldsymbol{v}^\star$, and we have shown that optimizing the Corrector Matching Objective is **equivalent to finding the optimal reverse-time dynamics that correct the SB forward-time SDE** such that it satisfies the terminal constraint. $\square$

The results from Proposition 6.13 and Proposition 6.14 indicate that alternating between optimizing the SB-AM Objective and Corrector Matching Objective is equivalent to alternating between solving the forward Schroedinger half-bridge that satisfies the initial marginal to solving the backward Schroedinger half-bridge that satisfies the target marginal. This alternating scheme is reminiscent of our discussion of **Sinkhorn's algorithm** from Section 1.5, but now adapted using an efficient *matching objective*. Just like the adjoint matching algorithm (Box 6.5), we can optimize the SB-AM Objective and Corrector Matching Objective by repeatedly optimizing over samples from a replay buffer.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.15</span><span class="math-callout__name">(Adjoint and Corrector Matching Doesn't Require Target Samples)</span></p>

We highlight that optimizing the SB-AM Objective and Corrector Matching Objective does not require explicit samples from $\boldsymbol{X}\_T \sim \pi\_T$ since the samples used to compute the objective are purely from sampling $(\boldsymbol{X}\_0, \boldsymbol{X}\_T) \sim p\_{0,T}^{\bar{u}}$ from the SDE induced by $\boldsymbol{u}$. The dependence on $\pi\_T$ only appears when computing the loss, which can be computed as a **probability under** $\pi\_T$, which can be empirical or a pre-defined potential energy function, as we will discuss further in Section 8.3.

</div>

Throughout this section, we have demonstrated how the adjoint state can be used as an efficient variational framework for solving Schroedinger bridge problems by alternating between two tractable objectives that correspond to the forward and backward Schroedinger half-bridges. Rather than directly optimizing over path measures, the method reduces the problem to learning the forward control drift and the backward correction through cheap matching objectives that can be evaluated using trajectories generated by the current dynamics. This formulation yields several practical advantages:

- **(i)** It avoids the need for expensive likelihood ratios or full path-space KL computations, replacing them with local drift-matching losses that are straightforward to estimate.
- **(ii)** The alternating optimization naturally mirrors the structure of Sinkhorn iterations in entropic optimal transport, providing convergence guarantees to the optimal SB control $\boldsymbol{u}^\star$.
- **(iii)** The corrector matching objective does not require explicit samples from the target distribution, enabling training on unknown energy-based target densities.
- **(iv)** Since the objectives do not require backpropagation through SDE trajectories or maintaining the full trajectory in memory, it can easily scale to high-dimensional systems.

These properties make adjoint matching a practical and flexible approach for learning Schroedinger bridges in complex generative modeling settings.

### 6.6 Closing Remarks for Section 6

In this section, we explored several generative modeling frameworks grounded in Schroedinger bridge (SB) theory. We began with the classical score-based generative modeling paradigm (Section 6.1), which can be interpreted as a specialized instance of the SB formulation in which the forward process is fixed, and learning focuses on estimating the reverse-time dynamics. Building on this perspective, we then leveraged SB theory to generalize to controlled forward processes, where both the forward and backward control drifts can be learned through likelihood-based training (Section 6.2).

Next, we introduced an alternative viewpoint through the Iterative Markovian Fitting (IMF) procedure and the diffusion Schroedinger bridge matching algorithm (Section 6.3), which draws on the Markov and reciprocal projection theory developed in Section 4.5. In this formulation, solving the Schroedinger bridge problem can be understood as performing iterative projections in path space.

To overcome the limitations in optimizing over full stochastic trajectories in the path space from the previous approaches, we conclude with two approaches for learning Schroedinger bridges with efficient **matching objectives**, including score and flow matching (Section 6.4) and adjoint matching (Section 6.5), which locally optimize the control drift to generate trajectories consistent with the optimal Schroedinger bridge dynamics.

Overall, this section builds the intuition behind the core generative modeling frameworks that leverage Schroedinger bridge theory. While it is not intended to be an exhaustive review of algorithmic developments in the field, it should provide the core theoretical foundations needed to understand a broad class of modern generative modeling techniques.

So far, we have restricted our attention to the **continuous state space**, where data is represented as continuous-valued vectors in $\mathbb{R}^d$. The structure of the continuous state space is *required* for many ideas developed throughout this guide, including stochastic differential equations, path measures, and Brownian motion. This naturally raises the question: *How does Schroedinger bridge theory extend to the discrete state space, where states belong to a finite or countable set rather than a continuous vector space?* This is precisely the question that we explore in the next section:

- **(i)** We will introduce the concept of stochastic processes in the discrete state space as **continuous-time Markov chains** (CTMCs), where the control drift in SDEs takes the analogous form of a *transition rate matrix* in discrete state spaces.
- **(ii)** While the structure of the discrete Schroedinger bridge problem remains the same, we will introduce fundamental differences in the KL divergence in discrete state spaces. This will provide the theoretical grounding for our discussion on solving the discrete Schroedinger bridge problem with generative modeling, which adapts several ideas developed in the previous sections for the discrete state space.

In doing so, we extend the Schroedinger bridge framework from diffusion processes on continuous spaces to jump processes on discrete state spaces, laying the groundwork for an even broader class of generative modeling methods.

## 7. From Continuous to Discrete State Space

Now that we have built the foundation required to understand and construct the Schroedinger bridge where the states exist as continuous latent vectors in some state space $\mathcal{X} \subseteq \mathbb{R}^d$, we will now take a detour to the **discrete state space**, where states exist as probabilities of existing in a finite set of discrete states. We will see that rather than representing dynamics with velocity fields that transport states via smooth lines, dynamics in the discrete state space are represented with **transition rates** that characterize the instantaneous change in the probabilities of existing in each state.

In this section, we introduce discrete state path measures not as SDEs but as **continuous-time Markov chains** (CTMCs) defined by their rate matrices (Section 7.1). Leveraging the theory of CTMCs, we define the **discrete Schroedinger bridge problem**, and extend the definitions for the Radon-Nikodym derivative (RND) and KL divergence to CTMCs (Section 7.2). Then, we analyze two methods of solving the discrete SB problem which mirror the continuous state space, starting with the stochastic optimal control formulation (Section 7.3 and 3.2) and concluding with the discrete analog of Iterative Markovian Fitting using Markovian and reciprocal projections (Sections 7.5 and 7.6).

### 7.1 Continuous-Time Markov Chains

The discrete state space can be defined as a finite set of states $\mathcal{X} = \lbrace 1, \ldots, d \rbrace$ and the probability simplex $\Delta^{d-1}$ over the $d$ discrete states given by:

$$
\Delta^{d-1} = \left\lbrace \boldsymbol{x} = (x_1, \ldots, x_d) \in \mathbb{R}^d \;\middle|\; x_i \in [0, 1], \; \sum_{i=1}^d x_i = 1 \right\rbrace
$$

To define a stochastic path measure $\mathbb{P}$ that lies on the simplex $\Delta^{d-1}$, we introduce the theory of **continuous-time Markov chains** (CTMCs) in the probability space $(\Omega, \text{Pr})$, where $\Omega \in D([0, T], \mathcal{X})$ is the space of left-limited and right-continuous (càdlàg) paths over $\mathcal{X}$ and $\text{Pr}$ is the probability measure over events. A CTMC is a stochastic process that evolves over time $\boldsymbol{X}\_{0:T}$ whose probability law is defined by a time-dependent **generator** or **transition rate matrix** $(\boldsymbol{Q}\_t \in \mathbb{R}^{\mathcal{X} \times \mathcal{X}})\_{t \in [0,T]}$ of the form:

$$
\boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) = \lim_{\Delta t \to 0} \frac{1}{\Delta t}\left(\text{Pr}(\boldsymbol{X}_{t+\Delta t} = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) - \mathbf{1}_{\boldsymbol{x} = \boldsymbol{y}}\right)
$$

which defines the instantaneous rate of transitioning from state $\boldsymbol{x} \in \mathcal{X}$ to state $\boldsymbol{y} \in \mathcal{X}$ at time $t$. Since the state transitions must remain on the probability simplex, the transition rates satisfy the following conditions:

$$
\forall \boldsymbol{x} \ne \boldsymbol{y}, \quad \boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) \ge 0, \qquad \sum_{\boldsymbol{y} \in \mathcal{X}} \boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) = 0
$$

A generator $\boldsymbol{Q}$ *uniquely* defines a path measure $\mathbb{P} \in \mathcal{P}(\Omega)$, under which we can define the transition probability over a discrete time interval $[t, t + \Delta t]$ as:

$$
\mathbb{P}(\boldsymbol{X}_{t+\Delta t} = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) = \begin{cases} \Delta t \boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) + O(\Delta t^2) & \boldsymbol{y} \ne \boldsymbol{x} \\ 1 - \Delta t \sum_{z \ne \boldsymbol{x}} \boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{z}) + O(\Delta t^2) & \boldsymbol{y} = \boldsymbol{x} \end{cases}
$$

Given that CTMCs are càdlàg paths, we denote the left limit of a state $\boldsymbol{X}\_t$ as $\boldsymbol{X}\_{t^-} = \lim\_{s \uparrow t} \boldsymbol{X}\_s$, where $\boldsymbol{X}\_{t^-} \ne \boldsymbol{X}\_t$ at jump times from state $\boldsymbol{X}\_{t^-}$ to state $\boldsymbol{X}\_t$. A **key property** of CTMCs is that they satisfy the **Kolmogorov forward equation** which define the time evolution of the path.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.1</span><span class="math-callout__name">(Kolmogorov Forward Equation for CTMCs)</span></p>

The forward-time dynamics of a CTMC $\boldsymbol{X}\_{0:T}$ with probability measure $p\_t(\cdot) := \text{Pr}(\boldsymbol{X}\_t = \cdot)$ and generator $\boldsymbol{Q}\_t$ satisfies the **Kolmogorov forward equation** defined as:

$$
\forall \boldsymbol{x} \in \mathcal{X}, \quad \partial_t p_t(\boldsymbol{x}) = \sum_{\boldsymbol{y} \in \mathcal{X}} \boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) p_t(\boldsymbol{y}) = \sum_{\boldsymbol{x} \ne \boldsymbol{y}} (\boldsymbol{Q}_t(\boldsymbol{y}, \boldsymbol{x}) p_t(\boldsymbol{y}) - \boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) p_t(\boldsymbol{x}))
$$

where that the probability measure $p\_t(\cdot)$ is **unique** given a pair of endpoint conditions at $t \in \lbrace 0, T \rbrace$ and $t \mapsto \boldsymbol{Q}\_t$ is continuous over time $t \in [0, T]$.

</div>

*Proof.* This proposition can be shown simply by defining a forward transition probability over the discrete time increment $[t, t + \Delta t]$ and taking a limit as $\Delta t \to 0$ to get the expression for $\partial\_t p\_t$. First, using the transition probability:

$$
p_{t+\Delta t}(\boldsymbol{x}) = \sum_{\boldsymbol{y} \in \mathcal{X}} \text{Pr}(\boldsymbol{X}_{t+\Delta t} = \boldsymbol{x} \mid \boldsymbol{X}_t = \boldsymbol{y}) p_t(\boldsymbol{y}) = p_t(\boldsymbol{x}) + \Delta t \sum_{\boldsymbol{y} \in \mathcal{X}} \boldsymbol{Q}_t(\boldsymbol{y}, \boldsymbol{x}) p_t(\boldsymbol{y}) + O(\Delta t^2)
$$

Taking the continuous time limit as $\Delta t \to 0$:

$$
\partial_t p_t(\boldsymbol{x}) = \sum_{\boldsymbol{y} \in \mathcal{X}} \boldsymbol{Q}_t(\boldsymbol{y}, \boldsymbol{x}) p_t(\boldsymbol{y}) = \boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{x}) p_t(\boldsymbol{x}) + \sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t(\boldsymbol{y}, \boldsymbol{x}) p_t(\boldsymbol{y}) = \sum_{\boldsymbol{x} \ne \boldsymbol{y}} (\boldsymbol{Q}_t(\boldsymbol{y}, \boldsymbol{x}) p_t(\boldsymbol{y}) - \boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) p_t(\boldsymbol{x}))
$$

which is exactly the Kolmogorov forward equation. To prove **uniqueness**, we can write it in vector form as $-\partial\_t \boldsymbol{p}\_t = \boldsymbol{Q}\_t \boldsymbol{p}\_t$, which is a linear ODE in $\mathbb{R}^{|\mathcal{X}|}$. Since $t \mapsto \boldsymbol{Q}\_t$ is continuous, linear ODEs have a unique solution. $\square$

Having derived the Kolmogorov forward equation, which describes how the state distribution $p\_t$ evolves *forward* under the time-dependent generator $\boldsymbol{Q}\_t$, we now ask how CTMC dynamics evolve *backwards* from a terminal constraint. As we have seen in the continuous state space, the idea of terminal conditioning is the foundation for solving the Schroedinger bridge problem, as it allows us to define the optimal dynamics that generate a target distribution. The idea of evolving CTMC dynamics backward is exactly captured by the **Kolmogorov backward equation**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.2</span><span class="math-callout__name">(Kolmogorov Backward Equation for CTMCs)</span></p>

Consider a CTMC $\boldsymbol{X}\_{0:T}$ under the measure $\mathbb{P}$ with generator $\boldsymbol{Q}\_t$ and let $\phi\_t(\boldsymbol{x}) := \mathbb{E}[\phi(\boldsymbol{X}\_T) \mid \boldsymbol{X}\_t = \boldsymbol{x}]$ given an arbitrary test function $\phi : \mathcal{X} \to \mathbb{R}$. Then, the reverse-time dynamics of $\phi\_t$ satisfies the **Kolmogorov backward equation** defined as:

$$
\forall \boldsymbol{x} \in \mathcal{X}, \quad -\partial_t\phi(\boldsymbol{x}) = \sum_{\boldsymbol{y} \in \mathcal{X}} \phi_t(\boldsymbol{y})\boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) = \sum_{\boldsymbol{y} \ne \boldsymbol{x}} (\phi_t(\boldsymbol{y}) - \phi(\boldsymbol{x}))\boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}), \quad \phi_T(\boldsymbol{x}) = \phi(\boldsymbol{x})
$$

which admits a **unique** solution $\phi$ when $t \mapsto \boldsymbol{Q}\_t$ is continuous over all $t \in [0, T]$.

</div>

*Proof.* Since $\phi\_t(\boldsymbol{x})$ can be considered some cost function to go from $\boldsymbol{X}\_t = \boldsymbol{x}$ to the terminal state $\boldsymbol{X}\_T$, we can expand the inner function $\phi(\boldsymbol{X}\_T)$ using the law of total expectation to write $\phi\_t(\boldsymbol{x})$ with respect to a discrete time step $\phi\_{t+\Delta t}(\boldsymbol{x})$:

$$
\phi_t(\boldsymbol{x}) = \mathbb{E}[\phi(\boldsymbol{X}_T) \mid \boldsymbol{X}_t = \boldsymbol{x}] = \mathbb{E}[\phi_{t+\Delta t}(\boldsymbol{X}_{t+\Delta t}) \mid \boldsymbol{X}_t = \boldsymbol{x}]
$$

Then, applying the transition probability:

$$
\phi_t(\boldsymbol{x}) = \sum_{\boldsymbol{y} \in \mathcal{X}} \phi_{t+\Delta t}(\boldsymbol{y})\mathbb{P}(\boldsymbol{X}_{t+\Delta t} = \boldsymbol{y} \mid \boldsymbol{X}_t = \boldsymbol{x}) = \phi_{t+\Delta t}(\boldsymbol{x}) + \Delta t \sum_{\boldsymbol{y} \in \mathcal{X}} \phi_{t+\Delta t}(\boldsymbol{y})\boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) + O(\Delta t^2)
$$

Now, subtracting $\phi\_{t+\Delta t}(\boldsymbol{x})$ from both sides, dividing by $\Delta t$, and taking the continuous time limit $\Delta t \to 0$, we get the reverse time dynamics $-\partial\_t\phi\_t(\boldsymbol{x})$:

$$
-\partial_t\phi_t(\boldsymbol{x}) = \sum_{\boldsymbol{y} \in \mathcal{X}} \phi_t(\boldsymbol{y})\boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y}) = \sum_{\boldsymbol{y} \ne \boldsymbol{x}} (\phi_t(\boldsymbol{y}) - \phi_t(\boldsymbol{x}))\boldsymbol{Q}_t(\boldsymbol{x}, \boldsymbol{y})
$$

which is the Kolmogorov backward equation. To prove **uniqueness**, we follow the proof of the forward equation and write it in vector form as $-\partial\_t\boldsymbol{\phi}\_t = \boldsymbol{Q}\_t\boldsymbol{\phi}\_t$, which is a linear ODE in $\mathbb{R}^{|\mathcal{X}|}$. Since $t \mapsto \boldsymbol{Q}\_t$ is continuous, linear ODEs have a unique solution. $\square$

CTMCs can be interpreted as the **discrete analog of stochastic differential equations (SDEs)** in the continuous state space, where the time-dependent generator conditioned on generator $\boldsymbol{Q}\_t(\boldsymbol{x}, \cdot) : \mathcal{X} \times [0, T] \to \mathbb{R}^d$ is analogous to the time-dependent drift $\boldsymbol{u}(\boldsymbol{x}, t)$ of an SDE. Since we have characterized the forward and reverse evolution of path-measure CTMCs, we are ready to formulate the discrete-state-space analog of the Schroedinger bridge problem, which aims to recover a CTMC that satisfies a pair of marginal distributions while remaining close to a reference CTMC.

### 7.2 Discrete Schroedinger Bridge Problem

In this section, we formulate the **discrete Schroedinger bridge problem** for continuous-time Markov chains (CTMCs) on a finite state space $\mathcal{X} = \lbrace 1, \ldots, d \rbrace$. Just like the continuous-space Schroedinger bridge problem, the objective is to find a path measure $\mathbb{P}^\star$ that is closest, in relative entropy or KL divergence, to a given reference process $\mathbb{Q}$, while matching prescribed marginal constraints $p\_0 = \pi\_0$ and $p\_T = \pi\_T$.

Since the *control drift* in continuous SB theory is denoted $\boldsymbol{u}(\boldsymbol{x}, t) : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$, we use $\boldsymbol{Q}\_t^u(\boldsymbol{x}, \cdot) : \mathcal{X} \times [0, T] \to \mathbb{R}^{d \times d}$ to denote the CTMC generator of the *controlled* path measure $\mathbb{P}^u \in \mathcal{P}(C([0, T]; \mathcal{X}))$, which we aim to optimize to recover the Schroedinger bridge. To denote the generator of the *reference process* $\mathbb{Q} \in \mathcal{P}(C([0, T]; \mathcal{X}))$, we use $\boldsymbol{Q}\_t^0(\boldsymbol{x}, \cdot) : \mathcal{X} \times [0, T] \to \mathbb{R}^{d \times d}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.3</span><span class="math-callout__name">(Discrete Schroedinger Bridge Problem)</span></p>

Consider a reference CTMC measure $\mathbb{Q}$ with generator $\boldsymbol{Q}\_t^0$ and a pair of marginal distributions in the discrete state space $\pi\_0, \pi\_T \in \mathcal{P}(\mathcal{X})$. The **discrete Schroedinger bridge problem** aims to determine the optimal CTMC path measure $\mathbb{P}^\star$ with generator $\boldsymbol{Q}\_t^\star$ where $p\_0 = \pi\_0$ and $p\_T = \pi\_T$ that solves the minimization problem:

$$
\mathbb{P}^\star = \arg\min_{\mathbb{P}^u \in \mathcal{P}(C([0,T]; \mathcal{X}))} \left\lbrace \text{KL}(\mathbb{P}^u \| \mathbb{Q}) : p_0 = \pi_0, \; p_T = \pi_T \right\rbrace
$$

where the KL divergence $\text{KL}(\cdot \| \cdot)$ between CTMCs is defined as:

$$
\text{KL}(\mathbb{P}^u \| \mathbb{Q}) = \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^u}\left[\int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \left(\boldsymbol{Q}_t^u \log\frac{\boldsymbol{Q}_t^u}{\boldsymbol{Q}_t^0} + \boldsymbol{Q}_t^0 - \boldsymbol{Q}_t^u\right)(\boldsymbol{X}_t, \boldsymbol{y}) \, dt\right]
$$

</div>

Solving the discrete SB problem yields the CTMC path measure that minimizes the discrepancy from the reference jump process, which penalizes differences in jump intensities, jump times, and transition structure over the time interval $t \in [0, T]$.

To define the form of the KL divergence, we first derive the **Radon-Nikodym derivative between CTMC path measures**, which defines the probability ratio of a CTMC $\boldsymbol{X}\_{0:T}$ under two path measures $\mathbb{P}$ and $\mathbb{P}'$ with **generators** $\boldsymbol{Q}\_t$ and $\boldsymbol{Q}\_t'$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.4</span><span class="math-callout__name">(Radon-Nikodym Derivative Between CTMCs)</span></p>

Consider two CTMC path measures $\mathbb{P}$ and $\mathbb{P}'$ with generators $\boldsymbol{Q}$ and $\boldsymbol{Q}\_t'$ and initial distributions $p\_0 = \pi\_0$ and $\mathbb{P}\_0' = \pi\_0'$. Assume that the $\pi\_0' \ll \pi\_0$ and $\mathbb{P}' \ll \mathbb{P}$, where the generators satisfy $\boldsymbol{Q}(\boldsymbol{x}, \boldsymbol{y}) = 0 \implies \boldsymbol{Q}'(\boldsymbol{x}, \boldsymbol{y}) = 0$. Then, the logarithm of the Radon-Nikodym derivative is given by:

$$
\log\frac{d\mathbb{P}'}{d\mathbb{P}}(\boldsymbol{X}_{0:T}) = \log\frac{d\pi_0'}{d\pi_0}(\boldsymbol{X}_0) + \sum_{t: \boldsymbol{X}_{t^-} \ne \boldsymbol{X}_t} \log\frac{\boldsymbol{Q}_t'(\boldsymbol{X}_{t^-}, \boldsymbol{X}_t)}{\boldsymbol{Q}_t(\boldsymbol{X}_{t^-}, \boldsymbol{X}_t)} + \int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} (\boldsymbol{Q}_t - \boldsymbol{Q}_t')(\boldsymbol{X}_t, \boldsymbol{y}) dt
$$

</div>

*Proof.* To derive the RND for CTMCs, we first consider the **discrete-time case**, where we break down the time horizon into time steps $0 = t\_0 < t\_1 < \cdots < t\_k < \cdots < t\_{K-1} < t\_K = T$ where the time intervals are separated by $\Delta t$. Then, we can write the log ratio of a discrete path $(\boldsymbol{X}\_{t\_k})\_{k \in \lbrace 0, \ldots, K \rbrace}$ under the probability measures $\mathbb{P}$ and $\mathbb{P}'$ as:

$$
\log\frac{d\mathbb{P}'}{d\mathbb{P}}(\boldsymbol{X}_{0:T}) = \log\frac{d\pi_0'}{d\pi_0}(\boldsymbol{X}_0) + \sum_{k=0}^{K-1}\log\frac{d\mathbb{P}'(\boldsymbol{X}_{t_{k+1}} \mid \boldsymbol{X}_{t_k})}{d\mathbb{P}(\boldsymbol{X}_{t_{k+1}} \mid \boldsymbol{X}_{t_k})} + O(\Delta t)
$$

From the transition probability, we can derive the probability ratio for two distinct cases given an interval $[t\_k, t\_{k+1}]$: when at least one change in state is made, the log ratio becomes:

$$
\log\frac{\mathbb{P}'(\boldsymbol{X}_{t_{k+1}} \mid \boldsymbol{X}_{t_k})}{\mathbb{P}(\boldsymbol{X}_{t_{k+1}} \mid \boldsymbol{X}_{t_k})} = \log\frac{\boldsymbol{Q}_{t_k}'(\boldsymbol{X}_{t_k}, \boldsymbol{X}_{t_{k+1}})}{\boldsymbol{Q}_{t_k}(\boldsymbol{X}_{t_k}, \boldsymbol{X}_{t_{k+1}})} + O(\Delta t^2)
$$

For the case when no jumps are made, the log ratio becomes:

$$
\log\frac{\mathbb{P}'(\boldsymbol{X}_{t_{k+1}} \mid \boldsymbol{X}_{t_k})}{\mathbb{P}(\boldsymbol{X}_{t_{k+1}} \mid \boldsymbol{X}_{t_k})} = \Delta t \sum_{\boldsymbol{y} \ne \boldsymbol{X}_{t_k}} (\boldsymbol{Q}_{t_k}(\boldsymbol{X}_{t_k}, \boldsymbol{y}) - \boldsymbol{Q}_{t_k}'(\boldsymbol{X}_{t_k}, \boldsymbol{y})) + O(\Delta t^2)
$$

Substituting both cases and taking the continuous time limit $\Delta t \to 0$ and $K \to \infty$, we have:

$$
\log\frac{d\mathbb{P}'}{d\mathbb{P}}(\boldsymbol{X}_{0:T}) = \log\frac{d\pi_0'}{d\pi_0}(\boldsymbol{X}_0) + \sum_{t: \boldsymbol{X}_{t^-} \ne \boldsymbol{X}_t} \log\frac{\boldsymbol{Q}_t'(\boldsymbol{X}_{t^-}, \boldsymbol{X}_t)}{\boldsymbol{Q}_t(\boldsymbol{X}_{t^-}, \boldsymbol{X}_t)} + \int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} (\boldsymbol{Q}_t - \boldsymbol{Q}_t')(\boldsymbol{X}_t, \boldsymbol{y}) dt
$$

which is the exact form of the log RND between $\mathbb{P}'$ and $\mathbb{P}$. $\square$

Using this result, we can easily derive the **KL divergence between CTMC path measures**, which is the foundation of the discrete Schroedinger bridge objective.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 7.5</span><span class="math-callout__name">(KL Divergence Between CTMCs)</span></p>

The KL divergence between two CTMC path measures $\mathbb{P}$ and $\mathbb{P}'$ with generators $\boldsymbol{Q}$ and $\boldsymbol{Q}\_t'$ is defined as:

$$
\text{KL}(\mathbb{P}' \| \mathbb{P}) = \text{KL}(\pi_0' \| \pi_0) + \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}'}\left[\int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \left(\boldsymbol{Q}_t' \log\frac{\boldsymbol{Q}_t'}{\boldsymbol{Q}_t}\right)(\boldsymbol{X}_t, \boldsymbol{y}) + (\boldsymbol{Q}_t - \boldsymbol{Q}_t')(\boldsymbol{X}_t, \boldsymbol{X}_t) \, dt\right]
$$

and can be equivalently written as:

$$
\text{KL}(\mathbb{P}' \| \mathbb{P}) = \text{KL}(\pi_0' \| \pi_0) + \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}'}\left[\int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \left(\boldsymbol{Q}_t' \log\frac{\boldsymbol{Q}_t'}{\boldsymbol{Q}_t}\right)(\boldsymbol{X}_t, \boldsymbol{y}) + \int_0^T (\boldsymbol{Q}_t' - \boldsymbol{Q}_t)(\boldsymbol{X}_t, \boldsymbol{X}_t) \, dt\right]
$$

</div>

*Proof.* Recalling that the KL divergence $\text{KL}(\mathbb{P}' \| \mathbb{P})$ is simply the expectation of the log RND over paths from the first argument, we have:

$$
\text{KL}(\mathbb{P}' \| \mathbb{P}) = \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}'}\left[\log\frac{d\mathbb{P}'}{d\mathbb{P}}\right]
$$

Substituting the log RND expression and separating into three terms: the first term $(\bigstar)$ gives $\text{KL}(\pi\_0' \| \pi\_0)$ since it only depends on $\boldsymbol{X}\_0$. For the second term $(\blacklozenge)$, we consider the discrete time case for $0 = t\_0 < \cdots < t\_K = T$, distribute the expectation, and take the continuous time limit:

$$
(\blacklozenge) \xrightarrow[\Delta t \to 0]{} \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}'}\left[\int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \left(\boldsymbol{Q}_t' \log\frac{\boldsymbol{Q}_t'}{\boldsymbol{Q}_t}\right)(\boldsymbol{X}_t, \boldsymbol{y}) \, dt\right]
$$

Plugging these expressions together, we get the KL divergence between CTMCs. We can also use the equality $\boldsymbol{Q}\_t(\boldsymbol{x}, \boldsymbol{x}) = 1 - \sum\_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}\_t(\boldsymbol{x}, \boldsymbol{y})$ to write $\sum\_{\boldsymbol{y} \ne \boldsymbol{X}\_t}(\boldsymbol{Q}\_t - \boldsymbol{Q}\_t')(\boldsymbol{X}\_t, \boldsymbol{y}) = (\boldsymbol{Q}\_t' - \boldsymbol{Q}\_t)(\boldsymbol{X}\_t, \boldsymbol{X}\_t)$, which is an equivalent expression for KL divergence between CTMCs using the rate of remaining at a position or the **stay rate**. $\square$

This decomposition reveals that the KL divergence between two CTMC path measures is the sum of the discrepancy between their initial marginals and a time-integrated KL divergence of the generator matrices at *jump times*, where the state $\boldsymbol{X}\_t$ jumps to a new state $\boldsymbol{y} \ne \boldsymbol{X}\_t$. This provides an intuitive interpretation for the discrete Schroedinger bridge problem, which selects, among all Markov processes matching the marginals, the one whose jump dynamics deviates minimally in KL divergence from the reference dynamics. While we defined the canonical reference process in the continuous state space as Brownian motion with a reference drift $\boldsymbol{f}$, which is often set to $\boldsymbol{f} := 0$, we can also define two common forms of the reference generator $\boldsymbol{Q}^0$ below.

#### Forms of the Reference Generator for CTMCs

The **reference generator** $\boldsymbol{Q}^0$ defining the CTMC $\mathbb{Q}$ is the baseline discrete stochastic dynamics that is minimally reweighted to match the prescribed marginal constraints. The choice of $\boldsymbol{Q}^0$ fundamentally changes the solution to the Discrete SB Problem. Here, we will highlight the **uniform generator** and the **pre-trained generator** as common choices.

**Uniform Generator.** The uniform generator $\boldsymbol{Q}\_t^0$ is defined as:

$$
\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}) = \begin{cases} \frac{\gamma(t)}{|\mathcal{X}| - 1} & \boldsymbol{y} \ne \boldsymbol{x} \\ -\gamma(t) & \boldsymbol{y} = \boldsymbol{x} \end{cases}
$$

which corresponds to a homogeneous jump process such that from any state $\boldsymbol{x} \in \mathcal{X}$ at time $t$, the state remains unchanged with rate $-\gamma(t)$ and jumps uniformly to any other state $\boldsymbol{y} \ne \boldsymbol{x}$. This means that the optimal generator $\boldsymbol{Q}^\star$ is entirely determined by the marginal constraints. It can be interpreted as the discrete state space analog of pure Brownian motion and is useful for settings where prior dynamics are unknown.

**Baseline Transition Generator.** A more structured alternative is to define the reference generator $\boldsymbol{Q}\_t^0$ with baseline transition rates of the system's natural dynamics or from a **pretrained model**, where the model is trained to capture the dynamics of the system from data samples. This choice embeds prior knowledge about plausible transitions into the reference process. The Schroedinger bridge then acts as a minimal correction of the pretrained dynamics to match the prescribed marginals.

### 7.3 Stochastic Optimal Control of CTMCs

Just like in the continuous state space, we can reframe the SB problem as a **stochastic optimal control** (SOC) problem, where the Schroedinger bridge aligns with the lowest cost path from any intermediate state $\boldsymbol{x} \in \mathcal{X}$ to a state in the target distribution. In this section, we build the theoretical foundations of SOC in the discrete state space, highlighting the deviations from the continuous state-space formulation, which will lead us to explicitly defining objectives for tractably solving the discrete SB problem with SOC in Section 7.4.

First, we define the **cost functional** $J(\boldsymbol{x}, t; \boldsymbol{u}) : \mathcal{X} \times [0, T] \to \mathbb{R}$ which returns the *cost-to-go* from an intermediate state $\boldsymbol{x} \in \mathcal{X}$ to the target distribution $\pi\_T$ under the **controlled** path measure $\mathbb{P}^u$:

$$
J(\boldsymbol{x}, t; \boldsymbol{u}) := \mathbb{E}_{\boldsymbol{X}_{0:T}^u \sim \mathbb{P}^u}\left[\int_t^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_s^u} C_t(\boldsymbol{X}_s^u, \boldsymbol{y}) ds + \Phi(\boldsymbol{X}_T^u) \;\middle|\; \boldsymbol{X}_t^u = \boldsymbol{x}\right]
$$

which measures the expected cumulative running cost $C\_t(\boldsymbol{x}, \boldsymbol{y})$ incurred from time $t$ until the terminal time $T$ and the terminal cost $\Phi(\boldsymbol{x}) : \mathcal{X} \to \mathbb{R}$. The running cost is defined as the instantaneous KL divergence between the controlled generator $\boldsymbol{Q}\_t^u$ and the reference generator $\boldsymbol{Q}\_t^0$ given by:

$$
C_t(\boldsymbol{x}, \boldsymbol{y}) := \text{KL}(p_t^u \| q_t) = \left(\boldsymbol{Q}_t^u \log\frac{\boldsymbol{Q}_t^u}{\boldsymbol{Q}_t^0} - \boldsymbol{Q}_t^u + \boldsymbol{Q}_t^0\right)(\boldsymbol{x}, \boldsymbol{y})
$$

The objective of the SOC problem is to determine the **optimal generator** $\boldsymbol{Q}^\star := \boldsymbol{Q}^{u^\star}$ that minimizes the cost-to-go functional:

$$
J^\star(\boldsymbol{x}, t; \boldsymbol{u}^\star) := \inf_{\mathbb{P}^u} J(\boldsymbol{x}, t; \boldsymbol{u}), \quad \forall (\boldsymbol{x}, t) \in \mathcal{X} \times [0, T]
$$

which generates the **optimal CTMC path measure** $\mathbb{P}^\star := \mathbb{P}^{u^\star}$ with generator $\boldsymbol{Q}\_t^\star := \boldsymbol{Q}\_t^{u^\star}$. Analogous to the continuous setting, the optimal cost-to-go satisfies **Bellman's Principle of Optimality**. For a small time increment $\Delta t$, the cost decomposes into the cost accumulated over $[t, t + \Delta t]$ and the optimal cost from $t + \Delta t$ onward:

$$
J^\star(\boldsymbol{x}, t; \boldsymbol{u}^\star) = \inf_{\mathbb{P}^u}\left\lbrace \Delta t \sum_{\boldsymbol{y} \ne \boldsymbol{x}} C_t(\boldsymbol{x}, \boldsymbol{y}) + O(\Delta t^2) + \mathbb{E}_{\boldsymbol{X}_{t+\Delta t:T}^u \sim \mathbb{P}^u}\left[J^\star(\boldsymbol{X}_{t+\Delta t}^u, t + \Delta t) \mid \boldsymbol{X}_t^u = \boldsymbol{x}\right]\right\rbrace
$$

The optimally controlled measure $\mathbb{P}^\star$ can be obtained by defining the **value function** $V\_t(\boldsymbol{x})$ as the *optimal cost-to-go* $V\_t(\boldsymbol{x}) := J^\star(\boldsymbol{x}, t; \boldsymbol{u}^\star)$, which yields the **dynamic programming** relation:

$$
V_t(\boldsymbol{x}) = \inf_{\mathbb{P}^u}\left\lbrace \Delta t \sum_{\boldsymbol{y} \ne \boldsymbol{x}} C_t(\boldsymbol{x}, \boldsymbol{y}) + O(\Delta t^2) + \mathbb{E}_{\boldsymbol{X}_{t+\Delta t:T}^u \sim \mathbb{P}^u}\left[V_{t+\Delta t}(\boldsymbol{X}_{t+\Delta t}^u) \mid \boldsymbol{X}_t^u = \boldsymbol{x}\right]\right\rbrace
$$

Since no more running cost $C\_t$ can be incurred at time $t = T$, the terminal value function is equal to the terminal cost $V\_T(\boldsymbol{x}) = \Phi(\boldsymbol{x})$.

To obtain an explicit characterization of the optimal CTMC dynamics solving the stochastic optimal control problem, we derive the structure of the optimal process in a sequence of steps. The key goal is to connect the dynamic programming formulation of the SOC problem with the change-of-measure perspective underlying Schroedinger bridges. Concretely, we first determine the form of the optimal controlled generator $\boldsymbol{Q}^\star$, then characterize the value function through the Hamilton-Jacobi-Bellman equation, and finally use these results to recover the optimal path measure $\mathbb{P}^\star$ and its likelihood ratio with respect to the reference process. **The derivation proceeds as follows:**

- **(i)** We derive the form of the optimal generator $\boldsymbol{Q}\_t^\star$ that defines the law of paths under the path measure $\mathbb{P}^\star$ that solves the SOC problem (Proposition 7.6).
- **(ii)** We show that the value function satisfies the Hamilton-Jacobi-Bellman equation (Corollary 7.8).
- **(iv)** We derive the form of the optimal path measure $\mathbb{P}^\star$ that solves the SOC problem (Proposition 7.9).
- **(iii)** Finally, we derive the Radon-Nikodym derivative (RND) between the optimal path measure $\mathbb{P}^\star$, and the reference path measure $\mathbb{Q}$ (Proposition 7.10).

Together, these theoretical ideas will form the basis for solving the discrete SB using SOC. We start with the derivation of the optimal generator $\boldsymbol{Q}\_t^\star$ with respect to the value function $V\_t$ which will naturally lead to the proof that the value function satisfies the HJB equation.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.6</span><span class="math-callout__name">(Optimal Generator)</span></p>

Given the generator $\boldsymbol{Q}\_t^0$ of the reference process $\mathbb{Q}$, the **optimal generator** $\boldsymbol{Q}\_t^\star$ of the process $\mathbb{P}^\star$ that solves the SOC problem takes the form:

$$
\boldsymbol{Q}_t^\star(\boldsymbol{x}, \boldsymbol{y}) = \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}) e^{V_t(\boldsymbol{x}) - V_t(\boldsymbol{y})}, \quad \forall \boldsymbol{x}, \boldsymbol{y} \in \mathcal{X}
$$

where $V\_t : \mathcal{X} \to \mathbb{R}$ is the value function defined in the dynamic programming relation.

</div>

*Proof.* First, we expand the second term of the value function in the Bellman recursion, which defines the expected future cost. Splitting into the case $\boldsymbol{y} = \boldsymbol{x}$ and $\boldsymbol{y} \ne \boldsymbol{x}$:

$$
\inf_{\mathbb{P}^u} \mathbb{E}_{\boldsymbol{X}_{t+\Delta t:T}^u \sim \mathbb{P}^u}\left[V_{t+\Delta t}(\boldsymbol{X}_{t+\Delta t}^u) \mid \boldsymbol{X}_t^u = \boldsymbol{x}\right] = \inf_{\mathbb{P}^u}\left[V_{t+\Delta t}(\boldsymbol{x})(1 + \Delta t \boldsymbol{Q}_t^u(\boldsymbol{x}, \boldsymbol{x})) + \sum_{\boldsymbol{y} \ne \boldsymbol{x}} V_{t+\Delta t}(\boldsymbol{y}) \Delta t \boldsymbol{Q}_t^u(\boldsymbol{x}, \boldsymbol{y}) + O(\Delta t^2)\right]
$$

For any CTMC generator, the sum of transition rates from a given state $\boldsymbol{x}$ to all states $\boldsymbol{y} \in \mathcal{X}$ must sum to zero, we have $\boldsymbol{Q}\_t(\boldsymbol{x}, \boldsymbol{x}) = -\sum\_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}\_t(\boldsymbol{x}, \boldsymbol{y})$, which we can substitute to get:

$$
= V_{t+\Delta t}(\boldsymbol{x}) + \Delta t \inf_{\mathbb{P}^u}\left[\sum_{\boldsymbol{x} \ne \boldsymbol{y}} \boldsymbol{Q}_t^u(\boldsymbol{x}, \boldsymbol{y})(V_{t+\Delta t}(\boldsymbol{y}) - V_{t+\Delta t}(\boldsymbol{x}))\right] + O(\Delta t^2)
$$

Substituting this back into the Bellman recursion, we have:

$$
V_t(\boldsymbol{x}) - V_{t+\Delta t}(\boldsymbol{x}) = -\inf_{\mathbb{P}^u}\left\lbrace \Delta t \sum_{\boldsymbol{y} \ne \boldsymbol{x}} C_t(\boldsymbol{x}, \boldsymbol{y}) + \Delta t \sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t^u(\boldsymbol{x}, \boldsymbol{y})(V_{t+\Delta t}(\boldsymbol{y}) - V_{t+\Delta t}(\boldsymbol{x}))\right\rbrace + O(\Delta t^2)
$$

where we rearrange the Bellman recursion to express the forward difference $V\_{t+\Delta t}(\boldsymbol{x}) - V\_t(\boldsymbol{x})$, which introduces a minus sign before the infimum. This expression is used to get the time derivative $\partial\_t V\_t$, by dividing both sides by $\Delta t$ and take the limit $\Delta t \to 0$:

$$
\partial_t V_t(\boldsymbol{x}) = -\inf_{\mathbb{P}^u}\left\lbrace \sum_{\boldsymbol{y} \ne \boldsymbol{x}} \left(C_t(\boldsymbol{x}, \boldsymbol{y}) + \boldsymbol{Q}_t^u(\boldsymbol{x}, \boldsymbol{y})(V_t(\boldsymbol{y}) - V_t(\boldsymbol{x}))\right)\right\rbrace
$$

Defining $f(\boldsymbol{Q}\_t^u)$ as the function inside the infimum for all $\boldsymbol{x} \ne \boldsymbol{y}$ and expanding $C\_t(\boldsymbol{x}, \boldsymbol{y})$ as the KL divergence, we can take the derivative $f'(\boldsymbol{Q}\_t^u)$ with respect to $\boldsymbol{Q}\_t^u$ to get:

$$
f(\boldsymbol{Q}_t^u) = \left(\boldsymbol{Q}_t^u \log\frac{\boldsymbol{Q}_t^u}{\boldsymbol{Q}_t^0}\right)(\boldsymbol{x}, \boldsymbol{y}) - \boldsymbol{Q}_t^u(\boldsymbol{x}, \boldsymbol{y}) + \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}) + \boldsymbol{Q}_t^u(\boldsymbol{x}, \boldsymbol{y})(V_t(\boldsymbol{y}) - V_t(\boldsymbol{x}))
$$

$$
f'(\boldsymbol{Q}_t^u) = \log\frac{\boldsymbol{Q}_t^u}{\boldsymbol{Q}_t^0}(\boldsymbol{x}, \boldsymbol{y}) + V_t(\boldsymbol{y}) - V_t(\boldsymbol{x})
$$

Setting $f'(\boldsymbol{Q}\_t^u) = 0$ to obtain the minimizer $\boldsymbol{Q}\_t^\star$:

$$
\log\frac{\boldsymbol{Q}_t^\star}{\boldsymbol{Q}_t^0}(\boldsymbol{x}, \boldsymbol{y}) = V_t(\boldsymbol{x}) - V_t(\boldsymbol{y}) \implies \boldsymbol{Q}_t^\star(\boldsymbol{x}, \boldsymbol{y}) = \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}) e^{V_t(\boldsymbol{x}) - V_t(\boldsymbol{y})}
$$

which is the form of the optimal generator. $\square$

From the result, we can also observe that by rewriting the exponential as a fraction, we recover a form analogous the **Doob's $h$-transform** described in Section 4.4 for CTMCs.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.7</span><span class="math-callout__name">(Doob's $h$-Transform of CTMCs)</span></p>

The optimal generator $\boldsymbol{Q}\_t^\star$ is the Doob $h$-transform of the reference generator $\boldsymbol{Q}\_t^0$ where the $h$ function is defined as the exponentiated value function $h(\boldsymbol{x}, t) := e^{-V\_t(\boldsymbol{x})}$:

$$
\boldsymbol{Q}_t^\star(\boldsymbol{x}, \boldsymbol{y}) = \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\frac{e^{V_t(\boldsymbol{x})}}{e^{V_t(\boldsymbol{y})}} = \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\frac{e^{-V_t(\boldsymbol{y})}}{e^{-V_t(\boldsymbol{x})}} =: \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\frac{h(\boldsymbol{y}, t)}{h(\boldsymbol{x}, t)}
$$

which can be interpreted as tilting the generator toward states $\boldsymbol{y}$ that minimize the optimal cost-to-go defined by $V\_t(\boldsymbol{x}) := J^\star(\boldsymbol{x}, t; \boldsymbol{u}^\star)$.

</div>

Now we will show that the value function satisfies the HJB equation, which is analogous to our derivation in Section 2.7 in continuous state spaces. Crucially, this defines the discrete SOC problem with a non-linear PDE which can be transformed to a linear equation via exponentiation, which acts as a **discrete analog of the Hopf-Cole transform** discussed in Section 2.8.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 7.8</span><span class="math-callout__name">(Hamilton-Jacobi-Bellman Equation)</span></p>

The value function satisfies the **Hamilton-Jacobi-Bellman equations**, defined as:

$$
\partial_t V_t(\boldsymbol{x}) = \sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\left(e^{V_t(\boldsymbol{x}) - V_t(\boldsymbol{y})} - 1\right) \iff \partial_t e^{-V_t(\boldsymbol{x})} = \sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\left(e^{-V_t(\boldsymbol{x})} - e^{-V_t(\boldsymbol{y})}\right)
$$

</div>

*Proof.* To prove this, we substitute the final form of the optimal generator $\boldsymbol{Q}\_t^\star = \boldsymbol{Q}\_t^0 e^{V\_t(\boldsymbol{x}) - V\_t(\boldsymbol{y})}$ defined in the Optimal Generator into the equation for $\partial\_t V\_t$:

$$
\partial_t V_t(\boldsymbol{x}) = -\sum_{\boldsymbol{y} \ne \boldsymbol{x}} \left[\boldsymbol{Q}_t^\star \log\frac{\boldsymbol{Q}_t^\star}{\boldsymbol{Q}_t^0} - \boldsymbol{Q}_t^\star + \boldsymbol{Q}_t^0 + \boldsymbol{Q}_t^\star(V_t(\boldsymbol{y}) - V_t(\boldsymbol{x}))\right]
$$

$$
= -\sum_{\boldsymbol{y} \ne \boldsymbol{x}} \left[\boldsymbol{Q}_t^0 e^{V_t(\boldsymbol{x}) - V_t(\boldsymbol{y})}(V_t(\boldsymbol{x}) - V_t(\boldsymbol{y})) - \boldsymbol{Q}_t^0 e^{V_t(\boldsymbol{y}) - V_t(\boldsymbol{x})} + \boldsymbol{Q}_t^0 + \boldsymbol{Q}_t^0 e^{V_t(\boldsymbol{x}) - V_t(\boldsymbol{y})}(V_t(\boldsymbol{y}) - V_t(\boldsymbol{x}))\right]
$$

$$
= -\sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t^0\left(1 - e^{V_t(\boldsymbol{x}) - V_t(\boldsymbol{y})}\right) = \sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\left(e^{V_t(\boldsymbol{x}) - V_t(\boldsymbol{y})} - 1\right)
$$

which gives us the first HJB equation. To get the second expression, we can differentiate $e^{-V\_t(\boldsymbol{x})}$ and apply the chain rule to get:

$$
\partial_t e^{-V_t(\boldsymbol{x})} = e^{-V_t(\boldsymbol{x})} \partial_t V_t(\boldsymbol{x}) = e^{-V_t(\boldsymbol{x})} \sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\left(e^{V_t(\boldsymbol{x}) - V_t(\boldsymbol{y})} - 1\right) = \sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\left(e^{-V_t(\boldsymbol{y})} - e^{-V_t(\boldsymbol{x})}\right)
$$

which is the second HJB equation in the Corollary. $\square$

Using this Corollary, we can derive the **optimal path measure**, which can be defined in terms of the value function $V\_t(\boldsymbol{x})$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.9</span><span class="math-callout__name">(Optimal Path Measure)</span></p>

The optimal path measure $\mathbb{P}^\star$ given the value function $V\_t(\boldsymbol{x})$ can be expressed as:

$$
p_t^\star(\boldsymbol{x}) = \frac{1}{Z_t} q_t(\boldsymbol{x}) e^{-V_t(\boldsymbol{x})}, \quad Z_t := \mathbb{E}_{\boldsymbol{x} \sim q_t}\left[e^{-V_t(\boldsymbol{x})}\right]
$$

</div>

*Proof.* From Remark 7.7, we can consider the probability of a state $\boldsymbol{x} \in \mathcal{X}$ under the optimal path measure as its probability under the reference measure tilted by the $h$-function defined as $h(\boldsymbol{x}, t) := e^{-V\_t(\boldsymbol{x})}$ which yields $\xi\_t(\boldsymbol{x}) = \frac{1}{Z}q\_t(\boldsymbol{x})e^{-V\_t(\boldsymbol{x})}$, where $Z$ is the normalization factor. To show that $\xi(\boldsymbol{x})$ is indeed the optimal path measure, we must check that it satisfies **Kolmogorov's forward equation** defined in Lemma 7.1 for the optimal generator $\boldsymbol{Q}\_t^\star$. First, taking the partial derivative, we get:

$$
\partial_t \xi_t(\boldsymbol{x}) = \frac{1}{Z}\left(e^{-V_t(\boldsymbol{x})}\partial_t q_t(\boldsymbol{x}) + q_t(\boldsymbol{x})\partial_t e^{-V_t(\boldsymbol{x})}\right)
$$

Applying the Kolmogorov forward equation from Lemma 7.1 to the reference path measure $\mathbb{Q}$ and the HJB equations from Corollary 7.8, we have:

$$
\partial_t p_t^0(\boldsymbol{x}) = \sum_{\boldsymbol{x} \ne \boldsymbol{y}} (\boldsymbol{Q}_t^0(\boldsymbol{y}, \boldsymbol{x})q_t(\boldsymbol{y}) - \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})q_t(\boldsymbol{x}))
$$

$$
\partial_t e^{-V_t(\boldsymbol{x})} = \sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\left(e^{-V_t(\boldsymbol{x})} - e^{-V_t(\boldsymbol{y})}\right)
$$

and substituting this back, we get:

$$
\partial_t \xi_t(\boldsymbol{x}) = \frac{1}{Z}\left[e^{-V_t(\boldsymbol{x})}\sum_{\boldsymbol{x} \ne \boldsymbol{y}} (\boldsymbol{Q}_t^0(\boldsymbol{y}, \boldsymbol{x})q_t(\boldsymbol{y}) - \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})q_t(\boldsymbol{x})) + q_t(\boldsymbol{x})\sum_{\boldsymbol{y} \ne \boldsymbol{x}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\left(e^{-V_t(\boldsymbol{x})} - e^{-V_t(\boldsymbol{y})}\right)\right]
$$

Expanding, cancelling matching terms, and recognizing $\frac{1}{Z}q\_t(\boldsymbol{y})e^{-V\_t(\boldsymbol{y})} = \xi\_t(\boldsymbol{y})$ and $\boldsymbol{Q}\_t^0(\boldsymbol{y}, \boldsymbol{x})e^{V\_t(\boldsymbol{x}) - V\_t(\boldsymbol{y})} = \boldsymbol{Q}\_t^\star(\boldsymbol{y}, \boldsymbol{x})$:

$$
\partial_t \xi_t(\boldsymbol{x}) = \sum_{\boldsymbol{x} \ne \boldsymbol{y}} (\boldsymbol{Q}_t^\star(\boldsymbol{y}, \boldsymbol{x})\xi_t(\boldsymbol{y}) - \boldsymbol{Q}_t^\star(\boldsymbol{x}, \boldsymbol{y})\xi_t(\boldsymbol{x}))
$$

which is exactly the Kolmogorov forward equation for the generator $\boldsymbol{Q}\_t^\star$ that defines the optimal path measure $\mathbb{P}^\star$. Since we prove in Lemma 7.1 that the solution to the Kolmogorov forward equation is **unique**, we have shown that $p\_t^\star(\boldsymbol{x}) = \frac{1}{Z\_t}q\_t(\boldsymbol{x})e^{-V\_t(\boldsymbol{x})}$. We derive $Z\_t$ such that the probability distribution is normalized, i.e. $\sum\_{\boldsymbol{x} \in \mathcal{X}} p\_t^\star(\boldsymbol{x}) = 1$:

$$
Z_t = \sum_{\boldsymbol{x} \in \mathcal{X}} q_t(\boldsymbol{x}) e^{-V_t(\boldsymbol{x})} = \mathbb{E}_{\boldsymbol{x} \sim q_t}\left[e^{-V_t(\boldsymbol{x})}\right]
$$

which concludes our proof of the optimal path measure. $\square$

Using this form of the optimal path measure $p\_t^\star(\boldsymbol{x}) = \frac{1}{Z}q\_t(\boldsymbol{x})e^{-V\_t(\boldsymbol{x})}$, we can now derive the RND between the optimal and reference path measures.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.10</span><span class="math-callout__name">(Radon-Nikodym Derivative of Optimal and Reference Path Measure)</span></p>

The **Radon-Nikodym Derivative** (RND) of the optimal path measure $\mathbb{P}^\star$ with generator $\boldsymbol{Q}\_t^\star$ and the reference path measure $\mathbb{Q}$ with generator $\boldsymbol{Q}\_t^0$ is given by:

$$
\frac{d\mathbb{P}^\star}{d\mathbb{Q}}(\boldsymbol{X}_{0:T}) = \frac{1}{Z}e^{-\Phi(\boldsymbol{X}_T)}, \quad Z := \mathbb{E}_{\boldsymbol{x} \sim q_T}\left[e^{-\Phi(\boldsymbol{X}_T)}\right]
$$

where $\Phi(\boldsymbol{x}) : \mathcal{X} \to \mathbb{R}$ is the terminal cost function.

</div>

*Proof.* Starting from the definition of RND between two CTMC path measures in Proposition 7.4, we can write:

$$
\log\frac{d\mathbb{P}^\star}{d\mathbb{Q}}(\boldsymbol{X}_{0:T}) = \log\frac{p_0^\star(\boldsymbol{X}_0)}{q_0(\boldsymbol{X}_0)} + \sum_{t: \boldsymbol{X}_{t^-} \ne \boldsymbol{X}_t} \log\frac{\boldsymbol{Q}_t^\star(\boldsymbol{X}_{t^-}, \boldsymbol{X}_t)}{\boldsymbol{Q}_t^0(\boldsymbol{X}_{t^-}, \boldsymbol{X}_t)} + \int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} (\boldsymbol{Q}_t^0 - \boldsymbol{Q}_t^\star)(\boldsymbol{X}_t, \boldsymbol{y}) dt
$$

Now, we can substitute the expression for the optimal path probability $p\_0^\star(\boldsymbol{x}) = \frac{1}{Z}q\_0(\boldsymbol{x})e^{-V\_0(\boldsymbol{x})}$ from Proposition 7.9 and the optimal generator $\boldsymbol{Q}^\star(\boldsymbol{x}, \boldsymbol{y}) = \boldsymbol{Q}\_t^0(\boldsymbol{x}, \boldsymbol{y})e^{V\_t(\boldsymbol{x}) - V\_t(\boldsymbol{y})}$ from Proposition 7.6 to get:

$$
\log\frac{d\mathbb{P}^\star}{d\mathbb{Q}}(\boldsymbol{X}_{0:T}) = -V_0(\boldsymbol{X}_0) - \log Z_0 + \sum_{t: \boldsymbol{X}_{t^-} \ne \boldsymbol{X}_t} (V_t(\boldsymbol{X}_{t^-}) - V_t(\boldsymbol{X}_t)) + \int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \boldsymbol{Q}_t^0(\boldsymbol{X}_t, \boldsymbol{y})\left(1 - e^{V_t(\boldsymbol{X}_t) - V_t(\boldsymbol{y})}\right) dt
$$

Since a CTMC is a piecewise càdlàg function, we can define jump times $0 = t\_0 < t\_1 < \cdots < t\_k < \cdots < t\_{K-1} < t\_K = T$. Then, we have that in the time interval $[t\_k, t\_{k+1}]$ the state $\boldsymbol{X}\_{t\_k}$ stays fixed until the left limit $\boldsymbol{X}\_{t\_{k+1}^-}$ where it jumps to state $\boldsymbol{X}\_{t\_{k+1}}$. Therefore, we can define the value difference over all time steps as the sum of changes in value at state $\boldsymbol{X}\_{t\_k}$ over the time interval $[t\_k, t\_{k+1}]$ and the value change over the jump between states $\boldsymbol{X}\_{t\_k}$ and $\boldsymbol{X}\_{t\_{k+1}}$:

$$
V_T(\boldsymbol{X}_T) - V_0(\boldsymbol{X}_0) = \int_0^T \partial_t V_t(\boldsymbol{X}_t) dt + \sum_{t: \boldsymbol{X}_{t^-} \ne \boldsymbol{X}_t} (V_t(\boldsymbol{X}_t) - V_t(\boldsymbol{X}_{t^-}))
$$

Isolating $V\_0(\boldsymbol{X}\_0)$ and substituting the HJB equation from Corollary 7.8, we get:

$$
-V_0(\boldsymbol{X}_0) = -V_T(\boldsymbol{X}_T) + \int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \boldsymbol{Q}_t^0(\boldsymbol{X}_t, \boldsymbol{y})\left(e^{V_t(\boldsymbol{X}_t) - V_t(\boldsymbol{y})} - 1\right) dt + \sum_{t: \boldsymbol{X}_{t^-} \ne \boldsymbol{X}_t} (V_t(\boldsymbol{X}_t) - V_t(\boldsymbol{X}_{t^-}))
$$

Finally, substituting back into the log RND expression and cancelling terms, we get:

$$
\log\frac{d\mathbb{P}^\star}{d\mathbb{Q}}(\boldsymbol{X}_{0:T}) = -V_T(\boldsymbol{X}_T) - \log Z_0 \implies \frac{d\mathbb{P}^\star}{d\mathbb{Q}}(\boldsymbol{X}_{0:T}) = \frac{1}{Z_0}e^{-V_T(\boldsymbol{X}_T)}
$$

which yields the form of the RND between the CTMC that solves the SOC problem and the reference path measure. Substituting $V\_T(\boldsymbol{X}\_T) = \Phi(\boldsymbol{X}\_T)$ yields the final result. $\square$

Having derived the stochastic optimal control formulation for CTMCs, we have shown that the optimal dynamics arise from an exponential tilting of the reference generator and that the resulting optimal path measure $\mathbb{P}^\star$ admits a simple Radon-Nikodym derivative with respect to the reference CTMC $\mathbb{Q}$. In particular, the likelihood ratio depends only on the terminal value function $V\_T(\boldsymbol{X}\_T)$, revealing that the SOC solution can be interpreted as an entropy-regularized change of measure on path space.

From this perspective, solving the SOC problem is equivalent to computing a KL projection of path measures, where the optimal controlled process is the closest process to the reference dynamics that satisfies the desired terminal value constraint. This observation provides a direct connection to the Schroedinger bridge problem. In the next section, we explicitly formulate discrete Schroedinger bridges using the SOC framework, allowing the objectives developed in Section 3.3 to be adapted to discrete state spaces and enabling practical algorithms for learning optimal CTMC dynamics.

### 7.4 Discrete Schroedinger Bridges with Stochastic Optimal Control

Given our construction of SOC for CTMCs in Section 7.3, we can now explicitly write the Discrete SB Problem in the form of a stochastic optimal control functional, which contains a running cost that corresponds to the KL divergence between the controlled bridge measure $\mathbb{P}^u$ and the reference measure $\mathbb{Q}$ and a terminal cost that ensures the optimal process satisfies the terminal constraint $\boldsymbol{X}\_T \sim \pi\_T$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.11</span><span class="math-callout__name">(Discrete Schroedinger Bridge Problem with SOC)</span></p>

Consider the discrete SB problem where $\boldsymbol{Q}\_t^0$ is the generator of the reference CTMC $\mathbb{Q}$ and $\pi\_0, \pi\_T \in \mathcal{P}(\mathcal{X})$ are the initial and terminal constraints on the finite state space $\mathcal{X}$. The **discrete Schroedinger bridge problem** can be formulated as a **stochastic optimal control** (SOC) problem which seeks a controlled generator $\boldsymbol{Q}\_t^u$ of a controlled CTMC $\mathbb{P}^u$ that minimizes:

$$
\inf_{\mathbb{P}^u} \mathbb{E}_{\boldsymbol{X}_{0:T}^u \sim \mathbb{P}^u}\left[\int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t^u} C_t(\boldsymbol{X}_t^u, \boldsymbol{y}) dt + \Phi(\boldsymbol{X}_T^u)\right], \quad \boldsymbol{X}_0^u \sim \pi_0
$$

where the running cost $C\_t(\boldsymbol{x}, \boldsymbol{y}) : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ and terminal cost $\Phi(\boldsymbol{x}) : \mathcal{X} \to \mathbb{R}$ are defined as:

$$
C_t(\boldsymbol{X}_t^u, \boldsymbol{y}) := \text{KL}(p_t^u \| q_t) = \left(\boldsymbol{Q}_t^u \log\frac{\boldsymbol{Q}_t^u}{\boldsymbol{Q}_t^0} - \boldsymbol{Q}_t^u + \boldsymbol{Q}_t^0\right)(\boldsymbol{X}_t^u, \boldsymbol{y}), \quad \Phi(\boldsymbol{X}_T^u) = \log\frac{\hat{\varphi}(\boldsymbol{X}_T^u)}{\pi_T(\boldsymbol{X}_T^u)}
$$

The optimal controlled process $\mathbb{P}^\star$ defines the Schroedinger bridge between $\pi\_0$ and $\pi\_T$ relative to the reference dynamics $\boldsymbol{Q}\_t^0$.

</div>

Now that we have established that the SOC problem can be solved in the discrete state space, we can easily adapt the objectives defined in Section 3.3 to solve the SB-SOC problem for discrete variables.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.12</span><span class="math-callout__name">(Discrete Relative Entropy (RE) Loss)</span></p>

The **relative entropy** (RE) loss between the controlled path measure $\mathbb{P}^u$ and the optimal path measure $\mathbb{P}^\star$ is defined as the KL divergence:

$$
\mathcal{L}_{\text{RE}}(\boldsymbol{Q}^u) := \mathbb{E}_{\boldsymbol{X}_{0:T}^u \sim \mathbb{P}^u}\left[\int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t^u} \left(\boldsymbol{Q}_t^u \log\frac{\boldsymbol{Q}_t^u}{\boldsymbol{Q}_t^0} + \boldsymbol{Q}_t^0 - \boldsymbol{Q}_t^u\right)(\boldsymbol{X}_t^u, \boldsymbol{y}) + V_T(\boldsymbol{X}_T^u) + \log Z_0\right]
$$

which can be written in terms of the controlled and reference generator $\boldsymbol{Q}^u$ and $\boldsymbol{Q}^0$.

</div>

As discussed in Section 3.3, the Discrete RE Objective requires backpropagating through the full stochastic trajectory simulations, so we can use the RERF Loss as a practical surrogate loss for the Discrete RE Objective.

Next, we will adapt the CE Loss to the discrete state space. Similarly to Definition 3.15, the expectation over the optimal path measure $\mathbb{P}^\star$ is generally intractable during training, so we define the expectation over an arbitrary path measure $\mathbb{P}^v$ and **reweighting** by the RND between $\mathbb{P}^\star$ and $\mathbb{P}^v$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.13</span><span class="math-callout__name">(Cross-Entropy Objective)</span></p>

The **cross-entropy** (CE) loss that optimizes a controlled path measure $\mathbb{P}^u$ to match the optimal path measure $\mathbb{P}^\star$ is defined as the reverse KL divergence:

$$
\mathcal{L}_{\text{CE}}(\mathbb{P}^u, \mathbb{P}^\star) := \text{KL}(\mathbb{P}^\star \| \mathbb{P}^u) = \mathbb{E}_{\mathbb{P}^v}\left[\frac{d\mathbb{P}^\star}{d\mathbb{P}^v}\left(\log\frac{d\mathbb{P}^\star}{d\mathbb{Q}} + \log\frac{d\mathbb{Q}}{d\mathbb{P}^u}\right)\right]
$$

where $\log\frac{d\mathbb{P}^\star}{d\mathbb{Q}}$ vanishes in the gradient as it is an additive constant with respect to $\mathbb{P}^u$. The CE objective can be written in terms of the controlled and reference generator $\boldsymbol{Q}^u$ and $\boldsymbol{Q}^0$ as:

$$
\mathcal{L}_{\text{CE}}(\boldsymbol{Q}^u) = \frac{1}{Z}\mathbb{E}_{\boldsymbol{X}_{0:T}^v \sim \mathbb{P}^v}\left[e^{W(\boldsymbol{X}_{0:T}^v)}\left(\int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t^v} \left(\boldsymbol{Q}_t^0 \log\frac{\boldsymbol{Q}_t^0}{\boldsymbol{Q}_t^u} + \boldsymbol{Q}_t^u - \boldsymbol{Q}_t^0\right)(\boldsymbol{X}_t, \boldsymbol{y})\right)\right] + C
$$

where $W(\boldsymbol{X}\_{0:T}^v) := \log\frac{d\mathbb{P}^\star}{d\mathbb{P}^v}(\boldsymbol{X}\_{0:T}^v)$ is a weight that can be expanded as:

$$
W(\boldsymbol{X}_{0:T}^v) = \int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t^v} \left(\boldsymbol{Q}_t^v \log\frac{\boldsymbol{Q}_t^v}{\boldsymbol{Q}_t^0} + \boldsymbol{Q}_t^0 - \boldsymbol{Q}_t^v\right)(\boldsymbol{X}_t^v, \boldsymbol{y}) + V_T(\boldsymbol{X}_T^v) + \log Z_0
$$

The sampling law is commonly defined in practice as the stop-gradient sampling measure $\mathbb{P}^{\bar{u}}$, where $\boldsymbol{Q}^{\bar{u}} := \text{stopgrad}(\boldsymbol{Q}^u)$ is the non-gradient-tracking controlled generator.

</div>

The final objective introduced for continuous SOC in Section 3.3 is the **variance and log-variance objectives**. Recall that the RND between the optimal and controlled path measure is high when the two distributions are dissimilar. On the other hand, the variance is low when the distributions are similar and is minimized at zero exactly when the RND evaluates to a constant regardless of the stochastic path, indicating that the two measures are equal. Since the variance objective is generally unstable, we only define the log-variance objective below.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.14</span><span class="math-callout__name">(Log-Variance Objectives)</span></p>

The **log-variance** (LV) loss that optimizes a controlled path measure $\mathbb{P}^u$ to match the optimal path measure $\mathbb{P}^\star$ is defined as:

$$
\mathcal{L}_{\text{LV}}(\mathbb{P}^u, \mathbb{P}^\star) := \text{Var}_{\mathbb{P}^v}\left(\log\frac{d\mathbb{P}^\star}{d\mathbb{P}^u}\right) = \text{Var}_{\mathbb{P}^v}\left(\log\frac{d\mathbb{P}^\star}{d\mathbb{Q}} - \log\frac{d\mathbb{P}^u}{d\mathbb{Q}}\right)
$$

which can be written in terms of the controlled and reference generator $\boldsymbol{Q}^u$ and $\boldsymbol{Q}^0$ as:

$$
\mathcal{L}_{\text{LV}}(\boldsymbol{Q}^u) = \text{Var}_{\boldsymbol{X}_{0:T}^v \sim \mathbb{P}^v}\left(\int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t^v} \left(\boldsymbol{Q}_t^u \log\frac{\boldsymbol{Q}_t^u}{\boldsymbol{Q}_t^0} + \boldsymbol{Q}_t^0 - \boldsymbol{Q}_t^u\right)(\boldsymbol{X}_t^v, \boldsymbol{y}) + V_T(\boldsymbol{X}_T^v) + \log Z_0\right)
$$

</div>

These objectives allow us to tractably solve the Discrete SB-SOC Objective by simulating the controlled CTMC over discrete time steps while simultaneously evaluating the transition rates of each jump under the reference generator $\boldsymbol{Q}^0$, computing the SOC objective using the time-discretized form of the log RND, and optimizing $\boldsymbol{u}$ using standard gradient descent. For uniform discrete time steps $0 = t\_0 < \cdots < t\_k < \cdots < t\_K = T$ with increments $\Delta t$, the log RND over the discrete states $(\boldsymbol{X}\_{t\_k})\_{k \in \lbrace 0, \ldots, K \rbrace}$ used to compute the losses can be computed as:

$$
\log\frac{d\mathbb{P}^u}{d\mathbb{Q}}((\boldsymbol{X}_{t_k})_{k \in \lbrace 0, \ldots, K \rbrace}) = \sum_{k=0}^{K-1}\left[\Delta t \left(\boldsymbol{Q}_{t_k}^u \log\frac{\boldsymbol{Q}_{t_k}^u}{\boldsymbol{Q}_{t_k}^0} + \boldsymbol{Q}_{t_k}^0 - \boldsymbol{Q}_{t_k}^u\right)(\boldsymbol{X}_{t_k}, \boldsymbol{X}_{t_{k+1}})\right] + O(\Delta t^2)
$$

Then, we can define a general form of the Discrete SB-SOC training procedure below:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Discrete SB-SOC Training Framework)</span></p>

Training a parameterized generator $\boldsymbol{Q}^u$ using one of the SOC objectives defined above typically iterates through the following steps:

- **(i)** Sample discretized trajectories $(\boldsymbol{X}\_{t\_k}^v)\_{t \in \lbrace 0, \ldots, K \rbrace}$ using the proposal generator $\boldsymbol{Q}^v$ for $K$ steps. Alternatively, if the proposal generator is not $\boldsymbol{u}$ and trajectories can be reused, then samples can be stored and subsequently sampled from a replay buffer $\mathcal{B}$.
- **(ii)** For each trajectory, compute the transition rates $\boldsymbol{Q}\_{t\_k}^0(\boldsymbol{X}\_{t\_k}, \boldsymbol{X}\_{t\_{k+1}})$ and $\boldsymbol{Q}\_{t\_k}^u(\boldsymbol{X}\_{t\_k}, \boldsymbol{X}\_{t\_{k+1}})$ over every interval $[t\_k, t\_{k+1}]$.
- **(iii)** Compute the Discretized Log RND for each trajectory and use it to compute one of the SOC losses $\mathcal{L}(\boldsymbol{Q}^u)$.
- **(iv)** Optimize the parameterized generator $\boldsymbol{Q}^u$ using gradient steps of $\nabla\_{\boldsymbol{u}}\mathcal{L}(\boldsymbol{Q}^u)$.
- **(v)** Repeat from Step (i).

In practice, the proposal generator $\boldsymbol{Q}^v$ is commonly defined as the same generator being optimized $\boldsymbol{Q}^u$ but without gradient-tracking, i.e., $\boldsymbol{Q}^v := \boldsymbol{Q}^{\bar{u}} = \text{stopgrad}(\boldsymbol{Q}^u)$.

</div>

The SOC framework provides a practical procedure for learning the controlled generator $\boldsymbol{Q}^u$ of a CTMC through sampling discrete trajectories from a proposal generator $\boldsymbol{Q}^v$ and optimizing the SOC objective with respect to $\boldsymbol{Q}^u$. While this process is well-suited for tasks where we have no access to samples from the target distribution $\pi\_T$ but only a way of evaluating the likelihood of a sample under it, optimizing a Schroedinger bridge on *paired samples* or an explicit optimal transport map requires a different approach. Next, we will describe the extension of Markovian and reciprocal projections in the discrete state space, which allows us to optimize discrete Schroedinger bridges directly on samples from the initial and target distributions, or pairs from an optimal transport coupling $\pi\_{0,T}^\star$.

### 7.5 Discrete Markov and Reciprocal Projections

In this section, we extend the theory of Markov and reciprocal projections to discrete state spaces, where stochastic processes are characterized by the generators of continuous-time Markov chains (CTMCs). Recall from Section 4.5 that the **Markovian projection** identifies the Markov process that is closest in KL divergence to a given reciprocal process within the class of Markov dynamics, while the **reciprocal projection** enforces the endpoint constraints through conditioning on the boundary distributions.

$$
\mathbb{M}^\star := \text{proj}_{\mathcal{M}}(\Pi) = \arg\min_{\mathbb{M} \in \mathcal{M}} \text{KL}(\Pi \| \mathbb{M})
$$

$$
\Pi^\star := \mathbb{Q}_{\cdot|0,T}\mathbb{M}_{0,T} = \text{proj}_{\mathcal{R}(\mathbb{Q})}(\mathbb{P}) = \int_{\mathbb{R}^d \times \mathbb{R}^d} \mathbb{Q}_{\cdot|0,T}(\cdot \mid \boldsymbol{x}_0, \boldsymbol{x}_T) d\mathbb{M}_{0,T}(\boldsymbol{x}_0, \boldsymbol{x}_T)
$$

We will derive the explicit form of the Markovian and reciprocal projections in terms of the generators of CTMCs, which will serve as the theoretical basis for extending the iterative Markovian fitting (IMF) procedure from Section 6.3 to the finite state space.

- **(i)** First, we define the Markov projection $\mathbb{M}^\star$ of a CTMC given a bridge measure in the reciprocal class $\Pi \in \mathcal{R}(\mathbb{Q})$. We define the explicit form of its generator $\boldsymbol{Q}^{\mathbb{M}^\star}$ and the KL divergence with the reciprocal process (Definition 7.15).
- **(ii)** We derive the definition for the reference generator $\boldsymbol{Q}\_t^0$ conditioned on a terminal state $\boldsymbol{x}\_T$, which appears in the expression for the generator of the Markovian projection (Lemma 7.16).
- **(iii)** We show that conditioning the bridge measure in the reciprocal class $\Pi \in \mathcal{R}(\mathbb{Q})$ on an initial state $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$ yields a Markov measure, which we denote $\Pi^{|0 = \boldsymbol{x}\_0}$ (Lemma 7.17).
- **(iv)** We derive the form of the KL divergence $\text{KL}(\Pi \| \mathbb{M})$ using the generator of the conditioned reciprocal process $\boldsymbol{Q}\_t^{\Pi^{|0 = \boldsymbol{x}\_0}}$ and the generator of the Markov process $\mathbb{M}$ (Lemma 7.18).
- **(v)** Finally, we define the **reverse-time** Markovian projection, which will allow us to condition on both the initial distribution $\pi\_0$ and target distribution $\pi\_T$, reducing error accumulation during the IMF procedure (Definition 7.19).

We start by defining the **Markovian projection** $\mathbb{M}^\star$ **of a CTMC path measure**, its explicit form in terms of the generator matrices and the KL minimization objective that yields $\mathbb{M}^\star$ given a reciprocal measure $\Pi$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.15</span><span class="math-callout__name">(Markovian Projection of CTMC Path Measure)</span></p>

Consider a reference path measure $\mathbb{Q}$ with generator $\boldsymbol{Q}\_t^0$ and a measure $\Pi \in \mathcal{R}(\mathbb{Q})$ in the reciprocal class of $\mathbb{Q}$ that preserves the bridge. The **Markovian projection** $\mathbb{M}^\star := \text{proj}\_{\mathcal{M}}(\Pi)$ has a generator $\boldsymbol{Q}\_t^{\mathbb{M}^\star}$ defined as the expectation over **conditional generators** $\boldsymbol{Q}\_t^0(\cdot, \cdot; \boldsymbol{x})$ under the reference process $\mathbb{Q}$ of the form:

$$
\boldsymbol{Q}_t^{\mathbb{M}^\star}(\boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_T \sim \mathbb{Q}_{T|t}(\cdot|\boldsymbol{x})}\left[\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

where each conditional generator is defined as:

$$
\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) = \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\frac{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{y})}{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{x})} - \mathbf{1}_{\boldsymbol{x} = \boldsymbol{y}}\sum_{\boldsymbol{z} \in \mathcal{X}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{z})\frac{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{z})}{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{x})}
$$

where $\mathbb{Q}\_{T|t}(\cdot \mid \cdot)$ is the conditional transition probability under the reference measure $\mathbb{Q}$. We can also define the KL divergence between $\Pi$ and its Markovian projection $\mathbb{M}^\star$ using Corollary 7.5 as:

$$
\text{KL}(\Pi \| \mathbb{M}^\star) = \int_0^T \mathbb{E}_{\Pi_{0,t}}\left[\sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \left(\boldsymbol{Q}_t^{\Pi^{|0}}\log\frac{\boldsymbol{Q}_t^{\Pi^{|0}}}{\boldsymbol{Q}_t^{\mathbb{M}^\star}} + \boldsymbol{Q}_t^{\mathbb{M}^\star} - \boldsymbol{Q}_t^{\Pi^{|0}}\right)(\boldsymbol{X}_t, \boldsymbol{y})\right] dt
$$

where we define $\Pi^{|0 = \boldsymbol{x}\_0}$ as the conditional bridge measure with generator $\boldsymbol{Q}\_t^{\Pi^{|0 = \boldsymbol{x}\_0}}$ defined as:

$$
\boldsymbol{Q}_t^{\Pi^{|0 = \boldsymbol{x}_0}}(\boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_T \sim \Pi_{T|0,t}}\left[\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) \mid \boldsymbol{X}_0 = \boldsymbol{x}_0, \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

where for any $t \in [0, T]$, the marginal distributions match $\mathbb{M}\_t = \Pi\_t$.

</div>

Breaking down Definition 7.15, we introduce several *unfamiliar* definitions, including the **endpoint-conditioned generator** of the reference process $\boldsymbol{Q}\_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}\_T)$, the reciprocal bridge measure $\Pi$ conditioned on $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$ denoted $\Pi^{|0 = \boldsymbol{x}\_0}$ and its **generator** defined as $\boldsymbol{Q}\_t^{\Pi^{|0 = \boldsymbol{x}\_0}}$. In the following sequence of Lemmas, we will break down these ideas more concretely to better understand the discrete analogue of the Markovian projection.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.16</span><span class="math-callout__name">(Conditional Generator of Markov Process)</span></p>

Consider a CTMC $\boldsymbol{X}\_{0:T}$ under the reference path measure $\mathbb{Q}$ with generator $\boldsymbol{Q}\_t^0$. Conditioning $\mathbb{Q}$ on a terminal state $\boldsymbol{X}\_T = \boldsymbol{x}\_T$ gives the conditioned path measure, denoted $\mathbb{Q}^{|T = \boldsymbol{x}\_T}$, which is **Markov** and is defined by the generator:

$$
\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) = \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\frac{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{y})}{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{x})} - \mathbf{1}_{\boldsymbol{x} = \boldsymbol{y}}\sum_{\boldsymbol{z} \in \mathcal{X}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{z})\frac{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{z})}{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{x})}
$$

</div>

*Proof.* To confirm the Markov property of $\mathbb{Q}^{|T = \boldsymbol{x}\_T}$, we first apply Bayes' rule to get:

$$
\mathbb{Q}_{t|s}(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{x}_T) = \frac{\mathbb{Q}_{t|s}(\boldsymbol{y} \mid \boldsymbol{x})\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{y})}{\mathbb{Q}_{T|s}(\boldsymbol{x}_T \mid \boldsymbol{x})} =: \mathbb{Q}_{t|s}^{|T = \boldsymbol{x}_T}(\boldsymbol{y} \mid \boldsymbol{x})
$$

which defines the transition kernel of the bridge process conditioned on $\boldsymbol{X}\_T = \boldsymbol{x}\_T$. This shows that the conditioned process is still Markov, that is given the current state $\boldsymbol{X}\_T = \boldsymbol{x}\_T$ and the fixed endpoint $\boldsymbol{X}\_T = \boldsymbol{x}\_T$, the law of the future state $\boldsymbol{X}\_t$ depends on the past only through the current state $\boldsymbol{X}\_s$.

Now, applying the **Kolmogorov forward equation** from Lemma 7.1 and the **Kolmogorov backward equation** from Lemma 7.2, we obtain the generator of the conditioned process as:

$$
\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) = \partial_s \mathbb{Q}_{t|s}^{|T = \boldsymbol{x}_T}(\boldsymbol{y} \mid \boldsymbol{x})\big|_{s = t} = \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y})\frac{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{y})}{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{x})} - \mathbf{1}_{\boldsymbol{x} = \boldsymbol{y}}\sum_{\boldsymbol{z} \in \mathcal{X}} \boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{z})\frac{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{z})}{\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{x})}
$$

which recovers the form of the generator $\boldsymbol{Q}\_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}\_T)$ given the terminal state $\boldsymbol{X}\_T = \boldsymbol{x}\_T$ under the reference process $\mathbb{Q}$. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.17</span><span class="math-callout__name">(KL Divergence Between Reciprocal and Markov CTMCs)</span></p>

Given the reciprocal process $\Pi \in \mathcal{R}(\mathbb{Q})$, **conditioning** on an initial state $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$, yields a **Markov** process $\Pi^{|0 = \boldsymbol{x}\_0}$ that is defined by the generator:

$$
\boldsymbol{Q}_t^{\Pi^{|0 = \boldsymbol{x}_0}}(\boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_T \sim \Pi_{T|0,t}}\left[\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) \mid \boldsymbol{X}_0 = \boldsymbol{x}_0, \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

Then, for any Markov measure $\mathbb{M} \in \mathcal{M}$ where $\mathbb{M}\_0 = \Pi\_0$ and $\Pi \ll \mathbb{M}$, the KL divergence between $\Pi$ and $\mathbb{M}$ is defined as:

$$
\text{KL}(\Pi \| \mathbb{M}) = \mathbb{E}_{\Pi_0}\left[\text{KL}(\Pi^{|0 = \boldsymbol{x}_0} \| \mathbb{M}^{|0 = \boldsymbol{x}_0})\right]
$$

where $\text{KL}(\Pi^{|0 = \boldsymbol{x}\_0} \| \mathbb{M}^{|0 = \boldsymbol{x}\_0})$ is a KL divergence between the Markov conditioned reciprocal process $\Pi^{|0 = \boldsymbol{x}\_0}$ and the conditioned Markov measure $\mathbb{M}^{|0 = \boldsymbol{x}\_0}$, which expands to:

$$
\text{KL}(\Pi^{|0} \| \mathbb{M}^{|0}) = \int_0^T \mathbb{E}_{\Pi_{0,t}}\left[\sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \left(\boldsymbol{Q}_t^{\Pi^{|0}}\log\frac{\boldsymbol{Q}_t^{\Pi^{|0}}}{\boldsymbol{Q}_t^{\mathbb{M}^{|0}}}\right)(\boldsymbol{X}_t, \boldsymbol{y}) + \left(\boldsymbol{Q}_t^{\Pi^{|0}} - \boldsymbol{Q}_t^{\mathbb{M}^{|0}}\right)(\boldsymbol{X}_t, \boldsymbol{X}_t)\right] dt
$$

</div>

*Proof.* We prove each part of the Lemma in steps. First, we derive the generator of the conditioned reciprocal measure, which takes the form of an expectation of the conditional reference generators derived in Lemma 7.16.

**Step 1: Derive the Conditioned Generator.** First, we establish that the pinned down bridge measure $\Pi^{|0 = \boldsymbol{x}\_0}$ is in the reciprocal class $\mathcal{R}(\mathbb{Q})$ and satisfies the Reciprocal Property:

$$
\Pi^{|0 = \boldsymbol{x}_0} = \int \mathbb{Q}_{\cdot|0,T}(\cdot \mid \boldsymbol{x}_0, \boldsymbol{x}_T)\Pi_{T|0}(d\boldsymbol{x}_T \mid \boldsymbol{x}_0)
$$

We first write the transition probability between states under the conditional process using the law of total probability:

$$
\Pi_{t|s,0}^{|0 = \boldsymbol{x}_0}(\boldsymbol{y} \mid \boldsymbol{x}) = \sum_{\boldsymbol{x}_T \in \mathcal{X}} \Pi_{T|s}^{|0 = \boldsymbol{x}_0}(\boldsymbol{x}_T \mid \boldsymbol{x})\Pi_{t|s,T}^{|0 = \boldsymbol{x}_0}(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{x}_T)
$$

which samples the terminal state $\boldsymbol{X}\_T = \boldsymbol{x}\_T$ and then samples the intermediate time $\boldsymbol{X}\_t = \boldsymbol{y}$ from the bridge. Given that $\Pi \in \mathcal{R}(\mathbb{Q})$ is in the reciprocal class with a shared bridge measure as $\mathbb{Q}$, by the Reciprocal Property, the second term in the sum is equal to $\Pi\_{t|s,T}^{|0 = \boldsymbol{x}\_0}(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{x}\_T) = \mathbb{Q}\_{t|s,T}(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{x}\_T)$.

We can break down $\Pi\_{T|s,0}(\boldsymbol{x}\_T \mid \boldsymbol{x}, \boldsymbol{x}\_0)$ using Bayes' rule and applying the reciprocal property to get:

$$
\Pi_{T|s,0}(\boldsymbol{x}_T \mid \boldsymbol{x}, \boldsymbol{x}_0) = \frac{\mathbb{Q}_{s|0}(\boldsymbol{x} \mid \boldsymbol{x}_0)}{\mathbb{Q}_{T|0}(\boldsymbol{x}_T \mid \boldsymbol{x}_0)} \cdot \frac{\Pi_{T|0}(\boldsymbol{x}_T \mid \boldsymbol{x}_0)}{\Pi_{s|0}(\boldsymbol{x} \mid \boldsymbol{x}_0)} \cdot \mathbb{Q}_{T|s}(\boldsymbol{x}_T \mid \boldsymbol{x})
$$

Substituting back into the expression for $\Pi\_{t|s,0}^{|0 = \boldsymbol{x}\_0}(\boldsymbol{y} \mid \boldsymbol{x})$, cancelling terms and factoring out terms not dependent on $\boldsymbol{x}\_T$:

$$
\Pi_{t|s}^{|0 = \boldsymbol{x}_0}(\boldsymbol{y} \mid \boldsymbol{x}) = \frac{\mathbb{Q}_{s|0}(\boldsymbol{x} \mid \boldsymbol{x}_0)}{\Pi_{s|0}(\boldsymbol{x} \mid \boldsymbol{x}_0)}\mathbb{Q}_{t|s}(\boldsymbol{y} \mid \boldsymbol{x})\sum_{\boldsymbol{x}_T \in \mathcal{X}} \frac{\Pi_{T|0}(\boldsymbol{x}_T \mid \boldsymbol{x}_0)}{\mathbb{Q}_{T|0}(\boldsymbol{x}_T \mid \boldsymbol{x}_0)}\mathbb{Q}_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{y})
$$

To derive the generator using this expression, we take the time derivative evaluated at $t = s$ to get:

$$
\boldsymbol{Q}_s^{\Pi^{|0 = \boldsymbol{x}_0}}(\boldsymbol{x}, \boldsymbol{y}) = \partial_t \Pi_{t|s}^{|0 = \boldsymbol{x}_0}(\boldsymbol{y} \mid \boldsymbol{x})\big|_{t = s}
$$

Applying the **backward Kolmogorov equation** from Lemma 7.2 to $\partial\_t \mathbb{Q}\_{T|t}(\boldsymbol{x}\_T \mid \boldsymbol{y})$, and combining all terms, we arrive at:

$$
\boldsymbol{Q}_s^{\Pi^{|0 = \boldsymbol{x}_0}}(\boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_T \sim \Pi_{T|0,t}}\left[\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) \mid \boldsymbol{X}_0 = \boldsymbol{x}_0, \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

which gives us our final form for the conditional generator. $\square$

**Step 2: Derive the KL Divergence.** In Corollary 7.5, we derived the expression for the KL divergence between CTMCs, which relies on the Markov structure. To derive the KL divergence between a *non-Markov reciprocal process* $\Pi$ and a Markov measure $\mathbb{M}$, we apply the KL Divergence Chain Rule from Lemma 1.4 to decompose the KL divergence as the sum of the KL divergence of the initial marginal and the expectation of the KL divergence of the conditioned measures:

$$
\text{KL}(\Pi \| \mathbb{M}) = \underbrace{\text{KL}(\Pi_0 \| \mathbb{M}_0)}_{= 0 \; (\Pi_0 = \mathbb{M}_0)} + \mathbb{E}_{\Pi_0}\left[\text{KL}(\Pi^{|0 = \boldsymbol{x}_0} \| \mathbb{M}^{|0 = \boldsymbol{x}_0})\right] = \mathbb{E}_{\Pi_0}\left[\text{KL}(\Pi^{|0 = \boldsymbol{x}_0} \| \mathbb{M}^{|0 = \boldsymbol{x}_0})\right]
$$

where the KL between the initial marginals vanishes by our definition $\Pi\_0 = \mathbb{M}\_0$. We have shown in Proposition 4.11 that conditioning a reciprocal process on an endpoint $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$ yields a **Markov process** and the KL divergence between two Markov measures is defined in Corollary 7.5, we can further expand as:

$$
\text{KL}(\Pi \| \mathbb{M}) = \int_0^T \mathbb{E}_{\Pi_{0,t}}\left[\sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \left(\boldsymbol{Q}_t^{\Pi^{|0}}\log\frac{\boldsymbol{Q}_t^{\Pi^{|0}}}{\boldsymbol{Q}_t^{\mathbb{M}^{|0}}}\right)(\boldsymbol{X}_t, \boldsymbol{y}) + \left(\boldsymbol{Q}_t^{\Pi^{|0}} - \boldsymbol{Q}_t^{\mathbb{M}^{|0}}\right)(\boldsymbol{X}_t, \boldsymbol{X}_t)\right] ds
$$

which is exactly the form of the KL divergence between the reciprocal and Markov measures defined in the Lemma. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.18</span><span class="math-callout__name">(Generator of Markovian Projection)</span></p>

The Markovian projection $\mathbb{M}^\star = \text{proj}\_{\mathcal{M}}(\Pi)$ of the reciprocal measure $\Pi \in \mathcal{R}(\mathbb{Q})$ that minimizes the KL divergence $\text{KL}(\Pi \| \mathbb{M})$ is defined by the generator:

$$
\boldsymbol{Q}_t^{\mathbb{M}^\star}(\boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_T \sim \Pi_{T|t}}\left[\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

</div>

*Proof.* First, we assume that for all $\boldsymbol{x} \ne \boldsymbol{y}, \boldsymbol{x}\_T \in \mathcal{X}$ and $t \in [0, T)$, we have $\boldsymbol{Q}^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}\_T) = 0 \iff \boldsymbol{Q}\_t^0(\boldsymbol{x}, \boldsymbol{y}) = 0$. To prove that the candidate generator is the generator of $\mathbb{M}^\star$, we first decompose it as follows:

$$
\boldsymbol{Q}_t^{\mathbb{M}^\star}(\boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_T \sim \Pi_{T|t}}\left[\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) \mid \boldsymbol{X}_t = \boldsymbol{x}\right] = \sum_{\boldsymbol{x}_T \in \mathcal{X}} \Pi_{T|t}(\boldsymbol{x}_T \mid \boldsymbol{x})\boldsymbol{Q}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T)
$$

Since the reciprocal process depends on both the initial state in addition to the target state, we apply the law of total expectation to insert conditioning on $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$ to get:

$$
\boldsymbol{Q}_t^{\mathbb{M}^\star}(\boldsymbol{x}, \boldsymbol{y}) = \sum_{\boldsymbol{x}_0 \in \mathcal{X}} \Pi_{0|t}(\boldsymbol{x}_0 \mid \boldsymbol{x})\boldsymbol{Q}_t^{\Pi^{|0 = \boldsymbol{x}_0}}(\boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}_{\Pi_{0|t}}\left[\boldsymbol{Q}_t^{\Pi^{|0 = \boldsymbol{x}_0}}(\boldsymbol{X}_t, \boldsymbol{y}) \mid \boldsymbol{X}_t = \boldsymbol{x}\right]
$$

and we have shown that the generator $\boldsymbol{Q}\_t^{\mathbb{M}^\star}(\boldsymbol{x}, \boldsymbol{y})$ is the expectation of bridge generators conditioned on the initial state or a **mixture of bridges** evaluated at $\boldsymbol{x}$.

**Step 2: Show the Markov Generator Minimizes the KL Divergence.** Let $\mathbb{M} \in \mathcal{M}$ be an arbitrary Markov measure with generator defined as the Markovian Projection Generator, then by optimality of $\mathbb{M}^\star$, we have $\text{KL}(\Pi \| \mathbb{M}) \ge \text{KL}(\Pi \| \mathbb{M}^\star)$. We aim to show that this inequality reduces to an **equality** where $\text{KL}(\Pi \| \mathbb{M}) = \text{KL}(\Pi \| \mathbb{M}^\star)$.

Using the definition of $\text{KL}(\Pi \| \mathbb{M})$ from Lemma 7.17 and some algebra, we can write the difference $\text{KL}(\Pi \| \mathbb{M}^\star) - \text{KL}(\Pi \| \mathbb{M})$:

$$
\text{KL}(\Pi \| \mathbb{M}^\star) - \text{KL}(\Pi \| \mathbb{M}) = \int_0^T \sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \left(\boldsymbol{Q}_t^{\mathbb{M}^\star}\log\frac{\boldsymbol{Q}_t^{\mathbb{M}^\star}}{\boldsymbol{Q}_t^{\mathbb{M}}} + \boldsymbol{Q}_t^{\mathbb{M}} - \boldsymbol{Q}_t^{\mathbb{M}^\star}\right)(\boldsymbol{X}_t, \boldsymbol{y}) dt = \text{KL}(\mathbb{M} \| \mathbb{M}^\star) \ge 0
$$

where the key step $(\bigstar)$ uses $\mathbb{E}\_{\Pi\_{0|t}}\left[\boldsymbol{Q}^{\Pi^{|0 = \boldsymbol{x}\_0}} \mid \boldsymbol{X}\_t\right] = \boldsymbol{Q}\_t^{\mathbb{M}^\star}$ from Step 1. Since $\mathbb{M}^\star$ is optimal, by definition $\text{KL}(\Pi \| \mathbb{M}^\star) - \text{KL}(\Pi \| \mathbb{M}) \le 0$. However, we have shown that $\text{KL}(\Pi \| \mathbb{M}^\star) - \text{KL}(\Pi \| \mathbb{M}) = \text{KL}(\mathbb{M} \| \mathbb{M}^\star) \ge 0$ by non-negativity of the KL divergence. Therefore, we can conclude that $\text{KL}(\Pi \| \mathbb{M}^\star) = \text{KL}(\Pi \| \mathbb{M})$ and $\mathbb{M} = \mathbb{M}^\star$ almost surely. $\square$

Now, we have seen that the Markovian projection $\mathbb{M}^\star := \text{proj}\_{\mathcal{R}(\mathbb{Q})}(\Pi)$ which minimizes the KL divergence from a reciprocal measure $\Pi$ is explicitly defined with the Generator of Markovian Projection. However, recall from our discussion of Iterative Markovian Fitting from Section 6.3 that simulating the Markov SDE in the forward-time can accumulate errors in matching the target distribution $\pi\_T$, motivating the simulation a corresponding reverse-time Markovian projection. Therefore, we will also extend the idea of a reverse-time Markovian projection for CTMCs.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.19</span><span class="math-callout__name">(Reverse-Time Markovian Projection)</span></p>

The generator of the **reverse-time Markovian projection** $\tilde{\mathbb{M}}^\star$ is given by:

$$
\tilde{\boldsymbol{Q}}_t^{\tilde{\mathbb{M}}^\star}(\boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_0 \sim \mathbb{Q}_{0|t}(\cdot|\boldsymbol{x})}\left[\tilde{\boldsymbol{Q}}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_0) \mid \tilde{\boldsymbol{X}}_t = \boldsymbol{x}\right]
$$

where each conditional generator is defined as:

$$
\tilde{\boldsymbol{Q}}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_0) = \boldsymbol{Q}_{T-t}^0(\boldsymbol{x}, \boldsymbol{y})\frac{\mathbb{Q}_{T-t|0}(\boldsymbol{y} \mid \boldsymbol{x}_0)}{\mathbb{Q}_{T-t|0}(\boldsymbol{x} \mid \boldsymbol{x}_0)} - \mathbf{1}_{\boldsymbol{x} = \boldsymbol{y}}\sum_{\boldsymbol{z} \in \mathcal{X}} \boldsymbol{Q}_{T-t}^0(\boldsymbol{z}, \boldsymbol{x})\frac{\mathbb{Q}_{T-t|0}(\boldsymbol{z} \mid \boldsymbol{x}_0)}{\mathbb{Q}_{T-t|0}(\boldsymbol{y} \mid \boldsymbol{x}_0)}
$$

where $\mathbb{Q}\_{T-t|0}(\cdot \mid \cdot)$ is the conditional transition probability under the reference measure $\mathbb{Q}$. Then, the reverse-time KL divergence that is minimized under the Markovian projection $\mathbb{M}^\star$ is given by:

$$
\text{KL}(\Pi \| \mathbb{M}^\star) = \int_0^T \mathbb{E}_{\Pi_{t,T}}\left[\sum_{\boldsymbol{y} \ne \tilde{\boldsymbol{X}}_t} \left(\tilde{\boldsymbol{Q}}_t^{\Pi^{|T}}\log\frac{\tilde{\boldsymbol{Q}}_t^{\Pi^{|T}}}{\tilde{\boldsymbol{Q}}_t^{\tilde{\mathbb{M}}^\star}}\right)(\tilde{\boldsymbol{X}}_t, \boldsymbol{y}) + \left(\tilde{\boldsymbol{Q}}_t^{\tilde{\mathbb{M}}^\star} - \tilde{\boldsymbol{Q}}_t^{\Pi^{|T}}\right)(\tilde{\boldsymbol{X}}_t, \tilde{\boldsymbol{X}}_t)\right] dt
$$

where the KL divergence between the terminal distributions at time $T$ vanishes as we initialize $\mathbb{M}\_T = \Pi\_T$. Specifically, we define $\Pi^{|T}$ as the bridge measure conditioned on $\boldsymbol{X}\_T = \boldsymbol{x}\_T$ with the reverse-time generator $\tilde{\boldsymbol{Q}}\_t^{\Pi^{|T = \boldsymbol{x}\_T}}$ defined as:

$$
\tilde{\boldsymbol{Q}}_t^{\Pi^{|T = \boldsymbol{x}_T}}(\boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_T \sim \Pi_{0|t,T}}\left[\tilde{\boldsymbol{Q}}_t^0(\boldsymbol{x}, \boldsymbol{y}; \boldsymbol{x}_T) \mid \tilde{\boldsymbol{X}}_t = \boldsymbol{x}, \tilde{\boldsymbol{X}}_0 = \boldsymbol{x}_T\right]
$$

where for any $t \in [0, T]$, the marginal distributions match $\mathbb{M}\_t^\star = \Pi\_t$.

</div>

*Proof Sketch.* The result follows from the same proof sequence as the forward-time Markovian projection, which consists of deriving the form of the conditional generator in reverse-time coordinate (Lemma 7.16), showing that the reciprocal process conditioned on an initial state $\tilde{\boldsymbol{X}}\_0 = \boldsymbol{x}\_T$ is Markov (Lemma 7.17), and deriving the form of the optimal reverse-time generator of the Markovian projection (Lemma 7.18). The only difference is the *direction of the generator*. By simply defining the reverse-time generator on the time-reversed CTMC process $\tilde{\boldsymbol{X}}\_t := \boldsymbol{X}\_{T-t}$, we can compute the KL divergence between the generators of the reciprocal process $\Pi^{|T}$ conditioned on a state from time $T$ and the Markov process $\mathbb{M}$ evaluated along the reversed trajectory $\bar{\boldsymbol{X}}\_{0:T}$. Taking expectation with respect to the trajectories from the reversed path measure $\bar{\boldsymbol{X}}\_{0:T} \sim \Pi$ yields the reverse KL divergence. $\square$

Now that we have written the Markovian and reciprocal projections explicitly with respect to the generators of CTMCs, we can extend the Iterative Markovian Fitting (IMF) procedure described in Section 6.3 to the discrete state space, where instead of parameterizing the forward and reverse Markov drifts, we parameterize the forward and reverse Markov generators $\boldsymbol{Q}^{\mathbb{M}^\theta}$ and $\tilde{\boldsymbol{Q}}^{\tilde{\mathbb{M}}^\phi}$.

### 7.6 Discrete Diffusion Schroedinger Bridge Matching

Using these theoretical foundations of Markov and reciprocal projections of CTMCs, we can define the discrete state space analog of the Iterative Markovian Fitting (IMF) algorithm from Section 6.3 called the **Discrete Diffusion Schroedinger Bridge Matching** (DDSBM). Just like the IMF algorithm, DDSBM alternates between performing a Markovian projection to obtain a Markov generator and reciprocal projections to preserve the marginal constraints. Concretely, the algorithm is outlined as follows.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Discrete Diffusion Schroedinger Bridge Matching)</span></p>

DDSBM generates a sequence of Markov and reciprocal measures $(\mathbb{M}^n, \Pi^n)\_{n \in \mathbb{N}}$ initialized at $\Pi^0 := \pi\_{0,T}\mathbb{Q}\_{\cdot|0,T}$ by alternating between the following steps:

- **(1a)** Solve the forward-time Markovian projection $\mathbb{M}^{2n+1} := \text{proj}\_{\mathcal{M}}(\Pi^{2n})$ by updating a parameterized generator $\boldsymbol{Q}^{\mathbb{M}^\theta}$ to minimize $\mathcal{L}\_{\text{DDSBM}}(\theta) := \text{KL}(\Pi^{2n} \| \mathbb{M}^\theta)$
- **(1b)** Define the reciprocal projection as $\Pi^{2n+1} := \mathbb{M}\_{0,T}^{2n+1}\mathbb{Q}\_{\cdot|0,T}$
- **(2a)** Solve the backward-time Markovian projection $\mathbb{M}^{2n+2} := \text{proj}\_{\mathcal{M}}(\Pi^{2n+1})$ by updating a parameterized generator $\tilde{\boldsymbol{Q}}^{\tilde{\mathbb{M}}^\phi}$ to minimize $\mathcal{L}\_{\text{DDSBM}}(\phi) := \text{KL}(\Pi^{2n+1} \| \mathbb{M}^\phi)$
- **(2b)** Define the reciprocal projection as $\Pi^{2n+2} := \mathbb{M}\_{0,T}^{2n+2}\mathbb{Q}\_{\cdot|0,T}$

</div>

While the high-level form of this algorithm is straightforward, we will build a deeper understanding of how the algorithm is implemented in practice by analyzing each step below.

**Step 1a: Forward-Time Markovian Projection.** To obtain the Markovian projection $\mathbb{M}^{2n+1} = \text{proj}\_{\mathcal{M}}(\Pi^{2n})$, we **minimize the KL divergence** $\text{KL}(\Pi^{2n} \| \mathbb{M}^\theta)$ derived in Lemma 7.17 between the reciprocal measure from the previous iteration $\Pi^{2n} = \mathbb{M}\_{0,T}^\phi \mathbb{Q}\_{\cdot|0,T}$.

To do this, we define the loss function $\mathcal{L}(\theta)$ which yields $\mathbb{M}\_\theta = \text{proj}\_{\mathcal{M}}(\Pi^{2n})$ at optimality:

$$
\mathcal{L}_{\text{DDSBM}}(\theta) := \text{KL}(\Pi^{2n} \| \mathbb{M}^\theta) = \int_0^T \mathbb{E}_{\Pi_{0,t}^{2n}}\left[\sum_{\boldsymbol{y} \ne \boldsymbol{X}_t} \boldsymbol{Q}_t^{\mathbb{Q}_{\cdot|T}}\log\frac{\boldsymbol{Q}_t^{\mathbb{Q}_{\cdot|T}}}{\boldsymbol{Q}_t^{\mathbb{M}^\theta}}(\boldsymbol{X}_t, \boldsymbol{y}) + (\boldsymbol{Q}_t^{\mathbb{Q}_{\cdot|T}} - \boldsymbol{Q}_t^{\mathbb{M}^\theta})(\boldsymbol{X}_t, \boldsymbol{X}_t)\right] dt
$$

where $\Pi^{2n} = \mathbb{M}\_{0,T}^\phi \mathbb{Q}\_{\cdot|0,T}$. To sample from $\Pi^{2n}$, we first sample endpoints by simulating the reverse-time Markov generator $\mathbb{M}^\phi$ to obtain $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \mathbb{M}\_{0,T}^\phi$ and then sampling the reference bridge $\boldsymbol{X}\_t \sim \mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$.

For each intermediate sample $\boldsymbol{X}\_t \sim \mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$, we condition the forward-time reference generator to $\boldsymbol{X}\_T = \boldsymbol{x}\_T$ defined in Lemma 7.16 as:

$$
\boldsymbol{Q}_t^{\mathbb{Q}_{\cdot|T}}(\boldsymbol{X}_t, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{X}_T \sim \Pi_{T|0,t}}\left[\boldsymbol{Q}_t^0(\boldsymbol{X}_t, \boldsymbol{y}; \boldsymbol{X}_T) \mid \boldsymbol{X}_t, \boldsymbol{X}_0\right]
$$

**Step 1b: Forward-Time Reciprocal Projection.** Then, we define the corresponding reciprocal projection $\Pi^{2n+1} = \text{proj}\_{\mathcal{R}(\mathbb{Q})}(\mathbb{M}^{2n+1})$ as the mixture of bridges under the reference measure conditioned on the endpoint law of $\mathbb{M}^\theta$ defined as:

$$
\Pi^{2n+1} = \mathbb{M}_{0,T}^\theta \mathbb{Q}_{\cdot|0,T}
$$

While we show that the Markovian projection preserves the bridge measure $\mathbb{M}\_t = \Pi\_t$ in theory (Proposition 4.10), in practice, parameterizing the generator of the forward CTMC with a neural network $\theta$ can result in mismatches in the terminal marginal constraint at time $t = T$. Therefore, similarly to the DSBM algorithm in Section 6.3, we also parameterize the *reverse-time* Markovian projection that explicitly constrains the terminal marginal.

**Step 2a: Reverse-Time Markovian Projection.** To obtain the reverse-time Markovian projection $\mathbb{M}^{2n+2} = \text{proj}\_{\mathcal{M}}(\Pi^{2n+1})$ parameterized by the reverse generator $\phi$, we **minimize the KL divergence** $\text{KL}(\Pi^{2n+1} \| \mathbb{M}^\phi)$ with reciprocal measure from the previous iteration $\Pi^{2n+1} = \mathbb{M}\_{0,T}^\theta \mathbb{Q}\_{\cdot|0,T}$ and define the loss function $\mathcal{L}(\phi)$ which yields $\mathbb{M}^\phi = \text{proj}\_{\mathcal{M}}(\Pi^{2n+1})$ at optimality:

$$
\mathcal{L}_{\text{DDSBM}}(\phi) := \text{KL}(\Pi^{2n+1} \| \mathbb{M}^\phi) = \int_0^T \mathbb{E}_{\Pi_{0,t}^{2n+1}}\left[\sum_{\boldsymbol{y} \ne \tilde{\boldsymbol{X}}_t} \tilde{\boldsymbol{Q}}_t^{\mathbb{Q}_{\cdot|0}}\log\frac{\tilde{\boldsymbol{Q}}_t^{\mathbb{Q}_{\cdot|0}}}{\tilde{\boldsymbol{Q}}_t^{\tilde{\mathbb{M}}^\phi}}(\tilde{\boldsymbol{X}}_t, \boldsymbol{y}) + (\tilde{\boldsymbol{Q}}_t^{\mathbb{Q}_{\cdot|0}} - \tilde{\boldsymbol{Q}}_t^{\tilde{\mathbb{M}}^\phi})(\tilde{\boldsymbol{X}}_t, \tilde{\boldsymbol{X}}_t)\right] dt
$$

where the expectation is over samples from $\Pi^{2n+1} = \mathbb{M}\_{0,T}^\theta \mathbb{Q}\_{\cdot|0,T}$. Each sample from $\Pi^{2n+1}$ is obtained by simulating the *forward-time* Markov generator $\mathbb{M}^\theta$ to obtain $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \mathbb{M}\_{0,T}^\theta$ and then sampling the reference bridge $\tilde{\boldsymbol{X}}\_t \sim \mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$.

For each intermediate sample $\tilde{\boldsymbol{X}}\_t \sim \mathbb{Q}(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$, we condition the reverse-time reference generator to $\boldsymbol{X}\_0 = \boldsymbol{x}\_0$ defined analogously to the forward-time generator as:

$$
\tilde{\boldsymbol{Q}}_t^{\mathbb{Q}_{\cdot|0}}(\tilde{\boldsymbol{X}}_t, \boldsymbol{y}) = \mathbb{E}_{\boldsymbol{X}_0 \sim \Pi_{0|t,T}}\left[\tilde{\boldsymbol{Q}}_t^0(\tilde{\boldsymbol{X}}_t, \boldsymbol{y}; \boldsymbol{X}_T) \mid \tilde{\boldsymbol{X}}_t, \boldsymbol{X}_0\right]
$$

**Step 2b: Reverse-Time Reciprocal Projection.** Then, we define the corresponding reciprocal projection $\Pi^{2n+2} = \text{proj}\_{\mathcal{R}(\mathbb{Q})}(\mathbb{M}^{2n+2})$ as the mixture of bridges under the reference measure conditioned on the endpoint law of $\mathbb{M}^\phi$ defined as:

$$
\Pi^{2n+2} = \mathbb{M}_{0,T}^\phi \mathbb{Q}_{\cdot|0,T}
$$

Now that we understand how the discrete diffusion SBM algorithm is implemented in practice, it is natural to question whether it yields the same convergence guarantees as the IMF procedure. Since we already establish the intuition and formal proofs for the convergence of the IMF procedure in $\mathbb{R}^d$ in Section 4.5, the discrete state space analog follows similar intuition, where the IMF sequence results in monotonically decreasing KL divergence with the optimal bridge $\mathbb{P}^\star$ which converges to zero at the limit when the number of iterations goes to infinity.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.20</span><span class="math-callout__name">(Comparison of Discrete SB-SOC and DDSBM)</span></p>

A **key distinction** between the discrete diffusion Schroedinger bridge method (DDSBM) and the discrete stochastic optimal control (SOC) formulation lies in their dependence on target samples. The DDSBM algorithm initializes the reciprocal measure as $\Pi^0 := \pi\_{0,T}\mathbb{Q}\_{\cdot|0,T}$, which requires access to pairs from both marginals, and then samples from $\pi\_0$ when optimizing the forward-time Markovian projection and $\pi\_T$ when optimizing the reverse-time Markovian projection.

In contrast, the discrete SB-SOC framework (Box 7.4) only requires computing the RND between the optimally-controlled path measure and the one generated from the current control, enabling matching of unknown target distributions, such as temperature-annealed or reward-tilted distributions.

</div>

### 7.7 Closing Remarks for Section 7

In this section, we introduced the theory of **continuous-time Markov chains** (CTMCs), which provide the natural analogue of stochastic differential equations for modeling stochastic processes in discrete state spaces $\mathcal{X} = \lbrace 1, \ldots, d \rbrace$. In this setting, the role of the control drift is replaced by a *generator* or *transition rate* matrix $\boldsymbol{Q}\_t^u$, which governs the jump dynamics of the process. Using this representation, we defined the **discrete Schroedinger bridge problem**, which seeks the controlled CTMC path measure $\mathbb{P}^u$ with generator $\boldsymbol{Q}\_t^u$ that remains closest, in KL divergence, to a reference path measure $\mathbb{Q}$ with generator $\boldsymbol{Q}\_t^0$, while satisfying the marginal constraints $p\_0 = \pi\_0$ and $p\_T = \pi\_T$.

Building on this formulation, we introduced two frameworks for solving the discrete Schroedinger bridge problem: **stochastic optimal control** (SOC) and **iterative Markovian fitting** (IMF). These approaches mirror their continuous-state counterparts, translating the ideas of controlled SDEs and path-space projections to the setting of jump processes governed by CTMC generators.

Taken together, we have shown that the Schroedinger bridge framework extends naturally from diffusion processes in continuous spaces to jump processes in discrete state spaces. With this theoretical foundation in place, we are now prepared to explore how all the ideas developed throughout this guide can be applied in practice to solve real data-driven and scientific problems. In the final section, we examine several **applications of generative modeling with Schroedinger bridges**, illustrating how the theoretical tools developed throughout this guide can be used to construct practical generative models across a variety of domains.

## 8. Applications of Generative Modeling with Schroedinger Bridges

While Schroedinger bridges can be seen as a unified framework that encompasses a large portion of generative modeling techniques, from denoising diffusion to score-based generative modeling to flow matching, there are several applications where Schroedinger bridge frameworks are specialized to solve. In this section, we highlight three prominent applications where advances in Schroedinger bridge--based generative modeling have led to principled and practically effective solutions, including **data translation** (Section 8.1), **single-cell modeling** (Section 8.2), and **sampling high-Boltzmann energy distributions** (Section 8.3). Our goal is not to provide an exhaustive review of applications, but to illustrate how learning entropy-regularized stochastic paths between structured distributions can be applied to diverse scientific and data-driven problems.

### 8.1 Data Translation

The goal of data translation is to transform samples from one unknown distribution to another. Specifically, we will focus on **image-to-image translation**, which aims to learn a mapping between pairs of images and can be applied to many tasks such as deblurring, inpainting, and image editing. In this setting, the source and target datasets correspond to two probability distributions defined over the space of images, and the goal is to learn a transformation that transports samples from the source distribution to the target distribution while *preserving the underlying structure of the data*.

The Schroedinger bridge framework provides a principled probabilistic formulation of this problem. Given empirical samples from the source distribution $\pi\_0$ and the target distribution $\pi\_T$, the Schroedinger bridge constructs the most likely stochastic process that connects the two distributions while remaining close to a reference diffusion. Rather than learning a deterministic mapping between individual samples, the bridge defines a stochastic transport process that transforms an image in the source distribution into many possible images in the target distribution.

In many image restoration tasks the degraded image already contains **substantial structural information about the target image**. Rather than generating from random noise, Schroedinger bridges allow the generative process to start directly from the degraded image distribution and gradually transform it into the clean image distribution.

Since the data translation task assumes *paired samples*, we can consider the form of the SB problem, where the clean distribution $\pi\_T = \delta\_x$ is a Dirac Delta function at the target data point (e.g., clean image), which corresponds to some distribution of initial samples $\boldsymbol{x}\_T \sim \pi\_T(\cdot \mid \boldsymbol{x}\_0)$ (e.g. degraded or corrupted images). This produces a *joint distribution* of paired samples defined as:

$$
\pi(\boldsymbol{x}_0, \boldsymbol{x}_T) = \pi_0(\boldsymbol{x}_0)\pi(\boldsymbol{x}_T \mid \boldsymbol{x}_0)
$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.1</span><span class="math-callout__name">(Data Translation with Schroedinger Bridge)</span></p>

Given data pairs sampled from the joint distribution $(\boldsymbol{x}\_0, \boldsymbol{x}\_T) \sim \pi\_0(\boldsymbol{x}\_0)\pi\_T(\boldsymbol{x}\_T \mid \boldsymbol{x}\_0)$, the optimal Schroedinger bridge density at intermediate times $t \in [0, T]$ takes the Gaussian form:

$$
\boldsymbol{X}_t \sim p_t(\boldsymbol{X}_t \mid \boldsymbol{x}_0, \boldsymbol{x}_T) = \mathcal{N}(\boldsymbol{X}_t; \boldsymbol{\mu}_t(\boldsymbol{x}_0, \boldsymbol{x}_T), \boldsymbol{\Sigma}_t)
$$

where $\boldsymbol{\mu}\_t \in \mathbb{R}^d$ is the conditional mean at time $t$, $\boldsymbol{\Sigma}\_t \in \mathbb{R}^{d \times d}$ is the covariance matrix. A choice of parameterization used in image restoration defines $\sigma\_t \in \mathbb{R}$ as the variance accumulated in forward time, and $\bar{\sigma}\_t \in \mathbb{R}$ as the variance accumulated in the reverse time coordinate, which explicitly define the mean and covariance as:

$$
\boldsymbol{\mu}_t = \frac{\bar{\sigma}_t^2}{\bar{\sigma}_t^2 + \sigma_t}\boldsymbol{x}_0 + \frac{\sigma_t^2}{\bar{\sigma}_t^2 + \sigma_t}\boldsymbol{x}_T, \quad \boldsymbol{\Sigma}_t = \frac{\sigma_t^2 \bar{\sigma}_t^2}{\bar{\sigma}_t^2 + \sigma_t^2}\boldsymbol{I}_d, \quad \sigma_t^2 := \int_0^t \beta_s ds, \quad \bar{\sigma}_t^2 := \int_t^T \beta_s ds
$$

When $\beta\_t \equiv \beta$ is constant over $t \in [0, T]$ and sufficiently small, the optimal SB reduces to the **dynamic optimal transport** map between the pair $(\boldsymbol{x}\_0, \boldsymbol{x}\_T)$ defined as:

$$
\boldsymbol{v}_t(\boldsymbol{X}_t) = \frac{\boldsymbol{X}_t - \boldsymbol{x}_0}{t}, \quad \boldsymbol{\mu}_t = \left(1 - \frac{t}{T}\right)\boldsymbol{x}_0 + \frac{t}{T}\boldsymbol{x}_T
$$

</div>

To learn the velocity $\boldsymbol{v}\_\theta$ that optimally transports samples from $\pi\_0$ to $\pi\_T$, we optimize the matching loss defined as:

$$
\mathcal{L}(\theta) := \mathbb{E}_{\boldsymbol{x}_0 \sim \pi_0(\boldsymbol{X}_0), \boldsymbol{x}_T \sim \pi_T(\boldsymbol{X}_T \mid \boldsymbol{X}_T)} \int_0^T \left\lVert \boldsymbol{v}_\theta(\boldsymbol{X}_t, t) - \underbrace{\frac{\boldsymbol{X}_t - \boldsymbol{x}_0}{\sigma_t}}_{\nabla\log\hat{\varphi}_t(\boldsymbol{X}_t \mid \boldsymbol{x}_0)} \right\rVert^2 dt
$$

where $\nabla\log\hat{\varphi}\_t(\boldsymbol{X}\_t \mid \boldsymbol{X}\_0)$ is the SB potential drift which is equivalent to the **conditional score function** $\nabla\log p\_t(\boldsymbol{X}\_t \mid \boldsymbol{x}\_0)$ where $\boldsymbol{X}\_t \sim p\_t(\cdot \mid \boldsymbol{x}\_0, \boldsymbol{x}\_T)$.

We observe that this objective resembles the standard **conditional score matching objective**, where the score function is given by $\nabla\log p\_t(\boldsymbol{X}\_t \mid \boldsymbol{y})$, where $\boldsymbol{y}$ is the conditioning information, which in the case of data translation is the initial or corrupted sample. However, in conditional score matching, the generation process starts from Gaussian noise $\boldsymbol{x}\_0 \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}\_d)$ and learns the score function using the corrupted sample as just another input to the model.

When the corrupted sample is already close to the target distribution, generating the clean target from pure noise is highly inefficient, which reveals the advantages of Schroedinger bridges as a way to learn the optimal interpolating bridge between a **structurally informative prior** $\pi\_0$ to the target distribution $\pi\_T$. Since the SB dynamics explicitly connect corrupted and clean image distributions, the generative trajectories correspond to progressive restoration processes rather than denoising from pure noise, leading to more interpretable and efficient generation.

### 8.2 Modeling Single-Cell State Dynamics

Cellular systems undergo dynamic state transitions from cell differentiation to responses under genetic or drug perturbations to adaptive changes during development and disease. These processes can be viewed as trajectories through a high-dimensional cell state space $\mathbb{R}^d$, typically defined as single-cell RNA sequencing (scRNA-seq) measurements of gene expression.

Due to the destructive nature of single-cell sequencing technologies, which kill the cell after measurement, data is often a collection of static snapshots of populations at discrete time points rather than continuous-time measurements. As a result, the underlying stochastic dynamics governing how cells transition between states are not directly observed. Instead, the problem becomes one of **inferring the dynamical process that transports a distribution of cells at an initial condition** $\pi\_0 \in \mathcal{P}(\mathbb{R}^d)$ **to a distribution observed at a later time** $\pi\_T \in \mathcal{P}(\mathbb{R}^d)$.

$$
\underbrace{\pi_0(\boldsymbol{x})}_{\text{initial cell distribution}} \xrightarrow{\text{differentiation, perturbation, etc.}} \underbrace{\pi_T(\boldsymbol{x})}_{\text{terminal cell distribution}}, \quad \boldsymbol{x} \in \mathbb{R}^d
$$

Since cell states naturally lie on a non-linear biological manifold, the Schroedinger bridge framework provides a natural approach to this problem. Given empirical distributions of cellular states at sequential time points $\lbrace \pi\_{t\_k} \rbrace\_{k=1}^K$, the Schroedinger bridge identifies the most likely stochastic process that connects these distributions while remaining close to a chosen reference diffusion. In this formulation, the inferred bridge represents a probabilistic model of cell state evolution that captures both the deterministic drift driving differentiation and the stochastic variability inherent in biological systems.

#### Modeling Cellular Dynamics with Schroedinger Bridges

Consider a sequence of snapshots of a cell population at discrete time intervals $\lbrace \pi\_{t\_k} \in \mathcal{P}(\mathbb{R}^d) \rbrace\_{k=1}^K$ for $0 = t\_0 < \cdots < t\_k < \cdots < t\_K = T$ where $\mathbb{R}^d$ represents a $d$-dimensional gene expression space. Then, the problem of modeling cellular dynamics between snapshots can be framed as optimizing a parameterized control $\boldsymbol{u}\_\theta(\boldsymbol{x}, t)$ that minimizes a combination of the Multi-Marginal SB Problem and the Generalized SB Problem:

$$
\inf_{\boldsymbol{u}_\theta} \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^u}\left[\int_0^T \left(\frac{1}{2}\lVert \boldsymbol{u}_\theta(\boldsymbol{X}, t) \rVert^2 + \mathcal{I}(\boldsymbol{X}, p_t, t)\right) dt\right]
$$

$$
\text{s.t.} \quad \begin{cases} d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{x}, t) + \sigma_t \boldsymbol{u}_\theta(\boldsymbol{X}_t, t)) dt + \sigma_t d\boldsymbol{B}_t \\ \boldsymbol{X}_0 \sim \pi_0, \quad \boldsymbol{X}_T \sim \pi_T \end{cases}
$$

where $\boldsymbol{f}(\boldsymbol{x}, t)$ is the underlying drift of the cell population, $\sigma\_t$ is the diffusion coefficient representing stochastic in cell evolution, and $\mathcal{I}(\boldsymbol{x}, p\_t) : \mathbb{R}^d \times \mathcal{P}(\mathbb{R}^d) \to \mathbb{R}$ is the interaction cost that measures how the cells evolve with respect to the overall population.

When the population of cells *changes over time*, we can formulate the cell modeling problem as an Unbalanced SB Problem with unbalanced marginal constraints:

$$
\inf_{p_t, \boldsymbol{u}_\theta, g_\phi} \int_0^T \int_{\mathbb{R}^d} \left[\frac{1}{2}\lVert \boldsymbol{u}_\theta(\boldsymbol{x}, t) \rVert^2 + \alpha\Psi(g_\phi(\boldsymbol{x}, t))\right] p_t(\boldsymbol{x}) d\boldsymbol{x} dt
$$

$$
\text{s.t.} \quad \begin{cases} \partial_t p_t = -\nabla \cdot (p_t(\boldsymbol{f} + \sigma_t \boldsymbol{u})) + \frac{\sigma_t^2}{2}\Delta p_t + g p_t \\ \forall k \in \lbrace 1, \ldots, K \rbrace, \quad p_{t_k} = \pi_{t_k} \end{cases}
$$

where $g\_\phi(\boldsymbol{x}, t)$ parametrizes the growth rate that determines how the cell population increase or decrease in mass and $\Psi(g\_\phi(\boldsymbol{x}, t))$ is the scalar function that penalizes cell growth and death.

Finally, we can consider the case of **branching cell dynamics**, where the terminal cell population contains multiple distinct modes $\pi\_T = \lbrace \pi\_{T,k} \rbrace\_{k=1}^K$ defining different sub-populations that have diverged as a result of differentiation or cellular perturbation. This problem can naturally be framed as the Branched SB Problem, which can be solved by defining a set of parameterized control drifts and growth rates $\lbrace \boldsymbol{u}\_{\theta,k}, g\_{\phi,k} \rbrace\_{k=1}^K$ that minimize:

$$
\inf_{\lbrace \boldsymbol{u}_{\theta,k}, g_{\phi,k} \rbrace_{k=1}^K} \int_0^T \left\lbrace \mathbb{E}_{p_{t,0}}\left[\frac{1}{2}\lVert \boldsymbol{u}_{\theta,0}(\boldsymbol{X}_{t,0}, t) \rVert^2 + c(\boldsymbol{X}_{t,0}, t)\right] w_{t,0} + \sum_{k=1}^K \mathbb{E}_{p_{t,k}}\left[\frac{1}{2}\lVert \boldsymbol{u}_{\theta,k}(\boldsymbol{X}_{t,k}, t) \rVert^2 + c(\boldsymbol{X}_{t,k}, t)\right] w_{t,k} \right\rbrace dt
$$

$$
\text{s.t.} \quad \begin{cases} d\boldsymbol{X}_{t,k} = (\boldsymbol{f}(\boldsymbol{X}_{t,k}, t) + \sigma_t \boldsymbol{u}_{\theta,k}(\boldsymbol{X}_{t,k}, t)) dt + \sigma_t d\boldsymbol{B}_t \\ \boldsymbol{X}_0 \sim \pi_0, \quad \boldsymbol{X}_{T,k} \sim \pi_{T,k} \\ w_{0,k} = \delta_{k=0}, \quad w_{T,k} = w_{T,k}^\star \end{cases}
$$

where the weight of the primary branch is given by $w\_{t,0} = 1 + \int\_0^t g\_{t,\phi}(\boldsymbol{X}\_{s,0}, s) ds$ and the weights of the $K$ secondary branches is given by $w\_{t,k} = \int\_0^t g\_{k,\phi}(\boldsymbol{X}\_{s,k}, s) ds$.

This shows that Schroedinger bridges defines a unified problem that can be specialized to several biologically relevant settings. In the balanced setting, the bridge captures smooth state transitions between multiple cell population distributions. In the unbalanced setting, it incorporates cell proliferation and death through growth terms, allowing the total population mass to vary over time, and finally, the branched Schroedinger bridge formulation models lineage diversification and diverse responses to perturbations by decomposing the population into multiple sub-trajectories, each governed by its own dynamics and weights.

Together, these formulations provide a principled and flexible framework for modeling complex cellular processes, including differentiation, perturbation responses, and lineage branching. By learning the underlying bridge dynamics, we not only interpolate intermediate cellular states but also recover the latent structure of population evolution, enabling predictive modeling of how cell populations evolve under changing biological conditions.

### 8.3 Sampling Boltzmann Distributions

The **sampling** problem aims to generate a complete reconstruction of a probability distribution. A specific class of distributions of particular interest in computational sciences are **Boltzmann distributions**, which are unnormalized and often high-dimensional and defined only by an **energy function** $U(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$ without explicit samples.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.2</span><span class="math-callout__name">(Sampling Boltzmann Distributions)</span></p>

Given an energy function $U(\boldsymbol{x}) : \mathbb{R}^d \to \mathbb{R}$ where $\mathbb{R}^d$, the **Boltzmann distribution** is defined as:

$$
\pi_T(\boldsymbol{x}) := \frac{e^{-U(\boldsymbol{x})}}{Z}, \quad \text{where} \quad Z := \int_{\mathbb{R}^d} e^{-U(\boldsymbol{x})} d\boldsymbol{x}
$$

such that $\int\_{\mathbb{R}^d} \pi\_T(\boldsymbol{x}) d\boldsymbol{x} = 1$ and $Z$ is some, often intractable, normalization factor.

</div>

This class of distributions are particularly difficult to sample due to the computational cost of evaluating the energy function of the molecular systems. Traditional approaches such as Markov chain Monte Carlo (MCMC) and Sequential Monte Carlo (SMC) while provably converges to the target distribution $\pi\_T$, exhibit **slow mixing times**, where the sampled distribution requires a significant number of simulation steps to converge to the target distribution, and can **remain trapped in local minima** with poor reconstruction of the global energy landscape, especially for large molecules with high-dimensional representations and multi-modal energy landscapes.

To overcome these challenges, generative modeling frameworks have been developed that approach the problem as transporting an easy-to-sample source distribution $\boldsymbol{X}\_0 \sim \pi\_0$ (e.g., a Gaussian or Dirac delta at the origin) to the target Boltzmann distribution $\pi\_T(\boldsymbol{x}) = \frac{e^{-U(\boldsymbol{x})}}{Z}$. Specifically, these frameworks define parameterized control drift $\boldsymbol{u}\_\theta(\boldsymbol{x}, t)$ that steers the sample toward the target Boltzmann distribution $\pi\_T(\boldsymbol{x}) = \frac{e^{-U(\boldsymbol{x})}}{Z}$ through the controlled SDE:

$$
d\boldsymbol{X}_t = [\boldsymbol{f}_t(\boldsymbol{X}_t) + \sigma_t \boldsymbol{u}_\theta(\boldsymbol{X}_t, t)] dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0 \sim \pi_0
$$

In this setting, we do not have access to explicit samples from the target distribution $\pi\_T$ and can only evaluate its unnormalized density through an energy function $\pi\_T(\boldsymbol{x}) = \frac{e^{-U(\boldsymbol{x})}}{Z}$. This implicit specification makes direct sampling intractable, and naturally motivates a formulation using **stochastic optimal control** (SOC) as described in Section 3, where we optimize the control $\boldsymbol{u}\_\theta(\boldsymbol{x}, t)$ such that the terminal distribution generated from the optimal control $\boldsymbol{u}^\star(\boldsymbol{x}, t)$ matches the Boltzmann distribution $p\_T^\star = \pi\_T$.

Specifically, we consider the SB-SOC Objective described in Section 3.2, where the terminal constraint is defined as $\log\frac{\hat{\varphi}\_T(\boldsymbol{X}\_T)}{\pi\_T(\boldsymbol{X}\_T)}$, where $\hat{\varphi}\_T$ is the backward Schroedinger potential evaluated at time $t = T$. When sampling from a Boltzmann distribution, this constraint becomes:

$$
\log\frac{\hat{\varphi}_T(\boldsymbol{X}_T)}{\pi_T(\boldsymbol{X}_T)} = \log\hat{\varphi}_T(\boldsymbol{X}_T) - \log\left(\frac{e^{-U(\boldsymbol{X}_T)}}{Z}\right) = \log\hat{\varphi}_T(\boldsymbol{X}_T) + U(\boldsymbol{X}_T) + \log Z
$$

Using this definition for the terminal constraint, we can define the Boltzmann SB-SOC problem.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.3</span><span class="math-callout__name">(Sampling the Boltzmann Distribution with Schroedinger Bridges and Stochastic Optimal Control)</span></p>

The optimal control drift $\boldsymbol{u}^\star(\boldsymbol{x}, t)$ that generates samples from the Boltzmann distribution $\pi\_T(\boldsymbol{X}\_T) = \frac{e^{-U(\boldsymbol{X}\_T)}}{Z}$ solves the **stochastic optimal control** (SOC) problem defined as:

$$
\inf_{\boldsymbol{u}} \mathbb{E}_{\boldsymbol{X}_{0:T} \sim \mathbb{P}^{u_\theta}}\left[\int_0^T \frac{1}{2}\lVert \boldsymbol{u}_\theta(\boldsymbol{X}_t, t) \rVert^2 dt + \log\hat{\varphi}_T(\boldsymbol{X}_T) + U(\boldsymbol{X}_T) + \log Z\right]
$$

$$
\text{s.t.} \quad d\boldsymbol{X}_t = (\boldsymbol{f}(\boldsymbol{X}_t, t) + \sigma_t \boldsymbol{u}_\theta(\boldsymbol{X}_t, t)) dt + \sigma_t d\boldsymbol{B}_t, \quad \boldsymbol{X}_0 \sim \pi_0
$$

where $\hat{\varphi}\_T(\boldsymbol{x})$ is the backward Schroedinger potential. Under the optimal control $\boldsymbol{u}^\star(\boldsymbol{x}, t)$, the generated distribution $p\_T^\star$ exactly reconstructs the Boltzmann distribution $p\_T^\star = \pi\_T$.

</div>

*Proof.* This result follows immediately from our proof of Proposition 3.11 by replacing the terminal value constraint with $\log\frac{\hat{\varphi}\_T(\boldsymbol{X}\_T)}{\pi\_T(\boldsymbol{X}\_T)} = \log\hat{\varphi}\_T(\boldsymbol{X}\_T) + U(\boldsymbol{X}\_T) + \log Z$ to get:

$$
\mathbb{P}^\star(\boldsymbol{X}_0, \boldsymbol{X}_T) = \frac{1}{Z}\mathbb{Q}(\boldsymbol{X}_0, \boldsymbol{X}_T)\exp\left(-\log\hat{\varphi}_T(\boldsymbol{X}_T) - U(\boldsymbol{X}_T) - \log Z - \log\varphi_0(\boldsymbol{X}_0)\right)
$$

and integrating over $\boldsymbol{X}\_T$ to get the terminal distribution generated from the optimal control drift $\boldsymbol{u}^\star$:

$$
p_T^\star(\boldsymbol{X}_T) = \exp(-U(\boldsymbol{X}_T) - \log Z) = \pi_T(\boldsymbol{X}_T)
$$

which exactly matches the target Boltzmann density without the initial value function bias. $\square$

One approach to solving this problem is using the SB-AM Objective and Corrector Matching Objective, defined specifically for sampling Boltzmann density as:

$$
\mathcal{L}_{\text{SB-AM}}(\boldsymbol{u}) := \mathbb{E}_{p_{0,T}^{\bar{u}}}\left[\frac{1}{2}\int_0^T \mathbb{E}_{p_{t|0,T}^{\bar{u}}}\left[\lVert \boldsymbol{u}(\boldsymbol{X}_t, t) + \sigma_t(\nabla U(\boldsymbol{X}_T) + \nabla\log\hat{\varphi}_T(\boldsymbol{X}_T)) \rVert^2\right] dt\right]
$$

$$
\mathcal{L}_{\text{SB-CM}}(\widehat{\boldsymbol{Z}}_T) := \mathbb{E}_{p_{0,T}^{\bar{u}}}\left[\lVert \widehat{\boldsymbol{Z}}_T(\boldsymbol{X}_T) - \nabla_{\boldsymbol{x}_T}\log\mathbb{Q}_{T|0}(\boldsymbol{X}_T \mid \boldsymbol{X}_0) \rVert^2\right]
$$

Since the prior distribution for the sampling problem is typically Gaussian and the intermediate dynamics can be arbitrarily defined as long as it converges to the target distribution, we consider the case where $\mathbb{Q}$ is pure Brownian motion with zero drift $\boldsymbol{f}\_t := 0$ and the conditional distribution $p\_{t|0,T}^{\bar{u}}$ takes a tractable form without requiring sampling the full path $\boldsymbol{X}\_{0:T} \sim \mathbb{P}^{\bar{u}}$.

This perspective reveals that sampling from Boltzmann distributions can be interpreted as learning the optimal Schroedinger bridge between a simple prior distribution and the target energy-based distribution. We show that formulation yields efficient algorithms that significantly reduce the number of energy evaluations and enables fast inference by simulating an SDE with a parameterized control drift. Notably, these examples highlight the broader potential of Schroedinger bridge methods as a **scalable framework for sampling in high-dimensional and complex energy landscapes**, with promising applications in modeling and understanding diverse molecular properties.

### 8.4 Closing Remarks for Section 8

This section illustrates notable examples of how Schroedinger bridge formulations extend beyond a unifying theoretical framework to become powerful, task-specific tools for generative modeling. Across data translation, single-cell dynamics, and sampling from complex energy landscapes, the common principle is the learning of entropy-regularized stochastic paths that respect both the structure of the data and the underlying physical or statistical constraints. By framing these problems as controlled stochastic processes, Schroedinger bridges provide a flexible mechanism for incorporating domain knowledge, handling distributional mismatch, and enabling efficient inference in high-dimensional settings. These successes highlight the versatility of the framework and suggest that Schroedinger bridge--based methods will continue to play a central role in advancing generative modeling across scientific and data-driven domains.
