---
layout: default
title: Mathematical Theory of Deep Learning
date: 2025-01-01
excerpt: Notes on the mathematical foundations of deep learning, covering approximation theory, optimization, generalization, and robustness of neural networks.
tags:
  - deep-learning
  - mathematics
  - neural-networks
---

# Mathematical Theory of Deep Learning

**Table of Contents**
- TOC
{:toc}

## Chapter 1: Introduction

### Mathematics of Deep Learning

In 2012, a deep learning architecture (AlexNet) revolutionized computer vision by achieving unprecedented performance in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). A few years later, in March 2016, AlphaGo defeated the best Go player at the time, Lee Sedol, in a five-game match. These breakthroughs, along with AlphaFold, GPT-3, Stable Diffusion, Midjourney, and DALL-E, have sparked interest in the theoretical underpinnings of deep learning.

Initially, there was a clear consensus in the mathematics community: *We do not understand why this technology works so well! In fact, there are many mathematical reasons that, at least superficially, should prevent the observed success.* Over the past decade the field has matured, though many open questions remain.

### High-Level Overview of Deep Learning

Deep learning refers to the application of deep neural networks trained by gradient-based methods to identify unknown input-output relationships. This approach has three key ingredients: **deep neural networks**, **gradient-based training**, and **prediction**.

**Deep Neural Networks.** A **neuron** is a function of the form

$$\mathbb{R}^d \ni \boldsymbol{x} \mapsto \nu(\boldsymbol{x}) = \sigma(\boldsymbol{w}^\top \boldsymbol{x} + b),$$

where $\boldsymbol{w} \in \mathbb{R}^d$ is a **weight vector**, $b \in \mathbb{R}$ is the **bias**, and $\sigma$ is the **activation function**. Neural networks are formed by connecting neurons, where the output of one neuron becomes the input to another.

A **shallow feedforward neural network** is an affine transformation applied to the output of a set of neurons that share the same input and the same activation function. Formally, it is a map $\Phi$ of the form

$$\mathbb{R}^d \ni \boldsymbol{x} \mapsto \Phi(\boldsymbol{x}) = T_1 \circ \sigma \circ T_0(\boldsymbol{x})$$

where $T_0$, $T_1$ are affine transformations. A **deep feedforward neural network** is constructed by compositions of shallow neural networks:

$$\mathbb{R}^d \ni \boldsymbol{x} \mapsto \Phi(\boldsymbol{x}) = T_{L+1} \circ \sigma \circ \cdots \circ T_1 \circ \sigma \circ T_0(\boldsymbol{x}),$$

where $L \in \mathbb{N}$ is the **number of layers**.

**Gradient-Based Training.** In **supervised learning**, the objective depends on a collection of input-output pairs $S = (\boldsymbol{x}_i, \boldsymbol{y}_i)_{i=1}^m$ called **training data**. The goal is to find a deep neural network $\Phi$ such that

$$\Phi(\boldsymbol{x}_i) \approx \boldsymbol{y}_i \quad \text{for all } i = 1, \ldots, m.$$

A standard way of achieving this is by minimizing the **empirical risk**:

$$\widehat{\mathcal{R}}_S(\Phi) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(\Phi(\boldsymbol{x}_i), \boldsymbol{y}_i),$$

where $\mathcal{L}$ is a **loss function**. The gradient of the empirical risk can be efficiently computed using **backpropagation**, enabling optimization via (stochastic) gradient descent.

**Prediction.** We assume existence of a **data distribution** $\mathcal{D}$ on $\mathbb{R}^d \times \mathbb{R}^k$ from which the sample $S$ and new data points are drawn i.i.d. The **risk** of a trained network $\Phi_*$ is

$$\mathcal{R}(\Phi_*) = \mathbb{E}_{(\boldsymbol{x}_{\text{new}}, \boldsymbol{y}_{\text{new}}) \sim \mathcal{D}}[\mathcal{L}(\Phi_*(\boldsymbol{x}_{\text{new}}), \boldsymbol{y}_{\text{new}})].$$

If the risk is not much larger than the empirical risk, we say $\Phi_*$ has a small **generalization error**. If the risk is much larger, we say $\Phi_*$ **overfits** the training data.

### Why Does It Work?

Three fundamental questions drive the mathematical study of deep learning:

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key Questions)</span></p>

**Approximation.** The *universal approximation theorem* states that every continuous function on a compact domain can be approximated arbitrarily well by a shallow neural network. However, this does not address *efficiency*: What is the role of the architecture for the expressive capabilities of neural networks? Why do deep networks appear to overcome the *curse of dimensionality*?

**Optimization.** The empirical risk is typically *highly non-linear and not convex* due to the repeated compositions with the nonlinear activation function. There is generally no guarantee that the optimization routine converges to a global minimum. *Why is the output of the optimization nonetheless often meaningful in practice?*

**Generalization.** In traditional statistical learning theory, the generalization error can be bounded in terms of a complexity measure divided by the number of training samples. In practice, neural networks operate in the *overparameterized regime*, where the number of parameters exceeds the number of training samples, rendering classical estimates void. *Why do deep overparameterized architectures generalize well?*

</div>

### Outline and Philosophy

The book addresses the above questions through the following chapters:

- **Chapter 2: Feedforward neural networks** -- introduces the main object of study.
- **Chapter 3: Universal approximation** -- the classical view of function approximation by neural networks.
- **Chapter 4: Splines** -- approximation rates and the link between neural networks and spline approximation.
- **Chapter 5: ReLU neural networks** -- the class of ReLU networks equals the set of continuous piecewise linear functions.
- **Chapter 6: Affine pieces for ReLU neural networks** -- deep networks can generate exponentially more affine regions than shallow ones.
- **Chapter 7: Deep ReLU neural networks** -- substantially better approximation rates than shallow networks.
- **Chapter 8: High-dimensional approximation** -- scenarios where neural networks can overcome the curse of dimensionality.
- **Chapter 9: Interpolation** -- conditions under which exact interpolation of training data is possible.
- **Chapter 10: Training of neural networks** -- (stochastic) gradient descent, accelerated methods, Adam, and backpropagation.
- **Chapter 11: Wide neural networks and the neural tangent kernel** -- training dynamics resemble kernel regression for sufficiently wide networks.
- **Chapter 12: Loss landscape analysis** -- overparameterization leads to favorable loss landscape geometry.
- **Chapter 13: Shape of neural network spaces** -- the set of neural networks of a fixed architecture is typically non-convex and lacks the best-approximation property.
- **Chapter 14: Generalization properties** -- classical statistical learning theory applied to neural network function classes.
- **Chapter 15: Generalization in the overparameterized regime** -- the phenomenon of double descent.
- **Chapter 16: Robustness and adversarial examples** -- theoretical explanations of adversarial examples.

---

## Chapter 2: Feedforward Neural Networks

### Formal Definition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.1</span><span class="math-callout__name">(Neural Network)</span></p>

Let $L \in \mathbb{N}$, $d_0, \ldots, d_{L+1} \in \mathbb{N}$, and let $\sigma \colon \mathbb{R} \to \mathbb{R}$. A function $\Phi \colon \mathbb{R}^{d_0} \to \mathbb{R}^{d_{L+1}}$ is called a **neural network** if there exist matrices $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$ and vectors $\boldsymbol{b}^{(\ell)} \in \mathbb{R}^{d_{\ell+1}}$, $\ell = 0, \ldots, L$, such that with

$$\boldsymbol{x}^{(0)} := \boldsymbol{x}$$

$$\boldsymbol{x}^{(\ell)} := \sigma(\boldsymbol{W}^{(\ell-1)} \boldsymbol{x}^{(\ell-1)} + \boldsymbol{b}^{(\ell-1)}) \quad \text{for } \ell \in \lbrace 1, \ldots, L \rbrace$$

$$\boldsymbol{x}^{(L+1)} := \boldsymbol{W}^{(L)} \boldsymbol{x}^{(L)} + \boldsymbol{b}^{(L)}$$

it holds that $\Phi(\boldsymbol{x}) = \boldsymbol{x}^{(L+1)}$ for all $\boldsymbol{x} \in \mathbb{R}^{d_0}$.

We call $L$ the **depth**, $d_{\max} = \max_{\ell=1,\ldots,L} d_\ell$ the **width**, $\sigma$ the **activation function**, and $(\sigma; d_0, d_1, \ldots, d_{L+1})$ the **architecture** of the neural network $\Phi$. The matrices $\boldsymbol{W}^{(\ell)}$ are the **weight matrices** and $\boldsymbol{b}^{(\ell)}$ the **bias vectors** of $\Phi$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.2</span></p>

Different choices of architectures, weights, and biases can yield the same function $\Phi \colon \mathbb{R}^{d_0} \to \mathbb{R}^{d_{L+1}}$. Therefore we cannot associate a unique meaning to the depth, width, etc. based solely on the function realized by $\Phi$. When we refer to properties of a neural network, it is always understood that there exists at least one construction satisfying those properties.

</div>

The architecture is often depicted as a connected graph. The **nodes** represent the neurons, arranged in **layers**, with $\boldsymbol{x}^{(\ell)}$ corresponding to layer $\ell$. We refer to $\boldsymbol{x}^{(0)}$ as the **input layer** and $\boldsymbol{x}^{(L+1)}$ as the **output layer**. All layers in between are the **hidden layers**. Networks of depth one are called **shallow**; if the depth is larger than one they are called **deep**.

Key terminology:

- **parameters**: The set of all entries of the weight matrices and bias vectors, often collected in a single vector $\boldsymbol{w} = ((\boldsymbol{W}^{(0)}, \boldsymbol{b}^{(0)}), \ldots, (\boldsymbol{W}^{(L)}, \boldsymbol{b}^{(L)}))$. These are learned during training.
- **hyperparameters**: Settings that define the network's architecture (depth, width, activation function). They are set before training begins.
- **weights**: Often used broadly to refer to *all* parameters, including bias vectors.
- **model**: For a fixed architecture, every choice of parameters $\boldsymbol{w}$ defines a specific function $\boldsymbol{x} \mapsto \Phi_{\boldsymbol{w}}(\boldsymbol{x})$.

Common variations of Definition 2.1 include:

- Using **different activation functions** $\sigma_\ell$ in each layer.
- **Residual** neural networks with "skip connections": nodes in layer $\ell$ may have $\boldsymbol{x}^{(0)}, \ldots, \boldsymbol{x}^{(\ell-1)}$ as their input.
- **Recurrent** neural networks where information flows backward, creating loops in the flow of information.

#### Basic Operations on Neural Networks

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.3</span><span class="math-callout__name">(Operations on Neural Networks)</span></p>

For two neural networks $\Phi_1$, $\Phi_2$ with architectures

$$(\sigma; d_0^1, d_1^1, \ldots, d_{L_1+1}^1) \quad \text{and} \quad (\sigma; d_0^2, d_1^2, \ldots, d_{L_2+1}^2)$$

respectively, it holds that:

**(i) Scaling:** For all $\alpha \in \mathbb{R}$, there exists a neural network $\Phi_\alpha$ with architecture $(\sigma; d_0^1, d_1^1, \ldots, d_{L_1+1}^1)$ such that

$$\Phi_\alpha(\boldsymbol{x}) = \alpha \Phi_1(\boldsymbol{x}) \quad \text{for all } \boldsymbol{x} \in \mathbb{R}^{d_0^1}.$$

**(ii) Parallelization:** If $d_0^1 = d_0^2 =: d_0$ and $L_1 = L_2 =: L$, then there exists a neural network $\Phi_{\text{parallel}}$ with architecture $(\sigma; d_0, d_1^1 + d_1^2, \ldots, d_{L+1}^1 + d_{L+1}^2)$ such that

$$\Phi_{\text{parallel}}(\boldsymbol{x}) = (\Phi_1(\boldsymbol{x}), \Phi_2(\boldsymbol{x})) \quad \text{for all } \boldsymbol{x} \in \mathbb{R}^{d_0}.$$

**(iii) Summation:** If $d_0^1 = d_0^2 =: d_0$, $L_1 = L_2 =: L$, and $d_{L+1}^1 = d_{L+1}^2 =: d_{L+1}$, then there exists a neural network $\Phi_{\text{sum}}$ with architecture $(\sigma; d_0, d_1^1 + d_1^2, \ldots, d_L^1 + d_L^2, d_{L+1})$ such that

$$\Phi_{\text{sum}}(\boldsymbol{x}) = \Phi_1(\boldsymbol{x}) + \Phi_2(\boldsymbol{x}) \quad \text{for all } \boldsymbol{x} \in \mathbb{R}^{d_0}.$$

**(iv) Composition:** If $d_{L_1+1}^1 = d_0^2$, then there exists a neural network $\Phi_{\text{comp}}$ with architecture $(\sigma; d_0^1, d_1^1, \ldots, d_{L_1}^1, d_1^2, \ldots, d_{L_2+1}^2)$ such that

$$\Phi_{\text{comp}}(\boldsymbol{x}) = \Phi_2 \circ \Phi_1(\boldsymbol{x}) \quad \text{for all } \boldsymbol{x} \in \mathbb{R}^{d_0^1}.$$

</div>

### Notion of Size

Neural networks provide a framework to parametrize functions. In practice, further restrictions on the weights and biases are often desirable:

- **weight sharing**: Specific entries of the weight matrices (or bias vectors) are constrained to be equal. Formally, $W_{k,l}^{(i)} = W_{s,t}^{(j)}$, denoted $(i,k,l) \sim (j,s,t)$. Shared weights are updated jointly during training.
- **sparsity**: Imposing a sparsity structure on the weight matrices, i.e., setting $W_{k,l}^{(i)} = 0$ for certain $(k,l,i)$. In the graph representation, this corresponds to not connecting certain nodes.

Both restrictions decrease the number of learnable parameters. The number of parameters serves as a measure of the complexity of the represented function class.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.4</span><span class="math-callout__name">(Size of a Neural Network)</span></p>

Let $\Phi$ be as in Definition 2.1. Then the **size** of $\Phi$ is

$$\text{size}(\Phi) := \left\lvert \left( \lbrace (i,k,l) \mid W_{k,l}^{(i)} \neq 0 \rbrace \cup \lbrace (i,k) \mid b_k^{(i)} \neq 0 \rbrace \right) / \!\sim \right\rvert.$$

</div>

The size counts the number of learnable (non-zero, up to sharing equivalence) parameters.

### Activation Functions

Activation functions are crucial as they introduce nonlinearity into the model. Without them, the network function would be affine and hence very restricted.

**Sigmoid.** The sigmoid activation function is given by

$$\sigma_{\text{sig}}(x) = \frac{1}{1 + e^{-x}} \quad \text{for } x \in \mathbb{R}.$$

Its output ranges between zero and one, making it interpretable as a probability. It is smooth, enabling gradient-based training. However, its derivative becomes very small as $\lvert x \rvert \to \infty$, leading to the **vanishing gradient problem**: for a network $\Phi_n(x) = \sigma \circ \cdots \circ \sigma(x + b)$ with $n$ compositions, if $\sup_{x \in \mathbb{R}} \lvert \sigma'(x) \rvert \le 1 - \delta$, then

$$\left\lvert \frac{\mathrm{d}}{\mathrm{d}b} \Phi_n(x) \right\rvert \le (1-\delta)^n.$$

The opposite effect (derivatives uniformly larger than one) leads to the **exploding gradient effect**. Since the sigmoid has areas with extremely small gradients, the vanishing gradient effect can be strongly exacerbated.

**ReLU (Rectified Linear Unit).** The ReLU is defined as

$$\sigma_{\text{ReLU}}(x) = \max\lbrace x, 0 \rbrace \quad \text{for } x \in \mathbb{R}.$$

It is piecewise linear and computationally very efficient. Its derivative is always zero or one, so it does not suffer from the vanishing gradient problem to the same extent as the sigmoid. However, ReLU can suffer from the **dead neurons** problem: if $b < 0$, the neuron $\Phi(x) = \sigma_{\text{ReLU}}(b - \sigma_{\text{ReLU}}(x))$ satisfies $\Phi(x) = 0$ for all $x$, and $\frac{\mathrm{d}}{\mathrm{d}b}\Phi(x) = 0$, so a gradient-based method cannot further train this parameter.

**SiLU (Sigmoid Linear Unit).** Also referred to as "swish", the SiLU is a smooth approximation to the ReLU:

$$\sigma_{\text{SiLU}}(x) := x \cdot \sigma_{\text{sig}}(x) = \frac{x}{1 + e^{-x}} \quad \text{for } x \in \mathbb{R}.$$

Other smooth ReLU-like activations include the **Softplus** $x \mapsto \log(1 + \exp(x))$, the **GELU** $x \mapsto x F(x)$ (where $F$ is the standard normal CDF), and the **Mish** $x \mapsto x \tanh(\log(1 + \exp(x)))$.

**Parametric ReLU or Leaky ReLU.** For some $a \in (0,1)$, the parametric ReLU is

$$\sigma_a(x) = \max\lbrace x, ax \rbrace \quad \text{for } x \in \mathbb{R}.$$

Since $\sigma_a$ does not have flat regions, the dying ReLU problem is mitigated and there is less of a vanishing gradient problem. Like the ReLU, the parametric ReLU is not differentiable at 0.

---

## Chapter 3: Universal Approximation

After introducing neural networks in Chapter 2, a natural question arises: are there inherent limitations to the type of functions a neural network can represent? Could neural networks be specialized tools suited only for certain relationships? In this chapter we show that this is not the case -- neural networks are indeed a *universal* tool. Given sufficiently large and complex architectures, they can approximate almost every sensible input-output relationship.

### A Universal Approximation Theorem

We begin by considering the uniform approximation of continuous functions $f \colon \mathbb{R}^d \to \mathbb{R}$ on compact sets.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.1</span><span class="math-callout__name">(Compact Convergence)</span></p>

Let $d \in \mathbb{N}$. A sequence of functions $f_n \colon \mathbb{R}^d \to \mathbb{R}$, $n \in \mathbb{N}$, is said to **converge compactly** to a function $f \colon \mathbb{R}^d \to \mathbb{R}$, if for every compact $K \subseteq \mathbb{R}^d$ it holds that $\lim_{n \to \infty} \sup_{\boldsymbol{x} \in K} \lvert f_n(\boldsymbol{x}) - f(\boldsymbol{x}) \rvert = 0$. In this case we write $f_n \xrightarrow{\text{cc}} f$.

</div>

Throughout what follows, we always consider $C^0(\mathbb{R}^d)$ equipped with the topology of compact convergence. If $D \subseteq \mathbb{R}^d$ is bounded, then convergence in $C^0(D)$ refers to uniform convergence.

#### Universal Approximators

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.2</span><span class="math-callout__name">(Universal Approximator)</span></p>

Let $d \in \mathbb{N}$. A set of functions $\mathcal{H}$ from $\mathbb{R}^d$ to $\mathbb{R}$ is a **universal approximator** (of $C^0(\mathbb{R}^d)$), if for every $\varepsilon > 0$, every compact $K \subseteq \mathbb{R}^d$, and every $f \in C^0(\mathbb{R}^d)$, there exists $g \in \mathcal{H}$ such that $\sup_{\boldsymbol{x} \in K} \lvert f(\boldsymbol{x}) - g(\boldsymbol{x}) \rvert < \varepsilon$.

</div>

We denote by $\overline{\mathcal{H}}^{\text{cc}}$ the closure of $\mathcal{H}$ with respect to compact convergence.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.3</span><span class="math-callout__name">(Characterization of Universal Approximators)</span></p>

Let $d \in \mathbb{N}$ and $\mathcal{H}$ be a set of functions from $\mathbb{R}^d$ to $\mathbb{R}$. Then $\mathcal{H}$ is a universal approximator of $C^0(\mathbb{R}^d)$ if and only if $C^0(\mathbb{R}^d) \subseteq \overline{\mathcal{H}}^{\text{cc}}$.

</div>

A key tool is the Stone-Weierstrass theorem:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.4</span><span class="math-callout__name">(Stone-Weierstrass)</span></p>

Let $d \in \mathbb{N}$, let $K \subseteq \mathbb{R}^d$ be compact, and let $\mathcal{H} \subseteq C^0(K, \mathbb{R})$ satisfy that

- **(a)** for all $\boldsymbol{x} \in K$ there exists $f \in \mathcal{H}$ such that $f(\boldsymbol{x}) \neq 0$,
- **(b)** for all $\boldsymbol{x} \neq \boldsymbol{y} \in K$ there exists $f \in \mathcal{H}$ such that $f(\boldsymbol{x}) \neq f(\boldsymbol{y})$,
- **(c)** $\mathcal{H}$ is an algebra of functions, i.e., $\mathcal{H}$ is closed under addition, multiplication and scalar multiplication.

Then $\mathcal{H}$ is dense in $C^0(K)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.5</span><span class="math-callout__name">(Polynomials are a Universal Approximator)</span></p>

For a multiindex $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_d) \in \mathbb{N}_0^d$ and a vector $\boldsymbol{x} = (x_1, \ldots, x_d) \in \mathbb{R}^d$, denote $\boldsymbol{x}^{\boldsymbol{\alpha}} := \prod_{j=1}^d x_j^{\alpha_j}$. The space of polynomials of degree at most $n$ is

$$\mathcal{P}_n := \text{span}\lbrace \boldsymbol{x}^{\boldsymbol{\alpha}} \mid \boldsymbol{\alpha} \in \mathbb{N}_0^d,\; \lvert\boldsymbol{\alpha}\rvert \le n \rbrace.$$

The set $\mathcal{P} := \bigcup_{n \in \mathbb{N}} \mathcal{P}_n$ satisfies the assumptions of Theorem 3.4 on every compact $K \subseteq \mathbb{R}^d$, so $\mathcal{P}$ is a universal approximator of $C^0(\mathbb{R}^d)$.

</div>

#### Shallow Neural Networks

We now show that shallow neural networks of arbitrary width form a universal approximator under certain (mild) conditions on the activation function.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.6</span><span class="math-callout__name">(Neural Network Function Classes)</span></p>

Let $d$, $m$, $L$, $n \in \mathbb{N}$ and $\sigma \colon \mathbb{R} \to \mathbb{R}$. The set of all functions realized by neural networks with $d$-dimensional input, $m$-dimensional output, depth at most $L$, width at most $n$, and activation function $\sigma$ is denoted by

$$\mathcal{N}_d^m(\sigma; L, n) := \lbrace \Phi \colon \mathbb{R}^d \to \mathbb{R}^m \mid \Phi \text{ as in Def. 2.1},\; \text{depth}(\Phi) \le L,\; \text{width}(\Phi) \le n \rbrace.$$

Furthermore, $\mathcal{N}_d^m(\sigma; L) := \bigcup_{n \in \mathbb{N}} \mathcal{N}_d^m(\sigma; L, n)$.

</div>

We require the activation function $\sigma$ to belong to the set of piecewise continuous and locally bounded functions:

$$\mathcal{M} := \left\lbrace \sigma \in L_{\text{loc}}^\infty(\mathbb{R}) \;\middle\vert\; \text{there exist intervals } I_1, \ldots, I_M \text{ partitioning } \mathbb{R}, \; \sigma \in C^0(I_j) \text{ for all } j \right\rbrace.$$

This includes all practically relevant activation functions (ReLU, SiLU, Sigmoid) as well as discontinuous functions like the Heaviside function.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.8</span><span class="math-callout__name">(Universal Approximation Theorem)</span></p>

Let $d \in \mathbb{N}$ and $\sigma \in \mathcal{M}$. Then $\mathcal{N}_d^1(\sigma; 1)$ is a universal approximator of $C^0(\mathbb{R}^d)$ if and only if $\sigma$ is not a polynomial.

</div>

The proof strategy relies on three claims:

1. If $C^0(\mathbb{R}^1) \subseteq \overline{\mathcal{N}_1^1(\sigma; 1)}^{\text{cc}}$ then $C^0(\mathbb{R}^d) \subseteq \overline{\mathcal{N}_d^1(\sigma; 1)}^{\text{cc}}$.
2. If $\sigma \in C^\infty(\mathbb{R})$ is not a polynomial then $C^0(\mathbb{R}^1) \subseteq \overline{\mathcal{N}_1^1(\sigma; 1)}^{\text{cc}}$.
3. If $\sigma \in \mathcal{M}$ is not a polynomial then there exists $\tilde{\sigma} \in C^\infty(\mathbb{R}) \cap \overline{\mathcal{N}_1^1(\sigma; 1)}^{\text{cc}}$ which is not a polynomial.

The key lemmas establishing these claims are:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.10</span><span class="math-callout__name">(Dimension Reduction)</span></p>

Assume that $\mathcal{H}$ is a universal approximator of $C^0(\mathbb{R})$. Then for every $d \in \mathbb{N}$

$$\text{span}\lbrace \boldsymbol{x} \mapsto g(\boldsymbol{w} \cdot \boldsymbol{x}) \mid \boldsymbol{w} \in \mathbb{R}^d,\; g \in \mathcal{H} \rbrace$$

is a universal approximator of $C^0(\mathbb{R}^d)$.

</div>

The proof uses the Hahn-Banach theorem to show that all $k$-homogeneous polynomials $\mathbb{H}_k$ belong to the closure $X := \overline{\text{span}\lbrace \boldsymbol{x} \mapsto g(\boldsymbol{w} \cdot \boldsymbol{x}) \mid \boldsymbol{w} \in \mathbb{R}^d,\; g \in \mathcal{H} \rbrace}^{\text{cc}}$. By the multinomial formula, $(\boldsymbol{w} \cdot \boldsymbol{x})^k \in \mathbb{H}_k$, and since all multivariate polynomials belong to $X$, the Stone-Weierstrass theorem concludes the proof.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.11</span><span class="math-callout__name">(Sigmoidal Activation Function)</span></p>

An activation function $\sigma \colon \mathbb{R} \to \mathbb{R}$ is called **sigmoidal** if $\sigma \in C^0(\mathbb{R})$, $\lim_{x \to \infty} \sigma(x) = 1$ and $\lim_{x \to -\infty} \sigma(x) = 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.12</span><span class="math-callout__name">(Sigmoidal Universality)</span></p>

Let $\sigma \colon \mathbb{R} \to \mathbb{R}$ be monotonically increasing and sigmoidal. Then $C^0(\mathbb{R}) \subseteq \overline{\mathcal{N}_1^1(\sigma; 1)}^{\text{cc}}$.

</div>

Lemmas 3.10 and 3.12 together establish Theorem 3.8 in the special case where $\sigma$ is monotonically increasing and sigmoidal.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.13</span><span class="math-callout__name">(Smooth Non-Polynomial Universality)</span></p>

If $\sigma \in C^\infty(\mathbb{R})$ and $\sigma$ is not a polynomial, then $\mathcal{N}_1^1(\sigma; 1)$ is dense in $C^0(\mathbb{R})$.

</div>

*Proof sketch.* Let $X := \overline{\mathcal{N}_1^1(\sigma; 1)}^{\text{cc}}$. Fix $b \in \mathbb{R}$ and denote $f_x(w) := \sigma(wx + b)$. By Taylor's theorem,

$$\frac{\sigma((w+h)x + b) - \sigma(wx + b)}{h} = f_x'(w) + \frac{h}{2} x^2 \sigma''(\xi x + b).$$

Since $\sigma'' \in C^0(\mathbb{R})$, letting $h \to 0$ shows $x \mapsto f_x'(w)$ belongs to $X$. By induction, $x \mapsto f_x^{(k)}(w) = x^k \sigma^{(k)}(wx + b)$ belongs to $X$ for all $k \in \mathbb{N}$. Since $\sigma$ is not a polynomial, there exist $b_k$ with $\sigma^{(k)}(b_k) \neq 0$. Choosing $w = 0$, we get $x \mapsto \sigma^{(k)}(b_k) x^k \in X$, and thus $x \mapsto x^k \in X$. The Stone-Weierstrass theorem concludes the proof.

For claim (iii) -- that every non-polynomial $\sigma \in \mathcal{M}$ has a smooth non-polynomial function in the closure of $\mathcal{N}_1^1(\sigma; 1)$ -- the argument uses convolutions:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.14</span><span class="math-callout__name">(Convolution Closure)</span></p>

Let $\sigma \in \mathcal{M}$. Then for each $\varphi \in C_c^\infty(\mathbb{R})$ it holds $\sigma * \varphi \in \overline{\mathcal{N}_1^1(\sigma; 1)}^{\text{cc}}$.

</div>

*Proof sketch.* For $\varphi$ with $\text{supp}\,\varphi \subseteq [-a, a]$, define $f_n(x) := \frac{2a}{n} \sum_{j=0}^{n-1} \sigma(x - y_j) \varphi(y_j)$ where $y_j := -a + 2aj/n$. Each $f_n \in \mathcal{N}_1^1(\sigma; 1)$. By uniform continuity arguments, $f_n \xrightarrow{\text{cc}} \sigma * \varphi$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.15</span></p>

If $\sigma \in \mathcal{M}$ and $\sigma * \varphi$ is a polynomial for all $\varphi \in C_c^\infty(\mathbb{R})$, then $\sigma$ is a polynomial.

</div>

*Proof sketch.* Uses Baire's category theorem. Define $V_k := \lbrace \varphi \in C_c^\infty(a,b) \mid \sigma * \varphi \in \mathcal{P}_k \rbrace$. By assumption $\bigcup_{k} V_k = C_c^\infty(a,b)$. Baire's theorem implies some $V_{k_0}$ contains an open set, hence $V_{k_0} = C_c^\infty(a,b)$. This forces $\sigma \in \mathcal{P}_{k_0}$.

*Proof of Theorem 3.8.* If $\sigma$ is a polynomial, Exercise 3.23 gives the "$\Rightarrow$" direction. For the other direction, assume $\sigma \in \mathcal{M}$ is not a polynomial. By Lemma 3.15, there exists $\varphi \in C_c^\infty(\mathbb{R})$ such that $\sigma * \varphi$ is not a polynomial. By Lemma 3.14, $\sigma * \varphi \in \overline{\mathcal{N}_1^1(\sigma; 1)}^{\text{cc}}$. By Lemma 3.13, $\mathcal{N}_1^1(\sigma; 1)$ is a universal approximator of $C^0(\mathbb{R})$. By Lemma 3.10, $\mathcal{N}_d^1(\sigma; 1)$ is a universal approximator of $C^0(\mathbb{R}^d)$.

#### Deep Neural Networks

Theorem 3.8 shows the universal approximation capability of single-hidden-layer neural networks. This extends directly to deeper networks: the identity function can be approximated with a shallow neural network, and composing a shallow approximation of $f$ with shallow approximations of the identity yields a deep approximation of $f$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.16</span><span class="math-callout__name">(Identity Approximation)</span></p>

Let $d$, $L \in \mathbb{N}$, let $K \subseteq \mathbb{R}^d$ be compact, and let $\sigma \colon \mathbb{R} \to \mathbb{R}$ be such that there exists an open set on which $\sigma$ is differentiable and not constant. Then, for every $\varepsilon > 0$, there exists a neural network $\Phi \in \mathcal{N}_d^d(\sigma; L, d)$ such that

$$\lVert \Phi(\boldsymbol{x}) - \boldsymbol{x} \rVert_\infty < \varepsilon \quad \text{for all } \boldsymbol{x} \in K.$$

</div>

*Proof sketch.* Let $x^* \in \mathbb{R}$ be such that $\sigma'(x^*) = \theta \neq 0$. Define $\Phi_\lambda(\boldsymbol{x}) := \frac{\lambda}{\theta} \sigma\!\left(\frac{\boldsymbol{x}}{\lambda} + \boldsymbol{x}^*\right) - \frac{\lambda}{\theta}\sigma(\boldsymbol{x}^*)$. By the definition of the derivative, $\lvert (\Phi_\lambda(\boldsymbol{x}) - \boldsymbol{x})_i \rvert \to 0$ as $\lambda \to \infty$ uniformly for all $\boldsymbol{x} \in K$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.17</span><span class="math-callout__name">(Deep Universal Approximation)</span></p>

Let $d \in \mathbb{N}$, $L \in \mathbb{N}$ and $\sigma \in \mathcal{M}$. Then $\mathcal{N}_d^1(\sigma; L)$ is a universal approximator of $C^0(\mathbb{R}^d)$ if and only if $\sigma$ is not a polynomial.

</div>

*Proof.* Assume $\sigma \in \mathcal{M}$ is not a polynomial, $K \subseteq \mathbb{R}^d$ compact, $f \in C^0(\mathbb{R}^d)$, $\varepsilon \in (0,1)$. For $L = 1$ this is Theorem 3.8. For $L > 1$: by Theorem 3.8, find $\Phi_{\text{shallow}} \in \mathcal{N}_d^1(\sigma; 1)$ with $\sup_{\boldsymbol{x} \in K} \lvert f(\boldsymbol{x}) - \Phi_{\text{shallow}}(\boldsymbol{x}) \rvert < \varepsilon/2$. By compactness, $\lbrace \Phi_{\text{shallow}}(\boldsymbol{x}) \mid \boldsymbol{x} \in K \rbrace \subseteq [-n, n]$. Let $\Phi_{\text{id}} \in \mathcal{N}_1^1(\sigma; L-1)$ approximate the identity on $[-n, n]$ to within $\varepsilon/2$. Then $\Phi := \Phi_{\text{id}} \circ \Phi_{\text{shallow}} \in \mathcal{N}_d^1(\sigma; L)$ by Proposition 2.3 (iv), and $\sup_{\boldsymbol{x} \in K} \lvert f(\boldsymbol{x}) - \Phi(\boldsymbol{x}) \rvert \le \varepsilon/2 + \varepsilon/2 = \varepsilon$.

#### Other Norms

Universal approximation results extend beyond the supremum norm:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.18</span><span class="math-callout__name">($L^p$ Universal Approximation)</span></p>

Let $d \in \mathbb{N}$, $L \in \mathbb{N}$, $p \in [1, \infty)$, and let $\sigma \in \mathcal{M}$ not be a polynomial. Then for every $\varepsilon > 0$, every compact $K \subseteq \mathbb{R}^d$, and every $f \in L^p(K)$ there exists $\Phi^{f,\varepsilon} \in \mathcal{N}_d^1(\sigma; L)$ such that

$$\left( \int_K \lvert f(\boldsymbol{x}) - \Phi(\boldsymbol{x}) \rvert^p \, \mathrm{d}\boldsymbol{x} \right)^{1/p} \le \varepsilon.$$

</div>

### Superexpressive Activations and Kolmogorov's Superposition Theorem

The previous section showed that a large class of activation functions allows for universal approximation, but provided no insight into the necessary neural network size. Here we present a remarkable result: with the right activation function, every $f \in C^0(K)$ on a compact $K \subseteq \mathbb{R}^d$ can be approximated to *any desired accuracy* $\varepsilon > 0$ using a neural network of size $O(d^2)$, independent of $\varepsilon$, $K$, and $f$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.19</span><span class="math-callout__name">(Superexpressive Activation -- 1D)</span></p>

There exists a continuous activation function $\sigma \colon \mathbb{R} \to \mathbb{R}$ such that for every compact $K \subseteq \mathbb{R}$, every $\varepsilon > 0$ and every $f \in C^0(K)$ there exists $\Phi(x) = \sigma(wx + b) \in \mathcal{N}_1^1(\sigma; 1, 1)$ such that

$$\sup_{x \in K} \lvert f(x) - \Phi(x) \rvert < \varepsilon.$$

</div>

*Proof sketch.* Let $(p_i)_{i \in \mathbb{Z}}$ be an enumeration of all polynomials with rational coefficients. Define $\sigma$ so that $\sigma$ equals $p_i$ on even intervals $[2i, 2i+1]$ and is linear on odd intervals. Since rational polynomials are dense in $C^0(K)$ (Example 3.5), for any $f$ and $\varepsilon$, we can find a polynomial $p_i$ close to $f$, then use $\sigma(wx + b)$ with $w = 1/(b-a)$ and appropriate $b$ to scale the input into the interval where $\sigma$ reproduces $p_i$.

The key to extending this to arbitrary dimension is Kolmogorov's superposition theorem:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.20</span><span class="math-callout__name">(Kolmogorov's Superposition Theorem)</span></p>

For every $d \in \mathbb{N}$ there exist $2d^2 + d$ monotonically increasing functions $\varphi_{i,j} \in C^0(\mathbb{R})$, $i = 1, \ldots, d$, $j = 1, \ldots, 2d+1$, such that for every $f \in C^0([0,1]^d)$ there exist functions $f_j \in C^0(\mathbb{R})$, $j = 1, \ldots, 2d+1$ satisfying

$$f(\boldsymbol{x}) = \sum_{j=1}^{2d+1} f_j\!\left( \sum_{i=1}^d \varphi_{i,j}(x_i) \right) \quad \text{for all } \boldsymbol{x} \in [0,1]^d.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.21</span><span class="math-callout__name">(Superexpressive Activation -- General)</span></p>

Let $d \in \mathbb{N}$. With the activation function $\sigma$ from Proposition 3.19, for every compact $K \subseteq \mathbb{R}^d$, every $\varepsilon > 0$ and every $f \in C^0(K)$ there exists $\Phi \in \mathcal{N}_d^1(\sigma; 2, 2d^2 + d)$ (i.e. $\text{width}(\Phi) = 2d^2 + d$ and $\text{depth}(\Phi) = 2$) such that

$$\sup_{\boldsymbol{x} \in K} \lvert f(\boldsymbol{x}) - \Phi(\boldsymbol{x}) \rvert < \varepsilon.$$

</div>

*Proof sketch.* By Kolmogorov's theorem, $f(\boldsymbol{x}) = \sum_{j=1}^{2d+1} f_j(\sum_{i=1}^d \varphi_{i,j}(x_i))$. By Proposition 3.19, each $\varphi_{i,j}$ and $f_j$ can be approximated by single neurons $\sigma(w_{i,j}x + b_{i,j})$ and $\sigma(w_j y + b_j)$ respectively. This yields a depth-2 network of width $2d^2 + d$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Relevance)</span></p>

Kolmogorov's superposition theorem is intriguing as it reduces approximating $d$-dimensional functions to the one-dimensional case through compositions. However, the functions $f_j$ in the decomposition can become very complex for large $d$. Similarly, the "magic" activation function in Proposition 3.19 encodes all rational polynomials, and no practical algorithm can efficiently identify appropriate weights and biases. The results of Section 3.2 should therefore be taken with caution as their practical relevance is limited. They highlight that while universal approximation is fundamental, many aspects remain unexplored. In the following chapters, the focus shifts to practically relevant architectures with activation functions such as the ReLU.

</div>

---

## Chapter 4: Splines

In Chapter 3, we saw that sufficiently large neural networks can approximate every continuous function to arbitrary accuracy. However, those results did not specify the meaning of "sufficiently large" or what constitutes a suitable architecture. Ideally, given a function $f$ and a desired accuracy $\varepsilon > 0$, we would like a (possibly sharp) bound on the required size, depth, and width guaranteeing the existence of a neural network approximating $f$ up to error $\varepsilon$.

The field of approximation theory establishes trade-offs between properties of $f$ (e.g., smoothness), the approximation accuracy, and the number of parameters needed. For example, given $k, d \in \mathbb{N}$, how many parameters are required to approximate a function $f \colon [0,1]^d \to \mathbb{R}$ with $\lVert f \rVert_{C^k([0,1]^d)} \le 1$ up to uniform error $\varepsilon$? Splines achieve this with a superposition of $O(\varepsilon^{-d/k})$ simple basis functions. In this chapter, we show that certain sigmoidal neural networks can match this performance, establishing that from an approximation-theoretic viewpoint, these neural networks are at least as expressive as superpositions of splines.

### B-Splines and Smooth Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1</span><span class="math-callout__name">(Cardinal B-Spline)</span></p>

For $n \in \mathbb{N}$, the **univariate cardinal B-spline** of order $n \in \mathbb{N}$ is given by

$$\mathcal{S}_n(x) := \frac{1}{(n-1)!} \sum_{\ell=0}^n (-1)^\ell \binom{n}{\ell} \sigma_{\text{ReLU}}(x - \ell)^{n-1} \quad \text{for } x \in \mathbb{R},$$

where $0^0 := 0$ and $\sigma_{\text{ReLU}}$ denotes the ReLU activation function.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.2</span><span class="math-callout__name">(Multivariate B-Spline)</span></p>

For $t \in \mathbb{R}$ and $n, \ell \in \mathbb{N}$ we define $\mathcal{S}_{\ell,t,n} := \mathcal{S}_n(2^\ell(\cdot - t))$. For $d \in \mathbb{N}$, $\boldsymbol{t} \in \mathbb{R}^d$, and $n, \ell \in \mathbb{N}$, the **multivariate B-spline** $\mathcal{S}_{\ell,\boldsymbol{t},n}^d$ is defined as

$$\mathcal{S}_{\ell,\boldsymbol{t},n}^d(\boldsymbol{x}) := \prod_{i=1}^d \mathcal{S}_{\ell,t_i,n}(x_i) \quad \text{for } \boldsymbol{x} = (x_1, \ldots, x_d) \in \mathbb{R}^d,$$

and $\mathcal{B}^n := \left\lbrace \mathcal{S}_{\ell,\boldsymbol{t},n}^d \;\middle\vert\; \ell \in \mathbb{N}, \boldsymbol{t} \in \mathbb{R}^d \right\rbrace$ is the **dictionary of B-splines** of order $n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.3</span><span class="math-callout__name">(B-Spline Approximation Rate)</span></p>

Let $d$, $n$, $k \in \mathbb{N}$ such that $0 < k \le n$. Then there exists $C$ such that for every $f \in C^k([0,1]^d)$ and every $N \in \mathbb{N}$, there exist $c_i \in \mathbb{R}$ with $\lvert c_i \rvert \le C \lVert f \rVert_{L^\infty([0,1]^d)}$ and $B_i \in \mathcal{B}^n$ for $i = 1, \ldots, N$, such that

$$\left\lVert f - \sum_{i=1}^N c_i B_i \right\rVert_{L^\infty([0,1]^d)} \le C N^{-k/d} \lVert f \rVert_{C^k([0,1]^d)}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.4</span><span class="math-callout__name">(Key Observations)</span></p>

There are several critical concepts in Theorem 4.3 that will reappear throughout this book:

- The number of parameters $N$ determines the approximation accuracy $N^{-k/d}$. Achieving accuracy $\varepsilon > 0$ requires $O(\varepsilon^{-d/k})$ parameters, which grows **exponentially in $d$**. This is the **curse of dimensionality**.
- The smoothness parameter $k$ has the opposite effect: smoother functions can be approximated with fewer B-splines. However, more efficient approximation requires B-splines of order $n$ with $n \ge k$.
- The order of the B-spline is closely linked to the concept of **depth** in neural networks.

</div>

### Reapproximation of B-Splines with Sigmoidal Activations

We now show that the approximation rates of B-splines can be transferred to certain neural networks.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.5</span><span class="math-callout__name">(Sigmoidal of Order $q$)</span></p>

A function $\sigma \colon \mathbb{R} \to \mathbb{R}$ is called **sigmoidal of order** $q \in \mathbb{N}$, if $\sigma \in C^{q-1}(\mathbb{R})$ and there exists $C > 0$ such that

$$\frac{\sigma(x)}{x^q} \to 0 \quad \text{as } x \to -\infty, \qquad \frac{\sigma(x)}{x^q} \to 1 \quad \text{as } x \to +\infty, \qquad \lvert \sigma(x) \rvert \le C \cdot (1 + \lvert x \rvert)^q \quad \text{for all } x \in \mathbb{R}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.6</span></p>

The rectified power unit $x \mapsto \sigma_{\text{ReLU}}(x)^q$ is sigmoidal of order $q$.

</div>

The strategy is to show that neural networks can approximate a linear combination of $N$ B-splines with a number of parameters proportional to $N$, and then apply Theorem 4.3 to obtain convergence rates.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.7</span><span class="math-callout__name">(Univariate B-Spline Approximation by Neural Networks)</span></p>

Let $n \in \mathbb{N}$, $n \ge 2$, $K > 0$, and let $\sigma \colon \mathbb{R} \to \mathbb{R}$ be sigmoidal of order $q \ge 2$. There exists a constant $C > 0$ such that for every $\varepsilon > 0$ there is a neural network $\Phi^{\mathcal{S}_n}$ with activation function $\sigma$, $\lceil \log_q(n-1) \rceil$ layers, and size $C$, such that

$$\left\lVert \mathcal{S}_n - \Phi^{\mathcal{S}_n} \right\rVert_{L^\infty([-K,K])} \le \varepsilon.$$

</div>

*Proof sketch.* By definition, $\mathcal{S}_n$ is a linear combination of $n+1$ shifts of $\sigma_{\text{ReLU}}^{n-1}$. The key step is approximating $\sigma_{\text{ReLU}}(x)^{n-1}$ using $t := \lceil \log_q(n-1) \rceil$ compositions of $\sigma$: since $\sigma$ is sigmoidal of order $q$, the $t$-fold composition $a^{-q^t} \underbrace{\sigma \circ \sigma \circ \cdots \circ \sigma}_{t \text{ times}}(ax)$ converges to $\sigma_{\text{ReLU}}(x)^{q^t}$ as $a \to \infty$. Since $q^t \ge n-1$, one can emulate approximate derivatives of these compositions (via finite differences) to reduce the power from $q^t$ down to any $p \le q^t$, including $p = n-1$. Each spatial translation $\Phi(\cdot - t)$ is a neural network of the same architecture, and sums of neural networks of the same depth are again neural networks of the same depth (Proposition 2.3).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.8</span><span class="math-callout__name">(Multivariate B-Spline Approximation by Neural Networks)</span></p>

Let $n, d \in \mathbb{N}$, $n \ge 2$, $K > 0$, and let $\sigma \colon \mathbb{R} \to \mathbb{R}$ be sigmoidal of order $q \ge 2$. Further let $\ell \in \mathbb{N}$ and $\boldsymbol{t} \in \mathbb{R}^d$.

Then, there exists a constant $C > 0$ such that for every $\varepsilon > 0$ there is a neural network $\Phi^{\mathcal{S}_{\ell,\boldsymbol{t},n}^d}$ with activation function $\sigma$, $\lceil \log_2(d) \rceil + \lceil \log_q(n-1) \rceil$ layers, and size $C$, such that

$$\left\lVert \mathcal{S}_{\ell,\boldsymbol{t},n}^d - \Phi^{\mathcal{S}_{\ell,\boldsymbol{t},n}^d} \right\rVert_{L^\infty([-K,K]^d)} \le \varepsilon.$$

</div>

*Proof sketch.* Since $\mathcal{S}_{\ell,\boldsymbol{t},n}^d(\boldsymbol{x}) = \prod_{i=1}^d \mathcal{S}_{\ell,t_i,n}(x_i)$, the problem reduces to approximating each univariate factor (via Proposition 4.7) and then approximating the $d$-fold product. The product of $d$ numbers is approximated by a neural network of depth $\lceil \log_2(d) \rceil$ using the identity

$$x_1 x_2 = \frac{1}{2}\left((x_1 + x_2)^2 - x_1^2 - x_2^2\right)$$

and the fact that $\sigma_{\text{ReLU}}(x)^2$ can be approximated by a sigmoidal network. The general $d$-fold product is handled by a divide-and-conquer strategy: split the product into $\prod_{i=1}^{\lfloor d/2 \rfloor} x_i \cdot \prod_{i=\lfloor d/2 \rfloor+1}^{d} x_i$ and recurse, requiring $\lceil \log_2(d) \rceil$ layers.

Combining Proposition 4.8 with Theorem 4.3 yields the main result:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.9</span><span class="math-callout__name">(Neural Network Approximation Rate via Splines)</span></p>

Let $d$, $n$, $k \in \mathbb{N}$ such that $0 < k \le n$ and $n \ge 2$. Let $q \ge 2$ and let $\sigma$ be sigmoidal of order $q$.

Then there exists $C$ such that for every $f \in C^k([0,1]^d)$ and every $N \in \mathbb{N}$ there exists a neural network $\Phi^N$ with activation function $\sigma$, $\lceil \log_2(d) \rceil + \lceil \log_q(k-1) \rceil$ layers, and size bounded by $CN$, such that

$$\left\lVert f - \Phi^N \right\rVert_{L^\infty([0,1]^d)} \le C N^{-k/d} \lVert f \rVert_{C^k([0,1]^d)}.$$

</div>

*Proof.* Fix $N \in \mathbb{N}$. By Theorem 4.3, there exist coefficients $\lvert c_i \rvert \le C \lVert f \rVert_{L^\infty}$ and B-splines $B_i \in \mathcal{B}^n$ such that $\lVert f - \sum_{i=1}^N c_i B_i \rVert_{L^\infty} \le C N^{-k/d} \lVert f \rVert_{C^k}$. By Proposition 4.8, for each $i = 1, \ldots, N$, there exists a neural network $\Phi^{B_i}$ with $\lceil \log_2(d) \rceil + \lceil \log_q(k-1) \rceil$ layers and a fixed size that approximates $B_i$ on $[-1,1]^d \supseteq [0,1]^d$ up to error $\varepsilon := N^{-k/d}/N$. The size of each $\Phi^{B_i}$ is independent of $i$ and $N$. By Proposition 2.3, $\Phi^N$ that uniformly approximates $\sum_{i=1}^N c_i B_i$ up to error $\varepsilon$ on $[0,1]^d$ has $\lceil \log_2(d) \rceil + \lceil \log_q(k-1) \rceil$ layers, and size linear in $N$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of Theorem 4.9)</span></p>

Theorem 4.9 shows that neural networks with higher-order sigmoidal activation functions can approximate smooth functions with the same accuracy as spline approximations while having a comparable number of parameters. The network depth is required to behave like $O(\log(k))$ in terms of the smoothness parameter $k$. This offers a first glimpse into the **importance of depth in deep learning**: achieving higher-order approximation (exploiting more smoothness) requires deeper networks, mirroring how higher-order B-splines require higher polynomial degree.

</div>

---

## Chapter 5: ReLU Neural Networks

In this chapter, we discuss feedforward neural networks using the ReLU activation function $\sigma_{\text{ReLU}}$. Due to its simplicity and the fact that it reduces the vanishing and exploding gradients phenomena, the ReLU is one of the most widely used activation functions in practice.

A key component of the proofs in the previous chapters was the approximation of derivatives of the activation function to emulate polynomials. Since the ReLU is piecewise linear, this trick is not applicable. This makes the analysis fundamentally different from the case of smoother activation functions. Nonetheless, even this extremely simple activation function yields a very rich class of functions possessing remarkable approximation capabilities.

### 5.1 Basic ReLU Calculus

The goal of this section is to formalize how to combine and manipulate ReLU neural networks. We sharpen Proposition 2.3 by adding bounds on the number of weights that the resulting neural networks have. The following four operations form the basis of all constructions:

- **Reproducing an identity:** For ReLUs, we can reproduce the identity exactly. This identity will play a crucial role in order to extend certain neural networks to deeper neural networks, and to facilitate an efficient composition operation.
- **Composition:** For ReLU activation functions, this composition can be done in a very efficient way leading to a neural network that has up to a constant not more than the number of weights of the two initial neural networks.
- **Parallelization:** We refine this notion and make precise the size of the resulting neural networks.
- **Linear combinations:** For the sum of two neural networks, we give precise bounds on the size of the resulting neural network.

#### 5.1.1 Identity

We start with expressing the identity on $\mathbb{R}^d$ as a neural network of depth $L \in \mathbb{N}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.1</span><span class="math-callout__name">(Identity)</span></p>

Let $L \in \mathbb{N}$. Then, there exists a ReLU neural network $\Phi_L^{\text{id}}$ such that $\Phi_L^{\text{id}}(\boldsymbol{x}) = \boldsymbol{x}$ for all $\boldsymbol{x} \in \mathbb{R}^d$. Moreover, $\text{depth}(\Phi_L^{\text{id}}) = L$, $\text{width}(\Phi_L^{\text{id}}) = 2d$, and $\text{size}(\Phi_L^{\text{id}}) = 2d \cdot (L+1)$.

</div>

*Proof.* Writing $\boldsymbol{I}_d \in \mathbb{R}^{d \times d}$ for the identity matrix, we choose the weights

$$(\boldsymbol{W}^{(0)}, \boldsymbol{b}^{(0)}), \ldots, (\boldsymbol{W}^{(L)}, \boldsymbol{b}^{(L)}) := \left(\begin{pmatrix} \boldsymbol{I}_d \\ -\boldsymbol{I}_d \end{pmatrix}, \boldsymbol{0}\right), \underbrace{(\boldsymbol{I}_{2d}, \boldsymbol{0}), \ldots, (\boldsymbol{I}_{2d}, \boldsymbol{0})}_{L-1 \text{ times}}, ((\boldsymbol{I}_d, -\boldsymbol{I}_d), \boldsymbol{0}).$$

Using that $x = \sigma_{\text{ReLU}}(x) - \sigma_{\text{ReLU}}(-x)$ for all $x \in \mathbb{R}$ and $\sigma_{\text{ReLU}}(x) = x$ for all $x \ge 0$, it is obvious that the neural network $\Phi_L^{\text{id}}$ associated to the weights above satisfies the assertion of the lemma.

The property to exactly represent the identity is not shared by sigmoidal activation functions. It does hold for polynomial activation functions though; also see Proposition 3.16.

#### 5.1.2 Composition

Assume we have two neural networks $\Phi_1$, $\Phi_2$ with architectures $(\sigma_{\text{ReLU}}; d_0^1, \ldots, d_{L_1+1}^1)$ and $(\sigma_{\text{ReLU}}; d_0^2, \ldots, d_{L_2+1}^2)$ respectively. If the output dimension $d_{L_1+1}^1$ of $\Phi_1$ equals the input dimension $d_0^2$ of $\Phi_2$, we can define two types of concatenations:

First, $\Phi_2 \circ \Phi_1$ is the neural network defined by directly composing the weight-bias tuples of $\Phi_1$ and $\Phi_2$, merging the output layer of $\Phi_1$ with the input layer of $\Phi_2$.

Second, $\Phi_2 \bullet \Phi_1$ is the neural network defined as $\Phi_2 \circ \Phi_1^{\text{id}} \circ \Phi_1$, which inserts an identity layer between the two networks by encoding the intermediate value using the ReLU identity construction.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.2</span><span class="math-callout__name">(Composition)</span></p>

Let $\Phi_1$, $\Phi_2$ be neural networks with architectures $(\sigma_{\text{ReLU}}; d_0^1, \ldots, d_{L_1+1}^1)$ and $(\sigma_{\text{ReLU}}; d_0^2, \ldots, d_{L_2+1}^2)$. Assume $d_{L_1+1}^1 = d_0^2$. Then $\Phi_2 \circ \Phi_1(\boldsymbol{x}) = \Phi_2 \bullet \Phi_1(\boldsymbol{x}) = \Phi_2(\Phi_1(\boldsymbol{x}))$ for all $\boldsymbol{x} \in \mathbb{R}^{d_0^1}$. Moreover,

$$\text{width}(\Phi_2 \circ \Phi_1) \le \max\lbrace\text{width}(\Phi_1), \text{width}(\Phi_2)\rbrace,$$

$$\text{depth}(\Phi_2 \circ \Phi_1) = \text{depth}(\Phi_1) + \text{depth}(\Phi_2),$$

$$\text{size}(\Phi_2 \circ \Phi_1) \le \text{size}(\Phi_1) + \text{size}(\Phi_2) + (d_{L_1}^1 + 1)d_1^2,$$

and

$$\text{width}(\Phi_2 \bullet \Phi_1) \le 2\max\lbrace\text{width}(\Phi_1), \text{width}(\Phi_2)\rbrace,$$

$$\text{depth}(\Phi_2 \bullet \Phi_1) = \text{depth}(\Phi_1) + \text{depth}(\Phi_2) + 1,$$

$$\text{size}(\Phi_2 \bullet \Phi_1) \le 2(\text{size}(\Phi_1) + \text{size}(\Phi_2)).$$

</div>

Interpreting linear transformations as neural networks of depth 0, the previous lemma is also valid in case $\Phi_1$ or $\Phi_2$ is a linear mapping.

#### 5.1.3 Parallelization

Let $(\Phi_i)_{i=1}^m$ be neural networks with architectures $(\sigma_{\text{ReLU}}; d_0^i, \ldots, d_{L_i+1}^i)$, respectively. We proceed to build a neural network $(\Phi_1, \ldots, \Phi_m)$ realizing the function

$$(\Phi_1, \ldots, \Phi_m) \colon \mathbb{R}^{\sum_{j=1}^m d_0^j} \to \mathbb{R}^{\sum_{j=1}^m d_{L_j+1}^j}$$

$$(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_m) \mapsto (\Phi_1(\boldsymbol{x}_1), \ldots, \Phi_m(\boldsymbol{x}_m)).$$

To do so we first assume $L_1 = \cdots = L_m = L$, and define $(\Phi_1, \ldots, \Phi_m)$ via block-diagonal weight matrices. For the general case where the $\Phi_j$ might have different depths, let $L_{\max} := \max_{1 \le i \le m} L_i$ and for each $j$ with $L_j < L_{\max}$ set $\widetilde{\Phi}_j := \Phi_{L_{\max} - L_j}^{\text{id}} \circ \Phi_j$ to equalize depths.

If all input dimensions $d_0^1 = \cdots = d_0^m =: d_0$ are the same, we will also use **parallelization with shared inputs**, realizing $\boldsymbol{x} \mapsto (\Phi_1(\boldsymbol{x}), \ldots, \Phi_m(\boldsymbol{x}))$ from $\mathbb{R}^{d_0} \to \mathbb{R}^{d_{L_1+1}^1 + \cdots + d_{L_m+1}^m}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.3</span><span class="math-callout__name">(Parallelization)</span></p>

Let $m \in \mathbb{N}$ and $(\Phi_i)_{i=1}^m$ be neural networks with architectures $(\sigma_{\text{ReLU}}; d_0^i, \ldots, d_{L_i+1}^i)$, respectively. Then the neural network $(\Phi_1, \ldots, \Phi_m)$ satisfies

$$(\Phi_1, \ldots, \Phi_m)(\boldsymbol{x}) = (\Phi_1(\boldsymbol{x}_1), \ldots, \Phi_m(\boldsymbol{x}_m)) \quad \text{for all } \boldsymbol{x} \in \mathbb{R}^{\sum_{j=1}^m d_0^j}.$$

Moreover, with $L_{\max} := \max_{j \le m} L_j$ it holds that

$$\text{width}((\Phi_1, \ldots, \Phi_m)) \le 2\sum_{j=1}^m \text{width}(\Phi_j),$$

$$\text{depth}((\Phi_1, \ldots, \Phi_m)) = \max_{j \le m} \text{depth}(\Phi_j),$$

$$\text{size}((\Phi_1, \ldots, \Phi_m)) \le 2\sum_{j=1}^m \text{size}(\Phi_j) + 2\sum_{j=1}^m (L_{\max} - L_j) d_{L_j+1}^j.$$

</div>

#### 5.1.4 Linear Combinations

Let $m \in \mathbb{N}$ and let $(\Phi_i)_{i=1}^m$ be ReLU neural networks with architectures $(\sigma_{\text{ReLU}}; d_0^i, \ldots, d_{L_i+1}^i)$, respectively. Assume that $d_{L_1+1}^1 = \cdots = d_{L_m+1}^m$, i.e., all $\Phi_1, \ldots, \Phi_m$ have the same output dimension. For scalars $\alpha_j \in \mathbb{R}$, we wish to construct a ReLU neural network $\sum_{j=1}^m \alpha_j \Phi_j$ realizing the function

$$(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_m) \mapsto \sum_{j=1}^m \alpha_j \Phi_j(\boldsymbol{x}_j).$$

This corresponds to the parallelization $(\Phi_1, \ldots, \Phi_m)$ composed with the linear transformation $(\boldsymbol{z}_1, \ldots, \boldsymbol{z}_m) \mapsto \sum_{j=1}^m \alpha_j \boldsymbol{z}_j$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.4</span><span class="math-callout__name">(Linear Combinations)</span></p>

Let $m \in \mathbb{N}$ and $(\Phi_i)_{i=1}^m$ be neural networks with architectures $(\sigma_{\text{ReLU}}; d_0^i, \ldots, d_{L_i+1}^i)$, respectively. Assume that $d_{L_1+1}^1 = \cdots = d_{L_m+1}^m$, let $\boldsymbol{\alpha} \in \mathbb{R}^m$ and set $L_{\max} := \max_{j \le m} L_j$. Then, there exists a neural network $\sum_{j=1}^m \alpha_j \Phi_j$ such that $\left(\sum_{j=1}^m \alpha_j \Phi_j\right)(\boldsymbol{x}) = \sum_{j=1}^m \alpha_j \Phi_j(\boldsymbol{x}_j)$ for all $\boldsymbol{x} = (\boldsymbol{x}_j)_{j=1}^m \in \mathbb{R}^{\sum_{j=1}^m d_0^j}$. Moreover,

$$\text{width}\left(\sum_{j=1}^m \alpha_j \Phi_j\right) \le 2\sum_{j=1}^m \text{width}(\Phi_j),$$

$$\text{depth}\left(\sum_{j=1}^m \alpha_j \Phi_j\right) = \max_{j \le m} \text{depth}(\Phi_j),$$

$$\text{size}\left(\sum_{j=1}^m \alpha_j \Phi_j\right) \le 2\sum_{j=1}^m \text{size}(\Phi_j) + 2\sum_{j=1}^m (L_{\max} - L_j) d_{L_j+1}^j.$$

</div>

### 5.2 Continuous Piecewise Linear Functions

In this section, we relate ReLU neural networks to a large class of functions. We first formally introduce the set of continuous piecewise linear functions from a set $\Omega \subseteq \mathbb{R}^d$ to $\mathbb{R}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.5</span><span class="math-callout__name">(Continuous Piecewise Linear Function)</span></p>

Let $\Omega \subseteq \mathbb{R}^d$, $d \in \mathbb{N}$. We call a function $f \colon \Omega \to \mathbb{R}$ **continuous, piecewise linear (cpwl)** if $f \in C^0(\Omega)$ and there exist $n \in \mathbb{N}$ affine functions $g_j \colon \mathbb{R}^d \to \mathbb{R}$, $g_j(\boldsymbol{x}) = \boldsymbol{w}_j^\top \boldsymbol{x} + b_j$ such that for each $\boldsymbol{x} \in \Omega$ it holds that $f(\boldsymbol{x}) = g_j(\boldsymbol{x})$ for at least one $j \in \lbrace 1, \ldots, n \rbrace$. For $m > 1$ we call $f \colon \Omega \to \mathbb{R}^m$ cpwl if and only if each component of $f$ is cpwl.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.6</span></p>

A "continuous piecewise linear function" as in Definition 5.5 is actually piecewise *affine*. To maintain consistency with the literature, we use the terminology cpwl. The connected domains on which $f$ is equal to one of the functions $g_j$ are called **regions** or **pieces**. If $f$ is cpwl with $q \in \mathbb{N}$ regions, then with $n \in \mathbb{N}$ denoting the number of affine functions it holds $n \le q$.

</div>

Note that the mapping $\boldsymbol{x} \mapsto \sigma_{\text{ReLU}}(\boldsymbol{w}^\top \boldsymbol{x} + b)$, which is a ReLU neural network with a single neuron, is cpwl (with two regions). Consequently, every ReLU neural network is a repeated composition of linear combinations of cpwl functions. It is not hard to see that the set of cpwl functions is closed under compositions and linear combinations. Hence, *every ReLU neural network is a cpwl function*. Interestingly, the reverse direction of this statement is also true, meaning that *every cpwl function can be represented by a ReLU neural network*. Therefore, we can identify the class of functions realized by arbitrary ReLU neural networks as the class of cpwl functions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.7</span><span class="math-callout__name">(ReLU Networks Realize cpwl Functions)</span></p>

Let $d \in \mathbb{N}$, let $\Omega \subseteq \mathbb{R}^d$ be convex, and let $f \colon \Omega \to \mathbb{R}$ be cpwl with $n \in \mathbb{N}$ as in Definition 5.5. Then, there exists a ReLU neural network $\Phi^f$ such that $\Phi^f(\boldsymbol{x}) = f(\boldsymbol{x})$ for all $\boldsymbol{x} \in \Omega$ and

$$\text{size}(\Phi^f) = O(dn2^n), \quad \text{width}(\Phi^f) = O(dn2^n), \quad \text{depth}(\Phi^f) = O(n).$$

</div>

The proof of Theorem 5.7 is based on the following well-known result: every cpwl function can be expressed as a finite maximum of a finite minimum of certain affine functions.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.8</span><span class="math-callout__name">(Max-Min Representation of cpwl Functions)</span></p>

Let $d \in \mathbb{N}$, $\Omega \subseteq \mathbb{R}^d$ be convex, and let $f \colon \Omega \to \mathbb{R}$ be cpwl with $n \in \mathbb{N}$ affine functions as in Definition 5.5. Then there exists $m \in \mathbb{N}$ and sets $s_j \subseteq \lbrace 1, \ldots, n \rbrace$ for $j \in \lbrace 1, \ldots, m \rbrace$, such that

$$f(\boldsymbol{x}) = \max_{1 \le j \le m} \min_{i \in s_j} (g_i(\boldsymbol{x})) \quad \text{for all } \boldsymbol{x} \in \Omega.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.9</span></p>

For any $a_1, \ldots, a_k \in \mathbb{R}$ holds $\min\lbrace -a_1, \ldots, -a_k \rbrace = -\max\lbrace a_1, \ldots, a_k \rbrace$. Thus, in the setting of Proposition 5.8, there exists $\tilde{m} \in \mathbb{N}$ and sets $\tilde{s}_j \subseteq \lbrace 1, \ldots, n \rbrace$ for $j = 1, \ldots, \tilde{m}$, such that for all $\boldsymbol{x} \in \Omega$:

$$f(\boldsymbol{x}) = \min_{1 \le j \le \tilde{m}} \left(\max_{i \in \tilde{s}_j} (g_i(\boldsymbol{x}))\right).$$

</div>

To prove Theorem 5.7, it therefore suffices to show that the minimum and the maximum are expressible by ReLU neural networks.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.10</span><span class="math-callout__name">(Min and Max via ReLU)</span></p>

For every $x, y \in \mathbb{R}$ it holds that

$$\min\lbrace x, y \rbrace = \sigma_{\text{ReLU}}(y) - \sigma_{\text{ReLU}}(-y) - \sigma_{\text{ReLU}}(y - x) \in \mathcal{N}_2^1(\sigma_{\text{ReLU}}; 1, 3)$$

and

$$\max\lbrace x, y \rbrace = \sigma_{\text{ReLU}}(y) - \sigma_{\text{ReLU}}(-y) + \sigma_{\text{ReLU}}(x - y) \in \mathcal{N}_2^1(\sigma_{\text{ReLU}}; 1, 3).$$

</div>

*Proof.* We have $\max\lbrace x, y \rbrace = y + \sigma_{\text{ReLU}}(x - y)$. Using $y = \sigma_{\text{ReLU}}(y) - \sigma_{\text{ReLU}}(-y)$, the claim for the maximum follows. For the minimum observe that $\min\lbrace x, y \rbrace = -\max\lbrace -x, -y \rbrace$.

The minimum of $n \ge 2$ inputs can be computed by repeatedly applying the construction of Lemma 5.10:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.11</span><span class="math-callout__name">(n-ary Min and Max Networks)</span></p>

For every $n \ge 2$ there exists a neural network $\Phi_n^{\min} \colon \mathbb{R}^n \to \mathbb{R}$ with

$$\text{size}(\Phi_n^{\min}) \le 16n, \quad \text{width}(\Phi_n^{\min}) \le 3n, \quad \text{depth}(\Phi_n^{\min}) \le \lceil \log_2(n) \rceil$$

such that $\Phi_n^{\min}(x_1, \ldots, x_n) = \min_{1 \le j \le n} x_j$. Similarly, there exists a neural network $\Phi_n^{\max} \colon \mathbb{R}^n \to \mathbb{R}$ realizing the maximum and satisfying the same complexity bounds.

</div>

*Proof sketch.* For $n = 2^k$, proceed by induction on $k$. Define $\Phi_{2^k}^{\min} := \Phi_2^{\min} \circ (\Phi_{2^{k-1}}^{\min}, \Phi_{2^{k-1}}^{\min})$, building a binary tree of pairwise minima. By Lemma 5.2 and Lemma 5.3, the depth satisfies $\text{depth}(\Phi_{2^k}^{\min}) \le k$. For general $n$, extend $\Phi_n^{\min}$ using identity networks on unused inputs. For the maximum, define $\Phi_n^{\max}(x_1, \ldots, x_n) := -\Phi_n^{\min}(-x_1, \ldots, -x_n)$.

*Proof (of Theorem 5.7).* By Proposition 5.8 the neural network

$$\Phi := \Phi_m^{\max} \bullet (\Phi_{\lvert s_j \rvert}^{\min})_{j=1}^m \bullet ((\boldsymbol{w}_i^\top \boldsymbol{x} + b_i)_{i \in s_j})_{j=1}^m$$

realizes the function $f$. Since the number of possibilities to choose subsets of $\lbrace 1, \ldots, n \rbrace$ equals $2^n$ we have $m \le 2^n$. Since each $s_j$ is a subset of $\lbrace 1, \ldots, n \rbrace$, the cardinality $\lvert s_j \rvert$ of $s_j$ is bounded by $n$. By Lemma 5.2, Lemma 5.3, and Lemma 5.11:

$$\text{depth}(\Phi) \le 1 + \lceil \log_2(2^n) \rceil + \lceil \log_2(n) \rceil = O(n),$$

$$\text{width}(\Phi) \le 2\max\left\lbrace 3m, 3mn, mdn \right\rbrace = O(dn2^n),$$

$$\text{size}(\Phi) \le 4\left(16m + 2\sum_{j=1}^m (16\lvert s_j \rvert + 2\lceil \log_2(n) \rceil) + nm(d+1)\right) = O(dn2^n).$$

### 5.3 Simplicial Pieces

This section studies the case where we do not have arbitrary cpwl functions, but where the regions on which $f$ is affine are simplices. Under this condition, we can construct neural networks that scale merely *linearly* in the number of such regions, which is a serious improvement from the *exponential* dependence of the size on the number of regions that was found in Theorem 5.7.

#### 5.3.1 Triangulations of $\Omega$

For the ensuing discussion, we will consider $\Omega \subseteq \mathbb{R}^d$ to be partitioned into simplices. This partitioning will be termed a **triangulation** of $\Omega$. For a set $S \subseteq \mathbb{R}^d$ we denote the **convex hull** of $S$ by

$$\text{co}(S) := \left\lbrace \sum_{j=1}^n \alpha_j \boldsymbol{x}_j \;\middle\vert\; n \in \mathbb{N},\; \boldsymbol{x}_j \in S, \alpha_j \ge 0,\; \sum_{j=1}^n \alpha_j = 1 \right\rbrace.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.12</span><span class="math-callout__name">(Affine Independence and Simplex)</span></p>

Let $n \in \mathbb{N}_0$, $d \in \mathbb{N}$ and $n \le d$. We call $\boldsymbol{x}_0, \ldots, \boldsymbol{x}_n \in \mathbb{R}^d$ **affinely independent** if and only if either $n = 0$ or $n \ge 1$ and the vectors $\boldsymbol{x}_1 - \boldsymbol{x}_0, \ldots, \boldsymbol{x}_n - \boldsymbol{x}_0$ are linearly independent. In this case, we call $\text{co}(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_n) := \text{co}(\lbrace \boldsymbol{x}_0, \ldots, \boldsymbol{x}_n \rbrace)$ an **$n$-simplex**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.13</span><span class="math-callout__name">(Regular Triangulation)</span></p>

Let $d \in \mathbb{N}$, and $\Omega \subseteq \mathbb{R}^d$ be compact. Let $\mathcal{T}$ be a finite set of $d$-simplices, and for each $\tau \in \mathcal{T}$ let $V(\tau) \subseteq \Omega$ have cardinality $d + 1$ such that $\tau = \text{co}(V(\tau))$. We call $\mathcal{T}$ a **regular triangulation** of $\Omega$, if and only if

(i) $\bigcup_{\tau \in \mathcal{T}} \tau = \Omega$,

(ii) for all $\tau, \tau' \in \mathcal{T}$ it holds that $\tau \cap \tau' = \text{co}(V(\tau) \cap V(\tau'))$.

We call $\boldsymbol{\eta} \in \mathcal{V} := \bigcup_{\tau \in \mathcal{T}} V(\tau)$ a **node** (or vertex) and $\tau \in \mathcal{T}$ an **element** of the triangulation.

</div>

For a regular triangulation $\mathcal{T}$ with nodes $\mathcal{V}$ we also introduce the constant

$$k_\mathcal{T} := \max_{\boldsymbol{\eta} \in \mathcal{V}} \lvert \lbrace \tau \in \mathcal{T} \mid \boldsymbol{\eta} \in \tau \rbrace \rvert$$

corresponding to the maximal number of elements shared by a single node.

#### 5.3.2 Size Bounds for Regular Triangulations

Throughout this subsection, let $\mathcal{T}$ be a regular triangulation of $\Omega$, and we adhere to the notation of Definition 5.13. We will say that $f \colon \Omega \to \mathbb{R}$ is cpwl with respect to $\mathcal{T}$ if $f$ is cpwl and $f\vert_\tau$ is affine for each $\tau \in \mathcal{T}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.14</span><span class="math-callout__name">(Size Bounds for Regular Triangulations)</span></p>

Let $d \in \mathbb{N}$, $\Omega \subseteq \mathbb{R}^d$ be a bounded domain, and let $\mathcal{T}$ be a regular triangulation of $\Omega$. Let $f \colon \Omega \to \mathbb{R}$ be cpwl with respect to $\mathcal{T}$ and $f\vert_{\partial \Omega} = 0$. Then there exists a ReLU neural network $\Phi \colon \Omega \to \mathbb{R}$ realizing $f$, and it holds

$$\text{size}(\Phi) = O(\lvert \mathcal{T} \rvert), \quad \text{width}(\Phi) = O(\lvert \mathcal{T} \rvert), \quad \text{depth}(\Phi) = O(1),$$

where the constants in the Landau notation depend on $d$ and $k_\mathcal{T}$.

</div>

The proof strategy is to introduce a basis of the space of cpwl functions on $\mathcal{T}$ that vanish on the boundary of $\Omega$. An affine function on a simplex is uniquely determined by its values at the nodes:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.15</span></p>

Let $d \in \mathbb{N}$. Let $\tau := \text{co}(\boldsymbol{\eta}_0, \ldots, \boldsymbol{\eta}_d)$ be a $d$-simplex. For every $y_0, \ldots, y_d \in \mathbb{R}$, there exists a unique $g \in \mathcal{P}_1(\mathbb{R}^d)$ such that $g(\boldsymbol{\eta}_i) = y_i$, $i = 0, \ldots, d$.

</div>

Since $\Omega$ is the union of the simplices $\tau \in \mathcal{T}$, every cpwl function with respect to $\mathcal{T}$ is thus uniquely defined through its values at the nodes. Hence, the desired basis consists of cpwl functions $\varphi_{\boldsymbol{\eta}} \colon \Omega \to \mathbb{R}$ with respect to $\mathcal{T}$ such that

$$\varphi_{\boldsymbol{\eta}}(\boldsymbol{\mu}) = \delta_{\boldsymbol{\eta}\boldsymbol{\mu}} \quad \text{for all } \boldsymbol{\mu} \in \mathcal{V},$$

where $\delta_{\boldsymbol{\eta}\boldsymbol{\mu}}$ denotes the Kronecker delta. We can then represent every cpwl function $f \colon \Omega \to \mathbb{R}$ that vanishes on the boundary $\partial \Omega$ as

$$f(\boldsymbol{x}) = \sum_{\boldsymbol{\eta} \in \mathcal{V} \cap \mathring{\Omega}} f(\boldsymbol{\eta}) \varphi_{\boldsymbol{\eta}}(\boldsymbol{x}) \quad \text{for all } \boldsymbol{x} \in \Omega.$$

For each $\boldsymbol{\eta} \in \mathcal{V}$, the **patch** $\omega(\boldsymbol{\eta})$ of the node $\boldsymbol{\eta}$ is defined as the union of all elements containing $\boldsymbol{\eta}$:

$$\omega(\boldsymbol{\eta}) := \bigcup_{\lbrace \tau \in \mathcal{T} \mid \boldsymbol{\eta} \in \tau \rbrace} \tau.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.16</span></p>

Let $\boldsymbol{\eta} \in \mathcal{V} \cap \mathring{\Omega}$ be an interior node. Then,

$$\partial \omega(\boldsymbol{\eta}) = \bigcup_{\lbrace \tau \in \mathcal{T} \mid \boldsymbol{\eta} \in \tau \rbrace} \text{co}(V(\tau) \setminus \lbrace \boldsymbol{\eta} \rbrace).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.17</span><span class="math-callout__name">(Basis Functions as ReLU Networks)</span></p>

For each interior node $\boldsymbol{\eta} \in \mathcal{V} \cap \mathring{\Omega}$ there exists a unique cpwl function $\varphi_{\boldsymbol{\eta}} \colon \Omega \to \mathbb{R}$ satisfying $\varphi_{\boldsymbol{\eta}}(\boldsymbol{\mu}) = \delta_{\boldsymbol{\eta}\boldsymbol{\mu}}$. Moreover, $\varphi_{\boldsymbol{\eta}}$ can be expressed by a ReLU neural network with size, width, and depth bounds that only depend on $d$ and $k_\mathcal{T}$.

</div>

*Proof sketch.* By Lemma 5.15, on each $\tau \in \mathcal{T}$, the affine function $\varphi_{\boldsymbol{\eta}}\vert_\tau$ is uniquely defined through the values at the nodes of $\tau$. This defines a continuous function $\varphi_{\boldsymbol{\eta}} \colon \Omega \to \mathbb{R}$. Using Lemma 5.16 and the fact that $\varphi_{\boldsymbol{\eta}}(\boldsymbol{\mu}) = 0$ whenever $\boldsymbol{\mu} \neq \boldsymbol{\eta}$, we find that $\varphi_{\boldsymbol{\eta}}$ vanishes on the boundary of the patch $\omega(\boldsymbol{\eta}) \subseteq \Omega$. Hence, it is a cpwl function with at most $n := k_\mathcal{T} + 1$ affine functions. By Theorem 5.7, $\varphi_{\boldsymbol{\eta}}$ can be expressed as a ReLU neural network with the claimed size, width and depth bounds.

*Proof (of Theorem 5.14).* With $\Phi(\boldsymbol{x}) := \sum_{\boldsymbol{\eta} \in \mathcal{V} \cap \mathring{\Omega}} f(\boldsymbol{\eta}) \varphi_{\boldsymbol{\eta}}(\boldsymbol{x})$, it holds that $\Phi$ equals $f$ on all of $\Omega$. Since each element $\tau$ is the convex hull of $d+1$ nodes $\boldsymbol{\eta} \in \mathcal{V}$, the cardinality of $\mathcal{V}$ is bounded by $(d+1)\lvert \mathcal{T} \rvert$. Thus, the summation is over $O(\lvert \mathcal{T} \rvert)$ terms. Using Lemma 5.4 and Lemma 5.17 we obtain the claimed bounds on size, width, and depth of the neural network.

#### 5.3.3 Size Bounds for Locally Convex Triangulations

Assuming local convexity of the triangulation, in this section we make the dependence of the constants in Theorem 5.14 explicit in the dimension $d$ and in the maximal number of simplices $k_\mathcal{T}$ touching a node.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.18</span><span class="math-callout__name">(Locally Convex Triangulation)</span></p>

A regular triangulation $\mathcal{T}$ is called **locally convex** if and only if $\omega(\boldsymbol{\eta})$ is convex for all interior nodes $\boldsymbol{\eta} \in \mathcal{V} \cap \mathring{\Omega}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.19</span><span class="math-callout__name">(Size Bounds for Locally Convex Triangulations)</span></p>

Let $d \in \mathbb{N}$, and let $\Omega \subseteq \mathbb{R}^d$ be a bounded domain. Let $\mathcal{T}$ be a locally convex regular triangulation of $\Omega$. Let $f \colon \Omega \to \mathbb{R}$ be cpwl with respect to $\mathcal{T}$ and $f\vert_{\partial \Omega} = 0$. Then, there exists a constant $C > 0$ (independent of $d$, $f$ and $\mathcal{T}$) and there exists a neural network $\Phi^f \colon \Omega \to \mathbb{R}$ such that $\Phi^f = f$,

$$\text{size}(\Phi^f) \le C \cdot (1 + d^2 k_\mathcal{T} \lvert \mathcal{T} \rvert),$$

$$\text{width}(\Phi^f) \le C \cdot (1 + d \log(k_\mathcal{T}) \lvert \mathcal{T} \rvert),$$

$$\text{depth}(\Phi^f) \le C \cdot (1 + \log_2(k_\mathcal{T})).$$

</div>

The key improvement relies on an explicit construction of the basis functions $\varphi_{\boldsymbol{\eta}}$ using the convexity of the patches. If $\omega(\boldsymbol{\eta})$ is convex, then it can be written as an intersection of finitely many half-spaces:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.20</span><span class="math-callout__name">(Convex Patch Characterization)</span></p>

Let $\boldsymbol{\eta}$ be an interior node. Then a patch $\omega(\boldsymbol{\eta})$ is convex if and only if

$$\omega(\boldsymbol{\eta}) = \bigcap_{\lbrace \tau \in \mathcal{T} \mid \boldsymbol{\eta} \in \tau \rbrace} H_+(\tau, \boldsymbol{\eta}),$$

where $H_+(\tau, \boldsymbol{\eta}) := \lbrace \boldsymbol{x} \in \mathbb{R}^d \mid \boldsymbol{x} \text{ is on the same side of } H_0(\tau, \boldsymbol{\eta}) \text{ as } \boldsymbol{\eta} \rbrace \cup H_0(\tau, \boldsymbol{\eta})$ and $H_0(\tau, \boldsymbol{\eta}) := \text{aff}(V(\tau) \setminus \lbrace \boldsymbol{\eta} \rbrace)$ is the affine hyperplane passing through all nodes in $V(\tau) \setminus \lbrace \boldsymbol{\eta} \rbrace$.

</div>

This allows us to explicitly construct the basis functions:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.21</span><span class="math-callout__name">(Explicit Basis Function Construction)</span></p>

Let $\boldsymbol{\eta} \in \mathcal{V} \cap \mathring{\Omega}$ be an interior node and let $\omega(\boldsymbol{\eta})$ be a convex patch. Then

$$\varphi_{\boldsymbol{\eta}}(\boldsymbol{x}) = \max\left\lbrace 0, \min_{\lbrace \tau \in \mathcal{T} \mid \boldsymbol{\eta} \in \tau \rbrace} g_{\tau, \boldsymbol{\eta}}(\boldsymbol{x}) \right\rbrace \quad \text{for all } \boldsymbol{x} \in \mathbb{R}^d,$$

where for $\tau \in \mathcal{T}$ and $\boldsymbol{\eta} \in V(\tau)$, $g_{\tau, \boldsymbol{\eta}} \in \mathcal{P}_1(\mathbb{R}^d)$ is the affine function such that $g_{\tau, \boldsymbol{\eta}}(\boldsymbol{\mu}) = 1$ if $\boldsymbol{\eta} = \boldsymbol{\mu}$ and $g_{\tau, \boldsymbol{\eta}}(\boldsymbol{\mu}) = 0$ if $\boldsymbol{\eta} \neq \boldsymbol{\mu}$ for all $\boldsymbol{\mu} \in V(\tau)$.

</div>

*Proof (of Theorem 5.19).* For every interior node $\boldsymbol{\eta} \in \mathcal{V} \cap \mathring{\Omega}$, the cpwl basis function $\varphi_{\boldsymbol{\eta}}$ can be expressed as in Lemma 5.21, i.e.,

$$\varphi_{\boldsymbol{\eta}}(\boldsymbol{x}) = \sigma \bullet \Phi_{\lvert \lbrace \tau \in \mathcal{T} \mid \boldsymbol{\eta} \in \tau \rbrace \rvert}^{\min} \bullet (g_{\tau, \boldsymbol{\eta}}(\boldsymbol{x}))_{\lbrace \tau \in \mathcal{T} \mid \boldsymbol{\eta} \in \tau \rbrace},$$

where $(g_{\tau, \boldsymbol{\eta}}(\boldsymbol{x}))_{\lbrace \tau \in \mathcal{T} \mid \boldsymbol{\eta} \in \tau \rbrace}$ denotes the parallelization with shared inputs of the functions $g_{\tau, \boldsymbol{\eta}}$ for all $\tau \in \mathcal{T}$ such that $\boldsymbol{\eta} \in \tau$. With $\lvert \lbrace \tau \in \mathcal{T} \mid \boldsymbol{\eta} \in \tau \rbrace \rvert \le k_\mathcal{T}$, we have by Lemma 5.2:

$$\text{size}(\varphi_{\boldsymbol{\eta}}) \le 4(2 + 16k_\mathcal{T} + k_\mathcal{T} d),$$

$$\text{depth}(\varphi_{\boldsymbol{\eta}}) \le 4 + \lceil \log_2(k_\mathcal{T}) \rceil, \quad \text{width}(\varphi_{\boldsymbol{\eta}}) \le \max\lbrace 1, 3k_\mathcal{T}, d \rbrace.$$

Since every interior node has at least $d$ simplices touching it, we can assume $\max\lbrace k_\mathcal{T}, d \rbrace = k_\mathcal{T}$. The neural network $\Phi(\boldsymbol{x}) := \sum_{\boldsymbol{\eta} \in \mathcal{V} \cap \mathring{\Omega}} f(\boldsymbol{\eta}) \varphi_{\boldsymbol{\eta}}(\boldsymbol{x})$ realizes $f$ on all of $\Omega$. Since $\lvert \mathcal{V} \rvert$ is bounded by $(d+1)\lvert \mathcal{T} \rvert$, an application of Lemma 5.4 yields the desired bounds.

### 5.4 Convergence Rates for Hölder Continuous Functions

Theorem 5.14 immediately implies convergence rates for certain classes of (low regularity) functions. Recall for example the space $C^{0,s}$ of **Hölder continuous** functions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.22</span><span class="math-callout__name">(Hölder Continuity)</span></p>

Let $s \in (0, 1]$ and $\Omega \subseteq \mathbb{R}^d$. Then for $f \colon \Omega \to \mathbb{R}$

$$\lVert f \rVert_{C^{0,s}(\Omega)} := \sup_{\boldsymbol{x} \in \Omega} \lvert f(\boldsymbol{x}) \rvert + \sup_{\boldsymbol{x} \neq \boldsymbol{y} \in \Omega} \frac{\lvert f(\boldsymbol{x}) - f(\boldsymbol{y}) \rvert}{\lVert \boldsymbol{x} - \boldsymbol{y} \rVert_2^s},$$

and we denote by $C^{0,s}(\Omega)$ the set of functions $f \in C^0(\Omega)$ for which $\lVert f \rVert_{C^{0,s}(\Omega)} < \infty$.

</div>

Hölder continuous functions can be approximated well by cpwl functions. This leads to the following result.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.23</span><span class="math-callout__name">(Approximation of Hölder Functions by ReLU Networks)</span></p>

Let $d \in \mathbb{N}$. There exists a constant $C = C(d)$ such that for every $f \in C^{0,s}([0,1]^d)$ and every $N$ there exists a ReLU neural network $\Phi_N^f$ with

$$\text{size}(\Phi_N^f) \le CN, \quad \text{width}(\Phi_N^f) \le CN, \quad \text{depth}(\Phi_N^f) = C$$

and

$$\sup_{\boldsymbol{x} \in [0,1]^d} \left\lvert f(\boldsymbol{x}) - \Phi_N^f(\boldsymbol{x}) \right\rvert \le C \lVert f \rVert_{C^{0,s}([0,1]^d)} N^{-s/d}.$$

</div>

*Proof sketch.* For $M \ge 2$, consider the set of nodes $\lbrace \boldsymbol{\nu}/M \mid \boldsymbol{\nu} \in \lbrace -1, \ldots, M+1 \rbrace^d \rbrace$. These nodes suggest a partition of $[-1/M, 1 + 1/M]^d$ into $(2+M)^d$ sub-hypercubes. Each such sub-hypercube can be partitioned into $d!$ simplices, such that we obtain a regular triangulation $\mathcal{T}$ with $d!(2+M)^d$ elements on $[0,1]^d$. According to Theorem 5.14 there exists a neural network $\Phi$ that is cpwl with respect to $\mathcal{T}$ and $\Phi(\boldsymbol{\nu}/M) = f(\boldsymbol{\nu}/M)$ whenever $\boldsymbol{\nu} \in \lbrace 0, \ldots, M \rbrace^d$ and $\Phi(\boldsymbol{\nu}/M) = 0$ for all other (boundary) nodes. It holds

$$\text{size}(\Phi) \le C\lvert \mathcal{T} \rvert = Cd!(2+M)^d, \quad \text{depth}(\Phi) \le C.$$

To bound the error: fix a point $\boldsymbol{x} \in [0,1]^d$. Then $\boldsymbol{x}$ belongs to one of the interior simplices $\tau$ of the triangulation. Two nodes of the simplex have distance at most $\varepsilon := \sqrt{d}/M$. Since $\Phi\vert_\tau$ is the linear interpolant of $f$ at the vertices $V(\tau)$, $\Phi(\boldsymbol{x})$ is a convex combination of the $(f(\boldsymbol{\eta}))_{\boldsymbol{\eta} \in V(\tau)}$. Thus

$$\lvert f(\boldsymbol{x}) - \Phi(\boldsymbol{x}) \rvert \le 2 \lVert f \rVert_{C^{0,s}([0,1]^d)} \varepsilon^s = 2d^{s/2} \lVert f \rVert_{C^{0,s}([0,1]^d)} M^{-s}.$$

Setting $N := M^d$ yields $\lvert f(\boldsymbol{x}) - \Phi(\boldsymbol{x}) \rvert \le 2d^{s/2} \lVert f \rVert_{C^{0,s}([0,1]^d)} N^{-s/d}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lifting Approximation Theory to ReLU Networks)</span></p>

The principle behind Theorem 5.23 can be applied in even more generality. Since we can represent every cpwl function on a regular triangulation with a neural network of size $O(N)$, where $N$ denotes the number of elements, most classical (e.g. finite element) approximation theory for cpwl functions can be lifted to generate statements about ReLU approximation. For instance, it is well-known that functions in the Sobolev space $H^2([0,1]^d)$ can be approximated by cpwl functions on a regular triangulation in terms of $L^2([0,1]^d)$ with the rate $2/d$. Similar as in the proof of Theorem 5.23, for every $f \in H^2([0,1]^d)$ and every $N$ there then exists a ReLU neural network $\Phi_N$ such that $\text{size}(\Phi_N) = O(N)$ and

$$\lVert f - \Phi_N \rVert_{L^2([0,1]^d)} \le C \lVert f \rVert_{H^2([0,1]^d)} N^{-2/d}.$$

Finally, we may consider how to approximate smoother functions such as $f \in C^k([0,1]^d)$, $k > 1$, with ReLU neural networks. As discussed in Chapter 4 for sigmoidal activation functions, larger $k$ can lead to faster convergence. However, we will see in the following chapter that the emulation of piecewise affine functions on regular triangulations will not yield improved approximation rates as $k$ increases. To leverage such smoothness with ReLU networks, in Chapter 7 we will first build networks that emulate polynomials. Surprisingly, it turns out that polynomials can be approximated very efficiently by *deep* ReLU neural networks.

</div>

---

## Chapter 6: Affine Pieces for ReLU Neural Networks

In the previous chapters, we observed some remarkable approximation results of shallow ReLU neural networks. In practice, however, deeper architectures are more common. To understand why, in this chapter we discuss some potential shortcomings of shallow ReLU networks compared to deep ReLU networks.

Traditionally, an insightful approach to study limitations of ReLU neural networks has been to analyze the number of linear regions these functions can generate.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.1</span><span class="math-callout__name">(Pieces / Linear Regions)</span></p>

Let $d \in \mathbb{N}$, $\Omega \subseteq \mathbb{R}^d$, and let $f \colon \Omega \to \mathbb{R}$ be cpwl (see Definition 5.5). We say that $f$ has $p \in \mathbb{N}$ **pieces** (or **linear regions**), if $p$ is the smallest number of connected open sets $(\Omega_i)_{i=1}^p$ such that $\bigcup_{i=1}^p \overline{\Omega}_i = \Omega$, and $f\vert_{\Omega_i}$ is an affine function for all $i = 1, \ldots, p$. We denote $\text{Pieces}(f, \Omega) := p$.

For $d = 1$ we call every point where $f$ is not differentiable a **break point** of $f$.

</div>

To get an accurate cpwl approximation of a function, the approximating function needs to have many pieces. The next theorem quantifies this statement.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.2</span><span class="math-callout__name">(Lower Bound on Approximation Error)</span></p>

Let $-\infty < a < b < \infty$ and $f \in C^3([a,b])$ so that $f$ is not affine. Then there exists a constant $C > 0$ depending only on $\int_a^b \sqrt{\lvert f''(x) \rvert} \, \mathrm{d}x$ so that

$$\lVert g - f \rVert_{L^\infty([a,b])} > C p^{-2}$$

for all cpwl $g$ with at most $p \in \mathbb{N}$ pieces.

</div>

Theorem 6.2 implies that for ReLU neural networks we need architectures allowing for many pieces, if we want to approximate non-linear functions to high accuracy. How many pieces can we create for a fixed depth and width?

### 6.1 Upper Bounds

Neural networks are based on the composition and addition of neurons. These two operations increase the possible number of pieces in a very specific way:

- **Summation:** Let $\Omega \subseteq \mathbb{R}$. The sum of two cpwl functions $f_1, f_2 \colon \Omega \to \mathbb{R}$ satisfies

$$\text{Pieces}(f_1 + f_2, \Omega) \le \text{Pieces}(f_1, \Omega) + \text{Pieces}(f_2, \Omega) - 1.$$

This holds because the sum is affine in every point where both $f_1$ and $f_2$ are affine. Therefore, the sum has at most as many break points as $f_1$ and $f_2$ combined. Moreover, the number of pieces of a univariate function equals the number of its break points plus one.

- **Composition:** Let again $\Omega \subseteq \mathbb{R}$. The composition of two functions $f_1 \colon \mathbb{R}^d \to \mathbb{R}$ and $f_2 \colon \Omega \to \mathbb{R}^d$ satisfies

$$\text{Pieces}(f_1 \circ f_2, \Omega) \le \text{Pieces}(f_1, \mathbb{R}^d) \cdot \text{Pieces}(f_2, \Omega).$$

This is because for each of the affine pieces of $f_2$ -- let us call one of those pieces $A \subseteq \mathbb{R}$ -- we have that $f_2$ is either constant or injective on $A$. If it is constant, then $f_1 \circ f_2$ is constant. If it is injective, then $\text{Pieces}(f_1 \circ f_2, A) = \text{Pieces}(f_1, f_2(A)) \le \text{Pieces}(f_1, \mathbb{R}^d)$. Since this holds for all pieces of $f_2$ we get the bound.

These considerations give the following result. The ReLU activation function corresponds to $p = 2$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.3</span><span class="math-callout__name">(Upper Bound on Number of Pieces)</span></p>

Let $L \in \mathbb{N}$. Let $\sigma$ be cpwl with $p$ pieces. Then, every neural network with architecture $(\sigma; 1, d_1, \ldots, d_L, 1)$ has at most $(p \cdot \text{width}(\Phi))^L$ pieces.

</div>

*Proof.* The proof is via induction over the depth $L$. Let $L = 1$, and let $\Phi \colon \mathbb{R} \to \mathbb{R}$ be a neural network of architecture $(\sigma; 1, d_1, 1)$. Then

$$\Phi(x) = \sum_{k=1}^{d_1} w_k^{(1)} \sigma(w_k^{(0)} x + b_k^{(0)}) + b^{(1)} \quad \text{for } x \in \mathbb{R},$$

for certain $\boldsymbol{w}^{(0)}, \boldsymbol{w}^{(1)}, \boldsymbol{b}^{(0)} \in \mathbb{R}^{d_1}$ and $b^{(1)} \in \mathbb{R}$. By the summation bound, $\text{Pieces}(\Phi) \le p \cdot \text{width}(\Phi)$.

For the induction step, assume the statement holds for $L \in \mathbb{N}$, and let $\Phi \colon \mathbb{R} \to \mathbb{R}$ be a neural network of architecture $(\sigma; 1, d_1, \ldots, d_{L+1}, 1)$. Then, we can write

$$\Phi(x) = \sum_{j=1}^{d_{L+1}} w_j \sigma(h_j(x)) + b \quad \text{for } x \in \mathbb{R},$$

for some $\boldsymbol{w} \in \mathbb{R}^{d_{L+1}}$, $b \in \mathbb{R}$, and where each $h_j$ is a neural network of architecture $(\sigma; 1, d_1, \ldots, d_L, 1)$. Using the induction hypothesis, each $\sigma \circ h_\ell$ has at most $p \cdot (p \cdot \text{width}(\Phi))^L$ affine pieces. Hence $\Phi$ has at most $\text{width}(\Phi) \cdot p \cdot (p \cdot \text{width}(\Phi))^L = (p \cdot \text{width}(\Phi))^{L+1}$ affine pieces.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Depth vs. Width)</span></p>

Theorem 6.3 shows that there are limits to how many pieces can be created with a certain architecture. It is noteworthy that the effects of the depth and the width of a neural network are vastly different. While increasing the width can polynomially increase the number of pieces, increasing the depth can result in exponential increase. This is a first indication of the prowess of depth of neural networks.

</div>

To understand the effect of this on the approximation problem, we apply the bound of Theorem 6.3 to Theorem 6.2.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.4</span><span class="math-callout__name">(Lower Bound on Approximation Error via Depth)</span></p>

Let $d_0 \in \mathbb{N}$ and $f \in C^3([0,1]^{d_0})$. Assume there exists a line segment $\mathfrak{s} \subseteq [0,1]^{d_0}$ of positive length such that $0 < c := \int_\mathfrak{s} \sqrt{\lvert f''(x) \rvert} \, \mathrm{d}x$. Then, there exists $C > 0$ solely depending on $c$, such that for all ReLU neural networks $\Phi \colon \mathbb{R}^{d_0} \to \mathbb{R}$ with $L$ hidden layers

$$\lVert f - \Phi \rVert_{L^\infty([0,1]^{d_0})} \ge C \cdot (2\text{width}(\Phi))^{-2L}.$$

</div>

Theorem 6.4 gives a lower bound on achievable approximation rates in dependence of the depth $L$. As target functions become smoother, we expect that we can achieve faster convergence rates (cp. Chapter 4). However, without increasing the depth, it seems to be impossible to leverage such additional smoothness. This observation strongly indicates that deeper architectures can be superior.

### 6.2 Tightness of Upper Bounds

We now construct a ReLU neural network that realizes the upper bound of Theorem 6.3. First let $h_1 \colon [0,1] \to \mathbb{R}$ be the hat function

$$h_1(x) := \begin{cases} 2x & \text{if } x \in [0, \tfrac{1}{2}] \\ 2 - 2x & \text{if } x \in [\tfrac{1}{2}, 1]. \end{cases}$$

This function can be expressed by a ReLU neural network of depth one and with two nodes:

$$h_1(x) = \sigma_{\text{ReLU}}(2x) - \sigma_{\text{ReLU}}(4x - 2) \quad \text{for all } x \in [0,1].$$

We recursively set

$$h_n := h_{n-1} \circ h_1 \quad \text{for all } n \ge 2,$$

i.e., $h_n = h_1 \circ \cdots \circ h_1$ is the $n$-fold composition of $h_1$. Since $h_1 \colon [0,1] \to [0,1]$, we have $h_n \colon [0,1] \to [0,1]$ and

$$h_n \in \mathcal{N}_1^1(\sigma_{\text{ReLU}}; n, 2).$$

It turns out that this function has a rather interesting behavior. It is a "sawtooth" function with $2^{n-1}$ spikes.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6.5</span><span class="math-callout__name">(Sawtooth Function)</span></p>

Let $n \in \mathbb{N}$. It holds for all $x \in [0,1]$

$$h_n(x) = \begin{cases} 2^n(x - i 2^{-n}) & \text{if } i \ge 0 \text{ is even and } x \in [i 2^{-n}, (i+1) 2^{-n}] \\ 2^n((i+1) 2^{-n} - x) & \text{if } i \ge 1 \text{ is odd and } x \in [i 2^{-n}, (i+1) 2^{-n}]. \end{cases}$$

</div>

*Proof.* The case $n = 1$ holds by definition. We proceed by induction, and assume the statement holds for $n$. Let $x \in [0, 1/2]$ and $i \ge 0$ even such that $x \in [i 2^{-(n+1)}, (i+1)2^{-(n+1)}]$. Then $2x \in [i 2^{-n}, (i+1)2^{-n}]$. Thus

$$h_n(h_1(x)) = h_n(2x) = 2^n(2x - i 2^{-n}) = 2^{n+1}(x - i 2^{-n+1}).$$

Similarly, if $x \in [0, 1/2]$ and $i \ge 1$ odd such that $x \in [i 2^{-(n+1)}, (i+1) 2^{-(n+1)}]$, then $h_1(x) = 2x \in [i 2^{-n}, (i+1) 2^{-n}]$ and

$$h_n(h_1(x)) = h_n(2x) = 2^n(2x - (i+1)2^{-n}) = 2^{n+1}(x - (i+1)2^{-n+1}).$$

The case $x \in [1/2, 1]$ follows by observing that $h_{n+1}$ is symmetric around $1/2$.

The neural network $h_n$ has size $O(n)$ and is piecewise linear on at least $2^n$ pieces. This shows that the number of pieces can indeed increase exponentially in the neural network size, also see the upper bound in Theorem 6.3.

### 6.3 Number of Pieces in Practice

We have seen in Theorem 6.3 that deep neural networks *can* have many more pieces than their shallow counterparts. This begs the question if deep neural networks tend to generate more pieces in practice. More formally: If we randomly initialize the weights of a neural network, what is the expected number of linear regions? Will this number scale exponentially with the depth?

Surprisingly, it was found that the number of pieces of randomly initialized neural networks typically does *not* depend exponentially on the depth. Both shallow and deep networks with random initialization can have essentially the same number of pieces. This means that the theoretical maximum from Theorem 6.3 is typically not achieved in practice.

We recall that pieces are generated through composition of two functions $f_1$ and $f_2$, if the values of $f_2$ cross a level that is associated to a break point of $f_1$. In the case of a simple neuron of the form

$$\boldsymbol{x} \mapsto \sigma_{\text{ReLU}}(\langle \boldsymbol{a}, h(\boldsymbol{x}) \rangle + b)$$

where $h$ is a cpwl function, $\boldsymbol{a}$ is a vector, and $b$ is a scalar, many pieces can be generated if $\langle \boldsymbol{a}, h(\boldsymbol{x}) \rangle$ crosses the $-b$ level often. If $\boldsymbol{a}$, $b$ are random variables, and we know that $h$ does not oscillate too much, then we can quantify the probability of $\langle \boldsymbol{a}, h(\boldsymbol{x}) \rangle$ crossing the $-b$ level often.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6.6</span><span class="math-callout__name">(Level Crossing Bound)</span></p>

Let $c > 0$ and let $h \colon [0, c] \to \mathbb{R}$ be a cpwl function on $[0, c]$. Let $t \in \mathbb{N}$, let $A \subseteq \mathbb{R}$ be a Lebesgue measurable set, and assume that for every $y \in A$

$$\lvert \lbrace x \in [0, c] \mid h(x) = y \rbrace \rvert \ge t.$$

Then, $c \lVert h' \rVert_{L^\infty} \ge \lVert h' \rVert_{L^1} \ge \lvert A \rvert \cdot t$, where $\lvert A \rvert$ is the Lebesgue measure of $A$. In particular, if $h$ has at most $P \in \mathbb{N}$ pieces and $\lVert h' \rVert_{L^1} < \infty$, then for all $\delta > 0$, $t \le P$

$$\mathbb{P}\left[\lvert \lbrace x \in [0, c] \mid h(x) = U \rbrace \rvert \ge t \right] \le \frac{\lVert h' \rVert_{L^1}}{\delta t},$$

$$\mathbb{P}\left[\lvert \lbrace x \in [0, c] \mid h(x) = U \rbrace \rvert > P \right] = 0,$$

where $U$ is a uniformly distributed variable on $[-\delta/2, \delta/2]$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.7</span><span class="math-callout__name">(Random-Bias Neural Network)</span></p>

Let $L \in \mathbb{N}$, $(d_0, d_1, \ldots, d_L, 1) \in \mathbb{N}^{L+2}$ and $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$ for $\ell = 0, \ldots, L$. Furthermore, let $\delta > 0$ and let the bias vectors $\boldsymbol{b}^{(\ell)} \in \mathbb{R}^{d_{\ell+1}}$, for $\ell = 0, \ldots, L$, be random variables such that each entry of each $\boldsymbol{b}^{(\ell)}$ is independently and uniformly distributed on the interval $[-\delta/2, \delta/2]$. We call the associated ReLU neural network a **random-bias neural network**.

</div>

To apply Lemma 6.6 to a single neuron with random biases, we also need some bound on the derivative of the input to the neuron.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.8</span><span class="math-callout__name">(Maximal Internal Derivative)</span></p>

Let $L \in \mathbb{N}$, $(d_0, d_1, \ldots, d_L, 1) \in \mathbb{N}^{L+2}$, and $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$ and $\boldsymbol{b}^{(\ell)} \in \mathbb{R}^{d_{\ell+1}}$ for $\ell = 0, \ldots, L$. Moreover let $\delta > 0$.

For $\ell = 1, \ldots, L+1$, $i = 1, \ldots, d_\ell$ introduce the functions

$$\eta_{\ell, i}(\boldsymbol{x}; (\boldsymbol{W}^{(j)}, \boldsymbol{b}^{(j)})_{j=0}^{\ell-1}) = (\boldsymbol{W}^{(\ell-1)} \boldsymbol{x}^{(\ell-1)})_i \quad \text{for } \boldsymbol{x} \in \mathbb{R}^{d_0},$$

where $\boldsymbol{x}^{(\ell-1)}$ is as in Definition 2.1. We call

$$\nu\left((\boldsymbol{W}^{(\ell)})_{\ell=1}^L, \delta\right) := \max\left\lbrace \left\lVert \eta_{\ell, i}'(\,\cdot\,; (\boldsymbol{W}^{(j)}, \boldsymbol{b}^{(j)})_{j=0}^{\ell-1}) \right\rVert_2 \;\middle\vert\; (\boldsymbol{b}^{(j)})_{j=0}^L \in \prod_{j=0}^L [-\delta/2, \delta/2]^{d_{j+1}}, \ell = 1, \ldots, L, i = 1, \ldots, d_\ell \right\rbrace$$

the **maximal internal derivative** of $\Phi$.

</div>

We can now formulate the main result of this section.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.9</span><span class="math-callout__name">(Expected Pieces of Random-Bias Networks)</span></p>

Let $L \in \mathbb{N}$ and let $(d_0, d_1, \ldots, d_L, 1) \in \mathbb{N}^{L+2}$. Let $\delta \in (0, 1]$. Let $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$, for $\ell = 0, \ldots, L$, be such that $\nu\left((\boldsymbol{W}^{(\ell)})_{\ell=0}^L, \delta\right) \le C_\nu$ for a $C_\nu > 0$.

For an associated random-bias neural network $\Phi$, we have that for a line segment $\mathfrak{s} \subseteq \mathbb{R}^{d_0}$ of length 1

$$\mathbb{E}[\text{Pieces}(\Phi, \mathfrak{s})] \le 1 + d_1 + \frac{C_\nu}{\delta}(1 + (L-1)\ln(2\text{width}(\Phi))) \sum_{j=2}^L d_j.$$

</div>

*Proof sketch.* Let $\boldsymbol{b}^{(\ell)} \in [-\delta/2, \delta/2]^{d_{\ell+1}}$ for $\ell = 0, \ldots, L$ be uniformly distributed random variables. Denote $\theta_\ell \colon \mathfrak{s} \to \mathbb{R}^{d_\ell}$ the function $\boldsymbol{x} \mapsto (\eta_{\ell, i}(\boldsymbol{x}; (\boldsymbol{W}^{(j)}, \boldsymbol{b}^{(j)})_{j=0}^{\ell-1}))_{i=1}^{d_\ell}$. Let $\kappa \colon \mathfrak{s} \to [0,1]$ be an isomorphism. Since each coordinate of $\theta_\ell$ is cpwl, there are points $\boldsymbol{x}_0, \ldots, \boldsymbol{x}_{q_\ell} \in \mathfrak{s}$ such that $\theta_\ell$ is affine (as a function into $\mathbb{R}^{d_\ell}$) on each interval $[\kappa(\boldsymbol{x}_j), \kappa(\boldsymbol{x}_{j+1})]$.

For $\ell = 2$, $\theta_2(\boldsymbol{x}) = \boldsymbol{W}^{(1)} \sigma_{\text{ReLU}}(\boldsymbol{W}^{(0)} \boldsymbol{x} + \boldsymbol{b}^{(0)})$. Since $\boldsymbol{W}^{(1)} \cdot + b^{(1)}$ is affine, $\theta_2$ can only be non-affine in points where $\sigma_{\text{ReLU}}(\boldsymbol{W}^{(0)} \cdot + \boldsymbol{b}^{(0)})$ is not affine. This can happen at most $d_1$ times. Hence $q_2 = d_1$.

For the inductive step, $\theta_{\ell+1}$ is affine in every point where $\theta_\ell$ is affine and $(\theta_\ell(\boldsymbol{x}) + \boldsymbol{b}^{(\ell-1)})_i \neq 0$ for all coordinates $i = 1, \ldots, d_\ell$. Thus

$$q_{\ell+1} \le q_\ell + \sum_{j=2}^\ell \sum_{i=1}^{d_j} \lvert \lbrace \boldsymbol{x} \in \mathfrak{s} \mid \eta_{j,i}(\boldsymbol{x}) = -b_i^{(j)} \rbrace \rvert.$$

By Theorem 6.3, $\text{Pieces}(\eta_{\ell, i}(\,\cdot\,; (\boldsymbol{W}^{(j)}, \boldsymbol{b}^{(j)})_{j=0}^{\ell-1}), \mathfrak{s}) \le (2\text{width}(\Phi))^{\ell-1}$. Setting $p_{k,\ell,i} := \mathbb{P}[\lvert \lbrace \boldsymbol{x} \in \mathfrak{s} \mid \eta_{\ell,i}(\boldsymbol{x}) = -b_i^{(\ell)} \rbrace \rvert \ge k]$, by Lemma 6.6: $p_{k,\ell,i} \le C_\nu / (\delta k)$ and $p_{k,\ell,i} = 0$ for $k > (2\text{width}(\Phi))^{\ell-1}$.

It holds

$$\mathbb{E}\left[\sum_{j=2}^L \sum_{i=1}^{d_j} \lvert \lbrace \boldsymbol{x} \in \mathfrak{s} \mid \eta_{j,i}(\boldsymbol{x}) = -b_i^{(j)} \rbrace \rvert \right] \le \sum_{j=2}^L \sum_{i=1}^{d_j} \sum_{k=1}^\infty p_{k,j,i}.$$

The inner sum can be bounded by

$$\sum_{k=1}^\infty p_{k,j,i} = \sum_{k=1}^{(2\text{width}(\Phi))^{L-1}} \frac{C_\nu}{\delta k} \le \frac{C_\nu}{\delta}\left(1 + \int_1^{(2\text{width}(\Phi))^{L-1}} \frac{1}{x}\,\mathrm{d}x\right) \le \frac{C_\nu}{\delta}(1 + (L-1)\ln(2\text{width}(\Phi))).$$

Since $\theta_L = \Phi_{L+1}\vert_\mathfrak{s}$, it follows that $\text{Pieces}(\Phi, \mathfrak{s}) \le q_{L+1} + 1$, which yields the result.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.10</span><span class="math-callout__name">(Observations on Theorem 6.9)</span></p>

- **Non-exponential dependence on depth:** Considering the bound, we see that the number of pieces scales in expectation essentially like $O(LN)$, where $N$ is the total number of neurons of the architecture. This shows that in expectation, the number of pieces is linear in the number of layers, as opposed to the exponential upper bound of Theorem 6.3.
- **Maximal internal derivative:** Theorem 6.9 requires the weights to be chosen such that the maximal internal derivative is bounded by a certain number. However, if they are randomly initialized in such a way that with high probability the maximal internal derivative is bounded by a small number, then similar results can be shown. In practice, weights in the $\ell$th layer are often initialized according to a centered normal distribution with standard deviation $\sqrt{2/d_\ell}$ (He initialization). Due to the anti-proportionality of the variance to the width of the layers it is achieved that the internal derivatives remain bounded with high probability, independent of the width of the neural networks. This explains the observation that randomly initialized shallow and deep networks can have similar numbers of pieces.

</div>

---

## Chapter 7: Deep ReLU Neural Networks

In the previous chapter, we observed that many layers are a necessary prerequisite for ReLU neural networks to approximate smooth functions with high rates. We now analyze whether depth is sufficient to achieve good approximation rates for smooth functions.

To approximate smooth functions efficiently, one of the main tools in Chapter 4 was to rebuild polynomial-based functions, such as higher-order B-splines. For smooth activation functions, we were able to reproduce polynomials by using the nonlinearity of the activation functions. This argument certainly cannot be repeated for the *piecewise linear* ReLU. On the other hand, deep ReLU neural networks are extremely efficient at producing the strongly oscillating sawtooth functions discussed in Lemma 6.5. The main observation in this chapter is that the sawtooth functions are intimately linked to the squaring function, which again leads to polynomials. This observation was first made by Dmitry Yarotsky in 2016.

### 7.1 The Square Function

We start with the approximation of the map $x \mapsto x^2$. The construction, first given in Yarotsky (2016), is based on the sawtooth functions $h_n$ defined in (6.2.1).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.1</span><span class="math-callout__name">(Piecewise Linear Interpolant of $x^2$)</span></p>

Let $n \in \mathbb{N}$. Then

$$s_n(x) := x - \sum_{j=1}^{n} \frac{h_j(x)}{2^{2j}}$$

is a piecewise linear function on $[0,1]$ with break points $x_{n,j} = j 2^{-n}$, $j = 0, \ldots, 2^n$. Moreover, $s_n(x_{n,k}) = x_{n,k}^2$ for all $k = 0, \ldots, 2^n$, i.e. $s_n$ is the piecewise linear interpolant of $x^2$ on $[0,1]$.

</div>

*Proof.* The statement holds for $n = 1$. We proceed by induction. Assume the statement holds for $s_n$ and let $k \in \lbrace 0, \ldots, 2^{n+1} \rbrace$. By Lemma 6.5, $h_{n+1}(x_{n+1,k}) = 0$ whenever $k$ is even. Hence for even $k \in \lbrace 0, \ldots, 2^{n+1} \rbrace$

$$s_{n+1}(x_{n+1,k}) = x_{n+1,k} - \sum_{j=1}^{n+1} \frac{h_j(x_{n+1,k})}{2^{2j}} = s_n(x_{n+1,k}) - \frac{h_{n+1}(x_{n+1,k})}{2^{2(n+1)}} = s_n(x_{n+1,k}) = x_{n+1,k}^2,$$

where we used the induction assumption $s_n(x_{n+1,k}) = x_{n+1,k}^2$ for $x_{n+1,k} = k 2^{-(n+1)} = \frac{k}{2} 2^{-n} = x_{n,k/2}$.

Now let $k \in \lbrace 1, \ldots, 2^{n+1} - 1 \rbrace$ be odd. Then by Lemma 6.5, $h_{n+1}(x_{n+1,k}) = 1$. Moreover, since $s_n$ is linear on $[x_{n,(k-1)/2}, x_{n,(k+1)/2}] = [x_{n+1,k-1}, x_{n+1,k+1}]$ and $x_{n+1,k}$ is the midpoint of this interval,

$$s_{n+1}(x_{n+1,k}) = s_n(x_{n+1,k}) - \frac{h_{n+1}(x_{n+1,k})}{2^{2(n+1)}} = \frac{1}{2}(x_{n+1,k-1}^2 + x_{n+1,k+1}^2) - \frac{1}{2^{2(n+1)}} = \frac{k^2}{2^{2(n+1)}} = x_{n+1,k}^2.$$

$\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.2</span><span class="math-callout__name">(Approximation of the Square Function)</span></p>

For $n \in \mathbb{N}$, it holds

$$\sup_{x \in [0,1]} \lvert x^2 - s_n(x) \rvert \le 2^{-2n-1}.$$

Moreover $s_n \in \mathcal{N}_1^1(\sigma_{\text{ReLU}}; n, 3)$, and $\text{size}(s_n) \le 7n$ and $\text{depth}(s_n) = n$.

</div>

*Proof.* Set $e_n(x) := x^2 - s_n(x)$. Let $x$ be in the interval $[x_{n,k}, x_{n,k+1}] = [k 2^{-n}, (k+1)2^{-n}]$ of length $2^{-n}$. Since $s_n$ is the linear interpolant of $x^2$ on this interval, we have

$$\lvert e_n'(x) \rvert = \left\lvert 2x - \frac{x_{n,k+1}^2 - x_{n,k}^2}{2^{-n}} \right\rvert = \left\lvert 2x - \frac{2k+1}{2^n} \right\rvert \le \frac{1}{2^n}.$$

Thus $e_n : [0,1] \to \mathbb{R}$ has Lipschitz constant $2^{-n}$. Since $e_n(x_{n,k}) = 0$ for all $k = 0, \ldots, 2^n$, and the length of the interval $[x_{n,k}, x_{n,k+1}]$ equals $2^{-n}$ we get

$$\sup_{x \in [0,1]} \lvert e_n(x) \rvert \le \frac{1}{2} 2^{-n} 2^{-n} = 2^{-2n-1}.$$

Finally, to see that $s_n$ can be represented by a neural network of the claimed architecture, note that for $n \ge 2$

$$s_n(x) = x - \sum_{j=1}^{n} \frac{h_j(x)}{2^{2j}} = s_{n-1}(x) - \frac{h_n(x)}{2^{2n}} = \sigma_{\text{ReLU}} \circ s_{n-1}(x) - \frac{h_1 \circ h_{n-1}(x)}{2^{2n}}.$$

Here we used that $s_{n-1}$ is the piecewise linear interpolant of $x^2$, so that $s_{n-1}(x) \ge 0$ and thus $s_{n-1}(x) = \sigma_{\text{ReLU}}(s_{n-1}(x))$ for all $x \in [0,1]$. Hence $s_n$ is of depth $n$ and width 3. $\square$

In conclusion, $s_n : [0,1] \to [0,1]$ approximates the square function uniformly on $[0,1]$ with exponentially decreasing error in the neural network size. Note that due to Theorem 6.4, this would not be possible with a shallow neural network, which can at best interpolate $x^2$ on a partition of $[0,1]$ with polynomially many (w.r.t. the neural network size) pieces.

### 7.2 Multiplication

According to Lemma 7.2, depth can help in the approximation of $x \mapsto x^2$, which, on first sight, seems like a rather specific example. However, this opens up a path towards fast approximation of functions with high regularity, e.g., $C^k([0,1]^d)$ for some $k > 1$. The crucial observation is that, via the polarization identity we can write the product of two numbers as a sum of squares

$$x \cdot y = \frac{(x+y)^2 - (x-y)^2}{4} \tag{7.2.1}$$

for all $x, y \in \mathbb{R}$. Efficient approximation of the operation of multiplication allows efficient approximation of polynomials. Those in turn are well-known to be good approximators for functions exhibiting $k \in \mathbb{N}$ derivatives. Before exploring this idea further in the next section, we first make precise the observation that neural networks can efficiently approximate the multiplication of real numbers.

We start with the multiplication of two numbers, in which case neural networks of logarithmic size in the desired accuracy are sufficient.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.3</span><span class="math-callout__name">(Approximate Multiplication by ReLU Networks)</span></p>

For every $\varepsilon > 0$ there exists a ReLU neural network $\Phi_\varepsilon^\times : [-1,1]^2 \to [-1,1]$ such that

$$\sup_{x,y \in [-1,1]} \lvert x \cdot y - \Phi_\varepsilon^\times(x,y) \rvert \le \varepsilon,$$

and it holds $\text{size}(\Phi_\varepsilon^\times) \le C \cdot (1 + \lvert \log(\varepsilon) \rvert)$ and $\text{depth}(\Phi_\varepsilon^\times) \le C \cdot (1 + \lvert \log(\varepsilon) \rvert)$ for a constant $C > 0$ independent of $\varepsilon$. Moreover, $\Phi_\varepsilon^\times(x,y) = 0$ if $x = 0$ or $y = 0$.

</div>

*Proof.* With $n = \lceil \lvert \log_4(\varepsilon) \rvert \rceil$, define the neural network

$$\Phi_\varepsilon^\times(x,y) := s_n\!\left(\frac{\sigma_{\text{ReLU}}(x+y) + \sigma_{\text{ReLU}}(-x-y)}{2}\right) - s_n\!\left(\frac{\sigma_{\text{ReLU}}(x-y) + \sigma_{\text{ReLU}}(y-x)}{2}\right). \tag{7.2.2}$$

Since $\lvert a \rvert = \sigma_{\text{ReLU}}(a) + \sigma_{\text{ReLU}}(-a)$, by (7.2.1) we have for all $x, y \in [-1,1]$

$$\lvert x \cdot y - \Phi_\varepsilon^\times(x,y) \rvert = \left\lvert \frac{(x+y)^2 - (x-y)^2}{4} - \left(s_n\!\left(\frac{\lvert x+y \rvert}{2}\right) - s_n\!\left(\frac{\lvert x-y \rvert}{2}\right)\right) \right\rvert \le \frac{4(2^{-2n-1} + 2^{-2n-1})}{4} = 4^{-n} \le \varepsilon,$$

where we used $\lvert x+y \rvert/2, \lvert x-y \rvert/2 \in [0,1]$. We have $\text{depth}(\Phi_\varepsilon^\times) = 1 + \text{depth}(s_n) = 1 + n \le 1 + \lceil \log_4(\varepsilon) \rceil$ and $\text{size}(\Phi_\varepsilon^\times) \le C + 2\text{size}(s_n) \le Cn \le C \cdot (1 - \log(\varepsilon))$ for some constant $C > 0$. $\square$

In a similar way as in Proposition 4.8 and Lemma 5.11, we can apply operations with two inputs in the form of a binary tree to extend them to an operation on arbitrary many inputs.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.4</span><span class="math-callout__name">(Approximate $n$-fold Multiplication)</span></p>

For every $n \ge 2$ and $\varepsilon > 0$ there exists a ReLU neural network $\Phi_{n,\varepsilon}^\times : [-1,1]^n \to [-1,1]$ such that

$$\sup_{x_j \in [-1,1]} \left\lvert \prod_{j=1}^{n} x_j - \Phi_{n,\varepsilon}^\times(x_1, \ldots, x_n) \right\rvert \le \varepsilon,$$

and it holds $\text{size}(\Phi_{n,\varepsilon}^\times) \le Cn \cdot (1 + \lvert \log(\varepsilon/n) \rvert)$ and $\text{depth}(\Phi_{n,\varepsilon}^\times) \le C \log(n)(1 + \lvert \log(\varepsilon/n) \rvert)$ for a constant $C > 0$ independent of $\varepsilon$ and $n$.

</div>

*Proof.* We begin with the case $n = 2^k$. For $k = 1$ let $\tilde{\Phi}_{2,\delta}^\times := \Phi_\delta^\times$. If $k \ge 2$ let

$$\tilde{\Phi}_{2^k,\delta}^\times := \Phi_\delta^\times \circ \left(\tilde{\Phi}_{2^{k-1},\delta}^\times, \tilde{\Phi}_{2^{k-1},\delta}^\times\right).$$

Using Lemma 7.3, this neural network has depth bounded by

$$\text{depth}\!\left(\tilde{\Phi}_{2^k,\delta}^\times\right) \le k \, \text{depth}(\Phi_\delta^\times) \le Ck \cdot (1 + \lvert \log(\delta) \rvert) \le C \log(n)(1 + \lvert \log(\delta) \rvert).$$

The number of occurrences of $\Phi_\delta^\times$ equals $\sum_{j=0}^{k-1} 2^j \le n$, so the size of $\tilde{\Phi}_{2^k,\delta}^\times$ can be bounded by $Cn \cdot (1 + \lvert \log(\delta) \rvert)$.

To estimate the approximation error, denote with $\boldsymbol{x} = (x_j)_{j=1}^{2^k}$

$$e_k := \sup_{x_j \in [-1,1]} \left\lvert \prod_{j \le 2^k} x_j - \tilde{\Phi}_{2^k,\delta}^\times(\boldsymbol{x}) \right\rvert.$$

Then $e_k \le \delta + 2e_{k-1} \le \cdots \le 2^k \delta = n\delta$.

The case for general $n \ge 2$ (not necessarily $n = 2^k$) is treated similarly as in Lemma 5.11, by replacing some $\Phi_\delta^\times$ neural networks with identity neural networks. Finally, setting $\delta := \varepsilon/n$ and $\Phi_{n,\varepsilon}^\times := \tilde{\Phi}_{n,\delta}^\times$ concludes the proof. $\square$

### 7.3 Polynomials and Depth Separation

As a first consequence of the above observations, we consider approximating the polynomial

$$p(x) = \sum_{j=0}^{n} c_j x^j. \tag{7.3.1}$$

One possibility to approximate $p$ is via the Horner scheme and the approximate multiplication $\Phi_\varepsilon^\times$ from Lemma 7.3, yielding

$$p(x) = c_0 + x \cdot (c_1 + x \cdot (\cdots + x \cdot c_n) \ldots) \simeq c_0 + \Phi_\varepsilon^\times(x, c_1 + \Phi_\varepsilon^\times(x, c_2 \cdots + \Phi_\varepsilon^\times(x, c_n)) \ldots).$$

This scheme requires depth $O(n)$ due to the nested multiplications. An alternative is to approximate all monomials $1, x, \ldots, x^n$ with a binary tree using approximate multiplications $\Phi_\varepsilon^\times$, and combining them in the output layer. This idea leads to a network of size $O(n \log(n))$ and depth $O(\log(n))$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.5</span><span class="math-callout__name">(Polynomial Approximation by ReLU Networks)</span></p>

There exists a constant $C > 0$, such that for any $\varepsilon \in (0,1)$ and any polynomial $p$ of degree $n \ge 2$ as in (7.3.1), there exists a neural network $\Phi_\varepsilon^p$ such that

$$\sup_{x \in [-1,1]} \lvert p(x) - \Phi_\varepsilon^p(x) \rvert \le C\varepsilon \sum_{j=0}^{n} \lvert c_j \rvert$$

and $\text{size}(\Phi_\varepsilon^p) \le Cn \log(n/\varepsilon)$ and $\text{depth}(\Phi_\varepsilon^p) \le C \log(n/\varepsilon)$.

</div>

Lemma 7.5 shows that deep ReLU networks can approximate polynomials efficiently. This leads to an interesting implication regarding the superiority of deep architectures. Recall that $f : [-1,1] \to \mathbb{R}$ is **analytic** if its Taylor series around any point $x \in [-1,1]$ converges to $f$ in a neighbourhood of $x$. For instance all polynomials, $\sin$, $\cos$, $\exp$ etc. are analytic. We now show that these functions (except linear ones) can be approximated much more efficiently with deep ReLU networks than by shallow ones: for fixed-depth networks, the number of parameters must grow faster than any polynomial compared to the required size of deep architectures.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.6</span><span class="math-callout__name">(Depth Separation for Analytic Functions)</span></p>

Let $L \in \mathbb{N}$ and let $f : [-1,1] \to \mathbb{R}$ be analytic but not linear. Then there exist constants $C$, $\beta > 0$ such that for every $\varepsilon > 0$, there exists a ReLU neural network $\Phi_{\text{deep}}$ satisfying

$$\sup_{x \in [-1,1]} \lvert f(x) - \Phi_{\text{deep}}(x) \rvert \le C \exp\!\left(-\beta \sqrt{\text{size}(\Phi_{\text{deep}})}\right) \le \varepsilon, \tag{7.3.2}$$

but for any ReLU neural network $\Phi_{\text{shallow}}$ of depth at most $L$ holds

$$\sup_{x \in [-1,1]} \lvert f(x) - \Phi_{\text{shallow}}(x) \rvert \ge C^{-1} \text{size}(\Phi_{\text{shallow}})^{-2L}. \tag{7.3.3}$$

</div>

*Proof.* The lower bound on (7.3.3) holds by Theorem 6.4. For the upper bound on the deep neural network, assume first that the convergence radius of the Taylor series of $f$ around $0$ is $r > 1$. Then for all $x \in [-1,1]$

$$f(x) = \sum_{j \in \mathbb{N}_0} c_j x^j \quad \text{where} \quad c_j = \frac{f^{(j)}(0)}{j!} \quad \text{and} \quad \lvert c_j \rvert \le C_r r^{-j},$$

for all $j \in \mathbb{N}_0$ and some $C_r > 0$. Hence $p_n(x) := \sum_{j=0}^{n} c_j x^j$ satisfies

$$\sup_{x \in [-1,1]} \lvert f(x) - p_n(x) \rvert \le \sum_{j > n} \lvert c_j \rvert \le C_r \sum_{j > n} r^{-j} \le \frac{C_r r^{-n}}{1 - r^{-1}}.$$

Let now $\Phi_\varepsilon^{p_n}$ be the network in Lemma 7.5. Then

$$\sup_{x \in [-1,1]} \lvert f(x) - \Phi_\varepsilon^{p_n}(x) \rvert \le \tilde{C} \cdot \left(\varepsilon + \frac{r^{-n}}{1 - r^{-1}}\right),$$

for some $\tilde{C} = \tilde{C}(r, C_r)$. Choosing $n = n(\varepsilon) = \lceil \log(\varepsilon) / \log(r) \rceil$, with the bounds from Lemma 7.5 we find that

$$\sup_{x \in [-1,1]} \lvert f(x) - \Phi_\varepsilon^{p_n}(x) \rvert \le 2\tilde{C}\varepsilon$$

and for another constant $\hat{C} = \hat{C}(r)$

$$\text{size}(\Phi_\varepsilon^{p_n}) \le \hat{C} \cdot (1 + \log(\varepsilon)^2) \quad \text{and} \quad \text{depth}(\Phi_\varepsilon^{p_n}) \le \hat{C} \cdot (1 + \log(\varepsilon)).$$

This implies the existence of $C$, $\beta > 0$ and $\Phi_{\text{deep}}$ as in (7.3.2). The general case, where the Taylor expansions of $f$ converge only locally, is left as Exercise 7.14. $\square$

The proposition shows that the approximation of certain (highly relevant) functions requires significantly more parameters when using shallow instead of deep architectures. Such statements are known as *depth separation* results.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.7</span><span class="math-callout__name">(Multivariate Analytic Functions)</span></p>

Proposition 7.6 shows in particular that for analytic $f : [-1,1] \to \mathbb{R}$, holds the error bound $\exp(-\beta\sqrt{N})$ in terms of the network size $N$. This can be generalized to multivariate analytic functions $f : [-1,1]^d \to \mathbb{R}$, in which case the bound reads $\exp(-\beta N^{1/(1+d)})$.

</div>

### 7.4 $C^{k,s}$ Functions

We will now discuss the implications of our observations in the previous sections for the approximation of functions in the class $C^{k,s}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.8</span><span class="math-callout__name">(Hölder Space $C^{k,s}$)</span></p>

Let $k \in \mathbb{N}_0$, $s \in [0,1]$ and $\Omega \subseteq \mathbb{R}^d$. Then for $f : \Omega \to \mathbb{R}$

$$\lVert f \rVert_{C^{k,s}(\Omega)} := \sup_{\boldsymbol{x} \in \Omega} \max_{\lbrace \boldsymbol{\alpha} \in \mathbb{N}_0^d \mid \lvert \boldsymbol{\alpha} \rvert \le k \rbrace} \lvert D^{\boldsymbol{\alpha}} f(\boldsymbol{x}) \rvert + \sup_{\boldsymbol{x} \neq \boldsymbol{y} \in \Omega} \max_{\lbrace \boldsymbol{\alpha} \in \mathbb{N}_0^d \mid \lvert \boldsymbol{\alpha} \rvert = k \rbrace} \frac{\lvert D^{\boldsymbol{\alpha}} f(\boldsymbol{x}) - D^{\boldsymbol{\alpha}} f(\boldsymbol{y}) \rvert}{\lVert \boldsymbol{x} - \boldsymbol{y} \rVert^s},$$

and we denote by $C^{k,s}(\Omega)$ the set of functions $f \in C^k(\Omega)$ for which $\lVert f \rVert_{C^{k,s}(\Omega)} < \infty$.

</div>

Note that these spaces are ordered according to

$$C^k(\Omega) \supseteq C^{k,s}(\Omega) \supseteq C^{k,t}(\Omega) \supseteq C^{k+1}(\Omega)$$

for all $0 < s \le t \le 1$.

In order to state our main result, we first recall a version of Taylor's remainder formula for $C^{k,s}(\Omega)$ functions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.9</span><span class="math-callout__name">(Taylor's Remainder for $C^{k,s}$ Functions)</span></p>

Let $d \in \mathbb{N}$, $k \in \mathbb{N}$, $s \in [0,1]$, $\Omega = [0,1]^d$ and $f \in C^{k,s}(\Omega)$. Then for all $\boldsymbol{a}$, $\boldsymbol{x} \in \Omega$

$$f(\boldsymbol{x}) = \sum_{\lbrace \boldsymbol{\alpha} \in \mathbb{N}_0^d \mid 0 \le \lvert \boldsymbol{\alpha} \rvert \le k \rbrace} \frac{D^{\boldsymbol{\alpha}} f(\boldsymbol{a})}{\boldsymbol{\alpha}!} (\boldsymbol{x} - \boldsymbol{a})^{\boldsymbol{\alpha}} + R_k(\boldsymbol{x})$$

where with $h := \max_{i \le d} \lvert a_i - x_i \rvert$ we have $\lvert R_k(\boldsymbol{x}) \rvert \le h^{k+s} \frac{d^{k+1/2}}{k!} \lVert f \rVert_{C^{k,s}(\Omega)}$.

</div>

We now come to the main statement of this section. Up to logarithmic terms, it shows the convergence rate $(k+s)/d$ for approximating functions in $C^{k,s}([0,1]^d)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.10</span><span class="math-callout__name">(Approximation of $C^{k,s}$ Functions by Deep ReLU Networks)</span></p>

Let $d \in \mathbb{N}$, $k \in \mathbb{N}_0$, $s \in [0,1]$, and $\Omega = [0,1]^d$. Then, there exists a constant $C > 0$ such that for every $f \in C^{k,s}(\Omega)$ and every $N \ge 2$ there exists a ReLU neural network $\Phi_N^f$ such that

$$\sup_{\boldsymbol{x} \in \Omega} \lvert f(\boldsymbol{x}) - \Phi_N^f(\boldsymbol{x}) \rvert \le C \lVert f \rVert_{C^{k,s}(\Omega)} N^{-\frac{k+s}{d}}, \tag{7.4.3}$$

$\text{size}(\Phi_N^f) \le CN \log(N)$ and $\text{depth}(\Phi_N^f) \le C \log(N)$.

</div>

*Proof sketch.* The idea of the proof is to use the "partition of unity method": First we construct a partition of unity $(\varphi_{\boldsymbol{\nu}})_{\boldsymbol{\nu}}$, such that for an appropriately chosen $M \in \mathbb{N}$ each $\varphi_{\boldsymbol{\nu}}$ has support on a $O(1/M)$ neighborhood of a point $\boldsymbol{\eta} \in \Omega$. On each of these neighborhoods we use the local Taylor polynomial $p_{\boldsymbol{\nu}}$ of $f$ around $\boldsymbol{\eta}$ to approximate the function. Then $\sum_{\boldsymbol{\nu}} \varphi_{\boldsymbol{\nu}} p_{\boldsymbol{\nu}}$ gives an approximation to $f$ on $\Omega$. This approximation can be emulated by a neural network of the type $\sum_{\boldsymbol{\nu}} \Phi_\varepsilon^\times(\varphi_{\boldsymbol{\nu}}, \hat{p}_{\boldsymbol{\nu}})$, where $\hat{p}_{\boldsymbol{\nu}}$ is a neural network approximation to the polynomial $p_{\boldsymbol{\nu}}$.

**Step 1.** Define $M := \lceil N^{1/d} \rceil$ and $\varepsilon := N^{-\frac{k+s}{d}}$. Consider a uniform simplicial mesh with nodes $\lbrace \boldsymbol{\nu}/M \mid \boldsymbol{\nu} \le M \rbrace$ where $\boldsymbol{\nu}/M := (\nu_1/M, \ldots, \nu_d/M)$. Let $\varphi_{\boldsymbol{\nu}}$ denote the cpwl basis function on this mesh. Then $\sum_{\boldsymbol{\nu} \le M} \varphi_{\boldsymbol{\nu}} \equiv 1$ on $\Omega$ (partition of unity), and $\text{supp}(\varphi_{\boldsymbol{\nu}}) \subseteq \lbrace \boldsymbol{x} \in \Omega \mid \lVert \boldsymbol{x} - \boldsymbol{\nu}/M \rVert_\infty \le 1/M \rbrace$. For each $\boldsymbol{\nu} \le M$ define the multivariate polynomial $p_{\boldsymbol{\nu}}(\boldsymbol{x}) := \sum_{\lvert \boldsymbol{\alpha} \rvert \le k} \frac{D^{\boldsymbol{\alpha}} f(\boldsymbol{\nu}/M)}{\boldsymbol{\alpha}!} (\boldsymbol{x} - \boldsymbol{\nu}/M)^{\boldsymbol{\alpha}}$ and its neural network approximation $\hat{p}_{\boldsymbol{\nu}}$ using $\Phi_{\lvert \boldsymbol{\alpha} \rvert, \varepsilon}^\times$. Define $\Phi_N^f := \sum_{\boldsymbol{\nu} \le M} \Phi_\varepsilon^\times(\varphi_{\boldsymbol{\nu}}, \hat{p}_{\boldsymbol{\nu}})$.

**Step 2.** Bound the approximation error. By Lemma 7.9, $\sup_{\boldsymbol{x} \in \Omega} \lvert f(\boldsymbol{x}) - \sum_{\boldsymbol{\nu} \le M} \varphi_{\boldsymbol{\nu}}(\boldsymbol{x}) p_{\boldsymbol{\nu}}(\boldsymbol{x}) \rvert \le M^{-(k+s)}$. The error from replacing $p_{\boldsymbol{\nu}}$ with $\hat{p}_{\boldsymbol{\nu}}$ and using the approximate multiplication $\Phi_\varepsilon^\times$ contributes at most $(d+2)\varepsilon$. In total, $\sup_{\boldsymbol{x} \in \Omega} \lvert f(\boldsymbol{x}) - \Phi_N^f(\boldsymbol{x}) \rvert \le M^{-(k+s)} + \varepsilon \cdot (d+2)$, which with our choices gives the error bound (7.4.3).

**Step 3.** Bound the size and depth of the neural network. By Lemma 5.17, $\text{size}(\varphi_{\boldsymbol{\nu}}) \le C \cdot (1 + k_\mathcal{T})$ and $\text{depth}(\varphi_{\boldsymbol{\nu}}) \le C \cdot (1 + \log(k_\mathcal{T}))$ where $k_\mathcal{T}$ is the maximal number of simplices attached to a node in the mesh, which is independent of $M$. Using Lemma 7.3 and Proposition 7.4, $\text{depth}(\Phi_N^f) \le C \cdot (1 + \log(N))$ and $\text{size}(\Phi_N^f) \le CN \log(N)$. $\square$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Discussion of Theorem 7.10)</span></p>

Theorem 7.10 is similar in spirit to Yarotsky (2017); the main differences are that Yarotsky considers the class $C^k([0,1]^d)$ instead of $C^{k,s}([0,1]^d)$, and uses an approximate partition of unity, while here we use the exact partition of unity constructed in Chapter 5. Up to logarithmic terms, the theorem shows the convergence rate $(k+s)/d$. As long as $k$ is large, in principle we can achieve arbitrarily large (and $d$-independent if $k \ge d$) convergence rates. In contrast to Theorem 5.23, achieving error $N^{-\frac{k+s}{d}}$ requires depth $O(\log(N))$, i.e. the neural network depth is required to increase. This can be avoided however, and networks of depth $O(k/d)$ suffice to attain these convergence rates.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.11</span><span class="math-callout__name">(Affine Transformation of the Domain)</span></p>

Let $L : \boldsymbol{x} \mapsto \boldsymbol{A}\boldsymbol{x} + \boldsymbol{b} : \mathbb{R}^d \to \mathbb{R}^d$ be a bijective affine transformation and set $\Omega := L([0,1]^d) \subseteq \mathbb{R}^d$. Then for a function $f \in C^{k,s}(\Omega)$, by Theorem 7.10 there exists a neural network $\Phi_N^f$ such that

$$\sup_{\boldsymbol{x} \in \Omega} \lvert f(\boldsymbol{x}) - \Phi_N^f(L^{-1}(\boldsymbol{x})) \rvert \le C \lVert f \circ L \rVert_{C^{k,s}([0,1]^d)} N^{-\frac{k+s}{d}}.$$

Since $\lVert f \circ L \rVert_{C^{k,s}([0,1]^d)} \le (1 + \lVert \boldsymbol{A} \rVert^{k+s}) \lVert f \rVert_{C^{k,s}(\Omega)}$, the convergence rate $N^{-\frac{k+s}{d}}$ is achieved on every set of the type $L([0,1]^d)$ for an affine map $L$, and in particular on every hypercube $\times_{j=1}^d [a_j, b_j]$.

</div>

---

## Chapter 8: High-Dimensional Approximation

In the previous chapters we established convergence rates for the approximation of a function $f : [0,1]^d \to \mathbb{R}$ by a neural network. For example, Theorem 7.10 provides the error bound $\mathcal{O}(N^{-(k+s)/d})$ in terms of the network size $N$ (up to logarithmic terms), where $k$ and $s$ describe the smoothness of $f$. Achieving an accuracy of $\varepsilon > 0$, therefore, necessitates a network size $N = O(\varepsilon^{-d/(k+s)})$ (according to this bound). Hence, the size of the network needs to increase exponentially in $d$. This exponential dependence on the dimension $d$ is referred to as the **curse of dimensionality**. For classical smoothness spaces, such exponential $d$ dependence cannot be avoided. However, functions $f$ that are of interest in practice may have additional properties, which allow for better convergence rates.

In this chapter, we discuss three scenarios under which the curse of dimensionality can be mitigated. First, we examine an assumption limiting the behavior of functions in their Fourier domain (the Barron class). Second, we consider functions with a specific compositional structure. Third, we study functions on lower-dimensional manifolds.

### 8.1 The Barron Class

In Barron (1993), a set of functions that can be approximated by neural networks without a curse of dimensionality was introduced. This set, known as the **Barron class**, is characterized by a specific type of bounded variation. To define it, for $g \in L^1(\mathbb{R}^d)$ we denote by

$$\check{g}(\boldsymbol{w}) := \int_{\mathbb{R}^d} g(\boldsymbol{x}) e^{i \boldsymbol{w}^\top \boldsymbol{x}} \, \mathrm{d}\boldsymbol{x}$$

its inverse Fourier transform. Then, for $C > 0$ the Barron class is defined as

$$\Gamma_C := \left\lbrace f \in C(\mathbb{R}^d) \;\middle\vert\; \exists g \in L^1(\mathbb{R}^d),\; \int_{\mathbb{R}^d} \lvert \boldsymbol{\xi} \rvert \lvert g(\boldsymbol{\xi}) \rvert \, \mathrm{d}\boldsymbol{\xi} \le C \text{ and } f = \check{g} \right\rbrace.$$

We say that a function $f \in \Gamma_C$ has a finite Fourier moment, even though technically the Fourier transform of $f$ may not be well-defined, since $f$ does not need to be integrable. By the Riemann-Lebesgue Lemma, the condition $f \in C(\mathbb{R}^d)$ in the definition of $\Gamma_C$ is automatically satisfied if $g \in L^1(\mathbb{R}^d)$ as in the definition exists.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.1</span><span class="math-callout__name">(Dimension-Independent Approximation of Barron Functions)</span></p>

Let $\sigma : \mathbb{R} \to \mathbb{R}$ be sigmoidal (see Definition 3.11) and let $f \in \Gamma_C$ for some $C > 0$. Denote by $B_1^d := \lbrace \boldsymbol{x} \in \mathbb{R}^d \mid \lVert \boldsymbol{x} \rVert \le 1 \rbrace$ the unit ball. Then, for every $c > 4C^2$ and every $N \in \mathbb{N}$ there exists a neural network $\Phi^f$ with architecture $(\sigma; d, N, 1)$ such that

$$\frac{1}{\lvert B_1^d \rvert} \int_{B_1^d} \left\lvert f(\boldsymbol{x}) - \Phi^f(\boldsymbol{x}) \right\rvert^2 \mathrm{d}\boldsymbol{x} \le \frac{c}{N}, \tag{8.1.1}$$

where $\lvert B_1^d \rvert$ is the Lebesgue measure of $B_1^d$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.2</span><span class="math-callout__name">(Dimension-Independence and Caveats)</span></p>

The approximation rate in (8.1.1) can be slightly improved under some assumptions on the activation function such as powers of the ReLU.

Importantly, the dimension $d$ does not enter on the right-hand side of (8.1.1), in particular the convergence rate is not directly affected by the dimension, which is in stark contrast to the results of the previous chapters. However, it should be noted, that the constant $C$ may still have some inherent $d$-dependence.

The dimension-independent approximation rate of Theorem 8.1 may seem surprising, especially when comparing to the results in Chapters 4 and 5. However, this can be explained by recognizing that the assumption of a finite Fourier moment is effectively a *dimension-dependent regularity assumption*. Indeed, the condition becomes more restrictive in higher dimensions and hence the complexity of $\Gamma_C$ does not grow with the dimension.

A sufficient condition is that all derivatives of order up to $\lfloor d/2 \rfloor + 2$ are square-integrable. In other words, if $f$ belongs to the Sobolev space $H^{\lfloor d/2 \rfloor + 2}(\mathbb{R}^d)$, then $f$ is a Barron function. Importantly, the functions must become smoother, as the dimension increases.

</div>

The proof of Theorem 8.1 is based on a peculiar property of high-dimensional convex sets, which is described by the (approximate) Carathéodory theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 8.3</span><span class="math-callout__name">(Approximate Carathéodory Theorem)</span></p>

Let $H$ be a Hilbert space, and let $G \subseteq H$ be such that for some $B > 0$ it holds that $\lVert g \rVert_H \le B$ for all $g \in G$. Let $f \in \overline{\text{co}}(G)$. Then, for every $N \in \mathbb{N}$ and every $c > B^2$ there exist $(g_i)_{i=1}^N \subseteq G$ such that

$$\left\lVert f - \frac{1}{N} \sum_{i=1}^{N} g_i \right\rVert_H^2 \le \frac{c}{N}. \tag{8.1.2}$$

</div>

*Proof.* Fix $\varepsilon > 0$ and $N \in \mathbb{N}$. Since $f \in \overline{\text{co}}(G)$, there exist coefficients $\alpha_1, \ldots, \alpha_m \in [0,1]$ summing to 1, and linearly independent elements $h_1, \ldots, h_m \in G$ such that $f^* := \sum_{j=1}^m \alpha_j h_j$ satisfies $\lVert f - f^* \rVert_H < \varepsilon$. We claim that there exist $g_1, \ldots, g_N$, each in $\lbrace h_1, \ldots, h_m \rbrace$, such that

$$\left\lVert f^* - \frac{1}{N} \sum_{j=1}^{N} g_j \right\rVert_H^2 \le \frac{B^2}{N}.$$

Let $X_i$, $i = 1, \ldots, N$, be i.i.d. $\mathbb{R}^m$-valued random variables with $\mathbb{P}[X_i = h_j] = \alpha_j$ for all $i = 1, \ldots, m$. In particular $\mathbb{E}[X_i] = \sum_{j=1}^m \alpha_j h_j = f^*$ for each $i$. Moreover,

$$\mathbb{E}\left[\left\lVert f^* - \frac{1}{N} \sum_{j=1}^{N} X_j \right\rVert_H^2\right] = \frac{1}{N} \mathbb{E}[\lVert f^* - X_1 \rVert_H^2] = \frac{1}{N} \mathbb{E}[\lVert X_1 \rVert_H^2 - \lVert f^* \rVert_H^2] \le \frac{B^2}{N}.$$

Since the expectation in (8.1.4) is bounded by $B^2/N$, there must exist at least one realization of the random variables $X_i \in \lbrace h_1, \ldots, h_m \rbrace$, denoted $g_i$, for which (8.1.3) holds. $\square$

Lemma 8.3 provides a powerful tool: If we want to approximate a function $f$ with a superposition of $N$ elements in a set $G$, then it is sufficient to show that $f$ can be represented as an arbitrary (infinite) convex combination of elements of $G$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 8.4</span><span class="math-callout__name">(Barron Functions as Convex Combinations of Heaviside Functions)</span></p>

Let $d \in \mathbb{N}$, $C > 0$ and $f \in \Gamma_C$. Then $f\vert_{B_1^d} - f(\boldsymbol{0}) \in \overline{\text{co}}(G_C)$, where the closure is taken with respect to the norm

$$\lVert g \rVert_{L^{2,\diamond}(B_1^d)} := \left(\frac{1}{\lvert B_1^d \rvert} \int_{B_1^d} \lvert g(\boldsymbol{x}) \rvert^2 \, \mathrm{d}\boldsymbol{x}\right)^{1/2},$$

and $G_C := \left\lbrace B_1^d \ni \boldsymbol{x} \mapsto \gamma \cdot \mathbb{1}_{\mathbb{R}_+}(\langle \boldsymbol{a}, \boldsymbol{x} \rangle + b) \;\middle\vert\; \boldsymbol{a} \in \mathbb{R}^d, b \in \mathbb{R}, \lvert \gamma \rvert \le 2C \right\rbrace$.

</div>

*Proof.* **Step 1.** We express $f(\boldsymbol{x})$ via an integral. Since $f \in \Gamma_C$, there exists $g \in L^1(\mathbb{R}^d)$ such that for all $\boldsymbol{x} \in \mathbb{R}^d$

$$f(\boldsymbol{x}) - f(\boldsymbol{0}) = \int_{\mathbb{R}^d} \lvert g(\boldsymbol{\xi}) \rvert \left(\cos(\langle \boldsymbol{x}, \boldsymbol{\xi} \rangle + \kappa(\boldsymbol{\xi})) - \cos(\kappa(\boldsymbol{\xi}))\right) \mathrm{d}\boldsymbol{\xi}, \tag{8.1.6}$$

where $\kappa(\boldsymbol{\xi})$ is the phase of $g(\boldsymbol{\xi})$, i.e. $g(\boldsymbol{\xi}) = \lvert g(\boldsymbol{\xi}) \rvert e^{i\kappa(\boldsymbol{\xi})}$, and the last equality follows since $f$ is real-valued. Define a measure $\mu$ on $\mathbb{R}^d$ via its Lebesgue density $\mathrm{d}\mu(\boldsymbol{\xi}) := \frac{1}{C'} \lvert \boldsymbol{\xi} \rvert \lvert g(\boldsymbol{\xi}) \rvert \, \mathrm{d}\boldsymbol{\xi}$, where $C' := \int \lvert \boldsymbol{\xi} \rvert \lvert g(\boldsymbol{\xi}) \rvert \, \mathrm{d}\boldsymbol{\xi} \le C$; this is possible since $f \in \Gamma_C$. Then (8.1.6) leads to

$$f(\boldsymbol{x}) - f(\boldsymbol{0}) = C' \int_{\mathbb{R}^d} \frac{\cos(\langle \boldsymbol{x}, \boldsymbol{\xi} \rangle + \kappa(\boldsymbol{\xi})) - \cos(\kappa(\boldsymbol{\xi}))}{\lvert \boldsymbol{\xi} \rvert} \, \mathrm{d}\mu(\boldsymbol{\xi}). \tag{8.1.7}$$

**Step 2.** We show that $\boldsymbol{x} \mapsto f(\boldsymbol{x}) - f(\boldsymbol{0})$ is in the $L^{2,\diamond}(B_1^d)$ closure of convex combinations of the functions $\boldsymbol{x} \mapsto q_{\boldsymbol{x}}(\boldsymbol{\theta})$, where $\boldsymbol{\theta} \in \mathbb{R}^d$, and

$$q_{\boldsymbol{x}}(\boldsymbol{\theta}) := C' \frac{\cos(\lvert \boldsymbol{\theta} \rvert z + \kappa(\boldsymbol{\theta})) - \cos(\kappa(\boldsymbol{\theta}))}{\lvert \boldsymbol{\theta} \rvert}, \quad z = \langle \boldsymbol{x}, \boldsymbol{\theta}/\lvert \boldsymbol{\theta} \rvert \rangle.$$

The cosine function is 1-Lipschitz. Hence for any $\boldsymbol{\xi} \in \mathbb{R}^d$ the map (8.1.8) is bounded by one. Since $\sum_{\boldsymbol{\theta} \in \frac{1}{n}\mathbb{Z}^d} \mu(I_{\boldsymbol{\theta}}) = \mu(\mathbb{R}^d) = 1$, the claim holds. Together with Step 2, this then concludes the proof.

**Step 3.** We prove that $\boldsymbol{x} \mapsto q_{\boldsymbol{x}}(\boldsymbol{\theta})$ is in the $L^{2,\diamond}(B_1^d)$ closure of convex combinations of $G_C$ for every $\boldsymbol{\theta} \in \mathbb{R}^d$. Setting $z = \langle \boldsymbol{x}, \boldsymbol{\theta}/\lvert \boldsymbol{\theta} \rvert \rangle$, the result follows if the maps $h_{\boldsymbol{\theta}} : [-1,1] \to \mathbb{R}$, $z \mapsto C' \frac{\cos(\lvert \boldsymbol{\theta} \rvert z + \kappa(\boldsymbol{\theta})) - \cos(\kappa(\boldsymbol{\theta}))}{\lvert \boldsymbol{\theta} \rvert}$ can be approximated arbitrarily well by convex combinations of functions of the form $[-1,1] \ni z \mapsto \gamma \mathbb{1}_{\mathbb{R}_+}(a'z + b')$, where $a', b' \in \mathbb{R}$ and $\lvert \gamma \rvert \le 2C$. This is achieved by constructing piecewise constant approximations $g_{T,-} + g_{T,+}$ that interpolate $h_{\boldsymbol{\theta}}$ and converge uniformly as $T \to \infty$. $\square$

*Proof (of Theorem 8.1).* Let $f \in \Gamma_C$. By Lemma 8.4, $f\vert_{B_1^d} - f(\boldsymbol{0}) \in \overline{\text{co}}(G_C)$, where the closure is understood with respect to the norm (8.1.5). It is not hard to see that for every $g \in G_C$ it holds that $\lVert g \rVert_{L^{2,\diamond}(B_1^d)} \le 2C$. Applying Lemma 8.3 with the Hilbert space $L^{2,\diamond}(B_1^d)$, we get that for every $N \in \mathbb{N}$ there exist $\lvert \gamma_i \rvert \le 2C$, $\boldsymbol{a}_i \in \mathbb{R}^d$, $b_i \in \mathbb{R}$, for $i = 1, \ldots, N$, so that

$$\frac{1}{\lvert B_1^d \rvert} \int_{B_1^d} \left\lvert f(\boldsymbol{x}) - f(\boldsymbol{0}) - \frac{1}{N} \sum_{i=1}^{N} \gamma_i \mathbb{1}_{\mathbb{R}_+}(\langle \boldsymbol{a}_i, \boldsymbol{x} \rangle + b_i) \right\rvert^2 \mathrm{d}\boldsymbol{x} \le \frac{4C^2}{N}.$$

By Exercise 3.24, it holds that $\sigma(\lambda \cdot) \to \mathbb{1}_{\mathbb{R}_+}$ for $\lambda \to \infty$ almost everywhere. Thus, for every $\delta > 0$ there exist $\tilde{\boldsymbol{a}}_i$, $\tilde{b}_i$, $i = 1, \ldots, N$, so that

$$\frac{1}{\lvert B_1^d \rvert} \int_{B_1^d} \left\lvert f(\boldsymbol{x}) - f(\boldsymbol{0}) - \frac{1}{N} \sum_{i=1}^{N} \gamma_i \sigma\!\left(\langle \tilde{\boldsymbol{a}}_i, \boldsymbol{x} \rangle + \tilde{b}_i\right) \right\rvert^2 \mathrm{d}\boldsymbol{x} \le \frac{4C^2}{N} + \delta.$$

The result follows by observing that $\frac{1}{N} \sum_{i=1}^{N} \gamma_i \sigma\!\left(\langle \tilde{\boldsymbol{a}}_i, \boldsymbol{x} \rangle + \tilde{b}_i\right) + f(\boldsymbol{0})$ is a neural network with architecture $(\sigma; d, N, 1)$. $\square$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-Compact Parameter Set)</span></p>

Another notable aspect of the approximation of Barron functions is that the absolute values of the weights other than the output weights are not bounded by a constant. To see this, we refer to (8.1.9) in the proof, where arbitrarily large $\theta$ need to be used. While $\Gamma_C$ is a compact set, the set of neural networks of the specified architecture for a fixed $N \in \mathbb{N}$ is not parameterized with a compact parameter set. In a certain sense, this is reminiscent of Proposition 3.19 and Theorem 3.20, where arbitrarily strong approximation rates were achieved by using a very complex activation function and a non-compact parameter space.

</div>

### 8.2 Functions with Compositional Structure

As a next instance of types of functions for which the curse of dimensionality can be overcome, we study functions with compositional structure. In words, this means that we study high-dimensional functions that are constructed by composing many low-dimensional functions. This point of view was proposed in Poggio et al. (2017). Note that this can be a realistic assumption in many cases, such as for sensor networks, where local information is first aggregated in smaller clusters of sensors before some information is sent to a processing unit for further evaluation.

We introduce a model for compositional functions next. Consider a directed acyclic graph $\mathcal{G}$ with $M$ vertices $\eta_1, \ldots, \eta_M$ such that

- exactly $d$ vertices, $\eta_1, \ldots, \eta_d$, have no ingoing edge,
- each vertex has at most $m \in \mathbb{N}$ ingoing edges,
- exactly one vertex, $\eta_M$, has no outgoing edge.

With each vertex $\eta_j$ for $j > d$ we associate a function $f_j : \mathbb{R}^{d_j} \to \mathbb{R}$. Here $d_j$ denotes the cardinality of the set $S_j$, which is defined as the set of indices $i$ corresponding to vertices $\eta_i$ for which we have an edge from $\eta_i$ to $\eta_j$. Without loss of generality, we assume that $m \ge d_j = \lvert S_j \rvert \ge 1$ for all $j > d$. Finally, we let

$$F_j := x_j \quad \text{for all } j \le d \tag{8.2.1a}$$

and

$$F_j := f_j((F_i)_{i \in S_j}) \quad \text{for all } j > d. \tag{8.2.1b}$$

Then $F_M(x_1, \ldots, x_d)$ is a function from $\mathbb{R}^d \to \mathbb{R}$. Assuming

$$\lVert f_j \rVert_{C^{k,s}(\mathbb{R}^{d_j})} \le 1 \quad \text{for all } j = d+1, \ldots, M, \tag{8.2.2}$$

we denote the set of all functions of the type $F_M$ by $\mathcal{F}^{k,s}(m, d, M)$.

Clearly, for $s = 0$, $\mathcal{F}^{k,0}(m, d, M) \subseteq C^k(\mathbb{R}^d)$ since the composition of functions in $C^k$ belongs again to $C^k$. A direct application of Theorem 7.10 allows to approximate $F_M \in \mathcal{F}^k(m, d, M)$ with a neural network of size $O(N \log(N))$ and error $O(N^{-k/d})$. Since each $f_j$ depends only on $m$ variables, intuitively we expect an error convergence of type $O(N^{-k/m})$ with the constant somehow depending on the number $M$ of vertices.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.5</span><span class="math-callout__name">(Approximation of Compositional Functions)</span></p>

Let $k$, $m$, $d$, $M \in \mathbb{N}$ and $s > 0$. Let $F_M \in \mathcal{F}^{k,s}(m, d, M)$. Then there exists a constant $C = C(m, k+s, M)$ such that for every $N \in \mathbb{N}$ there exists a ReLU neural network $\Phi^{F_M}$ such that

$$\text{size}(\Phi^{F_M}) \le CN \log(N), \qquad \text{depth}(\Phi^{F_M}) \le C \log(N)$$

and

$$\sup_{\boldsymbol{x} \in [0,1]^d} \lvert F_M(\boldsymbol{x}) - \Phi^{F_M}(\boldsymbol{x}) \rvert \le N^{-\frac{k+s}{m}}.$$

</div>

*Proof sketch.* We assume without loss of generality that the indices follow a topological ordering, i.e., $S_j \subseteq \lbrace 1, \ldots, j-1 \rbrace$ for all $j$.

**Step 1.** First assume that $\hat{f}_j$ are functions such that with $0 < \varepsilon \le 1$

$$\lvert f_j(\boldsymbol{x}) - \hat{f}_j(\boldsymbol{x}) \rvert \le \delta_j := \varepsilon \cdot (2m)^{-(M+1-j)} \quad \text{for all } \boldsymbol{x} \in [-2,2]^{d_j}.$$

Let $\hat{F}_j$ be defined as in (8.2.1), but with all $f_j$ replaced by $\hat{f}_j$. We check the error of the approximation $\hat{F}_M$ to $F_M$ by induction over $j$ and show that for all $\boldsymbol{x} \in [-1,1]^d$

$$\lvert F_j(\boldsymbol{x}) - \hat{F}_j(\boldsymbol{x}) \rvert \le (2m)^{-(M-j)} \varepsilon.$$

Note that due to $\lVert f_j \rVert_{C^k} \le 1$ we have $\lvert F_j(\boldsymbol{x}) \rvert \le 1$ and thus $\hat{F}_j(\boldsymbol{x}) \in [-2,2]$. For $j = 1$ it holds $F_1(x_1) = \hat{F}_1(x_1) = x_1$, and thus (8.2.4) is valid for all $x_1 \in [-1,1]$. For the induction step, for all $\boldsymbol{x} \in [-1,1]^d$ by (8.2.3) and the induction hypothesis

$$\lvert F_j(\boldsymbol{x}) - \hat{F}_j(\boldsymbol{x}) \rvert \le \sum_{i \in S_j} \lvert F_i - \hat{F}_i \rvert + \delta_j \le m \cdot (2m)^{-(M-(j-1))} \varepsilon + (2m)^{-(M+1-j)} \varepsilon = (2m)^{-(M-j)} \varepsilon.$$

Here we used that $\lvert \frac{d}{dx_r} f_j((x_i)_{i \in S_j}) \rvert \le 1$ for all $r \in S_j$ so that by the triangle inequality and the mean value theorem $\lvert f_j((x_i)_{i \in S_j}) - f_j((y_i)_{i \in S_j}) \rvert \le \sum_{r \in S_j} \lvert x_r - y_r \rvert$.

This shows that (8.2.4) holds, and thus for all $\boldsymbol{x} \in [-1,1]^d$: $\lvert F_M(\boldsymbol{x}) - \hat{F}_M(\boldsymbol{x}) \rvert \le \varepsilon$.

**Step 2.** We sketch a construction of how to write $\hat{F}_M$ from Step 1 as a neural network $\Phi^{F_M}$ of the asserted size and depth bounds. Fix $N \in \mathbb{N}$ and let $N_j := \lceil N(2m)^{\frac{m}{k+s}(M+1-j)} \rceil$. By Theorem 7.10, since $d_j \le m$, we can find a neural network $\Phi^{f_j}$ satisfying

$$\sup_{\boldsymbol{x} \in [-2,2]^{d_j}} \lvert f_j(\boldsymbol{x}) - \Phi^{f_j}(\boldsymbol{x}) \rvert \le N_j^{-\frac{k+s}{m}} \le N^{-\frac{k+s}{m}} (2m)^{-(M+1-j)}.$$

The function $\hat{F}_M$ from Step 1 then will yield error $N^{-\frac{k+s}{m}}$ by (8.2.3) and (8.2.5). We observe that $\hat{F}_M$ can be constructed inductively as a neural network $\Phi^{F_M}$ by propagating all values $\Phi^{F_1}, \ldots, \hat{\Phi}^{F_j}$ to all consecutive layers using identity neural networks and then using the outputs of $(\Phi^{F_i})_{i \in S_{j+1}}$ as input to $\Phi^{f_{j+1}}$. The depth of this neural network is bounded by $\sum_{j=1}^{M} \text{depth}(\Phi^{f_j}) = O(M \log(N))$. We have at most $\sum_{j=1}^{M} \lvert S_j \rvert \le mM$ values which need to be propagated through these $O(M \log(N))$ layers, amounting to an overhead $O(mM^2 \log(N)) = O(\log(N))$ for the identity neural networks. In all the neural network size is thus $O(N \log(N))$. $\square$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.6</span><span class="math-callout__name">(Dependence of the Constant on Graph Structure)</span></p>

From the proof we observe that the constant $C$ in Proposition 8.5 behaves like $O((2m)^{\frac{m(M+1)}{k+s}})$.

</div>

### 8.3 Functions on Manifolds

Another instance in which the curse of dimension can be mitigated, is if the input to the network belongs to $\mathbb{R}^d$, but stems from an $m$-dimensional manifold $\mathcal{M} \subseteq \mathbb{R}^d$. If we only measure the approximation error on $\mathcal{M}$, then we can again show that it is $m$ rather than $d$ that determines the rate of convergence.

To explain the idea, we assume in the following that $\mathcal{M}$ is a smooth, compact $m$-dimensional manifold in $\mathbb{R}^d$. Moreover, we suppose that there exists $\delta > 0$ and finitely many points $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_M \in \mathcal{M}$ such that the $\delta$-balls $B_{\delta/2}(\boldsymbol{x}_i) := \lbrace \boldsymbol{y} \in \mathbb{R}^d \mid \lVert \boldsymbol{y} - \boldsymbol{x} \rVert_2 < \delta/2 \rbrace$ for $j = 1, \ldots, M$ cover $\mathcal{M}$ (for every $\delta > 0$ such $\boldsymbol{x}_i$ exist since $\mathcal{M}$ is compact). Moreover, denoting by $T_{\boldsymbol{x}} \mathcal{M} \simeq \mathbb{R}^m$ the tangential space of $\mathcal{M}$ at $\boldsymbol{x}$, we assume $\delta > 0$ to be so small that the orthogonal projection

$$\pi_j : B_\delta(\boldsymbol{x}_j) \cap \mathcal{M} \to T_{\boldsymbol{x}_j} \mathcal{M}$$

is injective, the set $\pi_j(B_\delta(\boldsymbol{x}_j) \cap \mathcal{M})$ has $C^\infty$ boundary, and the inverse of $\pi_j$, i.e.

$$\pi_j^{-1} : \pi_j(B_\delta(\boldsymbol{x}_j) \cap \mathcal{M}) \to \mathcal{M}$$

is $C^\infty$ (this is possible because $\mathcal{M}$ is a smooth manifold). Note that $\pi_j$ is a linear map, whereas $\pi_j^{-1}$ is in general non-linear.

For a function $f : \mathcal{M} \to \mathbb{R}$ we can then write

$$f(\boldsymbol{x}) = f(\pi_j^{-1}(\pi_j(\boldsymbol{x}))) = f_j(\pi_j(\boldsymbol{x})) \quad \text{for all } \boldsymbol{x} \in B_\delta(\boldsymbol{x}_j) \cap \mathcal{M}$$

where $f_j := f \circ \pi_j^{-1} : \pi_j(B_\delta(\boldsymbol{x}_j) \cap \mathcal{M}) \to \mathbb{R}$. For $f : \mathcal{M} \to \mathbb{R}$, $k \in \mathbb{N}_0$, and $s \in [0,1)$ we let

$$\lVert f \rVert_{C^{k,s}(\mathcal{M})} := \sup_{j=1,\ldots,M} \lVert f_j \rVert_{C^{k,s}(\pi_j(B_\delta(\boldsymbol{x}_j) \cap \mathcal{M}))}.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.7</span><span class="math-callout__name">(Approximation on Manifolds)</span></p>

Let $d$, $k \in \mathbb{N}$, $s \ge 0$, and let $\mathcal{M}$ be a smooth, compact $m$-dimensional manifold in $\mathbb{R}^d$. Then there exists a constant $C > 0$ such that for all $f \in C^{k,s}(\mathcal{M})$ and every $N \in \mathbb{N}$ there exists a ReLU neural network $\Phi_N^f$ such that $\text{size}(\Phi_N^f) \le CN \log(N)$, $\text{depth}(\Phi_N^f) \le C \log(N)$ and

$$\sup_{\boldsymbol{x} \in \mathcal{M}} \lvert f(\boldsymbol{x}) - \Phi_N^f(\boldsymbol{x}) \rvert \le C \lVert f \rVert_{C^{k,s}(\mathcal{M})} N^{-\frac{k+s}{m}}.$$

</div>

*Proof sketch.* Since $\mathcal{M}$ is compact there exists $A > 0$ such that $\mathcal{M} \subseteq [-A, A]^d$. Similar as in the proof of Theorem 7.10, we consider a uniform mesh with nodes $\lbrace -A + 2A\frac{\boldsymbol{\nu}}{n} \mid \boldsymbol{\nu} \le n \rbrace$, and the corresponding piecewise linear basis functions forming the partition of unity $\sum_{\boldsymbol{\nu} \le n} \varphi_{\boldsymbol{\nu}} \equiv 1$ on $[-A, A]^d$. Since $\mathcal{M}$ is covered by the balls $(B_{\delta/2}(\boldsymbol{x}_j))_{j=1}^M$, fixing $n \in \mathbb{N}$ large enough, for each $\boldsymbol{\nu}$ such that $\text{supp}\,\varphi_{\boldsymbol{\nu}} \cap \mathcal{M} \neq \emptyset$ there exists $j(\boldsymbol{\nu}) \in \lbrace 1, \ldots, M \rbrace$ such that $\text{supp}\,\varphi_{\boldsymbol{\nu}} \subseteq B_\delta(\boldsymbol{x}_{j(\boldsymbol{\nu})})$ and we set $I_j := \lbrace \boldsymbol{\nu} \le n \mid j = j(\boldsymbol{\nu}) \rbrace$. Using (8.3.3) we then have for all $\boldsymbol{x} \in \mathcal{M}$

$$f(\boldsymbol{x}) = \sum_{\boldsymbol{\nu} \le n} \varphi_{\boldsymbol{\nu}}(\boldsymbol{x}) f(\boldsymbol{x}) = \sum_{j=1}^{M} \sum_{\boldsymbol{\nu} \in I_j} \varphi_{\boldsymbol{\nu}}(\boldsymbol{x}) f_j(\pi_j(\boldsymbol{x})).$$

Next, we approximate the functions $f_j$. Let $C_j$ be the smallest ($m$-dimensional) cube in $T_{\boldsymbol{x}_j} \mathcal{M} \simeq \mathbb{R}^m$ such that $\pi_j(B_\delta(\boldsymbol{x}_j) \cap \mathcal{M}) \subseteq C_j$. The function $f_j$ can be extended to a function on $C_j$ with $\lVert f \rVert_{C^{k,s}(C_j)} \le C \lVert f \rVert_{C^{k,s}(\pi_j(B_\delta(\boldsymbol{x}_j) \cap \mathcal{M}))}$. By Theorem 7.10 (also see Remark 7.11), there exists a neural network $\hat{f}_j : C_j \to \mathbb{R}$ such that

$$\sup_{\boldsymbol{x} \in C_j} \lvert f_j(\boldsymbol{x}) - \hat{f}_j(\boldsymbol{x}) \rvert \le CN^{-\frac{k+s}{m}}$$

and $\text{size}(\hat{f}_j) \le CN \log(N)$, $\text{depth}(\hat{f}_j) \le C \log(N)$.

To approximate $f$ in (8.3.4) we let with $\varepsilon := N^{-\frac{k+s}{m}}$

$$\Phi_N := \sum_{j=1}^{M} \sum_{\boldsymbol{\nu} \in I_j} \Phi_\varepsilon^\times(\varphi_{\boldsymbol{\nu}}, \hat{f}_i \circ \pi_j),$$

where we note that $\pi_j$ is linear and thus $\hat{f}_j \circ \pi_j$ can be expressed by a neural network. The approximation error satisfies

$$\lvert f(\boldsymbol{x}) - \Phi_N(\boldsymbol{x}) \rvert \le CN^{-\frac{k+s}{m}} + d\varepsilon \le CN^{-\frac{k+s}{m}},$$

where $C$ is a constant depending on $d$ and $\mathcal{M}$. Finally, $\text{size}(\Phi_N) = O(N \log(N))$ and $\text{depth}(\Phi_N) = O(\log(N))$. $\square$

---

## Chapter 9: Interpolation

The learning problem associated to minimizing the empirical risk of (1.2.3) is based on minimizing an error that results from evaluating a neural network on a *finite* set of (training) points. In contrast, all previous approximation results focused on achieving uniformly small errors across the entire domain. Finding neural networks that achieve a small training error appears to be much simpler, since, instead of $\lVert f - \Phi_n \rVert_\infty \to 0$ for a sequence of neural networks $\Phi_n$, it suffices to have $\Phi_n(\boldsymbol{x}_i) \to f(\boldsymbol{x}_i)$ for all $\boldsymbol{x}_i$ in the training set.

In this chapter, we study the extreme case: under which conditions is it possible to find a neural network that *coincides* with the target function $f$ at all training points. This is referred to as **interpolation**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.1</span><span class="math-callout__name">(Interpolation)</span></p>

Let $d, m \in \mathbb{N}$, and let $\Omega \subseteq \mathbb{R}^d$. We say that a set of functions $\mathcal{H} \subseteq \lbrace h : \Omega \to \mathbb{R} \rbrace$ **interpolates** $m$ **points in** $\Omega$, if for every $S = (\boldsymbol{x}_i, y_i)_{i=1}^m \subseteq \Omega \times \mathbb{R}$, such that $\boldsymbol{x}_i \neq \boldsymbol{x}_j$ for $i \neq j$, there exists a function $h \in \mathcal{H}$ such that $h(\boldsymbol{x}_i) = y_i$ for all $i = 1, \ldots, m$.

</div>

Knowing the interpolation properties of an architecture is valuable for two reasons:

- Consider an architecture that interpolates $m$ points and let the number of training samples be bounded by $m$. Then (1.2.3) always has a solution.
- Consider again an architecture that interpolates $m$ points and assume that the number of training samples is less than $m$. Then for every point $\tilde{\boldsymbol{x}}$ not in the training set and every $y \in \mathbb{R}$ there exists a minimizer $h$ of (1.2.3) that satisfies $h(\tilde{\boldsymbol{x}}) = y$. As a consequence, without further restrictions, such an architecture cannot generalize to unseen data.

The existence of solutions to the interpolation problem does not follow trivially from the approximation results provided in the previous chapters (even though there is a close connection). The question of how many points neural networks with a given architecture can interpolate is closely related to the so-called **VC dimension**, which we will study in Chapter 14.

### 9.1 Universal Interpolation

Under what conditions on the activation function and architecture can a set of neural networks interpolate $m \in \mathbb{N}$ points? According to Chapter 3, particularly Theorem 3.8, we know that shallow neural networks can approximate every continuous function with arbitrary accuracy, provided the neural network width is large enough. As the neural network's width and/or depth increases, the architectures become increasingly powerful, leading us to expect that at some point, they should be able to interpolate $m$ points. However, this intuition may not be correct:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.2</span></p>

Let $\mathcal{H} := \lbrace f \in C^0([0,1]) \mid f(0) \in \mathbb{Q} \rbrace$. Then $\mathcal{H}$ is dense in $C^0([0,1])$, but $\mathcal{H}$ does not even interpolate one point in $[0,1]$.

</div>

Moreover, Theorem 3.8 is an asymptotic result that only states that a given function can be approximated for sufficiently large neural network architectures, but it does not state how large the architecture needs to be.

Surprisingly, Theorem 3.8 can nonetheless be used to give a guarantee that a fixed-size architecture yields sets of neural networks that allow the interpolation of $m$ points. Due to its similarity to the universal approximation theorem and the fact that it uses the same assumptions, we call the following theorem the "Universal Interpolation Theorem". For its statement recall the definition of the set of allowed activation functions $\mathcal{M}$ in (3.1.1) and the class $\mathcal{N}_d^1(\sigma, 1, n)$ of shallow neural networks of width $n$ introduced in Definition 3.6.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.3</span><span class="math-callout__name">(Universal Interpolation Theorem)</span></p>

Let $d, n \in \mathbb{N}$ and let $\sigma \in \mathcal{M}$ not be a polynomial. Then $\mathcal{N}_d^1(\sigma, 1, n)$ interpolates $n + 1$ points in $\mathbb{R}^d$.

</div>

*Proof.* Fix $(\boldsymbol{x}_i)_{i=1}^{n+1} \subseteq \mathbb{R}^d$ arbitrary. We will show that for any $(y_i)_{i=1}^{n+1} \subseteq \mathbb{R}$ there exist weights and biases $(\boldsymbol{w}_j)_{j=1}^n \subseteq \mathbb{R}^d$, $(b_j)_{j=1}^n$, $(v_j)_{j=1}^n \subseteq \mathbb{R}$, $c \in \mathbb{R}$ such that

$$\Phi(\boldsymbol{x}_i) := \sum_{j=1}^n v_j \sigma(\boldsymbol{w}_j^\top \boldsymbol{x}_i + b_j) + c = y_i \quad \text{for all } i = 1, \ldots, n+1.$$

Since $\Phi \in \mathcal{N}_d^1(\sigma, 1, n)$ this then concludes the proof. Denote

$$\boldsymbol{A} := \begin{pmatrix} 1 & \sigma(\boldsymbol{w}_1^\top \boldsymbol{x}_1 + b_1) & \cdots & \sigma(\boldsymbol{w}_n^\top \boldsymbol{x}_1 + b_n) \\ \vdots & \vdots & \ddots & \vdots \\ 1 & \sigma(\boldsymbol{w}_1^\top \boldsymbol{x}_{n+1} + b_1) & \cdots & \sigma(\boldsymbol{w}_n^\top \boldsymbol{x}_{n+1} + b_n) \end{pmatrix} \in \mathbb{R}^{(n+1) \times (n+1)}.$$

Then $\boldsymbol{A}$ being regular implies that for each $(y_i)_{i=1}^{n+1}$ exist $c$ and $(v_j)_{j=1}^n$ such that the interpolation holds. Hence, it suffices to find $(\boldsymbol{w}_j)_{j=1}^n$ and $(b_j)_{j=1}^n$ such that $\boldsymbol{A}$ is regular.

We proceed by induction over $k = 0, \ldots, n$, to show that there exist $(\boldsymbol{w}_j)_{j=1}^k$ and $(b_j)_{j=1}^k$ such that the first $k + 1$ columns of $\boldsymbol{A}$ are linearly independent. The case $k = 0$ is trivial. Next let $0 < k < n$ and assume that the first $k$ columns of $\boldsymbol{A}$ are linearly independent. We wish to find $\boldsymbol{w}_k$, $b_k$ such that the first $k + 1$ columns are linearly independent. Suppose such $\boldsymbol{w}_k$, $b_k$ do not exist and denote by $Y_k \subseteq \mathbb{R}^{n+1}$ the space spanned by the first $k$ columns of $\boldsymbol{A}$. Then for all $\boldsymbol{w} \in \mathbb{R}^n$, $b \in \mathbb{R}$ the vector $(\sigma(\boldsymbol{w}^\top \boldsymbol{x}_i + b))_{i=1}^{n+1} \in \mathbb{R}^{n+1}$ must belong to $Y_k$. Fix $\boldsymbol{y} = (y_i)_{i=1}^{n+1} \in \mathbb{R}^{n+1} \setminus Y_k$. Then

$$\inf_{\tilde{\Phi} \in \mathcal{N}_d^1(\sigma,1)} \lVert (\tilde{\Phi}(\boldsymbol{x}_i))_{i=1}^{n+1} - \boldsymbol{y} \rVert_2^2 \ge \inf_{\tilde{\boldsymbol{y}} \in Y_k} \lVert \tilde{\boldsymbol{y}} - \boldsymbol{y} \rVert_2^2 > 0.$$

Since we can find a continuous function $f : \mathbb{R}^d \to \mathbb{R}$ such that $f(\boldsymbol{x}_i) = y_i$ for all $i = 1, \ldots, n+1$, this contradicts Theorem 3.8. $\square$

### 9.2 Optimal Interpolation and Reconstruction

Consider a bounded domain $\Omega \subseteq \mathbb{R}^d$, a function $f : \Omega \to \mathbb{R}$, distinct points $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_m \in \Omega$, and corresponding function values $y_i := f(\boldsymbol{x}_i)$. Our objective is to approximate $f$ based solely on the data pairs $(\boldsymbol{x}_i, y_i)$, $i = 1, \ldots, m$. Under certain assumptions on $f$, ReLU neural networks can express an "optimal" reconstruction which also turns out to be an interpolant of the data.

#### 9.2.1 Motivation

In the previous section, we observed that neural networks with $m - 1 \in \mathbb{N}$ hidden neurons can interpolate $m$ points for every reasonable activation function. However, not all interpolants are equally suitable for a given application. In accordance with Occam's razor, it seems reasonable to assume that $f$ does not exhibit extreme oscillations or behave erratically between interpolation points. One way to formalize this assumption is to assume that the **Lipschitz constant**

$$\text{Lip}(f) := \sup_{\boldsymbol{x} \neq \boldsymbol{y}} \frac{\lvert f(\boldsymbol{x}) - f(\boldsymbol{y}) \rvert}{\lVert \boldsymbol{x} - \boldsymbol{y} \rVert}$$

of $f$ is bounded by a fixed value $M \in \mathbb{R}$. Here $\lVert \cdot \rVert$ denotes an arbitrary fixed norm on $\mathbb{R}^d$.

For every function $f : \Omega \to \mathbb{R}$ satisfying $f(\boldsymbol{x}_i) = y_i$ for all $i = 1, \ldots, m$, we have

$$\text{Lip}(f) = \sup_{\boldsymbol{x} \neq \boldsymbol{y} \in \Omega} \frac{\lvert f(\boldsymbol{x}) - f(\boldsymbol{y}) \rvert}{\lVert \boldsymbol{x} - \boldsymbol{y} \rVert} \ge \sup_{i \neq j} \frac{\lvert y_i - y_j \rvert}{\lVert \boldsymbol{x}_i - \boldsymbol{x}_j \rVert} =: \tilde{M}.$$

Because of this, we fix $M$ as a real number greater than or equal to $\tilde{M}$ for the remainder of our analysis.

#### 9.2.2 Optimal Reconstruction for Lipschitz Continuous Functions

Given only the information that the function has Lipschitz constant at most $M$, what is the best reconstruction of $f$ based on the data? We consider the "best reconstruction" to be a function that minimizes the $L^\infty$-error in the worst case. With

$$\text{Lip}_M(\Omega) := \lbrace f : \Omega \to \mathbb{R} \mid \text{Lip}(f) \le M \rbrace,$$

denoting the set of all functions with Lipschitz constant at most $M$, we want to solve the following problem:

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem 9.4</span></p>

We wish to find an element

$$\Phi \in \operatorname{argmin}_{h : \Omega \to \mathbb{R}} \sup_{\substack{f \in \text{Lip}_M(\Omega) \\ f \text{ satisfies } (9.2.2)}} \sup_{\boldsymbol{x} \in \Omega} \lvert f(\boldsymbol{x}) - h(\boldsymbol{x}) \rvert.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.5</span><span class="math-callout__name">(Optimal Lipschitz Interpolant)</span></p>

Let $m, d \in \mathbb{N}$, $\Omega \subseteq \mathbb{R}^d$, $f : \Omega \to \mathbb{R}$, and let $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_m \in \Omega$, $y_1, \ldots, y_m \in \mathbb{R}$ satisfy (9.2.2) and (9.2.3) with $\tilde{M} \ge 0$. Further, let $M \ge \tilde{M}$.

Then, Problem 9.4 has at least one solution given by

$$\Phi(\boldsymbol{x}) := \frac{1}{2}(f_{\text{upper}}(\boldsymbol{x}) + f_{\text{lower}}(\boldsymbol{x})) \quad \text{for } \boldsymbol{x} \in \Omega,$$

where

$$f_{\text{upper}}(\boldsymbol{x}) := \min_{k=1,\ldots,m} (y_k + M \lVert \boldsymbol{x} - \boldsymbol{x}_k \rVert), \qquad f_{\text{lower}}(\boldsymbol{x}) := \max_{k=1,\ldots,m} (y_k - M \lVert \boldsymbol{x} - \boldsymbol{x}_k \rVert).$$

Moreover, $\Phi \in \text{Lip}_M(\Omega)$ and $\Phi$ interpolates the data (i.e. satisfies (9.2.2)).

</div>

*Proof.* First we claim that for all $h_1, h_2 \in \text{Lip}_M(\Omega)$ holds $\max\lbrace h_1, h_2 \rbrace \in \text{Lip}_M(\Omega)$ as well as $\min\lbrace h_1, h_2 \rbrace \in \text{Lip}_M(\Omega)$. Since $\min\lbrace h_1, h_2 \rbrace = -\max\lbrace -h_1, -h_2 \rbrace$, it suffices to show the claim for the maximum. We need to check that

$$\frac{\lvert \max\lbrace h_1(\boldsymbol{x}), h_2(\boldsymbol{x}) \rbrace - \max\lbrace h_1(\boldsymbol{y}), h_2(\boldsymbol{y}) \rbrace \rvert}{\lVert \boldsymbol{x} - \boldsymbol{y} \rVert} \le M$$

for all $\boldsymbol{x} \neq \boldsymbol{y} \in \Omega$. Fix $\boldsymbol{x} \neq \boldsymbol{y}$. Without loss of generality assume $\max\lbrace h_1(\boldsymbol{x}), h_2(\boldsymbol{x}) \rbrace \ge \max\lbrace h_1(\boldsymbol{y}), h_2(\boldsymbol{y}) \rbrace$ and $\max\lbrace h_1(\boldsymbol{x}), h_2(\boldsymbol{x}) \rbrace = h_1(\boldsymbol{x})$. If $\max\lbrace h_1(\boldsymbol{y}), h_2(\boldsymbol{y}) \rbrace = h_1(\boldsymbol{y})$ then the numerator equals $h_1(\boldsymbol{x}) - h_1(\boldsymbol{y})$ which is bounded by $M \lVert \boldsymbol{x} - \boldsymbol{y} \rVert$. If $\max\lbrace h_1(\boldsymbol{y}), h_2(\boldsymbol{y}) \rbrace = h_2(\boldsymbol{y})$, then the numerator equals $h_1(\boldsymbol{x}) - h_2(\boldsymbol{y})$ which is bounded by $h_1(\boldsymbol{x}) - h_1(\boldsymbol{y}) \le M \lVert \boldsymbol{x} - \boldsymbol{y} \rVert$. In either case the bound holds.

Clearly, $\boldsymbol{x} \mapsto y_k - M \lVert \boldsymbol{x} - \boldsymbol{x}_k \rVert \in \text{Lip}_M(\Omega)$ for each $k = 1, \ldots, m$ and thus $f_{\text{upper}}, f_{\text{lower}} \in \text{Lip}_M(\Omega)$ as well as $\Phi \in \text{Lip}_M(\Omega)$.

Next we claim that for all $f \in \text{Lip}_M(\Omega)$ satisfying (9.2.2) holds

$$f_{\text{lower}}(\boldsymbol{x}) \le f(\boldsymbol{x}) \le f_{\text{upper}}(\boldsymbol{x}) \quad \text{for all } \boldsymbol{x} \in \Omega.$$

This is true since for every $k \in \lbrace 1, \ldots, m \rbrace$ and $\boldsymbol{x} \in \Omega$

$$\lvert y_k - f(\boldsymbol{x}) \rvert = \lvert f(\boldsymbol{x}_k) - f(\boldsymbol{x}) \rvert \le M \lVert \boldsymbol{x} - \boldsymbol{x}_k \rVert$$

so that for all $\boldsymbol{x} \in \Omega$

$$f(\boldsymbol{x}) \le \min_{k=1,\ldots,m} (y_k + M \lVert \boldsymbol{x} - \boldsymbol{x}_k \rVert), \qquad f(\boldsymbol{x}) \ge \max_{k=1,\ldots,m} (y_k - M \lVert \boldsymbol{x} - \boldsymbol{x}_k \rVert).$$

Since $f_{\text{upper}}, f_{\text{lower}} \in \text{Lip}_M(\Omega)$ satisfy (9.2.2), we conclude that for every $h : \Omega \to \mathbb{R}$ holds

$$\sup_{\substack{f \in \text{Lip}_M(\Omega) \\ f \text{ satisfies } (9.2.2)}} \sup_{\boldsymbol{x} \in \Omega} \lvert f(\boldsymbol{x}) - h(\boldsymbol{x}) \rvert \ge \sup_{\boldsymbol{x} \in \Omega} \frac{\lvert f_{\text{lower}}(\boldsymbol{x}) - f_{\text{upper}}(\boldsymbol{x}) \rvert}{2}.$$

Moreover, using the sandwich bound,

$$\sup_{\substack{f \in \text{Lip}_M(\Omega) \\ f \text{ satisfies } (9.2.2)}} \sup_{\boldsymbol{x} \in \Omega} \lvert f(\boldsymbol{x}) - \Phi(\boldsymbol{x}) \rvert \le \sup_{\boldsymbol{x} \in \Omega} \frac{\lvert f_{\text{lower}}(\boldsymbol{x}) - f_{\text{upper}}(\boldsymbol{x}) \rvert}{2}.$$

These two inequalities together imply that $\Phi$ is a solution of Problem 9.4. $\square$

#### 9.2.3 Optimal ReLU Reconstructions

So far everything was valid with an arbitrary norm on $\mathbb{R}^d$. For the next theorem, we restrict ourselves to the $1$-norm $\lVert \boldsymbol{x} \rVert_1 = \sum_{j=1}^d \lvert x_j \rvert$. Using the explicit formula of Theorem 9.5, we now show the remarkable result that ReLU neural networks can exactly express an optimal reconstruction (in the sense of Problem 9.4) with a neural network whose size scales linearly in the product of the dimension $d$ and the number of data points $m$. Additionally, the proof is constructive, thus allowing in principle for an explicit construction of the neural network without the need for training.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.6</span><span class="math-callout__name">(Optimal Lipschitz Reconstruction)</span></p>

Let $m, d \in \mathbb{N}$, $\Omega \subseteq \mathbb{R}^d$, $f : \Omega \to \mathbb{R}$, and let $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_m \in \Omega$, $y_1, \ldots, y_m \in \mathbb{R}$ satisfy (9.2.2) and (9.2.3) with $\tilde{M} > 0$. Further, let $M \ge \tilde{M}$ and let $\lVert \cdot \rVert = \lVert \cdot \rVert_1$ in (9.2.3) and (9.2.4).

Then, there exists a ReLU neural network $\Phi \in \text{Lip}_M(\Omega)$ that interpolates the data (i.e. satisfies (9.2.2)) and satisfies

$$\Phi \in \operatorname{argmin}_{h : \Omega \to \mathbb{R}} \sup_{\substack{f \in \text{Lip}_M(\Omega) \\ f \text{ satisfies } (9.2.2)}} \sup_{\boldsymbol{x} \in \Omega} \lvert f(\boldsymbol{x}) - h(\boldsymbol{x}) \rvert.$$

Moreover, $\text{depth}(\Phi) = O(\log(m))$, $\text{width}(\Phi) = O(dm)$ and all weights of $\Phi$ are bounded in absolute value by $\max\lbrace M, \lVert \boldsymbol{y} \rVert_\infty \rbrace$.

</div>

*Proof.* We need to show that the function in (9.2.6) can be expressed as a ReLU neural network with the stated size bounds. First, there is a simple ReLU neural network that implements the $1$-norm. It holds for all $\boldsymbol{x} \in \mathbb{R}^d$ that

$$\lVert \boldsymbol{x} \rVert_1 = \sum_{i=1}^d (\sigma(x_i) + \sigma(-x_i)).$$

Thus, there exists a ReLU neural network $\Phi^{\lVert \cdot \rVert_1}$ such that for all $\boldsymbol{x} \in \mathbb{R}^d$

$$\text{width}(\Phi^{\lVert \cdot \rVert_1}) = 2d, \quad \text{depth}(\Phi^{\lVert \cdot \rVert_1}) = 1, \quad \Phi^{\lVert \cdot \rVert_1}(\boldsymbol{x}) = \lVert \boldsymbol{x} \rVert_1.$$

As a result, there exist ReLU neural networks $\Phi_k : \mathbb{R}^d \to \mathbb{R}$, $k = 1, \ldots, m$, such that

$$\text{width}(\Phi_k) = 2d, \quad \text{depth}(\Phi_k) = 1, \quad \Phi_k(\boldsymbol{x}) = y_k + M \lVert \boldsymbol{x} - \boldsymbol{x}_k \rVert_1$$

for all $\boldsymbol{x} \in \mathbb{R}^d$. Using the parallelization of neural networks introduced in Section 5.1.3, there exists a ReLU neural network $\Phi_{\text{all}} := (\Phi_1, \ldots, \Phi_m) : \mathbb{R}^d \to \mathbb{R}^m$ such that

$$\text{width}(\Phi_{\text{all}}) = 4md, \quad \text{depth}(\Phi_{\text{all}}) = 1$$

and $\Phi_{\text{all}}(\boldsymbol{x}) = (y_k + M \lVert \boldsymbol{x} - \boldsymbol{x}_k \rVert_1)_{k=1}^m$ for all $\boldsymbol{x} \in \mathbb{R}^d$.

Using Lemma 5.11, we can now find a ReLU neural network $\Phi_{\text{upper}}$ such that $\Phi_{\text{upper}} = f_{\text{upper}}(\boldsymbol{x})$ for all $\boldsymbol{x} \in \Omega$, $\text{width}(\Phi_{\text{upper}}) \le \max\lbrace 16m, 4md \rbrace$, and $\text{depth}(\Phi_{\text{upper}}) \le 1 + \log(m)$.

Essentially the same construction yields a ReLU neural network $\Phi_{\text{lower}}$ with the respective properties. Lemma 5.4 then completes the proof. $\square$

---

## Chapter 10: Training of Neural Networks

Up to this point, we have discussed the representation and approximation of certain function classes using neural networks. The second pillar of deep learning concerns the question of how to fit a neural network to given data, i.e., having fixed an architecture, how to find suitable weights and biases. This task amounts to minimizing a so-called **objective function** such as the empirical risk $\widehat{\mathcal{R}}_S$ in (1.2.3). Throughout this chapter we denote the objective function by

$$f : \mathbb{R}^n \to \mathbb{R},$$

and interpret it as a function of all neural network weights and biases collected in a vector in $\mathbb{R}^n$. The goal is to (approximately) determine a **minimizer**, i.e., some $\boldsymbol{w}_* \in \mathbb{R}^n$ satisfying

$$f(\boldsymbol{w}_*) \le f(\boldsymbol{w}) \quad \text{for all } \boldsymbol{w} \in \mathbb{R}^n.$$

Standard approaches primarily include variants of (stochastic) gradient descent. In Sections 10.1, 10.2, and 10.3, we explore gradient descent, stochastic gradient descent, and accelerated gradient descent, and provide convergence proofs for smooth and strongly convex objectives. Section 10.4 discusses adaptive step sizes and explains the core principles behind popular algorithms such as Adam. Finally, Section 10.5 introduces the backpropagation algorithm, which enables the efficient application of gradient-based methods to neural network training.

### 10.1 Gradient Descent

The general idea of gradient descent is to start with some $\boldsymbol{w}_0 \in \mathbb{R}^n$, and then apply sequential updates by moving in the direction of *steepest descent* of the objective function. Assume for the moment that $f \in C^2(\mathbb{R}^n)$, and denote the $k$th iterate by $\boldsymbol{w}_k$. Then

$$f(\boldsymbol{w}_k + \boldsymbol{v}) = f(\boldsymbol{w}_k) + \boldsymbol{v}^\top \nabla f(\boldsymbol{w}_k) + O(\lVert \boldsymbol{v} \rVert^2) \quad \text{for } \lVert \boldsymbol{v} \rVert^2 \to 0.$$

This shows that the change in $f$ around $\boldsymbol{w}_k$ is locally described by the gradient $\nabla f(\boldsymbol{w}_k)$. For small $\boldsymbol{v}$ the contribution of the second order term is negligible, and the direction $\boldsymbol{v}$ along which the decrease of the risk is maximized equals the negative gradient $-\nabla f(\boldsymbol{w}_k)$.

Thus, $-\nabla f(\boldsymbol{w}_k)$ is also called the direction of steepest descent. This leads to an update of the form

$$\boldsymbol{w}_{k+1} := \boldsymbol{w}_k - h_k \nabla f(\boldsymbol{w}_k),$$

where $h_k > 0$ is referred to as the **step size** or **learning rate**. We refer to this iterative algorithm as **gradient descent**.

By the Taylor expansion it holds that

$$f(\boldsymbol{w}_{k+1}) = f(\boldsymbol{w}_k) - h_k \lVert \nabla f(\boldsymbol{w}_k) \rVert^2 + O(h_k^2),$$

so that if $\nabla f(\boldsymbol{w}_k) \neq \boldsymbol{0}$, a small enough step size $h_k$ ensures that the algorithm decreases the value of the objective function. In practice, tuning the learning rate $h_k$ can be a subtle issue as it should strike a balance between two dissenting requirements:

1. $h_k$ needs to be sufficiently small so that the second-order term is not dominating, and the update decreases the objective function.
2. $h_k$ should be large enough to ensure significant decrease of the objective function, which facilitates faster convergence of the algorithm.

A learning rate that is too high might overshoot the minimum, while a rate that is too low results in slow convergence. Common strategies include constant learning rates ($h_k = h$ for all $k \in \mathbb{N}_0$), learning rate schedules such as decaying learning rates ($h_k \searrow 0$ as $k \to \infty$), and adaptive methods.

#### 10.1.1 Structural Conditions and Existence of Minimizers

We start our analysis by discussing three key notions for analyzing gradient descent algorithms. A continuously differentiable objective function $f : \mathbb{R}^n \to \mathbb{R}$ will be called

1. *smooth* if, at each $\boldsymbol{w} \in \mathbb{R}^n$, $f$ is bounded above and below by a quadratic function that touches its graph at $\boldsymbol{w}$,
2. *convex* if, at each $\boldsymbol{w} \in \mathbb{R}^n$, $f$ lies above its tangent at $\boldsymbol{w}$,
3. *strongly convex* if, at each $\boldsymbol{w} \in \mathbb{R}^n$, $f$ lies above its tangent at $\boldsymbol{w}$ plus a convex quadratic term.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.1</span><span class="math-callout__name">($L$-Smoothness)</span></p>

Let $n \in \mathbb{N}$ and $L > 0$. A function $f : \mathbb{R}^n \to \mathbb{R}$ is called **$L$-smooth** if $f \in C^1(\mathbb{R}^n)$ and

$$f(\boldsymbol{v}) \le f(\boldsymbol{w}) + \langle \nabla f(\boldsymbol{w}), \boldsymbol{v} - \boldsymbol{w} \rangle + \frac{L}{2} \lVert \boldsymbol{w} - \boldsymbol{v} \rVert^2 \quad \text{for all } \boldsymbol{w}, \boldsymbol{v} \in \mathbb{R}^n,$$

$$f(\boldsymbol{v}) \ge f(\boldsymbol{w}) + \langle \nabla f(\boldsymbol{w}), \boldsymbol{v} - \boldsymbol{w} \rangle - \frac{L}{2} \lVert \boldsymbol{w} - \boldsymbol{v} \rVert^2 \quad \text{for all } \boldsymbol{w}, \boldsymbol{v} \in \mathbb{R}^n.$$

</div>

By definition, $L$-smooth functions satisfy the geometric property (i). In the literature, $L$-smoothness is often instead defined as Lipschitz continuity of the gradient:

$$\lVert \nabla f(\boldsymbol{w}) - \nabla f(\boldsymbol{v}) \rVert \le L \lVert \boldsymbol{w} - \boldsymbol{v} \rVert \quad \text{for all } \boldsymbol{w}, \boldsymbol{v} \in \mathbb{R}^n.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 10.2</span></p>

Let $L > 0$. Then $f \in C^1(\mathbb{R}^n)$ is $L$-smooth if and only if $\lVert \nabla f(\boldsymbol{w}) - \nabla f(\boldsymbol{v}) \rVert \le L \lVert \boldsymbol{w} - \boldsymbol{v} \rVert$ for all $\boldsymbol{w}, \boldsymbol{v} \in \mathbb{R}^n$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.3</span><span class="math-callout__name">(Convexity)</span></p>

Let $n \in \mathbb{N}$. A function $f : \mathbb{R}^n \to \mathbb{R}$ is called **convex** if and only if

$$f(\lambda \boldsymbol{w} + (1 - \lambda) \boldsymbol{v}) \le \lambda f(\boldsymbol{w}) + (1 - \lambda) f(\boldsymbol{v})$$

for all $\boldsymbol{w}, \boldsymbol{v} \in \mathbb{R}^n$, $\lambda \in (0, 1)$.

</div>

In case $f$ is continuously differentiable, this is equivalent to the geometric property (ii):

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 10.4</span></p>

Let $f \in C^1(\mathbb{R}^n)$. Then $f$ is convex if and only if

$$f(\boldsymbol{v}) \ge f(\boldsymbol{w}) + \langle \nabla f(\boldsymbol{w}), \boldsymbol{v} - \boldsymbol{w} \rangle \quad \text{for all } \boldsymbol{w}, \boldsymbol{v} \in \mathbb{R}^n.$$

</div>

The concept of convexity is strengthened by so-called strong-convexity, which requires an additional positive quadratic term on the right-hand side, and thus corresponds to geometric property (iii) by definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10.5</span><span class="math-callout__name">($\mu$-Strong Convexity)</span></p>

Let $n \in \mathbb{N}$ and $\mu > 0$. A function $f : \mathbb{R}^n \to \mathbb{R}$ is called **$\mu$-strongly convex** if $f \in C^1(\mathbb{R}^n)$ and

$$f(\boldsymbol{v}) \ge f(\boldsymbol{w}) + \langle \nabla f(\boldsymbol{w}), \boldsymbol{v} - \boldsymbol{w} \rangle + \frac{\mu}{2} \lVert \boldsymbol{v} - \boldsymbol{w} \rVert^2 \quad \text{for all } \boldsymbol{w}, \boldsymbol{v} \in \mathbb{R}^n.$$

</div>

A convex function need not be bounded from below (e.g. $w \mapsto w$) and thus need not have any (global) minimizers. And even if it is bounded from below, there need not exist minimizers (e.g. $w \mapsto \exp(w)$). However, we have the following statement.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 10.6</span></p>

Let $f : \mathbb{R}^n \to \mathbb{R}$. If $f$ is

1. *convex*, then the set of minimizers of $f$ is convex and has cardinality $0$, $1$, or $\infty$,
2. *$\mu$-strongly convex*, then $f$ has exactly one minimizer.

</div>

*Proof.* Let $f$ be convex, and assume that $\boldsymbol{w}_*$ and $\boldsymbol{v}_*$ are two minimizers of $f$. Then every convex combination $\lambda \boldsymbol{w}_* + (1 - \lambda) \boldsymbol{v}_*$, $\lambda \in [0,1]$, is also a minimizer due to (10.1.7). This shows the first claim.

Now let $f$ be $\mu$-strongly convex. Then (10.1.9) implies $f$ to be lower bounded by a convex quadratic function. Hence there exists at least one minimizer $\boldsymbol{w}_*$, and $\nabla f(\boldsymbol{w}_*) = 0$. By (10.1.9) we then have $f(\boldsymbol{v}) > f(\boldsymbol{w}_*)$ for every $\boldsymbol{v} \neq \boldsymbol{w}_*$. $\square$

#### 10.1.2 Convergence of Gradient Descent

To analyze convergence, we focus on $\mu$-strongly convex and $L$-smooth objectives only. The following theorem establishes linear convergence of gradient descent.

Recall that a sequence $e_k$ is said to **converge linearly** to $0$, if and only if there exist constants $C > 0$ and $c \in [0, 1)$ such that $e_k \le C c^k$ for all $k \in \mathbb{N}_0$. The constant $c$ is also referred to as the **rate of convergence**. Note that comparing (10.1.4a) and (10.1.9) it necessarily holds $L \ge \mu$ and therefore $\kappa := L / \mu \ge 1$. This term, known as the **condition number** of $f$, determines the rate of convergence.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.7</span><span class="math-callout__name">(Convergence of Gradient Descent)</span></p>

Let $n \in \mathbb{N}$ and $L \ge \mu > 0$. Let $f : \mathbb{R}^n \to \mathbb{R}$ be $L$-smooth and $\mu$-strongly convex. Further, let $h_k = h \in (0, 1/L]$ for all $k \in \mathbb{N}_0$, let $(\boldsymbol{w}_k)_{k=0}^\infty \subseteq \mathbb{R}^n$ be defined by the gradient descent update, and let $\boldsymbol{w}_*$ be the unique minimizer of $f$.

Then, $f(\boldsymbol{w}_k) \to f(\boldsymbol{w}_*)$ and $\boldsymbol{w}_k \to \boldsymbol{w}_*$ converge linearly for $k \to \infty$. For the specific choice $h = 1/L$ it holds for all $k \in \mathbb{N}$

$$\lVert \boldsymbol{w}_k - \boldsymbol{w}_* \rVert^2 \le \left(1 - \frac{\mu}{L}\right)^k \lVert \boldsymbol{w}_0 - \boldsymbol{w}_* \rVert^2$$

$$f(\boldsymbol{w}_k) - f(\boldsymbol{w}_*) \le \frac{L}{2} \left(1 - \frac{\mu}{L}\right)^k \lVert \boldsymbol{w}_0 - \boldsymbol{w}_* \rVert^2.$$

</div>

*Proof.* It suffices to show the first bound, since the second follows directly from the first and $L$-smoothness. The case $k = 0$ is trivial, so let $k \in \mathbb{N}$.

Expanding $\boldsymbol{w}_k = \boldsymbol{w}_{k-1} - h \nabla f(\boldsymbol{w}_{k-1})$ and using $\mu$-strong convexity (10.1.9)

$$\lVert \boldsymbol{w}_k - \boldsymbol{w}_* \rVert^2 = \lVert \boldsymbol{w}_{k-1} - \boldsymbol{w}_* \rVert^2 - 2h \langle \nabla f(\boldsymbol{w}_{k-1}), \boldsymbol{w}_{k-1} - \boldsymbol{w}_* \rangle + h^2 \lVert \nabla f(\boldsymbol{w}_{k-1}) \rVert^2$$

$$\le (1 - \mu h) \lVert \boldsymbol{w}_{k-1} - \boldsymbol{w}_* \rVert^2 - 2h \cdot (f(\boldsymbol{w}_{k-1}) - f(\boldsymbol{w}_*)) + h^2 \lVert \nabla f(\boldsymbol{w}_{k-1}) \rVert^2.$$

To bound the sum of the last two terms, we use $L$-smoothness to get

$$f(\boldsymbol{w}_k) \le f(\boldsymbol{w}_{k-1}) + \left(\frac{L}{2} - \frac{1}{h}\right) h^2 \lVert \nabla f(\boldsymbol{w}_{k-1}) \rVert^2$$

so that for $h < 2/L$

$$h^2 \lVert \nabla f(\boldsymbol{w}_{k-1}) \rVert^2 \le \frac{1}{1/h - L/2} (f(\boldsymbol{w}_{k-1}) - f(\boldsymbol{w}_*)),$$

and therefore

$$-2h \cdot (f(\boldsymbol{w}_{k-1}) - f(\boldsymbol{w}_*)) + h^2 \lVert \nabla f(\boldsymbol{w}_{k-1}) \rVert^2 \le \left(-2h + \frac{1}{1/h - L/2}\right)(f(\boldsymbol{w}_{k-1}) - f(\boldsymbol{w}_*)).$$

If $2h \ge 1/(1/h - L/2)$, which is equivalent to $h \le 1/L$, then the last term is less or equal to zero. Hence for $h \le 1/L$

$$\lVert \boldsymbol{w}_k - \boldsymbol{w}_* \rVert^2 \le (1 - \mu h) \lVert \boldsymbol{w}_{k-1} - \boldsymbol{w}_* \rVert^2 \le \cdots \le (1 - \mu h)^k \lVert \boldsymbol{w}_0 - \boldsymbol{w}_* \rVert^2.$$

This concludes the proof. $\square$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.8</span><span class="math-callout__name">(Convex Objective Functions)</span></p>

Let $f : \mathbb{R}^n \to \mathbb{R}$ be a convex and $L$-smooth function with a minimizer at $\boldsymbol{w}_*$. As shown in Lemma 10.6, the minimizer need not be unique, so we cannot expect $\boldsymbol{w}_k \to \boldsymbol{w}_*$ in general. However, the objective values still converge. Specifically, if $h_k = h \in (0, 2/L)$ for all $k \in \mathbb{N}_0$ and $(\boldsymbol{w}_k)_{k=0}^\infty \subseteq \mathbb{R}^n$ is generated by gradient descent, then

$$f(\boldsymbol{w}_k) - f(\boldsymbol{w}_*) = O(k^{-1}) \quad \text{as } k \to \infty.$$

</div>

### 10.2 Stochastic Gradient Descent

We next discuss a stochastic variant of gradient descent. The idea, which originally goes back to Robbins and Monro, is to replace the gradient $\nabla f(\boldsymbol{w}_k)$ in the gradient descent update by a random variable that we denote by $\boldsymbol{G}_k$. We interpret $\boldsymbol{G}_k$ as an approximation to $\nabla f(\boldsymbol{w}_k)$. More precisely, throughout we assume that given $\boldsymbol{w}_k$, $\boldsymbol{G}_k$ is an unbiased estimator of $\nabla f(\boldsymbol{w}_k)$ conditionally independent of $\boldsymbol{w}_0, \ldots, \boldsymbol{w}_{k-1}$, so that

$$\mathbb{E}[\boldsymbol{G}_k \mid \boldsymbol{w}_k] = \mathbb{E}[\boldsymbol{G}_k \mid \boldsymbol{w}_k, \ldots, \boldsymbol{w}_0] = \nabla f(\boldsymbol{w}_k).$$

After choosing some initial value $\boldsymbol{w}_0 \in \mathbb{R}^n$, the update rule becomes

$$\boldsymbol{w}_{k+1} := \boldsymbol{w}_k - h_k \boldsymbol{G}_k,$$

where $h_k > 0$ denotes again the step size. Unlike in Section 10.1, we focus here on $k$-dependent step sizes $h_k$.

#### 10.2.1 Motivation and Decreasing Learning Rates

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.9</span><span class="math-callout__name">(Empirical Risk Minimization)</span></p>

Suppose we have a data sample $S := (\boldsymbol{x}_j, y_j)_{j=1}^m$, where $y_j \in \mathbb{R}$ is the label corresponding to the data point $\boldsymbol{x}_j \in \mathbb{R}^d$. Using the square loss, we wish to fit a neural network $\Phi(\cdot, \boldsymbol{w}) : \mathbb{R}^d \to \mathbb{R}$ depending on parameters $\boldsymbol{w} \in \mathbb{R}^n$, such that the empirical risk

$$f(\boldsymbol{w}) := \widehat{\mathcal{R}}_S(\boldsymbol{w}) = \frac{1}{m} \sum_{j=1}^m (\Phi(\boldsymbol{x}_j, \boldsymbol{w}) - y_j)^2$$

is minimized. Performing one step of gradient descent requires the computation of

$$\nabla f(\boldsymbol{w}) = \frac{2}{m} \sum_{j=1}^m (\Phi(\boldsymbol{x}_j, \boldsymbol{w}) - y_j) \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}_j, \boldsymbol{w}),$$

and thus the computation of $m$ gradients of the neural network $\Phi$. For large $m$ (in practice $m$ can be in the millions or even larger), this computation might be infeasible.

To reduce computational cost, we replace the full gradient by the stochastic gradient

$$\boldsymbol{G} := 2(\Phi(\boldsymbol{x}_j, \boldsymbol{w}) - y_j) \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}_j, \boldsymbol{w})$$

where $j \sim \text{uniform}(1, \ldots, m)$. Then $\mathbb{E}[\boldsymbol{G}] = \nabla f(\boldsymbol{w})$, meaning that $\boldsymbol{G}$ is an unbiased estimator of $\nabla f(\boldsymbol{w})$. Importantly, computing (a realization of) $\boldsymbol{G}$ merely requires a single gradient evaluation of the neural network.

More generally, one can choose **mini-batches** of size $m_b$ (where $m_b \ll m$) and let

$$\boldsymbol{G} = \frac{2}{m_b} \sum_{j \in J} (\Phi(\boldsymbol{x}_j, \boldsymbol{w}) - y_j) \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}_j, \boldsymbol{w}),$$

where $J$ is a random subset of $\lbrace 1, \ldots, m \rbrace$ of cardinality $m_b$. A larger mini-batch size reduces the variance of $\boldsymbol{G}$ (thus giving a more robust estimate of the true gradient) but increases the computational cost.

</div>

Using $L$-smoothness we have

$$\mathbb{E}[f(\boldsymbol{w}_{k+1}) \mid \boldsymbol{w}_k] - f(\boldsymbol{w}_k) \le -h_k \lVert \nabla f(\boldsymbol{w}_k) \rVert^2 + \frac{L}{2} \mathbb{E}[\lVert h_k \boldsymbol{G}_k \rVert^2 \mid \boldsymbol{w}_k].$$

For the objective function to decrease in expectation, the first term $h_k \lVert \nabla f(\boldsymbol{w}_k) \rVert^2$ should dominate the second term $\frac{L}{2} \mathbb{E}[\lVert h_k \boldsymbol{G}_k \rVert^2 \mid \boldsymbol{w}_k]$. As $\boldsymbol{w}_k$ approaches the minimum, we have $\lVert \nabla f(\boldsymbol{w}_k) \rVert \to 0$, which suggests that $\mathbb{E}[\lVert h_k \boldsymbol{G}_k \rVert^2 \mid \boldsymbol{w}_k]$ should also decrease over time.

With a *constant learning rate* $h_k = h$, the stochastic updates cause fluctuations in the optimization path. Since these fluctuations do not decrease as the algorithm approaches the minimum, the iterates will not converge. Instead they stabilize within a neighborhood of the minimum, and oscillate around it. To achieve convergence in the limit, the variance of the update vector $-h_k \boldsymbol{G}_k$ must decrease over time. This can be achieved either by reducing the variance of $\boldsymbol{G}_k$ (e.g. through larger mini-batches), or more commonly, by decreasing the step size $h_k$ as $k$ progresses.

#### 10.2.2 Convergence of Stochastic Gradient Descent

Since $\boldsymbol{w}_k$ in the SGD update is a random variable by construction, a convergence statement can only be stochastic, e.g., in expectation or with high probability. The result is stated under the assumption that bounds the second moments of the stochastic gradients $\boldsymbol{G}_k$ and ensures that they grow at most linearly with $\lVert \nabla f(\boldsymbol{w}_k) \rVert^2$. Moreover, the analysis relies on the following decreasing step sizes

$$h_k := \min\left(\frac{\mu}{L^2 \gamma}, \frac{1}{\mu} \frac{(k+1)^2 - k^2}{(k+1)^2}\right) \quad \text{for all } k \in \mathbb{N}_0.$$

Note that $h_k = O(k^{-1})$ as $k \to \infty$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.10</span><span class="math-callout__name">(Convergence of SGD)</span></p>

Let $n \in \mathbb{N}$ and $L$, $\mu$, $\gamma > 0$. Let $f : \mathbb{R}^n \to \mathbb{R}$ be $L$-smooth and $\mu$-strongly convex. Fix $\boldsymbol{w}_0 \in \mathbb{R}^n$, let $(h_k)_{k=0}^\infty$ be as in (10.2.4) and let $(\boldsymbol{G}_k)_{k=1}^\infty$, $(\boldsymbol{w}_k)_{k=1}^\infty$ be sequences of random variables as in (10.2.1) and (10.2.2). Assume that, for some fixed $\gamma > 0$, and all $k \in \mathbb{N}$

$$\mathbb{E}[\lVert \boldsymbol{G}_k \rVert^2 \mid \boldsymbol{w}_k] \le \gamma(1 + \lVert \nabla f(\boldsymbol{w}_k) \rVert^2).$$

Then there exists a constant $C = C(\gamma, \mu, L)$ such that for all $k \in \mathbb{N}$

$$\mathbb{E}[\lVert \boldsymbol{w}_k - \boldsymbol{w}_* \rVert^2] \le \frac{C}{k}, \qquad \mathbb{E}[f(\boldsymbol{w}_k)] - f(\boldsymbol{w}_*) \le \frac{C}{k}.$$

</div>

*Proof sketch.* Using the unbiasedness of $\boldsymbol{G}_k$ and $\mu$-strong convexity, one shows the recursion

$$\mathbb{E}[\lVert \boldsymbol{w}_k - \boldsymbol{w}_* \rVert^2 \mid \boldsymbol{w}_{k-1}] \le (1 - \mu h_{k-1}) \lVert \boldsymbol{w}_{k-1} - \boldsymbol{w}_* \rVert^2 + h_{k-1}^2 \gamma.$$

By unrolling this recursion and using the specific choice of $h_k$ from (10.2.4), one obtains

$$e_k \le e_0 \prod_{j=0}^{k-1} (1 - \mu h_j) + \gamma \sum_{j=0}^{k-1} h_j^2 \prod_{i=j+1}^{k-1} (1 - \mu h_i),$$

where $e_k := \mathbb{E}[\lVert \boldsymbol{w}_k - \boldsymbol{w}_* \rVert^2 \mid \boldsymbol{w}_{k-1}]$. The product terms can be bounded by $\tilde{C} j^2 / k^2$ for a constant $\tilde{C}$. This yields $e_k \le C/k$. Finally, using $L$-smoothness,

$$f(\boldsymbol{w}_k) - f(\boldsymbol{w}_*) \le \langle \nabla f(\boldsymbol{w}_*), \boldsymbol{w}_k - \boldsymbol{w}_* \rangle + \frac{L}{2} \lVert \boldsymbol{w}_k - \boldsymbol{w}_* \rVert^2 = \frac{L}{2} \lVert \boldsymbol{w}_k - \boldsymbol{w}_* \rVert^2,$$

and taking the expectation concludes the proof. $\square$

### 10.3 Acceleration

Acceleration is an important tool for the training of neural networks. The idea was first introduced by Polyak in 1964 under the name "heavy ball method". It is inspired by the dynamics of a heavy ball rolling down the valley of the loss landscape. Since then other types of acceleration have been proposed and analyzed, with Nesterov acceleration being the most prominent example.

#### 10.3.1 Heavy Ball Method

Consider the quadratic objective function in two dimensions

$$f(\boldsymbol{w}) := \frac{1}{2} \boldsymbol{w}^\top \boldsymbol{D} \boldsymbol{w} \quad \text{where} \quad \boldsymbol{D} = \begin{pmatrix} \zeta_1 & 0 \\ 0 & \zeta_2 \end{pmatrix}$$

with $\zeta_1 \ge \zeta_2 > 0$. Clearly, $f$ has a unique minimizer at $\boldsymbol{w}_* = \boldsymbol{0} \in \mathbb{R}^2$. Starting at some $\boldsymbol{w}_0 \in \mathbb{R}^2$, gradient descent with constant step size $h > 0$ computes the iterates

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - h \boldsymbol{D} \boldsymbol{w}_k = \begin{pmatrix} (1 - h\zeta_1)^{k+1} & 0 \\ 0 & (1 - h\zeta_2)^{k+1} \end{pmatrix} \boldsymbol{w}_0.$$

The method converges for arbitrary initialization $\boldsymbol{w}_0$ if and only if $\lvert 1 - h\zeta_1 \rvert < 1$ and $\lvert 1 - h\zeta_2 \rvert < 1$. The optimal step size balancing the rate of convergence in both coordinates is

$$h_* = \operatorname{argmin}_{h > 0} \max\lbrace \lvert 1 - h\zeta_1 \rvert, \lvert 1 - h\zeta_2 \rvert \rbrace = \frac{2}{\zeta_1 + \zeta_2}.$$

With $\kappa = \zeta_1 / \zeta_2$ we then obtain the convergence rate

$$\lvert 1 - h_* \zeta_1 \rvert = \lvert 1 - h_* \zeta_2 \rvert = \frac{\zeta_1 - \zeta_2}{\zeta_1 + \zeta_2} = \frac{\kappa - 1}{\kappa + 1} \in [0, 1).$$

If $\zeta_1 \gg \zeta_2$, this term is close to $1$, and thus the convergence will be slow. This is consistent with our analysis for strongly convex objective functions; the condition number of $f$ equals $\kappa = \zeta_1 / \zeta_2 \gg 1$.

The loss-landscape in this case looks like a ravine (the derivative is much larger in one direction than the other), and away from the floor, $\nabla f$ mainly points to the opposite side. Therefore the iterates oscillate back and forth in the first coordinate, and make little progress in the direction of the valley along the second coordinate axis. To address this problem, the heavy ball method introduces a "momentum" term which can mitigate this effect to some extent. The idea is to choose the update not just according to the gradient at the current location, but to add information from the previous steps. After initializing $\boldsymbol{w}_0$ and $\boldsymbol{w}_1 = \boldsymbol{w}_0 - \alpha \nabla f(\boldsymbol{w}_0)$, let for $k \in \mathbb{N}$

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \alpha \nabla f(\boldsymbol{w}_k) + \beta(\boldsymbol{w}_k - \boldsymbol{w}_{k-1}).$$

This is known as Polyak's heavy ball method. Here $\alpha > 0$ and $\beta \in (0, 1)$ are hyperparameters. Iteratively expanding with the given initialization, observe that for $k \ge 0$

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \alpha \left(\sum_{j=0}^k \beta^j \nabla f(\boldsymbol{w}_{k-j})\right).$$

Thus, $\boldsymbol{w}_k$ is updated using an *exponentially weighted moving average* of all past gradients. Choosing the momentum parameter $\beta$ in the interval $(0, 1)$ ensures that the influence of previous gradients on the update decays exponentially. The concrete value of $\beta$ determines the balance between the impact of recent and past gradients.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.12</span><span class="math-callout__name">(ODE Interpretation)</span></p>

For suitable choices of $\alpha$ and $\beta$, the heavy ball method (10.3.5) can be interpreted as a discretization of the second-order ODE

$$m \boldsymbol{w}''(t) = -\nabla f(\boldsymbol{w}(t)) - r \boldsymbol{w}'(t).$$

This equation describes the movement of a point mass $m$ under influence of the force field $-\nabla f(\boldsymbol{w}(t))$; the term $-\boldsymbol{w}'(t)$, which points in the negative direction of the current velocity, corresponds to friction, and $r > 0$ is the friction coefficient. The discretization

$$m \frac{\boldsymbol{w}_{k+1} - 2\boldsymbol{w}_k + \boldsymbol{w}_{k-1}}{h^2} = -\nabla f(\boldsymbol{w}_k) - \frac{\boldsymbol{w}_{k+1} - \boldsymbol{w}_k}{h}$$

then leads to

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \underbrace{\frac{h^2}{m - rh}}_{=\alpha} \nabla f(\boldsymbol{w}_k) + \underbrace{\frac{m}{m - rh}}_{=\beta} (\boldsymbol{w}_k - \boldsymbol{w}_{k-1}).$$

Letting $m = 0$ recovers the gradient descent update. Hence, positive mass $m > 0$ corresponds to the momentum term. The gradient descent update in turn can be interpreted as an Euler discretization of the gradient flow $\boldsymbol{w}'(t) = -\nabla f(\boldsymbol{w}(t))$.

</div>

#### 10.3.2 Nesterov Acceleration

Nesterov's accelerated gradient method (NAG) builds on the heavy ball method. After initializing $\boldsymbol{w}_0$, $\boldsymbol{v}_0 \in \mathbb{R}^n$, the update is formulated for $k \ge 0$ as the two-step process

$$\boldsymbol{w}_{k+1} = \boldsymbol{v}_k - \alpha \nabla f(\boldsymbol{v}_k)$$

$$\boldsymbol{v}_{k+1} = \boldsymbol{w}_{k+1} + \beta(\boldsymbol{w}_{k+1} - \boldsymbol{w}_k)$$

where again $\alpha > 0$ and $\beta \in (0, 1)$ are hyperparameters. Substituting the second line into the first we get for $k \ge 1$

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \alpha \nabla f(\boldsymbol{v}_k) + \beta(\boldsymbol{w}_k - \boldsymbol{w}_{k-1}).$$

Comparing with the heavy ball method, the key difference is that the gradient is not evaluated at the current position $\boldsymbol{w}_k$, but instead at the point $\boldsymbol{v}_k = \boldsymbol{w}_k + \beta(\boldsymbol{w}_k - \boldsymbol{w}_{k-1})$, which can be interpreted as an estimate of the position at the next iteration. This improves stability and robustness with respect to hyperparameter settings.

We now discuss the convergence of NAG for $L$-smooth and $\mu$-strongly convex objective functions $f$. To give the analysis, it is convenient to first rewrite the two-step update as a three sequence update: Let $\tau = \sqrt{\mu / L}$, $\alpha = 1/L$, and $\beta = (1 - \tau)/(1 + \tau)$. After initializing $\boldsymbol{w}_0$, $\boldsymbol{v}_0 \in \mathbb{R}^n$, the update can also be written as $\boldsymbol{u}_0 = ((1 + \tau)\boldsymbol{v}_0 - \boldsymbol{w}_0) / \tau$ and for $k \ge 0$

$$\boldsymbol{v}_k = \frac{\tau}{1 + \tau} \boldsymbol{u}_k + \frac{1}{1 + \tau} \boldsymbol{w}_k$$

$$\boldsymbol{w}_{k+1} = \boldsymbol{v}_k - \frac{1}{L} \nabla f(\boldsymbol{v}_k)$$

$$\boldsymbol{u}_{k+1} = \boldsymbol{u}_k + \tau \cdot (\boldsymbol{v}_k - \boldsymbol{u}_k) - \frac{\tau}{\mu} \nabla f(\boldsymbol{v}_k).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.13</span><span class="math-callout__name">(Convergence of Nesterov Acceleration)</span></p>

Let $n \in \mathbb{N}$, $0 < \mu \le L$, and let $f : \mathbb{R}^n \to \mathbb{R}$ be $L$-smooth and $\mu$-strongly convex. Further, let $\boldsymbol{w}_0$, $\boldsymbol{v}_0 \in \mathbb{R}^n$ and let $\tau = \sqrt{\mu / L}$. Let $(\boldsymbol{v}_k, \boldsymbol{w}_{k+1}, \boldsymbol{u}_{k+1})_{k=0}^\infty \subseteq \mathbb{R}^n$ be defined by the three-sequence update (10.3.11a), and let $\boldsymbol{w}_*$ be the unique minimizer of $f$.

Then, for all $k \in \mathbb{N}_0$, it holds that

$$\lVert \boldsymbol{u}_k - \boldsymbol{w}_* \rVert^2 \le \frac{2}{\mu} \left(1 - \sqrt{\frac{\mu}{L}}\right)^k \left(f(\boldsymbol{w}_0) - f(\boldsymbol{w}_*) + \frac{\mu}{2} \lVert \boldsymbol{u}_0 - \boldsymbol{w}_* \rVert^2\right),$$

$$f(\boldsymbol{w}_k) - f(\boldsymbol{w}_*) \le \left(1 - \sqrt{\frac{\mu}{L}}\right)^k \left(f(\boldsymbol{w}_0) - f(\boldsymbol{w}_*) + \frac{\mu}{2} \lVert \boldsymbol{u}_0 - \boldsymbol{w}_* \rVert^2\right).$$

</div>

*Proof sketch.* Define

$$e_k := (f(\boldsymbol{w}_k) - f(\boldsymbol{w}_*)) + \frac{\mu}{2} \lVert \boldsymbol{u}_k - \boldsymbol{w}_* \rVert^2.$$

With $c := 1 - \tau$, it suffices to prove that $e_{k+1} \le c \, e_k$ for all $k \in \mathbb{N}_0$.

**Step 1.** We bound the first term in $e_{k+1}$. Using $L$-smoothness and (10.3.11b)

$$f(\boldsymbol{w}_{k+1}) - f(\boldsymbol{v}_k) \le -\frac{1}{2L} \lVert \nabla f(\boldsymbol{v}_k) \rVert^2.$$

Thus, since $c + \tau = 1$,

$$f(\boldsymbol{w}_{k+1}) - f(\boldsymbol{w}_*) \le c \cdot (f(\boldsymbol{w}_k) - f(\boldsymbol{w}_*)) + \tau \cdot (f(\boldsymbol{v}_k) - f(\boldsymbol{w}_*)) - \frac{1}{2L} \lVert \nabla f(\boldsymbol{v}_k) \rVert^2.$$

**Step 2.** We bound the second term in $e_{k+1}$. By (10.3.11c) and using $\mu$-strong convexity

$$\frac{\mu}{2} \lVert \boldsymbol{u}_{k+1} - \boldsymbol{w}_* \rVert^2 \le c \frac{\mu}{2} \lVert \boldsymbol{u}_k - \boldsymbol{w}_* \rVert^2 + \tau \langle \nabla f(\boldsymbol{v}_k), \boldsymbol{v}_k - \boldsymbol{u}_k \rangle - \tau \cdot (f(\boldsymbol{v}_k) - f(\boldsymbol{w}_*)) - \frac{\tau \mu}{2}(\lVert \boldsymbol{u}_k - \boldsymbol{w}_* \rVert^2 + \lVert \boldsymbol{v}_k - \boldsymbol{u}_k \rVert^2).$$

**Step 3.** Adding the bounds from Steps 1 and 2, and using (10.3.11a) which gives $\tau \cdot (\boldsymbol{v}_k - \boldsymbol{u}_k) = \boldsymbol{w}_k - \boldsymbol{v}_k$, the cross terms involving $\nabla f(\boldsymbol{v}_k)$ and the terms involving $f(\boldsymbol{v}_k) - f(\boldsymbol{w}_*)$ cancel out. Since $\tau = \sqrt{\mu / L}$, the coefficients of $\lVert \nabla f(\boldsymbol{v}_k) \rVert^2$ and $\lVert \boldsymbol{w}_k - \boldsymbol{v}_k \rVert^2$ vanish, yielding $e_{k+1} \le c \, e_k$. $\square$

Comparing the result for gradient descent (Theorem 10.7) with NAG (Theorem 10.13), the improvement for strongly convex objectives lies in the convergence rate, which is $1 - \kappa^{-1}$ for gradient descent and $1 - \kappa^{-1/2}$ for NAG, where $\kappa = L / \mu$. For NAG the convergence rate depends only on the *square root* of the condition number $\kappa$. For ill-conditioned problems where $\kappa$ is large, we therefore expect much better performance for accelerated methods.

### 10.4 Adaptive and Coordinate-wise Learning Rates

In Section 10.3.1, we saw why plain gradient descent can be inefficient for ill-conditioned objective functions. This issue can be particularly pronounced in high-dimensional optimization problems, such as when training neural networks, where certain parameters influence the network output much more than others. A simpler and computationally efficient alternative is to scale each component of the gradient individually, corresponding to a diagonal preconditioning matrix. This allows different learning rates for different coordinates.

After initializing $\boldsymbol{u}_0 = \boldsymbol{0} \in \mathbb{R}^n$, $\boldsymbol{s}_0 = \boldsymbol{0} \in \mathbb{R}^n$, and $\boldsymbol{w}_0 \in \mathbb{R}^n$, all methods discussed below are special cases of

$$\boldsymbol{u}_{k+1} = \beta_1 \boldsymbol{u}_k + \beta_2 \nabla f(\boldsymbol{w}_k)$$

$$\boldsymbol{s}_{k+1} = \gamma_1 \boldsymbol{s}_k + \gamma_2 \nabla f(\boldsymbol{w}_k) \odot \nabla f(\boldsymbol{w}_k)$$

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \alpha_k \boldsymbol{u}_{k+1} \oslash \sqrt{\boldsymbol{s}_{k+1} + \varepsilon}$$

for $k \in \mathbb{N}_0$, and certain hyperparameters $\alpha_k$, $\beta_1$, $\beta_2$, $\gamma_1$, $\gamma_2$, and $\varepsilon$. Here $\odot$ and $\oslash$ denote the componentwise (Hadamard) multiplication and division, respectively, and $\sqrt{\boldsymbol{s}_{k+1} + \varepsilon}$ is understood as the vector $(\sqrt{v_{k+1,i} + \varepsilon})_i$. Equation (10.4.1a) defines an update vector and corresponds to heavy ball momentum if $\beta_1 \in (0, 1)$. Equation (10.4.1b) defines a scaling vector $\boldsymbol{s}_{k+1}$ that is used to set a coordinate-wise learning rate of the update vector in (10.4.1c). The constant $\varepsilon > 0$ is chosen small but positive to avoid division by zero in (10.4.1c).

#### 10.4.1 Coordinate-wise Scaling

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 10.14</span></p>

Consider an objective function $f : \mathbb{R}^n \to \mathbb{R}$, and its rescaled version

$$f_{\boldsymbol{\zeta}}(\boldsymbol{w}) := f(\boldsymbol{w} \odot \boldsymbol{\zeta}) \quad \text{with gradient} \quad \nabla f_{\boldsymbol{\zeta}}(\boldsymbol{w}) = \boldsymbol{\zeta} \odot \nabla f(\boldsymbol{w} \odot \boldsymbol{\zeta}),$$

for some $\boldsymbol{\zeta} \in (0, \infty)^n$. Gradient descent applied to $f_{\boldsymbol{\zeta}}$ performs the update $\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - h_k \boldsymbol{\zeta} \odot \nabla f(\boldsymbol{w} \odot \boldsymbol{\zeta})$. Setting $\varepsilon = 0$, the adaptive method (10.4.1) on the other hand performs the update

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \alpha_k \left(\beta_2 \sum_{j=0}^k \beta_1^j \nabla f(\boldsymbol{w}_{k-j} \odot \boldsymbol{\zeta})\right) \oslash \sqrt{\gamma_2 \sum_{j=0}^k \gamma_1^j \nabla f(\boldsymbol{w}_{k-j} \odot \boldsymbol{\zeta}) \odot \nabla f(\boldsymbol{w}_{k-j} \odot \boldsymbol{\zeta})}.$$

Note how the outer scaling factor $\boldsymbol{\zeta}$ has vanished due to the division, in this sense making the update invariant to a componentwise rescaling of the objective.

</div>

#### 10.4.2 Algorithms

**AdaGrad.** AdaGrad (Adaptive Gradient Algorithm) corresponds to (10.4.1) with

$$\beta_1 = 0, \quad \gamma_1 = \beta_2 = \gamma_2 = 1, \quad \alpha_k = \alpha \quad \text{for all } k \in \mathbb{N}_0.$$

This leaves the hyperparameters $\varepsilon > 0$ and $\alpha > 0$. Here $\alpha > 0$ can be understood as a "global" learning rate. The AdaGrad update then reads

$$\boldsymbol{s}_{k+1} = \boldsymbol{s}_k + \nabla f(\boldsymbol{w}_k) \odot \nabla f(\boldsymbol{w}_k)$$

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \alpha \nabla f(\boldsymbol{w}_k) \oslash \sqrt{\boldsymbol{s}_{k+1} + \varepsilon}.$$

Due to $\boldsymbol{s}_{k+1} = \sum_{j=0}^k \nabla f(\boldsymbol{w}_j) \odot \nabla f(\boldsymbol{w}_j)$, the algorithm scales the gradient $\nabla f(\boldsymbol{w}_k)$ in the update componentwise by the inverse square root of the sum over all past squared gradients plus $\varepsilon$. Note that the scaling factor $(s_{k+1,i} + \varepsilon)^{-1/2}$ for component $i$ will be large, if the previous gradients for that component were small, and vice versa.

**RMSProp.** RMSProp (Root Mean Squared Propagation) corresponds to (10.4.1) with

$$\beta_1 = 0, \quad \beta_2 = 1, \quad \gamma_2 = 1 - \gamma_1 \in (0, 1), \quad \alpha_k = \alpha \quad \text{for all } k \in \mathbb{N}_0,$$

effectively leaving the hyperparameters $\varepsilon > 0$, $\gamma_1 \in (0, 1)$ and $\alpha > 0$. The algorithm is thus given through

$$\boldsymbol{s}_{k+1} = \gamma_1 \boldsymbol{s}_k + (1 - \gamma_1) \nabla f(\boldsymbol{w}_k) \odot \nabla f(\boldsymbol{w}_k)$$

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \alpha \nabla f(\boldsymbol{w}_k) \oslash \sqrt{\boldsymbol{s}_{k+1} + \varepsilon}.$$

The scaling vector can be expressed as $\boldsymbol{s}_{k+1} = (1 - \gamma_1) \sum_{j=0}^k \gamma_1^j \nabla f(\boldsymbol{w}_{k-j}) \odot \nabla f(\boldsymbol{w}_{k-j})$ and corresponds to an exponentially weighted moving average over the past squared gradients. Unlike for AdaGrad, where past gradients accumulate indefinitely, RMSprop exponentially downweights older gradients, giving more weight to recent updates. This prevents the overly rapid decay of learning rates and slow convergence sometimes observed in AdaGrad.

**Adam.** Adam (Adaptive Moment Estimation) corresponds to (10.4.1) with

$$\beta_2 = 1 - \beta_1 \in (0, 1), \quad \gamma_2 = 1 - \gamma_1 \in (0, 1), \quad \alpha_k = \alpha \frac{\sqrt{1 - \gamma_1^{k+1}}}{1 - \beta_1^{k+1}}$$

for some $\alpha > 0$. The default values for the remaining parameters recommended are $\varepsilon = 10^{-8}$, $\alpha = 0.001$, $\beta_1 = 0.9$ and $\gamma_1 = 0.999$. The update can be formulated as

$$\boldsymbol{u}_{k+1} = \beta_1 \boldsymbol{u}_k + (1 - \beta_1) \nabla f(\boldsymbol{w}_k), \qquad \hat{\boldsymbol{u}}_{k+1} = \frac{\boldsymbol{u}_{k+1}}{1 - \beta_1^{k+1}}$$

$$\boldsymbol{s}_{k+1} = \gamma_1 \boldsymbol{s}_k + (1 - \gamma_1) \nabla f(\boldsymbol{w}_k) \odot \nabla f(\boldsymbol{w}_k), \qquad \hat{\boldsymbol{s}}_{k+1} = \frac{\boldsymbol{s}_{k+1}}{1 - \gamma_1^{k+1}}$$

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \alpha \hat{\boldsymbol{u}}_{k+1} \oslash \sqrt{\hat{\boldsymbol{s}}_{k+1} + \varepsilon}.$$

Compared to RMSProp, Adam introduces two modifications. First, due to $\beta_1 > 0$,

$$\boldsymbol{u}_{k+1} = (1 - \beta_1) \sum_{j=0}^k \beta_1^j \nabla f(\boldsymbol{w}_{k-j})$$

which corresponds to heavy ball momentum (cf. (10.3.6)). Second, to counteract the initialization bias from $\boldsymbol{u}_0 = \boldsymbol{0}$ and $\boldsymbol{s}_0 = \boldsymbol{0}$, Adam applies a bias correction via

$$\hat{\boldsymbol{u}}_k = \frac{\boldsymbol{u}_k}{1 - \beta_1^k}, \quad \hat{\boldsymbol{s}}_k = \frac{\boldsymbol{s}_k}{1 - \gamma_1^k}.$$

It should be noted that there exist specific settings and convex optimization problems for which Adam (and RMSProp and Adadelta) does not necessarily converge to a minimizer. Nonetheless, Adam remains a highly popular algorithm for the training of neural networks.

### 10.5 Backpropagation

We now explain how to apply gradient-based methods to the training of neural networks. Let $\Phi \in \mathcal{N}_{d_0}^{d_{L+1}}(\sigma; L, n)$ (see Definition 3.6) and assume that the activation function satisfies $\sigma \in C^1(\mathbb{R})$. As earlier, we denote the neural network parameters by

$$\boldsymbol{w} = ((\boldsymbol{W}^{(0)}, \boldsymbol{b}^{(0)}), \ldots, (\boldsymbol{W}^{(L)}, \boldsymbol{b}^{(L)}))$$

with weight matrices $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$ and bias vectors $\boldsymbol{b}^{(\ell)} \in \mathbb{R}^{d_{\ell+1}}$. Additionally, we fix a differentiable loss function $\mathcal{L} : \mathbb{R}^{d_{L+1}} \times \mathbb{R}^{d_{L+1}} \to \mathbb{R}$, e.g., $\mathcal{L}(\boldsymbol{w}, \tilde{\boldsymbol{w}}) = \lVert \boldsymbol{w} - \tilde{\boldsymbol{w}} \rVert^2 / 2$, and assume given data $(\boldsymbol{x}_j, \boldsymbol{y}_j)_{j=1}^m \subseteq \mathbb{R}^{d_0} \times \mathbb{R}^{d_{L+1}}$. The goal is to minimize an empirical risk of the form

$$f(\boldsymbol{w}) := \frac{1}{m} \sum_{j=1}^m \mathcal{L}(\Phi(\boldsymbol{x}_j, \boldsymbol{w}), \boldsymbol{y}_j).$$

An application of the gradient step requires the computation of $\nabla f(\boldsymbol{w}) = \frac{1}{m} \sum_{j=1}^m \nabla_{\boldsymbol{w}} \mathcal{L}(\Phi(\boldsymbol{x}_j, \boldsymbol{w}), \boldsymbol{y}_j)$. For stochastic methods, we only compute the average over a (random) subbatch of the dataset. In either case, we need an algorithm to determine $\nabla_{\boldsymbol{w}} \mathcal{L}(\Phi(\boldsymbol{x}, \boldsymbol{w}), \boldsymbol{y})$, i.e. the gradients

$$\nabla_{\boldsymbol{b}^{(\ell)}} \mathcal{L}(\Phi(\boldsymbol{x}, \boldsymbol{w}), \boldsymbol{y}) \in \mathbb{R}^{d_{\ell+1}}, \quad \nabla_{\boldsymbol{W}^{(\ell)}} \mathcal{L}(\Phi(\boldsymbol{x}, \boldsymbol{w}), \boldsymbol{y}) \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$$

for all $\ell = 0, \ldots, L$.

The backpropagation algorithm provides an *efficient* way to do so, by storing intermediate values in the calculation. For fixed input $\boldsymbol{x} \in \mathbb{R}^{d_0}$ introduce the notation

$$\bar{\boldsymbol{x}}^{(1)} := \boldsymbol{W}^{(0)} \boldsymbol{x} + \boldsymbol{b}^{(0)}$$

$$\bar{\boldsymbol{x}}^{(\ell+1)} := \boldsymbol{W}^{(\ell)} \sigma(\bar{\boldsymbol{x}}^{(\ell)}) + \boldsymbol{b}^{(\ell)} \quad \text{for } \ell \in \lbrace 1, \ldots, L \rbrace,$$

where the application of $\sigma$ to a vector is understood componentwise. With the notation of Definition 2.1, $\boldsymbol{x}^{(\ell)} = \sigma(\bar{\boldsymbol{x}}^{(\ell)}) \in \mathbb{R}^{d_\ell}$ for $\ell = 1, \ldots, L$ and $\bar{\boldsymbol{x}}^{(L+1)} = \Phi(\boldsymbol{x}, \boldsymbol{w}) \in \mathbb{R}^{d_{L+1}}$ is the output of the neural network. The $\bar{\boldsymbol{x}}^{(\ell)}$ for $\ell = 1, \ldots, L$ are sometimes also referred to as the **preactivations**.

In the following, we additionally fix $\boldsymbol{y} \in \mathbb{R}^{d_{L+1}}$ and write $\mathcal{L} := \mathcal{L}(\Phi(\boldsymbol{x}, \boldsymbol{w}), \boldsymbol{y}) = \mathcal{L}(\bar{\boldsymbol{x}}^{(L+1)}, \boldsymbol{y})$. Note that $\bar{\boldsymbol{x}}^{(k)}$ depends on $(\boldsymbol{W}^{(\ell)}, \boldsymbol{b}^{(\ell)})$ only if $k > \ell$. Since $\bar{\boldsymbol{x}}^{(\ell+1)}$ is a function of $\bar{\boldsymbol{x}}^{(\ell)}$ for each $\ell$, by repeated application of the chain rule

$$\frac{\partial \mathcal{L}}{\partial W_{ij}^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial \bar{\boldsymbol{x}}^{(L+1)}} \frac{\partial \bar{\boldsymbol{x}}^{(L+1)}}{\partial \bar{\boldsymbol{x}}^{(L)}} \cdots \frac{\partial \bar{\boldsymbol{x}}^{(\ell+2)}}{\partial \bar{\boldsymbol{x}}^{(\ell+1)}} \frac{\partial \bar{\boldsymbol{x}}^{(\ell+1)}}{\partial W_{ij}^{(\ell)}}.$$

To avoid unnecessary computations, the main idea of backpropagation is to introduce

$$\boldsymbol{\alpha}^{(\ell)} := \nabla_{\bar{\boldsymbol{x}}^{(\ell)}} \mathcal{L} \in \mathbb{R}^{d_\ell} \quad \text{for all } \ell = 1, \ldots, L+1.$$

As the following lemma shows, the $\boldsymbol{\alpha}^{(\ell)}$ can be computed recursively for $\ell = L+1, \ldots, 1$. This explains the name "backpropagation".

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 10.15</span><span class="math-callout__name">(Backpropagation Recursion)</span></p>

It holds

$$\boldsymbol{\alpha}^{(L+1)} = \nabla_{\bar{\boldsymbol{x}}^{(L+1)}} \mathcal{L}(\bar{\boldsymbol{x}}^{(L+1)}, \boldsymbol{y})$$

and

$$\boldsymbol{\alpha}^{(\ell)} = \sigma'(\bar{\boldsymbol{x}}^{(\ell)}) \odot (\boldsymbol{W}^{(\ell)})^\top \boldsymbol{\alpha}^{(\ell+1)} \quad \text{for all } \ell = L, \ldots, 1.$$

</div>

*Proof.* The first equation holds by definition. For $\ell \in \lbrace 1, \ldots, L \rbrace$ by the chain rule

$$\boldsymbol{\alpha}^{(\ell)} = \frac{\partial \mathcal{L}}{\partial \bar{\boldsymbol{x}}^{(\ell)}} = \left(\frac{\partial \bar{\boldsymbol{x}}^{(\ell+1)}}{\partial \bar{\boldsymbol{x}}^{(\ell)}}\right)^\top \frac{\partial \mathcal{L}}{\partial \bar{\boldsymbol{x}}^{(\ell+1)}} = \left(\frac{\partial \bar{\boldsymbol{x}}^{(\ell+1)}}{\partial \bar{\boldsymbol{x}}^{(\ell)}}\right)^\top \boldsymbol{\alpha}^{(\ell+1)}.$$

By (10.5.3b) for $i \in \lbrace 1, \ldots, d_{\ell+1} \rbrace$, $j \in \lbrace 1, \ldots, d_\ell \rbrace$

$$\left(\frac{\partial \bar{\boldsymbol{x}}^{(\ell+1)}}{\partial \bar{\boldsymbol{x}}^{(\ell)}}\right)_{ij} = \frac{\partial \bar{x}_i^{(\ell+1)}}{\partial \bar{x}_j^{(\ell)}} = W_{ij}^{(\ell)} \sigma'(\bar{x}_j^{(\ell)}).$$

Thus the claim follows. $\square$

Putting everything together, we obtain explicit formulas for the gradients.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10.16</span><span class="math-callout__name">(Gradient Formulas)</span></p>

It holds

$$\nabla_{\boldsymbol{b}^{(\ell)}} \mathcal{L} = \boldsymbol{\alpha}^{(\ell+1)} \in \mathbb{R}^{d_{\ell+1}} \quad \text{for } \ell = 0, \ldots, L$$

and

$$\nabla_{\boldsymbol{W}^{(0)}} \mathcal{L} = \boldsymbol{\alpha}^{(1)} \boldsymbol{x}^\top \in \mathbb{R}^{d_1 \times d_0}$$

and

$$\nabla_{\boldsymbol{W}^{(\ell)}} \mathcal{L} = \boldsymbol{\alpha}^{(\ell+1)} \sigma(\bar{\boldsymbol{x}}^{(\ell)})^\top \in \mathbb{R}^{d_{\ell+1} \times d_\ell} \quad \text{for } \ell = 1, \ldots, L.$$

</div>

Lemma 10.15 and Proposition 10.16 motivate the following algorithm, in which a forward pass computing $\bar{\boldsymbol{x}}^{(\ell)}$, $\ell = 1, \ldots, L+1$, is followed by a backward pass to determine the $\boldsymbol{\alpha}^{(\ell)}$, $\ell = L+1, \ldots, 1$, and the gradients of $\mathcal{L}$ with respect to the neural network parameters.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 1</span><span class="math-callout__name">(Backpropagation)</span></p>

**Input:** Network input $\boldsymbol{x}$, target output $\boldsymbol{y}$, neural network parameters $((\boldsymbol{W}^{(0)}, \boldsymbol{b}^{(0)}), \ldots, (\boldsymbol{W}^{(L)}, \boldsymbol{b}^{(L)}))$

**Output:** Gradients of the loss function $\mathcal{L}$ with respect to neural network parameters

**Forward pass**
- $\bar{\boldsymbol{x}}^{(1)} \leftarrow \boldsymbol{W}^{(0)} \boldsymbol{x} + \boldsymbol{b}^{(0)}$
- **for** $\ell = 1, \ldots, L$ **do** $\bar{\boldsymbol{x}}^{(\ell+1)} \leftarrow \boldsymbol{W}^{(\ell)} \sigma(\bar{\boldsymbol{x}}^{(\ell)}) + \boldsymbol{b}^{(\ell)}$

**Backward pass**
- $\boldsymbol{\alpha}^{(L+1)} \leftarrow \nabla_{\bar{\boldsymbol{x}}^{(L+1)}} \mathcal{L}(\bar{\boldsymbol{x}}^{(L+1)}, \boldsymbol{y})$
- **for** $\ell = L, \ldots, 1$ **do**
  - $\nabla_{\boldsymbol{b}^{(\ell)}} \mathcal{L} \leftarrow \boldsymbol{\alpha}^{(\ell+1)}$
  - $\nabla_{\boldsymbol{W}^{(\ell)}} \mathcal{L} \leftarrow \boldsymbol{\alpha}^{(\ell+1)} \sigma(\bar{\boldsymbol{x}}^{(\ell)})^\top$
  - $\boldsymbol{\alpha}^{(\ell)} \leftarrow \sigma'(\bar{\boldsymbol{x}}^{(\ell)}) \odot (\boldsymbol{W}^{(\ell)})^\top \boldsymbol{\alpha}^{(\ell+1)}$
- $\nabla_{\boldsymbol{b}^{(0)}} \mathcal{L} \leftarrow \boldsymbol{\alpha}^{(1)}$
- $\nabla_{\boldsymbol{W}^{(0)}} \mathcal{L} \leftarrow \boldsymbol{\alpha}^{(1)} \boldsymbol{x}^\top$

</div>

Two important remarks are in order. First, the objective function associated to neural networks is typically not convex as a function of the neural network weights and biases. Thus, the analysis of the previous sections will in general not be directly applicable. It may still give some insight about the convergence behavior locally around a (local) minimizer however. Second, we assumed the activation function to be continuously differentiable, which does not hold for ReLU. Using the concept of subgradients, gradient-based algorithms and their analysis may be generalized to some extent to also accommodate non-differentiable loss functions.

---

## Chapter 11: Wide Neural Networks and the Neural Tangent Kernel

This chapter explores the dynamics of training (shallow) neural networks of large width. Throughout we assume given data pairs

$$(\boldsymbol{x}_i, y_i) \in \mathbb{R}^d \times \mathbb{R}, \quad i \in \lbrace 1, \ldots, m \rbrace,$$

for distinct $\boldsymbol{x}_i$. We wish to train a model $\Phi(\boldsymbol{x}, \boldsymbol{w})$ depending on the input $\boldsymbol{x} \in \mathbb{R}^d$ and the parameters $\boldsymbol{w} \in \mathbb{R}^n$. To this end we consider either minimization of the **ridgeless** (unregularized) objective

$$f(\boldsymbol{w}) := \sum_{i=1}^m (\Phi(\boldsymbol{x}_i, \boldsymbol{w}) - y_i)^2,$$

or, for some regularization parameter $\lambda \ge 0$, of the **ridge** regularized objective

$$f_\lambda(\boldsymbol{w}) := f(\boldsymbol{w}) + \lambda \lVert \boldsymbol{w} \rVert^2.$$

The goal is to gain insight into the dynamics of $\Phi(\boldsymbol{x}, \boldsymbol{w}_k)$ as the parameter vector $\boldsymbol{w}_k$ progresses during training, and to understand the influence of regularization. We study this through so-called kernel methods, using gradient descent with constant step size.

If $\Phi(\boldsymbol{x}, \boldsymbol{w})$ depends linearly on the parameters $\boldsymbol{w}$, the objective $f_\lambda$ is convex. For typical neural network architectures, $\boldsymbol{w} \mapsto \Phi(\boldsymbol{x}, \boldsymbol{w})$ is not linear, and convergence to a global minimizer is in general not guaranteed. Recent research has shown that neural network behavior tends to linearize in $\boldsymbol{w}$ as network width increases, allowing transfer of results from the linear case to the training of neural networks.

### 11.1 Linear Least-Squares Regression

Given data $(11.0.1a)$, we fit a linear function $\boldsymbol{x} \mapsto \Phi(\boldsymbol{x}, \boldsymbol{w}) := \boldsymbol{x}^\top \boldsymbol{w}$ by minimizing $f$ or $f_\lambda$. With

$$\boldsymbol{A} = \begin{pmatrix} \boldsymbol{x}_1^\top \\ \vdots \\ \boldsymbol{x}_m^\top \end{pmatrix} \in \mathbb{R}^{m \times d} \qquad \text{and} \qquad \boldsymbol{y} = \begin{pmatrix} y_1 \\ \vdots \\ y_m \end{pmatrix} \in \mathbb{R}^m$$

it holds

$$f(\boldsymbol{w}) = \lVert \boldsymbol{A}\boldsymbol{w} - \boldsymbol{y} \rVert^2 \qquad \text{and} \qquad f_\lambda(\boldsymbol{w}) = f(\boldsymbol{w}) + \lambda \lVert \boldsymbol{w} \rVert^2.$$

The $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_m$ are referred to as the **training points** (or design points), and their span is denoted by

$$\tilde{H} := \operatorname{span}\lbrace \boldsymbol{x}_1, \ldots, \boldsymbol{x}_m \rbrace \subseteq \mathbb{R}^d.$$

This is the subspace spanned by the rows of $\boldsymbol{A}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 11.1</span><span class="math-callout__name">(Bias Term)</span></p>

More generally, the ansatz $\Phi(\boldsymbol{x}, (\boldsymbol{w}, b)) := \boldsymbol{w}^\top \boldsymbol{x} + b$ corresponds to

$$\Phi(\boldsymbol{x}, (\boldsymbol{w}, b)) = (1, \boldsymbol{x}^\top) \begin{pmatrix} b \\ \boldsymbol{w} \end{pmatrix}.$$

Therefore, additionally allowing for a bias can be treated similarly.

</div>

#### 11.1.1 Existence of Minimizers

We start with the ridgeless case $\lambda = 0$. The model $\Phi(\boldsymbol{x}, \boldsymbol{w}) = \boldsymbol{x}^\top \boldsymbol{w}$ is linear in both $\boldsymbol{x}$ and $\boldsymbol{w}$. If $\boldsymbol{A}$ is invertible, then $f$ has the unique minimizer $\boldsymbol{w}_* = \boldsymbol{A}^{-1}\boldsymbol{y}$. If $\operatorname{rank}(\boldsymbol{A}) < d$, then $\ker(\boldsymbol{A}) \neq \lbrace \boldsymbol{0} \rbrace$ and there exist infinitely many minimizers. To guarantee uniqueness, we consider the **minimum norm solution**

$$\boldsymbol{w}_* := \operatorname{argmin}_{\boldsymbol{w} \in M} \lVert \boldsymbol{w} \rVert, \qquad M := \lbrace \boldsymbol{w} \in \mathbb{R}^d \mid f(\boldsymbol{w}) \le f(\boldsymbol{v}) \; \forall \boldsymbol{v} \in \mathbb{R}^d \rbrace.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.2</span><span class="math-callout__name">(Minimum Norm Solution)</span></p>

There is a unique minimum norm solution $\boldsymbol{w}_* \in \mathbb{R}^d$. It lies in the subspace $\tilde{H}$, and is the unique minimizer of $f$ in $\tilde{H}$, i.e.

$$\boldsymbol{w}_* = \operatorname{argmin}_{\tilde{\boldsymbol{w}} \in \tilde{H}} f(\tilde{\boldsymbol{w}}).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.3</span><span class="math-callout__name">(Ridge Regression Minimizer)</span></p>

Let $\lambda > 0$. Then, with $f_\lambda$ as above, there exists a unique minimizer

$$\boldsymbol{w}_{*,\lambda} := \operatorname{argmin}_{\boldsymbol{w} \in \mathbb{R}^d} f_\lambda(\boldsymbol{w}).$$

It holds $\boldsymbol{w}_{*,\lambda} \in \tilde{H}$, and

$$\lim_{\lambda \to 0} \boldsymbol{w}_{*,\lambda} = \boldsymbol{w}_*.$$

</div>

The minimizer is reached at $\nabla f_\lambda(\boldsymbol{w}) = 0$, which yields

$$\boldsymbol{w}_{*,\lambda} = (\boldsymbol{A}^\top \boldsymbol{A} + \lambda \boldsymbol{I}_d)^{-1} \boldsymbol{A}^\top \boldsymbol{y}.$$

Using the singular value decomposition $\boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^\top$, this converges towards $\boldsymbol{A}^\dagger \boldsymbol{y}$ as $\lambda \to 0$, where $\boldsymbol{A}^\dagger$ denotes the pseudoinverse.

#### 11.1.2 Gradient Descent

Consider gradient descent to minimize $f_\lambda$. Starting from $\boldsymbol{w}_0 \in \mathbb{R}^d$, the iterative update with constant step size $h > 0$ reads

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - 2h\boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{w}_k - \boldsymbol{y}) - 2h\lambda \boldsymbol{w}_k \qquad \text{for all } k \in \mathbb{N}_0.$$

For $\lambda = 0$ and sufficiently small step size $h > 0$, gradient descent converges to the minimum norm solution $\boldsymbol{w}_*$ as long as $\boldsymbol{w}_0 \in \tilde{H}$ (e.g. $\boldsymbol{w}_0 = \boldsymbol{0}$).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.4</span><span class="math-callout__name">(Ridgeless Gradient Descent Convergence)</span></p>

Let $\lambda = 0$ and fix $h \in (0, s_{\max}(\boldsymbol{A})^{-2})$. Let $\boldsymbol{w}_0 = \tilde{\boldsymbol{w}}_0 + \hat{\boldsymbol{w}}_0$ where $\tilde{\boldsymbol{w}}_0 \in \tilde{H}$ and $\hat{\boldsymbol{w}}_0 \in \tilde{H}^\perp$, and let $(\boldsymbol{w}_k)_{k \in \mathbb{N}}$ be defined by the gradient descent iteration. Then

$$\lim_{k \to \infty} \boldsymbol{w}_k = \boldsymbol{w}_* + \hat{\boldsymbol{w}}_0.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.5</span><span class="math-callout__name">(Ridge Gradient Descent Convergence)</span></p>

Let $\lambda > 0$, and fix $h \in (0, (2\lambda + 2s_{\max}(\boldsymbol{A})^2)^{-1})$. Let $\boldsymbol{w}_0 \in \mathbb{R}^n$ and let $(\boldsymbol{w}_k)_{k \in \mathbb{N}}$ be defined by the gradient descent iteration. Then

$$\lim_{k \to \infty} \boldsymbol{w}_k = \boldsymbol{w}_{*,\lambda}$$

and

$$\lVert \boldsymbol{w}_* - \boldsymbol{w}_{*,\lambda} \rVert \le \left\lvert \frac{\lambda}{s_{\min}(\boldsymbol{A})^3 + s_{\min}(\boldsymbol{A})\lambda} \right\rvert \lVert \boldsymbol{y} \rVert = O(\lambda) \qquad \text{as } \lambda \to 0.$$

</div>

By Proposition 11.5, if we use ridge regression with a small regularization parameter $\lambda > 0$, then gradient descent converges to a vector $\boldsymbol{w}_{*,\lambda}$ which is $O(\lambda)$ close to the minimal norm solution $\boldsymbol{w}_*$, regardless of the initialization $\boldsymbol{w}_0$.

### 11.2 Feature Methods and Kernel Least-Squares Regression

Linear models are often too simplistic to capture the true relationship between $\boldsymbol{x}$ and $y$. Feature- and kernel-based methods address this by replacing $\boldsymbol{x} \mapsto \langle \boldsymbol{x}, \boldsymbol{w} \rangle$ with $\boldsymbol{x} \mapsto \langle \phi(\boldsymbol{x}), \boldsymbol{w} \rangle$ where $\phi : \mathbb{R}^d \to \mathbb{R}^n$ is a (typically nonlinear) map. This introduces nonlinearity in $\boldsymbol{x}$ while retaining linearity in the parameter $\boldsymbol{w} \in \mathbb{R}^n$.

Let $(H, \langle \cdot, \cdot \rangle_H)$ be a Hilbert space (the **feature space**), and let $\phi : \mathbb{R}^d \to H$ denote the **feature map**. The model is

$$\Phi(\boldsymbol{x}, w) := \langle \phi(\boldsymbol{x}), w \rangle_H$$

with $w \in H$. The components of $\phi$ are referred to as **features**. The goal is to minimize

$$f(w) := \sum_{j=1}^m \big( \langle \phi(\boldsymbol{x}_j), w \rangle_H - y_j \big)^2 \qquad \text{or} \qquad f_\lambda(w) := f(w) + \lambda \lVert w \rVert_H^2.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.6</span><span class="math-callout__name">(Polynomial Features)</span></p>

Let data $(x_i, y_i)_{i=1}^m \subseteq \mathbb{R} \times \mathbb{R}$ be given, and define for $x \in \mathbb{R}$

$$\phi(x) := (1, x, \ldots, x^{n-1})^\top \in \mathbb{R}^n.$$

For $\boldsymbol{w} \in \mathbb{R}^n$, the model $x \mapsto \langle \phi(x), \boldsymbol{w} \rangle = \sum_{j=0}^{n-1} w_j x^j$ can represent any polynomial of degree $n-1$.

</div>

#### 11.2.1 Existence of Minimizers

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.7</span><span class="math-callout__name">(Minimum Norm Solution in Feature Space)</span></p>

There is a unique minimum norm solution $w_* \in H$. It lies in the subspace $\tilde{H} := \operatorname{span}\lbrace \phi(\boldsymbol{x}_1), \ldots, \phi(\boldsymbol{x}_m) \rbrace \subseteq H$, and is the unique minimizer of $f$ in $\tilde{H}$, i.e.

$$w_* = \operatorname{argmin}_{\tilde{w} \in \tilde{H}} f(\tilde{w}).$$

</div>

Statements as in Theorems 11.7 and 11.8, which yield that the minimizer is attained in the finite dimensional subspace $\tilde{H}$, are known in the literature as **representer theorems**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.8</span><span class="math-callout__name">(Ridge Minimizer in Feature Space)</span></p>

Let $\lambda > 0$. Then, with $f_\lambda$ as above, there exists a unique minimizer

$$w_{*,\lambda} := \operatorname{argmin}_{w \in H} f_\lambda(w).$$

It holds $w_{*,\lambda} \in \tilde{H}$, and $\lim_{\lambda \to 0} w_{*,\lambda} = w_*$.

</div>

#### 11.2.2 The Kernel Trick

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 11.9</span><span class="math-callout__name">(Kernel)</span></p>

A symmetric function $K : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ is called a **kernel** if for any $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_k \in \mathbb{R}^d$, $k \in \mathbb{N}$, the **kernel matrix** $\boldsymbol{G} = (K(\boldsymbol{x}_i, \boldsymbol{x}_j))_{i,j=1}^k \in \mathbb{R}^{k \times k}$ is symmetric positive semidefinite.

</div>

Given a feature map $\phi : \mathbb{R}^d \to H$, one can verify that

$$K(\boldsymbol{x}, \boldsymbol{z}) := \langle \phi(\boldsymbol{x}), \phi(\boldsymbol{z}) \rangle_H \qquad \text{for all } \boldsymbol{x}, \boldsymbol{z} \in \mathbb{R}^d$$

defines a kernel. The corresponding kernel matrix $\boldsymbol{G} \in \mathbb{R}^{m \times m}$ is $G_{ij} = K(\boldsymbol{x}_i, \boldsymbol{x}_j)$.

The ansatz $w_* = \sum_{j=1}^m \alpha_j \phi(\boldsymbol{x}_j)$ turns the optimization into

$$\operatorname{argmin}_{\boldsymbol{\alpha} \in \mathbb{R}^m} \lVert \boldsymbol{G}\boldsymbol{\alpha} - \boldsymbol{y} \rVert^2 + \lambda \boldsymbol{\alpha}^\top \boldsymbol{G} \boldsymbol{\alpha}.$$

We refer to

$$\boldsymbol{x} \mapsto \Phi(\boldsymbol{x}, w_{*,\lambda}) = \langle \phi(\boldsymbol{x}), w_{*,\lambda} \rangle_H$$

as the **(ridge or ridgeless) kernel least-squares estimator**. By the above, its computation neither requires explicit knowledge of the feature map $\phi$ nor of $w_{*,\lambda} \in H$. It is sufficient to choose a kernel $K : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ and perform all computations in finite dimensional spaces. This is known as the **kernel trick**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 2</span><span class="math-callout__name">(Kernel Least-Squares Regression)</span></p>

**Input:** Data $(\boldsymbol{x}_i, y_i)_{i=1}^m \in \mathbb{R}^d \times \mathbb{R}$, kernel $K : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$, regularization parameter $\lambda \ge 0$, evaluation point $\boldsymbol{x} \in \mathbb{R}^d$

**Output:** (Ridge or ridgeless) kernel least-squares estimator at $\boldsymbol{x}$

1. Compute the kernel matrix $\boldsymbol{G} = (K(\boldsymbol{x}_i, \boldsymbol{x}_j))_{i,j=1}^m$
2. Determine a minimizer $\boldsymbol{\alpha} \in \mathbb{R}^m$ of $\lVert \boldsymbol{G}\boldsymbol{\alpha} - \boldsymbol{y} \rVert^2 + \lambda \boldsymbol{\alpha}^\top \boldsymbol{G}\boldsymbol{\alpha}$
3. Evaluate $\Phi(\boldsymbol{x}, w_{*,\lambda}) = \sum_{j=1}^m \alpha_j K(\boldsymbol{x}, \boldsymbol{x}_j)$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 11.10</span><span class="math-callout__name">(Mercer's Theorem)</span></p>

If $\Omega \subseteq \mathbb{R}^d$ is compact and $K : \Omega \times \Omega \to \mathbb{R}$ is a continuous kernel, then Mercer's theorem implies existence of a Hilbert space $H$ and a feature map $\phi : \mathbb{R}^d \to H$ such that $K(\boldsymbol{x}, \boldsymbol{z}) = \langle \phi(\boldsymbol{x}), \phi(\boldsymbol{z}) \rangle_H$ for all $\boldsymbol{x}, \boldsymbol{z} \in \Omega$.

</div>

#### 11.2.3 Gradient Descent

In practice we may either minimize $f_\lambda$ in the Hilbert space $H$ or the finite-dimensional objective. Assuming $H = \mathbb{R}^n$ equipped with the Euclidean inner product, gradient descent with constant step size $h > 0$ reads

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - 2h\boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{w}_k - \boldsymbol{y}) - 2h\lambda \boldsymbol{w}_k$$

where now $\boldsymbol{A}$ has rows $\phi(\boldsymbol{x}_i)^\top$. For $\lambda = 0$ and sufficiently small step size, by Proposition 11.4, for $\boldsymbol{x} \in \mathbb{R}^d$

$$\lim_{k \to \infty} \Phi(\boldsymbol{x}, \boldsymbol{w}_k) = \langle \phi(\boldsymbol{x}), \boldsymbol{p}_* \rangle + \langle \phi(\boldsymbol{x}), \hat{\boldsymbol{w}}_0 \rangle,$$

where $\boldsymbol{w}_0 = \tilde{\boldsymbol{w}}_0 + \hat{\boldsymbol{w}}_0$ with $\tilde{\boldsymbol{w}}_0 \in \tilde{H}$ and $\hat{\boldsymbol{w}}_0 \in \tilde{H}^\perp$. For $\lambda = 0$, gradient descent thus yields the ridgeless kernel least-squares estimator plus an additional term $\langle \phi(\boldsymbol{x}), \hat{\boldsymbol{w}}_0 \rangle$ depending on initialization.

For $\lambda > 0$ and sufficiently small step size, by Proposition 11.5

$$\lim_{k \to \infty} \Phi(\boldsymbol{x}, \boldsymbol{w}_k) = \langle \phi(\boldsymbol{x}), w_{*,\lambda} \rangle = \langle \phi(\boldsymbol{x}), w_* \rangle + O(\lambda) \qquad \text{as } \lambda \to 0.$$

Thus, for $\lambda > 0$ gradient descent determines the ridge kernel least-squares estimator regardless of the initialization.

### 11.3 Tangent Kernel

Consider a general model $\Phi(\boldsymbol{x}, \boldsymbol{w})$ with input $\boldsymbol{x} \in \mathbb{R}^d$ and parameters $\boldsymbol{w} \in \mathbb{R}^n$. The goal is to minimize the square loss objective $f(\boldsymbol{w})$ given data. If $\boldsymbol{w} \mapsto \Phi(\boldsymbol{x}, \boldsymbol{w})$ is not linear, then the objective function is in general not convex. We simplify the situation by **linearizing the model in the parameter** $\boldsymbol{w} \in \mathbb{R}^n$ around initialization: fixing $\boldsymbol{w}_0 \in \mathbb{R}^n$, let

$$\Phi^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{p}) := \Phi(\boldsymbol{x}, \boldsymbol{w}_0) + \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}_0)^\top (\boldsymbol{p} - \boldsymbol{w}_0) \qquad \text{for all } \boldsymbol{w} \in \mathbb{R}^n,$$

which is the first order Taylor approximation of $\Phi$ around $\boldsymbol{w}_0$. Introduce

$$\delta_j := y_j - \Phi(\boldsymbol{x}_i, \boldsymbol{w}_0) + \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}_j, \boldsymbol{w}_0)^\top \boldsymbol{w}_0 \qquad \text{for all } j = 1, \ldots, m.$$

The square loss objective for the linearized model reads

$$f^{\mathrm{lin}}(\boldsymbol{p}) := \sum_{j=1}^m (\Phi^{\mathrm{lin}}(\boldsymbol{x}_j, \boldsymbol{p}) - y_j)^2 = \sum_{j=1}^m \big( \langle \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}_j, \boldsymbol{w}_0), \boldsymbol{p} \rangle - \delta_j \big)^2.$$

Comparing with the kernel least-squares setting, minimizing $f^{\mathrm{lin}}$ corresponds to kernel regression with feature map $\phi(\boldsymbol{x}) = \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}_0) \in \mathbb{R}^n$. The corresponding kernel is

$$\hat{K}_n(\boldsymbol{x}, \boldsymbol{z}) = \langle \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}_0), \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{z}, \boldsymbol{w}_0) \rangle.$$

We refer to $\hat{K}_n$ as the empirical **tangent kernel**, as it arises from the first order Taylor approximation (the tangent) of $\Phi$ around $\boldsymbol{w}_0$. Note that $\hat{K}_n$ depends on the choice of $\boldsymbol{w}_0$.

### 11.4 Global Minimizers

Consider a general model $\Phi : \mathbb{R}^d \times \mathbb{R}^n \to \mathbb{R}$ and the ridgeless square loss

$$f(\boldsymbol{w}) = \sum_{j=1}^m (\Phi(\boldsymbol{x}_j, \boldsymbol{w}) - y_j)^2.$$

The key idea: if $\boldsymbol{w} \mapsto \Phi(\boldsymbol{x}, \boldsymbol{w})$ is nonlinear but sufficiently *close to its linearization* $\Phi^{\mathrm{lin}}$ within some region, the objective function behaves almost like a convex function there. If the region is *large enough* to contain both the initial value $\boldsymbol{w}_0$ and a global minimum, then gradient descent will never leave this (almost convex) basin and find a global minimizer.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption 11.11</span><span class="math-callout__name">(Regularity Conditions)</span></p>

Let $\Phi \in C^1(\mathbb{R}^d \times \mathbb{R}^n)$ and $\boldsymbol{w}_0 \in \mathbb{R}^n$. There exist constants $r, R, U, L > 0$ and $0 < \theta_{\min} \le \theta_{\max} < \infty$ such that $\lVert \boldsymbol{x}_i \rVert \le R$ for all $i = 1, \ldots, m$, and it holds:

**(a)** The kernel matrix of the empirical tangent kernel

$$(\hat{K}_n(\boldsymbol{x}_i, \boldsymbol{x}_j))_{i,j=1}^m = \big( \langle \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}_i, \boldsymbol{w}_0), \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}_j, \boldsymbol{w}_0) \rangle \big)_{i,j=1}^m \in \mathbb{R}^{m \times m}$$

is regular and its eigenvalues belong to $[\theta_{\min}, \theta_{\max}]$.

**(b)** For all $\boldsymbol{x} \in \mathbb{R}^d$ with $\lVert \boldsymbol{x} \rVert \le R$:

$$\lVert \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}) \rVert \le U \qquad \text{for all } \boldsymbol{w} \in B_r(\boldsymbol{w}_0)$$

$$\lVert \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}) - \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{v}) \rVert \le L \lVert \boldsymbol{w} - \boldsymbol{v} \rVert \qquad \text{for all } \boldsymbol{w}, \boldsymbol{v} \in B_r(\boldsymbol{w}_0)$$

**(c)**

$$L \le \frac{\theta_{\min}^2}{12 m^{3/2} U^2 \sqrt{f(\boldsymbol{w}_0)}} \qquad \text{and} \qquad r = \frac{2\sqrt{m} U \sqrt{f(\boldsymbol{w}_0)}}{\theta_{\min}}.$$

</div>

Condition (a) implies in particular that the Jacobian matrix has full rank $m \le n$ (thus we have at least as many parameters as training data $m$). Condition (b) formalizes the required closeness of $\Phi$ and its linearization. Condition (c) ties together all constants, ensuring the full model to be sufficiently close to its linearization in a large enough ball around $\boldsymbol{w}_0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.12</span><span class="math-callout__name">(Global Convergence of Gradient Descent)</span></p>

Let Assumption 11.11 hold. Fix a positive learning rate

$$h \le \frac{1}{\theta_{\min} + \theta_{\max}}.$$

Let $(\boldsymbol{w}_k)_{k \in \mathbb{N}}$ be generated by gradient descent, i.e. for all $k \in \mathbb{N}_0$

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - h \nabla f(\boldsymbol{w}_k).$$

It then holds for all $k \in \mathbb{N}$

$$\lVert \boldsymbol{w}_k - \boldsymbol{w}_0 \rVert \le r$$

$$f(\boldsymbol{w}_k) \le (1 - h\theta_{\min})^{2k} f(\boldsymbol{w}_0).$$

</div>

The loss bound implies that gradient descent achieves zero loss in the limit. Consequently, the limiting model interpolates the training data. This shows convergence to a global minimizer for the (generally nonconvex) optimization problem.

### 11.5 Proximity to Trained Linearized Model

The analysis in Section 11.4 was based on the observation that the linearization $\Phi^{\mathrm{lin}}$ closely mimics the behaviour of the full model $\Phi$ for parameters within distance $r$ of $\boldsymbol{w}_0$. Theorem 11.12 states that the parameters remain within this range throughout training. This suggests that the predictions of the trained full model $\lim_{k \to \infty} \Phi(\boldsymbol{x}, \boldsymbol{w}_k)$ are similar to those of the trained linear model $\lim_{k \to \infty} \Phi^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{p}_k)$.

#### 11.5.1 Evolution of Model Predictions

Define $\boldsymbol{\Phi}(\boldsymbol{X}, \boldsymbol{w}) := (\Phi(\boldsymbol{x}_i, \boldsymbol{w}))_{i=1}^m \in \mathbb{R}^m$ and similarly $\boldsymbol{\Phi}^{\mathrm{lin}}(\boldsymbol{X}, \boldsymbol{p})$. Under gradient descent, the model predictions evolve as:

**Full model:**

$$\Phi(\boldsymbol{x}, \boldsymbol{w}_{k+1}) = \Phi(\boldsymbol{x}, \boldsymbol{w}_k) - 2h \boldsymbol{G}^k(\boldsymbol{x}, \boldsymbol{X})(\boldsymbol{\Phi}(\boldsymbol{X}, \boldsymbol{w}_k) - \boldsymbol{y})$$

where $\boldsymbol{G}^k(\boldsymbol{x}, \boldsymbol{X}) := \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \tilde{\boldsymbol{w}}_k)^\top \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{X}, \boldsymbol{w}_k)^\top$ for some $\tilde{\boldsymbol{w}}_k$ in the convex hull of $\boldsymbol{w}_k$ and $\boldsymbol{w}_{k+1}$.

**Linearized model:**

$$\Phi^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{p}_{k+1}) = \Phi^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{p}_k) - 2h \boldsymbol{G}^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{X})(\boldsymbol{\Phi}^{\mathrm{lin}}(\boldsymbol{X}, \boldsymbol{p}_k) - \boldsymbol{y})$$

where $\boldsymbol{G}^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{X}) = \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}_0)^\top \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{X}, \boldsymbol{w}_0)^\top$.

The full dynamics are governed by the $k$-dependent kernel matrices $\boldsymbol{G}^k$. In contrast, the linear model's dynamics are entirely determined by the initial kernel matrix $\boldsymbol{G}^{\mathrm{lin}}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 11.13</span><span class="math-callout__name">(Kernel Matrix Deviation)</span></p>

Let $\boldsymbol{w}_0 = \boldsymbol{p}_0 \in \mathbb{R}^n$, and let Assumption 11.11 be satisfied. Let $(\boldsymbol{w}_k)_{k \in \mathbb{N}}$, $(\boldsymbol{p}_k)_{k \in \mathbb{N}}$ be generated by gradient descent with a positive step size $h < (\theta_{\min} + \theta_{\max})^{-1}$. Then for all $\boldsymbol{x} \in \mathbb{R}^d$ with $\lVert \boldsymbol{x} \rVert \le R$:

$$\sup_{k \in \mathbb{N}} \lVert \boldsymbol{G}^k(\boldsymbol{x}, \boldsymbol{X}) - \boldsymbol{G}^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{X}) \rVert \le 2\sqrt{m} U L r$$

$$\sup_{k \in \mathbb{N}} \lVert \boldsymbol{G}^k(\boldsymbol{X}, \boldsymbol{X}) - \boldsymbol{G}^{\mathrm{lin}}(\boldsymbol{X}, \boldsymbol{X}) \rVert \le 2m U L r.$$

</div>

#### 11.5.2 Limiting Model Predictions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.14</span><span class="math-callout__name">(Proximity of Full and Linearized Models)</span></p>

Consider the setting of Corollary 11.13, in particular let $r$, $R$, $\theta_{\min}$, $\theta_{\max}$ be as in Assumption 11.11. Then for all $\boldsymbol{x} \in \mathbb{R}^d$ with $\lVert \boldsymbol{x} \rVert \le R$

$$\sup_{k \in \mathbb{N}} \lVert \Phi(\boldsymbol{x}, \boldsymbol{w}_k) - \Phi^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{p}_k) \rVert \le \frac{4\sqrt{m} U L r}{\theta_{\min}} \left(1 + \frac{mU^2}{(h\theta_{\min})^2(\theta_{\min} + \theta_{\max})}\right) \sqrt{f(\boldsymbol{w}_0)}.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.15</span><span class="math-callout__name">(Training Data Discrepancy)</span></p>

Consider the setting of Corollary 11.13 and set

$$\alpha := \frac{2mULr}{h\theta_{\min}(\theta_{\min} + \theta_{\max})} \sqrt{f(\boldsymbol{w}_0)}.$$

Then for all $k \in \mathbb{N}$

$$\lVert \boldsymbol{\Phi}(\boldsymbol{X}, \boldsymbol{w}_k) - \boldsymbol{\Phi}^{\mathrm{lin}}(\boldsymbol{X}, \boldsymbol{p}_k) \rVert \le \alpha k (1 - h\theta_{\min})^{k-1}.$$

</div>

### 11.6 Training Dynamics for Shallow Neural Networks

We now discuss the implications of Theorems 11.12 and 11.14 for wide neural networks. We focus on a shallow architecture with only one hidden layer.

#### 11.6.1 Architecture

Let $\Phi : \mathbb{R}^d \to \mathbb{R}$ be a neural network of depth one and width $n \in \mathbb{N}$ of type

$$\Phi(\boldsymbol{x}, \boldsymbol{w}) = \boldsymbol{v}^\top \sigma(\boldsymbol{U}\boldsymbol{x} + \boldsymbol{b}) + c.$$

Here $\boldsymbol{x} \in \mathbb{R}^d$ is the input, and $\boldsymbol{U} \in \mathbb{R}^{n \times d}$, $\boldsymbol{v} \in \mathbb{R}^n$, $\boldsymbol{b} \in \mathbb{R}^n$ and $c \in \mathbb{R}$ are the parameters collected in $\boldsymbol{w} = (\boldsymbol{U}, \boldsymbol{b}, \boldsymbol{v}, c) \in \mathbb{R}^{n(d+2)+1}$. The gradients are:

$$\nabla_{\boldsymbol{U}} \Phi(\boldsymbol{x}, \boldsymbol{w}) = (\boldsymbol{v} \odot \sigma'(\boldsymbol{U}\boldsymbol{x} + \boldsymbol{b}))\boldsymbol{x}^\top \in \mathbb{R}^{n \times d}$$

$$\nabla_{\boldsymbol{b}} \Phi(\boldsymbol{x}, \boldsymbol{w}) = \boldsymbol{v} \odot \sigma'(\boldsymbol{U}\boldsymbol{x} + \boldsymbol{b}) \in \mathbb{R}^n$$

$$\nabla_{\boldsymbol{v}} \Phi(\boldsymbol{x}, \boldsymbol{w}) = \sigma(\boldsymbol{U}\boldsymbol{x} + \boldsymbol{b}) \in \mathbb{R}^n$$

$$\nabla_c \Phi(\boldsymbol{x}, \boldsymbol{w}) = 1 \in \mathbb{R}$$

where $\odot$ denotes the Hadamard product.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption 11.16</span><span class="math-callout__name">(LeCun Initialization)</span></p>

The distribution $\mathcal{W}$ on $\mathbb{R}$ has expectation zero, variance one, and finite moments up to order eight. The initial parameters $\boldsymbol{w}_0 = (\boldsymbol{U}_0, \boldsymbol{b}_0, \boldsymbol{v}_0, c_0)$ are randomly initialized with components

$$U_{0;ij} \stackrel{\text{iid}}{\sim} \mathcal{W}\!\left(0, \tfrac{1}{d}\right), \qquad v_{0;i} \stackrel{\text{iid}}{\sim} \mathcal{W}\!\left(0, \tfrac{1}{n}\right), \qquad b_{0;i} = 0, \quad c_0 = 0$$

independently for all $i = 1, \ldots, n$, $j = 1, \ldots, d$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.17</span><span class="math-callout__name">(Typical Initializations)</span></p>

Typical examples for $\mathcal{W}(0,1)$ are the standard normal distribution on $\mathbb{R}$ or the uniform distribution on $[-\sqrt{3}, \sqrt{3}]$.

</div>

#### 11.6.2 Neural Tangent Kernel

The empirical tangent kernel of the shallow network is

$$\hat{K}_n(\boldsymbol{x}, \boldsymbol{z}) = \langle \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}_0), \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{z}, \boldsymbol{w}_0) \rangle.$$

Scaled properly, it converges in the infinite width limit $n \to \infty$ towards a specific kernel known as the **neural tangent kernel** (NTK).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.18</span><span class="math-callout__name">(Convergence to NTK)</span></p>

Let $R < \infty$ such that $\lvert \sigma(x) \rvert \le R \cdot (1 + \lvert x \rvert)$ and $\lvert \sigma'(x) \rvert \le R \cdot (1 + \lvert x \rvert)$ for all $x \in \mathbb{R}$. For any $\boldsymbol{x}, \boldsymbol{z} \in \mathbb{R}^d$ and $u_i \stackrel{\text{iid}}{\sim} \mathcal{W}(0, 1/d)$, $i = 1, \ldots, d$, it then holds

$$\lim_{n \to \infty} \frac{1}{n} \hat{K}_n(\boldsymbol{x}, \boldsymbol{z}) = \mathbb{E}[\sigma(\boldsymbol{u}^\top \boldsymbol{x}) \sigma(\boldsymbol{u}^\top \boldsymbol{z})] =: K^{\mathrm{NTK}}(\boldsymbol{x}, \boldsymbol{z})$$

almost surely. Moreover, for every $\delta$, $\varepsilon > 0$ there exists $n_0(\delta, \varepsilon, R) \in \mathbb{N}$ such that for all $n \ge n_0$ and all $\boldsymbol{x}$, $\boldsymbol{z} \in \mathbb{R}^d$ with $\lVert \boldsymbol{x} \rVert, \lVert \boldsymbol{z} \rVert \le R$

$$\mathbb{P}\!\left[\left\lVert \frac{1}{n} \hat{K}_n(\boldsymbol{x}, \boldsymbol{z}) - K^{\mathrm{NTK}}(\boldsymbol{x}, \boldsymbol{z}) \right\rVert < \varepsilon\right] \ge 1 - \delta.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 11.19</span><span class="math-callout__name">($K^{\mathrm{NTK}}$ for ReLU)</span></p>

Let $\sigma(x) = \max\lbrace 0, x \rbrace$ and let $\mathcal{W}(0,1)$ be the standard normal distribution. For $\boldsymbol{x}, \boldsymbol{z} \in \mathbb{R}^d$ denote by

$$\vartheta = \arccos\!\left(\frac{\boldsymbol{x}^\top \boldsymbol{z}}{\lVert \boldsymbol{x} \rVert \lVert \boldsymbol{z} \rVert}\right)$$

the angle between these vectors. Then

$$K^{\mathrm{NTK}}(\boldsymbol{x}, \boldsymbol{z}) = \mathbb{E}[\sigma(\boldsymbol{u}^\top \boldsymbol{x})\sigma(\boldsymbol{u}^\top \boldsymbol{z})] = \frac{\lVert \boldsymbol{x} \rVert \lVert \boldsymbol{z} \rVert}{2\pi d} (\sin(\vartheta) + (\pi - \vartheta)\cos(\vartheta)).$$

</div>

#### 11.6.3 Training Dynamics and Model Predictions

We now show that the analysis in Sections 11.4--11.5 is applicable to the wide neural network with high probability under random initialization.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption 11.20</span><span class="math-callout__name">(Assumptions on Activation and Data)</span></p>

There exist $1 \le R < \infty$ and $0 < \theta_{\min}^{\mathrm{NTK}} \le \theta_{\max}^{\mathrm{NTK}} < \infty$ such that:

**(a)** $\sigma : \mathbb{R} \to \mathbb{R}$ belongs to $C^1(\mathbb{R})$ and $\lvert \sigma(0) \rvert$, $\operatorname{Lip}(\sigma)$, $\operatorname{Lip}(\sigma') \le R$.

**(b)** $\lVert \boldsymbol{x}_i \rVert, \lvert y_i \rvert \le R$ for all training data $(\boldsymbol{x}_i, y_i) \in \mathbb{R}^d \times \mathbb{R}$, $i = 1, \ldots, m$.

**(c)** The kernel matrix of the neural tangent kernel $(K^{\mathrm{NTK}}(\boldsymbol{x}_i, \boldsymbol{x}_j))_{i,j=1}^m \in \mathbb{R}^{m \times m}$ is regular and its eigenvalues belong to $[\theta_{\min}^{\mathrm{NTK}}, \theta_{\max}^{\mathrm{NTK}}]$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 11.21</span><span class="math-callout__name">(Eigenvalue Bounds for Empirical Tangent Kernel)</span></p>

Let Assumption 11.20 be satisfied. Then for every $\delta > 0$ there exists $n_0(\delta, \theta_{\min}^{\mathrm{NTK}}, m, R) \in \mathbb{R}$ such that for all $n \ge n_0$ it holds with probability at least $1 - \delta$ that all eigenvalues of

$$(\hat{K}_n(\boldsymbol{x}_i, \boldsymbol{x}_j))_{i,j=1}^m = \big( \langle \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}_i, \boldsymbol{w}_0), \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}_j, \boldsymbol{w}_0) \rangle \big)_{i,j=1}^m \in \mathbb{R}^{m \times m}$$

belong to $[n\theta_{\min}^{\mathrm{NTK}}/2, \; 2n\theta_{\max}^{\mathrm{NTK}}]$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 11.22</span><span class="math-callout__name">(Random Matrix Norm Bound)</span></p>

Let $\mathcal{W}(0,1)$ be as in Assumption 11.16, and let $\boldsymbol{W} \in \mathbb{R}^{n \times d}$ with $W_{ij} \stackrel{\text{iid}}{\sim} \mathcal{W}(0,1)$. Denote the fourth moment of $\mathcal{W}(0,1)$ by $\mu_4$. Then

$$\mathbb{P}\!\left[\lVert \boldsymbol{W} \rVert \le \sqrt{n(d+1)}\right] \ge 1 - \frac{d\mu_4}{n}.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 11.23</span><span class="math-callout__name">(Gradient Bounds)</span></p>

Let Assumption 11.20 (a) be satisfied with some constant $R$. Then there exists $M(R) > 0$ such that for all $c$, $\delta > 0$ there exists $n_0(c, d, \delta) \in \mathbb{N}$ such that for all $n \ge n_0$ it holds with probability at least $1 - \delta$ that

$$\lVert \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}) \rVert \le M\sqrt{n} \qquad \text{for all } \boldsymbol{w} \in B_{cn^{-1/2}}(\boldsymbol{w}_0)$$

$$\lVert \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}) - \nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{v}) \rVert \le M\sqrt{n} \lVert \boldsymbol{w} - \boldsymbol{v} \rVert \qquad \text{for all } \boldsymbol{w}, \boldsymbol{v} \in B_{cn^{-1/2}}(\boldsymbol{w}_0)$$

for all $\boldsymbol{x} \in \mathbb{R}^d$ with $\lVert \boldsymbol{x} \rVert \le R$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 11.24</span><span class="math-callout__name">(Initial Error Bound)</span></p>

Let Assumption 11.20 (a), (b) be satisfied. Then for every $\delta > 0$ exists $R_0(\delta, m, R) > 0$ such that for all $n \in \mathbb{N}$

$$\mathbb{P}[f(\boldsymbol{w}_0) \le R_0] \ge 1 - \delta.$$

</div>

The following theorem is the main result of this section. It summarizes: with high probability, gradient descent converges to a global minimizer, the network weights remain close to initialization, and the trained network gives predictions that are $O(n^{-1/2})$ close to those of the trained linearized model.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11.25</span><span class="math-callout__name">(Main Result for Wide Networks)</span></p>

Let Assumption 11.20 be satisfied, and let the parameters $\boldsymbol{w}_0$ of the width-$n$ neural network $\Phi$ be initialized according to LeCun initialization. Fix the learning rate

$$h = \frac{1}{\theta_{\min}^{\mathrm{NTK}} + 4\theta_{\max}^{\mathrm{NTK}}} \cdot \frac{1}{n},$$

set $\boldsymbol{p}_0 := \boldsymbol{w}_0$ and let for all $k \in \mathbb{N}_0$

$$\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - h\nabla f(\boldsymbol{w}_k) \qquad \text{and} \qquad \boldsymbol{p}_{k+1} = \boldsymbol{p}_k - h\nabla f^{\mathrm{lin}}(\boldsymbol{p}_k).$$

Then for every $\delta > 0$ there exist $C > 0$, $n_0 \in \mathbb{N}$ such that for all $n \ge n_0$ it holds with probability at least $1 - \delta$ that for all $k \in \mathbb{N}$ and all $\boldsymbol{x} \in \mathbb{R}^d$ with $\lVert \boldsymbol{x} \rVert \le R$

$$\lVert \boldsymbol{w}_k - \boldsymbol{w}_0 \rVert \le \frac{C}{\sqrt{n}}$$

$$f(\boldsymbol{w}_k) \le C\!\left(1 - \frac{hn}{2\theta_{\min}^{\mathrm{NTK}}}\right)^{2k}$$

$$\lVert \Phi(\boldsymbol{x}, \boldsymbol{w}_k) - \Phi^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{p}_k) \rVert \le \frac{C}{\sqrt{n}}.$$

</div>

Note that the convergence rate does not improve as $n$ grows, since $h$ is bounded by a constant times $1/n$.

#### 11.6.4 Connection to Kernel Least-Squares and Gaussian Processes

Theorem 11.25 establishes that the trained neural network mirrors the behaviour of the trained linearized model. Since the prediction of the trained linearized model corresponds to the ridgeless kernel least-squares estimator plus a term depending on random initialization, we can understand both $\boldsymbol{x} \mapsto \Phi(\boldsymbol{x}, \boldsymbol{w}_0)$ and the model after training as random draws of a certain distribution over functions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 11.26</span><span class="math-callout__name">(Gaussian Process)</span></p>

Let $(\Omega, \mathfrak{A}, \mathbb{P})$ be a probability space, and let $g : \mathbb{R}^d \times \Omega \to \mathbb{R}$. We call $g$ a **Gaussian process** with mean function $\mu : \mathbb{R}^d \to \mathbb{R}$ and covariance function $c : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ if

**(a)** for each $\boldsymbol{x} \in \mathbb{R}^d$ it holds that $\omega \mapsto g(\boldsymbol{x}, \omega)$ is a random variable,

**(b)** for all $k \in \mathbb{N}$ and all $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_k \in \mathbb{R}^d$ the random variables $g(\boldsymbol{x}_1, \cdot), \ldots, g(\boldsymbol{x}_k, \cdot)$ are jointly Gaussian distributed with

$$(g(\boldsymbol{x}_1, \omega), \ldots, g(\boldsymbol{x}_k, \omega)) \sim \mathrm{N}\!\left((\mu(\boldsymbol{x}_i))_{i=1}^k, \; (c(\boldsymbol{x}_i, \boldsymbol{x}_j))_{i,j=1}^k\right).$$

</div>

Fixing $\omega \in \Omega$, we can interpret $\boldsymbol{x} \mapsto g(\boldsymbol{x}, \omega)$ as a random draw from a distribution over functions.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.27</span><span class="math-callout__name">(Neural Networks Converge to Gaussian Processes)</span></p>

Let $\lvert \sigma(x) \rvert \le R(1 + \lvert x \rvert)^4$ for all $x \in \mathbb{R}$. Consider depth-$n$ networks $\Phi$ as above with LeCun initialization. Let $K^{\mathrm{NTK}}$ be as in Theorem 11.18. Then for all distinct $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_k \in \mathbb{R}^d$ it holds that

$$\lim_{n \to \infty} (\Phi(\boldsymbol{x}_1, \boldsymbol{w}_0), \ldots, \Phi(\boldsymbol{x}_k, \boldsymbol{w}_0)) \sim \mathrm{N}\!\left(\boldsymbol{0}, \; (K^{\mathrm{NTK}}(\boldsymbol{x}_i, \boldsymbol{x}_j))_{i,j=1}^k\right)$$

with convergence in distribution.

</div>

Since the full and linearized models coincide in the infinite width limit (Theorem 11.25), we can infer that wide networks post-training resemble draws from a Gaussian process. In particular, after sufficient training, the mean (over random initializations) of the trained neural network $\boldsymbol{x} \mapsto \Phi(\boldsymbol{x}, \boldsymbol{w}_k)$ resembles the kernel least-squares estimator with kernel $K^{\mathrm{NTK}}$:

$$\mathbb{E}\big[\Phi(\boldsymbol{x}, \boldsymbol{w}_k)\big] \simeq \mathbb{E}\big[\Phi^{\mathrm{lin}}(\boldsymbol{x}, \boldsymbol{p}_k)\big] \simeq \text{ridgeless kernel least-squares estimator with kernel } K^{\mathrm{NTK}} \text{ evaluated at } \boldsymbol{x}.$$

#### 11.6.5 Role of Initialization

Consider the gradient $\nabla_{\boldsymbol{w}} \Phi(\boldsymbol{x}, \boldsymbol{w}_0)$ with LeCun initialization. The expected squared norms in terms of the width $n$ are:

$$\mathbb{E}[\lVert \nabla_{\boldsymbol{U}} \Phi(\boldsymbol{x}, \boldsymbol{w}_0) \rVert^2] = O(1), \qquad \mathbb{E}[\lVert \nabla_{\boldsymbol{b}} \Phi(\boldsymbol{x}, \boldsymbol{w}_0) \rVert^2] = O(1),$$

$$\mathbb{E}[\lVert \nabla_{\boldsymbol{v}} \Phi(\boldsymbol{x}, \boldsymbol{w}_0) \rVert^2] = O(n), \qquad \mathbb{E}[\lVert \nabla_c \Phi(\boldsymbol{x}, \boldsymbol{w}_0) \rVert^2] = O(1).$$

Due to this different scaling, gradient descent with step size $O(n^{-1})$ will primarily adjust the weights $\boldsymbol{v}$ in the output layer, while only slightly modifying the remaining parameters $\boldsymbol{U}$, $\boldsymbol{b}$, and $c$. This is reflected in the expression for the NTK computed in Theorem 11.18, which corresponds only to the contribution of $\langle \nabla_{\boldsymbol{v}} \Phi, \nabla_{\boldsymbol{v}} \Phi \rangle$.

LeCun initialization sets the variance of the weight initialization inversely proportional to the input dimension of each layer, so that the variance of all node outputs remains stable and does not blow up as the width increases. However, it does not normalize the backward dynamics, i.e., it does not ensure that the gradients with respect to the parameters have similar variance. To balance both forward and backward dynamics, Glorot and Bengio proposed a normalized initialization, where the variance is chosen inversely proportional to the sum of the input and output dimensions of each layer.

---

## Chapter 12: Loss Landscape Analysis

In Chapter 10, we saw how the weights of neural networks get adapted during training, using variants of gradient descent. For certain cases, including the wide networks considered in Chapter 11, the corresponding iterative scheme converges to a global minimizer. In general, this is not guaranteed, and gradient descent can get stuck in non-global minima or saddle points.

To get a better understanding of these situations, we discuss the so-called **loss landscape**. This term refers to the graph of the empirical risk as a function of the weights. The loss landscape is a high-dimensional surface, with hills and valleys.

Questions of interest include: How likely is it that we find local instead of global minima? Are these local minima typically sharp, having small volume, or are they part of large flat valleys that are difficult to escape? Are most local minima as deep as the global minimum, or can they be significantly higher? How do these characteristics depend on the network architecture?

### Notation and Definitions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.1</span><span class="math-callout__name">(Parameter Space and Realization Map)</span></p>

Let $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$, let $\sigma : \mathbb{R} \to \mathbb{R}$ be an activation function, and let $B > 0$. We denote the set of neural networks $\Phi$ with $L$ layers, layer widths $d_0, d_1, \ldots, d_{L+1}$, all weights bounded in modulus by $B$, and using the activation function $\sigma$ by $\mathcal{N}(\sigma; \mathcal{A}, B)$. Additionally, we define

$$\mathcal{PN}(\mathcal{A}, B) := \bigtimes_{\ell=0}^L \left( [-B, B]^{d_{\ell+1} \times d_\ell} \times [-B, B]^{d_{\ell+1}} \right),$$

and the **realization map**

$$R_\sigma : \mathcal{PN}(\mathcal{A}, B) \to \mathcal{N}(\sigma; \mathcal{A}, B), \qquad (\boldsymbol{W}^{(\ell)}, \boldsymbol{b}^{(\ell)})_{\ell=0}^L \mapsto \Phi,$$

where $\Phi$ is the neural network with weights and biases given by $(\boldsymbol{W}^{(\ell)}, \boldsymbol{b}^{(\ell)})_{\ell=0}^L$.

</div>

We identify $\mathcal{PN}(\mathcal{A}, B)$ with the cube $[-B, B]^{n_\mathcal{A}}$, where $n_\mathcal{A} := \sum_{\ell=0}^L d_{\ell+1}(d_\ell + 1)$ is the total number of parameters.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.2</span><span class="math-callout__name">(Loss Landscape)</span></p>

Let $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$, let $\sigma : \mathbb{R} \to \mathbb{R}$. Let $m \in \mathbb{N}$, and $S = (\boldsymbol{x}_i, \boldsymbol{y}_i)_{i=1}^m \in (\mathbb{R}^{d_0} \times \mathbb{R}^{d_{L+1}})^m$ be a sample and let $\mathcal{L}$ be a loss function. Then, the **loss landscape** is the graph of the function $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}$ defined as

$$\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}} : \mathcal{PN}(\mathcal{A}; \infty) \to \mathbb{R}, \qquad \theta \mapsto \widehat{\mathcal{R}}_S(R_\sigma(\theta)).$$

</div>

Identifying $\mathcal{PN}(\mathcal{A}, \infty)$ with $\mathbb{R}^{n_\mathcal{A}}$, we can consider $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}$ as a map on $\mathbb{R}^{n_\mathcal{A}}$ and the loss landscape is a subset of $\mathbb{R}^{n_\mathcal{A}} \times \mathbb{R}$.

### 12.1 Visualization of Loss Landscapes

Visualizing loss landscapes can provide valuable insights into the effects of neural network depth, width, and activation functions. Since the loss landscape is a very high-dimensional object, we reduce its dimensionality by evaluating $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}$ on a two-dimensional subspace. Specifically, we choose three parameters $\mu$, $\theta_1$, $\theta_2$ and examine the function

$$\mathbb{R}^2 \ni (\alpha_1, \alpha_2) \mapsto \Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}(\mu + \alpha_1 \theta_1 + \alpha_2 \theta_2).$$

There are various natural choices for $\mu$, $\theta_1$, $\theta_2$:

- **Random directions:** $\theta_1$, $\theta_2$ are chosen randomly, while $\mu$ is either a minimum of $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}$ or also chosen randomly. This simple approach can offer a quick insight into how rough the surface can be. However, random directions will very likely be orthogonal to the trajectory of the optimization procedure, and hence will likely miss the most relevant features.

- **Principal components of learning trajectory:** If $\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(N)}$ are the parameters resulting from training by SGD, we may determine $\mu$, $\theta_1$, $\theta_2$ such that the hyperplane $\lbrace \mu + \alpha_1 \theta_1 + \alpha_2 \theta_2 \mid \alpha_1, \alpha_2 \in \mathbb{R} \rbrace$ minimizes the mean squared distance to the $\theta^{(j)}$. This is achieved by a principal component analysis.

- **Based on critical points:** $\mu$, $\theta_1$, $\theta_2$ can be chosen to ensure the observation of multiple critical points. Running the optimization procedure three times with final parameters $\theta^{(1)}$, $\theta^{(2)}$, $\theta^{(3)}$, we can set $\mu = \theta^{(1)}$, $\theta_1 = \theta^{(2)} - \mu$, $\theta_2 = \theta^{(3)} - \mu$.

For very wide and shallow neural networks, the widest minima also seem to belong to the same valley. With increasing depth and smaller width the minima get steeper and more disconnected.

### 12.2 Spurious Valleys

From the perspective of optimization, the ideal loss landscape has one global minimum in the center of a large valley, so that gradient descent converges towards the minimum irrespective of the chosen initialization.

This situation is not realistic for deep neural networks. Indeed, for a simple shallow neural network $\Phi(\boldsymbol{x}) = \boldsymbol{W}^{(1)} \sigma(\boldsymbol{W}^{(0)} \boldsymbol{x} + \boldsymbol{b}^{(0)}) + \boldsymbol{b}^{(1)}$, it is clear that for every permutation matrix $\boldsymbol{P}$

$$\Phi(\boldsymbol{x}) = \boldsymbol{W}^{(1)} \boldsymbol{P}^T \sigma(\boldsymbol{P}\boldsymbol{W}^{(0)} \boldsymbol{x} + \boldsymbol{P}\boldsymbol{b}^{(0)}) + \boldsymbol{b}^{(1)} \qquad \text{for all } \boldsymbol{x} \in \mathbb{R}^d.$$

Hence, in general there exist multiple parameterizations realizing the same output function. If at least one global minimum with non-permutation-invariant weights exists, then there are more than one global minima of the loss landscape. This is not problematic; having many global minima is beneficial. The larger issue is the existence of non-global minima.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12.3</span><span class="math-callout__name">(Spurious Valley)</span></p>

Let $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$ and $\sigma : \mathbb{R} \to \mathbb{R}$. Let $m \in \mathbb{N}$, and $S = (\boldsymbol{x}_i, \boldsymbol{y}_i)_{i=1}^m \in (\mathbb{R}^{d_0} \times \mathbb{R}^{d_{L+1}})^m$ be a sample and let $\mathcal{L}$ be a loss function. For $c \in \mathbb{R}$, we define the sub-level set of $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}$ as

$$\Omega_\Lambda(c) := \lbrace \theta \in \mathcal{PN}(\mathcal{A}, \infty) \mid \Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}(\theta) \le c \rbrace.$$

A path-connected component of $\Omega_\Lambda(c)$, which does not contain a global minimum of $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}$, is called a **spurious valley**.

</div>

The next proposition shows that spurious local minima do not exist for shallow overparameterized neural networks, i.e., for neural networks that have at least as many parameters in the hidden layer as there are training samples.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12.4</span><span class="math-callout__name">(No Spurious Valleys for Overparameterized Shallow Networks)</span></p>

Let $\mathcal{A} = (d_0, d_1, 1) \in \mathbb{N}^3$ and let $S = (\boldsymbol{x}_i, y_i)_{i=1}^m \in (\mathbb{R}^{d_0} \times \mathbb{R})^m$ be a sample such that $m \le d_1$. Furthermore, let $\sigma \in \mathcal{M}$ be not a polynomial, and let $\mathcal{L}$ be a convex loss function. Further assume that $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}$ has at least one global minimum. Then, $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}$ has no spurious valleys.

</div>

The proof proceeds by showing that for any $\theta_a$ with $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}(\theta_a) > \Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}(\theta_b)$, there exists a continuous path $\alpha$ from $\theta_a$ to some $\theta_c$ such that $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}$ is monotonically decreasing along the path and $\Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}(\theta_b) \ge \Lambda_{\mathcal{A}, \sigma, S, \mathcal{L}}(\theta_c)$. The key insight is that if the matrix of hidden layer activations at the training points does not have full rank $m$, one can always find a path to a parameter with higher rank while monotonically decreasing the loss (using that $\sigma$ is not a polynomial and Theorem 9.3 on interpolation). Once the activation matrix has full rank, one can directly construct a monotonically decreasing straight path to a point with lower or equal loss.

### 12.3 Saddle Points

Saddle points are critical points of the loss landscape at which the loss decreases in one direction. In this sense, saddle points are not as problematic as local minima or spurious valleys if the updates in the learning iteration have some stochasticity. Eventually, a random step in the right direction could be taken and the saddle point can be escaped.

If most of the critical points are saddle points, then, even though the loss landscape is challenging for optimization, one still has a good chance of eventually reaching the global minimum. The main observation is that, under some quite strong assumptions, *critical points in the loss landscape associated to a large loss are typically saddle points, whereas those associated to small loss correspond to minima*. This is encouraging for the prospects of optimization in deep learning, since even if we get stuck in a local minimum, it will very likely be such that the loss is close to optimal.

Let $\mathcal{A} = (d_0, d_1, 1) \in \mathbb{N}^3$. For a neural network parameter $\theta \in \mathcal{PN}(\mathcal{A}, \infty)$ and activation function $\sigma$, we set $\Phi_\theta := R_\sigma(\theta)$ and define for a sample $S = (\boldsymbol{x}_i, y_i)_{i=1}^m$ the errors

$$e_i = \Phi_\theta(\boldsymbol{x}_i) - y_i \qquad \text{for } i = 1, \ldots, m.$$

If we use the square loss, then

$$\widehat{\mathcal{R}}_S(\Phi_\theta) = \frac{1}{m} \sum_{i=1}^m e_i^2.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12.5</span><span class="math-callout__name">(Hessian Decomposition)</span></p>

Let $\mathcal{A} = (d_0, d_1, 1)$ and $\sigma : \mathbb{R} \to \mathbb{R}$. Then, for every $\theta \in \mathcal{PN}(\mathcal{A}, \infty)$ where $\widehat{\mathcal{R}}_S(\Phi_\theta)$ is twice continuously differentiable with respect to the weights, it holds that

$$\boldsymbol{H}(\theta) = \boldsymbol{H}_0(\theta) + \boldsymbol{H}_1(\theta),$$

where $\boldsymbol{H}(\theta)$ is the Hessian of $\widehat{\mathcal{R}}_S(\Phi_\theta)$ at $\theta$, $\boldsymbol{H}_0(\theta)$ is a positive semi-definite matrix which is independent from $(y_i)_{i=1}^m$, and $\boldsymbol{H}_1(\theta)$ is a symmetric matrix that for fixed $\theta$ and $(\boldsymbol{x}_i)_{i=1}^m$ depends linearly on $(e_i)_{i=1}^m$.

</div>

The Hessian decomposes as

$$\frac{\partial^2 \widehat{\mathcal{R}}_S(\Phi_\theta)}{\partial \theta_j \partial \theta_k} = \frac{2}{m} \sum_{i=1}^m \left( \frac{\partial \Phi_\theta(\boldsymbol{x}_i)}{\partial \theta_j} \frac{\partial \Phi_\theta(\boldsymbol{x}_i)}{\partial \theta_k} \right) + \frac{2}{m} \left( \sum_{i=1}^m e_i \frac{\partial^2 \Phi_\theta(\boldsymbol{x}_i)}{\partial \theta_j \partial \theta_k} \right) =: \boldsymbol{H}_0(\theta) + \boldsymbol{H}_1(\theta).$$

The term $\boldsymbol{H}_0(\theta) = \frac{2}{m} \sum_{i=1}^m J_{i,\theta} J_{i,\theta}^\top$ is a sum of positive semi-definite matrices (where $J_{i,\theta}$ is the Jacobian of $\Phi_\theta(\boldsymbol{x}_i)$ with respect to $\theta$), and hence is positive semi-definite. The term $\boldsymbol{H}_1(\theta)$ is symmetric and depends linearly on the errors $(e_i)_{i=1}^m$.

**How does this relate to saddle points?** Let $\theta$ correspond to a critical point. If $\boldsymbol{H}(\theta)$ has at least one negative eigenvalue, then $\theta$ cannot be a minimum, but instead must be either a saddle point or a maximum.

Consider a fixed parameter $\theta$. Let $S^0 = (\boldsymbol{x}_i, y_i^0)_{i=1}^m$ be a sample with associated errors $(e_i^0)_{i=1}^m$. Further let for $\lambda > 0$, $S^\lambda = (\boldsymbol{x}_i, y_i^\lambda)_{i=1}^m$ be such that the associated errors are $(e_i)_{i=1}^m = \lambda (e_i^0)_{i=1}^m$. The Hessian of $\widehat{\mathcal{R}}_{S^\lambda}(\Phi_\theta)$ at $\theta$ is then $\boldsymbol{H}^\lambda(\theta)$ satisfying

$$\boldsymbol{H}^\lambda(\theta) = \boldsymbol{H}_0^0(\theta) + \lambda \boldsymbol{H}_1^0(\theta).$$

Hence, if $\lambda$ is large, then $\boldsymbol{H}^\lambda(\theta)$ is a perturbation of an amplified version of $\boldsymbol{H}_1^0(\theta)$. If $\boldsymbol{v}$ is an eigenvector of $\boldsymbol{H}_1^0(\theta)$ with negative eigenvalue $-\mu$, then

$$\boldsymbol{v}^\top \boldsymbol{H}^\lambda(\theta) \boldsymbol{v} \le (\lVert \boldsymbol{H}_0^0(\theta) \rVert - \lambda \mu) \lVert \boldsymbol{v} \rVert^2,$$

which we can expect to be negative for large $\lambda$. Thus, $\boldsymbol{H}^\lambda(\theta)$ has a negative eigenvalue for large $\lambda$.

On the other hand, if $\lambda$ is small, then $\boldsymbol{H}^\lambda(\theta)$ is merely a perturbation of $\boldsymbol{H}_0^0(\theta)$ and we can expect its spectrum to resemble that of $\boldsymbol{H}_0^0$ more and more.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation)</span></p>

What we see is that, the same parameter is more likely to be a saddle point for a sample that produces a high empirical risk than for a sample with small risk. Note that, since $\boldsymbol{H}_0^0(\theta)$ was only shown to be *semi*-definite, the argument above does not rule out saddle points even for very small $\lambda$. But it does show that for small $\lambda$, every negative eigenvalue would be very small.

</div>

---

## Chapter 13: Shape of Neural Network Spaces

As we have seen in the previous chapter, the loss landscape of neural networks can be quite intricate and is typically not convex. The reason for this is that we take the point of view of a map from the parameterization of a neural network. Consider a convex loss function $\mathcal{L} : \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ and a sample $S = (\boldsymbol{x}_i, y_i)_{i=1}^m \in (\mathbb{R}^d \times \mathbb{R})^m$. Then, for two neural networks $\Phi_1, \Phi_2$ and for $\alpha \in (0, 1)$ it holds that

$$\widehat{\mathcal{R}}_S(\alpha \Phi_1 + (1 - \alpha)\Phi_2) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(\alpha \Phi_1(\boldsymbol{x}_i) + (1 - \alpha)\Phi_2(\boldsymbol{x}_i), y_i) \leq \alpha \widehat{\mathcal{R}}_S(\Phi_1) + (1 - \alpha)\widehat{\mathcal{R}}_S(\Phi_2).$$

Hence, the empirical risk is convex when considered as a map depending on the neural network *functions* rather than the neural network *parameters*. A convex function does not have spurious minima or saddle points, so the issues from the previous section are avoided if we take the perspective of neural network sets.

So why do we not optimize over the sets of neural networks instead of the parameters? To understand this, we now study the set of neural networks associated with a fixed architecture as a subset of other function spaces.

We start by investigating the realization map $R_\sigma$ introduced in Definition 12.1. Concretely, we show in Section 13.1 that if $\sigma$ is Lipschitz, then the set of neural networks is the image of $\mathcal{PN}(\mathcal{A}, \infty)$ under a locally Lipschitz map. We will use this fact in Section 13.2 to show that sets of neural networks are typically non-convex, and even have arbitrarily large holes. Finally, in Section 13.3, we study the extent to which there exist best approximations to arbitrary functions in the set of neural networks, and demonstrate that the lack of best approximations causes the weights of neural networks to grow infinitely during training.

### 13.1 Lipschitz Parameterizations

In this section, we study the realization map $R_\sigma$. The main result is the following simplified version of [207, Proposition 4].

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13.1</span><span class="math-callout__name">(Lipschitz Continuity of the Realization Map)</span></p>

Let $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$, let $\sigma : \mathbb{R} \to \mathbb{R}$ be $C_\sigma$-Lipschitz continuous with $C_\sigma \geq 1$, let $\lvert \sigma(x) \rvert \leq C_\sigma \lvert x \rvert$ for all $x \in \mathbb{R}$, and let $B \geq 1$. Then, for all $\theta, \theta' \in \mathcal{PN}(\mathcal{A}, B)$,

$$\lVert R_\sigma(\theta) - R_\sigma(\theta') \rVert_{L^\infty([-1,1]^{d_0})} \leq (2C_\sigma B d_{\max})^L n_\mathcal{A} \lVert \theta - \theta' \rVert_\infty,$$

where $d_{\max} = \max_{\ell = 0, \ldots, L+1} d_\ell$ and $n_\mathcal{A} = \sum_{\ell=0}^L d_{\ell+1}(d_\ell + 1)$.

</div>

*Proof.* Let $\theta, \theta' \in \mathcal{PN}(\mathcal{A}, B)$ and define $\delta := \lVert \theta - \theta' \rVert_\infty$. Repeatedly using the triangle inequality we find a sequence $(\theta_j)_{j=0}^{n_\mathcal{A}}$ such that $\theta_0 = \theta$, $\theta_{n_\mathcal{A}} = \theta'$, $\lVert \theta_j - \theta_{j+1} \rVert_\infty \leq \delta$, and $\theta_j$ and $\theta_{j+1}$ differ in one entry only for all $j = 0, \ldots n_\mathcal{A} - 1$. We conclude that for all $\boldsymbol{x} \in [-1,1]^{d_0}$

$$\lVert R_\sigma(\theta)(\boldsymbol{x}) - R_\sigma(\theta')(\boldsymbol{x}) \rVert_\infty \leq \sum_{j=0}^{n_\mathcal{A} - 1} \lVert R_\sigma(\theta_j)(\boldsymbol{x}) - R_\sigma(\theta_{j+1})(\boldsymbol{x}) \rVert_\infty.$$

To upper bound this, we now only need to understand the effect of changing one weight in a neural network by $\delta$. The proof uses two auxiliary lemmas.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 13.2</span></p>

Under the assumptions of Proposition 13.1, but with $B$ being allowed to be arbitrary positive, it holds for all $\Phi \in \mathcal{N}(\sigma; \mathcal{A}, B)$

$$\lVert \Phi(\boldsymbol{x}) - \Phi(\boldsymbol{x}') \rVert_\infty \leq C_\sigma^L \cdot (B d_{\max})^{L+1} \lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_\infty$$

for all $\boldsymbol{x}, \boldsymbol{x}' \in \mathbb{R}^{d_0}$.

</div>

*Proof.* For $L = 1$, we have $\Phi(\boldsymbol{x}) = \mathbf{W}^{(1)} \sigma(\mathbf{W}^{(0)} \boldsymbol{x} + \mathbf{b}^{(0)}) + \mathbf{b}^{(1)}$ with all entries bounded by $B$. Then

$$\lVert \Phi(\boldsymbol{x}) - \Phi(\boldsymbol{x}') \rVert_\infty \leq d_1 B \cdot C_\sigma \lVert \mathbf{W}^{(0)}(\boldsymbol{x} - \boldsymbol{x}') \rVert_\infty \leq C_\sigma \cdot (d_{\max} B)^2 \lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_\infty,$$

using the Lipschitz property of $\sigma$ and the fact that $\lVert \mathbf{A}\boldsymbol{x} \rVert_\infty \leq n \max_{i,j} \lvert A_{ij} \rvert \lVert \boldsymbol{x} \rVert_\infty$ for every matrix $\mathbf{A} = (A_{ij})_{i=1,j=1}^{m,n} \in \mathbb{R}^{m \times n}$. The induction step from $L$ to $L+1$ follows similarly. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 13.3</span></p>

Under the assumptions of Proposition 13.1 it holds that

$$\lVert \boldsymbol{x}^{(\ell)} \rVert_\infty \leq (2C_\sigma B d_{\max})^\ell \quad \text{for all } \boldsymbol{x} \in [-1,1]^{d_0}.$$

</div>

*Proof.* Per the definitions of a neural network, we have that for $\ell = 1, \ldots, L + 1$

$$\lVert \boldsymbol{x}^{(\ell)} \rVert_\infty \leq C_\sigma B d_{\max} \lVert \boldsymbol{x}^{(\ell-1)} \rVert_\infty + B C_\sigma \leq C_\sigma B d_{\max} \cdot (1 + \lVert \boldsymbol{x}^{(\ell-1)} \rVert_\infty) \leq 2C_\sigma B d_{\max} \cdot \max\lbrace 1, \lVert \boldsymbol{x}^{(\ell-1)} \rVert_\infty \rbrace.$$

Resolving the recursive estimate of $\lVert \boldsymbol{x}^{(\ell)} \rVert_\infty$ by $2C_\sigma B d_{\max} (\max\lbrace 1, \lVert \boldsymbol{x}^{(\ell-1)} \rVert_\infty \rbrace)$, we conclude that

$$\lVert \boldsymbol{x}^{(\ell)} \rVert_\infty \leq (2C_\sigma B d_{\max})^\ell \max\lbrace 1, \lVert \boldsymbol{x}^{(0)} \rVert_\infty \rbrace = (2C_\sigma B d_{\max})^\ell. \quad \square$$

We can now proceed with the proof of Proposition 13.1. Assume that $\theta_{j+1}$ and $\theta_j$ differ only in one entry. If this entry is in the $\ell$-th layer, with $\ell < L$, it holds

$$\lvert R_\sigma(\theta_j)(\boldsymbol{x}) - R_\sigma(\theta_{j+1})(\boldsymbol{x}) \rvert \leq C_\sigma^{L-\ell-1} (B d_{\max})^{L-\ell} \lVert \sigma(\mathbf{W}^{(\ell)} \boldsymbol{x}^{(\ell)} + \mathbf{b}^{(\ell)}) - \sigma(\overline{\mathbf{W}}^{(\ell)} \boldsymbol{x}^{(\ell)} + \overline{\mathbf{b}}^{(\ell)}) \rVert_\infty$$

where $(\mathbf{W}^{(\ell)}, \mathbf{b}^{(\ell)})$ and $(\overline{\mathbf{W}}^{(\ell)}, \overline{\mathbf{b}}^{(\ell)})$ differ in one entry only. Using the Lipschitz continuity of $\Phi^\ell$ from Lemma 13.2, and invoking Lemma 13.3, we conclude that

$$\lvert R_\sigma(\theta_j)(\boldsymbol{x}) - R_\sigma(\theta_{j+1})(\boldsymbol{x}) \rvert \leq (2C_\sigma B d_{\max})^L \lVert \theta - \theta' \rVert_\infty.$$

For the case $\ell = L$, a similar estimate can be shown. Combining this with the telescoping sum yields the result. $\square$

Using Proposition 13.1, we can now consider the set of neural networks with a fixed architecture $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ as a subset of $L^\infty([-1,1]^{d_0})$. What is more, $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is the image of $\mathcal{PN}(\mathcal{A}, \infty)$ under a **locally Lipschitz map**.

### 13.2 Convexity of Neural Network Spaces

As a first step towards understanding $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ as a subset of $L^\infty([-1,1]^{d_0})$, we notice that it is star-shaped with few centers. Let us first introduce the necessary terminology.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 13.4</span><span class="math-callout__name">(Center, Star-Shaped)</span></p>

Let $Z$ be a subset of a linear space. A point $x \in Z$ is called a **center** of $Z$ if, for every $y \in Z$ it holds that

$$\lbrace tx + (1 - t)y \mid t \in [0, 1] \rbrace \subseteq Z.$$

A set is called **star-shaped** if it has at least one center.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13.5</span><span class="math-callout__name">(Scaling Invariance)</span></p>

Let $L \in \mathbb{N}$ and $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$ and $\sigma : \mathbb{R} \to \mathbb{R}$. Then $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is scaling invariant, i.e. for every $\lambda \in \mathbb{R}$ it holds that $\lambda f \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ if $f \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$, and hence $0 \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ is a center of $\mathcal{N}(\sigma; \mathcal{A}, \infty)$.

</div>

Knowing that $\mathcal{N}(\sigma; \mathcal{A}, B)$ is star-shaped with center 0, we can also ask whether $\mathcal{N}(\sigma; \mathcal{A}, B)$ has more than this one center. It is not hard to see that also every constant function is a center. The following theorem, which corresponds to [207, Proposition C.4], yields an upper bound on the number of *linearly independent* centers.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13.6</span><span class="math-callout__name">(Bound on Linearly Independent Centers)</span></p>

Let $L \in \mathbb{N}$ and $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$, and let $\sigma : \mathbb{R} \to \mathbb{R}$ be Lipschitz continuous. Then, $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ contains at most $n_\mathcal{A} = \sum_{\ell=0}^L (d_\ell + 1) d_{\ell+1}$ linearly independent centers.

</div>

*Proof.* Assume by contradiction that there are functions $(g_i)_{i=1}^{n_\mathcal{A}+1} \subseteq \mathcal{N}(\sigma; \mathcal{A}, \infty) \subseteq L^\infty([-1,1]^{d_0})$ that are linearly independent and centers of $\mathcal{N}(\sigma; \mathcal{A}, \infty)$.

By the Theorem of Hahn-Banach, there exist $(g_i')_{i=1}^{n_\mathcal{A}+1} \subseteq (L^\infty([-1,1]^{d_0}))'$ such that $g_i'(g_j) = \delta_{ij}$ for all $i, j \in \lbrace 1, \ldots, L + 1 \rbrace$. We define

$$T : L^\infty([-1,1]^{d_0}) \to \mathbb{R}^{n_\mathcal{A}+1}, \quad g \mapsto \begin{pmatrix} g_1'(g) \\ g_2'(g) \\ \vdots \\ g_{n_\mathcal{A}+1}'(g) \end{pmatrix}.$$

Since $T$ is continuous and linear, we have that $T \circ R_\sigma$ is locally Lipschitz continuous by Proposition 13.1. Moreover, since the $(g_i)_{i=1}^{n_\mathcal{A}+1}$ are linearly independent, we have that $T(\text{span}((g_i)_{i=1}^{n_\mathcal{A}+1})) = \mathbb{R}^{n_\mathcal{A}+1}$.

Next, we establish that $\mathcal{N}(\sigma; \mathcal{A}, \infty) \supset V$ where $V := \text{span}((g_i)_{i=1}^{n_\mathcal{A}+1})$. Let $g \in V$, then $g = \sum_{\ell=1}^{n_\mathcal{A}+1} a_\ell g_\ell$ for some $a_1, \ldots, a_{n_\mathcal{A}+1} \in \mathbb{R}$. We show by induction that $\widetilde{g}^{(m)} := \sum_{\ell=1}^m a_\ell g_\ell \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ for every $m \leq n_\mathcal{A} + 1$. If $a_{m+1} \neq 0$, then

$$\widetilde{g}^{(m+1)} = 2a_{m+1} \cdot \left( \frac{1}{2} g_{m+1} + \frac{1}{2a_{m+1}} \widetilde{g}^{(m)} \right).$$

By the induction assumption $\widetilde{g}^{(m)} \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ and by Proposition 13.5 $\widetilde{g}^{(m)} / (a_{m+1}) \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$. Additionally, since $g_{m+1}$ is a center of $\mathcal{N}(\sigma; \mathcal{A}, \infty)$, we have $\frac{1}{2} g_{m+1} + \frac{1}{2a_{m+1}} \widetilde{g}^{(m)} \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$. By Proposition 13.5, we conclude that $\widetilde{g}^{(m+1)} \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$.

The induction shows that $g \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ and thus $V \subseteq \mathcal{N}(\sigma; \mathcal{A}, \infty)$. As a consequence, $T \circ R_\sigma(\mathcal{PN}(\mathcal{A}, \infty)) \supseteq T(V) = \mathbb{R}^{n_\mathcal{A}+1}$.

It is a well-known fact of basic analysis that for every $n \in \mathbb{N}$ there does not exist a surjective and locally Lipschitz continuous map from $\mathbb{R}^n$ to $\mathbb{R}^{n+1}$. We recall that $n_\mathcal{A} = \dim(\mathcal{PN}(\mathcal{A}, \infty))$. This yields the contradiction. $\square$

For a convex set $X$, the line between all two points of $X$ is a subset of $X$. Hence, every point of a convex set is a center. This yields the following corollary.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 13.7</span><span class="math-callout__name">(Non-Convexity of Neural Network Spaces)</span></p>

Let $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1})$, let, and let $\sigma : \mathbb{R} \to \mathbb{R}$ be Lipschitz continuous. If $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ contains more than $n_\mathcal{A} = \sum_{\ell=0}^L (d_\ell + 1) d_{\ell+1}$ linearly independent functions, then $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is **not convex**.

</div>

Corollary 13.7 tells us that we cannot expect convex sets of neural networks if the set of neural networks has many linearly independent elements. Sets of neural networks contain for each $f \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ also all shifts of this function, i.e. $f(\cdot + \boldsymbol{b})$ for a $\boldsymbol{b} \in \mathbb{R}^d$ are elements of $\mathcal{N}(\sigma; \mathcal{A}, \infty)$. For a set of functions being shift invariant and having only finitely many linearly independent functions is a very restrictive condition. Indeed, it was shown in [207, Proposition C.6] that if $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ has only finitely many linearly independent functions and $\sigma$ is differentiable at at least one point and has non-zero derivative there, then $\sigma$ is necessarily a polynomial.

We conclude that the set of neural networks is in general non-convex and star-shaped with $0$ and constant functions being centers.

The fact that the neural network space is not convex could also mean that it merely fails to be convex at one point. We will next observe that $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ does not have such a benign non-convexity and in fact has *arbitrarily large holes*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 13.8</span><span class="math-callout__name">($\varepsilon$-Convexity)</span></p>

For $\varepsilon > 0$, we say that a subset $A$ of a normed vector space $X$ is $\varepsilon$-**convex** if

$$\text{co}(A) \subseteq A + B_\varepsilon(0),$$

where $\text{co}(A)$ denotes the convex hull of $A$ and $B_\varepsilon(0)$ is an $\varepsilon$ ball around $0$ with respect to the norm of $X$.

</div>

Intuitively speaking, a set that is convex when one fills up all holes smaller than $\varepsilon$ is $\varepsilon$-convex. Now we show that there is no $\varepsilon > 0$ such that $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is $\varepsilon$-convex.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13.9</span><span class="math-callout__name">(Arbitrarily Large Holes)</span></p>

Let $L \in \mathbb{N}$ and $\mathcal{A} = (d_0, d_1, \ldots, d_L, 1) \in \mathbb{N}^{L+2}$. Let $K \subseteq \mathbb{R}^{d_0}$ be compact and let $\sigma \in \mathcal{M}$, with $\mathcal{M}$ as in (3.1.1) and assume that $\sigma$ is not a polynomial. Moreover, assume that there exists an open set, where $\sigma$ is differentiable and not constant.

If there exists an $\varepsilon > 0$ such that $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is $\varepsilon$-convex, then $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is dense in $C(K)$.

</div>

*Proof.* **Step 1.** We show that $\varepsilon$-convexity implies $\overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$ to be convex. By Proposition 13.5, we have that $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is scaling invariant. This implies that $\text{co}(\mathcal{N}(\sigma; \mathcal{A}, \infty))$ is scaling invariant as well. Hence, if there exists $\varepsilon > 0$ such that $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is $\varepsilon$-convex, then for every $\varepsilon' > 0$

$$\text{co}(\mathcal{N}(\sigma; \mathcal{A}, \infty)) = \frac{\varepsilon'}{\varepsilon} \text{co}(\mathcal{N}(\sigma; \mathcal{A}, \infty)) \subseteq \frac{\varepsilon'}{\varepsilon}(\mathcal{N}(\sigma; \mathcal{A}, \infty) + B_\varepsilon(0)) = \mathcal{N}(\sigma; \mathcal{A}, \infty) + B_{\varepsilon'}(0).$$

This yields that $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is $\varepsilon'$-convex for all $\varepsilon' > 0$. As a consequence,

$$\text{co}(\mathcal{N}(\sigma; \mathcal{A}, \infty)) \subseteq \bigcap_{\varepsilon > 0} (\mathcal{N}(\sigma; \mathcal{A}, \infty) + B_\varepsilon(0)) \subseteq \bigcap_{\varepsilon > 0} (\overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)} + B_\varepsilon(0)) = \overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}.$$

Hence, $\overline{\text{co}(\mathcal{N}(\sigma; \mathcal{A}, \infty))} \subseteq \overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$ and, by the well-known fact that in every metric vector space $\text{co}(\overline{A}) \subseteq \overline{\text{co}(A)}$, we conclude that $\overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$ is convex.

**Step 2.** We show that $\mathcal{N}_d^1(\sigma; 1) \subseteq \overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$. If $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is $\varepsilon$-convex, then by Step 1 $\overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$ is convex. The scaling invariance of $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ then shows that $\overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$ is a closed linear subspace of $C(K)$.

Note that, by Proposition 3.16, for every $\boldsymbol{w} \in \mathbb{R}^{d_0}$ and $b \in \mathbb{R}$ there exists a function $f \in \overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$ such that

$$f(\boldsymbol{x}) = \sigma(\boldsymbol{w}^\top \boldsymbol{x} + b) \qquad \text{for all } \boldsymbol{x} \in K.$$

By definition, every constant function is an element of $\overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$. Since $\overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$ is a closed vector space, this implies that for all $n \in \mathbb{N}$ and all $\boldsymbol{w}_1^{(1)}, \ldots, \boldsymbol{w}_n^{(1)} \in \mathbb{R}^{d_0}$, $w_1^{(2)}, \ldots, w_n^{(2)} \in \mathbb{R}$, $b_1^{(1)}, \ldots, b_n^{(1)} \in \mathbb{R}$, $b^{(2)} \in \mathbb{R}$

$$\boldsymbol{x} \mapsto \sum_{i=1}^n w_i^{(2)} \sigma((\boldsymbol{w}_i^{(1)})^\top \boldsymbol{x} + b_i^{(1)}) + b^{(2)} \in \overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}.$$

**Step 3.** From the above, we conclude that $\mathcal{N}_d^1(\sigma; 1) \subseteq \overline{\mathcal{N}(\sigma; \mathcal{A}, \infty)}$. In other words, the whole set of shallow neural networks of arbitrary width is contained in the closure of the set of neural networks with a fixed architecture. By Theorem 3.8, we have that $\mathcal{N}_d^1(\sigma; 1)$ is dense in $C(K)$, which yields the result. $\square$

For any activation function of practical relevance, a set of neural networks with fixed architecture is not dense in $C(K)$. This is only the case for very strange activation functions such as the one discussed in Subsection 3.2. Hence, Theorem 13.9 shows that in general, sets of neural networks of fixed architectures have arbitrarily large holes.

### 13.3 Closedness and Best-Approximation Property

The non-convexity of the set of neural networks can have some serious consequences for the way we think of the approximation or learning problem by neural networks.

Consider $\mathcal{A} = (d_0, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$ and an activation function $\sigma$. Let $H$ be a normed function space on $[-1,1]^{d_0}$ such that $\mathcal{N}(\sigma; \mathcal{A}, \infty) \subseteq H$. For $h \in H$ we would like to find a neural network $\Phi \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ such that

$$\lVert \Phi - h \rVert_H = \inf_{\Phi^* \in \mathcal{N}(\sigma; \mathcal{A}, \infty)} \lVert \Phi^* - h \rVert_H.$$

We say that $\mathcal{N}(\sigma; \mathcal{A}, \infty) \subseteq H$ has

- the **best approximation property**, if for all $h \in H$ there exists at least one $\Phi \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ such that the above holds,
- the **unique best approximation property**, if for all $h \in H$ there exists exactly one $\Phi \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ such that the above holds,
- the **continuous selection property**, if there exists a continuous function $\phi : H \to \mathcal{N}(\sigma; \mathcal{A}, \infty)$ such that $\Phi = \phi(h)$ satisfies the above for all $h \in H$.

In the absence of the best approximation property, we will be able to prove that the learning problem necessarily requires the weights of the neural networks to tend to infinity. Moreover, having a continuous selection procedure is desirable as it implies the existence of a stable selection algorithm; that is, an algorithm which, for similar problems, yields similar neural networks.

#### 13.3.1 Continuous Selection

As shown in [136], neural network spaces essentially never admit the continuous selection property. To give the argument, we first recall the following result from [136, Theorem 3.4] without proof.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13.10</span></p>

Let $p \in (1, \infty)$. Every subset of $L^p([-1,1]^{d_0})$ with the unique best approximation property is convex.

</div>

This allows to show the next proposition.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13.11</span><span class="math-callout__name">(No Continuous Selection)</span></p>

Let $L \in \mathbb{N}$, $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$, let $\sigma : \mathbb{R} \to \mathbb{R}$ be Lipschitz continuous and not a polynomial, and let $p \in (1, \infty)$.

Then, $\mathcal{N}(\sigma; \mathcal{A}, \infty) \subseteq L^p([-1,1]^{d_0})$ does not have the continuous selection property.

</div>

*Proof.* We observe from Theorem 13.6 and the discussion above that $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is not convex. We conclude from Theorem 13.10 that $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ does not have the unique best approximation property. Moreover, if the set $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ does not have the best approximation property, then it obviously cannot have continuous selection. Thus, assume without loss of generality that $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ has the best approximation property and there exists a point $h \in L^p([-1,1]^{d_0})$ and two different $\Phi_1, \Phi_2$ such that

$$\lVert \Phi_1 - h \rVert_{L^p} = \lVert \Phi_2 - h \rVert_{L^p} = \inf_{\Phi^* \in \mathcal{N}(\sigma; \mathcal{A}, \infty)} \lVert \Phi^* - h \rVert_{L^p}.$$

Note that $h \notin \mathcal{N}(\sigma; \mathcal{A}, \infty)$. Define the continuous path

$$[-1, 1] \ni \lambda \mapsto P(\lambda) = \begin{cases} (1 + \lambda) h - \lambda \Phi_1 & \text{for } \lambda \leq 0, \\ (1 - \lambda) h + \lambda \Phi_2 & \text{for } \lambda \geq 0. \end{cases}$$

One can show that for every $\lambda < 0$, $\Phi_1$ is the unique minimizer to $P(\lambda)$ in $\mathcal{N}(\sigma; \mathcal{A}, \infty)$. The same argument holds for $\lambda > 0$ and $\Phi_2$. We conclude that for every selection function $\phi : L^p([-1,1]^{d_0}) \to \mathcal{N}(\sigma; \mathcal{A}, \infty)$ it holds that

$$\lim_{\lambda \downarrow 0} \phi(P(\lambda)) = \Phi_2 \neq \Phi_1 = \lim_{\lambda \uparrow 0} \phi(P(\lambda)).$$

As a consequence, $\phi$ is not continuous, which shows the result. $\square$

#### 13.3.2 Existence of Best Approximations

We have seen in Proposition 13.11 that under very mild assumptions, the continuous selection property cannot hold. Moreover, the next result shows that in many cases, also the best approximation property fails to be satisfied. We provide below a simplified version of [207, Theorem 3.1].

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13.12</span><span class="math-callout__name">(Failure of Best Approximation)</span></p>

Let $\mathcal{A} = (1, 2, 1)$ and let $\sigma : \mathbb{R} \to \mathbb{R}$ be Lipschitz continuous. Additionally assume that there exist $r > 0$ and $\alpha' \neq \alpha$ such that $\sigma$ is differentiable for all $\lvert x \rvert > r$ and $\sigma'(x) \to \alpha$ for $x \to \infty$, $\sigma'(x) \to \alpha'$ for $x \to -\infty$.

Then, there exists a sequence in $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ which converges in $L^p([-1,1])$, for every $p \in (1, \infty)$, and the limit of this sequence is discontinuous. In particular, the limit of the sequence does not lie in $\mathcal{N}(\sigma; \mathcal{A}', \infty)$ for any $\mathcal{A}'$.

</div>

*Proof.* For all $n \in \mathbb{N}$ let $f_n(x) = \sigma(nx + 1) - \sigma(nx)$ for all $x \in \mathbb{R}$. Then $f_n$ can be written as a neural network with architecture $(\sigma; 1, 2, 1)$, i.e. $\mathcal{A} = (1, 2, 1)$. Moreover, for $x > 0$ we observe with the fundamental theorem of calculus and integration by substitution that

$$f_n(x) = \int_x^{x+1/n} n\sigma'(nz) dz = \int_{nx}^{nx+1} \sigma'(z) dz.$$

It is not hard to see that the right hand side converges to $\alpha$ for $n \to \infty$. Similarly, for $x < 0$, we observe that $f_n(x)$ converges to $\alpha'$ for $n \to \infty$. We conclude that

$$f_n \to \alpha \mathbb{1}_{\mathbb{R}_+} + \alpha' \mathbb{1}_{\mathbb{R}_-}$$

almost everywhere as $n \to \infty$. Since $\sigma$ is Lipschitz continuous, we have that $f_n$ is bounded. Therefore, we conclude that $f_n \to \alpha \mathbb{1}_{\mathbb{R}_+} + \alpha' \mathbb{1}_{\mathbb{R}_-}$ in $L^p$ for all $p \in [1, \infty)$ by the dominated convergence theorem. $\square$

There is a straight-forward extension of Proposition 13.12 to arbitrary architectures, that will be the content of Exercises 13.16 and 13.17.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 13.13</span></p>

The proof of Proposition 13.12 does not extend to the $L^\infty$ norm. This, of course, does not mean that generally $\mathcal{N}(\sigma; \mathcal{A}, \infty)$ is a closed set in $L^\infty([-1,1]^{d_0})$. In fact, almost all activation functions used in practice still give rise to non-closed neural network sets, see [207, Theorem 3.3]. However, there is one notable exception. For the ReLU activation function, it can be shown that $\mathcal{N}(\sigma_{\text{ReLU}}; \mathcal{A}, \infty)$ is a closed set in $L^\infty([-1,1]^{d_0})$ if $\mathcal{A}$ has only one hidden layer. The closedness of deep ReLU spaces in $L^\infty$ is an open problem.

</div>

#### 13.3.3 Exploding Weights Phenomenon

Finally, we discuss one of the consequences of the non-existence of best approximations of Proposition 13.12.

Consider a regression problem, where we aim to learn a function $f$ using neural networks with a fixed architecture $\mathcal{N}(\mathcal{A}; \sigma, \infty)$. As discussed in Chapters 10 and 11, we wish to produce a sequence of neural networks $(\Phi_n)_{n=1}^\infty$ such that the risk converges to $0$. If the loss $\mathcal{L}$ is the squared loss, $\mu$ is a probability measure on $[-1,1]^{d_0}$, and the data is given by $(\boldsymbol{x}, f(\boldsymbol{x}))$ for $\boldsymbol{x} \sim \mu$, then

$$\mathcal{R}(\Phi_n) = \lVert \Phi_n - f \rVert_{L^2([-1,1]^{d_0}, \mu)}^2 = \int_{[-1,1]^{d_0}} \lvert \Phi_n(\boldsymbol{x}) - f(\boldsymbol{x}) \rvert^2 d\mu(\boldsymbol{x}) \to 0 \quad \text{for } n \to \infty.$$

According to Proposition 13.12, for a given $\mathcal{A}$ and an activation function $\sigma$, it is possible that this holds, but $f \notin \mathcal{N}(\sigma; \mathcal{A}, \infty)$. The following result shows that in this situation, the weights of $\Phi_n$ diverge.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13.14</span><span class="math-callout__name">(Exploding Weights)</span></p>

Let $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$, let $\sigma : \mathbb{R} \to \mathbb{R}$ be Lipschitz continuous with $C_\sigma \geq 1$, and $\lvert \sigma(x) \rvert \leq C_\sigma \lvert x \rvert$ for all $x \in \mathbb{R}$, and let $\mu$ be a measure on $[-1,1]^{d_0}$.

Assume that there exists a sequence $\Phi_n \in \mathcal{N}(\sigma; \mathcal{A}, \infty)$ and $f \in L^2([-1,1]^{d_0}, \mu) \setminus \mathcal{N}(\sigma; \mathcal{A}, \infty)$ such that

$$\lVert \Phi_n - f \rVert_{L^2([-1,1]^{d_0}, \mu)}^2 \to 0.$$

Then

$$\limsup_{n \to \infty} \max \left\lbrace \lVert \mathbf{W}_n^{(\ell)} \rVert_\infty, \lVert \mathbf{b}_n^{(\ell)} \rVert_\infty \;\middle|\; \ell = 0, \ldots L \right\rbrace = \infty.$$

</div>

*Proof.* We assume towards a contradiction that the left-hand side is finite. As a result, there exists $C > 0$ such that $\Phi_n \in \mathcal{N}(\sigma; \mathcal{A}, C)$ for all $n \in \mathbb{N}$.

By Proposition 13.1, we conclude that $\mathcal{N}(\sigma; \mathcal{A}, C)$ is the image of a compact set under a continuous map and hence is itself a compact set in $L^2([-1,1]^{d_0}, \mu)$. In particular, we have that $\mathcal{N}(\sigma; \mathcal{A}, C)$ is closed. Hence, $f \in \mathcal{N}(\sigma; \mathcal{A}, C)$. This gives a contradiction. $\square$

Proposition 13.14 can be extended to all $f$ for which there is no best approximation in $\mathcal{N}(\sigma; \mathcal{A}, \infty)$, see Exercise 13.18. The results imply that for functions we wish to learn that lack a best approximation within a neural network set, we must expect the weights of the approximating neural networks to grow to infinity. This can be undesirable because, as we will see in the following sections on generalization, a bounded parameter space facilitates many generalization bounds.

---

## Chapter 14: Generalization Properties of Deep Neural Networks

As discussed in Section 1.2, we generally learn based on a finite data set. For example, given data $(x_i, y_i)_{i=1}^m$, we try to find a network $\Phi$ that satisfies $\Phi(x_i) = y_i$ for $i = 1, \ldots, m$. The field of generalization is concerned with how well such $\Phi$ performs on *unseen* data, which refers to any $x$ outside of training data $\lbrace x_1, \ldots, x_m \rbrace$. In this chapter we discuss generalization through the use of covering numbers.

### 14.1 Learning Setup

A general learning problem requires a **feature space** $X$ and a **label space** $Y$, which we assume throughout to be measurable spaces. We observe joint data pairs $(x_i, y_i)_{i=1}^m \subseteq X \times Y$, and aim to identify a connection between the $x$ and $y$ variables. Specifically, we assume a relationship between features $x$ and labels $y$ modeled by a probability distribution $\mathcal{D}$ over $X \times Y$, that generated the observed data $(x_i, y_i)_{i=1}^m$. While this distribution is unknown, our goal is to extract information from it, so that we can make possibly good predictions of $y$ for a given $x$. Importantly, the relationship between $x$ and $y$ need not be deterministic.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 14.1</span><span class="math-callout__name">(Coffee Quality)</span></p>

Our goal is to determine the quality of different coffees. We model the quality as a number in $Y = \lbrace 0/10, \ldots, 10/10 \rbrace$, with higher numbers indicating better quality. We assume that our subjective assessment of quality of coffee is related to six features: "Acidity", "Caffeine content", "Price", "Aftertaste", "Roast level", and "Origin". The feature space $X$ thus corresponds to the set of six-tuples describing these attributes, which can be either numeric or categorical.

We aim to understand the relationship between elements of $X$ and elements of $Y$, but we can neither afford, nor do we have the time to taste all the coffees in the world. Instead, we can sample some coffees, taste them, and grow our database accordingly. This way we obtain samples of pairs in $X \times Y$. The distribution $\mathcal{D}$ from which they are drawn depends on various external factors.

</div>

Characterizing how good a predictor is requires a notion of discrepancy in the label space. This is the purpose of the so-called **loss function**, which is a measurable mapping $\mathcal{L} \colon Y \times Y \to \mathbb{R}_+$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.2</span><span class="math-callout__name">(Population Risk)</span></p>

Let $\mathcal{L} \colon Y \times Y \to \mathbb{R}_+$ be a loss function and let $\mathcal{D}$ be a distribution on $X \times Y$. For a measurable function $h \colon X \to Y$ we call

$$\mathcal{R}(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\mathcal{L}(h(x), y)]$$

the **(population) risk** of $h$.

</div>

The best predictor is one such that its risk is as close as possible to the smallest that any function can achieve. More precisely, we would like a risk that is close to the so-called **Bayes risk**

$$R^* := \inf_{h \colon X \to Y} \mathcal{R}(h),$$

where the infimum is taken over all measurable $h \colon X \to Y$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 14.3</span><span class="math-callout__name">(Loss Functions)</span></p>

The choice of a loss function $\mathcal{L}$ usually depends on the application.

- For a **regression problem**, i.e., a learning problem where $Y$ is a non-discrete subset of a Euclidean space, a common choice is the **square loss** $\mathcal{L}_2(\boldsymbol{y}, \boldsymbol{y}') = \lVert \boldsymbol{y} - \boldsymbol{y}' \rVert^2$.

- For **binary classification** problems, i.e. when $Y$ is a discrete set of cardinality two, the **$0$-$1$ loss**

$$\mathcal{L}_{0\text{-}1}(y, y') = \begin{cases} 1 & y \neq y' \\ 0 & y = y' \end{cases}$$

seems more natural.

- Another frequently used loss for binary classification, especially when we want to predict probabilities (i.e., if $Y = [0,1]$ but all labels are binary), is the **binary cross-entropy loss**

$$\mathcal{L}_{ce}(y, y') = -(y \log(y') + (1-y)\log(1-y')).$$

In contrast to the $0$-$1$ loss, the cross-entropy loss is differentiable, which is desirable in deep learning as we saw in Chapter 10.

</div>

### 14.2 Empirical Risk Minimization

Finding a minimizer of the risk constitutes a considerable challenge. First, we cannot search through all measurable functions. Therefore, we need to restrict ourselves to a specific set $\mathcal{H} \subseteq \lbrace h \colon X \to Y \rbrace$ called the **hypothesis set**. In the following, this set will be some set of neural networks. Second, we are faced with the problem that we cannot evaluate $\mathcal{R}(h)$ for non-trivial loss functions, because the distribution $\mathcal{D}$ is typically unknown so that expectations with respect to $\mathcal{D}$ cannot be computed. To approximate the risk, we will assume access to an i.i.d. sample of $m$ observations drawn from $\mathcal{D}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.4</span><span class="math-callout__name">(Empirical Risk)</span></p>

Let $m \in \mathbb{N}$, let $\mathcal{L} \colon Y \times Y \to \mathbb{R}$ be a loss function and let $S = (x_i, y_i)_{i=1}^m \in (X \times Y)^m$ be a sample. For $h \colon X \to Y$, we call

$$\widehat{\mathcal{R}}_S(h) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(h(x_i), y_i)$$

the **empirical risk** of $h$.

</div>

If the sample $S$ is drawn i.i.d. according to $\mathcal{D}$, then we immediately see from the linearity of the expected value that $\widehat{\mathcal{R}}_S(h)$ is an unbiased estimator of $\mathcal{R}(h)$, i.e., $\mathbb{E}_{S \sim \mathcal{D}^m}[\widehat{\mathcal{R}}_S(h)] = \mathcal{R}(h)$. Moreover, the weak law of large numbers states that the sample mean of an i.i.d. sequence of integrable random variables converges to the expected value in probability. Hence, there is some hope that, at least for large $m \in \mathbb{N}$, minimizing the empirical risk instead of the population risk might lead to a good hypothesis.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.5</span><span class="math-callout__name">(Empirical Risk Minimizer)</span></p>

Let $\mathcal{H} \subseteq \lbrace h \colon X \to Y \rbrace$ be a hypothesis set. Let $m \in \mathbb{N}$, let $\mathcal{L} \colon Y \times Y \to \mathbb{R}$ be a loss function and let $S = (x_i, y_i)_{i=1}^m \in (X \times Y)^m$ be a sample. We call a function $h_S$ such that

$$\widehat{\mathcal{R}}_S(h_S) = \inf_{h \in \mathcal{H}} \widehat{\mathcal{R}}_S(h)$$

an **empirical risk minimizer**.

</div>

From a generalization perspective, supervised deep learning is empirical risk minimization over sets of neural networks. Let $\mathcal{H}$ be some hypothesis set, such that an empirical risk minimizer $h_S$ exists for all $S \in (X \times Y)^m$. Moreover, let $g \in \mathcal{H}$ be arbitrary. Then

$$\mathcal{R}(h_S) - R^* \leq 2 \sup_{h \in \mathcal{H}} \lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert + \inf_{g \in \mathcal{H}} \mathcal{R}(g) - R^* =: 2\varepsilon_{\text{gen}} + \varepsilon_{\text{approx}}.$$

Similarly, considering only the first inequality yields that

$$\mathcal{R}(h_S) \leq \sup_{h \in \mathcal{H}} \lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert + \inf_{g \in \mathcal{H}} \widehat{\mathcal{R}}_S(g) =: \varepsilon_{\text{gen}} + \varepsilon_{\text{int}}.$$

How to choose $\mathcal{H}$ to reduce the **approximation error** $\varepsilon_{\text{approx}}$ or the **interpolation error** $\varepsilon_{\text{int}}$ was discussed at length in the previous chapters. The final piece is to figure out how to bound the **generalization error** $\sup_{h \in \mathcal{H}} \lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert$.

### 14.3 Generalization Bounds

We have seen that one aspect of successful learning is to bound the generalization error $\varepsilon_{\text{gen}}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.6</span><span class="math-callout__name">(Generalization Bound)</span></p>

Let $\mathcal{H} \subseteq \lbrace h \colon X \to Y \rbrace$ be a hypothesis set, and let $\mathcal{L} \colon Y \times Y \to \mathbb{R}$ be a loss function. Let $\kappa \colon (0,1) \times \mathbb{N} \to \mathbb{R}_+$ be such that for every $\delta \in (0,1)$ it holds $\kappa(\delta, m) \to 0$ for $m \to \infty$. We call $\kappa$ a **generalization bound for $\mathcal{H}$** if for every distribution $\mathcal{D}$ on $X \times Y$, every $m \in \mathbb{N}$ and every $\delta \in (0,1)$, it holds with probability at least $1 - \delta$ over the random sample $S \sim \mathcal{D}^m$ that

$$\sup_{h \in \mathcal{H}} \lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert \leq \kappa(\delta, m).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 14.7</span></p>

For a generalization bound $\kappa$ it holds that

$$\mathbb{P}\left[\left\lvert \mathcal{R}(h_S) - \widehat{\mathcal{R}}_S(h_S)\right\rvert \leq \varepsilon\right] \geq 1 - \delta$$

as soon as $m$ is so large that $\kappa(\delta, m) \leq \varepsilon$. If there exists an empirical risk minimizer $h_S$ such that $\widehat{\mathcal{R}}_S(h_S) = 0$, then with high probability the empirical risk minimizer will also have a small risk $\mathcal{R}(h_S)$. Empirical risk minimization is often referred to as a "PAC" algorithm, which stands for *probably ($\delta$) approximately correct ($\varepsilon$)*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 14.8</span><span class="math-callout__name">(Generalization in the Coffee Quality Problem)</span></p>

In Example 14.1, the underlying distribution describes both our process of choosing coffees and the relation between the attributes and the quality. Suppose we do not enjoy drinking coffee that costs less than 1€. Consequently, we do not have a single sample of such coffee in the dataset, and therefore we have no chance of learning the quality of cheap coffees.

However, the absence of coffee samples costing less than 1€ in our dataset is due to our *general avoidance* of such coffee. As a result, we run a low risk of incorrectly classifying the quality of a coffee that is cheaper than 1€, since it is unlikely that we will choose such a coffee in the future.

</div>

To establish generalization bounds, we use stochastic tools that guarantee that the empirical risk converges to the true risk as the sample size increases. This is typically achieved through concentration inequalities. One of the simplest and most well-known is Hoeffding's inequality (Theorem A.24).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.9</span><span class="math-callout__name">(Finite Hypothesis Set)</span></p>

Let $\mathcal{H} \subseteq \lbrace h \colon X \mapsto Y \rbrace$ be a *finite* hypothesis set. Let $\mathcal{L} \colon Y \times Y \to \mathbb{R}$ be such that $\mathcal{L}(Y \times Y) \subseteq [c_1, c_2]$ with $c_2 - c_1 = C > 0$.

Then, for every $m \in \mathbb{N}$ and every distribution $\mathcal{D}$ on $X \times Y$ it holds with probability at least $1 - \delta$ over the sample $S \sim \mathcal{D}^m$ that

$$\sup_{h \in \mathcal{H}} \lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert \leq C \sqrt{\frac{\log(\lvert \mathcal{H} \rvert) + \log(2/\delta)}{2m}}.$$

</div>

*Proof.* Let $\mathcal{H} = \lbrace h_1, \ldots, h_n \rbrace$. Then it holds by a union bound that

$$\mathbb{P}\left[\exists h_i \in \mathcal{H} : \lvert \mathcal{R}(h_i) - \widehat{\mathcal{R}}_S(h_i) \rvert > \varepsilon\right] \leq \sum_{i=1}^n \mathbb{P}\left[\lvert \mathcal{R}(h_i) - \widehat{\mathcal{R}}_S(h_i) \rvert > \varepsilon\right].$$

Note that $\widehat{\mathcal{R}}_S(h_i)$ is the mean of independent random variables which take their values almost surely in $[c_1, c_2]$. Additionally, $\mathcal{R}(h_i)$ is the expectation of $\widehat{\mathcal{R}}_S(h_i)$. The proof can therefore be finished by applying Hoeffding's inequality (Theorem A.24). $\square$

### 14.4 Generalization Bounds from Covering Numbers

To derive a generalization bound for classes of neural networks, we start by introducing the notion of covering numbers.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.10</span><span class="math-callout__name">(Covering Number)</span></p>

Let $A$ be a relatively compact subset of a metric space $(X, d)$. For $\varepsilon > 0$, we call

$$\mathcal{G}(A, \varepsilon, (X, d)) := \min\left\lbrace n \in \mathbb{N} \;\middle|\; \exists\, (x_i)_{i=1}^n \subseteq X \text{ s.t. } \bigcup_{i=1}^n B_\varepsilon(x_i) \supset A \right\rbrace,$$

where $B_\varepsilon(x) = \lbrace z \in X \mid d(z, x) \leq \varepsilon \rbrace$, the **$\varepsilon$-covering number** of $A$ in $X$. In case $X$ or $d$ are clear from context, we also write $\mathcal{G}(A, \varepsilon, d)$ or $\mathcal{G}(A, \varepsilon, X)$ instead of $\mathcal{G}(A, \varepsilon, (X, d))$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.11</span><span class="math-callout__name">(Generalization via Covering Numbers)</span></p>

Let $C_Y, C_\mathcal{L} > 0$ and $\alpha > 0$. Let $Y \subseteq [-C_Y, C_Y]$, $X \subseteq \mathbb{R}^d$ for some $d \in \mathbb{N}$, and $\mathcal{H} \subseteq \lbrace h \colon X \to Y \rbrace$. Further, let $\mathcal{L} \colon Y \times Y \to \mathbb{R}$ be $C_\mathcal{L}$-Lipschitz.

Then, for every distribution $\mathcal{D}$ on $X \times Y$ and every $m \in \mathbb{N}$ it holds with probability at least $1 - \delta$ over the sample $S \sim \mathcal{D}^m$ that for all $h \in \mathcal{H}$

$$\lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert \leq 4 C_Y C_\mathcal{L} \sqrt{\frac{\log(\mathcal{G}(\mathcal{H}, m^{-\alpha}, L^\infty(X))) + \log(2/\delta)}{m}} + \frac{2C_\mathcal{L}}{m^\alpha}.$$

</div>

*Proof.* Let

$$M = \mathcal{G}(\mathcal{H}, m^{-\alpha}, L^\infty(X))$$

and let $\mathcal{H}_M = (h_i)_{i=1}^M \subseteq \mathcal{H}$ be such that for every $h \in \mathcal{H}$ there exists $h_i \in \mathcal{H}_M$ with $\lVert h - h_i \rVert_{L^\infty(X)} \leq 1/m^\alpha$. Fix for the moment such $h \in \mathcal{H}$ and $h_i \in \mathcal{H}_M$. By the reverse and normal triangle inequalities, we have

$$\lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert - \lvert \mathcal{R}(h_i) - \widehat{\mathcal{R}}_S(h_i) \rvert \leq \lvert \mathcal{R}(h) - \mathcal{R}(h_i) \rvert + \lvert \widehat{\mathcal{R}}_S(h) - \widehat{\mathcal{R}}_S(h_i) \rvert.$$

Moreover, from the monotonicity of the expected value and the Lipschitz property of $\mathcal{L}$ it follows that

$$\lvert \mathcal{R}(h) - \mathcal{R}(h_i) \rvert \leq \mathbb{E}\lvert \mathcal{L}(h(x), y) - \mathcal{L}(h_i(x), y) \rvert \leq C_\mathcal{L} \mathbb{E}\lvert h(x) - h_i(x) \rvert \leq \frac{C_\mathcal{L}}{m^\alpha}.$$

A similar estimate yields $\lvert \widehat{\mathcal{R}}_S(h) - \widehat{\mathcal{R}}_S(h_i) \rvert \leq C_\mathcal{L}/m^\alpha$. We thus conclude that for every $\varepsilon > 0$

$$\mathbb{P}_{S \sim \mathcal{D}^m}\left[\exists h \in \mathcal{H} : \lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert \geq \varepsilon\right] \leq \mathbb{P}_{S \sim \mathcal{D}^m}\left[\exists h_i \in \mathcal{H}_M : \lvert \mathcal{R}(h_i) - \widehat{\mathcal{R}}_S(h_i) \rvert \geq \varepsilon - \frac{2C_\mathcal{L}}{m^\alpha}\right].$$

From Proposition 14.9, we know that for $\varepsilon > 0$ and $\delta \in (0,1)$

$$\mathbb{P}_{S \sim \mathcal{D}^m}\left[\exists h_i \in H_M : \lvert \mathcal{R}(h_i) - \widehat{\mathcal{R}}_S(h_i) \rvert \geq \varepsilon - \frac{2C_\mathcal{L}}{m^\alpha}\right] \leq \delta$$

as long as

$$\varepsilon - \frac{2C_\mathcal{L}}{m^\alpha} > C\sqrt{\frac{\log(M) + \log(2/\delta)}{2m}},$$

where $C = 2\sqrt{2} C_\mathcal{L} C_Y$. The definition of $M$, together with the above estimates, give that with probability at least $1 - \delta$ it holds for all $h \in \mathcal{H}$

$$\lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert \leq 2\sqrt{2} C_\mathcal{L} C_Y \sqrt{\frac{\log(\mathcal{G}(\mathcal{H}, m^{-\alpha}, L^\infty)) + \log(2/\delta)}{2m}} + \frac{2C_\mathcal{L}}{m^\alpha}.$$

This concludes the proof. $\square$

### 14.5 Covering Numbers of Deep Neural Networks

We have seen in Theorem 14.11 that estimating $L^\infty$-covering numbers is crucial for understanding the generalization error. The following lemma suggests a simpler approach for bounding covering numbers of neural network classes.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 14.12</span><span class="math-callout__name">(Covering Numbers under Lipschitz Maps)</span></p>

Let $X_1$, $X_2$ be two metric spaces and let $f \colon X_1 \to X_2$ be Lipschitz continuous with Lipschitz constant $C_{\text{Lip}}$. For every relatively compact $A \subseteq X_1$ it holds that for all $\varepsilon > 0$

$$\mathcal{G}(f(A), C_{\text{Lip}} \varepsilon, X_2) \leq \mathcal{G}(A, \varepsilon, X_1).$$

</div>

Conveniently, we have already observed in Proposition 13.1, that the set of neural networks is the image of $\mathcal{PN}(\mathcal{A}, B)$ under the Lipschitz continuous realization map $R_\sigma$. It thus suffices to establish the $\varepsilon$-covering number of $\mathcal{PN}(\mathcal{A}, B)$ or equivalently of $[-B, B]^{n_\mathcal{A}}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14.13</span><span class="math-callout__name">(Covering Number of Hypercubes)</span></p>

Let $B$, $\varepsilon > 0$ and $q \in \mathbb{N}$. Then

$$\mathcal{G}([-B, B]^q, \varepsilon, (\mathbb{R}^q, \lVert \cdot \rVert_\infty)) \leq \lceil B/\varepsilon \rceil^q.$$

</div>

*Proof.* We start with the one-dimensional case $q = 1$. We choose $k = \lfloor B/\varepsilon \rfloor$,

$$x_0 = -B + \varepsilon \quad \text{and} \quad x_j = x_{j-1} + 2\varepsilon \text{ for } j = 1, \ldots, k-1.$$

It is clear that all points between $-B$ and $x_{k-1}$ have distance at most $\varepsilon$ to one of the $x_j$. Also, $x_{k-1} = -B + \varepsilon + 2(k-1)\varepsilon \geq B - \varepsilon$. We conclude that $\mathcal{G}([-B,B], \varepsilon, \mathbb{R}) \leq \lceil B/\varepsilon \rceil$. Set $X_k := \lbrace x_0, \ldots, x_{k-1} \rbrace$.

For arbitrary $q$, we observe that for every $x \in [-B, B]^q$ there is an element in $X_k^q = \bigotimes_{k=1}^q X_k$ with $\lVert \cdot \rVert_\infty$ distance less than $\varepsilon$. Clearly, $\lvert X_k^q \rvert = \lceil B/\varepsilon \rceil^q$, which completes the proof. $\square$

Having established a covering number for $[-B, B]^{n_\mathcal{A}}$ and hence $\mathcal{PN}(\mathcal{A}, B)$, we can now estimate the covering numbers of deep neural networks by combining Lemma 14.12 and Propositions 13.1 and 14.13.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.14</span><span class="math-callout__name">(Covering Numbers of Neural Networks)</span></p>

Let $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$, let $\sigma \colon \mathbb{R} \to \mathbb{R}$ be $C_\sigma$-Lipschitz continuous with $C_\sigma \geq 1$, let $\lvert \sigma(x) \rvert \leq C_\sigma \lvert x \rvert$ for all $x \in \mathbb{R}$, and let $B \geq 1$. Then

$$\mathcal{G}(\mathcal{N}(\sigma; \mathcal{A}, B), \varepsilon, L^\infty([0,1]^{d_0})) \leq \mathcal{G}([-B,B]^{n_\mathcal{A}}, \varepsilon / (n_\mathcal{A}(2C_\sigma B d_{\max})^L), (\mathbb{R}^{n_\mathcal{A}}, \lVert \cdot \rVert_\infty)) \leq \lceil n_\mathcal{A}/\varepsilon \rceil^{n_\mathcal{A}} \lceil 2C_\sigma B d_{\max} \rceil^{n_\mathcal{A} L}.$$

</div>

We end this section by applying the above to the generalization bound of Theorem 14.11 with $\alpha = 1/2$. To simplify the analysis, we restrict the discussion to neural networks with range $[-1, 1]$. To this end, denote

$$\mathcal{N}^*(\sigma; \mathcal{A}, B) := \left\lbrace \Phi \in \mathcal{N}(\sigma; \mathcal{A}, B) \;\middle|\; \Phi(\boldsymbol{x}) \in [-1, 1] \text{ for all } \boldsymbol{x} \in [0,1]^{d_0} \right\rbrace.$$

Since $\mathcal{N}^*(\sigma; \mathcal{A}, B) \subseteq \mathcal{N}(\sigma; \mathcal{A}, B)$ we can bound the covering numbers of $\mathcal{N}^*(\sigma; \mathcal{A}, B)$ by those of $\mathcal{N}(\sigma; \mathcal{A}, B)$. This yields the following result.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.15</span><span class="math-callout__name">(Generalization Bound for Neural Networks)</span></p>

Let $C_\mathcal{L} > 0$ and let $\mathcal{L} \colon [-1,1] \times [-1,1] \to \mathbb{R}$ be $C_\mathcal{L}$-Lipschitz continuous. Further, let $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$, let $\sigma \colon \mathbb{R} \to \mathbb{R}$ be $C_\sigma$-Lipschitz continuous with $C_\sigma \geq 1$, and $\lvert \sigma(x) \rvert \leq C_\sigma \lvert x \rvert$ for all $x \in \mathbb{R}$, and let $B \geq 1$.

Then, for every $m \in \mathbb{N}$, and every distribution $\mathcal{D}$ on $X \times [-1,1]$ it holds with probability at least $1 - \delta$ over $S \sim \mathcal{D}^m$ that for all $\Phi \in \mathcal{N}^*(\sigma; \mathcal{A}, B)$

$$\lvert \mathcal{R}(\Phi) - \widehat{\mathcal{R}}_S(\Phi) \rvert \leq 4C_\mathcal{L} \sqrt{\frac{n_\mathcal{A} \log(\lceil n_\mathcal{A} \sqrt{m} \rceil) + L n_\mathcal{A} \log(\lceil 2C_\sigma B d_{\max} \rceil) + \log(2/\delta)}{m}} + \frac{2C_\mathcal{L}}{\sqrt{m}}.$$

</div>

### 14.6 The Approximation-Complexity Trade-off

We recall the decomposition of the error:

$$\mathcal{R}(h_S) - R^* \leq 2\varepsilon_{\text{gen}} + \varepsilon_{\text{approx}},$$

where $R_*$ is the Bayes risk. We make the following observations about the approximation error $\varepsilon_{\text{approx}}$ and generalization error $\varepsilon_{\text{gen}}$ in the context of neural network based learning:

- *Scaling of generalization error:* By Theorem 14.15, for a hypothesis class $\mathcal{H}$ of neural networks with $n_\mathcal{A}$ weights and $L$ layers, and for sample of size $m \in \mathbb{N}$, the generalization error $\varepsilon_{\text{gen}}$ essentially scales like

$$\varepsilon_{\text{gen}} = O\!\left(\sqrt{(n_\mathcal{A} \log(n_\mathcal{A} m) + L n_\mathcal{A} \log(n_\mathcal{A}))/m}\right) \quad \text{as } m \to \infty.$$

- *Scaling of approximation error:* Assume there exists $h^*$ such that $\mathcal{R}(h^*) = R^*$, and let the loss function $\mathcal{L}$ be Lipschitz continuous in the first coordinate. Then

$$\varepsilon_{\text{approx}} = \inf_{h \in \mathcal{H}} \mathcal{R}(h) - \mathcal{R}(h^*) \leq C \inf_{h \in \mathcal{H}} \lVert h - h^* \rVert_{L^\infty},$$

for some constant $C > 0$. If we choose $\mathcal{H}$ as a set of neural networks with size $n_\mathcal{A}$ and $L$ layers, then $\inf_{h \in \mathcal{H}} \lVert h - h^* \rVert_{L^\infty}$ behaves like $n_\mathcal{A}^{-r}$ for appropriate activation functions and regularity conditions on $h^*$.

By these considerations, we conclude that for an empirical risk minimizer $\Phi_S$ from a set of neural networks with $n_\mathcal{A}$ weights and $L$ layers, it holds that

$$\mathcal{R}(\Phi_S) - R^* \leq O\!\left(\sqrt{(n_\mathcal{A} \log(m) + L n_\mathcal{A} \log(n_\mathcal{A}))/m}\right) + O(n_\mathcal{A}^{-r}),$$

for $m \to \infty$ and for some $r$ depending on the regularity of $h^*$. Note that enlarging the neural network set, i.e., increasing $n_\mathcal{A}$ has two effects: The term associated to approximation decreases, and the term associated to generalization increases. This trade-off is known as **approximation-complexity trade-off**.

Using this notion, we can separate all models into three classes:

- *Underfitting:* If the approximation error decays faster than the estimation error increases.
- *Optimal:* If the sum of approximation error and generalization error is at a minimum.
- *Overfitting:* If the approximation error decays slower than the estimation error increases.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Overparameterized Regime)</span></p>

In Chapter 15, we will see that deep learning often operates in the regime where the number of parameters $n_\mathcal{A}$ exceeds the optimal trade-off point. For certain architectures used in practice, $n_\mathcal{A}$ can be so large that the theory of the approximation-complexity trade-off suggests that learning should be impossible. However, the present analysis only provides upper bounds. It does not prove that learning is impossible or even impractical in the overparameterized regime. Moreover, in Chapter 11 we have already seen indications that learning in the overparametrized regime need not necessarily lead to large generalization errors.

</div>

### 14.7 PAC Learning from VC Dimension

In addition to covering numbers, there are several other tools to analyze the generalization capacity of hypothesis sets. In the context of classification problems, one of the most important is the so-called Vapnik–Chervonenkis (VC) dimension.

#### 14.7.1 Definition and Examples

Let $\mathcal{H}$ be a hypothesis set of functions mapping from $\mathbb{R}^d$ to $\lbrace 0, 1 \rbrace$. A set $S = \lbrace \boldsymbol{x}_1, \ldots, \boldsymbol{x}_n \rbrace \subseteq \mathbb{R}^d$ is said to be **shattered** by $\mathcal{H}$ if for every $(y_1, \ldots, y_n) \in \lbrace 0, 1 \rbrace^n$ there exists $h \in \mathcal{H}$ such that $h(\boldsymbol{x}_j) = y_j$ for all $j \in \mathbb{N}$.

The VC dimension quantifies the complexity of a function class via the number of points that can in principle be shattered.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14.16</span><span class="math-callout__name">(VC Dimension)</span></p>

The **VC dimension** of $\mathcal{H}$ is the cardinality of the largest set $S \subseteq \mathbb{R}^d$ that is shattered by $\mathcal{H}$. We denote the VC dimension by $\text{VCdim}(\mathcal{H})$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 14.17</span><span class="math-callout__name">(Intervals)</span></p>

Let $\mathcal{H} = \lbrace \mathbb{1}_{[a,b]} \mid a, b \in \mathbb{R} \rbrace$. It is clear that $\text{VCdim}(\mathcal{H}) \geq 2$ since for $x_1 < x_2$ the functions

$$\mathbb{1}_{[x_1-2, x_1-1]}, \quad \mathbb{1}_{[x_1-1, x_1]}, \quad \mathbb{1}_{[x_1, x_2]}, \quad \mathbb{1}_{[x_2, x_2+1]}$$

are all different when restricted to $S = (x_1, x_2)$.

On the other hand, if $x_1 < x_2 < x_3$ then, since $h^{-1}(\lbrace 1 \rbrace)$ is an interval for all $h \in \mathcal{H}$, we have that $h(x_1) = 1 = h(x_3)$ implies $h(x_2) = 1$. Hence, no set of three elements can be shattered. Therefore, $\text{VCdim}(\mathcal{H}) = 2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 14.18</span><span class="math-callout__name">(Half-spaces)</span></p>

Let $\mathcal{H}_2 = \lbrace \mathbb{1}_{[0, \infty)}(\langle \boldsymbol{w}, \cdot \rangle + b) \mid \boldsymbol{w} \in \mathbb{R}^2, b \in \mathbb{R} \rbrace$ be a hypothesis set of rotated and shifted two-dimensional half-spaces. $\mathcal{H}_2$ shatters a set of three points. More generally, for $d \geq 2$ with

$$\mathcal{H}_d := \lbrace \boldsymbol{x} \mapsto \mathbb{1}_{[0, \infty)}(\boldsymbol{w}^\top \boldsymbol{x} + b) \mid \boldsymbol{w} \in \mathbb{R}^d, \; b \in \mathbb{R} \rbrace$$

the VC dimension of $\mathcal{H}_d$ equals $d + 1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 14.19</span><span class="math-callout__name">(Infinite VC Dimension)</span></p>

Let for $x \in \mathbb{R}$

$$\mathcal{H} := \lbrace x \mapsto \mathbb{1}_{[0, \infty)}(\sin(wx)) \mid w \in \mathbb{R} \rbrace.$$

Then the VC dimension of $\mathcal{H}$ is infinite.

</div>

#### 14.7.2 Generalization Based on VC Dimension

In the following, we consider a classification problem. Denote by $\mathcal{D}$ the data-generating distribution on $\mathbb{R}^d \times \lbrace 0, 1 \rbrace$. Moreover, we let $\mathcal{H}$ be a set of functions from $\mathbb{R}^d \to \lbrace 0, 1 \rbrace$.

In the binary classification set-up, the natural choice of a loss function is the $0$-$1$ loss $\mathcal{L}_{0\text{-}1}(y, y') = \mathbb{1}_{y \neq y'}$. Thus, given a sample $S$, the empirical risk of a function $h \in \mathcal{H}$ is

$$\widehat{\mathcal{R}}_S(h) = \frac{1}{m} \sum_{i=1}^m \mathbb{1}_{h(\boldsymbol{x}_i) \neq y_i}.$$

Moreover, the risk can be written as

$$\mathcal{R}(h) = \mathbb{P}_{(\boldsymbol{x}, y) \sim \mathcal{D}}[h(\boldsymbol{x}) \neq y],$$

i.e., the probability under $(\boldsymbol{x}, y) \sim \mathcal{D}$ of $h$ misclassifying the label $y$ of $\boldsymbol{x}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.20</span><span class="math-callout__name">(VC Generalization Bound)</span></p>

Let $d, k \in \mathbb{N}$ and $\mathcal{H} \subseteq \lbrace h \colon \mathbb{R}^d \to \lbrace 0, 1 \rbrace \rbrace$ have VC dimension $k$. Let $\mathcal{D}$ be a distribution on $\mathbb{R}^d \times \lbrace 0, 1 \rbrace$. Then, for every $\delta > 0$ and $m \in \mathbb{N}$, it holds with probability at least $1 - \delta$ over a sample $S \sim \mathcal{D}^m$ that for every $h \in \mathcal{H}$

$$\lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert \leq \sqrt{\frac{2k \log(em/k)}{m}} + \sqrt{\frac{\log(1/\delta)}{2m}}.$$

</div>

In words, Theorem 14.20 tells us that if a hypothesis class has finite VC dimension, then a hypothesis with a small empirical risk will have a small risk if the number of samples is large. This shows that empirical risk minimization is a viable strategy in this scenario.

Will this approach also work if the VC dimension is not bounded? No, in fact, in that case, no learning algorithm will succeed in reliably producing a hypothesis for which the risk is close to the best possible.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.21</span><span class="math-callout__name">(No-Free-Lunch)</span></p>

Let $k \in \mathbb{N}$ and let $\mathcal{H} \subseteq \lbrace h \colon X \to \lbrace 0, 1 \rbrace \rbrace$ be a hypothesis set with VC dimension $k$. Then, for every $m \in \mathbb{N}$ and every learning algorithm $\text{A} \colon (X \times \lbrace 0, 1 \rbrace)^m \to \mathcal{H}$ there exists a distribution $\mathcal{D}$ on $X \times \lbrace 0, 1 \rbrace$ such that

$$\mathbb{P}_{S \sim \mathcal{D}^m}\left[\mathcal{R}(\text{A}(S)) - \inf_{h \in \mathcal{H}} \mathcal{R}(h) > \sqrt{\frac{k}{320m}}\right] \geq \frac{1}{64}.$$

</div>

Theorem 14.21 immediately implies the following statement for the generalization bound.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 14.22</span></p>

Let $k \in \mathbb{N}$ and let $\mathcal{H} \subseteq \lbrace h \colon X \to \lbrace 0, 1 \rbrace \rbrace$ be a hypothesis set with VC dimension $k$. Then, for every $m \in \mathbb{N}$ there exists a distribution $\mathcal{D}$ on $X \times \lbrace 0, 1 \rbrace$ such that

$$\mathbb{P}_{S \sim \mathcal{D}^m}\left[\sup_{h \in \mathcal{H}} \lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert > \sqrt{\frac{k}{1280 m}}\right] \geq \frac{1}{64}.$$

</div>

*Proof.* For a sample $S$, let $h_S \in \mathcal{H}$ be an empirical risk minimizer, i.e., $\widehat{\mathcal{R}}_S(h_S) = \min_{h \in \mathcal{H}} \widehat{\mathcal{R}}_S(h)$. Let $\mathcal{D}$ be the distribution of Theorem 14.21. Moreover, for $\delta > 0$, let $h_\delta \in \mathcal{H}$ be such that $\mathcal{R}(h_\delta) - \inf_{h \in \mathcal{H}} \mathcal{R}(h) < \delta$. Then, applying Theorem 14.21 with $\text{A}(S) = h_S$ it holds that

$$2 \sup_{h \in \mathcal{H}} \lvert \mathcal{R}(h) - \widehat{\mathcal{R}}_S(h) \rvert \geq \lvert \mathcal{R}(h_S) - \widehat{\mathcal{R}}_S(h_S) \rvert + \lvert \mathcal{R}(h_\delta) - \widehat{\mathcal{R}}_S(h_\delta) \rvert \geq \mathcal{R}(h_S) - \mathcal{R}(h_\delta) > \mathcal{R}(h_S) - \inf_{h \in \mathcal{H}} \mathcal{R}(h) - \delta,$$

where we used the definition of $h_S$ in the third inequality. The proof is completed by applying Theorem 14.21 and using that $\delta$ was arbitrary. $\square$

We have seen now, that we have a generalization bound scaling like $O(1/\sqrt{m})$ for $m \to \infty$ if and only if the VC dimension of the hypothesis class is finite. In more quantitative terms, we require the VC dimension of a neural network to be smaller than $m$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.23</span><span class="math-callout__name">(VC Dimension of ReLU Networks)</span></p>

Let $\mathcal{A} \in \mathbb{N}^{L+2}$, $L \in \mathbb{N}$ and set

$$\mathcal{H} := \lbrace \mathbb{1}_{[0, \infty)} \circ \Phi \mid \Phi \in \mathcal{N}(\sigma_{\text{ReLU}}; \mathcal{A}, \infty) \rbrace.$$

Then, there exists a constant $C > 0$ independent of $L$ and $\mathcal{A}$ such that

$$\text{VCdim}(\mathcal{H}) \leq C \cdot (n_\mathcal{A} L \log(n_\mathcal{A}) + n_\mathcal{A} L^2).$$

</div>

The bound from Theorem 14.20 is meaningful if $m \gg k$. For ReLU neural networks as in Theorem 14.23, this means $m \gg n_\mathcal{A} L \log(n_\mathcal{A}) + n_\mathcal{A} L^2$. Fixing $L = 1$ this amounts to $m \gg n_\mathcal{A} \log(n_\mathcal{A})$ for a shallow neural network with $n_\mathcal{A}$ parameters. This condition is contrary to what we assumed in Chapter 11, where it was crucial that $n_\mathcal{A} \gg m$. If the VC dimension of the neural network sets scale like $O(n_\mathcal{A} \log(n_\mathcal{A}))$, then Theorem 14.21 and Corollary 14.22 indicate that, at least for certain distributions, generalization should not be possible in this regime. We will discuss the resolution of this potential paradox in Chapter 15.

### 14.8 Lower Bounds on Achievable Approximation Rates

We conclude this chapter on the complexities and generalization bounds of neural networks by using the established VC dimension bound of Theorem 14.23 to deduce limitations to the approximation capacity of neural networks.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14.24</span><span class="math-callout__name">(Lower Bound on Approximation Rates)</span></p>

Let $k$, $d \in \mathbb{N}$. Assume that for every $\varepsilon > 0$ there exists $L_\varepsilon \in \mathbb{N}$ and $\mathcal{A}_\varepsilon$ with $L_\varepsilon$ layers and input dimension $d$ such that

$$\sup_{\lVert f \rVert_{C^k([0,1]^d)} \leq 1} \inf_{\Phi \in \mathcal{N}(\sigma_{\text{ReLU}}; \mathcal{A}_\varepsilon, \infty)} \lVert f - \Phi \rVert_{C^0([0,1]^d)} < \frac{\varepsilon}{2}.$$

Then there exists $C > 0$ solely depending on $k$ and $d$, such that for all $\varepsilon \in (0, 1)$

$$n_{\mathcal{A}_\varepsilon} L_\varepsilon \log(n_{\mathcal{A}_\varepsilon}) + n_{\mathcal{A}_\varepsilon} L_\varepsilon^2 \geq C \varepsilon^{-d/k}.$$

</div>

To interpret Theorem 14.24, we consider two situations:

- In case the depth is allowed to increase at most logarithmically, then reaching uniform error $\varepsilon$ for all $f \in C^k([0,1]^d)$ with $\lVert f \rVert_{C^k([0,1]^d)} \leq 1$ requires

$$n_{\mathcal{A}_\varepsilon} \log(n_{\mathcal{A}_\varepsilon}) \log(\varepsilon) + n_{\mathcal{A}_\varepsilon} \log(\varepsilon)^2 \geq C \varepsilon^{-d/k}.$$

In terms of the neural network size, this (necessary) condition becomes $n_{\mathcal{A}_\varepsilon} \geq C \varepsilon^{-d/k} / \log(\varepsilon)^2$. As we have shown in Chapter 7, in particular Theorem 7.10, up to log terms this condition is also sufficient. Hence, while the constructive proof of Theorem 7.10 might have seemed rather specific, under the assumption of the depth increasing at most logarithmically, it was essentially optimal! The neural networks in this proof are shown to have size $O(\varepsilon^{-d/k})$ up to log terms.

- If we allow the depth $L_\varepsilon$ to increase faster than logarithmically in $\varepsilon$, then the lower bound on the required neural network size improves. Fixing for example $\mathcal{A}_\varepsilon$ with $L_\varepsilon$ layers such that $n_{\mathcal{A}_\varepsilon} \leq W L_\varepsilon$ for some fixed $\varepsilon$ independent $W \in \mathbb{N}$, the (necessary) condition on the depth becomes

$$W \log(W L_\varepsilon) L_\varepsilon^2 + W L_\varepsilon^3 \geq C \varepsilon^{-d/k}$$

and hence $L_\varepsilon \gtrsim \varepsilon^{-d/(3k)}$.

To sum up, in order to get error $\varepsilon$ uniformly for all $\lVert f \rVert_{C^k([0,1]^d)} \leq 1$, the size of a ReLU neural network is required to increase at least like $O(\varepsilon^{-d/(2k)})$ as $\varepsilon \to 0$, i.e. the best possible attainable convergence rate is $2k/d$. It has been proven that this rate is also achievable, and thus the bound is sharp. Achieving this rate requires neural network architectures that grow faster in depth than in width.

---

## Chapter 15: Generalization in the Overparameterized Regime

In the previous chapter, we discussed the theory of generalization for deep networks trained by minimizing the empirical risk. A key conclusion was that good generalization is possible as long as we choose an architecture that has a moderate number of neural network parameters relative to the number of training samples. Moreover, we saw in Section 14.6 that the best performance can be expected when the neural network size is chosen to balance the generalization and approximation errors, by minimizing their sum.

Surprisingly, successful neural network architectures do not necessarily follow these theoretical observations. Consider the neural network architectures in the ImageNet Classification Competition: all architectures have a ratio of parameters to training samples larger than one, i.e. they all have more parameters than training samples. For the largest model, there are by a factor 1000 more neural network parameters than training samples. Given that the practical application of deep learning appears to operate in a regime significantly different from the one analyzed in Chapter 14, we must ask: Why do these methods still work effectively?

### 15.1 The Double Descent Phenomenon

The success of deep learning in a regime not covered by traditional statistical learning theory puzzled researchers for some time. An intriguing set of experiments showed that while the risk follows the upper bound from Section 14.6 for neural network architectures that do not interpolate the data, the curve does not expand to infinity in the way that the approximation-complexity trade-off suggests. Instead, after surpassing the so-called "interpolation threshold", the risk starts to decrease again. This behavior is known as **double descent**.

#### 15.1.1 Least-Squares Regression Revisited

To gain further insight, we consider ridgeless kernel least-squares regression as introduced in Section 11.2. Consider a data sample $(\boldsymbol{x}_j, y_j)_{j=1}^m \subseteq \mathbb{R}^d \times \mathbb{R}$ generated by some ground-truth function $f$, i.e.

$$y_j = f(\boldsymbol{x}_j) \quad \text{for } j = 1, \ldots, m.$$

Let $\phi_j \colon \mathbb{R}^d \to \mathbb{R}$, $j \in \mathbb{N}$, be a sequence of *ansatz functions*. For $n \in \mathbb{N}$, we wish to fit a function $\boldsymbol{x} \mapsto \sum_{i=1}^n w_i \phi_i(\boldsymbol{x})$ to the data using linear least-squares. To this end, we introduce the feature map

$$\mathbb{R}^d \ni \boldsymbol{x} \mapsto \boldsymbol{\phi}(\boldsymbol{x}) := (\phi_1(\boldsymbol{x}), \ldots, \phi_n(\boldsymbol{x}))^\top \in \mathbb{R}^n.$$

The goal is to determine coefficients $\boldsymbol{w} \in \mathbb{R}^n$ minimizing the empirical risk

$$\widehat{\mathcal{R}}_S(\boldsymbol{w}) = \frac{1}{m} \sum_{j=1}^m \left(\sum_{i=1}^n w_i \phi_i(\boldsymbol{x}_j) - y_j\right)^2 = \frac{1}{m} \sum_{j=1}^m (\langle \boldsymbol{\phi}(\boldsymbol{x}_j), \boldsymbol{w} \rangle - y_j)^2.$$

With

$$\boldsymbol{A}_n := \begin{pmatrix} \phi_1(\boldsymbol{x}_1) & \cdots & \phi_n(\boldsymbol{x}_1) \\ \vdots & \ddots & \vdots \\ \phi_1(\boldsymbol{x}_m) & \cdots & \phi_n(\boldsymbol{x}_m) \end{pmatrix} = \begin{pmatrix} \boldsymbol{\phi}(\boldsymbol{x}_1)^\top \\ \vdots \\ \boldsymbol{\phi}(\boldsymbol{x}_m)^\top \end{pmatrix} \in \mathbb{R}^{m \times n}$$

and $\boldsymbol{y} = (y_1, \ldots, y_m)^\top$ it holds

$$\widehat{\mathcal{R}}_S(\boldsymbol{w}) = \frac{1}{m} \lVert \boldsymbol{A}_n \boldsymbol{w} - \boldsymbol{y} \rVert^2.$$

As discussed in Sections 11.1--11.2, a unique minimizer of this problem only exists if $\boldsymbol{A}_n$ has rank $n$. For a minimizer $\boldsymbol{w}_n$, the fitted function reads

$$f_n(x) := \sum_{j=1}^n w_{n,j} \phi_j(x).$$

We are interested in the behavior of the $f_n$ as a function of $n$ (the number of ansatz functions/parameters of our model), and distinguish between two cases:

- *Underparameterized:* If $n < m$ we have fewer parameters $n$ than training points $m$. For the least squares problem of minimizing $\widehat{\mathcal{R}}_S$, this means that there are more conditions $m$ than free parameters $n$. Thus, in general, we cannot interpolate the data, and we have $\min_{\boldsymbol{w} \in \mathbb{R}^n} \widehat{\mathcal{R}}_S(\boldsymbol{w}) > 0$.

- *Overparameterized:* If $n \geq m$, then we have at least as many parameters $n$ as training points $m$. If the $\boldsymbol{x}_j$ and the $\phi_j$ are such that $\boldsymbol{A}_n \in \mathbb{R}^{m \times n}$ has full rank $m$, then there exists $\boldsymbol{w}$ such that $\widehat{\mathcal{R}}_S(\boldsymbol{w}) = 0$. If $n > m$, then $\boldsymbol{A}_n$ necessarily has a nontrivial kernel, and there exist infinitely many parameter choices $\boldsymbol{w}$ that yield zero empirical risk $\widehat{\mathcal{R}}_S$. Some of them lead to better, and some lead to worse prediction functions $f_n$.

In the overparameterized case, there exist many minimizers of $\widehat{\mathcal{R}}_S$. The training algorithm we use to compute a minimizer determines the type of prediction function $f_n$ we obtain. We argued in Chapter 11 that for suitable initialization, gradient descent converges towards the **minimal norm minimizer**

$$\boldsymbol{w}_{n,*} = \operatorname{argmin}_{\boldsymbol{w} \in M} \lVert \boldsymbol{w} \rVert \in \mathbb{R}^n, \qquad M = \lbrace \boldsymbol{w} \in \mathbb{R}^n \mid \widehat{\mathcal{R}}_S(\boldsymbol{w}) \leq \widehat{\mathcal{R}}_S(\boldsymbol{v}) \;\forall\, \boldsymbol{v} \in \mathbb{R}^n \rbrace.$$

#### 15.1.2 An Example

We consider a concrete example. We plot a set of 40 ansatz functions $\phi_1, \ldots, \phi_{40}$, which are drawn from a Gaussian process. The Runge function $f$ is used with $m = 18$ equispaced training data points. We then fit a function in $\text{span}\lbrace \phi_1, \ldots, \phi_n \rbrace$ via the minimal norm minimizer. The result for different numbers of ansatz functions $n$:

- $n = 2$: The model can only represent functions in $\text{span}\lbrace \phi_1, \phi_2 \rbrace$. It is not yet expressive enough to give a meaningful approximation of $f$.

- $n = 15$: The model has sufficient expressivity to capture the main characteristics of $f$. Since $n = 15 < 18 = m$, it is not yet able to interpolate the data. Thus it allows to strike a good balanced between the approximation and generalization error, which corresponds to the scenario discussed in Chapter 14.

- $n = 18$: We are at the interpolation threshold. The model is capable of interpolating the data, and there is a unique $\boldsymbol{w}$ such that $\widehat{\mathcal{R}}_S(\boldsymbol{w}) = 0$. Yet, in between data points the behavior of the predictor $f_{18}$ seems erratic, and displays strong oscillations. This is referred to as **overfitting**, and is to be expected due to our analysis in Chapter 14; while the approximation error at the data points has improved compared to the case $n = 15$, the generalization error has gotten worse.

- $n = 40$: This is the overparameterized regime, where we have significantly more parameters than data points. Our prediction $f_{40}$ interpolates the data and appears to be the best overall approximation to $f$ so far, due to a "good" choice of minimizer of $\widehat{\mathcal{R}}_S$, namely the minimal norm minimizer. We also note that, while quite good, the fit is not perfect. We cannot expect significant improvement in performance by further increasing $n$, since at this point the main limiting factor is the amount of available data.

The $L^2$-error $\lVert f - f_n \rVert_{L^2([-1,1])}$ over $n$ exhibits the characteristic **double descent curve**, where the error initially decreases and then peaks at the interpolation threshold, which is marked by $n = m$. Afterwards, in the overparameterized regime, it starts to decrease again. The Euclidean norm of the coefficient vector $\lVert \boldsymbol{w}_{n,*} \rVert$ also peaks at the interpolation threshold.

We emphasize that the precise nature of the convergence curves depends strongly on various factors, such as the distribution and number of training points $m$, the ground truth $f$, and the choice of ansatz functions $\phi_j$. For overparametrization ($n \gg m$), the precise choice of $n$ is less critical, potentially making the algorithm more stable in this regime.

### 15.2 Size of Weights

We observed that the norm of the coefficients $\lVert \boldsymbol{w}_{n,*} \rVert$ exhibits similar behavior to the $L^2$-error, peaking at the interpolation threshold $n = 18$. In machine learning, large weights are usually undesirable, as they are associated with large derivatives or oscillatory behavior. This is evident in the example shown for $n = 18$. Assuming that the data was generated by a "smooth" function $f$, e.g. a function with moderate Lipschitz constant, these large derivatives of the prediction function could lead to poor generalization. Such a smoothness assumption about $f$ may or may not be satisfied. However, if $f$ is not smooth, there is little hope of accurately recovering $f$ from limited data (see the discussion in Section 9.2).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15.1</span><span class="math-callout__name">(Monotonicity of Minimal Norm Solutions)</span></p>

Assume that $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_m$ and the $(\phi_j)_{j \in \mathbb{N}}$ are such that $\boldsymbol{A}_n$ in (15.1.2) has full rank $n$ for all $n \leq m$. Given $\boldsymbol{y} \in \mathbb{R}^m$, denote by $\boldsymbol{w}_{n,*}(\boldsymbol{y})$ the vector in (15.1.5). Then

$$n \mapsto \sup_{\lVert \boldsymbol{y} \rVert = 1} \lVert \boldsymbol{w}_{n,*}(\boldsymbol{y}) \rVert \quad \text{is monotonically} \quad \begin{cases} \text{increasing} & \text{for } n < m, \\ \text{decreasing} & \text{for } n \geq m. \end{cases}$$

</div>

*Proof.* We start with the case $n \geq m$. By assumption $\boldsymbol{A}_m$ has full rank $m$, and thus $\boldsymbol{A}_n$ has rank $m$ for all $n \geq m$. In particular, there exists $\boldsymbol{w}_n \in \mathbb{R}^n$ such that $\boldsymbol{A}_n \boldsymbol{w}_n = \boldsymbol{y}$. Now fix $\boldsymbol{y} \in \mathbb{R}^m$ and let $\boldsymbol{w}_n$ be any such vector. Then $\boldsymbol{w}_{n+1} := (\boldsymbol{w}_n, 0) \in \mathbb{R}^{n+1}$ satisfies $\boldsymbol{A}_{n+1} \boldsymbol{w}_{n+1} = \boldsymbol{y}$ and $\lVert \boldsymbol{w}_{n+1} \rVert = \lVert \boldsymbol{w}_n \rVert$. Thus necessarily $\lVert \boldsymbol{w}_{n+1,*} \rVert \leq \lVert \boldsymbol{w}_{n,*} \rVert$ for the minimal norm solutions. Since this holds for every $\boldsymbol{y}$, we obtain the statement for $n \geq m$.

Now let $n < m$. Recall that the minimal norm solution can be written through the pseudo inverse

$$\boldsymbol{w}_{n,*}(\boldsymbol{y}) = \boldsymbol{A}_n^\dagger \boldsymbol{y},$$

where

$$\boldsymbol{A}_n^\dagger = \boldsymbol{V}_n \begin{pmatrix} s_{n,1}^{-1} & & \\ & \ddots & \\ & & s_{n,n}^{-1} \\ & \boldsymbol{0} & \end{pmatrix} \boldsymbol{U}_n^\top \in \mathbb{R}^{n \times m}$$

and $\boldsymbol{A}_n = \boldsymbol{U}_n \boldsymbol{\Sigma}_n \boldsymbol{V}_n^\top$ is the singular value decomposition. The matrix $\boldsymbol{\Sigma}_n$ contains the singular values $s_{n,1} \geq \cdots \geq s_{n,n} > 0$ of $\boldsymbol{A}_n$. Since $\boldsymbol{V}_n$ and $\boldsymbol{U}_n$ are orthogonal matrices, we have

$$\sup_{\lVert \boldsymbol{y} \rVert = 1} \lVert \boldsymbol{w}_{n,*}(\boldsymbol{y}) \rVert = \sup_{\lVert \boldsymbol{y} \rVert = 1} \lVert \boldsymbol{A}_n^\dagger \boldsymbol{y} \rVert = s_{n,n}^{-1}.$$

Finally, since the minimal singular value $s_{n,n}$ of $\boldsymbol{A}_n$ can be written as

$$s_{n,n} = \inf_{\substack{\boldsymbol{x} \in \mathbb{R}^n \\ \lVert \boldsymbol{x} \rVert = 1}} \lVert \boldsymbol{A}_n \boldsymbol{x} \rVert \geq \inf_{\substack{\boldsymbol{x} \in \mathbb{R}^{n+1} \\ \lVert \boldsymbol{x} \rVert = 1}} \lVert \boldsymbol{A}_{n+1} \boldsymbol{x} \rVert = s_{n+1, n+1},$$

we observe that $n \mapsto s_{n,n}$ is monotonically decreasing for $n \leq m$. This concludes the proof. $\square$

### 15.3 Theoretical Justification

Let us now examine one possible explanation of the double descent phenomenon for neural networks. The key assumption underlying our analysis is that large overparameterized neural networks tend to be Lipschitz continuous with a Lipschitz constant independent of the size. This is a consequence of neural networks typically having relatively small weights. To motivate this, let us consider the class of neural networks $\mathcal{N}(\sigma; \mathcal{A}, B)$ for an architecture $\mathcal{A}$ of depth $d \in \mathbb{N}$ and width $L \in \mathbb{N}$. If $\sigma$ is $C_\sigma$-Lipschitz continuous with $C_\sigma \geq 1$, such that $B \leq c_B \cdot (d C_\sigma)^{-1}$ for some $c_B > 0$, then by Lemma 13.2

$$\mathcal{N}(\sigma; \mathcal{A}, B) \subseteq \text{Lip}_{c_B}(\mathbb{R}^{d_0}).$$

An assumption of the type $B \leq c_B \cdot (d C_\sigma)^{-1}$, i.e. a scaling of the weights by the reciprocal $1/d$ of the width, is not unreasonable in practice: Standard initialization schemes such as LeCun or He initialization use random weights with variance scaled inverse proportional to the input dimension of each layer. Moreover, as we saw in Chapter 11, for very wide neural networks, the weights do not move significantly from their initialization during training. Additionally, many training routines use regularization terms on the weights, thereby encouraging the optimization routine to find small weights.

We study the generalization capacity of Lipschitz functions through the covering-number-based learning results of Chapter 14. The set $\text{Lip}_C(\Omega)$ of $C$-Lipschitz functions on a compact $d$-dimensional Euclidean domain $\Omega$ has covering numbers bounded according to

$$\log(\mathcal{G}(\text{Lip}_C(\Omega), \varepsilon, L^\infty)) \leq C_{\text{cov}} \cdot \left(\frac{C}{\varepsilon}\right)^d \qquad \text{for all } \varepsilon > 0$$

for some constant $C_{\text{cov}}$ independent of $\varepsilon > 0$.

As a result of these considerations, we can identify two regimes:

- *Standard regime:* For small neural network size $n_\mathcal{A}$, we consider neural networks as a set parameterized by $n_\mathcal{A}$ parameters. As we have seen before, this yields a bound on the generalization error that scales linearly with $n_\mathcal{A}$. As long as $n_\mathcal{A}$ is small in comparison to the number of samples, we can expect good generalization by Theorem 14.15.

- *Overparameterized regime:* For large neural network size $n_\mathcal{A}$ and small weights, we consider neural networks as a subset of $\text{Lip}_C(\Omega)$ for a constant $C > 0$. This set has a covering number bound that is independent of the number of parameters $n_\mathcal{A}$.

Choosing the better of the two generalization bounds for each regime yields the following result. Recall that $\mathcal{N}^*(\sigma; \mathcal{A}, B)$ denotes all neural networks in $\mathcal{N}(\sigma; \mathcal{A}, B)$ with a range contained in $[-1, 1]$ (see (14.5.1)).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 15.2</span><span class="math-callout__name">(Generalization in Two Regimes)</span></p>

Let $C$, $C_\mathcal{L} > 0$ and let $\mathcal{L} \colon [-1,1] \times [-1,1] \to \mathbb{R}$ be $C_\mathcal{L}$-Lipschitz. Further, let $\mathcal{A} = (d_0, d_1, \ldots, d_{L+1}) \in \mathbb{N}^{L+2}$, let $\sigma \colon \mathbb{R} \to \mathbb{R}$ be $C_\sigma$-Lipschitz continuous with $C_\sigma \geq 1$, and $\lvert \sigma(x) \rvert \leq C_\sigma \lvert x \rvert$ for all $x \in \mathbb{R}$, and let $B > 0$.

Then, there exist $c_1$, $c_2 > 0$, such that for every $m \in \mathbb{N}$ and every distribution $\mathcal{D}$ on $[-1,1]^{d_0} \times [-1,1]$ it holds with probability at least $1 - \delta$ over $S \sim \mathcal{D}^m$ that for all $\Phi \in \mathcal{N}^*(\sigma; \mathcal{A}, B) \cap \text{Lip}_C([-1,1]^{d_0})$

$$\lvert \mathcal{R}(\Phi) - \widehat{\mathcal{R}}_S(\Phi) \rvert \leq g(\mathcal{A}, C_\sigma, B, m) + 4C_\mathcal{L} \sqrt{\frac{\log(4/\delta)}{m}},$$

where

$$g(\mathcal{A}, C_\sigma, B, m) = \min\left\lbrace c_1 \sqrt{\frac{n_\mathcal{A} \log(n_\mathcal{A} \lceil \sqrt{m} \rceil) + L n_\mathcal{A} \log(d_{\max})}{m}},\; c_2 m^{-\frac{1}{2 + d_0}} \right\rbrace.$$

</div>

*Proof.* Applying Theorem 14.11 with $\alpha = 1/(2 + d_0)$ and the covering number bound for Lipschitz functions, we obtain that with probability at least $1 - \delta/2$ it holds for all $\Phi \in \text{Lip}_C([-1,1]^{d_0})$

$$\lvert \mathcal{R}(\Phi) - \widehat{\mathcal{R}}_S(\Phi) \rvert \leq 4C_\mathcal{L} \sqrt{\frac{C_{\text{cov}} C^{d_0} (m^{d_0/(d_0+2)-1}) + \log(4/\delta)}{m}} + \frac{2C_\mathcal{L}}{m^\alpha} = \frac{(4C_\mathcal{L}\sqrt{C_{\text{cov}} C^{d_0}} + 2C_\mathcal{L})}{m^\alpha} + 4C_\mathcal{L}\sqrt{\frac{\log(4/\delta)}{m}}.$$

In addition, Theorem 14.15 yields that with probability at least $1 - \delta/2$ it holds for all $\Phi \in \mathcal{N}^*(\sigma; \mathcal{A}, B)$

$$\lvert \mathcal{R}(\Phi) - \widehat{\mathcal{R}}_S(\Phi) \rvert \leq 6C_\mathcal{L} \sqrt{\frac{n_\mathcal{A} \log(\lceil n_\mathcal{A} \sqrt{m} \rceil) + L n_\mathcal{A} \log(\lceil 2C_\sigma B d_{\max} \rceil)}{m}} + 4C_\mathcal{L}\sqrt{\frac{\log(4/\delta)}{m}}.$$

Then, for $\Phi \in \mathcal{N}^*(\sigma; \mathcal{A}, B) \cap \text{Lip}_C([-1,1]^{d_0})$ the minimum of both upper bounds holds with probability at least $1 - \delta$. $\square$

The two regimes in Theorem 15.2 correspond to the two terms comprising the minimum in the definition of $g(\mathcal{A}, C_\sigma, B, m)$. The first term increases with $n_\mathcal{A}$ while the second is constant. In the first regime, where the first term is smaller, the generalization gap $\lvert \mathcal{R}(\Phi) - \widehat{\mathcal{R}}_S(\Phi) \rvert$ increases with $n_\mathcal{A}$.

In the second regime, where the second term is smaller, the generalization gap is constant with $n_\mathcal{A}$. Moreover, it is reasonable to assume that the empirical risk $\widehat{\mathcal{R}}_S$ will decrease with increasing number of parameters $n_\mathcal{A}$.

By the bound

$$\mathcal{R}(\Phi) \leq \widehat{\mathcal{R}}_S + g(\mathcal{A}, C_\sigma, B, m) + 4C_\mathcal{L}\sqrt{\frac{\log(4/\delta)}{m}},$$

in the second regime, this upper bound is monotonically decreasing. In the first regime it may both decrease and increase. In some cases, this behavior can lead to an upper bound on the risk resembling the double descent curve.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 15.3</span></p>

Theorem 15.2 assumes $C$-Lipschitz continuity of the neural networks. As we saw in Sections 15.1.2 and 15.2, this assumption may not hold near the interpolation threshold. Hence, Theorem 15.2 likely gives a too optimistic upper bound near the interpolation threshold.

</div>

---

## Chapter 16: Robustness and Adversarial Examples

How sensitive is the output of a neural network to small changes in its input? Real-world observations of trained neural networks often reveal that even barely noticeable modifications of the input can lead to drastic variations in the network's predictions. This phenomenon, where a correctly classified image is misclassified after a slight perturbation, is termed an **adversarial example**.

In practice, such behavior is highly undesirable: it indicates that the learning algorithm might not be very reliable and poses a potential security risk. This chapter describes the basic mathematical principles behind adversarial examples and investigates simple conditions under which they might or might not occur. For simplicity, the treatment is restricted to a binary classification problem.

### 16.1 Adversarial Examples

We consider the problem of assigning a label $y \in \lbrace -1, 1 \rbrace$ to a vector $\boldsymbol{x} \in \mathbb{R}^d$. The relation between $\boldsymbol{x}$ and $y$ is described by a distribution $\mathcal{D}$ on $\mathbb{R}^d \times \lbrace -1, 1 \rbrace$. In particular, for a given $\boldsymbol{x}$, both values $-1$ and $1$ could have positive probability, i.e. the label is not necessarily deterministic. We let

$$D_{\boldsymbol{x}} := \lbrace \boldsymbol{x} \in \mathbb{R}^d \mid \exists y \text{ s.t. } (\boldsymbol{x}, y) \in \text{supp}(\mathcal{D}) \rbrace$$

and refer to $D_{\boldsymbol{x}}$ as the **feature support**.

Throughout this chapter we denote by

$$g \colon \mathbb{R}^d \to \lbrace -1, 0, 1 \rbrace$$

a fixed *ground-truth classifier*, satisfying

$$\mathbb{P}[y = g(\boldsymbol{x}) \mid \boldsymbol{x}] \geq \mathbb{P}[y = -g(\boldsymbol{x}) \mid \boldsymbol{x}] \quad \text{for all } \boldsymbol{x} \in D_{\boldsymbol{x}}.$$

We allow $g$ to take the value $0$, which corresponds to an additional label for nonrelevant or nonsensical input data $\boldsymbol{x}$. We refer to $g^{-1}(0)$ as the **nonrelevant class**. The ground truth $g$ is interpreted as how a human would classify the data.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 16.1</span></p>

We wish to classify whether an image shows a panda ($y = 1$) or a wombat ($y = -1$). Consider three images $\boldsymbol{x}_1$, $\boldsymbol{x}_2$, $\boldsymbol{x}_3$. The first image $\boldsymbol{x}_1$ is a photograph of a panda. Together with a label $y$, it can be interpreted as a draw $(\boldsymbol{x}_1, y)$ from a distribution of images $\mathcal{D}$, i.e. $\boldsymbol{x}_1 \in D_{\boldsymbol{x}}$ and $g(\boldsymbol{x}_1) = 1$. The second image $\boldsymbol{x}_2$ displays noise and corresponds to nonrelevant data, so $\boldsymbol{x}_2 \in D_{\boldsymbol{x}}^c$ and $g(\boldsymbol{x}_2) = 0$. The third (perturbed) image $\boldsymbol{x}_3$ also belongs to $D_{\boldsymbol{x}}^c$, as it is not a photograph but a noise-corrupted version of $\boldsymbol{x}_1$. Nonetheless, it is *not* nonrelevant, as a human would classify it as a panda. Thus $g(\boldsymbol{x}_3) = 1$.

</div>

In addition to the ground truth $g$, we denote by

$$h \colon \mathbb{R}^d \to \lbrace -1, 1 \rbrace$$

some trained classifier.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 16.2</span><span class="math-callout__name">(Adversarial Example)</span></p>

Let $g \colon \mathbb{R}^d \to \lbrace -1, 0, 1 \rbrace$ be the ground-truth classifier, let $h \colon \mathbb{R}^d \to \lbrace -1, 1 \rbrace$ be a classifier, and let $\lVert \cdot \rVert_*$ be a norm on $\mathbb{R}^d$. For $\boldsymbol{x} \in \mathbb{R}^d$ and $\delta > 0$, we call $\boldsymbol{x}' \in \mathbb{R}^d$ an **adversarial example** to $\boldsymbol{x} \in \mathbb{R}^d$ with perturbation $\delta$, if and only if

1. $\lVert \boldsymbol{x}' - \boldsymbol{x} \rVert_* \leq \delta$,
2. $g(\boldsymbol{x}) g(\boldsymbol{x}') > 0$,
3. $h(\boldsymbol{x}) = g(\boldsymbol{x})$ and $h(\boldsymbol{x}') \neq g(\boldsymbol{x}')$.

</div>

In words, $\boldsymbol{x}'$ is an adversarial example to $\boldsymbol{x}$ with perturbation $\delta$, if (i) the distance of $\boldsymbol{x}$ and $\boldsymbol{x}'$ is at most $\delta$, (ii) $\boldsymbol{x}$ and $\boldsymbol{x}'$ belong to the same (not nonrelevant) class according to the ground-truth classifier, and (iii) the classifier $h$ correctly classifies $\boldsymbol{x}$ but misclassifies $\boldsymbol{x}'$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 16.3</span></p>

The concept of a ground-truth classifier $g$ differs from a minimizer of the Bayes risk (14.1.1) for two reasons. First, we allow for an additional label $0$ corresponding to the nonrelevant class, which does not exist for the data generating distribution $\mathcal{D}$. Second, $g$ should correctly classify points *outside of* $D_{\boldsymbol{x}}$; small perturbations of images as we find them in adversarial examples are not regular images in $D_{\boldsymbol{x}}$. Nonetheless, a human classifier can still classify these images, and $g$ models this property of human classification.

</div>

### 16.2 Bayes Classifier

At first sight, an adversarial example seems to be no more than a misclassified sample. These exist if the model does not generalize well. To avoid edge cases, we assume that for all $\boldsymbol{x} \in D_{\boldsymbol{x}}$

$$\text{either} \quad \mathbb{P}[y = 1 \mid \boldsymbol{x}] > \mathbb{P}[y = -1 \mid \boldsymbol{x}] \quad \text{or} \quad \mathbb{P}[y = 1 \mid \boldsymbol{x}] < \mathbb{P}[y = -1 \mid \boldsymbol{x}]$$

so that $g(\boldsymbol{x})$ is uniquely defined for $\boldsymbol{x} \in D_{\boldsymbol{x}}$. We say that the distribution **exhausts the domain** if $D_{\boldsymbol{x}} \cup g^{-1}(0) = \mathbb{R}^d$. This means that every point is either in the feature support $D_{\boldsymbol{x}}$ or it belongs to the nonrelevant class. Moreover, we say that $h$ is a **Bayes classifier** if

$$\mathbb{P}[h(\boldsymbol{x}) \mid \boldsymbol{x}] \geq \mathbb{P}[-h(\boldsymbol{x}) \mid \boldsymbol{x}] \quad \text{for all } \boldsymbol{x} \in D_{\boldsymbol{x}}.$$

By definition, the ground truth $g$ is a Bayes classifier, and the assumption above ensures that $h$ coincides with $g$ on $D_{\boldsymbol{x}}$ if $h$ is a Bayes classifier. A Bayes classifier minimizes the Bayes risk. With these two notions, we distinguish between four cases:

**(i) Bayes classifier / exhaustive distribution:** If $h$ is a Bayes classifier and the data exhausts the domain, then there are *no adversarial examples*. Every $\boldsymbol{x} \in \mathbb{R}^d$ either belongs to the nonrelevant class or is classified the same by $h$ and $g$.

**(ii) Bayes classifier / non-exhaustive distribution:** If $h$ is a Bayes classifier and the distribution does not exhaust the domain, then *adversarial examples can exist*. Even though $h$ coincides with $g$ on the feature support, adversarial examples can be constructed for data points on the complement of $D_{\boldsymbol{x}} \cup g^{-1}(0)$, which is not empty.

**(iii) Not a Bayes classifier / exhaustive distribution:** The set $D_{\boldsymbol{x}}$ can be covered by the four subdomains

$$C_1 = h^{-1}(1) \cap g^{-1}(1), \quad F_1 = h^{-1}(-1) \cap g^{-1}(1),$$

$$C_{-1} = h^{-1}(-1) \cap g^{-1}(-1), \quad F_{-1} = h^{-1}(1) \cap g^{-1}(-1).$$

If $\text{dist}(C_1 \cap D_{\boldsymbol{x}}, F_1 \cap D_{\boldsymbol{x}})$ or $\text{dist}(C_{-1} \cap D_{\boldsymbol{x}}, F_{-1} \cap D_{\boldsymbol{x}})$ is smaller than $\delta$, then *adversarial examples in the feature support can exist*. However, even for classifiers with incorrect predictions, adversarial examples *do not need to exist*.

**(iv) Not a Bayes classifier / non-exhaustive distribution:** In this case *everything is possible*. Data points and their associated adversarial examples can appear in the feature support, and adversarial examples to elements in the feature support can be created by leaving the feature support.

### 16.3 Affine Classifiers

For linear classifiers, a simple argument shows that the high-dimensionality of the input, common in image classification problems, is a potential cause for the existence of adversarial examples.

A linear classifier is a map of the form

$$\boldsymbol{x} \mapsto \text{sign}(\boldsymbol{w}^\top \boldsymbol{x}) \quad \text{where } \boldsymbol{w}, \boldsymbol{x} \in \mathbb{R}^d.$$

Let

$$\boldsymbol{x}' := \boldsymbol{x} - 2 \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert \frac{\text{sign}(\boldsymbol{w}^\top \boldsymbol{x}) \text{sign}(\boldsymbol{w})}{\lVert \boldsymbol{w} \rVert_1}$$

where $\text{sign}(\boldsymbol{w})$ is understood coordinate-wise. Then $\lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_\infty \leq 2 \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert / \lVert \boldsymbol{w} \rVert_1$ and $\text{sign}(\boldsymbol{w}^\top \boldsymbol{x}') \neq \text{sign}(\boldsymbol{w}^\top \boldsymbol{x})$.

For high-dimensional vectors $\boldsymbol{w}$, $\boldsymbol{x}$ chosen at random but possibly dependently such that $\boldsymbol{w}$ is uniformly distributed on a $d-1$ dimensional sphere, it holds with high probability that

$$\frac{\lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert}{\lVert \boldsymbol{w} \rVert_1} \leq \frac{\lVert \boldsymbol{x} \rVert \lVert \boldsymbol{w} \rVert}{\lVert \boldsymbol{w} \rVert_1} \ll \lVert \boldsymbol{x} \rVert.$$

Thus, if $\boldsymbol{x}$ has a moderate Euclidean norm, the perturbation of $\boldsymbol{x}'$ is likely small for large dimensions. Below we give a sufficient condition for the existence of adversarial examples when both $h$ and the ground truth $g$ are linear classifiers.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 16.4</span><span class="math-callout__name">(Adversarial Examples for Linear Classifiers)</span></p>

Let $\boldsymbol{w}$, $\overline{\boldsymbol{w}} \in \mathbb{R}^d$ be nonzero. For $\boldsymbol{x} \in \mathbb{R}^d$, let $h(\boldsymbol{x}) = \text{sign}(\boldsymbol{w}^\top \boldsymbol{x})$ be a classifier and let $g(\boldsymbol{x}) = \text{sign}(\overline{\boldsymbol{w}}^\top \boldsymbol{x})$ be the ground-truth classifier.

For every $\boldsymbol{x} \in \mathbb{R}^d$ with $h(\boldsymbol{x}) g(\boldsymbol{x}) > 0$ and all $\varepsilon \in (0, \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert)$ such that

$$\frac{\lvert \overline{\boldsymbol{w}}^\top \boldsymbol{x} \rvert}{\lVert \overline{\boldsymbol{w}} \rVert} > \frac{\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert}{\lVert \boldsymbol{w} \rVert} \cdot \frac{\lvert \boldsymbol{w}^\top \overline{\boldsymbol{w}} \rvert}{\lVert \boldsymbol{w} \rVert \lVert \overline{\boldsymbol{w}} \rVert}$$

it holds that

$$\boldsymbol{x}' = \boldsymbol{x} - h(\boldsymbol{x}) \frac{\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert}{\lVert \boldsymbol{w} \rVert^2} \boldsymbol{w}$$

is an adversarial example to $\boldsymbol{x}$ with perturbation $\delta = (\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert) / \lVert \boldsymbol{w} \rVert$.

</div>

Note that $\lbrace \boldsymbol{x} \in \mathbb{R}^d \mid \boldsymbol{w}^\top \boldsymbol{x} = 0 \rbrace$ is the decision boundary of $h$, meaning that points lying on opposite sides of this hyperplane are classified differently by $h$. Due to $\lvert \boldsymbol{w}^\top \overline{\boldsymbol{w}} \rvert \leq \lVert \boldsymbol{w} \rVert \lVert \overline{\boldsymbol{w}} \rVert$, the condition implies that an adversarial example always exists whenever

$$\frac{\lvert \overline{\boldsymbol{w}}^\top \boldsymbol{x} \rvert}{\lVert \overline{\boldsymbol{w}} \rVert} > \frac{\lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert}{\lVert \boldsymbol{w} \rVert}.$$

The left term is the decision margin of $\boldsymbol{x}$ for $g$, i.e. the distance of $\boldsymbol{x}$ to the decision boundary of $g$. The right is the decision margin of $\boldsymbol{x}$ for $h$. Thus we conclude that adversarial examples exist if the decision margin of $\boldsymbol{x}$ for the ground truth $g$ is larger than that for the classifier $h$.

Second, the term $(\boldsymbol{w}^\top \overline{\boldsymbol{w}}) / (\lVert \boldsymbol{w} \rVert \lVert \overline{\boldsymbol{w}} \rVert)$ describes the alignment of the two classifiers. If the classifiers are not aligned, i.e. $\boldsymbol{w}$ and $\overline{\boldsymbol{w}}$ have a large angle between them, then adversarial examples exist even if the margin of the classifier is larger than that of the ground-truth classifier.

Finally, adversarial examples with small perturbation are possible if $\lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert \ll \lVert \boldsymbol{w} \rVert$, i.e. if $\boldsymbol{x}$ is close to the decision boundary of $h$.

*Proof (of Theorem 16.4).* We verify that $\boldsymbol{x}'$ satisfies the conditions of Definition 16.2. Since $h(\boldsymbol{x}) g(\boldsymbol{x}) > 0$, we have

$$g(\boldsymbol{x}) = \text{sign}(\overline{\boldsymbol{w}}^\top \boldsymbol{x}) = \text{sign}(\boldsymbol{w}^\top \boldsymbol{x}) = h(\boldsymbol{x}) \neq 0.$$

First, $\lVert \boldsymbol{x} - \boldsymbol{x}' \rVert = \frac{\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert}{\lVert \boldsymbol{w} \rVert} = \delta.$

Next we show $g(\boldsymbol{x}) g(\boldsymbol{x}') > 0$, i.e. that $(\overline{\boldsymbol{w}}^\top \boldsymbol{x})(\overline{\boldsymbol{w}}^\top \boldsymbol{x}')$ is positive. Plugging in the definition of $\boldsymbol{x}'$:

$$\overline{\boldsymbol{w}}^\top \boldsymbol{x} \left( \overline{\boldsymbol{w}}^\top \boldsymbol{x} - h(\boldsymbol{x}) \frac{\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert}{\lVert \boldsymbol{w} \rVert^2} \overline{\boldsymbol{w}}^\top \boldsymbol{w} \right) = \lvert \overline{\boldsymbol{w}}^\top \boldsymbol{x} \rvert^2 - \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert \frac{\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert}{\lVert \boldsymbol{w} \rVert^2} \lvert \overline{\boldsymbol{w}}^\top \boldsymbol{w} \rvert$$

which is positive by the assumed condition, after dividing by $\lvert \overline{\boldsymbol{w}}^\top \boldsymbol{x} \rvert \lVert \overline{\boldsymbol{w}} \rVert$.

Finally, we check that $h(\boldsymbol{x}') \neq h(\boldsymbol{x})$, i.e. $(\boldsymbol{w}^\top \boldsymbol{x})(\boldsymbol{w}^\top \boldsymbol{x}') < 0$:

$$(\boldsymbol{w}^\top \boldsymbol{x})(\boldsymbol{w}^\top \boldsymbol{x}') = \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert^2 - \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert (\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert) < 0.$$

This completes the proof. $\square$

Theorem 16.4 readily implies the following proposition for *affine* classifiers.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.5</span><span class="math-callout__name">(Adversarial Examples for Affine Classifiers)</span></p>

Let $\boldsymbol{w}$, $\overline{\boldsymbol{w}} \in \mathbb{R}^d$ and $b$, $\overline{b} \in \mathbb{R}$. For $\boldsymbol{x} \in \mathbb{R}^d$ let $h(\boldsymbol{x}) = \text{sign}(\boldsymbol{w}^\top \boldsymbol{x} + b)$ be a classifier and let $g(\boldsymbol{x}) = \text{sign}(\overline{\boldsymbol{w}}^\top \boldsymbol{x} + \overline{b})$ be the ground-truth classifier.

For every $\boldsymbol{x} \in \mathbb{R}^d$ with $\overline{\boldsymbol{w}}^\top \boldsymbol{x} \neq 0$, $h(\boldsymbol{x}) g(\boldsymbol{x}) > 0$, and all $\varepsilon \in (0, \lvert \boldsymbol{w}^\top \boldsymbol{x} + b \rvert)$ such that

$$\frac{\lvert \overline{\boldsymbol{w}}^\top \boldsymbol{x} + \overline{b} \rvert^2}{\lVert \overline{\boldsymbol{w}} \rVert^2 + b^2} > \frac{(\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} + b \rvert)^2}{\lVert \boldsymbol{w} \rVert^2 + b^2} \cdot \frac{(\boldsymbol{w}^\top \overline{\boldsymbol{w}} + b \overline{b})^2}{(\lVert \boldsymbol{w} \rVert^2 + b^2)(\lVert \overline{\boldsymbol{w}} \rVert^2 + \overline{b}^2)}$$

it holds that

$$\boldsymbol{x}' = \boldsymbol{x} - h(\boldsymbol{x}) \frac{\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} + b \rvert}{\lVert \boldsymbol{w} \rVert^2} \boldsymbol{w}$$

is an adversarial example with perturbation $\delta = (\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} + b \rvert) / \lVert \boldsymbol{w} \rVert$ to $\boldsymbol{x}$.

</div>

We now study two cases of linear classifiers which allow for different types of adversarial examples. In both examples, the ground-truth classifier $g \colon \mathbb{R}^d \to \lbrace -1, 1 \rbrace$ is given by $g(\boldsymbol{x}) = \text{sign}(\overline{\boldsymbol{w}}^\top \boldsymbol{x})$ for $\overline{\boldsymbol{w}} \in \mathbb{R}^d$ with $\lVert \overline{\boldsymbol{w}} \rVert = 1$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 16.6</span><span class="math-callout__name">(Bayes Classifier with Non-Exhaustive Distribution)</span></p>

Let $\mathcal{D}$ be the uniform distribution on $\lbrace (\lambda \overline{\boldsymbol{w}}, g(\lambda \overline{\boldsymbol{w}})) \mid \lambda \in [-1, 1] \setminus \lbrace 0 \rbrace \rbrace$. The feature support equals $D_{\boldsymbol{x}} = \lbrace \lambda \overline{\boldsymbol{w}} \mid \lambda \in [-1, 1] \setminus \lbrace 0 \rbrace \rbrace \subseteq \text{span}\lbrace \overline{\boldsymbol{w}} \rbrace$.

Fix $\alpha \in (0, 1)$ and set $\boldsymbol{w} := \alpha \overline{\boldsymbol{w}} + (1 - \alpha) \boldsymbol{v}$ for some $\boldsymbol{v} \in \overline{\boldsymbol{w}}^\perp$ with $\lVert \boldsymbol{v} \rVert = 1$, so that $\lVert \boldsymbol{w} \rVert = 1$. Let $h(\boldsymbol{x}) := \text{sign}(\boldsymbol{w}^\top \boldsymbol{x})$. Then $h(\boldsymbol{x}) = g(\boldsymbol{x})$ for every $\boldsymbol{x} \in D_{\boldsymbol{x}}$, so $h$ is a Bayes classifier. Moreover, $\lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert \leq \alpha \lvert \overline{\boldsymbol{w}}^\top \boldsymbol{x} \rvert$, so the condition of Theorem 16.4 is satisfied. For every $\varepsilon > 0$ it holds that

$$\delta := \frac{\varepsilon + \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert}{\lVert \boldsymbol{w} \rVert} \leq \varepsilon + \alpha.$$

Hence, for $\varepsilon < \lvert \boldsymbol{w}^\top \boldsymbol{x} \rvert$ there exists an adversarial example with perturbation less than $\varepsilon + \alpha$. For small $\alpha$, this corresponds to case (ii) in Section 16.2.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 16.7</span><span class="math-callout__name">(Non-Bayes Classifier with Global Feature Support)</span></p>

Let $\mathcal{D}_{\boldsymbol{x}}$ be a distribution on $\mathbb{R}^d$ with positive Lebesgue density everywhere outside the decision boundary $\text{DB}_g = \lbrace \boldsymbol{x} \mid \overline{\boldsymbol{w}}^\top \boldsymbol{x} = 0 \rbrace$ of $g$. We define $\mathcal{D}$ to be the distribution of $(X, g(X))$ for $X \sim \mathcal{D}_{\boldsymbol{x}}$. Let $\boldsymbol{w} \notin \lbrace \pm \overline{\boldsymbol{w}} \rbrace$, $\lVert \boldsymbol{w} \rVert = 1$ and $h(\boldsymbol{x}) = \text{sign}(\boldsymbol{w}^\top \boldsymbol{x})$. We exclude $\boldsymbol{w} = -\overline{\boldsymbol{w}}$ because in this case every prediction of $h$ is wrong, so no adversarial examples are possible.

By construction the feature support is $D_{\boldsymbol{x}} = \mathbb{R}^d$. Moreover, $h^{-1}(\lbrace -1 \rbrace)$, $h^{-1}(\lbrace 1 \rbrace)$ and $g^{-1}(\lbrace -1 \rbrace)$, $g^{-1}(\lbrace 1 \rbrace)$ are half spaces, which implies that

$$\text{dist}(C_{\pm 1} \cap D_{\boldsymbol{x}}, F_{\pm 1} \cap D_{\boldsymbol{x}}) = \text{dist}(C_{\pm 1}, F_{\pm 1}) = 0.$$

Hence, for every $\delta > 0$ there is a positive probability of observing $\boldsymbol{x}$ to which an adversarial example with perturbation $\delta$ exists. This corresponds to case (iv) in Section 16.2.

</div>

### 16.4 ReLU Neural Networks

So far we discussed classification by affine classifiers. A binary classifier based on a ReLU neural network is a function $\mathbb{R}^d \ni \boldsymbol{x} \mapsto \text{sign}(\Phi(\boldsymbol{x}))$, where $\Phi$ is a ReLU neural network. As noted in the literature, the arguments for affine classifiers (Proposition 16.5) can be applied to the affine pieces of $\Phi$ to show existence of adversarial examples.

Consider a ground-truth classifier $g \colon \mathbb{R}^d \to \lbrace -1, 0, 1 \rbrace$. For each $\boldsymbol{x} \in \mathbb{R}^d$ we define the **geometric margin** of $g$ at $\boldsymbol{x}$ as

$$\mu_g(\boldsymbol{x}) := \text{dist}(\boldsymbol{x}, g^{-1}(\lbrace g(\boldsymbol{x}) \rbrace)^c),$$

i.e. the distance of $\boldsymbol{x}$ to the closest element that is classified differently from $\boldsymbol{x}$. Additionally, we denote the distance of $\boldsymbol{x}$ to the closest adjacent affine piece by

$$\nu_\Phi(\boldsymbol{x}) := \text{dist}(\boldsymbol{x}, A_{\Phi, \boldsymbol{x}}^c),$$

where $A_{\Phi, \boldsymbol{x}}$ is the largest connected region on which $\Phi$ is affine and which contains $\boldsymbol{x}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 16.8</span><span class="math-callout__name">(Adversarial Examples for ReLU Networks)</span></p>

Let $\Phi \colon \mathbb{R}^d \to \mathbb{R}$ and for $\boldsymbol{x} \in \mathbb{R}^d$ let $h(\boldsymbol{x}) = \text{sign}(\Phi(\boldsymbol{x}))$. Denote by $g \colon \mathbb{R}^d \to \lbrace -1, 0, 1 \rbrace$ the ground-truth classifier. Let $\boldsymbol{x} \in \mathbb{R}^d$ and $\varepsilon > 0$ be such that $\nu_\Phi(\boldsymbol{x}) > 0$, $g(\boldsymbol{x}) \neq 0$, $\nabla \Phi(\boldsymbol{x}) \neq 0$ and

$$\mu_g(\boldsymbol{x}), \nu_\Phi(\boldsymbol{x}) > \frac{\varepsilon + \lvert \Phi(\boldsymbol{x}) \rvert}{\lVert \nabla \Phi(\boldsymbol{x}) \rVert}.$$

Then

$$\boldsymbol{x}' := \boldsymbol{x} - h(\boldsymbol{x}) \frac{\varepsilon + \lvert \Phi(\boldsymbol{x}) \rvert}{\lVert \nabla \Phi(\boldsymbol{x}) \rVert^2} \nabla \Phi(\boldsymbol{x})$$

is an adversarial example to $\boldsymbol{x}$ with perturbation $\delta = (\varepsilon + \lvert \Phi(\boldsymbol{x}) \rvert) / \lVert \nabla \Phi(\boldsymbol{x}) \rVert$.

</div>

*Proof.* We show that $\boldsymbol{x}'$ satisfies the properties in Definition 16.2. By construction $\lVert \boldsymbol{x} - \boldsymbol{x}' \rVert \leq \delta$. Since $\mu_g(\boldsymbol{x}) > \delta$ it follows that $g(\boldsymbol{x}) = g(\boldsymbol{x}')$. Moreover, by assumption $g(\boldsymbol{x}) \neq 0$, and thus $g(\boldsymbol{x}) g(\boldsymbol{x}') > 0$.

It only remains to show that $h(\boldsymbol{x}') \neq h(\boldsymbol{x})$. Since $\delta < \nu_\Phi(\boldsymbol{x})$, we have that $\Phi(\boldsymbol{x}) = \nabla \Phi(\boldsymbol{x})^\top \boldsymbol{x} + b$ and $\Phi(\boldsymbol{x}') = \nabla \Phi(\boldsymbol{x})^\top \boldsymbol{x}' + b$ for some $b \in \mathbb{R}$. Therefore,

$$\Phi(\boldsymbol{x}) - \Phi(\boldsymbol{x}') = \nabla \Phi(\boldsymbol{x})^\top (\boldsymbol{x} - \boldsymbol{x}') = h(\boldsymbol{x})(\varepsilon + \lvert \Phi(\boldsymbol{x}) \rvert).$$

Since $h(\boldsymbol{x}) \lvert \Phi(\boldsymbol{x}) \rvert = \Phi(\boldsymbol{x})$ it follows that $\Phi(\boldsymbol{x}') = -h(\boldsymbol{x}) \varepsilon$. Hence, $h(\boldsymbol{x}') = -h(\boldsymbol{x})$, which completes the proof. $\square$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 16.9</span></p>

We look at the key parameters in Theorem 16.8 to understand which factors facilitate adversarial examples:

- **The geometric margin of the ground-truth classifier $\mu_g(\boldsymbol{x})$:** To make the construction possible, we need to be sufficiently far away from points that belong to a different class than $\boldsymbol{x}$ or to the nonrelevant class.
- **The distance to the next affine piece $\nu_\Phi(\boldsymbol{x})$:** Since we are looking for an adversarial example within the same affine piece as $\boldsymbol{x}$, we need this piece to be sufficiently large.
- **The perturbation $\delta$:** The perturbation is given by $(\varepsilon + \lvert \Phi(\boldsymbol{x}) \rvert) / \lVert \nabla \Phi(\boldsymbol{x}) \rVert$, which depends on the classification margin $\lvert \Phi(\boldsymbol{x}) \rvert$ of the ReLU classifier and its sensitivity to inputs $\lVert \nabla \Phi(\boldsymbol{x}) \rVert$. For adversarial examples to be possible, we either want a small classification margin of $\Phi$ or a high sensitivity of $\Phi$ to its inputs.

</div>

### 16.5 Robustness

Having established that adversarial examples can arise in various ways under mild assumptions, we now turn our attention to conditions that prevent their existence.

#### 16.5.1 Global Lipschitz Regularity

We have repeatedly observed in the previous sections that a large value of $\lVert \boldsymbol{w} \rVert$ for linear classifiers $\text{sign}(\boldsymbol{w}^\top \boldsymbol{x})$, or $\lVert \nabla \Phi(\boldsymbol{x}) \rVert$ for ReLU classifiers $\text{sign}(\Phi(\boldsymbol{x}))$, facilitates the occurrence of adversarial examples. Naturally, both these values are upper bounded by the Lipschitz constant of the classifier's inner functions. Consequently, it was stipulated early on that bounding the Lipschitz constant could be an effective measure against adversarial examples.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.10</span><span class="math-callout__name">(Lipschitz Robustness)</span></p>

Let $\Phi \colon \mathbb{R}^d \to \mathbb{R}$ be $C_L$-Lipschitz with $C_L > 0$, and let $s > 0$. Let $h(\boldsymbol{x}) = \text{sign}(\Phi(\boldsymbol{x}))$ be a classifier, and let $g \colon \mathbb{R}^d \to \lbrace -1, 0, 1 \rbrace$ be a ground-truth classifier. Moreover, let $\boldsymbol{x} \in \mathbb{R}^d$ be such that

$$\Phi(\boldsymbol{x}) g(\boldsymbol{x}) \geq s.$$

Then there does not exist an adversarial example to $\boldsymbol{x}$ of perturbation $\delta < s / C_L$.

</div>

*Proof.* Let $\boldsymbol{x} \in \mathbb{R}^d$ satisfy the above and assume that $\lVert \boldsymbol{x}' - \boldsymbol{x} \rVert \leq \delta$. The Lipschitz continuity of $\Phi$ implies $\lvert \Phi(\boldsymbol{x}') - \Phi(\boldsymbol{x}) \rvert < s$. Since $\lvert \Phi(\boldsymbol{x}) \rvert \geq s$ we conclude that $\Phi(\boldsymbol{x}')$ has the same sign as $\Phi(\boldsymbol{x})$, which shows that $\boldsymbol{x}'$ cannot be an adversarial example to $\boldsymbol{x}$. $\square$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 16.11</span></p>

As we have seen in Lemma 13.2, we can bound the Lipschitz constant of ReLU neural networks by restricting the magnitude and number of their weights and the number of layers. There has been some criticism of results of this form, since an assumption on the Lipschitz constant may potentially restrict the capabilities of the neural network too much.

</div>

We next present a result showing that under assumptions on the training set, there exists a neural network that classifies the training set correctly but does not allow for adversarial examples within the training set.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 16.12</span><span class="math-callout__name">(Robust Interpolation)</span></p>

Let $m \in \mathbb{N}$, let $g \colon \mathbb{R}^d \to \lbrace -1, 0, 1 \rbrace$ be a ground-truth classifier, and let $(\boldsymbol{x}_i, g(\boldsymbol{x}_i))_{i=1}^m \in (\mathbb{R}^d \times \lbrace -1, 1 \rbrace)^m$. Assume that

$$\sup_{i \neq j} \frac{\lvert g(\boldsymbol{x}_i) - g(\boldsymbol{x}_j) \rvert}{\lVert \boldsymbol{x}_i - \boldsymbol{x}_j \rVert} =: \widetilde{M} > 0.$$

Then there exists a ReLU neural network $\Phi$ with $\text{depth}(\Phi) = O(\log(m))$ and $\text{width}(\Phi) = O(dm)$ such that for all $i = 1, \ldots, m$

$$\text{sign}(\Phi(\boldsymbol{x}_i)) = g(\boldsymbol{x}_i)$$

and there is no adversarial example of perturbation $\delta = 1/\widetilde{M}$ to $\boldsymbol{x}_i$.

</div>

*Proof.* The result follows directly from Theorem 9.6 and Proposition 16.10. $\square$

#### 16.5.2 Local Regularity

One issue with upper bounds involving global Lipschitz constants such as those in Proposition 16.10, is that these bounds may be quite large for deep neural networks. For example, the upper bound given in Lemma 13.2 is

$$\lVert \Phi(\boldsymbol{x}) - \Phi(\boldsymbol{x}') \rVert_\infty \leq C_\sigma^L \cdot (B d_{\max})^{L+1} \lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_\infty$$

which grows exponentially with the depth of the neural network. However, in practice this bound may be pessimistic, and locally the neural network might have significantly smaller gradients than the global Lipschitz constant.

Because of this, it is reasonable to study results preventing adversarial examples under *local* Lipschitz bounds.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 16.13</span><span class="math-callout__name">(Local Lipschitz Robustness)</span></p>

Let $h \colon \mathbb{R}^d \to \lbrace -1, 1 \rbrace$ be a classifier of the form $h(\boldsymbol{x}) = \text{sign}(\Phi(\boldsymbol{x}))$ and let $g \colon \mathbb{R}^d \to \lbrace -1, 0, 1 \rbrace$ be the ground-truth classifier. Let $\boldsymbol{x} \in \mathbb{R}^d$ satisfy $g(\boldsymbol{x}) \neq 0$, and set

$$\alpha := \max_{R > 0} \min \left\lbrace \Phi(\boldsymbol{x}) g(\boldsymbol{x}) \Big/ \sup_{\substack{\lVert \boldsymbol{y} - \boldsymbol{x} \rVert_\infty \leq R \\ \boldsymbol{y} \neq \boldsymbol{x}}} \frac{\lvert \Phi(\boldsymbol{y}) - \Phi(\boldsymbol{x}) \rvert}{\lVert \boldsymbol{x} - \boldsymbol{y} \rVert_\infty},\; R \right\rbrace,$$

where the minimum is understood to be $R$ in case the supremum is zero. Then there are no adversarial examples to $\boldsymbol{x}$ with perturbation $\delta < \alpha$.

</div>

*Proof.* Assume, towards a contradiction, that for $0 < \delta < \alpha$, there exists an adversarial example $\boldsymbol{x}'$ to $\boldsymbol{x}$ with perturbation $\delta$. If the supremum is zero, then $\Phi$ is constant on a ball of radius $R$ around $\boldsymbol{x}$, so $h(\boldsymbol{x}') = h(\boldsymbol{x})$ and $\boldsymbol{x}'$ cannot be an adversarial example.

Now assume the supremum is not zero. It holds by the definition of $\alpha$ for $\delta < R$ that

$$\delta < \Phi(\boldsymbol{x}) g(\boldsymbol{x}) \Big/ \sup_{\substack{\lVert \boldsymbol{y} - \boldsymbol{x} \rVert_\infty \leq R \\ \boldsymbol{y} \neq \boldsymbol{x}}} \frac{\lvert \Phi(\boldsymbol{y}) - \Phi(\boldsymbol{x}) \rvert}{\lVert \boldsymbol{x} - \boldsymbol{y} \rVert_\infty}.$$

Moreover,

$$\lvert \Phi(\boldsymbol{x}') - \Phi(\boldsymbol{x}) \rvert \leq \sup_{\substack{\lVert \boldsymbol{y} - \boldsymbol{x} \rVert_\infty \leq R \\ \boldsymbol{y} \neq \boldsymbol{x}}} \frac{\lvert \Phi(\boldsymbol{y}) - \Phi(\boldsymbol{x}) \rvert}{\lVert \boldsymbol{x} - \boldsymbol{y} \rVert_\infty} \lVert \boldsymbol{x} - \boldsymbol{x}' \rVert_\infty < \Phi(\boldsymbol{x}) g(\boldsymbol{x}).$$

It follows that $g(\boldsymbol{x}) \Phi(\boldsymbol{x}') = g(\boldsymbol{x}) \Phi(\boldsymbol{x}) + g(\boldsymbol{x})(\Phi(\boldsymbol{x}') - \Phi(\boldsymbol{x})) \geq g(\boldsymbol{x}) \Phi(\boldsymbol{x}) - \lvert \Phi(\boldsymbol{x}') - \Phi(\boldsymbol{x}) \rvert > 0$. This rules out $\boldsymbol{x}'$ as an adversarial example. $\square$

The supremum in the definition of $\alpha$ is bounded by the Lipschitz constant of $\Phi$ on $B_R(\boldsymbol{x})$. Thus Theorem 16.13 depends only on the local Lipschitz constant of $\Phi$. One criticism of this result is that the computation of $\alpha$ is potentially prohibitive. We next show a different result, for which the assumptions can immediately be checked by applying a simple algorithm.

For a continuous function $\Phi \colon \mathbb{R}^d \to \mathbb{R}$ and for $\boldsymbol{x} \in \mathbb{R}^d$ and $\delta > 0$ we define

$$z^{\delta, \max} := \max \lbrace \Phi(\boldsymbol{y}) \mid \lVert \boldsymbol{y} - \boldsymbol{x} \rVert_\infty \leq \delta \rbrace$$

$$z^{\delta, \min} := \min \lbrace \Phi(\boldsymbol{y}) \mid \lVert \boldsymbol{y} - \boldsymbol{x} \rVert_\infty \leq \delta \rbrace.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.14</span><span class="math-callout__name">(Robustness via Local Extrema)</span></p>

Let $h \colon \mathbb{R}^d \to \lbrace -1, 1 \rbrace$ be a classifier of the form $h(\boldsymbol{x}) = \text{sign}(\Phi(\boldsymbol{x}))$ and $g \colon \mathbb{R}^d \to \lbrace -1, 0, 1 \rbrace$ be the ground-truth classifier. Let $\boldsymbol{x}$ be such that $h(\boldsymbol{x}) = g(\boldsymbol{x})$. Then $\boldsymbol{x}$ does not have an adversarial example of perturbation $\delta$ if $z^{\delta, \max} z^{\delta, \min} > 0$.

</div>

*Proof.* The proof is immediate, since $z^{\delta, \max} z^{\delta, \min} > 0$ implies that all points in a $\delta$ neighborhood of $\boldsymbol{x}$ are classified the same. $\square$

To apply Proposition 16.14, we only have to compute $z^{\delta, \max}$ and $z^{\delta, \min}$. If $\Phi$ is a neural network, then $z^{\delta, \max}$, $z^{\delta, \min}$ can be approximated by a computation similar to a forward pass of $\Phi$. Denote by $\lvert \boldsymbol{A} \rvert$ the matrix obtained by taking the absolute value of each entry of the matrix $\boldsymbol{A}$. Additionally, we define

$$\boldsymbol{A}^+ = (\lvert \boldsymbol{A} \rvert + \boldsymbol{A}) / 2 \quad \text{and} \quad \boldsymbol{A}^- = (\lvert \boldsymbol{A} \rvert - \boldsymbol{A}) / 2.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 3</span><span class="math-callout__name">(Computing $\Phi(\boldsymbol{x})$, $z^{\delta, \max}$ and $z^{\delta, \min}$)</span></p>

**Input:** weight matrices $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$ and bias vectors $\boldsymbol{b}^{(\ell)} \in \mathbb{R}^{d_{\ell+1}}$ for $\ell = 0, \ldots, L$ with $d_{L+1} = 1$, monotonous activation function $\sigma$, input vector $\boldsymbol{x} \in \mathbb{R}^{d_0}$, neighborhood size $\delta > 0$

**Output:** Bounds for $z^{\delta, \max}$ and $z^{\delta, \min}$

$\boldsymbol{x}^{(0)} = \boldsymbol{x}$, $\delta^{(0), \text{up}} = \delta \mathbb{1} \in \mathbb{R}^{d_0}$, $\delta^{(0), \text{low}} = \delta \mathbb{1} \in \mathbb{R}^{d_0}$

**for** $\ell = 0, \ldots, L-1$ **do**
- $\boldsymbol{x}^{(\ell+1)} = \sigma(\boldsymbol{W}^{(\ell)} \boldsymbol{x}^{(\ell)} + \boldsymbol{b}^{(\ell)})$
- $\delta^{(\ell+1), \text{up}} = \sigma(\boldsymbol{W}^{(\ell)} \boldsymbol{x}^{(\ell)} + (\boldsymbol{W}^{(\ell)})^+ \delta^{(\ell), \text{up}} + (\boldsymbol{W}^{(\ell)})^- \delta^{(\ell), \text{low}} + \boldsymbol{b}^{(\ell)}) - \boldsymbol{x}^{(\ell+1)}$
- $\delta^{(\ell+1), \text{low}} = \boldsymbol{x}^{(\ell+1)} - \sigma(\boldsymbol{W}^{(\ell)} \boldsymbol{x}^{(\ell)} - (\boldsymbol{W}^{(\ell)})^+ \delta^{(\ell), \text{low}} - (\boldsymbol{W}^{(\ell)})^- \delta^{(\ell), \text{up}} + \boldsymbol{b}^{(\ell)})$

**end for**

$\boldsymbol{x}^{(L+1)} = \boldsymbol{W}^{(L)} \boldsymbol{x}^{(L)} + \boldsymbol{b}^{(L)}$

$\delta^{(L+1), \text{up}} = (\boldsymbol{W}^{(L)})^+ \delta^{(L), \text{up}} + (\boldsymbol{W}^{(L)})^- \delta^{(L), \text{low}}$

$\delta^{(L+1), \text{low}} = (\boldsymbol{W}^{(L)})^+ \delta^{(L), \text{low}} + (\boldsymbol{W}^{(L)})^- \delta^{(L), \text{up}}$

**return** $\boldsymbol{x}^{(L+1)}$, $\boldsymbol{x}^{(L+1)} + \delta^{(L+1), \text{up}}$, $\boldsymbol{x}^{(L+1)} - \delta^{(L+1), \text{low}}$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 16.15</span></p>

Up to constants, Algorithm 3 has the same computational complexity as a forward pass (see also Algorithm 1). In addition, in contrast to upper bounds based on estimating the global Lipschitz constant of $\Phi$ via its weights, the upper bounds found via Algorithm 3 include the effect of the activation function $\sigma$. For example, if $\sigma$ is the ReLU, then we may often end up in a situation where $\delta^{(\ell), \text{up}}$ or $\delta^{(\ell), \text{low}}$ can have many entries that are $0$. If an entry of $\boldsymbol{W}^{(\ell)} \boldsymbol{x}^{(\ell)} + \boldsymbol{b}^{(\ell)}$ is nonpositive, then the associated entry in $\delta^{(\ell), \text{low}}$ will be zero. Similarly, if $\boldsymbol{W}^{(\ell)}$ has only few positive entries, then most of the entries of $\delta^{(\ell), \text{up}}$ are not propagated to $\delta^{(\ell+1), \text{up}}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16.16</span><span class="math-callout__name">(Correctness of Algorithm 3)</span></p>

Let $\Phi$ be a neural network with weight matrices $\boldsymbol{W}^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$ and bias vectors $\boldsymbol{b}^{(\ell)} \in \mathbb{R}^{d_{\ell+1}}$ for $\ell = 0, \ldots, L$, and a monotonically increasing activation function $\sigma$.

Let $\boldsymbol{x} \in \mathbb{R}^d$. Then the output of Algorithm 3 satisfies

$$\boldsymbol{x}^{L+1} + \delta^{(L+1), \text{up}} > z^{\delta, \max} \quad \text{and} \quad \boldsymbol{x}^{L+1} - \delta^{(L+1), \text{low}} < z^{\delta, \min}.$$

</div>

*Proof.* Fix $\boldsymbol{y}$, $\boldsymbol{x} \in \mathbb{R}^d$ with $\lVert \boldsymbol{y} - \boldsymbol{x} \rVert_\infty \leq \delta$ and let $\boldsymbol{y}^{(\ell)}$, $\boldsymbol{x}^{(\ell)}$ for $\ell = 0, \ldots, L+1$ be as in Algorithm 3 applied to $\boldsymbol{y}$, $\boldsymbol{x}$, respectively. We prove by induction over $\ell = 0, \ldots, L+1$ that

$$\boldsymbol{y}^{(\ell)} - \boldsymbol{x}^{(\ell)} \leq \delta^{\ell, \text{up}} \quad \text{and} \quad \boldsymbol{x}^{(\ell)} - \boldsymbol{y}^{(\ell)} \leq \delta^{\ell, \text{low}},$$

where the inequalities are understood entry-wise for vectors. Since $\boldsymbol{y}$ was arbitrary, this proves the result.

The case $\ell = 0$ follows immediately from $\lVert \boldsymbol{y} - \boldsymbol{x} \rVert_\infty \leq \delta$. For the induction step, we use the identity $\boldsymbol{W}^{(\ell)}(\boldsymbol{y}^{(\ell)} - \boldsymbol{x}^{(\ell)}) = (\boldsymbol{W}^{(\ell)})^+(\boldsymbol{y}^{(\ell)} - \boldsymbol{x}^{(\ell)}) + (\boldsymbol{W}^{(\ell)})^-(\boldsymbol{x}^{(\ell)} - \boldsymbol{y}^{(\ell)})$ and the monotonicity of $\sigma$ to propagate the bounds through each layer. The case $\ell = L+1$ follows by the same argument, replacing $\sigma$ by the identity. $\square$

---

## Appendix A: Probability Theory

This appendix provides basic notions and results in probability theory required in the main text. It is intended as a revision for a reader already familiar with these concepts.

### A.1 Sigma-Algebras, Topologies, and Measures

Let $\Omega$ be a set, and denote by $2^\Omega$ the powerset of $\Omega$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.1</span><span class="math-callout__name">(Sigma-Algebra)</span></p>

A subset $\mathfrak{A} \subseteq 2^\Omega$ is called a **sigma-algebra** on $\Omega$ if it satisfies

1. $\Omega \in \mathfrak{A}$,
2. $A^c \in \mathfrak{A}$ whenever $A \in \mathfrak{A}$,
3. $\bigcup_{i \in \mathbb{N}} A_i \in \mathfrak{A}$ whenever $A_i \in \mathfrak{A}$ for all $i \in \mathbb{N}$.

</div>

For a sigma-algebra $\mathfrak{A}$ on $\Omega$, the tuple $(\Omega, \mathfrak{A})$ is also referred to as a **measurable space**. For a measurable space, a subset $A \subseteq \Omega$ is called **measurable**, if $A \in \mathfrak{A}$. Measurable sets are also called **events**.

Another key system of subsets of $\Omega$ is that of a topology.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.2</span><span class="math-callout__name">(Topology)</span></p>

A subset $\mathfrak{T} \subseteq 2^\Omega$ is called a **topology** on $\Omega$ if it satisfies

1. $\emptyset, \Omega \in \mathfrak{T}$,
2. $\bigcap_{j=1}^n O_j \in \mathfrak{T}$ whenever $n \in \mathbb{N}$ and $O_1, \ldots, O_n \in \mathfrak{T}$,
3. $\bigcup_{i \in I} O_i \in \mathfrak{T}$ whenever for an index set $I$ holds $O_i \in \mathfrak{T}$ for all $i \in I$.

If $\mathfrak{T}$ is a topology on $\Omega$, we call $(\Omega, \mathfrak{T})$ a **topological space**, and a set $O \subseteq \Omega$ is called **open** if and only if $O \in \mathfrak{T}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark A.3</span></p>

The two notions differ in that a topology allows for unions of *arbitrary* (possibly uncountably many) sets, but only for *finite* intersection, whereas a sigma-algebra allows for countable unions and intersections.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.4</span><span class="math-callout__name">(Euclidean Topology)</span></p>

Let $d \in \mathbb{N}$ and denote by $B_\varepsilon(\boldsymbol{x}) = \lbrace \boldsymbol{y} \in \mathbb{R}^d \mid \lVert \boldsymbol{y} - \boldsymbol{x} \rVert < \varepsilon \rbrace$ the set of points whose Euclidean distance to $\boldsymbol{x}$ is less than $\varepsilon$. Then for every $A \subseteq \mathbb{R}^d$, the smallest topology on $A$ containing $A \cap B_\varepsilon(\boldsymbol{x})$ for all $\varepsilon > 0$, $\boldsymbol{x} \in \mathbb{R}^d$, is called the **Euclidean topology** on $A$.

</div>

If $(\Omega, \mathfrak{T})$ is a topological space, then the **Borel sigma-algebra** refers to the smallest sigma-algebra on $\Omega$ containing all open sets, i.e. all elements of $\mathfrak{T}$. Throughout this book, subsets of $\mathbb{R}^d$ are always understood to be equipped with the Euclidean topology and the Borel sigma-algebra. The Borel sigma-algebra on $\mathbb{R}^d$ is denoted by $\mathfrak{B}_d$.

We can now introduce measures.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.5</span><span class="math-callout__name">(Measure)</span></p>

Let $(\Omega, \mathfrak{A})$ be a measurable space. A mapping $\mu \colon \mathfrak{A} \to [0, \infty]$ is called a **measure** if it satisfies

1. $\mu(\emptyset) = 0$,
2. for every sequence $(A_i)_{i \in \mathbb{N}} \subseteq \mathfrak{A}$ such that $A_i \cap A_j = \emptyset$ whenever $i \neq j$, it holds

$$\mu\!\left(\bigcup_{i \in \mathbb{N}} A_i\right) = \sum_{i \in \mathbb{N}} \mu(A_i).$$

We say that the measure is **finite** if $\mu(\Omega) < \infty$, and it is **sigma-finite** if there exists a sequence $(A_i)_{i \in \mathbb{N}} \subseteq \mathfrak{A}$ such that $\Omega = \bigcup_{i \in \mathbb{N}} A_i$ and $\mu(A_i) < 1$ for all $i \in \mathbb{N}$. In case $\mu(\Omega) = 1$, the measure is called a **probability measure**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.6</span><span class="math-callout__name">(Lebesgue Measure)</span></p>

One can show that there exists a unique measure $\lambda$ on $(\mathbb{R}^d, \mathfrak{B}_d)$, such that for all sets of the type $\times_{j=1}^d [a_i, b_i)$ with $-\infty < a_i \leq b_i < \infty$ holds

$$\lambda(\times_{i=1}^d [a_i, b_i)) = \prod_{i=1}^d (b_i - a_i).$$

This measure is called the **Lebesgue measure**.

</div>

If $\mu$ is a measure on the measurable space $(\Omega, \mathfrak{A})$, then the triplet $(\Omega, \mathfrak{A}, \mu)$ is called a **measure space**. In case $\mu$ is a probability measure, it is called a **probability space**.

Let $(\Omega, \mathfrak{A}, \mu)$ be a measure space. A subset $N \subseteq \Omega$ is called a **null-set**, if $N$ is measurable and $\mu(N) = 0$. Moreover, an equality or inequality is said to hold $\mu$-**almost everywhere** or $\mu$-**almost surely**, if it is satisfied on the complement of a null-set.

### A.2 Random Variables

#### A.2.1 Measurability of Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.7</span><span class="math-callout__name">(Measurable Function)</span></p>

Let $(\Omega_1, \mathfrak{A}_1)$ and $(\Omega_2, \mathfrak{A}_2)$ be two measurable spaces. A function $f \colon \Omega_1 \to \Omega_2$ is called **measurable** if

$$f^{-1}(A_2) := \lbrace \omega \in \Omega_1 \mid f(\omega) \in A_2 \rbrace \in \mathfrak{A}_1 \quad \text{for all } A_2 \in \mathfrak{A}_2.$$

A mapping $X \colon \Omega_1 \to \Omega_2$ is called a $\Omega_2$**-valued random variable** if it is measurable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark A.8</span></p>

We again point out the parallels to topological spaces: A function $f \colon \Omega_1 \to \Omega_2$ between two topological spaces $(\Omega_1, \mathfrak{T}_1)$ and $(\Omega_2, \mathfrak{T}_2)$ is called **continuous** if $f^{-1}(O_2) \in \mathfrak{T}_1$ for all $O_2 \in \mathfrak{T}_2$.

</div>

Let $\Omega_1$ be a set and let $(\Omega_2, \mathfrak{A}_2)$ be a measurable space. For $X \colon \Omega_1 \to \Omega_2$, we can ask for the smallest sigma-algebra $\mathfrak{A}_X$ on $\Omega_1$ such that $X$ is measurable as a mapping from $(\Omega_1, \mathfrak{A}_X)$ to $(\Omega_2, \mathfrak{A}_2)$. Clearly, for every sigma-algebra $\mathfrak{A}_1$ on $\Omega_1$, $X$ is measurable as a mapping from $(\Omega_1, \mathfrak{A}_1)$ to $(\Omega_2, \mathfrak{A}_2)$ if and only if every $A \in \mathfrak{A}_X$ belongs to $\mathfrak{A}_1$; or in other words, $\mathfrak{A}_X$ is a sub sigma-algebra of $\mathfrak{A}_1$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.9</span><span class="math-callout__name">(Induced Sigma-Algebra)</span></p>

Let $X \colon \Omega_1 \to \Omega_2$ be a random variable. Then

$$\mathfrak{A}_X := \lbrace X^{-1}(A_2) \mid A_2 \in \mathfrak{A}_2 \rbrace \subseteq 2^{\Omega_1}$$

is the **sigma-algebra induced by** $X$ on $\Omega_1$.

</div>

#### A.2.2 Distribution and Expectation

Now let $(\Omega_1, \mathfrak{A}_1, \mathbb{P})$ be a probability space, and let $(\Omega_2, \mathfrak{A}_2)$ be a measurable space. Then $X$ naturally induces a measure on $(\Omega_2, \mathfrak{A}_2)$ via

$$\mathbb{P}_X[A_2] := \mathbb{P}[X^{-1}(A_2)] \quad \text{for all } A_2 \in \mathfrak{A}_2.$$

Note that due to the measurability of $X$ it holds $X^{-1}(A_2) \in \mathfrak{A}_1$, so that $\mathbb{P}_X$ is well-defined.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.10</span><span class="math-callout__name">(Distribution and Lebesgue Density)</span></p>

The measure $\mathbb{P}_X$ is called the **distribution** of $X$. If $(\Omega_2, \mathfrak{A}_2) = (\mathbb{R}^d, \mathfrak{B}_d)$, and there exists a function $f_X \colon \mathbb{R}^d \to \mathbb{R}$ such that

$$\mathbb{P}[A] = \int_A f_X(\boldsymbol{x})\, \mathrm{d}\boldsymbol{x} \quad \text{for all } A \in \mathfrak{B}_d,$$

then $f_X$ is called the **(Lebesgue) density** of $X$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark A.11</span></p>

The term distribution is often used without specifying an underlying probability space and random variable. In this case, "distribution" stands interchangeably for "probability measure". For example, $\mu$ *is a distribution on* $\Omega_2$ states that $\mu$ is a probability measure on the measurable space $(\Omega_2, \mathfrak{A}_2)$. In this case, there always exists a probability space $(\Omega_1, \mathfrak{A}_1, \mathbb{P})$ and a random variable $X \colon \Omega_1 \to \Omega_2$ such that $\mathbb{P}_X = \mu$; namely $(\Omega_1, \mathfrak{A}_1, \mathbb{P}) = (\Omega_2, \mathfrak{A}_2, \mu)$ and $X(\omega) = \omega$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.12</span><span class="math-callout__name">(Important Distributions)</span></p>

Some important distributions include the following.

- **Bernoulli distribution**: A random variable $X \colon \Omega \to \lbrace 0, 1 \rbrace$ is Bernoulli distributed if there exists $p \in [0, 1]$ such that $\mathbb{P}[X = 1] = p$ and $\mathbb{P}[X = 0] = 1 - p$.

- **Uniform distribution**: A random variable $X \colon \Omega \to \mathbb{R}^d$ is uniformly distributed on a measurable set $A \in \mathfrak{B}_d$, if its density equals

$$f_X(\boldsymbol{x}) = \frac{1}{\lvert A \rvert} \mathbb{1}_A(\boldsymbol{x})$$

where $\lvert A \rvert < \infty$ is the Lebesgue measure of $A$.

- **Gaussian distribution**: A random variable $X \colon \Omega \to \mathbb{R}^d$ is Gaussian distributed with mean $\boldsymbol{m} \in \mathbb{R}^d$ and the regular covariance matrix $\boldsymbol{C} \in \mathbb{R}^{d \times d}$, if its density equals

$$f_X(\boldsymbol{x}) = \frac{1}{(2\pi \det(\boldsymbol{C}))^{d/2}} \exp\!\left(-\frac{1}{2}(\boldsymbol{x} - \boldsymbol{m})^\top \boldsymbol{C}^{-1}(\boldsymbol{x} - \boldsymbol{m})\right).$$

We denote this distribution by $\mathrm{N}(\boldsymbol{m}, \boldsymbol{C})$.

</div>

Let $(\Omega, \mathfrak{A}, \mathbb{P})$ be a probability space, let $X \colon \Omega \to \mathbb{R}^d$ be an $\mathbb{R}^d$-valued random variable. We then call the Lebesgue integral

$$\mathbb{E}[X] := \int_\Omega X(\omega)\, \mathrm{d}\mathbb{P}(\omega) = \int_{\mathbb{R}^d} \boldsymbol{x}\, \mathrm{d}\mathbb{P}_X(\boldsymbol{x})$$

the **expectation** of $X$. Moreover, for $k \in \mathbb{N}$ we say that $X$ has **finite $k$-th moment** if $\mathbb{E}[\lVert X \rVert^k] < \infty$. Similarly, for a probability measure $\mu$ on $\mathbb{R}^d$ and $k \in \mathbb{N}$, we say that $\mu$ has finite $k$-th moment if

$$\int_{\mathbb{R}^d} \lVert \boldsymbol{x} \rVert^k\, \mathrm{d}\mu(\boldsymbol{x}) < \infty.$$

Furthermore, the matrix

$$\int_\Omega (X(\omega) - \mathbb{E}[X])(X(\omega) - \mathbb{E}[X])^\top\, \mathrm{d}\mathbb{P}(\omega) \in \mathbb{R}^{d \times d}$$

is the **covariance** of $X \colon \Omega \to \mathbb{R}^d$. For $d = 1$, it is called the **variance** of $X$ and denoted by $\mathbb{V}[X]$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.13</span><span class="math-callout__name">(Convergence of Random Variables)</span></p>

Let $(\Omega, \mathfrak{A}, \mathbb{P})$ be a probability space, let $X_j \colon \Omega \to \mathbb{R}^d$, $j \in \mathbb{N}$, be a sequence of random variables, and let $X \colon \Omega \to \mathbb{R}^d$ also be a random variable. The sequence is said to

1. **converge almost surely** to $X$, if

$$\mathbb{P}\!\left[\left\lbrace \omega \in \Omega \;\middle|\; \lim_{j \to \infty} X_j(\omega) = X(\omega) \right\rbrace\right] = 1,$$

2. **converge in probability** to $X$, if

$$\text{for all } \varepsilon > 0: \quad \lim_{j \to \infty} \mathbb{P}\!\left[\lbrace \omega \in \Omega \mid \lvert X_j(\omega) - X(\omega) \rvert > \varepsilon \rbrace\right] = 0,$$

3. **converge in distribution** to $X$, if for all bounded continuous functions $f \colon \mathbb{R}^d \to \mathbb{R}$

$$\lim_{j \to \infty} \mathbb{E}[f \circ X_j] = \mathbb{E}[f \circ X].$$

</div>

The notions in Definition A.13 are ordered by decreasing strength, i.e. almost sure convergence implies convergence in probability, and convergence in probability implies convergence in distribution. Since $\mathbb{E}[f \circ X] = \int_{\mathbb{R}^d} f(x)\, \mathrm{d}\mathbb{P}_X(x)$, the notion of convergence in distribution only depends on the distribution $\mathbb{P}_X$ of $X$. We thus also say that a sequence of random variables converges in distribution towards a measure $\mu$.

### A.3 Conditionals, Marginals, and Independence

#### A.3.1 Joint and Marginal Distribution

Let again $(\Omega, \mathfrak{A}, \mathbb{P})$ be a probability space, and let $X \colon \Omega \to \mathbb{R}^{d_X}$, $Y \colon \Omega \to \mathbb{R}^{d_Y}$ be two random variables. Then

$$Z := (X, Y) \colon \Omega \to \mathbb{R}^{d_X + d_Y}$$

is also a random variable. Its distribution $\mathbb{P}_Z$ is a measure on the measurable space $(\mathbb{R}^{d_X + d_Y}, \mathfrak{B}_{d_X + d_Y})$, and $\mathbb{P}_Z$ is referred to as the **joint distribution** of $X$ and $Y$. On the other hand, $\mathbb{P}_X$, $\mathbb{P}_Y$ are called the **marginal distributions** of $X$, $Y$. Note that

$$\mathbb{P}_X[A] = \mathbb{P}_Z[A \times \mathbb{R}^{d_Y}] \quad \text{for all } A \in \mathfrak{B}_{d_X},$$

and similarly for $\mathbb{P}_Y$. Thus the marginals $\mathbb{P}_X$, $\mathbb{P}_Y$ can be constructed from the joint distribution $\mathbb{P}_Z$. In turn, knowledge of the marginals is not sufficient to construct the joint distribution.

#### A.3.2 Independence

The concept of independence serves to formalize the situation where knowledge of one random variable provides no information about another random variable.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.14</span><span class="math-callout__name">(Independence)</span></p>

Let $(\Omega, \mathfrak{A}, \mathbb{P})$ be a probability space. Then two events $A$, $B \in \mathfrak{A}$ are called **independent** if

$$\mathbb{P}[A \cap B] = \mathbb{P}[A] \mathbb{P}[B].$$

Two random variables $X \colon \Omega \to \mathbb{R}^{d_X}$ and $Y \colon \Omega \to \mathbb{R}^{d_Y}$ are called **independent**, if

$$A, B \text{ are independent for all } A \in \mathfrak{A}_X,\; B \in \mathfrak{A}_Y.$$

</div>

Two random variables are thus independent if and only if all events in their induced sigma-algebras are independent. This turns out to be equivalent to the joint distribution $\mathbb{P}_{(X,Y)}$ being equal to the product measure $\mathbb{P}_X \otimes \mathbb{P}_Y$; the latter is characterized as the unique measure $\mu$ on $\mathbb{R}^{d_X + d_Y}$ satisfying $\mu(A \times B) = \mathbb{P}_X[A] \mathbb{P}_Y[B]$ for all $A \in \mathfrak{B}_{d_X}$, $B \in \mathfrak{B}_{d_Y}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.15</span><span class="math-callout__name">(Independence and Dice Rolls)</span></p>

Let $\Omega = \lbrace 1, \ldots, 6 \rbrace$ represent the outcomes of rolling a fair die, let $\mathfrak{A} = 2^\Omega$ be the sigma-algebra, and let $\mathbb{P}[\omega] = 1/6$ for all $\omega \in \Omega$. Consider three random variables:

$$X_1(\omega) = \begin{cases} 0 & \text{if } \omega \text{ is odd} \\ 1 & \text{if } \omega \text{ is even} \end{cases} \quad X_2(\omega) = \begin{cases} 0 & \text{if } \omega \leq 3 \\ 1 & \text{if } \omega \geq 4 \end{cases} \quad X_3(\omega) = \begin{cases} 0 & \text{if } \omega \in \lbrace 1, 2 \rbrace \\ 1 & \text{if } \omega \in \lbrace 3, 4 \rbrace \\ 2 & \text{if } \omega \in \lbrace 5, 6 \rbrace \end{cases}$$

$X_1$ and $X_2$ are not independent, but $X_1$ and $X_3$ are independent. This reflects the fact that, for example, knowing the outcome to be odd makes it more likely that the number belongs to $\lbrace 1, 2, 3 \rbrace$ rather than $\lbrace 4, 5, 6 \rbrace$. However, this knowledge provides no information on the three categories $\lbrace 1, 2 \rbrace$, $\lbrace 3, 4 \rbrace$, and $\lbrace 5, 6 \rbrace$.

</div>

If $X \colon \Omega \to \mathbb{R}$, $Y \colon \Omega \to \mathbb{R}$ are two independent random variables, then due to $\mathbb{P}_{(X,Y)} = \mathbb{P}_X \otimes \mathbb{P}_Y$

$$\mathbb{E}[XY] = \int_\Omega X(\omega) Y(\omega)\, \mathrm{d}\mathbb{P}(\omega) = \int_{\mathbb{R}} x\, \mathrm{d}\mathbb{P}_X(x) \int_{\mathbb{R}} y\, \mathrm{d}\mathbb{P}_Y(y) = \mathbb{E}[X] \mathbb{E}[Y].$$

Using this observation, it is easy to see that for a sequence of independent $\mathbb{R}$-valued random variables $(X_i)_{i=1}^n$ with bounded second moments, there holds **Bienaymé's identity**

$$\mathbb{V}\!\left[\sum_{i=1}^n X_i\right] = \sum_{i=1}^n \mathbb{V}[X_i].$$

#### A.3.3 Conditional Distributions

Let $(\Omega, \mathfrak{A}, \mathbb{P})$ be a probability space, and let $A$, $B \in \mathfrak{A}$ be two events. In case $\mathbb{P}[B] > 0$, we define

$$\mathbb{P}[A \mid B] := \frac{\mathbb{P}[A \cap B]}{\mathbb{P}[B]},$$

and call $\mathbb{P}[A \mid B]$ the **conditional probability of $A$ given $B$**.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.16</span></p>

Consider the setting of Example A.15. Let $A = \lbrace \omega \in \Omega \mid X_1(\omega) = 0 \rbrace$ be the event that the die roll was an odd number and let $B = \lbrace \omega \in \Omega \mid X_2(\omega) = 0 \rbrace$ be the event that the outcome yielded a number at most 3. Then $\mathbb{P}[B] = 1/2$, and $\mathbb{P}[A \cap B] = 1/3$. Thus

$$\mathbb{P}[A \mid B] = \frac{\mathbb{P}[A \cap B]}{\mathbb{P}[B]} = \frac{1/3}{1/2} = \frac{2}{3}.$$

This reflects that, given we know the outcome to be at most 3, the probability of the number being odd, i.e. in $\lbrace 1, 3 \rbrace$, is larger than the probability of the number being even, i.e. equal to 2.

</div>

The conditional probability above is only well-defined if $\mathbb{P}[B] > 0$. In practice, we often encounter the case where we would like to condition on an event of probability zero.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example A.17</span></p>

Consider the following procedure: We first draw a random number $p \in [0, 1]$ according to a uniform distribution on $[0, 1]$. Afterwards we draw a random number $X \in \lbrace 0, 1 \rbrace$ according to a $p$-Bernoulli distribution, i.e. $\mathbb{P}[X = 1] = p$ and $\mathbb{P}[X = 0] = 1 - p$. Then $(p, X)$ is a joint random variable taking values in $[0, 1] \times \lbrace 0, 1 \rbrace$. What is $\mathbb{P}[X = 1 \mid p = 0.5]$ in this case? Intuitively, it should be $1/2$, but note that $\mathbb{P}[p = 0.5] = 0$, so that the conditional probability formula is not meaningful here.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.18</span><span class="math-callout__name">(Regular Conditional Distribution)</span></p>

Let $(\Omega, \mathfrak{A}, \mathbb{P})$ be a probability space, and let $X \colon \Omega \to \mathbb{R}^{d_X}$ and $Y \colon \Omega \to \mathbb{R}^{d_Y}$ be two random variables. Let $\tau_{X \mid Y} \colon \mathfrak{B}_{d_X} \times \mathbb{R}^{d_Y} \to [0, 1]$ satisfy

1. $y \mapsto \tau_{X \mid Y}(A, y) \colon \mathbb{R}^{d_Y} \to [0, 1]$ is measurable for every fixed $A \in \mathfrak{B}_{d_X}$,
2. $A \mapsto \tau_{X \mid Y}(A, y)$ is a probability measure on $(\mathbb{R}^{d_X}, \mathfrak{B}_{d_X})$ for every $y \in Y(\Omega)$,
3. for all $A \in \mathfrak{B}_{d_X}$ and all $B \in \mathfrak{B}_{d_Y}$ holds

$$\mathbb{P}[X \in A, Y \in B] = \int_B \tau_{X \mid Y}(A, y)\, \mathbb{P}_Y(y).$$

Then $\tau$ is called a **regular (version of the) conditional distribution of $X$ given $Y$**. In this case, we denote

$$\mathbb{P}[X \in A \mid Y = y] := \tau_{X \mid Y}(A, y),$$

and refer to this measure as the conditional distribution of $X \mid Y = y$.

</div>

Definition A.18 provides a mathematically rigorous way of assigning a distribution to a random variable conditioned on an event that may have probability zero.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.19</span><span class="math-callout__name">(Existence and Uniqueness of Regular Conditional Distributions)</span></p>

Let $(\Omega, \mathfrak{A}, \mathbb{P})$ be a probability space, and let $X \colon \Omega \to \mathbb{R}^{d_X}$, $Y \colon \Omega \to \mathbb{R}^{d_Y}$ be two random variables. Then there exists a regular version of the conditional distribution $\tau_1$.

Let $\tau_2$ be another regular version of the conditional distribution. Then there exists a $\mathbb{P}_Y$-null set $N \subseteq \mathbb{R}^{d_Y}$, such that for all $y \in N^c \cap Y(\Omega)$, the two probability measures $\tau_1(\cdot, y)$ and $\tau_2(\cdot, y)$ coincide.

</div>

In particular, conditional distributions are only well-defined in a $\mathbb{P}_Y$-almost everywhere sense.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition A.20</span><span class="math-callout__name">(Conditional Independence)</span></p>

Let $(\Omega, \mathfrak{A}, \mathbb{P})$ be a probability space, and let $X \colon \Omega \to \mathbb{R}^{d_X}$, $Y \colon \Omega \to \mathbb{R}^{d_Y}$, $Z \colon \Omega \to \mathbb{R}^{d_Z}$ be three random variables. We say that $X$ and $Z$ are **conditionally independent given $Y$**, if the two distributions $X \mid Y = y$ and $Z \mid Y = y$ are independent for $\mathbb{P}_Y$-almost every $y \in Y(\Omega)$.

</div>

### A.4 Concentration Inequalities

Let $X_i \colon \Omega \to \mathbb{R}$, $i \in \mathbb{N}$, be a sequence of random variables with finite first moments. The centered average over the first $n$ terms

$$S_n := \frac{1}{n} \sum_{i=1}^n (X_i - \mathbb{E}[X_i])$$

is another random variable, and by linearity of the expectation it holds $\mathbb{E}[S_n] = 0$. The sequence is said to satisfy the **strong law of large numbers** if

$$\mathbb{P}\!\left[\lim\sup_{n \to \infty} \lvert S_n \rvert = 0\right] = 1.$$

This is the case, for example, if there exists $C < \infty$ such that $\mathbb{V}[X_i] \leq C$ for all $i \in \mathbb{N}$. Concentration inequalities provide bounds on the rate of this convergence.

We start with Markov's inequality.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma A.21</span><span class="math-callout__name">(Markov's Inequality)</span></p>

Let $X \colon \Omega \to \mathbb{R}$ be a random variable, and let $\varphi \colon [0, \infty) \to [0, \infty)$ be monotonically increasing. Then for all $\varepsilon > 0$

$$\mathbb{P}[\lvert X \rvert \geq \varepsilon] \leq \frac{\mathbb{E}[\varphi(\lvert X \rvert)]}{\varphi(\varepsilon)}.$$

</div>

*Proof.* We have

$$\mathbb{P}[\lvert X \rvert \geq \varepsilon] = \int_{X^{-1}([\varepsilon, \infty))} 1\, \mathrm{d}\mathbb{P}(\omega) \leq \int_\Omega \frac{\varphi(\lvert X(\omega) \rvert)}{\varphi(\varepsilon)}\, \mathrm{d}\mathbb{P}(\omega) = \frac{\mathbb{E}[\varphi(\lvert X \rvert)]}{\varphi(\varepsilon)},$$

which gives the claim. $\square$

Applying Markov's inequality with $\varphi(x) := x^2$ to the random variable $X - \mathbb{E}[X]$ directly gives Chebyshev's inequality.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma A.22</span><span class="math-callout__name">(Chebyshev's Inequality)</span></p>

Let $X \colon \Omega \to \mathbb{R}$ be a random variable with finite variance. Then for all $\varepsilon > 0$

$$\mathbb{P}[\lvert X - \mathbb{E}[X] \rvert \geq \varepsilon] \leq \frac{\mathbb{V}[X]}{\varepsilon^2}.$$

</div>

From Chebyshev's inequality we obtain the next result, which is a quite general concentration inequality for random variables with finite variances.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.23</span><span class="math-callout__name">(Concentration via Chebyshev)</span></p>

Let $X_1, \ldots, X_n$ be $n \in \mathbb{N}$ independent real-valued random variables such that for some $\varsigma > 0$ holds $\mathbb{E}[\lvert X_i - \mu \rvert^2] \leq \varsigma^2$ for all $i = 1, \ldots, n$. Denote

$$\mu := \mathbb{E}\!\left[\frac{1}{n} \sum_{j=1}^n X_j\right].$$

Then for all $\varepsilon > 0$

$$\mathbb{P}\!\left[\left\lvert \frac{1}{n} \sum_{i=1}^n X_i - \mu \right\rvert \geq \varepsilon\right] \leq \frac{\varsigma^2}{\varepsilon^2 n}.$$

</div>

*Proof.* Let $S_n = \sum_{i=1}^n (X_i - \mathbb{E}[X_i]) / n = (\sum_{i=1}^n X_i)/n - \mu$. By Bienaymé's identity (A.3.1), it holds that

$$\mathbb{V}[S_n] = \frac{1}{n^2} \sum_{j=1}^n \mathbb{E}[(X_i - \mathbb{E}[X_i])^2] \leq \frac{\varsigma^2}{n}.$$

Since $\mathbb{E}[S_n] = 0$, Chebyshev's inequality applied to $S_n$ gives the statement. $\square$

If we have additional information about the random variables, then we can derive sharper bounds. In case of uniformly bounded random variables (rather than just bounded variance), Hoeffding's inequality shows an exponential rate of concentration around the mean.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.24</span><span class="math-callout__name">(Hoeffding's Inequality)</span></p>

Let $a$, $b \in \mathbb{R}$. Let $X_1, \ldots, X_n$ be $n \in \mathbb{N}$ independent real-valued random variables such that $a \leq X_i \leq b$ almost surely for all $i = 1, \ldots, n$, and let $\mu$ be as in (A.4.2). Then, for every $\varepsilon > 0$

$$\mathbb{P}\!\left[\left\lvert \frac{1}{n} \sum_{j=1}^n X_j - \mu \right\rvert > \varepsilon\right] \leq 2 e^{-\frac{2n\varepsilon^2}{(b-a)^2}}.$$

</div>

Finally, we recall the central limit theorem in its multivariate formulation. We say that $(X_j)_{j \in \mathbb{N}}$ is an **i.i.d. sequence of random variables**, if the random variables are (pairwise) independent and identically distributed.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem A.25</span><span class="math-callout__name">(Multivariate Central Limit Theorem)</span></p>

Let $(\boldsymbol{X}_n)_{n \in \mathbb{N}}$ be an i.i.d. sequence of $\mathbb{R}^d$-valued random variables, such that $\mathbb{E}[\boldsymbol{X}_n] = \boldsymbol{0} \in \mathbb{R}^d$ and $\mathbb{E}[X_{n,i} X_{n,j}] = C_{ij}$ for all $i$, $j = 1, \ldots, d$. Let

$$\boldsymbol{Y}_n := \frac{\boldsymbol{X}_1 + \cdots + \boldsymbol{X}_n}{\sqrt{n}} \in \mathbb{R}^d.$$

Then $\boldsymbol{Y}_n$ converges in distribution to $\mathrm{N}(\boldsymbol{0}, \boldsymbol{C})$ as $n \to \infty$.

</div>

---

## Appendix B: Linear Algebra and Functional Analysis

This appendix provides some basic notions and results in linear algebra and functional analysis required in the main text.

### B.1 Singular Value Decomposition and Pseudoinverse

Let $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, $m, n \in \mathbb{N}$. Then the square root of the positive eigenvalues of $\boldsymbol{A}^\top \boldsymbol{A}$ (or equivalently of $\boldsymbol{A}\boldsymbol{A}^\top$) are referred to as the **singular values** of $\boldsymbol{A}$. We denote them by $s_1 \geq s_2 \cdots \geq s_r > 0$, where $r := \operatorname{rank}(\boldsymbol{A})$, so that $r \leq \min\lbrace m, n \rbrace$. Every matrix allows for a **singular value decomposition (SVD)** as stated in the next theorem. Recall that a matrix $\boldsymbol{V} \in \mathbb{R}^{n \times n}$ is called **orthogonal**, if $\boldsymbol{V}^\top \boldsymbol{V}$ is the identity.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.1</span><span class="math-callout__name">(Singular Value Decomposition)</span></p>

Let $\boldsymbol{A} \in \mathbb{R}^{m \times n}$. Then there exist orthogonal matrices $\boldsymbol{U} \in \mathbb{R}^{m \times m}$, $\boldsymbol{V} \in \mathbb{R}^{n \times n}$ such that with

$$\boldsymbol{\Sigma} := \begin{pmatrix} s_1 & & & \\ & \ddots & & \boldsymbol{0} \\ & & s_r & \\ & \boldsymbol{0} & & \boldsymbol{0} \end{pmatrix} \in \mathbb{R}^{m \times n}$$

it holds that $\boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^\top$, where $\boldsymbol{0}$ stands for a zero block of suitable size.

</div>

Given $\boldsymbol{y} \in \mathbb{R}^m$, consider the linear system

$$\boldsymbol{A}\boldsymbol{w} = \boldsymbol{y}. \tag{B.1.1}$$

If $\boldsymbol{A}$ is not a regular square matrix, then in general there need not be a unique solution $\boldsymbol{w} \in \mathbb{R}^n$ to (B.1.1). However, there exists a unique **minimal norm solution**

$$\boldsymbol{w}_* = \operatorname{argmin}_{\boldsymbol{w} \in M} \lVert \boldsymbol{w} \rVert, \quad M = \lbrace \boldsymbol{w} \in \mathbb{R}^m \mid \lVert \boldsymbol{A}\boldsymbol{w} - \boldsymbol{y} \rVert \leq \lVert \boldsymbol{A}\boldsymbol{v} - \boldsymbol{y} \rVert\; \forall \boldsymbol{v} \in \mathbb{R}^n \rbrace. \tag{B.1.2}$$

The minimal norm solution can be expressed via the **Moore-Penrose pseudoinverse** $\boldsymbol{A}^\dagger \in \mathbb{R}^{n \times m}$ of $\boldsymbol{A}$; given an (arbitrary) SVD $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^\top$, it is defined as

$$\boldsymbol{A}^\dagger := \boldsymbol{V} \boldsymbol{\Sigma}^\dagger \boldsymbol{U}^\top \quad \text{where} \quad \boldsymbol{\Sigma}^\dagger := \begin{pmatrix} s_1^{-1} & & & \\ & \ddots & & \boldsymbol{0} \\ & & s_r^{-1} & \\ & \boldsymbol{0} & & \boldsymbol{0} \end{pmatrix} \in \mathbb{R}^{n \times m}. \tag{B.1.3}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.2</span></p>

Let $\boldsymbol{A} \in \mathbb{R}^{m \times n}$. Then there exists a unique minimum norm solution $\boldsymbol{w}_* \in \mathbb{R}^n$ in (B.1.2) and it holds $\boldsymbol{w}_* = \boldsymbol{A}^\dagger \boldsymbol{y}$.

</div>

*Proof.* Denote by $\boldsymbol{\Sigma}_r \in \mathbb{R}^{r \times r}$ the upper left quadrant of $\boldsymbol{\Sigma}$. Since $\boldsymbol{U} \in \mathbb{R}^{m \times m}$ is orthogonal,

$$\lVert \boldsymbol{A}\boldsymbol{w} - \boldsymbol{y} \rVert = \left\lVert \begin{pmatrix} \boldsymbol{\Sigma}_r & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{0} \end{pmatrix} \boldsymbol{V}^\top \boldsymbol{w} - \boldsymbol{U}^\top \boldsymbol{y} \right\rVert.$$

We can thus write $M$ in (B.1.2) as

$$M = \left\lbrace \boldsymbol{w} \in \mathbb{R}^n \;\middle|\; (\boldsymbol{V}^\top \boldsymbol{w})_{i=1}^r = \boldsymbol{\Sigma}_r^{-1} (\boldsymbol{U}^\top \boldsymbol{y})_{i=1}^r \right\rbrace = \left\lbrace \boldsymbol{V}\boldsymbol{z} \;\middle|\; \boldsymbol{z} \in \mathbb{R}^n,\; (\boldsymbol{z})_{i=1}^r = \boldsymbol{\Sigma}_r^{-1} (\boldsymbol{U}^\top \boldsymbol{y})_{i=1}^r \right\rbrace$$

where $(\boldsymbol{a})_{i=1}^r$ denotes the first $r$ entries of a vector $\boldsymbol{a}$, and for the last equality we used orthogonality of $\boldsymbol{V} \in \mathbb{R}^{n \times n}$. Since $\lVert \boldsymbol{V}\boldsymbol{z} \rVert = \lVert \boldsymbol{z} \rVert$, the unique minimal norm solution is obtained by setting components $r+1, \ldots, m$ of $\boldsymbol{z}$ to zero, which yields

$$\boldsymbol{w}_* = \boldsymbol{V} \begin{pmatrix} \boldsymbol{\Sigma}_r^{-1} (\boldsymbol{U}^\top \boldsymbol{y})_{i=1}^r \\ \boldsymbol{0} \end{pmatrix} = \boldsymbol{V} \boldsymbol{\Sigma}^\dagger \boldsymbol{U}^\top \boldsymbol{y} = \boldsymbol{A}^\dagger \boldsymbol{y}$$

as claimed. $\square$

### B.2 Vector Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.3</span><span class="math-callout__name">(Vector Space)</span></p>

Let $\mathbb{K} \in \lbrace \mathbb{R}, \mathbb{C} \rbrace$. A **vector space** (over $\mathbb{K}$) is a set $X$ such that the following holds:

**(i) Properties of addition:** For every $x, y \in X$ there exists $x + y \in X$ such that for all $z \in X$

$$x + y = y + x \quad \text{and} \quad x + (y + z) = (x + y) + z.$$

Moreover, there exists a unique element $0 \in X$ such that $x + 0 = x$ for all $x \in X$ and for each $x \in X$ there exists a unique $-x \in X$ such that $x + (-x) = 0$.

**(ii) Properties of scalar multiplication:** There exists a map $(\alpha, x) \mapsto \alpha x$ from $\mathbb{K} \times X$ to $X$ called scalar multiplication. It satisfies $1x = x$ and $(\alpha\beta)x = \alpha(\beta x)$ for all $x \in X$.

We call the elements of a vector space **vectors**.

</div>

If the field is clear from context, we simply refer to $X$ as a vector space. We will primarily consider the case $\mathbb{K} = \mathbb{R}$, and in this case we also say that $X$ is a **real vector space**.

To introduce a notion of convergence on a vector space $X$, it needs to be equipped with a topology. A **topological vector space** is a vector space which is also a topological space, and in which addition and scalar multiplication are continuous maps.

#### B.2.1 Metric Spaces

An important class of topological vector spaces consists of vector spaces that are also metric spaces.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.4</span><span class="math-callout__name">(Metric)</span></p>

For a set $X$, we call a map $d_X \colon X \times X \to [0, \infty)$ a **metric**, if

1. $d_X(x, y) = 0$ if and only if $x = y$,
2. $d_X(x, y) = d(y, x)$ for all $x, y \in X$,
3. $d_X(x, z) \leq d_X(x, y) + d_X(y, z)$ for all $x, y, z \in X$.

We call $(X, d_X)$ a **metric space**.

</div>

In a metric space $(X, d_X)$, we denote the **open ball** with center $x$ and radius $r > 0$ by

$$B_r(x) := \lbrace y \in X \mid d_X(x, y) < r \rbrace. \tag{B.2.1}$$

Every metric space is naturally equipped with a topology: A set $A \subseteq X$ is open if and only if for every $x \in A$ exists $\varepsilon > 0$ such that $B_\varepsilon(x) \subseteq A$. Therefore every metric vector space is a topological vector space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.5</span><span class="math-callout__name">(Complete Metric Space)</span></p>

A metric space $(X, d_X)$ is called **complete**, if every Cauchy sequence with respect to $d$ converges to an element in $X$.

</div>

For complete metric spaces, an immensely powerful tool is Baire's category theorem. Let $A, B \subseteq X$ for a topological space $X$. Then $A$ is **dense** in $B$ if the closure of $A$, denoted by $\overline{A}$, satisfies $\overline{A} \supseteq B$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.6</span><span class="math-callout__name">(Baire's Category Theorem)</span></p>

Let $X$ be a complete metric space. Then the intersection of every countable collection of dense open subsets of $X$ is dense in $X$.

</div>

Theorem B.6 implies that if $X = \bigcup_{i=1}^\infty V_i$ for a sequence of sets $V_i$, then at least one of the $V_i$ has to contain an open set. Indeed, assuming all $V_i$'s have empty interior implies that $V_i^c = X \setminus V_i$ is dense for all $i \in \mathbb{N}$. By De Morgan's laws, it then holds that $\emptyset = \bigcap_{i=1}^\infty V_i^c$ which contradicts Theorem B.6.

#### B.2.2 Normed Spaces

A norm is a way of assigning a length to a vector. A normed space is a vector space with a norm.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.7</span><span class="math-callout__name">(Norm)</span></p>

Let $X$ be a vector space over a field $\mathbb{K} \in \lbrace \mathbb{R}, \mathbb{C} \rbrace$. A map $\lVert \cdot \rVert_X \colon X \to [0, \infty)$ is called a **norm** if the following hold for all $x, y \in X$ and all $\alpha \in \mathbb{K}$:

1. **triangle inequality:** $\lVert x + y \rVert_X \leq \lVert x \rVert_X + \lVert y \rVert_X$,
2. **absolute homogeneity:** $\lVert \alpha x \rVert_X = \lvert \alpha \rvert \lVert x \rVert_X$,
3. **positive definiteness:** $\lVert x \rVert_X = 0$ if and only if $x = 0$.

We call $(X, \lVert \cdot \rVert_X)$ a **normed space**.

</div>

Every norm induces a metric $d_X$ and hence a topology via $d_X(x, y) := \lVert x - y \rVert_X$. In particular, every normed vector space is a topological vector space with respect to this topology.

#### B.2.3 Banach Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.8</span><span class="math-callout__name">(Banach Space)</span></p>

A normed vector space is called a **Banach space** if and only if it is complete.

</div>

Before presenting the main results on Banach spaces, we collect a couple of important examples.

- **Euclidean spaces:** Let $d \in \mathbb{N}$. Then $(\mathbb{R}^d, \lVert \cdot \rVert)$ is a Banach space.

- **Continuous functions:** Let $d \in \mathbb{N}$ and let $K \subseteq \mathbb{R}^d$ be compact. The set of continuous functions from $K$ to $\mathbb{R}$ is denoted by $C(K)$. For $\alpha, \beta \in \mathbb{R}$ and $f, g \in C(K)$, we define addition and scalar multiplication by $(\alpha f + \beta g)(\boldsymbol{x}) = \alpha f(\boldsymbol{x}) + \beta g(\boldsymbol{x})$ for all $\boldsymbol{x} \in K$. The vector space $C(K)$ equipped with the **supremum norm**

$$\lVert f \rVert_\infty := \sup_{\boldsymbol{x} \in K} \lvert f(\boldsymbol{x}) \rvert,$$

is a Banach space.

- **Lebesgue spaces:** Let $(\Omega, \mathfrak{A}, \mu)$ be a measure space and let $1 \leq p < \infty$. Then the **Lebesgue space** $L^p(\Omega, \mu)$ is defined as the vector space of all equivalence classes of measurable functions $f \colon \Omega \to \mathbb{R}$ that coincide $\mu$-almost everywhere and satisfy

$$\lVert f \rVert_{L^p(\Omega,\mu)} := \left( \int_\Omega \lvert f(x) \rvert^p \,\mathrm{d}\mu(x) \right)^{1/p} < \infty. \tag{B.2.2}$$

$L^p(\Omega, \mu)$ is a Banach space. If $\Omega$ is a measurable subset of $\mathbb{R}^d$, and $\mu$ is the Lebesgue measure, we typically write $L^p(\Omega)$. If $\Omega = \mathbb{N}$ and the measure is the counting measure, we denote these spaces by $\ell^p(\mathbb{N})$ or simply $\ell^p$.

- **Essentially bounded functions:** Let $(\Omega, \mathfrak{A}, \mu)$ be a measure space. The $L^p$ spaces can be extended to $p = \infty$ by defining the $L^\infty$-norm

$$\lVert f \rVert_{L^\infty(\Omega,\mu)} := \inf\lbrace C \geq 0 \mid \mu(\lbrace \lvert f \rvert > C \rbrace) = 0 \rbrace.$$

With this norm, $L^\infty(\Omega, \mu)$ is a Banach space. If $\Omega = \mathbb{N}$ and $\mu$ is the counting measure, we denote the resulting space by $\ell^\infty(\mathbb{N})$ or simply $\ell^\infty$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.9</span><span class="math-callout__name">(Dual Space)</span></p>

Let $(X, \lVert \cdot \rVert_X)$ be a normed vector space over $\mathbb{K} \in \lbrace \mathbb{R}, \mathbb{C} \rbrace$. Linear maps from $X \to \mathbb{K}$ are called **linear functionals**. The vector space of all continuous linear functionals on $X$ is called the **(topological) dual space** of $X$ and is denoted by $X'$.

Together with the natural addition and scalar multiplication, $X'$ is a vector space. We equip $X'$ with the norm

$$\lVert f \rVert_{X'} := \sup_{\substack{x \in X \\ \lVert x \rVert_X = 1}} \lvert f(x) \rvert.$$

The space $(X', \lVert \cdot \rVert_{X'})$ is always a Banach space, even if $(X, \lVert \cdot \rVert_X)$ is not complete.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.10</span><span class="math-callout__name">(Geometric Hahn-Banach, Subspace Version)</span></p>

Let $M$ be a subspace of a Banach space $X$ and let $x_0 \in X$. If $x_0$ is not in the closure of $M$, then there exists $f \in X'$ such that $f(x_0) = 1$ and $f(x) = 0$ for every $x \in M$.

</div>

An immediate consequence of Theorem B.10 is the existence of a **dual basis**. Let $X$ be a Banach space and let $(x_i)_{i \in \mathbb{N}} \subseteq X$ be such that for all $i \in \mathbb{N}$

$$x_i \notin \overline{\operatorname{span}\lbrace x_j \mid j \in \mathbb{N},\; j \neq i \rbrace}.$$

Then, for every $i \in \mathbb{N}$, there exists $f_i \in X'$ such that $f_i(x_j) = 0$ if $i \neq j$ and $f_i(x_i) = 1$.

#### B.2.4 Hilbert Spaces

Often, we require more structure than that provided by normed spaces. An inner product offers additional tools to compare vectors by introducing notions of angle and orthogonality.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.11</span><span class="math-callout__name">(Inner Product)</span></p>

Let $X$ be a real vector space. A map $\langle \cdot, \cdot \rangle_X \colon X \times X \to \mathbb{R}$ is called an **inner product** on $X$ if the following hold for all $x, y, z \in X$ and all $\alpha, \beta \in \mathbb{R}$:

1. **linearity:** $\langle \alpha x + \beta y, z \rangle_X = \alpha \langle x, z \rangle_X + \beta \langle y, z \rangle_X$,
2. **symmetry:** $\langle x, y \rangle_X = \langle y, x \rangle_X$,
3. **positive definiteness:** $\langle x, x \rangle_X > 0$ for all $x \neq 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example B.12</span></p>

For $p = 2$, the Lebesgue spaces $L^2(\Omega)$ and $\ell^2(\mathbb{N})$ are Hilbert spaces with inner products

$$\langle f, g \rangle_{L^2(\Omega)} = \int_\Omega f(x) g(x)\,\mathrm{d}x \quad \text{for all } f, g \in L^2(\Omega),$$

and

$$\langle \boldsymbol{x}, \boldsymbol{y} \rangle_{\ell^2(\mathbb{N})} = \sum_{j \in \mathbb{N}} x_j y_j \quad \text{for all } \boldsymbol{x} = (x_j)_{j \in \mathbb{N}},\; \boldsymbol{y} = (y_j)_{j \in \mathbb{N}} \in \ell^2(\mathbb{N}).$$

</div>

On inner product spaces the Cauchy-Schwarz inequality holds.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.13</span><span class="math-callout__name">(Cauchy-Schwarz Inequality)</span></p>

Let $X$ be a vector space with inner product $\langle \cdot, \cdot \rangle_X$. Then it holds for all $x, y \in X$

$$\lvert \langle x, y \rangle_X \rvert \leq \sqrt{\langle x, x \rangle_X\, \langle y, y \rangle_X}.$$

Moreover, equality holds if and only if $x$ and $y$ are linearly dependent.

</div>

*Proof.* Let $x, y \in X$. If $y = 0$ then $\langle x, y \rangle_X = 0$ and the statement is trivial. Assume $y \neq 0$, so that $\langle y, y \rangle_X > 0$. Using linearity and symmetry it holds for all $\alpha \in \mathbb{R}$

$$0 \leq \langle x - \alpha y, x - \alpha y \rangle_X = \langle x, x \rangle_X - 2\alpha \langle x, y \rangle_X + \alpha^2 \langle y, y \rangle_X.$$

Letting $\alpha := \langle x, y \rangle_X / \langle y, y \rangle_X$ we get

$$0 \leq \langle x, x \rangle_X - 2\frac{\langle x, y \rangle_X^2}{\langle y, y \rangle_X} + \frac{\langle x, y \rangle_X^2}{\langle y, y \rangle_X} = \langle x, x \rangle_X - \frac{\langle x, y \rangle_X^2}{\langle y, y \rangle_X}.$$

Rearranging terms gives the claim. $\square$

Every inner product $\langle \cdot, \cdot \rangle_X$ induces a norm via

$$\lVert x \rVert_X := \sqrt{\langle x, x \rangle} \quad \text{for all } x \in X. \tag{B.2.3}$$

The properties of the inner product immediately yield the **polar identity**

$$\lVert x + y \rVert_X^2 = \lVert x \rVert_X^2 + 2\langle x, y \rangle_X + \lVert y \rVert_X^2. \tag{B.2.4}$$

The fact that (B.2.3) indeed defines a norm follows by an application of the Cauchy-Schwarz inequality to (B.2.4), which yields that $\lVert \cdot \rVert_X$ satisfies the triangle inequality.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.14</span><span class="math-callout__name">(Hilbert Space)</span></p>

Let $H$ be a real vector space with inner product $\langle \cdot, \cdot \rangle_H$. Then $(H, \langle \cdot, \cdot \rangle_H)$ is called a **Hilbert space** if and only if $H$ is complete with respect to the norm $\lVert \cdot \rVert_H$ induced by the inner product.

</div>

A standard example of a Hilbert space is $L^2$: Let $(\Omega, \mathfrak{A}, \mu)$ be a measure space. Then

$$\langle f, g \rangle_{L^2(\Omega,\mu)} = \int_\Omega f(x) g(x)\,\mathrm{d}\mu(x) \quad \text{for all } f, g \in L^2(\Omega, \mu),$$

defines an inner product on $L^2(\Omega, \mu)$ compatible with the $L^2(\Omega, \mu)$-norm.

In a Hilbert space, we can compare vectors not only via their distance, measured by the norm, but also by using the inner product, which corresponds to their relative orientation. This leads to the concept of orthogonality.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition B.15</span><span class="math-callout__name">(Orthogonality)</span></p>

Let $(H, \langle \cdot, \cdot \rangle_H)$ be a Hilbert space and let $f, g \in H$. We say that $f$ and $g$ are **orthogonal** if $\langle f, g \rangle_H = 0$, denoted by $f \perp g$. For $F, G \subseteq H$ we write $F \perp G$ if $f \perp g$ for all $f \in F$, $g \in G$. Finally, for $F \subseteq H$, the set $F^\perp = \lbrace g \in H \mid g \perp f\; \forall f \in F \rbrace$ is called the **orthogonal complement** of $F$ in $H$.

</div>

For orthogonal vectors, the polar identity immediately implies the Pythagorean theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.16</span><span class="math-callout__name">(Pythagorean Theorem)</span></p>

Let $(H, \langle \cdot, \cdot \rangle_H)$ be a Hilbert space, $n \in \mathbb{N}$, and let $f_1, \ldots, f_n \in H$ be pairwise orthogonal vectors. Then,

$$\left\lVert \sum_{i=1}^n f_i \right\rVert_H^2 = \sum_{i=1}^n \lVert f_i \rVert_H^2.$$

</div>

A final property of Hilbert spaces that we encounter in this book is the existence of unique **projections** onto convex sets.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem B.17</span><span class="math-callout__name">(Projection onto Convex Sets)</span></p>

Let $(H, \langle \cdot, \cdot \rangle_H)$ be a Hilbert space and let $K \neq \emptyset$ be a closed convex subset of $H$. Then for all $h \in H$ exists a unique $k_0 \in K$ such that

$$\lVert h - k_0 \rVert_H = \inf\lbrace \lVert h - k \rVert_H \mid k \in K \rbrace.$$

</div>
