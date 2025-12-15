---
title: Neural Differential Equations
layout: default
noindex: true
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

# Neural Differential Equations

## Motivation

A **neural differential equation** is a differential equation whose vector field is parameterized by a neural network. The most common example is the **neural ordinary differential equation**:

$$y(0)=y_0, \qquad \frac{dy}{dt}(t)=f_\theta\bigl(t,y(t)\bigr)$$

Here, $\theta$ denotes a set of learned parameters, $f_\theta:\mathbb{R}\times \mathbb{R}^{d_1\times\cdots\times d_k}\to \mathbb{R}^{d_1\times\cdots\times d_k}$ can be any standard neural network architecture, and $y:[0,T]\to \mathbb{R}^{d_1\times\cdots\times d_k}$ is the resulting solution. In many practical settings, $f_\theta$ is simply a feedforward network.

The key idea is to incorporate a differential equation solver into a trainable, differentiable computation graph.

As a simple illustration, suppose we observe an image $y_0\in \mathbb{R}^{3\times 32\times 32}$ (an RGB image with ($32\times 32$) pixels) and want to classify it as either a cat or a dog. We set $y(0)=y_0$ as the initial condition for the neural ODE and evolve the system up to some time $T$. An affine (commonly referred to as a "linear" transformation in deep learning, although this is not technically correct in the mathematical sense of the word) map $\ell_\theta:\mathbb{R}^{3\times 32\times 32}\to \mathbb{R}^2$ is then applied, followed by a softmax, so that the output may be interpreted as a length-2 tuple ($\mathbb{P}$(picture is of a cat), $\mathbb{P}$(picture is of a dog)).

<div class="gd-grid">
  <img src="{{ '/assets/images/notes/dynamical-systems/simple_neural_ode.png' | relative_url }}" alt="simple neural ODE" loading="lazy">
  <figcaption>Computation graph for a simple neural ODE.</figcaption>
</div>

This is illustrated in Figure. In standard notation, the computation can be written as

$$\mathrm{softmax}\left(\ell_\theta\left(y(0)+\int_0^T f_\theta(t,y(t)),dt\right)\right)$$

### Continuous-depth neural networks

Recall the residual network formulation:

$$y_{j+1}=y_j+f_\theta(j,y_j), \quad (1.1)$$

where $f_\theta(j,\cdot)$ is the $j$-th residual block (with all block parameters bundled into $\theta$).

Now consider the neural ODE

$$\frac{dy}{dt}(t)=f_\theta(t,y(t)).$$

If we discretise this using the explicit Euler method at time points $t_j$ spaced by $\Delta t$, we obtain

$$\frac{y(t_{j+1})-y(t_j)}{\Delta t}\approx f_\theta(t_j,y(t_j)),$$

and therefore

$$y(t_{j+1})=y(t_j)+\Delta t, f_\theta(t_j,y(t_j)).$$

By absorbing $\Delta t$ into $f_\theta$, we recover equation (1.1).

This suggests that neural ODEs can be viewed as the continuous-depth limit of residual networks, which naturally encourages further links. In particular, the defining update rules of GRUs and LSTMs, compared to generic RNNs, look very much like discretised differential equations. StyleGAN2 and (score based) diﬀusion models are simply discretised SDEs. Coupling layers in invertible neural networks turn out to be related to reversible diﬀerential equation solvers. And so on.

### An important distinction

There are some works that seek numerical approximations to the solution $y$ of an ODE

$$\frac{dy}{dt}=f(t,y(t))$$

by representing the solution itself with a neural network, $y=y_\theta$.

If $f$ is known, one can fit $y_\theta$ by minimizing a loss such as

$$\min_\theta \frac{1}{N}\sum_{i=1}^N \left\lVert \frac{dy_\theta}{dt}(t_i)-f\bigl(t_i,y_\theta(t_i)\bigr)\right\rVert, \quad (1.2)$$

for sample points $t_i\in[0,T]$. In this approach, solving the differential equation becomes an optimization problem, reminiscent of collocation or finite element methods. This has been widely studied in the literature.

This framework is known as a **physics-informed neural network (PINN)**. PINNs can be especially useful for certain PDEs—particularly nonlocal or high-dimensional ones—where classical solvers may be very expensive, though in many settings traditional methods remain more efficient.

**Toy example: learning $y_\theta$ (PINN?):**
<div class="accordion">
  <details markdown="1">
    <summary>Toy example: learn $y_\theta$ (PINN?)</summary>

Sure — here’s a concrete **toy example** of learning $y_\theta$ directly.

### Goal

Solve
$$
\frac{dy}{dt} = -y, \qquad y(0)=1
$$
by **training a neural net to represent the solution**.

### Model

Let a small NN approximate the function:
$$
y_\theta(t) = 1 + t\cdot \text{NN}_\theta(t)
$$
This automatically enforces $y_\theta(0)=1$.

> Note: not every function $y_\theta(t)$ that you put in, for example, $\frac{dy_{\theta}}{dt}(t)=const+y_\theta(t)$ will give this equality. Such function $y_\theta(t)$ that satisfies this equality must be learned.

### ODE residual

At sampled points $t_i\in[0,T]$,
$$
r_\theta(t_i) = \frac{d y_\theta}{dt}(t_i) + y_\theta(t_i)
$$

### Loss

$$
\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^N \lvert r_\theta(t_i)\rvert^2
$$

Train with SGD/Adam. After training, $y_\theta(t)\approx e^{-t}$.

### Minimal PyTorch sketch

```python
import torch
import torch.nn as nn

# small NN
net = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

def y_theta(t):
    return 1.0 + t * net(t)   # hard-enforce y(0)=1

# collocation points
T = 2.0
N = 128
t = torch.rand(N, 1) * T
t.requires_grad_(True)

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

for _ in range(2000):
    y = y_theta(t)
    dy_dt = torch.autograd.grad(
        y, t, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]

    residual = dy_dt + y
    loss = (residual**2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
```

That’s the framework in action: **the NN is the solution**.
If you want, I can give a PDE example in the same style.
  </details>
</div>

Crucially, this idea is **different** from neural differential equations. NDEs use neural networks to *define* the differential equation itself, whereas (1.2) uses neural networks to *approximate the solution* of a pre-specified differential equation. This distinction is often blurred, especially because the PDE analogue of (1.2) is sometimes also called a “neural partial differential equation.”

### Inspiration

Traditional "discrete" deep learning is widely applicable, and rightly so. We have already seen the parallels between diﬀerential equations and deep learning: a highly successful strategy for the development of deep learning models is simply to take the appropriate diﬀerential equation, and then discretise it.

By coincidence (or, as the idea becomes more popular, by design) many of the most eﬀective and popular deep learning architectures resemble diﬀerential equations. Perhaps we should not be surprised: diﬀerential equations have been the dominant modelling paradigm for centuries; they are not so easily toppled.

## Neural Ordinary Diﬀerential Equations

<!-- Here’s a rephrased version of the text you shared, keeping the meaning and structure but changing the wording.

### Introduction

The most widely used form of a neural differential equation is the **neural ODE**:


$$y(0)=y_0, \qquad \frac{dy}{dt}(t)=f_\theta(t,y(t)).$$


Here, (y_0 \in \mathbb{R}^{d_1 \times \cdots \times d_k}) can be a tensor of any shape, (\theta) denotes the learned parameters, and (f_\theta: \mathbb{R}\times \mathbb{R}^{d_1 \times \cdots \times d_k}\to \mathbb{R}^{d_1 \times \cdots \times d_k}) is implemented by a neural network. In practice, (f_\theta) is usually a fairly standard architecture, such as a feedforward or convolutional model.

---

## 2.1.1 Existence and uniqueness

A natural first concern—especially from a mathematical viewpoint—is whether equation (2.1) has a solution, and whether that solution is unique. This is not hard to establish. If (f_\theta) is **Lipschitz** in (y)—a condition commonly satisfied by neural networks since they are compositions of typically Lipschitz components—then Picard’s theorem guarantees well-posedness.

**Theorem 2.1 (Picard).**
Let (f:[0,T]\times \mathbb{R}^d\to \mathbb{R}^d) be continuous in (t) and uniformly Lipschitz in (y). For any (y_0\in \mathbb{R}^d), there exists a unique differentiable solution (y:[0,T]\to \mathbb{R}^d) satisfying

$$y(0)=y_0, \qquad \frac{dy}{dt}(t)=f(t,y(t)).$$

“Uniformly Lipschitz” means the Lipschitz constant does not depend on time: there is some (C>0) such that for all (t,y_1,y_2),

$$\|f_\theta(t,y_1)-f_\theta(t,y_2)\|\le C\|y_1-y_2\|.$$

## 2.1.2 Evaluation and training

Compared with models that are not based on differential equations, neural ODEs introduce two additional practical requirements.

1. We need reliable **numerical solvers** to compute solutions, because closed-form solutions are almost never available.
2. We must be able to **differentiate through the ODE solve** to obtain gradients with respect to (\theta).

Since software for both forward solving and backpropagation is now well established, we can focus mainly on designing the architecture. A more detailed discussion of evaluation and differentiation appears later.

## 2.2 Applications

### 2.2.1 Image classification

Since CNN-based image classification is often the first deep learning task people encounter, it provides a convenient starting point for illustrating neural differential equations.

**Dataset.**
Assume we have images represented as tensors in (\mathbb{R}^{3\times 32\times 32}) (RGB channels with height and width of 32). Each image has a label in (\mathbb{R}^{10}), using one-hot encoding for ten possible categories (e.g., aeroplane, car, bird, cat, deer, dog, frog, horse, ship, lorry).

**Model.**
Let (f_\theta:\mathbb{R}\times \mathbb{R}^{3\times 32\times 32}\to \mathbb{R}^{3\times 32\times 32}) be a convolutional network defining the dynamics, and let (\ell_\theta:\mathbb{R}^{3\times 32\times 32}\to \mathbb{R}^{10}) be an affine map. Define the classifier

[
\phi:\mathbb{R}^{3\times 32\times 32}\to \mathbb{R}^{10},
\qquad
\phi(y_0)=\text{softmax}(\ell_\theta(y(T))),
]

where (y) is the solution to

[
y(0)=y_0,
\qquad
\frac{dy}{dt}(t)=f_\theta(t,y(t)).
]

**Loss function.**
Using cross-entropy, we can train (\theta) so that the output represents class probabilities. Concretely, for data ((a_i,b_i)) with (a_i\in \mathbb{R}^{3\times 32\times 32}) and (b_i\in \mathbb{R}^{10}),

$$-\frac{1}{N}\sum_{i=1}^N b_i\cdot \log \phi(a_i)$$

is minimized, where (\cdot) is a dot product and the logarithm is applied elementwise.

**Practical note.**
This is mainly a pedagogical example. For image classification, continuous-time models typically offer limited benefit. Standard residual networks—viewable as discretized versions of neural ODEs—are generally simpler and more practical. Hence, neural ODEs are not being recommended here as the best choice for this task.

**The manifold hypothesis.**
Neural ODEs mesh nicely with the manifold hypothesis, which proposes that high-dimensional data often lie close to a lower-dimensional manifold in feature space. The ODE can be interpreted as defining a smooth flow that evolves this data manifold over time.

If you want, I can also make this more compact, or rewrite it in a more “lecture notes” style. -->

### Introduction

#### The Core Concept
> **Neural ODEs.** A Neural ODE replaces the discrete layers of a traditional neural network with a continuous transformation. Instead of updating a hidden state in steps, the state $y(t)$ evolves according to an ordinary differential equation (ODE). 

The most common neural diﬀerential equation is a neural ODE:

$$y(0)=y_0, \qquad \frac{dy}{dt}(t)=f_\theta(t,y(t)).$$


* **$y(0) = y_0$:** The input data (e.g., an image tensor).
* **$f_\theta$:** A standard neural network (like a feedforward or a CNN) that learns the derivative—effectively deciding *how* the data changes over time.
* **Solution:** To get the output of the model, we calculate $y(T)$ by *integrating* this equation from time 0 to $T$.
  
Note, $y_0 \in \mathbb{R}^{d_1 \times \cdots \times d_k}$ can be a tensor of any shape, (\theta) denotes the learned parameters, and $f_\theta: \mathbb{R}\times \mathbb{R}^{d_1 \times \cdots \times d_k}\to \mathbb{R}^{d_1 \times \cdots \times d_k}$. The input space is $\mathbb{R}^{d_1 \times \cdots \times d_k}$ and the function $f_\theta$ takes the scalar time $t$ and $y(t)$ of dimensionality of the input and produces the output of the size of the input. 

For example, for the image we would have a CNN $f_\theta: \mathbb{R}\times \mathbb{R}^{3 \times 32 \times 32} \to \mathbb{R}^{3 \times 32 \times 32}$, affine part $l_\theta: \mathbb{R}^{3 \times 32 \times 32} \to \mathbb{R}^{10}$ and the whole model is a stack of them + softmax on the top of it: $\text{softmax} \circ l_\theta \circ f_\theta$.

#### Theoretical Validity (Existence and Uniqueness)

A common mathematical concern is whether a valid solution to this equation actually exists.

$\textbf{Theorem (Picard’s Existence Theorem):}$ Let $f:[0,T]\times \mathbb{R}^d\to \mathbb{R}^d$ be continuous in $t$ and *uniformly Lipschitz* in $y$. For any $y_0\in \mathbb{R}^d$, there exists a unique differentiable solution $y:[0,T]\to \mathbb{R}^d$ satisfying

$$y(0)=y_0, \qquad \frac{dy}{dt}(t)=f(t,y(t)).$$

**"Uniformly Lipschitz"** means the Lipschitz constant does not depend on time: there is some $C>0$ such that for all $t,y_1,y_2$,

$$\|f_\theta(t,y_1)-f_\theta(t,y_2)\|\le C\|y_1-y_2\|.$$

* Since most standard neural network layers (composed of linear maps and activation functions like ReLU or Tanh) are Lipschitz continuous (basically a neural network, which is usually a composition of Lipschitz functions), **Neural ODEs are mathematically sound**.
* **Numerical Solvers:** We cannot usually write down a simple formula for the solution, so we use numerical integration software to compute $y(T)$.

#### The Manifold Hypothesis
**Manifold Hypothesis:** Neural ODEs are, however, theoretically elegant. They align well with the "manifold hypothesis"—the idea that high-dimensional data actually lies on a lower-dimensional surface (manifold). The ODE models the "flow" of data evolving along this manifold.

### Physical modelling with inductive biases 
It refers to the practice of endowing a model with the known structure of a specific problem. In the context of physical modeling, this can be achieved in two ways:

* **Soft Biases:** Applied through penalty terms in the loss function.
* **Hard Biases:** Applied through explicit architectural choices.

#### Universal Differential Equations (UDEs)
A "Universal Differential Equation" is a hybrid model that augments an existing theoretical model with a neural network correction term.

**Motivation:** theoretical models are rarely perfectly accurate; there is usually a gap between the theoretical prediction and the behavior observed in practice. Using a neural network is an admission that *there is behavior we do not fully understand*, but this augmentation allows us to model that residual behavior explicitly.

##### Example: Augmented Lotka-Volterra

* **The Theoretical Base:** A standard ODE describes the interaction between prey population $x(t)$ and predator population $y(t)$ using parameters $\alpha, \beta, \gamma, \delta$:

$$\frac{dx}{dt}(t) = \alpha x(t) - \beta x(t) y(t) \in \mathbb{R},$$

$$\frac{dy}{dt}(t) = -\gamma x(t) + \delta x(t) y(t) \in \mathbb{R}.$$

* **The Augmented Model:** To close the gap between theory and observation, neural networks $f_\theta$ and $g_\theta$ are added to the differential equations:


$$\frac{dx}{dt}(t) = \alpha x(t) - \beta x(t) y(t) + f_\theta(x(t), y(t)) \in \mathbb{R},$$

$$\frac{dy}{dt}(t) = -\gamma x(t) + \delta x(t) y(t) + g_\theta(x(t), y(t)) \in \mathbb{R}.$$

We broadly refer to this approach as a *universal diﬀerential equation*. Contrary to standard deep learning, the networks used in UDEs are frequently very small. Examples cited include feedforward networks with only a single hidden layer of width $32$, or $10$ layers of width $10$.

After we learned the vector field, we (can) use an ODE solver (like Runge–Kutta, etc.) to integrate the chosen equation (RHS, which includes the neural nets). We never "solve for $x(t)$ and $y(t)$ in closed form"; instead you **numerically integrate** the learned ODE to get estimate of $x(t)$ and $y(t)$.

##### Loss function an training

Assume we observe data points $x_i(t_j) \in \mathbb{R}$ and $y_i(t_j) \in \mathbb{R}$, where

* $i = 1,\dots,N$ indexes different trajectories (e.g. different initial conditions), and
* $j = 1,\dots,M$ indexes observation times $t_j \in [0,T]$ with $t_1 = 0$.

In practice we might have only one trajectory ($N=1$), provided we observe it at enough time points ($M$ large).

For either system (preys or predators), let $x_{x_0,y_0}(t)$ denote the solution $x(t)$ with initial condition $x(0)=x_0$, $y(0)=y_0$; similarly for $y_{x_0,y_0}(t)$.

We can fit both models in exactly the same way, using stochastic gradient descent to minimise

$$
\frac{1}{NM}\sum_{i=1}^N\sum_{j=1}^M
\Big(x_{x_i(0),y_i(0)}(t_j) - x_i(t_j)\Big)^2
+
\Big(y_{x_i(0),y_i(0)}(t_j) - y_i(t_j)\Big)^2.
$$

Thus, moving from preys to predators does not change the basic training procedure.

##### Training Strategies and Challenges

* **Interpretability Issues:** If trained directly, the neural network might model parts of the behavior that *should* be captured by the physical parameters $\alpha, \beta, \gamma, \delta$, making those parameters uninterpretable and they may not necessarily correspond to their usual quantities because of that.
* *Solution:* Fit the theoretical model first to initialize the physical parameters, then train the neural network parameters $\theta$ to fit only the residual.
* *Alternative Solution:* Regularize the norm of the neural network so it is only utilized when necessary.
* **Local Minima:** Because the networks are small, the model may get stuck in local minima.
* *Solution:* Train on a small portion of the time series first (e.g., the first 10%) before expanding to the whole series.

##### Use Cases
This approach is ideal for modeling complex, poorly understood behavior where sufficient data exists to show that theoretical models are falling short.

* **Closure Relations:** In physics, UDEs can approximate terms that lack precise theoretical descriptions (often representing effects at scales smaller than the solver can resolve).
* **Specific Examples:**
  * **Turbulence Modelling:** Approximating Reynolds stresses in Reynolds-averaged Navier Stokes models.
  * **Climate Modelling:** Modeling turbulent vertical heat flux in ocean models using small Multilayer Perceptrons (MLPs).


#### Hamiltonian neural networks
#TODO:

#### Lagrangian neural networks
#TODO:

### Continuous normalising flows

#### Normalising Flows Recap

Fix $d \in \mathbb{N}$ and take a bijective, sufficiently smooth map $f : \mathbb{R}^d \to \mathbb{R}^d$.

Let $X$ be a random variable in $\mathbb{R}^d$ with density $p_X : \mathbb{R}^d \to [0,\infty)$, and define $Y = f(X)$.
By the change-of-variables formula, the density $p_Y : \mathbb{R}^d \to [0,\infty)$ of $Y$ is

$$p_Y(y) = p_X\big(f^{-1}(y)\big) \left|\det \frac{df}{dx}\big(f^{-1}(y)\big)\right|^{-1}.$$

Now choose $X$ to be multivariate Gaussian, let $Y$ represent the data distribution (empirical samples), and let $f = f_\theta$ be a flexible neural network. We train $f_\theta$ by maximum likelihood, i.e. we solve

$$
\max_\theta \mathbb{E}_{y \sim Y} \log p_Y(y)
= \max_\theta \mathbb{E}_{y \sim Y}
\left[
\log p_X\big(f_\theta^{-1}(y)\big)\log \left|\det \frac{\partial f}{\partial x}\big(f^{-1}(y)\big)\right|
  \right].
$$

Once training is finished, we have a generative model for $Y$: to draw approximate samples from $Y$, we first sample $X$ from its Gaussian prior and then compute $f(X)$.

The difficulty is that training is computationally heavy: in general, computing the Jacobian determinant (or its log) scales as $\mathcal{O}(d^3)$. As a result, a lot of research focuses on designing neural architectures $f_\theta$ that (a) have special structure making the Jacobian (log-det) cheap to evaluate, while (b) remaining expressive enough to model complex distributions well.

<div class="gd-grid">
  <img src="{{ '/assets/images/notes/dynamical-systems/normalising_flows.png' | relative_url }}" alt="NFs" loading="lazy">
  <figcaption>Normalising Flows</figcaption>
</div>


#### The Generative Goal
In this context, the objective is unsupervised learning. We assume there is an observable distribution $\mathbb{P}$ with a density $\pi$ over a state space $\mathbb{R}^d$ (e.g., a distribution of images of cats). Our goal is to learn a generative model that approximates this distribution, allowing us to synthesize new samples.

#### The Random Neural ODE Model
To achieve this, we define a "random neural ODE." Instead of a fixed input, we start with a simple noise distribution and evolve it over time.

The model is defined by:

$$y(0) \sim \mathcal{N}(0, I_{d \times d}), \quad \frac{dy}{dt}(t) = f_\theta(t, y(t)) \text{ for } t \in [0, T].$$

* **Initial State:** $y(0)$ is sampled from a standard Gaussian distribution.
* **Evolution:** The state evolves according to the vector field $f_\theta$.
* **Objective:** We train the model so that the distribution of the final state $y(1)$—which is the "pushforward" of the initial Gaussian—approximates the target distribution $\mathbb{P}$. This architecture is known as a **Continuous Normalising Flow (CNF)**.

**Sampling:**
Once trained, generating new data is straightforward: simply sample noise $y(0) \sim \mathcal{N}(0, I_{d \times d})$ and solve the differential equation forward to time $T$.


#### Instantaneous Change of Variables
To train this model via maximum likelihood, we must calculate the probability density of the output $y(1)$. This requires tracking how the density changes as the data flows through the differential equation.

$\textbf{Theorem (Instantaneous Change of Variables):}$ Let $p_\theta(t, \cdot)$ be the density of $y(t)$ at time $t$. If $f_\theta$ is Lipschitz continuous, then the log-density of a sample evolving along the flow changes according to the divergence of $f$:

$$\frac{d}{dt} \big( t \mapsto \log p_\theta(t, y(t)) \big)(t) = - \sum_{k=1}^d \frac{\partial f_{\theta, k}}{\partial y_k}(t, y(t)) \quad (2.5)$$

Here, the right-hand side represents the negative trace of the Jacobian of $f$ (or the divergence of $f$). The subscript $θ$ in $p_θ$ denotes the dependence on $f_θ$.

> **Note:** This equation is familiar to SDE theorists as the Fokker-Planck equation for deterministic dynamics (with random initial conditions), formulated so the right-hand side does not depend on the unknown density $p_\theta$.

<div class="gd-grid">
  <img src="{{ '/assets/images/notes/dynamical-systems/normalising_flows1.png' | relative_url }}" alt="NFs" loading="lazy">
  <figcaption>A continuous normalising flow continuously deforms one distribution into another distribution. The flow lines show how particles from the base distribution are perturbed until they approximate the target distribution.</figcaption>
</div>


#### Training and Evaluation
We train the CNF by maximizing the likelihood of the observed empirical samples $y_1, \dots, y_N \in \mathbb{R}^d$. This is equivalent to minimizing the negative log-likelihood:

$$-\frac{1}{N} \sum_{i=1}^N \log p_\theta(T, y_i).$$

##### Deriving the Tractable Loss
To evaluate $\log p_\theta(T, y_i)$, we solve the flow **backwards in time** from our observed data at $t=T$ to the latent space at $t=0$. Let $y(t, x)$ denote the solution to the ODE with terminal condition $y(T, x)$ = x.

By integrating the instantaneous change of variables formula (Equation 2.5) and substituting it into the loss function, we obtain a tractable expression:

##### The Evaluation Algorithm
The practical steps to evaluate this loss for a sample y_i are:

1. **Solve Backwards:** Start with the empirical sample y_i at t=T and solve the ODE backwards to t=0 to find the corresponding initial noise vector y(0, y_i).
2. **Integrate Divergence:** Simultaneously solve the integral of the divergence (the trace term in the equation above) during this backward pass by concatenating it to the system of differential equations.
3. **Compute Probability:** Evaluate \log p_\theta(0, \cdot) using the standard normal distribution formula on the resulting y(0, y_i), and add the value of the divergence integral.

##### Backpropagation
Once the loss (Equation 2.7) is evaluated, we update parameters \theta via gradient descent. Since the evaluation already involves a reverse-time solve (from T to 0), and backpropagation is inherently a "reverse time" procedure, the gradient calculation aligns naturally with the evaluation flow.

Would you like me to explain the connection between the "divergence" mentioned here and the Jacobian determinant used in discrete Normalizing Flows?

### Latent ODEs
### Residual networks
### Choice of parameterisation
### Approximation properties

## Neural Controlled Diﬀerential Equations
### Introduction

## Numerical Solutions of Neural Diﬀerential Equations

## Other Stuff

### Symbolic regression

Deep learning models (including neural differential equations) are often accurate but hard to interpret, and in science understanding the model can be as important as performance. That motivates seeking **symbolic expressions** — compact formulas built from simple operations like $+$, $\times$, exponentiation, etc.

**Symbolic regression** aims to automatically discover such expressions from data. But it’s difficult because the space of possible expressions is not smoothly differentiable (so search often relies on *genetic algorithms*) and it is **combinatorially huge** — the number of possible expression trees grows extremely fast with size: $\frac{(2n)!}{(n+1)!n!}$ binary trees with $n$ vertices and so as a rough approximation we may expect there to be a similar number of possible expressions to consider. 

As a result, symbolic regression tends to work best for simpler problems; beyond a certain complexity, the search becomes impractical.

### Symbolic regression for dynamical systems

$\text{Example (SINDy)}:$ Assume we have pairs  $(\frac{dy}{dt}(t), y(t))$, and that they satisfy an equation of the form

$$\frac{dy}{dt}(t) = f(y(t))$$

In this setting, [SINDy](/subpages/dynamical-systems/dynamical-systems-in-machine-learning/sindy) tries to find a symbolic form for $f$ by first choosing a set of candidate features $f_i$. It then represents the function as

$$f(y) = \sum_{i=1}^N \theta_i f_i(y)$$

and fits this model by regressing $\frac{dy}{dt}(t)$ onto the feature values ${f_i(y(t))}_{i=1}^N$. A sparsity-inducing penalty, such as $L^1$ regularization, is applied to $\theta$ so that the final expression contains only a small number of terms. 

SINDy is widely regarded as a leading approach for symbolic regression in dynamical systems, with many extensions and applications reported in the literature.

However, SINDy relies on two strong assumptions:
* that matched measurements of both $y$ and $\frac{dy}{dt}(t)$ are available, and
* $f$ can be expressed as a shallow formula — specifically, a sparse linear combination of a preselected set of candidate functions.

> SINDy sometimes works around the first assumption by approximating $\frac{dy}{dt(t)}$ using finite diﬀerences. However this requires densely-packed observations.

Next, we will see how **neural differential equations (NDEs)** can help relax both of these assumptions introduced in this example.

###TODO: finish the chapter
