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
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/simple_neural_ode.png' | relative_url }}" alt="simple neural ODE" loading="lazy">
    <figcaption>Computation graph for a simple neural ODE.</figcaption>
  </figure>
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

<div class="accordion">
  <details>
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

> Note: not every function $y_\theta(t)$ that you put in, for example, $[dy\dt](t)=const+y_\theta(t)$ will give this equality. Such function $y_\theta(t)$ that satisfies this equality must be learned.

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

---

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
