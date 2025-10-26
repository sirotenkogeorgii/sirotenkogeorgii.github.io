---
title: Dynamical Systems Theory in Machine Learning
date: 2024-11-01
excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
tags:
  - dynamical-systems
  - machine-learning
  - theory
---

## Executive Summary

Dynamical systems theory (DST) studies rules that evolve a state \(x(t)\) through time. The same mathematical ideas underpin classical mechanics, control theory, and key components of modern machine learning such as gradient-based training, recurrent networks, and neural differential equations. Viewing algorithms as dynamical systems unlocks geometric intuition—phase portraits, invariant sets, and stability criteria—that explains convergence, metastability, and chaotic behavior in high-dimensional optimizers.

* **Continuous-time systems** evolve according to ordinary differential equations (ODEs) \(\dot{x} = f(x, t)\).
* **Discrete-time systems** evolve via iterated maps \(x_{k+1} = F(x_k)\).
* **State space geometry** (fixed points, limit cycles, attractors) conveys long-term behavior without solving the dynamics explicitly.
* **Linearization** and spectral analysis give tractable approximations near equilibria, guiding both theoretical guarantees and practical design.

## 1. Building Blocks

### 1.1 Continuous vs. Discrete Time

A continuous-time dynamical system on a state space \(X \subseteq \mathbb{R}^n\) is defined by an ODE
\[
\dot{x}(t) = f(x(t), t), \qquad x(0) = x_0.
\]
The flow map \(\Phi_t(x_0)\) transports the initial state along trajectories: \(x(t) = \Phi_t(x_0)\).

Discrete-time systems replace derivatives with difference equations:
\[
x_{k+1} = F(x_k), \qquad x_0 \in X.
\]
Both viewpoints are linked: explicit Euler with step size \(h\) gives \(x_{k+1} = x_k + h f(x_k)\), while letting \(h \to 0\) recovers the continuous flow.

### 1.2 State Space and Trajectories

* **State space:** set of all admissible states.
* **Trajectory/orbit:** curve \(\{\Phi_t(x_0)\}_{t \ge 0}\) or sequence \(\{F^{\circ k}(x_0)\}_{k \ge 0}\).
* **Equilibrium (fixed point):** state \(x^\star\) with \(f(x^\star) = 0\) (continuous) or \(F(x^\star) = x^\star\) (discrete).
* **Invariant set:** subset \(S \subseteq X\) with \(\Phi_t(S) = S\) (or \(F(S) = S\)).

DST often focuses on **autonomous** systems, where \(f(x, t) = f(x)\). Non-autonomous systems can be rewritten as autonomous ones by augmenting time: define \(z = (t, x)\) and \(\dot{z} = (1, f(x, t))\).

### 1.3 First-Order Form

Higher-order ODEs can be written as first-order systems. For example, the third-order equation
\[
\frac{d^3 x}{dt^3} + 2 \frac{d^2 x}{dt^2} - \frac{dx}{dt} + 5x = \cos t
\]
becomes the first-order system in coordinates \(x_1 = x\), \(x_2 = \dot{x}\), \(x_3 = \ddot{x}\):
\[
\dot{x}_1 = x_2,\quad
\dot{x}_2 = x_3,\quad
\dot{x}_3 = -2 x_3 + x_2 - 5 x_1 + \cos t.
\]

## 2. Qualitative Analysis

### 2.1 Linearization and Stability

Near an equilibrium \(x^\star\), expand \(f\) to first order:
\[
f(x) \approx f(x^\star) + Df(x^\star)(x - x^\star) = A (x - x^\star), \quad A := Df(x^\star).
\]
For the linear system \(\dot{y} = A y\), the solution is \(y(t) = \exp(At) y(0)\). Spectral properties of \(A\) classify local behavior:

| Eigenvalues of \(A\) | Local behavior |
| --- | --- |
| \(\Re \lambda_i < 0\) for all \(i\) | Asymptotically stable node |
| Some \(\Re \lambda_i > 0\) | Unstable (repelling directions) |
| Purely imaginary, semisimple | Center (requires nonlinear terms to decide stability) |

This motivates **Hartman–Grobman:** non-linear systems behave like their linearization near a hyperbolic equilibrium (no eigenvalues on the imaginary axis).

### 2.2 Phase Portraits and Invariant Manifolds

* **Stable manifold \(W^s(x^\star)\):** states whose trajectories converge to \(x^\star\) as \(t \to \infty\).
* **Unstable manifold \(W^u(x^\star)\):** states that converge when time is reversed.
* **Limit cycles:** isolated periodic orbits. In planar systems, Poincaré–Bendixson ensures trajectories in compact regions either approach equilibria, a limit cycle, or a cycle of equilibria.
* **Lyapunov functions:** scalar \(V(x) \ge 0\) with \(\dot{V}(x) \le 0\) certify stability without solving the ODE.

## 3. Linear Systems Refresher

### 3.1 One-Dimensional Case

The scalar ODE \(\dot{x} = a x\) has solution \(x(t) = x_0 e^{a t}\).

* \(a < 0\): asymptotically stable (sink).
* \(a > 0\): unstable (source).
* \(a = 0\): neutrally stable; every point is an equilibrium.

Discrete-time analog \(x_{k+1} = \alpha x_k\) converges iff \(|\alpha| < 1\).

### 3.2 Multi-Dimensional Linear Flow

For \(\dot{x} = A x\), diagonalize or use Jordan form. If \(A = PDP^{-1}\) with diagonal \(D = \operatorname{diag}(\lambda_i)\), then
\[
x(t) = \sum_{i} c_i e^{\lambda_i t} v_i, \quad c_i = \langle w_i, x_0 \rangle.
\]
Complex eigenvalues \(\lambda = \sigma \pm i \omega\) yield spirals with angular frequency \(\omega\) and exponential growth/decay \(\sigma\).

### 3.3 Linear Systems as Approximations

Many machine learning proofs (e.g., linear convergence of gradient descent near a strongly convex minimizer) rely on the linear system
\[
\dot{e}(t) = -H e(t), \qquad H \succ 0,
\]
where \(e = x - x^\star\). Eigenvalues of \(H\) set contraction rates along principal directions.

## 4. Nonlinear Phenomena

### 4.1 Normal Forms and Bifurcations

When an eigenvalue crosses the imaginary axis as a parameter \(\mu\) varies, equilibria change stability:

* **Saddle-node:** two equilibria collide and annihilate.
* **Pitchfork:** symmetry creates one unstable and two stable equilibria (or vice versa).
* **Hopf:** a complex conjugate pair crosses, birthing a limit cycle with amplitude proportional to \(\sqrt{\mu}\).

Near a Hopf bifurcation, dynamics in polar coordinates satisfy
\[
\dot{r} = \mu r - \beta r^3, \qquad \dot{\theta} = \omega + \mathcal{O}(r^2),
\]
illustrating amplitude saturation.

### 4.2 Chaos and Sensitivity

Higher-dimensional nonlinear systems exhibit sensitive dependence on initial conditions. The Lorenz system
\[
\dot{x} = \sigma (y - x),\qquad \dot{y} = x(\rho - z) - y,\qquad \dot{z} = xy - \beta z
\]
supports a chaotic attractor for classic parameters (\(\sigma = 10\), \(\beta = 8/3\), \(\rho = 28\)). Lyapunov exponents quantify exponential separation of nearby trajectories.

## 5. Dynamical Systems in Machine Learning

### 5.1 Gradient Descent as a Discrete Flow

For loss \(J(\theta)\), gradient descent with step size \(\eta\) reads
\[
\theta_{k+1} = \theta_k - \eta \nabla J(\theta_k).
\]
Small \(\eta\) approximates the gradient flow ODE
\[
\dot{\theta} = - \nabla J(\theta),
\]
whose equilibria are critical points. Strong convexity (\(\nabla^2 J(\theta^\star) \succeq m I\)) implies exponential convergence: \(\|\theta(t) - \theta^\star\| \le e^{-m t} \|\theta(0) - \theta^\star\|\).

### 5.2 Stochastic Effects

Stochastic gradient descent (SGD) adds noise: \(\theta_{k+1} = \theta_k - \eta ( \nabla J(\theta_k) + \xi_k )\). In the small-step limit, the dynamics resemble a stochastic differential equation
\[
d\theta_t = - \nabla J(\theta_t) \, dt + \Sigma^{1/2} dW_t,
\]
where \(W_t\) is a Wiener process. Stationary distributions and escape times from saddle neighborhoods can be analyzed via stochastic stability theory.

### 5.3 Recurrent Models and Neural ODEs

* **RNNs:** \(h_{t+1} = \phi(W h_t + U x_{t+1} + b)\) defines a discrete-time system. Jacobian spectral radius controls exploding/vanishing gradients through backpropagation.
* **Neural ODEs:** learned vector fields \(f_\theta(x, t)\) generate flows \(\dot{x} = f_\theta(x, t)\). Adjoint sensitivities solve a reverse-time ODE to compute gradients.
* **Diffusion models:** forward SDE \(dx_t = f(x_t, t)\,dt + g(t) dW_t\) gradually destroys structure; training learns the reverse-time dynamics to synthesize data.

## 6. Exercises

1. A non-autonomous planar system \(\dot{x} = y + \cos t\), \(\dot{y} = -x + \sin t\) is linear. Convert it into an autonomous system by augmenting the state and classify the equilibrium of the augmented system.
2. Let \(F(x) = x + h f(x)\) be an explicit Euler step with step size \(h > 0\). Show that if all eigenvalues of \(Df(x^\star)\) have negative real parts, then \(x^\star\) remains stable for \(h < h_{\max}\), and derive \(h_{\max}\) in terms of the spectral radius of \(Df(x^\star)\).
3. For the two-dimensional system \(\dot{x} = y - x^3\), \(\dot{y} = -x\), construct a Lyapunov function that proves trajectories remain bounded.

## Further Reading

* **Strogatz, _Nonlinear Dynamics and Chaos_.** Gentle introduction with geometric intuition.
* **Khalil, _Nonlinear Systems_.** Rigorous Lyapunov and stability analysis.
* **Brunton & Kutz, _Data-Driven Science and Engineering_.** Connects sparse regression and operator learning to system identification.

